"""Compile pool (l3_farm/analysis/compile_pool.md): shared content-addressed build cache + remote builds.

GPU-free: a stub ``nvcc`` (a shell script that writes a deterministic "object"), a stub ``ssh`` that runs the
remote command locally and a stub ``scp`` that copies. The remote script needs ``flock`` (Linux; a small python
shim stands in where it is missing, e.g. on a laptop).
"""
import asyncio
import json
import multiprocessing as mp
import os
import shutil
import stat
import sys
import textwrap
import time
from pathlib import Path

import pytest

from kernelblaster.knowledge import remote_build as rb
from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem, SSHEvaluator
from kernelblaster.servers.utils import build_cache as bc
from kernelblaster.servers.utils.build_cache import (BuildCache, BuildError, CachedBuildError, cache_key,
                                                     deterministic_failure, external_inputs, summarize_stats)

FIX = Path(__file__).parent / "fixtures" / "mb_lenet"

STUB_NVCC = textwrap.dedent("""\
    #!/bin/bash
    D="$(cd "$(dirname "$0")" && pwd)"
    if [ "$1" = "--version" ]; then cat "$D/version"; exit 0; fi
    out=""; src=""; prev=""
    for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; case "$a" in *.cu|*.cpp) src="$a";; esac; done
    echo "$$ $*" >> "$D/calls.log"
    [ -f "$D/sleep" ] && sleep "$(cat "$D/sleep")"
    if [ -n "$src" ] && grep -q COMPILE_ERROR "$src"; then echo "$src(3): error: expected a ';'" >&2; exit 1; fi
    if [ -n "$src" ] && grep -q KILLED "$src"; then echo "Killed" >&2; exit 137; fi
    { echo "stub-object"; [ -f "$D/version" ] && cat "$D/version"; for a in "$@"; do case "$a" in -c|-o|*.o|*.cu) ;; *) echo "flag $a";; esac; done
      [ -n "$src" ] && cat "$src"; for a in "$@"; do case "$a" in *.o) [ "$a" != "$out" ] && [ -f "$a" ] && head -c 100000 "$a";; esac; done; true; } > "$out"
    """)
STUB_FLOCK = textwrap.dedent("""\
    #!/usr/bin/env python3
    import fcntl, os, subprocess, sys
    a = [x for x in sys.argv[1:] if x != "-n"]
    nb = "-n" in sys.argv[1:]
    flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nb else 0)
    try:
        if a[0].isdigit():
            fcntl.flock(int(a[0]), flags)
            sys.exit(0)
        fd = os.open(a[0], os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, flags)
        sys.exit(subprocess.call(a[1:]))
    except BlockingIOError:
        sys.exit(1)
    """)
STUB_SSH = textwrap.dedent("""\
    #!/bin/bash
    D="$(cd "$(dirname "$0")" && pwd)"
    while [ $# -gt 0 ]; do case "$1" in -o|-i) shift 2;; -*) shift;; *) break;; esac; done
    host="$1"; shift
    echo "$host" >> "$D/ssh_hosts.log"
    case " $(cat "$D/down" 2>/dev/null) " in *" $host "*) echo "ssh: connect to host $host port 22: Connection refused" >&2; exit 255;; esac
    if [ -f "$D/truncate" ]; then echo "partial output"; exit 0; fi
    exec bash -c "$*"
    """)
STUB_SCP = textwrap.dedent("""\
    #!/bin/bash
    D="$(cd "$(dirname "$0")" && pwd)"
    args=()
    while [ $# -gt 0 ]; do case "$1" in -o|-i) shift 2;; -*) shift;; *) args+=("$1"); shift;; esac; done
    [ -f "$D/scp_fail" ] && { echo "scp: lost connection" >&2; exit 1; }
    src="${args[0]}"; src="${src#*:}"; dst="${args[1]}"; dst="${dst#*:}"
    exec cp "$src" "$dst"
    """)


def _exe(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def stubs(tmp_path, monkeypatch):
    d = tmp_path / "stubbin"
    d.mkdir()
    _exe(d / "nvcc", STUB_NVCC)
    (d / "version").write_text("Cuda compilation tools, release 12.6, V12.6.77\nBuild cuda_12.6.r12.6/compiler.1\n")
    _exe(d / "ssh", STUB_SSH)
    _exe(d / "scp", STUB_SCP)
    if shutil.which("flock") is None:
        _exe(d / "flock", STUB_FLOCK)
    monkeypatch.setenv("PATH", f"{d}:{os.environ['PATH']}")
    for k in ("KERNELBLASTER_BUILD_CACHE", "KERNELBLASTER_REMOTE_BUILD", "KERNELBLASTER_BUILD_CACHE_MAX_GB",
              "KERNELBLASTER_BUILD_CACHE_NEG_TTL", "KERNELBLASTER_BUILD_CACHE_STATS_DIR", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(k, raising=False)
    bc._TOOL_ID.clear()
    return d


def _calls(stubs: Path) -> int:
    p = stubs / "calls.log"
    return len(p.read_text().splitlines()) if p.exists() else 0


def _sources(problem, **edit):
    src = problem.seed_sources()
    src.update(edit)
    return src


def _build(ev, problem, sources, work):
    return asyncio.run(ev._build_so(problem, sources, work))


def _local(stubs, tmp_path, **kw):
    return LocalEvaluator("89", work_root=tmp_path / "w", nvcc=str(stubs / "nvcc"), build_jobs=4, runtime="host", **kw)


# ---------------------------------------------------------------- the cache itself

def test_flag_off_keys_and_files_are_unchanged(stubs, tmp_path):
    """No KERNELBLASTER_BUILD_CACHE: per-problem cache dir, the legacy keys (nvcc command, mtime inputs), no sidecar /
    lock / stats files."""
    p = SlotProblem(FIX)
    ev = _local(stubs, tmp_path)
    so, bad = _build(ev, p, p.seed_sources(), tmp_path / "w1")
    assert bad is None and so.is_relative_to(tmp_path / "w" / ".build_cache")
    flags = ev._so_flags()
    for name, text in p.units(p.seed_sources()):
        k = cache_key("nvcc-c", str(stubs / "nvcc"), *flags, text, *external_inputs(text))
        assert (ev._build_cache(p).path(k, ".o")).exists(), name
    names = {f.name for f in (tmp_path / "w" / ".build_cache").rglob("*") if f.is_file()}
    assert not any(n.endswith((".meta", ".err", ".json")) for n in names)
    assert not (tmp_path / "w" / ".build_cache" / ".locks").exists()
    assert ev._cache.max_bytes is None and ev._cache.neg_ttl == 0


def test_flag_off_ssh_evaluator_never_touches_ssh(stubs, tmp_path):
    p = SlotProblem(FIX)
    ev = SSHEvaluator("h1", arch="89", remote_dir=str(tmp_path / "rd"), remote_path=str(stubs), ssh=str(stubs / "ssh"),
                      scp=str(stubs / "scp"), work_root=tmp_path / "w", nvcc=str(stubs / "nvcc"), build_jobs=4)
    assert ev.remote_build is False
    _, bad = _build(ev, p, p.seed_sources(), tmp_path / "w1")
    assert bad is None and not (stubs / "ssh_hosts.log").exists()
    assert "remote_build" not in ev.host_stats()


def test_shared_cache_hit_across_problems_and_evaluators(stubs, tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELBLASTER_BUILD_CACHE", str(tmp_path / "shared"))
    p = SlotProblem(FIX)
    ev1, ev2 = _local(stubs, tmp_path / "a"), _local(stubs, tmp_path / "b")
    so1, _ = _build(ev1, p, p.seed_sources(), tmp_path / "w1")
    n = _calls(stubs)
    assert n > 0 and so1.is_relative_to(tmp_path / "shared")
    so2, _ = _build(ev2, p, p.seed_sources(), tmp_path / "w2")
    assert so2 == so1 and _calls(stubs) == n                          # a second evaluator: no nvcc at all
    st = ev2._cache.stats()
    assert st["hits"] >= 1 and st["seconds_saved"] >= 0 and st["bytes_saved"] > 0


def _one_unit_edit(p):
    sid = next(iter(p.slots))
    return sid, p.seed_sources()[sid]


def test_key_inputs_each_make_a_miss(stubs, tmp_path, monkeypatch):
    """A changed slot text, flag, arch, nvcc version, compiler env or included header is a miss; the unchanged
    units of the same candidate stay hits (the incremental build)."""
    monkeypatch.setenv("KERNELBLASTER_BUILD_CACHE", str(tmp_path / "shared"))
    p = SlotProblem(FIX)
    base = _local(stubs, tmp_path / "a")
    _build(base, p, p.seed_sources(), tmp_path / "w1")
    n0 = _calls(stubs)
    nunits = len(p.units(p.seed_sources()))

    sid, text = _one_unit_edit(p)
    _build(_local(stubs, tmp_path / "b"), p, _sources(p, **{sid: text + "\n// edit\n"}), tmp_path / "w2")
    assert _calls(stubs) - n0 == 2                                    # the edited slot unit + the link only
    n1 = _calls(stubs)

    arch = LocalEvaluator("80", work_root=tmp_path / "c", nvcc=str(stubs / "nvcc"), build_jobs=4, runtime="host")
    _build(arch, p, p.seed_sources(), tmp_path / "w3")
    assert _calls(stubs) - n1 == nunits + 1                           # another arch: everything rebuilt
    n2 = _calls(stubs)

    g = LocalEvaluator("89", work_root=tmp_path / "d", nvcc=str(stubs / "nvcc"), build_jobs=4, runtime="host", graph=True)
    assert g._so_flags() == base._so_flags()                          # .so flags do not depend on graph: still a hit
    _build(g, p, p.seed_sources(), tmp_path / "w4")
    assert _calls(stubs) == n2

    (stubs / "version").write_text("Cuda compilation tools, release 12.8, V12.8.61\nBuild x\n")   # another toolkit
    bc._TOOL_ID.clear()
    _build(_local(stubs, tmp_path / "e"), p, p.seed_sources(), tmp_path / "w5")
    assert _calls(stubs) - n2 == nunits + 1
    n3 = _calls(stubs)

    monkeypatch.setenv("NVCC_APPEND_FLAGS", "-DFOO=1")                # compiler environment
    _build(_local(stubs, tmp_path / "f"), p, p.seed_sources(), tmp_path / "w6")
    assert _calls(stubs) - n3 == nunits + 1


def test_included_header_content_is_part_of_the_key(tmp_path, monkeypatch):
    h = tmp_path / "x.h"
    h.write_text("#define A 1\n")
    src = f'#include "{h}"\nint f();\n'
    k1 = cache_key(src, *external_inputs(src, content=True))
    st = os.stat(h)
    h.write_text("#define A 2\n")                                      # same size, new content
    os.utime(h, ns=(st.st_atime_ns, st.st_mtime_ns))                  # ... and the same mtime
    k2 = cache_key(src, *external_inputs(src, content=True))
    assert k1 != k2
    assert external_inputs(src)[0].startswith(f"{h}:")                # the per-problem mode keeps size:mtime
    data = tmp_path / "data.bin"
    data.write_bytes(b"1234")
    s2 = f'__asm__(".incbin \\"{data}\\"\\n");'
    a = external_inputs(s2, content=True)
    data.write_bytes(b"12345")
    assert external_inputs(s2, content=True) != a                     # incbin: size / mtime


def test_tool_id_names_version_and_env(stubs, monkeypatch):
    a = bc.tool_id(str(stubs / "nvcc"))
    assert "release 12.6" in a
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "-g")
    assert bc.tool_id(str(stubs / "nvcc")) != a
    assert bc.tool_id("/no/such/nvcc").startswith("nvcc-unavailable")


def test_negative_cache(tmp_path):
    c = BuildCache(tmp_path / "c", shared=True, neg_ttl=3600)
    calls = []

    async def bad(tmp: Path):
        calls.append(1)
        raise BuildError("u.cu", "u.cu(3): error: expected a ';'")

    async def timeout(tmp: Path):
        calls.append(2)
        raise BuildError("u.cu", "nvcc timed out after 900s", cacheable=False)

    async def good(tmp: Path):
        tmp.write_text("ok")

    async def main():
        for _ in range(3):
            with pytest.raises(BuildError) as e:
                await c.get_or_build("a" * 64, ".o", bad)
        assert isinstance(e.value, CachedBuildError) and "expected" in e.value.text
        for _ in range(2):                                            # never stored: asked again
            with pytest.raises(BuildError):
                await c.get_or_build("b" * 64, ".o", timeout)
        return await c.get_or_build("c" * 64, ".o", good)             # another key: unaffected
    path, hit = asyncio.run(main())
    assert calls == [1, 2, 2] and path.read_text() == "ok" and not hit
    assert c.st["neg_hits"] == 2 and c.st["failed_uncached"] == 2


def test_negative_entry_expires_and_is_off_by_default(tmp_path):
    async def bad(tmp: Path):
        raise BuildError("u", "error: x")
    c = BuildCache(tmp_path / "c", shared=True, neg_ttl=0.2)
    off = BuildCache(tmp_path / "d")

    async def main():
        with pytest.raises(BuildError):
            await c.get_or_build("k" * 64, ".o", bad)
        with pytest.raises(CachedBuildError):
            await c.get_or_build("k" * 64, ".o", bad)
        await asyncio.sleep(0.3)
        with pytest.raises(BuildError) as e:                           # expired: builds (and fails) again
            await c.get_or_build("k" * 64, ".o", bad)
        assert not isinstance(e.value, CachedBuildError)
        for _ in range(2):
            with pytest.raises(BuildError) as e:
                await off.get_or_build("k" * 64, ".o", bad)
            assert not isinstance(e.value, CachedBuildError)
    asyncio.run(main())


def test_compile_error_candidate_is_cached_and_changed_source_is_not(stubs, tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELBLASTER_BUILD_CACHE", str(tmp_path / "shared"))
    p = SlotProblem(FIX)
    sid, text = _one_unit_edit(p)
    bad = _sources(p, **{sid: text + "\nCOMPILE_ERROR\n"})
    ev = _local(stubs, tmp_path / "a")
    so, res = _build(ev, p, bad, tmp_path / "w1")
    assert so is None and res.stage == "compile" and "expected" in res.error_text()
    n = _calls(stubs)
    so, res2 = _build(_local(stubs, tmp_path / "b"), p, bad, tmp_path / "w2")           # same broken source again
    assert so is None and "expected" in res2.error_text() and res2.errors == res.errors
    assert _calls(stubs) == n                                                      # served from the negative cache
    fixed = _sources(p, **{sid: text + "\nint ok;\n"})
    so, res3 = _build(_local(stubs, tmp_path / "c"), p, fixed, tmp_path / "w3")
    assert res3 is None and so.exists() and _calls(stubs) > n                      # a different text: compiled


def test_killed_compiler_is_not_a_compile_error_and_not_cached(stubs, tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELBLASTER_BUILD_CACHE", str(tmp_path / "shared"))
    assert not deterministic_failure(137, "Killed") and not deterministic_failure(-1, "") and \
        not deterministic_failure(1, "fatal: No space left on device") and deterministic_failure(1, "error: x")
    p = SlotProblem(FIX)
    sid, text = _one_unit_edit(p)
    bad = _sources(p, **{sid: text + "\nKILLED\n"})
    _build(_local(stubs, tmp_path / "a"), p, bad, tmp_path / "w1")
    assert not list((tmp_path / "shared").rglob("*.err"))


def _worker(args):
    root, key, marker = args

    async def build(tmp: Path):
        with open(marker, "a") as f:
            f.write("x\n")
        await asyncio.sleep(0.4)
        tmp.write_text("obj")
    c = BuildCache(Path(root), shared=True)
    path, hit = asyncio.run(c.get_or_build(key, ".o", build))
    return str(path), hit, c.st["lock_waits"]


def test_concurrent_processes_build_a_key_once(tmp_path):
    marker = tmp_path / "marker"
    with mp.get_context("fork").Pool(6) as pool:
        res = pool.map(_worker, [(str(tmp_path / "c"), "f" * 64, str(marker))] * 6)
    assert len(marker.read_text().splitlines()) == 1                  # single flight across processes
    assert len({r[0] for r in res}) == 1 and sum(r[1] for r in res) == 5
    assert not list((tmp_path / "c").rglob("*.tmp"))
    assert Path(res[0][0]).read_text() == "obj"


def test_concurrent_distinct_keys_and_atomic_publish(tmp_path):
    """Readers never see a partial file: the product appears (os.replace) complete or not at all."""
    c = BuildCache(tmp_path / "c", shared=True)
    seen = []

    async def build(tmp: Path):
        tmp.write_text("a" * 1000)
        await asyncio.sleep(0.2)
        tmp.write_text("b" * 1000)

    async def watcher(dst: Path):
        for _ in range(60):
            if dst.exists():
                seen.append(dst.read_text())
            await asyncio.sleep(0.01)

    async def main():
        dst = c.path("d" * 64, ".o")
        await asyncio.gather(c.get_or_build("d" * 64, ".o", build), watcher(dst))
    asyncio.run(main())
    assert seen and set(seen) == {"b" * 1000}


def test_eviction_is_lru_and_respects_min_age(tmp_path):
    root = tmp_path / "c"
    c = BuildCache(root, shared=True, max_bytes=1000, min_age=100)
    now = time.time()
    keys = [f"{i:02x}" + "0" * 62 for i in range(6)]

    async def fill():
        for k in keys:
            async def build(tmp: Path, k=k):
                tmp.write_bytes(b"x" * 300)
            await c.get_or_build(k, ".o", build)
    asyncio.run(fill())
    # entry i last used i*1000 s ago, except the two newest which are "just built"
    ages = {keys[0]: 5000, keys[1]: 4000, keys[2]: 3000, keys[3]: 2000, keys[4]: 10, keys[5]: 5}
    for k, a in ages.items():
        for f in (c.path(k, ".o"), c._meta_path(c.path(k, ".o"))):
            os.utime(f, (now - a, now - a))
    asyncio.run(asyncio.sleep(0))
    rep = c.evict()
    left = {k for k in keys if c.path(k, ".o").exists()}
    assert rep["files"] >= 2 and keys[0] not in left and keys[1] not in left     # oldest gone
    assert keys[4] in left and keys[5] in left                                    # too young to evict
    assert not c._meta_path(c.path(keys[0], ".o")).exists()                       # sidecar goes with its entry
    assert c.size_bytes() <= 1000 or {keys[4], keys[5]} <= left
    # a hit refreshes the entry: it outlives an untouched older one
    c2 = BuildCache(tmp_path / "c2", shared=True, max_bytes=700, min_age=0)

    async def two():
        for k in ("aa" + "0" * 62, "bb" + "0" * 62):
            async def b(tmp: Path):
                tmp.write_bytes(b"y" * 300)
            await c2.get_or_build(k, ".o", b)
    asyncio.run(two())
    a, b = c2.path("aa" + "0" * 62, ".o"), c2.path("bb" + "0" * 62, ".o")
    os.utime(a, (now - 9000, now - 9000))
    os.utime(b, (now - 8000, now - 8000))

    async def touch():                                                             # use a (the older one)
        async def never(tmp):
            raise AssertionError
        await c2.get_or_build("aa" + "0" * 62, ".o", never)
    asyncio.run(touch())
    assert a.stat().st_mtime > b.stat().st_mtime
    c2.evict(500)
    assert a.exists() and not b.exists()


def test_eviction_triggered_by_get_or_build_and_cleans_stale(tmp_path):
    root = tmp_path / "c"
    c = BuildCache(root, shared=True, max_bytes=2000, min_age=0)

    async def fill():
        for i in range(30):
            async def build(tmp: Path):
                tmp.write_bytes(b"z" * 200)
            await c.get_or_build(f"{i:02x}" + "1" * 62, ".o", build)
    asyncio.run(fill())
    assert c.size_bytes() <= 2000 + 200 * 3 and c.st["evicted"] > 0
    stale = root / "00" / ".dead.tmp"
    stale.write_text("x")
    os.utime(stale, (time.time() - 7 * 3600,) * 2)
    c.evict()
    assert not stale.exists()


def test_stats_files_and_summary(tmp_path):
    c = BuildCache(tmp_path / "c", shared=True)

    async def build(tmp: Path):
        await asyncio.sleep(0.05)
        tmp.write_bytes(b"q" * 100)

    async def main():
        for _ in range(4):
            await c.get_or_build("e" * 64, ".o", build)
    asyncio.run(main())
    c.flush_stats()
    s = summarize_stats(tmp_path / "c")
    assert s["hits"] == 3 and s["misses"] == 1 and s["hit_rate"] == 0.75 and s["requests"] == 4
    assert s["bytes_saved"] == 300 and s["seconds_saved"] > 0.1 and s["entries"] == 1 and s["processes"] == 1


# ---------------------------------------------------------------- scheduler

def test_rank_hosts_fewest_in_flight_then_load_then_round_robin():
    hs = ["a", "b", "c"]
    r = rb.rank_hosts
    assert r(hs, {"a": 2, "b": 0, "c": 1}, {}, cap=3) == ["b", "c", "a"]
    assert r(hs, {"a": 1, "b": 1, "c": 1}, {"a": 3, "b": 1, "c": 2}, cap=3) == ["b", "c", "a"]
    assert r(hs, {}, {}, cap=3, rr=0) == ["a", "b", "c"] and r(hs, {}, {}, cap=3, rr=1) == ["b", "c", "a"]
    assert r(hs, {}, {}, cap=3, quarantined=["a"]) == ["b", "c"]
    assert r(hs, {}, {}, cap=3, quarantined=hs) == hs                       # all quarantined: still try them
    assert r(hs, {"a": 0, "b": 5, "c": 5}, {}, cap=3, avoid=["a"]) == ["b", "c"]
    assert r(hs, {}, {}, cap=3, avoid=hs) == hs


def test_remote_script_parts():
    s = rb.remote_script(cache_dir="/d/.kbcc", work_dir="/d/s/rb", name="u.cu", key="k" * 8, args=["-O3", "-arch=sm_89"],
                         cuda_bin="/cuda/bin", env={"NVCC_APPEND_FLAGS": "-g"}, timeout_s=900, slots=3, avoid_busy=True)
    assert "nice -n 19" in s and "ionice -c3" in s and "flock -n 9" in s and "KB_RB_END" in s
    assert "-lt 3" in s and "NVCC_APPEND_FLAGS=-g" in s and "kernelblaster_gpu_" in s and "timeout -k 5 900" in s
    assert "kernelblaster_gpu_" not in rb.remote_script(cache_dir="/d", work_dir="/w", name="u.cu", key="k", args=[],
                                                        cuda_bin="/c", env={}, timeout_s=1, slots=1, avoid_busy=False)
    r = rb.RemoteReply("x\nKB_RB_RC=1\nKB_RB_SECS=2.5\nKB_RB_LOG_BEGIN\nerror: boom\n\nKB_RB_LOG_END\nKB_RB_END\n")
    assert r.rc == 1 and r.seconds == 2.5 and r.end and "boom" in r.log and not r.full
    assert rb.RemoteReply("KB_RB_FULL=1\nKB_RB_END\n").full and not rb.RemoteReply("KB_RB_RC=0\n").end


# ---------------------------------------------------------------- remote builds (stub ssh)

def _ssh_ev(stubs, tmp_path, hosts=("h1",), rd=None, **kw):
    kw.setdefault("remote_build", True)
    rd = rd or tmp_path
    return SSHEvaluator(list(hosts), arch="89", remote_dir={h: str(rd / f"rd_{h}") for h in hosts}, remote_path=str(stubs),
                        ssh=str(stubs / "ssh"), scp=str(stubs / "scp"), work_root=tmp_path / "w",
                        nvcc=str(stubs / "nvcc"), build_jobs=4, runtime="host", **kw)


def _ssh_hosts(stubs):
    p = stubs / "ssh_hosts.log"
    return p.read_text().split() if p.exists() else []


def test_remote_build_equals_local_build(stubs, tmp_path, monkeypatch):
    p = SlotProblem(FIX)
    loc = _local(stubs, tmp_path / "l")
    so_l, _ = _build(loc, p, p.seed_sources(), tmp_path / "w0")
    ev = _ssh_ev(stubs, tmp_path)
    so_r, bad = _build(ev, p, p.seed_sources(), tmp_path / "w1")
    st = ev.remote_build_stats()
    assert bad is None and st["remote"] == len(p.units(p.seed_sources())) and st["local"] == 0
    # the object bytes of the stub embed the flags, the source text and the version: remote == local up to the
    # per-job paths in the link line (none are embedded by the stub)
    keep = lambda b: [l for l in b.read_bytes().splitlines()[1:] if not l.startswith(b"flag /")]
    assert keep(so_l) == keep(so_r)
    assert st["per_host"]["h1"]["builds"] == st["remote"] and st["bytes_back"] > 0 and st["eligible"] == ["h1"]


def test_remote_build_with_shared_cache_second_evaluator_needs_no_host(stubs, tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELBLASTER_BUILD_CACHE", str(tmp_path / "shared"))
    p = SlotProblem(FIX)
    ev = _ssh_ev(stubs, tmp_path / "a")
    _build(ev, p, p.seed_sources(), tmp_path / "w1")
    n = len(_ssh_hosts(stubs))
    assert n > 0
    ev2 = _ssh_ev(stubs, tmp_path / "b")
    so, bad = _build(ev2, p, p.seed_sources(), tmp_path / "w2")
    assert bad is None and len(_ssh_hosts(stubs)) == n               # all hits on the driver cache: no host contacted
    # a host keeps its own object cache by hash: the same unit asked of it again is not recompiled
    sid, text = _one_unit_edit(p)
    _build(ev, p, _sources(p, **{sid: text + "\n// e\n"}), tmp_path / "w3")
    calls = _calls(stubs)
    monkeypatch.setenv("KERNELBLASTER_BUILD_CACHE", str(tmp_path / "shared2"))   # driver cache empty, host cache warm
    c3 = _ssh_ev(stubs, tmp_path / "c", rd=tmp_path / "a")
    _build(c3, p, _sources(p, **{sid: text + "\n// e\n"}), tmp_path / "w4")
    assert c3.remote_build_stats()["remote_cached"] >= 1


def test_remote_compile_error_is_a_compile_error_not_infra(stubs, tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELBLASTER_BUILD_CACHE", str(tmp_path / "shared"))
    p = SlotProblem(FIX)
    sid, text = _one_unit_edit(p)
    bad = _sources(p, **{sid: text + "\nCOMPILE_ERROR\n"})
    ev = _ssh_ev(stubs, tmp_path)
    so, res = _build(ev, p, bad, tmp_path / "w1")
    st = ev.remote_build_stats()
    assert so is None and res.stage == "compile" and "expected" in res.error_text()
    assert st["compile_errors"] >= 1 and st["infra"] == 0
    n = len(_ssh_hosts(stubs))
    _build(_ssh_ev(stubs, tmp_path / "b"), p, bad, tmp_path / "w2")
    assert len(_ssh_hosts(stubs)) == n                                # negative entry on the driver: no host asked


@pytest.mark.parametrize("mode", ["down", "truncate", "scp_fail"])
def test_transport_failure_falls_back_to_the_driver_and_is_not_a_candidate_failure(stubs, tmp_path, mode):
    p = SlotProblem(FIX)
    ev = _ssh_ev(stubs, tmp_path)
    asyncio.run(ev._rb_eligible())                                    # the probe saw a healthy host
    assert ev.remote_build_stats()["eligible"] == ["h1"]
    (stubs / {"down": "down", "truncate": "truncate", "scp_fail": "scp_fail"}[mode]).write_text("h1" if mode == "down" else "1")
    so, bad = _build(ev, p, p.seed_sources(), tmp_path / "w1")
    st = ev.remote_build_stats()
    assert bad is None and so.exists()                                # the candidate built (on the driver)
    assert st["infra"] >= 1 and st["local"] >= 1 or mode == "scp_fail" and st["infra"] >= 1
    assert st["compile_errors"] == 0


def test_killed_remote_compiler_is_infra_and_driver_decides(stubs, tmp_path):
    p = SlotProblem(FIX)
    sid, text = _one_unit_edit(p)
    # the stub kills itself remotely AND locally for this text, so the driver build fails too: a compile-stage
    # failure that is NOT the remote infra error text
    bad = _sources(p, **{sid: text + "\nKILLED\n"})
    ev = _ssh_ev(stubs, tmp_path)
    so, res = _build(ev, p, bad, tmp_path / "w1")
    st = ev.remote_build_stats()
    assert st["infra"] >= 1 and st["compile_errors"] == 0 and st["local"] >= 1
    assert so is None and res.stage == "compile"


def test_toolchain_mismatch_host_is_not_used(stubs, tmp_path):
    p = SlotProblem(FIX)
    other = tmp_path / "otherbin"
    other.mkdir()
    shutil.copy(stubs / "nvcc", other / "nvcc")
    (other / "version").write_text("Cuda compilation tools, release 11.8, V11.8.0\nBuild y\n")
    for n in ("flock",):
        if (stubs / n).exists():
            shutil.copy(stubs / n, other / n)
    ev = _ssh_ev(stubs, tmp_path)
    ev.remote_path = str(other)                                       # the "host" has another toolkit
    _, bad = _build(ev, p, p.seed_sources(), tmp_path / "w1")
    st = ev.remote_build_stats()
    assert bad is None and st["remote"] == 0 and st["eligible"] == [] and st["local"] > 0


def test_scheduler_spreads_over_hosts_by_in_flight(stubs, tmp_path):
    """Concurrent units go to the host with the fewest builds in flight: 2 hosts, 4 slow units -> 2 + 2."""
    (stubs / "sleep").write_text("0.6")
    p = SlotProblem(FIX)
    ev = _ssh_ev(stubs, tmp_path, hosts=("h1", "h2"), remote_build_jobs=2)
    asyncio.run(ev._rb_eligible())
    (stubs / "ssh_hosts.log").unlink(missing_ok=True)
    work = tmp_path / "wk"
    work.mkdir()

    async def go():
        async def one(i):
            src = work / f"u{i}.cu"
            src.write_text(f"// unit {i}\n")
            await ev._compile_obj(src, ["-O3"], work / f"u{i}.o", f"// unit {i}\n")
        await asyncio.gather(*[one(i) for i in range(4)])
    asyncio.run(go())
    st = ev.remote_build_stats()
    assert st["per_host"]["h1"]["builds"] == 2 and st["per_host"]["h2"]["builds"] == 2 and st["local"] == 0


def test_host_slot_limit_full_then_spill_to_driver(stubs, tmp_path):
    """A host whose build slots are all taken answers FULL; after the spill time the unit builds on the driver."""
    (stubs / "sleep").write_text("1.5")
    ev = _ssh_ev(stubs, tmp_path, remote_build_jobs=1)
    ev.rb_spill_s = 0.5
    work = tmp_path / "wk"
    work.mkdir()

    async def go():
        async def one(i):
            src = work / f"u{i}.cu"
            src.write_text(f"// unit {i}\n")
            await ev._compile_obj(src, ["-O3"], work / f"u{i}.o", f"// unit {i}\n")
        await asyncio.gather(*[one(i) for i in range(3)])
    asyncio.run(go())
    st = ev.remote_build_stats()
    assert st["remote"] == 1 and st["full"] >= 1 and st["spilled"] == 2 and st["local"] == 2
    assert all((work / f"u{i}.o").exists() for i in range(3))


def test_policy_avoid_skips_a_host_with_a_timed_run(stubs, tmp_path, monkeypatch):
    """policy avoid: a held GPU flock on the host (a timed run) -> BUSY -> the unit builds elsewhere (here: driver)."""
    import fcntl
    lockdir = tmp_path / "shm"
    lockdir.mkdir()
    if not Path("/dev/shm").is_dir():
        pytest.skip("no /dev/shm (the lock probe reads it)")
    lock = Path("/dev/shm/kernelblaster_gpu_cptest.lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        pytest.skip("lock file busy")
    try:
        monkeypatch.setenv("KERNELBLASTER_REMOTE_BUILD_POLICY", "avoid")
        ev = _ssh_ev(stubs, tmp_path)
        ev.rb_spill_s = 0.3
        work = tmp_path / "wk"
        work.mkdir()
        src = work / "u.cu"
        src.write_text("// u\n")
        asyncio.run(ev._compile_obj(src, ["-O3"], work / "u.o", "// u\n"))
        st = ev.remote_build_stats()
        assert st["busy"] >= 1 and st["remote"] == 0 and st["local"] == 1 and (work / "u.o").exists()
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


def test_driver_object_with_absolute_inputs_stays_local(stubs, tmp_path):
    ev = _ssh_ev(stubs, tmp_path)
    inc = tmp_path / "x.h"
    inc.write_text("//\n")
    assert ev._rb_applicable(Path("slot.cu"), ["-O3"], "int x;\n")
    assert not ev._rb_applicable(Path("driver.cpp"), ["-O3"], f'#include "{inc}"\n')
    assert not ev._rb_applicable(Path("driver.cpp"), ["-O3", "-I/cuda/cupti/include"], "int x;\n")
