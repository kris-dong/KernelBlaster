# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reading a trace store back: steps, their story, plain-file and HTML export
(``scripts/trace_view.py``; docs/trace.md, "Viewer")."""
from __future__ import annotations

import difflib
import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .core import read_ref

FAIL = ("compile_error", "runtime_error", "wrong_output")


def find_stores(path) -> list[Path]:
    """Trace store roots at / below ``path`` (a run dir, a store, or a ModelBlaster work dir)."""
    p = Path(path)
    if (p / "index.jsonl").exists() and (p / "blobs").is_dir():
        return [p]
    out = []
    if (p / "trace" / "index.jsonl").exists():
        out.append(p / "trace")
    for f in sorted(p.rglob("trace/index.jsonl")):
        if f.parent not in out:
            out.append(f.parent)
    return out


def read_index(root: Path) -> list[dict]:
    recs = []
    try:
        with open(Path(root) / "index.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except ValueError:                     # a torn last line (killed writer)
                    pass
    except OSError:
        pass
    return recs


@dataclass
class Step:
    problem: str
    rollout: int
    step: int
    run: str = ""
    input: Optional[dict] = None
    llm: dict = field(default_factory=dict)          # attempt -> latest llm record
    llm_tries: dict = field(default_factory=dict)    # attempt -> every llm record (errors, retries)
    candidates: dict = field(default_factory=dict)   # attempt -> candidate record
    replays: dict = field(default_factory=dict)      # attempt -> llm_replay (response from the resume journal)
    evals: list = field(default_factory=list)        # eval records (candidate, fixes, confirm, profile)
    record: Optional[dict] = None                    # the decision ("step")
    error: Optional[dict] = None                     # step_error

    @property
    def key(self) -> tuple:
        return (self.problem, self.rollout, self.step)

    def attempts(self) -> list:
        ks = set(self.llm) | set(self.candidates) | set(self.replays) | {e.get("attempt") for e in self.evals
                                                     if isinstance(e.get("attempt"), int)}
        return sorted(k for k in ks if isinstance(k, int))


class Trace:
    """One store: records deduplicated by id (the latest wins: a resumed step
    re-recorded under the same id replaces the earlier copy)."""

    def __init__(self, root):
        self.root = Path(root)
        self.raw = read_index(self.root)
        by: dict[str, dict] = {}
        self.dupes: dict[str, int] = {}
        for r in self.raw:
            rid = r.get("id")
            if rid in by:
                self.dupes[rid] = self.dupes.get(rid, 0) + 1
            by[rid] = r
        self.recs = list(by.values())
        self.children: dict[str, list] = {}
        for r in self.recs:
            if r.get("t") in ("build", "driver", "ncu") and r.get("eval_occ"):
                self.children.setdefault(r["eval_occ"], []).append(r)

    def text(self, ref) -> Optional[str]:
        return read_ref(self.root, ref)

    def of(self, *kinds) -> list[dict]:
        return [r for r in self.recs if r.get("t") in kinds]

    def steps(self) -> list[Step]:
        out: dict[tuple, Step] = {}
        for r in self.recs:
            if r.get("rollout") is None or r.get("step") is None:
                continue
            k = (r.get("problem") or "?", r["rollout"], r["step"])
            s = out.get(k) or out.setdefault(k, Step(*k, run=r.get("run") or ""))
            t = r.get("t")
            if t == "step_input":
                s.input = r
            elif t == "llm":
                a = r.get("attempt", 0)
                s.llm_tries.setdefault(a, [])
                if r.get("ok") or a not in s.llm:
                    s.llm[a] = r
            elif t == "candidate":
                s.candidates[r.get("attempt", 0)] = r
            elif t == "llm_replay":
                s.replays[r.get("attempt", 0)] = r
            elif t == "eval":
                s.evals.append(r)
            elif t == "step":
                s.record = r
            elif t == "step_error":
                s.error = r
        for r in self.raw:                         # every call, also failed / retried ones
            if r.get("t") == "llm" and r.get("rollout") is not None:
                k = (r.get("problem") or "?", r["rollout"], r["step"])
                if k in out:
                    out[k].llm_tries.setdefault(r.get("attempt", 0), []).append(r)
        for s in out.values():
            s.evals.sort(key=lambda e: e.get("ts", 0))
        return sorted(out.values(), key=lambda s: (s.problem, s.rollout, s.step))

    def problem_evals(self) -> list[dict]:
        """Evaluations outside any step (baseline, store seeds, final composite)."""
        return sorted([r for r in self.of("eval") if r.get("step") is None], key=lambda r: r.get("ts", 0))

    def llm_by_request(self, rid: str) -> list[dict]:
        return [r for r in self.raw if r.get("t") == "llm" and str(r.get("request_id", "")).startswith(rid)]

    def mb_records(self) -> list[dict]:
        return sorted([r for r in self.recs if str(r.get("t", "")).startswith("mb.")], key=lambda r: r.get("ts", 0))


# ---------------------------------------------------------------- text rendering

def _abridge(text: str, head: int = 25, tail: int = 15) -> str:
    lines = text.splitlines()
    if len(lines) <= head + tail + 3:
        return text
    return "\n".join(lines[:head] + [f"    ... [{len(lines) - head - tail} lines; --full shows them] ..."]
                     + lines[-tail:])


def _ind(text: str, pre: str = "    ") -> str:
    return "\n".join(pre + l for l in (text or "").splitlines())


def step_row(s: Step) -> dict:
    r = s.record or {}
    tok_in = sum(((x.get("usage") or {}).get("input_tokens") or 0) for x in s.llm.values())
    tok_out = sum(((x.get("usage") or {}).get("output_tokens") or 0) for x in s.llm.values())
    return {"problem": s.problem, "r": s.rollout, "s": s.step,
            "slot": (s.input or {}).get("slot") or (r.get("steplog") or {}).get("slot"),
            "technique": (s.input or {}).get("technique") or (r.get("steplog") or {}).get("technique"),
            "status": r.get("status") or ("error" if s.error else "?"), "fixes": r.get("fix_attempts"),
            "decision": r.get("decision") or ("error" if s.error else "?"), "reward": r.get("reward"),
            "llm": sum(len(v) for v in s.llm_tries.values()) or len(s.llm), "tok_in": tok_in, "tok_out": tok_out,
            "seconds": r.get("seconds")}


def list_table(steps: Iterable[Step]) -> str:
    rows = [step_row(s) for s in steps]
    if not rows:
        return "(no steps)"
    cols = ["problem", "r", "s", "slot", "technique", "status", "fixes", "decision", "reward", "llm", "tok_in",
            "tok_out", "seconds"]

    def f(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return "" if v is None else str(v)
    w = {c: max(len(c), *(len(f(r[c])) for r in rows)) for c in cols}
    w["problem"] = min(w["problem"], 40)
    out = ["  ".join(c.ljust(w[c]) for c in cols)]
    for r in rows:
        out.append("  ".join(f(r[c])[:w[c]].ljust(w[c]) for c in cols))
    return "\n".join(out)


def _eval_line(e: dict) -> str:
    res = e.get("result") or {}
    bits = [f"[{e.get('purpose')}] {e.get('tag')}: " + ("ok" if res.get("ok") else f"FAILED ({res.get('stage')})")]
    for k in ("e2e_us", "partition_us"):
        if res.get(k):
            bits.append(f"{k}={res[k]:.2f}")
            break
    if res.get("e2e_ratio"):
        bits.append(f"ratio={res['e2e_ratio'].get('med')}")
    if e.get("seconds") is not None:
        bits.append(f"{e['seconds']:.2f}s")
    if e.get("evaluator"):
        bits.append(e["evaluator"])
    if res.get("runtime"):
        bits.append(res["runtime"])
    return " ".join(bits)


def render_eval(tr: Trace, e: dict, full: bool = False) -> list[str]:
    out = ["  " + _eval_line(e)]
    res = e.get("result") or {}
    if res.get("errors"):
        out.append("    errors:\n" + _ind("\n".join(map(str, res["errors"]))[-3000:], "      "))
    bad = {s: c for s, c in (res.get("slot_checks") or {}).items() if not c.get("ok")}
    if bad:
        out.append(f"    failing slot checks: {bad}")
    if e.get("error"):
        out.append(f"    evaluator raised: {e['error']}")
    for c in sorted(tr.children.get(e.get("occ"), []), key=lambda c: c.get("ts", 0)):
        t = tr.text(c.get("output")) or ""
        if c["t"] == "build":
            cmd = c.get("cmd")
            cmd = " ".join(cmd) if isinstance(cmd, list) else (cmd or c.get("tool"))
            failed = c.get("rc") not in (0, None)
            out.append(f"    build rc={c.get('rc')} {c.get('seconds') or ''}s: {cmd if full else cmd[:160]}")
            if failed or full:
                out.append(_ind(t if full else _abridge(t, 30, 10), "      | "))
        elif c["t"] == "driver":
            mb = [l for l in t.splitlines() if l.startswith(("MB_E2E", "MB_ERROR", "MB_CHECK", "MB_SLOT_CHECK",
                                                             "MB_PARTITION_US", "MB_GRAPH_SKIP", "passed"))]
            out.append(f"    driver output ({c.get('chars')} chars)" + (":" if mb or full else ""))
            out.append(_ind(t if full else "\n".join(mb[:30]), "      | "))
        elif c["t"] == "ncu":
            out.append(f"    ncu raw CSV ({c.get('chars')} chars)" + (":\n" + _ind(t, "      | ") if full else ""))
    return out


def render_messages(tr: Trace, rec: dict, full: bool) -> list[str]:
    out = []
    for m in rec.get("messages") or []:
        t = tr.text(m.get("content")) or "(not stored: meta level)"
        out.append(f"  --- {m.get('role')} ({len(t)} chars)")
        out.append(_ind(t if full else _abridge(t, 6 if m.get("role") == "system" else 30, 20)))
    return out


def render_step(tr: Trace, s: Step, *, full: bool = False, diff: bool = False) -> str:
    out = [f"=== {s.problem} rollout {s.rollout} step {s.step}"]
    inp = s.input or {}
    r = s.record or {}
    out.append(f"slot {inp.get('slot')}  technique {inp.get('technique')}  state {inp.get('state')}")
    before = tr.text(inp.get("source")) if inp else None
    for a in s.attempts():
        label = "candidate (codegen)" if a == 0 else f"fix attempt {a}"
        out.append(f"\n--- attempt {a}: {label}")
        llm = s.llm.get(a)
        if llm:
            u = llm.get("usage") or {}
            out.append(f"LLM {llm.get('role')} {llm.get('model')} request_id={llm.get('request_id')} "
                       f"{llm.get('seconds')}s in={u.get('input_tokens')} out={u.get('output_tokens')} "
                       f"stop={u.get('stop_reason')}" + ("" if llm.get("ok") else f" ERROR {llm.get('error')}"))
            tries = s.llm_tries.get(a) or []
            errs = [t for t in tries if not t.get("ok")]
            if errs:
                out.append(f"  ({len(errs)} failed provider call(s) before: {errs[-1].get('error')})")
            out.append("prompt:")
            out += render_messages(tr, llm, full)
            resp = tr.text(llm.get("response"))
            out.append(f"response ({len(resp or '')} chars):")
            out.append(_ind(resp if full or resp is None or len(resp) < 1500 else _abridge(resp, 20, 10)
                            or "(empty)"))
        rp = s.replays.get(a)
        if rp:
            out.append(f"response replayed from the resume journal (key {rp.get('journal_key')}; original call "
                       f"{rp.get('replay_of')}" + (")" if llm else ", not in this trace):"))
            if not llm:
                out.append(_ind(tr.text(rp.get("response")) or "(not stored)"))
        c = s.candidates.get(a)
        if c:
            if c.get("no_code"):
                out.append(f"candidate: NONE -- {c.get('reason')} (the raw response is above)")
            else:
                code = tr.text(c.get("code")) or ""
                out.append(f"candidate source ({c.get('lines')} lines, sha {str(c.get('code_sha'))[:12]}):")
                prev = before if a == 0 else tr.text((s.candidates.get(a - 1) or {}).get("code")) or before
                if diff and prev is not None:
                    d = difflib.unified_diff(prev.splitlines(), code.splitlines(), "before", f"attempt{a}",
                                             lineterm="", n=2)
                    out.append(_ind("\n".join(d) or "(identical)"))
                else:
                    out.append(_ind(code if full else _abridge(code, 40, 10)))
        for e in s.evals:
            if e.get("attempt") == a:
                out += render_eval(tr, e, full)
    other = [e for e in s.evals if not isinstance(e.get("attempt"), int)]
    if other:
        out.append("\n--- confirm / profile runs")
        for e in other:
            out += render_eval(tr, e, full)
    out.append("\n--- decision")
    if r:
        sl = r.get("steplog") or {}
        out.append(f"{r.get('decision')}: {r.get('why')}")
        out.append(f"status {r.get('status')} reward {r.get('reward')} fixes {r.get('fix_attempts')} "
                   f"slot {sl.get('slot_us_before')} -> {sl.get('slot_us_after')} us  e2e {sl.get('e2e_before')} -> "
                   f"{sl.get('e2e_after')} us  {r.get('seconds')}s")
        out.append(f"links: result.json steps[{r.get('steps_index')}] ({r.get('result_json')}), outcome "
                   f"{r.get('outcome_id')}, telemetry step span {r.get('tel_span')}")
    if s.error:
        out.append(f"step raised: {s.error.get('error')}")
        tb = tr.text(s.error.get("traceback"))
        if tb:
            out.append(_ind(tb if full else _abridge(tb, 5, 10)))
    if not r and not s.error:
        out.append("(no decision record: the step did not finish)")
    return "\n".join(out)


def render_llm(tr: Trace, rec: dict, full: bool = True) -> str:
    u = rec.get("usage") or {}
    out = [f"=== LLM call {rec.get('request_id')} ({rec.get('id')})",
           f"role {rec.get('role')} model {rec.get('model')} ok {rec.get('ok')} {rec.get('seconds')}s "
           f"in={u.get('input_tokens')} out={u.get('output_tokens')} stop={u.get('stop_reason')} "
           f"telemetry span {rec.get('tel_span')}"]
    if rec.get("error"):
        out.append(f"error: {rec['error']}")
    out += render_messages(tr, rec, full)
    out.append("response:")
    out.append(_ind(tr.text(rec.get("response")) or "(none)"))
    return "\n".join(out)


def render_mb(tr: Trace, full: bool = False) -> str:
    out = []
    for r in tr.mb_records():
        head = f"[{r.get('t')}] " + " ".join(f"{k}={r[k]}" for k in ("mb_round", "ir", "hint", "decision", "status",
                                                                       "rc", "seconds") if r.get(k) is not None)
        out.append(head)
        for k in ("why", "kb_trace", "run_dir", "work"):
            if r.get(k):
                out.append(f"    {k}: {r[k]}")
        for k, ref in (r.get("files") or {}).items():
            out.append(f"    file {k} ({(ref or {}).get('n')} bytes)")
        if full and r.get("output"):
            out.append(_ind(tr.text(r["output"]) or "", "    | "))
    return "\n".join(out) or "(no ModelBlaster records)"


# ---------------------------------------------------------------- export

def _safe(s) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(s))[:120]


def export(tr: Trace, dest, steps: Optional[list] = None) -> Path:
    """Plain files: ``dest/<problem>/r<R>_s<S>/`` with the input source, each attempt's
    prompt / response / candidate / evaluation / build and driver output, and step.json."""
    dest = Path(dest)
    ext = ".cu"
    for s in steps if steps is not None else tr.steps():
        d = dest / _safe(s.problem) / f"r{s.rollout}_s{s.step}"
        d.mkdir(parents=True, exist_ok=True)
        if s.input and tr.text(s.input.get("source")) is not None:
            (d / f"00_input{ext}").write_text(tr.text(s.input["source"]))
        for a in s.attempts():
            llm = s.llm.get(a)
            if llm:
                msgs = "\n\n".join(f"### {m.get('role')}\n\n{tr.text(m.get('content')) or ''}"
                                   for m in llm.get("messages") or [])
                (d / f"a{a}_prompt.md").write_text(msgs)
                (d / f"a{a}_response.md").write_text(tr.text(llm.get("response")) or "")
                (d / f"a{a}_llm.json").write_text(json.dumps({k: v for k, v in llm.items() if k != "messages"},
                                                             indent=1))
            c = s.candidates.get(a)
            if c and not c.get("no_code"):
                (d / f"a{a}_candidate{ext}").write_text(tr.text(c.get("code")) or "")
        for e in s.evals:
            tag = _safe(e.get("tag"))
            (d / f"eval_{tag}.json").write_text(json.dumps(e, indent=1))
            for c in tr.children.get(e.get("occ"), []):
                (d / f"eval_{tag}_{c['id'].rsplit('/', 1)[-1]}.txt").write_text(
                    (" ".join(c["cmd"]) + "\n\n" if isinstance(c.get("cmd"), list) else "")
                    + (tr.text(c.get("output")) or ""))
        if s.record:
            (d / "step.json").write_text(json.dumps(s.record, indent=1))
        (d / "story.txt").write_text(render_step(tr, s, full=True, diff=True))
    return dest


def export_html(tr: Trace, dest, steps: Optional[list] = None, title: str = "KernelBlaster trace") -> Path:
    """One self-contained HTML file (no scripts, no external resources): the step
    table and every step's story in collapsible sections."""
    steps = steps if steps is not None else tr.steps()
    e = html.escape
    css = (":root{--bg:#fff;--fg:#1d1d1f;--mute:#666;--line:#ddd;--bad:#b3261e;--good:#1b6e3a;--code:#f5f5f7}"
           "@media (prefers-color-scheme:dark){:root{--bg:#121212;--fg:#e8e8e8;--mute:#aaa;--line:#333;"
           "--bad:#f2b8b5;--good:#8fd6a5;--code:#1e1e1e}}"
           "body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:16px;max-width:1200px}"
           "table{border-collapse:collapse;font-size:12px;display:block;overflow-x:auto}"
           "td,th{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left;white-space:nowrap}"
           "pre{background:var(--code);padding:8px;overflow-x:auto;font-size:12px;white-space:pre-wrap}"
           "details{border-left:3px solid var(--line);padding-left:8px;margin:6px 0}"
           "summary{cursor:pointer}.bad{color:var(--bad)}.good{color:var(--good)}.mute{color:var(--mute)}")
    rows = []
    for s in steps:
        r = step_row(s)
        cls = "bad" if r["status"] in FAIL or r["decision"] == "error" else "good" if r["decision"] == "merged" else ""
        rows.append(f"<tr class='{cls}'>" + "".join(f"<td>{e('' if r[c] is None else str(r[c]))}</td>" for c in
                                                   ("problem", "r", "s", "slot", "technique", "status", "fixes",
                                                    "decision", "reward", "llm", "seconds")) + "</tr>")
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
             f"content='width=device-width,initial-scale=1'><title>{e(title)}</title><style>{css}</style></head><body>",
             f"<h1>{e(title)}</h1><p class='mute'>{e(str(tr.root))} &middot; {len(tr.recs)} records &middot; "
             "contains prompts and generated code</p>",
             "<table><tr>" + "".join(f"<th>{c}</th>" for c in ("problem", "r", "s", "slot", "technique", "status",
                                                              "fixes", "decision", "reward", "llm", "s"))
             + "</tr>" + "".join(rows) + "</table>"]
    for s in steps:
        r = step_row(s)
        parts.append(f"<details><summary>{e(s.problem)} r{s.rollout} s{s.step}: {e(str(r['slot']))} "
                     f"{e(str(r['technique']))} &rarr; <b>{e(str(r['status']))}</b> / {e(str(r['decision']))}"
                     f"</summary><pre>{e(render_step(tr, s, full=True, diff=True))}</pre></details>")
    parts.append("</body></html>")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(parts))
    return dest
