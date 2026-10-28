# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Region loop: a bounded, tool-using agent loop for REGION slots (ModelBlaster ``--region-loop on``).

A *region slot* (ModelBlaster ``fuse_region``; slot id ``region__...``) owns a whole sub-graph (an RNN stack, a dense block, a window of
a recurrence, the model) as ONE entry point, so the slot optimiser may replace it by one persistent kernel. The flow's normal slot
step (one prompt, one reply, one evaluation) did not exploit regions: ``region:all`` was rejected on 5 of 8 branchy pilot problems
(l3_farm/graph_agent_pilot/SUMMARY.txt). The pilot's Claude Code agent, with typed tools (profile, scratch experiments, paired timing,
verify, commit / revert), moved 7 of 8 programs 1.75x-3.6x past the flow headline at $2-7 per problem. This module ports the TOOL
SEMANTICS of the pilot's ``./gt`` (not its CLI isolation) to a loop driven through the flow's own LLM path:

    RegionLoop(problem, sid, backend, llm, ...).run()  ->  result dict      # committed sources are in ``loop.current``

Model protocol over the relay
-----------------------------
The LLM relay (``servers/llm_server.py`` ``POST /v1/generate``) forwards ``{messages, n_tasks, model, role, ...}`` only, ``messages``
being ``{role, content: str}``; the provider paths (``query.generate_code_bedrock`` / ``_openai``, ``agentic_llm``) take plain text
chat completions and drop everything else: there is NO tool / function-calling pass-through, and system messages are folded into
the provider's system prompt. So the loop uses a STRUCTURED-TEXT protocol over plain chat: every assistant turn holds ONE action,

    ```action
    {"tool": "verify", "args": {}}
    ```
    ```cuda
    ...the candidate source (verify / time) or the standalone program (scratch)...
    ```

the harness executes it and answers with the result as the next user message (strictly alternating roles). Tool-calling can replace
:func:`parse_reply` / :func:`render_result` if a relay ever carries it; the tools, budgets and rules do not change.

Tools (every call is timed and counted; the result states what is left)
-----------------------------------------------------------------------
``show``     the region (members, shapes, signature, instances), the committed source (paged) / its outline, the pending candidate,
             the reference, the knowledge section, the commit history
``profile``  per-slot / per-op time of the CURRENT program through the evaluator (the driver's CUPTI attribution; no ncu on the lab
             B200 hosts), launches, launch overhead; ``ncu`` only where the evaluator can
``scratch``  compile + run a standalone CUDA program on the leased GPU (experiment_tool.ExperimentTool): clock64 probes, spin-wait
             costs, occupancy, micro-benchmarks
``edit``     change the pending candidate (else the committed source) by exact SEARCH / REPLACE blocks instead of resending a 600-line
             megakernel (output tokens are the expensive ones); the result is the new pending candidate, unverified
``verify``   static lint of the candidate, then the flow driver's checks through the evaluator (compile, the region slot's isolated
             check against the reference walk, the partition outputs, CUDA-graph capturability), then the backend's extra checks
             (repeat-run stability; NaN-poisoned / replay-with-new-data / weight-swap checks exist only in the pilot's bench: see docs)
``time``     paired candidate-vs-current measurement, ``time_reps`` (5) alternating reps, needs a passing ``verify`` of the exact source
``commit``   accept the verified, timed candidate: gain >= ``commit_gain`` (15%) in one timing, or two consecutive timings both with a gain
``revert``   drop the pending candidate, or the last N commits
``status``   budgets (turns / tool calls / usd / wall), per-tool accounting, commits, pending state
``finish``   end the loop

Budgets: ``max_turns`` (LLM calls), ``max_tool_calls``, ``max_usd`` (from the relay's usage telemetry: provider cost when reported, else
tokens x the price table), ``max_wall_s``. A budget stops the loop; whatever is committed stays; a pending candidate is NOT part of
the result. Old turns are compacted (code blocks and long results elided) so a long session stays cheap.

Everything GPU-side is behind :class:`Backend` (``EvaluatorBackend`` = the flow's Evaluator: local GPU pool or the ssh evaluator; the
tests use a fake), the LLM behind a ``llm(messages, role) -> (text, usage)`` callable (the flow's ``generate_code_retry`` or a scripted
``stub:`` model). Off by default; nothing here runs unless the ModelBlaster flag / ``KERNELBLASTER_REGION_LOOP`` is on.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional

ENV = "KERNELBLASTER_REGION_LOOP"
ENV_BUDGET = "KERNELBLASTER_REGION_LOOP_BUDGET"          # "turns,tools,usd,wall_s"
ENV_COMMIT = "KERNELBLASTER_REGION_LOOP_COMMIT"          # "commit_gain,repeat_gain" (fractions; default 0.15,0.03; a smoke test sets -1,-1)
REGION_PREFIX = "region__"
TOOLS = ("show", "profile", "scratch", "edit", "verify", "time", "commit", "revert", "status", "finish")
CODE_TOOLS = ("verify", "time", "scratch")               # tools that take a code block (``edit`` takes SEARCH/REPLACE blocks)
DEFAULT_BUDGET = {"turns": 40, "tools": 120, "usd": 8.0, "wall_s": 3600.0}


def enabled(value: Optional[str] = None, env: Mapping = os.environ) -> bool:
    v = (value if value is not None else env.get(ENV, "off")) or "off"
    v = str(v).strip().lower()
    if v in ("", "off", "0", "false", "no"):
        return False
    if v in ("on", "1", "true", "yes"):
        return True
    raise ValueError(f"--region-loop must be on|off, not {v!r}")


def parse_budget(s: Optional[str]) -> dict:
    """``"turns,tools,usd,wall_s"`` (any suffix may be omitted; empty fields keep the default) -> dict."""
    out = dict(DEFAULT_BUDGET)
    if not s:
        return out
    keys = ("turns", "tools", "usd", "wall_s")
    parts = [p.strip() for p in str(s).split(",")]
    if len(parts) > len(keys):
        raise ValueError(f"region loop budget {s!r}: expected turns,tools,usd,wall_s")
    for k, p in zip(keys, parts):
        if p:
            try:
                out[k] = float(p) if k in ("usd", "wall_s") else int(p)
            except ValueError:
                raise ValueError(f"region loop budget {s!r}: {k}={p!r} is not a number") from None
            if out[k] <= 0:
                raise ValueError(f"region loop budget {s!r}: {k} must be > 0")
    return out


@dataclass
class LoopConfig:
    max_turns: int = DEFAULT_BUDGET["turns"]
    max_tool_calls: int = DEFAULT_BUDGET["tools"]
    max_usd: float = DEFAULT_BUDGET["usd"]
    max_wall_s: float = DEFAULT_BUDGET["wall_s"]
    commit_gain: float = 0.15            # one timing with at least this gain commits (the pilot's noise level)
    repeat_gain: float = 0.03            # two consecutive timings each with at least this gain commit
    time_reps: int = 5                   # alternating candidate / current reps per ``time``
    scratch_runs: int = 12               # standalone programs per loop
    scratch_s: float = 600.0             # compile + run seconds of all of them
    max_result_chars: int = 6000         # one tool result as shown to the model
    keep_full_turns: int = 2             # older turns are compacted (code / results elided)
    max_protocol_errors: int = 4         # consecutive unparsable replies before the loop gives up
    show_lines: int = 160                # default page of ``show what=source``
    role: str = "codegen_hard"

    @classmethod
    def from_env(cls, env: Mapping = os.environ, budget: Optional[str] = None, **kw) -> "LoopConfig":
        """Budget from ``budget`` (the flag) else ``$KERNELBLASTER_REGION_LOOP_BUDGET``; commit rule from ``$KERNELBLASTER_REGION_LOOP_COMMIT``."""
        b = parse_budget(budget or env.get(ENV_BUDGET))
        c = str(env.get(ENV_COMMIT, "") or "").strip()
        if c:
            try:
                cg, rg = (float(x) for x in c.split(","))
            except ValueError:
                raise ValueError(f"{ENV_COMMIT}={c!r}: expected commit_gain,repeat_gain (fractions)") from None
            kw.setdefault("commit_gain", cg)
            kw.setdefault("repeat_gain", rg)
        return cls(max_turns=int(b["turns"]), max_tool_calls=int(b["tools"]), max_usd=float(b["usd"]),
                   max_wall_s=float(b["wall_s"]), **kw)

    def describe(self) -> dict:
        return dict(self.__dict__)

    def scaled(self, frac: float, floor_turns: int = 6) -> "LoopConfig":
        """A copy with the turn / tool / usd / wall budgets multiplied by ``frac`` (0..1): one problem's budget shared by its region slots."""
        f = max(0.0, min(1.0, float(frac)))
        d = dict(self.__dict__)
        d.update(max_turns=max(floor_turns, int(round(self.max_turns * f))), max_tool_calls=max(2 * floor_turns, int(round(self.max_tool_calls * f))),
                 max_usd=self.max_usd * f, max_wall_s=self.max_wall_s * f)
        return LoopConfig(**d)


# ------------------------------------------------------------------------------------------------------ protocol
@dataclass
class Action:
    tool: str
    args: dict = field(default_factory=dict)
    code: Optional[str] = None
    thought: str = ""
    extra_actions: int = 0
    edits: list = field(default_factory=list)          # [(old, new)] from ```edit blocks (SEARCH / REPLACE) or args.edits


class ProtocolError(Exception):
    pass


_ACTION_RE = re.compile(r"```(?:action|json)\s*\n(.*?)```", re.S)
_CODE_RE = re.compile(r"```(?:cuda|cpp|c\+\+|cu|c)\s*\n(.*?)```", re.S)
_EDIT_BLOCK_RE = re.compile(r"```edit[^\n]*\n(.*?)```", re.S)
_SR_RE = re.compile(r"<{5,9} SEARCH\n(.*?)\n={5,9}\n(.*?)\n?>{5,9} REPLACE", re.S)


def parse_reply(text: str) -> Action:
    """The one action of an assistant turn: a ```action JSON block (or ```json, or a bare JSON object) ``{"tool": ..., "args": {...}}``
    and at most one ```cuda block (the last one wins). Raises :class:`ProtocolError` with the message shown to the model."""
    text = text or ""
    blocks = _ACTION_RE.findall(text)
    cands = []
    for b in blocks:
        try:
            d = json.loads(b)
        except ValueError as e:
            raise ProtocolError(f"the action block is not valid JSON ({e}); expected {{\"tool\": \"<name>\", \"args\": {{...}}}}")
        if isinstance(d, dict) and "tool" in d:
            cands.append(d)
    if not cands:
        s = text.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = json.loads(s)
                if isinstance(d, dict) and "tool" in d:
                    cands.append(d)
            except ValueError:
                pass
    if not cands:
        raise ProtocolError("no action: reply with one ```action block holding {\"tool\": \"<name>\", \"args\": {...}} "
                            f"(tools: {', '.join(TOOLS)}); put code in a ```cuda block next to it")
    d = cands[0]
    tool = str(d.get("tool", "")).strip().lower()
    if tool not in TOOLS:
        raise ProtocolError(f"unknown tool {tool!r} (tools: {', '.join(TOOLS)})")
    args = d.get("args") or {}
    if not isinstance(args, dict):
        raise ProtocolError("args must be a JSON object")
    codes = _CODE_RE.findall(text)
    code = (codes[-1].strip() + "\n") if codes else None
    if code is None and isinstance(args.get("source"), str) and args["source"].strip():
        code = args["source"].strip() + "\n"
    first = _ACTION_RE.search(text)
    thought = (text[:first.start()] if first else "").strip()[:1500]
    edits = [(o, n) for blk in _EDIT_BLOCK_RE.findall(text) for o, n in _SR_RE.findall(blk)]
    for e in args.get("edits") or []:
        if isinstance(e, dict) and isinstance(e.get("old"), str) and isinstance(e.get("new"), str):
            edits.append((e["old"], e["new"]))
    if tool == "edit" and not edits:
        raise ProtocolError("edit needs SEARCH / REPLACE blocks: an ```edit fence holding <<<<<<< SEARCH / old text / ======= / new text / "
                            ">>>>>>> REPLACE (exact text; it must match once), or args.edits = [{\"old\": ..., \"new\": ...}]")
    return Action(tool, args, code, thought, extra_actions=len(cands) - 1, edits=edits)


def render_result(name: str, ok: bool, body: str, left: str, hint: str = "") -> str:
    """The user message that answers an action."""
    head = f"[{name}] {'ok' if ok else 'FAILED'}"
    return f"{head}\n{body.rstrip()}\n\n{left}" + (f"\n{hint}" if hint else "")


# ------------------------------------------------------------------------------------------------------ accounting
@dataclass
class Ledger:
    turns: int = 0
    tool_calls: int = 0
    usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    t0: float = field(default_factory=time.time)
    per_tool: dict = field(default_factory=dict)
    scratch_runs: int = 0
    scratch_s: float = 0.0
    protocol_errors: int = 0
    llm_seconds: float = 0.0
    price_assumed: list = field(default_factory=list)      # models charged at FALLBACK_PRICE (no price in the table)
    truncated: int = 0                                     # replies cut by the output-token limit

    def wall(self) -> float:
        return time.time() - self.t0

    def charge_tool(self, name: str, seconds: float, ok: bool) -> None:
        d = self.per_tool.setdefault(name, {"calls": 0, "seconds": 0.0, "errors": 0})
        d["calls"] += 1
        d["seconds"] = round(d["seconds"] + seconds, 3)
        d["errors"] += 0 if ok else 1

    def exhausted(self, cfg: LoopConfig) -> Optional[str]:
        if self.turns >= cfg.max_turns:
            return "turns"
        if self.tool_calls >= cfg.max_tool_calls:
            return "tool_calls"
        if self.usd >= cfg.max_usd:
            return "usd"
        if self.wall() >= cfg.max_wall_s:
            return "wall"
        return None

    def left(self, cfg: LoopConfig) -> str:
        return (f"Budget left: {cfg.max_turns - self.turns} turns, {cfg.max_tool_calls - self.tool_calls} tool calls, "
                f"${max(cfg.max_usd - self.usd, 0):.2f}, {max(cfg.max_wall_s - self.wall(), 0) / 60:.0f} min.")

    def to_json(self) -> dict:
        return {"turns": self.turns, "tool_calls": self.tool_calls, "usd": round(self.usd, 4),
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "llm_seconds": round(self.llm_seconds, 2), "wall_s": round(self.wall(), 2),
                "per_tool": self.per_tool, "scratch_runs": self.scratch_runs, "scratch_s": round(self.scratch_s, 2),
                "protocol_errors": self.protocol_errors, "truncated_replies": self.truncated,
                **({"price_assumed": sorted(set(self.price_assumed))} if self.price_assumed else {})}


FALLBACK_PRICE = (5.0, 25.0)          # USD per M tokens (in, out) when the model has no entry in the price table (an Opus-class price: never 0)
ENV_PRICE = "KERNELBLASTER_REGION_LOOP_PRICE"          # "in,out" overrides the table


def usage_usd(usage: Mapping, model: str, assumed: Optional[list] = None) -> float:
    """The cost of one call from the relay's / provider's usage: the reported ``cost_usd`` when there is one, else tokens x the price table
    (``cost_tracker.lookup_pricing``). A model with no price (the table's 0, 0) is charged ``FALLBACK_PRICE`` (or ``$KERNELBLASTER_
    REGION_LOOP_PRICE``), so the dollar budget is never blind; ``assumed`` (a list) gets the model name appended when that happened."""
    c = float(usage.get("cost_usd") or 0.0)
    if c:
        return c
    pt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    ct = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    pin = pout = 0.0
    env = os.environ.get(ENV_PRICE, "").strip()
    if env:
        try:
            pin, pout = (float(x) for x in env.split(","))
        except ValueError:
            pin = pout = 0.0
    if not (pin or pout):
        try:
            from .cost_tracker import lookup_pricing
            pin, pout = lookup_pricing(model)
        except Exception:                               # noqa: BLE001
            pin = pout = 0.0
    if not (pin or pout) and (pt or ct):
        pin, pout = FALLBACK_PRICE
        if assumed is not None:
            assumed.append(model)
    return (pt * pin + ct * pout) / 1e6


def truncated(usage: Mapping) -> bool:
    """The provider cut the reply at the output-token limit (Bedrock ``stopReason: max_tokens`` / OpenAI ``length``)."""
    tel = usage.get("telemetry") or {}
    stops = list(tel.get("stop_reasons") or []) + [tel.get("stop_reason")]
    return any(str(x).lower() in ("max_tokens", "length") for x in stops if x)


# ------------------------------------------------------------------------------------------------------ static lint
_ENTRY_FORBIDDEN = [
    (r"\bcudaDeviceSynchronize\b", "cudaDeviceSynchronize: no host synchronisation in a slot (it breaks CUDA-graph capture)"),
    (r"\bcudaMemcpy(2D|3D)?\s*\(", "blocking cudaMemcpy: no host copies in a slot (device-to-device cudaMemcpyAsync is fine)"),
    (r"\bcudaMallocHost\b|\bcudaHostAlloc\b|\bcudaMallocManaged\b|__managed__", "host / managed memory is not allowed"),
    (r"\bint\s+main\s*\(", "do not define main()"),
    (r"\blaunch_gpu_implementation\b|\bmb_run_op\b|\bmb_num_ops\b", "do not reference the walker (launch_gpu_implementation / mb_run_op)"),
    (r"#\s*include\s*[<\"]\s*(cublas\w*|cudnn\w*|cufft\w*|cusparse\w*|cusolver\w*|curand\w*|nvrtc\w*|cutlass/|cute/|thrust/|cub/)",
     "CUDA math libraries / CUTLASS / CUB are not part of the slot contract"),
    (r"use_fast_math|-ftz\s*=\s*true", "no fast-math flags"),
]
_REGION_NOTES = [
    (r"\bcudaStream(Synchronize|Query)\b", "cudaStreamSynchronize / cudaStreamQuery: host synchronisation breaks capture"),
    (r"\bcudaFree\w*\s*\(", "cudaFree in the call path: allocate once (lazily, static) and keep"),
]


def entry_name(signature: str) -> Optional[str]:
    m = re.search(r"\b(\w+)\s*\(", signature or "")
    return m.group(1) if m else None


def lint_source(code: str, entry: Optional[str], *, region: bool = True) -> tuple[list, list]:
    """(errors, warnings) of a candidate source before it costs a build: the entry function is defined, no forbidden API. A REGION slot
    may fork / join side streams and helper threads (the flow's own region seed does); a non-region slot may not."""
    errs, warns = [], []
    if not code or not code.strip():
        return ["empty source"], []
    if entry and not re.search(r"\b" + re.escape(entry) + r"\s*\(", code):
        errs.append(f"the entry function `{entry}` is not defined (keep the exact signature)")
    for pat, msg in _ENTRY_FORBIDDEN:
        if re.search(pat, code):
            errs.append(msg)
    for pat, msg in _REGION_NOTES:
        if re.search(pat, code):
            warns.append(msg)
    if not region:
        if re.search(r"\bcuda(Stream\w*|Event\w*|Graph\w*)\b|std::thread|\bpthread_", code):
            errs.append("streams / events / graphs / helper threads are allowed in REGION slots only")
    if len(code) > 400_000:
        errs.append(f"source too large ({len(code)} bytes > 400000)")
    return errs, warns


def outline(code: str) -> str:
    """A cheap map of a (possibly 300 KB) composite seed: size, namespaces, kernels, entry-like functions."""
    lines = code.splitlines()
    bits = [f"{len(lines)} lines, {len(code)} bytes"]
    ns = re.findall(r"^\s*namespace\s+(\w+)\s*\{", code, re.M)
    if ns:
        bits.append(f"{len(ns)} namespace block(s): {', '.join(ns[:12])}{' ...' if len(ns) > 12 else ''}")
    ks = re.findall(r"__global__\s+(?:void\s+)?(?:__launch_bounds__\([^)]*\)\s*)?(?:void\s+)?(\w+)\s*\(", code)
    if ks:
        bits.append(f"{len(ks)} __global__ kernel(s): {', '.join(dict.fromkeys(ks[:20]))}{' ...' if len(ks) > 20 else ''}")
    ent = re.findall(r"^(?:extern \"C\"\s+)?void\s+(kernel_\w+)\s*\(", code, re.M)
    if ent:
        bits.append(f"entry-like functions: {', '.join(dict.fromkeys(ent[:6]))}")
    cm = re.findall(r"//\s*(current kernel of [^\n]{0,120})", code)
    if cm:
        bits.append("member sources: " + "; ".join(cm[:8]) + (" ..." if len(cm) > 8 else ""))
    return "\n".join(bits)


def sha(code: str) -> str:
    return hashlib.sha1((code or "").encode()).hexdigest()[:10]


# ------------------------------------------------------------------------------------------------------ backend
class Backend:
    """The GPU side of the loop: everything the tools need, so tests run with a fake and nothing here knows how a GPU is leased."""

    arch: str = "100"

    async def evaluate(self, sources: Mapping[str, str], tag: str, *, incumbent=None, screen: bool = False, only_slots=None):
        """-> EvalResult (knowledge.slots): compile, run, the driver's checks, timings."""
        raise NotImplementedError

    def pairs_incumbent(self) -> bool:
        return False

    async def scratch(self, code: str, seconds_left: float) -> dict:
        """-> {"ok", "stage" (refused|compile|run), "exit", "out", "wall_s"}"""
        return {"ok": False, "stage": "run", "out": "this backend runs no scratch programs"}

    async def ncu(self, sources: Mapping[str, str], tag: str, sid: str):
        """-> (summary text or None, why not)"""
        return None, "ncu is not available on this backend"

    async def extra_checks(self, sources: Mapping[str, str], tag: str, first) -> list:
        """Optional further checks after the driver's: -> [{"name", "ok", "detail"}]."""
        return []


class EvaluatorBackend(Backend):
    """The flow's Evaluator (local GPU pool, or the ssh evaluator) as the loop's backend."""

    def __init__(self, evaluator, problem, arch: str, work: Path):
        self.ev, self.p, self.arch, self.work = evaluator, problem, str(arch), Path(work)
        self._xtool = None

    def pairs_incumbent(self) -> bool:
        f = getattr(self.ev, "pairs_incumbent", None)
        return bool(f is not None and f(self.p))

    async def evaluate(self, sources, tag, *, incumbent=None, screen=False, only_slots=None):
        kw: dict = {}
        if only_slots:
            kw["only_slots"] = list(only_slots)
        if screen:
            kw["screen"] = True
        if incumbent is not None and self.pairs_incumbent():
            import inspect
            try:
                if "incumbent" in inspect.signature(self.ev.evaluate).parameters:
                    kw["incumbent"] = dict(incumbent)
            except (TypeError, ValueError):
                pass
        return await self.ev.evaluate(self.p, dict(sources), tag, **kw)

    async def scratch(self, code, seconds_left):
        if hasattr(self.ev, "_ssh_base"):                # the ssh evaluator: ExperimentTool would compile and run on THIS host
            return {"ok": False, "stage": "run", "out": "scratch programs need the GPU on the host that runs this loop (this run times on "
                                                        "remote GPUs over ssh): use `profile`, `verify` and paired `time` instead"}
        if self._xtool is None:
            from . import experiment_tool
            self._xtool = experiment_tool.ExperimentTool(self.ev, self.p, self.arch, self.work / "scratch")
        return await self._xtool.run(code, seconds_left)

    async def ncu(self, sources, tag, sid):
        try:
            by, err = await self.ev.profile(self.p, dict(sources), tag, slots=[sid])
        except Exception as e:                           # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"
        if err or not by:
            return None, err or "no profile"
        from ..knowledge import ncu as ncu_mod
        return ncu_mod.render(by.get(sid) or next(iter(by.values()))), None

    async def extra_checks(self, sources, tag, first):
        """Repeat-run stability: the candidate is built and checked a SECOND time (a persistent kernel with a stale flag / counter, or a
        race, passes the first launch and fails the next). The NaN-poisoned op-by-op run, the replay with new inputs and the in-place
        weight swap of the pilot's bench are not in the flow driver (docs/region_loop.md, "Not done")."""
        r2 = await self.evaluate(sources, tag + "_repeat")
        ok = bool(r2.ok)
        return [{"name": "repeat_run", "ok": ok, "detail": "" if ok else r2.error_text(600)}]


# ------------------------------------------------------------------------------------------------------ the loop
@dataclass
class Pending:
    code: str
    sha: str
    lint_warnings: list = field(default_factory=list)
    verified: Optional[dict] = None
    timings: list = field(default_factory=list)
    n_verify: int = 0


@dataclass
class Commit:
    code: str
    sha: str
    gain: float
    cand_us: Optional[float]
    cur_us: Optional[float]
    turn: int
    before_code: str
    before_us: Optional[float]
    timings: list = field(default_factory=list)


def _median(xs: list) -> Optional[float]:
    return statistics.median(xs) if xs else None


class RegionLoop:
    """One region slot, one bounded conversation. ``current`` is the full source dict of the problem (committed state)."""

    def __init__(self, *, problem, sid: str, backend: Backend, llm: Callable[..., Awaitable], model: str, sources: Mapping[str, str],
                 baseline_us: Optional[float], cfg: Optional[LoopConfig] = None, out_dir: Optional[Path] = None,
                 system: Optional[str] = None, knowledge_text: str = "", examples: Optional[list] = None,
                 on_event: Optional[Callable[[dict], None]] = None, logger=None, tag: str = "rl"):
        self.p, self.sid, self.be, self.llm, self.model = problem, sid, backend, llm, model
        self.cfg = cfg or LoopConfig()
        self.slot = problem.slots[sid]
        self.entry = entry_name(self.slot.signature)
        self.current: dict = dict(sources)
        self.start_sources: dict = dict(sources)
        self.cur_us = baseline_us                       # deployed time of the committed state (None until profiled)
        self.start_us = baseline_us
        self.out = Path(out_dir) if out_dir else None
        self.system = system or build_system_prompt(problem, sid, arch=getattr(backend, "arch", "100"))
        self.knowledge_text = knowledge_text
        self.examples = list(examples or [])
        self.on_event = on_event or (lambda ev: None)
        self.log = logger
        self.tag = tag
        self.led = Ledger()
        self.pending: Optional[Pending] = None
        self.commits: list[Commit] = []
        self.events: list[dict] = []                    # one per tool call (the transcript's index)
        self.stop: Optional[str] = None
        self.finished_summary = ""
        self._profile_cache: dict = {}
        self.messages: list[dict] = []
        self._n_eval = 0
        self.final_res = None                           # the EvalResult of the final re-verification of the committed program

    # ---- small helpers
    def _evtag(self, what: str) -> str:
        self._n_eval += 1
        return f"{self.tag}_{self.sid[:24]}_{what}{self._n_eval}"

    def _emit(self, kind: str, **kw) -> dict:
        ev = {"kind": kind, "turn": self.led.turns, "t": round(self.led.wall(), 2), **kw}
        self.events.append(ev)
        try:
            self.on_event(ev)
        except Exception:                               # noqa: BLE001 -- a sink never breaks the loop
            pass
        if self.out is not None:
            try:
                self.out.mkdir(parents=True, exist_ok=True)
                with open(self.out / "events.jsonl", "a") as f:
                    f.write(json.dumps(ev, default=str) + "\n")
            except OSError:
                pass
        return ev

    def _left(self) -> str:
        return self.led.left(self.cfg)

    # ---- messages: compaction keeps a long session cheap
    def _compacted(self) -> list[dict]:
        msgs = self.messages
        keep_from = max(2, len(msgs) - 2 * self.cfg.keep_full_turns)
        out = []
        for i, m in enumerate(msgs):
            if i < 2 or i >= keep_from:                 # system + the task, and the last turns, verbatim
                out.append(m)
                continue
            c = m["content"]
            if m["role"] == "assistant":
                c = _CODE_RE.sub(lambda mo: "```cuda\n// [elided: %d lines, sha %s; `show what=candidate|source` gets it back]\n```"
                                 % (mo.group(1).count("\n"), sha(mo.group(1).strip() + "\n")), c)
            elif len(c) > 700:
                c = c[:500] + f"\n[... {len(c) - 500} chars elided]"
            out.append({"role": m["role"], "content": c})
        return out

    # ---- tool implementations; each returns (ok, text)
    async def t_show(self, a: Action) -> tuple:
        what = str(a.args.get("what", "region")).lower()
        slot, p = self.slot, self.p
        if what == "region":
            ops = [o for o in p.manifest.get("ops", []) if o.get("slot") == self.sid]
            bufs = {b["name"]: b for b in p.manifest.get("buffers", [])}
            lines = [f"slot `{self.sid}` (op `{slot.op}`, class `{slot.op_class}`, dtype {slot.dtype}, isolated bound {slot.tolerance})",
                     f"signature:\n{slot.signature}", f"semantics (members, checked outputs, scratch, rules):\n{slot.semantics[:5000]}",
                     "instances (shape scalars per call):"]
            lines += [f"  {i.get('op_name', '')}: {json.dumps(i.get('shape', {}), sort_keys=True)[:300]}" for i in slot.instances[:6]]
            for o in ops[:3]:
                io = ", ".join(f"{b}[{bufs.get(b, {}).get('dtype', '?')} x{bufs.get(b, {}).get('numel', '?')}]"
                               for b in (o.get("inputs", []) + o.get("outputs", []))[:16])
                lines.append(f"call `{o.get('name', o.get('op'))}` buffers: {io}")
            cur = self.current[self.sid]
            lines.append(f"current committed source: {outline(cur)}")
            return True, "\n".join(lines)
        if what in ("source", "committed"):
            return True, self._page(self.current[self.sid], a.args)
        if what == "outline":
            return True, outline(self.current[self.sid])
        if what == "candidate":
            if self.pending is None:
                return False, "no pending candidate (send one with verify / time)"
            return True, f"pending candidate sha {self.pending.sha}\n" + self._page(self.pending.code, a.args)
        if what == "reference":
            ref = p.reference_text(self.sid)
            return True, (ref[:self.cfg.max_result_chars] if ref else "no C reference for this slot")
        if what == "knowledge":
            return True, self.knowledge_text or "(no knowledge section for this region)"
        if what == "history":
            if not self.commits:
                return True, "no commits yet"
            return True, "\n".join(f"#{i + 1} turn {c.turn}: sha {c.sha}, gain {100 * c.gain:.1f}%, "
                                   f"{c.before_us or 0:.1f} -> {c.cand_us or 0:.1f} us" for i, c in enumerate(self.commits))
        return False, "what: region | source | outline | candidate | reference | knowledge | history"

    def _page(self, code: str, args: dict) -> str:
        lines = code.splitlines()
        off = max(int(args.get("offset", 0) or 0), 0)
        lim = max(1, min(int(args.get("limit", self.cfg.show_lines) or self.cfg.show_lines), 400))
        page = lines[off:off + lim]
        txt = "\n".join(f"{off + i + 1:5d}  {l}" for i, l in enumerate(page))
        more = f"\n[lines {off + 1}-{off + len(page)} of {len(lines)}; offset={off + len(page)} for the next page]" \
            if off + len(page) < len(lines) else f"\n[end: {len(lines)} lines]"
        return txt[: self.cfg.max_result_chars * 2] + more

    async def t_profile(self, a: Action) -> tuple:
        if a.args.get("ncu"):
            txt, why = await self.be.ncu(self.current, self._evtag("ncu"), self.sid)
            return (True, txt) if txt else (False, f"ncu unavailable: {why} (use the per-slot times of `profile` and `scratch` probes)")
        key = sha(json.dumps(self.current, sort_keys=True))
        if key in self._profile_cache and not a.args.get("refresh"):
            return True, self._profile_cache[key] + "\n(cached; refresh=true to re-measure)"
        r = await self.be.evaluate(self.current, self._evtag("profile"))
        if not r.ok:
            return False, "the committed program does not pass its own checks:\n" + r.error_text(1500)
        if r.deployed_us:
            self.cur_us = r.deployed_us
        total = sum(v for v in r.slot_times().values() if v) or 1.0
        rows = sorted(((v or 0.0, s) for s, v in r.slot_times().items()), reverse=True)[:12]
        lines = [f"end to end (deployed, {r.e2e_basis or 'eager'}): {r.deployed_us or 0:.1f} us; partition (eager): {r.partition_us or 0:.1f} us",
                 f"launches per pass: {r.attr_launches}; launch overhead: {r.launch_overhead_us}",
                 "per slot (attributed GPU us, share):"]
        lines += [f"  {s}{'  <- this region' if s == self.sid else ''}: {v:.1f} us ({100 * v / total:.1f}%)"
                  f"{'  launches ' + str(round(r.attr_slot_launches[s])) if s in r.attr_slot_launches else ''}" for v, s in rows]
        ops = sorted(((v, o) for o, v in (r.attr_op_us or r.op_us or {}).items()), reverse=True)[:8]
        if ops:
            lines.append("largest ops: " + ", ".join(f"{o} {v:.1f}" for v, o in ops))
        txt = "\n".join(lines)
        self._profile_cache[key] = txt
        return True, txt

    async def t_scratch(self, a: Action) -> tuple:
        code = a.code
        if not code:
            return False, "scratch needs a complete standalone CUDA program with int main() in a ```cuda block"
        if self.led.scratch_runs >= self.cfg.scratch_runs:
            return False, f"scratch budget used up ({self.cfg.scratch_runs} programs)"
        left = self.cfg.scratch_s - self.led.scratch_s
        if left <= 5:
            return False, f"scratch time budget used up ({self.cfg.scratch_s:.0f} s)"
        t0 = time.time()
        r = await self.be.scratch(code, min(left, self.cfg.max_wall_s - self.led.wall()))
        dt = time.time() - t0
        if r.get("stage") != "refused":
            self.led.scratch_runs += 1
        self.led.scratch_s += dt
        head = {"refused": "REFUSED", "compile": "compile FAILED"}.get(r.get("stage"), f"exit {r.get('exit')}")
        return bool(r.get("ok")), f"{head} ({r.get('wall_s', round(dt, 1))} s)\n```\n{(r.get('out') or '').strip()[-3000:]}\n```"

    def _take_candidate(self, a: Action) -> Optional[str]:
        """Adopt the turn's code block as the pending candidate (a new source resets verify / timings)."""
        if a.code:
            s = sha(a.code)
            if self.pending is None or self.pending.sha != s:
                self.pending = Pending(a.code, s)
        return None if self.pending is None else self.pending.sha

    async def t_edit(self, a: Action) -> tuple:
        """Apply SEARCH / REPLACE edits to the pending candidate (else the committed source): the result is the new, unverified pending."""
        base = self.pending.code if self.pending is not None else self.current[self.sid]
        code, notes = base, []
        for i, (old, new) in enumerate(a.edits, 1):
            if not old.strip():
                return False, f"edit {i}: an empty SEARCH text (nothing was changed)"
            n = code.count(old)
            if n == 0:
                return False, f"edit {i}: the SEARCH text was not found (exact match, whitespace included; `show what=candidate` for the current text). Nothing was changed."
            if n > 1 and not a.args.get("all"):
                return False, f"edit {i}: the SEARCH text matches {n} places; add surrounding lines to make it unique (or args.all=true). Nothing was changed."
            code = code.replace(old, new) if a.args.get("all") else code.replace(old, new, 1)
            notes.append(f"edit {i}: {old.count(chr(10)) + 1} -> {new.count(chr(10)) + 1} lines" + (f" (x{n})" if n > 1 else ""))
        if code == base:
            return False, "the edits change nothing"
        s_ = sha(code)
        self.pending = Pending(code, s_)
        errs, warns = lint_source(code, self.entry, region=True)
        self.pending.lint_warnings = warns
        lines = [f"pending candidate is now {s_} ({len(code.splitlines())} lines; {len(a.edits)} edit(s) applied to "
                 f"{'the pending candidate' if base is not self.current[self.sid] else 'the committed source'}): " + "; ".join(notes)]
        lines += [f"  lint error: {e}" for e in errs] + [f"  note: {w}" for w in warns]
        lines.append("next: `verify`")
        return True, "\n".join(lines)

    async def t_verify(self, a: Action) -> tuple:
        s = self._take_candidate(a)
        if s is None:
            return False, "no candidate: send the complete region source in a ```cuda block with this action"
        pend = self.pending
        pend.n_verify += 1
        errs, warns = lint_source(pend.code, self.entry, region=True)
        pend.lint_warnings = warns
        if errs:
            pend.verified = {"ok": False, "stage": "lint", "errors": errs}
            self._emit("verify", sha=s, ok=False, stage="lint", error="; ".join(errs)[:600])
            return False, "static check failed (nothing was built):\n- " + "\n- ".join(errs)
        t0 = time.time()
        r = await self.be.evaluate({**self.current, self.sid: pend.code}, self._evtag("verify"))
        dt = time.time() - t0
        res: dict = {"ok": False, "stage": r.stage, "seconds": round(dt, 1), "warnings": warns}
        if not r.ok:
            res["errors"] = [r.error_text(2500)]
            pend.verified = res
            self._emit("verify", sha=s, ok=False, stage=r.stage, error=r.error_text(1200))
            return False, f"candidate {s} FAILED at stage `{r.stage}` ({dt:.0f} s):\n{r.error_text(2500)}"
        cap = _capture_defect(r)
        if cap:
            res["errors"] = [cap]
            pend.verified = res
            self._emit("verify", sha=s, ok=False, stage="capture", error=cap)
            return False, f"candidate {s} passes its checks but breaks CUDA-graph capture of the partition ({cap}): fix it, a slot that " \
                          "cannot be captured makes the whole partition fall back to eager launches"
        extras = await self.be.extra_checks({**self.current, self.sid: pend.code}, self._evtag("x"), r)
        bad = [x for x in extras if not x.get("ok")]
        res.update(ok=not bad, extras=extras, rel_err=(r.slot_checks.get(self.sid) or {}).get("rel_err"), e2e_us=r.deployed_us,
                   slot_us=r.slot_time(self.sid))
        pend.verified = res
        if bad:
            res["errors"] = [f"{x['name']}: {x.get('detail', '')}" for x in bad]
            self._emit("verify", sha=s, ok=False, stage="extra", error="; ".join(res["errors"])[:600])
            return False, f"candidate {s} passes the driver's checks but FAILS " + "; ".join(res["errors"])
        self._emit("verify", sha=s, ok=True, stage="ok", rel_err=res["rel_err"], e2e_us=r.deployed_us)
        lines = [f"candidate {s} PASSED ({dt:.0f} s): rel_err {res['rel_err']}, region slot {res['slot_us'] or 0:.1f} us, end to end "
                 f"{r.deployed_us or 0:.1f} us ({r.e2e_basis or 'eager'})"]
        lines += [f"  extra check {x['name']}: ok" for x in extras]
        lines += [f"  note: {w}" for w in warns]
        lines.append("next: `time` (paired against the committed program), then `commit` if it wins")
        return True, "\n".join(lines)

    @staticmethod
    def _clock_suspect(res) -> bool:
        """X-149: the run was timed on a GPU whose clocks were locked / throttled and no clean GPU was left: not evidence."""
        try:
            from ..knowledge.clock_guard import clock_suspect
            return bool(clock_suspect(res))
        except Exception:                               # noqa: BLE001
            return False

    async def _paired(self, cand_code: str) -> dict:
        cand = {**self.current, self.sid: cand_code}
        reps = self.cfg.time_reps
        ratios, cand_us, cur_us = [], [], []
        paired = self.be.pairs_incumbent()
        for i in range(reps):
            if self.led.wall() >= self.cfg.max_wall_s:
                break
            if paired:
                r = await self.be.evaluate(cand, self._evtag("time"), incumbent=self.current)
                if not r.ok:
                    return {"ok": False, "error": r.error_text(1200)}
                if self._clock_suspect(r):
                    return {"ok": False, "error": "timed on a GPU with locked / throttled clocks (clock guard): the number is not evidence; "
                                                  "run `time` again"}
                rr = r.e2e_ratio or {}
                if rr.get("n") and rr.get("med"):
                    ratios.append(float(rr["med"]))
                elif r.deployed_us and r.ref_e2e_us:
                    ratios.append(r.deployed_us / r.ref_e2e_us)
                cand_us.append(r.deployed_us)
                if r.ref_e2e_us:
                    cur_us.append(r.ref_e2e_us)
            else:                                        # alternate A B A B ...: current, candidate, current, ...
                a_ = await self.be.evaluate(self.current, self._evtag("timeA"))
                b_ = await self.be.evaluate(cand, self._evtag("timeB"))
                if not b_.ok:
                    return {"ok": False, "error": b_.error_text(1200)}
                if self._clock_suspect(a_) or self._clock_suspect(b_):
                    return {"ok": False, "error": "timed on a GPU with locked / throttled clocks (clock guard): the number is not evidence; "
                                                  "run `time` again"}
                if a_.ok and a_.deployed_us and b_.deployed_us:
                    ratios.append(b_.deployed_us / a_.deployed_us)
                    cur_us.append(a_.deployed_us)
                    cand_us.append(b_.deployed_us)
        if not ratios:
            return {"ok": False, "error": "no paired timing came back"}
        med = statistics.median(ratios)
        return {"ok": True, "gain": 1.0 - med, "ratios": [round(x, 4) for x in ratios], "n": len(ratios),
                "cand_us": _median([x for x in cand_us if x]), "cur_us": _median([x for x in cur_us if x]),
                "spread": round(max(ratios) - min(ratios), 4), "basis": "paired" if paired else "alternating"}

    async def t_time(self, a: Action) -> tuple:
        s = self._take_candidate(a)
        if s is None:
            return False, "no candidate: send the source with `verify` first"
        pend = self.pending
        if not pend.verified or not pend.verified.get("ok"):
            return False, f"candidate {s} has no passing `verify` (a timing of unverified code is refused): run `verify` first"
        t0 = time.time()
        r = await self._paired(pend.code)
        dt = time.time() - t0
        if not r["ok"]:
            return False, f"timing failed: {r['error']}"
        r["seconds"] = round(dt, 1)
        pend.timings.append(r)
        if r.get("cur_us"):
            self.cur_us = r["cur_us"]
        self._emit("timing", sha=s, gain=r["gain"], cand_us=r["cand_us"], cur_us=r["cur_us"], n=r["n"], basis=r["basis"])
        ok_now, why = self._commit_rule(pend)
        lines = [f"candidate {s}: gain {100 * r['gain']:+.1f}% (candidate {r['cand_us'] or 0:.1f} us vs committed {r['cur_us'] or 0:.1f} us), "
                 f"{r['n']} {r['basis']} reps, ratios {r['ratios']}, spread {r['spread']:.3f}, {dt:.0f} s",
                 f"commit rule: gain >= {100 * self.cfg.commit_gain:.0f}% in one timing, or two consecutive timings both >= "
                 f"{100 * self.cfg.repeat_gain:.0f}%: " + ("SATISFIED, `commit` now" if ok_now else f"not yet ({why})")]
        return True, "\n".join(lines)

    def _commit_rule(self, pend: Pending) -> tuple:
        t = pend.timings
        if not t:
            return False, "no timing yet"
        last = t[-1]["gain"]
        if last >= self.cfg.commit_gain:
            return True, f"gain {100 * last:.1f}% >= {100 * self.cfg.commit_gain:.0f}%"
        if len(t) >= 2 and t[-1]["gain"] >= self.cfg.repeat_gain and t[-2]["gain"] >= self.cfg.repeat_gain:
            return True, "two consecutive timings with a gain"
        return False, f"last gain {100 * last:.1f}%; time it again to confirm a smaller gain"

    async def t_commit(self, a: Action) -> tuple:
        pend = self.pending
        if pend is None:
            return False, "nothing to commit (no pending candidate)"
        if not pend.verified or not pend.verified.get("ok"):
            return False, f"refused: candidate {pend.sha} has no passing `verify`"
        ok, why = self._commit_rule(pend)
        if not ok:
            return False, f"refused: {why}. Needs gain >= {100 * self.cfg.commit_gain:.0f}% (one timing) or two consecutive timings with a gain."
        last = pend.timings[-1]
        c = Commit(pend.code, pend.sha, last["gain"], last["cand_us"], last["cur_us"], self.led.turns, self.current[self.sid], self.cur_us,
                   list(pend.timings))
        self.commits.append(c)
        self.current = {**self.current, self.sid: pend.code}
        if last.get("cand_us"):
            self.cur_us = last["cand_us"]
        self.pending = None
        self._profile_cache.clear()
        self._emit("commit", sha=c.sha, gain=c.gain, before_us=c.before_us, after_us=self.cur_us, code=c.code, timings=c.timings)
        return True, f"committed {c.sha}: {100 * c.gain:.1f}% faster; the committed program is now {self.cur_us or 0:.1f} us end to end " \
                     f"({len(self.commits)} commit(s) so far)"

    async def t_revert(self, a: Action) -> tuple:
        n = int(a.args.get("n", 0) or 0)
        if n <= 0:
            if self.pending is None:
                return True, "no pending candidate to drop"
            sha_ = self.pending.sha
            self.pending = None
            return True, f"dropped the pending candidate {sha_}"
        n = min(n, len(self.commits))
        if n == 0:
            return False, "no commits to revert"
        undone = self.commits[-n:]
        self.current = {**self.current, self.sid: undone[0].before_code}
        self.cur_us = undone[0].before_us
        self.commits = self.commits[:-n]
        self.pending = None
        self._profile_cache.clear()
        self._emit("revert", n=n, shas=[c.sha for c in undone])
        return True, f"reverted the last {n} commit(s): {[c.sha for c in undone]}; the committed program is again {self.cur_us or 0:.1f} us"

    async def t_status(self, a: Action) -> tuple:
        p = self.pending
        lines = [self._left(), f"commits: {len(self.commits)}; committed program {self.cur_us or 0:.1f} us (start {self.start_us or 0:.1f} us)",
                 "per tool: " + ", ".join(f"{k} x{v['calls']} {v['seconds']:.0f}s" for k, v in self.led.per_tool.items())]
        if p:
            lines.append(f"pending {p.sha}: verified={None if p.verified is None else p.verified.get('ok')}, timings "
                         f"{[round(100 * t['gain'], 1) for t in p.timings]}%")
        else:
            lines.append("no pending candidate")
        return True, "\n".join(lines)

    async def t_finish(self, a: Action) -> tuple:
        self.finished_summary = str(a.args.get("summary", ""))[:1000]
        return True, "finishing"

    # ---- the conversation
    def _first_user(self) -> str:
        slot = self.slot
        parts = [f"## Task\nOptimise the REGION slot `{self.sid}` of problem `{self.p.name}`: one entry point that computes a whole sub-graph "
                 f"({slot.op_class}). Start from the committed program (the members' current kernels called in order: parity). "
                 f"Current end-to-end time: {self.cur_us or 0:.1f} us. Beat it, keep it correct, commit only verified, timed wins.",
                 f"Signature (define EXACTLY this):\n```c\n{slot.signature}\n```", f"Semantics:\n{slot.semantics[:3500]}",
                 f"Committed source: {outline(self.current[self.sid])}"]
        if self.knowledge_text:
            parts.append("## Knowledge from earlier runs\n" + self.knowledge_text[:6000])
        for label, code in self.examples[:2]:
            parts.append(f"Example verified kernel ({label}):\n```cuda\n{code[:3500]}\n```")
        parts.append(self._left() + "\nStart with `show` / `profile`, then iterate. Reply with ONE action per turn.")
        return "\n\n".join(parts)

    async def _ask(self) -> Optional[tuple]:
        t0 = time.time()
        try:
            text, usage = await self.llm(self._compacted(), self.cfg.role)
        except asyncio.CancelledError:
            raise
        except Exception as e:                           # noqa: BLE001
            self.stop = "llm_error"
            self._emit("llm_error", error=f"{type(e).__name__}: {e}"[:300])
            return None
        self.led.llm_seconds += time.time() - t0
        self.led.turns += 1
        self.led.prompt_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        self.led.completion_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        self.led.usd += usage_usd(usage, self.model, self.led.price_assumed)
        return text, usage

    async def run(self) -> dict:
        cfg = self.cfg
        self.messages = [{"role": "system", "content": self.system}, {"role": "user", "content": self._first_user()}]
        self._emit("start", model=self.model, budget=cfg.describe(), start_us=self.start_us)
        while True:
            why = self.led.exhausted(cfg)
            if why:
                self.stop = why
                break
            got = await self._ask()
            if got is None:
                break
            text, usage = got
            self.messages.append({"role": "assistant", "content": text})
            if truncated(usage):                          # an unfinished reply is never executed (a cut code block would be a wrong candidate)
                self.led.truncated += 1
                self.led.protocol_errors += 1
                self._emit("truncated", reply_chars=len(text))
                if self.led.protocol_errors >= cfg.max_protocol_errors:
                    self.stop = "protocol_errors"
                    break
                self.messages.append({"role": "user", "content": render_result(
                    "protocol", False, "your reply hit the output-token limit and was cut: NOTHING was executed. Send a smaller change: "
                    "use `edit` with SEARCH / REPLACE blocks on the pending / committed source instead of the whole file, and keep the "
                    "reasoning short.", self._left())})
                continue
            try:
                act = parse_reply(text)
            except ProtocolError as e:
                self.led.protocol_errors += 1
                self._emit("protocol_error", error=str(e), reply=text[:400])
                if self.led.protocol_errors >= cfg.max_protocol_errors:
                    self.stop = "protocol_errors"
                    break
                self.messages.append({"role": "user", "content": render_result("protocol", False, str(e), self._left())})
                continue
            self.led.protocol_errors = 0
            if self.led.tool_calls >= cfg.max_tool_calls:
                self.stop = "tool_calls"
                break
            self.led.tool_calls += 1
            t0 = time.time()
            try:
                ok, body = await getattr(self, "t_" + act.tool)(act)
            except asyncio.CancelledError:
                raise
            except Exception as e:                        # noqa: BLE001 -- a tool crash is a failed call, not a dead loop
                ok, body = False, f"tool error: {type(e).__name__}: {e}"[:800]
            dt = time.time() - t0
            self.led.charge_tool(act.tool, dt, ok)
            self._emit("tool", tool=act.tool, ok=ok, seconds=round(dt, 2), args={k: v for k, v in act.args.items() if k != "source"},
                       sha=(self.pending.sha if self.pending else None), thought=act.thought[:300], result=body[:600])
            if act.tool == "finish":
                self.stop = "finished"
                break
            hint = f"(only the first of {act.extra_actions + 1} actions was run: one action per turn)" if act.extra_actions else ""
            self.messages.append({"role": "user", "content": render_result(act.tool, ok, body[:cfg.max_result_chars], self._left(), hint)})
        return await self._finish()

    async def _finish(self) -> dict:
        """The final re-verification of the committed program (when it changed), then the result."""
        final = None
        if self.commits:
            r = await self.be.evaluate(self.current, self._evtag("final"))
            final = {"ok": bool(r.ok), "e2e_us": r.deployed_us, "error": None if r.ok else r.error_text(600)}
            if not r.ok:                                   # never hand back a program that fails its own checks
                self.current = dict(self.start_sources)
                self.cur_us = self.start_us
                final["reverted_to_start"] = True
            elif r.deployed_us:
                self.cur_us = r.deployed_us
            self.final_res = r
        res = {"sid": self.sid, "stop": self.stop or "finished", "summary": self.finished_summary, "start_us": self.start_us,
               "final_us": self.cur_us, "commits": [{"sha": c.sha, "gain": round(c.gain, 4), "turn": c.turn, "cand_us": c.cand_us,
                                                       "cur_us": c.cur_us} for c in self.commits],
               "changed": self.current[self.sid] != self.start_sources[self.sid], "final_check": final,
               "ledger": self.led.to_json(), "config": self.cfg.describe(), "model": self.model}
        if self.start_us and self.cur_us:
            res["speedup"] = round(self.start_us / self.cur_us, 4)
        self._emit("end", **{k: v for k, v in res.items() if k != "config"})
        if self.out is not None:
            try:
                self.out.mkdir(parents=True, exist_ok=True)
                (self.out / "result.json").write_text(json.dumps(res, indent=1, default=str))
            except OSError:
                pass
        return res


def _capture_defect(res) -> Optional[str]:
    """A verified candidate that broke CUDA-graph capture of a partition that was capturable (opt_multislot._capture_defect)."""
    skip = getattr(res, "graph_skip", None)
    if skip and getattr(res, "e2e_basis", None) == "eager" and getattr(res, "graph_nodes", None) in (None, 0):
        return str(skip)[:300]
    return None


# ------------------------------------------------------------------------------------------------------ prompts
REGION_RULES = """You write the CUDA source of one REGION slot: a single entry function that computes a whole sub-graph (several ops, \
views included) of a model that ModelBlaster lowered. The harness calls it once per forward pass with explicit pointers: `in_*` \
(read-only inputs), `out_*` (checked outputs: compared against the reference walk of the members), `scratch_*` (harness-allocated \
internals of the region: yours to use, not checked), `w_*` (weights, read-only, the model's own layout: re-layouts inside the kernel \
cost time on EVERY call). Scalars (shapes, strides, flags) are compile-time constants of the region.

You start from the PARITY program: the members' current kernels called back to back. The goal is the END-TO-END time of the \
deployed program (a CUDA-graph replay of the whole partition): fewer launches, no DRAM round trip between members, overlap of \
independent branches, persistent kernels for recurrences.

Contract (checked by `verify`):
- Define the entry function with EXACTLY the given signature (name, parameter types, order). No main(), no walker references.
- Launch on the default (per-thread) stream. No cudaDeviceSynchronize / cudaStreamSynchronize / blocking cudaMemcpy / host or managed \
memory / cudaFree in the call path: the whole forward pass is captured into a CUDA graph. One-time setup (cudaMalloc of static device \
scratch, cudaFuncSetAttribute, stream/event creation) happens lazily on the first call (the harness warms up before capture) and is \
kept in statics; scratch must not carry results from one call to the next except through epoch / step tags you reset correctly.
- Dtypes: see the dtype rules below (storage dtype per tensor, fp32 accumulation, round once on store). `--quant native` problems mix \
dtypes per slot: keep exactly the signature's pointer types.
- No cuBLAS / cuDNN / CUTLASS / CUB / Thrust, no fast-math flags, no compiler flag changes. Tensor cores via wmma / mma.sync / (sm_90+) \
wgmma / (sm_100) tcgen05 with fp32 accumulators.

What a REGION may do that an ordinary slot may not, and why: an ordinary slot computes ONE op, so the flow forbids streams, events and \
threads in it (the walker owns scheduling and concurrency). A region spans a whole DAG with independent branches and recurrences, and the \
flow's own region seed already forks and joins concurrent members on helper threads' per-thread streams with events (that is capturable). \
So in a region you MAY: (1) write persistent / megakernel patterns (one launch for all time steps / layers, an on-device task queue, \
ready flags, tagged exchange buffers); (2) fork / join side streams with events inside the call, and use helper threads, provided every \
fork is joined before the function returns, streams and events are created once (static) and events are timing-disabled, and nothing \
synchronises the host. Spin-waiting grids need CO-RESIDENCY: every block must be resident at once; size the grid with mb_sm_budget() \
(SMs available to this op) and the occupancy API, never assume more. A replay that hangs is killed by the harness watchdog and the \
candidate fails.

Build and hardware facts: the harness compiles every slot with nvcc -O3 -std=c++17 (sm_100 / B200: -gencode=arch=compute_100a,code=sm_100a; \
sm_100a features such as tcgen05 need that exact form; older targets use -arch=sm_<n>) and `--default-stream per-thread`. Scratch programs \
are compiled with the same target flags. Nsight Compute (ncu) is DISABLED on the lab B200 hosts: you cannot get hardware counters; measure \
with the `profile` tool (per-slot CUPTI attribution of the whole program), `scratch` micro-benchmarks (clock64 phase counters, spin-wait \
cost, occupancy via cudaOccupancyMaxActiveBlocksPerMultiprocessor, __launch_bounds__ register / smem use) and paired `time`.

Protocol (plain chat, no native tool calling): every reply is your reasoning followed by EXACTLY ONE action block\n
```action\n{"tool": "<name>", "args": {...}}\n```\n
and, for `verify` / `time` / `scratch`, one ```cuda block holding the COMPLETE candidate source (or the standalone program); for `edit` one or \
more SEARCH / REPLACE blocks in an ```edit fence. The harness \
runs the action and answers with its result. Tools:
- show {"what": "region|source|outline|candidate|reference|knowledge|history", "offset": 0, "limit": 160}  read-only; sources are paged.
- profile {"refresh": false, "ncu": false}  per-slot / per-op time of the committed program (ncu only where available).
- scratch  + ```cuda program with int main()  compile and run it on the GPU (limited runs / seconds; no file, process or network APIs).
- edit     + ```edit block(s)  exact SEARCH / REPLACE edits of the pending candidate (else the committed source): send `<<<<<<< SEARCH` / old \
text / `=======` / new text / `>>>>>>> REPLACE`; the SEARCH text must match exactly once. Use it for every change after the first full \
source: a whole megakernel costs thousands of output tokens per resend, an edit costs a few lines.
- verify   + ```cuda candidate  static lint, build, the driver's checks (the region slot against the reference walk, the partition outputs, \
CUDA-graph capture), repeat-run stability. A new source resets its verification and timings.
- time     paired measurement of the verified candidate against the committed program, 5 alternating reps; refused without a passing verify.
- commit   accept the pending candidate. Rule: gain >= 15% in one timing, or two consecutive timings both with a gain (each `time` is one \
timing). Refused otherwise. Commit often: only committed code counts; a pending candidate is lost when the budget ends.
- revert {"n": 0}  drop the pending candidate (n=0) or the last n commits.     - status  budgets and state.     - finish {"summary": "..."}  stop.
Budgets (turns, tool calls, dollars, wall time) are enforced and shown after every result; plan your steps, batch your reasoning, and \
keep the best working program committed at all times. Timing noise on a shared GPU is large: trust a gain only when `time` agrees twice or \
exceeds 15%."""


def build_system_prompt(problem, sid: str, arch: str = "100") -> str:
    """The region system prompt: the region contract + the flow's dtype rules and hardware text (opt_multislot) + the protocol."""
    from . import hw_profile
    from .opt_multislot import DTYPE_RULES, MIXED_RULE, _bound
    slot = problem.slots[sid]
    if problem.dtype == "mixed":
        sd = {s.dtype: s.tolerance for s in problem.slots.values()}
        rules = "\n".join(DTYPE_RULES.get(d, DTYPE_RULES["fp32"]).format(slot_rel=_bound(sd[d])) for d in sorted(sd)) + "\n" + MIXED_RULE
        hw_dt = "fp16" if "fp16" in sd else "fp32"
    else:
        rules = DTYPE_RULES.get(slot.dtype, DTYPE_RULES["fp32"]).format(slot_rel=_bound(slot.tolerance or problem.tolerance))
        hw_dt = slot.dtype
    hw = hw_profile.hardware_text(arch, hw_dt) if arch else ""
    return REGION_RULES + f"\n\n## Dtype rules\n{rules}" + (f"\n\n## Target hardware\n{hw}" if hw else "")


# ------------------------------------------------------------------------------------------------------ llm glue
def default_llm(model: str, logger=None, count: Optional[Callable] = None) -> Callable:
    """The flow's LLM path: ``generate_code_retry`` (the relay when KB_LLM_SERVER_URL is set; ``stub:`` models in process). ``count``
    (usage, model, role) lets the caller fold the usage into its own totals."""
    async def call(messages: list, role: str):
        from .utils import generate_code_retry
        resp = await generate_code_retry(messages, model, logger, n_tasks=1)
        u = getattr(resp, "usage", None) or {}
        if not isinstance(u, dict):
            u = {k: getattr(u, k, 0) for k in ("prompt_tokens", "completion_tokens")}
        if count is not None:
            count(u, model, role)
        return (resp.generations[0] if resp.generations else ""), u
    return call


def region_slots(problem) -> list[str]:
    """Region slot ids of a problem (``region__...``), in assembly order."""
    return [s for s in problem.order if s.startswith(REGION_PREFIX) and s in problem.slots]
