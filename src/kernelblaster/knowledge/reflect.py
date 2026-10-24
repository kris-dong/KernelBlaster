# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The semantic update pass: outcome records -> validated claim edits.

A cheap model reads a bounded batch of :class:`OutcomeRecord` plus the
claims currently relevant to them and answers with JSON edits. Every edit
is schema-checked (:func:`validate_edit`) against the outcome log and the
claim store before it is applied; anything malformed, uncited, citing ids
that are not in the log, or crossing targets is rejected and logged. The
model never writes free text into the store directly.

Cost is bounded by construction: one LLM call per reflection, at most
``max_batch`` records and ``max_prompt_claims`` claims in the prompt, at
most ``max_edits`` edits applied, and reflections happen only every
``every_n`` new records (``maybe_reflect``). ``consolidate`` adds one more
call every ``consolidate_every`` reflections.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

from .claims import (
    ACTIVE,
    APPLICABILITY,
    CLAIM_KINDS,
    IMPL_NOTE,
    INTERACTION,
    MAX_TEXT_CHARS,
    PARAMETER,
    PITFALL,
    REVISE_DESCRIPTION,
    STATE_SPLIT,
    Claim,
    ClaimStore,
)
from .outcomes import (
    COMPILE_ERROR,
    IMPROVED,
    OP_CLASSES,
    RUNTIME_ERROR,
    WRONG_OUTPUT,
    OutcomeLog,
    OutcomeRecord,
)

SUPPORT = "support"
CONTRADICT = "contradict"
EDIT_KINDS = CLAIM_KINDS + (SUPPORT, CONTRADICT)

EFFECTS = ("helps", "no_effect", "hurts")
RELATIONS = ("works_after", "works_before", "undoes", "conflicts", "enables")
NOTE_TYPES = ("why_fast", "dead_end")
ERROR_STATUSES = (COMPILE_ERROR, WRONG_OUTPUT, RUNTIME_ERROR)
ANY = ("*", "any", "all")
#: kinds whose scope.state is not inherited from the cited records
STATELESS_KINDS = (PITFALL, REVISE_DESCRIPTION, IMPL_NOTE)

#: documented in docs/semantic_update_pass.md; also shown to the model
EDIT_SCHEMA = {
    "edits": [{
        "kind": "|".join(EDIT_KINDS),
        "text": "one sentence, <= 300 chars (not for support/contradict)",
        "evidence": ["outcome record ids that support this edit (required, >= 1)"],
        "counter_evidence": ["optional: record ids that cut against a NEW claim"],
        "claim_id": "existing claim id (support/contradict only)",
        "scope": {"arch": "...|*", "op_class": "...|*", "technique": "...|*", "state": "...|*",
                  "problem_id": "impl_note only", "slot": "impl_note only", "impl_id": "impl_note only"},
        "data": "flat object whose keys depend on kind, see DATA_BY_KIND",
    }]
}
#: the flat ``data`` object each kind takes (revise_description / support / contradict: none)
DATA_BY_KIND = {
    "applicability": {"effect": "|".join(EFFECTS)},
    "pitfall": {"symptom": "error / failure seen", "fix": "what fixed it (optional)"},
    "interaction": {"other": "technique name", "relation": "|".join(RELATIONS)},
    "parameter": {"params": {"name": "value"}},
    "state_split": {"proposed_state": "snake_case_name", "criterion": "what separates the groups"},
    "impl_note": {"note_type": "|".join(NOTE_TYPES)},
}
EXAMPLE_EDIT = {"kind": "applicability", "text": "<when it helps, with the condition>",
                "evidence": ["<record id>", "<record id>"],
                "scope": {"technique": "<technique>", "op_class": "gemm"},
                "data": {"effect": "helps"}}


class EditError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _common(values: Iterable[Optional[str]]) -> Optional[str]:
    vals = set(values)
    return vals.pop() if len(vals) == 1 else None


def _str(v: Any, name: str, *, max_len: int = MAX_TEXT_CHARS, required: bool = True) -> Optional[str]:
    if v is None or (isinstance(v, str) and not v.strip()):
        if required:
            raise EditError(f"missing {name}")
        return None
    if not isinstance(v, str):
        raise EditError(f"{name} must be a string")
    return v.strip()[:max_len]


def _ids(v: Any, name: str, index: dict, *, required: bool) -> list[str]:
    if v is None:
        v = []
    if not isinstance(v, list) or not all(isinstance(i, str) for i in v):
        raise EditError(f"{name} must be a list of record ids")
    if required and not v:
        raise EditError(f"{name} is empty (every edit must cite outcome records)")
    missing = [i for i in v if i not in index]
    if missing:
        raise EditError(f"{name} cites ids not in the outcome log: {missing[:5]}")
    return list(dict.fromkeys(v))


def validate_edit(edit: Any, *, index: dict[str, OutcomeRecord], claims: ClaimStore,
                  known_techniques: Optional[set] = None,
                  known_states: Optional[set] = None) -> dict:
    """Check one raw edit and resolve its scope. Returns a normalised edit; raises EditError."""
    if not isinstance(edit, dict):
        raise EditError("edit is not an object")
    kind = edit.get("kind")
    if kind not in EDIT_KINDS:
        raise EditError(f"unknown edit kind {kind!r}")
    evidence = _ids(edit.get("evidence"), "evidence", index, required=True)
    recs = [index[i] for i in evidence]
    targets = {r.target for r in recs}
    if len(targets) != 1:
        raise EditError(f"evidence spans several targets {sorted(targets)}")
    target = targets.pop()

    if kind in (SUPPORT, CONTRADICT):
        cid = edit.get("claim_id")
        c = claims.get(cid) if isinstance(cid, str) else None
        if c is None:
            raise EditError(f"unknown claim id {cid!r}")
        if c.status != ACTIVE:
            raise EditError(f"claim {cid} is retired")
        if c.target != target:
            raise EditError(f"evidence target {target} != claim target {c.target}")
        return {"kind": kind, "claim_id": cid, "evidence": evidence}

    text = _str(edit.get("text"), "text")
    scope_in = edit.get("scope") or {}
    if not isinstance(scope_in, dict):
        raise EditError("scope must be an object")
    if scope_in.get("target") not in (None, target):
        raise EditError(f"scope target {scope_in.get('target')!r} != evidence target {target}")
    data = edit.get("data") or {}
    if not isinstance(data, dict):
        raise EditError("data must be an object")
    if set(data) == {kind} and isinstance(data[kind], dict):
        data = data[kind]   # tolerate {"data": {"<kind>": {...}}} (seen from sonnet)
    counter = _ids(edit.get("counter_evidence"), "counter_evidence", index, required=False)
    if any(index[i].target != target for i in counter):
        raise EditError("counter_evidence crosses targets")
    counter = [i for i in counter if i not in evidence]

    def resolve(name: str, inherited: Optional[str], allowed: Optional[set]) -> Optional[str]:
        v = scope_in.get(name)
        if v is None:
            return inherited
        if not isinstance(v, str):
            raise EditError(f"scope.{name} must be a string")
        if v.strip().lower() in ANY:
            return None
        if allowed is not None and v not in allowed:
            raise EditError(f"scope.{name}={v!r} not known / not in the cited records")
        return v

    rec_techs = {r.technique for r in recs}
    rec_states = {s for r in recs for s in (r.state_before, r.state_after) if s}
    all_techs = (set(known_techniques) if known_techniques else set()) | {r.technique for r in index.values()}
    all_states = (set(known_states) if known_states else set()) | rec_states
    scope = {
        "target": target,
        "arch": resolve("arch", _common(r.arch for r in recs), {r.arch for r in recs if r.arch}),
        "op_class": resolve("op_class", _common(r.op_class for r in recs), set(OP_CLASSES)),
        "technique": resolve("technique", _common(rec_techs), all_techs),
        # failures / descriptions / impl notes are not bottleneck-state dependent
        "state": resolve("state", None if kind in STATELESS_KINDS else _common(r.state_before for r in recs),
                         all_states),
        "problem_id": None, "slot": None, "impl_id": None,
    }
    statuses = {r.status for r in recs}
    out_data: dict = {}

    if kind == APPLICABILITY:
        eff = data.get("effect")
        if eff not in EFFECTS:
            raise EditError(f"applicability.effect must be one of {EFFECTS}")
        if eff == "helps" and IMPROVED not in statuses:
            raise EditError("'helps' claim cites no improved record")
        if eff != "helps" and statuses == {IMPROVED}:
            raise EditError(f"'{eff}' claim cites only improved records")
        if scope["technique"] is None:
            raise EditError("applicability needs a technique")
        out_data = {"effect": eff}
    elif kind == PITFALL:
        if not statuses & set(ERROR_STATUSES):
            raise EditError("pitfall cites no compile/wrong-output/runtime failure")
        out_data = {"symptom": _str(data.get("symptom"), "data.symptom", max_len=200)}
        fix = _str(data.get("fix"), "data.fix", max_len=300, required=False)
        if fix:
            out_data["fix"] = fix
    elif kind == INTERACTION:
        rel, other = data.get("relation"), data.get("other")
        if rel not in RELATIONS:
            raise EditError(f"interaction.relation must be one of {RELATIONS}")
        if not isinstance(other, str) or other not in all_techs:
            raise EditError(f"interaction.other={other!r} is not a known technique")
        if scope["technique"] is None or other == scope["technique"]:
            raise EditError("interaction needs a technique distinct from data.other")
        out_data = {"relation": rel, "other": other}
    elif kind == PARAMETER:
        params = data.get("params")
        if (not isinstance(params, dict) or not params or len(params) > 8
                or not all(isinstance(k, str) and isinstance(v, (str, int, float, bool))
                           for k, v in params.items())):
            raise EditError("parameter.params must be a non-empty flat object (<= 8 keys)")
        if IMPROVED not in statuses:
            raise EditError("parameter claim cites no improved record")
        if scope["technique"] is None:
            raise EditError("parameter needs a technique")
        out_data = {"params": params}
    elif kind == REVISE_DESCRIPTION:
        if scope["technique"] is None:
            raise EditError("revise_description needs a technique")
    elif kind == STATE_SPLIT:
        ps = data.get("proposed_state")
        if not isinstance(ps, str) or not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", ps):
            raise EditError("state_split.proposed_state must be snake_case")
        if scope["state"] is None or scope["technique"] is None:
            raise EditError("state_split needs a state and a technique")
        if len(recs) < 2 or len({r.status for r in recs}) < 2:
            raise EditError("state_split needs >= 2 records with differing outcomes")
        out_data = {"proposed_state": ps,
                    "criterion": _str(data.get("criterion"), "data.criterion", max_len=300)}
    elif kind == IMPL_NOTE:
        nt = data.get("note_type")
        if nt not in NOTE_TYPES:
            raise EditError(f"impl_note.note_type must be one of {NOTE_TYPES}")
        impl_ids = {i for r in recs for i in (r.impl_id, r.parent_impl_id) if i}
        pid = scope_in.get("problem_id") or _common(r.problem_id for r in recs)
        if pid is not None and pid not in {r.problem_id for r in recs}:
            raise EditError(f"scope.problem_id={pid!r} not in the cited records")
        slot = scope_in.get("slot") if "slot" in scope_in else _common(r.slot for r in recs)
        if slot is not None and slot not in {r.slot for r in recs}:
            raise EditError(f"scope.slot={slot!r} not in the cited records")
        impl = scope_in.get("impl_id")
        if impl is not None and impl not in impl_ids:
            raise EditError(f"scope.impl_id={impl!r} not in the cited records")
        if impl is None and nt == "why_fast":
            impl = _common(r.impl_id for r in recs if r.impl_id)
        if pid is None and impl is None:
            raise EditError("impl_note needs a problem_id or impl_id")
        if nt == "why_fast" and IMPROVED not in statuses:
            raise EditError("why_fast note cites no improved record")
        scope.update(problem_id=pid, slot=slot, impl_id=impl)
        out_data = {"note_type": nt}

    return {"kind": kind, "text": text, "scope": scope, "data": out_data,
            "evidence": evidence, "counter_evidence": counter}


def apply_edit(e: dict, claims: ClaimStore, *, source: str = "reflect") -> dict:
    """Apply one *validated* edit; returns a record of what changed."""
    if e["kind"] == SUPPORT:
        n = claims.support(e["claim_id"], e["evidence"])
        c = claims.get(e["claim_id"])
        return {"kind": SUPPORT, "claim_id": c.id, "result": "supported", "new_ids": n,
                "support": c.support, "contradict": c.contradict, "evidence": e["evidence"]}
    if e["kind"] == CONTRADICT:
        n = claims.contradict(e["claim_id"], e["evidence"])
        c = claims.get(e["claim_id"])
        return {"kind": CONTRADICT, "claim_id": c.id, "result": "contradicted", "new_ids": n,
                "support": c.support, "contradict": c.contradict, "evidence": e["evidence"]}
    claim = Claim(kind=e["kind"], text=e["text"], data=e["data"], source=source, **e["scope"])
    c, result = claims.add(claim, e["evidence"], e.get("counter_evidence", ()))
    return {"kind": e["kind"], "claim_id": c.id, "result": result, "text": c.text,
            "scope": {k: v for k, v in e["scope"].items() if v is not None}, "data": c.data,
            "support": c.support, "contradict": c.contradict, "evidence": e["evidence"]}


def parse_json_response(text: str) -> Any:
    """Pull one JSON value out of a model response (tolerates fences / prose)."""
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_, close in (("{", "}"), ("[", "]")):
        i, j = t.find(open_), t.rfind(close)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object in response")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

REFLECT_SYSTEM = """You maintain a knowledge base of GPU / CPU kernel-optimisation lessons.
You read OUTCOME RECORDS (one per optimisation attempt: technique, op class, arch,
bottleneck state before/after, status, speedup vs the parent kernel, profiler metric
deltas, compiler/verify error, short diff summary) and the EXISTING CLAIMS relevant to
them, and you return structured edits to the claims. You never restate a record; you
extract reusable, falsifiable lessons.

Edit kinds:
- applicability: when a technique helps / has no effect / hurts (data.effect). Say WHEN
  (shape, access pattern, metric condition), not just that it did.
- pitfall: a recurring compile / correctness / runtime failure for a technique and the
  fix that worked (data.symptom, data.fix).
- interaction: technique A works_after / works_before / undoes / conflicts with / enables
  technique B (data.other = B, scope.technique = A).
- parameter: tile sizes / block shapes / unroll factors that won (data.params), per arch.
- revise_description: a better one-paragraph description of what a technique should do
  for this kind of kernel (text = the description).
- state_split: one technique shows two clearly different outcome groups inside one state;
  propose a finer state (data.proposed_state, data.criterion). Proposal only.
- impl_note: why a winning implementation is fast (note_type why_fast), or that an
  approach on this problem/slot is a dead end (note_type dead_end) so parallel searches
  stop repeating it.
- support / contradict: the batch confirms / refutes an EXISTING claim (claim_id).

Rules:
- Every edit cites the record ids that justify it in "evidence" (ids from the batch or
  ids already cited by an existing claim). Never invent ids.
- Prefer support/contradict on an existing claim over adding a near-duplicate.
  Contradicting evidence goes in a contradict edit; do not rewrite the claim.
- Scope narrowly. arch / op_class / technique / state default to the common value of the
  cited records; set a field to "*" only when the lesson clearly generalises. Never mix
  targets.
- One record alone rarely justifies an applicability or parameter claim; a single failure
  with a clear error is enough for a pitfall or a dead_end note.
- Neutral (+-2%) results are weak evidence; noise is not a lesson.
- At most {max_edits} edits. Fewer good edits beat many vague ones. Return {{"edits": []}}
  if nothing is worth recording.

Return ONLY a JSON object of this shape (no prose, no markdown fences):
{schema}

DATA_BY_KIND -- "data" is FLAT, e.g. "data": {{"effect": "helps"}}, never nested under the kind:
{data_by_kind}

Example edit:
{example}
"""


def _fmt_record(r: OutcomeRecord, err_chars: int = 300) -> str:
    d = {"id": r.id, "technique": r.technique, "status": r.status, "target": r.target}
    for k in ("arch", "op_class", "op_kind", "shape_bucket", "dtype", "problem_id", "slot",
              "state_before", "state_after", "impl_id", "parent_impl_id"):
        v = getattr(r, k)
        if v not in (None, ""):
            d[k] = v
    if r.speedup is not None:
        d["speedup"] = round(float(r.speedup), 3)
    if r.metrics_delta:
        top = sorted(r.metrics_delta.items(), key=lambda kv: -abs(kv[1]) if isinstance(kv[1], (int, float)) else 0)
        d["metrics_delta"] = dict(top[:6])
    if r.diff_summary:
        d["diff"] = r.diff_summary[:300]
    if r.error:
        d["error"] = r.error[:err_chars]
    return json.dumps(d, sort_keys=False)


def _fmt_claim(c: Claim) -> str:
    d = {"id": c.id, "kind": c.kind, "text": c.text, "support": c.support,
         "contradict": c.contradict}
    for f in ("arch", "op_class", "technique", "state", "problem_id", "slot", "impl_id"):
        if getattr(c, f):
            d[f] = getattr(c, f)
    if c.data:
        d["data"] = c.data
    d["cites"] = c.support_ids[-4:]
    return json.dumps(d)


def build_reflect_messages(records: list[OutcomeRecord], existing: list[Claim], *,
                           max_edits: int, known_techniques: Optional[Iterable[str]] = None) -> list:
    system = REFLECT_SYSTEM.format(max_edits=max_edits, schema=json.dumps(EDIT_SCHEMA, indent=1),
                                   data_by_kind=json.dumps(DATA_BY_KIND, indent=1),
                                   example=json.dumps(EXAMPLE_EDIT))
    parts = [f"OUTCOME RECORDS ({len(records)}):"]
    parts += [_fmt_record(r) for r in records]
    parts.append("")
    parts.append(f"EXISTING CLAIMS ({len(existing)}; support/contradict are counts of distinct records):")
    parts += [_fmt_claim(c) for c in existing] or ["(none)"]
    if known_techniques:
        parts.append("")
        parts.append("KNOWN TECHNIQUES: " + ", ".join(sorted(known_techniques)[:80]))
    parts.append("")
    parts.append('Return {"edits": [...]} now.')
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(parts)}]


CONSOLIDATE_SYSTEM = """You compact a knowledge base of kernel-optimisation claims.
Find claims that say the same thing in different words. Only merge claims that have the
same kind and the same scope (arch, op_class, technique, state, problem/slot) and the
same polarity (e.g. never merge "helps" with "hurts"). Optionally give a shorter, clearer
text for the kept claim (<= 300 chars) that preserves every condition and number.
Return ONLY JSON: {"merges": [{"keep": "<claim id>", "absorb": ["<claim id>", ...],
"text": "<optional new text>"}]}. Return {"merges": []} if nothing is redundant."""


# ---------------------------------------------------------------------------
# Updater
# ---------------------------------------------------------------------------


class SemanticUpdater:
    """Runs the reflection pass over an :class:`OutcomeLog` into a :class:`ClaimStore`."""

    def __init__(self, outcome_log: OutcomeLog, claims: ClaimStore, *, model: str,
                 every_n: int = 8, logger=None, max_batch: int = 24, max_edits: int = 12,
                 max_prompt_claims: int = 40, consolidate_every: int = 10,
                 known_techniques: Optional[Iterable[str]] = None,
                 known_states: Optional[Iterable[str]] = None,
                 trace_dir: Optional[Path] = None, max_retries: int = 2):
        self.log = outcome_log
        self.claims = claims
        self.model = model
        self.every_n = max(1, int(every_n))
        self.logger = logger
        self.max_batch = max_batch
        self.max_edits = max_edits
        self.max_prompt_claims = max_prompt_claims
        self.consolidate_every = consolidate_every
        self.known_techniques = set(known_techniques) if known_techniques else None
        self.known_states = set(known_states) if known_states else None
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.max_retries = max_retries
        self.last_rejected: list[dict] = []
        self.total_usage: Counter = Counter()
        self._lock = asyncio.Lock()

    # -- helpers ----------------------------------------------------------
    def _log(self, level: str, msg: str) -> None:
        if self.logger is not None:
            getattr(self.logger, level, self.logger.info)(f"[semantic-kb] {msg}")

    async def _llm(self, messages: list) -> str:
        from kernelblaster.agents.utils import generate_code_retry
        resp = await generate_code_retry(messages, self.model, self.logger, n_tasks=1,
                                         max_retries=self.max_retries)
        for k, v in (getattr(resp, "usage", None) or {}).items():
            if isinstance(v, (int, float)):
                self.total_usage[k] += v
        return resp.generations[0] if resp.generations else ""

    def _trace(self, name: str, payload: dict) -> None:
        if self.trace_dir is None:
            return
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            fp = self.trace_dir / f"{name}_e{self.claims.epoch:04d}_{int(time.time() * 1000)}.json"
            fp.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        except OSError:
            pass

    def _context_claims(self, records: list[OutcomeRecord]) -> list[Claim]:
        seen: dict[str, Claim] = {}
        scopes = {(r.target, r.arch, r.op_class, r.technique) for r in records}
        per = max(4, self.max_prompt_claims // max(1, len(scopes)))
        for tgt, arch, opc, tech in scopes:
            for c in self.claims.relevant(target=tgt, arch=arch, op_class=opc, technique=tech,
                                          k=per, include_ineligible=True):
                seen.setdefault(c.id, c)
        for pid, slot, tgt in {(r.problem_id, r.slot, r.target) for r in records if r.problem_id}:
            for c in self.claims.relevant(target=tgt, problem_id=pid, slot=slot, kinds=[IMPL_NOTE],
                                          k=6, include_ineligible=True):
                seen.setdefault(c.id, c)
        out = sorted(seen.values(), key=lambda c: -c.rank(self.claims.epoch))
        return out[: self.max_prompt_claims]

    def build_messages(self, records: list[OutcomeRecord], existing: list[Claim]) -> list:
        return build_reflect_messages(records, existing, max_edits=self.max_edits,
                                      known_techniques=self.known_techniques)

    def validate_and_apply(self, raw_edits: Any, *, index: dict) -> tuple[list[dict], list[dict]]:
        if isinstance(raw_edits, dict):
            raw_edits = raw_edits.get("edits", [])
        if not isinstance(raw_edits, list):
            return [], [{"edit": raw_edits, "reason": "edits is not a list"}]
        applied, rejected = [], []
        for raw in raw_edits:
            if len(applied) >= self.max_edits:
                rejected.append({"edit": raw, "reason": "over max_edits"})
                continue
            try:
                e = validate_edit(raw, index=index, claims=self.claims,
                                  known_techniques=self.known_techniques,
                                  known_states=self.known_states)
            except EditError as exc:
                rejected.append({"edit": raw, "reason": str(exc)})
                continue
            applied.append(apply_edit(e, self.claims))
        return applied, rejected

    # -- public API -------------------------------------------------------
    async def maybe_reflect(self, *, force: bool = False) -> list[dict]:
        """Reflect when >= every_n records arrived since the last pass (``force``: >= 1).

        Non-blocking for concurrent callers: if a pass is already running this
        returns ``[]`` and the next call picks the new records up.
        """
        if self._lock.locked():
            return []
        cursor = int(self.claims.meta.get("reflect_cursor", 0))
        new, end = self.log.since(cursor)
        if not new or (len(new) < self.every_n and not force):
            return []
        async with self._lock:
            self.claims.meta["reflect_cursor"] = end
            batch = new[-self.max_batch:]
            if len(new) > len(batch):
                self._log("info", f"{len(new) - len(batch)} older records skipped (max_batch)")
            applied = await self.reflect(batch)
            n = int(self.claims.meta.get("reflections", 0))
            if self.consolidate_every and n and n % self.consolidate_every == 0:
                try:
                    await self.consolidate()
                except Exception as exc:  # consolidation is best-effort
                    self._log("warning", f"consolidate failed: {exc}")
            return applied

    async def reflect(self, records: list[OutcomeRecord]) -> list[dict]:
        """One reflection call over *records*; returns the applied edits."""
        records = list(records)[-self.max_batch:]
        if not records:
            return []
        index = {r.id: r for r in self.log}
        for r in records:   # records passed in directly must be citable too
            index.setdefault(r.id, r)
        existing = self._context_claims(records)
        messages = self.build_messages(records, existing)
        raw_text, parsed, err = "", None, None
        try:
            raw_text = await self._llm(messages)
            parsed = parse_json_response(raw_text)
        except Exception as exc:
            err = str(exc)
            self._log("warning", f"reflection call/parse failed: {exc}")
        applied, rejected = ([], []) if parsed is None else self.validate_and_apply(parsed, index=index)
        self.last_rejected = rejected
        for rj in rejected:
            self._log("info", f"rejected edit: {rj['reason']}")
        self.claims.meta["reflections"] = int(self.claims.meta.get("reflections", 0)) + 1
        tick = self.claims.tick()
        self.claims.save()
        self._log("info", f"reflection over {len(records)} records: {len(applied)} applied, "
                          f"{len(rejected)} rejected, {tick}")
        self._trace("reflect", {"model": self.model, "messages": messages, "response": raw_text,
                                "error": err, "applied": applied, "rejected": rejected,
                                "tick": tick})
        return applied

    async def consolidate(self, *, max_claims: int = 60) -> list[dict]:
        """Deterministic dedupe, then (if the store is big enough) one LLM merge pass."""
        merged = self.claims.dedupe()
        results: list[dict] = [{"result": "dedupe", "absorbed": merged}] if merged else []
        by_target = Counter(c.target for c in self.claims.active())
        if not by_target:
            self.claims.save()
            return results
        target, n = by_target.most_common(1)[0]
        if n >= 6:
            cands = sorted((c for c in self.claims.active() if c.target == target),
                           key=lambda c: -c.rank(self.claims.epoch))[:max_claims]
            messages = [{"role": "system", "content": CONSOLIDATE_SYSTEM},
                        {"role": "user", "content": "CLAIMS:\n" + "\n".join(_fmt_claim(c) for c in cands)}]
            raw = ""
            try:
                raw = await self._llm(messages)
                merges = parse_json_response(raw)
                merges = merges.get("merges", []) if isinstance(merges, dict) else merges
            except Exception as exc:
                self._log("warning", f"consolidation call/parse failed: {exc}")
                merges = []
            for m in merges if isinstance(merges, list) else []:
                try:
                    keep = self.claims.get(m.get("keep"))
                    absorb = [self.claims.get(a) for a in m.get("absorb", [])]
                    if keep is None or keep.status != ACTIVE or not absorb or any(
                            a is None or a.status != ACTIVE or a.id == keep.id
                            or a.dedupe_key() != keep.dedupe_key() for a in absorb):
                        raise EditError("merge ids unknown, retired, or of different kind/scope")
                    text = m.get("text")
                    text = _str(text, "text", required=False) if text else None
                    self.claims.merge(keep.id, [a.id for a in absorb], text=text)
                    results.append({"result": "merged", "keep": keep.id,
                                    "absorbed": [a.id for a in absorb], "text": text})
                except (EditError, AttributeError, TypeError) as exc:
                    results.append({"result": "rejected", "merge": m, "reason": str(exc)})
            self._trace("consolidate", {"messages": messages, "response": raw, "results": results})
        self.claims.save()
        return results
