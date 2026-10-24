# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``KnowledgeV2``: the facade the agents hold (design sections 9-10).

    adapter -> classify -> select (techniques + hypotheses, hierarchical UCB,
    exploration budget) -> prompt from retrieval -> record (metric vectors,
    state chain, item id, mechanism check, stats, hypothesis attempt)

plus :meth:`maybe_upkeep`, the slower passes: hypothesis decisions and the
automatic rollback of promotions / lifts (no LLM),
hypothesis generation and state discovery (one LLM call each, on the
``upkeep_model``, only when deterministic triggers fire) and probation.

Everything lives in ``<root>/knowledge.db``; the audit log of automatic
changes (nothing waits on it) is mirrored to ``<root>/review.jsonl``. Seeds (targets, states, techniques, the
legacy technique DB) are loaded idempotently on first open.
"""
from __future__ import annotations

import asyncio
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..outcomes import IMPROVED, OutcomeRecord
from . import discovery as D
from . import metrics as M
from . import predicates as P
from .hypotheses import EXPLORE_FRAC, Hypothesis, HypothesisBook, build_generation_messages, validate_spec
from .ontology import MECH_UNKNOWN, Ontology, Technique, mechanism_check
from .states import Classification, State, StateHierarchy, load_seed
from .stats import HierStats, path as stat_path
from .store import KnowledgeDB, SQLiteClaimStore, SQLiteOutcomeLog
from .targets import DATA_DIR, TargetRegistry

SEED_VERSION = 3              # 2: scope tree, node-level techniques; 3: comprehensive seed


@dataclass
class Choice:
    item_id: str
    kind: str                      # technique | hypothesis
    desc: str                      # what the codegen prompt is told to do
    est: float = 0.0
    n_local: float = 0.0
    evidence: str = "untested"
    explore: bool = False
    ranked: list = field(default_factory=list)   # [(item, est, n_local)] for logs

    @property
    def hypothesis_id(self) -> Optional[str]:
        return self.item_id if self.kind == "hypothesis" else None


class KnowledgeV2:
    def __init__(self, root: Path, *, upkeep_model: Optional[str] = None, logger=None,
                 explore_frac: float = EXPLORE_FRAC, kappa: float = 3.0, ucb_c: float = 1.0,
                 upkeep_every: int = 8, max_new_hypotheses: int = 3, seed_mined: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.review_path = self.root / "review.jsonl"
        self.db = KnowledgeDB(self.root / "knowledge.db")
        self.registry = TargetRegistry()
        self.logger = logger
        self.upkeep_model = upkeep_model
        self.explore_frac, self.ucb_c = explore_frac, ucb_c
        self.upkeep_every, self.max_new_hyp = upkeep_every, max_new_hypotheses
        self._seed()
        self._load_scopes()
        self.hierarchy = self._load_hierarchy()
        self.ontology = self._load_ontology()
        self.outcomes = SQLiteOutcomeLog(self.db)
        self.claims = SQLiteClaimStore(self.db, registry=self.registry, hierarchy=self.hierarchy)
        if seed_mined:
            self._seed_mined()
        self.stats = HierStats(self.registry, self.db, kappa=kappa)
        self.hypotheses = HypothesisBook(self.db, self.registry, self.ontology)
        self.n_selected = self.n_explore = 0
        self.usage: Counter = Counter()
        self.events: list[dict] = []              # upkeep events of this process (for result.json)
        self._upkeep_lock = asyncio.Lock()

    # ---------------------------------------------------------------- seeding / loading
    def _seed(self) -> None:
        """Load the seed data idempotently; a newer SEED_VERSION refreshes seed-sourced rows only
        (discovered states, promoted techniques and runtime scopes are never touched)."""
        have = self.db.get_meta("seed_version") or 0
        if have >= SEED_VERSION:
            return
        for k, t in self.registry.targets.items():
            self.db.put_target(k, t.node, t.to_dict())
        states, legacy = load_seed()
        old_states = {d["id"]: d for d in self.db.rows("states")}
        for s in states:
            if old_states.get(s.id, {}).get("source", "seed") == "seed":
                self.db.put("states", s.id, s.to_dict())
        o = Ontology.seed(self.registry, legacy=True)       # reads the old JSON DB + footers once
        old_t = {d["id"]: d for d in self.db.techniques()}
        for t in o.techniques.values():
            src = old_t.get(t.id, {}).get("source", t.source)
            if not src.startswith("hypothesis:"):
                self.db.put("techniques", t.id, t.to_dict())
        self.db.set_meta("legacy_state_map", legacy)
        self.db.set_meta("seed_version", SEED_VERSION)

    def _seed_mined(self) -> None:
        """Fold ``mined.json`` (scripts/mine_knowledge.py) in once: its outcomes become statistics
        (evidence), its claims evidence-backed claims at their mined scope."""
        fp = DATA_DIR / "mined.json"
        if not fp.exists():
            return
        import hashlib
        sha = hashlib.sha1(fp.read_bytes()).hexdigest()[:16]
        if self.db.get_meta("mined_seed") == sha:
            return
        if self.db.get_meta("mined_seed"):        # a newer mined file: earlier mined stats are not re-added
            self._log("info", "mined.json changed; adding only claims not seen before")
            stats_too = False
        else:
            stats_too = True
        d = json.loads(fp.read_text())
        if stats_too:
            agg: dict = {}
            have = {r[0] for r in self.db.query("SELECT id FROM outcomes")}
            for o in d.get("outcomes", []):
                if o["item"] not in self.ontology.techniques or o["id"] in have:   # no double counting
                    continue
                for node in stat_path(self.registry, target_key=o["target"], op_class=o["op_class"],
                                      chain=o.get("chain") or [], dtype=o.get("dtype")):
                    a = agg.setdefault((o["item"], *node), [0.0, 0.0, 0.0])
                    a[0] += 1
                    a[1] += o["reward"]
                    a[2] += o["reward"] ** 2
            self.db.add_stats_bulk(agg)
        from ..claims import Claim
        from .reflect_v2 import set_claim_scope
        for c in d.get("claims", []):
            cl = Claim(kind=c["kind"], text=c["text"], technique=c["technique"], op_class=c["op_class"],
                       data={"effect": c["effect"]}, source="mined")
            first = c["targets"][0]
            set_claim_scope(cl, c["scope"] if c["scope"] in self.registry.nodes or c["scope"] == "universal"
                            else first)
            self.claims.add(cl, c["evidence"])
        self.claims.save()
        self.db.set_meta("mined_seed", sha)

    def _load_scopes(self) -> None:
        """Scope nodes and targets added at run time (DB) on top of targets.json."""
        for n in self.db.scopes():
            try:
                self.registry.add_node(n["id"], n.get("parent") or "universal", n.get("description", ""),
                                       n.get("kappa"))
            except ValueError as e:
                self._log("warning", f"scope {n.get('id')}: {e}")
        for key, node, data in self.db.target_rows():
            if key in self.registry.targets:
                continue
            node = data.get("node") or (node if node in self.registry.nodes else None)
            info = {k: v for k, v in data.items() if k not in ("key", "node", "family")}
            self.registry.add_target(key, node, **info)

    def add_scope(self, nid: str, parent: str = "universal", description: str = "",
                  kappa: Optional[float] = None) -> None:
        """Add (or move) a scope node at run time; persisted and audited."""
        n = self.registry.add_node(nid, parent, description, kappa)
        self.db.put_scope(nid, parent, n.to_dict())
        self.db.audit("scope_added", nid, n.to_dict(), self.review_path)

    def add_target(self, key: str, node: str, **info) -> None:
        """Add a target under ``node`` at run time; persisted and audited."""
        t = self.registry.add_target(key, node, **info)
        self.db.put_target(t.key, t.node, t.to_dict())
        self.db.audit("target_added", t.key, t.to_dict(), self.review_path)

    def _load_hierarchy(self) -> StateHierarchy:
        rows = [State.from_dict(d) for d in self.db.rows("states")]
        h = StateHierarchy([], self.registry, self.db.get_meta("legacy_state_map", {}))
        pending = sorted(rows, key=lambda s: s.order)
        while pending:                                  # parents before children
            rest = [s for s in pending if s.parent is not None and s.parent not in h.states]
            for s in pending:
                if s not in rest:
                    h.add(s)
            if len(rest) == len(pending):
                break
            pending = rest
        return h

    def _load_ontology(self) -> Ontology:
        return Ontology([Technique.from_dict(d) for d in self.db.techniques()], self.registry)

    def _log(self, level: str, msg: str) -> None:
        if self.logger is not None:
            getattr(self.logger, level, self.logger.info)(f"[knowledge-v2] {msg}")

    # ---------------------------------------------------------------- metrics + state
    def vector(self, target_key: str, *, estimate: Optional[M.RooflineInput] = None,
               profile: Optional[dict] = None, profile_kind: Optional[str] = None) -> dict:
        t = self.registry.get(target_key)
        return M.adapter_for(t, profile_kind).vector(profile=profile, estimate=estimate)

    def classify(self, vec: dict, *, target_key: str, op_class: str, dtype: Optional[str] = None) -> Classification:
        return self.hierarchy.classify(vec, target=target_key, op_class=op_class, dtype=dtype)

    def _ctx(self, target_key: str, op_class: str, dtype: Optional[str]) -> dict:
        t = self.registry.get(target_key)
        return {"op_class": op_class, "dtype": dtype or "fp32", "family": t.family, "target": t.key}

    # ---------------------------------------------------------------- selection
    def candidates(self, *, target_key: str, op_class: str, dtype: Optional[str], chain: list, vec: dict,
                   op_kind: Optional[str] = None):
        ctx = self._ctx(target_key, op_class, dtype)
        vals = M.values(vec)
        techs = self.ontology.candidates(target_key=target_key, op_class=op_class, chain=chain,
                                         values=vals, ctx=ctx, op_kind=op_kind)
        hyps = self.hypotheses.testable(target_key=target_key, op_class=op_class, chain=chain, values=vals, ctx=ctx)
        return techs, hyps

    def select(self, *, target_key: str, op_class: str, dtype: Optional[str], cls: Classification, vec: dict,
               rng: random.Random, exclude: tuple = (), op_kind: Optional[str] = None) -> Choice:
        chain = cls.chain
        techs, hyps = self.candidates(target_key=target_key, op_class=op_class, dtype=dtype, chain=chain, vec=vec,
                                      op_kind=op_kind)
        nodes = stat_path(self.registry, target_key=target_key, op_class=op_class, chain=chain, dtype=dtype)
        self.n_selected += 1
        # exploration budget: hypotheses get ~explore_frac of the steps when one is testable here
        if hyps and (self.n_explore + 1) <= self.explore_frac * (self.n_selected + 1):
            h = hyps[0]
            self.n_explore += 1
            return Choice(h.id, "hypothesis", h.as_technique_desc(), self.stats.estimate(h.id, nodes),
                          self.stats.count(h.id, nodes[0]), "hypothesis under test", explore=True)
        items = [t.id for t in techs if t.id not in exclude] or [t.id for t in techs]
        if not items:
            raise RuntimeError(f"no technique applies to {target_key} {op_class} {cls.label()}")
        ranked = self.stats.ucb(items, nodes, rng, c=self.ucb_c, priors=self.priors(items))
        _, est, n, tid = ranked[0]
        t = self.ontology.techniques[tid]
        ev, _ = self.stats.evidence_level(tid, nodes)
        return Choice(tid, "technique", self.technique_text(t, target_key), est, n, ev,
                      ranked=[(i, round(e, 4), k) for _, e, k, i in ranked[:8]])

    def priors(self, items) -> dict:
        return {i: self.ontology.techniques[i].prior for i in items
                if i in self.ontology.techniques and self.ontology.techniques[i].prior}

    def technique_text(self, t: Technique, target_key: str) -> str:
        r = self.ontology.realization(t, target_key) or {}
        bits = [r.get("how") or t.mechanism, f"Mechanism: {t.mechanism}."]
        if t.predicts:
            bits.append("Expected measurable effect: " + ", ".join(f"{k} {v.replace('_', ' ')}"
                                                                   for k, v in t.predicts.items()) + ".")
        if t.risks:
            bits.append("Risks: " + "; ".join(t.risks) + ".")
        if r.get("params_prior"):
            bits.append(f"Parameter prior: {json.dumps(r['params_prior'])}.")
        return " ".join(bits)

    # ---------------------------------------------------------------- retrieval (prompt section)
    def retrieve(self, *, target_key: str, op_class: str, dtype: Optional[str], cls: Classification, vec: dict,
                 problem_id: Optional[str] = None, slot: Optional[str] = None, technique: Optional[str] = None,
                 max_chars: int = 3500, k_tech: int = 6, k_claims: int = 10, op_kind: Optional[str] = None) -> str:
        """Budgeted, most specific first: state chain, ranked techniques, claims, hypotheses."""
        parts = []
        if cls.chain:
            mech = self.hierarchy.states[cls.leaf].mechanism
            parts.append(f"State: {cls.label()} ({M.render(vec, keys=list(cls.decided_by)) or 'n/a'}). "
                         f"Meaning: {mech}.")
        techs, hyps = self.candidates(target_key=target_key, op_class=op_class, dtype=dtype, chain=cls.chain, vec=vec,
                                      op_kind=op_kind)
        nodes = stat_path(self.registry, target_key=target_key, op_class=op_class, chain=cls.chain, dtype=dtype)
        ranked = sorted(((self.stats.estimate(t.id, nodes, t.prior or None), t) for t in techs), key=lambda x: -x[0])
        rows = []
        for est, t in ranked[:k_tech]:
            ev, n = self.stats.evidence_level(t.id, nodes)
            if n:
                rows.append(f"- {t.id}: est. reward {est:+.3f} (evidence: {ev}, n={int(n)}) -- {t.mechanism}")
        if rows:
            parts.append("Techniques for this state, by estimated reward (shrunk toward related targets/states):\n"
                         + "\n".join(rows))
        tk = self.registry.get(target_key)
        backend, arch = tk.key.split(":", 1)
        try:
            cl = self.claims.relevant(target=backend, arch=arch, op_class=op_class, state=cls.leaf if cls.chain else None,
                                      technique=technique, problem_id=problem_id, slot=slot, k=k_claims)
            if cl:
                parts.append("Lessons (+support/-contradict, level):\n" + self.claims.render(cl))
        except Exception as e:                      # knowledge must never block a step
            self._log("warning", f"claims lookup failed: {e}")
        if hyps:
            parts.append("Hypotheses under test for this state: "
                         + "; ".join(f"{h.name} ({len(h.attempts)}/{h.budget} tries)" for h in hyps[:4]))
        text = "\n\n".join(parts)
        return text[:max_chars]

    def already_tried(self, problem_id: str, slot: Optional[str]) -> str:
        return self.claims.already_tried(problem_id, slot)

    # ---------------------------------------------------------------- recording
    def record(self, rec: OutcomeRecord, *, choice: Choice, reward: float, cls: Classification,
               before: dict, after: Optional[dict] = None, op_class: Optional[str] = None,
               dtype: Optional[str] = None, extra: Optional[dict] = None) -> dict:
        """Append the outcome with its v2 extras, update stats and the hypothesis; -> v2 extras."""
        tk = self.registry.get(rec.target, rec.arch).key
        if choice.kind == "hypothesis":
            pred = self.hypotheses.items[choice.item_id].predicts
        else:                                        # legacy flows may name techniques the ontology lacks
            t = self.ontology.get(choice.item_id)
            pred = t.predicts if t is not None else {}
        mech = mechanism_check(pred, before, after) if after else {"label": MECH_UNKNOWN, "agree": 0,
                                                                  "disagree": 0, "metrics": {}}
        measured = bool(mech["metrics"])
        rec.state_before = cls.leaf if cls.chain else None
        if after:
            ac = self.classify(after, target_key=tk, op_class=op_class or rec.op_class, dtype=dtype)
            rec.state_after = ac.leaf if ac.chain else None
        rec.metrics_delta = M.delta(before, after) if after else {}
        v2 = {"item_id": choice.item_id, "kind": choice.kind, "hypothesis_id": choice.hypothesis_id,
              "reward": round(float(reward), 6), "chain": cls.chain, "classification": cls.status,
              "metrics_before": before, "metrics_after": after or {}, "mechanism": mech,
              "est": round(choice.est, 5), "explore": choice.explore, **(extra or {})}
        self.outcomes.append(rec, v2)
        nodes = stat_path(self.registry, target_key=tk, op_class=op_class or rec.op_class, chain=cls.chain,
                          dtype=dtype)
        self.stats.update(choice.item_id, nodes, reward)
        if choice.kind == "hypothesis":
            self.hypotheses.record_attempt(choice.item_id, outcome_id=rec.id, problem_id=rec.problem_id or "",
                                           target_key=tk, reward=reward, status=rec.status,
                                           mechanism=mech["label"], measured=measured)
        return v2

    def mechanism_lines(self, ids: list[str]) -> list[str]:
        if not ids:
            return []
        q = ",".join("?" * len(ids))
        out = []
        for r in self.outcomes.rows(f"id IN ({q})", tuple(ids)):
            m = r["v2"].get("mechanism") or {}
            if m.get("metrics"):
                out.append(f"- {r['id']} {r['v2'].get('item_id')}: {m['label']} "
                           + json.dumps({k: [v['pred'], v['delta']] for k, v in m['metrics'].items()}))
        return out

    # ---------------------------------------------------------------- upkeep (slower cadence)
    async def _llm(self, messages: list) -> str:
        from kernelblaster.agents.utils import generate_code_retry
        resp = await generate_code_retry(messages, self.upkeep_model, self.logger, n_tasks=1, max_retries=2)
        for k, v in (getattr(resp, "usage", None) or {}).items():
            if isinstance(v, (int, float)):
                self.usage[k] += v
        return resp.generations[0] if resp.generations else ""

    def _event(self, ev: dict) -> None:
        self.events.append(ev)
        self._log("info", json.dumps(ev, default=str)[:400])

    async def maybe_upkeep(self, *, force: bool = False) -> list[dict]:
        if self._upkeep_lock.locked():
            return []
        async with self._upkeep_lock:
            n = len(self.outcomes)
            last = int(self.db.get_meta("upkeep_cursor", 0) or 0)
            if not force and n - last < self.upkeep_every:
                return []
            self.db.set_meta("upkeep_cursor", n)
            evs = []
            for e in self.hypotheses.review(self.review_path):
                if e["event"] == "promoted":
                    self._copy_stats(e["id"], e["technique"])
                evs.append(e)
            rows = self.outcomes.rows()
            evs += D.review_probation(self.hierarchy, rows, self.db, self.review_path)
            # no human gate: automatic rollback of promotions and lifts later evidence contradicts
            evs += self.hypotheses.revalidate(rows, self.review_path)
            from .reflect_v2 import revalidate_lifts
            index = {r.id: r for r in self.outcomes}
            evs += revalidate_lifts(self.claims, index, self.registry, self.review_path, self.db)
            self.claims.save()
            if self.upkeep_model:
                evs += await self.discover_states(rows)
                evs += await self.generate_hypotheses(rows)
            for e in evs:
                self._event(e)
            return evs

    def _copy_stats(self, src: str, dst: str) -> None:
        for key, (n, s, ss) in list(self.stats.data.items()):
            if key[0] == src:
                self.stats.data[(dst, *key[1:])] = [n, s, ss]
                with self.db.tx() as c:
                    c.execute("INSERT OR REPLACE INTO stats VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (dst, *key[1:], n, s, ss))

    # -- state discovery
    async def discover_states(self, rows: list[dict], max_calls: int = 1) -> list[dict]:
        seen = self.db.get_meta("discovery_seen", {}) or {}
        trigs = D.find_triggers(rows, self.hierarchy, seen=seen)
        evs = []
        for trig in trigs[:max_calls]:
            seen[trig["key"]] = len(trig["rows"])
            prior = [p for p in self.db.rows("proposals") if p.get("key") == trig["key"]]
            msgs = D.build_discovery_messages(trig, self.hierarchy, [p for p in prior if p["status"] == "rejected"])
            raw, prop, err = "", None, None
            try:
                from ..reflect import parse_json_response
                raw = await self._llm(msgs)
                prop = (parse_json_response(raw) or {}).get("proposal")
            except Exception as e:
                err = str(e)
            if prop is None:
                ok, reason, nums = False, err or "model declined: " + str((locals().get("raw") or "")[:200]), {}
            else:
                ok, reason, nums = D.validate_proposal(prop, trig, self.hierarchy)
            pid = f"sp_{len(prior) + 1}_{trig['key'].replace('|', '_')}"[:80]
            rec = {"id": pid, "key": trig["key"], "trigger": {k: v for k, v in trig.items() if k != "rows"},
                   "n_rows": len(trig["rows"]), "proposal": prop, "status": "admitted" if ok else "rejected",
                   "reason": reason, "numbers": nums, "model": self.upkeep_model}
            self.db.put("proposals", pid, rec, status=rec["status"])
            if ok:
                s = D.admit(prop, trig, self.hierarchy, self.db, nums, seq=max(r["seq"] for r in rows),
                            mirror=self.review_path)
                evs.append({"event": "state_admitted", "state": s.id, "parent": s.parent,
                            "predicate": P.render(s.predicate), "reason": reason})
            else:
                self.db.audit("state_rejected", pid, {"proposal": prop, "reason": reason}, self.review_path)
                evs.append({"event": "state_rejected", "proposal": pid, "reason": reason,
                            "predicate": P.render(prop["predicate"]) if isinstance(prop, dict) and
                            isinstance(prop.get("predicate"), dict) else None})
        self.db.set_meta("discovery_seen", seen)
        return evs

    # -- hypothesis generation
    def hypothesis_triggers(self, rows: list[dict], max_triggers: int = 6) -> list[dict]:
        trig: list[dict] = []
        handled = set(self.db.get_meta("hyp_triggers_seen", []) or [])
        ctxs: dict[tuple, dict] = {}
        for r in rows[-200:]:
            v2 = r.get("v2") or {}
            if not v2.get("chain"):
                continue
            tk = f"{r['target']}:{r['arch']}"
            ctxs[(tk, r["op_class"], tuple(v2["chain"]), r.get("dtype") or "fp32")] = v2.get("metrics_before") or {}
        for (tk, opc, chain, dt), vec in ctxs.items():
            techs, _ = self.candidates(target_key=tk, op_class=opc, dtype=dt, chain=list(chain), vec=vec)
            nodes = stat_path(self.registry, target_key=tk, op_class=opc, chain=list(chain), dtype=dt)
            tried = [(t.id, self.stats.estimate(t.id, nodes), self.stats.count(t.id, nodes[0])) for t in techs]
            tried_local = [x for x in tried if x[2] > 0]
            key = f"gap|{tk}|{opc}|{chain[-1]}"
            if len(tried_local) >= 4 and all(e <= 0 for _, e, _ in tried_local) and key not in handled:
                trig.append({"source": "gap", "key": key, "target": tk, "op_class": opc, "state": list(chain),
                             "tried": [{"technique": i, "est": round(e, 4), "n": n} for i, e, n in tried_local]})
            pos = sorted([x for x in tried_local if x[1] > 0], key=lambda x: -x[1])[:3]
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    a, b = pos[i][0], pos[j][0]
                    key = f"combine|{tk}|{opc}|{a}+{b}"
                    if key in handled or self._stacked(rows, a, b):
                        continue
                    trig.append({"source": "combine", "key": key, "target": tk, "op_class": opc,
                                 "state": list(chain), "techniques": [a, b],
                                 "est": [round(pos[i][1], 4), round(pos[j][1], 4)]})
            for t in self.ontology.techniques.values():
                if t.realizations and self.ontology.realization(t, tk) is None and \
                        (not t.op_classes or opc in t.op_classes):
                    key = f"port|{self.registry.get(tk).backend}|{t.id}"
                    if key not in handled:
                        trig.append({"source": "port", "key": key, "target": tk, "op_class": opc,
                                     "technique": t.id, "mechanism": t.mechanism,
                                     "realizations": t.realizations[:2]})
        for r in rows[-200:]:
            v2 = r.get("v2") or {}
            tk = f"{r['target']}:{r['arch']}"
            for rule in v2.get("ncu_rules") or []:
                key = f"profiler|{tk}|{r['op_class']}|{rule.get('rule')}"
                if (rule.get("est_speedup_pct") or 0) >= 20 and key not in handled:
                    handled.add(key)
                    trig.append({"source": "profiler", "key": key, "target": tk, "op_class": r["op_class"],
                                 "state": v2.get("chain"), "rule": rule})
            m = v2.get("mechanism") or {}
            key = f"code_mining|{r['id']}"
            if r["status"] == IMPROVED and m.get("label") == "contradicted" and key not in handled:
                trig.append({"source": "code_mining", "key": key, "target": tk, "op_class": r["op_class"],
                             "technique": v2.get("item_id"), "speedup": r.get("speedup"),
                             "mechanism_check": m.get("metrics"), "diff": (v2.get("diff_excerpt") or "")[:1500]})
        # one of each source first, then the rest (diversity under the cap)
        order = {"gap": 0, "profiler": 1, "code_mining": 2, "combine": 3, "port": 4}
        dedup = {t["key"]: t for t in trig}
        firsts, rest, got = [], [], set()
        for t in sorted(dedup.values(), key=lambda t: order.get(t["source"], 9)):
            (rest if t["source"] in got else firsts).append(t)
            got.add(t["source"])
        return (firsts + rest)[:max_triggers]

    @staticmethod
    def _stacked(rows: list[dict], a: str, b: str) -> bool:
        for r in rows:
            v2 = r.get("v2") or {}
            lin = v2.get("lineage") or []
            it = v2.get("item_id") or r["technique"]
            if r["status"] == IMPROVED and ((it == b and a in lin) or (it == a and b in lin)):
                return True
        return False

    async def generate_hypotheses(self, rows: list[dict]) -> list[dict]:
        trigs = self.hypothesis_triggers(rows)
        if not trigs:
            return []
        seen = set(self.db.get_meta("hyp_triggers_seen", []) or []) | {t["key"] for t in trigs}
        self.db.set_meta("hyp_triggers_seen", sorted(seen))
        techs = [{"id": t.id, "mechanism": t.mechanism} for t in self.ontology.techniques.values() if t.slot]
        hyps = [h.to_dict() for h in self.hypotheses.items.values() if h.kind == "slot"]
        msgs = build_generation_messages(trigs, techniques=techs, hypotheses=hyps,
                                         metrics=[f"{k} ({u})" for k, (_, u) in M.CANONICAL.items()],
                                         k=self.max_new_hyp)
        from ..reflect import parse_json_response
        try:
            raw = await self._llm(msgs)
            specs = (parse_json_response(raw) or {}).get("hypotheses") or []
        except Exception as e:
            return [{"event": "hypothesis_generation_failed", "reason": str(e)}]
        evs = []
        for sp in specs[: self.max_new_hyp]:
            try:
                ti = sp.get("trigger") if isinstance(sp, dict) else None
                trig = trigs[ti] if isinstance(ti, int) and 0 <= ti < len(trigs) else trigs[0]
                h = validate_spec(sp, ontology=self.ontology, existing=list(self.hypotheses.items.values()),
                                  target_key=trig["target"], known_states=set(self.hierarchy.states))
                h.trigger = {k: v for k, v in trig.items() if k not in ("diff",)}
                if not h.op_classes and trig.get("op_class"):
                    h.op_classes = [trig["op_class"]]
                self.hypotheses.add(h, self.review_path)
                evs.append({"event": "hypothesis_proposed", "id": h.id, "name": h.name, "source": h.source,
                            "target": h.target_key, "states": h.states})
            except Exception as e:
                evs.append({"event": "hypothesis_rejected", "spec": (sp or {}).get("name") if isinstance(sp, dict)
                            else None, "reason": str(e)})
        return evs

    # ---------------------------------------------------------------- reporting
    def summary(self) -> dict:
        hs = Counter(h.status for h in self.hypotheses.items.values())
        st = Counter(s.status for s in self.hierarchy.states.values())
        return {"hypotheses": dict(hs), "states": dict(st), "events": self.events[-50:],
                "explore_steps": self.n_explore, "selected": self.n_selected,
                "upkeep_usage": dict(self.usage)}
