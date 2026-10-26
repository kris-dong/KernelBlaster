# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Claim consolidation in the global store (A5; opt-in, CLI ``merge consolidate``).

The bootstrap merged 53 problems and folded ZERO claims together: the merge's near-duplicate rule
(``claims.similar``: identical numbers, Jaccard >= 0.8 or SequenceMatcher >= 0.85 on the exact scope) never fires
across problems, because every problem's LLM words the same lesson differently and quotes its own shapes
("M16384xN64xK256" vs "M4,K4096: 28.7x"). This pass consolidates near-duplicates with evidence and provenance kept:

1. **blocks** -- only claims that could be merged by the claim model itself: same ``Claim.dedupe_key()`` (kind,
   level, family, full scope -- target, arch, op class, state, technique, problem / slot / impl -- and polarity:
   applicability effect, interaction relation, parameter values, ...). Problem-local claims (impl notes) are
   skipped unless ``include_local``.
2. **similarity** -- lexical, embedding-free: TF-IDF cosine over the claims' normalised content words (numbers and
   shape strings removed, light stemming) + bigrams; IDF over ALL of the partition's claims (active and retired),
   so a second pass sees the same similarities (idempotence).
3. **clusters** -- deterministic leader clustering: claims sorted by (support - contradict, support, id) desc; a claim
   joins the first leader with cosine >= ``threshold`` (compared with the leader only, no chaining).
   Optional LLM judge (``judge``: a callable, e.g. :func:`relay_judge` with a cheap model) for borderline pairs
   (``llm_low`` <= cosine < ``threshold``): it says whether two texts state the same lesson (and may give a merged
   text <= 400 chars).
4. **merge** -- the leader survives: support / contradict ids unioned (cited ids counted once), support +=
   member support - ids it already cited (same for contradict), provenance: problems / runs unioned, ``prov.consolidated``
   lists every member (id, support, contradict, first problem), ``data.consolidated_from``; members are retired with
   ``data.gk_consolidated_into`` / ``merged_into`` = survivor (rows kept: provenance is never deleted). A later merge of
   a run seeded before the consolidation passes evidence on a member to its survivor (``merge._apply_delta``).
5. **contradictions** -- opposite-polarity applicability claims of the same scope (helps vs hurts / no_effect) with
   similar text: the weak side is retired when the strong side has >= ``contra_ratio`` x its net support and the weak
   one has <= 2 (``gk_retired: contradicted by consolidated claim``); otherwise both get ``data.contested_with``.
6. **upkeep** -- the store's own claim rules run afterwards (``merge.upkeep``: contradicted >= support, stale
   ineligible, caps), so the snapshot is in the state a merge would leave it.

Prompt size stays bounded by retrieval (``KnowledgeV2.retrieve``: k claims, 3,500 chars); consolidation changes
WHICH claims fill it: fewer paraphrases of one lesson, more distinct lessons (measured in impl_db_knowledge.md).
"""
from __future__ import annotations

import contextlib
import json
import math
import re
from collections import Counter
from typing import Callable, Optional

_STOP = frozenset("a an the of to in on for and or is are be was were when with this that it its by as at from than "
                  "then use using used vs via into over under per each all any both only also not no more less most "
                  "very so such can could may might will would should does did do has have had been being which "
                  "while where these those there their they them our we you here".split())
_SHAPE = re.compile(r"\b[a-z]*\d+[a-z0-9_.x:,]*\b|\b\d+(\.\d+)?x\b", re.I)
_EFFECT_POS = {"helps", "help", "improves", "positive", "speedup", "win"}
_EFFECT_NEG = {"hurts", "regresses", "negative", "no_effect", "neutral", "slower"}


def _stem(w: str) -> str:
    for suf in ("ings", "ing", "ies", "es", "s", "ed"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def words(text: str) -> list:
    t = _SHAPE.sub(" ", (text or "").lower())
    t = re.sub(r"[^a-z_]+", " ", t)
    return [_stem(w) for w in t.split() if len(w) > 1 and w not in _STOP]


def features(text: str, bigrams: bool = False) -> Counter:
    """Content-word unigrams (calibrated on v0004: inside one exact-scope block, unigram cosine >= 0.35 pairs were the
    same lesson in a manual check of 24 sampled pairs; bigrams lowered every cosine without separating better)."""
    w = words(text)
    return Counter(w + ([f"{a} {b}" for a, b in zip(w, w[1:])] if bigrams else []))


class TfIdf:
    def __init__(self, docs: dict):
        df: Counter = Counter()
        self.tf = {i: features(t) for i, t in docs.items()}
        for tf in self.tf.values():
            df.update(tf.keys())
        n = max(len(self.tf), 1)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.vec = {}
        for i, tf in self.tf.items():
            v = {t: (1 + math.log(c)) * self.idf[t] for t, c in tf.items()}
            nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            self.vec[i] = {t: x / nrm for t, x in v.items()}

    def cos(self, a, b) -> float:
        va, vb = self.vec.get(a, {}), self.vec.get(b, {})
        if len(va) > len(vb):
            va, vb = vb, va
        return sum(x * vb.get(t, 0.0) for t, x in va.items())


def _net(d: dict) -> int:
    return int(d.get("support", 0)) - int(d.get("contradict", 0))


def _block(d: dict):
    from ...knowledge.claims import Claim
    try:
        return Claim.from_dict(d).dedupe_key()
    except (TypeError, ValueError):
        return None


def _polar_scope(d: dict):
    """Scope key without polarity (for the contradiction check) + the applicability sign."""
    if d.get("kind") != "applicability":
        return None, None
    eff = str((d.get("data") or {}).get("effect") or "").lower()
    sign = 1 if eff in _EFFECT_POS else -1 if eff in _EFFECT_NEG else 0
    if not sign:
        return None, None
    from ...knowledge.claims import SCOPE_FIELDS
    return (d.get("kind"), d.get("level"), d.get("family")) + tuple(d.get(f) for f in SCOPE_FIELDS), sign


def plan(rows: list, *, threshold: float = 0.35, include_local: bool = False, judge: Optional[Callable] = None,
         llm_low: float = 0.2, llm_max_pairs: int = 200, all_texts: Optional[dict] = None) -> dict:
    """Clusters for ``rows`` [(id, data, prov, status)] -> {"clusters": {leader: [members]}, "pairs_judged", ...}.
    ``all_texts`` (id -> text, active + retired) fixes the IDF corpus."""
    act = {cid: d for cid, d, _, st in rows if st == "active" and (include_local or not d.get("problem_id"))}
    tf = TfIdf(all_texts or {cid: d.get("text", "") for cid, d, _, _ in rows})
    blocks: dict = {}
    for cid, d in act.items():
        b = _block(d)
        if b is not None:
            blocks.setdefault(json.dumps(b, default=str), []).append(cid)
    clusters: dict = {}
    judged = 0
    llm_merges = 0
    texts: dict = {}
    for _, ids in sorted(blocks.items()):
        if len(ids) < 2:
            continue
        ids.sort(key=lambda i: (-_net(act[i]), -int(act[i].get("support", 0)), i))
        leaders: list = []
        for i in ids:
            best, bsim = None, -1.0
            for L in leaders:
                s = tf.cos(i, L)
                if s > bsim:
                    best, bsim = L, s
            if best is not None and bsim >= threshold:
                clusters.setdefault(best, []).append(i)
                continue
            if best is not None and judge is not None and bsim >= llm_low and judged < llm_max_pairs:
                judged += 1
                try:
                    same, text = judge(act[best].get("text", ""), act[i].get("text", ""))
                except Exception:                                   # noqa: BLE001 -- the judge is optional
                    same, text = False, None
                if same:
                    clusters.setdefault(best, []).append(i)
                    llm_merges += 1
                    if text:
                        texts[best] = str(text)[:400]
                    continue
            leaders.append(i)
    return {"clusters": clusters, "pairs_judged": judged, "llm_merges": llm_merges, "texts": texts,
            "blocks": sum(1 for v in blocks.values() if len(v) >= 2)}


def consolidate(g, part: str, *, seq: int, threshold: float = 0.35, include_local: bool = False,
                judge: Optional[Callable] = None, llm_low: float = 0.2, llm_max_pairs: int = 200,
                contra_ratio: float = 3.0, policy=None, dry_run: bool = False) -> dict:
    """One consolidation pass over partition ``part`` of an open snapshot connection ``g`` (in a transaction)."""
    from ...knowledge.claims import ACTIVE, MAX_CITED_IDS, RETIRED
    from .merge import Policy, _j, _now, upkeep
    pol = policy or Policy()
    rows = []
    for cid, st, data, prov in g.execute("SELECT id, status, data, prov FROM gk_claims WHERE part=? ORDER BY id", (part,)):
        try:
            rows.append((cid, json.loads(data), json.loads(prov or "{}"), st))
        except ValueError:
            continue
    by = {cid: (d, pv, st) for cid, d, pv, st in rows}
    all_texts = {cid: d.get("text", "") for cid, d, _, _ in rows}
    pl = plan(rows, threshold=threshold, include_local=include_local, judge=judge, llm_low=llm_low,
              llm_max_pairs=llm_max_pairs, all_texts=all_texts)
    before = sum(1 for *_, st in rows if st == ACTIVE)
    rep = {"active_before": before, "blocks_with_candidates": pl["blocks"], "clusters": len(pl["clusters"]),
           "merged_members": sum(len(v) for v in pl["clusters"].values()), "pairs_judged": pl["pairs_judged"],
           "llm_merges": pl["llm_merges"], "contradictions_retired": 0, "contested": 0}
    if dry_run:
        rep["examples"] = [{"survivor": by[L][0].get("text"), "members": [by[m][0].get("text") for m in ms[:3]]}
                           for L, ms in list(sorted(pl["clusters"].items(), key=lambda kv: -len(kv[1])))[:8]]
        return rep
    for L, members in sorted(pl["clusters"].items()):
        d, pv, _ = by[L]
        d, pv = dict(d), dict(pv)
        sup, con = list(d.get("support_ids") or []), list(d.get("contradict_ids") or [])
        have = set(sup) | set(con)
        cons = list(pv.get("consolidated") or [])
        for m in members:
            md, mpv, _ = by[m]
            ms_ids = [i for i in (md.get("support_ids") or []) if i not in have]
            mc_ids = [i for i in (md.get("contradict_ids") or []) if i not in have and i not in ms_ids]
            dup_s = sum(1 for i in (md.get("support_ids") or []) if i in have)
            dup_c = sum(1 for i in (md.get("contradict_ids") or []) if i in have)
            d["support"] = int(d.get("support", 0)) + max(0, int(md.get("support", 0)) - dup_s)
            d["contradict"] = int(d.get("contradict", 0)) + max(0, int(md.get("contradict", 0)) - dup_c)
            sup += ms_ids
            con += mc_ids
            have |= set(ms_ids) | set(mc_ids)
            for f in ("problems", "runs"):
                for x in mpv.get(f) or []:
                    if x not in (pv.get(f) or []):
                        pv[f] = ((pv.get(f) or []) + [x])[-20:]
            pv["n_runs"] = len(set(pv.get("runs") or []))
            pv["last_evidence"] = max(int(pv.get("last_evidence", pv.get("last_merge", 0)) or 0),
                                      int(mpv.get("last_evidence", mpv.get("last_merge", 0)) or 0))
            cons.append({"id": m, "support": md.get("support"), "contradict": md.get("contradict"),
                         "problem": (mpv.get("first") or {}).get("problem"), "seq": seq})
            md2 = dict(md, status=RETIRED)
            md2["data"] = dict(md.get("data") or {}, gk_retired=f"consolidated into {L}", gk_consolidated_into=L,
                               merged_into=L)
            g.execute("UPDATE gk_claims SET status=?, data=? WHERE part=? AND id=?", (RETIRED, _j(md2), part, m))
            by[m] = (md2, mpv, RETIRED)
        d["support_ids"] = sup[-MAX_CITED_IDS:]
        d["contradict_ids"] = con[-MAX_CITED_IDS:]
        d["data"] = dict(d.get("data") or {}, consolidated_from=sorted(set((d.get("data") or {}).get(
            "consolidated_from", [])) | set(members)))
        if L in pl["texts"]:
            d.setdefault("data", {})["text_before_consolidation"] = d.get("text")
            d["text"] = pl["texts"][L]
        d["updated_ts"] = _now()
        pv["consolidated"] = cons[-200:]
        g.execute("UPDATE gk_claims SET data=?, prov=? WHERE part=? AND id=?", (_j(d), _j(pv), part, L))
        by[L] = (d, pv, ACTIVE)
    # contradictions between opposite-polarity applicability claims of one scope
    tf = TfIdf(all_texts)
    sides: dict = {}
    for cid, (d, pv, st) in by.items():
        if st != ACTIVE or (d.get("problem_id") and not include_local):
            continue
        key, sign = _polar_scope(d)
        if key is not None:
            sides.setdefault(json.dumps(key, default=str), []).append((sign, cid))
    for _, lst in sorted(sides.items()):
        pos = [c for s, c in lst if s > 0]
        neg = [c for s, c in lst if s < 0]
        for a in sorted(pos):
            for b in sorted(neg):
                if by[a][2] != ACTIVE or by[b][2] != ACTIVE or tf.cos(a, b) < threshold:
                    continue
                na, nb = _net(by[a][0]), _net(by[b][0])
                strong, weak = (a, b) if na >= nb else (b, a)
                ns_, nw = max(_net(by[strong][0]), 0), max(_net(by[weak][0]), 0)
                wd, wpv, _ = by[weak]
                if ns_ >= contra_ratio * max(nw, 1) and int(wd.get("support", 0)) <= 2:
                    wd = dict(wd, status=RETIRED)
                    wd["data"] = dict(wd.get("data") or {}, gk_retired=f"contradicted by consolidated claim {strong} "
                                      f"(net support {ns_} vs {nw})")
                    g.execute("UPDATE gk_claims SET status=?, data=? WHERE part=? AND id=?",
                              (RETIRED, _j(wd), part, weak))
                    by[weak] = (wd, wpv, RETIRED)
                    rep["contradictions_retired"] += 1
                else:
                    for x, y in ((a, b), (b, a)):
                        xd = dict(by[x][0])
                        cw = sorted(set((xd.get("data") or {}).get("contested_with", [])) | {y})
                        if cw != sorted((xd.get("data") or {}).get("contested_with", [])):
                            xd["data"] = dict(xd.get("data") or {}, contested_with=cw)
                            g.execute("UPDATE gk_claims SET data=? WHERE part=? AND id=?", (_j(xd), part, x))
                            by[x] = (xd, by[x][1], by[x][2])
                            rep["contested"] += 1
    rep["upkeep"] = upkeep(g, part, seq, pol)
    rep["active_after"] = g.execute("SELECT COUNT(*) FROM gk_claims WHERE part=? AND status='active'", (part,)).fetchone()[0]
    return rep


def consolidate_store(store, part: Optional[str] = None, *, label: str = "consolidate", dry_run: bool = False,
                      **kw) -> dict:
    """A new snapshot = current + one consolidation pass per partition (or ``part``), under the store lock with a
    backup (``GlobalStore._backup``); ``current`` moves only after the snapshot is sealed. ``dry_run`` reports on a
    scratch copy and commits nothing."""
    from .merge import _connect, _j, _meta, _now, _set_meta
    with store.lock():
        cur = store.current_id()
        if not cur:
            return {"status": "empty"}
        if not dry_run:
            store._backup(label)
        tmp = store._start_from(cur)
        g = _connect(tmp)
        try:
            g.execute("BEGIN IMMEDIATE")
            seq = int(_meta(g, "merge_seq", 0, "gk_meta") or 0)
            parts = [part] if part else [r[0] for r in g.execute("SELECT DISTINCT part FROM gk_claims ORDER BY 1")]
            rep = {"parts": {p: consolidate(g, p, seq=seq, dry_run=dry_run, **kw) for p in parts}}
            changed = any(r.get("merged_members") or r.get("contradictions_retired") or r.get("contested")
                          or any((r.get("upkeep") or {}).values()) for r in rep["parts"].values())
            if dry_run or not changed:
                g.execute("ROLLBACK")
                g.close()
                tmp.unlink(missing_ok=True)
                rep["status"] = "dry_run" if dry_run else "no_change"
                rep["version"] = cur
                return rep
            g.execute("INSERT INTO gk_log (ts, kind, ref, part, data) VALUES (?,?,?,?,?)",
                      (_now(), "consolidate", label, part or "*", _j(rep)))
            from .refresh import lineage
            vid_next = store._next_id()
            lin = _meta(g, "lineage", None, "gk_meta")
            if lin is None:
                lin = lineage(store, cur)
            _set_meta(g, "lineage", (list(lin) + [vid_next])[-2000:], "gk_meta")
            g.execute("COMMIT")
        except BaseException:
            with contextlib.suppress(Exception):
                g.execute("ROLLBACK")
            with contextlib.suppress(Exception):
                g.close()
            tmp.unlink(missing_ok=True)
            raise
        g.close()
        vid = store._commit(tmp, parent=cur, label=label, extra={"consolidate": rep})
        with open(store.root / "history.jsonl", "a") as f:
            f.write(_j({"ts": _now(), "op": "consolidate", "version": vid, "parent": cur}) + "\n")
        rep.update(status="consolidated", version=vid, parent=cur)
        return rep


def relay_judge(model: str, *, max_tokens: int = 300, ledger: Optional[list] = None) -> Callable:
    """An LLM judge through KernelBlaster's query layer (``OPENAI_BASE_URL`` relay): (text a, text b) -> (same,
    merged text). Use a cheap model; every call's usage is appended to ``ledger``."""
    import asyncio
    import logging

    from ...agents.utils.query import generate_code_retry
    log = logging.getLogger("consolidate")

    def judge(a: str, b: str):
        msgs = [{"role": "system", "content": "You compare two lessons learned while optimizing GPU kernels."},
                {"role": "user", "content":
                 "Do these two statements express the SAME lesson (same technique, same condition, same effect; "
                 "different example shapes or numbers do not matter)?\n"
                 f"A: {a}\nB: {b}\n"
                 "Answer with JSON only: {\"same\": true|false, \"merged\": \"<one statement covering both, <= 300 "
                 "chars, keep the condition and the effect, cite at most one example>\"}"}]
        resp = asyncio.run(generate_code_retry(msgs, model, log, n_tasks=1))
        text = resp.generations[0] if resp.generations else ""
        if ledger is not None:
            ledger.append({"model": model, "usage": getattr(resp, "usage", None)})
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return False, None
        try:
            d = json.loads(m.group(0))
        except ValueError:
            return False, None
        return bool(d.get("same")), (d.get("merged") or None) if d.get("same") else None

    return judge
