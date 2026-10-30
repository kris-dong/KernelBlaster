"""Offline checks of knowledge v2 on recorded outcomes (docs/knowledge_architecture.md section 12).

    python scripts/eval_knowledge_v2.py --outcomes A/outcomes.jsonl [B/knowledge.db ...] \
        --problems DIR [DIR ...] [--out report.json] [--permutations 2000]

Inputs are v1 ``outcomes.jsonl`` logs (roofline vectors are rebuilt from the
problem manifests + the record's measured slot time; v1 NCU ``metrics_delta``
become measured canonical deltas) and/or v2 ``knowledge.db`` files (their
stored vectors are used). Reports:

* classification coverage: classified / unknown / unexplained, depth, and the
  cross-tab against the v1 state label;
* do states separate technique effects: share of the reward variance within
  each technique explained by the state (eta^2), seed roots and leaves vs
  v1 states, with a permutation p-value (labels shuffled within technique);
* mechanism check vs outcome: on records with measured deltas, how often
  ``confirmed`` goes with an improvement and ``contradicted`` with none,
  against the base rate.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernelblaster.agents.opt_multislot import _op_cost  # noqa: E402
from kernelblaster.knowledge.outcomes import IMPROVED  # noqa: E402
from kernelblaster.knowledge.slots import SlotProblem, find_problems  # noqa: E402
from kernelblaster.knowledge.v2 import metrics as M  # noqa: E402
from kernelblaster.knowledge.v2.ontology import MECH_UNKNOWN, Ontology, mechanism_from_delta  # noqa: E402
from kernelblaster.knowledge.v2.states import StateHierarchy  # noqa: E402
from kernelblaster.knowledge.v2.targets import TargetRegistry, target_key  # noqa: E402


def roofline_input(p: SlotProblem, sid: str, us: float) -> M.RooflineInput:
    byt = fl = work = 0.0
    n = 0
    bufs = p.manifest["buffers"]
    for op in p.manifest["ops"]:
        if op["slot"] == sid:
            b, f = _op_cost(p, op)
            byt, fl, n = byt + b, fl + f, n + 1
            work = max(work, float(sum(bufs[x]["numel"] for x in op["outputs"])))
    return M.RooflineInput(bytes=byt, flops=fl, time_us=us, work_items=work or None, launches=max(n, 1))


def reward_of(r: dict):
    if r.get("v2", {}).get("reward") is not None:
        return float(r["v2"]["reward"])
    sp = r.get("speedup")
    if sp is None or sp <= 0:
        return None
    return max(-2.0, min(2.0, math.log2(sp)))


def load_rows(paths, problems, reg, hier, onto) -> list[dict]:
    rows = []
    for p in paths:
        p = Path(p)
        if p.suffix == ".db":
            c = sqlite3.connect(str(p))
            for rec, v2 in c.execute("SELECT rec, v2 FROM outcomes ORDER BY seq"):
                d = json.loads(rec)
                d["v2"] = json.loads(v2)
                d["vec"] = d["v2"].get("metrics_before") or {}
                d["mdelta"] = M.delta(d["v2"].get("metrics_before"), d["v2"].get("metrics_after"))
                d["src_file"] = str(p)
                rows.append(d)
            continue
        for line in open(p):
            d = json.loads(line)
            d["src_file"] = str(p)
            # names can repeat across variants (``..__fused``): take the one that has the slot
            prob = next((q for q in problems.get(d.get("problem_id"), []) if d.get("slot") in q.slots), None)
            d["vec"] = {}
            if prob is not None and d.get("slot") in prob.slots and d.get("metric_before"):
                t = reg.get(target_key(d["target"], d.get("arch")))
                d["vec"] = M.roofline_vector(t, roofline_input(prob, d["slot"], d["metric_before"]))
            d["mdelta"] = M.canonical_ncu_delta(d.get("metrics_delta") or {})
            rows.append(d)
    for d in rows:
        tk = target_key(d["target"], d.get("arch"))
        cls = hier.classify(d["vec"], target=tk, op_class=d.get("op_class"), dtype=d.get("dtype"))
        d["chain"], d["cls"] = cls.chain, cls.status
        d["item"] = (d.get("v2") or {}).get("item_id") or onto.resolve(d["technique"]) or d["technique"]
        d["reward"] = reward_of(d)
    return rows


def eta2(groups: dict) -> float:
    xs = [x for g in groups.values() for x in g]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    tot = sum((x - m) ** 2 for x in xs)
    if tot <= 0:
        return 0.0
    between = sum(len(g) * (sum(g) / len(g) - m) ** 2 for g in groups.values() if g)
    return between / tot


def separation(rows, label_fn, perms: int, rng: random.Random) -> dict:
    by_t = defaultdict(list)
    for r in rows:
        if r["reward"] is not None and label_fn(r) is not None:
            by_t[r["item"]].append((label_fn(r), r["reward"]))
    usable = {t: v for t, v in by_t.items() if len(v) >= 4 and len({l for l, _ in v}) >= 2}

    def score(data):
        num = den = 0.0
        for v in data.values():
            g = defaultdict(list)
            for l, x in v:
                g[l].append(x)
            num += eta2(g) * len(v)
            den += len(v)
        return num / den if den else 0.0

    obs = score(usable)
    ge = 0
    for _ in range(perms):
        sh = {}
        for t, v in usable.items():
            labels = [l for l, _ in v]
            rng.shuffle(labels)
            sh[t] = list(zip(labels, [x for _, x in v]))
        ge += score(sh) >= obs - 1e-12
    return {"techniques": len(usable), "outcomes": sum(len(v) for v in usable.values()),
            "eta2_weighted": round(obs, 4), "perm_p": round((ge + 1) / (perms + 1), 4) if usable else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outcomes", nargs="+", required=True)
    ap.add_argument("--problems", nargs="*", default=[])
    ap.add_argument("--out")
    ap.add_argument("--permutations", type=int, default=2000)
    a = ap.parse_args()
    reg = TargetRegistry()
    hier = StateHierarchy.seed(reg)
    onto = Ontology.seed(reg)
    problems = {}
    for d in a.problems:
        for pd in find_problems(Path(d)):
            try:
                p = SlotProblem(pd)
                problems.setdefault(p.name, []).append(p)
            except Exception:
                continue
    rows = load_rows(a.outcomes, problems, reg, hier, onto)
    rng = random.Random(0)
    rep: dict = {"records": len(rows), "with_vector": sum(1 for r in rows if r["vec"]),
                 "problems_found": sorted(problems)}
    # does the state explain technique effects beyond the op class? (groups = technique x op class)
    rows_oc = [dict(r, item=f"{r['item']}|{r.get('op_class')}") for r in rows]
    rep["classification"] = dict(Counter(r["cls"] for r in rows))
    rep["unknown_rate"] = round(sum(1 for r in rows if r["cls"] != "classified") / max(len(rows), 1), 4)
    rep["depth"] = dict(Counter(len(r["chain"]) for r in rows))
    rep["leaf_states"] = dict(Counter(r["chain"][-1] if r["chain"] else r["cls"] for r in rows))
    xt = defaultdict(Counter)
    for r in rows:
        xt[r.get("state_before") or "-"][r["chain"][0] if r["chain"] else r["cls"]] += 1
    rep["v1_state_vs_v2_root"] = {k: dict(v) for k, v in xt.items()}
    # for v2 databases state_before IS the v2 leaf; v1_state is meaningful for v1 logs only
    lab = {"v1_state": lambda r: r.get("state_before") if not r.get("v2") else None,
           "v2_root": lambda r: r["chain"][0] if r["chain"] else None,
           "v2_leaf": lambda r: r["chain"][-1] if r["chain"] else None,
           "op_class_only": lambda r: r.get("op_class")}
    rep["state_separates_technique_effects"] = {k: separation(rows, f, a.permutations, rng) for k, f in lab.items()}
    rep["state_separates_within_technique_x_op_class"] = {
        k: separation(rows_oc, f, a.permutations, rng) for k, f in lab.items() if k != "op_class_only"}
    mech = []
    for r in rows:
        if not r["mdelta"]:
            continue
        t = onto.get(r["item"]) or onto.get(r["technique"])
        if t is None or r.get("speedup") is None:
            continue
        m = mechanism_from_delta(t.predicts, r["mdelta"])
        mech.append((m["label"], r["status"] == IMPROVED, r["id"], r["item"]))
    lab_n = Counter(l for l, _, _, _ in mech)
    dec = [(l, w) for l, w, _, _ in mech if l != MECH_UNKNOWN]
    agree = sum(1 for l, w in dec if (l == "confirmed") == w)
    base_w = sum(1 for _, w in dec if w) / len(dec) if dec else None
    chance = (base_w * sum(1 for l, _ in dec if l == "confirmed") / len(dec)
              + (1 - base_w) * sum(1 for l, _ in dec if l == "contradicted") / len(dec)) if dec else None
    rep["mechanism_check"] = {"records_with_measured_delta": len(mech), "labels": dict(lab_n),
                              "decided": len(dec), "agreement": round(agree / len(dec), 4) if dec else None,
                              "chance_agreement": round(chance, 4) if chance is not None else None,
                              "p_improved_given_confirmed": _rate([w for l, w in dec if l == "confirmed"]),
                              "p_improved_given_contradicted": _rate([w for l, w in dec if l == "contradicted"]),
                              "fisher_p_one_sided": round(fisher_one_sided(
                                  sum(1 for l, w in dec if l == "confirmed" and w),
                                  sum(1 for l, w in dec if l == "confirmed" and not w),
                                  sum(1 for l, w in dec if l == "contradicted" and w),
                                  sum(1 for l, w in dec if l == "contradicted" and not w)), 4) if dec else None,
                              "rows": [{"id": i, "item": it, "label": l, "improved": w} for l, w, i, it in mech][:60]}
    js = json.dumps(rep, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(js)
    print(js)


def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """P(X >= a) for the 2x2 table [[a, b], [c, d]] with fixed margins (hypergeometric)."""
    n1, n2, k = a + b, c + d, a + c
    tot = math.comb(n1 + n2, k)
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / tot if tot else 1.0


def _rate(xs):
    return {"n": len(xs), "rate": round(sum(xs) / len(xs), 3) if xs else None}


if __name__ == "__main__":
    main()
