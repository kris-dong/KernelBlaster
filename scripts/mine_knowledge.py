"""Mine recorded runs into the v2 seed (``data/kernelblaster/knowledge/mined.json``).

    python scripts/mine_knowledge.py --db RUN/knowledge/knowledge.db [...] \
        --outcomes OLD/outcomes.jsonl [...] --impl OLD/impl [...] --out data/kernelblaster/knowledge/mined.json

* ``outcomes``: every technique outcome of the v2 databases (deduplicated by id;
  hypothesis items skipped), with its target, op class / kind, state chain and
  reward. The seed loader folds them into the hierarchical statistics, so they
  are evidence, not priors.
* ``claims``: applicability claims aggregated per (technique, op class, target)
  from v2 rows, v1 ``outcomes.jsonl`` logs and implementation-store records
  (``impl/index.jsonl``: a record >= 1.05x over its parent counts for the last
  technique of its lineage, the step that produced it): ``helps`` when >= 2/3 of >= 3 attempts
  improved, ``hurts`` when >= 2/3 regressed or failed. A polarity seen on >= 2
  targets becomes one claim at their lowest common ancestor. Each claim cites
  the record / impl ids it was mined from.

Technique names are resolved through the ontology's aliases; names it does not
know are kept only if they are v2 hypothesis-promoted techniques (otherwise
dropped and counted).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernelblaster.knowledge.v2.ontology import Ontology  # noqa: E402
from kernelblaster.knowledge.v2.targets import TargetRegistry, target_key  # noqa: E402

FAIL = ("compile_error", "wrong_output", "runtime_error")
MIN_N, SHARE = 3, 2 / 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", nargs="*", default=[])
    ap.add_argument("--outcomes", nargs="*", default=[])
    ap.add_argument("--impl", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reg = TargetRegistry()
    onto = Ontology.seed(reg)
    dropped = Counter()
    samples, seen = [], set()
    events = defaultdict(list)                  # (item, op_class, target) -> [(id, status, speedup)]

    def resolve(name: str):
        tid = onto.resolve(name)
        if tid is None:
            dropped[name] += 1
        return tid

    for db in a.db:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for rec, v2 in con.execute("SELECT rec, v2 FROM outcomes ORDER BY seq"):
            r, x = json.loads(rec), json.loads(v2 or "{}")
            if r["id"] in seen or x.get("kind") == "hypothesis":
                continue
            seen.add(r["id"])
            item = resolve(x.get("item_id") or r["technique"])
            if item is None:
                continue
            tk = reg.get(target_key(r["target"], r["arch"])).key
            if x.get("reward") is not None:
                samples.append({"id": r["id"], "item": item, "target": tk, "op_class": r["op_class"],
                                "op_kind": r.get("op_kind"), "dtype": r.get("dtype") or "fp32",
                                "chain": x.get("chain") or [], "reward": round(float(x["reward"]), 5),
                                "status": r["status"]})
            events[(item, r["op_class"], tk)].append((r["id"], r["status"], r.get("speedup")))
    for f in a.outcomes:
        for line in open(f):
            r = json.loads(line)
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            item = resolve(r["technique"])
            if item is not None:
                events[(item, r["op_class"], reg.get(target_key(r["target"], r["arch"])).key)].append(
                    (r["id"], r["status"], r.get("speedup")))
    winners = 0
    for d in a.impl:
        idx = Path(d) / "index.jsonl"
        for line in (idx.read_text().splitlines() if idx.exists() else []):
            try:
                m = json.loads(line)
            except ValueError:
                continue
            # a stored record's own step: its last technique, speedup over the parent it was made from
            if m.get("id") in seen or not m.get("techniques") or (m.get("speedup") or 0) < 1.05:
                continue
            seen.add(m["id"])
            winners += 1
            tk = reg.get(target_key(m.get("target", "cuda"), m.get("arch"))).key
            item = resolve(m["techniques"][-1])
            if item is not None:
                events[(item, m.get("op_class", "other"), tk)].append((f"impl:{m['id']}", "improved",
                                                                      round(m["speedup"], 3)))

    # per-target polarity, then merge the same polarity across targets at their common ancestor
    per = defaultdict(dict)                      # (item, op_class, effect) -> {target: [ids]}
    speed = defaultdict(list)
    tot = Counter()
    for (item, opc, tk), evs in events.items():
        n = len(evs)
        good = [e for e in evs if e[1] == "improved"]
        bad = [e for e in evs if e[1] == "regressed" or e[1] in FAIL]
        for effect, grp in (("helps", good), ("hurts", bad)):
            if n >= MIN_N and len(grp) >= SHARE * n:
                per[(item, opc, effect)][tk] = [e[0] for e in grp]
                speed[(item, opc, effect)] += [e[2] for e in grp if e[2]]
                tot[(item, opc, effect)] += n
    claims = []
    for (item, opc, effect), by_t in sorted(per.items()):
        scope = reg.lca(by_t)
        ids = [i for t in sorted(by_t) for i in by_t[t]]
        sp = speed[(item, opc, effect)]
        med = f", median {statistics.median(sp):.2f}x" if sp else ""
        where = ", ".join(t.split(":", 1)[1] for t in sorted(by_t))
        verb = "improved" if effect == "helps" else "regressed or failed on"
        claims.append({"kind": "applicability", "technique": item, "op_class": opc, "effect": effect,
                       "scope": scope, "targets": sorted(by_t), "evidence": ids[:64], "n": len(ids),
                       "text": f"{item} {verb} {opc} slots in {len(ids)} of {tot[(item, opc, effect)]} recorded "
                               f"attempts on {where}{med}."})
    out = {"_doc": "Mined from recorded runs by scripts/mine_knowledge.py (do not hand-edit). outcomes: technique "
                   "outcomes (evidence for the hierarchical statistics); claims: aggregated applicability claims "
                   "citing the mined record / impl ids, scoped to the lowest common ancestor of their targets.",
           "sources": {"dbs": len(a.db), "v1_logs": len(a.outcomes), "impl_stores": len(a.impl), "winners": winners,
                       "dropped_unknown_techniques": dict(dropped.most_common(20))},
           "outcomes": samples, "claims": claims}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=False) + "\n")
    print(json.dumps({"outcomes": len(samples), "claims": len(claims), "winners": winners,
                      "dropped": sum(dropped.values())}))


if __name__ == "__main__":
    main()
