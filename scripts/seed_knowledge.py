"""Build (or refresh) a seeded knowledge v2 database from the data files.

    python scripts/seed_knowledge.py OUT_DIR [--fresh] [--no-mined] [--seed-dir DIR] [--report]

OUT_DIR/knowledge.db gets the scope tree (targets.json), the state seeds
(states.json), the curated techniques (techniques/*.json), the techniques of
the v1 technique DB and footers not folded into them, and the mined evidence
(mined.json: statistics + claims). Seeding is idempotent: on an existing DB
only seed-sourced rows are refreshed (discovered states, promoted techniques,
runtime scopes and recorded outcomes are kept). ``--fresh`` deletes OUT_DIR's
knowledge.db first. ``--seed-dir`` builds from another seed directory (e.g. an
older checkout's data/kernelblaster/knowledge, for A/B runs). The seed is data
files plus this loader; databases are never hand-edited.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--no-mined", action="store_true")
    ap.add_argument("--seed-dir")
    ap.add_argument("--report", action="store_true", help="print the technique coverage (op class x scope node)")
    a = ap.parse_args()
    if a.seed_dir:
        os.environ["KB_KNOWLEDGE_SEED_DIR"] = str(Path(a.seed_dir).resolve())
    from kernelblaster.knowledge.v2 import KnowledgeV2      # after the seed-dir override
    out = Path(a.out)
    if a.fresh:
        for f in ("knowledge.db", "knowledge.db-wal", "knowledge.db-shm"):
            (out / f).unlink(missing_ok=True)
    kv = KnowledgeV2(out, seed_mined=not a.no_mined)
    techs = [t for t in kv.ontology.techniques.values()]
    rep = {"db": str(out / "knowledge.db"), "scopes": len(kv.registry.nodes), "targets": len(kv.registry.targets),
           "states": len(kv.hierarchy.states), "techniques": len(techs),
           "slot_techniques": sum(1 for t in techs if t.slot), "claims": len(kv.claims.active()),
           "stats_rows": len(kv.stats.data), "seed_version": kv.db.get_meta("seed_version"),
           "mined_seed": kv.db.get_meta("mined_seed")}
    if a.report:
        cov = coverage(kv)
        rep["coverage"] = cov
    print(json.dumps(rep, indent=1))


def coverage(kv) -> dict:
    """Curated slot techniques per op class x scope node (universal techniques count under universal)."""
    from kernelblaster.knowledge.outcomes import OP_CLASSES
    cov: dict = {}
    for t in kv.ontology.techniques.values():
        if not t.slot or t.source != "seed":
            continue
        node = kv.registry.scope_id(t.level, t.scope)
        for oc in (t.op_classes or ["(any)"]):
            cov.setdefault(oc, Counter())[node] += 1
    order = list(OP_CLASSES) + ["(any)"]
    return {oc: dict(cov[oc]) for oc in order if oc in cov}


if __name__ == "__main__":
    main()
