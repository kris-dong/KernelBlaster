"""Fold ModelBlaster graph-optimiser decisions into a knowledge v2 database.

    python scripts/sync_graph_hints.py KNOWLEDGE_DIR GRAPH_HINTS.json [...] [--target cuda:sm_75]

GRAPH_HINTS.json is ModelBlaster's ``modelblaster.graph_hints/v1`` file
(``pipeline/graph_hints.py``; the graph optimiser keeps it in its work dir).
Accepted hints become promoted graph hypotheses, rejected ones retired, keyed
by the hint id (``fuse:<kind>``, ``split:<slot>``) -- the ids KernelBlaster's
kb-proposals use. Idempotent. ``run_multislot.py --graph-hints FILE`` does the
same at the start of every run, so a graph optimiser that passes its hints
file to each KernelBlaster call keeps the knowledge base in sync without a
separate step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernelblaster.knowledge.v2 import KnowledgeV2  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("knowledge_dir")
    ap.add_argument("hints", nargs="+")
    ap.add_argument("--target", default="cuda:sm_75")
    a = ap.parse_args()
    kv = KnowledgeV2(Path(a.knowledge_dir))
    n = sum(kv.sync_graph_hints(h, target_key=a.target) for h in a.hints)
    graph = [h for h in kv.hypotheses.items.values() if h.kind == "graph"]
    print(json.dumps({"synced": n, "graph_hypotheses": len(graph),
                      "by_status": {s: sum(1 for h in graph if h.status == s) for s in {h.status for h in graph}}}))


if __name__ == "__main__":
    main()
