"""Show what a multi-slot run did, step by step, from its trace store (docs/trace.md).

    python scripts/trace_view.py RUN_DIR --list                       # one row per step
    python scripts/trace_view.py RUN_DIR --problem P --rollout 0 --step 2 [--diff] [--full]
    python scripts/trace_view.py RUN_DIR --failed                     # every step that failed or needed a fix
    python scripts/trace_view.py RUN_DIR --llm REQUEST_ID             # one LLM call, whole prompt + response
    python scripts/trace_view.py RUN_DIR --export DIR                 # plain files per step
    python scripts/trace_view.py RUN_DIR --html FILE                  # one self-contained HTML file
    python scripts/trace_view.py MB_WORK --mb                         # ModelBlaster's records (rounds, hints)

RUN_DIR: a run_multislot --out dir (its trace/), a trace store, or a ModelBlaster
graph-optimiser work dir (every store below it: MB's own and each KernelBlaster run's).
A step's story: the slot's source before the step, each attempt's prompt (abridged
unless --full), raw response, extracted candidate (--diff: against the previous
source), build commands with compiler output on failure, driver MB_* lines, the
evaluation result, and the decision with its result.json / outcome / telemetry links.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernelblaster.trace.view import (Trace, export, export_html, find_stores, list_table, render_llm,  # noqa: E402
                                      render_mb, render_step, FAIL)


def _select(steps, a):
    out = []
    for s in steps:
        if a.problem and a.problem not in s.problem:
            continue
        if a.rollout is not None and s.rollout != a.rollout:
            continue
        if a.step is not None and s.step != a.step:
            continue
        if a.failed and not (s.error or (s.record or {}).get("status") in FAIL or (s.record or {}).get("fix_attempts")):
            continue
        out.append(s)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--list", action="store_true", help="table of steps")
    ap.add_argument("--problem", help="problem name (substring)")
    ap.add_argument("--rollout", type=int)
    ap.add_argument("--step", type=int)
    ap.add_argument("--failed", action="store_true", help="only steps that failed, raised or needed fixes")
    ap.add_argument("--llm", metavar="REQUEST_ID", help="one LLM call (request id prefix)")
    ap.add_argument("--diff", action="store_true", help="candidates as diffs against the previous source")
    ap.add_argument("--full", action="store_true", help="whole prompts, responses and outputs")
    ap.add_argument("--export", metavar="DIR", help="write plain files per step")
    ap.add_argument("--html", metavar="FILE", help="write one self-contained HTML file")
    ap.add_argument("--mb", action="store_true", help="ModelBlaster records (graph optimiser rounds, hints)")
    a = ap.parse_args(argv)
    stores = find_stores(a.run_dir)
    if not stores:
        print(f"no trace store at or below {a.run_dir}", file=sys.stderr)
        return 1
    shown = 0
    for root in stores:
        tr = Trace(root)
        head = f"## {root} ({len(tr.recs)} records)" if len(stores) > 1 else None
        if a.mb:
            if tr.mb_records():
                print(head or f"## {root}")
                print(render_mb(tr, a.full))
                shown += 1
            continue
        if a.llm:
            for r in tr.llm_by_request(a.llm):
                print(render_llm(tr, r, full=True))
                shown += 1
            continue
        steps = _select(tr.steps(), a)
        if a.export:
            dest = Path(a.export) / (root.parent.name if len(stores) > 1 else "")
            print(f"exported {len(steps)} step(s) to {export(tr, dest, steps)}")
            shown += 1
            continue
        if a.html:
            dest = Path(a.html)
            if len(stores) > 1:
                dest = dest.with_name(f"{dest.stem}_{root.parent.name}{dest.suffix}")
            print(f"wrote {export_html(tr, dest, steps, title=f'trace {root.parent.name}')}")
            shown += 1
            continue
        if not steps:
            continue
        if head:
            print(head)
        if a.list or (a.step is None and not a.failed):
            print(list_table(steps))
        else:
            for s in steps:
                print(render_step(tr, s, full=a.full, diff=a.diff))
                print()
        shown += 1
    if not shown:
        print("nothing matched", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
