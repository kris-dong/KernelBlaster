"""Summarise one or more telemetry runs (docs/telemetry.md).

    python scripts/telemetry_report.py OUT [OUT2 ...] [--by-problem] [--top 15] [--json]

Each argument is a ``telemetry.jsonl`` file or a directory searched for them
(a ``run_multislot.py --out`` dir, a ModelBlaster ``kb_graph_opt`` work dir).
Tables: where the wall clock goes (by category and by span), evaluate()
calls by purpose, LLM tokens by role / model, truncated outputs, the slowest
spans; several runs also get a combined table.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernelblaster.telemetry.report import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
