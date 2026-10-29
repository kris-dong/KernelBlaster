"""claude-sonnet-5-5 (the QRB5165 A6 run's model for every role): its price row is an ESTIMATE ($3 / $15 per M tokens;
cache write 3.75, read 0.30, the Sonnet-class rates), explicit in both tables so the relay's est_usd and --budget-usd count it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernelblaster.agents.cost_tracker import lookup_cache_pricing, lookup_pricing  # noqa: E402
from kernelblaster.telemetry import llm  # noqa: E402


def test_sonnet55_priced():
    for m in ("claude-sonnet-5-5", "azure/anthropic/claude-sonnet-5-5", "anthropic.claude-sonnet-5-5-v1:0"):
        assert lookup_pricing(m) == (3.0, 15.0)
        assert lookup_cache_pricing(m) == (3.75, 6.0, 0.3)
        assert llm._prices(m) == (3.0, 15.0, 0.3, 3.75)


def test_sonnet55_cost_of_a_call():
    rec = llm.normalize_usage({"prompt_tokens": 1_000_000, "completion_tokens": 100_000}, "azure/anthropic/claude-sonnet-5-5")
    assert abs(rec["est_usd"] - (3.0 + 1.5)) < 1e-9


def test_other_rows_unchanged():
    assert lookup_pricing("azure/anthropic/claude-opus-5-5") == (5.0, 25.0)
    assert lookup_pricing("azure/anthropic/claude-sonnet-4-6") == (3.0, 15.0)
