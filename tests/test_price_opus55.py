"""claude-opus-5-5 has a price row (it had none: the LLM relay's est_usd was $0 for every Opus call, so --budget-usd never counted Opus spend)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernelblaster.agents.cost_tracker import lookup_cache_pricing, lookup_pricing  # noqa: E402
from kernelblaster.telemetry import llm  # noqa: E402


def test_opus55_priced():
    for m in ("claude-opus-5-5", "azure/anthropic/claude-opus-5-5"):
        assert lookup_pricing(m) == (5.0, 25.0)
        assert lookup_cache_pricing(m) == (6.25, 10.0, 0.5)
        assert llm._prices(m) == (5.0, 25.0, 0.5, 6.25)


def test_haiku55_unchanged():
    assert lookup_pricing("azure/anthropic/claude-haiku-5-5") == (1.0, 5.0)
