# Make ``import kernelblaster`` work from a plain checkout (no install).
import sys
from pathlib import Path

# ``data.sources`` is a top-level package next to ``src``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
