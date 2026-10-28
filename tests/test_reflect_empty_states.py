"""A truly empty seed (KB_EMPTY_SEED=1) has no states: SemanticUpdater must keep an EMPTY known_states set a set.

It used to turn it into None (`set(x) if x else None`); reflect_v2 then computed `self.known_states | set(...)`, which raised
`unsupported operand type(s) for |` and rejected every reflection edit as malformed (empty-seed arm lost all its claims)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernelblaster.knowledge.reflect import SemanticUpdater  # noqa: E402


class _Stub:
    pass


def test_empty_known_states_stays_a_set():
    u = SemanticUpdater(_Stub(), _Stub(), model="stub", known_states=set())
    assert u.known_states == set()
    assert (u.known_states | {"a"}) == {"a"}          # the union reflect_v2 computes


def test_missing_known_states_is_unconstrained():
    assert SemanticUpdater(_Stub(), _Stub(), model="stub").known_states is None
    assert SemanticUpdater(_Stub(), _Stub(), model="stub", known_states={"s"}).known_states == {"s"}
