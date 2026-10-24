"""Per-job slicing of fused spike / firesim batch stdout."""
import pytest

from kernelblaster.servers.strategies.firesim_exec import FireSimExecStrategy
from kernelblaster.servers.strategies.spike_exec import SpikeExecStrategy
from kernelblaster.servers.utils.exec_strategy import ExecJob


def _parse(cls, jobs, model_names, stdout):
    # _parse_batch_output only touches its arguments; skip __init__
    # (which wants a modelblaster root / spike binary).
    strat = object.__new__(cls)
    return strat._parse_batch_output(jobs, model_names, stdout, "")


def _jobs(*stems):
    return [ExecJob(binary_data=b"", filename=f"/x/{s}.elf") for s in stems]


# Exactly what harness_shared_input/gen_shared_input_main.py prints.
SHARED_INPUT_LOG = """\
modelblaster harness: model=kb_19_ReLU shared_input_variants=3 in=4096 out=4096
=== MODELBLASTER_VERIFY [kb_19_ReLU@tmpaaa] === max_abs_err=0 max_rel_err=0 n=4096 PASS
=== MODELBLASTER_WALL_CYCLES [kb_19_ReLU@tmpaaa] === 111
=== MODELBLASTER_VERIFY [kb_19_ReLU@tmpbbb] === max_abs_err=3.5 max_rel_err=1 n=4096 FAIL
=== MODELBLASTER_WALL_CYCLES [kb_19_ReLU@tmpbbb] === 222
=== MODELBLASTER_VERIFY [kb_19_ReLU@tmpccc] === max_abs_err=0 max_rel_err=0 n=4096 PASS
=== MODELBLASTER_WALL_CYCLES [kb_19_ReLU@tmpccc] === 333
modelblaster harness: shared_input all variants done
"""


@pytest.mark.parametrize("cls", [SpikeExecStrategy, FireSimExecStrategy])
def test_shared_input_variants_map_to_their_own_job(cls):
    jobs = _jobs("tmpaaa", "tmpbbb", "tmpccc")
    names = [f"kb_19_ReLU@{t}" for t in ("tmpaaa", "tmpbbb", "tmpccc")]
    res = _parse(cls, jobs, names, SHARED_INPUT_LOG)
    assert [r.success for r in res] == [True, True, True]
    for r, tag, cyc, verdict in zip(
        res, ("tmpaaa", "tmpbbb", "tmpccc"), (111, 222, 333),
        ("PASS", "FAIL", "PASS"),
    ):
        assert r.stdout.endswith(f"[kb_19_ReLU@{tag}] === {cyc}")
        assert f"[kb_19_ReLU@{tag}] === max_abs_err" in r.stdout
        assert r.stdout.splitlines()[-2].endswith(verdict)
        # No sibling variant's markers leak into this job's block.
        others = {"tmpaaa", "tmpbbb", "tmpccc"} - {tag}
        assert not any(o in r.stdout for o in others)


@pytest.mark.parametrize("cls", [SpikeExecStrategy, FireSimExecStrategy])
def test_missing_variant_fails_instead_of_borrowing_sibling(cls):
    log = "\n".join(
        l for l in SHARED_INPUT_LOG.splitlines() if "tmpbbb" not in l
    )
    jobs = _jobs("tmpaaa", "tmpbbb", "tmpccc")
    names = [f"kb_19_ReLU@{t}" for t in ("tmpaaa", "tmpbbb", "tmpccc")]
    res = _parse(cls, jobs, names, log)
    assert [r.success for r in res] == [True, False, True]
    assert "kb_19_ReLU@tmpbbb" in res[1].message


@pytest.mark.parametrize("cls", [SpikeExecStrategy, FireSimExecStrategy])
def test_cross_problem_bare_names_still_match_quant_tags(cls):
    log = (
        "=== MODELBLASTER_VERIFY [kb_19_ReLU@fp32] === max_abs_err=0 max_rel_err=0 n=8\n"
        "=== MODELBLASTER_WALL_CYCLES [kb_19_ReLU@fp32] === 10\n"
        "=== MODELBLASTER_VERIFY [kb_20_LeakyReLU@fp32] === max_abs_err=0 max_rel_err=0 n=8\n"
        "=== MODELBLASTER_WALL_CYCLES [kb_20_LeakyReLU@fp32] === 20\n"
    )
    jobs = _jobs("kb_19_ReLU", "kb_20_LeakyReLU")
    res = _parse(cls, jobs, ["kb_19_ReLU", "kb_20_LeakyReLU"], log)
    assert [r.success for r in res] == [True, True]
    assert res[0].stdout.endswith("=== 10")
    assert res[1].stdout.endswith("=== 20")
    assert "LeakyReLU" not in res[0].stdout


@pytest.mark.parametrize("cls", [SpikeExecStrategy, FireSimExecStrategy])
def test_untagged_single_model_marker(cls):
    log = "boot\n=== MODELBLASTER_WALL_CYCLES === 42\n"
    res = _parse(cls, _jobs("m"), [""], log)
    assert res[0].success and res[0].stdout.endswith("=== 42")
