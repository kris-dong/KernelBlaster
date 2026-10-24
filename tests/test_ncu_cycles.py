"""NCU "Elapsed Cycles" totals must cover every kernel launch."""
import pytest

from kernelblaster.agents.utils import get_elapsed_cycles_ncu_log
from kernelblaster.backends import get_backend


def _section(name, grid, cycles, dur):
    return f"""\
  {name} ({grid}, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         6.79
    SM Frequency            cycle/usecond         1.35
    Elapsed Cycles                  cycle        {cycles}
    Memory Throughput                   %        85.21
    DRAM Throughput                     %        85.21
    Duration                      usecond        {dur}
    L1/TEX Cache Throughput             %        21.66
    L2 Cache Throughput                 %        30.91
    SM Active Cycles                cycle     2,876.35
    Compute (SM) Throughput             %        16.75
    ----------------------- ------------- ------------

    OPT   This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak
          performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak
          typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential
          reasons.

"""


# One op = 3 launches of one kernel + 1 launch of another (the
# `cat3`-style case: first-match reported 3,115 instead of the total).
MULTI_LAUNCH_LOG = (
    "==PROF== Connected to process 1488701 (/tmp/multi)\n"
    '==PROF== Profiling "scale_kernel" - 0: 0%....50%....100% - 8 passes\n'
    '==PROF== Profiling "scale_kernel" - 1: 0%....50%....100% - 8 passes\n'
    '==PROF== Profiling "scale_kernel" - 2: 0%....50%....100% - 8 passes\n'
    '==PROF== Profiling "add_kernel" - 3: 0%....50%....100% - 8 passes\n'
    "done\n"
    "==PROF== Disconnected from process 1488701\n"
    "[1488701] multi@127.0.0.1\n"
    + _section("scale_kernel(float *, int)", 4096, "3,115", "2.30")
    + _section("scale_kernel(float *, int)", 4096, "3,281", "2.43")
    + _section("scale_kernel(float *, int)", 4096, "3,285", "2.43")
    + _section("add_kernel(float *, int)", 4096, "998", "0.74")
)


def test_sums_every_launch():
    assert get_elapsed_cycles_ncu_log(MULTI_LAUNCH_LOG) == 3115 + 3281 + 3285 + 998


def test_single_launch_unchanged():
    log = _section("k(float *)", 1, "12675", "9.39")
    assert get_elapsed_cycles_ncu_log(log) == 12675


def test_cuda_backend_parse_profile_sums():
    pr = get_backend("cuda").parse_profile(MULTI_LAUNCH_LOG)
    assert pr.raw_metrics["elapsed_cycles"] == 10679


@pytest.mark.parametrize("log,expected", [
    ("Elapsed Cycles                cycle        98765", 98765),
    ("Elapsed Cycles,cycle,12675\nElapsed Cycles,cycle,25", 12700),
    ("// Elapsed Cycles: 12675\n", 12675),
])
def test_other_formats(log, expected):
    assert get_elapsed_cycles_ncu_log(log) == expected


def test_missing_raises():
    with pytest.raises(RuntimeError):
        get_elapsed_cycles_ncu_log("no cycles in this log")
