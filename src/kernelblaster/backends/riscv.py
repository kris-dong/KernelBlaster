# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""RISC-V + Zephyr + FPGA backend (skeleton).

Facade over the RISC-V SoC + Zephyr RTOS + FPGA-emulation flow. The
key architectural distinction from CUDA/OpenCL:

* **Compile**: cross-compiles each kernel into a Zephyr ELF (or static
  library) via :class:`servers.strategies.ZephyrCompileStrategy`.
* **Exec**: rides on top of :class:`servers.strategies.FPGAExecStrategy`,
  whose ``batch_exec`` primary path links N kernel ELFs into one boot
  ELF, flashes the FPGA bitstream **once** (minutes) if not cached, and
  dispatches each embedded kernel with per-kernel cycle counts read out
  over UART.

The Zephyr batch runner emits ``[PROFILE] <kernel_id>: <cycles>`` lines
into UART — same tag shape as the OpenCL ``[PROFILE]`` convention so
this backend's :meth:`parse_profile` is structurally identical, only
switching the unit (cycles instead of ms). ``mcycle`` from the RISC-V
control-and-status registers (CSRs) is the underlying source; the
batch runner samples it around each kernel's entry/exit.

Sections marked with ``# TODO(riscv-fpga)`` are the domain-specific
bits that require actual hardware / build-flow integration and are
NOT-yet-wired. The rest is production-shape (parse, prompt, metric
formatting) — enough to bring up a real RL loop end-to-end against
a mock target for shape validation before the FPGA path lights up.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from .base import Backend, ProfileResult, RLNodeConfig

if TYPE_CHECKING:
    from ..config import GPUType


# Fallback catalog: bottleneck → [(technique_id, predicted_improvement_%), ...].
# Placeholder set biased toward RISC-V-in-order-core / small-cache concerns.
# TODO(riscv-fpga): tune once we've observed real workloads on the target SoC.
_RISCV_DEFAULT_OPTIMIZATIONS: Mapping[str, list[tuple[str, float]]] = {
    "memory_bound": [
        ("1.1_loop_tiling_icache", 25.0),
        ("1.2_data_layout_soa", 20.0),
        ("1.3_reduce_indirect_loads", 15.0),
    ],
    "compute_bound": [
        ("2.1_loop_unrolling", 30.0),
        ("2.2_strength_reduction", 20.0),
        ("2.3_rvv_vectorization", 40.0),
    ],
    "latency_bound": [
        ("3.1_branch_prediction_hints", 25.0),
        ("3.2_load_use_scheduling", 20.0),
        ("3.3_prefetch_hints", 20.0),
    ],
    "hybrid_bound": [
        ("4.1_register_tiling", 30.0),
        ("4.2_software_pipelining", 30.0),
    ],
}


_RISCV_TECHNIQUE_MAP: Mapping[str, str] = {
    "1.1_loop_tiling_icache": (
        "Tile inner loops so the working set fits in the RISC-V core's small "
        "instruction cache. Prefer block sizes that keep hot loop bodies "
        "under the L1I capacity."
    ),
    "1.2_data_layout_soa": (
        "Reorganise struct-of-arrays vs array-of-structs to keep stride-1 "
        "access on hot paths. Avoids gather-style access patterns that "
        "trash the small D-cache."
    ),
    "1.3_reduce_indirect_loads": (
        "Hoist address computations out of the hot loop; replace pointer "
        "chases with pre-computed base+offset when the layout is known."
    ),
    "2.1_loop_unrolling": (
        "Unroll inner loops manually (compiler often under-unrolls for "
        "code-size budgets on embedded targets). Aim for enough independent "
        "chains to hide load-use latency on the in-order core."
    ),
    "2.2_strength_reduction": (
        "Replace multiplies/divisions in address computation with adds and "
        "shifts. In-order RISC-V cores make branch/mul latency painful."
    ),
    "2.3_rvv_vectorization": (
        "If the target SoC exposes the RISC-V V extension, use vsetvl-driven "
        "vector intrinsics for the inner loop. Falls back to scalar cleanly "
        "when V is absent."
    ),
    "3.1_branch_prediction_hints": (
        "Restructure conditionals so the predicted-taken path is the common "
        "one; consider __builtin_expect / predicate-style masking."
    ),
    "3.2_load_use_scheduling": (
        "Interleave independent loads with dependent computation so the "
        "in-order pipeline doesn't stall on load-use hazards."
    ),
    "3.3_prefetch_hints": (
        "Emit software prefetches (target-specific intrinsic or the RVA22 "
        "hint instruction) two iterations ahead of the consumer."
    ),
    "4.1_register_tiling": (
        "Increase work per innermost iteration so intermediate results stay "
        "in registers rather than spilling to the small stack."
    ),
    "4.2_software_pipelining": (
        "Overlap iteration k's compute with iteration k+1's load; requires "
        "manual epilogue/prologue but big win on in-order pipelines."
    ),
}


# modelblaster wire format (kept in sync with
# modelblaster/validation/runner_common.py):
#
#   === MODELBLASTER_WALL_CYCLES === 12345678
#   === MODELBLASTER_WALL_CYCLES [<model>@<quant>] === 12345678
#   === MODELBLASTER_PROFILE_BEGIN [<model>@<quant>] ===
#   dispatch_id,name,op,shape,cycles
#   0,relu,relu,n=49152,639138
#   === MODELBLASTER_PROFILE_END [<model>@<quant>] ===
#
# WALL_CYCLES is the mtime-tick wall time (mtime CSR). PROFILE lines
# give per-op ``mcycle`` deltas — the actually useful RL signal since
# they isolate individual kernel costs. :meth:`parse_profile` reports
# the sum of per-op cycles as ``total_cycles`` when PROFILE is present;
# falls back to WALL_CYCLES otherwise.
_WALL_CYCLES_RE = re.compile(
    r"=== MODELBLASTER_WALL_CYCLES(?: \[[^\]]+\])? === (\d+)",
)
_PROFILE_BLOCK_RE = re.compile(
    r"=== MODELBLASTER_PROFILE_BEGIN(?:\s*\[[^\]]+\])?\s*===\s*"
    r"(?P<body>.*?)"
    r"=== MODELBLASTER_PROFILE_END(?:\s*\[[^\]]+\])?\s*===",
    re.DOTALL,
)
# Individual dispatch row inside the CSV block. Column order (per
# modelblaster/pipeline/generate_skeleton.py): dispatch_id, name, op,
# shape, cycles.
_PROFILE_ROW_RE = re.compile(
    r"^(?P<idx>\d+),(?P<name>[^,]+),(?P<op>[^,]+),(?P<shape>[^,]*),(?P<cycles>\d+)$",
    re.MULTILINE,
)


def default_batch_runner_template() -> Path:
    """Path to the Zephyr batch-runner app template.

    The batch runner is a Zephyr app that boots on the RISC-V SoC,
    reads a queue of (kernel_id, args) records from UART, dispatches
    each embedded kernel, samples ``mcycle`` before/after, and emits
    ``[PROFILE] <kernel_id>: <cycles>`` on UART. See
    :class:`servers.strategies.FPGAExecStrategy` for the wiring.

    Overridable via ``KERNELBLASTER_ZEPHYR_BATCH_RUNNER`` env var. The
    default points into the repo's ``docker_qualcomm``-style vendor
    directory (``docker_riscv/zephyr_batch_runner``) — real path lands
    when the template is committed.
    """
    override = os.getenv("KERNELBLASTER_ZEPHYR_BATCH_RUNNER")
    if override:
        return Path(override)
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "docker_riscv" / "zephyr_batch_runner"


class RiscvZephyrBackend(Backend):
    """RISC-V + Zephyr RTOS + FPGA-emulated SoC backend.

    A single :class:`RiscvZephyrBackend` instance corresponds to one
    logical SoC/bitstream target — different bitstreams (different
    core configs, different peripherals) get distinct
    :class:`GPUType` enum entries mapped through
    :func:`backend_for_gpu`. Today only ``RISCV_FPGA_ZEPHYR`` is
    registered; add more entries when concrete SoCs need distinct
    identities.
    """

    name = "riscv"
    kernel_ext = ".c"
    driver_filename = "main.c"     # Zephyr apps use main.c as entry point

    # Heterogeneous-model tier dispatch (P5.5). Names match the technique
    # IDs in _RISCV_TECHNIQUE_MAP. SIMPLE = mostly-mechanical (unroll a loop,
    # add a prefetch hint, swap a branch for predication); HARD = structural
    # (vectorization strategy, register tiling, software pipelining).
    simple_technique_patterns = (
        "loop_unrolling",
        "strength_reduction",
        "branch_prediction",
        "load_use_scheduling",
        "prefetch_hints",
        "reduce_indirect_loads",
        "data_layout_soa",
        "predicated_arithmetic",
        "min_max_via_intrinsics",
    )
    hard_technique_patterns = (
        "rvv_vectorization",
        "register_tiling",
        "software_pipelining",
        "loop_tiling_icache",
        "im2col_transform",
        "fused_attention",
        "online_softmax",
        "tree_reduction",
        "blocked_layout_transform",
    )

    def __init__(
        self,
        *,
        gpu: "GPUType | None" = None,
        board_host: str | None = None,
        batch_runner_template: Path | None = None,
    ):
        self.gpu = gpu
        # ``board_host`` here is the SSH/serial-server host that owns the
        # FPGA (analogous to OpenCLBackend.board_host). Unlike Adreno,
        # this host talks to the FPGA over a JTAG/UART tunnel, not scp.
        self.board_host = board_host or os.getenv(
            "KERNELBLASTER_RISCV_BOARD_HOST", "root@192.0.2.101",
        )
        self.batch_runner_template = (
            Path(batch_runner_template)
            if batch_runner_template is not None
            else default_batch_runner_template()
        )

    # ---- assets ----
    @property
    def technique_map(self) -> Mapping[str, str]:
        return _RISCV_TECHNIQUE_MAP

    @property
    def database_footer_path(self) -> Path:
        repo_root = Path(__file__).resolve().parents[3]
        return (
            repo_root / "data" / "kernelblaster"
            / "optimization_database_footer_riscv.md"
        )

    # ---- compile + run ----
    async def compile_and_run(
        self,
        main_filepath: Path,
        kernel_filepath: Path,
        gpu: "GPUType",
        timer,
        logger,
        *,
        timeout: int = 3600,
        num_runs: int = 1,
        passed_keyword: str | None = None,
        profile: bool = False,
        extra_files: list[str] | None = None,
        extra_args: str = "",
    ):
        """Compile a Zephyr app around ``kernel_filepath`` and run it
        through the spike simulator (or FireSim / native, depending on
        the exec server's active strategy).

        ``extra_files[0]`` (when set) is interpreted as an ``io.npz``
        path — the modelblaster golden for verify. ``extra_args`` is
        forwarded as a comma-list of spike args (see
        :func:`agents.utils.commands.run_riscv_executable`).
        """
        from ..agents.utils import compile_and_run_riscv

        io_npz: Path | None = None
        if extra_files:
            for f in extra_files:
                if f.endswith(".npz"):
                    io_npz = Path(f)
                    break

        return await compile_and_run_riscv(
            main_filepath, kernel_filepath, gpu, timer, logger,
            timeout=timeout,
            num_runs=num_runs,
            io_npz_path=io_npz,
            spike_args_str=extra_args,
            passed_keyword=passed_keyword,
        )

    # ---- profile parsing ----
    def parse_profile(self, raw_log: str) -> ProfileResult:
        """Extract the RISC-V primary metric from modelblaster output.

        Two data sources:

        * ``=== MODELBLASTER_PROFILE_BEGIN ===`` CSV block — per-op
          ``mcycle`` deltas (``dispatch_id,name,op,shape,cycles``).
          When present, the sum of these is the ``total_cycles``
          reported to the RL loop, and each op's cycles land in
          ``per_kernel_ms`` (repurposed to carry cycles when
          :attr:`metric_name` == "cycles").
        * ``=== MODELBLASTER_WALL_CYCLES === N`` — fallback wall-clock
          in mtime ticks. Used when no PROFILE block appears (older
          harness / kernels without profile hooks).

        Missing both → ``ProfileResult`` with total_cycles=0. The RL
        loop still sees the raw log and can prompt on it.
        """
        per_kernel_cycles: dict[str, float] = {}
        for m_block in _PROFILE_BLOCK_RE.finditer(raw_log):
            body = m_block.group("body")
            for m_row in _PROFILE_ROW_RE.finditer(body):
                # Use ``<op>@<shape>`` as the key so the same op fired
                # multiple times with different shapes doesn't collide.
                op = m_row.group("op").strip()
                shape = m_row.group("shape").strip()
                key = f"{op}@{shape}" if shape else op
                # Multiple firings of the same key → sum (matches how
                # RL would want cumulative cost).
                per_kernel_cycles[key] = (
                    per_kernel_cycles.get(key, 0.0)
                    + float(m_row.group("cycles"))
                )

        wall_ticks = sum(
            int(m.group(1)) for m in _WALL_CYCLES_RE.finditer(raw_log)
        )

        if per_kernel_cycles:
            total_cycles = sum(per_kernel_cycles.values())
            raw_metrics = {
                "total_cycles": int(total_cycles),
                "wall_mtime_ticks": wall_ticks,
            }
        else:
            # Fall back to wall time. Synthesise a single per-op entry
            # so the RL database's per-op layer has something to key on.
            total_cycles = wall_ticks
            if wall_ticks:
                per_kernel_cycles["_wall"] = float(wall_ticks)
            raw_metrics = {
                "total_cycles": int(total_cycles),
                "wall_mtime_ticks": wall_ticks,
            }

        return ProfileResult(
            total_time_ms=total_cycles,
            per_kernel_ms=per_kernel_cycles,
            raw_metrics=raw_metrics,
            raw_log=raw_log,
        )

    # ---- artifact naming ----
    def step_filename(self, trajectory: int, step: int, technique: str) -> str:
        return f"step_{step}_{technique}.c"

    def best_filename(self) -> str:
        return "global_best_rl_optimization.c"

    # ---- default optimizations ----
    def get_default_optimizations(self) -> Mapping[str, list[tuple[str, float]]]:
        return _RISCV_DEFAULT_OPTIMIZATIONS

    # ---- primary metric ----
    @property
    def metric_name(self) -> str:
        return "cycles"

    def format_metric(self, value, *, with_unit: bool = True) -> str:
        if isinstance(value, (int, float)):
            s = f"{int(value)}"
        else:
            s = str(value)
        return f"{s} cycles" if with_unit else s

    def extract_primary_metric(self, profile_result: ProfileResult) -> float:
        # ``raw_metrics["total_cycles"]`` is authoritative; total_time_ms
        # mirrors it for the shared ProfileResult shape.
        return float(profile_result.raw_metrics.get(
            "total_cycles", profile_result.total_time_ms,
        ))

    # ---- State derivation glue ----
    def derive_metrics_for_state(self, profile_result: ProfileResult) -> dict:
        metrics = dict(profile_result.per_kernel_ms)   # per-kernel cycles
        metrics["total_cycles"] = self.extract_primary_metric(profile_result)
        return metrics

    def state_cycles_arg(self, profile_result: ProfileResult) -> int:
        return int(self.extract_primary_metric(profile_result))

    # ---- Prompt + DB glue ----
    def build_strategy_prompt(
        self,
        optimization_entry,
        code: str,
        profile_result: ProfileResult,
        database_content: str,
        description: str = "",
    ) -> str:
        # Deliberately minimal until we have a dedicated
        # generate_riscv_strategy_prompt. Uses the technique_map + the
        # profile log to steer the LLM.
        tech_id = getattr(optimization_entry, "technique_id", "")
        tech_desc = self.technique_map.get(tech_id, description or tech_id)
        return (
            f"Optimise the following C kernel for a RISC-V in-order core "
            f"running under Zephyr on an FPGA-emulated SoC.\n\n"
            f"Technique to apply: **{tech_id}** — {tech_desc}\n\n"
            f"Current profile:\n```\n{profile_result.raw_log}\n```\n\n"
            f"Database context:\n{database_content}\n\n"
            f"Original code:\n```c\n{code}\n```\n\n"
            f"Return the optimised kernel in a single ```c code block. "
            f"Keep the same entry-point function name and signature."
        )

    def build_fix_prompt(
        self,
        code: str,
        error_msg: str,
        database_footer: str = "",
    ) -> str:
        return (
            "The previously generated RISC-V kernel failed to compile "
            "or run under Zephyr on the FPGA target.\n\n"
            f"ERROR LOG:\n```\n{error_msg}\n```\n\n"
            f"ORIGINAL KERNEL CODE:\n```c\n{code}\n```\n\n"
            "Please provide a corrected, fully compilable version. "
            "Return complete C code in one ```c``` block. Keep the "
            "same function name and signature. Avoid host-only headers "
            "(<stdio.h> printf is available via Zephyr's minimal libc; "
            "no <math.h> transcendentals unless you can guarantee "
            "linkage against the target's libm)."
        )

    # ---- Metric-shape glue ----
    def parse_state_metrics(self, raw_log: str, current_metric) -> dict:
        # Re-parse the raw_log for per-kernel cycles then inject the
        # scalar current_metric (same shape as OpenCL's parse_state_metrics
        # so the shared analyze_performance_state code path Just Works).
        pr = self.parse_profile(raw_log)
        metrics = dict(pr.per_kernel_ms)
        metrics["total_cycles"] = float(current_metric or 0.0)
        return metrics

    def state_cycles_from_metric(self, current_metric) -> int:
        return int(current_metric or 0)

    # ---- RL graph-node config ----
    def rl_node_config(self) -> RLNodeConfig:
        # Deliberately reuses the CUDA-shape defaults (no
        # global_best preference, num_pgen=4). RL agent class is left
        # None so a caller can plug in an RLRiscvAgent when it's built
        # — mirrors how OpenCL grew its own RLOpenCLAgent alongside
        # RLNCUAgent.
        return RLNodeConfig(
            state_kernel_fp_input="kernel_c_fp",
            state_perf_fp_output="rl_riscv_perf_fp",
            agent_class=None,
            agent_kernel_fp_kwarg="code_to_optimize_fp",
            fb_config_agent_name="rl_riscv",
            num_pgen=4,
            final_filename="final_rl_riscv_perf.c",
            use_global_best_preference=False,
        )

    # ---- Deterministic-fix pre-pass (P5.5) ----
    # gcc / picolibc-flavored regex repairs for the most common
    # RISC-V + Zephyr failure modes. Covers what the seed markdown
    # warned the LLM to avoid:
    #   1. Host-only headers pulled into a Zephyr kernel (<math.h> for
    #      transcendentals, <malloc.h> for allocation).
    #   2. Missing <stdint.h> / <stddef.h> when the code uses fixed-width
    #      types or size_t. gcc reports "unknown type name 'int32_t'".
    #   3. Missing `static inline` on the RVV intrinsic helpers → linker
    #      "multiple definition" when included from more than one TU.
    _GCC_FIX_RULES: tuple[tuple[re.Pattern, str, str], ...] = (
        (re.compile(r"unknown type name '(int|uint)\d+_t'", re.IGNORECASE),
         "header_stdint", "#include <stdint.h>\n"),
        (re.compile(r"unknown type name 'size_t'", re.IGNORECASE),
         "header_stddef", "#include <stddef.h>\n"),
        (re.compile(r"implicit declaration of function 'memcpy'", re.IGNORECASE),
         "header_string", "#include <string.h>\n"),
    )

    def deterministic_fix(self, code: str, error_msg: str) -> "str | None":
        """gcc/picolibc-flavored regex repairs. Returns repaired code or None.

        Only fires for the two/three most common shapes — most real
        RISC-V failures need LLM understanding of the algorithm and
        fall through to the LLM fix path.
        """
        if not error_msg:
            return None
        repaired = code
        changed = False
        for rx, _label, header in self._GCC_FIX_RULES:
            if rx.search(error_msg) and header.strip() not in repaired:
                inc_match = re.search(
                    r"^\s*#include\s+[<\"][^>\"]+[>\"]", repaired, re.MULTILINE,
                )
                if inc_match:
                    idx = inc_match.end()
                    repaired = repaired[:idx] + "\n" + header + repaired[idx:]
                else:
                    repaired = header + repaired
                changed = True
        return repaired if changed else None

    # ---- LLM response handling ----
    def extract_code_from_response(self, response_text: str) -> str | None:
        """RISC-V: expect ```c fences (same as OpenCL). No fallback tag."""
        from ..agents.utils import extract_code_from_response as _extract
        return _extract(response_text, tag="c")

    # ---- result artifact formatting ----
    _CYCLES_FOOTER_RE = re.compile(
        r"\n*//\s*Elapsed Cycles:\s*\d+\s*$", re.IGNORECASE,
    )

    def format_result_artifact(self, code: str, metric_value: float) -> str:
        """Append ``// Elapsed Cycles: <int>`` (mirrors CUDA convention).

        Stripping any existing footer of the same shape first so
        re-annotating an already-annotated kernel doesn't double-stamp.
        """
        stripped = self._CYCLES_FOOTER_RE.sub("", code).rstrip()
        return f"{stripped}\n\n// Elapsed Cycles: {int(metric_value)}\n"
