# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hardware backend abstraction — Phase 2 (adapter layer).

The ``Backend`` ABC is the unified contract for compile / run / profile / artifact
naming across CUDA (NVIDIA + NCU) and OpenCL (Adreno + event timing). Phase 2
implementations are pure *facades* over the existing free functions in
``kernelblaster.agents.utils`` and the FastAPI servers under
``kernelblaster.servers``; they preserve current behavior so consumers can
migrate one call site at a time.

Phase 3+ will progressively move consumers (database, RL agents, graph nodes)
to call methods on this ABC rather than the backend-specific free functions.

``ProfileResult`` is the key normalization: CUDA currently produces an
integer ``elapsed_cycles`` while OpenCL produces a float ``total_kernel_time_ms``.
The canonical metric across backends is ``ProfileResult.total_time_ms``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class ProfileResult:
    """Normalized profile data across backends.

    ``total_time_ms`` is the canonical metric used for speedup comparison
    (replaces the CUDA-cycles vs OpenCL-ms split). ``raw_metrics`` carries
    backend-specific extras (NCU cycles, occupancy, SM throughput; OpenCL
    per-kernel events) for prompts and dashboards.
    """

    total_time_ms: float
    per_kernel_ms: dict[str, float] = field(default_factory=dict)
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    raw_log: str = ""

    # ---- serialization (Phase 3b) ----
    # Used by the backfill script and (eventually) by RL agents to persist a
    # structured profile.json next to each step's NCU/OpenCL raw log.

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view suitable for ``json.dumps``."""
        return {
            "total_time_ms": self.total_time_ms,
            "per_kernel_ms": dict(self.per_kernel_ms),
            "raw_metrics": dict(self.raw_metrics),
            "raw_log": self.raw_log,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProfileResult":
        return cls(
            total_time_ms=float(d.get("total_time_ms", 0.0)),
            per_kernel_ms=dict(d.get("per_kernel_ms", {})),
            raw_metrics=dict(d.get("raw_metrics", {})),
            raw_log=str(d.get("raw_log", "")),
        )

    def write_json(self, path) -> None:
        """Serialize to ``path``. Parent directory must exist.

        ``raw_log`` is included verbatim — the JSON encoder handles newlines.
        For large logs (~MB), callers may want to omit ``raw_log`` first via
        ``ProfileResult(raw_log="", ...other fields)`` before writing.
        """
        import json
        from pathlib import Path as _Path
        _Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def read_json(cls, path) -> "ProfileResult":
        """Deserialize from a ``profile.json`` produced by ``write_json``."""
        import json
        from pathlib import Path as _Path
        return cls.from_dict(json.loads(_Path(path).read_text()))


@dataclass
class RLNodeConfig:
    """Per-backend configuration consumed by the unified RL-optimization graph node.

    Carries the bits that genuinely diverge between the CUDA and OpenCL
    optimization-graph nodes (curated-artifact root, state-dict keys, agent
    class wiring, final filename, etc.). Each backend produces one of these
    so ``graph/nodes/_rl_node.py`` can drive both flows from a single
    function body — Phase 4e of the backend abstraction refactor.
    """

    # Curated artifact resolution (state-driven, with sensible defaults).
    curated_root_state_key: str          # state.get(this) overrides curated_root_default
    curated_root_default: "Path"

    # On-disk filenames inside a curated <root>/<tier>/<problem>/ directory.
    kernel_filename: str                 # e.g. "init.cu" / "kernel.cl"
    # ``driver_filename`` reuses ``Backend.driver_filename``.

    # State dict keys consumed/produced by the node.
    state_kernel_fp_input: str           # state[key] -> kernel-source path
    state_perf_fp_output: str            # node return dict key
    state_test_code_fp_key: str = "test_code_fp"

    # Agent wiring.
    agent_class: "type | None" = None     # RLNCUAgent or RLOpenCLAgent
    agent_kernel_fp_kwarg: str = "code_to_optimize_fp"
    fb_config_agent_name: str = "rl_ncu"
    num_pgen: int = 4

    # Output.
    final_filename: str = "final.cu"
    # ``best_filename`` (e.g. global_best_rl_optimization.cl) is read off
    # ``Backend.best_filename()`` directly.
    use_global_best_preference: bool = False  # OpenCL prefers global_best over per-traj best

    # Tier mapping: maps base_folder.parent.name to the directory under
    # ``curated_root_default`` that holds the problem. CUDA: identity
    # (level1/level2/...); OpenCL: tier-aware (sol-level2 -> L2 etc.).
    tier_resolver: "Callable[[str], str] | None" = None


class Backend(ABC):
    """Contract every hardware backend satisfies.

    Subclasses must set the four class-level identity attributes (``name``,
    ``kernel_ext``, ``driver_filename``, ``compile_server_env``) and implement
    the abstract methods below.
    """

    # ---- identity (class attributes, set by subclasses) ----
    name: str = ""               # "cuda" | "opencl"
    kernel_ext: str = ""         # ".cu" | ".cl"
    driver_filename: str = ""    # "driver.cpp" | "driver.c"

    # ---- prompts / database assets ----
    @property
    @abstractmethod
    def technique_map(self) -> Mapping[str, str]:
        """Technique-ID -> prose description, consumed by RL prompts."""

    @property
    @abstractmethod
    def database_footer_path(self) -> Path:
        """Path to the optimization-database footer markdown for this backend."""

    # ---- compile + run ----
    # Note: Phase 2 keeps the per-backend signature shape (mirroring
    # ``compile_and_run_cu_file`` / ``compile_and_run_opencl``) so consumers
    # can migrate with a mechanical diff. Phase 4 will normalize on a single
    # signature.
    @abstractmethod
    async def compile_and_run(self, **kwargs):
        """Compile + execute. Returns ``(stdout_list, stderr_list, binary_path, success)``."""

    # ---- profile parsing (this IS unified now) ----
    @abstractmethod
    def parse_profile(self, raw_log: str) -> ProfileResult:
        """Extract a normalized ``ProfileResult`` from a backend-specific raw log."""

    # ---- artifact naming ----
    @abstractmethod
    def step_filename(self, trajectory: int, step: int, technique: str) -> str:
        """Filename used for a single RL step's generated kernel (e.g. ``step_3_tiling.cu``)."""

    @abstractmethod
    def best_filename(self) -> str:
        """Filename for the trajectory-best kernel (e.g. ``global_best_rl_optimization.cu``)."""

    # ---- LLM response handling ----
    @abstractmethod
    def extract_code_from_response(self, response_text: str) -> str | None:
        """Pull kernel code out of an LLM response, using the backend's expected
        code-block tags (CUDA: ``cpp``; OpenCL: ``c`` falling back to ``opencl``).

        Returns ``None`` if no code block matched — callers raise a
        ``FeedbackError`` so the LLM can be re-prompted.
        """

    # ---- result artifact formatting ----
    @abstractmethod
    def format_result_artifact(self, code: str, metric_value: float) -> str:
        """Append the backend's canonical performance footer to ``code``.

        Any existing footer of the same shape is stripped first so re-formatting
        a previously-annotated kernel doesn't produce double footers.

        CUDA footer:  ``// Elapsed Cycles: <int>``
        OpenCL footer: ``// Kernel time: <float> ms``
        """

    # ---- default optimizations (RL fallback catalog) ----
    @abstractmethod
    def get_default_optimizations(self) -> "Mapping[str, list[tuple[str, float]]]":
        """Per-backend fallback catalog: ``bottleneck -> [(technique_id, predicted_pct), ...]``.

        Used by the RL agents' ``_try_add_default_optimizations`` when the LLM
        produces a state name we have no recorded optimizations for. The
        technique IDs are backend-specific (CUDA uses generic names like
        ``memory_coalescing_optimization``; OpenCL uses entries from
        ``technique_map`` like ``1.1_coalesced_access``).
        """

    # ---- primary performance metric ----
    # The "primary metric" is the single number each backend uses to rank
    # kernels: NCU elapsed-cycles for CUDA, OpenCL-event ms for OpenCL.
    # ``Backend.parse_profile`` returns a ``ProfileResult`` with both fields
    # populated; ``extract_primary_metric`` selects the one this backend uses.
    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Short label for the primary metric (``"cycles"`` / ``"ms"``)."""

    @abstractmethod
    def format_metric(self, value, *, with_unit: bool = True) -> str:
        """Format a primary-metric value for logs / artifacts.

        CUDA: ``"12345"`` (or ``"12345 cycles"`` with ``with_unit=True``).
        OpenCL: ``"0.058"`` (or ``"0.058 ms"`` with ``with_unit=True``).
        ``value`` can be a number or the string sentinel ``"N/A"``.
        """

    @abstractmethod
    def extract_primary_metric(self, profile_result: "ProfileResult") -> float:
        """Pull the canonical metric out of a ``ProfileResult``.

        CUDA: ``profile_result.raw_metrics["elapsed_cycles"]``.
        OpenCL: ``profile_result.total_time_ms``.
        """

    # ---- RL graph-node config ----
    @abstractmethod
    def rl_node_config(self) -> "RLNodeConfig":
        """Return the per-backend config consumed by the unified RL-optimization
        graph node — see ``RLNodeConfig`` and ``graph/nodes/_rl_node.py``."""

    # ---- State derivation glue (Phase 4f.3a) ----
    # These two methods carry the per-backend bits that the database needs
    # in order to compute a state for the current kernel. They keep
    # ``Backend`` pure (no database/agent dependency) — the agent calls them
    # to build the args, then passes them to its own ``self.database``.
    @abstractmethod
    def derive_metrics_for_state(self, profile_result: "ProfileResult") -> dict:
        """Build the metrics dict that ``database.get_state_from_ncu_report``
        expects for THIS backend.

        CUDA: parses ``profile_result.raw_log`` (an NCU report) via
        ``parse_ncu_metrics`` to extract Speed-Of-Light metrics.

        OpenCL: returns per-kernel ms (from ``per_kernel_ms``) plus
        ``total_kernel_time_ms`` — the dict shape the database has been
        consuming since the original OpenCL agent landed.
        """

    @abstractmethod
    def state_cycles_arg(self, profile_result: "ProfileResult") -> int:
        """Build the ``elapsed_cycles=`` integer that ``database.get_state_from_ncu_report``
        expects (the database API name is CUDA-flavored; OpenCL fakes it
        with microseconds-as-integer).

        CUDA: returns ``profile_result.raw_metrics["elapsed_cycles"]``.
        OpenCL: returns ``int(profile_result.total_time_ms * 1000)`` —
        the existing OpenCL hack of stuffing microseconds into the
        cycles parameter.
        """

    # ---- Prompt + DB glue (Phase 4f.3d.a) ----
    # Bits that vary per backend but are called from a lifted
    # ``RLAgentBase.apply_optimization``. Kept on the backend so the RL loop
    # body stays fully backend-agnostic.
    @abstractmethod
    def build_strategy_prompt(
        self,
        optimization_entry,
        code: str,
        profile_result: "ProfileResult",
        database_content: str,
        description: str = "",
    ) -> str:
        """Assemble the per-step LLM prompt that asks for optimised code.

        Backends dispatch to their existing prompt generator internally
        (CUDA: ``generate_strategy_guided_prompt``, OpenCL:
        ``generate_opencl_strategy_prompt``) with the shape they expect.
        ``profile_result`` carries the raw log + backend-specific extras
        (CUDA's ``annotated_ncu`` lives in ``raw_metrics``).
        """

    @abstractmethod
    def build_fix_prompt(
        self,
        code: str,
        error_msg: str,
        database_footer: str = "",
    ) -> str:
        """Assemble the "the previous code failed to compile, please fix it"
        prompt. CUDA uses a ```cpp fence and mentions ``cuda_fp16.h`` etc.;
        OpenCL uses a ```c fence and asks for the same signature."""

    def database_update_kwargs(self) -> dict:
        """Extra kwargs forwarded to ``database.update_optimization_result``.

        CUDA passes nothing (relies on the DB's default baseline-file
        speedup parse). OpenCL passes ``current_file_path=None`` to
        suppress that parse because it tracks percent improvement from
        measured ms directly.
        """
        return {}

    # ---- Metric-shape glue (Phase 4f.3d.b) ----
    # The trajectory loop holds two scalar state variables — ``current_metric``
    # (backend primary metric) and ``last_raw_log`` (backend profile stdout /
    # NCU report). These three hooks bridge the shape gap so the loop body
    # can be lifted verbatim.
    @abstractmethod
    def parse_state_metrics(self, raw_log: str, current_metric) -> dict:
        """Build the metrics dict that ``database.analyze_performance_state``
        expects for THIS backend.

        CUDA: ``parse_ncu_metrics(raw_log)`` (Speed-Of-Light section extract).
        OpenCL: ``parse_opencl_profile(raw_log)`` + inject
        ``total_kernel_time_ms=current_metric or 0.0`` — this is what the
        legacy in-loop code did for the OpenCL path.
        """

    @abstractmethod
    def state_cycles_from_metric(self, current_metric) -> int:
        """Build the ``elapsed_cycles=`` int for ``analyze_performance_state``
        from the loop's ``current_metric`` scalar (no ProfileResult available
        here; the loop keeps the metric+raw_log pair, not the full result).

        CUDA: ``int(current_metric or 0)`` (metric IS cycles).
        OpenCL: ``int((current_metric or 0) * 1000)`` (ms -> microseconds-as-int).
        """

    @abstractmethod
    def metric_to_traj_cycles(self, metric) -> int:
        """Convert a backend primary metric into the value stored in
        ``TrajectoryStep.cycles`` (typed ``int``).

        CUDA: ``int(metric)`` — cycles are already integer-valued.
        OpenCL: ``int(metric * 1000)`` — stuff microseconds into the field.

        Inverse of ``metric_from_traj_cycles``.
        """

    @abstractmethod
    def metric_from_traj_cycles(self, cycles: int) -> float:
        """Inverse of ``metric_to_traj_cycles``. Used by the top-level
        ``run()`` loop when it reads ``TrajectoryStep.cycles`` back out to
        select the fastest iteration.

        CUDA: ``float(cycles)``. OpenCL: ``cycles / 1000.0``.
        """
