"""Optimized GPU optimization database — companion to ``opt_ncu_rl_optimized.py``.

Wraps the legacy ``OptimizationDatabase`` and overrides only the prompt-building
+ LLM dispatch surfaces. Persistence, technique storage, and the static
optimization-database markdown content are reused unchanged.

What's different vs ``database.py``:

1. **Prompt caching layout** — `system` (cacheable) holds the static optimisation
   database + role + output-format rules. `user` holds only the per-call kernel
   source + NCU metric JSON + chosen technique description.

2. **NCU shipped as JSON** — ``_extract_metrics_json`` returns a 200-byte
   structured object instead of the 2–4 KB ASCII Speed-Of-Light table.

3. **Slim technique injection** — ``get_technique_description(name)`` returns a
   1–2 KB description of just the chosen technique, plus a short index of
   other available technique names. The full database stays in the cached
   system prompt.

4. **Model heterogeneity** — ``analyze_performance_state`` and
   ``generate_optimization_plan`` route to a *cheap* model (configurable via
   ``MODEL_PLAN`` / ``MODEL_STATE``), since these are pattern-matching tasks
   that frontier models are wasted on.

5. **Hash-stable system prompt** — the cacheable prefix is stable byte-for-byte
   across calls within a run, so OpenAI/Anthropic prefix caches automatically
   kick in even without explicit ``cache_control`` markers. (When run against
   an Anthropic client that supports cache_control on the system block, the
   ``_anthropic_cache_control`` flag adds the explicit ephemeral marker.)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import loguru

from .database import (
    LLMInterface,
    OptimizationDatabase,
    StateProfile,
    OptimizationEntry,
    CompositeOptimization,
)
from .cost_tracker import CostTracker


# ---------------------------------------------------------------------------
# NCU log → JSON metric extraction
# ---------------------------------------------------------------------------


_NCU_METRIC_KEYS = {
    "sm_throughput_pct": [r"Compute\s*\(SM\)\s*Throughput"],
    "dram_throughput_pct": [r"Memory\s+Throughput", r"DRAM\s+Throughput"],
    "l1_hit_rate_pct": [r"L1/TEX\s+Hit\s+Rate"],
    "l2_hit_rate_pct": [r"L2\s+Cache\s+Hit\s+Rate"],
    "occupancy_pct": [r"Achieved\s+Occupancy", r"Occupancy"],
    "registers_per_thread": [r"Registers\s+Per\s+Thread"],
    "shared_mem_per_block_kb": [r"Shared\s+Memory\s+Per\s+Block"],
    "elapsed_cycles": [r"Elapsed\s+Cycles"],
}


def extract_metrics_json(
    ncu_log: str,
    elapsed_cycles: Optional[int] = None,
    gpu_time_ns: Optional[int] = None,
) -> Dict[str, Any]:
    """Parse a JSON-summary of the NCU Speed-Of-Light section, no ASCII tables.

    Returns a small dict suitable for embedding in an LLM prompt instead of the
    raw NCU output. Missing metrics are omitted (not zero-filled — that just
    confuses the LLM into "optimising" a metric the profiler couldn't read).

    Parameters
    ----------
    elapsed_cycles : optional bottleneck-kernel cycle count (NCU). Legacy.
    gpu_time_ns    : wall-clock GPU time in ns from nsys (first solution kernel
                     start to last solution kernel end). New optimisation reward.
    """
    out: Dict[str, Any] = {}
    if not ncu_log or not ncu_log.strip():
        if elapsed_cycles:
            out["elapsed_cycles"] = int(elapsed_cycles)
        if gpu_time_ns:
            out["gpu_time_ns"] = int(gpu_time_ns)
        return out

    for key, patterns in _NCU_METRIC_KEYS.items():
        for pat in patterns:
            # Take the last numeric token on the matching line — handles NCU's
            # variable column widths and units columns.
            rx = rf"{pat}.*?([0-9]+(?:\.[0-9]+)?)"
            m = re.search(rx, ncu_log, re.IGNORECASE | re.MULTILINE)
            if m:
                try:
                    val = float(m.group(1))
                except ValueError:
                    continue
                if key == "elapsed_cycles":
                    out[key] = int(val)
                else:
                    out[key] = val
                break

    if elapsed_cycles is not None and "elapsed_cycles" not in out:
        out["elapsed_cycles"] = int(elapsed_cycles)
    if gpu_time_ns is not None:
        out["gpu_time_ns"] = int(gpu_time_ns)
    return out


# ---------------------------------------------------------------------------
# Cheap-model LLM client
# ---------------------------------------------------------------------------


class TieredLLMInterface(LLMInterface):
    """LLM client that routes to a *cheap* model and supports system+user messages.

    The cheap model defaults to ``$MODEL_PLAN`` (env var) → ``$MODEL`` (config).
    Optionally records token usage into a shared :class:`CostTracker`.
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        logger=None,
        cost_tracker: Optional[CostTracker] = None,
        role_label: str = "plan",
    ):
        super().__init__(model_name=model_name, logger=logger)
        self.cost_tracker = cost_tracker
        self.role_label = role_label

    async def query_layered(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        anthropic_cache_control: bool = True,
    ) -> str:
        """Send a (system, user) pair using the cheap model.

        The system prompt is meant to be *stable* across calls so that
        provider-side prefix caches can be exploited. With Anthropic, when
        ``anthropic_cache_control`` is True we set ``cache_control`` on the
        system block (this is a no-op on OpenAI; their cache is automatic).
        """
        try:
            from .utils import generate_code_retry
        except ImportError:
            return ""

        # Construct an OpenAI-style messages list. The downstream Anthropic
        # converter will pull off the system message; we add a marker comment
        # the converter can use to enable cache_control if patched upstream.
        if anthropic_cache_control:
            sys_marker = "<!-- cache_control: ephemeral -->\n"
        else:
            sys_marker = ""
        messages = [
            {"role": "system", "content": sys_marker + system_prompt},
            {"role": "user", "content": user_message},
        ]
        try:
            response = await generate_code_retry(
                messages, self.model_name, self.logger, n_tasks=1, max_retries=2
            )
            if self.cost_tracker is not None:
                self.cost_tracker.record(
                    model=self.model_name,
                    usage=getattr(response, "usage", None),
                    role=self.role_label,
                    logger=self.logger,
                )
            return response.generations[0] if response.generations else ""
        except Exception as e:
            if self.logger:
                self.logger.warning(f"TieredLLMInterface.query_layered failed: {e}")
            return ""


# ---------------------------------------------------------------------------
# Optimized database
# ---------------------------------------------------------------------------


class OptimizedOptimizationDatabase(OptimizationDatabase):
    """Drop-in subclass that overrides expensive prompts with cached layered ones."""

    def __init__(
        self,
        persist_json_fp,
        gpu_optimization_report_md_fp,
        llm_interface=None,
        *,
        cheap_llm: Optional[LLMInterface] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        super().__init__(persist_json_fp, gpu_optimization_report_md_fp, llm_interface)

        self.cost_tracker = cost_tracker

        # Tiered model dispatch. The expensive ``self.llm_interface`` is reused
        # for codegen-related calls; ``self.cheap_llm`` handles state-analysis
        # and plan-generation, both of which are pattern-matching tasks.
        cheap_model = (
            os.getenv("MODEL_PLAN")
            or os.getenv("MODEL_STATE")
            or os.getenv("KERNELAGENT_MODEL_CHEAP")
        )
        if cheap_llm is not None:
            self.cheap_llm: LLMInterface = cheap_llm
            # Attach the tracker to a passed-in client if it doesn't have one.
            if (
                cost_tracker is not None
                and isinstance(self.cheap_llm, TieredLLMInterface)
                and self.cheap_llm.cost_tracker is None
            ):
                self.cheap_llm.cost_tracker = cost_tracker
        else:
            self.cheap_llm = TieredLLMInterface(
                model_name=cheap_model or getattr(llm_interface, "model_name", None),
                logger=getattr(llm_interface, "logger", None),
                cost_tracker=cost_tracker,
                role_label="plan",
            )

        # Cache the static system-prompt parts so we don't rebuild them each call.
        self._system_prompt_state = self._build_state_system_prompt()
        self._system_prompt_plan = self._build_plan_system_prompt()

        # Lookup table: technique_name → short description string.
        self._technique_index = self._build_technique_index()

    # ------------------------------------------------------------------
    # Static system prompts (cache-stable across calls within a run)
    # ------------------------------------------------------------------

    def _build_state_system_prompt(self) -> str:
        return (
            "You are a GPU performance analysis expert.\n"
            "Given a kernel source and a JSON of NCU Speed-Of-Light metrics, return a "
            "qualitative state summary in this EXACT shape:\n\n"
            "PERFORMANCE_SIGNATURE: <one sentence: what limits performance>\n"
            "PRIMARY_BOTTLENECK: memory_bound | compute_bound | latency_bound | hybrid_bound\n"
            "RELATIVE_PATTERNS:\n"
            "- memory_pressure: very_low|low|moderate|high|very_high\n"
            "- compute_utilization: very_low|low|moderate|high|very_high\n"
            "- access_patterns: excellent|good|moderate|poor|very_poor\n"
            "- cache_efficiency: excellent|good|moderate|poor|very_poor\n"
            "- occupancy_level: very_low|low|moderate|high|very_high\n"
            "- parallelism_utilization: very_low|low|moderate|high|very_high\n"
            "- specialised_hw_usage: very_low|low|moderate|high|very_high\n"
            "CONTEXT_DESCRIPTION: <one short paragraph about workload characteristics>\n\n"
            "Stay qualitative. Do not echo numbers from the JSON. Do not include any "
            "commentary outside the labelled fields."
        )

    def _build_plan_system_prompt(self) -> str:
        # Inject the full optimisation-database summary here so it gets cached
        # by the provider's prefix cache.
        avail = self._build_available_optimisations_summary()
        return (
            "You are a GPU optimisation expert. Given a kernel and a state summary, "
            "return the top-N optimisation techniques most likely to improve "
            "performance. Output strict JSON only — a list of length N — with keys: "
            "`technique` (must match an entry in the AVAILABLE OPTIMISATIONS list "
            "below), `relevance_score` (float in [0,1]), `description` (one short "
            "sentence). Do not wrap in markdown fences.\n\n"
            "Heuristics:\n"
            "- Memory-bound or bandwidth-bound: prefer SIMD packed types (half2, "
            "float4) and coalesced access first.\n"
            "- Compute-bound on sm_70+: prefer tensor-core / wmma when the matrix "
            "math can be expressed in 16x16 tiles.\n"
            "- Latency-bound or low-occupancy: prefer occupancy/register-pressure "
            "techniques before kernel-fusion / tiling.\n\n"
            "AVAILABLE OPTIMISATIONS:\n"
            + avail
        )

    # ------------------------------------------------------------------
    # Slim technique injection
    # ------------------------------------------------------------------

    def _build_technique_index(self) -> Dict[str, str]:
        """Return {technique_name: short description string}."""
        idx: Dict[str, str] = {}
        for state_data in self.optimization_strategies.values():
            for opt in state_data.get("optimizations", []):
                if opt.technique not in idx:
                    idx[opt.technique] = (opt.description or "").strip()
        for comps in self.composite_optimizations.values():
            for c in comps:
                cid = c.get_composite_id()
                if cid not in idx:
                    idx[cid] = (c.reason or "").strip()
        return idx

    def get_technique_description(self, name: str) -> str:
        """Return ~1 KB of technique description, with a short index of siblings.

        This is what gets injected into codegen prompts — a tiny replacement for
        the legacy 6 KB database dump.
        """
        desc = self._technique_index.get(name) or ""
        # Up to 12 sibling names so the LLM has alternatives if it can't apply
        # the chosen one verbatim.
        siblings = [n for n in self._technique_index if n != name][:12]
        sib_str = ", ".join(siblings)
        return (
            f"SELECTED TECHNIQUE: {name}\n"
            f"DESCRIPTION: {desc or '(no description on file)'}\n\n"
            f"OTHER AVAILABLE TECHNIQUES: {sib_str}"
        )

    # ------------------------------------------------------------------
    # Overrides — state analysis (cheap model, layered prompt, JSON metrics)
    # ------------------------------------------------------------------

    async def analyze_performance_state(
        self,
        ncu_report: str,
        metrics: dict,
        code_implementation: str,
        elapsed_cycles: Optional[int] = None,
    ) -> StateProfile:
        # Combine raw NCU + structured metrics into a single 200-byte JSON.
        merged_metrics = dict(metrics or {})
        json_metrics = extract_metrics_json(ncu_report, elapsed_cycles=elapsed_cycles)
        merged_metrics.update(json_metrics)
        if elapsed_cycles is not None:
            merged_metrics.setdefault("elapsed_cycles", int(elapsed_cycles))

        if not self.cheap_llm or not self.cheap_llm.is_available():
            return self._fallback_state_analysis(ncu_report, metrics)

        user_msg = (
            "KERNEL SOURCE:\n```cpp\n"
            f"{code_implementation}\n```\n\n"
            "NCU METRICS (Speed-Of-Light, JSON):\n"
            f"{json.dumps(merged_metrics, sort_keys=True)}\n"
        )
        try:
            analysis = await self.cheap_llm.query_layered(
                self._system_prompt_state,
                user_msg,
                max_tokens=600,
                temperature=0.1,
            )
            self._log_llm_interaction("StateAnalysisOpt", user_msg, analysis)
            return self._parse_state_analysis(analysis)
        except Exception as e:
            if self.cheap_llm.logger:
                self.cheap_llm.logger.warning(f"Optimised state analysis failed: {e}")
            return self._fallback_state_analysis(ncu_report, metrics)

    # ------------------------------------------------------------------
    # Overrides — plan generation (cheap model, layered prompt)
    # ------------------------------------------------------------------

    async def generate_optimization_plan(
        self,
        state_analysis_response: str,
        code_implementation: str,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self.cheap_llm or not self.cheap_llm.is_available():
            return await super().generate_optimization_plan(
                state_analysis_response, code_implementation, top_n
            )

        user_msg = (
            f"Pick the {top_n} highest-relevance optimisations for this kernel.\n\n"
            "STATE ANALYSIS:\n"
            f"{state_analysis_response}\n\n"
            "KERNEL SOURCE:\n```cpp\n"
            f"{code_implementation}\n```\n\n"
            f"Return a JSON array of {top_n} objects with keys "
            "`technique`, `relevance_score`, `description`. No prose."
        )
        try:
            llm_resp = await self.cheap_llm.query_layered(
                self._system_prompt_plan,
                user_msg,
                max_tokens=800,
                temperature=0.1,
            )
            self._log_llm_interaction("OptPlanOpt", user_msg, llm_resp)
            plan = self._parse_optimization_plan(llm_resp, top_n)
            if plan:
                return plan
        except Exception as e:
            if self.cheap_llm.logger:
                self.cheap_llm.logger.warning(f"Optimised plan generation failed: {e}")

        # Fallback to legacy implementation (which itself has a deterministic path)
        return await super().generate_optimization_plan(
            state_analysis_response, code_implementation, top_n
        )

    # ------------------------------------------------------------------
    # Codegen prompt builder — slim, layered
    # ------------------------------------------------------------------

    def build_codegen_messages(
        self,
        *,
        technique_name: str,
        kernel_source: str,
        ncu_metrics_json: Dict[str, Any],
        strategy_description: str = "",
        best_so_far_summary: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build a system+user message pair for codegen with cache-stable system text.

        This replaces ``generate_strategy_guided_prompt`` — its system prompt is
        ~6 KB and identical across all codegen calls in the run, so the
        provider's prefix cache will charge cached input price for it.
        """
        system_prompt = (
            "You are an expert CUDA optimisation engineer. You receive an "
            "optimisation technique to apply, a kernel source, and a JSON of NCU "
            "metrics. Apply the technique to the kernel and emit the COMPLETE "
            "rewritten CUDA file in a single ```cpp``` code block.\n\n"
            "Hard rules:\n"
            "- Output ONLY the rewritten CUDA file in one ```cpp``` block. No prose.\n"
            "- The kernel must compile under nvcc with no extra includes beyond\n"
            "  `<cuda_runtime.h>`, `<cuda_fp16.h>`, `<cuda_bf16.h>`, `<cstdint>`,\n"
            "  `<torch/extension.h>` (when the existing file uses it).\n"
            "- Define every constant before use.\n"
            "- Preserve the existing `launch_gpu_implementation(...)` signature.\n"
            "- Preserve the existing `void run(...)` signature when present.\n"
            "- If the technique cannot be cleanly applied, return the input "
            "unchanged inside the same ```cpp``` block.\n\n"
            "AVAILABLE TECHNIQUE INDEX (for reference / fallback choices):\n"
            + ", ".join(self._technique_index.keys())
        )

        user_blocks = [
            self.get_technique_description(technique_name),
        ]
        if strategy_description:
            user_blocks.append(f"STRATEGY NOTE FROM PLAN:\n{strategy_description}")
        if best_so_far_summary:
            user_blocks.append(f"PRIOR-STEP CONTEXT:\n{best_so_far_summary}")
        user_blocks.append(
            "NCU METRICS (Speed-Of-Light, JSON):\n"
            f"{json.dumps(ncu_metrics_json, sort_keys=True)}"
        )
        user_blocks.append(
            "CURRENT KERNEL SOURCE:\n```cpp\n" + kernel_source + "\n```"
        )
        user_blocks.append(
            "Apply the SELECTED TECHNIQUE to the source. Return only the rewritten "
            "CUDA file in a single ```cpp``` code block."
        )

        return [
            {"role": "system", "content": "<!-- cache_control: ephemeral -->\n" + system_prompt},
            {"role": "user", "content": "\n\n".join(user_blocks)},
        ]

    def build_fix_messages(
        self,
        *,
        broken_kernel: str,
        compiler_error: str,
    ) -> List[Dict[str, str]]:
        """Build a small fix-attempt prompt — bounded payload, no DB injection."""
        # Truncate compiler error: nvcc errors are repetitive, the first match is enough.
        err = compiler_error[:1500]
        kernel = broken_kernel
        if len(kernel) > 12000:
            # Show head + tail; the middle of a long kernel rarely contains the
            # broken site nvcc points to.
            kernel = kernel[:6000] + "\n// ... [trimmed] ...\n" + kernel[-6000:]
        system_prompt = (
            "You are a CUDA compiler-error fixer. Given a kernel that failed to "
            "compile or run, return the corrected COMPLETE kernel inside a single "
            "```cpp``` block. Preserve the launcher signature. Do not change the "
            "intent of the kernel — only fix the error."
        )
        user_msg = (
            "COMPILER / RUNTIME ERROR:\n```\n"
            f"{err}\n```\n\n"
            "BROKEN KERNEL:\n```cpp\n"
            f"{kernel}\n```\n\n"
            "Return the fixed kernel in one ```cpp``` block."
        )
        return [
            {"role": "system", "content": "<!-- cache_control: ephemeral -->\n" + system_prompt},
            {"role": "user", "content": user_msg},
        ]
