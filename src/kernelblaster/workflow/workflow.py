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
from __future__ import annotations
import time
import asyncio
import loguru
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator
import shutil

from ..graph import build_graph
from ..config import config, WorkflowConfig
from ..graph.state import save_state_to_json

__all__ = ["WorkflowResult", "run_workflow"]


@dataclass
class WorkflowResult:
    """Outcome of a single workflow invocation.

    Item 2 cleanup Phase 5: the canonical RL-perf field is now
    ``rl_perf_filepath`` (backend-neutral — CUDA or OpenCL). The
    old ``rl_cuda_perf_filepath`` name is retained as a deprecated
    property alias for HTTP-API back-compat with clients that
    destructure the response dict.
    """

    config: WorkflowConfig
    rl_perf_filepath: Path = None  # RL-optimized kernel (CUDA or OpenCL)
    error: str = (
        "Failed code generation due to an error or reaching the maximum number of attempts."
    )
    timeout: bool = False

    def set_error(self, error: str, timeout: bool = False):
        self.error = error
        self.timeout = timeout

    @property
    def rl_cuda_perf_filepath(self) -> Path:
        """Deprecated alias for :attr:`rl_perf_filepath`.

        Kept so HTTP-API consumers of :meth:`generated_codes` that
        destructured ``{"rl_cuda_perf": ...}`` don't break. The dict
        still carries both keys — see :meth:`generated_codes`.
        """
        return self.rl_perf_filepath

    @property
    def success(self) -> bool:
        return self.rl_perf_filepath is not None

    def agents(self) -> Iterator[str]:
        """Names of the agents this result describes."""
        yield "rl_perf"

    def running_agents(self) -> Iterator[str]:
        """Names of the agents that are supposed to be running."""
        yield "rl_perf"

    @property
    def generated_codes(self) -> dict[str, str]:
        """Role-keyed dict of produced code artifacts.

        Contains both ``"rl_perf"`` (canonical) and ``"rl_cuda_perf"``
        (deprecated alias) pointing at the same path. HTTP-API clients
        destructuring the legacy key keep working; new clients should
        read ``"rl_perf"``.
        """
        def stringify(filepath: Path | None) -> str | None:
            if filepath is None:
                return None
            return str(filepath)

        path_str = stringify(self.rl_perf_filepath)
        return {"rl_perf": path_str, "rl_cuda_perf": path_str}

    def write_failures(self, folder: str):
        if self.rl_perf_filepath is None:
            (folder / "failed_rl_perf").write_text(self.error)

    def remove_existing_files(self, folder: Path):
        # Legacy marker name kept for cleanup so any pre-Phase-5 run
        # tree still gets its failure marker cleared on retry.
        for marker in ("failed_rl_perf", "failed_rl_cuda_perf"):
            failed_file = folder / marker
            if failed_file.exists() and self.config.retry_failed:
                # Remove the agent folder if the retry_failed flag is set.
                shutil.rmtree(folder / "rl_ncu", ignore_errors=True)
            failed_file.unlink(missing_ok=True)


async def run_workflow(
    task_id: str,
    user_message: str,
    reference_code: str,
    folder: Path,
    workflow_config: WorkflowConfig,
    job_logger: loguru.Logger,
    timeout_seconds: int,
    shared_database=None,
    problem=None,
) -> WorkflowResult:
    """Run the LangGraph workflow for a single problem.

    ``problem`` (``data.sources.Problem``) is optional for back-compat
    but strongly preferred — when set, it lands in ``state["problem"]``
    so :mod:`kernelblaster.graph.nodes._rl_node` uses the
    Problem-driven path (role-keyed curated_artifacts) instead of
    deriving paths from the run-folder name.
    """
    folder.mkdir(exist_ok=True, parents=True)
    start = time.time()

    job_logger.info(f"Starting workflow for task {task_id}.")
    config.print_config(job_logger)

    result = WorkflowResult(config=workflow_config)

    # Prepare output directory for the run
    result.remove_existing_files(folder)

    workflow = build_graph()
    workflow_input = {
        "user_message": user_message,
        "reference_code": reference_code,
        "folder": folder,
        "logger": job_logger,
        "model": workflow_config.model,
        # Pass shared database directly from caller (runner)
        "shared_optimization_database": shared_database,
        **workflow_config.dict(),
    }
    if problem is not None:
        workflow_input["problem"] = problem

    try:
        final_state = await asyncio.wait_for(
            workflow.ainvoke(workflow_input),
            timeout=timeout_seconds,
        )
        save_state_to_json(final_state, folder / "state.json")
        # Read whichever backend's RL-perf path is populated. The graph
        # only invokes ONE backend per run, so at most one of these keys
        # is set. ``rl_ncu_cuda_fp`` = CUDA (RLNodeConfig for CUDA),
        # ``rl_opencl_perf_fp`` = OpenCL. Non-null wins.
        rl_perf_fp = (
            final_state.get("rl_ncu_cuda_fp")
            or final_state.get("rl_opencl_perf_fp")
        )
        result = WorkflowResult(
            config=workflow_config,
            rl_perf_filepath=rl_perf_fp,
        )
    except asyncio.TimeoutError:
        result.set_error(
            f"Timeout after {timeout_seconds / 60} minutes",
            timeout=True,
        )

    # Successes will be written by the agents themselves
    # We write the failures here instead of inside the agents incase of exceptions or timeouts.
    result.write_failures(folder)
    duration = time.time() - start
    job_logger.info(f"Workflow completed in {duration:0.2f} seconds")
    return result
