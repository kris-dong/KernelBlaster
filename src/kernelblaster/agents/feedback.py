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

import asyncio
from pathlib import Path
import loguru
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional
import time
import re

from ..config import config, GPUType
from .utils import (
    generate_code_retry,
    LLMResponse,
    NamedTimer,
    write_jsonl,
    extract_code_from_response,
    write_code_to_file,
    FeedbackError,
    convert_messages_to_string,
)
from .utils.query import (
    extract_code_from_response,
    process_messages,
    trim_messages,
    TrimError,
)

__all__ = ["FeedbackConfig", "FeedbackAgent", "FeedbackError"]


@dataclass
class FeedbackConfig:
    """Configuration for FeedbackAgent and its subclasses."""

    agent_name: str
    base_folder: Path
    logger: loguru.Logger
    init_user_prompt: str
    model: str
    gpu: GPUType
    test_code_fp: Optional[Path] = None
    retry_failed: bool = False
    num_pgen: int = config.NUM_PARALLEL_GENERATIONS_PER_ATTEMPT
    max_attempts: int = config.MAX_ATTEMPTS
    system_prompt: Optional[str] = None
    file_rules: list[Callable] = field(default_factory=list)


@dataclass
class Feedback:
    new_messages: list[dict] = field(default_factory=list)
    llm_calls: list[LLMResponse] = field(default_factory=list)
    success: bool = False
    filename: str = None
    contents: str = None
    # A dictionary of the time taken for various steps in the agent.
    # The keys are the names of the steps and the values are the times in seconds.
    # Using the task_timer will automatically add the time taken to the durations dictionary.
    durations: dict[str, float] = field(default_factory=dict)
    feedback: str = None


def write_metrics(filepath: Path, threads: dict[int, dict]):
    metrics_file = [
        {
            "attempt_id": attempt_id,
            "thread_id": thread_id,
            **asdict(feedback),
            "version": 1.2,
        }
        for thread_id in threads
        for attempt_id, feedback in enumerate(threads[thread_id]["feedbacks"])
    ]
    metrics_file = list(
        sorted(metrics_file, key=lambda x: (x["attempt_id"], x["thread_id"]))
    )
    write_jsonl(filepath, metrics_file)


class FeedbackAgent:
    def __init__(
        self,
        fb_config: FeedbackConfig,
    ):
        self.fb_config = fb_config

        self.agent_name = fb_config.agent_name
        self.base_folder = Path(fb_config.base_folder)
        self.folder = self.base_folder / self.agent_name
        self.system_prompt = fb_config.system_prompt
        self.init_user_prompt = fb_config.init_user_prompt
        self.model = fb_config.model
        self.num_pgen = fb_config.num_pgen
        self.max_attempts = fb_config.max_attempts

        self.folder.mkdir(exist_ok=True, parents=True)
        self.timers = []

        # Add a custom logger for this agent
        self.agent_logger = fb_config.logger.bind(
            agent_name=self.agent_name, folder=str(self.folder)
        )

        self.task_loggers = []
        self.file_rules = fb_config.file_rules
        self.retry_failed = fb_config.retry_failed
        self.gpu = fb_config.gpu

    def check_rules(self, code: str):
        for rule in self.file_rules:
            rule(code)

    def get_intermediate_filepath(self, attempt_id, task_id) -> Path:
        return self.folder / f"attempt{attempt_id}_task{task_id}.cu"

    def get_ids_from_filepath(self, filepath: Path) -> tuple[int, int]:
        """
        Get the attempt_id and task_id from the filepath.
        The filepath is expected to be of the form *attempt<attempt_id>_task<task_id>.*
        """
        match = re.search(r"attempt(\d+)_task(\d+)", filepath.stem)
        assert match, f"Failed to parse filepath for attempt and task id: {filepath}"
        assert (
            len(match.groups()) == 2
        ), f"Failed to parse filepath for attempt and task id: {filepath}"
        return int(match.group(1)), int(match.group(2))

    def get_code_from_response(
        self, response, attempt_id, task_id, logger
    ) -> tuple[str, Path]:
        code = extract_code_from_response(response)
        if code is None:
            raise FeedbackError(
                "Error: The code should be contained within ```cpp and ``` tags."
            )
        filepath = self.get_intermediate_filepath(attempt_id, task_id)
        write_code_to_file(code, filepath, logger)
        return code, filepath

    async def get_feedback(self, response, attempt_id, task_id) -> Feedback:
        # implement in subclasses
        raise NotImplementedError

    async def initialize(self):
        # optionally implement in subclasses
        return

    def choose_best_task(self, successful_tasks: list[Path]) -> Path:
        # optionally implement in subclasses
        return successful_tasks[0]

    @staticmethod
    def raise_numerics_verification_error(
        stdouts: list[str], stderr: list[str], custom_msg=""
    ) -> FeedbackError:
        if not custom_msg:
            custom_msg = "The numerics verification failed."
        raise FeedbackError(
            f"{custom_msg}. Please check your implementation carefully and try again. \nstdout:\n{stdouts[0]}\nstderr:\n{stderr[0]}",
        )

    @staticmethod
    def raise_time_measurement_error(
        stdouts: list[str], stderr: list[str]
    ) -> FeedbackError:
        raise FeedbackError(
            f"The time measurement failed. Please check your implementation carefully and try again:\nstdout:\n{stdouts[0]}\nstderr:\n{stderr[0]}",
        )

    def check_for_existing_run(self) -> str | None:
        finished_fp = self.folder / ".finished"
        successful_files = list(self.folder.glob("success_*"))

        # The tasks are named success_attempt<attempt_id>_task<task_id>.cu
        # We want to sort the tasks based on attempt_id and then task_id
        # Sort the tasks by attempt_id and task_id
        successful_files.sort(key=lambda x: self.get_ids_from_filepath(x))

        attempt_ids = [self.get_ids_from_filepath(p)[0] for p in successful_files]

        # If we find successful files from multiple attempts, keep only the most recent attempt.
        # This situation can happen when a previous successful run exists and the workflow is restarted
        # (e.g., with retry_failed=True) producing new attempts in the same folder. Instead of failing, we
        # filter to the latest attempt and proceed.
        unique_attempt_ids = set(attempt_ids)
        if len(unique_attempt_ids) > 1:
            latest_attempt_id = max(unique_attempt_ids)
            self.agent_logger.warning(
                f"Detected successful files from multiple attempts {sorted(unique_attempt_ids)} in {self.folder}. "
                f"Using the latest attempt id {latest_attempt_id}."
            )
            successful_files = [
                p for p in successful_files if self.get_ids_from_filepath(p)[0] == latest_attempt_id
            ]
            # Recompute attempt_ids after filtering
            attempt_ids = [latest_attempt_id] * len(successful_files)

        if finished_fp.exists() and len(successful_files):
            filename = self.choose_best_task(successful_files)
            self.agent_logger.warning(
                f"Found an existing solution for this agent: {filename}"
            )
            return filename

        elif finished_fp.exists() and not len(successful_files):
            if self.retry_failed:
                finished_fp.unlink()
                return None
            else:
                self.agent_logger.warning(
                    f"No successful solutions found for this agent and retry_failed flag is not set. Skipping problem as failed."
                )
                return "__failed__"

        existing_files = list(self.folder.glob("*"))
        if len(existing_files):
            self.agent_logger.warning(
                f"No successful solutions found for this agent. Regenerating..."
            )
            import shutil
            for file in existing_files:
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    # Remove directories (like trajectory directories)
                    shutil.rmtree(file)
            return None

    async def __run(self, messages, attempt_id, task_id) -> Feedback:
        timer = self.timers[task_id]
        logger = self.task_loggers[task_id]

        prompt_path = self.folder / f"attempt{attempt_id}_task{task_id}_prompt.md"
        prompt_path.write_text(convert_messages_to_string(messages))

        timer.reset()
        timer.start("attempt")
        logger.info(f"Generating response with {self.model}...")
        try:
            response = await generate_code_retry(messages, self.model, logger)
        except Exception as e:
            logger.error(f"Failed to generate code: {e}")
            return Feedback()
        assert response.generations, "No generations found"
        logger.info(
            f"Response generation completed in {response.elapsed_time:0.2f} seconds"
        )

        generation = response.generations[0]
        prompt_path.write_text(
            convert_messages_to_string(
                messages, response=generation, usage=response.usage
            )
        )
        new_feedback = await self.get_feedback(generation, attempt_id, task_id, logger)
        timer.stop("attempt")

        new_feedback.llm_calls.insert(0, response)
        new_feedback.durations = self.timers[task_id].elapsed.copy()
        return new_feedback

    async def run(self) -> Path:
        """
        Run the agent and return the filename and the generated code.
        """
        existing_attempt = self.check_for_existing_run()
        if isinstance(existing_attempt, Path):
            return existing_attempt
        elif existing_attempt == "__failed__":
            return None

        successful_tasks = []
        chosen_task = None
        start_time = time.time()

        custom_logger_id = self.agent_logger.add(
            self.folder / "run.log",
            level=config.LOG_LEVEL,
            backtrace=True,
            diagnose=True,
            format=config.CUSTOM_LOGGER_FORMAT,
            filter=lambda record: record["extra"].get("folder") == str(self.folder),
        )
        self.agent_logger.info(f"Running {self.folder}...")

        try:
            await self.initialize()
        except FeedbackError as e:
            self.agent_logger.error(
                f"Failed to initialize agent for {self.folder}: {e}"
            )
            return None

        initial_messages = [
            {
                "role": "system",
                "content": self.system_prompt if self.system_prompt else "",
            },
            {
                "role": "user",
                "content": self.init_user_prompt if self.init_user_prompt else "",
            },
        ]
        threads = {
            i: {
                "messages": process_messages(initial_messages, self.model),
                "feedbacks": [],
                "running": True,
            }
            for i in range(self.num_pgen)
        }
        for attempt in range(self.max_attempts):
            tasks = {}
            self.task_loggers.clear()
            self.timers.clear()

            for i in range(self.num_pgen):
                if not threads[i]["running"]:
                    continue
                try:
                    threads[i]["messages"] = trim_messages(
                        threads[i]["messages"],
                        logger=self.agent_logger,
                    )
                    self.task_loggers.append(
                        self.agent_logger.bind(attempt_id=attempt, task_id=i)
                    )
                    self.timers.append(NamedTimer())
                    tasks[i] = asyncio.create_task(
                        self.__run(
                            deepcopy(threads[i]["messages"]),
                            attempt,
                            i,
                        )
                    )
                except TrimError as e:
                    self.agent_logger.warning(
                        f"Failed to trim messages: {e}, dropping this thread"
                    )
                    threads[i]["running"] = False
                    continue

            feedbacks = await asyncio.gather(
                *[task for task in tasks.values() if task is not None]
            )
            for i, feedback in enumerate(feedbacks):
                if not threads[i]["running"]:
                    continue
                threads[i]["feedbacks"].append(feedback)
                for message in feedback.new_messages:
                    threads[i]["messages"].append(message)
                if feedback.success:
                    successful_tasks.append(feedback.filename)

            # save metrics to the file
            write_metrics(self.folder / "metrics.jsonl", threads)

            if len(successful_tasks):
                # successful tasks found in this attempt. Choose the best one and break
                chosen_task = self.choose_best_task(successful_tasks)
                self.agent_logger.info(
                    f"Successfully generated and verified code in task {chosen_task}"
                )
                break

        (self.folder / ".finished").write_text("")
        duration = time.time() - start_time
        self.agent_logger.info(f"Agent completed in {duration:0.2f} seconds.")
        self.agent_logger.remove(custom_logger_id)
        if chosen_task is None:
            self.agent_logger.error(
                "Failed to generate and verify correct code after multiple attempts."
            )
            return None
        return chosen_task
