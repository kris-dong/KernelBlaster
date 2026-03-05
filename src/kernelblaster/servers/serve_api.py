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
import argparse
import asyncio
import sys
from loguru import logger
from pathlib import Path
import time
import uuid
import traceback
from typing import Optional, List
import json

import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..graph.state import load_state_from_json, compare_states
from ..agents.utils.file_operations import get_agent_status
from ..config import config, WorkflowConfig, GPUType
from ..resources import *
from ..workflow import *


app = FastAPI(title="KernelBlaster API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Task tracking
class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class Task:
    def __init__(self, task_id, params, user_id):
        self.task_id = task_id
        self.params = params
        self.folder = (
            OUTPUT_DIR / params.folder
            if params.folder
            else OUTPUT_DIR / user_id / task_id if user_id else OUTPUT_DIR / task_id
        )
        self.status = TaskStatus.PENDING
        self.creation_time = time.time()
        self.start_time = None
        self.end_time = None
        self.result: dict[str, str] = None
        self.task = None
        self.queue_position = None
        self.state = None


# Global state
TASKS: dict[str, Task] = {}
TASK_QUEUE: List[str] = []  # Queue of pending task_ids in order of submission
SEMAPHORE = None
OUTPUT_DIR: Path = None


class WorkflowRequest(BaseModel):
    user_message: str
    reference_code: str
    folder: Optional[str] = None
    user_id: Optional[str] = None
    model: str
    timeout: Optional[float] = None
    run_cuda: bool = False
    run_cuda_perf: bool = False
    benchmark: bool = False
    retry_failed: bool = False
    gpu: Optional[GPUType] = None


class TaskResponse(BaseModel):
    task_id: str
    status: str
    queue_position: Optional[int] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    creation_time: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    folder: str
    queue_position: Optional[int] = None
    state: Optional[dict] = None


class StatusResponse(BaseModel):
    tasks: dict[str, dict]
    running_count: int
    pending_count: int
    task_queue: List[str]


def get_queue_position(task_id: str) -> Optional[int]:
    """Get the position of a task in the queue (0 for running, 1+ for pending)"""
    assert task_id in TASKS, f"Task {task_id} not found"

    task = TASKS[task_id]
    if task.status == TaskStatus.RUNNING:
        return 0
    elif task.status == TaskStatus.PENDING:
        try:
            return TASK_QUEUE.index(task_id) + 1
        except ValueError:
            return None
    else:
        return None


def get_task_status_data(task_id: str) -> dict:
    """Get task status data in a format suitable for API responses."""
    if task_id not in TASKS:
        # Try to load from file for completed tasks
        task = load_task_status(task_id)
        if not task:
            return {"error": f"Task {task_id} not found"}
    else:
        task = TASKS[task_id]

    queue_position = get_queue_position(task_id) if task_id in TASKS else None

    duration = None
    if task.start_time:
        end = task.end_time if task.end_time else time.time()
        duration = end - task.start_time

    state_path = task.folder / "state.json"
    state = (
        load_state_from_json(state_path, read_fp=True) if state_path.exists() else None
    )
    agents_status = {
        "tests": get_agent_status(task.folder, "tests"),
        "cuda": get_agent_status(task.folder, "kgen"),
        "ncu": get_agent_status(task.folder, "ncu"),
        "ncu_annot": get_agent_status(task.folder, "ncu_annot"),
    }

    return {
        "task_id": task_id,
        "status": task.status,
        "queue_position": queue_position,
        "creation_time": task.creation_time,
        "start_time": task.start_time,
        "end_time": task.end_time,
        "duration": duration,
        "state": state,
        "agents": agents_status,
    }


def save_task_status(task: Task):
    """Save task status to a file in the task's folder."""
    task_folder = task.folder
    task_folder.mkdir(parents=True, exist_ok=True)

    # Create a serializable version of the task
    task_data = {
        "task_id": task.task_id,
        "status": task.status,
        "creation_time": task.creation_time,
        "start_time": task.start_time,
        "end_time": task.end_time,
        "result": task.result,
    }

    # Save to status.json in the task's folder
    with open(task_folder / "status.json", "w") as f:
        json.dump(task_data, f)


def load_task_status_from_folder(task_folder: Path) -> Optional[Task]:
    """Load task status from a specific folder path."""
    status_file = task_folder / "status.json"

    if not status_file.exists():
        return None

    try:
        with open(status_file, "r") as f:
            task_data = json.load(f)

        task_id = task_data["task_id"]

        # try extract user_id from task_folder
        user_id = task_folder.parent.name if task_folder.parent.name else None

        # Create a task object from the data
        task = Task(task_id, type("MockParams", (), {"folder": None}), user_id)
        task.folder = task_folder  # Set the folder directly since we know it
        task.status = task_data["status"]
        task.creation_time = task_data["creation_time"]
        task.start_time = task_data["start_time"]
        task.end_time = task_data["end_time"]
        task.result = task_data.get("result", None)
        return task
    except Exception as e:
        logger.error(f"Failed to load task status from folder {task_folder}: {e}")
        return None


def restore_previous_tasks(output_dir: Path):
    """Restore previous tasks from the output directory."""
    if not output_dir.exists():
        logger.info("Output directory does not exist, no previous tasks to restore")
        return

    restored_count = 0

    try:
        for item in output_dir.iterdir():
            if item.is_dir():
                tasks = []
                task = load_task_status_from_folder(item)
                if not task:
                    # this might be a username folder, which we need to check for children folders for tasks
                    for child in item.iterdir():
                        if child.is_dir():
                            task = load_task_status_from_folder(child)
                            tasks.append(task)
                else:
                    tasks.append(task)

                for task in tasks:
                    # For pending/running tasks, update to cancelled
                    if task.status not in [
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                        TaskStatus.CANCELLED,
                        TaskStatus.TIMEOUT,
                    ]:
                        task.status = TaskStatus.CANCELLED
                        task.result = "Task cancelled due to server restart"
                        save_task_status(task)
                        logger.debug(
                            f"Task {task.task_id} cancelled due to server restart"
                        )

                    TASKS[task.task_id] = task
                    restored_count += 1
                    logger.debug(
                        f"Restored task {task.task_id} from {task.folder} with status {task.status}"
                    )
    except Exception as e:
        logger.error(f"Error restoring previous tasks: {e}")

    if restored_count > 0:
        logger.info(f"Restored {restored_count} previous tasks from {output_dir}")
    else:
        logger.info("No previous tasks found to restore")


def load_task_status(task_id: str) -> Optional[Task]:
    """Load task status from the task's folder."""
    if task_id not in TASKS:
        return None

    task_folder = TASKS[task_id].folder
    return load_task_status_from_folder(task_folder)


async def process_task(task_id: str, params: WorkflowRequest):
    task = TASKS[task_id]
    job_logger_id = None
    job_logger = None
    try:
        async with SEMAPHORE:
            # Remove from pending queue when task starts running
            assert task_id in TASK_QUEUE, f"Task {task_id} not in queue"
            assert task.status == TaskStatus.PENDING, f"Task {task_id} is not pending"
            TASK_QUEUE.remove(task_id)

            task_folder = task.folder
            task_folder.mkdir(parents=True, exist_ok=True)

            logger.info(f"Task {task_id} started running: {task_folder}")

            # Update task status
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()

            # create logger for the job
            job_logger = logger.bind(job_id=task_id)
            job_logger_id = job_logger.add(
                task_folder / "run.log",
                level="DEBUG",
                backtrace=True,
                diagnose=True,
                format=config.CUSTOM_LOGGER_FORMAT,
                # task_id refers to the parallel generation thread in kernelblaster
                # so use the job_id to filter the logs here
                filter=lambda record: record["extra"].get("job_id") == task_id,
            )

            workflow_config = WorkflowConfig(
                model=params.model,
                run_cuda=params.run_cuda,
                run_cuda_perf=params.run_cuda_perf,
                run_cuda_bench=params.run_cuda and params.benchmark,
                run_cuda_perf_bench=params.run_cuda_perf and params.benchmark,
                retry_failed=params.retry_failed,
                gpu=params.gpu if params.gpu else GPUType.current(),
            )

            result = await run_workflow(
                task_id,
                params.user_message,
                params.reference_code,
                task_folder,
                workflow_config,
                job_logger,
                params.timeout,
            )

            if result.timeout:
                task.status = TaskStatus.TIMEOUT
                task.result = result.error
                logger.warning(f"Task {task_id} timed out: {task.result}")
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result.generated_codes
                logger.info(f"Task {task_id} completed successfully: {task.result}")

    except asyncio.CancelledError:
        logger.info(f"Task {task_id} was cancelled")
        task.status = TaskStatus.CANCELLED
    except Exception as e:
        logger.error(f"Unhandled error in task {task_id}: {e}")
        logger.error(f"Stacktrace for task {task_id}:\n{traceback.format_exc()}")
        task.status = TaskStatus.FAILED
        task.result = str(e)

    finally:
        task.end_time = time.time()
        if task_id in TASK_QUEUE:
            # Remove from the queue if the task is still remaining in the queue.
            # This can occur if process_task() is cancelled while waiting for the semaphore
            TASK_QUEUE.remove(task_id)
        if job_logger is not None:
            job_logger.remove(job_logger_id)

        # If task is in a terminal state, save status to file
        if task.status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        ]:
            # Save task status to a file in its folder
            save_task_status(task)
            logger.info(
                f"Task {task_id} with status {task.status} completed and saved to disk"
            )


async def create_task(task_id: str, params: WorkflowRequest):
    TASKS[task_id].task = asyncio.create_task(process_task(task_id, params))


@app.post("/submit", response_model=TaskResponse)
async def submit_workflow(request: WorkflowRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task = Task(task_id, request, request.user_id)
    logger.info(f"Task {task_id} created: {task.folder}")
    TASKS[task_id] = task

    # Add to queue of pending tasks
    TASK_QUEUE.append(task_id)

    background_tasks.add_task(create_task, task_id, request)

    task.queue_position = get_queue_position(task_id)
    logger.info(
        f"Task {task_id} status: {task.status}, queue position: {task.queue_position}"
    )

    return TaskResponse(
        task_id=task_id, status=task.status, queue_position=task.queue_position
    )


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """HTTP endpoint to get the status of a task."""
    status_data = get_task_status_data(task_id)
    if "error" in status_data:
        raise HTTPException(status_code=404, detail=status_data["error"])
    return status_data


@app.post("/cancel/{task_id}", response_model=TaskResponse)
async def cancel_task(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = TASKS[task_id]

    if task.status in [TaskStatus.RUNNING, TaskStatus.PENDING]:
        # The task's cancellation method should handle removing itself from the queue
        assert task.task is not None, "Task should be running"
        logger.info(f"Cancelling task {task_id}")
        task.task.cancel()
        await task.task

    task.status = TaskStatus.CANCELLED

    return TaskResponse(task_id=task_id, status=task.status, queue_position=None)


@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API server is running and connected to dependencies."""
    try:
        compile_server_status = "ok" if config.COMPILE_SERVER_URL else "not_connected"
        gpu_server_status = "ok" if config.GPU_SERVER_URL else "not_connected"

        return {
            "status": "ok",
            "dependencies": {
                "compile_server": compile_server_status,
                "gpu_server": gpu_server_status,
            },
            "tasks": {
                "running": sum(
                    1 for t in TASKS.values() if t.status == TaskStatus.RUNNING
                ),
                "pending": len(TASK_QUEUE),
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "message": str(e)}, 500


def run_server(host, port, output_dir, gpu: Optional[GPUType]):
    # Set up logging
    logger.configure(
        handlers=[
            dict(
                sink=sys.stderr,
                format=config.CUSTOM_LOGGER_FORMAT,
                level=config.LOG_LEVEL,
                colorize=True,
                backtrace=True,
                diagnose=True,
            ),
        ],
        extra=dict(agent_name="server", attempt_id=None, task_id=None),
    )

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Restore previous tasks
    restore_previous_tasks(output_dir)

    # Set up resources
    try:
        compile_server = CompileServer(logger, output_dir)
        gpu_server = GPUServer(logger, output_dir, gpu=gpu)
        assert VectorDB.initialize(logger)
        compile_server.wait_for_connection()
        gpu_server.wait_for_connection()
        if compile_server.is_managed:
            config.set_compile_server_url(compile_server.url)
        if gpu_server.is_managed:
            config.set_gpu_server_url(GPUType.current(), gpu_server.url)

        config.print_config(logger)
    except Exception as e:
        logger.error(f"Failed to initialize resources: {e}")
        sys.exit(1)

    # Start the server
    logger.info(
        f"Starting server on {host}:{port} with max concurrency {SEMAPHORE._value}"
    )
    uvicorn.run(app, host=host, port=port)


def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--concurrency", type=int, default=16, help="Maximum concurrent workflows"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/kernelblaster"),
        help="Output directory for compile server and GPU server logs",
    )
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir

    assert OUTPUT_DIR is not None, "Output directory must be specified"

    global SEMAPHORE
    SEMAPHORE = asyncio.Semaphore(args.concurrency)

    # Use None as the GPU type to default start GPU server on host machine with its current GPU type
    run_server(args.host, args.port, args.output_dir, None)


if __name__ == "__main__":
    main()
