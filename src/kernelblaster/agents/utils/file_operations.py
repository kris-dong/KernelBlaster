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
import json
from pathlib import Path
import dataclasses


def write_code_to_file(code, filename, logger=None):
    Path(filename).write_text(code)
    if logger:
        logger.info(f"Code written to {filename}")


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        elif isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_jsonl(filepath: str, lines: list):
    filepath = Path(filepath)
    filepath.write_text(
        "\n".join(json.dumps(d, cls=EnhancedJSONEncoder) for d in lines)
    )


def read_jsonl(filepath: str) -> list[dict]:
    filepath = Path(filepath)
    return [json.loads(line) for line in filepath.read_text().splitlines()]


def get_agent_status(output_dir: Path, agent_name: str) -> dict:
    metrics_path = output_dir / agent_name / "metrics.jsonl"
    if not metrics_path.exists():
        return {
            "iteration": 0,
            "success": False,
        }
    metrics_data = read_jsonl(metrics_path)
    max_iteration = max([x["attempt_id"] for x in metrics_data])
    success = any(metrics["success"] for metrics in metrics_data)
    details = []
    for metrics in metrics_data:
        detail = {
            "attempt_id": metrics["attempt_id"],
            "thread_id": metrics["thread_id"],
            "success": metrics["success"],
            "durations": metrics["durations"],
            "code": metrics["contents"],
            "feedback": metrics["feedback"],
        }
        if agent_name == "ncu":
            detail["ncu_metrics"] = {
                k: v
                for k, v in {
                    "elapsed_cycles": metrics.get("elapsed_cycles", None),
                    "ncu_log": metrics.get("ncu_log", None),
                    "init_cycles": metrics.get("init_cycles", None),
                    "best_cycles": metrics.get("best_cycles", None),
                    "num_improvements": metrics.get("num_improvements", None),
                    "is_faster": metrics.get("is_faster", None),
                }.items()
                if v is not None
            }
        details.append(detail)
    return {
        "iteration": max_iteration,
        "success": success,
        "details": details,
    }
