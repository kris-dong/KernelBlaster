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
from typing import Optional, TypedDict, Any, Dict
from pathlib import Path
import json

from ..config import GPUType


class GraphState(TypedDict):
    model: str  # model to use for generation
    gpu: GPUType  # GPU type to use for generation

    run_cuda: bool  # whether to run the CUDA generation agent
    run_cuda_perf: bool  # whether to run the performance agents
    run_cuda_bench: bool  # whether to run the CUDA benchmark agent
    run_cuda_perf_bench: bool  # whether to run the CUDA benchmark agent

    retry_failed: bool  # whether to retry failed agents
    reference_code: str  # reference code
    user_message: str  # user message
    folder: Path  # folder to save the generated code
    logger: Any  # logger for the problem
    
    # RL optimization parameters
    rl_iterations: int  # number of RL iterations to run
    rl_rollout_steps: int  # number of rollout steps per RL iteration
    rl_buffer_size: int  # size of RL replay buffer
    rl_update_frequency: int  # frequency of RL database updates

    filepath: str  # filename of the most recently generated file in the graph
    test_code_fp: Path  # Path to the generated test code for CUDA
    cuda_fp: Path  # Path to the generated CUDA kernel code
    cuda_bench_fp: Path  # Path to the generated CUDA benchmark code
    ncu_cuda_fp: Path  # Path to the optimized CUDA code based on NCU profiling
    ncu_cuda_bench_fp: (
        Path  # Path to the optimized CUDA code based on NCU profiling and benchmarking
    )
    rl_ncu_cuda_fp: Path  # Path to the RL-optimized CUDA code


def save_state_to_json(state: GraphState, output_path: str) -> None:
    """
    Serialize and write GraphState to a JSON file, handling non-serializable fields.

    Args:
        state: The GraphState to serialize
        output_path: Path to the output JSON file
    """
    # Create a serializable copy of the state dictionary
    serializable_state: Dict[str, Any] = {}

    # Fields to ignore during comparison (like logger which isn't serializable)
    ignore_fields = {"logger", "shared_optimization_database"}

    for key, value in state.items():
        # Skip logger as it's not serializable
        if key in ignore_fields:
            continue

        # Convert Path objects to strings
        if isinstance(value, Path):
            serializable_state[key] = str(value.resolve())
        else:
            # Include all other serializable values
            serializable_state[key] = value

    # Write to JSON file
    try:
        with open(output_path, "w") as f:
            json.dump(serializable_state, f, indent=2)
    except Exception as e:
        print(f"Error saving state to {output_path}: {e}")


def load_state_from_json(json_path: str, read_fp: bool = False) -> Dict[str, Any]:
    """
    Load a state dictionary from a JSON file and parse file pointers.

    Fields ending with '_fp' are treated as file paths, and their contents
    are loaded into fields without the '_fp' suffix.

    Args:
        json_path: Path to the JSON file
        read_fp: If True, read the file contents into fields without the '_fp' suffix
    Returns:
        Dictionary containing the loaded state with file contents

    Example:
        If JSON contains {"cuda_fp": "path/to/file.txt"},
        the result will include:
        {"cuda_fp": "path/to/file.txt", "cuda": "<file contents>"}
    """
    try:
        # Load the JSON file
        with open(json_path, "r") as f:
            state_dict = json.load(f)

        # Process fields ending with _fp
        fp_fields = [key for key in state_dict.keys() if key.endswith("_fp")]

        if read_fp:
            for fp_field in fp_fields:
                file_path = state_dict[fp_field]
                content_field = fp_field[:-3]  # Remove '_fp' suffix

                # Skip if the path is empty or None
                if not file_path:
                    continue

                try:
                    # Attempt to read the file content
                    with open(file_path, "r") as file:
                        state_dict[content_field] = file.read()
                except Exception as e:
                    print(f"Warning: Could not read file at {file_path}: {e}")
                    # Keep the field with None value to indicate attempted but failed loading
                    state_dict[content_field] = None

        return state_dict

    except Exception as e:
        print(f"Error loading state from {json_path}: {e}")
        return {}


def compare_states(state1: Optional[GraphState], state2: Optional[GraphState]) -> bool:
    """
    Compare two GraphState dictionaries.

    Args:
        state1: The first GraphState dictionary
        state2: The second GraphState dictionary

    Returns:
        True if the states are equal, False otherwise
    """
    # Handle None cases
    if state1 is None and state2 is None:
        return True
    if state1 is None or state2 is None:
        return False

    # Fields to ignore during comparison (like logger which isn't serializable)
    ignore_fields = {"logger"}

    # Compare all fields except those in ignore_fields
    for key in set(state1.keys()) | set(state2.keys()):
        # Skip ignored fields
        if key in ignore_fields:
            continue

        # Check if the key exists in both states
        if key not in state1 or key not in state2:
            return False

        value1, value2 = state1[key], state2[key]

        # Handle Path objects by converting to strings for comparison
        if isinstance(value1, Path) and isinstance(value2, Path):
            if str(value1.resolve()) != str(value2.resolve()):
                return False
        # Handle case where one is Path and other is not
        elif isinstance(value1, Path) or isinstance(value2, Path):
            return False
        # Compare other values directly
        elif value1 != value2:
            return False

    return True
