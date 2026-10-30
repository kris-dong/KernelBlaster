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
import os
from langgraph.graph import StateGraph, START, END

from .nodes import *
from .state import GraphState
from ..config import config


def route_kernel_generation(state: GraphState):
    routes = []
    if state["run_cuda"]:
        routes.append("Generate CUDA")
    if len(routes) > 0:
        return routes
    return "Skip"


def route_baseline_or_generation(state: GraphState):
    """
    Route to either baseline-driven optimization or traditional code generation.
    """
    # Check if we should use baseline-driven optimization
    rl_enabled = config.EXPERIMENTAL_FEATURES.OPT_RL_NCU
    
    # Check for RL parameters in state as indicator of baseline optimization mode
    has_rl_params = all(key in state for key in ['rl_iterations', 'rl_rollout_steps', 'rl_buffer_size'])
    use_baseline = state.get("use_baseline_optimization", False) or has_rl_params
    
    # Enhanced debugging
    logger = state.get("logger")
    if logger:
        logger.info(f"=== ROUTING DEBUG START ===")
        logger.info(f"RL_enabled: {rl_enabled}")
        logger.info(f"has_rl_params: {has_rl_params}")
        logger.info(f"use_baseline: {use_baseline}")
        logger.info(f"user_message: {state.get('user_message', 'N/A')}")
        logger.info(f"folder: {state.get('folder', 'N/A')}")
        logger.info(f"State keys: {list(state.keys())}")
        logger.info(f"run_cuda: {state.get('run_cuda', 'N/A')}")
        logger.info(f"run_cuda_perf: {state.get('run_cuda_perf', 'N/A')}")
    else:
        print(f"DEBUG ROUTING: RL_enabled={rl_enabled}, use_baseline={use_baseline}, has_rl_params={has_rl_params}")
        print(f"DEBUG ROUTING: user_message={state.get('user_message', 'N/A')}")
        print(f"DEBUG ROUTING: folder={state.get('folder', 'N/A')}")
        print(f"DEBUG ROUTING: state keys = {list(state.keys())}")
    
    if rl_enabled and use_baseline:
        # Check if we can actually use baseline optimization
        from ..utils.baseline_selector import find_best_baseline_for_problem
        import re
        
        # Try to extract problem number from user message or folder path
        problem_number = None
        user_message = state.get("user_message", "")
        
        # Extract problem number (FIXED LOGIC - more specific patterns)
        if "level" in user_message.lower():
            match = re.search(r"level\d+/(\d{3})_", user_message)  # Look for 3-digit pattern
            if match:
                problem_number = str(int(match.group(1)))
                if logger:
                    logger.info(f"Extracted problem_number from level pattern: {problem_number}")
        
        if not problem_number:
            match = re.search(r"task\s*(\d+)", user_message, re.IGNORECASE)
            if match:
                problem_number = match.group(1)
                if logger:
                    logger.info(f"Extracted problem_number from task pattern: {problem_number}")
        
        if not problem_number:
            from pathlib import Path
            base_folder = Path(state.get("folder", ""))
            folder_parts = str(base_folder).split("/")
            if logger:
                logger.info(f"Folder parts for extraction: {folder_parts}")
            for part in folder_parts:
                # FIXED: Look specifically for task pattern (3 digits followed by underscore and letters)
                match = re.search(r"(\d{3})_[A-Za-z]", part)
                if match:
                    problem_number = str(int(match.group(1)))
                    if logger:
                        logger.info(f"Extracted problem_number from folder part '{part}': {problem_number}")
                    break
        
        if logger:
            logger.info(f"Final problem_number: {problem_number}")
        
        # Try to find baseline
        baseline_file = None
        if problem_number:
            if logger:
                logger.info(f"Searching for baseline for problem {problem_number}")
            else:
                print(f"DEBUG ROUTING: Checking baseline for problem {problem_number}")
            baseline_file = find_best_baseline_for_problem(
                problem_number=problem_number,
                precision=state.get("precision", "fp16"),
                logger=logger
            )
            if logger:
                logger.info(f"Baseline search result: {baseline_file}")
        
        if baseline_file:
            if logger:
                logger.info(f"Found baseline {baseline_file}, routing to Baseline RL Optimization")
                logger.info(f"=== ROUTING DEBUG END: Going to Baseline RL Optimization ===")
            else:
                print(f"DEBUG ROUTING: Found baseline {baseline_file}, going to Baseline RL Optimization")
            return "Baseline RL Optimization"
        else:
            if logger:
                logger.info(f"No baseline found, checking for existing generated code")
            else:
                print(f"DEBUG ROUTING: No baseline found, checking for existing generated code")
            # No baseline found, check if we have existing generated code for RL optimization
            kgen_cuda_fp = state.get("kgen_cuda_fp")
            if logger:
                logger.info(f"kgen_cuda_fp in state: {kgen_cuda_fp}")
            
            if kgen_cuda_fp and os.path.exists(kgen_cuda_fp):
                if logger:
                    logger.info(f"Found existing generated code at {kgen_cuda_fp}, going to RL optimization")
                    logger.info(f"=== ROUTING DEBUG END: Going to RL optimization with existing code ===")
                else:
                    print(f"DEBUG ROUTING: Found existing generated code, going to RL optimization")
                return "Baseline RL Optimization"  # Use same node but it will handle original approach
            else:
                if logger:
                    logger.info(f"No baseline and no generated code, routing to traditional generation")
                    # FIXED: Ensure run_cuda is enabled for traditional generation
                    state["run_cuda"] = True
                    logger.info(f"Set run_cuda=True for traditional generation")
                    kernel_routes = route_kernel_generation(state)
                    logger.info(f"Kernel generation routes: {kernel_routes}")
                    logger.info(f"=== ROUTING DEBUG END: Going to traditional generation ===")
                else:
                    print(f"DEBUG ROUTING: No baseline and no generated code, routing to traditional generation")
                    state["run_cuda"] = True
                # Need to generate code first
                return route_kernel_generation(state)
    else:
        if logger:
            logger.info(f"RL not enabled or no baseline mode, going to traditional generation")
            # FIXED: Ensure run_cuda is enabled for traditional generation
            state["run_cuda"] = True  
            logger.info(f"Set run_cuda=True for traditional generation")
            kernel_routes = route_kernel_generation(state)
            logger.info(f"Kernel generation routes: {kernel_routes}")
            logger.info(f"=== ROUTING DEBUG END: Going to traditional generation ===")
        else:
            print("DEBUG ROUTING: Going to traditional generation")
            state["run_cuda"] = True
        # Traditional workflow - route to code generation
        return route_kernel_generation(state)



def route_ncu_benchmark(state: GraphState):
    routes = []
    if state["run_cuda_perf_bench"]:
        routes.append("Generate NCU CUDA Benchmark")
    if len(routes) > 0:
        return routes
    return "Skip"


def route_original_cuda_kernel(state: GraphState):
    routes = []
    
    # Check if RL optimization should be used after generation
    rl_enabled = config.EXPERIMENTAL_FEATURES.OPT_RL_NCU
    has_rl_params = all(key in state for key in ['rl_iterations', 'rl_rollout_steps', 'rl_buffer_size'])
    use_rl_optimization = rl_enabled and has_rl_params
    
    if use_rl_optimization:
        routes.append("RL Optimization")
    elif state["run_cuda_perf"]:
        routes.append("Perf Optimization")
    
    if state["run_cuda_bench"]:
        routes.append("Generate CUDA Benchmark")
    
    if len(routes) > 0:
        return routes
    return "Skip"


def build_graph():
    graph_builder = StateGraph(GraphState)
    
    # Baseline-driven RL optimization node
    graph_builder.add_node("Baseline RL Optimization", optimization_rl_ncu)

    # Workflow routing
    graph_builder.add_edge(START, "Baseline RL Optimization")

    # Baseline optimization ends workflow
    graph_builder.add_edge("Baseline RL Optimization", END)

    return graph_builder.compile()
