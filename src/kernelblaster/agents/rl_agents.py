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
"""
Reinforcement Learning Agents for LLM-based Code Optimization.
Implements PolicyEvaluation, PerfGapAnalysis, and ParameterUpdate agents.
"""
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import statistics

from .database import OptimizationDatabase, OptimizationEntry, CompositeOptimization
from .utils import generate_code_retry, LLMResponse
from ..config import config


@dataclass
class TrajectoryStep:
    """Represents a single step in an optimization trajectory.

    ``cycles`` retains its historical name for on-disk / JSON stability
    but is now ``float`` — Step 4 canonicalisation. Both backends store
    their primary metric here in the backend's native unit (CUDA cycles,
    OpenCL ms). Previously OpenCL scaled ms → int µs to fit an ``int``
    field; that shape-conversion (``Backend.metric_to_traj_cycles`` /
    ``metric_from_traj_cycles``) is now identity and the hooks are
    deleted.
    """
    state: str
    action: str  # optimization technique
    code: str
    cycles: float
    predicted_improvement: float
    actual_improvement: float
    reward: float


@dataclass
class Trajectory:
    """Represents a complete optimization trajectory."""
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0
    initial_cycles: float = 0.0
    final_cycles: float = 0.0
    
    def add_step(self, step: TrajectoryStep):
        self.steps.append(step)
        self.total_reward += step.reward
        if len(self.steps) == 1:
            self.initial_cycles = step.cycles
        self.final_cycles = step.cycles


class ReplayBuffer:
    """Stores trajectories for policy learning."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.trajectories: List[Trajectory] = []
    
    def add_trajectory(self, trajectory: Trajectory):
        """Add a trajectory to the buffer."""
        self.trajectories.append(trajectory)
        if len(self.trajectories) > self.max_size:
            # Remove oldest trajectory
            self.trajectories.pop(0)
    
    def get_recent_trajectories(self, n: int = None) -> List[Trajectory]:
        """Get the most recent n trajectories."""
        if n is None:
            return self.trajectories
        return self.trajectories[-n:]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics about the trajectories in the buffer."""
        if not self.trajectories:
            return {}
        
        rewards = [t.total_reward for t in self.trajectories]
        improvements = [(t.initial_cycles - t.final_cycles) / t.initial_cycles * 100 
                       for t in self.trajectories if t.initial_cycles > 0]
        
        return {
            'num_trajectories': len(self.trajectories),
            'avg_reward': statistics.mean(rewards),
            'std_reward': statistics.stdev(rewards) if len(rewards) > 1 else 0,
            'max_reward': max(rewards),
            'min_reward': min(rewards),
            'avg_improvement': statistics.mean(improvements) if improvements else 0,
            'success_rate': sum(1 for r in rewards if r > 0) / len(rewards)
        }



# PolicyEvaluationAgent / PerfGapAnalysisAgent / ParameterUpdateAgent
# deleted in the Step-3 cleanup — the CUDA agent's policy_update_cycle
# was their only consumer and was itself dead after Phase 4f. Cut ~400
# lines of unused RL-analysis scaffolding.
