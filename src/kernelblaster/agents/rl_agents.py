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
    """Represents a single step in an optimization trajectory."""
    state: str
    action: str  # optimization technique
    code: str
    cycles: int
    predicted_improvement: float
    actual_improvement: float
    reward: float


@dataclass
class Trajectory:
    """Represents a complete optimization trajectory."""
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0
    initial_cycles: int = 0
    final_cycles: int = 0
    
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


class PolicyEvaluationAgent:
    """Agent that evaluates policy performance by comparing predicted vs actual results."""
    
    def __init__(self, model: str = None):
        self.model = model or config.MODEL
        self.system_prompt = """You are a performance analysis expert specializing in CUDA optimization evaluation.

Your task is to analyze the performance discrepancies between predicted and actual optimization results.

Given a set of optimization attempts with their predicted improvements, actual results, and profiling data, 
you should:

1. Identify patterns in prediction accuracy
2. Summarize key performance discrepancies 
3. Highlight successful optimization strategies
4. Note any systematic biases in predictions

Focus on actionable insights that can improve future optimization predictions."""

    async def evaluate_policy(self, replay_buffer: ReplayBuffer, database: OptimizationDatabase) -> str:
        """Evaluate policy performance and return analysis in natural language."""
        
        recent_trajectories = replay_buffer.get_recent_trajectories(10)  # Analyze last 10 trajectories
        if not recent_trajectories:
            return "No trajectories available for evaluation."
        
        # Collect performance data
        performance_data = []
        for traj in recent_trajectories:
            for step in traj.steps:
                performance_data.append({
                    'state': step.state,
                    'technique': step.action,
                    'predicted_improvement': step.predicted_improvement,
                    'actual_improvement': step.actual_improvement,
                    'cycles': step.cycles,
                    'reward': step.reward
                })
        
        # Create evaluation prompt
        prompt = f"""Analyze the following optimization performance data:

RECENT OPTIMIZATION ATTEMPTS:
{json.dumps(performance_data, indent=2)}

BUFFER STATISTICS:
{json.dumps(replay_buffer.get_statistics(), indent=2)}

DATABASE STATISTICS:
{json.dumps(database.get_database_stats(), indent=2)}

Please provide a concise analysis focusing on:
1. Overall prediction accuracy trends
2. Which optimization techniques are over/under-performing
3. Patterns in successful vs failed optimizations
4. Recommendations for improving the optimization strategy database

Keep your response focused and actionable."""

        try:
            # Import logger for proper logging
            from loguru import logger
            response = await generate_code_retry(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                logger=logger,
                max_retries=3
            )
            return response.generations[0]
        except Exception as e:
            return f"Error in policy evaluation: {str(e)}"


class PerfGapAnalysisAgent:
    """Agent that analyzes performance gaps and identifies why predictions differed from reality."""
    
    def __init__(self, model: str = None):
        self.model = model or config.MODEL
        self.system_prompt = """You are a CUDA performance analysis expert specializing in understanding optimization failures and successes.

Your task is to analyze why optimization predictions differed from actual results and identify the root causes.

When analyzing performance gaps, consider:
1. Hardware-specific factors (memory bandwidth, compute units, cache behavior)
2. Code characteristics (memory access patterns, control flow, data dependencies)
3. Optimization technique assumptions vs reality
4. Profiling metric interpretation accuracy

Provide specific, technical insights about why certain optimizations succeeded or failed."""

    async def analyze_performance_gaps(self, evaluation_result: str, recent_failures: List[TrajectoryStep]) -> str:
        """Analyze performance gaps and provide insights on prediction errors."""
        
        failure_analysis = []
        for step in recent_failures:
            gap = step.predicted_improvement - step.actual_improvement
            failure_analysis.append({
                'state': step.state,
                'technique': step.action,
                'predicted_improvement': step.predicted_improvement,
                'actual_improvement': step.actual_improvement,
                'performance_gap': gap,
                'cycles': step.cycles
            })
        
        prompt = f"""POLICY EVALUATION RESULTS:
{evaluation_result}

DETAILED FAILURE ANALYSIS:
{json.dumps(failure_analysis, indent=2)}

Based on the policy evaluation and specific failure cases, analyze:

1. ROOT CAUSES: Why did these optimizations fail to meet predictions?
   - Were the assumptions about bottlenecks incorrect?
   - Did hardware characteristics differ from expectations?
   - Were there unexpected interactions between optimizations?

2. SYSTEMATIC ISSUES: Are there patterns in the prediction errors?
   - Which types of optimizations consistently over/under-perform?
   - Which code states are hardest to analyze correctly?

3. SPECIFIC CORRECTIONS: What specific changes should be made to improve predictions?
   - Adjustment factors for certain optimization types
   - New metrics to consider for state classification
   - Refined prediction models for specific scenarios

Provide concrete, actionable recommendations for database improvements."""

        try:
            # Import logger for proper logging
            from loguru import logger
            response = await generate_code_retry(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                logger=logger,
                max_retries=3
            )
            return response.generations[0]
        except Exception as e:
            return f"Error in performance gap analysis: {str(e)}"


class ParameterUpdateAgent:
    """Agent that updates the optimization database based on analysis results."""
    
    def __init__(self, model: str = None):
        self.model = model or config.MODEL
        self.system_prompt = """You are a creative database management expert for CUDA optimization strategies.

Your task is to update the optimization strategy database based on performance analysis results, and creatively discover new optimization approaches.

You should:
1. Adjust predicted performance values based on actual results
2. Update confidence scores for existing optimizations
3. Add new optimization strategies discovered through analysis
4. Create composite optimizations that combine multiple techniques in specific orders
5. Design parameter-tuned versions of existing optimizations
6. Identify new performance states that don't fit existing categories
7. Remove or deprecate consistently poor-performing strategies

CREATIVE CAPABILITIES:
- Combine 2-3 optimization techniques in specific orders
- Fine-tune parameters like unrolling factors, tile sizes, block dimensions
- Discover new performance states from unusual metric patterns
- Create adaptive optimizations that change based on problem characteristics
- Identify side effects and trade-offs between optimization techniques

Output your recommendations as structured JSON updates that can be applied to the database."""

    async def update_parameters(self, gap_analysis: str, database: OptimizationDatabase) -> Dict[str, Any]:
        """Update database parameters based on gap analysis."""
        
        current_stats = database.get_database_stats()
        
        # Get poorly performing optimizations
        poor_performers = []
        for state, optimizations in database.optimization_strategies.items():
            for opt in optimizations:
                if (opt.actual_improvement is not None and 
                    opt.predicted_improvement > 0 and
                    opt.actual_improvement < opt.predicted_improvement * 0.5):
                    poor_performers.append({
                        'state': state,
                        'technique': opt.technique,
                        'predicted': opt.predicted_improvement,
                        'actual': opt.actual_improvement,
                        'confidence': opt.confidence_score
                    })
        
        prompt = f"""PERFORMANCE GAP ANALYSIS:
{gap_analysis}

CURRENT DATABASE STATISTICS:
{json.dumps(current_stats, indent=2)}

POOR PERFORMING OPTIMIZATIONS:
{json.dumps(poor_performers, indent=2)}

Based on this analysis, provide comprehensive database update recommendations in the following JSON format:

{{
  "prediction_adjustments": [
    {{
      "state": "state_name",
      "technique1": "technique_name_1",
      "technique2": "technique_name_2", 
      "technique3": "technique_name_3",
      "order_of_techniques": ["1. technique_name_1", "2. technique_name_2", "3. technique_name_3"],
      "parameters_to_fine_tune": {{
        "unrolling_factor": 4,
        "tile_size": 32,
        "block_size": 256
      }},
      "new_predicted_improvement": 35.0,
      "reason": "explanation for this composite optimization",
      "side_effects": "potential negative effects or trade-offs to be aware of"
    }}
  ],
  "confidence_updates": [
    {{
      "state": "state_name",
      "technique": "technique_name",
      "new_confidence": 0.8,
      "reason": "explanation for confidence change"
    }}
  ],
  "new_optimizations": [
    {{
      "state": "state_name",
      "technique": "new_technique_name",
      "predicted_improvement": 20.0,
      "reason": "why this optimization should be added"
    }}
  ],
  "parameter_tuned_optimizations": [
    {{
      "base_technique": "loop_unrolling",
      "parameters": {{
        "unrolling_factor": 8,
        "vectorization": true
      }},
      "predicted_improvement": 25.0,
      "reason": "explanation for parameter choice",
      "applicable_states": ["compute_bound", "latency_bound"]
    }}
  ],
  "discovered_states": [
    {{
      "state_name": "new_state_pattern",
      "description": "description of performance characteristics",
      "characteristics": "metric ranges that define this state",
      "initial_optimizations": [
        {{
          "technique": "suggested_technique",
          "predicted_improvement": 30.0
        }}
      ]
    }}
  ],
  "deprecated_optimizations": [
    {{
      "state": "state_name", 
      "technique": "technique_name",
      "reason": "why this should be deprecated"
    }}
  ]
}}

CREATIVE THINKING GUIDELINES:
1. Look for patterns where combining techniques might yield better results
2. Consider parameter fine-tuning for techniques that partially worked
3. Identify new performance states from unusual metric combinations
4. Think about side effects and trade-offs between optimizations
5. Suggest adaptive approaches that change based on problem characteristics

Focus on innovative, data-driven updates that will improve future optimization performance."""

        try:
            # Import logger for proper logging
            from loguru import logger
            response = await generate_code_retry(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                logger=logger,
                max_retries=3
            )
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response.generations[0], re.DOTALL)
            if json_match:
                updates = json.loads(json_match.group())
                
                # Apply updates to database
                self._apply_database_updates(database, updates)
                return updates
            else:
                return {"error": "Could not parse update recommendations"}
                
        except Exception as e:
            return {"error": f"Error in parameter update: {str(e)}"}
    
    def _apply_database_updates(self, database: OptimizationDatabase, updates: Dict[str, Any]):
        """Apply the recommended updates to the database."""
        
        # Apply composite optimization predictions (create new composite optimizations)
        for adj in updates.get("prediction_adjustments", []):
            if "technique1" in adj:  # This is a composite optimization
                composite = CompositeOptimization(
                    state=adj["state"],
                    technique1=adj["technique1"],
                    technique2=adj.get("technique2"),
                    technique3=adj.get("technique3"),
                    order_of_techniques=adj.get("order_of_techniques", []),
                    parameters_to_fine_tune=adj.get("parameters_to_fine_tune", {}),
                    predicted_improvement=adj["new_predicted_improvement"],
                    reason=adj.get("reason", ""),
                    side_effects=adj.get("side_effects", "")
                )
                database.add_composite_optimization(composite)
            else:  # Traditional single technique adjustment
                state = adj["state"]
                technique = adj.get("technique", "")
                new_prediction = adj["new_predicted_improvement"]
                
                optimizations = database.get_optimizations_for_state(state)
                for opt in optimizations:
                    if opt.technique == technique:
                        opt.predicted_improvement = new_prediction
                        opt.last_updated = datetime.now().isoformat()
                        break
        
        # Apply confidence updates
        for conf in updates.get("confidence_updates", []):
            state = conf["state"]
            technique = conf["technique"]
            new_confidence = conf["new_confidence"]
            
            optimizations = database.get_optimizations_for_state(state)
            for opt in optimizations:
                if opt.technique == technique:
                    opt.confidence_score = new_confidence
                    opt.last_updated = datetime.now().isoformat()
                    break
        
        # Add new optimizations
        for new_opt in updates.get("new_optimizations", []):
            database.add_new_optimization(
                new_opt["state"],
                new_opt["technique"], 
                new_opt["predicted_improvement"]
            )
        
        # Add parameter-tuned optimizations
        for param_opt in updates.get("parameter_tuned_optimizations", []):
            new_technique = database.create_parameter_tuned_optimization(
                param_opt["base_technique"],
                param_opt["parameters"],
                param_opt["predicted_improvement"],
                param_opt.get("reason", "")
            )
            
            # Add to all applicable states
            for state in param_opt.get("applicable_states", []):
                database.add_new_optimization(
                    state,
                    new_technique,
                    param_opt["predicted_improvement"]
                )
        
        # Add discovered states
        for discovered in updates.get("discovered_states", []):
            state_name = discovered["state_name"]
            database.discovered_states[state_name] = {
                "description": discovered.get("description", ""),
                "characteristics": discovered.get("characteristics", ""),
                "discovery_context": "AI-discovered state"
            }
            
            # Add initial optimizations for the new state
            for opt in discovered.get("initial_optimizations", []):
                database.add_new_optimization(
                    state_name,
                    opt["technique"],
                    opt["predicted_improvement"]
                )
        
        # Mark deprecated optimizations (reduce confidence significantly)
        for dep in updates.get("deprecated_optimizations", []):
            state = dep["state"]
            technique = dep["technique"]
            
            optimizations = database.get_optimizations_for_state(state)
            for opt in optimizations:
                if opt.technique == technique:
                    opt.confidence_score = 0.1  # Very low confidence
                    opt.last_updated = datetime.now().isoformat()
                    break 