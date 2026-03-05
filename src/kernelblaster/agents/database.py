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
Enhanced Database utility for GPU optimization with LLM-powered qualitative state analysis.

This module implements a two-LLM agent system:
1. State Summarizer Agent: Analyzes NCU reports qualitatively
2. State Matcher Agent: Matches current state against known optimization patterns
"""
from __future__ import annotations
from pathlib import Path
import shutil
import re
import json
import itertools  # Added to support fallback logic using itertools.chain
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import threading

def get_elapsed_cycles_v2(text: str) -> int:
    groups = re.search(r"Elapsed Cycles: (\d+)", text)
    if groups is None:
        raise ValueError("No elapsed cycles found in text")
    return int(groups.group(1))

def get_speedup_from_files(soln_file: Path) -> Tuple[int, int, float]:
    final_text = soln_file.read_text()
    if (soln_file.parent /"ncu/0_init_ncu_log.txt").exists():
        initial_text = (soln_file.parent /"ncu/0_init_ncu_log.txt").read_text()
    else:
        initial_text = (soln_file.parent /"ncu_annot/init.cu").read_text()
    final_elapsed_cycles = get_elapsed_cycles_v2(final_text)
    initial_elapsed_cycles = get_elapsed_cycles_v2(initial_text)
    speedup_ratio = initial_elapsed_cycles / final_elapsed_cycles
    return initial_elapsed_cycles, final_elapsed_cycles, speedup_ratio

class LLMInterface:
    """Interface for LLM queries used in state analysis."""
    
    def __init__(self, model_name: str = None, logger = None):
        from ..config import config
        self.model_name = model_name or config.MODEL
        self.logger = logger
    
    async def query(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """Send a query to the LLM and return the response."""
        try:
            from .utils import generate_code_retry
        except ImportError:
            if self.logger:
                self.logger.error("Failed to import generate_code_retry from utils")
            return "Error: Could not import LLM utilities"
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await generate_code_retry(
                messages, 
                self.model_name, 
                self.logger, 
                n_tasks=1,
                max_retries=3
            )
            return response.generations[0] if response.generations else ""
        except Exception as e:
            if self.logger:
                self.logger.error(f"LLM query failed: {e}")
            return f"Error: {str(e)}"
    
    def query_sync(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """Synchronous wrapper for LLM queries."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return self._mock_response()
            else:
                return loop.run_until_complete(self.query(prompt, max_tokens, temperature))
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Async query failed, using mock response: {e}")
            return self._mock_response()
    
    def _mock_response(self) -> str:
        """Fallback mock response for when LLM is not available."""
        return """
        PRIMARY_BOTTLENECK: memory_bound
        SECONDARY_CHARACTERISTICS:
        - Inefficient memory access patterns
        - Low cache utilization
        - Moderate occupancy
        PERFORMANCE_SIGNATURE: Memory-intensive workload with room for optimization
        """
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        # This should be a lightweight availability check. We intentionally avoid
        # making network calls here; we only check for configured credentials.
        try:
            from ..config import config
        except Exception:
            config = None  # type: ignore

        import os

        # Prefer explicit config value if present.
        if config is not None and bool(getattr(config, "API_KEY", None)):
            return True

        # Common env vars used across our supported LLM backends.
        # Note: keep this in sync with the client selection logic in utils/query.py.
        if os.getenv("OAI_ATLAS_KEY") or os.getenv("OPENAI_API_KEY"):
            return True
        if os.getenv("NIM_KEY") or os.getenv("CHIPNEMO_KEY") or os.getenv("NGC_KEY"):
            return True
        if os.getenv("AZURE_ENDPOINT") and os.getenv("AZURE_KEY"):
            return True
        if os.getenv("EOS_BASE_URL"):
            return True

        # LLM gateway-style credentials (if used).
        if os.getenv("LLM_GATEWAY_URL") and (os.getenv("LLM_GATEWAY_KEY") or os.getenv("LLM_GATEWAY_TOKEN")):
            return True

        return False


@dataclass
class StateProfile:
    """Qualitative state profile with primary and secondary characteristics."""
    state_name: str
    primary_bottleneck: str  # memory_bound, compute_bound, latency_bound, hybrid_bound
    secondary_characteristics: List[str]
    performance_signature: str
    context_description: str
    relative_patterns: Dict[str, str]  # Qualitative patterns instead of numerical values

@dataclass
class OptimizationEntry:
    technique: str
    predicted_improvement: Optional[float] = None
    description: str = ""
    category: str = ""  # memory, compute, latency, etc.
    actual_improvement: Optional[float] = None
    confidence_score: float = 0.5
    last_updated: Optional[str] = None
    usage_count: int = 0
    # Speedup tracking fields
    predicted_speedup: float = 1.0  # Expected speedup (ratio)
    actual_speedup: Optional[float] = None  # Most recent speedup measurement
    initial_elapsed_cycles: Optional[int] = None  # Baseline elapsed cycles


@dataclass
class CompositeOptimization:
    """Represents a composite optimization with multiple techniques."""
    state: str
    technique1: str
    technique2: Optional[str] = None
    technique3: Optional[str] = None
    order_of_techniques: List[str] = field(default_factory=list)
    parameters_to_fine_tune: Dict[str, Any] = field(default_factory=dict)
    predicted_improvement: float = 0.0
    actual_improvement: Optional[float] = None
    reason: str = ""
    side_effects: str = ""
    confidence_score: float = 0.5
    last_updated: Optional[str] = None
    usage_count: int = 0
    
    def get_composite_id(self) -> str:
        """Generate a unique ID for this composite optimization."""
        techniques = [t for t in [self.technique1, self.technique2, self.technique3] if t]
        params_str = "_".join(f"{k}_{v}" for k, v in self.parameters_to_fine_tune.items())
        return f"composite_{'+'.join(techniques)}_{params_str}"


class GPUOptimizationDatabase:
    """Enhanced database with LLM-powered qualitative state analysis."""
    
    def __init__(self, optimization_db_path: Path, gpu_report_path: Path, llm_interface=None):
        import os
        self.optimization_db_path = optimization_db_path
        self.optimization_db_header_path = optimization_db_path.with_name(f"{optimization_db_path.stem}_header{optimization_db_path.suffix}") 
        self.optimization_db_footer_path = optimization_db_path.with_name(f"{optimization_db_path.stem}_footer{optimization_db_path.suffix}") 
        self.gpu_report_path = gpu_report_path
        self.llm_interface = llm_interface or LLMInterface()

        # Log env-driven behaviour once at startup so runs are easy to audit from run.log.
        # Note: this flag only affects the database *fallback* chooser, not LLM plan selection.
        try:
            raw_val = os.getenv("KERNELAGENT_DB_FALLBACK_TOP1", None)
            parsed_val = os.getenv("KERNELAGENT_DB_FALLBACK_TOP1", "0") in (
                "1",
                "true",
                "True",
                "yes",
                "YES",
                "y",
                "on",
                "ON",
            )
            msg = f"KERNELAGENT_DB_FALLBACK_TOP1={raw_val!r} (parsed={parsed_val})"
            if hasattr(self.llm_interface, "logger") and self.llm_interface.logger:
                self.llm_interface.logger.info(msg)
            else:
                print(msg)
        except Exception:
            # Never let env logging break DB init.
            pass

        # ---------- LLM interaction logging ----------
        # Prompts and outputs will be appended to this file so that we can
        # inspect the reasoning of the database-level agents.
        self._llm_log_fp: Path = self.optimization_db_path.parent / "database_llm_log.txt"
        # Log file that captures every change made to the optimisation database
        # (e.g. updates to measured improvements, newly added techniques, etc.).
        self._db_change_log_fp: Path = self.optimization_db_path.parent / "database_change_log.txt"
        # Path where a live JSON snapshot of the database will be stored.
        self._persist_json_fp: Path = self.optimization_db_path.with_suffix(".json")

        # Ensure the LLM log file exists so users can reliably find it even if
        # the run ends up taking deterministic fallback paths.
        try:
            self._llm_log_fp.parent.mkdir(parents=True, exist_ok=True)
            with open(self._llm_log_fp, "a", encoding="utf-8"):
                pass
        except Exception:
            pass

        # Serialize concurrent writes across tasks sharing this instance (re-entrant for nested writes)
        self._io_lock: threading.RLock = threading.RLock()

        # Database structures
        self.known_states: Dict[str, StateProfile] = {}
        self.optimization_strategies: Dict[str, Dict[str, Any]] = {} # Changed to Dict[str, Dict[str, Any]]
        self.composite_optimizations: Dict[str, List[CompositeOptimization]] = {}
        self.discovered_states: Dict[str, Dict[str, Any]] = {}  # Track AI-discovered states
        # Cache of LLM-recommended optimizations keyed by state name so that callers can
        # retrieve them directly without having to run additional selection logic.
        # self._llm_recommended_optimizations: Dict[str, OptimizationEntry | CompositeOptimization] = {}
        self._llm_recommended_optimizations: Dict[str, OptimizationEntry] = {}
        
        # Load comprehensive optimization knowledge
        self.gpu_optimization_knowledge = ""
        self.load_databases()

        # Create an iniital json from the loaded llm database using _persist_database
        self._persist_database()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(self.optimization_strategies)
        print(f"Persisted database to {self._persist_json_fp}")

        # exit(0)

    # ------------------------------------------------------------------
    # Helper: persist LLM prompt / response pairs for debugging
    # ------------------------------------------------------------------
    def _log_llm_interaction(self, label: str, prompt: str, response: str):
        """Append a labelled prompt / response pair to the shared log file."""
        try:
            with open(self._llm_log_fp, "a", encoding="utf-8") as f:
                f.write(f"=== {label} | {datetime.now().isoformat()} ===\n")
                f.write("--- PROMPT ---\n")
                f.write(prompt.strip() + "\n")
                f.write("--- RESPONSE ---\n")
                f.write((response or "<empty response>").strip() + "\n\n")
        except Exception as e:
            # Logging failure should never crash the optimisation process
            if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                self.llm_interface.logger.warning(f"Failed to write LLM log: {e}")

    # ------------------------------------------------------------------
    # Helper: persist structural changes to the optimisation database
    # ------------------------------------------------------------------
    def _log_db_change(self, action: str, details: Any):
        """Write a record of *action* together with *details* to the change log."""
        # Log the file write, with the file path
        # Write to logger
        if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
            self.llm_interface.logger.info(f"=== {action} | {datetime.now().isoformat()} ===\n")
        try:
            if not isinstance(details, str):
                import json as _json
                try:
                    details = _json.dumps(details, indent=2)
                except Exception:
                    details = str(details)

            with self._io_lock:
                with open(self._db_change_log_fp, "a", encoding="utf-8") as f:
                    f.write(f"=== {action} | {datetime.now().isoformat()} ===\n")
                    f.write(details.strip() + "\n\n")

            # After logging, persist the full database snapshot so that we
            # always have an up-to-date machine-readable version.
            self._persist_database()
        except Exception as e:
            # Never fail hard on logging – just emit a warning if possible.
            if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                self.llm_interface.logger.warning(f"Failed to write DB change log: {e}")

    # ------------------------------------------------------------------
    # Helper: persist the entire optimisation database as JSON
    # ------------------------------------------------------------------
    def _persist_database(self):
        """Dump the current in-memory database state to *self._persist_json_fp*."""

        print(f"_persist_database: Persisting database to {self._persist_json_fp}")
        # log this
        self.llm_interface.logger.info(f"_persist_database: Persisting database to {self._persist_json_fp}")
        try:
            import json as _json
            from dataclasses import asdict as _asdict

            # Convert optimization strategies to a list of dictionaries
            optimization_strategies = {}
            for k, v in self.optimization_strategies.items():
                # Normalize secondary_characteristics - it might be a string from markdown parsing
                secondary_chars = v.get("secondary_characteristics", [])
                if isinstance(secondary_chars, str):
                    # If it's a string, try to parse it as a comma-separated list
                    secondary_chars = [s.strip() for s in secondary_chars.split(",") if s.strip()]
                
                optimization_strategies[k] = {
                    "optimizations": [_asdict(o) for o in v.get("optimizations", [])],
                    "primary_bottleneck": v.get("primary_bottleneck", ""),
                    "secondary_characteristics": secondary_chars if isinstance(secondary_chars, list) else [],
                    # "performance_signature": v["performance_signature"],
                    # "context_description": v["context_description"]
                }

            data = {
                "known_states": {k: _asdict(v) for k, v in self.known_states.items()},
                "optimization_strategies": optimization_strategies,
                "composite_optimizations": {
                    k: [_asdict(o) for o in v] for k, v in self.composite_optimizations.items()
                },
                "discovered_states": self.discovered_states,
            }

            # Guard snapshot writes as well
            if hasattr(self, "_io_lock"):
                with self._io_lock:
                    with open(self._persist_json_fp, "w", encoding="utf-8") as fp:
                        _json.dump(data, fp, indent=2)
            else:
                with open(self._persist_json_fp, "w", encoding="utf-8") as fp:
                    _json.dump(data, fp, indent=2)
        except Exception as e:
            # Fail-soft – we only warn if persistence fails.
            print(f"_persist_database: Failed to persist database JSON: {e}")
            if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                self.llm_interface.logger.warning(f"Failed to persist database JSON: {e}")
    
    def load_databases(self):
        """Load optimization database and GPU optimization report."""
        print(f"Loading databases from {self.optimization_db_path} and {self.gpu_report_path}")
        # Load GPU optimization report as comprehensive knowledge base
        if self.gpu_report_path.exists():
            self.gpu_optimization_knowledge = self.gpu_report_path.read_text()
            print(f"Loaded GPU optimization report: {len(self.gpu_optimization_knowledge)} characters")
        else:
            print(f"Warning: GPU optimization report not found at {self.gpu_report_path}")
        
        # Get default location in data/kernelblaster for fallback
        # Repo root is the project root (e.g. /path/to/KernelBlaster)
        repo_root = Path(__file__).resolve().parents[3]
        default_json_path = repo_root / "data" / "kernelblaster" / "optimization_database.json"
        # Default header/footer live alongside the JSON template
        default_header_path = repo_root / "data" / "kernelblaster" / "optimization_database_header.md"
        default_footer_path = repo_root / "data" / "kernelblaster" / "optimization_database_footer.md"
        
        # Load current optimization database
        # Priority: 1) persisted JSON in output dir, 2) markdown in output dir,
        #           3) default JSON (and initialize persisted copy), 4) default markdown
        loaded = False
        if self._persist_json_fp.exists():
            print(f"Loading database from persisted JSON: {self._persist_json_fp}")
            self._regenerate_database_from_json()
            loaded = True
        elif self.optimization_db_path.exists():
            print(f"Loading database from markdown: {self.optimization_db_path}")
            self._parse_optimization_database()
            loaded = True
        elif default_json_path.exists():
            # Initialize persisted JSON by copying from the default template
            print(f"Initializing database JSON from default location: {default_json_path}")
            try:
                self._persist_json_fp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(default_json_path, self._persist_json_fp)
                print(f"Copied default database JSON to {self._persist_json_fp}")
                # Also initialise header/footer markdown if they don't exist yet
                if default_header_path.exists() and not self.optimization_db_header_path.exists():
                    shutil.copy2(default_header_path, self.optimization_db_header_path)
                    print(f"Copied default header markdown to {self.optimization_db_header_path}")
                if default_footer_path.exists() and not self.optimization_db_footer_path.exists():
                    shutil.copy2(default_footer_path, self.optimization_db_footer_path)
                    print(f"Copied default footer markdown to {self.optimization_db_footer_path}")
            except Exception as e:
                print(f"Failed to copy default optimization database assets: {e}")
            # Now load from the (newly created) persisted JSON
            if self._persist_json_fp.exists():
                print(f"Loading database from persisted JSON: {self._persist_json_fp}")
                self._regenerate_database_from_json()
                loaded = True
        
        if not loaded:
            print(f"Warning: No optimization database found. Starting with empty database.")
            print(f"  Searched paths:")
            print(f"    - {self._persist_json_fp}")
            print(f"    - {self.optimization_db_path}")
            print(f"    - {default_json_path}")
            print(f"    - {default_md_path}")
        else:
            # Log database statistics
            num_states = len(self.optimization_strategies)
            total_optimizations = sum(
                len(state_data.get("optimizations", []))
                for state_data in self.optimization_strategies.values()
            )
            print(f"Database loaded: {num_states} states, {total_optimizations} optimizations")
        
        # Extract known states from GPU optimization report
        self._extract_states_from_gpu_report()
    
    def _regenerate_database_from_json(self):
        """Regenerate the database from the JSON file."""
        try:
            import json as _json
            # 1) Load persisted JSON snapshot
            if not self._persist_json_fp.exists():
                print(f"_regenerate_database_from_json: JSON snapshot not found at {self._persist_json_fp}")
                return

            data = _json.loads(self._persist_json_fp.read_text(encoding="utf-8"))

            # 2) Rebuild in-memory structures from JSON
            self.known_states = {}
            for k, v in data.get("known_states", {}).items():
                try:
                    self.known_states[k] = StateProfile(**v)
                except Exception:
                    # Be robust against schema drift
                    self.known_states[k] = StateProfile(
                        state_name=v.get("state_name", k),
                        primary_bottleneck=v.get("primary_bottleneck", "unknown_bound"),
                        secondary_characteristics=v.get("secondary_characteristics", []),
                        performance_signature=v.get("performance_signature", ""),
                        context_description=v.get("context_description", ""),
                        relative_patterns=v.get("relative_patterns", {}),
                    )

            # Optimization strategies
            self.optimization_strategies = {}
            for state_name, state_data in data.get("optimization_strategies", {}).items():
                optim_dicts = state_data.get("optimizations", [])
                optim_entries: List[OptimizationEntry] = []
                for od in optim_dicts:
                    try:
                        optim_entries.append(OptimizationEntry(**od))
                    except Exception:
                        # Minimal compatible construction
                        optim_entries.append(
                            OptimizationEntry(
                                technique=od.get("technique", "unknown"),
                                predicted_improvement=(
                                    float(od.get("predicted_improvement"))
                                    if od.get("predicted_improvement") not in (None, "")
                                    else None
                                ),
                                description=od.get("description", ""),
                                category=od.get("category", "general"),
                                actual_improvement=od.get("actual_improvement"),
                                confidence_score=float(od.get("confidence_score", 0.5) or 0.5),
                                last_updated=od.get("last_updated"),
                                usage_count=int(od.get("usage_count", 0) or 0),
                            )
                        )

                self.optimization_strategies[state_name] = {
                    "optimizations": optim_entries,
                    "primary_bottleneck": state_data.get("primary_bottleneck", ""),
                    "secondary_characteristics": state_data.get("secondary_characteristics", []),
                }

            # Composite optimizations
            self.composite_optimizations = {}
            for state_name, comp_list in data.get("composite_optimizations", {}).items():
                comps: List[CompositeOptimization] = []
                for cd in comp_list:
                    try:
                        comps.append(CompositeOptimization(**cd))
                    except Exception:
                        comps.append(
                            CompositeOptimization(
                                state=cd.get("state", state_name),
                                technique1=cd.get("technique1", ""),
                                technique2=cd.get("technique2"),
                                technique3=cd.get("technique3"),
                                order_of_techniques=cd.get("order_of_techniques", []),
                                parameters_to_fine_tune=cd.get("parameters_to_fine_tune", {}),
                                predicted_improvement=float(cd.get("predicted_improvement", 0.0) or 0.0),
                                actual_improvement=cd.get("actual_improvement"),
                                reason=cd.get("reason", ""),
                                side_effects=cd.get("side_effects", ""),
                                confidence_score=float(cd.get("confidence_score", 0.5)),
                                last_updated=cd.get("last_updated"),
                                usage_count=int(cd.get("usage_count", 0)),
                            )
                        )
                self.composite_optimizations[state_name] = comps

            # Discovered states metadata
            self.discovered_states = data.get("discovered_states", {})

            # 3) Compose full markdown from current in-memory data and write it
            final_markdown = self.get_database_md_text(include_header_footer=True)
            self.optimization_db_path.write_text(final_markdown, encoding="utf-8")
            print(f"_regenerate_database_from_json: Regenerated markdown at {self.optimization_db_path}")
        except Exception as e:
            print(f"_regenerate_database_from_json: Failed to regenerate from JSON: {e}")

    def _build_states_markdown(self) -> str:
        """Build only the states section in markdown from in-memory structures."""
        def _fmt_chars(chars: Any) -> str:
            if isinstance(chars, list):
                return ", ".join(str(c) for c in chars)
            return str(chars) if chars is not None else ""

        def _fmt_impr(val: Any) -> str:
            if val is None:
                return "0"
            try:
                return f"{float(val):g}"
            except Exception:
                return "0"

        state_sections: List[str] = []
        for state_name, state_data in self.optimization_strategies.items():
            primary_bottleneck = state_data.get("primary_bottleneck", "")
            secondary_chars = _fmt_chars(state_data.get("secondary_characteristics", []))
            lines: List[str] = []
            lines.append(f"#### State: {state_name}")
            if secondary_chars:
                lines.append(f"**Characteristics**: {secondary_chars}")
            if primary_bottleneck:
                lines.append(f"**Primary Bottleneck**: {primary_bottleneck}")
            lines.append("**Optimizations**:")

            opts: List[OptimizationEntry] = state_data.get("optimizations", [])
            if opts:
                for opt in opts:
                    desc = f" - {opt.description}" if getattr(opt, "description", "") else ""
                    predicted_speedup = getattr(opt, "predicted_speedup", None)
                    if predicted_speedup in (None, 0.0):
                        # derive speedup from predicted_improvement percent as fallback
                        pred_impr = (getattr(opt, "predicted_improvement", 0.0) or 0.0)
                        predicted_speedup = 1.0 / max(1e-6, 1.0 - (pred_impr / 100.0))
                    lines.append(
                        f"- **{opt.technique}**: {predicted_speedup:.2f}x predicted speedup{desc}"
                    )
            else:
                lines.append("- (no optimizations available)")

            state_sections.append("\n".join(lines))

        return ("\n\n".join(state_sections).rstrip() + "\n") if state_sections else ""

    def get_database_footer_text(self) -> str:
        """Return the footer text for the database."""
        return self.optimization_db_footer_path.read_text(encoding="utf-8") if self.optimization_db_footer_path.exists() else ""

    def get_database_md_text(self, include_header_footer: bool = True) -> str:
        """Return the full database markdown text without writing to disk.

        When include_header_footer is True, raw header and footer files are
        included around the regenerated states markdown if present.
        """
        states_md = self._build_states_markdown()
        if not include_header_footer:
            return states_md

        header_text = (
            self.optimization_db_header_path.read_text(encoding="utf-8")
            if self.optimization_db_header_path.exists()
            else ""
        )
        footer_text = (
            self.optimization_db_footer_path.read_text(encoding="utf-8")
            if self.optimization_db_footer_path.exists()
            else ""
        )

        parts: List[str] = []
        if header_text.strip():
            parts.append(header_text.rstrip())
        parts.append(states_md)
        if footer_text.strip():
            parts.append(footer_text.rstrip())

        return "\n\n".join(parts).rstrip() + "\n"
    
    def _parse_optimization_database(self):
        """Parse the existing optimization database."""
        content = self.optimization_db_path.read_text()
        current_state = None
        print(f"Parsing optimization database from {self.optimization_db_path}")
        
        # Parse JSON sections for composite optimizations
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                json_data = json.loads(json_match.group(1))
                self._load_composite_optimizations(json_data)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON section: {e}")
        
        # Parse basic optimizations
        for line in content.split('\n'):
            line = line.strip()
            
            state_match = re.match(r'#### State: (.+)', line)
            if state_match:
                current_state = state_match.group(1).strip()
                if current_state not in self.optimization_strategies:
                    self.optimization_strategies[current_state] = {"optimizations": []}
                continue
            
            # Extended regex: captures optional description after the improvement figure.
            # Example line:
            # - **memory_compute_overlap**: 0% performance improvement - Pipeline memory and compute operations
            opt_match = re.match(
                r'- \*\*(.+?)\*\*: (\d+(?:\.\d+)?)% performance improvement(?:\s*-\s*(.+))?',
                line,
            )

            # if line starts with **Characteristics**:
            if line.startswith("**Characteristics**:"):
                current_state_characteristics = line.split(":", 1)[1].strip()
                self.optimization_strategies[current_state]["secondary_characteristics"] = current_state_characteristics
            # if line starts with **Primary Bottleneck**:
            if line.startswith("**Primary Bottleneck**:"):
                current_state_primary_bottleneck = line.split(":", 1)[1].strip()
                self.optimization_strategies[current_state]["primary_bottleneck"] = current_state_primary_bottleneck


            if opt_match and current_state:
                technique = opt_match.group(1).strip()
                improvement = float(opt_match.group(2))
                description = opt_match.group(3).strip() if opt_match.group(3) else ""
                
                entry = OptimizationEntry(
                    technique=technique,
                    predicted_improvement=improvement,
                    description=description,
                    category=self._categorize_technique(technique),
                )
                self.optimization_strategies[current_state]["optimizations"].append(entry)
                # print(f"Added optimization strategy for {current_state}: {technique} with improvement {improvement}")
    
    def _extract_states_from_gpu_report(self):
        """Extract state patterns from the comprehensive GPU optimization report."""
        # This creates qualitative state profiles from the GPU optimization report
        # Based on the decision tree structure
        
        memory_bound_profile = StateProfile(
            state_name="memory_bandwidth_limited",
            primary_bottleneck="memory_bound",
            secondary_characteristics=[
                "High memory throughput utilization",
                "Bandwidth saturation",
                "Potential coalescing issues",
                "Cache inefficiencies"
            ],
            performance_signature="Memory bandwidth is the primary limiting factor with potential for access pattern optimization",
            context_description="Workload is limited by memory bandwidth, showing signs of inefficient access patterns or cache behavior",
            relative_patterns={
                "memory_pressure": "high",
                "compute_utilization": "underutilized", 
                "access_patterns": "potentially_uncoalesced",
                "cache_behavior": "suboptimal"
            }
        )
        
        compute_bound_profile = StateProfile(
            state_name="compute_throughput_limited",
            primary_bottleneck="compute_bound",
            secondary_characteristics=[
                "High compute unit utilization",
                "Instruction throughput bottleneck",
                "Potential for specialized units",
                "Arithmetic intensity"
            ],
            performance_signature="Compute units are saturated, indicating opportunity for algorithmic or instruction-level optimization",
            context_description="Workload is compute-intensive with potential for specialized hardware utilization or algorithmic improvements",
            relative_patterns={
                "compute_pressure": "high",
                "memory_utilization": "adequate",
                "instruction_mix": "potentially_suboptimal",
                "parallelism": "high"
            }
        )
        
        latency_bound_profile = StateProfile(
            state_name="latency_occupancy_limited", 
            primary_bottleneck="latency_bound",
            secondary_characteristics=[
                "Low occupancy",
                "Insufficient parallelism",
                "Resource underutilization",
                "Synchronization overhead"
            ],
            performance_signature="Neither memory nor compute are saturated, indicating latency hiding or occupancy issues",
            context_description="Workload has insufficient parallelism or resource conflicts limiting occupancy and latency hiding",
            relative_patterns={
                "occupancy": "low",
                "parallelism": "insufficient",
                "resource_conflicts": "present",
                "latency_hiding": "poor"
            }
        )
        
        # Store known state profiles
        self.known_states = {
            "memory_bandwidth_limited": memory_bound_profile,
            "compute_throughput_limited": compute_bound_profile,
            "latency_occupancy_limited": latency_bound_profile
        }
    
    async def analyze_performance_state(self, ncu_report: str, metrics: dict, code_implementation: str, elapsed_cycles: Optional[int] = None) -> StateProfile:
        """
        LLM Agent 1: State Summarizer
        Analyzes NCU report and extracts qualitative performance characteristics.
        """
        
        # If ncu_report is empty but we have cycles, construct a minimal report
        # Only show cycles if they're > 0 (0 usually indicates parsing failure)
        if not ncu_report.strip() and elapsed_cycles is not None and elapsed_cycles > 0:
            ncu_report = f"""Elapsed Cycles: {elapsed_cycles:,}

Note: This is a cycles-only profiling mode. Detailed NCU metrics are not available.
Use the code implementation and elapsed cycles to infer performance characteristics."""
        elif not ncu_report.strip():
            # If cycles are 0 or None, indicate that profiling data is unavailable
            ncu_report = """Note: Cycles-only profiling mode is enabled, but elapsed cycles were not successfully parsed from the program output.
Detailed NCU metrics are not available. Please analyze based on the code implementation alone."""

        state_analysis_prompt = f"""
You are a GPU performance analysis expert. Analyze this NVIDIA NSight Compute (NCU) profiling report and provide a qualitative summary of the kernel's performance state.

CODE IMPLEMENTATION:
```cpp
{code_implementation}
```

NCU REPORT:
{ncu_report}  

Provide your analysis in this EXACT format:

PERFORMANCE_SIGNATURE: [2-3 sentence summary of what is limiting performance and the overall execution pattern]

RELATIVE_PATTERNS:
- memory_pressure: [very_low|low|moderate|high|very_high]
- compute_utilization: [very_low|low|moderate|high|very_high] 
- access_patterns: [excellent|good|moderate|poor|very_poor]
- cache_efficiency: [excellent|good|moderate|poor|very_poor]
- occupancy_level: [very_low|low|moderate|high|very_high]
- parallelism_utilization: [very_low|low|moderate|high|very_high]
- specialied_hw_usage: [very_low|low|moderate|high|very_high]
- [List 3-5 key secondary performance characteristics]
- [Focus on patterns you observe in the data]
- [Include cache behavior, memory access patterns, occupancy]
- [Note any resource conflicts or inefficiencies]

PRIMARY_BOTTLENECK: [memory_bound|compute_bound|latency_bound|hybrid_bound]

//code signiture: loop pattern /branches(in summary/generic)


CONTEXT_DESCRIPTION: [Brief description of the workload characteristics and optimization opportunities]

Focus on qualitative patterns and relationships rather than specific numbers. Look for the underlying performance characteristics that drive behavior.
"""
        
        if self.llm_interface and self.llm_interface.is_available():
            try:
                analysis = await self.llm_interface.query(state_analysis_prompt, max_tokens=800, temperature=0.1)
                # Log prompt/response for transparency
                self._log_llm_interaction("StateAnalysis", state_analysis_prompt, analysis)
                return self._parse_state_analysis(analysis)
            except Exception as e:
                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                    self.llm_interface.logger.warning(f"State analysis failed: {e}")
                return self._fallback_state_analysis(ncu_report, metrics)
        else:
            return self._fallback_state_analysis(ncu_report, metrics)
    
    def _parse_state_analysis(self, llm_response: str) -> StateProfile:
        """Parse LLM state analysis response into StateProfile."""
        lines = llm_response.split('\n')
        
        primary_bottleneck = "unknown_bound"
        secondary_characteristics = []
        performance_signature = ""
        context_description = ""
        relative_patterns = {}
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("PRIMARY_BOTTLENECK:"):
                primary_bottleneck = line.split(":", 1)[1].strip()
            elif line.startswith("SECONDARY_CHARACTERISTICS:"):
                current_section = "secondary"
            elif line.startswith("PERFORMANCE_SIGNATURE:"):
                current_section = "signature"
                performance_signature = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line.startswith("RELATIVE_PATTERNS:"):
                current_section = "patterns"
            elif line.startswith("CONTEXT_DESCRIPTION:"):
                current_section = "context"
                context_description = line.split(":", 1)[1].strip() if ":" in line else ""
            elif current_section == "secondary" and line.startswith("-"):
                secondary_characteristics.append(line[1:].strip())
            elif current_section == "signature" and line and not line.startswith(("RELATIVE", "CONTEXT")):
                performance_signature += " " + line
            elif current_section == "patterns" and ":" in line:
                key, value = line.split(":", 1)
                relative_patterns[key.strip().replace("- ", "")] = value.strip()
            elif current_section == "context" and line and not line.startswith(("PRIMARY", "SECONDARY")):
                context_description += " " + line
        
        return StateProfile(
            state_name="current_analysis",
            primary_bottleneck=primary_bottleneck,
            secondary_characteristics=secondary_characteristics,
            performance_signature=performance_signature.strip(),
            context_description=context_description.strip(),
            relative_patterns=relative_patterns
        )
    
    async def match_state_against_database(self, current_state: StateProfile) -> str:
        """
        LLM Agent 2: State Matcher
        Compares current state against known optimization patterns qualitatively.
        """
        
        # Prepare known states for comparison
        known_states_text = ""
        # Build text from optimisation_strategies (deprecated known_states removed)
        for state_name, state_data in self.optimization_strategies.items():
            primary_bottleneck = state_data.get("primary_bottleneck", "")
            secondary_chars = state_data.get("secondary_characteristics", "")
            optimization_list = ""
            opts = state_data.get("optimizations", [])[:3]
            if opts:
                optimization_list = "\n".join(
                    [
                        f"  - {opt.technique}: {((getattr(opt, 'predicted_speedup', None) if getattr(opt, 'predicted_speedup', None) not in (None, 0.0) else (1.0 / max(1e-6, 1.0 - ((getattr(opt, 'predicted_improvement', 0.0) or 0.0)/100.0))))):.2f}x speedup"
                        for opt in opts
                    ]
                )

            known_states_text += f"""
STATE: {state_name}
Primary Bottleneck: {primary_bottleneck}
Secondary Characteristics: {secondary_chars}
Available Optimizations:
{optimization_list}

"""
        
        matching_prompt = f"""
You are a GPU optimization expert. Compare the current kernel performance state against known optimization states and find the best match.

CURRENT STATE TO MATCH:
Primary Bottleneck: {current_state.primary_bottleneck}
Secondary Characteristics: {', '.join(current_state.secondary_characteristics)}
Performance Signature: {current_state.performance_signature}
Relative Patterns: {json.dumps(current_state.relative_patterns, indent=2)}
Context: {current_state.context_description}

KNOWN OPTIMIZATION STATES:
{known_states_text}

MATCHING INSTRUCTIONS:
1. Primary bottleneck must align (memory_bound with memory_bound, etc.)
2. Look for similar secondary characteristics and patterns
3. Consider the performance signature and context similarity
4. Focus on qualitative patterns rather than exact matches

Provide your analysis in this EXACT format:

BEST_MATCH: [state_name from database or "NEW_STATE_NEEDED"]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Explain why this state matches, focusing on bottleneck alignment and similar characteristics]

If confidence < 0.6, respond with BEST_MATCH: NEW_STATE_NEEDED
"""
        
        if self.llm_interface and self.llm_interface.is_available():
            try:
                matching_result = await self.llm_interface.query(matching_prompt, max_tokens=500, temperature=0.1)
                # Log interaction
                self._log_llm_interaction("StateMatching", matching_prompt, matching_result)
                return self._parse_matching_result(matching_result)
            except Exception as e:
                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                    self.llm_interface.logger.warning(f"State matching failed: {e}")
                return self._fallback_state_matching(current_state)
        else:
            return self._fallback_state_matching(current_state)
    
    def _parse_matching_result(self, llm_response: str) -> str:
        """Parse LLM matching response to extract best match."""
        lines = llm_response.split('\n')
        
        for line in lines:
            if line.startswith("BEST_MATCH:"):
                match = line.split(":", 1)[1].strip()
                return match
        
        return "unknown_state"
    
    async def get_state_from_ncu_report(
        self, ncu_report: str, metrics: dict, code_implementation: str = "", elapsed_cycles: Optional[int] = None
    ) -> str:
        """
        Main interface: Two-LLM agent system for state identification.
        
        Returns the matched state name for optimization selection.
        """

        # Agent 1: Analyze current state qualitatively
        current_state = await self.analyze_performance_state(
            ncu_report, metrics, code_implementation, elapsed_cycles=elapsed_cycles
        )
        
        # Agent 2: Match against known optimization patterns
        matched_state = await self.match_state_against_database(current_state)
        
        # Handle new state discovery - HYBRID APPROACH: Create new state + inherit strategies
        if matched_state == "NEW_STATE_NEEDED" or matched_state == "unknown_state":
            # Create new state to preserve unique characteristics
            new_state_name = f"discovered_{current_state.primary_bottleneck}_{len(self.optimization_strategies)}"
            
            # Find the best existing state to inherit optimization strategies from
            source_state = self._map_to_existing_state_with_strategies(current_state)
            
            if source_state and source_state in self.optimization_strategies:
                # Copy optimization strategies but with reduced confidence for the new state
                inherited_strategies = []
                for original_strategy in self.optimization_strategies[source_state].get("optimizations", []):
                    inherited_strategy = OptimizationEntry(
                        technique=original_strategy.technique,
                        predicted_improvement=original_strategy.predicted_improvement * 0.8 if original_strategy.predicted_improvement is not None else None,  # Reduce confidence
                        description=f"Inherited from {source_state}: {original_strategy.description}",
                        category=original_strategy.category,
                        confidence_score=original_strategy.confidence_score * 0.8,  # Lower confidence for inheritance
                        last_updated=datetime.now().isoformat(),
                        usage_count=original_strategy.usage_count  # Preserve accumulated usage count instead of resetting
                    )
                    inherited_strategies.append(inherited_strategy)
                
                # Assign inherited strategies to the new state (wrap with metadata)
                self.optimization_strategies[new_state_name] = {
                    "optimizations": inherited_strategies,
                    "primary_bottleneck": current_state.primary_bottleneck,
                    "secondary_characteristics": current_state.secondary_characteristics,
                }
                
                # Store detailed discovery metadata
                self.discovered_states[new_state_name] = {
                    "original_state": current_state.__dict__,
                    "inherited_from": source_state,
                    "inherited_strategies_count": len(inherited_strategies),
                    "discovery_timestamp": datetime.now().isoformat(),
                    "approach": "hybrid_create_and_inherit"
                }
                
                # Log the hybrid creation
                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                    self.llm_interface.logger.info(
                        f"Created new state '{new_state_name}' with {len(inherited_strategies)} "
                        f"strategies inherited from '{source_state}' (bottleneck: {current_state.primary_bottleneck})"
                    )
                
                # Pre-select and cache an optimisation for the newly discovered state
                try:
                    best_opt = await self._select_best_optimization_llm(new_state_name, current_state)
                    if best_opt:
                        self._llm_recommended_optimizations[new_state_name] = best_opt
                except Exception as e:
                    if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                        self.llm_interface.logger.warning(f"LLM optimisation selection failed: {e}")

                return new_state_name
            
            else:
                # Fallback: Create state with default strategies based on bottleneck type
                default_strategies = self._create_default_strategies_for_bottleneck(current_state.primary_bottleneck)
                if default_strategies:
                    self.optimization_strategies[new_state_name] = {
                        "optimizations": default_strategies,
                        "primary_bottleneck": current_state.primary_bottleneck,
                        "secondary_characteristics": current_state.secondary_characteristics,
                    }
                    
                    # Store metadata for default strategy creation
                    self.discovered_states[new_state_name] = {
                        "original_state": current_state.__dict__,
                        "strategy_source": "default_for_bottleneck",
                        "default_strategies_count": len(default_strategies),
                        "discovery_timestamp": datetime.now().isoformat(),
                        "approach": "create_with_default_strategies"
                    }
                    
                    if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                        self.llm_interface.logger.info(
                            f"Created new state '{new_state_name}' with {len(default_strategies)} "
                            f"default strategies for bottleneck: {current_state.primary_bottleneck}"
                        )
                    
                    # Pre-select and cache an optimisation for the default-strategy state
                    try:
                        best_opt = await self._select_best_optimization_llm(new_state_name, current_state)
                        if best_opt:
                            self._llm_recommended_optimizations[new_state_name] = best_opt
                    except Exception as e:
                        if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                            self.llm_interface.logger.warning(f"LLM optimisation selection failed: {e}")

                return new_state_name
        
        # Cache an optimisation for the matched existing state so that callers
        # can retrieve it immediately via select_best_optimization.
        try:
            best_opt = await self._select_best_optimization_llm(matched_state, current_state)
            if best_opt:
                self._llm_recommended_optimizations[matched_state] = best_opt
        except Exception as e:
            if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                self.llm_interface.logger.warning(f"LLM optimisation selection failed: {e}")

        return matched_state
    
    def _map_to_existing_state_with_strategies(self, current_state: StateProfile) -> Optional[str]:
        """
        Map a discovered state to an existing state that has optimization strategies.
        
        This prevents the issue where discovered states have no optimization strategies.
        """
        # Primary bottleneck mapping to database state names
        bottleneck_to_state_mapping = {
            "memory_bound": [
                "memory_bandwidth_saturated",
                "memory_latency_bound", 
                "memory_bank_conflicts",
                "cache_inefficient"
            ],
            "compute_bound": [
                "compute_throughput_saturated",
                "instruction_mix_suboptimal",
                "thread_divergence_high"
            ],
            "latency_bound": [
                "low_occupancy_register_pressure",
                "low_occupancy_shared_memory",
                "insufficient_parallelism"
            ],
            "hybrid_bound": [
                "memory_compute_balanced",
                "latency_memory_bound"
            ]
        }
        
        # Get candidate states based on primary bottleneck
        candidates = bottleneck_to_state_mapping.get(current_state.primary_bottleneck, [])
        
        # Filter to only states that have optimization strategies
        candidates_with_strategies = [
            state for state in candidates 
            if state in self.optimization_strategies and len(self.optimization_strategies[state].get("optimizations", [])) > 0
        ]
        
        if not candidates_with_strategies:
            # Try to find any state with optimization strategies as last resort
            candidates_with_strategies = [
                state for state in self.optimization_strategies.keys()
                if len(self.optimization_strategies[state].get("optimizations", [])) > 0
            ]
            
            # Log debugging information
            if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                self.llm_interface.logger.debug(
                    f"No direct candidates found for {current_state.primary_bottleneck}. "
                    f"Available states with strategies: {list(self.optimization_strategies.keys())}"
                )
        
        if candidates_with_strategies:
            # For now, return the first candidate. Could be improved with similarity scoring
            selected_state = candidates_with_strategies[0]
            
            # Log the selection
            if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                self.llm_interface.logger.info(
                    f"Selected state '{selected_state}' for bottleneck '{current_state.primary_bottleneck}' "
                    f"with {len(self.optimization_strategies[selected_state].get('optimizations', []))} optimization strategies"
                )
            
            return selected_state
        
        return None
    
    def _fallback_state_analysis(self, ncu_report: str, metrics: dict) -> StateProfile:
        """Fallback analysis when LLM is not available."""
        memory_throughput = metrics.get('memory_throughput', 0)
        compute_throughput = metrics.get('compute_throughput', 0)
        
        if memory_throughput > compute_throughput * 1.2:
            primary_bottleneck = "memory_bound"
        elif compute_throughput > memory_throughput * 1.2:
            primary_bottleneck = "compute_bound"  
        else:
            primary_bottleneck = "hybrid_bound"
        
        return StateProfile(
            state_name="fallback_analysis",
            primary_bottleneck=primary_bottleneck,
            secondary_characteristics=["Fallback analysis - limited detail"],
            performance_signature=f"Fallback analysis indicates {primary_bottleneck} workload",
            context_description="Basic analysis due to LLM unavailability",
            relative_patterns={"analysis_quality": "basic"}
        )
    
    def _fallback_state_matching(self, current_state: StateProfile) -> str:
        """Fallback matching when LLM is not available."""
        # Simple matching based on primary bottleneck
        for state_name, state_data in self.optimization_strategies.items():
            if state_data.get("primary_bottleneck") == current_state.primary_bottleneck:
                return state_name
        return "unknown_state"

    async def _select_best_optimization_llm(
        self,
        state: str,
        current_state_profile: StateProfile,
        include_composite: bool = True,
    ) -> Optional[OptimizationEntry | CompositeOptimization]:
        """Let the LLM pick the single best optimisation globally.

        Instead of limiting the choice to *state*-specific techniques we now
        expose **all** optimisations found in the database (across every
        state).  This gives the selector complete freedom and removes the
        dependency on an explicit state-matching phase.

        The helper still caches the choice under the provided *state* key so
        external callers can access it transparently via
        ``select_best_optimization``.
        """

        def _collect_all_opts(include_composite_flag: bool):
            opts: List[OptimizationEntry | CompositeOptimization] = []
            for state_data in self.optimization_strategies.values():
                opts.extend(state_data.get("optimizations", []))
            if include_composite_flag:
                for lst in self.composite_optimizations.values():
                    opts.extend(lst)
            return opts

        all_opts: List[OptimizationEntry | CompositeOptimization] = _collect_all_opts(include_composite)
        
        if not all_opts:
            return None

        # --------------- Attempt LLM-driven choice ---------------
        chosen_name: Optional[str] = None
        if self.llm_interface and self.llm_interface.is_available():
            # Sample up to 15 representative options to keep the prompt size
            # manageable – choose those with highest predicted improvement.
            top_opts = sorted(
                all_opts,
                key=lambda o: (
                    getattr(o, "predicted_speedup", None)
                    if getattr(o, "predicted_speedup", None) not in (None, 0.0)
                    else 1.0 / max(1e-6, 1.0 - ((getattr(o, "predicted_improvement", 0.0) or 0.0) / 100.0))
                ),
                reverse=True,
            )

            options_text = "\n".join(
                [
                    f"- {opt.technique}: {((getattr(opt, 'predicted_speedup', None) if getattr(opt, 'predicted_speedup', None) not in (None, 0.0) else (1.0 / max(1e-6, 1.0 - ((getattr(opt, 'predicted_improvement', 0.0) or 0.0)/100.0))))):.2f}x (confidence {getattr(opt,'confidence_score',0.5)})"
                    for opt in top_opts
                ]
            )

            prompt = f"""
You are a GPU optimisation expert. A kernel has been analysed and its qualitative
performance characteristics are shown below. From the list of available
optimisation techniques pick the ONE technique that you judge will yield the
largest performance gain. Respond STRICTLY in the format:

BEST_OPTIMIZATION: <technique name>
REASONING: <brief rationale>

CURRENT STATE SUMMARY
Primary Bottleneck: {current_state_profile.primary_bottleneck}
Secondary Characteristics: {', '.join(current_state_profile.secondary_characteristics)}
Performance Signature: {current_state_profile.performance_signature}

AVAILABLE OPTIMISATIONS:
{self._build_available_optimisations_summary()}
            """

            try:
                response = await self.llm_interface.query(prompt, max_tokens=400, temperature=0.1)
                self._log_llm_interaction("OptSelection", prompt, response)

                for line in response.split("\n"):
                    if line.strip().startswith("BEST_OPTIMIZATION"):
                        chosen_name = line.split(":", 1)[1].strip()
                        break
            except Exception as exc:
                if hasattr(self.llm_interface, "logger") and self.llm_interface.logger:
                    self.llm_interface.logger.warning(f"LLM optimisation selection failed: {exc}")

        # --------------- Fallback deterministic choice ---------------
        if not chosen_name:
            def _score(o):
                pred_speedup = getattr(o, "predicted_speedup", None)
                if pred_speedup in (None, 0.0):
                    pred_impr = (getattr(o, "predicted_improvement", 0.0) or 0.0)
                    pred_speedup = 1.0 / max(1e-6, 1.0 - (pred_impr / 100.0))
                return pred_speedup * getattr(o, "confidence_score", 0.5)

            best_opt = max(all_opts, key=_score)
            return best_opt

        # Map the name chosen by the LLM back to the object.
        for opt in all_opts:
            if opt.technique == chosen_name:
                return opt
            if isinstance(opt, CompositeOptimization) and opt.get_composite_id() == chosen_name:
                return opt

        # If we reach here the LLM responded with an unknown name – fall back to deterministic.
        def _score(o):
            pred_speedup = getattr(o, "predicted_speedup", None)
            if pred_speedup in (None, 0.0):
                pred_impr = (getattr(o, "predicted_improvement", 0.0) or 0.0)
                pred_speedup = 1.0 / max(1e-6, 1.0 - (pred_impr / 100.0))
            return pred_speedup * getattr(o, "confidence_score", 0.5)

        return max(all_opts, key=_score)
    
    async def generate_optimization_plan(
        self,
        state_analysis_response: str,
        code_implementation: str,
        top_n: int = 5, #generated top 5 optimisations options
    ) -> List[Dict[str, Any]]:
        """Ask the LLM to pick the *top_n* most relevant optimisation techniques.

        Parameters
        ----------
        state_analysis_response:
            The RAW text returned by :py:meth:`analyze_performance_state` (or an
            equivalent manual analysis).  It provides the qualitative bottleneck
            and performance characteristics of the current kernel.
        code_implementation:
            The CUDA/C++ implementation of the kernel to be optimised – will be
            embedded inside a `````cpp````` block so syntax highlighting is
            preserved for the LLM.
        top_n:
            Number of optimisation candidates to request from the LLM (default 3).

        Returns
        -------
        List[Dict[str, Any]]
            A list with length *top_n* where every element is a dictionary with the
            keys ``technique``, ``relevance_score`` and ``reasoning``.
        """

        # ------------------------- LLM prompt -------------------------
        prompt = f"""
You are a world-class GPU optimisation expert.  Based on the kernel implementation
and the qualitative state analysis below, choose the **{top_n}** optimisation
techniques that are most likely to improve performance.  From the list of
AVAILABLE OPTIMISATIONS pick only those with the highest relevance to the
observed performance characteristics **and** the specific code patterns you see.

For *each* chosen technique provide a concise explanation *why* it is relevant
and *how* it should be applied to the given code.  Also assign a numerical
RELEVANCE_SCORE between 0.0 (not relevant) and 1.0 (perfect match).

Return your answer as **valid JSON** in the EXACT format (no extra keys, no
comments, do not wrap the JSON in markdown fences):

[
  {{
    "technique": "<name>",
    "relevance_score": <float 0-1>,
    "description": "<explain what the optimisation does and *why* it applies to the current code>"
  }},
  ... (exactly {top_n} entries)
]


Besides the available optimisation techniques in Performance State Categories, you should always consider using the following techniques: 

- For memory bandwidth bound, or compute bandwidth bound kernels: prioritize using **SIMD_operations**: Use packed SIMD datatypes such as half2
- For compute bandwidth bound and compute throughput bound kernels: prioritize using **tensor_core_utilization**: Use tensor core library such as wmma when there exist a tensor cores (sm_70+) in the target GPU archecture. 
----------------------- CURRENT KERNEL CONTEXT -----------------------
STATE ANALYSIS RESPONSE:
{state_analysis_response}

CODE IMPLEMENTATION:
```cpp
{code_implementation}
```

----------------------- AVAILABLE OPTIMISATIONS ----------------------
{self._build_available_optimisations_summary()}
"""

        # -------------------- Attempt LLM inference --------------------
        if self.llm_interface and self.llm_interface.is_available():
            try:
                llm_resp = await self.llm_interface.query(prompt, max_tokens=800, temperature=0.1)
                self._log_llm_interaction("OptPlan", prompt, llm_resp)
                plan = self._parse_optimization_plan(llm_resp, top_n)
                if plan:
                    return plan
            except Exception as exc:
                if hasattr(self.llm_interface, "logger") and self.llm_interface.logger:
                    self.llm_interface.logger.warning(f"LLM optimisation-plan generation failed: {exc}")

        # ------------------ Fallback deterministic path ------------------
        def _score(opt):
            pred_speedup = getattr(opt, "predicted_speedup", None)
            if pred_speedup in (None, 0.0):
                pred_impr = (getattr(opt, "predicted_improvement", 0.0) or 0.0)
                pred_speedup = 1.0 / max(1e-6, 1.0 - (pred_impr / 100.0))
            return pred_speedup * getattr(opt, "confidence_score", 0.5)

        # Flatten all optimisation objects
        all_opts: List[OptimizationEntry | CompositeOptimization] = list(
            itertools.chain.from_iterable(
                state_data.get("optimizations", []) for state_data in self.optimization_strategies.values()
            )
        )
        all_opts.extend(itertools.chain.from_iterable(self.composite_optimizations.values()))
        if not all_opts:
            return []

        best_opts = sorted(all_opts, key=_score, reverse=True)[:top_n]
        fallback_plan: List[Dict[str, Any]] = []
        for opt in best_opts:
            fallback_plan.append(
                {
                    "technique": opt.technique if isinstance(opt, OptimizationEntry) else opt.get_composite_id(),
                    "relevance_score": min(1.0, _score(opt) / 100.0),  # crude normalisation
                    "description": "Selected via deterministic fallback based on predicted speedup.",
                }
            )
        return fallback_plan

    # ------------------------------------------------------------------
    # Helper: parse optimisation plan JSON returned by the LLM
    # ------------------------------------------------------------------
    def _parse_optimization_plan(self, llm_response: str, expected_n: int) -> List[Dict[str, Any]]:
        """Attempt to JSON-decode *llm_response* and validate structure."""
        import json as _json

        try:
            plan = _json.loads(llm_response)
            if (
                isinstance(plan, list)
                and len(plan) == expected_n
                and all(isinstance(p, dict) for p in plan)
            ):
                return plan  # type: ignore[return-value]
        except Exception:
            pass  # fallthrough – caller will trigger deterministic fallback

        return []
    
    # ------------------------------------------------------------------
    # Helper: build human-readable summary of all optimisation techniques
    # ------------------------------------------------------------------
    def _build_available_optimisations_summary(self) -> str:
        """Return a multi-line string enumerating all optimisation techniques.

        Format:
        STATE: <state_name>
          - <technique> (pred <x>% | conf <y>): <description>
        """

        lines: List[str] = []
        for state, state_data in self.optimization_strategies.items():
            opts = state_data.get("optimizations", [])
            if not opts:
                continue
            lines.append(f"STATE: {state}")
            for opt in opts:
                desc = opt.description or "(no description)"
                predicted_speedup = getattr(opt, 'predicted_speedup', None)
                if predicted_speedup in (None, 0.0):
                    pred_impr = (getattr(opt, 'predicted_improvement', 0.0) or 0.0)
                    predicted_speedup = 1.0 / max(1e-6, 1.0 - (pred_impr / 100.0))
                lines.append(
                    f"  - {opt.technique} (pred {predicted_speedup:.2f}x | conf {opt.confidence_score}): {desc}"
                )
            lines.append("")  # blank line between states

        # Include composite optimisations
        for state, comps in self.composite_optimizations.items():
            if not comps:
                continue
            lines.append(f"STATE (composite): {state}")
            for comp in comps:
                desc = comp.reason or "(no description)"
                predicted_speedup = getattr(comp, 'predicted_speedup', None)
                if predicted_speedup in (None, 0.0):
                    pred_impr = (getattr(comp, 'predicted_improvement', 0.0) or 0.0)
                    predicted_speedup = 1.0 / max(1e-6, 1.0 - (pred_impr / 100.0))
                lines.append(
                    f"  - {comp.get_composite_id()} (pred {predicted_speedup:.2f}x | conf {comp.confidence_score}): {desc}"
                )
            lines.append("")

        return "\n".join(lines)
    
    def get_optimizations_for_state(self, state: str) -> List[OptimizationEntry]:
        """Get optimization strategies for a given state."""
        return self.optimization_strategies.get(state, {}).get("optimizations", [])
    
    def get_composite_optimizations_for_state(self, state: str) -> List[CompositeOptimization]:
        """Get composite optimizations for a given state."""
        return self.composite_optimizations.get(state, [])
    
    def select_best_optimization(self, state: str, exclude_used: bool = False, 
                                include_composite: bool = True) -> Optional[OptimizationEntry | CompositeOptimization]:
        """Return the optimisation previously selected by the LLM for *state*.

        This method is now a lightweight accessor so that existing external
        code can remain unchanged.  If the user requests to exclude already
        used techniques we honour that contract; otherwise the cached
        recommendation is returned directly.  If, for any reason, no cached
        recommendation exists we fall back to the legacy stochastic chooser
        to preserve behaviour.
        """

        # ----------------------------------------------------------
        # 1) Fast path – the LLM has already made a recommendation.
        # ----------------------------------------------------------
        if state in self._llm_recommended_optimizations:
            recommended = self._llm_recommended_optimizations[state]
            if exclude_used and getattr(recommended, "usage_count", 0) > 0:
                return None
            try:
                tech = (
                    recommended.get_composite_id()
                    if isinstance(recommended, CompositeOptimization)
                    else getattr(recommended, "technique", str(recommended))
                )
                self.llm_interface.logger.info(
                    f"[select_best_optimization] Using cached LLM recommendation for state='{state}': {tech}"
                )
            except Exception:
                # Best-effort logging only
                pass
            return recommended

        # ----------------------------------------------------------
        # 2) Fallback – use the old probabilistic scoring mechanism.
        # ----------------------------------------------------------
        try:
            self.llm_interface.logger.info(
                f"[select_best_optimization] No cached LLM recommendation for state='{state}'. Falling back to global chooser."
            )
        except Exception:
            pass

        import math, random, itertools, os

        # Log env-driven behaviour once so runs are easy to audit from run.log.
        if not getattr(self, "_logged_fallback_top1_env", False):
            self._logged_fallback_top1_env = True
            raw_val = os.getenv("KERNELAGENT_DB_FALLBACK_TOP1", None)
            parsed_val = os.getenv("KERNELAGENT_DB_FALLBACK_TOP1", "0") in (
                "1",
                "true",
                "True",
                "yes",
                "YES",
                "y",
                "on",
                "ON",
            )
            msg = f"KERNELAGENT_DB_FALLBACK_TOP1={raw_val!r} (parsed={parsed_val})"
            if hasattr(self.llm_interface, "logger") and self.llm_interface.logger:
                self.llm_interface.logger.info(msg)
            else:
                print(msg)

        # Use the global optimisation pool for the fallback path as well.
        optimizations = [
            opt
            for state_data in self.optimization_strategies.values()
            for opt in state_data.get("optimizations", [])
        ]
        composite_opts = (
            list(itertools.chain.from_iterable(self.composite_optimizations.values()))
            if include_composite
            else []
        )

        if exclude_used:
            optimizations = [opt for opt in optimizations if getattr(opt, "usage_count", 0) == 0]
            composite_opts = [opt for opt in composite_opts if getattr(opt, "usage_count", 0) == 0]

        if not optimizations and not composite_opts:
            return None

        def score_optimization(opt) -> float:
            pred_speedup = getattr(opt, "predicted_speedup", None)
            if pred_speedup in (None, 0.0):
                pred_impr = (getattr(opt, "predicted_improvement", 0.0) or 0.0)
                pred_speedup = 1.0 / max(1e-6, 1.0 - (pred_impr / 100.0))
            base_score = pred_speedup * getattr(opt, "confidence_score", 0.5)
            usage_penalty = min(getattr(opt, "usage_count", 0) * 0.1, 0.5)
            composite_bonus = 0.1 if isinstance(opt, CompositeOptimization) else 0
            return base_score - usage_penalty + composite_bonus

        all_opts = optimizations + composite_opts

        scores = [score_optimization(o) for o in all_opts]

        if all(abs(s) < 1e-6 for s in scores):
            scores = [1.0 for _ in scores]

        # Temperature for exploration (smaller => greedier)
        tau = 0.5  # can be exposed as parameter later

        max_s = max(scores)
        exp_scores = [math.exp((s - max_s) / max(tau, 1e-6)) for s in scores]
        total = sum(exp_scores)
        probs = [es / total for es in exp_scores]

        # Optional deterministic fallback for debugging/repro.
        # If set, we choose the single best-scoring optimisation instead of sampling.
        # Env var is intentionally scoped to this fallback path (LLM recommendations are unchanged).
        force_top1 = os.getenv("KERNELAGENT_DB_FALLBACK_TOP1", "0") in (
            "1",
            "true",
            "True",
            "yes",
            "YES",
            "y",
            "on",
            "ON",
        )
        if force_top1:
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            return all_opts[best_idx]

        # Random choice according to probabilities – eliminates order bias
        chosen_opt = random.choices(all_opts, weights=probs, k=1)[0]
        return chosen_opt
        
    def _categorize_technique(self, technique: str) -> str:
        """Categorize optimization technique by type."""
        technique_lower = technique.lower()
        if any(term in technique_lower for term in ['memory', 'coalesced', 'cache', 'tiling']):
            return 'memory'
        elif any(term in technique_lower for term in ['compute', 'tensor', 'instruction']):
            return 'compute'
        elif any(term in technique_lower for term in ['occupancy', 'thread', 'block']):
            return 'latency'
        else:
            return 'general'
        
    def _create_default_strategies_for_bottleneck(self, bottleneck_type: str) -> List[OptimizationEntry]:
        """
        Create default optimization strategies for a given bottleneck type.
        
        Args:
            bottleneck_type: The type of bottleneck (memory_bound, compute_bound, latency_bound, hybrid_bound)
            
        Returns:
            List of OptimizationEntry objects with default strategies
        """
        
        default_strategies = []
        
        if bottleneck_type == "memory_bound":
            default_strategies = [
                OptimizationEntry(
                    technique="memory_coalescing_optimization",
                    predicted_improvement=20.0,
                    description="Optimize memory access patterns for coalesced reads/writes",
                    category="memory",
                    confidence_score=0.8,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="shared_memory_tiling",
                    predicted_improvement=25.0,
                    description="Use shared memory tiling to reduce global memory accesses",
                    category="memory",
                    confidence_score=0.7,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="vectorized_memory_access",
                    predicted_improvement=15.0,
                    description="Use vectorized loads/stores to improve memory bandwidth utilization",
                    category="memory",
                    confidence_score=0.6,
                    last_updated=datetime.now().isoformat()
                )
            ]
        
        elif bottleneck_type == "compute_bound":
            default_strategies = [
                OptimizationEntry(
                    technique="instruction_level_parallelism",
                    predicted_improvement=30.0,
                    description="Optimize instruction scheduling and parallelism",
                    category="compute",
                    confidence_score=0.8,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="fast_math_optimization",
                    predicted_improvement=20.0,
                    description="Use fast math operations where precision allows",
                    category="compute",
                    confidence_score=0.7,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="tensor_core_utilization",
                    predicted_improvement=40.0,
                    description="Utilize tensor cores for supported operations",
                    category="compute",
                    confidence_score=0.6,
                    last_updated=datetime.now().isoformat()
                )
            ]
        
        elif bottleneck_type == "latency_bound":
            default_strategies = [
                OptimizationEntry(
                    technique="occupancy_optimization",
                    predicted_improvement=35.0,
                    description="Optimize thread block size and resource usage for higher occupancy",
                    category="latency",
                    confidence_score=0.8,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="register_pressure_reduction",
                    predicted_improvement=30.0,
                    description="Reduce register usage to improve occupancy",
                    category="latency",
                    confidence_score=0.7,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="work_per_thread_increase",
                    predicted_improvement=25.0,
                    description="Increase work per thread to hide latency",
                    category="latency",
                    confidence_score=0.6,
                    last_updated=datetime.now().isoformat()
                )
            ]
        
        elif bottleneck_type == "hybrid_bound":
            default_strategies = [
                OptimizationEntry(
                    technique="memory_compute_overlap",
                    predicted_improvement=40.0,
                    description="Overlap memory operations with compute to hide latency",
                    category="general",
                    confidence_score=0.8,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="algorithmic_optimization",
                    predicted_improvement=35.0,
                    description="Optimize algorithm structure for better resource utilization",
                    category="general",
                    confidence_score=0.7,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="adaptive_block_sizing",
                    predicted_improvement=30.0,
                    description="Dynamically adjust block size based on workload characteristics",
                    category="general",
                    confidence_score=0.6,
                    last_updated=datetime.now().isoformat()
                )
            ]
        
        else:
            # Fallback strategies for unknown bottleneck types
            default_strategies = [
                OptimizationEntry(
                    technique="general_optimization",
                    predicted_improvement=20.0,
                    description="General optimization techniques",
                    category="general",
                    confidence_score=0.5,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="performance_tuning",
                    predicted_improvement=15.0,
                    description="Basic performance tuning",
                    category="general",
                    confidence_score=0.4,
                    last_updated=datetime.now().isoformat()
                )
            ]
        
        return default_strategies
    
    def _load_composite_optimizations(self, json_data: dict):
        """Load composite optimizations from JSON data."""
        for adjustment in json_data.get("prediction_adjustments", []):
            state = adjustment["state"]
            if state not in self.composite_optimizations:
                self.composite_optimizations[state] = []
            
            composite = CompositeOptimization(
                state=state,
                technique1=adjustment["technique1"],
                technique2=adjustment.get("technique2"),
                technique3=adjustment.get("technique3"),
                order_of_techniques=adjustment.get("order_of_techniques", []),
                parameters_to_fine_tune=adjustment.get("parameters_to_fine_tune", {}),
                predicted_improvement=adjustment.get("new_predicted_improvement", 0.0),
                reason=adjustment.get("reason", ""),
                side_effects=adjustment.get("side_effects", "")
            )
            self.composite_optimizations[state].append(composite)

    def update_optimization_result(self, state: str, technique: str, actual_improvement: float,
                                    current_file_path: Optional[Path] = None):
            # Log the update attempt
            # Update the optimization entry with actual results
            if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                self.llm_interface.logger.info(f"Attempting to update optimization result for {technique} in state {state} with actual improvement {actual_improvement}")
            if state in self.optimization_strategies:
                for opt in self.optimization_strategies[state].get("optimizations", []):
                    if opt.technique == technique:
                        prev_usage = opt.usage_count
                        new_usage = prev_usage + 1
                        # Store the most recent measurement
                        opt.actual_improvement = actual_improvement
                        opt.usage_count = new_usage
                        opt.last_updated = datetime.now().isoformat()

                        self.llm_interface.logger.info(f"Updating database entry for {technique} in state {state}")
                        # ----------------- Calculate speedup -----------------
                        speedup_of_cur_optimization = 1.0  # Default to no speedup
                        if current_file_path:
                            try:
                                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                    self.llm_interface.logger.info(
                                        f"Speedup calc: current_file_path={current_file_path}"
                                    )
                                # For first iteration, get initial baseline from files
                                if opt.initial_elapsed_cycles is None:
                                    baseline_ncu = current_file_path.parent / "ncu/0_init_ncu_log.txt"
                                    init_cu = current_file_path.parent / "ncu_annot/init.cu"
                                    if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                        self.llm_interface.logger.info(
                                            f"Speedup calc: baseline paths exist? ncu={baseline_ncu.exists()} init_cu={init_cu.exists()}"
                                        )
                                    if baseline_ncu.exists():
                                        initial_text = baseline_ncu.read_text()
                                        used_path = baseline_ncu
                                    else:
                                        initial_text = init_cu.read_text()
                                        used_path = init_cu
                                    opt.initial_elapsed_cycles = get_elapsed_cycles_v2(initial_text)
                                    if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                        self.llm_interface.logger.info(
                                            f"Speedup calc: parsed initial_elapsed_cycles={opt.initial_elapsed_cycles} from {used_path}"
                                        )
                                # Calculate speedup using the passed in actual_improvement as current elapsed cycles
                                # and the stored initial_elapsed_cycles as baseline
                                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                    self.llm_interface.logger.info(
                                        f"Speedup calc: actual_improvement arg value={actual_improvement} (type={type(actual_improvement)})"
                                    )
                                current_elapsed_cycles = int(actual_improvement)
                                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                    self.llm_interface.logger.info(
                                        f"Speedup calc: current_elapsed_cycles={current_elapsed_cycles}, baseline={opt.initial_elapsed_cycles}"
                                    )
                                if current_elapsed_cycles <= 0:
                                    raise ValueError(f"Non-positive current_elapsed_cycles={current_elapsed_cycles}")
                                speedup_of_cur_optimization = abs(float(opt.initial_elapsed_cycles) / float(current_elapsed_cycles))
                                opt.actual_speedup = speedup_of_cur_optimization
                            except (ValueError, FileNotFoundError, AttributeError) as e:
                                # Fall back to no speedup calculation if files can't be read
                                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                    self.llm_interface.logger.warning(f"Could not calculate speedup (baseline flow): {e}")
                                speedup_of_cur_optimization = 1.0
                        else:
                            # No file path provided, cannot read baseline. Attempt to infer from percent improvement if applicable.
                            if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                self.llm_interface.logger.info("Speedup calc: no current_file_path provided; attempting percent-based inference.")
                            try:
                                denom = 1.0 - (float(actual_improvement) / 100.0)
                                if abs(denom) < 1e-6:
                                    denom = 1e-6 if denom >= 0 else -1e-6
                                inferred_speedup = abs(1.0 / denom)
                                # Use inferred speedup as measured value when baseline file path is unavailable
                                speedup_of_cur_optimization = inferred_speedup
                                opt.actual_speedup = inferred_speedup
                                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                    self.llm_interface.logger.info(
                                        f"Speedup calc: inferred speedup from percent improvement={inferred_speedup:.4f}x (actual_improvement={actual_improvement})"
                                    )
                            except Exception as e:
                                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                    self.llm_interface.logger.info(
                                        f"Speedup calc: percent-based inference failed: {e} (actual_improvement={actual_improvement})"
                                    )
                        # Log measured speedups
                        if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                            self.llm_interface.logger.info(
                                f"Measured speedup for {technique} in state {state}: {speedup_of_cur_optimization:.4f}x"
                            )
                        # ----------------- Update predicted_speedup using rolling average -------------
                        # Use a running weighted average for speedup tracking (separate from predicted_improvement)
                        if opt.predicted_speedup is None:
                            opt.predicted_speedup = 1.0
                        # Inverse weighting: give more weight to later iterations
                        # cur_num_iter is the current usage count (1-based)
                        cur_num_iter = new_usage
                        # inverse_weight = 1.0 / max(1, 100 - cur_num_iter)
                        try:
                        #     numerator = opt.predicted_speedup * float(prev_usage) + speedup_of_cur_optimization * inverse_weight
                        #     denom = max(float(prev_usage) + inverse_weight, 1e-6)
                            numerator = opt.predicted_speedup * float(prev_usage) + speedup_of_cur_optimization 
                            denom = max(float(new_usage), 1e-6)

                            opt.predicted_speedup = numerator / denom
                        except ZeroDivisionError:
                            # Should never happen, but guard just in case.
                            opt.predicted_speedup = speedup_of_cur_optimization
                        # Log predicted vs actual speedup and weights
                        if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                            self.llm_interface.logger.info(
                                f"Predicted speedup now {opt.predicted_speedup:.4f}x; actual speedup {getattr(opt, 'actual_speedup', None)} | prev_usage={prev_usage}"
                            )
                        # ----------------- Update predicted_improvement -------------
                        # Use a running weighted average where the existing
                        # predicted value is treated as the mean of *prev_usage*
                        # historical data points.  This keeps the value stable
                        # yet allows it to converge as more real data arrives.
                        if opt.predicted_improvement is None:
                            opt.predicted_improvement = 0.0
                        try:
                            opt.predicted_improvement = (
                                opt.predicted_improvement * prev_usage + actual_improvement
                            ) / max(new_usage, 1)
                        except ZeroDivisionError:
                            # Should never happen, but guard just in case.
                            opt.predicted_improvement = actual_improvement
                        # ----------------- Adjust confidence score -----------------
                        # Compute accuracy using the *previous* prediction so the
                        # adjustment reflects the quality of that prior estimate.
                        if prev_usage > 0 and opt.predicted_improvement > 0:
                            # Use the *old* predicted value (before update) which
                            # is   (new_pred * new_usage - actual) / prev_usage
                            prev_pred = (
                                opt.predicted_improvement * new_usage - actual_improvement
                            ) / max(prev_usage, 1)
                            accuracy = actual_improvement / prev_pred if prev_pred else 0.0
                            if 0.8 <= accuracy <= 1.2:  # Good prediction (±20%)
                                opt.confidence_score = min(1.0, opt.confidence_score + 0.1)
                            else:  # Poor prediction
                                opt.confidence_score = max(0.1, opt.confidence_score - 0.1)
                        # -- Log change --------------------------------------------------
                        self._log_db_change(
                            "update_optimization_result",
                            {
                                "state": state,
                                "technique": technique,
                                "actual_improvement": actual_improvement,
                                "predicted_improvement": opt.predicted_improvement,
                                "confidence_score": opt.confidence_score,
                                "usage_count": opt.usage_count,
                                "speedup_of_cur_optimization": speedup_of_cur_optimization,
                                "predicted_speedup": opt.predicted_speedup,
                                "actual_speedup": opt.actual_speedup,
                                "initial_elapsed_cycles": opt.initial_elapsed_cycles,
                            },
                        )
                        # For first-use entries we leave confidence as is.
                        break

    def update_composite_optimization_result(self, state: str, composite_id: str, actual_improvement: float):
        """Update composite optimization result for tracking and learning."""
        if state in self.composite_optimizations:
            for comp_opt in self.composite_optimizations[state]:
                if comp_opt.get_composite_id() == composite_id:
                    comp_opt.actual_improvement = actual_improvement
                    comp_opt.usage_count += 1
                    comp_opt.last_updated = datetime.now().isoformat()
                    # Update confidence score
                    if comp_opt.predicted_improvement > 0:
                        accuracy = actual_improvement / comp_opt.predicted_improvement
                        if 0.8 <= accuracy <= 1.2:
                            comp_opt.confidence_score = min(1.0, comp_opt.confidence_score + 0.1)
                        else:
                            comp_opt.confidence_score = max(0.1, comp_opt.confidence_score - 0.1)
                    # -- Log change --------------------------------------------------
                    self._log_db_change(
                        "update_composite_optimization_result",
                        {
                            "state": state,
                            "composite_id": composite_id,
                            "actual_improvement": actual_improvement,
                            "predicted_improvement": comp_opt.predicted_improvement,
                            "confidence_score": comp_opt.confidence_score,
                            "usage_count": comp_opt.usage_count,
                        },
                    )
                    break

    def add_composite_optimization(self, composite: CompositeOptimization):
        """Add a composite optimization to the database."""
        state = composite.state
        if state not in self.composite_optimizations:
            self.composite_optimizations[state] = []
        
        # Check if this composite already exists
        for existing_comp in self.composite_optimizations[state]:
            if existing_comp.get_composite_id() == composite.get_composite_id():
                # Update existing composite
                existing_comp.predicted_improvement = composite.predicted_improvement
                existing_comp.reason = composite.reason
                existing_comp.side_effects = composite.side_effects
                existing_comp.last_updated = datetime.now().isoformat()

                # Log update event
                self._log_db_change(
                    "update_composite_optimization",
                    {
                        "state": state,
                        "composite_id": existing_comp.get_composite_id(),
                        "predicted_improvement": existing_comp.predicted_improvement,
                    },
                )
                return
        
        # Add new composite optimization
        composite.last_updated = datetime.now().isoformat()
        self.composite_optimizations[state].append(composite)

        # Log creation event
        self._log_db_change(
            "add_composite_optimization",
            {
                "state": state,
                "composite_id": composite.get_composite_id(),
                "predicted_improvement": composite.predicted_improvement,
            },
        )

    def add_new_optimization(self, state: str, technique: str, predicted_improvement: float):
        """Add a new optimization technique to the database."""
        if state not in self.optimization_strategies:
            self.optimization_strategies[state] = {"optimizations": []}
        
        # Check if this technique already exists for this state
        for existing_opt in self.optimization_strategies[state].get("optimizations", []):
            if existing_opt.technique == technique:
                # Update existing optimization
                existing_opt.predicted_improvement = predicted_improvement
                existing_opt.last_updated = datetime.now().isoformat()

                self._log_db_change(
                    "update_optimization",
                    {
                        "state": state,
                        "technique": technique,
                        "predicted_improvement": predicted_improvement,
                    },
                )
                return
        
        # Add new optimization
        new_opt = OptimizationEntry(
            technique=technique,
            predicted_improvement=predicted_improvement,
            category=self._categorize_technique(technique),
            last_updated=datetime.now().isoformat()
        )
        self.optimization_strategies[state]["optimizations"].append(new_opt)

        self._log_db_change(
            "add_new_optimization",
            {
                "state": state,
                "technique": technique,
                "predicted_improvement": predicted_improvement,
            },
        )

    def create_parameter_tuned_optimization(self, base_technique: str, parameters: Dict[str, Any], 
                                          predicted_improvement: float, reason: str = "") -> str:
        """Create a parameter-tuned optimization technique name."""
        # Create a unique technique name that includes parameters
        param_str = "_".join(f"{k}_{v}" for k, v in parameters.items())
        tuned_technique = f"{base_technique}_tuned_{param_str}"
        
        # Store parameter information for later use
        if not hasattr(self, 'parameter_tuned_techniques'):
            self.parameter_tuned_techniques = {}
        
        self.parameter_tuned_techniques[tuned_technique] = {
            "base_technique": base_technique,
            "parameters": parameters,
            "predicted_improvement": predicted_improvement,
            "reason": reason,
            "created_at": datetime.now().isoformat()
        }

        # Log tuned technique creation
        self._log_db_change(
            "create_parameter_tuned_optimization",
            {
                "tuned_technique": tuned_technique,
                "base_technique": base_technique,
                "parameters": parameters,
                "predicted_improvement": predicted_improvement,
            },
        )
        
        return tuned_technique

    # Legacy compatibility
    def get_state_from_metrics(self, metrics: dict, performance_pattern: str = "") -> str:
        """Legacy method - use get_state_from_ncu_report instead."""
        print("Warning: get_state_from_metrics is deprecated. Use get_state_from_ncu_report instead.")
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.get_state_from_ncu_report(performance_pattern, metrics))
        except:
            return self._fallback_state_matching(self._fallback_state_analysis(performance_pattern, metrics))

    def get_database_stats(self) -> dict:
        """Get statistics about the optimization database."""
        
        # Count total optimizations
        total_optimizations = 0
        total_composite_optimizations = 0
        state_counts = {}
        
        for state, state_data in self.optimization_strategies.items():
            opts = state_data.get("optimizations", [])
            total_optimizations += len(opts)
            state_counts[state] = len(opts)
        
        for state, composite_opts in self.composite_optimizations.items():
            total_composite_optimizations += len(composite_opts)
            if state in state_counts:
                state_counts[state] += len(composite_opts)
            else:
                state_counts[state] = len(composite_opts)
        
        # Calculate average improvements
        improvements = []
        for state_data in self.optimization_strategies.values():
            for opt in state_data.get("optimizations", []):
                if opt.actual_improvement is not None:
                    improvements.append(opt.actual_improvement)
        
        for state_opts in self.composite_optimizations.values():
            for opt in state_opts:
                if opt.actual_improvement is not None:
                    improvements.append(opt.actual_improvement)
        
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0
        
        return {
            "total_states": len(self.optimization_strategies),
            "total_optimizations": total_optimizations,
            "total_composite_optimizations": total_composite_optimizations,
            "state_counts": state_counts,
            "average_improvement": avg_improvement,
            "total_measured_optimizations": len(improvements),
            "last_updated": datetime.now().isoformat(),
            "database_health": "healthy" if total_optimizations > 0 else "empty"
        }


# Maintain backward compatibility
OptimizationDatabase = GPUOptimizationDatabase 


def print_database_summary(db: GPUOptimizationDatabase):
    """Print a human-readable summary of the optimisation database."""
    import json as _json

    print("\n=== Optimisation Database Summary ===\n")
    if not db.optimization_strategies:
        print("No optimisation strategies loaded.")
        return

    for state, state_data in db.optimization_strategies.items():
        print(f"State: {state}")
        opts = state_data.get("optimizations", [])
        if opts:
            for opt in opts:
                imp = getattr(opt, "predicted_improvement", 0.0) or 0.0
                print(f"  - {opt.technique}: {imp}% predicted improvement")
        else:
            print("  (no optimisation strategies)")
        print()

    # Optional: print simple database stats
    try:
        stats = db.get_database_stats()
        print("Database statistics:\n" + _json.dumps(stats, indent=2))
    except Exception as exc:
        print(f"Could not compute database stats: {exc}")



def _main():
    """Entry-point for quick manual testing via the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load an optimisation database markdown file and list states/optimisations."
    )
    parser.add_argument(
        "--optimization_db",
        help="Path to the optimisation database markdown file (.md)",
    )
    parser.add_argument(
        "--gpu_report",
        default="",
        help="Optional path to a GPU optimisation knowledge report (text/markdown)",
    )
    parser.add_argument(
        "--prompt_file",
        default="",
        help="Optional path to a text file containing a prompt to send to the LLM for quick testing",
    )
    # ------------ Development test for analyse+plan workflow ------------
    parser.add_argument(
        "--ncu_report_file",
        default="",
        help="Path to an NSight Compute (NCU) report text file used for test analysis",
    )
    parser.add_argument(
        "--code_impl_file",
        default="",
        help="Path to a CUDA/C++ kernel implementation file for optimisation planning",
    )
    parser.add_argument(
        "--metrics_json",
        default="",
        help="Optional path to a JSON file providing numeric metrics for the NCU report test",
    )
    parser.add_argument(
        "--regenerate_from_json",
        action="store_true",
        help="Regenerate the optimisation database markdown using the persisted JSON snapshot and optional header/footer",
    )
    args = parser.parse_args()

    optimisation_db_path = Path(args.optimization_db).expanduser().resolve()
    gpu_report_path = (
        Path(args.gpu_report).expanduser().resolve() if args.gpu_report else Path("/dev/null")
    )

    if not optimisation_db_path.exists():
        raise FileNotFoundError(f"Optimisation database file not found: {optimisation_db_path}")

    db = GPUOptimizationDatabase(
        optimization_db_path=optimisation_db_path,
        gpu_report_path=gpu_report_path,
        llm_interface=LLMInterface(),
    )

    print_database_summary(db)

    # ------------------------------------------------------------
    # Test helper: regenerate markdown from JSON and show a preview
    # ------------------------------------------------------------
    if getattr(args, "regenerate_from_json", False):
        print("\n>>> Testing markdown regeneration from JSON snapshot...\n")
        db._regenerate_database_from_json()
        try:
            md_text = db.optimization_db_path.read_text(encoding="utf-8")
            print(f"Regenerated markdown written to: {db.optimization_db_path}")
            preview_lines = md_text.splitlines()
            preview = "\n".join(preview_lines[: min(25, len(preview_lines))])
            print("\n--- Preview (first 25 lines) ---\n" + preview + "\n--- End preview ---\n")
        except Exception as exc:
            print(f"Could not read regenerated markdown: {exc}")

    # ------------------------------------------------------------
    # Quick test: print the available-optimisations summary after
    # the database has been loaded.  This helps developers verify
    # that techniques are parsed correctly and the helper renders
    # them in the expected format.
    # ------------------------------------------------------------
    optim_summary = db._build_available_optimisations_summary()
    print("\n=== Available Optimisations Summary ===\n")
    if optim_summary.strip():
        print(optim_summary)
    else:
        print("(no optimisation techniques found)")

    # ==============================================================
    # Developer test: end-to-end analyse_performance_state +
    # generate_optimization_plan flow.
    # ==============================================================
    if args.ncu_report_file and args.code_impl_file:
        ncu_path = Path(args.ncu_report_file).expanduser().resolve()
        code_path = Path(args.code_impl_file).expanduser().resolve()

        if not ncu_path.exists():
            print(f"NCU report file not found: {ncu_path}")
        elif not code_path.exists():
            print(f"Code implementation file not found: {code_path}")
        else:
            ncu_report_text = ncu_path.read_text(encoding="utf-8")
            code_impl_text = code_path.read_text(encoding="utf-8")

            # Load optional metrics JSON
            metrics: dict = {}
            if args.metrics_json:
                metrics_path = Path(args.metrics_json).expanduser().resolve()
                if metrics_path.exists():
                    import json as _json
                    try:
                        metrics = _json.loads(metrics_path.read_text())
                    except Exception as exc:
                        print(f"Could not parse metrics JSON ({metrics_path}): {exc}")

            import asyncio, json as _json

            async def _run_flow():
                print("\n>>> Running analyse_performance_state...")
                profile = await db.analyze_performance_state(
                    ncu_report_text, metrics, code_impl_text
                )

                # Represent the analysis as JSON for the plan generator
                analysis_json_str = _json.dumps(asdict(profile), indent=2)

                print("Analysis result:\n" + analysis_json_str + "\n")

                print(">>> Generating optimisation plan...\n")
                plan = await db.generate_optimization_plan(
                    analysis_json_str, code_impl_text
                )

                print("Optimisation plan (top suggestions):\n" + _json.dumps(plan, indent=2))

            try:
                asyncio.run(_run_flow())
            except RuntimeError:
                # If we're already inside an asyncio loop (e.g. Jupyter) fall back
                loop = asyncio.get_event_loop()
                loop.run_until_complete(_run_flow())

    # --------------------------------------------------------------
    # Optional quick test: send user-supplied prompt to the LLM and
    # print the raw response.  This is **not** part of the normal
    # optimisation workflow – it is merely a convenience helper for
    # developers who want to sanity-check LLM connectivity.
    # --------------------------------------------------------------
    if args.prompt_file:
        prompt_path = Path(args.prompt_file).expanduser().resolve()
        if not prompt_path.exists():
            print(f"Prompt file not found: {prompt_path}")
        elif not db.llm_interface.is_available():
            print("LLM interface is not available – skipping test query.")
        else:
            prompt_text = prompt_path.read_text(encoding="utf-8")
            print("\n=== Prompt to LLM ===\n" + prompt_text.strip() + "\n")
            print("Querying LLM... (this may take a moment)\n")
            response = db.llm_interface.query_sync(prompt_text, max_tokens=800, temperature=0.1)
            print("=== LLM response ===\n" + response.strip() + "\n")


if __name__ == "__main__":
    _main() 