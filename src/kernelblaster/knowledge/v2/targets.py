# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The scope tree: an open-ended hierarchy of scope nodes with targets as leaves.

    universal
    ├── simt_gpu ── nvidia_gpu ── turing (cuda:sm_75), ampere, ada (cuda:sm_89), hopper
    │           └── adreno (opencl:adreno650, opencl:adreno740)
    ├── vector_cpu ── rvv (rvv:vlen128, rvv:vlen256, rvv:vlen512)
    └── systolic ── gemmini (gemmini:default)

Every node has an id, a parent and a description; the root is ``universal``.
Targets hang off any node. Nodes and targets are data (``targets.json``) and
can be added at run time (:meth:`TargetRegistry.add_node` / ``add_target``;
``KnowledgeV2`` persists them in the DB ``scopes`` / ``targets`` tables).

A *target key* is ``<backend>:<arch>`` (``cuda:sm_89``, ``opencl:adreno740``).
An unknown target is placed under the node its backend maps to (the
``backends`` map in ``targets.json``, data not code) and has no peaks until
someone provides them (the roofline adapter then leaves utilisations absent).

Knowledge is scoped to one *scope id*: a target key, a node id or
``universal``; it applies to every target whose chain (target -> parent ->
... -> universal) contains that id. The v1 constants stay as aliases:
``level="family"`` with ``scope=X`` means node X; ``level="universal"`` is
the root.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

UNIVERSAL, NODE, TARGET = "universal", "node", "target"
FAMILY = "family"                               # v1 alias of NODE (old DBs, claims, seeds)
LEVELS = (TARGET, NODE, UNIVERSAL)
ROOT = UNIVERSAL

#: seed data directory (``KB_KNOWLEDGE_SEED_DIR`` overrides it, e.g. to build a DB from another seed)
REPO_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "kernelblaster" / "knowledge"
DATA_DIR = Path(os.environ.get("KB_KNOWLEDGE_SEED_DIR") or REPO_DATA_DIR)


def empty_seed() -> bool:
    """``KB_EMPTY_SEED=1`` (default off): no built-in knowledge at all. No curated techniques / states (whatever
    ``DATA_DIR`` holds), no legacy technique DB / footer techniques, no recurrent-class seeds; only the hardware
    description (``targets.json``) loads. ``KnowledgeV2.select`` then falls back to a free-form step when no
    technique applies (instead of raising)."""
    return os.environ.get("KB_EMPTY_SEED", "0").strip().lower() in ("1", "on", "true", "yes")


def target_key(target: str, arch: Optional[str] = None) -> str:
    """``("cuda", "sm_75")`` / ``("cuda", "75")`` -> ``cuda:sm_75``; keys pass through."""
    if ":" in (target or ""):
        return target
    a = arch or "unknown"
    if target == "cuda" and a.isdigit():
        a = f"sm_{a}"
    return f"{target}:{a}"


@dataclass
class Node:
    id: str
    parent: Optional[str]
    description: str = ""
    kappa: Optional[float] = None               # shrinkage strength of this node toward its parent

    def to_dict(self) -> dict:
        return {"id": self.id, "parent": self.parent, "description": self.description, "kappa": self.kappa}


@dataclass
class Target:
    key: str
    node: str                                   # parent scope node
    name: str = ""
    peak_gflops: Optional[float] = None
    peak_gbps: Optional[float] = None
    sms: int = 1
    threads_per_sm: int = 1024
    warp: int = 32
    smem_kb: Optional[int] = None
    l2_kb: Optional[int] = None
    launch_us: float = 3.0
    profiler: str = "roofline"
    aliases: list = field(default_factory=list)
    kappa: Optional[float] = None
    limits: dict = field(default_factory=dict)  # resource limits (regs, smem, tensor cores, VLEN, scratchpad ...)
    family: str = ""                            # top node under universal (filled by the registry)

    @property
    def backend(self) -> str:
        return self.key.split(":", 1)[0]

    @property
    def width(self) -> int:
        """Parallel lanes needed to fill the machine once."""
        return max(1, self.sms * self.threads_per_sm)

    @property
    def ridge(self) -> Optional[float]:
        if self.peak_gflops and self.peak_gbps:
            return self.peak_gflops / self.peak_gbps
        return None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("family", None)
        return d


_TARGET_FIELDS = set(Target.__dataclass_fields__) - {"key"}


class TargetRegistry:
    def __init__(self, path: Optional[Path] = None):
        path = Path(path) if path else DATA_DIR / "targets.json"
        d = json.loads(path.read_text()) if path.exists() else {}
        self.nodes: dict[str, Node] = {ROOT: Node(ROOT, None, "everything")}
        self.backends: dict[str, str] = dict(d.get("backends", {}))
        self.targets: dict[str, Target] = {}
        self._alias: dict[str, str] = {}
        for nid, n in (d.get("families") or {}).items():          # v1 format: families under universal
            self.nodes.setdefault(nid, Node(nid, ROOT, n.get("description", "")))
            for b in n.get("backends", []):
                self.backends.setdefault(b, nid)
        pending = dict(d.get("scopes") or {})
        while pending:                                            # parents before children
            ready = {k: v for k, v in pending.items() if (v.get("parent") or ROOT) in self.nodes}
            if not ready:
                raise ValueError(f"scope nodes with unknown parents: {sorted(pending)}")
            for nid, n in ready.items():
                if nid != ROOT:
                    self.nodes[nid] = Node(nid, n.get("parent") or ROOT, n.get("description", ""), n.get("kappa"))
                del pending[nid]
        for k, t in (d.get("targets") or {}).items():
            self.add_target(k, **t)

    # -- structure ------------------------------------------------------------
    def add_node(self, nid: str, parent: str = ROOT, description: str = "", kappa: Optional[float] = None) -> Node:
        if nid in self.targets:
            raise ValueError(f"{nid!r} is a target key, not a node id")
        if parent not in self.nodes:
            raise ValueError(f"node {nid!r}: unknown parent {parent!r}")
        if nid == ROOT:
            return self.nodes[ROOT]
        n = self.nodes.get(nid)
        if n is None:
            n = self.nodes[nid] = Node(nid, parent, description, kappa)
        elif n.parent != parent and nid in self.ancestors(parent):
            raise ValueError(f"node {nid!r}: parent {parent!r} would make a cycle")
        else:
            n.parent, n.description, n.kappa = parent, description or n.description, kappa or n.kappa
        for t in self.targets.values():          # top-level family may have changed
            t.family = self._top(t.node)
        return n

    def add_target(self, key: str, node: Optional[str] = None, **info) -> Target:
        node = node or info.pop("family", None) or self.node_for_backend(key)
        info.pop("family", None)
        if node not in self.nodes:
            raise ValueError(f"target {key!r}: unknown node {node!r}")
        t = Target(key=key, node=node, **{k: v for k, v in info.items() if k in _TARGET_FIELDS})
        t.family = self._top(node)
        self.targets[key] = t
        for a in t.aliases:
            self._alias[a] = key
        return t

    def node_for_backend(self, key: str) -> str:
        return self.backends.get(key.split(":", 1)[0], ROOT)

    def get(self, key: str, arch: Optional[str] = None) -> Target:
        key = target_key(key, arch)
        key = self._alias.get(key, key)
        t = self.targets.get(key)
        if t is None:                            # unknown target: under its backend's node
            t = self.add_target(key, self.node_for_backend(key))
        return t

    def ensure(self, key: str, *, peaks: Optional[tuple] = None) -> Target:
        """The target for ``key``; fill missing (GFLOP/s, GB/s) peaks, e.g. from the agent's TargetProfile."""
        t = self.get(key)
        if peaks and not (t.peak_gflops and t.peak_gbps):
            t.peak_gflops, t.peak_gbps = float(peaks[0]), float(peaks[1])
        return t

    # -- chains -----------------------------------------------------------------
    def ancestors(self, nid: str) -> list[str]:
        """``nid`` and its parents up to universal."""
        out, cur = [], nid
        while cur is not None and cur in self.nodes and cur not in out:
            out.append(cur)
            cur = self.nodes[cur].parent
        return out

    def chain(self, key: str) -> list[str]:
        """Scope ids from the target up: [target key, node, ..., universal]."""
        t = self.get(key)
        return [t.key] + self.ancestors(t.node)

    def _top(self, nid: str) -> str:
        a = self.ancestors(nid)
        return a[-2] if len(a) >= 2 else ROOT

    def family_of(self, key: str) -> str:
        """v1 alias: the target's top node under universal (e.g. simt_gpu)."""
        return self.get(key).family

    def scope_id(self, level: Optional[str], scope: Optional[str]) -> str:
        """(level, scope) in any vocabulary -> one scope id (target key | node id | universal)."""
        if level == UNIVERSAL or (level is None and scope is None):
            return ROOT
        if level in (FAMILY, NODE):
            return scope or ROOT
        if scope is None:
            return ROOT
        return self.get(scope).key if ":" in scope else scope

    def level_of(self, sid: str) -> str:
        return UNIVERSAL if sid == ROOT else NODE if sid in self.nodes else TARGET

    def covers(self, level: Optional[str], scope: Optional[str], key: str) -> bool:
        """Does knowledge at (level, scope) apply to target ``key``?"""
        return self.scope_id(level, scope) in self.chain(key)

    def depth(self, sid: str) -> int:
        """Edges from universal (targets are one below their node)."""
        if sid in self.nodes:
            return len(self.ancestors(sid)) - 1
        return len(self.chain(sid)) - 1

    def lca(self, keys: Iterable[str]) -> str:
        """Lowest common scope of targets / nodes: a single target is its own scope."""
        chains = [self.chain(k) if k not in self.nodes else self.ancestors(k) for k in dict.fromkeys(keys)]
        if not chains:
            return ROOT
        common = set(chains[0]).intersection(*map(set, chains[1:]))
        return next(s for s in chains[0] if s in common)

    def is_above(self, a: str, b: str) -> bool:
        """Is scope ``a`` a strict ancestor of scope ``b``?"""
        up = self.ancestors(b) if b in self.nodes else self.chain(b)
        return a != b and a in up

    def label(self, sid: str) -> str:
        """Prompt label: ``sm_89 only`` / ``nvidia_gpu (3 targets)`` / ``general lesson``."""
        if sid == ROOT:
            return "general lesson"
        if sid in self.nodes:
            n = sum(1 for t in self.targets if sid in self.chain(t))
            return f"{sid} ({n} known target{'s' if n != 1 else ''})"
        return f"{sid.split(':', 1)[-1]} only"

    def scopes(self, key: str) -> list[tuple[str, Optional[str]]]:
        """(level, scope) from the target up."""
        return [(self.level_of(s), None if s == ROOT else s) for s in self.chain(key)]

    def kappa_of(self, sid: str, default: float) -> float:
        k = (self.nodes[sid].kappa if sid in self.nodes else
             self.targets[sid].kappa if sid in self.targets else None)
        return float(k) if k else default
