# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hierarchical reward statistics with shrinkage (design section 7).

Statistics are kept per (item, node) where an item is a technique or
hypothesis id and a node is (level, scope, op, state). One outcome updates
every node on its path, most specific first: the target, then every scope
node of its chain (any depth), then universal, then the state chain::

    (target, cuda:sm_89, conv|fp16, leaf)          only for non-fp32 dtypes
    (target, cuda:sm_89, conv, leaf)
    (node,   ada,        conv, leaf)
    (node,   nvidia_gpu, conv, leaf)
    (node,   simt_gpu,   conv, leaf)
    (universal, -,       conv, leaf)
    (universal, -,       conv, parent) ... (universal, -, conv, root)
    (universal, -,       conv, *)
    (universal, -,       *,    *)

and the estimate shrinks each node toward the next one up::

    est(k) = (n_k * mean_k + kappa * est(parent(k))) / (n_k + kappa)

with ``kappa`` per edge: a target or scope node may set its own ``kappa`` (its pull
toward its parent; ``targets.json`` / runtime nodes), every other edge uses the
default. A new target starts from its nearest scope's evidence. v1 rows with
level ``family`` are read as ``node``. The
bandit's UCB adds ``c * sqrt(log(N + 1) / (n_local + 1))`` on the local count.
"""
from __future__ import annotations

import math
import random
from typing import Optional

from .targets import FAMILY, NODE, ROOT, TARGET, UNIVERSAL, TargetRegistry

ANY = "*"
KAPPA = 3.0


def path(registry: TargetRegistry, *, target_key: str, op_class: str, chain: list,
         dtype: Optional[str] = None) -> list[tuple]:
    """Nodes (level, scope, op, state), most specific first."""
    t = registry.get(target_key)
    leaf = chain[-1] if chain else ANY
    nodes = []
    if dtype and dtype != "fp32":
        nodes.append((TARGET, t.key, f"{op_class}|{dtype}", leaf))
    nodes.append((TARGET, t.key, op_class, leaf))
    nodes += [(NODE, n, op_class, leaf) for n in registry.ancestors(t.node) if n != ROOT]
    nodes.append((UNIVERSAL, "", op_class, leaf))
    for s in reversed(chain[:-1]):
        nodes.append((UNIVERSAL, "", op_class, s))
    if leaf != ANY:
        nodes.append((UNIVERSAL, "", op_class, ANY))
    nodes.append((UNIVERSAL, "", ANY, ANY))
    return nodes


class HierStats:
    def __init__(self, registry: Optional[TargetRegistry] = None, db=None, kappa: float = KAPPA,
                 prior: float = 0.0):
        self.registry = registry or TargetRegistry()
        self.db = db
        self.kappa, self.prior = kappa, prior
        self.data: dict[tuple, list] = {}           # (item, *node) -> [n, sum, sumsq]
        if db is not None:
            self.reload()

    def reload(self) -> None:
        if self.db is not None:
            self.data = {}
            for k, v in self.db.all_stats().items():
                if k[1] == FAMILY:                       # v1 DB rows
                    k = (k[0], NODE, *k[2:])
                a = self.data.setdefault(k, [0.0, 0.0, 0.0])
                for i in range(3):
                    a[i] += v[i]

    def update(self, item: str, nodes: list[tuple], reward: float) -> None:
        for node in nodes:
            a = self.data.setdefault((item, *node), [0.0, 0.0, 0.0])
            a[0] += 1
            a[1] += reward
            a[2] += reward * reward
        if self.db is not None:
            self.db.add_stats([(item, *n) for n in nodes], reward)

    def count(self, item: str, node: tuple) -> float:
        return self.data.get((item, *node), [0.0])[0]

    def mean(self, item: str, node: tuple) -> Optional[float]:
        a = self.data.get((item, *node))
        return a[1] / a[0] if a and a[0] else None

    def estimate(self, item: str, nodes: list[tuple]) -> float:
        est = self.prior
        for node in reversed(nodes):
            a = self.data.get((item, *node))
            if a and a[0]:
                k = self.registry.kappa_of(node[1], self.kappa) if node[0] in (TARGET, NODE) else self.kappa
                est = (a[1] + k * est) / (a[0] + k)
        return est

    def evidence_level(self, item: str, nodes: list[tuple]) -> tuple[str, float]:
        """Where the evidence for ``item`` comes from: (label, n) at the most specific node with data."""
        for node in nodes:
            n = self.count(item, node)
            if n:
                level, scope, op, state = node
                where = {TARGET: f"{scope.split(':')[-1]}", NODE: f"{scope}", UNIVERSAL: "all targets"}[level]
                ctx = "" if state == ANY else f", {state}"
                return f"{where}{ctx}", n
        return "untested", 0.0

    def local_total(self, nodes: list[tuple], items) -> float:
        return sum(self.count(i, nodes[0]) for i in items)

    def ucb(self, items: list[str], nodes: list[tuple], rng: random.Random, c: float = 1.0) -> list[tuple]:
        """[(score, est, n_local, item)] sorted best first."""
        total = self.local_total(nodes, items) + 1
        out = []
        for it in items:
            n = self.count(it, nodes[0])
            est = self.estimate(it, nodes)
            score = est + c * math.sqrt(math.log(total + 1) / (n + 1)) + 1e-6 * rng.random()
            out.append((score, est, n, it))
        return sorted(out, reverse=True)

    def summary(self) -> dict:
        return {"|".join(map(str, k)): [round(x, 4) for x in v] for k, v in self.data.items()}
