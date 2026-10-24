# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Knowledge architecture v2 (``docs/knowledge_architecture.md``).

``targets``     fixed levels (universal > family > target); families + targets as data
``metrics``     canonical metrics; NCU / NCU-text / roofline adapters (Adreno, RVV stubs)
``predicates``  the JSON predicate DSL (three-valued)
``states``      predicate states in a hierarchy; deterministic classifier
``ontology``    techniques with mechanisms, predictions, realizations; mechanism check
``stats``       hierarchical reward statistics with shrinkage
``store``       one SQLite DB (WAL); outcome log + claim store on top
``hypotheses``  proposed techniques with a test budget; promotion / retirement
``discovery``   evidence-driven state discovery with validation
``reflect_v2``  reflection with ``lift`` / ``mechanism`` edits and the upkeep passes
``knowledge``   :class:`KnowledgeV2`, the facade the agents hold

v1 (``knowledge.claims`` / ``reflect`` / the JSON technique DB) stays the
default; agents take v2 behind a switch (``run_multislot.py --knowledge v2``).
"""
from .knowledge import Choice, KnowledgeV2  # noqa: F401
