# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-problem optimisation knowledge.

``outcomes``     one record per optimisation attempt (the raw evidence)
``impl_store``   verified kernel implementations, retrievable by op/shape/class
``claims``       semantic knowledge distilled from outcomes (applicability,
                 pitfalls, interactions, parameters), with support counts
``reflect``      the semantic update pass: outcomes -> claim edits

The technique database (``agents/database.py``) stays the source of *what*
to try; this package records *how it went* and *how it was done*.
"""
