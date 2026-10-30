# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Precision-injection helper shared by torch-flavoured sources.

Extracted from :class:`KernelBenchSource` in the SOL split so
:class:`SOLExecBenchTorchSource` can use the exact same snippet + guard
without importing the sibling class (avoids a circular between
``kernelbench_source`` and ``sol_execbench_torch_source``).

The pre-refactor CUDA-path variant in ``KernelBenchDataset._load_dataset``
lacked the idempotency guard and the ``class Model`` fallback; the
:func:`inject_precision` below is the union (fp32/fp16/bf16 + guard +
fallback) — same shape both torch sources emit.
"""
from __future__ import annotations

from typing import Mapping

ALLOWED_PRECISION: frozenset[str] = frozenset({"fp32", "fp16", "bf16"})

_DTYPE_SNIPPETS: Mapping[str, str] = {
    "fp32": "# Use fp32 datatype for all tensors\ntorch.set_default_dtype(torch.float32)",
    "fp16": "# Use fp16 datatype for all tensors\ntorch.set_default_dtype(torch.float16)",
    "bf16": "# Use bf16 datatype for all tensors\ntorch.set_default_dtype(torch.bfloat16)",
}


def inject_precision(reference_code: str, precision: str) -> str:
    """Inject a ``torch.set_default_dtype`` snippet before ``class Model``.

    Contract:
      - Unknown precision -> returns input unchanged (silent no-op).
      - Snippet already present -> returns input unchanged (idempotent).
      - ``class Model`` missing -> prepends the snippet (previously
        silently corrupted the source in the CUDA-path variant).
    """
    snippet = _DTYPE_SNIPPETS.get(precision)
    if not snippet:
        return reference_code
    if "set_default_dtype" in reference_code:
        return reference_code
    insertion_point = reference_code.find("class Model")
    if insertion_point == -1:
        return snippet + "\n\n" + reference_code
    return (
        reference_code[:insertion_point]
        + snippet
        + "\n\n"
        + reference_code[insertion_point:]
    )
