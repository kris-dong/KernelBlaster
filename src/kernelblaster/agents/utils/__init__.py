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
from .error import FeedbackError
from .file_operations import write_code_to_file, write_jsonl, read_jsonl
from .query import *
from .timer import *
from .commands import *
from .parsing import *
from .annotate_ncu import *

# Default to the instrumented LLM wrapper so token usage + timing are logged in run.log.
# This is a lightweight wrapper around the normal implementation; it remains compatible
# with existing call sites and becomes a no-op for aggregation unless a timing collector
# is installed (see timing_patches.py).
try:
    from .query_instrumented import generate_code_retry_instrumented as generate_code_retry
except Exception:
    # Fall back to the standard implementation if instrumentation cannot be imported.
    pass
