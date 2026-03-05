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
import signal


async def safe_kill_process(proc, logger=None):
    if proc.returncode is None:
        forbidden_groups = [
            os.getpgid(0),  # Current shell's group
            os.getpgid(1),  # init/systemd group
            0,
            1,  # init/systemd group
        ]
        current_pgid = os.getpgid(proc.pid)
        if logger:
            logger.info(
                f"Current PGID: {current_pgid} ; Process PID: {proc.pid} ; Forbidden groups: {forbidden_groups}"
            )

        # Critical safety check
        if current_pgid not in forbidden_groups:
            if logger:
                logger.info(f"Safe to kill - KILLING PGID: {current_pgid}")
            os.killpg(current_pgid, signal.SIGKILL)
        else:
            if logger:
                logger.warning(f"Refusing to kill protected group {current_pgid}")
            proc.kill()

    await proc.wait()
