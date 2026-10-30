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
import uvicorn.config


def get_log_config(log_filepath: str = None):
    base_config = uvicorn.config.LOGGING_CONFIG.copy()
    log_format = "%(asctime)s | %(levelprefix)s | %(message)s"
    base_config["formatters"]["default"]["fmt"] = log_format
    if log_filepath is not None:
        base_config["handlers"]["file"] = {
            "formatter": "file",  # Use a custom formatter without colors
            "class": "logging.FileHandler",
            "filename": log_filepath,
            "mode": "a",
        }
        # Add a formatter without colors for file logging
        base_config["formatters"]["file"] = {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": log_format,
            "use_colors": False,  # Explicitly disable colors
        }
        base_config["loggers"]["uvicorn"]["handlers"].append("file")
    return base_config
