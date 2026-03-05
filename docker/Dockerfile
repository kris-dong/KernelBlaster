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
FROM nvcr.io/nvidia/pytorch:25.01-py3

# Create /kernelblaster directory with proper permissions
RUN mkdir -p /kernelblaster && chmod 777 /kernelblaster

# Set WORKDIR to /workspace to avoid permission issues during container init
# The entrypoint script will handle changing to /kernelblaster after fixing permissions
WORKDIR /workspace

# Add CUDA to PATH and set environment variables
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV CUDACXX=/usr/local/cuda/bin/nvcc

# Install system dependencies including those needed for bitsandbytes
RUN apt-get update && apt-get -y install sudo ccache zsh sshfs cmake build-essential git

# Verify CUDA installation
RUN nvcc --version


RUN cd /kernelblaster
COPY docker/requirements.txt /tmp/requirements.txt

# Install Python dependencies
# Note: bitsandbytes requires CUDA toolkit to be available during installation
RUN python -m pip install --no-cache-dir -U pip setuptools wheel && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Verify bitsandbytes installation (optional, for debugging)
RUN python -c "import bitsandbytes; print('bitsandbytes version:', bitsandbytes.__version__)" || echo "bitsandbytes import check skipped"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
