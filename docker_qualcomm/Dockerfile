FROM ubuntu:22.04

# Create /kernelblaster directory with proper permissions
RUN mkdir -p /kernelblaster && chmod 777 /kernelblaster

# Set WORKDIR to /workspace to avoid permission issues during container init
# The entrypoint script will handle changing to /kernelblaster after fixing permissions
WORKDIR /workspace

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCL development
RUN apt-get update && apt-get -y install \
    sudo \
    ccache \
    zsh \
    sshfs \
    cmake \
    build-essential \
    git \
    wget \
    curl \
    unzip \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    # OpenCL ICD loader
    ocl-icd-opencl-dev \
    ocl-icd-libopencl1 \
    clinfo \
    # SSH client for connecting to dev board
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install Qualcomm Adreno GPU SDK (OpenCL headers + Adreno libOpenCL)
ENV QUALCOMM_SDK_ROOT=/opt/qualcomm/adreno-gpu-sdk
RUN mkdir -p /opt/qualcomm
COPY docker_qualcomm/sdk/Adreno_OpenCL_SDK*.zip /tmp/adreno-sdk.zip
RUN unzip -q /tmp/adreno-sdk.zip -d /opt/qualcomm/adreno-gpu-sdk && \
    rm /tmp/adreno-sdk.zip

# Install Adreno OpenCL headers system-wide (includes cl_ext_qcom.h for Adreno extensions)
RUN cp -f ${QUALCOMM_SDK_ROOT}/opencl-sdk/inc/CL/* /usr/include/CL/

# Set up OpenCL environment
# Note: The SDK's libOpenCL.so is ARM64 (for the board). On this x86 build host we link
# against the system ICD loader for syntax/compilation checks. Actual GPU execution
# happens on the RB5 board via SSH.
ENV ADRENO_SDK_ROOT=${QUALCOMM_SDK_ROOT}
ENV OPENCL_INCLUDE_DIR=/usr/include/CL

COPY docker_qualcomm/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN python -m pip install --no-cache-dir -U pip setuptools wheel && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    python -m pip install --no-cache-dir 'anthropic[bedrock]>=0.40.0'

# Install pyopencl for testing kernels from Python
RUN python -m pip install --no-cache-dir pyopencl

# Verify OpenCL headers are available
RUN ls /usr/include/CL/cl.h && echo "OpenCL headers found"
RUN ls /usr/include/CL/cl_ext_qcom.h && echo "Adreno extensions header found"

COPY docker_qualcomm/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
