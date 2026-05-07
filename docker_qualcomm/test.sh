#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${SCRIPT_DIR}/test"
BUILD_DIR="${SCRIPT_DIR}/test/build"

# Dev board SSH target
BOARD_HOST="${BOARD_HOST:-root@10.44.120.201}"
BOARD_WORK_DIR="/tmp/kernelblaster_test"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

echo "=== Adreno OpenCL Compilation Test Suite ==="
echo "Platform: $(uname -m) / $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2)"
echo "Target board: ${BOARD_HOST}"
echo ""

# -------------------------------------------------------
# Step 1: Verify OpenCL headers are present
# -------------------------------------------------------
echo "[1/5] Checking OpenCL headers..."
if [ -f "/usr/include/CL/cl.h" ]; then
    echo "  Found: /usr/include/CL/cl.h"
else
    echo "  ERROR: /usr/include/CL/cl.h not found!"
    exit 1
fi

if [ -f "/usr/include/CL/cl_ext_qcom.h" ]; then
    echo "  Found: /usr/include/CL/cl_ext_qcom.h (Adreno extensions)"
else
    echo "  WARNING: cl_ext_qcom.h not found (Adreno-specific extensions unavailable)"
fi
echo "  PASS"
echo ""

# -------------------------------------------------------
# Step 2: Verify OpenCL libraries
# -------------------------------------------------------
echo "[2/5] Checking OpenCL libraries..."
LIBOPENCL=""
for path in /usr/lib/x86_64-linux-gnu/libOpenCL.so /usr/lib/aarch64-linux-gnu/libOpenCL.so /usr/lib/libOpenCL.so; do
    if [ -e "$path" ]; then
        echo "  Found: $path"
        LIBOPENCL="$path"
        file "$path" 2>/dev/null | sed 's/^/    /'
    fi
done

if [ -z "$LIBOPENCL" ]; then
    echo "  WARNING: No libOpenCL.so found — compilation will work but local runtime tests will skip"
fi
echo "  PASS"
echo ""

# -------------------------------------------------------
# Step 3: Compile test program locally (x86, links against ICD loader)
# -------------------------------------------------------
echo "[3/5] Building OpenCL test program (local x86)..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

cmake -S "${TEST_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_C_COMPILER=gcc \
    -DOPENCL_INCLUDE_DIR="/usr/include" \
    -DADRENO_SDK_ROOT="${QUALCOMM_SDK_ROOT:-/opt/qualcomm/adreno-gpu-sdk}/opencl-sdk" \
    2>&1 | sed 's/^/  /'

cmake --build "${BUILD_DIR}" 2>&1 | sed 's/^/  /'
echo "  PASS: Local test binary compiled"
echo ""

# -------------------------------------------------------
# Step 4: Run locally (no GPU — validates kernel source loading)
# -------------------------------------------------------
echo "[4/5] Running local compilation test (no GPU, syntax check only)..."
if "${BUILD_DIR}/test_opencl_compile" "${TEST_DIR}" 2>&1 | sed 's/^/  /'; then
    echo "  PASS"
else
    echo "  FAIL (local)"
    exit 1
fi
echo ""

# -------------------------------------------------------
# Step 5: Build and run on the dev board via SSH (actual Adreno GPU)
# -------------------------------------------------------
echo "[5/5] Running on dev board (${BOARD_HOST}) with Adreno GPU..."

# Check SSH connectivity
if ! ssh ${SSH_OPTS} "${BOARD_HOST}" "echo ok" > /dev/null 2>&1; then
    echo "  WARNING: Cannot reach ${BOARD_HOST} via SSH — skipping on-device test"
    echo "  Set BOARD_HOST=user@ip to configure the target board"
    echo ""
    echo "=== LOCAL TESTS PASSED (on-device skipped) ==="
    exit 0
fi

echo "  Connected to board"

# Create work directory on board
ssh ${SSH_OPTS} "${BOARD_HOST}" "mkdir -p ${BOARD_WORK_DIR}"

# Copy test sources and host program source to the board
scp ${SSH_OPTS} -q \
    "${TEST_DIR}/test_host.c" \
    "${TEST_DIR}/vector_add.cl" \
    "${TEST_DIR}/matmul_tiled.cl" \
    "${BOARD_HOST}:${BOARD_WORK_DIR}/"

echo "  Copied test files to board"

# Compile and run on the board (native ARM64 with real Adreno GPU)
ssh ${SSH_OPTS} "${BOARD_HOST}" bash -s "${BOARD_WORK_DIR}" <<'REMOTE_SCRIPT'
set -e
WORK_DIR="$1"
cd "$WORK_DIR"

echo "  Compiling test on board..."
gcc -o test_opencl_compile test_host.c \
    -I/usr/include \
    -L/usr/lib -lOpenCL -lm \
    -DCL_TARGET_OPENCL_VERSION=200 \
    2>&1 | sed 's/^/    /'

echo "  Running test on Adreno GPU (with profiling)..."
./test_opencl_compile "$WORK_DIR" --profile 2>&1 | sed 's/^/    /'
REMOTE_SCRIPT

echo ""

# -------------------------------------------------------
# Step 6: Profile kernel execution using KGSL ftrace
# -------------------------------------------------------
echo "[6/6] Profiling kernel execution on Adreno GPU (ftrace)..."

# Copy profiling script to board
scp ${SSH_OPTS} -q \
    "${TEST_DIR}/profile_kernels.sh" \
    "${BOARD_HOST}:${BOARD_WORK_DIR}/"

# Run profiler on board (requires root for tracefs access)
ssh ${SSH_OPTS} "${BOARD_HOST}" bash -s "${BOARD_WORK_DIR}" <<'REMOTE_PROFILE'
set -e
WORK_DIR="$1"
cd "$WORK_DIR"
chmod +x profile_kernels.sh

# Compile test binary if not already there
if [ ! -x test_opencl_compile ]; then
    gcc -o test_opencl_compile test_host.c \
        -I/usr/include -L/usr/lib -lOpenCL -lm \
        -DCL_TARGET_OPENCL_VERSION=200
fi

# Run the profiler
./profile_kernels.sh "$WORK_DIR" "$WORK_DIR" 2>&1 | sed 's/^/    /'
REMOTE_PROFILE

echo ""
echo "=== ALL TESTS PASSED (including on-device profiling) ==="
