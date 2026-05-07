#!/bin/bash
# Profile OpenCL kernel execution on Adreno GPU using KGSL ftrace tracepoints.
# This captures per-dispatch GPU timing with zero application instrumentation.
#
# Usage (on the board directly, or via SSH):
#   ./profile_kernels.sh [path_to_test_binary] [path_to_kernels_dir]
set -e

WORK_DIR="${1:-.}"
TEST_BIN="${WORK_DIR}/test_opencl_compile"
KERNEL_DIR="${2:-$WORK_DIR}"
TRACE_OUTPUT="${WORK_DIR}/profile_trace.txt"

echo "=== Adreno GPU Kernel Profiler ==="
echo "Test binary: ${TEST_BIN}"
echo "Kernel dir: ${KERNEL_DIR}"
echo ""

if [ ! -x "${TEST_BIN}" ]; then
    echo "ERROR: Test binary not found or not executable: ${TEST_BIN}"
    echo "Compile it first: gcc -o test_opencl_compile test_host.c -lOpenCL -lm -DCL_TARGET_OPENCL_VERSION=200"
    exit 1
fi

# Check we have access to tracing infrastructure
TRACEFS="/sys/kernel/debug/tracing"
if [ ! -d "${TRACEFS}" ]; then
    echo "ERROR: tracefs not accessible at ${TRACEFS}"
    echo "Make sure you're running as root"
    exit 1
fi

echo "[1/4] Configuring ftrace for KGSL tracepoints..."

# Clear any existing trace
echo 0 > ${TRACEFS}/tracing_on
echo > ${TRACEFS}/trace

# Enable the relevant KGSL tracepoints
echo 1 > ${TRACEFS}/events/kgsl/adreno_cmdbatch_submitted/enable
echo 1 > ${TRACEFS}/events/kgsl/adreno_cmdbatch_retired/enable

echo "  Enabled: adreno_cmdbatch_submitted, adreno_cmdbatch_retired"
echo ""

echo "[2/4] Running test binary with tracing enabled..."
echo 1 > ${TRACEFS}/tracing_on

# Run the actual test with --profile to get OpenCL event timings
"${TEST_BIN}" "${KERNEL_DIR}" --profile 2>&1 | sed 's/^/  /'

echo 0 > ${TRACEFS}/tracing_on
echo ""

echo "[3/4] Capturing trace..."
cat ${TRACEFS}/trace > "${TRACE_OUTPUT}"

# Disable tracepoints
echo 0 > ${TRACEFS}/events/kgsl/adreno_cmdbatch_submitted/enable
echo 0 > ${TRACEFS}/events/kgsl/adreno_cmdbatch_retired/enable

echo "  Trace saved to: ${TRACE_OUTPUT}"
echo ""

echo "[4/4] Parsing kernel dispatch timings..."
echo ""
echo "  -----------------------------------------------------------------------"
printf "  %-6s  %-12s  %-12s  %-14s  %-10s\n" "CTX" "TIMESTAMP" "START" "RETIRE" "DURATION"
echo "  -----------------------------------------------------------------------"

# Parse the retired events which contain both start and retire ticks
grep "adreno_cmdbatch_retired" "${TRACE_OUTPUT}" | while read -r line; do
    # Extract fields from the trace line
    ctx=$(echo "$line" | grep -oP 'ctx=\K[0-9]+')
    ts=$(echo "$line" | grep -oP 'ts=\K[0-9]+')
    start=$(echo "$line" | grep -oP 'start=\K[0-9]+')
    retire=$(echo "$line" | grep -oP 'retire=\K[0-9]+')

    if [ -n "$start" ] && [ -n "$retire" ] && [ "$start" != "0" ] && [ "$retire" != "0" ]; then
        duration=$((retire - start))
        printf "  %-6s  %-12s  %-12s  %-14s  %s ticks\n" "$ctx" "$ts" "$start" "$retire" "$duration"
    fi
done

echo "  -----------------------------------------------------------------------"
echo ""

# Also show GPU clock for converting ticks to time
GPU_CLK=$(cat /sys/class/kgsl/kgsl-3d0/gpuclk 2>/dev/null || echo "unknown")
GPU_CLK_BUSY=$(cat /sys/class/kgsl/kgsl-3d0/gpu_clock_stats 2>/dev/null | head -1 || echo "")
echo "  GPU clock: ${GPU_CLK} Hz"

if [ "$GPU_CLK" != "unknown" ] && [ "$GPU_CLK" != "0" ]; then
    echo ""
    echo "  Timing summary (estimated from GPU clock):"
    echo "  -----------------------------------------------------------------------"
    printf "  %-6s  %-12s  %-14s  %-10s\n" "CTX" "TIMESTAMP" "DURATION(ticks)" "TIME(us)"
    echo "  -----------------------------------------------------------------------"

    grep "adreno_cmdbatch_retired" "${TRACE_OUTPUT}" | while read -r line; do
        ctx=$(echo "$line" | grep -oP 'ctx=\K[0-9]+')
        ts=$(echo "$line" | grep -oP 'ts=\K[0-9]+')
        start=$(echo "$line" | grep -oP 'start=\K[0-9]+')
        retire=$(echo "$line" | grep -oP 'retire=\K[0-9]+')

        if [ -n "$start" ] && [ -n "$retire" ] && [ "$start" != "0" ] && [ "$retire" != "0" ]; then
            duration=$((retire - start))
            # Convert ticks to microseconds: ticks / (clock_hz / 1000000)
            time_us=$(awk "BEGIN { printf \"%.2f\", $duration / ($GPU_CLK / 1000000.0) }")
            printf "  %-6s  %-12s  %-14s  %s us\n" "$ctx" "$ts" "$duration" "$time_us"
        fi
    done
    echo "  -----------------------------------------------------------------------"
fi

echo ""
echo "  Raw trace: ${TRACE_OUTPUT}"
echo ""
echo "=== Profiling complete ==="
