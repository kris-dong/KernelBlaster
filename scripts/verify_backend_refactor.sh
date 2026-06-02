#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Backend-refactor verification suite.
#
# Exercises the Phase 1+2+4a+5 changes against real hardware so we can catch
# regressions before they hit a production run. Designed to be expanded into
# proper unit/integration tests later; for now it's a single bash driver that
# prints PASS/FAIL per test and exits non-zero on any failure.
#
# What it covers:
#   T1 — Unit: imports + Backend.parse_profile + run_subprocess_shell +
#        queue_worker_loop. No servers, no network.
#   T2 — CUDA E2E: spawns compile + gpu servers, compiles a kernelbench-cuda
#        L1 problem, executes the binary on the local NVIDIA GPU, checks the
#        driver prints "passed". Tears the servers down on exit.
#   T3 — OpenCL E2E: spawns compile_opencl + gpu_adreno servers, compiles a
#        benchmark-opencl L1 kernel via SSH to the Adreno board, verifies the
#        remote binary exists on the board. (Full exec needs a precomputed
#        reference_output.bin which the RL flow generates lazily — we skip
#        that here to keep the test self-contained.)
#
# Required environment (set externally or sourced before running):
#   AWS_BEARER_TOKEN_BEDROCK         — only consumed by full RL flow, not by
#                                       these plumbing tests; tolerated absent.
#   KERNELBLASTER_ADRENO_BOARD_HOST  — e.g. root@10.44.120.201. Required for T3.
#
# Required hardware:
#   - NVIDIA GPU with CUDA toolchain + libtorch (host or container) for T2
#   - SSH key reaching $KERNELBLASTER_ADRENO_BOARD_HOST for T3
#
# Required venv (auto-detected):
#   /scratch/kris/local-llm/.venv  if available (has fastapi/torch/loguru),
#   otherwise system python (will likely fail on import of fastapi).
#
# Ports:
#   2200 — CUDA compile server
#   2201 — CUDA GPU server
#   2202 — OpenCL compile server
#   2203 — Adreno GPU server
# (Deliberately offset from the existing container's 2101/2102.)
#
# Usage:
#   bash scripts/verify_backend_refactor.sh         # run all tests
#   SKIP_T2=1 bash scripts/verify_backend_refactor.sh   # skip CUDA tests
#   SKIP_T3=1 bash scripts/verify_backend_refactor.sh   # skip OpenCL tests
#   KEEP_LOGS=1 bash scripts/verify_backend_refactor.sh # don't clean up /tmp logs

set -u  # don't set -e — we want to count failures, not abort

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=""
for cand in \
    /scratch/kris/local-llm/.venv/bin/python \
    "$(command -v python3 2>/dev/null)" \
    "$(command -v python 2>/dev/null)"; do
    if [ -x "$cand" ]; then
        if "$cand" -c "import fastapi, torch" >/dev/null 2>&1; then
            PY="$cand"
            break
        fi
    fi
done
if [ -z "$PY" ]; then
    echo "FATAL: no python interpreter found with fastapi + torch installed"
    exit 2
fi

TMPDIR="$(mktemp -d -t kb_verify_XXXXXX)"
echo "==== Backend-refactor verification ===="
echo "repo:      $REPO_ROOT"
echo "python:    $PY"
echo "tmp:       $TMPDIR"
echo "board:     ${KERNELBLASTER_ADRENO_BOARD_HOST:-<unset>}"
echo "skip_T2:   ${SKIP_T2:-0}"
echo "skip_T3:   ${SKIP_T3:-0}"
echo

PASS_COUNT=0
FAIL_COUNT=0
SPAWNED_PIDS=()

cleanup() {
    local rc=$?
    if [ ${#SPAWNED_PIDS[@]} -gt 0 ]; then
        echo
        echo "---- cleaning up ${#SPAWNED_PIDS[@]} spawned server(s) ----"
        for pid in "${SPAWNED_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -TERM "$pid" 2>/dev/null
            fi
        done
        # Give them a moment to shut down cleanly, then SIGKILL stragglers.
        sleep 1
        for pid in "${SPAWNED_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL "$pid" 2>/dev/null
            fi
        done
    fi
    if [ "${KEEP_LOGS:-0}" = "1" ]; then
        echo "logs preserved at $TMPDIR"
    else
        rm -rf "$TMPDIR"
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

SKIP_COUNT=0
record() {
    local name="$1" status="$2"
    case "$status" in
        PASS) echo "  ✅ PASS $name"; PASS_COUNT=$((PASS_COUNT + 1)) ;;
        FAIL) echo "  ❌ FAIL $name"; FAIL_COUNT=$((FAIL_COUNT + 1)) ;;
        SKIP) echo "  ⏭️  SKIP $name"; SKIP_COUNT=$((SKIP_COUNT + 1)) ;;
    esac
}

wait_for_health() {
    local url="$1" timeout="${2:-30}"
    local start=$SECONDS
    while [ $((SECONDS - start)) -lt "$timeout" ]; do
        if curl -sf --max-time 2 "$url/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

# ---------------------------------------------------------------------------
# T1 — Unit / import smoke (no servers)
# ---------------------------------------------------------------------------

echo "==== T1: unit + import smoke ===="

T1_OUT="$TMPDIR/t1.log"
if "$PY" - >"$T1_OUT" 2>&1 <<'PYEOF'
import sys
sys.path.insert(0, ".")

# === Imports ===
from src.kernelblaster.servers.utils.subprocess import (
    SubprocessRunError, run_subprocess_shell, tail_bytes,
)
from src.kernelblaster.servers.utils.queue_server import queue_worker_loop
from src.kernelblaster.servers.utils import safe_kill_process
from src.kernelblaster.backends import (
    Backend, ProfileResult, CUDABackend, OpenCLBackend, get_backend, backend_for_gpu,
)
from src.kernelblaster.config.gpu_config import GPUType

# === Phase 2: parse_profile ===
cuda = get_backend("cuda")
ocl = get_backend("opencl")

# CUDA: cycles extracted from NCU log
pr = cuda.parse_profile("Elapsed Cycles                cycle        98765")
assert pr.raw_metrics["elapsed_cycles"] == 98765, pr.raw_metrics
assert pr.total_time_ms == 0.0

# CUDA: missing cycles -> RuntimeError (strict contract preserved)
try:
    cuda.parse_profile("no cycles in this log")
except RuntimeError:
    pass
else:
    raise AssertionError("expected RuntimeError for missing NCU cycles")

# OpenCL: [PROFILE] markers
ocl_log = "[PROFILE] vec_add: 0.5 ms\n[PROFILE] matmul: 2.25 ms"
pr = ocl.parse_profile(ocl_log)
assert pr.per_kernel_ms == {"vec_add": 0.5, "matmul": 2.25}
assert abs(pr.total_time_ms - 2.75) < 1e-9

# === gpu_config wiring ===
assert GPUType.L40S.target_lang == "cuda"
assert GPUType.ADRENO_650.target_lang == "opencl"
b = GPUType.L40S.backend()
assert isinstance(b, CUDABackend)
assert b.gpu == GPUType.L40S

# === Phase 1: run_subprocess_shell ===
import asyncio, time
from pathlib import Path

class _Err(Exception):
    def __init__(self, m): self.message = m; super().__init__(m)

async def t1_subprocess():
    # 1) happy
    out, _ = await run_subprocess_shell(stage="t", cmd="echo hello",
                                         cwd=Path("/tmp"), timeout_s=5)
    assert b"hello" in out

    # 2) non-zero exit -> factory
    try:
        await run_subprocess_shell(stage="t", cmd="exit 11",
                                    cwd=Path("/tmp"), timeout_s=5,
                                    error_factory=_Err)
    except _Err as e:
        assert "rc=11" in e.message

    # 3) timeout -> factory
    try:
        await run_subprocess_shell(stage="t", cmd="sleep 5",
                                    cwd=Path("/tmp"), timeout_s=0.3,
                                    error_factory=_Err)
    except _Err as e:
        assert "Timeout" in e.message

asyncio.run(t1_subprocess())

# === Phase 1: queue_worker_loop ===
async def t1_queue():
    q = asyncio.Queue()
    async def handler(wid, args):
        if args[0] == "bad":
            raise _Err("expected")
        return args[0]
    task = asyncio.create_task(queue_worker_loop(
        worker_id=0, queue=q, handler=handler, domain_error=_Err,
    ))
    # happy
    f = asyncio.Future()
    await q.put(("ok", f, time.time()))
    assert await f == "ok"
    # error
    f = asyncio.Future()
    await q.put(("bad", f, time.time()))
    try: await f
    except _Err: pass
    task.cancel()
    try: await task
    except asyncio.CancelledError: pass

asyncio.run(t1_queue())

print("T1: OK")
PYEOF
then
    record "T1 imports + parse_profile + run_subprocess_shell + queue_worker_loop" PASS
else
    cat "$T1_OUT" | tail -40
    record "T1 imports + parse_profile + run_subprocess_shell + queue_worker_loop" FAIL
fi

# ---------------------------------------------------------------------------
# T2 — CUDA E2E: spawn servers, compile, execute, check "passed"
# ---------------------------------------------------------------------------

if [ "${SKIP_T2:-0}" != "1" ]; then
    echo
    echo "==== T2: CUDA E2E (compile + execute) ===="

    CUDA_ARTIFACTS="$TMPDIR/cuda_artifacts"
    mkdir -p "$CUDA_ARTIFACTS"
    PROBLEM_DIR="$REPO_ROOT/data/kernelbench-cuda/level1/005_Matrix_scalar_multiplication"

    # Pre-flight: CUDA toolchain available?
    SKIP_T2_REASON=""
    if ! command -v nvcc >/dev/null 2>&1 && \
       ! [ -d "/usr/local/cuda/bin" ] && \
       ! [ -d "/usr/local/cuda-12/bin" ] && \
       ! [ -d "/usr/local/cuda-13/bin" ]; then
        SKIP_T2_REASON="no nvcc on PATH (CUDA toolchain not on host — try running this script inside the kernelblaster docker container)"
    fi
    # Pre-flight: is the GPU busy with non-KB processes (vllm, training, etc.)?
    # The gpu server refuses to start in this case (safety check in gpu.py).
    if [ -z "$SKIP_T2_REASON" ] && command -v nvidia-smi >/dev/null 2>&1; then
        BUSY_PROCS="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v '^\s*$' || true)"
        if [ -n "$BUSY_PROCS" ]; then
            SKIP_T2_REASON="GPU has pre-existing compute processes (gpu.py safety check would reject startup): $(echo "$BUSY_PROCS" | head -1)..."
        fi
    fi

    if [ -n "$SKIP_T2_REASON" ]; then
        echo "  Skipping T2: $SKIP_T2_REASON"
        record "T2.a compile server boot (port 2200)" SKIP
        record "T2.b gpu server boot (port 2201)" SKIP
        record "T2.c CUDA compile of 005_Matrix_scalar_multiplication" SKIP
        record "T2.d CUDA binary execute -> 'passed'" SKIP
    elif [ ! -f "$PROBLEM_DIR/driver.cpp" ] || [ ! -f "$PROBLEM_DIR/init.cu" ]; then
        echo "  driver.cpp / init.cu not found at $PROBLEM_DIR — skipping T2"
        record "T2 CUDA E2E" FAIL
    else
        # Spawn compile server
        "$PY" -m src.kernelblaster.servers.compile \
            --port 2200 \
            --num-workers 2 \
            --artifacts-dir "$CUDA_ARTIFACTS" \
            >"$TMPDIR/cuda_compile.log" 2>&1 &
        SPAWNED_PIDS+=($!)

        # Spawn GPU server
        "$PY" -m src.kernelblaster.servers.gpu \
            --port 2201 \
            --log_path "$TMPDIR/cuda_gpu.log" \
            >"$TMPDIR/cuda_gpu_stdout.log" 2>&1 &
        SPAWNED_PIDS+=($!)

        if wait_for_health "http://localhost:2200" 30; then
            record "T2.a compile server boot (port 2200)" PASS
        else
            echo "  compile server failed to start — log tail:"
            tail -20 "$TMPDIR/cuda_compile.log"
            record "T2.a compile server boot (port 2200)" FAIL
        fi

        if wait_for_health "http://localhost:2201" 30; then
            record "T2.b gpu server boot (port 2201)" PASS
        else
            echo "  gpu server failed to start — log tail:"
            tail -20 "$TMPDIR/cuda_gpu.log" 2>/dev/null || tail -20 "$TMPDIR/cuda_gpu_stdout.log"
            record "T2.b gpu server boot (port 2201)" FAIL
        fi

        # Issue compile request
        COMPILE_RESP="$TMPDIR/cuda_compile_resp.json"
        if curl -sf --max-time 120 -G "http://localhost:2200/compile" \
            --data-urlencode "job_name=verify_t2" \
            --data-urlencode "main_file=$PROBLEM_DIR/driver.cpp" \
            --data-urlencode "cuda_file=$PROBLEM_DIR/init.cu" \
            --data-urlencode "sm_version=sm_89" \
            >"$COMPILE_RESP" 2>"$TMPDIR/cuda_compile_curl.err"; then
            if "$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get('success') else 1)" "$COMPILE_RESP"; then
                record "T2.c CUDA compile of 005_Matrix_scalar_multiplication" PASS
                BINARY_PATH="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('output_path',''))" "$COMPILE_RESP")"
                echo "  -> binary: $BINARY_PATH"
            else
                echo "  compile returned success=false; response:"
                cat "$COMPILE_RESP" | tail -10
                record "T2.c CUDA compile of 005_Matrix_scalar_multiplication" FAIL
                BINARY_PATH=""
            fi
        else
            echo "  curl /compile failed; stderr:"
            cat "$TMPDIR/cuda_compile_curl.err"
            record "T2.c CUDA compile of 005_Matrix_scalar_multiplication" FAIL
            BINARY_PATH=""
        fi

        # Execute the binary via /gpu/binary
        if [ -n "$BINARY_PATH" ] && [ -f "$BINARY_PATH" ]; then
            EXEC_RESP="$TMPDIR/cuda_exec_resp.json"
            if curl -sf --max-time 60 -X POST "http://localhost:2201/gpu/binary" \
                -F "binary=@$BINARY_PATH" \
                -F "args=" \
                -F "n_runs=1" \
                -F "timeout=30" \
                >"$EXEC_RESP" 2>"$TMPDIR/cuda_exec_curl.err"; then
                if "$PY" -c "
import json,sys
r = json.load(open(sys.argv[1]))
if not r.get('success'):
    print('exec returned success=false:', r.get('message','')); sys.exit(1)
out = r.get('stdout','')
if isinstance(out, list): out = ''.join(out)
print('stdout snippet:', repr(out[-200:]))
sys.exit(0 if 'passed' in out.lower() else 2)
                " "$EXEC_RESP"; then
                    record "T2.d CUDA binary execute -> 'passed'" PASS
                else
                    echo "  binary executed but didn't print 'passed'"
                    record "T2.d CUDA binary execute -> 'passed'" FAIL
                fi
            else
                echo "  curl /gpu/binary failed; stderr:"
                cat "$TMPDIR/cuda_exec_curl.err"
                record "T2.d CUDA binary execute -> 'passed'" FAIL
            fi
        else
            echo "  skipping execute — no binary"
            record "T2.d CUDA binary execute -> 'passed'" FAIL
        fi
    fi
else
    echo
    echo "==== T2 skipped (SKIP_T2=1) ===="
fi

# ---------------------------------------------------------------------------
# T3 — OpenCL E2E: spawn servers, compile, verify remote binary on board
# ---------------------------------------------------------------------------

if [ "${SKIP_T3:-0}" != "1" ]; then
    echo
    echo "==== T3: OpenCL E2E (compile via SSH to Adreno board) ===="

    if [ -z "${KERNELBLASTER_ADRENO_BOARD_HOST:-}" ]; then
        echo "  KERNELBLASTER_ADRENO_BOARD_HOST not set — skipping T3"
        record "T3 OpenCL E2E" FAIL
    elif ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$KERNELBLASTER_ADRENO_BOARD_HOST" "echo ok" >/dev/null 2>&1; then
        echo "  cannot SSH to $KERNELBLASTER_ADRENO_BOARD_HOST — skipping T3"
        record "T3 OpenCL E2E" FAIL
    else
        OCL_ARTIFACTS="$TMPDIR/ocl_artifacts"
        mkdir -p "$OCL_ARTIFACTS"
        OCL_PROBLEM_DIR="$REPO_ROOT/data/benchmark-opencl/L1/19_ReLU"

        # Spawn OpenCL compile server
        "$PY" -m src.kernelblaster.servers.compile_opencl \
            --port 2202 \
            --num-workers 1 \
            --board-host "$KERNELBLASTER_ADRENO_BOARD_HOST" \
            --artifacts-dir "$OCL_ARTIFACTS" \
            >"$TMPDIR/ocl_compile.log" 2>&1 &
        SPAWNED_PIDS+=($!)

        # Spawn Adreno GPU server
        "$PY" -m src.kernelblaster.servers.gpu_adreno \
            --port 2203 \
            --board-host "$KERNELBLASTER_ADRENO_BOARD_HOST" \
            --log_path "$TMPDIR/ocl_gpu.log" \
            >"$TMPDIR/ocl_gpu_stdout.log" 2>&1 &
        SPAWNED_PIDS+=($!)

        if wait_for_health "http://localhost:2202" 30; then
            record "T3.a opencl compile server boot (port 2202)" PASS
        else
            echo "  opencl compile server failed to start — log tail:"
            tail -20 "$TMPDIR/ocl_compile.log"
            record "T3.a opencl compile server boot (port 2202)" FAIL
        fi

        if wait_for_health "http://localhost:2203" 30; then
            record "T3.b adreno gpu server boot (port 2203, board=$KERNELBLASTER_ADRENO_BOARD_HOST)" PASS
        else
            echo "  adreno gpu server failed to start — log tail:"
            tail -20 "$TMPDIR/ocl_gpu.log" 2>/dev/null || tail -20 "$TMPDIR/ocl_gpu_stdout.log"
            record "T3.b adreno gpu server boot (port 2203, board=$KERNELBLASTER_ADRENO_BOARD_HOST)" FAIL
        fi

        # Issue OpenCL compile request
        OCL_COMPILE_RESP="$TMPDIR/ocl_compile_resp.json"
        if curl -sf --max-time 120 -G "http://localhost:2202/compile_opencl" \
            --data-urlencode "job_name=verify_t3" \
            --data-urlencode "main_file=$OCL_PROBLEM_DIR/driver.c" \
            --data-urlencode "kernel_file=$OCL_PROBLEM_DIR/kernel.cl" \
            --data-urlencode "opencl_version=opencl_2.0" \
            --data-urlencode "remote=1" \
            >"$OCL_COMPILE_RESP" 2>"$TMPDIR/ocl_compile_curl.err"; then
            if "$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get('success') else 1)" "$OCL_COMPILE_RESP"; then
                record "T3.c OpenCL compile of 19_ReLU via SSH to board" PASS
                REMOTE_BIN="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('remote_binary_path','') or '')" "$OCL_COMPILE_RESP")"
                echo "  -> remote binary: $REMOTE_BIN"
            else
                echo "  opencl compile returned success=false; response:"
                cat "$OCL_COMPILE_RESP" | tail -10
                record "T3.c OpenCL compile of 19_ReLU via SSH to board" FAIL
                REMOTE_BIN=""
            fi
        else
            echo "  curl /compile_opencl failed; stderr:"
            cat "$TMPDIR/ocl_compile_curl.err"
            record "T3.c OpenCL compile of 19_ReLU via SSH to board" FAIL
            REMOTE_BIN=""
        fi

        # Verify the remote binary exists on the board
        LOCAL_BIN=""
        if [ -n "$REMOTE_BIN" ]; then
            if ssh -o BatchMode=yes -o ConnectTimeout=5 "$KERNELBLASTER_ADRENO_BOARD_HOST" "test -x '$REMOTE_BIN'" 2>/dev/null; then
                record "T3.d remote binary exists + is executable" PASS
                LOCAL_BIN="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('output_path','') or '')" "$OCL_COMPILE_RESP")"
            else
                echo "  remote binary check failed at $REMOTE_BIN"
                record "T3.d remote binary exists + is executable" FAIL
            fi
        else
            echo "  skipping remote check — no remote_binary_path"
            record "T3.d remote binary exists + is executable" FAIL
        fi

        # T3.e — execute binary via gpu_adreno.py /gpu/binary endpoint, no profile.
        # This is the most direct exercise of Phase 5's actual code changes:
        # upload_and_exec_binary's scp invocations now go through
        # run_subprocess_shell with AdrenoExecutionError on non-zero rc, and
        # exec_remote_command is fully migrated. The binary's CPU fallback for
        # the missing reference_output.bin means a single fresh run does
        # compile-on-CPU-reference + GPU kernel + tolerance check end-to-end.
        if [ -n "$LOCAL_BIN" ] && [ -f "$LOCAL_BIN" ]; then
            OCL_EXEC_RESP="$TMPDIR/ocl_exec_resp.json"
            if curl -sf --max-time 120 -X POST "http://localhost:2203/gpu/binary" \
                -F "binary=@$LOCAL_BIN" \
                -F "args=" \
                -F "n_runs=1" \
                -F "timeout=60" \
                -F "kernel_files=[\"$OCL_PROBLEM_DIR/kernel.cl\"]" \
                -F "profile=false" \
                >"$OCL_EXEC_RESP" 2>"$TMPDIR/ocl_exec_curl.err"; then
                if "$PY" -c "
import json, sys
r = json.load(open(sys.argv[1]))
if not r.get('success'):
    print('exec returned success=false:', r.get('message', '')); sys.exit(1)
out = r.get('stdout', '')
if isinstance(out, list): out = ''.join(out)
print('stdout snippet:', repr(out[-200:]))
sys.exit(0 if 'passed' in out.lower() else 2)
                " "$OCL_EXEC_RESP"; then
                    record "T3.e adreno binary execute via /gpu/binary -> 'passed'" PASS
                else
                    echo "  exec didn't print 'passed' — response tail:"
                    cat "$OCL_EXEC_RESP" | tail -3
                    record "T3.e adreno binary execute via /gpu/binary -> 'passed'" FAIL
                fi
            else
                echo "  curl /gpu/binary failed; stderr:"
                cat "$TMPDIR/ocl_exec_curl.err"
                record "T3.e adreno binary execute via /gpu/binary -> 'passed'" FAIL
            fi

            # T3.f — execute with profile=true, verify [PROFILE] marker is present
            # and OpenCLBackend.parse_profile extracts the kernel timing.
            OCL_PROF_RESP="$TMPDIR/ocl_prof_resp.json"
            if curl -sf --max-time 120 -X POST "http://localhost:2203/gpu/binary" \
                -F "binary=@$LOCAL_BIN" \
                -F "args=" \
                -F "n_runs=1" \
                -F "timeout=60" \
                -F "kernel_files=[\"$OCL_PROBLEM_DIR/kernel.cl\"]" \
                -F "profile=true" \
                >"$OCL_PROF_RESP" 2>"$TMPDIR/ocl_prof_curl.err"; then
                if "$PY" -c "
import json, sys
sys.path.insert(0, '.')
r = json.load(open(sys.argv[1]))
if not r.get('success'):
    print('exec returned success=false:', r.get('message', '')); sys.exit(1)
out = r.get('stdout', '')
if isinstance(out, list): out = ''.join(out)
print('stdout snippet:', repr(out[-300:]))
# Verify [PROFILE] marker present
if '[PROFILE]' not in out:
    print('no [PROFILE] marker in stdout'); sys.exit(2)
# Verify OpenCLBackend.parse_profile picks it up
from src.kernelblaster.backends import get_backend
pr = get_backend('opencl').parse_profile(out)
if not pr.per_kernel_ms:
    print('parse_profile produced empty per_kernel_ms'); sys.exit(3)
print('parse_profile result:', dict(pr.per_kernel_ms), 'total_ms:', pr.total_time_ms)
                " "$OCL_PROF_RESP"; then
                    record "T3.f adreno binary execute with profile -> [PROFILE] parsed" PASS
                else
                    echo "  profile run failed; response tail:"
                    cat "$OCL_PROF_RESP" | tail -3
                    record "T3.f adreno binary execute with profile -> [PROFILE] parsed" FAIL
                fi
            else
                echo "  curl /gpu/binary (profile) failed; stderr:"
                cat "$TMPDIR/ocl_prof_curl.err"
                record "T3.f adreno binary execute with profile -> [PROFILE] parsed" FAIL
            fi
        else
            echo "  no local binary available — skipping T3.e/T3.f"
            record "T3.e adreno binary execute via /gpu/binary -> 'passed'" SKIP
            record "T3.f adreno binary execute with profile -> [PROFILE] parsed" SKIP
        fi
    fi
else
    echo
    echo "==== T3 skipped (SKIP_T3=1) ===="
fi

# ---------------------------------------------------------------------------
# T4 — Direct test of refactored Phase 5 subprocess code paths
# ---------------------------------------------------------------------------
# gpu.exec_command and gpu_adreno.exec_remote_command were rewritten in
# Phase 5 to route through run_subprocess_shell. This test calls them
# directly without spinning up FastAPI, so it works even when the GPU is
# blocked by other processes (T2 environment). T3 covers gpu_adreno but
# nothing else covers the host-side exec_command rewrite end-to-end.

if [ "${SKIP_T4:-0}" != "1" ]; then
    echo
    echo "==== T4: Phase 5 direct subprocess plumbing (no servers) ===="
    T4_OUT="$TMPDIR/t4.log"
    if "$PY" - >"$T4_OUT" 2>&1 <<'PYEOF'
import asyncio, sys
sys.path.insert(0, ".")

# Bring in the refactored gpu.exec_command (uses run_subprocess_shell internally).
from src.kernelblaster.servers import gpu as gpu_mod

async def main():
    # 1) happy path: echo
    stdout, stderr = await gpu_mod.exec_command("echo hello-from-T4", timeout=10)
    assert "hello-from-T4" in stdout, f"unexpected stdout: {stdout!r}"

    # 2) non-zero exit -> raises GpuCommandError (factory preserved across refactor)
    try:
        await gpu_mod.exec_command("exit 23", timeout=10)
    except gpu_mod.GpuCommandError as e:
        assert "rc=23" in e.error_message, f"missing rc in error: {e.error_message!r}"
    else:
        raise AssertionError("expected GpuCommandError on non-zero exit")

    # 3) timeout -> raises GpuCommandError
    try:
        await gpu_mod.exec_command("sleep 10", timeout=0.5)
    except gpu_mod.GpuCommandError as e:
        assert "Timeout" in e.error_message, f"missing timeout marker: {e.error_message!r}"
    else:
        raise AssertionError("expected GpuCommandError on timeout")

    # 4) n_runs=3 returns lists
    stdout_list, stderr_list = await gpu_mod.exec_command("echo run", timeout=10, n_runs=3)
    assert isinstance(stdout_list, list) and len(stdout_list) == 3, stdout_list
    assert all("run" in s for s in stdout_list), stdout_list

asyncio.run(main())
print("T4: OK")
PYEOF
    then
        record "T4 gpu.exec_command direct (happy / rc!=0 / timeout / n_runs)" PASS
    else
        cat "$T4_OUT" | tail -30
        record "T4 gpu.exec_command direct (happy / rc!=0 / timeout / n_runs)" FAIL
    fi
else
    echo
    echo "==== T4 skipped (SKIP_T4=1) ===="
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo
echo "==== Summary ===="
echo "  passed:  $PASS_COUNT"
echo "  skipped: $SKIP_COUNT"
echo "  failed:  $FAIL_COUNT"
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo
    echo "FAIL — review logs:"
    if [ "${KEEP_LOGS:-0}" = "1" ]; then
        echo "  $TMPDIR/"
    else
        echo "  (rerun with KEEP_LOGS=1 to preserve logs)"
    fi
    exit 1
fi
if [ "$SKIP_COUNT" -gt 0 ]; then
    echo "OK (with skips — check messages above; re-run in an env that satisfies them for full coverage)"
else
    echo "OK"
fi
exit 0
