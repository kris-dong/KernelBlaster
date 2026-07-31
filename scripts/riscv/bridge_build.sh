#!/usr/bin/env bash
# Bridge script the ZephyrCompileStrategy invokes for RISC-V builds
# against the modelblaster pipeline (P5.12).
#
# Semantics: the RL loop generates a candidate kernel implementation
# (a single .c file). We drop it into modelblaster's pre-staged
# ``generated/<TARGET>/kernels.c`` slot and run an INCREMENTAL west
# build (``-p never``) — the harness, driver, weights.c, io header,
# and every Zephyr subsystem are already compiled; only ``kernels.c``
# changes each step, so this is a fast recompile (typically < 20 s
# vs 60+ s for a pristine build).
#
# Contract with ZephyrCompileStrategy:
#   KERNELBLASTER_ZEPHYR_BUILD_CMD='bash <this> {job_name} {source_file} {output_path} {board}'
#
# Args this script parses:
#   $1 job_name       — kb_<problem>/opt_ncu_rl_optimized/<step>_<tech>.c-shaped
#   $2 source_file    — the RL-generated kernel .c
#   $3 output_path    — where to write the resulting zephyr.elf
#   $4 board          — Zephyr board id (spike_riscv64 for now)
#
# Env the caller MUST set (usually the runner script):
#   KERNELBLASTER_MODELBLASTER_STAGE_DIR  — modelblaster/examples/kernelbench/kb_<name>/<quant>
#   KERNELBLASTER_MODELBLASTER_TARGET     — scalar | rvv | rvv_f16 ...
#   ZEPHYR_BASE / ZEPHYR_SDK_INSTALL_DIR  — normal Zephyr env
#   PATH                                   — spike + Zephyr SDK on PATH
#   PYTHONPATH                             — modelblaster root (for west build's cmake python calls)

set -euo pipefail

JOB_NAME="$1"
SRC="$2"
OUT="$3"
BOARD="$4"

: "${KERNELBLASTER_MODELBLASTER_STAGE_DIR:?must be set (path to modelblaster/examples/kernelbench/kb_<name>/<quant>)}"
: "${KERNELBLASTER_MODELBLASTER_TARGET:=scalar}"

STAGE="${KERNELBLASTER_MODELBLASTER_STAGE_DIR}"
TARGET="${KERNELBLASTER_MODELBLASTER_TARGET}"

GEN_DIR="${STAGE}/generated/${TARGET}"
# Separate build dir per-worker so concurrent workers don't collide.
BUILD_DIR="${STAGE}/build/${TARGET}_rl_$(basename "${OUT%.elf}")"
KERNELS_C="${GEN_DIR}/kernels.c"

if [[ ! -f "${KERNELS_C}" ]]; then
    echo "ERROR: modelblaster stage not primed: ${KERNELS_C} does not exist." >&2
    echo "       Run 'BACKEND=reference bash modelblaster/examples/kernelbench/run_one.sh'" >&2
    echo "       once before starting the RL loop." >&2
    exit 1
fi

# Drop the RL-generated kernel into the staged slot. Back up the original
# once so a botched RL kernel never clobbers the reference baseline.
if [[ ! -f "${KERNELS_C}.orig" ]]; then
    cp "${KERNELS_C}" "${KERNELS_C}.orig"
fi
cp "${SRC}" "${KERNELS_C}"

# west build must be invoked from within the workspace. We chdir to the
# zephyr-chipyard-sw repo root and give it a repo-relative harness path.
# ``STAGE`` = ``.../modelblaster/examples/kernelbench/kb_<name>/<quant>``
# — 5 levels up lands at the chipyard-sw root, NOT 4 (that's
# ``modelblaster/``). Prefer the explicit
# ``KERNELBLASTER_MODELBLASTER_ROOT`` env var when set — the runner
# script always exports it.
if [[ -n "${KERNELBLASTER_MODELBLASTER_ROOT:-}" ]]; then
    REPO_ROOT="${KERNELBLASTER_MODELBLASTER_ROOT}"
else
    REPO_ROOT="$(cd "${STAGE}/../../../../.." && pwd)"
fi
cd "${REPO_ROOT}"

# If the build_dir doesn't exist yet, west needs a full config pass first
# (with -p auto). Subsequent calls use -p never for the incremental rebuild.
if [[ -d "${BUILD_DIR}" ]]; then
    PRISTINE=never
else
    PRISTINE=auto
fi

# Pull the target-specific kernel cflags (RVV, gemmini, etc. need them).
KERNEL_CFLAGS=$(python -c "
from modelblaster.pipeline.backends import get
b = get('${TARGET}')
print(';'.join(b.resolved_kernel_cflags('${REPO_ROOT}/modelblaster')))
" 2>/dev/null || echo "")

WEST_CMAKE_ARGS=(
    "-DMODEL_DIR=${GEN_DIR}"
    "-DMODELBLASTER_BACKEND=${TARGET}"
)
if [[ -n "${KERNEL_CFLAGS}" ]]; then
    WEST_CMAKE_ARGS+=("-DMODELBLASTER_KERNEL_CFLAGS=${KERNEL_CFLAGS}")
fi

# modelblaster's utilization-aware sizing (BENCH_TARGET_MB=256 default) can
# overflow the stock 256MB ram0 region. _run_lib.sh's auto-ram0 logic writes
# a ``ram0.overlay`` alongside the generated skeleton that resizes the RAM
# region for BOTH runners. The RL loop's bridge bypasses _run_lib.sh (goes
# straight to west build), so pick up the overlay here explicitly.
RAM0_OVERLAY="${STAGE}/generated/ram0.overlay"
if [[ -f "${RAM0_OVERLAY}" ]]; then
    WEST_CMAKE_ARGS+=("-DDTC_OVERLAY_FILE=${RAM0_OVERLAY}")
fi

# Baked io > 2GB (or the .mb_bigio section above the stock .bss layout)
# needs the RISC-V large code model. Honor CMODEL_LARGE=1 the same way
# _run_lib.sh does. Default off so small problems stay in the ±2GiB
# medany span; setting it once per RL run enables the same fitness the
# baseline build was configured with.
if [[ "${CMODEL_LARGE:-0}" == "1" ]]; then
    WEST_CMAKE_ARGS+=("-DCONFIG_RISCV_CMODEL_LARGE=y")
fi

west build -p "${PRISTINE}" -b "${BOARD}" modelblaster/harness \
    --build-dir "${BUILD_DIR}" \
    -- "${WEST_CMAKE_ARGS[@]}"

# Copy the built elf to the RL loop's expected output path.
ELF="${BUILD_DIR}/zephyr/zephyr.elf"
if [[ ! -f "${ELF}" ]]; then
    echo "ERROR: west build did not produce ${ELF}" >&2
    exit 2
fi
mkdir -p "$(dirname "${OUT}")"
cp "${ELF}" "${OUT}"

# Optional: stash the RL kernel source next to the build dir for
# post-mortem review.
cp "${SRC}" "${BUILD_DIR}/rl_kernel_source.c" 2>/dev/null || true
