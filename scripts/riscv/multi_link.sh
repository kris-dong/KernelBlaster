#!/usr/bin/env bash
# Fuse N same-problem kernel variants into one Zephyr boot via
# modelblaster/harness_shared_input. Bridges T3's fuse contract to the
# harness-shared-input west build.
#
# Contract (called by spike_exec/firesim_exec ``_link_batch_elf``):
#   - Reads stdin manifest lines of the form ``<tag> <staged_dir>``.
#   - Each ``staged_dir`` has a ``kernels.c`` with the LLM's variant.
#   - Writes fused ELF to ``$FUSED_OUT``.
#   - Env: ``KB_MULTI_MODEL_DIR`` — the shared base per-problem
#     ``generated/<target>/`` dir (holds model.c, weights.c, test_io.S
#     + baselines that all variants share).
#     ``KB_MULTI_TARGET`` — modelblaster backend name (rvv, scalar).
#     ``KERNELBLASTER_MODELBLASTER_ROOT`` — repo root that contains
#     ``modelblaster/harness_shared_input/``.
#     Standard Zephyr env: ``ZEPHYR_BASE``, ``ZEPHYR_SDK_INSTALL_DIR``,
#     ``PATH`` (west + gcc).
#   - Exits non-zero on any build failure. Emits BATCH_RELOC_ERROR_RE
#     substring on RISC-V PC-relative overflow so the caller can
#     shrink+retry.
set -euo pipefail

: "${FUSED_OUT:?FUSED_OUT env must point at the fused elf destination}"
: "${KB_MULTI_MODEL_DIR:?KB_MULTI_MODEL_DIR env must be set (base problem stage)}"
: "${KB_MULTI_TARGET:=rvv}"
: "${KERNELBLASTER_MODELBLASTER_ROOT:?KERNELBLASTER_MODELBLASTER_ROOT env must be set}"

REPO_ROOT="${KERNELBLASTER_MODELBLASTER_ROOT}"
HARNESS="${REPO_ROOT}/modelblaster/harness_shared_input"
if [[ ! -f "${HARNESS}/CMakeLists.txt" ]]; then
    echo "ERROR: harness_shared_input not present at ${HARNESS}" >&2
    exit 1
fi

# Parse the stdin manifest into parallel semicolon-lists that
# harness_shared_input's CMakeLists consumes as -DVARIANT_SOURCES /
# -DVARIANT_TAGS.
tags=""
srcs=""
while IFS=' ' read -r tag dir; do
    [[ -z "${tag}" ]] && continue
    src="${dir}/kernels.c"
    if [[ ! -f "${src}" ]]; then
        echo "ERROR: manifest row '${tag} ${dir}': missing ${src}" >&2
        exit 2
    fi
    tags+="${tags:+;}${tag}"
    srcs+="${srcs:+;}${src}"
done

if [[ -z "${tags}" ]]; then
    echo "ERROR: empty manifest on stdin" >&2
    exit 3
fi

# Board: chipyard_riscv64 for firesim, spike_riscv64 for spike. We
# infer from KB_MULTI_BOARD when set (T3 strategy fills it), else
# default to spike (safe for local smokes).
: "${KB_MULTI_BOARD:=spike_riscv64}"

# Kernel cflags pulled dynamically from modelblaster's backend (RVV
# needs -march=rv64gcv etc.). Fall back to empty if the query fails —
# the harness build will still succeed on scalar builds.
CF=$(python -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from modelblaster.pipeline.backends import get
print(';'.join(get('${KB_MULTI_TARGET}').resolved_kernel_cflags('${REPO_ROOT}/modelblaster')))
" 2>/dev/null || echo "")

BUILD_DIR="${FUSE_BUILD_DIR:-${FUSED_OUT%.elf}_build_dir}"
rm -rf "${BUILD_DIR}"

# Prime the auto-ram0 overlay for the shared-input case: shared input +
# shared golden + N * output. modelblaster's _run_lib.sh has the exact
# formula, but for simplicity we hand-write the overlay when the
# baseline stage has an io.npz > 1MB (small workloads skip it).
_IO_NPZ="${KB_MULTI_MODEL_DIR%/generated/*}/generated/io.npz"
_OVERLAY=""
if [[ -f "${_IO_NPZ}" ]]; then
    _BYTES=$(python -c "
import numpy as np, sys
d = np.load(sys.argv[1])
n = int(d['input'].nbytes) + int(d['output'].nbytes)
# input + golden shared once; per-variant output * N.
n_vars = int(sys.argv[2])
out_bytes = int(d['output'].nbytes)
total = n + n_vars * out_bytes + 128*1024*1024
# Round up to a 64MB boundary matching _run_lib.sh's convention.
mask = 0x3FFFFFF
total = ((total + mask) // (mask + 1)) * (mask + 1)
print(hex(total))
" "${_IO_NPZ}" "$(echo "${tags}" | tr ';' '\n' | wc -l)" 2>/dev/null || echo "")
    if [[ -n "${_BYTES}" && "${_BYTES}" != "0x0" ]]; then
        _OVERLAY="${BUILD_DIR%/*}/ram0_shared.overlay"
        mkdir -p "$(dirname "${_OVERLAY}")"
        printf '&ram0 { reg = < 0x80000000 %s >; };\n' "${_BYTES}" > "${_OVERLAY}"
    fi
fi

echo "multi_link: harness_shared_input build board=${KB_MULTI_BOARD} tags=${tags} ram0_overlay=${_OVERLAY:-none}" >&2

WEST_EXTRA=()
[[ -n "${_OVERLAY}" ]] && WEST_EXTRA+=("-DEXTRA_DTC_OVERLAY_FILE=${_OVERLAY}")
[[ -n "${CF}" ]] && WEST_EXTRA+=("-DMODELBLASTER_KERNEL_CFLAGS=${CF}")
[[ "${CMODEL_LARGE:-0}" == "1" ]] && WEST_EXTRA+=("-DCONFIG_RISCV_CMODEL_LARGE=y")

cd "${REPO_ROOT}"
if ! west build -p auto -b "${KB_MULTI_BOARD}" modelblaster/harness_shared_input \
        --build-dir "${BUILD_DIR}" -- \
        -DMODELBLASTER_BACKEND="${KB_MULTI_TARGET}" \
        -DMODEL_DIR="${KB_MULTI_MODEL_DIR}" \
        -DVARIANT_SOURCES="${srcs}" \
        -DVARIANT_TAGS="${tags}" \
        "${WEST_EXTRA[@]}" >&2 ; then
    # Re-run just cmake+ninja capturing full output so caller can
    # detect BATCH_RELOC_ERROR_RE (R_RISCV_PCREL_HI20 overflow).
    exit 4
fi

ELF="${BUILD_DIR}/zephyr/zephyr.elf"
if [[ ! -f "${ELF}" ]]; then
    echo "ERROR: west build did not produce ${ELF}" >&2
    exit 5
fi
mkdir -p "$(dirname "${FUSED_OUT}")"
cp "${ELF}" "${FUSED_OUT}"
echo "multi_link: fused ELF -> ${FUSED_OUT}" >&2
