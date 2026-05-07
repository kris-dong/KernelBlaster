#!/bin/bash
# Verify the 6 patched sol-level2 drivers via the existing kgen pipeline:
#   - validate     : static rules + launcher contract
#   - run-dummy    : compile+run driver against a no-op kernel; MUST print 'failed'
#                    (this is the anti-stub check — if it prints 'passed', the
#                     driver's verification is bypassable, which was the bug we
#                     just fixed)
#   - run-real     : compile+run driver against the genuine final_cuda.cu;
#                    MUST print 'passed' (proves the in-driver torch-op
#                    reference matches the kernel within tolerance)
#
# Usage:
#   scripts/verify_patched_drivers.sh                   # all 6 patched problems
#   PROBLEMS="002_..." scripts/verify_patched_drivers.sh # subset
#
# Env overrides:
#   CONTAINER_NAME (default: kernelblaster-serve)
#   GPU (default: l40s)
#   TIMEOUT (per-step seconds, default: 600)
#
# Exit codes: 0 = all 6 verified, 1 = at least one failed.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONTAINER_NAME="${CONTAINER_NAME:-kernelblaster-serve}"
GPU="${GPU:-l40s}"
TIMEOUT="${TIMEOUT:-600}"

DEFAULT_PROBLEMS=(
    001_fused_vision_multihead_attention_with_norms_backward
    002_decoder_layer_full_block
    004_fused_residual_rms_mlp
    007_multimodal_rotary_embedding_attention
    019_decoder_layer_fused_attention_mlp
    021_cross_attention_text_video_conditioning_backward
)
if [ -n "${PROBLEMS:-}" ]; then
    # shellcheck disable=SC2206
    PROBLEMS=( $PROBLEMS )
else
    PROBLEMS=( "${DEFAULT_PROBLEMS[@]}" )
fi

echo "================================================================"
echo "  KernelBlaster — verify patched drivers"
echo "================================================================"
echo "  container: $CONTAINER_NAME"
echo "  gpu:       $GPU"
echo "  problems:  ${#PROBLEMS[@]}"
echo "  timeout:   ${TIMEOUT}s"
echo "================================================================"

if ! docker ps --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "ERROR: container '$CONTAINER_NAME' is not running."
    echo "Start it via:  ./docker/run.sh serve"
    exit 2
fi

# Per-problem results (in order)
RESULTS=()
LOG_DIR="${REPO_ROOT}/out/verify_patched_drivers"
mkdir -p "$LOG_DIR"

for prob in "${PROBLEMS[@]}"; do
    echo ""
    echo "----------------------------------------------------------------"
    echo "  $prob"
    echo "----------------------------------------------------------------"
    folder_in_container="data/kernelbench-cuda/sol-level2/$prob"
    folder_on_host="$REPO_ROOT/$folder_in_container"

    if [ ! -d "$folder_on_host" ]; then
        echo "  SKIP: folder not found at $folder_on_host"
        RESULTS+=( "skip:$prob" )
        continue
    fi
    if [ ! -f "$folder_on_host/driver.cpp" ] || [ ! -f "$folder_on_host/final_cuda.cu" ]; then
        echo "  SKIP: driver.cpp or final_cuda.cu missing"
        RESULTS+=( "skip:$prob" )
        continue
    fi

    log="$LOG_DIR/$prob.log"
    rc=0
    # The repo is bind-mounted at /kernelblaster inside the container, but
    # the default WORKDIR is /workspace. Set -w explicitly.
    docker exec -u "$(id -u):$(id -g)" -w /kernelblaster "$CONTAINER_NAME" \
        python scripts/kgen_step_cuda.py full \
        --folder "$folder_in_container" \
        --gpu "$GPU" \
        --timeout "$TIMEOUT" > "$log" 2>&1 || rc=$?

    tail_lines=$(tail -n 8 "$log")
    echo "$tail_lines" | sed 's/^/    /'

    if [ "$rc" -eq 0 ]; then
        echo "  ✓ verified (validate + run-dummy + run-real)"
        RESULTS+=( "ok:$prob" )
    else
        echo "  ✗ FAILED (rc=$rc) — full log: $log"
        RESULTS+=( "fail:$prob:$rc" )
    fi
done

echo ""
echo "================================================================"
echo "  Summary"
echo "================================================================"
ok=0; fail=0; skip=0
for r in "${RESULTS[@]}"; do
    case "$r" in
        ok:*)   echo "   ✓ ${r#ok:}";   ok=$((ok+1)) ;;
        fail:*) name_rc="${r#fail:}";   echo "   ✗ ${name_rc%:*} (rc=${name_rc##*:})"; fail=$((fail+1)) ;;
        skip:*) echo "   - ${r#skip:} (skipped)";  skip=$((skip+1)) ;;
    esac
done
echo "----------------------------------------------------------------"
echo "  total: ${#RESULTS[@]}  passed: $ok  failed: $fail  skipped: $skip"
echo "  per-problem logs: $LOG_DIR/"

if [ "$fail" -gt 0 ]; then
    exit 1
fi
exit 0
