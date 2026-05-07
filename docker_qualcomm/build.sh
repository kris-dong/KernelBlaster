#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-kernelblaster-qualcomm}"

echo "Building Qualcomm Adreno OpenCL development image: ${IMAGE_NAME}"
echo "Context: ${REPO_ROOT}"

docker build \
    "${REPO_ROOT}" \
    -t "${IMAGE_NAME}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "$@"

echo ""
echo "Build complete: ${IMAGE_NAME}"
echo "Run with: ${SCRIPT_DIR}/run.sh"
