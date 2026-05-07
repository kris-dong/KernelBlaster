#!/bin/bash
# Launch the Qualcomm docker container with the KernelBlaster repo mounted
# and run the OpenCL compile + GPU server test flow.
#
# Usage:
#   ./docker_qualcomm/run_server_test.sh              # Run the server test
#   ./docker_qualcomm/run_server_test.sh dev          # Get a shell inside the container
#   BOARD_HOST=root@1.2.3.4 ./docker_qualcomm/run_server_test.sh  # Custom board
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-kernelblaster-qualcomm}"
CONTAINER_NAME="${CONTAINER_NAME:-kernelblaster-qualcomm-test}"
BOARD_HOST="${BOARD_HOST:-root@10.44.120.201}"
CMD="${1:-test}"

LOCAL_USER="$(whoami)"
LOCAL_UID="$(id -u)"
LOCAL_GID="$(id -g)"

echo "=== KernelBlaster OpenCL Server Test ==="
echo "Image: ${IMAGE_NAME}"
echo "Container: ${CONTAINER_NAME}"
echo "Repo root: ${REPO_ROOT}"
echo "Board host: ${BOARD_HOST}"
echo "User: ${LOCAL_USER} (uid=${LOCAL_UID}, gid=${LOCAL_GID})"
echo "Command: ${CMD}"
echo ""

SSH_FLAGS=""
if [ -d "${HOME}/.ssh" ]; then
    echo "Mounting host SSH keys (read-only)"
    SSH_FLAGS="-v ${HOME}/.ssh:/tmp/host_ssh_keys:ro"
fi

case "${CMD}" in
    "test")
        echo "Running server test flow..."
        docker run --rm \
            --name="${CONTAINER_NAME}" \
            ${SSH_FLAGS} \
            -e USER_NAME="${LOCAL_USER}" \
            -e USER_ID="${LOCAL_UID}" \
            -e GROUP_ID="${LOCAL_GID}" \
            -e BOARD_HOST="${BOARD_HOST}" \
            -v "${REPO_ROOT}:/kernelblaster" \
            -p 127.0.0.1:6003:6003 \
            -p 127.0.0.1:6004:6004 \
            "${IMAGE_NAME}" \
            bash -c "cd /kernelblaster && bash docker_qualcomm/test_servers.sh"
        ;;
    "dev")
        echo "Starting dev shell (repo mounted at /kernelblaster)..."
        docker run --rm -it \
            --name="${CONTAINER_NAME}" \
            ${SSH_FLAGS} \
            -e USER_NAME="${LOCAL_USER}" \
            -e USER_ID="${LOCAL_UID}" \
            -e GROUP_ID="${LOCAL_GID}" \
            -e BOARD_HOST="${BOARD_HOST}" \
            -v "${REPO_ROOT}:/kernelblaster" \
            -p 127.0.0.1:6003:6003 \
            -p 127.0.0.1:6004:6004 \
            "${IMAGE_NAME}" \
            dev
        ;;
    *)
        echo "Running custom command: ${@:2}"
        docker run --rm -it \
            --name="${CONTAINER_NAME}" \
            ${SSH_FLAGS} \
            -e USER_NAME="${LOCAL_USER}" \
            -e USER_ID="${LOCAL_UID}" \
            -e GROUP_ID="${LOCAL_GID}" \
            -e BOARD_HOST="${BOARD_HOST}" \
            -v "${REPO_ROOT}:/kernelblaster" \
            -p 127.0.0.1:6003:6003 \
            -p 127.0.0.1:6004:6004 \
            "${IMAGE_NAME}" \
            "${@}"
        ;;
esac
