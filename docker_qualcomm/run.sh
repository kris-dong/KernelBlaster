#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-kernelblaster-qualcomm}"
CMD="${1:-dev}"

# `serve` uses a distinct name so it can coexist with dev containers.
if [ "$CMD" = "serve" ]; then
    CONTAINER_NAME="${CONTAINER_NAME:-kernelblaster-qualcomm-serve}"
else
    CONTAINER_NAME="${CONTAINER_NAME:-kernelblaster-qualcomm}"
fi

DEVICE_FLAGS=""
# Pass the Adreno GPU device into the container
if [ -e /dev/kgsl-3d0 ]; then
    echo "Adreno GPU detected (/dev/kgsl-3d0)"
    DEVICE_FLAGS="--device /dev/kgsl-3d0"
fi

# The user's ~/.ssh lives on an NFS mount with root_squash, so container-root
# cannot read it. Stage keys onto local disk (the repo lives on /scratch/*,
# typically ext4) and mount that instead. This is the location the updated
# entrypoint.sh expects at /tmp/host_ssh_keys.
SSH_FLAGS=""
SSH_STAGE_DIR="${REPO_ROOT}/.ssh_stage"
if [ -d "${HOME}/.ssh" ]; then
    mkdir -p "$SSH_STAGE_DIR"
    chmod 700 "$SSH_STAGE_DIR"
    # Only copy what ssh actually needs — skip authorized_keys etc.
    for f in id_rsa id_rsa.pub id_ed25519 id_ed25519.pub known_hosts config; do
        if [ -e "${HOME}/.ssh/$f" ]; then
            cp -a "${HOME}/.ssh/$f" "$SSH_STAGE_DIR/"
        fi
    done
    chmod 600 "$SSH_STAGE_DIR"/* 2>/dev/null || true
    SSH_FLAGS="-v ${SSH_STAGE_DIR}:/tmp/host_ssh_keys:ro"
    echo "Staged SSH keys at ${SSH_STAGE_DIR} (container-readable)"
fi

# Always bind-mount the up-to-date entrypoint.sh over the baked-in copy —
# the current image predates the /tmp/host_ssh_keys staging support.
ENTRYPOINT_OVERRIDE="-v ${REPO_ROOT}/docker_qualcomm/entrypoint.sh:/entrypoint.sh:ro"
chmod +x "${REPO_ROOT}/docker_qualcomm/entrypoint.sh" 2>/dev/null || true

if [ "$CMD" = "serve" ]; then
    # Detached daemon that exposes 6003/6004 on localhost for `docker exec` clients.
    # If a serve container already exists, reuse it rather than clobbering state.
    if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
        EXISTING_STATE=$(docker inspect -f '{{.State.Status}}' "${CONTAINER_NAME}")
        if [ "$EXISTING_STATE" = "running" ]; then
            echo "Serve container ${CONTAINER_NAME} already running — leaving it."
            docker ps --filter "name=${CONTAINER_NAME}"
            exit 0
        fi
        echo "Removing stopped container ${CONTAINER_NAME} (state: $EXISTING_STATE)"
        docker rm "${CONTAINER_NAME}" >/dev/null
    fi

    BOARD_HOST="${BOARD_HOST:-root@10.44.120.201}"
    COMPILE_PORT="${COMPILE_PORT:-6003}"
    GPU_PORT="${GPU_PORT:-6004}"

    echo "Launching detached serve container: ${CONTAINER_NAME}"
    CID=$(docker run -d \
        --name="${CONTAINER_NAME}" \
        --restart=unless-stopped \
        ${DEVICE_FLAGS} \
        ${SSH_FLAGS} \
        ${ENTRYPOINT_OVERRIDE} \
        -e USER_NAME="$(whoami)" \
        -e USER_ID="$(id -u)" \
        -e GROUP_ID="$(id -g)" \
        -e BOARD_HOST="${BOARD_HOST}" \
        -e COMPILE_PORT="${COMPILE_PORT}" \
        -e GPU_PORT="${GPU_PORT}" \
        -e KERNELBLASTER_OPENCL_COMPILE_SERVER_URL="http://localhost:${COMPILE_PORT}" \
        -e KERNELBLASTER_ADRENO_GPU_SERVER_URL="http://localhost:${GPU_PORT}" \
        -v "${REPO_ROOT}:/kernelblaster" \
        -p "127.0.0.1:${COMPILE_PORT}:${COMPILE_PORT}" \
        -p "127.0.0.1:${GPU_PORT}:${GPU_PORT}" \
        "${IMAGE_NAME}" \
        serve)
    echo "Container: ${CID}"
    echo ""
    echo "Tailing first 30s of startup (logs: out/kgen_serve/{compile,gpu}.log)..."
    timeout 30 docker logs -f "${CONTAINER_NAME}" 2>&1 | sed 's/^/  /' || true
    echo ""
    echo "To run the kgen helper:"
    echo "  docker exec -u $(whoami) ${CONTAINER_NAME} python scripts/kgen_step.py --help"
    echo "To stop:"
    echo "  docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}"
    exit 0
fi

echo "Launching container: ${CONTAINER_NAME} (command: ${CMD})"

docker run --rm -it \
    --name="${CONTAINER_NAME}" \
    ${DEVICE_FLAGS} \
    ${SSH_FLAGS} \
    ${ENTRYPOINT_OVERRIDE} \
    -e USER_NAME="$(whoami)" \
    -e USER_ID="$(id -u)" \
    -e GROUP_ID="$(id -g)" \
    -v "${REPO_ROOT}:/kernelblaster" \
    "${IMAGE_NAME}" \
    "${CMD}" "${@:2}"
