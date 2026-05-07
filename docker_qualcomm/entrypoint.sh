#!/bin/bash
set -e

# Default values if not provided
USER_NAME=${USER_NAME:-root}
USER_ID=${USER_ID:-0}
GROUP_ID=${GROUP_ID:-0}
HOME_DIR="/home/$USER_NAME"
RUN_AS_ROOT=${RUN_AS_ROOT:-false}

# Create group if it doesn't exist
GROUP_NAME="${GROUP_NAME:-$USER_NAME}"
if ! getent group "$GROUP_ID" > /dev/null 2>&1 && ! getent group "$GROUP_NAME" > /dev/null 2>&1; then
    groupadd -g "$GROUP_ID" "$GROUP_NAME" 2>/dev/null || \
    groupadd -g "$GROUP_ID" "g${GROUP_ID}" 2>/dev/null || true
fi

# Get the actual group name for the GID (in case it already existed)
ACTUAL_GROUP=$(getent group "$GROUP_ID" | cut -d: -f1)
if [ -z "$ACTUAL_GROUP" ]; then
    ACTUAL_GROUP="$GROUP_NAME"
fi

# Create user if it doesn't exist
if ! id -u "$USER_NAME" > /dev/null 2>&1; then
    useradd -u "$USER_ID" -g "$ACTUAL_GROUP" -m "$USER_NAME" -s /bin/bash
    echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
fi

echo "User: ${USER_NAME} with uid ${USER_ID} and gid ${GROUP_ID}."
echo "Run as root: ${RUN_AS_ROOT}"
echo " - To switch to a different user, call docker run with the following environment variables:"
echo "   - USER_NAME=\$(whoami)"
echo "   - USER_ID=\$(id -u)"
echo "   - GROUP_ID=\$(id -g)"
echo "   - RUN_AS_ROOT=true (to run as root instead of user)"

# Ensure home dir ownership (if re-used volumes are mounted)
chown "$USER_ID":"$GROUP_ID" "$HOME_DIR"

# Ensure /kernelblaster directory has proper permissions (if mounted volume)
if [ -d "/kernelblaster" ]; then
    chmod 755 /kernelblaster 2>/dev/null || true
    chown -R "$USER_ID":"$GROUP_ID" /kernelblaster 2>/dev/null || true
    chmod -R u+rwX,go+rX /kernelblaster 2>/dev/null || true
    chmod 755 /kernelblaster 2>/dev/null || true
fi

# Switch to user with proper environment
export HOME="$HOME_DIR"
cd "$HOME_DIR"

# Function to execute command with appropriate user
execute_command() {
    if [[ "$RUN_AS_ROOT" == "true" ]]; then
        echo "Executing as root..."
        exec sudo -E sh -c "cd /kernelblaster && exec \"\$@\"" -- "$@"
    else
        echo "Executing as user: $USER_NAME"
        exec sudo -E -u "$USER_NAME" sh -c "cd /kernelblaster && exec \"\$@\"" -- "$@"
    fi
}

# Set up SSH keys from host-mounted staging dir (/tmp/host_ssh_keys)
# or from a direct .ssh mount (legacy).
HOST_SSH_STAGING="/tmp/host_ssh_keys"
if [ -d "$HOST_SSH_STAGING" ]; then
    echo "Copying SSH keys from staging mount to $HOME_DIR/.ssh..."
    mkdir -p "$HOME_DIR/.ssh"
    cp -a "$HOST_SSH_STAGING/"* "$HOME_DIR/.ssh/" 2>/dev/null || true
    chown -R "$USER_NAME":"$GROUP_ID" "$HOME_DIR/.ssh"
    chmod 700 "$HOME_DIR/.ssh"
    chmod -R go-rwx "$HOME_DIR/.ssh"
elif [ -d "$HOME_DIR/.ssh" ]; then
    if sudo chown -R "$USER_NAME":"$GROUP_ID" "$HOME_DIR/.ssh" 2>/dev/null && \
       sudo chmod 700 "$HOME_DIR/.ssh" 2>/dev/null && \
       sudo chmod -R go-rwx "$HOME_DIR/.ssh" 2>/dev/null; then
        true
    else
        echo "WARNING: SSH dir exists but permissions cannot be fixed."
    fi
fi

# Create sshfs mount point (optional, disabled by default)
if [[ "${ENABLE_SSHFS:-false}" == "true" ]]; then
    sudo -u "$USER_NAME" mkdir -p "$HOME_DIR/sshfs_mnt" 2>/dev/null || true
    sudo chown "$USER_NAME":"$GROUP_ID" "$HOME_DIR/sshfs_mnt" 2>/dev/null || true
    sudo -u "$USER_NAME" sshfs -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER_NAME"@avs:/data/"$USER_NAME" "$HOME_DIR/sshfs_mnt" 2>/dev/null || \
        echo "Note: sshfs mount failed (optional feature)"
fi

# Ensure we can access /kernelblaster before trying to cd
if [ ! -d "/kernelblaster" ]; then
    mkdir -p /kernelblaster
    chown "$USER_ID":"$GROUP_ID" /kernelblaster
    chmod 755 /kernelblaster
fi

# Verify we can access the directory
if [ ! -r "/kernelblaster" ] || [ ! -x "/kernelblaster" ]; then
    echo "Warning: /kernelblaster is not accessible, attempting to fix permissions..."
    chmod 755 /kernelblaster 2>/dev/null || true
    chown "$USER_ID":"$GROUP_ID" /kernelblaster 2>/dev/null || true
fi

# Print OpenCL environment info
echo "=== Qualcomm Adreno OpenCL Development Environment ==="
echo "Platform: $(uname -m)"
echo "OpenCL headers: /usr/include/CL/"
echo "Adreno SDK: ${QUALCOMM_SDK_ROOT:-not set}"
if [ -e /dev/kgsl-3d0 ]; then
    echo "Adreno GPU: /dev/kgsl-3d0 (available)"
    clinfo 2>/dev/null | head -10 || echo "  (clinfo not available, but device present)"
else
    echo "Adreno GPU: not available (compile-only mode)"
fi
echo "======================================================="

# Handle different commands
case "${1:-api}" in
    "dev")
        echo "Starting development environment..."
        execute_command bash
        ;;
    "compile")
        echo "Starting OpenCL kernel compilation server..."
        execute_command python -m src.kernelblaster.servers.gpu --port 2002 --target adreno
        ;;
    "serve")
        # Long-lived: run both the OpenCL compile server and Adreno GPU server
        # so Claude Code (or any host client) can drive the kgen flow via `docker exec`.
        BOARD_HOST="${BOARD_HOST:-root@10.44.120.201}"
        COMPILE_PORT="${COMPILE_PORT:-6003}"
        GPU_PORT="${GPU_PORT:-6004}"
        ARTIFACTS_DIR="${ARTIFACTS_DIR:-/tmp/kernelblaster_kgen_serve}"
        echo "Starting kgen serve lifecycle (compile:${COMPILE_PORT} + gpu:${GPU_PORT})..."
        mkdir -p "$ARTIFACTS_DIR"
        chown -R "$USER_ID":"$GROUP_ID" "$ARTIFACTS_DIR" 2>/dev/null || true
        # Export server URLs so `docker exec` invocations of kgen_step.py pick up the right ports.
        export KERNELBLASTER_OPENCL_COMPILE_SERVER_URL="http://localhost:${COMPILE_PORT}"
        export KERNELBLASTER_ADRENO_GPU_SERVER_URL="http://localhost:${GPU_PORT}"
        execute_command bash -c "
            set -e
            cd /kernelblaster
            mkdir -p /kernelblaster/out/kgen_serve
            python -m src.kernelblaster.servers.compile_opencl \
                --port $COMPILE_PORT --num-workers 2 \
                --board-host '$BOARD_HOST' \
                --artifacts-dir '$ARTIFACTS_DIR' \
                > /kernelblaster/out/kgen_serve/compile.log 2>&1 &
            COMPILE_PID=\$!
            python -m src.kernelblaster.servers.gpu_adreno \
                --port $GPU_PORT --num-workers 1 \
                --board-host '$BOARD_HOST' \
                > /kernelblaster/out/kgen_serve/gpu.log 2>&1 &
            GPU_PID=\$!
            echo \"compile_opencl pid \$COMPILE_PID on :$COMPILE_PORT\"
            echo \"gpu_adreno     pid \$GPU_PID on :$GPU_PORT\"
            trap 'kill \$COMPILE_PID \$GPU_PID 2>/dev/null || true' TERM INT
            # Health probe — fail fast if either server never comes up
            for i in \$(seq 1 30); do
                sleep 1
                curl -sf http://localhost:$COMPILE_PORT/health >/dev/null || continue
                curl -sf http://localhost:$GPU_PORT/health     >/dev/null || continue
                echo 'Both servers healthy.'
                break
            done
            wait \$COMPILE_PID \$GPU_PID
        "
        ;;
    "api")
        echo "Starting API server..."
        execute_command python -m src.kernelblaster.servers.serve_api --port 8000 --output-dir out/server/ "${@:2}"
        ;;
    *)
        echo "Running custom command: $@"
        execute_command "$@"
        ;;
esac
