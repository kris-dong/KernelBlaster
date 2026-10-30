#!/bin/bash
# local port forward Mac:PORT -> board 127.0.0.1:PORT (the RPC server listens on the board's localhost only)
PORT=${1:-9191}
exec ssh -N -o BatchMode=yes -o ConnectTimeout=25 -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=6 -L 127.0.0.1:$PORT:127.0.0.1:$PORT qrb5165
