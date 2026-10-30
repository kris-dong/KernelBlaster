#!/bin/bash
# prints LISTENING / WAITING_LOCK / DEAD
PORT=${1:-9191}
ssh -o BatchMode=yes -o ConnectTimeout=25 qrb5165 "bash -s $PORT" <<'R'
D=/root/krisd_mb_t
if ! { [ -f $D/rpc.pid ] && kill -0 $(cat $D/rpc.pid) 2>/dev/null; }; then echo DEAD; tail -3 $D/rpc.log 2>/dev/null; exit 0; fi
if ss -ltn 2>/dev/null | grep -q "127.0.0.1:$1 "; then echo "LISTENING $(ss -ltn | grep ":$1 " | head -2 | tr '\n' ' ')"; else echo WAITING_LOCK; fi
R
