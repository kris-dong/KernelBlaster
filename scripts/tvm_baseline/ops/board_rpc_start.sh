#!/bin/bash
# usage: board_rpc_start.sh [PORT=9191] [MAX_RUN_S=1500] [LOCK_WAIT_S=7200]
# Refuses (exit 3) when a foreign user / GPU process is present; otherwise starts tvm_rpc (127.0.0.1 ONLY) UNDER the board lock
# (it only starts serving once flock is acquired). Returns immediately; poll with board_rpc_status.sh.
PORT=${1:-9191}; MAXRUN=${2:-1500}; LW=${3:-7200}
ssh -o BatchMode=yes -o ConnectTimeout=25 qrb5165 "bash -s $PORT $MAXRUN $LW" <<'R'
PORT=$1; MAXRUN=$2; LW=$3
D=/root/krisd_mb_t
if [ -n "$(who)" ]; then echo "FOREIGN who: $(who | tr '\n' ';')"; exit 3; fi
for p in /proc/[0-9]*; do
  pid=$(basename $p)
  if ls -l $p/fd 2>/dev/null | grep -q kgsl; then
    c=$(tr '\0' ' ' < $p/cmdline)
    case "$c" in weston*) ;; *krisd_mb*) ;; *) echo "FOREIGN gpu-proc $pid: ${c:0:100}"; exit 3;; esac
  fi
done
if [ -f $D/rpc.pid ] && kill -0 $(cat $D/rpc.pid) 2>/dev/null; then echo "ALREADY pid $(cat $D/rpc.pid)"; exit 0; fi
mkdir -p $D/work
cd $D
LD_LIBRARY_PATH=$D/rt nohup flock -w $LW /root/mb_board.lock timeout $MAXRUN $D/rt/tvm_rpc server --host=127.0.0.1 --port=$PORT --port-end=$((PORT+1)) --work-dir=$D/work > $D/rpc.log 2>&1 < /dev/null &
echo $! > $D/rpc.pid
echo "STARTED pid $(cat $D/rpc.pid)"
R
