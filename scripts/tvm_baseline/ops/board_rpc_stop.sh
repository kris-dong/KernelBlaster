#!/bin/bash
# stop ONLY our own server (pid file) and its children; remove uploaded .so files from the work dir
ssh -o BatchMode=yes -o ConnectTimeout=25 qrb5165 'bash -s' <<'R'
D=/root/user_mb_t
if [ -f $D/rpc.pid ]; then P=$(cat $D/rpc.pid); pkill -P $P 2>/dev/null; kill $P 2>/dev/null; sleep 1; kill -9 $P 2>/dev/null; rm -f $D/rpc.pid; fi
pkill -f "$D/rt/tvm_rpc" 2>/dev/null; sleep 1
rm -rf $D/work/* 2>/dev/null
pgrep -f "$D/rt/tvm_rpc" >/dev/null && echo "STILL RUNNING" || echo STOPPED
du -sh $D | tail -1
R
