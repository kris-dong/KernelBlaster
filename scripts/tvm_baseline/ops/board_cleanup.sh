#!/bin/bash
ssh -o BatchMode=yes -o ConnectTimeout=25 qrb5165 'pgrep -f "user_mb_t/rt/tvm_rpc" && echo "RPC STILL RUNNING"; rm -rf /root/user_mb_t; ls -d /root/user_mb_t 2>&1; df -h / | tail -1; df -BM / | tail -1'
