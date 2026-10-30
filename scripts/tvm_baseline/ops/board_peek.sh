#!/bin/bash
ssh -o BatchMode=yes -o ConnectTimeout=25 qrb5165 'D=/root/krisd_mb_t; ls -la $D $D/work; tail -5 $D/rpc.log; echo busy=$(cat /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage); ps -eo pid,etime,args | grep -E "tvm_rpc|flock" | grep -v grep | cut -c1-140'
