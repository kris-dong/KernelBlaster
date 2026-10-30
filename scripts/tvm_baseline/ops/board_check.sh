#!/bin/bash
# read-only: foreign users / GPU users / lock holder / busy %. exit 0 = clear, 3 = foreign activity
ssh -o BatchMode=yes -o ConnectTimeout=25 qrb5165 'bash -s' <<'R'
echo "who: [$(who | tr '\n' ';')]"
echo "busy=$(cat /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage)"
for p in /proc/[0-9]*; do
  if ls -l $p/fd 2>/dev/null | grep -q kgsl; then echo "gpu-proc $(basename $p): $(tr '\0' ' ' < $p/cmdline | cut -c1-100)"; fi
done
echo "lock holders: $(fuser /root/mb_board.lock 2>&1 | head -2)"
ps -eo pid,etime,args | grep -E "user_mb|flock|driver" | grep -v grep | cut -c1-150
R
