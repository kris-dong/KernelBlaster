#!/bin/bash
ssh -o BatchMode=yes -o ConnectTimeout=25 qrb5165 'ps -eo pid,args | grep "[k]risd_mb_t" || echo none; ls -d /root/krisd_mb_t 2>&1'
