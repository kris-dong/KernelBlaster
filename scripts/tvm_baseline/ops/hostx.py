#!/usr/bin/env python3
"""Run things on a lab CPU host through the farm's ssh backend (CPU only, never a GPU).
  hostx.py [-H 277|278] run FILE.sh [timeout_s]     run a local script file on the host (stdin)
  hostx.py [-H 277|278] put LOCAL REMOTE            write a local file (under /raid/user/mb/...)
  hostx.py [-H 277|278] get REMOTE LOCAL            fetch a file
"""
import sys, shlex, os
from pathlib import Path
L = Path(os.environ.get("FARM_DIR", "/home/user/Documents/ModelBlaster-clone/l3_farm"))
sys.path.insert(0, str(L / "ssh_backend_share/tools")); sys.path.insert(0, str(L / "ssh_backend_share"))
import sshb_ctl
a = sys.argv[1:]
host = "277"
if a[0] == "-H": host = a[1]; a = a[2:]
b = sshb_ctl.make("tvmbase", "/raid/user/mb", hosts=[host], share=2)
nid = b.nid(host, 4, 0)
cmd = a[0]
if cmd == "run":
    body = Path(a[1]).read_bytes()
    rc, out, err = b.sh(nid, "bash -s", body, timeout=int(a[2]) if len(a) > 2 else 600)
    sys.stdout.write(out.decode(errors="replace")); sys.stderr.write(err.decode(errors="replace")[-2000:]); sys.exit(rc)
if cmd == "put":
    data = Path(a[1]).read_bytes()
    rc, out, err = b.sh(nid, "bash -c %s" % shlex.quote("mkdir -p $(dirname %s) && cat > %s" % (shlex.quote(a[2]), shlex.quote(a[2]))), data, timeout=900)
    print(rc, err.decode()[-300:]); sys.exit(rc)
if cmd == "get":
    rc, out, err = b.sh(nid, "cat %s" % shlex.quote(a[1]), b"", timeout=900)
    Path(a[2]).write_bytes(out); print(rc, len(out)); sys.exit(rc)
