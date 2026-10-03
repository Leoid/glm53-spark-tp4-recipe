#!/usr/bin/env bash
# keys wrapper: drop page caches on all 4 ranks (NFS weight reads inflate "used" and trip vLLM's
# startup free-memory guard on unified-memory GB10), then launch with the standard skip flags.
# Usage: [ENFORCE_EAGER=0] [EXL3_FUSED_MOE=1] [GPU_MEM_UTIL=0.86] ./keys-launch.sh [logname]
set -u
cd ~/glm53-mia
LOG=${1:-launch-tp4-keys.log}
./start-tp4.sh stop 2>&1 | tail -1
for ip in 10.100.10.1 10.100.10.2 10.100.10.3 10.100.10.5; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$ip" "sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; free -g | awk '/Mem:/{print \"\" \" free=\" \$4 \"G avail=\" \$7 \"G\"}'" 2>/dev/null || echo "$ip: drop_caches failed"
done
SKIP_PULL=1 SKIP_SHIP=1 SKIP_BUILD=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 SKIP_OVERLAY_VERIFY=1 \
  ENFORCE_EAGER=${ENFORCE_EAGER:-0} EXL3_FUSED_MOE=${EXL3_FUSED_MOE:-1} \
  setsid nohup ./start-tp4.sh > "$LOG" 2>&1 < /dev/null &
echo "launched -> $LOG"
