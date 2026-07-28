#!/bin/bash
# Sequential (never two ~21 GB servers on one card). Order = decision value:
#   AC  = the PREFERRED config as it SHOULD run (chat template) -> answers the directive
#   QC pair = the f16-vs-q8_0 KV quality check the operator explicitly asked for
#   AC4 = the K=4 shape, the only one comparable to the 340-495 s baseline
#   FC/IC = fallbacks (a) FFN offload and (b) iGPU
cd /tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/fitgrid
PY=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
# WAIT BY PID, NOT BY PATTERN. A `pgrep -f "phase2.py A_egpu..."` here matched this script's own
# parent wrapper (whose command line contains the pattern as literal text) and would have waited
# on itself forever -- the same self-match hazard as `pkill -f` killing your own command.
WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  while [ -d "/proc/$WAIT_PID" ]; do sleep 10; done
fi
echo "=== raw-arm A (pid $WAIT_PID) done; starting chat queue ==="
for c in AC_egpu_81920_q8_chat QC_egpu_24576_f16_chat QC_egpu_24576_q8_chat AC4_egpu_81920_q8_chat_K4 FC_egpu_81920_q8_ffn12_chat; do
  echo "=== QUEUE: $c ==="
  timeout 5400 $PY phase2.py "$c" 2>&1
done
echo "=== QUEUE COMPLETE ==="
