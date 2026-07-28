#!/bin/bash
# Re-run the arms lost to the external pattern-kill, now that our server runs as `p2srv`
# (a same-directory symlink to the identical llama-server binary) and is therefore no longer
# matched by a `pkill -f llama-server`.
#
# Sequential: each config tears its own server down before the next launches, so we can never
# put two large servers on one card.
cd /tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/fitgrid || exit 1
PY=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
for c in QC_egpu_24576_f16_chat QC_egpu_24576_q8_chat AC4_egpu_81920_q8_chat_K4 FC_egpu_81920_q8_ffn12_chat; do
  echo "=== RERUN: $c ==="
  timeout 5400 "$PY" phase2.py "$c" 2>&1
done
echo "=== RERUN QUEUE COMPLETE ==="
