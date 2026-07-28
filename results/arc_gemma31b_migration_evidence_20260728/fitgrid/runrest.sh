#!/bin/bash
# Wait for the in-flight config A to exit, then run the remaining configs SEQUENTIALLY.
# Sequential matters: each config tears its server down before the next launches, so we can
# never put two ~21 GB servers on one card.
cd /tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/fitgrid
PY=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
while pgrep -f "phase2.py A_egpu_81920_q8" > /dev/null; do sleep 10; done
echo "=== A done, starting queue ==="
# Order = decision value. The quality pair first (the operator explicitly asked and it is the
# cheapest), then the K=4 arm matched to the 340-495 s baseline, then the fallbacks.
for c in Q_egpu_24576_f16 Q_egpu_24576_q8 A4_egpu_81920_q8_K4 B_egpu_32768_q8 F_egpu_81920_q8_ffn12; do
  echo "=== QUEUE: $c ==="
  timeout 5400 $PY phase2.py "$c" 2>&1
done
echo "=== QUEUE COMPLETE ==="
