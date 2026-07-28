#!/bin/bash
cd /tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/fitgrid || exit 1
PY=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
for c in QC_egpu_24576_f16_chat QC_egpu_24576_q8_chat; do
  echo "=== QC: $c ==="
  timeout 1200 "$PY" phase2.py "$c" 2>&1
done
echo "=== QC COMPLETE ==="
