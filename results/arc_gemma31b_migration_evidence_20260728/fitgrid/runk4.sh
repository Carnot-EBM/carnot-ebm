#!/bin/bash
cd /tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/fitgrid || exit 1
PY=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
timeout 2400 "$PY" phase2.py AC4_egpu_81920_q8_chat_K4 2>&1
echo "=== K4 COMPLETE ==="
