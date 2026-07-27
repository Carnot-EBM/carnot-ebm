#!/bin/bash
# Wait for the fixed-generator arm to finish, THEN run the pre-fix contention control.
# The two LLM arms must never overlap: they need different -c pools, and two servers would
# both double VRAM and change the very contention being measured.
while kill -0 2261356 2>/dev/null; do sleep 20; done
echo "=== llm_on_fix process exited; starting llm_on_16k contention control ==="
exec /home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python -u \
  results/first_win_llm_on_20260727/firstwin.py --arm llm_on_16k --k 4 --variants 1 \
  --budget 200 --port 8953 --gpu 1
