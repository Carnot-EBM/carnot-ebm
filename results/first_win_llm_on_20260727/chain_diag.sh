#!/bin/bash
V=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
D=results/first_win_llm_on_20260727
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if [ -f $D/run_llm_on_fix_diag.json ]; then echo "=== run file present; arm complete ==="; break; fi
  n=$(ls $D/cells/ | grep -c '^llm_on_fix_diag__')
  echo "=== attempt $attempt: $n/25 cells banked, port $((9010+attempt)) ==="
  $V -u $D/firstwin.py --arm llm_on_fix_diag --k 4 --variants 1 --budget 200 \
     --port $((9010+attempt)) --gpu 1
  echo "=== attempt $attempt returned (exit $?) ==="
  sleep 10
done
echo "=== DIAG SUPERVISOR DONE: $(ls $D/cells/ | grep -c '^llm_on_fix_diag__')/25 ==="
