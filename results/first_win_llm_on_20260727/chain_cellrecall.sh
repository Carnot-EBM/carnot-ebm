#!/bin/bash
# Supervisor loop for the cell_recall diagnostic arm -- same pattern, and the same reason, as
# finish_fix.sh: this box terminates long ARC arms with an external SIGTERM (no traceback, no
# OOM evidence). Resumption is lossless because run_cell() returns the cached row for any cell
# already banked, so a retry only executes the genuinely-missing cells and can never re-run or
# overwrite a completed one. Bounded so a genuine hard failure surfaces instead of looping.
V=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
D=results/first_win_llm_on_20260727
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if [ -f $D/run_llm_on_fix_cellrecall.json ]; then echo "=== run file present; arm complete ==="; break; fi
  n=$(ls $D/cells/ | grep -c '^llm_on_fix_cellrecall__')
  echo "=== attempt $attempt: $n/25 cells banked, port $((8990+attempt)) ==="
  $V -u $D/firstwin.py --arm llm_on_fix_cellrecall --k 4 --variants 1 --budget 200 \
     --port $((8990+attempt)) --gpu 1
  echo "=== attempt $attempt returned (exit $?) ==="
  sleep 10
done
echo "=== CELLRECALL SUPERVISOR DONE: $(ls $D/cells/ | grep -c '^llm_on_fix_cellrecall__')/25 ==="
