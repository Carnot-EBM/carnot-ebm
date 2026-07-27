#!/bin/bash
# Finish the pre-specified variant-1 slice for the FIXED generator.
#
# WHY A SUPERVISOR LOOP: this arm has now been terminated twice by an external SIGTERM
# (log_llm_on_fix.txt truncated at 10/25 with no traceback; log_rest.txt recorded a literal
# "Terminated" at 11/25), on both occasions with no OOM evidence and once while already in its
# own setsid session. Cause remains unproven, so rather than assert a diagnosis this simply
# retries until the arm actually writes its run file. Resumption is lossless -- run_cell()
# returns the cached row for every cell already banked -- so a retry only ever executes the
# cells that are genuinely missing, and it can never re-run or overwrite a completed cell.
# Bounded at 8 attempts so a genuine hard failure surfaces instead of looping forever.
V=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
D=results/first_win_llm_on_20260727
for attempt in 1 2 3 4 5 6 7 8; do
  if [ -f $D/run_llm_on_fix.json ]; then echo "=== run file present; arm complete ==="; break; fi
  n=$(ls $D/cells/ | grep -c '^llm_on_fix__')
  echo "=== attempt $attempt: $n/25 cells banked, port $((8970+attempt)) ==="
  $V -u $D/firstwin.py --arm llm_on_fix --k 4 --variants 1 --budget 200 \
     --port $((8970+attempt)) --gpu 1
  echo "=== attempt $attempt returned (exit $?) ==="
  sleep 10
done
echo "=== FIX ARM SUPERVISOR DONE: $(ls $D/cells/ | grep -c '^llm_on_fix__')/25 cells ==="
