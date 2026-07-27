#!/bin/bash
# CONTROL-WINNER PROBE. Runs AFTER the 16k contention control so only one generator server
# exists at a time.
#
# The target games are computed AT RUN TIME from the llm_off control's ACTUAL winning cells,
# not hardcoded: the winner set under today's agent code is not the June baseline's (it has
# already moved from {lp85} to {lp85, sp80}), so a hardcoded list would have silently probed
# the wrong cells. Probing each winning GAME's full 4 variants guarantees >= 4 concurrent
# cells (so K=4 is actually reachable) and guarantees every winning signature is covered.
#
# These cells are written under *_probe arm labels so they can never pool into the
# pre-specified variant-1 slice's rate -- they are selected on the control's outcome and are
# therefore biased by construction. See analyse.py's control_winner_probe.
V=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
D=results/first_win_llm_on_20260727

while kill -0 2262470 2>/dev/null; do sleep 20; done
echo "=== 16k arm exited; computing control-winner games ==="
GAMES=$($V - <<'PY'
import glob, json
games = set()
for f in glob.glob("results/first_win_llm_on_20260727/cells/llm_off__*.json"):
    d = json.load(open(f))
    if d.get("first_win"):
        games.add(d["game"])
print(",".join(sorted(games)))
PY
)
echo "=== control-winner games: [$GAMES] ==="
if [ -z "$GAMES" ]; then
  echo "BLOCKED: the control arm recorded no wins, so there is no reachable-win cell to probe."
  exit 3
fi
echo "=== probe (FIXED generator) ==="
$V -u $D/firstwin.py --arm llm_on_fix_probe --k 4 --games "$GAMES" --variants 1,2,3,4 \
   --budget 200 --port 8954 --gpu 1
echo "=== probe (PRE-FIX 16k generator) ==="
$V -u $D/firstwin.py --arm llm_on_16k_probe --k 4 --games "$GAMES" --variants 1,2,3,4 \
   --budget 200 --port 8955 --gpu 1
echo "=== ALL PROBE ARMS DONE ==="
