#!/bin/bash
# Sequential remainder: resume the truncated fixed arm, then both control-winner probes.
# Only ONE generator server exists at any time (each run tears its own down by explicit pid).
#
# WHY setsid (see RUN_LOG.md). The first llm_on_fix relaunch died SILENTLY at 10/25 cells --
# no traceback, no OOM evidence (125GB RAM, ~92GB available, zero memory pressure), log
# truncated mid-batch at 12:11:48 UTC. The conductor SIGKILLs whole PROCESS GROUPS
# (research_conductor.py:517, :940 os.killpg(pgid, SIGKILL)) and its iteration cycled at
# 12:10/12:12 UTC. Cause is NOT proven -- it is a timing coincidence plus a plausible
# mechanism -- but running each arm in its OWN session makes the whole class of
# group-directed signal collateral impossible, which is cheap insurance either way.
#
# Resumption is safe and lossless: run_cell() returns the cached row when its cell file
# already exists, so the 10 completed cells are reused untouched and only the missing 15 run.
# measurement_wall_s_from_rows sums every row's OWN elapsed_s, cached ones included, so the
# published measurement clock covers the whole arm, not just the resumed tail.
V=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
D=results/first_win_llm_on_20260727

true  # 16k arm already complete; its leaked server was reaped by port ownership
echo "=== 16k arm finished; RESUMING the truncated fixed arm (10/25 cells already banked) ==="
$V -u $D/firstwin.py --arm llm_on_fix --k 4 --variants 1 --budget 200 --port 8961 --gpu 1
echo "=== fixed arm complete; computing control-winner games ==="
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
  echo "BLOCKED: control arm recorded no wins; no reachable-win cell exists to probe."
  exit 3
fi
echo "=== probe (FIXED generator) ==="
$V -u $D/firstwin.py --arm llm_on_fix_probe --k 4 --games "$GAMES" --variants 1,2,3,4 \
   --budget 200 --port 8962 --gpu 1
echo "=== probe (PRE-FIX 16k generator) ==="
$V -u $D/firstwin.py --arm llm_on_16k_probe --k 4 --games "$GAMES" --variants 1,2,3,4 \
   --budget 200 --port 8963 --gpu 1
echo "=== ALL REMAINING ARMS DONE ==="
