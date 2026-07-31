#!/usr/bin/env bash
# PHASE 2, STEP 4c -- run the three-arm repair A/B across the five audited games.
#
# Two games at a time, one per idle 3090, on NON-DEFAULT ports (8919 is the default and a stale
# server there is silently adopted). Each cell proves its own CUDA build from /proc/<pid>/exe and
# its own per-PID VRAM row before generating anything, and writes its own ab.json -- so a cell
# that dies leaves a missing observation rather than corrupting a shared file.
#
# Games are NOT interleaved on one GPU: each cell starts its own llama-server, and Phase 1
# measured that a seed does not reach across server instances, so mixing games through one server
# would leave the arms comparable within a game but not the wall-clock accounting between them.
set -u

REPO=/home/ianblenke/github.com/ianblenke/carnot
PY=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
H="$REPO/results/arc_engine_validation_20260731/harness"
LOGS="${P2_LOGS:-/tmp/arc_p2_logs}"
mkdir -p "$LOGS"

ATTEMPTS="${AB_ATTEMPTS:-3}"

run_one() {  # game gpu port
  local game=$1 gpu=$2 port=$3
  AB_GAME="$game" AB_GPU="$gpu" AB_PORT="$port" AB_ATTEMPTS="$ATTEMPTS" \
    timeout 7200 "$PY" "$H/repair_ab.py" > "$LOGS/$game.log" 2>&1
  echo "$game exit=$?"
}

wave() {  # gameA gameB
  run_one "$1" 0 8941 &
  local p1=$!
  if [ -n "${2:-}" ]; then run_one "$2" 1 8942 & local p2=$!; fi
  wait $p1
  if [ -n "${2:-}" ]; then wait $p2; fi
}

wave ft09 tu93
wave lp85 tn36
wave sc25 ""

"$PY" "$H/build_ab_artifact.py"
