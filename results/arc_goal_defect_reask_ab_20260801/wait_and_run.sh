#!/usr/bin/env bash
# Wait for a 3090 to free, then run the whole pipeline: A/B -> score -> analyse -> artifact.
#
# NEVER EVICTS. A concurrent session owned both cards for this session; the rule is to wait,
# so this only ever binds a card that ALREADY has headroom and it kills nothing. `run_ab.py`
# re-checks the same headroom precondition itself and refuses on its own, so this loop is
# convenience and not the guard.
#
# Polls at 15s, not minutes: the other session cycles its servers, and a card was observed
# free and retaken inside two minutes. A coarse poll sees "always busy" and waits forever.
#
# CHAINED so that a window arriving late still yields a COMPLETE result rather than a
# directory of raw cells nobody turned into a finding. Each stage is bounded and each stage's
# failure is recorded rather than swallowed -- build_artifact runs unconditionally at the end
# precisely so a partial or failed run still writes an honest artifact.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
NEED_MB=${NEED_MB:-20500}
LOG="$HERE/out/run.log"
mkdir -p "$(dirname "$LOG")"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits |
    while IFS=', ' read -r idx total used; do
      free=$((total - used))
      if [ "$free" -ge "$NEED_MB" ]; then echo "$idx"; return 0; fi
    done
}

while true; do
  GPU=$(pick_gpu | head -1)
  if [ -n "${GPU:-}" ]; then
    echo "$(date -u +%FT%TZ) launching on GPU $GPU" | tee -a "$LOG"
    cd "$REPO" || exit 1
    GDAB_GPU="$GPU" CARNOT_ARC_OFFLINE=1 JAX_PLATFORMS=cpu \
      "$PY" "$HERE/run_ab.py" >>"$LOG" 2>&1
    echo "$(date -u +%FT%TZ) run_ab exited rc=$?" | tee -a "$LOG"

    if [ -s "$HERE/out/rows.json" ]; then
      echo "$(date -u +%FT%TZ) scoring" | tee -a "$LOG"
      CARNOT_ARC_OFFLINE=1 JAX_PLATFORMS=cpu "$PY" "$HERE/score_cells.py" >>"$LOG" 2>&1
      echo "$(date -u +%FT%TZ) score_cells exited rc=$?" | tee -a "$LOG"
      CARNOT_ARC_OFFLINE=1 JAX_PLATFORMS=cpu "$PY" "$HERE/analyse.py" >>"$LOG" 2>&1
      echo "$(date -u +%FT%TZ) analyse exited rc=$?" | tee -a "$LOG"
    else
      echo "$(date -u +%FT%TZ) no rows.json -- skipping score/analyse" | tee -a "$LOG"
    fi

    # Unconditional: a partial or failed run still deserves an honest artifact.
    "$PY" "$HERE/build_artifact.py" >>"$LOG" 2>&1
    echo "$(date -u +%FT%TZ) PIPELINE DONE rc=$?" | tee -a "$LOG"
    exit 0
  fi
  sleep 15
done
