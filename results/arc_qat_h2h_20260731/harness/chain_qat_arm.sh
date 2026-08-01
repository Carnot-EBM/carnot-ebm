#!/usr/bin/env bash
# Launch the QAT arm the moment the control arm finishes -- but ONLY if the control arm
# actually produced a usable shard. Chaining unconditionally would burn another 5 hours
# measuring a treatment arm whose control had wedged, and the pair is worthless without both.
set -u
REPO=/home/ianblenke/github.com/ianblenke/carnot
D="$REPO/results/arc_qat_h2h_20260731"
cd "$REPO"

# Wait for the control arm process to exit (it is already running).
while pgrep -f "h2h_arm_runner.py --arm q4km" >/dev/null; do sleep 60; done
echo "[chain] control arm exited at $(date -u +%H:%M:%SZ)"

ROWS=$(wc -l < "$D/h2h_shard_q4km.jsonl" 2>/dev/null || echo 0)
echo "[chain] control shard rows: $ROWS / 39"
if [ "$ROWS" -lt 39 ]; then
  echo "[chain] REFUSING to start the QAT arm: control arm is incomplete."
  echo "[chain] A wedge is a recorded fact -- inspect h2h_meta_q4km.json before rerunning."
  exit 2
fi

echo "[chain] starting QAT arm at $(date -u +%H:%M:%SZ)"
CARNOT_ARC_E3_DIR=/tmp/qat_h2h/e3_qat JAX_PLATFORMS=cpu \
  "$REPO/.venv/bin/python" "$D/harness/h2h_arm_runner.py" \
    --arm qat --shard "$D/h2h_shard_qat.jsonl" --meta "$D/h2h_meta_qat.json"
echo "[chain] QAT arm exited rc=$? at $(date -u +%H:%M:%SZ)"
