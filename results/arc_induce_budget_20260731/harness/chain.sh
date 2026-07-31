#!/bin/bash
# Wait for the two budget lanes, then run the two follow-on lanes on the freed cards.
#
# WHY CHAINED AND NOT CONCURRENT WITH THE BUDGET LANES. At the pinned n_ctx=32768 a second
# concurrent request turns K=1 into K=2, which drops the per-slot allowance to 16384 cells --
# below `ft09 prompt (4343) + 16384 budget = 20727`. The 16384-budget rows would then be
# silently truncated by the shared pool (mode C) and would read as "the model stopped early"
# when in fact the harness starved it. Running these after is the only way the 16384 tier means
# what it says.
set -u
P5=/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/p5
PY=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
cd "$P5" || exit 1

while ! grep -q "DONE" lane_engine.log || ! grep -q "DONE" lane_combined.log; do
  sleep 30
done
echo "budget lanes done at $(date -u +%H:%M:%SZ)"

# GPU 1 / port 8933: the REFACTOR call -- the one that actually truncated in the live ft09 run,
# and the one REQ-ARC-FCP-5699-34 measured 8192 fixing on a 27B.
SWEEP_OUT=sweep_refactor SWEEP_GPU=1 SWEEP_PORT=8933 \
  SWEEP_BUDGETS=4096,8192,16384 SWEEP_ATTEMPTS=3 \
  nohup "$PY" refactor_sweep.py > lane_refactor.log 2>&1 &

# GPU 0 / port 8934: the SAMPLER control -- is the wall a budget problem or a decode-degeneration
# problem? Held at ONE budget (4096) so the only axis that varies is the repetition control.
SWEEP_OUT=sweep_sampler SWEEP_GPU=0 SWEEP_PORT=8934 SWEEP_PROMPT=engine \
  SWEEP_BUDGETS=4096 SWEEP_ATTEMPTS=3 \
  nohup "$PY" sampler_sweep.py > lane_sampler.log 2>&1 &

wait
echo "follow-on lanes done at $(date -u +%H:%M:%SZ)"
