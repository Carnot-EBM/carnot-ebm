#!/bin/bash
# Follow-up queue, run AFTER the main chat queue exits (waits by PID, never by pattern --
# a pgrep pattern here would match this script's own wrapper, the self-match hazard).
#
# Why this exists: the QC f16 arm lost 2 of its 3 prompts when its llama-server was killed by
# an EXTERNAL signal mid-run ("Received second interrupt, terminating immediately" x3 in the
# server log, while our own teardown sends exactly one SIGTERM and only at arm end). The most
# likely source is a pattern-based kill from the other workflow sharing this box. The arm is
# re-run here rather than reported as a partial, because a 1-of-3 arm cannot support the
# matched f16-vs-q8_0 comparison it exists to provide.
cd /tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/fitgrid || exit 1
PY=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  while [ -d "/proc/$WAIT_PID" ]; do sleep 10; done
fi
echo "=== main queue done; follow-up starting ==="
echo "=== FOLLOWUP: QC_egpu_24576_f16_chat (re-run) ==="
timeout 5400 "$PY" phase2.py QC_egpu_24576_f16_chat 2>&1
echo "=== FOLLOWUP COMPLETE ==="
