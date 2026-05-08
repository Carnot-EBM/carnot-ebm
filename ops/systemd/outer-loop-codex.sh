#!/bin/bash
# outer-loop-codex.sh — hourly outer-loop monitor running on codex/gpt-5.5
#
# Hybrid pattern (Path 2 from 2026-05-08 conversation):
# - Codex runs hourly via systemd timer for routine status checks
# - On ESCALATE verdict, operator pings claude-code session for structural fix
# - Most weeks: zero escalations expected (8 conductor structural fixes already
#   shipped this session arc + systemd cgroup wrap + orphan janitor + classifier
#   patches cover most known failure modes)
#
# Escalation criteria (codex writes "ESCALATE: ..." in output):
# - Conductor service inactive AND systemd auto-restart didn't recover
# - Single codex subprocess >2hr elapsed without log activity
# - Load average >10 (orphan accumulation despite cgroup wrap)
# - New structural failure mode: Traceback / persistent FAIL across multiple
#   distinct tasks / conductor crash with no recovery
# - Task stuck 3+ retries bootstrap-only with no log activity (>3 hours)
#
# Non-escalation routine work codex handles autonomously:
# - Push commits >30min old
# - Schedule note that next .NN milestone planner failed once but retrying
# - One-off bootstrap-only fail that retries naturally
#
# Logs: /tmp/carnot-outer-loop.log (rolling, journalctl --user -u carnot-outer-loop)

# NB: do NOT use set -e — best-effort monitor; errors in any single check
# (git push when nothing to push, awk on empty input, codex transient stall)
# must not abort the whole run.
set -u

cd /home/ianblenke/github.com/ianblenke/carnot

# Source codex env (AGENT_TYPE_PLANNER, CODEX_FORCE_EXPERIMENTS, etc.)
if [ -f /home/ianblenke/.carnot/conductor_state.sh ]; then
  . /home/ianblenke/.carnot/conductor_state.sh
fi

LOG=/tmp/carnot-outer-loop.log
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

{
  echo "==== $TS outer-loop check ===="

  # Routine: push any commit >30 min old
  UNPUSHED_OLD=$(git log origin/main..HEAD --format="%H %cr" 2>/dev/null | awk '$NF~"hour|day|month|year" || $(NF-1)+0 > 30' | wc -l)
  if [ "$UNPUSHED_OLD" -gt 0 ]; then
    echo "Pushing $UNPUSHED_OLD old commit(s)..."
    git push 2>&1 | tail -3
  fi

  # Build status snapshot for codex
  HEARTBEAT=$(cat ops/conductor-heartbeat.json 2>/dev/null || echo "missing")
  SVC=$(systemctl --user is-active carnot-conductor.service 2>&1 || echo "inactive")
  LOG_TAIL=$(tail -5 ops/conductor-log.md 2>/dev/null | sed 's/|/_/g')
  GIT_S=$(git status -s 2>/dev/null | head -5)
  UNPUSHED=$(git log origin/main..HEAD --oneline 2>/dev/null | head -5)
  PROCS=$(ps -eo pid,etime,cmd 2>/dev/null | grep -E "codex exec --|pytest tests" | grep -v grep | head -3)
  LOAD=$(uptime)

  echo "--- snapshot ---"
  echo "heartbeat: $HEARTBEAT"
  echo "service: $SVC"
  echo "load: $LOAD"
  echo "log_tail:"
  echo "$LOG_TAIL"
  echo "unpushed: $UNPUSHED"
  echo "procs: $PROCS"

  # Codex evaluation — single JSON-ish line on stdout
  PROMPT=$(cat <<EOF
You are the Carnot autonomous research conductor's hourly outer-loop monitor.
Read the status snapshot and decide one of:

ESCALATE if any of: conductor service inactive (systemd should auto-restart;
escalate only if it didn't); a single codex subprocess elapsed >2 hours with
no recent log activity; load average >10; unrecovered Traceback/Exception in
recent log; persistent FAIL across multiple distinct tasks (not just one
retry); task retried 3+ times bootstrap-only across >3 hours.

OK if conductor is healthy and producing terminal verdicts (OK or honest
research-finding) at expected cadence (~10-30 min per task).

Output exactly ONE line, format:
ESCALATE: <one-sentence reason naming the specific anomaly>
or
OK: iter=<N> recent=<one-line summary of last 1-3 task outcomes>

STATUS SNAPSHOT:
heartbeat: $HEARTBEAT
service: $SVC
load: $LOAD
recent_log:
$LOG_TAIL
unpushed_commits: $UNPUSHED
active_procs: $PROCS

Respond with the single-line verdict. No preamble, no markdown.
EOF
  )

  echo "--- codex verdict ---"
  echo "$PROMPT" | timeout 300 codex exec \
    --dangerously-bypass-approvals-and-sandbox \
    --color never \
    --model gpt-5.5 \
    --cd /home/ianblenke/github.com/ianblenke/carnot \
    --ephemeral \
    - 2>&1 | tail -20

  echo ""
} >> $LOG 2>&1
