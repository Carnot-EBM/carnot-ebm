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
# - Single codex/gemini subprocess >2hr elapsed without log activity
# - Load average >10 (orphan accumulation despite cgroup wrap)
# - New structural failure mode: Traceback / persistent FAIL across multiple
#   distinct tasks / conductor crash with no recovery
# - Task stuck 3+ retries bootstrap-only with no log activity (>3 hours)
# - Heartbeat last_beat >60 minutes stale (NEW 2026-05-13)
# - Current-milestone retire rate >60% with ≥10 tasks (NEW 2026-05-13)
# - Adversarial-verify flag rate >50% in last 10 experiments (NEW 2026-05-13)
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

  # NEW (2026-05-13): heartbeat staleness in minutes — proxy for "conductor advancing"
  HEARTBEAT_AGE_MIN="n/a"
  if [ -f ops/conductor-heartbeat.json ]; then
    LAST_BEAT=$(python3 -c "import json,sys; d=json.load(open('ops/conductor-heartbeat.json')); print(d.get('last_beat',''))" 2>/dev/null || echo "")
    if [ -n "$LAST_BEAT" ]; then
      HEARTBEAT_AGE_MIN=$(python3 -c "
from datetime import datetime, timezone
last = datetime.fromisoformat('$LAST_BEAT'.replace('Z','+00:00'))
now = datetime.now(timezone.utc)
print(int((now - last).total_seconds() / 60))
" 2>/dev/null || echo "n/a")
    fi
  fi

  # NEW (2026-05-13): current-milestone retire rate from log
  # Find the latest "Milestone YYYY.MM.NNN activated" line, then count OK vs
  # RETIRED outcomes since that line. Retired = DOOMED_RERUN_BLOCK, GATE_BLOCK
  # appearing 3+ times, FAIL 3+ times, or SKIP 3+ times for the same task.
  MILESTONE_STATS=$(python3 <<'PYEOF' 2>/dev/null || echo "milestone_stats: n/a"
import re
from collections import Counter

try:
    with open('ops/conductor-log.md') as f:
        lines = f.readlines()
except FileNotFoundError:
    print("milestone_stats: log_missing")
    raise SystemExit

# Walk backward to find latest "Milestone YYYY.MM.NNN activated"
activation_idx = None
milestone = None
for i in range(len(lines) - 1, -1, -1):
    m = re.search(r'Milestone (\d{4}\.\d{2}\.\d+) activated', lines[i])
    if m:
        activation_idx = i
        milestone = m.group(1)
        break

if activation_idx is None:
    print("milestone_stats: no_activation_in_log")
    raise SystemExit

# Count outcomes for tasks since activation
ok_count = 0
task_attempts = Counter()
for line in lines[activation_idx + 1:]:
    # Skip non-table lines
    if not line.startswith('|') or '|' not in line[1:]:
        continue
    parts = [p.strip() for p in line.split('|')]
    if len(parts) < 5:
        continue
    # parts: ['', date_utc, title, status, details, '']
    title = parts[2]
    status = parts[3]
    if not title.startswith('Exp '):
        continue
    if status == 'OK':
        ok_count += 1
    elif status in ('DOOMED_RERUN_BLOCK', 'GATE_BLOCK', 'FAIL', 'SKIP'):
        task_attempts[(title, status)] += 1

# Retired tasks: 3+ retries with same FAIL/GATE_BLOCK/SKIP status, OR any
# DOOMED_RERUN_BLOCK count (one DRB = retired)
retired = set()
for (title, status), cnt in task_attempts.items():
    if status == 'DOOMED_RERUN_BLOCK' and cnt >= 1:
        retired.add(title)
    elif cnt >= 3 and status in ('GATE_BLOCK', 'FAIL', 'SKIP'):
        retired.add(title)

retired_count = len(retired)
total_attempted = ok_count + retired_count
retire_rate = (100 * retired_count / total_attempted) if total_attempted else 0

print(f"milestone={milestone} ok={ok_count} retired={retired_count} retire_rate={retire_rate:.0f}%")
PYEOF
)

  # NEW (2026-05-13): adversarial-verify sweep on last 10 experiments
  # Compares against the baseline saved at results/adversarial_verify_sweep_20260512.json
  ADV_SWEEP="adv_sweep: n/a"
  if [ -f scripts/adversarial_verify.py ]; then
    # Find latest experiment number from log
    LATEST_EXP=$(python3 -c "
import re
try:
    with open('ops/conductor-log.md') as f:
        lines = f.readlines()
except FileNotFoundError:
    raise SystemExit
for line in reversed(lines):
    m = re.search(r'Exp (\d+):', line)
    if m:
        print(m.group(1)); raise SystemExit
print('')
" 2>/dev/null || echo "")
    if [ -n "$LATEST_EXP" ]; then
      LOW=$((LATEST_EXP - 10))
      ADV_SWEEP=$(timeout 60 python3 scripts/adversarial_verify.py --milestone-range "$LOW" "$LATEST_EXP" --json 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    n = len(d['reports'])
    f = d['flagged_count']
    rate = (100 * f / n) if n else 0
    print(f'adv_sweep: scanned={n} flagged={f} rate={rate:.0f}%')
except Exception:
    print('adv_sweep: parse_error')
" 2>/dev/null || echo "adv_sweep: timeout")
    fi
  fi

  echo "--- snapshot ---"
  echo "heartbeat: $HEARTBEAT"
  echo "heartbeat_age_min: $HEARTBEAT_AGE_MIN"
  echo "service: $SVC"
  echo "load: $LOAD"
  echo "log_tail:"
  echo "$LOG_TAIL"
  echo "unpushed: $UNPUSHED"
  echo "procs: $PROCS"
  echo "$MILESTONE_STATS"
  echo "$ADV_SWEEP"

  # Codex evaluation — single JSON-ish line on stdout
  PROMPT=$(cat <<EOF
You are the Carnot autonomous research conductor's hourly outer-loop monitor.
Read the status snapshot and decide one of:

ESCALATE if any of these anomalies are present:

  STRUCTURAL/PROCESS:
  - conductor service inactive (systemd should auto-restart; escalate only
    if it didn't)
  - a single codex/gemini subprocess elapsed >2 hours with no recent log
    activity
  - load average >10
  - unrecovered Traceback/Exception in recent log
  - persistent FAIL across multiple distinct tasks (not just one retry)
  - task retried 3+ times bootstrap-only across >3 hours

  LIVENESS (new 2026-05-13):
  - heartbeat_age_min > 60 (conductor not advancing; >120 is critical)

  MILESTONE HEALTH (new 2026-05-13):
  - milestone_stats retire_rate > 60% AND (ok + retired) >= 10
    (planner is emitting scope-similar repeats or harness-fit failures
    are dominating; needs operator-pre-staged roadmap or routing fix)

  RESEARCH INTEGRITY (new 2026-05-13):
  - adv_sweep rate > 50%
    (artifact fabrication / tautology / sample-size violations exceeding
    baseline ~30%; paper-v6 disclosure discipline urgent)

OK if conductor is healthy: heartbeat fresh (<60min), milestone retire
rate within historical band (~30-40%), adv_sweep rate near baseline
(~30%), agents producing terminal verdicts at expected cadence
(~10-30 min per task).

Output exactly ONE line, format:
ESCALATE: <one-sentence reason naming the specific anomaly + recommended action>
or
OK: heartbeat=Xmin retire=Y% adv=Z% iter=N — <one-line summary>

Recommended operator actions for the new anomaly classes:
- heartbeat_age_min > 60: systemctl --user restart carnot-conductor.service
- retire_rate > 60% with task IDs repeating: pre-stage research-roadmap-next.yaml
  with proper prior_failures: blocks for the repeated tasks
- adv_sweep rate > 50%: review the flagged artifacts at
  results/adversarial_verify_sweep_<latest>.json; common cause is
  agent fabrication (duration_s too short for claimed model invocation)

STATUS SNAPSHOT:
heartbeat: $HEARTBEAT
heartbeat_age_min: $HEARTBEAT_AGE_MIN
service: $SVC
load: $LOAD
recent_log:
$LOG_TAIL
unpushed_commits: $UNPUSHED
active_procs: $PROCS
$MILESTONE_STATS
$ADV_SWEEP

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
