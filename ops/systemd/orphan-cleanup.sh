#!/bin/bash
# orphan-cleanup.sh — kill stale pytest/python workers not under the active
# conductor. Runs as a cron-style janitor every 30 min.
#
# Origin: 2026-05-05 10:12 UTC incident — load average reached 90 with
# accumulated orphan pytest workers from prior conductor instances. Process
# tree showed multiple python3 procs at PPID=1 with 2-day elapsed time +
# zombie-parent pytest-xdist pools.
#
# This script is the cheap mechanical janitor that catches whatever escapes
# the systemd cgroup wrapping (~/.config/systemd/user/carnot-conductor.service).
# Together they form Layer 1 + Layer 2 defense in depth.
#
# Behavior:
#   1. Read active conductor PID from ops/conductor-heartbeat.json
#   2. Enumerate all python3 / pytest processes with elapsed >2hr
#   3. For each, walk up the parent tree; if conductor is NOT an ancestor,
#      kill the process
#   4. Conductor and its descendants are never touched
#   5. (2026-08-23) If the conductor is DEAD and no hold marker exists,
#      START it (REQ-CONDUCTOR-RESTART-1). The hold marker is the
#      machine-readable form of "stopped on purpose".
#
# Logs to /tmp/orphan-cleanup.log so accumulation is observable.
#
# Env overrides exist for TESTS ONLY (tests/python/test_janitor_conductor_
# restart.py runs this script against a tmp heartbeat + stub systemctl):
#   CARNOT_JANITOR_HEARTBEAT, CARNOT_JANITOR_LOG, CARNOT_JANITOR_HOLD,
#   CARNOT_JANITOR_SYSTEMCTL, CARNOT_JANITOR_CONDUCTOR_LOG,
#   CARNOT_JANITOR_SKIP_SWEEPS=1 (skip the maintenance blocks so a test
#   never invokes the real sentinel/authority/sweeps against the live box).

set -euo pipefail

HEARTBEAT=${CARNOT_JANITOR_HEARTBEAT:-/home/ianblenke/github.com/ianblenke/carnot/ops/conductor-heartbeat.json}
LOG=${CARNOT_JANITOR_LOG:-/tmp/orphan-cleanup.log}
HOLD=${CARNOT_JANITOR_HOLD:-/home/ianblenke/.carnot/conductor-hold}
SYSTEMCTL=${CARNOT_JANITOR_SYSTEMCTL:-systemctl}
CONDUCTOR_LOG=${CARNOT_JANITOR_CONDUCTOR_LOG:-/home/ianblenke/github.com/ianblenke/carnot/ops/conductor-log.md}
THRESHOLD_MIN=120  # 2 hours

if [ -z "${CARNOT_JANITOR_SKIP_SWEEPS:-}" ]; then

# --- /tmp inode-cruft reaping (added 2026-06-03 after the tmpfs inode-exhaustion
# incident) — runs FIRST, independent of conductor liveness ---
# Root cause: the conductor runs pytest on every task, so /tmp/pytest-of-$USER
# (plus carnot-trace-* dirs and stray coverage JSON) accumulate on the /tmp
# tmpfs, which has a HARD 1,048,576-inode cap. Over a few days these reached
# ~788k inodes (75% of the cap) and exhausted INODES (not space) — which makes
# EVERY /tmp write fail with ENOSPC, silently breaking the conductor's own
# checkpoint writes AND agent subprocesses (one EBT training run blocked at 0
# steps and the kill-gate recorded a FALSE NEGATIVE). This MUST run even when the
# conductor PID is stale/dead (cruft accumulates regardless), so it sits above
# the conductor-liveness early-exits below. Anything older than 30 min is from a
# completed run (tasks finish in minutes), so this never races an in-flight
# pytest. All lines are `|| true` so a no-match never trips set -e.
find "/tmp/pytest-of-$USER" -mindepth 1 -maxdepth 1 -type d -mmin +30 -exec rm -rf {} + 2>/dev/null || true
find /tmp -maxdepth 1 -mmin +30 \( -name 'carnot-trace-*' -o -name '*cov*.json' \) -exec rm -rf {} + 2>/dev/null || true
# Safety net: if /tmp inodes are still under pressure (>85% used), reap ALL
# pytest/trace cruft regardless of age — an exhausted tmpfs is a hard outage.
TMP_IUSE=$(df -i /tmp 2>/dev/null | awk 'NR==2{gsub("%","",$5); print $5}' || echo 0)
if [ -n "${TMP_IUSE:-}" ] && [ "${TMP_IUSE:-0}" -gt 85 ]; then
  rm -rf "/tmp/pytest-of-$USER" /tmp/carnot-trace-* 2>/dev/null || true
  echo "$(date -u +%FT%TZ) INODE PRESSURE ${TMP_IUSE}% -> full pytest/trace reap; now $(df -i /tmp 2>/dev/null | awk 'NR==2{print $5}')" >> "$LOG"
fi

# --- Anomaly-Escalation (added 2026-06-03, Deep Think P3) ---
# Scan recent result artifacts for FRAME-VIOLATING anomalies (infra false-negatives
# where the method didn't actually run, fabrication flags, invariant regressions) and
# append them to ops/anomaly-escalations.md for HUMAN triage — instead of the loop
# silently auto-reconciling them as dead-ends (the Verification Trap that buried the
# Thesis-A false-negative this session). De-duped; `|| true` so it never trips set -e.
_CARNOT_VENV=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
[ -x "$_CARNOT_VENV" ] && "$_CARNOT_VENV" \
  /home/ianblenke/github.com/ianblenke/carnot/scripts/anomaly_escalation.py \
  --scan --since-hours 1 --apply >> /tmp/anomaly-escalation.log 2>&1 || true

# --- In-flight run sentinel (added 2026-08-22, REQ-CONDUCTOR-SENTINEL-1/2/3) ---
# Reads the validity signals live runs already write: llm_on_row_valid row
# streaks, llama-server stderr allocation failures, stranded VRAM, orphaned
# servers, wrong loaded model. Escalates to ops/conductor-log.md (+
# known-issues.md for CRITICAL), deduplicated via ops/.run_sentinel_state.json.
# Runs ABOVE the conductor-liveness early-exit on purpose: outer-loop-launched
# runs are exactly the class the conductor cannot see, and the 2026-08-22
# invalid-rows A/B (2.5h of llm_on_row_valid=false nobody read) was one.
# Read-only except those ops appends; it NEVER kills a process. `|| true` so a
# failure never trips set -e; the state-file receipt makes a dead sentinel
# visible to the conductor's own receipt check.
[ -x "$_CARNOT_VENV" ] && timeout 180 "$_CARNOT_VENV" \
  /home/ianblenke/github.com/ianblenke/carnot/scripts/conductor_run_sentinel.py \
  >> /tmp/run-sentinel.log 2>&1 || true

# --- Stop authority (added 2026-08-23, REQ-CONDUCTOR-AUTHORITY-1/2) ---
# The ACTION half of the sentinel's findings: reaps a provably-unowned
# llama-server (six conjunctive conditions + two-scan persistence), and —
# ONLY when the operator has armed ~/.carnot/stop-authority-armed — stops a
# provably-dead-tier run (all-invalid rows AND server failure evidence).
# Disarmed, it writes a yes/no packet instead of acting. Every kill writes a
# durable actor line: this project's own reapers never add to the
# unexplained-dead-process class (2026-08-09). Runs above the liveness
# early-exit for the same reason the sentinel does.
[ -x "$_CARNOT_VENV" ] && timeout 120 "$_CARNOT_VENV" \
  /home/ianblenke/github.com/ianblenke/carnot/scripts/run_stop_authority.py \
  >> /tmp/stop-authority.log 2>&1 || true

# --- Root-clutter sweep (added 2026-06-09) — relocate untracked agent/experiment
# scratch out of the REPO ROOT. The conductor launches every subagent with
# cwd=PROJECT_ROOT, so quick probe.py / check_*.py / test_*.py debug scripts accrete
# at the repo root (~155 had piled up before the 2026-06-09 cleanup). Conservative:
# 120-min age guard (never sweeps in-flight scratch), allowlist of legit root files,
# NEVER touches tracked files; untracked root *.py -> .root-scratch-trash/ (reversible),
# regenerable build artifacts (main.*, vivado*, clockInfo.txt) -> deleted. Runs
# regardless of conductor liveness (clutter accrues regardless), above the heartbeat
# early-exits. `|| true` so a failure never trips set -e. Uses stdlib only -> python3. ---
python3 /home/ianblenke/github.com/ianblenke/carnot/scripts/root_clutter_sweep.py --apply \
  >> /tmp/root-clutter-sweep.log 2>&1 || true

fi  # CARNOT_JANITOR_SKIP_SWEEPS

# --- Conductor liveness (rewritten 2026-08-23, REQ-CONDUCTOR-RESTART-1) ---
# Old behavior: heartbeat missing or PID dead -> log "skipping" and exit 0.
# On 2026-08-22 that meant NO conductor ran for 4h39m+ while this janitor
# logged "not alive — skipping" every 31 minutes; the deliberate-stop intent
# lived only in ops/status.md prose. Now: a dead conductor is STARTED unless
# the hold marker exists. The hold marker is the machine-readable "stopped
# on purpose"; create it (one line of intent inside) before a deliberate
# stop, remove it to resume.
_dead_reason=""
CONDUCTOR_PID=""
if [ ! -f "$HEARTBEAT" ]; then
  _dead_reason="no heartbeat file"
else
  CONDUCTOR_PID=$(python3 -c "import json,sys; print(json.load(open('$HEARTBEAT')).get('pid',''))" 2>/dev/null || echo "")
  if [ -z "$CONDUCTOR_PID" ] || ! kill -0 "$CONDUCTOR_PID" 2>/dev/null; then
    _dead_reason="conductor PID ${CONDUCTOR_PID:-unparseable} not alive"
  fi
fi

if [ -n "$_dead_reason" ]; then
  if [ -f "$HOLD" ]; then
    echo "$(date -u +%FT%TZ) $_dead_reason; hold marker present — respecting operator stop" >> "$LOG"
    # A forgotten hold is a silent outage: WARN durably at 48h, deduped by day
    # (one row per calendar day, not one per 30-min cycle).
    if [ -n "$(find "$HOLD" -mmin +2880 2>/dev/null)" ]; then
      _today=$(date -u '+%Y-%m-%d')
      if ! grep -q "JANITOR: conductor hold stale | WARN | $_today" "$CONDUCTOR_LOG" 2>/dev/null; then
        printf '| %s | %s | WARN | %s |\n' \
          "$(date -u '+%Y-%m-%d %H:%M UTC')" \
          "JANITOR: conductor hold stale" \
          "$_today hold marker older than 48h; conductor stays down until it is removed" \
          >> "$CONDUCTOR_LOG" 2>/dev/null || true
      fi
    fi
  else
    # Fail direction: if systemd state is unreadable, do NOTHING — a start
    # during an operator intervention is exactly what the hold protocol
    # prevents, and absence of information is not permission.
    UNIT_STATE=$("$SYSTEMCTL" --user is-active carnot-conductor.service 2>/dev/null || true)
    if [ "$UNIT_STATE" = "active" ] || [ "$UNIT_STATE" = "activating" ]; then
      echo "$(date -u +%FT%TZ) $_dead_reason but unit is $UNIT_STATE — systemd owns it; skipping" >> "$LOG"
    elif [ -z "$UNIT_STATE" ]; then
      echo "$(date -u +%FT%TZ) $_dead_reason; systemd state unreadable — doing nothing" >> "$LOG"
    else
      if "$SYSTEMCTL" --user start carnot-conductor.service 2>>"$LOG"; then
        echo "$(date -u +%FT%TZ) $_dead_reason; no hold marker -> started carnot-conductor.service" >> "$LOG"
        printf '| %s | %s | WARN | %s |\n' \
          "$(date -u '+%Y-%m-%d %H:%M UTC')" \
          "JANITOR: conductor auto-start" \
          "$_dead_reason; unit was $UNIT_STATE; started (REQ-CONDUCTOR-RESTART-1)" \
          >> "$CONDUCTOR_LOG" 2>/dev/null || true
      else
        echo "$(date -u +%FT%TZ) $_dead_reason; start FAILED — see systemctl --user status" >> "$LOG"
      fi
    fi
  fi
  exit 0
fi

# Test hook: liveness-only tests must never run the reap half against
# the real process table (a stray kill -9 from a test is unrecoverable).
[ -n "${CARNOT_JANITOR_SKIP_REAP:-}" ] && exit 0

# Enumerate descendants (recursive) of conductor
declare -A under_conductor
under_conductor[$CONDUCTOR_PID]=1
# BFS via /proc — repeat until stable
changed=1
while [ "$changed" = "1" ]; do
  changed=0
  while read -r pid ppid; do
    [ -z "$pid" ] && continue
    if [ -n "${under_conductor[$ppid]:-}" ] && [ -z "${under_conductor[$pid]:-}" ]; then
      under_conductor[$pid]=1
      changed=1
    fi
  done < <(ps -eo pid,ppid --no-headers 2>/dev/null)
done

# Kill stale python3 / pytest not under conductor
killed=0
while read -r pid etime comm; do
  [ -z "$pid" ] && continue
  case "$comm" in
    python3|pytest) ;;
    *) continue ;;
  esac
  # parse elapsed: HH:MM:SS, DD-HH:MM:SS, MM:SS
  case "$etime" in
    *-*) mins=999999 ;;  # days
    *:*:*)
      h=${etime%%:*}
      mins=$((10#$h * 60))
      ;;
    *:*) mins=0 ;;  # MM:SS, under threshold
    *) mins=0 ;;
  esac
  [ "$mins" -lt "$THRESHOLD_MIN" ] && continue
  # Skip if under conductor
  [ -n "${under_conductor[$pid]:-}" ] && continue
  if kill -9 "$pid" 2>/dev/null; then
    killed=$((killed+1))
  fi
done < <(ps -eo pid,etime,comm --no-headers 2>/dev/null)

if [ "$killed" -gt 0 ]; then
  echo "$(date -u +%FT%TZ) killed $killed orphan workers (conductor=$CONDUCTOR_PID)" >> "$LOG"
fi
