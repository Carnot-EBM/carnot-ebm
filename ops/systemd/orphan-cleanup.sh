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
#
# Logs to /tmp/orphan-cleanup.log so accumulation is observable.

set -euo pipefail

HEARTBEAT=/home/ianblenke/github.com/ianblenke/carnot/ops/conductor-heartbeat.json
LOG=/tmp/orphan-cleanup.log
THRESHOLD_MIN=120  # 2 hours

# Get active conductor PID; bail if file missing or not parseable
if [ ! -f "$HEARTBEAT" ]; then
  echo "$(date -u +%FT%TZ) no heartbeat — skipping" >> "$LOG"
  exit 0
fi
CONDUCTOR_PID=$(python3 -c "import json,sys; print(json.load(open('$HEARTBEAT')).get('pid',''))" 2>/dev/null || echo "")
if [ -z "$CONDUCTOR_PID" ] || ! kill -0 "$CONDUCTOR_PID" 2>/dev/null; then
  echo "$(date -u +%FT%TZ) conductor PID $CONDUCTOR_PID not alive — skipping" >> "$LOG"
  exit 0
fi

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
