#!/bin/bash
# Conductor commit watchdog — surfaces stuck commits for autonomous recovery.
#
# Why this exists:
#   The conductor's commit pipeline truncates / fails periodically, leaving
#   uncommitted dirty files in the working tree. The 2026-04-27 → 2026-04-28
#   sessions saw this 4× — each instance required a manual `git commit` to
#   unblock. Without an automated watchdog, the conductor can run for hours
#   accumulating uncommitted state that vanishes on a hard reset.
#
# What this does:
#   1. Probes the working tree for files dirty more than $STALE_MIN minutes.
#   2. If any files match: produce a summary on stderr + write an alert
#      JSONL entry to ops/conductor-watchdog/alerts.jsonl.
#   3. If $AUTO_COMMIT is set to 1: attempt a `git commit --no-verify` of the
#      dirty files with a clearly-tagged message. This is the last-resort
#      bypass; the alerts.jsonl entry records that the bypass fired.
#   4. Exit 0 always (informational) unless --strict is passed (exit 1 on
#      stale files).
#
# Usage:
#   bash scripts/conductor_commit_watchdog.sh
#   AUTO_COMMIT=1 bash scripts/conductor_commit_watchdog.sh    # last-resort bypass
#   STALE_MIN=120 bash scripts/conductor_commit_watchdog.sh
#
# Run periodically: cron (every 30 min) or systemd-timer.

set -u

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ -z "$REPO_ROOT" ]; then
    echo "watchdog: not in a git repo" >&2
    exit 2
fi
cd "$REPO_ROOT" || exit 2

STALE_MIN="${STALE_MIN:-60}"
AUTO_COMMIT="${AUTO_COMMIT:-0}"
ALERT_DIR="$REPO_ROOT/ops/conductor-watchdog"
ALERT_FILE="$ALERT_DIR/alerts.jsonl"
STRICT=0
for arg in "$@"; do
    case "$arg" in
        --strict) STRICT=1 ;;
    esac
done

mkdir -p "$ALERT_DIR"

# Find dirty files (tracked + modified, plus untracked matching repo gitignore rules)
DIRTY_FILES=$(git status --porcelain | awk '{print $2}')
if [ -z "$DIRTY_FILES" ]; then
    exit 0
fi

# Filter to files modified more than STALE_MIN minutes ago
STALE_FILES=()
NOW_EPOCH=$(date +%s)
for f in $DIRTY_FILES; do
    [ -e "$f" ] || continue
    MTIME=$(stat -c '%Y' "$f" 2>/dev/null || echo "$NOW_EPOCH")
    AGE_MIN=$(( (NOW_EPOCH - MTIME) / 60 ))
    if [ "$AGE_MIN" -gt "$STALE_MIN" ]; then
        STALE_FILES+=("$f")
    fi
done

if [ "${#STALE_FILES[@]}" -eq 0 ]; then
    exit 0
fi

# Emit alert
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
N_STALE="${#STALE_FILES[@]}"
echo "watchdog: $N_STALE files dirty > ${STALE_MIN}min" >&2
printf '  %s\n' "${STALE_FILES[@]}" >&2

# Append to alerts.jsonl
ALERT_JSON=$(printf '{"ts":"%s","stale_minutes":%d,"file_count":%d,"files":[' \
    "$TS" "$STALE_MIN" "$N_STALE")
for i in "${!STALE_FILES[@]}"; do
    if [ "$i" -gt 0 ]; then ALERT_JSON+=","; fi
    ALERT_JSON+="\"${STALE_FILES[$i]}\""
done
ALERT_JSON+="],\"auto_commit_triggered\":$([ "$AUTO_COMMIT" = "1" ] && echo true || echo false)}"
echo "$ALERT_JSON" >> "$ALERT_FILE"
sync 2>/dev/null || true

# Last-resort auto-commit if requested
if [ "$AUTO_COMMIT" = "1" ]; then
    git add "${STALE_FILES[@]}" 2>&1 | tail -2 >&2
    if git commit --no-verify -m "[conductor-watchdog] Force-commit ${N_STALE} stale files (>${STALE_MIN}min)

Triggered by scripts/conductor_commit_watchdog.sh as last-resort bypass.
Files were dirty in the working tree without being committed by the
conductor's normal commit pipeline. See ops/conductor-watchdog/alerts.jsonl
for the alert log.

[no-verify because pre-commit hooks may be the failing step]" 2>&1 | tail -3 >&2; then
        echo "watchdog: auto-commit succeeded" >&2
    else
        echo "watchdog: auto-commit FAILED — escalate to human" >&2
        exit 2
    fi
fi

if [ "$STRICT" = "1" ]; then
    exit 1
fi
exit 0
