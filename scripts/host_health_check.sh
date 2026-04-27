#!/usr/bin/env bash
# host_health_check.sh — quick health probe for the early symptoms of the
# 2026-04-26 swap-saturation incident.
#
# Reports four metrics that tracked the incident's onset:
#   swap_gb_used       — RAM pressure spilling to swap
#   orphan_experiments — count of experiment_NNN.py processes
#   orphan_pytests     — count of pytest tests/python invocations
#   zombie_count       — defunct child processes never reaped
#
# Exits 0 with status=OK if all four are below threshold, or 1 with
# status=ALERT and a short reason. Designed to run unattended via
# ScheduleWakeup or a cron and surface a single line that's easy to
# decide on.

# Note on shell options: we deliberately do NOT use `set -e` or pipefail.
# The pgrep calls below return non-zero when there are no matching
# processes (the OK case), and pipefail would mistakenly treat that as
# a script failure. We do `set -u` for unset-variable safety.
set -u

# Thresholds tuned against the 2026-04-26 incident peak:
#   - swap reached 123 GB / 143 GB total (86%)
#   - 3 concurrent experiment_942 + 7 concurrent pytest accumulated over hours
#   - 161 defunct multiprocessing zombies
# The thresholds below catch the situation while it is still building, before
# the host becomes unresponsive.
readonly SWAP_GB_ALERT=30
readonly ORPHAN_EXP_ALERT=2          # one in flight is normal; two is suspect
readonly ORPHAN_PYTEST_ALERT=2       # same shape
readonly ZOMBIE_ALERT=50             # incident hit 161; alert well below that

swap_gb=$(free -g | awk '/^Swap:/ {print $3}')
orphan_exp=$(pgrep -af 'python.*scripts/experiment_[0-9]+_' 2>/dev/null \
              | grep -v "host_health_check\|grep " | wc -l)
orphan_pytest=$(pgrep -af 'pytest.*tests/python' 2>/dev/null \
                 | grep -v "host_health_check\|grep " | wc -l)
zombie_count=$(ps -e -o stat= | awk '$1 ~ /^Z/' | wc -l)

reasons=()
if [ "$swap_gb" -gt "$SWAP_GB_ALERT" ]; then
    reasons+=("swap=${swap_gb}GB>${SWAP_GB_ALERT}")
fi
if [ "$orphan_exp" -gt "$ORPHAN_EXP_ALERT" ]; then
    reasons+=("orphan_experiments=${orphan_exp}>${ORPHAN_EXP_ALERT}")
fi
if [ "$orphan_pytest" -gt "$ORPHAN_PYTEST_ALERT" ]; then
    reasons+=("orphan_pytests=${orphan_pytest}>${ORPHAN_PYTEST_ALERT}")
fi
if [ "$zombie_count" -gt "$ZOMBIE_ALERT" ]; then
    reasons+=("zombies=${zombie_count}>${ZOMBIE_ALERT}")
fi

ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

if [ ${#reasons[@]} -gt 0 ]; then
    printf 'ALERT %s swap_gb=%d orphan_experiments=%d orphan_pytests=%d zombies=%d reason=%s\n' \
        "$ts" "$swap_gb" "$orphan_exp" "$orphan_pytest" "$zombie_count" \
        "$(IFS=,; echo "${reasons[*]}")"
    exit 1
fi

printf 'OK    %s swap_gb=%d orphan_experiments=%d orphan_pytests=%d zombies=%d\n' \
    "$ts" "$swap_gb" "$orphan_exp" "$orphan_pytest" "$zombie_count"
exit 0
