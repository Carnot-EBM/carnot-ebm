#!/usr/bin/env bash
# run_experiment_with_timeout.sh — wrapper that enforces a hard timeout on
# the research conductor (or any other long-running command).
#
# REQ-INFRA-001: conductor timeout ≤ 45 min (default).
# RETRO-001 carried forward from milestones 2026.04.22 and 2026.04.23.
# Exp 308 post-test failure loop consumed 138 min; a 45-min cap saves 93 min.
#
# Usage:
#   CARNOT_CONDUCTOR_TIMEOUT_MINUTES=30 ./scripts/run_experiment_with_timeout.sh \
#       python scripts/research_conductor.py --push
#
# Exit codes:
#   0     — command completed successfully
#   1-123 — command's own exit code
#   124   — timeout fired (standard Unix timeout sentinel)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Read timeout from environment; default to 45 minutes (RETRO-001 target).
TIMEOUT_MINUTES="${CARNOT_CONDUCTOR_TIMEOUT_MINUTES:-45}"

# Grace period before SIGKILL is sent after SIGTERM (REQ-INFRA-001 -k flag).
KILL_AFTER="60s"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command...>" >&2
    echo "  CARNOT_CONDUCTOR_TIMEOUT_MINUTES defaults to 45" >&2
    exit 1
fi

# Run the command under timeout.
# -k KILL_AFTER  — send SIGKILL if the process is still alive after this grace
#                  period following the initial SIGTERM.
# timeout exits with 124 when the timer fires.
timeout -k "${KILL_AFTER}" "${TIMEOUT_MINUTES}m" "$@"
EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 124 ]]; then
    echo "CONDUCTOR TIMEOUT after ${TIMEOUT_MINUTES} min — process was killed." >&2
fi

exit ${EXIT_CODE}
