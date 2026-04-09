#!/bin/bash
# PostToolUse:Bash hook — validate after Bash commands.
#
# Triggers:
#   1. After git commit: run full reconciliation check (docs + specs + tests)
#   2. After pytest: coverage is enforced by pytest --cov-fail-under (no extra check needed)
#
# The hook receives tool input/output via environment variables:
#   TOOL_INPUT  — the bash command that was run
#   TOOL_OUTPUT — the command's stdout
#
# Exit 0: all good (or non-blocking warning)
# Exit 2: blocking error (prevents proceeding)

set -euo pipefail

COMMAND="${TOOL_INPUT:-}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Skip if running in research mode (conductor handles its own checks)
if [[ "${CARNOT_MODE:-}" == "research" ]]; then
    exit 0
fi

# After git commit: run reconciliation check (non-blocking warning)
if echo "$COMMAND" | grep -qE "^git commit"; then
    if [[ -x "$PROJECT_ROOT/scripts/validate-reconciliation.sh" ]]; then
        if ! "$PROJECT_ROOT/scripts/validate-reconciliation.sh" 2>&1; then
            echo "" >&2
            echo "═══════════════════════════════════════════════════════" >&2
            echo "RECONCILIATION NEEDED (per CLAUDE.md mandatory workflow)" >&2
            echo "═══════════════════════════════════════════════════════" >&2
            echo "Before reporting work as done:" >&2
            echo "  1. Update ops/status.md and ops/changelog.md" >&2
            echo "  2. Update _bmad/traceability.md if specs changed" >&2
            echo "  3. Ensure all tests reference REQ-*/SCENARIO-*" >&2
            echo "  4. Run: scripts/validate-reconciliation.sh" >&2
            echo "═══════════════════════════════════════════════════════" >&2
            # Non-blocking — exit 0 so the commit isn't prevented.
            # The warning is visible to the agent and should trigger action.
        fi
    fi
fi

exit 0
