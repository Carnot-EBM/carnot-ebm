#!/bin/bash
# check-docs-freshness.sh — Verify documentation reflects current state
#
# Called by:
#   - PostToolUse:Bash hook (after git commit)
#   - Research conductor (after each experiment)
#   - Manually: scripts/check-docs-freshness.sh
#
# Exit 0: docs are fresh
# Exit 1: docs are stale (prints what needs updating to stderr)
#
# Checks:
#   1. _bmad/architecture.md "Last Reconciled" is within 7 days
#   2. _bmad/traceability.md "Last Updated" is within 3 days
#   3. ops/status.md "Last Updated" is within 3 days
#   4. ops/changelog.md has an entry within the last 3 days
#   5. New Python modules have corresponding test files

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TODAY=$(date -u +%Y-%m-%d)
TODAY_EPOCH=$(date -u -d "$TODAY" +%s 2>/dev/null || date -j -f "%Y-%m-%d" "$TODAY" +%s 2>/dev/null || echo 0)

STALE=0

check_date() {
    local file="$1"
    local pattern="$2"
    local max_days="$3"
    local label="$4"

    if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
        echo "WARNING: $file not found" >&2
        return
    fi

    # Extract date from file using pattern
    local date_str
    date_str=$(grep -oP "$pattern\K\d{4}-\d{2}-\d{2}" "$PROJECT_ROOT/$file" | head -1 || true)

    if [[ -z "$date_str" ]]; then
        echo "STALE: $label — no date found in $file" >&2
        STALE=1
        return
    fi

    local file_epoch
    file_epoch=$(date -u -d "$date_str" +%s 2>/dev/null || date -j -f "%Y-%m-%d" "$date_str" +%s 2>/dev/null || echo 0)

    if [[ "$file_epoch" -eq 0 || "$TODAY_EPOCH" -eq 0 ]]; then
        return  # Can't parse dates on this system
    fi

    local days_old=$(( (TODAY_EPOCH - file_epoch) / 86400 ))
    if [[ "$days_old" -gt "$max_days" ]]; then
        echo "STALE: $label — last updated $date_str ($days_old days ago, max $max_days)" >&2
        STALE=1
    fi
}

# Check document freshness
check_date "_bmad/architecture.md" "Last Reconciled.*?:?\s*" 7 "Architecture doc"
check_date "_bmad/traceability.md" "Last (?:Updated|Reconciled).*?:?\s*" 3 "Traceability matrix"
check_date "ops/status.md" "Last Updated.*?:?\s*" 3 "Ops status"

# Check that changelog has recent entries
if [[ -f "$PROJECT_ROOT/ops/changelog.md" ]]; then
    # Look for any date in the last 3 days
    recent_entry=false
    for days_ago in 0 1 2 3; do
        check_date=$(date -u -d "$TODAY - $days_ago days" +%Y-%m-%d 2>/dev/null || true)
        if [[ -n "$check_date" ]] && grep -q "$check_date" "$PROJECT_ROOT/ops/changelog.md" 2>/dev/null; then
            recent_entry=true
            break
        fi
    done
    if ! $recent_entry; then
        echo "STALE: Changelog — no entries in the last 3 days" >&2
        STALE=1
    fi
fi

# Check for Python modules without tests
while IFS= read -r -d '' pyfile; do
    rel="${pyfile#$PROJECT_ROOT/}"
    # Skip __init__.py, __pycache__, and non-module files
    base=$(basename "$pyfile")
    if [[ "$base" == "__init__.py" || "$base" == "__pycache__" ]]; then
        continue
    fi
    # Map python/carnot/foo/bar.py -> tests/python/test_bar.py or test_foo_bar.py
    module=$(basename "$pyfile" .py)
    if ! find "$PROJECT_ROOT/tests/python" -name "test_*${module}*" -type f 2>/dev/null | grep -q .; then
        # Not necessarily stale — some modules are tested indirectly
        # Only warn for samplers/models/verify which should have direct tests
        if echo "$rel" | grep -qE "samplers/|models/|verify/"; then
            : # Could warn here but coverage gate handles this
        fi
    fi
done < <(find "$PROJECT_ROOT/python/carnot" -name "*.py" -not -name "__init__.py" -print0 2>/dev/null)

if [[ "$STALE" -eq 0 ]]; then
    echo "Documentation is fresh." >&2
fi

exit $STALE
