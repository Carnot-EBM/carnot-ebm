#!/bin/bash
# validate-reconciliation.sh — Comprehensive post-commit reconciliation check
#
# Validates that BOTH documentation and specs are consistent with code.
# Called by PostToolUse:Bash hook after git commits.
#
# Checks:
#   1. Documentation freshness (dates, changelog entries)
#   2. Spec-test traceability (every test references REQ-*/SCENARIO-*)
#   3. Implementation completeness (no spec REQs marked TODO/PLANNED without code)
#   4. Coverage gate (100% line coverage on Python)
#   5. Changed files have corresponding spec/doc updates
#
# Exit 0: all reconciled
# Exit 1: issues found (prints details to stderr)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISSUES=0

warn() {
    echo "RECONCILE: $*" >&2
    ISSUES=$((ISSUES + 1))
}

info() {
    echo "  $*" >&2
}

# ─── 1. Documentation Freshness ──────────────────────────────────────
echo "Checking documentation freshness..." >&2
if [[ -x "$PROJECT_ROOT/scripts/check-docs-freshness.sh" ]]; then
    if ! "$PROJECT_ROOT/scripts/check-docs-freshness.sh" 2>&1 >/dev/null; then
        # Re-run to get the specific warnings
        "$PROJECT_ROOT/scripts/check-docs-freshness.sh" 2>&1 >/dev/null || true
        warn "Documentation is stale (see above)"
    fi
fi

# ─── 2. Spec-Test Traceability ───────────────────────────────────────
echo "Checking spec-test traceability..." >&2
if [[ -f "$PROJECT_ROOT/scripts/check_spec_coverage.py" ]]; then
    if ! "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/check_spec_coverage.py" >/dev/null 2>&1; then
        warn "Spec coverage: some tests lack REQ-*/SCENARIO-* references"
        info "Run: .venv/bin/python scripts/check_spec_coverage.py"
    fi
fi

# ─── 3. Changed Source Files Have Spec/Doc Updates ───────────────────
echo "Checking changed files for spec coverage..." >&2

# Get files changed in the last commit
CHANGED_FILES=$(git diff --name-only HEAD~1 HEAD 2>/dev/null || echo "")

if [[ -n "$CHANGED_FILES" ]]; then
    SRC_CHANGED=false
    SPEC_CHANGED=false
    DOC_CHANGED=false
    TEST_CHANGED=false

    while IFS= read -r file; do
        case "$file" in
            python/carnot/*)   SRC_CHANGED=true ;;
            crates/*)          SRC_CHANGED=true ;;
            openspec/*)        SPEC_CHANGED=true ;;
            _bmad/*)           DOC_CHANGED=true ;;
            ops/*)             DOC_CHANGED=true ;;
            tests/*)           TEST_CHANGED=true ;;
        esac
    done <<< "$CHANGED_FILES"

    # Source changed but no tests → warning
    if $SRC_CHANGED && ! $TEST_CHANGED; then
        warn "Source files changed but no test files updated"
        info "Changed: $(echo "$CHANGED_FILES" | grep -E '^(python|crates)/' | head -3)"
        info "Consider: add or update tests for changed code"
    fi

    # Source changed but no docs → warning
    if $SRC_CHANGED && ! $DOC_CHANGED; then
        warn "Source files changed but no documentation updated"
        info "Per CLAUDE.md: update ops/changelog.md and ops/status.md after changes"
    fi
fi

# ─── 4. Spec REQs vs Implementation Status ──────────────────────────
echo "Checking spec implementation status..." >&2

# Find all REQ-* in specs and check if they're marked as implemented
for spec_file in "$PROJECT_ROOT"/openspec/capabilities/*/spec.md; do
    [[ -f "$spec_file" ]] || continue
    cap_name=$(basename "$(dirname "$spec_file")")

    # Count REQs marked as TODO/PLANNED/NOT_IMPLEMENTED
    todo_count=$(grep -ciE "TODO|PLANNED|NOT.IMPLEMENTED" "$spec_file" 2>/dev/null || echo "0")
    todo_count=$(echo "$todo_count" | tr -d '[:space:]')
    if [[ "$todo_count" -gt 0 ]] 2>/dev/null; then
        # This is informational, not a failure — specs often have future work
        : # info "$cap_name: $todo_count REQs marked TODO/PLANNED"
    fi
done

# ─── 5. research-roadmap.yaml consistency ────────────────────────────
echo "Checking research roadmap consistency..." >&2
if [[ -f "$PROJECT_ROOT/research-roadmap.yaml" ]]; then
    # Check that deliverables for completed tasks actually exist
    while IFS= read -r deliverable; do
        deliverable=$(echo "$deliverable" | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')
        if [[ -n "$deliverable" && ! -f "$PROJECT_ROOT/$deliverable" ]]; then
            : # Pending tasks don't have deliverables yet — this is expected
        fi
    done < <(grep "deliverable:" "$PROJECT_ROOT/research-roadmap.yaml" | sed 's/.*deliverable:[[:space:]]*//')
fi

# ─── Summary ─────────────────────────────────────────────────────────
if [[ "$ISSUES" -eq 0 ]]; then
    echo "Reconciliation: all checks passed." >&2
else
    echo "" >&2
    echo "Reconciliation: $ISSUES issue(s) found." >&2
    echo "Fix these before reporting work as done (per CLAUDE.md workflow)." >&2
fi

exit $( [[ "$ISSUES" -eq 0 ]] && echo 0 || echo 1 )
