#!/usr/bin/env bash
# validate-task-completion.sh — repository task-completion validation hook
#
# Validates that the project is in a healthy state after changes:
#   1. Python tests pass with 100% coverage
#   2. Rust tests pass
#   3. A handoff artifact exists in .harness/handoffs/
#
# Exit 0  — all checks pass (or tools not available)
# Exit 2  — one or more checks failed; prints specific feedback to stderr
#
# Design: tolerant of missing infrastructure. If pytest or cargo are not
# installed, the corresponding check is skipped (exit 0). This avoids
# blocking work in environments where only one language toolchain is present.

set -uo pipefail

# --------------------------------------------------------------------------
# Autoresearch bypass: when running in research mode, skip validation.
# The research conductor manages its own test/commit/revert cycle.
# --------------------------------------------------------------------------
if [[ "${CARNOT_MODE:-}" == "research" ]]; then
  exit 0
fi

# --------------------------------------------------------------------------
# Find the repo root so paths resolve correctly regardless of cwd
# --------------------------------------------------------------------------
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "")"
if [[ -z "$REPO_ROOT" ]]; then
  # Not in a git repo; can't validate reliably, so allow
  exit 0
fi

cd "$REPO_ROOT"

FAILURES=()

# --------------------------------------------------------------------------
# 1. Python tests with 100% coverage
# --------------------------------------------------------------------------
# Run pytest only if the pytest command exists AND the test directory is present.
# The --cov-fail-under=100 flag causes pytest to return non-zero if coverage
# drops below 100%, which is the project's standard.

if command -v pytest &>/dev/null && [[ -d "tests/python" ]]; then
  if ! pytest tests/python \
       --cov=python/carnot \
       --cov-report=term-missing \
       --cov-fail-under=100 \
       -q 2>/dev/null; then
    FAILURES+=("Python tests failed or coverage is below 100%. Run: pytest tests/python --cov=python/carnot --cov-report=term-missing --cov-fail-under=100")
  fi
else
  # pytest not available or test dir missing — skip gracefully
  :
fi

# --------------------------------------------------------------------------
# 2. Rust workspace tests
# --------------------------------------------------------------------------
# Run cargo test only if cargo is available AND a Cargo.toml exists at root.
# The carnot-python crate is excluded because it requires a Python
# interpreter linked via PyO3, which may not be set up in all environments.

if command -v cargo &>/dev/null && [[ -f "Cargo.toml" ]]; then
  if ! cargo test --workspace --exclude carnot-python -q 2>/dev/null; then
    FAILURES+=("Rust tests failed. Run: cargo test --workspace --exclude carnot-python")
  fi
else
  # cargo not available or no Cargo.toml — skip gracefully
  :
fi

# --------------------------------------------------------------------------
# 3. Handoff artifact check
# --------------------------------------------------------------------------
# The .harness/handoffs/ directory should contain at least one file when
# a task is complete. This ensures the session leaves a handoff document
# for the next session (human or AI), as required by the workflow.

HANDOFF_DIR=".harness/handoffs"

if [[ -d "$HANDOFF_DIR" ]]; then
  # Count non-hidden files in the handoffs directory
  FILE_COUNT=$(find "$HANDOFF_DIR" -maxdepth 1 -type f ! -name '.*' 2>/dev/null | wc -l)
  if [[ "$FILE_COUNT" -eq 0 ]]; then
    FAILURES+=("No handoff artifact found in ${HANDOFF_DIR}/. Create a handoff document summarizing what was done and what's next.")
  fi
else
  FAILURES+=("Handoff directory ${HANDOFF_DIR}/ does not exist. Create it and add a handoff document.")
fi

# --------------------------------------------------------------------------
# Report results
# --------------------------------------------------------------------------

if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo "TASK COMPLETION VALIDATION FAILED" >&2
  echo "==================================" >&2
  echo "" >&2
  for i in "${!FAILURES[@]}"; do
    echo "$((i + 1)). ${FAILURES[$i]}" >&2
    echo "" >&2
  done
  echo "Fix the above issues before marking the task as complete." >&2
  exit 2
fi

# All checks passed
exit 0
