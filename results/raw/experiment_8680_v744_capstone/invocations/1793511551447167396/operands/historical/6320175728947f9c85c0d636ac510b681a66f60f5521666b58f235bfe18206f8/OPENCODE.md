# OPENCODE.md

This file provides guidance to OpenCode when working with code in this repository.

## OpenCode Guidelines

- If the user's request is based on a misconception, say so.
- Never claim tests passed when command output shows failures.
- Keep progress notes concise and focused on actions and outcomes.
- Treat the repository validation scripts as part of the workflow, even when OpenCode is not auto-wired to call them.

## Required Workflow

1. **Spec First** — Update `openspec/capabilities/*/spec.md` before changing implementation behavior.
2. **Tests First** — Add or update tests that trace directly to `REQ-*` and `SCENARIO-*`.
3. **Implement** — Make the code satisfy the documented spec and tests.
4. **Verify** — Run the relevant build, lint, type-check, and coverage commands.
5. **E2E Verify** — Run the applicable end-to-end checks from `ops/e2e-test-plan.md`.
6. **Reconcile** — Keep `openspec/`, `_bmad/traceability.md`, stories, and ops docs aligned with the code.
7. **Update Ops** — Record the change in `ops/changelog.md` and refresh `ops/status.md`.

## Validation Scripts

- Run `scripts/validate-phase-gate.sh <path>` before writing new implementation files if hook coverage is uncertain.
- Run `scripts/validate-test-run.sh` after significant shell-driven verification work.
- Run `scripts/validate-reconciliation.sh` before reporting completion or after commits.

## Core Commands

```bash
cargo build --workspace --exclude carnot-python
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p carnot-python
cargo test --workspace --exclude carnot-python
pytest tests/python --cov=python/carnot --cov-report=term-missing --cov-fail-under=100
python scripts/check_spec_coverage.py
cargo fmt --all -- --check
cargo clippy --workspace --exclude carnot-python -- -D warnings
ruff check python/ tests/
ruff format --check python/ tests/
mypy python/carnot
pre-commit run --all-files
```

## Read First

- `_bmad/architecture.md` before architectural changes.
- `_bmad/traceability.md` before closing out capability work.
- `ops/e2e-test-plan.md` before declaring task completion.
- `ops/known-issues.md` before deployment or deep debugging work.
- `CLAUDE.md` and `GEMINI.md` contain the same canonical workflow, written for their CLIs.
