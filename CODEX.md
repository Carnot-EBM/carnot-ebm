# CODEX.md

This file provides guidance to Codex when working with code in this repository.

## Codex Guidelines

- If the user's request is based on a misconception, say so.
- Never claim tests passed when command output shows failures.
- Keep progress updates concise and high-signal.
- Use parallel helpers only when they materially improve the result.

## Required Workflow

1. **Spec First** — Update `openspec/capabilities/*/spec.md` with new `REQ-*` and `SCENARIO-*` items before implementation work that changes behavior.
2. **Write Tests First** — Tests must reference the `REQ-*` or `SCENARIO-*` they verify.
3. **Implement** — Change code only after the spec and tests describe the intended behavior.
4. **Verify** — Run the relevant build, test, lint, type-check, and spec-coverage commands.
5. **E2E Verify** — Run the applicable checks from `ops/e2e-test-plan.md` before reporting completion.
6. **Reconcile** — Update implementation status in specs, `_bmad/traceability.md`, and any relevant story/docs files.
7. **Update Ops** — Record what changed in `ops/changelog.md` and the current state in `ops/status.md`.

## Validation Hooks

- `scripts/validate-phase-gate.sh` blocks implementation writes when the matching OpenSpec capability has no spec or no `REQ-*`.
- `scripts/validate-test-run.sh` warns after shell commands that need reconciliation.
- `scripts/validate-reconciliation.sh` is the final consistency check before reporting work done.

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

- `AGENTS.md` for the auto-loaded Codex entrypoint.
- `_bmad/architecture.md` before architectural changes.
- `_bmad/traceability.md` before closing out capability work.
- `ops/e2e-test-plan.md` before declaring task completion.
- `ops/known-issues.md` before deployment or deep debugging work.
- `CLAUDE.md` and `GEMINI.md` contain the same canonical workflow, written for their CLIs.
