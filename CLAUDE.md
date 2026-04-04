# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Anthropic internal prompt augmentation
If you notice the user's request is based on a misconception, say so.
Never claim 'all tests pass' when output shows failures."
Keep text between tool calls to <=25 words.
Spawn an adversarial sub-agent to review non-trivial changes before reporting completion.

## Development Workflow (MANDATORY)

This project uses **spec-anchored development** (BMAD + OpenSpec). Every code change follows:

1. **Spec First** — Update `openspec/capabilities/*/spec.md` with new REQ-* and SCENARIO-*. Create/update story in `epics/stories/`.
2. **Write Tests** — Tests reference REQ-* and SCENARIO-* in comments.
3. **Implement** — Code to satisfy spec requirements.
4. **Verify** — Run unit tests, type checks, builds per commands below.
5. **E2E Verify (MANDATORY)** — Run end-to-end tests per `ops/e2e-test-plan.md`. All changes derived from user instruction MUST be verified E2E before reporting done. See E2E Testing below.
6. **Reconcile Specs** — Update Implementation Status in spec.md. Update story status. Update `_bmad/traceability.md` impl status column. If implementation diverged from spec, update spec to match reality with rationale.
7. **Update Ops** — Update `ops/status.md` (what's working/next) and `ops/changelog.md` (what you did).
8. **Update `_bmad`** — Update any part of `_bmad` that is relevant to the changes you made. Never leave specs and code disagreeing silently.

### Architecture Freshness Check

If `_bmad/architecture.md` "Last Reconciled" date is >30 days old, flag to user before starting new capability work.

## E2E Testing (MANDATORY)

**Every change derived from user instruction must be verified end-to-end.** This means:

- **EBM models**: Full training + sampling pipeline producing statistically correct distributions
- **Cross-language**: Rust and Python implementations producing equivalent results for same inputs
- **Serialization**: Model saved in one language loads correctly in the other

E2E tests must exercise the full stack, not just unit tests. The test plan lives at `ops/e2e-test-plan.md` and results are documented at `ops/test-results.md`.

## Build / Test / Deploy

```bash
# Build (Rust)
cargo build --workspace --exclude carnot-python
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p carnot-python

# Test (Rust unit)
cargo test --workspace --exclude carnot-python

# Test (Python unit with 100% coverage)
pytest tests/python --cov=python/carnot --cov-report=term-missing --cov-fail-under=100

# Test (spec coverage — every test must trace to REQ-*/SCENARIO-*)
python scripts/check_spec_coverage.py

# Lint/Type-check (Rust)
cargo fmt --all -- --check
cargo clippy --workspace --exclude carnot-python -- -D warnings

# Lint/Type-check (Python)
ruff check python/ tests/
ruff format --check python/ tests/
mypy python/carnot

# Pre-commit (all of the above)
pre-commit run --all-files

# Test (Rust with coverage via tarpaulin)
cargo tarpaulin --workspace --exclude carnot-python --out Html --fail-under 100
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Core compute (Rust) | Rust stable, ndarray, rayon |
| Core compute (Python) | Python 3.11+, JAX, Flax, Optax |
| Python-Rust bridge | PyO3 0.24+, maturin |
| Serialization | safetensors (both languages) |
| Rust testing | cargo test, cargo-tarpaulin |
| Python testing | pytest, pytest-cov |
| Rust linting | rustfmt, clippy |
| Python linting | ruff, mypy (strict) |
| Pre-commit | .pre-commit-config.yaml |

## Model Tiers

| Tier | Name | Crate | Python Module |
|------|------|-------|---------------|
| Large | Boltzmann | `carnot-boltzmann` | `carnot.models.boltzmann` |
| Medium | Gibbs | `carnot-gibbs` | `carnot.models.gibbs` |
| Small | Ising | `carnot-ising` | `carnot.models.ising` |

## Session Metrics (MANDATORY)

Track execution time and token consumption every turn:

1. **Turn start**: Run `date -u +"%Y-%m-%dT%H:%M:%SZ"` at start of each response
2. **Turn end**: Run `date -u +"%Y-%m-%dT%H:%M:%SZ"` right before responding to user
3. **Log both** in `ops/metrics.md` turn log table
4. **Subagent metrics**: Record tokens and duration from agent result metadata
5. **On context compaction or session end**: Run `python3 scripts/session-metrics.py` to extract authoritative token counts and costs from the session JSONL, then update `ops/metrics.md` Session Summary

## User Input Tracking (MANDATORY)

Every user instruction must be captured and traceable to outcomes:

1. **Log user instructions**: At the start of each turn, record a 1-line summary of the user's request in `ops/metrics.md` turn log (Description column)
2. **Cycle time**: The turn log's Start/End columns capture wall-clock time between user input and agent completion — this IS the cycle time. Review it to identify slow turns.
3. **Instruction → outcome mapping**: Each entry in `ops/changelog.md` should be traceable to the user instruction that triggered it. If a change was agent-initiated (refactoring, cleanup), note that explicitly.
4. **Session handoff**: Before session ends, update `ops/status.md` with what's working and what's next. This is the handoff document for the next session — human or AI.

## Build Environment

- Rust: stable toolchain
- Python: 3.11+ (3.14 requires `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`)
- JAX: CPU by default, CUDA 12 via `pip install carnot[cuda]`

## Key Paths

| What | Where |
|------|-------|
| BMAD strategic docs | `_bmad/` |
| Capability specs | `openspec/capabilities/*/spec.md` |
| Capability designs | `openspec/capabilities/*/design.md` |
| Change proposals | `openspec/change-proposals/` |
| Epics & stories | `epics/` |
| Operational status | `ops/status.md` |
| Work log | `ops/changelog.md` |
| Known issues | `ops/known-issues.md` |
| E2E test plan | `ops/e2e-test-plan.md` |
| Test results | `ops/test-results.md` |
| Session metrics | `ops/metrics.md` |
| Spec coverage script | `scripts/check_spec_coverage.py` |
| Rust crates | `crates/carnot-*/` |
| Python package | `python/carnot/` |

## When to Read Deeper

- **Before starting a new capability**: First review all documents in `_bmad/` and determine if the new capability is already implemented or if there are any relevant change proposals, or if the new capability implies an evolution of the architecture. Read the relevant `openspec/capabilities/*/spec.md` and `design.md`
- **Before deploying or debugging server issues**: Read `ops/known-issues.md`
- **Before architectural decisions or adding new components**: Read `_bmad/architecture.md`
- **To understand project scope or requirements**: Read `_bmad/prd.md`
- **To check what's built vs. spec'd**: Read `_bmad/traceability.md` (has implementation status per FR)
- **Before reporting work as done**: Read `ops/e2e-test-plan.md` and execute relevant E2E tests
