# Contributing to Carnot

Thank you for your interest in contributing to Carnot. This guide will help you get started.

## Development Environment Setup

### Prerequisites

- **Rust**: stable toolchain (install via [rustup](https://rustup.rs/))
- **Python**: 3.11 or later
- **pre-commit**: Install with `pip install pre-commit`, then run `pre-commit install` in the repo root

### Quick Start

```bash
git clone https://github.com/Carnot-EBM/carnot-ebm.git
cd carnot

# Rust
cargo build --workspace --exclude carnot-python

# Python
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Pre-commit hooks
pre-commit install
```

## Development Workflow

This project uses **spec-anchored development**. Every change should trace back to a specification requirement. See `CLAUDE.md` for the full workflow, but the short version is:

1. Update or create specs in `openspec/capabilities/` with REQ-* and SCENARIO-* identifiers.
2. Write tests that reference those identifiers.
3. Implement the change.
4. Verify with tests and lints (see below).
5. Reconcile specs and update operational docs.

## Running Tests

### Rust

```bash
# Unit tests
cargo test --workspace --exclude carnot-python

# Formatting check
cargo fmt --all -- --check

# Linting
cargo clippy --workspace --exclude carnot-python -- -D warnings
```

### Python

```bash
# Tests with 100% coverage requirement
PYTHONPATH=python:$PYTHONPATH pytest tests/python -v \
  --cov=python/carnot \
  --cov-report=term-missing \
  --cov-fail-under=100

# Lint
ruff check python/ tests/
ruff format --check python/ tests/

# Type checking
mypy python/carnot
```

### All Checks

```bash
pre-commit run --all-files
```

### Spec Coverage

Every test must trace to a REQ-* or SCENARIO-* identifier:

```bash
python3 scripts/check_spec_coverage.py
```

## Code Style

- **Rust**: Enforced by `rustfmt` and `clippy`. Run `cargo fmt` before committing.
- **Python**: Enforced by `ruff` (linting and formatting) and `mypy` (type checking).
- **Comments**: Use verbose, layman-friendly explanations in code comments -- not terse research shorthand.

## Pull Request Process

1. Create a feature branch from `main`.
2. Make your changes following the spec-anchored workflow above.
3. Ensure all tests pass and coverage requirements are met.
4. Ensure `pre-commit run --all-files` passes cleanly.
5. Update relevant specs, stories, and operational docs as described in `CLAUDE.md`.
6. Open a pull request against `main` with a clear description of what changed and why.
7. Link to the relevant REQ-* or SCENARIO-* identifiers in your PR description.

## Reporting Issues

Open a GitHub issue with a clear description of the problem, steps to reproduce, and expected vs. actual behavior. If relevant, include the model tier (Ising, Gibbs, or Boltzmann) and whether the issue is in Rust, Python, or both.

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 license.
