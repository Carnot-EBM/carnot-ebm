# Carnot — Operational Status

**Last Updated:** 2026-04-03

## What's Working

- BMAD strategic documents created (PRD, architecture, traceability)
- OpenSpec capability specs for core-ebm, model-tiers, training-inference
- Rust workspace compiles: carnot-core, carnot-ising, carnot-gibbs, carnot-boltzmann, carnot-samplers, carnot-training
- Python/JAX package structure with core abstractions, Ising model, samplers
- Pre-commit hooks configured (rustfmt, clippy, ruff, mypy, pytest coverage, spec coverage)
- PyO3 binding crate compiles (with PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 for Python 3.14)

## What's Next

- Run `cargo test` and fix any test failures
- Set up Python test suite with 100% coverage
- Install pre-commit hooks and verify they pass
- Implement Gibbs and Boltzmann models in Python/JAX
- Add score matching and NCE training algorithms
- Set up CI/CD (GitHub Actions)
- Design autoresearch loop (JAX prototype -> Rust transpilation pipeline)

## Known Constraints

- Python 3.14 requires PyO3 ABI3 forward compatibility flag
- PyO3 bindings are skeleton only — no model classes exposed yet
