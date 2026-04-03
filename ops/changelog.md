# Carnot — Changelog

## 2026-04-03: Project Bootstrap (user instruction: initial project setup)

### Added
- **BMAD strategic documents**: `_bmad/prd.md`, `_bmad/architecture.md`, `_bmad/traceability.md`
- **OpenSpec capability specs**: core-ebm, model-tiers, training-inference (spec.md + design.md each)
- **Rust workspace** with 7 crates:
  - `carnot-core`: EnergyFunction trait, ModelState, serialization, initialization
  - `carnot-ising`: Ising (small) tier with pairwise interaction energy
  - `carnot-gibbs`: Gibbs (medium) tier with multi-layer energy network
  - `carnot-boltzmann`: Boltzmann (large) tier with residual blocks
  - `carnot-samplers`: Langevin dynamics + HMC samplers with Sampler trait
  - `carnot-training`: CD-k training + Adam optimizer
  - `carnot-python`: PyO3 binding skeleton
- **Python/JAX package**:
  - Core energy function protocol and AutoGradMixin
  - ModelState with safetensors serialization
  - IsingModel in JAX
  - Langevin and HMC samplers in JAX (using jax.lax.scan)
- **Pre-commit hooks**: rustfmt, clippy, ruff, mypy, pytest coverage, spec coverage checker
- **Ops documents**: status.md, changelog.md, known-issues.md, e2e-test-plan.md, test-results.md, metrics.md
- **Spec coverage script**: `scripts/check_spec_coverage.py`
- Context from deep research PDF on EBM ecosystem (EB-JEPA, THRML, TorchEBM, Kona, Extropic TSU)
