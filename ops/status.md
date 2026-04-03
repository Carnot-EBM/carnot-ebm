# Carnot — Operational Status

**Last Updated:** 2026-04-03 (reconciled)

## What's Working

### Core Framework
- EnergyFunction trait (Rust) and protocol (Python) — fully implemented
- Three model tiers: Ising (both languages), Gibbs (Rust + partial Python), Boltzmann (Rust only)
- Samplers: Langevin dynamics and HMC in both Rust and Python/JAX
- Serialization: safetensors cross-language model persistence
- PyO3 bindings: all 3 tiers + 2 samplers exposed to Python from Rust

### Training
- Contrastive Divergence CD-k (Rust)
- Denoising Score Matching (Rust + Python/JAX)
- Adam optimizer with gradient clipping (Rust)

### Verifiable Reasoning
- ConstraintTerm trait/protocol — constraints as energy terms
- ComposedEnergy — weighted constraint composition with decomposition
- Verification certificates — VERIFIED/VIOLATED with per-constraint reports
- Gradient-based repair — descend only on violated constraints
- Sudoku example — full constraint satisfaction demo

### Autoresearch Pipeline
- Benchmark suite: DoubleWell, Rosenbrock, Ackley, Rastrigin, GaussianMixture
- Benchmark runner with baseline recording (JSON persistence)
- Process-level sandbox (dev/test): import blocking, timeout, I/O capture
- Docker+gVisor sandbox (production): network isolation, read-only FS, memory/CPU limits
- Three-gate evaluator: energy, time, memory
- Experiment log: full audit trail with rejected registry
- Orchestrator: propose → sandbox → evaluate → log → update baselines
- Circuit breaker: halts after N consecutive failures
- End-to-end demo: `python scripts/demo_autoresearch.py`

### Quality Infrastructure
- 218 tests (93 Rust + 125 Python), 100% code coverage, 100% spec coverage
- Pre-commit hooks: rustfmt, clippy, ruff, mypy, pytest coverage, spec coverage
- Gitea CI: 5 parallel jobs (rust-check, rust-test, python-test, pyo3-check, spec-coverage)
- Agent team: security-auditor, test-runner, lint-checker, spec-validator, evaluator, docs-keeper
- SOPS configured for encrypted secrets
- Comprehensive inline documentation (4,475 lines)

## What's Next

### Unimplemented Requirements
- REQ-TRAIN-003: Noise Contrastive Estimation (NCE) — neither language
- REQ-VERIFY-006: Energy landscape certification (Hessian analysis, basin estimation)
- REQ-AUTO-006: Cross-language transpilation pipeline (JAX → Rust)
- REQ-AUTO-007: Automatic rollback mechanism
- REQ-AUTO-010: Improvement composition (testing multiple hypotheses together)

### Gaps
- Boltzmann tier missing Python/JAX implementation
- CD-k training missing Python implementation
- Training loop (REQ-TRAIN-004) only partial — missing LR scheduling, checkpointing
- PyO3 bindings not integration-tested end-to-end

## Known Constraints
- Python 3.14 requires `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`
- gVisor not yet installed on dev machine (Docker sandbox falls back to runc)
- Ackley and GaussianMixture benchmarks use numerical gradients (analytical is complex)
