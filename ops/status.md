# Carnot — Operational Status

**Last Updated:** 2026-04-03 — ALL REQUIREMENTS IMPLEMENTED + LLM AUTORESEARCH WORKING

## What's Working — Everything

### Core Framework (REQ-CORE-001–006)
- EnergyFunction trait (Rust) and protocol (Python/JAX)
- Three model tiers: Ising (both), Gibbs (Rust + partial Python), Boltzmann (both)
- Samplers: Langevin + HMC in both languages
- Serialization: safetensors cross-language persistence
- PyO3 bindings: all 3 tiers + 2 samplers exposed to Python
- Configurable precision: f32 (default) / f64

### Training (REQ-TRAIN-001–004)
- Contrastive Divergence CD-k (Rust)
- Denoising Score Matching (Rust + Python/JAX)
- Noise Contrastive Estimation (Rust + Python/JAX)
- Adam optimizer with gradient clipping (Rust)

### Verifiable Reasoning (REQ-VERIFY-001–007)
- ConstraintTerm trait/protocol — constraints as energy terms
- ComposedEnergy — weighted composition with decomposition
- Verification certificates — VERIFIED/VIOLATED with per-constraint reports
- Gradient-based repair — descend only on violated constraints
- Energy landscape certification — Hessian eigenvalue analysis, basin estimation
- Deterministic reproducibility
- Sudoku example — full constraint satisfaction demo

### Autoresearch Pipeline (REQ-AUTO-001–010)
- Benchmark suite: DoubleWell, Rosenbrock, Ackley, Rastrigin, GaussianMixture
- Benchmark runner with baseline recording (JSON)
- Process-level sandbox (dev): import blocking, timeout, I/O capture
- Docker+gVisor sandbox (production): 5-layer defense in depth
- Three-gate evaluator: energy, time, memory
- Experiment log: append-only audit trail with rejected registry
- Orchestrator: full propose → sandbox → evaluate → log → update loop
- Circuit breaker: halts after N consecutive failures
- Cross-language validation: test vector generation + conformance checking
- Automatic rollback: git-based revert on production energy regression
- End-to-end demo: `python scripts/demo_autoresearch.py`

### Gibbs Python/JAX (NEW)
- Full GibbsModel with SiLU/ReLU/Tanh activations in JAX
- AutoGradMixin for automatic gradient computation
- Completes cross-language parity for all 3 model tiers

### PyO3 Integration Tests (NEW)
- 24 tests validating Rust↔Python bridge for all tiers + samplers
- Fixed module name mismatch (carnot_python → _rust)

### Claude Code API Bridge (NEW)
- `tools/claude-api-bridge/` — Docker container wrapping `claude -p` as OpenAI API
- Streaming SSE + non-streaming JSON + MCP config support
- Uses Max subscription OAuth credentials via volume mount

### LLM-Powered Autoresearch (NEW)
- Hypothesis generator using OpenAI-compatible API (Claude, Qwen, etc.)
- Generator-based orchestrator loop with failure feedback
- Successfully ran 3 iterations with Sonnet generating real Carnot sampler code

### Quality Infrastructure
- 408 tests (100 Rust + 284 Python + 24 PyO3 integration), 100% code coverage
- Pre-commit hooks: rustfmt, clippy, ruff, mypy, pytest, spec coverage
- 8-agent team: security-auditor, test-runner, lint-checker, spec-validator, spec-reconciler, evaluator, docs-keeper, **adversarial-reviewer** (red team)
- Gitea CI: 5 parallel jobs
- 7-agent team: security-auditor, test-runner, lint-checker, spec-validator, spec-reconciler, evaluator, docs-keeper
- SOPS encrypted secrets
- Comprehensive inline documentation (~6,000 lines of docs)

## What's Next

All 39 requirements implemented. Gibbs JAX, PyO3 tests, and LLM autoresearch now working. Remaining:

- ~~**Gibbs Python/JAX**~~: DONE — full implementation with AutoGradMixin
- ~~**PyO3 integration tests**~~: DONE — 24 tests, all tiers + samplers
- ~~**Real autoresearch run**~~: DONE — Claude API bridge + LLM hypothesis generator
- ~~**Benchmark energy functions**~~: DONE — All 5 benchmarks (DoubleWell, Rosenbrock, Ackley, Rastrigin, GaussianMixture) implemented as JAX EnergyFunction classes with known minima. Autoresearch now evaluates against real mathematical landscapes.
- **Autoresearch tuning**: JIT compilation overhead makes first-call timing unfair; consider warm-up or more generous time budgets
- **GPU benchmarking**: JAX CUDA performance vs Rust CPU
- **Attention in Boltzmann tier**: multi-head attention (num_heads is reserved)
- **GitHub public mirror**: visibility for the open-source project

## Known Constraints
- Python 3.14 requires `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`
- gVisor not yet installed on dev machine (Docker sandbox falls back to runc)
- Ackley and GaussianMixture benchmarks use numerical gradients
