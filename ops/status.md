# Carnot — Operational Status

**Last Updated:** 2026-04-04 — LLM-EBM INFERENCE PIPELINE BUILT + ALL REQUIREMENTS IMPLEMENTED

## What's Working — Everything

### Core Framework (REQ-CORE-001–006)
- EnergyFunction trait (Rust) and protocol (Python/JAX)
- Three model tiers: Ising (both), Gibbs (both), Boltzmann (both)
- Samplers: Langevin + HMC in both languages, with optional gradient clipping (REQ-SAMPLE-004)
- Serialization: safetensors cross-language persistence
- PyO3 bindings: all 3 tiers + 2 samplers exposed to Python, integration tested
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

### Autoresearch Pipeline (REQ-AUTO-001–014)
- Benchmark suite: DoubleWell, Rosenbrock, Ackley, Rastrigin, GaussianMixture (Rust + Python/JAX)
- Benchmark runner with baseline recording (JSON)
- Process-level sandbox (dev): import blocking, timeout, I/O capture
- Docker+gVisor sandbox (production): 5-layer defense in depth
- Three-gate evaluator: energy, time (with JIT grace period), memory
- Experiment log: append-only audit trail with rejected registry
- Orchestrator: full propose → sandbox → evaluate → log → update loop
- Generator-based orchestrator: lazy LLM hypothesis generation with failure feedback
- Claude Code API bridge: Docker container wrapping `claude -p` as OpenAI API
- Circuit breaker: halts after N consecutive failures
- Cross-language validation: test vector generation + conformance checking
- Automatic rollback: git-based revert on production energy regression
- **Trace2Skill learning layer** (REQ-AUTO-011–014):
  - Trajectory analyst: parallel error/success sub-agents extract structured Lessons via LLM
  - Skill directory: persistent optimization playbook (SKILL.md + lessons.json + scripts/ + references/)
  - Hierarchical consolidation: tree-reduction merge deduplicates, resolves conflicts, filters low-confidence
  - Cross-tier transfer: Ising lessons available when generating for Gibbs/Boltzmann
  - `run_loop_with_skills()`: enhanced orchestrator loop integrating all of the above

### 10-Iteration Autoresearch Results (Sonnet, 2026-04-04)
- DoubleWell: **0.9483 → 0.1604 (83% energy reduction)** via 3 accepted hypotheses
- Sonnet proposed: larger step sizes, HMC for curved valleys, annealing schedules
- By iteration 9, identified per-benchmark sampler selection as the right approach
- Rising baseline (from accepted improvements) made later iterations harder — by design
- Rosenbrock remains NaN (gradient explosion from steep `100*(...)^2` — needs sampler API extension)

### LLM-EBM Inference Pipeline (REQ-INFER-001–005)
- SAT constraints: product relaxation, DIMACS parser, binary penalty
- Graph coloring constraints: pairwise repulsion, range penalty
- LLM output parsing: multiple format support (space-separated, named, T/F)
- Verify-and-repair pipeline: parse → verify → gradient repair → round → certify
- Benchmark harness: random SAT/coloring generation, aggregated statistics
- **This is the first concrete anti-hallucination pipeline**: LLM proposes, EBM verifies and repairs

### Quality Infrastructure
- 408 tests (100 Rust + 284 Python + 24 PyO3 integration), 100% code coverage, 100% spec coverage
- Pre-commit hooks: rustfmt, clippy, ruff, mypy, pytest, spec coverage
- 8-agent BMAD team: security-auditor, test-runner, lint-checker, spec-validator, spec-reconciler, evaluator, docs-keeper, adversarial-reviewer (red team)
- E2E tests: training+sampling pipeline, serialization round-trip, PyO3 binding, API bridge, autoresearch
- SOPS encrypted secrets
- Comprehensive inline documentation (~8,000 lines of docs)

## What's Next

### High Priority
- ~~**Gradient clipping**~~: DONE — `clip_norm` parameter on LangevinSampler and HMCSampler. Rosenbrock no longer diverges.
- **E2E-001: Rust training pipeline test**: Only remaining E2E test gap. Requires Rust-side integration test.

### Medium Priority
- **GPU benchmarking**: JAX CUDA performance vs Rust CPU
- **Attention in Boltzmann tier**: multi-head attention (num_heads is reserved)
- **Longer autoresearch runs**: With gradient clipping, run 50+ iterations to test convergence

### Low Priority
- **GitHub public mirror**: open-source visibility
- **Autoresearch CI**: automate the LLM autoresearch loop in CI (requires API bridge in pipeline)

## Known Constraints
- Python 3.14 requires `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`
- gVisor not yet installed on dev machine (Docker sandbox falls back to runc)
- Ackley Python/JAX uses epsilon=1e-10 in sqrt for gradient stability (documented in spec, differs from Rust numerical gradient approach)
- Rosenbrock diverges with Langevin step_size > ~0.001 due to steep gradients (needs gradient clipping)
