# Carnot — Architecture

**Last Reconciled:** 2026-04-18

## Overview

Carnot is a dual-language (Rust + Python/JAX) Energy Based Model framework organized as a Cargo workspace with a companion Python package. The Rust side provides performance-critical compute; the Python side provides JAX-based research workflows and exposes Rust internals via PyO3.

## System Architecture

```
carnot/
├── crates/                    # Rust workspace
│   ├── carnot-core/           # Core energy function traits, types
│   ├── carnot-boltzmann/      # Large-tier EBM
│   ├── carnot-gibbs/          # Medium-tier EBM
│   ├── carnot-ising/          # Small-tier EBM
│   ├── carnot-samplers/       # MCMC samplers (Langevin, HMC)
│   ├── carnot-training/       # Training algorithms (CD, SM, NCE)
│   └── carnot-python/         # PyO3 bindings
├── python/
│   └── carnot/                # Python package
│       ├── core/              # JAX energy functions
│       ├── models/            # Boltzmann, Gibbs, Ising in JAX
│       ├── phase3/            # Phase 3 seed — bridging Ising to Kona continuous latent space
│       │   ├── continuous_ebm.py  # ContinuousEBM, sample_continuous, sample_langevin,
│       │   │                  #   sample_energy_matching, compare_samplers, compare_minima
│       │   │                  #   (Exp 435a baseline; Exp 446 Langevin+EnergyMatching)
│       │   └── __init__.py    # Exports all 8 public symbols
│       ├── samplers/          # JAX MCMC samplers
│       │   ├── parallel_ising.py  # Parallel Ising Gibbs (checkerboard, annealing, thrml-compatible)
│       │   └── backend.py     # SamplerBackend protocol (CPU, TSU stub)
│       ├── training/          # JAX training loops
│       ├── pipeline/          # Production verify-repair pipeline (Exp 74-75)
│       │   ├── extract.py     # ConstraintExtractor: Arithmetic, Code, Logic, NL, Auto
│       │   ├── verify_repair.py  # VerifyRepairPipeline — main user API
│       │   └── errors.py      # CarnotError hierarchy, timeouts, degradation
│       ├── mcp/               # Production MCP server (Exp 76)
│       │   └── server.py      # verify_llm_output, verify_and_repair, health_check
│       ├── verify/            # ComposedEnergy, ConstraintTerm, repair
│       ├── inference/         # EBM loader, composite scorer, LLM solver
│       └── bindings/          # PyO3 bridge to Rust
├── crates/carnot-constraints/ # Rust constraint verification (Exp 70)
├── examples/                  # 5 integration examples (Exp 79)
├── tests/
│   ├── rust/                  # Rust integration tests
│   ├── python/                # Python/pytest tests (1353 tests, 100% coverage)
│   └── integration/           # Full pipeline integration tests (Exp 81)
├── openspec/                  # Capability specs
├── _bmad/                     # Strategic docs
├── ops/                       # Operational docs
├── epics/                     # Epics and stories
├── research-program.md        # Declarative research goals and priorities
├── research-references.md     # Technologies and ideas for future milestones
├── research-roadmap.yaml      # Active research roadmap
└── research-complete.yaml     # Completed experiments (85+ across 4 milestones)
```

## Key Design Decisions

### DD-01: Cargo Workspace
Each logical component is a separate crate for compile-time isolation, independent versioning, and clear dependency boundaries.

### DD-02: Trait-Based Core
`carnot-core` defines traits (`EnergyFunction`, `Sampler`, `Trainer`) that all tiers implement. This enables generic algorithms over any tier.

### DD-03: JAX for Python
JAX is chosen over PyTorch for the Python side because JAX is the first-class citizen of EBM research — its functional transform model (vmap, grad, jit) maps naturally to energy function composition.

### DD-04: PyO3 Bindings
A dedicated `carnot-python` crate exposes Rust implementations to Python via PyO3/maturin, enabling researchers to use Rust performance from familiar Python workflows.

### DD-05: Tier Separation
Each tier (Boltzmann, Gibbs, Ising) is a separate crate/module to enable independent development, testing, and deployment. Users can depend on only the tier they need.

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Core compute (Rust) | Rust stable, ndarray, rayon |
| Core compute (Python) | Python 3.11+, JAX, Flax/NNX |
| Bindings | PyO3, maturin |
| Testing (Rust) | cargo test, cargo-tarpaulin |
| Testing (Python) | pytest, pytest-cov |
| Linting (Rust) | rustfmt, clippy |
| Linting (Python) | ruff, mypy |
| Pre-commit | pre-commit framework |
| CI | GitHub Actions |

## Data Flow

```
Training:
  Data → Sampler(energy_fn) → Gradient Estimator → Parameter Update → Model

Inference:
  Model + Noise → MCMC Sampler(energy_fn) → Samples
```

## Cross-Cutting Concerns

- **Logging**: `tracing` (Rust), `logging` (Python)
- **Serialization**: `serde` (Rust), `safetensors` (both)
- **Numerics**: `f32` default, `f64` configurable
- **Parallelism**: `rayon` (Rust), `jax.pmap` (Python)

## thrml Integration

The `parallel_ising.py` sampler provides a `parallel_sample_states` function that wraps thrml's `IsingEBM` interface. It extracts coupling matrices and biases from an `IsingEBM` instance and runs the parallel checkerboard Gibbs sampler (with optional simulated annealing) as a drop-in replacement for thrml's built-in sampling — achieving 183x speedup on CPU at 100 variables and 572x at 500 variables.

## Verification Pipeline Tiers

The LLM output verification pipeline uses a cascade architecture where cheaper tiers run first:

| Tier | Name | Class | Cost | Signal Source | Skip Condition |
|------|------|-------|------|---------------|----------------|
| 0a | CarnotThinkProbe | `CarnotThinkProbe` | ~0 ms (CI stub) / ~50-200 ms (GPU) | Generative 3-step CoT verdict (ThinkPRM, arXiv 2504.16828) | `verdict == 'incorrect'` → skip all downstream (fast-path violation) |
| 0b | SpilledEnergyDetector | `SpilledEnergyDetector` | ~0 ms (text hash) | Per-token logit-discrepancy (arXiv 2602.18671) | `high_spill_fraction <= threshold` (confident model) |
| 1 | SinkProbe | `SinkProbe` | ~0 ms (attention reuse) | Attention sink concentration (arXiv 2604.10697) | `mean_sink_score >= sink_threshold` |
| 2 | EORM | `EORMModel` | ~10 ms | CoT energy reward model (55M params) | `energy < eorm_threshold` |
| 3 | Ising | `VerifyRepairPipeline` | ~0.006 ms/constraint | Full constraint verification | Always runs if tiers 0-2 pass |

Each tier returns early if it can clear the response, avoiding subsequent more expensive tiers. Tier 0a (CarnotThinkProbe) was added in Exp 444 (arXiv 2504.16828, ThinkPRM). Tier 0b was added in Exp 433 (arXiv 2602.18671, ICLR 2026). Tiers 1-3 were designed in Exps 346-348/360.

CarnotThinkProbe is optional in VerifyRepairPipeline.verify() — pass `think_probe=CarnotThinkProbe(llm_caller=caller)` to enable. When enabled: if verdict='incorrect', returns VerificationResult(verified=False, mode='THINK_PROBE_FAST_PATH', skipped=True) without running Ising. CI stub (llm_caller=None, the default) returns 'uncertain' and falls through to Ising.

## KAN Fast-Path Tier — KAEMEnergy (Exp 447)

`python/carnot/models/kaem_energy.py` adds a **KAEMEnergy** model that replaces the iterative Ising/Gibbs MCMC inner loop with exact inverse-transform sampling (arXiv 2506.14167, June 2025).

| Component | Class | Key property |
|-----------|-------|--------------|
| Per-variable splines | `UnivariateKAEMLayer` | Energy = sum_i e_i(x_i); variables are independent under Gibbs distribution |
| Marginal CDF | `marginal_cdf(var_idx, x)` | Numerical integration (trapezoidal, 256-point grid) |
| Exact sampling | `sample_exact(n_samples, rng_key)` | Inverse-transform via binary search on precomputed CDF table; O(log 256) per variable |
| Full model | `KAEMEnergy` | energy() / sample() / fit(); no MCMC required |
| Benchmark | `benchmark_kaem_vs_mcmc` | Wall-clock comparison vs ParallelIsingSampler; returns speedup_ratio |

**Why this matters:** The Kolmogorov-Arnold theorem decomposes the energy into univariate per-variable terms. Each marginal CDF can be computed and inverted in closed form — no burn-in, no autocorrelation. Target speedup: 10-100x for sub-100-variable problems. Hardware path: bisection is pure arithmetic, FPGA-native. Exported from `carnot.models.__init__`.

Spec: REQ-SAMPLE-015, REQ-SAMPLE-016, SCENARIO-SAMPLE-027/028/029.
