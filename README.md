# Carnot

**Open-source Energy Based Model framework — Rust + Python/JAX**

Carnot is an EBM framework built to move beyond the hallucination-prone paradigm of autoregressive large language models, and toward autonomous systems capable of directed self-learning.

## The Problem with LLMs

Large language models generate text by predicting the most probable next token. This produces fluent output, but it's fundamentally guessing — there is no mechanism to verify that the output is logically consistent, physically valid, or factually correct. When an early token is wrong, the error cascades irrecoverably. This is why LLMs hallucinate: they optimize for plausibility, not truth.

## Why Energy Based Models?

EBMs take a fundamentally different approach. Instead of generating outputs sequentially, they assign a scalar energy to every possible configuration of variables. Low energy = valid/consistent; high energy = invalid/contradictory. Inference is optimization: find the configuration that minimizes energy across all constraints simultaneously.

This enables capabilities that autoregressive models structurally cannot provide:

- **Verifiable reasoning** — mathematically prove a solution satisfies all constraints by showing it sits at an energy minimum
- **Surgical error correction** — when a constraint is violated, gradient descent fixes the broken part without discarding the rest
- **Autonomous self-improvement** — the energy function is an objective ground truth that cannot be gamed, enabling closed-loop self-learning without human feedback
- **Hardware acceleration** — energy landscapes map directly to thermodynamic sampling hardware (Extropic TSU), promising 10,000x efficiency gains

## The Path to Self-Learning

Carnot is designed from the ground up to support an autonomous self-improvement loop:

1. **Propose** — candidate improvements to architecture, training, or hyperparameters are prototyped in Python/JAX
2. **Evaluate** — the energy landscape on held-out data serves as the objective judge (did energy decrease? real improvement. did it not? rejected.)
3. **Deploy** — proven improvements are transpiled to Rust for production performance
4. **Repeat** — the loop runs without human supervision, with safety guardrails

The EBM itself is the evaluator. No LLM needed to judge quality — the math provides ground truth.

## Model Tiers

| Tier | Name | Scale | Use Case |
|------|------|-------|----------|
| Large | **Boltzmann** | Deep residual + attention | Research frontiers, large-scale generation |
| Medium | **Gibbs** | Multi-layer (2-4 hidden) | Applied ML, domain adaptation |
| Small | **Ising** | Pairwise interactions | Edge deployment, teaching |

All tiers implement the same `EnergyFunction` trait (Rust) / protocol (Python), so algorithms written against the interface work with any tier.

## Architecture

```
carnot/
├── crates/                        # Rust workspace
│   ├── carnot-core/               # EnergyFunction trait, types, serialization
│   ├── carnot-ising/              # Ising tier: E(x) = -0.5 x^T J x - b^T x
│   ├── carnot-gibbs/              # Gibbs tier: multi-layer energy network
│   ├── carnot-boltzmann/          # Boltzmann tier: deep residual energy network
│   ├── carnot-samplers/           # Langevin dynamics + HMC samplers
│   ├── carnot-training/           # CD-k, score matching, optimizers
│   └── carnot-python/             # PyO3 bindings
├── python/carnot/                 # Python/JAX package
│   ├── core/                      # Energy function protocol, model state
│   ├── models/                    # Ising, Gibbs, Boltzmann in JAX
│   ├── samplers/                  # Langevin, HMC via jax.lax.scan
│   └── training/                  # JAX training loops with Optax
├── openspec/capabilities/         # Specification-driven contracts
│   ├── core-ebm/                  # REQ-CORE-*, SCENARIO-CORE-*
│   ├── model-tiers/               # REQ-TIER-*, SCENARIO-TIER-*
│   └── training-inference/        # REQ-TRAIN-*, REQ-SAMPLE-*
├── _bmad/                         # Strategic docs (PRD, architecture, traceability)
└── ops/                           # Operational status, changelog, test results
```

## Quick Start

### Rust

```bash
cargo build --workspace --exclude carnot-python
cargo test --workspace --exclude carnot-python
```

### Python

```bash
pip install -e ".[dev]"
pytest tests/python
```

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Development Philosophy

Carnot follows **spec-anchored development**:

1. **Spec first** — every feature starts as REQ-* and SCENARIO-* in OpenSpec
2. **Tests trace to specs** — every test references the requirement it verifies
3. **100% coverage** — code coverage and spec coverage enforced by pre-commit hooks
4. **Dual implementation** — Rust for performance, Python/JAX for research iteration
5. **Cross-language interop** — safetensors serialization + PyO3 bindings

See [CLAUDE.md](CLAUDE.md) for the full development workflow.

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Core compute (Rust) | ndarray, rayon |
| Core compute (Python) | JAX, Flax, Optax |
| Python-Rust bridge | PyO3, maturin |
| Serialization | safetensors |
| Testing | cargo test, pytest, cargo-tarpaulin |
| Linting | rustfmt, clippy, ruff, mypy (strict) |

## Related Work

Carnot exists alongside other open-source EBM frameworks, each with different focuses:

| Framework | Org | Language | Focus |
|-----------|-----|----------|-------|
| [EB-JEPA](https://github.com/facebookresearch/eb_jepa) | Meta FAIR | PyTorch | Self-supervised world modeling (JEPA) |
| [THRML](https://github.com/extropic-ai/thrml) | Extropic | JAX | Probabilistic graphical models for TSU hardware |
| [TorchEBM](https://github.com/soran-ghaderi/torchebm) | Independent | PyTorch | General-purpose EBM toolkit |
| **Carnot** | Open source | Rust + JAX | Spec-driven, dual-language, production + research |

## License

Apache 2.0 — see [LICENSE](LICENSE).
