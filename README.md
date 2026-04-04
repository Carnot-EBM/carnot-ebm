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

## Research References

Papers and resources that have informed Carnot's design and direction.

### Foundational EBM Theory

- [LeCun et al. (2006) — A Tutorial on Energy-Based Learning](https://web.stanford.edu/class/cs379c/archive/2012/suggested_reading_list/documents/LeCunetal06.pdf) — The foundational EBM tutorial establishing energy functions as a unifying framework for ML
- [Gutmann & Hyvarinen (2010) — Noise-Contrastive Estimation](https://proceedings.mlr.press/v9/gutmann10a.html) — NCE training for EBMs, used in Carnot's `nce_loss()`

### EBM Architecture and Scaling

- [Energy-Based Transformers are Scalable Learners and Thinkers (2025)](https://arxiv.org/abs/2507.02092) — EBTs: train transformers to assign energy to (input, prediction) pairs, infer via gradient descent. 35% faster scaling than Transformer++. Validates Carnot's verify-and-repair architecture at transformer scale.
- [Autoregressive Language Models are Secretly EBMs (2025)](https://arxiv.org/abs/2512.15605) — Explicit bijection between ARMs and EBMs via soft Bellman equation. Every LLM is already an EBM. Theoretical foundation for extracting energy signals directly from LLM logits.
- [Learning EBMs by Self-Normalising the Likelihood (2025)](https://arxiv.org/abs/2503.07021) — SNL: single learnable parameter for partition function. Lower bound of log-likelihood, concave for exponential families. Potential alternative to NCE for training learned verifiers.

### EBM + LLM Hallucination Detection

- [Semantic Energy: Detecting LLM Hallucination Beyond Entropy (2025)](https://arxiv.org/abs/2508.14496) — Energy = negative logit from penultimate layer. High energy = hallucination. 4-5% AUROC improvement over entropy methods. Directly applicable to Carnot's verification pipeline.
- [Spilled Energy in Large Language Models (2026)](https://arxiv.org/abs/2602.18671) — Energy-based analysis of LLM internals for hallucination detection.
- [Energy-Based Calibration for Implicit Chain-of-Thought (2025)](https://arxiv.org/abs/2511.07124) — EBM-CoT: refine latent reasoning toward low-energy regions. Gradient descent on reasoning trajectories.

### EBM for Physical Systems

- [Hybrid EBMs for Physical AI: Port-Hamiltonian Dynamics (2026)](https://arxiv.org/abs/2604.00277) — Separates visible (dynamical) from hidden (feedforward) layers. Absorbing invariant sets for stability. Validates Carnot's architecture of constraint evaluation (feedforward) + repair dynamics (gradient descent).
- [Cognitively Inspired Energy-Based World Models (2024)](https://arxiv.org/abs/2406.08862) — EBMs as cognitive world models.

### Agent Skill Learning

- [Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills (2026)](https://arxiv.org/abs/2603.25158) — Parallel analyst sub-agents extract lessons from execution traces, hierarchical consolidation merges them. Integrated into Carnot's autoresearch as the Trace2Skill learning layer.

### Open-Source EBM Frameworks

| Framework | Org | Language | Focus |
|-----------|-----|----------|-------|
| [EB-JEPA](https://github.com/facebookresearch/eb_jepa) | Meta FAIR | PyTorch | Self-supervised world modeling (JEPA) |
| [THRML](https://github.com/extropic-ai/thrml) | Extropic | JAX | Probabilistic graphical models for TSU hardware |
| [TorchEBM](https://github.com/soran-ghaderi/torchebm) | Independent | PyTorch | General-purpose EBM toolkit |
| [mini-ebm](https://github.com/yataobian/mini-ebm) | Educational | PyTorch | Minimal educational EBM implementation |
| [Kona 1.0](https://logicalintelligence.com/kona-ebms-energy-based-models) | Logical Intelligence | — | Continuous latent reasoning via EBMs |
| [UvA Deep Energy Models Tutorial](https://github.com/phlippe/uvadlc_notebooks) | UvA | PyTorch | Tutorial 8: deep energy-based models |
| [Equilibrium Matching](https://energy-based-model.github.io/) | — | — | EBM training via equilibrium matching |

### Hardware

- [Extropic TSU/XTR-0](https://extropic.ai/writing/inside-x0-and-xtr-0) — Thermodynamic Sampling Unit for native EBM inference in hardware

## License

Apache 2.0 — see [LICENSE](LICENSE).
