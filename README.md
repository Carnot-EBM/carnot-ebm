# Carnot

**Open-source Energy Based Model framework — Rust + Python/JAX**

Carnot is an EBM framework that combines energy-based verification with large language models to reduce hallucinations. Through 25 systematic experiments on real models (Qwen3-0.6B and Qwen3.5-0.8B), we established what works: logprob rejection sampling (+10% accuracy on 20 questions; see [Limitations](docs/technical-writeup.md#8-limitations)), composite scoring (0% → 30% for code on 10 tasks vs. unmodified LLM baseline), and per-token EBM training (84.5% on base model). We discovered that instruction tuning compresses hallucination signals (84.5% → 67.2%) — meaning the models most deployed in production are hardest to monitor — and that chain-of-thought reasoning compresses them further (75.5% without thinking vs 61.3% with). Ships with an MCP server and CLI for Python code verification. See the [technical report](docs/technical-report.md) for full results.

## The Problem with LLMs

Large language models generate text by predicting the most probable next token. This produces fluent output, but it's fundamentally guessing — there is no mechanism to verify that the output is logically consistent, physically valid, or factually correct. When an early token is wrong, the error cascades irrecoverably. This is why LLMs hallucinate: they optimize for plausibility, not truth.

## Why Energy Based Models?

EBMs take a fundamentally different approach. Instead of generating outputs sequentially, they assign a scalar energy to every possible configuration of variables. Low energy = valid/consistent; high energy = invalid/contradictory. Inference is optimization: find the configuration that minimizes energy across all constraints simultaneously.

This enables capabilities that autoregressive models structurally cannot provide:

- **Verifiable reasoning** — mathematically prove a solution satisfies all constraints by showing it sits at an energy minimum
- **Surgical error correction** — when a constraint is violated, gradient descent fixes the broken part without discarding the rest
- **Autonomous self-improvement** — the energy function is an objective ground truth that cannot be gamed, enabling closed-loop self-learning without human feedback
- **Hardware acceleration** — energy landscapes map directly to thermodynamic sampling hardware (Extropic TSU), promising 10,000x efficiency gains

## How Carnot Uses EBMs (Introspection, Not Fine-Tuning)

**Carnot never modifies the LLM's weights.** The target language model remains completely frozen throughout all experiments and deployment. Instead, Carnot works by introspecting the LLM's existing internal representations:

1. **Logprob-based methods** — read the LLM's own per-token log-probabilities as an energy signal. The model is already an EBM (per the ARM↔EBM bijection); we simply read the energy it already computes.
2. **Activation-based methods** — extract hidden state activations from a frozen forward pass (`output_hidden_states=True`), then train a small separate EBM classifier (a lightweight Gibbs model, typically [1024→256→64→1]) on those extracted features via Noise Contrastive Estimation.
3. **Structural verification** — execute the LLM's generated code against test cases. No model weights involved at all.

The "training" in Carnot refers to training the small EBM classifier on activation features extracted from a frozen LLM — not gradient descent on the LLM itself. This is fundamentally different from fine-tuning, RLHF, or DPO, which modify the language model's parameters. Carnot's approach is closer to probing or introspection: we observe what the model already knows internally and build a lightweight detector on top of it.

## The Path to Self-Learning

Carnot is designed from the ground up to support an automated self-improvement loop (LLM proposes, energy function evaluates):

1. **Propose** — candidate improvements to architecture, training, or hyperparameters are prototyped in Python/JAX
2. **Evaluate** — the energy landscape on held-out data serves as the objective judge (did energy decrease? real improvement. did it not? rejected.)
3. **Deploy** — proven improvements are transpiled to Rust for production performance
4. **Repeat** — the loop runs without human supervision, with safety guardrails

The EBM itself is the evaluator. No LLM needed to judge quality — the math provides ground truth.

## Key Results (25 experiments)

| Approach | Domain | Result |
|----------|--------|--------|
| Logprob rejection sampling | QA/Factual | **+10% accuracy** (45% → 55%, n=20 questions) |
| Composite scoring (logprob + tests) | Code | **0% → 30% accuracy** (n=10 tasks, vs. unmodified baseline) |
| SAT gradient repair | Constraint satisfaction | **60% → 80%** (Haiku benchmark) |
| Per-token EBM (base model) | Activation analysis | **84.5% test accuracy** (Qwen3-0.6B) |
| Per-token EBM (no thinking) | Activation analysis | **75.5% test accuracy** (Qwen3.5-0.8B, thinking disabled) |
| Per-token EBM (with thinking) | Activation analysis | 61.3% test accuracy (Qwen3.5-0.8B, thinking enabled) |
| Activation steering | In-generation | 0% effect (negative result) |
| EBM rejection sampling | Adversarial QA | -3% to -6% (negative result) |

**What works:** The model's own logprobs + structural test execution, combined as a composite energy score. **What doesn't:** Activation-based steering during generation, and rejection sampling on adversarial questions. **Key insights:** (1) Instruction tuning compresses hallucination signals, making the most-deployed models hardest to monitor (Principle 8), (2) chain-of-thought compresses them further — disabling thinking improves detection by 14.2% (Principle 10), (3) adversarial questions defeat post-hoc detection entirely (Principle 9). **Caveat:** QA results are on small samples (20 questions) without statistical significance testing. Code results compare against unmodified LLM output, not against fine-tuning or RLHF. See [Limitations](docs/technical-writeup.md#8-limitations).

See the [technical writeup](docs/technical-writeup.md) for the full write-up, or the [technical report](docs/technical-report.md) for a summary of all 25 experiments.

## 10 Principles Learned

Hard-won lessons from 25 experiments on real models:

1. **Simpler is better in small-data regimes.** Linear projections outperform nonlinear models when you have fewer than 100 training examples.
2. **Token-level features > sequence-level.** Mean-pooling across tokens destroys the signal. Per-token features preserve it.
3. **The model's own logprobs are the best energy.** No external EBM needed for rejection sampling — the LLM's own confidence is already an energy function.
4. **Overfitting is the main enemy.** Every approach that trains on calibration data overfits when examples < dimensions.
5. **Extract features from generated tokens, not prompts.** The hallucination signal lives in the GENERATED tokens, not the input.
6. **Different energy signals dominate in different domains.** Logprobs for QA, structural tests for code. The composite combines both.
7. **Statistical difference ≠ causal influence.** A direction that separates correct from hallucinated activations does NOT steer the model when injected during generation.
8. **Instruction tuning compresses the hallucination signal.** Base models: 84.5%. Instruction-tuned: 67.2%. RLHF makes models sound confident even when wrong. **This is a fundamental limitation:** the models most in need of hallucination detection (instruction-tuned models deployed in production) are precisely the ones where activation-based detection is hardest.
9. **Adversarial questions defeat post-hoc detection.** On TruthfulQA, neither logprob nor EBM rejection improves over greedy. Detection must move upstream.
10. **Chain-of-thought compresses the hallucination signal.** Disabling thinking improves detection from 61.3% → 75.5% (+14.2%). Thinking makes hidden states more uniform.

## Tools

### CLI

```bash
pip install -e .
carnot verify examples/math_funcs.py --func gcd --test "(12,8):4" --test "(7,13):1"
```

### MCP Server

Configure with `cp .mcp.json.example .mcp.json` for Claude Code integration. Exposes `verify_code`, `verify_with_properties`, and `score_candidates` tools via stdio JSON-RPC. These tools perform **Python code verification** (structural tests, property-based testing, candidate ranking) — they do not implement the activation-based EBM hallucination detection described in the research sections.

See [docs/usage-guide.md](docs/usage-guide.md) for detailed setup and usage instructions.

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

## Pre-trained Models

Pre-trained EBM models and activation datasets are available on HuggingFace at [huggingface.co/Carnot-EBM](https://huggingface.co/Carnot-EBM):

| Model | Accuracy | Source Model |
|-------|----------|-------------|
| `per-token-ebm-qwen3-06b` | 84.5% | Qwen3-0.6B (base) |
| `per-token-ebm-qwen35-08b-nothink` | 75.5% | Qwen3.5-0.8B (no thinking) |
| `per-token-ebm-qwen35-08b-think` | 61.3% | Qwen3.5-0.8B (with thinking) |

See [docs/huggingface-plan.md](docs/huggingface-plan.md) for the full publishing plan.

## License

Apache 2.0 — see [LICENSE](LICENSE).
