# Carnot

**Open-source Energy Based Model framework — Rust + Python/JAX**

Carnot is an Energy-Based Model framework for **verifying and repairing LLM outputs**. All headline benchmark results below are from **live GPU inference**. The repository also preserves simulated, unverified, and software-model artifacts for provenance, and labels them explicitly instead of folding them into headline claims.

**Headline result:** Full 164-problem HumanEval with property-based testing: **+3.0pp** [+0.6, +6.1] 95% CI (Exp 226). PBT detects 99.3% of wrong code. The latest same-cohort cross-model check (Exp 227) keeps Qwen3.5-0.8B flat at **23.3% -> 23.3%**, still detects **17/23** wrong baselines, and catches **2** official-test misses beyond the weak harness. Held-out replay keeps success flat at **32.74%** while cutting false positives **7 -> 1** (Exp 223). Trace learning distills **164** learnable PBT histories into property and repair rankings (VERIFY-030). Typed IR constraints give **+4.9pp** on Gemma4 (Exp 221).

**What ships today:** `VerifyRepairPipeline` plus standalone `verify_code()` for end-user code checks. CLI (`carnot pipeline verify`, `carnot verify-code`), a hardened **7-tool** MCP server for Claude Code, 5 integration examples, and full API docs. Constraint extraction spans arithmetic, code, logic, and natural language domains.

**What we learned:** Activation-based EBMs detect confidence, not correctness (50% practical). The 14 principles from our systematic negative results save other researchers months of dead ends. Structural constraint verification is what actually works. See the [technical report](docs/technical-report.md) for full results.

## Install

```bash
# Python (3.11+)
pip install -e ".[dev]"

# Verify it works
carnot verify examples/math_funcs.py --func gcd --test "(12,8):4" --test "(7,13):1"
```

> GPU: `pip install carnot[cuda]` for CUDA 12. On AMD/ROCm, use `JAX_PLATFORMS=cpu`.
> Rust bindings (optional): `pip install carnot[rust]` with Rust toolchain installed.

### Quick start (Python API)

```python
from carnot.pipeline import VerifyRepairPipeline

pipeline = VerifyRepairPipeline()

# Correct answer — passes verification
result = pipeline.verify("What is 15 + 27?", "15 + 27 = 42")
print(result.verified)    # True

# Wrong answer — caught by constraint extraction
result = pipeline.verify("What is 15 + 27?", "15 + 27 = 43")
print(result.verified)    # False
print(result.violations)  # [ConstraintResult: "15 + 27 = 43 (correct: 42)"]
```

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

## Key Results (230+ experiments, 22 completed milestones)

All benchmark results below are from **live GPU inference**. Simulated and software-model artifacts remain in the repo, but they are labeled explicitly and are not mixed into the headline tables. See the [technical report](docs/technical-report.md) for the full history including what didn't work.

### Simulation vs Reality

Provenance snapshot: **13 live GPU artifacts**, **3 simulated artifacts**, **72 unverified artifacts**, and **1 software-model artifact** (Exp 228, software simulation). Only the live GPU subset informs the benchmark tables below.

## PBT Verification

Carnot's strongest live evidence is now the Hypothesis-backed code-verification path. The full Gemma run is positive and statistically significant; the seeded Qwen transfer check is intentionally honest about a flat repair delta while still showing additive verifier signal.

| Live PBT artifact | Baseline | +Carnot | Added verifier signal | Experiment |
|-------------------|----------|---------|-----------------------|------------|
| Full HumanEval 164 (Gemma4) | 11.6% | 14.6% | 144/145 wrong baselines detected; 6 official-test misses caught; 5 repairs | Exp 226 |
| Seeded HumanEval 30 (Qwen3.5-0.8B) | 23.3% | 23.3% | 17/23 wrong baselines detected; 2 official-test misses caught; 0 repairs | Exp 227 |
| Dual-model HumanEval 50 | 18.0% / 10.0% | 20.0% / 12.0% | 144/145 wrong-code detections across the paired slice | Exp 220 |

### Code verification (strongest domain)

| Benchmark | Baseline | +Carnot | Delta | Experiment |
|-----------|----------|---------|-------|------------|
| HumanEval 164 problems (PBT) | 11.6% | 14.6% | **+3.0pp** [+0.6, +6.1] CI | Exp 226 |
| HumanEval 30 problems (PBT, seeded Qwen cohort) | 23.3% | 23.3% | +0.0pp; 2 harness misses caught | Exp 227 |
| HumanEval 50 problems (PBT) | 18.0% / 10.0% | 20.0% / 12.0% | +2.0pp both models | Exp 220 |
| HumanEval 30 problems (execution) | 16.7% | 20.0% | +3.3pp | Exp 208 |

PBT detects 99.3% of wrong code (144/145) on the full Gemma run and catches 6 official-test misses. Exp 227 shows the same verifier still adds signal cross-model even when the repair loop itself stays flat.
On deterministic HumanEval-style probes, the same Hypothesis-backed verifier also catches **5/5** under-specified bugs that execution-only checks miss while preserving **5/5** matching correct solutions (Exp 224).

### Constraint verification (math/instruction)

| Benchmark | Baseline | +Carnot | Delta | Experiment |
|-----------|----------|---------|-------|------------|
| Typed IR constraints (81 tasks) | 61.7% | 66.7% | **+4.9pp** (Gemma4) | Exp 221 |
| GSM8K semantic (200 questions) | 37.5% | 38.0% | +0.5pp (Gemma4) | Exp 219 |
| GSM8K arithmetic (100 questions) | 91.0% | 91.0% | 0.0pp | Exp 206 |

Semantic grounding detects 22-23% of wrong math answers (vs 0% for regex). The bottleneck is false positive rate (31% on Gemma4) — addressed by self-learning below.

### Self-learning

| Metric | Without learning | With learning | Improvement | Experiment |
|--------|-----------------|---------------|-------------|------------|
| Held-out false positives | 7 | 1 | **-86%** at flat **32.74%** success | Exp 223 |

On **168** held-out cases, tracker gating keeps held-out success flat at **32.74%** while reducing false positives **7 -> 1**. Memory reuse is not yet strong enough to add task gains under the stricter provenance gates: hit rate is **9.9%**, precision is **5.8%**, and the practical win is still false-positive control rather than higher task success.

### Latest follow-ons

- **Live trace memory (Exp 222):** **662** live trace events ingest into **230** accepted memory entries, **43** learned patterns, **29** mature patterns, **14** reusable repair snippets, and **12** monitorability-policy updates; replay precision is still only **12.6%**.
- **Hypothesis-backed verifier (Exp 224):** deterministic generated-code checks catch **5/5** under-specified bugs versus **0/5** for execution-only verification while preserving **5/5** matching correct solutions.
- **Dual-GPU runner (Exp 225):** paired fresh-process generation on the local **2x RTX 3090** host improves from **37.371s** to **32.774s** on a **10**-question microbenchmark for a measured **1.14x** speedup.
- **FPGA software model (Exp 228):** the KV260-class sparse **4,096-spin** backend now has a checked-in `FPGAIsingSampler` control-path design; the current **128**-spin benchmark is explicitly **software simulation** at **0.824549s** for `fpga_sim` versus **0.288092s** on CPU.
- **Trace learning (VERIFY-030):** `TraceAnalyzer`, `PropertyRanker`, and `RepairStrategy` normalize **164** Exp 226 histories, with signature-derived checks covering **163** cases and accounting for **6** official-test misses beyond the weak harness.
- **Packaged verification (VERIFY-031):** the strongest code path now ships as `verify_code()`, `carnot verify-code`, and `verify_code_with_pbt` on the hardened **7-tool** MCP surface.

### Infrastructure

| Component | Result | Experiment |
|-----------|--------|------------|
| Three extractors | Regex 5/91 FP, Z3 3/91 FP, LLM 1/91 FP | Exp 206-207 |
| Verify latency | 0.006ms per constraint check | Exp 102 |
| Parallel Ising sampler | 183x faster than thrml | Exp 102 |
| CoT monitorability | Free-form 100% parseable, JSON 18% | Exp 213 |
| Dual-GPU parallel | 1.14x speedup (Thunderbolt bottleneck) | Exp 225 |

### What works on test sets but fails in practice

| Approach | Test Accuracy | Practical Result | Why It Fails |
|----------|-------------|-----------------|-------------|
| Per-token EBM (best) | 88.5% | 50% on real questions | Detects confidence, not correctness |
| Multi-layer concat | 81.3% | Not tested in deployment | Same fundamental limitation |
| Activation steering | 0% effect | N/A | Statistical ≠ causal |
| Cross-model transfer | ~50% (chance) | N/A | Model-specific representations |
| Cross-domain training | 70.8% (worse) | N/A | Domain-specific signals |

**The core problem:** activation-based EBMs measure how confident the model is, not whether it's right. A model that confidently says "Neil Armstrong walked on Mars" produces activations indistinguishable from "Neil Armstrong walked on the Moon." The EBM rewards confident hallucination and penalizes correct hedging — the exact opposite of what a hallucination detector should do.

See the [technical report](docs/technical-report.md) for the full research record.

## 14 Principles Learned

Hard-won lessons from the activation-based phase of a research program that now spans 230+ experiments across 22 milestones and 16 model families. These negative results are the project's primary contribution — they document what doesn't work and why, saving other researchers months of dead ends.

### What works
1. **The model's own logprobs are the best energy.** No external EBM needed for rejection sampling — the LLM's own confidence is already an energy function. Simple, practical, +10%.
2. **Different energy signals dominate in different domains.** Logprobs for QA, structural tests for code. The composite combines both and is never worse than either alone.
3. **Multi-layer concatenation improves test-set detection by ~6%.** Concatenating activations from layers 4+12+24 achieves 81.3% vs 75.5% for the final layer alone.

### What doesn't work (and why)
4. **Activation EBMs detect confidence, not correctness.** The fundamental limitation. Confident hallucinations produce activations indistinguishable from confident correct answers. Test-set accuracy (75-88%) does not translate to practical detection (50%).
5. **Instruction tuning compresses the hallucination signal.** Base models: 86.8%. Instruction-tuned: 75.0%. RLHF makes models sound confident even when wrong.
6. **Chain-of-thought compresses it further.** Disabling thinking improves detection from 61.3% → 75.5%. Thinking makes hidden states more uniform.
7. **Statistical difference ≠ causal influence.** A direction that separates correct from hallucinated activations does NOT steer the model when injected during generation.
8. **Adversarial questions defeat post-hoc detection.** On TruthfulQA, neither logprob nor EBM rejection improves over greedy.
9. **Hallucination representations are model-specific.** Cross-model transfer is at chance (~50%). Each model needs its own EBM.
10. **EBM detection is domain-specific.** Mixing datasets hurts (70.8% < 75.5%). Mixing temperatures hurts. Train on your target domain only.
11. **Normalization doesn't enable transfer.** Z-score, L2, and PCA whitening all destroy signal without improving cross-domain or cross-model transfer.
12. **Upstream question-level detection is weak.** The model's representation of the question partially predicts hallucination (62.6%) but not usefully.

### Scaling observations
13. **EBM accuracy scales with model size** within a family. Qwen3.5: 75.5% (0.8B) → 88.5% (27B). But this is test-set accuracy — the confidence-vs-correctness problem applies at all scales.
14. **MoE architectures vary wildly.** Qwen3.5-35B has 256 genuinely specialized experts (0.008 overlap). Mixtral has 8 near-identical experts (0.997 overlap). Fundamentally different knowledge organization.

## Tools

### CLI

```bash
pip install -e .
carnot verify examples/math_funcs.py --func gcd --test "(12,8):4" --test "(7,13):1"
```

### MCP Server

Configure with `cp .mcp.json.example .mcp.json` for Claude Code integration. The hardened stdio JSON-RPC server now exposes **7** tools: `verify_code`, `verify_with_properties`, `verify_code_with_pbt`, `verify_llm_output`, `verify_and_repair`, `list_domains`, and `health_check`. These tools perform **Python code verification** and pipeline checks — they do not implement the activation-based EBM hallucination detection described in the research sections.

See [docs/usage-guide.md](docs/usage-guide.md) for detailed setup and usage instructions.

## Model Tiers

| Tier | Name | Scale | Use Case |
|------|------|-------|----------|
| Large | **Boltzmann** | Deep residual + attention | Research frontiers, large-scale generation |
| Medium | **Gibbs** | Multi-layer MLP (2-4 hidden) | Applied ML, complex pattern learning |
| Efficient | **KAN** | Learnable B-spline edges | **Default for verification** — best accuracy/param ratio (0.994 AUROC, 2.3K params) |
| Hardware | **Ising** | Pairwise quadratic | **Real-time sampling + FPGA/TSU** — direct hardware mapping, fastest parallel Gibbs |

All tiers implement the same `EnergyFunction` trait (Rust) / protocol (Python), so algorithms written against the interface work with any tier.

**When to use which:** KAN is the default for constraint verification (most accurate per parameter). Ising is for real-time guided decoding and hardware deployment (fastest sampling, maps to physical p-bits). They complement each other — KAN for accuracy, Ising for speed.

### Hardware Path

The current hardware track is the [FPGA Ising design](docs/fpga-ising-design.md) for a KV260-class sparse **4,096-spin** backend. Exp 228 validates the AXI-Lite upload/trigger/readback contract in **software simulation** with the new `FPGAIsingSampler` backend; the checked-in `fpga_sim` timing (`0.824549s` on a 128-spin sparse problem) is explicitly a software-model artifact, not a synthesized FPGA throughput claim. Exp 242 now attempts the real KV260 round trip and records the honest blocker in this environment: no `CARNOT_KV260_BITFILE` path was configured, so no board timings were fabricated. Exp 243 then replays **460** saved semantic and code repair cases through the sampler path: the CPU reranker stays neutral overall on top-1 quality and verifier precision, while the KV260-backed path remains blocked in the same environment.

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
│   ├── samplers/                  # Langevin, HMC, ParallelIsingSampler
│   ├── training/                  # JAX training loops with Optax
│   ├── pipeline/                  # VerifyRepairPipeline, extractors, errors
│   ├── mcp/                       # MCP server for Claude Code integration
│   ├── verify/                    # ComposedEnergy, ConstraintTerm, repair
│   └── inference/                 # EBM loader, composite scorer, LLM solver
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

- [FPGA Ising design](docs/fpga-ising-design.md) — KV260-class sparse **4,096-spin** overlay contract with `FPGAIsingSampler`, `SoftwareFPGAOverlay`, and AXI-Lite upload/trigger/readback semantics. Provenance: **software simulation** (Exp 228), not a live hardware-speed claim.
- [Extropic TSU/XTR-0](https://extropic.ai/writing/inside-x0-and-xtr-0) — Thermodynamic Sampling Unit for native EBM inference in hardware

## Pre-trained Models

16 per-token EBM models are available on HuggingFace at [huggingface.co/Carnot-EBM](https://huggingface.co/Carnot-EBM).

**Important caveat:** These models achieve 75-88% accuracy on held-out TruthfulQA test sets, but this metric is misleading. In practical deployment (8 real questions), the EBM agreed with ground truth only 50% of the time. The EBMs detect model *confidence*, not *correctness* — they are research artifacts for studying activation-space structure, not production hallucination detectors. See the [practical test results](scripts/experiment_practical_mcp_test.py).

| Model | Test Set Accuracy | Source Model | Notes |
|-------|----------|-------------|-------|
| `per-token-ebm-qwen35-27b-nothink` | 88.5% | Qwen3.5-27B | Highest test accuracy |
| `per-token-ebm-gemma4-e2b-nothink` | 86.8% | Gemma 4 E2B (base) | Best base model |
| `per-token-ebm-qwen35-9b-nothink` | 85.8% | Qwen3.5-9B | |
| `per-token-ebm-qwen35-35b-nothink` | 84.5% | Qwen3.5-35B-A3B | MoE, 256 experts |
| ... | 73-84% | 11 more models | See HuggingFace |

## License

Apache 2.0 — see [LICENSE](LICENSE).
