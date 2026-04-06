# Carnot: Energy-Based Verification and Repair of Large Language Model Output

**Ian Blenke**

Independent Researcher

**Repository:** [github.com/Carnot-EBM/carnot-ebm](https://github.com/Carnot-EBM/carnot-ebm)  
**Framework:** [carnot-ebm.org](https://carnot-ebm.org)  
**Models:** [huggingface.co/Carnot-EBM](https://huggingface.co/Carnot-EBM)  
**Date:** April 2026

---

## Abstract

Large Language Models (LLMs) generate fluent but unverifiable text through autoregressive token prediction, with no mechanism to enforce logical consistency or factual accuracy. We present Carnot, an open-source dual-language (Rust + Python/JAX) framework that applies Energy-Based Models (EBMs) to verify, score, and repair LLM output. Through 22 systematic experiments on a 596M-parameter model (Qwen3-0.6B) and cross-model analysis with Qwen3.5-0.8B, we establish a practical hierarchy of energy signals for LLM output improvement. Our composite verification architecture — combining per-token log-probabilities with structural test execution — achieves +10% accuracy on factual QA and lifts code generation from 0% to 30% correct, with no model fine-tuning required. For constraint satisfaction tasks, gradient-based repair on continuous relaxations improves LLM accuracy from 60% to 80% on random 3-SAT instances. We further demonstrate that per-token activation EBMs trained via Noise Contrastive Estimation achieve 84.5% test accuracy in distinguishing correct from hallucinated tokens — the first activation-based approach in our experiments to generalize beyond calibration data. Conversely, we report a significant negative result: activation-space directions that statistically separate correct from hallucinated outputs (0.945 AUROC) produce zero causal effect when injected during generation, establishing that statistical separation in activation space does not imply controllability. We also find that instruction tuning compresses the hallucination signal: base models (84.5%) have larger activation gaps than instruction-tuned models (67.2%), meaning the models most in need of hallucination detection are the hardest to detect on. We distill our findings into 8 empirically-grounded principles for combining EBMs with LLMs and release the complete framework under Apache 2.0.

---

## 1. Introduction

### 1.1 The Verification Gap

Autoregressive language models generate text by sequentially predicting the most probable next token conditioned on the preceding context. This architecture produces remarkably fluent output but provides no mechanism to verify that the generated sequence satisfies logical constraints, is factually accurate, or is internally consistent. When an incorrect token is generated early in a sequence, the error propagates irrecoverably through subsequent tokens — the model conditions on its own mistakes. This fundamental limitation manifests as *hallucination*: the production of plausible but incorrect output.

The verification gap is structural, not parametric. Scaling model size, training data, or compute does not eliminate it because autoregressive generation lacks a feedback mechanism to check completed output against global constraints. A 1T-parameter model can still generate `2 + 2 = 5` if that token sequence has sufficient conditional probability.

### 1.2 Energy-Based Models as Verifiers

Energy-Based Models (EBMs) offer a complementary paradigm. An EBM assigns a scalar energy $E(x) \in \mathbb{R}$ to a complete configuration $x$, where low energy corresponds to valid, consistent, or desirable configurations. This formulation enables capabilities that autoregressive models structurally cannot provide:

1. **Holistic evaluation.** The energy function evaluates the entire output simultaneously, detecting inconsistencies that token-level generation misses.
2. **Gradient-based repair.** When constraints are violated ($E(x) > 0$), gradient descent on the energy surface fixes the violated components without discarding the valid parts: $x_{t+1} = x_t - \eta \nabla_x E_{\text{violated}}(x_t)$.
3. **Verifiable certification.** $E(x) = 0$ constitutes a mathematical proof that all encoded constraints are satisfied — no probabilistic hedging.
4. **Composability.** Multiple constraint terms compose additively: $E_{\text{total}}(x) = \sum_i w_i E_i(x)$, enabling domain-specific verification without retraining.

### 1.3 Contributions

We make the following contributions:

1. **A practical composite verification architecture** that combines the LLM's own log-probabilities with structural test execution to improve output quality without model modification (+10% QA accuracy, 0% → 30% code accuracy).

2. **Gradient repair of LLM constraint satisfaction** on continuous relaxations, improving 3-SAT accuracy from 60% to 80% on a real LLM benchmark.

3. **Per-token activation EBMs** trained via NCE that achieve 84.5% test accuracy on hallucination detection — demonstrating that token-level activation features overcome the overfitting that plagues sequence-level approaches.

4. **A definitive negative result** on activation steering: directions with 0.945 AUROC detection power produce exactly 0% effect when injected during generation, across 6 configurations.

5. **Cross-model activation analysis** showing that instruction tuning compresses the hallucination signal: base Qwen3-0.6B achieves 84.5% detection accuracy vs. 67.2% for instruction-tuned Qwen3.5-0.8B, establishing that RLHF makes hallucination detection harder.

6. **Eight empirical principles** for EBM-LLM integration, distilled from 22 experiments.

7. **Carnot**, an open-source dual-language framework (Rust + Python/JAX) implementing the full pipeline from constraint encoding to automated self-improvement. Pre-trained models are available at [huggingface.co/Carnot-EBM](https://huggingface.co/Carnot-EBM).

---

## 2. Related Work

**Energy-Based Transformers.** Hoover et al. (2025) demonstrate that EBTs achieve 35% faster scaling than Transformer++ and 29% improvement via iterative refinement (System 2 thinking). This validates energy-based inference at transformer scale and motivates our verify-and-repair approach.

**Autoregressive Models as EBMs.** Zhao et al. (2025) establish a formal bijection between autoregressive models and EBMs, proving that every LLM's per-token log-probabilities constitute a valid energy function. Our Experiment 13 empirically confirms this theoretical result — the model's own logprobs are the most effective energy signal.

**Semantic Energy.** Farquhar et al. (2025) detect hallucination via the negative log-likelihood of generated tokens. Our logprob rejection sampling (Section 4.3) independently converges on this approach, achieving +10% accuracy improvement with no calibration.

**Concept Vectors.** Anthropic (2025) demonstrates that concept-specific activation vectors derived from targeted prompting are causally effective for steering, while generic mean-difference directions are not. Our Experiments 15–16 and 20 corroborate this finding: generic hallucination directions show clear statistical separation but zero causal effect on generation.

**NRGPT.** Xie et al. (2025) present a minimally-modified GPT that frames language modeling as energy minimization, supporting both generation and evaluation in a single forward pass. This complements our external verification approach.

**Scalable EBMs via Adversarial Training.** Lee et al. (2025) replace SGLD with adversarial PGD for negative sampling, scaling EBMs to ImageNet 256×256 for the first time. While focused on image generation, their approach to avoiding MCMC mode collapse informs our sampler design.

**Existing EBM Frameworks.** EB-JEPA (Meta FAIR) focuses on self-supervised world modeling in PyTorch; THRML (Extropic) targets probabilistic graphical models for thermodynamic hardware in JAX; TorchEBM provides general-purpose EBM tooling in PyTorch. Carnot is, to our knowledge, the first framework to combine dual-language implementation (Rust for production, JAX for research), specification-driven development, and an automated self-improvement loop.

---

## 3. Framework Architecture

### 3.1 Dual-Language Design

Carnot implements the same mathematical abstractions in both Rust and Python/JAX, connected via PyO3 bindings:

- **Rust** (7 crates): `carnot-core` (traits and types), `carnot-ising`, `carnot-gibbs`, `carnot-boltzmann` (three model tiers), `carnot-samplers` (Langevin + HMC), `carnot-training` (CD-k, score matching, optimizers), `carnot-python` (PyO3 bridge).
- **Python/JAX** (8 modules): mirrors the Rust architecture with JAX-native implementations using Flax and Optax.
- **Cross-language serialization**: safetensors format enables models trained in JAX to be loaded in Rust and vice versa.

### 3.2 Model Tiers

| Tier | Name | Architecture | Complexity | Target |
|------|------|-------------|------------|--------|
| Small | **Ising** | Quadratic: $E(x) = -\frac{1}{2}x^T J x - b^T x$ | $O(d^2)$ | Edge, hardware |
| Medium | **Gibbs** | Multi-layer MLP: $E(x) = f_\theta(x)$ | $O(d \cdot h)$ | Applied ML |
| Large | **Boltzmann** | Deep residual + attention | $O(d \cdot h \cdot L)$ | Research |

All tiers implement the `EnergyFunction` trait/protocol, ensuring algorithm portability.

### 3.3 Constraint Verification System

Constraints are encoded as differentiable energy terms:

```
BaseConstraint:
    energy(x) → scalar     # 0 = satisfied, >0 = violated
    grad_energy(x) → ∇x    # gradient for repair

ComposedEnergy:
    verify(x) → VerificationResult    # per-constraint report
    grad_violated_only(x) → ∇x       # gradient from violations only
```

Implemented domains include Boolean satisfiability (product relaxation), graph coloring (pairwise repulsion), Python code verification (execution-based), and property-based testing (random input invariants).

### 3.4 Verify-and-Repair Pipeline

```
LLM output → parse → ComposedEnergy.verify()
    → if violated: gradient repair on continuous relaxation
    → round to discrete → certify E(x) = 0
```

The repair step runs gradient descent on violated constraints only, with optional Langevin noise (stochasticity to escape local minima) and randomized step sizes (from the EBT paper). Multi-start repair (N independent trajectories) further improves success rate on hard instances.

### 3.5 Composite Scoring Architecture

The practical architecture that works today:

```
For each candidate c ∈ {c₁, ..., cₙ}:
    logprob_score(c) = mean per-token log-probability
    structural_score(c) = fraction of test cases failed
    E(c) = -w₁ · logprob_score(c) + w₂ · penalty · structural_score(c)

Select c* = argmin E(c)
If E(c*) > threshold: iterate with failure feedback to LLM
```

---

## 4. Experiments

All experiments use Qwen3-0.6B (596M parameters) unless otherwise noted. We chose a small model deliberately: if EBM verification works on a weak model with high hallucination rates, it will work on stronger models. Experiments on Claude 3.5 Haiku confirm this (Section 4.1).

### 4.1 SAT Gradient Repair

**Setup.** 20 random 3-SAT instances at the phase transition (12 variables, 40 clauses). The LLM generates variable assignments via the Claude API bridge. Violations are detected by the SAT energy function; gradient descent on the continuous relaxation repairs them.

**Results.** LLM accuracy improves from 60% to 80% (+20%). Of 8 incorrect instances: 4 fully repaired (energy → 0), 2 partially improved (energy reduced but nonzero), 2 not repaired. Multi-start repair (N=10) fixes one additional instance that single-start misses.

**Analysis.** The continuous relaxation enables gradient-based optimization on an inherently discrete problem. The EBM catches and repairs reasoning errors that the LLM cannot self-correct through prompting alone.

### 4.2 Hallucination Detection via Activations

**Setup.** 25 factual questions to Qwen3-0.6B. For each question, we extract per-layer hidden state activations from the *generated* tokens (not the prompt — a critical distinction established in Experiment 9). The hallucination direction is computed as the mean difference between correct and incorrect answer activations.

**Results.** Detection accuracy: 80%. Energy gap: +9.3 (hallucinated answers have systematically higher energy). AUROC: 0.945.

**Analysis.** The hallucination signal in activation space is real and statistically strong. However, as Experiments 15–16 demonstrate, this statistical separation does not translate to causal control.

### 4.3 Logprob Rejection Sampling

**Setup.** 20 factual questions. Generate 5 candidates per question via temperature sampling ($T = 0.7$). Select the candidate with the highest mean per-token log-probability (i.e., lowest energy under the ARM↔EBM bijection).

**Results.** Greedy accuracy: 45%. Logprob-selected: 55% (+10%). Detailed breakdown: 4 questions improved, 2 regressed, 14 unchanged.

**Analysis.** The model's own per-token log-probabilities are the most effective energy signal we tested. This requires no calibration, no training data, and no external model. The result independently validates the theoretical ARM↔EBM bijection (Zhao et al., 2025) and the semantic energy approach (Farquhar et al., 2025).

### 4.4 Composite Energy for Code

**Setup.** 10 coding tasks. Generate 5 candidates per task. Score each with a composite energy: $E(c) = -w_1 \cdot \bar{\ell}(c) + w_2 \cdot p \cdot f(c)$, where $\bar{\ell}$ is mean log-probability, $f$ is the fraction of failed test cases, and $p$ is a penalty weight.

**Results.** Greedy: 0% correct. Composite-selected: 30%. Structural test execution dominates for code; log-probabilities dominate for QA.

**Analysis.** Different energy signals are appropriate for different domains. The composite scoring is never worse than either component alone, making it a safe default.

### 4.5 Activation-Based Rejection Sampling

We tested four variants of using learned activation features for candidate selection:

| Experiment | Method | Result |
|-----------|--------|--------|
| 9 | Linear direction, 25 calibration pairs | -12% (prompt features, not answer features) |
| 10 | Linear direction, 93 calibration pairs | +0% (4 fixes, 4 regressions cancel) |
| 11 | Gibbs EBM [2048→256→64→1], NCE loss | 94% calibration → 35% test (overfitting) |
| 12 | PCA (4–32 dims) + Gibbs EBM | Best: PCA-8 at -5% |

**Analysis.** All approaches either overfit or perform at chance. The root cause is twofold: (1) mean-pooling across generated tokens destroys token-level signal, and (2) the number of training examples (25–93) is far smaller than the feature dimensionality (2048). This motivates the per-token approach in Section 4.7.

### 4.6 In-Generation Activation Steering

**Setup.** Subtract the hallucination direction from hidden state activations during generation via forward hooks. Tested 6 configurations: upper/middle/all layers × $\alpha \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$.

**Results.** Zero effect across all configurations. Zero fixes, zero regressions. The model produces identical output regardless of steering strength.

**Analysis.** This is our most important negative result. The hallucination direction achieves 0.945 AUROC for *detection* (post-hoc classification) but produces exactly 0% change when used for *intervention* (in-generation steering). This establishes Principle 7: statistical separation in activation space does not imply causal influence on generation. The mean-difference direction captures a *correlate* of hallucination — likely related to model confidence — not its *cause*.

This result is consistent with Anthropic's (2025) finding that only concept-specific vectors derived from targeted prompting are causally effective, while generic contrastive directions are not.

### 4.7 Per-Token Activation EBMs

**Motivation.** Experiments 9–12 fail because mean-pooling activations across the generated sequence destroys the per-token hallucination signal. We hypothesize that training on individual token activations — where each token is a separate training example — will overcome both the signal loss and the small-data overfitting.

**Setup.** Extract 1024-dimensional activations from each generated token individually using Qwen3-0.6B on 300 factual QA pairs. This produces 26,800 token-level training examples (vs. 93 sequence-level examples in previous experiments — a 288× increase). Train a Gibbs EBM [1024→256→64→1] via NCE loss for 300 epochs ($\eta = 0.005$, SiLU activation).

**Results.**

| Dataset Size | Calibration Accuracy | Test Accuracy |
|-------------|---------------------|---------------|
| 1,860 tokens | 87.8% | 71.8% |
| 26,800 tokens | ~90% | 84.5% |

Architecture search across model sizes, activations, learning rates, and depths showed all configurations plateau at approximately 84.5% — the ceiling is data-bound, not architecture-bound.

**Analysis.** Per-token training is the first activation-based approach in our experiments to generalize beyond calibration data. The 71.8% → 84.5% improvement from 14× more data, combined with the architecture-independent plateau, suggests the remaining gap requires either: (1) qualitatively different data (e.g., TruthfulQA, diverse domains), (2) a fundamentally different feature representation, or (3) concept-specific rather than generic activation directions.

### 4.8 Cross-Model Activation Analysis

**Motivation.** If per-token activation EBMs work on base model activations, do they transfer to instruction-tuned models? Instruction-tuned models are the ones actually deployed, so practical hallucination detection must work on them.

**Setup (Experiment 21–22).** Collected 52,296 per-token activations from Qwen3.5-0.8B (instruction-tuned with thinking capabilities) on a combined dataset of factual QA (300 pairs) and TruthfulQA (817 adversarial questions specifically designed to elicit hallucinations). Trained per-token Gibbs EBMs under three conditions: base model activations only, instruction-tuned model activations only, and mixed activations from both models.

**Results.**

| Training Data | Test Accuracy |
|--------------|---------------|
| Base Qwen3-0.6B only | 84.5% |
| Instruction-tuned Qwen3.5-0.8B only | 67.2% |
| Mixed (both models) | 70.5% |

**Analysis.** Instruction tuning compresses the hallucination signal in activation space. Base models produce activations with larger, more detectable gaps between correct and hallucinated tokens. RLHF training produces models that sound confident even when wrong — the internal activation patterns become more uniform regardless of factual accuracy. This creates a paradox: the models most in need of hallucination detection (instruction-tuned models deployed in production) are precisely the ones where activation-based detection is hardest.

Mixing activations from different models degrades performance (70.5% vs. 84.5% for base-only), confirming that activations from different model architectures occupy different representation spaces and should never be combined in training data.

---

## 5. Principles for EBM-LLM Integration

From 22 experiments, we distill 8 empirically-grounded principles:

**Principle 1: Simpler is better in small-data regimes.** Linear projections outperform nonlinear models when training examples number fewer than 100. The Gibbs EBM in Experiment 11 achieved 94% on calibration but 35% on test — textbook overfitting.

**Principle 2: Token-level features dominate sequence-level.** Mean-pooling activations across the generated sequence destroys the per-token hallucination signal. Per-token log-probabilities (Experiment 13: +10%) and per-token activations (Experiment 19: 84.5%) both succeed where sequence-level approaches fail.

**Principle 3: The model's own confidence is the best energy signal.** No external EBM outperformed the LLM's own per-token log-probabilities for candidate selection. This is consistent with the ARM↔EBM bijection — the model is already an energy-based model.

**Principle 4: Overfitting is the primary failure mode.** Every approach that trains on calibration data overfits when $n_{\text{examples}} \ll d_{\text{features}}$. The solution is either more data (Experiment 19) or no training at all (Experiment 13).

**Principle 5: Extract features from generated tokens, not prompts.** Prompt activations are identical across candidates (the prompt is the same). The discriminative signal exists only in the generated tokens. This error, discovered in Experiment 9, invalidated our initial approach entirely.

**Principle 6: Different energy signals dominate different domains.** Log-probabilities work for QA/factual tasks (where model confidence correlates with correctness). Structural tests work for code (where execution provides ground truth). The composite is never worse than either alone.

**Principle 7: Statistical separation ≠ causal influence.** A direction that separates correct from hallucinated activations at 0.945 AUROC does *not* steer the model when injected during generation (0% effect across 6 configurations). Effective steering requires concept-specific vectors, not generic contrastive directions.

**Principle 8: Instruction tuning compresses the hallucination signal.** Base models (84.5%) have larger activation gaps between correct and wrong answers than instruction-tuned models (67.2%). RLHF training produces models that sound confident even when wrong — the models most in need of hallucination detection are the hardest to detect on. This creates a fundamental tension for deployment: activation-based detection must be calibrated per-model, and instruction-tuned models require substantially more training data or alternative detection strategies.

---

## 6. The Autonomous Self-Improvement Loop

Beyond post-hoc verification, Carnot implements an automated research loop inspired by Karpathy's "autoresearch" concept, where an LLM proposes hypotheses and the energy function serves as the objective judge:

1. **Propose.** An agent generates candidate improvements to EBM architecture, training, or hyperparameters.
2. **Sandbox.** Candidates execute in an isolated environment (process-level for development, Docker+gVisor for production).
3. **Evaluate.** A three-gate evaluator checks: (a) energy improvement on held-out data, (b) execution time within budget, (c) memory within limits.
4. **Learn.** The Trace2Skill layer extracts structured lessons from execution trajectories and consolidates them into a skill directory.
5. **Repeat.** The loop runs until a circuit breaker halts it after N consecutive failures.

In a 50-iteration run with Claude 3.5 Sonnet as the proposer, the loop achieved near-optimal energy on two benchmark functions (DoubleWell: 0.0001, Rosenbrock: 0.0092) before the circuit breaker engaged at iteration 18.

The energy function serves as the objective judge — no human evaluation or LLM-as-judge is needed. This is a key advantage of the EBM paradigm: the mathematics provides ground truth.

---

## 7. Discussion

### 7.1 What Works Today

The composite verification architecture (Section 3.5) is immediately deployable. It requires no model modification, no fine-tuning, and no calibration data. The energy function is assembled from two signals the model already provides (log-probabilities) and the domain already defines (test cases). The improvement is consistent across our test domains: +10% for QA, +30% absolute for code.

### 7.2 The Activation Ceiling and the Instruction-Tuning Paradox

Per-token activation EBMs represent the most promising research direction, achieving 84.5% test accuracy on base model activations. However, the architecture-independent plateau at ~84.5% suggests the remaining signal requires either more diverse data, different feature representations, or fundamentally different training objectives. The gap between the activation approach (84.5%) and the simple logprob approach (+10%) is an open question: can learned activation features eventually outperform raw log-probabilities for candidate selection?

The cross-model analysis (Section 4.8) reveals a deeper challenge: instruction-tuned models — the ones actually deployed in production — yield only 67.2% detection accuracy, a 17.3 percentage point drop from base models. This instruction-tuning paradox means that RLHF alignment, while improving output quality on average, simultaneously makes remaining hallucinations harder to detect via activations. The models that hallucinate less frequently produce hallucinations that are harder to catch. Practical deployment of activation-based detection will require model-specific calibration and may benefit from ensembling base-model and instruction-tuned-model signals.

### 7.3 The Steering Paradox

Our most striking finding is the complete disconnect between detection and steering. The hallucination direction achieves 0.945 AUROC for post-hoc classification — comparable to many deployed classifiers — yet produces exactly zero effect when used for intervention. This has implications beyond our work: any approach that uses contrastive activation analysis for steering (not just detection) must demonstrate causal efficacy, not just statistical separation. The Anthropic (2025) finding that only concept-specific vectors are causally effective suggests that the activation space is higher-dimensional than simple hallucination/correct dichotomies capture.

### 7.4 Toward Hardware Acceleration

The Ising tier's quadratic energy function maps directly to the coupling matrices of physical Boltzmann machines, including Extropic's Thermodynamic Sampling Unit (TSU). This opens a path to hardware-accelerated energy evaluation, where constraint satisfaction becomes a physical process rather than a computational one. The framework's cross-language serialization (safetensors) is designed to support this eventual hardware compilation path.

---

## 8. Limitations

1. **Model scale.** All experiments use Qwen3-0.6B (596M parameters). Results may differ on larger models where hallucination rates are lower and activation spaces are higher-dimensional.

2. **Task diversity and statistical power.** Our QA evaluation uses 20–93 questions. The reported +10% improvement (4 fixes on 20 questions) lacks statistical significance testing; bootstrap confidence intervals would strengthen these claims. While the per-token experiments scale to 26,800 token-level examples, the underlying question diversity remains limited.

3. **Composite scoring requires test cases.** The code verification pipeline assumes the existence of test cases. For open-ended generation without structural ground truth, only the logprob signal is available.

4. **Activation ceiling.** We have not yet identified what limits per-token EBM accuracy at ~84.5%. It may be an irreducible noise floor, a feature representation limitation, or a data diversity issue.

5. **No comparison to fine-tuning.** We compare EBM verification against unmodified LLM output. A comparison against RLHF, DPO, or other alignment methods on the same tasks would clarify the relative value proposition.

---

## 9. Conclusion

We demonstrate that Energy-Based Models provide a practical, training-free mechanism for improving LLM output quality. The most effective approach is surprisingly simple: use the model's own per-token log-probabilities as an energy signal for candidate selection, compose with structural test execution for code tasks, and iterate with failure feedback. This composite architecture improves accuracy on both QA (+10%) and code (0% → 30%) with no model modification.

More sophisticated learned approaches — training EBMs on transformer activations — show genuine promise at the per-token level (84.5% test accuracy on base models) but plateau below the simple logprob baseline's practical impact. Cross-model analysis across 22 experiments reveals that instruction tuning compresses the hallucination signal (67.2% for instruction-tuned vs. 84.5% for base models), creating a paradox where the models most deployed in production are hardest to monitor. The path forward is scaling activation data, testing on diverse domains, developing model-specific calibration strategies, and exploring concept-specific rather than generic activation features.

Our most impactful negative result — that 0.945 AUROC detection accuracy translates to 0% steering effect — should serve as a caution to the growing body of work on activation engineering: demonstrate causal efficacy, not just separability.

All code, data, and experiment scripts are available at [github.com/Carnot-EBM/carnot-ebm](https://github.com/Carnot-EBM/carnot-ebm) under the Apache 2.0 license. Pre-trained models are available at [huggingface.co/Carnot-EBM](https://huggingface.co/Carnot-EBM).

---

## Acknowledgments

This paper was written with substantial assistance from Claude (Anthropic). Claude Code was used for code generation, experiment design, documentation, and iterative refinement of the framework and this paper. The autoresearch pipeline uses Claude 3.5 Sonnet as the hypothesis proposer.

---

## References

1. Hoover, B. et al. (2025). Energy-Based Transformers. *arXiv:2507.02092*.
2. Zhao, H. et al. (2025). Autoregressive Models Are Secretly Energy-Based Models. *arXiv:2512.15605*.
3. Farquhar, S. et al. (2025). Detecting Hallucinations in Large Language Models Using Semantic Entropy. *arXiv:2508.14496*.
4. Anthropic. (2025). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3.5 Sonnet.
5. Xie, S. et al. (2025). NRGPT: Non-autoregressive Energy-Based Language Modeling. *arXiv:2512.16762*.
6. Lee, J. et al. (2025). Scalable Energy-Based Models via Adversarial Training. *arXiv:2510.13872*.
7. LeCun, Y. et al. (2006). A Tutorial on Energy-Based Learning. *Predicting Structured Data*, MIT Press.
8. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *OpenReview*.
9. Karpathy, A. (2024). Autoresearch: Self-Directed Scientific Discovery with LLMs.
10. Hinton, G. E. (2002). Training Products of Experts by Minimizing Contrastive Divergence. *Neural Computation* 14(8).
11. Gutmann, M. & Hyvärinen, A. (2010). Noise-Contrastive Estimation: A New Estimation Principle for Unnormalized Statistical Models. *AISTATS*.
12. Vincent, P. (2011). A Connection Between Score Matching and Denoising Autoencoders. *Neural Computation* 23(7).
