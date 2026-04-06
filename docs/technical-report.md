# Carnot: Energy-Based Verification for LLM Output

## A Technical Report on Combining Energy-Based Models with Large Language Models

**Author:** Ian Blenke
**Date:** 2026-04-06
**Repository:** github.com/Carnot-EBM/carnot-ebm

---

## Abstract

We present Carnot, an open-source framework that combines Energy-Based Models (EBMs) with Large Language Models (LLMs) to reduce hallucinations in generated output. Through 25 systematic experiments on real models (Qwen3-0.6B 596M-parameter base and Qwen3.5-0.8B instruction-tuned), we establish what works and what doesn't for EBM-based verification and correction of LLM output. Our key findings: (1) the model's own per-token log-probabilities are the most effective energy signal for candidate selection (+10% accuracy improvement), (2) structural test execution dominates for code verification (0% → 30% accuracy), (3) composite scoring combining both signals is never worse than either alone, (4) activation-space approaches show detectable signals but fail to improve output quality as a candidate filter on adversarial questions, (5) statistical separation in activation space does not imply causal influence on generation, (6) per-token EBM training on 52,296 tokens achieves 84.5% test accuracy on base models with architecture search showing data-bound plateau, (7) instruction tuning compresses the hallucination signal — base models (84.5%) have larger activation gaps than instruction-tuned models (67.2%), (8) adversarial questions defeat post-hoc detection entirely, (9) hallucination signal follows a U-curve across transformer layers, concentrating at the final layer, and (10) chain-of-thought reasoning further compresses hallucination signals — disabling thinking improves EBM detection from 61.3% to 75.5% (+14.2%). We release the complete framework including constraint verification, gradient repair, learned verifiers, autoresearch self-improvement, GPU compute (Vulkan/WebGPU), an MCP server, and a CLI tool.

---

## 1. Introduction

### 1.1 The Hallucination Problem

Large Language Models generate text by predicting the most probable next token. This produces fluent output but provides no mechanism to verify logical consistency, factual accuracy, or constraint satisfaction. When an LLM generates an incorrect early token, the error cascades irrecoverably through the remaining sequence.

### 1.2 The EBM Alternative

Energy-Based Models assign a scalar energy E(x) to complete configurations. Low energy = valid/consistent; high energy = invalid/contradictory. This enables:
- **Holistic evaluation**: assess the entire output at once, not token-by-token
- **Gradient-based repair**: when constraints are violated, gradient descent fixes the broken parts
- **Verifiable certification**: energy = 0 mathematically proves all constraints are satisfied

### 1.3 Introspection, Not Fine-Tuning

**Carnot never modifies the target LLM's weights.** The language model remains completely frozen. Our approach works by introspecting the model's existing internal representations:

- **Logprob methods** read the LLM's own per-token log-probabilities — energy the model already computes. Per the ARM↔EBM bijection, every autoregressive model is already an EBM.
- **Activation methods** extract hidden state activations from a frozen forward pass, then train a small separate EBM classifier (a lightweight Gibbs model [1024→256→64→1]) on those features via NCE.
- **Structural verification** executes generated output against domain constraints. No model weights involved.

When we say "EBM training," we mean training the small classifier on features from a frozen LLM — not gradient descent on the language model itself. This is closer to probing/introspection than to fine-tuning, RLHF, or DPO.

### 1.4 This Work

We investigate whether EBMs can practically improve LLM output through:
1. Post-hoc verification and repair (verify after generation)
2. Rejection sampling (generate N candidates, select by energy)
3. In-generation steering (modify activations during generation)

We report positive results for (1) and (2) with specific energy signals, and negative results for (3) and for activation-based energy signals.

---

## 2. Framework Architecture

### 2.1 Core EBM Framework

Carnot provides EBM implementations in both Rust (for production performance) and Python/JAX (for research iteration):

- **Three model tiers**: Ising (quadratic, O(d²)), Gibbs (multi-layer MLP), Boltzmann (deep residual)
- **Samplers**: Langevin dynamics + HMC, both with gradient clipping (REQ-SAMPLE-004)
- **Training**: Contrastive Divergence, Denoising Score Matching, Noise Contrastive Estimation, Self-Normalised Likelihood
- **Serialization**: safetensors for cross-language model sharing

### 2.2 Constraint Verification

The `verify` module encodes domain constraints as differentiable energy terms:

```python
class BaseConstraint:
    def energy(self, x) -> scalar    # 0 = satisfied, >0 = violated
    def grad_energy(self, x) -> grad  # gradient for repair

class ComposedEnergy:
    def verify(self, x) -> VerificationResult   # per-constraint breakdown
    def grad_violated_only(self, x) -> grad     # gradient from violations only
```

Implemented domains: SAT (product relaxation), graph coloring (pairwise repulsion), Python code (execution-based type/test checking), property-based testing (random input invariants).

### 2.3 Verify-and-Repair Pipeline

```
LLM output → parse → ComposedEnergy.verify() → if violated: repair() → round → certify
```

The `repair()` function runs gradient descent on violated constraints only, with optional Langevin noise and randomized step sizes (from the EBT paper).

### 2.4 GPU Compute

- **carnot-gpu**: wgpu-based Vulkan/Metal/DX12 compute for batch energy evaluation
- **carnot-webgpu-gateway**: distributed browser GPU compute via WebSocket

---

## 3. Experiments and Results

### 3.1 SAT Gradient Repair (Experiment 2)

**Setup:** 20 random 3-SAT instances (12 variables, 40 clauses). Haiku generates assignments via Claude API bridge.

**Result:** LLM accuracy 60% → repaired accuracy 80% (+20%). 4 instances fully repaired, 2 partially reduced, 2 not repaired. Multi-start repair (N=10) fixed an additional instance that single-start missed.

**Finding:** Gradient repair on continuous relaxation of discrete constraints works. The EBM catches and fixes LLM reasoning errors.

### 3.2 Real Hallucination Detection (Experiment 8)

**Setup:** 25 factual questions to Qwen3-0.6B. Extract mean-pooled activations from last + middle transformer layers. Compute hallucination direction via mean difference.

**Result:** Detection accuracy 64%. Energy gap +9.3 (hallucinated answers have higher energy).

**Finding:** The hallucination direction in activation space IS real. But 64% is insufficient for practical use.

### 3.3 Logprob Rejection Sampling (Experiment 13)

**Setup:** 20 factual questions. Generate 5 candidates per question via temperature sampling. Select the candidate with highest mean per-token log-probability.

**Result:** Greedy 45% → logprob-selected 55% (+10%). 4 fixes, 2 regressions, net +2.

**Finding:** The model's own logprobs are the best energy signal. No calibration, no training, no external EBM needed.

### 3.4 Composite Energy for Code (Experiment 14)

**Setup:** 10 coding tasks. Generate 5 candidates. Score each with: composite = -logprob_weight × mean_logprob + structural_weight × failure_penalty × n_test_failures.

**Result:** Greedy 0% → composite-selected 30%. Structural tests dominate for code; logprobs dominate for QA.

**Finding:** Different energy signals work for different domains. The composite handles both and is never worse than either alone.

### 3.5 Activation-Based Rejection Sampling (Experiments 9-12)

| Experiment | Approach | Result |
|-----------|----------|--------|
| 9 | Linear direction, 25 calibration | -12% |
| 10 | Linear direction, 93 calibration | +0% (4 fixes, 4 regressions) |
| 11 | Gibbs EBM, 2048-dim | 94% cal → 35% test (overfitting) |
| 12 | PCA + Gibbs, dim 4-32 | Best: PCA-8 at -5% |

**Finding:** Activation mean-pooling destroys the token-level signal. All approaches overfit or fail to generalize at small data scale.

### 3.6 In-Generation Activation Steering (Experiments 15-16)

**Setup:** Subtract hallucination direction from hidden states during generation via forward hooks. Tested on 25 QA questions across 6 configurations (upper/mid/all layers, alpha 0.1-5.0).

**Result:** 0% change across ALL configurations. Zero fixes, zero regressions.

**Finding:** Statistical separation in activation space does NOT imply causal influence on generation. This is our Principle #7. The mean-difference direction captures a correlate of hallucination, not its cause.

### 3.7 Scaled Per-Token EBM (Experiment 21)

**Setup:** Train per-token EBM on 26,800 tokens from Qwen3-0.6B (base model) across QA and TruthfulQA datasets. Architecture search across linear, 2-layer MLP, 3-layer MLP, and residual network models.

**Result:** 84.5% test accuracy. All architectures plateau at ~84.5% — the performance ceiling is data-bound, not architecture-bound.

**Finding:** Per-token features (validating Principle #2) scale well. The 84.5% ceiling suggests the remaining errors require either more diverse training data or richer features (e.g., attention patterns, cross-layer interactions), not deeper models.

### 3.8 Cross-Model TruthfulQA (Experiment 22)

**Setup:** Train per-token EBM on 52,296 tokens from Qwen3.5-0.8B (instruction-tuned model) across QA and TruthfulQA datasets. Same architecture as Experiment 21 for direct comparison.

**Result:** 67.2% test accuracy (vs 84.5% for the base model in Experiment 21).

**Finding:** Instruction tuning compresses the hallucination signal. RLHF training teaches the model to produce confident-sounding activations regardless of correctness, reducing the activation gap between truthful and hallucinated outputs. This is our Principle #8: base models are better EBM targets because their activations more honestly reflect uncertainty.

---

## 4. Principles Learned

From 25 experiments, we distilled 10 principles (Principles 9-10 from Experiments 23-25):

1. **Simpler is better in small-data regimes.** Linear projections outperform nonlinear models when you have <100 training examples.

2. **Token-level features > sequence-level.** Mean-pooling activations across generated tokens destroys the signal. Per-token logprobs preserve it.

3. **The model's own confidence is the best energy.** No external EBM outperformed the LLM's own logprobs for candidate selection.

4. **Overfitting is the main enemy.** Every approach that trains on calibration data overfits when examples < dimensions (42 examples in 2048-dim space).

5. **Extract features from generated tokens, not prompts.** Prompt activations are identical across candidates. The signal is in the GENERATED tokens.

6. **Different energy signals dominate in different domains.** Logprobs for QA/factual. Structural tests for code. Composite for both.

7. **Statistical difference ≠ causal influence.** A direction that separates correct from hallucinated activations (64% detection) does NOT steer the model when injected during generation (0% effect).

8. **Instruction tuning compresses the hallucination signal.** Base models (84.5%) have larger activation gaps between truthful and hallucinated outputs than instruction-tuned models (67.2%). RLHF makes the model sound confident even when wrong, reducing the energy separation that EBMs rely on. Target base models for EBM verification.

9. **Adversarial questions defeat post-hoc detection.** On TruthfulQA, neither logprob nor EBM rejection sampling improves over greedy (Experiment 23: -3% to -6%). Detection must move upstream of generation.

10. **Chain-of-thought reasoning compresses the hallucination signal.** Disabling thinking improves detection from 61.3% → 75.5% (+14.2%, Experiment 25). Thinking makes hidden states more uniform, with a 5.8× reduction in energy gap.

---

## 5. What Works: The Composite Verification Architecture

The practical "married pair" that works today:

```
LLM generates candidate(s)
    ↓
For each candidate:
    logprob_score = mean per-token log-probability
    structural_score = fraction of test cases failed
    composite = -logprob_weight × logprob + structural_weight × penalty × failures
    ↓
Select candidate with lowest composite score
    ↓
If still failing: feed violations back to LLM (iterative refinement)
    ↓
Repeat until verified or max iterations
```

This combines:
- **LLM confidence** (logprobs) — catches uncertain answers
- **Structural verification** (test execution) — catches wrong answers
- **Iterative refinement** (feedback loop) — the LLM learns from specific failures
- **Property-based testing** (random inputs) — catches edge cases beyond test suites

---

## 6. What Didn't Work and Why

### 6.1 Activation-Based Approaches

Every approach that used transformer hidden state activations for candidate SELECTION or generation STEERING failed at the data scales we tested (25-93 examples). The reasons:

- **Mean-pooling**: aggregating per-token activations into one vector loses the signal
- **Overfitting**: 42 examples in 2048-dim space → memorization, not generalization
- **Correlation ≠ causation**: activation patterns correlate with hallucination but don't cause it

### 6.2 What Might Fix Activation Approaches

- **More data**: 1000+ examples may overcome overfitting (Track C)
- **Per-token training**: train on individual token activations (5000+ examples from 100 QA pairs) (Track B)
- **Concept-specific vectors**: targeted prompting (Anthropic approach) instead of generic mean-difference (Track A)

---

## 7. Related Work

- **Energy-Based Transformers** (arxiv 2507.02092): EBTs achieve 35% faster scaling and 29% improvement via System 2 thinking. Validates energy-based inference at transformer scale.
- **Autoregressive Models as EBMs** (arxiv 2512.15605): Establishes bijection between ARMs and EBMs. Every LLM is already an EBM — the logprobs ARE the energy.
- **Semantic Energy** (arxiv 2508.14496): Detects hallucination via negative logits. Our Experiment 13 confirms this approach works (+10%).
- **Emotion Concept Vectors** (Anthropic 2025): Concept-specific activation vectors are causally effective for steering. Generic directions are not. Consistent with our Principle #7.
- **Trace2Skill** (arxiv 2603.25158): Parallel analyst sub-agents extract structured lessons from execution traces. Integrated into Carnot's autoresearch as the Trace2Skill learning layer.

---

## 8. Framework Summary

| Component | Files | Tests | Status |
|-----------|-------|-------|--------|
| Core EBM (Rust + JAX) | 12 crates + 8 Python modules | 100 Rust + 1053 Python | Alpha |
| Constraint verification | SAT, coloring, code, property tests | Full coverage | Alpha |
| LLM-EBM inference | Composite scorer, iterative refinement | Full coverage | Alpha |
| Learned verifiers | NCE/SNL/optimization training | Full coverage | Research |
| Activation analysis | Extraction, direction, steering, concepts | Full coverage | Research |
| GPU compute | wgpu Vulkan + WebGPU gateway | 4 Rust tests | Experimental |
| Autoresearch | 50-iteration self-improvement, Trace2Skill | Full coverage | Alpha |
| Research conductor | Autonomous research via Claude Code | N/A | Experimental |

---

## 9. Reproduction

```bash
# Clone and setup
git clone https://github.com/Carnot-EBM/carnot-ebm
cd carnot
pip install -e ".[dev]"

# Run experiments
make up                          # Start Claude API bridge + WebGPU gateway
python scripts/experiment_logprob_rejection.py        # Experiment 13
python scripts/experiment_composite_energy_rejection.py  # Experiment 14
python scripts/experiment_real_hallucination_detection.py # Experiment 8
python scripts/collect_truthfulqa_activations.py           # Experiment 21
python scripts/collect_qa_activations_qwen35.py            # Experiment 22
python scripts/train_per_token_ebm_combined.py             # Train per-token EBM
python scripts/experiment_23_ebm_rejection.py              # Experiment 23
python scripts/experiment_24_layer_probing.py              # Experiment 24
python scripts/experiment_25_no_thinking.py                # Experiment 25

# Verify code with CLI
carnot verify examples/math_funcs.py --func gcd --test "(12,8):4"

# Run full test suite
make test

# Start autonomous research
make research-loop
```

---

## 10. Conclusion

Across 25 experiments on two model families (Qwen3-0.6B base and Qwen3.5-0.8B instruction-tuned), the most effective LLM+EBM combination is surprisingly simple: use the model's own logprobs as energy for candidate selection, combine with structural test execution for code verification, and iterate with feedback. This "composite scorer" architecture improves accuracy on both QA (+10%) and code (0% → 30%) tasks, with no training required.

More sophisticated approaches — training EBMs on activations, steering generation via hooks — show detectable signals but fail to improve practical output quality. EBM rejection sampling on adversarial questions (TruthfulQA) actually hurt accuracy by 3-6% (Experiment 23). Multi-layer probing (Experiment 24) confirmed the final transformer layer is optimal, with hallucination signal following a U-curve across layers.

Three compression effects compound against activation-based detection:

1. **Instruction tuning** compresses the signal (84.5% → 67.2%, Experiment 22)
2. **Adversarial questions** make correct/wrong answers indistinguishable (Experiment 23)
3. **Chain-of-thought reasoning** compresses it further — disabling thinking improves detection from 61.3% to 75.5% (+14.2%, Experiment 25)

The thinking mode finding (Principle 10) is our most actionable discovery: for activation-based hallucination detection, **disable chain-of-thought**. The thinking process makes hidden states more uniform across correct and wrong answers, with a 5.8x reduction in energy gap. This creates a fundamental tension: thinking may improve answer quality but makes detection harder.

The framework ships with an MCP server (3 Python code verification tools) and CLI (`carnot verify`) for structural code verification. The composite scorer (logprob + structural tests) is the most mature component and does not require activation analysis. Note: this is alpha-stage research software (v0.1.0), not production-hardened infrastructure.

### 10 Principles Learned

1. Simpler is better in small-data regimes
2. Token-level features > sequence-level (mean-pooling kills signal)
3. The model's own logprobs are the best energy
4. Overfitting is the main enemy when examples < dimensions
5. Extract features from generated tokens, not prompts
6. Different energy signals dominate in different domains
7. Statistical difference ≠ causal influence
8. Instruction tuning compresses the hallucination signal
9. Adversarial questions defeat post-hoc detection
10. Chain-of-thought reasoning compresses the hallucination signal
