# Carnot Experiment Log

Documenting all experiments — what worked, what failed, and what we learned.

## Summary Table

| # | Experiment | Result | Key Learning |
|---|-----------|--------|-------------|
| 1 | SAT verify-and-repair (random) | ✅ +91.6% energy reduction | Gradient repair works on continuous relaxation |
| 2 | Haiku SAT benchmark (20 inst) | ✅ 60% → 80% (+20%) | First real LLM hallucination → EBM repair |
| 3 | Multi-start repair on stubborn | ✅ Fixed instance single-start missed | P2 self-verification adds value on hard cases |
| 4 | Sonnet SAT (15v/50c) | ✅ 100% | Sonnet is too strong for SAT — need harder tasks or weaker model |
| 5 | Sonnet SAT (20v/85c) | ✅ 100% | Sonnet still perfect at phase transition |
| 6 | Sonnet SAT (30v/128c) | ⚠️ 1/3 parsed, correct | Parse failures on long output, not reasoning failures |
| 7 | Code verification (11 tasks) | ✅ 100% (both Sonnet and Haiku) | LLMs ace well-known algorithms — need novel tasks |
| 8 | Real hallucination detection | ✅ 64% detection accuracy | Activation-space direction IS real (energy gap +9.3) |
| 9 | Rejection: linear direction (25 cal) | ❌ 68% → 56% (-12%) | Prompt activations ≠ answer activations |
| 10 | Rejection: linear direction (93 cal) | ⚠️ 60% → 60% (net zero) | 4 fixes + 4 regressions cancel out |
| 11 | Rejection: Gibbs EBM (2048-dim) | ❌ 94% cal → 35% test | Classic overfitting: 42 examples in 2048-dim space |
| 12 | Rejection: PCA + Gibbs (4/8/16/32) | ❌ Best: PCA-8 at -5% | PCA reduces overfitting but mean-pooling is wrong feature |
| 13 | Rejection: logprob-based | ✅ 45% → 55% (+10%) | **Simplest approach wins.** Model's own confidence IS the energy. |
| 14 | Composite: logprob + structural (code) | ✅ 0% → 30% | Structural tests dominate for code; logprobs dominate for QA |
| 15 | Activation steering (in-generation) | ⚠️ 76% → 76% | Zero effect — mean-difference direction doesn't causally drive token choices |
| 16 | Concept steering (6 configs) | ⚠️ 75% → 75% | Zero effect across all layer/alpha combos |
| 17 | Concept-specific vectors (targeted prompting) | ❌ All < 56%, generic 80% | Targeted prompts produce WORSE directions than generic mean-diff |
| 18 | Per-token activation dataset | ✅ 1860 tokens | Saved to safetensors; first per-token dataset |
| 19 | Per-token EBM (1860 tokens, Qwen3-0.6B) | ✅ 71.8% test | **First activation method that generalizes** |
| 20 | Concept steering (expanded) | ❌ 0% change | Confirms #15-16: statistical ≠ causal |
| 21 | Scaled per-token EBM (26,800 tokens, Qwen3-0.6B) | ✅ 84.5% test | More data helps — 71.8% → 84.5% |
| 22 | TruthfulQA + Qwen3.5-0.8B | ⚠️ 67.2% test | Better models have subtler hallucination signatures |
| 23 | EBM rejection sampling (TruthfulQA) | ❌ -3% to -6% | Neither logprob nor EBM rejection helps on adversarial QA with tuned models |
| 24 | Multi-layer probing (Qwen3.5-0.8B) | ⚠️ Final layer best (64%) | Hallucination signal follows U-curve: appears early, compresses mid-network, reconcentrates at final layer |
| 25 | **Thinking vs no-thinking** | **75.5% no-think vs 61.3% think** | **✅ Thinking compresses hallucination signal by 14.2%** |
| 26 | Cross-model EBM transfer | ❌ 49.8% cross vs 86.2% self | Hallucination representations are model-specific; no universal detector |
| 27 | Upstream detection (question-level) | ⚠️ 62.6% mean (72.1% best) | Weak signal — question reps partially predict hallucination but much weaker than per-token |
| 28 | **Multi-layer concatenation** | **81.3% (3 layers) vs 75.5% (1 layer)** | **✅ Concatenating layers 4+12+24 improves by 5.8%** |
| 29 | Layer gating vs concat | All-concat 79.2%, 3-layer 78.3%, gating 62.8% | 3-layer concat is the sweet spot; learned gating fails with limited data |
| 30 | Temperature diversity | 78.7% best single-temp, 70.2% combined | ❌ Mixing temperatures hurts; more questions > more temperatures |
| 31 | Multi-dataset (TruthfulQA+MMLU+SimpleQA+HaluEval) | 70.8% combined vs 75.5% TruthfulQA-only | ❌ Mixing domains hurts; domain-specific training wins |
| 32 | Weight structure profiling (dense + MoE) | Qwen3.5-35B experts overlap 0.008 vs Mixtral 0.997 | ✅ MoE architectures differ fundamentally in expert specialization |
| — | Scaling curve analysis | 75.5% → 88.5% (0.8B → 27B Qwen3.5) | ✅ EBM accuracy scales with model size; IT tax shrinks at larger scale |
| 34 | MoE routing entropy as hallucination signal | Router hooks didn't capture routing decisions | ⚠️ Qwen3.5-35B router not accessible via standard forward hooks |
| — | Practical deployment test | EBM agreement 50%, energy gap inverted | ❌ Confident hallucinations have LOW energy — indistinguishable from correct |

## Detailed Experiment Notes

| 17 | Concept-specific vectors (targeted prompting) | ❌ All < 56%, generic 80% | Targeted prompts produce WORSE directions; generic mean-diff with generated-token activations reaches 80%/0.945 AUROC |
| 18 | Per-token activation dataset | ✅ 1860 tokens | Saved to safetensors; fewer than 5000 target but usable |

### Experiment 9: Linear Hallucination Direction — Wrong Features
**Date:** 2026-04-05
**Hypothesis:** Mean-pooled activations from correct vs hallucinated answers define a linear separation direction. Use it to rank rejection sampling candidates.
**Result:** 68% → 56% (-12%)
**Root cause:** Extracted activations from the INPUT PROMPT (model(**inputs)), not from the GENERATED TOKENS. The prompt activations are identical across candidates since the prompt is the same.
**Fix applied:** Extract from generated tokens (model(full_sequence), slice to prompt_len:).

### Experiment 10: Scaled Calibration — Confidence ≠ Correctness
**Date:** 2026-04-05
**Hypothesis:** More calibration data (93 vs 25) will give a better direction.
**Result:** 60% → 60% (net zero: 4 fixes, 4 regressions)
**Learning:** The mean-difference direction captures confidence (uncertain vs certain), not correctness (right vs wrong). A confident hallucination looks identical to a confident correct answer. The direction can't distinguish them.

### Experiment 11: Nonlinear Gibbs EBM — Overfitting
**Date:** 2026-04-05
**Hypothesis:** A nonlinear Gibbs model [2048→256→64→1] trained via NCE can learn the curved boundary between correct and hallucinated activation patterns.
**Result:** 94% calibration accuracy, 35% test accuracy (-25%)
**Learning:** 42 examples in 2048-dim space → severe overfitting. The model memorizes calibration data perfectly but learns nothing generalizable. Need either: much more data (500+), dimensionality reduction, or regularization.

### Experiment 12: PCA + Gibbs — Mean-Pooling Is Wrong
**Date:** 2026-04-05
**Hypothesis:** PCA to 8-32 dims before Gibbs training prevents overfitting.
**Result:** PCA-8 best at -5%, all dims still regress
**Learning:** The problem isn't dimensionality — it's that mean-pooled activations lose token-level signal. The model's state while generating "Paris" vs "garbage" differs at individual token positions, but mean-pooling washes this out. Token-level features needed.

### Experiment 13: Logprob Selection — Simplicity Wins
**Date:** 2026-04-05
**Hypothesis:** Per-token log-probabilities (the model's own confidence per token) are a better energy signal than activation mean-pooling.
**Result:** 45% → 55% (+10%), 4 fixes, 2 regressions, net +2
**Learning:** The simplest approach works best. No calibration, no training, no PCA needed. The model's logprobs at each generated token ARE the right energy function. Higher total logprob = more confident = more likely correct. This validates the semantic energy paper (arxiv 2508.14496).

| 14 | Composite: logprob + structural (code) | ✅ 0% → 30% | Structural tests dominate for code; logprobs dominate for QA |
| 15 | Activation steering (in-generation) | ⚠️ 76% → 76% | Zero effect — mean-difference direction doesn't causally drive token choices |
| 16 | Concept steering (6 configs) | ⚠️ 75% → 75% | Zero effect across all layer/alpha combos — needs concept-specific prompting |

### Experiment 21: Scaled Per-Token EBM (26,800 tokens, Qwen3-0.6B)
**Date:** 2026-04-06
**Hypothesis:** More per-token data (26,800 vs 1,860) from QA dataset improves EBM accuracy past 71.8%.
**Result:** 84.5% test accuracy (+12.7% over experiment 19)
**Learning:** More data directly improves generalization. Architecture search (512-128, 256-128-32, etc.) shows all models plateau at ~84% — the bottleneck is data diversity, not model capacity. The [256, 64] silu architecture remains optimal.

### Experiment 22: TruthfulQA + Qwen3.5-0.8B (52,296 tokens)
**Date:** 2026-04-06
**Hypothesis:** Adding TruthfulQA (adversarial QA) from an instruction-tuned model (Qwen3.5-0.8B) improves per-token EBM with more diverse data.
**Results:**
- Qwen3-0.6B QA only: 84.5% (baseline)
- Qwen3.5-0.8B TruthfulQA only: 63.1%
- Qwen3.5-0.8B QA + TruthfulQA merged: 67.2%
- Mixed models (Qwen3-0.6B + Qwen3.5-0.8B): 70.5% (worse than either alone)

**Key Learnings:**
1. **Better models have subtler hallucination signatures.** Qwen3.5-0.8B (instruction-tuned with thinking) produces more uniform representations across correct/wrong answers than base Qwen3-0.6B. The hallucination signal exists (67% > 50%) but is weaker.
2. **Never mix activations from different models.** Different models occupy different representation spaces even with the same hidden dimension (1024). Combining them degrades performance.
3. **TruthfulQA is genuinely harder.** Adversarial questions designed to elicit hallucination produce less distinguishable activations. The model's 53% accuracy on TruthfulQA (vs 57% on QA) means it's nearly guessing, so correct/wrong activations overlap more.
4. **Principle 8: Instruction tuning compresses the hallucination signal.** Base models have bigger activation gaps between right/wrong because they're more "confused" when wrong. RLHF/instruction tuning makes the model sound confident even when wrong — this is literally what makes hallucination dangerous and what makes detection harder.

### Experiment 23: EBM Rejection Sampling on TruthfulQA
**Date:** 2026-04-06
**Hypothesis:** Combining per-token EBM energy with logprob scores improves candidate selection on adversarial QA.
**Results:**
- Greedy: 50% (baseline)
- Logprob-only: 49% (-1%)
- EBM-only: 44% (-6%)
- Composite: 47% (-3%)

**Learning:** Rejection sampling that works on simple QA (+10% with Qwen3-0.6B) fails on adversarial QA with instruction-tuned models. TruthfulQA questions are designed so all answer options sound equally plausible. The model's logprobs and activation patterns don't distinguish correct from confident-but-wrong answers on adversarial inputs. This reinforces Principle 8.

### Experiment 24: Multi-Layer Hallucination Probing
**Date:** 2026-04-06
**Hypothesis:** Intermediate transformer layers retain hallucination signal that gets compressed in the final layer.
**Results (Qwen3.5-0.8B, 200 TruthfulQA questions, ~3.6K tokens/layer):**

| Layer | Test Accuracy |
|-------|--------------|
| 0 | 52.7% (near chance) |
| 4 | 60.0% |
| 8 | 56.8% |
| 12 | 57.8% |
| 16 | 55.8% |
| 20 | 53.0% |
| 24 | 64.0% (best) |

**Learning:**
1. The final layer IS the best single layer — no intermediate layer rescues the signal.
2. Hallucination signal follows a **U-curve**: appears in early layers (4: 60%), gets compressed in middle layers (16-20: ~54%), then reconcentrates at the final layer (24: 64%).
3. This pattern suggests the model's residual stream accumulates hallucination-relevant features gradually, with the most meaningful representation at the output.
4. **Principle 9: Adversarial questions defeat post-hoc detection.** When questions are designed to make LLMs hallucinate, the model's internal representations during correct and incorrect generation become indistinguishable. Detection must move upstream — to the question itself, not the answer.

### Experiment 25: Thinking vs No-Thinking (Qwen3.5-0.8B)
**Date:** 2026-04-06
**Hypothesis:** Thinking mode (chain-of-thought before answering) compresses hallucination signal by making the model's hidden states more uniform.
**Results (200 TruthfulQA questions):**

| Mode | Model Accuracy | EBM Test Accuracy | Energy Gap |
|------|---------------|-------------------|-----------|
| With thinking | 54% | 61.3% | 0.4206 |
| **Without thinking** | **54%** | **75.5%** | **2.4248** |

**Key Learnings:**
1. **Thinking compresses hallucination signal by 14.2%.** The chain-of-thought process makes hidden states more uniform, washing out the differences between correct and wrong answers.
2. **Energy gap is 5.8x larger without thinking** (2.4248 vs 0.4206). The EBM has much more to work with.
3. **Model accuracy is the same** (54% either way on TruthfulQA). Thinking doesn't help on adversarial questions — it just makes detection harder.
4. **Principle 10: Chain-of-thought reasoning compresses the hallucination signal.** For activation-based detection, disable thinking. For answer quality, enable it. These goals are in tension.

### Experiment 26: Cross-Model EBM Transfer
**Date:** 2026-04-07
**Hypothesis:** If the hallucination "direction" in activation space is universal, an EBM trained on one model should detect hallucinations in another model with the same hidden dimension.
**Results:**

| Dimension | Trained On | Tested On | Accuracy |
|-----------|-----------|-----------|----------|
| 1024 | LFM2.5-350M | Self | 90.4% |
| 1024 | LFM2.5-350M | Qwen3.5-0.8B | 44.7% |
| 1024 | Qwen3.5-0.8B | LFM2.5-350M | 50.4% |
| 1536 | Gemma4-E2B | Self | 89.5% |
| 1536 | Gemma4-E2B | Gemma4-E2B-it | 54.5% |
| 1536 | Gemma4-E2B-it | Gemma4-E2B | 51.0% |
| 2048 | Bonsai-1.7B | Self | 81.7% |
| 2048 | Bonsai-1.7B | Qwen3.5-2B | 50.7% |
| 2048 | Qwen3.5-2B | Self | 89.8% |
| 2048 | Qwen3.5-2B | LFM2.5-1.2B | 47.8% |

**Mean self-accuracy:** 86.2%. **Mean cross-accuracy:** 49.8%. **Transfer rate:** 57.8%.

**Key Learnings:**
1. **Cross-model transfer is at chance (~50%).** EBMs trained on one model are useless for another, even with identical hidden dimensions.
2. **Even same-architecture pairs don't transfer.** Gemma4-E2B base → Gemma4-E2B-it is only 54.5%. Instruction tuning doesn't just compress the signal — it *rotates* the representation space.
3. **Cross-family transfer is no worse than within-family.** Bonsai (Qwen3-based) → Qwen3.5-2B is 50.7%, same as random. Shared architecture heritage doesn't help.
4. **Principle 11: Hallucination representations are model-specific.** Each model must have its own trained EBM. There is no universal hallucination detector via activation analysis. The good news: training takes only ~5 minutes per model.

## Key Principles Learned

1. **Simpler is better for small-data regimes.** Linear projections > nonlinear models when you have <100 training examples.

2. **Token-level features > sequence-level.** Mean-pooling destroys the signal. Per-token logprobs preserve it.

3. **The model's own confidence is the best energy.** No external EBM needed for rejection sampling — the LLM's logprobs are already an energy function (as the ARM↔EBM bijection paper proved).

4. **Overfitting is the main enemy.** Every approach that trains on calibration data overfits when examples < dimensions.

5. **Extract features from the RIGHT part of the computation.** Prompt activations ≠ answer activations. The signal is in the GENERATED tokens, not the input.

6. **Different energy signals dominate in different domains.** Logprobs (LLM confidence) work best for QA/factual. Structural tests (EBM execution) work best for code. The composite combines both and is never worse than either alone.

7. **Statistical difference ≠ causal influence.** A direction that separates correct from hallucinated activations (experiment 8: 64% detection) does NOT necessarily steer the model when injected during generation (experiments 15-16: 0% effect). Effective steering requires concept-specific vectors found via targeted prompting (Anthropic's approach), not generic contrastive means.

8. **Instruction tuning compresses the hallucination signal.** Base models (Qwen3-0.6B: 84.5%) have larger activation gaps between correct/wrong answers than instruction-tuned models (Qwen3.5-0.8B: 67.2%). RLHF makes models sound confident even when wrong — exactly what makes hallucination dangerous and detection harder. This has profound implications: the models most in need of hallucination detection are the hardest to detect on.

9. **Adversarial questions defeat post-hoc detection.** On TruthfulQA (designed to elicit hallucination), neither logprob rejection (-1%), EBM rejection (-6%), nor composite (-3%) improves over greedy. Multi-layer probing confirms: the final layer is already best (64%), no hidden layer rescues the signal. Detection must move upstream — to recognizing adversarial question patterns, not analyzing answer activations.

10. **Chain-of-thought reasoning compresses the hallucination signal.** Disabling thinking in Qwen3.5-0.8B improves EBM detection from 61.3% → 75.5% (+14.2%) with a 5.8x larger energy gap. Thinking makes hidden states more uniform across correct/wrong answers. For activation-based detection, disable thinking. For answer quality, enable it. These goals are in tension — a key design constraint for production hallucination detection systems.

11. **Hallucination representations are model-specific, not universal.** Cross-model EBM transfer is at chance (~50%) even between models with identical hidden dimensions and shared architecture heritage. Instruction tuning doesn't just compress the signal — it rotates the representation space. Each target model needs its own trained EBM. Training takes ~5 minutes per model.

12. **Multi-layer concatenation improves detection by ~6%.** Concatenating activations from layers 4+12+24 (early + middle + late) achieves 81.3% vs 75.5% for the final layer alone. The sweet spot is 3 layers; adding all 7 sampled layers (80.0%) slightly underperforms due to noise. Early layers carry complementary hallucination signal that the final layer compresses.

13. **EBM detection is domain-specific, not universal.** Mixing datasets (experiment 31) and mixing temperatures (experiment 30) both hurt accuracy. The EBM learns domain-specific correct/wrong activation patterns. Train on your target domain, not on "everything." Same-model cross-domain transfer is poor (57.2%).

14. **Upstream (question-level) detection is weak but model-dependent.** The model's representation of the question partially predicts hallucination (62.6% mean, 72.1% best on Qwen3.5-0.8B) but is much weaker than per-token post-hoc detection (75-78%). Instruction-tuned models (Qwen: 72.1%) show more signal than base models (LFM: 55.9%), suggesting IT models develop internal "uncertainty representations" that base models lack.
