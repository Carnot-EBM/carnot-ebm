# Carnot Research Roadmap v3: What Works, What Failed, What's Next

**Created:** 2026-04-06
**Status:** Active
**Supersedes:** research-roadmap-v2.md

## Experiment Scorecard (16 experiments)

| # | Approach | Result | Verdict |
|---|----------|--------|---------|
| 2 | SAT gradient repair (Haiku) | 60% → 80% | ✅ Works |
| 8 | Activation detection (post-hoc) | 64% detection | ✅ Detection works |
| 9-12 | Activation rejection sampling | -5% to -25% | ❌ Overfits / wrong features |
| 13 | Logprob rejection sampling | 45% → 55% (+10%) | ✅ Works |
| 14 | Composite (logprob + structural) | 0% → 30% on code | ✅ Works |
| 15-16 | Activation steering (in-generation) | 0% change | ❌ No causal effect |

## 7 Principles Learned

1. Simpler is better in small-data regimes
2. Token-level features > sequence-level (mean-pooling kills signal)
3. Model's own logprobs are the best energy
4. Overfitting is the main enemy when examples < dimensions
5. Extract features from generated tokens, not prompt
6. Different energy signals dominate in different domains
7. Statistical difference ≠ causal influence

## What's Next: Four Research Tracks

### Track A: Concept-Specific Vectors via Targeted Prompting

**Priority:** HIGH — most likely to succeed based on Anthropic emotion research
**Why:** Generic mean-difference failed (experiments 9-16). Anthropic showed concept-specific vectors (found via targeted prompting) ARE causally effective. We built concept_vectors.py but never ran it with actual concept-specific model generations.

#### A1: Generate concept-specific text from the model

**What:** Instead of correct-vs-wrong means, prompt the model to generate text in specific cognitive modes:
- "I am certain that..." → extract "certainty" activations
- "I'm not sure, but I think..." → extract "uncertainty" activations  
- "Actually, let me make something up..." → extract "confabulation" activations
- "Let me reason step by step..." → extract "reasoning" activations
- "As everyone knows..." → extract "memorization" activations

**Deliverable:** `scripts/experiment_concept_specific_vectors.py`
**Success metric:** Concept vectors are more orthogonal than the generic direction, and at least one concept (confabulation) separates correct from hallucinated better than the generic direction.

#### A2: Steer with confabulation-specific vector

**What:** Having identified the confabulation vector via targeted prompting, subtract it during generation (same hook mechanism as experiment 15, but with the RIGHT direction).

**Deliverable:** `scripts/experiment_concept_specific_steering.py`
**Success metric:** Accuracy improves (any positive delta beats the 0% from experiment 15-16).

#### A3: Monitor confabulation energy in real-time

**What:** During generation, compute the projection of each token's hidden state onto the confabulation vector. If it spikes, flag that token. This is per-token hallucination detection — not mean-pooled.

**Deliverable:** `python/carnot/embeddings/realtime_monitor.py`
**Success metric:** Confabulation energy is higher on tokens where the model is wrong vs right (per-token, not per-sequence).

### Track B: Per-Token Activation EBM

**Priority:** HIGH — addresses the mean-pooling problem directly
**Why:** Experiments 9-12 failed because mean-pooling activations across the generated sequence destroyed the token-level signal. Logprobs work because they ARE per-token. Train the EBM on individual token activations instead.

#### B1: Collect per-token activation dataset

**What:** For each generated token, save: (token_id, layer_activations, is_correct_token). A token is "correct" if it appears in a verified correct answer at that position. This gives thousands of training examples from just 100 QA pairs.

**Deliverable:** `scripts/collect_token_activations.py`
**Success metric:** Dataset of 5000+ per-token activation examples with labels.

#### B2: Train Gibbs EBM on per-token activations

**What:** Instead of training on mean-pooled sequence activations (which overfit at 42 examples in 2048-dim), train on individual token activations. With 5000+ examples, the Gibbs model should generalize.

**Deliverable:** `scripts/experiment_per_token_ebm.py`
**Success metric:** Test accuracy > 60% (beats the 50% random baseline and the 64% from generic detection).

#### B3: Per-token energy rejection sampling

**What:** For each candidate answer, compute per-token energy. The candidate's score = mean per-token energy. This is like logprobs but from the EBM's perspective — it should capture patterns logprobs miss.

**Deliverable:** `scripts/experiment_per_token_rejection.py`
**Success metric:** Rejection sampling accuracy > 55% (beats or matches logprob-only from experiment 13).

### Track C: Scale Up

**Priority:** MEDIUM — more data may solve the overfitting problem
**Why:** All activation experiments used 25-93 examples. At scale (1000+), even simple approaches may work.

#### C1: Generate 1000+ QA pairs programmatically

**What:** Use a knowledge base (e.g., Wikidata simple questions, SQuAD, TriviaQA) to generate 1000+ QA pairs with known correct answers. Run Qwen3-0.6B on all of them, label as correct/hallucinated.

**Deliverable:** `data/qa_activations_1000.safetensors`
**Success metric:** At least 300+ correct and 300+ hallucinated examples.

#### C2: Retrain activation-based approaches with 1000+ examples

**What:** Re-run experiments 10-12 (linear direction, Gibbs EBM, PCA+Gibbs) with the larger dataset. The overfitting that killed them at 42-93 examples may not occur at 1000+.

**Deliverable:** `scripts/experiment_scaled_activation_rejection.py`
**Success metric:** Any activation-based rejection sampling achieves > 55% (beats experiment 13's +10%).

#### C3: SVD top-k directions

**What:** Instead of one mean-difference direction, compute SVD on the correct-vs-hallucinated activation difference matrix. Use the top 5-10 principal components as a multi-dimensional energy function.

**Deliverable:** `scripts/experiment_svd_topk_rejection.py`
**Success metric:** Multi-direction energy separates better than single direction (higher AUROC).

### Track D: Ship What Works

**Priority:** HIGH — turn research into a usable tool
**Why:** Composite scoring (logprob + structural tests) works today. Iterative refinement with property testing works. These should be packaged as a tool developers use.

#### D1: MCP server for code verification

**What:** Build an MCP server that exposes `verify_code(code, test_cases)` and `refine_code(code, test_cases, max_iterations)` as tools. Claude Code can call these automatically when generating code.

**Deliverable:** `tools/verify-mcp/server.py`
**Success metric:** Claude Code can call the verifier during code generation.

#### D2: CLI tool

**What:** `carnot verify <file>` runs property-based testing + structural verification on a Python file. Reports energy score and specific failures.

**Deliverable:** `scripts/carnot_cli.py`
**Success metric:** Can verify a Python file from the command line.

#### D3: Write up findings

**What:** Document the 16 experiments, 7 principles, and the composite scoring architecture as a technical report or blog post.

**Deliverable:** `docs/technical-report.md`
**Success metric:** Publishable document with clear methodology and reproducible results.

## Implementation Priority

| Phase | Track | Tasks | Est. Sessions |
|-------|-------|-------|---------------|
| Next | A1-A2 | Concept-specific prompting + steering | 1 |
| Next | D1-D2 | Ship MCP server + CLI | 1-2 |
| Then | B1-B3 | Per-token activation EBM | 2-3 |
| Then | C1-C3 | Scale to 1000+ examples | 2-3 |
| Then | D3 | Technical report | 1 |

Total: ~7-10 sessions for the complete roadmap.

## Conductor Tasks

These replace the completed Phase 2.5/3 tasks in the conductor.
