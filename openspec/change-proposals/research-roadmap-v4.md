> **SUPERSEDED** by research-roadmap-v6.md. Activation-based approaches proven insufficient by experiments 36-38.

# Carnot Research Roadmap v4: From Detector to Reasoning Engine

**Created:** 2026-04-06
**Status:** Active
**Supersedes:** research-roadmap-v3.md

## Where We Are

25 experiments established a clear hierarchy of what works for EBM-based hallucination detection:

| What Works | What Doesn't |
|-----------|-------------|
| Logprob rejection sampling (+10% QA) | Activation steering (0% causal effect) |
| Composite scoring (0% → 30% code) | Mean-pooled activation features (overfit) |
| Per-token activation EBM (84.5% base) | EBM rejection on adversarial QA (-3% to -6%) |
| SAT gradient repair (60% → 80%) | Concept-specific vectors (worse than generic) |

**Key constraints discovered:**
- 84.5% accuracy ceiling is data-bound, not architecture-bound
- Instruction tuning compresses the signal (84.5% → 67.2%)
- Chain-of-thought compresses it further (75.5% → 61.3%)
- Adversarial questions defeat post-hoc detection entirely

**Core principle: Carnot never modifies the target LLM's weights.** We introspect frozen models — reading logprobs, extracting activations, executing generated output against constraints. "EBM training" means training a small classifier on features from frozen LLMs.

## The Long-Term Vision

**The LLM is an I/O device. The EBM is the reasoning engine.**

Today's LLMs are language interfaces — they map human language to and from an internal representation space. The actual "thinking" (constraint satisfaction, logical consistency, factual verification) should happen in an energy landscape where correctness is mathematically verifiable, not statistically approximated.

The path:
1. **Now:** EBM as detector — introspect frozen LLMs to catch hallucinations after generation
2. **Next:** EBM as guide — steer generation in real-time via energy gradients in activation space
3. **Then:** EBM as reasoner — the EBM operates in semantic space, finds minimum-energy configurations, and the LLM translates the result to/from human language
4. **Eventually:** EBM on hardware — the energy minimization runs on thermodynamic sampling hardware (Extropic TSU), with the LLM as a thin language interface

The key insight enabling this: **every open-source model has already learned something about the structure of language and knowledge.** Their frozen weights encode this knowledge in their activation patterns. We can extract and compose this knowledge into an EBM rapidly and efficiently — no trillion-token pretraining required.

---

## Tier 1: Improve the Detector

**Goal:** Push per-token EBM accuracy past 84.5% and make it robust on instruction-tuned models.
**Timeline:** Near-term (1-3 sessions each)
**Risk:** Low — incremental extensions of proven approaches

### Experiment 26: Multi-Layer Concatenated Features

**Hypothesis:** Concatenating activations from the 2-3 most informative layers (per Experiment 24's U-curve: layers 4 and 24) gives the EBM a richer input signal than the final layer alone.

**Method:**
1. Extract activations from layers 4, 16, and 24 of Qwen3-0.6B on the existing 26,800-token QA dataset
2. Concatenate into a 3072-dim feature vector per token (3 × 1024)
3. Train Gibbs EBM [3072→512→128→1] via NCE
4. Compare against single-layer baseline (84.5%)

**Success:** >86% test accuracy. If the ceiling doesn't move, the bottleneck is confirmed as data diversity, not feature richness.

**Deliverable:** `scripts/experiment_26_multi_layer.py`

### Experiment 27: Domain-Diverse Training Data

**Hypothesis:** The 84.5% ceiling comes from training on a single domain (factual QA). Hallucination patterns differ by domain — a multi-domain EBM is more robust.

**Method:**
1. Collect per-token activations from Qwen3-0.6B on 4 domains: factual QA (existing), math/arithmetic, science claims, code explanation
2. Generate 5,000+ tokens per domain using standard datasets (GSM8K, SciQ, code docstrings)
3. Train per-domain EBMs and one combined multi-domain EBM
4. Evaluate cross-domain transfer: does training on math help detect QA hallucinations?

**Success:** Multi-domain EBM >87% on QA (beating single-domain 84.5%) and >70% on each individual domain.

**Deliverable:** `scripts/experiment_27_multi_domain.py`

### Experiment 28: Temporal/Sequential Token Features

**Hypothesis:** Hallucinations have a trajectory signature — a confident start followed by increasing uncertainty. Per-token features miss this because each token is scored independently.

**Method:**
1. For each token position t, construct a feature vector: [activation(t), activation(t) - activation(t-1), running_mean(t-3:t)]
2. This gives the EBM temporal context without mean-pooling
3. Train on sliding windows of the existing dataset
4. Compare against static per-token features

**Success:** >86% test accuracy with temporal features. Look especially for improvement on tokens deep into hallucinated sequences (where the trajectory signature should be strongest).

**Deliverable:** `scripts/experiment_28_temporal_features.py`

### Experiment 29: Larger Base Model Activations

**Hypothesis:** Larger base models (Qwen3-4B, 8B) have higher-dimensional activations with more separable hallucination representations.

**Method:**
1. Run Qwen3-4B (or largest model that fits in 67GB unified memory) on the same QA dataset
2. Extract per-token activations from the final layer
3. Train per-token EBM, compare accuracy against Qwen3-0.6B baseline

**Success:** >87% test accuracy. If larger models show more separable representations, this informs the Tier 2 strategy of which frozen models to distill from.

**Deliverable:** `scripts/experiment_29_larger_model.py`

---

## Tier 2: EBM as Reasoning Guide

**Goal:** Move from post-hoc detection to real-time energy-guided generation. Use multiple frozen models as "knowledge donors" to train a compact EBM that captures their collective understanding.
**Timeline:** Medium-term (multiple sessions per experiment)
**Risk:** Medium — requires new architecture patterns, not just more data

### Experiment 30: Cross-Model Activation Consensus

**Hypothesis:** When multiple frozen models agree internally on an answer, it's more likely correct. The pattern of agreement/disagreement across models is a stronger energy signal than any single model's activations.

**Method:**
1. Run the same 300 QA pairs through 3-4 frozen models: Qwen3-0.6B, Phi-3-mini, Gemma-2B, Llama-3.2-1B (all small enough to run locally)
2. For each question, extract per-token activations from each model's final layer
3. Compute inter-model agreement features: cosine similarity of activation trajectories, variance across models, PCA of the cross-model activation matrix
4. Train an EBM on these consensus features

**Why this matters:** This is the first step toward "distill frozen weights into an EBM." Instead of learning from one model's internal patterns, we learn from the *shared structure* across models. What all models agree on is more likely to reflect genuine knowledge rather than model-specific artifacts.

**Success:** Consensus EBM >80% accuracy (beats any single instruction-tuned model's 67.2%). If cross-model agreement is a better signal, this validates the multi-model distillation approach.

**Deliverable:** `scripts/experiment_30_cross_model_consensus.py`

### Experiment 31: Energy-Guided Decoding

**Hypothesis:** Instead of generate-then-verify, we can guide token selection during generation by using the EBM's energy as a supplementary scoring signal alongside the LLM's logprobs.

**Method:**
1. At each generation step, compute top-k candidate tokens from the LLM's logits
2. For each candidate, run a forward pass to get the hidden state that would result from selecting that token
3. Score each candidate: composite = -w₁·logprob + w₂·EBM_energy(hidden_state)
4. Select the token with lowest composite energy
5. Compare against greedy decoding and pure logprob rejection sampling

**Why this matters:** This is the transition from post-hoc to in-generation. The EBM isn't steering activations (which failed in Experiments 15-16) — it's influencing token *selection* by predicting which tokens will lead to high-energy (hallucinated) continuations.

**Success:** >60% accuracy on QA (beats greedy 45% and logprob-only 55%). Must not be catastrophically slow (< 5x overhead vs. greedy).

**Deliverable:** `python/carnot/inference/energy_guided_decoding.py`

### Experiment 32: Iterative Refinement in Embedding Space

**Hypothesis:** For structured problems, we can map LLM output to a continuous embedding, run gradient descent in that space using the EBM's energy, and decode back to text.

**Method:**
1. Start with a structured domain: math word problems (GSM8K) where answers are numerical
2. LLM generates an initial answer → extract the final-layer embedding of the answer tokens
3. Define energy: E(embedding) = EBM_score + constraint_energy (e.g., the answer must be a valid number, intermediate steps must be consistent)
4. Run gradient descent on the embedding: embedding' = embedding - η·∇E
5. Find the nearest valid token sequence to the refined embedding (nearest-neighbor in the model's vocabulary space)
6. Compare the refined answer against the original

**Why this matters:** This is the first test of "EBM as reasoning engine." The LLM provides the initial embedding and the decode-back-to-text step. The actual reasoning (finding the right answer) happens in energy space.

**Success:** Refined answers are more often correct than original LLM answers on GSM8K. Even marginal improvement validates the architecture.

**Deliverable:** `scripts/experiment_32_embedding_refinement.py`

### Experiment 33: Knowledge Distillation as Energy

**Hypothesis:** The KL divergence between a frozen model's output distribution and a reference distribution is a meaningful energy term. Multiple frozen models = multiple composable energy terms.

**Method:**
1. For a given input, compute the output distribution (logits) from N frozen models
2. Define per-model energy: E_i(answer) = KL(P_model_i(answer|input) || P_uniform) — how "surprised" each model is by the answer
3. Composite energy: E_total = Σ w_i · E_i — weighted sum across models
4. The weights w_i can be learned or set proportional to model size/quality
5. Test as a candidate selector: generate N answers, score by composite energy, select lowest

**Why this matters:** This directly implements "use frozen weights from open-source models to train our EBM." Each model contributes an energy term. The EBM is the composition. No model is modified. Training the EBM = learning the weights w_i, which is a tiny optimization problem.

**Success:** Composite multi-model energy outperforms any single model's logprobs for candidate selection.

**Deliverable:** `python/carnot/inference/multi_model_energy.py`

---

## Tier 3: Standalone EBM with LLM as I/O

**Goal:** Build an EBM whose energy landscape encodes the collective knowledge of many frozen models. The LLM becomes a thin translation layer between human language and the EBM's semantic space.
**Timeline:** Long-term (multi-week research program)
**Risk:** High — requires breakthroughs in representation learning. But the potential payoff is enormous: a reasoning system that is verifiable, composable, and hardware-acceleratable.

### 3a: Universal Activation Encoder

**Concept:** Train a shared encoder that maps activations from *any* frozen model into a common semantic space. In this space, the same concept has the same representation regardless of which model produced it.

**Why:** Currently, activations from different models can't be mixed (Experiment 22). A universal encoder removes this limitation, allowing the EBM to learn from all models simultaneously.

**Approach:**
1. Collect (input, output, activations) triples from many models on the same inputs
2. Train a contrastive encoder: activations from different models on the SAME input should map to nearby points; activations on DIFFERENT inputs should map far apart
3. The encoder's output space is the EBM's operating space

**Key challenge:** Different models have different hidden dimensions, layer counts, and internal representations. The encoder must handle this heterogeneity. Possible approaches: projection to a fixed dimension, attention over variable-length layer sequences, or using only the final layer (which Experiment 24 showed is most informative).

**Deliverable:** `python/carnot/embeddings/universal_encoder.py`

### 3b: EBM Energy Landscape from Frozen Weights

**Concept:** Train an EBM whose energy function over the universal semantic space captures what all frozen models collectively "know." Low energy = all models agree this is correct. High energy = models disagree or are uncertain.

**Why:** This is the core of the vision. The EBM becomes a standalone knowledge store that doesn't depend on any single LLM. It can be queried, composed with domain constraints, and optimized over — all properties LLMs lack.

**Approach:**
1. Use the universal encoder (3a) to project activations from N models into shared space
2. For each input, compute the distribution of encodings across models
3. Train the EBM: low energy where models agree (tight cluster), high energy where they disagree (dispersed)
4. The resulting energy landscape is a map of "what is known" — verified by cross-model consensus
5. For reasoning: gradient descent in this space finds the point of maximum consensus

**Key challenge:** The energy landscape must be smooth enough for gradient descent but expressive enough to capture complex knowledge. The Boltzmann tier (deep residual) is designed for this, but may need architectural innovations (e.g., attention over energy terms, hierarchical energy decomposition).

**Deliverable:** `python/carnot/models/consensus_ebm.py`

### 3c: LLM as Language Interface

**Concept:** The LLM's role reduces to two operations:
1. **Encode:** Human question → semantic embedding (via frozen LLM forward pass)
2. **Decode:** Semantic embedding → human-readable answer (via frozen LLM generation conditioned on the embedding)

Between encode and decode, all reasoning happens in energy space:
1. Encode the question
2. EBM computes energy gradient at the encoded point
3. Gradient descent navigates to the minimum-energy answer configuration
4. Decode the result back to language

**Why:** This separates the "language competence" of LLMs (which they're good at) from the "reasoning competence" (which they're bad at). The LLM handles syntax; the EBM handles semantics.

**Approach:**
1. Start with a constrained domain (math, logic, code) where we can verify the decoded answer
2. Use the frozen LLM's encoder (hidden states from the prompt) as the "question embedding"
3. Define the target: find an embedding in the same space that, when decoded, satisfies all constraints
4. Run gradient descent on the EBM's energy + constraint energy
5. Decode via constrained generation: force the LLM to generate text consistent with the refined embedding

**Key challenge:** The decode step. Moving an embedding via gradient descent may land in a region the LLM can't decode coherently. Possible solutions: project back to the LLM's learned manifold after each gradient step, or use the LLM's own decoder as a regularizer (energy += distance from the LLM's manifold).

**Deliverable:** `python/carnot/inference/ebm_reasoning_engine.py`

### 3d: Hardware Path (Extropic TSU)

**Concept:** The Ising tier's quadratic energy E(x) = -½xᵀJx - bᵀx maps directly to the coupling matrices of physical Boltzmann machines. Once the EBM is the reasoning engine, energy minimization can run on thermodynamic hardware at potentially 10,000x efficiency.

**Prerequisites:** Tiers 1-3 must demonstrate that the EBM energy landscape captures useful knowledge. The hardware path is only worthwhile if the software EBM works.

**Approach:**
1. Distill the full EBM (Boltzmann/Gibbs tier) into an Ising-tier approximation
2. Export the Ising coupling matrix J and bias b as hardware-compatible format
3. Interface with Extropic TSU API (when available) for native energy minimization
4. Compare: software EBM inference time vs. hardware TSU inference time

**Deliverable:** `python/carnot/hardware/ising_compiler.py`

---

## Dependency Graph

```
Tier 1 (Improve Detector)
  ├── Exp 26: Multi-layer features
  ├── Exp 27: Multi-domain data
  ├── Exp 28: Temporal features
  └── Exp 29: Larger model activations
        │
        ▼
Tier 2 (EBM as Guide)
  ├── Exp 30: Cross-model consensus  ← uses multiple frozen models
  ├── Exp 31: Energy-guided decoding  ← uses improved detector from Tier 1
  ├── Exp 32: Embedding refinement    ← first test of EBM-as-reasoner
  └── Exp 33: KL distillation energy  ← composable multi-model energy
        │
        ▼
Tier 3 (Standalone EBM)
  ├── 3a: Universal activation encoder  ← needs cross-model data from Exp 30
  ├── 3b: Consensus energy landscape    ← needs encoder from 3a
  ├── 3c: LLM as language interface     ← needs energy landscape from 3b
  └── 3d: Hardware compilation           ← needs working EBM from 3b/3c
```

Tier 1 experiments are independent of each other and can run in parallel.
Tier 2 experiments depend on Tier 1 results but are mostly independent of each other.
Tier 3 is sequential: each step builds on the previous.

## Implementation Priority

| Phase | Experiments | Est. Sessions | What We Learn |
|-------|-----------|---------------|---------------|
| Now | 26-28 | 1-2 each | Whether the 84.5% ceiling is breakable with richer features |
| Next | 29-30 | 2-3 each | Whether larger/multiple models provide better signal |
| Then | 31-33 | 3-4 each | Whether energy-guided generation outperforms post-hoc |
| Long-term | 3a-3c | 5+ each | Whether a standalone EBM can reason without an LLM |
| Future | 3d | TBD | Whether hardware acceleration is practical |

## Success Criteria (per tier)

**Tier 1 success:** Per-token EBM accuracy >90% on base models, >80% on instruction-tuned models. If achieved, the detector is production-useful.

**Tier 2 success:** Energy-guided decoding outperforms greedy+rejection on at least one domain. Cross-model consensus is a better energy signal than any single model. If achieved, the EBM is adding reasoning value, not just detecting errors.

**Tier 3 success:** The EBM can find correct answers to structured problems (math, logic) via gradient descent in semantic space, with the LLM only used for encoding/decoding. If achieved, we have demonstrated that reasoning can be separated from language generation.

## Relationship to Carnot's Core Architecture

These experiments use and extend existing Carnot components:

| Component | Current Use | Extended Use |
|-----------|------------|-------------|
| Gibbs model | Per-token hallucination detector | Cross-model consensus EBM, energy-guided decoding |
| Boltzmann model | Deep energy network (research) | Universal energy landscape over semantic space |
| Ising model | Edge/hardware tier | Hardware compilation target for TSU |
| NCE training | Train per-token EBM | Train consensus EBM, universal encoder |
| Verify-and-repair | Post-hoc constraint repair | In-generation energy guidance |
| Composite scorer | Logprob + structural tests | Multi-model KL divergence composition |
| Autoresearch | Architecture/hyperparameter search | Automated experiment execution for Tier 1-2 |

The framework is designed for this progression. The `EnergyFunction` trait/protocol means any new energy function (consensus, KL distillation, universal) plugs into existing samplers, training loops, and verification infrastructure.
