> **SUPERSEDED** by research-roadmap-v6.md. Activation-based approaches proven insufficient by experiments 36-38.

# Carnot Research Roadmap v5: Weight-First EBM — Reading Knowledge from Frozen Parameters

**Created:** 2026-04-07
**Status:** Active
**Supersedes:** research-roadmap-v4.md
**Informed by:** Nemotron 3 Super technical report (NVIDIA, 2026-04-03), experiments 26-31

## The Core Insight: The Weights ARE the Energy Model

Every autoregressive model already IS an energy-based model: P(token | context) = exp(-E(token, context)) / Z. The weights define E(). We don't need to train a separate EBM on labeled hallucination data — we need to learn to **read the energy landscape that the frozen weights already define**.

Our first 31 experiments trained EBM classifiers on labeled (hallucinated / correct) activation data. This works (84.5% on base models) but doesn't scale: it requires domain-specific labels, is model-specific (50% cross-model transfer = chance), and degrades when domains are mixed (70.8% < 75.5% single-domain).

The new direction: **derive hallucination signal directly from frozen weight structure and unlabeled forward passes.** Labeled data becomes a validation tool, not a training dependency.

### What the frozen weights contain (no inference needed)

| Signal | How to Extract | What It Tells You |
|--------|---------------|-------------------|
| Channel magnitude patterns | L2-norm each FC1/FC2 column per expert | What each expert "knows" — its knowledge shape |
| Expert routing matrices | Analyze W_gate partitioning of input space | How the model organizes knowledge into domains |
| Unembedding geometry | SVD / clustering of the output projection matrix | Token similarity structure — what the model considers "close" |
| Weight norm profiles | Per-layer L2 norms, variance across neurons | Confidence distribution across depth |
| Inter-model weight alignment | CKA or SVCCA between models | Shared vs. unique knowledge across models |
| Ising coupling structure | WᵀW for any weight matrix W | Quadratic energy landscape — IS an Ising model |
| Expert specialization overlap | Cosine similarity of expert weight matrices | Redundancy and coverage of knowledge domains |

### What requires forward passes on arbitrary text (no labels needed)

| Signal | How to Extract | What It Tells You |
|--------|---------------|-------------------|
| Logprobs | Model output distribution | Model's own energy/confidence |
| Activation patterns | Hidden states per token | Internal representation trajectory |
| Expert routing patterns | Which experts activate per token | Which knowledge bank the model is accessing |
| MTP head predictions | Auxiliary prediction head outputs | How far ahead the model can "see" |
| Cross-model consensus | Same input → multiple models → compare | Whether knowledge is "real" (shared) or artifact (model-specific) |
| Activation deviation from baseline | Per-token hidden state vs. sequence mean | Whether the model is in familiar vs. unfamiliar territory |

### What labeled data is actually needed for

**Validation only.** A labeled hallucination dataset lets us measure whether our weight-derived features actually predict hallucinations. But the features themselves should come from the weights and unlabeled inference.

---

## Where We Are

31 experiments established what works and what the bottlenecks are:

| What Works | What Doesn't |
|-----------|-------------|
| Logprob rejection (+10% QA) | Activation steering (0% causal effect) |
| Composite scoring (0% → 30% code) | Mean-pooled features (overfit) |
| Per-token activation EBM (84.5% base) | Cross-model transfer (49.8% = chance) |
| Multi-layer concat (81.3%, +5.8%) | Naive domain mixing (70.8% < 75.5%) |
| SAT gradient repair (60% → 80%) | Temperature diversity (hurts) |

**13 principles discovered.** The most important for this roadmap:
- The model's own logprobs ARE the best single energy signal (Principle 3)
- Hallucination representations are model-specific (Principle 11)
- Naive domain mixing hurts (Experiment 31)
- The accuracy ceiling is data-bound at current approach (Principle: 84.5%)

**Core constraint: Carnot never modifies the target LLM's weights.**

## The Long-Term Vision

**The LLM is an I/O device. The EBM is the reasoning engine.**

1. **Now:** Read the energy landscape from frozen weights — detect hallucinations without labeled training data
2. **Next:** Use multi-model weight structure to build a consensus energy field — what all models "agree on" via their weight geometry
3. **Then:** Navigate the consensus energy landscape to find correct answers — the EBM reasons, the LLM translates
4. **Eventually:** Run the energy minimization on thermodynamic hardware (Extropic TSU)

---

## Phase 1: Weight Anatomy (No Labels Needed)

**Goal:** Extract hallucination-predictive features from frozen weights and unlabeled forward passes. Validate against existing labeled datasets but do NOT use labels for training.
**Timeline:** Near-term (1-2 sessions each)
**Risk:** Low-Medium — novel analysis methods, but no training infrastructure needed

### Experiment 32: Weight Structure Profiling

**Hypothesis:** A frozen model's weight matrices contain structural signatures that predict where it will hallucinate — before any inference occurs. Layers with low-rank weight matrices, low-norm neurons, or high weight variance are less "certain" and more prone to hallucination on related inputs.

**Method:**
1. For each layer of Qwen3-0.6B (and Qwen3.5-0.8B for comparison), compute:
   - Per-layer weight matrix rank (effective rank via singular value decay)
   - Per-neuron L2 norm distribution
   - Weight matrix condition number (ratio of largest to smallest singular value)
   - Spectral gap (gap between top singular values)
2. For MoE models (Mixtral-8x7B), additionally compute:
   - Per-expert specialization: L2 distance between each expert's weight matrix and the mean expert
   - Expert overlap: cosine similarity matrix between all expert pairs
   - Router weight clustering: what input regions map to which experts
3. Correlate these static weight features with hallucination rates on existing labeled data (validation, not training)
4. Output: a "weight health map" per model — which layers/experts are confident, which are fragile

**Why this matters:** This is pure weight analysis. Zero inference needed. If weight structure predicts hallucination zones, we can identify model weaknesses before deployment — and target those zones for energy-based monitoring during inference.

**Data needed:** Labeled hallucination data for **validation only** (use existing 52K token dataset)
**Success:** Weight structure features correlate with per-layer hallucination rates (Spearman ρ > 0.5). MoE expert specialization predicts which domains a model will hallucinate on.

**Deliverable:** `scripts/experiment_32_weight_profiling.py`, `python/carnot/introspection/weight_profiler.py`

### Experiment 33: Channel Magnitude Introspection (Nemotron-Inspired)

**Hypothesis:** Nemotron's training revealed that expert FC1/FC2 weights develop structured channel magnitude patterns (Figure 6 in the paper). Low-norm output channels of FC1 align with low-norm input channels of FC2. These patterns encode what each expert has learned. When a token's activation doesn't align with the channel structure of the selected experts, the model is operating outside its trained knowledge.

**Method:**
1. For a frozen MoE model (Mixtral-8x7B or quantized Nemotron), precompute the channel magnitude profile:
   - For each expert: L2 norm of each FC1 output channel, L2 norm of each FC2 input channel
   - Identify the "active channels" (high norm) and "dead channels" (near-zero norm)
   - Map the alignment between FC1 output and FC2 input norms
2. At inference time (unlabeled text), measure:
   - Channel alignment score: cosine similarity between the token's activation and the expert's expected magnitude pattern
   - Dead channel activation: how much signal flows through "dead" channels (should be near zero for well-matched inputs)
   - Expert coherence: does the token activate the same "knowledge pathway" (high-norm channels) as training data for that expert?
3. Validate: do low-alignment tokens correlate with hallucinated tokens in labeled data?

**Data needed:** Unlabeled text for inference + labeled data for **validation only**
**Success:** Channel alignment score < threshold predicts hallucination with >60% precision. Dead channel activation spikes on hallucinated tokens.

**Deliverable:** `scripts/experiment_33_channel_magnitude.py`, `python/carnot/introspection/channel_profiler.py`

### Experiment 34: MoE Routing Entropy as Self-Supervised Energy

**Hypothesis:** When a MoE model is uncertain about which expert to route a token to (high router entropy), it doesn't have a clear "knowledge bank" for that input — a signal of potential hallucination. This requires only a forward pass on unlabeled text.

**Method:**
1. Load Mixtral-8x7B (or any MoE model that fits locally)
2. For each token in arbitrary (unlabeled) input text, extract:
   - Router logits for each MoE layer
   - Top-K expert indices and weights
   - Router entropy: H = -Σ p_i log p_i over expert probabilities
   - Expert consistency: across a sequence, how often the same experts are selected (sliding window)
   - Rare expert frequency: how often an infrequently-used expert is selected
3. Define the routing energy: E_routing(token) = w₁·router_entropy + w₂·expert_inconsistency + w₃·rare_expert_score
4. This is a **self-supervised energy** — no hallucination labels needed to define it
5. Validate: does high routing energy correlate with hallucinated tokens in labeled data?

**Data needed:** Unlabeled text for inference + labeled data for **validation only**
**Success:** Routing energy alone predicts hallucination >60% (better than chance). Combined with logprobs >70%.

**Deliverable:** `scripts/experiment_34_routing_energy.py`, `python/carnot/introspection/routing_extractor.py`

### Experiment 35: Activation Normalization for Domain-Invariant Features

**Hypothesis:** Cross-model and cross-domain transfer fail because raw activations encode domain/model identity, not just hallucination signal. Normalizing activations relative to the model's own per-sequence baseline removes the domain confound, making features domain-invariant without labels.

**Method:**
1. For each input sequence, compute:
   - Mean activation across all tokens in the sequence
   - Per-token deviation: activation(t) - mean(all tokens)
   - Z-score normalization: (activation(t) - mean) / std
   - Whitening: project through the inverse square root of the empirical covariance
2. The EBM input becomes the **relative** activation — the deviation from the model's own "normal" for this input
3. Train on the multi-domain dataset from Experiment 31 and evaluate cross-domain transfer
4. Re-test cross-model transfer (Experiment 26) with normalized features

**Why this matters:** This is the cheapest experiment that could unlock "everything" domain. If hallucination is encoded as deviation from the model's own baseline, normalization makes it domain-invariant. No domain labels needed.

**Data needed:** Unlabeled text for inference + labeled data for **validation and training** (this one still uses labeled data for EBM training, but the normalization itself is unsupervised)
**Success:** Cross-domain accuracy >75% (vs 70.8% unnormalized). Cross-model accuracy >60% (vs 49.8%).

**Deliverable:** `scripts/experiment_35_activation_normalization.py`, `python/carnot/introspection/normalizer.py`

---

## Phase 2: Self-Supervised Energy Composition (Minimal Labels)

**Goal:** Combine weight-derived and forward-pass features into composite energy functions that detect hallucinations without labeled training data. Labels used only for weight tuning and validation.
**Timeline:** Medium-term (2-3 sessions each)

### Experiment 36: Composite Self-Supervised Energy

**Hypothesis:** Each of the weight-derived signals (logprob, routing entropy, channel alignment, activation deviation) captures a different aspect of "the model doesn't know this." Composing them produces a domain-general hallucination detector without any labeled hallucination training data.

**Method:**
1. For each token, compute the full feature vector (all from unlabeled forward passes):
   - Logprob (model's own confidence)
   - Router entropy (from Exp 34)
   - Channel alignment score (from Exp 33)
   - Activation deviation from sequence mean (from Exp 35)
   - Weight layer health score at the layer where this token's activation is most extreme (from Exp 32)
2. Define the composite energy: E_composite = Σ w_i · feature_i
3. The weights w_i can be:
   - **Fully unsupervised:** Set w_i = 1 (equal weight) or w_i proportional to feature variance
   - **Lightly supervised:** Optimize w_i on a small labeled calibration set (100-500 labeled tokens)
   - **Fully supervised baseline:** Train a full EBM (our existing approach) for comparison
4. Evaluate all three on the full multi-domain dataset

**Why this matters:** This is the test of the weight-first thesis. If the unsupervised composite energy performs within 10% of the fully supervised EBM, the labeled data dependency is broken. The "everything" domain becomes achievable because no domain-specific labels are needed.

**Data needed:** Unlabeled text for all features + small calibration set (100-500 labels) + full labeled set for **validation**
**Success:** Unsupervised composite >70% on multi-domain (vs 70.8% labeled EBM). Lightly supervised >80%. Within 10% of full-supervised baseline on each domain.

**Deliverable:** `scripts/experiment_36_composite_energy.py`, `python/carnot/inference/self_supervised_energy.py`

### Experiment 37: Multi-Token Prediction Confidence (Nemotron-Inspired)

**Hypothesis:** A model's ability to predict multiple tokens ahead signals reasoning confidence. Models with MTP heads expose this directly. For models without MTP heads, we can simulate it by running the model autoregressively from position t and checking whether it can "see ahead."

**Method:**
1. For models WITH MTP heads (Nemotron 3 Super, DeepSeek-V3): extract auxiliary prediction head outputs at each position
2. Compute MTP confidence features:
   - Agreement between main head and MTP heads (token-level match)
   - Entropy of each MTP head's prediction
   - Confidence degradation rate: how quickly entropy increases with prediction horizon
3. For models WITHOUT MTP heads: simulate by caching the model's hidden state at position t, then running a speculative k-step rollout. Compare predicted token at t+k against what was actually generated.
4. MTP features are self-supervised — no hallucination labels needed to compute them
5. Add to composite energy from Experiment 36

**Data needed:** Unlabeled text for inference + labeled data for **validation**
**Success:** MTP confidence alone >65%. Added to composite energy, total >75%.

**Deliverable:** `scripts/experiment_37_mtp_confidence.py`

### Experiment 38: Cross-Architecture Consensus Energy

**Hypothesis:** When architecturally diverse frozen models (dense transformer + MoE + Mamba hybrid) agree on an answer, it's more likely correct. Disagreement is energy. This requires only forward passes on unlabeled text through multiple frozen models — no hallucination labels.

**Method:**
1. Select 3-4 architecturally diverse frozen models:
   - Dense transformer: Qwen3-0.6B or Llama-3.2-1B
   - MoE: Mixtral-8x7B
   - Mamba hybrid: Mamba-2.8B or Jamba
2. Run the same inputs through all models, extract:
   - Per-model logprobs for the generated tokens
   - Top-k predicted token overlap across models
   - Logprob variance across models (high variance = disagreement)
   - For MoE: routing entropy per model
3. Define consensus energy:
   - E_consensus = logprob_variance + (1 - top_k_overlap) + Σ routing_entropies
   - High energy = models disagree = likely hallucination
4. This is fully self-supervised — the "label" is agreement/disagreement across models
5. Validate against labeled hallucination data

**Why this matters:** Experiment 26 showed cross-model transfer fails at the activation level (model-specific representations). But consensus at the **output** level (do they agree on the answer?) is architecture-independent. Adding MoE and Mamba models — with fundamentally different internals — makes consensus stronger than comparing similar dense transformers.

**Data needed:** Unlabeled text + labeled data for **validation only**
**Success:** Consensus energy alone >70% across all domains. Combined with per-model features >80%.

**Deliverable:** `scripts/experiment_38_consensus_energy.py`, `python/carnot/inference/consensus_scorer.py`

### Experiment 39: Unembedding Geometry as Semantic Energy

**Hypothesis:** The unembedding matrix (the model's output projection) defines a similarity structure over all tokens. The "logit lens" technique projects intermediate hidden states through this matrix to see what the model "thinks" at each layer. The trajectory of these projections through the layer stack reveals whether the model is converging toward an answer (low energy) or wandering (high energy).

**Method:**
1. For each layer l in the frozen model, project the hidden state through the unembedding matrix: pseudo_logits(l) = W_unembed · hidden_state(l)
2. Compute trajectory features:
   - Convergence rate: how quickly the top-1 prediction stabilizes across layers
   - Prediction consistency: does the model's "opinion" at layer 4 match layer 24?
   - Entropy trajectory: how does prediction entropy change across layers?
   - The "logit lens U-curve": does confidence follow our discovered U-curve pattern (high at early/late layers, low in middle)?
3. These features are self-supervised — they describe the model's own internal deliberation process
4. Define logit lens energy: E_lens = (1 - convergence_rate) + entropy_variance_across_layers

**Data needed:** Unlabeled text + labeled data for **validation only**
**Success:** Logit lens energy >60% for hallucination detection. Convergence rate feature alone >55%.

**Deliverable:** `scripts/experiment_39_logit_lens_energy.py`, `python/carnot/introspection/logit_lens.py`

---

## Phase 3: Consensus Energy Landscape (No Labels)

**Goal:** Build a multi-model energy landscape where the "correct" answer is the minimum-energy configuration — defined by cross-model weight structure agreement. No hallucination labels needed anywhere in this phase.
**Timeline:** Medium-to-long-term (3-5 sessions each)

### Experiment 40: Weight-Space Model Similarity Map

**Hypothesis:** Before running any inference, we can compute a "knowledge map" from frozen weights alone. Models with similar weight structure in a given layer/expert share knowledge for the same input types. Models with different structure have complementary knowledge. This map tells us which models to trust for which domains.

**Method:**
1. For each pair of frozen models, compute:
   - CKA (Centered Kernel Alignment) between corresponding layers
   - SVCCA (Singular Vector Canonical Correlation Analysis) for cross-architecture comparison
   - For MoE models: expert-to-expert similarity matrix across models
2. Build a "knowledge graph": nodes = models, edges = similarity per layer/domain
3. The knowledge graph tells us: "For math, models A and B share knowledge. For history, A and C are complementary."
4. Use this to weight the consensus energy (Exp 38): models that share knowledge for a given domain provide correlated votes (downweight). Models with complementary knowledge provide independent votes (upweight).

**Data needed:** **None** — pure weight-space analysis. Validation requires labeled data.
**Success:** The knowledge map predicts which model pairs will agree/disagree on specific domains (measured by output correlation on held-out data, Spearman ρ > 0.5).

**Deliverable:** `scripts/experiment_40_weight_similarity.py`, `python/carnot/introspection/knowledge_map.py`

### Experiment 41: Energy-Guided Decoding with Self-Supervised Energy

**Hypothesis:** The composite self-supervised energy (from Phase 2) can guide token selection during generation. Instead of generate-then-verify, we score candidate tokens in real-time using weight-derived features — without any labeled training data.

**Method:**
1. At each generation step, compute top-k candidate tokens from the LLM's logits
2. For each candidate, score using the self-supervised composite energy:
   - Logprob component (from the LLM itself)
   - Routing entropy component (if MoE model)
   - Channel alignment component (precomputed weight profiles)
   - MTP confidence component (if model has MTP heads)
   - Consensus component (if running multi-model)
3. Select the token with lowest composite energy
4. Compare against: greedy decoding, logprob-only rejection, and fully-supervised EBM guidance

**Data needed:** Unlabeled text + labeled data for **comparison only**
**Success:** Self-supervised guided decoding matches or exceeds logprob-only rejection (+10% baseline). Within 5% of supervised EBM guidance.

**Deliverable:** `python/carnot/inference/self_supervised_guided_decoding.py`

### Experiment 42: KL Distillation as Composable Energy

**Hypothesis:** The KL divergence between frozen models' output distributions is a label-free energy term. When model A is "surprised" by what model B generates, that surprise IS energy. Multiple models = multiple composable energy terms.

**Method:**
1. For a given input, compute output distribution (logits) from N frozen models
2. Per-model energy: E_i(answer) = KL(P_model_i(answer|input) || P_uniform) — how "surprised" each model is
3. Cross-model energy: E_ij = KL(P_model_i || P_model_j) — how much models disagree
4. Composite energy: E_total = Σ w_i · E_i + Σ w_ij · E_ij
5. Weights w_i, w_ij from the knowledge map (Exp 40) — upweight independent models, downweight correlated ones
6. Test as candidate selector: generate N answers from one model, score by composite multi-model energy

**Data needed:** Unlabeled text + labeled data for **validation only**
**Success:** Composite multi-model KL energy outperforms any single model's logprobs for candidate selection.

**Deliverable:** `python/carnot/inference/multi_model_energy.py`

---

## Phase 4: Standalone EBM with LLM as I/O

**Goal:** Build an EBM whose energy landscape encodes the collective knowledge of many frozen models. The LLM becomes a thin translation layer.
**Timeline:** Long-term (multi-week research program)
**Risk:** High — requires breakthroughs in representation learning

### 4a: Universal Activation Encoder

**Concept:** Train a shared encoder that maps activations from *any* frozen model into a common semantic space. In this space, the same concept has the same representation regardless of source model.

**Approach (informed by Nemotron's LatentMoE + our normalization work):**
1. Use activation normalization (Exp 35) as preprocessing — remove model-specific baseline
2. Use the knowledge map (Exp 40) to identify corresponding layers across architectures
3. Train a contrastive encoder per architecture type: normalized activations on the SAME input → nearby points; DIFFERENT inputs → far apart
4. The encoder's output space is the EBM's operating space
5. Key insight from Nemotron: their LatentMoE W↓ ∈ R^{ℓ×d} compresses while preserving routing quality. Our per-model encoder W_model ∈ R^{shared_dim × model_dim} does the same for cross-model alignment.

**Data needed:** Unlabeled text (same inputs through multiple models) — contrastive learning is self-supervised
**Deliverable:** `python/carnot/embeddings/universal_encoder.py`

### 4b: Consensus Energy Landscape

**Concept:** The EBM's energy over the universal semantic space captures what all frozen models collectively "know." Low energy = consensus. High energy = disagreement.

**Approach:**
1. Use universal encoder (4a) to project activations from N models into shared space
2. Energy = dispersion of encodings across models (tight cluster = low energy, dispersed = high energy)
3. Add structural terms: channel magnitude alignment (Exp 33), routing coherence (Exp 34), logit lens convergence (Exp 39)
4. Gradient descent in this space finds maximum consensus — the point where all models agree most

**Data needed:** Unlabeled text through multiple frozen models — fully self-supervised
**Deliverable:** `python/carnot/models/consensus_ebm.py`

### 4c: LLM as Language Interface

The LLM's role reduces to two operations:
1. **Encode:** Human question → semantic embedding (via frozen LLM forward pass)
2. **Decode:** Semantic embedding → human-readable answer (via frozen LLM generation)

Between encode and decode, all reasoning happens in the consensus energy landscape.

**Deliverable:** `python/carnot/inference/ebm_reasoning_engine.py`

### 4d: Hardware Path (Extropic TSU)

The Ising tier's quadratic energy maps directly to physical Boltzmann machines.

**Deliverable:** `python/carnot/hardware/ising_compiler.py`

---

## Dependency Graph

```
Phase 1: Weight Anatomy (NO LABELS for training)
  ├── Exp 32: Weight structure profiling         ← pure weight analysis, zero inference
  ├── Exp 33: Channel magnitude introspection    ← weight analysis + unlabeled forward pass
  ├── Exp 34: MoE routing entropy as energy      ← unlabeled forward pass only
  └── Exp 35: Activation normalization           ← unlabeled forward pass (uses labels for EBM comparison)
        │
        ▼
Phase 2: Self-Supervised Energy Composition (MINIMAL LABELS)
  ├── Exp 36: Composite self-supervised energy   ← combines Phase 1 features, labels for calibration only
  ├── Exp 37: MTP confidence features            ← unlabeled forward pass, Nemotron-inspired
  ├── Exp 38: Cross-architecture consensus       ← multi-model, fully self-supervised
  └── Exp 39: Unembedding geometry / logit lens  ← unlabeled, per-layer trajectory analysis
        │
        ▼
Phase 3: Consensus Energy Landscape (NO LABELS)
  ├── Exp 40: Weight-space model similarity map  ← PURE weight analysis, zero inference
  ├── Exp 41: Energy-guided decoding             ← uses self-supervised energy from Phase 2
  └── Exp 42: KL distillation energy             ← multi-model output comparison, no labels
        │
        ▼
Phase 4: Standalone EBM
  ├── 4a: Universal activation encoder           ← contrastive (self-supervised) + normalization (35)
  ├── 4b: Consensus energy landscape             ← combines all Phase 1-3 signals
  ├── 4c: LLM as language interface              ← uses energy landscape (4b)
  └── 4d: Hardware compilation                   ← uses working EBM (4b/4c)
```

**Phase 1** experiments are independent and can run in parallel.
**Phase 2** depends on Phase 1 features but experiments are mostly independent.
**Phase 3** depends on Phase 2 composites.
**Phase 4** is sequential.

## Implementation Priority

| Phase | Experiments | Labels Needed | Est. Sessions | What We Learn |
|-------|-----------|--------------|---------------|---------------|
| **Now** | **32, 33** | **Validation only** | **1-2 each** | **Do weight-structure features predict hallucination zones?** |
| **Now** | **34, 35** | **Validation only** | **1-2 each** | **Do routing entropy and normalization work label-free?** |
| Next | 36, 37 | Calibration (100-500) | 2-3 each | Does the composite self-supervised energy match supervised? |
| Next | 38, 39 | Validation only | 2-3 each | Does cross-architecture consensus work? Is logit lens viable? |
| Then | 40-42 | None / Validation | 3-4 each | Can we build an energy landscape from weight geometry alone? |
| Long-term | 4a-4c | Self-supervised | 5+ each | Can the EBM reason without the LLM? |
| Future | 4d | N/A | TBD | Hardware acceleration |

## Label Dependency Summary

| Category | Experiments | Training Labels | Inference Labels |
|----------|-----------|----------------|-----------------|
| **Pure weight analysis** | 32, 33 (weight parts), 40 | None | None |
| **Self-supervised inference** | 34, 37, 38, 39, 42 | None | None |
| **Normalization + comparison** | 35 | For EBM baseline only | None |
| **Calibration (tiny label set)** | 36 | 100-500 for weight tuning | None |
| **Validation** | All | N/A | Existing 52K dataset |

**10 of 11 experiments need zero training labels.** The existing 52K labeled token dataset is sufficient for all validation. Only Experiment 36's lightly-supervised variant needs any labeled data at all (100-500 tokens for weight calibration).

## New Models to Acquire

| Model | Architecture | Size | Why |
|-------|-------------|------|-----|
| **Mixtral-8x7B** | Dense MoE | 46.7B (12.9B active) | **Priority 1:** Unlocks routing features (Exp 33, 34), MoE channel analysis |
| **Mamba-2.8B or Jamba-1.5-Mini** | SSM / Mamba hybrid | 2.8-24B | **Priority 2:** Architectural diversity for consensus (Exp 38) |
| Nemotron 3 Super NVFP4 | Mamba-2 + LatentMoE | ~30GB quantized | MTP heads (Exp 37), richest routing structure |
| Qwen3-4B or 8B | Dense transformer | 4-8B | Scale comparison, larger activation space |

**Mixtral-8x7B is the single most important model to acquire.** It unlocks 4 experiments (32 MoE parts, 33, 34, 38) and is small enough to run locally with quantization.

## Success Criteria

**Phase 1 success:** Weight-derived features correlate with hallucination (ρ > 0.5). Routing entropy alone >60%. Channel alignment alone >60%.

**Phase 2 success:** Unsupervised composite energy >70% on multi-domain (matching supervised EBM without labels). Cross-architecture consensus >70%. Within 10% of fully supervised baseline.

**Phase 3 success:** Self-supervised energy-guided decoding matches or exceeds logprob-only rejection. Weight-space knowledge map predicts model agreement patterns.

**Phase 4 success:** The EBM can find correct answers to structured problems via gradient descent in a label-free consensus energy landscape, with the LLM only for encoding/decoding.

## Relationship to Carnot's Core Architecture

| Component | Current Use | Weight-First Extension |
|-----------|------------|----------------------|
| Gibbs model | Per-token hallucination detector (supervised) | Self-supervised composite energy scorer |
| Boltzmann model | Deep energy network (supervised) | Consensus energy landscape (self-supervised) |
| Ising model | Edge/hardware tier | Weight-derived coupling matrices (WᵀW) |
| NCE training | Train per-token EBM | **Replaced by** self-supervised energy composition |
| Composite scorer | Logprob + structural tests | Multi-signal self-supervised composition |
| **NEW: Weight profiler** | — | Static weight analysis: rank, norms, condition |
| **NEW: Channel profiler** | — | MoE expert channel magnitude structure |
| **NEW: Routing extractor** | — | MoE router entropy and expert patterns |
| **NEW: Logit lens** | — | Unembedding projection trajectory analysis |
| **NEW: Knowledge map** | — | Cross-model weight similarity graph |
| **NEW: Activation normalizer** | — | Per-sequence normalization for domain invariance |
| **NEW: Consensus scorer** | — | Multi-model output agreement as energy |

The `EnergyFunction` trait/protocol means all new self-supervised energy functions plug into existing samplers, training loops, and verification infrastructure. The key shift: **training** the EBM becomes optional — the energy function is defined by the frozen weights themselves.
