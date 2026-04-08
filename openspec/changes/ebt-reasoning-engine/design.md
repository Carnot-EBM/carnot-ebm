## Context

36 experiments proved activation-based hallucination detection fails in practice — EBMs detect confidence, not correctness. Logit lens (experiment 36) showed the model's intermediate layers don't disagree differently on correct vs wrong tokens. The computation dynamics are identical whether the model is right or wrong.

Kona (Logical Intelligence) demonstrates the alternative: make the EBM the reasoning engine itself, operating in continuous latent space. We have the EBM infrastructure (Gibbs models, NCE training, gradient descent, samplers) — we need to repurpose it from "activation classifier" to "reasoning energy function."

Current state: Carnot has per-token EBMs trained on 15 models (75-88% test accuracy, 50% practical). The Gibbs tier ([hidden_dims] → scalar) is well-tested. The gap is operating on semantic embeddings rather than raw activations.

## Goals / Non-Goals

**Goals:**
- Train a Gibbs EBM on sentence embeddings of (question, answer) pairs where E(q, correct_a) < E(q, wrong_a)
- Demonstrate gradient descent in embedding space moves wrong-answer embeddings toward low-energy regions
- Decode optimized embeddings back to text via nearest-neighbor lookup
- Measure: does energy-guided repair produce better answers than the original wrong answer?

**Non-Goals:**
- Full autoregressive text generation (that's a much harder problem)
- Training our own sentence encoder (use frozen off-the-shelf)
- Real-time inference speed (research prototype)
- Rust implementation (Python/JAX only for now)
- Replacing LLMs (this is a verification/repair tool, not a generator)

## Decisions

### 1. Sentence encoder: all-MiniLM-L6-v2

**Choice:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings, 22M params)
**Why:** Smallest high-quality sentence encoder. Fast inference, well-tested, 384-dim is manageable for Gibbs EBM training. Available on HuggingFace, no API key needed.
**Alternative considered:** Larger encoders (768-dim, 1024-dim) — rejected for first experiment. Can scale up if 384-dim works.

### 2. Input representation: concatenated (question, answer) embeddings

**Choice:** Concatenate question embedding (384-dim) + answer embedding (384-dim) → 768-dim input to Gibbs EBM.
**Why:** Simple, preserves both question and answer information independently. The EBM learns the joint distribution.
**Alternative considered:** Cross-encoding (encode question+answer as single string) — loses the ability to fix the question and optimize only the answer embedding. Difference encoding (answer - question) — loses absolute position information.

### 3. Energy function: Gibbs EBM [768 → 256 → 64 → 1]

**Choice:** Reuse existing GibbsModel with input_dim=768.
**Why:** Already tested, trained via NCE, differentiable via JAX. The same architecture that achieved 75-88% on activations. The key difference is the input (semantic embeddings, not raw activations).
**Alternative considered:** New architecture — rejected for first experiment. If Gibbs doesn't work on embeddings, architectural changes won't help.

### 4. Answer repair: gradient descent on answer embedding only

**Choice:** Fix question embedding, optimize answer embedding via `jax.grad` on the energy function. Use Langevin dynamics (gradient + noise) for exploration.
**Why:** This is the core hypothesis — can we move a wrong answer toward a correct one by following the energy gradient? The question is fixed (we know the question), the answer is the variable.
**Alternative considered:** Joint optimization of question + answer — rejected because we don't want to change the question.

### 5. Decoding: nearest-neighbor in embedding space

**Choice:** After optimization, find the nearest answer string from a candidate pool by cosine similarity.
**Why:** Simple, avoids the hard problem of continuous→discrete decoding. The candidate pool is pre-embedded answer strings from the dataset.
**Limitation:** Can only produce answers that exist in the candidate pool. This is fine for validation but not for open-ended generation.

### 6. Serialization: safetensors for trained EBM, no cross-language needed yet

**Choice:** Save trained Gibbs EBM weights via safetensors (existing infrastructure).
**Why:** Consistent with current export pipeline. Rust implementation deferred.

## Risks / Trade-offs

- **[Embedding space may not be smooth]** → Sentence embeddings might not have continuous paths from wrong to right answers. Mitigation: use Langevin dynamics with noise to escape local minima.
- **[384-dim may be too low]** → Semantic distinctions might require higher dimensionality. Mitigation: scale to 768/1024-dim encoders if 384 fails.
- **[Candidate pool limits decoding]** → Can only decode to known answers. Mitigation: acceptable for research validation. Production would need learned decoders.
- **[NCE training may overfit on small QA set]** → TruthfulQA has 817 questions. Mitigation: use MMLU/HaluEval data from experiment 31 to supplement.
- **[Energy landscape may have many local minima]** → Gradient descent might get stuck. Mitigation: multi-start repair (existing infrastructure from P2).
