# Carnot Research Roadmap v2: From Toolkit to State-of-the-Art EBM

**Created:** 2026-04-05
**Status:** Active
**Goal:** Build energy-based models that are natively better at reasoning than autoregressive LLMs — not wrappers around them.

## The Problem With v1

The current Carnot is infrastructure: constraints, repair, verification, autoresearch. It works, but LLMs beat it on every benchmark because our tasks are well-defined and LLMs have memorized the solutions. Post-hoc verification adds a mathematical guarantee but doesn't make the output better.

The fundamental shift: stop verifying LLM outputs. Start building EBMs that **generate better outputs natively**.

---

## Phase 1: Learned Energy in Latent Space

**Goal:** Train EBMs that operate on real embeddings, not bag-of-tokens frequency vectors.

**Why this matters:** The current `code_to_embedding()` is a bag-of-tokens — it loses all structural information. A real energy function needs to score *meaning*, not token frequencies. By operating in the latent space of a pretrained language model, the energy function inherits semantic understanding.

### Milestone 1.1: Local Model Embeddings

**What:** Use a small local model (Qwen3-0.6B, Phi-3-mini, or similar) to extract embeddings. Train a Carnot Gibbs/Boltzmann model to score these embeddings for correctness.

**Concrete steps:**
1. Add a dependency on `transformers` or `mlx` for local model inference
2. Write `python/carnot/embeddings/model_embeddings.py`:
   - `extract_embeddings(code: str, model_name: str) -> jax.Array` — get last-hidden-state from a local LM
   - Returns a real embedding vector (768-dim or 1024-dim) instead of bag-of-tokens (256-dim)
3. Retrain the learned code verifier on real embeddings
4. Compare accuracy: bag-of-tokens vs real embeddings on (correct, buggy) code classification

**Success metric:** Learned verifier accuracy improves >10% with real embeddings vs bag-of-tokens.

**Conductor task:**
```
Install a small local model (Qwen3-0.6B or similar via transformers library).
Write python/carnot/embeddings/model_embeddings.py with extract_embeddings().
Retrain the learned code verifier using real embeddings instead of bag-of-tokens.
Compare classification accuracy on 100 (correct, buggy) code pairs.
```

### Milestone 1.2: Energy-Scored Embedding Prediction (JEPA-style)

**What:** Given partial context (e.g., function signature + first few lines), predict the embedding of the missing part. The energy function scores how well the prediction matches the actual continuation.

**Concrete steps:**
1. Write `python/carnot/embeddings/jepa_energy.py`:
   - `ContextPredictionEnergy`: takes (context_embedding, prediction_embedding) → scalar energy
   - Low energy = prediction is coherent continuation of context
   - High energy = prediction is incoherent
2. Generate training data: take real Python functions, split into (context, continuation)
3. Train with NCE: real continuations are data, random continuations are noise
4. Test: given a function signature, which continuation has lower energy?

**Success metric:** The energy function correctly ranks real continuations over random ones >80% of the time.

**Conductor task:**
```
Implement JEPA-style context prediction energy in the embedding space.
Given a function signature embedding and a body embedding, the energy
function should assign low energy when the body correctly implements
the signature and high energy otherwise. Train on real Python functions.
```

### Milestone 1.3: Embedding-Space Repair

**What:** Instead of repairing discrete tokens (which doesn't work for code), repair in embedding space, then decode back to tokens.

**Concrete steps:**
1. Start from a bad prediction embedding
2. Run gradient descent on the energy function to improve the embedding
3. Find the nearest real code embedding to the repaired embedding
4. Decode back to tokens via the local model

This is the EBM analog of "editing in latent space" — gradient descent produces better embeddings than the LLM's original prediction.

**Success metric:** Repaired embeddings decode to code that passes more tests than the original LLM output.

---

## Phase 2: Energy-Based Transformer (EBT)

**Goal:** Build a transformer where the output IS energy and inference IS optimization — not an autoregressive model.

**Why this matters:** This is the architectural shift. An EBT evaluates complete configurations holistically (no left-to-right commitment, no error cascading). The EBT paper showed 35% faster scaling than Transformer++ and 29% improvement from System 2 thinking.

### Milestone 2.1: Minimal EBT in JAX

**What:** Implement a small EBT (4-8 layers, 64-128 dim) in JAX that can be trained on toy tasks.

**Concrete steps:**
1. Write `python/carnot/models/ebt.py`:
   - `EBTConfig`: n_layers, d_model, n_heads, d_ff
   - `EBTransformer(AutoGradMixin)`: takes (input, candidate_output) → scalar energy
   - Uses standard transformer attention but outputs a scalar, not a sequence
2. Training via the optimization-through-training approach (P5):
   - Start from random prediction
   - Run N gradient descent steps on energy
   - Loss = MSE between optimized prediction and ground truth
   - Backpropagate through all N steps (Hessian-vector products via JAX)
3. Test on sequence completion: predict the next 4 tokens of a simple sequence

**Success metric:** EBT achieves >70% accuracy on toy sequence prediction via energy minimization.

**Conductor task:**
```
Implement a minimal Energy-Based Transformer in python/carnot/models/ebt.py.
The model takes (input_sequence, candidate_output) and returns scalar energy.
Train using optimization-through-training (P5): gradient descent from random
predictions toward the correct output. Test on simple sequence completion.
```

### Milestone 2.2: EBT for Code

**What:** Train the EBT on code completion: given a function signature, generate the body by energy minimization.

**Concrete steps:**
1. Tokenize Python functions into integer sequences
2. Split into (signature_tokens, body_tokens)
3. Train EBT: energy(signature, candidate_body) should be low for correct body
4. Inference: start from random tokens, gradient descent on energy to find the body
5. Compare: EBT-generated code vs Haiku-generated code on test cases

**Success metric:** EBT generates code that passes test cases on at least 30% of tasks.

### Milestone 2.3: System 2 Thinking

**What:** Add the self-verification pattern from the EBT paper: generate M candidates, select minimum energy.

**Concrete steps:**
1. Generate M=10 candidates from different random starts
2. Run N=100 gradient descent steps on each
3. Select the candidate with lowest energy
4. Compare: M=1 vs M=10 vs M=100 accuracy

**Success metric:** M=10 improves accuracy >15% over M=1.

---

## Phase 3: Self-Supervised Energy Training (JEPA)

**Goal:** Train the energy function on unlabeled data — no test cases, no labels. The energy function learns what "coherent code" means from code alone.

### Milestone 3.1: Masked Code Prediction

**What:** Mask parts of real code, train the energy function to score whether a candidate fill-in is coherent with the context.

**Concrete steps:**
1. Collect a corpus of Python functions (stdlib, popular packages)
2. For each function: randomly mask 20-50% of tokens
3. Train: energy(context, real_fill) < energy(context, random_fill)
4. No labels needed — the real code IS the training signal

**Success metric:** The energy function can distinguish real code from shuffled code >90% of the time.

### Milestone 3.2: Scaling

**What:** Scale the EBT to useful size (125M-350M parameters) and train on a large code corpus.

**Concrete steps:**
1. Use the wgpu GPU backend for training
2. Distribute training across WebGPU workers
3. Train on GitHub code corpus (or The Stack)
4. Evaluate on HumanEval, MBPP, or similar code benchmarks

**Success metric:** Competitive with similarly-sized autoregressive models on code benchmarks, with the added benefit of self-verification (System 2 thinking).

---

## Phase 4: Hardware-Native EBM

**Goal:** Deploy on thermodynamic sampling hardware (Extropic TSU) for orders-of-magnitude speedup.

### Milestone 4.1: WGSL Shader Compilation for EBT

**What:** Compile the trained EBT's energy function into WGSL compute shaders for the WebGPU gateway. This enables distributed inference across browser GPUs.

### Milestone 4.2: Extropic TSU Integration

**What:** Map the energy landscape directly to Extropic's Thermodynamic Sampling Unit for native analog inference.

---

## Conductor Integration

The research conductor should cycle through these milestones in order. Each milestone has a concrete task prompt for Claude Code. The conductor:

1. Checks which milestone is current (from ops/status.md)
2. Sends the corresponding prompt to Claude Code
3. Verifies tests pass
4. Commits and pushes
5. Moves to the next milestone

### Phase 1 Conductor Tasks (ready to execute)

```python
PHASE_1_TASKS = [
    {
        "id": "p1-m1.1-local-embeddings",
        "milestone": "1.1",
        "title": "Add local model embeddings",
        "prompt": "...",  # detailed above
        "depends_on": None,
    },
    {
        "id": "p1-m1.2-jepa-energy",
        "milestone": "1.2",
        "title": "JEPA-style context prediction energy",
        "prompt": "...",
        "depends_on": "p1-m1.1-local-embeddings",
    },
    {
        "id": "p1-m1.3-embedding-repair",
        "milestone": "1.3",
        "title": "Embedding-space repair",
        "prompt": "...",
        "depends_on": "p1-m1.2-jepa-energy",
    },
]
```

---

## Timeline Estimate

| Phase | Milestones | Estimated Sessions | Key Deliverable |
|-------|-----------|-------------------|-----------------|
| 1 | 1.1-1.3 | 5-8 | Energy function on real embeddings |
| 2 | 2.1-2.3 | 8-12 | Working EBT that generates code |
| 3 | 3.1-3.2 | 10-15 | Self-supervised EBT on code corpus |
| 4 | 4.1-4.2 | 5-8 | Hardware-accelerated inference |

Total: ~28-43 sessions to go from current state to a publishable, state-of-the-art energy-based code generation system.

---

## What Makes This State-of-the-Art

No one has built:
1. An EBT specifically for code generation (the EBT paper does language + images)
2. JEPA-style self-supervised training on code
3. System 2 thinking for code (multi-start energy minimization)
4. Autoresearch that improves the EBT architecture itself

Carnot's unique position: Rust+JAX dual implementation, wgpu GPU backend, WebGPU distributed compute, and the autoresearch self-improvement loop. The combination doesn't exist elsewhere.
