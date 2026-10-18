# LLM Weight Introspection for Targeted EBM Training

**Created:** 2026-04-05
**Status:** Proposed
**Priority:** High — this is a shortcut to effective EBMs

## The Insight

Instead of training an EBM from scratch on (correct, incorrect) pairs, **introspect the LLM's own weights to find which layers and activations correlate with hallucination vs reliable reasoning**. Use that knowledge to train a focused EBM that monitors the specific internal signals that predict failure.

This is fundamentally more efficient than black-box training: the LLM already "knows" when it's uncertain — the information is in its activations. We just need to extract it.

## Key Techniques from Research

### 1. Layer-wise Semantic Dynamics (LSD)
- Tracks how semantic representations evolve across transformer layers
- Hallucinations show distinctive geometric patterns (embedding drift, sudden direction changes)
- **For Carnot**: monitor the embedding trajectory during generation. Energy = geometric deviation from "confident" trajectories.

### 2. LayerNavigator
- Scores each layer's "steerability" — how much intervening at that layer affects output
- Identifies which layers are most effective for correction without testing every layer
- **For Carnot**: run LayerNavigator on a local model to find the 2-3 layers most correlated with hallucination. Train the EBM to monitor ONLY those layers (focused, not whole-model).

### 3. Activation Engineering / Activation Editing
- Activation vectors mediate specific behaviors
- Techniques: additive steering, norm-preserving rotation, dynamic masking, multi-objective subspace manipulation
- **For Carnot**: extract "hallucination direction" in activation space. The EBM's energy function = projection onto this direction. High projection = likely hallucination.

### 4. Contrastive Weight Steering (CWS)
- Modifies LLM behavior via weight arithmetic (add/subtract weight vectors) without retraining
- **For Carnot**: combine with the EBM loop — when the EBM detects high energy (hallucination), apply CWS to steer the LLM's weights away from the hallucination direction for the next generation.

### 5. Intra-Layer Local Information Scores
- Uncertainty estimation from local information within individual layers
- **For Carnot**: this IS semantic energy (P1) but at the layer level instead of output level. More fine-grained hallucination detection.

## Architecture: EBM as Activation Monitor

```
LLM generates token-by-token
    ↓
At each layer, extract activations
    ↓
EBM scores activation trajectory:
  - Low energy: activations follow "confident" pattern
  - High energy: activations deviate (hallucination signal)
    ↓
If energy spikes mid-generation:
  - Option A: halt and repair (current approach)
  - Option B: steer activations via CWS (in-generation correction)
  - Option C: backtrack to last low-energy token and regenerate
```

This is **real-time EBM guidance during generation**, not post-hoc verification.

## Implementation Plan

### Milestone A: Activation Extraction

**What:** Extract per-layer activations from a local model during inference.

**Concrete steps:**
1. Use a small local model (Qwen3-0.6B or Phi-3-mini) via transformers
2. Hook into model.forward() to capture hidden states at each layer
3. Write `python/carnot/embeddings/activation_extractor.py`:
   - `extract_layer_activations(model, tokenizer, text) -> dict[int, jax.Array]`
   - Returns {layer_num: activation_tensor} for all layers
4. Compute per-layer statistics: norm, direction change, entropy

### Milestone B: Hallucination Direction Identification

**What:** Find the "hallucination direction" in activation space.

**Concrete steps:**
1. Generate (correct_output, hallucinated_output) pairs using the LLM
2. Extract activations for both
3. Compute the difference vector at each layer
4. PCA/SVD to find the principal direction of hallucination
5. This direction IS the EBM's energy function: E(x) = projection of x onto hallucination direction

### Milestone C: Layer-Targeted EBM

**What:** Train a small EBM that monitors only the critical layers.

**Concrete steps:**
1. Use LayerNavigator approach to identify top-3 most steerability layers
2. Extract activations from ONLY those layers (fast, focused)
3. Train Gibbs model on (correct_activations, hallucination_activations) at those layers
4. The trained EBM is a compact hallucination detector

### Milestone D: In-Generation Steering

**What:** Use the EBM to steer the LLM during generation, not just after.

**Concrete steps:**
1. Hook into the LLM's generation loop
2. After each token: extract activations → EBM scores → if energy high:
   - Apply CWS: subtract hallucination direction from the current layer's weights
   - Or: add "confidence" direction to steer back to reliable generation
3. Continue generation with steered activations

## Why This Is Significant

Current approaches (our verify-and-repair, RLHF, etc.) are **post-hoc**: generate first, check later. This is **in-generation**: the EBM monitors the LLM's internal state in real-time and corrects it before the hallucination materializes in the output.

It's the difference between:
- A spell-checker that corrects after you type (post-hoc)
- A system that guides your fingers to the right keys (in-generation)

## Relationship to Roadmap v2

This slots between Phase 1 and Phase 2:

```
Phase 1 (current): Learned energy on embeddings
  → Phase 1.5 (NEW): Energy on LLM activations (this proposal)
Phase 2: Energy-Based Transformer
Phase 3: Self-supervised training
Phase 4: Hardware
```

Phase 1.5 is faster to results than Phase 2 because we're using an EXISTING LLM's internal representations rather than training a new model from scratch.

## Conductor Tasks

```python
{
    "id": "p1.5-activation-extraction",
    "title": "Extract per-layer activations from local model",
    "prompt": "...",
},
{
    "id": "p1.5-hallucination-direction",
    "title": "Find hallucination direction in activation space",
    "prompt": "...",
},
{
    "id": "p1.5-layer-targeted-ebm",
    "title": "Train layer-targeted hallucination detector",
    "prompt": "...",
},
{
    "id": "p1.5-in-generation-steering",
    "title": "Real-time EBM steering during LLM generation",
    "prompt": "...",
},
```
