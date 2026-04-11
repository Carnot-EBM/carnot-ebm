---
tags:
  - energy-based-model
  - guided-decoding
  - constraint-satisfaction
  - jax
  - carnot
license: apache-2.0
---

> **Research Artifact — Not Production-Ready**
>
> Real-model validation is pending (Exp-111). Exp-110 results use a mock LLM
> with deterministic error injection. The constraint checker works correctly
> (0.006 ms/check on CPU); the guidance logic is unvalidated on live models.

# guided-decoding-adapter

Energy-guided decoding adapter for any HuggingFace causal LM.

Attaches Carnot's constraint energy pipeline to the token generation loop.
Each token step runs a constraint violation check on the text generated so far;
violating tokens are penalised by subtracting `alpha × violation_count` from
all logits before sampling.

## How It Works

```
prompt → encode → [forward pass → check constraints → penalise logits → sample] × N → text
```

The constraint checker (`AutoExtractor`) detects violations across four domains:

| Domain | Constraint types |
|--------|-----------------|
| Arithmetic | addition, multiplication, bounds |
| Code | type checks, return types, initialisation |
| Logic | implication, exclusion, disjunction, negation, universal |
| Natural language | NL consistency |

Energy is a plain violation count (not a calibrated probability).  The penalty
is applied uniformly across the vocabulary — token ranking is preserved while
overall entropy increases, discouraging the model from continuing down a
constraint-violating path.

## Latency Profile

From Exp-102 (CPU, JAX_PLATFORMS=cpu, 1000-iteration benchmark):

| Measurement | Value |
|---|---|
| Constraint check p50 | 0.006 ms |
| Constraint check p99 | 0.034 ms |
| Extraction p50 | 0.276 ms |
| Per-token budget fraction | 0.04% of 20 ms/token |
| Verdict | **Fits in real-time generation budget** |

## Usage

```python
from carnot.inference.guided_decoding import GuidedDecoder
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model (any HF causal LM)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
model.eval()

# Load adapter from local directory or HuggingFace Hub
decoder = GuidedDecoder.from_pretrained("Carnot-EBM/guided-decoding-adapter")

# Generate with constraint guidance
result = decoder.generate(model, tokenizer, "What is 47 + 28?")
print(result.text)
print(f"Energy checks: {result.energy_checks}, final energy: {result.final_energy}")
```

### Override defaults

```python
decoder = GuidedDecoder.from_pretrained(
    "Carnot-EBM/guided-decoding-adapter",
    alpha=1.0,           # stronger guidance
    check_every_k=5,     # check every 5 tokens (faster, less precise)
    energy_threshold=0.5 # only penalise when violations > 0.5
)
```

### Load from a local export directory

```python
decoder = GuidedDecoder.from_pretrained("./exports/guided-decoding-adapter")
```

## Return Value

`generate()` returns a `GuidedDecodingResult`:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Generated text (prompt excluded) |
| `tokens_generated` | `int` | Number of tokens produced |
| `energy_checks` | `int` | Times constraint check ran |
| `mean_penalty` | `float` | Average logit penalty applied |
| `latency_seconds` | `float` | Wall-clock time |
| `final_energy` | `float` | Violation count after last check |

## Constraint Weights

Default weights are stored in `constraint_weights.safetensors`.  Load and inspect:

```python
from safetensors.numpy import load_file
weights = load_file("constraint_weights.safetensors")
print(weights["all_weights"])   # shape (12,) float32
print(weights["default_alpha"]) # [0.5]
```

## Compatible Models

Tested target models (Exp-110):
- `Qwen/Qwen3.5-0.8B`
- `google/gemma-4-E4B-it`

Any HuggingFace `AutoModelForCausalLM` with `.logits` output should work.
The adapter does not modify model weights.


## Benchmark Results (Exp-138 & Exp-140)

> **Note — Simulated Inference**: All benchmark numbers below were produced
> with a *simulated* (mock) LLM, not a real transformer model.  The constraint
> checker and logit-penalty logic are real; the generation loop uses a
> deterministic stand-in.  Live-model E2E validation is pending (Exp-111).

### Accuracy (Exp-138, n=200/50/100, simulated inference)

| Dataset | Baseline | Guided | Guided+Verify-Repair | Delta (guided) |
|---------|----------|--------|----------------------|----------------|
| GSM8K (math) | 55.5% | 62.5% | 65.0% | **+7.0%** |
| HumanEval (code) | 100.0% | 100.0% | — | **+0.0%** |
| TruthfulQA | 55.0% | 56.0% | 61.0% | **+1.0%** |

### Latency (Exp-138, n=485 samples, CPU)

| Metric | Value |
|--------|-------|
| Constraint-check p50 | 0.0719 ms |
| Constraint-check p99 | 0.1275 ms |

### Latency — KAN Projection Mode (Exp-140, batch=1, CPU)

| Operation | p50 | p99 |
|-----------|-----|-----|
| Logit projection (energy gradient) | 0.077 ms | 0.271 ms |
| Total per-token (grad + projection) | 0.405 ms | 0.924 ms |

Exp-140 pass criterion: total p50 < 5 ms — **PASSED**
(actual 0.4054 ms vs 5.0 ms threshold).

## Installation

```bash
pip install carnot
```

Requires Python 3.11+.  See [pypi.org/project/carnot](https://pypi.org/project/carnot)
for the full package including the verify-repair pipeline.

## Limitations

1. **Simulated inference benchmark**: Exp-138 and Exp-140 used a mock LLM.
   Numbers show constraint-checker and logit-penalty overhead, not end-to-end
   accuracy on real models.  Treat accuracy deltas as directional, not final.
2. **No KV-cache**: Full forward pass every token.  Keep `max_tokens < 256`.
3. **Uniform penalty**: Adjusts entropy across the whole vocabulary; does not
   steer towards specific correct tokens.
4. **Energy is a violation count**: Not a calibrated probability.  High `alpha`
   + many violations → very flat distribution (model may repeat or stall).
5. **Min-text guard**: `AutoExtractor` skips texts < 5 chars (early tokens).
6. **Live-model E2E pending**: Exp-111 validation against Qwen/Gemma not done yet.

## Spec

- REQ-VERIFY-001: Constraint energy computed from partial text at each step.
- SCENARIO-VERIFY-004: Energy penalises logits before sampling.

## Citation

```bibtex
@misc{carnot2026guided,
  title  = {Carnot Guided Decoding Adapter},
  author = {Carnot-EBM},
  year   = {2026},
  url    = {https://github.com/Carnot-EBM/carnot-ebm}
}
```
