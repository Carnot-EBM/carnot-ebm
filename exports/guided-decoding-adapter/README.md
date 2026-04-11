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

## Known Limitations

1. **No KV-cache**: Full forward pass every token. Keep `max_tokens < 256`.
2. **Uniform penalty**: Does not steer vocabulary — only adjusts entropy.
3. **Energy is a count**: Not calibrated; high alpha + many violations = very
   flat distribution (model may repeat or stall).
4. **Min-text guard**: AutoExtractor skips texts < 5 chars (early tokens).
5. **Mock-only validation**: Exp-110 used a mock LLM. Live-model E2E pending.

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
