# Energy-Guided Decoding Adapter

> **Research Prototype — Not Production Quality**
> From Carnot Experiment 110. Tested on template prompts with Qwen3.5-0.8B only.

## What This Is

`guided_decoding_adapter.py` is a **self-contained, model-agnostic adapter** that
wraps any HuggingFace causal LM and steers its generation away from
constraint-violating text using Carnot's constraint energy.

This is a **novel research artifact**: to our knowledge, token-by-token energy
guidance using an explicitly trained constraint EBM (rather than a reward model)
has not been published as a standalone adapter in this form before.

## Why This Is Novel

Most constrained decoding approaches:
1. Use a separately trained reward model (expensive, model-specific)
2. Apply beam search with external constraint checkers (requires reranking N-best)
3. Modify the model's weights (fine-tuning, RLHF)

This adapter instead:
- Uses a **pre-trained EBM** (the KAN model in this directory) as the constraint oracle
- Applies **per-token logit penalties** — no reranking, no beam search required
- Is **model-agnostic** — works with any HuggingFace causal LM
- Costs **< 0.01 ms overhead per token** (Exp 102 benchmark: 0.008 ms JIT-compiled)

## Quick Start (No Install Required)

```python
# Just copy guided_decoding_adapter.py to your project
from guided_decoding_adapter import EnergyGuidedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model.eval()

# alpha=0.5 is a good starting point
sampler = EnergyGuidedSampler(alpha=0.5, check_every_k=5)
result = sampler.generate(
    prompt="What is 42 + 17?",
    model=model,
    tokenizer=tokenizer,
    max_tokens=64,
    temperature=0.7,
)
print(result.text)
print(f"Energy checks: {result.energy_checks}")
print(f"Mean penalty applied: {result.mean_penalty:.3f}")
print(f"Final constraint energy: {result.final_energy:.3f}")
```

## With Full Carnot Integration

Install carnot for richer constraint extraction across arithmetic, logic, and code:

```bash
pip install carnot
```

```python
from guided_decoding_adapter import EnergyGuidedSampler
from carnot.pipeline.verify_repair import VerifyRepairPipeline

# Pre-warm the pipeline (loads extractors once)
pipeline = VerifyRepairPipeline()

sampler = EnergyGuidedSampler(
    pipeline=pipeline,  # uses Carnot's AutoExtractor
    alpha=0.5,
    check_every_k=5,
)

result = sampler.generate(
    prompt="Prove that all integers > 1 can be represented as a sum of primes.",
    model=model,
    tokenizer=tokenizer,
    max_tokens=256,
    temperature=1.0,
    domain="logic",
)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.5 | Guidance strength. 0 = no guidance. Higher = stronger penalty. |
| `check_every_k` | 1 | Recompute energy every k tokens. k=5 is a good tradeoff. |
| `energy_threshold` | 0.0 | Minimum violation count to trigger penalty. |
| `pipeline` | None | Optional Carnot VerifyRepairPipeline for richer extraction. |

## alpha Tuning Guide

| alpha | Effect |
|-------|--------|
| 0.0 | Baseline (no guidance) |
| 0.1–0.3 | Light nudge — minimal fluency impact |
| 0.5–1.0 | Moderate guidance (recommended starting range) |
| 2.0+ | Aggressive — may reduce fluency; use for hard constraints only |

## Performance Characteristics

| Metric | Value | Source |
|--------|-------|--------|
| Overhead per token (with Carnot) | 0.008 ms | Exp 102 (JIT-compiled) |
| Overhead per token (built-in) | < 0.001 ms | Pattern matching only |
| Memory overhead | Negligible | No additional model weights loaded |
| Compatible models | Any HuggingFace causal LM | Tested: Qwen3.5-0.8B |

## Fallback Behavior

When carnot is NOT installed, the adapter uses a built-in heuristic that detects:
- Self-contradicting arithmetic claims (e.g., "the answer is 5" then "the answer is 7")

When carnot IS installed, the adapter uses `AutoExtractor` which supports:
- Arithmetic constraint checking
- Logical consistency checking
- Python type/syntax constraint checking

## Limitations

- **No KV-cache**: The default `generate()` implementation re-runs the full forward
  pass each step. For long sequences, add `past_key_values` support.
- **Uniform penalty**: The logit penalty is applied uniformly to all tokens. This
  reduces confidence globally, not selectively. Token-specific penalties would
  require gradient access.
- **Heuristic extraction**: The built-in constraint extractor is a simple pattern
  matcher. It will miss many violation types.
- **Prototype only**: Not validated beyond arithmetic template prompts.

## Source

The canonical implementation lives in `python/carnot/inference/guided_decoding.py`
in the [Carnot repository](https://github.com/ianblenke/carnot).
This file is a standalone copy with an added built-in fallback extractor.
