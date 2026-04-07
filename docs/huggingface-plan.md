# HuggingFace Publishing Plan

## Organization: Carnot-EBM

### Repositories to Publish

#### 1. `Carnot-EBM/per-token-ebm-qwen3-06b`

**Description:** Per-token hallucination detection EBM trained on Qwen3-0.6B activations. 84.5% test accuracy.

**Contents:**
- `model.safetensors` — Gibbs EBM weights (input_dim=1024, hidden_dims=[256, 64], activation=silu)
- `config.json` — Model configuration
- `README.md` — Model card with usage, training data, limitations
- `training_metadata.json` — Training hyperparameters, dataset stats

**Training Data:** 26,800 per-token activations from 1,340 QA pairs (Qwen3-0.6B base model). Labels: correct answer = 1, wrong answer = 0.

**Source files:**
- Weights: trained by `scripts/train_per_token_ebm_large.py`
- Data: `data/token_activations_large.safetensors`

---

#### 2. `Carnot-EBM/per-token-ebm-qwen35-08b-nothink`

**Description:** Per-token hallucination detection EBM trained on Qwen3.5-0.8B activations WITHOUT thinking. 75.5% test accuracy.

**Contents:**
- `model.safetensors` — Gibbs EBM weights
- `config.json` — Model configuration
- `README.md` — Model card
- `training_metadata.json` — Training hyperparameters

**Training Data:** ~6,928 per-token activations from 200 TruthfulQA questions (Qwen3.5-0.8B, enable_thinking=False).

**Source files:**
- Weights: trained by `scripts/experiment_25_no_thinking.py`
- Data: collected inline during experiment

**Action needed:** Save the trained EBM weights from experiment 25 to safetensors.

---

#### 3. `Carnot-EBM/per-token-ebm-qwen35-08b-think`

**Description:** Per-token hallucination detection EBM trained on Qwen3.5-0.8B activations WITH thinking. 61.3% test accuracy.

**Contents:**
- `model.safetensors` — Gibbs EBM weights
- `config.json` — Model configuration
- `README.md` — Model card

**Training Data:** 52,296 per-token activations from QA + TruthfulQA (Qwen3.5-0.8B, thinking enabled).

**Source files:**
- Data: `data/token_activations_qwen35_merged.safetensors`
- Weights: trained by `scripts/train_per_token_ebm_combined.py --source merged`

---

#### 4. `Carnot-EBM/token-activations-qwen35`

**Description:** Raw per-token activation datasets for research. Includes correct/wrong labels.

**Contents:**
- `token_activations_qa_qwen35.safetensors` — 23,238 tokens from QA (Qwen3.5-0.8B)
- `token_activations_tqa_qwen35.safetensors` — 29,058 tokens from TruthfulQA (Qwen3.5-0.8B)
- `token_activations_qwen35_merged.safetensors` — 52,296 tokens combined
- `token_activations_large.safetensors` — 26,800 tokens (Qwen3-0.6B)
- `README.md` — Dataset card with collection methodology

**Fields per file:**
- `activations`: float32 [n_tokens, 1024]
- `labels`: int32 [n_tokens] (1=correct, 0=wrong)
- `token_ids`: int32 [n_tokens]
- `question_ids`: int32 [n_tokens]

---

#### 5. `Carnot-EBM/carnot-framework` (optional)

**Description:** The full Carnot framework as a pip-installable package.

This would be the `carnot` PyPI package once stabilized.

---

### Model Card Template

Each model repository should include a README.md with:

```markdown
---
tags:
  - energy-based-model
  - hallucination-detection
  - jax
license: apache-2.0
---

# {Model Name}

## Description
Per-token EBM for hallucination detection. Assigns low energy to activation
vectors from correct LLM responses and high energy to hallucinated ones.

## Usage
[Code example from usage-guide.md]

## Training
- Architecture: Gibbs EBM [1024 → 256 → 64 → 1], SiLU activation
- Loss: Noise Contrastive Estimation (NCE)
- Epochs: 300, lr=0.005
- Training data: {n_tokens} per-token activations from {model_name}

## Limitations
- Only works with activations from {model_name}
- Trained on {dataset_name} — may not generalize to other domains
- {accuracy}% test accuracy — not suitable as sole verification signal

## Principles Learned
See the [technical report](https://github.com/Carnot-EBM/carnot-ebm/docs/technical-report.md)
for all 10 principles from 25 experiments.
```

### Publishing Steps

1. **Create HuggingFace organization** `Carnot-EBM`
2. **Export model weights** — write a script to save trained Gibbs EBM parameters to safetensors with proper key names
3. **Write model cards** — use template above
4. **Upload datasets** — the activation safetensors files
5. **Upload models** — EBM weights + configs
6. **Link from README** — update repository URLs

### Export Script Needed

```python
# scripts/export_ebm_to_hf.py
# Train the EBM, save weights with standardized key names:
# - layer_0_weight, layer_0_bias
# - layer_1_weight, layer_1_bias
# - output_weight, output_bias
# Plus config.json with input_dim, hidden_dims, activation
```
