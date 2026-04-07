#!/usr/bin/env python3
"""Export trained per-token EBMs to HuggingFace-compatible format.

Trains each EBM variant and saves weights + config + model card
in a directory structure ready for `huggingface-hub` upload.

Output: exports/
  per-token-ebm-qwen3-06b/
  per-token-ebm-qwen35-08b-nothink/
  per-token-ebm-qwen35-08b-think/
  token-activations-qwen35/

Usage:
    .venv/bin/python scripts/export_ebm_to_hf.py [--upload --org Carnot-EBM]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

EXPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "exports")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def train_and_export_ebm(
    name: str,
    activations: np.ndarray,
    labels: np.ndarray,
    description: str,
    source_model: str,
    thinking_mode: str,
) -> dict:
    """Train a Gibbs EBM and export weights + config + model card."""
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    from safetensors.numpy import save_file

    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss

    out_dir = os.path.join(EXPORT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"  Tokens: {len(labels)}, dim={activations.shape[1]}")
    print(f"  Correct: {int(labels.sum())}, Wrong: {int(len(labels) - labels.sum())}")

    acts = jnp.array(activations)
    correct = acts[labels == 1]
    wrong = acts[labels == 0]
    min_n = min(len(correct), len(wrong))

    rng = np.random.default_rng(42)
    correct = correct[rng.permutation(len(correct))[:min_n]]
    wrong = wrong[rng.permutation(len(wrong))[:min_n]]
    split = int(min_n * 0.8)
    tc, tw = correct[:split], wrong[:split]
    vc, vw = correct[split:], wrong[split:]

    # Train
    hidden_dims = [256, 64]
    key = jrandom.PRNGKey(42)
    config = GibbsConfig(input_dim=1024, hidden_dims=hidden_dims, activation="silu")
    ebm = GibbsModel(config, key=key)

    def get_p(m):
        return {"layers": [(w, b) for w, b in m.layers],
                "output_weight": m.output_weight, "output_bias": m.output_bias}

    def set_p(m, p):
        m.layers = list(p["layers"])
        m.output_weight = p["output_weight"]
        m.output_bias = p["output_bias"]

    params = get_p(ebm)

    def loss_fn(p):
        old = get_p(ebm)
        set_p(ebm, p)
        r = nce_loss(ebm, tc, tw)
        set_p(ebm, old)
        return r

    start = time.time()
    for ep in range(300):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - 0.005 * g, params, grads)
        if ep % 100 == 0:
            set_p(ebm, params)
            loss = float(nce_loss(ebm, tc, tw))
            print(f"  Epoch {ep}: loss={loss:.4f}")
    set_p(ebm, params)
    elapsed = time.time() - start

    # Evaluate
    n_eval = min(500, len(vc))
    ce = [float(ebm.energy(vc[i])) for i in range(n_eval)]
    we = [float(ebm.energy(vw[i])) for i in range(n_eval)]
    thresh = (sum(ce) / len(ce) + sum(we) / len(we)) / 2
    tp = sum(1 for e in we if e > thresh)
    tn = sum(1 for e in ce if e <= thresh)
    test_acc = (tp + tn) / (len(ce) + len(we))
    gap = sum(we) / len(we) - sum(ce) / len(ce)

    print(f"  Test accuracy: {test_acc:.1%}")
    print(f"  Energy gap: {gap:.4f}")
    print(f"  Training time: {elapsed:.1f}s")

    # Save weights
    weights = {}
    for i, (w, b) in enumerate(ebm.layers):
        weights[f"layer_{i}_weight"] = np.array(w)
        weights[f"layer_{i}_bias"] = np.array(b)
    weights["output_weight"] = np.array(ebm.output_weight)
    weights["output_bias"] = np.array(ebm.output_bias)
    save_file(weights, os.path.join(out_dir, "model.safetensors"))

    # Save config
    config_dict = {
        "model_type": "gibbs_ebm",
        "input_dim": 1024,
        "hidden_dims": hidden_dims,
        "activation": "silu",
        "n_layers": len(hidden_dims),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save training metadata
    metadata = {
        "source_model": source_model,
        "thinking_mode": thinking_mode,
        "n_tokens": int(len(labels)),
        "n_correct": int(labels.sum()),
        "n_wrong": int(len(labels) - labels.sum()),
        "n_train": int(split),
        "n_test": int(min_n - split),
        "test_accuracy": round(test_acc, 4),
        "energy_gap": round(gap, 4),
        "training_epochs": 300,
        "learning_rate": 0.005,
        "training_time_seconds": round(elapsed, 1),
        "loss_function": "nce",
        "seed": 42,
    }
    with open(os.path.join(out_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Write model card
    model_card = f"""---
tags:
  - energy-based-model
  - hallucination-detection
  - jax
  - carnot
license: apache-2.0
---

# {name}

{description}

## Key Stats

| Metric | Value |
|--------|-------|
| Test accuracy | {test_acc:.1%} |
| Energy gap | {gap:.4f} |
| Source model | {source_model} |
| Thinking mode | {thinking_mode} |
| Training tokens | {len(labels):,} |
| Architecture | Gibbs [1024 → 256 → 64 → 1], SiLU |

## Usage

```python
from safetensors.numpy import load_file
import jax.numpy as jnp
import jax.random as jrandom
from carnot.models.gibbs import GibbsConfig, GibbsModel

# Load weights
weights = load_file("{name}/model.safetensors")
config = GibbsConfig(input_dim=1024, hidden_dims=[256, 64], activation="silu")
ebm = GibbsModel(config, key=jrandom.PRNGKey(0))

# Set weights
ebm.layers = [
    (jnp.array(weights["layer_0_weight"]), jnp.array(weights["layer_0_bias"])),
    (jnp.array(weights["layer_1_weight"]), jnp.array(weights["layer_1_bias"])),
]
ebm.output_weight = jnp.array(weights["output_weight"])
ebm.output_bias = jnp.array(weights["output_bias"])

# Score an activation vector (from {source_model} hidden states)
energy = float(ebm.energy(activation_vector))
# Low energy = likely correct, high energy = likely hallucination
```

## Training

- **Loss:** Noise Contrastive Estimation (NCE)
- **Epochs:** 300, lr=0.005
- **Data:** {len(labels):,} per-token activations from {source_model}
- **Labels:** correct answer tokens = low energy (data), wrong answer tokens = high energy (noise)

## Limitations

- Only works with activations from **{source_model}** — different models have different representation spaces
- {test_acc:.1%} accuracy — use as one signal among many, not as sole verification
- Trained on QA/TruthfulQA — may not generalize to all domains

## 10 Principles from Carnot Research

1. Simpler is better in small-data regimes
2. Token-level features > sequence-level
3. The model's own logprobs are the best energy
4. Overfitting is the main enemy
5. Extract features from generated tokens, not prompts
6. Different energy signals dominate in different domains
7. Statistical difference ≠ causal influence
8. Instruction tuning compresses the hallucination signal
9. Adversarial questions defeat post-hoc detection
10. Chain-of-thought compresses the hallucination signal

See the [Carnot technical report](https://github.com/Carnot-EBM/carnot-ebm/blob/main/docs/technical-report.md) for all 25 experiments.
"""
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(model_card)

    size = os.path.getsize(os.path.join(out_dir, "model.safetensors"))
    print(f"  Exported to: {out_dir} ({size / 1e6:.1f} MB)")

    return metadata


def main() -> int:
    from safetensors.numpy import load_file

    parser = argparse.ArgumentParser(description="Export EBMs to HuggingFace format")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace after export")
    parser.add_argument("--org", default="Carnot-EBM", help="HuggingFace organization name")
    args = parser.parse_args()

    os.makedirs(EXPORT_DIR, exist_ok=True)

    results = {}

    # --- Model 1: Qwen3-0.6B ---
    qa_file = os.path.join(DATA_DIR, "token_activations_large.safetensors")
    if os.path.exists(qa_file):
        data = load_file(qa_file)
        results["qwen3-06b"] = train_and_export_ebm(
            name="per-token-ebm-qwen3-06b",
            activations=data["activations"],
            labels=data["labels"],
            description="Per-token hallucination detection EBM trained on Qwen3-0.6B (base model) activations. Highest accuracy (84.5%) due to base model's larger activation gaps between correct and wrong answers.",
            source_model="Qwen/Qwen3-0.6B",
            thinking_mode="N/A (base model)",
        )
    else:
        print(f"SKIP: {qa_file} not found")

    # --- Model 2: Qwen3.5-0.8B with thinking ---
    merged_file = os.path.join(DATA_DIR, "token_activations_qwen35_merged.safetensors")
    if os.path.exists(merged_file):
        data = load_file(merged_file)
        # TruthfulQA portion only for think model (question_ids >= 10000)
        # But merged has both QA + TruthfulQA, all with thinking
        results["qwen35-08b-think"] = train_and_export_ebm(
            name="per-token-ebm-qwen35-08b-think",
            activations=data["activations"],
            labels=data["labels"],
            description="Per-token hallucination detection EBM trained on Qwen3.5-0.8B activations with thinking (chain-of-thought) enabled. Lower accuracy (67.2%) because thinking compresses the hallucination signal (Principle 10).",
            source_model="Qwen/Qwen3.5-0.8B",
            thinking_mode="enabled",
        )
    else:
        print(f"SKIP: {merged_file} not found")

    # --- Model 3: Qwen3.5-0.8B without thinking ---
    # This requires running experiment 25 to collect data.
    # For now, check if we have the no-think activations saved.
    nothink_file = os.path.join(DATA_DIR, "token_activations_qwen35_nothink.safetensors")
    if os.path.exists(nothink_file):
        data = load_file(nothink_file)
        results["qwen35-08b-nothink"] = train_and_export_ebm(
            name="per-token-ebm-qwen35-08b-nothink",
            activations=data["activations"],
            labels=data["labels"],
            description="Per-token hallucination detection EBM trained on Qwen3.5-0.8B activations WITHOUT thinking. Best accuracy for instruction-tuned models (75.5%) because disabling chain-of-thought preserves the hallucination signal (Principle 10).",
            source_model="Qwen/Qwen3.5-0.8B",
            thinking_mode="disabled (enable_thinking=False)",
        )
    else:
        print(f"\nNOTE: {nothink_file} not found.")
        print("  Run experiment 25 with --save-activations to generate it,")
        print("  or collect no-thinking activations separately.")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("EXPORT SUMMARY")
    print(f"{'='*60}")
    for model_id, meta in results.items():
        print(f"  {model_id}: {meta['test_accuracy']:.1%} accuracy, {meta['n_tokens']} tokens")
    print(f"\n  Export directory: {os.path.abspath(EXPORT_DIR)}")

    # --- Upload if requested ---
    if args.upload:
        print(f"\n--- Uploading to HuggingFace ({args.org}) ---")
        try:
            from huggingface_hub import HfApi
            api = HfApi()

            for model_id in results:
                repo_name = f"per-token-ebm-{model_id}"
                repo_id = f"{args.org}/{repo_name}"
                local_dir = os.path.join(EXPORT_DIR, repo_name)

                print(f"\n  Creating repo: {repo_id}")
                api.create_repo(repo_id, exist_ok=True, repo_type="model")

                print(f"  Uploading: {local_dir}")
                api.upload_folder(
                    folder_path=local_dir,
                    repo_id=repo_id,
                    repo_type="model",
                )
                print(f"  Done: https://huggingface.co/{repo_id}")

        except Exception as e:
            print(f"  Upload failed: {e}")
            print("  Run 'huggingface-cli login' first to authenticate.")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
