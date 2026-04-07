#!/usr/bin/env python3
"""Train per-token EBMs across multiple LLM architectures.

Collects activations from each model on TruthfulQA (200 questions),
trains a Gibbs EBM, exports to HuggingFace format.

Models are processed sequentially (one at a time to fit in GPU memory).

Usage:
    sg render -c 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
        PYTHONUNBUFFERED=1 .venv/bin/python scripts/train_ebm_multi_model.py'

    # Single model:
    sg render -c '... scripts/train_ebm_multi_model.py --only qwen35-2b'
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "exports")

# Model registry: short_id -> (hf_name, has_chat_template, enable_thinking_param)
MODEL_REGISTRY = {
    # LiquidAI
    "lfm25-350m": ("LiquidAI/LFM2.5-350M", False, False),
    "lfm25-12b": ("LiquidAI/LFM2.5-1.2B-Instruct", True, False),
    # Bonsai (prism-ml, Qwen3-based)
    "bonsai-17b": ("prism-ml/Bonsai-1.7B-unpacked", True, True),
    # Qwen3.5 family
    "qwen35-08b": ("Qwen/Qwen3.5-0.8B", True, True),
    "qwen35-2b": ("Qwen/Qwen3.5-2B", True, True),
    "qwen35-4b": ("Qwen/Qwen3.5-4B", True, True),
    "qwen35-9b": ("Qwen/Qwen3.5-9B", True, True),
    "qwen35-27b": ("Qwen/Qwen3.5-27B", True, True),
    "qwen35-35b": ("Qwen/Qwen3.5-35B-A3B", True, True),
    # Gemma 4 (base and instruction-tuned)
    "gemma4-e2b": ("google/gemma-4-E2B", False, False),
    "gemma4-e2b-it": ("google/gemma-4-E2B-it", True, False),
    "gemma4-e4b": ("google/gemma-4-E4B", False, False),
    "gemma4-e4b-it": ("google/gemma-4-E4B-it", True, False),
    # OpenAI
    "gptoss-20b": ("openai/gpt-oss-20b", True, False),
}


def check_truthfulqa_answer(response: str, correct_answers: list[str],
                            incorrect_answers: list[str], best_answer: str = "") -> bool:
    """Check if response matches any correct answer."""
    response_lower = response.lower().strip()
    for correct in correct_answers:
        if correct.lower() in response_lower:
            return True
    if best_answer:
        words = best_answer.lower().split()
        if len(words) >= 3:
            for i in range(len(words) - 2):
                phrase = " ".join(words[i:i+3])
                if phrase in response_lower:
                    return True
    for incorrect in incorrect_answers:
        if incorrect.lower() in response_lower:
            return False
    return False


def collect_activations(
    model_name: str,
    short_id: str,
    has_chat: bool,
    has_thinking: bool,
    n_questions: int = 200,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Load model, generate on TruthfulQA, collect last-layer activations.

    Returns (activations, labels, hidden_dim, n_correct, n_wrong).
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        output_hidden_states=True,
        torch_dtype="auto" if device == "cuda" else None,
    )
    if device == "cuda":
        model = model.cuda()
    model.eval()

    # Get hidden dim from config
    config = model.config
    if hasattr(config, 'get_text_config'):
        tc = config.get_text_config()
    else:
        tc = config
    hidden_dim = getattr(tc, 'hidden_size', getattr(tc, 'd_model', 1024))
    print(f"  Hidden dim: {hidden_dim}, Device: {device}")

    ds = load_dataset("truthful_qa", "generation")
    questions = list(ds["validation"])[:n_questions]

    all_activations = []
    all_labels = []
    n_correct = 0
    n_wrong = 0

    for qi, example in enumerate(questions):
        question = example["question"]
        correct_answers = example["correct_answers"]
        incorrect_answers = example["incorrect_answers"]
        best_answer = example.get("best_answer", "")

        prompt = f"Answer briefly and factually in one sentence. {question}"

        if has_chat and hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            kwargs = {}
            if has_thinking:
                kwargs["enable_thinking"] = False  # Disable thinking for cleaner activations
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs,
            )
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=80, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen_ids = outputs[0, prompt_len:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()

            is_correct = check_truthfulqa_answer(response, correct_answers, incorrect_answers, best_answer)
            if is_correct:
                n_correct += 1
            else:
                n_wrong += 1

            with torch.no_grad():
                ho = model(outputs, output_hidden_states=True)
                hs = ho.hidden_states

            last_layer = hs[-1][0, prompt_len:, :].float().cpu().numpy()
            gen_tok_ids = gen_ids.cpu().numpy()

            for t in range(len(gen_tok_ids)):
                all_activations.append(last_layer[t])
                all_labels.append(1 if is_correct else 0)

        except Exception as e:
            print(f"  WARNING: Question {qi} failed: {e}")
            n_wrong += 1
            continue

        if (qi + 1) % 50 == 0:
            print(f"    [{qi+1:3d}/{n_questions}] tokens={len(all_activations)} "
                  f"correct={n_correct} wrong={n_wrong} ({n_correct/(qi+1)*100:.0f}%)")

    # Free GPU memory
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    activations = np.stack(all_activations).astype(np.float32)
    labels = np.array(all_labels, dtype=np.int32)

    return activations, labels, hidden_dim, n_correct, n_wrong


def train_ebm(activations: np.ndarray, labels: np.ndarray, hidden_dim: int) -> tuple:
    """Train a Gibbs EBM on the activations. Returns (ebm, test_acc, gap)."""
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss

    acts = jnp.array(activations)
    correct = acts[labels == 1]
    wrong = acts[labels == 0]
    min_n = min(len(correct), len(wrong))

    if min_n < 20:
        print(f"  WARNING: Only {min_n} balanced samples, may overfit")
        if min_n < 5:
            return None, 0.5, 0.0

    rng = np.random.default_rng(42)
    correct = correct[rng.permutation(len(correct))[:min_n]]
    wrong = wrong[rng.permutation(len(wrong))[:min_n]]
    split = int(min_n * 0.8)
    tc, tw = correct[:split], wrong[:split]
    vc, vw = correct[split:], wrong[split:]

    # Scale hidden dims with input_dim
    if hidden_dim <= 1024:
        hdims = [256, 64]
    elif hidden_dim <= 2560:
        hdims = [512, 128]
    else:
        hdims = [1024, 256]

    key = jrandom.PRNGKey(42)
    config = GibbsConfig(input_dim=hidden_dim, hidden_dims=hdims, activation="silu")
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

    for ep in range(300):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - 0.005 * g, params, grads)
    set_p(ebm, params)

    # Evaluate
    n_eval = min(300, len(vc))
    ce = [float(ebm.energy(vc[i])) for i in range(n_eval)]
    we = [float(ebm.energy(vw[i])) for i in range(n_eval)]
    thresh = (sum(ce) / len(ce) + sum(we) / len(we)) / 2
    tp = sum(1 for e in we if e > thresh)
    tn = sum(1 for e in ce if e <= thresh)
    test_acc = (tp + tn) / (len(ce) + len(we))
    gap = sum(we) / len(we) - sum(ce) / len(ce)

    return ebm, test_acc, gap


def export_model(
    ebm, short_id: str, model_name: str, hidden_dim: int,
    test_acc: float, gap: float, n_tokens: int, n_correct: int, n_wrong: int,
):
    """Save trained EBM to exports/ directory."""
    from safetensors.numpy import save_file

    hdims = [w.shape[0] for w, _ in ebm.layers]
    export_name = f"per-token-ebm-{short_id}-nothink"
    out_dir = os.path.join(EXPORT_DIR, export_name)
    os.makedirs(out_dir, exist_ok=True)

    # Weights
    weights = {}
    for i, (w, b) in enumerate(ebm.layers):
        weights[f"layer_{i}_weight"] = np.array(w)
        weights[f"layer_{i}_bias"] = np.array(b)
    weights["output_weight"] = np.array(ebm.output_weight)
    weights["output_bias"] = np.array(ebm.output_bias)
    save_file(weights, os.path.join(out_dir, "model.safetensors"))

    # Config
    config = {
        "model_type": "gibbs_ebm",
        "input_dim": hidden_dim,
        "hidden_dims": hdims,
        "activation": "silu",
        "n_layers": len(hdims),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Metadata
    metadata = {
        "source_model": model_name,
        "thinking_mode": "disabled",
        "n_tokens": n_tokens,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "test_accuracy": round(test_acc, 4),
        "energy_gap": round(gap, 4),
        "hidden_dim": hidden_dim,
        "ebm_hidden_dims": hdims,
    }
    with open(os.path.join(out_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Model card
    card = f"""---
tags:
  - energy-based-model
  - hallucination-detection
  - jax
  - carnot
license: apache-2.0
---

# {export_name}

Per-token hallucination detection EBM for {model_name}.

| Metric | Value |
|--------|-------|
| Test accuracy | {test_acc:.1%} |
| Energy gap | {gap:.4f} |
| Source model | {model_name} |
| Hidden dim | {hidden_dim} |
| Architecture | Gibbs [{hidden_dim} → {' → '.join(str(h) for h in hdims)} → 1], SiLU |
| Training tokens | {n_tokens:,} |
| Thinking | disabled |

## Usage

```python
from carnot.inference.ebm_loader import load_ebm
ebm = load_ebm("{export_name}")
energy = float(ebm.energy(activation_vector))
# Low energy = likely correct, high energy = likely hallucination
```

Trained with [Carnot](https://github.com/Carnot-EBM/carnot-ebm).
"""
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(card)

    size = os.path.getsize(os.path.join(out_dir, "model.safetensors"))
    print(f"  Exported: {out_dir} ({size / 1e6:.1f} MB)")
    return export_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Train EBMs across multiple LLMs")
    parser.add_argument("--only", help="Train only this model (short ID)")
    parser.add_argument("--n-questions", type=int, default=200, help="TruthfulQA questions per model")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace after training")
    parser.add_argument("--org", default="Carnot-EBM", help="HuggingFace org")
    args = parser.parse_args()

    os.makedirs(EXPORT_DIR, exist_ok=True)

    if args.only:
        if args.only not in MODEL_REGISTRY:
            print(f"Unknown model: {args.only}")
            print(f"Available: {', '.join(MODEL_REGISTRY.keys())}")
            return 1
        models_to_train = {args.only: MODEL_REGISTRY[args.only]}
    else:
        models_to_train = MODEL_REGISTRY

    results = {}
    start_all = time.time()

    for short_id, (hf_name, has_chat, has_thinking) in models_to_train.items():
        print(f"\n{'=' * 60}")
        print(f"MODEL: {short_id} ({hf_name})")
        print(f"{'=' * 60}")

        start = time.time()

        # Collect activations
        try:
            activations, labels, hidden_dim, n_correct, n_wrong = collect_activations(
                hf_name, short_id, has_chat, has_thinking, args.n_questions,
            )
        except Exception as e:
            print(f"  FAILED to collect activations: {e}")
            results[short_id] = {"status": "FAILED", "error": str(e)}
            continue

        n_tokens = len(labels)
        print(f"  Collected: {n_tokens} tokens, {n_correct} correct, {n_wrong} wrong")

        # Save activations
        from safetensors.numpy import save_file
        acts_file = os.path.join(DATA_DIR, f"token_activations_{short_id}_nothink.safetensors")
        save_file({
            "activations": activations,
            "labels": labels,
        }, acts_file)
        print(f"  Saved activations: {acts_file}")

        # Train EBM
        print(f"  Training EBM (hidden_dim={hidden_dim})...")
        ebm, test_acc, gap = train_ebm(activations, labels, hidden_dim)

        if ebm is None:
            print(f"  SKIPPED: insufficient data")
            results[short_id] = {"status": "SKIPPED", "n_tokens": n_tokens}
            continue

        print(f"  Test accuracy: {test_acc:.1%}, gap: {gap:.4f}")

        # Export
        export_name = export_model(
            ebm, short_id, hf_name, hidden_dim,
            test_acc, gap, n_tokens, n_correct, n_wrong,
        )

        elapsed = time.time() - start
        results[short_id] = {
            "status": "OK",
            "hf_name": hf_name,
            "export_name": export_name,
            "hidden_dim": hidden_dim,
            "n_tokens": n_tokens,
            "accuracy": f"{n_correct}/{args.n_questions}",
            "test_acc": f"{test_acc:.1%}",
            "gap": f"{gap:.4f}",
            "time": f"{elapsed:.0f}s",
        }

        # Upload immediately after each model if requested
        if args.upload:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                repo_id = f"{args.org}/{export_name}"
                local_dir = os.path.join(EXPORT_DIR, export_name)
                api.create_repo(repo_id, exist_ok=True, repo_type="model")
                api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
                print(f"  Uploaded: https://huggingface.co/{repo_id}")
            except Exception as e:
                print(f"  Upload failed: {e}")

    # Summary
    total_time = time.time() - start_all
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"MULTI-MODEL EBM TRAINING COMPLETE ({total_time:.0f}s)")
    print(sep)
    print(f"{'Model':20s} {'Status':8s} {'Tokens':>8s} {'Hidden':>8s} {'EBM Acc':>8s} {'Gap':>8s} {'Time':>8s}")
    print("-" * 80)
    for short_id, r in results.items():
        if r["status"] == "OK":
            print(f"{short_id:20s} {'OK':8s} {r['n_tokens']:>8d} {r['hidden_dim']:>8d} "
                  f"{r['test_acc']:>8s} {r['gap']:>8s} {r['time']:>8s}")
        else:
            print(f"{short_id:20s} {r['status']:8s}")
    print(sep)

    # Upload
    if args.upload:
        print("\n--- Uploading to HuggingFace ---")
        from huggingface_hub import HfApi
        api = HfApi()
        for short_id, r in results.items():
            if r["status"] != "OK":
                continue
            repo_id = f"{args.org}/{r['export_name']}"
            local_dir = os.path.join(EXPORT_DIR, r["export_name"])
            try:
                api.create_repo(repo_id, exist_ok=True, repo_type="model")
                api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
                print(f"  Uploaded: https://huggingface.co/{repo_id}")
            except Exception as e:
                print(f"  FAILED {repo_id}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
