#!/usr/bin/env python3
"""Experiment 24: Multi-layer hallucination probing on Qwen3.5-0.8B.

Tests whether intermediate transformer layers retain hallucination signal
that gets compressed in the final layer (Principle 8).

Probes every 4th layer for speed: layers 0, 4, 8, 12, 16, 20, 24.

Usage:
    sg render -c 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
        .venv/bin/python scripts/experiment_24_layer_probing.py'
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def main() -> int:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from carnot.embeddings.layer_probing import probe_all_layers
    # Import the answer checker
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from collect_truthfulqa_activations import check_truthfulqa_answer

    print("=" * 60)
    print("EXPERIMENT 24: Multi-Layer Hallucination Probing")
    print("=" * 60)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3.5-0.8B"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        output_hidden_states=True,
        dtype=torch.float16 if device == "cuda" else None,
    )
    if device == "cuda":
        model = model.cuda()
    model.eval()

    # Load TruthfulQA (first 200 for speed)
    print("Loading TruthfulQA...")
    ds = load_dataset("truthful_qa", "generation")
    questions = list(ds["validation"])[:200]

    # Collect per-layer activations for correct and wrong responses
    # Probe every 4th layer (Qwen3.5-0.8B has 24 layers → indices 0-24 in hidden_states)
    probe_layers = list(range(0, 25, 4))  # 0, 4, 8, 12, 16, 20, 24
    print(f"Probing layers: {probe_layers}")

    correct_by_layer: dict[int, list] = {l: [] for l in probe_layers}
    wrong_by_layer: dict[int, list] = {l: [] for l in probe_layers}
    n_correct = 0
    n_wrong = 0

    for qi, example in enumerate(questions):
        question = example["question"]
        correct_answers = example["correct_answers"]
        incorrect_answers = example["incorrect_answers"]
        best_answer = example.get("best_answer", "")

        messages = [{"role": "user", "content": f"Answer briefly and factually in one sentence. {question}"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)

        gen_ids = outputs[0, prompt_len:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()

        is_correct = check_truthfulqa_answer(response, correct_answers, incorrect_answers, best_answer)

        # Get hidden states from ALL layers
        with torch.no_grad():
            ho = model(outputs, output_hidden_states=True)
            hs = ho.hidden_states

        # Collect per-token activations from each probed layer
        for layer_idx in probe_layers:
            if layer_idx < len(hs):
                layer_act = hs[layer_idx][0, prompt_len:, :].float().cpu().numpy()
                # Add each token as a separate sample
                target = correct_by_layer if is_correct else wrong_by_layer
                for t in range(len(layer_act)):
                    target[layer_idx].append(layer_act[t])

        if is_correct:
            n_correct += 1
        else:
            n_wrong += 1

        if (qi + 1) % 50 == 0:
            total_tokens = sum(len(v) for v in correct_by_layer.values())
            print(f"  [{qi+1:3d}/{len(questions)}] correct={n_correct} wrong={n_wrong} "
                  f"tokens/layer≈{total_tokens // len(probe_layers)}")

    print(f"\nCollection: {n_correct} correct, {n_wrong} wrong responses")

    # Convert lists to numpy arrays
    correct_acts = {}
    wrong_acts = {}
    for layer_idx in probe_layers:
        if correct_by_layer[layer_idx] and wrong_by_layer[layer_idx]:
            correct_acts[layer_idx] = np.array(correct_by_layer[layer_idx])
            wrong_acts[layer_idx] = np.array(wrong_by_layer[layer_idx])
            print(f"  Layer {layer_idx}: {len(correct_acts[layer_idx])} correct, "
                  f"{len(wrong_acts[layer_idx])} wrong tokens")

    # Probe all layers
    print("\nTraining probes at each layer...")
    start = time.time()
    results = probe_all_layers(
        correct_acts, wrong_acts,
        hidden_dim=1024,
        n_epochs=200,
        lr=0.005,
        model_name=model_name,
        layers=probe_layers,
    )
    elapsed = time.time() - start

    # Print results
    sep = "=" * 60
    print(f"\n{sep}")
    print("EXPERIMENT 24 RESULTS: Multi-Layer Probing")
    print(sep)
    print(results.summary())
    print(f"\nTraining time: {elapsed:.1f}s")
    print(sep)

    # Compare to final layer baseline
    final_layer = probe_layers[-1]
    final_acc = next((r.test_accuracy for r in results.layer_results if r.layer_index == final_layer), 0)
    best_acc = results.best_test_accuracy

    if results.best_layer != final_layer and best_acc > final_acc + 0.02:
        print(f"VERDICT: ✅ Layer {results.best_layer} ({best_acc:.1%}) beats final layer ({final_acc:.1%})")
        print("  Intermediate layers retain hallucination signal!")
    elif best_acc > final_acc:
        print(f"VERDICT: ⚠️ Small difference: layer {results.best_layer} ({best_acc:.1%}) vs final ({final_acc:.1%})")
    else:
        print(f"VERDICT: ❌ Final layer ({final_acc:.1%}) is already best or tied")

    return 0


if __name__ == "__main__":
    sys.exit(main())
