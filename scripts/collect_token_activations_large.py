#!/usr/bin/env python3
"""Collect 20,000+ per-token activations from Qwen3-0.6B on 1000+ QA pairs.

Uses ROCm GPU (3.3x speedup) and the generated QA dataset.

Usage:
    sg render -c 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
        .venv/bin/python scripts/collect_token_activations_large.py'
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

QA_DATASET = os.path.join(os.path.dirname(__file__), "..", "data", "qa_dataset_1000.json")
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data", "token_activations_large.safetensors")


def check_answer(response: str, expected: str) -> bool:
    return expected.lower() in response.lower().strip()


def main() -> int:
    import torch
    from safetensors.numpy import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load QA dataset
    with open(QA_DATASET) as f:
        qa_pairs = json.load(f)
    print(f"Loaded {len(qa_pairs)} QA pairs from {QA_DATASET}")

    # Load model on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        output_hidden_states=True,
        dtype=torch.float16 if device == "cuda" else None,
    )
    if device == "cuda":
        model = model.cuda()
    model.eval()
    print(f"Loaded in {time.time()-start:.1f}s")

    # Collect activations
    all_token_ids = []
    all_activations = []
    all_labels = []
    all_question_ids = []
    n_correct = 0
    n_wrong = 0

    for qi, pair in enumerate(qa_pairs):
        question = pair["question"]
        expected = pair.get("expected_answer_substring", pair.get("expected", ""))

        prompt = f"Answer in one word or number only. {question}"
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
            )

        generated_ids = outputs[0, prompt_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        is_correct = check_answer(response, expected)

        if is_correct:
            n_correct += 1
        else:
            n_wrong += 1

        # Get hidden states for full sequence
        with torch.no_grad():
            hidden_out = model(outputs)
            hs = hidden_out.hidden_states

        # Extract per-token activations from last layer (generated tokens only)
        last_layer = hs[-1][0, prompt_len:, :].float().cpu().numpy()
        gen_ids = generated_ids.cpu().numpy()

        for t in range(len(gen_ids)):
            all_token_ids.append(int(gen_ids[t]))
            all_activations.append(last_layer[t])
            all_labels.append(1 if is_correct else 0)
            all_question_ids.append(qi)

        if (qi + 1) % 50 == 0:
            total = len(all_token_ids)
            print(f"  [{qi+1:4d}/{len(qa_pairs)}] "
                  f"tokens={total:6d} "
                  f"correct={n_correct} wrong={n_wrong} "
                  f"({n_correct/(qi+1)*100:.0f}%)")

    # Save
    total_tokens = len(all_token_ids)
    act_dim = all_activations[0].shape[0] if all_activations else 0

    print(f"\n{'='*60}")
    print(f"Collection complete:")
    print(f"  Questions: {len(qa_pairs)}")
    print(f"  Model accuracy: {n_correct}/{len(qa_pairs)} ({n_correct/len(qa_pairs)*100:.0f}%)")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Activation dim: {act_dim}")
    print(f"  Correct tokens: {sum(all_labels)}")
    print(f"  Wrong tokens: {total_tokens - sum(all_labels)}")

    save_file(
        {
            "token_ids": np.array(all_token_ids, dtype=np.int32),
            "activations": np.stack(all_activations).astype(np.float32),
            "labels": np.array(all_labels, dtype=np.int32),
            "question_ids": np.array(all_question_ids, dtype=np.int32),
        },
        OUTPUT,
    )
    print(f"  Saved to: {OUTPUT}")
    print(f"  File size: {os.path.getsize(OUTPUT) / 1e6:.1f} MB")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
