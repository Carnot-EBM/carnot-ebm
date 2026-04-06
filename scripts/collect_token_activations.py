#!/usr/bin/env python3
"""Collect per-token activations from Qwen3-0.6B for EBM training.

WHY PER-TOKEN: Mean-pooled activations (experiments 9-12) destroyed the
token-level signal that makes logprobs effective at hallucination detection.
This script collects the raw, per-token hidden states so we can train an EBM
that operates at the same granularity as logprobs — one energy score per token.

For each generated token we store:
  - token_id: the vocabulary index of the generated token
  - activation: the last-layer hidden state vector at that token position
  - is_correct: 1 if the full answer was verified correct, 0 otherwise

The "is_correct" label applies to every token in the answer — this is a
sequence-level label projected onto tokens. The EBM will learn which
activation patterns correlate with correct vs. hallucinated completions.

Output: data/token_activations.safetensors
  - token_ids:    int32  [N]           vocabulary indices
  - activations:  float32 [N, D]       last-layer hidden states
  - labels:       int32  [N]           1=correct, 0=hallucinated
  - question_ids: int32  [N]           which question each token came from

Usage:
    python scripts/collect_token_activations.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# The 93 factual QA pairs from experiment_scaled_rejection_sampling.py.
# Each is (question_string, expected_answer_substring).  A generated answer
# is "correct" when expected_answer_substring appears (case-insensitive) in
# the model's decoded output.
# ---------------------------------------------------------------------------
QA_PAIRS = [
    # Math — easy
    ("What is 2+3?", "5"), ("What is 7*8?", "56"), ("What is 100/4?", "25"),
    ("What is 15-9?", "6"), ("What is 3^3?", "27"), ("What is 50+50?", "100"),
    ("What is 12*12?", "144"), ("What is 81/9?", "9"), ("What is 1000-1?", "999"),
    ("What is 2^10?", "1024"),
    # Math — medium
    ("What is the square root of 169?", "13"), ("What is 17*19?", "323"),
    ("What is 256/16?", "16"), ("What is 99*99?", "9801"),
    ("What is the cube root of 27?", "3"), ("What is 7! (7 factorial)?", "5040"),
    ("What is 2^16?", "65536"), ("What is 1+2+3+4+5+6+7+8+9+10?", "55"),
    # Math — hard (likely hallucination)
    ("What is 37*43?", "1591"), ("What is 123*456?", "56088"),
    ("What is the 20th prime number?", "71"), ("What is 11! (11 factorial)?", "39916800"),
    ("What is 2^20?", "1048576"), ("What is the square root of 1764?", "42"),
    ("What is 97*103?", "9991"), ("What is 19^2?", "361"),
    ("What is 23*29?", "667"), ("What is the 12th Fibonacci number?", "144"),
    # Geography — easy
    ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"),
    ("What is the capital of Spain?", "Madrid"),
    ("What is the capital of France?", "Paris"),
    ("What is the capital of the UK?", "London"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("What is the capital of Russia?", "Moscow"),
    # Geography — medium
    ("What is the capital of South Korea?", "Seoul"),
    ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of Turkey?", "Ankara"),
    ("What is the capital of Thailand?", "Bangkok"),
    ("What is the capital of Argentina?", "Buenos Aires"),
    # Geography — hard
    ("What is the capital of Myanmar?", "Naypyidaw"),
    ("What is the capital of Sri Lanka?", "Colombo"),
    ("What is the capital of Kazakhstan?", "Astana"),
    ("What is the capital of Nigeria?", "Abuja"),
    ("What is the capital of Pakistan?", "Islamabad"),
    # Science — easy
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("What is the chemical symbol for sodium?", "Na"),
    ("How many chromosomes do humans have?", "46"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("What is the atomic number of hydrogen?", "1"),
    ("What is the atomic number of oxygen?", "8"),
    ("What is the speed of sound in m/s approximately?", "343"),
    ("How many planets are in our solar system?", "8"),
    # Science — medium
    ("What is the atomic number of gold?", "79"),
    ("What is the chemical formula for table salt?", "NaCl"),
    ("What is the chemical formula for methane?", "CH4"),
    ("What is Avogadro's number approximately?", "6.022"),
    ("What is the charge of an electron in coulombs?", "1.6"),
    # History — easy
    ("In what year did the Titanic sink?", "1912"),
    ("In what year did WW1 start?", "1914"),
    ("In what year did WW2 end?", "1945"),
    ("In what year did humans first land on the Moon?", "1969"),
    ("In what year was the Declaration of Independence signed?", "1776"),
    # History — medium
    ("In what year did the Berlin Wall fall?", "1989"),
    ("In what year did the French Revolution begin?", "1789"),
    ("In what year did Columbus reach the Americas?", "1492"),
    ("In what year was the Magna Carta signed?", "1215"),
    ("In what year did the Roman Empire fall?", "476"),
    # General knowledge
    ("How many sides does a hexagon have?", "6"),
    ("How many sides does an octagon have?", "8"),
    ("How many continents are there?", "7"),
    ("How many strings does a standard guitar have?", "6"),
    ("How many cards are in a standard deck?", "52"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is the smallest planet in our solar system?", "Mercury"),
    ("What is the closest star to Earth?", "Sun"),
    ("What is the largest organ in the human body?", "skin"),
    ("How many teeth does an adult human typically have?", "32"),
    # CS / Tech
    ("What year was Python first released?", "1991"),
    ("What year was Java first released?", "1995"),
    ("What does HTML stand for?", "HyperText"),
    ("What does CPU stand for?", "Central Processing"),
    ("What does RAM stand for?", "Random Access"),
    ("Who created Linux?", "Torvalds"),
    ("What company created Java?", "Sun"),
    ("What year was the iPhone first released?", "2007"),
    ("How many bits in a byte?", "8"),
    ("What is the binary representation of 10?", "1010"),
]


def check_answer(response: str, expected: str) -> bool:
    """Case-insensitive substring match — same logic as the experiment scripts."""
    return expected.lower() in response.lower().strip()


def main() -> int:
    print("=" * 60)
    print("COLLECT PER-TOKEN ACTIVATIONS FOR EBM TRAINING")
    print(f"QA pairs: {len(QA_PAIRS)}")
    print("=" * 60)

    import torch
    from safetensors.numpy import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-0.6B"
    print(f"\nLoading {model_name}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # We need hidden states from the forward pass (not generate).
        output_hidden_states=True,
    )
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Accumulators — we append numpy arrays and stack at the end.
    all_token_ids: list[np.ndarray] = []      # [n_tokens_i] int32
    all_activations: list[np.ndarray] = []    # [n_tokens_i, D] float32
    all_labels: list[np.ndarray] = []         # [n_tokens_i] int32
    all_question_ids: list[np.ndarray] = []   # [n_tokens_i] int32

    n_correct = 0
    n_wrong = 0
    total_tokens = 0

    for q_idx, (question, expected) in enumerate(QA_PAIRS):
        prompt = f"Answer in one word or number only. {question}"
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]

        # -----------------------------------------------------------
        # Step 1: Generate an answer (greedy, deterministic).
        # We use generate() because it handles autoregressive decoding,
        # then run a second forward pass on the full sequence to get
        # hidden states at every position.
        # -----------------------------------------------------------
        with torch.no_grad():
            gen_output = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )

        # gen_output is [1, prompt_len + gen_len]
        gen_token_ids = gen_output[0, prompt_len:]  # only generated tokens
        response = tokenizer.decode(gen_token_ids, skip_special_tokens=True)
        is_correct = check_answer(response, expected)
        label_val = 1 if is_correct else 0

        if is_correct:
            n_correct += 1
        else:
            n_wrong += 1

        # -----------------------------------------------------------
        # Step 2: Forward pass on the full sequence to get hidden states.
        # model() with the complete token sequence gives us hidden_states
        # at every position.  We only keep the generated-token positions.
        #
        # hidden_states is a tuple of (n_layers+1) tensors, each shaped
        # [batch=1, seq_len, hidden_dim].  We take the LAST layer (index -1)
        # which is the representation the LM head sees.
        # -----------------------------------------------------------
        with torch.no_grad():
            fwd = model(gen_output)

        # fwd.hidden_states[-1] is the last transformer layer output.
        # Shape: [1, full_seq_len, hidden_dim].
        # We slice out only the generated-token positions.
        last_layer = fwd.hidden_states[-1]  # [1, seq_len, D]
        gen_activations = last_layer[0, prompt_len:, :]  # [gen_len, D]

        # bfloat16 → float32 → numpy
        gen_acts_np = gen_activations.float().cpu().numpy()  # [gen_len, D]
        gen_ids_np = gen_token_ids.cpu().numpy().astype(np.int32)  # [gen_len]
        n_gen = gen_acts_np.shape[0]

        all_token_ids.append(gen_ids_np)
        all_activations.append(gen_acts_np)
        all_labels.append(np.full(n_gen, label_val, dtype=np.int32))
        all_question_ids.append(np.full(n_gen, q_idx, dtype=np.int32))
        total_tokens += n_gen

        icon = "✓" if is_correct else "✗"
        if (q_idx + 1) % 10 == 0 or q_idx == len(QA_PAIRS) - 1:
            print(
                f"  [{q_idx+1:3d}/{len(QA_PAIRS)}] {icon} "
                f"tokens={n_gen:2d}  total={total_tokens:5d}  "
                f"q=\"{question[:35]}...\""
            )

    # -----------------------------------------------------------
    # Stack and save
    # -----------------------------------------------------------
    token_ids = np.concatenate(all_token_ids)        # [N]
    activations = np.concatenate(all_activations)     # [N, D]
    labels = np.concatenate(all_labels)               # [N]
    question_ids = np.concatenate(all_question_ids)   # [N]

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "token_activations.safetensors")
    out_path = os.path.abspath(out_path)

    save_file(
        {
            "token_ids": token_ids,
            "activations": activations,
            "labels": labels,
            "question_ids": question_ids,
        },
        out_path,
    )

    # -----------------------------------------------------------
    # Report
    # -----------------------------------------------------------
    hidden_dim = activations.shape[1]
    print(f"\n{'='*60}")
    print("COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total tokens:     {total_tokens}")
    print(f"  Hidden dim:       {hidden_dim}")
    print(f"  Correct answers:  {n_correct}/{len(QA_PAIRS)} ({100*n_correct/len(QA_PAIRS):.0f}%)")
    print(f"  Wrong answers:    {n_wrong}/{len(QA_PAIRS)} ({100*n_wrong/len(QA_PAIRS):.0f}%)")
    correct_tokens = int(labels.sum())
    wrong_tokens = total_tokens - correct_tokens
    print(f"  Correct tokens:   {correct_tokens}")
    print(f"  Wrong tokens:     {wrong_tokens}")
    ratio = correct_tokens / max(wrong_tokens, 1)
    print(f"  Correct/wrong:    {ratio:.2f}")
    print(f"  Activations dtype: {activations.dtype}")
    print(f"  File size:        {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")
    print(f"  Saved to:         {out_path}")

    if total_tokens < 5000:
        print(f"\n  WARNING: Only {total_tokens} tokens collected (target: 5000+).")
        print("  Consider adding more QA pairs or increasing max_new_tokens.")
    else:
        print(f"\n  Target met: {total_tokens} >= 5000 tokens.")

    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
