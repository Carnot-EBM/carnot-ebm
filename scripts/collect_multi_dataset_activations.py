#!/usr/bin/env python3
"""Collect per-token activations from multiple QA datasets.

Pulls questions from TruthfulQA, MMLU, SimpleQA, and HaluEval,
generates answers at multiple temperatures, and collects labeled
activation data for EBM training.

Datasets:
  - TruthfulQA (generation split): 817 adversarial questions
  - MMLU (test split): ~14K questions across 57 domains (sample N)
  - SimpleQA: ~4K straightforward factual QA
  - HaluEval (qa split): hallucination-focused QA pairs

Each question gets responses at configurable temperatures.
Labels are determined by answer-checking against ground truth.

Usage:
    sg render -c 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
        PYTHONUNBUFFERED=1 .venv/bin/python scripts/collect_multi_dataset_activations.py \
        --model Qwen/Qwen3.5-0.8B --n-per-dataset 200 --temps 0.0,0.7'

    # Full collection (slow):
    sg render -c '... --n-per-dataset 1000 --temps 0.0,0.3,0.5,0.7,0.9,1.0'
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def check_answer_substring(response: str, expected: str) -> bool:
    """Check if expected answer appears as substring in response."""
    return expected.lower().strip() in response.lower().strip()


def check_truthfulqa_answer(
    response: str, correct_answers: list[str],
    incorrect_answers: list[str], best_answer: str = "",
) -> bool:
    """TruthfulQA-specific answer checking with multiple strategies."""
    response_lower = response.lower().strip()
    for correct in correct_answers:
        if correct.lower() in response_lower:
            return True
    if best_answer:
        words = best_answer.lower().split()
        if len(words) >= 3:
            for i in range(len(words) - 2):
                phrase = " ".join(words[i:i + 3])
                if phrase in response_lower:
                    return True
    for incorrect in incorrect_answers:
        if incorrect.lower() in response_lower:
            return False
    return False


def check_mmlu_answer(response: str, correct_letter: str, choices: list[str]) -> bool:
    """Check MMLU answer — look for correct letter or full answer text."""
    response_lower = response.lower().strip()
    # Check for letter match (A, B, C, D)
    if correct_letter.lower() in response_lower[:5]:
        return True
    # Check for full answer text
    correct_idx = ord(correct_letter.upper()) - ord("A")
    if 0 <= correct_idx < len(choices):
        if choices[correct_idx].lower() in response_lower:
            return True
    return False


def load_truthfulqa(n: int) -> list[dict]:
    """Load TruthfulQA questions."""
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "generation")
    questions = []
    for ex in list(ds["validation"])[:n]:
        questions.append({
            "source": "truthfulqa",
            "question": ex["question"],
            "check_fn": lambda resp, ex=ex: check_truthfulqa_answer(
                resp, ex["correct_answers"], ex["incorrect_answers"],
                ex.get("best_answer", ""),
            ),
        })
    print(f"  TruthfulQA: {len(questions)} questions")
    return questions


def load_mmlu(n: int) -> list[dict]:
    """Load MMLU questions (sample across subjects)."""
    from datasets import load_dataset

    try:
        ds = load_dataset("cais/mmlu", "all", split="test")
    except Exception:
        try:
            ds = load_dataset("lukaemon/mmlu", "all", split="test")
        except Exception:
            print("  MMLU: SKIP — dataset not available")
            return []

    # Sample evenly
    indices = np.random.default_rng(42).permutation(len(ds))[:n]
    questions = []
    for idx in indices:
        ex = ds[int(idx)]
        q_text = ex["question"]
        choices = ex["choices"] if isinstance(ex["choices"], list) else [ex.get(f"choice_{i}", "") for i in range(4)]
        answer_raw = ex["answer"]
        correct = answer_raw if isinstance(answer_raw, str) else chr(ord("A") + int(answer_raw))
        formatted = f"{q_text}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
        questions.append({
            "source": "mmlu",
            "question": formatted,
            "check_fn": lambda resp, c=correct, ch=choices: check_mmlu_answer(resp, c, ch),
        })
    print(f"  MMLU: {len(questions)} questions")
    return questions


def load_simpleqa(n: int) -> list[dict]:
    """Load SimpleQA questions (google/simpleqa-verified)."""
    from datasets import load_dataset

    try:
        ds = load_dataset("google/simpleqa-verified", split="eval")
        indices = np.random.default_rng(42).permutation(len(ds))[:n]
        questions = []
        for idx in indices:
            ex = ds[int(idx)]
            q = ex.get("problem", "")
            a = ex.get("answer", "")
            if q and a:
                questions.append({
                    "source": "simpleqa",
                    "question": q,
                    "check_fn": lambda resp, expected=a: check_answer_substring(resp, expected),
                })
        print(f"  SimpleQA: {len(questions)} questions")
        return questions
    except Exception as e:
        print(f"  SimpleQA: SKIP — {e}")
        return []


def load_halueval(n: int) -> list[dict]:
    """Load HaluEval QA questions."""
    from datasets import load_dataset

    try:
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
        indices = np.random.default_rng(42).permutation(len(ds))[:n]
        questions = []
        for idx in indices:
            ex = ds[int(idx)]
            q = ex.get("question", "")
            a = ex.get("answer", ex.get("right_answer", ""))
            if q and a:
                questions.append({
                    "source": "halueval",
                    "question": q,
                    "check_fn": lambda resp, expected=a: check_answer_substring(resp, expected),
                })
        print(f"  HaluEval: {len(questions)} questions")
        return questions
    except Exception as e:
        print(f"  HaluEval: SKIP — {e}")
        return []


def collect_activations(
    model_name: str,
    questions: list[dict],
    temperatures: list[float],
    enable_thinking: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate answers and collect activations.

    Returns (activations, labels, temp_ids, sources).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        output_hidden_states=True,
        torch_dtype="auto" if device == "cuda" else None,
    )
    if device == "cuda":
        model = model.cuda()
    model.eval()

    config = model.config
    if hasattr(config, "get_text_config"):
        tc = config.get_text_config()
    else:
        tc = config
    hidden_dim = getattr(tc, "hidden_size", getattr(tc, "d_model", 1024))
    print(f"  Hidden dim: {hidden_dim}, Device: {device}")

    all_activations = []
    all_labels = []
    all_temps = []
    all_sources = []
    stats: dict[str, dict[str, int]] = {}

    for qi, qdata in enumerate(questions):
        source = qdata["source"]
        question = qdata["question"]
        check_fn = qdata["check_fn"]

        if source not in stats:
            stats[source] = {"correct": 0, "wrong": 0}

        prompt = f"Answer briefly and factually in one sentence. {question}"

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            kwargs = {}
            if not enable_thinking:
                try:
                    kwargs["enable_thinking"] = False
                except TypeError:
                    pass
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **kwargs,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        for temp in temperatures:
            gen_kwargs = {"max_new_tokens": 80, "pad_token_id": tokenizer.eos_token_id}
            if temp == 0.0:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temp
                gen_kwargs["top_p"] = 0.95

            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)

                gen_ids = outputs[0, prompt_len:]
                response = tokenizer.decode(gen_ids, skip_special_tokens=True)
                if "</think>" in response:
                    response = response.split("</think>")[-1].strip()

                is_correct = check_fn(response)

                with torch.no_grad():
                    ho = model(outputs, output_hidden_states=True)
                    hs = ho.hidden_states

                last_layer = hs[-1][0, prompt_len:, :].float().cpu().numpy()

                for t in range(len(gen_ids)):
                    all_activations.append(last_layer[t])
                    all_labels.append(1 if is_correct else 0)
                    all_temps.append(temp)
                    all_sources.append(source)

                if is_correct:
                    stats[source]["correct"] += 1
                else:
                    stats[source]["wrong"] += 1

            except Exception:
                stats[source]["wrong"] += 1

        if (qi + 1) % 100 == 0:
            n_tok = len(all_activations)
            print(f"  [{qi+1:4d}/{len(questions)}] tokens={n_tok}")
            for src, s in stats.items():
                total = s["correct"] + s["wrong"]
                acc = s["correct"] / total * 100 if total > 0 else 0
                print(f"    {src}: {s['correct']}/{total} ({acc:.0f}%)")

    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return (
        np.array(all_activations, dtype=np.float32),
        np.array(all_labels, dtype=np.int32),
        np.array(all_temps, dtype=np.float32),
        all_sources,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect activations from multiple datasets")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="HuggingFace model name")
    parser.add_argument("--n-per-dataset", type=int, default=200, help="Questions per dataset")
    parser.add_argument("--temps", default="0.0,0.7", help="Comma-separated temperatures")
    parser.add_argument("--output-suffix", default="", help="Output file suffix")
    args = parser.parse_args()

    temperatures = [float(t) for t in args.temps.split(",")]
    model_short = args.model.split("/")[-1].lower().replace(".", "").replace("-", "")

    print("=" * 70)
    print(f"MULTI-DATASET ACTIVATION COLLECTION")
    print(f"  Model: {args.model}")
    print(f"  Questions per dataset: {args.n_per_dataset}")
    print(f"  Temperatures: {temperatures}")
    print("=" * 70)

    # Load all datasets
    print("\nLoading datasets...")
    all_questions = []
    all_questions.extend(load_truthfulqa(args.n_per_dataset))
    all_questions.extend(load_mmlu(args.n_per_dataset))
    all_questions.extend(load_simpleqa(args.n_per_dataset))
    all_questions.extend(load_halueval(args.n_per_dataset))

    # Shuffle
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(all_questions))
    all_questions = [all_questions[i] for i in indices]

    print(f"\nTotal questions: {len(all_questions)}")

    # Collect activations
    start = time.time()
    activations, labels, temp_ids, sources = collect_activations(
        args.model, all_questions, temperatures,
    )
    elapsed = time.time() - start

    # Save
    from safetensors.numpy import save_file

    suffix = args.output_suffix or f"_{model_short}_multi"
    output_file = os.path.join(DATA_DIR, f"token_activations{suffix}.safetensors")

    # Encode sources as integers
    source_names = sorted(set(sources))
    source_map = {name: i for i, name in enumerate(source_names)}
    source_ids = np.array([source_map[s] for s in sources], dtype=np.int32)

    save_file({
        "activations": activations,
        "labels": labels,
        "temperatures": temp_ids,
        "source_ids": source_ids,
    }, output_file)

    # Summary
    n_total = len(labels)
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"COLLECTION COMPLETE ({elapsed:.0f}s)")
    print(sep)
    print(f"  Total tokens: {n_total}")
    print(f"  Correct: {int(labels.sum())}, Wrong: {n_total - int(labels.sum())}")
    print(f"  File: {output_file} ({os.path.getsize(output_file) / 1e6:.1f} MB)")
    print(f"")
    print(f"  Per source:")
    for name in source_names:
        mask = source_ids == source_map[name]
        n = int(mask.sum())
        c = int(labels[mask].sum())
        print(f"    {name}: {n} tokens, {c} correct, {n - c} wrong")
    print(f"")
    print(f"  Per temperature:")
    for t in sorted(set(temp_ids)):
        mask = temp_ids == t
        n = int(mask.sum())
        c = int(labels[mask].sum())
        print(f"    temp={t:.1f}: {n} tokens, {c} correct")
    print(f"")
    print(f"  Source mapping: {source_map}")
    print(sep)

    return 0


if __name__ == "__main__":
    sys.exit(main())
