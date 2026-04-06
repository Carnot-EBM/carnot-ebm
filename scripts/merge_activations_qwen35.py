#!/usr/bin/env python3
"""Merge QA + TruthfulQA activations (both from Qwen3.5-0.8B).

Combines:
  - data/token_activations_qa_qwen35.safetensors (QA, question_ids 0-1339)
  - data/token_activations_combined.safetensors (TruthfulQA portion, question_ids >= 10000)

Output: data/token_activations_qwen35_merged.safetensors

Usage:
    .venv/bin/python scripts/merge_activations_qwen35.py
"""

import os
import sys

import numpy as np
from safetensors.numpy import load_file, save_file

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
QA_FILE = os.path.join(DATA_DIR, "token_activations_qa_qwen35.safetensors")
TQA_FILE = os.path.join(DATA_DIR, "token_activations_combined.safetensors")
OUTPUT = os.path.join(DATA_DIR, "token_activations_qwen35_merged.safetensors")


def main():
    # Load QA data (all from Qwen3.5-0.8B)
    print("Loading QA activations (Qwen3.5-0.8B)...")
    qa = load_file(QA_FILE)
    qa_n = len(qa["labels"])
    print(f"  QA: {qa_n} tokens, correct={qa['labels'].sum()}, wrong={qa_n - qa['labels'].sum()}")

    # Load TruthfulQA portion from combined file
    print("Loading TruthfulQA activations (Qwen3.5-0.8B)...")
    combined = load_file(TQA_FILE)
    tqa_mask = combined["question_ids"] >= 10000
    tqa_n = int(tqa_mask.sum())
    print(f"  TruthfulQA: {tqa_n} tokens, correct={combined['labels'][tqa_mask].sum()}, "
          f"wrong={tqa_n - combined['labels'][tqa_mask].sum()}")

    # Merge
    merged = {
        "token_ids": np.concatenate([qa["token_ids"], combined["token_ids"][tqa_mask]]),
        "activations": np.concatenate([qa["activations"], combined["activations"][tqa_mask]]),
        "labels": np.concatenate([qa["labels"], combined["labels"][tqa_mask]]),
        "question_ids": np.concatenate([qa["question_ids"], combined["question_ids"][tqa_mask]]),
    }

    total = len(merged["labels"])
    correct = int(merged["labels"].sum())
    print(f"\nMerged: {total} tokens (correct={correct}, wrong={total - correct})")
    print(f"  Dim: {merged['activations'].shape[1]}")

    save_file(merged, OUTPUT)
    print(f"  Saved to: {OUTPUT}")
    print(f"  File size: {os.path.getsize(OUTPUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
