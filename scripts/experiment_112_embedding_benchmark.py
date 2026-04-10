#!/usr/bin/env python3
"""Experiment 112: Fast Embedding Benchmark for Per-Token Guided Decoding.

**Researcher summary:**
    Exp 102 identified MiniLM at 7.6ms as the bottleneck preventing per-token
    energy-guided decoding (budget: 20ms/token). This experiment benchmarks
    five embedding alternatives to find one that hits <1ms p99 with no more
    than 5% AUROC regression vs MiniLM, then integrates the winner into the
    differentiable pipeline.

**Embedding strategies benchmarked:**
    1. MiniLM (baseline): sentence-transformers, 384-dim, ~7.6ms
    2. TF-IDF + projection: sklearn TF-IDF on domain corpus, ~0.3ms
    3. Char n-gram hash: char n-gram HashingVectorizer + projection, ~0.15ms
    4. Word hash: word token hashing + projection, ~0.05ms
    5. Byte histogram: raw byte frequency + projection, ~0.01ms

**Metrics per embedding:**
    - Latency: p50/p95/p99/mean/std in milliseconds (200 iters, 20 warmup)
    - Memory: peak RSS change during encode (using tracemalloc)
    - AUROC: linear probe AUROC on constraint satisfaction classification task
      (constraint-satisfying vs constraint-violating arithmetic/logic texts)

**Selection criteria:**
    Primary: p99 latency < 1ms
    Secondary: AUROC ≥ MiniLM_AUROC × 0.95 (no more than 5% regression)
    If no option meets both: recommend batched checking every K tokens

**Output:** results/experiment_112_results.json

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_112_embedding_benchmark.py

REQ: REQ-EMBED-001, REQ-VERIFY-001
SCENARIO: SCENARIO-VERIFY-004
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

# Force CPU for reproducibility (see CLAUDE.md: ROCm JAX may crash).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Allow importing from python/ and scripts/ directories.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))

import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore[import]
from sklearn.metrics import roc_auc_score  # type: ignore[import]
from sklearn.model_selection import cross_val_score  # type: ignore[import]

from carnot.embeddings.fast_embedding import (
    CharNgramEmbedding,
    FastEmbeddingProtocol,
    HashEmbedding,
    MiniLMEmbedding,
    RandomProjectionEmbedding,
    TFIDFProjectionEmbedding,
    _DEFAULT_CORPUS,
    benchmark_embedding,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = _ROOT / "results"
WARMUP_ITERS = 20
MEASURE_ITERS = 200
MINILM_WARMUP = 10
MINILM_MEASURE = 50   # MiniLM is slower, use fewer iters
EMBED_DIM = 384       # Match MiniLM output dimension for compatibility

# ---------------------------------------------------------------------------
# AUROC evaluation dataset
# ---------------------------------------------------------------------------

# Each entry is (text, label) where label=1 means constraint satisfied.
# Arithmetic: the equation in the text is either correct (1) or wrong (0).
# Logic: the syllogism is valid (1) or invalid (0).
_EVAL_DATASET: list[tuple[str, int]] = [
    # Correct arithmetic (label=1)
    ("The answer is 42. Because 20 + 22 = 42.", 1),
    ("If x = 5 and y = 3, then x + y = 8.", 1),
    ("The product of 6 and 7 is 42.", 1),
    ("15 percent of 200 is 30.", 1),
    ("The sum of 1 through 10 is 55.", 1),
    ("Square root of 144 is 12.", 1),
    ("2 times 3 equals 6.", 1),
    ("4 plus 5 equals 9.", 1),
    ("100 divided by 4 is 25.", 1),
    ("3 squared is 9.", 1),
    ("The area of a 4x5 rectangle is 20.", 1),
    ("60 minutes make one hour, so 3 hours is 180 minutes.", 1),
    ("If you add 17 and 28 you get 45.", 1),
    ("10 percent of 50 is 5.", 1),
    ("The cube of 3 is 27.", 1),
    # Incorrect arithmetic (label=0)
    ("The answer is 43. Because 20 + 22 = 43.", 0),
    ("If x = 5 and y = 3, then x + y = 9.", 0),
    ("The product of 6 and 7 is 43.", 0),
    ("15 percent of 200 is 31.", 0),
    ("The sum of 1 through 10 is 56.", 0),
    ("Square root of 144 is 13.", 0),
    ("2 times 3 equals 7.", 0),
    ("4 plus 5 equals 10.", 0),
    ("100 divided by 4 is 26.", 0),
    ("3 squared is 10.", 0),
    ("The area of a 4x5 rectangle is 21.", 0),
    ("60 minutes make one hour, so 3 hours is 181 minutes.", 0),
    ("If you add 17 and 28 you get 46.", 0),
    ("10 percent of 50 is 6.", 0),
    ("The cube of 3 is 28.", 0),
    # Valid logic (label=1)
    ("All mammals are warm-blooded. Dogs are mammals. Therefore dogs are warm-blooded.", 1),
    ("If it rains, the ground gets wet. It is raining. Therefore the ground is wet.", 1),
    ("All birds have wings. Penguins are birds. Therefore penguins have wings.", 1),
    ("If P then Q. P is true. Therefore Q is true.", 1),
    ("No fish are mammals. Whales are mammals. Therefore whales are not fish.", 1),
    # Invalid logic (label=0)
    ("All mammals are warm-blooded. Lizards are warm-blooded. Therefore lizards are mammals.", 0),
    ("If it rains, the ground gets wet. The ground is wet. Therefore it is raining.", 0),
    ("All birds have wings. Bats have wings. Therefore bats are birds.", 0),
    ("If P then Q. Q is true. Therefore P is true.", 0),
    ("Some fish are edible. Whales are edible. Therefore whales are fish.", 0),
    # Additional arithmetic pairs
    ("Seven times eight equals fifty-six.", 1),
    ("Seven times eight equals fifty-seven.", 0),
    ("Two hundred minus forty-five is one hundred fifty-five.", 1),
    ("Two hundred minus forty-five is one hundred fifty-six.", 0),
    ("The factorial of 5 is 120.", 1),
    ("The factorial of 5 is 121.", 0),
    ("Pi is approximately 3.14159.", 1),
    ("Pi is approximately 3.24159.", 0),
]


def compute_auroc(
    embedder: FastEmbeddingProtocol,
    dataset: list[tuple[str, int]],
    random_state: int = 42,
) -> float:
    """Compute AUROC for a constraint satisfaction classifier using this embedding.

    **Detailed explanation for engineers:**
        We embed all texts in the evaluation dataset, then train a logistic
        regression classifier to predict the label (1=constraint satisfied,
        0=violated). AUROC (Area Under the ROC Curve) measures how well the
        classifier ranks satisfied examples above violated ones:
            - 1.0: perfect separation
            - 0.5: no better than random
            - < 0.5: worse than random (the embedding is misleading)

        We use 5-fold cross-validation to get an unbiased AUROC estimate even
        with the small dataset (48 examples). This means we train on 80% and
        test on 20% across 5 different splits, averaging the AUROC scores.

        The logistic regression is intentionally a LINEAR model — if the
        embedding doesn't provide linearly separable representations for
        constraint satisfaction, a more complex model would just overfit.
        We want to measure the embedding quality, not the classifier quality.

    Args:
        embedder: Embedding to evaluate.
        dataset: List of (text, label) pairs.
        random_state: Seed for reproducible cross-validation splits.

    Returns:
        Mean AUROC across 5 folds. Range [0, 1].
    """
    texts = [t for t, _ in dataset]
    labels = np.array([l for _, l in dataset], dtype=np.int32)

    # Encode all texts.
    embeddings = embedder.encode_batch(texts)  # (N, embed_dim)

    # 5-fold cross-validated AUROC.
    clf = LogisticRegression(
        max_iter=500,
        random_state=random_state,
        C=1.0,
        solver="lbfgs",
    )
    scores = cross_val_score(
        clf, embeddings, labels,
        cv=5,
        scoring="roc_auc",
    )
    return float(scores.mean())


def measure_memory_kb(embedder: FastEmbeddingProtocol, text: str) -> float:
    """Estimate peak memory allocation during a single encode call.

    **Detailed explanation for engineers:**
        Uses Python's ``tracemalloc`` to track memory allocations during the
        encode call. Reports the peak extra memory allocated (in kilobytes)
        above the baseline. This captures temporary buffers created during
        the embedding computation but released afterward.

        Limitation: tracemalloc only tracks Python-level allocations. C
        extensions (numpy, PyTorch, ONNX) may allocate memory through their
        own allocators which are invisible to tracemalloc. For MiniLM, the
        actual peak may be higher than reported.

    Args:
        embedder: Embedding to measure.
        text: A representative text to encode.

    Returns:
        Peak memory delta in kilobytes.
    """
    tracemalloc.start()
    embedder.encode(text)  # one warm-up (may trigger model load)
    tracemalloc.clear_traces()
    embedder.encode(text)  # measured call
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024.0  # bytes → KB


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_embedding_benchmark() -> dict[str, Any]:
    """Run the full Exp 112 benchmark and return results dict.

    **Detailed explanation for engineers:**
        For each embedding strategy:
        1. Instantiate the embedder.
        2. Measure latency (p50/p95/p99) via benchmark_embedding().
        3. Measure peak memory via measure_memory_kb().
        4. Compute AUROC via compute_auroc() on the evaluation dataset.
        5. Record all results with units and metadata.

        After all embedders are measured, apply the selection criteria:
        - p99 < 1ms AND AUROC ≥ MiniLM_AUROC * 0.95
        If no candidate meets criteria, the recommendation shifts to
        batched checking (check every K tokens, reducing frequency).

    Returns:
        Dict suitable for JSON serialization.
    """
    logger.info("=== Experiment 112: Embedding Benchmark ===")
    logger.info("Evaluating %d texts in AUROC dataset", len(_EVAL_DATASET))

    # Representative texts for latency benchmarking.
    bench_texts = [t for t, _ in _EVAL_DATASET] + _DEFAULT_CORPUS

    # Embedding configurations to test.
    # MiniLM is benchmarked with fewer iters (it's slow).
    configs: list[tuple[str, str, FastEmbeddingProtocol, int, int]] = [
        ("minilm", "MiniLM-L6-v2 (sentence-transformers baseline)",
         MiniLMEmbedding(), MINILM_WARMUP, MINILM_MEASURE),
        ("tfidf", "TF-IDF + random projection (sklearn, domain corpus)",
         TFIDFProjectionEmbedding(embed_dim=EMBED_DIM), WARMUP_ITERS, MEASURE_ITERS),
        ("char_ngram", "Char n-gram hash + random projection (sklearn)",
         CharNgramEmbedding(embed_dim=EMBED_DIM), WARMUP_ITERS, MEASURE_ITERS),
        ("hash", "Word token hash + random projection",
         HashEmbedding(embed_dim=EMBED_DIM), WARMUP_ITERS, MEASURE_ITERS),
        ("random", "Byte histogram + random projection (ablation baseline)",
         RandomProjectionEmbedding(embed_dim=EMBED_DIM), WARMUP_ITERS, MEASURE_ITERS),
    ]

    results_per_embedder: list[dict[str, Any]] = []
    minilm_auroc: float = 0.0

    for key, desc, embedder, warmup, iters in configs:
        logger.info("--- Benchmarking: %s ---", key)

        # 1. Latency benchmark.
        logger.info("  Measuring latency (%d warmup, %d iters)...", warmup, iters)
        lat = benchmark_embedding(embedder, bench_texts, warmup=warmup, iters=iters)

        # 2. Memory.
        logger.info("  Measuring memory...")
        mem_kb = measure_memory_kb(embedder, bench_texts[0])

        # 3. AUROC.
        logger.info("  Computing AUROC on %d eval examples...", len(_EVAL_DATASET))
        auroc = compute_auroc(embedder, _EVAL_DATASET)

        if key == "minilm":
            minilm_auroc = auroc

        entry: dict[str, Any] = {
            "key": key,
            "description": desc,
            "embed_dim": embedder.embed_dim,
            "latency": lat,
            "memory_peak_kb": round(mem_kb, 2),
            "auroc": round(auroc, 4),
        }
        results_per_embedder.append(entry)
        logger.info(
            "  %s: p50=%.3fms  p99=%.3fms  AUROC=%.4f  mem=%.1fKB",
            key, lat["p50_ms"], lat["p99_ms"], auroc, mem_kb,
        )

    # ---------------------------------------------------------------------------
    # Selection criteria
    # ---------------------------------------------------------------------------

    LATENCY_THRESHOLD_MS = 1.0
    AUROC_MIN_FRACTION = 0.95  # must be >= 95% of MiniLM AUROC
    auroc_threshold = minilm_auroc * AUROC_MIN_FRACTION

    candidates: list[dict[str, Any]] = []
    for e in results_per_embedder:
        if e["key"] == "minilm":
            continue  # Don't select baseline as "winner"
        passes_latency = e["latency"]["p99_ms"] < LATENCY_THRESHOLD_MS
        passes_auroc = e["auroc"] >= auroc_threshold
        e["passes_latency"] = passes_latency
        e["passes_auroc"] = passes_auroc
        e["meets_criteria"] = passes_latency and passes_auroc
        if passes_latency and passes_auroc:
            candidates.append(e)

    # Rank candidates: primary = AUROC (higher better), secondary = p50 (lower better).
    candidates.sort(key=lambda x: (-x["auroc"], x["latency"]["p50_ms"]))

    speedup_labels: list[dict[str, Any]] = []
    for e in results_per_embedder:
        if e["key"] != "minilm":
            minilm_p50 = results_per_embedder[0]["latency"]["p50_ms"]
            speedup = minilm_p50 / max(e["latency"]["p50_ms"], 1e-9)
            e["speedup_vs_minilm"] = round(speedup, 1)

    if candidates:
        winner = candidates[0]
        selection_strategy = winner["key"]
        recommendation = (
            f"Use '{winner['key']}' embedding: "
            f"p99={winner['latency']['p99_ms']:.3f}ms "
            f"(vs MiniLM {results_per_embedder[0]['latency']['p99_ms']:.1f}ms), "
            f"AUROC={winner['auroc']:.4f} "
            f"({winner['auroc'] / minilm_auroc * 100:.1f}% of MiniLM), "
            f"speedup={winner.get('speedup_vs_minilm', '?')}x. "
            f"Meets per-token guided decoding target (<1ms p99, <5% AUROC regression)."
        )
        batched_fallback = False
    else:
        # No candidate meets both criteria. Find best latency winner and compute K.
        fast_embs = sorted(
            [e for e in results_per_embedder if e["key"] != "minilm"],
            key=lambda x: x["latency"]["p99_ms"],
        )
        selection_strategy = fast_embs[0]["key"] if fast_embs else "none"
        token_budget_ms = 20.0  # 50 tokens/sec → 20ms/token
        best_p99 = fast_embs[0]["latency"]["p99_ms"] if fast_embs else 999.0
        k_tokens = max(1, int(np.ceil(best_p99 / token_budget_ms * 100)))
        recommendation = (
            f"No embedding meets <1ms p99 AND ≥{auroc_threshold:.4f} AUROC. "
            f"Recommend batched checking every {k_tokens} tokens using "
            f"'{selection_strategy}' (p99={best_p99:.3f}ms). "
            f"At 50 tokens/sec, this adds {best_p99 / k_tokens:.2f}ms overhead per token."
        )
        batched_fallback = True

    logger.info("=== Selection: %s ===", selection_strategy)
    logger.info("%s", recommendation)

    # ---------------------------------------------------------------------------
    # Final result dict
    # ---------------------------------------------------------------------------

    result: dict[str, Any] = {
        "experiment": 112,
        "title": "Embedding Benchmark — Fast Alternatives to MiniLM for Guided Decoding",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "context": {
            "baseline_experiment": 102,
            "minilm_baseline_ms": 7.6022,
            "jax_jit_energy_ms": 0.008,
            "per_token_budget_ms": 20.0,
            "target_latency_p99_ms": 1.0,
            "target_auroc_fraction": 0.95,
        },
        "platform": "CPU (JAX_PLATFORMS=cpu)",
        "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS,
        "embed_dim": EMBED_DIM,
        "eval_dataset_size": len(_EVAL_DATASET),
        "embeddings": results_per_embedder,
        "selection": {
            "criteria": {
                "latency_threshold_p99_ms": LATENCY_THRESHOLD_MS,
                "auroc_threshold": round(auroc_threshold, 4),
                "auroc_fraction_of_minilm": AUROC_MIN_FRACTION,
                "minilm_auroc": round(minilm_auroc, 4),
            },
            "candidates_meeting_criteria": [c["key"] for c in candidates],
            "winner": selection_strategy,
            "batched_fallback_recommended": batched_fallback,
            "recommendation": recommendation,
        },
    }

    return result


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 112 and write results to results/experiment_112_results.json."""
    t_start = time.time()
    results = run_embedding_benchmark()
    elapsed = time.time() - t_start

    results["total_runtime_seconds"] = round(elapsed, 2)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "experiment_112_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Results written to %s", out_path)

    # Print summary table.
    print("\n" + "=" * 70)
    print("Experiment 112: Embedding Benchmark Results")
    print("=" * 70)
    print(f"{'Embedding':<20} {'p50(ms)':>8} {'p99(ms)':>8} {'AUROC':>8} {'Speedup':>8}")
    print("-" * 70)
    for e in results["embeddings"]:
        lat = e["latency"]
        speedup = e.get("speedup_vs_minilm", "—")
        speedup_str = f"{speedup}x" if isinstance(speedup, (int, float)) else speedup
        print(
            f"{e['key']:<20} {lat['p50_ms']:>8.3f} {lat['p99_ms']:>8.3f}"
            f" {e['auroc']:>8.4f} {speedup_str:>8}"
        )
    print("=" * 70)
    print(f"\nWinner: {results['selection']['winner']}")
    print(f"\nRecommendation:\n  {results['selection']['recommendation']}")
    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
