#!/usr/bin/env python3
"""Experiment 1084: Step-level PRM training data generation via MCTS-inspired cascade scoring.

**Background:**
    ThinkPRM (exp1033) achieved AUROC 0.9885 on the FoVer corpus using full-step
    hidden-state probing. arXiv 2604.17957 (PRMs Meet Planning, April 2026) shows
    MCTS-based synthetic data generation can create 50k+ step-level examples without
    human annotation by using a verifier's signal at each partial-trajectory step.

    Carnot's cascade (validated end-to-end in exp1073) IS the step scorer. This
    experiment uses the cascade to label partial CoT trajectories — generating training
    data to improve ThinkPRM's step-level discrimination.

**What this experiment does:**
    1. Loads FoVer corpus v4 (6548 entries: question_id, step_text, label, confidence).
    2. Decomposes each step_text into sub-steps (sentence/paragraph splitting).
    3. For each prefix of sub-steps, computes a cascade score (lightweight heuristic
       calibrated to exp1073 energy distribution) and assigns a step label.
    4. Writes labeled examples to data/step_level_prm_training.jsonl.
    5. If >= 2000 non-ambiguous examples are generated, retrains ThinkPRM on a
       sample using Qwen3-0.6B hidden states and evaluates AUROC on a held-out set.

**Honest verdict logic:**
    - "step_data_generated_thinkprm_improved"  : retrain succeeded, AUROC improved
    - "step_data_generated_thinkprm_unchanged" : retrain succeeded, AUROC same/worse
    - "step_data_generated_retrain_skipped"    : generated < 2000 examples or sample
                                                  extraction failed
    - "step_data_insufficient"                 : generated < 100 examples (data problem)
    - "failed"                                 : exception or corpus missing

Spec: REQ-LEARN-011, REQ-VERIFY-098
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure the project root is on the path so carnot imports work
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "python"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

EXPERIMENT_ID = "exp1084_step_level_prm_data_generation"
SCHEMA = "carnot.experiment.v1"
CORPUS_PATH = "data/fover_corpus_v4.json"
TEST_PATH = "data/fover_test_v4.json"
OUTPUT_JSONL = "data/step_level_prm_training.jsonl"
DELIVERABLE = "results/experiment_1084_step_level_prm_data_generation.json"

BASELINE_AUROC = 0.9885  # exp1033 result
RETRAIN_THRESHOLD = 2000  # minimum non-ambiguous examples to attempt retraining

# Sample sizes for retraining (keep small for CPU-only runs)
RETRAIN_SAMPLE = 300  # training examples to draw from generated data
TEST_SAMPLE = 200  # test examples from fover_test_v4.json


def _load_jsonl(path: str, n: int | None = None) -> list[dict]:
    """Load up to n records from a JSONL file."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if n is not None and len(rows) >= n:
                break
    return rows


def _run_retrain(
    generated_jsonl: str,
    test_corpus_path: str,
) -> dict:
    """Retrain ThinkPRM on step-level data and evaluate on held-out FoVer test set.

    Uses Qwen3-0.6B hidden states (real inference, not text features) via the
    ThinkPRMProbe class from exp1033. Retraining uses 10 epochs as specified.

    Returns dict with: auroc_after, n_train, n_test, retrain_ok, error (if any).
    """
    result: dict = {
        "retrain_ok": False,
        "auroc_after": None,
        "n_train": 0,
        "n_test": 0,
        "error": None,
    }
    try:
        import numpy as np
        from carnot.verify.thinkprm_probe import ThinkPRMProbe

        # Load training sample — balanced: equal correct/wrong up to RETRAIN_SAMPLE/2
        all_train = _load_jsonl(generated_jsonl)
        correct_samples = [r for r in all_train if r["step_label"] == "correct"]
        wrong_samples = [r for r in all_train if r["step_label"] == "wrong"]

        # Balance classes to reduce majority-class bias
        half = RETRAIN_SAMPLE // 2
        import random

        random.seed(42)
        train_correct = random.sample(correct_samples, min(half, len(correct_samples)))
        train_wrong = random.sample(wrong_samples, min(half, len(wrong_samples)))
        train_rows = train_correct + train_wrong
        random.shuffle(train_rows)

        train_texts = [r["partial_cot"] for r in train_rows]
        train_labels = np.array(
            [1.0 if r["step_label"] == "correct" else 0.0 for r in train_rows], dtype=np.float64
        )
        result["n_train"] = len(train_rows)

        # Load test sample from original FoVer test corpus
        test_corpus = json.load(open(test_corpus_path))
        import random as rnd

        rnd.seed(0)
        test_sample = rnd.sample(test_corpus, min(TEST_SAMPLE, len(test_corpus)))
        test_texts = [r["step_text"] for r in test_sample]
        test_labels = np.array(
            [1.0 if r["label"] == "correct" else 0.0 for r in test_sample], dtype=np.float64
        )
        result["n_test"] = len(test_sample)

        print(
            f"[exp1084] Retraining ThinkPRM on {result['n_train']} step examples "
            f"(correct={len(train_correct)}, wrong={len(train_wrong)})"
        )
        print(f"[exp1084] Test set: {result['n_test']} examples from fover_test_v4.json")

        # Feature extraction with Qwen3-0.6B via transformers (real hidden-state inference).
        # Explicitly set model_id so ThinkPRMProbe skips the GGUF llama_cpp path (which
        # returns a None tokenizer and breaks the transformers batch_encode API).
        probe = ThinkPRMProbe(model_id="Qwen/Qwen3-0.6B", n_pca_dims=16, seed=42)
        # Monkey-patch _find_gemma31b_gguf to return None: force transformers path
        probe._find_gemma31b_gguf = lambda: None  # noqa: E731
        all_texts = train_texts + test_texts
        print(f"[exp1084] Extracting hidden states for {len(all_texts)} texts ...")
        X_all = probe.fit_features(all_texts, batch_size=16, max_length=128)

        X_train = X_all[: result["n_train"]]
        X_test = X_all[result["n_train"] :]

        # Retrain with 10 epochs (quick check, not full convergence)
        print("[exp1084] Training LogisticProbe (10 epochs) ...")
        probe.fit_classifier(X_train, train_labels, n_epochs=10, lr=0.05, reg=0.01)

        auroc = probe.auroc(X_test, test_labels)
        result["auroc_after"] = round(float(auroc), 4)
        result["retrain_ok"] = True
        print(f"[exp1084] Retrain AUROC: {auroc:.4f} (baseline: {BASELINE_AUROC})")

    except Exception as exc:
        result["error"] = str(exc)
        print(f"[exp1084] Retrain failed: {exc}")

    return result


def main() -> None:
    """Run the experiment end-to-end."""
    run_start = time.time()
    print(f"[exp1084] Starting {EXPERIMENT_ID}")

    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Verify corpus is present
    # -----------------------------------------------------------------------
    if not Path(CORPUS_PATH).exists():
        artifact = {
            "experiment": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": _utcnow(),
            "duration_s": round(time.time() - run_start, 2),
            "honest_verdict": "failed",
            "error": f"FoVer corpus not found: {CORPUS_PATH}",
            "n_fover_pairs_processed": 0,
            "n_step_examples_generated": 0,
            "n_correct_step_examples": 0,
            "n_wrong_step_examples": 0,
            "n_ambiguous_excluded": 0,
            "output_file": OUTPUT_JSONL,
            "thinkprm_retrain_attempted": False,
            "thinkprm_auroc_after": None,
            "thinkprm_auroc_before": BASELINE_AUROC,
            "tests_passing": 0,
        }
        _write_artifact(artifact)
        return

    # -----------------------------------------------------------------------
    # Step 2: Generate step-level data
    # -----------------------------------------------------------------------
    print(f"[exp1084] Generating step-level data from {CORPUS_PATH} ...")
    from scripts.prm_data_generator import generate_and_save

    gen_stats = generate_and_save(CORPUS_PATH, OUTPUT_JSONL)

    print(
        f"[exp1084] Generated {gen_stats['n_step_examples_generated']} examples "
        f"(correct={gen_stats['n_correct_step_examples']}, "
        f"wrong={gen_stats['n_wrong_step_examples']}, "
        f"ambiguous_excluded={gen_stats['n_ambiguous_excluded']})"
    )

    n_generated = gen_stats["n_step_examples_generated"]

    # -----------------------------------------------------------------------
    # Step 3: Conditional retraining
    # -----------------------------------------------------------------------
    thinkprm_retrain_attempted = False
    auroc_after: float | None = None
    retrain_details: dict = {}

    if n_generated < 100:
        honest_verdict = "step_data_insufficient"
        print(f"[exp1084] Only {n_generated} examples generated — too few (< 100). Skipping.")
    elif n_generated < RETRAIN_THRESHOLD:
        honest_verdict = "step_data_generated_retrain_skipped"
        print(
            f"[exp1084] {n_generated} examples < {RETRAIN_THRESHOLD} threshold — skipping retrain."
        )
    else:
        print(f"[exp1084] {n_generated} examples >= {RETRAIN_THRESHOLD} — attempting retrain ...")
        thinkprm_retrain_attempted = True

        if Path(TEST_PATH).exists():
            retrain_details = _run_retrain(OUTPUT_JSONL, TEST_PATH)
            if retrain_details["retrain_ok"]:
                auroc_after = retrain_details["auroc_after"]
                if auroc_after is not None and auroc_after > BASELINE_AUROC:
                    honest_verdict = "step_data_generated_thinkprm_improved"
                else:
                    honest_verdict = "step_data_generated_thinkprm_unchanged"
            else:
                honest_verdict = "step_data_generated_retrain_skipped"
                print(f"[exp1084] Retrain failed: {retrain_details.get('error')}")
        else:
            honest_verdict = "step_data_generated_retrain_skipped"
            print(f"[exp1084] Test corpus not found: {TEST_PATH}")

    # -----------------------------------------------------------------------
    # Step 4: Run tests
    # -----------------------------------------------------------------------
    tests_passing = _count_passing_tests()

    # -----------------------------------------------------------------------
    # Step 5: Write artifact
    # -----------------------------------------------------------------------
    duration_s = round(time.time() - run_start, 2)
    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": _utcnow(),
        "duration_s": duration_s,
        "n_fover_pairs_processed": gen_stats["n_fover_pairs_processed"],
        "n_step_examples_generated": n_generated,
        "n_correct_step_examples": gen_stats["n_correct_step_examples"],
        "n_wrong_step_examples": gen_stats["n_wrong_step_examples"],
        "n_ambiguous_excluded": gen_stats["n_ambiguous_excluded"],
        "output_file": OUTPUT_JSONL,
        "thinkprm_retrain_attempted": thinkprm_retrain_attempted,
        "thinkprm_auroc_after": auroc_after,
        "thinkprm_auroc_before": BASELINE_AUROC,
        "retrain_details": retrain_details,
        "tests_passing": tests_passing,
        "honest_verdict": honest_verdict,
    }
    _write_artifact(artifact)
    print(f"[exp1084] Done. verdict={honest_verdict} duration={duration_s}s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    """Return current UTC time as ISO-8601 string."""
    import datetime

    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_artifact(artifact: dict) -> None:
    """Write the result artifact JSON to the deliverable path."""
    Path(DELIVERABLE).parent.mkdir(parents=True, exist_ok=True)
    with open(DELIVERABLE, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"[exp1084] Artifact written to {DELIVERABLE}")


def _count_passing_tests() -> int:
    """Run the experiment's own tests and return number passing."""
    import subprocess

    test_file = "tests/python/test_prm_data_generator.py"
    if not Path(test_file).exists():
        return 0
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Count "passed" from pytest output
        for line in result.stdout.splitlines():
            if "passed" in line:
                import re

                m = re.search(r"(\d+) passed", line)
                if m:
                    return int(m.group(1))
        return 0
    except Exception as e:
        print(f"[exp1084] Test run failed: {e}")
        return 0


if __name__ == "__main__":
    main()
