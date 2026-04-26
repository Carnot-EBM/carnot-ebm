#!/usr/bin/env python3
"""Experiment 899: DRIFTProbe — Hidden-State Representational Drift (Tier 0i).

**Researcher summary:**
    arXiv 2601.14210 (DRIFT) demonstrates that hallucinating LLMs show lower
    cosine similarity between adjacent Transformer layer hidden states than
    truthful completions.  This experiment implements DRIFTProbe (Tier 0i),
    extracts drift signatures from Qwen/Qwen3.5-0.8B, trains a linear probe
    on 50 FoVer pairs, and evaluates on 10 held-out pairs plus 20 synthetic OOD pairs.

    Target: probe_auc > 0.65 to declare "drift_probe_viable".

Spec: REQ-TIER0-009, SCENARIO-TIER0-009
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

DELIVERABLE = "results/experiment_899_drift_hidden_state_probe.json"

tmpl = ExperimentTemplate(
    exp_id=899,
    title="DRIFTProbe — Hidden-State Representational Drift Tier 0i",
    deliverable=DELIVERABLE,
    requires_gpu=False,
)
tmpl.setup()

t_start = time.time()


def _load_fover_pairs(n: int = 60, seed: int = 42) -> list[dict]:
    """Load FoVer pairs from fover_corpus_v2.json, balanced where possible.

    Each returned dict has keys: text (str), label (int: 1=hallucinating, 0=truthful).
    When fewer than n pairs are available, returns as many as exist.

    Args:
        n: Target number of pairs to return.
        seed: Random seed for reproducible sampling.

    Returns:
        List of dicts with "text" and "label" keys.
    """
    corpus_path = os.path.join(PROJECT_ROOT, "results", "fover_corpus_v2.json")
    with open(corpus_path) as f:
        raw = json.load(f)

    # Convert corpus entries to (text, label) pairs.
    # is_correct=True → truthful (label 0); is_correct=False → hallucinating (label 1).
    pairs = []
    for entry in raw:
        # Use the main response text; fall back to the first cot_step text.
        text = entry.get("response", "").strip()
        if not text:
            steps = entry.get("cot_steps", [])
            text = steps[0].get("step_text", "") if steps else ""
        if not text:
            continue
        label = 0 if entry.get("is_correct", False) else 1
        pairs.append({"text": text, "label": label})

    # Separate by class and balance if possible.
    rng = random.Random(seed)
    truthful = [p for p in pairs if p["label"] == 0]
    hallucinating = [p for p in pairs if p["label"] == 1]

    rng.shuffle(truthful)
    rng.shuffle(hallucinating)

    # Take up to n//2 from each class; if one class is short, fill with the other.
    half = n // 2
    t_take = min(half, len(truthful))
    h_take = min(n - t_take, len(hallucinating))
    selected = truthful[:t_take] + hallucinating[:h_take]
    rng.shuffle(selected)
    return selected[:n]


def _build_synthetic_ood_pairs() -> list[dict]:
    """Generate 20 synthetic OOD pairs for out-of-distribution evaluation.

    10 pairs are intentionally wrong answers to simple arithmetic questions.
    10 pairs are correct Qwen answers (or ground-truth answers) to the same questions.

    These are text-only pairs — no LLM inference needed.  They exercise the
    drift-signature pathway with minimal arithmetic content to test OOD generalisation.

    Returns:
        List of 20 dicts with "text" (str) and "label" (int) keys.
    """
    correct_pairs = [
        "2 + 2 = 4. The answer is 4.",
        "5 * 3 = 15. The answer is 15.",
        "10 - 7 = 3. The answer is 3.",
        "8 / 2 = 4. The answer is 4.",
        "6 + 9 = 15. The answer is 15.",
        "4 * 4 = 16. The answer is 16.",
        "20 - 8 = 12. The answer is 12.",
        "9 + 6 = 15. The answer is 15.",
        "7 * 3 = 21. The answer is 21.",
        "100 / 5 = 20. The answer is 20.",
    ]
    wrong_pairs = [
        "2 + 2 = 5. The answer is 5.",
        "5 * 3 = 14. Wait, no. The answer is 17.",
        "10 - 7 = 4. The answer is 4.",
        "8 / 2 = 3. Actually I think it is 3.",
        "6 + 9 = 14. The answer is 14.",
        "4 * 4 = 15. The answer is 15.",
        "20 - 8 = 13. The answer is 13.",
        "9 + 6 = 14. Wait, 9 + 6... I believe it is 14.",
        "7 * 3 = 24. Let me verify: 7 * 3 = 24.",
        "100 / 5 = 25. The answer is 25.",
    ]
    pairs = (
        [{"text": t, "label": 0} for t in correct_pairs]
        + [{"text": t, "label": 1} for t in wrong_pairs]
    )
    return pairs


def main() -> None:
    """Run Exp 899: DRIFTProbe training, evaluation, and artifact writing."""
    print("[Exp 899] Loading FoVer pairs...")
    all_pairs = _load_fover_pairs(n=60)
    print(f"  Loaded {len(all_pairs)} pairs.")

    # Split 50 train / 10 eval.
    train_pairs = all_pairs[:50]
    eval_pairs = all_pairs[50:]
    if not eval_pairs:
        # Fallback: if corpus is small, use the last 5 of train as eval.
        eval_pairs = train_pairs[-10:]
        train_pairs = train_pairs[:-10]

    label_counts = {0: sum(1 for p in train_pairs if p["label"] == 0),
                    1: sum(1 for p in train_pairs if p["label"] == 1)}
    print(f"  Train: {len(train_pairs)} pairs (truthful={label_counts[0]}, "
          f"hallucinating={label_counts[1]})")
    print(f"  Eval:  {len(eval_pairs)} pairs")

    from carnot.probes.drift_probe import DRIFTProbe

    probe = DRIFTProbe(model_name="Qwen/Qwen3.5-0.8B", probe_layers=[4, 8, 12, 16])

    print("[Exp 899] Extracting drift signatures for training set (CPU)...")
    probe.fit(train_pairs)
    print("  Probe fitted.")

    # Evaluate on held-out FoVer pairs.
    print("[Exp 899] Evaluating on held-out FoVer pairs...")
    from sklearn.metrics import roc_auc_score

    eval_labels = [p["label"] for p in eval_pairs]
    eval_probas = [probe.predict_proba(p["text"]) for p in eval_pairs]

    # roc_auc_score requires both classes present; handle degenerate case.
    if len(set(eval_labels)) < 2:
        probe_auc = 0.5
        print("  WARNING: eval set has only one class; probe_auc set to 0.5.")
    else:
        probe_auc = float(roc_auc_score(eval_labels, eval_probas))
    print(f"  probe_auc = {probe_auc:.4f}")

    # Out-of-distribution evaluation on 20 synthetic arithmetic pairs.
    print("[Exp 899] Evaluating on OOD synthetic pairs...")
    ood_pairs = _build_synthetic_ood_pairs()
    ood_labels = [p["label"] for p in ood_pairs]
    ood_probas = [probe.predict_proba(p["text"]) for p in ood_pairs]

    if len(set(ood_labels)) < 2:
        ood_auc = 0.5
    else:
        ood_auc = float(roc_auc_score(ood_labels, ood_probas))
    print(f"  ood_auc = {ood_auc:.4f}")

    # Collect probe metadata.
    drift_signature_shape = list(probe.extract_drift_signature("test").shape)
    linear_probe_coef = (
        probe.linear_probe.coef_.tolist() if probe.linear_probe is not None else None
    )

    # Determine honest verdict.
    if probe_auc > 0.65:
        honest_verdict = "drift_probe_viable"
    elif probe_auc > 0.55:
        honest_verdict = "drift_probe_marginal"
    else:
        honest_verdict = "drift_probe_not_viable"

    print(f"  honest_verdict = {honest_verdict}")

    # Check if model actually loaded (determines inference mode reliability).
    model_loaded = not probe._model_load_failed and probe._model is not None
    inference_mode = "cpu_live" if model_loaded else "cpu_synthetic_fallback"
    print(f"  inference_mode = {inference_mode}")

    # Build standardised artifact.
    artifact = tmpl.build_result(
        {
            "probe_auc": probe_auc,
            "ood_auc": ood_auc,
            "probe_layers": probe.probe_layers,
            "drift_signature_shape": drift_signature_shape,
            "linear_probe_coef": linear_probe_coef,
            "inference_mode": inference_mode,
            "train_n": len(train_pairs),
            "eval_n": len(eval_pairs),
            "ood_n": len(ood_pairs),
            "model_name": probe.model_name,
            "model_loaded": model_loaded,
            "honest_verdict": honest_verdict,
            "label_counts_train": label_counts,
        },
        status="success",
    )

    # Write deliverable.
    out_path = os.path.join(PROJECT_ROOT, DELIVERABLE)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"[Exp 899] Deliverable written: {out_path}")

    tmpl.assert_deliverable_written()
    print("[Exp 899] Done.")


if __name__ == "__main__":
    main()
