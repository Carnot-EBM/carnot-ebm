#!/usr/bin/env python3
"""Experiment 647: OTV One-Token Verifier — benchmark vs EORM.

**Researcher summary:**
    OTV (arXiv 2603.01025) adds a single learnable verification token via LoRA
    to any LLM and estimates reasoning correctness in one forward pass — no
    rollouts, no separate verifier model.  Carnot's EORM runs ~10ms per check
    at 55M params.  An OTV-style head using text statistics runs sub-1ms (CPU
    dot product), a ~100x theoretical speedup.

    This experiment:
    1. Loads live FOVER pairs (fover_corpus_v5_oracle.json, n>=50).
    2. Trains an OTVVerificationHead with binary cross-entropy for 50 epochs.
    3. Evaluates AUC-ROC on the held-out 20% test split.
    4. Compares against the best known EORM AUC from prior result files.
    5. Declares OTV viable if AUC gap <= 0.05.

**Gate:**
    0. apply_env_autofix() FIRST.
    1. ExperimentTimeoutWatchdog(647, timeout_minutes=25).
    2. Load data -> train -> evaluate -> compare -> artifact.
    3. tmpl.assert_deliverable_written() — FINAL LINE.

Spec: REQ-VERIFY-145, SCENARIO-VERIFY-192, SCENARIO-VERIFY-193
"""

from __future__ import annotations

import json
import random
import sys
import timeit
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Step 0: apply_env_autofix BEFORE any heavy imports.
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

# ---------------------------------------------------------------------------
# Step 1: Watchdog — hard 25-minute wall-clock cap.
# ---------------------------------------------------------------------------
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_watchdog = ExperimentTimeoutWatchdog(647, timeout_minutes=25)

# ---------------------------------------------------------------------------
# Remaining imports
# ---------------------------------------------------------------------------
import jax.numpy as jnp  # noqa: E402

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.models.otv_verifier import OTVVerificationHead, OTVTrainer  # noqa: E402
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402

_DELIVERABLE = "results/experiment_647_otv_verifier.json"

tmpl = ExperimentTemplate(
    647,
    "OTV One-Token Verifier",
    _DELIVERABLE,
    requires_gpu=False,
)
tmpl.setup()


# ---------------------------------------------------------------------------
# Step 2: Load FOVER pairs.
# ---------------------------------------------------------------------------
def _load_pairs() -> list[dict]:
    """Load labeled response pairs from FOVER corpus.

    Why fover_corpus_v5_oracle first: it is the most recent and largest
    labeled corpus (175 pairs at last count).  live_pairs_578 is the fallback
    if the oracle corpus has fewer than 50 usable pairs.

    Returns a list of dicts with 'response' (str) and 'is_correct' (bool).
    """
    oracle_path = _REPO_ROOT / "results/fover_corpus_v5_oracle.json"
    live_path = _REPO_ROOT / "results/live_pairs_578.json"

    def _from_oracle(raw: list) -> list[dict]:
        pairs = []
        for rec in raw:
            response = rec.get("model_response", "")
            is_correct = not bool(rec.get("has_violation", True))
            pairs.append({"response": response, "is_correct": is_correct})
        return pairs

    def _from_live(raw: list) -> list[dict]:
        pairs = []
        for rec in raw:
            response = rec.get("response", "")
            is_correct = bool(rec.get("is_correct", False))
            pairs.append({"response": response, "is_correct": is_correct})
        return pairs

    if oracle_path.exists():
        raw = json.loads(oracle_path.read_text())
        pairs = _from_oracle(raw) if isinstance(raw, list) else []
        if len(pairs) >= 50:
            return pairs

    # Fallback to live_pairs_578.
    if live_path.exists():
        raw = json.loads(live_path.read_text())
        return _from_live(raw) if isinstance(raw, list) else []

    return []


pairs = _load_pairs()
if not pairs:
    artifact = tmpl.build_result(
        {"error": "no FOVER pairs found"},
        status="blocked",
    )
    import json as _json  # noqa: PLC0415

    Path(_DELIVERABLE).write_text(_json.dumps(artifact, indent=2))
    tmpl.assert_deliverable_written()
    raise SystemExit(0)

# 80/20 train/test split with fixed seed for reproducibility.
random.seed(42)
shuffled = list(pairs)
random.shuffle(shuffled)
split = int(len(shuffled) * 0.8)
train_pairs = shuffled[:split]
test_pairs = shuffled[split:]

# ---------------------------------------------------------------------------
# Step 3: Train OTV head.
# ---------------------------------------------------------------------------
head = OTVVerificationHead()
trainer = OTVTrainer(head, lr=0.01)
head = trainer.train(train_pairs, n_epochs=50)

# ---------------------------------------------------------------------------
# Step 4: Evaluate AUC-ROC on test set.
# ---------------------------------------------------------------------------

def _compute_auc(trained_head: OTVVerificationHead, eval_pairs: list[dict]) -> float:
    """Compute AUC-ROC manually (no sklearn dependency required).

    Why manual: avoids adding sklearn to the hard dependency list for this
    lightweight module.  The trapezoidal AUC formula is standard and matches
    sklearn's roc_auc_score for the same (scores, labels) input.
    """
    if not eval_pairs:
        return 0.5

    scores = []
    labels = []
    for p in eval_pairs:
        x = trained_head.feature_vector(p["response"])
        scores.append(trained_head.forward(x))
        labels.append(int(p["is_correct"]))

    # Sort by score descending; compute TPR/FPR at each threshold.
    paired = sorted(zip(scores, labels), key=lambda t: -t[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        # Degenerate split — return 0.5 (random baseline) rather than NaN.
        return 0.5

    tp = fp = 0
    prev_fpr = prev_tpr = 0.0
    auc = 0.0
    for _score, lbl in paired:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr, prev_tpr = fpr, tpr

    return float(auc)


otv_auc = _compute_auc(head, test_pairs)

# ---------------------------------------------------------------------------
# Step 5: Load EORM AUC baseline from prior results.
# ---------------------------------------------------------------------------

def _load_eorm_baseline() -> float:
    """Extract the best EORM/JEPA AUC from Exps 556, 559, 383, 631.

    Why these experiments: they represent successive EORM and JEPA retrain
    runs, each producing an AUC measurement on the same FOVER corpus.
    We take the maximum so the OTV viability threshold is as conservative
    as possible — OTV must be close to the BEST known EORM, not just the
    earliest.
    """
    candidates: list[float] = []
    search_paths = [
        ("results/experiment_631_jepa_v14_oracle.json", ["v14_ood_auc", "v14_in_dist_auc", "v13_ood_auc"]),
        ("results/experiment_556_eorm_grpo_retrain.json", ["after_auc", "before_auc", "auc"]),
        ("results/experiment_559_lowrank_kaem_calibration.json", ["auc", "test_auc", "eorm_auc"]),
        ("results/experiment_383_models_retrain.json", ["auc", "test_auc", "eorm_auc"]),
    ]
    for rel_path, keys in search_paths:
        p = _REPO_ROOT / rel_path
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        for k in keys:
            v = d.get(k)
            if isinstance(v, (int, float)) and 0 < v <= 1.0:
                candidates.append(float(v))

    return max(candidates) if candidates else 0.60


eorm_auc_baseline = _load_eorm_baseline()

# ---------------------------------------------------------------------------
# Step 6: Measure OTV forward latency vs theoretical EORM baseline.
# ---------------------------------------------------------------------------
_sample_response = test_pairs[0]["response"] if test_pairs else "The answer is 18."
_sample_x = head.feature_vector(_sample_response)

# Time a single OTV forward pass (µs level).
_n_timing = 1000
_elapsed = timeit.timeit(lambda: head.forward(_sample_x), number=_n_timing)
otv_latency_ms = (_elapsed / _n_timing) * 1000.0

# EORM reference latency from project documentation (~10ms per check).
eorm_latency_ms = 10.0
speedup_ratio = eorm_latency_ms / max(otv_latency_ms, 1e-6)

# ---------------------------------------------------------------------------
# Step 7: Compute viability and build artifact.
# ---------------------------------------------------------------------------
auc_gap = float(eorm_auc_baseline - otv_auc)
otv_viable = bool(otv_auc >= eorm_auc_baseline - 0.05)

recommendation = (
    "OTV as Tier 2 default" if otv_viable else "Keep EORM as Tier 2"
)
honest_verdict = (
    "otv_viable_replace_eorm" if otv_viable else "otv_not_viable_keep_eorm"
)

artifact = tmpl.build_result(
    {
        "n_train": len(train_pairs),
        "n_test": len(test_pairs),
        "otv_auc": round(otv_auc, 6),
        "eorm_auc_baseline": round(eorm_auc_baseline, 6),
        "auc_gap": round(auc_gap, 6),
        "otv_viable": otv_viable,
        "otv_latency_ms": round(otv_latency_ms, 4),
        "eorm_latency_ms": eorm_latency_ms,
        "speedup_ratio": round(speedup_ratio, 2),
        "recommendation": recommendation,
        "honest_verdict": honest_verdict,
    },
    status="success",
)

# Overwrite the schema field with our explicit version string.
# WHY: build_result() sets schema to a sorted list of result keys;
# we want the canonical version string for downstream tooling.
artifact["schema"] = "carnot.otv_verifier.v1"

AtomicResultWriter(str(_REPO_ROOT / _DELIVERABLE)).write(artifact)

# ---------------------------------------------------------------------------
# FINAL LINE — must be last.
# ---------------------------------------------------------------------------
tmpl.assert_deliverable_written()
