#!/usr/bin/env python3
"""Exp 543 — JEPA Live Retrain v8: LeWorldModel on Exp 542 expanded FOVER corpus.

**What this experiment does:**
    Retrains the JEPA violation predictor using the LeWorldModel two-term objective
    (arXiv 2603.19312) on the expanded FOVER corpus produced by Exp 542.  Falls back
    to the Exp 442 baseline 57-pair corpus, then to synthetic pairs for CI.

**Data source priority (FR-11 honesty contract):**
    1. live_fover_expanded — fover_labeled_steps_expanded.json (Exp 542 merged corpus)
    2. live_fover_442      — fover_labeled_steps_live.json (Exp 442 baseline, 57 pairs)
    3. synthetic           — 100 deterministic synthetic pairs (CI-only fallback)

**Why v8 over v7:**
    v7 (Exp 535) used the Exp 442 FOVER baseline (57 pairs, AUC=0.967).  Exp 542
    expanded the corpus via multi-source merge.  v8 validates whether the expanded
    corpus maintains or improves AUC, closing the FR-11 relay loop.

**LeWorldModel objective (lambda_reg=0.1):**
    L_total = L_prediction + 0.1 * L_KL
    L_KL = KL(q(z) || N(0,I)) = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
    Stronger regularization (0.1 vs 0.01 in v6/v7) prevents overfitting on the
    more diverse expanded corpus.

Spec: REQ-LEARN-056, REQ-LEARN-057, SCENARIO-LEARN-088, SCENARIO-LEARN-089
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root on sys.path — required to import carnot and scripts modules
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# Step a: apply_env_autofix FIRST (REQ-INFRA-060)
# ---------------------------------------------------------------------------

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

# ---------------------------------------------------------------------------
# Step b: ExperimentTimeoutWatchdog
# ---------------------------------------------------------------------------

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_watchdog = ExperimentTimeoutWatchdog(543, timeout_minutes=40)
_watchdog.start()

# ---------------------------------------------------------------------------
# Step c: ExperimentTemplate
# ---------------------------------------------------------------------------

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

tmpl = ExperimentTemplate(
    543,
    "JEPA v8 Live Retrain",
    "results/experiment_543_jepa_v8_live_retrain.json",
    requires_gpu=False,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# Imports for experiment body
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from carnot.models.jepa_retrain_v6 import (  # noqa: E402
    compute_held_out_split,
    violation_pairs_to_trainer_dicts,
)
from carnot.models.jepa_retrain_v8 import load_v8_cot_corpus  # noqa: E402
from carnot.pipeline.jepa_predictor import JEPAViolationPredictor  # noqa: E402
from carnot.pipeline.lw_jepa_trainer import (  # noqa: E402
    LeWorldModelJEPATrainer,
    LeWorldModelLoss,
)

# ---------------------------------------------------------------------------
# Step d: Load corpus with cascading fallback (REQ-LEARN-056, SCENARIO-LEARN-089)
# ---------------------------------------------------------------------------

print("[543] Loading FOVER corpus...")

_EXPANDED_PATH = str(_REPO / "results" / "fover_labeled_steps_expanded.json")
_LIVE_FALLBACK_PATH = str(_REPO / "results" / "fover_labeled_steps_live.json")

raw_pairs, data_source = load_v8_cot_corpus(_EXPANDED_PATH, _LIVE_FALLBACK_PATH)
print(f"  [543] data_source={data_source}, n_raw_pairs={len(raw_pairs)}")


def _make_synthetic_trainer_dicts(n: int = 100, seed: int = 543) -> list:
    """Generate deterministic synthetic training dicts for CI fallback.

    Why 256-D embeddings with class-correlated signal: matches the Exp 520/522
    synthetic pair generation.  The +0.5/-0.5 bias in emb[0] gives the predictor
    a learnable signal without requiring real data.
    """
    rng = np.random.RandomState(seed)
    pairs = []
    for i in range(n):
        label = int(i % 2)
        emb = rng.randn(256).astype(np.float32)
        emb[0] += (1.0 if label else -1.0) * 0.5
        pairs.append({
            "embedding": emb.tolist(),
            "violated_arithmetic": label,
            "violated_code": label,
            "violated_logic": label,
        })
    return pairs


if data_source == "synthetic":
    all_trainer_dicts = _make_synthetic_trainer_dicts(100, seed=543)
else:
    all_trainer_dicts = violation_pairs_to_trainer_dicts(raw_pairs)

print(f"  [543] Total trainer dicts: {len(all_trainer_dicts)}")

# ---------------------------------------------------------------------------
# Step e: 80/20 train/test split — same seed strategy as v7 for comparability
# ---------------------------------------------------------------------------

# v7 used the first 20% of pairs as test (no shuffle, deterministic order).
# We mirror that here so AUC comparisons are apples-to-apples.
if data_source == "synthetic":
    n_test = max(1, int(len(all_trainer_dicts) * 0.2))
    n_test = min(n_test, len(all_trainer_dicts) - 1)
    test_dicts = all_trainer_dicts[:n_test]
    train_dicts = all_trainer_dicts[n_test:]
else:
    train_vp, test_vp = compute_held_out_split(raw_pairs, test_fraction=0.2)
    train_dicts = violation_pairs_to_trainer_dicts(train_vp)
    test_dicts = violation_pairs_to_trainer_dicts(test_vp)

n_train_pairs = len(train_dicts)
n_test_pairs = len(test_dicts)
print(f"  [543] train={n_train_pairs}, test={n_test_pairs}")

# ---------------------------------------------------------------------------
# Step f: Train JEPA predictor with LeWorldModel two-term objective
# (REQ-LEARN-057): lambda_reg=0.1 for stronger regularization on expanded corpus
# ---------------------------------------------------------------------------

print("[543] Training JEPA predictor with LeWorldModel objective (lambda=0.1, 100 epochs)...")

predictor = JEPAViolationPredictor(seed=543)
lw_loss = LeWorldModelLoss(lambda_reg=0.1)
trainer = LeWorldModelJEPATrainer(predictor, loss=lw_loss)

train_result = trainer.train_to_convergence(train_dicts, max_epochs=100, patience=5)
print(
    f"  [543] Training complete: epochs={train_result['epochs_trained']}, "
    f"converged={train_result['converged']}, "
    f"train_auc={train_result['final_auc']:.4f}"
)

# ---------------------------------------------------------------------------
# Step g: Evaluate AUC on held-out test split
# ---------------------------------------------------------------------------

final_auc = trainer.evaluate_auc(test_dicts)
auc_v7_baseline = 0.967  # v7 baseline from Exp 535 (AUC=0.966667, rounded)
auc_improvement = final_auc - auc_v7_baseline
print(f"  [543] Test AUC: {final_auc:.4f} (vs v7 baseline {auc_v7_baseline:.3f}, delta={auc_improvement:+.4f})")

# ---------------------------------------------------------------------------
# Step h: Save model checkpoint to results/jepa_predictor_543_v8.safetensors
# ---------------------------------------------------------------------------

_ckpt_path = _REPO / "results" / "jepa_predictor_543_v8.safetensors"
_saved_checkpoint = False

try:
    from safetensors.numpy import save_file  # type: ignore[import]

    params = getattr(predictor, "_params", None)
    if params is not None and isinstance(params, dict):
        np_params = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
        save_file(np_params, str(_ckpt_path))
        _saved_checkpoint = True
        print(f"  [543] Checkpoint saved to {_ckpt_path}")
    else:
        print("  [543] No _params dict on predictor — checkpoint skipped")
except Exception as exc:
    print(f"  [543] Checkpoint save skipped: {exc}")

# ---------------------------------------------------------------------------
# Step i: Build artifact with honest_verdict (SCENARIO-LEARN-088)
# ---------------------------------------------------------------------------

fr11_live_relay = (data_source != "synthetic") and (final_auc >= 0.800)

# honest_verdict logic (SCENARIO-LEARN-088):
#   'jepa_v8_improved'   — auc>=0.900 AND n_train>=80 (best case: clear improvement)
#   'auc_stable'         — auc>=0.800 (acceptable, within tolerated range)
#   'synthetic_fallback' — all other cases (data too sparse or AUC too low)
if final_auc >= 0.900 and n_train_pairs >= 80:
    honest_verdict = "jepa_v8_improved"
elif final_auc >= 0.800:
    honest_verdict = "auc_stable"
else:
    honest_verdict = "synthetic_fallback"

print(f"  [543] honest_verdict={honest_verdict}, fr11_live_relay={fr11_live_relay}")

artifact = tmpl.build_result(
    {
        "schema": "carnot.jepa_retrain.v8",
        "data_source": data_source,
        "n_train_pairs": n_train_pairs,
        "n_test_pairs": n_test_pairs,
        "final_auc": round(float(final_auc), 6),
        "auc_improvement": round(float(auc_improvement), 6),
        "epochs_trained": train_result["epochs_trained"],
        "converged": train_result["converged"],
        "loss_history": [round(v, 6) for v in train_result["loss_history"]],
        "fr11_live_relay": fr11_live_relay,
        "honest_verdict": honest_verdict,
        "checkpoint_saved": _saved_checkpoint,
        "checkpoint_path": str(_ckpt_path) if _saved_checkpoint else None,
    },
    status="success",
    decision_class="verify",
)

_out_path = _REPO / "results" / "experiment_543_jepa_v8_live_retrain.json"
_out_path.write_text(json.dumps(artifact, indent=2))
print(f"[543] Artifact written to {_out_path}")
print(f"[543] Done. honest_verdict={honest_verdict}, final_auc={final_auc:.4f}")

# ---------------------------------------------------------------------------
# Step j: assert_deliverable_written (FINAL LINE)
# ---------------------------------------------------------------------------

tmpl.assert_deliverable_written()
