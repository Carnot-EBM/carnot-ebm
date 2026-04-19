#!/usr/bin/env python3
"""Exp 522 — JEPA Live Retrain v6: LeWorldModel two-term objective on real CoT data.

**What this experiment does:**
    Trains the JEPA violation predictor using the LeWorldModel two-term objective
    (arXiv 2603.19312) on the best available real CoT pairs, then evaluates AUC
    on a held-out test set.  FR-11 (self-learning milestone) is relayed only when
    real data is used AND final_auc >= 0.800.

**Data source priority (FR-11 honesty contract):**
    1. live_exp514_515 — results/exp514_cot_pairs.json + exp515_cot_pairs.json
    2. live_fover_442  — results/fover_labeled_steps_live.json (57 real pairs from Exp 442)
    3. synthetic       — 100 deterministic synthetic pairs (CI fallback)

**Why the two-term objective prevents collapse:**
    Standard BCE collapses when positive and negative embedding pairs are similar
    (AUC regression from 0.667 to 0.400 in Exp 472).  The Gaussian KL regularization
    term forces the latent distribution to stay near N(0,I), maintaining embedding
    diversity.  See REQ-LEARN-046 and lw_jepa_trainer.py for the math.

Spec: REQ-LEARN-048, SCENARIO-LEARN-076, SCENARIO-LEARN-077
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

_watchdog = ExperimentTimeoutWatchdog(522, timeout_minutes=30)
_watchdog.start()

# ---------------------------------------------------------------------------
# Step c: ExperimentTemplate
# ---------------------------------------------------------------------------

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

tmpl = ExperimentTemplate(
    522,
    "JEPA Live Retrain v6",
    "results/experiment_522_jepa_live_retrain_v6.json",
    requires_gpu=False,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# Step d: DeliverableGuard
# ---------------------------------------------------------------------------

from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402

_guard = DeliverableGuard(str(_REPO / "results" / "experiment_522_jepa_live_retrain_v6.json"))

# ---------------------------------------------------------------------------
# Imports for experiment body
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from carnot.models.jepa_retrain_v6 import (  # noqa: E402
    compute_held_out_split,
    load_cot_pairs_from_experiments,
    violation_pairs_to_trainer_dicts,
)
from carnot.pipeline.jepa_predictor import JEPAViolationPredictor  # noqa: E402
from carnot.pipeline.lw_jepa_trainer import (  # noqa: E402
    LeWorldModelJEPATrainer,
    LeWorldModelLoss,
)

# ---------------------------------------------------------------------------
# Step e: Load CoT pairs with cascading fallback
# ---------------------------------------------------------------------------

print("[522] Loading CoT pairs...")

_FOVER_PATH = str(_REPO / "results" / "fover_labeled_steps_live.json")
_EXP514_PATH = _REPO / "results" / "exp514_cot_pairs.json"
_EXP515_PATH = _REPO / "results" / "exp515_cot_pairs.json"

# Try live experiment data first (514 + 515)
_live_exp_pairs = load_cot_pairs_from_experiments([514, 515], _FOVER_PATH)

# Determine data_source based on what was loaded
if _EXP514_PATH.exists() or _EXP515_PATH.exists():
    # At least one exp file exists — use whatever was loaded (even if partially empty)
    _raw_pairs = _live_exp_pairs
    data_source = "live_exp514_515"
    print(f"  [522] Using live_exp514_515 data: {len(_raw_pairs)} pairs")
elif _live_exp_pairs:
    # Fell back to FOVER (Exp 442) real data
    _raw_pairs = _live_exp_pairs
    data_source = "live_fover_442"
    print(f"  [522] Using live_fover_442 fallback: {len(_raw_pairs)} pairs")
else:
    # Final fallback: 100 synthetic pairs
    data_source = "synthetic"
    _raw_pairs = []
    print("  [522] No real data found — using synthetic fallback")


def _make_synthetic_trainer_dicts(n: int = 100, seed: int = 522) -> list[dict]:
    """Generate deterministic synthetic training dicts (final fallback).

    Why 256-D embeddings with class-correlated signal: matches the Exp 520
    synthetic pair generation.  The +0.5/-0.5 bias in emb[0] gives the JEPA
    predictor a learnable signal even in CI without real data.
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
    all_trainer_dicts = _make_synthetic_trainer_dicts(100, seed=522)
else:
    # Convert ViolationPair objects to trainer dicts (hash-based 256-D embeddings)
    all_trainer_dicts = violation_pairs_to_trainer_dicts(_raw_pairs)

print(f"  [522] Total pairs available: {len(all_trainer_dicts)}")

# ---------------------------------------------------------------------------
# Step f: Record data_source (already done above)
# ---------------------------------------------------------------------------

# data_source is one of: 'live_exp514_515', 'live_fover_442', 'synthetic'
print(f"  [522] data_source={data_source}")

# ---------------------------------------------------------------------------
# Step g: Split 80/20 train/test
# ---------------------------------------------------------------------------

if data_source == "synthetic":
    # Synthetic: split the dicts directly (no ViolationPair objects)
    n_test = max(1, int(len(all_trainer_dicts) * 0.2))
    n_test = min(n_test, len(all_trainer_dicts) - 1)
    test_dicts = all_trainer_dicts[:n_test]
    train_dicts = all_trainer_dicts[n_test:]
else:
    # Real data: use compute_held_out_split on ViolationPair objects for determinism,
    # then convert splits to trainer dicts separately.
    train_vp, test_vp = compute_held_out_split(_raw_pairs, test_fraction=0.2)
    train_dicts = violation_pairs_to_trainer_dicts(train_vp)
    test_dicts = violation_pairs_to_trainer_dicts(test_vp)

n_train_pairs = len(train_dicts)
n_test_pairs = len(test_dicts)
print(f"  [522] train={n_train_pairs}, test={n_test_pairs}")

# ---------------------------------------------------------------------------
# Step h: LeWorldModelJEPATrainer.train_to_convergence
# ---------------------------------------------------------------------------

print("[522] Training JEPA predictor with LeWorldModel two-term objective...")

predictor = JEPAViolationPredictor(seed=522)
lw_loss = LeWorldModelLoss(lambda_reg=0.01)
trainer = LeWorldModelJEPATrainer(predictor, loss=lw_loss)

# Compute AUC before training (baseline)
training_auc = trainer.evaluate_auc(train_dicts)
print(f"  [522] Pre-train AUC (on train set): {training_auc:.4f}")

# Train to convergence with max_epochs=50, patience=5
train_result = trainer.train_to_convergence(train_dicts, max_epochs=50, patience=5)
print(
    f"  [522] Training complete: epochs={train_result['epochs_trained']}, "
    f"converged={train_result['converged']}, "
    f"final_train_auc={train_result['final_auc']:.4f}"
)

# ---------------------------------------------------------------------------
# Step i: Evaluate AUC on test pairs
# ---------------------------------------------------------------------------

final_auc = trainer.evaluate_auc(test_dicts)
auc_improvement = final_auc - training_auc
print(f"  [522] Test AUC: {final_auc:.4f} (improvement: {auc_improvement:+.4f})")

# ---------------------------------------------------------------------------
# Step j: Save checkpoint to results/jepa_predictor_522_live.safetensors
# ---------------------------------------------------------------------------

_ckpt_path = _REPO / "results" / "jepa_predictor_522_live.safetensors"
_saved_checkpoint = False

try:
    from safetensors.numpy import save_file  # type: ignore[import]

    # Package the predictor's learned weights as numpy arrays for safetensors.
    # JEPAViolationPredictor stores weights in self._params (a dict of jnp arrays).
    import jax.numpy as jnp  # noqa: PLC0415

    params = getattr(predictor, "_params", None)
    if params is not None and isinstance(params, dict):
        np_params = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
        save_file(np_params, str(_ckpt_path))
        _saved_checkpoint = True
        print(f"  [522] Checkpoint saved to {_ckpt_path}")
    else:
        print("  [522] No params to save (predictor may not expose _params)")
except Exception as exc:
    print(f"  [522] Checkpoint save skipped: {exc}")

# ---------------------------------------------------------------------------
# Step k: Build artifact
# ---------------------------------------------------------------------------

fr11_live_relay = (data_source != "synthetic") and (final_auc >= 0.800)
fr11_synthetic_only = data_source == "synthetic"

if fr11_live_relay:
    honest_verdict = "fr11_live_relay"
elif fr11_synthetic_only:
    honest_verdict = "fr11_synthetic_fallback"
else:
    honest_verdict = "fr11_partial"

print(f"  [522] honest_verdict={honest_verdict}, fr11_live_relay={fr11_live_relay}")

artifact = tmpl.build_result(
    {
        "schema": "carnot.jepa_retrain.v6",
        "data_source": data_source,
        "n_train_pairs": n_train_pairs,
        "n_test_pairs": n_test_pairs,
        "training_auc": round(float(training_auc), 6),
        "final_auc": round(float(final_auc), 6),
        "auc_improvement": round(float(auc_improvement), 6),
        "epochs_trained": train_result["epochs_trained"],
        "converged": train_result["converged"],
        "loss_history": [round(v, 6) for v in train_result["loss_history"]],
        "fr11_live_relay": fr11_live_relay,
        "fr11_synthetic_only": fr11_synthetic_only,
        "honest_verdict": honest_verdict,
        "checkpoint_saved": _saved_checkpoint,
        "checkpoint_path": str(_ckpt_path) if _saved_checkpoint else None,
    },
    status="success",
    decision_class="verify",
)

# Write artifact to disk
_out_path = _REPO / "results" / "experiment_522_jepa_live_retrain_v6.json"
_out_path.write_text(json.dumps(artifact, indent=2))
print(f"[522] Artifact written to {_out_path}")
print(f"[522] Done. honest_verdict={honest_verdict}, final_auc={final_auc:.4f}")

# ---------------------------------------------------------------------------
# Step l: assert_deliverable_written (FINAL LINE)
# ---------------------------------------------------------------------------

tmpl.assert_deliverable_written()
