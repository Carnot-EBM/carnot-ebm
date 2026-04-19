#!/usr/bin/env python3
"""Exp 520 — LeWorldModel-JEPA: compare stability of BCE vs. two-term LeWorldModel loss.

**What this experiment measures:**
    Does the Gaussian KL regularization term from arXiv 2603.19312 (LeWorldModel)
    reduce AUC variance across independent training runs, compared to standard BCE loss?

    Three runs of each approach are trained on 100 synthetic CoT pairs.
    AUC mean and variance are compared.  Stable training = low variance = AUC variance < 0.05.

    Root cause context: BCE collapses when positive/negative pairs have similar embeddings
    (Exps 472, 510).  The KL term forces latent diversity and prevents the collapse.

Spec: REQ-LEARN-046, REQ-LEARN-047
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

_watchdog = ExperimentTimeoutWatchdog(520, timeout_minutes=25)
_watchdog.start()

# ---------------------------------------------------------------------------
# Step c: ExperimentTemplate
# ---------------------------------------------------------------------------

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

tmpl = ExperimentTemplate(
    520,
    "LeWorldModel-JEPA",
    "results/experiment_520_leworldmodel_jepa.json",
    requires_gpu=False,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# Step d: DeliverableGuard
# ---------------------------------------------------------------------------

from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402

_guard = DeliverableGuard(str(_REPO / "results" / "experiment_520_leworldmodel_jepa.json"))

# ---------------------------------------------------------------------------
# Imports for the experiment body
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from carnot.pipeline.jepa_predictor import JEPAViolationPredictor  # noqa: E402
from carnot.pipeline.lw_jepa_trainer import (  # noqa: E402
    LeWorldModelJEPATrainer,
    LeWorldModelLoss,
)


# ---------------------------------------------------------------------------
# Step e: Generate 100 synthetic CoT pairs
# ---------------------------------------------------------------------------


def _make_synthetic_pairs(n: int = 100, seed: int = 0) -> list[dict]:
    """Generate synthetic CoT violation pairs for training.

    Why synthetic: live FOVER corpus may not be available in CI.  Synthetic pairs
    with known labels let us measure AUC stability without model inference.
    Each pair has a 256-D random embedding and a deterministic label.
    """
    rng = np.random.RandomState(seed)
    pairs = []
    for i in range(n):
        # Positive and negative classes separated by a small margin in embedding space.
        # label=1 embeddings have a positive first component; label=0 have negative.
        label = int(i % 2)
        emb = rng.randn(256).astype(np.float32)
        # Inject a small class-correlated signal so AUC > 0.5 is achievable.
        emb[0] += (1.0 if label else -1.0) * 0.5
        pairs.append({
            "embedding": emb.tolist(),
            "violated_arithmetic": label,
            "violated_code": label,
            "violated_logic": label,
        })
    return pairs


PAIRS = _make_synthetic_pairs(100, seed=42)

# ---------------------------------------------------------------------------
# Helper: compute AUC for a trained predictor on the pairs
# ---------------------------------------------------------------------------


def _compute_auc_for_predictor(predictor: JEPAViolationPredictor, pairs: list[dict]) -> float:
    """Compute macro-AUC for a predictor using sklearn roc_auc_score.

    Returns 0.5 if AUC is undefined (single class in fold).
    """
    try:
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415
    except ImportError:
        return 0.5

    domains = {
        "arithmetic": "violated_arithmetic",
        "code": "violated_code",
        "logic": "violated_logic",
    }
    import numpy as np  # noqa: PLC0415

    aucs = []
    for domain, label_key in domains.items():
        y_true = [float(p[label_key]) for p in pairs]
        y_score = [predictor.predict(np.asarray(p["embedding"], dtype=np.float32)).get(domain, 0.5) for p in pairs]
        if len(set(y_true)) < 2:
            aucs.append(0.5)
        else:
            try:
                aucs.append(float(roc_auc_score(y_true, y_score)))
            except Exception:
                aucs.append(0.5)
    return float(np.mean(aucs)) if aucs else 0.5


# ---------------------------------------------------------------------------
# Step f: 3 independent runs with standard BCE loss
# ---------------------------------------------------------------------------

print("[520] Running 3 BCE-loss training runs...")

standard_bce_aucs: list[float] = []
for run_i in range(3):
    predictor = JEPAViolationPredictor(seed=run_i * 100)
    log = predictor.train(PAIRS, n_epochs=20, lr=1e-3, batch_size=32, seed=run_i)
    auc = log["macro_auroc"]
    standard_bce_aucs.append(float(auc))
    print(f"  BCE run {run_i}: AUC={auc:.4f}")

standard_bce_variance = float(np.var(standard_bce_aucs))
print(f"  BCE variance={standard_bce_variance:.6f}")

# ---------------------------------------------------------------------------
# Step g: 3 independent runs with LeWorldModelLoss
# ---------------------------------------------------------------------------

print("[520] Running 3 LeWorldModel-loss training runs...")

leworldmodel_aucs: list[float] = []
for run_i in range(3):
    predictor = JEPAViolationPredictor(seed=run_i * 100 + 50)
    lw_loss = LeWorldModelLoss(lambda_reg=0.01)
    trainer = LeWorldModelJEPATrainer(predictor, loss=lw_loss)
    result = trainer.train_to_convergence(PAIRS, max_epochs=20, patience=5)
    # Use the AUC from the trainer's evaluate_auc (which calls the underlying predictor)
    auc = result["final_auc"]
    leworldmodel_aucs.append(float(auc))
    print(f"  LW run {run_i}: AUC={auc:.4f}, epochs={result['epochs_trained']}, converged={result['converged']}")

leworldmodel_variance = float(np.var(leworldmodel_aucs))
print(f"  LW variance={leworldmodel_variance:.6f}")

# ---------------------------------------------------------------------------
# Step h+i: Compute verdict and build artifact
# ---------------------------------------------------------------------------

stability_improvement = bool(leworldmodel_variance < standard_bce_variance)
if stability_improvement:
    honest_verdict = "stable_training"
else:
    honest_verdict = "no_stability_gain"

print(f"[520] stability_improvement={stability_improvement}, honest_verdict={honest_verdict}")

artifact = tmpl.build_result(
    {
        "standard_bce_aucs": standard_bce_aucs,
        "standard_bce_mean": float(np.mean(standard_bce_aucs)),
        "standard_bce_variance": standard_bce_variance,
        "leworldmodel_aucs": leworldmodel_aucs,
        "leworldmodel_mean": float(np.mean(leworldmodel_aucs)),
        "leworldmodel_variance": leworldmodel_variance,
        "stability_improvement": stability_improvement,
        "honest_verdict": honest_verdict,
        "n_pairs": len(PAIRS),
        "n_runs": 3,
    },
    status="success",
    schema="carnot.lw_jepa.v1",
)

# Write the deliverable
out_path = _REPO / "results" / "experiment_520_leworldmodel_jepa.json"
out_path.write_text(json.dumps(artifact, indent=2))
print(f"[520] Wrote deliverable to {out_path}")

# ---------------------------------------------------------------------------
# Step j: assert deliverable written — FINAL LINE
# ---------------------------------------------------------------------------

tmpl.assert_deliverable_written()
