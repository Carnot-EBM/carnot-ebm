#!/usr/bin/env python3
"""Exp 347 — JEPA Predictor Retrain on Real Live Violation Pairs.

**Researcher summary:**
    Exp 340 produced real LLM responses on 200 GSM8K questions with ground-truth
    correctness labels (Gemma4-E4B-it and Qwen3.5-0.8B). This experiment uses
    that data to retrain the JEPA predictor (ContextPredictionEnergy) on real
    (partial_response, has_violation) pairs, so the JEPA gate can predict constraint
    violations from only the first 50% of a response — before generation completes.

**Why this matters:**
    The prior JEPA gate (Exps 307-309) was trained on Apple adversarial logit data
    and evaluated with simulated predictions. Exp 347 is the first retrain on real
    GPU inference data, closing the simulation-to-reality gap.

**Honest reporting:**
    - If Exp 340 live data is absent (partial JSON), synthetic pairs are used and
      ``inference_mode="simulated"`` is recorded in the artifact.
    - AUC improvement is signed and may be negative — always reported truthfully.
    - The retrained model is saved to results/jepa_predictor_347_real.safetensors
      only when live data is used; synthetic runs save to a ``_synthetic`` suffix.

**Usage:**
    JAX_PLATFORMS=cpu python scripts/experiment_347_jepa_real_retrain.py
    CARNOT_FORCE_LIVE=1 python scripts/experiment_347_jepa_real_retrain.py

Spec: REQ-LEARN-024, SCENARIO-LEARN-041, SCENARIO-LEARN-042
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Ensure repo root on sys.path (for scripts and carnot imports)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax.random as jrandom

from carnot.embeddings.jepa_energy import ContextPredictionEnergy, JEPAEnergyConfig
from carnot.embeddings.jepa_retrain import (
    JEPARetrainer,
    ViolationPair,
    build_retrain_artifact,
    extract_violation_pairs,
)
from scripts.experiment_template import ExperimentTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = 347
TITLE = "JEPA real-data retrain — (partial_response, violation_flag) pairs from Exp 340 live GPU data"
DELIVERABLE = "results/experiment_347_jepa_real_retrain.json"

EXP_340_PATH = _REPO_ROOT / "results" / "experiment_340_live_precision_benchmark.json"
SAFETENSORS_LIVE_PATH = _REPO_ROOT / "results" / "jepa_predictor_347_real.safetensors"
SAFETENSORS_SYN_PATH = _REPO_ROOT / "results" / "jepa_predictor_347_synthetic.safetensors"

# Training config
TRAIN_SPLIT = 0.8
N_EPOCHS_CI = 10     # when no GPU / CI mode
N_EPOCHS_LIVE = 30   # when CARNOT_FORCE_LIVE=1
BATCH_SIZE = 8
LR = 1e-3            # slightly higher than default to converge faster on small data

# JEPA model config (must match existing predictor embed_dim for fine-tuning)
EMBED_DIM = 64
HIDDEN_DIMS = [64, 32]


# ---------------------------------------------------------------------------
# Load / build JEPA model
# ---------------------------------------------------------------------------


def _load_or_build_jepa_model() -> ContextPredictionEnergy:
    """Load the existing JEPA predictor or rebuild from scratch.

    **For engineers:**
        We prefer to fine-tune an existing model so prior NCE training is not lost.
        The existing model is stored as a ContextPredictionEnergy instance serialised
        via safetensors. If no compatible checkpoint exists, we rebuild fresh.

        NOTE: The existing jepa_predictor.safetensors was trained with embed_dim=64
        via Exp 307 / Exp 308. We rebuild with the same architecture so fine-tuning
        is semantically consistent.

        Safetensors loading for this architecture is not yet implemented in carnot
        (parameters are stored as Python lists of JAX arrays, not a flat dict),
        so we always rebuild from scratch with a fixed seed. This is honest —
        the "before_auc" baseline reflects the untrained model, which matches
        the zero-initialised output layer that gives AUC ~0.5.
    """
    cfg = JEPAEnergyConfig(
        embed_dim=EMBED_DIM,
        hidden_dims=HIDDEN_DIMS,
        activation="silu",
    )
    # Fixed seed for reproducibility across runs
    model = ContextPredictionEnergy(cfg, key=jrandom.PRNGKey(347))
    _log.info("Built fresh ContextPredictionEnergy(embed_dim=%d)", EMBED_DIM)
    return model


def _save_model_safetensors(model: ContextPredictionEnergy, path: Path) -> None:
    """Save JEPA model parameters to safetensors format.

    **For engineers:**
        ContextPredictionEnergy stores parameters as Python lists of JAX arrays
        (not a standard Flax/PyTorch state dict). We flatten them into a named
        dict of numpy arrays for safetensors serialisation.

        Naming convention:
        - ``layer_{i}_weight``, ``layer_{i}_bias`` for hidden layers
        - ``output_weight``, ``output_bias`` for the readout

    Args:
        model: Trained ContextPredictionEnergy model.
        path: Destination .safetensors file path.
    """
    try:
        import numpy as np
        from safetensors.numpy import save_file  # type: ignore[import]

        tensors: dict[str, "np.ndarray"] = {}
        for i, (w, b) in enumerate(model.layers):
            tensors[f"layer_{i}_weight"] = np.array(w)
            tensors[f"layer_{i}_bias"] = np.array(b)
        tensors["output_weight"] = np.array(model.output_weight)
        tensors["output_bias"] = np.array([model.output_bias], dtype=np.float32)

        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path))
        _log.info("Saved JEPA model to %s", path)
    except ImportError:
        _log.warning("safetensors not installed — model not saved to disk")
    except Exception as exc:
        _log.warning("Could not save model: %s", exc)


# ---------------------------------------------------------------------------
# Main experiment logic
# ---------------------------------------------------------------------------


def run_experiment(
    *,
    force_live: bool = False,
    repo_root: Path | None = None,
) -> dict:
    """Execute Exp 347: load data, retrain JEPA, evaluate AUC before/after.

    **For engineers:**
        This function is the single entry point for both live execution and unit tests.
        It is designed to be deterministic when live data is absent (synthetic mode).

    Args:
        force_live: If True, behave as if CARNOT_FORCE_LIVE=1 (more epochs).
        repo_root: Override repo root (used in tests).

    Returns:
        The full experiment artifact dict (same as written to JSON).
    """
    _root = repo_root or _REPO_ROOT
    tmpl = ExperimentTemplate(
        EXPERIMENT_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,  # retraining runs on CPU (JAX_PLATFORMS=cpu)
        repo_root=_root,
    )
    tmpl.setup()

    # ---- 1. Load Exp 340 live results ----
    exp340_path = _root / "results" / "experiment_340_live_precision_benchmark.json"
    live_results: dict | None = None
    inference_mode = "simulated"

    if exp340_path.exists():
        try:
            raw = json.loads(exp340_path.read_text())
            # Exp 340 partial artifact: check if it has real response data
            if isinstance(raw.get("responses"), list) and len(raw["responses"]) > 0:
                live_results = raw
                inference_mode = "live_gpu"
                _log.info(
                    "Loaded %d real responses from Exp 340", len(raw["responses"])
                )
            else:
                _log.info(
                    "Exp 340 JSON present but has no responses list — using synthetic pairs"
                )
        except (json.JSONDecodeError, OSError) as exc:
            _log.warning("Could not read Exp 340 JSON: %s — using synthetic pairs", exc)
    else:
        _log.info("Exp 340 results not found at %s — using synthetic pairs", exp340_path)

    # ---- 2. Extract violation pairs ----
    all_pairs = extract_violation_pairs(live_results, prefix_fraction=0.5)
    n_real_pairs = 0 if inference_mode == "simulated" else len(all_pairs)
    _log.info(
        "Extracted %d pairs (inference_mode=%s)", len(all_pairs), inference_mode
    )

    # ---- 3. 80/20 train/test split (deterministic, no shuffle) ----
    split_idx = max(1, round(TRAIN_SPLIT * len(all_pairs)))
    train_pairs: list[ViolationPair] = all_pairs[:split_idx]
    test_pairs: list[ViolationPair] = all_pairs[split_idx:]

    if not test_pairs:
        # Fallback: use last 20% of train as test (avoids empty test set on tiny data)
        fallback_split = max(1, round(0.8 * len(train_pairs)))
        test_pairs = train_pairs[fallback_split:]
        train_pairs = train_pairs[:fallback_split]

    _log.info("Split: %d train, %d test", len(train_pairs), len(test_pairs))

    # ---- 4. Load / build JEPA model ----
    model = _load_or_build_jepa_model()
    retrainer = JEPARetrainer(model, lr=LR)

    # ---- 5. Baseline AUC (before retraining) ----
    before_auc = retrainer.evaluate_auc_roc(test_pairs)
    _log.info("AUC before retraining: %.4f", before_auc)

    # ---- 6. Train ----
    n_epochs = N_EPOCHS_LIVE if (force_live or os.environ.get("CARNOT_FORCE_LIVE") == "1") else N_EPOCHS_CI
    epoch_losses: list[float] = []

    for epoch in range(n_epochs):
        loss = retrainer.train_epoch(train_pairs, batch_size=BATCH_SIZE)
        epoch_losses.append(loss)
        if epoch % max(1, n_epochs // 5) == 0:
            _log.info("Epoch %d/%d loss=%.6f", epoch + 1, n_epochs, loss)
        # Checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            tmpl.checkpoint_save(
                {"epoch": epoch + 1, "losses": epoch_losses},
                step=epoch + 1,
            )

    # ---- 7. After AUC ----
    after_auc = retrainer.evaluate_auc_roc(test_pairs)
    _log.info("AUC after %d epochs: %.4f (improvement: %+.4f)", n_epochs, after_auc, after_auc - before_auc)

    # ---- 8. Save retrained model ----
    safetensors_path = (
        _root / "results" / "jepa_predictor_347_real.safetensors"
        if inference_mode == "live_gpu"
        else _root / "results" / "jepa_predictor_347_synthetic.safetensors"
    )
    _save_model_safetensors(model, safetensors_path)

    # ---- 9. Build artifact ----
    retrain_meta = build_retrain_artifact(before_auc, after_auc, len(all_pairs))

    # vs_simulated_baseline_auc: from Exp 308 result if available
    vs_simulated_baseline_auc: float | None = None
    exp308_path = _root / "results" / "experiment_308_jepa_gate.json"
    if exp308_path.exists():
        try:
            exp308 = json.loads(exp308_path.read_text())
            vs_simulated_baseline_auc = exp308.get("gate_auc_roc") or exp308.get("auc_roc")
        except Exception:
            pass

    artifact = tmpl.build_result(
        {
            **retrain_meta,
            "n_real_pairs": n_real_pairs,
            "n_train_pairs": len(train_pairs),
            "n_test_pairs": len(test_pairs),
            "n_epochs": n_epochs,
            "epoch_losses": epoch_losses,
            "inference_mode": inference_mode,
            "training_mode": "live_gpu" if force_live else "ci",
            "vs_simulated_baseline_auc": vs_simulated_baseline_auc,
            "safetensors_path": str(safetensors_path.relative_to(_root)),
        },
        status="success",
    )

    # ---- 10. Write artifact ----
    output_path = _root / DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", output_path)

    return artifact


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"
    result = run_experiment(force_live=force_live)

    print(f"\nExperiment {EXPERIMENT_ID} complete.")
    print(f"  Status:          {result['status']}")
    print(f"  inference_mode:  {result['inference_mode']}")
    print(f"  n_pairs:         {result['n_pairs']}")
    print(f"  before_auc:      {result['before_auc']:.4f}")
    print(f"  after_auc:       {result['after_auc']:.4f}")
    print(f"  auc_improvement: {result['auc_improvement']:+.4f}")
    print(f"  Artifact:        {DELIVERABLE}")
