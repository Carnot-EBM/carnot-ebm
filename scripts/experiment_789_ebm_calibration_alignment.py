#!/usr/bin/env python3
"""Exp 789: EBM Calibration Alignment — ECE measurement + isotonic regression fix.

**Researcher summary:**
    Two arXiv papers motivate this experiment:
    1. arXiv 2603.06604 "Know When You're Wrong" (March 2026): SFT yields
       well-calibrated confidence; RL induces overconfidence.  Calibration gap
       is 15-25pp.  Carnot energy should be a calibration target.
    2. arXiv 2602.11364 "Energy of Falsehood" (February 2026): diffusion
       reconstruction energy detects hallucinations at AUROC 0.725 on FEVER.

    Current Carnot pipeline uses energy as a DISCRIMINATIVE signal (violated/not).
    This experiment measures whether energy is also a CALIBRATED signal and, if
    not, applies isotonic regression to fix the miscalibration.

**Why this matters:**
    If energy is calibrated, "energy decile 2" means "roughly 80% P(correct)."
    If uncalibrated, energy is only a binary detector — not a reasoning tool.
    Isotonic regression is the standard post-hoc fix for probability calibration
    (Zadrozny & Elkan 2002, used in sklearn.calibration).

**honest_verdict logic:**
    - "energy_well_calibrated"     if ECE_before <= 0.10 (already good)
    - "calibration_improved"       if ece_improvement >= 0.05 (significant gain)
    - "calibration_marginal"       if 0 < ece_improvement < 0.05 (minor gain)
    - "calibration_no_improvement" if ece_improvement <= 0 (isotonic doesn't help)
    - "insufficient_data"          if n_total_steps < 20

Spec: REQ-CALIB-001, REQ-CALIB-002, SCENARIO-CALIB-001, SCENARIO-CALIB-002
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from python.carnot.pipeline.ebm_calibrator import EBMCalibrator  # noqa: E402
from python.carnot.pipeline.experiment_watchdog import (  # noqa: E402
    ExperimentTimeoutWatchdog,
)

_DELIVERABLE = "results/experiment_789_ebm_calibration_alignment.json"
_CURVE_PATH = "results/ebm_calibration_curve.json"
_FOVER_V1 = "results/fover_labeled_steps_live.json"
_FOVER_V2 = "results/fover_labeled_steps_live_v2.json"


def load_labeled_steps(repo_root: Path) -> List[Tuple[str, int]]:
    """Load pooled FoVer labeled steps from v1 and optionally v2.

    Each step is a (step_text, label) pair where label=1 means correct,
    label=0 means incorrect.  We pool v1 (57 steps) and v2 if it exists.

    Returns:
        List of (step_text, binary_label) tuples.
    """
    steps: List[Tuple[str, int]] = []

    for path_rel in [_FOVER_V1, _FOVER_V2]:
        p = repo_root / path_rel
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        for item in data:
            text = item.get("step_text", "")
            raw_label = item.get("label", "")
            # Normalize: "correct" -> 1, anything else -> 0
            label = 1 if str(raw_label).strip().lower() == "correct" else 0
            steps.append((text, label))

    return steps


def _tfidf_energy_proxy(texts: List[str]) -> np.ndarray:
    """Compute a TF-IDF + IsingEBM energy proxy for plain text steps.

    When the full VerifyRepairPipeline is unavailable (no LLM), we need a
    deterministic energy signal.  We use TF-IDF to embed each step text into
    a fixed-dimension vector, normalize to [-1, 1], and compute the Ising
    quadratic energy E(x) = -0.5 x^T J x - b^T x with random parameters.

    The random seed is fixed (42) for reproducibility.  While this proxy does
    not reflect a trained EBM, it provides a non-trivial energy distribution
    that can test calibration machinery.

    Args:
        texts: List of step text strings.

    Returns:
        1-D numpy array of energy values, one per text.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    from python.carnot.models.ising import IsingConfig, IsingModel

    # Fit TF-IDF to get fixed-dim embeddings (max 50 features for speed)
    vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
    X = vectorizer.fit_transform(texts).toarray()  # shape: (N, 50)

    # Normalize each row to [-1, 1] so Ising spin convention is met
    row_max = np.abs(X).max(axis=1, keepdims=True)
    row_max = np.where(row_max == 0, 1.0, row_max)
    X_norm = X / row_max

    # Build a small Ising model on 50 dimensions
    cfg = IsingConfig(input_dim=50)
    key = jrandom.PRNGKey(42)
    model = IsingModel(cfg, key)

    energies = []
    for row in X_norm:
        x = jnp.array(row, dtype=jnp.float32)
        e = float(model.energy(x))
        energies.append(e)

    return np.array(energies, dtype=np.float64)


def classify_verdict(
    ece_before: float,
    ece_after: float,
    ece_improvement: float,
    n_total: int,
) -> str:
    """Map ECE metrics to a human-readable honest_verdict string.

    Args:
        ece_before: ECE before calibration.
        ece_after: ECE after isotonic regression.
        ece_improvement: ECE_before - ECE_after (positive = better).
        n_total: Total number of labeled steps used.

    Returns:
        One of: "insufficient_data", "energy_well_calibrated",
                "calibration_improved", "calibration_marginal",
                "calibration_no_improvement".
    """
    if n_total < 20:
        return "insufficient_data"
    if ece_before <= 0.10:
        return "energy_well_calibrated"
    if ece_improvement >= 0.05:
        return "calibration_improved"
    if ece_improvement > 0:
        return "calibration_marginal"
    return "calibration_no_improvement"


def run_experiment(tmpl: ExperimentTemplate) -> Dict[str, Any]:
    """Execute EBM calibration alignment experiment.

    Steps:
    1. Load pooled FoVer labeled steps.
    2. Compute Carnot energy proxy for each step.
    3. Measure ECE_before.
    4. Fit isotonic regression.
    5. Apply calibration -> calibrated_probs.
    6. Measure ECE_after.
    7. Save calibration curve.
    8. Classify verdict.
    """
    repo_root = _REPO

    # Step 1: Load labeled steps
    steps = load_labeled_steps(repo_root)
    n_total = len(steps)
    texts = [s[0] for s in steps]
    labels = [s[1] for s in steps]

    if n_total < 20:
        return tmpl.build_result(
            {
                "n_total_steps": n_total,
                "ECE_before": None,
                "ECE_after": None,
                "ece_improvement": None,
                "calibration_curve_saved": False,
                "honest_verdict": "insufficient_data",
            },
            status="blocked",
        )

    # Step 2: Compute energy proxy
    energies_np = _tfidf_energy_proxy(texts)
    energies_list = energies_np.tolist()

    calibrator = EBMCalibrator(n_bins=10)

    # Step 3: ECE before calibration
    ece_before = calibrator.compute_ece(energies_list, labels)

    # Step 4: Fit isotonic regression
    iso_reg = calibrator.fit_isotonic(energies_list, labels)

    # Step 5: Apply calibration
    calibrated_probs = iso_reg.predict(-energies_np).tolist()

    # Step 6: ECE after
    ece_after = calibrator.compute_ece_from_probs(calibrated_probs, labels)

    ece_improvement = round(ece_before - ece_after, 6)

    # Step 7: Save calibration curve
    curve_path = str(repo_root / _CURVE_PATH)
    bins = calibrator.build_curve(energies_list, labels)
    calibrator.save_curve(bins, curve_path)
    calibration_curve_saved = Path(curve_path).exists()

    # Step 8: Classify verdict
    honest_verdict = classify_verdict(ece_before, ece_after, ece_improvement, n_total)

    return tmpl.build_result(
        {
            "n_total_steps": n_total,
            "ECE_before": round(ece_before, 6),
            "ECE_after": round(ece_after, 6),
            "ece_improvement": ece_improvement,
            "calibration_curve_saved": calibration_curve_saved,
            "honest_verdict": honest_verdict,
        },
        status="success",
    )


def main() -> None:
    """Entry point for Exp 789."""
    tmpl = ExperimentTemplate(
        789,
        "EBM Calibration Alignment — ECE + Isotonic Regression",
        _DELIVERABLE,
    )
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(789, timeout_minutes=20, result_path=_DELIVERABLE)
    with watchdog:
        artifact = run_experiment(tmpl)

    with open(_REPO / _DELIVERABLE, "w") as f:
        json.dump(artifact, f, indent=2)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
