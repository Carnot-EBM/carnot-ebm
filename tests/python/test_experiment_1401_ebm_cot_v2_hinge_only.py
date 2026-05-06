"""Tests for Exp 1401 EBM-CoT v2 hinge-only KAN calibration.

Spec: REQ-KAN-1401, SCENARIO-KAN-1401
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.models.ebm_cot_energy_calibration_probe import (  # noqa: E402
    FoVerSplit,
    FoVerStepCase,
    KANCheckpointInfo,
    build_hinge_only_v2_artifact,
    ebm_cot_loss_components,
    load_exp1384_reference,
)


def _tiny_split() -> FoVerSplit:
    return FoVerSplit(
        train_positive=[FoVerStepCase("p0", "q", "1 + 1 = 2", 1)],
        train_negative=[FoVerStepCase("n0", "q", "1 + 1 = 3", 0)],
        test_cases=[
            FoVerStepCase("p1", "q", "2 + 2 = 4", 1),
            FoVerStepCase("n1", "q", "2 + 2 = 5", 0),
        ],
    )


def test_hinge_only_loss_total_ignores_consistency_component():
    """REQ-KAN-1401: v2 training objective is hinge-only when weight is zero."""

    components = ebm_cot_loss_components(
        e_positive=np.array([0.0, 0.0]),
        e_negative=np.array([0.25, 2.0]),
        e_positive_paraphrase=np.array([10.0, -10.0]),
        hinge_margin=1.0,
        consistency_weight=0.0,
    )

    assert components["consistency"] == 10.0
    assert components["contrastive"] == 0.375
    assert components["total"] == components["contrastive"]


def test_load_exp1384_reference_reads_baseline_and_consistency_details(tmp_path: Path):
    """REQ-KAN-1401: v2 delta is anchored to the Exp1384 baseline artifact."""

    path = tmp_path / "experiment_1384.json"
    path.write_text(
        json.dumps(
            {
                "baseline_auroc": 0.799,
                "consistency_regularization_effect": -0.061,
                "consistency_regularization_weight": 0.1,
                "checkpoint_path": "model.json",
            }
        ),
        encoding="utf-8",
    )

    reference = load_exp1384_reference(path)

    assert reference["baseline_auroc"] == 0.799
    assert reference["consistency_regularization_effect"] == -0.061
    assert reference["consistency_regularization_weight"] == 0.1


def test_build_hinge_only_v2_artifact_uses_required_fields_and_viability_gate():
    """SCENARIO-KAN-1401: artifact reports v2 AUROC, delta, and variance gate."""

    artifact = build_hinge_only_v2_artifact(
        split=_tiny_split(),
        checkpoint_info=KANCheckpointInfo(True, "model.json", "schema", "loaded"),
        exp1384_reference={
            "baseline_auroc": 0.70,
            "consistency_regularization_effect": -0.05,
            "consistency_regularization_weight": 0.1,
        },
        ebm_cot_v2_auroc=0.80,
        variance_before=0.20,
        variance_after=0.10,
        loss_history=[{"loss": 0.5, "total": 0.5}],
        started_at=0.0,
        duration_s=12.5,
    )

    required = {
        "status",
        "corpus_cases_used",
        "training_method",
        "hinge_margin",
        "consistency_regularization_weight",
        "baseline_auroc",
        "ebm_cot_v2_auroc",
        "calibration_auroc_delta",
        "paraphrase_energy_variance_before",
        "paraphrase_energy_variance_after",
        "variance_worsened",
        "implicit_cot_energy_viable",
        "honest_verdict",
    }
    assert required <= set(artifact)
    assert artifact["run_date"] == "20260506"
    assert artifact["consistency_regularization_weight"] == 0.0
    assert artifact["baseline_auroc"] == 0.70
    assert artifact["ebm_cot_v2_auroc"] == 0.80
    assert artifact["calibration_auroc_delta"] == 0.10000000000000009
    assert artifact["variance_worsened"] is False
    assert artifact["implicit_cot_energy_viable"] is True
    assert (
        artifact["honest_verdict"]
        == "hinge_only_confirmed_positive_calibration_without_variance_worsening"
    )


def test_build_hinge_only_v2_artifact_marks_variance_worsened():
    """SCENARIO-KAN-1401: variance_worsened follows after > before exactly."""

    artifact = build_hinge_only_v2_artifact(
        split=_tiny_split(),
        checkpoint_info=KANCheckpointInfo(True, "model.json", "schema", "loaded"),
        exp1384_reference={"baseline_auroc": 0.70},
        ebm_cot_v2_auroc=0.69,
        variance_before=0.01,
        variance_after=0.02,
        loss_history=[],
        started_at=0.0,
        duration_s=1.0,
    )

    assert artifact["variance_worsened"] is True
    assert artifact["implicit_cot_energy_viable"] is False
    assert (
        artifact["honest_verdict"]
        == "hinge_only_did_not_confirm_positive_calibration_variance_worsened"
    )
