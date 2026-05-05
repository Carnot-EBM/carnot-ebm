"""Tests for Exp 1384 EBM-CoT KAN energy calibration.

Spec: REQ-KAN-1384, SCENARIO-KAN-1384
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
    EBMCoTKANEnergyCalibrator,
    FoVerStepCase,
    KANCheckpointInfo,
    build_artifact,
    ebm_cot_loss_components,
    encode_fover_features,
    load_fover_verified_cases,
    make_balanced_split,
)


def test_loss_components_match_ebm_cot_hinge_formula():
    """REQ-KAN-1384: loss is hinge gap plus positive paraphrase consistency."""

    components = ebm_cot_loss_components(
        e_positive=np.array([0.2, 0.4]),
        e_negative=np.array([1.5, 0.9]),
        e_positive_paraphrase=np.array([0.1, 0.7]),
        hinge_margin=1.0,
        consistency_weight=0.1,
    )

    # Pair 1 has gap 1.3 -> zero hinge; pair 2 has gap 0.5 -> hinge 0.5.
    assert components["contrastive"] == 0.25
    assert abs(components["consistency"] - 0.2) < 1e-12
    assert abs(components["total"] - 0.27) < 1e-12
    assert abs(components["mean_energy_gap"] - 0.9) < 1e-12


def test_load_fover_cases_accepts_jsonl_labels(tmp_path: Path):
    """REQ-KAN-1384: FoVer rows load from local labels without LLM inference."""

    path = tmp_path / "fover.jsonl"
    rows = [
        {"question_id": "a", "question": "Q1", "step_text": "1 + 1 = 2", "label": "correct"},
        {"question_id": "b", "question": "Q2", "step_text": "2 + 2 = 5", "label": "incorrect"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    cases = load_fover_verified_cases(path)

    assert [case.label for case in cases] == [1, 0]
    assert cases[0].step_text == "1 + 1 = 2"


def test_balanced_split_uses_equal_positive_negative_counts():
    """SCENARIO-KAN-1384: train pairs are balanced correct/incorrect steps."""

    cases = [
        FoVerStepCase(str(i), "q", f"correct {i}", 1)
        for i in range(6)
    ] + [
        FoVerStepCase(str(i + 10), "q", f"wrong {i}", 0)
        for i in range(4)
    ]

    split = make_balanced_split(cases, test_fraction=0.25, seed=1)

    assert len(split.train_positive) == 3
    assert len(split.train_negative) == 3
    assert len(split.test_cases) == 2
    assert split.corpus_cases_used == 8


def test_encode_fover_features_marks_arithmetic_error():
    """REQ-KAN-1384: feature extraction exposes arithmetic error signal to KAN."""

    correct = FoVerStepCase("ok", "Alice has 2 and gets 2.", "2 + 2 = 4", 1)
    wrong = FoVerStepCase("bad", "Alice has 2 and gets 2.", "2 + 2 = 5", 0)

    correct_features = encode_fover_features(correct)
    wrong_features = encode_fover_features(wrong)

    assert correct_features.shape == (32,)
    assert wrong_features.shape == (32,)
    assert correct_features[4] == 0.0
    assert wrong_features[4] == 1.0
    assert np.all((wrong_features >= 0.0) & (wrong_features <= 1.0))


def test_load_current_checkpoint_restores_compatible_kan_json(tmp_path: Path):
    """REQ-KAN-1384: compatible local KAN checkpoint warm-starts the probe."""

    n_features = 32
    n_hidden = 2
    n_knots = 4
    degree = 1
    payload = {
        "schema": "test.kan",
        "n_features": n_features,
        "n_hidden": n_hidden,
        "n_knots": n_knots,
        "degree": degree,
        "edge_ctrl": np.zeros((n_hidden, n_features, n_knots + degree)).tolist(),
        "output_ctrl": np.zeros((n_hidden, n_knots + degree)).tolist(),
    }
    (tmp_path / "prompt_injection_kan_weights.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    calibrator = EBMCoTKANEnergyCalibrator.load_current_checkpoint(tmp_path)

    assert calibrator.checkpoint_info.loaded is True
    assert calibrator.checkpoint_info.schema == "test.kan"
    assert calibrator.edge_ctrl.shape == (n_hidden, n_features, n_knots + degree)


def test_build_artifact_sets_viability_from_auroc_delta():
    """SCENARIO-KAN-1384: viability is true exactly when AUROC improves."""

    split = make_balanced_split(
        [
            FoVerStepCase("p0", "q", "1 + 1 = 2", 1),
            FoVerStepCase("p1", "q", "2 + 2 = 4", 1),
            FoVerStepCase("p2", "q", "3 + 3 = 6", 1),
            FoVerStepCase("p3", "q", "4 + 4 = 8", 1),
            FoVerStepCase("n0", "q", "1 + 1 = 3", 0),
            FoVerStepCase("n1", "q", "2 + 2 = 5", 0),
            FoVerStepCase("n2", "q", "3 + 3 = 9", 0),
            FoVerStepCase("n3", "q", "4 + 4 = 9", 0),
        ],
        test_fraction=0.25,
    )

    artifact = build_artifact(
        split=split,
        checkpoint_info=KANCheckpointInfo(True, "model.json", "schema", "loaded"),
        baseline_auroc=0.50,
        ebm_cot_auroc=0.75,
        variance_before=0.20,
        variance_after=0.05,
        loss_history=[{"loss": 1.0}],
        started_at=0.0,
    )

    assert artifact["status"] == "complete"
    assert artifact["calibration_auroc_delta"] == 0.25
    assert artifact["implicit_cot_energy_viable"] is True
    assert artifact["consistency_regularization_effect"] == 0.15000000000000002
