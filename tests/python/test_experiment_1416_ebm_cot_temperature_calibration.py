"""Tests for Exp 1416 EBM-CoT v3 post-hoc temperature calibration.

Spec: REQ-VERIFY-1416, SCENARIO-VERIFY-1416
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.models import ebm_cot_temperature_calibration as tempcal
from carnot.models.ebm_cot_energy_calibration_probe import FoVerSplit, FoVerStepCase
from carnot.models.ebm_cot_temperature_calibration import (
    DEFAULT_TEMPERATURE_CANDIDATES,
    TemperatureCalibrationScores,
    auroc_from_energies,
    apply_temperature_to_energies,
    build_temperature_calibration_artifact,
    calibrate_temperature_scores,
    fit_best_temperature,
    paraphrase_variance_after_temperature,
    regenerate_temperature_calibration_scores,
    run_temperature_calibration_pass,
    write_in_progress_artifact,
    write_temperature_calibration_artifact,
)


def _scores_for_tests() -> TemperatureCalibrationScores:
    return TemperatureCalibrationScores(
        validation_labels=np.array([1, 0], dtype=np.float64),
        validation_energies=np.array([-2.0, -4.0], dtype=np.float64),
        validation_paraphrase_deltas=np.array([2.0, -2.0], dtype=np.float64),
        test_labels=np.array([1, 1, 0, 0], dtype=np.float64),
        test_energies=np.array([-4.0, -2.0, 1.0, 3.0], dtype=np.float64),
        test_paraphrase_deltas=np.array([2.0, -2.0], dtype=np.float64),
        baseline_auroc=0.60,
        exp1401_reference_delta=0.18,
        corpus_cases_used=12,
        validation_cases_used=2,
        test_cases_used=4,
    )


def test_apply_temperature_to_energies_requires_positive_scalar():
    """REQ-VERIFY-1416: post-hoc scaling uses a single positive scalar T*."""

    scaled = apply_temperature_to_energies(np.array([2.0, -4.0]), 2.0)

    assert np.allclose(scaled, np.array([1.0, -2.0]))
    try:
        apply_temperature_to_energies(np.array([1.0]), 0.0)
    except ValueError as exc:
        assert "temperature" in str(exc)
    else:
        raise AssertionError("temperature <= 0 must be rejected")


def test_fit_best_temperature_uses_validation_scores_not_test_scores():
    """REQ-VERIFY-1416: T* is fit on validation scores, not held-out test scores."""

    validation_labels = np.array([1, 0], dtype=np.float64)
    validation_energies = np.array([-2.0, -4.0], dtype=np.float64)
    candidates = (1.0, 2.0, 4.0)

    fitted = fit_best_temperature(
        validation_labels,
        validation_energies,
        candidate_temperatures=candidates,
    )

    assert fitted.best_temperature == 4.0
    assert set(fitted.validation_losses) == set(candidates)


def test_invalid_inputs_raise_clear_errors():
    """REQ-VERIFY-1416: calibration helpers reject malformed score arrays."""

    with pytest.raises(ValueError, match="one-dimensional"):
        auroc_from_energies(np.array([[1, 0]]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="same length"):
        auroc_from_energies(np.array([1, 0]), np.array([1.0]))
    with pytest.raises(ValueError, match="same length"):
        fit_best_temperature(np.array([1, 0]), np.array([1.0]))
    with pytest.raises(ValueError, match="must not be empty"):
        fit_best_temperature(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="positive finite"):
        fit_best_temperature(np.array([1]), np.array([1.0]), candidate_temperatures=(0.0,))
    with pytest.raises(ValueError, match="at least one"):
        fit_best_temperature(np.array([1]), np.array([1.0]), candidate_temperatures=())

    assert auroc_from_energies(np.array([1, 1]), np.array([0.1, 0.2])) == 0.5
    assert paraphrase_variance_after_temperature(np.array([]), 2.0) == 0.0


def test_temperature_scaling_preserves_auroc_and_reduces_paraphrase_variance():
    """SCENARIO-VERIFY-1416: positive scaling preserves ranking and reduces variance."""

    result = calibrate_temperature_scores(
        _scores_for_tests(),
        candidate_temperatures=(1.0, 2.0, 4.0),
    )

    assert result.best_temperature == 4.0
    assert result.auroc_before == result.auroc_after
    assert result.calibration_auroc_delta_before == 0.4
    assert result.calibration_auroc_delta_after == 0.4
    assert result.variance_after_temp_scaling < result.variance_before_temp_scaling
    assert result.auroc_preserved is True
    assert result.variance_worsened is False


def test_split_and_verdict_edge_cases():
    """SCENARIO-VERIFY-1416: gates distinguish variance and AUROC failures."""

    split = FoVerSplit(
        train_positive=[FoVerStepCase("p0", "q", "ok", 1)],
        train_negative=[FoVerStepCase("n0", "q", "bad", 0)],
        test_cases=[],
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        tempcal._split_train_validation(split, validation_fraction=0.0)
    with pytest.raises(ValueError, match="consume all"):
        tempcal._split_train_validation(split, validation_fraction=0.5)

    base = calibrate_temperature_scores(_scores_for_tests(), candidate_temperatures=(1.0, 4.0))
    worsened = tempcal.TemperatureCalibrationResult(
        **{**base.__dict__, "variance_worsened": True}
    )
    lost_auroc = tempcal.TemperatureCalibrationResult(
        **{**base.__dict__, "auroc_preserved": False}
    )
    failed_both = tempcal.TemperatureCalibrationResult(
        **{**base.__dict__, "auroc_preserved": False, "variance_worsened": True}
    )

    assert tempcal._honest_verdict(worsened) == (
        "temperature_scaling_preserved_auroc_but_variance_worsened"
    )
    assert tempcal._honest_verdict(lost_auroc) == (
        "temperature_scaling_reduced_variance_but_lost_auroc"
    )
    assert tempcal._honest_verdict(failed_both) == (
        "temperature_scaling_failed_variance_and_auroc_gates"
    )


def test_build_temperature_calibration_artifact_sets_required_fields():
    """SCENARIO-VERIFY-1416: artifact exposes gates and required schema fields."""

    artifact = build_temperature_calibration_artifact(
        result=calibrate_temperature_scores(
            TemperatureCalibrationScores(
                validation_labels=np.array([1, 0], dtype=np.float64),
                validation_energies=np.array([-2.0, -4.0], dtype=np.float64),
                validation_paraphrase_deltas=np.array([2.0, -2.0], dtype=np.float64),
                test_labels=np.array([1, 0], dtype=np.float64),
                test_energies=np.array([-2.0, 2.0], dtype=np.float64),
                test_paraphrase_deltas=np.array([1.5, -1.5], dtype=np.float64),
                baseline_auroc=0.50,
                exp1401_reference_delta=0.18,
                corpus_cases_used=6,
                validation_cases_used=2,
                test_cases_used=2,
            ),
            candidate_temperatures=(1.0, 2.0, 4.0),
        ),
        exp1401_reference={
            "calibration_auroc_delta": 0.18,
            "ebm_cot_v2_auroc": 0.98,
            "paraphrase_energy_variance_after": 0.16,
        },
        duration_s=1.25,
    )

    required = {
        "status",
        "temperature_scaling_applied",
        "best_temperature",
        "calibration_auroc_delta_before",
        "calibration_auroc_delta_after",
        "paraphrase_energy_variance_before_temp_scaling",
        "paraphrase_energy_variance_after_temp_scaling",
        "variance_worsened",
        "auroc_preserved",
        "honest_verdict",
    }
    assert required <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["temperature_scaling_applied"] is True
    assert artifact["best_temperature"] == 4.0
    assert artifact["variance_worsened"] is False
    assert artifact["auroc_preserved"] is True
    assert artifact["honest_verdict"] == "temperature_scaling_reduced_variance_and_preserved_auroc"
    assert artifact["candidate_temperatures"] == list(DEFAULT_TEMPERATURE_CANDIDATES)


def test_build_artifact_records_noop_temperature_when_best_temperature_is_one():
    """REQ-VERIFY-1416: artifact truthfully records when the fitted T* is 1.0."""

    result = calibrate_temperature_scores(_scores_for_tests(), candidate_temperatures=(1.0,))
    artifact = build_temperature_calibration_artifact(
        result=result,
        exp1401_reference={"calibration_auroc_delta": 0.18},
        duration_s=0.5,
    )

    assert artifact["best_temperature"] == 1.0
    assert artifact["temperature_scaling_applied"] is False


def test_write_temperature_calibration_artifact_round_trips_json(tmp_path: Path):
    """REQ-VERIFY-1416: final JSON artifact is written with status complete."""

    target = tmp_path / "experiment_1416.json"
    artifact = {
        "status": "complete",
        "temperature_scaling_applied": True,
        "best_temperature": 2.0,
        "calibration_auroc_delta_before": 0.18,
        "calibration_auroc_delta_after": 0.18,
        "paraphrase_energy_variance_before_temp_scaling": 0.16,
        "paraphrase_energy_variance_after_temp_scaling": 0.04,
        "variance_worsened": False,
        "auroc_preserved": True,
        "honest_verdict": "temperature_scaling_reduced_variance_and_preserved_auroc",
    }

    write_temperature_calibration_artifact(target, artifact)

    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == artifact


def test_write_in_progress_artifact_round_trips_bootstrap_json(tmp_path: Path):
    """REQ-VERIFY-1416: runner can write the required in-progress bootstrap."""

    target = tmp_path / "experiment_1416.json"

    write_in_progress_artifact(target)

    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["status"] == "in_progress"
    assert loaded["honest_verdict"] == "in_progress"


def test_regenerate_scores_uses_validation_slice_without_fresh_inference(monkeypatch):
    """REQ-VERIFY-1416: score regeneration uses local FoVer rows and a validation split."""

    split = FoVerSplit(
        train_positive=[
            FoVerStepCase("p0", "q", "ok 0", 1),
            FoVerStepCase("p1", "q", "ok 1", 1),
            FoVerStepCase("p2", "q", "ok 2", 1),
            FoVerStepCase("p3", "q", "ok 3", 1),
        ],
        train_negative=[
            FoVerStepCase("n0", "q", "bad 0", 0),
            FoVerStepCase("n1", "q", "bad 1", 0),
            FoVerStepCase("n2", "q", "bad 2", 0),
            FoVerStepCase("n3", "q", "bad 3", 0),
        ],
        test_cases=[
            FoVerStepCase("pt", "q", "ok test", 1),
            FoVerStepCase("nt", "q", "bad test", 0),
        ],
    )

    class FakeCalibrator:
        def evaluate_auroc(self, cases):
            assert cases == split.test_cases
            return 0.5

        def train_ebm_cot(self, positive_cases, negative_cases, **kwargs):
            assert [case.case_id for case in positive_cases] == ["p0", "p1", "p2"]
            assert [case.case_id for case in negative_cases] == ["n0", "n1", "n2"]
            assert kwargs["consistency_weight"] == 0.0
            return []

        def energy(self, case):
            base = -2.0 if case.label == 1 else 2.0
            return base + (0.5 if case.case_id.endswith(":paraphrase") else 0.0)

    loader = type(
        "FakeLoader",
        (),
        {"load_current_checkpoint": staticmethod(lambda models_dir: FakeCalibrator())},
    )
    monkeypatch.setattr(tempcal, "load_fover_verified_cases", lambda path: [])
    monkeypatch.setattr(tempcal, "make_balanced_split", lambda cases, **kwargs: split)
    monkeypatch.setattr(tempcal, "EBMCoTKANEnergyCalibrator", loader)

    scores = regenerate_temperature_calibration_scores(
        exp1401_reference={"calibration_auroc_delta": 0.18},
        validation_fraction=0.25,
    )

    assert scores.validation_cases_used == 2
    assert scores.test_cases_used == 2
    assert scores.corpus_cases_used == 10
    assert np.allclose(scores.test_paraphrase_deltas, np.array([-0.5]))


def test_run_temperature_calibration_pass_writes_final_artifact_with_injected_scores(
    tmp_path: Path,
):
    """REQ-VERIFY-1416: runner writes complete artifact after in-progress bootstrap."""

    target = tmp_path / "experiment_1416.json"

    artifact = run_temperature_calibration_pass(
        artifact_path=target,
        scores=_scores_for_tests(),
        exp1401_reference={"calibration_auroc_delta": 0.18, "ebm_cot_v2_auroc": 0.98},
        candidate_temperatures=(1.0, 2.0, 4.0),
    )

    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert artifact == loaded
    assert loaded["status"] == "complete"
    assert loaded["test_split_used_for_temperature_fit"] is False


def test_run_temperature_calibration_pass_can_load_reference_and_regenerate_scores(
    tmp_path: Path,
    monkeypatch,
):
    """REQ-VERIFY-1416: production branch loads Exp1401 JSON and regenerates scores."""

    target = tmp_path / "experiment_1416.json"
    reference_path = tmp_path / "experiment_1401.json"
    reference_path.write_text(
        json.dumps({"calibration_auroc_delta": 0.18, "ebm_cot_v2_auroc": 0.98}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        tempcal,
        "regenerate_temperature_calibration_scores",
        lambda **kwargs: _scores_for_tests(),
    )

    artifact = run_temperature_calibration_pass(
        artifact_path=target,
        exp1401_artifact_path=reference_path,
        candidate_temperatures=(1.0, 2.0, 4.0),
    )

    assert artifact["exp1401_reference_calibration_auroc_delta"] == 0.18
    assert json.loads(target.read_text(encoding="utf-8"))["status"] == "complete"
