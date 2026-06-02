"""Tests for Exp 3683 detector code operating point hardening.

Spec: REQ-SPOE-3683, SCENARIO-SPOE-3683.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from carnot.pipeline import detector_code_operating_point_3683 as exp


def _metric(point: float, low: float, high: float) -> dict[str, object]:
    return {
        "point": point,
        "ci95": [low, high],
        "n": 20,
        "n_positive_errors": 10,
        "n_negative_correct": 10,
        "bootstrap_seeds": [3683],
        "seed_mean_aurocs": [point],
    }


def _baseline() -> dict[str, object]:
    return {
        "fused": _metric(0.5, 0.38, 0.62),
        "ensemble": _metric(0.5, 0.38, 0.62),
        "confidence": _metric(0.5, 0.38, 0.62),
        "calibration_brier_ece": {"brier": 0.31, "ece": 0.28},
        "n_holdout": 20,
    }


@pytest.mark.parametrize(
    (
        "case_name",
        "blocked",
        "recalibrated",
        "calibration_after",
        "expected_verdict",
        "expected_recovered",
    ),
    [
        (
            "code_operating_point_recovered",
            False,
            _metric(0.72, 0.61, 0.83),
            {"brier": 0.18, "ece": 0.07},
            "complete: code_operating_point_recovered_detector_now_math_and_code",
            True,
        ),
        (
            "code_remains_math_only",
            False,
            _metric(0.56, 0.43, 0.69),
            {"brier": 0.19, "ece": 0.08},
            "complete: code_remains_math_only_detector_scoped_honestly",
            False,
        ),
        (
            "blocked",
            True,
            {},
            {},
            "complete: blocked_no_balanced_code_corpus_or_detector_module",
            False,
        ),
    ],
)
def test_scenario_spoe_3683_parametrized_honest_outcomes(
    case_name: str,
    blocked: bool,
    recalibrated: dict[str, object],
    calibration_after: dict[str, float],
    expected_verdict: str,
    expected_recovered: bool,
) -> None:
    """SCENARIO-SPOE-3683: synthetic fixtures cover recovered, math-only, blocked."""

    artifact = exp.build_artifact_from_metrics(
        blocked=blocked,
        code_auroc_baseline={} if blocked else _baseline(),
        code_auroc_dependency_aware={} if blocked else _metric(0.64, 0.51, 0.76),
        code_auroc_recalibrated=recalibrated,
        code_calibration_brier_ece_after=calibration_after,
        code_recall_at_fixed_fpr={} if blocked else {"0.10": {"fused_recall": 0.4}},
        n_examples_code=0 if blocked else 60,
        module_code_path_updated=not blocked,
        e2e_test_passed=not blocked,
        started_s=1.0,
        now_s=2.25,
        tests_run=[f"SCENARIO-SPOE-3683 {case_name}"],
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["code_operating_point_recovered"] is expected_recovered
    assert type(artifact["code_operating_point_recovered"]) is bool
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["tests_run"] == [f"SCENARIO-SPOE-3683 {case_name}"]
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)


def test_req_spoe_3683_chance_floor_and_calibration_gates() -> None:
    """REQ-SPOE-3683: AUROC must exclude chance and calibration must improve."""

    assert exp.auroc_signal_excludes_chance(_metric(0.72, 0.51, 0.9)) is True
    assert exp.auroc_signal_excludes_chance(_metric(0.72, 0.5, 0.9)) is False
    assert exp.auroc_signal_excludes_chance(_metric(0.49, 0.4, 0.6)) is False

    before = {"brier": 0.3, "ece": 0.25}
    assert exp.calibration_improved(before, {"brier": 0.2, "ece": 0.1}) is True
    assert exp.calibration_improved(before, {"brier": 0.31, "ece": 0.1}) is False
    assert exp.calibration_improved(before, {"brier": 0.2, "ece": 0.26}) is False


def test_req_spoe_3683_metric_helpers_and_code_recalibration() -> None:
    """REQ-SPOE-3683: metrics are measured from scores, not replayed constants."""

    labels = [1, 1, 0, 0, 1, 0]
    scores = [0.9, 0.8, 0.3, 0.2, 0.7, 0.1]
    metric = exp.auroc_metric(labels, scores, seeds=[1, 2], n_bootstrap=8)
    calibration = exp.calibration_bundle(labels, scores)

    assert metric["point"] == pytest.approx(1.0)
    assert metric["n"] == 6
    assert metric["n_positive_errors"] == 3
    assert calibration["brier"] < 0.1
    assert calibration["ece"] >= 0.0

    recalibrated = exp.measure_code_recalibration(
        labels=[1, 1, 1, 0, 0, 0, 1, 0],
        ensemble_scores=[0.95, 0.9, 0.88, 0.08, 0.12, 0.2, 0.85, 0.1],
        confidence_scores=[0.6, 0.55, 0.65, 0.35, 0.4, 0.3, 0.7, 0.25],
        seeds=[3],
        n_bootstrap=8,
    )

    assert recalibrated["code_auroc_recalibrated"]["point"] >= 0.5
    assert set(recalibrated["code_recall_at_fixed_fpr"]) == {"0.05", "0.10", "0.20"}
    assert recalibrated["code_calibration_brier_ece_after"]["brier"] >= 0.0


def test_req_spoe_3683_validation_and_write_artifact(tmp_path: Path) -> None:
    """REQ-SPOE-3683: artifact writing and bare-bool validation are strict."""

    output = exp.write_artifact_from_metrics(
        tmp_path,
        output_path="results/exp3683.json",
        blocked=False,
        code_auroc_baseline=_baseline(),
        code_auroc_dependency_aware=_metric(0.64, 0.51, 0.76),
        code_auroc_recalibrated=_metric(0.72, 0.61, 0.83),
        code_calibration_brier_ece_after={"brier": 0.18, "ece": 0.07},
        code_recall_at_fixed_fpr={"0.10": {"fused_recall": 0.4}},
        n_examples_code=60,
        module_code_path_updated=True,
        e2e_test_passed=True,
        started_s=0.0,
        now_s=1.0,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["code_operating_point_recovered"] is True
    assert artifact["acceptance_gate"]["passed"] is True

    broken = dict(artifact)
    broken["code_operating_point_recovered"] = {"value": True}
    with pytest.raises(ValueError, match="code_operating_point_recovered"):
        exp.validate_artifact(broken)

    missing = dict(artifact)
    missing.pop("code_auroc_baseline")
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="complete: unexpected")
    with pytest.raises(ValueError, match="terminal verdict"):
        exp.validate_artifact(bad_verdict)

    bad_duration = dict(artifact, duration_s=-1.0)
    with pytest.raises(ValueError, match="duration_s"):
        exp.validate_artifact(bad_duration)


def test_req_spoe_3683_preconditions_block_missing_corpus(tmp_path: Path) -> None:
    """REQ-SPOE-3683: missing balanced corpus produces the blocked terminal artifact."""

    artifact = exp.build_artifact(tmp_path, started_s=0.0, now_s=1.0)

    assert artifact["honest_verdict"] == "complete: blocked_no_balanced_code_corpus_or_detector_module"
    assert artifact["n_examples_code"] == 0
    assert artifact["code_auroc_baseline"] == {}
    assert artifact["module_code_path_updated"] is False


def test_req_spoe_3683_branch_guards_and_io_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SPOE-3683: helper guards fail closed on malformed or partial inputs."""

    assert exp._load_second_pair_detector() is exp.spd
    saved_module = exp.sys.modules.pop(exp._SPD_MODULE_NAME, None)
    monkeypatch.setattr(exp.importlib.util, "spec_from_file_location", lambda *args: None)
    try:
        with pytest.raises(ImportError, match="second_pair_detector"):
            exp._load_second_pair_detector()
    finally:
        if saved_module is not None:
            exp.sys.modules[exp._SPD_MODULE_NAME] = saved_module

    assert exp.auroc_signal_excludes_chance({}) is False
    assert exp.calibration_improved({"brier": 0.2}, {"brier": 0.1}) is False
    assert exp.auroc_metric([1, 1], [0.8, 0.7], seeds=[1]) == exp.empty_metric([1])
    no_bootstrap = exp.auroc_metric([1, 0], [0.9, 0.1], seeds=[1], n_bootstrap=0)
    assert no_bootstrap["ci95"] == [1.0, 1.0]
    assert exp.empty_metric([5])["n"] == 0

    blocked_recalibration = exp.measure_code_recalibration(
        labels=[1, 1],
        ensemble_scores=[0.8, 0.7],
        confidence_scores=[0.5, 0.5],
        seeds=[1],
        n_bootstrap=1,
    )
    assert blocked_recalibration["code_recalibration_protocol"]["blocked_reason"]

    assert (
        exp.measure_baseline_code_operating_point(
            [exp.spd.LabeledDetectorExample("math", 1, 0.8, 0.5, "m")],
            seeds=[1],
            n_bootstrap=1,
        )
        == {}
    )

    from carnot.verify import code_corpus_verifiers_fire_transfer_v3 as code_transfer

    real_structural = exp.structural_dependency_scores_aligned
    monkeypatch.setattr(code_transfer, "ast_structure_scores", lambda rows: [0.1])
    monkeypatch.setattr(exp, "structural_dependency_scores_aligned", lambda rows, root: [math.nan])
    monkeypatch.setattr(code_transfer, "score_math_signal", lambda rows, score_overrides: [0.2])
    with pytest.raises(ValueError, match="finite"):
        exp.code_verifier_score_panel([{"candidate_code": "def f():\n    pass"}], tmp_path)

    monkeypatch.setattr(exp, "structural_dependency_scores_aligned", lambda rows, root: [0.1])
    with pytest.raises(ValueError, match="returned"):
        exp._require_length([0.1], 2, "fixture")

    monkeypatch.setattr(exp, "structural_dependency_scores_aligned", real_structural)
    monkeypatch.setattr(code_transfer, "load_manifest_lookup", lambda rows, root: {})
    assert exp.structural_dependency_scores_aligned(
        [{"candidate_code": "def f():\n    return 1", "metadata": {}}],
        tmp_path,
    ) == [0.0]

    def boom(root: Path):
        raise RuntimeError("fixture")

    monkeypatch.setattr(exp, "load_balanced_code_rows", boom)
    checks = exp.check_preconditions(tmp_path)
    assert checks[0]["available"] is False

    assert exp.run_score_candidates_e2e(tmp_path, []) is False

    code_example = exp.spd.LabeledDetectorExample("code", 1, 0.8, 0.5, "c")

    def raises(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("fixture")

    monkeypatch.setattr(exp.spd, "score_candidates", raises)
    assert exp.run_score_candidates_e2e(tmp_path, [code_example]) is False

    monkeypatch.setattr(exp.spd, "score_candidates", lambda *args, **kwargs: {"scores": []})
    assert exp.run_score_candidates_e2e(tmp_path, [code_example]) is False

    assert exp._precondition_n_examples([]) == 0
    assert exp._read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert exp._read_json_object(bad_json) == {}
    assert exp._read_jsonl(tmp_path / "missing.jsonl") == []
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('\n{\n{"ok": true}\n', encoding="utf-8")
    assert exp._read_jsonl(jsonl) == [{"ok": True}]
    assert exp._round(float("inf")) == float("inf")
