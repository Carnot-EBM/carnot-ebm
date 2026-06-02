"""Tests for Exp 3696 math+code detector re-ship.

Spec: REQ-SPOE-3696, SCENARIO-SPOE-3696.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.pipeline import reship_detector_math_plus_code_3696 as exp
from carnot.pipeline import second_pair_detector as spd


def _ship_artifact(*, code_auroc: float, math_auroc: float, math_ece: float) -> dict[str, object]:
    return {
        "honest_verdict": "complete: fixture",
        "fused_detector_auroc_per_domain": {"code": code_auroc, "math": math_auroc},
        "calibration_brier_ece_per_domain": {
            "code": {"brier": 0.04, "ece": 0.03},
            "math": {"brier": 0.014, "ece": math_ece},
        },
        "operating_points": {
            "code": {"threshold": 0.98, "fpr_budget": 0.1, "expected_recall": 1.0},
            "math": {"threshold": 0.013, "fpr_budget": 0.1, "expected_recall": 0.97},
        },
        "e2e_test_passed": True,
        "detector_shipped": True,
        "n_examples_per_domain": {"code": 60, "math": 6548},
    }


@pytest.mark.parametrize(
    (
        "case_name",
        "blocked",
        "ship_artifact",
        "baseline_artifact",
        "module_updated",
        "e2e_passed",
        "expected_verdict",
    ),
    [
        (
            "detector_math_plus_code_shipped",
            False,
            _ship_artifact(code_auroc=1.0, math_auroc=0.979656, math_ece=0.0083),
            _ship_artifact(code_auroc=0.5, math_auroc=0.979656, math_ece=0.0086),
            True,
            True,
            "complete: detector_reshipped_math_plus_code_operating_point_e2e_green",
        ),
        (
            "blocked",
            True,
            {},
            {},
            False,
            False,
            "complete: blocked_code_signal_not_recovered_or_module_unavailable",
        ),
    ],
)
def test_scenario_spoe_3696_parametrized_honest_outcomes(
    case_name: str,
    blocked: bool,
    ship_artifact: dict[str, object],
    baseline_artifact: dict[str, object],
    module_updated: bool,
    e2e_passed: bool,
    expected_verdict: str,
) -> None:
    """SCENARIO-SPOE-3696: fixtures cover shipped and blocked outcomes."""

    artifact = exp.build_artifact_from_measurements(
        blocked=blocked,
        ship_artifact=ship_artifact,
        baseline_ship_artifact=baseline_artifact,
        module_code_path_updated=module_updated,
        e2e_test_passed=e2e_passed,
        adversarial_verify_clean=not blocked,
        started_s=1.0,
        now_s=2.25,
        tests_run=[f"SCENARIO-SPOE-3696 {case_name}"],
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["module_code_path_updated"] is module_updated
    assert artifact["e2e_test_passed"] is e2e_passed
    assert type(artifact["module_code_path_updated"]) is bool
    assert type(artifact["math_operating_point_unchanged"]) is bool
    assert type(artifact["e2e_test_passed"]) is bool
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["tests_run"] == [f"SCENARIO-SPOE-3696 {case_name}"]
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)


def test_req_spoe_3696_code_loader_uses_exp3695_code_native_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SPOE-3696: balanced code rows use the code-native verifier score."""

    data = tmp_path / "data"
    data.mkdir()
    (data / "code_verification_corpus_v2.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"candidate_code": "def ok():\n    return 1", "label": True}),
                json.dumps({"candidate_code": "def bad(:", "label": False}),
            ]
        ),
        encoding="utf-8",
    )

    class FakeVerifier:
        def score_rows(self, rows):
            return [
                SimpleNamespace(score=0.11 + idx * 0.7)
                for idx, _row in enumerate(rows)
            ]

    monkeypatch.setattr(spd.code_native_verifier_3695, "CodeNativeVerifier", FakeVerifier)
    examples, status = spd.load_cached_labeled_examples(
        tmp_path,
        use_balanced_code_corpus=True,
    )

    assert status["code"]["code_native_exp3695"] is True
    assert [example.ensemble_energy for example in examples] == pytest.approx([0.11, 0.81])


def test_req_spoe_3696_runtime_code_candidate_uses_code_native_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SPOE-3696: runtime code candidates use the code-native verifier score."""

    class FakeVerifier:
        def score_rows(self, rows):
            assert rows[0]["candidate_code"] == "def f():\n    return None"
            return [SimpleNamespace(score=0.73)]

    monkeypatch.setattr(spd.code_native_verifier_3695, "CodeNativeVerifier", FakeVerifier)
    candidate = spd.CandidateScoreInput(
        candidate_id="candidate",
        domain="code",
        text="def f():\n    return None",
    )

    assert spd._candidate_ensemble_energy(candidate, Path(".")) == pytest.approx(0.73)


def test_req_spoe_3696_validation_and_write_artifact(tmp_path: Path) -> None:
    """REQ-SPOE-3696: artifact writing and bare-bool validation are strict."""

    output = exp.write_artifact_from_measurements(
        tmp_path,
        output_path="results/exp3696.json",
        blocked=False,
        ship_artifact=_ship_artifact(code_auroc=1.0, math_auroc=0.979656, math_ece=0.0083),
        baseline_ship_artifact=_ship_artifact(
            code_auroc=0.5,
            math_auroc=0.979656,
            math_ece=0.0086,
        ),
        module_code_path_updated=True,
        e2e_test_passed=True,
        adversarial_verify_clean=True,
        started_s=0.0,
        now_s=1.0,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["acceptance_gate"]["passed"] is True
    assert artifact["code_operating_point_auroc"] == pytest.approx(1.0)

    broken = dict(artifact, module_code_path_updated={"value": True})
    with pytest.raises(ValueError, match="module_code_path_updated"):
        exp.validate_artifact(broken)

    missing = dict(artifact)
    missing.pop("code_operating_point_auroc")
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="complete: unexpected")
    with pytest.raises(ValueError, match="terminal verdict"):
        exp.validate_artifact(bad_verdict)

    bad_duration = dict(artifact, duration_s=-1.0)
    with pytest.raises(ValueError, match="duration_s"):
        exp.validate_artifact(bad_duration)


def test_req_spoe_3696_blocked_precondition_for_unrecovered_code_signal(
    tmp_path: Path,
) -> None:
    """REQ-SPOE-3696: missing or unrecovered Exp 3695 precondition blocks."""

    artifact = exp.build_artifact(tmp_path, started_s=0.0, now_s=1.0)

    assert artifact["honest_verdict"] == (
        "complete: blocked_code_signal_not_recovered_or_module_unavailable"
    )
    assert artifact["module_code_path_updated"] is False
    assert artifact["code_operating_point_auroc"] is None


def test_req_spoe_3696_success_build_and_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SPOE-3696: success build path and defensive helpers stay covered."""

    (tmp_path / "results").mkdir()
    exp3695 = tmp_path / "results/experiment_3695_code_native_verifier.json"
    exp3695.write_text(json.dumps({"code_signal_recovered": True}), encoding="utf-8")
    baseline = _ship_artifact(code_auroc=0.5, math_auroc=0.979656, math_ece=0.0086)
    (tmp_path / "results/experiment_3671_ship_second_pair_of_eyes_detector.json").write_text(
        json.dumps(baseline),
        encoding="utf-8",
    )
    ship = _ship_artifact(code_auroc=1.0, math_auroc=0.979656, math_ece=0.0083)
    monkeypatch.setattr(exp.spd, "build_ship_artifact", lambda *args, **kwargs: dict(ship))

    artifact = exp.build_artifact(tmp_path, started_s=0.0, now_s=1.0)

    assert artifact["honest_verdict"] == (
        "complete: detector_reshipped_math_plus_code_operating_point_e2e_green"
    )
    assert artifact["ship_artifact_summary"]["fused_detector_auroc_per_domain"]["code"] == 1.0
    assert exp.module_code_path_updated() is True
    assert exp.math_operating_point_unchanged({}, baseline) is False
    assert exp.compact_adversarial_report({"flags": [{"severity": "warn"}, "skip"]}) == {
        "flag_count": 1,
        "flags": [{"severity": "warn"}],
    }
    assert exp.adversarial_report_is_clean({"flags": 3}) is False
    assert exp.adversarial_report_is_clean(
        {"flags": [{"severity": "critical"}]}
    ) is False
    assert exp._round(float("inf")) == float("inf")

    bad_auroc = dict(artifact, code_operating_point_auroc="1.0")
    with pytest.raises(ValueError, match="code_operating_point_auroc"):
        exp.validate_artifact(bad_auroc)


def test_req_spoe_3696_write_artifact_runs_adversarial_verify(tmp_path: Path) -> None:
    """REQ-SPOE-3696: write_artifact performs the adversarial cleanliness pass."""

    output = exp.write_artifact(
        tmp_path,
        output_path="results/blocked-exp3696.json",
        tests_run=["REQ-SPOE-3696 write_artifact"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["tests_run"] == ["REQ-SPOE-3696 write_artifact"]
    assert "adversarial_verify_report" in artifact
    assert "flags" in exp.run_adversarial_verify_report(output)

    saved = exp.importlib.util.spec_from_file_location
    try:
        exp.importlib.util.spec_from_file_location = lambda *args, **kwargs: None
        with pytest.raises(ImportError, match="adversarial_verify"):
            exp.run_adversarial_verify_report(output)
    finally:
        exp.importlib.util.spec_from_file_location = saved
