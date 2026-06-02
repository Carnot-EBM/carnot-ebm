"""Tests for Exp 3706 held-out reconciliation of the shipped detector.

Spec: REQ-SPOE-3706, SCENARIO-SPOE-3706.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline import reconcile_shipped_detector_heldout_3706 as exp
from carnot.pipeline import second_pair_detector as spd


def _audit(
    *,
    survives: bool,
    leak_detected: bool,
    heldout_auroc: float = 0.74,
) -> dict[str, object]:
    return {
        "honest_verdict": "complete: fixture",
        "code_signal_survives_heldout": survives,
        "leak_detected": leak_detected,
        "heldout_code_auroc": heldout_auroc,
        "heldout_calibration_brier_ece": {"brier": 0.18, "ece": 0.08},
        "adversarial_verify_clean": True,
    }


def _ship_artifact(*, e2e: bool = True) -> dict[str, object]:
    return {
        "honest_verdict": "complete: fixture",
        "fused_detector_auroc_per_domain": {"math": 0.979656},
        "calibration_brier_ece_per_domain": {"math": {"brier": 0.0135, "ece": 0.0085}},
        "operating_points": {
            "math": {
                "threshold": 0.013,
                "fpr_budget": 0.1,
                "expected_recall": 0.970588,
            }
        },
        "e2e_test_passed": e2e,
    }


@pytest.mark.parametrize(
    (
        "case_name",
        "audit_artifact",
        "module_available",
        "code_surface_abstains",
        "expected_action",
        "expected_verdict",
        "expected_code_auroc",
    ),
    [
        (
            "code_operating_point_recalibrated_to_heldout",
            _audit(survives=True, leak_detected=False, heldout_auroc=0.742),
            True,
            False,
            "recalibrated_to_heldout",
            "complete: shipped_detector_code_recalibrated_to_heldout_e2e_green",
            0.742,
        ),
        (
            "narrowed_to_math_only_abstain_on_code",
            _audit(survives=False, leak_detected=True, heldout_auroc=0.993243),
            True,
            True,
            "narrowed_to_math_only_abstain",
            "complete: shipped_detector_narrowed_to_math_only_abstain_on_code_e2e_green",
            None,
        ),
        (
            "blocked",
            {},
            False,
            False,
            "blocked",
            "complete: blocked_heldout_audit_unavailable",
            None,
        ),
    ],
)
def test_scenario_spoe_3706_parametrized_honest_outcomes(
    case_name: str,
    audit_artifact: dict[str, object],
    module_available: bool,
    code_surface_abstains: bool,
    expected_action: str,
    expected_verdict: str,
    expected_code_auroc: float | None,
) -> None:
    """SCENARIO-SPOE-3706: fixtures cover recalibrated, narrowed, and blocked."""

    artifact = exp.build_artifact_from_measurements(
        audit_artifact=audit_artifact,
        ship_artifact={} if expected_action == "blocked" else _ship_artifact(),
        module_available=module_available,
        code_surface_abstains=code_surface_abstains,
        operating_envelope_docstring_updated=expected_action != "blocked",
        e2e_test_passed=expected_action != "blocked",
        adversarial_verify_clean=expected_action != "blocked",
        started_s=1.0,
        now_s=2.75,
        tests_run=[f"SCENARIO-SPOE-3706 {case_name}"],
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["reconciliation_action"] == expected_action
    assert artifact["shipped_code_operating_point_auroc"] == expected_code_auroc
    assert type(artifact["overclaim_removed"]) is bool
    assert type(artifact["math_operating_point_unchanged"]) is bool
    assert type(artifact["e2e_test_passed"]) is bool
    assert artifact["duration_s"] == pytest.approx(1.75)
    assert artifact["tests_run"] == [f"SCENARIO-SPOE-3706 {case_name}"]
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)


def test_req_spoe_3706_shipped_surface_abstains_on_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SPOE-3706: code candidates return no code verdict after narrowing."""

    class ForbiddenVerifier:
        def score_rows(self, rows):  # pragma: no cover - failure path.
            raise AssertionError("code verifier must not run when code abstains")

    monkeypatch.setattr(spd.code_native_verifier_3695, "CodeNativeVerifier", ForbiddenVerifier)
    response = spd.score_candidates(
        [
            spd.CandidateScoreInput(
                candidate_id="code-abstain",
                domain="code",
                text="def f():\n    return None",
            ),
            spd.CandidateScoreInput(
                candidate_id="math-score",
                domain="math",
                text="1 + 1 = 3",
                confidence_error=0.8,
                ensemble_energy=0.9,
            ),
        ],
        examples=(
            _domain_examples("math", "fusion_wins")
            + _domain_examples("code", "fusion_wins")
        ),
    )

    rows = {row["candidate_id"]: row for row in response["scores"]}
    assert rows["code-abstain"]["code_verdict"] == "no_code_verdict"
    assert rows["code-abstain"]["abstained"] is True
    assert rows["code-abstain"]["calibrated_error_score"] is None
    assert rows["code-abstain"]["operating_point"] is None
    assert 0.0 <= rows["math-score"]["calibrated_error_score"] <= 1.0


def test_req_spoe_3706_write_artifact_and_validation(tmp_path: Path) -> None:
    """REQ-SPOE-3706: writing persists schema fields and bare-bool guards."""

    output = exp.write_artifact_from_measurements(
        tmp_path,
        output_path="results/exp3706.json",
        audit_artifact=_audit(survives=False, leak_detected=True, heldout_auroc=0.993243),
        ship_artifact=_ship_artifact(),
        module_available=True,
        code_surface_abstains=True,
        operating_envelope_docstring_updated=True,
        e2e_test_passed=True,
        adversarial_verify_clean=True,
        started_s=0.0,
        now_s=1.0,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["acceptance_gate"]["passed"] is True
    assert artifact["overclaim_removed"] is True

    broken = dict(artifact, overclaim_removed={"value": True})
    with pytest.raises(ValueError, match="overclaim_removed"):
        exp.validate_artifact(broken)

    missing = dict(artifact)
    missing.pop("reconciliation_action")
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="complete: unexpected")
    with pytest.raises(ValueError, match="terminal verdict"):
        exp.validate_artifact(bad_verdict)

    bad_code_auroc = dict(artifact, shipped_code_operating_point_auroc="0.7")
    with pytest.raises(ValueError, match="shipped_code_operating_point_auroc"):
        exp.validate_artifact(bad_code_auroc)

    bad_action = dict(artifact, reconciliation_action="surprise")
    with pytest.raises(ValueError, match="reconciliation_action"):
        exp.validate_artifact(bad_action)

    bad_narrowed = dict(
        artifact,
        reconciliation_action=exp.ACTION_NARROWED,
        shipped_code_operating_point_auroc=0.7,
    )
    with pytest.raises(ValueError, match="narrowed"):
        exp.validate_artifact(bad_narrowed)

    bad_duration = dict(artifact, duration_s=-1.0)
    with pytest.raises(ValueError, match="duration_s"):
        exp.validate_artifact(bad_duration)


def test_req_spoe_3706_build_artifact_paths_and_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SPOE-3706: real build paths and helper edge cases are covered."""

    blocked = exp.build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    assert blocked["honest_verdict"] == "complete: blocked_heldout_audit_unavailable"
    assert blocked["preconditions_checked"][0]["available"] is False

    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_3705_code_native_leak_audit_heldout.json").write_text(
        json.dumps(_audit(survives=False, leak_detected=True, heldout_auroc=0.993243)),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp.spd, "build_ship_artifact", lambda *args, **kwargs: _ship_artifact())
    narrowed = exp.build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    assert narrowed["honest_verdict"] == (
        "complete: shipped_detector_narrowed_to_math_only_abstain_on_code_e2e_green"
    )
    assert narrowed["exp3705_audit_summary"]["leak_detected"] is True
    assert narrowed["ship_artifact_summary"]["e2e_test_passed"] is True

    assert exp.detector_module_available() is True
    assert exp.code_surface_abstains() is True
    assert exp.operating_envelope_docstring_updated() is True
    assert exp.reconciliation_action({"unexpected": True}, module_available=True) == exp.ACTION_BLOCKED
    assert exp.math_operating_point_unchanged({}) is False
    assert exp.compact_adversarial_report({"flags": [{"severity": "warn"}, "skip"]}) == {
        "flag_count": 1,
        "flags": [{"severity": "warn"}],
    }
    assert exp.adversarial_report_is_clean({"flags": 3}) is False
    assert exp.adversarial_report_is_clean({"flags": [{"severity": "critical"}]}) is False
    assert exp._read_json_object(tmp_path / "missing.json") == {}
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert exp._read_json_object(invalid) == {}
    assert exp._coerce_optional_float("not-a-number") is None
    assert exp._round(float("inf")) == float("inf")


def test_req_spoe_3706_write_artifact_runs_adversarial_verify(tmp_path: Path) -> None:
    """REQ-SPOE-3706: write_artifact performs the adversarial cleanliness pass."""

    output = exp.write_artifact(
        tmp_path,
        output_path="results/blocked-exp3706.json",
        tests_run=["REQ-SPOE-3706 write_artifact"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["tests_run"] == ["REQ-SPOE-3706 write_artifact"]
    assert "adversarial_verify_report" in artifact
    assert "flags" in exp.run_adversarial_verify_report(output)

    saved = exp.importlib.util.spec_from_file_location
    try:
        exp.importlib.util.spec_from_file_location = lambda *args, **kwargs: None
        with pytest.raises(ImportError, match="adversarial_verify"):
            exp.run_adversarial_verify_report(output)
    finally:
        exp.importlib.util.spec_from_file_location = saved


def _domain_examples(
    domain: str,
    outcome: str,
    *,
    n: int = 80,
) -> list[spd.LabeledDetectorExample]:
    examples: list[spd.LabeledDetectorExample] = []
    for idx in range(n):
        label = 1 if idx < n // 2 else 0
        if outcome == "fusion_wins":
            ensemble = 0.92 - 0.003 * idx if label else 0.08 + 0.001 * (idx - n // 2)
            confidence = 0.50
        else:  # pragma: no cover - guarded by test data.
            raise ValueError(outcome)
        examples.append(
            spd.LabeledDetectorExample(
                domain=domain,
                label=label,
                ensemble_energy=ensemble,
                confidence_error=confidence,
                example_id=f"{domain}-{outcome}-{idx}",
            )
        )
    return examples
