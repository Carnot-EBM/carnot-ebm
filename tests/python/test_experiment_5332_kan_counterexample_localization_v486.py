"""Tests for Exp 5332 bounded KAN counterexample localization.

Spec refs: REQ-KAN-5332, SCENARIO-KAN-5332.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5332_kan_counterexample_localization_v486 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def test_req_kan_5332_spec_declares_localization_contract() -> None:
    """REQ-KAN-5332: OpenSpec anchors the bounded localization diagnostic."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("## REQ-KAN-5332") : spec.index("## Implementation Status")
    ]

    for marker in (
        "REQ-KAN-5332",
        "SCENARIO-KAN-5332",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "false-property perturbations",
        "localization accuracy",
        "`no_broad_certificate_claim` MUST be true",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_kan_5332_defines_one_bounded_false_perturbation_per_unit() -> None:
    """REQ-KAN-5332: perturbations have deterministic expected units and regions."""

    perturbations = mod.define_false_property_perturbations()

    assert len(perturbations) == mod.FIXTURE_COUNT
    assert [row.expected_unit_index for row in perturbations] == [0, 1, 2]
    assert [row.perturbation_id for row in perturbations] == [
        "unit_0_constant_shift_false_threshold",
        "unit_1_constant_shift_false_threshold",
        "unit_2_constant_shift_false_threshold",
    ]
    assert all(row.constant_shift == pytest.approx(mod.PERTURBATION_DELTA) for row in perturbations)
    assert all(row.false_threshold == pytest.approx(mod.FALSE_PROPERTY_THRESHOLD) for row in perturbations)
    assert all(row.true_threshold == pytest.approx(mod.TRUE_PROPERTY_THRESHOLD) for row in perturbations)
    assert perturbations[0].expected_region == pytest.approx((0.35, 0.6))
    assert perturbations[1].expected_region == pytest.approx((0.2666666667, 0.6))
    assert perturbations[2].expected_region == pytest.approx((0.2666666667, 0.6))


def test_scenario_kan_5332_rejects_false_properties_and_localizes_regions() -> None:
    """SCENARIO-KAN-5332: diagnostic rejects false perturbations and localizes them."""

    diagnostic = mod.run_localization_diagnostic()

    assert diagnostic["fixture_count"] == mod.FIXTURE_COUNT
    assert diagnostic["false_property_rejection_rate"] == pytest.approx(1.0)
    assert diagnostic["true_property_preservation_rate"] == pytest.approx(1.0)
    assert diagnostic["counterexample_localization_accuracy"] == pytest.approx(1.0)
    assert diagnostic["envelope_gap_delta"] > 0.0
    assert diagnostic["certificate_success_delta"] == pytest.approx(0.0)
    assert diagnostic["counterexample_localization_ready"] is True
    assert diagnostic["no_broad_certificate_claim"] is True
    assert diagnostic["piece_budget"] == mod.PIECE_BUDGET
    assert diagnostic["inference_substrate"] == mod.INFERENCE_SUBSTRATE

    for row in diagnostic["perturbation_results"]:
        assert row["false_property_rejected"] is True
        assert row["true_property_preserved"] is True
        assert row["localized"] is True
        assert row["predicted_unit_index"] == row["expected_unit_index"]
        assert row["predicted_region"] == pytest.approx(row["expected_region"])
        assert row["false_property_slack"] < 0.0
        assert row["true_property_slack"] > 0.0
        assert row["sensitivity_margin"] > 0.0
        assert row["counterexample_inputs"] == pytest.approx([0.6, 0.6, 0.6])
        assert row["envelope_gap"] == pytest.approx(diagnostic["envelope_gap"])


def test_req_kan_5332_artifact_schema_and_validation(tmp_path: Path) -> None:
    """REQ-KAN-5332: artifact exposes required wrapped fields and bare scalars."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit exp5332", "outcome": "passed"}]
    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        duration_s=0.2,
        tests_run=tests_run,
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert _value(artifact, "experiment_id") == mod.EXPERIMENT_ID
    assert _value(artifact, "milestone") == mod.MILESTONE
    assert _value(artifact, "status") == "complete"
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "tests_run") == tests_run
    assert artifact["fixture_count"] == mod.FIXTURE_COUNT
    assert artifact["false_property_rejection_rate"] == pytest.approx(1.0)
    assert artifact["true_property_preservation_rate"] == pytest.approx(1.0)
    assert artifact["counterexample_localization_accuracy"] == pytest.approx(1.0)
    assert artifact["envelope_gap_delta"] > 0.0
    assert artifact["certificate_success_delta"] == pytest.approx(0.0)
    assert artifact["counterexample_localization_ready"] is True
    assert artifact["no_broad_certificate_claim"] is True
    assert "REQ-KAN-5332" in artifact["spec_refs"]
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_kan_5332_validation_fails_closed_on_schema_drift() -> None:
    """REQ-KAN-5332: validation rejects drift in scope, substrate, and localization."""

    artifact = mod.build_artifact(
        duration_s=0.1,
        tests_run=[{"command": "unit exp5332", "outcome": "passed"}],
    )

    broken = copy.deepcopy(artifact)
    broken["no_broad_certificate_claim"] = False
    with pytest.raises(AssertionError, match="broad certificate"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["counterexample_localization_ready"] = {"value": True}
    with pytest.raises(AssertionError, match="bare bool"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["fixture_count"] = {"value": mod.FIXTURE_COUNT}
    with pytest.raises(AssertionError, match="bare integer"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["inference_substrate"] = mod.wrap_field("inference_substrate", "wrong")
    with pytest.raises(AssertionError, match="inference"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["perturbation_results"][0]["localized"] = False
    with pytest.raises(AssertionError, match="localization"):
        mod.validate_artifact(broken)


def test_req_kan_5332_honest_verdict_prefixes_cover_blocked_cases() -> None:
    """REQ-KAN-5332: terminal verdict prefixes cover ready and blocked diagnostics."""

    diagnostic = mod.run_localization_diagnostic()

    assert mod.honest_verdict(diagnostic).startswith("complete:")

    blocked = copy.deepcopy(diagnostic)
    blocked["no_broad_certificate_claim"] = False
    assert mod.honest_verdict(blocked).startswith("blocked_")

    blocked = copy.deepcopy(diagnostic)
    blocked["false_property_rejection_rate"] = 0.0
    assert mod.honest_verdict(blocked).startswith("blocked_")

    blocked = copy.deepcopy(diagnostic)
    blocked["counterexample_localization_ready"] = False
    assert mod.honest_verdict(blocked).startswith("blocked_")


def test_deliverable_file_validates_for_scenario_kan_5332() -> None:
    """SCENARIO-KAN-5332: committed deliverable JSON satisfies the V486 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["counterexample_localization_ready"] is True
    assert artifact["no_broad_certificate_claim"] is True
    assert artifact["fixture_count"] == mod.FIXTURE_COUNT
