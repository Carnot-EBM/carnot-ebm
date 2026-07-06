"""Tests for Exp 5291 low-order factor certificate curriculum.

Spec refs: REQ-KAN-5291, SCENARIO-KAN-5291.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5291_low_order_factor_certificate_curriculum_v483 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def test_req_kan_5291_spec_declares_curriculum_contract() -> None:
    """REQ-KAN-5291: OpenSpec anchors the low-order curriculum artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-KAN-5291") : spec.index("## Implementation Status")]

    for marker in (
        "REQ-KAN-5291",
        "SCENARIO-KAN-5291",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "low-order",
        "medium-order",
        "higher-order",
        "false-property",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_kan_5291_defines_ordered_factor_stages() -> None:
    """SCENARIO-KAN-5291: stages report factor order and property targets."""

    stages = mod.define_factor_stages()

    assert [stage.stage_id for stage in stages] == [
        "low_order_unary",
        "medium_order_pair",
        "higher_order_triple",
    ]
    assert [stage.factor_order for stage in stages] == [1, 2, 3]
    assert [stage.component_indices for stage in stages] == [(0,), (0, 1), (0, 1, 2)]
    assert all(stage.true_property_target > stage.false_property_target for stage in stages)
    assert any("experiment_5277" in ref for ref in stages[0].source_fixture_refs)
    assert any("experiment_5278" in ref for ref in stages[1].source_fixture_refs)


def test_req_kan_5291_false_property_rejection_and_slack_accounting() -> None:
    """REQ-KAN-5291: every stage certifies true target and rejects false target."""

    outcomes = [mod.evaluate_stage(stage) for stage in mod.define_factor_stages()]

    assert [outcome.stage_id for outcome in outcomes] == [
        "low_order_unary",
        "medium_order_pair",
        "higher_order_triple",
    ]
    assert all(outcome.certificate_success for outcome in outcomes)
    assert all(outcome.false_property_rejected for outcome in outcomes)
    assert all(outcome.failure_class == "none" for outcome in outcomes)
    assert [outcome.factor_order for outcome in outcomes] == [1, 2, 3]
    assert [outcome.piece_count for outcome in outcomes] == [2, 4, 6]
    assert [outcome.true_property_slack for outcome in outcomes] == pytest.approx(
        [0.011, 0.0112, 0.0166]
    )
    assert outcomes[0].false_property_slack < 0.0
    assert outcomes[1].false_property_slack < 0.0
    assert outcomes[2].false_property_slack < 0.0


def test_req_kan_5291_factor_order_reporting_compares_shuffled_order() -> None:
    """REQ-KAN-5291: low-order-first is compared against a shuffled order."""

    curriculum = mod.run_curriculum()
    factor_metrics = curriculum["factor_order_metrics"]
    success_metrics = curriculum["certificate_success_by_order"]

    assert factor_metrics["low_order_first_sequence"] == [
        "low_order_unary",
        "medium_order_pair",
        "higher_order_triple",
    ]
    assert factor_metrics["shuffled_sequence"] == [
        "higher_order_triple",
        "low_order_unary",
        "medium_order_pair",
    ]
    assert factor_metrics["low_before_high_in_curriculum"] is True
    assert factor_metrics["lowest_order_success_step"] == 1
    assert factor_metrics["highest_order_success_step"] == 3
    assert success_metrics["all_curriculum_stages_certified"] is True
    assert success_metrics["all_shuffled_stages_certified"] is True
    assert success_metrics["success_advantage_over_shuffled"] == 0.0
    assert success_metrics["helped_certificate_success"] is False


def test_req_kan_5291_artifact_fields_and_schema_validate(tmp_path: Path) -> None:
    """REQ-KAN-5291: artifact exposes required wrapped fields and bare ready bool."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_outputs(
        artifact_path=result_path,
        duration_s=0.5,
        tests_run=[{"command": "unit exp5291", "outcome": "passed"}],
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert "did not improve certificate success" in _value(artifact, "honest_verdict")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert artifact["low_order_curriculum_ready"] is True
    assert isinstance(artifact["low_order_curriculum_ready_principle"], str)
    assert _value(artifact, "false_property_rejected") is True
    assert _value(artifact, "factor_order_metrics")["factor_orders_seen"] == [1, 2, 3]
    assert _value(artifact, "slack_metrics")["minimum_true_property_slack"] == pytest.approx(0.011)
    assert _value(artifact, "piece_count_metrics")["piece_count_by_stage"] == {
        "low_order_unary": 2,
        "medium_order_pair": 4,
        "higher_order_triple": 6,
    }
    assert artifact["tests_run"] == [{"command": "unit exp5291", "outcome": "passed"}]
    assert "REQ-KAN-5291" in artifact["spec_refs"]
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_kan_5291_validation_rejects_false_control_and_slack_drift() -> None:
    """REQ-KAN-5291: validation fails closed on false-control or slack drift."""

    artifact = mod.build_artifact(
        duration_s=0.1,
        tests_run=[{"command": "unit exp5291", "outcome": "passed"}],
    )

    broken = copy.deepcopy(artifact)
    broken["false_property_rejected"] = mod.wrap_field("false_property_rejected", False)
    with pytest.raises(AssertionError, match="false property"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["slack_metrics"] = mod.wrap_field(
        "slack_metrics",
        {"minimum_true_property_slack": -0.1, "true_property_slack_by_stage": {}},
    )
    with pytest.raises(AssertionError, match="slack"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["low_order_curriculum_ready"] = {"value": True}
    with pytest.raises(AssertionError, match="bare bool"):
        mod.validate_artifact(broken)


def test_deliverable_file_validates_for_scenario_kan_5291() -> None:
    """SCENARIO-KAN-5291: committed deliverable satisfies the V483 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["low_order_curriculum_ready"] is True
    assert _value(artifact, "false_property_rejected") is True
    assert _value(artifact, "certificate_success_by_order")["helped_certificate_success"] is False
    assert _value(artifact, "piece_count_metrics")["piece_count_increases_with_order"] is True
