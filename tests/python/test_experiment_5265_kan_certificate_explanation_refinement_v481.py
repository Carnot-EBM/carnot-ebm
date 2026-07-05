"""Tests for Exp 5265 KAN certificate explanation/refinement.

Spec refs: REQ-KAN-5265, SCENARIO-KAN-5265.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from carnot import experiment_5254_kan_convex_envelope_certificate_v480 as v480
from carnot import experiment_5265_kan_certificate_explanation_refinement_v481 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
EXPLANATION_PATH = REPO / mod.EXPLANATION_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def test_req_kan_5265_spec_declares_explanation_refinement_contract() -> None:
    """REQ-KAN-5265: OpenSpec anchors explanation and refinement before code."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("## REQ-KAN-5265")
    end = spec.index("## Implementation Status", start)
    section = spec[start:end]

    for marker in (
        "REQ-KAN-5265",
        "SCENARIO-KAN-5265",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "near-threshold expected-false property",
        "envelope-gap bound",
        "broad KAN verification",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_kan_5265_identifies_driver_and_refines_gap() -> None:
    """SCENARIO-KAN-5265: largest component drives the bound and is refined."""

    report = mod.build_refinement_report()

    assert report.driver.variable_index == 0
    assert report.driver.bound_contribution == pytest.approx(0.45)
    assert report.driver.contribution_fraction > 0.6
    assert report.driver.max_envelope_gap == pytest.approx(0.078125)
    assert report.refinement_decisions[0].variable_index == 0
    assert report.refinement_decisions[0].decision == "split_interval"
    assert report.no_refinement.output_upper_bound == pytest.approx(0.69875)
    assert report.refined.output_upper_bound == pytest.approx(
        report.no_refinement.output_upper_bound
    )
    assert report.refined.envelope_gap_bound < report.no_refinement.envelope_gap_bound
    assert report.refined.refined_interval_count == 3


def test_req_kan_5265_certifies_true_property_and_rejects_near_false_property() -> None:
    """REQ-KAN-5265: true property remains certified and near-threshold false rejects."""

    report = mod.build_refinement_report()

    assert report.true_property.certified is True
    assert report.true_property.threshold == pytest.approx(v480.TRUE_PROPERTY_THRESHOLD)
    assert report.true_property.certificate_slack == pytest.approx(0.02125)
    assert report.near_false_property.certified is False
    assert report.near_false_property.rejected is True
    assert report.near_false_property.threshold == pytest.approx(mod.NEAR_FALSE_PROPERTY_THRESHOLD)
    assert report.near_false_property.certificate_slack < 0.0
    assert report.near_false_property.actual_witness_value > report.near_false_property.threshold
    assert report.near_false_property.counterexample_inputs == pytest.approx([0.75, 0.75])


def test_req_kan_5265_compares_refinement_and_no_refinement() -> None:
    """REQ-KAN-5265: refinement comparison reports tighter abstraction gap."""

    coarse = mod.build_abstraction_summary(refine_driver=False)
    refined = mod.build_abstraction_summary(refine_driver=True)
    driver = mod.build_refinement_report().driver

    assert coarse.refined_interval_count == 2
    assert refined.refined_interval_count == 3
    assert coarse.output_upper_bound == pytest.approx(refined.output_upper_bound)
    assert coarse.same_property_slack == pytest.approx(refined.same_property_slack)
    assert refined.envelope_gap_bound == pytest.approx(coarse.envelope_gap_bound / 2.0)
    assert refined.envelope_gap_reduction == pytest.approx(
        coarse.envelope_gap_bound - refined.envelope_gap_bound
    )
    no_op_decision = mod._refinement_decisions(driver, coarse, coarse)[0]
    assert no_op_decision.decision == "no_refinement"
    not_ready = replace(
        mod.build_refinement_report(),
        near_false_property=replace(
            mod.build_refinement_report().near_false_property, rejected=False
        ),
    )
    assert mod._honest_verdict(not_ready).startswith("blocked_")


def test_req_kan_5265_artifacts_have_required_schema_and_principles(tmp_path: Path) -> None:
    """REQ-KAN-5265: main and explanation artifacts expose required fields."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    explanation_path = tmp_path / mod.EXPLANATION_RELATIVE_PATH
    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        explanation_path=explanation_path,
        duration_s=0.2,
        tests_run=[{"command": "unit fixture", "outcome": "passed"}],
    )
    explanation = json.loads(explanation_path.read_text(encoding="utf-8"))

    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    mod.validate_explanation_artifact(explanation)
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert "refinement added certificate value" in _value(artifact, "honest_verdict")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert artifact["certificate_refinement_ready"] is True
    assert artifact["certificate_refinement_ready_principle"] == mod.CERTIFICATE_READY_PRINCIPLE
    assert _value(artifact, "true_property_certified") is True
    assert _value(artifact, "false_property_rejected") is True
    assert _value(artifact, "explanation_artifact_path") == str(mod.EXPLANATION_RELATIVE_PATH)
    assert _value(artifact, "spec_updated") is True
    slack = artifact["slack_before_after"]["value"]
    assert slack["same_property_slack_before"] == pytest.approx(0.02125)
    assert slack["same_property_slack_after"] == pytest.approx(0.02125)
    assert slack["same_property_slack_delta"] == pytest.approx(0.0)
    assert slack["envelope_gap_bound_after"] < slack["envelope_gap_bound_before"]
    assert artifact["tests_run"] == [{"command": "unit fixture", "outcome": "passed"}]
    assert explanation["bound_contributors"][0]["variable_index"] == 0
    assert explanation["refinement_decisions"][0]["decision"] == "split_interval"


def test_req_kan_5265_validation_rejects_drift() -> None:
    """REQ-KAN-5265: validation fails closed on substrate and readiness drift."""

    artifact = mod.build_artifact(tests_run=[{"command": "unit fixture", "outcome": "passed"}])

    broken = copy.deepcopy(artifact)
    broken["inference_substrate"] = mod.wrap_field("inference_substrate", "live_llm_inference")
    with pytest.raises(AssertionError, match="offline deterministic"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["certificate_refinement_ready"] = False
    with pytest.raises(AssertionError, match="refinement ready"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["slack_before_after"]["value"]["envelope_gap_bound_after"] = broken[
        "slack_before_after"
    ]["value"]["envelope_gap_bound_before"]
    with pytest.raises(AssertionError, match="gap reduction"):
        mod.validate_artifact(broken)


def test_deliverable_file_validates_for_scenario_kan_5265() -> None:
    """SCENARIO-KAN-5265: committed deliverable satisfies the V481 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    explanation = json.loads(EXPLANATION_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    mod.validate_explanation_artifact(explanation)
    assert artifact["certificate_refinement_ready"] is True
    assert _value(artifact, "true_property_certified") is True
    assert _value(artifact, "false_property_rejected") is True
    assert explanation["source_artifact"] == str(v480.RESULT_RELATIVE_PATH)
