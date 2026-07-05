"""Tests for Exp 5254 bounded KAN convex-envelope certificate.

Spec refs: REQ-KAN-5254, SCENARIO-KAN-5254.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5254_kan_convex_envelope_certificate_v480 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def test_req_kan_5254_spec_declares_convex_envelope_contract() -> None:
    """REQ-KAN-5254: OpenSpec anchors the bounded convex-envelope certificate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-KAN-5254") : spec.index("## Implementation Status")]

    for marker in (
        "REQ-KAN-5254",
        "SCENARIO-KAN-5254",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "false-property rejection",
        "no hardware speedup claim",
        "broad KAN verification",
    ):
        assert marker in section
    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert f"`{field}`" in section


def test_scenario_kan_5254_loads_exp5242_baseline_and_builds_two_envelopes() -> None:
    """SCENARIO-KAN-5254: the convex prototype compares against Exp5242 dimensions."""

    baseline = mod.load_exp5242_baseline()
    relaxation = mod.build_convex_relaxation()

    assert baseline.reproduced is True
    assert baseline.variables_verified == 2
    assert baseline.max_pwa_segments_verified == 10
    assert baseline.input_box == mod.INPUT_BOX
    assert baseline.false_property_rejected is True
    assert relaxation.variable_count == 2
    assert relaxation.envelope_count == 2
    assert relaxation.input_box == mod.INPUT_BOX
    assert relaxation.envelope_upper_bound == pytest.approx(0.69875)


def test_req_kan_5254_certificate_true_bound_and_false_rejection() -> None:
    """REQ-KAN-5254: true bounded property certifies and false property rejects."""

    result = mod.run_certificate_checks()

    assert result.true_property_certified is True
    assert result.false_property_rejected is True
    assert result.certificate_slack_min == pytest.approx(0.02125)
    assert result.true_property.certified_upper_bound == pytest.approx(0.69875)
    assert result.true_property.threshold == pytest.approx(mod.TRUE_PROPERTY_THRESHOLD)
    assert result.false_property.threshold == pytest.approx(mod.FALSE_PROPERTY_THRESHOLD)
    assert result.false_property.counterexample_inputs == pytest.approx([0.75, 0.75])
    assert result.false_property.actual_witness_value == pytest.approx(0.69875)
    assert result.solve_time_s >= 0.0


def test_req_kan_5254_artifact_fields_are_principle_wrapped(tmp_path: Path) -> None:
    """REQ-KAN-5254: artifact emits every required principle-wrapped field."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        duration_s=0.2,
        tests_run=[{"command": "unit fixture", "passed": True}],
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert "convex" in _value(artifact, "certificate_method")
    assert _value(artifact, "variables_verified") == 2
    assert _value(artifact, "max_segments_or_envelopes_verified") == 2
    assert _value(artifact, "input_box") == [[-0.5, 0.75], [-0.5, 0.75]]
    assert _value(artifact, "true_property_certified") is True
    assert _value(artifact, "false_property_rejected") is True
    assert _value(artifact, "certificate_slack_min") == pytest.approx(0.02125)
    assert _value(artifact, "solve_time_s") >= 0.0
    assert _value(artifact, "no_hardware_speedup_claim") is True
    assert artifact["baseline_comparison"]["max_pwa_segments_verified"] == 10
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_kan_5254_validation_rejects_overclaiming_edges() -> None:
    """REQ-KAN-5254: validation fails closed on substrate drift or hardware claims."""

    artifact = mod.build_artifact(
        duration_s=0.1,
        tests_run=[{"command": "unit fixture", "passed": True}],
    )

    broken = copy.deepcopy(artifact)
    broken["inference_substrate"] = mod.wrap_field(
        "inference_substrate",
        "live_llm_or_hardware",
    )
    with pytest.raises(AssertionError, match="inference_substrate"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["no_hardware_speedup_claim"] = mod.wrap_field(
        "no_hardware_speedup_claim",
        False,
    )
    with pytest.raises(AssertionError, match="hardware speedup"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    del broken["certificate_method"]
    with pytest.raises(AssertionError, match="missing required field"):
        mod.validate_artifact(broken)


def test_deliverable_file_validates_for_req_kan_5254() -> None:
    """SCENARIO-KAN-5254: committed deliverable JSON is the bounded certificate."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert _value(artifact, "true_property_certified") is True
    assert _value(artifact, "false_property_rejected") is True
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
