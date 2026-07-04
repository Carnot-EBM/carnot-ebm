"""Tests for Exp 5242 KAEM certificate abstraction scale boundary.

Spec refs: REQ-KAN-5242, SCENARIO-KAN-5242.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5242_kan_certificate_abstraction_scale_v479 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def test_req_kan_5242_spec_declares_bounded_stress_contract() -> None:
    """REQ-KAN-5242: OpenSpec anchors the bounded certificate stress contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-KAN-5242") : spec.index("## Implementation Status")]

    for marker in (
        "REQ-KAN-5242",
        "SCENARIO-KAN-5242",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "false property",
        "hardware readiness",
        "broad KAN verification",
    ):
        assert marker in section
    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert f"`{field}`" in section


def test_scenario_kan_5242_reproduces_exp5230_baseline_dimensions() -> None:
    """SCENARIO-KAN-5242: baseline reproduction records exact Exp 5230 dimensions."""

    baseline = mod.reproduce_exp5230_baseline()

    assert baseline.reproduced is True
    assert baseline.dimensions["input_dimension"] == 2
    assert baseline.dimensions["pwa_piece_count"] == 6
    assert baseline.dimensions["binary_variable_count"] == 6
    assert baseline.dimensions["constraint_count"] == 43
    assert baseline.dimensions["input_box"] == [[-0.25, 0.5], [-0.25, 0.5]]
    assert baseline.dimensions["baseline_slack"] == pytest.approx(0.0750000037)


def test_req_kan_5242_stress_cases_extend_segments_domain_and_reject_false_property() -> None:
    """REQ-KAN-5242: bounded stress verifies true cases and rejects a false property."""

    if mod.detect_solver() != "z3":
        pytest.skip("Z3 is not available in this environment")

    rows = mod.run_stress_cases(solver_name="z3")
    by_id = {row.case_id: row for row in rows}

    more_segments = by_id["more_pwa_segments"]
    wider_domain = by_id["wider_input_bounds"]
    false_property = by_id["deliberate_false_property"]

    assert more_segments.verified is True
    assert more_segments.pwa_piece_count == 10
    assert more_segments.certificate_slack is not None
    assert wider_domain.verified is True
    assert wider_domain.input_box == ((-0.5, 0.75), (-0.5, 0.75))
    assert wider_domain.certificate_slack is not None
    assert false_property.verified is False
    assert false_property.false_property_rejected is True
    assert false_property.certificate_slack is not None
    assert false_property.certificate_slack < 0.0
    assert all(row.numerical_instability == "none_detected" for row in rows)

    summary = mod.summarize_boundary(mod.reproduce_exp5230_baseline(), rows)
    assert summary["kan_certificate_extended"] is True
    assert summary["max_pwa_segments_verified"] == 10
    assert summary["false_property_rejected"] is True
    assert summary["certificate_slack_min"] == pytest.approx(
        min(
            row.certificate_slack
            for row in (more_segments, wider_domain)
            if row.certificate_slack is not None
        )
    )


def test_req_kan_5242_artifact_fields_are_principle_wrapped(tmp_path: Path) -> None:
    """REQ-KAN-5242: artifact emits required principle-wrapped schema fields."""

    if mod.detect_solver() != "z3":
        pytest.skip("Z3 is not available in this environment")

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        duration_s=0.5,
        tests_run=[{"command": "unit fixture", "passed": True}],
        solver_name="z3",
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert _value(artifact, "kan_certificate_baseline_reproduced") is True
    assert _value(artifact, "kan_certificate_extended") is True
    assert "more_pwa_segments" in _value(artifact, "stress_axes")
    assert _value(artifact, "max_pwa_segments_verified") == 10
    assert _value(artifact, "false_property_rejected") is True
    assert _value(artifact, "certificate_slack_min") > 0.0
    assert _value(artifact, "solve_time_s") >= 0.0
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "honest_verdict").startswith("success:")
    assert "hardware" in artifact["claim_limits"][0]
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_kan_5242_blocks_without_solver_and_validates_edges() -> None:
    """REQ-KAN-5242: missing solver blocks extension without losing baseline provenance."""

    blocked = mod.build_artifact(
        duration_s=0.1,
        tests_run=[{"command": "solver forced absent", "passed": True}],
        solver_name="",
    )

    mod.validate_artifact(blocked)
    assert _value(blocked, "kan_certificate_baseline_reproduced") is True
    assert _value(blocked, "kan_certificate_extended") is False
    assert _value(blocked, "max_pwa_segments_verified") == 0
    assert _value(blocked, "false_property_rejected") is False
    assert _value(blocked, "certificate_slack_min") is None
    assert _value(blocked, "honest_verdict").startswith("complete:")
    assert blocked["blocked_reason"] == "blocked_kan_pwa_milp_solver_unavailable"

    broken = copy.deepcopy(blocked)
    broken["inference_substrate"] = mod.wrap_field(
        "inference_substrate",
        "deterministic_enumeration",
    )
    with pytest.raises(AssertionError, match="inference_substrate"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(blocked)
    del broken["stress_axes"]
    with pytest.raises(AssertionError, match="missing required field"):
        mod.validate_artifact(broken)


def test_deliverable_file_validates_for_req_kan_5242() -> None:
    """SCENARIO-KAN-5242: committed deliverable reports the bounded certificate boundary."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert _value(artifact, "kan_certificate_baseline_reproduced") is True
    assert _value(artifact, "kan_certificate_extended") is True
    assert _value(artifact, "max_pwa_segments_verified") >= 10
    assert _value(artifact, "false_property_rejected") is True
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
