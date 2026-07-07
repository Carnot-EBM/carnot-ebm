"""Tests for Exp5370 overwrite-capable solver guidance matrix.

Spec refs: REQ-VERIFY-5370, SCENARIO-VERIFY-5370.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5370_overwrite_solver_guidance_matrix_v489 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_verify_5370_spec_declares_solver_guidance_matrix_contract() -> None:
    """REQ-VERIFY-5370: OpenSpec anchors the full guidance-mode matrix."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5370") : spec.index("### REQ-VERIFY-5345")]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5370",
        "SCENARIO-VERIFY-5370",
        str(exp.RESULT_RELATIVE_PATH),
        "no hints",
        "forced hints",
        "overwrite-capable hints",
        "aligned hints",
        "partially wrong hints",
        "misleading high-confidence hints",
        "overwrite_solver_guidance_ready",
        "unsafe_false_accepts=0",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_verify_5370_loads_existing_solver_fixture_sources() -> None:
    """REQ-VERIFY-5370: source fixtures reuse prior SAT/QSTR diagnostics."""

    sources = exp.load_source_fixtures()
    fixtures = exp.build_constraint_fixtures(sources)

    assert sources["solver_projection_ready"] is True
    assert sources["pbit_schedule_diagnostic_ready"] is True
    assert sources["qstr_ready"] is True
    assert sources["sat_cdcl_available"] is True
    assert str(exp.exp5358.RESULT_RELATIVE_PATH) in sources["source_artifacts"]
    assert str(exp.exp5359.RESULT_RELATIVE_PATH) in sources["source_artifacts"]
    assert {fixture.domain for fixture in fixtures} == {"qstr", "sat_cdcl"}
    assert len(fixtures) == exp.EXPECTED_FIXTURE_COUNT
    assert tuple(mode.name for mode in exp.build_guidance_modes()) == exp.GUIDANCE_MODE_NAMES
    assert tuple(proposal.name for proposal in exp.build_proposal_classes()) == exp.PROPOSAL_CLASS_NAMES


def test_scenario_verify_5370_forced_hints_compared_to_overwrite_routing() -> None:
    """SCENARIO-VERIFY-5370: bad hints are measured and overwritten safely."""

    diagnostic = exp.run_guidance_matrix()
    rows = diagnostic["matrix_results"]
    forced_rows = [row for row in rows if row["guidance_mode"] == "forced_hint"]
    overwrite_rows = [row for row in rows if row["guidance_mode"] == "overwrite_capable"]
    misleading_overwrite = [
        row
        for row in overwrite_rows
        if row["proposal_class"] == "misleading_high_confidence_hints"
    ]

    assert diagnostic["solver_authoritative"] is True
    assert diagnostic["fixture_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert diagnostic["proposal_class_count"] == 4
    assert diagnostic["guidance_mode_count"] == 3
    assert diagnostic["unsafe_false_accepts"] == 0
    assert diagnostic["fallback_completeness_rate"] == pytest.approx(1.0)
    assert diagnostic["forced_hint_harm_rate"] > diagnostic["overwrite_hint_harm_rate"]
    assert diagnostic["overwrite_rate"] > 0.0
    assert diagnostic["conflict_delta_vs_solver_only"] > 0
    assert 0.0 < diagnostic["post_projection_validity_rate"] < 1.0
    assert diagnostic["harmful_hint_classes"] == [
        "misleading_high_confidence_hints",
        "partially_wrong_hints",
    ]
    assert diagnostic["overwrite_solver_guidance_ready"] is True

    assert forced_rows
    assert overwrite_rows
    assert any(row["projection_valid"] is False for row in forced_rows)
    assert all(row["unsafe_false_accept"] is False for row in rows)
    assert all(row["fallback_complete"] is True for row in misleading_overwrite)
    assert all(row["overwritten_decisions"] > 0 for row in misleading_overwrite)


def test_req_verify_5370_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5370: artifact exposes required fields and principles."""

    tests_run = [{"command": "unit exp5370", "outcome": "passed"}]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["overwrite_solver_guidance_ready"] is True
    assert artifact["solver_authoritative"] is True
    assert artifact["fixture_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert artifact["proposal_class_count"] == 4
    assert artifact["overwrite_rate"] > 0.0
    assert artifact["conflict_delta_vs_solver_only"] > 0
    assert artifact["forced_hint_harm_rate"] > artifact["overwrite_hint_harm_rate"]
    assert 0.0 < artifact["post_projection_validity_rate"] < 1.0
    assert artifact["fallback_completeness_rate"] == pytest.approx(1.0)
    assert artifact["harmful_hint_classes"] == [
        "misleading_high_confidence_hints",
        "partially_wrong_hints",
    ]
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["tests_run"] == tests_run
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES


def test_req_verify_5370_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5370: checked-in result is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(tests_run=result["tests_run"])

    assert result == replay
    assert result["status"] == "complete"
    assert result["overwrite_solver_guidance_ready"] is True
    assert result["solver_authoritative"] is True
    assert result["unsafe_false_accepts"] == 0
    exp.validate_artifact(result)


def test_req_verify_5370_validation_rejects_unsafe_or_unprincipled_drift() -> None:
    """REQ-VERIFY-5370: validation fails closed on schema and safety drift."""

    artifact = exp.build_artifact(
        tests_run=[{"command": "unit exp5370", "outcome": "passed"}]
    )

    bad_status = deepcopy(artifact)
    bad_status["status"] = "done"
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_status)

    bad_authority = deepcopy(artifact)
    bad_authority["solver_authoritative"] = False
    with pytest.raises(ValueError, match="solver_authoritative"):
        exp.validate_artifact(bad_authority)

    bad_accept = deepcopy(artifact)
    bad_accept["unsafe_false_accepts"] = 1
    with pytest.raises(ValueError, match="unsafe_false_accepts"):
        exp.validate_artifact(bad_accept)

    bad_modes = deepcopy(artifact)
    bad_modes["guidance_modes_measured"] = ["no_hint", "forced_hint"]
    with pytest.raises(ValueError, match="guidance_modes_measured"):
        exp.validate_artifact(bad_modes)

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_principles)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    bad_tests["overwrite_solver_guidance_ready"] = True
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)
