"""Tests for Exp5383 scaled overwrite-guidance validity report.

Spec refs: REQ-VERIFY-5383, SCENARIO-VERIFY-5383.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5383_overwrite_guidance_scale_validity_v490 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_verify_5383_spec_declares_scaled_validity_contract() -> None:
    """REQ-VERIFY-5383: OpenSpec anchors scaled overwrite-guidance validity."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5383") : spec.index("### REQ-VERIFY-5371")]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5383",
        "SCENARIO-VERIFY-5383",
        str(exp.RESULT_RELATIVE_PATH),
        "benign hints",
        "harmful hints",
        "incomplete hints",
        "contradictory hints",
        "overwrite_guidance_scale_ready",
        "unsafe_false_accepts=0",
        "honest_blocked",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_verify_5383_reuses_projection_and_overwrite_sources() -> None:
    """REQ-VERIFY-5383: fixture scale-up reuses existing solver guidance inputs."""

    sources = exp.load_source_fixtures()
    fixtures = exp.build_scaled_hint_fixtures(sources)

    assert sources["exp5358_solver_projection_ready"] is True
    assert sources["exp5370_overwrite_solver_guidance_ready"] is True
    assert str(exp.exp5358.RESULT_RELATIVE_PATH) in sources["source_artifacts"]
    assert str(exp.exp5370.RESULT_RELATIVE_PATH) in sources["source_artifacts"]
    assert sources["base_constraint_count"] == exp.BASE_CONSTRAINT_COUNT
    assert len(fixtures) == exp.EXPECTED_FIXTURE_COUNT
    assert tuple(mode.name for mode in exp.build_guidance_modes()) == exp.GUIDANCE_MODE_NAMES
    assert tuple(hint.name for hint in exp.build_hint_classes()) == exp.HINT_CLASS_NAMES
    assert {fixture.hint_class for fixture in fixtures} == set(exp.HINT_CLASS_NAMES)


def test_scenario_verify_5383_forced_invalidity_is_root_caused_not_accepted() -> None:
    """SCENARIO-VERIFY-5383: overwrite-capable guidance fixes invalid projections."""

    diagnostic = exp.run_scaled_validity_matrix()
    rows = diagnostic["matrix_results"]
    forced_rows = [row for row in rows if row["guidance_mode"] == "forced_hint"]
    overwrite_rows = [row for row in rows if row["guidance_mode"] == "overwrite_capable"]
    forced_invalid = [row for row in forced_rows if not row["projection_valid"]]
    root_cause_counts = {
        cause["root_cause"]: cause["count"]
        for cause in diagnostic["invalid_projection_root_causes"]
    }

    assert diagnostic["solver_authoritative"] is True
    assert diagnostic["fixture_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert diagnostic["guidance_mode_count"] == 3
    assert diagnostic["hint_class_count"] == 4
    assert diagnostic["forced_hint_harm_rate"] == pytest.approx(0.75)
    assert diagnostic["overwrite_rate"] == pytest.approx(1.0)
    assert diagnostic["post_projection_validity_rate"] == pytest.approx(1.0)
    assert diagnostic["fallback_completeness_rate"] == pytest.approx(1.0)
    assert diagnostic["unsafe_false_accepts"] == 0
    assert diagnostic["conflict_delta_vs_no_hint"] > 0
    assert diagnostic["convergence_delta_vs_no_hint"] > 0
    assert diagnostic["overwrite_guidance_scale_ready"] is True
    assert diagnostic["harmful_hint_classes"] == [
        "contradictory_hints",
        "harmful_hints",
        "incomplete_hints",
    ]

    assert len(forced_invalid) == 3 * exp.BASE_CONSTRAINT_COUNT
    assert all(row["accepted_as_valid"] is False for row in forced_invalid)
    assert all(row["unsafe_false_accept"] is False for row in rows)
    assert all(row["projection_valid"] is True for row in overwrite_rows)
    assert all(row["final_matches_baseline"] is True for row in overwrite_rows)
    assert all(row["fallback_complete"] is True for row in overwrite_rows)
    assert root_cause_counts == {
        "forced_contradictory_hint": exp.BASE_CONSTRAINT_COUNT,
        "forced_harmful_hint": exp.BASE_CONSTRAINT_COUNT,
        "forced_incomplete_hint": exp.BASE_CONSTRAINT_COUNT,
    }


def test_req_verify_5383_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5383: artifact exposes the required validity report fields."""

    tests_run = [{"command": "unit exp5383", "outcome": "passed"}]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["overwrite_guidance_scale_ready"] is True
    assert artifact["solver_authoritative"] is True
    assert artifact["fixture_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert artifact["forced_hint_harm_rate"] == pytest.approx(0.75)
    assert artifact["overwrite_rate"] == pytest.approx(1.0)
    assert artifact["post_projection_validity_rate"] == pytest.approx(1.0)
    assert artifact["fallback_completeness_rate"] == pytest.approx(1.0)
    assert artifact["conflict_delta_vs_no_hint"] > 0
    assert artifact["convergence_delta_vs_no_hint"] > 0
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == tests_run
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES


def test_req_verify_5383_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5383: checked-in result is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(tests_run=result["tests_run"])

    assert result == replay
    assert result["status"] == "complete"
    assert result["overwrite_guidance_scale_ready"] is True
    assert result["solver_authoritative"] is True
    assert result["unsafe_false_accepts"] == 0
    exp.validate_artifact(result)


def test_req_verify_5383_validation_rejects_unsafe_or_unprincipled_drift() -> None:
    """REQ-VERIFY-5383: validation fails closed on safety and schema drift."""

    artifact = exp.build_artifact(
        tests_run=[{"command": "unit exp5383", "outcome": "passed"}]
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

    bad_validity = deepcopy(artifact)
    bad_validity["post_projection_validity_rate"] = 0.99
    with pytest.raises(ValueError, match="post_projection_validity_rate"):
        exp.validate_artifact(bad_validity)

    bad_causes = deepcopy(artifact)
    bad_causes["invalid_projection_root_causes"] = []
    with pytest.raises(ValueError, match="invalid_projection_root_causes"):
        exp.validate_artifact(bad_causes)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    bad_tests["overwrite_guidance_scale_ready"] = True
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)
