"""Tests for Exp5344 deterministic solver-guidance overwrite telemetry.

Spec refs: REQ-VERIFY-5344, SCENARIO-VERIFY-5344.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5344_solver_guidance_overwrite_telemetry_v487 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_verify_5344_spec_declares_overwrite_telemetry_contract() -> None:
    """REQ-VERIFY-5344: OpenSpec anchors the overwrite telemetry diagnostic."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5344") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5344",
        "SCENARIO-VERIFY-5344",
        str(exp.RESULT_RELATIVE_PATH),
        exp.INFERENCE_SUBSTRATE,
        "perfect hints",
        "partial hints",
        "stale hints",
        "misleading hints",
        "no hints",
        "solver_guidance_telemetry_ready",
        "misleading_hint_false_accepts=0",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_verify_5344_loads_qstr_and_sat_fixture_sources() -> None:
    """REQ-VERIFY-5344: diagnostic loads QSTR and bounded SAT/CDCL fixtures."""

    fixtures = exp.load_source_fixtures()
    instances = exp.build_diagnostic_instances(fixtures)

    assert fixtures["qstr_ready"] is True
    assert fixtures["sat_cdcl_available"] is True
    assert str(exp.qstr.RESULT_RELATIVE_PATH) in fixtures["source_artifacts"]
    assert str(exp.cdcl.RESULT_RELATIVE_PATH) in fixtures["source_artifacts"]
    assert {instance.domain for instance in instances} == {"qstr", "sat_cdcl"}
    assert [hint.name for hint in exp.build_hint_classes()] == list(exp.HINT_CLASS_NAMES)
    assert len(exp.build_hint_classes()) == 5


def test_scenario_verify_5344_bad_hints_are_overwritten_without_false_accepts() -> None:
    """SCENARIO-VERIFY-5344: stale and misleading hints fall back safely."""

    diagnostic = exp.run_diagnostic()
    stale_or_misleading = [
        row
        for row in diagnostic["per_hint_results"]
        if row["hint_class"] in {"stale_hints", "misleading_hints"}
    ]

    assert diagnostic["solver_authoritative"] is True
    assert diagnostic["hint_class_count"] == 5
    assert diagnostic["hint_validity_rate"] == pytest.approx(0.6)
    assert diagnostic["hint_overwrite_rate"] == pytest.approx(0.5)
    assert diagnostic["fallback_completeness_rate"] == pytest.approx(1.0)
    assert diagnostic["conflict_delta_vs_no_hint"] == 4
    assert diagnostic["search_delta_vs_no_hint"] == 46
    assert diagnostic["misleading_hint_false_accepts"] == 0
    assert diagnostic["blocked_instance_class_count"] == 4
    assert diagnostic["solver_guidance_telemetry_ready"] is True
    assert stale_or_misleading
    assert all(row["fallback_used"] is True for row in stale_or_misleading)
    assert all(row["fallback_preserved_baseline"] is True for row in stale_or_misleading)
    assert all(row["overwrite_count"] > 0 for row in stale_or_misleading)


def test_req_verify_5344_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5344: artifact exposes principle fields and bare metrics."""

    tests_run = [{"command": "unit exp5344", "outcome": "passed"}]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_NAME
    assert artifact["milestone"]["value"] == exp.MILESTONE
    assert artifact["status"]["value"] == "solver_guidance_telemetry_ready"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert artifact["solver_authoritative"] is True
    assert artifact["hint_class_count"] == 5
    assert artifact["hint_validity_rate"] == pytest.approx(0.6)
    assert artifact["hint_overwrite_rate"] == pytest.approx(0.5)
    assert artifact["fallback_completeness_rate"] == pytest.approx(1.0)
    assert artifact["conflict_delta_vs_no_hint"] == 4
    assert artifact["misleading_hint_false_accepts"] == 0
    assert artifact["blocked_instance_class_count"] == 4
    assert artifact["solver_guidance_telemetry_ready"] is True
    assert artifact["tests_run"]["value"] == tests_run


def test_req_verify_5344_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5344: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert result["solver_authoritative"] is True
    assert result["misleading_hint_false_accepts"] == 0
    assert result["solver_guidance_telemetry_ready"] is True
    exp.validate_artifact(result)


def test_req_verify_5344_validation_rejects_schema_drift() -> None:
    """REQ-VERIFY-5344: artifact validation rejects wrapped and bare field drift."""

    artifact = exp.build_artifact(tests_run=[{"command": "unit exp5344", "outcome": "passed"}])

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_authority = deepcopy(artifact)
    bad_authority["solver_authoritative"] = False
    with pytest.raises(ValueError, match="solver_authoritative"):
        exp.validate_artifact(bad_authority)

    bad_numeric = deepcopy(artifact)
    bad_numeric["hint_validity_rate"] = "0.6"
    with pytest.raises(ValueError, match="hint_validity_rate"):
        exp.validate_artifact(bad_numeric)

    bad_ready = deepcopy(artifact)
    bad_ready["solver_guidance_telemetry_ready"] = {"value": True}
    with pytest.raises(ValueError, match="solver_guidance_telemetry_ready"):
        exp.validate_artifact(bad_ready)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = [{"command": "lost principle", "outcome": "passed"}]
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)
