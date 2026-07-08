"""Tests for Exp5400 PRD-aligned evidence table synthesis.

Spec refs: REQ-REPORT-5400, SCENARIO-REPORT-5400,
SCENARIO-REPORT-5400-MISSING-INPUT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5400_evidence_table_prd_gap_analysis_v491 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_report_5400_spec_declares_evidence_table_contract() -> None:
    """REQ-REPORT-5400: OpenSpec anchors the PRD evidence table."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5400") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5400",
        "SCENARIO-REPORT-5400",
        "SCENARIO-REPORT-5400-MISSING-INPUT",
        str(exp.RESULT_RELATIVE_PATH),
        "structured local SOTA",
        "formal-encoding safety",
        "hardware speedup without repeatability",
        "`scripts/research_conductor.py`",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5400_available_artifacts_emit_guarded_table() -> None:
    """SCENARIO-REPORT-5400: available .491 artifacts produce guarded rows."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5400", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["missing_artifacts"] == []
    assert artifact["artifacts_read"] == [str(path) for path in exp.EXPECTED_ARTIFACTS]
    assert [row["row_id"] for row in artifact["evidence_rows"]] == list(exp.ROW_IDS)
    assert artifact["claim_boundary_checks"] == {
        "external_text_scoring_relied_on": False,
        "cpu_only_legacy_headline_evidence_relied_on": False,
        "duplicate_arc_solve_relied_on": False,
        "hardware_speedup_without_repeatability_relied_on": False,
    }
    assert artifact["honest_verdict"].startswith("complete:")

    rows = {row["row_id"]: row for row in artifact["evidence_rows"]}
    assert rows["structured_local_sota"]["evidence_strength"] == "closed"
    assert rows["structured_local_sota"]["principal_metric"] == {
        "fixture_count": 24,
        "constrained_semantic_validity_rate": 1.0,
        "unconstrained_semantic_validity_rate": 0.0,
        "wrong_valid_reduction": 20,
        "unsafe_false_accept_count": 0,
    }
    assert rows["formal_encoding_safety"]["evidence_strength"] == "partial"
    assert "adversarial TAUTOLOGY flag remains pending" in rows["formal_encoding_safety"]["claim_blocked"]
    assert rows["solver_corrigendum"]["evidence_strength"] == "closed"
    assert rows["pbit_ablation"]["evidence_strength"] == "partial"
    assert "hardware p-bit or speedup claim" in rows["pbit_ablation"]["claim_blocked"]
    assert rows["continuous_self_learning_router"]["evidence_strength"] == "closed"
    assert rows["memory_guard"]["evidence_strength"] == "closed"
    assert rows["arc"]["evidence_strength"] == "blocked"
    assert "new ARC level banked" in rows["arc"]["claim_blocked"]
    assert rows["hardware"]["evidence_strength"] == "partial"
    assert "hardware speedup" in rows["hardware"]["claim_blocked"]
    assert rows["kan_certificate"]["evidence_strength"] == "closed"
    assert rows["token_internal_features"]["evidence_strength"] == "blocked"
    assert rows["prd_alignment"]["evidence_strength"] == "partial"

    assert any(gap["gap_id"] == "FR-12-local-verifiable-reasoning" for gap in artifact["closed_gaps"])
    assert any(gap["gap_id"] == "formal-encoding-safety-methodology" for gap in artifact["partial_gaps"])
    assert any(gap["gap_id"] == "ARC-live-level-bank" for gap in artifact["blocked_gaps"])
    assert "external generated-text scoring as final authority" in artifact["disallowed_claims"]
    assert "hardware speedup without repeated same-workload timing" in artifact["disallowed_claims"]


def test_req_report_5400_rows_have_required_guardrails() -> None:
    """REQ-REPORT-5400: each row records source, claims, metrics, and guardrails."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5400", "outcome": "passed"}],
    )

    for row in artifact["evidence_rows"]:
        assert set(exp.REQUIRED_ROW_FIELDS) <= set(row)
        assert row["source_artifact"]
        assert isinstance(row["claim_allowed"], list)
        assert isinstance(row["claim_blocked"], list)
        assert isinstance(row["next_action"], str)
        assert row["evidence_strength"] in exp.EVIDENCE_STRENGTHS
        assert row["guardrail_checks"] == {
            "external_text_scoring_relied_on": False,
            "cpu_only_legacy_headline_evidence_relied_on": False,
            "duplicate_arc_solve_relied_on": False,
            "hardware_speedup_without_repeatability_relied_on": False,
        }


def test_scenario_report_5400_missing_inputs_stay_partial(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5400-MISSING-INPUT: absent inputs are not inferred."""

    artifact = exp.build_artifact(
        root=tmp_path,
        tests_run=[{"command": "unit exp5400 missing", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["status"] == "partial"
    assert artifact["artifacts_read"] == []
    assert artifact["missing_artifacts"] == [str(path) for path in exp.EXPECTED_ARTIFACTS]
    assert artifact["honest_verdict"].startswith("partial:")
    assert artifact["closed_gaps"] == []
    assert artifact["partial_gaps"] == []
    assert len(artifact["blocked_gaps"]) == len(exp.ROW_IDS)

    for row in artifact["evidence_rows"]:
        assert row["evidence_strength"] == "missing_inputs"
        assert row["claim_allowed"] == []
        assert row["claim_blocked"] == ["missing upstream artifact; no outcome inferred"]


def test_req_report_5400_run_writes_stable_repository_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-5400: run() writes the deterministic evidence artifact."""

    tests_run = [
        {
            "command": (
                ".venv/bin/pytest "
                "tests/python/test_experiment_5400_evidence_table_prd_gap_analysis_v491.py -q"
            ),
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage run "
                "--include=python/carnot/experiment_5400_evidence_table_prd_gap_analysis_v491.py "
                "-m pytest tests/python/test_experiment_5400_evidence_table_prd_gap_analysis_v491.py "
                "-q --no-cov -n 0"
            ),
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage report "
                "--include=python/carnot/experiment_5400_evidence_table_prd_gap_analysis_v491.py "
                "--fail-under=100"
            ),
            "outcome": "passed",
        },
        {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    ]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH

    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["tests_run"] == tests_run
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["spec_refs"] == list(exp.SPEC_REFS)
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    exp.validate_artifact(artifact)


def test_req_report_5400_committed_result_matches_replay() -> None:
    """REQ-REPORT-5400: checked-in result is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay


def test_req_report_5400_validation_rejects_claim_drift() -> None:
    """REQ-REPORT-5400: validation fails closed on schema or claim drift."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5400", "outcome": "passed"}],
    )

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(missing)

    bad_principle = deepcopy(artifact)
    bad_principle["field_principles"]["status"] = "changed"
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_principle)

    bad_milestone = deepcopy(artifact)
    bad_milestone["milestone"] = "2026.07.490"
    with pytest.raises(ValueError, match="milestone"):
        exp.validate_artifact(bad_milestone)

    bad_row_count = deepcopy(artifact)
    bad_row_count["evidence_rows"] = bad_row_count["evidence_rows"][:-1]
    with pytest.raises(ValueError, match="row_ids"):
        exp.validate_artifact(bad_row_count)

    bad_strength = deepcopy(artifact)
    bad_strength["evidence_rows"][0]["evidence_strength"] = "headline"
    with pytest.raises(ValueError, match="evidence_strength"):
        exp.validate_artifact(bad_strength)

    bad_external = deepcopy(artifact)
    bad_external["evidence_rows"][0]["guardrail_checks"]["external_text_scoring_relied_on"] = True
    with pytest.raises(ValueError, match="guardrail"):
        exp.validate_artifact(bad_external)

    bad_speedup = deepcopy(artifact)
    bad_speedup["disallowed_claims"].remove("hardware speedup without repeated same-workload timing")
    with pytest.raises(ValueError, match="disallowed_claims"):
        exp.validate_artifact(bad_speedup)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "complete"
    bad_status["missing_artifacts"] = ["results/missing.json"]
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_status)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)

    assert exp.unwrap("plain") == "plain"
    assert exp.json_ready((Path("a"), Path("b"))) == ["a", "b"]
