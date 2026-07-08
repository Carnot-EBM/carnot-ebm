"""Tests for Exp5413 .492 PRD evidence table synthesis.

Spec refs: REQ-REPORT-5413, SCENARIO-REPORT-5413,
SCENARIO-REPORT-5413-MISSING-INPUT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5413_evidence_table_prd_gap_analysis_v492 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_report_5413_spec_declares_v492_gap_table_contract() -> None:
    """REQ-REPORT-5413: OpenSpec anchors the .492 evidence table."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5413") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5413",
        "SCENARIO-REPORT-5413",
        "SCENARIO-REPORT-5413-MISSING-INPUT",
        str(exp.RESULT_RELATIVE_PATH),
        str(exp.PRD_GAP_TABLE_RELATIVE_PATH),
        "FR-11 continuous self-learning",
        "FR-12 verifiable reasoning",
        "`scripts/research_conductor.py`",
        "`ops/status.md`",
        "`ops/changelog.md`",
        "`_bmad/traceability.md`",
    ):
        assert marker in section

    for field, principle in exp.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5413_available_artifacts_emit_guarded_gap_table() -> None:
    """SCENARIO-REPORT-5413: actual .492 artifacts produce guarded rows."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5413", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["missing_artifacts"] == []
    assert artifact["artifacts_read"] == [str(path) for path in exp.EXPECTED_ARTIFACTS]
    assert artifact["closed_gap_count"] == 5
    assert artifact["partial_gap_count"] == 4
    assert artifact["blocked_gap_count"] == 3
    assert artifact["missing_gap_count"] == 0
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["honest_verdict"].startswith("complete:")

    assert artifact["headline_ready_lanes"] == [
        "formal_encoding_corrigendum",
        "structured_safety_action_panel",
        "resource_accounted_csl",
        "uncertainty_gated_promotion",
    ]
    assert "arc_live_path" in artifact["non_headline_lanes"]
    assert "hardware_repeatability" in artifact["non_headline_lanes"]
    assert "token_internal_feature_backend" in artifact["non_headline_lanes"]

    rows = {row["row_id"]: row for row in artifact["gap_rows"]}
    assert list(rows) == list(exp.ROW_IDS)
    assert rows["formal_encoding_corrigendum"]["evidence_status"] == "closed"
    assert rows["formal_encoding_corrigendum"]["claim_strength"] == "headline_ready"
    assert rows["formal_encoding_corrigendum"]["principal_metric"] == {
        "fixture_count": 18,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
        "forbidden_leak_rate": 0.0,
        "gpu_offload_verified": True,
        "deterministic_policy_authority": True,
    }
    assert rows["structured_safety_action_panel"]["principal_metric"] == {
        "fixture_count": 42,
        "constrained_validity": 1.0,
        "unconstrained_validity": 0.285714,
        "unsafe_false_accept_rate": 0.0,
        "wrong_valid_delta": 26,
        "tool_action_reachability": 1.0,
        "gpu_offload_verified": True,
    }
    assert rows["active_constraint_warmstart"]["evidence_status"] == "closed"
    assert rows["active_constraint_warmstart"]["claim_strength"] == "bounded"
    assert rows["pbit_qubo_boundary"]["evidence_status"] == "partial"
    assert rows["pbit_qubo_boundary"]["principal_metric"]["hardware_speedup_claim"] is False
    assert rows["resource_accounted_csl"]["evidence_status"] == "closed"
    assert rows["resource_accounted_csl"]["principal_metric"]["no_weight_mutation"] is True
    assert rows["uncertainty_gated_promotion"]["principal_metric"] == {
        "accepted_promotion_count": 3,
        "rejected_retained_count": 6,
        "promotion_candidate_count": 9,
        "rollback_success_rate": 1.0,
        "routing_effect_row_count": 24,
        "no_weight_mutation": True,
    }
    assert rows["arc_live_path"]["evidence_status"] == "blocked"
    assert rows["arc_live_path"]["principal_metric"] == {
        "arc_new_level_banked": False,
        "attempt_count": 35,
        "frontier_expansion_count": 17,
        "reproduced_levels": 0,
        "offline_reproduced": False,
        "registry_total_before": 69,
        "registry_total_after": 69,
    }
    assert rows["hardware_repeatability"]["evidence_status"] == "partial"
    assert rows["hardware_repeatability"]["principal_metric"] == {
        "kv260_ssh_reachable": False,
        "polarfire_reachable": True,
        "polarfire_repeat_count": 3,
        "gatemate_reachable": False,
        "repeated_same_workload_ready": True,
        "hardware_speedup_claim": False,
    }
    assert rows["kan_active_constraint_certificate"]["principal_metric"] == {
        "kan_active_constraint_certificate_ready": True,
        "true_property_count": 3,
        "false_property_count": 8,
        "false_property_rejection_rate": 1.0,
        "counterexample_region_count": 8,
        "broad_kan_verification_claim": False,
    }
    assert rows["source_delta_watch_only"]["claim_strength"] == "watch_only"
    assert rows["hardware_speedup_claim"]["evidence_status"] == "blocked"
    assert rows["token_internal_feature_backend"]["claim_blocked"] == [
        "no .492 backend feature artifact authorizes a token/internal-feature headline claim"
    ]
    assert artifact["claim_boundary_checks"] == exp.CLAIM_BOUNDARY_CHECKS


def test_req_report_5413_rows_have_required_prd_and_architecture_fields() -> None:
    """REQ-REPORT-5413: each row preserves source, status, and claim boundary."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5413", "outcome": "passed"}],
    )

    for row in artifact["gap_rows"]:
        assert set(exp.REQUIRED_ROW_FIELDS) <= set(row)
        assert row["source_artifacts"]
        assert isinstance(row["prd_refs"], list)
        assert isinstance(row["architecture_refs"], list)
        assert row["evidence_status"] in exp.EVIDENCE_STATUSES
        assert row["claim_strength"] in exp.CLAIM_STRENGTHS
        assert isinstance(row["claim_allowed"], list)
        assert isinstance(row["claim_blocked"], list)
        assert isinstance(row["next_action"], str)


def test_scenario_report_5413_missing_inputs_stay_blocked(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5413-MISSING-INPUT: absent inputs are counted as missing."""

    artifact = exp.build_artifact(
        root=tmp_path,
        tests_run=[{"command": "unit exp5413 missing", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["status"] == "blocked_missing_inputs"
    assert artifact["artifacts_read"] == []
    assert artifact["missing_artifacts"] == [str(path) for path in exp.EXPECTED_ARTIFACTS]
    assert artifact["closed_gap_count"] == 0
    assert artifact["partial_gap_count"] == 0
    assert artifact["blocked_gap_count"] == 0
    assert artifact["missing_gap_count"] == len(exp.ROW_IDS)
    assert artifact["headline_ready_lanes"] == []
    assert artifact["non_headline_lanes"] == list(exp.ROW_IDS)
    assert artifact["honest_verdict"].startswith("blocked:")

    for row in artifact["gap_rows"]:
        assert row["evidence_status"] == "missing"
        assert row["claim_allowed"] == []
        assert row["claim_blocked"] == ["missing upstream artifact; no outcome inferred"]


def test_req_report_5413_run_writes_stable_repository_artifacts(tmp_path: Path) -> None:
    """REQ-REPORT-5413: run() writes deterministic artifact and table JSON."""

    tests_run = [
        {
            "command": (
                ".venv/bin/pytest "
                "tests/python/test_experiment_5413_evidence_table_prd_gap_analysis_v492.py -q "
                "--no-cov -n 0"
            ),
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage run "
                "--include=python/carnot/experiment_5413_evidence_table_prd_gap_analysis_v492.py "
                "-m pytest tests/python/test_experiment_5413_evidence_table_prd_gap_analysis_v492.py "
                "-q --no-cov -n 0"
            ),
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage report "
                "--include=python/carnot/experiment_5413_evidence_table_prd_gap_analysis_v492.py "
                "--fail-under=100"
            ),
            "outcome": "passed",
        },
        {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    ]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    table_path = tmp_path / exp.PRD_GAP_TABLE_RELATIVE_PATH

    artifact = exp.run(
        root=REPO,
        result_path=result_path,
        prd_gap_table_path=table_path,
        tests_run=tests_run,
    )

    table = json.loads(table_path.read_text(encoding="utf-8"))
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["prd_gap_table_path"] == str(exp.PRD_GAP_TABLE_RELATIVE_PATH)
    assert artifact["gap_rows"] == table["gap_rows"]
    assert table["milestone"] == exp.MILESTONE
    assert table["row_count"] == len(exp.ROW_IDS)
    assert artifact["tests_run"] == tests_run
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    assert artifact["spec_refs"] == list(exp.SPEC_REFS)
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    exp.validate_artifact(artifact)


def test_req_report_5413_committed_result_matches_replay() -> None:
    """REQ-REPORT-5413: checked-in result is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay


def test_req_report_5413_validation_rejects_claim_drift() -> None:
    """REQ-REPORT-5413: validation fails closed on schema or claim drift."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5413", "outcome": "passed"}],
    )

    missing_field = deepcopy(artifact)
    missing_field.pop("closed_gap_count")
    with pytest.raises(ValueError, match="closed_gap_count"):
        exp.validate_artifact(missing_field)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_counts = deepcopy(artifact)
    bad_counts["partial_gap_count"] = 99
    with pytest.raises(ValueError, match="gap counts"):
        exp.validate_artifact(bad_counts)

    bad_headline = deepcopy(artifact)
    bad_headline["headline_ready_lanes"].append("arc_live_path")
    with pytest.raises(ValueError, match="headline_ready_lanes"):
        exp.validate_artifact(bad_headline)

    bad_speedup = deepcopy(artifact)
    rows = {row["row_id"]: row for row in bad_speedup["gap_rows"]}
    rows["hardware_speedup_claim"]["principal_metric"]["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        exp.validate_artifact(bad_speedup)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)

    assert exp.unwrap({"value": "wrapped"}) == "wrapped"
    assert exp.unwrap("plain") == "plain"
    assert exp.json_ready((Path("a"), Path("b"))) == ["a", "b"]
