"""Tests for Exp6435 V553 adversarial capstone.

Spec refs: REQ-CAPSTONE-6435,
SCENARIO-CAPSTONE-6435-HASHES,
SCENARIO-CAPSTONE-6435-PER-TASK,
SCENARIO-CAPSTONE-6435-ROW-RECHECKS,
SCENARIO-CAPSTONE-6435-CLAIM-ELIGIBILITY,
SCENARIO-CAPSTONE-6435-RETIREMENT-AND-ATTACKS,
SCENARIO-CAPSTONE-6435-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6435_v553_adversarial_capstone as exp6435


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / exp6435.SPEC_RELATIVE_PATH


def _artifact(tmp_path: Path) -> dict[str, Any]:
    commands = (
        exp6435.RUN_COMMAND,
        exp6435.FOCUSED_TEST_COMMAND,
        exp6435.COVERAGE_RUN_COMMAND,
        exp6435.COVERAGE_REPORT_COMMAND,
    )
    return exp6435.build_artifact(
        repo_root=REPO,
        date="20260814",
        result_path=tmp_path / exp6435.RESULT_RELATIVE_PATH.name,
        duration_s=2.0,
        tests_run=[{"command": command, "exit_code": 0} for command in commands],
        write=True,
    )


def _with_checksum(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = exp6435.payload_checksum(payload)
    return payload


def test_req_capstone_6435_spec_declares_v553_contract() -> None:
    """REQ-CAPSTONE-6435: OpenSpec owns the V553 capstone contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CAPSTONE-6435") :]
    for marker in (
        "SCENARIO-CAPSTONE-6435-HASHES",
        "SCENARIO-CAPSTONE-6435-PER-TASK",
        "SCENARIO-CAPSTONE-6435-ROW-RECHECKS",
        "SCENARIO-CAPSTONE-6435-CLAIM-ELIGIBILITY",
        "SCENARIO-CAPSTONE-6435-RETIREMENT-AND-ATTACKS",
        "SCENARIO-CAPSTONE-6435-FIELD-PRINCIPLES",
        exp6435.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in exp6435.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_capstone_6435_hashes_record_missing_and_malformed_inputs() -> None:
    """SCENARIO-CAPSTONE-6435-HASHES: absent inputs stay visible."""

    hashes = exp6435.hash_required_inputs(REPO)

    assert hashes["roadmaps"]["active_roadmap"]["exists"] is True
    assert hashes["roadmaps"]["staged_roadmap"]["exists"] is False
    assert hashes["artifacts"]["exp6424"]["exists"] is True
    assert hashes["artifacts"]["exp6434"]["exists"] is True
    assert hashes["artifacts"]["exp6434"]["json_loadable"] is False
    assert hashes["data_rows_and_manifests"]["exp6427_rows"]["file_count"] > 100
    assert hashes["sources"]["exp6435_source"]["exists"] is True
    assert hashes["tests"]["exp6435_tests"]["exists"] is True
    assert any(
        item["path"] == "research-roadmap-next.yaml"
        for item in hashes["missing_inputs"]
    )
    assert any(
        item["path"] == exp6435.EXPECTED_ARTIFACTS["exp6434"].as_posix()
        for item in hashes["malformed_inputs"]
    )


def test_req_capstone_6435_helper_branches_fail_closed(tmp_path: Path) -> None:
    """REQ-CAPSTONE-6435: helper edge cases preserve blockers."""

    missing_dir = exp6435._directory_entry(tmp_path, Path("absent"), "rows")
    assert missing_dir["exists"] is False

    git_error = exp6435._git(["not-a-real-git-subcommand"], REPO)
    assert git_error.startswith("git_exit_")

    missing_json = exp6435._load_json_record(tmp_path / "missing.json")
    assert missing_json == {
        "payload": {},
        "exists": False,
        "json_loadable": False,
        "load_error": "missing",
    }

    no_log = exp6435._conductor_outcomes(tmp_path)
    assert all(row["status"] == "missing" for row in no_log.values())

    for marker, expected in (
        ("complete_retired: same verdict", "retired"),
        ("complete_skipped: not run", "skipped"),
        ("complete_blocked: missing input", "blocked"),
    ):
        record = {"json_loadable": True, "payload": {"honest_verdict": marker}}
        assert exp6435._classification(record, {}) == expected

    invalid_factor = exp6435._scientific_eligibility(
        "exp6427",
        {"json_loadable": True, "payload": {"per_unit_rows": {"row_count": 0}}},
        {},
        "complete",
    )
    assert "factor_corpus_rows_or_ready_score_invalid" in invalid_factor["blockers"]

    invalid_admission = exp6435._scientific_eligibility(
        "exp6428",
        {"json_loadable": True, "payload": {"delta_future_exact_yield": 0.0}},
        {},
        "complete",
    )
    assert "write_time_admission_gate_invalid" in invalid_admission["blockers"]

    eligibility = exp6435.claim_eligibility(
        factor={"scientific_eligibility": False},
        admission={"scientific_eligibility": False},
        verification={"scientific_eligibility": True},
        csl={
            "exp6432": {"current_critical_flag_count": 0},
            "exp6433": {"audit_ready_score": 1.0, "open_critical_attack_ids": []},
        },
        arc={"scientific_eligibility": True},
    )
    assert eligibility["claim_blockers_by_class"]["public_factor"] == [
        "exp6427_factor_corpus_not_eligible",
        "exp6428_write_time_admission_not_eligible",
    ]
    assert eligibility["claim_blockers_by_class"]["prospective_csl"] == [
        "unexpected_audit_ready_score"
    ]


def test_scenario_capstone_6435_task_rows_preserve_flags_and_missing_arc() -> None:
    """SCENARIO-CAPSTONE-6435-PER-TASK: flags and malformed rows fail closed."""

    artifacts = exp6435.load_upstream_artifacts(REPO)
    findings = exp6435.current_adversarial_findings(REPO, artifacts)
    tasks = exp6435.per_task_reconciliations(
        repo_root=REPO,
        artifacts=artifacts,
        adversarial_findings=findings,
    )
    rollup = exp6435.expected_task_rollup(tasks)

    assert rollup["expected_upstream_task_count"] == 11
    assert rollup["counts"]["completed"] == 10
    assert rollup["counts"]["flagged"] == 2
    assert rollup["counts"]["missing"] == 1
    assert rollup["counts"]["null"] == 1
    assert rollup["counts"]["retired"] == 0
    assert rollup["counts"]["underpowered"] == 4
    assert rollup["counts"]["substantive"] == 7

    assert tasks["exp6429"]["classification"] == "flagged"
    assert tasks["exp6429"]["current_adversarial_findings"]["highest_severity"] == "critical"
    assert tasks["exp6432"]["classification"] == "flagged"
    assert tasks["exp6432"]["stamped_flags"]["flagged_adversarial"] is True
    assert tasks["exp6434"]["classification"] == "missing"
    assert tasks["exp6434"]["row_availability"]["json_loadable"] is False
    assert tasks["exp6434"]["scientific_eligibility"]["eligible"] is False
    assert tasks["exp6428"]["scientific_eligibility"]["eligible"] is True


def test_scenario_capstone_6435_row_rechecks_and_claim_boundaries(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6435-ROW-RECHECKS: rows control claim decisions."""

    artifact = _artifact(tmp_path)

    assert artifact["status"] == "complete_blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked:")
    assert artifact["factor_corpus_recheck"]["row_count"] == 144
    assert artifact["factor_corpus_recheck"]["exact_yield"]["evaluable"] == 64
    assert artifact["factor_corpus_recheck"]["per_constraint_success"]["correct"] == 206
    assert artifact["write_time_admission_recheck"]["delta_future_exact_yield"] == pytest.approx(
        4 / 48
    )
    assert artifact["write_time_admission_recheck"]["false_accept_delta"] == 0.0
    assert artifact["write_time_admission_recheck"]["protected_retention_delta"] == 0.0
    assert artifact["verification_cost_recheck"]["row_count"] == 144
    assert artifact["verification_cost_recheck"]["checker_calls_delta_selective_vs_always"] == -80
    assert artifact["verification_cost_recheck"]["current_critical_flag_count"] == 1
    assert artifact["csl_capacity_interference_held_and_audit_rechecks"]["exp6430"]["best_capacity"] == 16
    assert artifact["csl_capacity_interference_held_and_audit_rechecks"]["exp6432"]["held_future_exact_yield_delta"] == pytest.approx(
        59 / 72
    )
    assert artifact["csl_capacity_interference_held_and_audit_rechecks"]["exp6433"]["audit_ready_score"] == 0.0
    assert artifact["arc_reachability_no_solve_and_registry_rechecks"]["artifact_json_loadable"] is False

    assert artifact["public_factor_claim_eligibility"]["eligible"] is True
    assert artifact["verification_cost_claim_eligibility"]["eligible"] is False
    assert artifact["prospective_csl_claim_eligibility"]["eligible"] is False
    assert artifact["internal_arc_reachability_claim_eligibility"]["eligible"] is False
    assert artifact["public_arc_claim_eligibility"]["eligible"] is False
    assert artifact["hardware_claim_eligibility"]["eligible"] is False


def test_scenario_capstone_6435_per_unit_claim_rows_and_attack_matrix(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6435-CLAIM-ELIGIBILITY: decisions stay separate."""

    artifact = _artifact(tmp_path)
    rows = artifact["per_unit_rows"]
    attacks = artifact[
        "claim_pooling_missing_flagged_duration_row_mismatch_leakage_oracle_retuning_offpath_solve_and_hardware_attack_matrix"
    ]

    assert len([row for row in rows if row["row_type"] == "task"]) == 11
    assert len([row for row in rows if row["row_type"] == "claim_decision"]) == 6
    assert {row["claim_class"] for row in rows if row["row_type"] == "claim_decision"} == {
        "public_factor",
        "verification_cost",
        "prospective_csl",
        "internal_arc_reachability",
        "public_arc",
        "hardware",
    }
    assert {row["attack"] for row in attacks} == set(exp6435.ATTACK_IDS)
    assert all(row["fail_closed"] is True for row in attacks)
    assert all(row["claim_promoted_by_attack"] is False for row in attacks)
    assert artifact["claim_blockers_by_class"]["verification_cost"]
    assert "exp6434_artifact_missing_or_malformed" in artifact["claim_blockers_by_class"]["internal_arc_reachability"]
    assert artifact["same_verdict_retirement_decisions"]["exp6427_vs_exp6414"]["retired"] is False
    assert artifact["same_verdict_retirement_decisions"]["exp6428_vs_exp6417"]["retired"] is False
    assert artifact["same_verdict_retirement_decisions"]["exp6430_6433_vs_exp6420"]["retired"] is False
    assert artifact[
        "openspec_traceability_status_changelog_known_issues_exclusion_and_claim_reconciliation"
    ]["ops_and_traceability_edits_deferred_by_stop_rule"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == exp6435.INFERENCE_SUBSTRATE


def test_scenario_capstone_6435_field_principles_and_checksum(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6435-FIELD-PRINCIPLES: schema is self-checking."""

    artifact = _artifact(tmp_path)

    assert set(exp6435.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert set(exp6435.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_provenance"])
    for key in (
        "task_state.completed",
        "task_state.flagged",
        "task_state.missing",
        "task_state.underpowered",
        "claim_class.public_factor",
        "claim_class.verification_cost",
        "claim_class.prospective_csl",
        "claim_class.internal_arc_reachability",
        "claim_class.hardware",
        "retirement_decision.exp6427_vs_exp6414",
        "next_falsifiable_research_question.question",
    ):
        assert key in artifact["field_principles"]
    assert artifact["reproducibility_checksum"] == exp6435.payload_checksum(artifact)
    exp6435.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("public_factor_claim_eligibility", {"eligible": False}, "public_factor_claim_eligibility"),
        ("verification_cost_claim_eligibility", {"eligible": True}, "verification_cost_claim_eligibility"),
        ("prospective_csl_claim_eligibility", {"eligible": True}, "prospective_csl_claim_eligibility"),
        ("internal_arc_reachability_claim_eligibility", {"eligible": True}, "internal_arc_reachability_claim_eligibility"),
        ("public_arc_claim_eligibility", {"eligible": True}, "public_arc_claim_eligibility"),
        ("hardware_claim_eligibility", {"eligible": True}, "hardware_claim_eligibility"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
        ("honest_verdict", "blocked", "honest_verdict"),
    ],
)
def test_scenario_capstone_6435_validation_rejects_claim_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """REQ-CAPSTONE-6435: manual claim edits fail validation."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    bad[field] = value
    _with_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6435.validate_artifact(bad)


def test_req_capstone_6435_validation_rejects_nested_drift(tmp_path: Path) -> None:
    """REQ-CAPSTONE-6435: blockers and rows cannot be erased."""

    artifact = _artifact(tmp_path)

    checksum = copy.deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp6435.validate_artifact(checksum)

    missing = copy.deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing fields"):
        exp6435.validate_artifact(missing)

    drift_cases = [
        (
            lambda a: a["per_unit_rows"].pop(),
            "per_unit_rows",
        ),
        (
            lambda a: a["per_unit_rows"].pop(0),
            "per_unit_rows",
        ),
        (
            lambda a: a.__setitem__("status", "complete"),
            "status",
        ),
        (
            lambda a: a.__setitem__("inference_substrate", "live_llm"),
            "inference_substrate",
        ),
        (
            lambda a: a["verification_cost_recheck"].__setitem__(
                "current_critical_flag_count", 0
            ),
            "verification_cost_recheck",
        ),
        (
            lambda a: a["csl_capacity_interference_held_and_audit_rechecks"]["exp6433"].__setitem__(
                "audit_ready_score", 1.0
            ),
            "csl_capacity_interference_held_and_audit_rechecks",
        ),
        (
            lambda a: a["arc_reachability_no_solve_and_registry_rechecks"].__setitem__(
                "artifact_json_loadable", True
            ),
            "arc_reachability_no_solve_and_registry_rechecks",
        ),
        (
            lambda a: a[
                "claim_pooling_missing_flagged_duration_row_mismatch_leakage_oracle_retuning_offpath_solve_and_hardware_attack_matrix"
            ][0].__setitem__("attack", "missing_attack"),
            "attack_matrix",
        ),
        (
            lambda a: a[
                "claim_pooling_missing_flagged_duration_row_mismatch_leakage_oracle_retuning_offpath_solve_and_hardware_attack_matrix"
            ][0].__setitem__("claim_promoted_by_attack", True),
            "attack_matrix",
        ),
        (
            lambda a: next(iter(a["same_verdict_retirement_decisions"].values())).__setitem__(
                "retired", True
            ),
            "same_verdict_retirement_decisions",
        ),
        (
            lambda a: a["hardware_status"].__setitem__(
                "authenticated_hardware_artifact_present", True
            ),
            "hardware_status",
        ),
        (
            lambda a: a["next_falsifiable_research_question"].__setitem__(
                "version_only_continuation", True
            ),
            "next_falsifiable_research_question",
        ),
        (
            lambda a: a["field_principles"].pop("task_state.flagged"),
            "field_principles",
        ),
        (
            lambda a: a["field_provenance"].pop("status"),
            "field_provenance",
        ),
        (
            lambda a: next(iter(a["protected_files_unchanged"].values())).__setitem__(
                "unchanged", False
            ),
            "protected_files_unchanged",
        ),
    ]
    for mutate, message in drift_cases:
        bad = copy.deepcopy(artifact)
        mutate(bad)
        _with_checksum(bad)
        with pytest.raises(ValueError, match=message):
            exp6435.validate_artifact(bad)


def test_scenario_capstone_6435_writer_and_cli_emit_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CAPSTONE-6435-HASHES: writer and CLI produce stable JSON."""

    output = tmp_path / "capstone.json"
    artifact = exp6435.build_artifact(
        repo_root=REPO,
        date="20260814",
        result_path=output,
        duration_s=2.0,
        tests_run=[{"command": exp6435.RUN_COMMAND, "exit_code": 0}],
        write=True,
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert output.read_text(encoding="utf-8").endswith("\n")

    cli_out = tmp_path / "cli.json"
    assert exp6435.main(["--date", "20260814", "--output", str(cli_out)]) == 0
    assert cli_out.is_file()

    monkeypatch.setattr(exp6435, "build_artifact", lambda **_kwargs: {"schema": "bad"})
    with pytest.raises(ValueError, match="missing fields"):
        exp6435.write_artifact(repo_root=REPO, result_path=tmp_path / "bad.json")
