"""Tests for Exp6423 V552 adversarial capstone.

Spec refs: REQ-CAPSTONE-6423,
SCENARIO-CAPSTONE-6423-HASHES,
SCENARIO-CAPSTONE-6423-PER-TASK,
SCENARIO-CAPSTONE-6423-RECHECKS,
SCENARIO-CAPSTONE-6423-ATTACKS-AND-ELIGIBILITY,
SCENARIO-CAPSTONE-6423-PRD-NEXT-QUESTION,
SCENARIO-CAPSTONE-6423-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6423_v552_adversarial_capstone as exp6423


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / exp6423.SPEC_RELATIVE_PATH


def _artifact(tmp_path: Path) -> dict[str, Any]:
    commands = (
        exp6423.RUN_COMMAND,
        exp6423.FOCUSED_TEST_COMMAND,
        exp6423.COVERAGE_RUN_COMMAND,
        exp6423.COVERAGE_REPORT_COMMAND,
    )
    return exp6423.build_artifact(
        repo_root=REPO,
        date="20260814",
        result_path=tmp_path / exp6423.RESULT_RELATIVE_PATH.name,
        duration_s=2.0,
        tests_run=[{"command": command, "exit_code": 0} for command in commands],
        write=True,
    )


def _with_checksum(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = exp6423.payload_checksum(payload)
    return payload


def test_req_capstone_6423_spec_declares_v552_contract() -> None:
    """REQ-CAPSTONE-6423: OpenSpec owns the V552 capstone contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CAPSTONE-6423") :]
    for marker in (
        "SCENARIO-CAPSTONE-6423-HASHES",
        "SCENARIO-CAPSTONE-6423-PER-TASK",
        "SCENARIO-CAPSTONE-6423-RECHECKS",
        "SCENARIO-CAPSTONE-6423-ATTACKS-AND-ELIGIBILITY",
        "SCENARIO-CAPSTONE-6423-PRD-NEXT-QUESTION",
        "SCENARIO-CAPSTONE-6423-FIELD-PRINCIPLES",
        exp6423.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in exp6423.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_capstone_6423_hashes_present_and_missing_inputs() -> None:
    """SCENARIO-CAPSTONE-6423-HASHES: missing requested inputs are visible."""

    hashes = exp6423.hash_required_inputs(REPO)

    assert hashes["artifacts"]["exp6410"]["exists"] is True
    assert hashes["artifacts"]["exp6422"]["exists"] is True
    assert hashes["sources"]["exp6423_source"]["exists"] is True
    assert hashes["specs"]["capstone"]["exists"] is True
    assert hashes["ops"]["status"]["exists"] is True
    assert hashes["registries_and_ledgers"]["arc_solve_registry"]["exists"] is True
    assert hashes["registries_and_ledgers"]["requested_claim_eligibility_ledger"]["exists"] is False
    assert hashes["checkers"]["verify_research_coverage"]["exists"] is False
    assert any(
        item["path"] == "ops/claim-eligibility-ledger.json"
        for item in hashes["missing_inputs"]
    )
    assert exp6423._path_entry(REPO, Path("not-present-exp6423"), "missing")["sha256"] is None


def test_scenario_capstone_6423_per_task_reconciliation_preserves_flags() -> None:
    """SCENARIO-CAPSTONE-6423-PER-TASK: flagged positives stay ineligible."""

    artifacts = exp6423.load_upstream_artifacts(REPO)
    findings = exp6423.current_adversarial_findings(REPO, artifacts)
    tasks = exp6423.per_task_reconciliations(
        repo_root=REPO,
        artifacts=artifacts,
        adversarial_findings=findings,
    )
    rollup = exp6423.expected_task_rollup(tasks)

    assert rollup["expected_upstream_task_count"] == 13
    assert rollup["counts"]["flagged"] == 2
    assert rollup["counts"]["null"] == 1
    assert rollup["counts"]["missing"] == 0
    assert rollup["counts"]["retired"] == 0

    exp6414 = tasks["exp6414"]
    exp6417 = tasks["exp6417"]
    assert exp6414["classification"] == "flagged"
    assert exp6417["classification"] == "flagged"
    assert exp6414["conductor_outcome"]["status"] == "FLAGGED"
    assert exp6417["conductor_outcome"]["status"] == "FLAGGED"
    assert exp6414["current_adversarial_findings"]["highest_severity"] == "critical"
    assert exp6417["current_adversarial_findings"]["highest_severity"] == "critical"
    assert exp6414["scientific_eligibility"]["eligible"] is False
    assert exp6417["scientific_eligibility"]["eligible"] is False
    assert tasks["exp6413"]["scientific_eligibility"]["eligible"] is True


def test_scenario_capstone_6423_rechecks_and_claim_boundaries(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6423-RECHECKS: gates recompute from primary fields."""

    artifact = _artifact(tmp_path)

    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["v551_corrigendum_boundary_applied"]["applied"] is True
    assert artifact["v551_corrigendum_boundary_applied"]["exp6408_counts_as_powered_evidence"] is False
    assert artifact["authentic_family_and_receipt_coverage_recheck"]["authentic_family_count"] == 3
    assert artifact["authentic_family_and_receipt_coverage_recheck"]["receipt_coverage_ready"] is True
    assert artifact["ccg_optimum_preservation_recheck"]["optimum_preservation_rate"] == 1.0
    assert artifact["selective_refinement_recheck"]["selective_vs_always_work_delta"] < 0
    assert artifact["authentic_admission_recheck"]["eligible_after_flag_check"] is False
    assert artifact["prospective_and_held_csl_rechecks"]["prospective_csl_claim_eligible_after_audit"] is False
    assert artifact["csl_audit_recheck"]["ready_score"] == 0.0
    assert artifact["arc_policy_and_held_audit_rechecks"]["internal_policy_claim_eligible"] is True
    assert artifact["arc_no_solve_and_registry_checks"]["solve_registry_modified"] is False
    assert artifact["arc_no_solve_and_registry_checks"]["level_solve_claimed"] is False
    assert artifact["deterministic_protocol_claim_eligibility"]["eligible"] is False
    assert artifact["authentic_powered_factor_claim_eligibility"]["eligible"] is False
    assert artifact["prospective_csl_claim_eligibility"]["eligible"] is False
    assert artifact["public_factor_claim_eligibility"] is False
    assert artifact["internal_arc_policy_claim_eligibility"]["eligible"] is True
    assert artifact["public_arc_claim_eligibility"] is False
    assert artifact["hardware_status"]["hardware_speedup_claimed"] is False
    assert len(artifact["remaining_prd_gaps"]) == 3
    assert artifact["next_falsifiable_research_question"]["version_only_continuation"] is False


def test_scenario_capstone_6423_attack_matrix_and_field_principles(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-6423-ATTACKS-AND-ELIGIBILITY: attacks fail closed."""

    artifact = _artifact(tmp_path)
    attacks = artifact[
        "claim_pooling_missing_flagged_duration_inheritance_leakage_oracle_retuning_offpath_solve_and_public_attack_matrix"
    ]

    assert {row["attack"] for row in attacks} == set(exp6423.ATTACK_IDS)
    assert all(row["claim_promoted"] is False for row in attacks)
    assert all(row["fail_closed"] is True for row in attacks)
    assert artifact["same_verdict_retirement_decisions"][0]["retirement_triggered"] is False
    assert artifact[
        "openspec_traceability_status_changelog_known_issues_and_architecture_reconciliation"
    ]["ops_and_traceability_edits_deferred_by_stop_rule"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == exp6423.INFERENCE_SUBSTRATE
    assert set(exp6423.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert set(exp6423.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_provenance"])
    for key in (
        "same_verdict_retirement_decisions.exp6403_v550_prior_failure",
        "remaining_prd_gaps.scientific_provenance",
        "remaining_prd_gaps.prospective_self_learning",
        "remaining_prd_gaps.public_arc_and_hardware",
        "next_falsifiable_research_question.question",
    ):
        assert key in artifact["field_principles"]
    assert artifact["reproducibility_checksum"] == exp6423.payload_checksum(artifact)
    exp6423.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("public_factor_claim_eligibility", True, "public_factor_claim_eligibility"),
        ("public_arc_claim_eligibility", True, "public_arc_claim_eligibility"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("status", "partial", "status"),
        ("honest_verdict", "blocked", "honest_verdict"),
    ],
)
def test_scenario_capstone_6423_validation_rejects_overclaims(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """SCENARIO-CAPSTONE-6423-FIELD-PRINCIPLES: overclaims are rejected."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    bad[field] = value
    _with_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6423.validate_artifact(bad)


def test_req_capstone_6423_validation_rejects_nested_drift(tmp_path: Path) -> None:
    """REQ-CAPSTONE-6423: nested blockers cannot be cleared by edit."""

    artifact = _artifact(tmp_path)

    checksum = copy.deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp6423.validate_artifact(checksum)

    missing = copy.deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing fields"):
        exp6423.validate_artifact(missing)

    drift_cases = [
        (
            lambda a: a["authentic_admission_recheck"].__setitem__(
                "eligible_after_flag_check", True
            ),
            "authentic_admission_recheck",
        ),
        (
            lambda a: a["prospective_and_held_csl_rechecks"].__setitem__(
                "prospective_csl_claim_eligible_after_audit", True
            ),
            "prospective_and_held_csl_rechecks",
        ),
        (
            lambda a: a["csl_audit_recheck"].__setitem__("ready_score", 1.0),
            "csl_audit_recheck",
        ),
        (
            lambda a: a["arc_no_solve_and_registry_checks"].__setitem__(
                "level_solve_claimed", True
            ),
            "arc_no_solve_and_registry_checks",
        ),
        (
            lambda a: a["arc_no_solve_and_registry_checks"].__setitem__(
                "solve_registry_modified", True
            ),
            "arc_no_solve_and_registry_checks",
        ),
        (
            lambda a: a["same_verdict_retirement_decisions"][0].__setitem__(
                "retirement_triggered", True
            ),
            "same_verdict_retirement_decisions",
        ),
        (
            lambda a: a[
                "claim_pooling_missing_flagged_duration_inheritance_leakage_oracle_retuning_offpath_solve_and_public_attack_matrix"
            ][0].__setitem__("claim_promoted", True),
            "attack_matrix",
        ),
        (
            lambda a: a[
                "claim_pooling_missing_flagged_duration_inheritance_leakage_oracle_retuning_offpath_solve_and_public_attack_matrix"
            ][0].__setitem__("attack", "not_declared"),
            "attack_matrix",
        ),
        (
            lambda a: a["remaining_prd_gaps"].pop(),
            "remaining_prd_gaps",
        ),
        (
            lambda a: a["next_falsifiable_research_question"].__setitem__(
                "version_only_continuation", True
            ),
            "next_falsifiable_research_question",
        ),
        (
            lambda a: next(iter(a["protected_files_unchanged"].values())).__setitem__(
                "unchanged", False
            ),
            "protected_files_unchanged",
        ),
        (
            lambda a: a["field_principles"].pop("public_factor_claim_eligibility"),
            "field_principles",
        ),
        (
            lambda a: a["field_provenance"].pop("public_arc_claim_eligibility"),
            "field_provenance",
        ),
    ]
    for mutate, message in drift_cases:
        bad = copy.deepcopy(artifact)
        mutate(bad)
        _with_checksum(bad)
        with pytest.raises(ValueError, match=message):
            exp6423.validate_artifact(bad)


def test_req_capstone_6423_defensive_branches_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6423-PER-TASK: missing and bad rows fail closed."""

    findings = exp6423.current_adversarial_findings(REPO, {"exp6410": {}})
    assert findings["exp6410"]["highest_severity"] == "missing"

    missing_log = exp6423._conductor_outcomes(tmp_path)
    assert missing_log["exp6410"]["status"] == "missing"

    assert exp6423._classification("exp6410", {}, {}) == "missing"
    assert (
        exp6423._classification(
            "exp6410",
            {"status": "retired"},
            {"highest_severity": "clean"},
        )
        == "retired"
    )
    assert (
        exp6423._classification(
            "exp6410",
            {"honest_verdict": "blocked by precondition"},
            {"highest_severity": "clean"},
        )
        == "blocked"
    )
    assert (
        exp6423._classification(
            "exp6410",
            {"status": "partial"},
            {"highest_severity": "clean"},
        )
        == "partial"
    )

    assert exp6423._eligible_field({"row": 1}, "row", "nested") is None
    assert "authentic_receipt_contract_incomplete" in exp6423._scientific_eligibility(
        "exp6413",
        {"authentic_family_count": 2},
        "complete",
    )["blockers"]
    assert "arc_public_boundary_not_false" in exp6423._scientific_eligibility(
        "exp6421",
        {"public_arc_claim_eligibility": True},
        "complete",
    )["blockers"]


def test_scenario_capstone_6423_writer_and_cli_emit_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CAPSTONE-6423-HASHES: writer and CLI emit stable JSON."""

    output = tmp_path / "capstone.json"
    artifact = exp6423.build_artifact(
        repo_root=REPO,
        date="20260814",
        result_path=output,
        duration_s=2.0,
        tests_run=[{"command": exp6423.RUN_COMMAND, "exit_code": 0}],
        write=True,
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert output.read_text(encoding="utf-8").endswith("\n")

    cli_out = tmp_path / "cli.json"
    assert exp6423.main(["--date", "20260814", "--output", str(cli_out)]) == 0
    assert cli_out.is_file()

    monkeypatch.setattr(exp6423, "build_artifact", lambda **_kwargs: {"schema": "bad"})
    with pytest.raises(ValueError, match="missing fields"):
        exp6423.write_artifact(repo_root=REPO, result_path=tmp_path / "bad.json")
