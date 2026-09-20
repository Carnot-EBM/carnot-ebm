"""Tests for the V652 thirteen-disposition capstone.

Spec refs: REQ-REPORT-7446 and SCENARIO-REPORT-7446-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7446_v652_capstone as capstone
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash


@pytest.fixture(scope="module")
def contract() -> dict[str, object]:
    """Load both real V652 authorities once for focused tests."""

    return capstone.load_contract(capstone.REPO_ROOT)


@pytest.fixture(scope="module")
def evidence(contract: dict[str, object]) -> dict[str, dict[str, object]]:
    """Authenticate the twelve real predecessors once for focused tests."""

    tasks = contract["tasks"]
    assert isinstance(tasks, list)
    return capstone.collect_evidence(capstone.REPO_ROOT, tasks)


@pytest.fixture(scope="module")
def artifact(
    contract: dict[str, object], evidence: dict[str, dict[str, object]]
) -> dict[str, object]:
    """Build one deterministic terminal artifact for mutation tests."""

    return capstone.build_artifact(
        capstone.REPO_ROOT,
        contract,
        evidence,
        _passing_validation(),
        capstone.evaluate_publication_gates(),
        started_at_utc="2026-09-20T12:00:00+00:00",
        completed_at_utc="2026-09-20T12:00:01+00:00",
        duration_s=1.0,
        phase_spans=capstone.zero_test_phase_spans(),
    )


def _passing_validation() -> dict[str, object]:
    names = (*capstone.REQUIRED_CHECK_NAMES, *capstone.TERMINAL_CHECK_NAMES)
    return {
        "required_checks_passed": True,
        "terminal_validation_passed": True,
        "validation_receipts": [
            {
                "name": name,
                "required": True,
                "passed": True,
                "exit_code": 0,
                "duration_s": 0.0,
                "log_path": f"/tmp/{name}.log",
                "log_sha256": "sha256:" + "0" * 64,
            }
            for name in names
        ],
        "repository_health": {"status": "not_part_of_affected_validity"},
    }


def test_exact_contract_and_gate_declarations_are_independently_bound(
    contract: dict[str, object],
) -> None:
    # REQ-REPORT-7446; SCENARIO-REPORT-7446-CONTRACT.
    assert contract["comparison_passed"] is True
    assert contract["errors"] == []
    tasks = contract["tasks"]
    assert isinstance(tasks, list)
    assert [row["id"] for row in tasks] == list(capstone.EXPECTED_TASK_IDS)
    assert len(contract["contract_rows"]) == 13
    assert sum(len(row["gates"]) for row in contract["contract_rows"]) == 12
    assert all(row["passed"] for row in contract["contract_rows"])


def test_contract_mutations_fail_without_using_the_contract_audit_artifact() -> None:
    # REQ-REPORT-7446; SCENARIO-REPORT-7446-CONTRACT.
    markdown = (capstone.REPO_ROOT / capstone.DESIGN_PATH).read_text(encoding="utf-8")
    roadmap = capstone.load_yaml_mapping(capstone.REPO_ROOT / capstone.ROADMAP_PATH)
    changed_markdown = markdown.replace("| 13 | exp7446-capstone |", "| 13 | exp7999-capstone |", 1)
    result = capstone.compare_contract_authorities(changed_markdown, roadmap)
    assert result["comparison_passed"] is False
    assert "markdown_task_order" in result["errors"]

    changed_yaml = deepcopy(roadmap)
    changed_yaml["tasks"][5]["gated_on"][0]["artifact_field"] = "undeclared_score"
    result = capstone.compare_contract_authorities(markdown, changed_yaml)
    assert result["comparison_passed"] is False
    assert "producer_field_declaration" in result["errors"]

    wrong_markdown_milestone = markdown.replace(
        "**Milestone:** `2026.09.652`", "**Milestone:** `2026.09.650`", 1
    )
    result = capstone.compare_contract_authorities(wrong_markdown_milestone, roadmap)
    assert "markdown_milestone" in result["errors"]

    changed_yaml = deepcopy(roadmap)
    changed_yaml["milestone"] = "2026.09.650"
    changed_yaml["tasks"] = changed_yaml["tasks"][:-1]
    changed_yaml["tasks"][0], changed_yaml["tasks"][1] = (
        changed_yaml["tasks"][1],
        changed_yaml["tasks"][0],
    )
    result = capstone.compare_contract_authorities(markdown, changed_yaml)
    assert {"yaml_milestone", "yaml_task_order", "task_count"} <= set(result["errors"])


def test_real_predecessors_preserve_verdict_flags_rows_and_receipts(
    evidence: dict[str, dict[str, object]],
) -> None:
    # REQ-REPORT-7446; SCENARIO-REPORT-7446-EVIDENCE.
    assert list(evidence) == list(capstone.EXPECTED_TASK_IDS[:-1])
    assert all(row["authenticated"] for row in evidence.values())
    assert all(row["row_manifest_authenticated"] for row in evidence.values())
    assert all(row["validation_receipts_authenticated"] for row in evidence.values())
    assert evidence["exp7441-decision-audit"]["verdict_class"] == "disqualified"
    assert evidence["exp7442-span-capture"]["flagged_adversarial"] is True
    assert evidence["exp7443-span-audit"]["flagged_adversarial"] is True
    assert evidence["exp7439-certified-decisions"]["raw_row_count"] == 21
    assert evidence["exp7440-mixture-learning"]["raw_shard_count"] == 17


def test_row_and_validation_manifests_are_hash_bound(tmp_path: Path) -> None:
    # REQ-REPORT-7446; SCENARIO-REPORT-7446-EVIDENCE.
    shard = tmp_path / "rows.jsonl"
    shard.write_text('{"unit_id":"a"}\n', encoding="utf-8")
    log = tmp_path / "check.log"
    log.write_text("passed\n", encoding="utf-8")
    payload = {
        "rows": [{"unit_id": "a", "metric": 0}],
        "row_shards": [
            {
                "path": shard.name,
                "rows": 1,
                "size_bytes": shard.stat().st_size,
                "sha256": capstone.sha256_file(shard),
            }
        ],
        "validation_receipts": [
            {
                "name": "focused_pytest",
                "required": True,
                "passed": True,
                "exit_code": 0,
                "log_path": log.name,
                "log_sha256": capstone.sha256_file(log),
            }
        ],
    }
    rows = capstone.authenticate_row_manifest(tmp_path, payload)
    receipts = capstone.authenticate_validation_receipts(tmp_path, payload)
    assert rows == {
        "authenticated": True,
        "inline_row_count": 1,
        "inline_rows_sha256": canonical_hash(payload["rows"]),
        "raw_shard_count": 1,
        "raw_shard_row_count": 1,
        "failures": [],
    }
    assert receipts["authenticated"] is True
    assert receipts["required_passed"] is True
    shard.write_text("changed\n", encoding="utf-8")
    assert capstone.authenticate_row_manifest(tmp_path, payload)["authenticated"] is False
    log.write_text("failed\n", encoding="utf-8")
    assert capstone.authenticate_validation_receipts(tmp_path, payload)["authenticated"] is False

    assert capstone.authenticate_row_manifest(tmp_path, {})["failures"] == ["inline_rows_missing"]
    missing_shard = deepcopy(payload)
    missing_shard["row_shards"][0]["path"] = "absent.jsonl"
    assert (
        "row_shard_missing:0"
        in capstone.authenticate_row_manifest(tmp_path, missing_shard)["failures"]
    )
    invalid_count = deepcopy(payload)
    invalid_count["row_shards"][0]["rows"] = True
    assert (
        "row_shard_rows:0"
        in capstone.authenticate_row_manifest(tmp_path, invalid_count)["failures"]
    )
    assert capstone.authenticate_validation_receipts(tmp_path, {})["failures"] == [
        "validation_receipts_missing"
    ]
    bad_receipt = {
        "validation_receipts": [
            {
                "name": None,
                "passed": "yes",
                "exit_code": "zero",
                "log_path": "absent.log",
                "log_sha256": "sha256:bad",
            }
        ]
    }
    failures = capstone.authenticate_validation_receipts(tmp_path, bad_receipt)["failures"]
    assert failures == ["validation_receipt_shape:0", "validation_log_missing:0"]


def test_missing_declared_artifact_uses_only_an_exact_pre_gate(tmp_path: Path) -> None:
    # REQ-REPORT-7446; SCENARIO-REPORT-7446-EVIDENCE.
    task = {
        "id": "exp7442-span-capture",
        "deliverable": "results/experiment_7442_v652_span_capture.json",
    }
    absent = capstone.load_evidence_slot(tmp_path, task)
    assert absent["source_kind"] == "missing"
    assert absent["verdict_class"] == "blocked"
    assert absent["authenticated"] is False
    fallback = tmp_path / "results/experiment_7442_span_capture.json"
    fallback.parent.mkdir(parents=True)
    fallback.write_text(
        json.dumps(
            {
                "schema": "blocked_gate_check_v1",
                "experiment_id": "exp7442-span-capture",
                "status": "blocked",
                "honest_verdict": "blocked_gate_check_failed",
                "failed_upstream": "exp7437-span-protocol",
                "failed_field": "span_protocol_ready_score",
                "failed_expected": 1,
                "failed_observed": 0,
            }
        ),
        encoding="utf-8",
    )
    blocked = capstone.load_evidence_slot(tmp_path, task)
    assert blocked["source_kind"] == "structured_pre_gate"
    assert blocked["authenticated"] is True
    assert blocked["declared_path"] == task["deliverable"]
    assert blocked["source_path"] == "results/experiment_7442_span_capture.json"
    assert blocked["gate_check_summary"]["field"] == "span_protocol_ready_score"


def test_gate_eligibility_keeps_null_and_rejects_invalid_scores(
    contract: dict[str, object], evidence: dict[str, dict[str, object]]
) -> None:
    # REQ-REPORT-7446; SCENARIO-REPORT-7446-GATES.
    rows = capstone.audit_structured_gates(contract, evidence)
    assert len(rows) == 12
    assert all(row["producer_in_roadmap"] for row in rows)
    assert all(row["producer_field_declared"] for row in rows)
    assert all(row["scalar_passed"] for row in rows)
    assert all(row["source_admissible"] for row in rows)
    assert capstone.gate_eligibility(1, "null", False) is True
    assert capstone.gate_eligibility(1, "disqualified", False) is False
    assert capstone.gate_eligibility(1, "null", True) is False


def test_auditor_cross_checks_preserve_static_online_and_extraction_defects(
    evidence: dict[str, dict[str, object]],
) -> None:
    # REQ-REPORT-7446; SCENARIO-REPORT-7446-BRANCHES.
    rows = capstone.auditor_cross_checks(evidence)
    by_branch = {row["branch"]: row for row in rows}
    assert set(by_branch) == {"static_decision", "online_learning", "extraction"}
    assert by_branch["static_decision"]["auditor_valid"] is True
    assert by_branch["online_learning"]["auditor_valid"] is False
    assert by_branch["online_learning"]["errors"] == [
        "missing_expert_predictions",
        "weight_update_replay_incomplete",
    ]
    assert by_branch["extraction"]["evaluation_coverage"] == 0.0
    assert by_branch["extraction"]["audited_outcome_count"] == 104
    assert all(row["passed"] for row in rows)


def test_claims_separate_completion_from_scientific_benefit(
    evidence: dict[str, dict[str, object]],
) -> None:
    # REQ-REPORT-7446; SCENARIO-REPORT-7446-BRANCHES.
    claims = capstone.reduce_claim_matrix(evidence)
    assert list(claims) == list(capstone.CLAIM_BRANCHES)
    assert claims["contract"]["completion_score"] == 1
    assert claims["contract"]["benefit_score"] == 0
    assert claims["runtime_recovery"]["scientific_progress_claimed"] is False
    assert claims["static_decision"]["disposition"] == "null"
    assert claims["online_learning"]["disposition"] == "disqualified"
    assert claims["extraction"]["disposition"] == "disqualified"
    assert claims["arc"]["benefit_score"] == 0
    assert claims["hardware"]["benefit_score"] == 0


def test_terminal_precedence_never_uses_partial_for_upstream_state(
    contract: dict[str, object], evidence: dict[str, dict[str, object]]
) -> None:
    # REQ-REPORT-7446; SCENARIO-REPORT-7446-BRANCHES.
    terminal = capstone.classify_terminal(contract, evidence, _passing_validation())
    assert terminal["verdict_class"] == "disqualified"
    assert terminal["honest_verdict"] == (
        "complete_disqualified_required_v652_science_with_thirteen_dispositions"
    )

    valid = deepcopy(evidence)
    for row in valid.values():
        row["valid"] = True
        row["available"] = True
        row["authenticated"] = True
        row["flagged_adversarial"] = False
        if row["verdict_class"] == "disqualified":
            row["verdict_class"] = "null"
    assert (
        capstone.classify_terminal(contract, valid, _passing_validation())["verdict_class"]
        == "null"
    )
    valid["exp7440-mixture-learning"]["available"] = False
    valid["exp7440-mixture-learning"]["authenticated"] = True
    assert (
        capstone.classify_terminal(contract, valid, _passing_validation())["verdict_class"]
        == "blocked"
    )
    broken_validation = _passing_validation()
    broken_validation["required_checks_passed"] = False
    assert (
        capstone.classify_terminal(contract, valid, broken_validation)["verdict_class"]
        == "disqualified"
    )
    broken_contract = deepcopy(contract)
    broken_contract["comparison_passed"] = False
    assert (
        capstone.classify_terminal(broken_contract, valid, _passing_validation())["verdict_class"]
        == "disqualified"
    )
    unauthenticated = deepcopy(valid)
    unauthenticated["exp7434-contract-methods"]["authenticated"] = False
    assert (
        capstone.classify_terminal(contract, unauthenticated, _passing_validation())[
            "verdict_class"
        ]
        == "disqualified"
    )


def test_retirement_requires_the_same_exact_declared_verdict(
    contract: dict[str, object], evidence: dict[str, dict[str, object]]
) -> None:
    # REQ-REPORT-7446; SCENARIO-REPORT-7446-DECISIONS.
    terminal = capstone.classify_terminal(contract, evidence, _passing_validation())
    rows = capstone.retirement_rows(contract["tasks"], evidence, terminal)
    assert rows
    assert not any(row["permanent_retirement"] for row in rows)
    changed = deepcopy(evidence)
    changed["exp7442-span-capture"]["honest_verdict"] = (
        "complete_disqualified_internal_artifact_validation_failed"
    )
    rows = capstone.retirement_rows(contract["tasks"], changed, terminal)
    repeated = next(row for row in rows if row["task_id"] == "exp7442-span-capture")
    assert repeated["same_exact_verdict"] is True
    assert repeated["permanent_retirement"] is True
    assert repeated["decision"] == "retire"


def test_continuations_name_changed_causes_and_forbidden_work(
    evidence: dict[str, dict[str, object]],
) -> None:
    # REQ-REPORT-7446; SCENARIO-REPORT-7446-DECISIONS.
    claims = capstone.reduce_claim_matrix(evidence)
    rows = capstone.continuation_rows(claims, [])
    assert {row["decision"] for row in rows} <= {"continue", "defer", "retire"}
    assert all(row["changed_cause_or_prerequisite"] for row in rows)
    assert not any(row["decision"] == "retire" for row in rows)
    by_mechanism = {row["mechanism"]: row for row in rows}
    assert by_mechanism["online_learning"]["decision"] == "continue"
    assert "expert prediction" in by_mechanism["online_learning"]["changed_cause_or_prerequisite"]
    obligations = capstone.unresolved_obligations(evidence)
    size_gate = next(row for row in obligations if row["obligation_id"] == "conductor_size_gate")
    assert size_gate["prohibited_path"] == "scripts/research_conductor.py"
    assert size_gate["current_task_action"] == "none"
    retired = capstone.continuation_rows(
        claims,
        [{"task_id": "exp7440-mixture-learning", "permanent_retirement": True}],
    )
    online = next(row for row in retired if row["mechanism"] == "online_learning")
    assert online["decision"] == "retire"


def test_artifact_has_thirteen_currently_reduced_dispositions(
    artifact: dict[str, object],
) -> None:
    # REQ-REPORT-7446; SCENARIO-REPORT-7446-ARTIFACT.
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == ZERO_INVOCATION_COUNTS
    assert artifact["verdict_class"] == "disqualified"
    assert len(artifact["task_dispositions"]) == 13
    assert artifact["task_dispositions"][-1]["source_kind"] == "current_work"
    assert artifact["rows"] == artifact["task_dispositions"]
    assert artifact["promotion_score"] == 0
    assert artifact["capstone_complete_score"] == 1
    assert artifact["publication_gates"]["headline_scope"] == "FoVer dual-condition AUROC"
    assert capstone.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["task_dispositions"].pop()
    assert "task_dispositions_invalid" in capstone.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["claim_matrix"]["static_decision"]["benefit_score"] = 1
    assert "claim_matrix_invalid" in capstone.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["promotion_score"] = 1
    assert "promotion_invalid" in capstone.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_invalid" in capstone.validate_artifact(changed)


def test_artifact_validator_rejects_each_derived_surface(
    artifact: dict[str, object],
) -> None:
    # REQ-REPORT-7446; SCENARIO-REPORT-7446-ARTIFACT.
    mutations = (
        ("schema", "bad", "identity_invalid"),
        ("status", "running", "lifecycle_invalid"),
        ("MODEL_SPECS", ["unexpected"], "model_contract_invalid"),
        ("execution_venue", "device", "substrate_invalid"),
        ("publication_gates", {}, "publication_gates_invalid"),
        ("auditor_cross_checks", [], "auditor_cross_checks_invalid"),
        ("structured_gate_audit", [], "structured_gate_audit_invalid"),
        ("retirement_rows", [], "retirement_rows_invalid"),
        ("continuation_rows", [], "continuation_rows_invalid"),
        ("unresolved_obligations", [], "unresolved_obligations_invalid"),
        ("honest_verdict", "complete_wrong", "terminal_reduction_invalid"),
        ("capstone_complete_score", 0, "capstone_score_invalid"),
        ("acceptance_gate_results", [], "acceptance_gates_invalid"),
        ("field_principles", {}, "field_principles_invalid"),
    )
    for field, value, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in capstone.validate_artifact(changed), field

    changed = deepcopy(artifact)
    changed["source_artifact_hashes"]["current-module"]["sha256"] = "sha256:bad"
    assert "source_hash_mismatch" in capstone.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"]["contract-comparison"]["sha256"] = "sha256:bad"
    assert "source_hash_mismatch" in capstone.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"] = []
    assert "source_hash_mismatch" in capstone.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"]["current-module"]["path"] = 7
    assert "source_hash_mismatch" in capstone.validate_artifact(changed)
    changed = deepcopy(artifact)
    del changed["schema"]
    assert capstone.validate_artifact(changed) == ["missing_required_field:schema"]
    assert capstone._receipt_passed(None, capstone.REQUIRED_CHECK_NAMES) is False

    incomplete = deepcopy(artifact)
    incomplete["validation_receipts"] = []
    incomplete["capstone_complete_score"] = 0
    incomplete["task_dispositions"][-1]["authenticated"] = False
    incomplete["task_dispositions"][-1]["valid"] = False
    incomplete["task_dispositions"][-1]["validation_receipts_authenticated"] = False
    incomplete["rows"] = deepcopy(incomplete["task_dispositions"])
    assert "terminal_validation_incomplete" in capstone.validate_artifact(incomplete)
    assert capstone.independent_reduce(incomplete)


def test_validation_plan_is_exact_and_date_is_frozen(tmp_path: Path) -> None:
    # REQ-REPORT-7446; SCENARIO-REPORT-7446-ARTIFACT.
    tmp_path.joinpath("basetemp").mkdir()
    tmp_path.joinpath("coverage").mkdir()
    commands = capstone.build_validation_plan(capstone.REPO_ROOT, tmp_path)
    assert [row.name for row in commands] == list(capstone.REQUIRED_CHECK_NAMES)
    assert capstone.validate_validation_plan(capstone.REPO_ROOT, commands) == []
    focused = next(row for row in commands if row.name == "focused_pytest")
    assert "-n" in focused.argv
    assert "--no-cov" in focused.argv
    assert capstone.TEST_PATH.as_posix() in focused.argv
    assert capstone.date_argument(capstone.RUN_DATE) == capstone.RUN_DATE
    with pytest.raises(Exception, match="run date must be"):
        capstone.date_argument("20260919")


def test_small_helpers_reject_malformed_identity_and_candidate(tmp_path: Path) -> None:
    # REQ-REPORT-7446; SCENARIO-REPORT-7446-EVIDENCE/ARTIFACT.
    assert capstone.numeric_experiment_id("experiment_7446_v652") == 7446
    assert capstone.numeric_experiment_id("none") is None
    assert capstone.terminal_status({"honest_verdict": "complete_null"}) is True
    assert capstone.terminal_status({"status": "running"}) is False
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="unreadable JSON object"):
        capstone.load_json_object(missing)
    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        capstone.load_json_object(list_path)
    assert capstone.validate_artifact([]) == ["artifact_mapping_required"]
    assert capstone.main(["--validate", str(list_path)]) == 1


def test_cli_dispatches_both_cold_modes_and_the_public_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-REPORT-7446; SCENARIO-REPORT-7446-ARTIFACT.
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(capstone, "validate_artifact", lambda *_args, **_kwargs: [])
    assert capstone.main(["--validate", str(candidate)]) == 0
    monkeypatch.setattr(capstone, "independent_reduce", lambda *_args, **_kwargs: ["bad"])
    assert capstone.main(["--independent-reduce", str(candidate)]) == 1
    monkeypatch.setattr(
        capstone,
        "run_experiment",
        lambda *_args, **_kwargs: {
            "status": "complete",
            "verdict_class": "null",
            "capstone_complete_score": 1,
        },
    )
    assert capstone.main(["--date", capstone.RUN_DATE]) == 0
