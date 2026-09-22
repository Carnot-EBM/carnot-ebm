"""Tests for REQ-REPORT-7515 and SCENARIO-REPORT-7515-*.

The real V657 artifacts exercise custody and claim reduction. Private copies
exercise fail-closed readers without changing the research record.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from carnot import experiment_7515_v657_capstone as capstone


def test_contract_has_exactly_thirteen_v657_tasks() -> None:
    """SCENARIO-REPORT-7515-CONTRACT: both authorities fix task identity."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    assert contract["comparison_passed"] is True
    assert contract["selected_roadmap_path"] == "research-roadmap.yaml"
    assert [task["id"] for task in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)

    changed = deepcopy(contract["roadmap"])
    changed["tasks"][12]["deliverable"] = "results/wrong.json"
    markdown = (capstone.REPO_ROOT / capstone.DESIGN_PATH).read_text()
    comparison = capstone.compare_contract_authorities(markdown, changed)
    assert comparison["passed"] is False


def test_inventory_authenticates_rows_receipts_flags_and_scores() -> None:
    """SCENARIO-REPORT-7515-INVENTORY: producer limits remain exact."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    assert len(evidence) == 12
    interface = evidence["exp7504-evidence-interface"]
    assert interface["evidence_state"] == "terminal"
    assert interface["ready_value_fields"] == {"evidence_ready_score": 1}
    assert interface["row_receipt"]["inline_row_count"] == 511
    assert interface["validation_receipt"]["required_passed"] is True
    assert interface["source_sha256"].startswith("sha256:")

    static_audit = evidence["exp7508-static-audit"]
    assert static_audit["evidence_state"] == "invalid"
    assert static_audit["verdict_class"] == "disqualified"
    assert static_audit["original_verdict_class"] == "disqualified"
    assert static_audit["flagged_adversarial"] is True
    assert static_audit["validation_receipt"]["required_passed"] is False
    assert static_audit["ready_value_fields"]["static_audit_complete_score"] == 1
    assert static_audit["ready_value_fields"]["static_claims_qualified_score"] == 0

    service = evidence["exp7514-service-trace"]
    assert service["model_invoked"] is True
    assert service["ready_value_fields"]["durable_service_claim_ready_score"] == 1
    assert service["inference_substrate_class"] == "model_load_no_generation"


def test_terminal_precedence_reserves_partial_for_owned_work() -> None:
    """SCENARIO-REPORT-7515-COMPLETION: external states never become partial."""

    valid = [{"evidence_state": "terminal", "verdict_class": "null"}]
    blocked = [*valid, {"evidence_state": "missing", "verdict_class": "blocked"}]
    invalid = [*blocked, {"evidence_state": "invalid", "verdict_class": "disqualified"}]
    assert capstone.classify_terminal(valid, current_validation_complete=True)["verdict_class"] == (
        "null"
    )
    assert (
        capstone.classify_terminal(blocked, current_validation_complete=True)["verdict_class"]
        == "blocked"
    )
    assert (
        capstone.classify_terminal(invalid, current_validation_complete=True)["verdict_class"]
        == "disqualified"
    )
    assert (
        capstone.classify_terminal(valid, current_validation_complete=False)["verdict_class"]
        == "partial"
    )


def test_claim_ledger_separates_static_causal_arc_and_hardware() -> None:
    """SCENARIO-REPORT-7515-CLAIMS: one evidence class cannot support another."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    claims = capstone.build_claim_ledger(evidence)
    by_claim = {row["claim"]: row for row in claims}
    assert by_claim["native_evidence_quality"]["ready_value"] == 1
    assert by_claim["static_probability_value"]["qualified_value"] == 0
    assert by_claim["selective_decision_value"]["qualified_value"] == 0
    assert by_claim["static_probability_value"]["state"] == "disqualified"
    assert by_claim["causal_information"]["qualified_value"] == 0
    assert by_claim["retention"]["ready_value"] == 1
    assert by_claim["arc_reachability"]["ready_value"] == 1
    assert by_claim["arc_opportunity"]["qualified_value"] == 0
    assert by_claim["host_quantization"]["ready_value"] == 1
    assert by_claim["historical_board_continuity"]["ready_value"] == 1
    assert by_claim["current_machine_service_timing"]["ready_value"] == 1
    assert by_claim["current_machine_service_timing"]["current_model_invoked"] is True
    assert by_claim["arc_reachability"]["live_benchmark_result"] is False


def test_prior_failures_keep_four_fields_and_do_not_overretire() -> None:
    """SCENARIO-REPORT-7515-RETIREMENT: only exact repeated failures retire."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    rows = capstone.reduce_prior_failures(contract["tasks"], evidence)
    assert len(rows) == 16
    assert all(row["prior_experiment"] for row in rows)
    assert all(row["prior_honest_verdict"] for row in rows)
    assert all(row["addressed_by"] for row in rows)
    assert all(row["retire_if_same_verdict"] is True for row in rows)
    assert not any(row["retirement_triggered"] for row in rows)
    assert rows[-1]["comparison_state"] == "current_work"
    assert capstone.retirement_rows(rows) == []

    repeated = deepcopy(rows)
    repeated[0]["current_honest_verdict"] = repeated[0]["prior_honest_verdict"]
    repeated[0]["exact_text_match"] = True
    repeated[0]["retirement_triggered"] = True
    retirements = capstone.retirement_rows(repeated)
    assert retirements[0]["retired_task_id"] == "exp7503-contract-methods"
    assert retirements[0]["scope"] == "bounded_mechanism_only"


def test_publication_gate_is_read_only_and_records_actual_outcomes() -> None:
    """SCENARIO-REPORT-7515-COMPLETION: readiness does not publish."""

    result = capstone.evaluate_publication_gates(capstone.REPO_ROOT)
    assert result["exit_code"] == 0
    assert result["paper_ready"] is True
    assert result["unmet_gates"] == []
    assert set(result["gates"]) == {"G1", "G2", "G3", "G4"}
    assert all(row["pass"] is True for row in result["gates"].values())
    assert result["publication_performed"] is False
    assert result["stdout_sha256"].startswith("sha256:")


def test_retrospective_has_thirteen_dispositions_and_three_gap_updates() -> None:
    """REQ-REPORT-7515: the retrospective is complete without promotion."""

    artifact = capstone.build_artifact_for_test()
    retrospective = capstone.build_retrospective(artifact)
    assert len(retrospective["task_dispositions"]) == 13
    assert len(retrospective["prd_gap_updates"]) == 3
    assert retrospective["aggregate_verdict_class"] == "disqualified"
    assert retrospective["capstone_complete_score"] == 1
    text = capstone.retrospective_markdown(retrospective)
    assert "# V657 retrospective" in text
    assert text.count("| exp75") == 13
    assert "Static calibrated decisions" in text
    assert "Causal feedback learning" in text
    assert "ARC service and hardware scope" in text


def test_schema_complete_artifact_closes_accounting_without_laundering() -> None:
    """SCENARIO-REPORT-7515-COMPLETION: complete accounting can be disqualified."""

    artifact = capstone.build_artifact_for_test()
    assert capstone.validate_artifact(artifact, require_terminal=True) == []
    assert len(artifact["task_dispositions"]) == 13
    assert artifact["rows"] == artifact["task_dispositions"]
    assert artifact["capstone_complete_score"] == 1
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["honest_verdict"] == "complete_disqualified_required_v657_evidence"
    assert artifact["flagged_adversarial"] is True
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["publication_gate_results"]["paper_ready"] is True
    assert artifact["publication_performed"] is False
    assert artifact["sample_size_budget"]["planned"] == 13
    assert artifact["sample_size_budget"]["complete"] == 13
    assert artifact["sample_size_budget"]["failed"] == 1
    assert artifact["gate_check_summary"]["first_failure"]["upstream"] == ("exp7508-static-audit")
    assert set(artifact["field_principles"]) == set(artifact) - {
        "field_principles",
        "reproducibility_checksum",
    }


def test_dispositions_retain_original_verdicts_and_exact_ready_values() -> None:
    """SCENARIO-REPORT-7515-INVENTORY: source conclusions stay literal."""

    artifact = capstone.build_artifact_for_test()
    by_task = {row["task_id"]: row for row in artifact["task_dispositions"]}
    assert by_task["exp7507-static-evaluation"]["honest_verdict"] == (
        "complete_null_static_evaluation_exploratory_prior_exposure"
    )
    assert by_task["exp7507-static-evaluation"]["ready_value_fields"] == {
        "selective_decision_value_score": 0,
        "static_evaluation_complete_score": 1,
        "static_probability_value_score": 0,
    }
    assert by_task["exp7508-static-audit"]["excluded_from_positive_aggregate"] is True
    assert (
        by_task["exp7510-causal-audit"]["ready_value_fields"]["causal_claims_qualified_score"] == 1
    )
    current = by_task["exp7515-capstone"]
    assert current["evidence_state"] == "current_work"
    assert current["completed"] is True


def test_validator_rejects_protected_mutations() -> None:
    """SCENARIO-REPORT-7515-E2E: derived evidence fails closed."""

    artifact = capstone.build_artifact_for_test()
    mutations = {
        "milestone": "wrong",
        "honest_verdict": "bad",
        "MODEL_SPECS": ["wrong"],
        "inference_substrate": "wrong",
        "task_dispositions": [],
        "claim_ledger": [],
        "prior_failure_rows": [],
        "retirement_rows": [{}],
        "next_conditions": [],
        "publication_gate_results": {},
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "capstone_complete_score": 0,
        "field_principles": {},
        "reproducibility_checksum": "sha256:" + "0" * 64,
    }
    for field, replacement in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = replacement
        assert capstone.validate_artifact(changed, require_terminal=True), field

    wrong_hash = deepcopy(artifact)
    wrong_hash["source_artifact_hashes"][0]["sha256"] = "sha256:" + "0" * 64
    assert "source_hash_mismatch" in capstone.validate_artifact(wrong_hash)
    assert capstone.validate_artifact([]) == ["artifact_mapping_required"]
    missing = deepcopy(artifact)
    del missing["schema"]
    assert capstone.validate_artifact(missing)[0].startswith("missing_required_field")


def test_validation_plan_is_scoped_and_date_is_frozen(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7515-E2E: affected checks cannot broaden silently."""

    private = tmp_path / "private"
    private.mkdir()
    plan = capstone.build_validation_plan(capstone.REPO_ROOT, private)
    assert capstone.validate_validation_plan(capstone.REPO_ROOT, plan) == []
    command_text = "\n".join(" ".join(command.argv) for command in plan)
    assert capstone.TEST_PATH.as_posix() in command_text
    assert "tests/python " not in command_text
    assert capstone.date_argument(capstone.RUN_DATE) == capstone.RUN_DATE
    with pytest.raises(ValueError, match="run date"):
        capstone.date_argument("20260921")


def test_fail_closed_source_reader_and_prior_shape(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7515-INVENTORY: malformed or absent bytes stay blocked."""

    task = {"id": "exp9001-example", "deliverable": "results/example.json"}
    missing = capstone.load_producer(tmp_path, task)
    assert missing["evidence_state"] == "missing"
    assert missing["verdict_class"] == "blocked"
    assert missing["gate_check_summary"]["observed"] is False

    path = tmp_path / "results/example.json"
    path.parent.mkdir()
    path.write_text("not json")
    invalid = capstone.load_producer(tmp_path, task)
    assert invalid["evidence_state"] == "invalid"
    assert invalid["verdict_class"] == "disqualified"

    with pytest.raises(ValueError, match="at least one prior failure"):
        capstone.reduce_prior_failures([{"id": "exp1", "prior_failures": []}], {})


def test_independent_reduction_requires_terminal_receipts() -> None:
    """SCENARIO-REPORT-7515-E2E: cold replay checks final command receipts."""

    artifact = capstone.build_artifact_for_test()
    assert capstone.independent_reduce(artifact) == []
    no_terminal = deepcopy(artifact)
    no_terminal["validation_receipts"] = [
        row
        for row in no_terminal["validation_receipts"]
        if row["name"] not in capstone.TERMINAL_CHECK_NAMES
    ]
    errors = capstone.validate_artifact(no_terminal, require_terminal=True)
    assert "terminal_validation_incomplete" in errors


def test_helper_boundaries_do_not_accept_ambiguous_values() -> None:
    """REQ-REPORT-7515: helper boundaries reject ambiguous evidence."""

    assert capstone._receipt_set_passed({}, capstone.REQUIRED_CHECK_NAMES) is False
    assert capstone._source_hashes_match({}, capstone.REPO_ROOT) is False
    assert capstone._source_hashes_match(
        {"source_artifact_hashes": ["bad"]}, capstone.REPO_ROOT
    ) is (False)
    assert capstone._ready_value_fields({"not_ready_text": "x", "ready": True}) == {}
    gate = capstone._gate_summary([])
    assert gate == {"passed": True, "failed_count": 0, "first_failure": None, "failed_checks": []}

    malformed = deepcopy(capstone.build_artifact_for_test())
    malformed["task_dispositions"][0]["task_id"] = "wrong"
    assert "task_dispositions_invalid" in capstone.validate_artifact(malformed)


def test_defensive_authority_and_publication_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7515-CONTRACT: malformed readers fail closed."""

    design = tmp_path / "design.md"
    design.write_text("design")
    selected = tmp_path / "roadmap.yaml"
    selected.write_text("milestone: test")
    monkeypatch.setattr(capstone, "DESIGN_PATH", Path("design.md"))
    monkeypatch.setattr(
        capstone,
        "resolve_v657_roadmap",
        lambda _root: (selected, {"milestone": capstone.MILESTONE, "tasks": {}}, []),
    )
    monkeypatch.setattr(capstone, "compare_contract_authorities", lambda _text, _roadmap: {})
    with pytest.raises(ValueError, match="task list"):
        capstone.load_contract(tmp_path)

    capstone.evaluate_publication_gates.cache_clear()
    monkeypatch.setattr(
        capstone.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "not-json", "reader error"),
    )
    failed = capstone.evaluate_publication_gates(tmp_path)
    assert failed["unmet_gates"] == ["reader_failed"]
    assert failed["exit_code"] == 1
    capstone.evaluate_publication_gates.cache_clear()


def test_defensive_receipt_flag_and_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7515-INVENTORY: ambiguous producer state remains explicit."""

    assert capstone._validation_rows({}) == []
    assert capstone._gate_rows({}) == []
    assert capstone._producer_receipts_pass({}) is False
    flagged = capstone._first_producer_failure(
        "exp1",
        Path("results/one.json"),
        {"validation_receipts": [], "flagged_adversarial": True},
        {},
    )
    assert flagged["check"] == "producer_adversarial_flag"
    disqualified = capstone._first_producer_failure(
        "exp2",
        Path("results/two.json"),
        {"validation_receipts": [], "flagged_adversarial": False, "verdict_class": "disqualified"},
        {"failures": ["invalid"]},
    )
    assert disqualified["check"] == "producer_validity"

    prior = {
        "experiment_id": "exp0",
        "verdict": "FAIL",
        "addressed_by": "changed mechanism",
        "retire_if_same_verdict": True,
    }
    rows = capstone.reduce_prior_failures(
        [{"id": "exp1", "prior_failures": [prior]}],
        {
            "exp1": {
                "evidence_state": "missing",
                "honest_verdict": "blocked_missing_declared_producer_evidence",
                "verdict_class": "blocked",
            }
        },
    )
    assert rows[0]["comparison_state"] == "external_absence"

    failures = capstone.failure_rows(
        {"comparison_passed": False, "selected_roadmap_path": "roadmap.yaml"},
        {
            "exp1": {
                "evidence_state": "terminal",
                "verdict_class": "blocked",
                "task_id": "exp1",
                "artifact_path": "results/one.json",
            }
        },
        {"required_checks_passed": False, "terminal_validation_passed": False},
    )
    assert [row["check"] for row in failures] == [
        "contract_authorities_agree",
        "producer_terminal_disposition",
        "current_required_checks_passed",
        "current_terminal_validation_passed",
    ]

    monkeypatch.setattr(capstone, "RETROSPECTIVE_PATH", Path("CODEX.md"))
    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    hashes = capstone._source_hashes(capstone.REPO_ROOT, contract, evidence)
    assert sum(row["path"] == "CODEX.md" for row in hashes) == 2
    assert capstone._source_hashes_match(
        {"source_artifact_hashes": [{"path": "missing", "sha256": None}]}, capstone.REPO_ROOT
    )


def test_validator_reports_independent_reduction_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7515-E2E: a reducer exception cannot validate bytes."""

    artifact = capstone.build_artifact_for_test()
    monkeypatch.setattr(
        capstone, "load_contract", lambda _root: (_ for _ in ()).throw(ValueError())
    )
    assert "independent_reduction_failed" in capstone.validate_artifact(artifact)
