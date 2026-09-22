"""Tests for the V658 fourteen-disposition capstone.

Spec refs: REQ-REPORT-7529 and SCENARIO-REPORT-7529-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7529_v658_capstone as capstone


def test_contract_and_inventory_preserve_fourteen_exact_rows() -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-CONTRACT/INVENTORY."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    assert contract["comparison_passed"] is True
    assert [row["id"] for row in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)

    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    assert list(evidence) == list(capstone.EXPECTED_TASK_IDS[:-1])
    assert {task_id for task_id, row in evidence.items() if row["evidence_state"] == "missing"} == {
        "exp7518-source-pilot",
        "exp7519-source-fit-capture",
        "exp7520-source-eval-capture",
        "exp7521-consistency-energy",
        "exp7522-source-evaluation",
        "exp7523-count-memory",
        "exp7524-count-online",
    }
    assert evidence["exp7517-source-protocol"]["verdict_class"] == "blocked"
    assert (
        evidence["exp7526-arc-eligibility"]["ready_value_fields"]["eligibility_receipt_ready_score"]
        == 1
    )
    assert (
        evidence["exp7528-service-boundary"]["ready_value_fields"][
            "board_continuity_complete_score"
        ]
        == 1
    )


def test_authority_mutations_fail_closed() -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-CONTRACT."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    markdown = (capstone.REPO_ROOT / capstone.DESIGN_PATH).read_text()
    for mutate in (
        lambda road: road["tasks"].pop(),
        lambda road: road["tasks"].__setitem__(0, {**road["tasks"][0], "phase": 9}),
        lambda road: road["tasks"].__setitem__(1, {**road["tasks"][1], "title": "changed"}),
        lambda road: road["tasks"].__setitem__(2, {**road["tasks"][2], "gated_on": []}),
    ):
        roadmap = deepcopy(contract["roadmap"])
        mutate(roadmap)
        assert capstone.compare_authorities(markdown, roadmap)["passed"] is False


def test_missing_and_malformed_producers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-INVENTORY."""

    task = {"id": "exp7518-source-pilot", "deliverable": "results/pilot.json"}
    missing = capstone.load_producer(tmp_path, task)
    assert missing["evidence_state"] == "missing"
    assert missing["verdict_class"] == "blocked"
    assert missing["gate_check_summary"] == {
        "check": "producer_artifact_exists",
        "upstream": "exp7518-source-pilot",
        "path": "results/pilot.json",
        "field": "path",
        "op": "exists",
        "expected": True,
        "observed": False,
        "passed": False,
    }

    path = tmp_path / "results/pilot.json"
    path.parent.mkdir()
    path.write_text("[]")
    malformed = capstone.load_producer(tmp_path, task)
    assert malformed["evidence_state"] == "invalid"
    assert malformed["verdict_class"] == "disqualified"
    assert malformed["gate_check_summary"]["field"] == "json_object"


def test_claims_keep_audit_opportunity_and_cost_independent() -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-CLAIMS."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    claims = {row["claim"]: row for row in capstone.build_claim_ledger(evidence)}

    assert claims["static_probability_value"]["source_task_id"] == "exp7525-decision-audit"
    assert claims["static_probability_value"]["qualified_value"] == 0
    assert claims["causal_online_learning"]["source_task_id"] == "exp7525-decision-audit"
    assert claims["causal_online_learning"]["qualified_value"] == 0
    assert claims["live_agent_opportunity"]["source_task_id"] == "exp7527-arc-opportunities"
    assert claims["live_agent_opportunity"]["qualified_value"] == 0
    assert claims["exact_cpu_operation_cost"]["source_task_id"] == "exp7528-service-boundary"
    assert claims["exact_cpu_operation_cost"]["ready_value"] == 0
    assert claims["dated_board_continuity"]["ready_value"] == 1
    assert claims["dated_board_continuity"]["positive_scientific_claim"] is False


def test_prior_failures_inspect_literal_and_substantive_states() -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-RETIREMENT."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    rows = capstone.reduce_prior_failures(contract["tasks"], evidence)

    assert len(rows) == 15
    assert all(row["retirement_triggered"] is False for row in rows)
    assert capstone.retirement_rows(rows) == []
    gpu = next(
        row
        for row in rows
        if row["task_id"] == "exp7527-arc-opportunities"
        and row["prior_experiment"] == "exp7512-arc-opportunity"
    )
    assert gpu["current_substantive_state"] == "environmental_owned_gpu_absence"
    assert gpu["substantive_repeat"] is False

    repeated = deepcopy(rows[0])
    repeated.update(
        exact_text_match=True,
        substantive_repeat=True,
        environmental_absence=False,
        retirement_triggered=True,
    )
    retired = capstone.retirement_rows([repeated])
    assert retired[0]["scope"] == "bounded_scientific_mechanism_only"


@pytest.mark.parametrize(
    ("affected", "terminal", "state", "verdict"),
    [
        (True, False, "partial", "partial_retryable_current_capstone_validation_unfinished"),
        (False, True, "disqualified", "complete_disqualified_required_validation"),
    ],
)
def test_current_validation_precedes_scientific_classification(
    affected: bool, terminal: bool, state: str, verdict: str
) -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-E2E."""

    source = {"evidence_state": "terminal", "verdict_class": "null"}
    result = capstone.classify_terminal(
        [source], affected_complete=affected, terminal_complete=terminal
    )
    assert result == {"verdict_class": state, "honest_verdict": verdict, "status": verdict}


def test_scientific_classification_precedence() -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-INVENTORY."""

    valid = {"evidence_state": "terminal", "verdict_class": "null"}
    blocked = {"evidence_state": "terminal", "verdict_class": "blocked"}
    missing = {"evidence_state": "missing", "verdict_class": "blocked"}
    invalid = {"evidence_state": "invalid", "verdict_class": "disqualified"}
    classify = capstone.classify_terminal

    assert (
        classify([valid], affected_complete=True, terminal_complete=True)["verdict_class"] == "null"
    )
    assert (
        classify([valid, blocked], affected_complete=True, terminal_complete=True)["verdict_class"]
        == "blocked"
    )
    assert (
        classify([valid, missing], affected_complete=True, terminal_complete=True)["verdict_class"]
        == "blocked"
    )
    assert (
        classify([blocked, invalid], affected_complete=True, terminal_complete=True)[
            "verdict_class"
        ]
        == "disqualified"
    )


def test_terminal_artifact_has_complete_accounting_and_blocked_science() -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-INVENTORY/CLAIMS/E2E."""

    artifact = capstone.build_artifact_for_test()
    assert artifact["schema"] == "carnot.exp7529.v658.capstone.v1"
    assert artifact["run_date"] == "20260922"
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == capstone.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["execution_venue"] == "host"
    assert artifact["execution_venue_detail"] == "host_cpu"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["capstone_complete_score"] == 1
    assert len(artifact["task_dispositions"]) == 14
    assert artifact["task_dispositions"][-1]["evidence_state"] == "current_terminal"
    assert artifact["sample_size_budget"] == {
        "independent_unit": "ordered_v658_task_disposition",
        "planned": 14,
        "attempted": 7,
        "completed": 7,
        "excluded": 12,
        "failed": 0,
        "censored": 7,
        "unstarted": 7,
    }
    assert artifact["gate_check_summary"]["first_failure"]["path"] == (
        "results/raw/experiment_7517_v658_source_protocol/exposure_inventory.json"
        "#/fresh_eligible_groups"
    )
    assert capstone.validate_artifact(artifact) == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value.update(model_invoked=True), "model_contract_invalid"),
        (
            lambda value: value.update(inference_substrate_class="no_model_load"),
            "substrate_invalid",
        ),
        (
            lambda value: value["task_dispositions"].pop(),
            "task_dispositions_invalid",
        ),
        (
            lambda value: value["claim_ledger"][0].update(qualified_value=1),
            "claim_ledger_invalid",
        ),
        (
            lambda value: value["retirement_rows"].append({"retired_task_id": "fake"}),
            "retirement_rows_invalid",
        ),
        (
            lambda value: value["gate_check_summary"].update(passed=True),
            "terminal_reduction_invalid",
        ),
        (
            lambda value: value.update(capstone_complete_score=True),
            "capstone_score_invalid",
        ),
        (
            lambda value: value["source_artifact_hashes"][0].update(sha256="sha256:bad"),
            "source_hash_mismatch",
        ),
        (
            lambda value: value["field_principles"].pop("schema"),
            "field_principles_invalid",
        ),
        (
            lambda value: value.update(reproducibility_checksum="sha256:bad"),
            "reproducibility_checksum_invalid",
        ),
    ],
)
def test_protected_artifact_mutations_fail(mutation: object, expected: str) -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-E2E."""

    artifact = capstone.build_artifact_for_test()
    assert callable(mutation)
    mutation(artifact)  # type: ignore[operator]
    assert expected in capstone.validate_artifact(artifact)


def test_missing_fields_and_non_mapping_fail_closed() -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-E2E."""

    assert capstone.validate_artifact([]) == ["artifact_mapping_required"]
    artifact = capstone.build_artifact_for_test()
    artifact.pop("schema")
    assert capstone.validate_artifact(artifact) == ["missing_required_field:schema"]


def test_running_self_row_never_reads_future_artifact() -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-CONTRACT."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    terminal = capstone.classify_terminal(
        list(evidence.values()), affected_complete=True, terminal_complete=False
    )
    rows = capstone.task_dispositions(
        contract["tasks"], evidence, terminal, current_validation_complete=False
    )
    assert rows[-1]["artifact_path"] is None
    assert rows[-1]["artifact_sha256"] is None
    assert rows[-1]["evidence_state"] == "current_running"
    assert rows[-1]["completed"] is False


def test_validation_plan_is_private_scoped_and_complete(tmp_path: Path) -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-E2E."""

    private = tmp_path / "private"
    private.mkdir()
    commands = capstone.build_validation_plan(capstone.REPO_ROOT, private)
    assert capstone.validate_validation_plan(capstone.REPO_ROOT, commands) == []
    assert [row.name for row in commands] == list(capstone.REQUIRED_CHECK_NAMES)
    focused = next(row for row in commands if row.name == "focused_pytest")
    assert "-n" in focused.argv and "0" in focused.argv
    assert "--no-cov" in focused.argv
    assert any(str(private) in arg for arg in focused.argv)

    changed = list(commands)
    changed[1] = capstone.validation_scope.CommandSpec(
        "focused_pytest", ("pytest", "tests/python"), "too_broad"
    )
    assert capstone.validate_validation_plan(capstone.REPO_ROOT, changed)


def test_retrospective_records_three_gaps_and_reopen_conditions() -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-CLAIMS/RETIREMENT."""

    artifact = capstone.build_artifact_for_test()
    retrospective = capstone.build_retrospective(artifact)
    assert len(retrospective["prd_gap_updates"]) == 3
    assert retrospective["wall_time_phase_budget"]["current_duration_s"] == 0.0
    text = capstone.retrospective_markdown(retrospective)
    assert "# V658 capstone" in text
    assert "Probability and causal learning" in text
    assert "Live-agent opportunity" in text
    assert "Service cost and board continuity" in text
    assert "Publication and roadmap activation remain operator actions." in text


def test_date_and_checksum_contracts() -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-E2E."""

    assert capstone.date_argument("20260922") == "20260922"
    with pytest.raises(ValueError, match="20260922"):
        capstone.date_argument("20260923")
    artifact = capstone.build_artifact_for_test()
    assert capstone.reproducibility_checksum(artifact) == artifact["reproducibility_checksum"]


def test_result_json_shape_if_present() -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-E2E."""

    path = capstone.REPO_ROOT / capstone.RESULT_PATH
    if path.exists():
        value = json.loads(path.read_text())
        assert capstone.validate_artifact(value, require_terminal=True) == []


def test_defensive_contract_and_receipt_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-CONTRACT/INVENTORY."""

    monkeypatch.setattr(
        capstone,
        "resolve_v658_roadmap",
        lambda root: (root / "research-roadmap.yaml", {"tasks": None}, []),
    )
    monkeypatch.setattr(capstone, "compare_authorities", lambda text, road: {"passed": True})
    with pytest.raises(ValueError, match="task list"):
        capstone.load_contract(capstone.REPO_ROOT)

    assert capstone._receipt_rows({}) == []
    assert capstone._required_receipts_pass({}) is False
    assert capstone._acceptance_rows({}) == []
    assert capstone._raw_receipts({}) == []
    assert capstone._receipt_set_passed({}, capstone.REQUIRED_CHECK_NAMES) is False


def test_present_producer_failure_causes_are_exact() -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-INVENTORY."""

    receipt = {
        "validation_receipts": [
            {
                "name": "strict",
                "required": True,
                "passed": False,
                "exit_code": 2,
                "log_path": "raw/strict.log",
            }
        ]
    }
    failed = capstone._producer_failure("task", "result.json", receipt, {})
    assert failed["field"] == "strict"
    assert failed["observed"] == {"passed": False, "exit_code": 2}

    flagged = capstone._producer_failure("task", "result.json", {"flagged_adversarial": True}, {})
    assert flagged["field"] == "flagged_adversarial"
    invalid = capstone._producer_failure(
        "task", "result.json", {"verdict_class": "disqualified"}, {"failures": ["x"]}
    )
    assert invalid["validation_failures"] == ["x"]


def test_invalid_claims_and_prior_shapes_cannot_promote_or_retire() -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-CLAIMS/RETIREMENT."""

    evidence = {
        "task": {
            "valid": False,
            "flagged_adversarial": True,
            "artifact_path": "result.json",
            "source_sha256": "sha256:x",
            "original_verdict_class": "positive",
            "honest_verdict": "complete_positive_bad",
        }
    }
    claim = capstone._claim(evidence, "task", "bad", qualified_value=1)
    assert claim["positive_aggregate_eligible"] is False

    with pytest.raises(ValueError, match="prior failure"):
        capstone.reduce_prior_failures([{"id": "task"}], {})


def test_failure_rows_cover_contract_current_and_summary_fallback() -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-INVENTORY/E2E."""

    blocked = {
        "task_id": "task",
        "artifact_path": "result.json",
        "expected_path": "result.json",
        "evidence_state": "terminal",
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_x",
        "gate_check_summary": None,
    }
    fallback = capstone._normalized_blocked_failure(blocked)
    assert fallback["field"] == "verdict_class"
    failures = capstone.failure_rows(
        {"comparison_passed": False, "selected_roadmap_path": "roadmap.yaml"},
        {"task": blocked},
        {"required_checks_passed": False, "terminal_validation_passed": False},
    )
    assert [row["check"] for row in failures] == [
        "contract_authorities_agree",
        "producer_terminal_disposition",
        "current_required_checks_passed",
        "current_terminal_validation_passed",
    ]


def test_source_hash_shapes_and_note_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-E2E."""

    assert (
        capstone._source_hashes_match({"source_artifact_hashes": {}}, capstone.REPO_ROOT) is False
    )
    assert (
        capstone._source_hashes_match({"source_artifact_hashes": [1]}, capstone.REPO_ROOT) is False
    )
    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    monkeypatch.setattr(capstone, "NOTE_PATH", capstone.TEST_PATH)
    rows = capstone._source_hashes(capstone.REPO_ROOT, contract, evidence)
    assert any(row["path"] == capstone.TEST_PATH.as_posix() for row in rows)


def test_publication_reader_malformed_json_is_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-E2E."""

    class Result:
        stdout = "not json"
        stderr = "reader error"
        returncode = 3

    capstone.evaluate_publication_gates.cache_clear()
    monkeypatch.setattr(capstone.subprocess, "run", lambda *args, **kwargs: Result())
    result = capstone.evaluate_publication_gates(Path("/tmp"))
    assert result["unmet_gates"] == ["reader_failed"]
    assert result["exit_code"] == 3
    capstone.evaluate_publication_gates.cache_clear()


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value.update(schema="wrong"), "identity_invalid"),
        (
            lambda value: value.update(honest_verdict="blocked_without_prefix"),
            "terminal_identity_invalid",
        ),
        (
            lambda value: value["prior_failure_rows"][0].update(exact_text_match=True),
            "prior_failure_rows_invalid",
        ),
        (lambda value: value["next_conditions"].pop(), "next_conditions_invalid"),
        (
            lambda value: value["acceptance_gate_results"].pop(),
            "acceptance_gates_invalid",
        ),
        (
            lambda value: value["publication_gate_results"].update(
                paper_ready=not value["publication_gate_results"]["paper_ready"]
            ),
            "publication_gate_results_invalid",
        ),
    ],
)
def test_additional_cold_reduction_mutations_fail(mutation: object, expected: str) -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-E2E."""

    artifact = capstone.build_artifact_for_test()
    assert callable(mutation)
    mutation(artifact)  # type: ignore[operator]
    assert expected in capstone.validate_artifact(artifact)


def test_terminal_requirement_and_independent_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7529; SCENARIO-REPORT-7529-E2E."""

    artifact = capstone.build_artifact_for_test()
    artifact["validation_receipts"] = [
        row
        for row in artifact["validation_receipts"]
        if row["name"] not in capstone.TERMINAL_CHECK_NAMES
    ]
    assert "terminal_validation_incomplete" in capstone.validate_artifact(
        artifact, require_terminal=True
    )

    good = capstone.build_artifact_for_test()
    monkeypatch.setattr(
        capstone, "load_contract", lambda root: (_ for _ in ()).throw(ValueError("bad"))
    )
    assert "independent_reduction_failed" in capstone.independent_reduce(good)
