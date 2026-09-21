"""Tests for REQ-REPORT-7474 and SCENARIO-REPORT-7474-*.

The tests use current immutable V654 artifacts for scientific reductions. Small
temporary fixtures test missing, invalid, and conductor pre-gate boundaries.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7474_v654_capstone as capstone


def test_contract_resolves_exact_fourteen_tasks_and_rejects_drift() -> None:
    """SCENARIO-REPORT-7474-CONTRACT: both authorities need exact rows."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    assert contract["comparison_passed"] is True
    assert [row["id"] for row in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)
    assert contract["selected_roadmap_path"] == "research-roadmap.yaml"

    roadmap = capstone.load_yaml_mapping(capstone.REPO_ROOT / capstone.ROADMAP_PATH)
    markdown = (capstone.REPO_ROOT / capstone.DESIGN_PATH).read_text()
    changed = deepcopy(roadmap)
    changed["tasks"][4]["deliverable"] = "results/wrong.json"
    comparison = capstone.compare_contract_authorities(markdown, changed)
    assert comparison["comparison_passed"] is False
    assert "row_mismatch" in comparison["errors"]


def test_evidence_slots_keep_terminal_pre_gate_missing_and_invalid_states(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7474-DISPOSITIONS: no evidence state is manufactured."""

    task = {
        "id": "exp9001-example",
        "deliverable": "results/experiment_9001_v654_example.json",
    }
    results = tmp_path / "results"
    results.mkdir()
    missing = capstone.load_evidence_slot(tmp_path, task)
    assert missing["evidence_state"] == "missing"
    assert missing["verdict_class"] == "blocked"

    upstream = results / "upstream.json"
    upstream.write_text("{}")
    pre_gate = results / "experiment_9001_example.json"
    pre_gate.write_text(
        json.dumps(
            {
                "experiment": 9001,
                "schema": "blocked_gate_check_v1",
                "status": "blocked",
                "honest_verdict": "blocked_gate_check_failed",
                "failed_upstream": "exp9000-source",
                "failed_field": "ready_score",
                "failed_operator": "==",
                "failed_expected": 1,
                "failed_observed": 0,
                "failed_evidence_path": str(upstream),
                "failed_evidence_sha256": capstone.sha256_file(upstream),
                "gates_evaluated": [],
                "blocked_at_layer": "conductor_pre_gate",
            }
        )
    )
    blocked = capstone.load_evidence_slot(tmp_path, task)
    assert blocked["evidence_state"] == "pre_gate"
    assert blocked["authenticated"] is True
    assert blocked["found_path"] == "results/experiment_9001_example.json"

    pre_gate.unlink()
    declared = tmp_path / task["deliverable"]
    declared.write_text(
        json.dumps(
            {
                "experiment_id": "exp9002-wrong",
                "milestone": capstone.MILESTONE,
                "status": "complete_null",
                "honest_verdict": "complete_null_wrong_identity",
                "verdict_class": "null",
                "flagged_adversarial": False,
                "rows": [],
                "validation_receipts": [],
            }
        )
    )
    invalid = capstone.load_evidence_slot(tmp_path, task)
    assert invalid["evidence_state"] == "invalid"
    assert invalid["verdict_class"] == "disqualified"


def test_real_dispositions_preserve_invalid_blocked_and_missing_slots() -> None:
    """SCENARIO-REPORT-7474-DISPOSITIONS: all thirteen inputs stay distinct."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    assert evidence["exp7462-option-protocol"]["verdict_class"] == "disqualified"
    assert evidence["exp7462-option-protocol"]["flagged_adversarial"] is True
    assert evidence["exp7465-source-option-capture"]["evidence_state"] == "pre_gate"
    assert evidence["exp7466-typed-energy-calibration"]["evidence_state"] == "missing"
    assert evidence["exp7469-continuous-residual-learning"]["evidence_state"] == "missing"
    assert evidence["exp7472-prefix-service"]["evidence_state"] == "missing"
    board = evidence["exp7473-board-continuity"]
    assert board["evidence_state"] == "terminal"
    assert board["valid"] is True
    assert board["validation_receipts_authenticated"] is False
    assert "validation_log_missing:12" in board["validation_authentication_failures"]
    assert len(evidence) == 13


def test_science_reduction_keeps_probability_cost_extraction_and_arc_separate() -> None:
    """SCENARIO-REPORT-7474-BRANCHES: unrelated claims cannot certify each other."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    reduced = capstone.reduce_science(evidence)
    assert reduced["probability_evidence"]["local_runtime_parity"] is False
    assert reduced["probability_evidence"]["scored_runtime_parity"] is False
    assert reduced["typed_decision_utility"]["availability"] == "missing"
    assert reduced["online_retention_and_benefit"]["updates_replayed"] == 0
    assert reduced["extraction_coverage"]["planned"] == 108
    assert reduced["extraction_coverage"]["unstarted"] == 96
    assert reduced["decision_cost"]["replaceable_share_upper"] > 0.89
    assert reduced["decision_cost"]["efficacy_established"] is False
    assert reduced["arc_self_discovery"]["current_live_episode_count"] == 8
    assert reduced["arc_self_discovery"]["hidden_game_efficacy_claim"] is False
    assert reduced["external_text_verifier_moat"]["reopened"] is False


def test_terminal_precedence_never_uses_partial_for_external_absence() -> None:
    """REQ-REPORT-7474: invalid, missing, null, and owned partial are distinct."""

    rows = [{"valid": True, "verdict_class": "null", "evidence_state": "terminal"}]
    assert capstone.classify_terminal(rows, validation_complete=True)["verdict_class"] == "null"
    absent = [*rows, {"valid": False, "verdict_class": "blocked", "evidence_state": "missing"}]
    assert (
        capstone.classify_terminal(absent, validation_complete=True)["verdict_class"] == "blocked"
    )
    invalid = [
        *absent,
        {"valid": False, "verdict_class": "disqualified", "evidence_state": "invalid"},
    ]
    assert (
        capstone.classify_terminal(invalid, validation_complete=True)["verdict_class"]
        == "disqualified"
    )
    assert capstone.classify_terminal(rows, validation_complete=False)["verdict_class"] == "partial"


def test_continuations_scope_retirement_and_preserve_external_obligations() -> None:
    """SCENARIO-REPORT-7474-RETIREMENT: only the measured construction retires."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    science = capstone.reduce_science(evidence)
    rows = capstone.continuation_rows(evidence, science)
    by_branch = {row["branch"]: row for row in rows}
    assert by_branch["four_expert_mixture"]["decision"] == "retire"
    assert by_branch["four_expert_mixture"]["scope"] == "unchanged four-expert mixture only"
    assert by_branch["typed_decision"]["decision"] == "defer"
    assert by_branch["arc_self_discovery"]["decision"] == "continue"
    assert by_branch["board_continuity"]["decision"] == "defer"
    assert all(row["changed_prerequisite"] for row in rows)

    retirements = capstone.retirement_rows(contract["tasks"], evidence)
    assert all(
        {"task_id", "prior_experiment_id", "previous_verdict", "addressed_by"} <= set(row)
        for row in retirements
    )
    assert capstone.unresolved_obligations()[-1]["state"] == "user_forbidden"


def test_schema_complete_artifact_has_fourteen_rows_and_fover_only_gate() -> None:
    """SCENARIO-REPORT-7474-ARTIFACT: completion does not certify V654 science."""

    artifact = capstone.build_artifact_for_test()
    assert capstone.validate_artifact(artifact, require_terminal=True) == []
    assert len(artifact["task_dispositions"]) == 14
    assert artifact["task_dispositions"] == artifact["rows"]
    assert artifact["capstone_complete_score"] == 1
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["flagged_adversarial"] is True
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["publication_gates"]["certifies_v654"] is False
    assert artifact["field_principles"]["task_dispositions"]


def test_cold_validator_rejects_identity_rows_source_and_checksum_mutations() -> None:
    """SCENARIO-REPORT-7474-ARTIFACT: protected evidence fails closed."""

    artifact = capstone.build_artifact_for_test()
    cases = []
    for field, value in (("milestone", "wrong"), ("MODEL_SPECS", ["wrong"])):
        changed = deepcopy(artifact)
        changed[field] = value
        cases.append(changed)
    changed = deepcopy(artifact)
    changed["task_dispositions"][0]["verdict_class"] = "positive"
    changed["rows"] = deepcopy(changed["task_dispositions"])
    cases.append(changed)
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"][0]["sha256"] = "sha256:" + "0" * 64
    cases.append(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    cases.append(changed)
    assert all(capstone.validate_artifact(case, require_terminal=True) for case in cases)


def test_validation_plan_is_scoped_and_date_is_frozen(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7474-ARTIFACT: validation names only affected files."""

    private = tmp_path / "private"
    private.mkdir()
    plan = capstone.build_validation_plan(capstone.REPO_ROOT, private)
    assert capstone.validate_validation_plan(capstone.REPO_ROOT, plan) == []
    commands = "\n".join(" ".join(command.argv) for command in plan)
    assert capstone.TEST_PATH.as_posix() in commands
    assert "tests/python " not in commands
    assert capstone.date_argument(capstone.RUN_DATE) == capstone.RUN_DATE
    with pytest.raises(ValueError, match="run date"):
        capstone.date_argument("20260920")


def test_contract_parser_and_resolution_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7474-CONTRACT: malformed and drifted authorities fail."""

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("- not-a-mapping\n")
    with pytest.raises(ValueError, match="mapping"):
        capstone.load_yaml_mapping(invalid_yaml)
    assert capstone.compare_contract_authorities("not a contract", {})["comparison_passed"] is False

    roadmap = capstone.load_yaml_mapping(capstone.REPO_ROOT / capstone.ROADMAP_PATH)
    markdown = (capstone.REPO_ROOT / capstone.DESIGN_PATH).read_text()
    cases = []
    wrong_milestone = markdown.replace("2026.09.654", "2026.09.999", 1)
    cases.append(capstone.compare_contract_authorities(wrong_milestone, roadmap))
    wrong_markdown_id = markdown.replace("exp7461-contract-methods", "exp9999-wrong", 1)
    cases.append(capstone.compare_contract_authorities(wrong_markdown_id, roadmap))
    changed = deepcopy(roadmap)
    changed["milestone"] = "2026.09.999"
    cases.append(capstone.compare_contract_authorities(markdown, changed))
    changed = deepcopy(roadmap)
    changed["tasks"][0]["id"] = "exp9999-wrong"
    cases.append(capstone.compare_contract_authorities(markdown, changed))
    changed = deepcopy(roadmap)
    changed["tasks"] = changed["tasks"][:-1]
    cases.append(capstone.compare_contract_authorities(markdown, changed))
    assert all(case["comparison_passed"] is False for case in cases)

    root = tmp_path / "no-milestone"
    (root / capstone.DESIGN_PATH.parent).mkdir(parents=True)
    (root / capstone.DESIGN_PATH).write_text(markdown)
    (root / capstone.ROADMAP_PATH).write_text("milestone: wrong\ntasks: []\n")
    fallback = capstone.load_contract(root)
    assert fallback["comparison_passed"] is False
    (root / capstone.ROADMAP_PATH).write_text("milestone: wrong\ntasks: nope\n")
    with pytest.raises(ValueError, match="task list"):
        capstone.load_contract(root)
    (root / capstone.NEXT_ROADMAP_PATH).write_text(
        (capstone.REPO_ROOT / capstone.ROADMAP_PATH).read_text()
    )
    selected_next = capstone.load_contract(root)
    assert selected_next["selected_roadmap_path"] == capstone.NEXT_ROADMAP_PATH.as_posix()
    monkeypatch.setattr(capstone, "SOURCE_PATHS", ())
    monkeypatch.setattr(capstone, "MODULE_PATH", capstone.NEXT_ROADMAP_PATH)
    monkeypatch.setattr(capstone, "WRAPPER_PATH", capstone.NEXT_ROADMAP_PATH)
    monkeypatch.setattr(capstone, "TEST_PATH", capstone.NEXT_ROADMAP_PATH)
    assert any(
        row["evidence_class"] == "selected_contract_authority"
        for row in capstone._source_hashes(root, selected_next, {})
    )


def test_small_helper_fail_closed_branches(tmp_path: Path) -> None:
    """REQ-REPORT-7474: helper shape errors cannot become usable evidence."""

    assert capstone._validation_rows({}) == []
    assert capstone._audit_branch({}, "typed_decision") == {}
    assert capstone._receipt_set_passed({}, capstone.REQUIRED_CHECK_NAMES) is False
    assert capstone._source_hashes_match({}, tmp_path) is False
    assert capstone._source_hashes_match({"source_artifact_hashes": ["bad"]}, tmp_path) is False

    source = tmp_path / "source.json"
    source.write_text("{}")
    task = {"id": "exp9003-relative", "deliverable": "results/unused.json"}
    payload = {
        "experiment": 9003,
        "blocked_at_layer": "conductor_pre_gate",
        "status": "blocked",
        "failed_upstream": "exp9002-source",
        "failed_field": "ready",
        "failed_operator": "==",
        "failed_expected": 1,
        "failed_observed": 0,
        "failed_evidence_path": "source.json",
        "failed_evidence_sha256": capstone.sha256_file(source),
    }
    pre_gate_path = tmp_path / "results/pre.json"
    pre_gate_path.parent.mkdir()
    pre_gate_path.write_text("{}")
    row = capstone._pre_gate_evidence(tmp_path, task, Path("results/pre.json"), payload)
    assert row["authenticated"] is True


def test_terminal_and_failure_control_branches() -> None:
    """REQ-REPORT-7474: controls cover benefit and current-validity branches."""

    circular = [{"valid": True, "verdict_class": "circular_positive", "evidence_state": "terminal"}]
    terminal = capstone.classify_terminal(circular, validation_complete=True)
    assert terminal["verdict_class"] == "null"
    contract = {
        "comparison_passed": False,
        "selected_roadmap_path": capstone.ROADMAP_PATH.as_posix(),
    }
    failures = capstone._failure_rows(contract, {}, False)
    assert [row["category"] for row in failures] == ["current_validity", "current_validity"]

    task = {
        "id": "exp1-control",
        "prior_failures": ["malformed", {"experiment_id": "exp0", "verdict": "complete_null"}],
    }
    retirements = capstone.retirement_rows([task], {})
    assert len(retirements) == 1


def test_validator_reports_each_protected_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7474-ARTIFACT: every protected derived field fails closed."""

    artifact = capstone.build_artifact_for_test()
    assert capstone.independent_reduce(artifact) == []
    assert capstone.validate_artifact([]) == ["artifact_mapping_required"]
    missing = deepcopy(artifact)
    del missing["schema"]
    assert capstone.validate_artifact(missing)[0].startswith("missing_required_field")

    mutations = {
        "status": "bad",
        "inference_substrate": "model",
        "promotion_score": 1,
        "science_reductions": {},
        "continuation_rows": [],
        "retirement_rows": [],
        "unresolved_obligations": [],
        "gate_check_summary": {},
        "acceptance_gate_results": [],
        "publication_gates": {},
        "capstone_complete_score": 0,
        "flagged_adversarial": False,
        "field_principles": {},
    }
    for field, replacement in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = replacement
        assert capstone.validate_artifact(changed), field

    wrong_rows = deepcopy(artifact)
    wrong_rows["task_dispositions"] = wrong_rows["task_dispositions"][:-1]
    assert "task_dispositions_invalid" in capstone.validate_artifact(wrong_rows)

    incomplete = deepcopy(artifact)
    incomplete["validation_receipts"] = [
        row for row in incomplete["validation_receipts"] if row["name"] != "publication_gate_json"
    ]
    incomplete["task_dispositions"][-1]["authenticated"] = False
    incomplete["task_dispositions"][-1]["valid"] = False
    incomplete["task_dispositions"][-1]["completed"] = False
    incomplete["task_dispositions"][-1]["failed"] = True
    incomplete["rows"] = deepcopy(incomplete["task_dispositions"])
    incomplete["capstone_complete_score"] = 0
    incomplete["reproducibility_checksum"] = capstone.reproducibility_checksum(incomplete)
    assert "terminal_validation_incomplete" in capstone.validate_artifact(incomplete)

    monkeypatch.setattr(
        capstone, "load_contract", lambda _root: (_ for _ in ()).throw(ValueError())
    )
    assert "independent_reduction_failed" in capstone.validate_artifact(artifact)
