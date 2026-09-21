"""Tests for REQ-REPORT-7488 and SCENARIO-REPORT-7488-*.

The tests use immutable V655 sources for the real reduction. Small fixtures
exercise evidence and interval boundaries without writing repository results.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7488_v655_capstone as capstone


def test_contract_authenticates_fourteen_rows_and_rejects_drift() -> None:
    """SCENARIO-REPORT-7488-CONTRACT: both authorities must match exactly."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    assert contract["comparison_passed"] is True
    assert [task["id"] for task in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)
    assert contract["selected_roadmap_path"] == "research-roadmap.yaml"

    roadmap = capstone.load_yaml_mapping(capstone.REPO_ROOT / capstone.ROADMAP_PATH)
    markdown = (capstone.REPO_ROOT / capstone.DESIGN_PATH).read_text()
    changed = deepcopy(roadmap)
    changed["tasks"][11]["deliverable"] = "results/wrong.json"
    result = capstone.compare_contract_authorities(markdown, changed)
    assert result["comparison_passed"] is False
    assert "row_mismatch" in result["errors"]


def test_real_evidence_keeps_invalid_missing_and_valid_slots_distinct() -> None:
    """SCENARIO-REPORT-7488-DISPOSITIONS: source states cannot be promoted."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    assert len(evidence) == 13
    assert evidence["exp7475-contract-methods"]["evidence_state"] == "invalid"
    assert evidence["exp7475-contract-methods"]["verdict_class"] == "disqualified"
    assert evidence["exp7484-decision-audit"]["required_validation_passed"] is False
    assert evidence["exp7485-arc-cost-panel-a"]["valid"] is True
    assert evidence["exp7486-arc-cost-panel-b"]["evidence_state"] == "missing"
    assert evidence["exp7486-arc-cost-panel-b"]["verdict_class"] == "blocked"


def test_evidence_fixture_authenticates_conductor_pre_gate_and_bad_identity(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7488-DISPOSITIONS: pre-gate bytes retain their cause."""

    task = {"id": "exp9001-example", "deliverable": "results/declared.json"}
    (tmp_path / "results").mkdir()
    assert capstone.load_evidence_slot(tmp_path, task)["evidence_state"] == "missing"

    source = tmp_path / "source.json"
    source.write_text("{}")
    fallback = tmp_path / "results/experiment_9001_example.json"
    fallback.write_text(
        json.dumps(
            {
                "schema": "blocked_gate_check_v1",
                "experiment": 9001,
                "status": "blocked",
                "blocked_at_layer": "conductor_pre_gate",
                "failed_upstream": "exp9000-source",
                "failed_field": "ready_score",
                "failed_operator": "==",
                "failed_expected": 1,
                "failed_observed": 0,
                "failed_evidence_path": str(source),
                "failed_evidence_sha256": capstone.sha256_file(source),
            }
        )
    )
    blocked = capstone.load_evidence_slot(tmp_path, task)
    assert blocked["evidence_state"] == "pre_gate"
    assert blocked["gate_check_summary"]["observed"] == 0

    fallback.unlink()
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
    assert capstone.load_evidence_slot(tmp_path, task)["evidence_state"] == "invalid"


def test_terminal_precedence_reserves_partial_for_owned_unfinished_work() -> None:
    """REQ-REPORT-7488: present invalid evidence outranks external absence."""

    valid = [{"valid": True, "evidence_state": "terminal"}]
    missing = [*valid, {"valid": False, "evidence_state": "missing"}]
    invalid = [*missing, {"valid": False, "evidence_state": "invalid"}]
    assert (
        capstone.classify_terminal(valid, current_validation_complete=True)["verdict_class"]
        == "null"
    )
    assert (
        capstone.classify_terminal(missing, current_validation_complete=True)["verdict_class"]
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


def test_interval_union_and_real_arc_reduction_use_measured_units_only() -> None:
    """SCENARIO-REPORT-7488-ARC: overlap and planned rows cannot inflate E6."""

    assert capstone.interval_union_ns([(0, 10), (4, 12), (20, 25), (25, 26)]) == 18
    assert capstone.interval_union_ns([(4, 4), (8, 2)]) == 0

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    reduced = capstone.reduce_arc_panels(capstone.REPO_ROOT, evidence)
    assert reduced["completed_episode_count"] == 18
    assert reduced["independent_game_count"] == 6
    assert len(reduced["episode_rows"]) == 18
    assert len(reduced["game_cluster_rows"]) == 6
    assert reduced["panels_pooled"] is False
    assert reduced["support_floor_passed"] is False
    assert reduced["scientific_verdict"] == "sample_limited_null"
    assert reduced["actions_to_progress_censored_count"] == 18
    assert reduced["hidden_game_efficacy_tested"] is False
    assert all(row["eligible_replaceable_union_ns"] == 0 for row in reduced["episode_rows"])
    assert reduced["strictly_positive_lower_bound"] is False


def test_science_reduction_separates_invalid_audit_and_service_limit() -> None:
    """SCENARIO-REPORT-7488-SCIENCE: one branch cannot certify another."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    science = capstone.reduce_science(evidence)
    assert science["independent_audit"]["usable_for_positive_aggregate"] is False
    assert science["static_calibration"]["external_group_count"] == 74
    assert science["typed_cost_grid"]["producer_benefit_score"] == 1
    assert science["typed_cost_grid"]["independently_supported"] is False
    assert science["prequential_improvement"]["producer_benefit_score"] == 0
    assert science["retention"]["reported_separately"] is True
    assert science["complete_service_cost"]["complete_denominator"] is False
    assert "adversarial_verify" in science["independent_audit"]["missing_checks"]
    assert science["positive_aggregate"] is False


def test_continuations_keep_retirements_and_external_blocks_scoped() -> None:
    """SCENARIO-REPORT-7488-CONTINUATION: changed evidence controls reopening."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    science = capstone.reduce_science(evidence)
    arc = capstone.reduce_arc_panels(capstone.REPO_ROOT, evidence)
    rows = capstone.continuation_rows(evidence, science, arc)
    by_branch = {row["branch"]: row for row in rows}
    assert by_branch["compact_span_extraction"]["decision"] == "retire"
    assert by_branch["four_expert_mixture"]["decision"] == "retire"
    assert by_branch["general_external_text_reranking"]["decision"] == "retire"
    assert by_branch["arc_cost_and_efficacy"]["decision"] == "defer"
    assert by_branch["gatemate_physical_retry"]["decision"] == "defer"
    assert all(row["measured_cause"] and row["changed_prerequisite"] for row in rows)
    assert all(
        {"task", "prior_experiment", "prior_verdict", "changed_mechanism"} <= set(row)
        for row in capstone.retirement_rows()
    )


def test_schema_complete_artifact_has_fourteen_rows_and_separate_completion() -> None:
    """SCENARIO-REPORT-7488-ARTIFACT: complete reporting is not valid science."""

    artifact = capstone.build_artifact_for_test()
    assert capstone.validate_artifact(artifact, require_terminal=True) == []
    assert len(artifact["task_dispositions"]) == 14
    assert artifact["rows"] == artifact["task_dispositions"]
    assert artifact["capstone_complete_score"] == 1
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["arc_combined_reduction"]["scientific_verdict"] == "sample_limited_null"
    assert artifact["publication_gates"]["certifies_v655"] is False
    assert all(row["principle"] for row in artifact["acceptance_gate_results"])


def test_cold_validator_rejects_identity_rows_arc_and_checksum_mutations() -> None:
    """SCENARIO-REPORT-7488-ARTIFACT: protected reductions fail closed."""

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
    changed["arc_combined_reduction"]["completed_episode_count"] = 36
    cases.append(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    cases.append(changed)
    assert all(capstone.validate_artifact(case, require_terminal=True) for case in cases)


def test_validation_plan_is_scoped_and_date_is_frozen(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7488-ARTIFACT: the manifest cannot broaden validation."""

    private = tmp_path / "private"
    private.mkdir()
    plan = capstone.build_validation_plan(capstone.REPO_ROOT, private)
    assert capstone.validate_validation_plan(capstone.REPO_ROOT, plan) == []
    command_text = "\n".join(" ".join(command.argv) for command in plan)
    assert capstone.TEST_PATH.as_posix() in command_text
    assert "tests/python " not in command_text
    assert capstone.date_argument(capstone.RUN_DATE) == capstone.RUN_DATE
    with pytest.raises(ValueError, match="run date"):
        capstone.date_argument("20260920")


def test_fail_closed_helper_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7488: malformed sources cannot become usable evidence."""

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("- not-a-mapping\n")
    with pytest.raises(ValueError, match="mapping"):
        capstone.load_yaml_mapping(invalid_yaml)
    assert capstone._validation_rows({}) == []
    assert capstone._percentile([], 0.5) is None
    assert capstone._cluster_interval([])["draws"] == 0
    assert capstone._episode_cost_rows({"exclusive_cost_rows": [None]}, "A") == []
    assert capstone._mean_loss_rows([{"wrong": 1}], "losses") == {}
    assert capstone._receipt_set_passed({}, capstone.REQUIRED_CHECK_NAMES) is False

    root = tmp_path / "contract"
    (root / capstone.DESIGN_PATH.parent).mkdir(parents=True)
    (root / capstone.DESIGN_PATH).write_text(
        (capstone.REPO_ROOT / capstone.DESIGN_PATH).read_text()
    )
    (root / capstone.ROADMAP_PATH).write_text("milestone: wrong\ntasks: []\n")
    assert capstone.load_contract(root)["comparison_passed"] is False
    (root / capstone.ROADMAP_PATH).write_text("milestone: wrong\ntasks: bad\n")
    with pytest.raises(ValueError, match="task list"):
        capstone.load_contract(root)

    source = tmp_path / "source.json"
    source.write_text("{}")
    relative = Path("results/pre.json")
    (tmp_path / relative.parent).mkdir(exist_ok=True)
    (tmp_path / relative).write_text("{}")
    pre_gate = capstone._pre_gate_evidence(
        tmp_path,
        {"id": "exp9001-example", "deliverable": "results/unused.json"},
        relative,
        {
            "experiment": 9001,
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "failed_upstream": "exp9000-source",
            "failed_field": "ready",
            "failed_operator": "==",
            "failed_expected": 1,
            "failed_observed": 0,
            "failed_evidence_path": "source.json",
            "failed_evidence_sha256": capstone.sha256_file(source),
        },
    )
    assert pre_gate["authenticated"] is True

    failures = capstone._failure_rows(
        {"comparison_passed": False, "selected_roadmap_path": "wrong.yaml"},
        {
            "exp9001": {
                "evidence_state": "invalid",
                "gate_check_summary": None,
                "found_path": "bad.json",
                "expected_path": "bad.json",
                "required_validation_passed": False,
            },
            "exp9002": {
                "evidence_state": "missing",
                "gate_check_summary": None,
                "found_path": None,
                "expected_path": "missing.json",
                "required_validation_passed": False,
            },
        },
        {"required_checks_passed": False, "terminal_validation_passed": False},
    )
    assert [row["check"] for row in failures] == [
        "contract_authorities_agree",
        "required_source_validation",
        "source_validity",
        "affected_validation",
        "terminal_validation",
    ]

    staged = tmp_path / "staged"
    (staged / capstone.DESIGN_PATH.parent).mkdir(parents=True)
    (staged / capstone.DESIGN_PATH).write_text(
        (capstone.REPO_ROOT / capstone.DESIGN_PATH).read_text()
    )
    (staged / capstone.NEXT_ROADMAP_PATH).write_text(
        (capstone.REPO_ROOT / capstone.ROADMAP_PATH).read_text()
    )
    contract = capstone.load_contract(staged)
    monkeypatch.setattr(capstone, "SOURCE_PATHS", ())
    monkeypatch.setattr(capstone, "MODULE_PATH", capstone.NEXT_ROADMAP_PATH)
    monkeypatch.setattr(capstone, "WRAPPER_PATH", capstone.NEXT_ROADMAP_PATH)
    monkeypatch.setattr(capstone, "TEST_PATH", capstone.NEXT_ROADMAP_PATH)
    assert any(
        row["evidence_class"] == "selected_contract_authority"
        for row in capstone._source_hashes(staged, contract, {})
    )


def test_validator_reports_each_derived_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7488-ARTIFACT: each protected derived field is checked."""

    artifact = capstone.build_artifact_for_test()
    assert capstone.independent_reduce(artifact) == []
    assert capstone.validate_artifact([]) == ["artifact_mapping_required"]
    missing = deepcopy(artifact)
    del missing["schema"]
    assert capstone.validate_artifact(missing)[0].startswith("missing_required_field")
    assert capstone._source_hashes_match({}, capstone.REPO_ROOT) is False
    assert (
        capstone._source_hashes_match({"source_artifact_hashes": ["bad"]}, capstone.REPO_ROOT)
        is False
    )
    wrong_hash = deepcopy(artifact)
    wrong_hash["source_artifact_hashes"][0]["sha256"] = "sha256:" + "0" * 64
    assert capstone._source_hashes_match(wrong_hash, capstone.REPO_ROOT) is False
    assert "source_hash_mismatch" in capstone.validate_artifact(wrong_hash)

    mutations = {
        "honest_verdict": "bad",
        "MODEL_SPECS": ["bad"],
        "inference_substrate": "model",
        "task_dispositions": [],
        "science_reductions": {},
        "continuation_rows": [],
        "retirement_rows": [],
        "unresolved_obligations": [],
        "gate_check_summary": {},
        "acceptance_gate_results": [],
        "publication_gates": {},
        "capstone_complete_score": 0,
        "field_principles": {},
    }
    for field, replacement in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = replacement
        assert capstone.validate_artifact(changed), field

    no_terminal = deepcopy(artifact)
    no_terminal["validation_receipts"] = [
        row
        for row in no_terminal["validation_receipts"]
        if row["name"] not in capstone.TERMINAL_CHECK_NAMES
    ]
    assert "terminal_validation_incomplete" in capstone.validate_artifact(
        no_terminal, require_terminal=True
    )

    monkeypatch.setattr(
        capstone, "load_contract", lambda _root: (_ for _ in ()).throw(ValueError())
    )
    assert "independent_reduction_failed" in capstone.validate_artifact(artifact)
