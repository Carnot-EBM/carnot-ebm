"""Tests for REQ-REPORT-7502 and SCENARIO-REPORT-7502-*.

The real repository artifacts exercise the capstone reduction. Small private
fixtures exercise missing evidence and conductor pre-gate authentication.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7502_v656_capstone as capstone


def test_contract_resolves_exactly_fourteen_v656_tasks() -> None:
    """SCENARIO-REPORT-7502-DISPOSITIONS: contract order is authoritative."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    assert contract["comparison_passed"] is True
    assert contract["selected_roadmap_path"] == "research-roadmap.yaml"
    assert [task["id"] for task in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)

    changed = deepcopy(contract["roadmap"])
    changed["tasks"][4]["deliverable"] = "results/wrong.json"
    markdown = (capstone.REPO_ROOT / capstone.DESIGN_PATH).read_text()
    comparison = capstone.compare_contract_authorities(markdown, changed)
    assert comparison["passed"] is False


def test_real_evidence_distinguishes_terminal_and_missing_slots() -> None:
    """SCENARIO-REPORT-7502-DISPOSITIONS: absence remains blocked evidence."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    assert len(evidence) == 13
    assert evidence["exp7489-contract-methods"]["evidence_state"] == "terminal"
    assert evidence["exp7494-window-eval-capture"]["row_count"] == 280
    assert evidence["exp7495-window-calibration"]["evidence_state"] == "missing"
    assert evidence["exp7495-window-calibration"]["verdict_class"] == "blocked"
    assert evidence["exp7498-independent-audit"]["verdict_class"] == "blocked"
    assert evidence["exp7500-arc-opportunity-audit"]["required_validation_passed"] is True
    assert evidence["exp7501-service-placement"]["evidence_state"] == "missing"


def test_private_evidence_fixture_authenticates_pre_gate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7502-DISPOSITIONS: exact pre-gates keep their cause."""

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
                "milestone": capstone.MILESTONE,
                "status": "blocked",
                "honest_verdict": "blocked_gate_check_failed",
                "blocked_at_layer": "conductor_pre_gate",
                "failed_upstream": "exp9000-source",
                "failed_field": "ready_score",
                "failed_operator": "==",
                "failed_expected": 1,
                "failed_observed": 0,
                "failed_evidence_path": "source.json",
                "failed_evidence_sha256": capstone.sha256_file(source),
            }
        )
    )
    blocked = capstone.load_evidence_slot(tmp_path, task)
    assert blocked["evidence_state"] == "pre_gate"
    assert blocked["honest_verdict"] == "blocked_gate_check_failed"
    assert blocked["support_gate_status"] == "not_run"

    fallback.write_text("{}")
    assert capstone.load_evidence_slot(tmp_path, task)["evidence_state"] == "invalid"


def test_terminal_precedence_reserves_partial_for_current_work() -> None:
    """SCENARIO-REPORT-7502-COMPLETION: external absence is never partial."""

    terminal = [{"evidence_state": "terminal"}]
    missing = [*terminal, {"evidence_state": "missing"}]
    invalid = [*missing, {"evidence_state": "invalid"}]
    assert (
        capstone.classify_terminal(terminal, current_validation_complete=True)["verdict_class"]
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
        capstone.classify_terminal(terminal, current_validation_complete=False)["verdict_class"]
        == "partial"
    )


def test_claim_reduction_keeps_probability_utility_and_arc_separate() -> None:
    """SCENARIO-REPORT-7502-CLAIMS: one claim type cannot support another."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    claims = capstone.reduce_supported_claims(evidence)
    assert claims["historical_probability_quality"]["finding"] == "null"
    assert claims["historical_typed_decision_utility"]["finding"] == "historical_observed"
    assert claims["historical_typed_decision_utility"]["passed_cost_cells"] == 7
    assert claims["current_probability_quality"]["finding"] == "blocked_missing"
    assert claims["causal_feedback_benefit"]["finding"] == "blocked_missing"
    assert claims["retention"]["finding"] == "blocked_missing"
    assert claims["arc_generalization"]["valid_episode_count"] == 18
    assert claims["arc_generalization"]["valid_game_count"] == 6
    assert claims["arc_generalization"]["efficacy_estimate"] is None
    assert claims["hardware_placement"]["finding"] == "blocked_missing"
    assert claims["fixture_efficacy_promoted"] is False


def test_prior_failure_rows_apply_same_disposition_without_text_equality() -> None:
    """SCENARIO-REPORT-7502-RETIREMENT: no-benefit wording cannot reopen scope."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    rows = capstone.reduce_prior_failures(contract["tasks"], evidence)
    assert len(rows) == 14
    by_task = {row["task_id"]: row for row in rows}
    assert by_task["exp7492-window-pilot"]["same_scientific_disposition"] is True
    assert by_task["exp7492-window-pilot"]["retirement_triggered"] is True
    assert by_task["exp7495-window-calibration"]["comparison_state"] == "current_absent"
    assert by_task["exp7495-window-calibration"]["retirement_triggered"] is False
    assert by_task["exp7502-capstone"]["comparison_state"] == "current_work"
    assert all(row["retire_if_same_verdict"] is True for row in rows)


def test_closed_branches_have_one_changed_reopen_condition() -> None:
    """SCENARIO-REPORT-7502-RETIREMENT: closed mechanisms need a real change."""

    conditions = capstone.next_reopen_conditions()
    assert {row["branch"] for row in conditions} == {
        "importance_anchoring",
        "compact_generated_spans",
        "four_expert_reweighting",
        "generic_external_text_reranking",
        "current_window_probability",
        "causal_feedback_learning",
        "arc_cross_game_generalization",
        "durable_service_and_hardware_placement",
    }
    assert all(row["state"] == "closed" for row in conditions)
    assert all(row["changed_prerequisite"] for row in conditions)


def test_schema_complete_artifact_closes_reporting_without_promoting_science() -> None:
    """SCENARIO-REPORT-7502-COMPLETION: complete ledger can stay blocked."""

    artifact = capstone.build_artifact_for_test()
    assert capstone.validate_artifact(artifact, require_terminal=True) == []
    assert len(artifact["task_dispositions"]) == 14
    assert artifact["rows"] == artifact["task_dispositions"]
    assert artifact["capstone_complete_score"] == 1
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["supported_claims"]["fixture_efficacy_promoted"] is False
    assert all(row["principle"] for row in artifact["acceptance_gate_results"])
    assert set(artifact["field_principles"]) == set(artifact) - {
        "field_principles",
        "reproducibility_checksum",
    }


def test_dispositions_retain_gates_rows_and_hashes() -> None:
    """SCENARIO-REPORT-7502-DISPOSITIONS: every row keeps audit operands."""

    artifact = capstone.build_artifact_for_test()
    by_task = {row["task_id"]: row for row in artifact["task_dispositions"]}
    ready = by_task["exp7493-window-fit-capture"]
    assert ready["artifact_path"] == "results/experiment_7493_v656_window_fit_capture.json"
    assert ready["artifact_sha256"].startswith("sha256:")
    assert ready["rows_available"] is True
    assert ready["required_validation_status"] == "passed"
    assert ready["support_gate_status"] == "passed"
    missing = by_task["exp7497-causal-online-learning"]
    assert missing["artifact_path"] is None
    assert missing["verdict_class"] == "blocked"
    assert missing["required_validation_status"] == "not_run"
    current = by_task["exp7502-capstone"]
    assert current["evidence_state"] == "current_work"
    assert current["completed"] is True


def test_validator_rejects_protected_mutations() -> None:
    """SCENARIO-REPORT-7502-ARTIFACT: derived evidence fails closed."""

    artifact = capstone.build_artifact_for_test()
    mutations = {
        "milestone": "wrong",
        "honest_verdict": "bad",
        "MODEL_SPECS": ["wrong"],
        "inference_substrate": "wrong",
        "task_dispositions": [],
        "supported_claims": {},
        "prior_failure_rows": [],
        "retirement_rows": [],
        "next_reopen_conditions": [],
        "retrospective": {},
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
    """SCENARIO-REPORT-7502-ARTIFACT: validation cannot broaden silently."""

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


def test_retrospective_data_has_three_gaps_and_dominant_timings() -> None:
    """REQ-REPORT-7502: retrospective limits and timing remain evidence based."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    retrospective = capstone.reduce_retrospective(evidence)
    assert len(retrospective["prd_gaps"]) == 3
    assert retrospective["support_limits"]["arc_valid_episode_count"] == 18
    assert retrospective["support_limits"]["arc_valid_game_count"] == 6
    timings = {row["task_id"]: row for row in retrospective["dominant_actual_timings"]}
    assert timings["exp7493-window-fit-capture"]["dominant_stage"] == "prefill"
    assert timings["exp7494-window-eval-capture"]["dominant_stage"] == "prefill"
    assert timings["exp7494-window-eval-capture"]["dominant_duration_s"] > 2000
    assert len(retrospective["next_reopen_conditions"]) == 8


def test_fail_closed_helper_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7502: malformed bytes cannot become usable evidence."""

    invalid_yaml = tmp_path / "bad.yaml"
    invalid_yaml.write_text("- not-a-mapping\n")
    with pytest.raises(ValueError, match="mapping"):
        capstone.load_yaml_mapping(invalid_yaml)
    valid_yaml = tmp_path / "valid.yaml"
    valid_yaml.write_text("milestone: test\n")
    assert capstone.load_yaml_mapping(valid_yaml) == {"milestone": "test"}
    assert capstone._validation_rows({}) == []
    assert capstone._receipt_set_passed({}, capstone.REQUIRED_CHECK_NAMES) is False
    assert capstone._source_hashes_match({}, capstone.REPO_ROOT) is False
    assert capstone._source_hashes_match({"source_artifact_hashes": ["bad"]}, tmp_path) is False
    assert capstone._claim_state({"evidence_state": "invalid"}) == "disqualified"
    assert capstone._claim_state({"evidence_state": "terminal", "verdict_class": "blocked"}) == (
        "blocked_upstream"
    )
    assert capstone._claim_state({"evidence_state": "terminal", "verdict_class": "null"}) == (
        "available"
    )
    assert capstone._dominant_timing("missing", {}) is None
    assert capstone._dominant_timing("empty", {"payload": {"duration_components_s": {}}}) is None
    assert (
        capstone._dominant_timing(
            "total-only", {"payload": {"duration_components_s": {"total": 1.0}}}
        )
        is None
    )

    with pytest.raises(ValueError, match="exactly one prior failure"):
        capstone.reduce_prior_failures([{"id": "exp1", "prior_failures": []}], {})
    invalid_prior = {
        "task_id": "exp1",
        "evidence_state": "invalid",
        "honest_verdict": "complete_null_invalid",
        "verdict_class": "disqualified",
    }
    prior_rows = capstone.reduce_prior_failures(
        [
            {
                "id": "exp1",
                "prior_failures": [
                    {
                        "experiment_id": "exp0",
                        "verdict": "complete_null_prior",
                        "addressed_by": "changed",
                        "retire_if_same_verdict": True,
                    }
                ],
            }
        ],
        {"exp1": invalid_prior},
    )
    assert prior_rows[0]["comparison_state"] == "current_invalid"

    malformed_root = tmp_path / "malformed"
    (malformed_root / "results").mkdir(parents=True)
    malformed = malformed_root / "results/declared.json"
    malformed.write_text("not json")
    malformed_task = {"id": "exp9001-example", "deliverable": "results/declared.json"}
    assert capstone.load_evidence_slot(malformed_root, malformed_task)["evidence_state"] == (
        "invalid"
    )

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    failures = capstone.failure_rows(
        {**contract, "comparison_passed": False},
        evidence,
        {"required_checks_passed": False, "terminal_validation_passed": False},
    )
    checks = [row["check"] for row in failures]
    assert checks[0] == "contract_authorities_agree"
    assert "producer_evidence" in checks
    assert checks[-2:] == ["affected_validation", "terminal_validation"]

    invalid_failure = capstone.failure_rows(
        contract,
        {
            "exp1": {
                "evidence_state": "invalid",
                "found_path": "bad.json",
                "expected_path": "bad.json",
                "required_validation_passed": False,
                "verdict_class": "disqualified",
            }
        },
        {"required_checks_passed": True, "terminal_validation_passed": True},
    )
    assert invalid_failure[0]["check"] == "required_source_validation"

    staged = tmp_path / "staged"
    staged.mkdir()
    selected = staged / "selected.yaml"
    selected.write_text("milestone: test\n")
    monkeypatch.setattr(capstone, "SOURCE_PATHS", ())
    monkeypatch.setattr(capstone, "MODULE_PATH", Path("selected.yaml"))
    monkeypatch.setattr(capstone, "WRAPPER_PATH", Path("selected.yaml"))
    monkeypatch.setattr(capstone, "TEST_PATH", Path("selected.yaml"))
    hashes = capstone._source_hashes(staged, {"selected_roadmap_path": "selected.yaml"}, {})
    assert any(row["evidence_class"] == "selected_contract_authority" for row in hashes)

    monkeypatch.setattr(
        capstone,
        "resolve_v656_roadmap",
        lambda _root: (
            staged / "selected.yaml",
            {"milestone": capstone.MILESTONE, "tasks": {}},
            [],
        ),
    )
    monkeypatch.setattr(capstone, "compare_contract_authorities", lambda _text, _roadmap: {})
    monkeypatch.setattr(capstone, "DESIGN_PATH", Path("selected.yaml"))
    with pytest.raises(ValueError, match="task list"):
        capstone.load_contract(staged)


def test_independent_reducer_and_terminal_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7502-ARTIFACT: cold replay covers terminal receipts."""

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

    monkeypatch.setattr(
        capstone, "load_contract", lambda _root: (_ for _ in ()).throw(ValueError())
    )
    assert "independent_reduction_failed" in capstone.validate_artifact(artifact)
