"""Tests for Exp 1609 residual-drift context induction.

Spec: REQ-VERIFY-1609, SCENARIO-VERIFY-1609.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.pipeline import context_induction as exp


def test_req_verify_1609_mines_drift_and_excludes_contradictions(tmp_path: Path) -> None:
    """REQ-VERIFY-1609: mining keeps satisfiable drift separate from contradictions."""

    ledger = tmp_path / "ledger.jsonl"
    repairs = tmp_path / "repairs.jsonl"
    conductor = tmp_path / "conductor.log"
    _write_jsonl(
        ledger,
        [
            _satquest_drift_row("sat-drift"),
            _satquest_assignment_drift_row("sat-assignment"),
            _satquest_contradiction_row("sat-contradiction"),
            {"row_type": "residual_drift_case", "failure_classification": "other_blocker"},
            {"row_type": "residual_drift_summary", "satisfiable_drift_cases": 3},
            _product_line_drift_row("pl-drift"),
            _runtime_drift_row("runtime-drift"),
        ],
    )
    _write_jsonl(
        repairs,
        [
            {
                "row_type": "residual_drift_repair_case",
                "case_id": "runtime-unrepaired",
                "source_domain": "runtime_contract",
                "failure_classification": "satisfiable_drift",
                "attempted": True,
                "accepted": False,
                "localized": True,
                "localization": {
                    "localized_span": "commitments[1].evidence.root_cause_category",
                    "repair_kind": "runtime_contract_root_cause_patch",
                },
                "replay": {"validator": "runtime_contract", "reason": "runtime_contract_replay_failed"},
            },
            {
                "row_type": "residual_drift_repair_case",
                "case_id": "repair-contradiction",
                "source_domain": "satquest",
                "failure_classification": "true_contradiction",
                "rejection_reason": "true_contradiction_untouched",
                "replay": {"validator": "sat_oracle"},
            },
            {
                "row_type": "residual_drift_repair_case",
                "case_id": "repair-accepted",
                "source_domain": "satquest",
                "failure_classification": "satisfiable_drift",
                "accepted": True,
            },
            {
                "row_type": "residual_drift_repair_summary",
                "repair_attempts": 1,
            }
        ],
    )
    conductor.write_text(
        "2026-05-09 unrelated failure line\n"
        "2026-05-09 residual_drift failure "
        "source_domain=runtime_contract case_id=runtime-log "
        "localized_span=commitments[1].evidence.root_cause_category "
        "repair_kind=runtime_contract_root_cause_patch validator=runtime_contract\n"
        "2026-05-09 residual_drift failure source_domain=custom case_id=custom-log\n",
        encoding="utf-8",
    )

    mined = exp.mine_failure_logs(
        project_root=tmp_path,
        ledger_path=ledger,
        repair_manifest_path=repairs,
        conductor_log_path=conductor,
    )

    positives = [item for item in mined.evidence if item.is_positive]
    exclusions = [item for item in mined.evidence if item.is_exclusion]
    assert mined.source_counts["ledger_rows"] == 7
    assert mined.source_counts["repair_rows"] == 4
    assert mined.source_counts["conductor_failure_lines"] == 2
    assert {item.case_id for item in positives} == {
        "sat-drift",
        "sat-assignment",
        "pl-drift",
        "runtime-drift",
        "runtime-unrepaired",
        "runtime-log",
        "custom-log",
    }
    assert [item.case_id for item in exclusions] == ["sat-contradiction", "repair-contradiction"]
    assert positives[0].localized_span == "commitments[1].evidence.answer"
    assert [item for item in positives if item.case_id == "sat-assignment"][0].localized_span == "commitments[1].evidence.assignment"
    assert all(item.failure_classification != exp.CLASS_TRUE_CONTRADICTION for item in positives)


def test_scenario_verify_1609_generates_context_sensitive_candidate() -> None:
    """SCENARIO-VERIFY-1609: candidates are grouped by localized trigger context."""

    evidence = [
        exp.FailureEvidence(
            case_id="runtime-a",
            source_path="results/residual_drift_commitment_ledger_1538.jsonl",
            source_kind="ledger",
            source_domain="runtime_contract",
            failure_classification=exp.CLASS_SATISFIABLE_DRIFT,
            localized_span="commitments[1].evidence.root_cause_category",
            repair_kind="runtime_contract_root_cause_patch",
            validator="runtime_contract",
            contract_family="tool_ordering",
            signal="ledger_satisfiable_drift",
            is_positive=True,
        ),
        exp.FailureEvidence(
            case_id="runtime-b",
            source_path="logs/conductor.log",
            source_kind="conductor_log",
            source_domain="runtime_contract",
            failure_classification=exp.CLASS_SATISFIABLE_DRIFT,
            localized_span="commitments[1].evidence.root_cause_category",
            repair_kind="runtime_contract_root_cause_patch",
            validator="runtime_contract",
            contract_family="tool_ordering",
            signal="conductor_residual_drift_failure",
            is_positive=True,
        ),
        exp.FailureEvidence(
            case_id="sat-contradiction",
            source_path="results/residual_drift_commitment_ledger_1538.jsonl",
            source_kind="ledger",
            source_domain="satquest",
            failure_classification=exp.CLASS_TRUE_CONTRADICTION,
            localized_span="commitments[1].evidence.answer",
            repair_kind="sat_answer_or_assignment_patch",
            validator="sat_oracle",
            contract_family=None,
            signal="true_contradiction_exclusion",
            is_positive=False,
            is_exclusion=True,
        ),
    ]

    candidates = exp.generate_constraint_candidates(evidence)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["constraint_id"].startswith("ctx1609_runtime_contract_")
    assert candidate["trigger_context"] == {
        "contract_family": "tool_ordering",
        "localized_span": "commitments[1].evidence.root_cause_category",
        "repair_kind": "runtime_contract_root_cause_patch",
        "source_domain": "runtime_contract",
        "validator": "runtime_contract",
    }
    assert candidate["support_count"] == 2
    assert candidate["positive_evidence_case_ids"] == ["runtime-a", "runtime-b"]
    assert candidate["negative_evidence_case_ids"] == ["sat-contradiction"]
    assert "true_contradiction" in " ".join(candidate["guardrails"])
    assert 0.0 < candidate["confidence"] <= 1.0

    solo = exp.generate_constraint_candidates(evidence[:1])
    assert solo[0]["confidence"] == pytest.approx(1.0)


def test_scenario_verify_1609_runner_writes_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1609: runner writes the bounded context-induction artifact."""

    ledger = tmp_path / "ledger.jsonl"
    repairs = tmp_path / "repairs.jsonl"
    conductor = tmp_path / "conductor.log"
    output = tmp_path / "experiment_1609_context_induction.json"
    _write_jsonl(ledger, [_satquest_drift_row("sat-drift"), _satquest_contradiction_row("sat-contradiction")])
    _write_jsonl(repairs, [])
    conductor.write_text("", encoding="utf-8")

    artifact = exp.run_experiment(
        project_root=tmp_path,
        ledger_path=ledger,
        repair_manifest_path=repairs,
        conductor_log_path=conductor,
        output_path=output,
        focused_tests_passed=True,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == "1609"
    assert artifact["context_induction_ready"] is True
    assert artifact["failure_logs_mined"] == 2
    assert artifact["candidate_constraints_generated"] == 1
    assert artifact["selected_candidate"]["positive_evidence_case_ids"] == ["sat-drift"]
    assert artifact["true_contradiction_exclusions"] == ["sat-contradiction"]
    assert artifact["focused_tests_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1609_missing_sources_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-1609: missing logs are blockers and do not fabricate candidates."""

    output = tmp_path / "experiment_1609_context_induction.json"

    artifact = exp.run_experiment(
        project_root=tmp_path,
        ledger_path=tmp_path / "missing-ledger.jsonl",
        repair_manifest_path=tmp_path / "missing-repairs.jsonl",
        conductor_log_path=tmp_path / "missing-conductor.log",
        output_path=output,
        focused_tests_passed=False,
    )

    assert artifact["status"] == "blocked"
    assert artifact["context_induction_ready"] is False
    assert artifact["failure_logs_mined"] == 0
    assert artifact["candidate_constraints_generated"] == 0
    assert artifact["selected_candidate"] is None
    assert "focused_tests_not_passed" in artifact["blockers"]
    assert all(str(item).startswith("missing_") for item in artifact["blockers"][:3])
    assert exp._commitments_by_name({"commitments": "malformed"}) == {}  # noqa: SLF001


def _satquest_drift_row(case_id: str) -> dict[str, Any]:
    return {
        "row_type": "residual_drift_case",
        "source_domain": "satquest",
        "source_case_id": case_id,
        "failure_classification": "satisfiable_drift",
        "false_accept": False,
        "commitments": [
            {"turn": 1, "name": "cnf_constraints", "evidence": {"n_vars": 1, "clauses": [[1]]}},
            {"turn": 2, "name": "model_answer", "evidence": {"answer": "UNSAT", "parse_ok": True}},
            {
                "turn": 3,
                "name": "solver_oracle_validation",
                "evidence": {"label": "SAT", "satisfiable": True, "satisfying_assignment": [True]},
            },
        ],
        "solver_oracle": {"used": True, "label": "SAT", "satisfiable": True},
        "deterministic_validator": {"classification": "wrong_label"},
    }


def _satquest_contradiction_row(case_id: str) -> dict[str, Any]:
    row = _satquest_drift_row(case_id)
    row["failure_classification"] = "true_contradiction"
    row["commitments"][0]["evidence"] = {"n_vars": 1, "clauses": [[1], [-1]]}
    row["commitments"][1]["evidence"] = {"answer": "SAT", "parse_ok": True}
    row["commitments"][2]["evidence"] = {
        "label": "UNSAT",
        "satisfiable": False,
        "satisfying_assignment": None,
    }
    row["solver_oracle"] = {"used": True, "label": "UNSAT", "satisfiable": False}
    return row


def _satquest_assignment_drift_row(case_id: str) -> dict[str, Any]:
    row = _satquest_drift_row(case_id)
    row["deterministic_validator"] = {"classification": "invalid_assignment"}
    return row


def _runtime_drift_row(case_id: str) -> dict[str, Any]:
    return {
        "row_type": "residual_drift_case",
        "source_domain": "runtime_contract",
        "source_case_id": case_id,
        "failure_classification": "satisfiable_drift",
        "false_accept": False,
        "commitments": [
            {"turn": 1, "name": "runtime_contract_prompt", "evidence": {"prompt_or_case_id": case_id}},
            {
                "turn": 2,
                "name": "cdg_failure_localization",
                "evidence": {
                    "root_cause_category": "structural_dependency",
                    "failure_categories": ["structural_dependency", "final_accept"],
                },
            },
            {
                "turn": 3,
                "name": "deterministic_contract_validation",
                "evidence": {
                    "expected_label": False,
                    "final_deterministic_accept": False,
                    "structural_contract_result": {
                        "contract_family": "tool_ordering",
                        "detected_violation": True,
                    },
                },
            },
        ],
        "solver_oracle": {"used": True, "satisfiable": True, "root_cause_category": "structural_dependency"},
        "deterministic_validator": {
            "deterministic_validator_accept": True,
            "expected_label": False,
            "final_deterministic_accept": False,
        },
    }


def _product_line_drift_row(case_id: str) -> dict[str, Any]:
    return {
        "row_type": "residual_drift_case",
        "source_domain": "product_line",
        "source_case_id": case_id,
        "failure_classification": "satisfiable_drift",
        "false_accept": False,
        "commitments": [
            {
                "turn": 1,
                "name": "feature_model_contract",
                "evidence": {"model_id": "UnitProduct", "operation": {"budget": 3}},
            },
            {
                "turn": 2,
                "name": "baseline_selection",
                "evidence": {"parse_ok": True, "oracle_agrees": False},
            },
            {
                "turn": 3,
                "name": "solver_oracle_validation",
                "evidence": {"oracle_agrees": True, "optimal_features": ["Core", "FastPath"]},
            },
        ],
        "solver_oracle": {"used": True, "satisfiable": True, "oracle_agrees_after_repair": True},
        "deterministic_validator": {
            "baseline_oracle_agrees": False,
            "oracle_agrees_after_repair": True,
            "policy_false_accept": False,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
