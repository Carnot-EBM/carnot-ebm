"""Tests for Exp 3137 exact-safe accept/abstain/reject contract.

Spec refs: REQ-VERIFY-3137, SCENARIO-VERIFY-3137.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import exact_safe_accept_abstain_contract_v1 as mod


REQUIRED_FIELDS = {
    "acceptance_contract_v1_ready",
    "contract_rules",
    "known_false_accept_rows_blocked",
    "replay_false_accept_rate",
    "replay_abstention_rate",
    "replay_false_reject_rate",
    "repair_gate_prerequisites",
    "regression_row_set",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(root: Path, rel_path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _monitor_events(
    fixture_id: str,
    exact_label: str,
    extracted_answer: str,
    live_decision: str,
) -> list[dict[str, Any]]:
    expected_action = mod.expected_action_from_label(exact_label)
    consistent = live_decision == expected_action
    return [
        {
            "event_type": "constraint_ledger",
            "event_index": 1,
            "fixture_id": fixture_id,
            "payload": {
                "ledger_action": expected_action,
                "ledger_source": "unit_exact_ledger",
            },
        },
        {
            "event_type": "candidate_final_answer",
            "event_index": 2,
            "fixture_id": fixture_id,
            "payload": {
                "extracted_answer": extracted_answer,
                "live_decision": live_decision,
                "expected_action": expected_action,
                "ledger_action": expected_action,
                "final_answer_consistent_with_exact": consistent,
                "final_answer_consistent_with_ledger": consistent,
                "has_returned_answer": True,
            },
        },
        {
            "event_type": "drift_classification",
            "event_index": 3,
            "fixture_id": fixture_id,
            "payload": {
                "failure_mechanism": "no_failure" if consistent else "contradiction",
                "is_monitor_violation": not consistent,
            },
        },
    ]


def _live_row(
    fixture_id: str,
    exact_label: str,
    extracted_answer: str,
    live_decision: str,
    *,
    mechanism: str = "no_failure",
    family: str = "arithmetic_code_assertions",
) -> dict[str, Any]:
    return {
        "row_id": fixture_id,
        "fixture_id": fixture_id,
        "exact_label": exact_label,
        "expected_action": mod.expected_action_from_label(exact_label),
        "extracted_answer": extracted_answer,
        "answer_extraction_format": mod.token_family_for_label(exact_label),
        "live_decision": live_decision,
        "failure_mechanism_from_exp3124": mechanism,
        "primary_mechanism": mechanism,
        "fixture_family": family,
        "task_family": family,
        "label_source": "unit_exact_authority",
    }


def _manifest_row(
    fixture_id: str,
    expected_answer: str,
    *,
    family: str = "arithmetic_code_assertions",
) -> dict[str, Any]:
    return {
        "schema": "carnot.exact_fixture_eval_manifest.v1",
        "source_fixture_id": fixture_id,
        "source_prompt_payload_sha256": f"{fixture_id}-prompt",
        "task_family": family,
        "expected_answer": expected_answer,
        "solver_label": expected_answer.lower(),
        "label_source": "unit_exact_authority",
        "verifier_target": {"expected_action": mod.expected_action_from_label(expected_answer)},
        "leakage_safe_prompt_payload": {"fixture": fixture_id, "expected": expected_answer},
    }


def _write_sources(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("exact verifier authenticity\n", encoding="utf-8")
    (root / "research-references.md").write_text("safe abstention references\n", encoding="utf-8")
    (root / "openspec/capabilities/verification").mkdir(parents=True, exist_ok=True)
    (root / "openspec/capabilities/verification/spec.md").write_text(
        "REQ-VERIFY-3137\nSCENARIO-VERIFY-3137\n", encoding="utf-8"
    )

    live_rows = [
        _live_row("live-valid", "VALID", "VALID", "accept"),
        _live_row("live-invalid", "INVALID", "INVALID", "reject"),
        _live_row("fa-arith", "INVALID", "VALID", "accept", mechanism="contradiction miss"),
        _live_row(
            "fa-smt",
            "UNSAT",
            "VALID",
            "accept",
            mechanism="SAT/validity-token confusion",
            family="smt_constraints",
        ),
    ]
    false_rows = [row for row in live_rows if row["row_id"].startswith("fa-")]
    _write_json(
        root,
        mod.EXP3136_REL_PATH,
        {
            "artifact": "experiment_3136_false_accept_root_cause_autopsy_v1",
            "false_accept_autopsy_v1_ready": True,
            "source_false_accept_rate": 0.5,
            "false_accept_row_ids": ["fa-arith", "fa-smt"],
            "regression_row_set": ["fa-arith", "fa-smt"],
            "false_accept_rows": false_rows,
            "verifier_rows": live_rows,
            "recommended_contract_changes": ["add regression gates", "add token-family checks"],
            "inference_substrate": {"fresh_live_model_calls": 0, "upstream_live_model_calls_reused": 4},
        },
    )
    _write_json(
        root,
        mod.EXP3125_REL_PATH,
        {
            "artifact": "experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1",
            "prefix_closed_bound_pilot_ready": True,
            "fixture_details": [
                {"fixture_id": "pc-valid", "expected_answer": "VALID"},
                {"fixture_id": "pc-invalid", "expected_answer": "INVALID"},
                {"fixture_id": "pc-sat", "expected_answer": "SAT"},
            ],
            "semantic_coverage": {"answer_label_semantics": {"covered": True}},
            "bound_width": 0.01,
        },
    )
    monitor_events: list[dict[str, Any]] = []
    for row in live_rows:
        monitor_events.extend(
            _monitor_events(
                row["fixture_id"],
                row["exact_label"],
                row["extracted_answer"],
                row["live_decision"],
            )
        )
    _write_json(
        root,
        mod.EXP3126_REL_PATH,
        {
            "artifact": "experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1",
            "fragment_time_monitor_v1_ready": True,
            "monitor_events": monitor_events,
            "monitor_violation_count": 2,
            "ledger_consistency_rate": 0.5,
        },
    )
    _write_json(
        root,
        mod.EXP3098_REL_PATH,
        {
            "artifact": "experiment_3098_maxsat_abstention_routing_policy_v1",
            "maxsat_policy_ready": True,
            "fallback_evaluator": {
                "kind": "deterministic_reference_evaluator",
                "fail_closed_default": "abstain",
                "tie_break_order": ["abstain", "reject", "accept"],
            },
        },
    )
    manifest_rows = [
        _manifest_row("fixture-valid", "VALID"),
        _manifest_row("fixture-invalid", "INVALID"),
        _manifest_row("fixture-sat", "SAT", family="smt_constraints"),
        _manifest_row("fixture-unsat", "UNSAT", family="smt_constraints"),
        _manifest_row("fixture-repair", "REPAIRABLE", family="repairable_invalid_candidates"),
    ]
    _write_jsonl(root, mod.MANIFEST_REL_PATH, manifest_rows)
    _write_json(
        root,
        mod.EXP3097_REL_PATH,
        {
            "artifact": "experiment_3097_exact_fixture_eval_protocol_audit_v1",
            "eval_protocol_ready": True,
            "stratified_eval_manifest_path": mod.MANIFEST_REL_PATH.as_posix(),
            "usable_fixture_count": len(manifest_rows),
        },
    )


def test_req_verify_3137_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3137: OpenSpec declares the exact-safe contract."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3137" in spec
    assert "SCENARIO-VERIFY-3137" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "known_false_accept_rows_blocked" in spec
    assert "replay_false_accept_rate" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3137_replay_blocks_regression_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3137: .291 false accepts are blocked from accept."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=12.5,
        tests_run=["REQ-VERIFY-3137 focused"],
    )
    decisions = {row["row_id"]: row for row in artifact["replay_rows"] if row["row_source"] == "live"}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["acceptance_contract_v1_ready"] is True
    assert artifact["known_false_accept_rows_blocked"] is True
    assert artifact["regression_row_set"] == ["fa-arith", "fa-smt"]
    assert artifact["replay_false_accept_rate"] == 0.0
    assert artifact["replay_abstention_rate"] > 0.0
    assert artifact["replay_false_reject_rate"] == 0.0
    assert decisions["fa-arith"]["decision"] == "abstain"
    assert decisions["fa-smt"]["decision"] == "abstain"
    assert decisions["live-valid"]["decision"] == "accept"
    assert decisions["live-invalid"]["decision"] == "reject"
    assert artifact["repair_gate_prerequisites"]["must_load_contract_path"] == mod.OUTPUT_REL_PATH.as_posix()
    assert artifact["repair_gate_prerequisites"]["replay_false_accept_rate_must_equal"] == 0.0
    assert artifact["tests_run"] == ["REQ-VERIFY-3137 focused"]
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["no_live_llm_inference"] is True
    assert artifact["self_checks"]["regression_rows_blocked_from_accept"] is True

    rule_decisions = {rule["decision"] for rule in artifact["contract_rules"]}
    assert {"accept", "abstain", "reject"} <= rule_decisions
    mod.validate_artifact(artifact)


def test_req_verify_3137_contract_order_abstains_before_reject() -> None:
    """REQ-VERIFY-3137: known false-accept rows hit abstain before reject."""

    context = mod.ContractContext(
        prefix_covered_labels=frozenset({"VALID", "INVALID", "SAT"}),
        regression_row_set=frozenset({"known"}),
        monitor_by_fixture={
            "known": _monitor_events("known", "INVALID", "VALID", "accept"),
            "unknown": _monitor_events("unknown", "INVALID", "VALID", "accept"),
        },
    )

    known = mod.evaluate_row(
        _live_row("known", "INVALID", "VALID", "accept", mechanism="contradiction miss"),
        context,
        row_source="live",
    )
    unknown = mod.evaluate_row(
        _live_row("unknown", "INVALID", "VALID", "accept", mechanism="no_failure"),
        context,
        row_source="live",
    )

    assert known["decision"] == "abstain"
    assert known["matched_rule_id"] == "ABSTAIN_KNOWN_FALSE_ACCEPT_REGRESSION"
    assert unknown["decision"] == "reject"
    assert unknown["matched_rule_id"] == "REJECT_EXACT_REJECT_CONTRADICTION"


def test_req_verify_3137_write_artifact_and_validate_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3137: writer persists the contract and validation blocks overclaiming."""

    _write_sources(tmp_path)

    out_path = mod.write_artifact(
        tmp_path,
        started_s=1.0,
        now_s=1.25,
        tests_run=["REQ-VERIFY-3137 focused"],
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["acceptance_contract_v1_ready"] is True
    assert artifact["source_checksums"][mod.EXP3136_REL_PATH.as_posix()]

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="known_false_accept_rows_blocked"):
        mod.validate_artifact(artifact | {"known_false_accept_rows_blocked": False})
    with pytest.raises(ValueError, match="replay_false_accept_rate"):
        mod.validate_artifact(artifact | {"replay_false_accept_rate": 0.1})
    with pytest.raises(ValueError, match="rate outside"):
        mod.validate_artifact(artifact | {"replay_abstention_rate": 1.5})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "ready: no"})


def test_req_verify_3137_defensive_fallbacks(tmp_path: Path) -> None:
    """REQ-VERIFY-3137: incomplete rows fail closed instead of accepting."""

    context = mod.ContractContext(
        prefix_covered_labels=frozenset({"VALID"}),
        regression_row_set=frozenset(),
        monitor_by_fixture={},
    )
    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text('\nnot-json\n[]\n{"source_fixture_id":"ok"}\n', encoding="utf-8")

    missing_exact = mod.evaluate_row({"fixture_id": "missing", "extracted_answer": "VALID"}, context)
    missing_monitor = mod.evaluate_row(
        {
            "fixture_id": "live",
            "exact_label": "VALID",
            "expected_action": "accept",
            "extracted_answer": "VALID",
        },
        context,
        row_source="live",
    )
    low_parse = mod.evaluate_row(
        {
            "fixture_id": "low",
            "exact_label": "VALID",
            "expected_action": "accept",
            "extracted_answer": "VALID",
            "parse_confidence": 0.5,
        },
        context,
    )
    uncovered = mod.evaluate_row(
        {
            "fixture_id": "sat",
            "exact_label": "SAT",
            "expected_action": "accept",
            "extracted_answer": "SAT",
        },
        context,
    )
    token_mismatch = mod.evaluate_row(
        {
            "fixture_id": "token",
            "exact_label": "SAT",
            "expected_action": "accept",
            "extracted_answer": "VALID",
        },
        context,
    )
    default_abstain = mod.evaluate_row(
        {
            "fixture_id": "default",
            "exact_label": "INVALID",
            "expected_action": "reject",
            "extracted_answer": "VALID",
        },
        context,
    )

    assert mod.read_jsonl_rows(jsonl_path) == [{"source_fixture_id": "ok"}]
    assert mod.live_rows_from_autopsy({"false_accept_rows": [{"fixture_id": "only"}]}) == [
        {"fixture_id": "only"}
    ]
    assert mod.exact_fixture_rows_from_manifest([{}]) == []
    assert (
        mod.honest_verdict({"acceptance_contract_v1_ready": False})
        == "blocked_exact_safe_contract_missing_required_replay_evidence"
    )
    assert missing_exact["matched_rule_id"] == "ABSTAIN_MISSING_EXACT_LABEL"
    assert missing_monitor["matched_rule_id"] == "ABSTAIN_MISSING_LIVE_MONITOR_REPLAY"
    assert low_parse["matched_rule_id"] == "ABSTAIN_LOW_PARSE_CONFIDENCE"
    assert uncovered["matched_rule_id"] == "ABSTAIN_ACCEPT_LABEL_OUTSIDE_PREFIX_COVERAGE"
    assert token_mismatch["matched_rule_id"] == "ABSTAIN_TOKEN_FAMILY_MISMATCH"
    assert default_abstain["matched_rule_id"] == "ABSTAIN_DEFAULT_FAIL_CLOSED"
