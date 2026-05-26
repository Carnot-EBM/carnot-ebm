"""Tests for Exp 3138 canonical answer and VeriCoT grounding pilot.

Spec refs: REQ-VERIFY-3138, SCENARIO-VERIFY-3138.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import canonical_answer_vericot_grounding_pilot_v1 as mod


REQUIRED_FIELDS = {
    "canonical_grounding_pilot_v1_ready",
    "canonicalizer_implemented",
    "premise_grounding_rows",
    "regression_rows_evaluated",
    "false_accept_rows_blocked",
    "canonicalization_block_count",
    "premise_grounding_block_count",
    "ledger_replay_block_count",
    "residual_false_accept_rows",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _monitor_events(
    fixture_id: str,
    exact_label: str,
    extracted_answer: str,
    live_decision: str,
    *,
    ledger_action: str = "reject",
    constraints: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    constraints = [] if constraints is None else constraints
    grounded_state = "fragment_trace_present" if constraints else "no_fragment_trace"
    consistent = live_decision == ledger_action and extracted_answer == exact_label
    return [
        {
            "event_index": 1,
            "event_type": "partial_trace_state",
            "fixture_id": fixture_id,
            "payload": {
                "fragment_count": len(constraints),
                "fragment_ids": [row["constraint_id"] for row in constraints],
                "fragment_status_counts": {
                    status: sum(row.get("status") == status for row in constraints)
                    for status in ("pass", "fail")
                },
                "partial_state": grounded_state,
            },
        },
        {
            "event_index": 2,
            "event_type": "constraint_ledger",
            "fixture_id": fixture_id,
            "payload": {
                "constraints": constraints,
                "constraint_count": len(constraints),
                "ledger_action": ledger_action,
                "ledger_source": "unit_ledger",
            },
        },
        {
            "event_index": 3,
            "event_type": "exact_test_z3_result",
            "fixture_id": fixture_id,
            "payload": {
                "exact_label": exact_label,
                "expected_action": mod.expected_action_from_label(exact_label),
                "solver_label": "unsat" if exact_label == "UNSAT" else "assertion_fails",
            },
        },
        {
            "event_index": 4,
            "event_type": "candidate_final_answer",
            "fixture_id": fixture_id,
            "payload": {
                "expected_action": mod.expected_action_from_label(exact_label),
                "extracted_answer": extracted_answer,
                "final_answer_consistent_with_exact": extracted_answer == exact_label,
                "final_answer_consistent_with_ledger": consistent,
                "has_returned_answer": True,
                "ledger_action": ledger_action,
                "live_decision": live_decision,
            },
        },
        {
            "event_index": 5,
            "event_type": "drift_classification",
            "fixture_id": fixture_id,
            "payload": {
                "exact_label": exact_label,
                "expected_action": mod.expected_action_from_label(exact_label),
                "failure_mechanism": "contradiction" if not consistent else "no_failure",
                "is_monitor_violation": not consistent,
                "ledger_action": ledger_action,
                "live_decision": live_decision,
            },
        },
    ]


def _false_row(row_id: str, exact_label: str, extracted_answer: str) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "fixture_id": row_id,
        "exact_label": exact_label,
        "expected_action": mod.expected_action_from_label(exact_label),
        "extracted_answer": extracted_answer,
        "live_decision": "accept",
        "primary_mechanism": "SAT/validity-token confusion"
        if exact_label == "UNSAT"
        else "contradiction miss",
    }


def _certificate(fixture_id: str, exact_label: str) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "exact_label": exact_label,
        "solver_label": "unsat" if exact_label == "UNSAT" else "assertion_fails",
        "solver_authority": "z3_solver" if exact_label == "UNSAT" else "python_ast_runtime_execution",
        "coherence_status": "incoherent",
        "maxsat_route": {"action": "reject"},
        "minimal_correction_set": {"kind": "unit_correction"},
        "unsat_core": ["constraint_0"],
    }


def _write_sources(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("no fake model-backed verifier\n", encoding="utf-8")
    (root / "research-references.md").write_text("VeriCoT premise grounding\n", encoding="utf-8")
    (root / "openspec/capabilities/verification").mkdir(parents=True, exist_ok=True)
    (root / "openspec/capabilities/verification/spec.md").write_text(
        "REQ-VERIFY-3138\nSCENARIO-VERIFY-3138\n", encoding="utf-8"
    )
    false_rows = [
        _false_row("fa-arith", "INVALID", "VALID"),
        _false_row("fa-smt", "UNSAT", "VALID"),
    ]
    _write_json(
        root,
        mod.EXP3136_REL_PATH,
        {
            "artifact": "experiment_3136_false_accept_root_cause_autopsy_v1",
            "false_accept_autopsy_v1_ready": True,
            "false_accept_row_ids": ["fa-arith", "fa-smt"],
            "false_accept_rows": false_rows,
            "regression_row_set": ["fa-arith", "fa-smt"],
            "inference_substrate": {"upstream_live_model_calls_reused": 2},
        },
    )
    _write_json(
        root,
        mod.EXP3137_REL_PATH,
        {
            "artifact": "experiment_3137_exact_safe_accept_abstain_contract_v1",
            "acceptance_contract_v1_ready": True,
            "replay_rows": [
                {
                    "row_id": "fa-arith",
                    "row_source": "live",
                    "decision": "abstain",
                    "matched_rule_id": "ABSTAIN_KNOWN_FALSE_ACCEPT_REGRESSION",
                },
                {
                    "row_id": "fa-smt",
                    "row_source": "live",
                    "decision": "abstain",
                    "matched_rule_id": "ABSTAIN_KNOWN_FALSE_ACCEPT_REGRESSION",
                },
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3111_REL_PATH,
        {
            "artifact": "experiment_3111_certified_coherence_z3_mcs_feedback_v3",
            "certified_coherence_feedback_v3_ready": True,
            "certificates": [_certificate("fa-arith", "INVALID"), _certificate("fa-smt", "UNSAT")],
        },
    )
    events = []
    events.extend(_monitor_events("fa-arith", "INVALID", "VALID", "accept"))
    events.extend(_monitor_events("fa-smt", "UNSAT", "VALID", "accept"))
    _write_json(
        root,
        mod.EXP3126_REL_PATH,
        {
            "artifact": "experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1",
            "fragment_time_monitor_v1_ready": True,
            "monitor_events": events,
        },
    )


def test_req_verify_3138_spec_anchor_exists() -> None:
    """REQ-VERIFY-3138: OpenSpec declares the canonical grounding pilot."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3138" in spec
    assert "SCENARIO-VERIFY-3138" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "canonicalization_block_count" in spec
    assert "premise_grounding_block_count" in spec
    assert mod.REPO_ROOT.joinpath("scripts/research_conductor.py").exists()


def test_req_verify_3138_canonical_forms_and_equivalence() -> None:
    """REQ-VERIFY-3138: canonical forms distinguish labels, numbers, JSON, and code."""

    valid = mod.canonicalize_answer(" valid ")
    sat = mod.canonicalize_answer("satisfiable")
    number = mod.canonicalize_answer("2.00")
    json_answer = mod.canonicalize_answer('{"b": 2, "a": [1, true]}')
    code_fail = mod.canonicalize_answer("assertion_fails")
    too_deep = mod.canonicalize_answer({"a": {"b": {"c": 1}}}, json_max_depth=2)

    assert valid.to_dict() == {
        "kind": "label",
        "value": "VALID",
        "family": "validity_token",
        "normalized": "VALID",
        "parse_status": "parsed",
    }
    assert sat.family == "sat_token"
    assert not mod.answers_equivalent("VALID", "SAT")
    assert mod.answers_equivalent(number, 2)
    assert json_answer.normalized == '{"a":[1,true],"b":2}'
    assert code_fail.to_dict()["value"] == "fail"
    assert code_fail.to_dict()["family"] == "code_fragment"
    assert too_deep.parse_status == "too_large"


def test_req_verify_3138_edge_canonicalization_paths() -> None:
    """REQ-VERIFY-3138: edge inputs fail closed into non-equivalent canonical forms."""

    assert mod.canonicalize_answer("{bad").parse_status == "unparsed"
    assert mod.canonicalize_answer("not-an-answer").kind == "unknown"
    assert mod.canonicalize_answer(True).kind == "unknown"
    assert mod.canonicalize_answer("NaN").parse_status == "unparsed"
    assert mod.known_false_accept_rows(
        {
            "false_accept_row_ids": ["fallback"],
            "verifier_rows": [_false_row("fallback", "INVALID", "VALID")],
        }
    )[0]["row_id"] == "fallback"
    assert mod.honest_verdict({"canonical_grounding_pilot_v1_ready": False}).startswith("blocked_")
    assert mod.expected_action_from_label("MAYBE") == "abstain"
    assert mod.action_from_canonical(mod.canonicalize_answer("assertion_fails")) == "reject"
    assert mod.action_from_canonical(mod.canonicalize_answer("not-an-answer")) == "abstain"
    assert mod.label_family("REPAIRABLE") == "repairability_token"


def test_req_verify_3138_premise_records_mark_absent_grounding() -> None:
    """REQ-VERIFY-3138: absent and available premise grounding are explicit."""

    grounded = mod.premise_records_from_events(
        _monitor_events(
            "ok",
            "VALID",
            "VALID",
            "accept",
            ledger_action="accept",
            constraints=[
                {
                    "constraint_id": "ok:assertion",
                    "status": "pass",
                    "solver_evidence": {"computed_value": 5},
                }
            ],
        ),
        "ok",
    )
    absent = mod.premise_records_from_events(
        _monitor_events("missing", "INVALID", "VALID", "accept"),
        "missing",
    )

    assert grounded[0]["grounded"] is True
    assert grounded[0]["status"] == "pass"
    assert absent == [
        {
            "premise_id": "missing:premise_absent",
            "source": "monitor_ledger",
            "grounded": False,
            "status": "absent",
            "reason": "no_fragment_trace",
        }
    ]


def test_scenario_verify_3138_contradictions_are_blocked_by_all_gates() -> None:
    """SCENARIO-VERIFY-3138: canonical, premise, and ledger checks block false accepts."""

    analysis = mod.evaluate_regression_row(
        _false_row("fa-smt", "UNSAT", "VALID"),
        solver_certificate=_certificate("fa-smt", "UNSAT"),
        monitor_events=_monitor_events("fa-smt", "UNSAT", "VALID", "accept"),
        contract_replay={
            "decision": "abstain",
            "matched_rule_id": "ABSTAIN_KNOWN_FALSE_ACCEPT_REGRESSION",
        },
    )
    clean = mod.evaluate_regression_row(
        {
            "row_id": "ok",
            "fixture_id": "ok",
            "exact_label": "VALID",
            "expected_action": "accept",
            "extracted_answer": "VALID",
            "live_decision": "accept",
        },
        solver_certificate={
            "fixture_id": "ok",
            "exact_label": "VALID",
            "solver_label": "assertion_passes",
            "maxsat_route": {"action": "accept"},
        },
        monitor_events=_monitor_events(
            "ok",
            "VALID",
            "VALID",
            "accept",
            ledger_action="accept",
            constraints=[{"constraint_id": "ok:claim", "status": "pass"}],
        ),
    )

    assert analysis["canonicalization_blocked"] is True
    assert analysis["premise_grounding_blocked"] is True
    assert analysis["ledger_replay_blocked"] is True
    assert analysis["candidate_canonical"]["family"] == "validity_token"
    assert analysis["exact_canonical"]["family"] == "sat_token"
    assert analysis["premise_to_answer_consistent"] is False
    assert analysis["answer_to_premise_consistent"] is False
    assert analysis["solver_certificate_summary"]["certificate_consistent_with_exact"] is True
    assert set(analysis["blocked_by"]) == {
        "canonicalization",
        "premise_grounding",
        "ledger_replay",
    }
    assert clean["blocked_by"] == []


def test_req_verify_3138_builds_and_validates_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-3138: replay artifact quantifies blocked false-accept rows."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=5.0,
        now_s=7.25,
        tests_run=["REQ-VERIFY-3138 focused"],
    )
    rows = {row["row_id"]: row for row in artifact["regression_row_replay"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["canonical_grounding_pilot_v1_ready"] is True
    assert artifact["canonicalizer_implemented"] is True
    assert artifact["premise_grounding_rows"] == 2
    assert artifact["regression_rows_evaluated"] == 2
    assert artifact["false_accept_rows_blocked"] == 2
    assert artifact["canonicalization_block_count"] == 2
    assert artifact["premise_grounding_block_count"] == 2
    assert artifact["ledger_replay_block_count"] == 2
    assert artifact["residual_false_accept_rows"] == []
    assert artifact["tests_run"] == ["REQ-VERIFY-3138 focused"]
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["no_live_llm_inference"] is True
    assert artifact["inference_substrate"]["model_backed_verifier_invoked"] is False
    assert rows["fa-smt"]["contract_replay"]["decision"] == "abstain"
    assert rows["fa-arith"]["solver_certificate_summary"]["solver_authority"]
    mod.validate_artifact(artifact)


def test_req_verify_3138_write_artifact_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3138: writer persists JSON and validation blocks overclaiming."""

    _write_sources(tmp_path)

    out_path = mod.write_artifact(
        tmp_path,
        started_s=1.0,
        now_s=1.5,
        tests_run=["REQ-VERIFY-3138 focused"],
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["source_checksums"][mod.EXP3136_REL_PATH.as_posix()]

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="canonicalizer_implemented"):
        mod.validate_artifact(artifact | {"canonicalizer_implemented": False})
    with pytest.raises(ValueError, match="residual_false_accept_rows"):
        mod.validate_artifact(artifact | {"residual_false_accept_rows": ["fa-smt"]})
    with pytest.raises(ValueError, match="regression row count"):
        mod.validate_artifact(artifact | {"false_accept_rows_blocked": 1})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "ready: no"})
