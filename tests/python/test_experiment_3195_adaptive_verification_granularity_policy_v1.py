"""Tests for Exp 3195 adaptive verification granularity policy v1.

Spec refs: REQ-VERIFY-3195, SCENARIO-VERIFY-3195.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import adaptive_verification_granularity_policy_v1 as mod


REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "policy_version",
    "source_artifacts",
    "exact_rows_used",
    "policy_features",
    "policy_actions",
    "simulated_rows",
    "estimated_verifier_call_delta",
    "false_accept_risk_increase",
    "redundant_recheck_suppression_rule",
    "promotion_allowed",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "source\n") -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _record(
    row_id: str,
    *,
    family: str = "exact_row:arithmetic_code_assertions",
    label: str = "VALID",
    checker: str = "exact_authority_replay",
    result: str = "accept",
    answers: list[str] | None = None,
    known_false: bool = False,
    exact_complete: bool = True,
    pilot: dict[str, Any] | None = None,
    flagged: bool = False,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "counterexample_family": family,
        "exact_label": label,
        "checker_authority": checker,
        "checker_result": result,
        "candidate_answers": answers or [label],
        "known_false_accept_or_regression": known_false,
        "exact_authority_complete": exact_complete,
        "depends_on_flagged_live_verifier": flagged,
        "pilot_certificate": pilot or {},
        "source_artifact": "fixture.exp3183",
    }


def _write_common_sources(root: Path) -> None:
    records = [
        _record("row-clean-valid"),
        _record(
            "row-simple-invalid",
            label="INVALID",
            result="reject",
            answers=["INVALID"],
        ),
        _record(
            "row-known-false",
            family="known_false_accept:arithmetic_code_assertions",
            label="INVALID",
            checker="python_ast_runtime_execution",
            result="reject",
            answers=["INVALID", "VALID"],
            known_false=True,
            pilot={
                "certificate_type": "ast_execution",
                "minimal_failing_assignment": {"claimed_value": 47, "computed_value": 43},
                "mcs": {"kind": "replace_claimed_value", "from": 47, "to": 43},
                "unsat_core": ["computed_value", "claimed_value"],
            },
        ),
        _record(
            "row-repair-fragment",
            family="exact_row:json_fragment_repair",
            label="REPAIRABLE",
            result="reject",
            answers=["REPAIRABLE"],
        ),
        _record(
            "row-unknown-family",
            family="exact_row:unknown",
            label="REPAIRABLE",
            result="reject",
            answers=["REPAIRABLE"],
        ),
        _record(
            "row-drift-anchor",
            family="satisfiable_drift_anchor",
            label="SAT",
            checker="z3_solver",
            result="accept",
            answers=["SAT"],
            pilot={
                "certificate_type": "solver_mcs",
                "minimal_failing_assignment": {"model": "sat"},
                "mcs": {"kind": "preserve_satisfiable_anchor"},
            },
        ),
    ]
    receipts = [
        {
            "index": 0,
            "transcript_hash": "receipt-a",
            "acceptance_authority": False,
            "substrate_used": "cpu_fallback_receipt_only",
        },
        {
            "index": 1,
            "transcript_hash": "receipt-b",
            "acceptance_authority": False,
            "substrate_used": "cpu_fallback_receipt_only",
        },
    ]

    _write_text(root, "AGENTS.md", "Read CODEX.md before non-trivial changes.\n")
    _write_text(root, "CODEX.md", "Spec First\nWrite Tests First\n")
    _write_text(root, "CLAUDE.md", "Exact authority remains final.\n")
    _write_text(
        root,
        "research-references.md",
        "Variable Granularity Search routes final answer, step chunk, and counterexample fragment checks.\n",
    )
    _write_text(
        root,
        "openspec/capabilities/verification/spec.md",
        "REQ-VERIFY-3195\nSCENARIO-VERIFY-3195\n"
        "results/experiment_3195_adaptive_verification_granularity_policy_v1.json\n",
    )
    _write_json(
        root,
        mod.EXP3180_REL_PATH,
        {
            "controlled_invariance_executor_v2_ready": True,
            "controlled_invariance_passed": True,
            "exact_row_count": len(records),
            "known_false_accept_regression_count": 1,
            "known_false_accept_regression_ids": ["row-known-false"],
            "receipt_backed_transcript_count": len(receipts),
            "receipt_backed_transcripts": receipts,
            "exact_rows_evaluated": [
                {
                    "row_id": row["row_id"],
                    "exact_label": row["exact_label"],
                    "exact_authority_decision": row["checker_result"],
                    "known_false_accept_regression": row["known_false_accept_or_regression"],
                }
                for row in records
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3183_REL_PATH,
        {
            "counterexample_certificate_expansion_v3_ready": True,
            "exact_row_count": len(records),
            "counterexample_count": 2,
            "certificate_records": records,
            "bounded_frontier_records": [
                {"frontier_id": "f0", "exact_status": "viable"},
                {"frontier_id": "f1", "exact_status": "pruned"},
                {"frontier_id": "f2", "exact_status": "bounded"},
            ],
            "known_false_accept_rows_covered": 1,
            "known_false_accept_row_ids_expected": ["row-known-false"],
            "known_false_accept_row_ids_covered": ["row-known-false"],
            "flagged_adversarial": True,
            "repair_call_ready": False,
        },
    )
    _write_json(
        root,
        mod.EXP3189_REL_PATH,
        {
            "cross_corpus_matrix_v29_ready": True,
            "rows_total": 159,
            "status_counts": {
                "clean": 39,
                "flagged": 18,
                "blocked": 37,
                "gated_skipped": 14,
                "diagnostic_only": 9,
            },
            "publication_blocker_count": 80,
            "next_top_gap": "full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock",
            "verifier_status": "gated_skip_cpu_fallback_receipt_only_flagged_adversarial",
            "repair_status": "blocked_receipt_precondition",
        },
    )


def test_req_verify_3195_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3195: OpenSpec declares the adaptive policy artifact."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3195" in spec
    assert "SCENARIO-VERIFY-3195" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3195_routes_rows_without_new_authority(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3195: row/risk features choose deterministic granularity."""

    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        tests_run=["SCENARIO-VERIFY-3195 focused"],
    )
    rows = {row["row_id"]: row for row in artifact["simulated_policy_rows"]}

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3195"
    assert artifact["policy_version"] == "v1"
    assert artifact["exact_rows_used"] == 6
    assert artifact["simulated_rows"] == 6
    assert artifact["promotion_allowed"] is False
    assert artifact["honest_verdict"].startswith("complete:")

    assert artifact["policy_actions"] == list(mod.POLICY_ACTIONS)
    assert set(artifact["policy_features"]) == set(mod.POLICY_FEATURES)
    assert artifact["evidence_inventory"]["receipt_backed_transcripts"] == 2
    assert artifact["evidence_inventory"]["counterexample_certificates"] == 2
    assert artifact["evidence_inventory"]["bounded_frontier_records"] == 3
    assert artifact["evidence_inventory"]["false_accept_families"] == [
        "known_false_accept:arithmetic_code_assertions"
    ]

    assert rows["row-clean-valid"]["selected_action"] == "skip redundant recheck"
    assert rows["row-simple-invalid"]["selected_action"] == "final-answer-only"
    assert rows["row-known-false"]["selected_action"] == "counterexample-fragment"
    assert rows["row-repair-fragment"]["selected_action"] == "step-chunk"
    assert rows["row-unknown-family"]["selected_action"] == "abstain/escalate"
    assert rows["row-drift-anchor"]["selected_action"] == "counterexample-fragment"

    assert rows["row-known-false"]["feature_values"]["answer_ambiguity"] is True
    assert (
        rows["row-known-false"]["feature_values"]["certificate_depth"]
        > rows["row-clean-valid"]["feature_values"]["certificate_depth"]
    )
    assert rows["row-clean-valid"]["routing_reason"] == "exact_accept_recheck_redundant"

    assert artifact["schedule_counts"] == {
        "abstain/escalate": 1,
        "counterexample-fragment": 2,
        "final-answer-only": 1,
        "skip redundant recheck": 1,
        "step-chunk": 1,
    }
    accounting = artifact["verifier_call_accounting"]
    assert accounting["baseline_policy"] == "uniform_step_chunk"
    assert accounting["baseline_verifier_calls"] == 12
    assert accounting["adaptive_verifier_calls"] == 10
    assert artifact["estimated_verifier_call_delta"] == -2

    risk = artifact["risk_tradeoffs"]
    assert risk["known_false_accept_rows_skipped"] == 0
    assert risk["known_false_accept_rows_below_counterexample_fragment"] == 0
    assert risk["ambiguous_rows_below_counterexample_fragment"] == 0
    assert artifact["false_accept_risk_increase"] == 0.0

    rule = artifact["redundant_recheck_suppression_rule"]
    assert rule["action"] == "skip redundant recheck"
    assert "known_false_accept_risk=true" in rule["excluded_when"]
    assert rule["authority_preserved"] == "prior exact-authority outcome remains final"
    assert artifact["inference_substrate"]["new_live_model_calls"] == 0
    assert artifact["authority_boundary"]["ebm_or_llm_authority"] is False


def test_req_verify_3195_writer_and_validation_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-3195: writer persists JSON and validation rejects overclaiming."""

    _write_common_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        tests_run=["REQ-VERIFY-3195 writer"],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["experiment_id"] == "exp3195"
    assert saved["honest_verdict"].startswith("complete:")

    with pytest.raises(ValueError, match="promotion_allowed"):
        mod.validate_artifact(saved | {"promotion_allowed": True})

    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(saved | {"honest_verdict": "blocked: no"})

    bad_rows = [
        row | {"selected_action": "skip redundant recheck"}
        if row["row_id"] == "row-known-false"
        else row
        for row in saved["simulated_policy_rows"]
    ]
    with pytest.raises(ValueError, match="known false-accept"):
        mod.validate_artifact(saved | {"simulated_policy_rows": bad_rows})


def test_req_verify_3195_missing_sources_materialize_honest_empty_policy(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3195: malformed or missing inputs do not fabricate rows."""

    _write_text(tmp_path, "AGENTS.md", "Read CODEX.md\n")
    _write_text(tmp_path, "CODEX.md", "Spec First\n")
    _write_text(tmp_path, "CLAUDE.md", "Exact authority remains final.\n")
    _write_text(tmp_path, "research-references.md", "VG-Search\n")
    _write_text(tmp_path, "openspec/capabilities/verification/spec.md", "REQ-VERIFY-3195\n")
    _write_text(tmp_path, mod.EXP3183_REL_PATH, "not-json\n")

    artifact = mod.build_artifact(tmp_path)

    assert artifact["adaptive_verification_granularity_policy_v1_ready"] is True
    assert artifact["exact_rows_used"] == 0
    assert artifact["simulated_rows"] == 0
    assert artifact["estimated_verifier_call_delta"] == 0
    assert artifact["false_accept_risk_increase"] is None
    assert artifact["source_errors"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_3195_defensive_helpers_and_cli_edges(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-3195: defensive helper edges are explicit and covered."""

    assert mod.normalized_answers(None) == ["unknown"]
    assert mod.normalized_answers("VALID") == ["VALID"]
    assert mod.normalized_answers([""]) == ["unknown"]

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})

    valid = {
        "schema_version": mod.SCHEMA_VERSION,
        "experiment_id": "exp3195",
        "policy_version": "v1",
        "source_artifacts": [],
        "exact_rows_used": 0,
        "policy_features": list(mod.POLICY_FEATURES),
        "policy_actions": list(mod.POLICY_ACTIONS),
        "simulated_rows": 0,
        "estimated_verifier_call_delta": 0,
        "false_accept_risk_increase": None,
        "redundant_recheck_suppression_rule": mod.redundant_recheck_suppression_rule(),
        "promotion_allowed": False,
        "honest_verdict": "complete: fixture",
        "simulated_policy_rows": [],
    }

    with pytest.raises(ValueError, match="simulated_policy_rows"):
        mod.validate_artifact(valid | {"simulated_policy_rows": ["bad-row"]})

    with pytest.raises(ValueError, match="feature_values"):
        mod.validate_artifact(
            valid | {"simulated_policy_rows": [{"selected_action": "final-answer-only"}]}
        )

    with pytest.raises(ValueError, match="redundant recheck"):
        mod.validate_artifact(
            valid
            | {
                "simulated_policy_rows": [
                    {
                        "selected_action": "skip redundant recheck",
                        "feature_values": {
                            "known_false_accept_risk": False,
                            "answer_ambiguity": True,
                            "exact_authority_complete": True,
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(mod, "write_artifact", lambda: Path("results/fake-exp3195.json"))
    mod.main()
    assert capsys.readouterr().out.strip() == "results/fake-exp3195.json"
