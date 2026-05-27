"""Tests for Exp 3180 controlled-invariance executor v2.

Spec refs: REQ-VERIFY-3180, SCENARIO-VERIFY-3180.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import controlled_invariance_executor_v2 as mod


REQUIRED_FIELDS = {
    "controlled_invariance_executor_v2_ready",
    "exact_row_count",
    "receipt_backed_transcript_count",
    "control_results",
    "shortcut_failure_count",
    "known_false_accept_regression_count",
    "token_suspicion_used_as_triage_only",
    "controlled_invariance_passed",
    "blocker_reasons",
    "inference_substrate",
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


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "row_id": "row-valid",
            "exact_label": "VALID",
            "expected_action": "accept",
            "extracted_answer": "VALID",
            "candidate_answers": ["VALID"],
            "fixture_family": "arithmetic_code_assertions",
        },
        {
            "row_id": "row-known-false-accept",
            "exact_label": "INVALID",
            "expected_action": "reject",
            "extracted_answer": "VALID",
            "candidate_answers": ["VALID"],
            "fixture_family": "contradiction",
            "answer_extraction_format": "validity_token",
        },
        {
            "row_id": "row-sat",
            "exact_label": "SAT",
            "expected_action": "accept",
            "extracted_answer": "SAT",
            "candidate_answers": ["SAT"],
            "fixture_family": "smt_constraints",
        },
    ]


def _receipts(*, duplicate: bool = False) -> list[dict[str, Any]]:
    second_hash = "receipt-a" if duplicate else "receipt-b"
    return [
        {
            "selected_model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "prompt_hash": "prompt-a",
            "response_hash": "response-a",
            "transcript_hash": "receipt-a",
            "token_counts": {"prompt_tokens": 6, "completion_tokens": 1, "total_tokens": 7},
            "substrate_used": "cpu_fallback_receipt_only",
            "subprocess_return_code": 0,
        },
        {
            "selected_model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "prompt_hash": "prompt-b",
            "response_hash": "response-b",
            "transcript_hash": second_hash,
            "token_counts": {"prompt_tokens": 6, "completion_tokens": 1, "total_tokens": 7},
            "substrate_used": "cpu_fallback_receipt_only",
            "subprocess_return_code": 0,
        },
    ]


def _write_docs(root: Path) -> None:
    _write_text(root, "AGENTS.md", "Read CODEX.md before non-trivial changes.\n")
    _write_text(root, "CODEX.md", "Spec First\nWrite Tests First\n")
    _write_text(root, "CLAUDE.md", "Token suspicion is triage only.\n")
    _write_text(root, "scripts/experiment_template.py", "DEFAULT_BATCH_SIZE = 8\n")
    _write_text(
        root,
        "openspec/capabilities/verification/spec.md",
        "REQ-VERIFY-3180\nSCENARIO-VERIFY-3180\n"
        "results/experiment_3180_controlled_invariance_executor_v2.json\n"
        "force-answer remove-answer shuffled-trace answer-only trace-only transcript-hash\n",
    )


def _write_common_sources(
    root: Path,
    *,
    receipts: list[dict[str, Any]] | None = None,
    token_suspicion_may_accept: bool = False,
    include_regression: bool = True,
) -> None:
    rows = _rows()
    regression_ids = ["row-known-false-accept"] if include_regression else []
    _write_docs(root)
    _write_json(
        root,
        mod.EXP3136_REL_PATH,
        {
            "false_accept_autopsy_v1_ready": True,
            "false_accept_row_ids": regression_ids,
            "regression_row_set": regression_ids,
            "false_accept_rows": [rows[1]] if include_regression else [],
            "verifier_rows": rows,
            "honest_verdict": "complete: fixture",
        },
    )
    _write_json(
        root,
        mod.EXP3137_REL_PATH,
        {
            "acceptance_contract_v1_ready": True,
            "known_false_accept_rows_blocked": include_regression,
            "regression_row_set": regression_ids,
            "replay_rows": [
                row
                | {
                    "decision": "reject" if row["row_id"] == "row-known-false-accept" else "accept",
                    "matched_rule_id": "REJECT_EXACT_REJECT_MATCH"
                    if row["row_id"] == "row-known-false-accept"
                    else "ACCEPT_EXACT_COVERED_CONSISTENT",
                }
                for row in rows
            ],
            "honest_verdict": "complete: fixture",
        },
    )
    _write_json(
        root,
        mod.EXP3138_REL_PATH,
        {
            "canonical_grounding_pilot_v1_ready": True,
            "false_accept_rows_blocked": 1 if include_regression else 0,
            "regression_rows_evaluated": 1 if include_regression else 0,
            "residual_false_accept_rows": [],
            "regression_row_replay": [
                {
                    "row_id": "row-known-false-accept",
                    "exact_label": "INVALID",
                    "expected_action": "reject",
                    "candidate_answer": "VALID",
                    "canonical_equivalent": False,
                }
            ]
            if include_regression
            else [],
            "honest_verdict": "complete: fixture",
        },
    )
    _write_json(
        root,
        mod.EXP3166_REL_PATH,
        {
            "verifier_invariance_token_suspicion_audit_ready": True,
            "controlled_invariance_checks": [
                {
                    "name": name,
                    "routes_to_exact_checks": True,
                    "can_authorize_acceptance": False,
                }
                for name in ("force", "remove", "shuffled_trace", "answer_only", "trace_only")
            ],
            "token_suspicion_fields": [
                {
                    "name": "first_token_entropy",
                    "acceptance_authority": False,
                    "may_accept": token_suspicion_may_accept,
                    "may_route_exact_checks": True,
                }
            ],
            "trusted_exact_rows": rows,
            "honest_verdict": "complete: fixture",
        },
    )
    _write_json(
        root,
        mod.EXP3167_REL_PATH,
        {
            "clean_live_verifier_rerun_v9_ready": True,
            "gated_skip": True,
            "exact_ground_truth_count": len(rows),
            "regression_rows_included": include_regression,
            "controlled_invariance_passed": False,
            "planned_rerun_set": {
                "row_ids": [row["row_id"] for row in rows],
                "regression_row_ids": regression_ids,
            },
            "inference_substrate": {"live_model_calls": 0, "executes_models": False},
            "honest_verdict": "complete: fixture",
        },
    )
    proof_receipts = _receipts() if receipts is None else receipts
    _write_json(
        root,
        mod.EXP3179_REL_PATH,
        {
            "local_sota_receipt_smoke_v3_ready": True,
            "preflight_passed": True,
            "live_call_count": len(proof_receipts),
            "proof_receipts": proof_receipts,
            "transcript_hashes": [row["transcript_hash"] for row in proof_receipts],
            "clean_rerun_allowed": False,
            "inference_substrate": {"live_model_calls": len(proof_receipts)},
            "honest_verdict": "complete: fixture",
        },
    )


def test_req_verify_3180_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3180: OpenSpec declares the executor before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3180" in spec
    assert "SCENARIO-VERIFY-3180" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "controlled-invariance executor" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3180_executes_all_controls_with_exact_authority(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3180: exact authority executes controls without live calls."""

    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=13.25,
        tests_run=["SCENARIO-VERIFY-3180 focused"],
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["controlled_invariance_executor_v2_ready"] is True
    assert artifact["exact_row_count"] == 3
    assert artifact["receipt_backed_transcript_count"] == 2
    assert artifact["known_false_accept_regression_count"] == 1
    assert artifact["shortcut_failure_count"] == 0
    assert artifact["semantic_false_accept_count"] == 0
    assert artifact["token_suspicion_used_as_triage_only"] is True
    assert artifact["controlled_invariance_passed"] is True
    assert artifact["blocker_reasons"] == []
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["tests_run"] == ["SCENARIO-VERIFY-3180 focused"]
    assert artifact["honest_verdict"].startswith("complete:")

    assert set(artifact["control_results"]) == set(mod.CONTROL_NAMES)
    assert all(row["executed"] is True for row in artifact["control_results"].values())
    assert all(row["passed"] is True for row in artifact["control_results"].values())
    assert artifact["control_results"]["answer_only"]["semantic_false_accept_count"] == 0
    assert artifact["control_results"]["trace_only"]["shortcut_failure_count"] == 0
    assert artifact["regression_row_results"][0]["exact_authority_decision"] == "reject"
    assert artifact["inference_substrate"]["new_live_model_calls"] == 0
    assert artifact["inference_substrate"]["executes_models"] is False

    output = mod.write_artifact(
        tmp_path,
        started_s=20.0,
        now_s=21.0,
        tests_run=["REQ-VERIFY-3180 write"],
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["controlled_invariance_executor_v2_ready"] is True
    assert written["tests_run"] == ["REQ-VERIFY-3180 write"]


def test_req_verify_3180_token_suspicion_cannot_accept(tmp_path: Path) -> None:
    """REQ-VERIFY-3180: token suspicion features are triage only, never authority."""

    _write_common_sources(tmp_path, token_suspicion_may_accept=True)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["token_suspicion_used_as_triage_only"] is False
    assert artifact["controlled_invariance_passed"] is False
    assert artifact["control_results"]["token_suspicion_triage"]["passed"] is False
    assert any("token suspicion" in reason for reason in artifact["blocker_reasons"])
    assert artifact["semantic_false_accept_count"] == 0


def test_req_verify_3180_known_false_accept_regressions_are_load_bearing(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3180: known false-accept regressions cannot be silently absent."""

    _write_common_sources(tmp_path, include_regression=False)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["known_false_accept_regression_count"] == 0
    assert artifact["controlled_invariance_passed"] is False
    assert any("known false-accept" in reason for reason in artifact["blocker_reasons"])


def test_req_verify_3180_transcript_hash_shortcut_failure_is_not_semantic_false_accept(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3180: transcript shortcut failures are separate from semantics."""

    _write_common_sources(tmp_path, receipts=_receipts(duplicate=True))

    artifact = mod.build_artifact(tmp_path)

    assert artifact["control_results"]["transcript_hash"]["passed"] is False
    assert artifact["shortcut_failure_count"] == 1
    assert artifact["semantic_false_accept_count"] == 0
    assert artifact["controlled_invariance_passed"] is False
    assert any("transcript hash" in reason for reason in artifact["blocker_reasons"])


def test_req_verify_3180_defensive_branches_and_validation_guards(tmp_path: Path) -> None:
    """REQ-VERIFY-3180: helper guards preserve blocked artifacts as explicit failures."""

    _write_common_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path)

    fallback_receipts = mod.collect_receipt_backed_transcripts(
        {
            "proof_receipts": [{}],
            "transcript_hashes": [{"transcript_hash": "fallback-a"}, "fallback-b"],
        }
    )
    assert [row["transcript_hash"] for row in fallback_receipts] == ["fallback-a", "fallback-b"]
    assert mod.token_suspicion_is_triage_only({}) is False
    assert mod.exact_authority_decision({"exact_label": "SAT"}) == "accept"
    assert mod.exact_authority_decision({"exact_label": "INVALID"}) == "reject"
    assert mod.exact_authority_decision({"exact_label": "UNKNOWN"}) == "abstain"
    assert mod.honest_verdict(
        {"controlled_invariance_executor_v2_ready": False, "blocker_reasons": ["missing"]}
    ).startswith("blocked_precondition:")

    conflict_rows = mod.collect_exact_rows(
        {
            "exp3166": {"trusted_exact_rows": [{"row_id": "conflict", "exact_label": "VALID"}]},
            "exp3167": {"exact_rows_evaluated": [{"row_id": "", "exact_label": ""}]},
            "exp3137": {
                "replay_rows": [
                    {"row_id": "conflict", "exact_label": "INVALID", "expected_action": "reject"}
                ]
            },
            "exp3136": {},
            "exp3138": {},
        }
    )
    assert conflict_rows[0]["exact_label_conflict"] is True
    assert conflict_rows[0]["expected_action"] == "reject"

    blockers = mod.blockers(
        source_errors=[{"path": "missing"}],
        exact_rows=[],
        regression_ids=["missing-regression"],
        regression_results=[],
        control_results={"force_answer": {"passed": False}},
        token_triage_only=False,
        shortcut_failure_count=1,
        semantic_false_accept_count=1,
    )
    assert "required source artifacts are missing or malformed" in blockers
    assert "not all known false-accept regression rows were included" in blockers
    assert "semantic false accept count=1" in blockers

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})

    bad_verdict = dict(artifact, honest_verdict="not-terminal")
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(bad_verdict)

    bad_inference = dict(
        artifact,
        inference_substrate=dict(artifact["inference_substrate"], new_live_model_calls=1),
    )
    with pytest.raises(ValueError, match="must not make new live model calls"):
        mod.validate_artifact(bad_inference)

    bad_blocker = dict(artifact, blocker_reasons=["late blocker"])
    with pytest.raises(ValueError, match="must not contain blockers"):
        mod.validate_artifact(bad_blocker)

    bad_shortcut = dict(artifact, shortcut_failure_count=1)
    with pytest.raises(ValueError, match="zero shortcut failures"):
        mod.validate_artifact(bad_shortcut)

    bad_semantic = dict(artifact, semantic_false_accept_count=1)
    with pytest.raises(ValueError, match="zero semantic false accepts"):
        mod.validate_artifact(bad_semantic)
