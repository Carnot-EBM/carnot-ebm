"""Tests for Exp 3112 logic-regularized verifier pilot.

Spec refs: REQ-VERIFY-3112, SCENARIO-VERIFY-3112.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import logic_regularized_verifier_pilot_v1 as mod


REQUIRED_FIELDS = {
    "logic_regularized_verifier_pilot_ready",
    "model_specs",
    "mandatory_headline_model_ids",
    "selected_headline_model_ids",
    "live_llm_inference",
    "exact_ground_truth_count",
    "negation_consistency_rate",
    "answer_group_consistency_rate",
    "verifier_recall_delta",
    "false_positive_delta",
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


def _manifest_row(
    fixture_id: str,
    *,
    expected_answer: str,
    perturbation: str,
    expected_action: str,
) -> dict[str, Any]:
    return {
        "schema": "carnot.exact_fixture_eval_manifest.v1",
        "source_fixture_id": fixture_id,
        "source_prompt_payload_sha256": f"{fixture_id}-hash",
        "task_family": "smt_constraints" if expected_answer in {"SAT", "UNSAT"} else "arithmetic_code_assertions",
        "task_axis": "verifying",
        "perturbation_type": perturbation,
        "expected_answer": expected_answer,
        "solver_label": expected_answer.lower(),
        "label_source": "unit_exact_authority",
        "exact_label_kind": "unit",
        "leakage_safe_prompt_payload": {"fixture": fixture_id},
        "verifier_target": {
            "expected_action": expected_action,
            "expected_reject": expected_action == "reject",
        },
        "repair_target": {"applicable": False, "reason": "not_a_repair_fixture"},
        "evaluation_tasks": ["logic_regularized_verifier_pilot_v1"],
        "stratum_key": f"{perturbation}|{expected_answer}",
    }


def _certificate(
    fixture_id: str,
    *,
    exact_label: str,
    coherence_status: str,
    route_action: str,
) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "exact_label": exact_label,
        "solver_label": exact_label.lower(),
        "solver_authority": "unit_exact_authority",
        "task_family": "unit",
        "perturbation_type": "unit",
        "coherence_status": coherence_status,
        "coherence_gap": 0 if coherence_status == "coherent" else 1,
        "diagnostics": {"unit": fixture_id},
        "unsat_core": [] if coherence_status == "coherent" else ["expected", "claimed"],
        "minimal_correction_set": {}
        if coherence_status == "coherent"
        else {"kind": "replace_claimed_value"},
        "repair_distance": 0 if coherence_status == "coherent" else 1,
        "maxsat_route": {
            "action": route_action,
            "hard_constraints": [{"id": "HC_EXACT_LABEL_PRESENT", "satisfied": True}],
            "soft_constraints": [{"id": "SC_EXACT_LABEL_MATCH", "satisfied": True}],
            "soft_score": 100,
        },
    }


def _panel_row(
    fixture_id: str,
    *,
    expected_answer: str,
    expected_action: str,
    route_decision: str,
    exact_answer_match: bool,
) -> dict[str, Any]:
    return {
        "row_index": 0,
        "source_fixture_id": fixture_id,
        "task_family": "unit",
        "perturbation_type": "unit",
        "expected_answer": expected_answer,
        "expected_action": expected_action,
        "parsed_answer": expected_answer if exact_answer_match else None,
        "exact_answer_match": exact_answer_match,
        "route_decision": route_decision,
        "route_scores": {"accept": 100, "reject": 120, "abstain": 80},
        "maxsat_policy_used": True,
    }


def _fixture_rows() -> list[dict[str, Any]]:
    return [
        _manifest_row(
            "case-valid",
            expected_answer="VALID",
            perturbation="arithmetic_true_verification",
            expected_action="accept",
        ),
        _manifest_row(
            "case-invalid",
            expected_answer="INVALID",
            perturbation="arithmetic_false_verification",
            expected_action="reject",
        ),
        _manifest_row(
            "case-sat",
            expected_answer="SAT",
            perturbation="smt_sat_solving",
            expected_action="accept",
        ),
        _manifest_row(
            "case-unsat",
            expected_answer="UNSAT",
            perturbation="smt_unsat_abstention",
            expected_action="reject",
        ),
    ]


def _write_sources(root: Path) -> None:
    rows = _fixture_rows()
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("No fake verifier recovery\n", encoding="utf-8")
    (root / "research-references.md").write_text("LOVER arXiv:2605.05893\n", encoding="utf-8")
    _write_jsonl(root, mod.MANIFEST_REL_PATH, rows)
    _write_json(
        root,
        mod.EXP3097_REL_PATH,
        {
            "artifact": "experiment_3097_exact_fixture_eval_protocol_audit_v1",
            "eval_protocol_ready": True,
            "usable_fixture_count": len(rows),
            "stratified_eval_manifest_path": mod.MANIFEST_REL_PATH.as_posix(),
            "honest_verdict": "complete: eval_protocol_ready=true",
        },
    )
    panel_rows = [
        _panel_row(
            "case-valid",
            expected_answer="VALID",
            expected_action="accept",
            route_decision="reject",
            exact_answer_match=False,
        ),
        _panel_row(
            "case-invalid",
            expected_answer="INVALID",
            expected_action="reject",
            route_decision="reject",
            exact_answer_match=True,
        ),
        _panel_row(
            "case-sat",
            expected_answer="SAT",
            expected_action="accept",
            route_decision="reject",
            exact_answer_match=False,
        ),
        _panel_row(
            "case-unsat",
            expected_answer="UNSAT",
            expected_action="reject",
            route_decision="reject",
            exact_answer_match=True,
        ),
    ]
    _write_jsonl(root, mod.EXP3099_ROWS_REL_PATH, panel_rows)
    _write_json(
        root,
        mod.EXP3099_REL_PATH,
        {
            "artifact": "experiment_3099_local_sota_confidence_abstention_panel_v3",
            "abstention_panel_v3_ready": True,
            "panel_rows_path": mod.EXP3099_ROWS_REL_PATH.as_posix(),
            "exact_ground_truth_count": len(rows),
            "false_accept_rate": 0.0,
            "false_reject_rate": 1.0,
            "rejection_recall": 1.0,
            "model_specs": [
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "cache_status": "cached",
                    "selected": True,
                    "model_path": "/tmp/gemma.gguf",
                }
            ],
            "selected_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "inference_substrate": {"executes_models": True},
            "honest_verdict": "complete: abstention_panel_v3_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3110_REL_PATH,
        {
            "artifact": "experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1",
            "mandatory_headline_model_ids": list(mod.MANDATORY_MODEL_IDS),
            "selected_headline_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "present_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "sota_model_manifest_ready": True,
            "honest_verdict": "complete: sota_model_manifest_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3111_REL_PATH,
        {
            "artifact": "experiment_3111_certified_coherence_z3_mcs_feedback_v3",
            "certified_coherence_feedback_v3_ready": True,
            "certificates": [
                _certificate(
                    "case-valid",
                    exact_label="VALID",
                    coherence_status="coherent",
                    route_action="accept",
                ),
                _certificate(
                    "case-invalid",
                    exact_label="INVALID",
                    coherence_status="incoherent",
                    route_action="reject",
                ),
                _certificate(
                    "case-sat",
                    exact_label="SAT",
                    coherence_status="coherent",
                    route_action="accept",
                ),
                _certificate(
                    "case-unsat",
                    exact_label="UNSAT",
                    coherence_status="incoherent",
                    route_action="reject",
                ),
            ],
            "honest_verdict": "complete: certified_coherence_feedback_v3_ready=true",
        },
    )


def test_req_verify_3112_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3112: OpenSpec declares the pilot before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3112" in spec
    assert "SCENARIO-VERIFY-3112" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "logic_regularized_verifier_pilot_ready" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3112_scores_exact_contrastive_paths(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3112: exact fixtures produce logic movement diagnostics."""

    _write_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        rows_path=tmp_path / mod.ROWS_REL_PATH,
        min_exact_count=4,
        started_s=10.0,
        now_s=12.0,
        tests_run=["focused-unit"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    rows = mod.read_jsonl_rows(tmp_path / artifact["diagnostic_rows_path"])

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["logic_regularized_verifier_pilot_ready"] is True
    assert artifact["promotion_claim_made"] is False
    assert artifact["live_llm_inference"] is False
    assert artifact["exact_ground_truth_count"] == 4
    assert artifact["path_count"] == 12
    assert artifact["negation_consistency_rate"] == pytest.approx(1.0)
    assert artifact["intra_answer_group_consistency_rate"] == pytest.approx(1.0)
    assert artifact["inter_answer_group_consistency_rate"] == pytest.approx(1.0)
    assert artifact["answer_group_consistency_rate"] == pytest.approx(1.0)
    assert artifact["exact_label_agreement_rate"] == pytest.approx(1.0)
    assert artifact["baseline_metrics"]["recall"] == pytest.approx(0.0)
    assert artifact["pilot_metrics"]["recall"] == pytest.approx(1.0)
    assert artifact["verifier_recall_delta"] == pytest.approx(1.0)
    assert artifact["false_positive_delta"] == pytest.approx(0.0)
    assert artifact["false_negative_movement"]["delta_count"] == -2
    assert artifact["false_positive_movement"]["delta_count"] == 0
    assert artifact["selected_headline_model_ids"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["tests_run"] == ["focused-unit"]
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(rows) == 4
    assert all(row["certified_feedback_v3_fields_present"] is True for row in rows)
    assert all(len(row["candidate_paths"]) == 3 for row in rows)
    assert rows[0]["candidate_paths"][0]["path_id"] == "certified_exact_path"
    mod.validate_artifact(artifact)


def test_req_verify_3112_validation_and_blocked_paths(tmp_path: Path) -> None:
    """REQ-VERIFY-3112: missing authorities fail closed with explicit diagnostics."""

    blocked = mod.build_artifact(tmp_path, min_exact_count=4, tests_run=["blocked-test"])

    assert blocked["logic_regularized_verifier_pilot_ready"] is False
    assert blocked["exact_ground_truth_count"] == 0
    assert blocked["blocked_reasons"]
    assert blocked["honest_verdict"].startswith("blocked_logic_regularized_verifier_pilot")
    mod.validate_artifact(blocked)

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, min_exact_count=4, tests_run=["validate-test"])
    relative_rows = mod.build_artifact(
        tmp_path,
        rows_path=mod.ROWS_REL_PATH,
        min_exact_count=4,
    )
    relative_output = mod.write_artifact(
        tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        rows_path=mod.ROWS_REL_PATH,
        min_exact_count=4,
    )

    assert relative_rows["diagnostic_rows_path"] == mod.ROWS_REL_PATH.as_posix()
    assert relative_output == tmp_path / mod.OUTPUT_REL_PATH

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="success prefix"):
        mod.validate_artifact(artifact | {"honest_verdict": "ready"})
    with pytest.raises(ValueError, match="finite rate"):
        mod.validate_artifact(artifact | {"negation_consistency_rate": 2.0})
    with pytest.raises(ValueError, match="promotion"):
        mod.validate_artifact(artifact | {"promotion_claim_made": True})
    with pytest.raises(ValueError, match="model_specs"):
        mod.validate_artifact(artifact | {"model_specs": []})
    with pytest.raises(ValueError, match="blocked"):
        mod.validate_artifact(artifact | {"logic_regularized_verifier_pilot_ready": False})
    with pytest.raises(ValueError, match="blocked_reasons"):
        mod.validate_artifact(
            blocked
            | {
                "honest_verdict": "blocked_logic_regularized_verifier_pilot: x",
                "blocked_reasons": [],
            }
        )


def test_req_verify_3112_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3112: helper parsing and movement math stay deterministic."""

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_json) == {}
    malformed_jsonl = '\nnot-json\n{"source_fixture_id": "ok"}\n[]\n'
    assert mod.read_jsonl_rows_from_text(malformed_jsonl) == [{"source_fixture_id": "ok"}]
    assert mod.read_jsonl_rows(tmp_path / "missing.jsonl") == []
    assert mod.contrastive_answer("VALID") == "INVALID"
    assert mod.contrastive_answer("UNSAT") == "SAT"
    assert mod.contrastive_answer("REPAIRABLE") == "UNKNOWN"
    assert mod.expected_action_from_answer("SAT") == "accept"
    assert mod.expected_action_from_answer("INVALID") == "reject"
    assert mod.expected_action_from_answer("UNKNOWN") == "abstain"
    assert mod.rate(0, 0) == 0.0
    assert mod.relative_path(tmp_path, tmp_path / "nested" / "rows.jsonl") == "nested/rows.jsonl"
    assert mod.relative_path(tmp_path, Path("/outside/rows.jsonl")) == "/outside/rows.jsonl"

    case = mod.score_case(
        _manifest_row(
            "edge",
            expected_answer="VALID",
            perturbation="arithmetic_true_verification",
            expected_action="accept",
        ),
        _certificate(
            "edge",
            exact_label="VALID",
            coherence_status="coherent",
            route_action="accept",
        ),
        _panel_row(
            "edge",
            expected_answer="VALID",
            expected_action="accept",
            route_decision="accept",
            exact_answer_match=True,
        ),
    )
    assert case["logic_decision"] == "accept"
    assert mod.case_rates([case])["negation_consistency_rate"] == pytest.approx(1.0)
    assert mod.movement_summary([case])["recall_delta"] == pytest.approx(0.0)
