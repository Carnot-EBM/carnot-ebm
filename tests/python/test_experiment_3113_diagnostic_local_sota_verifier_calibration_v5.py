"""Tests for Exp 3113 diagnostic local SOTA verifier calibration v5.

Spec refs: REQ-VERIFY-3113, SCENARIO-VERIFY-3113.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import diagnostic_local_sota_verifier_calibration_v5 as mod


REQUIRED_FIELDS = {
    "diagnostic_verifier_calibration_v5_ready",
    "model_specs",
    "mandatory_headline_model_ids",
    "selected_headline_model_ids",
    "live_llm_inference",
    "exact_ground_truth_count",
    "verifier_gain_delta",
    "verifier_gain_delta_with_certified_coherence",
    "false_accept_rate",
    "false_reject_rate",
    "calibration_error",
    "abstention_precision",
    "rejection_recall",
    "repair_gate_state",
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


def _manifest_row(fixture_id: str, answer: str, action: str) -> dict[str, Any]:
    return {
        "schema": "carnot.exact_fixture_eval_manifest.v1",
        "source_fixture_id": fixture_id,
        "source_prompt_payload_sha256": f"{fixture_id}-hash",
        "task_family": "smt_constraints" if answer in {"SAT", "UNSAT"} else "unit",
        "task_axis": "verifying",
        "perturbation_type": f"unit_{answer.lower()}",
        "expected_answer": answer,
        "solver_label": answer.lower(),
        "label_source": "unit_exact_authority",
        "exact_label_kind": "unit",
        "leakage_safe_prompt_payload": {"fixture": fixture_id},
        "verifier_target": {"expected_action": action, "expected_reject": action == "reject"},
        "repair_target": {"applicable": False},
        "evaluation_tasks": ["diagnostic_local_sota_verifier_calibration_v5"],
        "stratum_key": f"unit|{answer}",
    }


def _panel_row(fixture_id: str, answer: str, action: str, route: str) -> dict[str, Any]:
    return {
        "row_index": 0,
        "source_fixture_id": fixture_id,
        "task_family": "unit",
        "perturbation_type": "unit",
        "expected_answer": answer,
        "expected_action": action,
        "parsed_answer": answer if route == action else None,
        "exact_answer_match": route == action,
        "route_decision": route,
        "route_scores": {"accept": 20, "reject": 80},
        "maxsat_policy_used": True,
    }


def _certificate(fixture_id: str, answer: str, route: str) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "exact_label": answer,
        "solver_label": answer.lower(),
        "solver_authority": "unit_exact_authority",
        "task_family": "unit",
        "perturbation_type": "unit",
        "coherence_status": "coherent" if route != "reject" or answer in {"INVALID", "UNSAT"} else "incoherent",
        "coherence_gap": 0,
        "diagnostics": {"unit": fixture_id},
        "unsat_core": [],
        "minimal_correction_set": {},
        "repair_distance": 0,
        "maxsat_route": {"action": route, "soft_score": 100},
    }


def _logic_row(fixture_id: str, answer: str, action: str, baseline: str, logic: str) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "task_family": "unit",
        "perturbation_type": "unit",
        "expected_answer": answer,
        "exact_label": answer,
        "expected_action": action,
        "baseline_decision": baseline,
        "logic_decision": logic,
        "coherence_status": "coherent",
        "certified_feedback_v3_fields_present": True,
        "candidate_paths": [
            {"path_id": "certified_exact_path", "answer_group": action, "label_agrees": True},
            {"path_id": "contrastive_negation_path", "answer_group": baseline, "label_agrees": False},
        ],
    }


def _fixture_rows() -> list[tuple[str, str, str]]:
    return [
        ("case-valid", "VALID", "accept"),
        ("case-invalid", "INVALID", "reject"),
        ("case-sat", "SAT", "accept"),
        ("case-unsat", "UNSAT", "reject"),
    ]


def _write_sources(
    root: Path,
    *,
    selected_models: bool = True,
    logic_matches_exact: bool = True,
    certified_matches_exact: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = _fixture_rows()
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("No fake verifier recovery\n", encoding="utf-8")
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/experiment_template.py").write_text("def cached_sota_pair(): pass\n", encoding="utf-8")
    _write_jsonl(root, mod.MANIFEST_REL_PATH, [_manifest_row(*row) for row in rows])
    _write_json(
        root,
        mod.EXP3097_REL_PATH,
        {
            "artifact": "experiment_3097_exact_fixture_eval_protocol_audit_v1",
            "eval_protocol_ready": True,
            "usable_fixture_count": len(rows),
            "minimum_live_eval_count": len(rows),
            "stratified_eval_manifest_path": mod.MANIFEST_REL_PATH.as_posix(),
            "honest_verdict": "complete: eval_protocol_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3098_REL_PATH,
        {
            "artifact": "experiment_3098_maxsat_abstention_routing_policy_v1",
            "maxsat_policy_ready": True,
            "routing_policy_path": mod.POLICY_REL_PATH.as_posix(),
            "hard_constraints": [{"id": "HC_EXACT_LABEL_AGREEMENT"}],
            "soft_constraints": [{"id": "SC_MINIMIZE_FALSE_REJECTS", "weight": 40}],
            "honest_verdict": "complete: maxsat_policy_ready=true",
        },
    )
    _write_json(
        root,
        mod.POLICY_REL_PATH,
        {"schema": "unit.maxsat.policy", "fallback_evaluator": "deterministic"},
    )
    panel_rows = [_panel_row(fixture_id, answer, action, "reject") for fixture_id, answer, action in rows]
    _write_jsonl(root, mod.EXP3099_ROWS_REL_PATH, panel_rows)
    model_specs = [
        {
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "cache_status": "cached" if selected_models else "cache_missing",
            "cache_present": selected_models,
            "selected": selected_models,
            "model_path": "/tmp/gemma.gguf" if selected_models else None,
        }
    ]
    _write_json(
        root,
        mod.EXP3099_REL_PATH,
        {
            "artifact": "experiment_3099_local_sota_confidence_abstention_panel_v3",
            "abstention_panel_v3_ready": True,
            "panel_rows_path": mod.EXP3099_ROWS_REL_PATH.as_posix(),
            "exact_ground_truth_count": len(rows),
            "model_specs": model_specs,
            "selected_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"] if selected_models else [],
            "models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"] if selected_models else [],
            "inference_substrate": {"executes_models": selected_models, "live_llm_calls_planned": len(rows)},
            "honest_verdict": "complete: abstention_panel_v3_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3110_REL_PATH,
        {
            "artifact": "experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1",
            "sota_model_manifest_ready": True,
            "mandatory_headline_model_ids": list(mod.MANDATORY_MODEL_IDS),
            "selected_headline_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"] if selected_models else [],
            "present_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"] if selected_models else [],
            "honest_verdict": "complete: sota_model_manifest_ready=true",
        },
    )
    certificates = []
    logic_rows = []
    for fixture_id, answer, action in rows:
        exact_or_reject = action if certified_matches_exact else "reject"
        logic_or_reject = action if logic_matches_exact else "reject"
        certificates.append(_certificate(fixture_id, answer, exact_or_reject))
        logic_rows.append(_logic_row(fixture_id, answer, action, "reject", logic_or_reject))
    _write_json(
        root,
        mod.EXP3111_REL_PATH,
        {
            "artifact": "experiment_3111_certified_coherence_z3_mcs_feedback_v3",
            "certified_coherence_feedback_v3_ready": True,
            "exact_ground_truth_count": len(rows),
            "certificates": certificates,
            "honest_verdict": "complete: certified_coherence_feedback_v3_ready=true",
        },
    )
    _write_jsonl(root, mod.EXP3112_ROWS_REL_PATH, logic_rows)
    _write_json(
        root,
        mod.EXP3112_REL_PATH,
        {
            "artifact": "experiment_3112_logic_regularized_verifier_pilot_v1",
            "logic_regularized_verifier_pilot_ready": True,
            "diagnostic_rows_path": mod.EXP3112_ROWS_REL_PATH.as_posix(),
            "exact_ground_truth_count": len(logic_rows),
            "verifier_recall_delta": 1.0 if logic_matches_exact else 0.0,
            "honest_verdict": "complete: logic_regularized_verifier_pilot_ready=true",
        },
    )


def test_req_verify_3113_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3113: OpenSpec declares the calibration before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3113" in spec
    assert "SCENARIO-VERIFY-3113" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "repair_gate_state" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3113_builds_unblocked_diagnostic_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3113: positive certified lift unblocks the repair gate."""

    _write_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        min_exact_count=4,
        started_s=10.0,
        now_s=12.25,
        tests_run=["focused-unit"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["diagnostic_verifier_calibration_v5_ready"] is True
    assert artifact["repair_gate_state"] == "unblocked"
    assert artifact["exact_ground_truth_count"] == 4
    assert artifact["verifier_gain_delta"] == pytest.approx(0.5)
    assert artifact["verifier_gain_delta_with_certified_coherence"] == pytest.approx(0.5)
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["false_reject_rate"] == pytest.approx(0.0)
    assert artifact["calibration_error"] == pytest.approx(0.5)
    assert artifact["abstention_precision"] == pytest.approx(0.0)
    assert artifact["rejection_recall"] == pytest.approx(1.0)
    assert artifact["selected_headline_model_ids"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["live_llm_inference"] is False
    assert artifact["inference_substrate"]["cached_trace_source_executed_models"] is True
    assert artifact["exp3115_repair_gate_explanation"]["delta_sign"] == "positive"
    assert artifact["exp3115_repair_gate_explanation"]["downstream_action"] == "repair_gate_unblocked"
    assert artifact["tests_run"] == ["focused-unit"]
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_verify_3113_blocks_non_positive_delta_without_hiding_evidence(tmp_path: Path) -> None:
    """REQ-VERIFY-3113: zero certified lift blocks repair but keeps diagnostics ready."""

    _write_sources(tmp_path, logic_matches_exact=False, certified_matches_exact=False)

    artifact = mod.build_artifact(tmp_path, min_exact_count=4, tests_run=["zero-delta"])

    assert artifact["diagnostic_verifier_calibration_v5_ready"] is True
    assert artifact["repair_gate_state"] == "blocked_negative_delta"
    assert artifact["verifier_gain_delta"] == pytest.approx(0.0)
    assert artifact["verifier_gain_delta_with_certified_coherence"] == pytest.approx(0.0)
    assert artifact["exp3115_repair_gate_explanation"]["delta_sign"] == "zero"
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_verify_3113_fail_closed_gate_states_and_validation(tmp_path: Path) -> None:
    """REQ-VERIFY-3113: missing, tiny, and cache-blocked inputs produce explicit gates."""

    missing = mod.build_artifact(tmp_path, min_exact_count=4, tests_run=["missing"])
    assert missing["diagnostic_verifier_calibration_v5_ready"] is False
    assert missing["repair_gate_state"] == "blocked_missing_inputs"
    assert missing["honest_verdict"].startswith("blocked_missing_inputs")
    mod.validate_artifact(missing)

    _write_sources(tmp_path)
    tiny = mod.build_artifact(tmp_path, min_exact_count=5, tests_run=["tiny"])
    relative_output = mod.write_artifact(
        tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        min_exact_count=5,
        tests_run=["tiny-write"],
    )
    assert tiny["diagnostic_verifier_calibration_v5_ready"] is True
    assert tiny["repair_gate_state"] == "blocked_tiny_panel"
    assert tiny["honest_verdict"].startswith("complete_blocked_tiny_panel")
    assert relative_output == tmp_path / mod.OUTPUT_REL_PATH

    cache_root = tmp_path / "cache"
    _write_sources(cache_root, selected_models=False)
    cache_blocked = mod.build_artifact(cache_root, min_exact_count=4, tests_run=["cache"])
    assert cache_blocked["diagnostic_verifier_calibration_v5_ready"] is True
    assert cache_blocked["repair_gate_state"] == "blocked_model_cache"
    assert cache_blocked["honest_verdict"].startswith("complete_blocked_headline")

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="repair_gate_state"):
        mod.validate_artifact(cache_blocked | {"repair_gate_state": "maybe"})
    with pytest.raises(ValueError, match="finite metric"):
        mod.validate_artifact(cache_blocked | {"calibration_error": float("nan")})
    with pytest.raises(ValueError, match="selected model"):
        mod.validate_artifact(cache_blocked | {"repair_gate_state": "unblocked"})
    with pytest.raises(ValueError, match="success prefix"):
        mod.validate_artifact(cache_blocked | {"honest_verdict": "blocked_model_cache"})
    with pytest.raises(ValueError, match="blocked_missing_inputs"):
        mod.validate_artifact(missing | {"honest_verdict": "complete: bad"})


def test_req_verify_3113_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3113: parser, routing, and metric helpers stay deterministic."""

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_json) == {}
    malformed_jsonl = '\nnot-json\n{"fixture_id": "ok"}\n[]\n'
    assert mod.read_jsonl_rows_from_text(malformed_jsonl) == [{"fixture_id": "ok"}]
    assert mod.read_jsonl_rows(tmp_path / "missing.jsonl") == []
    assert mod.expected_action_from_answer("SAT") == "accept"
    assert mod.expected_action_from_answer("INVALID") == "reject"
    assert mod.expected_action_from_answer("REPAIRABLE") == "reject"
    assert mod.expected_action_from_answer("UNKNOWN") == "abstain"
    assert mod.rate(0, 0) == 0.0
    assert mod.relative_path(tmp_path, tmp_path / "nested" / "rows.jsonl") == "nested/rows.jsonl"
    assert mod.relative_path(tmp_path, Path("/outside/rows.jsonl")) == "/outside/rows.jsonl"
    assert mod.delta_sign(0.2) == "positive"
    assert mod.delta_sign(0.0) == "zero"
    assert mod.delta_sign(-0.1) == "negative"
    assert mod.normalized_route_confidence({"confidence": 0.7}, "accept") == pytest.approx(0.7)
    assert mod.normalized_route_confidence({"confidence": 2.0}, "accept") == pytest.approx(1.0)
    assert mod.normalized_route_confidence({"route_scores": {"accept": 1, "reject": 3}}, "reject") == pytest.approx(0.75)
    assert mod.normalized_route_confidence({}, "reject") == pytest.approx(0.0)

    rows = [
        {"expected_action": "reject", "decision": "abstain"},
        {"expected_action": "accept", "decision": "accept"},
    ]
    metrics = mod.decision_metrics(rows, "decision")
    assert metrics["abstention_precision"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(0.5)
