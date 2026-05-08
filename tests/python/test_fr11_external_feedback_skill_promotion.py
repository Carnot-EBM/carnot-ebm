"""Tests for Exp 1539 FR-11 external-feedback skill graph promotion.

Spec: REQ-LEARN-1539, SCENARIO-LEARN-1539, SCENARIO-LEARN-1540,
SCENARIO-LEARN-1541.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import fr11_external_feedback_skill_promotion as exp


def test_req_learn_1539_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1539-1/8: bootstrap artifact exposes every required field."""

    output = tmp_path / exp.OUTPUT_FILE
    skill_graph = tmp_path / exp.SKILL_GRAPH_FILE
    rollback_plan = tmp_path / exp.ROLLBACK_PLAN_FILE

    artifact = exp.write_in_progress_artifact(
        output,
        skill_graph_path=skill_graph,
        rollback_plan_path=rollback_plan,
        project_root=tmp_path,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "in_progress"
    assert artifact["milestone"] == ".118"
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["fr11_external_feedback_ready"] is False
    assert artifact["positive_utility_promotion_ready"] is False
    assert artifact["skill_graph_path"] == exp.SKILL_GRAPH_FILE
    assert artifact["rollback_plan_path"] == exp.ROLLBACK_PLAN_FILE
    assert artifact["no_model_weight_mutation"] is True
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    exp.validate_artifact(artifact)
    with pytest.raises(AssertionError, match="honest_verdict"):
        exp.validate_artifact(dict(artifact, honest_verdict="not-terminal"))
    with pytest.raises(AssertionError, match="skill graph artifact"):
        exp.validate_artifact(
            dict(
                artifact,
                fr11_external_feedback_ready=True,
                honest_verdict="complete: external feedback ready",
            ),
            skill_graph_path=tmp_path / "missing-graph.json",
        )


def test_scenario_learn_1539_replays_external_feedback_into_skill_graph() -> None:
    """SCENARIO-LEARN-1539: external verifier feedback promotes a graph node."""

    candidates = exp.extract_candidate_updates(
        promotion_rows=[
            _promotion_row("eligible", baseline_success=False, promoted_success=True),
            _promotion_row("self-only", baseline_success=False, promoted_success=True, external=False),
            _promotion_row(
                "unsafe",
                baseline_success=True,
                promoted_success=False,
                false_accept_delta=1,
                soundness_mistakes=1,
            ),
        ],
        rollback_rows=[
            _rollback_row("eligible"),
            _rollback_row("self-only"),
            _rollback_row("unsafe", false_accept_delta=1, soundness_mistakes=1),
        ],
        residual_drift_rows=[_residual_drift_row("eligible")],
    )
    graph = exp.build_skill_graph(
        candidates,
        skill_graph_path=Path("results/fr11_external_feedback_skill_graph_1539.json"),
    )

    promoted = graph["nodes"]
    candidate_by_id = {candidate["policy_update_id"]: candidate for candidate in candidates}

    assert [node["policy_update_id"] for node in promoted] == ["daily_eval:eligible"]
    assert candidate_by_id["daily_eval:eligible"]["explicit_inputs"]["contract_case_id"] == (
        "contract-eligible"
    )
    assert candidate_by_id["daily_eval:eligible"]["expected_outputs"] == {
        "final_deterministic_accept": False,
        "final_deterministic_decision": "reject",
    }
    assert candidate_by_id["daily_eval:eligible"]["model_outputs"]["promoted_sha256"] == (
        "sha-promoted-eligible"
    )
    assert candidate_by_id["daily_eval:eligible"]["verifier_reward"] == pytest.approx(1.0)
    assert candidate_by_id["daily_eval:self-only"]["external_deterministic_feedback"] is False
    assert candidate_by_id["daily_eval:unsafe"]["promotion_decision"] == "rollback_required"

    node = promoted[0]
    assert node["node_id"] == "skill:fr11_v13/eligible"
    assert node["lineage"]["parent_policy_update_id"] == "daily_eval:eligible"
    assert node["external_verifier_feedback"]["feedback_source"] == (
        "runtime_contract_deterministic_verifier"
    )
    assert node["external_verifier_feedback"]["self_feedback_only"] is False
    assert node["replay_evidence"]["rollback_decision"] == "keep"
    assert node["replay_evidence"]["residual_drift_cases_observed"] == 1
    assert node["promotion_decision"]["positive_utility"] is True


def test_scenario_learn_1540_zero_utility_blocks_headline_readiness(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1540: safe zero utility completes without headline success."""

    paths = _write_sources(
        tmp_path,
        [_promotion_row("zero", baseline_success=False, promoted_success=False)],
        [_rollback_row("zero")],
        [_residual_drift_row("zero")],
    )

    artifact = exp.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        output_path=paths["output"],
        skill_graph_path=paths["skill_graph"],
        rollback_plan_path=paths["rollback_plan"],
        live_policy_artifact_path=paths["policy_artifact"],
        live_policy_manifest_path=paths["policy_manifest"],
        rollback_manifest_path=paths["rollback_manifest"],
        residual_drift_artifact_path=paths["drift_artifact"],
        residual_drift_ledger_path=paths["drift_ledger"],
        focused_tests_passed=True,
    )

    graph = json.loads(paths["skill_graph"].read_text(encoding="utf-8"))
    rollback_plan = json.loads(paths["rollback_plan"].read_text(encoding="utf-8"))

    assert json.loads(paths["output"].read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["fr11_external_feedback_ready"] is True
    assert artifact["positive_utility_promotion_ready"] is False
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["baseline_task_success_rate"] == pytest.approx(0.0)
    assert artifact["promoted_task_success_rate"] == pytest.approx(0.0)
    assert artifact["utility_delta"] == pytest.approx(0.0)
    assert artifact["soundness_mistakes"] == 0
    assert artifact["promoted_updates"] == ["daily_eval:zero"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert graph["summary"]["promoted_node_count"] == 1
    assert rollback_plan["rollback_entries"][0]["policy_update_id"] == "daily_eval:zero"
    exp.validate_artifact(artifact, skill_graph_path=paths["skill_graph"])


def test_scenario_learn_1541_rollback_plan_rejects_self_feedback_and_unsafe_rows() -> None:
    """SCENARIO-LEARN-1541: self-only or unsafe replay cannot be promoted."""

    candidates = exp.extract_candidate_updates(
        promotion_rows=[
            _promotion_row("self-only", external=False),
            _promotion_row("unsafe", false_accept_delta=1, soundness_mistakes=1),
            _promotion_row("missing-rollback"),
            _promotion_row("legacy-model", model_hf_id="legacy/small-GGUF"),
            _promotion_row("rolled-back"),
            _promotion_row("unreachable"),
            _promotion_row("stale"),
            _promotion_row("no-validator"),
        ],
        rollback_rows=[
            _rollback_row("self-only"),
            _rollback_row("unsafe", false_accept_delta=1, soundness_mistakes=1),
            _rollback_row("legacy-model"),
            _rollback_row("rolled-back", decision="rollback"),
            _rollback_row("unreachable", reachable=False),
            _rollback_row("stale", stale=True),
            _rollback_row("no-validator", deterministic=False),
        ],
        residual_drift_rows=[],
    )
    graph = exp.build_skill_graph(candidates)
    plan = exp.build_rollback_plan(candidates, graph)

    assert graph["nodes"] == []
    reasons_by_update = {
        entry["policy_update_id"]: entry["rollback_triggers"] for entry in plan["rollback_entries"]
    }
    assert "missing_external_deterministic_verifier_feedback" in reasons_by_update[
        "daily_eval:self-only"
    ]
    assert "false_accept_delta_positive" in reasons_by_update["daily_eval:unsafe"]
    assert "soundness_mistake" in reasons_by_update["daily_eval:unsafe"]
    assert "missing_rollback_replay_evidence" in reasons_by_update["daily_eval:missing-rollback"]
    assert "missing_live_mandated_sota_evidence" in reasons_by_update[
        "daily_eval:legacy-model"
    ]
    assert "rollback_decision_not_keep" in reasons_by_update["daily_eval:rolled-back"]
    assert "source_evidence_unreachable" in reasons_by_update["daily_eval:unreachable"]
    assert "source_evidence_stale" in reasons_by_update["daily_eval:stale"]
    assert "missing_deterministic_validator_support" in reasons_by_update[
        "daily_eval:no-validator"
    ]
    assert all(entry["action"] == "do_not_promote_or_demote" for entry in plan["rollback_entries"])


def test_req_learn_1539_artifact_positive_and_blocked_verdicts(tmp_path: Path) -> None:
    """REQ-LEARN-1539-7/8: artifact readiness distinguishes positive and blocked."""

    graph_path = tmp_path / exp.SKILL_GRAPH_FILE
    graph_path.write_text("{}\n", encoding="utf-8")
    candidates = exp.extract_candidate_updates(
        promotion_rows=[_promotion_row("positive", baseline_success=False, promoted_success=True)],
        rollback_rows=[_rollback_row("positive")],
        residual_drift_rows=[_residual_drift_row("positive")],
    )
    graph = exp.build_skill_graph(candidates, skill_graph_path=graph_path, project_root=tmp_path)
    positive = exp.build_artifact(
        candidates=candidates,
        graph=graph,
        rollback_plan=exp.build_rollback_plan(candidates, graph),
        skill_graph_path=graph_path,
        rollback_plan_path=tmp_path / exp.ROLLBACK_PLAN_FILE,
        focused_tests_passed=True,
        project_root=tmp_path,
    )
    blocked = exp.build_artifact(
        candidates=[],
        graph={"nodes": [], "summary": {"no_model_weight_mutation": True}},
        rollback_plan={"rollback_entries": []},
        skill_graph_path=graph_path,
        rollback_plan_path=tmp_path / exp.ROLLBACK_PLAN_FILE,
        focused_tests_passed=True,
        project_root=tmp_path,
    )

    assert positive["positive_utility_promotion_ready"] is True
    assert positive["utility_delta"] == pytest.approx(1.0)
    assert positive["honest_verdict"].startswith("complete: fr11 external-feedback")
    assert blocked["status"] == "blocked"
    assert blocked["fr11_external_feedback_ready"] is False
    assert blocked["honest_verdict"] == "complete: fr11 external-feedback skill graph blocked"


def _promotion_row(
    case_id: str,
    *,
    baseline_success: bool = False,
    promoted_success: bool = True,
    false_accept_delta: int = 0,
    soundness_mistakes: int = 0,
    external: bool = True,
    model_hf_id: str | None = None,
) -> dict[str, Any]:
    model = model_hf_id or exp.MANDATED_MODEL_SPECS[0]
    return {
        "row_type": "policy_promotion_evaluation",
        "spec": ["REQ-LEARN-1524", "SCENARIO-LEARN-1524"],
        "model_hf_id": model,
        "model_name": "Qwen3.6-35B-A3B",
        "policy_update_id": f"daily_eval:{case_id}",
        "policy_action": "retrieval_boost",
        "skill_id": f"fr11_v10_trace2skill/{case_id}",
        "contract_case_id": f"contract-{case_id}",
        "prompt_or_case_id": f"prompt-{case_id}",
        "source_family": "runtime_contract",
        "baseline_task_success": baseline_success,
        "promoted_task_success": promoted_success,
        "utility_delta": int(promoted_success) - int(baseline_success),
        "baseline_false_accept": False,
        "promoted_false_accept": false_accept_delta > 0,
        "false_accept_delta": false_accept_delta,
        "soundness_mistakes": soundness_mistakes,
        "runtime_contract_validation": _runtime_validation(case_id, external=external),
    }


def _runtime_validation(case_id: str, *, external: bool) -> dict[str, Any]:
    promoted = {
        "mode": "promoted",
        "contract_case_id": f"contract-{case_id}",
        "model_hf_id": exp.MANDATED_MODEL_SPECS[0],
        "policy_update_id": f"daily_eval:{case_id}",
        "raw_output_sha256": f"sha-promoted-{case_id}",
        "raw_output_excerpt": f"promoted output for {case_id}",
        "parsed_contract_output": {
            "contract_case_id": f"contract-{case_id}",
            "final_deterministic_decision": "reject",
        },
        "parse_status": "ok",
        "case_id_matches": True,
        "expected_label": False,
        "proposed_final_deterministic_accept": False,
        "task_success": True,
        "false_accept": False,
    }
    if external:
        promoted["contract_validation_row"] = _contract_validation_row(case_id)
    return {
        "baseline": {
            "mode": "baseline",
            "contract_case_id": f"contract-{case_id}",
            "raw_output_sha256": f"sha-baseline-{case_id}",
            "raw_output_excerpt": f"baseline output for {case_id}",
            "expected_label": False,
            "proposed_final_deterministic_accept": True,
            "task_success": False,
            "false_accept": False,
            "contract_validation_row": _contract_validation_row(case_id),
        },
        "promoted": promoted,
    }


def _contract_validation_row(case_id: str) -> dict[str, Any]:
    return {
        "row_type": "contract_case",
        "contract_schema_version": "runtime-contract-e2e/v1",
        "contract_case_id": f"contract-{case_id}",
        "prompt_or_case_id": f"prompt-{case_id}",
        "expected_label": False,
        "final_deterministic_accept": False,
        "final_deterministic_decision": "reject",
        "source_family": "runtime_contract",
    }


def _rollback_row(
    case_id: str,
    *,
    decision: str = "keep",
    reachable: bool = True,
    stale: bool = False,
    deterministic: bool = True,
    false_accept_delta: int = 0,
    soundness_mistakes: int = 0,
) -> dict[str, Any]:
    return {
        "schema": "fr11_policy_rollback_replay_row_v1",
        "source_event_id": f"daily_eval:{case_id}",
        "source_case_id": case_id,
        "source_kind": "daily_eval",
        "skill_id": f"fr11_v10_trace2skill/{case_id}",
        "policy_action": "retrieval_boost",
        "decision": decision,
        "source_evidence_reachable": reachable,
        "source_evidence_stale": stale,
        "deterministic_validator_supported": deterministic,
        "soundness_mistakes": soundness_mistakes,
        "false_accept_delta": false_accept_delta,
        "utility_delta": 1,
        "rollback_reasons": [],
    }


def _residual_drift_row(case_id: str) -> dict[str, Any]:
    return {
        "row_type": "residual_drift_case",
        "source_case_id": case_id,
        "source_domain": "runtime_contract",
        "failure_classification": "satisfiable_drift",
        "repaired_drift": True,
        "false_accept": False,
    }


def _write_sources(
    tmp_path: Path,
    promotion_rows: list[dict[str, Any]],
    rollback_rows: list[dict[str, Any]],
    drift_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    paths = {
        "output": tmp_path / exp.OUTPUT_FILE,
        "skill_graph": tmp_path / exp.SKILL_GRAPH_FILE,
        "rollback_plan": tmp_path / exp.ROLLBACK_PLAN_FILE,
        "policy_artifact": tmp_path / "experiment_1524.json",
        "policy_manifest": tmp_path / "promotion.jsonl",
        "rollback_manifest": tmp_path / "rollback.jsonl",
        "drift_artifact": tmp_path / "experiment_1538.json",
        "drift_ledger": tmp_path / "drift.jsonl",
    }
    _write_json(
        paths["policy_artifact"],
        {
            "status": "complete",
            "continuous_self_learning_task": True,
            "live_sota_model_inference_used": True,
            "no_model_weight_mutation": True,
        },
    )
    _write_jsonl(paths["policy_manifest"], [*promotion_rows, {"row_type": "summary"}])
    _write_json(
        paths["drift_artifact"],
        {
            "status": "complete",
            "residual_drift_ledger_ready": True,
            "live_sota_model_inference_used": True,
        },
    )
    _write_jsonl(paths["rollback_manifest"], rollback_rows)
    _write_jsonl(paths["drift_ledger"], [*drift_rows, {"row_type": "residual_drift_summary"}])
    return paths


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
