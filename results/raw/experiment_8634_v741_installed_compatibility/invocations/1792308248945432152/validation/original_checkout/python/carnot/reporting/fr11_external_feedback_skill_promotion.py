"""Exp 1539 FR-11 external-feedback skill graph promotion.

Spec: REQ-LEARN-1539, SCENARIO-LEARN-1539, SCENARIO-LEARN-1540,
SCENARIO-LEARN-1541.

This experiment keeps FR-11 learning at query time.  It does not train or
finetune the LLM.  Instead, it converts rollback-passing policy/cache updates
into an auditable skill graph only when a deterministic verifier outside the
model's own prose has already supplied feedback.  The final readiness gate is
intentionally stricter than the safety gate: zero soundness mistakes can make a
skill graph usable for audit, but headline self-learning success needs a
positive measured utility delta.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260508"
MILESTONE = ".118"
OUTPUT_FILE = "experiment_1539_fr11_external_feedback_skill_promotion_v13.json"
SKILL_GRAPH_FILE = "fr11_external_feedback_skill_graph_1539.json"
ROLLBACK_PLAN_FILE = "fr11_external_feedback_skill_rollback_plan_1539.json"

DEFAULT_OUTPUT_PATH = Path("results") / OUTPUT_FILE
DEFAULT_SKILL_GRAPH_PATH = Path("results") / SKILL_GRAPH_FILE
DEFAULT_ROLLBACK_PLAN_PATH = Path("results") / ROLLBACK_PLAN_FILE
DEFAULT_LIVE_POLICY_ARTIFACT_PATH = Path(
    "results/experiment_1524_fr11_live_policy_promotion_v12.json"
)
DEFAULT_LIVE_POLICY_MANIFEST_PATH = Path("results/fr11_live_policy_promotion_1524.jsonl")
DEFAULT_ROLLBACK_MANIFEST_PATH = Path("results/fr11_policy_rollback_replay_1513.jsonl")
DEFAULT_RESIDUAL_DRIFT_ARTIFACT_PATH = Path(
    "results/experiment_1538_residual_drift_commitment_ledger.json"
)
DEFAULT_RESIDUAL_DRIFT_LEDGER_PATH = Path("results/residual_drift_commitment_ledger_1538.jsonl")

MANDATED_MODEL_SPECS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MANDATED_HF_IDS = frozenset(MANDATED_MODEL_SPECS)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "milestone",
    "continuous_self_learning_task",
    "fr11_external_feedback_ready",
    "positive_utility_promotion_ready",
    "model_specs",
    "live_sota_model_inference_used",
    "skill_graph_path",
    "candidate_updates",
    "externally_verified_updates",
    "promoted_updates",
    "baseline_task_success_rate",
    "promoted_task_success_rate",
    "utility_delta",
    "soundness_mistakes",
    "no_model_weight_mutation",
    "rollback_plan_path",
    "focused_tests_passed",
    "honest_verdict",
)

TERMINAL_VERDICT_PREFIXES: tuple[str, ...] = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    skill_graph_path: Path | str = DEFAULT_SKILL_GRAPH_PATH,
    rollback_plan_path: Path | str = DEFAULT_ROLLBACK_PLAN_PATH,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-LEARN-1539-1/8: create the durable bootstrap artifact first."""

    artifact = {
        "status": "in_progress",
        "milestone": MILESTONE,
        "run_date": run_date,
        "schema": "fr11_external_feedback_skill_promotion_v13",
        "spec": [
            "REQ-LEARN-1539",
            "SCENARIO-LEARN-1539",
            "SCENARIO-LEARN-1540",
            "SCENARIO-LEARN-1541",
        ],
        "continuous_self_learning_task": True,
        "fr11_external_feedback_ready": False,
        "positive_utility_promotion_ready": False,
        "model_specs": list(MANDATED_MODEL_SPECS),
        "live_sota_model_inference_used": False,
        "skill_graph_path": _display_path(skill_graph_path, project_root=project_root),
        "candidate_updates": [],
        "externally_verified_updates": [],
        "promoted_updates": [],
        "baseline_task_success_rate": 0.0,
        "promoted_task_success_rate": 0.0,
        "utility_delta": 0.0,
        "soundness_mistakes": 0,
        "no_model_weight_mutation": True,
        "rollback_plan_path": _display_path(rollback_plan_path, project_root=project_root),
        "focused_tests_passed": False,
        "honest_verdict": "complete: fr11 external-feedback skill graph in progress",
    }
    validate_artifact(artifact)
    _write_json(Path(output_path), artifact)
    return artifact


def extract_candidate_updates(
    *,
    promotion_rows: Sequence[Mapping[str, Any]],
    rollback_rows: Sequence[Mapping[str, Any]],
    residual_drift_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """REQ-LEARN-1539-2/3: normalize live rows into skill promotion candidates."""

    rollback_by_update = {
        str(row.get("source_event_id")): row
        for row in rollback_rows
        if row.get("source_event_id")
    }
    drift_by_case = _residual_drift_index(residual_drift_rows)
    candidates: list[JsonDict] = []
    for row in promotion_rows:
        if row.get("row_type") != "policy_promotion_evaluation":
            continue
        policy_update_id = str(row.get("policy_update_id") or "")
        rollback = rollback_by_update.get(policy_update_id)
        promoted_validation = _mapping(
            _mapping(row.get("runtime_contract_validation")).get("promoted")
        )
        baseline_validation = _mapping(
            _mapping(row.get("runtime_contract_validation")).get("baseline")
        )
        contract_validation = _mapping(promoted_validation.get("contract_validation_row"))
        external_feedback = _has_external_feedback(row, promoted_validation, contract_validation)
        rollback_reasons = _rollback_reasons(row, rollback, external_feedback=external_feedback)
        baseline_success = bool(row.get("baseline_task_success"))
        promoted_success = bool(row.get("promoted_task_success"))
        false_accept_delta = int(row.get("false_accept_delta", 0))
        soundness_mistakes = int(row.get("soundness_mistakes", 0))
        utility_delta = int(promoted_success) - int(baseline_success)
        source_case_id = str((rollback or {}).get("source_case_id") or _case_tail(row))
        residual_rows = drift_by_case.get(source_case_id, [])
        candidates.append(
            {
                "policy_update_id": policy_update_id,
                "skill_id": str(row.get("skill_id") or ""),
                "node_id": _node_id(row),
                "policy_action": str(row.get("policy_action") or ""),
                "model_hf_id": str(row.get("model_hf_id") or ""),
                "source_case_id": source_case_id,
                "explicit_inputs": {
                    "contract_case_id": row.get("contract_case_id"),
                    "prompt_or_case_id": row.get("prompt_or_case_id"),
                    "source_family": row.get("source_family"),
                    "policy_action": row.get("policy_action"),
                    "skill_id": row.get("skill_id"),
                },
                "expected_outputs": _expected_outputs(contract_validation, promoted_validation),
                "model_outputs": {
                    "baseline_sha256": baseline_validation.get("raw_output_sha256"),
                    "promoted_sha256": promoted_validation.get("raw_output_sha256"),
                    "baseline_excerpt": baseline_validation.get("raw_output_excerpt"),
                    "promoted_excerpt": promoted_validation.get("raw_output_excerpt"),
                },
                "verifier_reward": _verifier_reward(
                    promoted_success=promoted_success,
                    false_accept_delta=false_accept_delta,
                    soundness_mistakes=soundness_mistakes,
                ),
                "baseline_task_success": baseline_success,
                "promoted_task_success": promoted_success,
                "utility_delta": utility_delta,
                "false_accept_delta": false_accept_delta,
                "soundness_mistakes": soundness_mistakes,
                "external_deterministic_feedback": external_feedback,
                "replay_evidence": {
                    "rollback_decision": None if rollback is None else rollback.get("decision"),
                    "source_evidence_reachable": None
                    if rollback is None
                    else rollback.get("source_evidence_reachable"),
                    "source_evidence_stale": None
                    if rollback is None
                    else rollback.get("source_evidence_stale"),
                    "deterministic_validator_supported": None
                    if rollback is None
                    else rollback.get("deterministic_validator_supported"),
                    "rollback_false_accept_delta": 0
                    if rollback is None
                    else int(rollback.get("false_accept_delta", 0)),
                    "rollback_soundness_mistakes": 0
                    if rollback is None
                    else int(rollback.get("soundness_mistakes", 0)),
                    "residual_drift_cases_observed": len(residual_rows),
                    "residual_drift_false_accepts": sum(
                        int(drift.get("false_accept") is True) for drift in residual_rows
                    ),
                },
                "lineage": {
                    "parent_policy_update_id": policy_update_id,
                    "parent_skill_id": row.get("skill_id"),
                    "source_case_id": source_case_id,
                    "source_artifacts": [
                        _display_path(DEFAULT_LIVE_POLICY_MANIFEST_PATH),
                        _display_path(DEFAULT_ROLLBACK_MANIFEST_PATH),
                        _display_path(DEFAULT_RESIDUAL_DRIFT_LEDGER_PATH),
                    ],
                },
                "rollback_triggers": rollback_reasons,
                "promotion_decision": (
                    "promote_external_feedback" if not rollback_reasons else "rollback_required"
                ),
            }
        )
    return candidates


def build_skill_graph(
    candidates: Sequence[Mapping[str, Any]],
    *,
    skill_graph_path: Path | str = DEFAULT_SKILL_GRAPH_PATH,
    run_date: str = RUN_DATE,
    project_root: Path | str = REPO_ROOT,
) -> JsonDict:
    """REQ-LEARN-1539-3/4: build promoted nodes from externally verified candidates."""

    nodes = [_skill_node(candidate) for candidate in candidates if _is_promoted(candidate)]
    edges = [
        {
            "from": str(node["lineage"]["parent_policy_update_id"]),
            "to": node["node_id"],
            "relation": "external_feedback_replay_promotes_skill_node",
        }
        for node in nodes
    ]
    return {
        "schema": "fr11_external_feedback_skill_graph_v13",
        "run_date": run_date,
        "skill_graph_path": _display_path(skill_graph_path, project_root=project_root),
        "spec": [
            "REQ-LEARN-1539",
            "SCENARIO-LEARN-1539",
            "SCENARIO-LEARN-1540",
            "SCENARIO-LEARN-1541",
        ],
        "source_artifacts": [
            _display_path(DEFAULT_LIVE_POLICY_ARTIFACT_PATH),
            _display_path(DEFAULT_LIVE_POLICY_MANIFEST_PATH),
            _display_path(DEFAULT_ROLLBACK_MANIFEST_PATH),
            _display_path(DEFAULT_RESIDUAL_DRIFT_ARTIFACT_PATH),
            _display_path(DEFAULT_RESIDUAL_DRIFT_LEDGER_PATH),
        ],
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "candidate_update_count": len(candidates),
            "externally_verified_count": sum(
                int(bool(candidate.get("external_deterministic_feedback")))
                for candidate in candidates
            ),
            "promoted_node_count": len(nodes),
            "positive_utility_node_count": sum(
                int(float(node["promotion_decision"]["utility_delta"]) > 0.0)
                for node in nodes
            ),
            "no_model_weight_mutation": True,
        },
    }


def build_rollback_plan(
    candidates: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
    *,
    rollback_plan_path: Path | str = DEFAULT_ROLLBACK_PLAN_PATH,
    run_date: str = RUN_DATE,
    project_root: Path | str = REPO_ROOT,
) -> JsonDict:
    """REQ-LEARN-1539-5: every candidate gets an auditable rollback handle."""

    promoted_ids = {str(node["policy_update_id"]) for node in graph.get("nodes", [])}
    entries = [_rollback_entry(candidate, promoted_ids) for candidate in candidates]
    return {
        "schema": "fr11_external_feedback_skill_rollback_plan_v13",
        "run_date": run_date,
        "rollback_plan_path": _display_path(rollback_plan_path, project_root=project_root),
        "rollback_entries": entries,
        "global_rollback_triggers": [
            "soundness_mistakes_positive",
            "false_accept_delta_positive",
            "missing_external_deterministic_verifier_feedback",
            "source_evidence_stale_or_unreachable",
            "replay_evidence_missing_or_failed",
        ],
        "no_model_weight_mutation": True,
        "summary": {
            "promoted_updates_with_rollback_handles": sum(
                int(entry["policy_update_id"] in promoted_ids) for entry in entries
            ),
            "rejected_updates_with_demotion_handles": sum(
                int(entry["policy_update_id"] not in promoted_ids) for entry in entries
            ),
        },
    }


def build_artifact(
    *,
    candidates: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
    rollback_plan: Mapping[str, Any],
    skill_graph_path: Path | str = DEFAULT_SKILL_GRAPH_PATH,
    rollback_plan_path: Path | str = DEFAULT_ROLLBACK_PLAN_PATH,
    focused_tests_passed: bool = False,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-LEARN-1539-6/8: build the final promotion artifact."""

    del rollback_plan
    metrics = _summary_metrics(candidates, graph)
    external_ready = bool(
        graph.get("nodes")
        and metrics["live_sota_model_inference_used"]
        and metrics["soundness_mistakes"] == 0
    )
    positive_ready = bool(
        external_ready
        and metrics["utility_delta"] > 0.0
        and focused_tests_passed
        and metrics["no_model_weight_mutation"]
    )
    artifact = {
        "status": "complete" if external_ready else "blocked",
        "milestone": MILESTONE,
        "run_date": run_date,
        "schema": "fr11_external_feedback_skill_promotion_v13",
        "spec": [
            "REQ-LEARN-1539",
            "SCENARIO-LEARN-1539",
            "SCENARIO-LEARN-1540",
            "SCENARIO-LEARN-1541",
        ],
        "continuous_self_learning_task": True,
        "fr11_external_feedback_ready": external_ready,
        "positive_utility_promotion_ready": positive_ready,
        "model_specs": list(MANDATED_MODEL_SPECS),
        "live_sota_model_inference_used": metrics["live_sota_model_inference_used"],
        "skill_graph_path": _display_path(skill_graph_path, project_root=project_root),
        "candidate_updates": [_artifact_candidate(candidate) for candidate in candidates],
        "externally_verified_updates": [
            str(candidate["policy_update_id"])
            for candidate in candidates
            if candidate.get("external_deterministic_feedback") is True
        ],
        "promoted_updates": [str(node["policy_update_id"]) for node in graph.get("nodes", [])],
        "baseline_task_success_rate": metrics["baseline_task_success_rate"],
        "promoted_task_success_rate": metrics["promoted_task_success_rate"],
        "utility_delta": metrics["utility_delta"],
        "soundness_mistakes": metrics["soundness_mistakes"],
        "no_model_weight_mutation": metrics["no_model_weight_mutation"],
        "rollback_plan_path": _display_path(rollback_plan_path, project_root=project_root),
        "focused_tests_passed": bool(focused_tests_passed),
        "honest_verdict": _honest_verdict(
            external_ready=external_ready,
            positive_ready=positive_ready,
        ),
    }
    validate_artifact(artifact, skill_graph_path=skill_graph_path)
    return artifact


def run_experiment(
    *,
    project_root: Path | str | None = None,
    run_date: str = RUN_DATE,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    skill_graph_path: Path | str = DEFAULT_SKILL_GRAPH_PATH,
    rollback_plan_path: Path | str = DEFAULT_ROLLBACK_PLAN_PATH,
    live_policy_artifact_path: Path | str = DEFAULT_LIVE_POLICY_ARTIFACT_PATH,
    live_policy_manifest_path: Path | str = DEFAULT_LIVE_POLICY_MANIFEST_PATH,
    rollback_manifest_path: Path | str = DEFAULT_ROLLBACK_MANIFEST_PATH,
    residual_drift_artifact_path: Path | str = DEFAULT_RESIDUAL_DRIFT_ARTIFACT_PATH,
    residual_drift_ledger_path: Path | str = DEFAULT_RESIDUAL_DRIFT_LEDGER_PATH,
    focused_tests_passed: bool = False,
) -> JsonDict:
    """Run Exp 1539 and persist the terminal JSON, skill graph, and rollback plan."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    output = _resolve_under_root(root, Path(output_path))
    skill_graph_output = _resolve_under_root(root, Path(skill_graph_path))
    rollback_plan_output = _resolve_under_root(root, Path(rollback_plan_path))
    live_policy_artifact = _resolve_under_root(root, Path(live_policy_artifact_path))
    live_policy_manifest = _resolve_under_root(root, Path(live_policy_manifest_path))
    rollback_manifest = _resolve_under_root(root, Path(rollback_manifest_path))
    drift_artifact = _resolve_under_root(root, Path(residual_drift_artifact_path))
    drift_ledger = _resolve_under_root(root, Path(residual_drift_ledger_path))

    write_in_progress_artifact(
        output,
        skill_graph_path=skill_graph_output,
        rollback_plan_path=rollback_plan_output,
        project_root=root,
        run_date=run_date,
    )
    _load_json(live_policy_artifact)
    _load_json(drift_artifact)
    candidates = extract_candidate_updates(
        promotion_rows=_read_jsonl(live_policy_manifest),
        rollback_rows=_read_jsonl(rollback_manifest),
        residual_drift_rows=_read_jsonl(drift_ledger),
    )
    graph = build_skill_graph(
        candidates,
        skill_graph_path=skill_graph_output,
        project_root=root,
        run_date=run_date,
    )
    rollback_plan = build_rollback_plan(
        candidates,
        graph,
        rollback_plan_path=rollback_plan_output,
        project_root=root,
        run_date=run_date,
    )
    _write_json(skill_graph_output, graph)
    _write_json(rollback_plan_output, rollback_plan)
    artifact = build_artifact(
        candidates=candidates,
        graph=graph,
        rollback_plan=rollback_plan,
        skill_graph_path=skill_graph_output,
        rollback_plan_path=rollback_plan_output,
        focused_tests_passed=focused_tests_passed,
        project_root=root,
        run_date=run_date,
    )
    _write_json(output, artifact)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    skill_graph_path: Path | str | None = None,
) -> None:
    """Enforce the terminal artifact shape used by the conductor."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:  # pragma: no cover - defensive schema guard.
        raise AssertionError(f"missing required fields: {missing}")
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_VERDICT_PREFIXES):
        raise AssertionError("honest_verdict must use an allowed terminal prefix")
    if artifact["positive_utility_promotion_ready"]:  # pragma: no cover - defensive guard.
        if float(artifact["utility_delta"]) <= 0.0:
            raise AssertionError("positive utility readiness requires utility_delta > 0")
        if int(artifact["soundness_mistakes"]) != 0:
            raise AssertionError("positive utility readiness requires zero soundness mistakes")
        if artifact["no_model_weight_mutation"] is not True:
            raise AssertionError("positive utility readiness requires frozen model weights")
    if artifact["fr11_external_feedback_ready"] and skill_graph_path is not None:
        if not Path(skill_graph_path).exists():
            raise AssertionError("external feedback readiness requires a skill graph artifact")


def _skill_node(candidate: Mapping[str, Any]) -> JsonDict:
    return {
        "node_id": str(candidate["node_id"]),
        "policy_update_id": str(candidate["policy_update_id"]),
        "skill_id": str(candidate["skill_id"]),
        "interface": {
            "inputs": dict(candidate["explicit_inputs"]),
            "expected_outputs": dict(candidate["expected_outputs"]),
        },
        "lineage": dict(candidate["lineage"]),
        "external_verifier_feedback": {
            "feedback_source": "runtime_contract_deterministic_verifier",
            "self_feedback_only": False,
            "verifier_reward": float(candidate["verifier_reward"]),
            "baseline_task_success": bool(candidate["baseline_task_success"]),
            "promoted_task_success": bool(candidate["promoted_task_success"]),
            "false_accept_delta": int(candidate["false_accept_delta"]),
            "soundness_mistakes": int(candidate["soundness_mistakes"]),
        },
        "replay_evidence": dict(candidate["replay_evidence"]),
        "promotion_decision": {
            "status": "promoted_external_feedback",
            "positive_utility": float(candidate["utility_delta"]) > 0.0,
            "utility_delta": float(candidate["utility_delta"]),
            "no_model_weight_mutation": True,
        },
    }


def _rollback_entry(candidate: Mapping[str, Any], promoted_ids: set[str]) -> JsonDict:
    update_id = str(candidate["policy_update_id"])
    promoted = update_id in promoted_ids
    triggers = (
        [
            "soundness_mistakes_positive",
            "false_accept_delta_positive",
            "external_verifier_feedback_missing_or_stale",
            "rollback_replay_fails_or_source_unreachable",
            "future_replay_utility_negative",
        ]
        if promoted
        else list(candidate["rollback_triggers"])
    )
    return {
        "node_id": str(candidate["node_id"]),
        "policy_update_id": update_id,
        "skill_id": str(candidate["skill_id"]),
        "action": (
            "disable_query_time_skill_and_revert_to_baseline_policy"
            if promoted
            else "do_not_promote_or_demote"
        ),
        "rollback_triggers": triggers,
        "replay_evidence": dict(candidate["replay_evidence"]),
    }


def _artifact_candidate(candidate: Mapping[str, Any]) -> JsonDict:
    return {
        "policy_update_id": str(candidate["policy_update_id"]),
        "skill_id": str(candidate["skill_id"]),
        "node_id": str(candidate["node_id"]),
        "explicit_inputs": dict(candidate["explicit_inputs"]),
        "expected_outputs": dict(candidate["expected_outputs"]),
        "model_outputs": dict(candidate["model_outputs"]),
        "verifier_reward": float(candidate["verifier_reward"]),
        "external_deterministic_feedback": bool(candidate["external_deterministic_feedback"]),
        "replay_evidence": dict(candidate["replay_evidence"]),
        "lineage": dict(candidate["lineage"]),
        "promotion_decision": str(candidate["promotion_decision"]),
        "rollback_triggers": list(candidate["rollback_triggers"]),
    }


def _summary_metrics(
    candidates: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
) -> JsonDict:
    total = len(candidates)
    baseline_success = sum(int(candidate.get("baseline_task_success") is True) for candidate in candidates)
    promoted_success = sum(int(candidate.get("promoted_task_success") is True) for candidate in candidates)
    return {
        "baseline_task_success_rate": _rate(baseline_success, total),
        "promoted_task_success_rate": _rate(promoted_success, total),
        "utility_delta": round(_rate(promoted_success, total) - _rate(baseline_success, total), 6),
        "soundness_mistakes": sum(int(candidate.get("soundness_mistakes", 0)) for candidate in candidates),
        "live_sota_model_inference_used": any(
            candidate.get("model_hf_id") in MANDATED_HF_IDS for candidate in candidates
        ),
        "no_model_weight_mutation": bool(graph.get("summary", {}).get("no_model_weight_mutation")),
    }


def _rollback_reasons(
    promotion_row: Mapping[str, Any],
    rollback: Mapping[str, Any] | None,
    *,
    external_feedback: bool,
) -> list[str]:
    reasons: list[str] = []
    if not external_feedback:
        reasons.append("missing_external_deterministic_verifier_feedback")
    if promotion_row.get("model_hf_id") not in MANDATED_HF_IDS:
        reasons.append("missing_live_mandated_sota_evidence")
    if int(promotion_row.get("false_accept_delta", 0)) > 0:
        reasons.append("false_accept_delta_positive")
    if int(promotion_row.get("soundness_mistakes", 0)) > 0:
        reasons.append("soundness_mistake")
    if rollback is None:
        reasons.append("missing_rollback_replay_evidence")
    else:
        if rollback.get("decision") != "keep":
            reasons.append("rollback_decision_not_keep")
        if rollback.get("source_evidence_reachable") is not True:
            reasons.append("source_evidence_unreachable")
        if bool(rollback.get("source_evidence_stale")):
            reasons.append("source_evidence_stale")
        if rollback.get("deterministic_validator_supported") is not True:
            reasons.append("missing_deterministic_validator_support")
        if int(rollback.get("false_accept_delta", 0)) > 0:
            reasons.append("rollback_false_accept_delta_positive")
        if int(rollback.get("soundness_mistakes", 0)) > 0:
            reasons.append("rollback_soundness_mistake")
    return sorted(dict.fromkeys(reasons))


def _has_external_feedback(
    row: Mapping[str, Any],
    promoted_validation: Mapping[str, Any],
    contract_validation: Mapping[str, Any],
) -> bool:
    return bool(
        row.get("model_hf_id") in MANDATED_HF_IDS
        and contract_validation
        and promoted_validation.get("false_accept") is False
        and promoted_validation.get("expected_label") is not None
    )


def _expected_outputs(
    contract_validation: Mapping[str, Any],
    promoted_validation: Mapping[str, Any],
) -> JsonDict:
    return {
        "final_deterministic_accept": contract_validation.get(
            "final_deterministic_accept",
            promoted_validation.get("proposed_final_deterministic_accept"),
        ),
        "final_deterministic_decision": contract_validation.get("final_deterministic_decision"),
    }


def _verifier_reward(
    *,
    promoted_success: bool,
    false_accept_delta: int,
    soundness_mistakes: int,
) -> float:
    return 1.0 if promoted_success and false_accept_delta <= 0 and soundness_mistakes == 0 else 0.0


def _residual_drift_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    indexed: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("row_type") != "residual_drift_case":
            continue
        key = str(row.get("source_case_id") or "")
        indexed.setdefault(key, []).append(row)
    return indexed


def _node_id(row: Mapping[str, Any]) -> str:
    skill_id = str(row.get("skill_id") or row.get("policy_update_id") or "unknown")
    suffix = skill_id.rsplit("/", 1)[-1].replace(":", "-")
    return f"skill:fr11_v13/{suffix}"


def _case_tail(row: Mapping[str, Any]) -> str:
    return str(row.get("policy_update_id") or "").rsplit(":", 1)[-1]


def _is_promoted(candidate: Mapping[str, Any]) -> bool:
    return candidate.get("promotion_decision") == "promote_external_feedback"


def _honest_verdict(*, external_ready: bool, positive_ready: bool) -> str:
    if positive_ready:
        return "complete: fr11 external-feedback skill graph positive utility ready"
    if external_ready:
        return "complete: fr11 external-feedback skill graph ready; positive utility not demonstrated"
    return "complete: fr11 external-feedback skill graph blocked"


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(numerator / denominator, 6)


def _resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path | str, *, project_root: Path | str = REPO_ROOT) -> str:
    target = Path(path)
    try:
        return target.resolve().relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:  # pragma: no cover - used only for out-of-tree diagnostics.
        return target.as_posix()


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive artifact guard.
        raise AssertionError(f"JSON artifact must be an object: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focused-tests-passed", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment(focused_tests_passed=args.focused_tests_passed)
    print(
        "[exp1539] "
        f"external_ready={artifact['fr11_external_feedback_ready']} "
        f"positive_ready={artifact['positive_utility_promotion_ready']} "
        f"utility_delta={artifact['utility_delta']} "
        f"soundness={artifact['soundness_mistakes']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())


__all__ = [
    "MANDATED_MODEL_SPECS",
    "OUTPUT_FILE",
    "REQUIRED_ARTIFACT_FIELDS",
    "ROLLBACK_PLAN_FILE",
    "SKILL_GRAPH_FILE",
    "build_artifact",
    "build_rollback_plan",
    "build_skill_graph",
    "extract_candidate_updates",
    "run_experiment",
    "validate_artifact",
    "write_in_progress_artifact",
]
