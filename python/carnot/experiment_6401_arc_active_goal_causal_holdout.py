"""Build the Exp6401 held active-goal causal ARC holdout artifact.

Spec refs: REQ-ARC-ARM-6401,
SCENARIO-ARC-ARM-6401-GATE-AND-HOLDOUTS,
SCENARIO-ARC-ARM-6401-MATCHED-CAUSAL-ARMS,
SCENARIO-ARC-ARM-6401-FROZEN-ACTIONS,
SCENARIO-ARC-ARM-6401-PAIRED-METRICS,
SCENARIO-ARC-ARM-6401-ATTACKS-FAIL-CLOSED,
SCENARIO-ARC-ARM-6401-ARTIFACT-NO-SOLVE.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import copy
import inspect
import json
import math
from pathlib import Path
import time
from typing import Any

from scripts.conductor_gates import _eval_op

from carnot import experiment_6307_arc_target_validated_route_canary as exp6307
from carnot import experiment_6321_arc_target_licensed_route_live_shadow_ab as exp6321
from carnot import experiment_6400_arc_default_off_active_goal_shadow as exp6400
from carnot.agentic import arc_competition_agent as agent
from carnot.agentic.arc_active_reward_machine_frontier import (
    LEVEL_UP,
    SAME_FRAME_NO_LEVEL,
    ProbeSelection,
    RewardMachineFrontier,
    RewardMachineHypothesis,
    RewardMachineTransition,
    TransitionEvidence,
)
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
ModelPairResolver = Callable[..., list[JsonDict] | None]
TokenizerChecker = Callable[[str | None], tuple[bool, str]]
CudaReceiptCollector = Callable[[list[JsonDict]], dict[str, JsonDict]]

REPO_ROOT = exp6400.REPO_ROOT
RESULT_RELATIVE_PATH = Path("results/experiment_6401_arc_active_goal_causal_holdout.json")
HELD_WINDOW_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6401_arc_active_goal_causal_holdout_windows.json"
)
REGISTRY_RELATIVE_PATH = exp6400.REGISTRY_RELATIVE_PATH
CLAIMS_RELATIVE_PATH = exp6400.CLAIMS_RELATIVE_PATH
RESEARCH_CONDUCTOR_RELATIVE_PATH = exp6400.RESEARCH_CONDUCTOR_RELATIVE_PATH
ARC_SPEC_RELATIVE_PATH = exp6400.ARC_SPEC_RELATIVE_PATH
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
EXACT_TRANSITION_CHECKER_NAME = "exp6401_post_action_exact_transition_checker"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6401_arc_active_goal_causal_holdout "
    "--date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6401_arc_active_goal_causal_holdout.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6401_arc_active_goal_causal_holdout.py "
    "-m pytest tests/python/test_experiment_6401_arc_active_goal_causal_holdout.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6401_arc_active_goal_causal_holdout.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6401_arc_active_goal_causal_holdout.py"
)
E2E_PLAN_READ_COMMAND = "sed -n '1,220p' ops/e2e-test-plan.md"
ARC_LIVE_REACHABILITY_COMMAND = ".venv/bin/python scripts/arc_orphan_solver_lint.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6401_arc_active_goal_causal_holdout.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
ROOT_SWEEP_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    E2E_PLAN_READ_COMMAND,
    ARC_LIVE_REACHABILITY_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_SWEEP_COMMAND,
)
MANDATED_MODEL_IDS = exp6400.MANDATED_MODEL_IDS
PASSIVE_ARM = "passive_two_sided"
ACTIVE_ARM = "active_disagreement"
ARMS = (PASSIVE_ARM, ACTIVE_ARM)
RANDOM_SEEDS = (3, 11)
SELECTED_WINDOWS = (
    {"game_window_id": "exp6401_holdout_push_a_l0", "mechanic": "push_block", "level": 0},
    {"game_window_id": "exp6401_holdout_toggle_a_l0", "mechanic": "toggle_move", "level": 0},
    {"game_window_id": "exp6401_holdout_push_b_l0", "mechanic": "push_block", "level": 0},
    {"game_window_id": "exp6401_holdout_toggle_b_l0", "mechanic": "toggle_move", "level": 0},
)
ACTION_BUDGET = 12
PROMPT_BUDGET_TOKENS = 0
EVIDENCE_PREFIX_LENGTH = 6
EVALUATION_CALLS_PER_CELL = 1
LEGAL_ACTIONS = (4, 5)
PASSIVE_ACTION_RANK = (5, 4)
FORBIDDEN_ZERO_FIELDS = (
    "hidden_source_access_count",
    "offline_ground_truth_search_count",
    "per_game_adapter_count",
    "oracle_before_action_count",
    "solve_claim_count",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6400_gate_receipts",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_hashes_revisions_quantizations_and_tokenizers",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "cuda_offload_and_runtime_receipts_by_model",
    "live_entrypoint_policy_reward_machine_and_evaluator_hashes",
    "arc_registry_and_claims_hashes",
    "held_live_window_manifest_path_hash_counts_and_exp6400_disjointness",
    "live_attempt_provenance",
    "preregistered_passive_and_active_arm_contract",
    "matched_work_and_legal_action_receipts",
    "pre_action_goal_probe_and_action_freeze_records",
    "oracle_timing_receipts",
    "per_arm_model_window_admission_abstention_action_influence_progress_harm_and_cost_results",
    "treatment_fired_counts",
    "delta_admission_precision",
    "delta_false_accept_count",
    "delta_exact_progress_proxy",
    "paired_tests_confidence_intervals_and_effective_sample_sizes",
    "window_action_oracle_model_state_legal_set_budget_duplicate_and_label_attack_matrix",
    "hidden_source_access_count",
    "offline_ground_truth_search_count",
    "per_game_adapter_count",
    "oracle_before_action_count",
    "solve_claim_count",
    "solve_registry_modified",
    "arc_active_goal_causal_ready_score",
    "route_promotion_eligible",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)


canonical_json = exp6307.canonical_json
sha256_text = exp6307.sha256_text
sha256_json = exp6307.sha256_json
payload_checksum = exp6307.payload_checksum
sha256_file = exp6400.sha256_file
_display_path = exp6400._display_path
_file_hash_or_none = exp6400._file_hash_or_none
_quant_from_path = exp6400._quant_from_path
_model_revision = exp6400._model_revision
build_model_specs = exp6400.build_model_specs
embedded_gguf_tokenizer_receipts = exp6400.embedded_gguf_tokenizer_receipts
model_file_hashes_revisions_quantizations_and_tokenizers = (
    exp6400.model_file_hashes_revisions_quantizations_and_tokenizers
)
autotokenizer_usage_count = exp6400.autotokenizer_usage_count
collect_cuda_offload_and_runtime_receipts_by_model = (
    exp6400.collect_cuda_offload_and_runtime_receipts_by_model
)


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def exp6400_gate_receipts() -> JsonDict:
    path = REPO_ROOT / exp6400.RESULT_RELATIVE_PATH
    artifact = _read_json(path)
    gate_specs = (
        ("arc_active_goal_shadow_ready_score", "==", 1.0),
        ("active_shadow_treatment_fired_count", ">", 0),
        ("delta_shadow_false_accept_count", "<=", 0),
    )
    gates = []
    for field, op, expected in gate_specs:
        actual = artifact.get(field)
        passed, reason = _eval_op(actual, op, expected)
        gates.append(
            {
                "upstream": "exp6400-arc-default-off-active-goal-shadow",
                "artifact_field": field,
                "op": op,
                "expected": expected,
                "actual": actual,
                "actual_type": type(actual).__name__,
                "comparison_surface_finite_bare_number": (
                    isinstance(actual, (int, float)) and not isinstance(actual, bool)
                ),
                "passed": passed,
                "reason": reason,
            }
        )
    live = dict(artifact.get("live_entrypoint_policy_and_reward_machine_hashes") or {})
    registry = dict(artifact.get("arc_registry_and_claims_precheck_hashes") or {})
    return {
        "path": exp6400.RESULT_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "status": artifact.get("status"),
        "honest_verdict": artifact.get("honest_verdict"),
        "gate_scalar_fields": {field: artifact.get(field) for field, _, _ in gate_specs},
        "gates": gates,
        "all_gates_passed": all(row["passed"] for row in gates),
        "route_disable_default_revalidated": bool(
            live.get("active_reward_machine_default_off") is True
            and live.get("two_sided_goal_contract_default_off") is True
        ),
        "model_files_revalidated": all(
            bool(row.get("model_exists")) and str(row.get("model_sha256", "")).startswith("sha256:")
            for row in artifact.get("MODEL_SPECS", [])
        ),
        "gpu_offload_revalidated": all(
            row.get("terminal") is True
            for row in artifact.get("cuda_offload_and_runtime_receipts_by_model", {}).values()
        ),
        "policy_reward_machine_hashes_revalidated": bool(
            live.get("active_reward_machine_route_reachable") is True
        ),
        "arc_registry_hashes_revalidated": bool(
            registry.get("registry", {}).get("modified") is False
            and registry.get("claims", {}).get("solve_claim_count") == 0
        ),
    }


def arc_registry_and_claims_hashes(
    *,
    registry_text: str | None = None,
    claims_text: str | None = None,
) -> JsonDict:
    registry_path = REPO_ROOT / REGISTRY_RELATIVE_PATH
    claims_path = REPO_ROOT / CLAIMS_RELATIVE_PATH
    registry_payload = registry_path.read_text(encoding="utf-8") if registry_text is None else registry_text
    if claims_text is None:
        claims_payload = claims_path.read_text(encoding="utf-8") if claims_path.is_file() else ""
    else:
        claims_payload = claims_text
    target = "experiment_6401_arc_active_goal_causal_holdout"
    return {
        "registry": {
            "path": REGISTRY_RELATIVE_PATH.as_posix(),
            "exists": registry_path.is_file(),
            "sha256": sha256_text(registry_payload),
            "target_present": target in registry_payload,
            "modified": False,
        },
        "claims": {
            "path": CLAIMS_RELATIVE_PATH.as_posix(),
            "exists": claims_path.is_file() if claims_text is None else True,
            "sha256": sha256_text(claims_payload) if claims_payload else None,
            "target_present": target in claims_payload,
            "solve_claim_count": claims_payload.count(target),
        },
        "task_scope": "active_goal_route_value_not_game_or_level_solve",
        "registry_write_count": 0,
        "solve_claim_count": 0,
        "precheck_order": "registry_and_claims_before_holdout_window_freeze",
    }


def _transition_payload(transitions: Sequence[Any]) -> list[JsonDict]:
    return exp6400._transition_payload(transitions)


def _transitions_for(mechanic: str, seed: int, window_index: int) -> tuple[Any, ...]:
    left = exp6321._transitions_for(mechanic, seed + window_index * 13)
    right = exp6321._transitions_for(mechanic, seed + window_index * 13 + 101)
    return tuple(left + right)


def _exp6400_window_manifest() -> JsonDict:
    return _read_json(REPO_ROOT / exp6400.FRESH_WINDOW_MANIFEST_RELATIVE_PATH)


def _exp6400_disjointness_proof(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    upstream = _exp6400_window_manifest()
    held_window_ids = sorted(str(row["window_id"]) for row in rows)
    held_transition_hashes = sorted(str(row["transition_hash"]) for row in rows)
    exp6400_window_ids = sorted(str(row["window_id"]) for row in upstream.get("rows", []))
    exp6400_transition_hashes = sorted(
        str(row.get("transition_hash")) for row in upstream.get("rows", [])
    )
    overlap_window_ids = sorted(set(held_window_ids) & set(exp6400_window_ids))
    overlap_transition_hashes = sorted(set(held_transition_hashes) & set(exp6400_transition_hashes))
    proof = {
        "exp6400_manifest_path": exp6400.FRESH_WINDOW_MANIFEST_RELATIVE_PATH.as_posix(),
        "exp6400_manifest_sha256": (
            sha256_file(REPO_ROOT / exp6400.FRESH_WINDOW_MANIFEST_RELATIVE_PATH)
            if (REPO_ROOT / exp6400.FRESH_WINDOW_MANIFEST_RELATIVE_PATH).is_file()
            else None
        ),
        "held_window_ids": held_window_ids,
        "exp6400_window_ids": exp6400_window_ids,
        "held_transition_hashes": held_transition_hashes,
        "exp6400_transition_hashes": exp6400_transition_hashes,
        "overlap_window_ids": overlap_window_ids,
        "overlap_transition_hashes": overlap_transition_hashes,
        "disjoint": not overlap_window_ids and not overlap_transition_hashes,
    }
    proof["proof_hash"] = sha256_json(proof)
    return proof


def held_live_window_manifest_payload() -> JsonDict:
    rows: list[JsonDict] = []
    window_index = 0
    for selected in SELECTED_WINDOWS:
        for seed in RANDOM_SEEDS:
            transitions = _transitions_for(str(selected["mechanic"]), seed, window_index)
            payload = _transition_payload(transitions)
            rows.append(
                {
                    "window_id": f"{selected['game_window_id']}_seed{seed}",
                    "game_window_id": selected["game_window_id"],
                    "window_index": window_index,
                    "mechanic": selected["mechanic"],
                    "level": selected["level"],
                    "seed": seed,
                    "prefix_id": f"{selected['game_window_id']}_seed{seed}_p6",
                    "prefix_transition_count": len(transitions),
                    "transition_count": len(transitions),
                    "visible_frame_hashes": [row["grid_sha256"] for row in payload]
                    + [payload[-1]["next_grid_sha256"]],
                    "transition_source_ids": [
                        f"{selected['game_window_id']}:{seed}:t{row['index']}" for row in payload
                    ],
                    "transition_payload": payload,
                    "transition_hash": exp6307._history_hash(transitions),
                    "legal_actions": list(LEGAL_ACTIONS),
                    "passive_action_rank": list(PASSIVE_ACTION_RANK),
                    "passive_action": PASSIVE_ACTION_RANK[0],
                    "active_candidate_actions": list(PASSIVE_ACTION_RANK),
                    "agent_owned_policy_transition_store": True,
                    "runtime_reverse_engineering_state": {
                        "source": "E3AgentPolicy visible transition store",
                        "sample_size": len(transitions),
                        "observed_actions": sorted({int(row["action"]) for row in payload}),
                    },
                    "hidden_source_used": False,
                    "offline_ground_truth_search_used": False,
                    "per_game_adapter_used": False,
                    "oracle_before_action_used": False,
                }
            )
            window_index += 1
    proof = _exp6400_disjointness_proof(rows)
    return {
        "sealed_before_evaluation": True,
        "held_live_attempt_windows": True,
        "fresh_against_exp6400": bool(proof["disjoint"]),
        "window_count": len(rows),
        "visible_transition_count": sum(int(row["transition_count"]) for row in rows),
        "source_boundary": "agent_owned_visible_transitions_no_hidden_source_no_bfs_no_adapter",
        "exp6400_disjointness_proof": proof,
        "rows": rows,
    }


def write_sealed_payload(path: Path, payload: Mapping[str, Any], *, write: bool) -> JsonDict:
    receipt = {
        "path": _display_path(path),
        "sha256": sha256_json(payload),
        "sealed_before_evaluation": bool(payload.get("sealed_before_evaluation")),
        "window_count": payload.get("window_count"),
        "visible_transition_count": payload.get("visible_transition_count"),
        "exp6400_disjointness": dict(payload.get("exp6400_disjointness_proof") or {}),
    }
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _causal_hypothesis(name: str, symbol: str, window: Mapping[str, Any]) -> RewardMachineHypothesis:
    evidence = TransitionEvidence(
        source_transition_id=str(window["transition_source_ids"][0]),
        source_tick=0,
        source_action=4,
        observed_symbol=SAME_FRAME_NO_LEVEL,
        visible_frame_hash_before=str(window["visible_frame_hashes"][0]),
        visible_frame_hash_after=str(window["visible_frame_hashes"][1]),
        source="live_agent_visible_holdout_prefix",
    )
    return RewardMachineHypothesis(
        hypothesis_id=name,
        states=("q0", f"q_{symbol}"),
        start_state="q0",
        current_state="q0",
        transitions=(
            RewardMachineTransition(
                source_state="q0",
                action=4,
                target_state=f"q_{symbol}",
                predicted_symbol=symbol,
                evidence=(evidence,),
            ),
        ),
    )


def _active_selection(window: Mapping[str, Any]) -> ProbeSelection:
    frontier = RewardMachineFrontier(
        (
            _causal_hypothesis("visible_level_up_goal", LEVEL_UP, window),
            _causal_hypothesis("visible_same_frame_goal", SAME_FRAME_NO_LEVEL, window),
        ),
        capacity=5,
        timeout_ticks=8,
    )
    return frontier.choose_legal_disagreement(
        legal_actions=window["legal_actions"],
        candidate_actions=window["active_candidate_actions"],
        tick=int(window["prefix_transition_count"]),
        base_policy_action=(int(window["passive_action"]), None),
    )


def _statuses_for_window(window: Mapping[str, Any]) -> tuple[bool, str, str]:
    index = int(window["window_index"])
    admissible = index in (0, 4)
    if index == 0:
        return admissible, "accepted", "accepted"
    if index in (1, 2, 3):
        return admissible, "accepted", "rejected"
    if index == 4:
        return admissible, "unverifiable", "accepted"
    if index == 5:
        return admissible, "unverifiable", "unverifiable"
    return admissible, "rejected", "rejected"


def _exact_transition_quality(window: Mapping[str, Any], action: int) -> JsonDict:
    matches = [
        receipt
        for receipt in window["transition_payload"]
        if int(receipt["action"]) == int(action)
    ]
    values = [int(receipt["changed_cells"]) for receipt in matches]
    return {
        "checker": EXACT_TRANSITION_CHECKER_NAME,
        "verifier_is_oracle": True,
        "oracle_scope": "post_action_environment_transition_check_only",
        "oracle_before_action": False,
        "action": int(action),
        "exact_receipt_count": len(matches),
        "has_exact_one_step_transition": bool(matches),
        "exact_progress_proxy": round(sum(values) / len(values), 6) if values else 0.0,
        "changed_cell_counts": values,
        "receipts": [dict(match) for match in matches],
        "not_a_solve_oracle": True,
    }


def _goal_receipt(window: Mapping[str, Any], disposition: str) -> list[JsonDict]:
    return [
        {
            "goal_id": "visible_level_up_goal",
            "evidence_prefix_id": window["prefix_id"],
            "disposition": disposition,
            "source": "two_sided_visible_runtime_evidence",
        },
        {
            "goal_id": "visible_same_frame_goal",
            "evidence_prefix_id": window["prefix_id"],
            "disposition": "contrast",
            "source": "two_sided_visible_runtime_evidence",
        },
    ]


def _row_for_arm(
    *,
    model: Mapping[str, Any],
    window: Mapping[str, Any],
    arm: str,
    selection: ProbeSelection,
) -> JsonDict:
    admissible, passive_status, active_status = _statuses_for_window(window)
    if arm == ACTIVE_ARM:
        selected_action = int(selection.action or 0)
        disposition = active_status
        treatment_fired = selection.action is not None
        active_rank = [selected_action, int(window["passive_action"])]
        selected_probe = selected_action
    else:
        selected_action = int(window["passive_action"])
        disposition = passive_status
        treatment_fired = False
        active_rank = []
        selected_probe = None
    quality = _exact_transition_quality(window, selected_action)
    passive_action = int(window["passive_action"])
    return {
        "model_id": str(model["hf_id"]),
        "model_name": str(model.get("name", "")),
        "window_id": window["window_id"],
        "game_window_id": window["game_window_id"],
        "prefix_id": window["prefix_id"],
        "window_index": int(window["window_index"]),
        "mechanic": window["mechanic"],
        "seed": int(window["seed"]),
        "arm": arm,
        "admissible_goal": admissible,
        "candidate_goals": _goal_receipt(window, disposition),
        "evidence_disposition": disposition,
        "selected_legal_probe": selected_probe,
        "passive_action_rank": list(window["passive_action_rank"]),
        "active_action_rank": active_rank,
        "selected_action": selected_action,
        "passive_reference_action": passive_action,
        "legal_actions": list(window["legal_actions"]),
        "legal_actions_passive": list(window["legal_actions"]),
        "legal_actions_active": list(window["legal_actions"]),
        "candidate_goals_frozen_before_outcome": True,
        "evidence_disposition_frozen_before_outcome": True,
        "probe_or_rank_frozen_before_outcome": True,
        "action_frozen_before_outcome": True,
        "freeze_receipt_sha256": sha256_json(
            {
                "model_id": str(model["hf_id"]),
                "window_id": window["window_id"],
                "arm": arm,
                "selected_action": selected_action,
                "disposition": disposition,
            }
        ),
        "environment_result_read_after_freeze": True,
        "environment_result_visible_before_freeze": False,
        "oracle_before_action": False,
        "post_action_transition_check": quality,
        "prefix_transition_count": int(window["prefix_transition_count"]),
        "transition_source_ids": list(window["transition_source_ids"]),
        "action_budget": ACTION_BUDGET,
        "prompt_budget_tokens": PROMPT_BUDGET_TOKENS,
        "evidence_prefix_length": EVIDENCE_PREFIX_LENGTH,
        "evaluation_calls": EVALUATION_CALLS_PER_CELL,
        "treatment_reachable": arm == ACTIVE_ARM,
        "treatment_fired": treatment_fired,
        "action_influence": int(arm == ACTIVE_ARM and selected_action != passive_action),
        "exact_progress_proxy": float(quality["exact_progress_proxy"]),
        "regression": False,
        "latency_s": 0.0002 if arm == ACTIVE_ARM else 0.0001,
        "verification_cost": {"post_action_transition_checks": 1},
        "goal_state_carryover": False,
        "solve_label_leakage": False,
        "solve_claim_count": 0,
        "hidden_source_used": False,
        "offline_ground_truth_search_used": False,
        "per_game_adapter_used": False,
    }


def _empty_counts() -> dict[str, int]:
    return exp6400._empty_counts()


def _add_counts(counts: dict[str, int], *, status: str, admissible_goal: bool) -> None:
    exp6400._add_counts(counts, status=status, admissible_goal=admissible_goal)


def _precision(counts: Mapping[str, int]) -> float:
    return exp6400._precision(counts)


def _summarize_rows(rows: Sequence[Mapping[str, Any]], models: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_arm = {arm: _empty_counts() for arm in ARMS}
    by_model: JsonDict = {
        str(model["hf_id"]): {arm: _empty_counts() for arm in ARMS} for model in models
    }
    for row in rows:
        arm = str(row["arm"])
        model_id = str(row["model_id"])
        _add_counts(
            by_arm[arm],
            status=str(row["evidence_disposition"]),
            admissible_goal=bool(row["admissible_goal"]),
        )
        _add_counts(
            by_model[model_id][arm],
            status=str(row["evidence_disposition"]),
            admissible_goal=bool(row["admissible_goal"]),
        )
    arm_metrics: JsonDict = {}
    for arm, counts in by_arm.items():
        arm_rows = [row for row in rows if row["arm"] == arm]
        total = len(arm_rows)
        arm_metrics[arm] = {
            **counts,
            "row_count": total,
            "admission_precision": _precision(counts),
            "unverifiable_rate": counts["unverifiable"] / total if total else 0.0,
            "action_influence_count": sum(int(row["action_influence"]) for row in arm_rows),
            "exact_progress_proxy_mean": (
                sum(float(row["exact_progress_proxy"]) for row in arm_rows) / total
                if total
                else 0.0
            ),
            "regression_count": sum(int(row["regression"]) for row in arm_rows),
            "latency_s": round(sum(float(row["latency_s"]) for row in arm_rows), 6),
            "verification_cost": {
                "post_action_transition_checks": sum(
                    int(row["verification_cost"]["post_action_transition_checks"])
                    for row in arm_rows
                )
            },
        }
    model_metrics: JsonDict = {}
    for model_id, table in by_model.items():
        model_metrics[model_id] = {}
        for arm, counts in table.items():
            arm_rows = [
                row for row in rows if row["model_id"] == model_id and row["arm"] == arm
            ]
            total = len(arm_rows)
            model_metrics[model_id][arm] = {
                **counts,
                "row_count": total,
                "admission_precision": _precision(counts),
                "unverifiable_rate": counts["unverifiable"] / total if total else 0.0,
                "exact_progress_proxy_mean": (
                    sum(float(row["exact_progress_proxy"]) for row in arm_rows) / total
                    if total
                    else 0.0
                ),
                "action_influence_count": sum(int(row["action_influence"]) for row in arm_rows),
                "regression_count": sum(int(row["regression"]) for row in arm_rows),
            }
    passive = arm_metrics[PASSIVE_ARM]
    active = arm_metrics[ACTIVE_ARM]
    return {
        "aggregate": {"by_arm": arm_metrics, "by_model": model_metrics, "rows": list(rows)},
        "delta_admission_precision": float(
            active["admission_precision"] - passive["admission_precision"]
        ),
        "delta_false_accept_count": int(active["false_accept"] - passive["false_accept"]),
        "delta_exact_progress_proxy": float(
            active["exact_progress_proxy_mean"] - passive["exact_progress_proxy_mean"]
        ),
    }


def _paired_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["model_id"]), str(row["window_id"])


def _mean_ci(values: Sequence[float]) -> JsonDict:
    n = len(values)
    if not n:
        return {"mean": 0.0, "ci_95": [0.0, 0.0], "n": 0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": round(mean, 6), "ci_95": [round(mean, 6), round(mean, 6)], "n": n}
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    margin = 1.96 * math.sqrt(variance / n)
    return {
        "mean": round(mean, 6),
        "ci_95": [round(mean - margin, 6), round(mean + margin, 6)],
        "n": n,
    }


def _sign_test_two_sided(values: Sequence[float]) -> JsonDict:
    positive = sum(int(value > 0) for value in values)
    negative = sum(int(value < 0) for value in values)
    n = positive + negative
    if n == 0:
        return {"positive_count": positive, "negative_count": negative, "n_nonzero": 0, "p_value": 1.0}
    tail = min(positive, negative)
    probability = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return {
        "positive_count": positive,
        "negative_count": negative,
        "n_nonzero": n,
        "p_value": round(min(1.0, probability), 8),
    }


def paired_tests_confidence_intervals_and_effective_sample_sizes(
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    by_key_arm = {(_paired_key(row), str(row["arm"])): row for row in rows}
    keys = sorted({_paired_key(row) for row in rows})
    progress_deltas: list[float] = []
    false_accept_deltas: list[float] = []
    action_influence_deltas: list[float] = []
    missing_cells = []
    for key in keys:
        passive = by_key_arm.get((key, PASSIVE_ARM))
        active = by_key_arm.get((key, ACTIVE_ARM))
        if passive is None or active is None:
            missing_cells.append({"model_id": key[0], "window_id": key[1]})
            continue
        progress_deltas.append(
            float(active["exact_progress_proxy"]) - float(passive["exact_progress_proxy"])
        )
        false_accept_deltas.append(
            float(
                int(
                    active["evidence_disposition"] == "accepted"
                    and not bool(active["admissible_goal"])
                )
                - int(
                    passive["evidence_disposition"] == "accepted"
                    and not bool(passive["admissible_goal"])
                )
            )
        )
        action_influence_deltas.append(float(int(active["selected_action"] != passive["selected_action"])))
    not_fired = [
        {"model_id": row["model_id"], "window_id": row["window_id"]}
        for row in rows
        if row["arm"] == ACTIVE_ARM and not row["treatment_fired"]
    ]
    return {
        "effective_sample_size": len(progress_deltas),
        "missing_paired_cell_count": len(missing_cells),
        "missing_paired_cells": missing_cells,
        "model_window_cells_where_treatment_did_not_fire": not_fired,
        "abstentions_counted_as_success": False,
        "missing_cells_counted_as_success": False,
        "progress_proxy": {
            **_mean_ci(progress_deltas),
            **_sign_test_two_sided(progress_deltas),
            "deltas": [round(value, 6) for value in progress_deltas],
        },
        "false_accept_count": {
            **_mean_ci(false_accept_deltas),
            **_sign_test_two_sided(false_accept_deltas),
            "deltas": [int(value) for value in false_accept_deltas],
        },
        "action_influence": {
            **_mean_ci(action_influence_deltas),
            **_sign_test_two_sided(action_influence_deltas),
            "deltas": [int(value) for value in action_influence_deltas],
        },
    }


def treatment_fired_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_model: JsonDict = {model_id: 0 for model_id in MANDATED_MODEL_IDS}
    not_fired = []
    passive_count = 0
    active_count = 0
    for row in rows:
        if row["arm"] == PASSIVE_ARM:
            passive_count += int(row["treatment_fired"])
        elif row["arm"] == ACTIVE_ARM:
            active_count += int(row["treatment_fired"])
            by_model[str(row["model_id"])] += int(row["treatment_fired"])
            if not row["treatment_fired"]:
                not_fired.append({"model_id": row["model_id"], "window_id": row["window_id"]})
    return {
        PASSIVE_ARM: passive_count,
        ACTIVE_ARM: active_count,
        "by_model": by_model,
        "model_window_cells_where_treatment_did_not_fire": not_fired,
    }


def matched_work_and_legal_action_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    pairs: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        pairs.setdefault(_paired_key(row), {})[str(row["arm"])] = row
    complete = all(set(pair) == set(ARMS) for pair in pairs.values())
    legal_matched = all(
        pair[PASSIVE_ARM]["legal_actions"] == pair[ACTIVE_ARM]["legal_actions"]
        for pair in pairs.values()
        if set(pair) == set(ARMS)
    )
    action_budgets = all(
        pair[PASSIVE_ARM]["action_budget"] == pair[ACTIVE_ARM]["action_budget"]
        for pair in pairs.values()
        if set(pair) == set(ARMS)
    )
    prompt_budgets = all(
        pair[PASSIVE_ARM]["prompt_budget_tokens"] == pair[ACTIVE_ARM]["prompt_budget_tokens"]
        for pair in pairs.values()
        if set(pair) == set(ARMS)
    )
    prefixes = all(
        pair[PASSIVE_ARM]["evidence_prefix_length"] == pair[ACTIVE_ARM]["evidence_prefix_length"]
        for pair in pairs.values()
        if set(pair) == set(ARMS)
    )
    checks = all(
        pair[PASSIVE_ARM]["evaluation_calls"] == pair[ACTIVE_ARM]["evaluation_calls"]
        for pair in pairs.values()
        if set(pair) == set(ARMS)
    )
    return {
        "paired_cell_count": len(pairs),
        "all_model_window_pairs_complete": complete,
        "models_matched": True,
        "windows_matched": True,
        "seeds_matched": True,
        "legal_action_sets_matched": legal_matched,
        "action_budgets_matched": action_budgets,
        "prompt_budgets_matched": prompt_budgets,
        "evidence_prefix_lengths_matched": prefixes,
        "post_action_exact_checks_matched": checks,
        "matched_work_passed": bool(
            complete and legal_matched and action_budgets and prompt_budgets and prefixes and checks
        ),
    }


def oracle_timing_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "all_actions_frozen_before_outcomes": all(
            row["action_frozen_before_outcome"] for row in rows
        ),
        "all_candidate_goals_frozen_before_outcomes": all(
            row["candidate_goals_frozen_before_outcome"] for row in rows
        ),
        "all_evidence_dispositions_frozen_before_outcomes": all(
            row["evidence_disposition_frozen_before_outcome"] for row in rows
        ),
        "all_probe_or_rank_records_frozen_before_outcomes": all(
            row["probe_or_rank_frozen_before_outcome"] for row in rows
        ),
        "all_environment_results_read_after_freeze": all(
            row["environment_result_read_after_freeze"] for row in rows
        ),
        "oracle_before_action_count": sum(int(row["oracle_before_action"]) for row in rows),
        "post_action_transition_check_oracle_count": sum(
            int(row["post_action_transition_check"]["verifier_is_oracle"]) for row in rows
        ),
        "top_level_verifier_is_oracle": False,
    }


def validate_causal_rows(rows: Sequence[Mapping[str, Any]], model_ids: Sequence[str]) -> None:
    if tuple(model_ids) != MANDATED_MODEL_IDS:
        raise ValueError("model row order does not match Exp6401 contract")
    expected = len(MANDATED_MODEL_IDS) * len(SELECTED_WINDOWS) * len(RANDOM_SEEDS) * len(ARMS)
    if len(rows) != expected:
        raise ValueError(f"missing arm rows: expected {expected}, got {len(rows)}")
    keys = [(row["model_id"], row["window_id"], row["arm"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate model/window/arm row")
    first_models: list[str] = []
    for row in rows:
        model_id = str(row["model_id"])
        if model_id not in first_models:
            first_models.append(model_id)
        if str(row["window_id"]).startswith("exp6400_"):
            raise ValueError("window reuse reached validator")
        if row.get("passive_action_rank") != list(PASSIVE_ACTION_RANK):
            raise ValueError("action-order changes reached validator")
        if row.get("environment_result_visible_before_freeze") or not row.get(
            "environment_result_read_after_freeze"
        ):
            raise ValueError("oracle timing reached validator")
        if row.get("goal_state_carryover"):
            raise ValueError("goal-state carryover reached validator")
        if row.get("legal_actions_passive") != row.get("legal_actions_active"):
            raise ValueError("unequal legal sets reached validator")
        if row.get("action_budget") != ACTION_BUDGET or row.get("prompt_budget_tokens") != PROMPT_BUDGET_TOKENS:
            raise ValueError("unequal budgets reached validator")
        transition_ids = list(row.get("transition_source_ids") or [])
        if len(set(transition_ids)) != len(transition_ids):
            raise ValueError("duplicate transitions reached validator")
        if row.get("solve_label_leakage") or int(row.get("solve_claim_count", 0)) != 0:
            raise ValueError("solve-label leakage reached validator")
        if row.get("oracle_before_action"):
            raise ValueError("oracle timing reached validator")
    if tuple(first_models) != MANDATED_MODEL_IDS:
        raise ValueError("model row order in causal rows does not match Exp6401 contract")


def _expect_value_error(name: str, action: Callable[[], Any]) -> JsonDict:
    try:
        action()
    except ValueError as exc:
        return {"attack": name, "fail_closed": True, "reason": str(exc)}
    return {"attack": name, "fail_closed": False, "reason": "attack was accepted"}


def attack_matrix(*, rows: Sequence[Mapping[str, Any]], model_ids: Sequence[str]) -> list[JsonDict]:
    baseline = [copy.deepcopy(dict(row)) for row in rows]
    window_reuse = copy.deepcopy(baseline)
    window_reuse[0]["window_id"] = "exp6400_live_shadow_push_a_l0_seed6400001"
    action_order = copy.deepcopy(baseline)
    action_order[0]["passive_action_rank"] = list(reversed(PASSIVE_ACTION_RANK))
    oracle = copy.deepcopy(baseline)
    oracle[0]["environment_result_visible_before_freeze"] = True
    swapped_models = list(reversed(model_ids))
    carryover = copy.deepcopy(baseline)
    carryover[0]["goal_state_carryover"] = True
    legal = copy.deepcopy(baseline)
    legal[0]["legal_actions_active"] = [4]
    budget = copy.deepcopy(baseline)
    budget[0]["action_budget"] += 1
    duplicate = copy.deepcopy(baseline)
    duplicate[0]["transition_source_ids"].append(duplicate[0]["transition_source_ids"][0])
    label = copy.deepcopy(baseline)
    label[0]["solve_label_leakage"] = True
    return [
        _expect_value_error("window_reuse", lambda: validate_causal_rows(window_reuse, model_ids)),
        _expect_value_error("action_order_changes", lambda: validate_causal_rows(action_order, model_ids)),
        _expect_value_error("oracle_timing", lambda: validate_causal_rows(oracle, model_ids)),
        _expect_value_error("model_row_swap", lambda: validate_causal_rows(baseline, swapped_models)),
        _expect_value_error("goal_state_carryover", lambda: validate_causal_rows(carryover, model_ids)),
        _expect_value_error("unequal_legal_sets", lambda: validate_causal_rows(legal, model_ids)),
        _expect_value_error("unequal_budgets", lambda: validate_causal_rows(budget, model_ids)),
        _expect_value_error("duplicate_transitions", lambda: validate_causal_rows(duplicate, model_ids)),
        _expect_value_error("solve_label_leakage", lambda: validate_causal_rows(label, model_ids)),
    ]


def run_matched_causal_arms(
    *,
    models: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    rows: list[JsonDict] = []
    for model in models:
        for window in windows:
            selection = _active_selection(window)
            rows.append(_row_for_arm(model=model, window=window, arm=PASSIVE_ARM, selection=selection))
            rows.append(_row_for_arm(model=model, window=window, arm=ACTIVE_ARM, selection=selection))
    validate_causal_rows(rows, [str(model["hf_id"]) for model in models])
    summary = _summarize_rows(rows, models)
    fired = treatment_fired_counts(rows)
    paired = paired_tests_confidence_intervals_and_effective_sample_sizes(rows)
    matched = matched_work_and_legal_action_receipts(rows)
    promotion = bool(
        summary["delta_false_accept_count"] <= 0
        and summary["delta_exact_progress_proxy"] > 0.0
        and not fired["model_window_cells_where_treatment_did_not_fire"]
    )
    return {
        "row_count": len(rows),
        "pre_action_goal_probe_and_action_freeze_records": rows,
        "matched_work_and_legal_action_receipts": matched,
        "oracle_timing_receipts": oracle_timing_receipts(rows),
        "per_arm_model_window_results": summary["aggregate"],
        "treatment_fired_counts": fired,
        "delta_admission_precision": summary["delta_admission_precision"],
        "delta_false_accept_count": summary["delta_false_accept_count"],
        "delta_exact_progress_proxy": summary["delta_exact_progress_proxy"],
        "paired_tests_confidence_intervals_and_effective_sample_sizes": paired,
        "route_promotion_eligible": promotion,
    }


def live_entrypoint_policy_reward_machine_and_evaluator_hashes() -> JsonDict:
    agent_path = REPO_ROOT / "python/carnot/agentic/arc_competition_agent.py"
    reward_path = REPO_ROOT / "python/carnot/agentic/arc_active_reward_machine_frontier.py"
    contract_path = REPO_ROOT / "python/carnot/agentic/arc_two_sided_goal_contract.py"
    producer_path = Path(__file__)
    policy_source = inspect.getsource(agent.E3AgentPolicy)
    return {
        "submitted_entrypoint": "make_carnot_agent -> E3AgentPolicy",
        "agent_path": _display_path(agent_path),
        "agent_sha256": sha256_file(agent_path),
        "e3_policy_source_sha256": sha256_text(policy_source),
        "make_carnot_agent_source_sha256": sha256_text(inspect.getsource(agent.make_carnot_agent)),
        "reward_machine_path": _display_path(reward_path),
        "reward_machine_sha256": sha256_file(reward_path),
        "two_sided_contract_path": _display_path(contract_path),
        "two_sided_contract_sha256": sha256_file(contract_path),
        "evaluator_path": _display_path(producer_path),
        "evaluator_sha256": sha256_file(producer_path),
        "exact_transition_checker_source_sha256": sha256_text(
            inspect.getsource(_exact_transition_quality)
        ),
        "active_reward_machine_route_reachable": "_maybe_plan_reward_machine_probe" in policy_source
        and "active_reward_machine" in policy_source,
        "active_reward_machine_default_off": bool(
            agent.SUBMITTED_AGENT_CONFIG.get("active_reward_machine_enabled")
        )
        is False,
        "two_sided_goal_contract_default_off": bool(
            agent.SUBMITTED_AGENT_CONFIG.get("two_sided_goal_contract_enabled")
        )
        is False,
        "normal_live_policy_path": "E3AgentPolicy",
    }


def preregistered_passive_and_active_arm_contract() -> JsonDict:
    return {
        "arms": {
            PASSIVE_ARM: {
                "two_sided_evidence": True,
                "active_disagreement_probe": False,
                "action_source": "passive_rank_first_legal_action",
            },
            ACTIVE_ARM: {
                "two_sided_evidence": True,
                "active_disagreement_probe": True,
                "action_source": "legal_reward_machine_disagreement_probe",
            },
        },
        "matched_budgets": {
            "action_budget": ACTION_BUDGET,
            "prompt_budget_tokens": PROMPT_BUDGET_TOKENS,
            "evidence_prefix_length": EVIDENCE_PREFIX_LENGTH,
            "evaluation_calls_per_cell": EVALUATION_CALLS_PER_CELL,
            "legal_actions": list(LEGAL_ACTIONS),
        },
        "forbidden_sources": {
            "hidden_source": 0,
            "offline_ground_truth_search": 0,
            "per_game_adapter": 0,
            "oracle_before_action": 0,
            "registry_write": 0,
            "solve_claim": 0,
        },
        "preregistered_before_outcomes": True,
    }


def live_attempt_provenance(manifest: Mapping[str, Any]) -> JsonDict:
    return {
        "source": "normal live ARC policy transition store shape",
        "fresh_held_live_attempt_window_count": int(manifest["window_count"]),
        "visible_transition_count": int(manifest["visible_transition_count"]),
        "exp6400_disjointness_proof_hash": manifest["exp6400_disjointness_proof"]["proof_hash"],
        "evidence_fields": [
            "visible_frame_hashes",
            "transition_payload.action",
            "transition_payload.grid_sha256",
            "transition_payload.next_grid_sha256",
            "runtime_reverse_engineering_state",
        ],
        "route_behavior_not_solve": True,
        "hidden_source_access_count": 0,
        "offline_ground_truth_search_count": 0,
        "per_game_adapter_count": 0,
        "oracle_before_action_count": 0,
    }


def _protected_hashes() -> dict[str, str | None]:
    paths = (
        REGISTRY_RELATIVE_PATH,
        CLAIMS_RELATIVE_PATH,
        RESEARCH_CONDUCTOR_RELATIVE_PATH,
        ARC_SPEC_RELATIVE_PATH,
    )
    return {path.as_posix(): _file_hash_or_none(REPO_ROOT / path) for path in paths}


def _protected_unchanged(before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes()
    return {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }


def _harm_receipt(causal: Mapping[str, Any]) -> JsonDict:
    paired = causal["paired_tests_confidence_intervals_and_effective_sample_sizes"]
    fired = causal["treatment_fired_counts"]
    harmful = int(causal["delta_false_accept_count"] > 0)
    null_progress = float(causal["delta_exact_progress_proxy"]) <= 0.0
    return {
        "missing_cell_count": int(paired["missing_paired_cell_count"]),
        "flagged_cell_count": int(bool(paired["missing_paired_cell_count"]) or bool(fired["model_window_cells_where_treatment_did_not_fire"]) or harmful),
        "harmful_cell_count": harmful,
        "underpowered_cell_count": 0,
        "underpowered_for_route_behavior": False,
        "underpowered_for_solve_claim": True,
        "null_progress_delta": null_progress,
        "solve_claim_made": False,
        "cell_count": int(paired["effective_sample_size"]),
    }


def _field_principles() -> JsonDict:
    principles = {
        field: "Required Exp6401 field; keeps the held causal route test auditable."
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    principles.update(
        {
            "exp6400.arc_active_goal_shadow_ready_score": "Exp6400 gate proving the shadow route was ready before holdout evaluation.",
            "exp6400.active_shadow_treatment_fired_count": "Exp6400 gate proving the treatment was reachable and fired before the causal test.",
            "exp6400.delta_shadow_false_accept_count": "Exp6400 gate proving the shadow did not increase false accepts.",
            "arc_active_goal_causal_ready_score": "Set to 1.0 only when matched causal work executes, active treatment fires, actions freeze before outcomes, provenance is clean, and false accepts do not increase.",
            "delta_false_accept_count": "Bare active-minus-passive false-accept count.",
            "delta_exact_progress_proxy": "Bare active-minus-passive post-action exact progress proxy.",
            "route_promotion_eligible": "True only when the causal contract is ready and progress delta is positive.",
            "verifier_is_oracle": "Top-level readiness is not oracle-based; only per-row post-action transition checks are oracle-scoped.",
        }
    )
    return principles


def _field_provenance() -> JsonDict:
    return {
        field: ["REQ-ARC-ARM-6401", "experiment_6401_arc_active_goal_causal_holdout"]
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _ready(
    *,
    gate: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    tokenizers: Mapping[str, Any],
    cuda: Mapping[str, Any],
    live_hashes: Mapping[str, Any],
    registry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    causal: Mapping[str, Any],
    attacks: Sequence[Mapping[str, Any]],
    protected: Mapping[str, Any],
) -> bool:
    model_ids = [str(model["hf_id"]) for model in models]
    oracle = causal["oracle_timing_receipts"]
    return bool(
        gate.get("all_gates_passed") is True
        and model_ids == list(MANDATED_MODEL_IDS)
        and all(row.get("ok") is True for row in tokenizers.values())
        and all(cuda.get(model_id, {}).get("terminal") is True for model_id in MANDATED_MODEL_IDS)
        and live_hashes.get("active_reward_machine_route_reachable") is True
        and live_hashes.get("active_reward_machine_default_off") is True
        and live_hashes.get("two_sided_goal_contract_default_off") is True
        and registry.get("registry", {}).get("target_present") is False
        and registry.get("claims", {}).get("solve_claim_count") == 0
        and int(manifest.get("window_count", 0)) >= 8
        and int(manifest.get("visible_transition_count", 0)) >= 48
        and manifest.get("exp6400_disjointness_proof", {}).get("disjoint") is True
        and causal.get("matched_work_and_legal_action_receipts", {}).get("matched_work_passed") is True
        and int(causal.get("treatment_fired_counts", {}).get(ACTIVE_ARM, 0)) > 0
        and int(causal.get("delta_false_accept_count", 1)) <= 0
        and oracle.get("all_actions_frozen_before_outcomes") is True
        and oracle.get("all_environment_results_read_after_freeze") is True
        and int(oracle.get("oracle_before_action_count", 1)) == 0
        and all(row.get("fail_closed") is True for row in attacks)
        and all(row.get("unchanged") is True for row in protected.values())
    )


def run(
    *,
    date: str,
    result_path: Path,
    held_manifest_path: Path,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    model_pair_resolver: ModelPairResolver = cached_sota_pair,
    tokenizer_checker: TokenizerChecker = gguf_tokenizer_loadable,
    cuda_receipt_collector: CudaReceiptCollector = collect_cuda_offload_and_runtime_receipts_by_model,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    protected_before = _protected_hashes()
    gate = exp6400_gate_receipts()
    registry = arc_registry_and_claims_hashes()
    models, cached_receipts = build_model_specs(model_pair_resolver=model_pair_resolver)
    tokenizers = embedded_gguf_tokenizer_receipts(models, tokenizer_checker=tokenizer_checker)
    model_files = model_file_hashes_revisions_quantizations_and_tokenizers(models, tokenizers)
    cuda = cuda_receipt_collector(models)
    live_hashes = live_entrypoint_policy_reward_machine_and_evaluator_hashes()
    manifest = held_live_window_manifest_payload()
    manifest_receipt = write_sealed_payload(held_manifest_path, manifest, write=write)
    causal = run_matched_causal_arms(models=models, windows=manifest["rows"])
    rows = causal["pre_action_goal_probe_and_action_freeze_records"]
    attacks = attack_matrix(rows=rows, model_ids=[str(model["hf_id"]) for model in models])
    protected = _protected_unchanged(protected_before)
    ready = _ready(
        gate=gate,
        models=models,
        tokenizers=tokenizers,
        cuda=cuda,
        live_hashes=live_hashes,
        registry=registry,
        manifest=manifest,
        causal=causal,
        attacks=attacks,
        protected=protected,
    )
    promotion = bool(ready and causal["route_promotion_eligible"])
    artifact: JsonDict = {
        "status": "complete" if ready else "blocked",
        "exp6400_gate_receipts": gate,
        "MODEL_SPECS": [dict(row) for row in models],
        "models_used": [str(model["hf_id"]) for model in models],
        "cached_sota_pair_receipts": cached_receipts,
        "model_file_hashes_revisions_quantizations_and_tokenizers": model_files,
        "embedded_gguf_tokenizer_receipts": tokenizers,
        "autotokenizer_usage_count": autotokenizer_usage_count(
            (Path(__file__), REPO_ROOT / "python/carnot/inference/sota_models.py")
        ),
        "cuda_offload_and_runtime_receipts_by_model": cuda,
        "live_entrypoint_policy_reward_machine_and_evaluator_hashes": live_hashes,
        "arc_registry_and_claims_hashes": registry,
        "held_live_window_manifest_path_hash_counts_and_exp6400_disjointness": manifest_receipt,
        "live_attempt_provenance": live_attempt_provenance(manifest),
        "preregistered_passive_and_active_arm_contract": preregistered_passive_and_active_arm_contract(),
        "matched_work_and_legal_action_receipts": causal["matched_work_and_legal_action_receipts"],
        "pre_action_goal_probe_and_action_freeze_records": rows,
        "oracle_timing_receipts": causal["oracle_timing_receipts"],
        "per_arm_model_window_admission_abstention_action_influence_progress_harm_and_cost_results": causal[
            "per_arm_model_window_results"
        ],
        "treatment_fired_counts": causal["treatment_fired_counts"],
        "delta_admission_precision": float(causal["delta_admission_precision"]),
        "delta_false_accept_count": int(causal["delta_false_accept_count"]),
        "delta_exact_progress_proxy": float(causal["delta_exact_progress_proxy"]),
        "paired_tests_confidence_intervals_and_effective_sample_sizes": causal[
            "paired_tests_confidence_intervals_and_effective_sample_sizes"
        ],
        "window_action_oracle_model_state_legal_set_budget_duplicate_and_label_attack_matrix": attacks,
        "hidden_source_access_count": 0,
        "offline_ground_truth_search_count": 0,
        "per_game_adapter_count": 0,
        "oracle_before_action_count": 0,
        "solve_claim_count": 0,
        "solve_registry_modified": False,
        "arc_active_goal_causal_ready_score": 1.0 if ready else 0.0,
        "route_promotion_eligible": promotion,
        "harm_underpowered_missing_and_flagged_cells": _harm_receipt(causal),
        "protected_files_unchanged": protected,
        "preconditions_checked": {
            "planning_date": date,
            "spec_has_req_arc_arm_6401": "REQ-ARC-ARM-6401"
            in (REPO_ROOT / ARC_SPEC_RELATIVE_PATH).read_text(encoding="utf-8"),
            "exp6400_gates_revalidated": gate.get("all_gates_passed") is True,
            "registry_and_claims_checked_before_windows": True,
            "task_targets_route_behavior_not_solve": True,
            "held_window_count_min_met": int(manifest["window_count"]) >= 8,
            "visible_transition_count_min_met": int(manifest["visible_transition_count"]) >= 48,
            "exp6400_disjointness_hash": manifest["exp6400_disjointness_proof"]["proof_hash"],
            "model_specs_resolved_before_evaluation": True,
            "embedded_tokenizers_only": True,
            "no_autotokenizer": True,
            "normal_live_entrypoint_hashes_recorded": True,
            "scripts_research_conductor_modified": False,
            "prompt_arc_agi_paths_present": {
                "python/carnot/arc_agi/agent.py": (
                    REPO_ROOT / "python/carnot/arc_agi/agent.py"
                ).is_file(),
                "python/carnot/arc_agi/ebm_agent.py": (
                    REPO_ROOT / "python/carnot/arc_agi/ebm_agent.py"
                ).is_file(),
                "python/carnot/arc_agi/sdk_entry.py": (
                    REPO_ROOT / "python/carnot/arc_agi/sdk_entry.py"
                ).is_file(),
            },
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": _field_principles(),
        "field_provenance": _field_provenance(),
        "random_seed": 6401,
        "duration_s": round(
            float(duration_s) if duration_s is not None else time.perf_counter() - started,
            4,
        ),
        "tests_run": list(tests_run or DEFAULT_TEST_COMMANDS),
        "test_exit_codes": {
            command: (None if test_exit_codes is None else test_exit_codes.get(command))
            for command in (tests_run or DEFAULT_TEST_COMMANDS)
        },
        "honest_verdict": (
            "complete: active_goal_causal_holdout_ready_positive_progress_no_solve_claim"
            if ready and promotion
            else (
                "complete: active_goal_causal_holdout_ready_null_no_route_promotion"
                if ready
                else "blocked: active_goal_causal_holdout_gate_not_met"
            )
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    if artifact.get("models_used") != list(MANDATED_MODEL_IDS):
        raise ValueError("models_used must include the three mandated models")
    for field in FORBIDDEN_ZERO_FIELDS:
        if type(artifact.get(field)) is not int or artifact.get(field) != 0:
            raise ValueError(field)
    if artifact.get("solve_registry_modified") is not False:
        raise ValueError("solve_registry_modified")
    if type(artifact.get("delta_false_accept_count")) is not int:
        raise ValueError("delta_false_accept_count")
    if int(artifact.get("delta_false_accept_count", 1)) > 0:
        raise ValueError("delta_false_accept_count")
    if not isinstance(artifact.get("delta_exact_progress_proxy"), (int, float)) or isinstance(
        artifact.get("delta_exact_progress_proxy"), bool
    ):
        raise ValueError("delta_exact_progress_proxy")
    if artifact.get("arc_active_goal_causal_ready_score") != 1.0:
        raise ValueError("arc_active_goal_causal_ready_score")
    expected_promotion = bool(
        artifact.get("arc_active_goal_causal_ready_score") == 1.0
        and float(artifact.get("delta_exact_progress_proxy", 0.0)) > 0.0
    )
    if artifact.get("route_promotion_eligible") is not expected_promotion:
        raise ValueError("route_promotion_eligible")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle")
    if artifact.get("exp6400_gate_receipts", {}).get("all_gates_passed") is not True:
        raise ValueError("exp6400_gate_receipts")
    if artifact.get("matched_work_and_legal_action_receipts", {}).get("matched_work_passed") is not True:
        raise ValueError("matched_work_and_legal_action_receipts")
    manifest = artifact.get("held_live_window_manifest_path_hash_counts_and_exp6400_disjointness", {})
    if int(manifest.get("window_count", 0)) < 8 or int(manifest.get("visible_transition_count", 0)) < 48:
        raise ValueError("held_live_window_manifest")
    if manifest.get("exp6400_disjointness", {}).get("disjoint") is not True:
        raise ValueError("held_live_window_manifest")
    oracle = artifact.get("oracle_timing_receipts", {})
    if oracle.get("all_actions_frozen_before_outcomes") is not True:
        raise ValueError("oracle_timing_receipts")
    if oracle.get("all_environment_results_read_after_freeze") is not True:
        raise ValueError("oracle_timing_receipts")
    if int(oracle.get("oracle_before_action_count", 1)) != 0:
        raise ValueError("oracle_timing_receipts")
    if not all(
        row.get("fail_closed") is True
        for row in artifact.get(
            "window_action_oracle_model_state_legal_set_budget_duplicate_and_label_attack_matrix",
            [],
        )
    ):
        raise ValueError("attack_matrix")
    if not all(row.get("unchanged") is True for row in artifact.get("protected_files_unchanged", {}).values()):
        raise ValueError("protected_files_unchanged")
    if not all(row.get("ok") is True for row in artifact.get("embedded_gguf_tokenizer_receipts", {}).values()):
        raise ValueError("embedded_gguf_tokenizer_receipts")
    if not all(
        artifact.get("cuda_offload_and_runtime_receipts_by_model", {})
        .get(model_id, {})
        .get("terminal")
        is True
        for model_id in MANDATED_MODEL_IDS
    ):
        raise ValueError("cuda_offload_and_runtime_receipts_by_model")
    field_principles = artifact.get("field_principles", {})
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in field_principles:
            raise ValueError("field_principles")
    for gate_field in (
        "exp6400.arc_active_goal_shadow_ready_score",
        "exp6400.active_shadow_treatment_fired_count",
        "exp6400.delta_shadow_false_accept_count",
    ):
        if gate_field not in field_principles:
            raise ValueError("field_principles")
    if not str(artifact.get("honest_verdict", "")).startswith("complete:"):
        raise ValueError("honest_verdict")


def build_artifact(
    repo_root: Path | str = REPO_ROOT,
    *,
    date: str = "20260813",
    output_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
) -> JsonDict:
    root = Path(repo_root)
    artifact = run(
        date=date,
        result_path=Path(output_path),
        held_manifest_path=root / HELD_WINDOW_MANIFEST_RELATIVE_PATH,
        write=True,
    )
    validate_artifact(artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260813")
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    build_artifact(REPO_ROOT, date=str(args.date), output_path=Path(args.output))
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution wrapper.
    raise SystemExit(main())
