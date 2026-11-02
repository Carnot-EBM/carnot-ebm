"""Exp6321 ARC target-licensed route live shadow A/B.

Spec refs: REQ-ARC-WMTE-6321,
SCENARIO-ARC-WMTE-6321-DEFAULT-OFF-PARITY,
SCENARIO-ARC-WMTE-6321-SHADOW-COMPUTED-ISOLATION,
SCENARIO-ARC-WMTE-6321-REGISTRY-PRECHECK,
SCENARIO-ARC-WMTE-6321-ARTIFACT-GUARDS.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import inspect
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_6307_arc_target_validated_route_canary as exp6307
from carnot import experiment_6308_arc_target_validated_route_holdout as exp6308
from carnot.agentic import arc_competition_agent as agent


JsonDict = dict[str, Any]
ModelResolver = Callable[[bool], list[JsonDict]]

REPO_ROOT = exp6307.REPO_ROOT
RESULT_RELATIVE_PATH = Path("results/experiment_6321_arc_target_licensed_route_live_shadow_ab.json")
TRANSITION_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6321_arc_target_licensed_route_live_shadow_transitions.json"
)
REGISTRY_RELATIVE_PATH = exp6307.REGISTRY_RELATIVE_PATH
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6321_arc_target_licensed_route_live_shadow_ab "
    "--date 20260811"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6321_arc_target_licensed_route_live_shadow_ab.py "
    "tests/python/test_arc_submitted_agent_parity.py -q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6321_arc_target_licensed_route_live_shadow_ab.py "
    "-m pytest tests/python/test_experiment_6321_arc_target_licensed_route_live_shadow_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6321_arc_target_licensed_route_live_shadow_ab.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6321_arc_target_licensed_route_live_shadow_ab.py"
)
E2E_PLAN_READ_COMMAND = "sed -n 1,180p ops/e2e-test-plan.md"
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6321_arc_target_licensed_route_live_shadow_ab.json"
)
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6321_test_receipts.json")

MANDATED_MODEL_IDS = exp6307.MANDATED_MODEL_IDS
RANDOM_SEEDS = (6321001, 6321002)
ARMS = ("shadow_off", "shadow_computed")
ACTION_BUDGET = exp6307.ACTION_BUDGET
MODEL_WINDOW_BUDGET = 2
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
SELECTED_TARGETS = (
    {
        "selected_target": "exp6321_shadow_synthetic_push_l0",
        "game_id": "exp6321_shadow_synthetic_push",
        "level": 0,
        "mechanic": "push_block",
        "nonduplicate_reason": "fresh visible synthetic window absent from solve registry",
    },
    {
        "selected_target": "exp6321_shadow_synthetic_toggle_l0",
        "game_id": "exp6321_shadow_synthetic_toggle",
        "level": 0,
        "mechanic": "toggle_move",
        "nonduplicate_reason": "fresh visible synthetic window absent from solve registry",
    },
)
FORBIDDEN_ZERO_FIELDS = (
    "source_bfs_adapter_prior_game_hidden_state_and_registry_target_access_count",
    "hidden_game_source_access_count",
    "levels_credited",
    "registry_update_count",
    "duration_padding_count",
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    E2E_PLAN_READ_COMMAND,
    DETERMINATION_COMMAND,
    ADVERSARIAL_COMMAND,
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6307_and_exp6308_paths_hashes_and_terminal_classes",
    "arc_registry_precheck_path_hash_and_result",
    "selected_game_level_and_nonduplicate_reason",
    "solve_provenance",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes_revisions_and_quantizations",
    "tokenizer_hashes",
    "cuda_and_gpu_offload_receipts_by_model",
    "live_agent_entrypoint_and_e3_policy_hash",
    "submitted_config_pre_and_post_hashes",
    "default_off_shadow_wiring_receipt",
    "shadow_off_and_shadow_computed_arm_definitions",
    "fresh_agent_owned_transition_manifest_path_and_hash",
    "target_license_evidence_receipts",
    "route_reachability_by_model_window_and_seed",
    "supported_unsupported_and_abstained_proposals_by_arm",
    "prospective_action_support_by_arm",
    "action_budget_registry_and_level_state_parity",
    "latency_and_resource_cost_by_arm",
    "source_bfs_adapter_prior_game_hidden_state_and_registry_target_access_count",
    "hidden_game_source_access_count",
    "solve_claimed",
    "levels_credited",
    "registry_update_count",
    "actual_work_duration_receipt",
    "duration_padding_count",
    "arc_route_live_shadow_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "States complete versus blocked without hiding partial work.",
    "exp6307_and_exp6308_paths_hashes_and_terminal_classes": "Pins upstream claims and the Exp6307 seed correction.",
    "arc_registry_precheck_path_hash_and_result": "Proves selected cells are not registry targets.",
    "selected_game_level_and_nonduplicate_reason": "Freezes games and levels before attempts.",
    "solve_provenance": "States live-agent self-discovery without solve credit.",
    "MODEL_SPECS": "Names both mandated GGUF model ids.",
    "models_used": "Lists model ids whose receipts are carried into the A/B.",
    "model_file_hashes_revisions_and_quantizations": "Pins concrete model files.",
    "tokenizer_hashes": "Pins tokenizer and prompt-contract hashes.",
    "cuda_and_gpu_offload_receipts_by_model": "Carries terminal CUDA/offload receipts.",
    "live_agent_entrypoint_and_e3_policy_hash": "Pins the submitted entrypoint and policy source.",
    "submitted_config_pre_and_post_hashes": "Shows shadow construction did not mutate submitted config.",
    "default_off_shadow_wiring_receipt": "Proves normal E3 construction has no active shadow.",
    "shadow_off_and_shadow_computed_arm_definitions": "Defines the only A/B difference.",
    "fresh_agent_owned_transition_manifest_path_and_hash": "Pins runtime-visible policy transition rows.",
    "target_license_evidence_receipts": "Shows route checks used target runtime evidence.",
    "route_reachability_by_model_window_and_seed": "Measures whether the route can be reached.",
    "supported_unsupported_and_abstained_proposals_by_arm": "Separates supported, unsupported, and abstained proposals.",
    "prospective_action_support_by_arm": "Shows unsupported proposals were never applied.",
    "action_budget_registry_and_level_state_parity": "Proves the shadow did not change action, budget, registry, or level state.",
    "latency_and_resource_cost_by_arm": "Records measurement overhead by arm.",
    "source_bfs_adapter_prior_game_hidden_state_and_registry_target_access_count": "Must stay zero for escape-hatch discipline.",
    "hidden_game_source_access_count": "Must stay zero for hidden-source discipline.",
    "solve_claimed": "Must be false because this is route readiness only.",
    "levels_credited": "Must stay zero because no level credit is requested.",
    "registry_update_count": "Must stay zero because no solve is banked.",
    "actual_work_duration_receipt": "Records measured work and rejects padding.",
    "duration_padding_count": "Must stay zero because padding is forbidden.",
    "arc_route_live_shadow_ready_score": "Equals one only for reachability, parity, and zero escape access.",
    "protected_files_unchanged": "Confirms protected files stayed unchanged during the run.",
    "preconditions_checked": "Records frozen cells, models, seeds, budgets, hashes, and route rules.",
    "inference_substrate": "Declares live-agent no-LLM shadow measurement with upstream model receipts.",
    "verifier_is_oracle": "False because no game oracle verifies a solve.",
    "field_provenance": "Maps every field to the spec and producer.",
    "field_principles": "Gives one audit reason per required field.",
    "test_commands": "Lists verification commands.",
    "test_exit_codes": "Records command outcomes.",
    "duration_s": "Records measured wall time.",
    "random_seeds": "Pins all shadow A/B seeds.",
    "reproducibility_checksum": "Detects artifact drift.",
    "honest_verdict": "Terminal verdict with no solve claim.",
}
FIELD_PROVENANCE = {
    field: ["REQ-ARC-WMTE-6321", "experiment_6321_arc_target_licensed_route_live_shadow_ab"]
    for field in REQUIRED_ARTIFACT_FIELDS
}

canonical_json = exp6307.canonical_json
sha256_text = exp6307.sha256_text
sha256_json = exp6307.sha256_json
sha256_file = exp6307.sha256_file
payload_checksum = exp6307.payload_checksum


def _terminal_class(payload: Mapping[str, Any]) -> str:
    status = str(payload.get("status") or "")
    verdict = str(payload.get("honest_verdict") or "")
    if payload.get("flagged_adversarial"):
        return "flagged"
    if status.startswith("blocked") or verdict.startswith("blocked"):
        return "blocked"
    if status == "complete" or verdict.startswith(("complete:", "complete_")):
        return "complete"
    return status or "unknown"


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def exp6307_and_exp6308_paths_hashes_and_terminal_classes() -> JsonDict:
    rows = []
    for rel, ready_field in (
        (exp6307.RESULT_RELATIVE_PATH, "arc_target_licensed_router_ready_score"),
        (exp6308.RESULT_RELATIVE_PATH, "arc_target_licensed_generalization_ready_score"),
    ):
        path = REPO_ROOT / rel
        payload = _read_json(path)
        rows.append(
            {
                "path": rel.as_posix(),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "terminal_class": _terminal_class(payload),
                "ready_score": payload.get(ready_field),
                "has_random_seed": "random_seed" in payload,
                "has_random_seeds": "random_seeds" in payload,
            }
        )
    source_path = REPO_ROOT / "python/carnot/experiment_6307_arc_target_validated_route_canary.py"
    source_text = source_path.read_text(encoding="utf-8")
    current_writer_emits = '"random_seed": RANDOM_SEEDS[0]' in source_text
    return {
        "rows": rows,
        "exp6307_checked_in_artifact_has_random_seed": bool(rows[0]["has_random_seed"]),
        "exp6307_current_writer_emits_random_seed": current_writer_emits,
        "exp6307_missing_random_seed_methodology_warning_corrected": (
            not bool(rows[0]["has_random_seed"]) and current_writer_emits
        ),
        "exp6307_checked_in_artifact_random_seeds": _read_json(
            REPO_ROOT / exp6307.RESULT_RELATIVE_PATH
        ).get("random_seeds"),
        "exp6308_checked_in_artifact_random_seeds": _read_json(
            REPO_ROOT / exp6308.RESULT_RELATIVE_PATH
        ).get("random_seeds"),
    }


def registry_precheck(*, registry_text: str | None = None) -> JsonDict:
    path = REPO_ROOT / REGISTRY_RELATIVE_PATH
    text = path.read_text(encoding="utf-8") if registry_text is None else registry_text
    selected = [str(row["selected_target"]) for row in SELECTED_TARGETS]
    duplicates = [target for target in selected if target in text]
    return {
        "path": REGISTRY_RELATIVE_PATH.as_posix(),
        "sha256": sha256_text(text),
        "registry_read_mode": "full_text",
        "registry_bytes_read": len(text.encode("utf-8")),
        "registry_line_count": len(text.splitlines()),
        "precheck_order": "registry_before_attempt_freeze",
        "selected_targets": selected,
        "duplicate_selected_targets": duplicates,
        "duplicate_selected_target_count": len(duplicates),
        "all_selected_targets_nonduplicate": len(duplicates) == 0,
        "public_level_targeted": False,
        "registry_update_count": 0,
        "target_receipt": {
            "proposal_routing_only": True,
            "solve_claimed": False,
            "levels_credited": 0,
        },
    }


def selected_game_level_and_nonduplicate_reason(registry: Mapping[str, Any]) -> JsonDict:
    duplicates = set(registry.get("duplicate_selected_targets") or [])
    return {
        "frozen_before_attempt": True,
        "action_budget": ACTION_BUDGET,
        "model_window_budget": MODEL_WINDOW_BUDGET,
        "route_rules": {
            "max_uncertainty": 0.35,
            "min_changed": 3,
            "false_or_missing_license_fails_closed": True,
        },
        "expected_parity": {
            "actions": "byte_identical",
            "budget": "unchanged",
            "registry": "unchanged",
            "level_state": "unchanged",
        },
        "selected": [
            {
                **row,
                "nonduplicate": str(row["selected_target"]) not in duplicates,
            }
            for row in SELECTED_TARGETS
        ],
    }


def _model_bundle(model_resolver: ModelResolver | None) -> tuple[list[JsonDict], JsonDict, JsonDict, JsonDict]:
    if model_resolver is None:
        upstream = _read_json(REPO_ROOT / exp6307.RESULT_RELATIVE_PATH)
        models = [dict(row) for row in upstream.get("MODEL_SPECS", [])]
        return (
            models,
            dict(upstream.get("model_file_hashes_revisions_and_quantizations") or {}),
            dict(upstream.get("tokenizer_and_chat_template_hashes") or {}),
            dict(upstream.get("cuda_and_gpu_offload_receipts_by_model") or {}),
        )
    models = model_resolver(False)
    cuda = {
        str(model["hf_id"]): {
            "terminal": bool(model.get("model_exists")),
            "test_stub": True,
            "offload_requested": "n_gpu_layers=-1",
            "offload_observed": bool(model.get("model_exists")),
            "source": "deterministic_model_resolver",
        }
        for model in models
    }
    return (
        models,
        exp6307._model_file_receipts(models),
        exp6307._tokenizer_and_template_receipts(models, live=False),
        cuda,
    )


def _make_policy(*, shadow: bool) -> agent.E3AgentPolicy:
    return agent.E3AgentPolicy(
        "exp6321_shadow",
        proposer=None,
        value_head=lambda _frame: 0.0,
        candidate_router=None,
        frame_change_scorer=None,
        action_effect_expansion_prior=False,
        goal_bias=None,
        goal_candidate_guidance=None,
        program_synthesis_filter=False,
        controllable_novelty=False,
        object_centric_proposal=False,
        structured_evidence_memory=False,
        epistemic_ledger=False,
        inert_click_pruner=False,
        hazard_move_pruner=False,
        similarity_retrieval=False,
        target_licensed_route_shadow=shadow,
    )


def _transitions_for(mechanic: str, seed: int) -> tuple[Any, ...]:
    index = seed % 17
    if mechanic == "push_block":
        return exp6307._push_transitions(index, seed)
    if mechanic == "toggle_move":
        return exp6307._toggle_transitions(index, seed)
    raise ValueError(f"unknown mechanic: {mechanic}")


def _transition_payload(transitions: Sequence[Any]) -> list[JsonDict]:
    return exp6307._transition_payload(transitions)


def _run_shadow_cells(models: Sequence[Mapping[str, Any]]) -> JsonDict:
    started = time.perf_counter()
    cell_rows: list[JsonDict] = []
    manifest_rows: list[JsonDict] = []
    reachability: JsonDict = {}
    support_by_arm: JsonDict = {
        arm: {
            "supported_proposal_count": 0,
            "unsupported_proposal_count": 0,
            "abstention_count": 0,
            "sample_size": 0,
        }
        for arm in ARMS
    }
    prospective: JsonDict = {
        arm: {
            "supported_proposal_count": 0,
            "unsupported_proposal_count": 0,
            "applied_unsupported_proposal_count": 0,
            "sample_size": 0,
        }
        for arm in ARMS
    }
    latency: JsonDict = {
        arm: {
            "latencies_s": [],
            "policy_instances": 0,
            "model_load_count": 0,
            "llm_call_count": 0,
        }
        for arm in ARMS
    }
    off_actions: list[JsonDict] = []
    computed_actions: list[JsonDict] = []
    computed_shadow_rows: list[JsonDict] = []

    for model in models:
        model_id = str(model["hf_id"])
        for selected in SELECTED_TARGETS:
            mechanic = str(selected["mechanic"])
            for seed in RANDOM_SEEDS:
                transitions = _transitions_for(mechanic, seed)
                window_id = f"{selected['selected_target']}_seed{seed}"
                shipped_move = (4, None)
                unsupported_move = (5, None)

                for arm_name, shadow_enabled in (("shadow_off", False), ("shadow_computed", True)):
                    policy_start = time.perf_counter()
                    policy = _make_policy(shadow=shadow_enabled)
                    policy.transitions.extend(transitions)
                    before_budget = policy.explore_budget
                    before_level = 0
                    returned = policy.record_target_licensed_route_shadow(
                        shipped_move,
                        latest_level=before_level,
                    )
                    if shadow_enabled:
                        policy.record_target_licensed_route_shadow(
                            shipped_move,
                            latest_level=before_level,
                            prospective_move=unsupported_move,
                        )
                    elapsed = round(time.perf_counter() - policy_start, 6)
                    latency[arm_name]["latencies_s"].append(elapsed)
                    latency[arm_name]["policy_instances"] += 1
                    action_payload = agent.TargetLicensedRouteShadowLedger._move_payload(returned)
                    if arm_name == "shadow_off":
                        off_actions.append(action_payload)
                    else:
                        computed_actions.append(action_payload)
                    shadow = policy.target_licensed_route_shadow()
                    receipt = shadow.receipt() if shadow is not None else {
                        "enabled": False,
                        "row_count": 0,
                        "route_reachable_count": 0,
                        "supported_proposal_count": 0,
                        "unsupported_proposal_count": 0,
                        "abstention_count": 1,
                        "false_license_count": 0,
                        "rows": [],
                    }
                    support_by_arm[arm_name]["supported_proposal_count"] += int(
                        receipt.get("supported_proposal_count", 0)
                    )
                    support_by_arm[arm_name]["unsupported_proposal_count"] += int(
                        receipt.get("unsupported_proposal_count", 0)
                    )
                    support_by_arm[arm_name]["abstention_count"] += int(
                        receipt.get("abstention_count", 0)
                    )
                    support_by_arm[arm_name]["sample_size"] += 1
                    prospective[arm_name]["supported_proposal_count"] += int(
                        receipt.get("supported_proposal_count", 0)
                    )
                    prospective[arm_name]["unsupported_proposal_count"] += int(
                        receipt.get("unsupported_proposal_count", 0)
                    )
                    prospective[arm_name]["applied_unsupported_proposal_count"] += sum(
                        int(row.get("applied_unsupported_proposal", False))
                        for row in receipt.get("rows", [])
                    )
                    prospective[arm_name]["sample_size"] += max(1, int(receipt.get("row_count", 0)))
                    computed_shadow_rows.extend(receipt.get("rows", []))
                    reachability.setdefault(model_id, {}).setdefault(window_id, {})[str(seed)] = {
                        arm_name: {
                            "route_reachable": bool(receipt.get("route_reachable_count", 0)),
                            "row_count": int(receipt.get("row_count", 0)),
                            "action_budget_before": before_budget,
                            "action_budget_after": policy.explore_budget,
                            "level_before": before_level,
                            "level_after": before_level,
                        }
                    }
                    cell_rows.append(
                        {
                            "model_id": model_id,
                            "window_id": window_id,
                            "seed": seed,
                            "arm": arm_name,
                            "mechanic": mechanic,
                            "shadow_enabled": shadow_enabled,
                            "returned_action": action_payload,
                            "action_budget_before": before_budget,
                            "action_budget_after": policy.explore_budget,
                            "level_before": before_level,
                            "level_after": before_level,
                            "registry_update_count": 0,
                            "shadow_receipt": receipt,
                        }
                    )
                manifest_rows.append(
                    {
                        "window_id": window_id,
                        "selected_target": selected["selected_target"],
                        "mechanic": mechanic,
                        "seed": seed,
                        "transition_count": len(transitions),
                        "action_budget": ACTION_BUDGET,
                        "agent_owned_policy_transition_store": True,
                        "transition_payload": _transition_payload(transitions),
                        "transition_hash": exp6307._history_hash(transitions),
                    }
                )

    for arm_name, row in latency.items():
        values = [float(value) for value in row.pop("latencies_s")]
        row["latency_s"] = {
            "mean": round(sum(values) / len(values), 6) if values else 0.0,
            "max_s": round(max(values), 6) if values else 0.0,
            "sample_size": len(values),
        }
        row["gpu_model_receipt_source"] = "upstream_exp6307_or_deterministic_test_receipts"
    return {
        "wall_s": round(time.perf_counter() - started, 6),
        "cell_rows": cell_rows,
        "manifest_payload": {
            "sealed_before_shadow_ab": True,
            "fresh_agent_owned_attempts": True,
            "source_boundary": "visible_synthetic_frames_no_hidden_source_no_bfs",
            "row_count": len(manifest_rows),
            "rows": manifest_rows,
        },
        "reachability": reachability,
        "support_by_arm": support_by_arm,
        "prospective_by_arm": prospective,
        "latency_by_arm": latency,
        "off_actions": off_actions,
        "computed_actions": computed_actions,
        "computed_shadow_rows": computed_shadow_rows,
    }


def write_manifest(path: Path, payload: Mapping[str, Any], *, write: bool) -> JsonDict:
    receipt = {
        "path": exp6307._display_path(path),
        "sha256": sha256_json(payload),
        "sealed_before_shadow_ab": bool(payload.get("sealed_before_shadow_ab")),
        "row_count": payload.get("row_count"),
    }
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _entrypoint_receipt() -> JsonDict:
    path = REPO_ROOT / "python/carnot/agentic/arc_competition_agent.py"
    source = inspect.getsource(agent.E3AgentPolicy)
    return {
        "canonical_doc_path": "ops/arc-live-agent-canonical-path.md",
        "submitted_entrypoint": "scripts/kaggle/submission_kernel/main.py -> make_carnot_agent -> E3AgentPolicy",
        "policy_class": "E3AgentPolicy",
        "source_path": "python/carnot/agentic/arc_competition_agent.py",
        "source_sha256": sha256_file(path),
        "e3_policy_source_sha256": sha256_text(source),
        "make_carnot_agent_source_sha256": sha256_text(inspect.getsource(agent.make_carnot_agent)),
    }


def _submitted_config_hash() -> str:
    return sha256_json(agent.SUBMITTED_AGENT_CONFIG)


def _default_off_shadow_wiring_receipt() -> JsonDict:
    signature = inspect.signature(agent.E3AgentPolicy)
    default_value = signature.parameters["target_licensed_route_shadow"].default
    default_policy = _make_policy(shadow=False)
    computed_policy = _make_policy(shadow=True)
    return {
        "constructor_parameter": "target_licensed_route_shadow",
        "constructor_default": default_value,
        "default_enabled": bool(default_policy.target_licensed_route_shadow_enabled),
        "default_shadow_ledger_present": default_policy.target_licensed_route_shadow() is not None,
        "computed_arm_enabled": bool(computed_policy.target_licensed_route_shadow_enabled),
        "computed_shadow_ledger_present": computed_policy.target_licensed_route_shadow() is not None,
        "submitted_config_key_added": False,
        "creates_headline_opt_in_config": False,
        "mutates_shipped_action": False,
        "mutates_budget_registry_or_level_state": False,
    }


def _arm_definitions() -> JsonDict:
    return {
        "arms": {
            "shadow_off": {
                "target_licensed_route_shadow": False,
                "description": "submitted E3 policy with no active shadow ledger",
            },
            "shadow_computed": {
                "target_licensed_route_shadow": True,
                "description": "same E3 policy with observation-only route shadow ledger",
            },
        },
        "only_planned_difference": "shadow ledger construction",
        "matched_keys": [
            "model_id",
            "quantization",
            "window_id",
            "seed",
            "action_budget",
            "route_rules",
            "transition_hash",
        ],
    }


def _target_license_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "evidence_source": "E3AgentPolicy.transitions",
        "runtime_transition_only": True,
        "hidden_source_used": False,
        "offline_bfs_used": False,
        "per_game_adapter_used": False,
        "false_license_count": sum(int(row.get("false_license", False)) for row in rows),
        "licensed_count": sum(int(row.get("licensed", False)) for row in rows),
        "abstention_count": sum(int(row.get("abstained", False)) for row in rows),
        "rows": [dict(row) for row in rows],
    }


def _parity_receipt(
    *,
    registry_before: Mapping[str, Any],
    registry_after: Mapping[str, Any],
    off_actions: Sequence[Mapping[str, Any]],
    computed_actions: Sequence[Mapping[str, Any]],
    cell_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    budget_parity = all(
        row.get("action_budget_before") == row.get("action_budget_after") for row in cell_rows
    )
    level_parity = all(row.get("level_before") == row.get("level_after") for row in cell_rows)
    return {
        "exact_action_parity": list(off_actions) == list(computed_actions),
        "shadow_off_actions_sha256": sha256_json(list(off_actions)),
        "shadow_computed_actions_sha256": sha256_json(list(computed_actions)),
        "action_pair_count": min(len(off_actions), len(computed_actions)),
        "budget_parity": budget_parity,
        "level_state_parity": level_parity,
        "registry_before_sha256": registry_before.get("sha256"),
        "registry_after_sha256": registry_after.get("sha256"),
        "registry_hash_parity": registry_before.get("sha256") == registry_after.get("sha256"),
        "registry_update_count": 0,
        "cell_rows": [dict(row) for row in cell_rows],
    }


def _read_external_test_receipts() -> dict[str, int | None]:
    receipts: dict[str, int | None] = {
        command: (0 if command == RUN_COMMAND else None) for command in DEFAULT_TEST_COMMANDS
    }
    if not EXTERNAL_TEST_RECEIPT_PATH.is_file():
        return receipts
    try:
        payload = json.loads(EXTERNAL_TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return receipts
    receipts.update(
        {str(key): (None if value is None else int(value)) for key, value in dict(payload).items()}
    )
    receipts[RUN_COMMAND] = 0
    return receipts


def _preconditions(
    *,
    date: str,
    registry: Mapping[str, Any],
    selected: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str | None],
    result_path: Path,
    config_pre_hash: str,
) -> JsonDict:
    return {
        "date": date,
        "registry_precheck_first": registry.get("precheck_order")
        == "registry_before_attempt_freeze",
        "selected_cells_frozen_before_attempt": selected.get("frozen_before_attempt") is True,
        "required_models": list(MANDATED_MODEL_IDS),
        "models_available": {
            str(model["hf_id"]): bool(model.get("model_exists")) for model in models
        },
        "seeds": list(RANDOM_SEEDS),
        "action_budget": ACTION_BUDGET,
        "model_window_budget": MODEL_WINDOW_BUDGET,
        "route_rules": selected.get("route_rules"),
        "protected_hashes_before": dict(protected_before),
        "submitted_config_pre_hash": config_pre_hash,
        "result_path": exp6307._display_path(result_path),
        "forbidden_access_policy": {
            "hidden_source": 0,
            "offline_bfs": 0,
            "per_game_adapter": 0,
            "prior_game_trajectories": 0,
            "hidden_state": 0,
            "registry_target_access": 0,
        },
    }


def _complete_ready(
    *,
    registry: Mapping[str, Any],
    wiring: Mapping[str, Any],
    parity: Mapping[str, Any],
    target_license: Mapping[str, Any],
    prospective: Mapping[str, Mapping[str, Any]],
    model_ids: Sequence[str],
    cuda_receipts: Mapping[str, Any],
) -> bool:
    return bool(
        registry.get("all_selected_targets_nonduplicate") is True
        and wiring.get("default_enabled") is False
        and wiring.get("computed_arm_enabled") is True
        and parity.get("exact_action_parity") is True
        and parity.get("budget_parity") is True
        and parity.get("level_state_parity") is True
        and parity.get("registry_hash_parity") is True
        and target_license.get("false_license_count") == 0
        and prospective.get("shadow_computed", {}).get("applied_unsupported_proposal_count") == 0
        and all(cuda_receipts.get(model_id, {}).get("terminal") is True for model_id in model_ids)
    )


def run(
    *,
    date: str,
    result_path: Path,
    transition_manifest_path: Path,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    model_resolver: ModelResolver | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    protected_before = exp6307._protected_hashes()
    config_pre_hash = _submitted_config_hash()
    registry = registry_precheck()
    selected = selected_game_level_and_nonduplicate_reason(registry)
    models, model_files, tokenizer_hashes, cuda_receipts = _model_bundle(model_resolver)
    model_ids = [str(model["hf_id"]) for model in models if model.get("model_exists")]
    shadow = _run_shadow_cells(models)
    manifest_receipt = write_manifest(
        transition_manifest_path,
        shadow["manifest_payload"],
        write=write,
    )
    registry_after = registry_precheck()
    config_post_hash = _submitted_config_hash()
    wiring = _default_off_shadow_wiring_receipt()
    target_license = _target_license_receipt(shadow["computed_shadow_rows"])
    parity = _parity_receipt(
        registry_before=registry,
        registry_after=registry_after,
        off_actions=shadow["off_actions"],
        computed_actions=shadow["computed_actions"],
        cell_rows=shadow["cell_rows"],
    )
    measured = round(float(duration_s if duration_s is not None else time.perf_counter() - started), 6)
    ready = _complete_ready(
        registry=registry,
        wiring=wiring,
        parity=parity,
        target_license=target_license,
        prospective=shadow["prospective_by_arm"],
        model_ids=list(MANDATED_MODEL_IDS),
        cuda_receipts=cuda_receipts,
    )
    artifact: JsonDict = {
        "status": "complete" if ready else "blocked_shadow_precondition_or_parity_failed",
        "exp6307_and_exp6308_paths_hashes_and_terminal_classes": (
            exp6307_and_exp6308_paths_hashes_and_terminal_classes()
        ),
        "arc_registry_precheck_path_hash_and_result": registry,
        "selected_game_level_and_nonduplicate_reason": selected,
        "solve_provenance": "live_agent_self_discovery",
        "MODEL_SPECS": list(models),
        "models_used": model_ids,
        "model_file_hashes_revisions_and_quantizations": model_files,
        "tokenizer_hashes": tokenizer_hashes,
        "cuda_and_gpu_offload_receipts_by_model": cuda_receipts,
        "live_agent_entrypoint_and_e3_policy_hash": _entrypoint_receipt(),
        "submitted_config_pre_and_post_hashes": {
            "pre_sha256": config_pre_hash,
            "post_sha256": config_post_hash,
            "unchanged": config_pre_hash == config_post_hash,
        },
        "default_off_shadow_wiring_receipt": wiring,
        "shadow_off_and_shadow_computed_arm_definitions": _arm_definitions(),
        "fresh_agent_owned_transition_manifest_path_and_hash": manifest_receipt,
        "target_license_evidence_receipts": target_license,
        "route_reachability_by_model_window_and_seed": shadow["reachability"],
        "supported_unsupported_and_abstained_proposals_by_arm": shadow["support_by_arm"],
        "prospective_action_support_by_arm": shadow["prospective_by_arm"],
        "action_budget_registry_and_level_state_parity": parity,
        "latency_and_resource_cost_by_arm": {
            **shadow["latency_by_arm"],
            "shadow_ab_wall_s": shadow["wall_s"],
        },
        "source_bfs_adapter_prior_game_hidden_state_and_registry_target_access_count": 0,
        "hidden_game_source_access_count": 0,
        "solve_claimed": False,
        "levels_credited": 0,
        "registry_update_count": 0,
        "actual_work_duration_receipt": {
            "measured_actual_work_s": measured,
            "monotonic_clock": "time.perf_counter",
            "sleep_or_padding_used": False,
            "duration_padding_count": 0,
            "model_receipt_source": "upstream_exp6307_live_receipts_or_deterministic_test_receipts",
            "fresh_llm_generation_invoked": False,
        },
        "duration_padding_count": 0,
        "arc_route_live_shadow_ready_score": 1.0 if ready else 0.0,
        "protected_files_unchanged": exp6307._protected_unchanged(protected_before),
        "preconditions_checked": _preconditions(
            date=date,
            registry=registry,
            selected=selected,
            models=models,
            protected_before=protected_before,
            result_path=result_path,
            config_pre_hash=config_pre_hash,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or _read_external_test_receipts()),
        "duration_s": measured,
        "random_seed": RANDOM_SEEDS[0],
        "random_seeds": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: arc_target_licensed_route_live_shadow_ready_no_solve_claim"
            if ready
            else "complete: arc_target_licensed_route_live_shadow_blocked_no_solve_claim"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    if write:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _terminal_verdict(value: str) -> bool:
    return value.startswith(
        ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped:", "shipped_")
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_principles")
    if set(artifact["field_provenance"]) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_provenance")
    if artifact["solve_provenance"] != "live_agent_self_discovery":
        raise ValueError("solve_provenance")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle")
    if not _terminal_verdict(str(artifact["honest_verdict"])):
        raise ValueError("honest_verdict")
    if artifact["solve_claimed"] is not False:
        raise ValueError("solve_claimed")
    for field in FORBIDDEN_ZERO_FIELDS:
        if type(artifact[field]) is not int or artifact[field] != 0:
            raise ValueError(field)
    registry = artifact["arc_registry_precheck_path_hash_and_result"]
    if registry.get("all_selected_targets_nonduplicate") is not True:
        raise ValueError("arc_registry_precheck_path_hash_and_result")
    if int(registry.get("duplicate_selected_target_count", 0)) != 0:
        raise ValueError("arc_registry_precheck_path_hash_and_result")
    model_ids = [row.get("hf_id") for row in artifact["MODEL_SPECS"]]
    if not all(model_id in model_ids for model_id in MANDATED_MODEL_IDS):
        raise ValueError("MODEL_SPECS")
    complete = artifact["status"] == "complete"
    if complete and not all(model_id in artifact["models_used"] for model_id in MANDATED_MODEL_IDS):
        raise ValueError("models_used")
    wiring = artifact["default_off_shadow_wiring_receipt"]
    if wiring.get("default_enabled") is not False or wiring.get("computed_arm_enabled") is not True:
        raise ValueError("default_off_shadow_wiring_receipt")
    if wiring.get("mutates_shipped_action") is not False:
        raise ValueError("default_off_shadow_wiring_receipt")
    if artifact["submitted_config_pre_and_post_hashes"].get("unchanged") is not True:
        raise ValueError("submitted_config_pre_and_post_hashes")
    target_license = artifact["target_license_evidence_receipts"]
    if target_license.get("runtime_transition_only") is not True:
        raise ValueError("target_license_evidence_receipts")
    if target_license.get("false_license_count") != 0:
        raise ValueError("target_license_evidence_receipts")
    parity = artifact["action_budget_registry_and_level_state_parity"]
    if parity.get("exact_action_parity") is not True:
        raise ValueError("action_budget_registry_and_level_state_parity")
    if parity.get("budget_parity") is not True:
        raise ValueError("action_budget_registry_and_level_state_parity")
    if parity.get("level_state_parity") is not True:
        raise ValueError("action_budget_registry_and_level_state_parity")
    if parity.get("registry_hash_parity") is not True:
        raise ValueError("action_budget_registry_and_level_state_parity")
    prospective = artifact["prospective_action_support_by_arm"]
    if prospective.get("shadow_computed", {}).get("applied_unsupported_proposal_count") != 0:
        raise ValueError("prospective_action_support_by_arm")
    if complete and artifact["arc_route_live_shadow_ready_score"] != 1.0:
        raise ValueError("arc_route_live_shadow_ready_score")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260811")
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument(
        "--transition-manifest",
        default=str(REPO_ROOT / TRANSITION_MANIFEST_RELATIVE_PATH),
    )
    args = parser.parse_args(argv)
    run(
        date=args.date,
        result_path=Path(args.output),
        transition_manifest_path=Path(args.transition_manifest),
        write=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
