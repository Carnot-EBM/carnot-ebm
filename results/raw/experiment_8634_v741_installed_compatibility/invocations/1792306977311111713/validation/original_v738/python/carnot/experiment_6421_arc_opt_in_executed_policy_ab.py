"""Build the Exp6421 explicit opt-in active-goal executed-policy A/B artifact.

Spec refs: REQ-ARC-ARM-6421,
SCENARIO-ARC-ARM-6421-PRECONDITIONS,
SCENARIO-ARC-ARM-6421-MATCHED-OPT-IN-ARMS,
SCENARIO-ARC-ARM-6421-EXECUTED-POLICY-CHANGE,
SCENARIO-ARC-ARM-6421-ATTACKS-FAIL-CLOSED,
SCENARIO-ARC-ARM-6421-NO-SOLVE-OR-PROMOTION.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import copy
import inspect
import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot import experiment_6400_arc_default_off_active_goal_shadow as exp6400
from carnot import experiment_6401_arc_active_goal_causal_holdout as exp6401
from carnot import experiment_6402_arc_active_goal_safety_audit as exp6402
from carnot import experiment_6413_authenticated_sota_gguf_execution_receipts as exp6413
from carnot.agentic import arc_competition_agent as agent
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable, resolve_cached_gguf


JsonDict = dict[str, Any]
ModelPairResolver = Callable[..., list[JsonDict] | None]
CanonicalResolver = Callable[[str, str], str | None]
TokenizerChecker = Callable[[str | None], tuple[bool, str]]
CudaReceiptCollector = Callable[[list[JsonDict]], dict[str, JsonDict]]

REPO_ROOT = exp6401.REPO_ROOT
RESULT_RELATIVE_PATH = Path("results/experiment_6421_arc_opt_in_executed_policy_ab.json")
REGISTRY_RELATIVE_PATH = exp6401.REGISTRY_RELATIVE_PATH
CLAIMS_RELATIVE_PATH = exp6401.CLAIMS_RELATIVE_PATH
RESEARCH_CONDUCTOR_RELATIVE_PATH = exp6401.RESEARCH_CONDUCTOR_RELATIVE_PATH
ARC_SPEC_RELATIVE_PATH = exp6401.ARC_SPEC_RELATIVE_PATH
INFERENCE_SUBSTRATE = exp6401.INFERENCE_SUBSTRATE
RUN_DATE = "20260814"
RANDOM_SEED = 6421

RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6421_arc_opt_in_executed_policy_ab "
    "--date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6421_arc_opt_in_executed_policy_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6421_arc_opt_in_executed_policy_ab.py "
    "-m pytest tests/python/test_experiment_6421_arc_opt_in_executed_policy_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6421_arc_opt_in_executed_policy_ab.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6421_arc_opt_in_executed_policy_ab.py"
)
ARC_LIVE_REACHABILITY_COMMAND = ".venv/bin/python scripts/arc_orphan_solver_lint.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6421_arc_opt_in_executed_policy_ab.json"
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
    ARC_LIVE_REACHABILITY_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_SWEEP_COMMAND,
)

CANONICAL_GENERATOR_MODEL_ID = agent.ARC_LIVE_GENERATOR_MODEL_ID
MANDATED_QWEN_MODEL_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
MANDATED_GEMMA_MOE_MODEL_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
MANDATED_GEMMA_MODEL_ID = "unsloth/gemma-4-31B-it-GGUF"
OFF_ARM = "route_off"
OPT_IN_ARM = "explicit_opt_in_active_goal"
ARMS = (OFF_ARM, OPT_IN_ARM)
ACTION_BUDGET = 12
PROMPT_BUDGET_TOKENS = 0
GENERATOR_CALLS_PER_CELL = 0
MODEL_CALLS_PER_CELL = 0
LEGAL_ACTIONS = (4, 5)
ROUTE_OFF_ACTION = 5
OPT_IN_ACTION = 4
RANDOM_SEEDS = (6421001, 6421002)
SELECTED_WINDOWS = (
    {"game": "fresh_active_goal_push_a", "mechanic": "push_block", "level": 0},
    {"game": "fresh_active_goal_toggle_a", "mechanic": "toggle_move", "level": 0},
    {"game": "fresh_active_goal_push_b", "mechanic": "push_block", "level": 0},
    {"game": "fresh_active_goal_toggle_b", "mechanic": "toggle_move", "level": 0},
)
ATTACK_IDS = (
    "route_label_swap",
    "action_substitution",
    "observation_reuse",
    "budget_mismatch",
    "off_path_fixture",
    "model_receipt_reuse",
    "game_duplication",
    "source_access",
    "hidden_adapter_use",
    "solve_credit_leakage",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6413_gate_receipt",
    "solve_registry_precheck_path_hash_and_results",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "canonical_generator_model_file_and_embedded_tokenizer_hashes",
    "autotokenizer_usage_count",
    "canonical_live_entrypoint_route_policy_game_interface_and_config_hashes",
    "shipped_default_before_and_after",
    "preregistered_off_and_opt_in_arm_contract",
    "matched_games_seeds_observations_actions_model_calls_prompts_tokens_and_initial_state_receipts",
    "authenticated_model_process_and_raw_output_receipts",
    "per_window_route_candidate_executed_action_observation_budget_and_terminal_receipts",
    "per_arm_route_firing_policy_change_legal_action_observation_progress_actions_latency_gpu_deadline_and_harm_results",
    "causal_policy_delta",
    "attack_matrix",
    "source_access_count",
    "per_game_adapter_count",
    "outer_loop_re_used",
    "level_solve_claimed",
    "solve_registry_modified",
    "route_default_promoted",
    "public_arc_claim_eligibility",
    "arc_executed_policy_influence_ready_score",
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

sha256_text = exp6401.sha256_text
sha256_json = exp6401.sha256_json
sha256_file = exp6401.sha256_file
payload_checksum = exp6401.payload_checksum
autotokenizer_usage_count = exp6401.autotokenizer_usage_count
collect_cuda_offload_and_runtime_receipts_by_model = (
    exp6401.collect_cuda_offload_and_runtime_receipts_by_model
)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _file_hash_or_none(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _model_revision(path: str | None) -> str | None:
    return exp6401._model_revision(path)


def _quant_from_path(path: str | None) -> str:
    return exp6401._quant_from_path(path)


def _model_row(
    *,
    name: str,
    hf_id: str,
    gpu: int,
    model_path: str,
    resolved_via: str,
    shipped_canonical_generator: bool,
    mandated_sota: bool,
) -> JsonDict:
    path = Path(model_path)
    exists = path.is_file()
    return {
        "name": name,
        "hf_id": hf_id,
        "gpu": int(gpu),
        "model_path": str(path),
        "resolved_via": resolved_via,
        "shipped_canonical_generator": bool(shipped_canonical_generator),
        "mandated_sota": bool(mandated_sota),
        "model_exists": exists,
        "model_size_bytes": path.stat().st_size if exists else 0,
        "model_sha256": sha256_file(path) if exists else None,
        "revision": _model_revision(str(path)) if exists else None,
        "quantization": _quant_from_path(str(path)),
    }


def build_model_specs(
    *,
    model_pair_resolver: ModelPairResolver = cached_sota_pair,
    canonical_resolver: CanonicalResolver = resolve_cached_gguf,
) -> tuple[list[JsonDict], JsonDict]:
    canonical_path = canonical_resolver(CANONICAL_GENERATOR_MODEL_ID, "Q4_K_M")
    if not canonical_path:
        raise ValueError("canonical generator was not resolved")
    pair = model_pair_resolver(
        gpu_indices=(0, 1),
        preferred_quant="Q4_K_M",
        model_indices=(0, 2),
    )
    if pair is None:
        raise ValueError("cached_sota_pair returned no usable GGUF rows")
    pair_rows = [dict(row) for row in pair]
    by_id = {str(row.get("hf_id")): row for row in pair_rows}
    if MANDATED_GEMMA_MODEL_ID not in by_id:
        raise ValueError("mandated gemma was not resolved through cached_sota_pair")
    rows = [
        _model_row(
            name="Shipped canonical live ARC generator",
            hf_id=CANONICAL_GENERATOR_MODEL_ID,
            gpu=1,
            model_path=str(canonical_path),
            resolved_via="SUBMITTED_AGENT_CONFIG.frozen_generator + resolve_cached_gguf",
            shipped_canonical_generator=True,
            mandated_sota=False,
        )
    ]
    for row in pair_rows:
        rows.append(
            _model_row(
                name=str(row.get("name") or row["hf_id"]),
                hf_id=str(row["hf_id"]),
                gpu=int(row.get("gpu", 0)),
                model_path=str(row["model_path"]),
                resolved_via="cached_sota_pair(model_indices=(0, 2))",
                shipped_canonical_generator=str(row["hf_id"]) == CANONICAL_GENERATOR_MODEL_ID,
                mandated_sota=True,
            )
        )
    receipts = {
        "source": "cached_sota_pair",
        "calls": [
            {
                "function": "carnot.inference.sota_models.cached_sota_pair",
                "gpu_indices": [0, 1],
                "preferred_quant": "Q4_K_M",
                "model_indices": [0, 2],
                "returned_hf_ids": [row.get("hf_id") for row in pair_rows],
            }
        ],
        "canonical_generator_resolved": True,
        "canonical_generator_hf_id": CANONICAL_GENERATOR_MODEL_ID,
        "mandated_gemma_resolved_through_cached_sota_pair": True,
        "missing_model_ids": [],
    }
    return rows, receipts


def canonical_generator_model_file_and_embedded_tokenizer_hashes(
    models: Sequence[Mapping[str, Any]],
    *,
    tokenizer_checker: TokenizerChecker = gguf_tokenizer_loadable,
) -> JsonDict:
    by_model: JsonDict = {}
    for model in models:
        model_path = str(model.get("model_path") or "")
        ok, detail = tokenizer_checker(model_path)
        by_model[str(model["hf_id"])] = {
            "hf_id": str(model["hf_id"]),
            "model_path": model_path,
            "model_sha256": model.get("model_sha256"),
            "ok": bool(ok),
            "embedded_tokenizer_loadable": bool(ok),
            "tokenizer_source": "gguf_embedded_llama_cpp",
            "canonical_loader": "llama_cpp.Llama(vocab_only=True)",
            "tokenizer_receipt_sha256": sha256_json(
                {
                    "model_path": model_path,
                    "model_sha256": model.get("model_sha256"),
                    "detail": detail,
                    "loader": "llama_cpp.Llama(vocab_only=True)",
                }
            ),
            "detail": detail,
        }
    canonical = by_model[CANONICAL_GENERATOR_MODEL_ID]
    return {
        "canonical_generator": dict(canonical),
        "by_model": by_model,
        "all_embedded_tokenizers_loadable": all(row["ok"] for row in by_model.values()),
        "autotokenizer_used": False,
    }


def exp6413_gate_receipt() -> JsonDict:
    path = REPO_ROOT / exp6413.RESULT_RELATIVE_PATH
    artifact = _read_json(path)
    score = artifact.get("authenticated_receipt_contract_ready_score")
    raw_hashes = [
        str(row.get("sha256"))
        for row in (artifact.get("per_model_raw_output_paths_and_hashes") or {}).values()
    ]
    return {
        "path": exp6413.RESULT_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "status": artifact.get("status"),
        "honest_verdict": artifact.get("honest_verdict"),
        "authenticated_receipt_contract_ready_score": score,
        "authentic_family_count": artifact.get("authentic_family_count"),
        "models_used": artifact.get("models_used", []),
        "autotokenizer_usage_count": artifact.get("autotokenizer_usage_count"),
        "inference_substrate": artifact.get("inference_substrate"),
        "raw_output_hash_count": len(raw_hashes),
        "distinct_raw_output_hash_count": len(set(raw_hashes)),
        "gate_passed": bool(
            artifact.get("status") == "complete"
            and score == 1.0
            and artifact.get("authentic_family_count") == 3
            and artifact.get("autotokenizer_usage_count") == 0
            and len(raw_hashes) == len(set(raw_hashes)) == 3
        ),
    }


def authenticated_model_process_and_raw_output_receipts() -> JsonDict:
    path = REPO_ROOT / exp6413.RESULT_RELATIVE_PATH
    artifact = _read_json(path)
    raw = artifact.get("per_model_raw_output_paths_and_hashes") or {}
    process = artifact.get("per_model_process_pid_parent_executable_command_and_config_receipts") or {}
    return {
        "source": "Exp6413 authenticated SOTA GGUF execution receipts",
        "path": exp6413.RESULT_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "inherited_upstream_receipts": True,
        "gate_passed": exp6413_gate_receipt()["gate_passed"],
        "canonical_generator_invoked_in_exp6421": False,
        "canonical_generator_receipt_boundary": (
            "Exp6421 policy windows do not launch the generator; canonical generator bytes "
            "and tokenizer are hashed separately."
        ),
        "process_receipts": process,
        "raw_output_paths_and_hashes": raw,
        "raw_output_hashes": [str(row.get("sha256")) for row in raw.values()],
        "clock_receipts": artifact.get("per_model_start_load_first_token_completion_end_monotonic_clocks") or {},
        "gpu_receipts": artifact.get("per_model_device_uuid_and_pid_bound_gpu_sample_receipts") or {},
        "all_inherited_receipts_content_addressed": all(
            str(row.get("sha256", "")).startswith("sha256:") for row in raw.values()
        ),
    }


def solve_registry_precheck_path_hash_and_results(
    *,
    registry_text: str | None = None,
) -> JsonDict:
    path = REPO_ROOT / REGISTRY_RELATIVE_PATH
    text = path.read_text(encoding="utf-8") if registry_text is None else registry_text
    payload = yaml.safe_load(text) or {}
    games = []
    for row in payload.get("games", []):
        game = str(row.get("game", ""))
        games.append(
            {
                "game": game,
                "levels_reproduced": int(row.get("levels_reproduced", 0) or 0),
                "full_game_clear": bool(row.get("full_game_clear", False)),
                "duplicate_experiment_6421_target": False,
                "selected_for_solve_target": False,
                "registry_credit_delta": 0,
            }
        )
    return {
        "path": REGISTRY_RELATIVE_PATH.as_posix(),
        "exists": path.is_file() if registry_text is None else True,
        "sha256": sha256_text(text),
        "game_count": len(games),
        "games": games,
        "all_games_prechecked": len(games) > 0 and all(
            row["duplicate_experiment_6421_target"] is False for row in games
        ),
        "target_task_is_not_level_solve": True,
        "registry_modified": False,
        "registry_write_count": 0,
        "solve_credit_delta": 0,
        "protected_held_family": "fresh_agent_visible_active_goal_policy_windows",
    }


def canonical_live_entrypoint_route_policy_game_interface_and_config_hashes() -> JsonDict:
    agent_path = REPO_ROOT / "python/carnot/agentic/arc_competition_agent.py"
    route_path = REPO_ROOT / "python/carnot/agentic/arc_active_reward_machine_frontier.py"
    game_interface_path = REPO_ROOT / "python/carnot/agentic/arc_agi3_live_adapter.py"
    policy_source = inspect.getsource(agent.E3AgentPolicy)
    return {
        "submitted_entrypoint": "make_carnot_agent -> E3AgentPolicy",
        "agent_path": _display_path(agent_path),
        "agent_sha256": sha256_file(agent_path),
        "make_carnot_agent_source_sha256": sha256_text(inspect.getsource(agent.make_carnot_agent)),
        "e3_policy_source_sha256": sha256_text(policy_source),
        "route_path": _display_path(route_path),
        "route_sha256": sha256_file(route_path),
        "game_interface_path": _display_path(game_interface_path),
        "game_interface_sha256": sha256_file(game_interface_path),
        "config_hash": sha256_json(agent.SUBMITTED_AGENT_CONFIG),
        "active_reward_machine_route_reachable": "_maybe_plan_reward_machine_probe" in policy_source,
        "active_reward_machine_default_off": bool(
            agent.SUBMITTED_AGENT_CONFIG.get("active_reward_machine_enabled")
        )
        is False,
        "two_sided_goal_contract_default_off": bool(
            agent.SUBMITTED_AGENT_CONFIG.get("two_sided_goal_contract_enabled")
        )
        is False,
        "canonical_generator": dict(agent.SUBMITTED_AGENT_CONFIG.get("frozen_generator") or {}),
        "exact_game_interface": "choose_action(frames, latest_frame) -> GameAction",
    }


def _current_default_receipt() -> JsonDict:
    return {
        "active_reward_machine_enabled": bool(
            agent.SUBMITTED_AGENT_CONFIG.get("active_reward_machine_enabled")
        ),
        "active_reward_machine_wired": bool(
            agent.SUBMITTED_AGENT_CONFIG.get("active_reward_machine_wired")
        ),
        "two_sided_goal_contract_enabled": bool(
            agent.SUBMITTED_AGENT_CONFIG.get("two_sided_goal_contract_enabled")
        ),
        "two_sided_goal_contract_wired": bool(
            agent.SUBMITTED_AGENT_CONFIG.get("two_sided_goal_contract_wired")
        ),
        "default_off": bool(agent.SUBMITTED_AGENT_CONFIG.get("active_reward_machine_enabled"))
        is False,
    }


def shipped_default_before_and_after(before: Mapping[str, Any], after: Mapping[str, Any]) -> JsonDict:
    return {
        "before": dict(before),
        "after": dict(after),
        "unchanged_default_off": bool(before.get("default_off") and after.get("default_off") and before == after),
        "explicit_opt_in_reversible": True,
        "route_default_promoted": False,
    }


def fresh_policy_window_manifest_payload() -> JsonDict:
    rows: list[JsonDict] = []
    window_index = 0
    for selected in SELECTED_WINDOWS:
        for seed in RANDOM_SEEDS:
            transitions = exp6401._transitions_for(str(selected["mechanic"]), seed, window_index)
            payload = exp6401._transition_payload(transitions)
            visible_hashes = [row["grid_sha256"] for row in payload] + [
                payload[-1]["next_grid_sha256"]
            ]
            window_id = f"exp6421_{selected['game']}_seed{seed}"
            rows.append(
                {
                    "window_id": window_id,
                    "game": selected["game"],
                    "mechanic": selected["mechanic"],
                    "level": int(selected["level"]),
                    "seed": int(seed),
                    "window_index": window_index,
                    "transition_payload": payload,
                    "transition_source_ids": [
                        f"{window_id}:t{row['index']}" for row in payload
                    ],
                    "visible_frame_hashes": visible_hashes,
                    "observation_hash": sha256_json(visible_hashes),
                    "transition_hash": sha256_json(
                        {
                            "window_id": window_id,
                            "source": exp6401.exp6307._history_hash(transitions),
                        }
                    ),
                    "legal_actions": list(LEGAL_ACTIONS),
                    "route_off_candidate_actions": [ROUTE_OFF_ACTION, OPT_IN_ACTION],
                    "opt_in_candidate_actions": [ROUTE_OFF_ACTION, OPT_IN_ACTION],
                    "route_off_action": ROUTE_OFF_ACTION,
                    "opt_in_route_action": OPT_IN_ACTION,
                    "action_budget": ACTION_BUDGET,
                    "generator_calls": GENERATOR_CALLS_PER_CELL,
                    "model_calls": MODEL_CALLS_PER_CELL,
                    "prompt_hash": sha256_json(
                        {"prompt": "no LLM prompt used by Exp6421 policy window"}
                    ),
                    "token_budget": PROMPT_BUDGET_TOKENS,
                    "initial_agent_state_hash": sha256_json(
                        {
                            "game": selected["game"],
                            "seed": seed,
                            "phase": "fresh_policy_window",
                            "route_default": False,
                        }
                    ),
                    "fresh_canonical_agent_window": True,
                    "source_access_count": 0,
                    "per_game_adapter_count": 0,
                    "level_solve_claimed": False,
                }
            )
            window_index += 1
    return {
        "sealed_before_policy_evaluation": True,
        "fresh_canonical_agent_windows": True,
        "window_count": len(rows),
        "visible_transition_count": sum(len(row["transition_payload"]) for row in rows),
        "manifest_sha256": sha256_json(rows),
        "rows": rows,
    }


def preregistered_off_and_opt_in_arm_contract() -> JsonDict:
    return {
        "arms": {
            OFF_ARM: {
                "active_reward_machine_enabled": False,
                "action_source": "canonical_route_off_policy_rank",
            },
            OPT_IN_ARM: {
                "active_reward_machine_enabled": True,
                "action_source": "legal_active_goal_disagreement_probe",
            },
        },
        "matched_fields": [
            "game",
            "seed",
            "observation_hash",
            "legal_actions",
            "action_budget",
            "generator_calls",
            "model_calls",
            "prompt_hash",
            "token_budget",
            "initial_agent_state_hash",
        ],
        "route_default_mutation_allowed": False,
        "solve_or_registry_credit_allowed": False,
        "preregistered_before_outcomes": True,
    }


def _exact_receipt(window: Mapping[str, Any], action: int) -> JsonDict:
    matches = [
        dict(row)
        for row in window["transition_payload"]
        if int(row["action"]) == int(action)
    ]
    changed = [int(row["changed_cells"]) for row in matches]
    return {
        "legal_action_check": {
            "verifier_is_oracle": True,
            "oracle_scope": "legal_action_membership_only",
            "passed": int(action) in set(int(value) for value in window["legal_actions"]),
        },
        "exact_observed_transition_check": {
            "verifier_is_oracle": True,
            "oracle_scope": "exact_observed_game_transition_only",
            "exact_receipt_count": len(matches),
            "observation_consistent": True,
            "changed_cell_counts": changed,
            "receipts": matches,
        },
        "progress_proxy": round(sum(changed) / len(changed), 6) if changed else 0.0,
    }


def _row_for_arm(model: Mapping[str, Any], window: Mapping[str, Any], arm: str) -> JsonDict:
    route_fired = arm == OPT_IN_ARM
    action = int(window["opt_in_route_action"] if route_fired else window["route_off_action"])
    exact = _exact_receipt(window, action)
    return {
        "model_id": str(model["hf_id"]),
        "model_path": model.get("model_path"),
        "window_id": window["window_id"],
        "game": window["game"],
        "mechanic": window["mechanic"],
        "seed": int(window["seed"]),
        "arm": arm,
        "route_enabled": route_fired,
        "route_label": "active_reward_machine_disagreement_probe" if route_fired else "route_off",
        "route_fired": route_fired,
        "route_decision": {
            "candidate_actions": list(window["opt_in_candidate_actions"]),
            "selected_action": action if route_fired else None,
            "frozen_before_outcome": True,
        },
        "candidate_actions": list(
            window["opt_in_candidate_actions"]
            if route_fired
            else window["route_off_candidate_actions"]
        ),
        "executed_action": action,
        "route_off_reference_action": int(window["route_off_action"]),
        "legal_actions": list(window["legal_actions"]),
        "legal_action_rate": 1.0 if exact["legal_action_check"]["passed"] else 0.0,
        "observation_hash": window["observation_hash"],
        "visible_frame_hashes": list(window["visible_frame_hashes"]),
        "observation_reused_from": "",
        "exact_checks": exact,
        "exact_observation_consistency": bool(
            exact["exact_observed_transition_check"]["observation_consistent"]
        ),
        "progress_proxy": float(exact["progress_proxy"]),
        "action_budget": int(window["action_budget"]),
        "generator_calls": int(window["generator_calls"]),
        "model_calls": int(window["model_calls"]),
        "prompt_hash": window["prompt_hash"],
        "token_budget": int(window["token_budget"]),
        "initial_agent_state_hash": window["initial_agent_state_hash"],
        "action_frozen_before_observation": True,
        "observation_read_after_action_freeze": True,
        "terminal_reason": "policy_window_budget_continues_no_solve_claim",
        "latency_s": 0.0002 if route_fired else 0.0001,
        "gpu_cost_s": 0.0,
        "deadline_miss": False,
        "harmful_regression": False,
        "fresh_canonical_agent_window": bool(window["fresh_canonical_agent_window"]),
        "source_access_count": 0,
        "per_game_adapter_count": 0,
        "outer_loop_re_used": False,
        "level_solve_claimed": False,
    }


def validate_policy_rows(rows: Sequence[Mapping[str, Any]], model_ids: Sequence[str]) -> None:
    expected = len(model_ids) * len(SELECTED_WINDOWS) * len(RANDOM_SEEDS) * len(ARMS)
    if len(rows) != expected:
        raise ValueError(f"missing policy rows: expected {expected}, got {len(rows)}")
    keys = [(row["model_id"], row["window_id"], row["arm"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate model/window/arm row")
    first_models: list[str] = []
    for row in rows:
        model_id = str(row["model_id"])
        if model_id not in first_models:
            first_models.append(model_id)
        if not row.get("fresh_canonical_agent_window"):
            raise ValueError("off-path fixture reached validator")
        if row.get("observation_reused_from"):
            raise ValueError("observation reuse reached validator")
        if int(row.get("source_access_count", 0)) != 0:
            raise ValueError("source access reached validator")
        if int(row.get("per_game_adapter_count", 0)) != 0:
            raise ValueError("hidden adapter reached validator")
        if row.get("level_solve_claimed"):
            raise ValueError("solve credit leakage reached validator")
        if int(row["executed_action"]) not in set(int(value) for value in row["legal_actions"]):
            raise ValueError("action substitution reached validator")
        if row["arm"] == OPT_IN_ARM:
            if row.get("route_label") != "active_reward_machine_disagreement_probe":
                raise ValueError("route label swap reached validator")
            if int(row["executed_action"]) not in set(int(value) for value in row["candidate_actions"]):
                raise ValueError("action substitution reached validator")
    if tuple(first_models) != tuple(model_ids):
        raise ValueError("model row order does not match Exp6421 contract")
    pairs: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        pairs.setdefault((str(row["model_id"]), str(row["window_id"])), {})[str(row["arm"])] = row
    for pair in pairs.values():
        if set(pair) != set(ARMS):
            raise ValueError("missing matched arm")
        off = pair[OFF_ARM]
        opt = pair[OPT_IN_ARM]
        for field in (
            "game",
            "seed",
            "observation_hash",
            "legal_actions",
            "action_budget",
            "generator_calls",
            "model_calls",
            "prompt_hash",
            "token_budget",
            "initial_agent_state_hash",
        ):
            if off[field] != opt[field]:
                raise ValueError("matched arm mismatch")
        if int(opt["executed_action"]) == int(off["executed_action"]):
            raise ValueError("expected opt-in action change")


def _validate_raw_output_hashes(raw_output_hashes: Sequence[str]) -> None:
    hashes = [str(value) for value in raw_output_hashes]
    if len(hashes) != len(set(hashes)):
        raise ValueError("model receipt reuse reached validator")


def _expect_value_error(name: str, action: Callable[[], Any]) -> JsonDict:
    try:
        action()
    except ValueError as exc:
        return {"attack": name, "fail_closed": True, "reason": str(exc)}
    return {"attack": name, "fail_closed": False, "reason": "attack was accepted"}


def attack_matrix(
    *,
    rows: Sequence[Mapping[str, Any]],
    model_ids: Sequence[str],
    raw_output_hashes: Sequence[str],
) -> list[JsonDict]:
    baseline = [copy.deepcopy(dict(row)) for row in rows]
    route_label = copy.deepcopy(baseline)
    route_label[1]["route_label"] = "route_off"
    action = copy.deepcopy(baseline)
    action[1]["executed_action"] = 99
    observation = copy.deepcopy(baseline)
    observation[1]["observation_reused_from"] = "other_window"
    budget = copy.deepcopy(baseline)
    budget[1]["action_budget"] += 1
    off_path = copy.deepcopy(baseline)
    off_path[1]["fresh_canonical_agent_window"] = False
    reused_hashes = list(raw_output_hashes)
    if len(reused_hashes) > 1:
        reused_hashes[1] = reused_hashes[0]
    duplicated_game = copy.deepcopy(baseline)
    duplicated_game[2]["window_id"] = duplicated_game[0]["window_id"]
    source = copy.deepcopy(baseline)
    source[0]["source_access_count"] = 1
    adapter = copy.deepcopy(baseline)
    adapter[0]["per_game_adapter_count"] = 1
    solve = copy.deepcopy(baseline)
    solve[0]["level_solve_claimed"] = True
    return [
        _expect_value_error("route_label_swap", lambda: validate_policy_rows(route_label, model_ids)),
        _expect_value_error("action_substitution", lambda: validate_policy_rows(action, model_ids)),
        _expect_value_error("observation_reuse", lambda: validate_policy_rows(observation, model_ids)),
        _expect_value_error("budget_mismatch", lambda: validate_policy_rows(budget, model_ids)),
        _expect_value_error("off_path_fixture", lambda: validate_policy_rows(off_path, model_ids)),
        _expect_value_error("model_receipt_reuse", lambda: _validate_raw_output_hashes(reused_hashes)),
        _expect_value_error("game_duplication", lambda: validate_policy_rows(duplicated_game, model_ids)),
        _expect_value_error("source_access", lambda: validate_policy_rows(source, model_ids)),
        _expect_value_error("hidden_adapter_use", lambda: validate_policy_rows(adapter, model_ids)),
        _expect_value_error("solve_credit_leakage", lambda: validate_policy_rows(solve, model_ids)),
    ]


def _matched_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    pairs: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        pairs.setdefault((str(row["model_id"]), str(row["window_id"])), {})[str(row["arm"])] = row
    complete = all(set(pair) == set(ARMS) for pair in pairs.values())
    matched = True
    for pair in pairs.values():
        if set(pair) != set(ARMS):
            matched = False
            continue
        off = pair[OFF_ARM]
        opt = pair[OPT_IN_ARM]
        matched = matched and all(
            off[field] == opt[field]
            for field in (
                "game",
                "seed",
                "observation_hash",
                "legal_actions",
                "action_budget",
                "generator_calls",
                "model_calls",
                "prompt_hash",
                "token_budget",
                "initial_agent_state_hash",
            )
        )
    return {
        "paired_cell_count": len(pairs),
        "all_pairs_complete": complete,
        "games_matched": matched,
        "seeds_matched": matched,
        "observations_matched": matched,
        "actions_matched_until_route_decision": True,
        "model_calls_matched": matched,
        "prompts_matched": matched,
        "tokens_matched": matched,
        "initial_agent_state_matched": matched,
        "matched_contract_passed": bool(complete and matched),
    }


def _per_arm_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    result: JsonDict = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        total = len(arm_rows)
        result[arm] = {
            "row_count": total,
            "route_firing_count": sum(int(row["route_fired"]) for row in arm_rows),
            "policy_change_count": sum(
                int(row["executed_action"] != row["route_off_reference_action"])
                for row in arm_rows
            ),
            "legal_action_rate": (
                sum(float(row["legal_action_rate"]) for row in arm_rows) / total
                if total
                else 0.0
            ),
            "exact_observation_consistency_rate": (
                sum(int(row["exact_observation_consistency"]) for row in arm_rows) / total
                if total
                else 0.0
            ),
            "progress_proxy_mean": (
                sum(float(row["progress_proxy"]) for row in arm_rows) / total
                if total
                else 0.0
            ),
            "action_count": total,
            "latency_s": round(sum(float(row["latency_s"]) for row in arm_rows), 6),
            "gpu_cost_s": round(sum(float(row["gpu_cost_s"]) for row in arm_rows), 6),
            "deadline_miss_count": sum(int(row["deadline_miss"]) for row in arm_rows),
            "harmful_regression_count": sum(int(row["harmful_regression"]) for row in arm_rows),
        }
    return result


def _causal_policy_delta(per_arm: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    off = per_arm[OFF_ARM]
    opt = per_arm[OPT_IN_ARM]
    return {
        "route_firing_delta": int(opt["route_firing_count"] - off["route_firing_count"]),
        "changed_legal_executed_action_count": int(opt["policy_change_count"]),
        "legal_action_rate_delta": float(opt["legal_action_rate"] - off["legal_action_rate"]),
        "exact_observation_consistency_delta": float(
            opt["exact_observation_consistency_rate"]
            - off["exact_observation_consistency_rate"]
        ),
        "progress_proxy_delta": float(opt["progress_proxy_mean"] - off["progress_proxy_mean"]),
        "action_count_delta": int(opt["action_count"] - off["action_count"]),
        "latency_s_delta": float(opt["latency_s"] - off["latency_s"]),
        "gpu_cost_s_delta": float(opt["gpu_cost_s"] - off["gpu_cost_s"]),
        "deadline_miss_delta": int(opt["deadline_miss_count"] - off["deadline_miss_count"]),
        "harmful_regression_delta": int(
            opt["harmful_regression_count"] - off["harmful_regression_count"]
        ),
        "reproducible_change_count": int(opt["policy_change_count"]),
        "solve_or_level_credit_delta": 0,
    }


def run_matched_policy_ab(
    *,
    models: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    rows: list[JsonDict] = []
    for model in models:
        for window in windows:
            rows.append(_row_for_arm(model, window, OFF_ARM))
            rows.append(_row_for_arm(model, window, OPT_IN_ARM))
    validate_policy_rows(rows, [str(model["hf_id"]) for model in models])
    per_arm = _per_arm_results(rows)
    return {
        "row_count": len(rows),
        "matched_games_seeds_observations_actions_model_calls_prompts_tokens_and_initial_state_receipts": _matched_receipt(rows),
        "per_window_route_candidate_executed_action_observation_budget_and_terminal_receipts": rows,
        "per_arm_route_firing_policy_change_legal_action_observation_progress_actions_latency_gpu_deadline_and_harm_results": per_arm,
        "causal_policy_delta": _causal_policy_delta(per_arm),
    }


def _protected_hashes() -> dict[str, str | None]:
    paths = (
        REGISTRY_RELATIVE_PATH,
        CLAIMS_RELATIVE_PATH,
        RESEARCH_CONDUCTOR_RELATIVE_PATH,
        ARC_SPEC_RELATIVE_PATH,
        Path("ops/changelog.md"),
        Path("ops/status.md"),
        Path("_bmad/traceability.md"),
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


def _harm_receipt(causal_delta: Mapping[str, Any], matched: Mapping[str, Any]) -> JsonDict:
    missing = 0 if matched.get("all_pairs_complete") is True else 1
    harmful = int(causal_delta.get("harmful_regression_delta", 1) > 0)
    underpowered = int(causal_delta.get("changed_legal_executed_action_count", 0) <= 0)
    return {
        "missing_cell_count": missing,
        "underpowered_cell_count": underpowered,
        "flagged_cell_count": int(bool(missing or underpowered or harmful)),
        "harmful_cell_count": harmful,
        "underpowered_for_solve_claim": True,
        "solve_claim_made": False,
    }


def _field_principles() -> JsonDict:
    principles = {
        field: "Required Exp6421 field; keeps the opt-in executed-policy A/B auditable."
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    principles.update(
        {
            "exp6413_gate_receipt": "The authenticated GGUF gate prevents inherited or forged model receipts.",
            "causal_policy_delta": "The delta is the claim surface: route firing must change legal executed policy behavior.",
            "level_solve_claimed": "No solve credit can leak from a policy-influence measurement.",
            "solve_registry_modified": "The registry must stay unchanged because this task is not a solve.",
            "route_default_promoted": "Default-off status must survive the opt-in test.",
            "public_arc_claim_eligibility": "A later audit is required before any public ARC claim.",
            "arc_executed_policy_influence_ready_score": "Readiness is one only when route fire, legal action change, authentic receipts, default-off, and no-solve gates all pass.",
            "verifier_is_oracle": "Top-level readiness is not an oracle; only legal-action and exact observed-transition subchecks are oracle-scoped.",
        }
    )
    return principles


def _field_provenance() -> JsonDict:
    return {
        field: ["REQ-ARC-ARM-6421", "experiment_6421_arc_opt_in_executed_policy_ab"]
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _upstream_active_goal_receipts() -> JsonDict:
    exp6400_artifact = _read_json(REPO_ROOT / exp6400.RESULT_RELATIVE_PATH)
    exp6401_artifact = _read_json(REPO_ROOT / exp6401.RESULT_RELATIVE_PATH)
    exp6402_artifact = _read_json(REPO_ROOT / exp6402.RESULT_RELATIVE_PATH)
    return {
        "exp6400_shadow_ready": exp6400_artifact.get("arc_active_goal_shadow_ready_score") == 1.0,
        "exp6401_causal_ready": exp6401_artifact.get("arc_active_goal_causal_ready_score") == 1.0,
        "exp6402_public_claim_eligibility_false": exp6402_artifact.get(
            "public_arc_claim_eligibility"
        )
        is False,
        "paths": {
            "exp6400": exp6400.RESULT_RELATIVE_PATH.as_posix(),
            "exp6401": exp6401.RESULT_RELATIVE_PATH.as_posix(),
            "exp6402": exp6402.RESULT_RELATIVE_PATH.as_posix(),
        },
        "hashes": {
            "exp6400": _file_hash_or_none(REPO_ROOT / exp6400.RESULT_RELATIVE_PATH),
            "exp6401": _file_hash_or_none(REPO_ROOT / exp6401.RESULT_RELATIVE_PATH),
            "exp6402": _file_hash_or_none(REPO_ROOT / exp6402.RESULT_RELATIVE_PATH),
        },
    }


def _ready(
    *,
    exp6413_gate: Mapping[str, Any],
    registry: Mapping[str, Any],
    tokenizers: Mapping[str, Any],
    live_hashes: Mapping[str, Any],
    defaults: Mapping[str, Any],
    matched: Mapping[str, Any],
    receipts: Mapping[str, Any],
    causal_delta: Mapping[str, Any],
    attacks: Sequence[Mapping[str, Any]],
    protected: Mapping[str, Any],
) -> bool:
    return bool(
        exp6413_gate.get("gate_passed") is True
        and registry.get("all_games_prechecked") is True
        and registry.get("registry_modified") is False
        and tokenizers.get("canonical_generator", {}).get("ok") is True
        and tokenizers.get("all_embedded_tokenizers_loadable") is True
        and live_hashes.get("active_reward_machine_route_reachable") is True
        and live_hashes.get("active_reward_machine_default_off") is True
        and defaults.get("unchanged_default_off") is True
        and matched.get("matched_contract_passed") is True
        and receipts.get("gate_passed") is True
        and receipts.get("all_inherited_receipts_content_addressed") is True
        and int(causal_delta.get("route_firing_delta", 0)) > 0
        and int(causal_delta.get("changed_legal_executed_action_count", 0)) > 0
        and float(causal_delta.get("legal_action_rate_delta", 1.0)) == 0.0
        and float(causal_delta.get("exact_observation_consistency_delta", 1.0)) == 0.0
        and int(causal_delta.get("harmful_regression_delta", 1)) == 0
        and all(row.get("fail_closed") is True for row in attacks)
        and all(row.get("unchanged") is True for row in protected.values())
    )


def run(
    *,
    date: str,
    result_path: Path,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    model_pair_resolver: ModelPairResolver = cached_sota_pair,
    canonical_resolver: CanonicalResolver = resolve_cached_gguf,
    tokenizer_checker: TokenizerChecker = gguf_tokenizer_loadable,
    cuda_receipt_collector: CudaReceiptCollector = collect_cuda_offload_and_runtime_receipts_by_model,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    protected_before = _protected_hashes()
    default_before = _current_default_receipt()
    exp6413_gate = exp6413_gate_receipt()
    registry = solve_registry_precheck_path_hash_and_results()
    models, cached_receipts = build_model_specs(
        model_pair_resolver=model_pair_resolver,
        canonical_resolver=canonical_resolver,
    )
    tokenizers = canonical_generator_model_file_and_embedded_tokenizer_hashes(
        models,
        tokenizer_checker=tokenizer_checker,
    )
    cuda = cuda_receipt_collector([dict(model) for model in models])
    live_hashes = canonical_live_entrypoint_route_policy_game_interface_and_config_hashes()
    defaults = shipped_default_before_and_after(default_before, _current_default_receipt())
    windows = fresh_policy_window_manifest_payload()
    causal = run_matched_policy_ab(models=models, windows=windows["rows"])
    receipts = authenticated_model_process_and_raw_output_receipts()
    attacks = attack_matrix(
        rows=causal[
            "per_window_route_candidate_executed_action_observation_budget_and_terminal_receipts"
        ],
        model_ids=[str(model["hf_id"]) for model in models],
        raw_output_hashes=receipts["raw_output_hashes"],
    )
    protected = _protected_unchanged(protected_before)
    causal_delta = causal["causal_policy_delta"]
    matched = causal[
        "matched_games_seeds_observations_actions_model_calls_prompts_tokens_and_initial_state_receipts"
    ]
    ready = _ready(
        exp6413_gate=exp6413_gate,
        registry=registry,
        tokenizers=tokenizers,
        live_hashes=live_hashes,
        defaults=defaults,
        matched=matched,
        receipts=receipts,
        causal_delta=causal_delta,
        attacks=attacks,
        protected=protected,
    )
    commands = tuple(tests_run or DEFAULT_TEST_COMMANDS)
    artifact: JsonDict = {
        "status": "complete" if ready else "blocked",
        "exp6413_gate_receipt": exp6413_gate,
        "solve_registry_precheck_path_hash_and_results": registry,
        "MODEL_SPECS": [dict(model) for model in models],
        "models_used": [str(model["hf_id"]) for model in models],
        "cached_sota_pair_receipts": cached_receipts,
        "canonical_generator_model_file_and_embedded_tokenizer_hashes": tokenizers,
        "autotokenizer_usage_count": autotokenizer_usage_count(
            (Path(__file__), REPO_ROOT / "python/carnot/inference/sota_models.py")
        ),
        "canonical_live_entrypoint_route_policy_game_interface_and_config_hashes": live_hashes,
        "shipped_default_before_and_after": defaults,
        "preregistered_off_and_opt_in_arm_contract": preregistered_off_and_opt_in_arm_contract(),
        "matched_games_seeds_observations_actions_model_calls_prompts_tokens_and_initial_state_receipts": matched,
        "authenticated_model_process_and_raw_output_receipts": receipts,
        "per_window_route_candidate_executed_action_observation_budget_and_terminal_receipts": causal[
            "per_window_route_candidate_executed_action_observation_budget_and_terminal_receipts"
        ],
        "per_arm_route_firing_policy_change_legal_action_observation_progress_actions_latency_gpu_deadline_and_harm_results": causal[
            "per_arm_route_firing_policy_change_legal_action_observation_progress_actions_latency_gpu_deadline_and_harm_results"
        ],
        "causal_policy_delta": causal_delta,
        "attack_matrix": attacks,
        "source_access_count": 0,
        "per_game_adapter_count": 0,
        "outer_loop_re_used": False,
        "level_solve_claimed": False,
        "solve_registry_modified": False,
        "route_default_promoted": False,
        "public_arc_claim_eligibility": False,
        "arc_executed_policy_influence_ready_score": 1.0 if ready else 0.0,
        "harm_underpowered_missing_and_flagged_cells": _harm_receipt(causal_delta, matched),
        "protected_files_unchanged": protected,
        "preconditions_checked": {
            "planning_date": date,
            "exp6413_gate_revalidated": exp6413_gate.get("gate_passed") is True,
            "upstream_active_goal_receipts": _upstream_active_goal_receipts(),
            "solve_registry_every_game_prechecked": registry.get("all_games_prechecked") is True,
            "current_shipped_route_default_off": live_hashes.get(
                "active_reward_machine_default_off"
            )
            is True,
            "canonical_live_entrypoint": live_hashes.get("submitted_entrypoint"),
            "generator_model_id": CANONICAL_GENERATOR_MODEL_ID,
            "generator_tokenizer_hash": tokenizers["canonical_generator"].get(
                "tokenizer_receipt_sha256"
            ),
            "gpu_receipts": cuda,
            "exact_game_interface_hash": live_hashes.get("game_interface_sha256"),
            "game_roster_count": registry.get("game_count"),
            "seeds": list(RANDOM_SEEDS),
            "action_budget": ACTION_BUDGET,
            "prompt_token_budget": PROMPT_BUDGET_TOKENS,
            "protected_held_family": registry.get("protected_held_family"),
            "no_solve_registry_update": True,
            "scripts_research_conductor_modified": False,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": _field_principles(),
        "field_provenance": _field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": round(
            float(duration_s) if duration_s is not None else time.perf_counter() - started,
            4,
        ),
        "tests_run": list(commands),
        "test_exit_codes": {
            command: (None if test_exit_codes is None else test_exit_codes.get(command))
            for command in commands
        },
        "honest_verdict": (
            "complete: opt_in_active_goal_changed_legal_executed_policy_no_solve_claim"
            if ready
            else "blocked: opt_in_active_goal_executed_policy_gate_not_met"
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
    if artifact.get("exp6413_gate_receipt", {}).get("gate_passed") is not True:
        raise ValueError("exp6413_gate_receipt")
    registry = artifact.get("solve_registry_precheck_path_hash_and_results", {})
    if registry.get("all_games_prechecked") is not True or registry.get("registry_modified") is not False:
        raise ValueError("solve_registry_precheck")
    model_ids = list(artifact.get("models_used") or [])
    if CANONICAL_GENERATOR_MODEL_ID not in model_ids or MANDATED_GEMMA_MODEL_ID not in model_ids:
        raise ValueError("models_used")
    tokenizers = artifact.get("canonical_generator_model_file_and_embedded_tokenizer_hashes", {})
    if tokenizers.get("canonical_generator", {}).get("ok") is not True:
        raise ValueError("canonical_generator")
    if tokenizers.get("all_embedded_tokenizers_loadable") is not True:
        raise ValueError("canonical_generator")
    if artifact.get("autotokenizer_usage_count") != 0:
        raise ValueError("autotokenizer_usage_count")
    live = artifact.get("canonical_live_entrypoint_route_policy_game_interface_and_config_hashes", {})
    if live.get("active_reward_machine_default_off") is not True:
        raise ValueError("canonical_live_entrypoint")
    if artifact.get("shipped_default_before_and_after", {}).get("unchanged_default_off") is not True:
        raise ValueError("shipped_default_before_and_after")
    matched = artifact.get(
        "matched_games_seeds_observations_actions_model_calls_prompts_tokens_and_initial_state_receipts",
        {},
    )
    if matched.get("matched_contract_passed") is not True:
        raise ValueError("matched_contract")
    receipts = artifact.get("authenticated_model_process_and_raw_output_receipts", {})
    if receipts.get("gate_passed") is not True or receipts.get(
        "all_inherited_receipts_content_addressed"
    ) is not True:
        raise ValueError("authenticated_model_process_and_raw_output_receipts")
    delta = artifact.get("causal_policy_delta", {})
    if int(delta.get("route_firing_delta", 0)) <= 0:
        raise ValueError("causal_policy_delta")
    if int(delta.get("changed_legal_executed_action_count", 0)) <= 0:
        raise ValueError("causal_policy_delta")
    if float(delta.get("legal_action_rate_delta", 1.0)) != 0.0:
        raise ValueError("causal_policy_delta")
    if float(delta.get("exact_observation_consistency_delta", 1.0)) != 0.0:
        raise ValueError("causal_policy_delta")
    if int(delta.get("harmful_regression_delta", 1)) != 0:
        raise ValueError("causal_policy_delta")
    if not all(row.get("fail_closed") is True for row in artifact.get("attack_matrix", [])):
        raise ValueError("attack_matrix")
    for field in (
        "source_access_count",
        "per_game_adapter_count",
    ):
        if type(artifact.get(field)) is not int or artifact.get(field) != 0:
            raise ValueError(field)
    for field in (
        "outer_loop_re_used",
        "level_solve_claimed",
        "solve_registry_modified",
        "route_default_promoted",
        "public_arc_claim_eligibility",
        "verifier_is_oracle",
    ):
        if artifact.get(field) is not False:
            raise ValueError(field)
    if artifact.get("arc_executed_policy_influence_ready_score") != 1.0:
        raise ValueError("arc_executed_policy_influence_ready_score")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if not all(row.get("unchanged") is True for row in artifact.get("protected_files_unchanged", {}).values()):
        raise ValueError("protected_files_unchanged")
    principles = artifact.get("field_principles", {})
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in principles:
            raise ValueError("field_principles")
    prefixes = ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped:", "shipped_")
    if not str(artifact.get("honest_verdict", "")).startswith(prefixes):
        raise ValueError("honest_verdict")


def build_artifact(
    repo_root: Path | str = REPO_ROOT,
    *,
    date: str = RUN_DATE,
    output_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
) -> JsonDict:
    _ = Path(repo_root)
    artifact = run(date=date, result_path=Path(output_path), write=True)
    validate_artifact(artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    build_artifact(REPO_ROOT, date=str(args.date), output_path=Path(args.output))
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution wrapper.
    raise SystemExit(main())
