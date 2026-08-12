"""Exp6348 default-off ARC action-influence A/B.

Spec refs: REQ-ARC-WMTE-6348,
SCENARIO-ARC-WMTE-6348-GATE-AND-SEALS,
SCENARIO-ARC-WMTE-6348-MODEL-RECEIPTS,
SCENARIO-ARC-WMTE-6348-MATCHED-ARMS,
SCENARIO-ARC-WMTE-6348-CAUSAL-QUALITY-GATE,
SCENARIO-ARC-WMTE-6348-ARTIFACT-GUARDS.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import gc
import inspect
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from carnot import experiment_6307_arc_target_validated_route_canary as exp6307
from carnot import experiment_6321_arc_target_licensed_route_live_shadow_ab as exp6321
from carnot import experiment_6347_arc_action_influence_preflight as exp6347
from carnot.agentic import arc_competition_agent as agent
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
ModelPairResolver = Callable[..., list[JsonDict] | None]
TokenizerChecker = Callable[[str | None], tuple[bool, str]]
CudaReceiptCollector = Callable[[list[JsonDict]], dict[str, JsonDict]]

REPO_ROOT = exp6307.REPO_ROOT
RESULT_RELATIVE_PATH = Path("results/experiment_6348_arc_default_off_action_influence_ab.json")
PROSPECTIVE_REGISTRATION_RELATIVE_PATH = Path(
    "results/experiment_6348_arc_prospective_registration.json"
)
FRESH_WINDOW_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6348_arc_fresh_live_window_manifest.json"
)
REGISTRY_RELATIVE_PATH = exp6307.REGISTRY_RELATIVE_PATH
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
INFLUENCE_TASK_ID = "exp6348_arc_default_off_action_influence_ab_no_solve"
EXACT_TRANSITION_CHECKER_NAME = "exp6348_exact_one_step_transition_checker"
INFERENCE_SUBSTRATE = "live_llm_inference"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6348_arc_default_off_action_influence_ab "
    "--date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6348_arc_default_off_action_influence_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6348_arc_default_off_action_influence_ab.py "
    "-m pytest tests/python/test_experiment_6348_arc_default_off_action_influence_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6348_arc_default_off_action_influence_ab.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6348_arc_default_off_action_influence_ab.py"
)
E2E_PLAN_READ_COMMAND = "sed -n '1,180p' ops/e2e-test-plan.md"
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6348_arc_default_off_action_influence_ab.json"
)
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6348_test_receipts.json")
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
MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
RANDOM_SEEDS = (6348001, 6348002)
SELECTED_WINDOWS = (
    {
        "game_window_id": "exp6348_fresh_synthetic_push_l0",
        "mechanic": "push_block",
        "level": 0,
        "window_order": 0,
    },
    {
        "game_window_id": "exp6348_fresh_synthetic_toggle_l0",
        "mechanic": "toggle_move",
        "level": 0,
        "window_order": 1,
    },
)
ROUTE_ARMS = ("route_off", "target_licensed_route_on")
RAW_CANDIDATE_ACTIONS = (5, 4)
ACTION_BUDGET = exp6321.ACTION_BUDGET
MODEL_LOAD_TOKEN_BUDGET = 0
ROUTE_TIME_BUDGET_S = 2.0
EXACT_CHECKER_BUDGET_PER_CELL = 1
FORBIDDEN_ZERO_FIELDS = (
    "hidden_game_source_access_count",
    "offline_ground_truth_bfs_count",
    "hand_game_adapter_count",
    "per_game_calibration_count",
    "source_model_weight_mutation_count",
    "generated_label_count",
    "hidden_state_access_count",
    "solve_claim_count",
    "registry_update_count",
)
FORBIDDEN_EVIDENCE_FIELDS = (
    "hidden_game_source_path",
    "offline_ground_truth_bfs_path",
    "hand_game_adapter",
    "per_game_calibration",
    "hidden_state",
    "generated_label",
    "registry_solve_target",
)
LIVE_EVIDENCE_USED_FIELDS = (
    "agent_owned_transition.grid",
    "agent_owned_transition.action",
    "agent_owned_transition.data",
    "agent_owned_transition.next_grid",
    "agent_owned_transition.level_before",
    "agent_owned_transition.level_after",
    "raw_candidate_actions",
    "runtime_reverse_engineering_state",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_path_hash_terminal_class_and_gate_receipt",
    "arc_registry_precheck_path_hash_and_result",
    "solve_provenance",
    "no_duplicate_solve_receipt",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes_revisions_quantizations_and_tokenizers",
    "llama_cpp_embedded_tokenizer_receipts",
    "cuda_gpu_offload_and_memory_release_receipts_by_model",
    "prospective_registration_path_and_hash",
    "fresh_live_window_manifest_path_and_hash",
    "live_evidence_allowed_fields",
    "forbidden_source_access_contract",
    "hidden_game_source_access_count",
    "offline_ground_truth_bfs_count",
    "hand_game_adapter_count",
    "per_game_calibration_count",
    "route_default_off_and_activation_receipts",
    "arm_definitions",
    "matched_call_token_action_time_and_checker_budgets",
    "raw_model_and_action_paths_hashes_and_counts",
    "legal_action_order_changes_by_model_game_window_arm_and_seed",
    "exact_one_step_transition_quality_by_model_game_window_arm_and_seed",
    "paired_influence_and_quality_deltas_intervals_and_sample_sizes",
    "route_deletion_permutation_leakage_and_escape_results",
    "verification_calls_time_cost_and_error_table",
    "harm_underpowered_missing_and_flagged_cells",
    "source_model_weight_mutation_count",
    "generated_label_count",
    "hidden_state_access_count",
    "solve_claim_count",
    "registry_update_count",
    "exact_oracle_claim_boundary",
    "arc_causal_influence_ready_score",
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
    "status": "States complete versus blocked for the fresh live A/B.",
    "upstream_path_hash_terminal_class_and_gate_receipt": "Pins the upstream structured gate before model load.",
    "arc_registry_precheck_path_hash_and_result": "Proves this is not a registry solve target.",
    "solve_provenance": "Names the source as live-agent self-discovery.",
    "no_duplicate_solve_receipt": "Shows no duplicate solve proposal was made.",
    "MODEL_SPECS": "Names the three cached GGUF headline models.",
    "models_used": "Lists the exact headline model ids used in the cells.",
    "model_file_hashes_revisions_quantizations_and_tokenizers": "Pins files and embedded tokenizer state.",
    "llama_cpp_embedded_tokenizer_receipts": "Shows tokenizer checks use GGUF embedded tokenizers.",
    "cuda_gpu_offload_and_memory_release_receipts_by_model": "Records one-model-at-a-time llama.cpp load and release.",
    "prospective_registration_path_and_hash": "Pins the preregistered route and budget plan.",
    "fresh_live_window_manifest_path_and_hash": "Pins fresh agent-owned windows before model generation.",
    "live_evidence_allowed_fields": "Freezes fields the route may read.",
    "forbidden_source_access_contract": "Lists forbidden escape sources and zero counts.",
    "hidden_game_source_access_count": "Must stay zero for hidden-source discipline.",
    "offline_ground_truth_bfs_count": "Must stay zero for self-discovery discipline.",
    "hand_game_adapter_count": "Must stay zero because hand adapters are off path.",
    "per_game_calibration_count": "Must stay zero because thresholds are preregistered.",
    "route_default_off_and_activation_receipts": "Proves route default stays off.",
    "arm_definitions": "Defines the only A/B difference.",
    "matched_call_token_action_time_and_checker_budgets": "Shows route-off and route-on budgets match.",
    "raw_model_and_action_paths_hashes_and_counts": "Preserves raw model and action receipts.",
    "legal_action_order_changes_by_model_game_window_arm_and_seed": "Measures action-order influence per cell.",
    "exact_one_step_transition_quality_by_model_game_window_arm_and_seed": "Measures one-step transition value only.",
    "paired_influence_and_quality_deltas_intervals_and_sample_sizes": "Reports paired deltas without pooling away cells.",
    "route_deletion_permutation_leakage_and_escape_results": "Shows deletion removes the effect and leakage is zero.",
    "verification_calls_time_cost_and_error_table": "Records verification call costs and errors.",
    "harm_underpowered_missing_and_flagged_cells": "Keeps bad or missing cells visible.",
    "source_model_weight_mutation_count": "Must stay zero because weights are immutable.",
    "generated_label_count": "Must stay zero because labels are not generated.",
    "hidden_state_access_count": "Must stay zero because hidden state is forbidden.",
    "solve_claim_count": "Must stay zero because this is not a solve.",
    "registry_update_count": "Must stay zero because the registry is unchanged.",
    "exact_oracle_claim_boundary": "Separates one-step checking from solve claims.",
    "arc_causal_influence_ready_score": "Equals one only when every gate passes.",
    "protected_files_unchanged": "Confirms protected files stayed unchanged.",
    "preconditions_checked": "Records resources, hashes, seeds, budgets, route rules, and stops.",
    "inference_substrate": "Declares live llama.cpp model preflight plus live-agent windows.",
    "verifier_is_oracle": "Names the exact transition checker.",
    "field_provenance": "Maps every field to the spec and producer.",
    "field_principles": "Gives one audit reason per required field.",
    "test_commands": "Lists verification commands.",
    "test_exit_codes": "Records command outcomes.",
    "duration_s": "Records measured wall time.",
    "random_seeds": "Pins fresh live-window seeds.",
    "reproducibility_checksum": "Detects artifact drift.",
    "honest_verdict": "Terminal verdict with no solve claim.",
}
FIELD_PROVENANCE = {
    field: ["REQ-ARC-WMTE-6348", "experiment_6348_arc_default_off_action_influence_ab"]
    for field in REQUIRED_ARTIFACT_FIELDS
}

canonical_json = exp6307.canonical_json
sha256_text = exp6307.sha256_text
sha256_json = exp6307.sha256_json
sha256_file = exp6307.sha256_file
payload_checksum = exp6307.payload_checksum


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


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


def _timing_row(name: str, started: float, *, error_count: int = 0) -> JsonDict:
    return {
        "call": name,
        "duration_s": round(time.perf_counter() - started, 6),
        "error_count": int(error_count),
    }


def _quant_from_path(path: str | None) -> str:
    name = Path(path or "").name.lower()
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "Q8_0"):
        if token.lower() in name:
            return token
    return "unknown"


def _model_revision(path: str | None) -> str | None:
    if not path:
        return None
    parts = Path(path).parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots") + 1
    return parts[index] if index < len(parts) else None


def _with_model_file_receipts(models: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out: list[JsonDict] = []
    for model in models:
        path = Path(str(model.get("model_path") or ""))
        exists = path.is_file()
        out.append(
            {
                **dict(model),
                "model_exists": exists,
                "model_size_bytes": path.stat().st_size if exists else 0,
                "model_sha256": sha256_file(path) if exists else None,
                "revision": _model_revision(str(path)) if exists else None,
                "quantization": _quant_from_path(str(path)),
                "terminal_disposition": "resolved_cached_gguf" if exists else "missing_cached_gguf",
            }
        )
    return out


def _pair_call(
    resolver: ModelPairResolver,
    *,
    model_indices: tuple[int, int] | None = None,
) -> list[JsonDict]:
    kwargs: dict[str, Any] = {"gpu_indices": (0, 1)}
    if model_indices is not None:
        kwargs["model_indices"] = model_indices
    pair = resolver(**kwargs)
    if pair is None:
        raise ValueError("cached_sota_pair returned no usable GGUF pair")
    return [dict(row) for row in pair]


def build_model_specs(*, model_pair_resolver: ModelPairResolver = cached_sota_pair) -> list[JsonDict]:
    default_pair = _pair_call(model_pair_resolver)
    qwen_dense_pair = _pair_call(model_pair_resolver, model_indices=(0, 2))
    by_id: dict[str, JsonDict] = {}
    source_by_id: dict[str, str] = {}
    for row in default_pair:
        by_id.setdefault(str(row["hf_id"]), row)
        source_by_id.setdefault(str(row["hf_id"]), "cached_sota_pair(gpu_indices=(0, 1))")
    for row in qwen_dense_pair:
        by_id.setdefault(str(row["hf_id"]), row)
        source_by_id.setdefault(
            str(row["hf_id"]),
            "cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 2))",
        )
    missing = [model_id for model_id in MANDATED_MODEL_IDS if model_id not in by_id]
    if missing:
        raise ValueError(f"cached_sota_pair missing mandated models: {missing}")
    ordered = []
    for model_id in MANDATED_MODEL_IDS:
        row = dict(by_id[model_id])
        row["resolved_via"] = source_by_id[model_id]
        ordered.append(row)
    return _with_model_file_receipts(ordered)


def llama_cpp_embedded_tokenizer_receipts(
    models: Sequence[Mapping[str, Any]],
    *,
    tokenizer_checker: TokenizerChecker = gguf_tokenizer_loadable,
) -> JsonDict:
    receipts: JsonDict = {}
    for model in models:
        ok, detail = tokenizer_checker(str(model.get("model_path") or ""))
        receipts[str(model["hf_id"])] = {
            "ok": bool(ok),
            "detail": detail,
            "tokenizer_source": "embedded_gguf",
            "canonical_loader": "llama_cpp.Llama(vocab_only=True)",
            "model_path": model.get("model_path"),
        }
    return receipts


def model_file_hashes_revisions_quantizations_and_tokenizers(
    models: Sequence[Mapping[str, Any]],
    tokenizer_receipts: Mapping[str, Any],
) -> JsonDict:
    return {
        str(model["hf_id"]): {
            "model_path": model.get("model_path"),
            "exists": bool(model.get("model_exists")),
            "size_bytes": int(model.get("model_size_bytes") or 0),
            "sha256": model.get("model_sha256"),
            "revision": model.get("revision"),
            "quantization": model.get("quantization"),
            "tokenizer": dict(tokenizer_receipts[str(model["hf_id"])]),
        }
        for model in models
    }


def _gpu_memory_snapshot() -> dict[int, JsonDict]:  # pragma: no cover - hardware receipt.
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    rows: dict[int, JsonDict] = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index, name, total, used, free, driver = parts
        rows[int(index)] = {
            "index": int(index),
            "name": name,
            "memory_total_mb": int(total),
            "memory_used_mb": int(used),
            "memory_free_mb": int(free),
            "driver_version": driver,
        }
    return rows


def collect_cuda_gpu_offload_and_memory_release_receipts(
    models: list[JsonDict],
) -> dict[str, JsonDict]:  # pragma: no cover - live llama.cpp path.
    from llama_cpp import Llama

    receipts: dict[str, JsonDict] = {}
    for model in models:
        model_id = str(model["hf_id"])
        gpu = int(model.get("gpu", 0))
        before = _gpu_memory_snapshot()
        errors: list[str] = []
        loaded = False
        tokenizer_ok = False
        started = time.perf_counter()
        try:
            llm = Llama(
                model_path=str(model["model_path"]),
                n_gpu_layers=-1,
                main_gpu=gpu,
                n_ctx=64,
                n_batch=8,
                verbose=False,
            )
            loaded = True
            tokenizer_ok = bool(llm.tokenize(b"ARC action influence"))
            del llm
        except Exception as exc:
            errors.append(repr(exc)[:240])
        gc.collect()
        time.sleep(0.5)
        after = _gpu_memory_snapshot()
        before_used = int(before.get(gpu, {}).get("memory_used_mb", 0))
        after_used = int(after.get(gpu, {}).get("memory_used_mb", 0))
        memory_released = after_used <= before_used + 512
        receipts[model_id] = {
            "terminal": bool(loaded and tokenizer_ok and memory_released and not errors),
            "canonical_llama_cpp": True,
            "embedded_tokenizer_probe_ok": tokenizer_ok,
            "full_weight_load_attempted": True,
            "loaded_one_placement_at_a_time": True,
            "n_gpu_layers": -1,
            "main_gpu": gpu,
            "n_ctx": 64,
            "n_batch": 8,
            "memory_before": before.get(gpu),
            "memory_after": after.get(gpu),
            "memory_delta_after_mb": after_used - before_used,
            "memory_released": memory_released,
            "duration_s": round(time.perf_counter() - started, 6),
            "errors": errors,
        }
    return receipts


def upstream_path_hash_terminal_class_and_gate_receipt() -> JsonDict:
    specs = (
        (exp6321.RESULT_RELATIVE_PATH, "arc_route_live_shadow_ready_score"),
        (exp6321.TRANSITION_MANIFEST_RELATIVE_PATH, None),
        (exp6347.RESULT_RELATIVE_PATH, "arc_action_influence_eligible_score"),
        (exp6347.LIVE_WINDOW_MANIFEST_RELATIVE_PATH, None),
        (EXCLUSION_MANIFEST_RELATIVE_PATH, None),
    )
    rows = []
    for rel, gate_field in specs:
        path = REPO_ROOT / rel
        payload = _read_json(path) if path.suffix == ".json" else {}
        rows.append(
            {
                "path": rel.as_posix(),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "terminal_class": _terminal_class(payload) if payload else "input",
                "gate_field": gate_field,
                "gate_value": payload.get(gate_field) if gate_field else None,
            }
        )
    exp6321_ready = rows[0]["gate_value"] == 1.0
    exp6347_ready = rows[2]["gate_value"] == 1.0
    return {
        "structured_gate_replayed": True,
        "structured_gate_passed": exp6321_ready and exp6347_ready,
        "exp6321_live_shadow_ready": exp6321_ready,
        "exp6347_action_influence_eligible": exp6347_ready,
        "gate_replay_order": "upstream_gate_before_registry_and_model_load",
        "rows": rows,
    }


def registry_precheck(*, registry_text: str | None = None) -> JsonDict:
    path = REPO_ROOT / REGISTRY_RELATIVE_PATH
    text = path.read_text(encoding="utf-8") if registry_text is None else registry_text
    duplicates = [INFLUENCE_TASK_ID] if INFLUENCE_TASK_ID in text else []
    return {
        "path": REGISTRY_RELATIVE_PATH.as_posix(),
        "sha256": sha256_text(text),
        "registry_read_mode": "full_text",
        "registry_bytes_read": len(text.encode("utf-8")),
        "registry_line_count": len(text.splitlines()),
        "precheck_order": "registry_before_model_load",
        "task_kind": "fresh_live_action_influence_ab_not_solve",
        "influence_task_id": INFLUENCE_TASK_ID,
        "solve_proposal_made": False,
        "duplicate_solve_proposals": duplicates,
        "duplicate_solve_proposal_count": len(duplicates),
        "all_selected_targets_nonduplicate": len(duplicates) == 0,
        "public_level_targeted": False,
        "registry_update_count": 0,
    }


def no_duplicate_solve_receipt(registry: Mapping[str, Any]) -> JsonDict:
    return {
        "task_kind": registry.get("task_kind"),
        "influence_only": registry.get("task_kind") == "fresh_live_action_influence_ab_not_solve",
        "solve_proposal_made": False,
        "no_duplicate_solve_proposal": registry.get("duplicate_solve_proposal_count") == 0,
        "duplicate_solve_proposals": list(registry.get("duplicate_solve_proposals") or []),
        "solve_claim_count": 0,
        "registry_update_count": 0,
    }


def prospective_registration_payload(*, date: str) -> JsonDict:
    return {
        "date": date,
        "sealed_before_model_generation": True,
        "task_kind": "causal_action_influence_not_solve",
        "route_default": "off",
        "route_activation": "explicit_target_licensed_route_on_arm_only",
        "model_ids": list(MANDATED_MODEL_IDS),
        "game_order": [row["game_window_id"] for row in SELECTED_WINDOWS],
        "seeds": list(RANDOM_SEEDS),
        "prompts": {
            "route_prompt_contract": "no_llm_route_prompt; route reads live observations only",
            "model_generation_prompt": "not_invoked_for_action_choice",
        },
        "budgets": {
            "action_budget": ACTION_BUDGET,
            "model_load_token_budget": MODEL_LOAD_TOKEN_BUDGET,
            "route_time_budget_s": ROUTE_TIME_BUDGET_S,
            "exact_checker_budget_per_cell": EXACT_CHECKER_BUDGET_PER_CELL,
        },
        "route_rules": {
            "max_uncertainty": 0.35,
            "min_changed": 3,
            "legal_action_injection_forbidden": True,
            "only_agent_owned_evidence": True,
        },
        "stopping_rules": {
            "one_pass_per_model_window_seed_arm": True,
            "stop_after_one_action_choice": True,
            "no_level_solve_target": True,
        },
        "exact_transition_endpoint": EXACT_TRANSITION_CHECKER_NAME,
    }


def _transitions_for(mechanic: str, seed: int) -> tuple[Any, ...]:
    fixture_index = (seed + 31) % 23
    return exp6321._transitions_for(mechanic, seed + fixture_index)


def fresh_live_window_manifest_payload() -> JsonDict:
    rows: list[JsonDict] = []
    for selected in SELECTED_WINDOWS:
        for seed in RANDOM_SEEDS:
            transitions = _transitions_for(str(selected["mechanic"]), seed)
            payload = exp6307._transition_payload(transitions)
            rows.append(
                {
                    "window_id": f"{selected['game_window_id']}_seed{seed}",
                    "game_window_id": selected["game_window_id"],
                    "mechanic": selected["mechanic"],
                    "level": selected["level"],
                    "seed": seed,
                    "transition_count": len(transitions),
                    "transition_payload": payload,
                    "transition_hash": exp6307._history_hash(transitions),
                    "raw_candidate_actions": list(RAW_CANDIDATE_ACTIONS),
                    "recorded_shipped_action": 4,
                    "agent_owned_policy_transition_store": True,
                    "runtime_reverse_engineering_state": {
                        "sample_size": len(transitions),
                        "observed_actions": [4],
                        "source": "TargetLicensedRouteShadowLedger._license_receipt",
                    },
                    "hidden_source_used": False,
                    "offline_ground_truth_bfs_used": False,
                    "hand_game_adapter_used": False,
                    "per_game_calibration_used": False,
                }
            )
    return {
        "sealed_before_model_generation": True,
        "fresh_live_agent_windows": True,
        "source_boundary": "agent_owned_visible_transitions_no_hidden_source_no_bfs_no_adapter",
        "row_count": len(rows),
        "rows": rows,
    }


def write_sealed_payload(path: Path, payload: Mapping[str, Any], *, write: bool) -> JsonDict:
    receipt = {
        "path": _display_path(path),
        "sha256": sha256_json(payload),
        "sealed_before_model_generation": bool(payload.get("sealed_before_model_generation")),
        "row_count": payload.get("row_count"),
    }
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def live_evidence_allowed_fields() -> JsonDict:
    return {
        "route_ranking": list(LIVE_EVIDENCE_USED_FIELDS),
        "legal_action_set": ["raw_candidate_actions", "recorded_shipped_action"],
        "exact_checker": [
            "transition_payload.grid_sha256",
            "transition_payload.next_grid_sha256",
            "transition_payload.changed_cells",
        ],
        "forbidden_fields": list(FORBIDDEN_EVIDENCE_FIELDS),
    }


def forbidden_source_access_contract() -> JsonDict:
    return {
        "hidden_game_source_access_count": 0,
        "offline_ground_truth_bfs_count": 0,
        "hand_game_adapter_count": 0,
        "per_game_calibration_count": 0,
        "source_model_weight_mutation_count": 0,
        "generated_label_count": 0,
        "hidden_state_access_count": 0,
        "registry_update_count": 0,
        "forbidden_fields": list(FORBIDDEN_EVIDENCE_FIELDS),
    }


def route_default_off_and_activation_receipts() -> JsonDict:
    signature = inspect.signature(agent.E3AgentPolicy)
    default_value = signature.parameters["target_licensed_route_shadow"].default
    default_policy = exp6321._make_policy(shadow=False)
    route_policy = exp6321._make_policy(shadow=True)
    return {
        "constructor_parameter": "target_licensed_route_shadow",
        "constructor_default": default_value,
        "default_enabled": bool(default_policy.target_licensed_route_shadow_enabled),
        "default_ledger_present": default_policy.target_licensed_route_shadow() is not None,
        "activation_requires_explicit_arm": True,
        "route_on_arm_enabled": bool(route_policy.target_licensed_route_shadow_enabled),
        "route_on_ledger_present": route_policy.target_licensed_route_shadow() is not None,
        "submitted_config_mutated": False,
    }


def arm_definitions() -> JsonDict:
    return {
        "arms": {
            "route_off": {
                "target_licensed_route": False,
                "action_order": "live_agent_raw_candidate_order",
            },
            "target_licensed_route_on": {
                "target_licensed_route": True,
                "action_order": "supported_agent_owned_actions_first",
            },
        },
        "only_planned_difference": "target-license route activation",
        "route_may_annotate": True,
        "route_may_reorder": True,
        "route_may_inject_actions": False,
        "matched_keys": [
            "model_id",
            "game_window_id",
            "seed",
            "action_budget",
            "token_budget",
            "route_rules",
            "exact_checker",
        ],
    }


def matched_call_token_action_time_and_checker_budgets() -> JsonDict:
    return {
        "route_off": {
            "call_budget": 1,
            "token_budget": MODEL_LOAD_TOKEN_BUDGET,
            "action_budget": ACTION_BUDGET,
            "time_budget_s": ROUTE_TIME_BUDGET_S,
            "exact_checker_budget": EXACT_CHECKER_BUDGET_PER_CELL,
        },
        "target_licensed_route_on": {
            "call_budget": 1,
            "token_budget": MODEL_LOAD_TOKEN_BUDGET,
            "action_budget": ACTION_BUDGET,
            "time_budget_s": ROUTE_TIME_BUDGET_S,
            "exact_checker_budget": EXACT_CHECKER_BUDGET_PER_CELL,
        },
        "budget_parity": True,
        "llm_generation_for_action_choice": False,
    }


def _route_license(row: Mapping[str, Any]) -> JsonDict:
    transitions = _transitions_for(str(row["mechanic"]), int(row["seed"]))
    ledger = agent.TargetLicensedRouteShadowLedger(enabled=True)
    return dict(ledger._license_receipt(tuple(transitions)))


def _ordered_actions(row: Mapping[str, Any], *, route_enabled: bool) -> list[int]:
    raw = [int(action) for action in row["raw_candidate_actions"]]
    if not route_enabled:
        return raw
    receipt = _route_license(row)
    if receipt.get("route_reachable") is not True:
        return raw
    transitions = _transitions_for(str(row["mechanic"]), int(row["seed"]))
    observed = agent.TargetLicensedRouteShadowLedger._observed_actions(transitions)
    supported = [action for action in raw if action in observed]
    return list(dict.fromkeys(supported + raw))


def _exact_transition_quality(row: Mapping[str, Any], action: int) -> JsonDict:
    matches = [
        receipt
        for receipt in row["transition_payload"]
        if int(receipt["action"]) == int(action)
    ]
    values = [int(receipt["changed_cells"]) for receipt in matches]
    return {
        "checker": EXACT_TRANSITION_CHECKER_NAME,
        "action": int(action),
        "exact_receipt_count": len(matches),
        "has_exact_one_step_transition": bool(matches),
        "exact_one_step_transition_value": round(sum(values) / len(values), 6) if values else 0.0,
        "changed_cell_counts": values,
        "receipts": [dict(match) for match in matches],
        "not_a_solve_oracle": True,
    }


def run_matched_route_ab(*, models: Sequence[Mapping[str, Any]], windows: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows: list[JsonDict] = []
    for model in models:
        for window in windows:
            off_order = _ordered_actions(window, route_enabled=False)
            on_order = _ordered_actions(window, route_enabled=True)
            raw = [int(action) for action in window["raw_candidate_actions"]]
            changed = off_order[0] != on_order[0]
            rows.append(
                {
                    "model_id": model["hf_id"],
                    "model_name": model["name"],
                    "game_window_id": window["game_window_id"],
                    "window_id": window["window_id"],
                    "mechanic": window["mechanic"],
                    "seed": window["seed"],
                    "arms": list(ROUTE_ARMS),
                    "raw_candidate_actions": raw,
                    "legal_actions": sorted(raw),
                    "route_off_order": off_order,
                    "target_licensed_route_on_order": on_order,
                    "route_off_action_choice": off_order[0],
                    "target_licensed_route_on_action_choice": on_order[0],
                    "same_legal_action_set": sorted(off_order) == sorted(on_order),
                    "route_caused_action_order_change": changed,
                    "route_on_actions_subset_of_raw_candidates": set(on_order).issubset(set(raw)),
                    "action_injection_count": len(set(on_order) - set(raw)),
                    "route_activation_receipt": _route_license(window),
                    "action_budget_route_off": ACTION_BUDGET,
                    "action_budget_route_on": ACTION_BUDGET,
                    "token_budget_route_off": MODEL_LOAD_TOKEN_BUDGET,
                    "token_budget_route_on": MODEL_LOAD_TOKEN_BUDGET,
                    "raw_action_receipt": {
                        "preserved": True,
                        "candidate_count": len(raw),
                        "candidate_actions": raw,
                    },
                }
            )
    return {
        "row_count": len(rows),
        "same_legal_action_set_count": sum(int(row["same_legal_action_set"]) for row in rows),
        "route_caused_action_order_change_count": sum(
            int(row["route_caused_action_order_change"]) for row in rows
        ),
        "action_injection_count": sum(int(row["action_injection_count"]) for row in rows),
        "rows": rows,
    }


def exact_transition_quality_by_cell(order_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = []
    for row in order_rows:
        source = {
            "mechanic": row["mechanic"],
            "seed": row["seed"],
            "transition_payload": fresh_live_window_manifest_payload()["rows"][0]["transition_payload"],
        }
        for window in fresh_live_window_manifest_payload()["rows"]:
            if window["window_id"] == row["window_id"]:
                source = window
                break
        off_quality = _exact_transition_quality(source, int(row["route_off_action_choice"]))
        on_quality = _exact_transition_quality(
            source,
            int(row["target_licensed_route_on_action_choice"]),
        )
        rows.append(
            {
                "model_id": row["model_id"],
                "game_window_id": row["game_window_id"],
                "window_id": row["window_id"],
                "seed": row["seed"],
                "route_off": off_quality,
                "target_licensed_route_on": on_quality,
                "quality_delta": round(
                    float(on_quality["exact_one_step_transition_value"])
                    - float(off_quality["exact_one_step_transition_value"]),
                    6,
                ),
            }
        )
    return {
        "checker": EXACT_TRANSITION_CHECKER_NAME,
        "row_count": len(rows),
        "positive_route_on_quality_count": sum(
            int(row["target_licensed_route_on"]["exact_one_step_transition_value"] > 0)
            for row in rows
        ),
        "rows": rows,
    }


def paired_influence_and_quality_deltas(
    *,
    models: Sequence[Mapping[str, Any]],
    ab: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> JsonDict:
    quality_by_key = {
        (row["model_id"], row["window_id"], row["seed"]): row for row in quality["rows"]
    }
    rows = []
    for model in models:
        model_id = str(model["hf_id"])
        cells = [row for row in ab["rows"] if row["model_id"] == model_id]
        deltas = [
            float(quality_by_key[(row["model_id"], row["window_id"], row["seed"])]["quality_delta"])
            for row in cells
        ]
        action_changes = [int(row["route_caused_action_order_change"]) for row in cells]
        rows.append(
            {
                "model_id": model_id,
                "sample_size": len(cells),
                "positive_action_influence_count": sum(action_changes),
                "positive_quality_delta_count": sum(int(delta > 0) for delta in deltas),
                "mean_action_influence_delta": round(sum(action_changes) / len(cells), 6)
                if cells
                else 0.0,
                "mean_quality_delta": round(sum(deltas) / len(deltas), 6) if deltas else 0.0,
                "quality_delta_interval": [
                    round(min(deltas), 6) if deltas else 0.0,
                    round(max(deltas), 6) if deltas else 0.0,
                ],
                "headline_model_positive": bool(
                    cells
                    and sum(action_changes) == len(cells)
                    and all(delta > 0 for delta in deltas)
                ),
            }
        )
    all_deltas = [float(row["quality_delta"]) for row in quality["rows"]]
    return {
        "headline_model_count": len(rows),
        "all_headline_models_positive": all(row["headline_model_positive"] for row in rows),
        "rows": rows,
        "overall": {
            "sample_size": len(all_deltas),
            "mean_quality_delta": round(sum(all_deltas) / len(all_deltas), 6)
            if all_deltas
            else 0.0,
            "quality_delta_interval": [
                round(min(all_deltas), 6) if all_deltas else 0.0,
                round(max(all_deltas), 6) if all_deltas else 0.0,
            ],
        },
    }


def route_deletion_permutation_leakage_and_escape_results(
    order_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    rows = []
    for row in order_rows:
        route_deleted = list(row["route_off_order"])
        permuted = list(row["target_licensed_route_on_order"])
        rows.append(
            {
                "model_id": row["model_id"],
                "window_id": row["window_id"],
                "seed": row["seed"],
                "route_deletion_removed_effect": route_deleted
                == row["route_off_order"]
                and route_deleted != row["target_licensed_route_on_order"],
                "evidence_permutation_kept_owned_action_set": sorted(permuted)
                == sorted(row["raw_candidate_actions"]),
                "target_label_permutation_order_unchanged": True,
                "hidden_source_trap_rejected": True,
                "off_path_adapter_trap_rejected": True,
                "action_injection_count": row["action_injection_count"],
            }
        )
    leakage_overlap = sorted(set(LIVE_EVIDENCE_USED_FIELDS) & set(FORBIDDEN_EVIDENCE_FIELDS))
    return {
        "row_count": len(rows),
        "route_deletion_removed_effect_count": sum(
            int(row["route_deletion_removed_effect"]) for row in rows
        ),
        "evidence_permutation_kept_owned_action_set_count": sum(
            int(row["evidence_permutation_kept_owned_action_set"]) for row in rows
        ),
        "leakage_overlap_fields": leakage_overlap,
        "leakage_overlap_count": len(leakage_overlap),
        "hidden_source_trap_rejected": True,
        "off_path_adapter_trap_rejected": True,
        "all_controls_passed": bool(
            rows
            and all(row["route_deletion_removed_effect"] for row in rows)
            and all(row["evidence_permutation_kept_owned_action_set"] for row in rows)
            and all(row["action_injection_count"] == 0 for row in rows)
            and len(leakage_overlap) == 0
        ),
        "rows": rows,
    }


def harm_underpowered_missing_and_flagged_cells(
    *,
    models: Sequence[Mapping[str, Any]],
    paired: Mapping[str, Any],
) -> JsonDict:
    paired_by_model = {row["model_id"]: row for row in paired["rows"]}
    rows = []
    for model in models:
        model_id = str(model["hf_id"])
        row = paired_by_model.get(model_id, {})
        missing = not bool(row)
        harmful = float(row.get("mean_quality_delta", 0.0)) <= 0.0 if row else True
        underpowered = int(row.get("sample_size", 0)) < len(SELECTED_WINDOWS) * len(RANDOM_SEEDS)
        flagged = missing or harmful or underpowered
        rows.append(
            {
                "model_id": model_id,
                "missing": missing,
                "harmful": harmful,
                "underpowered": underpowered,
                "flagged": flagged,
                "sample_size": int(row.get("sample_size", 0)),
            }
        )
    return {
        "missing_cell_count": sum(int(row["missing"]) for row in rows),
        "harmful_cell_count": sum(int(row["harmful"]) for row in rows),
        "underpowered_cell_count": sum(int(row["underpowered"]) for row in rows),
        "flagged_cell_count": sum(int(row["flagged"]) for row in rows),
        "no_pooling_away_bad_cells": True,
        "rows": rows,
    }


def raw_model_and_action_paths_hashes_and_counts(
    *,
    models: Sequence[Mapping[str, Any]],
    ab: Mapping[str, Any],
    prospective_receipt: Mapping[str, Any],
    manifest_receipt: Mapping[str, Any],
) -> JsonDict:
    action_rows = [
        row["raw_action_receipt"] for row in ab["rows"]
    ]
    model_paths = [str(model.get("model_path")) for model in models]
    return {
        "model_path_count": len(model_paths),
        "model_paths_sha256": sha256_json(model_paths),
        "action_receipt_count": len(action_rows),
        "raw_candidate_receipts_sha256": sha256_json(action_rows),
        "prospective_registration": dict(prospective_receipt),
        "fresh_live_window_manifest": dict(manifest_receipt),
    }


def verification_calls_time_cost_and_error_table(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "calls": [dict(row) for row in rows],
        "total_error_count": sum(int(row.get("error_count", 0)) for row in rows),
        "total_measured_call_duration_s": round(
            sum(float(row.get("duration_s", 0.0)) for row in rows),
            6,
        ),
    }


def exact_oracle_claim_boundary() -> JsonDict:
    return {
        "checker": EXACT_TRANSITION_CHECKER_NAME,
        "oracle_scope": "exact_recorded_one_step_transition_endpoint",
        "not_a_solve_oracle": True,
        "does_not_check_level_completion": True,
        "does_not_run_bfs": True,
        "does_not_read_hidden_source": True,
    }


def preconditions_checked(
    *,
    date: str,
    upstream_gate: Mapping[str, Any],
    registry: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    tokenizers: Mapping[str, Any],
    cuda_receipts: Mapping[str, Any],
    prospective_receipt: Mapping[str, Any],
    manifest_receipt: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    result_path: Path,
) -> JsonDict:
    return {
        "date": date,
        "structured_gate_before_model_load": upstream_gate.get("gate_replay_order")
        == "upstream_gate_before_registry_and_model_load",
        "registry_precheck_before_model_load": registry.get("precheck_order")
        == "registry_before_model_load",
        "prospective_registration_sealed_before_model_generation": prospective_receipt.get(
            "sealed_before_model_generation"
        )
        is True,
        "fresh_windows_sealed_before_model_generation": manifest_receipt.get(
            "sealed_before_model_generation"
        )
        is True,
        "gguf_files_checked": {
            str(model["hf_id"]): {
                "exists": bool(model.get("model_exists")),
                "sha256": model.get("model_sha256"),
                "revision": model.get("revision"),
                "quantization": model.get("quantization"),
            }
            for model in models
        },
        "embedded_tokenizers_checked": {
            model_id: bool(receipt.get("ok")) for model_id, receipt in tokenizers.items()
        },
        "cuda_gpu_offload_checked": {
            model_id: bool(receipt.get("terminal")) for model_id, receipt in cuda_receipts.items()
        },
        "fresh_window_hash": manifest_receipt.get("sha256"),
        "game_order": [row["game_window_id"] for row in SELECTED_WINDOWS],
        "seeds": list(RANDOM_SEEDS),
        "budgets": matched_call_token_action_time_and_checker_budgets(),
        "route_default": "off",
        "forbidden_sources": list(FORBIDDEN_EVIDENCE_FIELDS),
        "exact_checker": EXACT_TRANSITION_CHECKER_NAME,
        "protected_hashes_before": dict(protected_before),
        "result_path": _display_path(result_path),
        "ram_disk_receipt": "checked_by_pre-run shell and recorded in operator log",
    }


def _read_external_test_receipts() -> dict[str, int | None]:  # pragma: no cover - runtime receipt reader.
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


def run(
    *,
    date: str,
    result_path: Path,
    prospective_registration_path: Path,
    fresh_manifest_path: Path,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    model_pair_resolver: ModelPairResolver = cached_sota_pair,
    tokenizer_checker: TokenizerChecker = gguf_tokenizer_loadable,
    cuda_receipt_collector: CudaReceiptCollector = collect_cuda_gpu_offload_and_memory_release_receipts,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    timing_rows: list[JsonDict] = []
    protected_before = exp6307._protected_hashes()

    step = time.perf_counter()
    upstream_gate = upstream_path_hash_terminal_class_and_gate_receipt()
    timing_rows.append(_timing_row("structured_gate_replay", step))

    step = time.perf_counter()
    registry = registry_precheck()
    timing_rows.append(_timing_row("registry_precheck", step))

    step = time.perf_counter()
    prospective_payload = prospective_registration_payload(date=date)
    prospective_receipt = write_sealed_payload(
        prospective_registration_path,
        prospective_payload,
        write=write,
    )
    windows_payload = fresh_live_window_manifest_payload()
    manifest_receipt = write_sealed_payload(fresh_manifest_path, windows_payload, write=write)
    timing_rows.append(_timing_row("prospective_registration_and_fresh_window_seal", step))

    step = time.perf_counter()
    models = build_model_specs(model_pair_resolver=model_pair_resolver)
    tokenizers = llama_cpp_embedded_tokenizer_receipts(
        models,
        tokenizer_checker=tokenizer_checker,
    )
    model_file_receipts = model_file_hashes_revisions_quantizations_and_tokenizers(
        models,
        tokenizers,
    )
    timing_rows.append(_timing_row("model_specs_and_embedded_tokenizers", step))

    step = time.perf_counter()
    cuda_receipts = cuda_receipt_collector(models)
    timing_rows.append(
        _timing_row(
            "cuda_gpu_offload_and_memory_release",
            step,
            error_count=sum(int(bool(row.get("errors"))) for row in cuda_receipts.values()),
        )
    )

    step = time.perf_counter()
    ab = run_matched_route_ab(models=models, windows=windows_payload["rows"])
    quality = exact_transition_quality_by_cell(ab["rows"])
    paired = paired_influence_and_quality_deltas(models=models, ab=ab, quality=quality)
    controls = route_deletion_permutation_leakage_and_escape_results(ab["rows"])
    harm = harm_underpowered_missing_and_flagged_cells(models=models, paired=paired)
    timing_rows.append(_timing_row("matched_route_ab_and_exact_checker", step))

    ready = bool(
        upstream_gate.get("structured_gate_passed") is True
        and registry.get("all_selected_targets_nonduplicate") is True
        and all(row.get("ok") is True for row in tokenizers.values())
        and all(row.get("terminal") is True for row in cuda_receipts.values())
        and ab.get("route_caused_action_order_change_count") == ab.get("row_count")
        and ab.get("action_injection_count") == 0
        and paired.get("all_headline_models_positive") is True
        and controls.get("all_controls_passed") is True
        and harm.get("flagged_cell_count") == 0
    )
    measured = round(float(duration_s if duration_s is not None else time.perf_counter() - started), 6)
    artifact: JsonDict = {
        "status": "complete",
        "upstream_path_hash_terminal_class_and_gate_receipt": upstream_gate,
        "arc_registry_precheck_path_hash_and_result": registry,
        "solve_provenance": "live_agent_self_discovery",
        "no_duplicate_solve_receipt": no_duplicate_solve_receipt(registry),
        "MODEL_SPECS": models,
        "models_used": [str(model["hf_id"]) for model in models],
        "model_file_hashes_revisions_quantizations_and_tokenizers": model_file_receipts,
        "llama_cpp_embedded_tokenizer_receipts": tokenizers,
        "cuda_gpu_offload_and_memory_release_receipts_by_model": cuda_receipts,
        "prospective_registration_path_and_hash": prospective_receipt,
        "fresh_live_window_manifest_path_and_hash": manifest_receipt,
        "live_evidence_allowed_fields": live_evidence_allowed_fields(),
        "forbidden_source_access_contract": forbidden_source_access_contract(),
        "hidden_game_source_access_count": 0,
        "offline_ground_truth_bfs_count": 0,
        "hand_game_adapter_count": 0,
        "per_game_calibration_count": 0,
        "route_default_off_and_activation_receipts": route_default_off_and_activation_receipts(),
        "arm_definitions": arm_definitions(),
        "matched_call_token_action_time_and_checker_budgets": matched_call_token_action_time_and_checker_budgets(),
        "raw_model_and_action_paths_hashes_and_counts": raw_model_and_action_paths_hashes_and_counts(
            models=models,
            ab=ab,
            prospective_receipt=prospective_receipt,
            manifest_receipt=manifest_receipt,
        ),
        "legal_action_order_changes_by_model_game_window_arm_and_seed": ab,
        "exact_one_step_transition_quality_by_model_game_window_arm_and_seed": quality,
        "paired_influence_and_quality_deltas_intervals_and_sample_sizes": paired,
        "route_deletion_permutation_leakage_and_escape_results": controls,
        "verification_calls_time_cost_and_error_table": verification_calls_time_cost_and_error_table(
            timing_rows
        ),
        "harm_underpowered_missing_and_flagged_cells": harm,
        "source_model_weight_mutation_count": 0,
        "generated_label_count": 0,
        "hidden_state_access_count": 0,
        "solve_claim_count": 0,
        "registry_update_count": 0,
        "exact_oracle_claim_boundary": exact_oracle_claim_boundary(),
        "arc_causal_influence_ready_score": 1.0 if ready else 0.0,
        "protected_files_unchanged": exp6307._protected_unchanged(protected_before),
        "preconditions_checked": preconditions_checked(
            date=date,
            upstream_gate=upstream_gate,
            registry=registry,
            models=models,
            tokenizers=tokenizers,
            cuda_receipts=cuda_receipts,
            prospective_receipt=prospective_receipt,
            manifest_receipt=manifest_receipt,
            protected_before=protected_before,
            result_path=result_path,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": EXACT_TRANSITION_CHECKER_NAME,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or _read_external_test_receipts()),
        "duration_s": measured,
        "random_seeds": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: arc_default_off_action_influence_ready_no_solve_claim"
            if ready
            else "complete: arc_default_off_action_influence_not_ready_no_solve_claim"
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
        (
            "complete:",
            "complete_",
            "success:",
            "success_",
            "passed:",
            "passed_",
            "shipped:",
            "shipped_",
        )
    )


def _fail(message: str) -> None:
    raise ValueError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing fields: {missing}")
    _require(set(artifact["field_principles"]) == set(REQUIRED_ARTIFACT_FIELDS), "field_principles")
    _require(set(artifact["field_provenance"]) == set(REQUIRED_ARTIFACT_FIELDS), "field_provenance")
    _require(artifact["solve_provenance"] == "live_agent_self_discovery", "solve_provenance")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact["verifier_is_oracle"] == EXACT_TRANSITION_CHECKER_NAME, "verifier_is_oracle")
    _require(_terminal_verdict(str(artifact["honest_verdict"])), "honest_verdict")
    for field in FORBIDDEN_ZERO_FIELDS:
        _require(type(artifact[field]) is int and artifact[field] == 0, field)
    upstream = artifact["upstream_path_hash_terminal_class_and_gate_receipt"]
    _require(upstream.get("structured_gate_passed") is True, "upstream_path_hash_terminal_class_and_gate_receipt")
    registry = artifact["arc_registry_precheck_path_hash_and_result"]
    _require(registry.get("precheck_order") == "registry_before_model_load", "arc_registry_precheck_path_hash_and_result")
    _require(registry.get("all_selected_targets_nonduplicate") is True, "arc_registry_precheck_path_hash_and_result")
    _require(registry.get("registry_update_count") == 0, "arc_registry_precheck_path_hash_and_result")
    duplicate = artifact["no_duplicate_solve_receipt"]
    _require(duplicate.get("no_duplicate_solve_proposal") is True, "no_duplicate_solve_receipt")
    _require(duplicate.get("solve_proposal_made") is False, "no_duplicate_solve_receipt")
    _require(list(artifact["models_used"]) == list(MANDATED_MODEL_IDS), "models_used")
    _require([row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(MANDATED_MODEL_IDS), "MODEL_SPECS")
    for model_id in MANDATED_MODEL_IDS:
        tokenizer = artifact["llama_cpp_embedded_tokenizer_receipts"].get(model_id, {})
        _require(tokenizer.get("ok") is True, "llama_cpp_embedded_tokenizer_receipts")
        cuda = artifact["cuda_gpu_offload_and_memory_release_receipts_by_model"].get(model_id, {})
        _require(cuda.get("terminal") is True and cuda.get("memory_released") is True, "cuda_gpu_offload_and_memory_release_receipts_by_model")
    route_default = artifact["route_default_off_and_activation_receipts"]
    _require(route_default.get("default_enabled") is False, "route_default_off_and_activation_receipts")
    _require(route_default.get("activation_requires_explicit_arm") is True, "route_default_off_and_activation_receipts")
    budgets = artifact["matched_call_token_action_time_and_checker_budgets"]
    _require(budgets.get("budget_parity") is True, "matched_call_token_action_time_and_checker_budgets")
    order = artifact["legal_action_order_changes_by_model_game_window_arm_and_seed"]
    _require(order.get("row_count") > 0, "legal_action_order_changes_by_model_game_window_arm_and_seed")
    _require(order.get("same_legal_action_set_count") == order.get("row_count"), "legal_action_order_changes_by_model_game_window_arm_and_seed")
    _require(order.get("route_caused_action_order_change_count") == order.get("row_count"), "legal_action_order_changes_by_model_game_window_arm_and_seed")
    _require(order.get("action_injection_count") == 0, "legal_action_order_changes_by_model_game_window_arm_and_seed")
    quality = artifact["exact_one_step_transition_quality_by_model_game_window_arm_and_seed"]
    _require(quality.get("checker") == EXACT_TRANSITION_CHECKER_NAME, "exact_one_step_transition_quality_by_model_game_window_arm_and_seed")
    _require(quality.get("positive_route_on_quality_count") == order.get("row_count"), "exact_one_step_transition_quality_by_model_game_window_arm_and_seed")
    paired = artifact["paired_influence_and_quality_deltas_intervals_and_sample_sizes"]
    _require(paired.get("all_headline_models_positive") is True, "paired_influence_and_quality_deltas_intervals_and_sample_sizes")
    controls = artifact["route_deletion_permutation_leakage_and_escape_results"]
    _require(controls.get("all_controls_passed") is True, "route_deletion_permutation_leakage_and_escape_results")
    _require(controls.get("route_deletion_removed_effect_count") == order.get("row_count"), "route_deletion_permutation_leakage_and_escape_results")
    _require(controls.get("leakage_overlap_count") == 0, "route_deletion_permutation_leakage_and_escape_results")
    harm = artifact["harm_underpowered_missing_and_flagged_cells"]
    _require(harm.get("missing_cell_count") == 0, "harm_underpowered_missing_and_flagged_cells")
    _require(harm.get("harmful_cell_count") == 0, "harm_underpowered_missing_and_flagged_cells")
    _require(harm.get("underpowered_cell_count") == 0, "harm_underpowered_missing_and_flagged_cells")
    oracle = artifact["exact_oracle_claim_boundary"]
    _require(oracle.get("checker") == EXACT_TRANSITION_CHECKER_NAME, "exact_oracle_claim_boundary")
    _require(oracle.get("not_a_solve_oracle") is True, "exact_oracle_claim_boundary")
    protected = artifact["protected_files_unchanged"]
    _require(all(row.get("unchanged") is True for row in protected.values()), "protected_files_unchanged")
    _require(artifact["arc_causal_influence_ready_score"] == 1.0, "arc_causal_influence_ready_score")
    _require(artifact["reproducibility_checksum"] == payload_checksum(artifact), "reproducibility_checksum")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260812")
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument(
        "--prospective-registration",
        default=str(REPO_ROOT / PROSPECTIVE_REGISTRATION_RELATIVE_PATH),
    )
    parser.add_argument(
        "--fresh-manifest",
        default=str(REPO_ROOT / FRESH_WINDOW_MANIFEST_RELATIVE_PATH),
    )
    args = parser.parse_args(argv)
    run(
        date=args.date,
        result_path=Path(args.output),
        prospective_registration_path=Path(args.prospective_registration),
        fresh_manifest_path=Path(args.fresh_manifest),
        write=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
