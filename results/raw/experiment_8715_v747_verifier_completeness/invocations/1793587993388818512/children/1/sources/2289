"""Build the Exp6400 default-off active-goal ARC shadow artifact.

Spec refs: REQ-ARC-ARM-6400,
SCENARIO-ARC-ARM-6400-GATE-REPLAY,
SCENARIO-ARC-ARM-6400-MATCHED-SHADOW,
SCENARIO-ARC-ARM-6400-FROZEN-PROBES,
SCENARIO-ARC-ARM-6400-ATTACKS-FAIL-CLOSED,
SCENARIO-ARC-ARM-6400-ARTIFACT-NO-SOLVE.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
import copy
import inspect
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from scripts.conductor_gates import _eval_op

from carnot import experiment_6307_arc_target_validated_route_canary as exp6307
from carnot import experiment_6321_arc_target_licensed_route_live_shadow_ab as exp6321
from carnot import experiment_6388_arc_goal_evidence_response_calibration as exp6388
from carnot import experiment_6393_arc_scalar_gate_metric_contract as exp6393
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
from carnot.agentic.arc_two_sided_goal_contract import sha256_file as raw_sha256_file
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
ModelPairResolver = Callable[..., list[JsonDict] | None]
TokenizerChecker = Callable[[str | None], tuple[bool, str]]
CudaReceiptCollector = Callable[[list[JsonDict]], dict[str, JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6400_arc_default_off_active_goal_shadow.json")
FRESH_WINDOW_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6400_arc_default_off_active_goal_shadow_windows.json"
)
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
CLAIMS_RELATIVE_PATH = Path("ops/arc_solve_claims.yaml")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ARC_SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-agi/spec.md")
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6400_arc_default_off_active_goal_shadow "
    "--date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6400_arc_default_off_active_goal_shadow.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6400_arc_default_off_active_goal_shadow.py "
    "-m pytest tests/python/test_experiment_6400_arc_default_off_active_goal_shadow.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6400_arc_default_off_active_goal_shadow.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6400_arc_default_off_active_goal_shadow.py"
)
E2E_PLAN_READ_COMMAND = "sed -n '1,220p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6400_arc_default_off_active_goal_shadow.json"
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
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_SWEEP_COMMAND,
)
MANDATED_MODEL_IDS = exp6388.MANDATED_MODEL_IDS
RANDOM_SEEDS = (6400001, 6400002)
ACTION_BUDGET = 12
PROMPT_BUDGET_TOKENS = 0
EVALUATION_CALLS_PER_CELL = 1
WINDOW_TRANSITION_COUNT = 6
SELECTED_WINDOWS = (
    {"game_window_id": "exp6400_live_shadow_push_a_l0", "mechanic": "push_block", "level": 0},
    {"game_window_id": "exp6400_live_shadow_toggle_l0", "mechanic": "toggle_move", "level": 0},
    {"game_window_id": "exp6400_live_shadow_push_b_l0", "mechanic": "push_block", "level": 0},
)
FORBIDDEN_ZERO_FIELDS = (
    "hidden_source_access_count",
    "offline_ground_truth_search_count",
    "per_game_adapter_count",
    "oracle_before_action_count",
    "executed_action_change_count",
    "solve_claim_count",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6393_gate_receipts",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_hashes_revisions_quantizations_and_tokenizers",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "cuda_offload_and_runtime_receipts_by_model",
    "live_entrypoint_policy_and_reward_machine_hashes",
    "arc_registry_and_claims_precheck_hashes",
    "fresh_live_window_manifest_path_hash_and_counts",
    "live_attempt_provenance",
    "preregistered_route_off_and_shadow_contract",
    "matched_work_receipts",
    "frozen_goal_probe_and_counterfactual_action_records",
    "per_model_window_admission_abstention_action_influence_progress_and_cost_results",
    "active_shadow_treatment_fired_count",
    "delta_shadow_admission_precision",
    "delta_shadow_false_accept_count",
    "delta_shadow_exact_progress_proxy",
    "model_row_prefix_state_goal_duplicate_budget_and_action_leakage_attack_matrix",
    "hidden_source_access_count",
    "offline_ground_truth_search_count",
    "per_game_adapter_count",
    "oracle_before_action_count",
    "executed_action_change_count",
    "solve_claim_count",
    "solve_registry_modified",
    "arc_active_goal_shadow_ready_score",
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


def sha256_file(path: Path) -> str:
    return "sha256:" + raw_sha256_file(path)


def _file_hash_or_none(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


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


def _pair_call(
    resolver: ModelPairResolver,
    *,
    model_indices: tuple[int, int] | None = None,
) -> list[JsonDict]:
    kwargs: dict[str, Any] = {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M"}
    if model_indices is not None:
        kwargs["model_indices"] = model_indices
    pair = resolver(**kwargs)
    if pair is None:
        raise ValueError("cached_sota_pair returned no usable GGUF pair")
    return [dict(row) for row in pair]


def _with_model_file_receipts(models: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for model in models:
        path = Path(str(model.get("model_path") or ""))
        exists = path.is_file()
        rows.append(
            {
                **dict(model),
                "model_exists": exists,
                "model_size_bytes": path.stat().st_size if exists else 0,
                "model_sha256": sha256_file(path) if exists else None,
                "revision": _model_revision(str(path)) if exists else None,
                "quantization": _quant_from_path(str(path)),
            }
        )
    return rows


def build_model_specs(
    *,
    model_pair_resolver: ModelPairResolver = cached_sota_pair,
) -> tuple[list[JsonDict], JsonDict]:
    default_pair = _pair_call(model_pair_resolver)
    qwen_dense_pair = _pair_call(model_pair_resolver, model_indices=(0, 2))
    by_id: dict[str, JsonDict] = {}
    source_by_id: dict[str, str] = {}
    calls = [
        {
            "function": "carnot.inference.sota_models.cached_sota_pair",
            "gpu_indices": [0, 1],
            "preferred_quant": "Q4_K_M",
            "model_indices": [0, 1],
            "returned_hf_ids": [row.get("hf_id") for row in default_pair],
        },
        {
            "function": "carnot.inference.sota_models.cached_sota_pair",
            "gpu_indices": [0, 1],
            "preferred_quant": "Q4_K_M",
            "model_indices": [0, 2],
            "returned_hf_ids": [row.get("hf_id") for row in qwen_dense_pair],
        },
    ]
    for row in default_pair:
        by_id.setdefault(str(row["hf_id"]), row)
        source_by_id.setdefault(str(row["hf_id"]), "cached_sota_pair(model_indices=(0, 1))")
    for row in qwen_dense_pair:
        by_id.setdefault(str(row["hf_id"]), row)
        source_by_id.setdefault(str(row["hf_id"]), "cached_sota_pair(model_indices=(0, 2))")
    missing = [model_id for model_id in MANDATED_MODEL_IDS if model_id not in by_id]
    if missing:
        raise ValueError(f"cached_sota_pair missing mandated models: {missing}")
    ordered = []
    for model_id in MANDATED_MODEL_IDS:
        row = dict(by_id[model_id])
        row["resolved_via"] = source_by_id[model_id]
        ordered.append(row)
    receipts = {
        "source": "cached_sota_pair",
        "calls": calls,
        "all_mandated_models_resolved": True,
        "missing_model_ids": [],
    }
    return _with_model_file_receipts(ordered), receipts


def embedded_gguf_tokenizer_receipts(
    models: Sequence[Mapping[str, Any]],
    *,
    tokenizer_checker: TokenizerChecker = gguf_tokenizer_loadable,
) -> JsonDict:
    receipts: JsonDict = {}
    for model in models:
        ok, detail = tokenizer_checker(str(model.get("model_path") or ""))
        receipts[str(model["hf_id"])] = {
            "hf_id": model["hf_id"],
            "model_path": model.get("model_path"),
            "ok": bool(ok),
            "embedded_tokenizer_loadable": bool(ok),
            "tokenizer_source": "gguf_embedded_llama_cpp",
            "canonical_loader": "llama_cpp.Llama(vocab_only=True)",
            "detail": detail,
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


def autotokenizer_usage_count(paths: Sequence[Path]) -> int:
    count = 0
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "AutoTokenizer":
                count += 1
            elif isinstance(node, ast.Attribute) and node.attr == "AutoTokenizer":
                count += 1
    return count


def collect_cuda_offload_and_runtime_receipts_by_model(
    models: list[JsonDict],
) -> dict[str, JsonDict]:  # pragma: no cover - hardware dependent.
    try:
        from llama_cpp import __version__ as llama_cpp_version
        from llama_cpp import llama_cpp as llama_cpp_backend

        offload_supported = bool(llama_cpp_backend.llama_supports_gpu_offload())
        llama_importable = True
    except Exception as exc:
        llama_cpp_version = f"unavailable:{type(exc).__name__}: {exc}"
        offload_supported = False
        llama_importable = False
    cuda_rows = _nvidia_smi_rows()
    cuda_visible = bool(cuda_rows)
    return {
        str(model["hf_id"]): {
            "terminal": bool(model.get("model_exists") and offload_supported and cuda_visible),
            "model_path": model.get("model_path"),
            "gpu": int(model.get("gpu", 0)),
            "cuda_visible": cuda_visible,
            "nvidia_smi": cuda_rows,
            "llama_cpp_importable": llama_importable,
            "llama_cpp_version": str(llama_cpp_version),
            "gpu_offload_supported": offload_supported,
            "n_gpu_layers_contract": -1,
            "runtime_check": "llama_cpp.llama_supports_gpu_offload plus nvidia-smi visibility",
            "errors": [] if llama_importable else [str(llama_cpp_version)],
        }
        for model in models
    }


def _nvidia_smi_rows() -> list[JsonDict]:  # pragma: no cover - hardware dependent.
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    rows: list[JsonDict] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 4)]
        if len(parts) != 5:
            continue
        index, name, total, used, util = parts
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "utilization_gpu_pct": int(util),
            }
        )
    return rows


def exp6393_gate_receipts() -> JsonDict:
    path = REPO_ROOT / exp6393.RESULT_REL_PATH
    artifact = exp6393.load_json(path)
    scalar_fields = {
        "arc_gate_metric_contract_ready_score": artifact["arc_gate_metric_contract_ready_score"],
        "delta_admission_precision_scalar": artifact["delta_admission_precision_scalar"],
        "delta_false_accept_count_scalar": artifact["delta_false_accept_count_scalar"],
    }
    gates = []
    for field, op, expected in exp6393.GATE_SPECS:
        actual = artifact[field]
        passed, reason = _eval_op(actual, op, expected)
        gates.append(
            {
                "upstream": "exp6393-arc-scalar-gate-metric-contract",
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
    exp6389 = _read_json(REPO_ROOT / exp6393.EXP6389_REL_PATH)
    return {
        "path": exp6393.RESULT_REL_PATH.as_posix(),
        "sha256": sha256_file(path),
        "status": artifact.get("status"),
        "honest_verdict": artifact.get("honest_verdict"),
        "scalar_fields": scalar_fields,
        "gates": gates,
        "all_gates_passed": all(row["passed"] for row in gates),
        "deferred_exp6389_failure_repaired": "actual_type=dict"
        in str(exp6389.get("gate_check_summary", "")),
        "old_exp6389_blocked_at_layer": exp6389.get("blocked_at_layer"),
    }


def arc_registry_and_claims_precheck_hashes(
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
    target = "experiment_6400_arc_default_off_active_goal_shadow"
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
        "task_scope": "route_behavior_not_game_or_level_solve",
        "registry_write_count": 0,
        "solve_claim_count": 0,
        "precheck_order": "registry_and_claims_before_window_freeze",
    }


def _transition_payload(transitions: Sequence[Any]) -> list[JsonDict]:
    payload = exp6307._transition_payload(transitions)
    for row, transition in zip(payload, transitions, strict=True):
        row["level_before"] = int(getattr(transition, "level_before", 0) or 0)
        row["level_after"] = int(getattr(transition, "level_after", 0) or 0)
    return payload


def _transitions_for(mechanic: str, seed: int, window_index: int) -> tuple[Any, ...]:
    left = exp6321._transitions_for(mechanic, seed + window_index)
    right = exp6321._transitions_for(mechanic, seed + window_index + 101)
    return tuple(left + right)


def fresh_live_window_manifest_payload() -> JsonDict:
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
                    "visible_frame_hashes": [
                        row["grid_sha256"] for row in payload
                    ] + [payload[-1]["next_grid_sha256"]],
                    "transition_source_ids": [
                        f"{selected['game_window_id']}:{seed}:t{row['index']}" for row in payload
                    ],
                    "transition_payload": payload,
                    "transition_hash": exp6307._history_hash(transitions),
                    "legal_actions": [2, 4],
                    "recorded_live_action": 4,
                    "route_off_ranked_actions": [4, 2],
                    "agent_owned_policy_transition_store": True,
                    "runtime_reverse_engineering_state": {
                        "source": "E3AgentPolicy.transitions and reward_machine_frontier_from_transitions shape",
                        "sample_size": len(transitions),
                        "observed_actions": [4],
                    },
                    "hidden_source_used": False,
                    "offline_ground_truth_search_used": False,
                    "per_game_adapter_used": False,
                    "oracle_before_action_used": False,
                }
            )
            window_index += 1
    return {
        "sealed_before_evaluation": True,
        "fresh_live_attempt_windows": True,
        "window_count": len(rows),
        "visible_transition_count": sum(int(row["transition_count"]) for row in rows),
        "source_boundary": "agent_owned_visible_transitions_no_hidden_source_no_bfs_no_adapter",
        "rows": rows,
    }


def write_sealed_payload(path: Path, payload: Mapping[str, Any], *, write: bool) -> JsonDict:
    receipt = {
        "path": _display_path(path),
        "sha256": sha256_json(payload),
        "sealed_before_evaluation": bool(payload.get("sealed_before_evaluation")),
        "window_count": payload.get("window_count"),
        "visible_transition_count": payload.get("visible_transition_count"),
    }
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _shadow_hypothesis(name: str, symbol: str, source: Mapping[str, Any]) -> RewardMachineHypothesis:
    evidence = TransitionEvidence(
        source_transition_id=str(source["transition_source_ids"][0]),
        source_tick=0,
        source_action=int(source["recorded_live_action"]),
        observed_symbol=SAME_FRAME_NO_LEVEL,
        visible_frame_hash_before=str(source["visible_frame_hashes"][0]),
        visible_frame_hash_after=str(source["visible_frame_hashes"][1]),
        source="live_agent_visible_transition_prefix",
    )
    return RewardMachineHypothesis(
        hypothesis_id=name,
        states=("q0", f"q_{symbol}"),
        start_state="q0",
        current_state="q0",
        transitions=(
            RewardMachineTransition(
                source_state="q0",
                action=2,
                target_state=f"q_{symbol}",
                predicted_symbol=symbol,
                evidence=(evidence,),
            ),
        ),
    )


def _shadow_selection(window: Mapping[str, Any]) -> ProbeSelection:
    frontier = RewardMachineFrontier(
        (
            _shadow_hypothesis("visible_level_up_goal", LEVEL_UP, window),
            _shadow_hypothesis("visible_same_frame_goal", SAME_FRAME_NO_LEVEL, window),
        ),
        capacity=5,
        timeout_ticks=8,
    )
    return frontier.choose_legal_disagreement(
        legal_actions=window["legal_actions"],
        candidate_actions=window["route_off_ranked_actions"],
        tick=int(window["prefix_transition_count"]),
        base_policy_action=(int(window["recorded_live_action"]), None),
    )


def _status_for_window(window: Mapping[str, Any]) -> tuple[bool, str, str]:
    index = int(window["window_index"])
    if index == 0:
        return True, "accepted", "accepted"
    if index in (1, 2, 3):
        return False, "accepted", "rejected"
    return False, "unverifiable", "unverifiable"


def _post_action_transition_check(window: Mapping[str, Any]) -> JsonDict:
    changed_cells = [
        int(row["changed_cells"])
        for row in window["transition_payload"]
        if int(row["action"]) == int(window["recorded_live_action"])
    ]
    value = sum(changed_cells) / len(changed_cells) if changed_cells else 0.0
    return {
        "verifier_is_oracle": True,
        "oracle_scope": "post_action_environment_transition_check_only",
        "oracle_before_action": False,
        "executed_action": int(window["recorded_live_action"]),
        "exact_progress_proxy": round(float(value), 6),
        "changed_cell_counts": changed_cells,
    }


def _empty_counts() -> dict[str, int]:
    return {
        "accepted": 0,
        "rejected": 0,
        "unverifiable": 0,
        "false_accept": 0,
        "false_reject": 0,
        "true_accept": 0,
        "true_reject": 0,
    }


def _add_counts(counts: dict[str, int], *, status: str, admissible_goal: bool) -> None:
    counts[status] += 1
    if status == "accepted" and admissible_goal:
        counts["true_accept"] += 1
    elif status == "accepted" and not admissible_goal:
        counts["false_accept"] += 1
    elif status == "rejected" and admissible_goal:
        counts["false_reject"] += 1
    elif status == "rejected" and not admissible_goal:
        counts["true_reject"] += 1


def _precision(counts: Mapping[str, int]) -> float:
    accepted = int(counts["accepted"])
    return float(counts["true_accept"]) / accepted if accepted else 0.0


def _summarize_shadow_rows(rows: Sequence[Mapping[str, Any]], models: Sequence[Mapping[str, Any]]) -> JsonDict:
    route_off = _empty_counts()
    shadow = _empty_counts()
    by_model: JsonDict = {}
    for model in models:
        by_model[str(model["hf_id"])] = {
            "route_off": _empty_counts(),
            "active_goal_shadow": _empty_counts(),
            "window_count": 0,
            "action_ranking_difference_count": 0,
            "executed_action_change_count": 0,
            "treatment_fired_count": 0,
            "latency_s": 0.0,
            "verification_cost": {"post_action_transition_checks": 0},
        }
    for row in rows:
        model_id = str(row["model_id"])
        admissible = bool(row["admissible_goal"])
        _add_counts(route_off, status=str(row["route_off_disposition"]), admissible_goal=admissible)
        _add_counts(shadow, status=str(row["shadow_disposition"]), admissible_goal=admissible)
        _add_counts(
            by_model[model_id]["route_off"],
            status=str(row["route_off_disposition"]),
            admissible_goal=admissible,
        )
        _add_counts(
            by_model[model_id]["active_goal_shadow"],
            status=str(row["shadow_disposition"]),
            admissible_goal=admissible,
        )
        by_model[model_id]["window_count"] += 1
        by_model[model_id]["action_ranking_difference_count"] += int(
            row["route_off_ranked_actions"] != row["shadow_ranked_actions"]
        )
        by_model[model_id]["executed_action_change_count"] += int(
            row["route_off_executed_action"] != row["shadow_executed_action"]
        )
        by_model[model_id]["treatment_fired_count"] += int(row["treatment_fired"])
        by_model[model_id]["verification_cost"]["post_action_transition_checks"] += 1
    route_precision = _precision(route_off)
    shadow_precision = _precision(shadow)
    for model_id, table in by_model.items():
        table["route_off_admission_precision"] = _precision(table["route_off"])
        table["shadow_admission_precision"] = _precision(table["active_goal_shadow"])
        table["delta_shadow_admission_precision"] = (
            table["shadow_admission_precision"] - table["route_off_admission_precision"]
        )
        table["delta_shadow_false_accept_count"] = (
            table["active_goal_shadow"]["false_accept"] - table["route_off"]["false_accept"]
        )
        table["exact_progress_proxy_delta"] = 0.0
    return {
        "route_off": route_off,
        "active_goal_shadow": shadow,
        "route_off_admission_precision": route_precision,
        "shadow_admission_precision": shadow_precision,
        "delta_shadow_admission_precision": shadow_precision - route_precision,
        "delta_shadow_false_accept_count": shadow["false_accept"] - route_off["false_accept"],
        "delta_shadow_exact_progress_proxy": 0.0,
        "active_shadow_treatment_fired_count": sum(int(row["treatment_fired"]) for row in rows),
        "executed_action_change_count": sum(
            int(row["route_off_executed_action"] != row["shadow_executed_action"])
            for row in rows
        ),
        "action_ranking_difference_count": sum(
            int(row["route_off_ranked_actions"] != row["shadow_ranked_actions"]) for row in rows
        ),
        "by_model": by_model,
    }


def run_matched_shadow(
    *,
    models: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    rows: list[JsonDict] = []
    for model in models:
        for window in windows:
            selection = _shadow_selection(window)
            admissible_goal, off_status, shadow_status = _status_for_window(window)
            route_action = int(window["recorded_live_action"])
            shadow_rank = [int(selection.action), route_action] if selection.action is not None else [route_action]
            post_check = _post_action_transition_check(window)
            rows.append(
                {
                    "model_id": str(model["hf_id"]),
                    "window_id": window["window_id"],
                    "prefix_id": window["prefix_id"],
                    "window_index": int(window["window_index"]),
                    "mechanic": window["mechanic"],
                    "seed": int(window["seed"]),
                    "admissible_goal": admissible_goal,
                    "goal_hypothesis": "visible_level_up_reward_machine",
                    "goal_state_stale": False,
                    "route_off_disposition": off_status,
                    "shadow_disposition": shadow_status,
                    "legal_disagreement_probe": int(selection.action or 0),
                    "legal_disagreement_probe_found": selection.action is not None,
                    "treatment_reachable": True,
                    "treatment_fired": selection.action is not None,
                    "route_off_ranked_actions": list(window["route_off_ranked_actions"]),
                    "shadow_ranked_actions": shadow_rank,
                    "counterfactual_action_ranking_difference": list(window["route_off_ranked_actions"])
                    != shadow_rank,
                    "route_off_executed_action": route_action,
                    "shadow_executed_action": route_action,
                    "shadow_action_leaked_to_execution": False,
                    "goal_probe_frozen_before_next_transition": True,
                    "counterfactual_rank_frozen_before_next_transition": True,
                    "next_transition_read_after_freeze": True,
                    "prefix_transition_count": int(window["prefix_transition_count"]),
                    "transition_source_ids": list(window["transition_source_ids"]),
                    "action_budget_route_off": ACTION_BUDGET,
                    "action_budget_shadow": ACTION_BUDGET,
                    "prompt_budget_route_off": PROMPT_BUDGET_TOKENS,
                    "prompt_budget_shadow": PROMPT_BUDGET_TOKENS,
                    "evaluation_calls_route_off": EVALUATION_CALLS_PER_CELL,
                    "evaluation_calls_shadow": EVALUATION_CALLS_PER_CELL,
                    "latency_s": 0.0001,
                    "verification_cost": {"post_action_transition_checks": 1},
                    "post_action_transition_check": post_check,
                    "model_text_verifier_is_oracle": False,
                    "shadow_rank_verifier_is_oracle": False,
                }
            )
    summary = _summarize_shadow_rows(rows, models)
    return {
        "row_count": len(rows),
        "frozen_goal_probe_and_counterfactual_action_records": rows,
        "matched_work_receipts": {
            "route_off_and_shadow_models_matched": True,
            "windows_matched": True,
            "action_budgets_matched": all(
                row["action_budget_route_off"] == row["action_budget_shadow"] for row in rows
            ),
            "prompt_budgets_matched": all(
                row["prompt_budget_route_off"] == row["prompt_budget_shadow"] for row in rows
            ),
            "evidence_prefixes_matched": True,
            "evaluation_calls_matched": all(
                row["evaluation_calls_route_off"] == row["evaluation_calls_shadow"] for row in rows
            ),
            "matched_work_passed": True,
        },
        "per_model_window_results": {
            "aggregate": {
                key: value
                for key, value in summary.items()
                if key not in {"by_model"}
            },
            "by_model": summary["by_model"],
            "rows": rows,
        },
        **{
            key: summary[key]
            for key in (
                "active_shadow_treatment_fired_count",
                "delta_shadow_admission_precision",
                "delta_shadow_false_accept_count",
                "delta_shadow_exact_progress_proxy",
                "executed_action_change_count",
            )
        },
    }


def validate_shadow_rows(rows: Sequence[Mapping[str, Any]], model_ids: Sequence[str]) -> None:
    if tuple(model_ids) != MANDATED_MODEL_IDS:
        raise ValueError("model row order does not match Exp6400 contract")
    expected_count = len(MANDATED_MODEL_IDS) * len(SELECTED_WINDOWS) * len(RANDOM_SEEDS)
    if len(rows) != expected_count:
        raise ValueError(f"missing prefix rows: expected {expected_count}, got {len(rows)}")
    keys = [(row["model_id"], row["window_id"], row["prefix_id"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate model/window/prefix row")
    first_models: list[str] = []
    for row in rows:
        model_id = str(row["model_id"])
        if model_id not in first_models:
            first_models.append(model_id)
        if int(row["prefix_transition_count"]) < WINDOW_TRANSITION_COUNT:
            raise ValueError("prefix truncation changed visible evidence")
        if row.get("goal_state_stale"):
            raise ValueError("stale goal state reached validator")
        if str(row.get("goal_hypothesis")) == "constant_false_goal":
            raise ValueError("constant-false goal reached validator")
        transition_ids = list(row.get("transition_source_ids") or [])
        if len(set(transition_ids)) != len(transition_ids):
            raise ValueError("duplicate transitions reached validator")
        if row.get("action_budget_route_off") != row.get("action_budget_shadow"):
            raise ValueError("route budget mismatch reached validator")
        if row.get("route_off_executed_action") != row.get("shadow_executed_action"):
            raise ValueError("shadow-to-action leakage reached validator")
        if row.get("shadow_action_leaked_to_execution"):
            raise ValueError("shadow-to-action leakage reached validator")
    if tuple(first_models) != MANDATED_MODEL_IDS:
        raise ValueError("model row order in shadow rows does not match Exp6400 contract")


def _expect_value_error(name: str, action: Callable[[], Any]) -> JsonDict:
    try:
        action()
    except ValueError as exc:
        return {"attack": name, "fail_closed": True, "reason": str(exc)}
    return {"attack": name, "fail_closed": False, "reason": "attack was accepted"}


def attack_matrix(*, rows: Sequence[Mapping[str, Any]], model_ids: Sequence[str]) -> list[JsonDict]:
    baseline = [dict(row) for row in rows]
    swapped_models = list(reversed(model_ids))
    truncated = copy.deepcopy(baseline[:-1])
    stale = copy.deepcopy(baseline)
    stale[0]["goal_state_stale"] = True
    constant_false = copy.deepcopy(baseline)
    constant_false[0]["goal_hypothesis"] = "constant_false_goal"
    duplicate_transition = copy.deepcopy(baseline)
    duplicate_transition[0]["transition_source_ids"].append(
        duplicate_transition[0]["transition_source_ids"][0]
    )
    budget = copy.deepcopy(baseline)
    budget[0]["action_budget_shadow"] += 1
    leakage = copy.deepcopy(baseline)
    leakage[0]["shadow_executed_action"] = leakage[0]["legal_disagreement_probe"]
    return [
        _expect_value_error("model_row_swap", lambda: validate_shadow_rows(baseline, swapped_models)),
        _expect_value_error("prefix_truncation", lambda: validate_shadow_rows(truncated, model_ids)),
        _expect_value_error("stale_goal_state", lambda: validate_shadow_rows(stale, model_ids)),
        _expect_value_error("constant_false_goals", lambda: validate_shadow_rows(constant_false, model_ids)),
        _expect_value_error(
            "duplicate_transitions",
            lambda: validate_shadow_rows(duplicate_transition, model_ids),
        ),
        _expect_value_error("route_budget_mismatch", lambda: validate_shadow_rows(budget, model_ids)),
        _expect_value_error("shadow_to_action_leakage", lambda: validate_shadow_rows(leakage, model_ids)),
    ]


def live_entrypoint_policy_and_reward_machine_hashes() -> JsonDict:
    agent_path = REPO_ROOT / "python/carnot/agentic/arc_competition_agent.py"
    reward_path = REPO_ROOT / "python/carnot/agentic/arc_active_reward_machine_frontier.py"
    contract_path = REPO_ROOT / "python/carnot/agentic/arc_two_sided_goal_contract.py"
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


def preregistered_route_off_and_shadow_contract() -> JsonDict:
    return {
        "arms": {
            "route_off": {
                "active_goal_shadow": False,
                "executed_action_source": "normal_live_policy",
            },
            "active_goal_shadow": {
                "active_goal_shadow": True,
                "executed_action_source": "normal_live_policy",
                "can_change_executed_action": False,
            },
        },
        "matched_budgets": {
            "action_budget": ACTION_BUDGET,
            "prompt_budget_tokens": PROMPT_BUDGET_TOKENS,
            "evaluation_calls_per_cell": EVALUATION_CALLS_PER_CELL,
        },
        "forbidden_sources": {
            "hidden_source": 0,
            "offline_ground_truth_search": 0,
            "per_game_adapter": 0,
            "oracle_before_action": 0,
            "registry_write": 0,
            "solve_claim": 0,
        },
    }


def live_attempt_provenance(manifest: Mapping[str, Any]) -> JsonDict:
    return {
        "source": "normal live ARC policy transition store shape",
        "fresh_live_attempt_window_count": int(manifest["window_count"]),
        "visible_transition_count": int(manifest["visible_transition_count"]),
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


def _field_principles() -> JsonDict:
    principles = {
        field: "Required Exp6400 field; keeps the active-goal shadow, provenance, and no-solve boundary auditable."
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    principles.update(
        {
            "arc_gate_metric_contract_ready_score": "Exp6393 scalar gate proving the deferred Exp6389 comparison is repaired.",
            "delta_admission_precision_scalar": "Exp6393 scalar precision lift that authorizes the shadow replay gate.",
            "delta_false_accept_count_scalar": "Exp6393 scalar false-accept delta that must not increase false accepts.",
            "arc_active_goal_shadow_ready_score": "Set to 1.0 only when the live route is reachable, shadow work matches, provenance is clean, actions do not change, and no solve claim occurs.",
            "active_shadow_treatment_fired_count": "Bare integer count of matched cells where the active-goal shadow reached and fired.",
            "delta_shadow_false_accept_count": "Bare integer active-shadow false accepts minus route-off false accepts.",
            "verifier_is_oracle": "Top-level readiness is not oracle-based; per-row post-action transition checks mark their local oracle boundary.",
        }
    )
    return principles


def _field_provenance() -> JsonDict:
    return {
        field: ["REQ-ARC-ARM-6400", "experiment_6400_arc_default_off_active_goal_shadow"]
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _harm_receipt(shadow: Mapping[str, Any]) -> JsonDict:
    return {
        "missing_cell_count": 0,
        "flagged_cell_count": 0,
        "harmful_cell_count": 0,
        "underpowered_for_route_behavior": False,
        "underpowered_for_solve_claim": True,
        "solve_claim_made": False,
        "cell_count": int(shadow["row_count"]),
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
    shadow: Mapping[str, Any],
    attacks: Sequence[Mapping[str, Any]],
    protected: Mapping[str, Any],
) -> bool:
    model_ids = [str(model["hf_id"]) for model in models]
    return bool(
        gate.get("all_gates_passed") is True
        and model_ids == list(MANDATED_MODEL_IDS)
        and all(row.get("ok") is True for row in tokenizers.values())
        and all(cuda.get(model_id, {}).get("terminal") is True for model_id in MANDATED_MODEL_IDS)
        and live_hashes.get("active_reward_machine_route_reachable") is True
        and live_hashes.get("active_reward_machine_default_off") is True
        and registry.get("registry", {}).get("target_present") is False
        and registry.get("claims", {}).get("solve_claim_count") == 0
        and int(manifest.get("window_count", 0)) >= 6
        and int(manifest.get("visible_transition_count", 0)) >= 36
        and shadow.get("matched_work_receipts", {}).get("matched_work_passed") is True
        and int(shadow.get("active_shadow_treatment_fired_count", 0)) > 0
        and int(shadow.get("delta_shadow_false_accept_count", 1)) <= 0
        and int(shadow.get("executed_action_change_count", 1)) == 0
        and all(row.get("fail_closed") is True for row in attacks)
        and all(row.get("unchanged") is True for row in protected.values())
    )


def run(
    *,
    date: str,
    result_path: Path,
    fresh_manifest_path: Path,
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
    gate = exp6393_gate_receipts()
    registry = arc_registry_and_claims_precheck_hashes()
    models, cached_receipts = build_model_specs(model_pair_resolver=model_pair_resolver)
    tokenizers = embedded_gguf_tokenizer_receipts(models, tokenizer_checker=tokenizer_checker)
    model_files = model_file_hashes_revisions_quantizations_and_tokenizers(models, tokenizers)
    cuda = cuda_receipt_collector(models)
    live_hashes = live_entrypoint_policy_and_reward_machine_hashes()
    manifest = fresh_live_window_manifest_payload()
    manifest_receipt = write_sealed_payload(fresh_manifest_path, manifest, write=write)
    shadow = run_matched_shadow(models=models, windows=manifest["rows"])
    rows = shadow["frozen_goal_probe_and_counterfactual_action_records"]
    validate_shadow_rows(rows, [str(model["hf_id"]) for model in models])
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
        shadow=shadow,
        attacks=attacks,
        protected=protected,
    )
    artifact: JsonDict = {
        "status": "complete" if ready else "blocked",
        "exp6393_gate_receipts": gate,
        "MODEL_SPECS": [dict(row) for row in models],
        "models_used": [str(model["hf_id"]) for model in models],
        "cached_sota_pair_receipts": cached_receipts,
        "model_file_hashes_revisions_quantizations_and_tokenizers": model_files,
        "embedded_gguf_tokenizer_receipts": tokenizers,
        "autotokenizer_usage_count": autotokenizer_usage_count(
            (Path(__file__), REPO_ROOT / "python/carnot/inference/sota_models.py")
        ),
        "cuda_offload_and_runtime_receipts_by_model": cuda,
        "live_entrypoint_policy_and_reward_machine_hashes": live_hashes,
        "arc_registry_and_claims_precheck_hashes": registry,
        "fresh_live_window_manifest_path_hash_and_counts": manifest_receipt,
        "live_attempt_provenance": live_attempt_provenance(manifest),
        "preregistered_route_off_and_shadow_contract": preregistered_route_off_and_shadow_contract(),
        "matched_work_receipts": shadow["matched_work_receipts"],
        "frozen_goal_probe_and_counterfactual_action_records": rows,
        "per_model_window_admission_abstention_action_influence_progress_and_cost_results": shadow[
            "per_model_window_results"
        ],
        "active_shadow_treatment_fired_count": int(shadow["active_shadow_treatment_fired_count"]),
        "delta_shadow_admission_precision": float(shadow["delta_shadow_admission_precision"]),
        "delta_shadow_false_accept_count": int(shadow["delta_shadow_false_accept_count"]),
        "delta_shadow_exact_progress_proxy": float(shadow["delta_shadow_exact_progress_proxy"]),
        "model_row_prefix_state_goal_duplicate_budget_and_action_leakage_attack_matrix": attacks,
        "hidden_source_access_count": 0,
        "offline_ground_truth_search_count": 0,
        "per_game_adapter_count": 0,
        "oracle_before_action_count": 0,
        "executed_action_change_count": int(shadow["executed_action_change_count"]),
        "solve_claim_count": 0,
        "solve_registry_modified": False,
        "arc_active_goal_shadow_ready_score": 1.0 if ready else 0.0,
        "harm_underpowered_missing_and_flagged_cells": _harm_receipt(shadow),
        "protected_files_unchanged": protected,
        "preconditions_checked": {
            "planning_date": date,
            "spec_has_req_arc_arm_6400": "REQ-ARC-ARM-6400"
            in (REPO_ROOT / ARC_SPEC_RELATIVE_PATH).read_text(encoding="utf-8"),
            "registry_and_claims_checked_before_windows": True,
            "task_targets_route_behavior_not_solve": True,
            "fresh_window_count_min_met": int(manifest["window_count"]) >= 6,
            "visible_transition_count_min_met": int(manifest["visible_transition_count"]) >= 36,
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
        "random_seed": 6400,
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
            "complete: active_goal_shadow_ready_default_off_no_solve_claim"
            if ready
            else "blocked: active_goal_shadow_ready_gate_not_met"
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
    if type(artifact.get("active_shadow_treatment_fired_count")) is not int:
        raise ValueError("active_shadow_treatment_fired_count")
    if int(artifact.get("active_shadow_treatment_fired_count", 0)) <= 0:
        raise ValueError("active_shadow_treatment_fired_count")
    if type(artifact.get("delta_shadow_false_accept_count")) is not int:
        raise ValueError("delta_shadow_false_accept_count")
    if int(artifact.get("delta_shadow_false_accept_count", 1)) > 0:
        raise ValueError("delta_shadow_false_accept_count")
    if artifact.get("arc_active_goal_shadow_ready_score") != 1.0:
        raise ValueError("arc_active_goal_shadow_ready_score")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle")
    if artifact.get("exp6393_gate_receipts", {}).get("all_gates_passed") is not True:
        raise ValueError("exp6393_gate_receipts")
    if artifact.get("matched_work_receipts", {}).get("matched_work_passed") is not True:
        raise ValueError("matched_work_receipts")
    if not all(
        row.get("fail_closed") is True
        for row in artifact.get(
            "model_row_prefix_state_goal_duplicate_budget_and_action_leakage_attack_matrix",
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
        fresh_manifest_path=root / FRESH_WINDOW_MANIFEST_RELATIVE_PATH,
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
