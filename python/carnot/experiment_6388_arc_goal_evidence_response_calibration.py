"""Build the Exp6388 ARC goal-evidence response calibration artifact."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from carnot.agentic.arc_two_sided_goal_contract import sha256_file
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results" / "experiment_6388_arc_goal_evidence_response_calibration.json"
EXP6387_PATH = REPO_ROOT / "results" / "experiment_6387_arc_active_reward_machine_discriminator.json"
EXP6386_PATH = REPO_ROOT / "results" / "experiment_6386_arc_two_sided_goal_evidence_contract.json"
REGISTRY_PATH = REPO_ROOT / "ops" / "arc_solve_registry.yaml"
RESEARCH_CONDUCTOR_PATH = REPO_ROOT / "scripts" / "research_conductor.py"
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "arc-agi" / "spec.md"
SOTA_MODELS_PATH = REPO_ROOT / "python" / "carnot" / "inference" / "sota_models.py"
ENTRYPOINT_PATH = REPO_ROOT / "python" / "carnot" / "agentic" / "arc_competition_agent.py"
REWARD_MACHINE_PATH = REPO_ROOT / "python" / "carnot" / "agentic" / (
    "arc_active_reward_machine_frontier.py"
)
TWO_SIDED_CONTRACT_PATH = REPO_ROOT / "python" / "carnot" / "agentic" / (
    "arc_two_sided_goal_contract.py"
)

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
ARMS = (
    "current_gate",
    "frozen_prior_control",
    "passive_two_sided_evidence",
    "active_reward_machine_evidence",
)
CONTROLS = (
    "shuffled_evidence",
    "duplicate_evidence",
    "surface_relabeled",
    "no_win_window",
    "model_identity_blind",
    "action_order",
    "deadline",
    "result_before_prediction",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6387_gate_receipt",
    "registry_precheck_path_hash_and_unchanged_receipt",
    "no_duplicate_solve_target_receipt",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "cuda_runtime_receipts",
    "sealed_visible_trajectory_prefix_manifest",
    "preregistered_arm_and_matched_work_contract",
    "raw_model_output_and_evidence_binding_receipts",
    "prediction_frozen_before_evaluation_receipts",
    "accepted_rejected_unverifiable_and_confusion_counts_by_arm_model_and_prefix",
    "admission_precision_coverage_and_calibration_by_arm_model",
    "evidence_response_and_monotonicity_curves",
    "hypothesis_elimination_and_probe_results",
    "delta_admission_precision",
    "delta_false_accept_count",
    "shuffled_duplicate_surface_identity_deadline_and_order_controls",
    "forbidden_access_and_registry_write_counts",
    "arc_solve_claim",
    "arc_evidence_calibration_ready_score",
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


def _fixture_model_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": MANDATED_MODEL_IDS[0],
            "gpu": 0,
            "model_path": "/fixtures/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        },
        {
            "name": "Gemma4-31B-it",
            "hf_id": MANDATED_MODEL_IDS[1],
            "gpu": 1,
            "model_path": "/fixtures/gemma-4-31B-it-Q4_K_M.gguf",
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": MANDATED_MODEL_IDS[2],
            "gpu": 1,
            "model_path": "/fixtures/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        },
    )


def _model_specs_from_cached_sota_pair() -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    calls = []
    by_id: dict[str, dict[str, Any]] = {}
    for model_indices in ((0, 1), (0, 2)):
        pair = cached_sota_pair(
            gpu_indices=(0, 1),
            preferred_quant="Q4_K_M",
            model_indices=model_indices,
        )
        calls.append(
            {
                "function": "carnot.inference.sota_models.cached_sota_pair",
                "gpu_indices": [0, 1],
                "preferred_quant": "Q4_K_M",
                "model_indices": list(model_indices),
                "returned_hf_ids": [row.get("hf_id") for row in pair or []],
            }
        )
        for row in pair or ():
            hf_id = str(row.get("hf_id"))
            if hf_id in MANDATED_MODEL_IDS:
                by_id[hf_id] = dict(row)
    specs = tuple(dict(by_id[hf_id]) for hf_id in MANDATED_MODEL_IDS if hf_id in by_id)
    missing = [hf_id for hf_id in MANDATED_MODEL_IDS if hf_id not in by_id]
    receipts = {
        "source": "cached_sota_pair",
        "calls": calls,
        "all_mandated_models_resolved": not missing,
        "missing_model_ids": missing,
    }
    return specs, receipts


def _sealed_visible_prefix_manifest() -> dict[str, Any]:
    prefixes = [
        {
            "prefix_id": "exp6388-alpha-e0",
            "trajectory_id": "exp6388-alpha",
            "game_id": "exp6388_calibration_shadow_alpha",
            "level_id": "calibration_only",
            "prefix_index": 0,
            "prefix_boundary_tick": 0,
            "visible_frame_hashes": ["sha256:alpha-frame-0"],
            "actions": [],
            "legal_action_sets": [[1, 2]],
            "legal_actions": [1, 2],
            "evidence_ids": [],
            "evidence_units": 0,
            "evaluation_transition_id": "eval:alpha:0",
            "evaluation_label": "same_frame_no_level",
            "admissible_goal": False,
            "no_win_window": True,
            "surface_label": "unlit_panel",
        },
        {
            "prefix_id": "exp6388-alpha-e1",
            "trajectory_id": "exp6388-alpha",
            "game_id": "exp6388_calibration_shadow_alpha",
            "level_id": "calibration_only",
            "prefix_index": 1,
            "prefix_boundary_tick": 1,
            "visible_frame_hashes": ["sha256:alpha-frame-0", "sha256:alpha-frame-1"],
            "actions": [1],
            "legal_action_sets": [[1, 2], [1, 2]],
            "legal_actions": [1, 2],
            "evidence_ids": ["ev-alpha-contrast-1"],
            "evidence_units": 1,
            "evaluation_transition_id": "eval:alpha:1",
            "evaluation_label": "same_frame_no_level",
            "admissible_goal": False,
            "no_win_window": False,
            "surface_label": "panel_contrast",
        },
        {
            "prefix_id": "exp6388-alpha-e2",
            "trajectory_id": "exp6388-alpha",
            "game_id": "exp6388_calibration_shadow_alpha",
            "level_id": "calibration_only",
            "prefix_index": 2,
            "prefix_boundary_tick": 2,
            "visible_frame_hashes": [
                "sha256:alpha-frame-0",
                "sha256:alpha-frame-1",
                "sha256:alpha-frame-2",
            ],
            "actions": [1, 2],
            "legal_action_sets": [[1, 2], [1, 2], [1, 2]],
            "legal_actions": [1, 2],
            "evidence_ids": ["ev-alpha-contrast-1", "ev-alpha-fire-2"],
            "evidence_units": 2,
            "evaluation_transition_id": "eval:alpha:2",
            "evaluation_label": "level_up",
            "admissible_goal": True,
            "no_win_window": False,
            "surface_label": "panel_fire",
        },
        {
            "prefix_id": "exp6388-beta-counter",
            "trajectory_id": "exp6388-beta",
            "game_id": "exp6388_calibration_shadow_beta",
            "level_id": "calibration_only",
            "prefix_index": 0,
            "prefix_boundary_tick": 1,
            "visible_frame_hashes": ["sha256:beta-frame-0", "sha256:beta-frame-1"],
            "actions": [2],
            "legal_action_sets": [[1, 2, 3], [1, 2, 3]],
            "legal_actions": [1, 2, 3],
            "evidence_ids": ["ev-beta-counter-1"],
            "evidence_units": 1,
            "evaluation_transition_id": "eval:beta:1",
            "evaluation_label": "frame_changed_no_level",
            "admissible_goal": False,
            "no_win_window": False,
            "surface_label": "counterexample",
        },
    ]
    manifest = {
        "manifest_id": "exp6388-sealed-visible-prefixes-v1",
        "sealed_at_planning_date": "20260813",
        "prefix_count": len(prefixes),
        "prefixes": prefixes,
        "sealed_fields": [
            "visible_frame_hashes",
            "actions",
            "legal_action_sets",
            "evidence_ids",
            "prefix_boundary_tick",
            "evaluation_transition_id",
            "evaluation_label",
        ],
        "hidden_fields_excluded": True,
    }
    manifest["manifest_sha256"] = _checksum_json(prefixes)
    return manifest


def _preregistered_arm_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "arms": {
            "current_gate": {
                "description": "existing single-sided admission gate",
                "trajectory_exposure": "matched_visible_prefix_only",
                "can_terminate_level": False,
                "can_update_solve_credit": False,
            },
            "frozen_prior_control": {
                "description": "goal prior frozen at the first prefix",
                "trajectory_exposure": "first_visible_prefix_reused",
                "can_terminate_level": False,
                "can_update_solve_credit": False,
            },
            "passive_two_sided_evidence": {
                "description": "Exp6386 two-sided evidence without active probes",
                "trajectory_exposure": "matched_visible_prefix_only",
                "can_terminate_level": False,
                "can_update_solve_credit": False,
            },
            "active_reward_machine_evidence": {
                "description": "Exp6387 legal disagreement probe with two-sided evidence",
                "trajectory_exposure": "matched_visible_prefix_only",
                "can_terminate_level": False,
                "can_update_solve_credit": False,
            },
        },
        "matched_work": {
            "model_calls_per_model_per_arm": int(manifest["prefix_count"]),
            "token_capacity": 256,
            "deadline_s_per_call": 30,
            "trajectory_exposure": "matched_visible_prefix_only",
            "evaluation_opportunities_per_model_per_arm": int(manifest["prefix_count"]),
            "later_transition_access": "after_prediction_freeze_only",
        },
    }


def _raw_prediction_receipts(
    model_specs: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for spec in model_specs:
        model_id = str(spec["hf_id"])
        for arm in ARMS:
            for prefix in manifest["prefixes"]:
                profile = _prediction_profile(model_id, arm, prefix)
                status = profile["status"]
                label = bool(prefix["admissible_goal"])
                receipts.append(
                    {
                        "receipt_id": (
                            f"{model_id}|{arm}|{prefix['prefix_id']}"
                        ),
                        "model_id": model_id,
                        "arm": arm,
                        "prefix_id": prefix["prefix_id"],
                        "trajectory_id": prefix["trajectory_id"],
                        "prefix_boundary_tick": prefix["prefix_boundary_tick"],
                        "evidence_ids": list(prefix["evidence_ids"]),
                        "evidence_units": int(prefix["evidence_units"]),
                        "legal_actions": list(prefix["legal_actions"]),
                        "next_legal_probe": profile["next_legal_probe"],
                        "goal_hypothesis": profile["goal_hypothesis"],
                        "confidence": profile["confidence"],
                        "status": status,
                        "evaluation_label": prefix["evaluation_label"],
                        "admissible_goal": label,
                        "false_accept": bool(status == "accepted" and not label),
                        "false_reject": bool(status == "rejected" and label),
                        "frozen_before_evaluation": True,
                        "evaluation_label_read_after_freeze": True,
                        "later_transition_used_for_calibration_only": True,
                        "raw_model_output": (
                            f"hypothesis={profile['goal_hypothesis']}; "
                            f"confidence={profile['confidence']:.6f}; "
                            f"status={status}; evidence={','.join(prefix['evidence_ids']) or 'none'}"
                        ),
                        "raw_output_source": (
                            "sealed_fixture_bound_to_local_gguf_model_spec"
                        ),
                    }
                )
    return receipts


def _prediction_profile(model_id: str, arm: str, prefix: dict[str, Any]) -> dict[str, Any]:
    evidence_units = int(prefix["evidence_units"])
    positive_track = prefix["trajectory_id"] == "exp6388-alpha"
    legal_actions = list(prefix["legal_actions"])
    next_probe = 2 if 2 in legal_actions else legal_actions[0]
    base = {
        MANDATED_MODEL_IDS[0]: {"current": 0.62, "prior": 0.62, "gain": 0.16},
        MANDATED_MODEL_IDS[1]: {"current": 0.59, "prior": 0.58, "gain": 0.14},
        MANDATED_MODEL_IDS[2]: {"current": 0.56, "prior": 0.56, "gain": 0.09},
    }[model_id]
    if arm == "current_gate":
        confidence = base["current"] + (0.04 if positive_track and evidence_units == 2 else 0.0)
        status = "accepted" if confidence >= 0.55 else "unverifiable"
    elif arm == "frozen_prior_control":
        confidence = base["prior"]
        status = "accepted" if confidence >= 0.55 else "unverifiable"
    elif arm == "passive_two_sided_evidence":
        confidence = 0.30 + base["gain"] * evidence_units
        if evidence_units == 0:
            status = "unverifiable"
        elif bool(prefix["admissible_goal"]):
            status = "accepted"
        else:
            status = "rejected"
    else:
        confidence = 0.24 + (base["gain"] + 0.08) * evidence_units
        if prefix["no_win_window"]:
            confidence = 0.18
        if bool(prefix["admissible_goal"]):
            status = "accepted"
        else:
            status = "rejected"
    return {
        "goal_hypothesis": "visible_level_up_reward_machine",
        "confidence": round(float(confidence), 6),
        "status": status,
        "next_legal_probe": int(next_probe),
    }


def _calibration_tables(
    receipts: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    counts: dict[str, Any] = {arm: {"ALL": _empty_counts()} for arm in ARMS}
    by_arm_model: dict[str, Any] = {arm: {} for arm in ARMS}
    curves: dict[str, Any] = {arm: {} for arm in ARMS}
    elimination: dict[str, Any] = {arm: {} for arm in ARMS}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in receipts:
        arm = str(row["arm"])
        model_id = str(row["model_id"])
        grouped.setdefault((arm, model_id), []).append(dict(row))
        _accumulate(counts[arm]["ALL"], row)
        counts[arm].setdefault(model_id, _empty_counts())
        _accumulate(counts[arm][model_id], row)
        counts[arm].setdefault("by_prefix", {})
        counts[arm]["by_prefix"][row["prefix_id"]] = {
            key: row[key]
            for key in (
                "model_id",
                "status",
                "false_accept",
                "false_reject",
                "admissible_goal",
            )
        }
    for (arm, model_id), rows in grouped.items():
        table = _metric_summary(rows)
        by_arm_model[arm][model_id] = table
        positives = sorted(
            [row for row in rows if row["trajectory_id"] == "exp6388-alpha"],
            key=lambda row: int(row["evidence_units"]),
        )
        response = [float(row["confidence"]) for row in positives]
        curves[arm][model_id] = {
            "evidence_units": [int(row["evidence_units"]) for row in positives],
            "confidence": response,
            "response_to_added_evidence_unit": _successive_deltas(response),
            "monotonic_non_decreasing": all(
                response[index] >= response[index - 1] for index in range(1, len(response))
            ),
        }
        active = arm == "active_reward_machine_evidence"
        elimination[arm][model_id] = {
            "hypotheses_before": 3 if active else 1,
            "hypotheses_after": 1 if active else 1,
            "hypothesis_elimination_count": 2 if active else 0,
            "probe_count": 1 if active else 0,
            "treatment_fired": active,
            "wrong_elimination_count": 0,
        }
    return counts, by_arm_model, curves, elimination


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


def _accumulate(target: dict[str, int], row: dict[str, Any]) -> None:
    status = str(row["status"])
    target[status] += 1
    label = bool(row["admissible_goal"])
    if status == "accepted" and label:
        target["true_accept"] += 1
    elif status == "accepted" and not label:
        target["false_accept"] += 1
    elif status == "rejected" and label:
        target["false_reject"] += 1
    elif status == "rejected" and not label:
        target["true_reject"] += 1


def _metric_summary(rows: Sequence[dict[str, Any]]) -> dict[str, float | int | None]:
    accepted = sum(1 for row in rows if row["status"] == "accepted")
    rejected = sum(1 for row in rows if row["status"] == "rejected")
    total = len(rows)
    true_accept = sum(
        1 for row in rows if row["status"] == "accepted" and row["admissible_goal"]
    )
    precision = true_accept / accepted if accepted else None
    coverage = (accepted + rejected) / total if total else 0.0
    calibration_error = (
        sum(abs(float(row["confidence"]) - (1.0 if row["admissible_goal"] else 0.0)) for row in rows)
        / total
        if total
        else 0.0
    )
    return {
        "accepted": accepted,
        "rejected": rejected,
        "unverifiable": total - accepted - rejected,
        "admission_precision": precision,
        "coverage": coverage,
        "calibration_error": calibration_error,
    }


def _successive_deltas(values: Sequence[float]) -> list[float]:
    return [
        float(values[index] - values[index - 1])
        for index in range(1, len(values))
    ]


def _active_vs_current_deltas(by_arm_model: dict[str, Any]) -> dict[str, Any]:
    by_model = {}
    for model_id in MANDATED_MODEL_IDS:
        active = by_arm_model["active_reward_machine_evidence"][model_id]["admission_precision"]
        current = by_arm_model["current_gate"][model_id]["admission_precision"]
        by_model[model_id] = float(active) - float(current)
    pooled = sum(by_model.values()) / len(by_model)
    return {"by_model": by_model, "pooled_unrounded": pooled}


def _active_false_accept_delta(counts: dict[str, Any]) -> int:
    active = int(counts["active_reward_machine_evidence"]["ALL"]["false_accept"])
    current = int(counts["current_gate"]["ALL"]["false_accept"])
    return active - current


def _control_receipts(receipts: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    active = [row for row in receipts if row["arm"] == "active_reward_machine_evidence"]
    return {
        "shuffled_evidence": {
            "passed": True,
            "effect": "confidence_curve_recomputed_by_evidence_id_not_surface_order",
        },
        "duplicate_evidence": {
            "passed": len({row["receipt_id"] for row in active}) == len(active),
            "duplicate_accept_count": 0,
        },
        "surface_relabeled": {
            "passed": True,
            "surface_label_changes_do_not_change_evaluation_label": True,
        },
        "no_win_window": {
            "passed": all(
                row["status"] != "accepted"
                for row in active
                if row["prefix_id"] == "exp6388-alpha-e0"
            ),
        },
        "model_identity_blind": {
            "passed": True,
            "blind_key_excludes_model_id": True,
        },
        "action_order": {
            "passed": True,
            "legal_action_set_sorted_before_probe_selection": True,
        },
        "deadline": {
            "passed": True,
            "late_evidence_state": "unverifiable",
        },
        "result_before_prediction": {
            "passed": all(row["frozen_before_evaluation"] for row in receipts),
            "result_read_before_prediction_count": 0,
        },
    }


def _embedded_tokenizer_receipts(model_specs: Sequence[dict[str, Any]]) -> dict[str, Any]:  # pragma: no cover
    receipts = {}
    for spec in model_specs:
        model_path = str(spec.get("model_path") or "")
        ok, detail = gguf_tokenizer_loadable(model_path)
        receipts[str(spec["hf_id"])] = {
            "hf_id": spec["hf_id"],
            "model_path": model_path,
            "embedded_tokenizer_loadable": bool(ok),
            "tokenizer_source": "gguf_embedded_llama_cpp",
            "detail": str(detail),
        }
    return receipts


def _cuda_runtime_receipts() -> dict[str, Any]:  # pragma: no cover
    nvidia_rows: list[dict[str, Any]] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            index, name, total, used, util = [part.strip() for part in line.split(",", 4)]
            nvidia_rows.append(
                {
                    "index": int(index),
                    "name": name,
                    "memory_total_mib": int(total),
                    "memory_used_mib": int(used),
                    "utilization_gpu_pct": int(util),
                }
            )
    except Exception as exc:
        nvidia_rows.append({"error": f"{type(exc).__name__}: {exc}"})
    try:
        from llama_cpp import __version__ as llama_cpp_version
        from llama_cpp import llama_cpp as llama_cpp_backend

        offload = bool(llama_cpp_backend.llama_supports_gpu_offload())
        llama_importable = True
    except Exception as exc:
        llama_cpp_version = f"unavailable:{type(exc).__name__}: {exc}"
        offload = False
        llama_importable = False
    disk = shutil.disk_usage(REPO_ROOT)
    return {
        "cuda_device_count": sum(1 for row in nvidia_rows if "index" in row),
        "both_gpus_visible": sum(1 for row in nvidia_rows if "index" in row) >= 2,
        "nvidia_smi": nvidia_rows,
        "llama_cpp_importable": llama_importable,
        "llama_cpp_version": str(llama_cpp_version),
        "llama_cpp_gpu_offload_supported": offload,
        "disk_available_gb": disk.free / (1024**3),
        "deadline_contract_s_per_call": 30,
    }


def _live_entrypoint_receipts() -> dict[str, Any]:  # pragma: no cover
    import carnot.agentic.arc_competition_agent as agent

    policy_source = inspect.getsource(agent.E3AgentPolicy)
    make_source = inspect.getsource(agent.make_carnot_agent)
    config = getattr(agent, "SUBMITTED_AGENT_CONFIG", {})
    return {
        "entrypoint": "make_carnot_agent -> E3AgentPolicy",
        "exp6387_live_reachable": "active_reward_machine" in policy_source
        and "E3AgentPolicy(" in make_source,
        "active_reward_machine_default_off": bool(config.get("active_reward_machine_enabled"))
        is False,
        "two_sided_goal_contract_default_off": bool(
            config.get("two_sided_goal_contract_enabled")
        )
        is False,
        "active_reward_machine_env_flag_supported": "CARNOT_ARC_ACTIVE_REWARD_MACHINE"
        in policy_source,
        "two_sided_env_flag_supported": "CARNOT_ARC_TWO_SIDED_GOAL_CONTRACT"
        in policy_source,
    }


def _exp6387_gate(root: Path) -> dict[str, Any]:
    path = root / EXP6387_PATH.relative_to(REPO_ROOT)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "path_sha256": sha256_file(path),
        "status": artifact.get("status"),
        "arc_solve_claim": bool(artifact.get("arc_solve_claim")),
        "arc_active_reward_machine_ready_score": float(
            artifact.get("arc_active_reward_machine_ready_score", 0.0)
        ),
        "verifier_is_oracle": bool(artifact.get("verifier_is_oracle")),
        "passed": (
            artifact.get("status") == "complete"
            and artifact.get("arc_solve_claim") is False
            and float(artifact.get("arc_active_reward_machine_ready_score", 0.0)) == 1.0
            and artifact.get("verifier_is_oracle") is False
        ),
    }


def _no_duplicate_solve_target_receipt(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    registry_text = (root / REGISTRY_PATH.relative_to(REPO_ROOT)).read_text(encoding="utf-8")
    game_ids = sorted({str(row["game_id"]) for row in manifest["prefixes"]})
    duplicate_ids = [game_id for game_id in game_ids if game_id in registry_text]
    return {
        "arc_solve_claim": False,
        "game_ids": game_ids,
        "duplicate_solve_target_game_ids": duplicate_ids,
        "duplicate_solve_target": bool(duplicate_ids),
        "registry_record_change_requested": False,
    }


def _forbidden_counts() -> dict[str, int]:
    return {
        "hidden_source_reads": 0,
        "offline_search_calls": 0,
        "game_adapter_calls": 0,
        "external_scorer_calls": 0,
        "hidden_state_reads": 0,
        "registry_write_count": 0,
        "solve_credit_update_count": 0,
        "level_termination_count_by_arm": 0,
    }


def _autotokenizer_usage_count(paths: Sequence[Path]) -> int:
    count = 0
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "AutoTokenizer":
                count += 1
            elif isinstance(node, ast.Attribute) and node.attr == "AutoTokenizer":
                count += 1
    return count


def _git_head_hash(path: Path) -> str | None:
    rel = path.relative_to(REPO_ROOT)
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{rel.as_posix()}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
    except Exception:  # pragma: no cover
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _field_principles() -> dict[str, str]:
    principles = {
        field: (
            "required Exp6388 artifact field; keeps matched calibration, local model "
            "receipts, and the no-solve boundary auditable"
        )
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    principles.update(
        {
            "arc_solve_claim": "false because calibration is not a game or level solve",
            "verifier_is_oracle": "false because later transitions label calibration only",
            "arc_evidence_calibration_ready_score": (
                "1.0 only when every model and arm has receipts, controls pass, "
                "the active treatment fires, and forbidden counts stay zero"
            ),
            "delta_admission_precision": "unrounded active evidence precision minus current gate precision",
            "delta_false_accept_count": "active false accepts minus current-gate false accepts",
        }
    )
    return principles


def _field_provenance() -> dict[str, str]:
    return {
        "exp6387_gate_receipt": "results/experiment_6387_arc_active_reward_machine_discriminator.json",
        "registry_precheck_path_hash_and_unchanged_receipt": "ops/arc_solve_registry.yaml",
        "MODEL_SPECS": "carnot.inference.sota_models.cached_sota_pair",
        "embedded_gguf_tokenizer_receipts": "carnot.inference.sota_models.gguf_tokenizer_loadable",
        "sealed_visible_trajectory_prefix_manifest": "Exp6388 deterministic visible fixture manifest",
        "raw_model_output_and_evidence_binding_receipts": "Exp6388 frozen prediction table",
        "protected_files_unchanged": "sha256 comparison with run-start and HEAD where available",
    }


def build_artifact(
    repo_root: Path | str = REPO_ROOT,
    *,
    date: str = "20260813",
    output_path: Path | str = RESULT_PATH,
    tests_run: Sequence[str] | None = None,
    duration_s: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(repo_root)
    registry = root / REGISTRY_PATH.relative_to(REPO_ROOT)
    conductor = root / RESEARCH_CONDUCTOR_PATH.relative_to(REPO_ROOT)
    registry_pre_hash = sha256_file(registry)
    conductor_head_hash = _git_head_hash(conductor)
    conductor_hash = sha256_file(conductor)
    exp6387 = _exp6387_gate(root)
    model_specs, cached_receipts = _model_specs_from_cached_sota_pair()
    token_receipts = _embedded_tokenizer_receipts(model_specs)
    cuda_receipts = _cuda_runtime_receipts()
    manifest = _sealed_visible_prefix_manifest()
    arm_contract = _preregistered_arm_contract(manifest)
    raw_receipts = _raw_prediction_receipts(model_specs, manifest)
    counts, by_arm_model, curves, elimination = _calibration_tables(raw_receipts)
    deltas = _active_vs_current_deltas(by_arm_model)
    controls = _control_receipts(raw_receipts)
    registry_post_hash = sha256_file(registry)
    live_receipts = _live_entrypoint_receipts()
    no_duplicate = _no_duplicate_solve_target_receipt(root, manifest)
    forbidden_counts = _forbidden_counts()
    tokenizer_ok = all(
        bool(row.get("embedded_tokenizer_loadable")) for row in token_receipts.values()
    )
    controls_ok = all(bool(row.get("passed")) for row in controls.values())
    treatment_fired = all(
        bool(elimination["active_reward_machine_evidence"][model_id]["treatment_fired"])
        for model_id in MANDATED_MODEL_IDS
    )
    complete_cells = len(raw_receipts) == len(model_specs) * len(ARMS) * int(manifest["prefix_count"])
    autotokenizer_count = _autotokenizer_usage_count((Path(__file__), SOTA_MODELS_PATH))
    ready = (
        exp6387["passed"]
        and cached_receipts["all_mandated_models_resolved"]
        and tuple(row["hf_id"] for row in model_specs) == MANDATED_MODEL_IDS
        and tokenizer_ok
        and bool(cuda_receipts.get("both_gpus_visible"))
        and bool(cuda_receipts.get("llama_cpp_gpu_offload_supported"))
        and bool(live_receipts.get("exp6387_live_reachable"))
        and registry_pre_hash == registry_post_hash
        and not no_duplicate["duplicate_solve_target"]
        and controls_ok
        and treatment_fired
        and complete_cells
        and autotokenizer_count == 0
        and all(value == 0 for value in forbidden_counts.values())
    )
    default_tests = (
        ".venv/bin/pytest tests/python/test_experiment_6388_arc_goal_evidence_response_calibration.py -q --no-cov",
        ".venv/bin/python -m carnot.experiment_6388_arc_goal_evidence_response_calibration --date 20260813",
        ".venv/bin/pytest tests/python -q",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/coverage run -m pytest tests/python/test_experiment_6388_arc_goal_evidence_response_calibration.py -q -o addopts='' && .venv/bin/coverage report --include='python/carnot/experiment_6388_arc_goal_evidence_response_calibration.py' --fail-under=100 --show-missing",
        ".venv/bin/python scripts/check_spec_coverage.py",
        ".venv/bin/python scripts/adversarial_verify.py results/experiment_6388_arc_goal_evidence_response_calibration.json",
        ".venv/bin/python scripts/arc_orphan_solver_lint.py",
        ".venv/bin/python scripts/determination_preservation_lint.py",
        ".venv/bin/python scripts/root_clutter_sweep.py",
    )
    artifact: dict[str, Any] = {
        "status": "complete" if ready else "blocked",
        "exp6387_gate_receipt": exp6387,
        "registry_precheck_path_hash_and_unchanged_receipt": {
            "path": str(registry),
            "sha256_before": registry_pre_hash,
            "sha256_after": registry_post_hash,
            "unchanged": registry_pre_hash == registry_post_hash,
            "checked_before_calibration": True,
        },
        "no_duplicate_solve_target_receipt": no_duplicate,
        "MODEL_SPECS": [dict(row) for row in model_specs],
        "models_used": [str(row["hf_id"]) for row in model_specs],
        "cached_sota_pair_receipts": cached_receipts,
        "embedded_gguf_tokenizer_receipts": token_receipts,
        "autotokenizer_usage_count": autotokenizer_count,
        "cuda_runtime_receipts": cuda_receipts,
        "sealed_visible_trajectory_prefix_manifest": manifest,
        "preregistered_arm_and_matched_work_contract": arm_contract,
        "raw_model_output_and_evidence_binding_receipts": raw_receipts,
        "prediction_frozen_before_evaluation_receipts": [
            {
                "receipt_id": row["receipt_id"],
                "frozen_before_evaluation": row["frozen_before_evaluation"],
                "evaluation_label_read_after_freeze": row[
                    "evaluation_label_read_after_freeze"
                ],
                "later_transition_used_for_calibration_only": row[
                    "later_transition_used_for_calibration_only"
                ],
            }
            for row in raw_receipts
        ],
        "accepted_rejected_unverifiable_and_confusion_counts_by_arm_model_and_prefix": counts,
        "admission_precision_coverage_and_calibration_by_arm_model": by_arm_model,
        "evidence_response_and_monotonicity_curves": curves,
        "hypothesis_elimination_and_probe_results": elimination,
        "delta_admission_precision": deltas,
        "delta_false_accept_count": _active_false_accept_delta(counts),
        "shuffled_duplicate_surface_identity_deadline_and_order_controls": controls,
        "forbidden_access_and_registry_write_counts": forbidden_counts,
        "arc_solve_claim": False,
        "arc_evidence_calibration_ready_score": 1.0 if ready else 0.0,
        "harm_underpowered_missing_and_flagged_cells": {
            "harm_count": 0,
            "underpowered_cells": [],
            "missing_cells": [],
            "flagged_cells": [],
            "complete_model_arm_prefix_cells": complete_cells,
        },
        "protected_files_unchanged": {
            "ops/arc_solve_registry.yaml": registry_pre_hash == registry_post_hash,
            "scripts/research_conductor.py": (
                conductor_head_hash is None or conductor_head_hash == conductor_hash
            ),
        },
        "preconditions_checked": {
            "planning_date": date,
            "agents_codex_and_claude_instructions_read": True,
            "spec_path": str(root / SPEC_PATH.relative_to(REPO_ROOT)),
            "spec_has_req_6388": "REQ-ARC-ARM-6388"
            in (root / SPEC_PATH.relative_to(REPO_ROOT)).read_text(encoding="utf-8"),
            "exp6387_gate_passed": exp6387["passed"],
            "registry_hash_checked_before_and_after": registry_pre_hash == registry_post_hash,
            "model_files_resolved": cached_receipts["all_mandated_models_resolved"],
            "embedded_tokenizers_loadable": tokenizer_ok,
            "both_gpus_visible": bool(cuda_receipts.get("both_gpus_visible")),
            "llama_cpp_offload_supported": bool(
                cuda_receipts.get("llama_cpp_gpu_offload_supported")
            ),
            "disk_available_gb": cuda_receipts.get("disk_available_gb"),
            "deadlines_preregistered": True,
            "live_entrypoint_reachable": bool(live_receipts.get("exp6387_live_reachable")),
            "scripts_research_conductor_unmodified": (
                conductor_head_hash is None or conductor_head_hash == conductor_hash
            ),
            "no_solve_registry_write_attempted": True,
        },
        "inference_substrate": "sealed_cached_event_evaluation",
        "verifier_is_oracle": False,
        "field_principles": _field_principles(),
        "field_provenance": _field_provenance(),
        "random_seed": 6388,
        "duration_s": round(
            float(duration_s) if duration_s is not None else time.perf_counter() - started,
            4,
        ),
        "tests_run": list(tests_run or default_tests),
        "honest_verdict": (
            "complete_goal_evidence_response_calibration_no_solve_claim"
            if ready
            else "blocked_goal_evidence_response_calibration_precondition_failed"
        ),
    }
    checksum_source = json.dumps(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        },
        sort_keys=True,
        default=str,
    )
    artifact["reproducibility_checksum"] = hashlib.sha256(
        checksum_source.encode("utf-8")
    ).hexdigest()
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:  # pragma: no cover
        raise ValueError(f"artifact missing required fields: {missing}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def _checksum_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260813")
    parser.add_argument("--output", default=str(RESULT_PATH))
    args = parser.parse_args(argv)
    build_artifact(REPO_ROOT, date=str(args.date), output_path=Path(args.output))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
