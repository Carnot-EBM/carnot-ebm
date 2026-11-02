"""Build the Exp 4254 v393 oracle-distinct frontier capstone.

Spec refs: REQ-CAPSTONE-4254, SCENARIO-CAPSTONE-4254.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_4254_capstone_v393.json")
EXPERIMENT_ID = 4254
RANDOM_SEED = 4254
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
MIN_HELD_OUT_TASK_N = 40
MIN_TOTAL_ARC_LEVELS = 18
BASELINE_392_POSITIVE_CANDIDATE_N = 20
BASELINE_392_WRONG_MAJORITY_N = 9
SPEC_REFS = ["REQ-CAPSTONE-4254", "SCENARIO-CAPSTONE-4254"]


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4243_pool": Upstream(4243, Path("results/experiment_4243_arc_candidate_pool_grow.json")),
    "4244_build": Upstream(
        4244, Path("results/experiment_4244_arc_set_encoder_aggregator_build.json")
    ),
    "4244_model": Upstream(
        4244, Path("results/experiment_4244_arc_set_encoder_aggregator_model.json")
    ),
    "4245_arc_gate": Upstream(
        4245, Path("results/experiment_4245_arc_set_encoder_beats_vote.json")
    ),
    "4246_code": Upstream(
        4246, Path("results/experiment_4246_code_oracle_distinct_replication.json")
    ),
    "4247_reward_retire": Upstream(
        4247, Path("results/experiment_4247_verifier_reward_offline_harness_retire_livelora.json")
    ),
    "4248_reward_offline": Upstream(
        4248, Path("results/experiment_4248_verifier_as_reward_offline_3arm.json")
    ),
    "4249_arc_progress": Upstream(
        4249, Path("results/experiment_4249_arc_incremental_progress.json")
    ),
    "4250_live_solver": Upstream(
        4250, Path("results/experiment_4250_arc_live_env_solver_accuracy.json")
    ),
    "4251_sota": Upstream(
        4251, Path("results/experiment_4251_sota_ingestion_set_encoder_offline_rft.json")
    ),
    "4252_registry": Upstream(
        4252, Path("results/experiment_4252_verifier_registry_gaps_hygiene.json")
    ),
    "4253_hardware": Upstream(4253, Path("results/experiment_4253_hardware_continuity.json")),
}

HEADLINE_OUTCOMES = {
    "arc_oracle_distinct_set_encoder_beats_vote_first_arc_win",
    "arc_oracle_distinct_ties_vote_at_power_on_grown_pool_real_bound",
    "oracle_distinct_code_robust_arc_still_data_bound",
    "verifier_reward_offline_real_label_carries_signal",
    "verifier_reward_offline_null_distillation",
    "verifier_reward_live_lora_retired_offline_pending",
}

ORACLE_DISTINCT_STATUSES = {
    "ARC-MOAT-WON",
    "TIES-AT-POWER-ON-GROWN-POOL",
    "CODE-ROBUST-ARC-BOUND",
    "NO-HEADROOM",
}

VERIFIER_AS_REWARD_STATUSES = {
    "OFFLINE-REAL",
    "OFFLINE-NULL",
    "INVALID-or-UNDERPOWERED",
    "LIVE-LORA-RETIRED-OFFLINE-PENDING",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "oracle_distinct_status",
    "verifier_as_reward_status",
    "diffusiongemma_gate_resolvable",
    "total_arc_levels_solved",
    "upstream_provenance",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'first ARC win', "
        "'stronger ties-at-power-on-grown-pool bound', or "
        "'code-robust-ARC-still-bound' is COMPLETE and decision-grade -- it "
        "tells .394 whether the oracle-distinct frontier is reached on the "
        "north-star domain."
    ),
    "headline_outcome": (
        "One of the enumerated set -- forces a single unambiguous read of the "
        "oracle-distinct frontier (ARC + code) + the verifier-as-reward pivot "
        "after .393."
    ),
    "oracle_distinct_status": (
        "ARC-MOAT-WON (set-encoder beats vote off-oracle, CI excl 0, n>=40) / "
        "TIES-AT-POWER-ON-GROWN-POOL (real selection bound, data-sparsity "
        "removed) / CODE-ROBUST-ARC-BOUND / NO-HEADROOM -- the 2026-06-14 P0 "
        "directive's standing after .393; a circular result does NOT count here."
    ),
    "verifier_as_reward_status": (
        "OFFLINE-REAL (A-vs-B label carries signal) / OFFLINE-NULL "
        "(distillation/spurious) / INVALID-or-UNDERPOWERED / "
        "LIVE-LORA-RETIRED-OFFLINE-PENDING -- the owed 2026-06-11 pivot's "
        "standing after .393."
    ),
    "diffusiongemma_gate_resolvable": (
        "BARE bool: true ONLY if an ARC oracle-distinct win landed with a "
        "matched control (verifier_is_oracle=false, CI95-excl-0); a code-only "
        "win + ARC tie keeps the north-star-domain gate STILL-PENDING."
    ),
    "total_arc_levels_solved": "The monotonic ARC progress metric after .393 (must be >= 18).",
    "upstream_provenance": (
        "{experiment_id, fields_imported, sha256} per cited upstream; the audit "
        "trail that a capstone synthesizes nothing from nothing."
    ),
}


def read_json_object(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def selected_upstream_paths(root: Path | str) -> dict[str, Path | None]:
    root_path = Path(root)
    return {
        key: path if (path := root_path / upstream.path).exists() else None
        for key, upstream in DEFAULT_UPSTREAMS.items()
    }


def flagged(payload: Mapping[str, Any] | None) -> bool:
    return isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True


def verdict_text(payload: Mapping[str, Any] | None) -> str:
    value = payload.get("honest_verdict") if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def bool_metric(payload: Mapping[str, Any] | None, field: str) -> bool | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, bool) else None


def int_metric(payload: Mapping[str, Any] | None, field: str) -> int:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def float_metric(payload: Mapping[str, Any] | None, field: str) -> float | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def str_metric(payload: Mapping[str, Any] | None, field: str) -> str:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def nested_map(payload: Mapping[str, Any] | None, field: str) -> Mapping[str, Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def ci95(payload: Mapping[str, Any] | None, field: str) -> list[float] | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    low, high = value
    if not isinstance(low, (int, float)) or isinstance(low, bool):
        return None
    if not isinstance(high, (int, float)) or isinstance(high, bool):
        return None
    return [float(low), float(high)]


def ci_excludes_zero(interval: Sequence[float] | None) -> bool:
    return interval is not None and (interval[0] > 0.0 or interval[1] < 0.0)


def ci_includes_zero(interval: Sequence[float] | None) -> bool:
    return interval is not None and interval[0] <= 0.0 <= interval[1]


def _matched_control_present(payload: Mapping[str, Any] | None) -> bool:
    return (
        bool_metric(payload, "matched_control") is True
        or bool_metric(payload, "matched_control_present") is True
        or float_metric(payload, "matched_control_delta") is not None
        or bool(str_metric(payload, "matched_control_policy"))
    )


def _headroom_present(payload: Mapping[str, Any] | None) -> bool:
    value = bool_metric(payload, "headroom_present")
    if value is None:
        value = bool_metric(payload, "headroom_exists")
    if value is not None:
        return value
    oracle_minus_vote = float_metric(payload, "oracle_minus_vote")
    return oracle_minus_vote is not None and oracle_minus_vote > 0.0


def _blocked(payload: Mapping[str, Any] | None) -> bool:
    return str_metric(payload, "status") == "blocked" or verdict_text(payload).startswith("blocked")


def wrong_majority_count(payload: Mapping[str, Any] | None) -> int:
    direct = int_metric(payload, "wrong_majority_n")
    if direct:
        return direct
    task_rows = payload.get("task_rows") if isinstance(payload, Mapping) else None
    if not isinstance(task_rows, Sequence) or isinstance(task_rows, (str, bytes)):
        return 0
    return sum(
        1
        for row in task_rows
        if isinstance(row, Mapping)
        and row.get("vote_correct") is False
        and row.get("oracle_hit") is True
    )


def pool_growth(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "arc_pool_grown": False}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "arc_pool_grown": False}
    positive_n = int_metric(payload, "positive_candidate_n")
    wrong_n = int_metric(payload, "wrong_majority_n")
    held_out_n = int_metric(payload, "held_out_task_n")
    model_specs = nested_map(payload, "model_specs")
    baseline = nested_map(model_specs, "baseline_392")
    positive_baseline = int_metric(baseline, "positive_candidate_n") or (
        BASELINE_392_POSITIVE_CANDIDATE_N
    )
    wrong_baseline = int_metric(baseline, "wrong_majority_n") or BASELINE_392_WRONG_MAJORITY_N
    return {
        "status": "included",
        "arc_pool_grown": bool_metric(payload, "arc_pool_grown") is True,
        "positive_candidate_n": positive_n,
        "wrong_majority_n": wrong_n,
        "held_out_task_n": held_out_n,
        "grew_over_392_positive_candidate_baseline": positive_n > positive_baseline,
        "grew_over_392_wrong_majority_baseline": wrong_n > wrong_baseline,
        "baseline_392_positive_candidate_n": positive_baseline,
        "baseline_392_wrong_majority_n": wrong_baseline,
        "pool_artifact_path": str_metric(payload, "pool_artifact_path"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": verdict_text(payload),
    }


def set_encoder_build(
    build_payload: Mapping[str, Any] | None,
    model_payload: Mapping[str, Any] | None,
    *,
    was_build_skipped: bool,
    was_model_skipped: bool,
) -> JsonDict:
    if was_build_skipped or was_model_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "off_fold_auroc": None,
            "set_encoder_vs_logistic_auroc_delta": None,
            "beat_logistic_ablation": False,
        }
    if not isinstance(build_payload, Mapping):
        return {
            "status": "missing",
            "off_fold_auroc": None,
            "set_encoder_vs_logistic_auroc_delta": None,
            "beat_logistic_ablation": False,
        }
    model_oof = nested_map(model_payload, "set_encoder_oof")
    logistic = nested_map(model_payload, "logistic_ablation")
    off_fold_auroc = float_metric(build_payload, "oracle_distinct_auroc")
    if off_fold_auroc is None:
        off_fold_auroc = float_metric(model_oof, "auroc")
    logistic_auroc = float_metric(build_payload, "logistic_auroc")
    if logistic_auroc is None:
        logistic_auroc = float_metric(logistic, "auroc")
    delta = float_metric(build_payload, "set_encoder_vs_logistic_auroc_delta")
    if delta is None and off_fold_auroc is not None and logistic_auroc is not None:
        delta = off_fold_auroc - logistic_auroc
    return {
        "status": "included",
        "aggregator_trained": bool_metric(build_payload, "aggregator_trained") is True,
        "off_fold_auroc": off_fold_auroc,
        "oracle_distinct_auroc_ci95": ci95(build_payload, "oracle_distinct_auroc_ci95")
        or ci95(model_oof, "ci95"),
        "logistic_auroc": logistic_auroc,
        "logistic_auroc_ci95": ci95(build_payload, "logistic_auroc_ci95")
        or ci95(logistic, "ci95"),
        "set_encoder_vs_logistic_auroc_delta": delta,
        "beat_logistic_ablation": delta is not None and delta > 0.0,
        "positive_candidate_n": int_metric(build_payload, "positive_candidate_n"),
        "wrong_majority_n": int_metric(build_payload, "wrong_majority_n"),
        "held_out_task_n": int_metric(build_payload, "held_out_task_n"),
        "learned_verifier_path": str_metric(build_payload, "learned_verifier_path"),
        "verifier_is_oracle": bool_metric(build_payload, "verifier_is_oracle"),
        "model_type": str_metric(model_payload, "model_type"),
        "model_artifact_status": "included" if isinstance(model_payload, Mapping) else "missing",
        "honest_verdict": verdict_text(build_payload),
    }


def arc_set_encoder_gate(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "arc_status": "NO-HEADROOM",
            "oracle_distinct_beats_vote": False,
            "gate_ran": False,
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "missing",
            "arc_status": "NO-HEADROOM",
            "oracle_distinct_beats_vote": False,
            "gate_ran": False,
        }
    interval = ci95(payload, "set_encoder_minus_vote_ci95")
    delta = float_metric(payload, "set_encoder_minus_vote_delta")
    beat_flag = bool_metric(payload, "oracle_distinct_beats_vote")
    held_out_n = int_metric(payload, "held_out_task_n")
    verifier_is_oracle = bool_metric(payload, "verifier_is_oracle")
    matched_control = _matched_control_present(payload)
    headroom = _headroom_present(payload)
    powered = held_out_n >= MIN_HELD_OUT_TASK_N
    gate_ran = not _blocked(payload) and (
        delta is not None or interval is not None or beat_flag is not None
    )
    won = (
        verifier_is_oracle is False
        and matched_control
        and headroom
        and powered
        and gate_ran
        and ci_excludes_zero(interval)
        and (beat_flag is True or (delta is not None and delta > 0.0))
    )
    tied = (
        verifier_is_oracle is False
        and matched_control
        and headroom
        and powered
        and gate_ran
        and not won
        and ci_includes_zero(interval)
    )
    if won:
        status = "ARC-MOAT-WON"
    elif tied:
        status = "TIES-AT-POWER-ON-GROWN-POOL"
    else:
        status = "NO-HEADROOM"
    return {
        "status": "included",
        "arc_status": status,
        "oracle_distinct_beats_vote": won,
        "gate_ran": gate_ran,
        "verifier_is_oracle": verifier_is_oracle,
        "matched_control_present": matched_control,
        "headroom_present": headroom,
        "powered_held_out_n": powered,
        "held_out_task_n": held_out_n,
        "set_encoder_minus_vote_delta": delta,
        "set_encoder_minus_vote_ci95": interval,
        "ci95_excludes_zero": ci_excludes_zero(interval),
        "margin_override_minus_vote": float_metric(payload, "margin_override_minus_vote"),
        "matched_control_delta": float_metric(payload, "matched_control_delta"),
        "matched_control_policy": str_metric(payload, "matched_control_policy"),
        "oracle_at_k": float_metric(payload, "oracle_at_k"),
        "oracle_minus_vote": float_metric(payload, "oracle_minus_vote"),
        "wrong_majority_n": wrong_majority_count(payload),
        "pass_rates": dict(nested_map(payload, "pass_rates")),
        "candidate_count": int_metric(payload, "candidate_count"),
        "honest_verdict": verdict_text(payload),
    }


def code_replication(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "code_status": "NO-HEADROOM",
            "code_replication_beats_vote": False,
        }
    if not isinstance(payload, Mapping):
        return {"status": "missing", "code_status": "NO-HEADROOM", "code_replication_beats_vote": False}
    interval = ci95(payload, "code_predictor_minus_vote_ci95")
    delta = float_metric(payload, "code_predictor_minus_vote_delta")
    win_flag = bool_metric(payload, "code_replication_beats_vote")
    if win_flag is None:
        win_flag = bool_metric(payload, "code_oracle_distinct_beats_vote")
    verifier_is_oracle = bool_metric(payload, "verifier_is_oracle")
    matched_control = _matched_control_present(payload)
    headroom = _headroom_present(payload)
    held_out_n = int_metric(payload, "held_out_task_n")
    powered = held_out_n >= MIN_HELD_OUT_TASK_N
    replication_read = str_metric(payload, "replication_read")
    blocked = _blocked(payload) or replication_read.startswith("blocked")
    gate_ran = not blocked and (delta is not None or interval is not None or win_flag is not None)
    won = (
        verifier_is_oracle is False
        and matched_control
        and headroom
        and powered
        and gate_ran
        and ci_excludes_zero(interval)
        and (win_flag is True or (delta is not None and delta > 0.0))
    )
    tied = (
        verifier_is_oracle is False
        and matched_control
        and headroom
        and powered
        and gate_ran
        and not won
        and ci_includes_zero(interval)
    )
    if blocked:
        status = "BLOCKED"
    elif won:
        status = "CODE-ROBUST"
    elif tied:
        status = "CODE-BOUNDED"
    else:
        status = "NO-HEADROOM"
    return {
        "status": "included",
        "code_status": status,
        "code_replication_beats_vote": won,
        "gate_ran": gate_ran,
        "verifier_is_oracle": verifier_is_oracle,
        "matched_control_present": matched_control,
        "headroom_present": headroom,
        "powered_held_out_n": powered,
        "held_out_task_n": held_out_n,
        "code_predictor_minus_vote_delta": delta,
        "code_predictor_minus_vote_ci95": interval,
        "ci95_excludes_zero": ci_excludes_zero(interval),
        "matched_control_delta": float_metric(payload, "matched_control_delta"),
        "off_fold_auroc": float_metric(payload, "off_fold_auroc"),
        "oracle_at_k": float_metric(payload, "oracle_at_k"),
        "oracle_minus_vote": float_metric(payload, "oracle_minus_vote"),
        "replication_read": replication_read,
        "pass_rates": dict(nested_map(payload, "pass_rates")),
        "candidate_pool": dict(nested_map(payload, "candidate_pool")),
        "honest_verdict": verdict_text(payload),
    }


def verifier_as_reward(
    reward_payload: Mapping[str, Any] | None,
    retirement_payload: Mapping[str, Any] | None,
    registry_payload: Mapping[str, Any] | None,
    *,
    reward_skipped: bool,
    retirement_skipped: bool,
) -> JsonDict:
    registry_reward = nested_map(registry_payload, "verifier_reward_outcome")
    clean_reward = isinstance(reward_payload, Mapping) and not reward_skipped
    interval = ci95(reward_payload, "a_vs_b_ci95") if clean_reward else None
    delta = float_metric(reward_payload, "a_vs_b_delta") if clean_reward else None
    positive_control = (
        bool_metric(reward_payload, "positive_control_confirmed") is True if clean_reward else False
    )
    label_carries = (
        bool_metric(reward_payload, "verifier_label_carries_signal") is True
        if clean_reward
        else False
    )
    offline_real = (
        positive_control
        and label_carries
        and delta is not None
        and delta > 0.0
        and ci_excludes_zero(interval)
    )
    offline_null = positive_control and interval is not None and ci_includes_zero(interval)
    live_lora_retired_recorded = (
        bool_metric(registry_payload, "live_lora_retired_recorded") is True
        or bool_metric(registry_reward, "live_lora_retired") is True
    )
    if offline_real:
        status = "OFFLINE-REAL"
    elif offline_null:
        status = "OFFLINE-NULL"
    elif live_lora_retired_recorded:
        status = "LIVE-LORA-RETIRED-OFFLINE-PENDING"
    else:
        status = "INVALID-or-UNDERPOWERED"
    retirement_status = (
        "skipped_flagged_adversarial"
        if retirement_skipped
        else "included"
        if isinstance(retirement_payload, Mapping)
        else "missing"
    )
    return {
        "status": "included" if clean_reward else "skipped_flagged_adversarial" if reward_skipped else "missing",
        "verifier_as_reward_status": status,
        "offline_a_vs_b_ran": interval is not None,
        "a_vs_b_delta": delta,
        "a_vs_b_ci95": interval,
        "a_vs_b_ci_excludes_zero": ci_excludes_zero(interval),
        "verifier_label_carries_signal": label_carries if clean_reward else None,
        "positive_control_confirmed": positive_control if clean_reward else None,
        "blocked_at_layer": str_metric(reward_payload, "blocked_at_layer") if clean_reward else "",
        "gate_check_summary": str_metric(reward_payload, "gate_check_summary") if clean_reward else "",
        "live_lora_retired_recorded": live_lora_retired_recorded,
        "retirement_artifact_status": retirement_status,
        "retirement_artifact_skipped": retirement_skipped,
        "registry_status": str_metric(registry_reward, "status"),
        "honest_verdict": verdict_text(reward_payload) if clean_reward else "",
    }


def arc_progress(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "total_arc_levels_solved": 0}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "total_arc_levels_solved": 0}
    return {
        "status": "included",
        "total_arc_levels_solved": int_metric(payload, "total_levels_solved"),
        "total_arc_games_solved": int_metric(payload, "total_games_solved"),
        "new_levels_solved_this_task": int_metric(payload, "new_levels_solved_this_task"),
        "levels_completed": int_metric(payload, "levels_completed"),
        "prior_total_levels_solved": int_metric(payload, "prior_total_levels_solved"),
        "target_game": str_metric(payload, "target_game"),
        "target_level": int_metric(payload, "target_level"),
        "real_env_confirmed": bool_metric(payload, "real_env_confirmed") is True,
        "acceptance_gate_passed": bool_metric(payload, "acceptance_gate_passed") is True,
        "honest_verdict": verdict_text(payload),
    }


def live_solver_accuracy(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "solver_completes_level": False}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "solver_completes_level": False}
    metrics = nested_map(payload, "live_env_metrics")
    score = nested_map(metrics, "environment_score")
    beats = nested_map(payload, "solver_beats_floor")
    return {
        "status": "included",
        "solver_completes_level": bool_metric(payload, "solver_completes_level") is True,
        "levels_completed": int_metric(metrics, "levels_completed"),
        "observed_frame_levels_completed": int_metric(metrics, "observed_frame_levels_completed"),
        "score": float_metric(metrics, "score"),
        "scorecard_closed": bool_metric(payload, "scorecard_closed") is True,
        "scorecard_levels_completed": int_metric(score, "levels_completed"),
        "live_env_reachable": bool_metric(payload, "live_env_reachable") is True,
        "solver_beats_floor_accuracy": bool_metric(nested_map(beats, "accuracy"), "beats") is True,
        "solver_beats_floor_efficiency": bool_metric(nested_map(beats, "efficiency"), "beats")
        is True,
        "solver_beats_floor_overall": bool_metric(beats, "overall") is True,
        "honest_verdict": verdict_text(payload),
    }


def sota_v394(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "flagged_for_v394": ""}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "flagged_for_v394": ""}
    methods = payload.get("methods_mapped")
    first = (
        methods[0]
        if isinstance(methods, list) and methods and isinstance(methods[0], Mapping)
        else {}
    )
    return {
        "status": "included",
        "flagged_for_v394": str_metric(payload, "flagged_for_v394"),
        "strongest_method_name": str_metric(first, "name"),
        "strongest_method_url": str_metric(first, "url"),
        "methods_mapped_count": len(methods) if isinstance(methods, list) else 0,
        "honest_verdict": verdict_text(payload),
    }


def registry_hygiene(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "regression_guard_passed": False}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "regression_guard_passed": False}
    return {
        "status": "included",
        "regression_guard_passed": bool_metric(payload, "regression_guard_passed") is True,
        "oracle_distinct_status": str_metric(
            nested_map(payload, "oracle_distinct_outcome"), "status"
        ),
        "code_replication_status": str_metric(
            nested_map(payload, "code_replication_outcome"), "status"
        ),
        "verifier_reward_status": str_metric(
            nested_map(payload, "verifier_reward_outcome"), "status"
        ),
        "live_lora_retired_recorded": bool_metric(payload, "live_lora_retired_recorded") is True,
        "honest_verdict": verdict_text(payload),
    }


def hardware_continuity(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "kv260_terminal_confirmed": False}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "kv260_terminal_confirmed": False}
    return {
        "status": "included",
        "per_board_reachability": dict(nested_map(payload, "per_board_reachability")),
        "per_board_status": dict(nested_map(payload, "per_board_status")),
        "gatemate_step_taken": str_metric(payload, "gatemate_step_taken"),
        "polarfire_step_taken": str_metric(payload, "polarfire_step_taken"),
        "kv260_terminal_confirmed": bool_metric(payload, "kv260_terminal_confirmed") is True,
        "fabric_acceleration_claimed": bool_metric(payload, "fabric_acceleration_claimed") is True,
        "speedup_claim_made": bool_metric(payload, "speedup_claim_made") is True,
        "honest_verdict": verdict_text(payload),
    }


def oracle_distinct_status(arc_gate: Mapping[str, Any], code: Mapping[str, Any]) -> str:
    arc_status = str(arc_gate.get("arc_status", "NO-HEADROOM"))
    code_status = str(code.get("code_status", "NO-HEADROOM"))
    if arc_status == "ARC-MOAT-WON":
        return "ARC-MOAT-WON"
    if code_status == "CODE-ROBUST":
        return "CODE-ROBUST-ARC-BOUND"
    if arc_status == "TIES-AT-POWER-ON-GROWN-POOL":
        return "TIES-AT-POWER-ON-GROWN-POOL"
    return "NO-HEADROOM"


def diffusiongemma_resolvable(arc_gate: Mapping[str, Any]) -> bool:
    return bool(arc_gate.get("oracle_distinct_beats_vote"))


def headline_outcome(oracle_status: str, reward_status: str) -> str:
    if oracle_status == "ARC-MOAT-WON":
        return "arc_oracle_distinct_set_encoder_beats_vote_first_arc_win"
    if oracle_status == "CODE-ROBUST-ARC-BOUND":
        return "oracle_distinct_code_robust_arc_still_data_bound"
    if oracle_status == "TIES-AT-POWER-ON-GROWN-POOL":
        return "arc_oracle_distinct_ties_vote_at_power_on_grown_pool_real_bound"
    if reward_status == "OFFLINE-REAL":
        return "verifier_reward_offline_real_label_carries_signal"
    if reward_status == "OFFLINE-NULL":
        return "verifier_reward_offline_null_distillation"
    if reward_status == "LIVE-LORA-RETIRED-OFFLINE-PENDING":
        return "verifier_reward_live_lora_retired_offline_pending"
    return "arc_oracle_distinct_ties_vote_at_power_on_grown_pool_real_bound"


def honest_verdict(
    outcome: str,
    oracle_status: str,
    reward_status: str,
    total_levels: int,
    skipped_count: int,
    gate_resolvable: bool,
) -> str:
    gate = "resolvable" if gate_resolvable else "still_pending"
    return (
        f"complete: capstone_v393_{outcome}_oracle_{oracle_status}_"
        f"reward_{reward_status}_arc_levels{total_levels}_"
        f"flagged_skipped{skipped_count}_diffusiongemma_{gate}"
    )


def imported_fields_by_key(clean_keys: set[str]) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {key: [] for key in DEFAULT_UPSTREAMS}
    if "4243_pool" in clean_keys:
        fields["4243_pool"] = [
            "arc_pool_grown",
            "positive_candidate_n",
            "wrong_majority_n",
            "held_out_task_n",
            "pool_artifact_path",
            "verifier_is_oracle",
            "model_specs",
        ]
    if "4244_build" in clean_keys:
        fields["4244_build"] = [
            "aggregator_trained",
            "oracle_distinct_auroc",
            "oracle_distinct_auroc_ci95",
            "logistic_auroc",
            "logistic_auroc_ci95",
            "set_encoder_vs_logistic_auroc_delta",
            "positive_candidate_n",
            "wrong_majority_n",
            "held_out_task_n",
            "learned_verifier_path",
            "verifier_is_oracle",
        ]
    if "4244_model" in clean_keys:
        fields["4244_model"] = [
            "set_encoder_oof.auroc",
            "set_encoder_oof.ci95",
            "logistic_ablation",
            "model_type",
            "positive_candidate_n",
            "wrong_majority_n",
            "held_out_task_n",
            "verifier_is_oracle",
        ]
    if "4245_arc_gate" in clean_keys:
        fields["4245_arc_gate"] = [
            "verifier_is_oracle",
            "oracle_distinct_beats_vote",
            "set_encoder_minus_vote_delta",
            "set_encoder_minus_vote_ci95",
            "margin_override_minus_vote",
            "matched_control_delta",
            "matched_control_policy",
            "headroom_exists",
            "held_out_task_n",
            "oracle_at_k",
            "pass_rates",
            "task_rows",
        ]
    if "4246_code" in clean_keys:
        fields["4246_code"] = [
            "verifier_is_oracle",
            "code_replication_beats_vote",
            "code_predictor_minus_vote_delta",
            "code_predictor_minus_vote_ci95",
            "matched_control_delta",
            "headroom_exists",
            "held_out_task_n",
            "off_fold_auroc",
            "replication_read",
            "candidate_pool",
        ]
    if "4247_reward_retire" in clean_keys:
        fields["4247_reward_retire"] = [
            "harness_smoke_passed",
            "live_lora_retired",
            "steps_run",
            "trainable_param_count",
            "lora_attach_path",
            "loss_initial",
            "loss_final",
        ]
    if "4248_reward_offline" in clean_keys:
        fields["4248_reward_offline"] = [
            "a_vs_b_delta",
            "a_vs_b_ci95",
            "verifier_label_carries_signal",
            "positive_control_confirmed",
            "blocked_at_layer",
            "gate_check_summary",
            "status",
        ]
    if "4249_arc_progress" in clean_keys:
        fields["4249_arc_progress"] = [
            "total_levels_solved",
            "total_games_solved",
            "new_levels_solved_this_task",
            "levels_completed",
            "prior_total_levels_solved",
            "real_env_confirmed",
            "acceptance_gate_passed",
        ]
    if "4250_live_solver" in clean_keys:
        fields["4250_live_solver"] = [
            "solver_completes_level",
            "live_env_metrics",
            "solver_beats_floor",
            "live_env_reachable",
            "scorecard_closed",
        ]
    if "4251_sota" in clean_keys:
        fields["4251_sota"] = ["flagged_for_v394", "methods_mapped"]
    if "4252_registry" in clean_keys:
        fields["4252_registry"] = [
            "regression_guard_passed",
            "oracle_distinct_outcome.status",
            "code_replication_outcome.status",
            "verifier_reward_outcome.status",
            "live_lora_retired_recorded",
        ]
    if "4253_hardware" in clean_keys:
        fields["4253_hardware"] = [
            "per_board_reachability",
            "per_board_status",
            "gatemate_step_taken",
            "polarfire_step_taken",
            "kv260_terminal_confirmed",
            "fabric_acceleration_claimed",
            "speedup_claim_made",
        ]
    return fields


def flagged_artifacts_skipped(
    root: Path,
    paths: Mapping[str, Path | None],
    upstreams: Mapping[str, Mapping[str, Any] | None],
    skipped_keys: set[str],
) -> list[JsonDict]:
    rows = []
    for key, upstream in DEFAULT_UPSTREAMS.items():
        path = paths[key]
        if path is not None and key in skipped_keys:
            rows.append(
                {
                    "artifact_key": key,
                    "experiment_id": upstream.experiment_id,
                    "path": str(path.relative_to(root)),
                    "reason": "flagged_adversarial:true",
                    "sha256": sha256_file(path),
                    "honest_verdict": verdict_text(upstreams[key]),
                }
            )
    return rows


def upstream_provenance(
    root: Path,
    paths: Mapping[str, Path | None],
    upstreams: Mapping[str, Mapping[str, Any] | None],
    skipped_keys: set[str],
    fields_by_key: Mapping[str, list[str]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for key, upstream in DEFAULT_UPSTREAMS.items():
        path = paths[key]
        if path is None:
            continue
        skipped = key in skipped_keys
        rows.append(
            {
                "artifact_key": key,
                "experiment_id": upstream.experiment_id,
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "fields_imported": [] if skipped else list(fields_by_key.get(key, [])),
                "skipped": skipped,
                "skip_reason": "flagged_adversarial:true" if skipped else "",
                "honest_verdict": verdict_text(upstreams[key]),
            }
        )
    return rows


def missing_upstream_artifacts(paths: Mapping[str, Path | None]) -> list[JsonDict]:
    return [
        {"artifact_key": key, "experiment_id": upstream.experiment_id}
        for key, upstream in DEFAULT_UPSTREAMS.items()
        if paths[key] is None
    ]


def duration_from(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return max(0.0001, end - started_s)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable.pop("reproducibility_checksum", None)
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    paths = selected_upstream_paths(root_path)
    upstreams: dict[str, Mapping[str, Any] | None] = {
        key: read_json_object(path) if path is not None else None for key, path in paths.items()
    }
    skipped_keys = {key for key, payload in upstreams.items() if flagged(payload)}
    clean_keys = {
        key
        for key, payload in upstreams.items()
        if isinstance(payload, Mapping) and key not in skipped_keys
    }
    clean = {key: upstreams[key] for key in clean_keys}

    pool = pool_growth(clean.get("4243_pool"), was_skipped="4243_pool" in skipped_keys)
    build = set_encoder_build(
        clean.get("4244_build"),
        clean.get("4244_model"),
        was_build_skipped="4244_build" in skipped_keys,
        was_model_skipped="4244_model" in skipped_keys,
    )
    arc_gate = arc_set_encoder_gate(
        clean.get("4245_arc_gate"), was_skipped="4245_arc_gate" in skipped_keys
    )
    code = code_replication(clean.get("4246_code"), was_skipped="4246_code" in skipped_keys)
    registry = registry_hygiene(
        clean.get("4252_registry"), was_skipped="4252_registry" in skipped_keys
    )
    reward = verifier_as_reward(
        clean.get("4248_reward_offline"),
        clean.get("4247_reward_retire"),
        clean.get("4252_registry"),
        reward_skipped="4248_reward_offline" in skipped_keys,
        retirement_skipped="4247_reward_retire" in skipped_keys,
    )
    arc = arc_progress(
        clean.get("4249_arc_progress"), was_skipped="4249_arc_progress" in skipped_keys
    )
    live = live_solver_accuracy(
        clean.get("4250_live_solver"), was_skipped="4250_live_solver" in skipped_keys
    )
    sota = sota_v394(clean.get("4251_sota"), was_skipped="4251_sota" in skipped_keys)
    hardware = hardware_continuity(
        clean.get("4253_hardware"), was_skipped="4253_hardware" in skipped_keys
    )

    oracle_status = oracle_distinct_status(arc_gate, code)
    reward_status = str(reward["verifier_as_reward_status"])
    gate_resolvable = diffusiongemma_resolvable(arc_gate)
    outcome = headline_outcome(oracle_status, reward_status)
    total_levels = int(arc["total_arc_levels_solved"])
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_keys)
    fields_by_key = imported_fields_by_key(clean_keys)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v393_4254.v1",
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict(
            outcome, oracle_status, reward_status, total_levels, len(skipped), gate_resolvable
        ),
        "headline_outcome": outcome,
        "oracle_distinct_status": oracle_status,
        "pool_growth": pool,
        "set_encoder_build": build,
        "arc_set_encoder_gate": arc_gate,
        "code_replication": code,
        "verifier_as_reward_status": reward_status,
        "verifier_as_reward": reward,
        "diffusiongemma_gate_resolvable": gate_resolvable,
        "total_arc_levels_solved": total_levels,
        "arc_progress": arc,
        "live_solver_accuracy": live,
        "strongest_sota_flagged_for_v394": str(sota.get("flagged_for_v394") or ""),
        "sota_v394": sota,
        "registry_hygiene": registry,
        "hardware_continuity": hardware,
        "reading_results_discipline": {
            "summarizer": "scripts/summarize_artifact.py",
            "upstream_count": len(DEFAULT_UPSTREAMS),
            "flagged_adversarial_policy": "skip_before_importing_metrics",
        },
        "flagged_artifacts_skipped": skipped,
        "upstream_provenance": upstream_provenance(
            root_path, paths, upstreams, skipped_keys, fields_by_key
        ),
        "missing_upstream_artifacts": missing_upstream_artifacts(paths),
        "duration_s": duration_from(start, now_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("complete:", "success:", "blocked:")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")
    if artifact.get("headline_outcome") not in HEADLINE_OUTCOMES:
        raise ValueError("headline_outcome must be enumerated")
    if artifact.get("oracle_distinct_status") not in ORACLE_DISTINCT_STATUSES:
        raise ValueError("oracle_distinct_status must be enumerated")
    if artifact.get("verifier_as_reward_status") not in VERIFIER_AS_REWARD_STATUSES:
        raise ValueError("verifier_as_reward_status must be enumerated")
    if not isinstance(artifact.get("diffusiongemma_gate_resolvable"), bool):
        raise ValueError("DiffusionGemma gate flag must be a bool")
    if (
        artifact.get("diffusiongemma_gate_resolvable") is True
        and artifact.get("oracle_distinct_status") != "ARC-MOAT-WON"
    ):
        raise ValueError("DiffusionGemma gate is resolvable only on an ARC oracle-distinct win")
    total_levels = artifact.get("total_arc_levels_solved")
    if (
        not isinstance(total_levels, int)
        or isinstance(total_levels, bool)
        or total_levels < MIN_TOTAL_ARC_LEVELS
    ):
        raise ValueError(f"total ARC levels must be an integer >= {MIN_TOTAL_ARC_LEVELS}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be an object")
    for field, principle in FIELD_PRINCIPLES.items():
        if principles.get(field) != principle:
            raise ValueError(f"{field} principle mismatch")
    provenance = artifact.get("upstream_provenance")
    if not isinstance(provenance, list):
        raise ValueError("upstream_provenance must be a list")
    for row in provenance:
        if not isinstance(row, Mapping):
            raise ValueError("upstream_provenance entries must be objects")
        if not isinstance(row.get("artifact_key"), str):
            raise ValueError("upstream_provenance entries need artifact_key")
        if not isinstance(row.get("experiment_id"), int):
            raise ValueError("upstream_provenance entries need integer experiment_id")
        if not isinstance(row.get("fields_imported"), list):
            raise ValueError("upstream_provenance fields_imported must be a list")
        if row.get("skipped") is True and row.get("fields_imported") != []:
            raise ValueError("skipped upstreams must import no fields")
        if not is_sha256(row.get("sha256")):
            raise ValueError("upstream_provenance entries need sha256")
    if not isinstance(artifact.get("flagged_artifacts_skipped"), list):
        raise ValueError("flagged_artifacts_skipped must be a list")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be sha256")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    root_path = Path(root)
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    validate_artifact(artifact)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    output = write_artifact(REPO_ROOT)
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
