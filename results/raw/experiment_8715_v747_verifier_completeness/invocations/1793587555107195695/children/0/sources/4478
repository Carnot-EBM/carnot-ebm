"""Build the Exp 4241 v392 oracle-distinct frontier capstone.

Spec refs: REQ-CAPSTONE-4241, SCENARIO-CAPSTONE-4241.
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
OUTPUT_REL_PATH = Path("results/experiment_4241_capstone_v392.json")
EXPERIMENT_ID = 4241
RANDOM_SEED = 4241
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
BASELINE_AUROC_391 = 0.778980279
MIN_HELD_OUT_TASK_N = 30
MIN_TOTAL_ARC_LEVELS = 17
SPEC_REFS = ["REQ-CAPSTONE-4241", "SCENARIO-CAPSTONE-4241"]


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4208_detector": Upstream(
        4208, Path("results/experiment_4208_verifier_as_detector_auroc.json")
    ),
    "4231_build": Upstream(
        4231, Path("results/experiment_4231_oracle_distinct_arc_aggregator_build.json")
    ),
    "4231_model": Upstream(
        4231, Path("results/experiment_4231_oracle_distinct_arc_aggregator_model.json")
    ),
    "4232_arc_gate": Upstream(
        4232, Path("results/experiment_4232_oracle_distinct_arc_aggregator_beats_vote.json")
    ),
    "4233_code": Upstream(
        4233, Path("results/experiment_4233_oracle_distinct_code_beats_vote.json")
    ),
    "4234_smoke": Upstream(
        4234, Path("results/experiment_4234_verifier_reward_lora_harness_real_training_smoke.json")
    ),
    "4235_reward": Upstream(
        4235, Path("results/experiment_4235_verifier_as_reward_3arm_window_boxed.json")
    ),
    "4236_arc_progress": Upstream(
        4236, Path("results/experiment_4236_arc_incremental_progress.json")
    ),
    "4237_live_solver": Upstream(
        4237, Path("results/experiment_4237_arc_live_env_solver_accuracy.json")
    ),
    "4238_sota": Upstream(
        4238, Path("results/experiment_4238_sota_ingestion_cross_candidate_aggregator.json")
    ),
    "4239_registry": Upstream(
        4239, Path("results/experiment_4239_verifier_registry_gaps_hygiene.json")
    ),
    "4240_hardware": Upstream(4240, Path("results/experiment_4240_hardware_continuity.json")),
}

HEADLINE_OUTCOMES = {
    "oracle_distinct_aggregator_beats_vote_first_moat",
    "oracle_distinct_aggregator_ties_vote_at_power_stronger_null",
    "oracle_distinct_arc_null_is_data_sparsity_code_wins",
    "oracle_distinct_selection_thesis_bounded_both_tie",
    "verifier_reward_real_label_carries_signal",
    "verifier_reward_null_distillation",
    "verifier_reward_live_lora_retired",
}

ORACLE_DISTINCT_STATUSES = {
    "MOAT-WON",
    "TIES-AT-POWER-NULL",
    "ARC-NULL-IS-DATA-SPARSITY",
    "THESIS-BOUNDED",
    "NO-HEADROOM",
}

VERIFIER_AS_REWARD_STATUSES = {
    "REAL",
    "NULL",
    "INVALID-or-UNDERPOWERED",
    "RETIRED-LIVE-LORA",
    "HARNESS-DEFERRED",
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
        "Terminal-prefixed. An honest 'ties-at-power null' or "
        "'ARC-null-is-data-sparsity' or 'thesis-bounded' is COMPLETE and "
        "decision-grade -- it tells .393 whether the oracle-distinct frontier "
        "is reachable, and where."
    ),
    "headline_outcome": (
        "One of the enumerated set -- forces a single unambiguous read of the "
        "oracle-distinct frontier (ARC + code) + the verifier-as-reward pivot."
    ),
    "oracle_distinct_status": (
        "MOAT-WON (aggregator beats vote off-oracle, CI excl 0, n>=30) / "
        "TIES-AT-POWER-NULL / ARC-NULL-IS-DATA-SPARSITY (code wins) / "
        "THESIS-BOUNDED (both tie) / NO-HEADROOM -- the 2026-06-14 P0 "
        "directive's standing after .392; a circular result does NOT count here."
    ),
    "verifier_as_reward_status": (
        "REAL (A-vs-B label carries signal) / NULL (distillation/spurious) / "
        "INVALID-or-UNDERPOWERED / RETIRED-LIVE-LORA / HARNESS-DEFERRED -- the "
        "owed 2026-06-11 pivot's standing after .392."
    ),
    "diffusiongemma_gate_resolvable": (
        "BARE bool: true ONLY if an oracle-distinct win landed with a matched "
        "control (verifier_is_oracle=false, CI95-excl-0); a circular execution "
        "win keeps it STILL-PENDING."
    ),
    "total_arc_levels_solved": "The monotonic ARC progress metric after .392 (must be >= 17).",
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


def rank_auc(rows: Sequence[Mapping[str, Any]]) -> float | None:
    pairs: list[tuple[float, bool]] = []
    for row in rows:
        score = float_metric(row, "score")
        label = bool_metric(row, "correct")
        if score is not None and label is not None:
            pairs.append((score, label))
    positives = sum(1 for _, label in pairs if label)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        next_index = index + 1
        while next_index < len(pairs) and pairs[next_index][0] == pairs[index][0]:
            next_index += 1
        average_rank = (index + 1 + next_index) / 2.0
        rank_sum += average_rank * sum(1 for _, label in pairs[index:next_index] if label)
        index = next_index
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def wrong_majority_count(gate_payload: Mapping[str, Any] | None) -> int:
    task_rows = gate_payload.get("task_rows") if isinstance(gate_payload, Mapping) else None
    if not isinstance(task_rows, Sequence) or isinstance(task_rows, (str, bytes)):
        return 0
    return sum(
        1
        for row in task_rows
        if isinstance(row, Mapping)
        and row.get("vote_correct") is False
        and row.get("oracle_hit") is True
    )


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


def detector_selection_divergence(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial"}
    if not isinstance(payload, Mapping):
        return {"status": "missing"}
    return {
        "status": "included",
        "detection_auroc_by_domain": dict(nested_map(payload, "detection_auroc_by_domain")),
        "detection_auroc_ci95_by_domain": dict(
            nested_map(payload, "detection_auroc_ci95_by_domain")
        ),
        "selector_headroom_by_domain": dict(nested_map(payload, "selector_headroom_by_domain")),
        "verifier_is_oracle_by_domain": dict(nested_map(payload, "verifier_is_oracle_by_domain")),
        "n_by_domain": dict(nested_map(payload, "n_by_domain")),
        "divergence_domains": list(payload.get("divergence_domains", []))
        if isinstance(payload.get("divergence_domains"), list)
        else [],
        "honest_verdict": verdict_text(payload),
    }


def arc_aggregator_model(
    model_payload: Mapping[str, Any] | None,
    build_payload: Mapping[str, Any] | None,
    *,
    was_model_skipped: bool,
    was_build_skipped: bool,
) -> JsonDict:
    build_status = "skipped_flagged_adversarial" if was_build_skipped else "not_skipped"
    if was_model_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "build_artifact_status": build_status,
            "off_fold_auroc": None,
            "held_out_task_n": 0,
            "improved_over_391": False,
        }
    if not isinstance(model_payload, Mapping):
        return {
            "status": "missing",
            "build_artifact_status": build_status,
            "off_fold_auroc": None,
            "held_out_task_n": 0,
            "improved_over_391": False,
        }
    raw_rows = model_payload.get("oof_rows")
    rows = (
        [row for row in raw_rows if isinstance(row, Mapping)]
        if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes))
        else []
    )
    auroc = rank_auc(rows)
    held_out_n = int_metric(model_payload, "held_out_task_n") or int_metric(
        build_payload, "held_out_task_n"
    )
    return {
        "status": "included",
        "build_artifact_status": build_status,
        "metric_source": (
            "computed_from_clean_4231_model_oof_rows"
            if was_build_skipped
            else "clean_4231_build_and_model"
        ),
        "off_fold_auroc": auroc,
        "held_out_task_n": held_out_n,
        "baseline_auroc_391": BASELINE_AUROC_391,
        "improved_over_391": auroc is not None and auroc > BASELINE_AUROC_391,
        "wrong_majority_n": int_metric(build_payload, "wrong_majority_n"),
        "positive_candidate_n": int_metric(build_payload, "positive_candidate_n"),
        "accepted_rejected_n": dict(nested_map(model_payload, "accepted_rejected_n")),
        "oof_row_n": len(rows),
        "model_type": str_metric(model_payload, "model_type"),
        "verifier_is_oracle": bool_metric(model_payload, "verifier_is_oracle"),
        "build_honest_verdict": verdict_text(build_payload) if not was_build_skipped else "",
    }


def arc_aggregator_gate(
    gate_payload: Mapping[str, Any] | None,
    model_report: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    if was_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "arc_status": "NO-HEADROOM",
            "oracle_distinct_beats_vote": False,
            "gate_ran": False,
        }
    if not isinstance(gate_payload, Mapping):
        return {
            "status": "missing",
            "arc_status": "NO-HEADROOM",
            "oracle_distinct_beats_vote": False,
            "gate_ran": False,
        }
    interval = ci95(gate_payload, "aggregator_minus_vote_ci95") or ci95(
        gate_payload, "verifier_minus_vote_ci95"
    )
    delta = float_metric(gate_payload, "aggregator_minus_vote_delta")
    if delta is None:
        delta = float_metric(gate_payload, "verifier_minus_vote_delta")
    blocked = str_metric(gate_payload, "status") == "blocked" or verdict_text(
        gate_payload
    ).startswith("blocked")
    beat_flag = bool_metric(gate_payload, "oracle_distinct_beats_vote")
    held_out_n = int_metric(gate_payload, "held_out_task_n") or int_metric(
        model_report, "held_out_task_n"
    )
    verifier_is_oracle = bool_metric(gate_payload, "verifier_is_oracle")
    matched_control = _matched_control_present(gate_payload)
    headroom = _headroom_present(gate_payload)
    powered = held_out_n >= MIN_HELD_OUT_TASK_N
    gate_ran = not blocked and (delta is not None or interval is not None or beat_flag is not None)
    moat_won = (
        verifier_is_oracle is False
        and matched_control
        and headroom
        and powered
        and gate_ran
        and ci_excludes_zero(interval)
        and (beat_flag is True or (delta is not None and delta > 0.0))
    )
    ties_at_power = (
        verifier_is_oracle is False
        and matched_control
        and headroom
        and powered
        and gate_ran
        and not moat_won
        and ci_includes_zero(interval)
    )
    if moat_won:
        status = "MOAT-WON"
    elif ties_at_power:
        status = "TIES-AT-POWER-NULL"
    else:
        status = "NO-HEADROOM"
    return {
        "status": "included",
        "arc_status": status,
        "oracle_distinct_beats_vote": moat_won,
        "gate_ran": gate_ran,
        "verifier_is_oracle": verifier_is_oracle,
        "matched_control_present": matched_control,
        "headroom_present": headroom,
        "powered_held_out_n": powered,
        "held_out_task_n": held_out_n,
        "aggregator_minus_vote_delta": delta,
        "aggregator_minus_vote_ci95": interval,
        "ci95_excludes_zero": ci_excludes_zero(interval),
        "margin_override_minus_vote": float_metric(gate_payload, "margin_override_minus_vote"),
        "matched_control_delta": float_metric(gate_payload, "matched_control_delta"),
        "oracle_at_k": float_metric(gate_payload, "oracle_at_k"),
        "oracle_minus_vote": float_metric(gate_payload, "oracle_minus_vote"),
        "wrong_majority_n": wrong_majority_count(gate_payload),
        "pass_rates": dict(nested_map(gate_payload, "pass_rates")),
        "candidate_count": int_metric(gate_payload, "candidate_count"),
        "honest_verdict": verdict_text(gate_payload),
    }


def code_disambiguation(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "code_status": "NO-HEADROOM"}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "code_status": "NO-HEADROOM"}
    interval = ci95(payload, "code_predictor_minus_vote_ci95")
    delta = float_metric(payload, "code_predictor_minus_vote_delta")
    verifier_is_oracle = bool_metric(payload, "verifier_is_oracle")
    matched_control = _matched_control_present(payload)
    headroom = _headroom_present(payload)
    held_out_n = int_metric(payload, "held_out_task_n")
    blocked = str_metric(payload, "status") == "blocked" or verdict_text(payload).startswith(
        "blocked"
    )
    win_flag = bool_metric(payload, "code_oracle_distinct_beats_vote")
    powered = held_out_n >= MIN_HELD_OUT_TASK_N
    gate_ran = not blocked and (delta is not None or interval is not None or win_flag is not None)
    code_won = (
        verifier_is_oracle is False
        and matched_control
        and headroom
        and powered
        and gate_ran
        and ci_excludes_zero(interval)
        and (win_flag is True or (delta is not None and delta > 0.0))
    )
    code_ties = (
        verifier_is_oracle is False
        and matched_control
        and headroom
        and powered
        and gate_ran
        and not code_won
        and ci_includes_zero(interval)
    )
    code_status = "CODE-WON" if code_won else "CODE-TIES" if code_ties else "NO-HEADROOM"
    return {
        "status": "included",
        "code_status": code_status,
        "code_oracle_distinct_beats_vote": code_won,
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
        "pass_rates": dict(nested_map(payload, "pass_rates")),
        "candidate_pool": dict(nested_map(payload, "candidate_pool")),
        "disambiguation_read": str_metric(payload, "disambiguation_read"),
        "honest_verdict": verdict_text(payload),
    }


def _b1_smoke(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "harness_smoke_passed": False}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "harness_smoke_passed": False}
    trainable = int_metric(payload, "trainable_param_count")
    steps = int_metric(payload, "steps_run")
    loss_initial = float_metric(payload, "loss_initial")
    loss_final = float_metric(payload, "loss_final")
    loss_moved = loss_initial is not None and loss_final is not None and loss_final < loss_initial
    passed = (
        bool_metric(payload, "harness_smoke_passed") is True
        and trainable > 0
        and steps >= 20
        and loss_moved
    )
    return {
        "status": "included",
        "harness_smoke_passed": passed,
        "reported_harness_smoke_passed": bool_metric(payload, "harness_smoke_passed") is True,
        "steps_run": steps,
        "trainable_param_count": trainable,
        "lora_attach_path": str_metric(payload, "lora_attach_path"),
        "loss_initial": loss_initial,
        "loss_final": loss_final,
        "smoke_failure_reason": str_metric(payload, "smoke_failure_reason"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": verdict_text(payload),
    }


def verifier_as_reward(
    reward_payload: Mapping[str, Any] | None,
    smoke_payload: Mapping[str, Any] | None,
    registry_payload: Mapping[str, Any] | None,
    *,
    was_reward_skipped: bool,
    was_smoke_skipped: bool,
) -> JsonDict:
    smoke = _b1_smoke(smoke_payload, was_skipped=was_smoke_skipped)
    registry_reward = nested_map(registry_payload, "verifier_reward_outcome")
    clean_reward = isinstance(reward_payload, Mapping) and not was_reward_skipped
    live_lora_retired = (
        (bool_metric(reward_payload, "live_lora_retired") is True if clean_reward else False)
        or bool_metric(registry_reward, "live_lora_retired") is True
    )
    if live_lora_retired:
        status = "RETIRED-LIVE-LORA"
    elif smoke["harness_smoke_passed"] is not True:
        status = "HARNESS-DEFERRED"
    elif was_reward_skipped or not isinstance(reward_payload, Mapping):
        status = "INVALID-or-UNDERPOWERED"
    else:
        interval = ci95(reward_payload, "a_vs_b_ci95")
        delta = float_metric(reward_payload, "a_vs_b_delta")
        positive_control = bool_metric(reward_payload, "positive_control_confirmed") is True
        label_carries = bool_metric(reward_payload, "verifier_label_carries_signal") is True
        if (
            positive_control
            and label_carries
            and delta is not None
            and delta > 0.0
            and ci_excludes_zero(interval)
        ):
            status = "REAL"
        elif positive_control and interval is not None and ci_includes_zero(interval):
            status = "NULL"
        else:
            status = "INVALID-or-UNDERPOWERED"
    return {
        "status": "included"
        if clean_reward
        else "skipped_flagged_adversarial"
        if was_reward_skipped
        else "missing",
        "verifier_as_reward_status": status,
        "b1_real_training_smoke": smoke,
        "a_vs_b_delta": float_metric(reward_payload, "a_vs_b_delta") if clean_reward else None,
        "a_vs_b_ci95": ci95(reward_payload, "a_vs_b_ci95") if clean_reward else None,
        "a_vs_b_ci_excludes_zero": ci_excludes_zero(ci95(reward_payload, "a_vs_b_ci95"))
        if clean_reward
        else False,
        "verifier_label_carries_signal": bool_metric(
            reward_payload, "verifier_label_carries_signal"
        )
        if clean_reward
        else None,
        "positive_control_confirmed": bool_metric(reward_payload, "positive_control_confirmed")
        if clean_reward
        else None,
        "live_lora_retired": live_lora_retired,
        "blocked_at_layer": str_metric(reward_payload, "blocked_at_layer") if clean_reward else "",
        "gate_check_summary": str_metric(reward_payload, "gate_check_summary")
        if clean_reward
        else "",
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


def sota_v393(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "flagged_for_v393": ""}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "flagged_for_v393": ""}
    methods = payload.get("methods_mapped")
    first = (
        methods[0]
        if isinstance(methods, list) and methods and isinstance(methods[0], Mapping)
        else {}
    )
    return {
        "status": "included",
        "flagged_for_v393": str_metric(payload, "flagged_for_v393"),
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
        "code_disambiguation_read": str_metric(
            nested_map(payload, "code_disambiguation_outcome"), "disambiguation_read"
        ),
        "verifier_reward_status": str_metric(
            nested_map(payload, "verifier_reward_outcome"), "status"
        ),
        "live_lora_retired": bool_metric(
            nested_map(payload, "verifier_reward_outcome"), "live_lora_retired"
        )
        is True,
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
    if arc_status == "MOAT-WON":
        return "MOAT-WON"
    if arc_status == "TIES-AT-POWER-NULL" and code_status == "CODE-WON":
        return "ARC-NULL-IS-DATA-SPARSITY"
    if arc_status == "TIES-AT-POWER-NULL" and code_status == "CODE-TIES":
        return "THESIS-BOUNDED"
    if arc_status == "TIES-AT-POWER-NULL":
        return "TIES-AT-POWER-NULL"
    return "NO-HEADROOM"


def diffusiongemma_resolvable(arc_gate: Mapping[str, Any], code: Mapping[str, Any]) -> bool:
    return bool(arc_gate.get("oracle_distinct_beats_vote")) or bool(
        code.get("code_oracle_distinct_beats_vote")
    )


def headline_outcome(oracle_status: str, reward_status: str) -> str:
    if oracle_status == "MOAT-WON":
        return "oracle_distinct_aggregator_beats_vote_first_moat"
    if oracle_status == "ARC-NULL-IS-DATA-SPARSITY":
        return "oracle_distinct_arc_null_is_data_sparsity_code_wins"
    if oracle_status == "THESIS-BOUNDED":
        return "oracle_distinct_selection_thesis_bounded_both_tie"
    if oracle_status == "TIES-AT-POWER-NULL":
        return "oracle_distinct_aggregator_ties_vote_at_power_stronger_null"
    if reward_status == "REAL":
        return "verifier_reward_real_label_carries_signal"
    if reward_status == "NULL":
        return "verifier_reward_null_distillation"
    if reward_status == "RETIRED-LIVE-LORA":
        return "verifier_reward_live_lora_retired"
    return "oracle_distinct_aggregator_ties_vote_at_power_stronger_null"


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
        f"complete: capstone_v392_{outcome}_oracle_{oracle_status}_"
        f"reward_{reward_status}_arc_levels{total_levels}_"
        f"flagged_skipped{skipped_count}_diffusiongemma_{gate}"
    )


def imported_fields_by_key(clean_keys: set[str]) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {key: [] for key in DEFAULT_UPSTREAMS}
    if "4208_detector" in clean_keys:
        fields["4208_detector"] = [
            "detection_auroc_by_domain",
            "detection_auroc_ci95_by_domain",
            "selector_headroom_by_domain",
            "verifier_is_oracle_by_domain",
            "n_by_domain",
            "divergence_domains",
        ]
    if "4231_build" in clean_keys:
        fields["4231_build"] = [
            "oracle_distinct_auroc",
            "held_out_task_n",
            "baseline_auroc_391",
            "wrong_majority_n",
            "verifier_is_oracle",
        ]
    if "4231_model" in clean_keys:
        fields["4231_model"] = [
            "oof_rows",
            "held_out_task_n",
            "accepted_rejected_n",
            "model_type",
            "verifier_is_oracle",
        ]
    if "4232_arc_gate" in clean_keys:
        fields["4232_arc_gate"] = [
            "verifier_is_oracle",
            "oracle_distinct_beats_vote",
            "aggregator_minus_vote_delta",
            "aggregator_minus_vote_ci95",
            "margin_override_minus_vote",
            "matched_control_delta",
            "matched_control_policy",
            "headroom_exists",
            "held_out_task_n",
            "oracle_at_k",
            "pass_rates",
            "task_rows",
        ]
    if "4233_code" in clean_keys:
        fields["4233_code"] = [
            "verifier_is_oracle",
            "code_oracle_distinct_beats_vote",
            "code_predictor_minus_vote_delta",
            "code_predictor_minus_vote_ci95",
            "matched_control_delta",
            "headroom_exists",
            "held_out_task_n",
            "off_fold_auroc",
            "disambiguation_read",
            "candidate_pool",
        ]
    if "4234_smoke" in clean_keys:
        fields["4234_smoke"] = [
            "harness_smoke_passed",
            "steps_run",
            "trainable_param_count",
            "lora_attach_path",
            "loss_initial",
            "loss_final",
        ]
    if "4235_reward" in clean_keys:
        fields["4235_reward"] = [
            "a_vs_b_delta",
            "a_vs_b_ci95",
            "verifier_label_carries_signal",
            "positive_control_confirmed",
            "live_lora_retired",
            "blocked_at_layer",
            "gate_check_summary",
        ]
    if "4236_arc_progress" in clean_keys:
        fields["4236_arc_progress"] = [
            "total_levels_solved",
            "total_games_solved",
            "new_levels_solved_this_task",
            "levels_completed",
            "real_env_confirmed",
        ]
    if "4237_live_solver" in clean_keys:
        fields["4237_live_solver"] = [
            "solver_completes_level",
            "live_env_metrics",
            "solver_beats_floor",
            "live_env_reachable",
        ]
    if "4238_sota" in clean_keys:
        fields["4238_sota"] = ["flagged_for_v393", "methods_mapped"]
    if "4239_registry" in clean_keys:
        fields["4239_registry"] = [
            "regression_guard_passed",
            "oracle_distinct_outcome.status",
            "code_disambiguation_outcome.disambiguation_read",
            "verifier_reward_outcome.status",
            "verifier_reward_outcome.live_lora_retired",
        ]
    if "4240_hardware" in clean_keys:
        fields["4240_hardware"] = [
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

    detector = detector_selection_divergence(
        clean.get("4208_detector"), was_skipped="4208_detector" in skipped_keys
    )
    model = arc_aggregator_model(
        clean.get("4231_model"),
        clean.get("4231_build"),
        was_model_skipped="4231_model" in skipped_keys,
        was_build_skipped="4231_build" in skipped_keys,
    )
    arc_gate = arc_aggregator_gate(
        clean.get("4232_arc_gate"), model, was_skipped="4232_arc_gate" in skipped_keys
    )
    code = code_disambiguation(clean.get("4233_code"), was_skipped="4233_code" in skipped_keys)
    registry = registry_hygiene(
        clean.get("4239_registry"), was_skipped="4239_registry" in skipped_keys
    )
    reward = verifier_as_reward(
        clean.get("4235_reward"),
        clean.get("4234_smoke"),
        clean.get("4239_registry"),
        was_reward_skipped="4235_reward" in skipped_keys,
        was_smoke_skipped="4234_smoke" in skipped_keys,
    )
    arc = arc_progress(
        clean.get("4236_arc_progress"), was_skipped="4236_arc_progress" in skipped_keys
    )
    live = live_solver_accuracy(
        clean.get("4237_live_solver"), was_skipped="4237_live_solver" in skipped_keys
    )
    sota = sota_v393(clean.get("4238_sota"), was_skipped="4238_sota" in skipped_keys)
    hardware = hardware_continuity(
        clean.get("4240_hardware"), was_skipped="4240_hardware" in skipped_keys
    )

    oracle_status = oracle_distinct_status(arc_gate, code)
    reward_status = str(reward["verifier_as_reward_status"])
    gate_resolvable = diffusiongemma_resolvable(arc_gate, code)
    outcome = headline_outcome(oracle_status, reward_status)
    total_levels = int(arc["total_arc_levels_solved"])
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_keys)
    fields_by_key = imported_fields_by_key(clean_keys)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v392_4241.v1",
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict(
            outcome, oracle_status, reward_status, total_levels, len(skipped), gate_resolvable
        ),
        "headline_outcome": outcome,
        "oracle_distinct_status": oracle_status,
        "arc_aggregator_gate": arc_gate,
        "arc_aggregator_model": model,
        "code_disambiguation": code,
        "detector_selection_divergence": detector,
        "verifier_as_reward_status": reward_status,
        "verifier_as_reward": reward,
        "diffusiongemma_gate_resolvable": gate_resolvable,
        "total_arc_levels_solved": total_levels,
        "arc_progress": arc,
        "live_solver_accuracy": live,
        "strongest_sota_flagged_for_v393": str(sota.get("flagged_for_v393") or ""),
        "sota_v393": sota,
        "registry_hygiene": registry,
        "hardware_continuity": hardware,
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
