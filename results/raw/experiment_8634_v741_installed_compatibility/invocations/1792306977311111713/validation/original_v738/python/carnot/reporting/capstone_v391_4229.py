"""Build the Exp 4229 v391 oracle-distinct verifier capstone aggregation.

Spec refs: REQ-CAPSTONE-4229, SCENARIO-CAPSTONE-4229.
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
OUTPUT_REL_PATH = Path("results/experiment_4229_capstone_v391.json")
EXPERIMENT_ID = 4229
RANDOM_SEED = 4229
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4229", "SCENARIO-CAPSTONE-4229"]


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4208_detector": Upstream(
        4208, Path("results/experiment_4208_verifier_as_detector_auroc.json")
    ),
    "4220_build_labeled": Upstream(
        4220, Path("results/experiment_4220_oracle_distinct_arc_verifier_build_labeled.json")
    ),
    "4220_model": Upstream(
        4220, Path("results/experiment_4220_oracle_distinct_arc_verifier_model.json")
    ),
    "4221_gate": Upstream(
        4221, Path("results/experiment_4221_oracle_distinct_arc_verifier_beats_vote.json")
    ),
    "4222_harness": Upstream(
        4222, Path("results/experiment_4222_verifier_reward_lora_harness_fix_smoke.json")
    ),
    "4223_reward": Upstream(
        4223, Path("results/experiment_4223_verifier_as_reward_3arm_synchronous.json")
    ),
    "4224_arc_progress": Upstream(
        4224, Path("results/experiment_4224_arc_incremental_progress.json")
    ),
    "4225_live_solver": Upstream(
        4225, Path("results/experiment_4225_arc_live_env_solver_accuracy.json")
    ),
    "4226_sota": Upstream(
        4226, Path("results/experiment_4226_sota_ingestion_learned_aggregator.json")
    ),
    "4227_registry": Upstream(
        4227, Path("results/experiment_4227_verifier_registry_gaps_hygiene.json")
    ),
    "4228_hardware": Upstream(4228, Path("results/experiment_4228_hardware_continuity.json")),
}

HEADLINE_OUTCOMES = {
    "oracle_distinct_verifier_beats_vote_first_moat",
    "oracle_distinct_verifier_ties_vote_with_headroom_null",
    "oracle_distinct_no_headroom_or_no_learnable_signal",
    "verifier_reward_real_label_carries_signal",
    "verifier_reward_null_distillation",
}

ORACLE_DISTINCT_STATUSES = {
    "MOAT-WON",
    "TIES-VOTE-NULL",
    "NO-HEADROOM-OR-NO-SIGNAL",
}

VERIFIER_AS_REWARD_STATUSES = {
    "REAL",
    "NULL",
    "INVALID-or-UNDERPOWERED",
    "ACCUMULATING",
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
        "Terminal-prefixed. An honest 'ties-vote null' or 'no learnable oracle-distinct "
        "signal' is COMPLETE and decision-grade -- it tells .392 whether the "
        "oracle-distinct frontier is reachable on ARC (and this time the gate RAN)."
    ),
    "headline_outcome": (
        "One of the enumerated set -- forces a single unambiguous read of the "
        "oracle-distinct frontier + the verifier-as-reward pivot."
    ),
    "oracle_distinct_status": (
        "MOAT-WON (learned verifier beats vote off-oracle, CI excl 0) / TIES-VOTE-NULL "
        "/ NO-HEADROOM-OR-NO-SIGNAL -- the 2026-06-14 P0 directive's standing after "
        ".391; a circular result does NOT count here."
    ),
    "verifier_as_reward_status": (
        "REAL (A-vs-B label carries signal) / NULL (distillation/spurious) / "
        "INVALID-or-UNDERPOWERED / ACCUMULATING / HARNESS-DEFERRED -- the owed "
        "2026-06-11 pivot's standing after .391."
    ),
    "diffusiongemma_gate_resolvable": (
        "BARE bool: true ONLY if an oracle-distinct win landed with a matched control "
        "(verifier_is_oracle=false, CI95-excl-0); a circular execution win keeps it "
        "STILL-PENDING."
    ),
    "total_arc_levels_solved": "The monotonic ARC progress metric after .391 (must be >= 16).",
    "upstream_provenance": (
        "{experiment_id, fields_imported, sha256} per cited upstream; the audit trail "
        "that a capstone synthesizes nothing from nothing."
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
    count = 0
    for row in task_rows:
        if (
            isinstance(row, Mapping)
            and row.get("vote_correct") is False
            and row.get("oracle_hit") is True
        ):
            count += 1
    return count


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


def learned_arc_verifier(
    model_payload: Mapping[str, Any] | None,
    gate_payload: Mapping[str, Any] | None,
    *,
    was_model_skipped: bool,
    was_summary_skipped: bool,
) -> JsonDict:
    summary_status = "skipped_flagged_adversarial" if was_summary_skipped else "not_skipped"
    if was_model_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "summary_artifact_status": summary_status,
            "off_fold_auroc": None,
            "wrong_majority_n": None,
        }
    if not isinstance(model_payload, Mapping):
        return {
            "status": "missing",
            "summary_artifact_status": summary_status,
            "off_fold_auroc": None,
            "wrong_majority_n": None,
        }
    oof_rows = model_payload.get("oof_rows")
    rows = (
        oof_rows
        if isinstance(oof_rows, Sequence) and not isinstance(oof_rows, (str, bytes))
        else []
    )
    return {
        "status": "included",
        "summary_artifact_status": summary_status,
        "metric_source": "computed_from_clean_model_oof_rows_and_clean_gate_task_rows",
        "off_fold_auroc": rank_auc([row for row in rows if isinstance(row, Mapping)]),
        "wrong_majority_n": wrong_majority_count(gate_payload),
        "oof_row_n": len(rows),
        "accepted_rejected_n": dict(nested_map(model_payload, "accepted_rejected_n")),
        "model_type": str_metric(model_payload, "model_type"),
        "verifier_is_oracle": bool_metric(model_payload, "verifier_is_oracle"),
    }


def _gate_interval(payload: Mapping[str, Any] | None) -> list[float] | None:
    return (
        ci95(payload, "verifier_minus_vote_ci95")
        or ci95(payload, "oracle_distinct_ci95")
        or ci95(payload, "delta_ci95")
        or ci95(payload, "ci95")
    )


def oracle_distinct_frontier(
    gate_payload: Mapping[str, Any] | None,
    detector: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    if was_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "oracle_distinct_status": "NO-HEADROOM-OR-NO-SIGNAL",
            "gate_ran": False,
        }
    if not isinstance(gate_payload, Mapping):
        return {
            "status": "missing",
            "oracle_distinct_status": "NO-HEADROOM-OR-NO-SIGNAL",
            "gate_ran": False,
        }
    verifier_is_oracle = bool_metric(gate_payload, "verifier_is_oracle")
    matched_control = (
        bool_metric(gate_payload, "matched_control") is True
        or bool_metric(gate_payload, "matched_control_present") is True
        or float_metric(gate_payload, "matched_control_delta") is not None
        or bool(str_metric(gate_payload, "matched_control_policy"))
    )
    headroom_present = bool_metric(gate_payload, "headroom_present")
    if headroom_present is None:
        headroom_present = bool_metric(gate_payload, "headroom_exists")
    if headroom_present is None:
        arc_headroom = nested_map(detector, "selector_headroom_by_domain").get("arc")
        headroom_present = (
            isinstance(arc_headroom, (int, float))
            and not isinstance(arc_headroom, bool)
            and arc_headroom > 0.0
        )
    interval = _gate_interval(gate_payload)
    delta = float_metric(gate_payload, "verifier_minus_vote_delta")
    if delta is None:
        delta = float_metric(gate_payload, "oracle_distinct_delta")
    blocked = str_metric(gate_payload, "status") == "blocked" or verdict_text(
        gate_payload
    ).startswith("blocked")
    beat_flag = bool_metric(gate_payload, "oracle_distinct_beats_vote")
    gate_ran = not blocked and (delta is not None or interval is not None or beat_flag is not None)
    moat_won = (
        verifier_is_oracle is False
        and matched_control
        and headroom_present is True
        and gate_ran
        and ci_excludes_zero(interval)
        and (beat_flag is True or (delta is not None and delta > 0.0))
    )
    ties_vote = (
        verifier_is_oracle is False
        and matched_control
        and headroom_present is True
        and gate_ran
        and not moat_won
        and ci_includes_zero(interval)
    )
    if moat_won:
        status = "MOAT-WON"
    elif ties_vote:
        status = "TIES-VOTE-NULL"
    else:
        status = "NO-HEADROOM-OR-NO-SIGNAL"
    return {
        "status": "included",
        "oracle_distinct_status": status,
        "oracle_distinct_beats_vote": moat_won,
        "gate_ran": gate_ran,
        "verifier_is_oracle": verifier_is_oracle,
        "matched_control_present": matched_control,
        "headroom_present": headroom_present is True,
        "verifier_minus_vote_delta": delta,
        "verifier_minus_vote_ci95": interval,
        "ci95_excludes_zero": ci_excludes_zero(interval),
        "matched_control_delta": float_metric(gate_payload, "matched_control_delta"),
        "arbiter_override_minus_vote": float_metric(gate_payload, "arbiter_override_minus_vote"),
        "oracle_at_k": float_metric(gate_payload, "oracle_at_k"),
        "pass_rates": dict(nested_map(gate_payload, "pass_rates")),
        "n_tasks": int_metric(gate_payload, "n_tasks"),
        "honest_verdict": verdict_text(gate_payload),
    }


def _harness_smoke(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "harness_smoke_passed": False}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "harness_smoke_passed": False}
    trainable = int_metric(payload, "trainable_param_count")
    passed = bool_metric(payload, "harness_smoke_passed") is True and trainable > 0
    return {
        "status": "included",
        "harness_smoke_passed": passed,
        "trainable_param_count": trainable,
        "lora_attach_path": str_metric(payload, "lora_attach_path"),
        "honest_verdict": verdict_text(payload),
    }


def verifier_as_reward(
    reward_payload: Mapping[str, Any] | None,
    harness_payload: Mapping[str, Any] | None,
    *,
    was_reward_skipped: bool,
    was_harness_skipped: bool,
) -> JsonDict:
    harness = _harness_smoke(harness_payload, was_skipped=was_harness_skipped)
    if harness["harness_smoke_passed"] is not True:
        status = "HARNESS-DEFERRED"
    elif was_reward_skipped or not isinstance(reward_payload, Mapping):
        status = "INVALID-or-UNDERPOWERED"
    else:
        interval = ci95(reward_payload, "a_vs_b_ci95")
        delta = float_metric(reward_payload, "a_vs_b_delta")
        positive_control = bool_metric(reward_payload, "positive_control_confirmed") is True
        label_carries = bool_metric(reward_payload, "verifier_label_carries_signal") is True
        accumulated = nested_map(reward_payload, "accumulated_n")
        evaluation = nested_map(reward_payload, "evaluation")
        no_eval_yet = (
            accumulated.get("eval") == 0
            or str_metric(evaluation, "status").startswith("pending")
            or verdict_text(reward_payload).startswith("progress: accumulating")
        )
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
        elif no_eval_yet:
            status = "ACCUMULATING"
        else:
            status = "INVALID-or-UNDERPOWERED"
    clean_reward = isinstance(reward_payload, Mapping) and not was_reward_skipped
    return {
        "status": "included"
        if clean_reward
        else "skipped_flagged_adversarial"
        if was_reward_skipped
        else "missing",
        "verifier_as_reward_status": status,
        "b1_harness_smoke": harness,
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
        "accumulated_n": dict(nested_map(reward_payload, "accumulated_n")) if clean_reward else {},
        "training": dict(nested_map(reward_payload, "training")) if clean_reward else {},
        "evaluation": dict(nested_map(reward_payload, "evaluation")) if clean_reward else {},
        "truncation_guard": dict(nested_map(reward_payload, "truncation_guard"))
        if clean_reward
        else {},
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


def sota_v392(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "flagged_for_v392": ""}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "flagged_for_v392": ""}
    methods = payload.get("methods_mapped")
    first = (
        methods[0]
        if isinstance(methods, list) and methods and isinstance(methods[0], Mapping)
        else {}
    )
    return {
        "status": "included",
        "flagged_for_v392": str_metric(payload, "flagged_for_v392"),
        "strongest_method_name": str_metric(first, "name"),
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
        "verifier_reward_status": str_metric(
            nested_map(payload, "verifier_reward_outcome"), "status"
        ),
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


def headline_outcome(oracle_status: str, reward_status: str) -> str:
    if oracle_status == "MOAT-WON":
        return "oracle_distinct_verifier_beats_vote_first_moat"
    if oracle_status == "TIES-VOTE-NULL":
        return "oracle_distinct_verifier_ties_vote_with_headroom_null"
    if reward_status == "REAL":
        return "verifier_reward_real_label_carries_signal"
    if reward_status == "NULL":
        return "verifier_reward_null_distillation"
    return "oracle_distinct_no_headroom_or_no_learnable_signal"


def honest_verdict(
    outcome: str,
    oracle_status: str,
    reward_status: str,
    total_levels: int,
    skipped_count: int,
) -> str:
    gate = "resolvable" if oracle_status == "MOAT-WON" else "still_pending"
    return (
        f"complete: capstone_v391_{outcome}_oracle_{oracle_status}_"
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
    if "4220_model" in clean_keys:
        fields["4220_model"] = [
            "oof_rows",
            "accepted_rejected_n",
            "model_type",
            "verifier_is_oracle",
        ]
    if "4221_gate" in clean_keys:
        fields["4221_gate"] = [
            "verifier_is_oracle",
            "oracle_distinct_beats_vote",
            "verifier_minus_vote_delta",
            "verifier_minus_vote_ci95",
            "matched_control_delta",
            "matched_control_policy",
            "headroom_exists",
            "arbiter_override_minus_vote",
            "pass_rates",
            "task_rows",
        ]
    if "4222_harness" in clean_keys:
        fields["4222_harness"] = [
            "harness_smoke_passed",
            "trainable_param_count",
            "lora_attach_path",
        ]
    if "4223_reward" in clean_keys:
        fields["4223_reward"] = [
            "a_vs_b_delta",
            "a_vs_b_ci95",
            "verifier_label_carries_signal",
            "positive_control_confirmed",
            "accumulated_n",
            "evaluation",
            "training",
            "truncation_guard",
        ]
    if "4224_arc_progress" in clean_keys:
        fields["4224_arc_progress"] = [
            "total_levels_solved",
            "total_games_solved",
            "new_levels_solved_this_task",
            "levels_completed",
            "real_env_confirmed",
        ]
    if "4225_live_solver" in clean_keys:
        fields["4225_live_solver"] = [
            "solver_completes_level",
            "live_env_metrics",
            "solver_beats_floor",
            "live_env_reachable",
        ]
    if "4226_sota" in clean_keys:
        fields["4226_sota"] = ["flagged_for_v392", "methods_mapped"]
    if "4227_registry" in clean_keys:
        fields["4227_registry"] = [
            "regression_guard_passed",
            "oracle_distinct_outcome.status",
            "verifier_reward_outcome.status",
        ]
    if "4228_hardware" in clean_keys:
        fields["4228_hardware"] = [
            "per_board_reachability",
            "per_board_status",
            "gatemate_step_taken",
            "polarfire_step_taken",
            "kv260_terminal_confirmed",
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
    learned = learned_arc_verifier(
        clean.get("4220_model"),
        clean.get("4221_gate"),
        was_model_skipped="4220_model" in skipped_keys,
        was_summary_skipped="4220_build_labeled" in skipped_keys,
    )
    oracle = oracle_distinct_frontier(
        clean.get("4221_gate"), detector, was_skipped="4221_gate" in skipped_keys
    )
    reward = verifier_as_reward(
        clean.get("4223_reward"),
        clean.get("4222_harness"),
        was_reward_skipped="4223_reward" in skipped_keys,
        was_harness_skipped="4222_harness" in skipped_keys,
    )
    arc = arc_progress(
        clean.get("4224_arc_progress"), was_skipped="4224_arc_progress" in skipped_keys
    )
    live = live_solver_accuracy(
        clean.get("4225_live_solver"), was_skipped="4225_live_solver" in skipped_keys
    )
    sota = sota_v392(clean.get("4226_sota"), was_skipped="4226_sota" in skipped_keys)
    registry = registry_hygiene(
        clean.get("4227_registry"), was_skipped="4227_registry" in skipped_keys
    )
    hardware = hardware_continuity(
        clean.get("4228_hardware"), was_skipped="4228_hardware" in skipped_keys
    )

    oracle_status = str(oracle["oracle_distinct_status"])
    reward_status = str(reward["verifier_as_reward_status"])
    outcome = headline_outcome(oracle_status, reward_status)
    total_levels = int(arc["total_arc_levels_solved"])
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_keys)
    fields_by_key = imported_fields_by_key(clean_keys)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v391_4229.v1",
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict(
            outcome, oracle_status, reward_status, total_levels, len(skipped)
        ),
        "headline_outcome": outcome,
        "oracle_distinct_status": oracle_status,
        "oracle_distinct_frontier": oracle,
        "learned_arc_verifier": learned,
        "detector_selection_divergence": detector,
        "verifier_as_reward_status": reward_status,
        "verifier_as_reward": reward,
        "diffusiongemma_gate_resolvable": oracle_status == "MOAT-WON",
        "total_arc_levels_solved": total_levels,
        "arc_progress": arc,
        "live_solver_accuracy": live,
        "strongest_sota_flagged_for_v392": str(sota.get("flagged_for_v392") or ""),
        "sota_v392": sota,
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
    if not isinstance(total_levels, int) or isinstance(total_levels, bool) or total_levels < 16:
        raise ValueError("total ARC levels must be an integer >= 16")
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
    output = root_path / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    output = write_artifact(REPO_ROOT)
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
