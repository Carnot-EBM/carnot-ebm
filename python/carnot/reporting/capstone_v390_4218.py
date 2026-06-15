"""Build the Exp 4218 v390 oracle-distinct verifier capstone aggregation.

Spec refs: REQ-CAPSTONE-4218, SCENARIO-CAPSTONE-4218.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_4218_capstone_v390.json")
EXPERIMENT_ID = 4218
RANDOM_SEED = 4218
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4218", "SCENARIO-CAPSTONE-4218"]

DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    4208: Path("results/experiment_4208_verifier_as_detector_auroc.json"),
    4209: Path("results/experiment_4209_oracle_distinct_arc_verifier_build.json"),
    4210: Path("results/experiment_4210_oracle_distinct_arc_verifier_beats_vote.json"),
    4211: Path("results/experiment_4211_verifier_as_reward_finish_synchronous.json"),
    4212: Path("results/experiment_4212_certified_arc_corpus_distill_lift.json"),
    4213: Path("results/experiment_4213_arc_incremental_progress.json"),
    4214: Path("results/experiment_4214_arc_live_env_solver_accuracy.json"),
    4215: Path("results/experiment_4215_sota_ingestion_oracle_distinct.json"),
    4216: Path("results/experiment_4216_verifier_registry_gaps_hygiene.json"),
    4217: Path("results/experiment_4217_hardware_continuity.json"),
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
        "signal' is COMPLETE and decision-grade -- it tells .391 whether the "
        "oracle-distinct frontier is reachable on ARC."
    ),
    "headline_outcome": (
        "One of the enumerated set -- forces a single unambiguous read of the "
        "oracle-distinct frontier + the verifier-as-reward pivot."
    ),
    "oracle_distinct_status": (
        "MOAT-WON (learned verifier beats vote off-oracle, CI excl 0) / "
        "TIES-VOTE-NULL / NO-HEADROOM-OR-NO-SIGNAL -- the 2026-06-14 P0 directive's "
        "standing after .390; a circular result does NOT count here."
    ),
    "verifier_as_reward_status": (
        "REAL (A-vs-B label carries signal) / NULL (distillation/spurious) / "
        "INVALID-or-UNDERPOWERED / ACCUMULATING -- the owed 2026-06-11 pivot's "
        "standing after .390."
    ),
    "diffusiongemma_gate_resolvable": (
        "BARE bool: true ONLY if an oracle-distinct win landed with a matched control "
        "(verifier_is_oracle=false, CI95-excl-0); a circular execution win keeps it "
        "STILL-PENDING."
    ),
    "total_arc_levels_solved": (
        "The monotonic ARC progress metric after .390 (must be >= 15)."
    ),
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


def selected_upstream_paths(root: Path | str) -> dict[int, Path | None]:
    root_path = Path(root)
    return {
        experiment_id: path if (path := root_path / rel_path).exists() else None
        for experiment_id, rel_path in DEFAULT_UPSTREAM_PATHS.items()
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
        "honest_verdict": verdict_text(payload),
    }


def learned_arc_verifier(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "off_fold_auroc": None}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "off_fold_auroc": None}
    return {
        "status": "included",
        "selector_trained": bool_metric(payload, "selector_trained") is True,
        "off_fold_auroc": float_metric(payload, "oracle_distinct_auroc"),
        "off_fold_auroc_ci95": ci95(payload, "oracle_distinct_auroc_ci95"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "learned_verifier_path": str_metric(payload, "learned_verifier_path"),
        "accepted_rejected_n": dict(nested_map(payload, "accepted_rejected_n")),
        "honest_verdict": verdict_text(payload),
    }


def _gate_interval(payload: Mapping[str, Any] | None) -> list[float] | None:
    return (
        ci95(payload, "oracle_distinct_ci95")
        or ci95(payload, "oracle_distinct_delta_ci95")
        or ci95(payload, "ci95")
    )


def oracle_distinct_frontier(
    gate_payload: Mapping[str, Any] | None,
    build_payload: Mapping[str, Any] | None,
    detector: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "oracle_distinct_status": "NO-HEADROOM-OR-NO-SIGNAL"}
    if not isinstance(gate_payload, Mapping):
        return {"status": "missing", "oracle_distinct_status": "NO-HEADROOM-OR-NO-SIGNAL"}

    verifier_is_oracle = bool_metric(gate_payload, "verifier_is_oracle")
    if verifier_is_oracle is None:
        verifier_is_oracle = bool_metric(build_payload, "verifier_is_oracle")
    matched_control = bool_metric(gate_payload, "matched_control") is True or bool_metric(
        gate_payload, "matched_control_present"
    ) is True
    headroom_present = bool_metric(gate_payload, "headroom_present")
    if headroom_present is None:
        headroom = nested_map(detector, "selector_headroom_by_domain").get("arc")
        headroom_present = isinstance(headroom, (int, float)) and not isinstance(headroom, bool) and headroom > 0.0
    interval = _gate_interval(gate_payload)
    delta = float_metric(gate_payload, "oracle_distinct_delta")
    if delta is None:
        delta = float_metric(gate_payload, "verifier_lift_vs_vote")
    blocked = str_metric(gate_payload, "status") == "blocked" or verdict_text(gate_payload).startswith(
        "blocked"
    )
    comparison_ran = not blocked and (
        delta is not None or bool_metric(gate_payload, "oracle_distinct_beats_vote") is not None
    )
    beat_flag = bool_metric(gate_payload, "oracle_distinct_beats_vote") is True
    moat_won = (
        verifier_is_oracle is False
        and matched_control
        and headroom_present is True
        and comparison_ran
        and ci_excludes_zero(interval)
        and (beat_flag or (delta is not None and delta > 0.0))
    )
    ties_vote = (
        verifier_is_oracle is False
        and matched_control
        and headroom_present is True
        and comparison_ran
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
        "comparison_ran": comparison_ran,
        "verifier_is_oracle": verifier_is_oracle,
        "matched_control": matched_control,
        "headroom_present": headroom_present is True,
        "oracle_distinct_delta": delta,
        "oracle_distinct_ci95": interval,
        "ci95_excludes_zero": ci_excludes_zero(interval),
        "gate_check_summary": str_metric(gate_payload, "gate_check_summary"),
        "blocked_at_layer": str_metric(gate_payload, "blocked_at_layer"),
        "honest_verdict": verdict_text(gate_payload),
    }


def verifier_as_reward(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "verifier_as_reward_status": "INVALID-or-UNDERPOWERED"}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "verifier_as_reward_status": "INVALID-or-UNDERPOWERED"}
    interval = ci95(payload, "a_vs_b_ci95")
    delta = float_metric(payload, "a_vs_b_delta")
    positive_control = bool_metric(payload, "positive_control_confirmed") is True
    label_carries = bool_metric(payload, "verifier_label_carries_signal") is True
    accumulated = nested_map(payload, "accumulated_n")
    eval_n = accumulated.get("eval")
    evaluation = nested_map(payload, "evaluation")
    no_eval_yet = (
        eval_n == 0
        or str_metric(evaluation, "status").startswith("pending")
        or verdict_text(payload).startswith("progress: accumulating")
    )
    if positive_control and label_carries and delta is not None and delta > 0.0 and ci_excludes_zero(interval):
        status = "REAL"
    elif positive_control and interval is not None and ci_includes_zero(interval):
        status = "NULL"
    elif no_eval_yet:
        status = "ACCUMULATING"
    else:
        status = "INVALID-or-UNDERPOWERED"
    return {
        "status": "included",
        "verifier_as_reward_status": status,
        "a_vs_b_delta": delta,
        "a_vs_b_ci95": interval,
        "a_vs_b_ci_excludes_zero": ci_excludes_zero(interval),
        "verifier_label_carries_signal": status == "REAL",
        "positive_control_confirmed": positive_control,
        "accumulated_n": dict(accumulated),
        "pass_at_1": dict(nested_map(payload, "pass_at_1")),
        "training": dict(nested_map(payload, "training")),
        "evaluation": dict(evaluation),
        "truncation_guard": dict(nested_map(payload, "truncation_guard")),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "youden_j": float_metric(payload, "youden_j"),
        "honest_verdict": verdict_text(payload),
    }


def certified_corpus_latent_or_absent(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "latent_or_absent": "SKIPPED-FLAGGED"}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "latent_or_absent": "MISSING"}
    latent = str_metric(payload, "distill_lift_latent_vs_absent").upper()
    if not latent:
        interval = ci95(payload, "distill_lift_ci95")
        latent = "LATENT" if interval is not None and interval[0] > 0.0 else "ABSENT"
    return {
        "status": "included",
        "latent_or_absent": latent,
        "certified_corpus_size": int_metric(payload, "certified_corpus_size"),
        "certification_precision": dict(nested_map(payload, "certification_precision")),
        "distill_lift_delta": float_metric(payload, "distill_lift_delta"),
        "distill_lift_ci95": ci95(payload, "distill_lift_ci95"),
        "honest_verdict": verdict_text(payload),
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
    beats = nested_map(payload, "solver_beats_floor")
    return {
        "status": "included",
        "solver_completes_level": bool_metric(payload, "solver_completes_level") is True,
        "levels_completed": int_metric(metrics, "levels_completed"),
        "score": float_metric(metrics, "score"),
        "live_env_reachable": bool_metric(payload, "live_env_reachable") is True,
        "solver_beats_floor_accuracy": bool_metric(nested_map(beats, "accuracy"), "beats") is True,
        "solver_beats_floor_efficiency": bool_metric(nested_map(beats, "efficiency"), "beats") is True,
        "solver_beats_floor_overall": bool_metric(beats, "overall") is True,
        "honest_verdict": verdict_text(payload),
    }


def sota_v391(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "flagged_for_v391": ""}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "flagged_for_v391": ""}
    methods = payload.get("methods_mapped")
    return {
        "status": "included",
        "flagged_for_v391": str_metric(payload, "flagged_for_v391"),
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
        "gatemate_step_taken": str_metric(payload, "gatemate_step_taken"),
        "polarfire_step_taken": str_metric(payload, "polarfire_step_taken"),
        "kv260_terminal_confirmed": bool_metric(payload, "kv260_terminal_confirmed") is True,
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


def honest_verdict(outcome: str, oracle_status: str, reward_status: str, total_levels: int, skipped_count: int) -> str:
    gate = "resolvable" if oracle_status == "MOAT-WON" else "still_pending"
    return (
        f"complete: capstone_v390_{outcome}_oracle_{oracle_status}_"
        f"reward_{reward_status}_arc_levels{total_levels}_"
        f"flagged_skipped{skipped_count}_diffusiongemma_{gate}"
    )


def imported_fields_by_id(clean_ids: set[int]) -> dict[int, list[str]]:
    fields: dict[int, list[str]] = {experiment_id: [] for experiment_id in DEFAULT_UPSTREAM_PATHS}
    if 4208 in clean_ids:
        fields[4208] = [
            "detection_auroc_by_domain",
            "detection_auroc_ci95_by_domain",
            "selector_headroom_by_domain",
            "verifier_is_oracle_by_domain",
            "n_by_domain",
        ]
    if 4209 in clean_ids:
        fields[4209] = [
            "selector_trained",
            "oracle_distinct_auroc",
            "oracle_distinct_auroc_ci95",
            "verifier_is_oracle",
            "learned_verifier_path",
            "accepted_rejected_n",
        ]
    if 4210 in clean_ids:
        fields[4210] = [
            "status",
            "gate_check_summary",
            "blocked_at_layer",
            "oracle_distinct_beats_vote",
            "oracle_distinct_delta",
            "oracle_distinct_ci95",
            "matched_control",
            "headroom_present",
            "verifier_is_oracle",
        ]
    if 4211 in clean_ids:
        fields[4211] = [
            "a_vs_b_delta",
            "a_vs_b_ci95",
            "verifier_label_carries_signal",
            "positive_control_confirmed",
            "accumulated_n",
            "pass_at_1",
            "evaluation",
            "training",
            "verifier_is_oracle",
            "youden_j",
        ]
    if 4212 in clean_ids:
        fields[4212] = [
            "certified_corpus_size",
            "certification_precision",
            "distill_lift_delta",
            "distill_lift_ci95",
            "distill_lift_latent_vs_absent",
        ]
    if 4213 in clean_ids:
        fields[4213] = [
            "total_levels_solved",
            "total_games_solved",
            "new_levels_solved_this_task",
            "levels_completed",
            "real_env_confirmed",
        ]
    if 4214 in clean_ids:
        fields[4214] = [
            "solver_completes_level",
            "live_env_metrics",
            "solver_beats_floor",
            "live_env_reachable",
        ]
    if 4215 in clean_ids:
        fields[4215] = ["flagged_for_v391", "methods_mapped"]
    if 4216 in clean_ids:
        fields[4216] = ["regression_guard_passed"]
    if 4217 in clean_ids:
        fields[4217] = [
            "per_board_reachability",
            "gatemate_step_taken",
            "polarfire_step_taken",
            "kv260_terminal_confirmed",
        ]
    return fields


def flagged_artifacts_skipped(
    root: Path,
    paths: Mapping[int, Path | None],
    upstreams: Mapping[int, Mapping[str, Any] | None],
    skipped_ids: set[int],
) -> list[JsonDict]:
    return [
        {
            "experiment_id": experiment_id,
            "path": str(paths[experiment_id].relative_to(root)),
            "reason": "flagged_adversarial:true",
            "sha256": sha256_file(paths[experiment_id]),
            "honest_verdict": verdict_text(upstreams[experiment_id]),
        }
        for experiment_id in DEFAULT_UPSTREAM_PATHS
        if paths[experiment_id] is not None and experiment_id in skipped_ids
    ]


def upstream_provenance(
    root: Path,
    paths: Mapping[int, Path | None],
    upstreams: Mapping[int, Mapping[str, Any] | None],
    skipped_ids: set[int],
    fields_by_id: Mapping[int, list[str]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for experiment_id in DEFAULT_UPSTREAM_PATHS:
        path = paths[experiment_id]
        if path is None:
            continue
        skipped = experiment_id in skipped_ids
        rows.append(
            {
                "experiment_id": experiment_id,
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "fields_imported": [] if skipped else list(fields_by_id.get(experiment_id, [])),
                "skipped": skipped,
                "skip_reason": "flagged_adversarial:true" if skipped else "",
                "honest_verdict": verdict_text(upstreams[experiment_id]),
            }
        )
    return rows


def missing_upstream_artifacts(paths: Mapping[int, Path | None]) -> list[JsonDict]:
    return [
        {"experiment_id": experiment_id}
        for experiment_id in DEFAULT_UPSTREAM_PATHS
        if paths[experiment_id] is None
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
    upstreams: dict[int, Mapping[str, Any] | None] = {
        experiment_id: read_json_object(path) if path is not None else None
        for experiment_id, path in paths.items()
    }
    skipped_ids = {experiment_id for experiment_id, payload in upstreams.items() if flagged(payload)}
    clean_ids = {
        experiment_id
        for experiment_id, payload in upstreams.items()
        if isinstance(payload, Mapping) and experiment_id not in skipped_ids
    }
    clean = {experiment_id: upstreams[experiment_id] for experiment_id in clean_ids}

    detector = detector_selection_divergence(clean.get(4208), was_skipped=4208 in skipped_ids)
    learned = learned_arc_verifier(clean.get(4209), was_skipped=4209 in skipped_ids)
    oracle = oracle_distinct_frontier(
        clean.get(4210), clean.get(4209), detector, was_skipped=4210 in skipped_ids
    )
    reward = verifier_as_reward(clean.get(4211), was_skipped=4211 in skipped_ids)
    certified = certified_corpus_latent_or_absent(
        clean.get(4212), was_skipped=4212 in skipped_ids
    )
    arc = arc_progress(clean.get(4213), was_skipped=4213 in skipped_ids)
    live = live_solver_accuracy(clean.get(4214), was_skipped=4214 in skipped_ids)
    sota = sota_v391(clean.get(4215), was_skipped=4215 in skipped_ids)
    registry = registry_hygiene(clean.get(4216), was_skipped=4216 in skipped_ids)
    hardware = hardware_continuity(clean.get(4217), was_skipped=4217 in skipped_ids)

    oracle_status = str(oracle["oracle_distinct_status"])
    reward_status = str(reward["verifier_as_reward_status"])
    outcome = headline_outcome(oracle_status, reward_status)
    total_levels = int(arc["total_arc_levels_solved"])
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_ids)
    fields_by_id = imported_fields_by_id(clean_ids)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v390_4218.v1",
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
        "certified_corpus_latent_or_absent": certified,
        "diffusiongemma_gate_resolvable": oracle_status == "MOAT-WON",
        "total_arc_levels_solved": total_levels,
        "arc_progress": arc,
        "live_solver_accuracy": live,
        "strongest_sota_flagged_for_v391": str(sota.get("flagged_for_v391") or ""),
        "sota_v391": sota,
        "registry_hygiene": registry,
        "hardware_continuity": hardware,
        "flagged_artifacts_skipped": skipped,
        "upstream_provenance": upstream_provenance(
            root_path, paths, upstreams, skipped_ids, fields_by_id
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
    if not isinstance(total_levels, int) or isinstance(total_levels, bool) or total_levels < 15:
        raise ValueError("total ARC levels must be an integer >= 15")
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
