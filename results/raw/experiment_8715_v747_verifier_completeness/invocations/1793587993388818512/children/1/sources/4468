"""Build the Exp 4133 v382 capstone aggregation.

Spec refs: REQ-CAPSTONE-4133, SCENARIO-CAPSTONE-4133.

This module is deliberately a reader, not a runner. It does not retrain TRM,
rerun the Sudoku graft, launch ARC exploration, or probe hardware. It reads the
landed .382 artifacts, excludes adversarial-flagged inputs before any metric
import, and writes the shortest decision-grade summary that can be audited from
sha256 provenance.
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
OUTPUT_REL_PATH = Path("results/experiment_4133_capstone_v382.json")
EXPERIMENT_ID = 4133
RANDOM_SEED = 4133
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
PUBLISHED_087_TARGET = 0.87
PUBLISHED_MATCH_TOLERANCE = 0.02
V381_REFERENCE_DELTA = 0.01

INITIAL_REFERENCE_PATH = Path("results/experiment_4108_nanotrm_sudoku_extreme_baseline.json")
UPSTREAM_IDS = (4126, 4127, 4128, 4129, 4130, 4131, 4132)
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    4126: Path("results/experiment_4126_lr_resume_correctness_fix.json"),
    4127: Path("results/experiment_4127_sudoku_extreme_accumulate_fixed.json"),
    4128: Path("results/experiment_4128_carnot_verifier_graft_sudoku.json"),
    4129: Path("results/experiment_4129_fourteenth_game_explore_first.json"),
    4130: Path("results/experiment_4130_sota_ingestion_resumable_training.json"),
    4131: Path("results/experiment_4131_verifier_registry_gaps_hygiene.json"),
    4132: Path("results/experiment_4132_hardware_continuity.json"),
}

HEADLINE_OUTCOMES = {
    "lr_fixed_baseline_reproduced_graft_validated",
    "lr_fixed_baseline_reproduced_graft_null",
    "lr_fixed_accumulating_v383_continues",
    "lr_fix_failed_contiguous_run_recommended",
    "baseline_still_blocked",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "baseline_val_trajectory",
    "upstream_provenance",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'lr fixed, still accumulating, .383 continues' "
        "is COMPLETE and valuable."
    ),
    "headline_outcome": (
        "One of the enumerated set -- forces a single unambiguous read of the .382 result."
    ),
    "baseline_val_trajectory": (
        "Val across .381+.382 (0.0232 -> 0.106 -> ...); shows whether the LR fix "
        "changed the convergence rate."
    ),
    "upstream_provenance": (
        "{experiment_id, fields_imported, sha256} per cited upstream -- the audit trail "
        "proving the capstone synthesizes real measurements."
    ),
}


def is_sha256(value: object) -> bool:
    """Return true only for lowercase SHA-256 hex digests."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def read_json_object(path: Path) -> JsonDict:
    """Load one JSON artifact and fail closed if it is not an object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")  # pragma: no cover
    return payload


def sha256_file(path: Path) -> str:
    """Hash an artifact so the capstone can be checked against landed bytes."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_to_root(root: Path, path: Path) -> str:
    """Return stable repository-relative paths when possible."""

    try:
        return str(path.relative_to(root))
    except ValueError:  # pragma: no cover
        return str(path)


def selected_upstream_paths(root: Path | str) -> dict[int, Path | None]:
    """Resolve the .382 upstream artifact paths under the selected root."""

    root_path = Path(root)
    return {
        experiment_id: path if (path := root_path / rel_path).exists() else None
        for experiment_id, rel_path in DEFAULT_UPSTREAM_PATHS.items()
    }


def flagged(payload: Mapping[str, Any] | None) -> bool:
    """Return whether an upstream is stamped adversarial."""

    return isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True


def verdict_text(payload: Mapping[str, Any] | None) -> str:
    """Read an upstream honest verdict without coercing malformed values."""

    value = payload.get("honest_verdict") if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def bool_metric(payload: Mapping[str, Any] | None, field: str) -> bool:
    """Read a JSON boolean without treating integers or strings as booleans."""

    return isinstance(payload, Mapping) and payload.get(field) is True


def int_metric(payload: Mapping[str, Any] | None, field: str) -> int:
    """Read an integer counter while rejecting bool, because bool is an int subclass."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def float_metric(payload: Mapping[str, Any] | None, field: str) -> float:
    """Read a numeric metric while rejecting bools and strings."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def str_metric(payload: Mapping[str, Any] | None, field: str) -> str:
    """Read a string metric without guessing from other types."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def list_float_metric(payload: Mapping[str, Any] | None, field: str) -> list[float]:
    """Read numeric list entries and drop malformed endpoints."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    if not isinstance(value, list):
        return []
    return [
        float(item)
        for item in value
        if isinstance(item, int | float) and not isinstance(item, bool)
    ]


def _optional_float(payload: Mapping[str, Any] | None, field: str) -> float | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _nested_map(payload: Mapping[str, Any] | None, field: str) -> Mapping[str, Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _ci_excludes_zero_positive(ci: list[float]) -> bool:
    return len(ci) == 2 and ci[0] > 0.0 and ci[1] > 0.0


def clean_val_points(rows: object) -> list[JsonDict]:
    """Keep only val-trajectory rows that contain real numeric validation values."""

    if not isinstance(rows, list):
        return []
    points: list[JsonDict] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        value = _optional_float(row, "val_exact_accuracy")
        if value is None:
            continue
        pass_index = int_metric(row, "pass_index")
        points.append(
            {
                "experiment_id": 4127,
                "pass_id": "v381_starting_baseline"
                if pass_index == 0
                else f"v382_fixed_lr_pass_{pass_index}",
                "kind": str_metric(row, "kind"),
                "source": str_metric(row, "source"),
                "source_field": "val_trajectory[].val_exact_accuracy",
                "pass_index": pass_index,
                "val_exact_accuracy": value,
                "val_exact_accuracy_rounded": round(value, 4),
                "delta_vs_previous": _optional_float(row, "delta_vs_previous"),
                "checkpoint_reload_ok": bool_metric(row, "checkpoint_reload_ok")
                if "checkpoint_reload_ok" in row
                else None,
            }
        )
    return points


def initial_reference_point(root: Path) -> JsonDict | None:
    """Include the pre-resume 0.0232 context when the prior artifact is available."""

    path = root / INITIAL_REFERENCE_PATH
    if not path.exists():
        return None
    payload = read_json_object(path)
    value = _optional_float(payload, "reproduced_exact_accuracy")
    if value is None:
        return None
    return {
        "experiment_id": 4108,
        "path": relative_to_root(root, path),
        "sha256": sha256_file(path),
        "pass_id": "pre_v381_reference",
        "source_field": "reproduced_exact_accuracy",
        "val_exact_accuracy": value,
        "val_exact_accuracy_rounded": round(value, 4),
    }


def baseline_val_trajectory(
    clean_upstreams: Mapping[int, Mapping[str, Any]],
    *,
    root: Path,
) -> JsonDict:
    """Build the .381 + .382 validation trajectory and acceleration answer."""

    exp4127 = clean_upstreams.get(4127)
    upstream_points = clean_val_points(
        exp4127.get("val_trajectory") if isinstance(exp4127, Mapping) else None
    )
    points: list[JsonDict] = []
    initial = initial_reference_point(root)
    if initial is not None:
        points.append(initial)
    points.extend(upstream_points)
    values = [float(point["val_exact_accuracy"]) for point in points]
    upstream_values = [float(point["val_exact_accuracy"]) for point in upstream_points]
    deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
    upstream_deltas = [
        upstream_values[index] - upstream_values[index - 1]
        for index in range(1, len(upstream_values))
    ]

    comparison = _nested_map(exp4127, "per_pass_delta_vs_v381")
    comparison_deltas = list_float_metric(comparison, "deltas")
    mean_delta = _optional_float(comparison, "mean_delta")
    reference_delta = _optional_float(comparison, "reference_delta")
    beats_v381 = bool_metric(comparison, "beats_v381")
    final_value = upstream_values[-1] if upstream_values else (values[-1] if values else None)
    start_value = upstream_values[0] if upstream_values else None
    moved_toward_target = (
        final_value is not None
        and start_value is not None
        and final_value > start_value
        and final_value < PUBLISHED_087_TARGET + PUBLISHED_MATCH_TOLERANCE
    )
    accelerated = beats_v381 and moved_toward_target

    if accelerated and bool_metric(exp4127, "matches_published_087"):
        status = "accelerated_and_reproduced"
    elif accelerated:
        status = "accelerated_but_still_accumulating"
    elif not upstream_values:
        status = "missing_clean_val_trajectory"
    else:
        status = "not_accelerated"

    return {
        "status": status,
        "values": values,
        "rounded_values": [round(value, 4) for value in values],
        "points": points,
        "deltas": deltas,
        "rounded_deltas": [round(delta, 4) for delta in deltas],
        "upstream_values": upstream_values,
        "upstream_deltas": upstream_deltas,
        "accelerated_vs_v381": accelerated,
        "moved_toward_087": moved_toward_target,
        "final_val_exact_accuracy": final_value,
        "published_exact_accuracy_target": PUBLISHED_087_TARGET,
        "published_match_tolerance": PUBLISHED_MATCH_TOLERANCE,
        "final_to_target_gap": PUBLISHED_087_TARGET - final_value
        if final_value is not None
        else None,
        "per_pass_delta_vs_v381": {
            "beats_v381": beats_v381,
            "comparison": str_metric(comparison, "comparison"),
            "deltas": comparison_deltas,
            "mean_delta": mean_delta if mean_delta is not None else 0.0,
            "reference_delta": reference_delta
            if reference_delta is not None
            else V381_REFERENCE_DELTA,
        },
    }


def lr_fix_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Answer whether Exp 4126 landed the LR-continuity prerequisite."""

    if was_skipped:
        status = "skipped_flagged_adversarial"
    elif not isinstance(payload, Mapping):
        status = "missing"
    elif bool_metric(payload, "lr_continuous_across_resume"):
        status = "fixed_lr_resume_continuous"
    else:
        status = "lr_fix_failed"
    return {
        "status": status,
        "lr_continuous_across_resume": status == "fixed_lr_resume_continuous",
        "validation_first_lr": _optional_float(payload, "validation_first_lr")
        if not was_skipped
        else None,
        "fresh_warmup_lr": _optional_float(payload, "fresh_warmup_lr")
        if not was_skipped
        else None,
        "prior_pass_last_lr": _optional_float(payload, "prior_pass_last_lr")
        if not was_skipped
        else None,
        "stable_checkpoint_path": str_metric(payload, "stable_checkpoint_path")
        if not was_skipped
        else "",
        "manual_lr_step_restored": int_metric(payload, "manual_lr_step_restored")
        if not was_skipped
        else 0,
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def baseline_answer(
    payload: Mapping[str, Any] | None,
    *,
    trajectory: Mapping[str, Any],
    was_skipped: bool,
) -> JsonDict:
    """Answer whether the corrected schedule reproduced the 0.87 baseline."""

    if was_skipped:
        status = "skipped_flagged_adversarial"
    elif not isinstance(payload, Mapping):
        status = "missing"
    elif bool_metric(payload, "matches_published_087"):
        status = "baseline_reproduced"
    elif trajectory.get("accelerated_vs_v381") is True:
        status = "still_accumulating"
    else:
        status = "blocked_or_not_accelerating"
    return {
        "status": status,
        "matches_published_087": status == "baseline_reproduced",
        "accelerated_vs_v381": trajectory.get("accelerated_vs_v381") is True,
        "val_exact_accuracy": trajectory.get("final_val_exact_accuracy"),
        "stable_checkpoint_path": str_metric(payload, "stable_checkpoint_path")
        if not was_skipped
        else "",
        "contiguous_run_recommendation": payload.get("contiguous_run_recommendation")
        if isinstance(payload, Mapping) and not was_skipped
        else None,
        "acceptance_gate_passed": bool_metric(payload, "acceptance_gate_passed")
        if not was_skipped
        else False,
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def graft_answer(
    payload: Mapping[str, Any] | None,
    *,
    baseline_matches: bool,
    was_skipped: bool,
) -> JsonDict:
    """Answer whether the verifier graft added value, while honoring skip flags."""

    artifact_status = (
        "skipped_flagged_adversarial"
        if was_skipped
        else "present"
        if isinstance(payload, Mapping)
        else "missing"
    )
    if not baseline_matches:
        return {
            "status": "deferred_by_baseline_not_reproduced",
            "exp4128_artifact_status": artifact_status,
            "graft_deferred": True,
            "verifier_value_added": False,
            "rft_vs_ablation_delta": None,
            "rft_vs_ablation_ci95": None,
            "rerank_lift_vs_vote": None,
            "estimated_passes_to_converge_for_383": None,
            "honest_verdict": "",
        }
    if was_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "exp4128_artifact_status": artifact_status,
            "graft_deferred": None,
            "verifier_value_added": None,
            "rft_vs_ablation_delta": None,
            "rft_vs_ablation_ci95": None,
            "rerank_lift_vs_vote": None,
            "estimated_passes_to_converge_for_383": None,
            "honest_verdict": "",
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "missing",
            "exp4128_artifact_status": artifact_status,
            "graft_deferred": None,
            "verifier_value_added": None,
            "rft_vs_ablation_delta": None,
            "rft_vs_ablation_ci95": None,
            "rerank_lift_vs_vote": None,
            "estimated_passes_to_converge_for_383": None,
            "honest_verdict": "",
        }
    if bool_metric(payload, "graft_deferred"):
        return {
            "status": "graft_deferred",
            "exp4128_artifact_status": artifact_status,
            "graft_deferred": True,
            "verifier_value_added": False,
            "rft_vs_ablation_delta": None,
            "rft_vs_ablation_ci95": None,
            "rerank_lift_vs_vote": None,
            "estimated_passes_to_converge_for_383": _nested_map(
                payload, "estimated_passes_to_converge_for_383"
            ),
            "honest_verdict": verdict_text(payload),
        }

    rft = _nested_map(payload, "rft_vs_ablation_delta")
    rerank = _nested_map(payload, "rerank_lift_vs_vote")
    ci = list_float_metric(rft, "ci95")
    value_added = bool_metric(payload, "verifier_value_added") and _ci_excludes_zero_positive(ci)
    return {
        "status": "verifier_value_added" if value_added else "null_or_inconclusive",
        "exp4128_artifact_status": artifact_status,
        "graft_deferred": False,
        "verifier_value_added": value_added,
        "rft_vs_ablation_delta": float_metric(rft, "delta"),
        "rft_vs_ablation_ci95": ci,
        "rerank_lift_vs_vote": {
            "delta": float_metric(rerank, "delta"),
            "ci95": list_float_metric(rerank, "ci95"),
            "metric": str_metric(rerank, "metric"),
        },
        "estimated_passes_to_converge_for_383": {},
        "honest_verdict": verdict_text(payload),
    }


def arc_games_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Carry the clean Exp 4129 ARC game count without inferring a solve."""

    if was_skipped:
        status = "skipped_flagged_adversarial"
    elif not isinstance(payload, Mapping):
        status = "missing"
    elif (
        bool_metric(payload, "game_solved")
        and bool_metric(payload, "real_env_confirmed")
        and int_metric(payload, "total_games_solved")
        > int_metric(payload, "prior_total_games_solved")
    ):
        status = "new_game_solved"
    else:
        status = "measured_no_new_solve"
    return {
        "status": status,
        "prior_total_games_solved": int_metric(payload, "prior_total_games_solved")
        if isinstance(payload, Mapping) and not was_skipped
        else 12,
        "total_games_solved": int_metric(payload, "total_games_solved")
        if isinstance(payload, Mapping) and not was_skipped
        else 12,
        "game_solved": bool_metric(payload, "game_solved") if not was_skipped else False,
        "real_env_confirmed": bool_metric(payload, "real_env_confirmed")
        if not was_skipped
        else False,
        "levels_completed": int_metric(payload, "levels_completed") if not was_skipped else 0,
        "target_game": str_metric(payload, "target_game") if not was_skipped else "",
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def sota_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Summarize clean Exp 4130 SOTA ingestion as supporting context only."""

    methods = (
        payload.get("methods_mapped") if isinstance(payload, Mapping) and not was_skipped else None
    )
    return {
        "included": isinstance(payload, Mapping) and not was_skipped,
        "methods_mapped": [dict(item) for item in methods if isinstance(item, Mapping)]
        if isinstance(methods, list)
        else [],
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def registry_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Summarize clean Exp 4131 registry hygiene when it is not flagged."""

    return {
        "included": isinstance(payload, Mapping) and not was_skipped,
        "status": "skipped_flagged_adversarial"
        if was_skipped
        else "included"
        if isinstance(payload, Mapping)
        else "missing",
        "regression_guard_passed": bool_metric(payload, "regression_guard_passed")
        if not was_skipped
        else False,
        "lr_resume_fix_status": str_metric(_nested_map(payload, "lr_resume_fix"), "status")
        if not was_skipped
        else "",
        "sudoku_baseline_status": str_metric(_nested_map(payload, "sudoku_baseline"), "status")
        if not was_skipped
        else "",
        "sudoku_graft_status": str_metric(_nested_map(payload, "sudoku_graft"), "status")
        if not was_skipped
        else "",
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def hardware_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Carry clean Exp 4132 hardware continuity apart from science claims."""

    reachability = (
        payload.get("per_board_reachability")
        if isinstance(payload, Mapping) and not was_skipped
        else None
    )
    return {
        "included": isinstance(payload, Mapping) and not was_skipped,
        "kv260_terminal_confirmed": bool_metric(payload, "kv260_terminal_confirmed")
        if not was_skipped
        else False,
        "per_board_reachability": dict(reachability) if isinstance(reachability, Mapping) else {},
        "gatemate_step_taken": str_metric(payload, "gatemate_step_taken")
        if not was_skipped
        else "",
        "polarfire_step_taken": str_metric(payload, "polarfire_step_taken")
        if not was_skipped
        else "",
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def headline_outcome(
    lr_fix: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    baseline: Mapping[str, Any],
    graft: Mapping[str, Any],
) -> str:
    """Choose the single enumerated .382 headline."""

    if lr_fix.get("lr_continuous_across_resume") is not True:
        return "lr_fix_failed_contiguous_run_recommended"
    if baseline.get("matches_published_087") is True:
        if graft.get("status") == "verifier_value_added":
            return "lr_fixed_baseline_reproduced_graft_validated"
        return "lr_fixed_baseline_reproduced_graft_null"
    if trajectory.get("accelerated_vs_v381") is True:
        return "lr_fixed_accumulating_v383_continues"
    return "baseline_still_blocked"


def headline_answers(
    lr_fix: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    baseline: Mapping[str, Any],
    graft: Mapping[str, Any],
    arc_games: Mapping[str, Any],
) -> JsonDict:
    """Expose the concrete operator questions in machine-checkable fields."""

    return {
        "exp4126_lr_resume_fix_landed": lr_fix.get("lr_continuous_across_resume") is True,
        "exp4127_corrected_schedule_accelerated": trajectory.get("accelerated_vs_v381")
        is True,
        "exp4127_matches_published_087": baseline.get("matches_published_087") is True,
        "exp4128_graft_or_defer": str(graft.get("status", "")),
        "exp4128_verifier_value_added": graft.get("verifier_value_added") is True,
        "total_arc_games_solved": int(arc_games.get("total_games_solved", 0)),
    }


def verdict(
    outcome: str,
    lr_fix: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    baseline: Mapping[str, Any],
    graft: Mapping[str, Any],
    games_solved_total: int,
    skipped_count: int,
) -> str:
    """Build a terminal-prefixed verdict from the already-selected headline."""

    prefix = (
        "success:"
        if outcome == "lr_fixed_baseline_reproduced_graft_validated"
        else "blocked:"
        if outcome in {"lr_fix_failed_contiguous_run_recommended", "baseline_still_blocked"}
        else "complete:"
    )
    lr_fixed_flag = int(lr_fix.get("lr_continuous_across_resume") is True)
    accelerated_flag = int(trajectory.get("accelerated_vs_v381") is True)
    baseline_flag = int(baseline.get("matches_published_087") is True)
    graft_status = str(graft.get("status", "missing")) or "missing"
    return (
        f"{prefix} capstone_v382_{outcome}_lr_fixed{lr_fixed_flag}_"
        f"accelerated{accelerated_flag}_baseline087{baseline_flag}_graft_{graft_status}_"
        f"games{games_solved_total}_flagged_skipped{skipped_count}"
    )


def flagged_artifacts_skipped(
    root: Path,
    paths: Mapping[int, Path | None],
    upstreams: Mapping[int, Mapping[str, Any] | None],
    skipped_ids: set[int],
) -> list[JsonDict]:
    """Record every upstream excluded before metric import."""

    rows: list[JsonDict] = []
    for experiment_id in sorted(skipped_ids):
        path = paths[experiment_id]
        rows.append(
            {
                "experiment_id": experiment_id,
                "path": relative_to_root(root, path) if path is not None else "",
                "reason": "flagged_adversarial:true"
                if flagged(upstreams[experiment_id])
                else "unknown",
                "sha256": sha256_file(path) if path is not None else "",
            }
        )
    return rows


def imported_fields_by_id(clean_ids: set[int]) -> dict[int, list[str]]:
    """Name exactly which fields each clean upstream contributes."""

    fields: dict[int, list[str]] = {experiment_id: [] for experiment_id in UPSTREAM_IDS}
    if 4126 in clean_ids:
        fields[4126] = [
            "lr_continuous_across_resume",
            "validation_first_lr",
            "fresh_warmup_lr",
            "prior_pass_last_lr",
            "stable_checkpoint_path",
            "manual_lr_step_restored",
        ]
    if 4127 in clean_ids:
        fields[4127] = [
            "val_trajectory",
            "per_pass_delta_vs_v381",
            "matches_published_087",
            "stable_checkpoint_path",
            "acceptance_gate_passed",
            "contiguous_run_recommendation",
        ]
    if 4128 in clean_ids:
        fields[4128] = [
            "graft_deferred",
            "verifier_value_added",
            "rft_vs_ablation_delta.delta",
            "rft_vs_ablation_delta.ci95",
            "rerank_lift_vs_vote.delta",
            "rerank_lift_vs_vote.ci95",
        ]
    if 4129 in clean_ids:
        fields[4129] = [
            "prior_total_games_solved",
            "total_games_solved",
            "game_solved",
            "real_env_confirmed",
            "levels_completed",
            "target_game",
        ]
    if 4130 in clean_ids:
        fields[4130] = ["methods_mapped"]
    if 4131 in clean_ids:
        fields[4131] = [
            "regression_guard_passed",
            "lr_resume_fix.status",
            "sudoku_baseline.status",
            "sudoku_graft.status",
        ]
    if 4132 in clean_ids:
        fields[4132] = [
            "kv260_terminal_confirmed",
            "per_board_reachability",
            "gatemate_step_taken",
            "polarfire_step_taken",
        ]
    return fields


def upstream_provenance(
    root: Path,
    paths: Mapping[int, Path | None],
    upstreams: Mapping[int, Mapping[str, Any] | None],
    skipped_ids: set[int],
    fields_by_id: Mapping[int, list[str]],
) -> list[JsonDict]:
    """Cite every existing .382 upstream and list imported fields."""

    rows: list[JsonDict] = []
    for experiment_id in UPSTREAM_IDS:
        path = paths[experiment_id]
        if path is None:
            continue
        skipped = experiment_id in skipped_ids
        rows.append(
            {
                "experiment_id": experiment_id,
                "path": relative_to_root(root, path),
                "sha256": sha256_file(path),
                "fields_imported": [] if skipped else list(fields_by_id.get(experiment_id, [])),
                "skipped": skipped,
                "skip_reason": "flagged_adversarial:true" if skipped else "",
                "honest_verdict": verdict_text(upstreams[experiment_id])
                if isinstance(upstreams[experiment_id], Mapping)
                else "",
            }
        )
    return rows


def missing_upstream_artifacts(paths: Mapping[int, Path | None]) -> list[JsonDict]:
    """Record absent upstream artifacts without inventing their numbers."""

    return [
        {"experiment_id": experiment_id}
        for experiment_id in UPSTREAM_IDS
        if paths[experiment_id] is None
    ]


def upstream_artifact_state(
    root: Path,
    paths: Mapping[int, Path | None],
    upstreams: Mapping[int, Mapping[str, Any] | None],
    skipped_ids: set[int],
    clean_ids: set[int],
) -> dict[str, JsonDict]:
    """Expose inclusion state for skipped, clean, and missing inputs."""

    state: dict[str, JsonDict] = {}
    for experiment_id in UPSTREAM_IDS:
        path = paths[experiment_id]
        payload = upstreams[experiment_id]
        state[str(experiment_id)] = {
            "exists": path is not None,
            "path": relative_to_root(root, path) if path is not None else "",
            "honest_verdict": verdict_text(payload) if isinstance(payload, Mapping) else "missing",
            "flagged_adversarial": flagged(payload),
            "included": experiment_id in clean_ids,
            "skipped": experiment_id in skipped_ids,
        }
    return state


def duration_from(started_s: float, now_s: float | None) -> float:
    """Compute aggregation duration with a nonzero floor for reproducible tests."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return max(0.0001, end - started_s)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return a reproducibility checksum excluding the checksum field itself."""

    stable = dict(payload)
    stable.pop("reproducibility_checksum", None)
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Build the .382 capstone from landed upstream artifacts."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    paths = selected_upstream_paths(root_path)
    upstreams: dict[int, Mapping[str, Any] | None] = {
        experiment_id: read_json_object(path) if path is not None else None
        for experiment_id, path in paths.items()
    }
    skipped_ids = {
        experiment_id for experiment_id, payload in upstreams.items() if flagged(payload)
    }
    clean_ids = {
        experiment_id
        for experiment_id, payload in upstreams.items()
        if isinstance(payload, Mapping) and experiment_id not in skipped_ids
    }
    clean_upstreams = {experiment_id: upstreams[experiment_id] for experiment_id in clean_ids}

    lr_fix = lr_fix_answer(clean_upstreams.get(4126), was_skipped=4126 in skipped_ids)
    trajectory = baseline_val_trajectory(clean_upstreams, root=root_path)
    baseline = baseline_answer(
        clean_upstreams.get(4127),
        trajectory=trajectory,
        was_skipped=4127 in skipped_ids,
    )
    graft = graft_answer(
        clean_upstreams.get(4128),
        baseline_matches=baseline["matches_published_087"] is True,
        was_skipped=4128 in skipped_ids,
    )
    games = arc_games_answer(clean_upstreams.get(4129), was_skipped=4129 in skipped_ids)
    sota = sota_answer(clean_upstreams.get(4130), was_skipped=4130 in skipped_ids)
    registry = registry_answer(clean_upstreams.get(4131), was_skipped=4131 in skipped_ids)
    hardware = hardware_answer(clean_upstreams.get(4132), was_skipped=4132 in skipped_ids)
    outcome = headline_outcome(lr_fix, trajectory, baseline, graft)
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_ids)
    total_games = int(games["total_games_solved"])
    fields_by_id = imported_fields_by_id(clean_ids)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v382_4133.v1",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": verdict(
            outcome, lr_fix, trajectory, baseline, graft, total_games, len(skipped)
        ),
        "headline_outcome": outcome,
        "headline_answers": headline_answers(lr_fix, trajectory, baseline, graft, games),
        "baseline_val_trajectory": trajectory,
        "lr_resume_fix": lr_fix,
        "baseline_reproduction": baseline,
        "sudoku_verifier_graft": graft,
        "arc_games": games,
        "total_arc_games_solved": total_games,
        "sota_ingestion": sota,
        "registry_gap_hygiene": registry,
        "hardware_continuity": hardware,
        "flagged_artifacts_skipped": skipped,
        "upstream_provenance": upstream_provenance(
            root_path, paths, upstreams, skipped_ids, fields_by_id
        ),
        "missing_upstream_artifacts": missing_upstream_artifacts(paths),
        "upstream_artifact_state": upstream_artifact_state(
            root_path, paths, upstreams, skipped_ids, clean_ids
        ),
        "duration_s": duration_from(start, now_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the fields that keep the .382 headline auditable."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover
    verdict_value = str(artifact.get("honest_verdict", ""))
    if not verdict_value.startswith(("complete:", "success:", "blocked:")):
        raise ValueError("honest_verdict must use a terminal prefix")  # pragma: no cover
    if artifact.get("headline_outcome") not in HEADLINE_OUTCOMES:
        raise ValueError("headline_outcome must be one of the enumerated values")
    trajectory = artifact.get("baseline_val_trajectory")
    if not isinstance(trajectory, Mapping) or not isinstance(trajectory.get("values"), list):
        raise ValueError("baseline_val_trajectory must contain numeric values")
    if not all(
        isinstance(value, int | float) and not isinstance(value, bool)
        for value in trajectory["values"]
    ):
        raise ValueError("baseline_val_trajectory values must be numeric")  # pragma: no cover
    if not isinstance(artifact.get("total_arc_games_solved"), int) or isinstance(
        artifact.get("total_arc_games_solved"), bool
    ):
        raise ValueError("total_arc_games_solved must be a bare int")  # pragma: no cover
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError(
            "inference_substrate must be aggregation_from_upstream_artifacts"
        )  # pragma: no cover
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be an object")  # pragma: no cover
    for field, principle in FIELD_PRINCIPLES.items():
        if principles.get(field) != principle:
            raise ValueError(f"field_principles.{field} mismatch")  # pragma: no cover
    provenance = artifact.get("upstream_provenance")
    if not isinstance(provenance, list):
        raise ValueError("upstream_provenance must be a list")  # pragma: no cover
    for row in provenance:
        if not isinstance(row, Mapping):
            raise ValueError("upstream_provenance entries must be objects")  # pragma: no cover
        if not isinstance(row.get("experiment_id"), int):
            raise ValueError(
                "upstream_provenance entries need integer experiment_id"
            )  # pragma: no cover
        if not isinstance(row.get("fields_imported"), list) or not all(
            isinstance(item, str) for item in row.get("fields_imported", [])
        ):
            raise ValueError(
                "upstream_provenance fields_imported must be strings"
            )  # pragma: no cover
        if row.get("skipped") is True and row.get("fields_imported") != []:
            raise ValueError("skipped upstreams must import no fields")  # pragma: no cover
        if not is_sha256(row.get("sha256")):
            raise ValueError("upstream_provenance entries need sha256")
    if not isinstance(artifact.get("flagged_artifacts_skipped"), list):
        raise ValueError("flagged_artifacts_skipped must be a list")  # pragma: no cover
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be sha256")  # pragma: no cover


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and write the Exp 4133 capstone artifact."""

    root_path = Path(root)
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    validate_artifact(artifact)
    output = root_path / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
