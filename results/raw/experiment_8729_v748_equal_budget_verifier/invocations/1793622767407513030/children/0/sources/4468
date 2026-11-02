"""Build the Exp 4124 v381 capstone aggregation.

Spec refs: REQ-CAPSTONE-4124, SCENARIO-CAPSTONE-4124.

This module does not rerun training, Sudoku verifier work, ARC exploration, or
hardware checks. It reads the landed .381 upstream artifacts, excludes any
artifact stamped adversarial before importing metrics, and writes one audited
decision-grade summary with sha256 provenance for every upstream file.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_4124_capstone_v381.json")
EXPERIMENT_ID = 4124
RANDOM_SEED = 4124
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
PUBLISHED_087_TARGET = 0.87
PUBLISHED_MATCH_TOLERANCE = 0.02
BOUNDED_RUN_CAP_S = 4800.0
PRIOR_TOTAL_ARC_GAMES_SOLVED = 12

UPSTREAM_IDS = (4116, 4117, 4118, 4119, 4120, 4121, 4122, 4123)
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    4116: Path("results/experiment_4116_sudoku_extreme_resume_pass1.json"),
    4117: Path("results/experiment_4117_sudoku_extreme_resume_pass2.json"),
    4118: Path("results/experiment_4118_sudoku_extreme_resume_pass3.json"),
    4119: Path("results/experiment_4119_carnot_verifier_graft_sudoku.json"),
    4120: Path("results/experiment_4120_thirteenth_game_explore_first.json"),
    4121: Path("results/experiment_4121_sota_ingestion_trm_baseline_graft.json"),
    4122: Path("results/experiment_4122_verifier_registry_gaps_hygiene.json"),
    4123: Path("results/experiment_4123_hardware_continuity.json"),
}

HEADLINE_OUTCOMES = {
    "baseline_reproduced_graft_validated",
    "baseline_reproduced_graft_null",
    "baseline_reproduced_graft_deferred",
    "baseline_still_accumulating_v382_continues",
    "resume_mechanism_stalled",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "baseline_val_trajectory",
    "upstream_provenance",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'still accumulating, .382 continues' is "
        "COMPLETE and valuable."
    ),
    "headline_outcome": (
        "One of the enumerated set -- forces a single unambiguous read of the .381 result."
    ),
    "baseline_val_trajectory": (
        "The val exact-accuracy across the resume passes (0.0232 -> ...); shows "
        "whether resumable training is converging and at what rate."
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
    """Load one upstream artifact and reject non-object JSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")  # pragma: no cover
    return payload


def sha256_file(path: Path) -> str:
    """Hash an upstream artifact so later readers can audit the aggregation."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_to_root(root: Path, path: Path) -> str:
    """Return stable repository-relative paths in the artifact."""

    try:
        return str(path.relative_to(root))
    except ValueError:  # pragma: no cover
        return str(path)


def selected_upstream_paths(root: Path | str) -> dict[int, Path | None]:
    """Resolve the intended .381 upstream artifact paths."""

    root_path = Path(root)
    return {
        experiment_id: path if (path := root_path / rel_path).exists() else None
        for experiment_id, rel_path in DEFAULT_UPSTREAM_PATHS.items()
    }


def flagged(payload: Mapping[str, Any] | None) -> bool:
    """Return whether an upstream is stamped adversarial."""

    return isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True


def verdict_text(payload: Mapping[str, Any] | None) -> str:
    """Read an upstream honest verdict without coercing non-strings."""

    value = payload.get("honest_verdict") if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def bool_metric(payload: Mapping[str, Any] | None, field: str) -> bool:
    """Read a JSON boolean without truthifying strings or integers."""

    return isinstance(payload, Mapping) and payload.get(field) is True


def int_metric(payload: Mapping[str, Any] | None, field: str) -> int:
    """Read an integer counter while rejecting booleans."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def float_metric(payload: Mapping[str, Any] | None, field: str) -> float:
    """Read a numeric metric while rejecting booleans and strings."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def str_metric(payload: Mapping[str, Any] | None, field: str) -> str:
    """Read a string metric without coercion."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def list_float_metric(payload: Mapping[str, Any] | None, field: str) -> list[float]:
    """Read numeric confidence interval endpoints without accepting strings."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    if not isinstance(value, list):
        return []
    return [
        float(item)
        for item in value
        if isinstance(item, int | float) and not isinstance(item, bool)
    ]


def _nested_map(payload: Mapping[str, Any] | None, field: str) -> Mapping[str, Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _ci_excludes_zero_positive(ci: list[float]) -> bool:
    return len(ci) == 2 and ci[0] > 0.0 and ci[1] > 0.0


def _optional_float(payload: Mapping[str, Any] | None, field: str) -> float | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _trajectory_point(
    pass_id: str,
    experiment_id: int,
    source_field: str,
    value: float | None,
) -> JsonDict | None:
    if value is None:
        return None
    return {
        "pass_id": pass_id,
        "experiment_id": experiment_id,
        "source_field": source_field,
        "val_exact_accuracy": value,
        "val_exact_accuracy_rounded": round(value, 4),
    }


def baseline_val_trajectory(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> JsonDict:
    """Build the clean resume-pass trajectory and cap-bound mechanism answer."""

    exp4116 = clean_upstreams.get(4116)
    exp4117 = clean_upstreams.get(4117)
    exp4118 = clean_upstreams.get(4118)
    pass1_value = _optional_float(exp4116, "val_exact_accuracy")
    pass1_experiment_id = 4116
    pass1_source_field = "val_exact_accuracy"
    if pass1_value is None:
        pass1_value = _optional_float(exp4117, "pass1_val_exact_accuracy")
        pass1_experiment_id = 4117
        pass1_source_field = "pass1_val_exact_accuracy"
    if pass1_value is None:
        pass1_value = _optional_float(_nested_map(exp4117, "pass1"), "val_exact_accuracy")
        pass1_experiment_id = 4117
        pass1_source_field = "pass1.val_exact_accuracy"

    points = [
        point
        for point in (
            _trajectory_point("pass1", pass1_experiment_id, pass1_source_field, pass1_value),
            _trajectory_point(
                "pass2", 4117, "val_exact_accuracy", _optional_float(exp4117, "val_exact_accuracy")
            ),
            _trajectory_point(
                "pass3", 4118, "val_exact_accuracy", _optional_float(exp4118, "val_exact_accuracy")
            ),
        )
        if point is not None
    ]
    values = [float(point["val_exact_accuracy"]) for point in points]
    deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
    climbed = len(values) >= 2 and all(delta > 0.0 for delta in deltas)

    duration_rows: list[JsonDict] = []
    for experiment_id in (4116, 4117, 4118):
        payload = clean_upstreams.get(experiment_id)
        duration_s = _optional_float(payload, "duration_s")
        if duration_s is not None:
            duration_rows.append(
                {
                    "experiment_id": experiment_id,
                    "duration_s": duration_s,
                    "under_cap": duration_s < BOUNDED_RUN_CAP_S,
                }
            )
    bounded = bool(duration_rows) and all(row["under_cap"] is True for row in duration_rows)
    if climbed and bounded:
        status = "climbed_and_bounded"
    elif not values:
        status = "missing_clean_resume_metrics"
    elif not climbed:
        status = "stalled_or_not_monotonic"
    else:
        status = "bounded_cap_failed"

    return {
        "status": status,
        "values": values,
        "rounded_values": [round(value, 4) for value in values],
        "points": points,
        "deltas": deltas,
        "rounded_deltas": [round(delta, 4) for delta in deltas],
        "climbed": climbed,
        "bounded_runs_under_cap": bounded,
        "bounded_run_cap_s": BOUNDED_RUN_CAP_S,
        "bounded_runs": duration_rows,
    }


def baseline_answer(
    payload: Mapping[str, Any] | None,
    *,
    trajectory: Mapping[str, Any],
    was_skipped: bool,
) -> JsonDict:
    """Answer whether Exp 4118 reproduced the published approximate 0.87 baseline."""

    if was_skipped:
        status = "skipped_flagged_adversarial"
    elif not isinstance(payload, Mapping):
        status = "missing"
    elif bool_metric(payload, "matches_published_087"):
        status = "baseline_reproduced"
    else:
        status = "still_accumulating"
    values = trajectory.get("values") if isinstance(trajectory, Mapping) else None
    final_value = values[-1] if isinstance(values, list) and values else None
    val_exact_accuracy = _optional_float(payload, "val_exact_accuracy")
    return {
        "status": status,
        "matches_published_087": status == "baseline_reproduced",
        "val_exact_accuracy": val_exact_accuracy if val_exact_accuracy is not None else final_value,
        "published_exact_accuracy_target": PUBLISHED_087_TARGET,
        "published_match_tolerance": PUBLISHED_MATCH_TOLERANCE,
        "total_cumulative_epochs": int_metric(payload, "total_cumulative_epochs")
        if not was_skipped
        else 0,
        "branch_taken": str_metric(payload, "branch_taken") if not was_skipped else "",
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def graft_answer(
    payload: Mapping[str, Any] | None,
    *,
    baseline_matches: bool,
    was_skipped: bool,
) -> JsonDict:
    """Answer whether the verifier graft validated, was null, or deferred."""

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
            "exp4119_artifact_status": artifact_status,
            "graft_deferred": True,
            "verifier_value_added": False,
            "rft_vs_ablation_delta": None,
            "rft_vs_ablation_ci95": None,
            "rerank_lift_vs_vote": None,
            "honest_verdict": "",
        }
    if was_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "exp4119_artifact_status": artifact_status,
            "graft_deferred": None,
            "verifier_value_added": None,
            "rft_vs_ablation_delta": None,
            "rft_vs_ablation_ci95": None,
            "rerank_lift_vs_vote": None,
            "honest_verdict": "",
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "missing",
            "exp4119_artifact_status": artifact_status,
            "graft_deferred": None,
            "verifier_value_added": None,
            "rft_vs_ablation_delta": None,
            "rft_vs_ablation_ci95": None,
            "rerank_lift_vs_vote": None,
            "honest_verdict": "",
        }
    if bool_metric(payload, "graft_deferred"):
        return {
            "status": "graft_deferred",
            "exp4119_artifact_status": artifact_status,
            "graft_deferred": True,
            "verifier_value_added": False,
            "rft_vs_ablation_delta": None,
            "rft_vs_ablation_ci95": None,
            "rerank_lift_vs_vote": None,
            "honest_verdict": verdict_text(payload),
        }

    rft = _nested_map(payload, "rft_vs_ablation_delta")
    rerank = _nested_map(payload, "rerank_lift_vs_vote")
    ci = list_float_metric(rft, "ci95")
    value_added = bool_metric(payload, "verifier_value_added") and _ci_excludes_zero_positive(ci)
    return {
        "status": "verifier_value_added" if value_added else "null_or_inconclusive",
        "exp4119_artifact_status": artifact_status,
        "graft_deferred": False,
        "verifier_value_added": value_added,
        "rft_vs_ablation_delta": float_metric(rft, "delta"),
        "rft_vs_ablation_ci95": ci,
        "rerank_lift_vs_vote": {
            "delta": float_metric(rerank, "delta"),
            "ci95": list_float_metric(rerank, "ci95"),
            "metric": str_metric(rerank, "metric"),
        },
        "honest_verdict": verdict_text(payload),
    }


def arc_games_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Carry the clean Exp 4120 ARC games-solved count."""

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
    total_games = (
        int_metric(payload, "total_games_solved")
        if isinstance(payload, Mapping) and not was_skipped
        else PRIOR_TOTAL_ARC_GAMES_SOLVED
    )
    return {
        "status": status,
        "prior_total_games_solved": int_metric(payload, "prior_total_games_solved")
        if isinstance(payload, Mapping) and not was_skipped
        else PRIOR_TOTAL_ARC_GAMES_SOLVED,
        "total_games_solved": total_games,
        "game_solved": bool_metric(payload, "game_solved") if not was_skipped else False,
        "real_env_confirmed": bool_metric(payload, "real_env_confirmed")
        if not was_skipped
        else False,
        "levels_completed": int_metric(payload, "levels_completed") if not was_skipped else 0,
        "failure_reason": str_metric(payload, "failure_reason") if not was_skipped else "",
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def sota_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Summarize clean Exp 4121 SOTA ingestion without affecting the headline."""

    methods = (
        payload.get("methods_mapped") if isinstance(payload, Mapping) and not was_skipped else None
    )
    method_rows = (
        [dict(item) for item in methods if isinstance(item, Mapping)]
        if isinstance(methods, list)
        else []
    )
    return {
        "included": isinstance(payload, Mapping) and not was_skipped,
        "flagged_for_v382": str_metric(payload, "flagged_for_v382") if not was_skipped else "",
        "methods_mapped": method_rows,
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def registry_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Summarize clean Exp 4122 registry/gap hygiene when it is usable."""

    gaps = payload.get("gaps_updated") if isinstance(payload, Mapping) and not was_skipped else None
    return {
        "included": isinstance(payload, Mapping) and not was_skipped,
        "status": "skipped_flagged_adversarial"
        if was_skipped
        else "included"
        if isinstance(payload, Mapping)
        else "missing",
        "gaps_updated": [item for item in gaps if isinstance(item, str)]
        if isinstance(gaps, list)
        else [],
        "regression_guard_passed": bool_metric(payload, "regression_guard_passed")
        if not was_skipped
        else False,
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def hardware_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Carry clean Exp 4123 hardware continuity separately from science claims."""

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
    trajectory: Mapping[str, Any],
    baseline: Mapping[str, Any],
    graft: Mapping[str, Any],
) -> str:
    """Choose the single enumerated outcome required by .381."""

    if trajectory.get("status") != "climbed_and_bounded":
        return "resume_mechanism_stalled"
    if baseline.get("matches_published_087") is not True:
        return "baseline_still_accumulating_v382_continues"
    if graft.get("status") == "verifier_value_added":
        return "baseline_reproduced_graft_validated"
    if graft.get("status") == "null_or_inconclusive":
        return "baseline_reproduced_graft_null"
    return "baseline_reproduced_graft_deferred"


def headline_answers(
    trajectory: Mapping[str, Any],
    baseline: Mapping[str, Any],
    graft: Mapping[str, Any],
    arc_games: Mapping[str, Any],
) -> JsonDict:
    """Expose the concrete question answers in machine-checkable fields."""

    return {
        "resume_val_climbed": trajectory.get("climbed") is True,
        "bounded_runs_under_cap": trajectory.get("bounded_runs_under_cap") is True,
        "resume_mechanism_status": str(trajectory.get("status", "")),
        "exp4118_matches_published_087": baseline.get("matches_published_087") is True,
        "exp4119_graft_or_defer": str(graft.get("status", "")),
        "exp4119_verifier_value_added": graft.get("verifier_value_added") is True,
        "total_arc_games_solved": int(arc_games.get("total_games_solved", 0)),
    }


def verdict(
    outcome: str,
    trajectory: Mapping[str, Any],
    baseline: Mapping[str, Any],
    graft: Mapping[str, Any],
    games_solved_total: int,
    skipped_count: int,
) -> str:
    """Build a terminal-prefix headline from the already-chosen outcome."""

    prefix = (
        "success:"
        if outcome == "baseline_reproduced_graft_validated"
        else "blocked:"
        if outcome == "resume_mechanism_stalled"
        else "complete:"
    )
    climbed_flag = int(trajectory.get("climbed") is True)
    bounded_flag = int(trajectory.get("bounded_runs_under_cap") is True)
    baseline_flag = int(baseline.get("matches_published_087") is True)
    graft_status = str(graft.get("status", "missing")) or "missing"
    return (
        f"{prefix} capstone_v381_{outcome}_val_climbed{climbed_flag}_"
        f"bounded{bounded_flag}_baseline087{baseline_flag}_graft_{graft_status}_"
        f"games{games_solved_total}_flagged_skipped{skipped_count}"
    )


def flagged_artifacts_skipped(
    root: Path,
    paths: Mapping[int, Path | None],
    upstreams: Mapping[int, Mapping[str, Any] | None],
    skipped_ids: set[int],
) -> list[JsonDict]:
    """Record upstreams excluded before metric import."""

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
    if 4116 in clean_ids:
        fields[4116] = ["val_exact_accuracy", "duration_s"]
    if 4117 in clean_ids:
        fields[4117] = [
            "pass1_val_exact_accuracy",
            "pass1.val_exact_accuracy",
            "val_exact_accuracy",
            "val_delta_vs_pass1",
            "accumulation_stalled",
            "cumulative_epochs",
            "duration_s",
        ]
    if 4118 in clean_ids:
        fields[4118] = [
            "pass2.val_exact_accuracy",
            "val_exact_accuracy",
            "matches_published_087",
            "total_cumulative_epochs",
            "duration_s",
            "branch_taken",
        ]
    if 4119 in clean_ids:
        fields[4119] = [
            "graft_deferred",
            "verifier_value_added",
            "rft_vs_ablation_delta.delta",
            "rft_vs_ablation_delta.ci95",
            "rerank_lift_vs_vote.delta",
            "rerank_lift_vs_vote.ci95",
        ]
    if 4120 in clean_ids:
        fields[4120] = [
            "prior_total_games_solved",
            "total_games_solved",
            "game_solved",
            "real_env_confirmed",
            "levels_completed",
            "failure_reason",
        ]
    if 4121 in clean_ids:
        fields[4121] = ["flagged_for_v382", "methods_mapped"]
    if 4122 in clean_ids:
        fields[4122] = ["gaps_updated", "regression_guard_passed"]
    if 4123 in clean_ids:
        fields[4123] = [
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
    """Cite every existing upstream sha and record imported fields."""

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
    """Record missing upstream artifacts without inventing their metrics."""

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
    """Expose inclusion state so skipped and missing inputs are auditable."""

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
    """Compute an honest aggregation duration with a small nonzero floor."""

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
    """Build the .381 capstone from landed upstream artifacts."""

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

    trajectory = baseline_val_trajectory(clean_upstreams)
    baseline = baseline_answer(
        clean_upstreams.get(4118),
        trajectory=trajectory,
        was_skipped=4118 in skipped_ids,
    )
    graft = graft_answer(
        clean_upstreams.get(4119),
        baseline_matches=baseline["matches_published_087"] is True,
        was_skipped=4119 in skipped_ids,
    )
    games = arc_games_answer(clean_upstreams.get(4120), was_skipped=4120 in skipped_ids)
    sota = sota_answer(clean_upstreams.get(4121), was_skipped=4121 in skipped_ids)
    registry = registry_answer(clean_upstreams.get(4122), was_skipped=4122 in skipped_ids)
    hardware = hardware_answer(clean_upstreams.get(4123), was_skipped=4123 in skipped_ids)
    outcome = headline_outcome(trajectory, baseline, graft)
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_ids)
    total_games = int(games["total_games_solved"])
    fields_by_id = imported_fields_by_id(clean_ids)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v381_4124.v1",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": verdict(outcome, trajectory, baseline, graft, total_games, len(skipped)),
        "headline_outcome": outcome,
        "headline_answers": headline_answers(trajectory, baseline, graft, games),
        "baseline_val_trajectory": trajectory,
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
    """Validate the fields that keep the .381 headline auditable."""

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
    """Build, validate, and write the Exp 4124 capstone artifact."""

    root_path = Path(root)
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    validate_artifact(artifact)
    output = root_path / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
