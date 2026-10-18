"""Build the Exp 4155 v384 capstone aggregation.

Spec refs: REQ-CAPSTONE-4155, SCENARIO-CAPSTONE-4155.

This module is a reader, not another experiment runner. It loads the landed
.384 artifacts, excludes every upstream stamped ``flagged_adversarial: true``
before importing metrics, and emits the single decision-grade headline the next
planner needs. The prior clean ARC total is carried only from an explicitly
cited prior capstone, because the .384 ARC artifact can itself be flagged.
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
OUTPUT_REL_PATH = Path("results/experiment_4155_capstone_v384.json")
EXPERIMENT_ID = 4155
RANDOM_SEED = 4155
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
PUBLISHED_087_TARGET = 0.87
PUBLISHED_MATCH_TOLERANCE = 0.02

PRIOR_BASELINE_REFERENCE_PATH = Path("results/experiment_4133_capstone_v382.json")
PRIOR_ARC_REFERENCE_PATH = Path("results/experiment_4144_capstone_v383.json")
UPSTREAM_IDS = (4146, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154)
BASELINE_PASS_IDS = (4146, 4147, 4148, 4149)
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    4146: Path("results/experiment_4146_sudoku_accumulate_pass1_epochfix.json"),
    4147: Path("results/experiment_4147_sudoku_accumulate_pass2.json"),
    4148: Path("results/experiment_4148_sudoku_accumulate_pass3.json"),
    4149: Path("results/experiment_4149_sudoku_accumulate_pass4_convergence.json"),
    4150: Path("results/experiment_4150_decisive_verifier_graft_sudoku.json"),
    4151: Path("results/experiment_4151_arc_incremental_progress.json"),
    4152: Path("results/experiment_4152_sota_ingestion_recursive_reasoner_verifier.json"),
    4153: Path("results/experiment_4153_verifier_registry_gaps_hygiene.json"),
    4154: Path("results/experiment_4154_hardware_continuity.json"),
}

HEADLINE_OUTCOMES = {
    "baseline_converged_graft_validated",
    "baseline_converged_graft_null",
    "baseline_converged_graft_deferred",
    "accumulation_unstalled_still_climbing_v385",
    "accumulation_still_blocked",
}
DIFFUSIONGEMMA_GATE_STATUSES = {"RESOLVED-positive", "RESOLVED-null", "STILL-PENDING"}
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "baseline_val_trajectory",
    "diffusiongemma_gate_status",
    "upstream_provenance",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'unstalled, still climbing, .385 continues' "
        "is COMPLETE and valuable."
    ),
    "headline_outcome": (
        "One of the enumerated set -- forces a single unambiguous read of the .384 result."
    ),
    "baseline_val_trajectory": (
        "Val across .382-.384 (0.106 -> 0.278 -> ...); shows whether the epoch-fix "
        "restored convergence."
    ),
    "diffusiongemma_gate_status": (
        "RESOLVED-positive / RESOLVED-null / STILL-PENDING -- the explicit status of "
        "the queued DiffusionGemma gate after the decisive graft."
    ),
    "upstream_provenance": (
        "{experiment_id, fields_imported, sha256} per cited upstream -- the audit trail."
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
    """Load one artifact and fail closed when it is not a JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")  # pragma: no cover
    return payload


def sha256_file(path: Path) -> str:
    """Hash landed artifact bytes so later readers can audit this synthesis."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_to_root(root: Path, path: Path) -> str:
    """Return stable repository-relative paths when the path is under root."""

    try:
        return str(path.relative_to(root))
    except ValueError:  # pragma: no cover
        return str(path)


def selected_upstream_paths(root: Path | str) -> dict[int, Path | None]:
    """Resolve the .384 upstream artifact paths under the selected root."""

    root_path = Path(root)
    return {
        experiment_id: path if (path := root_path / rel_path).exists() else None
        for experiment_id, rel_path in DEFAULT_UPSTREAM_PATHS.items()
    }


def flagged(payload: Mapping[str, Any] | None) -> bool:
    """Return whether an upstream is stamped adversarial."""

    return isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True


def verdict_text(payload: Mapping[str, Any] | None) -> str:
    """Read an upstream honest verdict without inventing a fallback verdict."""

    value = payload.get("honest_verdict") if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def bool_metric(payload: Mapping[str, Any] | None, field: str) -> bool | None:
    """Read a JSON boolean without treating integers or strings as booleans."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, bool) else None


def int_metric(payload: Mapping[str, Any] | None, field: str) -> int:
    """Read an integer counter while rejecting bool, because bool subclasses int."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def float_metric(payload: Mapping[str, Any] | None, field: str) -> float | None:
    """Read a numeric metric while rejecting bools and strings."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


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


def nested_map(payload: Mapping[str, Any] | None, field: str) -> Mapping[str, Any]:
    """Return a nested mapping or an empty mapping for malformed input."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def ci_excludes_zero_positive(ci: list[float]) -> bool:
    """Return whether a two-sided CI is strictly positive."""

    return len(ci) == 2 and ci[0] > 0.0 and ci[1] > 0.0


def experiment_id_from_name(name: str) -> int:
    """Extract an experiment id from the conventional artifact name."""

    marker = "experiment_"
    if not name.startswith(marker):
        return 0
    suffix = name[len(marker) :].split("_", 1)[0]
    return int(suffix) if suffix.isdigit() else 0


def prior_baseline_values(root: Path) -> JsonDict:
    """Read the clean .382 baseline lineage used as the .384 starting point."""

    path = root / PRIOR_BASELINE_REFERENCE_PATH
    if not path.exists():
        return {"values": [], "points": [], "path": "", "sha256": ""}
    payload = read_json_object(path)
    trajectory = nested_map(payload, "baseline_val_trajectory")
    raw_values = trajectory.get("upstream_values")
    if not isinstance(raw_values, list):
        raw_values = trajectory.get("values")
    values = [
        float(value)
        for value in raw_values
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    if len(values) > 2:
        values = values[-2:]
    points = [
        {
            "experiment_id": 4133,
            "label": ".382_reference",
            "source_field": "baseline_val_trajectory.upstream_values",
            "val_exact_accuracy": value,
            "val_exact_accuracy_rounded": round(value, 4),
        }
        for value in values
    ]
    return {
        "values": values,
        "points": points,
        "path": relative_to_root(root, path),
        "sha256": sha256_file(path),
    }


def clean_v384_points(rows: object) -> list[JsonDict]:
    """Keep v384 trajectory rows with real validation values."""

    if not isinstance(rows, list):
        return []
    points: list[JsonDict] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        value = float_metric(row, "val_exact_accuracy")
        if value is None:
            value = float_metric(row, "effective_val_exact_accuracy")
        if value is None:
            continue
        points.append(
            {
                "experiment_id": experiment_id_from_name(str_metric(row, "experiment")),
                "label": str_metric(row, "pass_label"),
                "post_epoch": int_metric(row, "post_epoch") or None,
                "source_field": "val_trajectory_v384[].val_exact_accuracy",
                "val_exact_accuracy": value,
                "val_exact_accuracy_rounded": round(value, 4),
            }
        )
    return points


def epoch_fix_unstalled(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> bool:
    """Answer whether Exp 4146 proved real epoch advance plus validation."""

    if was_skipped or not isinstance(payload, Mapping):
        return False
    seed_epoch = int_metric(payload, "seed_epoch")
    post_epoch = int_metric(payload, "post_epoch")
    return post_epoch > seed_epoch and float_metric(payload, "val_exact_accuracy") is not None


def attempted_baseline_passes(
    clean_upstreams: Mapping[int, Mapping[str, Any]],
    skipped_ids: set[int],
) -> list[JsonDict]:
    """Expose which .384 baseline passes were usable without importing skipped metrics."""

    rows: list[JsonDict] = []
    for offset, experiment_id in enumerate(BASELINE_PASS_IDS, start=1):
        payload = clean_upstreams.get(experiment_id)
        if experiment_id in skipped_ids:
            included = False
            status = "skipped_flagged_adversarial"
        elif isinstance(payload, Mapping):
            included = True
            status = verdict_text(payload) or "included"
        else:
            included = False
            status = "missing"
        rows.append(
            {
                "experiment_id": experiment_id,
                "pass_index": offset,
                "included": included,
                "status": status,
            }
        )
    return rows


def baseline_val_trajectory(
    clean_upstreams: Mapping[int, Mapping[str, Any]],
    *,
    root: Path,
    skipped_ids: set[int],
) -> JsonDict:
    """Build the .382-.384 validation trajectory and convergence answer."""

    prior = prior_baseline_values(root)
    points = list(prior["points"])
    exp4149 = clean_upstreams.get(4149)
    rows = clean_v384_points(
        exp4149.get("val_trajectory_v384") if isinstance(exp4149, Mapping) else None
    )
    if not rows:
        for experiment_id in BASELINE_PASS_IDS:
            payload = clean_upstreams.get(experiment_id)
            value = float_metric(payload, "val_exact_accuracy")
            if value is not None:
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "label": f"pass{experiment_id - 4145}",
                        "post_epoch": int_metric(payload, "post_epoch") or None,
                        "source_field": "val_exact_accuracy",
                        "val_exact_accuracy": value,
                        "val_exact_accuracy_rounded": round(value, 4),
                    }
                )
    if points and rows and rows[0]["val_exact_accuracy"] == points[-1]["val_exact_accuracy"]:
        rows = rows[1:]
    points.extend(rows)
    values = [float(point["val_exact_accuracy"]) for point in points]
    deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
    unstalled = epoch_fix_unstalled(
        clean_upstreams.get(4146),
        was_skipped=4146 in skipped_ids,
    )
    matches = bool_metric(exp4149, "matches_published_087") if isinstance(exp4149, Mapping) else None
    final_value = float_metric(exp4149, "val_exact_accuracy") if isinstance(exp4149, Mapping) else None
    if final_value is None and values:
        final_value = values[-1]
    moved_after_382 = len(values) >= 3 and values[-1] > values[1]
    baseline_skipped = any(experiment_id in skipped_ids for experiment_id in BASELINE_PASS_IDS)

    if matches is True:
        status = "baseline_converged"
    elif unstalled and moved_after_382:
        status = "accumulation_unstalled_still_climbing_v385"
    elif baseline_skipped or isinstance(exp4149, Mapping):
        status = "accumulation_still_blocked"
    else:
        status = "missing"

    return {
        "status": status,
        "values": values,
        "rounded_values": [round(value, 4) for value in values],
        "points": points,
        "deltas": deltas,
        "rounded_deltas": [round(delta, 4) for delta in deltas],
        "exp4146_epoch_fix_unstalled": unstalled,
        "matches_published_087": matches,
        "final_val_exact_accuracy": final_value,
        "published_exact_accuracy_target": PUBLISHED_087_TARGET,
        "published_match_tolerance": PUBLISHED_MATCH_TOLERANCE,
        "final_to_target_gap": PUBLISHED_087_TARGET - final_value
        if final_value is not None
        else None,
        "prior_reference": {
            "path": prior["path"],
            "sha256": prior["sha256"],
            "fields_imported": ["baseline_val_trajectory.upstream_values"]
            if prior["path"]
            else [],
        },
        "attempted_passes": attempted_baseline_passes(clean_upstreams, skipped_ids),
    }


def metric_summary(payload: Mapping[str, Any] | None) -> JsonDict:
    """Return the metric fields used by the decisive graft gate."""

    return {
        "metric": str_metric(payload, "metric"),
        "delta": float_metric(payload, "delta"),
        "ci95": list_float_metric(payload, "ci95"),
        "status": str_metric(payload, "status"),
        "n_matched": int_metric(payload, "n_matched"),
    }


def metric_positive(summary: Mapping[str, Any]) -> bool:
    """Return whether a graft metric reports positive value with positive CI."""

    delta = summary.get("delta")
    return (
        isinstance(delta, int | float)
        and not isinstance(delta, bool)
        and delta > 0.0
        and ci_excludes_zero_positive(
            [float(value) for value in summary.get("ci95", [])]
            if isinstance(summary.get("ci95"), list)
            else []
        )
    )


def graft_verdict_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Answer whether Exp 4150 ran a decisive verifier graft or deferred."""

    artifact_status = (
        "skipped_flagged_adversarial"
        if was_skipped
        else "present"
        if isinstance(payload, Mapping)
        else "missing"
    )
    base = {
        "artifact_status": artifact_status,
        "honest_verdict": "" if was_skipped else verdict_text(payload),
        "graft_deferred": False,
        "verifier_value_added": False,
        "rerank_lift_vs_vote": {},
        "rft_vs_ablation_delta": {},
    }
    if was_skipped or not isinstance(payload, Mapping):
        base["status"] = artifact_status
        return base

    graft_deferred = bool_metric(payload, "graft_deferred") is True
    rerank = metric_summary(nested_map(payload, "rerank_lift_vs_vote"))
    rft = metric_summary(nested_map(payload, "rft_vs_ablation_delta"))
    value_added = bool_metric(payload, "verifier_value_added") is True or metric_positive(rerank) or metric_positive(rft)
    if graft_deferred:
        status = "deferred"
        value_added = False
    elif value_added:
        status = "ran_value_added"
    else:
        status = "ran_null"
    return {
        **base,
        "status": status,
        "graft_deferred": graft_deferred,
        "verifier_value_added": value_added,
        "rerank_lift_vs_vote": rerank,
        "rft_vs_ablation_delta": rft,
    }


def diffusiongemma_gate_status(graft: Mapping[str, Any]) -> str:
    """Resolve the queued DiffusionGemma gate only from a clean decisive graft."""

    if graft.get("status") == "ran_value_added":
        return "RESOLVED-positive"
    if graft.get("status") == "ran_null":
        return "RESOLVED-null"
    return "STILL-PENDING"


def prior_arc_reference(root: Path) -> JsonDict | None:
    """Read the prior clean ARC games total for an honest carry-forward."""

    path = root / PRIOR_ARC_REFERENCE_PATH
    if not path.exists():
        return None
    payload = read_json_object(path)
    total = int_metric(payload, "total_arc_games_solved")
    if total <= 0:
        return None
    return {
        "experiment_id": 4144,
        "path": relative_to_root(root, path),
        "sha256": sha256_file(path),
        "fields_imported": ["total_arc_games_solved"],
        "total_arc_games_solved": total,
        "total_arc_levels_solved": int_metric(payload, "total_arc_levels_solved"),
    }


def arc_games_answer(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
    root: Path,
) -> JsonDict:
    """Return clean .384 ARC count or a cited prior clean carry-forward."""

    if not was_skipped and isinstance(payload, Mapping):
        total = int_metric(payload, "total_games_solved")
        if total > 0:
            return {
                "status": "clean_384_imported",
                "total_arc_games_solved": total,
                "real_env_confirmed": bool_metric(payload, "real_env_confirmed") is True,
                "verifier_validated": bool_metric(payload, "verifier_validated") is True,
                "honest_verdict": verdict_text(payload),
                "carry_forward_provenance": None,
            }
    prior = prior_arc_reference(root)
    if prior is not None:
        return {
            "status": "prior_clean_carry_forward",
            "total_arc_games_solved": int(prior["total_arc_games_solved"]),
            "real_env_confirmed": False,
            "verifier_validated": False,
            "honest_verdict": "" if was_skipped else verdict_text(payload),
            "carry_forward_provenance": prior,
        }
    return {
        "status": "missing",
        "total_arc_games_solved": 0,
        "real_env_confirmed": False,
        "verifier_validated": False,
        "honest_verdict": "",
        "carry_forward_provenance": None,
    }


def sota_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Summarize clean Exp 4152 as context for the .385 queue only."""

    methods = (
        payload.get("methods_mapped") if isinstance(payload, Mapping) and not was_skipped else None
    )
    return {
        "included": isinstance(payload, Mapping) and not was_skipped,
        "methods_mapped": [dict(item) for item in methods if isinstance(item, Mapping)]
        if isinstance(methods, list)
        else [],
        "flagged_for_v385": str_metric(payload, "flagged_for_v385") if not was_skipped else "",
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def registry_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Summarize Exp 4153 only when it is not adversarial-flagged."""

    gate = nested_map(payload, "diffusiongemma_gate_state") if not was_skipped else {}
    return {
        "included": isinstance(payload, Mapping) and not was_skipped,
        "status": "skipped_flagged_adversarial"
        if was_skipped
        else "included"
        if isinstance(payload, Mapping)
        else "missing",
        "regression_guard_passed": bool_metric(payload, "regression_guard_passed") is True
        if not was_skipped
        else False,
        "diffusiongemma_gate_state": dict(gate),
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def hardware_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Carry clean Exp 4154 hardware continuity apart from science claims."""

    reachability = (
        payload.get("per_board_reachability")
        if isinstance(payload, Mapping) and not was_skipped
        else None
    )
    return {
        "included": isinstance(payload, Mapping) and not was_skipped,
        "kv260_terminal_confirmed": bool_metric(payload, "kv260_terminal_confirmed") is True
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


def headline_outcome(baseline: Mapping[str, Any], graft: Mapping[str, Any]) -> str:
    """Choose the single enumerated .384 headline."""

    if baseline.get("matches_published_087") is True:
        if graft.get("status") == "ran_value_added":
            return "baseline_converged_graft_validated"
        if graft.get("status") == "ran_null":
            return "baseline_converged_graft_null"
        return "baseline_converged_graft_deferred"
    if baseline.get("status") == "accumulation_unstalled_still_climbing_v385":
        return "accumulation_unstalled_still_climbing_v385"
    return "accumulation_still_blocked"


def headline_answers(
    baseline: Mapping[str, Any],
    graft: Mapping[str, Any],
    gate_status: str,
    arc_games: Mapping[str, Any],
) -> JsonDict:
    """Expose the five concrete operator questions in machine-checkable fields."""

    return {
        "exp4146_epoch_fix_unstalled": baseline.get("exp4146_epoch_fix_unstalled") is True,
        "exp4149_matches_published_087": baseline.get("matches_published_087"),
        "exp4150_decisive_graft_status": graft.get("status"),
        "diffusiongemma_gate_status": gate_status,
        "total_arc_games_solved": int(arc_games.get("total_arc_games_solved", 0)),
    }


def verdict(
    outcome: str,
    baseline: Mapping[str, Any],
    graft: Mapping[str, Any],
    gate_status: str,
    games_solved_total: int,
    skipped_count: int,
) -> str:
    """Build a terminal-prefixed verdict from the already-selected headline."""

    prefix = "success:" if outcome == "baseline_converged_graft_validated" else "blocked:" if outcome == "accumulation_still_blocked" else "complete:"
    epochfix_flag = int(baseline.get("exp4146_epoch_fix_unstalled") is True)
    baseline_flag = int(baseline.get("matches_published_087") is True)
    graft_status = str(graft.get("status", "missing")) or "missing"
    return (
        f"{prefix} capstone_v384_{outcome}_epochfix{epochfix_flag}_"
        f"baseline087{baseline_flag}_graft_{graft_status}_diffusiongemma_{gate_status}_"
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
    if 4146 in clean_ids:
        fields[4146] = ["seed_epoch", "post_epoch", "val_exact_accuracy", "max_epochs_cap_confirmed"]
    for experiment_id in (4147, 4148):
        if experiment_id in clean_ids:
            fields[experiment_id] = ["post_epoch", "val_exact_accuracy", "native_trainer_launched"]
    if 4149 in clean_ids:
        fields[4149] = [
            "val_trajectory_v384",
            "val_exact_accuracy",
            "matches_published_087",
            "native_trainer_launched",
        ]
    if 4150 in clean_ids:
        fields[4150] = [
            "graft_deferred",
            "verifier_value_added",
            "rerank_lift_vs_vote",
            "rft_vs_ablation_delta",
        ]
    if 4151 in clean_ids:
        fields[4151] = ["total_games_solved", "real_env_confirmed", "verifier_validated"]
    if 4152 in clean_ids:
        fields[4152] = ["methods_mapped", "flagged_for_v385"]
    if 4153 in clean_ids:
        fields[4153] = ["regression_guard_passed", "diffusiongemma_gate_state"]
    if 4154 in clean_ids:
        fields[4154] = [
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
    """Cite every existing .384 upstream and list imported fields."""

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
    """Build the .384 capstone from landed upstream artifacts."""

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

    baseline = baseline_val_trajectory(clean_upstreams, root=root_path, skipped_ids=skipped_ids)
    graft = graft_verdict_answer(clean_upstreams.get(4150), was_skipped=4150 in skipped_ids)
    gate_status = diffusiongemma_gate_status(graft)
    arc_games = arc_games_answer(
        clean_upstreams.get(4151),
        was_skipped=4151 in skipped_ids,
        root=root_path,
    )
    sota = sota_answer(clean_upstreams.get(4152), was_skipped=4152 in skipped_ids)
    registry = registry_answer(clean_upstreams.get(4153), was_skipped=4153 in skipped_ids)
    hardware = hardware_answer(clean_upstreams.get(4154), was_skipped=4154 in skipped_ids)
    outcome = headline_outcome(baseline, graft)
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_ids)
    total_games = int(arc_games["total_arc_games_solved"])
    fields_by_id = imported_fields_by_id(clean_ids)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v384_4155.v1",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": verdict(outcome, baseline, graft, gate_status, total_games, len(skipped)),
        "headline_outcome": outcome,
        "headline_answers": headline_answers(baseline, graft, gate_status, arc_games),
        "baseline_val_trajectory": baseline,
        "graft_verdict": graft,
        "diffusiongemma_gate_status": gate_status,
        "diffusiongemma_gate": {
            "status": gate_status,
            "basis": "clean_exp4150_verifier_value_added",
            "verifier_value_added": graft.get("status") == "ran_value_added",
            "resolved": gate_status != "STILL-PENDING",
        },
        "arc_games": arc_games,
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
    """Validate the fields that keep the .384 headline auditable."""

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
    if artifact.get("diffusiongemma_gate_status") not in DIFFUSIONGEMMA_GATE_STATUSES:
        raise ValueError("diffusiongemma_gate_status must be enumerated")
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
    """Build, validate, and write the Exp 4155 capstone artifact."""

    root_path = Path(root)
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    validate_artifact(artifact)
    output = root_path / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    """Write the default Exp 4155 capstone artifact and print its path."""

    output = write_artifact(REPO_ROOT)
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
