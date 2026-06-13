"""Build the Exp 4144 v383 capstone aggregation.

Spec refs: REQ-CAPSTONE-4144, SCENARIO-CAPSTONE-4144.

This module reads the landed .383 artifacts, excludes any artifact stamped
``flagged_adversarial: true`` before importing metrics, and writes the compact
decision-grade verdict for the verifier-as-reward milestone. The executable
Sudoku oracle is carried only as context; it never defines verifier value
added.
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
OUTPUT_REL_PATH = Path("results/experiment_4144_capstone_v383.json")
EXPERIMENT_ID = 4144
RANDOM_SEED = 4144
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
PUBLISHED_087_TARGET = 0.87
PUBLISHED_MATCH_TOLERANCE = 0.02

PRIOR_BASELINE_REFERENCE_PATH = Path("results/experiment_4133_capstone_v382.json")
UPSTREAM_IDS = (4135, 4136, 4137, 4138, 4139, 4140, 4141, 4142, 4143)
BASELINE_PASS_IDS = (4135, 4136, 4137, 4138)
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    4135: Path("results/experiment_4135_sudoku_accumulate_pass1_fixed_lr.json"),
    4136: Path("results/experiment_4136_sudoku_accumulate_pass2_fixed_lr.json"),
    4137: Path("results/experiment_4137_sudoku_accumulate_pass3_fixed_lr.json"),
    4138: Path("results/experiment_4138_sudoku_accumulate_pass4_convergence_check.json"),
    4139: Path("results/experiment_4139_decisive_verifier_graft_sudoku.json"),
    4140: Path("results/experiment_4140_arc_incremental_progress.json"),
    4141: Path("results/experiment_4141_sota_ingestion_recursive_reasoner_verifier.json"),
    4142: Path("results/experiment_4142_verifier_registry_gaps_hygiene.json"),
    4143: Path("results/experiment_4143_hardware_continuity.json"),
}

HEADLINE_OUTCOMES = {
    "baseline_converged_verifier_value_added",
    "baseline_converged_verifier_null_honest",
    "baseline_near_faithful_rft_measured",
    "baseline_accumulating_graft_deferred_v384_continues",
    "baseline_config_blocked",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "baseline_val_trajectory",
    "verifier_value_added_verdict",
    "diffusiongemma_unlocks",
    "upstream_provenance",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'baseline converging, graft deferred, .384 "
        "continues' OR 'verifier null with headroom present' is COMPLETE and valuable."
    ),
    "headline_outcome": (
        "One of the enumerated set -- forces a single unambiguous read of the .383 result."
    ),
    "baseline_val_trajectory": (
        "Val across .382+.383 (0.278 -> ... across 4 passes); shows whether the fixed "
        "schedule converged."
    ),
    "verifier_value_added_verdict": (
        "The decisive answer (true / false-with-headroom / deferred), defined on the "
        "transferable ensemble + RFT (NOT the oracle); the DiffusionGemma gate signal."
    ),
    "diffusiongemma_unlocks": (
        "Bare bool: verifier_value_added==true (transferable ensemble + RFT) on the "
        "executable Sudoku domain -> the queued scale-up unlocks for .384."
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
    """Resolve the .383 upstream artifact paths under the selected root."""

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


def bool_metric(payload: Mapping[str, Any] | None, field: str) -> bool | None:
    """Read a JSON boolean without treating integers or strings as booleans."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, bool) else None


def int_metric(payload: Mapping[str, Any] | None, field: str) -> int:
    """Read an integer counter while rejecting bool, because bool is an int subclass."""

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


def clean_trajectory_points(rows: object) -> list[JsonDict]:
    """Keep only val-trajectory rows that contain real numeric validation values."""

    if not isinstance(rows, list):
        return []
    points: list[JsonDict] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        value = float_metric(row, "val_exact_accuracy")
        if value is None:
            continue
        pass_index = int_metric(row, "pass_index")
        points.append(
            {
                "experiment_id": experiment_id_from_name(str_metric(row, "experiment")),
                "label": str_metric(row, "label"),
                "pass_index": pass_index,
                "status": str_metric(row, "status"),
                "source_field": "val_trajectory_383[].val_exact_accuracy",
                "val_exact_accuracy": value,
                "val_exact_accuracy_rounded": round(value, 4),
                "delta_vs_previous": float_metric(row, "delta_vs_previous"),
            }
        )
    return points


def experiment_id_from_name(name: str) -> int:
    """Extract an experiment id from the conventional artifact name."""

    marker = "experiment_"
    if not name.startswith(marker):
        return 0
    suffix = name[len(marker) :].split("_", 1)[0]
    return int(suffix) if suffix.isdigit() else 0


def prior_baseline_reference(root: Path) -> JsonDict | None:
    """Include the clean .382 anchor when the prior capstone is available."""

    path = root / PRIOR_BASELINE_REFERENCE_PATH
    if not path.exists():
        return None
    payload = read_json_object(path)
    trajectory = nested_map(payload, "baseline_val_trajectory")
    value = float_metric(trajectory, "final_val_exact_accuracy")
    if value is None:
        return None
    return {
        "experiment_id": 4133,
        "path": relative_to_root(root, path),
        "sha256": sha256_file(path),
        "label": ".382_capstone_anchor",
        "pass_index": 0,
        "source_field": "baseline_val_trajectory.final_val_exact_accuracy",
        "val_exact_accuracy": value,
        "val_exact_accuracy_rounded": round(value, 4),
        "delta_vs_previous": None,
    }


def attempted_baseline_passes(
    clean_upstreams: Mapping[int, Mapping[str, Any]],
    skipped_ids: set[int],
) -> list[JsonDict]:
    """Expose which .383 baseline passes were usable without importing skipped metrics."""

    rows: list[JsonDict] = []
    for experiment_id in BASELINE_PASS_IDS:
        payload = clean_upstreams.get(experiment_id)
        if experiment_id in skipped_ids:
            included = False
            status = "skipped_flagged_adversarial"
        elif isinstance(payload, Mapping):
            included = True
            status = str_metric(payload, "baseline_status") or "included"
        else:
            included = False
            status = "missing"
        rows.append(
            {
                "experiment_id": experiment_id,
                "pass_index": experiment_id - 4134,
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
    """Build the .382 + .383 validation trajectory and convergence answer."""

    exp4138 = clean_upstreams.get(4138)
    prior = prior_baseline_reference(root)
    points: list[JsonDict] = [prior] if prior is not None else []
    rows = clean_trajectory_points(
        exp4138.get("val_trajectory_383") if isinstance(exp4138, Mapping) else None
    )
    if prior is not None:
        rows = [row for row in rows if row.get("pass_index") != 0]
    points.extend(rows)
    values = [float(point["val_exact_accuracy"]) for point in points]
    deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
    matches = bool_metric(exp4138, "matches_published_087") if isinstance(exp4138, Mapping) else None
    near = bool_metric(exp4138, "near_faithful_080") if isinstance(exp4138, Mapping) else None
    final_value = float_metric(exp4138, "val_exact_accuracy") if isinstance(exp4138, Mapping) else None
    if final_value is None and values:
        final_value = values[-1]
    moved_toward_target = len(values) >= 2 and values[-1] > values[0]
    baseline_skipped = any(experiment_id in skipped_ids for experiment_id in BASELINE_PASS_IDS)

    if matches is True:
        status = "faithful_baseline_reproduced"
    elif near is True:
        status = "near_faithful_baseline"
    elif baseline_skipped:
        status = "baseline_config_blocked"
    elif isinstance(exp4138, Mapping) and moved_toward_target:
        status = "still_accumulating"
    elif isinstance(exp4138, Mapping):
        status = "blocked_or_not_accelerating"
    else:
        status = "missing"

    return {
        "status": status,
        "values": values,
        "rounded_values": [round(value, 4) for value in values],
        "points": points,
        "deltas": deltas,
        "rounded_deltas": [round(delta, 4) for delta in deltas],
        "matches_published_087": matches,
        "near_faithful_080": near,
        "converged_toward_087": (matches is True or near is True) and moved_toward_target,
        "final_val_exact_accuracy": final_value,
        "published_exact_accuracy_target": PUBLISHED_087_TARGET,
        "published_match_tolerance": PUBLISHED_MATCH_TOLERANCE,
        "final_to_target_gap": PUBLISHED_087_TARGET - final_value
        if final_value is not None
        else None,
        "attempted_passes": attempted_baseline_passes(clean_upstreams, skipped_ids),
    }


def metric_summary(payload: Mapping[str, Any] | None) -> JsonDict:
    """Return the transferable metric fields used by the verifier gate."""

    return {
        "metric": str_metric(payload, "metric"),
        "delta": float_metric(payload, "delta"),
        "ci95": list_float_metric(payload, "ci95"),
        "status": str_metric(payload, "status"),
        "meaningful": bool_metric(payload, "meaningful"),
        "n_matched": int_metric(payload, "n_matched"),
        "uses_exact_validity_check": bool_metric(payload, "uses_exact_validity_check"),
    }


def verifier_value_added_answer(
    payload: Mapping[str, Any] | None,
    *,
    baseline: Mapping[str, Any],
    was_skipped: bool,
) -> JsonDict:
    """Decide verifier value added from transferable signals, never from the oracle."""

    baseline_usable = (
        baseline.get("matches_published_087") is True or baseline.get("near_faithful_080") is True
    )
    artifact_status = (
        "skipped_flagged_adversarial"
        if was_skipped
        else "present"
        if isinstance(payload, Mapping)
        else "missing"
    )
    if not baseline_usable:
        return {
            "status": "deferred",
            "reason": "baseline_not_faithful_or_near_faithful",
            "artifact_status": artifact_status,
            "headroom_present": None,
            "verifier_value_added": False,
            "transferable_ensemble_value_added": False,
            "rft_label_deconfound_value_added": False,
            "ensemble_rerank_lift_vs_vote": {},
            "rft_vs_ablation_delta": {},
            "oracle_context": {},
            "uses_executable_oracle_upper_bound_for_gate": False,
            "honest_verdict": "",
        }
    if was_skipped or not isinstance(payload, Mapping):
        return {
            "status": "deferred",
            "reason": artifact_status,
            "artifact_status": artifact_status,
            "headroom_present": None,
            "verifier_value_added": False,
            "transferable_ensemble_value_added": False,
            "rft_label_deconfound_value_added": False,
            "ensemble_rerank_lift_vs_vote": {},
            "rft_vs_ablation_delta": {},
            "oracle_context": {},
            "uses_executable_oracle_upper_bound_for_gate": False,
            "honest_verdict": "",
        }

    headroom = bool_metric(payload, "headroom_present")
    ensemble = metric_summary(nested_map(payload, "ensemble_rerank_lift_vs_vote"))
    rft = metric_summary(nested_map(payload, "rft_vs_ablation_delta"))
    ensemble_positive = (
        headroom is True
        and ensemble["uses_exact_validity_check"] is not True
        and (ensemble["meaningful"] is True or ci_excludes_zero_positive(ensemble["ci95"]))
        and (ensemble["delta"] or 0.0) > 0.0
    )
    rft_positive = headroom is True and ci_excludes_zero_positive(rft["ci95"])
    value_added = ensemble_positive or rft_positive
    if value_added:
        status = "true"
        reason = "transferable_ensemble_or_rft_positive"
    elif headroom is True:
        status = "false-with-headroom"
        reason = "headroom_present_no_transferable_lift"
    else:
        status = "deferred"
        reason = "no_headroom_false_negative_risk"

    oracle = dict(nested_map(payload, "executable_oracle_upper_bound"))
    if oracle:
        oracle["used_for_gate"] = False
    return {
        "status": status,
        "reason": reason,
        "artifact_status": artifact_status,
        "headroom_present": headroom,
        "verifier_value_added": value_added,
        "transferable_ensemble_value_added": ensemble_positive,
        "rft_label_deconfound_value_added": rft_positive,
        "ensemble_rerank_lift_vs_vote": ensemble,
        "rft_vs_ablation_delta": rft,
        "oracle_context": oracle,
        "uses_executable_oracle_upper_bound_for_gate": False,
        "honest_verdict": verdict_text(payload),
    }


def arc_levels_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Carry the clean Exp 4140 ARC level count without inferring a solve."""

    if was_skipped:
        status = "skipped_flagged_adversarial"
    elif not isinstance(payload, Mapping):
        status = "missing"
    elif int_metric(payload, "new_levels_solved_this_task") > 0 and (
        bool_metric(payload, "real_env_confirmed") is True
    ):
        status = "new_level_solved"
    else:
        status = "measured_no_new_level"
    return {
        "status": status,
        "prior_total_levels_solved": int_metric(payload, "prior_total_levels_solved")
        if isinstance(payload, Mapping) and not was_skipped
        else 0,
        "total_levels_solved": int_metric(payload, "total_levels_solved")
        if isinstance(payload, Mapping) and not was_skipped
        else 0,
        "total_games_solved": int_metric(payload, "total_games_solved")
        if isinstance(payload, Mapping) and not was_skipped
        else 0,
        "new_levels_solved_this_task": int_metric(payload, "new_levels_solved_this_task")
        if isinstance(payload, Mapping) and not was_skipped
        else 0,
        "real_env_confirmed": bool_metric(payload, "real_env_confirmed") is True
        if not was_skipped
        else False,
        "verifier_validated": bool_metric(payload, "verifier_validated") is True
        if not was_skipped
        else False,
        "target_game": str_metric(payload, "target_game") if not was_skipped else "",
        "target_level": int_metric(payload, "target_level") if not was_skipped else 0,
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def sota_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Summarize clean Exp 4141 SOTA ingestion as supporting context only."""

    methods = (
        payload.get("methods_mapped") if isinstance(payload, Mapping) and not was_skipped else None
    )
    return {
        "included": isinstance(payload, Mapping) and not was_skipped,
        "methods_mapped": [dict(item) for item in methods if isinstance(item, Mapping)]
        if isinstance(methods, list)
        else [],
        "flagged_for_v384": str_metric(payload, "flagged_for_v384") if not was_skipped else "",
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def registry_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Summarize clean Exp 4142 registry hygiene when it is not flagged."""

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
        "sudoku_baseline_status": str_metric(nested_map(payload, "sudoku_baseline"), "status")
        if not was_skipped
        else "",
        "sudoku_decisive_graft_status": str_metric(
            nested_map(payload, "sudoku_decisive_graft"), "status"
        )
        if not was_skipped
        else "",
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def hardware_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Carry clean Exp 4143 hardware continuity apart from science claims."""

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


def headline_outcome(
    baseline: Mapping[str, Any],
    verifier: Mapping[str, Any],
) -> str:
    """Choose the single enumerated .383 headline."""

    if baseline.get("status") == "baseline_config_blocked":
        return "baseline_config_blocked"
    if baseline.get("matches_published_087") is True:
        if verifier.get("status") == "true":
            return "baseline_converged_verifier_value_added"
        return "baseline_converged_verifier_null_honest"
    if baseline.get("near_faithful_080") is True and verifier.get("status") in {
        "true",
        "false-with-headroom",
    }:
        return "baseline_near_faithful_rft_measured"
    if baseline.get("status") in {"still_accumulating", "near_faithful_baseline"}:
        return "baseline_accumulating_graft_deferred_v384_continues"
    return "baseline_config_blocked"


def headline_answers(
    baseline: Mapping[str, Any],
    verifier: Mapping[str, Any],
    arc_levels: Mapping[str, Any],
    diffusiongemma_unlocks: bool,
) -> JsonDict:
    """Expose the concrete operator questions in machine-checkable fields."""

    return {
        "exp4138_fixed_lr_baseline_converged": baseline.get("converged_toward_087") is True,
        "exp4138_matches_published_087": baseline.get("matches_published_087"),
        "exp4138_near_faithful_080": baseline.get("near_faithful_080"),
        "exp4139_headroom_present": verifier.get("headroom_present"),
        "exp4139_transferable_verifier_value_added": verifier.get("verifier_value_added") is True,
        "total_arc_levels_solved": int(arc_levels.get("total_levels_solved", 0)),
        "diffusiongemma_unlocks": diffusiongemma_unlocks,
    }


def diffusiongemma_gate_state(verifier: Mapping[str, Any]) -> JsonDict:
    """Represent the scale-up gate solely from transferable verifier value added."""

    unlocks = verifier.get("verifier_value_added") is True
    return {
        "state": "unlocked" if unlocks else "kept_gated",
        "verifier_value_added": unlocks,
        "basis": "transferable_ensemble_rerank_plus_rft_label_deconfound_not_oracle",
        "uses_executable_oracle_upper_bound": False,
        "reason": "transferable_verifier_value_added" if unlocks else str(verifier.get("reason", "")),
    }


def verdict(
    outcome: str,
    baseline: Mapping[str, Any],
    verifier: Mapping[str, Any],
    levels_solved_total: int,
    skipped_count: int,
    diffusiongemma_unlocks: bool,
) -> str:
    """Build a terminal-prefixed verdict from the already-selected headline."""

    prefix = (
        "success:"
        if outcome == "baseline_converged_verifier_value_added"
        else "blocked:"
        if outcome == "baseline_config_blocked"
        else "complete:"
    )
    baseline_flag = int(baseline.get("matches_published_087") is True)
    near_flag = int(baseline.get("near_faithful_080") is True)
    headroom_flag = int(verifier.get("headroom_present") is True)
    verifier_status = str(verifier.get("status", "deferred")) or "deferred"
    diffusion_flag = int(diffusiongemma_unlocks)
    return (
        f"{prefix} capstone_v383_{outcome}_baseline_converged{baseline_flag}_"
        f"near_faithful{near_flag}_headroom{headroom_flag}_verifier_{verifier_status}_"
        f"diffusiongemma{diffusion_flag}_levels{levels_solved_total}_"
        f"flagged_skipped{skipped_count}"
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
    for experiment_id in (4135, 4136, 4137):
        if experiment_id in clean_ids:
            fields[experiment_id] = [
                "pass_index",
                "val_exact_accuracy",
                "delta_vs_previous",
                "matches_published_087",
            ]
    if 4138 in clean_ids:
        fields[4138] = [
            "val_trajectory_383",
            "matches_published_087",
            "near_faithful_080",
            "val_exact_accuracy",
            "baseline_status",
        ]
    if 4139 in clean_ids:
        fields[4139] = [
            "headroom_present",
            "ensemble_rerank_lift_vs_vote.delta",
            "ensemble_rerank_lift_vs_vote.ci95",
            "ensemble_rerank_lift_vs_vote.uses_exact_validity_check",
            "rft_vs_ablation_delta.delta",
            "rft_vs_ablation_delta.ci95",
            "executable_oracle_upper_bound.context_only",
        ]
    if 4140 in clean_ids:
        fields[4140] = [
            "prior_total_levels_solved",
            "total_levels_solved",
            "total_games_solved",
            "new_levels_solved_this_task",
            "real_env_confirmed",
            "verifier_validated",
            "target_game",
            "target_level",
        ]
    if 4141 in clean_ids:
        fields[4141] = ["methods_mapped", "flagged_for_v384"]
    if 4142 in clean_ids:
        fields[4142] = [
            "regression_guard_passed",
            "diffusiongemma_gate_state",
            "sudoku_baseline.status",
            "sudoku_decisive_graft.status",
        ]
    if 4143 in clean_ids:
        fields[4143] = [
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
    """Cite every existing .383 upstream and list imported fields."""

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
    """Build the .383 capstone from landed upstream artifacts."""

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
    verifier = verifier_value_added_answer(
        clean_upstreams.get(4139),
        baseline=baseline,
        was_skipped=4139 in skipped_ids,
    )
    arc_levels = arc_levels_answer(clean_upstreams.get(4140), was_skipped=4140 in skipped_ids)
    sota = sota_answer(clean_upstreams.get(4141), was_skipped=4141 in skipped_ids)
    registry = registry_answer(clean_upstreams.get(4142), was_skipped=4142 in skipped_ids)
    hardware = hardware_answer(clean_upstreams.get(4143), was_skipped=4143 in skipped_ids)
    gate = diffusiongemma_gate_state(verifier)
    unlocks = gate["state"] == "unlocked"
    outcome = headline_outcome(baseline, verifier)
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_ids)
    total_levels = int(arc_levels["total_levels_solved"])
    fields_by_id = imported_fields_by_id(clean_ids)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v383_4144.v1",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": verdict(outcome, baseline, verifier, total_levels, len(skipped), unlocks),
        "headline_outcome": outcome,
        "headline_answers": headline_answers(baseline, verifier, arc_levels, unlocks),
        "baseline_val_trajectory": baseline,
        "verifier_value_added_verdict": verifier,
        "diffusiongemma_unlocks": unlocks,
        "diffusiongemma_gate_state": gate,
        "arc_levels": arc_levels,
        "total_arc_levels_solved": total_levels,
        "total_arc_games_solved": int(arc_levels["total_games_solved"]),
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
    """Validate the fields that keep the .383 headline auditable."""

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
    verifier = artifact.get("verifier_value_added_verdict")
    if not isinstance(verifier, Mapping) or verifier.get("status") not in {
        "true",
        "false-with-headroom",
        "deferred",
    }:
        raise ValueError("verifier_value_added_verdict must be decisive")  # pragma: no cover
    if verifier.get("uses_executable_oracle_upper_bound_for_gate") is not False:
        raise ValueError("executable oracle cannot define verifier value added")  # pragma: no cover
    if not isinstance(artifact.get("diffusiongemma_unlocks"), bool):
        raise ValueError("diffusiongemma_unlocks must be a bare bool")
    if artifact.get("diffusiongemma_unlocks") != (verifier.get("verifier_value_added") is True):
        raise ValueError("diffusiongemma_unlocks must equal transferable verifier value added")
    if not isinstance(artifact.get("total_arc_levels_solved"), int) or isinstance(
        artifact.get("total_arc_levels_solved"), bool
    ):
        raise ValueError("total_arc_levels_solved must be a bare int")  # pragma: no cover
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
    """Build, validate, and write the Exp 4144 capstone artifact."""

    root_path = Path(root)
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    validate_artifact(artifact)
    output = root_path / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    """Write the default Exp 4144 capstone artifact and print its path."""

    output = write_artifact(REPO_ROOT)
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
