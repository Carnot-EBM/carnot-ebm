"""Build the Exp 4165 v385 capstone aggregation.

Spec refs: REQ-CAPSTONE-4165, SCENARIO-CAPSTONE-4165.
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
OUTPUT_REL_PATH = Path("results/experiment_4165_capstone_v385.json")
EXPERIMENT_ID = 4165
RANDOM_SEED = 4165
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4165", "SCENARIO-CAPSTONE-4165"]

SEED_VAL = 0.278172343969
LINEAGE_VALUES = [0.105989582837, SEED_VAL, 0.42]
FAITHFUL_THRESHOLD = 0.85

UPSTREAM_IDS = (4157, 4158, 4159, 4160, 4161, 4162, 4163, 4164)
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    4157: Path("results/experiment_4157_baseline_harvest_contiguous_continue.json"),
    4158: Path("results/experiment_4158_verifier_rerank_recovery_moat.json"),
    4159: Path("results/experiment_4159_decisive_verifier_reward_graft.json"),
    4160: Path("results/experiment_4160_arc_action_efficiency_harness.json"),
    4161: Path("results/experiment_4161_observability_timing_detector_fix.json"),
    4162: Path("results/experiment_4162_sota_ingestion_verifier_moat_guidance.json"),
    4163: Path("results/experiment_4163_verifier_registry_gaps_hygiene.json"),
    4164: Path("results/experiment_4164_hardware_continuity.json"),
}

HEADLINE_OUTCOMES = {
    "baseline_advancing_moat_rerank_confirmed",
    "baseline_advancing_moat_rerank_null",
    "baseline_advancing_moat_rerank_no_headroom",
    "baseline_faithful_graft_validated",
    "baseline_faithful_graft_null",
    "accumulation_still_blocked",
}
MOAT_STATUSES = {
    "confirmed",
    "null",
    "no_headroom",
    "skipped_flagged_adversarial",
    "missing",
}
GRAFT_STATUSES = {
    "deferred",
    "ran_value_added",
    "ran_null",
    "skipped_flagged_adversarial",
    "missing",
}
DIFFUSIONGEMMA_GATE_STATUSES = {"RESOLVED-positive", "RESOLVED-null", "STILL-PENDING"}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "baseline_val_trajectory",
    "moat_rerank_verdict",
    "diffusiongemma_gate_status",
    "upstream_provenance",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'baseline advancing via contiguous run; moat "
        "confirmed/null at the rerank level; .386 continues to 0.85 then grafts' is "
        "COMPLETE and valuable."
    ),
    "headline_outcome": (
        "One of the enumerated set -- forces a single unambiguous read of the .385 "
        "result, and (unlike .382-.384) is NOT all-or-nothing on the faithful baseline."
    ),
    "baseline_val_trajectory": (
        "Val across .382-.385 (0.106 -> 0.278 -> 0.42+ -> ?); shows the contiguous "
        "run restored convergence after the bounded-pass chain failed."
    ),
    "moat_rerank_verdict": (
        "The decision-grade rerank-recovery result (CI95 excl 0 / null / no-headroom) "
        "-- the .385 moat signal that does NOT depend on reaching 0.85."
    ),
    "diffusiongemma_gate_status": (
        "RESOLVED-positive / RESOLVED-null / STILL-PENDING -- the explicit status "
        "after the rerank + graft work."
    ),
    "upstream_provenance": (
        "{experiment_id, fields_imported, sha256} per cited upstream -- the audit trail."
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


def relative_to_root(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:  # pragma: no cover - only used with external absolute paths.
        return str(path)


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
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def str_metric(payload: Mapping[str, Any] | None, field: str) -> str:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def nested_map(payload: Mapping[str, Any] | None, field: str) -> Mapping[str, Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def list_float_metric(payload: Mapping[str, Any] | None, field: str) -> list[float]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    if not isinstance(value, list):
        return []
    return [
        float(item)
        for item in value
        if isinstance(item, int | float) and not isinstance(item, bool)
    ]


def ci_excludes_zero_positive(ci95: list[float]) -> bool:
    return len(ci95) == 2 and ci95[0] > 0.0 and ci95[1] > 0.0


def ci_includes_zero(ci95: list[float]) -> bool:
    return len(ci95) == 2 and ci95[0] <= 0.0 <= ci95[1]


def metric_summary(payload: Mapping[str, Any] | None) -> JsonDict:
    return {
        "delta": float_metric(payload, "delta"),
        "ci95": list_float_metric(payload, "ci95"),
        "status": str_metric(payload, "status"),
        "n_puzzles": int_metric(payload, "n_puzzles"),
        "n_matched": int_metric(payload, "n_matched"),
        "verifier_pass_at_1": float_metric(payload, "verifier_pass_at_1"),
    }


def metric_positive(metric: Mapping[str, Any]) -> bool:
    delta = metric.get("delta")
    return (
        isinstance(delta, int | float)
        and not isinstance(delta, bool)
        and delta > 0.0
        and ci_excludes_zero_positive(metric.get("ci95", []))
    )


def trajectory_current_marker(payload: Mapping[str, Any] | None) -> float | None:
    rows = payload.get("val_trajectory") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        value = float_metric(row, "val_exact_accuracy")
        if value is not None and value >= 0.42:
            return value
    return None


def baseline_val_trajectory(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    values = list(LINEAGE_VALUES)
    if was_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "source_experiment_id": 4157,
            "source_status": "skipped_flagged_adversarial",
            "values": values,
            "rounded_values": [round(value, 3) for value in values],
            "seed_val": SEED_VAL,
            "current_val": None,
            "max_val": None,
            "run_alive": None,
            "advanced_vs_seed": None,
            "baseline_faithful": False,
            "honest_verdict": verdict_text(payload),
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "missing",
            "source_experiment_id": 4157,
            "source_status": "missing",
            "values": values,
            "rounded_values": [round(value, 3) for value in values],
            "seed_val": SEED_VAL,
            "current_val": None,
            "max_val": None,
            "run_alive": None,
            "advanced_vs_seed": None,
            "baseline_faithful": False,
            "honest_verdict": "",
        }

    current = float_metric(payload, "current_val")
    max_val = float_metric(payload, "max_val")
    marker = trajectory_current_marker(payload)
    if marker is not None and abs(marker - LINEAGE_VALUES[-1]) > 1e-6:
        values.append(marker)
    if current is not None and all(abs(current - value) > 1e-9 for value in values):
        values.append(current)
    advanced = current is not None and current > SEED_VAL
    faithful = bool_metric(payload, "baseline_faithful") is True or (
        current is not None and current >= FAITHFUL_THRESHOLD
    )
    return {
        "status": "baseline_advancing" if advanced else "accumulation_still_blocked",
        "source_experiment_id": 4157,
        "source_status": "included",
        "values": values,
        "rounded_values": [round(value, 3) for value in values],
        "seed_val": SEED_VAL,
        "current_val": current,
        "max_val": max_val,
        "run_alive": bool_metric(payload, "run_alive"),
        "advanced_vs_seed": advanced,
        "baseline_faithful": faithful,
        "manual_lr_step": int_metric(payload, "manual_lr_step"),
        "estimated_passes_to_085": nested_map(payload, "estimated_passes_to_085"),
        "honest_verdict": verdict_text(payload),
    }


def moat_rerank_verdict(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    base = {
        "source_experiment_id": 4158,
        "honest_verdict": "" if was_skipped else verdict_text(payload),
        "headroom_present": None,
        "oracle_at_k": None,
        "vote_at_1": None,
        "delta": None,
        "ci95": [],
        "n_puzzles": 0,
        "verifier_recovers_outvoted": 0,
        "decision": "uninformative",
    }
    if was_skipped:
        return {**base, "status": "skipped_flagged_adversarial"}
    if not isinstance(payload, Mapping):
        return {**base, "status": "missing"}

    metric = metric_summary(nested_map(payload, "rerank_lift_vs_vote"))
    headroom = bool_metric(payload, "headroom_present") is True
    ci95 = metric["ci95"]
    if not headroom:
        status = "no_headroom"
        decision = "no_headroom"
    elif metric_positive(metric):
        status = "confirmed"
        decision = "moat_real_at_rerank"
    elif ci_includes_zero(ci95):
        status = "null"
        decision = "moat_absent_at_rerank"
    else:
        status = "null"
        decision = "moat_absent_at_rerank"
    return {
        **base,
        "status": status,
        "decision": decision,
        "headroom_present": headroom,
        "oracle_at_k": float_metric(payload, "oracle_at_k"),
        "vote_at_1": float_metric(payload, "vote_at_1"),
        "delta": metric["delta"],
        "ci95": ci95,
        "n_puzzles": metric["n_puzzles"],
        "verifier_pass_at_1": metric["verifier_pass_at_1"],
        "verifier_recovers_outvoted": int_metric(payload, "verifier_recovers_outvoted"),
        "rerank_lift_vs_vote": metric,
        "cost_ratio_vs_llm_judge": nested_map(payload, "cost_ratio_vs_llm_judge"),
    }


def graft_verdict(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    base = {
        "source_experiment_id": 4159,
        "honest_verdict": "" if was_skipped else verdict_text(payload),
        "graft_deferred": False,
        "verifier_value_added": False,
        "rft_vs_ablation_delta": {},
    }
    if was_skipped:
        return {**base, "status": "skipped_flagged_adversarial"}
    if not isinstance(payload, Mapping):
        return {**base, "status": "missing"}

    rft = metric_summary(nested_map(payload, "rft_vs_ablation_delta"))
    deferred = bool_metric(payload, "graft_deferred") is True
    value_added = bool_metric(payload, "verifier_value_added") is True or metric_positive(rft)
    if deferred:
        status = "deferred"
        value_added = False
    elif value_added:
        status = "ran_value_added"
    else:
        status = "ran_null"
    return {
        **base,
        "status": status,
        "graft_deferred": deferred,
        "verifier_value_added": value_added,
        "current_val": float_metric(payload, "current_val"),
        "rft_vs_ablation_delta": rft,
    }


def diffusiongemma_gate_status(moat: Mapping[str, Any], graft: Mapping[str, Any]) -> str:
    if graft.get("status") == "ran_value_added":
        return "RESOLVED-positive"
    if graft.get("status") == "ran_null":
        return "RESOLVED-null"
    if moat.get("status") == "confirmed":
        return "RESOLVED-positive"
    if moat.get("status") == "null":
        return "RESOLVED-null"
    return "STILL-PENDING"


def arc_action_efficiency(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    if was_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "action_efficiency_ratio": 0.0,
            "total_games_solved": 0,
            "fields_imported": [],
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "missing",
            "action_efficiency_ratio": 0.0,
            "total_games_solved": 0,
            "fields_imported": [],
        }
    return {
        "status": "included",
        "action_efficiency_ratio": float_metric(payload, "action_efficiency_ratio") or 0.0,
        "baseline_actions": int_metric(payload, "baseline_actions"),
        "verifier_actions": int_metric(payload, "verifier_actions"),
        "actions_saved_vs_baseline": int_metric(payload, "actions_saved_vs_baseline"),
        "total_games_solved": int_metric(payload, "total_games_solved"),
        "new_levels_solved_this_task": int_metric(payload, "new_levels_solved_this_task"),
        "real_env_confirmed": bool_metric(payload, "real_env_confirmed") is True,
        "honest_verdict": verdict_text(payload),
        "fields_imported": imported_fields_by_id({4160})[4160],
    }


def observability_status(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "fixed": False}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "fixed": False}
    fixed = bool_metric(payload, "fix_applied") is True
    return {
        "status": "fixed" if fixed else "not_fixed",
        "fixed": fixed,
        "fallback_added": bool_metric(payload, "fallback_added") is True,
        "research_conductor_touched": bool_metric(payload, "research_conductor_touched") is True,
        "honest_verdict": verdict_text(payload),
    }


def sota_guidance(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "flagged_for_v386": ""}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "flagged_for_v386": ""}
    methods = payload.get("methods_mapped")
    return {
        "status": "included",
        "flagged_for_v386": str_metric(payload, "flagged_for_v386"),
        "methods_mapped_count": len(methods) if isinstance(methods, list) else 0,
        "honest_verdict": verdict_text(payload),
    }


def registry_status(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "regression_guard_passed": False}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "regression_guard_passed": False}
    return {
        "status": "included",
        "regression_guard_passed": bool_metric(payload, "regression_guard_passed") is True,
        "diffusiongemma_gate_state": dict(nested_map(payload, "diffusiongemma_gate_state")),
        "honest_verdict": verdict_text(payload),
    }


def hardware_continuity(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "kv260_terminal_confirmed": False}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "kv260_terminal_confirmed": False}
    reachability = payload.get("per_board_reachability")
    return {
        "status": "included",
        "kv260_terminal_confirmed": bool_metric(payload, "kv260_terminal_confirmed") is True,
        "per_board_reachability": dict(reachability) if isinstance(reachability, Mapping) else {},
        "gatemate_step_taken": str_metric(payload, "gatemate_step_taken"),
        "polarfire_step_taken": str_metric(payload, "polarfire_step_taken"),
        "honest_verdict": verdict_text(payload),
    }


def headline_outcome(
    baseline: Mapping[str, Any],
    moat: Mapping[str, Any],
    graft: Mapping[str, Any],
) -> str:
    if baseline.get("advanced_vs_seed") is not True:
        return "accumulation_still_blocked"
    if baseline.get("baseline_faithful") is True:
        if graft.get("status") == "ran_value_added":
            return "baseline_faithful_graft_validated"
        if graft.get("status") == "ran_null":
            return "baseline_faithful_graft_null"
    if moat.get("status") == "confirmed":
        return "baseline_advancing_moat_rerank_confirmed"
    if moat.get("status") == "null":
        return "baseline_advancing_moat_rerank_null"
    if moat.get("status") == "no_headroom":
        return "baseline_advancing_moat_rerank_no_headroom"
    return "accumulation_still_blocked"


def honest_verdict(
    outcome: str,
    baseline: Mapping[str, Any],
    moat: Mapping[str, Any],
    graft: Mapping[str, Any],
    gate_status: str,
    games_solved: int,
    skipped_count: int,
) -> str:
    prefix = "success:" if outcome in {
        "baseline_advancing_moat_rerank_confirmed",
        "baseline_faithful_graft_validated",
    } else "blocked:" if outcome == "accumulation_still_blocked" else "complete:"
    return (
        f"{prefix} capstone_v385_{outcome}_"
        f"baseline_{baseline.get('status', 'missing')}_"
        f"rerank_{moat.get('status', 'missing')}_"
        f"graft_{graft.get('status', 'missing')}_"
        f"diffusiongemma_{gate_status}_games{games_solved}_"
        f"flagged_skipped{skipped_count}"
    )


def headline_answers(
    baseline: Mapping[str, Any],
    moat: Mapping[str, Any],
    graft: Mapping[str, Any],
    gate_status: str,
    arc: Mapping[str, Any],
    observability: Mapping[str, Any],
) -> JsonDict:
    return {
        "contiguous_run_advanced_baseline": baseline.get("advanced_vs_seed"),
        "current_val_vs_seed": {
            "seed_val": baseline.get("seed_val"),
            "current_val": baseline.get("current_val"),
            "run_alive": baseline.get("run_alive"),
        },
        "rerank_moat_status": moat.get("status"),
        "full_graft_status": graft.get("status"),
        "diffusiongemma_gate_status": gate_status,
        "arc_action_efficiency_ratio": arc.get("action_efficiency_ratio"),
        "total_games_solved": arc.get("total_games_solved"),
        "observability_fixed": observability.get("fixed"),
    }


def imported_fields_by_id(clean_ids: set[int]) -> dict[int, list[str]]:
    fields: dict[int, list[str]] = {experiment_id: [] for experiment_id in UPSTREAM_IDS}
    if 4157 in clean_ids:
        fields[4157] = [
            "current_val",
            "max_val",
            "run_alive",
            "baseline_faithful",
            "manual_lr_step",
            "val_trajectory",
            "estimated_passes_to_085",
        ]
    if 4158 in clean_ids:
        fields[4158] = [
            "headroom_present",
            "oracle_at_k",
            "vote_at_1",
            "rerank_lift_vs_vote",
            "verifier_recovers_outvoted",
            "cost_ratio_vs_llm_judge",
        ]
    if 4159 in clean_ids:
        fields[4159] = [
            "graft_deferred",
            "verifier_value_added",
            "current_val",
            "rft_vs_ablation_delta",
        ]
    if 4160 in clean_ids:
        fields[4160] = [
            "action_efficiency_ratio",
            "baseline_actions",
            "verifier_actions",
            "actions_saved_vs_baseline",
            "total_games_solved",
            "new_levels_solved_this_task",
            "real_env_confirmed",
        ]
    if 4161 in clean_ids:
        fields[4161] = ["fix_applied", "fallback_added", "research_conductor_touched"]
    if 4162 in clean_ids:
        fields[4162] = ["methods_mapped", "flagged_for_v386"]
    if 4163 in clean_ids:
        fields[4163] = ["regression_guard_passed", "diffusiongemma_gate_state"]
    if 4164 in clean_ids:
        fields[4164] = [
            "kv260_terminal_confirmed",
            "per_board_reachability",
            "gatemate_step_taken",
            "polarfire_step_taken",
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
            "path": relative_to_root(root, paths[experiment_id]) if paths[experiment_id] else "",
            "reason": "flagged_adversarial:true",
            "sha256": sha256_file(paths[experiment_id]) if paths[experiment_id] else "",
        }
        for experiment_id in sorted(skipped_ids)
        if flagged(upstreams[experiment_id])
    ]


def upstream_provenance(
    root: Path,
    paths: Mapping[int, Path | None],
    upstreams: Mapping[int, Mapping[str, Any] | None],
    skipped_ids: set[int],
    fields_by_id: Mapping[int, list[str]],
) -> list[JsonDict]:
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
                "honest_verdict": verdict_text(upstreams[experiment_id]),
            }
        )
    return rows


def missing_upstream_artifacts(paths: Mapping[int, Path | None]) -> list[JsonDict]:
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
    return {
        str(experiment_id): {
            "exists": paths[experiment_id] is not None,
            "path": relative_to_root(root, paths[experiment_id])
            if paths[experiment_id] is not None
            else "",
            "honest_verdict": verdict_text(upstreams[experiment_id]),
            "flagged_adversarial": flagged(upstreams[experiment_id]),
            "included": experiment_id in clean_ids,
            "skipped": experiment_id in skipped_ids,
        }
        for experiment_id in UPSTREAM_IDS
    }


def duration_from(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return max(0.0001, end - started_s)


def payload_checksum(payload: Mapping[str, Any]) -> str:
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
    clean = {experiment_id: upstreams[experiment_id] for experiment_id in clean_ids}

    baseline = baseline_val_trajectory(clean.get(4157), was_skipped=4157 in skipped_ids)
    moat = moat_rerank_verdict(clean.get(4158), was_skipped=4158 in skipped_ids)
    graft = graft_verdict(clean.get(4159), was_skipped=4159 in skipped_ids)
    gate_status = diffusiongemma_gate_status(moat, graft)
    arc = arc_action_efficiency(clean.get(4160), was_skipped=4160 in skipped_ids)
    observability = observability_status(clean.get(4161), was_skipped=4161 in skipped_ids)
    sota = sota_guidance(clean.get(4162), was_skipped=4162 in skipped_ids)
    registry = registry_status(clean.get(4163), was_skipped=4163 in skipped_ids)
    hardware = hardware_continuity(clean.get(4164), was_skipped=4164 in skipped_ids)
    outcome = headline_outcome(baseline, moat, graft)
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_ids)
    total_games = int(arc["total_games_solved"])
    fields_by_id = imported_fields_by_id(clean_ids)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v385_4165.v1",
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict(
            outcome, baseline, moat, graft, gate_status, total_games, len(skipped)
        ),
        "headline_outcome": outcome,
        "headline_answers": headline_answers(
            baseline, moat, graft, gate_status, arc, observability
        ),
        "baseline_val_trajectory": baseline,
        "moat_rerank_verdict": moat,
        "graft_verdict": graft,
        "diffusiongemma_gate_status": gate_status,
        "diffusiongemma_gate": {
            "status": gate_status,
            "basis": "clean_exp4158_rerank_or_clean_exp4159_graft",
            "positive_verifier_discrimination": gate_status == "RESOLVED-positive",
            "resolved": gate_status != "STILL-PENDING",
        },
        "arc_action_efficiency": arc,
        "total_games_solved": total_games,
        "observability_status": observability,
        "sota_guidance": sota,
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
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not str(artifact.get("honest_verdict", "")).startswith(("complete:", "success:", "blocked:")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if artifact.get("headline_outcome") not in HEADLINE_OUTCOMES:
        raise ValueError("headline_outcome must be one of the enumerated values")
    trajectory = artifact.get("baseline_val_trajectory")
    if not isinstance(trajectory, Mapping) or not isinstance(trajectory.get("values"), list):
        raise ValueError("baseline_val_trajectory must contain numeric values")
    if not all(isinstance(value, int | float) and not isinstance(value, bool) for value in trajectory["values"]):
        raise ValueError("baseline_val_trajectory values must be numeric")
    moat = artifact.get("moat_rerank_verdict")
    if not isinstance(moat, Mapping) or moat.get("status") not in MOAT_STATUSES:
        raise ValueError("moat_rerank_verdict status must be enumerated")
    graft = artifact.get("graft_verdict")
    if not isinstance(graft, Mapping) or graft.get("status") not in GRAFT_STATUSES:
        raise ValueError("graft_verdict status must be enumerated")
    if artifact.get("diffusiongemma_gate_status") not in DIFFUSIONGEMMA_GATE_STATUSES:
        raise ValueError("diffusiongemma_gate_status must be enumerated")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
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
