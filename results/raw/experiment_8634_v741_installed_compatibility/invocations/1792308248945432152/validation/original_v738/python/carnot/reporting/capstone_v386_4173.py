"""Build the Exp 4173 v386 capstone aggregation.

Spec refs: REQ-CAPSTONE-4173, SCENARIO-CAPSTONE-4173.
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
OUTPUT_REL_PATH = Path("results/experiment_4173_capstone_v386.json")
EXPERIMENT_ID = 4173
RANDOM_SEED = 4173
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4173", "SCENARIO-CAPSTONE-4173"]

SEED_VAL = 0.278172343969
TARGET_VAL = 0.87
FAITHFUL_THRESHOLD = 0.85

UPSTREAM_IDS = (4165, 4167, 4168, 4169, 4170, 4171, 4172)
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    4165: Path("results/experiment_4165_capstone_v385.json"),
    4167: Path("results/experiment_4167_outerloop_training_monitor.json"),
    4168: Path("results/experiment_4168_decisive_verifier_graft_defensive.json"),
    4169: Path("results/experiment_4169_arc_incremental_progress.json"),
    4170: Path("results/experiment_4170_sota_ingestion_verifier_moat_guidance.json"),
    4171: Path("results/experiment_4171_verifier_registry_gaps_hygiene.json"),
    4172: Path("results/experiment_4172_hardware_continuity.json"),
}

HEADLINE_OUTCOMES = {
    "baseline_converged_graft_validated",
    "baseline_converged_graft_null",
    "outerloop_training_in_progress",
    "baseline_stalled",
}
GRAFT_STATUSES = {
    "deferred",
    "ran_value_added",
    "ran_null",
    "missing",
    "skipped_flagged_adversarial",
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
        "Terminal-prefixed. An honest 'outer-loop training in progress' is COMPLETE "
        "and valuable."
    ),
    "headline_outcome": "One of the enumerated set -- forces a single unambiguous read.",
    "baseline_val_trajectory": "Val from the outer-loop run; shows convergence toward 0.87.",
    "diffusiongemma_gate_status": "RESOLVED-positive / RESOLVED-null / STILL-PENDING.",
    "upstream_provenance": "{experiment_id, fields_imported, sha256} per cited upstream.",
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
    return str(path.relative_to(root))


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


def baseline_val_trajectory(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    base = {
        "source_experiment_id": 4167,
        "seed_val": SEED_VAL,
        "target_val": TARGET_VAL,
        "faithful_threshold": FAITHFUL_THRESHOLD,
        "current_val_exact_accuracy": None,
        "checkpoint_mtime": "",
        "outerloop_train_alive": None,
        "advanced_past_seed": None,
        "val_crossed_085": False,
        "val_crossed_087": False,
        "baseline_faithful": False,
        "values": [SEED_VAL],
        "rounded_values": [round(SEED_VAL, 3)],
        "honest_verdict": "" if was_skipped else verdict_text(payload),
    }
    if was_skipped:
        return {**base, "status": "skipped_flagged_adversarial"}
    if not isinstance(payload, Mapping):
        return {**base, "status": "missing"}

    current = float_metric(payload, "current_val_exact_accuracy")
    values = [SEED_VAL] + ([] if current is None else [current])
    advanced = current is not None and current > SEED_VAL
    crossed_085 = bool_metric(payload, "val_crossed_085") is True or (
        current is not None and current >= FAITHFUL_THRESHOLD
    )
    crossed_087 = current is not None and current >= TARGET_VAL
    faithful = bool_metric(payload, "baseline_faithful") is True or crossed_085
    if not advanced:
        status = "baseline_stalled"
    elif faithful:
        status = "baseline_converged"
    else:
        status = "outerloop_training_in_progress"
    return {
        **base,
        "status": status,
        "current_val_exact_accuracy": current,
        "checkpoint_mtime": str_metric(payload, "checkpoint_mtime"),
        "outerloop_train_alive": bool_metric(payload, "outerloop_train_alive"),
        "advanced_past_seed": advanced,
        "val_crossed_085": crossed_085,
        "val_crossed_087": crossed_087,
        "baseline_faithful": faithful,
        "values": values,
        "rounded_values": [round(value, 3) for value in values],
    }


def defensive_graft_verdict(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    base = {
        "source_experiment_id": 4168,
        "graft_deferred": False,
        "verifier_value_added": False,
        "baseline_current_val_exact_accuracy": None,
        "honest_verdict": "" if was_skipped else verdict_text(payload),
    }
    if was_skipped:
        return {**base, "status": "skipped_flagged_adversarial"}
    if not isinstance(payload, Mapping):
        return {**base, "status": "missing"}

    deferred = bool_metric(payload, "graft_deferred") is True
    value_added = bool_metric(payload, "verifier_value_added") is True
    baseline_status = nested_map(payload, "baseline_status")
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
        "baseline_current_val_exact_accuracy": float_metric(
            baseline_status, "current_val_exact_accuracy"
        ),
    }


def diffusiongemma_gate_status(graft: Mapping[str, Any]) -> str:
    if graft.get("status") == "ran_value_added":
        return "RESOLVED-positive"
    if graft.get("status") == "ran_null":
        return "RESOLVED-null"
    return "STILL-PENDING"


def arc_games_solved(
    current_arc: Mapping[str, Any] | None,
    carry_forward: Mapping[str, Any] | None,
    *,
    current_was_skipped: bool,
) -> JsonDict:
    if current_was_skipped:
        carried = int_metric(carry_forward, "total_games_solved") or int_metric(
            carry_forward, "total_arc_games_solved"
        )
        return {
            "status": "included_carry_forward" if carried else "missing_clean_source",
            "source": "prior_clean_carry_forward" if carried else "none",
            "source_experiment_id": 4165 if carried else None,
            "current_arc_experiment_id": 4169,
            "current_arc_status": "skipped_flagged_adversarial",
            "total_arc_games_solved": carried,
        }
    if isinstance(current_arc, Mapping):
        return {
            "status": "included_current",
            "source": "current_clean_exp4169",
            "source_experiment_id": 4169,
            "current_arc_experiment_id": 4169,
            "current_arc_status": "included",
            "total_arc_games_solved": int_metric(current_arc, "total_games_solved"),
        }
    carried = int_metric(carry_forward, "total_games_solved") or int_metric(
        carry_forward, "total_arc_games_solved"
    )
    return {
        "status": "included_carry_forward" if carried else "missing",
        "source": "prior_clean_carry_forward" if carried else "none",
        "source_experiment_id": 4165 if carried else None,
        "current_arc_experiment_id": 4169,
        "current_arc_status": "missing",
        "total_arc_games_solved": carried,
    }


def sota_guidance(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "methods_mapped_count": 0}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "methods_mapped_count": 0}
    methods = payload.get("methods_mapped")
    return {
        "status": "included",
        "flagged_for_v387": str_metric(payload, "flagged_for_v387"),
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


def headline_outcome(baseline: Mapping[str, Any], graft: Mapping[str, Any]) -> str:
    if baseline.get("advanced_past_seed") is not True:
        return "baseline_stalled"
    if baseline.get("baseline_faithful") is not True:
        return "outerloop_training_in_progress"
    if graft.get("status") == "ran_value_added":
        return "baseline_converged_graft_validated"
    return "baseline_converged_graft_null"


def honest_verdict(
    outcome: str,
    baseline: Mapping[str, Any],
    graft: Mapping[str, Any],
    gate_status: str,
    games_solved: int,
    skipped_count: int,
) -> str:
    return (
        f"complete: capstone_v386_{outcome}_"
        f"val_{baseline.get('current_val_exact_accuracy')}_"
        f"checkpoint_{baseline.get('checkpoint_mtime')}_"
        f"graft_{graft.get('status', 'missing')}_"
        f"diffusiongemma_{gate_status}_arc_games{games_solved}_"
        f"flagged_skipped{skipped_count}"
    )


def headline_answers(
    baseline: Mapping[str, Any],
    graft: Mapping[str, Any],
    gate_status: str,
    arc: Mapping[str, Any],
) -> JsonDict:
    return {
        "outerloop_advanced_past_0278_toward_087": baseline.get("advanced_past_seed") is True
        and baseline.get("val_crossed_087") is not True,
        "defensive_graft_result": graft.get("status"),
        "diffusiongemma_gate_status": gate_status,
        "total_arc_games_solved": arc.get("total_arc_games_solved"),
    }


def imported_fields_by_id(clean_ids: set[int], arc_source_id: int | None) -> dict[int, list[str]]:
    fields: dict[int, list[str]] = {experiment_id: [] for experiment_id in UPSTREAM_IDS}
    if 4165 in clean_ids and arc_source_id == 4165:
        fields[4165] = ["total_games_solved"]
    if 4167 in clean_ids:
        fields[4167] = [
            "current_val_exact_accuracy",
            "checkpoint_mtime",
            "outerloop_train_alive",
            "baseline_faithful",
            "val_crossed_085",
        ]
    if 4168 in clean_ids:
        fields[4168] = ["graft_deferred", "verifier_value_added", "baseline_status"]
    if 4169 in clean_ids and arc_source_id == 4169:
        fields[4169] = ["total_games_solved", "real_env_confirmed", "game_solved"]
    if 4170 in clean_ids:
        fields[4170] = ["methods_mapped", "flagged_for_v387"]
    if 4171 in clean_ids:
        fields[4171] = ["regression_guard_passed", "diffusiongemma_gate_state"]
    if 4172 in clean_ids:
        fields[4172] = [
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
            "path": relative_to_root(root, paths[experiment_id]),
            "reason": "flagged_adversarial:true",
            "sha256": sha256_file(paths[experiment_id]),
        }
        for experiment_id in sorted(skipped_ids)
        if paths[experiment_id] is not None and flagged(upstreams[experiment_id])
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
    skipped_ids = {
        experiment_id for experiment_id, payload in upstreams.items() if flagged(payload)
    }
    clean_ids = {
        experiment_id
        for experiment_id, payload in upstreams.items()
        if isinstance(payload, Mapping) and experiment_id not in skipped_ids
    }
    clean = {experiment_id: upstreams[experiment_id] for experiment_id in clean_ids}

    baseline = baseline_val_trajectory(clean.get(4167), was_skipped=4167 in skipped_ids)
    graft = defensive_graft_verdict(clean.get(4168), was_skipped=4168 in skipped_ids)
    gate_status = diffusiongemma_gate_status(graft)
    arc = arc_games_solved(
        clean.get(4169),
        clean.get(4165),
        current_was_skipped=4169 in skipped_ids,
    )
    sota = sota_guidance(clean.get(4170), was_skipped=4170 in skipped_ids)
    registry = registry_status(clean.get(4171), was_skipped=4171 in skipped_ids)
    hardware = hardware_continuity(clean.get(4172), was_skipped=4172 in skipped_ids)
    outcome = headline_outcome(baseline, graft)
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_ids)
    total_games = int(arc["total_arc_games_solved"])
    arc_source = arc.get("source_experiment_id")
    fields_by_id = imported_fields_by_id(
        clean_ids, arc_source if isinstance(arc_source, int) else None
    )

    artifact: JsonDict = {
        "schema": "carnot.capstone_v386_4173.v1",
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict(
            outcome, baseline, graft, gate_status, total_games, len(skipped)
        ),
        "headline_outcome": outcome,
        "headline_answers": headline_answers(baseline, graft, gate_status, arc),
        "baseline_val_trajectory": baseline,
        "defensive_graft_verdict": graft,
        "diffusiongemma_gate_status": gate_status,
        "diffusiongemma_gate": {
            "status": gate_status,
            "basis": "clean_exp4168_defensive_graft_verifier_value_added",
            "resolved": gate_status != "STILL-PENDING",
        },
        "arc_games_solved": arc,
        "total_arc_games_solved": total_games,
        "sota_guidance": sota,
        "registry_gap_hygiene": registry,
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
    if not str(artifact.get("honest_verdict", "")).startswith(("complete:", "success:", "blocked:")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if artifact.get("headline_outcome") not in HEADLINE_OUTCOMES:
        raise ValueError("headline_outcome must be one of the enumerated values")
    trajectory = artifact.get("baseline_val_trajectory")
    if not isinstance(trajectory, Mapping):
        raise ValueError("baseline_val_trajectory must be an object")
    current = trajectory.get("current_val_exact_accuracy")
    if current is not None and not isinstance(current, int | float):
        raise ValueError("current validation value must be numeric or null")
    graft = artifact.get("defensive_graft_verdict")
    if not isinstance(graft, Mapping) or graft.get("status") not in GRAFT_STATUSES:
        raise ValueError("defensive graft status must be enumerated")
    if artifact.get("diffusiongemma_gate_status") not in DIFFUSIONGEMMA_GATE_STATUSES:
        raise ValueError("DiffusionGemma gate status must be enumerated")
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
