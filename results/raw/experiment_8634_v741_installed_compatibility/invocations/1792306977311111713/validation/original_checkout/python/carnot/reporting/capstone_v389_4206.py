"""Build the Exp 4206 v389 verifier-as-reward capstone aggregation.

Spec refs: REQ-CAPSTONE-4206, SCENARIO-CAPSTONE-4206.
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
OUTPUT_REL_PATH = Path("results/experiment_4206_capstone_v389.json")
EXPERIMENT_ID = 4206
RANDOM_SEED = 4206
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4206", "SCENARIO-CAPSTONE-4206"]

DEFAULT_UPSTREAM_PATHS: Mapping[str, Path] = {
    "4197_smoke": Path("results/experiment_4197_verifier_reward_code_lora_rft_3arm_smoke.json"),
    "4197_phase0": Path(
        "results/experiment_4197_verifier_reward_phase0_headroom_harness_build.json"
    ),
    "4198": Path("results/experiment_4198_verifier_reward_3arm_rft_launch.json"),
    "4199": Path("results/experiment_4199_verifier_reward_decisive_a_vs_b_collect.json"),
    "4200": Path("results/experiment_4200_certified_arc_corpus_distill_lift.json"),
    "4201": Path("results/experiment_4201_arc_incremental_progress.json"),
    "4202": Path("results/experiment_4202_arc_live_env_solver_vs_floor.json"),
    "4203": Path("results/experiment_4203_sota_ingestion_verifier_as_reward.json"),
    "4204": Path("results/experiment_4204_verifier_registry_gaps_hygiene.json"),
    "4205": Path("results/experiment_4205_hardware_continuity.json"),
}

UPSTREAM_EXPERIMENT_IDS: Mapping[str, int] = {
    "4197_smoke": 4197,
    "4197_phase0": 4197,
    "4198": 4198,
    "4199": 4199,
    "4200": 4200,
    "4201": 4201,
    "4202": 4202,
    "4203": 4203,
    "4204": 4204,
    "4205": 4205,
}

HEADLINE_OUTCOMES = {
    "verifier_reward_real_label_carries_training_signal",
    "verifier_reward_null_equals_distillation_or_spurious",
    "verifier_reward_invalid_or_underpowered",
    "verifier_reward_no_code_operating_point",
}

VERIFIER_AS_REWARD_STATUSES = {
    "REAL",
    "NULL",
    "INVALID-or-UNDERPOWERED",
    "NO-OPERATING-POINT",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "verifier_as_reward_status",
    "phase0_operating_point",
    "arc_distill_latent_or_absent",
    "total_arc_levels_solved",
    "upstream_provenance",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'A~=B = distillation null' or 'no operating point "
        "cleared' is COMPLETE and decision-grade -- it tells .390 exactly whether the "
        "pivot's mechanism works."
    ),
    "headline_outcome": (
        "One of the enumerated set -- forces a single unambiguous read of the pivot's "
        "first clean test."
    ),
    "verifier_as_reward_status": (
        "REAL (label carries signal) / NULL (distillation/spurious) / "
        "INVALID-or-UNDERPOWERED / NO-OPERATING-POINT -- the operator-pivot's standing "
        "after .389."
    ),
    "phase0_operating_point": (
        "The code Phase-0 precision + Youden J the test cleared (exp4197) -- the "
        "precondition that grids failed and code met."
    ),
    "arc_distill_latent_or_absent": (
        "Whether the certified-ARC-corpus in-context lift (exp4200) shows the "
        "abstraction latent (distill viable .390) or absent (need a stronger base)."
    ),
    "total_arc_levels_solved": "The monotonic ARC progress metric after .389 (must be >= 15).",
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


def relative_to_root(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:  # pragma: no cover - supports external output roots.
        return str(path)


def selected_upstream_paths(root: Path | str) -> dict[str, Path | None]:
    root_path = Path(root)
    return {
        upstream_key: path if (path := root_path / rel_path).exists() else None
        for upstream_key, rel_path in DEFAULT_UPSTREAM_PATHS.items()
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


def phase0_operating_point(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    empty = {
        "phase0_precision": None,
        "youden_j": None,
        "phase0_clears": False,
        "training_headroom_present": False,
        "operating_point": {},
    }
    if was_skipped:
        return {**empty, "status": "skipped_flagged_adversarial"}
    if not isinstance(payload, Mapping):
        return {**empty, "status": "missing"}

    detail = nested_map(payload, "phase0_detail")
    precision = float_metric(payload, "phase0_precision")
    youden = float_metric(payload, "youden_j")
    phase0_clears = detail.get("phase0_clears") is True or (
        (precision or 0.0) >= 0.85
        and (youden or 0.0) > 0.0
        and bool_metric(payload, "training_headroom_present") is True
    )
    return {
        "status": "included",
        "phase0_precision": precision,
        "youden_j": youden,
        "phase0_clears": phase0_clears,
        "training_headroom_present": bool_metric(payload, "training_headroom_present") is True,
        "operating_point": dict(nested_map(payload, "operating_point")),
        "phase0_detail": dict(detail),
        "honest_verdict": verdict_text(payload),
    }


def a_vs_b_training_signal(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    empty = {
        "a_vs_b_delta": None,
        "a_vs_b_ci95": None,
        "a_vs_b_ci_excludes_zero": False,
        "verifier_label_carries_signal": False,
        "positive_control_confirmed": False,
        "truncation_guard_confirmed": False,
        "controls_confirmed": False,
    }
    if was_skipped:
        return {**empty, "status": "skipped_flagged_adversarial"}
    if not isinstance(payload, Mapping):
        return {**empty, "status": "missing"}

    delta = float_metric(payload, "a_vs_b_delta")
    interval = ci95(payload, "a_vs_b_ci95")
    ci_excludes_zero = interval is not None and (interval[0] > 0.0 or interval[1] < 0.0)
    label_signal = bool_metric(payload, "verifier_label_carries_signal") is True or (
        delta is not None and delta > 0.0 and interval is not None and interval[0] > 0.0
    )
    positive_control = bool_metric(payload, "positive_control_confirmed") is True
    truncation_guard = bool_metric(payload, "truncation_guard_confirmed") is True
    blocked = str_metric(payload, "status") == "blocked" or verdict_text(payload).startswith(
        "blocked:"
    )
    status = "blocked_a_vs_b_not_collected" if blocked else "included"
    return {
        **empty,
        "status": status,
        "a_vs_b_delta": delta,
        "a_vs_b_ci95": interval,
        "a_vs_b_ci_excludes_zero": ci_excludes_zero,
        "verifier_label_carries_signal": label_signal and ci_excludes_zero,
        "positive_control_confirmed": positive_control,
        "truncation_guard_confirmed": truncation_guard,
        "controls_confirmed": positive_control and truncation_guard,
        "gate_check_summary": str_metric(payload, "gate_check_summary"),
        "blocked_at_layer": str_metric(payload, "blocked_at_layer"),
        "honest_verdict": verdict_text(payload),
    }


def arc_distill_latent_or_absent(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    empty = {
        "latent_or_absent": "UNINFORMATIVE",
        "certification_precision": {},
        "certified_corpus_size": 0,
        "distill_lift_ci95": None,
    }
    if was_skipped:
        return {
            **empty,
            "status": "skipped_flagged_adversarial",
            "latent_or_absent": "UNINFORMATIVE-skipped_flagged_adversarial",
        }
    if not isinstance(payload, Mapping):
        return {**empty, "status": "missing"}

    interval = ci95(payload, "distill_lift_ci95")
    diagnosis = str_metric(payload, "invisible_leash_diagnosis").lower()
    if diagnosis == "latent" or (interval is not None and interval[0] > 0.0):
        latent_or_absent = "LATENT"
    elif diagnosis == "absent" or (interval is not None and interval[1] <= 0.0):
        latent_or_absent = "ABSENT"
    else:
        latent_or_absent = "UNINFORMATIVE"

    return {
        "status": "included",
        "latent_or_absent": latent_or_absent,
        "certification_precision": dict(nested_map(payload, "certification_precision")),
        "certified_corpus_size": int_metric(payload, "certified_corpus_size"),
        "distill_lift_ci95": interval,
        "invisible_leash_diagnosis": str_metric(payload, "invisible_leash_diagnosis"),
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
        "prior_total_levels_solved": int_metric(payload, "prior_total_levels_solved"),
        "acceptance_gate_passed": bool_metric(payload, "acceptance_gate_passed") is True,
        "real_env_confirmed": bool_metric(payload, "real_env_confirmed") is True,
        "honest_verdict": verdict_text(payload),
    }


def live_solver_vs_floor(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    empty = {
        "live_env_reachable": False,
        "solver_beats_floor_overall": False,
        "solver_beats_floor_accuracy": False,
        "solver_beats_floor_efficiency": False,
    }
    if was_skipped:
        return {**empty, "status": "skipped_flagged_adversarial"}
    if not isinstance(payload, Mapping):
        return {**empty, "status": "missing"}

    beats = nested_map(payload, "solver_beats_floor")
    accuracy = nested_map(beats, "accuracy")
    efficiency = nested_map(beats, "efficiency")
    return {
        "status": "included",
        "live_env_reachable": bool_metric(payload, "live_env_reachable") is True,
        "solver_beats_floor_overall": bool_metric(beats, "overall") is True,
        "solver_beats_floor_accuracy": bool_metric(accuracy, "beats") is True,
        "solver_beats_floor_efficiency": bool_metric(efficiency, "beats") is True,
        "live_env_metrics": dict(nested_map(payload, "live_env_metrics")),
        "random_greedy_floor": dict(nested_map(payload, "random_greedy_floor")),
        "honest_verdict": verdict_text(payload),
    }


def sota_v390(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "flagged_for_v390": ""}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "flagged_for_v390": ""}
    return {
        "status": "included",
        "flagged_for_v390": str_metric(payload, "flagged_for_v390"),
        "methods_mapped_count": len(payload.get("methods_mapped", []))
        if isinstance(payload.get("methods_mapped"), list)
        else 0,
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
        return {"status": "skipped_flagged_adversarial", "kv260_reachable": False}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "kv260_reachable": False}
    return {
        "status": "included",
        "kv260_reachable": bool_metric(payload, "kv260_reachable") is True,
        "gatemate_reachable": bool_metric(payload, "gatemate_reachable") is True,
        "polarfire_reachable": bool_metric(payload, "polarfire_reachable") is True,
        "fabric_acceleration_claimed": bool_metric(payload, "fabric_acceleration_claimed") is True,
        "speedup_claim_made": bool_metric(payload, "speedup_claim_made") is True,
        "honest_verdict": verdict_text(payload),
    }


def smoke_harness(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "harness_ready": False}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "harness_ready": False}
    return {
        "status": "included",
        "smoke": bool_metric(payload, "smoke") is True,
        "harness_ready": bool_metric(payload, "harness_ready") is True,
        "truncation_guard": dict(nested_map(payload, "truncation_guard")),
        "honest_verdict": verdict_text(payload),
    }


def headline_outcome(phase0: Mapping[str, Any], a_vs_b: Mapping[str, Any]) -> str:
    if phase0.get("status") != "included" or phase0.get("phase0_clears") is not True:
        return "verifier_reward_no_code_operating_point"
    if a_vs_b.get("controls_confirmed") is not True:
        return "verifier_reward_invalid_or_underpowered"
    if a_vs_b.get("a_vs_b_ci95") is None:
        return "verifier_reward_invalid_or_underpowered"
    if (
        a_vs_b.get("verifier_label_carries_signal") is True
        and a_vs_b.get("a_vs_b_ci_excludes_zero") is True
    ):
        return "verifier_reward_real_label_carries_training_signal"
    return "verifier_reward_null_equals_distillation_or_spurious"


def verifier_as_reward_status(outcome: str) -> str:
    return {
        "verifier_reward_real_label_carries_training_signal": "REAL",
        "verifier_reward_null_equals_distillation_or_spurious": "NULL",
        "verifier_reward_invalid_or_underpowered": "INVALID-or-UNDERPOWERED",
        "verifier_reward_no_code_operating_point": "NO-OPERATING-POINT",
    }[outcome]


def honest_verdict(outcome: str, total_levels: int, skipped_count: int) -> str:
    return (
        f"complete: capstone_v389_{outcome}_"
        f"status_{verifier_as_reward_status(outcome)}_"
        f"arc_levels{total_levels}_flagged_skipped{skipped_count}"
    )


def imported_fields_by_key(clean_keys: set[str]) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {upstream_key: [] for upstream_key in DEFAULT_UPSTREAM_PATHS}
    if "4197_smoke" in clean_keys:
        fields["4197_smoke"] = ["smoke", "harness_ready", "truncation_guard"]
    if "4197_phase0" in clean_keys:
        fields["4197_phase0"] = [
            "phase0_precision",
            "youden_j",
            "training_headroom_present",
            "operating_point",
            "phase0_detail",
        ]
    if "4198" in clean_keys:
        fields["4198"] = ["training_launched", "gold_control_early_read", "truncation_guard"]
    if "4199" in clean_keys:
        fields["4199"] = [
            "status",
            "gate_check_summary",
            "gates_evaluated",
            "blocked_at_layer",
            "a_vs_b_delta",
            "a_vs_b_ci95",
            "verifier_label_carries_signal",
            "positive_control_confirmed",
            "truncation_guard_confirmed",
        ]
    if "4200" in clean_keys:
        fields["4200"] = [
            "certification_precision",
            "certified_corpus_size",
            "distill_lift_ci95",
            "invisible_leash_diagnosis",
        ]
    if "4201" in clean_keys:
        fields["4201"] = [
            "total_levels_solved",
            "total_games_solved",
            "new_levels_solved_this_task",
            "prior_total_levels_solved",
        ]
    if "4202" in clean_keys:
        fields["4202"] = [
            "live_env_reachable",
            "solver_beats_floor",
            "live_env_metrics",
            "random_greedy_floor",
        ]
    if "4203" in clean_keys:
        fields["4203"] = ["flagged_for_v390", "methods_mapped"]
    if "4204" in clean_keys:
        fields["4204"] = ["regression_guard_passed"]
    if "4205" in clean_keys:
        fields["4205"] = [
            "kv260_reachable",
            "gatemate_reachable",
            "polarfire_reachable",
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
    rows: list[JsonDict] = []
    for upstream_key in DEFAULT_UPSTREAM_PATHS:
        path = paths[upstream_key]
        if path is not None and upstream_key in skipped_keys:
            rows.append(
                {
                    "experiment_id": UPSTREAM_EXPERIMENT_IDS[upstream_key],
                    "upstream_key": upstream_key,
                    "path": relative_to_root(root, path),
                    "reason": "flagged_adversarial:true",
                    "sha256": sha256_file(path),
                    "honest_verdict": verdict_text(upstreams[upstream_key]),
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
    for upstream_key in DEFAULT_UPSTREAM_PATHS:
        path = paths[upstream_key]
        if path is None:
            continue
        skipped = upstream_key in skipped_keys
        rows.append(
            {
                "experiment_id": UPSTREAM_EXPERIMENT_IDS[upstream_key],
                "upstream_key": upstream_key,
                "path": relative_to_root(root, path),
                "sha256": sha256_file(path),
                "fields_imported": [] if skipped else list(fields_by_key.get(upstream_key, [])),
                "skipped": skipped,
                "skip_reason": "flagged_adversarial:true" if skipped else "",
                "honest_verdict": verdict_text(upstreams[upstream_key]),
            }
        )
    return rows


def missing_upstream_artifacts(paths: Mapping[str, Path | None]) -> list[JsonDict]:
    return [
        {"experiment_id": UPSTREAM_EXPERIMENT_IDS[upstream_key], "upstream_key": upstream_key}
        for upstream_key in DEFAULT_UPSTREAM_PATHS
        if paths[upstream_key] is None
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
        upstream_key: read_json_object(path) if path is not None else None
        for upstream_key, path in paths.items()
    }
    skipped_keys = {upstream_key for upstream_key, payload in upstreams.items() if flagged(payload)}
    clean_keys = {
        upstream_key
        for upstream_key, payload in upstreams.items()
        if isinstance(payload, Mapping) and upstream_key not in skipped_keys
    }
    clean = {upstream_key: upstreams[upstream_key] for upstream_key in clean_keys}

    smoke = smoke_harness(clean.get("4197_smoke"), was_skipped="4197_smoke" in skipped_keys)
    phase0 = phase0_operating_point(
        clean.get("4197_phase0"), was_skipped="4197_phase0" in skipped_keys
    )
    a_vs_b = a_vs_b_training_signal(clean.get("4199"), was_skipped="4199" in skipped_keys)
    distill = arc_distill_latent_or_absent(clean.get("4200"), was_skipped="4200" in skipped_keys)
    arc = arc_progress(clean.get("4201"), was_skipped="4201" in skipped_keys)
    live = live_solver_vs_floor(clean.get("4202"), was_skipped="4202" in skipped_keys)
    sota = sota_v390(clean.get("4203"), was_skipped="4203" in skipped_keys)
    registry = registry_hygiene(clean.get("4204"), was_skipped="4204" in skipped_keys)
    hardware = hardware_continuity(clean.get("4205"), was_skipped="4205" in skipped_keys)

    outcome = headline_outcome(phase0, a_vs_b)
    status = verifier_as_reward_status(outcome)
    total_levels = int(arc["total_arc_levels_solved"])
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_keys)
    fields_by_key = imported_fields_by_key(clean_keys)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v389_4206.v1",
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict(outcome, total_levels, len(skipped)),
        "headline_outcome": outcome,
        "verifier_as_reward_status": status,
        "phase0_operating_point": phase0,
        "three_arm_smoke": smoke,
        "a_vs_b_training_signal": a_vs_b,
        "arc_distill_latent_or_absent": distill,
        "total_arc_levels_solved": total_levels,
        "arc_progress": arc,
        "live_solver_vs_floor": live,
        "strongest_sota_flagged_for_v390": str(sota.get("flagged_for_v390") or ""),
        "sota_v390": sota,
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
        raise ValueError("headline_outcome must be one of the enumerated values")
    if artifact.get("verifier_as_reward_status") not in VERIFIER_AS_REWARD_STATUSES:
        raise ValueError("verifier_as_reward_status must be enumerated")
    if not isinstance(artifact.get("phase0_operating_point"), Mapping):
        raise ValueError("phase0_operating_point must be an object")
    if not isinstance(artifact.get("arc_distill_latent_or_absent"), Mapping):
        raise ValueError("arc_distill_latent_or_absent must be an object")
    total_levels = artifact.get("total_arc_levels_solved")
    if not isinstance(total_levels, int) or isinstance(total_levels, bool) or total_levels < 15:
        raise ValueError("total ARC levels must be an integer >= 15")
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
        if not isinstance(row.get("upstream_key"), str):
            raise ValueError("upstream_provenance entries need upstream_key")
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
