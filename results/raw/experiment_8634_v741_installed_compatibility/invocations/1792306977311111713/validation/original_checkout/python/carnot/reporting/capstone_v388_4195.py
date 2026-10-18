"""Build the Exp 4195 v388 capstone aggregation.

Spec refs: REQ-CAPSTONE-4195, SCENARIO-CAPSTONE-4195.
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
OUTPUT_REL_PATH = Path("results/experiment_4195_capstone_v388.json")
EXPERIMENT_ID = 4195
RANDOM_SEED = 4195
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4195", "SCENARIO-CAPSTONE-4195"]

UPSTREAM_IDS = tuple(range(4185, 4195))
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    4185: Path("results/experiment_4185_headroom_recensus_llm_judge_harness.json"),
    4186: Path("results/experiment_4186_efficiency_moat_verifier_vs_llm_judge.json"),
    4187: Path("results/experiment_4187_gap4_graded_execution_gate_hardening.json"),
    4188: Path("results/experiment_4188_sovereign_local_generator_gap4_self_distill.json"),
    4189: Path("results/experiment_4189_diffusiongemma_verifier_guided_decoding.json"),
    4190: Path("results/experiment_4190_arc_incremental_progress.json"),
    4191: Path("results/experiment_4191_arc_live_env_grounding_probe.json"),
    4192: Path("results/experiment_4192_sota_ingestion_efficiency_gap4_diffusion.json"),
    4193: Path("results/experiment_4193_verifier_registry_gaps_hygiene.json"),
    4194: Path("results/experiment_4194_hardware_continuity.json"),
}

HEADLINE_OUTCOMES = {
    "efficiency_moat_won",
    "efficiency_moat_bounded_no_cost_advantage",
    "efficiency_moat_judge_more_accurate",
    "gap4_production_safe_and_sovereign",
    "gap4_sovereign_under_induces",
}
EFFICIENCY_MOAT_STATUSES = {
    "WON",
    "BOUNDED-no-cost-advantage",
    "JUDGE-MORE-ACCURATE",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "efficiency_moat_status",
    "gap4_production_safe",
    "gap4_sovereign",
    "diffusiongemma_feasible",
    "total_arc_levels_solved",
    "upstream_provenance",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'efficiency unwon / sovereign under-induces' is "
        "COMPLETE and decision-grade -- it tells .389 exactly which moat axis remains."
    ),
    "headline_outcome": "One of the enumerated set -- forces a single unambiguous read.",
    "efficiency_moat_status": (
        "WON / BOUNDED-no-cost-advantage / JUDGE-MORE-ACCURATE -- the north-star §5 "
        "efficiency-parity question's standing after .388."
    ),
    "gap4_production_safe": (
        "Whether the graded gate (exp4187) holds the +4/-0 safety property with the "
        "vote-aware guard."
    ),
    "gap4_sovereign": (
        "Whether a fully-local generator + GAP-4 verifier (exp4188) recovers ARC "
        "headroom with no closed-weight call (decentralization rule 1)."
    ),
    "diffusiongemma_feasible": (
        "Whether verifier-guided DiffusionGemma fired (exp4189) or honestly blocked "
        "-- the .389 scale-up readiness."
    ),
    "total_arc_levels_solved": ("The monotonic ARC progress metric after .388 (must be >= 14)."),
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
    except ValueError:  # pragma: no cover - defensive for external roots.
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
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def str_metric(payload: Mapping[str, Any] | None, field: str) -> str:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def nested_map(payload: Mapping[str, Any] | None, field: str) -> Mapping[str, Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _empty_efficiency(status: str) -> JsonDict:
    return {
        "status": status,
        "verifier_efficiency_win": False,
        "positive_control_confirmed": False,
        "accuracy_parity_vs_judge": {},
        "cost_ratio_vs_judge": {},
        "efficiency_moat_status": "BOUNDED-no-cost-advantage",
    }


def efficiency_moat(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return _empty_efficiency("skipped_flagged_adversarial")
    if not isinstance(payload, Mapping):
        return _empty_efficiency("missing")

    accuracy = dict(nested_map(payload, "accuracy_parity_vs_judge"))
    cost = dict(nested_map(payload, "cost_ratio_vs_judge"))
    verifier_win = bool_metric(payload, "verifier_efficiency_win") is True
    positive_control = bool_metric(payload, "positive_control_confirmed") is True
    within_ci = accuracy.get("within_ci_or_better") is True
    ten_x = cost.get("ten_x_cheaper_on_both_axes") is True
    pareto = cost.get("strictly_pareto_dominant") is True
    arm_a = float_metric(accuracy, "arm_a_pass1") or 0.0
    arm_j = float_metric(accuracy, "arm_j_pass1") or 0.0

    if verifier_win and positive_control and within_ci and (ten_x or pareto):
        status = "WON"
    elif arm_j > arm_a:
        status = "JUDGE-MORE-ACCURATE"
    else:
        status = "BOUNDED-no-cost-advantage"

    return {
        "status": "included",
        "efficiency_moat_status": status,
        "verifier_efficiency_win": verifier_win,
        "positive_control_confirmed": positive_control,
        "accuracy_parity_vs_judge": accuracy,
        "cost_ratio_vs_judge": cost,
        "north_star_reference": "ops/north-star.md §5",
        "honest_verdict": verdict_text(payload),
    }


def gap4_production_safety(
    payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    if was_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "safe": False,
            "gross_recovery_ledger": {"recovered": 0, "lost": 0},
        }
    if not isinstance(payload, Mapping):
        return {"status": "missing", "safe": False, "gross_recovery_ledger": {}}

    ledger = dict(nested_map(payload, "gross_recovery_ledger"))
    recovered = int_metric(ledger, "recovered")
    lost = int_metric(ledger, "lost")
    vote_wins_lost = int_metric(payload, "pass2_vote_wins_lost")
    guard_blocked = bool_metric(payload, "vote_aware_guard_blocked_mispromotion") is True
    safe = recovered >= 4 and lost == 0 and vote_wins_lost == 0 and guard_blocked
    return {
        "status": "HOLDS-plus4-minus0" if safe else "BOUNDED",
        "safe": safe,
        "graded_gate_pass2_vs_vote": float_metric(payload, "graded_gate_pass2_vs_vote"),
        "gross_recovery_ledger": {"recovered": recovered, "lost": lost},
        "pass2_vote_wins_lost": vote_wins_lost,
        "vote_aware_guard_blocked_mispromotion": guard_blocked,
        "vote_aware_guard": dict(nested_map(payload, "vote_aware_guard")),
        "honest_verdict": verdict_text(payload),
    }


def gap4_sovereignty(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    empty = {
        "recovered_arc_headroom": False,
        "no_closed_weight_call": None,
        "self_distillation_corpus_size": None,
        "sovereign_pool_pass2": {},
    }
    if was_skipped:
        return {**empty, "status": "skipped_flagged_adversarial"}
    if not isinstance(payload, Mapping):
        return {**empty, "status": "missing"}

    pool = dict(nested_map(payload, "sovereign_pool_pass2"))
    corpus_size = int_metric(payload, "self_distillation_corpus_size")
    delta_vs_vote = float_metric(pool, "delta_vs_vote") or 0.0
    no_closed = bool_metric(payload, "no_closed_weight_call") is True
    recovered = no_closed and corpus_size > 0 and delta_vs_vote > 0.0
    return {
        "status": "RECOVERS-headroom" if recovered else "UNDER-INDUCES",
        "recovered_arc_headroom": recovered,
        "no_closed_weight_call": no_closed,
        "self_distillation_corpus_size": corpus_size,
        "sovereign_pool_pass2": pool,
        "honest_verdict": verdict_text(payload),
    }


def diffusiongemma(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "diffusiongemma_feasible": False,
        }
    if not isinstance(payload, Mapping):
        return {"status": "missing", "diffusiongemma_feasible": False}
    feasible = bool_metric(payload, "diffusiongemma_feasible") is True
    return {
        "status": "feasible" if feasible else verdict_text(payload),
        "diffusiongemma_feasible": feasible,
        "guided_vs_unguided_delta": dict(nested_map(payload, "guided_vs_unguided_delta")),
        "model_specs": dict(nested_map(payload, "model_specs")),
        "honest_verdict": verdict_text(payload),
    }


def arc_progress(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "total_arc_levels_solved": 0,
        }
    if not isinstance(payload, Mapping):
        return {"status": "missing", "total_arc_levels_solved": 0}
    return {
        "status": "included",
        "total_arc_levels_solved": int_metric(payload, "total_levels_solved"),
        "total_arc_games_solved": int_metric(payload, "total_games_solved"),
        "new_levels_solved_this_task": int_metric(payload, "new_levels_solved_this_task"),
        "real_env_confirmed": bool_metric(payload, "real_env_confirmed") is True,
        "honest_verdict": verdict_text(payload),
    }


def live_env(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "live_env_reachable": False}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "live_env_reachable": False}
    return {
        "status": "included",
        "live_env_reachable": bool_metric(payload, "live_env_reachable") is True,
        "environment_count": int_metric(payload, "environment_count"),
        "random_greedy_baseline": dict(nested_map(payload, "random_greedy_baseline")),
        "honest_verdict": verdict_text(payload),
    }


def sota_v389(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "flagged_for_v389": ""}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "flagged_for_v389": ""}
    return {
        "status": "included",
        "flagged_for_v389": str_metric(payload, "flagged_for_v389"),
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
        "honest_verdict": verdict_text(payload),
    }


def headroom_harness(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "headroom_present": False}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "headroom_present": False}
    domain = str_metric(payload, "headroom_present_domain")
    headroom = float_metric(payload, "max_selectable_headroom") or 0.0
    return {
        "status": "included",
        "headroom_present": bool(domain) and headroom > 0.0,
        "headroom_present_domain": domain,
        "max_selectable_headroom": headroom,
        "acceptance_gate": bool_metric(payload, "acceptance_gate") is True,
        "honest_verdict": verdict_text(payload),
    }


def headline_outcome(
    efficiency_status: str,
    production_safe: bool,
    sovereign: bool,
) -> str:
    if efficiency_status == "WON":
        return "efficiency_moat_won"
    if efficiency_status == "JUDGE-MORE-ACCURATE":
        return "efficiency_moat_judge_more_accurate"
    if production_safe and sovereign:
        return "gap4_production_safe_and_sovereign"
    if production_safe and not sovereign:
        return "gap4_sovereign_under_induces"
    return "efficiency_moat_bounded_no_cost_advantage"


def honest_verdict(
    outcome: str,
    efficiency_status: str,
    production_safe: bool,
    sovereign: bool,
    diffusion_feasible: bool,
    total_levels: int,
    skipped_count: int,
) -> str:
    return (
        f"complete: capstone_v388_{outcome}_"
        f"efficiency_{efficiency_status}_"
        f"gap4_safe_{str(production_safe).lower()}_"
        f"sovereign_{str(sovereign).lower()}_"
        f"diffusiongemma_{str(diffusion_feasible).lower()}_"
        f"arc_levels{total_levels}_flagged_skipped{skipped_count}"
    )


def imported_fields_by_id(clean_ids: set[int]) -> dict[int, list[str]]:
    fields: dict[int, list[str]] = {experiment_id: [] for experiment_id in UPSTREAM_IDS}
    if 4185 in clean_ids:
        fields[4185] = [
            "acceptance_gate",
            "headroom_present_domain",
            "max_selectable_headroom",
            "per_domain_headroom",
        ]
    if 4186 in clean_ids:
        fields[4186] = [
            "verifier_efficiency_win",
            "positive_control_confirmed",
            "accuracy_parity_vs_judge",
            "cost_ratio_vs_judge",
        ]
    if 4187 in clean_ids:
        fields[4187] = [
            "graded_gate_pass2_vs_vote",
            "gross_recovery_ledger",
            "pass2_vote_wins_lost",
            "vote_aware_guard_blocked_mispromotion",
        ]
    if 4188 in clean_ids:
        fields[4188] = [
            "no_closed_weight_call",
            "self_distillation_corpus_size",
            "sovereign_pool_pass2",
        ]
    if 4189 in clean_ids:
        fields[4189] = [
            "diffusiongemma_feasible",
            "guided_vs_unguided_delta",
            "model_specs",
        ]
    if 4190 in clean_ids:
        fields[4190] = [
            "total_levels_solved",
            "total_games_solved",
            "new_levels_solved_this_task",
            "real_env_confirmed",
        ]
    if 4191 in clean_ids:
        fields[4191] = [
            "live_env_reachable",
            "environment_count",
            "random_greedy_baseline",
        ]
    if 4192 in clean_ids:
        fields[4192] = ["flagged_for_v389"]
    if 4193 in clean_ids:
        fields[4193] = ["regression_guard_passed"]
    if 4194 in clean_ids:
        fields[4194] = ["kv260_reachable", "gatemate_reachable", "polarfire_reachable"]
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
            "honest_verdict": verdict_text(upstreams[experiment_id]),
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

    headroom = headroom_harness(clean.get(4185), was_skipped=4185 in skipped_ids)
    efficiency = efficiency_moat(clean.get(4186), was_skipped=4186 in skipped_ids)
    production = gap4_production_safety(clean.get(4187), was_skipped=4187 in skipped_ids)
    sovereign_detail = gap4_sovereignty(clean.get(4188), was_skipped=4188 in skipped_ids)
    diffusion = diffusiongemma(clean.get(4189), was_skipped=4189 in skipped_ids)
    arc = arc_progress(clean.get(4190), was_skipped=4190 in skipped_ids)
    live = live_env(clean.get(4191), was_skipped=4191 in skipped_ids)
    sota = sota_v389(clean.get(4192), was_skipped=4192 in skipped_ids)
    registry = registry_hygiene(clean.get(4193), was_skipped=4193 in skipped_ids)
    hardware = hardware_continuity(clean.get(4194), was_skipped=4194 in skipped_ids)

    efficiency_status = str(efficiency["efficiency_moat_status"])
    production_safe = production.get("safe") is True
    sovereign = sovereign_detail.get("recovered_arc_headroom") is True
    diffusion_feasible = diffusion.get("diffusiongemma_feasible") is True
    total_levels = int(arc["total_arc_levels_solved"])
    outcome = headline_outcome(efficiency_status, production_safe, sovereign)
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_ids)
    fields_by_id = imported_fields_by_id(clean_ids)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v388_4195.v1",
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict(
            outcome,
            efficiency_status,
            production_safe,
            sovereign,
            diffusion_feasible,
            total_levels,
            len(skipped),
        ),
        "headline_outcome": outcome,
        "efficiency_moat_status": efficiency_status,
        "efficiency_moat": efficiency,
        "gap4_production_safe": production_safe,
        "gap4_production_safety": production,
        "gap4_sovereign": sovereign,
        "gap4_sovereign_detail": sovereign_detail,
        "diffusiongemma_feasible": diffusion_feasible,
        "diffusiongemma_detail": diffusion,
        "total_arc_levels_solved": total_levels,
        "arc_progress": arc,
        "live_env_reachable": live.get("live_env_reachable") is True,
        "live_env": live,
        "strongest_sota_flagged_for_v389": str(sota.get("flagged_for_v389") or ""),
        "sota_v389": sota,
        "headroom_harness": headroom,
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
        raise ValueError("headline_outcome must be one of the enumerated values")
    if artifact.get("efficiency_moat_status") not in EFFICIENCY_MOAT_STATUSES:
        raise ValueError("efficiency_moat_status must be enumerated")
    if not isinstance(artifact.get("gap4_production_safe"), bool):
        raise ValueError("gap4_production_safe must be a bool")
    if not isinstance(artifact.get("gap4_sovereign"), bool):
        raise ValueError("gap4_sovereign must be a bool")
    if not isinstance(artifact.get("diffusiongemma_feasible"), bool):
        raise ValueError("diffusiongemma_feasible must be a bool")
    total_levels = artifact.get("total_arc_levels_solved")
    if not isinstance(total_levels, int) or isinstance(total_levels, bool) or total_levels < 14:
        raise ValueError("total ARC levels must be an integer >= 14")
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
