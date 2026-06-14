"""Build the Exp 4183 v387 capstone aggregation.

Spec refs: REQ-CAPSTONE-4183, SCENARIO-CAPSTONE-4183.
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
OUTPUT_REL_PATH = Path("results/experiment_4183_capstone_v387.json")
EXPERIMENT_ID = 4183
RANDOM_SEED = 4183
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4183", "SCENARIO-CAPSTONE-4183"]

HEADROOM_THRESHOLD = 0.10
EXECUTABLE_DOMAINS = {"code", "sudoku", "math"}

UPSTREAM_IDS = (4175, 4177, 4178, 4179, 4180, 4181, 4182)
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    4175: Path("results/experiment_4175_headroom_gate_executable_census.json"),
    4177: Path("results/experiment_4177_decisive_headroom_controlled_moat_test.json"),
    4178: Path("results/experiment_4178_gap3_stage1_model_native_arc_energy.json"),
    4179: Path("results/experiment_4179_arc_incremental_progress.json"),
    4180: Path("results/experiment_4180_sota_ingestion_moat_gap3_diffusion.json"),
    4181: Path("results/experiment_4181_verifier_registry_gaps_hygiene.json"),
    4182: Path("results/experiment_4182_hardware_continuity.json"),
}

HEADLINE_OUTCOMES = {
    "moat_proven_headroom_present",
    "moat_bounded_clean_null",
    "moat_deferred_no_headroom",
    "gap3_reaches_headroom",
    "gap3_bounded",
}
VERIFIER_MOAT_STATUSES = {
    "PROVEN-headroom-present",
    "BOUNDED-clean-null",
    "DEFERRED-no-headroom",
}
GAP3_STAGE1_STATUSES = {"REACHES-headroom", "BOUNDED"}
DIFFUSIONGEMMA_GATE_STATUSES = {
    "MET",
    "STILL-PENDING-headroom-present-null",
    "STILL-PENDING-no-headroom",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "verifier_moat_status",
    "gap3_stage1_status",
    "diffusiongemma_gate_status",
    "upstream_provenance",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest bounded-null is COMPLETE and decision-grade -- "
        "it tells .388 the bottleneck is verifier discrimination (GAP-3), not the "
        "substrate."
    ),
    "headline_outcome": "One of the enumerated set -- forces a single unambiguous read.",
    "verifier_moat_status": (
        "PROVEN-headroom-present / BOUNDED-clean-null / DEFERRED-no-headroom -- "
        "the existential question's standing after .387."
    ),
    "gap3_stage1_status": (
        "Whether GAP-3 Stage-1 reached the ~13pp headroom (advances toward `filled`) "
        "or is bounded."
    ),
    "diffusiongemma_gate_status": (
        "MET / STILL-PENDING-headroom-present-null / STILL-PENDING-no-headroom; "
        "MET only on a positive-controlled executable win."
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


def list_float_metric(payload: Mapping[str, Any] | None, field: str) -> list[float]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    if not isinstance(value, list):
        return []
    return [
        float(item)
        for item in value
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    ]


def headroom_census(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "headroom_present": False,
            "headroom_present_domain": "",
            "max_selectable_headroom": 0.0,
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "missing",
            "headroom_present": False,
            "headroom_present_domain": "",
            "max_selectable_headroom": 0.0,
        }
    domain = str_metric(payload, "headroom_present_domain")
    max_headroom = float_metric(payload, "max_selectable_headroom") or 0.0
    return {
        "status": "included",
        "headroom_present": bool(domain) and max_headroom >= HEADROOM_THRESHOLD,
        "headroom_present_domain": domain,
        "max_selectable_headroom": max_headroom,
        "honest_verdict": verdict_text(payload),
    }


def moat_verdict(
    payload: Mapping[str, Any] | None,
    census: Mapping[str, Any],
    *,
    was_skipped: bool,
) -> JsonDict:
    base = {
        "headroom_present": census.get("headroom_present") is True,
        "headroom_present_domain": str(census.get("headroom_present_domain") or ""),
        "max_selectable_headroom": census.get("max_selectable_headroom", 0.0),
        "domain": "",
        "executable_domain": False,
        "verifier_value_added": False,
        "positive_control_confirmed": False,
        "honest_verdict": "" if was_skipped else verdict_text(payload),
    }
    if was_skipped:
        return {**base, "status": "skipped_flagged_adversarial"}
    if not isinstance(payload, Mapping):
        return {**base, "status": "missing"}

    domain = str_metric(payload, "domain") or str(base["headroom_present_domain"])
    positive_control = bool_metric(payload, "positive_control_confirmed") is True
    value_added = bool_metric(payload, "verifier_value_added") is True
    headroom_present = base["headroom_present"] and domain in EXECUTABLE_DOMAINS
    if not headroom_present:
        status = "DEFERRED-no-headroom"
    elif value_added and positive_control:
        status = "PROVEN-headroom-present"
    else:
        status = "BOUNDED-clean-null"
    return {
        **base,
        "status": status,
        "domain": domain,
        "executable_domain": domain in EXECUTABLE_DOMAINS,
        "verifier_value_added": value_added,
        "positive_control_confirmed": positive_control,
        "moat_delta_vs_vote": dict(nested_map(payload, "moat_delta_vs_vote")),
        "moat_vs_matched_control": dict(nested_map(payload, "moat_vs_matched_control")),
        "accuracy_cost_pareto": dict(nested_map(payload, "accuracy_cost_pareto")),
        "positive_control": dict(nested_map(payload, "positive_control")),
    }


def gap3_stage1(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "reaches_proven_arc_headroom": False,
            "pass2_energy_vs_vote": 0.0,
            "all_four_gates_pass": False,
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "missing",
            "reaches_proven_arc_headroom": False,
            "pass2_energy_vs_vote": 0.0,
            "all_four_gates_pass": False,
        }
    delta = float_metric(payload, "pass2_energy_vs_vote") or 0.0
    all_gates = bool_metric(payload, "all_four_gates_pass") is True
    reaches = delta > 0.0 and all_gates
    return {
        "status": "REACHES-headroom" if reaches else "BOUNDED",
        "reaches_proven_arc_headroom": reaches,
        "pass2_energy_vs_vote": delta,
        "all_four_gates_pass": all_gates,
        "headroom_capture_fraction": float_metric(payload, "headroom_capture_fraction"),
        "candidate_auroc": float_metric(payload, "candidate_auroc"),
        "gates": dict(nested_map(payload, "gates")),
        "honest_verdict": verdict_text(payload),
    }


def arc_progress(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "total_arc_levels_solved": 0,
            "total_arc_games_solved": 0,
        }
    if not isinstance(payload, Mapping):
        return {"status": "missing", "total_arc_levels_solved": 0, "total_arc_games_solved": 0}
    return {
        "status": "included",
        "total_arc_levels_solved": int_metric(payload, "total_levels_solved"),
        "total_arc_games_solved": int_metric(payload, "total_games_solved"),
        "new_levels_solved_this_task": int_metric(payload, "new_levels_solved_this_task"),
        "real_env_confirmed": bool_metric(payload, "real_env_confirmed") is True,
        "honest_verdict": verdict_text(payload),
    }


def sota_v388(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "flagged_for_v388": ""}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "flagged_for_v388": ""}
    return {
        "status": "included",
        "flagged_for_v388": str_metric(payload, "flagged_for_v388"),
        "honest_verdict": verdict_text(payload),
    }


def registry_gap_hygiene(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "regression_guard_passed": False}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "regression_guard_passed": False}
    return {
        "status": "included",
        "regression_guard_passed": bool_metric(payload, "regression_guard_passed") is True,
        "moat_verdict": dict(nested_map(payload, "moat_verdict")),
        "gap3_stage1_result": dict(nested_map(payload, "gap3_stage1_result")),
        "honest_verdict": verdict_text(payload),
    }


def hardware_continuity(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    if was_skipped:
        return {"status": "skipped_flagged_adversarial", "kv260_terminal_confirmed": False}
    if not isinstance(payload, Mapping):
        return {"status": "missing", "kv260_terminal_confirmed": False}
    return {
        "status": "included",
        "kv260_terminal_confirmed": bool_metric(payload, "kv260_terminal_confirmed") is True,
        "gatemate_step_taken": str_metric(payload, "gatemate_step_taken"),
        "polarfire_step_taken": str_metric(payload, "polarfire_step_taken"),
        "honest_verdict": verdict_text(payload),
    }


def verifier_moat_status(moat: Mapping[str, Any]) -> str:
    status = str(moat.get("status") or "")
    return status if status in VERIFIER_MOAT_STATUSES else "DEFERRED-no-headroom"


def gap3_stage1_status(gap3: Mapping[str, Any]) -> str:
    status = str(gap3.get("status") or "")
    return status if status in GAP3_STAGE1_STATUSES else "BOUNDED"


def diffusiongemma_gate_status(moat_status: str) -> str:
    if moat_status == "PROVEN-headroom-present":
        return "MET"
    if moat_status == "BOUNDED-clean-null":
        return "STILL-PENDING-headroom-present-null"
    return "STILL-PENDING-no-headroom"


def headline_outcome(moat_status: str, gap3_status: str, *, moat_available: bool) -> str:
    if moat_status == "PROVEN-headroom-present":
        return "moat_proven_headroom_present"
    if moat_status == "BOUNDED-clean-null":
        return "moat_bounded_clean_null"
    if gap3_status == "REACHES-headroom":
        return "gap3_reaches_headroom"
    if not moat_available:
        return "gap3_bounded"
    return "moat_deferred_no_headroom"


def honest_verdict(
    outcome: str,
    moat_status: str,
    gap3_status: str,
    gate_status: str,
    total_levels: int,
    flagged_for_v388: str,
    skipped_count: int,
) -> str:
    return (
        f"complete: capstone_v387_{outcome}_"
        f"moat_{moat_status}_gap3_{gap3_status}_"
        f"diffusiongemma_{gate_status}_arc_levels{total_levels}_"
        f"sota_{flagged_for_v388 or 'missing'}_flagged_skipped{skipped_count}"
    )


def headline_answers(
    moat: Mapping[str, Any],
    gap3: Mapping[str, Any],
    arc: Mapping[str, Any],
    sota: Mapping[str, Any],
) -> JsonDict:
    return {
        "headroom_controlled_moat_verifier_value_added": moat.get("verifier_value_added") is True,
        "headroom_controlled_moat_positive_control_confirmed": (
            moat.get("positive_control_confirmed") is True
        ),
        "headroom_controlled_moat_domain": str(moat.get("domain") or ""),
        "gap3_stage1_reaches_13pp_headroom": gap3.get("reaches_proven_arc_headroom") is True,
        "gap3_pass2_energy_vs_vote": gap3.get("pass2_energy_vs_vote", 0.0),
        "gap3_all_four_gates_pass": gap3.get("all_four_gates_pass") is True,
        "total_arc_levels_solved": arc.get("total_arc_levels_solved", 0),
        "strongest_sota_method_flagged_for_v388": str(sota.get("flagged_for_v388") or ""),
    }


def imported_fields_by_id(clean_ids: set[int]) -> dict[int, list[str]]:
    fields: dict[int, list[str]] = {experiment_id: [] for experiment_id in UPSTREAM_IDS}
    if 4175 in clean_ids:
        fields[4175] = [
            "headroom_present_domain",
            "max_selectable_headroom",
            "per_domain_headroom",
        ]
    if 4177 in clean_ids:
        fields[4177] = [
            "domain",
            "verifier_value_added",
            "positive_control_confirmed",
            "moat_delta_vs_vote",
            "moat_vs_matched_control",
            "accuracy_cost_pareto",
        ]
    if 4178 in clean_ids:
        fields[4178] = [
            "pass2_energy_vs_vote",
            "all_four_gates_pass",
            "gates",
            "headroom_capture_fraction",
            "candidate_auroc",
        ]
    if 4179 in clean_ids:
        fields[4179] = [
            "total_levels_solved",
            "total_games_solved",
            "new_levels_solved_this_task",
            "real_env_confirmed",
        ]
    if 4180 in clean_ids:
        fields[4180] = ["flagged_for_v388"]
    if 4181 in clean_ids:
        fields[4181] = ["regression_guard_passed", "moat_verdict", "gap3_stage1_result"]
    if 4182 in clean_ids:
        fields[4182] = [
            "kv260_terminal_confirmed",
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

    census = headroom_census(clean.get(4175), was_skipped=4175 in skipped_ids)
    moat = moat_verdict(clean.get(4177), census, was_skipped=4177 in skipped_ids)
    gap3 = gap3_stage1(clean.get(4178), was_skipped=4178 in skipped_ids)
    arc = arc_progress(clean.get(4179), was_skipped=4179 in skipped_ids)
    sota = sota_v388(clean.get(4180), was_skipped=4180 in skipped_ids)
    registry = registry_gap_hygiene(clean.get(4181), was_skipped=4181 in skipped_ids)
    hardware = hardware_continuity(clean.get(4182), was_skipped=4182 in skipped_ids)

    moat_status = verifier_moat_status(moat)
    gap_status = gap3_stage1_status(gap3)
    gate_status = diffusiongemma_gate_status(moat_status)
    outcome = headline_outcome(
        moat_status,
        gap_status,
        moat_available=4177 in clean_ids,
    )
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_ids)
    fields_by_id = imported_fields_by_id(clean_ids)
    total_levels = int(arc["total_arc_levels_solved"])
    flagged_for_v388 = str(sota.get("flagged_for_v388") or "")

    artifact: JsonDict = {
        "schema": "carnot.capstone_v387_4183.v1",
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict(
            outcome,
            moat_status,
            gap_status,
            gate_status,
            total_levels,
            flagged_for_v388,
            len(skipped),
        ),
        "headline_outcome": outcome,
        "headline_answers": headline_answers(moat, gap3, arc, sota),
        "verifier_moat_status": moat_status,
        "verifier_moat": moat,
        "gap3_stage1_status": gap_status,
        "gap3_stage1": gap3,
        "diffusiongemma_gate_status": gate_status,
        "diffusiongemma_gate": {
            "status": gate_status,
            "basis": "clean_exp4177_positive_controlled_executable_headroom_moat",
            "met": gate_status == "MET",
        },
        "arc_progress": arc,
        "total_arc_levels_solved": total_levels,
        "sota_v388": sota,
        "strongest_sota_method_flagged_for_v388": flagged_for_v388,
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
    if artifact.get("verifier_moat_status") not in VERIFIER_MOAT_STATUSES:
        raise ValueError("verifier moat status must be enumerated")
    if artifact.get("gap3_stage1_status") not in GAP3_STAGE1_STATUSES:
        raise ValueError("GAP-3 Stage-1 status must be enumerated")
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
