"""Build the Exp 4114 v380 capstone aggregation.

Spec refs: REQ-CAPSTONE-4114, SCENARIO-CAPSTONE-4114.

This module is a small reporting layer. It does not rerun training, Sudoku, ARC,
or hardware work. It reads the landed .380 upstream artifacts, excludes any
artifact marked adversarial before importing metrics, and writes one audited
headline artifact with sha256 provenance for every upstream file.
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
OUTPUT_REL_PATH = Path("results/experiment_4114_capstone_v380.json")
EXPERIMENT_ID = 4114
RANDOM_SEED = 4114
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

UPSTREAM_IDS = (4107, 4108, 4109, 4110, 4111, 4112, 4113)
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    4107: Path("results/experiment_4107_nanotrm_mechanism_smoke.json"),
    4108: Path("results/experiment_4108_nanotrm_sudoku_extreme_baseline.json"),
    4109: Path("results/experiment_4109_carnot_verifier_graft_sudoku.json"),
    4110: Path("results/experiment_4110_twelfth_game_explore_first.json"),
    4111: Path("results/experiment_4111_sota_ingestion_trm_verifier_training.json"),
    4112: Path("results/experiment_4112_verifier_registry_gaps_hygiene.json"),
    4113: Path("results/experiment_4113_hardware_continuity.json"),
}

HEADLINE_OUTCOMES = {
    "verifier_as_reward_validated_on_executable_domain",
    "honest_null_verifier_no_added_value",
    "baseline_reproduced_graft_inconclusive",
    "mechanism_still_blocked",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "upstream_provenance",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. The milestone's decision-grade summary; an honest null "
        "is COMPLETE and valuable."
    ),
    "headline_outcome": (
        "One of the enumerated set — forces a single unambiguous read of the .380 result."
    ),
    "upstream_provenance": (
        "{experiment_id, fields_imported, sha256} per cited upstream — the audit trail "
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
    """Hash an upstream artifact so the result can be audited later."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_to_root(root: Path, path: Path) -> str:
    """Return stable repository-relative paths in the artifact."""

    try:
        return str(path.relative_to(root))
    except ValueError:  # pragma: no cover
        return str(path)


def selected_upstream_paths(root: Path | str) -> dict[int, Path | None]:
    """Resolve the intended .380 upstream artifact paths."""

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


def _ci_excludes_zero_positive(ci: list[float]) -> bool:
    return len(ci) == 2 and ci[0] > 0.0 and ci[1] > 0.0


def _nested_map(payload: Mapping[str, Any] | None, field: str) -> Mapping[str, Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def mechanism_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Answer whether Exp 4107 de-risked the native nano-trm trainer."""

    if was_skipped:
        status = "skipped_flagged_adversarial"
    elif not isinstance(payload, Mapping):
        status = "missing"
    else:
        status = "derisked" if bool_metric(payload, "nanotrm_trainer_checkpoint_ok") else "blocked"
    exact_accuracy = float_metric(payload, "exact_accuracy")
    checkpoint_ok = status == "derisked"
    return {
        "status": status,
        "derisked": checkpoint_ok and exact_accuracy > 0.0,
        "checkpoint_ok": checkpoint_ok,
        "exact_accuracy": exact_accuracy if isinstance(payload, Mapping) and not was_skipped else None,
        "exact_accuracy_metric": str_metric(payload, "exact_accuracy_metric") if not was_skipped else "",
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def baseline_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Answer whether Exp 4108 reproduced the published approximate 0.87 baseline."""

    if was_skipped:
        status = "skipped_flagged_adversarial"
    elif not isinstance(payload, Mapping):
        status = "missing"
    elif bool_metric(payload, "matches_published_087"):
        status = "baseline_reproduced"
    else:
        status = "baseline_not_reproduced"
    reproduced = status == "baseline_reproduced"
    return {
        "status": status,
        "published_087_baseline_reproduced": reproduced,
        "mechanism_checkpoint_ok": bool_metric(payload, "mechanism_checkpoint_ok") if not was_skipped else False,
        "checkpoint_reload_ok": bool_metric(payload, "checkpoint_reload_ok") if not was_skipped else False,
        "reproduced_exact_accuracy": float_metric(payload, "reproduced_exact_accuracy")
        if isinstance(payload, Mapping) and not was_skipped
        else None,
        "published_exact_accuracy_target": float_metric(payload, "published_exact_accuracy_target")
        if isinstance(payload, Mapping) and not was_skipped
        else None,
        "published_match_tolerance": float_metric(payload, "published_match_tolerance")
        if isinstance(payload, Mapping) and not was_skipped
        else None,
        "return_code": int_metric(payload, "return_code") if not was_skipped else 0,
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def graft_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Answer whether a clean Exp 4109 verifier graft beat the vote ablation."""

    if was_skipped:
        return {
            "status": "skipped_flagged_adversarial",
            "beat_vote_ablation": False,
            "verifier_value_added": None,
            "a_vs_b_delta": None,
            "a_vs_b_ci95": None,
            "a_vs_b_status": "",
            "rerank_lift_vs_vote": None,
            "honest_verdict": "",
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "missing",
            "beat_vote_ablation": False,
            "verifier_value_added": None,
            "a_vs_b_delta": None,
            "a_vs_b_ci95": None,
            "a_vs_b_status": "",
            "rerank_lift_vs_vote": None,
            "honest_verdict": "",
        }

    delta_payload = _nested_map(payload, "rft_vs_ablation_delta")
    rerank_payload = _nested_map(payload, "rerank_lift_vs_vote")
    ci = list_float_metric(delta_payload, "ci95")
    value_added = bool_metric(payload, "verifier_value_added")
    beat = value_added and _ci_excludes_zero_positive(ci)
    return {
        "status": "verifier_value_added" if beat else "honest_null_or_inconclusive",
        "beat_vote_ablation": beat,
        "verifier_value_added": value_added,
        "a_vs_b_delta": float_metric(delta_payload, "delta"),
        "a_vs_b_ci95": ci,
        "a_vs_b_status": str_metric(delta_payload, "status"),
        "rerank_lift_vs_vote": {
            "delta": float_metric(rerank_payload, "delta"),
            "ci95": list_float_metric(rerank_payload, "ci95"),
            "metric": str_metric(rerank_payload, "metric"),
        },
        "honest_verdict": verdict_text(payload),
    }


def arc_games_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Carry the clean Exp 4110 ARC games-solved count."""

    if was_skipped:
        status = "skipped_flagged_adversarial"
    elif not isinstance(payload, Mapping):
        status = "missing"
    elif bool_metric(payload, "game_solved") and bool_metric(payload, "real_env_confirmed"):
        status = "new_game_solved"
    else:
        status = "measured_no_new_solve"
    return {
        "status": status,
        "prior_total_games_solved": int_metric(payload, "prior_total_games_solved") if not was_skipped else 0,
        "total_games_solved": int_metric(payload, "total_games_solved") if not was_skipped else 0,
        "game_solved": bool_metric(payload, "game_solved") if not was_skipped else False,
        "real_env_confirmed": bool_metric(payload, "real_env_confirmed") if not was_skipped else False,
        "target_game": str_metric(payload, "target_game") if not was_skipped else "",
        "first_solve_at_action": int_metric(payload, "first_solve_at_action") if not was_skipped else 0,
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def sota_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Summarize clean Exp 4111 next-method ingestion without affecting the headline."""

    methods = payload.get("methods_mapped") if isinstance(payload, Mapping) and not was_skipped else None
    method_rows = [dict(item) for item in methods if isinstance(item, Mapping)] if isinstance(methods, list) else []
    return {
        "included": isinstance(payload, Mapping) and not was_skipped,
        "flagged_for_v381": str_metric(payload, "flagged_for_v381") if not was_skipped else "",
        "methods_mapped": [
            {"name": str_metric(row, "name"), "arxiv_id": str_metric(row, "arxiv_id")}
            for row in method_rows
        ],
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def registry_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Summarize clean Exp 4112 registry/gap hygiene."""

    gaps = payload.get("gaps_updated") if isinstance(payload, Mapping) and not was_skipped else None
    return {
        "included": isinstance(payload, Mapping) and not was_skipped,
        "gaps_updated": [item for item in gaps if isinstance(item, str)] if isinstance(gaps, list) else [],
        "regression_guard_passed": bool_metric(payload, "regression_guard_passed") if not was_skipped else False,
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def hardware_answer(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Carry clean Exp 4113 hardware continuity separately from science claims."""

    reachability = (
        payload.get("per_board_reachability")
        if isinstance(payload, Mapping) and not was_skipped
        else None
    )
    return {
        "included": isinstance(payload, Mapping) and not was_skipped,
        "kv260_terminal_confirmed": bool_metric(payload, "kv260_terminal_confirmed") if not was_skipped else False,
        "per_board_reachability": dict(reachability) if isinstance(reachability, Mapping) else {},
        "gatemate_step_taken": str_metric(payload, "gatemate_step_taken") if not was_skipped else "",
        "polarfire_step_taken": str_metric(payload, "polarfire_step_taken") if not was_skipped else "",
        "honest_verdict": verdict_text(payload) if not was_skipped else "",
    }


def headline_outcome(
    mechanism: Mapping[str, Any],
    baseline: Mapping[str, Any],
    graft: Mapping[str, Any],
) -> str:
    """Choose the single enumerated outcome required by .380."""

    if mechanism.get("derisked") is not True:
        return "mechanism_still_blocked"
    if (
        baseline.get("published_087_baseline_reproduced") is True
        and graft.get("beat_vote_ablation") is True
    ):
        return "verifier_as_reward_validated_on_executable_domain"
    if baseline.get("published_087_baseline_reproduced") is True and graft.get("status") in {
        "missing",
        "skipped_flagged_adversarial",
    }:
        return "baseline_reproduced_graft_inconclusive"
    return "honest_null_verifier_no_added_value"


def headline_answers(
    mechanism: Mapping[str, Any],
    baseline: Mapping[str, Any],
    graft: Mapping[str, Any],
    arc_games: Mapping[str, Any],
) -> JsonDict:
    """Expose the four concrete question answers in machine-checkable fields."""

    return {
        "nanotrm_trainer_mechanism_derisked": mechanism.get("derisked") is True,
        "published_087_baseline_reproduced": baseline.get("published_087_baseline_reproduced")
        is True,
        "carnot_verifier_graft_beat_vote_ablation_on_sudoku": graft.get("beat_vote_ablation")
        is True,
        "carnot_verifier_graft_evidence_status": str(graft.get("status", "")),
        "total_arc_games_solved": int(arc_games.get("total_games_solved", 0)),
    }


def verdict(
    outcome: str,
    mechanism: Mapping[str, Any],
    baseline: Mapping[str, Any],
    graft: Mapping[str, Any],
    games_solved_total: int,
    skipped_count: int,
) -> str:
    """Build a terminal-prefix headline from the already-chosen outcome."""

    prefix = (
        "success:"
        if outcome == "verifier_as_reward_validated_on_executable_domain"
        else "blocked:"
        if outcome == "mechanism_still_blocked"
        else "complete:"
    )
    mechanism_flag = int(mechanism.get("derisked") is True)
    baseline_flag = int(baseline.get("published_087_baseline_reproduced") is True)
    graft_status = str(graft.get("status", "missing")) or "missing"
    return (
        f"{prefix} capstone_v380_{outcome}_mechanism_derisked{mechanism_flag}_"
        f"baseline087{baseline_flag}_graft_{graft_status}_games{games_solved_total}_"
        f"flagged_skipped{skipped_count}"
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
                "reason": "flagged_adversarial:true" if flagged(upstreams[experiment_id]) else "unknown",
                "sha256": sha256_file(path) if path is not None else "",
            }
        )
    return rows


def imported_fields_by_id(clean_ids: set[int]) -> dict[int, list[str]]:
    """Name exactly which fields each clean upstream contributes."""

    fields: dict[int, list[str]] = {experiment_id: [] for experiment_id in UPSTREAM_IDS}
    if 4107 in clean_ids:
        fields[4107] = [
            "nanotrm_trainer_checkpoint_ok",
            "exact_accuracy",
            "exact_accuracy_metric",
        ]
    if 4108 in clean_ids:
        fields[4108] = [
            "mechanism_checkpoint_ok",
            "checkpoint_reload_ok",
            "reproduced_exact_accuracy",
            "published_exact_accuracy_target",
            "published_match_tolerance",
            "matches_published_087",
            "return_code",
        ]
    if 4109 in clean_ids:
        fields[4109] = [
            "verifier_value_added",
            "rft_vs_ablation_delta.delta",
            "rft_vs_ablation_delta.ci95",
            "rft_vs_ablation_delta.status",
            "rerank_lift_vs_vote.delta",
            "rerank_lift_vs_vote.ci95",
        ]
    if 4110 in clean_ids:
        fields[4110] = [
            "prior_total_games_solved",
            "total_games_solved",
            "game_solved",
            "real_env_confirmed",
            "target_game",
            "first_solve_at_action",
        ]
    if 4111 in clean_ids:
        fields[4111] = ["flagged_for_v381", "methods_mapped"]
    if 4112 in clean_ids:
        fields[4112] = ["gaps_updated", "regression_guard_passed"]
    if 4113 in clean_ids:
        fields[4113] = [
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
    """Build the .380 capstone from landed upstream artifacts."""

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

    mechanism = mechanism_answer(clean_upstreams.get(4107), was_skipped=4107 in skipped_ids)
    baseline = baseline_answer(clean_upstreams.get(4108), was_skipped=4108 in skipped_ids)
    graft = graft_answer(clean_upstreams.get(4109), was_skipped=4109 in skipped_ids)
    games = arc_games_answer(clean_upstreams.get(4110), was_skipped=4110 in skipped_ids)
    sota = sota_answer(clean_upstreams.get(4111), was_skipped=4111 in skipped_ids)
    registry = registry_answer(clean_upstreams.get(4112), was_skipped=4112 in skipped_ids)
    hardware = hardware_answer(clean_upstreams.get(4113), was_skipped=4113 in skipped_ids)
    outcome = headline_outcome(mechanism, baseline, graft)
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_ids)
    total_games = int(games["total_games_solved"])
    fields_by_id = imported_fields_by_id(clean_ids)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v380_4114.v1",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": verdict(outcome, mechanism, baseline, graft, total_games, len(skipped)),
        "headline_outcome": outcome,
        "headline_answers": headline_answers(mechanism, baseline, graft, games),
        "nanotrm_trainer_mechanism": mechanism,
        "published_baseline_reproduction": baseline,
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
    """Validate the fields that keep the .380 headline auditable."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover
    if "gated_on" in artifact:
        raise ValueError("capstone must not emit gated_on")  # pragma: no cover
    verdict_value = str(artifact.get("honest_verdict", ""))
    if not verdict_value.startswith(("complete:", "success:", "blocked_", "blocked:")):
        raise ValueError("honest_verdict must use a terminal prefix")  # pragma: no cover
    if artifact.get("headline_outcome") not in HEADLINE_OUTCOMES:
        raise ValueError("headline_outcome must be one of the enumerated values")
    if not isinstance(artifact.get("total_arc_games_solved"), int) or isinstance(
        artifact.get("total_arc_games_solved"), bool
    ):
        raise ValueError("total_arc_games_solved must be a bare int")  # pragma: no cover
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")  # pragma: no cover
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
            raise ValueError("upstream_provenance entries need integer experiment_id")  # pragma: no cover
        if not isinstance(row.get("fields_imported"), list) or not all(
            isinstance(item, str) for item in row.get("fields_imported", [])
        ):
            raise ValueError("upstream_provenance fields_imported must be strings")  # pragma: no cover
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
    """Build, validate, and write the Exp 4114 capstone artifact."""

    root_path = Path(root)
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    validate_artifact(artifact)
    output = root_path / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
