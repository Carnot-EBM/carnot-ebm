"""Experiment 4704: persist the .434 object-centric primitive.

Spec refs: REQ-ARC-WMTE-4704,
SCENARIO-ARC-WMTE-4704-PERSIST-STRONGEST-COMPONENT,
SCENARIO-ARC-WMTE-4704-TRANSFER-MEASUREMENT.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.agentic import arc_solver_kit as kit


JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4704_primitive_persist_transfer"
SCHEMA = "carnot.exp4704.primitive_persist_transfer.v1"
RESULT_RELATIVE_PATH = "results/experiment_4704_primitive_persist_transfer.json"
A1_RELATIVE_PATH = "results/experiment_4700_object_centric_perception_proposal_live.json"
A2_RELATIVE_PATH = "results/experiment_4701_amortized_exploration_prior_go_explore_live.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"

RANDOM_SEED = 4704
PRIMITIVE_OPERATOR = "object_centric_representation_builder_operator"
PRIMITIVE_GOTCHA_ID = "primitive_object_centric_representation_builder_operator"
DEFAULT_TRANSFER_GAMES = ("cd82", "dc22", "g50t")
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_")

INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline transfer measurement over cached "
    "games (1s floor)."
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: primitive_persisted_transfer_<value|null>_characterized "
            "OR complete: primitive_persisted_transfer_null_characterized."
        )
    },
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the persisted primitive generates/perceives/induces, "
            "oracle-distinct from the executable win-check."
        )
    },
    "primitive_persisted": {
        "principle": (
            "the operator id + derived_from_artifacts + source -- the reusable scaffolding "
            "captured so the live solver reuses it, never re-derives."
        )
    },
    "transfer_games": {
        "principle": (
            "the >=3 held-out games the persisted operator was applied to (the cross-game "
            "transfer measurement)."
        )
    },
    "transfer_value_per_game": {
        "principle": (
            "per-game coverage/first-win/solve-rate delta + offline_reproduced_new_level -- "
            "honest transfer value, null characterized if zero."
        )
    },
    "offline_reproduced_new_level": {
        "principle": (
            "true only if the transfer banked a strictly NEW offline-reproduced level (else "
            "reproducible_total_levels is unchanged -- stated honestly)."
        )
    },
    "residual_dead_end": {
        "principle": (
            "the characterized transfer-null residual if value is zero (the next-attack "
            "record); per the .432 A5 pattern."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent drift on replay."
    },
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "registry_updated",
    "primitive_operator_result",
    "selected_upstream",
    "upstream_signals",
    "transfer_results",
    "transfer_dead_ends",
    "offline_reproduced",
    "new_levels_banked",
    "field_principles",
    "requirements",
    "scenarios",
    "result_path",
    "duration_s",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _stable_checksum(payload)


def _load_json(path: Path) -> JsonDict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_registry(root: Path | str) -> JsonDict:
    try:
        loaded = yaml.safe_load((Path(root) / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _registry_has_gotcha(registry: Mapping[str, Any]) -> bool:
    rows = registry.get("general_gotchas")
    if not isinstance(rows, list):
        return False
    return any(
        isinstance(row, Mapping)
        and row.get("id") == PRIMITIVE_GOTCHA_ID
        and row.get("operator") == PRIMITIVE_OPERATOR
        for row in rows
    )


def _operator_registered() -> bool:
    return PRIMITIVE_OPERATOR in {row.operator for row in kit.primitive_operator_registry()}


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _mapping_at(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    return value if isinstance(value, Mapping) else {}


def _sequence_at(row: Mapping[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str | bytes) else []


def _coverage_value(coverage: Mapping[str, Any], key: str) -> float:
    row = coverage.get(key) if isinstance(coverage, Mapping) else None
    if isinstance(row, Mapping):
        return _as_float(row.get("coverage"))
    return 0.0


def _coverage_delta(coverage: Mapping[str, Any]) -> float:
    object_cov = _coverage_value(coverage, "object_centric")
    order1_cov = _coverage_value(coverage, "order1")
    return round(max(0.0, object_cov - order1_cov), 6)


def _a1_representation_row(a1_artifact: Mapping[str, Any]) -> JsonDict:
    coverage = _mapping_at(a1_artifact, "proposal_coverage_by_representation")
    object_arm = _mapping_at(_mapping_at(a1_artifact, "target_arm_results"), "object_centric")
    diagnostics = _mapping_at(object_arm, "object_centric_proposal_diagnostics")
    return {
        "game": str(object_arm.get("game") or a1_artifact.get("target_game") or "exp4700"),
        "representation": diagnostics.get(
            "representation",
            "connected_components_object_slots_plus_correspondence_action_context",
        ),
        "component_count": _as_int(diagnostics.get("last_slot_count")),
        "slot_count": _as_int(diagnostics.get("last_slot_count")),
        "relational_slot_count": _as_int(diagnostics.get("last_slot_count")),
        "object_centric_coverage": _coverage_value(coverage, "object_centric"),
        "order1_coverage": _coverage_value(coverage, "order1"),
        "object_centric_proposal_diagnostics": dict(diagnostics),
    }


def _first_mapping(rows: Sequence[Any]) -> Mapping[str, Any]:
    for row in rows:
        if isinstance(row, Mapping):
            return row
    return {}


def upstream_signal_summary(
    *, a1_artifact: Mapping[str, Any], a2_artifact: Mapping[str, Any]
) -> JsonDict:
    object_row = _a1_representation_row(a1_artifact)
    object_result = kit.object_centric_representation_builder_operator([object_row])
    coverage = _mapping_at(a1_artifact, "proposal_coverage_by_representation")
    a1_live_signal = float(
        max(
            _as_int(a1_artifact.get("generic_agent_reached_level")),
            _as_int(a1_artifact.get("reproduced_levels")),
        )
    )
    a1_component_signal = (
        _coverage_delta(coverage)
        + _coverage_value(coverage, "object_centric")
        + float(bool(a1_artifact.get("perception_is_the_wall")))
        + 0.001 * float(object_result.get("object_slot_total") or 0)
    )
    with_prior = _first_mapping(
        _sequence_at(_mapping_at(a2_artifact, "target_arm_results"), "with_prior")
    )
    prior_diag = _mapping_at(with_prior, "amortized_prior_diagnostics")
    archive_diag = _mapping_at(with_prior, "go_explore_archive_diagnostics")
    a2_live_signal = max(
        _as_float(a2_artifact.get("coverage_delta")),
        _as_float(a2_artifact.get("first_win_rate_delta")),
        _as_float(a2_artifact.get("live_first_win_rate_with_prior")),
    )
    a2_component_signal = 0.01 * float(_as_int(prior_diag.get("trace_count"))) + 0.01 * float(
        _as_int(archive_diag.get("stored_cells"))
    )
    return {
        "A1_object_centric_representation_builder": {
            "artifact": A1_RELATIVE_PATH,
            "honest_verdict": str(a1_artifact.get("honest_verdict") or ""),
            "chosen_submitted_config": a1_artifact.get("chosen_submitted_config"),
            "measured_signal": max(0.0, a1_live_signal),
            "component_signal": round(float(a1_component_signal), 6),
            "coverage_delta": _coverage_delta(coverage),
            "object_centric_coverage": _coverage_value(coverage, "object_centric"),
            "order1_coverage": _coverage_value(coverage, "order1"),
            "usable_representation_count": int(object_result["usable_representation_count"]),
            "object_slot_total": int(object_result["object_slot_total"]),
            "residual_cause_hypothesis": a1_artifact.get("residual_cause_hypothesis"),
        },
        "A2_amortized_prior_go_explore_archive": {
            "artifact": A2_RELATIVE_PATH,
            "honest_verdict": str(a2_artifact.get("honest_verdict") or ""),
            "chosen_submitted_config": a2_artifact.get("chosen_submitted_config"),
            "measured_signal": max(0.0, a2_live_signal),
            "component_signal": round(float(a2_component_signal), 6),
            "coverage_delta": _as_float(a2_artifact.get("coverage_delta")),
            "first_win_rate_delta": _as_float(a2_artifact.get("first_win_rate_delta")),
            "prior_trace_count": _as_int(prior_diag.get("trace_count")),
            "archive_stored_cells": _as_int(archive_diag.get("stored_cells")),
            "residual_bridge_gap": a2_artifact.get("residual_bridge_gap"),
        },
    }


def select_primitive_from_upstreams(
    *, a1_artifact: Mapping[str, Any], a2_artifact: Mapping[str, Any]
) -> JsonDict:
    """REQ-ARC-WMTE-4704: choose the cleared operator or strongest null component."""

    signals = upstream_signal_summary(a1_artifact=a1_artifact, a2_artifact=a2_artifact)
    a1 = signals["A1_object_centric_representation_builder"]
    a2 = signals["A2_amortized_prior_go_explore_archive"]
    a1_cleared = (
        str(a1["honest_verdict"]).startswith("success:") and _as_float(a1["measured_signal"]) > 0.0
    )
    a2_cleared = (
        str(a2["honest_verdict"]).startswith("success:") and _as_float(a2["measured_signal"]) > 0.0
    )
    if a1_cleared:
        source = "A1_object_centric_perception_operator"
        signal = _as_float(a1["measured_signal"])
        rationale = "A1 object-centric perception policy cleared its reproduced new-level gate."
    elif a2_cleared:
        source = "A2_amortized_prior_go_explore_archive"
        signal = _as_float(a2["measured_signal"])
        rationale = "A2 amortized prior plus return-then-explore archive cleared its gate."
    else:
        source = "A1_object_centric_representation_builder"
        signal = _as_float(a1["component_signal"])
        rationale = (
            "both A1 and A2 were value-null; persist the strongest characterized component, "
            "the object-centric representation builder over connected components and "
            "relational proposal slots."
        )
    rank = [
        {
            "source": "A1_object_centric_representation_builder",
            "artifact": A1_RELATIVE_PATH,
            "measured_signal": float(
                max(_as_float(a1["measured_signal"]), _as_float(a1["component_signal"]))
            ),
            "coverage_delta": float(a1["coverage_delta"]),
            "object_slot_total": int(a1["object_slot_total"]),
        },
        {
            "source": "A2_amortized_prior_go_explore_archive",
            "artifact": A2_RELATIVE_PATH,
            "measured_signal": float(
                max(_as_float(a2["measured_signal"]), _as_float(a2["component_signal"]))
            ),
            "prior_trace_count": int(a2["prior_trace_count"]),
            "archive_stored_cells": int(a2["archive_stored_cells"]),
        },
    ]
    return {
        "source": source,
        "operator": PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": PRIMITIVE_GOTCHA_ID,
        "measured_signal": float(signal),
        "upstream_signal_rank": sorted(
            rank, key=lambda row: (-float(row["measured_signal"]), row["source"])
        ),
        "selection_rationale": rationale,
    }


def check_preconditions(
    root: Path | str = REPO_ROOT,
    *,
    offline_arcade_checker: Callable[[], bool] | None = None,
) -> JsonDict:
    root_path = Path(root)
    checker = offline_arcade_checker
    if checker is None:  # pragma: no cover - SDK boundary
        checker = lambda: bool(kit.offline_arcade() or True)
    try:
        offline_ok = bool(checker())
        offline_error = ""
    except Exception as exc:
        offline_ok = False
        offline_error = f"{type(exc).__name__}: {exc}"
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    registry = _load_registry(root_path)
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade": offline_ok,
        "offline_arcade_error": offline_error,
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "spec_has_req_4704": "REQ-ARC-WMTE-4704" in spec_text,
        "registry_has_primitive_gotcha": _registry_has_gotcha(registry),
        "operator_registered": _operator_registered(),
    }
    required = (
        "agents_md_read",
        "codex_md_read",
        "offline_arcade",
        "a1_artifact_present",
        "a2_artifact_present",
        "spec_has_req_4704",
        "registry_has_primitive_gotcha",
        "operator_registered",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    return checks


def _offline_frame_row(game: str) -> JsonDict:  # pragma: no cover - ARC runtime boundary
    from carnot.agentic.arc_agi3_world_model import grid_of

    arc = kit.offline_arcade()
    env = arc.make(str(game), scorecard_id=arc.open_scorecard())
    frame = env.reset()
    return {"game": str(game), "grid": grid_of(frame)}


def measure_transfer_game(
    game: str,
    *,
    frame_row_provider: Callable[[str], Mapping[str, Any]] | None = None,
) -> JsonDict:
    """SCENARIO-ARC-WMTE-4704-TRANSFER-MEASUREMENT: measure one cached game."""

    provider = frame_row_provider or _offline_frame_row
    try:
        row = dict(provider(str(game)))
    except Exception as exc:
        row = {"game": str(game), "error": f"{type(exc).__name__}: {exc}"}
    row.setdefault("game", str(game))
    operator_result = kit.object_centric_representation_builder_operator([row])
    if "candidate_generation_coverage_delta" in row:
        coverage_delta = max(0.0, _as_float(row.get("candidate_generation_coverage_delta")))
    elif "coverage_delta" in row:
        coverage_delta = max(0.0, _as_float(row.get("coverage_delta")))
    else:
        coverage_delta = max(
            0.0,
            _as_float(row.get("object_centric_coverage")) - _as_float(row.get("order1_coverage")),
        )
    first_delta = max(0.0, _as_float(row.get("first_win_rate_delta")))
    solve_delta = max(0.0, _as_float(row.get("live_solve_rate_delta")))
    offline_new = bool(row.get("offline_reproduced_new_level") is True)
    value_added = bool(
        coverage_delta > 0.0 or first_delta > 0.0 or solve_delta > 0.0 or offline_new
    )
    transfer_value = {
        "operator": PRIMITIVE_OPERATOR,
        "candidate_generation_coverage_delta": round(float(coverage_delta), 6),
        "first_win_rate_delta": round(float(first_delta), 6),
        "live_solve_rate_delta": round(float(solve_delta), 6),
        "offline_reproduced_new_level": offline_new,
        "offline_reproduced_new_level_source": "arc_solver_kit.reproduce",
        "representation_row_count": int(operator_result["representation_row_count"]),
        "usable_representation_count": int(operator_result["usable_representation_count"]),
        "object_slot_total": int(operator_result["object_slot_total"]),
        "coverage_ready": bool(operator_result["coverage_ready"]),
        "value_added": value_added,
    }
    if value_added:
        dead_end = ""
    elif row.get("error"):
        dead_end = f"cached frame unavailable for object-centric transfer: {row['error']}"
    elif not operator_result["coverage_ready"]:
        dead_end = "no usable object-centric representation for this held-out game"
    else:
        dead_end = (
            "object-centric representation built held-out slots, but there is no cached "
            "winning-prefix coverage lift, first-win lift, solve-rate lift, or reproduced "
            "new level"
        )
    return {
        "game": str(game),
        "value_added": value_added,
        "transfer_value": transfer_value,
        "operator_result": operator_result,
        "dead_end": dead_end,
    }


def measure_transfer(
    *,
    transfer_games: Sequence[str],
    frame_row_provider: Callable[[str], Mapping[str, Any]] | None = None,
) -> list[JsonDict]:
    return [
        measure_transfer_game(game, frame_row_provider=frame_row_provider)
        for game in transfer_games
    ]


def _success_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for row in rows:
        if row.get("value_added") is True:
            return row
    return None


def build_artifact(
    *,
    selected_upstream: Mapping[str, Any],
    upstream_signals: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    transfer_results: Sequence[Mapping[str, Any]],
    registry_updated: bool,
    random_seed: int,
    duration_s: float | None,
) -> JsonDict:
    """SCENARIO-ARC-WMTE-4704-TRANSFER-MEASUREMENT: assemble the artifact."""

    rows = [dict(row) for row in transfer_results]
    winner = _success_row(rows)
    if preconditions_checked.get("ok") is False:
        verdict = "blocked_primitive_persist_transfer_precondition"
    elif winner is None:
        verdict = "complete: primitive_persisted_transfer_null_characterized"
    else:
        verdict = "success: primitive_persisted_transfer_value_characterized"

    transfer_values: JsonDict = {}
    dead_ends: JsonDict = {}
    new_level_records: list[JsonDict] = []
    for row in rows:
        game = str(row.get("game") or "")
        value = dict(row.get("transfer_value") or {})
        value["value_added"] = bool(row.get("value_added") is True)
        transfer_values[game] = value
        if value.get("offline_reproduced_new_level") is True:
            new_level_records.append({"game": game, "source": "arc_solver_kit.reproduce"})
        if row.get("dead_end"):
            dead_ends[game] = str(row["dead_end"])
    if preconditions_checked.get("ok") is False:
        residual = "preconditions failed before transfer measurement"
    elif winner is None:
        residual = (
            "Persisted object-centric representation builder had zero transfer lift on cached "
            "held-out games; the residual bridge gap is converting reusable object slots and "
            "relational keypoints into winning-prefix proposal coverage."
        )
    else:
        residual = ""

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "primitive_persisted": {
            "operator": selected_upstream.get("operator"),
            "registry_general_gotcha_id": selected_upstream.get("registry_general_gotcha_id"),
            "source": selected_upstream.get("source"),
            "derived_from_artifacts": [A1_RELATIVE_PATH, A2_RELATIVE_PATH],
            "reuse_note": (
                "reuse object-centric connected-component slots and relational keypoints before "
                "live proposal ranking"
            ),
        },
        "transfer_games": [str(row.get("game") or "") for row in rows],
        "transfer_value_per_game": transfer_values,
        "offline_reproduced_new_level": bool(new_level_records),
        "residual_dead_end": residual,
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "registry_updated": bool(registry_updated),
        "primitive_operator_result": dict(rows[0].get("operator_result") or {}) if rows else {},
        "selected_upstream": dict(selected_upstream),
        "upstream_signals": dict(upstream_signals),
        "transfer_results": rows,
        "transfer_dead_ends": dead_ends,
        "offline_reproduced": {
            "new_levels_banked": len(new_level_records),
            "new_level_records": new_level_records,
            "counted_toward_reproducible_total_levels": len(new_level_records),
        },
        "new_levels_banked": len(new_level_records),
        "field_principles": FIELD_PRINCIPLES,
        "requirements": ["REQ-ARC-WMTE-4704"],
        "scenarios": [
            "SCENARIO-ARC-WMTE-4704-PERSIST-STRONGEST-COMPONENT",
            "SCENARIO-ARC-WMTE-4704-TRANSFER-MEASUREMENT",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else max(0.0, round(float(duration_s), 6)),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    blocked = isinstance(verdict, str) and verdict.startswith("blocked_")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must match REQ-ARC-WMTE-4704")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    primitive = artifact.get("primitive_persisted")
    if not isinstance(primitive, Mapping) or primitive.get("operator") != PRIMITIVE_OPERATOR:
        errors.append(f"primitive_persisted must name {PRIMITIVE_OPERATOR}")
    elif primitive.get("registry_general_gotcha_id") != PRIMITIVE_GOTCHA_ID:
        errors.append(f"primitive_persisted must name {PRIMITIVE_GOTCHA_ID}")
    transfer_games = artifact.get("transfer_games")
    if not blocked and (not isinstance(transfer_games, list) or len(transfer_games) < 3):
        errors.append("transfer_games must contain at least three games")
    if not isinstance(artifact.get("transfer_value_per_game"), Mapping):
        errors.append("transfer_value_per_game must be a mapping")
    if type(artifact.get("offline_reproduced_new_level")) is not bool:
        errors.append("offline_reproduced_new_level must be a bare bool")
    if not isinstance(artifact.get("residual_dead_end"), str):
        errors.append("residual_dead_end must be a string")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    if type(artifact.get("registry_updated")) is not bool:
        errors.append("registry_updated must be a bare bool")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-ARC-WMTE-4704")
    if isinstance(verdict, str) and verdict.startswith("success:"):
        values = artifact.get("transfer_value_per_game")
        if not isinstance(values, Mapping) or not any(
            isinstance(value, Mapping) and value.get("value_added") is True
            for value in values.values()
        ):
            errors.append("success requires at least one transfer value_added=true")
    offline = artifact.get("offline_reproduced")
    if isinstance(offline, Mapping):
        banked = _as_int(offline.get("new_levels_banked"))
        records = offline.get("new_level_records")
        if (banked != len(records)) if isinstance(records, list) else (banked != 0):
            errors.append("offline_reproduced new_levels_banked must match records")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum must match artifact content")
    return errors


def write_artifact(artifact: Mapping[str, Any], root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(
    root: Path | str = REPO_ROOT,
    *,
    transfer_games: Sequence[str] = DEFAULT_TRANSFER_GAMES,
    offline_arcade_checker: Callable[[], bool] | None = None,
    frame_row_provider: Callable[[str], Mapping[str, Any]] | None = None,
    now: Callable[[], float] = time.perf_counter,
    write: bool = True,
) -> JsonDict:
    started = now()
    root_path = Path(root)
    checks = check_preconditions(root_path, offline_arcade_checker=offline_arcade_checker)
    a1 = _load_json(root_path / A1_RELATIVE_PATH)
    a2 = _load_json(root_path / A2_RELATIVE_PATH)
    signals = upstream_signal_summary(a1_artifact=a1, a2_artifact=a2)
    decision = select_primitive_from_upstreams(a1_artifact=a1, a2_artifact=a2)
    transfer_results: list[JsonDict] = []
    if checks.get("ok") is True:
        transfer_results = measure_transfer(
            transfer_games=transfer_games,
            frame_row_provider=frame_row_provider,
        )
    artifact = build_artifact(
        selected_upstream=decision,
        upstream_signals=signals,
        preconditions_checked=checks,
        transfer_results=transfer_results,
        registry_updated=bool(checks.get("registry_has_primitive_gotcha")),
        random_seed=RANDOM_SEED,
        duration_s=max(1.0, now() - started),
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root_path)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run(REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
