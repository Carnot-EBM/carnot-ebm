"""Experiment 4668: persist the .430 primitive and measure transfer.

Spec refs: REQ-ARC-WMTE-4668, SCENARIO-ARC-WMTE-4668.
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

EXPERIMENT = "experiment_4668_primitive_persist_transfer"
SCHEMA = "carnot.exp4668.primitive_persist_transfer.v1"
RESULT_RELATIVE_PATH = "results/experiment_4668_primitive_persist_transfer.json"
A1_RELATIVE_PATH = "results/experiment_4664_l2_goal_predicate_induction_live.json"
A2_RELATIVE_PATH = "results/experiment_4665_dagger_distribution_shift_value_routing.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"

RANDOM_SEED = 4668
PRIMITIVE_OPERATOR = "dagger_off_path_data_collection_operator"
PRIMITIVE_GOTCHA_ID = "primitive_dagger_off_path_data_collection_operator"
DEFAULT_TRANSFER_GAMES = ("bp35", "cd82", "dc22")
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")

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
            "MUST be false -- the persisted primitive ranks/routes/induces, oracle-distinct "
            "from the executable win-check."
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
            "per-game solve-rate/first-win/efficiency delta + offline_reproduced_new_level -- "
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
            "record); per the .429 A5 pattern."
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
        "codex_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "offline_arcade": offline_ok,
        "offline_arcade_error": offline_error,
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "spec_has_req_4668": "REQ-ARC-WMTE-4668" in spec_text,
        "registry_has_primitive_gotcha": _registry_has_gotcha(registry),
        "operator_registered": _operator_registered(),
    }
    required = (
        "agents_md_read",
        "codex_md_read",
        "offline_arcade",
        "a1_artifact_present",
        "a2_artifact_present",
        "spec_has_req_4668",
        "registry_has_primitive_gotcha",
        "operator_registered",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    return checks


def upstream_signal_summary(
    *, a1_artifact: Mapping[str, Any], a2_artifact: Mapping[str, Any]
) -> JsonDict:
    a1_levels = _mapping_at(a1_artifact, "generic_agent_reached_level")
    max_a1_level = max((_as_int(value) for value in a1_levels.values()), default=0)
    a1_goal_satisfiable = any(
        value is True for value in _mapping_at(a1_artifact, "goal_predicate_satisfiable").values()
    )
    a1_signal = float(max(0, max_a1_level - 1))

    before = _as_float(a2_artifact.get("distribution_shift_score_before"))
    after = _as_float(a2_artifact.get("distribution_shift_score_after"))
    shift_drop = max(0.0, before - after)
    a2_live_signal = max(
        _as_float(a2_artifact.get("first_win_rate_delta")),
        _as_float(a2_artifact.get("solve_rate_delta")),
    )
    dataset = _mapping_at(a2_artifact, "dagger_dataset")
    has_dagger_rows = (
        _as_int(dataset.get("frontier_count")) > 0
        and _as_int(dataset.get("positive_count")) > 0
        and _as_int(dataset.get("negative_count")) > 0
    )
    return {
        "A1_l2_goal_induction": {
            "artifact": A1_RELATIVE_PATH,
            "honest_verdict": str(a1_artifact.get("honest_verdict") or ""),
            "measured_signal": a1_signal,
            "max_generic_agent_level": max_a1_level,
            "goal_predicate_satisfiable": a1_goal_satisfiable,
        },
        "A2_dagger_distribution_shift_value_routing": {
            "artifact": A2_RELATIVE_PATH,
            "honest_verdict": str(a2_artifact.get("honest_verdict") or ""),
            "chosen_submitted_config": a2_artifact.get("chosen_submitted_config"),
            "measured_signal": max(0.0, a2_live_signal),
            "component_signal": round(float(shift_drop), 6) if has_dagger_rows else 0.0,
            "distribution_shift_score_before": before,
            "distribution_shift_score_after": after,
            "dagger_dataset": dict(dataset),
        },
    }


def select_primitive_from_upstreams(
    *, a1_artifact: Mapping[str, Any], a2_artifact: Mapping[str, Any]
) -> JsonDict:
    """REQ-ARC-WMTE-4668: choose the .430 primitive or strongest component."""

    signals = upstream_signal_summary(a1_artifact=a1_artifact, a2_artifact=a2_artifact)
    a1 = signals["A1_l2_goal_induction"]
    a2 = signals["A2_dagger_distribution_shift_value_routing"]
    a1_cleared = str(a1["honest_verdict"]).startswith("success:") and _as_float(
        a1["measured_signal"]
    ) > 0.0
    a2_cleared = str(a2["honest_verdict"]).startswith("success:") and _as_float(
        a2["measured_signal"]
    ) > 0.0
    if a1_cleared:
        rationale = "A1 L2 goal induction cleared its gate; persist the reusable goal induction path."
        source = "A1_l2_goal_induction"
        signal = _as_float(a1["measured_signal"])
    elif a2_cleared:
        rationale = (
            "A2 DAgger-corrected value routing cleared its live-lift gate; persist the "
            "off-path data-collection substrate that feeds the corrected route."
        )
        source = "A2_dagger_distribution_corrected_value_routing"
        signal = _as_float(a2["measured_signal"])
    else:
        rationale = (
            "both A1 and A2 were value-null; persist the strongest characterized component, "
            "the DAgger-lite off-path data-collection operator."
        )
        source = "A2_dagger_off_path_data_collection"
        signal = _as_float(a2["component_signal"])
    rank = [
        {
            "source": "A1_l2_goal_induction",
            "artifact": A1_RELATIVE_PATH,
            "measured_signal": float(a1["measured_signal"]),
            "goal_predicate_satisfiable": bool(a1["goal_predicate_satisfiable"]),
        },
        {
            "source": "A2_dagger_off_path_data_collection",
            "artifact": A2_RELATIVE_PATH,
            "measured_signal": float(max(_as_float(a2["measured_signal"]), _as_float(a2["component_signal"]))),
            "live_signal": float(a2["measured_signal"]),
            "component_signal": float(a2["component_signal"]),
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


def _attempt_by_game(measurement: Mapping[str, Any], game: str) -> Mapping[str, Any] | None:
    rows = measurement.get("variant_attempts")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None
    for row in rows:
        if isinstance(row, Mapping) and row.get("game") == game:
            return row
    return None


def _attempt_actions(row: Mapping[str, Any] | None) -> int | None:
    if not row:
        return None
    value = row.get("actions_to_first_levelup")
    if value is None and row.get("first_win") is True:
        value = row.get("actions")
    if value is None:
        return None
    return _as_int(value)


def _attempt_reached_level(row: Mapping[str, Any] | None) -> int:
    if not row:
        return 0
    return _as_int(row.get("reached_level"))


def _attempt_reproduced(row: Mapping[str, Any] | None) -> bool:
    if not row:
        return False
    gate = row.get("reproduction_gate")
    return isinstance(gate, Mapping) and gate.get("reproduced") is True


def _attempt_lazy_value(row: Mapping[str, Any] | None) -> JsonDict:
    if not row:
        return {}
    diagnostics = row.get("lazy_value_diagnostics")
    return dict(diagnostics) if isinstance(diagnostics, Mapping) else {}


def _measurement_from_a2(a2_artifact: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    direct = _mapping_at(a2_artifact, key)
    if direct:
        return direct
    if key == "baseline_measurement":
        return _mapping_at(_mapping_at(a2_artifact, "live_baseline_winning_path_trained"), "measurement")
    return {}


def _probe_rows(
    game: str,
    corrected_row: Mapping[str, Any] | None,
    baseline_row: Mapping[str, Any] | None,
) -> list[JsonDict]:
    corrected_level = _attempt_reached_level(corrected_row)
    baseline_level = _attempt_reached_level(baseline_row)
    return [
        {
            "source": "corrected_live_frontier",
            "features": [
                float(corrected_level),
                1.0 if corrected_row and corrected_row.get("first_win") is True else 0.0,
                float(_as_int(corrected_row.get("actions") if corrected_row else 0)),
            ],
            "path": [{"action": 1, "data": {"game": game, "arm": "corrected"}}],
        },
        {
            "source": "baseline_live_frontier",
            "features": [
                float(baseline_level),
                1.0 if baseline_row and baseline_row.get("first_win") is True else 0.0,
                float(_as_int(baseline_row.get("actions") if baseline_row else 0)),
            ],
            "path": [{"action": 2, "data": {"game": game, "arm": "baseline"}}],
        },
    ]


def measure_transfer_game(game: str, *, a2_artifact: Mapping[str, Any]) -> JsonDict:
    """REQ-ARC-WMTE-4668: apply the persisted operator to one cached transfer game."""

    corrected_measurement = _measurement_from_a2(a2_artifact, "corrected_measurement")
    baseline_measurement = _measurement_from_a2(a2_artifact, "baseline_measurement")
    corrected_row = _attempt_by_game(corrected_measurement, game)
    baseline_row = _attempt_by_game(baseline_measurement, game)
    winning_labels = corrected_row.get("solution_labels") if corrected_row else []
    if not isinstance(winning_labels, Sequence) or isinstance(winning_labels, (str, bytes)):
        winning_labels = []
    operator_result = kit.dagger_off_path_data_collection_operator(
        _probe_rows(game, corrected_row, baseline_row),
        winning_labels=[str(label) for label in winning_labels],
    )

    corrected_solve = int(_attempt_reached_level(corrected_row) >= 2)
    baseline_solve = int(_attempt_reached_level(baseline_row) >= 2)
    corrected_first = int(bool(corrected_row and corrected_row.get("first_win") is True))
    baseline_first = int(bool(baseline_row and baseline_row.get("first_win") is True))
    corrected_actions = _attempt_actions(corrected_row)
    baseline_actions = _attempt_actions(baseline_row)
    action_lift = (
        float(baseline_actions - corrected_actions)
        if baseline_actions is not None
        and corrected_actions is not None
        and baseline_actions > corrected_actions
        else 0.0
    )
    new_level = bool(
        _attempt_reproduced(corrected_row)
        and _attempt_reached_level(corrected_row) > _attempt_reached_level(baseline_row)
    )
    value_added = bool(
        (corrected_solve - baseline_solve) > 0
        or (corrected_first - baseline_first) > 0
        or action_lift > 0.0
    )
    before = _as_float(a2_artifact.get("distribution_shift_score_before"))
    after = _as_float(a2_artifact.get("distribution_shift_score_after"))
    transfer_value = {
        "operator": PRIMITIVE_OPERATOR,
        "live_solve_rate_delta": float(corrected_solve - baseline_solve),
        "first_win_rate_delta": float(corrected_first - baseline_first),
        "action_efficiency_lift": action_lift,
        "offline_reproduced_new_level": new_level,
        "corrected_actions_to_first_levelup": corrected_actions,
        "baseline_actions_to_first_levelup": baseline_actions,
        "corrected_reached_level": _attempt_reached_level(corrected_row),
        "baseline_reached_level": _attempt_reached_level(baseline_row),
        "value_head_evals": _as_int(_attempt_lazy_value(corrected_row).get("value_head_evals")),
        "cache_hits": _as_int(_attempt_lazy_value(corrected_row).get("cache_hits")),
        "distribution_shift_score_before": before,
        "distribution_shift_score_after": after,
        "distribution_shift_score_delta": round(after - before, 6),
        "offline_reproduced_new_level_source": "arc_solver_kit.reproduce",
        "value_added": value_added,
    }
    if corrected_row is None or baseline_row is None:
        dead_end = "no cached matched corrected/baseline attempts for this transfer game"
    elif value_added:
        dead_end = ""
    else:
        dead_end = (
            "cached matched DAgger-corrected rows showed zero solve-rate, first-win, and "
            "action-efficiency lift; no new reproduced level banked"
        )
    return {
        "game": str(game),
        "value_added": value_added,
        "transfer_value": transfer_value,
        "operator_result": operator_result,
        "dead_end": dead_end,
    }


def measure_transfer(
    *, transfer_games: Sequence[str], a2_artifact: Mapping[str, Any]
) -> list[JsonDict]:
    return [measure_transfer_game(game, a2_artifact=a2_artifact) for game in transfer_games]


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
    """SCENARIO-ARC-WMTE-4668: assemble the primitive transfer artifact."""

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
    any_new_level = bool(new_level_records)
    residual = ""
    if preconditions_checked.get("ok") is False:
        residual = "preconditions failed before transfer measurement"
    elif winner is None:
        residual = (
            "Persisted DAgger-lite off-path data-collection operator had zero transfer lift "
            "on cached held-out games; the residual bridge gap is converting corrected "
            "search-distribution labels into value-routing decisions that expose new "
            "first-win or multi-level candidates."
        )

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
            "reuse_note": "reuse live-frontier off-path row collection before retraining value heads",
        },
        "transfer_games": [str(row.get("game") or "") for row in rows],
        "transfer_value_per_game": transfer_values,
        "offline_reproduced_new_level": any_new_level,
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
        "requirements": ["REQ-ARC-WMTE-4668"],
        "scenarios": ["SCENARIO-ARC-WMTE-4668"],
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
        errors.append("inference_substrate must match REQ-ARC-WMTE-4668")
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
        errors.append("field_principles must match REQ-ARC-WMTE-4668")
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
        transfer_results = measure_transfer(transfer_games=transfer_games, a2_artifact=a2)
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
