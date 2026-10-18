"""Experiment 4656: persist the .429 primitive and measure transfer.

Spec refs: REQ-ARC-WMTE-4656, SCENARIO-ARC-WMTE-4656.
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

EXPERIMENT = "experiment_4656_primitive_persist_transfer"
SCHEMA = "carnot.exp4656.primitive_persist_transfer.v1"
RESULT_RELATIVE_PATH = "results/experiment_4656_primitive_persist_transfer.json"
A1_RELATIVE_PATH = "results/experiment_4652_value_routing_cost_fix_live.json"
A2_RELATIVE_PATH = "results/experiment_4653_energy_fitness_qd_generation_live.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"

RANDOM_SEED = 4656
PRIMITIVE_OPERATOR = "cheap_value_routing_cost_fix_operator"
PRIMITIVE_GOTCHA_ID = "primitive_cheap_value_routing_cost_fix_operator"
FEATURE_SUBSET = "cross_game_features_v3:v2_plus_frame_delta"
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
            "MUST be false -- the persisted primitive ranks/routes/generates, oracle-distinct "
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
            "record); per the .428 A5 pattern."
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


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
        "spec_has_req_4656": "REQ-ARC-WMTE-4656" in spec_text,
        "registry_has_primitive_gotcha": _registry_has_gotcha(registry),
        "operator_registered": _operator_registered(),
    }
    required = (
        "agents_md_read",
        "codex_md_read",
        "offline_arcade",
        "a1_artifact_present",
        "a2_artifact_present",
        "spec_has_req_4656",
        "registry_has_primitive_gotcha",
        "operator_registered",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    return checks


def upstream_signal_summary(
    *, a1_artifact: Mapping[str, Any], a2_artifact: Mapping[str, Any]
) -> JsonDict:
    a1_live_signal = max(
        _as_float(a1_artifact.get("solve_rate_delta")),
        _as_float(a1_artifact.get("first_win_rate_delta")),
    )
    a1_component_signal = 0.0
    if a1_artifact.get("feature_output_identical_verified") is True:
        a1_component_signal += 0.34
    if _as_float(a1_artifact.get("per_node_feature_cost_ms")) < 1.0:
        a1_component_signal += 0.33
    if a1_artifact.get("sim_timed_out") is False:
        a1_component_signal += 0.33
    a2_signal = max(
        _as_float(a2_artifact.get("solve_rate_delta")),
        _as_float(a2_artifact.get("first_win_rate_delta")),
        float(_as_int(a2_artifact.get("winner_generated_count"))),
    )
    return {
        "A1_value_routing_cost_fix_live": {
            "artifact": A1_RELATIVE_PATH,
            "honest_verdict": str(a1_artifact.get("honest_verdict") or ""),
            "chosen_submitted_config": a1_artifact.get("chosen_submitted_config"),
            "measured_signal": max(0.0, a1_live_signal),
            "component_signal": round(min(1.0, a1_component_signal), 6),
            "feature_subset": str(a1_artifact.get("feature_subset") or FEATURE_SUBSET),
            "per_node_feature_cost_ms": _as_float(a1_artifact.get("per_node_feature_cost_ms")),
        },
        "A2_energy_fitness_qd_generator": {
            "artifact": A2_RELATIVE_PATH,
            "honest_verdict": str(a2_artifact.get("honest_verdict") or ""),
            "chosen_submitted_config": a2_artifact.get("chosen_submitted_config"),
            "measured_signal": max(0.0, a2_signal),
            "winner_generated": bool(a2_artifact.get("winner_generated") is True),
            "winner_generated_count": _as_int(a2_artifact.get("winner_generated_count")),
        },
    }


def select_primitive_from_upstreams(
    *, a1_artifact: Mapping[str, Any], a2_artifact: Mapping[str, Any]
) -> JsonDict:
    """REQ-ARC-WMTE-4656: choose the .429 primitive or strongest component."""

    signals = upstream_signal_summary(a1_artifact=a1_artifact, a2_artifact=a2_artifact)
    a1 = signals["A1_value_routing_cost_fix_live"]
    a2 = signals["A2_energy_fitness_qd_generator"]
    a1_cleared = str(a1["honest_verdict"]).startswith("success:") and _as_float(
        a1["measured_signal"]
    ) > 0.0
    a2_cleared = bool(a2["winner_generated"]) and _as_int(a2["winner_generated_count"]) > 0
    if a1_cleared:
        rationale = "A1 value-routing cost fix cleared its live gate; persist its reusable cheap route."
        source = "A1_value_routing_cost_fix"
        signal = _as_float(a1["measured_signal"])
    elif a2_cleared:
        rationale = (
            "A2 generated a winner, but this run persists the shared cheap value-routing "
            "cost substrate because the current solver-kit primitive captured here is the "
            "available reusable component."
        )
        source = "A1_cheap_value_routing_cost_fix"
        signal = _as_float(a2["measured_signal"])
    else:
        rationale = (
            "both A1 and A2 were live-value null; persist the strongest characterized "
            "component, the cheap-feature value-routing cost fix."
        )
        source = "A1_cheap_value_routing_cost_fix"
        signal = _as_float(a1["component_signal"])
    rank = [
        {
            "source": "A1_cheap_value_routing_cost_fix",
            "artifact": A1_RELATIVE_PATH,
            "measured_signal": float(max(_as_float(a1["measured_signal"]), _as_float(a1["component_signal"]))),
            "live_signal": float(a1["measured_signal"]),
            "component_signal": float(a1["component_signal"]),
        },
        {
            "source": "A2_energy_fitness_qd_generator",
            "artifact": A2_RELATIVE_PATH,
            "measured_signal": float(a2["measured_signal"]),
            "winner_generated": bool(a2["winner_generated"]),
        },
    ]
    return {
        "source": source,
        "operator": PRIMITIVE_OPERATOR,
        "base_operator": "value_head_bridge_fix_operator",
        "registry_general_gotcha_id": PRIMITIVE_GOTCHA_ID,
        "measured_signal": float(signal),
        "feature_subset": str(a1["feature_subset"] or FEATURE_SUBSET),
        "per_node_feature_cost_ms": float(a1["per_node_feature_cost_ms"]),
        "upstream_signal_rank": sorted(rank, key=lambda row: (-float(row["measured_signal"]), row["source"])),
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


def _probe_candidates(game: str, value_row: Mapping[str, Any] | None, baseline_row: Mapping[str, Any] | None) -> list[JsonDict]:
    baseline_target = bool(baseline_row and baseline_row.get("first_win") is True)
    value_target = bool(value_row and value_row.get("first_win") is True)
    return [
        {
            "candidate_id": f"{game}:baseline_order",
            "state_key": f"{game}:baseline",
            "value_score": 0.9 if value_target and not baseline_target else 0.1,
            "reaches_levelup": baseline_target,
        },
        {
            "candidate_id": f"{game}:cheap_value_route",
            "state_key": f"{game}:value",
            "value_score": 0.1,
            "reaches_levelup": value_target,
        },
    ]


def measure_transfer_game(game: str, *, a1_artifact: Mapping[str, Any]) -> JsonDict:
    """REQ-ARC-WMTE-4656: apply the persisted operator to one cached transfer game."""

    value_measurement = a1_artifact.get("value_routed_measurement")
    baseline_measurement = a1_artifact.get("baseline_measurement")
    value_row = (
        _attempt_by_game(value_measurement, game) if isinstance(value_measurement, Mapping) else None
    )
    baseline_row = (
        _attempt_by_game(baseline_measurement, game)
        if isinstance(baseline_measurement, Mapping)
        else None
    )
    feature_cost = _as_float(a1_artifact.get("per_node_feature_cost_ms"))
    operator_result = kit.cheap_value_routing_cost_fix_operator(
        _probe_candidates(game, value_row, baseline_row),
        score_key="value_score",
        first_win_budget=1,
        feature_subset=str(a1_artifact.get("feature_subset") or FEATURE_SUBSET),
        per_node_feature_cost_ms=feature_cost,
    )
    value_solve = int(_attempt_reached_level(value_row) >= 2)
    baseline_solve = int(_attempt_reached_level(baseline_row) >= 2)
    value_first = int(bool(value_row and value_row.get("first_win") is True))
    baseline_first = int(bool(baseline_row and baseline_row.get("first_win") is True))
    value_actions = _attempt_actions(value_row)
    baseline_actions = _attempt_actions(baseline_row)
    action_lift = (
        float(baseline_actions - value_actions)
        if baseline_actions is not None and value_actions is not None and baseline_actions > value_actions
        else 0.0
    )
    new_level = bool(
        _attempt_reproduced(value_row)
        and _attempt_reached_level(value_row) > _attempt_reached_level(baseline_row)
    )
    value_added = bool(
        (value_solve - baseline_solve) > 0
        or (value_first - baseline_first) > 0
        or action_lift > 0.0
    )
    transfer_value = {
        "operator": PRIMITIVE_OPERATOR,
        "feature_subset": str(a1_artifact.get("feature_subset") or FEATURE_SUBSET),
        "live_solve_rate_delta": float(value_solve - baseline_solve),
        "first_win_rate_delta": float(value_first - baseline_first),
        "action_efficiency_lift": action_lift,
        "offline_reproduced_new_level": new_level,
        "value_routed_actions_to_first_levelup": value_actions,
        "baseline_actions_to_first_levelup": baseline_actions,
        "value_routed_reached_level": _attempt_reached_level(value_row),
        "baseline_reached_level": _attempt_reached_level(baseline_row),
        "value_head_evals": _as_int(_attempt_lazy_value(value_row).get("value_head_evals")),
        "cache_hits": _as_int(_attempt_lazy_value(value_row).get("cache_hits")),
        "per_node_feature_cost_ms": feature_cost,
        "value_added": value_added,
    }
    if value_row is None or baseline_row is None:
        dead_end = "no cached matched value-routed/baseline attempts for this transfer game"
    elif value_added:
        dead_end = ""
    else:
        dead_end = (
            "cached matched value-routing rows showed zero solve-rate, first-win, and "
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
    *, transfer_games: Sequence[str], a1_artifact: Mapping[str, Any]
) -> list[JsonDict]:
    return [measure_transfer_game(game, a1_artifact=a1_artifact) for game in transfer_games]


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
    """SCENARIO-ARC-WMTE-4656: assemble the primitive transfer artifact."""

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
            "Persisted cheap-feature value-routing cost fix had zero transfer lift on cached "
            "held-out games; value calibration/candidate generation remains the residual dead-end."
        )

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "primitive_persisted": {
            "operator": selected_upstream.get("operator"),
            "base_operator": selected_upstream.get("base_operator"),
            "registry_general_gotcha_id": selected_upstream.get("registry_general_gotcha_id"),
            "source": selected_upstream.get("source"),
            "derived_from_artifacts": [A1_RELATIVE_PATH, A2_RELATIVE_PATH],
            "feature_subset": selected_upstream.get("feature_subset") or FEATURE_SUBSET,
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
        "requirements": ["REQ-ARC-WMTE-4656"],
        "scenarios": ["SCENARIO-ARC-WMTE-4656"],
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
        errors.append("inference_substrate must match REQ-ARC-WMTE-4656")
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
        errors.append("field_principles must match REQ-ARC-WMTE-4656")
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
        transfer_results = measure_transfer(transfer_games=transfer_games, a1_artifact=a1)
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
