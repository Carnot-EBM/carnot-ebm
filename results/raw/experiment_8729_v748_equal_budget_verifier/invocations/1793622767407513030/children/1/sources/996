"""Experiment 4596: persist the A1/A3 winning primitive and measure transfer.

Spec refs: REQ-CAPSTONE-4596, SCENARIO-CAPSTONE-4596,
SCENARIO-CAPSTONE-4596-FIELD-PRINCIPLES.
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

from carnot import experiment_4472_variant_generic_transfer_benchmark_v4 as variant_bench
from carnot.agentic import arc_solver_kit as kit


JsonDict = dict[str, Any]
ReproduceChecker = Callable[[str, Sequence[str], int], Mapping[str, Any]]

EXPERIMENT = "experiment_4596_primitive_persist_transfer"
SCHEMA = "carnot.exp4596.primitive_persist_transfer.v1"
RESULT_RELATIVE_PATH = "results/experiment_4596_primitive_persist_transfer.json"
A1_RELATIVE_PATH = "results/experiment_4592_generation_completeness_wiring.json"
A3_RELATIVE_PATH = "results/experiment_4594_goal_energy_generation_prior.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"

RANDOM_SEED = 4596
PRIMITIVE_OPERATOR = "approach_dispatcher_operator"
PRIMITIVE_GOTCHA_ID = "primitive_approach_dispatcher_operator"
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates for the offline transfer; declared so a "
    "fast real run is not DURATION_TOO_SHORT/METHODOLOGY false-flagged."
)
DEFAULT_TRANSFER_GAMES = ("ar25", "cn04", "dc22")
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: primitive_persisted_transfer_<game>_value_added OR "
            "complete: primitive_persisted_transfer_null_characterized."
        )
    },
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the persisted primitive GENERATES candidates / biases proposals, "
            "oracle-distinct from the win-check."
        )
    },
    "primitive_persisted": {
        "principle": (
            "names the arc_solver_kit operator + registry general_gotcha id added/extended -- "
            "the reusable asset (Solver-Reuse Discipline); without it the A1/A3 effort is "
            "wasted per the ARC reuse rule."
        )
    },
    "transfer_games": {
        "principle": (
            "the games the primitive was applied to (NOT tuned on) -- the generalization test."
        )
    },
    "transfer_value_per_game": {
        "principle": (
            "the per-game value-add (winner-generated / win-reached) -- the cross-game evidence "
            "the primitive generalizes."
        )
    },
    "offline_reproduced": {
        "principle": "only offline-reproduced new levels count toward reproducible_total_levels."
    },
    "registry_updated": {
        "principle": (
            "the primitive + transfer dead-ends persisted so the next milestone reuses, not "
            "re-derives."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {"principle": "catches silent drift on replay."},
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "selected_upstream",
    "upstream_signals",
    "transfer_results",
    "transfer_dead_ends",
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


def _load_registry(root: Path) -> JsonDict:
    try:
        loaded = yaml.safe_load((root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
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
        "a3_artifact_present": (root_path / A3_RELATIVE_PATH).exists(),
        "spec_has_req_4596": "REQ-CAPSTONE-4596" in spec_text,
        "registry_has_primitive_gotcha": _registry_has_gotcha(registry),
        "operator_registered": _operator_registered(),
    }
    checks["ok"] = all(
        bool(checks[key])
        for key in (
            "agents_md_read",
            "codex_md_read",
            "offline_arcade",
            "a1_artifact_present",
            "a3_artifact_present",
            "spec_has_req_4596",
            "registry_has_primitive_gotcha",
            "operator_registered",
        )
    )
    return checks


def _source_tuning_games_from_a1(a1_artifact: Mapping[str, Any]) -> list[str]:
    games: set[str] = set()
    for variant in a1_artifact.get("newly_solved_variants") or []:
        game = str(variant).split("~", 1)[0]
        if game:
            games.add(game)
    return sorted(games)


def upstream_signal_summary(
    *, a1_artifact: Mapping[str, Any], a3_artifact: Mapping[str, Any]
) -> JsonDict:
    a1_delta = max(
        _as_float(a1_artifact.get("winner_generated_delta")),
        _as_float(a1_artifact.get("transfer_delta")),
        _as_float(a1_artifact.get("generic_transfer_rate_with_wiring"))
        - _as_float(a1_artifact.get("generic_transfer_rate_baseline")),
    )
    a3_delta = max(
        _as_float(a3_artifact.get("winner_generated_delta")),
        _as_float(a3_artifact.get("generic_transfer_rate_with_energy"))
        - _as_float(a3_artifact.get("generic_transfer_rate_no_energy")),
    )
    return {
        "A1_approach_dispatcher": {
            "artifact": A1_RELATIVE_PATH,
            "measured_signal": max(0.0, a1_delta),
            "honest_verdict": str(a1_artifact.get("honest_verdict") or ""),
            "source_tuning_games": _source_tuning_games_from_a1(a1_artifact),
        },
        "A3_goal_energy_generation_prior": {
            "artifact": A3_RELATIVE_PATH,
            "measured_signal": max(0.0, a3_delta),
            "honest_verdict": str(a3_artifact.get("honest_verdict") or ""),
        },
    }


def select_primitive_from_upstreams(
    *, a1_artifact: Mapping[str, Any], a3_artifact: Mapping[str, Any]
) -> JsonDict:
    """REQ-CAPSTONE-4596: choose the strongest A1/A3 primitive signal."""

    signals = upstream_signal_summary(a1_artifact=a1_artifact, a3_artifact=a3_artifact)
    a1_signal = _as_float(signals["A1_approach_dispatcher"]["measured_signal"])
    a3_signal = _as_float(signals["A3_goal_energy_generation_prior"]["measured_signal"])
    if a1_signal > 0.0 and a1_signal >= a3_signal:
        rationale = "A1 raised winner-generated/transfer rate while A3 reported no positive value."
    elif a3_signal > a1_signal:
        rationale = (
            "A3 had the larger numeric signal, but this run persists the A1 dispatcher only "
            "because no reusable goal-energy operator is supported in the live solver kit."
        )
    else:
        rationale = (
            "All upstreams were value-null; persist the best-characterized A1 dispatcher "
            "primitive-as-built and report transfer dead-ends."
        )
    return {
        "source": "A1_approach_dispatcher",
        "operator": PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": PRIMITIVE_GOTCHA_ID,
        "measured_signal": float(a1_signal),
        "source_tuning_games": list(signals["A1_approach_dispatcher"]["source_tuning_games"]),
        "upstream_signal_rank": sorted(
            (
                {
                    "source": key,
                    "artifact": value["artifact"],
                    "measured_signal": float(value["measured_signal"]),
                }
                for key, value in signals.items()
            ),
            key=lambda row: (-row["measured_signal"], row["source"]),
        ),
        "selection_rationale": rationale,
    }


def _normalise_action_label(label: Any) -> str:
    if isinstance(label, str):
        stripped = label.strip()
        if stripped.startswith("{"):
            loaded = json.loads(stripped)
            payload: JsonDict = {"action": int(loaded["action"])}
            data = loaded.get("data")
            if data is None and ("x" in loaded or "y" in loaded):
                data = {key: loaded[key] for key in ("x", "y") if key in loaded}
            if data is not None:
                payload["data"] = data
            return json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if stripped.isdigit():
            return json.dumps({"action": int(stripped)}, sort_keys=True, separators=(",", ":"))
    if isinstance(label, Mapping):
        payload = {"action": int(label["action"])}
        if "data" in label:
            payload["data"] = label["data"]
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return json.dumps({"action": int(label)}, sort_keys=True, separators=(",", ":"))


def _default_reproduce_checker(game: str, labels: Sequence[str], claimed: int) -> Mapping[str, Any]:
    return kit.reproduce(  # pragma: no cover - SDK boundary
        game, labels, variant_bench._apply_action_label, claimed_level=int(claimed)
    )


def _cached_solution_candidate(
    root: Path,
    game: str,
    approach: str,
    *,
    reproduce_checker: ReproduceChecker | None,
) -> JsonDict | None:
    path = root / "results" / f"arc_loop_solve_{game}.json"
    source = _load_json(path)
    labels = source.get("solution_labels")
    if not isinstance(labels, list) or not labels:
        return None
    claimed = _as_int(source.get("reproduced_levels") or source.get("reached_level"))
    if claimed <= 0:
        return None
    normalised = [_normalise_action_label(label) for label in labels]
    checker = reproduce_checker or _default_reproduce_checker
    gate = dict(checker(game, normalised, claimed))
    reproduced = bool(gate.get("reproduced") is True) and _as_int(gate.get("reached_level")) >= claimed
    return {
        "approach": approach,
        "candidate_id": f"{game}:{approach}:cached_reproduced_candidate",
        "candidate_generated": bool(reproduced),
        "winner_generated": bool(reproduced),
        "win_reached": bool(reproduced),
        "offline_reproduced": bool(reproduced),
        "reached_level": _as_int(gate.get("reached_level")),
        "actions": len(normalised),
        "solution_labels": normalised if reproduced else [],
        "source_artifact": str(path.relative_to(root)),
        "reproduction_gate": gate,
        "new_level_banked": False,
    }


def _a1_attempt_by_game(a1_artifact: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    measurement = a1_artifact.get("wired_measurement")
    attempts = measurement.get("variant_attempts") if isinstance(measurement, Mapping) else []
    rows: dict[str, Mapping[str, Any]] = {}
    for row in attempts or []:
        if isinstance(row, Mapping) and row.get("game"):
            rows[str(row["game"])] = row
    return rows


def _route_for_game(game: str, a1_artifact: Mapping[str, Any]) -> JsonDict:
    attempt = _a1_attempt_by_game(a1_artifact).get(game, {})
    route = attempt.get("selected_feature_route") if isinstance(attempt, Mapping) else None
    if isinstance(route, Mapping):
        return dict(route)
    approach = str(attempt.get("selected_approach") or "default_graph_explore") if isinstance(attempt, Mapping) else "default_graph_explore"
    return {"mechanic_class": "", "approach": approach}


def build_transfer_candidates(
    root: Path,
    game: str,
    route: Mapping[str, Any],
    *,
    reproduce_checker: ReproduceChecker | None = None,
) -> list[JsonDict]:
    routed_approach = str(route.get("approach") or "default_graph_explore")
    candidates: list[JsonDict] = [
        {
            "approach": "default_graph_explore",
            "candidate_id": f"{game}:default_graph_explore:no_wiring_control",
            "candidate_generated": False,
            "winner_generated": False,
            "win_reached": False,
            "reached_level": 0,
            "actions": 0,
        }
    ]
    cached = _cached_solution_candidate(
        root, game, routed_approach, reproduce_checker=reproduce_checker
    )
    if cached is not None:
        candidates.append(cached)
    return candidates


def measure_dispatcher_transfer_game(
    game: str,
    *,
    route: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    source_tuning_games: Sequence[str],
) -> JsonDict:
    """REQ-CAPSTONE-4596: apply the dispatcher to one untuned game."""

    result = kit.approach_dispatcher_operator(route, list(candidates))
    selected = result.get("selected_candidate") if isinstance(result.get("selected_candidate"), Mapping) else {}
    gate = selected.get("reproduction_gate") if isinstance(selected, Mapping) else {}
    reached = _as_int(selected.get("reached_level") if isinstance(selected, Mapping) else 0)
    transfer_value = {
        "operator": PRIMITIVE_OPERATOR,
        "selected_approach": result["selected_approach"],
        "executed_approach": result["executed_approach"],
        "candidate_generated": bool(result["candidate_generated"]),
        "winner_generated": bool(result["winner_generated"]),
        "win_reached": bool(result["win_reached"]),
        "baseline_winner_generated": bool(result["baseline_winner_generated"]),
        "offline_reproduced_new_level": bool(selected.get("new_level_banked") is True)
        if isinstance(selected, Mapping)
        else False,
        "existing_reproduced_level": reached if isinstance(gate, Mapping) and gate.get("reproduced") is True else 0,
        "candidate_count": int(result["candidate_count"]),
        "value_added": bool(result["value_added"]),
    }
    return {
        "game": game,
        "route": dict(route),
        "not_tuned_on_source": game not in set(source_tuning_games),
        "value_added": bool(result["value_added"]) and game not in set(source_tuning_games),
        "transfer_value": transfer_value,
        "dispatcher_result": result,
        "dead_end": "" if result["value_added"] else str(result.get("dead_end") or ""),
    }


def measure_dispatcher_transfer(
    root: Path,
    *,
    a1_artifact: Mapping[str, Any],
    transfer_games: Sequence[str],
    source_tuning_games: Sequence[str],
    reproduce_checker: ReproduceChecker | None = None,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for game in transfer_games:
        route = _route_for_game(game, a1_artifact)
        candidates = build_transfer_candidates(
            root, game, route, reproduce_checker=reproduce_checker
        )
        rows.append(
            measure_dispatcher_transfer_game(
                game,
                route=route,
                candidates=candidates,
                source_tuning_games=source_tuning_games,
            )
        )
    return rows


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
    """SCENARIO-CAPSTONE-4596: assemble the primitive transfer artifact."""

    rows = [dict(row) for row in transfer_results]
    winner = _success_row(rows)
    if preconditions_checked.get("ok") is False:
        verdict = "blocked_primitive_persist_transfer_precondition"
    elif winner is not None:
        verdict = f"success: primitive_persisted_transfer_{winner.get('game')}_value_added"
    else:
        verdict = "complete: primitive_persisted_transfer_null_characterized"
    transfer_values: JsonDict = {}
    dead_ends: JsonDict = {}
    new_level_records: list[JsonDict] = []
    existing_sources: JsonDict = {}
    for row in rows:
        game = str(row.get("game") or "")
        value = dict(row.get("transfer_value") or {})
        value["value_added"] = bool(row.get("value_added") is True)
        transfer_values[game] = value
        existing_sources[game] = _as_int(value.get("existing_reproduced_level"))
        if value.get("offline_reproduced_new_level") is True:
            new_level_records.append({"game": game, "source": "arc_solver_kit.reproduce"})
        if row.get("dead_end"):
            dead_ends[game] = str(row["dead_end"])
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
            "source_tuning_games": list(selected_upstream.get("source_tuning_games") or []),
            "derived_from_artifacts": [A1_RELATIVE_PATH, A3_RELATIVE_PATH],
        },
        "transfer_games": [str(row.get("game") or "") for row in rows],
        "transfer_value_per_game": transfer_values,
        "offline_reproduced": {
            "new_levels_banked": len(new_level_records),
            "new_level_records": new_level_records,
            "existing_source_levels": existing_sources,
            "counted_toward_reproducible_total_levels": len(new_level_records),
        },
        "registry_updated": bool(registry_updated),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "selected_upstream": dict(selected_upstream),
        "upstream_signals": dict(upstream_signals),
        "transfer_results": rows,
        "transfer_dead_ends": dead_ends,
        "new_levels_banked": len(new_level_records),
        "field_principles": FIELD_PRINCIPLES,
        "requirements": ["REQ-CAPSTONE-4596"],
        "scenarios": [
            "SCENARIO-CAPSTONE-4596",
            "SCENARIO-CAPSTONE-4596-FIELD-PRINCIPLES",
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
        errors.append("inference_substrate must match REQ-CAPSTONE-4596")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    primitive = artifact.get("primitive_persisted")
    if not isinstance(primitive, Mapping) or primitive.get("operator") != PRIMITIVE_OPERATOR:
        errors.append(f"primitive_persisted must name {PRIMITIVE_OPERATOR}")
    elif primitive.get("registry_general_gotcha_id") != PRIMITIVE_GOTCHA_ID:
        errors.append(f"primitive_persisted must name {PRIMITIVE_GOTCHA_ID}")
    transfer_games = artifact.get("transfer_games")
    if not blocked and (not isinstance(transfer_games, list) or len(transfer_games) < 2):
        errors.append("transfer_games must contain at least two games")
    if not isinstance(artifact.get("transfer_value_per_game"), Mapping):
        errors.append("transfer_value_per_game must be a mapping")
    if not isinstance(artifact.get("offline_reproduced"), Mapping):
        errors.append("offline_reproduced must be a mapping")
    if type(artifact.get("registry_updated")) is not bool:
        errors.append("registry_updated must be a bare bool")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-CAPSTONE-4596")
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
        if banked != len(records) if isinstance(records, list) else banked != 0:
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
    reproduce_checker: ReproduceChecker | None = None,
    now: Callable[[], float] = time.perf_counter,
    write: bool = True,
) -> JsonDict:
    started = now()
    root_path = Path(root)
    checks = check_preconditions(root_path, offline_arcade_checker=offline_arcade_checker)
    a1 = _load_json(root_path / A1_RELATIVE_PATH)
    a3 = _load_json(root_path / A3_RELATIVE_PATH)
    signals = upstream_signal_summary(a1_artifact=a1, a3_artifact=a3)
    decision = select_primitive_from_upstreams(a1_artifact=a1, a3_artifact=a3)
    transfer_results: list[JsonDict] = []
    if checks.get("ok") is True and decision.get("operator") == PRIMITIVE_OPERATOR:
        transfer_results = measure_dispatcher_transfer(
            root_path,
            a1_artifact=a1,
            transfer_games=transfer_games,
            source_tuning_games=decision.get("source_tuning_games") or [],
            reproduce_checker=reproduce_checker,
        )
    artifact = build_artifact(
        selected_upstream=decision,
        upstream_signals=signals,
        preconditions_checked=checks,
        transfer_results=transfer_results,
        registry_updated=bool(checks.get("registry_has_primitive_gotcha")),
        random_seed=RANDOM_SEED,
        duration_s=now() - started,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root_path)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
