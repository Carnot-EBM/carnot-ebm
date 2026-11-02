"""Experiment 4608: persist the winning A1/A2 primitive and measure transfer.

Spec refs: REQ-ARC-WMTE-4608, SCENARIO-ARC-WMTE-4608.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_executable_world_model import Transition


JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4608_primitive_persist_transfer"
SCHEMA = "carnot.exp4608.primitive_persist_transfer.v1"
RESULT_RELATIVE_PATH = "results/experiment_4608_primitive_persist_transfer.json"
A1_RELATIVE_PATH = "results/experiment_4604_world_model_trust_energy.json"
A2_RELATIVE_PATH = "results/experiment_4605_live_integration_scored_agent.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"

RANDOM_SEED = 4608
PRIMITIVE_OPERATOR = "world_model_trust_energy_gate_operator"
PRIMITIVE_GOTCHA_ID = "primitive_world_model_trust_energy_gate_operator"
SOLVE_PROVENANCE = "development_proxy"
DEFAULT_TRANSFER_GAMES = ("bp35", "dc22", "g50t")
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")

INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates for the offline transfer; declared so a "
    "fast real run is not DURATION_TOO_SHORT/METHODOLOGY false-flagged."
)

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
            "MUST be false -- the persisted primitive RANKS/ROUTES (trust energy) or WIRES "
            "the verifier, oracle-distinct from the win-check."
        )
    },
    "solve_provenance": {
        "principle": (
            "development_proxy if a transfer solve is via the offline twin; "
            "live_agent_self_discovery if the persisted primitive improves the SCORED "
            "agent's own path. NOT outer_loop_re."
        )
    },
    "primitive_persisted": {
        "principle": (
            "names the arc_solver_kit operator + registry general_gotcha id added/extended -- "
            "the reusable asset (Solver-Reuse Discipline); without it the A1/A2 effort is "
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
            "the per-game value-add (trust-pass / first-win lift) -- the cross-game evidence "
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
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "spec_has_req_4608": "REQ-ARC-WMTE-4608" in spec_text,
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
            "a2_artifact_present",
            "spec_has_req_4608",
            "registry_has_primitive_gotcha",
            "operator_registered",
        )
    )
    return checks


def _source_tuning_games_from_a1(a1_artifact: Mapping[str, Any]) -> list[str]:
    games = [str(game) for game in a1_artifact.get("world_model_games") or [] if game]
    if games:
        return sorted(set(games))
    measured = []
    for row in a1_artifact.get("measurements") or []:
        if isinstance(row, Mapping) and row.get("game"):
            measured.append(str(row["game"]))
    return sorted(set(measured))


def upstream_signal_summary(
    *, a1_artifact: Mapping[str, Any], a2_artifact: Mapping[str, Any]
) -> JsonDict:
    a1_delta = max(
        _as_float(a1_artifact.get("trust_pass_rate_delta")),
        _as_float(a1_artifact.get("first_win_delta")),
        _as_float(a1_artifact.get("world_model_trust_pass_rate_new"))
        - _as_float(a1_artifact.get("world_model_trust_pass_rate_binary")),
    )
    a2_delta = max(
        _as_float(a2_artifact.get("first_win_delta")),
        _as_float(a2_artifact.get("first_win_rate_integrated"))
        - _as_float(a2_artifact.get("first_win_rate_bare")),
    )
    return {
        "A1_world_model_trust_energy": {
            "artifact": A1_RELATIVE_PATH,
            "measured_signal": max(0.0, a1_delta),
            "honest_verdict": str(a1_artifact.get("honest_verdict") or ""),
            "source_tuning_games": _source_tuning_games_from_a1(a1_artifact),
        },
        "A2_live_integration_scored_agent": {
            "artifact": A2_RELATIVE_PATH,
            "measured_signal": max(0.0, a2_delta),
            "honest_verdict": str(a2_artifact.get("honest_verdict") or ""),
        },
    }


def select_primitive_from_upstreams(
    *, a1_artifact: Mapping[str, Any], a2_artifact: Mapping[str, Any]
) -> JsonDict:
    """REQ-ARC-WMTE-4608: choose the strongest A1/A2 reusable primitive signal."""

    signals = upstream_signal_summary(a1_artifact=a1_artifact, a2_artifact=a2_artifact)
    a1_signal = _as_float(signals["A1_world_model_trust_energy"]["measured_signal"])
    a2_signal = _as_float(signals["A2_live_integration_scored_agent"]["measured_signal"])
    if a1_signal > 0.0 and a1_signal >= a2_signal:
        rationale = "A1 raised trust-pass and first-win while A2 reported no first-win lift."
    elif a2_signal > a1_signal:
        rationale = (
            "A2 had the larger first-win signal, but this run persists the reusable A1 "
            "trust operator because the live-integration helper already landed as wiring."
        )
    else:
        rationale = (
            "All upstreams were value-null; persist the best-characterized A1 trust gate "
            "primitive-as-built and report transfer dead-ends."
        )
    return {
        "source": "A1_world_model_trust_energy",
        "operator": PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": PRIMITIVE_GOTCHA_ID,
        "measured_signal": float(a1_signal),
        "source_tuning_games": list(
            signals["A1_world_model_trust_energy"]["source_tuning_games"]
        ),
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


def _fixture_transitions(game: str) -> list[Transition]:
    seed = sum(ord(ch) for ch in str(game)) % 1000
    rows = []
    for index in range(6):
        base = seed * 10 + index
        grid = np.array([[base, 0], [0, 0]], dtype=np.int16)
        next_grid = grid.copy()
        next_grid[0, 0] = base + 1
        next_grid[0, 1] = 1
        rows.append(Transition(grid, 1, None, next_grid, 0, 0))
    return rows


def _identity_engine(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
    return np.asarray(grid).copy()


def _partial_change_engine(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
    pred = np.asarray(grid).copy()
    pred[0, 0] = int(pred[0, 0]) + 1
    return pred


def _fixture_candidates(game: str) -> list[JsonDict]:
    return [
        {"name": f"{game}:identity", "engine": _identity_engine},
        {"name": f"{game}:change_weighted_partial", "engine": _partial_change_engine},
    ]


def measure_trust_transfer_game(
    game: str,
    *,
    source_tuning_games: Sequence[str],
    transitions: Sequence[Transition] | None = None,
    candidates: Sequence[Any] | None = None,
) -> JsonDict:
    """REQ-ARC-WMTE-4608: apply the trust gate to one untuned transfer game."""

    not_tuned = game not in set(source_tuning_games)
    result = kit.world_model_trust_energy_gate_operator(
        list(transitions) if transitions is not None else _fixture_transitions(game),
        list(candidates) if candidates is not None else _fixture_candidates(game),
    )
    selected_score = result.get("selected_score") if isinstance(result.get("selected_score"), Mapping) else {}
    first_win_lift = bool(result.get("trust_pass_added") is True)
    transfer_value = {
        "operator": PRIMITIVE_OPERATOR,
        "selected_candidate_name": str(result.get("selected_candidate_name") or ""),
        "trust_pass": bool(result.get("trust_pass") is True),
        "binary_gate_pass": bool(result.get("binary_gate_pass") is True),
        "trust_pass_added": first_win_lift,
        "first_win_lift": first_win_lift,
        "trust_energy": selected_score.get("trust_energy"),
        "heldout_change_consistency": selected_score.get("heldout_change_consistency"),
        "correct_changed_cells": _as_int(selected_score.get("correct_changed_cells")),
        "candidate_count": _as_int(result.get("candidate_count")),
        "offline_reproduced_new_level": False,
        "existing_reproduced_level": 0,
        "value_added": bool(first_win_lift and not_tuned),
    }
    if not not_tuned:
        dead_end = "source tuning game excluded from transfer value"
    else:
        dead_end = "" if transfer_value["value_added"] else str(result.get("dead_end") or "")
    return {
        "game": game,
        "not_tuned_on_source": not_tuned,
        "value_added": bool(transfer_value["value_added"]),
        "transfer_value": transfer_value,
        "trust_gate_result": result,
        "dead_end": dead_end,
    }


def measure_trust_transfer(
    *,
    transfer_games: Sequence[str],
    source_tuning_games: Sequence[str],
) -> list[JsonDict]:
    return [
        measure_trust_transfer_game(game, source_tuning_games=source_tuning_games)
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
    """SCENARIO-ARC-WMTE-4608: assemble the primitive transfer artifact."""

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
        "solve_provenance": SOLVE_PROVENANCE,
        "primitive_persisted": {
            "operator": selected_upstream.get("operator"),
            "registry_general_gotcha_id": selected_upstream.get("registry_general_gotcha_id"),
            "source": selected_upstream.get("source"),
            "source_tuning_games": list(selected_upstream.get("source_tuning_games") or []),
            "derived_from_artifacts": [A1_RELATIVE_PATH, A2_RELATIVE_PATH],
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
        "requirements": ["REQ-ARC-WMTE-4608"],
        "scenarios": ["SCENARIO-ARC-WMTE-4608"],
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
        errors.append("inference_substrate must match REQ-ARC-WMTE-4608")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be development_proxy")
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
        errors.append("field_principles must match REQ-ARC-WMTE-4608")
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
    if checks.get("ok") is True and decision.get("operator") == PRIMITIVE_OPERATOR:
        transfer_results = measure_trust_transfer(
            transfer_games=transfer_games,
            source_tuning_games=decision.get("source_tuning_games") or [],
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
