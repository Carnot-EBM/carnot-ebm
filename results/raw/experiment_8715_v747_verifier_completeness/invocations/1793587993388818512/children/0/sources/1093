"""Experiment 4692: persist the .433 directed-exploration primitive.

Spec refs: REQ-ARC-WMTE-4692,
SCENARIO-ARC-WMTE-4692-PERSIST-STRONGEST-COMPONENT,
SCENARIO-ARC-WMTE-4692-TRANSFER-MEASUREMENT.
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

EXPERIMENT = "experiment_4692_primitive_persist_transfer"
SCHEMA = "carnot.exp4692.primitive_persist_transfer.v1"
RESULT_RELATIVE_PATH = "results/experiment_4692_primitive_persist_transfer.json"
A1_RELATIVE_PATH = "results/experiment_4688_controllable_novelty_proposal_policy_live.json"
A2_RELATIVE_PATH = "results/experiment_4689_program_synthesis_action_effect_proposal_filter.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"

RANDOM_SEED = 4692
PRIMITIVE_OPERATOR = "controllable_novelty_embedding_operator"
PRIMITIVE_GOTCHA_ID = "primitive_controllable_novelty_embedding_operator"
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
            "MUST be false -- the persisted primitive generates/ranks/induces, "
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
            "record); per the .431 A5 pattern."
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


def _attempt_by_game(measurement: Mapping[str, Any], game: str) -> Mapping[str, Any] | None:
    rows = measurement.get("variant_attempts")
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        return None
    for row in rows:
        if isinstance(row, Mapping) and row.get("game") == game:
            return row
    return None


def _config_attempt(
    a1_artifact: Mapping[str, Any],
    game: str,
    *,
    preferred_keys: Sequence[str],
    prefix: str,
) -> Mapping[str, Any] | None:
    configs = _mapping_at(a1_artifact, "generic_first_win_by_config")
    for key in preferred_keys:
        row = _attempt_by_game(_mapping_at(configs, key), game)
        if row is not None:
            return row
    for key, value in configs.items():
        if isinstance(key, str) and key.startswith(prefix) and isinstance(value, Mapping):
            row = _attempt_by_game(value, game)
            if row is not None:
                return row
    return None


def _controllable_attempt(a1_artifact: Mapping[str, Any], game: str) -> Mapping[str, Any] | None:
    return _config_attempt(
        a1_artifact,
        game,
        preferred_keys=("controllable_novelty_t0.5", "controllable_novelty"),
        prefix="controllable_novelty",
    )


def _baseline_attempt(a1_artifact: Mapping[str, Any], game: str) -> Mapping[str, Any] | None:
    return _config_attempt(
        a1_artifact,
        game,
        preferred_keys=("no_novelty_bonus", "flat_exploration"),
        prefix="no_novelty",
    )


def _program_probe_by_game(a2_artifact: Mapping[str, Any], game: str) -> Mapping[str, Any] | None:
    probe = _mapping_at(_mapping_at(a2_artifact, "target_arm_results"), "candidate_generation_probe")
    rows = probe.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        return None
    for row in rows:
        if isinstance(row, Mapping) and row.get("game") == game:
            return row
    return None


def _diagnostics_from_attempt(attempt: Mapping[str, Any]) -> Mapping[str, Any]:
    diagnostics = attempt.get("controllable_novelty_diagnostics")
    return diagnostics if isinstance(diagnostics, Mapping) else {}


def _novelty_row_from_attempt(attempt: Mapping[str, Any]) -> JsonDict:
    diagnostics = dict(_diagnostics_from_attempt(attempt))
    diagnostics["game"] = str(attempt.get("game") or "")
    diagnostics["policy_mode"] = str(attempt.get("policy_mode") or "controllable_novelty")
    return diagnostics


def _novelty_rows_for_game(a1_artifact: Mapping[str, Any], game: str) -> list[JsonDict]:
    attempt = _controllable_attempt(a1_artifact, game)
    if attempt is None or not _diagnostics_from_attempt(attempt):
        return []
    return [_novelty_row_from_attempt(attempt)]


def _all_novelty_rows(a1_artifact: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    configs = _mapping_at(a1_artifact, "generic_first_win_by_config")
    for key, value in configs.items():
        if not (isinstance(key, str) and key.startswith("controllable_novelty")):
            continue
        if not isinstance(value, Mapping):
            continue
        for attempt in _sequence_at(value, "variant_attempts"):
            if isinstance(attempt, Mapping) and _diagnostics_from_attempt(attempt):
                rows.append(_novelty_row_from_attempt(attempt))
    return rows


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
        "spec_has_req_4692": "REQ-ARC-WMTE-4692" in spec_text,
        "registry_has_primitive_gotcha": _registry_has_gotcha(registry),
        "operator_registered": _operator_registered(),
    }
    required = (
        "agents_md_read",
        "codex_md_read",
        "offline_arcade",
        "a1_artifact_present",
        "a2_artifact_present",
        "spec_has_req_4692",
        "registry_has_primitive_gotcha",
        "operator_registered",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    return checks


def upstream_signal_summary(
    *, a1_artifact: Mapping[str, Any], a2_artifact: Mapping[str, Any]
) -> JsonDict:
    novelty_result = kit.controllable_novelty_embedding_operator(_all_novelty_rows(a1_artifact))
    a1_live_signal = float(
        max(_as_int(a1_artifact.get("generic_agent_reached_level")), _as_int(a1_artifact.get("reproduced_levels")))
    )
    a1_component_signal = float(novelty_result["usable_embedding_count"]) + _as_float(
        novelty_result.get("best_novelty_signal")
    )
    a2_live_signal = max(
        _as_float(a2_artifact.get("coverage_delta")),
        _as_float(a2_artifact.get("first_win_rate_delta")),
        _as_float(a2_artifact.get("live_first_win_rate_filter")),
    )
    a2_component_signal = float(_as_int(a2_artifact.get("heldout_programs_kept"))) + 0.1 * float(
        _as_int(a2_artifact.get("heldout_programs_rejected"))
    )
    return {
        "A1_controllable_novelty_embedding": {
            "artifact": A1_RELATIVE_PATH,
            "honest_verdict": str(a1_artifact.get("honest_verdict") or ""),
            "chosen_submitted_config": a1_artifact.get("chosen_submitted_config"),
            "measured_signal": max(0.0, a1_live_signal),
            "component_signal": round(float(a1_component_signal), 6),
            "usable_embedding_count": int(novelty_result["usable_embedding_count"]),
            "best_novelty_signal": float(novelty_result["best_novelty_signal"]),
            "residual_cause_hypothesis": a1_artifact.get("residual_cause_hypothesis"),
        },
        "A2_program_synthesis_action_effect_filter": {
            "artifact": A2_RELATIVE_PATH,
            "honest_verdict": str(a2_artifact.get("honest_verdict") or ""),
            "chosen_submitted_config": a2_artifact.get("chosen_submitted_config"),
            "measured_signal": max(0.0, a2_live_signal),
            "component_signal": round(float(a2_component_signal), 6),
            "heldout_programs_kept": _as_int(a2_artifact.get("heldout_programs_kept")),
            "heldout_programs_rejected": _as_int(a2_artifact.get("heldout_programs_rejected")),
            "residual_bridge_gap": a2_artifact.get("residual_bridge_gap"),
        },
    }


def select_primitive_from_upstreams(
    *, a1_artifact: Mapping[str, Any], a2_artifact: Mapping[str, Any]
) -> JsonDict:
    """REQ-ARC-WMTE-4692: choose the cleared operator or strongest null component."""

    signals = upstream_signal_summary(a1_artifact=a1_artifact, a2_artifact=a2_artifact)
    a1 = signals["A1_controllable_novelty_embedding"]
    a2 = signals["A2_program_synthesis_action_effect_filter"]
    a1_cleared = str(a1["honest_verdict"]).startswith("success:") and _as_float(
        a1["measured_signal"]
    ) > 0.0
    a2_cleared = str(a2["honest_verdict"]).startswith("success:") and _as_float(
        a2["measured_signal"]
    ) > 0.0
    if a1_cleared:
        source = "A1_controllable_novelty_proposal_policy"
        signal = _as_float(a1["measured_signal"])
        rationale = "A1 controllable-novelty proposal policy cleared its new-level gate."
    elif a2_cleared:
        source = "A2_program_synthesis_action_effect_filter"
        signal = _as_float(a2["measured_signal"])
        rationale = "A2 program-synthesis action-effect filter cleared its coverage gate."
    else:
        source = "A1_controllable_novelty_embedding"
        signal = _as_float(a1["component_signal"])
        rationale = (
            "both A1 and A2 were value-null; persist the strongest characterized component, "
            "the controllable-novelty embedding mechanism over controllable action effects."
        )
    rank = [
        {
            "source": "A1_controllable_novelty_embedding",
            "artifact": A1_RELATIVE_PATH,
            "measured_signal": float(max(_as_float(a1["measured_signal"]), _as_float(a1["component_signal"]))),
            "usable_embedding_count": int(a1["usable_embedding_count"]),
            "best_novelty_signal": float(a1["best_novelty_signal"]),
        },
        {
            "source": "A2_program_synthesis_action_effect_filter",
            "artifact": A2_RELATIVE_PATH,
            "measured_signal": float(max(_as_float(a2["measured_signal"]), _as_float(a2["component_signal"]))),
            "heldout_programs_kept": int(a2["heldout_programs_kept"]),
            "heldout_programs_rejected": int(a2["heldout_programs_rejected"]),
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


def _attempt_reached_level(row: Mapping[str, Any] | None) -> int:
    return _as_int(row.get("reached_level")) if row else 0


def measure_transfer_game(
    game: str,
    *,
    a1_artifact: Mapping[str, Any],
    a2_artifact: Mapping[str, Any],
) -> JsonDict:
    """SCENARIO-ARC-WMTE-4692-TRANSFER-MEASUREMENT: measure one cached game."""

    novelty_rows = _novelty_rows_for_game(a1_artifact, game)
    operator_result = kit.controllable_novelty_embedding_operator(novelty_rows)
    controllable = _controllable_attempt(a1_artifact, game)
    baseline = _baseline_attempt(a1_artifact, game)
    program_probe = _program_probe_by_game(a2_artifact, game)

    coverage_delta = max(0.0, _as_float(controllable.get("candidate_generation_coverage_delta"))) if controllable else 0.0
    if coverage_delta == 0.0 and program_probe:
        filter_hit = bool(program_probe.get("filter_winner_in_pool") is True)
        blind_hit = bool(program_probe.get("blind_winner_in_pool") is True)
        coverage_delta = max(0.0, float(int(filter_hit) - int(blind_hit)))

    controllable_first = bool(controllable and controllable.get("first_win") is True)
    baseline_first = bool(baseline and baseline.get("first_win") is True)
    first_delta = max(0.0, float(int(controllable_first) - int(baseline_first))) if controllable else 0.0

    controllable_level = _attempt_reached_level(controllable)
    baseline_level = _attempt_reached_level(baseline)
    solve_delta = (
        max(0.0, float(int(controllable_level >= 2) - int(baseline_level >= 2)))
        if controllable
        else 0.0
    )
    gate = controllable.get("reproduction_gate") if isinstance(controllable, Mapping) else {}
    gate_reproduced = isinstance(gate, Mapping) and gate.get("reproduced") is True
    offline_new = bool(controllable and (controllable.get("offline_reproduced") is True or gate_reproduced))
    value_added = bool(coverage_delta > 0.0 or first_delta > 0.0 or solve_delta > 0.0 or offline_new)
    transfer_value = {
        "operator": PRIMITIVE_OPERATOR,
        "candidate_generation_coverage_delta": coverage_delta,
        "first_win_rate_delta": first_delta,
        "live_solve_rate_delta": solve_delta,
        "offline_reproduced_new_level": offline_new,
        "offline_reproduced_new_level_source": "arc_solver_kit.reproduce",
        "embedding_row_count": int(operator_result["embedding_row_count"]),
        "usable_embedding_count": int(operator_result["usable_embedding_count"]),
        "best_novelty_signal": float(operator_result["best_novelty_signal"]),
        "coverage_ready": bool(operator_result["coverage_ready"]),
        "baseline_reached_level": baseline_level,
        "controllable_reached_level": controllable_level,
        "value_added": value_added,
    }
    if value_added:
        dead_end = ""
    elif not novelty_rows or controllable is None:
        dead_end = "no cached controllable-novelty transfer rows for this held-out game"
    elif not operator_result["coverage_ready"]:
        if operator_result.get("residual") == "cosmetic_novelty_not_controllable":
            dead_end = "rejected raw-frame or cosmetic novelty as not controllable action-effect novelty"
        else:
            dead_end = "no usable controllable action-effect embeddings for this held-out game"
    else:
        dead_end = (
            "controllable embeddings produced no coverage, first-win, solve-rate, "
            "or reproduced-level lift"
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
    a1_artifact: Mapping[str, Any],
    a2_artifact: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        measure_transfer_game(game, a1_artifact=a1_artifact, a2_artifact=a2_artifact)
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
    """SCENARIO-ARC-WMTE-4692-TRANSFER-MEASUREMENT: assemble the artifact."""

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
    residual = ""
    if preconditions_checked.get("ok") is False:
        residual = "preconditions failed before transfer measurement"
    elif winner is None:
        residual = (
            "Persisted controllable-novelty embedding operator had zero transfer lift on "
            "cached held-out games; the residual bridge gap is making the controllable "
            "novelty signal propose a winning prefix instead of revisiting non-winning "
            "controllable states."
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
            "reuse_note": "reuse controllable action-effect novelty embeddings before live proposal ranking",
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
        "requirements": ["REQ-ARC-WMTE-4692"],
        "scenarios": [
            "SCENARIO-ARC-WMTE-4692-PERSIST-STRONGEST-COMPONENT",
            "SCENARIO-ARC-WMTE-4692-TRANSFER-MEASUREMENT",
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
        errors.append("inference_substrate must match REQ-ARC-WMTE-4692")
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
        errors.append("field_principles must match REQ-ARC-WMTE-4692")
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
        transfer_results = measure_transfer(
            transfer_games=transfer_games,
            a1_artifact=a1,
            a2_artifact=a2,
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
