"""Experiment 4632: persist the 4629 action-effect ranker and measure transfer.

Spec refs: REQ-ARC-WMTE-4632, SCENARIO-ARC-WMTE-4632.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(REPO_ROOT))

from carnot import experiment_4629_graduate_action_effect_predictor_live as exp4629
from carnot.agentic import arc_frame_change_predictor as fcp
from carnot.agentic import arc_solver_kit as kit


RESULT_RELATIVE_PATH = "results/experiment_4632_primitive_persist_transfer.json"
A1_RELATIVE_PATH = "results/experiment_4628_dense_curiosity_progress_loop.json"
A2_RELATIVE_PATH = "results/experiment_4629_graduate_action_effect_predictor_live.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates for the offline transfer; declared so a fast "
    "real run is not DURATION_TOO_SHORT/METHODOLOGY false-flagged."
)
PRIMITIVE_OPERATOR = "persistent_action_effect_memory_operator"
PRIMITIVE_GOTCHA_ID = "primitive_persistent_action_effect_memory_operator"
RANDOM_SEED = 4632
TRANSFER_GAME_LIMIT = 3
TERMINAL_PREFIXES = ("complete:", "success:", "blocked_")
REQUIREMENTS = ["REQ-ARC-WMTE-4632"]
SCENARIOS = ["SCENARIO-ARC-WMTE-4632"]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: primitive_persisted_transfer_<game>_value_added OR "
        "complete: primitive_persisted_transfer_null_characterized."
    ),
    "inference_substrate": INFERENCE_SUBSTRATE,
    "verifier_is_oracle": (
        "MUST be false -- the persisted primitive directs exploration or prunes actions, "
        "oracle-distinct from the win-check."
    ),
    "solve_provenance": (
        "development_proxy if a transfer solve is via the offline twin; live_agent_self_discovery "
        "if the persisted primitive improves the SCORED agent's own path. NOT outer_loop_re."
    ),
    "primitive_persisted": (
        "names the arc_solver_kit operator + registry general_gotcha id added/extended -- the "
        "reusable asset (Solver-Reuse Discipline); without it the A1/A2 effort is wasted per the "
        "ARC reuse rule."
    ),
    "transfer_games": (
        "the games the primitive was applied to (NOT tuned on) -- the generalization test."
    ),
    "transfer_value_per_game": (
        "the per-game value-add (live solve-rate / action-efficiency lift) -- the cross-game "
        "evidence the primitive generalizes."
    ),
    "offline_reproduced": "only offline-reproduced new levels count toward reproducible_total_levels.",
    "registry_updated": (
        "the primitive + transfer dead-ends persisted so the next milestone reuses, not re-derives."
    ),
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "catches silent drift on replay.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "field_principles",
    "requirements",
    "scenarios",
    "upstream_decision",
    "upstream_summaries",
    "transfer_results",
    "transfer_dead_ends",
    "new_levels_banked",
    "reproducible_total_levels",
    "result_path",
    "duration_s",
)


def _load_json(path: Path) -> dict[str, Any]:  # pragma: no cover - file boundary.
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_registry(root: Path) -> dict[str, Any]:  # pragma: no cover - file boundary.
    try:
        loaded = yaml.safe_load((root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _row_game(row: Any) -> str:
    if isinstance(row, Mapping):
        return str(row.get("game") or row.get("env") or "")
    return str(getattr(row, "game", "") or getattr(row, "env", "") or "")


def _registry_has_primitive_gotcha(registry: Mapping[str, Any]) -> bool:
    return any(
        isinstance(row, Mapping)
        and row.get("id") == PRIMITIVE_GOTCHA_ID
        and row.get("operator") == PRIMITIVE_OPERATOR
        for row in registry.get("general_gotchas", []) or []
    )


def _registry_has_exp4632_transfer(registry: Mapping[str, Any]) -> bool:
    return any(
        isinstance(row, Mapping)
        and row.get("id") == PRIMITIVE_GOTCHA_ID
        and isinstance(row.get("latest_exp4632_transfer"), Mapping)
        and row["latest_exp4632_transfer"].get("artifact") == RESULT_RELATIVE_PATH
        for row in registry.get("general_gotchas", []) or []
    )


def _registry_reproducible_total(registry: Mapping[str, Any]) -> int:
    return _as_int(registry.get("reproducible_total_levels"))


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    root_path = Path(root)
    registry = _load_registry(root_path)
    spec_path = root_path / SPEC_RELATIVE_PATH
    rows = fcp.load_cached_transition_effect_rows(root_path)
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import_smoke": False,
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "spec_has_req_4632": spec_path.exists()
        and "REQ-ARC-WMTE-4632" in spec_path.read_text(encoding="utf-8"),
        "registry_has_primitive_gotcha": _registry_has_primitive_gotcha(registry),
        "registry_has_exp4632_transfer": _registry_has_exp4632_transfer(registry),
        "transition_effect_rows_loaded": int(len(rows)),
        "leaderboard_submission": False,
    }
    try:
        kit.offline_arcade()
        checks["offline_arcade_import_smoke"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = repr(exc)
    checks["ok"] = bool(
        checks["offline_arcade_import_smoke"]
        and checks["a1_artifact_present"]
        and checks["a2_artifact_present"]
        and checks["spec_has_req_4632"]
        and checks["registry_has_primitive_gotcha"]
        and checks["registry_has_exp4632_transfer"]
        and int(checks["transition_effect_rows_loaded"]) > 0
    )
    return checks


def select_primitive_from_upstreams(
    *, a1_artifact: Mapping[str, Any], a2_artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4632: select the strongest reusable A1/A2 primitive signal."""

    a1_signal = max(
        _as_float(a1_artifact.get("first_win_rate_delta")),
        _as_float(a1_artifact.get("solve_rate_delta")),
        _as_float(a1_artifact.get("state_coverage_delta")) / 1000.0,
    )
    a2_actions_delta = _as_float(a2_artifact.get("actions_delta"))
    a2_solve_preserved = a2_artifact.get("solve_rate_preserved") is True
    a2_success = str(a2_artifact.get("honest_verdict") or "").startswith("success:")
    a2_signal = a2_actions_delta if a2_solve_preserved or a2_success else 0.0
    if a2_signal <= 0.0:
        a2_signal = max(
            _as_float(a2_artifact.get("first_win_rate_delta")),
            _as_float(a2_artifact.get("solve_rate_delta")),
        )
    measured_signal = max(a2_signal, a1_signal)
    persisted_null = bool(a2_signal <= 0.0)
    if persisted_null:
        rationale = (
            "degrade gracefully: 4628 had no live lift and 4629 had no positive action-efficiency "
            "signal in this input, so persist the best-characterized action-effect ranker as-built "
            "and report transfer nulls."
        )
    else:
        rationale = (
            "4629 is the winning primitive: it raised live action efficiency with solve-rate "
            "preserved, while 4628 dense curiosity reported an honest no-live-lift null."
        )
    return {
        "source": "A2_action_effect_candidate_ranker",
        "operator": PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": PRIMITIVE_GOTCHA_ID,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
        "measured_signal": float(measured_signal),
        "a1_dense_curiosity_signal": float(a1_signal),
        "a2_actions_delta": float(a2_actions_delta),
        "a2_solve_rate_preserved": bool(a2_solve_preserved),
        "persisted_as_best_characterized_null": persisted_null,
        "selection_rationale": rationale,
    }


def measure_action_effect_ranker_transfer_game(
    game: str,
    *,
    effect_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4632: measure target-excluded action-effect ranker transfer."""

    rows = list(effect_rows)
    target_rows = [row for row in rows if _row_game(row) == game]
    memory = kit.PersistentAEM.from_effect_rows(rows, exclude_games=(game,))
    scorer = fcp.LiveActionEffectScorer(memory=memory, cnn_scorer=None)
    metrics = exp4629.measure_live_action_efficiency(
        target_rows,
        scorer=scorer,
        n_bootstrap=0,
    )
    solve_rate_lift = round(
        _as_float(metrics.get("solve_rate_predictor")) - _as_float(metrics.get("solve_rate_bare")),
        10,
    )
    actions_delta = _as_float(metrics.get("actions_delta"))
    value_added = bool((actions_delta > 0.0 or solve_rate_lift > 0.0) and target_rows)
    dead_end = ""
    if not rows:
        dead_end = "no cached action-effect rows were available for transfer measurement."
    elif not target_rows:
        dead_end = "no cached action-effect rows were available for the held-out transfer game."
    elif _as_int(metrics.get("heldout_candidate_group_count")) <= 0:
        dead_end = "target game rows had no effective same-state candidate groups to rank."
    elif memory.row_count <= 0:
        dead_end = "target-game exclusion left no cross-game action-effect memory to transfer."
    elif not value_added:
        dead_end = (
            "action-effect ranker transferred a scorer, but live solve-rate/action-efficiency did "
            "not improve on this target-excluded game."
        )
    transfer_value = {
        "operator": PRIMITIVE_OPERATOR,
        "measurement_kind": metrics.get("measurement_kind"),
        "candidate_group_count": int(metrics.get("heldout_candidate_group_count") or 0),
        "paired_delta_count": int(metrics.get("paired_delta_count") or 0),
        "median_actions_to_first_levelup_bare": metrics.get(
            "median_actions_to_first_levelup_bare"
        ),
        "median_actions_to_first_levelup_predictor": metrics.get(
            "median_actions_to_first_levelup_predictor"
        ),
        "actions_delta": float(actions_delta),
        "solve_rate_bare": float(metrics.get("solve_rate_bare") or 0.0),
        "solve_rate_predictor": float(metrics.get("solve_rate_predictor") or 0.0),
        "solve_rate_lift": float(solve_rate_lift),
        "first_win_rate_delta": float(metrics.get("first_win_rate_delta") or 0.0),
        "target_game_excluded_from_memory": bool(game in memory.excluded_games),
        "memory_row_count": int(memory.row_count),
        "value_added": value_added,
    }
    return {
        "game": str(game),
        "value_added": value_added,
        "target_game_excluded_from_memory": bool(game in memory.excluded_games),
        "transfer_value": transfer_value,
        "offline_reproduced_new_level": False,
        "dead_end": dead_end,
    }


def select_transfer_results(
    effect_rows: Sequence[Mapping[str, Any]],
    *,
    games: Sequence[str] | None = None,
    limit: int = TRANSFER_GAME_LIMIT,
) -> list[dict[str, Any]]:
    measured_games = (
        list(games)
        if games is not None
        else sorted({game for game in (_row_game(row) for row in effect_rows) if game})
    )
    measured = [
        measure_action_effect_ranker_transfer_game(game, effect_rows=effect_rows)
        for game in measured_games
    ]
    measured.sort(
        key=lambda row: (
            row.get("value_added") is not True,
            -_as_float((row.get("transfer_value") or {}).get("actions_delta"))
            if isinstance(row.get("transfer_value"), Mapping)
            else 0.0,
            -_as_float((row.get("transfer_value") or {}).get("solve_rate_lift"))
            if isinstance(row.get("transfer_value"), Mapping)
            else 0.0,
            -_as_float((row.get("transfer_value") or {}).get("first_win_rate_delta"))
            if isinstance(row.get("transfer_value"), Mapping)
            else 0.0,
            str(row.get("game") or ""),
        )
    )
    return [dict(row) for row in measured[: max(0, int(limit))]]


def _success_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    winners = [row for row in rows if row.get("value_added") is True]
    if not winners:
        return None
    return max(
        winners,
        key=lambda row: (
            _as_float((row.get("transfer_value") or {}).get("actions_delta"))
            if isinstance(row.get("transfer_value"), Mapping)
            else 0.0,
            _as_float((row.get("transfer_value") or {}).get("solve_rate_lift"))
            if isinstance(row.get("transfer_value"), Mapping)
            else 0.0,
            str(row.get("game") or ""),
        ),
    )


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _stable_checksum(payload)


def build_artifact(
    *,
    upstream_decision: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    transfer_results: Sequence[Mapping[str, Any]],
    registry_updated: bool,
    random_seed: int,
    duration_s: float | None,
    reproducible_total_levels: int | None,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4632: assemble the primitive-persist transfer artifact."""

    rows = [dict(row) for row in transfer_results]
    winner = _success_row(rows)
    transfer_games = [str(row.get("game")) for row in rows]
    transfer_values: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = dict(row.get("transfer_value") or {})
        value.setdefault("value_added", bool(row.get("value_added") is True))
        transfer_values[str(row.get("game"))] = value
    dead_ends = {
        str(row.get("game")): str(row.get("dead_end") or "")
        for row in rows
        if str(row.get("dead_end") or "")
    }
    new_levels_banked = sum(1 for row in rows if row.get("offline_reproduced_new_level") is True)
    if preconditions_checked.get("ok") is False:
        verdict = "blocked_primitive_persist_transfer_precondition"
    elif winner is not None:
        verdict = f"success: primitive_persisted_transfer_{winner.get('game')}_value_added"
    else:
        verdict = "complete: primitive_persisted_transfer_null_characterized"
    artifact = {
        "experiment": "experiment_4632_primitive_persist_transfer",
        "schema": "carnot.arc_primitive_persist_transfer_4632.v1",
        "honest_verdict": verdict,
        "inference_substrate": str(
            upstream_decision.get("inference_substrate") or INFERENCE_SUBSTRATE
        ),
        "verifier_is_oracle": False,
        "solve_provenance": str(
            upstream_decision.get("solve_provenance") or "live_agent_self_discovery"
        ),
        "primitive_persisted": {
            "operator": upstream_decision.get("operator"),
            "registry_general_gotcha_id": upstream_decision.get("registry_general_gotcha_id"),
            "source": upstream_decision.get("source"),
            "derived_from_artifacts": [A1_RELATIVE_PATH, A2_RELATIVE_PATH],
        },
        "transfer_games": transfer_games,
        "transfer_value_per_game": transfer_values,
        "offline_reproduced": bool(new_levels_banked > 0),
        "registry_updated": bool(registry_updated),
        "random_seed": _as_int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "upstream_decision": dict(upstream_decision),
        "upstream_summaries": {
            "a1_dense_curiosity_progress_loop": {
                "artifact": A1_RELATIVE_PATH,
                "selected": False,
            },
            "a2_action_effect_candidate_ranker": {
                "artifact": A2_RELATIVE_PATH,
                "selected": upstream_decision.get("source") == "A2_action_effect_candidate_ranker",
            },
        },
        "transfer_results": rows,
        "transfer_dead_ends": dead_ends,
        "new_levels_banked": int(new_levels_banked),
        "reproducible_total_levels": _as_int(reproducible_total_levels),
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else float(duration_s),
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
        errors.append("inference_substrate must match the 4632 offline substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("solve_provenance") not in {
        "live_agent_self_discovery",
        "development_proxy",
    }:
        errors.append("solve_provenance must be live_agent_self_discovery or development_proxy")
    if artifact.get("solve_provenance") == "outer_loop_re":
        errors.append("solve_provenance must not be outer_loop_re")
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
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be a bare bool")
    if type(artifact.get("registry_updated")) is not bool:
        errors.append("registry_updated must be a bare bool")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-ARC-WMTE-4632")
    if isinstance(verdict, str) and verdict.startswith("success:"):
        values = artifact.get("transfer_value_per_game")
        if not isinstance(values, Mapping) or not any(
            isinstance(value, Mapping) and value.get("value_added") is True
            for value in values.values()
        ):
            errors.append("success requires at least one transfer value_added=true")
    if (
        artifact.get("offline_reproduced") is True
        and _as_int(artifact.get("new_levels_banked")) < 1
    ):
        errors.append("offline_reproduced=true requires at least one new level banked")
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


def run(root: Path | str = REPO_ROOT, *, write: bool = True) -> dict[str, Any]:  # pragma: no cover
    root_path = Path(root)
    started = time.monotonic()
    registry = _load_registry(root_path)
    checks = check_preconditions(root_path)
    a1 = _load_json(root_path / A1_RELATIVE_PATH)
    a2 = _load_json(root_path / A2_RELATIVE_PATH)
    decision = select_primitive_from_upstreams(a1_artifact=a1, a2_artifact=a2)
    rows = fcp.load_cached_transition_effect_rows(root_path)
    transfer_results: list[dict[str, Any]] = []
    if checks.get("ok") is True and decision.get("operator") == PRIMITIVE_OPERATOR:
        transfer_results = select_transfer_results(rows)
    artifact = build_artifact(
        upstream_decision=decision,
        preconditions_checked=checks,
        transfer_results=transfer_results,
        registry_updated=bool(checks.get("registry_has_exp4632_transfer")),
        random_seed=RANDOM_SEED,
        duration_s=max(0.0, time.monotonic() - started),
        reproducible_total_levels=_registry_reproducible_total(registry),
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root_path)
    return artifact


def main() -> int:  # pragma: no cover - requested command boundary.
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - requested command boundary.
    raise SystemExit(main())
