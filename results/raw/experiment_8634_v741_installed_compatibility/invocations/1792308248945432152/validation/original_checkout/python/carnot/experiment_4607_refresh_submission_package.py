"""Experiment 4607: refresh the ARC operator-resubmit package.

Spec refs: REQ-CAPSTONE-4607, SCENARIO-CAPSTONE-4607,
SCENARIO-CAPSTONE-4607-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot import experiment_4595_refresh_submission_package as base


JsonDict = dict[str, Any]
ActionList = list[JsonDict]

EXPERIMENT = "experiment_4607_refresh_submission_package"
SCHEMA = "carnot.exp4607.refresh_submission_package.v1"
PACKAGE_SCHEMA = "carnot.exp4607.submission_package.v1"
RANDOM_SEED = 4607

RESULT_RELATIVE_PATH = "results/experiment_4607_refresh_submission_package.json"
PACKAGE_RELATIVE_PATH = "results/experiment_4607_submission_package_operator_resubmit.json"
BASELINE_RESULT_RELATIVE_PATH = "results/experiment_4595_refresh_submission_package.json"
BASELINE_PACKAGE_RELATIVE_PATH = "results/experiment_4595_submission_package_operator_resubmit.json"
A1_WORLD_MODEL_RELATIVE_PATH = "results/experiment_4604_world_model_trust_energy.json"
A2_LIVE_INTEGRATION_RELATIVE_PATH = "results/experiment_4605_live_integration_scored_agent.json"
A3_LEVELUP_RELATIVE_PATH = "results/experiment_4606_levelup_selfplay.json"
REGISTRY_RELATIVE_PATH = base.REGISTRY_RELATIVE_PATH
BANK_DIR_RELATIVE_PATH = base.BANK_DIR_RELATIVE_PATH

LIVE_SUBMITTABLE_PREV = 55
SUBMISSION_SCORE_GATE = 33
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline reproduction-gated packaging, "
    "no LLM load (1s floor)."
)
SPEC_REFS = [
    "REQ-CAPSTONE-4607",
    "SCENARIO-CAPSTONE-4607",
    "SCENARIO-CAPSTONE-4607-FIELD-PRINCIPLES",
]
TERMINAL_PREFIXES = base.TERMINAL_PREFIXES

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "live_submittable_level_count",
    "live_submittable_count_prev",
    "count_delta",
    "levels_folded_in",
    "refreshed_package_path",
    "per_game_submittable",
    "ready_for_operator_submit",
    "offline_reproduced",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: package_refreshed_live_submittable_<n>_above_33 "
            "OR complete: package_refreshed_unchanged_depth."
        )
    },
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- packaging + env-adaptive replay bank already-earned solves "
            "(no circular moat claim)."
        )
    },
    "live_submittable_level_count": {
        "principle": (
            "the offline-reproduction-gated count of levels with a replayable trajectory or "
            "env-adaptive re-solver (the honest leaderboard score, must stay > 33)."
        )
    },
    "live_submittable_count_prev": {
        "principle": "55 -- the .424 A4 count, the apples-to-apples comparison."
    },
    "count_delta": {
        "principle": (
            "live_submittable_level_count - 55 (>=0; positive = more submittable levels "
            "folded in), emitted explicitly so a null is annotated."
        )
    },
    "levels_folded_in": {
        "principle": (
            "names the games whose new banks (A3 + A1/A2 level solves) were folded into "
            "the refreshed package this milestone."
        )
    },
    "refreshed_package_path": {
        "principle": (
            "the path to the refreshed validated package the operator's live-submit driver "
            "will load -- the deliverable the operator resubmits."
        )
    },
    "per_game_submittable": {
        "principle": (
            "per-game offline-reproduced level + has-trajectory + drift-robust flags -- the "
            "audit trail that no level is over-claimed above its offline depth."
        )
    },
    "ready_for_operator_submit": {
        "principle": (
            "True if the refreshed package's live-submittable count beats 33 and every claim "
            "is offline-reproduction-gated; the task NEVER submits (operator-only)."
        )
    },
    "offline_reproduced": {
        "principle": (
            "every claimed-submittable level must offline-reproduce to count (no "
            "frozen-trajectory over-claim)."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent registry/trajectory drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (offline arcade, registry loadable); pre-empts "
            "missing-resource fabrication."
        )
    },
}

BASE_ENV_ADAPTIVE_RESOLVERS = {
    "sc25": "sc25_dynamic_cast_grid_origin_step",
    "s5i5": "env_adaptive_resolve_operator:s5i5",
    "ft09": "env_adaptive_resolve_operator:ft09",
    "sb26": "env_adaptive_resolve_operator:sb26",
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _as_int(value: Any, default: int = 0) -> int:
    return base._as_int(value, default)


def _load_json(path: Path) -> JsonDict:
    return base._load_json(path)


def _load_yaml(path: Path) -> JsonDict:
    return base._load_yaml(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    base._write_json(path, payload)


def _package_rows(package: Mapping[str, Any]) -> dict[str, JsonDict]:
    return base._package_rows(package)


def _registry_levels(registry: Mapping[str, Any]) -> dict[str, int]:
    return base._registry_levels(registry)


def _row_order(
    registry_levels: Mapping[str, int], package_by_game: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    return base._row_order(registry_levels, package_by_game)


def _trajectory_actions(root: Path, trajectory_path: str) -> ActionList:
    return base._trajectory_actions(root, trajectory_path)


def _terminal_prefixed(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(TERMINAL_PREFIXES)


def _solution_label_actions(labels: Any) -> ActionList:
    actions: ActionList = []
    if not isinstance(labels, list):
        return actions
    for item in labels:
        parsed: Any = item
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, Mapping):
            actions.append(dict(parsed))
    return actions


def _candidate_actions(payload: Mapping[str, Any]) -> ActionList:
    actions = base._actions_from_payload(payload)
    if actions:
        return actions
    return _solution_label_actions(payload.get("solution_labels"))


def _candidate_level(payload: Mapping[str, Any], *, fallback: int = 0) -> int:
    gate = payload.get("reproduction_gate")
    gate = gate if isinstance(gate, Mapping) else {}
    for value in (
        gate.get("reached_level"),
        payload.get("reached_level"),
        payload.get("reproduced_levels"),
    ):
        level = _as_int(value)
        if level > 0:
            return level
    return int(fallback)


def _is_candidate_reproduced(payload: Mapping[str, Any], target_level: int) -> bool:
    gate = payload.get("reproduction_gate")
    gate = gate if isinstance(gate, Mapping) else {}
    reached = _candidate_level(payload, fallback=target_level)
    if gate:
        return bool(gate.get("reproduced") is True and reached >= target_level)
    return bool(payload.get("offline_reproduced") is True and reached >= target_level)


def _write_banked_trajectory(
    root: Path, game: str, actions: Sequence[Mapping[str, Any]], source: str
) -> str:
    relative = f"{BANK_DIR_RELATIVE_PATH}/{game}.json"
    payload = {
        "schema": "carnot.arc3.flat_trajectory_bank.v1",
        "game": game,
        "source": source,
        "action_count": len(actions),
        "solution": [dict(action) for action in actions],
    }
    _write_json(root / relative, payload)
    return relative


def _env_adaptive_resolvers(registry: Mapping[str, Any]) -> dict[str, str]:
    resolvers = dict(BASE_ENV_ADAPTIVE_RESOLVERS)
    gotchas = registry.get("general_gotchas")
    if not isinstance(gotchas, list):
        return resolvers
    for gotcha in gotchas:
        if (
            not isinstance(gotcha, Mapping)
            or gotcha.get("id") != "primitive_env_adaptive_resolve_operator"
        ):
            continue
        latest = gotcha.get("latest_exp4584_transfer")
        latest = latest if isinstance(latest, Mapping) else {}
        for game in latest.get("transfer_games") or []:
            if isinstance(game, str) and game:
                resolvers.setdefault(game, f"env_adaptive_resolve_operator:{game}")
    return resolvers


def check_preconditions(
    root: Path,
    *,
    offline_arcade_checker: Callable[[], bool] | None = None,
) -> JsonDict:
    checker = offline_arcade_checker or _default_offline_arcade_checker
    checked: JsonDict = {
        "offline_arcade": {"ok": False, "resource": "arc_solver_kit.offline_arcade"},
        "registry_loadable": {"ok": False, "path": REGISTRY_RELATIVE_PATH},
    }
    try:
        checked["offline_arcade"]["ok"] = bool(checker())
    except Exception as exc:
        checked["offline_arcade"]["error"] = str(exc)
    registry = _load_yaml(root / REGISTRY_RELATIVE_PATH)
    levels = _registry_levels(registry)
    checked["registry_loadable"]["ok"] = bool(levels)
    checked["registry_loadable"]["reproduced_game_count"] = len(levels)
    return checked


def first_precondition_miss(checked: Mapping[str, Any]) -> str | None:
    offline = checked.get("offline_arcade")
    registry = checked.get("registry_loadable")
    if not (isinstance(offline, Mapping) and offline.get("ok") is True):
        return "offline_arcade"
    if not (isinstance(registry, Mapping) and registry.get("ok") is True):
        return "registry"
    return None


def _default_offline_arcade_checker() -> bool:  # pragma: no cover - SDK boundary
    from carnot.agentic import arc_solver_kit as kit

    kit.offline_arcade()
    return True


def _previous_live_count(previous_result: Mapping[str, Any]) -> int:
    return _as_int(previous_result.get("live_submittable_level_count"), LIVE_SUBMITTABLE_PREV)


def _candidate(
    *,
    game: str,
    source_artifact: str,
    payload: Mapping[str, Any],
    source_payload_path: str = "",
    level: int | None = None,
) -> JsonDict:
    target = int(level if level is not None else _candidate_level(payload, fallback=1))
    return {
        "game": game,
        "source_artifact": source_artifact,
        "source_payload_path": source_payload_path,
        "target_level": target,
        "offline_reproduced": _is_candidate_reproduced(payload, target),
        "payload": dict(payload),
    }


def collect_upstream_candidates(root: Path) -> dict[str, list[JsonDict]]:
    out: dict[str, list[JsonDict]] = {}

    a3 = _load_json(root / A3_LEVELUP_RELATIVE_PATH)
    game = str(a3.get("target_game") or "")
    if game:
        out.setdefault(game, []).append(
            _candidate(
                game=game,
                source_artifact=A3_LEVELUP_RELATIVE_PATH,
                source_payload_path=str(a3.get("standing_loop_result_path") or ""),
                payload=a3,
                level=_candidate_level(a3),
            )
        )

    a1 = _load_json(root / A1_WORLD_MODEL_RELATIVE_PATH)
    measurements = a1.get("measurements")
    if isinstance(measurements, list):
        for row in measurements:
            if not isinstance(row, Mapping) or not row.get("game"):
                continue
            if row.get("offline_reproduced") is True and row.get("new_first_win") is True:
                game = str(row["game"])
                out.setdefault(game, []).append(
                    _candidate(
                        game=game,
                        source_artifact=A1_WORLD_MODEL_RELATIVE_PATH,
                        payload=row,
                        level=1,
                    )
                )

    a2 = _load_json(root / A2_LIVE_INTEGRATION_RELATIVE_PATH)
    integrated = a2.get("integrated_measurement")
    integrated = integrated if isinstance(integrated, Mapping) else {}
    attempts = integrated.get("variant_attempts")
    if isinstance(attempts, list):
        for row in attempts:
            if not isinstance(row, Mapping) or not row.get("game"):
                continue
            game = str(row["game"])
            out.setdefault(game, []).append(
                _candidate(
                    game=game, source_artifact=A2_LIVE_INTEGRATION_RELATIVE_PATH, payload=row
                )
            )
    return out


def _candidate_payload(root: Path, candidate: Mapping[str, Any]) -> tuple[JsonDict, str]:
    payload_path = str(candidate.get("source_payload_path") or "")
    if payload_path:
        payload = _load_json(root / payload_path)
        if payload:
            return payload, payload_path
    payload = candidate.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else {}, str(
        candidate.get("source_artifact") or ""
    )


def _refreshed_actions(
    root: Path,
    *,
    game: str,
    registry_level: int,
    previous_level: int,
    previous_row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[ActionList, str, str, int, list[JsonDict]]:
    trajectory_path = str(previous_row.get("trajectory_path") or "")
    previous_actions = _trajectory_actions(root, trajectory_path)
    source = str(previous_row.get("source") or trajectory_path)
    trajectory_level = previous_level if previous_actions else 0
    refreshes: list[JsonDict] = []

    for candidate in sorted(
        candidates, key=lambda item: _as_int(item.get("target_level")), reverse=True
    ):
        target_level = min(registry_level, _as_int(candidate.get("target_level")))
        if target_level <= trajectory_level or candidate.get("offline_reproduced") is not True:
            continue
        payload, payload_source = _candidate_payload(root, candidate)
        if not _is_candidate_reproduced(payload, target_level):
            continue
        actions = _candidate_actions(payload)
        if not actions:
            continue
        trajectory_path = _write_banked_trajectory(root, game, actions, payload_source)
        refresh = {
            "game": game,
            "source": payload_source,
            "trajectory_path": trajectory_path,
            "action_count": len(actions),
            "reached_level": target_level,
        }
        refreshes.append(refresh)
        return actions, trajectory_path, payload_source, target_level, refreshes
    return previous_actions, trajectory_path, source, trajectory_level, refreshes


def _candidate_audit(
    *,
    candidates_by_game: Mapping[str, Sequence[Mapping[str, Any]]],
    registry_levels: Mapping[str, int],
    previous_by_game: Mapping[str, Mapping[str, Any]],
    rows_by_game: Mapping[str, Mapping[str, Any]],
    folded: Sequence[str],
) -> list[JsonDict]:
    folded_set = set(folded)
    audit: list[JsonDict] = []
    for game in sorted(candidates_by_game):
        previous_level = _as_int(previous_by_game.get(game, {}).get("levels"))
        registry_level = _as_int(registry_levels.get(game))
        row = rows_by_game.get(game, {})
        for candidate in candidates_by_game[game]:
            offline = candidate.get("offline_reproduced") is True
            target = _as_int(candidate.get("target_level"))
            if not offline:
                reason = "upstream_not_offline_reproduced"
            elif registry_level <= 0:
                reason = "not_in_current_registry"
            elif target > registry_level:
                reason = "above_current_registry_depth"
            elif game in folded_set:
                reason = "folded"
            elif registry_level <= previous_level:
                reason = "already_submittable"
            elif _as_int(row.get("submittable_level")) <= previous_level:
                reason = "no_new_replayable_depth"
            else:
                reason = "covered_without_delta"
            audit.append(
                {
                    "game": game,
                    "source_artifact": str(candidate.get("source_artifact") or ""),
                    "target_level": target,
                    "registry_reproduced_level": registry_level,
                    "previous_package_level": previous_level,
                    "offline_reproduced": offline,
                    "folded": game in folded_set and offline,
                    "reason": reason,
                }
            )
    return audit


def build_refreshed_rows(
    root: Path,
    *,
    registry: Mapping[str, Any],
    previous_package: Mapping[str, Any],
    candidates_by_game: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[list[JsonDict], list[str], list[JsonDict], dict[str, JsonDict], list[JsonDict]]:
    registry_levels = _registry_levels(registry)
    previous_by_game = _package_rows(previous_package)
    candidates_by_game = candidates_by_game or collect_upstream_candidates(root)
    adaptive_resolvers = _env_adaptive_resolvers(registry)
    rows: list[JsonDict] = []
    folded: list[str] = []
    trajectory_refreshes: list[JsonDict] = []
    claimed_caps: dict[str, JsonDict] = {}

    for game in _row_order(registry_levels, previous_by_game):
        registry_level = registry_levels[game]
        previous_row = previous_by_game.get(game, {})
        previous_level = _as_int(
            previous_row.get("levels") or previous_row.get("submittable_level")
        )
        previous_offline = _as_int(previous_row.get("offline_reproduced_level"))
        actions, trajectory_path, source, trajectory_level, refreshes = _refreshed_actions(
            root,
            game=game,
            registry_level=registry_level,
            previous_level=previous_level,
            previous_row=previous_row,
            candidates=candidates_by_game.get(game, ()),
        )
        trajectory_refreshes.extend(refreshes)
        adaptive_solver = str(previous_row.get("adaptive_solver") or "")
        if game in adaptive_resolvers:
            adaptive_solver = adaptive_resolvers[game]
        has_adaptive = bool(adaptive_solver)
        path_level = max(trajectory_level, registry_level if has_adaptive else 0)
        has_replayable_for_claim = bool(actions and trajectory_level > 0)
        env_matchable = bool(
            previous_row.get("env_matched") is True
            or previous_row.get("env_match") is True
            or previous_row.get("env_match_basis")
            or has_replayable_for_claim
            or has_adaptive
        )
        submittable = min(registry_level, path_level) if env_matchable else 0
        has_submission_path = bool(
            (has_replayable_for_claim and submittable <= trajectory_level) or has_adaptive
        )
        if not has_submission_path:
            submittable = 0
        exclusion_reason = "" if has_submission_path else "missing_trajectory_or_adaptive_resolver"
        if submittable > previous_level:
            folded.append(game)
        claim_capped = bool(previous_level > registry_level or previous_offline > registry_level)
        claim_capped = claim_capped or (
            registry_level > submittable and registry_level > previous_level
        )
        claimed_caps[game] = {
            "package_claimed_level": previous_level,
            "package_offline_reproduced_level": previous_offline,
            "registry_reproduced_level": registry_level,
            "capped_claimed_level": submittable,
            "cap_applied": claim_capped or submittable < previous_level,
        }
        rows.append(
            {
                "game": game,
                "previous_package_level": previous_level,
                "claimed_level": submittable,
                "registry_reproduced_level": registry_level,
                "offline_reproduced_level": registry_level,
                "submittable_level": submittable,
                "has_trajectory": bool(actions),
                "has_replayable_trajectory": bool(
                    actions and submittable > 0 and submittable <= trajectory_level
                ),
                "trajectory_reaches_level": trajectory_level,
                "has_env_adaptive_resolver": has_adaptive,
                "drift_robust": has_adaptive,
                "env_matchable": env_matchable,
                "trajectory_path": trajectory_path,
                "trajectory_action_count": len(actions),
                "adaptive_solver": adaptive_solver,
                "source": source,
                "claim_capped": bool(claim_capped),
                "exclusion_reason": exclusion_reason,
            }
        )
    rows_by_game = {str(row["game"]): row for row in rows}
    audit = _candidate_audit(
        candidates_by_game=candidates_by_game,
        registry_levels=registry_levels,
        previous_by_game=previous_by_game,
        rows_by_game=rows_by_game,
        folded=folded,
    )
    return rows, sorted(dict.fromkeys(folded)), trajectory_refreshes, claimed_caps, audit


def build_package_payload(rows: Sequence[Mapping[str, Any]], *, result_path: str) -> JsonDict:
    manifest: list[JsonDict] = []
    for row in rows:
        levels = _as_int(row.get("submittable_level"))
        if levels <= 0:
            continue
        manifest.append(
            {
                "game": str(row.get("game") or ""),
                "levels": levels,
                "offline_reproduced_level": _as_int(row.get("offline_reproduced_level")),
                "registry_reproduced_level": _as_int(row.get("registry_reproduced_level")),
                "trajectory_path": str(row.get("trajectory_path") or ""),
                "action_count": _as_int(row.get("trajectory_action_count")),
                "source": str(row.get("source") or ""),
                "env_matched": True,
                "env_match_basis": "offline_reproduction_gated_package_refresh_4607",
                "has_trajectory": bool(row.get("has_replayable_trajectory")),
                "has_env_adaptive_resolver": bool(row.get("has_env_adaptive_resolver")),
                "adaptive_solver": str(row.get("adaptive_solver") or ""),
                "claim_capped": bool(row.get("claim_capped")),
            }
        )
    return {
        "experiment": "experiment_4607_submission_package_operator_resubmit",
        "schema": PACKAGE_SCHEMA,
        "source_result_path": result_path,
        "package_manifest": manifest,
        "claimed_total_levels": sum(_as_int(row.get("levels")) for row in manifest),
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "reproducibility_checksum": _sha256(manifest),
    }


def _ready_for_operator(rows: Sequence[Mapping[str, Any]], live_count: int) -> bool:
    every_claim_gated = all(
        _as_int(row.get("submittable_level")) <= _as_int(row.get("offline_reproduced_level"))
        and _as_int(row.get("submittable_level")) <= _as_int(row.get("registry_reproduced_level"))
        and (
            _as_int(row.get("submittable_level")) == 0
            or row.get("has_replayable_trajectory") is True
            or row.get("has_env_adaptive_resolver") is True
        )
        for row in rows
    )
    return bool(live_count > SUBMISSION_SCORE_GATE and every_claim_gated)


def _honest_verdict(live_count: int, count_delta: int) -> str:
    if live_count > SUBMISSION_SCORE_GATE and count_delta > 0:
        return f"success: package_refreshed_live_submittable_{live_count}_above_33"
    return "complete: package_refreshed_unchanged_depth."


def compute_reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    return _sha256(
        {
            "live_submittable_level_count": artifact.get("live_submittable_level_count"),
            "live_submittable_count_prev": artifact.get("live_submittable_count_prev"),
            "count_delta": artifact.get("count_delta"),
            "levels_folded_in": artifact.get("levels_folded_in"),
            "per_game_submittable": artifact.get("per_game_submittable"),
            "claimed_caps": artifact.get("claimed_caps"),
            "trajectory_refreshes": artifact.get("trajectory_refreshes"),
            "env_adaptive_recovery": artifact.get("env_adaptive_recovery"),
            "upstream_fold_audit": artifact.get("upstream_fold_audit"),
            "random_seed": artifact.get("random_seed"),
        }
    )


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    previous_count: int,
    levels_folded_in: Sequence[str],
    trajectory_refreshes: Sequence[Mapping[str, Any]],
    claimed_caps: Mapping[str, Mapping[str, Any]],
    upstream_fold_audit: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    live_count = sum(_as_int(row.get("submittable_level")) for row in rows)
    count_delta = live_count - previous_count
    ready = _ready_for_operator(rows, live_count)
    package_payload = build_package_payload(rows, result_path=RESULT_RELATIVE_PATH)
    env_adaptive_recovery = [
        {
            "game": str(row.get("game")),
            "adaptive_solver": str(row.get("adaptive_solver") or ""),
            "drift_robust": bool(row.get("drift_robust")),
        }
        for row in rows
        if row.get("has_env_adaptive_resolver") and _as_int(row.get("submittable_level")) > 0
    ]
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": _honest_verdict(live_count, count_delta),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "live_submittable_level_count": live_count,
        "live_submittable_count_prev": previous_count,
        "count_delta": count_delta,
        "levels_folded_in": list(levels_folded_in),
        "refreshed_package_path": PACKAGE_RELATIVE_PATH,
        "per_game_submittable": [dict(row) for row in rows],
        "ready_for_operator_submit": ready,
        "offline_reproduced": ready,
        "offline_reproduction_by_game": {
            str(row.get("game")): _as_int(row.get("offline_reproduced_level"))
            for row in rows
            if _as_int(row.get("submittable_level")) > 0
        },
        "random_seed": RANDOM_SEED,
        "preconditions_checked": dict(preconditions_checked),
        "null_delta_methodology_note": (
            "count_delta==0 is an unchanged-depth refresh measured against the same .424 A4 count."
            if count_delta == 0
            else ""
        ),
        "submitted_to_leaderboard": False,
        "package_manifest": package_payload["package_manifest"],
        "claimed_caps": {key: dict(value) for key, value in claimed_caps.items()},
        "trajectory_refreshes": [dict(item) for item in trajectory_refreshes],
        "env_adaptive_recovery": env_adaptive_recovery,
        "upstream_fold_audit": [dict(item) for item in upstream_fold_audit],
        "duration_s": max(0.0, round(float(duration_s), 6)),
        "result_path": RESULT_RELATIVE_PATH,
    }
    artifact["reproducibility_checksum"] = compute_reproducibility_checksum(artifact)
    return artifact


def _blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": f"blocked_{reason}",
        "inference_substrate": "precondition_check_no_inference",
        "verifier_is_oracle": False,
        "live_submittable_level_count": 0,
        "live_submittable_count_prev": LIVE_SUBMITTABLE_PREV,
        "count_delta": -LIVE_SUBMITTABLE_PREV,
        "levels_folded_in": [],
        "refreshed_package_path": "",
        "per_game_submittable": [],
        "ready_for_operator_submit": False,
        "offline_reproduced": False,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": dict(preconditions_checked),
        "null_delta_methodology_note": "",
        "submitted_to_leaderboard": False,
        "package_manifest": [],
        "claimed_caps": {},
        "trajectory_refreshes": [],
        "env_adaptive_recovery": [],
        "upstream_fold_audit": [],
        "duration_s": max(0.0, round(float(duration_s), 6)),
        "result_path": RESULT_RELATIVE_PATH,
    }
    artifact["reproducibility_checksum"] = compute_reproducibility_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    if not _terminal_prefixed(artifact.get("honest_verdict")):
        errors.append("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") not in {
        INFERENCE_SUBSTRATE,
        "precondition_check_no_inference",
    }:
        errors.append("inference_substrate must equal the declared offline packaging substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    for field in (
        "live_submittable_level_count",
        "live_submittable_count_prev",
        "count_delta",
        "random_seed",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
    for field in ("levels_folded_in", "per_game_submittable"):
        if not isinstance(artifact.get(field), list):
            errors.append(f"{field} must be list")
    if type(artifact.get("ready_for_operator_submit")) is not bool:
        errors.append("ready_for_operator_submit must be bare bool")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be mapping")
    if not _sha256_hex(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be sha256 hex")

    live = artifact.get("live_submittable_level_count")
    prev = artifact.get("live_submittable_count_prev")
    delta = artifact.get("count_delta")
    if type(live) is int and type(prev) is int and type(delta) is int:
        if delta != live - prev:
            errors.append(
                "count_delta must equal live_submittable_level_count - live_submittable_count_prev"
            )
        if artifact.get("ready_for_operator_submit") is True and live <= SUBMISSION_SCORE_GATE:
            errors.append("ready_for_operator_submit requires count strictly above 33")
        if delta == 0 and not artifact.get("null_delta_methodology_note"):
            errors.append("null_delta_methodology_note required when count_delta is zero")

    rows = artifact.get("per_game_submittable")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                errors.append("per_game_submittable rows must be mappings")
                continue
            game = str(row.get("game") or "<unknown>")
            submittable = _as_int(row.get("submittable_level"))
            offline = _as_int(row.get("offline_reproduced_level"))
            registry = _as_int(row.get("registry_reproduced_level"))
            if submittable > offline:
                errors.append(f"{game} submittable exceeds offline reproduction")
            if submittable > registry:
                errors.append(f"{game} submittable exceeds registry claim")
            if submittable > 0 and not (
                row.get("has_replayable_trajectory") is True
                or row.get("has_env_adaptive_resolver") is True
            ):
                errors.append(f"{game} counted without trajectory or adaptive resolver")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / RESULT_RELATIVE_PATH
    _write_json(path, artifact)
    return path


def write_package(root: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    payload = build_package_payload(rows, result_path=RESULT_RELATIVE_PATH)
    path = root / PACKAGE_RELATIVE_PATH
    _write_json(path, payload)
    return path


def run(
    root: Path = REPO_ROOT,
    *,
    offline_arcade_checker: Callable[[], bool] | None = None,
    now: Callable[[], float] = time.perf_counter,
) -> JsonDict:
    started = now()
    preconditions = check_preconditions(root, offline_arcade_checker=offline_arcade_checker)
    miss = first_precondition_miss(preconditions)
    if miss is not None:
        artifact = _blocked_artifact(
            reason=miss,
            preconditions_checked=preconditions,
            duration_s=now() - started,
        )
        write_artifact(root, artifact)
        return artifact

    registry = _load_yaml(root / REGISTRY_RELATIVE_PATH)
    previous_package = _load_json(root / BASELINE_PACKAGE_RELATIVE_PATH)
    previous_result = _load_json(root / BASELINE_RESULT_RELATIVE_PATH)
    rows, folded, trajectory_refreshes, claimed_caps, upstream_audit = build_refreshed_rows(
        root,
        registry=registry,
        previous_package=previous_package,
    )
    artifact = build_artifact(
        rows=rows,
        previous_count=_previous_live_count(previous_result),
        levels_folded_in=folded,
        trajectory_refreshes=trajectory_refreshes,
        claimed_caps=claimed_caps,
        upstream_fold_audit=upstream_audit,
        preconditions_checked=preconditions,
        duration_s=now() - started,
    )
    write_artifact(root, artifact)
    write_package(root, rows)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run(REPO_ROOT)
    print(REPO_ROOT / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    print(f"live_submittable_level_count={artifact['live_submittable_level_count']}")
    print(f"ready_for_operator_submit={artifact['ready_for_operator_submit']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
