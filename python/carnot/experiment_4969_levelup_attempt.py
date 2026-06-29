"""Experiment 4969: fresh deep ARC level-up attempt.

Spec refs: REQ-ARC-WMTE-4969,
SCENARIO-ARC-WMTE-4969-FRESH-DEEP-TARGET,
SCENARIO-ARC-WMTE-4969-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4969-STABLE-ARTIFACT.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4969_levelup_attempt"
SCHEMA = "carnot.arc_levelup_attempt_4969.v1"
RESULT_RELATIVE_PATH = "results/experiment_4969_levelup_attempt.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 20260629
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
STANDING_LOOP_TIMEOUT_S = 120
SPEC_REFS = [
    "REQ-ARC-WMTE-4969",
    "SCENARIO-ARC-WMTE-4969-FRESH-DEEP-TARGET",
    "SCENARIO-ARC-WMTE-4969-REPRODUCTION-GATE",
    "SCENARIO-ARC-WMTE-4969-STABLE-ARTIFACT",
]

PREFERRED_TARGETS = ("tu93", "tn36", "cn04")
RECENT_EXCLUDED_TARGETS = (
    "tr87",
    "s5i5",
    "ar25",
    "vc33",
    "lf52",
    "sb26",
    "sp80",
    "su15",
    "m0r0",
    "dc22",
    "g50t",
)
HIDDEN_STATE_TARGETS = ("ka59", "wa30")

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; banked is success_<game>_levelup_banked, no-bank is "
            "complete_<game>_no_new_level_residual_<cause>."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery -- the agent advanced via its own attempts/runtime "
            "RE; NOT outer_loop_re (CRITICAL) and NOT a re-solve of an already-banked "
            "level (duplicate CRITICAL)."
        )
    },
    "target_game": {
        "principle": (
            "the rotated FRESH grounded DEEP deepen target (differs from .457 tr87/s5i5 "
            "/ .456 ar25/vc33 AND from A2's and A3's targets)."
        )
    },
    "offline_reproduced": {
        "principle": "only reproduced levels count toward reproducible_total_levels."
    },
    "reproduced_levels": {
        "principle": "the new reproducible depth; the monotonic ARC progress metric."
    },
    "new_levels_banked": {
        "principle": (
            ">=1 for a PASS; 0 records the honest rotation dead-end for the next planner "
            "(the deepen well is dry across all regimes)."
        )
    },
    "live_path_reachable": {
        "principle": (
            "true -- the deepen runs through arc_loop_solve + a live GameAdapter "
            "(arc_orphan_solver_lint passes), not a parallel solver."
        )
    },
    "verifier_is_oracle": {
        "principle": "the reproduction gate is the executable oracle (circularity discipline)."
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference if induction runs (60s floor); else "
            "verifier_ensemble_against_cached_candidates / the honest offline arcade substrate."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records arcade/env/generator checks; a missing resource emits blocked_, never "
            "a fabricated solve."
        )
    },
    "random_seed": {"principle": "determinism for the offline search."},
    "reproducibility_checksum": {
        "principle": (
            "content hash of (game, plan, claimed level) so a replication catches drift."
        )
    },
}

REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "field_principles",
    "rotation_selection",
    "approach_recommendation",
    "attempted_games",
    "dead_ends",
    "registry_update",
    "retire_if_same_verdict",
    "loop_command",
    "loop_artifact",
    "schema_errors",
)


JsonDict = dict[str, Any]
RecommendFn = Callable[[str], Mapping[str, Any]]


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _load_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_registry(root: Path = REPO) -> JsonDict:
    return yaml.safe_load((root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")) or {}


def _game_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for row in registry.get("games", []) or []:
        if isinstance(row, Mapping) and row.get("game"):
            rows[str(row["game"])] = row
    return rows


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dead_ends(row: Mapping[str, Any]) -> list[str]:
    values = row.get("dead_ends") or []
    if isinstance(values, list):
        return [str(value) for value in values]
    return [str(values)]


def _row_text(row: Mapping[str, Any]) -> str:
    return _json_dumps(row).lower()


def _has_grounded_next_level_delta(row: Mapping[str, Any], next_level: int) -> tuple[bool, str | None]:
    text = _row_text(row)
    blocking_tokens = (
        f"no_grounded_l{next_level}_delta",
        f"no grounded l{next_level} delta",
        "hidden-state-bound",
        "hidden state bound",
    )
    for token in blocking_tokens:
        if token in text:
            return False, token.replace(" ", "_").replace("-", "_")
    return True, None


def reproducibility_checksum(game: str, plan: Sequence[Any], claimed_level: int) -> str:
    payload = {
        "claimed_level": int(claimed_level),
        "game": str(game),
        "plan": list(plan),
    }
    digest = hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def loop_result_relative_path(game: str) -> str:
    return f"results/arc_loop_solve_{game}.json"


def loop_command(selection: Mapping[str, Any]) -> list[str]:
    return [
        ".venv/bin/python",
        "scripts/arc_loop_solve.py",
        "--game",
        str(selection["target_game"]),
        "--target-level",
        str(selection["target_level"]),
    ]


def select_target(
    registry: Mapping[str, Any],
    *,
    recommend_fn: RecommendFn,
) -> JsonDict:
    """Select the fresh deep target and attach the required routing recommendation.

    The script default target level is too low for deep rows, so this function
    always emits `prior + 1`. That is the duplicate-depth guard for Exp 4969.
    """

    rows = _game_rows(registry)
    excluded = sorted(set(RECENT_EXCLUDED_TARGETS) | set(HIDDEN_STATE_TARGETS))
    candidate_audit: list[JsonDict] = []
    for game in PREFERRED_TARGETS:
        row = rows.get(game)
        if row is None:
            candidate_audit.append({"game": game, "status": "missing_registry_row"})
            continue
        prior = _int_value(row.get("levels_reproduced"))
        next_level = prior + 1
        grounded, residual = _has_grounded_next_level_delta(row, next_level)
        audit_row = {
            "game": game,
            "prior_reproduced_levels": prior,
            "target_level": next_level,
            "dead_end_count": len(_dead_ends(row)),
            "dead_ends_consulted": _dead_ends(row),
            "grounded_next_level_delta": grounded,
            "status": "candidate" if grounded else "skip_no_grounded_delta",
            "residual_cause": residual,
        }
        candidate_audit.append(audit_row)
        if not grounded:
            continue
        recommendation = dict(recommend_fn(game))
        return {
            "target_game": game,
            "prior_reproduced_levels": prior,
            "target_level": next_level,
            "grounded_next_level_delta": True,
            "candidate_order": list(PREFERRED_TARGETS),
            "candidate_audit": candidate_audit,
            "excluded_lanes": excluded,
            "approach_recommendation": recommendation,
            "selection_reason": "fresh_deep_rotated_grounded_target",
        }
    return {
        "target_game": "none",
        "prior_reproduced_levels": 0,
        "target_level": 0,
        "grounded_next_level_delta": False,
        "candidate_order": list(PREFERRED_TARGETS),
        "candidate_audit": candidate_audit,
        "excluded_lanes": excluded,
        "approach_recommendation": {},
        "selection_reason": "no_grounded_fresh_deep_target",
        "residual_cause": "no_grounded_next_level_delta",
    }


def _loop_reproduced_level(loop_result: Mapping[str, Any]) -> int:
    gate = loop_result.get("reproduction_gate")
    gate = gate if isinstance(gate, Mapping) else {}
    return max(
        _int_value(loop_result.get("reproduced_levels")),
        _int_value(loop_result.get("reached_level")),
        _int_value(gate.get("reached_level")),
        _int_value(gate.get("claimed_level")),
    )


def _loop_offline_reproduced(loop_result: Mapping[str, Any]) -> bool:
    gate = loop_result.get("reproduction_gate")
    gate = gate if isinstance(gate, Mapping) else {}
    gate_reproduced = gate.get("reproduced")
    return bool(loop_result.get("offline_reproduced") is True and gate_reproduced is not False)


def _loop_plan(loop_result: Mapping[str, Any]) -> list[Any]:
    labels = loop_result.get("solution_labels")
    if isinstance(labels, list):
        return list(labels)
    solution = loop_result.get("solution")
    if isinstance(solution, list):
        return list(solution)
    return []


def _residual_cause(loop_result: Mapping[str, Any], prior: int) -> str:
    if loop_result.get("status") == "needs_per_game_RE":
        return "needs_per_game_re"
    if loop_result.get("status") == "standing_loop_timeout":
        return "standing_loop_timeout"
    if not _loop_offline_reproduced(loop_result):
        return "offline_reproduction_failed"
    if _loop_reproduced_level(loop_result) <= prior:
        return "duplicate_depth"
    return "unknown"


def _base_artifact(
    *,
    selection: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    loop_command_value: Sequence[str],
    loop_artifact: str | None,
) -> JsonDict:
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "solve_provenance": SOLVE_PROVENANCE,
        "target_game": selection.get("target_game"),
        "live_path_reachable": True,
        "verifier_is_oracle": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "rotation_selection": dict(selection),
        "approach_recommendation": dict(selection.get("approach_recommendation") or {}),
        "attempted_games": [],
        "dead_ends": [],
        "registry_update": {},
        "retire_if_same_verdict": True,
        "loop_command": list(loop_command_value),
        "loop_artifact": loop_artifact,
    }


def build_artifact(
    *,
    selection: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    loop_result: Mapping[str, Any],
    loop_artifact: str | None,
    loop_command: Sequence[str],
) -> JsonDict:
    game = str(selection["target_game"])
    prior = _int_value(selection.get("prior_reproduced_levels"))
    reproduced_level = _loop_reproduced_level(loop_result)
    banked = max(0, reproduced_level - prior) if _loop_offline_reproduced(loop_result) else 0
    success = banked >= 1 and reproduced_level >= _int_value(selection.get("target_level"))
    residual = _residual_cause(loop_result, prior)
    plan = _loop_plan(loop_result)

    artifact = _base_artifact(
        selection=selection,
        preconditions_checked=preconditions_checked,
        loop_command_value=loop_command,
        loop_artifact=loop_artifact,
    )
    artifact.update(
        {
            "honest_verdict": (
                f"success_{game}_levelup_banked"
                if success
                else f"complete_{game}_no_new_level_residual_{residual}"
            ),
            "offline_reproduced": bool(success),
            "reproduced_levels": int(reproduced_level if reproduced_level else prior),
            "new_levels_banked": int(banked if success else 0),
            "attempted_games": [
                {
                    "game": game,
                    "prior_reproduced_levels": prior,
                    "target_level": _int_value(selection.get("target_level")),
                    "loop_reproduced_level": reproduced_level,
                    "offline_reproduced": _loop_offline_reproduced(loop_result),
                }
            ],
            "dead_ends": []
            if success
            else [
                {
                    "game": game,
                    "attempted_level": _int_value(selection.get("target_level")),
                    "residual_cause": residual,
                    "loop_artifact": loop_artifact,
                }
            ],
            "registry_update": {
                "action": "bank_new_level" if success else "record_dead_end_only",
                "path": REGISTRY_RELATIVE_PATH,
                "target_game": game,
                "prior_levels": prior,
                "new_total_levels": reproduced_level if success else prior,
                "banked_levels": int(banked if success else 0),
                "residual_cause": None if success else residual,
            },
            "retire_if_same_verdict": not success,
            "reproduction_gate": dict(loop_result.get("reproduction_gate") or {}),
            "solution_labels": plan,
            "states_expanded": _int_value(loop_result.get("states_expanded")),
            "standing_loop_subprocess": dict(loop_result.get("standing_loop_subprocess") or {}),
            "verifier_source": loop_result.get("verifier_src"),
            "reproducibility_checksum": reproducibility_checksum(game, plan, reproduced_level),
        }
    )
    artifact["schema_errors"] = []
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    return artifact


def build_no_grounded_delta_artifact(
    *,
    selection: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
) -> JsonDict:
    game = str(selection.get("target_game") or "none")
    prior = _int_value(selection.get("prior_reproduced_levels"))
    residual = str(selection.get("residual_cause") or "no_grounded_next_level_delta")
    artifact = _base_artifact(
        selection=selection,
        preconditions_checked=preconditions_checked,
        loop_command_value=[],
        loop_artifact=None,
    )
    artifact.update(
        {
            "honest_verdict": f"complete_{game}_no_new_level_residual_{residual}",
            "offline_reproduced": False,
            "reproduced_levels": prior,
            "new_levels_banked": 0,
            "live_path_reachable": False,
            "attempted_games": [],
            "dead_ends": [
                {
                    "game": game,
                    "attempted_level": _int_value(selection.get("target_level")),
                    "residual_cause": residual,
                    "loop_artifact": None,
                }
            ],
            "registry_update": {
                "action": "record_dead_end_only",
                "path": REGISTRY_RELATIVE_PATH,
                "target_game": game,
                "prior_levels": prior,
                "new_total_levels": prior,
                "banked_levels": 0,
                "residual_cause": residual,
            },
            "reproducibility_checksum": reproducibility_checksum(game, [], prior),
        }
    )
    artifact["schema_errors"] = []
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    return artifact


def build_blocked_artifact(
    *,
    selection: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    residual_cause: str,
) -> JsonDict:
    game = str(selection.get("target_game") or "none")
    prior = _int_value(selection.get("prior_reproduced_levels"))
    artifact = _base_artifact(
        selection=selection,
        preconditions_checked=preconditions_checked,
        loop_command_value=[],
        loop_artifact=None,
    )
    artifact.update(
        {
            "honest_verdict": f"blocked_{game}_{residual_cause}",
            "offline_reproduced": False,
            "reproduced_levels": prior,
            "new_levels_banked": 0,
            "live_path_reachable": False,
            "registry_update": {
                "action": "blocked_no_registry_change",
                "path": REGISTRY_RELATIVE_PATH,
                "target_game": game,
                "prior_levels": prior,
                "new_total_levels": prior,
                "banked_levels": 0,
                "residual_cause": residual_cause,
            },
            "reproducibility_checksum": reproducibility_checksum(game, [], prior),
        }
    )
    artifact["schema_errors"] = []
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing:{field}")
    if artifact.get("experiment") != EXPERIMENT:
        errors.append("experiment_mismatch")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs_mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    verdict = str(artifact.get("honest_verdict") or "")
    if not re.match(r"^(success|complete|blocked)_", verdict):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance_mismatch")
    if not isinstance(artifact.get("target_game"), str):
        errors.append("target_game_type")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced_type")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels_type")
    if type(artifact.get("new_levels_banked")) is not int:
        errors.append("new_levels_banked_type")
    if type(artifact.get("live_path_reachable")) is not bool:
        errors.append("live_path_reachable_type")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_not_true")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked_type")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", checksum):
        errors.append("reproducibility_checksum_format")
    if verdict.startswith("success_"):
        if artifact.get("offline_reproduced") is not True:
            errors.append("success_requires_offline_reproduced")
        if _int_value(artifact.get("new_levels_banked")) < 1:
            errors.append("success_requires_new_level")
        prior = _int_value(
            (artifact.get("rotation_selection") or {}).get("prior_reproduced_levels")
            if isinstance(artifact.get("rotation_selection"), Mapping)
            else 0
        )
        if _int_value(artifact.get("reproduced_levels")) <= prior:
            errors.append("success_requires_strictly_deeper_level")
    if artifact.get("offline_reproduced") is True and _int_value(artifact.get("new_levels_banked")) < 1:
        errors.append("offline_reproduced_without_bank")
    if artifact.get("retire_if_same_verdict") is False and not verdict.startswith("success_"):
        errors.append("retire_flag_false_on_non_success")
    return errors


def check_preconditions(
    selection: Mapping[str, Any],
    root: Path = REPO,
    offline_arcade_factory: Callable[[], Any] | None = None,
) -> JsonDict:
    game = str(selection.get("target_game") or "")
    checked: JsonDict = {
        "offline_arcade": {"ok": False, "backend": "arc_solver_kit.offline_arcade"},
        "target_env": {"ok": False, "game": game},
        "generator": {"required": False, "checked": False},
        "gpu_policy": {
            "cuda_gpu0_allowed": True,
            "igpu_hip_allowed": True,
            "igpu_pin_required": False,
        },
    }
    if not (root / "environment_files" / game).exists():
        checked["target_env"]["error"] = "environment_files entry missing"
        return checked
    try:
        if offline_arcade_factory is None:
            from carnot.agentic import arc_solver_kit as kit

            offline_arcade_factory = kit.offline_arcade

        arc = offline_arcade_factory()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        env.reset()
        checked["offline_arcade"]["ok"] = True
        checked["target_env"]["ok"] = True
    except Exception as exc:  # pragma: no cover - environment boundary
        checked["offline_arcade"]["error"] = str(exc)
        checked["target_env"]["error"] = str(exc)
    return checked


def run_standing_loop(
    *,
    selection: Mapping[str, Any],
    root: Path = REPO,
) -> JsonDict:  # pragma: no cover - subprocess boundary
    command = loop_command(selection)
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=STANDING_LOOP_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "game": selection["target_game"],
            "offline_reproduced": False,
            "reproduced_levels": selection.get("prior_reproduced_levels", 0),
            "status": "standing_loop_timeout",
            "standing_loop_subprocess": {
                "returncode": "timeout",
                "timeout_s": STANDING_LOOP_TIMEOUT_S,
                "stdout_tail": str(exc.stdout or "")[-4000:],
                "stderr_tail": str(exc.stderr or "")[-4000:],
            },
        }
    result_path = root / loop_result_relative_path(str(selection["target_game"]))
    if result_path.exists():
        result = _load_json(result_path)
    else:
        result = {
            "game": selection["target_game"],
            "offline_reproduced": False,
            "reproduced_levels": selection.get("prior_reproduced_levels", 0),
            "status": "standing_loop_failed_no_artifact",
        }
    result["standing_loop_subprocess"] = {
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    return result


def _write_artifact(artifact: Mapping[str, Any], root: Path = REPO) -> None:
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def run_experiment(
    *,
    root: Path = REPO,
    registry: Mapping[str, Any] | None = None,
    recommend_fn: RecommendFn | None = None,
    loop_result: Mapping[str, Any] | None = None,
) -> JsonDict:
    if recommend_fn is None:  # pragma: no cover - CLI default path
        from carnot.agentic import arc_solve_learning

        recommend_fn = arc_solve_learning.recommend_approach

    loaded_registry = dict(registry or _load_registry(root))
    selection = select_target(loaded_registry, recommend_fn=recommend_fn)
    preconditions = check_preconditions(selection, root=root)

    if preconditions.get("offline_arcade", {}).get("ok") is not True:
        artifact = build_blocked_artifact(
            selection=selection,
            preconditions_checked=preconditions,
            residual_cause="offline_env_missing",
        )
        _write_artifact(artifact, root=root)
        return artifact
    if preconditions.get("target_env", {}).get("ok") is not True:
        artifact = build_blocked_artifact(
            selection=selection,
            preconditions_checked=preconditions,
            residual_cause="offline_env_missing",
        )
        _write_artifact(artifact, root=root)
        return artifact
    if selection.get("grounded_next_level_delta") is not True:
        artifact = build_no_grounded_delta_artifact(
            selection=selection,
            preconditions_checked=preconditions,
        )
        _write_artifact(artifact, root=root)
        return artifact

    loop = dict(loop_result or run_standing_loop(selection=selection, root=root))
    command = loop_command(selection)
    loop_artifact = (
        None
        if loop.get("status") == "standing_loop_timeout"
        else loop_result_relative_path(str(selection["target_game"]))
    )
    artifact = build_artifact(
        selection=selection,
        preconditions_checked=preconditions,
        loop_result=loop,
        loop_artifact=loop_artifact,
        loop_command=command,
    )
    _write_artifact(artifact, root=root)
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    del argv
    artifact = run_experiment(root=REPO)
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "target_game": artifact["target_game"],
                "offline_reproduced": artifact["offline_reproduced"],
                "reproduced_levels": artifact["reproduced_levels"],
                "new_levels_banked": artifact["new_levels_banked"],
                "result_path": RESULT_RELATIVE_PATH,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
