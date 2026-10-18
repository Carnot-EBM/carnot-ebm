"""Experiment 4894: rotated ARC level-up attempt.

Spec refs: REQ-REPORT-4894, SCENARIO-REPORT-4894,
SCENARIO-REPORT-4894-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4894_levelup_attempt"
SCHEMA = "carnot.arc_levelup_attempt_4894.v1"
TARGET_GAME = "dc22"
RESULT_RELATIVE_PATH = "results/experiment_4894_levelup_attempt.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4894
SPEC_REFS = [
    "REQ-REPORT-4894",
    "SCENARIO-REPORT-4894",
    "SCENARIO-REPORT-4894-BLOCKED-PRECONDITION",
]
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "offline_arcade_reproduction_gate_no_llm"
RECENT_EXCLUDED_TARGETS = ("g50t", "s5i5", "r11l")
HIDDEN_STATE_TARGETS = ("ka59", "wa30")
CANDIDATE_GAMES = ("dc22", "sp80", "su15", "cn04")

REQUIRED_FIELDS = (
    "honest_verdict",
    "solve_provenance",
    "target_game",
    "offline_reproduced",
    "reproduced_levels",
    "new_levels_banked",
    "verifier_is_oracle",
    "inference_substrate",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; banked is success_<game>_levelup_banked, no-bank is "
            "complete_<game>_no_new_level_residual_<cause>."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery -- the agent solved via its own attempts/RE; "
            "NOT outer_loop_re."
        )
    },
    "target_game": {
        "principle": (
            "the rotated target must differ from .450 g50t, .449 s5i5, and .448 r11l."
        )
    },
    "offline_reproduced": {
        "principle": "only reproduced levels count toward reproducible_total_levels."
    },
    "reproduced_levels": {
        "principle": "the new reproducible depth; the monotonic ARC progress metric."
    },
    "new_levels_banked": {
        "principle": ">=1 for a PASS; 0 records the rotation dead-end for the next planner."
    },
    "verifier_is_oracle": {
        "principle": "the reproduction gate is the executable oracle."
    },
    "inference_substrate": {
        "principle": "live_llm_inference if induction runs; this path uses no LLM induction."
    },
    "preconditions_checked": {
        "principle": (
            "records arcade/env/generator checks; missing resources emit blocked_, never a fabricated solve."
        )
    },
}


def loop_result_relative_path(game: str = TARGET_GAME) -> str:
    return f"results/arc_loop_solve_{game}.json"


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Return a stable hash over replay evidence, excluding self-referential fields."""

    checksum_payload = {
        "experiment": payload.get("experiment"),
        "schema": payload.get("schema"),
        "target_game": payload.get("target_game"),
        "prior_reproduced_level": payload.get("prior_reproduced_level"),
        "reproduced_levels": payload.get("reproduced_levels"),
        "new_levels_banked": payload.get("new_levels_banked"),
        "offline_reproduced": payload.get("offline_reproduced"),
        "reproduction_gate": payload.get("reproduction_gate"),
        "solution_labels": list(payload.get("solution_labels") or []),
        "registry_update": dict(payload.get("registry_update") or {}),
        "approach_recommendation": dict(payload.get("approach_recommendation") or {}),
        "random_seed": payload.get("random_seed"),
    }
    raw = json.dumps(checksum_payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:  # pragma: no cover - CLI boundary
    return json.loads(path.read_text(encoding="utf-8"))


def _registry(root: Path = REPO) -> dict[str, Any]:
    return yaml.safe_load((root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")) or {}


def _game_row(registry: Mapping[str, Any], game: str) -> Mapping[str, Any]:
    for row in registry.get("games", []) or []:
        if isinstance(row, Mapping) and row.get("game") == game:
            return row
    raise ValueError(f"registry missing game row: {game}")


def _dead_ends(row: Mapping[str, Any]) -> list[str]:
    values = row.get("dead_ends") or []
    if isinstance(values, list):
        return [str(value) for value in values]
    return [str(values)]


def _has_recorded_l3_dead_end(game: str, row: Mapping[str, Any]) -> bool:
    for item in _dead_ends(row):
        lowered = item.lower()
        has_blocking_phrase = any(
            phrase in lowered
            for phrase in (
                "no grounded l3",
                "no_grounded_l3",
                "current adapter has no grounded l3",
                "stalled before a gate",
            )
        )
        if not has_blocking_phrase:
            continue
        if game in lowered or lowered.startswith(("current adapter", "exp")):
            return True
    return False


def registry_level(game: str = TARGET_GAME, root: Path = REPO) -> int:
    try:
        return int(_game_row(_registry(root), game).get("levels_reproduced") or 0)
    except (TypeError, ValueError):
        return 0


def registry_total_levels(root: Path = REPO) -> int:
    try:
        return int(_registry(root).get("reproducible_total_levels") or 0)
    except (TypeError, ValueError):
        return 0


def select_rotation_target(
    registry: Mapping[str, Any],
    recommendations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = {
        str(row.get("game")): row
        for row in registry.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    }
    audit: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for game in CANDIDATE_GAMES:
        row = rows.get(game, {})
        prior = int(row.get("levels_reproduced") or 0)
        status = "candidate_unselected"
        reason = "lower_priority_grounded_delta"
        if game in RECENT_EXCLUDED_TARGETS:
            status, reason = "skip_recent_target", "recently_banked_target"
        elif game in HIDDEN_STATE_TARGETS:
            status, reason = "skip_hidden_state_bound", "hidden_state_bound"
        elif prior != 2:
            status, reason = "skip_not_l2", "not_current_l2_depth"
        elif _has_recorded_l3_dead_end(game, row):
            status, reason = "skip_recorded_dead_end", "recorded_l3_dead_end"
        elif selected is None:
            status, reason = "selected", "grounded_rotated_l2_to_l3_candidate"
        audit_row = {
            "game": game,
            "prior_level": prior,
            "target_level": prior + 1 if prior else 1,
            "status": status,
            "reason": reason,
            "dead_ends_consulted": _dead_ends(row),
        }
        audit.append(audit_row)
        if status == "selected" and selected is None:
            selected = audit_row

    if selected is None:
        return {
            "game": "none",
            "prior_level": 0,
            "target_level": 0,
            "reason": "no_grounded_rotated_l2_to_l3_candidate",
            "candidate_audit": audit,
            "excluded_recent_targets": list(RECENT_EXCLUDED_TARGETS),
            "hidden_state_targets_avoided": list(HIDDEN_STATE_TARGETS),
            "approach_recommendation": {},
        }
    game = str(selected["game"])
    return {
        "game": game,
        "prior_level": int(selected["prior_level"]),
        "target_level": int(selected["target_level"]),
        "reason": str(selected["reason"]),
        "candidate_audit": audit,
        "excluded_recent_targets": list(RECENT_EXCLUDED_TARGETS),
        "hidden_state_targets_avoided": list(HIDDEN_STATE_TARGETS),
        "approach_recommendation": dict((recommendations or {}).get(game) or {}),
    }


def _loop_reached_level(loop_result: Mapping[str, Any]) -> int:
    gate = loop_result.get("reproduction_gate") if isinstance(loop_result.get("reproduction_gate"), Mapping) else {}
    return int(gate.get("reached_level") or loop_result.get("reached_level") or 0)


def _loop_reproduced(loop_result: Mapping[str, Any]) -> bool:
    gate = loop_result.get("reproduction_gate") if isinstance(loop_result.get("reproduction_gate"), Mapping) else {}
    return bool(loop_result.get("offline_reproduced") is True and gate.get("reproduced") is True)


def _success(loop_result: Mapping[str, Any], prior_level: int, registry_update: Mapping[str, Any]) -> bool:
    return bool(
        _loop_reproduced(loop_result)
        and _loop_reached_level(loop_result) > int(prior_level)
        and int(registry_update.get("banked_levels") or 0) >= 1
    )


def _residual_cause(loop_result: Mapping[str, Any], prior_level: int) -> str:
    if _loop_reproduced(loop_result) and _loop_reached_level(loop_result) <= int(prior_level):
        return "duplicate_depth"
    if loop_result.get("status") == "needs_per_game_RE":
        return "needs_per_game_re"
    if not _loop_reproduced(loop_result):
        return "offline_reproduction_failed"
    return "unknown"


def build_artifact(
    *,
    loop_result: Mapping[str, Any],
    prior_level: int,
    prior_total_levels: int,
    preconditions_checked: Mapping[str, Any],
    approach_recommendation: Mapping[str, Any] | None,
    registry_update: Mapping[str, Any],
) -> dict[str, Any]:
    game = str(loop_result.get("game") or TARGET_GAME)
    reached_level = _loop_reached_level(loop_result)
    banked_levels = max(0, reached_level - int(prior_level)) if _loop_reproduced(loop_result) else 0
    if int(registry_update.get("banked_levels") or 0) > 0:
        banked_levels = int(registry_update.get("banked_levels") or 0)
    success = _success(loop_result, prior_level, {"banked_levels": banked_levels})
    residual = str(registry_update.get("reason") or _residual_cause(loop_result, prior_level))
    verdict = f"success_{game}_levelup_banked" if success else f"complete_{game}_no_new_level_residual_{residual}"
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": verdict,
        "solve_provenance": SOLVE_PROVENANCE,
        "target_game": game,
        "offline_reproduced": bool(success),
        "reproduced_levels": int(reached_level if success else 0),
        "new_levels_banked": int(banked_levels if success else 0),
        "verifier_is_oracle": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "prior_reproduced_level": int(prior_level),
        "reproducible_total_levels_before": int(prior_total_levels),
        "reproducible_total_levels_after": int(prior_total_levels + (banked_levels if success else 0)),
        "random_seed": RANDOM_SEED,
        "standing_loop_result_path": loop_result_relative_path(game),
        "approach_recommendation": dict(approach_recommendation or {}),
        "registry_update": dict(registry_update),
        "reproduction_gate": dict(loop_result.get("reproduction_gate") or {}),
        "solution_labels": list(loop_result.get("solution_labels") or []),
        "solution": list(loop_result.get("solution") or []),
        "states_expanded": int(loop_result.get("states_expanded") or 0),
        "retire_if_same_verdict": not success,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    return artifact


def blocked_artifact(
    *,
    target_game: str,
    reason: str,
    preconditions_checked: Mapping[str, Any],
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": f"blocked_{target_game}_{reason}",
        "solve_provenance": SOLVE_PROVENANCE,
        "target_game": target_game,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "new_levels_banked": 0,
        "verifier_is_oracle": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "prior_reproduced_level": 0,
        "reproducible_total_levels_before": 0,
        "reproducible_total_levels_after": 0,
        "random_seed": RANDOM_SEED,
        "standing_loop_result_path": loop_result_relative_path(target_game),
        "approach_recommendation": {},
        "registry_update": {"updated": False, "banked_levels": 0},
        "reproduction_gate": {},
        "solution_labels": [],
        "solution": [],
        "states_expanded": 0,
        "retire_if_same_verdict": True,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    if artifact.get("experiment") != EXPERIMENT:
        errors.append("experiment mismatch")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    if artifact.get("target_game") in set(RECENT_EXCLUDED_TARGETS) | set(HIDDEN_STATE_TARGETS):
        errors.append("target_game violates rotation exclusions")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    if type(artifact.get("new_levels_banked")) is not int:
        errors.append("new_levels_banked must be bare int")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        errors.append("reproducibility_checksum must be 64 hex chars")
    elif checksum != reproducibility_checksum(artifact):
        errors.append("checksum mismatch")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (
        verdict.startswith("success_")
        or verdict.startswith("complete_")
        or verdict.startswith("blocked_")
    ):
        errors.append("honest_verdict must use a terminal prefix")
    if verdict.startswith("success_"):
        if artifact.get("offline_reproduced") is not True:
            errors.append("success requires offline_reproduced true")
        if int(artifact.get("new_levels_banked") or 0) < 1:
            errors.append("success requires new_levels_banked >= 1")
        if int(artifact.get("reproduced_levels") or 0) <= int(artifact.get("prior_reproduced_level") or 0):
            errors.append("success requires reproduced_levels > prior_reproduced_level")
    if artifact.get("offline_reproduced") is True and int(artifact.get("new_levels_banked") or 0) < 1:
        errors.append("offline_reproduced true requires new_levels_banked >= 1")
    return errors


def _game_block_bounds(text: str, game: str) -> tuple[int, int]:
    marker = f"- game: {game}\n"
    start = text.index(marker)
    next_match = re.search(r"\n- game: ", text[start + len(marker) :])
    end = start + len(marker) + next_match.start() + 1 if next_match else len(text)
    return start, end


def _registry_update_for_loop(
    registry_text: str,
    loop_result: Mapping[str, Any],
) -> dict[str, Any]:
    registry = yaml.safe_load(registry_text) or {}
    game = str(loop_result.get("game") or TARGET_GAME)
    prior = int(_game_row(registry, game).get("levels_reproduced") or 0)
    reached = _loop_reached_level(loop_result)
    reproduced = _loop_reproduced(loop_result)
    banked = max(0, reached - prior) if reproduced else 0
    prior_total = int(registry.get("reproducible_total_levels") or 0)
    return {
        "updated": True,
        "path": REGISTRY_RELATIVE_PATH,
        "target_game": game,
        "prior_game_levels": prior,
        "new_game_levels": reached if banked else prior,
        "banked_levels": banked,
        "prior_total_declared": prior_total,
        "new_total_declared": prior_total + banked,
        "reason": "banked_offline_reproduced_level" if banked else _residual_cause(loop_result, prior),
    }


def _replace_game_row(registry_text: str, game: str, row: Mapping[str, Any], total_levels: int) -> str:
    start, end = _game_block_bounds(registry_text, game)
    block = yaml.safe_dump([dict(row)], sort_keys=False, width=1000)
    updated = registry_text[:start] + block + registry_text[end:]
    updated = re.sub(
        r"(?m)^(reproducible_total_levels:\s*)\d+\s*$",
        rf"\g<1>{total_levels}",
        updated,
        count=1,
    )
    updated = re.sub(r"(?m)^updated: '[^']+'", "updated: '2026-06-28'", updated, count=1)
    return updated


def apply_registry_result(
    registry_text: str,
    *,
    artifact: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    registry = yaml.safe_load(registry_text) or {}
    game = str(artifact.get("target_game") or TARGET_GAME)
    row = dict(_game_row(registry, game))
    prior = int(row.get("levels_reproduced") or 0)
    reached = int(artifact.get("reproduced_levels") or 0)
    prior_total = int(registry.get("reproducible_total_levels") or 0)
    banked = max(0, reached - prior) if artifact.get("offline_reproduced") is True else 0
    update = {
        "updated": True,
        "path": REGISTRY_RELATIVE_PATH,
        "target_game": game,
        "prior_game_levels": prior,
        "new_game_levels": reached if banked else prior,
        "banked_levels": banked,
        "prior_total_declared": prior_total,
        "new_total_declared": prior_total + banked,
        "reason": "banked_offline_reproduced_level" if banked else _artifact_residual_reason(artifact),
    }
    if banked >= 1:
        row["reproducibility"] = "reproduced"
        row["levels_reproduced"] = reached
        row["reproduce"] = (
            f"Exp4894 {RESULT_RELATIVE_PATH} re-gated {loop_result_relative_path(game)} "
            f"offline_reproduced=True, reached_level={reached}, banked +{banked}, "
            f"checksum {artifact.get('reproducibility_checksum')}."
        )
        row["latest_exp4894_levelup_attempt"] = {
            "artifact": RESULT_RELATIVE_PATH,
            "loop_artifact": loop_result_relative_path(game),
            "offline_reproduced": True,
            "reproduced_levels": reached,
            "new_levels_banked": banked,
            "solve_provenance": SOLVE_PROVENANCE,
            "target_rotation": "rotated_off_g50t_s5i5_r11l_hidden_state_bound_targets",
            "reproducibility_checksum": artifact.get("reproducibility_checksum"),
        }
    else:
        dead_ends = list(row.get("dead_ends") or [])
        reason = _artifact_residual_reason(artifact)
        note = f"Exp4894 {game} no-bank {reason}: {artifact.get('honest_verdict')}."
        if note not in dead_ends:
            dead_ends.append(note)
        row["dead_ends"] = dead_ends
        row["latest_exp4894_levelup_attempt"] = {
            "artifact": RESULT_RELATIVE_PATH,
            "loop_artifact": loop_result_relative_path(game),
            "offline_reproduced": False,
            "reproduced_levels": prior,
            "new_levels_banked": 0,
            "residual_cause": reason,
            "solve_provenance": SOLVE_PROVENANCE,
            "reproducibility_checksum": artifact.get("reproducibility_checksum"),
        }
    return _replace_game_row(registry_text, game, row, prior_total + banked), update


def _artifact_residual_reason(artifact: Mapping[str, Any]) -> str:
    verdict = str(artifact.get("honest_verdict") or "")
    marker = "_no_new_level_residual_"
    if marker in verdict:
        return verdict.split(marker, 1)[1]
    return str((artifact.get("registry_update") or {}).get("reason") or "unknown")


def precondition_probe(target_game: str = TARGET_GAME, root: Path = REPO) -> dict[str, Any]:  # pragma: no cover
    checked: dict[str, Any] = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "target_env_present": (root / "environment_files" / target_game).exists(),
        "generator_required": False,
        "rotated_off": list(RECENT_EXCLUDED_TARGETS),
        "hidden_state_targets_avoided": list(HIDDEN_STATE_TARGETS),
        "standing_loop_command": f".venv/bin/python scripts/arc_loop_solve.py --game {target_game}",
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checked["offline_arcade_exits_0"] = True
    except Exception as exc:
        checked["offline_arcade_exits_0"] = False
        checked["offline_arcade_error"] = str(exc)
    return checked


def run_experiment(
    *,
    root: Path = REPO,
    target_game: str = TARGET_GAME,
    loop_result: Mapping[str, Any] | None = None,
    approach_recommendation: Mapping[str, Any] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    checked = dict(preconditions_checked or precondition_probe(target_game, root))
    if checked.get("offline_arcade_exits_0") is not True or checked.get("target_env_present") is False:
        reason = "offline_env_missing" if checked.get("target_env_present") is False else "offline_arcade_missing"
        artifact = blocked_artifact(
            target_game=target_game,
            reason=reason,
            preconditions_checked=checked,
        )
        _write_artifact(root, artifact)
        return artifact

    registry_text = (root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")
    registry = yaml.safe_load(registry_text) or {}
    prior_level = int(_game_row(registry, target_game).get("levels_reproduced") or 0)
    prior_total = int(registry.get("reproducible_total_levels") or 0)
    loop_path = root / loop_result_relative_path(target_game)
    loop = dict(loop_result or _read_json(loop_path))
    loop.setdefault("game", target_game)
    registry_update = _registry_update_for_loop(registry_text, loop)
    artifact = build_artifact(
        loop_result=loop,
        prior_level=prior_level,
        prior_total_levels=prior_total,
        preconditions_checked=checked,
        approach_recommendation=approach_recommendation,
        registry_update=registry_update,
    )
    updated_text, final_update = apply_registry_result(registry_text, artifact=artifact)
    (root / REGISTRY_RELATIVE_PATH).write_text(updated_text, encoding="utf-8")
    artifact = build_artifact(
        loop_result=loop,
        prior_level=prior_level,
        prior_total_levels=prior_total,
        preconditions_checked=checked,
        approach_recommendation=approach_recommendation,
        registry_update=final_update,
    )
    _write_artifact(root, artifact)
    return artifact


def _write_artifact(root: Path, artifact: Mapping[str, Any]) -> None:
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    del argv
    from carnot.agentic import arc_solve_learning

    loop_path = REPO / loop_result_relative_path(TARGET_GAME)
    if not loop_path.exists():
        print(f"missing {loop_result_relative_path(TARGET_GAME)}; run scripts/arc_loop_solve.py first", file=sys.stderr)
        return 2
    artifact = run_experiment(
        root=REPO,
        target_game=TARGET_GAME,
        loop_result=_read_json(loop_path),
        approach_recommendation=arc_solve_learning.recommend_approach(TARGET_GAME),
    )
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "target_game": artifact["target_game"],
                "offline_reproduced": artifact["offline_reproduced"],
                "reproduced_levels": artifact["reproduced_levels"],
                "new_levels_banked": artifact["new_levels_banked"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
