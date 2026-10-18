"""Experiment 4884: rotated g50t ARC level-up attempt.

Spec refs: REQ-REPORT-4884, SCENARIO-REPORT-4884.
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
EXPERIMENT = "experiment_4884_levelup_attempt"
SCHEMA = "carnot.arc_levelup_attempt_4884.v1"
TARGET_GAME = "g50t"
RESULT_RELATIVE_PATH = "results/experiment_4884_levelup_attempt.json"
LOOP_RESULT_RELATIVE_PATH = "results/arc_loop_solve_g50t.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4884
SPEC_REFS = ["REQ-REPORT-4884", "SCENARIO-REPORT-4884"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "offline_arcade_reproduction_gate_no_llm"
DISALLOWED_TARGETS = {"s5i5", "r11l", "re86", "ka59", "wa30"}

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
            "the rotated target differs from .449 s5i5, .448 r11l, and re86 so "
            "coverage sweeps the corpus."
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
        "initial_loop_result": dict(payload.get("initial_loop_result") or {}),
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
    initial_loop_result: Mapping[str, Any] | None,
    approach_recommendation: Mapping[str, Any] | None,
    registry_update: Mapping[str, Any],
) -> dict[str, Any]:
    game = str(loop_result.get("game") or TARGET_GAME)
    reached_level = _loop_reached_level(loop_result)
    banked_levels = max(0, reached_level - int(prior_level)) if _loop_reproduced(loop_result) else 0
    if int(registry_update.get("banked_levels") or 0) > 0:
        banked_levels = int(registry_update.get("banked_levels") or 0)
    success = _success(loop_result, prior_level, {"banked_levels": banked_levels})
    verdict = (
        f"success_{game}_levelup_banked"
        if success
        else f"complete_{game}_no_new_level_residual_{_residual_cause(loop_result, prior_level)}"
    )
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
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
        "reproduction_gate": dict(loop_result.get("reproduction_gate") or {}),
        "solution_labels": list(loop_result.get("solution_labels") or []),
        "solution": list(loop_result.get("solution") or []),
        "standing_loop_result_path": LOOP_RESULT_RELATIVE_PATH,
        "initial_loop_result": dict(initial_loop_result or {}),
        "approach_recommendation": dict(approach_recommendation or {}),
        "registry_update": dict(registry_update),
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
        "reproduction_gate": {},
        "solution_labels": [],
        "solution": [],
        "standing_loop_result_path": LOOP_RESULT_RELATIVE_PATH,
        "initial_loop_result": {},
        "approach_recommendation": {},
        "registry_update": {"updated": False, "banked_levels": 0},
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
    if artifact.get("target_game") in DISALLOWED_TARGETS:
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
    if not isinstance(checksum, str) or len(checksum) != 64:
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
        "updated": bool(banked >= 1),
        "path": REGISTRY_RELATIVE_PATH,
        "target_game": game,
        "prior_game_levels": prior,
        "new_game_levels": reached if banked else prior,
        "banked_levels": banked,
        "prior_total_declared": prior_total,
        "new_total_declared": prior_total + banked,
        "reason": "banked_offline_reproduced_level" if banked else _residual_cause(loop_result, prior),
    }


def apply_g50t_registry_bank(
    registry_text: str,
    *,
    artifact: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    registry = yaml.safe_load(registry_text) or {}
    row = dict(_game_row(registry, TARGET_GAME))
    prior = int(row.get("levels_reproduced") or 0)
    reached = int(artifact.get("reproduced_levels") or 0)
    banked = max(0, reached - prior) if artifact.get("offline_reproduced") is True else 0
    prior_total = int(registry.get("reproducible_total_levels") or 0)
    update = {
        "updated": False,
        "path": REGISTRY_RELATIVE_PATH,
        "target_game": TARGET_GAME,
        "prior_game_levels": prior,
        "new_game_levels": prior,
        "banked_levels": 0,
        "prior_total_declared": prior_total,
        "new_total_declared": prior_total,
        "reason": "duplicate_depth" if reached <= prior else "not_offline_reproduced",
    }
    if banked < 1:
        return registry_text, update

    row["reproducibility"] = "reproduced"
    row["levels_reproduced"] = reached
    row["mechanic_class"] = "config_toggle_target_offset"
    row["win_condition"] = (
        "All banked levels use the grounded target-offset predicate: the active player "
        "must reach target.x + 1 and target.y + 1, verified only by the offline "
        "arc_solver_kit.reproduce gate."
    )
    row["action_model"] = (
        "Keyboard ACTION1 up, ACTION2 down, ACTION3 left, ACTION4 right, ACTION5 "
        "commit/rewind clone. L2 uses two clone cycles: first clone holds the "
        "right-hand plate, second clone holds the left-hand plate, then the active "
        "player takes the top route to the target-offset cell."
    )
    row["solver"] = (
        "GameAdapter _g50t in python/carnot/agentic/arc_game_adapters.py plus "
        "scripts/arc_loop_solve.py --game g50t."
    )
    row["reproduce"] = (
        f"Exp4884 {RESULT_RELATIVE_PATH} re-gated {LOOP_RESULT_RELATIVE_PATH} "
        f"offline_reproduced=True, reached_level={reached}, banked +{banked}, "
        f"checksum {artifact.get('reproducibility_checksum')}."
    )
    row["latest_exp4884_levelup_attempt"] = {
        "artifact": RESULT_RELATIVE_PATH,
        "loop_artifact": LOOP_RESULT_RELATIVE_PATH,
        "offline_reproduced": True,
        "reproduced_levels": reached,
        "new_levels_banked": banked,
        "target_rotation": "rotated_off_s5i5_r11l_re86_hidden_state_bound_targets",
        "reproducibility_checksum": artifact.get("reproducibility_checksum"),
    }
    dead_ends = list(row.get("dead_ends") or [])
    retired = "Exp4884 retired the prior g50t adapter-free L2 bounded-search dead end by registering _g50t."
    if retired not in dead_ends:
        dead_ends.append(retired)
    row["dead_ends"] = dead_ends

    start, end = _game_block_bounds(registry_text, TARGET_GAME)
    block = yaml.safe_dump([row], sort_keys=False, width=1000)
    updated = registry_text[:start] + block + registry_text[end:]
    updated = re.sub(
        r"(?m)^(reproducible_total_levels:\s*)\d+\s*$",
        rf"\g<1>{prior_total + banked}",
        updated,
        count=1,
    )
    updated = re.sub(r"(?m)^updated: '[^']+'", "updated: '2026-06-27'", updated, count=1)
    update.update(
        {
            "updated": True,
            "new_game_levels": reached,
            "banked_levels": banked,
            "new_total_declared": prior_total + banked,
            "reason": "banked_offline_reproduced_level",
        }
    )
    return updated, update


def precondition_probe(root: Path = REPO) -> dict[str, Any]:  # pragma: no cover - ARC boundary
    checked: dict[str, Any] = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "target_env_present": (root / "environment_files" / TARGET_GAME).exists(),
        "generator_required": False,
        "rotated_off": ["s5i5", "r11l", "re86"],
        "hidden_state_targets_avoided": ["ka59", "wa30"],
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
    loop_result: Mapping[str, Any] | None = None,
    initial_loop_result: Mapping[str, Any] | None = None,
    approach_recommendation: Mapping[str, Any] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    checked = dict(preconditions_checked or precondition_probe(root))
    if checked.get("offline_arcade_exits_0") is not True or checked.get("target_env_present") is False:
        reason = "offline_env_missing" if checked.get("target_env_present") is False else "offline_arcade_missing"
        artifact = blocked_artifact(
            target_game=TARGET_GAME,
            reason=reason,
            preconditions_checked=checked,
        )
        _write_artifact(root, artifact)
        return artifact

    registry_text = (root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")
    registry = yaml.safe_load(registry_text) or {}
    prior_level = int(_game_row(registry, TARGET_GAME).get("levels_reproduced") or 0)
    prior_total = int(registry.get("reproducible_total_levels") or 0)
    loop = dict(loop_result or _read_json(root / LOOP_RESULT_RELATIVE_PATH))
    loop.setdefault("game", TARGET_GAME)
    initial = dict(initial_loop_result or {})
    registry_update = _registry_update_for_loop(registry_text, loop)
    artifact = build_artifact(
        loop_result=loop,
        prior_level=prior_level,
        prior_total_levels=prior_total,
        preconditions_checked=checked,
        initial_loop_result=initial,
        approach_recommendation=approach_recommendation,
        registry_update=registry_update,
    )
    if artifact["offline_reproduced"]:
        updated_text, final_update = apply_g50t_registry_bank(registry_text, artifact=artifact)
        (root / REGISTRY_RELATIVE_PATH).write_text(updated_text, encoding="utf-8")
        artifact = build_artifact(
            loop_result=loop,
            prior_level=prior_level,
            prior_total_levels=prior_total,
            preconditions_checked=checked,
            initial_loop_result=initial,
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

    current_path = REPO / LOOP_RESULT_RELATIVE_PATH
    if not current_path.exists():
        print(f"missing {LOOP_RESULT_RELATIVE_PATH}; run scripts/arc_loop_solve.py first", file=sys.stderr)
        return 2
    artifact = run_experiment(
        root=REPO,
        loop_result=_read_json(current_path),
        initial_loop_result={},
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
