"""Experiment 4822: ARC rotated level-up attempt ledger.

Spec refs: REQ-ARC-WMTE-4822,
SCENARIO-ARC-WMTE-4822-ROTATION-TARGET,
SCENARIO-ARC-WMTE-4822-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4822-STABLE-ARTIFACT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_4762_levelup_attempt as ledger


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"
ARTIFACT = RESULTS / "experiment_4822_levelup_attempt.json"

EXPERIMENT = "experiment_4822_levelup_attempt"
SCHEMA = "carnot.exp4822.levelup_attempt.v1"
RESULT_RELATIVE_PATH = "results/experiment_4822_levelup_attempt.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4822
PUBLIC_FIRST_CONTACT_TARGETS = ("bp35", "sb26", "lf52")
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "adapter_search_only_no_induction"

SPEC_REFS = [
    "REQ-ARC-WMTE-4822",
    "SCENARIO-ARC-WMTE-4822-ROTATION-TARGET",
    "SCENARIO-ARC-WMTE-4822-REPRODUCTION-GATE",
    "SCENARIO-ARC-WMTE-4822-STABLE-ARTIFACT",
]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; a banked level is success_, a no-bank is "
        "complete_<game>_no_new_level_residual_<cause>."
    ),
    "solve_provenance": (
        "live_agent_self_discovery -- the agent advanced via its OWN attempts; "
        "NOT outer_loop_re (CRITICAL)."
    ),
    "offline_reproduced": "only reproduced levels count.",
    "reproduced_levels": "the new reproducible depth; the monotonic ARC progress metric.",
    "inference_substrate": "live_llm_inference if induction runs (60s floor).",
    "verifier_is_oracle": "the reproduction gate is execution-grounded (true); this is a SOLVE task.",
    "preconditions_checked": (
        "records arcade/env/generator checks so a missing-resource run emits blocked_, "
        "never a fabricated solve."
    ),
}

REQUIRED_FIELDS = (
    "experiment",
    "schema",
    "spec_refs",
    "field_principles",
    "honest_verdict",
    "solve_provenance",
    "offline_reproduced",
    "reproduced_levels",
    "new_levels_banked",
    "inference_substrate",
    "verifier_is_oracle",
    "preconditions_checked",
    "target_game",
    "rotation_selection",
    "approach_recommendation",
    "attempted_games",
    "dead_ends",
    "registry_update",
    "retire_if_same_verdict",
    "reproducibility_checksum",
    "schema_errors",
)

stable_checksum = ledger.stable_checksum
registry_levels = ledger.registry_levels
registry_total_levels = ledger.registry_total_levels


def load_registry(path: Path | None = None) -> dict[str, Any]:
    registry_path = REGISTRY if path is None else Path(path)
    try:
        data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _checksum_is_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _adaptered_games() -> set[str]:  # pragma: no cover - live adapter registry boundary.
    from carnot.agentic import arc_game_adapters

    return set(arc_game_adapters.adaptered_games())


def _recommend_approach(game: str) -> dict[str, Any]:  # pragma: no cover - live registry boundary.
    from carnot.agentic import arc_solve_learning

    return dict(arc_solve_learning.recommend_approach(game))


def _public_rotation(levels: dict[str, int]) -> list[dict[str, Any]]:
    rows = []
    for game in PUBLIC_FIRST_CONTACT_TARGETS:
        prior = int(levels.get(game, 0))
        rows.append(
            {
                "game": game,
                "known": game in levels,
                "prior_level": prior,
                "status": "unreproduced" if prior < 1 else "already_reproduced",
            }
        )
    return rows


def _candidate_from_public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "game": str(row["game"]),
        "prior_level": int(row.get("prior_level") or 0),
        "target_level": 1,
        "reason": "preferred_public_first_contact",
    }


def _shallowest_solved_rows(
    levels: dict[str, int], adaptered_games: set[str]
) -> list[dict[str, Any]]:
    candidates = [
        (int(level), game)
        for game, level in levels.items()
        if game in adaptered_games and int(level) > 0
    ]
    return [
        {
            "game": game,
            "prior_level": prior,
            "target_level": prior + 1,
            "reason": "shallowest_already_solved_deepen",
        }
        for prior, game in sorted(candidates)
    ]


def select_rotation_target(
    registry: dict[str, Any],
    adaptered_games: set[str] | None = None,
    approach_recommendation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    levels = registry_levels(registry)
    adaptered = _adaptered_games() if adaptered_games is None else set(adaptered_games)
    public_rows = _public_rotation(levels)
    first_contact = [
        _candidate_from_public_row(row)
        for row in public_rows
        if bool(row["known"]) and int(row["prior_level"]) < 1
    ]
    deepen_rows = _shallowest_solved_rows(levels, adaptered)

    if first_contact:
        selected = first_contact[0]
        rotate_after = first_contact[1:] + deepen_rows
    elif deepen_rows:
        selected = deepen_rows[0]
        rotate_after = [row for row in deepen_rows[1:] if row["game"] != selected["game"]]
    else:
        return {
            "game": "none",
            "prior_level": 0,
            "target_level": 0,
            "reason": "no_reproduced_standing_loop_target",
            "public_rotation": public_rows,
            "rotate_if_no_bank": [],
            "shallowest_solved_candidates": [],
            "approach_recommendation": {},
        }

    return {
        **selected,
        "public_rotation": public_rows,
        "rotate_if_no_bank": rotate_after,
        "shallowest_solved_candidates": deepen_rows,
        "approach_recommendation": dict(approach_recommendation or {}),
    }


def _attempt_selections(selection: dict[str, Any]) -> list[dict[str, Any]]:
    if selection.get("game") == "none":
        return []
    first = {
        "game": selection["game"],
        "prior_level": int(selection.get("prior_level") or 0),
        "target_level": int(selection.get("target_level") or 0),
        "reason": selection.get("reason"),
    }
    return [first] + [dict(row) for row in selection.get("rotate_if_no_bank", [])]


def summarize_loop_attempt(
    *,
    selection: dict[str, Any],
    loop_result: dict[str, Any],
    loop_result_path: str,
) -> dict[str, Any]:
    attempt = ledger.summarize_loop_attempt(
        game=str(selection["game"]),
        prior_level=int(selection["prior_level"]),
        target_level=int(selection["target_level"]),
        loop_result=loop_result,
        loop_result_path=loop_result_path,
    )
    attempt["target_selection_reason"] = selection.get("reason")
    return attempt


def collect_attempt(selection: dict[str, Any], results_dir: Path | None = None) -> dict[str, Any]:
    root = RESULTS if results_dir is None else Path(results_dir)
    game = str(selection["game"])
    relative = f"results/arc_loop_solve_{game}.json"
    result_path = root / f"arc_loop_solve_{game}.json"
    if not result_path.exists():
        return {
            "game": game,
            "prior_level": int(selection.get("prior_level") or 0),
            "target_level": int(selection.get("target_level") or 0),
            "reached_level": 0,
            "loop_result_path": relative,
            "reproduction_gate": {},
            "offline_reproduced_existing_depth": False,
            "offline_reproduced_new_depth": False,
            "new_levels_banked": 0,
            "residual_cause": "missing_loop_result",
            "loop_solve_provenance": None,
            "learned_verifier_checkpoint": None,
            "solution_labels": [],
            "dead_end": f"{game}: missing standing-loop result; no new reproduced level banked",
            "target_selection_reason": selection.get("reason"),
        }
    data = json.loads(result_path.read_text(encoding="utf-8"))
    loop_result = data if isinstance(data, dict) else {}
    return summarize_loop_attempt(
        selection=selection,
        loop_result=loop_result,
        loop_result_path=relative,
    )


def collect_attempts(
    selection: dict[str, Any], results_dir: Path | None = None
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for attempt_selection in _attempt_selections(selection):
        attempt = collect_attempt(attempt_selection, results_dir=results_dir)
        attempts.append(attempt)
        if int(attempt.get("new_levels_banked") or 0) > 0 and attempt.get(
            "offline_reproduced_new_depth"
        ):
            break
    return attempts


def _best_success(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    for attempt in attempts:
        if int(attempt.get("new_levels_banked") or 0) > 0 and attempt.get(
            "offline_reproduced_new_depth"
        ):
            return attempt
    return None


def _residual_for_verdict(attempts: list[dict[str, Any]]) -> str:
    if any(
        attempt.get("residual_cause") == "reproduced_existing_or_lower_level"
        for attempt in attempts
    ):
        return "existing_depth"
    if attempts:
        return str(attempts[0].get("residual_cause") or "unknown")
    return "no_attempts"


def _target_env_missing(preconditions_checked: dict[str, Any]) -> bool:
    target_env = preconditions_checked.get("target_offline_env")
    return isinstance(target_env, dict) and target_env.get("ok") is False


def build_artifact(
    *,
    registry: dict[str, Any],
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
    preconditions_checked: dict[str, Any],
) -> dict[str, Any]:
    total_before = registry_total_levels(registry)
    success = _best_success(attempts)
    selected_game = str(selection.get("game") or "none")
    blocked_env = _target_env_missing(preconditions_checked)

    if blocked_env:
        target_game = selected_game
        verdict = f"blocked_{target_game}_offline_env_missing"
        offline_reproduced = False
        reproduced_levels = 0
        new_levels = 0
        registry_updated = False
        dead_ends = [f"{target_game}: offline env missing; no reproduced level banked"]
    elif success is not None:
        target_game = str(success["game"])
        reached_level = int(success.get("reached_level") or 0)
        new_levels = int(success.get("new_levels_banked") or 0)
        verdict = f"success_{target_game}_L{reached_level}_offline_reproduced"
        offline_reproduced = True
        reproduced_levels = reached_level
        registry_updated = True
        dead_ends = [str(attempt.get("dead_end")) for attempt in attempts if attempt.get("dead_end")]
    else:
        target_game = str(attempts[0]["game"] if attempts else selected_game)
        cause = _residual_for_verdict(attempts)
        verdict = f"complete_{target_game}_no_new_level_residual_{cause}"
        offline_reproduced = False
        reproduced_levels = 0
        new_levels = 0
        registry_updated = False
        dead_ends = [str(attempt.get("dead_end")) for attempt in attempts if attempt.get("dead_end")]

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": verdict,
        "solve_provenance": SOLVE_PROVENANCE,
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": int(reproduced_levels),
        "new_levels_banked": int(new_levels),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "preconditions_checked": dict(preconditions_checked),
        "target_game": target_game,
        "rotation_selection": dict(selection),
        "approach_recommendation": dict(selection.get("approach_recommendation") or {}),
        "attempted_games": list(attempts),
        "dead_ends": dead_ends,
        "registry_update": {
            "updated": registry_updated,
            "path": REGISTRY_RELATIVE_PATH,
            "reproducible_total_levels_before": int(total_before),
            "reproducible_total_levels_after": int(total_before) + int(new_levels),
            "reason": "banked_new_level" if registry_updated else "no_new_level_banked",
        },
        "retire_if_same_verdict": True,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "schema_errors": [],
    }
    artifact["reproducibility_checksum"] = stable_checksum(artifact)
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    return artifact


def artifact_schema_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")
    principles = payload.get("field_principles")
    for field, principle in FIELD_PRINCIPLES.items():
        if not isinstance(principles, dict) or principles.get(field) != principle:
            errors.append(f"missing_principle:{field}")
    verdict = str(payload.get("honest_verdict") or "")
    if not verdict.startswith(("success_", "complete_", "blocked_")):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance_mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_must_be_true")
    checksum = payload.get("reproducibility_checksum")
    if not _checksum_is_hex(checksum):
        errors.append("invalid_reproducibility_checksum")
    elif checksum != stable_checksum(dict(payload)):
        errors.append("checksum_mismatch")
    if int(payload.get("new_levels_banked") or 0) > 0 and payload.get("offline_reproduced") is not True:
        errors.append("bank_without_offline_reproduction")
    if int(payload.get("new_levels_banked") or 0) == 0 and payload.get("offline_reproduced") is True:
        errors.append("offline_reproduced_true_without_new_bank")
    if payload.get("retire_if_same_verdict") is not True:
        errors.append("retire_if_same_verdict_must_be_true")
    return errors


def write_artifact(payload: dict[str, Any], path: Path | None = None) -> Path:
    output = ARTIFACT if path is None else Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def check_preconditions(selection: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - live environment boundary.
    from carnot.agentic import arc_solver_kit as kit

    target_game = str(selection.get("game") or "none")
    try:
        arc = kit.offline_arcade()
        offline_arcade = {"ok": True, "check": "arc_solver_kit.offline_arcade()"}
    except Exception as exc:
        arc = None
        offline_arcade = {
            "ok": False,
            "check": "arc_solver_kit.offline_arcade()",
            "error": str(exc),
        }

    target_env: dict[str, Any] = {"game": target_game, "ok": False}
    if arc is not None and target_game != "none":
        try:
            arc.make(target_game, scorecard_id=arc.open_scorecard())
            target_env["ok"] = True
        except Exception as exc:
            target_env["error"] = str(exc)
    elif target_game == "none":
        target_env["error"] = "no_target_selected"

    return {
        "AGENTS.md": (REPO / "AGENTS.md").exists(),
        "CODEX.md": (REPO / "CODEX.md").exists(),
        "offline_arcade": offline_arcade,
        "registry_loadable": {"ok": REGISTRY.exists(), "path": REGISTRY_RELATIVE_PATH},
        "target_offline_env": target_env,
        "induction_needed": False,
        "qwen_igpu": {"needed": False, "ok": None},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)

    registry = load_registry(REGISTRY)
    base_selection = select_rotation_target(registry)
    recommendation = (
        _recommend_approach(str(base_selection["game"]))
        if base_selection.get("game") != "none"
        else {}
    )
    selection = select_rotation_target(
        registry,
        approach_recommendation=recommendation,
    )
    preconditions = check_preconditions(selection)
    attempts = collect_attempts(selection)
    artifact = build_artifact(
        registry=registry,
        selection=selection,
        attempts=attempts,
        preconditions_checked=preconditions,
    )
    write_artifact(artifact, ARTIFACT)
    print(f"target_game={artifact['target_game']}")
    print(f"honest_verdict={artifact['honest_verdict']}")
    print(f"new_levels_banked={artifact['new_levels_banked']}")
    print(f"schema_errors={artifact['schema_errors']}")
    return 0 if not artifact["schema_errors"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
