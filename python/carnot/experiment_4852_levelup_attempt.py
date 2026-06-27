"""Experiment 4852: ARC rotated level-up attempt ledger.

Spec refs: REQ-ARC-WMTE-4852,
SCENARIO-ARC-WMTE-4852-ROTATED-TARGET,
SCENARIO-ARC-WMTE-4852-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4852-STABLE-ARTIFACT.
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
ARTIFACT = RESULTS / "experiment_4852_levelup_attempt.json"

EXPERIMENT = "experiment_4852_levelup_attempt"
SCHEMA = "carnot.exp4852.levelup_attempt.v1"
RESULT_RELATIVE_PATH = "results/experiment_4852_levelup_attempt.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4852
PREVIOUS_EXCLUDED_TARGET = "ka59"
SHALLOW_L1_CANDIDATES = ("g50t", "s5i5", "wa30", "r11l")
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "adapter_free_graph_explore_no_induction"

SPEC_REFS = [
    "REQ-ARC-WMTE-4852",
    "SCENARIO-ARC-WMTE-4852-ROTATED-TARGET",
    "SCENARIO-ARC-WMTE-4852-REPRODUCTION-GATE",
    "SCENARIO-ARC-WMTE-4852-STABLE-ARTIFACT",
]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; banked is success_<game>_levelup_banked, "
            "no-bank is complete_<game>_no_new_level_residual_<cause>."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery -- the agent solved via its own attempts/RE; "
            "NOT outer_loop_re (CRITICAL)."
        )
    },
    "target_game": {
        "principle": (
            "the rotated target (must differ from .446 ka59) so coverage sweeps the corpus."
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
    "inference_substrate": {
        "principle": "live_llm_inference if induction runs (60s floor)."
    },
    "preconditions_checked": {
        "principle": (
            "records arcade/env/generator checks; a missing resource emits blocked_, "
            "never a fabricated solve."
        )
    },
}

REQUIRED_FIELDS = (
    "experiment",
    "schema",
    "spec_refs",
    "result_path",
    "field_principles",
    "honest_verdict",
    "solve_provenance",
    "target_game",
    "offline_reproduced",
    "reproduced_levels",
    "new_levels_banked",
    "inference_substrate",
    "verifier_is_oracle",
    "preconditions_checked",
    "rotation_selection",
    "approach_recommendation",
    "attempted_games",
    "dead_ends",
    "registry_update",
    "retire_if_same_verdict",
    "random_seed",
    "reproducibility_checksum",
    "schema_errors",
)

stable_checksum = ledger.stable_checksum
registry_levels = ledger.registry_levels
registry_total_levels = ledger.registry_total_levels


def load_registry(path: Path | None = None) -> dict[str, Any]:  # pragma: no cover
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


def _registry_rows(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("game")): row
        for row in registry.get("games", [])
        if isinstance(row, dict) and row.get("game")
    }


def _dead_ends(row: dict[str, Any]) -> list[str]:
    values = row.get("dead_ends") or []
    return [str(value) for value in values] if isinstance(values, list) else [str(values)]


def _candidate_status(game: str, prior_level: int, dead_ends: list[str]) -> tuple[str, str]:
    joined = "\n".join(dead_ends).lower()
    if prior_level != 1:
        return "skip_not_l1_only", "not_shallow_l1_only"
    if "clone_replay" in joined or "no_bank" in joined:
        return "skip_prior_no_bank_wall", "prior_no_bank_wall"
    if "hidden-state" in joined or "hidden_state" in joined:
        return "skip_hidden_state_bound", "hidden_state_bound_wall"
    if "prefix_rooted" in joined or "stalled" in joined:
        return "skip_prior_stalled_wall", "prior_stalled_wall"
    if game == "s5i5":
        return "selected", "grounded_marker_coverage_delta_adapter_needed"
    return "candidate_unselected", "lower_priority_grounded_delta"


def _candidate_audit(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _registry_rows(registry)
    levels = registry_levels(registry)
    audit = []
    for game in SHALLOW_L1_CANDIDATES:
        row = rows.get(game, {})
        prior = int(levels.get(game, 0))
        dead_ends = _dead_ends(row)
        status, reason = _candidate_status(game, prior, dead_ends)
        audit.append(
            {
                "game": game,
                "prior_level": prior,
                "target_level": prior + 1 if prior > 0 else 1,
                "status": status,
                "reason": reason,
                "dead_ends_consulted": dead_ends,
            }
        )
    return audit


def select_rotation_target(
    registry: dict[str, Any],
    approach_recommendation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audit = _candidate_audit(registry)
    selected = next((row for row in audit if row["status"] == "selected"), None)
    if selected is None:
        return {
            "game": "none",
            "prior_level": 0,
            "target_level": 0,
            "reason": "no_grounded_rotated_l1_delta_available",
            "excluded_previous_target": PREVIOUS_EXCLUDED_TARGET,
            "candidate_audit": audit,
            "approach_recommendation": dict(approach_recommendation or {}),
        }
    return {
        "game": str(selected["game"]),
        "prior_level": int(selected["prior_level"]),
        "target_level": int(selected["target_level"]),
        "reason": str(selected["reason"]),
        "excluded_previous_target": PREVIOUS_EXCLUDED_TARGET,
        "candidate_audit": audit,
        "approach_recommendation": dict(approach_recommendation or {}),
    }


def _gate(loop_result: dict[str, Any]) -> dict[str, Any]:
    gate = loop_result.get("reproduction_gate")
    return dict(gate) if isinstance(gate, dict) else {}


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _reached_level(loop_result: dict[str, Any]) -> int:
    gate = _gate(loop_result)
    return _int_value(
        gate.get("reached_level")
        or loop_result.get("reached_level")
        or loop_result.get("reproduced_levels")
        or 0
    )


def _gate_reproduced(loop_result: dict[str, Any]) -> bool:
    gate = _gate(loop_result)
    return bool(loop_result.get("offline_reproduced") and gate.get("reproduced", True))


def _residual(loop_result: dict[str, Any], reached: int, prior_level: int, new_levels: int) -> str:
    if loop_result.get("status") == "needs_per_game_RE":
        return "needs_per_game_RE"
    if not _gate_reproduced(loop_result):
        return "offline_reproduction_failed"
    if reached <= prior_level or new_levels < 1:
        return "reproduced_existing_or_lower_level"
    return "banked_new_level"


def summarize_loop_attempt(
    *,
    selection: dict[str, Any],
    loop_result: dict[str, Any],
    loop_result_path: str,
) -> dict[str, Any]:
    game = str(selection["game"])
    prior = int(selection.get("prior_level") or 0)
    target = int(selection.get("target_level") or 0)
    reached = _reached_level(loop_result)
    gate_reproduced = _gate_reproduced(loop_result)
    new_levels = max(0, reached - prior) if gate_reproduced else 0
    residual = _residual(loop_result, reached, prior, new_levels)
    if residual == "needs_per_game_RE":
        dead_end = (
            f"{game}: needs_per_game_RE from standing loop; marker-coverage L2 delta "
            "remains unadaptered; no new reproduced level banked"
        )
    elif residual == "banked_new_level":
        dead_end = f"{game}: banked L{reached} over prior L{prior}"
    else:
        dead_end = f"{game}: {residual}; no new reproduced level banked"
    return {
        "game": game,
        "prior_level": prior,
        "target_level": target,
        "reached_level": reached,
        "loop_result_path": loop_result_path,
        "loop_status": loop_result.get("status"),
        "reproduction_gate": _gate(loop_result),
        "offline_reproduced_existing_depth": bool(gate_reproduced and new_levels < 1),
        "offline_reproduced_new_depth": bool(gate_reproduced and new_levels > 0),
        "new_levels_banked": int(new_levels),
        "residual_cause": residual,
        "loop_solve_provenance": loop_result.get("solve_provenance"),
        "learned_verifier_checkpoint": loop_result.get("learned_verifier_checkpoint"),
        "solution_labels": list(loop_result.get("solution_labels") or []),
        "target_selection_reason": selection.get("reason"),
        "dead_end": dead_end,
    }


def collect_attempt(
    selection: dict[str, Any],
    results_dir: Path | None = None,
) -> dict[str, Any]:  # pragma: no cover
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
            "target_selection_reason": selection.get("reason"),
            "dead_end": f"{game}: missing standing-loop result; no new reproduced level banked",
        }
    data = json.loads(result_path.read_text(encoding="utf-8"))
    loop_result = data if isinstance(data, dict) else {}
    return summarize_loop_attempt(
        selection=selection,
        loop_result=loop_result,
        loop_result_path=relative,
    )


def collect_attempts(
    selection: dict[str, Any],
    results_dir: Path | None = None,
) -> list[dict[str, Any]]:  # pragma: no cover
    if selection.get("game") == "none":
        return []
    return [collect_attempt(selection, results_dir=results_dir)]


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
    offline_arcade = preconditions_checked.get("offline_arcade")
    target_missing = isinstance(target_env, dict) and target_env.get("ok") is False
    arcade_missing = isinstance(offline_arcade, dict) and offline_arcade.get("ok") is False
    return target_missing or arcade_missing


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
        verdict = f"success_{target_game}_levelup_banked"
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
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": verdict,
        "solve_provenance": SOLVE_PROVENANCE,
        "target_game": target_game,
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": int(reproduced_levels),
        "new_levels_banked": int(new_levels),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "preconditions_checked": dict(preconditions_checked),
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
    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in payload]
    principles = payload.get("field_principles")
    errors.extend(
        f"missing_principle:{field}"
        for field, principle in FIELD_PRINCIPLES.items()
        if not isinstance(principles, dict) or principles.get(field) != principle
    )
    verdict = str(payload.get("honest_verdict") or "")
    checksum = payload.get("reproducibility_checksum")
    checksum_error = (
        "invalid_reproducibility_checksum"
        if not _checksum_is_hex(checksum)
        else "checksum_mismatch"
        if checksum != stable_checksum(dict(payload))
        else ""
    )
    checks = [
        ("honest_verdict_missing_terminal_prefix", not verdict.startswith(("success_", "complete_", "blocked_"))),
        ("solve_provenance_mismatch", payload.get("solve_provenance") != SOLVE_PROVENANCE),
        ("rotated_target_must_not_be_ka59", payload.get("target_game") == PREVIOUS_EXCLUDED_TARGET),
        ("target_game_missing", not payload.get("target_game")),
        ("inference_substrate_mismatch", payload.get("inference_substrate") != INFERENCE_SUBSTRATE),
        ("verifier_is_oracle_must_be_true", payload.get("verifier_is_oracle") is not True),
        (
            "bank_without_offline_reproduction",
            int(payload.get("new_levels_banked") or 0) > 0 and payload.get("offline_reproduced") is not True,
        ),
        (
            "offline_reproduced_true_without_new_bank",
            int(payload.get("new_levels_banked") or 0) == 0 and payload.get("offline_reproduced") is True,
        ),
        ("retire_if_same_verdict_must_be_true", payload.get("retire_if_same_verdict") is not True),
        ("experiment_mismatch", payload.get("experiment") != EXPERIMENT),
        ("schema_mismatch", payload.get("schema") != SCHEMA),
        ("spec_refs_mismatch", payload.get("spec_refs") != SPEC_REFS),
        ("result_path_mismatch", payload.get("result_path") != RESULT_RELATIVE_PATH),
    ]
    errors.extend([checksum_error] if checksum_error else [])
    errors.extend(name for name, failed in checks if failed)
    return errors


def write_artifact(payload: dict[str, Any], path: Path | None = None) -> Path:
    output = ARTIFACT if path is None else Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def check_preconditions(selection: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
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


def _recommend_approach(game: str) -> dict[str, Any]:  # pragma: no cover
    from carnot.agentic import arc_solve_learning

    return dict(arc_solve_learning.recommend_approach(game))


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)

    registry = load_registry(REGISTRY)
    base_selection = select_rotation_target(registry)
    recommendation = (
        _recommend_approach(str(base_selection["game"]))
        if base_selection.get("game") != "none"
        else {}
    )
    selection = select_rotation_target(registry, approach_recommendation=recommendation)
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
