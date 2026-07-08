"""Experiment 5410: ARC live trajectory-frontier level-up attempt.

Spec refs: REQ-ARC-FCP-5410, SCENARIO-ARC-FCP-5410.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from carnot.agentic.arc_live_trajectory_frontier import LiveTrajectoryFrontierGenerator


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5410
EXPERIMENT = "experiment_5410_arc_live_trajectory_frontier_levelup_v492"
MILESTONE = "2026.07.492"
RESULT_RELATIVE_PATH = "results/experiment_5410_arc_live_trajectory_frontier_levelup_v492.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5410", "SCENARIO-ARC-FCP-5410"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
DEFAULT_BUDGET = 36
DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5410_arc_live_trajectory_frontier_levelup_v492.py -q --no-cov",
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5410_arc_live_trajectory_frontier_levelup_v492.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/agentic/arc_live_trajectory_frontier.py "
        "python/carnot/experiment_5410_arc_live_trajectory_frontier_levelup_v492.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "registry_precheck_done": {
        "principle": "bare bool proving no duplicate solve attempt starts before reading the registry."
    },
    "target_game": {"principle": "selected game id for reproducibility."},
    "target_level": {"principle": "selected target level label for reproducibility."},
    "solve_provenance": {
        "principle": "must be live_agent_self_discovery for credited progress."
    },
    "offline_reproduced": {
        "principle": "legacy ARC lint field; true only for a live-agent self-discovered registry-compatible new level."
    },
    "attempt_count": {"principle": "bounded live-agent effort count."},
    "frontier_expansion_count": {
        "principle": "number of trajectory/frontier prefixes emitted by the new mechanism."
    },
    "salience_routes_used": {
        "principle": "auditable blob/color salience routes used by prefix generation."
    },
    "uncertainty_rejections": {
        "principle": "low-support inferred dynamics rejected by the gate."
    },
    "reproduced_levels": {"principle": "registry-compatible new level count."},
    "arc_new_level_banked": {"principle": "standing ARC floor success flag."},
    "duplicate_solve_avoided": {
        "principle": "registry discipline prevents duplicate solved-level credit."
    },
    "no_offline_bfs": {"principle": "must be true; forbidden solve path was not used."},
    "no_per_game_adapter": {
        "principle": "must be true; no hand per-game shortcut was used."
    },
    "inference_substrate": {
        "principle": "must be offline_arcade_live_agent_runtime_self_discovery_no_llm."
    },
    "honest_verdict": {
        "principle": "terminal status starts with complete:, honest_null:, or blocked:."
    },
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


class _NoOpProposer:  # pragma: no cover - ARC runtime boundary
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "disabled_exp5410_no_live_llm"

    def world_model_candidates(self, _game: str) -> list[Any]:
        return []


def _registry_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("game")): row
        for row in registry.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    }


def _registry_total(registry: Mapping[str, Any]) -> int:
    return int(registry.get("reproducible_total_levels") or 0)


def _level_label(level: int) -> str:
    return f"L{max(1, int(level))}"


def select_target_after_precheck(
    registry: Mapping[str, Any],
    *,
    preferred: str = "re86",
    preferred_level: int = 3,
    alternates: Sequence[str] = ("sb26", "bp35", "lf52", "g50t", "cd82", "sp80", "su15"),
) -> dict[str, Any]:
    """REQ-ARC-FCP-5410: choose a target without duplicating a solved level."""

    rows = _registry_rows(registry)
    preferred_before = int((rows.get(str(preferred)) or {}).get("levels_reproduced") or 0)
    if preferred_before < int(preferred_level):
        target_number = min(int(preferred_level), preferred_before + 1)
        return {
            "status": "selected",
            "registry_precheck_done": True,
            "target_game": str(preferred),
            "target_level_before": preferred_before,
            "target_level_number": target_number,
            "target_level": _level_label(target_number),
            "duplicate_solve_avoided": True,
            "selection_reason": "preferred_re86_l3_not_live_reached",
        }
    for game in alternates:
        row = rows.get(str(game))
        if row is None:
            continue
        before = int(row.get("levels_reproduced") or 0)
        return {
            "status": "selected",
            "registry_precheck_done": True,
            "target_game": str(game),
            "target_level_before": before,
            "target_level_number": before + 1,
            "target_level": _level_label(before + 1),
            "duplicate_solve_avoided": True,
            "selection_reason": "preferred_re86_l3_already_reached_rotated_target",
        }
    return {
        "status": "blocked_duplicate_solve",
        "registry_precheck_done": True,
        "target_game": str(preferred),
        "target_level_before": preferred_before,
        "target_level_number": int(preferred_level),
        "target_level": _level_label(preferred_level),
        "duplicate_solve_avoided": True,
        "selection_reason": "preferred_target_already_reached_no_alternate_available",
    }


def offline_arcade_available() -> bool:  # pragma: no cover - environment probe
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
    except Exception:
        return False
    return True


def _action_label(action: int | str, data: Any) -> str:
    return json.dumps({"action": action, "data": data}, sort_keys=True, separators=(",", ":"))


def _apply_action_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover
    if label == "RESET":
        return env.reset()
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


def run_live_trajectory_frontier_attempt(  # pragma: no cover - ARC runtime boundary
    *,
    root: Path,
    selection: Mapping[str, Any],
    budget: int = DEFAULT_BUDGET,
) -> dict[str, Any]:
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of

    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    game = str(selection["target_game"])
    target_before = int(selection["target_level_before"])
    target_number = int(selection["target_level_number"])
    generator = LiveTrajectoryFrontierGenerator(min_support=1, max_uncertainty=0.51)
    try:
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        policy = E3AgentPolicy(
            game,
            proposer=_NoOpProposer(),
            explore_budget=max(1, int(budget)),
            target_levels=max(1, target_number),
            value_head=None,
            frame_change_scorer=None,
            candidate_router=None,
            action_effect_expansion_prior=False,
            action_prior=generator,
            qd_generator=generator,
            goal_bias=None,
            goal_candidate_guidance=False,
            active_probe_controller=False,
            go_explore_archive=False,
        )
        frames: list[Any] = []
        latest = None
        labels: list[str] = []
        max_level = target_before
        newly_reached: list[str] = []
        for _index in range(max(1, int(budget))):
            if policy.is_done(frames, latest):
                break
            kind, data = policy.next_move(frames, latest)
            if kind == "RESET":
                latest = env.reset()
                if labels:
                    labels.append("RESET")
            elif kind is None:
                break
            else:
                latest = env.step(getattr(GameAction, f"ACTION{int(kind)}"), data=data)
                labels.append(_action_label(int(kind), data))
            observed_level = int(_level_of(latest))
            if observed_level > max_level:
                newly_reached.extend(_level_label(level) for level in range(max_level + 1, observed_level + 1))
            max_level = max(max_level, observed_level)
            frames.append(latest)
            if max_level >= target_number or latest is None:
                break
        gate: dict[str, Any] = {
            "game": game,
            "claimed_level": 0,
            "reached_level": 0,
            "reproduced": False,
            "mode": "offline_reproduction_gate_no_new_level_claim",
        }
        if max_level > target_before and labels:
            gate = dict(kit.reproduce(game, labels, _apply_action_label, claimed_level=max_level))
        reproduced = bool(gate.get("reproduced")) and max_level >= target_number
        diagnostics = generator.diagnostics()
        salience_routes = list(
            dict.fromkeys(
                list(diagnostics["salience_routes_used"]) + [generator.as_dict()["source"]]
            )
        )
        return {
            "target_game": game,
            "target_level_before": target_before,
            "target_level": _level_label(target_number),
            "attempt_count": len([label for label in labels if label != "RESET"]),
            "max_level_reached": int(max_level),
            "offline_reproduced": bool(reproduced),
            "new_reproduced_levels": 1 if reproduced else 0,
            "failure_mode": None if reproduced else "bounded_budget_no_levelup",
            "frontier_expansion_count": int(diagnostics["frontier_expansion_count"]),
            "frontier_expansions": list(diagnostics["frontier_expansions"]),
            "salience_routes_used": salience_routes,
            "uncertainty_rejections": int(diagnostics["uncertainty_rejections"]),
            "verifier_observations": list(diagnostics["verifier_observations"]),
            "newly_reached_levels": newly_reached,
            "solution_labels": list(labels) if reproduced else [],
            "reproduction_gate": gate,
            "runtime_self_discovery": True,
            "no_offline_bfs": True,
            "no_per_game_adapter": True,
            "root": str(root),
        }
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def _new_reproduced_levels(attempt: Mapping[str, Any]) -> int:
    if attempt.get("offline_reproduced") is not True:
        return 0
    return max(0, int(attempt.get("new_reproduced_levels") or 0))


def build_artifact(
    *,
    selection: Mapping[str, Any],
    registry_total_before: int,
    attempt: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    blocked = bool(attempt.get("blocked")) or selection.get("status") == "blocked_duplicate_solve"
    target_game = str(selection.get("target_game") or "re86")
    target_level = str(selection.get("target_level") or "L1")
    no_offline_bfs = attempt.get("no_offline_bfs", True) is True
    no_per_game_adapter = attempt.get("no_per_game_adapter", True) is True
    new_reproduced = _new_reproduced_levels(attempt)
    solution_labels = bool(attempt.get("solution_labels"))
    can_bank = bool(
        new_reproduced >= 1
        and solution_labels
        and attempt.get("runtime_self_discovery") is True
        and bool(selection.get("duplicate_solve_avoided"))
        and no_offline_bfs
        and no_per_game_adapter
    )
    status = "complete" if can_bank else "blocked" if blocked else "honest_null"
    failure_mode = attempt.get("failure_mode")
    if status == "blocked" and not failure_mode:
        failure_mode = (
            "duplicate_solve_precheck"
            if selection.get("status") == "blocked_duplicate_solve"
            else "missing_harness_access"
        )
    if status == "honest_null" and not failure_mode:
        failure_mode = "bounded_budget_no_levelup"
    frontier_count = int(attempt.get("frontier_expansion_count") or 0)
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "registry_precheck_done": bool(selection.get("registry_precheck_done")),
        "target_game": target_game,
        "target_level": target_level,
        "solve_provenance": SOLVE_PROVENANCE,
        "offline_reproduced": bool(can_bank),
        "attempt_count": int(attempt.get("attempt_count") or 0),
        "frontier_expansion_count": frontier_count,
        "salience_routes_used": list(attempt.get("salience_routes_used") or []),
        "uncertainty_rejections": int(attempt.get("uncertainty_rejections") or 0),
        "reproduced_levels": int(new_reproduced if can_bank else 0),
        "arc_new_level_banked": bool(can_bank),
        "duplicate_solve_avoided": bool(selection.get("duplicate_solve_avoided")),
        "no_offline_bfs": bool(no_offline_bfs),
        "no_per_game_adapter": bool(no_per_game_adapter),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"complete: {target_game} {target_level} live trajectory frontier reproduced"
            if can_bank
            else f"blocked: {failure_mode}"
            if status == "blocked"
            else f"honest_null: {target_game} {target_level} {failure_mode}"
        ),
        "registry_total_before": int(registry_total_before),
        "registry_total_after": int(registry_total_before) + int(new_reproduced if can_bank else 0),
        "target_selection": dict(selection),
        "attempts": [dict(attempt)] if attempt else [],
        "frontier_expansions": list(attempt.get("frontier_expansions") or []),
        "verifier_observations": list(attempt.get("verifier_observations") or []),
        "newly_reached_levels": list(attempt.get("newly_reached_levels") or []),
        "failure_mode": None if can_bank else str(failure_mode or ""),
        "preconditions_checked": dict(preconditions_checked),
        "tests_run": list(tests_run),
        "duration_s": float(duration_s),
    }
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in artifact
    ]
    if artifact.get("status") not in {"complete", "honest_null", "blocked"}:
        errors.append("status must be complete, honest_null, or blocked")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    for field in (
        "registry_precheck_done",
        "offline_reproduced",
        "arc_new_level_banked",
        "duplicate_solve_avoided",
        "no_offline_bfs",
        "no_per_game_adapter",
    ):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
    for field in ("registry_precheck_done", "duplicate_solve_avoided", "no_offline_bfs", "no_per_game_adapter"):
        if artifact.get(field) is not True:
            errors.append(f"{field} must be true")
    for field in (
        "attempt_count",
        "frontier_expansion_count",
        "uncertainty_rejections",
        "reproduced_levels",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
    for field in ("target_game", "target_level"):
        if not isinstance(artifact.get(field), str) or not artifact.get(field):
            errors.append(f"{field} must be non-empty string")
    if not isinstance(artifact.get("salience_routes_used"), list):
        errors.append("salience_routes_used must be list")
    if artifact.get("offline_reproduced") is True and artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("offline_reproduced true requires live_agent_self_discovery")
    if artifact.get("status") == "complete":
        if artifact.get("offline_reproduced") is not True:
            errors.append("complete artifact requires offline_reproduced true")
        if artifact.get("arc_new_level_banked") is not True:
            errors.append("complete artifact requires arc_new_level_banked true")
        if type(artifact.get("reproduced_levels")) is int and artifact["reproduced_levels"] < 1:
            errors.append("complete artifact requires reproduced_levels >= 1")
    else:
        if artifact.get("offline_reproduced") is True:
            errors.append("non-complete artifact cannot set offline_reproduced true")
        if artifact.get("arc_new_level_banked") is True:
            errors.append("arc_new_level_banked requires complete status")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "honest_null:", "blocked:")):
        errors.append("honest_verdict must start with complete:, honest_null:, or blocked:")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def _write_artifact(root: Path, artifact: Mapping[str, Any]) -> None:
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    *,
    root: Path = REPO,
    budget: int = DEFAULT_BUDGET,
    attempt_runner: Callable[..., Mapping[str, Any]] = run_live_trajectory_frontier_attempt,
    offline_arcade_check: Callable[[], bool] = offline_arcade_available,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(root)
    spec_path = root / SPEC_RELATIVE_PATH
    registry_path = root / REGISTRY_RELATIVE_PATH
    preconditions = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "spec_has_req_5410": (
            "REQ-ARC-FCP-5410" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
            else False
        ),
        "registry_present": registry_path.exists(),
        "offline_arcade_available": False,
        "no_offline_bfs": True,
        "no_per_game_adapter": True,
    }
    registry = (
        yaml.safe_load(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    )
    registry = registry or {}
    selection = select_target_after_precheck(registry)
    ready_without_arcade = (
        preconditions["AGENTS.md"]
        and preconditions["CODEX.md"]
        and preconditions["spec_has_req_5410"]
        and preconditions["registry_present"]
        and selection.get("status") != "blocked_duplicate_solve"
    )
    if not ready_without_arcade:
        attempt: Mapping[str, Any] = {
            "blocked": True,
            "failure_mode": "duplicate_solve_precheck"
            if selection.get("status") == "blocked_duplicate_solve"
            else "missing_harness_access",
            "attempt_count": 0,
            "no_offline_bfs": True,
            "no_per_game_adapter": True,
        }
    else:
        preconditions["offline_arcade_available"] = bool(offline_arcade_check())
        if not preconditions["offline_arcade_available"]:
            attempt = {
                "blocked": True,
                "failure_mode": "missing_harness_access",
                "attempt_count": 0,
                "no_offline_bfs": True,
                "no_per_game_adapter": True,
            }
        else:
            attempt = attempt_runner(root=root, selection=selection, budget=budget)
    artifact = build_artifact(
        selection=selection,
        registry_total_before=_registry_total(registry),
        attempt=attempt,
        preconditions_checked=preconditions,
        tests_run=tests_run,
        duration_s=time.monotonic() - started,
    )
    validate_artifact(artifact)
    _write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
