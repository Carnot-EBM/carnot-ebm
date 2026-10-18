"""Experiment 5385: ARC geometric salience live-path level-up attempt.

Spec refs: REQ-ARC-FCP-5385, SCENARIO-ARC-FCP-5385.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from carnot.agentic.arc_agi3_live_adapter import ArcAction
from carnot.agentic.arc_frame_change_predictor import rank_arc_actions
from carnot.agentic.arc_geometric_salience import GeometricSaliencePrior


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5385
EXPERIMENT = "experiment_5385_arc_geometric_salience_live_path_v490"
MILESTONE = "2026.07.490"
RESULT_RELATIVE_PATH = "results/experiment_5385_arc_geometric_salience_live_path_v490.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5385", "SCENARIO-ARC-FCP-5385"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
DEFAULT_BUDGET = 36
DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5385_arc_geometric_salience_live_path_v490.py -q --no-cov",
    ".venv/bin/pytest tests/python/test_arc_submitted_agent_parity.py -q --no-cov",
    ".venv/bin/pytest tests/python/test_experiment_5385_arc_geometric_salience_live_path_v490.py -q -o addopts= --cov=python/carnot/agentic/arc_geometric_salience.py --cov=python/carnot/experiment_5385_arc_geometric_salience_live_path_v490.py --cov-report=term-missing --cov-fail-under=100",
    ".venv/bin/ruff check python/carnot/agentic/arc_geometric_salience.py python/carnot/agentic/arc_competition_agent.py python/carnot/experiment_5385_arc_geometric_salience_live_path_v490.py tests/python/test_experiment_5385_arc_geometric_salience_live_path_v490.py",
    ".venv/bin/ruff format --check python/carnot/agentic/arc_geometric_salience.py python/carnot/agentic/arc_competition_agent.py python/carnot/experiment_5385_arc_geometric_salience_live_path_v490.py tests/python/test_experiment_5385_arc_geometric_salience_live_path_v490.py",
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "status": {"principle": "complete, honest_null, or duplicate_blocked with evidence."},
    "solve_provenance": {
        "principle": "must equal live_agent_self_discovery for any credited solve."
    },
    "registry_precheck_done": {"principle": "must be true."},
    "target_game": {"principle": "selected game id."},
    "target_level_before": {"principle": "reproduced level count before this task."},
    "attempted_level": {"principle": "target level attempted."},
    "geometric_salience_live_reachable": {
        "principle": "true only if the live agent can use the salience signal without outer-loop help."
    },
    "hyperbolic_or_geodesic_ranking_enabled": {
        "principle": "whether the GeoWorld-inspired ranking was active."
    },
    "live_attempt_count": {"principle": "number of live attempts."},
    "offline_reproduced": {
        "principle": "true only if a live-agent-discovered solve was replayed/reproduced for banking; must not mean offline BFS or outer-loop reverse engineering."
    },
    "no_outer_loop_re": {"principle": "must be true."},
    "no_per_game_adapter": {"principle": "must be true."},
    "no_duplicate_solve": {"principle": "must be true for credited deliverables."},
    "reproduced_levels": {
        "principle": "level count after this task; for a credited first-contact solve this must satisfy reproduced_levels>=1, and for a deeper solve it must be at least target_level_before+1."
    },
    "new_level_banked": {
        "principle": "true only if the live agent self-discovered a new reproducible level."
    },
    "failure_mode": {"principle": "if no bank, concrete live-path blocker."},
    "honest_verdict": {"principle": "one-line ARC outcome."},
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


class _NoOpProposer:  # pragma: no cover - ARC runtime boundary
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "disabled_exp5385_no_live_llm"

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


def select_target_after_precheck(
    registry: Mapping[str, Any],
    *,
    preferred: str = "re86",
    alternates: Sequence[str] = ("sb26", "bp35", "lf52", "g50t", "cd82", "sp80", "su15"),
) -> dict[str, Any]:
    """REQ-ARC-FCP-5385: pick a non-duplicate next-level live-path target."""

    rows = _registry_rows(registry)
    preferred_row = rows.get(str(preferred), {})
    preferred_before = int(preferred_row.get("levels_reproduced") or 0)
    if preferred_before < 3:
        return {
            "status": "selected",
            "registry_precheck_done": True,
            "target_game": str(preferred),
            "target_level_before": preferred_before,
            "attempted_level": preferred_before + 1,
            "no_duplicate_solve": True,
            "selection_reason": "preferred_re86_l3_not_banked",
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
            "attempted_level": before + 1,
            "no_duplicate_solve": True,
            "selection_reason": "re86_l3_already_banked_rotated_target",
        }
    return {
        "status": "duplicate_blocked",
        "registry_precheck_done": True,
        "target_game": str(preferred),
        "target_level_before": preferred_before,
        "attempted_level": 3,
        "no_duplicate_solve": False,
        "selection_reason": "preferred_re86_l3_already_banked_no_alternate",
    }


def _default_geometric_frame() -> SimpleNamespace:
    grid = np.zeros((20, 20), dtype=np.int16)
    grid[0, :] = 16
    grid[3:5, 3:5] = 8
    grid[14:16, 14:16] = 8
    return SimpleNamespace(frame=grid, available_actions=[6])


def _default_geometric_candidates() -> list[ArcAction]:
    return [
        ArcAction(6, {"x": 3, "y": 3}, "far_equal_button"),
        ArcAction(6, {"x": 14, "y": 14}, "near_changed_button"),
    ]


def _default_observed_transition() -> tuple[SimpleNamespace, SimpleNamespace]:
    before = _default_geometric_frame().frame.copy()
    after = before.copy()
    after[14:16, 14:16] = 9
    return SimpleNamespace(frame=before), SimpleNamespace(frame=after)


def _rank_of_source(candidates: Sequence[Any], source: str) -> int | None:
    for index, candidate in enumerate(candidates):
        if getattr(candidate, "source", None) == source:
            return index
    return None


def measure_geometric_rank_delta(
    frame: Any | None = None,
    candidates: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Measure whether observed transition geometry changes live candidate order."""

    frame = _default_geometric_frame() if frame is None else frame
    rows = list(_default_geometric_candidates() if candidates is None else candidates)
    prior = GeometricSaliencePrior(geodesic_weight=500.0)
    before_ranked = rank_arc_actions(frame, rows, prior=prior.base_prior)
    before, after = _default_observed_transition()
    prior.observe_transition(before, 6, {"x": 14, "y": 14}, after)
    after_ranked = rank_arc_actions(frame, rows, prior=prior)
    before_rank = _rank_of_source(before_ranked, "near_changed_button")
    after_rank = _rank_of_source(after_ranked, "near_changed_button")
    delta = (
        None if before_rank is None or after_rank is None else int(before_rank) - int(after_rank)
    )
    return {
        "before_rank": before_rank,
        "after_rank": after_rank,
        "geometric_rank_delta": delta,
        "legacy_first_source": getattr(before_ranked[0], "source", None) if before_ranked else None,
        "geometric_first_source": getattr(after_ranked[0], "source", None)
        if after_ranked
        else None,
        "prior_diagnostics": prior.as_dict(),
    }


def geometric_salience_live_diagnostics() -> dict[str, Any]:
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    prior = GeometricSaliencePrior()
    policy = E3AgentPolicy(
        "re86",
        proposer=None,
        value_head=None,
        frame_change_scorer=None,
        candidate_router=None,
        action_effect_expansion_prior=False,
        action_prior=prior,
        goal_bias=None,
        goal_candidate_guidance=False,
        active_probe_controller=False,
    )
    action_prior = policy.explorer.action_prior
    return {
        "geometric_salience_live_reachable": isinstance(action_prior, GeometricSaliencePrior),
        "hyperbolic_or_geodesic_ranking_enabled": bool(
            isinstance(action_prior, GeometricSaliencePrior)
            and action_prior.hyperbolic_or_geodesic_ranking_enabled
        ),
        "action_prior_source": action_prior.as_dict().get("source")
        if hasattr(action_prior, "as_dict")
        else None,
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


def run_live_geometric_salience_attempt(  # pragma: no cover - ARC runtime boundary
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
    target_level_before = int(selection["target_level_before"])
    attempted_level = int(selection["attempted_level"])
    prior = GeometricSaliencePrior()
    try:
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        policy = E3AgentPolicy(
            game,
            proposer=_NoOpProposer(),
            explore_budget=max(1, int(budget)),
            target_levels=max(1, attempted_level),
            value_head=None,
            frame_change_scorer=None,
            candidate_router=None,
            action_effect_expansion_prior=False,
            action_prior=prior,
            goal_bias=None,
            goal_candidate_guidance=False,
            active_probe_controller=False,
            go_explore_archive=False,
        )
        frames: list[Any] = []
        latest = None
        labels: list[str] = []
        actions_taken = 0
        max_level = target_level_before
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
                actions_taken += 1
            observed_level = int(_level_of(latest))
            max_level = max(max_level, observed_level)
            frames.append(latest)
            if max_level >= attempted_level or latest is None:
                break
        gate: dict[str, Any] = {
            "game": game,
            "claimed_level": 0,
            "reached_level": 0,
            "reproduced": False,
            "mode": "offline_reproduction_gate_no_new_level_claim",
        }
        if max_level > target_level_before and labels:
            gate = dict(kit.reproduce(game, labels, _apply_action_label, claimed_level=max_level))
        offline_reproduced = bool(gate.get("reproduced")) and max_level >= attempted_level
        reproduced_levels = int(max_level if offline_reproduced else target_level_before)
        return {
            "target_game": game,
            "target_level_before": target_level_before,
            "attempted_level": attempted_level,
            "actions_taken": int(actions_taken),
            "max_level_reached": int(max_level),
            "offline_reproduced": bool(offline_reproduced),
            "reproduced_levels": reproduced_levels,
            "new_level_banked": bool(offline_reproduced),
            "failure_mode": "" if offline_reproduced else "bounded_budget_no_levelup",
            "solution_labels": list(labels) if offline_reproduced else [],
            "reproduction_gate": gate,
            "self_discovery_lever": "geometric_salience_action_prior",
            "runtime_self_discovery": True,
            "offline_source_reading_used": False,
            "offline_ground_truth_bfs_used": False,
            "per_game_adapter_used": False,
            "no_outer_loop_re": True,
            "root": str(root),
            "geometric_prior_diagnostics": prior.as_dict(),
        }
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def build_artifact(
    *,
    selection: Mapping[str, Any],
    registry_total_before: int,
    live_diagnostics: Mapping[str, Any],
    rank_measurement: Mapping[str, Any],
    attempt: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    target_game = str(selection.get("target_game") or "none")
    target_level_before = int(selection.get("target_level_before") or 0)
    attempted_level = int(selection.get("attempted_level") or target_level_before + 1)
    no_outer_loop_re = True
    no_per_game_adapter = True
    no_duplicate = bool(selection.get("no_duplicate_solve"))
    live_attempt_count = 0 if selection.get("status") == "duplicate_blocked" else 1
    reproduced_after = int(
        attempt.get("reproduced_levels") or attempt.get("max_level_reached") or target_level_before
    )
    reproduced_after = max(target_level_before, reproduced_after)
    live_solution_labels = bool(attempt.get("solution_labels"))
    new_level_banked = bool(
        attempt.get("offline_reproduced") is True
        and reproduced_after >= attempted_level
        and live_solution_labels
        and no_outer_loop_re
        and no_per_game_adapter
        and no_duplicate
    )
    status = (
        "complete"
        if new_level_banked
        else "duplicate_blocked"
        if selection.get("status") == "duplicate_blocked"
        else "honest_null"
    )
    failure_mode = str(attempt.get("failure_mode") or "")
    if not new_level_banked and not failure_mode:
        failure_mode = (
            "duplicate_target_already_reproducible"
            if status == "duplicate_blocked"
            else "bounded_budget_no_levelup"
        )
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "solve_provenance": SOLVE_PROVENANCE,
        "registry_precheck_done": bool(selection.get("registry_precheck_done")),
        "target_game": target_game,
        "target_level_before": target_level_before,
        "attempted_level": attempted_level,
        "geometric_salience_live_reachable": bool(
            live_diagnostics.get("geometric_salience_live_reachable")
        ),
        "hyperbolic_or_geodesic_ranking_enabled": bool(
            live_diagnostics.get("hyperbolic_or_geodesic_ranking_enabled")
        ),
        "live_attempt_count": int(live_attempt_count),
        "offline_reproduced": bool(new_level_banked),
        "no_outer_loop_re": no_outer_loop_re,
        "no_per_game_adapter": no_per_game_adapter,
        "no_duplicate_solve": no_duplicate,
        "reproduced_levels": int(reproduced_after if new_level_banked else target_level_before),
        "new_level_banked": bool(new_level_banked),
        "failure_mode": "" if new_level_banked else failure_mode,
        "honest_verdict": (
            f"banked: {target_game} L{attempted_level} live geometric salience reproduced"
            if new_level_banked
            else f"duplicate-blocked: {target_game} L{attempted_level} already reproducible"
            if status == "duplicate_blocked"
            else f"no-bank: {target_game} L{attempted_level} {failure_mode}"
        ),
        "registry_total_before": int(registry_total_before),
        "registry_total_after": int(registry_total_before) + (1 if new_level_banked else 0),
        "preconditions_checked": dict(preconditions_checked),
        "target_selection": dict(selection),
        "rank_measurement": dict(rank_measurement),
        "live_diagnostics": dict(live_diagnostics),
        "attempt": dict(attempt),
        "tests_run": list(tests_run),
        "duration_s": float(duration_s),
    }
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in artifact
    ]
    bool_fields = (
        "registry_precheck_done",
        "geometric_salience_live_reachable",
        "hyperbolic_or_geodesic_ranking_enabled",
        "live_attempt_count",
        "offline_reproduced",
        "no_duplicate_solve",
        "new_level_banked",
    )
    int_fields = ("target_level_before", "attempted_level", "reproduced_levels")
    for field in bool_fields:
        if field == "live_attempt_count":
            continue
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
    for field in ("no_outer_loop_re", "no_per_game_adapter"):
        if artifact.get(field) is not True:
            errors.append(f"{field} must be bare true")
    for field in int_fields:
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
    if type(artifact.get("live_attempt_count")) is not int:
        errors.append("live_attempt_count must be bare int")
    if artifact.get("status") not in {"complete", "honest_null", "duplicate_blocked"}:
        errors.append("status must be complete, honest_null, or duplicate_blocked")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    if artifact.get("registry_precheck_done") is not True:
        errors.append("registry_precheck_done must be true")
    if not isinstance(artifact.get("failure_mode"), str):
        errors.append("failure_mode must be a string")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("banked:", "no-bank:", "duplicate-blocked:")):
        errors.append("honest_verdict must be a one-line ARC outcome")
    if artifact.get("status") == "complete":
        if artifact.get("offline_reproduced") is not True:
            errors.append("credited solve must set offline_reproduced true")
        if artifact.get("new_level_banked") is not True:
            errors.append("credited solve must set new_level_banked true")
        if (
            type(artifact.get("reproduced_levels")) is int
            and type(artifact.get("attempted_level")) is int
            and artifact["reproduced_levels"] < artifact["attempted_level"]
        ):
            errors.append("credited solve must reproduce at least attempted_level")
        attempt = artifact.get("attempt")
        if not isinstance(attempt, Mapping) or not attempt.get("solution_labels"):
            errors.append("credited solve requires live-agent solution labels")
    elif artifact.get("offline_reproduced") is True:
        errors.append("non-complete artifact cannot set offline_reproduced true")
    if artifact.get("new_level_banked") is True and artifact.get("status") != "complete":
        errors.append("new_level_banked requires complete status")
    if artifact.get("status") != "complete" and not artifact.get("failure_mode"):
        errors.append("no-bank or duplicate-blocked artifact must record failure_mode")
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
    attempt_runner: Callable[..., Mapping[str, Any]] = run_live_geometric_salience_attempt,
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
        "spec_has_req_5385": (
            "REQ-ARC-FCP-5385" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
            else False
        ),
        "registry_present": registry_path.exists(),
        "offline_arcade_available": False,
        "no_outer_loop_re": True,
        "no_per_game_adapter": True,
    }
    registry = (
        yaml.safe_load(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    )
    registry = registry or {}
    selection = select_target_after_precheck(registry)
    live_diagnostics = geometric_salience_live_diagnostics()
    rank_measurement = measure_geometric_rank_delta()
    if selection.get("status") == "duplicate_blocked":
        attempt: Mapping[str, Any] = {
            "offline_reproduced": False,
            "reproduced_levels": int(selection.get("target_level_before") or 0),
            "new_level_banked": False,
            "failure_mode": "duplicate_target_already_reproducible",
        }
    else:
        preconditions["offline_arcade_available"] = bool(offline_arcade_check())
        if not (
            preconditions["AGENTS.md"]
            and preconditions["CODEX.md"]
            and preconditions["spec_has_req_5385"]
            and preconditions["registry_present"]
            and preconditions["offline_arcade_available"]
        ):
            attempt = {
                "offline_reproduced": False,
                "reproduced_levels": int(selection.get("target_level_before") or 0),
                "new_level_banked": False,
                "failure_mode": "preconditions_missing_or_offline_arcade_unavailable",
            }
        else:
            attempt = attempt_runner(root=root, selection=selection, budget=budget)
    artifact = build_artifact(
        selection=selection,
        registry_total_before=_registry_total(registry),
        live_diagnostics=live_diagnostics,
        rank_measurement=rank_measurement,
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
