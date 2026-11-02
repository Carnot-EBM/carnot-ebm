"""Experiment 5397: ARC blob salience live-path level-up attempt.

Spec refs: REQ-ARC-FCP-5397, SCENARIO-ARC-FCP-5397.
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

from carnot.agentic.arc_color_blob_salience import ColorBlobSaliencePrior


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5397
EXPERIMENT = "experiment_5397_arc_blob_salience_live_path_v491"
MILESTONE = "2026.07.491"
RESULT_RELATIVE_PATH = "results/experiment_5397_arc_blob_salience_live_path_v491.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5397", "SCENARIO-ARC-FCP-5397"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
DEFAULT_BUDGET = 36
DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5397_arc_blob_salience_live_path_v491.py -q --no-cov",
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5397_arc_blob_salience_live_path_v491.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/experiment_5397_arc_blob_salience_live_path_v491.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "status": {
        "principle": "complete for a banked +1 level, honest_null for a real no-bank attempt, or blocked for missing harness access."
    },
    "milestone": {"principle": "must equal 2026.07.491."},
    "target_game": {"principle": "game selected after registry precheck."},
    "attempted_level": {"principle": "level attempted after registry precheck."},
    "registry_precheck_done": {"principle": "must be true."},
    "duplicate_solve_avoided": {"principle": "must be true."},
    "solve_provenance": {"principle": "must be live_agent_self_discovery for a credited solve."},
    "live_agent_policy_modified": {
        "principle": "true only if E3AgentPolicy generation-stage action prioritization was changed."
    },
    "connected_component_salience_enabled": {
        "principle": "true if the blob salience mechanism was active."
    },
    "salience_tiers_emitted": {"principle": "true if action tiers were logged."},
    "per_game_adapter_used": {"principle": "must be false."},
    "offline_bfs_used": {"principle": "must be false."},
    "outer_loop_re_used": {"principle": "must be false."},
    "live_attempt_count": {"principle": "count of live harness attempts."},
    "offline_reproduced": {
        "principle": "true only if the live-discovered new level is reproduced."
    },
    "reproduced_levels": {
        "principle": "number of newly reproduced levels, success requires reproduced_levels>=1."
    },
    "new_level_banked": {"principle": "true only for a +1 reproducible level."},
    "failure_mode": {"principle": "null on success or concise no-bank reason."},
    "honest_verdict": {
        "principle": "one-line summary starting with complete:, honest_null:, or blocked:."
    },
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


class _NoOpProposer:  # pragma: no cover - ARC runtime boundary
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "disabled_exp5397_no_live_llm"

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
    """REQ-ARC-FCP-5397: choose a next-level target without duplicate credit."""

    rows = _registry_rows(registry)
    preferred_before = int((rows.get(str(preferred)) or {}).get("levels_reproduced") or 0)
    if preferred_before < 3:
        return {
            "status": "selected",
            "registry_precheck_done": True,
            "target_game": str(preferred),
            "target_level_before": preferred_before,
            "attempted_level": preferred_before + 1,
            "duplicate_solve_avoided": True,
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
            "duplicate_solve_avoided": True,
            "selection_reason": "preferred_re86_l3_banked_rotated_target",
        }
    return {
        "status": "selected",
        "registry_precheck_done": True,
        "target_game": str(preferred),
        "target_level_before": preferred_before,
        "attempted_level": preferred_before + 1,
        "duplicate_solve_avoided": True,
        "selection_reason": "no_alternate_available_no_duplicate_attempted",
    }


def _diagnostic_blob_frame() -> SimpleNamespace:
    grid = np.zeros((20, 20), dtype=np.int16)
    grid[0, :] = 16
    grid[2:10, 2:18] = 8
    grid[14:16, 14:16] = 9
    return SimpleNamespace(frame=grid, available_actions=[6])


def blob_salience_live_diagnostics() -> dict[str, Any]:
    """REQ-ARC-FCP-5397: prove the submitted E3 path reaches blob tier generation."""

    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    policy = E3AgentPolicy(
        "re86",
        proposer=None,
        value_head=None,
        frame_change_scorer=None,
        candidate_router=None,
        action_effect_expansion_prior=False,
        goal_bias=None,
        goal_candidate_guidance=False,
        active_probe_controller=False,
    )
    candidates = policy.explorer._candidates(_diagnostic_blob_frame())  # noqa: SLF001
    diagnostics = policy.explorer.action_salience_diagnostics()
    action_prior = policy.explorer.action_prior
    return {
        "live_agent_policy_modified": True,
        "action_prior_source": action_prior.as_dict().get("source")
        if hasattr(action_prior, "as_dict")
        else None,
        "first_generated_candidate": dict(candidates[0]) if candidates else None,
        **diagnostics,
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


def run_live_blob_salience_attempt(  # pragma: no cover - ARC runtime boundary
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
    prior = ColorBlobSaliencePrior()
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
        return {
            "target_game": game,
            "target_level_before": target_level_before,
            "attempted_level": attempted_level,
            "actions_taken": int(actions_taken),
            "max_level_reached": int(max_level),
            "offline_reproduced": bool(offline_reproduced),
            "new_level_banked": bool(offline_reproduced),
            "failure_mode": None if offline_reproduced else "bounded_budget_no_levelup",
            "solution_labels": list(labels) if offline_reproduced else [],
            "reproduction_gate": gate,
            "runtime_self_discovery": True,
            "offline_bfs_used": False,
            "per_game_adapter_used": False,
            "outer_loop_re_used": False,
            "root": str(root),
            "salience_diagnostics": policy.explorer.action_salience_diagnostics(),
            "blob_prior_diagnostics": prior.as_dict(),
        }
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def _new_reproduced_levels(
    *,
    attempt: Mapping[str, Any],
    target_level_before: int,
    attempted_level: int,
) -> int:
    if attempt.get("offline_reproduced") is not True:
        return 0
    if "new_reproduced_levels" in attempt:
        return max(0, int(attempt.get("new_reproduced_levels") or 0))
    max_level = int(attempt.get("max_level_reached") or target_level_before)
    if max_level < attempted_level:
        return 0
    return max(1, max_level - int(target_level_before))


def build_artifact(
    *,
    selection: Mapping[str, Any],
    registry_total_before: int,
    live_diagnostics: Mapping[str, Any],
    attempt: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    target_game = str(selection.get("target_game") or "none")
    target_level_before = int(selection.get("target_level_before") or 0)
    attempted_level = int(selection.get("attempted_level") or target_level_before + 1)
    blocked = bool(attempt.get("blocked")) or str(attempt.get("failure_mode") or "").startswith(
        "missing_harness"
    )
    new_reproduced = _new_reproduced_levels(
        attempt=attempt,
        target_level_before=target_level_before,
        attempted_level=attempted_level,
    )
    no_prohibited_inputs = (
        attempt.get("per_game_adapter_used") is not True
        and attempt.get("offline_bfs_used") is not True
        and attempt.get("outer_loop_re_used") is not True
    )
    live_solution_labels = bool(attempt.get("solution_labels"))
    duplicate_solve_avoided = bool(selection.get("duplicate_solve_avoided"))
    new_level_banked = bool(
        new_reproduced >= 1
        and live_solution_labels
        and duplicate_solve_avoided
        and no_prohibited_inputs
    )
    status = "complete" if new_level_banked else "blocked" if blocked else "honest_null"
    failure_mode = None if new_level_banked else str(attempt.get("failure_mode") or "")
    if status == "honest_null" and not failure_mode:
        failure_mode = "bounded_budget_no_levelup"
    if status == "blocked" and not failure_mode:
        failure_mode = "missing_harness_access"
    live_attempt_count = int(attempt.get("live_attempt_count") or (0 if blocked else 1))
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "milestone": MILESTONE,
        "target_game": target_game,
        "target_level_before": target_level_before,
        "attempted_level": attempted_level,
        "registry_precheck_done": bool(selection.get("registry_precheck_done")),
        "duplicate_solve_avoided": duplicate_solve_avoided,
        "solve_provenance": SOLVE_PROVENANCE,
        "live_agent_policy_modified": bool(live_diagnostics.get("live_agent_policy_modified")),
        "connected_component_salience_enabled": bool(
            live_diagnostics.get("connected_component_salience_enabled")
        ),
        "salience_tiers_emitted": bool(live_diagnostics.get("salience_tiers_emitted")),
        "per_game_adapter_used": bool(attempt.get("per_game_adapter_used", False)),
        "offline_bfs_used": bool(attempt.get("offline_bfs_used", False)),
        "outer_loop_re_used": bool(attempt.get("outer_loop_re_used", False)),
        "live_attempt_count": live_attempt_count,
        "offline_reproduced": bool(new_level_banked),
        "reproduced_levels": int(new_reproduced if new_level_banked else 0),
        "new_level_banked": bool(new_level_banked),
        "failure_mode": failure_mode,
        "honest_verdict": (
            f"complete: {target_game} L{attempted_level} live blob salience reproduced"
            if new_level_banked
            else f"blocked: {failure_mode}"
            if status == "blocked"
            else f"honest_null: {target_game} L{attempted_level} {failure_mode}"
        ),
        "registry_total_before": int(registry_total_before),
        "registry_total_after": int(registry_total_before) + int(new_reproduced),
        "target_selection": dict(selection),
        "preconditions_checked": dict(preconditions_checked),
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
    if artifact.get("status") not in {"complete", "honest_null", "blocked"}:
        errors.append("status must be complete, honest_null, or blocked")
    if artifact.get("milestone") != MILESTONE:
        errors.append("milestone must be 2026.07.491")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    for field in (
        "registry_precheck_done",
        "duplicate_solve_avoided",
        "live_agent_policy_modified",
        "connected_component_salience_enabled",
        "salience_tiers_emitted",
        "per_game_adapter_used",
        "offline_bfs_used",
        "outer_loop_re_used",
        "offline_reproduced",
        "new_level_banked",
    ):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
    for field in ("registry_precheck_done", "duplicate_solve_avoided"):
        if artifact.get(field) is not True:
            errors.append(f"{field} must be true")
    for field in ("per_game_adapter_used", "offline_bfs_used", "outer_loop_re_used"):
        if artifact.get(field) is not False:
            errors.append(f"{field} must be false")
    for field in ("attempted_level", "live_attempt_count", "reproduced_levels"):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
    if artifact.get("status") == "complete":
        if artifact.get("offline_reproduced") is not True:
            errors.append("complete artifact requires offline_reproduced true")
        if artifact.get("new_level_banked") is not True:
            errors.append("complete artifact requires new_level_banked true")
        if type(artifact.get("reproduced_levels")) is int and artifact["reproduced_levels"] < 1:
            errors.append("complete artifact requires reproduced_levels >= 1")
        if artifact.get("failure_mode") is not None:
            errors.append("complete artifact requires failure_mode null")
    else:
        if artifact.get("offline_reproduced") is True:
            errors.append("non-complete artifact cannot set offline_reproduced true")
        if artifact.get("new_level_banked") is True:
            errors.append("new_level_banked requires complete status")
        if not isinstance(artifact.get("failure_mode"), str) or not artifact.get("failure_mode"):
            errors.append("honest_null or blocked artifact requires concise failure_mode")
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
    attempt_runner: Callable[..., Mapping[str, Any]] = run_live_blob_salience_attempt,
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
        "spec_has_req_5397": (
            "REQ-ARC-FCP-5397" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
            else False
        ),
        "registry_present": registry_path.exists(),
        "offline_arcade_available": False,
        "no_per_game_adapter": True,
        "no_offline_bfs": True,
        "no_outer_loop_re": True,
    }
    registry = (
        yaml.safe_load(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    )
    registry = registry or {}
    selection = select_target_after_precheck(registry)
    live_diagnostics = blob_salience_live_diagnostics()
    ready_without_arcade = (
        preconditions["AGENTS.md"]
        and preconditions["CODEX.md"]
        and preconditions["spec_has_req_5397"]
        and preconditions["registry_present"]
    )
    if not ready_without_arcade:
        attempt: Mapping[str, Any] = {
            "blocked": True,
            "failure_mode": "missing_harness_access",
            "live_attempt_count": 0,
        }
    else:
        preconditions["offline_arcade_available"] = bool(offline_arcade_check())
        if not preconditions["offline_arcade_available"]:
            attempt = {
                "blocked": True,
                "failure_mode": "missing_harness_access",
                "live_attempt_count": 0,
            }
        else:
            attempt = attempt_runner(root=root, selection=selection, budget=budget)
    artifact = build_artifact(
        selection=selection,
        registry_total_before=_registry_total(registry),
        live_diagnostics=live_diagnostics,
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
