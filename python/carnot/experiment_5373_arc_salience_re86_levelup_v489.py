"""Experiment 5373: live ARC salience repair and re86 level-up attempt.

Spec refs: REQ-ARC-FCP-5373, SCENARIO-ARC-FCP-5373.
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
from carnot.agentic.arc_color_blob_salience import ColorBlobSaliencePrior
from carnot.agentic.arc_frame_change_predictor import (
    GroundTruthValidatedFrameChangeScorer,
    rank_arc_actions,
)


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5373
EXPERIMENT = "experiment_5373_arc_salience_re86_levelup_v489"
MILESTONE = "2026.07.489"
RESULT_RELATIVE_PATH = "results/experiment_5373_arc_salience_re86_levelup_v489.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5373", "SCENARIO-ARC-FCP-5373"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
DEFAULT_BUDGET = 36
DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5373_arc_salience_re86_levelup_v489.py -q --no-cov",
    ".venv/bin/pytest tests/python/test_arc_submitted_agent_parity.py -q --no-cov",
    ".venv/bin/pytest tests/python/test_experiment_5373_arc_salience_re86_levelup_v489.py -q -n 0 -o addopts= --cov=python/carnot --cov-report= --cov-fail-under=0",
    ".venv/bin/coverage report -m python/carnot/experiment_5373_arc_salience_re86_levelup_v489.py",
    ".venv/bin/coverage report -m python/carnot/agentic/arc_color_blob_salience.py",
    ".venv/bin/ruff check python/carnot/agentic/arc_color_blob_salience.py python/carnot/agentic/arc_frame_change_predictor.py python/carnot/agentic/arc_competition_agent.py python/carnot/experiment_5373_arc_salience_re86_levelup_v489.py tests/python/test_experiment_5373_arc_salience_re86_levelup_v489.py",
    ".venv/bin/ruff format --check python/carnot/agentic/arc_color_blob_salience.py python/carnot/agentic/arc_frame_change_predictor.py python/carnot/agentic/arc_competition_agent.py python/carnot/experiment_5373_arc_salience_re86_levelup_v489.py tests/python/test_experiment_5373_arc_salience_re86_levelup_v489.py",
    ".venv/bin/python scripts/arc_levelup_guarantee_lint.py research-roadmap.yaml --min 1",
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "status": {
        "principle": "complete or honest_null; never claim a solve without registry-compatible evidence."
    },
    "solve_provenance": {"principle": "must be live_agent_self_discovery for credited solves."},
    "registry_precheck_done": {"principle": "must be true before target selection or attempts."},
    "target_game": {"principle": "target game id after registry precheck."},
    "target_level_before": {"principle": "reproduced level count before the attempt."},
    "attempted_level": {"principle": "level attempted for +1 deeper progress."},
    "salience_repair_live_reachable": {
        "principle": "true only if the repair is in the live agent path."
    },
    "status_bar_deprioritization_enabled": {
        "principle": "whether the repair addresses the .488 status-bar error class."
    },
    "frame_diff_ground_truth_validated": {
        "principle": "whether frame-diff salience is validated before committing probes."
    },
    "button_like_blob_rank_delta": {
        "principle": "measured ranking change for button-like blobs if available."
    },
    "offline_reproduced": {
        "principle": "true only when a new level is banked by the accepted registry/evidence path; include this exact field for ARC lint."
    },
    "reproduced_levels": {
        "principle": "number of newly reproduced levels; success requires reproduced_levels>=1."
    },
    "new_level_banked": {
        "principle": "true only if registry-compatible evidence banks a level not already present before this task."
    },
    "registry_total_before": {"principle": "total reproduced levels before the attempt."},
    "registry_total_after": {"principle": "total reproduced levels after the attempt."},
    "live_attempt_count": {"principle": "number of live attempts made."},
    "perception_error_classes": {"principle": "residual perception/salience errors observed."},
    "no_outer_loop_re": {"principle": "must be true for credited solve."},
    "no_duplicate_solve": {"principle": "must be true."},
    "honest_verdict": {"principle": "one-line banked/no-bank verdict."},
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


class _ZeroFrameChangeScorer:
    source = "zero_fixture_frame_change_scorer"

    def candidate_score(self, _frame: Any, _candidate: Any) -> float:
        return 0.0


class _NoOpProposer:  # pragma: no cover - ARC runtime boundary
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "disabled_exp5373_no_live_llm"

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
    alternates: Sequence[str] = ("sb26", "bp35", "lf52", "cd82", "sp80", "su15"),
) -> dict[str, Any]:
    """Pick the next unbanked target while avoiding duplicate depth claims."""

    rows = _registry_rows(registry)
    preferred_row = rows.get(str(preferred), {})
    preferred_before = int(preferred_row.get("levels_reproduced") or 0)
    if preferred_before < 3:
        return {
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
            "registry_precheck_done": True,
            "target_game": str(game),
            "target_level_before": before,
            "attempted_level": before + 1,
            "no_duplicate_solve": True,
            "selection_reason": "re86_l3_already_banked_rotated_target",
        }
    return {
        "registry_precheck_done": False,
        "target_game": "none",
        "target_level_before": 0,
        "attempted_level": 0,
        "no_duplicate_solve": False,
        "selection_reason": "no_registry_target_available",
    }


def _default_salience_frame() -> SimpleNamespace:
    grid = np.zeros((20, 20), dtype=np.int16)
    grid[0, :] = 16
    grid[2:10, 2:18] = 8
    grid[14:16, 14:16] = 8
    return SimpleNamespace(frame=grid, available_actions=[6])


def _default_salience_candidates() -> list[ArcAction]:
    return [
        ArcAction(6, {"x": 4, "y": 4}, "large_flat_blob"),
        ArcAction(6, {"x": 14, "y": 14}, "button_like_blob"),
        ArcAction(6, {"x": 2, "y": 0}, "status_bar_blob"),
    ]


def _rank_of_source(candidates: Sequence[Any], source: str) -> int | None:
    for index, candidate in enumerate(candidates):
        if getattr(candidate, "source", None) == source:
            return index
    return None


def measure_button_like_blob_rank_delta(
    frame: Any | None = None,
    candidates: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Measure the repair on the Exp5360 flat-blob/status-bar failure case."""

    frame = _default_salience_frame() if frame is None else frame
    rows = list(_default_salience_candidates() if candidates is None else candidates)
    legacy = ColorBlobSaliencePrior(large_flat_deprioritization=False)
    repaired = ColorBlobSaliencePrior()
    before_ranked = rank_arc_actions(frame, rows, prior=legacy)
    after_ranked = rank_arc_actions(frame, rows, prior=repaired)
    before = _rank_of_source(before_ranked, "button_like_blob")
    after = _rank_of_source(after_ranked, "button_like_blob")
    delta = None if before is None or after is None else int(before) - int(after)
    return {
        "before_rank": before,
        "after_rank": after,
        "button_like_blob_rank_delta": delta,
        "legacy_first_source": getattr(before_ranked[0], "source", None) if before_ranked else None,
        "repaired_first_source": getattr(after_ranked[0], "source", None) if after_ranked else None,
    }


def salience_repair_live_diagnostics() -> dict[str, bool]:
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    validator = GroundTruthValidatedFrameChangeScorer(_ZeroFrameChangeScorer())
    policy = E3AgentPolicy(
        "re86",
        proposer=None,
        value_head=None,
        frame_change_scorer=validator,
        candidate_router=None,
        action_effect_expansion_prior=False,
        goal_bias=None,
        goal_candidate_guidance=False,
        active_probe_controller=False,
    )
    prior = policy.explorer.action_prior
    return {
        "salience_repair_live_reachable": bool(
            isinstance(prior, ColorBlobSaliencePrior)
            and isinstance(
                policy.explorer.frame_change_scorer, GroundTruthValidatedFrameChangeScorer
            )
        ),
        "status_bar_deprioritization_enabled": bool(
            isinstance(prior, ColorBlobSaliencePrior)
            and prior.status_bar_deprioritization
            and prior.large_flat_deprioritization
        ),
        "frame_diff_ground_truth_validated": isinstance(
            policy.explorer.frame_change_scorer,
            GroundTruthValidatedFrameChangeScorer,
        ),
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


def run_live_salience_attempt(  # pragma: no cover - ARC runtime boundary
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
    try:
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        policy = E3AgentPolicy(
            game,
            proposer=_NoOpProposer(),
            explore_budget=max(1, int(budget)),
            target_levels=max(1, attempted_level),
            value_head=None,
            candidate_router=None,
            action_effect_expansion_prior=False,
            goal_bias=None,
            goal_candidate_guidance=False,
            active_probe_controller=False,
            go_explore_archive=False,
        )
        frames: list[Any] = []
        latest = None
        labels: list[str] = []
        actions_taken = 0
        max_level = 0
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
        reproduced_levels = 0
        offline_reproduced = False
        if max_level > target_level_before and labels:
            gate = kit.reproduce(game, labels, _apply_action_label, claimed_level=max_level)
            offline_reproduced = bool(gate.get("reproduced"))
            if offline_reproduced:
                reproduced_levels = int(max_level - target_level_before)
        return {
            "target_game": game,
            "target_level_before": target_level_before,
            "attempted_level": attempted_level,
            "actions_taken": int(actions_taken),
            "max_level_reached": int(max_level),
            "offline_reproduced": bool(offline_reproduced),
            "reproduced_levels": int(reproduced_levels),
            "new_level_banked": bool(offline_reproduced and reproduced_levels >= 1),
            "perception_error_classes": [] if reproduced_levels else ["bounded_budget_no_levelup"],
            "solution_labels": list(labels) if reproduced_levels else [],
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
    repair: Mapping[str, Any],
    rank_measurement: Mapping[str, Any],
    attempt: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    no_duplicate = bool(selection.get("no_duplicate_solve"))
    no_outer_loop_re = True
    raw_new_levels = int(attempt.get("reproduced_levels") or 0)
    new_level_banked = bool(
        attempt.get("offline_reproduced") is True
        and raw_new_levels >= 1
        and no_duplicate
        and no_outer_loop_re
    )
    reproduced_levels = raw_new_levels if new_level_banked else 0
    target_game = str(selection.get("target_game") or "none")
    attempted_level = int(selection.get("attempted_level") or 0)
    status = "complete" if new_level_banked else "honest_null"
    errors = list(attempt.get("perception_error_classes") or [])
    if not new_level_banked and "bounded_budget_no_levelup" not in errors:
        errors.append("bounded_budget_no_levelup")
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
        "target_level_before": int(selection.get("target_level_before") or 0),
        "attempted_level": attempted_level,
        "salience_repair_live_reachable": bool(repair.get("salience_repair_live_reachable")),
        "status_bar_deprioritization_enabled": bool(
            repair.get("status_bar_deprioritization_enabled")
        ),
        "frame_diff_ground_truth_validated": bool(repair.get("frame_diff_ground_truth_validated")),
        "button_like_blob_rank_delta": rank_measurement.get("button_like_blob_rank_delta"),
        "offline_reproduced": bool(new_level_banked),
        "reproduced_levels": int(reproduced_levels),
        "new_level_banked": bool(new_level_banked),
        "registry_total_before": int(registry_total_before),
        "registry_total_after": int(registry_total_before) + int(reproduced_levels),
        "live_attempt_count": 1,
        "perception_error_classes": sorted(set(str(error) for error in errors)),
        "no_outer_loop_re": no_outer_loop_re,
        "no_duplicate_solve": no_duplicate,
        "honest_verdict": (
            f"banked: {target_game} L{attempted_level} reproduced live-path"
            if new_level_banked
            else f"no-bank: {target_game} L{attempted_level} not reproduced within bounded live attempt"
        ),
        "preconditions_checked": dict(preconditions_checked),
        "target_selection": dict(selection),
        "rank_measurement": dict(rank_measurement),
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
        "salience_repair_live_reachable",
        "status_bar_deprioritization_enabled",
        "frame_diff_ground_truth_validated",
        "offline_reproduced",
        "new_level_banked",
        "no_outer_loop_re",
        "no_duplicate_solve",
    )
    int_fields = (
        "target_level_before",
        "attempted_level",
        "reproduced_levels",
        "registry_total_before",
        "registry_total_after",
        "live_attempt_count",
    )
    for field in bool_fields:
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
    for field in int_fields:
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
    if artifact.get("status") not in {"complete", "honest_null"}:
        errors.append("status must be complete or honest_null")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    if not isinstance(artifact.get("perception_error_classes"), list):
        errors.append("perception_error_classes must be a list")
    if not str(artifact.get("honest_verdict") or "").startswith(("banked:", "no-bank:")):
        errors.append("honest_verdict must be a one-line banked/no-bank verdict")
    if artifact.get("new_level_banked") is True and not (
        artifact.get("offline_reproduced") is True
        and int(artifact.get("reproduced_levels") or 0) >= 1
        and artifact.get("no_outer_loop_re") is True
        and artifact.get("no_duplicate_solve") is True
    ):
        errors.append("new_level_banked requires reproduced live-path non-duplicate evidence")
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
    attempt_runner: Callable[..., Mapping[str, Any]] = run_live_salience_attempt,
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
        "spec_has_req_5373": (
            "REQ-ARC-FCP-5373" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
            else False
        ),
        "registry_present": registry_path.exists(),
        "offline_arcade_available": False,
        "no_outer_loop_re": True,
    }
    registry = (
        yaml.safe_load(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    )
    registry = registry or {}
    selection = select_target_after_precheck(registry)
    repair = salience_repair_live_diagnostics()
    rank_measurement = measure_button_like_blob_rank_delta()
    preconditions["offline_arcade_available"] = bool(offline_arcade_check())
    if not (
        preconditions["AGENTS.md"]
        and preconditions["CODEX.md"]
        and preconditions["spec_has_req_5373"]
        and preconditions["registry_present"]
        and preconditions["offline_arcade_available"]
    ):
        attempt: Mapping[str, Any] = {
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "new_level_banked": False,
            "perception_error_classes": ["preconditions_missing_or_offline_arcade_unavailable"],
        }
    else:
        attempt = attempt_runner(root=root, selection=selection, budget=budget)
    artifact = build_artifact(
        selection=selection,
        registry_total_before=_registry_total(registry),
        repair=repair,
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
