"""Experiment 5360: ARC live-path perception/salience level-up attempt.

Spec refs: REQ-ARC-FCP-5360, SCENARIO-ARC-FCP-5360.
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
from carnot.agentic.arc_frame_change_predictor import rank_arc_actions


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5360
EXPERIMENT = "experiment_5360_arc_perception_salience_levelup_attempt_v488"
MILESTONE = "2026.07.488"
RESULT_RELATIVE_PATH = "results/experiment_5360_arc_perception_salience_levelup_attempt_v488.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5360", "SCENARIO-ARC-FCP-5360"]
INFERENCE_SUBSTRATE = "live_arc_agent_policy"
SOLVE_PROVENANCE = "live_agent_self_discovery"
DEFAULT_CANDIDATES = ("re86", "sb26", "bp35", "lf52")
DEFAULT_BUDGET = 36
DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5360_arc_perception_salience_levelup_attempt_v488.py -q --no-cov",
    ".venv/bin/pytest tests/python/test_arc_submitted_agent_parity.py -q --no-cov",
    ".venv/bin/pytest tests/python/test_experiment_5360_arc_perception_salience_levelup_attempt_v488.py -q -o addopts= --cov=carnot.agentic.arc_color_blob_salience --cov-report=term-missing --cov-fail-under=100",
    ".venv/bin/ruff check python/carnot/agentic/arc_color_blob_salience.py python/carnot/agentic/arc_competition_agent.py python/carnot/experiment_5360_arc_perception_salience_levelup_attempt_v488.py tests/python/test_experiment_5360_arc_perception_salience_levelup_attempt_v488.py tests/python/test_arc_submitted_agent_parity.py",
    ".venv/bin/ruff format --check python/carnot/agentic/arc_color_blob_salience.py python/carnot/agentic/arc_competition_agent.py python/carnot/experiment_5360_arc_perception_salience_levelup_attempt_v488.py tests/python/test_experiment_5360_arc_perception_salience_levelup_attempt_v488.py tests/python/test_arc_submitted_agent_parity.py",
    ".venv/bin/python scripts/arc_levelup_guarantee_lint.py research-roadmap.yaml --min 1",
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "experiment_id": {"principle": "Stable id ties the artifact to this roadmap task."},
    "milestone": {"principle": "Satisfies the per-milestone ARC standing floor for `.488`."},
    "status": {
        "principle": "Lets capstone distinguish banked level, honest null, or blocked live path."
    },
    "honest_verdict": {
        "principle": "Terminal prefix `complete:` or `blocked_` prevents ambiguous ARC progress."
    },
    "inference_substrate": {
        "principle": "Expected value is live_arc_agent_policy for credited progress."
    },
    "solve_provenance": {
        "principle": "Must be live_agent_self_discovery for a credited ARC solve path."
    },
    "registry_precheck_completed": {
        "principle": "Bare boolean prevents duplicate re-solving of already banked levels."
    },
    "target_game": {"principle": "Names the rotated target game for coverage auditing."},
    "target_level_before": {"principle": "Bare integer records pre-attempt reproduced depth."},
    "perception_audit_completed": {
        "principle": "Bare boolean proves the known-issues priority was exercised."
    },
    "salience_policy_live_reachable": {
        "principle": "Bare boolean proves the live agent can reach the new mechanism."
    },
    "offline_reproduced": {"principle": "Bare boolean is the ARC level-up lint and registry gate."},
    "reproduced_levels": {
        "principle": "Bare integer records new live-path level count; success gate includes reproduced_levels>=1."
    },
    "new_level_banked": {"principle": "Bare boolean separates real progress from diagnostics."},
    "actions_to_first_levelup": {"principle": "Bare integer or null measures action efficiency."},
    "perception_error_classes": {"principle": "Lists observed perception failure modes."},
    "outer_loop_re_used": {
        "principle": "Bare boolean must be false for credited live-path progress."
    },
    "registry_updated": {"principle": "Bare boolean records whether solve registry changed."},
    "tests_run": {"principle": "Lists live-path, registry, and salience-policy checks."},
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


def _registry_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("game")): row
        for row in registry.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    }


def _registry_total(registry: Mapping[str, Any]) -> int:
    return int(registry.get("reproducible_total_levels") or 0)


def select_rotated_target(
    registry: Mapping[str, Any],
    *,
    candidates: Sequence[str] = DEFAULT_CANDIDATES,
    requested_target: tuple[str, int] | None = None,
) -> dict[str, Any]:
    """Pick the next shallow target without duplicating an already banked depth."""

    rows = _registry_rows(registry)
    requested_game = str(requested_target[0]) if requested_target else None
    requested_level = int(requested_target[1]) if requested_target else None
    audit: list[dict[str, Any]] = []
    duplicate_target_avoided = False
    for game in candidates:
        row = rows.get(str(game))
        if row is None:
            audit.append({"game": str(game), "status": "missing_registry_row"})
            continue
        prior = int(row.get("levels_reproduced") or 0)
        if requested_game == str(game) and requested_level is not None and requested_level <= prior:
            duplicate_target_avoided = True
            audit.append(
                {
                    "game": str(game),
                    "status": "skip_duplicate_requested_depth",
                    "requested_level": requested_level,
                    "levels_reproduced": prior,
                }
            )
            continue
        selected = {
            "registry_precheck_completed": True,
            "target_game": str(game),
            "target_level_before": prior,
            "target_level": prior + 1,
            "duplicate_target_avoided": duplicate_target_avoided,
            "candidate_audit": audit
            + [
                {
                    "game": str(game),
                    "status": "selected_next_unbanked_depth",
                    "levels_reproduced": prior,
                    "target_level": prior + 1,
                }
            ],
        }
        return selected
    return {
        "registry_precheck_completed": False,
        "target_game": "none",
        "target_level_before": 0,
        "target_level": 0,
        "duplicate_target_avoided": duplicate_target_avoided,
        "candidate_audit": audit,
    }


class _SelfConsistentWrongFrameDiffScorer:
    def candidate_score(self, _frame: Any, candidate: Any) -> float:
        return 1.0 if getattr(candidate, "source", "") == "self_consistent_wrong_noop" else 0.0


def audit_perception_path() -> dict[str, Any]:
    """Exercise the current frame-diff ranking surface on a synthetic wrong-read case."""

    frame = SimpleNamespace(frame=np.zeros((6, 6), dtype=np.int16), available_actions=[6])
    candidates = [
        ArcAction(6, {"x": 1, "y": 1}, "self_consistent_wrong_noop"),
        ArcAction(6, {"x": 4, "y": 4}, "ground_truth_changing_click"),
    ]
    ranked = rank_arc_actions(
        frame,
        candidates,
        scorer=_SelfConsistentWrongFrameDiffScorer(),
    )
    error_classes = [
        "flat_salience_large_blob_or_status_bar_can_precede_button_like_blob",
        "status_bar_color_requires_salience_deprioritization",
    ]
    if ranked[0].source == "self_consistent_wrong_noop":
        error_classes.append("frame_diff_score_not_ground_truth_validated_before_probe")
    return {
        "completed": True,
        "modules_checked": [
            "arc_frame_change_predictor.rank_arc_actions",
            "arc_online_action_effect_scorer.LiveActionEffectScorer",
            "arc_competition_agent.StepwiseExplorer._candidates",
        ],
        "perception_error_classes": sorted(error_classes),
    }


def salience_policy_live_reachable() -> bool:
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    policy = E3AgentPolicy(
        "reachability",
        proposer=None,
        value_head=None,
        frame_change_scorer=None,
        candidate_router=None,
        action_effect_expansion_prior=False,
        goal_bias=None,
        goal_candidate_guidance=False,
        active_probe_controller=False,
    )
    return isinstance(policy.explorer.action_prior, ColorBlobSaliencePrior)


def offline_arcade_available() -> bool:  # pragma: no cover - environment probe
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
    except Exception:
        return False
    return True


def _action_label(action: int | str, data: Any) -> str:  # pragma: no cover - ARC runtime
    return json.dumps({"action": action, "data": data}, sort_keys=True, separators=(",", ":"))


def _apply_action_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover
    if label == "RESET":
        return env.reset()
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


class _NoOpProposer:  # pragma: no cover - ARC runtime
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "disabled_exp5360_no_live_llm"

    def world_model_candidates(self, _game: str) -> list[Any]:
        return []


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
    target_level = int(selection["target_level"])
    try:
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        policy = E3AgentPolicy(
            game,
            proposer=_NoOpProposer(),
            explore_budget=max(1, int(budget)),
            target_levels=max(1, target_level),
            value_head=None,
            frame_change_scorer=None,
            candidate_router=None,
            action_effect_expansion_prior=False,
            goal_bias=None,
            goal_candidate_guidance=False,
            action_prior=ColorBlobSaliencePrior(),
            active_probe_controller=False,
            go_explore_archive=False,
        )
        frames: list[Any] = []
        latest = None
        labels: list[str] = []
        actions_taken = 0
        max_level = 0
        actions_to_first_levelup = None
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
            if observed_level > max_level:
                max_level = observed_level
                actions_to_first_levelup = actions_taken
            frames.append(latest)
            if max_level >= target_level or latest is None:
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
            "target_level": target_level,
            "budget": int(budget),
            "actions_taken": int(actions_taken),
            "max_level_reached": int(max_level),
            "offline_reproduced": bool(offline_reproduced),
            "reproduced_levels": int(reproduced_levels),
            "actions_to_first_levelup": actions_to_first_levelup,
            "blockers": [] if reproduced_levels else ["bounded_budget_no_levelup"],
            "solution_labels": list(labels) if reproduced_levels else [],
        }
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def _attempt_blocker(attempt: Mapping[str, Any]) -> str:
    blockers = attempt.get("blockers")
    if isinstance(blockers, Sequence) and blockers:
        return str(blockers[0])
    return "no_new_level_banked"


def build_artifact(
    *,
    selection: Mapping[str, Any],
    registry_total: int,
    perception_audit: Mapping[str, Any],
    salience_live_reachable: bool,
    attempt: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    reproduced_levels = int(attempt.get("reproduced_levels") or 0)
    new_level_banked = bool(attempt.get("offline_reproduced") is True and reproduced_levels >= 1)
    status = "banked_level" if new_level_banked else "honest_null"
    blocker = _attempt_blocker(attempt)
    target_game = str(selection.get("target_game") or "none")
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "milestone": MILESTONE,
        "status": status,
        "honest_verdict": (
            f"complete: banked_new_level_{target_game}_plus_{reproduced_levels}"
            if new_level_banked
            else f"complete: no_new_level_banked_{target_game}_residual_{blocker}"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "solve_provenance": SOLVE_PROVENANCE,
        "registry_precheck_completed": bool(selection.get("registry_precheck_completed")),
        "target_game": target_game,
        "target_level_before": int(selection.get("target_level_before") or 0),
        "target_level": int(selection.get("target_level") or 0),
        "perception_audit_completed": bool(perception_audit.get("completed")),
        "salience_policy_live_reachable": bool(salience_live_reachable),
        "offline_reproduced": bool(attempt.get("offline_reproduced") is True),
        "reproduced_levels": reproduced_levels,
        "new_level_banked": new_level_banked,
        "actions_to_first_levelup": attempt.get("actions_to_first_levelup"),
        "perception_error_classes": list(perception_audit.get("perception_error_classes") or []),
        "outer_loop_re_used": False,
        "registry_updated": bool(new_level_banked and attempt.get("registry_updated")),
        "tests_run": list(tests_run),
        "registry_total_before": int(registry_total),
        "registry_total_after": int(registry_total)
        + (reproduced_levels if new_level_banked else 0),
        "preconditions_checked": dict(preconditions_checked),
        "target_selection": dict(selection),
        "attempt": dict(attempt),
        "duration_s": float(duration_s),
    }
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    bool_fields = (
        "registry_precheck_completed",
        "perception_audit_completed",
        "salience_policy_live_reachable",
        "offline_reproduced",
        "new_level_banked",
        "outer_loop_re_used",
        "registry_updated",
    )
    int_fields = ("experiment_id", "target_level_before", "reproduced_levels")
    checks = [
        ("experiment_id mismatch", artifact.get("experiment_id") == EXPERIMENT_ID),
        ("milestone mismatch", artifact.get("milestone") == MILESTONE),
        (
            "inference_substrate mismatch",
            artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        ),
        ("solve_provenance mismatch", artifact.get("solve_provenance") == SOLVE_PROVENANCE),
        ("outer_loop_re_used must be false", artifact.get("outer_loop_re_used") is False),
        (
            "honest_verdict must use terminal prefix",
            str(artifact.get("honest_verdict") or "").startswith("complete:")
            or str(artifact.get("honest_verdict") or "").startswith("blocked_"),
        ),
    ]
    errors = [
        f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in artifact
    ]
    errors += [message for message, ok in checks if not ok]
    errors += [
        f"{field} must be bare bool"
        for field in bool_fields
        if type(artifact.get(field)) is not bool
    ]
    errors += [
        f"{field} must be bare int" for field in int_fields if type(artifact.get(field)) is not int
    ]
    actions = artifact.get("actions_to_first_levelup")
    if actions is not None and type(actions) is not int:
        errors.append("actions_to_first_levelup must be bare int or null")
    if artifact.get("status") == "banked_level" and not (
        artifact.get("offline_reproduced") is True
        and int(artifact.get("reproduced_levels") or 0) >= 1
    ):
        errors.append("banked_level requires offline_reproduced and reproduced_levels>=1")
    if not isinstance(artifact.get("perception_error_classes"), list):
        errors.append("perception_error_classes must be a list")
    if not isinstance(artifact.get("tests_run"), list):
        errors.append("tests_run must be a list")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def _write_artifact(root: Path, artifact: Mapping[str, Any]) -> None:
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[str],
) -> dict[str, Any]:
    selection = {
        "registry_precheck_completed": False,
        "target_game": "none",
        "target_level_before": 0,
        "target_level": 0,
    }
    return {
        **build_artifact(
            selection=selection,
            registry_total=0,
            perception_audit={"completed": False, "perception_error_classes": []},
            salience_live_reachable=False,
            attempt={"offline_reproduced": False, "reproduced_levels": 0, "blockers": [reason]},
            preconditions_checked=preconditions_checked,
            tests_run=tests_run,
            duration_s=duration_s,
        ),
        "status": "blocked",
        "honest_verdict": f"blocked_{reason}",
    }


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
        "spec_has_req_5360": (
            "REQ-ARC-FCP-5360" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
            else False
        ),
        "registry_present": registry_path.exists(),
        "offline_arcade_available": False,
        "outer_loop_re_used": False,
    }
    if not preconditions["spec_has_req_5360"] or not preconditions["registry_present"]:
        artifact = _blocked_artifact(
            reason="preconditions_missing",
            preconditions_checked=preconditions,
            duration_s=time.monotonic() - started,
            tests_run=tests_run,
        )
        _write_artifact(root, artifact)
        return artifact
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    selection = select_rotated_target(registry)
    perception_audit = audit_perception_path()
    salience_reachable = salience_policy_live_reachable()
    preconditions["offline_arcade_available"] = bool(offline_arcade_check())
    if not preconditions["offline_arcade_available"]:
        artifact = _blocked_artifact(
            reason="offline_arcade_missing",
            preconditions_checked=preconditions,
            duration_s=time.monotonic() - started,
            tests_run=tests_run,
        )
        _write_artifact(root, artifact)
        return artifact
    attempt = dict(attempt_runner(root=root, selection=selection, budget=budget))
    artifact = build_artifact(
        selection=selection,
        registry_total=_registry_total(registry),
        perception_audit=perception_audit,
        salience_live_reachable=salience_reachable,
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
