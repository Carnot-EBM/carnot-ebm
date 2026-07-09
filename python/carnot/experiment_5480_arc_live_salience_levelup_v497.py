"""Experiment 5480: rotated ARC live salience level-up attempt.

Spec refs: REQ-ARC-FCP-5480, SCENARIO-ARC-FCP-5480.

This module is the accounting shell around one bounded live-agent attempt. The
important guardrail is that a level only counts when the live attempt produces a
candidate that the official reproduction gate confirms beyond the Exp5479
precheck depth. Salience can guide the search, but source reading, offline BFS,
hand adapters, and outer-loop reverse engineering cannot receive credit here.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5480
EXPERIMENT = "experiment_5480_arc_live_salience_levelup_v497"
MILESTONE = "2026.07.497"
RESULT_RELATIVE_PATH = "results/experiment_5480_arc_live_salience_levelup_v497.json"
FIRST_WIN_TRACE_TEMPLATE = "results/experiment_5480_first_win_trace_{game}_L{level}.json"
EXP5479_RELATIVE_PATH = "results/experiment_5479_arc_target_rotation_precheck_v497.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5480", "SCENARIO-ARC-FCP-5480"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "arc_live_agent_self_discovery"
RANDOM_SEED = 5480
DEFAULT_BUDGET = 48
DEFAULT_TESTS_RUN = [
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_5480_arc_live_salience_levelup_v497.py "
        "-q --no-cov"
    ),
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5480_arc_live_salience_levelup_v497.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/experiment_5480_arc_live_salience_levelup_v497.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "game": {
        "principle": "selected Exp5479 game, or none when the target precondition blocks the attempt."
    },
    "target_level": {
        "principle": "selected Exp5479 target level; success must reproduce at least this level."
    },
    "solve_provenance": {"principle": "must equal live_agent_self_discovery."},
    "hidden_source_reading": {
        "principle": "must be false; hidden/public game source is not part of the credited path."
    },
    "offline_bfs_used": {
        "principle": "must be false; exhaustive offline ground-truth BFS is not the credited path."
    },
    "hand_adapter_used": {
        "principle": "must be false; a hand per-game adapter is not credited."
    },
    "outer_loop_re_used": {
        "principle": "must be false; outer-loop reverse engineering is not credited."
    },
    "action_count": {
        "principle": "bare integer count of bounded live-agent actions actually executed."
    },
    "explored_state_count": {
        "principle": "bare integer count of live-agent states observed or tracked during the attempt."
    },
    "failed_hypotheses": {
        "principle": "list of rejected salience/runtime hypotheses when no target level is banked."
    },
    "offline_reproduced": {
        "principle": "true only when the live-agent candidate reproduces beyond the precheck depth."
    },
    "reproduced_levels": {
        "principle": "new reproduced levels banked beyond the precheck depth; success requires >=1."
    },
    "new_level_banked": {
        "principle": "true only when offline_reproduced=true and reproduced_levels>=1."
    },
    "reproduced_levels_before": {
        "principle": "registry reproduced depth for the selected game before Exp5480."
    },
    "reproduced_levels_after": {
        "principle": "registry reproduced depth after Exp5480; unchanged on honest null."
    },
    "registry_updated": {
        "principle": "true only when ops/arc_solve_registry.yaml was updated for a newly reproduced level."
    },
    "first_win_trace_path": {
        "principle": "relative path to the first reproduced winning trace, or empty string when none exists."
    },
    "inference_substrate": {"principle": "must equal arc_live_agent_self_discovery."},
    "random_seed": {"principle": "deterministic seed for the bounded live attempt."},
    "honest_verdict": {
        "principle": "one-line verdict starting complete:, honest_null:, or blocked:."
    },
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _level_label(level: int) -> str:
    return f"L{max(0, int(level))}"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry(root: Path = REPO) -> dict[str, Any]:
    path = root / REGISTRY_RELATIVE_PATH
    if not path.exists():
        return {"reproducible_total_levels": 0, "games": []}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _registry_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("game")): row
        for row in registry.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    }


def select_exp5479_target(
    exp5479_artifact: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """REQ-ARC-FCP-5480: select Exp5479's rotated target before attempting."""

    total = _as_int(registry.get("reproducible_total_levels"))
    game = str(exp5479_artifact.get("selected_game") or "")
    target_level = _as_int(exp5479_artifact.get("selected_target_level"))
    if not game or game == "none" or target_level <= 0:
        return {
            "blocked": True,
            "blocker": "missing_exp5479_target",
            "game": "none",
            "target_level": 0,
            "reproduced_levels_before": 0,
            "registry_total_before": total,
        }
    if exp5479_artifact.get("arc_target_rotation_ready") is not True:
        return {
            "blocked": True,
            "blocker": "exp5479_target_rotation_not_ready",
            "game": game,
            "target_level": target_level,
            "reproduced_levels_before": 0,
            "registry_total_before": total,
        }
    before = _as_int((_registry_rows(registry).get(game) or {}).get("levels_reproduced"))
    if before >= target_level:
        return {
            "blocked": True,
            "blocker": "target_already_reproduced",
            "game": game,
            "target_level": target_level,
            "reproduced_levels_before": before,
            "registry_total_before": total,
        }
    return {
        "blocked": False,
        "game": game,
        "target_level": target_level,
        "reproduced_levels_before": before,
        "registry_total_before": total,
        "selection_reason": "exp5479_rotated_eligible_target",
    }


def _build_failed_hypotheses(
    exp5479_artifact: Mapping[str, Any],
    attempt: Mapping[str, Any],
    target: Mapping[str, Any],
) -> list[dict[str, Any]]:
    summary = exp5479_artifact.get("salience_feature_summary")
    candidates = []
    if isinstance(summary, Mapping):
        raw_candidates = summary.get("target_region_candidates") or []
        candidates = [row for row in raw_candidates if isinstance(row, Mapping)]
    failed: list[dict[str, Any]] = []
    if not candidates:
        failed.append(
            {
                "hypothesis": "exp5479_salience_candidates_absent",
                "evidence": "no target-region candidates were available from the rotated precheck",
            }
        )
    for row in candidates[:3]:
        failed.append(
            {
                "hypothesis": (
                    f"tier_{_as_int(row.get('tier'))}_color_{_as_int(row.get('color'))}_click_candidate"
                ),
                "action": _as_int(row.get("action")),
                "data": dict(row.get("data") or {}),
                "evidence": (
                    f"bounded live attempt did not reproduce {target.get('game')} "
                    f"{_level_label(_as_int(target.get('target_level')))} from this prior"
                ),
            }
        )
    failed.append(
        {
            "hypothesis": "bounded_salience_runtime_sequence",
            "action_count": _as_int(attempt.get("action_count"), _as_int(attempt.get("actions_taken"))),
            "explored_state_count": _as_int(attempt.get("explored_state_count")),
            "failure_mode": str(attempt.get("failure_mode") or "bounded_budget_no_target_level_reproduction"),
        }
    )
    return failed


def _trace_path_for(game: str, target_level: int) -> str:
    return FIRST_WIN_TRACE_TEMPLATE.format(game=game, level=int(target_level))


def run_rotated_live_salience_attempt(  # pragma: no cover - ARC runtime boundary
    *,
    root: Path,
    target: Mapping[str, Any],
    exp5479_artifact: Mapping[str, Any],
    budget: int = DEFAULT_BUDGET,
) -> dict[str, Any]:
    """Run the bounded live salience path without source, BFS, or adapter credit."""

    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_color_blob_salience import ColorBlobSaliencePrior
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of
    from carnot.experiment_5465_gated_arc_connected_component_salience_levelup_v496 import (
        _NoOpProposer,
        _action_label,
        _apply_action_label,
    )

    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    game = str(target["game"])
    before = _as_int(target.get("reproduced_levels_before"))
    target_level = _as_int(target.get("target_level"))
    labels: list[str] = []
    action_count = 0
    observed_effects: list[dict[str, Any]] = []
    try:
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        prior = ColorBlobSaliencePrior()
        policy = E3AgentPolicy(
            game,
            proposer=_NoOpProposer(),
            explore_budget=max(1, int(budget)),
            target_levels=max(1, target_level),
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
        max_level = before
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
                previous = latest
                latest = env.step(getattr(GameAction, f"ACTION{int(kind)}"), data=data)
                labels.append(_action_label(int(kind), data))
                action_count += 1
                if previous is not None and len(observed_effects) < 12:
                    observed_effects.append(
                        {
                            "action": int(kind),
                            "data": data,
                            "level_after": _as_int(_level_of(latest), default=max_level),
                        }
                    )
            observed_level = _as_int(_level_of(latest), default=max_level)
            max_level = max(max_level, observed_level)
            frames.append(latest)
            if latest is None or max_level >= target_level:
                break

        first_win_trace_path = ""
        gate: dict[str, Any] = {
            "game": game,
            "claimed_level": max_level,
            "reached_level": before,
            "reproduced": False,
            "mode": "official_reproduction_gate_not_run_no_new_target_candidate",
        }
        if max_level > before and labels:
            first_win_trace_path = _trace_path_for(game, target_level)
            trace = {
                "game": game,
                "target_level": target_level,
                "labels": labels,
                "source": "exp5480_live_agent_self_discovery_candidate",
            }
            trace_path = root / first_win_trace_path
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            gate = dict(kit.reproduce(game, labels, _apply_action_label, claimed_level=max_level))
        reached = _as_int(gate.get("reached_level"), default=max_level)
        offline_reproduced = bool(gate.get("reproduced")) and reached > before
        if not offline_reproduced:
            first_win_trace_path = ""
        explored_state_count = int(len(getattr(policy.explorer, "graph", {}) or frames))
        failure_mode = None if offline_reproduced else "bounded_budget_no_target_level_reproduction"
        diagnostic_attempt = {
            "action_count": int(action_count),
            "explored_state_count": explored_state_count,
            "failure_mode": failure_mode,
        }
        return {
            "action_count": int(action_count),
            "explored_state_count": explored_state_count,
            "offline_reproduced": bool(offline_reproduced),
            "reproduced_levels_after": int(reached if offline_reproduced else before),
            "first_win_trace_path": first_win_trace_path,
            "failed_hypotheses": [] if offline_reproduced else _build_failed_hypotheses(exp5479_artifact, diagnostic_attempt, target),
            "failure_mode": failure_mode,
            "reproduction_gate": gate,
            "observed_action_effects": observed_effects,
            "salience_diagnostics": policy.explorer.action_salience_diagnostics(),
            "source_reading_used": False,
            "offline_bfs_used": False,
            "hand_adapter_used": False,
            "outer_loop_re_used": False,
        }
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def offline_arcade_available() -> bool:  # pragma: no cover - environment probe
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
    except Exception:
        return False
    return True


def build_artifact(
    *,
    target: Mapping[str, Any],
    exp5479_artifact: Mapping[str, Any],
    attempt: Mapping[str, Any],
    registry_updated: bool,
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    before = _as_int(target.get("reproduced_levels_before"))
    target_level = _as_int(target.get("target_level"))
    attempted_after = _as_int(
        attempt.get("reproduced_levels_after"),
        _as_int(attempt.get("reproduced_levels"), before),
    )
    blocked = bool(target.get("blocked")) or bool(attempt.get("blocked"))
    no_prohibited_inputs = (
        attempt.get("source_reading_used", False) is False
        and attempt.get("offline_bfs_used", False) is False
        and attempt.get("hand_adapter_used", False) is False
        and attempt.get("outer_loop_re_used", False) is False
    )
    reproduced_delta = max(0, int(attempted_after) - int(before))
    new_level_banked = bool(
        not blocked
        and attempt.get("offline_reproduced") is True
        and no_prohibited_inputs
        and reproduced_delta >= 1
        and attempted_after >= target_level
    )
    after = int(attempted_after) if new_level_banked else int(before)
    status = "complete" if new_level_banked else "blocked" if blocked else "honest_null"
    failure_mode = str(
        attempt.get("failure_mode")
        or target.get("blocker")
        or "bounded_budget_no_target_level_reproduction"
    )
    trace_path = str(attempt.get("first_win_trace_path") or "") if new_level_banked else ""
    failed_hypotheses = (
        []
        if new_level_banked
        else list(attempt.get("failed_hypotheses") or _build_failed_hypotheses(exp5479_artifact, attempt, target))
    )
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "carnot.experiment_5480_arc_live_salience_levelup_v497.v1",
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "game": str(target.get("game") or "none"),
        "target_level": int(target_level),
        "solve_provenance": SOLVE_PROVENANCE,
        "hidden_source_reading": bool(attempt.get("hidden_source_reading", False)),
        "offline_bfs_used": bool(attempt.get("offline_bfs_used", False)),
        "hand_adapter_used": bool(attempt.get("hand_adapter_used", False)),
        "outer_loop_re_used": bool(attempt.get("outer_loop_re_used", False)),
        "action_count": _as_int(attempt.get("action_count"), _as_int(attempt.get("actions_taken"))),
        "explored_state_count": _as_int(
            attempt.get("explored_state_count"),
            _as_int(attempt.get("states_expanded")),
        ),
        "failed_hypotheses": failed_hypotheses,
        "offline_reproduced": bool(new_level_banked),
        "reproduced_levels": int(after - before if new_level_banked else 0),
        "new_level_banked": bool(new_level_banked),
        "reproduced_levels_before": int(before),
        "reproduced_levels_after": int(after),
        "registry_updated": bool(registry_updated),
        "first_win_trace_path": trace_path,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "honest_verdict": (
            f"complete: {target.get('game')} {_level_label(target_level)} live salience reproduced and banked"
            if new_level_banked
            else f"blocked: {failure_mode}"
            if status == "blocked"
            else f"honest_null: {target.get('game')} {_level_label(target_level)} {failure_mode}"
        ),
        "failure_mode": "" if new_level_banked else failure_mode,
        "target_selection": dict(target),
        "exp5479_salience_feature_summary": dict(
            exp5479_artifact.get("salience_feature_summary") or {}
        ),
        "attempt": dict(attempt),
        "preconditions_checked": dict(preconditions_checked),
        "tests_run": list(tests_run),
        "duration_s": float(duration_s),
    }
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in artifact
    ]
    if not isinstance(artifact.get("game"), str) or not artifact.get("game"):
        errors.append("game must be a non-empty string")
    for field in (
        "target_level",
        "action_count",
        "explored_state_count",
        "reproduced_levels",
        "reproduced_levels_before",
        "reproduced_levels_after",
        "random_seed",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
        elif field in ("action_count", "explored_state_count", "reproduced_levels") and artifact[field] < 0:
            errors.append(f"{field} must be non-negative")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    for field in (
        "hidden_source_reading",
        "offline_bfs_used",
        "hand_adapter_used",
        "outer_loop_re_used",
    ):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
        elif artifact.get(field) is not False:
            errors.append(f"{field} must be false")
    for field in ("offline_reproduced", "new_level_banked", "registry_updated"):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
    if not isinstance(artifact.get("failed_hypotheses"), list):
        errors.append("failed_hypotheses must be a list")
    if not isinstance(artifact.get("first_win_trace_path"), str):
        errors.append("first_win_trace_path must be a string")

    before = _as_int(artifact.get("reproduced_levels_before"))
    after = _as_int(artifact.get("reproduced_levels_after"))
    target_level = _as_int(artifact.get("target_level"))
    if artifact.get("offline_reproduced") is True:
        if _as_int(artifact.get("reproduced_levels")) < 1:
            errors.append("offline_reproduced requires reproduced_levels >= 1")
        if after <= before:
            errors.append("offline_reproduced requires reproduced_levels_after > reproduced_levels_before")
        if after < target_level:
            errors.append("offline_reproduced requires reproduced_levels_after >= target_level")
    if artifact.get("new_level_banked") is True:
        if artifact.get("offline_reproduced") is not True:
            errors.append("new_level_banked requires offline_reproduced true")
        if artifact.get("registry_updated") is not True:
            errors.append("new_level_banked requires registry_updated true")
        if not artifact.get("first_win_trace_path"):
            errors.append("new_level_banked requires first_win_trace_path")
    if artifact.get("registry_updated") is True and artifact.get("new_level_banked") is not True:
        errors.append("registry_updated requires new_level_banked true")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "honest_null:", "blocked:")):
        errors.append("honest_verdict must start with complete:, honest_null:, or blocked:")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def update_registry_if_banked(
    *,
    root: Path,
    artifact: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> bool:
    if artifact.get("new_level_banked") is not True:
        return False
    updated = dict(registry)
    games = list(updated.get("games") or [])
    game = str(artifact["game"])
    after = _as_int(artifact.get("reproduced_levels_after"))
    delta = _as_int(artifact.get("reproduced_levels"))
    row_found = False
    for row in games:
        if isinstance(row, dict) and row.get("game") == game:
            row["reproducibility"] = "reproduced"
            row["levels_reproduced"] = after
            row["latest_exp5480_levelup_attempt"] = {
                "artifact": RESULT_RELATIVE_PATH,
                "offline_reproduced": True,
                "reproduced_levels_before": _as_int(artifact.get("reproduced_levels_before")),
                "reproduced_levels_after": after,
                "new_levels_banked": delta,
                "solve_provenance": SOLVE_PROVENANCE,
                "first_win_trace_path": str(artifact.get("first_win_trace_path") or ""),
            }
            row_found = True
            break
    if not row_found:
        games.append(
            {
                "game": game,
                "reproducibility": "reproduced",
                "levels_reproduced": after,
                "latest_exp5480_levelup_attempt": {
                    "artifact": RESULT_RELATIVE_PATH,
                    "offline_reproduced": True,
                    "reproduced_levels_after": after,
                    "new_levels_banked": delta,
                    "solve_provenance": SOLVE_PROVENANCE,
                    "first_win_trace_path": str(artifact.get("first_win_trace_path") or ""),
                },
            }
        )
    updated["games"] = games
    updated["reproducible_total_levels"] = _as_int(
        updated.get("reproducible_total_levels")
    ) + delta
    path = root / REGISTRY_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")
    return True


def _write_artifact(root: Path, artifact: Mapping[str, Any]) -> None:
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    *,
    root: Path = REPO,
    budget: int = DEFAULT_BUDGET,
    attempt_runner: Callable[..., Mapping[str, Any]] = run_rotated_live_salience_attempt,
    offline_arcade_check: Callable[[], bool] = offline_arcade_available,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(root)
    spec_path = root / SPEC_RELATIVE_PATH
    registry_path = root / REGISTRY_RELATIVE_PATH
    exp5479_path = root / EXP5479_RELATIVE_PATH
    preconditions = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "spec_has_req_5480": (
            "REQ-ARC-FCP-5480" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
            else False
        ),
        "registry_present": registry_path.exists(),
        "exp5479_present": exp5479_path.exists(),
        "offline_arcade_available": False,
        "hidden_source_reading": False,
        "offline_bfs_used": False,
        "hand_adapter_used": False,
        "outer_loop_re_used": False,
    }
    registry = load_registry(root)
    exp5479_artifact = load_json(exp5479_path)
    target = select_exp5479_target(exp5479_artifact, registry)
    ready_without_arcade = (
        preconditions["AGENTS.md"]
        and preconditions["CODEX.md"]
        and preconditions["spec_has_req_5480"]
        and preconditions["registry_present"]
        and preconditions["exp5479_present"]
        and not target.get("blocked")
    )
    if not ready_without_arcade:
        attempt: Mapping[str, Any] = {
            "blocked": True,
            "failure_mode": str(target.get("blocker") or "missing_exp5480_precondition"),
            "action_count": 0,
            "explored_state_count": 0,
            "offline_reproduced": False,
            "reproduced_levels_after": _as_int(target.get("reproduced_levels_before")),
            "source_reading_used": False,
            "offline_bfs_used": False,
            "hand_adapter_used": False,
            "outer_loop_re_used": False,
        }
    else:
        preconditions["offline_arcade_available"] = bool(offline_arcade_check())
        if not preconditions["offline_arcade_available"]:
            attempt = {
                "blocked": True,
                "failure_mode": "missing_harness_access",
                "action_count": 0,
                "explored_state_count": 0,
                "offline_reproduced": False,
                "reproduced_levels_after": _as_int(target.get("reproduced_levels_before")),
                "source_reading_used": False,
                "offline_bfs_used": False,
                "hand_adapter_used": False,
                "outer_loop_re_used": False,
            }
        else:
            attempt = attempt_runner(
                root=root,
                target=target,
                exp5479_artifact=exp5479_artifact,
                budget=budget,
            )

    artifact = build_artifact(
        target=target,
        exp5479_artifact=exp5479_artifact,
        attempt=attempt,
        registry_updated=False,
        preconditions_checked=preconditions,
        tests_run=tests_run,
        duration_s=time.monotonic() - started,
    )
    if artifact["new_level_banked"]:
        registry_updated = update_registry_if_banked(
            root=root,
            artifact=artifact,
            registry=registry,
        )
        artifact["registry_updated"] = bool(registry_updated)
    validate_artifact(artifact)
    _write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
