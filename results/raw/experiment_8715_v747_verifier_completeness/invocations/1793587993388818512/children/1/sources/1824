"""Exp5521: ARC live action-diverse level-up attempt.

Spec refs: REQ-ARC-FCP-5521, SCENARIO-ARC-FCP-5521.

This module is the accounting shell for one bounded live-agent attempt after
Exp5520 selected a registry-safe target. The credited path is deliberately
narrow: the agent may use only runtime frames, the action-diverse
connected-component/color-blob generator, and the standard replay gate for a
receipt. Source reading, offline ground-truth BFS, and hand-built per-game
adapters are tracked as explicit disqualifiers because they answer a different
question than live hidden-game self-discovery.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from carnot.agentic.arc_perception_generation import ClassicalPerceptionGenerator


JsonDict = dict[str, Any]

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5521
EXPERIMENT = "experiment_5521_arc_live_action_diverse_levelup"
MILESTONE = "2026.07.500"
RESULT_RELATIVE_PATH = "results/experiment_5521_arc_live_action_diverse_levelup.json"
TRAJECTORY_LOG_RELATIVE_PATH = (
    "results/experiment_5521_arc_live_action_diverse_levelup_trajectory.json"
)
PRECHECK_RELATIVE_PATH = "results/experiment_5520_arc_action_diversity_target_precheck.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5521", "SCENARIO-ARC-FCP-5521"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "arc_live_agent_self_discovery"
DEFAULT_BUDGET = 48
DEFAULT_TESTS_RUN = [
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_5521_arc_live_action_diverse_levelup.py "
        "-q --no-cov"
    ),
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5521_arc_live_action_diverse_levelup.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/experiment_5521_arc_live_action_diverse_levelup.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "selected_game": "Exp5520-selected registry-safe game id; empty only when the readiness gate blocks before live runtime.",
    "selected_level": "Exp5520-selected unreproduced level label or int; success must be strictly deeper than the registry depth.",
    "offline_reproduced": "true only when the live-discovered trajectory passes the standard offline replay gate.",
    "reproduced_levels": "integer new levels banked from the live-discovered trajectory; success requires >=1.",
    "banking_gate": "bare bool equal to offline_reproduced=true and reproduced_levels>=1 for solve_provenance=live_agent_self_discovery.",
    "registry_delta": "bare int registry total delta; nonzero only when the banking gate is true.",
    "solve_provenance": "must equal live_agent_self_discovery.",
    "live_attempts": "bare int count of runtime actions executed by the live agent.",
    "action_entropy": "Shannon entropy over executed live action/coordinate choices as a bare float.",
    "repeated_coordinate_rate": "fraction of executed live coordinate choices that repeated a prior coordinate, as a bare float.",
    "salience_coverage_rate": "fraction of executed live coordinate choices covering proposed salience coordinates, as a bare float.",
    "trajectory_log_path": "path to the detailed trajectory log containing observations, proposed actions, verifier feedback, and diversity metrics.",
    "reproduction_command": "exact replay command when a live trajectory was reproduced, else null.",
    "arc_live_levelup_ready": "bare bool proving Exp5520 and the registry reread allowed the live attempt.",
    "inference_substrate": "must equal arc_live_agent_self_discovery.",
    "honest_verdict": "one-line verdict starting complete:, honest_null:, or blocked:.",
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _level_label(level: int) -> str:
    return f"L{max(0, int(level))}"


def _parse_level_label(value: Any) -> int:
    text = str(value or "").strip().upper()
    if text.startswith("L") and text[1:].isdigit():
        return int(text[1:])
    if text.isdigit():
        return int(text)
    return 0


def _read_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> JsonDict:
    if not path.exists():
        return {"reproducible_total_levels": 0, "games": []}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {
        "reproducible_total_levels": 0,
        "games": [],
    }


def _registry_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("game")): row
        for row in registry.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    }


def _registry_depth(registry: Mapping[str, Any], game: str) -> int:
    return _as_int((_registry_rows(registry).get(game) or {}).get("levels_reproduced"))


def _registry_total(registry: Mapping[str, Any]) -> int:
    return _as_int(registry.get("reproducible_total_levels"))


def _row_data(row: Mapping[str, Any]) -> Mapping[str, Any]:
    data = row.get("data")
    return data if isinstance(data, Mapping) else {}


def _row_coordinate(row: Mapping[str, Any]) -> tuple[int, int] | None:
    data = _row_data(row)
    if "x" in data and "y" in data:
        return _as_int(data.get("x")), _as_int(data.get("y"))
    if "x" in row and "y" in row:
        return _as_int(row.get("x")), _as_int(row.get("y"))
    return None


def _row_signature(row: Mapping[str, Any]) -> str:
    coord = _row_coordinate(row)
    if coord is not None:
        return f"A{_as_int(row.get('action'))}@{coord[0]},{coord[1]}"
    return f"A{_as_int(row.get('action'))}"


def _pattern_coordinates(precheck: Mapping[str, Any]) -> set[tuple[int, int]]:
    pattern = precheck.get("exp5508_pattern")
    rows = pattern.get("coordinates") if isinstance(pattern, Mapping) else []
    coords: set[tuple[int, int]] = set()
    for row in rows or []:
        if isinstance(row, Mapping) and "x" in row and "y" in row:
            coords.add((_as_int(row.get("x")), _as_int(row.get("y"))))
    return coords


def _policy_salience_diagnostics(policy: Any) -> JsonDict:
    if hasattr(policy, "action_salience_diagnostics"):
        return dict(policy.action_salience_diagnostics())
    explorer = getattr(policy, "explorer", None)
    if explorer is not None and hasattr(explorer, "action_salience_diagnostics"):
        return dict(explorer.action_salience_diagnostics())
    return {
        "connected_component_salience_enabled": False,
        "salience_tiers_emitted": False,
        "generation_stage_action_prioritization": False,
        "tier_rows": [],
        "action_tier_rows": [],
    }


class ActionDiverseLiveGenerator(ClassicalPerceptionGenerator):
    """Runtime perception generator with conservative coordinate diversity.

    The generator still derives candidates from rendered frames through the base
    classical perception pass. The extra state is just live bookkeeping: avoid
    Exp5508's known failed coordinates and avoid click coordinates already tried
    during this attempt, so a short budget does not collapse into the same small
    coordinate loop.
    """

    def __init__(
        self,
        *,
        max_candidates: int = 8,
        avoid_coordinates: set[tuple[int, int]] | frozenset[tuple[int, int]] = frozenset(),
    ) -> None:
        super().__init__(max_candidates=max_candidates)
        self._avoid_coordinates = set(avoid_coordinates)
        self._attempted_coordinates: set[tuple[int, int]] = set()
        self._suppressed_coordinate_count = 0

    def for_path(self, path: Sequence[Mapping[str, Any]]) -> "ActionDiverseLiveGenerator":
        for row in path:
            coord = _row_coordinate(row) if isinstance(row, Mapping) else None
            if coord is not None:
                self._attempted_coordinates.add(coord)
        return self

    def _coordinate_blocked(self, point: tuple[int, int], seen: set[tuple[int, int]]) -> bool:
        blocked = point in self._avoid_coordinates or point in self._attempted_coordinates or point in seen
        if blocked:
            self._suppressed_coordinate_count += 1
        return blocked

    def click_points(self, frame: Any, *, max_points: int | None = None) -> list[tuple[int, int]]:
        limit = int(self.max_candidates if max_points is None else max_points)
        raw_limit = max(limit + len(self._avoid_coordinates) + len(self._attempted_coordinates), limit)
        raw = super().click_points(frame, max_points=raw_limit)
        selected: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for x, y in raw:
            point = (int(x), int(y))
            if self._coordinate_blocked(point, seen):
                continue
            seen.add(point)
            selected.append(point)
            if len(selected) >= limit:
                break
        return selected

    def action_tier_rows(self, frame: Any, candidates: Sequence[Any]) -> list[dict[str, Any]]:
        rows = super().action_tier_rows(frame, candidates)
        filtered: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for row in rows:
            coord = _row_coordinate(row)
            if coord is not None and self._coordinate_blocked(coord, seen):
                continue
            if coord is not None:
                seen.add(coord)
            filtered.append(row)
        return filtered or rows[:1]

    def observe_transition(self, before: Any, action: int, data: Any, after: Any) -> None:
        super().observe_transition(before, action, data, after)
        if int(action) == 6 and isinstance(data, Mapping) and "x" in data and "y" in data:
            self._attempted_coordinates.add((_as_int(data.get("x")), _as_int(data.get("y"))))

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = super().diagnostics()
        diagnostics["source"] = "action_diverse_connected_component_color_blob_perception_generation"
        diagnostics["action_diversity"] = {
            "avoid_coordinates": [
                {"x": x, "y": y} for x, y in sorted(self._avoid_coordinates)
            ],
            "attempted_coordinates": [
                {"x": x, "y": y} for x, y in sorted(self._attempted_coordinates)
            ],
            "suppressed_coordinate_count": int(self._suppressed_coordinate_count),
        }
        return diagnostics


def select_target_from_precheck(
    precheck: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> JsonDict:
    """REQ-ARC-FCP-5521: require Exp5520 readiness and a fresh registry depth."""

    total = _registry_total(registry)
    game = str(precheck.get("selected_game") or "")
    selected_level = precheck.get("selected_level")
    target_level = _parse_level_label(selected_level)
    if not precheck:
        return _blocked_target("exp5520_target_missing", total, game, selected_level, target_level)
    if precheck.get("arc_levelup_candidate_ready") is not True:
        return _blocked_target("exp5520_candidate_not_ready", total, game, selected_level, target_level)
    if precheck.get("solve_provenance") != SOLVE_PROVENANCE:
        return _blocked_target("exp5520_wrong_provenance", total, game, selected_level, target_level)
    if not game:
        return _blocked_target("exp5520_target_missing", total, game, selected_level, target_level)
    if target_level <= 0:
        return _blocked_target("exp5520_selected_level_malformed", total, game, selected_level, target_level)
    prior = _registry_depth(registry, game)
    if prior >= target_level:
        return _blocked_target(
            "selected_level_already_reproducible",
            total,
            game,
            _level_label(target_level),
            target_level,
            prior,
        )
    return {
        "blocked": False,
        "selected_game": game,
        "selected_level": _level_label(target_level),
        "target_level": int(target_level),
        "prior_levels_reproduced": int(prior),
        "registry_before_levels": int(total),
        "selection_reason": "exp5520_action_diversity_target_survived_registry_reread",
    }


def _blocked_target(
    blocker: str,
    total: int,
    game: str,
    selected_level: Any,
    target_level: int,
    prior: int = 0,
) -> JsonDict:
    return {
        "blocked": True,
        "blocker": str(blocker),
        "selected_game": str(game or ""),
        "selected_level": str(selected_level or ""),
        "target_level": int(max(0, target_level)),
        "prior_levels_reproduced": int(max(0, prior)),
        "registry_before_levels": int(total),
        "selection_reason": str(blocker),
    }


def offline_arcade_available() -> bool:  # pragma: no cover - environment probe
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
    except Exception:
        return False
    return True


class _NoOpProposer:  # pragma: no cover - ARC runtime boundary
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "disabled_exp5521_no_llm"

    def world_model_candidates(self, _game: str) -> list[Any]:
        return []


def _action_label(action: int, data: Any) -> str:  # pragma: no cover - ARC runtime boundary
    from carnot.experiment_5494_arc_live_trajectory_option_induction_v498 import (
        _action_label as labeler,
    )

    return labeler(action, data)


def _apply_action_label(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - ARC runtime boundary
    from carnot.experiment_5494_arc_live_trajectory_option_induction_v498 import (
        _apply_action_label as applier,
    )

    return applier(*args, **kwargs)


def run_live_action_diverse_attempt(  # pragma: no cover - ARC runtime boundary
    *,
    root: Path,
    target: Mapping[str, Any],
    precheck: Mapping[str, Any],
    budget: int = DEFAULT_BUDGET,
) -> JsonDict:
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of

    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    game = str(target["selected_game"])
    prior = _as_int(target.get("prior_levels_reproduced"))
    target_level = _as_int(target.get("target_level"))
    labels: list[str] = []
    frames: list[Any] = []
    observations: list[JsonDict] = []
    proposed_rows: list[JsonDict] = []
    action_rows: list[JsonDict] = []
    latest = None
    max_level = prior
    generator = ActionDiverseLiveGenerator(
        max_candidates=8,
        avoid_coordinates=_pattern_coordinates(precheck),
    )
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
            action_prior=generator,
            qd_generator=generator,
            goal_bias=None,
            goal_candidate_guidance=False,
            active_probe_controller=False,
            go_explore_archive=False,
        )
        for index in range(1, max(1, int(budget)) + 1):
            if policy.is_done(frames, latest):
                break
            before = latest
            before_level = _as_int(_level_of(before), default=max_level) if before is not None else max_level
            kind, data = policy.next_move(frames, latest)
            salience = _policy_salience_diagnostics(policy)
            for row in salience.get("action_tier_rows") or []:
                if isinstance(row, Mapping):
                    proposed_rows.append({"step": int(index), **dict(row)})
            if kind == "RESET":
                latest = env.reset()
                observations.append(
                    {"step": int(index), "event": "reset", "level": _as_int(_level_of(latest))}
                )
                if labels:
                    labels.append("RESET")
            elif kind is None:
                observations.append(
                    {"step": int(index), "event": "policy_exhausted", "level": int(max_level)}
                )
                break
            else:
                latest = env.step(getattr(GameAction, f"ACTION{int(kind)}"), data=data)
                label = _action_label(int(kind), data)
                labels.append(label)
                after_level = _as_int(_level_of(latest), default=max_level)
                max_level = max(max_level, after_level)
                row = {
                    "step": int(index),
                    "action": int(kind),
                    "data": dict(data) if isinstance(data, Mapping) else data,
                    "label": label,
                    "level_before": int(before_level),
                    "level_after": int(after_level),
                }
                action_rows.append(row)
                observations.append({"step": int(index), "event": "action", **row})
            frames.append(latest)
            if latest is None or max_level >= target_level:
                break

        gate: JsonDict = {
            "game": game,
            "claimed_level": max_level,
            "reached_level": prior,
            "reproduced": False,
            "mode": "standard_reproduction_gate_not_run_no_new_target_candidate",
        }
        if max_level > prior and labels:
            gate = dict(kit.reproduce(game, labels, _apply_action_label, claimed_level=max_level))
        reached = _as_int(gate.get("reached_level"), default=max_level)
        reproduced = bool(gate.get("reproduced")) and reached >= target_level
        if not reproduced:
            generator.record_scope_failure("bounded_budget_no_target_level_reproduction")
        diagnostics = generator.diagnostics()
        return {
            "live_agent_attempts": int(len(action_rows)),
            "post_levels_reproduced": int(reached if reproduced else prior),
            "offline_reproduced": bool(reproduced),
            "reproduced_levels": max(0, int(reached) - int(prior)) if reproduced else 0,
            "action_rows": action_rows,
            "proposed_action_rows": proposed_rows,
            "observations": observations,
            "verifier_feedback": {"reproduction_gate": gate},
            "reproduction_gate": gate,
            "solution_labels": list(labels) if reproduced else [],
            "failure_mode": "" if reproduced else "bounded_budget_no_target_level_reproduction",
            "offline_bfs_used": False,
            "game_source_read": False,
            "hand_built_per_game_adapter_used": False,
            "methodology_receipt": (
                f"bounded_live_runtime budget={int(budget)} "
                "mechanism=action_diverse_perception_generation "
                "gate=standard_reproduction prohibited_inputs=false"
            ),
            "generator_diagnostics": diagnostics,
            "root": str(root),
        }
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def _blocked_attempt(reason: str) -> JsonDict:
    return {
        "blocked": True,
        "live_agent_attempts": 0,
        "post_levels_reproduced": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "action_rows": [],
        "proposed_action_rows": [],
        "observations": [{"event": "blocked", "reason": str(reason)}],
        "verifier_feedback": {"reproduction_gate": {"reproduced": False, "reason": str(reason)}},
        "reproduction_gate": {"reproduced": False, "reason": str(reason)},
        "solution_labels": [],
        "failure_mode": str(reason),
        "offline_bfs_used": False,
        "game_source_read": False,
        "hand_built_per_game_adapter_used": False,
        "methodology_receipt": (
            f"blocked_before_live_runtime reason={reason} "
            "mechanism=action_diverse_perception_generation "
            "gate=standard_reproduction prohibited_inputs=false"
        ),
    }


def trajectory_metrics(
    action_rows: Sequence[Mapping[str, Any]],
    proposed_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """SCENARIO-ARC-FCP-5521: measure live action diversity from executed rows."""

    signatures = [_row_signature(row) for row in action_rows if isinstance(row, Mapping)]
    counts = Counter(signatures)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        probability = float(count) / float(total or 1)
        if probability:
            entropy -= probability * math.log2(probability)

    seen: set[tuple[int, int]] = set()
    repeated = 0
    coordinate_count = 0
    for row in action_rows:
        coord = _row_coordinate(row) if isinstance(row, Mapping) else None
        if coord is None:
            continue
        coordinate_count += 1
        if coord in seen:
            repeated += 1
        seen.add(coord)

    proposed_coords = {
        coord
        for row in proposed_rows
        if isinstance(row, Mapping) and (coord := _row_coordinate(row)) is not None
    }
    coverage = float(len(seen & proposed_coords)) / float(max(1, len(proposed_coords)))
    return {
        "action_entropy": float(entropy),
        "repeated_coordinate_rate": float(repeated) / float(max(1, coordinate_count)),
        "salience_coverage_rate": min(1.0, max(0.0, coverage)),
    }


def _accepted_reproduced_levels(target: Mapping[str, Any], attempt: Mapping[str, Any]) -> int:
    if attempt.get("offline_reproduced") is not True:
        return 0
    if (
        attempt.get("offline_bfs_used", False)
        or attempt.get("game_source_read", False)
        or attempt.get("hand_built_per_game_adapter_used", False)
    ):
        return 0
    prior = _as_int(target.get("prior_levels_reproduced"))
    target_level = _as_int(target.get("target_level"))
    post = _as_int(attempt.get("post_levels_reproduced"), prior + _as_int(attempt.get("reproduced_levels")))
    if post <= prior or post < target_level:
        return 0
    return max(0, _as_int(attempt.get("reproduced_levels"), post - prior))


def _reproduction_command(trajectory_log_path: str, banking_gate: bool) -> str | None:
    if not banking_gate:
        return None
    return (
        ".venv/bin/python -m carnot.experiment_5521_arc_live_action_diverse_levelup "
        f"--reproduce-log {trajectory_log_path}"
    )


def build_artifact(
    *,
    target: Mapping[str, Any],
    attempt: Mapping[str, Any],
    registry_updated: bool,
    trajectory_log_path: str,
    precheck: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
) -> JsonDict:
    """SCENARIO-ARC-FCP-5521: build the required live-attempt artifact."""

    del precheck
    blocked = bool(target.get("blocked")) or bool(attempt.get("blocked"))
    accepted_delta = _accepted_reproduced_levels(target, attempt)
    can_bank = bool(accepted_delta >= 1 and registry_updated and not blocked)
    before_total = _as_int(target.get("registry_before_levels"))
    registry_delta = int(accepted_delta if can_bank else 0)
    metrics = trajectory_metrics(
        [row for row in attempt.get("action_rows") or [] if isinstance(row, Mapping)],
        [row for row in attempt.get("proposed_action_rows") or [] if isinstance(row, Mapping)],
    )
    selected_game = str(target.get("selected_game") or "") if not blocked else ""
    selected_level = str(target.get("selected_level") or "") if not blocked else ""
    failure_mode = str(
        attempt.get("failure_mode")
        or target.get("blocker")
        or "bounded_budget_no_target_level_reproduction"
    )
    status = "complete" if can_bank else "blocked" if blocked else "honest_null"
    reproduction_command = _reproduction_command(trajectory_log_path, can_bank)
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "carnot.experiment_5521_arc_live_action_diverse_levelup.v1",
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "selected_game": selected_game,
        "selected_level": selected_level,
        "offline_reproduced": bool(can_bank),
        "reproduced_levels": int(accepted_delta if can_bank else 0),
        "banking_gate": bool(can_bank),
        "registry_delta": registry_delta,
        "solve_provenance": SOLVE_PROVENANCE,
        "live_attempts": _as_int(attempt.get("live_agent_attempts")),
        "action_entropy": float(metrics["action_entropy"]),
        "repeated_coordinate_rate": float(metrics["repeated_coordinate_rate"]),
        "salience_coverage_rate": float(metrics["salience_coverage_rate"]),
        "trajectory_log_path": str(trajectory_log_path),
        "reproduction_command": reproduction_command,
        "arc_live_levelup_ready": bool(not blocked),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"complete: {target.get('selected_game')} {target.get('selected_level')} live action-diverse trajectory reproduced and banked"
            if can_bank
            else f"blocked: {failure_mode}"
            if status == "blocked"
            else (
                f"honest_null: {target.get('selected_game')} {target.get('selected_level')} "
                f"{failure_mode}; entropy={metrics['action_entropy']:.3f}; "
                f"repeat_rate={metrics['repeated_coordinate_rate']:.3f}; registry_delta=0"
            )
        ),
        "selected_target_level": _as_int(target.get("target_level")),
        "prior_levels_reproduced": _as_int(target.get("prior_levels_reproduced")),
        "registry_before_levels": int(before_total),
        "registry_after_levels": int(before_total + registry_delta),
        "post_levels_reproduced": (
            _as_int(attempt.get("post_levels_reproduced"))
            if can_bank
            else _as_int(target.get("prior_levels_reproduced"))
        ),
        "registry_updated": bool(registry_updated),
        "offline_bfs_used": bool(attempt.get("offline_bfs_used", False)),
        "game_source_read": bool(attempt.get("game_source_read", False)),
        "hand_built_per_game_adapter_used": bool(
            attempt.get("hand_built_per_game_adapter_used", False)
        ),
        "methodology_receipt": str(attempt.get("methodology_receipt") or ""),
        "target_selection": dict(target),
        "attempt_summary": {
            "failure_mode": failure_mode,
            "action_count": len(attempt.get("action_rows") or []),
            "proposed_action_count": len(attempt.get("proposed_action_rows") or []),
            "reproduction_gate": dict(attempt.get("reproduction_gate") or {}),
        },
        "preconditions_checked": dict(preconditions_checked),
        "tests_run": list(tests_run),
        "duration_s": float(duration_s),
    }


def build_trajectory_log(
    *,
    target: Mapping[str, Any],
    attempt: Mapping[str, Any],
    artifact: Mapping[str, Any],
    precheck: Mapping[str, Any],
) -> JsonDict:
    """SCENARIO-ARC-FCP-5521: keep enough trajectory detail for audit."""

    return {
        "schema": "carnot.experiment_5521_arc_live_action_diverse_levelup.trajectory.v1",
        "experiment": EXPERIMENT,
        "selected_game": artifact.get("selected_game") or target.get("selected_game") or "",
        "selected_level": artifact.get("selected_level") or target.get("selected_level") or "",
        "target_selection": dict(target),
        "observations": list(attempt.get("observations") or []),
        "proposed_actions": list(attempt.get("proposed_action_rows") or []),
        "executed_actions": list(attempt.get("action_rows") or []),
        "verifier_feedback": dict(attempt.get("verifier_feedback") or {}),
        "solution_labels": list(attempt.get("solution_labels") or []),
        "metrics": {
            "action_entropy": artifact.get("action_entropy"),
            "repeated_coordinate_rate": artifact.get("repeated_coordinate_rate"),
            "salience_coverage_rate": artifact.get("salience_coverage_rate"),
        },
        "exp5508_comparison": {
            "selected_game": (precheck.get("exp5508_pattern") or {}).get("selected_game")
            if isinstance(precheck.get("exp5508_pattern"), Mapping)
            else "",
            "selected_level": (precheck.get("exp5508_pattern") or {}).get("selected_level")
            if isinstance(precheck.get("exp5508_pattern"), Mapping)
            else "",
            "action_entropy": (precheck.get("exp5508_pattern") or {}).get("action_entropy")
            if isinstance(precheck.get("exp5508_pattern"), Mapping)
            else None,
            "repeated_coordinate_rate": (precheck.get("exp5508_pattern") or {}).get(
                "repeated_coordinate_rate"
            )
            if isinstance(precheck.get("exp5508_pattern"), Mapping)
            else None,
        },
        "prohibited_inputs": {
            "offline_bfs_used": bool(attempt.get("offline_bfs_used", False)),
            "game_source_read": bool(attempt.get("game_source_read", False)),
            "hand_built_per_game_adapter_used": bool(
                attempt.get("hand_built_per_game_adapter_used", False)
            ),
        },
    }


def write_trajectory_log(path: Path, log: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in artifact
    ]
    if not isinstance(artifact.get("selected_game"), str):
        errors.append("selected_game must be a string")
    if not isinstance(artifact.get("selected_level"), (str, int)):
        errors.append("selected_level must be a string or int")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    elif artifact["reproduced_levels"] < 0:
        errors.append("reproduced_levels must be non-negative")
    if type(artifact.get("banking_gate")) is not bool:
        errors.append("banking_gate must be bare bool")
    if type(artifact.get("registry_delta")) is not int:
        errors.append("registry_delta must be bare int")
    elif artifact["registry_delta"] < 0:
        errors.append("registry_delta must be non-negative")
    if artifact.get("offline_reproduced") is True:
        if _as_int(artifact.get("reproduced_levels")) < 1:
            errors.append("offline_reproduced requires reproduced_levels >= 1")
        if artifact.get("banking_gate") is not True:
            errors.append("offline_reproduced true requires banking_gate true")
    if artifact.get("banking_gate") is True:
        if artifact.get("offline_reproduced") is not True:
            errors.append("banking_gate true requires offline_reproduced true")
        if _as_int(artifact.get("registry_delta")) != _as_int(artifact.get("reproduced_levels")):
            errors.append("banking_gate true requires registry_delta == reproduced_levels")
        if not isinstance(artifact.get("reproduction_command"), str):
            errors.append("banking_gate true requires reproduction_command")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    if type(artifact.get("live_attempts")) is not int:
        errors.append("live_attempts must be bare int")
    elif artifact["live_attempts"] < 0:
        errors.append("live_attempts must be non-negative")
    for field in ("action_entropy", "repeated_coordinate_rate", "salience_coverage_rate"):
        if type(artifact.get(field)) is not float:
            errors.append(f"{field} must be bare float")
    for field in ("repeated_coordinate_rate", "salience_coverage_rate"):
        if type(artifact.get(field)) in (float, int) and not (0.0 <= float(artifact[field]) <= 1.0):
            errors.append(f"{field} must be in [0, 1]")
    if not isinstance(artifact.get("trajectory_log_path"), str) or not artifact.get(
        "trajectory_log_path"
    ):
        errors.append("trajectory_log_path must be a non-empty string")
    if artifact.get("reproduction_command") is not None and not isinstance(
        artifact.get("reproduction_command"), str
    ):
        errors.append("reproduction_command must be string or null")
    if type(artifact.get("arc_live_levelup_ready")) is not bool:
        errors.append("arc_live_levelup_ready must be bare bool")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be arc_live_agent_self_discovery")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "honest_null:", "blocked:")):
        errors.append("honest_verdict must start with complete:, honest_null:, or blocked:")
    for field in ("offline_bfs_used", "game_source_read", "hand_built_per_game_adapter_used"):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
        elif artifact.get(field) is not False:
            errors.append(f"{field} must be false")
    if not isinstance(artifact.get("methodology_receipt"), str) or not artifact.get(
        "methodology_receipt"
    ):
        errors.append("methodology_receipt must be a non-empty string")
    if artifact.get("registry_updated") is True and artifact.get("banking_gate") is not True:
        errors.append("registry_updated requires banking_gate true")
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
    if artifact.get("offline_reproduced") is not True or artifact.get("banking_gate") is not True:
        return False
    updated = dict(registry)
    games = list(updated.get("games") or [])
    game = str(artifact["selected_game"])
    post = _as_int(artifact.get("post_levels_reproduced"))
    delta = _as_int(artifact.get("reproduced_levels"))
    found = False
    for row in games:
        if isinstance(row, dict) and row.get("game") == game:
            row["reproducibility"] = "reproduced"
            row["levels_reproduced"] = post
            row["latest_exp5521_levelup_attempt"] = {
                "artifact": RESULT_RELATIVE_PATH,
                "trajectory_log_path": artifact.get("trajectory_log_path"),
                "offline_reproduced": True,
                "reproduced_levels": delta,
                "prior_levels_reproduced": _as_int(artifact.get("prior_levels_reproduced")),
                "post_levels_reproduced": post,
                "registry_delta": delta,
                "solve_provenance": SOLVE_PROVENANCE,
                "reproduction_command": artifact.get("reproduction_command"),
            }
            found = True
            break
    if not found:
        games.append(
            {
                "game": game,
                "reproducibility": "reproduced",
                "levels_reproduced": post,
                "latest_exp5521_levelup_attempt": {
                    "artifact": RESULT_RELATIVE_PATH,
                    "trajectory_log_path": artifact.get("trajectory_log_path"),
                    "offline_reproduced": True,
                    "reproduced_levels": delta,
                    "post_levels_reproduced": post,
                    "solve_provenance": SOLVE_PROVENANCE,
                    "reproduction_command": artifact.get("reproduction_command"),
                },
            }
        )
    updated["games"] = games
    updated["reproducible_total_levels"] = _registry_total(updated) + delta
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
    attempt_runner: Callable[..., Mapping[str, Any]] = run_live_action_diverse_attempt,
    offline_arcade_check: Callable[[], bool] = offline_arcade_available,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    started = time.monotonic()
    root = Path(root)
    spec_path = root / SPEC_RELATIVE_PATH
    registry_path = root / REGISTRY_RELATIVE_PATH
    precheck_path = root / PRECHECK_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    preconditions = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "OPENCODE.md": (root / "OPENCODE.md").exists(),
        "CLAUDE.md": (root / "CLAUDE.md").exists(),
        "spec_has_req_5521": "REQ-ARC-FCP-5521" in spec_text,
        "registry_present": registry_path.exists(),
        "exp5520_precheck_present": precheck_path.exists(),
        "exp5520_ready": False,
        "offline_arcade_available": False,
        "offline_bfs_used": False,
        "game_source_read": False,
        "hand_built_per_game_adapter_used": False,
    }
    registry = _read_yaml(registry_path)
    precheck = _read_json(precheck_path)
    preconditions["exp5520_ready"] = precheck.get("arc_levelup_candidate_ready") is True
    target = select_target_from_precheck(precheck, registry)
    ready_without_arcade = (
        preconditions["AGENTS.md"]
        and (preconditions["CODEX.md"] or preconditions["OPENCODE.md"])
        and preconditions["CLAUDE.md"]
        and preconditions["spec_has_req_5521"]
        and preconditions["registry_present"]
        and preconditions["exp5520_precheck_present"]
        and not target.get("blocked")
    )
    if not ready_without_arcade:
        reason = str(target.get("blocker") or "missing_exp5521_precondition")
        attempt: Mapping[str, Any] = _blocked_attempt(reason)
    else:
        preconditions["offline_arcade_available"] = bool(offline_arcade_check())
        if not preconditions["offline_arcade_available"]:
            attempt = _blocked_attempt("missing_harness_access")
        else:
            attempt = attempt_runner(root=root, target=target, precheck=precheck, budget=budget)

    trajectory_log_path = TRAJECTORY_LOG_RELATIVE_PATH
    registry_updated = False
    if _accepted_reproduced_levels(target, attempt) >= 1:
        preliminary = build_artifact(
            target=target,
            attempt=attempt,
            registry_updated=True,
            trajectory_log_path=trajectory_log_path,
            precheck=precheck,
            preconditions_checked=preconditions,
            tests_run=tests_run,
            duration_s=time.monotonic() - started,
        )
        registry_updated = update_registry_if_banked(
            root=root,
            artifact=preliminary,
            registry=registry,
        )
    artifact = build_artifact(
        target=target,
        attempt=attempt,
        registry_updated=registry_updated,
        trajectory_log_path=trajectory_log_path,
        precheck=precheck,
        preconditions_checked=preconditions,
        tests_run=tests_run,
        duration_s=time.monotonic() - started,
    )
    validate_artifact(artifact)
    log = build_trajectory_log(
        target=target,
        attempt=attempt,
        artifact=artifact,
        precheck=precheck,
    )
    write_trajectory_log(root / trajectory_log_path, log)
    _write_artifact(root, artifact)
    return artifact


def reproduce_from_log(  # pragma: no cover - replay CLI boundary
    log_path: Path,
    *,
    reproducer: Callable[[str, Sequence[str], Callable[..., Any], int], Mapping[str, Any]] | None = None,
) -> JsonDict:
    log = _read_json(Path(log_path))
    labels = [str(label) for label in log.get("solution_labels") or []]
    game = str(log.get("selected_game") or "")
    claimed_level = _parse_level_label(log.get("selected_level"))
    if reproducer is None:
        from carnot.agentic import arc_solver_kit as kit

        def reproducer(game: str, labels: Sequence[str], applier: Callable[..., Any], claimed: int) -> Mapping[str, Any]:
            return kit.reproduce(game, labels, applier, claimed_level=claimed)

    return dict(reproducer(game, labels, _apply_action_label, claimed_level))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--reproduce-log")
    args = parser.parse_args(argv)
    if args.reproduce_log:
        receipt = reproduce_from_log(Path(args.reproduce_log))
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    artifact = run_experiment(budget=args.budget)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
