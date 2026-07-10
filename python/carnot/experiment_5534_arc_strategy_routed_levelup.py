"""Exp5534: ARC strategy-routed live level-up attempt.

Spec refs: REQ-ARC-FCP-5534, SCENARIO-ARC-FCP-5534.

This module is the credit-bearing follow-up to Exp5533. It deliberately keeps a
narrow boundary: target selection comes only from the Exp5533 artifact, live
action selection installs the bounded strategy router with repeated-coordinate
suppression, and solve credit is granted only after the standard offline replay
gate reproduces the live-discovered trajectory.
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

from carnot.agentic.arc_bounded_strategy_router import BoundedStrategyCandidateRouter


JsonDict = dict[str, Any]

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5534
EXPERIMENT = "experiment_5534_arc_strategy_routed_levelup"
MILESTONE = "2026.07.501"
RESULT_RELATIVE_PATH = "results/experiment_5534_arc_strategy_routed_levelup.json"
TRAJECTORY_RELATIVE_PATH = "results/experiment_5534_arc_strategy_routed_levelup_trajectory.json"
EXP5533_RELATIVE_PATH = "results/experiment_5533_arc_strategy_routing_precheck.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5534", "SCENARIO-ARC-FCP-5534"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "arc_live_agent_self_discovery"
DEFAULT_BUDGET = 48
MIN_STRATEGY_COUNT = 3
MODEL_SPECS = [
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
]
DEFAULT_TESTS_RUN = [
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_5534_arc_strategy_routed_levelup.py "
        "-q --no-cov"
    ),
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5534_arc_strategy_routed_levelup.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/experiment_5534_arc_strategy_routed_levelup.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "selected_game": "Exp5533-selected registry-safe game id; empty only when Exp5533 blocks before runtime.",
    "selected_level": "Exp5533-selected adjacent frontier level label; duplicate targets block rather than rotate.",
    "solve_provenance": "must equal live_agent_self_discovery.",
    "strategy_portfolio_used": "bounded live-path-compatible strategy descriptors actually installed on the candidate router.",
    "strategy_switch_count": "integer count of changes between executed strategy labels after repeated-coordinate suppression.",
    "attempts": "bare int count of runtime actions executed by the live agent.",
    "action_entropy": "Shannon entropy over executed live action/coordinate choices as a bare float.",
    "repeated_coordinate_rate": "fraction of executed live coordinate choices that repeated an earlier executed coordinate.",
    "repeated_coordinate_suppression_events": "bare int count of candidate-router repeated-coordinate suppressions recorded during live selection.",
    "salience_coverage_rate": "fraction of executed live coordinate choices covering proposed salience coordinates.",
    "offline_reproduced": "true only when the live-discovered trajectory passes the standard offline replay gate.",
    "reproduced_levels": "integer new levels banked from the live-discovered trajectory; success requires >=1.",
    "registry_delta": "bare int registry total delta; nonzero only when the live reproduction gate passes.",
    "trajectory_path": "path to the detailed trajectory log containing attempts, strategies, verifier routes, suppression events, and level changes.",
    "model_specs": "allowed local-GGUF proposer specs recorded for audit; no model is invoked when llm_strategy_proposer_used=false.",
    "llm_strategy_proposer_used": "bare bool; false means deterministic strategy templates were used and no GGUF tokenizer/model path was loaded.",
    "arc_live_levelup_ready": "bare bool proving Exp5533 and registry reread allowed live runtime.",
    "tests_added_or_reused": "list of focused tests that cover the Exp5534 schema, duplicate block, live routing trace, and registry gate.",
    "inference_substrate": "must equal arc_live_agent_self_discovery.",
    "honest_verdict": "one-line verdict starting complete:, honest_null:, or blocked:.",
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _parse_level_label(value: Any) -> int:
    text = str(value or "").strip().upper()
    if text.startswith("L") and text[1:].isdigit():
        return int(text[1:])
    if text.isdigit():
        return int(text)
    return 0


def _level_label(level: int) -> str:
    return f"L{max(0, int(level))}"


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


def _default_strategy_portfolio() -> list[JsonDict]:
    return BoundedStrategyCandidateRouter().portfolio_descriptors()


def _strategy_portfolio_from_exp5533(exp5533: Mapping[str, Any]) -> list[JsonDict]:
    rows = exp5533.get("strategy_portfolio")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        portfolio = [dict(row) for row in rows if isinstance(row, Mapping)]
        if portfolio:
            return portfolio
    return _default_strategy_portfolio()


def select_target_from_exp5533(
    exp5533: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> JsonDict:
    """REQ-ARC-FCP-5534: use Exp5533's target and block duplicates."""

    total = _registry_total(registry)
    game = str(exp5533.get("selected_game") or "")
    selected_level = str(exp5533.get("selected_level") or "")
    target_level = _parse_level_label(selected_level)
    portfolio = _strategy_portfolio_from_exp5533(exp5533)
    if not exp5533:
        return _blocked_target("exp5533_target_missing", total, "", "", 0, 0, portfolio)
    if exp5533.get("arc_sge_candidate_ready") is not True:
        return _blocked_target(
            "exp5533_candidate_not_ready",
            total,
            game,
            selected_level,
            target_level,
            0,
            portfolio,
        )
    if exp5533.get("solve_provenance") != SOLVE_PROVENANCE:
        return _blocked_target(
            "exp5533_wrong_provenance",
            total,
            game,
            selected_level,
            target_level,
            0,
            portfolio,
        )
    if not game:
        return _blocked_target("exp5533_target_missing", total, game, selected_level, target_level, 0, portfolio)
    if target_level <= 0:
        return _blocked_target(
            "exp5533_selected_level_malformed",
            total,
            game,
            selected_level,
            target_level,
            0,
            portfolio,
        )
    prior = _registry_depth(registry, game)
    if prior >= target_level:
        return _blocked_target(
            "blocked_duplicate_target",
            total,
            game,
            _level_label(target_level),
            target_level,
            prior,
            portfolio,
        )
    return {
        "blocked": False,
        "selected_game": game,
        "selected_level": _level_label(target_level),
        "target_level": int(target_level),
        "prior_levels_reproduced": int(prior),
        "registry_before_levels": int(total),
        "selection_reason": "exp5533_strategy_routed_target_survived_registry_reread",
        "strategy_portfolio": portfolio,
    }


def _blocked_target(
    blocker: str,
    total: int,
    game: str,
    selected_level: Any,
    target_level: int,
    prior: int,
    portfolio: Sequence[Mapping[str, Any]],
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
        "strategy_portfolio": [dict(row) for row in portfolio],
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
        return False, "disabled_exp5534_no_llm"

    def world_model_candidates(self, _game: str) -> list[Any]:
        return []


def _policy_salience_diagnostics(policy: Any) -> JsonDict:  # pragma: no cover - ARC runtime boundary
    if hasattr(policy, "action_salience_diagnostics"):
        return dict(policy.action_salience_diagnostics())
    explorer = getattr(policy, "explorer", None)
    if explorer is not None and hasattr(explorer, "action_salience_diagnostics"):
        return dict(explorer.action_salience_diagnostics())
    return {"action_tier_rows": []}


def _action_label(action: int, data: Any) -> str:  # pragma: no cover - ARC runtime boundary
    from carnot.experiment_5521_arc_live_action_diverse_levelup import _action_label as labeler

    return labeler(action, data)


def _apply_action_label(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - ARC runtime boundary
    from carnot.experiment_5521_arc_live_action_diverse_levelup import (
        _apply_action_label as applier,
    )

    return applier(*args, **kwargs)


def run_live_strategy_routed_attempt(  # pragma: no cover - ARC runtime boundary
    *,
    root: Path,
    target: Mapping[str, Any],
    exp5533: Mapping[str, Any],
    strategy_portfolio: Sequence[Mapping[str, Any]],
    budget: int = DEFAULT_BUDGET,
) -> JsonDict:
    del exp5533
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of
    from carnot.experiment_5521_arc_live_action_diverse_levelup import (
        ActionDiverseLiveGenerator,
    )

    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    game = str(target["selected_game"])
    prior = _as_int(target.get("prior_levels_reproduced"))
    target_level = _as_int(target.get("target_level"))
    router = BoundedStrategyCandidateRouter(
        strategies=strategy_portfolio,
        max_candidates=8,
        per_strategy_limit=1,
        suppress_repeated_coordinates=True,
    )
    generator = ActionDiverseLiveGenerator(max_candidates=8)
    labels: list[str] = []
    frames: list[Any] = []
    observations: list[JsonDict] = []
    proposed_rows: list[JsonDict] = []
    action_rows: list[JsonDict] = []
    verifier_routes: list[JsonDict] = []
    suppression_events: list[JsonDict] = []
    level_changes: list[JsonDict] = []
    latest = None
    max_level = prior
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
            candidate_router=router,
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
            before_level = _as_int(_level_of(latest), default=max_level) if latest is not None else max_level
            kind, data = policy.next_move(frames, latest)
            router_diag = dict(router.last_diagnostics)
            strategies_used = list(router_diag.get("strategies_used") or [])
            selected_strategy = next(
                (name for name in strategies_used if name != "fallback_fill"),
                "explorer_pending_or_unrouted",
            )
            salience = _policy_salience_diagnostics(policy)
            for row in salience.get("action_tier_rows") or []:
                if isinstance(row, Mapping):
                    proposed_rows.append({"step": int(index), **dict(row)})
            suppressed = _as_int(router_diag.get("suppressed_coordinate_count"))
            if suppressed > 0:
                suppression_events.append(
                    {
                        "step": int(index),
                        "suppressed_coordinate_count": int(suppressed),
                        "strategies_used": strategies_used,
                        "selected_signatures": list(router_diag.get("selected_signatures") or []),
                        "unsuppressed_signatures": list(
                            router_diag.get("unsuppressed_signatures") or []
                        ),
                    }
                )
            verifier_routes.append(
                {
                    "step": int(index),
                    "route": "candidate_router.rank",
                    "strategy": selected_strategy,
                    "phase": str(getattr(policy, "phase", "explore")),
                    "suppression_enabled": bool(router_diag.get("suppression_enabled")),
                }
            )
            if kind == "RESET":
                latest = env.reset()
                after_level = _as_int(_level_of(latest), default=max_level)
                observations.append({"step": int(index), "event": "reset", "level": after_level})
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
                    "strategy": selected_strategy,
                    "verifier_route": "candidate_router.rank",
                    "level_before": int(before_level),
                    "level_after": int(after_level),
                }
                action_rows.append(row)
                observations.append({"step": int(index), "event": "action", **row})
            if after_level != before_level:
                level_changes.append(
                    {
                        "step": int(index),
                        "level_before": int(before_level),
                        "level_after": int(after_level),
                    }
                )
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
        return {
            "attempts": int(len(action_rows)),
            "post_levels_reproduced": int(reached if reproduced else prior),
            "offline_reproduced": bool(reproduced),
            "reproduced_levels": max(0, int(reached) - int(prior)) if reproduced else 0,
            "action_rows": action_rows,
            "proposed_action_rows": proposed_rows,
            "observations": observations,
            "verifier_routes": verifier_routes,
            "suppression_events": suppression_events,
            "level_counter_changes": level_changes,
            "verifier_feedback": {"reproduction_gate": gate},
            "reproduction_gate": gate,
            "solution_labels": list(labels) if reproduced else [],
            "failure_mode": "" if reproduced else "bounded_budget_no_target_level_reproduction",
            "offline_bfs_used": False,
            "game_source_read": False,
            "hand_built_per_game_adapter_used": False,
            "methodology_receipt": (
                f"bounded_live_runtime budget={int(budget)} "
                "mechanism=strategy_routed_live_agent "
                "gate=standard_reproduction prohibited_inputs=false"
            ),
        }
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def _blocked_attempt(reason: str) -> JsonDict:
    return {
        "blocked": True,
        "attempts": 0,
        "post_levels_reproduced": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "action_rows": [],
        "proposed_action_rows": [],
        "observations": [{"event": "blocked", "reason": str(reason)}],
        "verifier_routes": [],
        "suppression_events": [],
        "level_counter_changes": [],
        "verifier_feedback": {"reproduction_gate": {"reproduced": False, "reason": str(reason)}},
        "reproduction_gate": {"reproduced": False, "reason": str(reason)},
        "solution_labels": [],
        "failure_mode": str(reason),
        "offline_bfs_used": False,
        "game_source_read": False,
        "hand_built_per_game_adapter_used": False,
        "methodology_receipt": (
            f"blocked_before_live_runtime reason={reason} "
            "mechanism=strategy_routed_live_agent "
            "gate=standard_reproduction prohibited_inputs=false"
        ),
    }


def trajectory_metrics(
    action_rows: Sequence[Mapping[str, Any]],
    proposed_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """SCENARIO-ARC-FCP-5534: measure live action diversity from executed rows."""

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


def _strategy_switch_count(action_rows: Sequence[Mapping[str, Any]]) -> int:
    labels = [
        str(row.get("strategy") or "")
        for row in action_rows
        if isinstance(row, Mapping) and row.get("strategy")
    ]
    return sum(1 for left, right in zip(labels, labels[1:]) if left != right)


def _suppression_event_count(attempt: Mapping[str, Any]) -> int:
    total = 0
    for event in attempt.get("suppression_events") or []:
        if isinstance(event, Mapping):
            total += max(0, _as_int(event.get("suppressed_coordinate_count")))
    return total


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


def build_artifact(
    *,
    target: Mapping[str, Any],
    attempt: Mapping[str, Any],
    registry_updated: bool,
    trajectory_path: str,
    exp5533: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
) -> JsonDict:
    """SCENARIO-ARC-FCP-5534: build the required live-attempt artifact."""

    action_rows = [row for row in attempt.get("action_rows") or [] if isinstance(row, Mapping)]
    proposed_rows = [
        row for row in attempt.get("proposed_action_rows") or [] if isinstance(row, Mapping)
    ]
    metrics = trajectory_metrics(action_rows, proposed_rows)
    blocked = bool(target.get("blocked")) or bool(attempt.get("blocked"))
    accepted_delta = _accepted_reproduced_levels(target, attempt)
    can_bank = bool(accepted_delta >= 1 and registry_updated and not blocked)
    before_total = _as_int(target.get("registry_before_levels"))
    registry_delta = int(accepted_delta if can_bank else 0)
    failure_mode = str(
        attempt.get("failure_mode")
        or target.get("blocker")
        or "bounded_budget_no_target_level_reproduction"
    )
    status = "complete" if can_bank else "blocked" if blocked else "honest_null"
    selected_game = str(target.get("selected_game") or "")
    selected_level = str(target.get("selected_level") or "")
    portfolio = _strategy_portfolio_from_exp5533(exp5533)
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "carnot.experiment_5534_arc_strategy_routed_levelup.v1",
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "selected_game": selected_game,
        "selected_level": selected_level,
        "solve_provenance": SOLVE_PROVENANCE,
        "strategy_portfolio_used": portfolio,
        "strategy_switch_count": int(_strategy_switch_count(action_rows)),
        "attempts": _as_int(attempt.get("attempts"), len(action_rows)),
        "action_entropy": float(metrics["action_entropy"]),
        "repeated_coordinate_rate": float(metrics["repeated_coordinate_rate"]),
        "repeated_coordinate_suppression_events": int(_suppression_event_count(attempt)),
        "salience_coverage_rate": float(metrics["salience_coverage_rate"]),
        "offline_reproduced": bool(can_bank),
        "reproduced_levels": int(accepted_delta if can_bank else 0),
        "registry_delta": int(registry_delta),
        "trajectory_path": str(trajectory_path),
        "model_specs": list(MODEL_SPECS),
        "llm_strategy_proposer_used": False,
        "arc_live_levelup_ready": bool(not blocked),
        "tests_added_or_reused": list(tests_run or DEFAULT_TESTS_RUN),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"complete: {selected_game} {selected_level} strategy-routed live trajectory reproduced and banked"
            if can_bank
            else f"blocked: {failure_mode}"
            if status == "blocked"
            else (
                f"honest_null: {selected_game} {selected_level} {failure_mode}; "
                f"entropy={metrics['action_entropy']:.3f}; "
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
            "action_count": len(action_rows),
            "proposed_action_count": len(proposed_rows),
            "strategy_switch_count": int(_strategy_switch_count(action_rows)),
            "suppression_events": int(_suppression_event_count(attempt)),
            "reproduction_gate": dict(attempt.get("reproduction_gate") or {}),
        },
        "preconditions_checked": dict(preconditions_checked),
        "duration_s": float(duration_s),
    }


def build_trajectory_log(
    *,
    target: Mapping[str, Any],
    attempt: Mapping[str, Any],
    artifact: Mapping[str, Any],
    exp5533: Mapping[str, Any],
) -> JsonDict:
    """SCENARIO-ARC-FCP-5534: keep enough trajectory detail for audit."""

    return {
        "schema": "carnot.experiment_5534_arc_strategy_routed_levelup.trajectory.v1",
        "experiment": EXPERIMENT,
        "selected_game": artifact.get("selected_game") or target.get("selected_game") or "",
        "selected_level": artifact.get("selected_level") or target.get("selected_level") or "",
        "target_selection": dict(target),
        "strategy_portfolio_used": list(artifact.get("strategy_portfolio_used") or []),
        "observations": list(attempt.get("observations") or []),
        "proposed_actions": list(attempt.get("proposed_action_rows") or []),
        "executed_actions": list(attempt.get("action_rows") or []),
        "strategy_choices": [
            {
                "step": row.get("step"),
                "strategy": row.get("strategy"),
                "action": row.get("action"),
                "data": row.get("data"),
            }
            for row in attempt.get("action_rows") or []
            if isinstance(row, Mapping)
        ],
        "verifier_routes": list(attempt.get("verifier_routes") or []),
        "suppression_events": list(attempt.get("suppression_events") or []),
        "level_counter_changes": list(attempt.get("level_counter_changes") or []),
        "verifier_feedback": dict(attempt.get("verifier_feedback") or {}),
        "solution_labels": list(attempt.get("solution_labels") or []),
        "metrics": {
            "action_entropy": artifact.get("action_entropy"),
            "repeated_coordinate_rate": artifact.get("repeated_coordinate_rate"),
            "repeated_coordinate_suppression_events": artifact.get(
                "repeated_coordinate_suppression_events"
            ),
            "salience_coverage_rate": artifact.get("salience_coverage_rate"),
            "strategy_switch_count": artifact.get("strategy_switch_count"),
        },
        "exp5533_source": {
            "selected_game": exp5533.get("selected_game"),
            "selected_level": exp5533.get("selected_level"),
            "arc_sge_candidate_ready": exp5533.get("arc_sge_candidate_ready"),
            "strategy_probe": exp5533.get("strategy_probe"),
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
    if not isinstance(artifact.get("selected_level"), str):
        errors.append("selected_level must be a string")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    portfolio = artifact.get("strategy_portfolio_used")
    if not isinstance(portfolio, list) or len(portfolio) < MIN_STRATEGY_COUNT:
        errors.append("strategy_portfolio_used must contain at least three strategies")
    for field in ("strategy_switch_count", "attempts", "repeated_coordinate_suppression_events"):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
        elif artifact[field] < 0:
            errors.append(f"{field} must be non-negative")
    for field in ("action_entropy", "repeated_coordinate_rate", "salience_coverage_rate"):
        if type(artifact.get(field)) is not float:
            errors.append(f"{field} must be bare float")
    for field in ("repeated_coordinate_rate", "salience_coverage_rate"):
        if type(artifact.get(field)) in (float, int) and not (0.0 <= float(artifact[field]) <= 1.0):
            errors.append(f"{field} must be in [0, 1]")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    elif artifact["reproduced_levels"] < 0:
        errors.append("reproduced_levels must be non-negative")
    if type(artifact.get("registry_delta")) is not int:
        errors.append("registry_delta must be bare int")
    elif artifact["registry_delta"] < 0:
        errors.append("registry_delta must be non-negative")
    if artifact.get("offline_reproduced") is True:
        if _as_int(artifact.get("reproduced_levels")) < 1:
            errors.append("offline_reproduced requires reproduced_levels >= 1")
        if _as_int(artifact.get("registry_delta")) != _as_int(artifact.get("reproduced_levels")):
            errors.append("offline_reproduced true requires registry_delta == reproduced_levels")
    if not isinstance(artifact.get("trajectory_path"), str) or not artifact.get("trajectory_path"):
        errors.append("trajectory_path must be a non-empty string")
    if not isinstance(artifact.get("model_specs"), list):
        errors.append("model_specs must be a list")
    if type(artifact.get("llm_strategy_proposer_used")) is not bool:
        errors.append("llm_strategy_proposer_used must be bare bool")
    if type(artifact.get("arc_live_levelup_ready")) is not bool:
        errors.append("arc_live_levelup_ready must be bare bool")
    tests = artifact.get("tests_added_or_reused")
    if not isinstance(tests, list) or not tests:
        errors.append("tests_added_or_reused must be a non-empty list")
    if not isinstance(artifact.get("field_principles"), Mapping):
        errors.append("field_principles must be a mapping")
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
    if artifact.get("registry_updated") is True and artifact.get("offline_reproduced") is not True:
        errors.append("registry_updated requires offline_reproduced true")
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
    if artifact.get("offline_reproduced") is not True or _as_int(artifact.get("registry_delta")) <= 0:
        return False
    updated = dict(registry)
    games = list(updated.get("games") or [])
    game = str(artifact["selected_game"])
    post = _as_int(artifact.get("post_levels_reproduced"))
    delta = _as_int(artifact.get("reproduced_levels"))
    found = False
    evidence = {
        "artifact": RESULT_RELATIVE_PATH,
        "trajectory_path": artifact.get("trajectory_path"),
        "offline_reproduced": True,
        "reproduced_levels": delta,
        "prior_levels_reproduced": _as_int(artifact.get("prior_levels_reproduced")),
        "post_levels_reproduced": post,
        "registry_delta": delta,
        "solve_provenance": SOLVE_PROVENANCE,
        "strategy_switch_count": _as_int(artifact.get("strategy_switch_count")),
        "repeated_coordinate_suppression_events": _as_int(
            artifact.get("repeated_coordinate_suppression_events")
        ),
    }
    for row in games:
        if isinstance(row, dict) and row.get("game") == game:
            row["reproducibility"] = "reproduced"
            row["levels_reproduced"] = post
            row["latest_exp5534_strategy_routed_levelup"] = evidence
            found = True
            break
    if not found:
        games.append(
            {
                "game": game,
                "reproducibility": "reproduced",
                "levels_reproduced": post,
                "latest_exp5534_strategy_routed_levelup": evidence,
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
    attempt_runner: Callable[..., Mapping[str, Any]] = run_live_strategy_routed_attempt,
    offline_arcade_check: Callable[[], bool] = offline_arcade_available,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    started = time.monotonic()
    root = Path(root)
    spec_path = root / SPEC_RELATIVE_PATH
    registry_path = root / REGISTRY_RELATIVE_PATH
    exp5533_path = root / EXP5533_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    registry = _read_yaml(registry_path)
    exp5533 = _read_json(exp5533_path)
    target = select_target_from_exp5533(exp5533, registry)
    preconditions = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "OPENCODE.md": (root / "OPENCODE.md").exists(),
        "CLAUDE.md": (root / "CLAUDE.md").exists(),
        "spec_has_req_5534": "REQ-ARC-FCP-5534" in spec_text,
        "registry_present": registry_path.exists(),
        "exp5533_precheck_present": exp5533_path.exists(),
        "exp5533_ready": exp5533.get("arc_sge_candidate_ready") is True,
        "offline_arcade_available": False,
        "offline_bfs_used": False,
        "game_source_read": False,
        "hand_built_per_game_adapter_used": False,
        "llm_strategy_proposer_used": False,
    }
    ready_without_arcade = (
        preconditions["AGENTS.md"]
        and (preconditions["CODEX.md"] or preconditions["OPENCODE.md"])
        and preconditions["CLAUDE.md"]
        and preconditions["spec_has_req_5534"]
        and preconditions["registry_present"]
        and preconditions["exp5533_precheck_present"]
        and not target.get("blocked")
    )
    if not ready_without_arcade:
        attempt: Mapping[str, Any] = _blocked_attempt(
            str(target.get("blocker") or "missing_exp5534_precondition")
        )
    else:
        preconditions["offline_arcade_available"] = bool(offline_arcade_check())
        if not preconditions["offline_arcade_available"]:
            attempt = _blocked_attempt("missing_harness_access")
        else:
            attempt = attempt_runner(
                root=root,
                target=target,
                exp5533=exp5533,
                strategy_portfolio=target.get("strategy_portfolio") or _default_strategy_portfolio(),
                budget=budget,
            )

    trajectory_path = TRAJECTORY_RELATIVE_PATH
    registry_updated = False
    if _accepted_reproduced_levels(target, attempt) >= 1:
        preliminary = build_artifact(
            target=target,
            attempt=attempt,
            registry_updated=True,
            trajectory_path=trajectory_path,
            exp5533=exp5533,
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
        trajectory_path=trajectory_path,
        exp5533=exp5533,
        preconditions_checked=preconditions,
        tests_run=tests_run,
        duration_s=time.monotonic() - started,
    )
    validate_artifact(artifact)
    log = build_trajectory_log(
        target=target,
        attempt=attempt,
        artifact=artifact,
        exp5533=exp5533,
    )
    write_trajectory_log(root / trajectory_path, log)
    _write_artifact(root, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    args = parser.parse_args(argv)
    artifact = run_experiment(budget=args.budget)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
