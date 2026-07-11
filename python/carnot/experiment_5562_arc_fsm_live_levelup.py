"""Exp5562: gated ARC FSM live level-up attempt.

Spec refs: REQ-ARC-FCP-5562, SCENARIO-ARC-FCP-5562.

This module is the credit-bearing follow-up to Exp5561. It rereads the ARC
registry immediately before runtime, prevents duplicate target re-solves, and
then runs the live E3 policy with Exp5561's finite-state candidate router. Solve
credit is deliberately narrow: only the live path's own attempts can produce a
candidate trajectory, and the offline reproduction gate is used only after that
trajectory exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from carnot.experiment_5534_arc_strategy_routed_levelup import (
    _action_label,
    _apply_action_label,
    _policy_salience_diagnostics,
    offline_arcade_available,
    trajectory_metrics,
)
from carnot.experiment_5561_arc_fsm_target_rotation_precheck import FSMActionAbstraction


JsonDict = dict[str, Any]

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5562
EXPERIMENT = "experiment_5562_arc_fsm_live_levelup"
MILESTONE = "2026.07.503"
RESULT_RELATIVE_PATH = "results/experiment_5562_arc_fsm_live_levelup.json"
TRAJECTORY_RELATIVE_PATH = "results/experiment_5562_arc_fsm_live_levelup_trajectory.json"
UPSTREAM_RELATIVE_PATH = "results/experiment_5561_arc_fsm_target_rotation_precheck.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5562", "SCENARIO-ARC-FCP-5562"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "arc_live_agent_self_discovery_no_llm"
DEFAULT_BUDGET = 48
DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5562_arc_fsm_live_levelup.py -q --no-cov",
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5562_arc_fsm_live_levelup.py -q -n0 -o addopts= && "
        ".venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/experiment_5562_arc_fsm_live_levelup.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "llm_invoked": "bare bool false proving the credited FSM live attempt did not invoke any LLM.",
    "no_model_specs_required": "bare bool true because this no-LLM live-agent substrate has no model invocation to name.",
    "upstream_arc_precheck": "path to the Exp5561 gate that selected the target and proved FSM live-path reachability before runtime.",
    "solve_provenance": "must equal live_agent_self_discovery so only the live runtime's own attempts and runtime reverse engineering receive solve credit.",
    "llm_strategy_proposer_used": "bare bool false proving no LLM strategy proposer text contributed to the credited solver path.",
    "random_seed": "deterministic seed used for target recheck, bounded live routing, trajectory logging, and checksum replay.",
    "reproducibility_checksum": "content-addressed hash over target, seed, trajectory metrics, duplicate gate, and banking result to catch silent drift.",
    "selected_game": "Exp5561-selected registry-safe game id rechecked immediately before the live attempt.",
    "selected_level": "Exp5561-selected adjacent unreproduced frontier level label, or the duplicate-prevented target label.",
    "attempts": "bare int count of live-agent runtime actions executed; zero means the duplicate or readiness gate prevented runtime.",
    "trajectory_path": "path to the detailed live trajectory or duplicate-prevented trajectory receipt.",
    "action_entropy": "Shannon entropy over executed live action/coordinate choices as a bare float.",
    "repeated_coordinate_rate": "fraction of executed coordinate actions that repeated an earlier executed coordinate.",
    "offline_reproduced": "true only when the live-discovered trajectory passes the post-solve offline reproduction gate.",
    "reproduced_levels": "integer new levels banked from the live-discovered trajectory; success requires at least one.",
    "registry_delta": "bare int registry total delta; nonzero only when the accepted reproduction gate passes after live discovery.",
    "arc_live_levelup_ready": "bare bool proving Exp5561, registry reread, live-reachability, no-LLM metadata, and duplicate gates allowed runtime.",
    "tests_added_or_reused": "list of focused tests covering the Exp5562 schema, duplicate prevention, live trajectory metrics, checksum, and banking gate.",
    "field_principles": "mapping of one-line principle annotations for each headline and gate field.",
    "inference_substrate": "must equal arc_live_agent_self_discovery_no_llm.",
    "honest_verdict": "one-line verdict starting complete:, honest_null:, duplicate_prevented:, or blocked:.",
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive parsing.
        return int(default)


def parse_level_label(value: Any) -> int:
    text = str(value or "").strip().upper()
    if text.startswith("L") and text[1:].isdigit():
        return int(text[1:])
    if text.isdigit():
        return int(text)
    return 0


def _level_label(level: int) -> str:
    return f"L{max(0, int(level))}"


def read_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> JsonDict:
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


def row_signature(row: Mapping[str, Any]) -> str:
    coord = _row_coordinate(row)
    if coord is not None:
        return f"A{_as_int(row.get('action'))}@{coord[0]},{coord[1]}"
    return f"A{_as_int(row.get('action'))}"


def _precheck_clean(precheck: Mapping[str, Any]) -> bool:
    return bool(
        precheck
        and precheck.get("arc_fsm_precheck_ready") is True
        and precheck.get("llm_invoked") is False
        and precheck.get("no_model_specs_required") is True
        and precheck.get("fsm_action_abstraction_ready") is True
        and precheck.get("repeated_coordinate_suppression_enabled") is True
        and precheck.get("solve_provenance") == SOLVE_PROVENANCE
        and precheck.get("inference_substrate") == "arc_live_path_precheck_no_llm"
    )


def _blocked_target(
    blocker: str,
    registry: Mapping[str, Any],
    *,
    game: str = "",
    selected_level: str = "",
    target_level: int = 0,
    prior: int = 0,
    precheck: Mapping[str, Any] | None = None,
    duplicate_prevented: bool = False,
) -> JsonDict:
    return {
        "blocked": True,
        "blocker": str(blocker),
        "duplicate_prevented": bool(duplicate_prevented),
        "selected_game": str(game or ""),
        "selected_level": str(selected_level or ""),
        "target_level": int(max(0, target_level)),
        "prior_levels_reproduced": int(max(0, prior)),
        "registry_before_levels": _registry_total(registry),
        "random_seed": _as_int((precheck or {}).get("random_seed"), EXPERIMENT_ID),
        "arc_live_levelup_ready": False,
        "fsm_action_abstraction_ready": bool((precheck or {}).get("fsm_action_abstraction_ready")),
        "selection_reason": str(blocker),
        "target_audit": {
            f"{game}:{_level_label(target_level)}": {
                "game": str(game or ""),
                "target_level": int(max(0, target_level)),
                "registry_depth": int(max(0, prior)),
                "already_reproduced": bool(duplicate_prevented),
                "decision": str(blocker),
            }
        }
        if game and target_level
        else {},
    }


def select_target_from_precheck(
    precheck: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> JsonDict:
    """REQ-ARC-FCP-5562: reuse Exp5561's target unless the registry banks it."""

    if not precheck:
        return _blocked_target("exp5561_precheck_missing", registry, precheck=precheck)
    game = str(precheck.get("selected_game") or "")
    selected_level = str(precheck.get("selected_level") or "")
    target_level = parse_level_label(selected_level)
    prior = _registry_depth(registry, game)
    if not _precheck_clean(precheck):
        return _blocked_target(
            "exp5561_precheck_not_ready",
            registry,
            game=game,
            selected_level=selected_level,
            target_level=target_level,
            prior=prior,
            precheck=precheck,
        )
    if not game:
        return _blocked_target(
            "exp5561_target_missing",
            registry,
            selected_level=selected_level,
            target_level=target_level,
            precheck=precheck,
        )
    if target_level <= 0:
        return _blocked_target(
            "exp5561_selected_level_malformed",
            registry,
            game=game,
            selected_level=selected_level,
            target_level=target_level,
            prior=prior,
            precheck=precheck,
        )
    if game not in _registry_rows(registry):
        return _blocked_target(
            "exp5561_target_missing_from_registry",
            registry,
            game=game,
            selected_level=_level_label(target_level),
            target_level=target_level,
            prior=0,
            precheck=precheck,
        )
    if prior >= target_level:
        return _blocked_target(
            "target_already_reproduced_duplicate_prevented",
            registry,
            game=game,
            selected_level=_level_label(target_level),
            target_level=target_level,
            prior=prior,
            precheck=precheck,
            duplicate_prevented=True,
        )
    if prior + 1 != target_level:
        return _blocked_target(
            "exp5561_target_not_adjacent_frontier",
            registry,
            game=game,
            selected_level=_level_label(target_level),
            target_level=target_level,
            prior=prior,
            precheck=precheck,
        )
    return {
        "blocked": False,
        "blocker": "",
        "duplicate_prevented": False,
        "selected_game": game,
        "selected_level": _level_label(target_level),
        "target_level": int(target_level),
        "prior_levels_reproduced": int(prior),
        "registry_before_levels": _registry_total(registry),
        "random_seed": _as_int(precheck.get("random_seed"), EXPERIMENT_ID),
        "arc_live_levelup_ready": True,
        "fsm_action_abstraction_ready": True,
        "selection_reason": "exp5561_fsm_target_survived_registry_reread",
        "target_audit": {
            f"{game}:{_level_label(target_level)}": {
                "game": game,
                "target_level": int(target_level),
                "registry_depth": int(prior),
                "already_reproduced": False,
                "decision": "selected",
            }
        },
    }


def blocked_attempt(reason: str) -> JsonDict:
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
        "reproduction_gate": {"reproduced": False, "reason": str(reason)},
        "verifier_feedback": {"reproduction_gate": {"reproduced": False, "reason": str(reason)}},
        "runtime_reverse_engineering": [],
        "solution_labels": [],
        "failure_mode": str(reason),
        "terminal_state": {"status": "blocked", "reason": str(reason), "max_level": 0},
        "offline_bfs_used": False,
        "game_source_read": False,
        "hand_built_per_game_adapter_used": False,
        "llm_strategy_proposer_used": False,
    }


class _NoOpProposer:  # pragma: no cover - ARC runtime boundary.
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "disabled_exp5562_no_llm"

    def world_model_candidates(self, _game: str) -> list[Any]:
        return []


def _dominant_fsm_phase(diagnostics: Mapping[str, Any]) -> str:  # pragma: no cover - ARC runtime boundary.
    counts = diagnostics.get("fsm_phase_counts")
    if isinstance(counts, Mapping) and counts:
        return max(counts.items(), key=lambda item: (_as_int(item[1]), str(item[0])))[0]
    return "unclassified_fsm_phase"


def run_fsm_live_attempt(  # pragma: no cover - ARC runtime boundary.
    *,
    root: Path,
    target: Mapping[str, Any],
    precheck: Mapping[str, Any],
    budget: int = DEFAULT_BUDGET,
    random_seed: int,
) -> JsonDict:
    """SCENARIO-ARC-FCP-5562: execute the live path with the FSM router."""

    del root, precheck
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of
    from carnot.experiment_5521_arc_live_action_diverse_levelup import (
        ActionDiverseLiveGenerator,
    )

    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    random.seed(int(random_seed))
    game = str(target["selected_game"])
    prior = _as_int(target.get("prior_levels_reproduced"))
    target_level = _as_int(target.get("target_level"))
    router = FSMActionAbstraction(max_candidates=8, suppress_repeated_coordinates=True)
    generator = ActionDiverseLiveGenerator(max_candidates=8)
    labels: list[str] = []
    frames: list[Any] = []
    observations: list[JsonDict] = []
    proposed_rows: list[JsonDict] = []
    action_rows: list[JsonDict] = []
    verifier_routes: list[JsonDict] = []
    suppression_events: list[JsonDict] = []
    level_changes: list[JsonDict] = []
    runtime_re: list[JsonDict] = []
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
            fsm_phase = _dominant_fsm_phase(router_diag)
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
                    "fsm_phase": fsm_phase,
                    "suppression_enabled": bool(router_diag.get("suppression_enabled")),
                }
            )
            runtime_re.append(
                {
                    "step": int(index),
                    "signal": "fsm_candidate_router_runtime_induction",
                    "fsm_phase": fsm_phase,
                    "suppressed_coordinate_count": int(suppressed),
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
                    "fsm_phase": fsm_phase,
                    "fsm_action": f"{fsm_phase}:{label}",
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
            "runtime_reverse_engineering": runtime_re,
            "solution_labels": list(labels) if reproduced else [],
            "failure_mode": "" if reproduced else "bounded_budget_no_target_level_reproduction",
            "terminal_state": {
                "status": "reproduced" if reproduced else "budget_exhausted",
                "max_level": int(max_level),
                "target_level": int(target_level),
            },
            "offline_bfs_used": False,
            "game_source_read": False,
            "hand_built_per_game_adapter_used": False,
            "llm_strategy_proposer_used": False,
        }
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def accepted_reproduced_levels(target: Mapping[str, Any], attempt: Mapping[str, Any]) -> int:
    if attempt.get("offline_reproduced") is not True:
        return 0
    if (
        attempt.get("offline_bfs_used", False)
        or attempt.get("game_source_read", False)
        or attempt.get("hand_built_per_game_adapter_used", False)
        or attempt.get("llm_strategy_proposer_used", False)
    ):
        return 0
    prior = _as_int(target.get("prior_levels_reproduced"))
    target_level = _as_int(target.get("target_level"))
    post = _as_int(attempt.get("post_levels_reproduced"), prior)
    if post <= prior or post < target_level:
        return 0
    return max(0, _as_int(attempt.get("reproduced_levels"), post - prior))


def compute_reproducibility_checksum(
    *,
    selected_game: str,
    selected_level: str,
    random_seed: int,
    attempts: int,
    action_entropy: float,
    repeated_coordinate_rate: float,
    offline_reproduced: bool,
    reproduced_levels: int,
    registry_delta: int,
    duplicate_prevented: bool,
    terminal_state: Mapping[str, Any],
    reproduction_gate: Mapping[str, Any],
) -> str:
    payload = {
        "action_entropy": float(action_entropy),
        "attempts": int(attempts),
        "duplicate_prevented": bool(duplicate_prevented),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "offline_reproduced": bool(offline_reproduced),
        "random_seed": int(random_seed),
        "registry_delta": int(registry_delta),
        "repeated_coordinate_rate": float(repeated_coordinate_rate),
        "reproduced_levels": int(reproduced_levels),
        "reproduction_gate": dict(reproduction_gate),
        "selected_game": str(selected_game),
        "selected_level": str(selected_level),
        "solve_provenance": SOLVE_PROVENANCE,
        "terminal_state": dict(terminal_state),
        "upstream_arc_precheck": UPSTREAM_RELATIVE_PATH,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_artifact(
    *,
    target: Mapping[str, Any],
    attempt: Mapping[str, Any],
    registry_updated: bool,
    trajectory_path: str,
    precheck: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
    attempt_budget: int,
) -> JsonDict:
    """SCENARIO-ARC-FCP-5562: build the provenance-clean live-attempt artifact."""

    action_rows = [row for row in attempt.get("action_rows") or [] if isinstance(row, Mapping)]
    proposed_rows = [
        row for row in attempt.get("proposed_action_rows") or [] if isinstance(row, Mapping)
    ]
    metrics = trajectory_metrics(action_rows, proposed_rows)
    blocked = bool(target.get("blocked")) or bool(attempt.get("blocked"))
    duplicate_prevented = bool(target.get("duplicate_prevented"))
    accepted_delta = accepted_reproduced_levels(target, attempt)
    can_bank = bool(accepted_delta >= 1 and registry_updated and not blocked)
    registry_delta = int(accepted_delta if can_bank else 0)
    selected_game = str(target.get("selected_game") or "")
    selected_level = str(target.get("selected_level") or "")
    attempts = _as_int(attempt.get("attempts"), len(action_rows))
    terminal_state = dict(
        attempt.get("terminal_state")
        or {
            "status": "reproduced" if can_bank else "budget_exhausted",
            "max_level": _as_int(attempt.get("post_levels_reproduced")),
            "target_level": _as_int(target.get("target_level")),
        }
    )
    reproduction_gate = dict(attempt.get("reproduction_gate") or {})
    status = (
        "complete"
        if can_bank
        else "duplicate_prevented"
        if duplicate_prevented
        else "blocked"
        if blocked
        else "honest_null"
    )
    reproduced_levels = int(accepted_delta if can_bank else 0)
    offline_reproduced = bool(can_bank)
    random_seed = _as_int(target.get("random_seed"), _as_int(precheck.get("random_seed"), EXPERIMENT_ID))
    checksum = compute_reproducibility_checksum(
        selected_game=selected_game,
        selected_level=selected_level,
        random_seed=random_seed,
        attempts=attempts,
        action_entropy=float(metrics["action_entropy"]),
        repeated_coordinate_rate=float(metrics["repeated_coordinate_rate"]),
        offline_reproduced=offline_reproduced,
        reproduced_levels=reproduced_levels,
        registry_delta=registry_delta,
        duplicate_prevented=duplicate_prevented,
        terminal_state=terminal_state,
        reproduction_gate=reproduction_gate,
    )
    failure_mode = str(
        attempt.get("failure_mode")
        or target.get("blocker")
        or "bounded_budget_no_target_level_reproduction"
    )
    before_total = _as_int(target.get("registry_before_levels"))
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "carnot.experiment_5562_arc_fsm_live_levelup.v1",
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "status": status,
        "llm_invoked": False,
        "no_model_specs_required": True,
        "upstream_arc_precheck": UPSTREAM_RELATIVE_PATH,
        "solve_provenance": SOLVE_PROVENANCE,
        "llm_strategy_proposer_used": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "selected_game": selected_game,
        "selected_level": selected_level,
        "attempts": int(attempts),
        "trajectory_path": str(trajectory_path),
        "action_entropy": float(metrics["action_entropy"]),
        "repeated_coordinate_rate": float(metrics["repeated_coordinate_rate"]),
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": reproduced_levels,
        "registry_delta": int(registry_delta),
        "arc_live_levelup_ready": bool(target.get("arc_live_levelup_ready") and not blocked),
        "tests_added_or_reused": list(tests_run or DEFAULT_TESTS_RUN),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"complete: {selected_game} {selected_level} FSM live trajectory reproduced and banked"
            if can_bank
            else f"duplicate_prevented: {selected_game} {selected_level} already in registry; runtime skipped"
            if duplicate_prevented
            else f"blocked: {failure_mode}"
            if status == "blocked"
            else (
                f"honest_null: {selected_game} {selected_level} {failure_mode}; "
                f"entropy={metrics['action_entropy']:.3f}; "
                f"repeat_rate={metrics['repeated_coordinate_rate']:.3f}; registry_delta=0"
            )
        ),
        "attempt_budget": int(attempt_budget),
        "terminal_state": terminal_state,
        "reproduction_gate": reproduction_gate,
        "exact_replay_result": {
            "offline_reproduced": offline_reproduced,
            "reproduced_levels": reproduced_levels,
            "registry_delta": int(registry_delta),
            "gate": reproduction_gate,
        },
        "duplicate_prevented": duplicate_prevented,
        "selected_target_level": _as_int(target.get("target_level")),
        "prior_levels_reproduced": _as_int(target.get("prior_levels_reproduced")),
        "post_levels_reproduced": (
            _as_int(attempt.get("post_levels_reproduced"))
            if can_bank
            else _as_int(target.get("prior_levels_reproduced"))
        ),
        "registry_before_levels": int(before_total),
        "registry_after_levels": int(before_total + registry_delta),
        "registry_updated": bool(registry_updated),
        "target_selection": dict(target),
        "fsm_action_abstraction_ready": bool(target.get("fsm_action_abstraction_ready")),
        "repeated_coordinate_suppression_enabled": bool(
            precheck.get("repeated_coordinate_suppression_enabled")
        ),
        "repeated_coordinate_suppression_events": sum(
            max(0, _as_int(event.get("suppressed_coordinate_count")))
            for event in attempt.get("suppression_events") or []
            if isinstance(event, Mapping)
        ),
        "offline_bfs_used": bool(attempt.get("offline_bfs_used", False)),
        "game_source_read": bool(attempt.get("game_source_read", False)),
        "hand_built_per_game_adapter_used": bool(
            attempt.get("hand_built_per_game_adapter_used", False)
        ),
        "runtime_reverse_engineering": list(attempt.get("runtime_reverse_engineering") or []),
        "input_artifacts": [UPSTREAM_RELATIVE_PATH, REGISTRY_RELATIVE_PATH],
        "preconditions_checked": dict(preconditions_checked),
        "duration_s": float(duration_s),
    }


def build_trajectory_log(
    *,
    target: Mapping[str, Any],
    attempt: Mapping[str, Any],
    artifact: Mapping[str, Any],
    precheck: Mapping[str, Any],
) -> JsonDict:
    """SCENARIO-ARC-FCP-5562: persist live trajectory evidence for audit."""

    return {
        "schema": "carnot.experiment_5562_arc_fsm_live_levelup.trajectory.v1",
        "experiment": EXPERIMENT,
        "selected_game": artifact.get("selected_game") or target.get("selected_game") or "",
        "selected_level": artifact.get("selected_level") or target.get("selected_level") or "",
        "random_seed": artifact.get("random_seed"),
        "attempt_budget": artifact.get("attempt_budget"),
        "target_selection": dict(target),
        "duplicate_prevented": bool(artifact.get("duplicate_prevented")),
        "observations": list(attempt.get("observations") or []),
        "proposed_actions": list(attempt.get("proposed_action_rows") or []),
        "executed_actions": list(attempt.get("action_rows") or []),
        "verifier_routes": list(attempt.get("verifier_routes") or []),
        "suppression_events": list(attempt.get("suppression_events") or []),
        "level_counter_changes": list(attempt.get("level_counter_changes") or []),
        "runtime_reverse_engineering": list(attempt.get("runtime_reverse_engineering") or []),
        "verifier_feedback": dict(attempt.get("verifier_feedback") or {}),
        "solution_labels": list(attempt.get("solution_labels") or []),
        "terminal_state": dict(artifact.get("terminal_state") or {}),
        "exact_replay_result": dict(artifact.get("exact_replay_result") or {}),
        "metrics": {
            "action_entropy": artifact.get("action_entropy"),
            "repeated_coordinate_rate": artifact.get("repeated_coordinate_rate"),
            "repeated_coordinate_suppression_events": artifact.get(
                "repeated_coordinate_suppression_events"
            ),
        },
        "prohibited_inputs": {
            "offline_bfs_used": bool(attempt.get("offline_bfs_used", False)),
            "game_source_read": bool(attempt.get("game_source_read", False)),
            "hand_built_per_game_adapter_used": bool(
                attempt.get("hand_built_per_game_adapter_used", False)
            ),
            "llm_strategy_proposer_used": bool(attempt.get("llm_strategy_proposer_used", False)),
        },
        "upstream_precheck": {
            "path": UPSTREAM_RELATIVE_PATH,
            "selected_game": precheck.get("selected_game"),
            "selected_level": precheck.get("selected_level"),
            "arc_fsm_precheck_ready": precheck.get("arc_fsm_precheck_ready"),
            "random_seed": precheck.get("random_seed"),
        },
    }


def write_trajectory_log(path: Path, log: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checksum_looks_valid(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in artifact
    ]
    if artifact.get("llm_invoked") is not False:
        errors.append("llm_invoked must be false")
    if artifact.get("no_model_specs_required") is not True:
        errors.append("no_model_specs_required must be true")
    if not isinstance(artifact.get("upstream_arc_precheck"), str) or not artifact.get(
        "upstream_arc_precheck"
    ):
        errors.append("upstream_arc_precheck must be a non-empty string")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    if artifact.get("llm_strategy_proposer_used") is not False:
        errors.append("llm_strategy_proposer_used must be false")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be an int")
    if not _checksum_looks_valid(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be a sha256 string")
    if not isinstance(artifact.get("selected_game"), str):
        errors.append("selected_game must be a string")
    if not isinstance(artifact.get("selected_level"), str):
        errors.append("selected_level must be a string")
    if type(artifact.get("attempts")) is not int:
        errors.append("attempts must be bare int")
    elif artifact["attempts"] < 0:
        errors.append("attempts must be non-negative")
    if not isinstance(artifact.get("trajectory_path"), str) or not artifact.get(
        "trajectory_path"
    ):
        errors.append("trajectory_path must be a non-empty string")
    if type(artifact.get("action_entropy")) is not float:
        errors.append("action_entropy must be bare float")
    if type(artifact.get("repeated_coordinate_rate")) is not float:
        errors.append("repeated_coordinate_rate must be bare float")
    elif not (0.0 <= artifact["repeated_coordinate_rate"] <= 1.0):
        errors.append("repeated_coordinate_rate must be in [0, 1]")
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
    if type(artifact.get("arc_live_levelup_ready")) is not bool:
        errors.append("arc_live_levelup_ready must be bare bool")
    tests = artifact.get("tests_added_or_reused")
    if not isinstance(tests, list) or not tests:
        errors.append("tests_added_or_reused must be a non-empty list")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be a mapping")
    elif any(field not in principles for field in REQUIRED_FIELDS):
        errors.append("field_principles must cover every required field")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be arc_live_agent_self_discovery_no_llm")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "honest_null:", "duplicate_prevented:", "blocked:")):
        errors.append(
            "honest_verdict must start with complete:, honest_null:, duplicate_prevented:, or blocked:"
        )
    if "model_specs" in artifact:
        errors.append("model_specs must be omitted for no-LLM substrate")
    if "target_model" in artifact:
        errors.append("target_model must be omitted for no-LLM substrate")
    for field in ("offline_bfs_used", "game_source_read", "hand_built_per_game_adapter_used"):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
        elif artifact.get(field) is not False:
            errors.append(f"{field} must be false")
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
    evidence = {
        "artifact": RESULT_RELATIVE_PATH,
        "trajectory_path": artifact.get("trajectory_path"),
        "offline_reproduced": True,
        "reproduced_levels": delta,
        "prior_levels_reproduced": _as_int(artifact.get("prior_levels_reproduced")),
        "post_levels_reproduced": post,
        "registry_delta": delta,
        "solve_provenance": SOLVE_PROVENANCE,
        "random_seed": _as_int(artifact.get("random_seed")),
        "reproducibility_checksum": artifact.get("reproducibility_checksum"),
    }
    found = False
    for row in games:
        if isinstance(row, dict) and row.get("game") == game:
            row["reproducibility"] = "reproduced"
            row["levels_reproduced"] = post
            row["latest_exp5562_fsm_live_levelup"] = evidence
            found = True
            break
    if not found:
        games.append(
            {
                "game": game,
                "reproducibility": "reproduced",
                "levels_reproduced": post,
                "latest_exp5562_fsm_live_levelup": evidence,
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
    attempt_runner: Callable[..., Mapping[str, Any]] = run_fsm_live_attempt,
    offline_arcade_check: Callable[[], bool] = offline_arcade_available,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    started = time.monotonic()
    root = Path(root)
    spec_path = root / SPEC_RELATIVE_PATH
    registry_path = root / REGISTRY_RELATIVE_PATH
    upstream_path = root / UPSTREAM_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    registry = read_yaml(registry_path)
    precheck = read_json(upstream_path)
    target = select_target_from_precheck(precheck, registry)
    preconditions = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "OPENCODE.md": (root / "OPENCODE.md").exists(),
        "CLAUDE.md": (root / "CLAUDE.md").exists(),
        "spec_has_req_5562": "REQ-ARC-FCP-5562" in spec_text,
        "registry_present": registry_path.exists(),
        "exp5561_precheck_present": upstream_path.exists(),
        "exp5561_ready": precheck.get("arc_fsm_precheck_ready") is True,
        "registry_target_duplicate_prevented": bool(target.get("duplicate_prevented")),
        "fsm_action_abstraction_ready": precheck.get("fsm_action_abstraction_ready") is True,
        "offline_arcade_available": False,
        "offline_bfs_used": False,
        "game_source_read": False,
        "hand_built_per_game_adapter_used": False,
        "llm_invoked": False,
        "llm_strategy_proposer_used": False,
        "model_specs_present": False,
    }
    ready_without_arcade = (
        preconditions["AGENTS.md"]
        and (preconditions["CODEX.md"] or preconditions["OPENCODE.md"])
        and preconditions["CLAUDE.md"]
        and preconditions["spec_has_req_5562"]
        and preconditions["registry_present"]
        and preconditions["exp5561_precheck_present"]
        and _precheck_clean(precheck)
        and not target.get("blocked")
        and target.get("arc_live_levelup_ready") is True
    )
    if not ready_without_arcade:
        attempt: Mapping[str, Any] = blocked_attempt(
            str(target.get("blocker") or "missing_exp5562_precondition")
        )
    else:
        preconditions["offline_arcade_available"] = bool(offline_arcade_check())
        if not preconditions["offline_arcade_available"]:
            attempt = blocked_attempt("missing_harness_access")
        else:
            attempt = attempt_runner(
                root=root,
                target=target,
                precheck=precheck,
                budget=budget,
                random_seed=_as_int(target.get("random_seed"), EXPERIMENT_ID),
            )

    trajectory_path = TRAJECTORY_RELATIVE_PATH
    registry_updated = False
    if accepted_reproduced_levels(target, attempt) >= 1:
        preliminary = build_artifact(
            target=target,
            attempt=attempt,
            registry_updated=True,
            trajectory_path=trajectory_path,
            precheck=precheck,
            preconditions_checked=preconditions,
            tests_run=tests_run,
            duration_s=time.monotonic() - started,
            attempt_budget=budget,
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
        precheck=precheck,
        preconditions_checked=preconditions,
        tests_run=tests_run,
        duration_s=time.monotonic() - started,
        attempt_budget=0 if target.get("duplicate_prevented") else budget,
    )
    validate_artifact(artifact)
    log = build_trajectory_log(
        target=target,
        attempt=attempt,
        artifact=artifact,
        precheck=precheck,
    )
    write_trajectory_log(root / trajectory_path, log)
    _write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
