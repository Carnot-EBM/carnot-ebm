"""Bounded live-LLM re-induction helper for ARC level-up episodes.

Spec refs: REQ-ARC-WMTE-4544, SCENARIO-ARC-WMTE-4544,
REQ-ARC-WMTE-4557, SCENARIO-ARC-WMTE-4557-POSITIVE-CONTROL-FIRST,
REQ-ARC-WMTE-4664, SCENARIO-ARC-WMTE-4664-GOAL-SATISFIABILITY,
REQ-ARC-WMTE-4676, SCENARIO-ARC-WMTE-4676-HIERARCHICAL-PLAN,
REQ-ARC-WMTE-4677, SCENARIO-ARC-WMTE-4677-PRODUCT-PLANNING,
REQ-ARC-WMTE-4712, SCENARIO-ARC-WMTE-4712-LIVE-REINDUCTION-WIRING.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from carnot.agentic.arc_executable_world_model import (
    Transition,
    WorldModelVerifier,
    score_goal_predicate_consistency,
)
from carnot.agentic.arc_world_model_trust_energy import (
    WorldModelCandidate,
    select_trusted_world_model,
)


MAX_REFINEMENT_ROUNDS = 3


@dataclass
class SubgoalCandidate:
    """REQ-ARC-WMTE-4676: one oracle-distinct candidate predicate for a local leg."""

    name: str
    predicate: Callable[[np.ndarray], bool]
    source: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchicalSubgoalPlanResult:
    """REQ-ARC-WMTE-4676: planned low-level legs plus mechanism diagnostics."""

    planned: bool
    plan: list[dict[str, Any]] = field(default_factory=list)
    subgoal_decomposition: list[dict[str, Any]] = field(default_factory=list)
    per_subgoal_reachable: list[dict[str, Any]] = field(default_factory=list)
    final_grid: np.ndarray | None = None
    residual: str = ""


@dataclass
class LlmReinductionResult:
    """Result of the GOAL+DYNAMICS proposer plus verifier-guided refinement loop."""

    planned: bool
    plan: list[dict[str, Any]] = field(default_factory=list)
    goal_predicate: Callable[[np.ndarray], bool] | None = None
    engine: Callable[[np.ndarray, int, Any], np.ndarray] | None = None
    selected_candidate_name: str = ""
    goal_candidate_names: list[str] = field(default_factory=list)
    dynamics_candidate_names: list[str] = field(default_factory=list)
    refinement_rounds_used: int = 0
    verifier_is_oracle: bool = False
    model_specs: str = ""
    heldout_accuracy: float | None = None
    accepted_by_heldout_verifier: bool = False
    goal_predicate_satisfiable: bool = False
    goal_satisfiability: dict[str, Any] = field(default_factory=dict)
    goal_expression: str = ""
    structural_goal_diagnostics: dict[str, Any] = field(default_factory=dict)
    subgoal_decomposition: list[dict[str, Any]] = field(default_factory=list)
    per_subgoal_reachable: list[dict[str, Any]] = field(default_factory=list)
    subgoal_search_used: bool = False
    factored_planner_used: bool = False
    expert_trust_weights: list[dict[str, Any]] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    counterexamples: list[dict[str, Any]] = field(default_factory=list)
    skipped: str = ""
    # REQ-ARC-WMTE-6035: what best-engine retention actually did on this call (which round
    # was retained, whether the on-disk store had to be rolled back to it, and why not if
    # not). Diagnostic only -- nothing branches on it -- but without it a reader cannot tell
    # a call where retention was a no-op from one where it was silently disabled.
    engine_retention: dict[str, Any] = field(default_factory=dict)


def _model_specs(proposer: Any) -> str:
    value = getattr(proposer, "model_specs", None)
    if value:
        return str(value)
    repo = getattr(proposer, "repo_substr", "")
    path = getattr(proposer, "model_path", None)
    label = repo or proposer.__class__.__name__
    return f"{label} GGUF ({path})" if path else str(label)


def _normalise_structural_goal_candidate(
    provider: Callable[[np.ndarray], Any] | None,
    start_grid: np.ndarray,
) -> dict[str, Any] | None:
    """REQ-ARC-WMTE-4712: normalize an optional perception-grounded goal candidate."""

    if provider is None:
        return None
    try:
        row = provider(np.asarray(start_grid))
    except Exception as exc:
        return {
            "error": repr(exc)[:160],
            "predicate": None,
            "goal_expression": "",
            "diagnostics": {},
        }
    if row is None:
        return None
    if callable(row):
        return {
            "predicate": row,
            "goal_expression": getattr(row, "__name__", "structural_goal"),
            "diagnostics": {},
        }
    if isinstance(row, Mapping):
        predicate = row.get("predicate")
        if not callable(predicate):
            return None
        return {
            "predicate": predicate,
            "goal_expression": str(
                row.get("goal_expression") or row.get("name") or "structural_goal"
            ),
            "diagnostics": dict(row.get("diagnostics") or {}),
        }
    return None


def _normalise_candidates(rows: Sequence[Any], engine: Any, goal: Any) -> list[WorldModelCandidate]:
    candidates: list[WorldModelCandidate] = []
    source = list(rows) if rows else [("loaded_world_model.py", engine, goal)]
    for index, row in enumerate(source):
        if isinstance(row, WorldModelCandidate):
            candidates.append(row)
        elif isinstance(row, Mapping):
            candidates.append(
                WorldModelCandidate(
                    str(row.get("name") or f"candidate_{index}"),
                    row.get("engine", engine),
                    row.get("is_level_complete") or row.get("goal_predicate") or goal,
                )
            )
        else:
            name, candidate_engine, *rest = row
            candidates.append(
                WorldModelCandidate(
                    str(name),
                    candidate_engine,
                    rest[0] if rest else goal,
                )
            )
    return candidates


def _counterexample_result(counterexample: Mapping[str, Any]) -> Any:
    """REQ-ARC-WMTE-4544: build the VerifyResult-shaped object passed to proposer.refactor().

    When real per-transition mismatch evidence (BEFORE/PREDICTED/OBSERVED deltas from
    WorldModelVerifier.score(), the same shape refactor_prompt() is built to consume) was
    computed by the caller, use it -- the LLM refactor step needs concrete failing cases to
    fix, not just a scalar heldout_accuracy summary. Falls back to the scalar-only summary
    (as its own single mismatch entry) when no real evidence is available, so this remains
    safe for any caller that has not (yet) attached real evidence.
    """
    real_mismatches = counterexample.get("real_mismatches")
    if real_mismatches:
        return SimpleNamespace(
            n=int(counterexample.get("real_n") or len(real_mismatches)),
            n_correct=int(counterexample.get("real_n_correct") or 0),
            accuracy=float(counterexample.get("real_accuracy") or 0.0),
            mismatches=list(real_mismatches),
        )
    return SimpleNamespace(n=1, n_correct=0, accuracy=0.0, mismatches=[dict(counterexample)])


def _supports_kwarg(fn: Any, name: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return any(
        param.kind is inspect.Parameter.VAR_KEYWORD or param.name == name
        for param in signature.parameters.values()
    )


def _call_induce(
    proposer: Any,
    game: str,
    transitions: Sequence[Any],
    cell: int,
    previous_level_complete_grid: np.ndarray | None,
) -> tuple[bool, str]:
    if previous_level_complete_grid is not None and _supports_kwarg(
        proposer.induce,
        "previous_level_complete_grid",
    ):
        return proposer.induce(
            game,
            list(transitions),
            int(cell),
            previous_level_complete_grid=np.asarray(previous_level_complete_grid),
        )
    return proposer.induce(game, list(transitions), int(cell))


def _proposal_prefix(transitions: Sequence[Any]) -> list[Any]:
    """REQ-ARC-WMTE-4557: keep a held-out suffix out of the proposer prompt."""

    rows = list(transitions)
    if len(rows) < 2:
        return rows
    n_heldout = max(1, int(round(len(rows) / 3.0)))
    n_heldout = min(n_heldout, len(rows) - 1)
    return rows[:-n_heldout]


def _normalise_subgoal_candidates(rows: Sequence[Any]) -> list[SubgoalCandidate]:
    candidates: list[SubgoalCandidate] = []
    for index, row in enumerate(rows):
        if isinstance(row, SubgoalCandidate):
            candidates.append(row)
        elif isinstance(row, Mapping):
            predicate = row.get("predicate") or row.get("is_level_complete")
            if callable(predicate):
                candidates.append(
                    SubgoalCandidate(
                        name=str(row.get("name") or f"subgoal_{index}"),
                        predicate=predicate,
                        source=str(row.get("source") or "a1_goal_induction"),
                        score=float(row.get("score") or 0.0),
                        metadata=dict(row.get("metadata") or {}),
                    )
                )
        else:
            try:
                name, predicate, *rest = row
            except Exception:
                continue
            if callable(predicate):
                metadata = rest[1] if len(rest) > 1 and isinstance(rest[1], Mapping) else {}
                candidates.append(
                    SubgoalCandidate(
                        name=str(name),
                        predicate=predicate,
                        source="a1_goal_induction",
                        score=float(rest[0]) if rest else 0.0,
                        metadata=dict(metadata),
                    )
                )
    return candidates


def _exact_grid_predicate(target: np.ndarray) -> Callable[[np.ndarray], bool]:
    target_grid = np.asarray(target).copy()

    def _predicate(grid: np.ndarray) -> bool:
        candidate = np.asarray(grid)
        return candidate.shape == target_grid.shape and bool(np.array_equal(candidate, target_grid))

    return _predicate


def _nonzero_count_predicate(target: np.ndarray) -> Callable[[np.ndarray], bool]:
    target_count = int(np.count_nonzero(np.asarray(target)))

    def _predicate(grid: np.ndarray) -> bool:
        return int(np.count_nonzero(np.asarray(grid))) >= target_count

    return _predicate


def propose_hierarchical_subgoals(
    *,
    game: str,
    transitions: Sequence[Any],
    proposer: Any,
    previous_level_complete_grid: np.ndarray | None = None,
    max_subgoals: int = 4,
) -> list[SubgoalCandidate]:
    """REQ-ARC-WMTE-4676: mine failed-tree states and optional A1 subgoal proposals."""

    candidates: list[SubgoalCandidate] = []
    provider = getattr(proposer, "propose_subgoals", None)
    if callable(provider):
        try:
            proposed = provider(
                game=game,
                transitions=list(transitions),
                previous_level_complete_grid=previous_level_complete_grid,
                max_subgoals=max_subgoals,
            )
        except TypeError:
            proposed = provider(game, list(transitions))
        candidates.extend(_normalise_subgoal_candidates(list(proposed or [])))

    for index, transition in enumerate(list(transitions)[-int(max_subgoals) :]):
        grid = getattr(transition, "next_grid", None)
        if grid is None:
            continue
        grid_array = np.asarray(grid).copy()
        candidates.append(
            SubgoalCandidate(
                name=f"failed_tree_state_{index}",
                predicate=_exact_grid_predicate(grid_array),
                source="failed_search_tree",
                score=0.4 + 0.01 * index,
                metadata={
                    "nonzero_cells": int(np.count_nonzero(grid_array)),
                    "shape": list(grid_array.shape),
                },
            )
        )

    if previous_level_complete_grid is not None:
        exemplar = np.asarray(previous_level_complete_grid).copy()
        candidates.append(
            SubgoalCandidate(
                name="previous_level_complete_shape",
                predicate=_nonzero_count_predicate(exemplar),
                source="previous_level_complete_exemplar",
                score=0.6,
                metadata={
                    "nonzero_cells": int(np.count_nonzero(exemplar)),
                    "shape": list(exemplar.shape),
                },
            )
        )

    deduped: list[SubgoalCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = f"{candidate.source}:{candidate.name}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= int(max_subgoals):
            break
    return deduped


def _apply_model_plan(
    engine: Callable[[np.ndarray, int, Any], np.ndarray],
    start_grid: np.ndarray,
    plan: Sequence[Mapping[str, Any]] | None,
) -> np.ndarray:
    grid = np.asarray(start_grid)
    for step in list(plan or []):
        grid = np.asarray(engine(grid.copy(), int(step["action"]), step.get("data")))
    return grid


def _candidate_sort_key(
    candidate: SubgoalCandidate,
    value_head: Callable[[np.ndarray], float] | None,
) -> tuple[float, float, str]:
    local_value = 0.0
    target_grid = candidate.metadata.get("target_grid")
    if value_head is not None and target_grid is not None:
        try:
            local_value = float(value_head(np.asarray(target_grid)))
        except Exception:
            local_value = 0.0
    return (float(candidate.score), local_value, candidate.name)


def plan_hierarchical_subgoals(
    *,
    engine: Callable[[np.ndarray, int, Any], np.ndarray],
    final_goal: Callable[[np.ndarray], bool],
    start_grid: np.ndarray,
    subgoals: Sequence[SubgoalCandidate],
    plan_in_model: Callable[[Any, Any, np.ndarray], list[dict[str, Any]] | None],
    value_head: Callable[[np.ndarray], float] | None = None,
    max_subgoals: int = 3,
) -> HierarchicalSubgoalPlanResult:
    """SCENARIO-ARC-WMTE-4676-HIERARCHICAL-PLAN: chain bounded low-level plans."""

    ordered = sorted(
        list(subgoals),
        key=lambda candidate: _candidate_sort_key(candidate, value_head),
        reverse=True,
    )[: max(0, int(max_subgoals))]
    current = np.asarray(start_grid)
    full_plan: list[dict[str, Any]] = []
    decomposition: list[dict[str, Any]] = []
    reachable: list[dict[str, Any]] = []

    for candidate in ordered:
        leg = plan_in_model(engine, candidate.predicate, current)
        reached = leg is not None
        row = {
            "name": candidate.name,
            "source": candidate.source,
            "reachable": bool(reached),
            "plan_length": len(leg or []),
            "score": round(float(candidate.score), 6),
        }
        reachable.append(dict(row))
        decomposition.append(dict(row))
        if not reached:
            return HierarchicalSubgoalPlanResult(
                planned=False,
                plan=full_plan,
                subgoal_decomposition=decomposition,
                per_subgoal_reachable=reachable,
                final_grid=current,
                residual="bounded_search_cannot_reach_subgoal",
            )
        full_plan.extend(dict(step) for step in list(leg or []))
        current = _apply_model_plan(engine, current, leg)

    final_leg = plan_in_model(engine, final_goal, current)
    final_reached = final_leg is not None
    final_row = {
        "name": "final_goal",
        "source": "terminal_goal_predicate",
        "reachable": bool(final_reached),
        "plan_length": len(final_leg or []),
        "score": 1.0,
    }
    reachable.append(dict(final_row))
    decomposition.append(dict(final_row))
    if not final_reached:
        return HierarchicalSubgoalPlanResult(
            planned=False,
            plan=full_plan,
            subgoal_decomposition=decomposition,
            per_subgoal_reachable=reachable,
            final_grid=current,
            residual="subgoals_mechanically_irrelevant",
        )
    full_plan.extend(dict(step) for step in list(final_leg or []))
    final_grid = _apply_model_plan(engine, current, final_leg)
    return HierarchicalSubgoalPlanResult(
        planned=True,
        plan=full_plan,
        subgoal_decomposition=decomposition,
        per_subgoal_reachable=reachable,
        final_grid=final_grid,
        residual="none",
    )


def _plan_reaches_goal(
    *,
    engine: Callable[[np.ndarray, int, Any], np.ndarray],
    goal: Callable[[np.ndarray], bool] | None,
    start_grid: np.ndarray,
    plan: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if goal is None:
        return {"reaches_goal": False, "counterexample": {"kind": "missing_goal_predicate"}}
    if not plan:
        try:
            if bool(goal(np.asarray(start_grid))):
                return {"reaches_goal": True, "final_grid": np.asarray(start_grid)}
        except Exception as exc:
            return {
                "reaches_goal": False,
                "counterexample": {"kind": "goal_predicate_error", "error": repr(exc)[:160]},
            }
        return {"reaches_goal": False, "counterexample": {"kind": "no_reachable_plan"}}

    grid = np.asarray(start_grid)
    for step_index, step in enumerate(plan):
        try:
            grid = np.asarray(engine(grid.copy(), int(step["action"]), step.get("data")))
        except Exception as exc:
            return {
                "reaches_goal": False,
                "counterexample": {
                    "kind": "plan_execution",
                    "step_index": step_index,
                    "step": dict(step),
                    "error": repr(exc)[:160],
                },
            }
    try:
        if bool(goal(grid)):
            return {"reaches_goal": True, "final_grid": grid}
    except Exception as exc:
        return {
            "reaches_goal": False,
            "counterexample": {"kind": "goal_predicate_error", "error": repr(exc)[:160]},
        }
    return {
        "reaches_goal": False,
        "counterexample": {
            "kind": "plan_execution",
            "step_index": len(plan) - 1,
            "reason": "plan_finished_before_goal",
        },
    }


def _goal_satisfiability_check(
    *,
    engine: Callable[[np.ndarray, int, Any], np.ndarray],
    goal: Callable[[np.ndarray], bool] | None,
    start_grid: np.ndarray,
    max_nodes: int = 20000,
    max_depth: int = 40,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4664: reject constant-false goals before invoking the planner."""

    if goal is None:
        return {
            "satisfiable": False,
            "counterexample": {"kind": "missing_goal_predicate"},
            "reachable_grids_evaluated": 0,
        }

    from collections import Counter, deque
    from carnot.agentic.arc_executable_world_model import to_ascii

    start = np.asarray(start_grid)

    def _probe_candidates(grid: np.ndarray) -> list[dict[str, Any]]:
        candidates = [{"action": action, "data": None} for action in (1, 2, 3, 4, 5)]
        flat = [int(value) for value in np.asarray(grid).flatten().tolist()]
        background = Counter(flat).most_common(1)[0][0] if flat else 0
        coords = np.argwhere(np.asarray(grid) != background)
        if coords.size == 0:
            coords = np.argwhere(np.asarray(grid) != 0)
        for r, c in coords[:32]:
            candidates.append({"action": 6, "data": {"x": int(c), "y": int(r)}})
        return candidates

    def _eval_goal(grid: np.ndarray, depth: int, evaluated: int) -> dict[str, Any] | None:
        try:
            if bool(goal(grid)):
                return {
                    "satisfiable": True,
                    "reachable_grids_evaluated": int(evaluated),
                    "first_true_depth": int(depth),
                    "counterexample": {},
                }
        except Exception as exc:
            return {
                "satisfiable": False,
                "reachable_grids_evaluated": int(evaluated),
                "counterexample": {
                    "kind": "goal_predicate_error",
                    "error": repr(exc)[:160],
                },
            }
        return None

    evaluated = 1
    start_result = _eval_goal(start, 0, evaluated)
    if start_result is not None:
        return start_result

    seen = {to_ascii(start)}
    q = deque([(start, 0)])
    engine_errors = 0
    while q and evaluated < int(max_nodes):
        grid, depth = q.popleft()
        if depth >= int(max_depth):
            continue
        for candidate in _probe_candidates(grid):
            try:
                next_grid = np.asarray(
                    engine(grid.copy(), int(candidate["action"]), candidate.get("data"))
                )
            except Exception:
                engine_errors += 1
                continue
            if next_grid.shape != start.shape:
                continue
            key = to_ascii(next_grid)
            if key in seen:
                continue
            seen.add(key)
            evaluated += 1
            result = _eval_goal(next_grid, depth + 1, evaluated)
            if result is not None:
                return result
            q.append((next_grid, depth + 1))
            if evaluated >= int(max_nodes):
                break

    return {
        "satisfiable": False,
        "reachable_grids_evaluated": int(evaluated),
        "engine_errors": int(engine_errors),
        "max_nodes": int(max_nodes),
        "max_depth": int(max_depth),
        "counterexample": {
            "kind": "degenerate_goal_predicate",
            "reachable_grids_evaluated": int(evaluated),
            "max_nodes": int(max_nodes),
            "max_depth": int(max_depth),
        },
    }


def _repair_degenerate_goal(
    *,
    engine: Callable[[np.ndarray, int, Any], np.ndarray],
    previous_level_complete_grid: np.ndarray | None,
    root_grid: np.ndarray,
) -> dict[str, Any] | None:
    """GOAL-REPAIR (2026-06-25 operator directive): rescue a DEGENERATE induced goal predicate.

    The LLM-induced ``is_level_complete`` frequently comes out degenerate -- a constant ``return
    False``, or an exact-match against a hardcoded win grid that the induced engine can never reach.
    In those cases ``_goal_satisfiability_check`` returns ``degenerate_goal_predicate`` and the
    planner has NO satisfiable target, so L1->L2 deepening stalls. This is the dominant lp85 failure
    mode even AFTER the truncation fix (which only guarantees the model EMITS code, not that the
    win-condition it writes is any good).

    Re-refactoring the engine (the loop's default fallback) does NOT help, because refactor prompts
    target the dynamics, not the goal predicate. Instead we substitute the exemplar-derived
    ``_nonzero_count_predicate`` fallback (already used as a hierarchical subgoal) and re-check
    satisfiability against the SAME engine. If that fallback is reachable, return it so the caller
    can plan toward it; otherwise return ``None`` (no exemplar available, or the fallback is also
    unreachable -> genuinely give up this round).

    WHY this is a real repair and not a cheat: the fallback is a NON-DEGENERATE, REACHABLE proxy
    (the level is "complete" once the grid has at least as many filled cells as the L1-completion
    exemplar). It is a heuristic, not an exact win oracle -- it can admit some non-win grids -- but
    it gives the planner a satisfiable target where a constant-false predicate blocks planning
    entirely. The downstream plan-reaches-goal check and the offline reproduction gate still decide
    whether the resulting plan is a REAL level-up, so a loose proxy cannot fabricate a solve; it can
    only unblock the search so a genuine deepening has a chance to be found and then verified.
    """
    if previous_level_complete_grid is None:
        return None
    exemplar = np.asarray(previous_level_complete_grid).copy()
    fallback = _nonzero_count_predicate(exemplar)
    check = _goal_satisfiability_check(
        engine=engine,
        goal=fallback,
        start_grid=np.asarray(root_grid),
    )
    if not bool(check.get("satisfiable")):
        return None
    return {
        "predicate": fallback,
        "satisfiability": dict(check),
        "source": "exemplar_nonzero_count_fallback",
    }


# ==============================================================================================
# REQ-ARC-WMTE-6035: BEST-ENGINE RETENTION ACROSS REFINEMENT ROUNDS
# ==============================================================================================
#
# THE DEFECT, IN PLAIN TERMS. This function runs up to MAX_REFINEMENT_ROUNDS=3 attempts at a
# world model. Round 1 induces; rounds 2..N call `proposer.refactor()`, and EVERY one of those
# calls overwrites `results/arc_e3/<game>/world_model.py` in place. Before this REQ, the loop
# then did two things that both assume the LAST round is the BEST round:
#
#   (i)  it left whatever round N wrote sitting on disk, so the next `load_engine()` -- next
#        stall, next level, next episode -- starts from the last refactor, however bad; and
#   (ii) it reassigned `last_engine = selected.engine` unconditionally every round, so even a
#        caller that never touches disk was handed round N's engine.
#
# Neither assumption holds. Refinement is not monotone: `refactor()` is a language model
# rewriting a program from a counterexample, and it frequently makes the model WORSE.
#
# THE MEASUREMENT (2026-07-28/29, offline, no LLM). Replaying the real historical write
# sequence of `results/arc_e3/<game>/world_model.py` across 6 games:
#
#     last-write-wins (shipped)     mean change_fidelity 0.0042, change-gate 0 of 12
#     retain-best (this REQ)        mean change_fidelity 0.3979, change-gate 5 of 12
#
# and retain-best matched the CIRCULAR oracle ceiling (select on the evaluation metric itself)
# in 6 of 6 games. Concretely: ka59's engine peaked at change_fidelity 1.0000 and is 0.0000 on
# disk today; ar25 peaked at 0.8157 and is 0.0000. Three of the 15 regressive writes happened
# under the STRONG codex/gpt-5.5 proposer, so this is a store defect, not a proposer defect.
#
# THE SAME DEFECT, DEMONSTRATED ON THIS FUNCTION rather than on the store's git history: a
# scripted proposer replaying real recorded ka59 induction blobs at the LIVE thresholds
# produced per-round heldout accuracies 0.65 -> 0.15 -> 0.075. The loop returned round 3
# (change_fidelity 0.0000) and left round 3 on disk. Round 1 was change_fidelity 1.0000.
#
# WHY THE SELECTION SIGNAL IS `heldout_change_consistency` AND NOTHING ELSE. The whole point is
# a fix that works on a HIDDEN game mid-episode, where there is no ground truth to peek at. So
# retention is only allowed to rank rounds on a number the loop ALREADY computes at runtime:
# `select_trusted_world_model(...).selected_score.heldout_change_consistency` -- agreement on
# the held-out split of the agent's own observed transitions, restricted to cells that actually
# changed. It never sees the evaluation metric, a different seed, a level counter, or the game
# source. That is exactly what made the counterfactual above non-circular; a version of this
# code that reached for anything else would be a fabricated win.
#
# WHY DEFAULT-ON, STATED HONESTLY. Retention is not free: it can hold an engine that a later
# round would have improved on. The counterfactual bounded that downside directly by replaying
# 55 sliding 3-write windows (the bound MAX_REFINEMENT_ROUNDS=3 imposes on a single live
# episode): retention HELPED 24, HURT 3, tied 28; sign test over the 27 discordant windows
# p = 4.9e-5; mean delta +0.0898 over all 55 windows, +0.1829 over the 27 discordant ones.
# Default-on is defensible because 3-of-55 is the measured downside, not because the downside is
# zero. `CARNOT_ARC_ENGINE_RETENTION=0` restores exact last-write-wins behaviour for anyone who
# needs the old path back.
#
# TIES GO TO THE INCUMBENT (strict `>` below). A tie is no evidence to overwrite on, and it is
# what the counterfactual replayed.
#
# THREE HONEST LIMITS, NONE OF THEM FIXED HERE.
#
#   1. AN ALL-NO-OP HELD-OUT WINDOW MAKES THE SIGNAL UNINFORMATIVE. `consistency` is
#      correct_changed_cells / max(1, true_changed_cells), so a held-out split containing no
#      state change at all scores 0.0 for EVERY round -- including a perfect engine. Retention
#      then degenerates to "keep round 1" rather than to "keep the best round". This is the
#      same FALSE_NEGATIVE_RISK shape the goal-consistency veto guards against by requiring
#      `n_real_levelups >= 1`. Do NOT read that regime as "harmless": in the counterfactual, of
#      the 23 windows whose winning signal was 0.0, retention helped 4, HURT 3 and tied 16 --
#      an arbitrary coin flip, and ALL 3 of the change's measured hurt windows live here. The
#      case FOR default-on is the other side of the same split: 0 of the 32 INFORMATIVE windows
#      were hurt. Named here so nobody later reads a 0.0 retention signal as evidence about the
#      engine.
#
#   2. ON A PLANNING RETURN THE STORE AND THE RETURNED ENGINE CAN DIVERGE. The planned-return
#      paths return the PLANNING round's engine (the plan was validated in-model against it),
#      while the store is rolled back to the retained round. The shipped `min_heldout_accuracy
#      =1.0` bounds that divergence ONLY when the held-out split contains at least one CHANGING
#      transition: exact prediction of a changing transition gets every changed cell right, so
#      consistency is 1.0 -- maximal, hence retained. When the split has NO changing transition
#      the bound evaporates, because consistency is then 0.0 for every engine including a
#      perfect one, while heldout_accuracy is 1.0 for a do-nothing engine and clears the gate.
#      Measured counterexample: ft09 seed 0, 0 of 40 held-out transitions change, a do-nothing
#      engine scores heldout_accuracy 1.0 / consistency 0.0 (hostile_noop_heldout_results.json,
#      55cbd98a7baf3cd4). Every round then ties, round 1 is kept, and a planning round >= 2
#      returns its own engine while the store holds round 1's. Two things keep that honest
#      rather than silent: the planned-return paths report THIS round's scalar and goal
#      diagnostics (so the returned plan, goal and engine always describe each other), and
#      `engine_retention["best_round"]` names which round the STORE ended up holding.
#
#   3. THE SIGNAL IS A RECALL, SO IT IS NOT RANK-CONSISTENT WITH THE SYMMETRIC
#      `change_fidelity` THE COUNTERFACTUAL EVALUATED IT BY. `heldout_change_consistency`
#      counts correctly-predicted changed cells over TRUE changed cells; it does not penalise
#      spurious changes the way fidelity does. They can therefore disagree: ka59 `9d36f9f25`
#      scores signal 0.9830 / fidelity 0.9119, ranking ABOVE `a7488e97a` at 0.9575 / 1.0000 --
#      ranking by the shipped signal would prefer the worse engine. This never bit the measured
#      result: no 3-version window isolates that pair without the peak `341f776c9`, and
#      recall-argmax equalled fidelity-argmax in 6 of 6 games with 0.000 fidelity left on the
#      table. It is a real residual, not a hypothetical one -- it just has no measured victim.
#      Fixing it would mean ranking on a signal the live loop does not currently compute, which
#      is exactly the peeking this REQ refuses to do.

_RETENTION_ENV = "CARNOT_ARC_ENGINE_RETENTION"


def engine_retention_enabled() -> bool:
    """REQ-ARC-WMTE-6035: retention is ON unless explicitly disabled (see the block above)."""

    raw = os.environ.get(_RETENTION_ENV)
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _retention_signal(selection: Any) -> float:
    """The ONLY quantity retention is allowed to rank rounds on.

    Deliberately reads `heldout_change_consistency` and nothing else: it is computed by
    `select_trusted_world_model` on the held-out split of the agent's OWN transitions, which is
    the only kind of signal a hidden-game episode actually has. A missing field yields 0.0 for
    every round rather than a fallback to some other metric -- every round then ties, and the
    tie rule keeps the incumbent, which is still retention. Silently falling back to a
    different metric would make the shipped selection rule differ from the measured one.
    """

    score = getattr(selection, "selected_score", None)
    try:
        return float(getattr(score, "heldout_change_consistency", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _retention_signal_true_changed_cells(selection: Any) -> int:
    """DIAGNOSTIC ONLY -- never read by the comparison. Tells the two 0.0s apart.

    The retention signal is `correct_changed_cells / max(1, true_changed_cells)`, so a 0.0 is
    ambiguous between "the engine got every changed cell wrong" and "the held-out split had no
    changed cell to get right" (honest limit 1 above; ft09 seed 0 is the measured instance).
    Recording the denominator makes an artifact reader able to distinguish them without a
    re-run.

    IT IS DELIBERATELY NOT PART OF `is_best`, and it is NOT a safe tiebreak either. It looks
    like a pure property of the corpus -- reality's changed-cell count -- but it is not:
    `score_change_weighted_consistency`'s accumulation loop `continue`s BEFORE
    `true_changed_cells += n_changed_cells` when the engine RAISES or returns a wrong-shaped
    grid. A CRASHING engine therefore reports a SMALLER denominator, so ranking on it would
    reward an engine for failing to be scored. That was verified the hard way: a mutant wiring
    this into `is_best` survived the first version of the test suite, and the rationalisation
    "it's an equivalent mutant, the denominator is a corpus constant" was wrong. See
    `test_the_informativeness_denominator_never_enters_the_comparison`.
    """

    score = getattr(selection, "selected_score", None)
    try:
        return int(getattr(score, "true_changed_cells", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _engine_store_path(game: str) -> Path:
    """Resolve `E3_DIR` at CALL time so `CARNOT_ARC_E3_DIR` / the `e3.E3_DIR` monkeypatch both
    redirect retention writes exactly like they redirect every other store access
    (REQ-ARC-WMTE-6016)."""

    from carnot.agentic import arc_executable_world_model as e3

    return Path(e3.E3_DIR) / str(game) / "world_model.py"


def _read_engine_source(game: str) -> str | None:
    """Snapshot the engine source the proposer just wrote. `None` when there is no file --
    an injected `load_engine` (tests, the structured-engine path) need not use the store at
    all, and in-memory retention still applies in that case."""

    try:
        path = _engine_store_path(game)
        return path.read_text() if path.exists() else None
    except OSError:
        return None


def _retain_engine_source_on_disk(
    game: str,
    source: str | None,
    *,
    enabled: bool,
    best_round: int,
    rounds_seen: int,
    signal: float,
    true_changed_cells: int = 0,
) -> dict[str, Any]:
    """Roll the store back to the retained round's source (defect (i) above).

    Returns the diagnostic recorded on `LlmReinductionResult.engine_retention`. Never raises:
    a store that cannot be written is a degraded-but-live episode, not a crashed one, and the
    in-memory half of the fix (defect (ii)) is unaffected by it.
    """

    info: dict[str, Any] = {
        "enabled": bool(enabled),
        "selection_signal": "heldout_change_consistency",
        "best_round": int(best_round),
        "rounds_seen": int(rounds_seen),
        "best_round_signal": (None if signal == float("-inf") else round(float(signal), 6)),
        # Diagnostic denominator (honest limit 1): 0 here means the signal COULD NOT
        # discriminate, not that the engine was bad. Never read by the comparison.
        "best_round_true_changed_cells": int(true_changed_cells),
        "signal_informative": bool(int(true_changed_cells) > 0),
        "restored": False,
    }
    if not enabled:
        info["reason"] = "disabled_by_env"
        return info
    if best_round <= 0 or source is None:
        info["reason"] = "no_store_snapshot"
        return info
    try:
        path = _engine_store_path(game)
        info["store_path"] = str(path)
        current = path.read_text() if path.exists() else None
        if current == source:
            info["reason"] = "store_already_holds_retained_engine"
            return info
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        info["restored"] = True
    except OSError as exc:
        info["error"] = repr(exc)[:160]
    return info


def execute_bounded_llm_reinduction(
    *,
    game: str,
    transitions: Sequence[Any],
    cell: int,
    root_grid: np.ndarray | None,
    proposer: Any,
    candidate_provider: Callable[[Any, Any], Sequence[Any]],
    load_engine: Callable[[str], tuple[Any, Any]],
    plan_in_model: Callable[[Any, Any, np.ndarray], list[dict[str, Any]] | None],
    max_rounds: int = MAX_REFINEMENT_ROUNDS,
    min_heldout_accuracy: float = 0.0,
    min_goal_predicate_consistency: float = 0.0,
    proposal_transitions: Sequence[Any] | None = None,
    previous_level_complete_grid: np.ndarray | None = None,
    enable_subgoal_search: bool = False,
    subgoal_budget: int = 3,
    value_head: Callable[[np.ndarray], float] | None = None,
    subgoal_candidates: Sequence[SubgoalCandidate] | None = None,
    enable_factored_planner: bool = False,
    factored_trust_threshold: float = 0.75,
    structural_goal_provider: Callable[[np.ndarray], Any] | None = None,
    goal_exemplar_grading: bool = False,
    # REQ-ARC-WMTE-6010: LOGICAL-coordinate HUD mask, threaded from the caller's explorer.
    # Default None keeps every existing caller byte-identical.
    hud_mask: Any = None,
) -> LlmReinductionResult:
    """REQ-ARC-WMTE-4544/4557: run executable proposal with K<=3 refinements."""

    rounds_limit = min(int(max_rounds), MAX_REFINEMENT_ROUNDS)
    specs = _model_specs(proposer)
    verifier_threshold = max(0.0, min(1.0, float(min_heldout_accuracy)))
    goal_consistency_threshold = max(0.0, min(1.0, float(min_goal_predicate_consistency)))
    if root_grid is None:
        return LlmReinductionResult(
            planned=False,
            refinement_rounds_used=0,
            model_specs=specs,
            skipped="missing_root_grid",
        )
    if not transitions:
        return LlmReinductionResult(
            planned=False,
            refinement_rounds_used=0,
            model_specs=specs,
            skipped="no_active_transitions",
        )

    induction_evidence = (
        list(proposal_transitions)
        if proposal_transitions is not None
        else _proposal_prefix(list(transitions))
    )
    rounds: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []
    last_counterexample: dict[str, Any] = {"kind": "initial_induction"}
    last_goal_names: list[str] = []
    last_dynamics_names: list[str] = []
    last_selected = ""
    last_engine = None
    last_goal = None
    last_goal_expression = ""
    last_structural_goal_diagnostics: dict[str, Any] = {}
    last_goal_satisfiable = False
    last_goal_satisfiability: dict[str, Any] = {}
    last_heldout_accuracy: float | None = None
    last_accepted = False
    skipped = "no_reachable_plan_after_refinement"
    structural_goal_candidate = _normalise_structural_goal_candidate(
        structural_goal_provider,
        np.asarray(root_grid),
    )
    # REQ-ARC-WMTE-6035 retention state. `best_signal` starts at -inf so round 1 always wins
    # its own comparison; every later round must be STRICTLY better to displace it.
    retention_on = engine_retention_enabled()
    best_signal = float("-inf")
    best_round_no = 0
    best_source: str | None = None
    best_true_changed_cells = 0

    def _finalise_engine_retention() -> dict[str, Any]:
        return _retain_engine_source_on_disk(
            game,
            best_source,
            enabled=retention_on,
            best_round=best_round_no,
            rounds_seen=len(rounds),
            signal=best_signal,
            true_changed_cells=best_true_changed_cells,
        )

    for round_index in range(rounds_limit):
        round_no = round_index + 1
        if round_index == 0:
            ok, message = _call_induce(
                proposer,
                game,
                induction_evidence,
                int(cell),
                previous_level_complete_grid,
            )
            action = "induce"
        else:
            ok, message = proposer.refactor(game, _counterexample_result(last_counterexample))
            action = "refactor"

        row: dict[str, Any] = {
            "round": round_no,
            "action": action,
            "proposer_ok": bool(ok),
        }
        if message:
            row["message"] = str(message)[:240]
        if not ok:
            row["skipped"] = "proposer_failed"
            rounds.append(row)
            skipped = "proposer_failed"
            break

        try:
            engine, goal = load_engine(game)
            candidates = _normalise_candidates(candidate_provider(engine, goal), engine, goal)
            selection = select_trusted_world_model(
                list(transitions),
                candidates,
                hidden_state=True,
            )
            selected = selection.selected
            selected_goal = selected.is_level_complete or goal
            if structural_goal_candidate is not None:
                if structural_goal_candidate.get("predicate") is not None:
                    selected_goal = structural_goal_candidate["predicate"]
                    last_goal_expression = str(
                        structural_goal_candidate.get("goal_expression") or ""
                    )
                    last_structural_goal_diagnostics = dict(
                        structural_goal_candidate.get("diagnostics") or {}
                    )
                    row["goal_expression"] = last_goal_expression
                    row["structural_goal_diagnostics"] = dict(last_structural_goal_diagnostics)
                elif structural_goal_candidate.get("error"):
                    row["structural_goal_error"] = structural_goal_candidate["error"]
            heldout_accuracy = float(selection.selected_score.heldout_accuracy)
            accepted = heldout_accuracy >= verifier_threshold
            names = [candidate.name for candidate in candidates]
            # REQ-ARC-WMTE-6035: this is the ONE place a round is compared against the best
            # round so far. `not retention_on` reproduces the pre-REQ unconditional
            # reassignment exactly, which is what makes the env flag a true A/B switch.
            retention_signal = _retention_signal(selection)
            retention_true_changed = _retention_signal_true_changed_cells(selection)
            is_best = (not retention_on) or (retention_signal > best_signal)
            if is_best:
                best_signal = retention_signal
                best_true_changed_cells = retention_true_changed
                best_round_no = round_no
                # Snapshot AFTER the proposer wrote and the engine loaded, so the bytes on
                # disk are known to be the bytes that produced this round's signal.
                best_source = _read_engine_source(game)
                last_heldout_accuracy = heldout_accuracy
                last_accepted = bool(accepted)
                last_goal_names = list(names)
                last_dynamics_names = list(names)
                last_selected = selected.name
                last_engine = selected.engine
                last_goal = selected_goal
            row.update(
                {
                    "retention_signal_heldout_change_consistency": round(retention_signal, 6),
                    "retention_signal_true_changed_cells": int(retention_true_changed),
                    "retained_as_best_engine": bool(is_best),
                    "selected_candidate_name": selected.name,
                    "goal_candidate_names": list(names),
                    "dynamics_candidate_names": list(names),
                    "prefix_accuracy": round(float(selection.selected_score.prefix_accuracy), 6),
                    "heldout_accuracy": round(heldout_accuracy, 6),
                    "heldout_threshold": round(verifier_threshold, 6),
                    "accepted_by_heldout_verifier": bool(accepted),
                    "trust_energy": round(float(selection.selected_score.trust_energy), 6),
                }
            )
            if not accepted:
                # REQ-ARC-WMTE-4544: attach REAL per-transition mismatch evidence (BEFORE/
                # PREDICTED/OBSERVED deltas), not just the scalar heldout_accuracy summary --
                # refactor_prompt() is built to consume concrete failing cases, and a bare
                # accuracy number gives the LLM nothing actionable to fix. Scoring against
                # the full transitions list (not just the held-out split) is deliberate: every
                # mismatch returned is still a genuinely observed transition the engine gets
                # wrong, which is what CEGIS refinement needs.
                # REQ-ARC-WMTE-6010: grade the counterexample evidence on the SAME
                # HUD-collapsed comparison the gate used. Feeding the LLM mismatches that are
                # only HUD-counter deltas is worse than useless -- it teaches the proposer to
                # model the step counter instead of the mechanic.
                real_verify = WorldModelVerifier(list(transitions), hud_mask=hud_mask).score(
                    selected.engine
                )
                last_counterexample = {
                    "kind": "heldout_transition_verification_failed",
                    "selected_candidate_name": selected.name,
                    "heldout_accuracy": round(heldout_accuracy, 6),
                    "heldout_threshold": round(verifier_threshold, 6),
                    "real_n": real_verify.n,
                    "real_n_correct": real_verify.n_correct,
                    "real_accuracy": round(float(real_verify.accuracy), 6),
                    "real_mismatches": list(real_verify.mismatches),
                }
                counterexamples.append(last_counterexample)
                row["counterexample"] = dict(last_counterexample)
                row["skipped"] = "heldout_transition_verification_failed"
                rounds.append(row)
                continue
            goal_check = _goal_satisfiability_check(
                engine=selected.engine,
                goal=selected_goal,
                start_grid=np.asarray(root_grid),
            )
            # REQ-ARC-WMTE-6035: the reported goal diagnostics must describe the engine that
            # is actually returned. Binding them to a round whose engine was discarded would
            # report round N's goal alongside round 1's dynamics -- a lie in the artifact.
            round_goal_satisfiable = bool(goal_check.get("satisfiable"))
            round_goal_satisfiability = dict(goal_check)
            if is_best:
                last_goal_satisfiable = round_goal_satisfiable
                last_goal_satisfiability = dict(goal_check)
            row["goal_predicate_satisfiable"] = bool(round_goal_satisfiable)
            row["goal_satisfiability"] = {
                key: value for key, value in goal_check.items() if key != "counterexample"
            }
            if not round_goal_satisfiable:
                # GOAL-REPAIR: the LLM-induced goal is degenerate (constant-false / unreachable
                # exact-match). Before giving up this round to an engine-refactor (which cannot fix
                # the goal), try the exemplar-derived nonzero-count fallback against THIS engine.
                repaired = _repair_degenerate_goal(
                    engine=selected.engine,
                    previous_level_complete_grid=previous_level_complete_grid,
                    root_grid=np.asarray(root_grid),
                )
                if repaired is not None:
                    selected_goal = repaired["predicate"]
                    round_goal_satisfiable = True
                    round_goal_satisfiability = dict(repaired["satisfiability"])
                    if is_best:
                        last_goal = selected_goal
                        last_goal_satisfiable = True
                        last_goal_satisfiability = dict(repaired["satisfiability"])
                    row["goal_repaired"] = repaired["source"]
                    row["goal_predicate_satisfiable"] = True
                    row["goal_satisfiability"] = {
                        key: value
                        for key, value in repaired["satisfiability"].items()
                        if key != "counterexample"
                    }
                    # fall through to planning with the repaired goal (no `continue`).
                else:
                    last_counterexample = dict(
                        goal_check.get("counterexample") or {"kind": "degenerate_goal_predicate"}
                    )
                    counterexamples.append(last_counterexample)
                    row["counterexample"] = dict(last_counterexample)
                    row["skipped"] = str(
                        last_counterexample.get("kind") or "degenerate_goal_predicate"
                    )
                    skipped = row["skipped"]
                    rounds.append(row)
                    continue
            if goal_consistency_threshold > 0.0:
                # REQ-ARC-WMTE-5593-3: `is_level_complete` gets installed as `plan_in_model`'s
                # search termination condition on the strength of the proposer's own code --
                # nothing checked it against real observed level-progress before this. Mirrors
                # the dynamics veto above (attach real mismatch evidence, skip this round so
                # `refactor()` gets a chance to fix it) rather than inventing a new pattern.
                # Only fires when the window contains at least one real level-up
                # (CLAUDE.md FALSE_NEGATIVE_RISK discipline) -- an all-no-op window makes
                # ANY predicate, including a constant-False stub, score a trivial 1.0, so a
                # veto there would be judging the predicate on uninformative data.
                # LEVER #2 (REQ-ARC-WMTE-5593-4, default-off via goal_exemplar_grading): the veto above
                # only fires when the window has a real level-up, but the toward-NEXT-level window has
                # zero (the episode-transition-start reset at each level boundary,
                # arc_competition_agent.py) -> the goal veto is structurally INERT exactly when it is
                # needed (a correct-dynamics / WRONG-win-predicate model at a deepening boundary is still
                # trusted -> GAP-ARCH-GOAL-NOT-VERIFIED). Fix (exp4020 insight): inject the already-captured
                # PRIOR-LEVEL win-state grid as one synthetic ground-truth POSITIVE so n_real_levelups>=1
                # and the induced is_level_complete must return True on a real win state (catches the
                # false-negative predicate) while still returning False on the real no-level-up window
                # (catches the too-loose predicate). Oracle-distinct: the predicate never reads the level
                # counter; the exemplar is a grid the agent itself banked live.
                consistency_window = list(transitions)
                if goal_exemplar_grading and previous_level_complete_grid is not None:
                    _exemplar = np.asarray(previous_level_complete_grid)
                    consistency_window = [
                        Transition(
                            grid=_exemplar,
                            action=0,
                            data=None,
                            next_grid=_exemplar,
                            level_before=0,
                            level_after=1,
                        ),
                        *consistency_window,
                    ]
                consistency = score_goal_predicate_consistency(selected_goal, consistency_window)
                row["goal_predicate_consistency_accuracy"] = round(float(consistency.accuracy), 6)
                row["goal_predicate_consistency_n_real_levelups"] = int(consistency.n_real_levelups)
                if (
                    consistency.n_real_levelups >= 1
                    and consistency.accuracy < goal_consistency_threshold
                ):
                    last_counterexample = {
                        "kind": "goal_predicate_consistency_failed",
                        "accuracy": round(float(consistency.accuracy), 6),
                        "threshold": round(goal_consistency_threshold, 6),
                        "n": consistency.n,
                        "n_correct": consistency.n_correct,
                        "n_real_levelups": consistency.n_real_levelups,
                        "mismatches": list(consistency.mismatches),
                    }
                    counterexamples.append(last_counterexample)
                    row["counterexample"] = dict(last_counterexample)
                    row["skipped"] = "goal_predicate_consistency_failed"
                    skipped = row["skipped"]
                    rounds.append(row)
                    continue
            plan = plan_in_model(selected.engine, selected_goal, np.asarray(root_grid))
            check = _plan_reaches_goal(
                engine=selected.engine,
                goal=selected_goal,
                start_grid=np.asarray(root_grid),
                plan=plan,
            )
        except Exception as exc:
            last_counterexample = {
                "kind": "selection_or_planning_exception",
                "error": repr(exc)[:160],
            }
            counterexamples.append(last_counterexample)
            row["skipped"] = last_counterexample["kind"]
            rounds.append(row)
            continue

        row.update(
            {
                "plan_length": len(plan or []),
                "plan_reaches_goal": bool(check["reaches_goal"]),
            }
        )
        if check["reaches_goal"]:
            rounds.append(row)
            return LlmReinductionResult(
                planned=True,
                plan=list(plan or []),
                goal_predicate=selected_goal,
                engine=selected.engine,
                selected_candidate_name=selected.name,
                goal_candidate_names=list(names),
                dynamics_candidate_names=list(names),
                refinement_rounds_used=round_no,
                verifier_is_oracle=False,
                model_specs=specs,
                # REQ-ARC-WMTE-6035: a PLANNED return describes THIS round -- its engine, its
                # goal predicate, its plan -- so its scalar and goal diagnostics must be this
                # round's too. Reporting the retained round's numbers here would contradict the
                # `goal_predicate` and `engine` returned alongside them. The retained-round
                # values are what the NON-planned return at the bottom reports, because that
                # one really does hand back the best engine found rather than this round's.
                heldout_accuracy=heldout_accuracy,
                accepted_by_heldout_verifier=bool(accepted),
                goal_predicate_satisfiable=round_goal_satisfiable,
                goal_satisfiability=dict(round_goal_satisfiability),
                goal_expression=last_goal_expression,
                structural_goal_diagnostics=last_structural_goal_diagnostics,
                engine_retention=_finalise_engine_retention(),
                rounds=rounds,
                counterexamples=counterexamples,
                skipped="",
            )
        if enable_subgoal_search:
            candidates = list(subgoal_candidates or []) or propose_hierarchical_subgoals(
                game=game,
                transitions=list(transitions),
                proposer=proposer,
                previous_level_complete_grid=previous_level_complete_grid,
                max_subgoals=max(1, int(subgoal_budget)),
            )
            subgoal_result = plan_hierarchical_subgoals(
                engine=selected.engine,
                final_goal=selected_goal,
                start_grid=np.asarray(root_grid),
                subgoals=candidates,
                plan_in_model=plan_in_model,
                value_head=value_head,
                max_subgoals=max(1, int(subgoal_budget)),
            )
            row.update(
                {
                    "subgoal_search_used": True,
                    "subgoal_decomposition": list(subgoal_result.subgoal_decomposition),
                    "per_subgoal_reachable": list(subgoal_result.per_subgoal_reachable),
                    "subgoal_residual": subgoal_result.residual,
                    "hierarchical_plan_length": len(subgoal_result.plan),
                }
            )
            if subgoal_result.planned:
                rounds.append(row)
                return LlmReinductionResult(
                    planned=True,
                    plan=list(subgoal_result.plan),
                    goal_predicate=selected_goal,
                    engine=selected.engine,
                    selected_candidate_name=selected.name,
                    goal_candidate_names=list(names),
                    dynamics_candidate_names=list(names),
                    refinement_rounds_used=round_no,
                    verifier_is_oracle=False,
                    model_specs=specs,
                    # REQ-ARC-WMTE-6035: see the twin comment on the direct-plan return.
                    heldout_accuracy=heldout_accuracy,
                    accepted_by_heldout_verifier=bool(accepted),
                    goal_predicate_satisfiable=round_goal_satisfiable,
                    goal_satisfiability=dict(round_goal_satisfiability),
                    goal_expression=last_goal_expression,
                    structural_goal_diagnostics=last_structural_goal_diagnostics,
                    subgoal_decomposition=list(subgoal_result.subgoal_decomposition),
                    per_subgoal_reachable=list(subgoal_result.per_subgoal_reachable),
                    subgoal_search_used=True,
                    engine_retention=_finalise_engine_retention(),
                    rounds=rounds,
                    counterexamples=counterexamples,
                    skipped="",
                )
        if enable_factored_planner:
            try:
                from carnot.agentic.arc_executable_world_model import (
                    induce_programmatic_object_experts,
                    plan_factored_subgoal_sequence,
                )

                expert_result = induce_programmatic_object_experts(
                    game=game,
                    transitions=list(transitions),
                    proposer=proposer,
                    cell=int(cell),
                    trust_threshold=float(factored_trust_threshold),
                )
                candidates = list(subgoal_candidates or []) or propose_hierarchical_subgoals(
                    game=game,
                    transitions=list(transitions),
                    proposer=proposer,
                    previous_level_complete_grid=previous_level_complete_grid,
                    max_subgoals=max(1, int(subgoal_budget)),
                )
                factored_result = plan_factored_subgoal_sequence(
                    start_grid=np.asarray(root_grid),
                    final_goal=selected_goal,
                    experts=expert_result.experts,
                    subgoals=candidates,
                    value_head=value_head,
                    max_subgoals=max(1, int(subgoal_budget)),
                )
                row.update(
                    {
                        "factored_planner_used": True,
                        "expert_trust_weights": list(expert_result.expert_trust_weights),
                        "factored_subgoal_decomposition": list(
                            factored_result.subgoal_decomposition
                        ),
                        "factored_per_subgoal_reachable": list(
                            factored_result.per_subgoal_reachable
                        ),
                        "factored_plan_length": len(factored_result.plan),
                        "factored_residual": factored_result.residual or expert_result.residual,
                    }
                )
                if factored_result.planned:
                    rounds.append(row)
                    return LlmReinductionResult(
                        planned=True,
                        plan=list(factored_result.plan),
                        goal_predicate=selected_goal,
                        engine=selected.engine,
                        selected_candidate_name=selected.name,
                        goal_candidate_names=list(names),
                        dynamics_candidate_names=list(names),
                        refinement_rounds_used=round_no,
                        verifier_is_oracle=False,
                        model_specs=specs,
                        # REQ-ARC-WMTE-6035: see the twin comment on the direct-plan return.
                        heldout_accuracy=heldout_accuracy,
                        accepted_by_heldout_verifier=bool(accepted),
                        goal_predicate_satisfiable=round_goal_satisfiable,
                        goal_satisfiability=dict(round_goal_satisfiability),
                        goal_expression=last_goal_expression,
                        structural_goal_diagnostics=last_structural_goal_diagnostics,
                        subgoal_decomposition=list(factored_result.subgoal_decomposition),
                        per_subgoal_reachable=list(factored_result.per_subgoal_reachable),
                        factored_planner_used=True,
                        expert_trust_weights=list(expert_result.expert_trust_weights),
                        engine_retention=_finalise_engine_retention(),
                        rounds=rounds,
                        counterexamples=counterexamples,
                        skipped="",
                    )
            except Exception as exc:
                row["factored_planner_used"] = True
                row["factored_residual"] = "product_model_plans_live_invalid"
                row["factored_error"] = repr(exc)[:160]
        last_counterexample = dict(check["counterexample"])
        counterexamples.append(last_counterexample)
        row["counterexample"] = dict(last_counterexample)
        rounds.append(row)

    return LlmReinductionResult(
        planned=False,
        plan=[],
        goal_predicate=last_goal,
        engine=last_engine,
        selected_candidate_name=last_selected,
        goal_candidate_names=last_goal_names,
        dynamics_candidate_names=last_dynamics_names,
        refinement_rounds_used=len(rounds),
        verifier_is_oracle=False,
        model_specs=specs,
        heldout_accuracy=last_heldout_accuracy,
        accepted_by_heldout_verifier=last_accepted,
        goal_predicate_satisfiable=last_goal_satisfiable,
        goal_satisfiability=last_goal_satisfiability,
        goal_expression=last_goal_expression,
        structural_goal_diagnostics=last_structural_goal_diagnostics,
        engine_retention=_finalise_engine_retention(),
        rounds=rounds,
        counterexamples=counterexamples,
        skipped=skipped,
    )
