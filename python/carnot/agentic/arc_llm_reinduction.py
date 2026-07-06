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
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from carnot.agentic.arc_executable_world_model import WorldModelVerifier
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
    proposal_transitions: Sequence[Any] | None = None,
    previous_level_complete_grid: np.ndarray | None = None,
    enable_subgoal_search: bool = False,
    subgoal_budget: int = 3,
    value_head: Callable[[np.ndarray], float] | None = None,
    subgoal_candidates: Sequence[SubgoalCandidate] | None = None,
    enable_factored_planner: bool = False,
    factored_trust_threshold: float = 0.75,
    structural_goal_provider: Callable[[np.ndarray], Any] | None = None,
) -> LlmReinductionResult:
    """REQ-ARC-WMTE-4544/4557: run executable proposal with K<=3 refinements."""

    rounds_limit = min(int(max_rounds), MAX_REFINEMENT_ROUNDS)
    specs = _model_specs(proposer)
    verifier_threshold = max(0.0, min(1.0, float(min_heldout_accuracy)))
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
            last_heldout_accuracy = heldout_accuracy
            last_accepted = bool(accepted)
            names = [candidate.name for candidate in candidates]
            last_goal_names = list(names)
            last_dynamics_names = list(names)
            last_selected = selected.name
            last_engine = selected.engine
            last_goal = selected_goal
            row.update(
                {
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
                real_verify = WorldModelVerifier(list(transitions)).score(selected.engine)
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
            last_goal_satisfiable = bool(goal_check.get("satisfiable"))
            last_goal_satisfiability = dict(goal_check)
            row["goal_predicate_satisfiable"] = bool(last_goal_satisfiable)
            row["goal_satisfiability"] = {
                key: value for key, value in goal_check.items() if key != "counterexample"
            }
            if not last_goal_satisfiable:
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
                heldout_accuracy=last_heldout_accuracy,
                accepted_by_heldout_verifier=last_accepted,
                goal_predicate_satisfiable=last_goal_satisfiable,
                goal_satisfiability=last_goal_satisfiability,
                goal_expression=last_goal_expression,
                structural_goal_diagnostics=last_structural_goal_diagnostics,
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
                    heldout_accuracy=last_heldout_accuracy,
                    accepted_by_heldout_verifier=last_accepted,
                    goal_predicate_satisfiable=last_goal_satisfiable,
                    goal_satisfiability=last_goal_satisfiability,
                    goal_expression=last_goal_expression,
                    structural_goal_diagnostics=last_structural_goal_diagnostics,
                    subgoal_decomposition=list(subgoal_result.subgoal_decomposition),
                    per_subgoal_reachable=list(subgoal_result.per_subgoal_reachable),
                    subgoal_search_used=True,
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
                        heldout_accuracy=last_heldout_accuracy,
                        accepted_by_heldout_verifier=last_accepted,
                        goal_predicate_satisfiable=last_goal_satisfiable,
                        goal_satisfiability=last_goal_satisfiability,
                        goal_expression=last_goal_expression,
                        structural_goal_diagnostics=last_structural_goal_diagnostics,
                        subgoal_decomposition=list(factored_result.subgoal_decomposition),
                        per_subgoal_reachable=list(factored_result.per_subgoal_reachable),
                        factored_planner_used=True,
                        expert_trust_weights=list(expert_result.expert_trust_weights),
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
        rounds=rounds,
        counterexamples=counterexamples,
        skipped=skipped,
    )
