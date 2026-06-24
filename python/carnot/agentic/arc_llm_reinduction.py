"""Bounded live-LLM re-induction helper for ARC level-up episodes.

Spec refs: REQ-ARC-WMTE-4544, SCENARIO-ARC-WMTE-4544,
REQ-ARC-WMTE-4557, SCENARIO-ARC-WMTE-4557-POSITIVE-CONTROL-FIRST,
REQ-ARC-WMTE-4664, SCENARIO-ARC-WMTE-4664-GOAL-SATISFIABILITY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from carnot.agentic.arc_world_model_trust_energy import (
    WorldModelCandidate,
    select_trusted_world_model,
)


MAX_REFINEMENT_ROUNDS = 3


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
    last_goal_satisfiable = False
    last_goal_satisfiability: dict[str, Any] = {}
    last_heldout_accuracy: float | None = None
    last_accepted = False
    skipped = "no_reachable_plan_after_refinement"

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
                last_counterexample = {
                    "kind": "heldout_transition_verification_failed",
                    "selected_candidate_name": selected.name,
                    "heldout_accuracy": round(heldout_accuracy, 6),
                    "heldout_threshold": round(verifier_threshold, 6),
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
                last_counterexample = dict(
                    goal_check.get("counterexample") or {"kind": "degenerate_goal_predicate"}
                )
                counterexamples.append(last_counterexample)
                row["counterexample"] = dict(last_counterexample)
                row["skipped"] = str(last_counterexample.get("kind") or "degenerate_goal_predicate")
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
                rounds=rounds,
                counterexamples=counterexamples,
                skipped="",
            )
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
        rounds=rounds,
        counterexamples=counterexamples,
        skipped=skipped,
    )
