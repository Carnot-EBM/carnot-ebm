"""Structured ProductWorldModel loader for ARC level-up re-induction.

Spec refs: REQ-ARC-WMTE-4749,
SCENARIO-ARC-WMTE-4749-STRUCTURED-ENGINE-ADAPTER,
SCENARIO-ARC-WMTE-4749-LIVE-WIRING.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from carnot.agentic.arc_executable_world_model import (
    ProductWorldModel,
    ProgrammaticExpertInductionResult,
    Transition,
    induce_programmatic_object_experts,
    load_engine,
)


@dataclass
class StructuredEngineBuildResult:
    """REQ-ARC-WMTE-4749: product engine plus measurement diagnostics."""

    engine: Callable[[np.ndarray, int, Any], np.ndarray]
    goal: Any
    expert_result: ProgrammaticExpertInductionResult
    non_degenerate: bool
    heldout_accuracy: float

    @property
    def expert_trust_weights(self) -> list[dict[str, Any]]:
        return list(self.expert_result.expert_trust_weights)


class StructuredEngineReinductionProposer:
    """No-op free-form proposer shim for the structured engine path."""

    def __init__(self, base: Any) -> None:
        self.base = base
        self.model_specs = getattr(base, "model_specs", None) or getattr(base, "repo_substr", "")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return True, "structured_product_world_model_loader"

    def refactor(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return True, "structured_product_world_model_refactor_noop"


def _transition_value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def normalise_transition(row: Any) -> Transition:
    """Convert a transition-like object into the local `Transition` dataclass."""

    if isinstance(row, Transition):
        return row
    return Transition(
        grid=np.asarray(_transition_value(row, "grid")).copy(),
        action=int(_transition_value(row, "action", 0)),
        data=_transition_value(row, "data"),
        next_grid=np.asarray(_transition_value(row, "next_grid")).copy(),
        level_before=int(_transition_value(row, "level_before", 0) or 0),
        level_after=int(_transition_value(row, "level_after", 0) or 0),
    )


def normalise_transitions(rows: Sequence[Any]) -> list[Transition]:
    return [normalise_transition(row) for row in rows]


def heldout_transition_split(
    transitions: Sequence[Any],
    *,
    heldout_fraction: float = 0.34,
) -> tuple[list[Transition], list[Transition]]:
    """Return a deterministic prefix/held-out split for engine comparison."""

    rows = normalise_transitions(transitions)
    if len(rows) < 2:
        return rows, rows
    fraction = max(0.0, min(1.0, float(heldout_fraction)))
    suffix = max(1, int(round(len(rows) * fraction)))
    cut = max(1, len(rows) - suffix)
    return rows[:cut], rows[cut:]


def measure_engine_accuracy(
    engine: Callable[[np.ndarray, int, Any], Any],
    transitions: Sequence[Any],
) -> float:
    """Exact held-out transition accuracy for an engine on observed transitions."""

    rows = normalise_transitions(transitions)
    if not rows:
        return 0.0
    correct = 0
    for transition in rows:
        try:
            pred = np.asarray(
                engine(np.asarray(transition.grid).copy(), int(transition.action), transition.data)
            )
        except Exception:
            continue
        target = np.asarray(transition.next_grid)
        if pred.shape == target.shape and np.array_equal(pred, target):
            correct += 1
    return round(float(correct) / float(len(rows)), 6)


def structured_engine_non_degenerate(
    engine: Callable[[np.ndarray, int, Any], Any],
    transitions: Sequence[Any],
) -> bool:
    """True iff the engine changes at least one cell on a real observed frame/action."""

    for transition in normalise_transitions(transitions):
        start = np.asarray(transition.grid)
        try:
            pred = np.asarray(engine(start.copy(), int(transition.action), transition.data))
        except Exception:
            continue
        if pred.shape == start.shape and int(np.count_nonzero(pred != start)) > 0:
            return True
    return False


def _fallback_goal(game: str, fallback_goal_loader: Callable[[str], tuple[Any, Any]] | None) -> Any:
    loader = fallback_goal_loader or load_engine
    try:
        _engine, goal = loader(game)
    except Exception:
        return lambda _grid: False
    return goal if callable(goal) else (lambda _grid: False)


def build_structured_engine(
    game: str,
    *,
    transitions: Sequence[Any],
    proposer: Any = None,
    cell: int = 1,
    goal: Any = None,
    trust_threshold: float = 0.75,
    heldout_fraction: float = 0.34,
    max_experts: int = 8,
    fallback_goal_loader: Callable[[str], tuple[Any, Any]] | None = None,
) -> StructuredEngineBuildResult:
    """Induce trusted experts, compose ProductWorldModel, and score held-out accuracy."""

    rows = normalise_transitions(transitions)
    _prefix, heldout = heldout_transition_split(rows, heldout_fraction=heldout_fraction)
    expert_result = induce_programmatic_object_experts(
        game=str(game),
        transitions=rows,
        proposer=proposer,
        cell=int(cell),
        trust_threshold=float(trust_threshold),
        heldout_fraction=float(heldout_fraction),
        max_experts=int(max_experts),
    )
    product = ProductWorldModel(expert_result.experts)
    engine = product.engine
    selected_goal = goal if callable(goal) else _fallback_goal(str(game), fallback_goal_loader)
    return StructuredEngineBuildResult(
        engine=engine,
        goal=selected_goal,
        expert_result=expert_result,
        non_degenerate=structured_engine_non_degenerate(engine, rows),
        heldout_accuracy=measure_engine_accuracy(engine, heldout),
    )


def structured_load_engine(
    game: str,
    *,
    transitions: Sequence[Any],
    proposer: Any = None,
    cell: int = 1,
    goal: Any = None,
    trust_threshold: float = 0.75,
    heldout_fraction: float = 0.34,
    max_experts: int = 8,
    fallback_goal_loader: Callable[[str], tuple[Any, Any]] | None = None,
) -> tuple[Callable[[np.ndarray, int, Any], np.ndarray], Any]:
    """Load-engine-shaped adapter returning `(ProductWorldModel.engine, goal)`."""

    built = build_structured_engine(
        str(game),
        transitions=transitions,
        proposer=proposer,
        cell=int(cell),
        goal=goal,
        trust_threshold=float(trust_threshold),
        heldout_fraction=float(heldout_fraction),
        max_experts=int(max_experts),
        fallback_goal_loader=fallback_goal_loader,
    )
    return built.engine, built.goal


def make_structured_load_engine(
    *,
    game: str,
    transitions: Sequence[Any],
    proposer: Any = None,
    cell: int = 1,
    goal: Any = None,
    trust_threshold: float = 0.75,
    heldout_fraction: float = 0.34,
    max_experts: int = 8,
    fallback_goal_loader: Callable[[str], tuple[Any, Any]] | None = None,
) -> Callable[[str], tuple[Callable[[np.ndarray, int, Any], np.ndarray], Any]]:
    """Freeze live episode context into the `load_engine(game)` shape."""

    rows = normalise_transitions(transitions)

    def _load(requested_game: str) -> tuple[Callable[[np.ndarray, int, Any], np.ndarray], Any]:
        return structured_load_engine(
            str(requested_game or game),
            transitions=rows,
            proposer=proposer,
            cell=int(cell),
            goal=goal,
            trust_threshold=float(trust_threshold),
            heldout_fraction=float(heldout_fraction),
            max_experts=int(max_experts),
            fallback_goal_loader=fallback_goal_loader,
        )

    return _load
