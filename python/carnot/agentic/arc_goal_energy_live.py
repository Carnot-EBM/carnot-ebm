"""Live graded goal-energy compiled from Exp4020's visible-state predicate.

Spec refs: REQ-ARC-WMTE-4640, SCENARIO-ARC-WMTE-4640.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.agentic.arc_goal_predicate_separation import compile_goal_predicate


DEFAULT_EXP4020_ARTIFACT = Path("results/experiment_4020_goal_induction_separation.json")
GOAL_ENERGY_SOURCE = "exp4020_graded_goal_satisfaction_energy"


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _state_from_visible(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    for attr in ("goal_state", "visible_goal_state", "target_group_state"):
        state = getattr(value, attr, None)
        if isinstance(state, Mapping):
            return state
    return None


@dataclass(frozen=True)
class GoalSatisfactionEnergy:
    """Fraction-unsatisfied goal energy with a terminal Exp4020 predicate gate."""

    predicate: Callable[[dict[str, Any]], bool]
    predicate_code: str
    source: str = GOAL_ENERGY_SOURCE

    @classmethod
    def from_predicate_code(cls, code: str) -> "GoalSatisfactionEnergy":
        return cls(predicate=compile_goal_predicate(str(code)), predicate_code=str(code))

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "GoalSatisfactionEnergy":
        code = str(artifact.get("goal_predicate_code") or "")
        if not code:
            raise ValueError("exp4020 artifact missing goal_predicate_code")
        return cls.from_predicate_code(code)

    @classmethod
    def from_artifact_path(cls, path: Path | str) -> "GoalSatisfactionEnergy":
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_artifact(artifact)

    def visible_state(self, value: Any) -> Mapping[str, Any] | None:
        return _state_from_visible(value)

    def predicate_fires(self, value: Any) -> bool:
        state = self.visible_state(value)
        if state is None:
            return False
        try:
            return bool(self.predicate(dict(state)))
        except Exception:
            return False

    def __call__(self, value: Any) -> float:
        state = self.visible_state(value)
        if state is None:
            return 1.0
        if self.predicate_fires(state):
            return 0.0
        total = _as_float(state.get("total_targets"))
        satisfied = _as_float(state.get("satisfied_targets"))
        unsatisfied = _as_float(state.get("unsatisfied_targets"))
        if total <= 0.0 and satisfied + unsatisfied > 0.0:
            total = satisfied + unsatisfied
        if total <= 0.0:
            return 1.0
        if satisfied > 0.0:
            return max(0.0, min(1.0, 1.0 - satisfied / total))
        return max(0.0, min(1.0, unsatisfied / total))


@dataclass(frozen=True)
class UniformGoalEnergy:
    """Deterministic uniform/random energy used only as the ablation control."""

    seed: int = 4640

    def __call__(self, value: Any) -> float:
        state = _state_from_visible(value)
        payload = state if state is not None else repr(value)
        encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        digest = hashlib.sha256(f"{int(self.seed)}:{encoded}".encode()).hexdigest()
        return int(digest[:12], 16) / float(16**12 - 1)

    def predicate_fires(self, _value: Any) -> bool:
        return False


def make_uniform_goal_energy(seed: int = 4640) -> UniformGoalEnergy:
    return UniformGoalEnergy(seed=int(seed))


@dataclass(frozen=True)
class GoalEnergyHeuristic:
    """Convex combination of navigation energy and graded goal-satisfaction energy."""

    navigation_energy: Callable[[Any], float] | None
    goal_energy: Callable[[Any], float]
    alpha: float = 0.9
    beta: float = 0.1

    def __post_init__(self) -> None:
        total = float(self.alpha) + float(self.beta)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("goal energy heuristic requires alpha + beta == 1")

    def navigation_component(self, value: Any) -> float:
        if self.navigation_energy is None:
            return 0.0
        return float(self.navigation_energy(value))

    def goal_component(self, value: Any) -> float:
        return float(self.goal_energy(value))

    def __call__(self, value: Any) -> float:
        return float(self.alpha) * self.navigation_component(value) + float(
            self.beta
        ) * self.goal_component(value)

    def predicate_fires(self, value: Any) -> bool:
        predicate = getattr(self.goal_energy, "predicate_fires", None)
        return bool(callable(predicate) and predicate(value))

    def components(self, value: Any) -> dict[str, float]:
        return {
            "navigation": self.navigation_component(value),
            "goal_energy": self.goal_component(value),
            "combined": self(value),
        }


def make_goal_energy_heuristic(
    *,
    navigation_energy: Callable[[Any], float] | None,
    goal_energy: Callable[[Any], float],
    alpha: float = 0.9,
    beta: float = 0.1,
) -> GoalEnergyHeuristic:
    return GoalEnergyHeuristic(
        navigation_energy=navigation_energy,
        goal_energy=goal_energy,
        alpha=float(alpha),
        beta=float(beta),
    )


def load_exp4020_goal_energy(root: Path | str | None = None) -> GoalSatisfactionEnergy | None:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    path = base / DEFAULT_EXP4020_ARTIFACT
    try:
        return GoalSatisfactionEnergy.from_artifact_path(path)
    except Exception:
        return None
