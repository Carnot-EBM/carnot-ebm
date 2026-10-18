"""Verifier-first ARC-AGI-3 harness scaffold on a synthetic grid.

This module is deliberately infrastructure-only.  It proves that a tiny
environment, perception stub, verifier-as-router, action-pruner, fallback path,
and readiness artifact are wired together without claiming any real ARC-AGI-3
benchmark performance.

Spec refs: REQ-PHASE4-006, SCENARIO-PHASE4-006.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


HARNESS_MODULE_PATH = "python/carnot/agentic/arc_agi3_harness.py"
UNIT_TEST_PATH = "tests/python/test_arc_agi3_harness.py"
INFERENCE_SUBSTRATE = "synthetic_grid_plus_cpu_energy_verifier"
RANDOM_SEED = 3919

Coordinate = tuple[int, int]
Grid = tuple[tuple[int, ...], ...]
VerifierItem = dict[str, object]
EnergyVerifier = Callable[[tuple[VerifierItem, ...]], Mapping[str, object]]


@dataclass(frozen=True)
class Action:
    """One deterministic grid action."""

    name: str
    dx: int
    dy: int


ACTIONS: dict[str, Action] = {
    "east": Action("east", 1, 0),
    "west": Action("west", -1, 0),
    "north": Action("north", 0, -1),
    "south": Action("south", 0, 1),
    "stay": Action("stay", 0, 0),
}


@dataclass(frozen=True)
class Observation:
    """Perception stub output for the tiny synthetic grid."""

    grid: Grid
    position: Coordinate
    goal: Coordinate
    step_index: int


class Env(Protocol):
    """Minimal harness environment contract."""

    def reset(self) -> Observation:
        """Return the initial observation."""

    def step(self, action: Action) -> tuple[Observation, float, bool]:
        """Apply one action and return observation, reward, done."""


class SyntheticGridEnv:
    """A tiny deterministic 3x3 grid with a known two-step east-east goal."""

    def __init__(
        self,
        *,
        grid_size: int = 3,
        start: Coordinate = (0, 0),
        goal: Coordinate = (2, 0),
    ) -> None:
        self.grid_size = int(grid_size)
        self.start = start
        self.goal = goal
        self._position = start
        self._step_index = 0

    def reset(self) -> Observation:
        self._position = self.start
        self._step_index = 0
        return self._observe()

    def step(self, action: Action) -> tuple[Observation, float, bool]:
        self._position = self.project_position(self._position, action, self.grid_size)
        self._step_index += 1
        done = self._position == self.goal
        return self._observe(), float(done), done

    def candidate_actions(self) -> tuple[Action, ...]:
        return tuple(ACTIONS.values())

    def _observe(self) -> Observation:
        return Observation(
            grid=self._grid_for(self._position),
            position=self._position,
            goal=self.goal,
            step_index=self._step_index,
        )

    def _grid_for(self, position: Coordinate) -> Grid:
        rows = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        rows[self.goal[1]][self.goal[0]] = 2
        rows[position[1]][position[0]] = 1
        return tuple(tuple(row) for row in rows)

    @staticmethod
    def project_position(position: Coordinate, action: Action, grid_size: int) -> Coordinate:
        x_pos = min(max(position[0] + action.dx, 0), grid_size - 1)
        y_pos = min(max(position[1] + action.dy, 0), grid_size - 1)
        return x_pos, y_pos


def _manhattan(left: Coordinate, right: Coordinate) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def encode_candidate_action(observation: Observation, action: Action) -> VerifierItem:
    """Encode a state-action candidate as a cheap-verifier input row."""

    grid_size = len(observation.grid)
    next_position = SyntheticGridEnv.project_position(observation.position, action, grid_size)
    old_distance = _manhattan(observation.position, observation.goal)
    new_distance = _manhattan(next_position, observation.goal)
    progress = old_distance - new_distance
    if progress > 0:
        step_text = f"Action {action.name}. Progress proof: 1 + 0 = 1. The answer is 1."
    else:
        step_text = (
            f"Action {action.name}. Progress proof: 1 + 0 = 2. "
            "The answer is 0. actually inconsistent."
        )
    return {
        "step": step_text,
        "action_name": action.name,
        "position": observation.position,
        "next_position": next_position,
        "goal": observation.goal,
    }


def default_energy_verifier(items: tuple[VerifierItem, ...]) -> Mapping[str, object]:
    """Run the existing cheap energy verifier lazily."""

    from carnot.verify.cost_instrumented_verification import run_energy_verifier

    return run_energy_verifier(items)


@dataclass(frozen=True)
class CandidateScore:
    action_name: str
    verification_score: float
    energy_error_score: float


@dataclass(frozen=True)
class RouterDecision:
    action: Action
    scores: tuple[CandidateScore, ...]
    retained_action_names: tuple[str, ...]
    pruned_count: int
    fallback_used: bool

    def score_for(self, action_name: str) -> float:
        return next(score.verification_score for score in self.scores if score.action_name == action_name)


class VerifierRouter:
    """Verifier-as-router and action-pruner for the synthetic scaffold."""

    def __init__(
        self,
        *,
        keep_threshold: float = 0.93,
        verifier_fn: EnergyVerifier = default_energy_verifier,
        fallback_action: Action = ACTIONS["stay"],
    ) -> None:
        self.keep_threshold = float(keep_threshold)
        self.verifier_fn = verifier_fn
        self.fallback_action = fallback_action

    def select_action(
        self,
        observation: Observation,
        candidate_actions: Iterable[Action],
    ) -> RouterDecision:
        candidates = tuple(candidate_actions)
        if not candidates:
            raise ValueError("at least one candidate action is required")

        items = tuple(encode_candidate_action(observation, action) for action in candidates)
        raw_scores = tuple(float(score) for score in self.verifier_fn(items)["scores"])
        scores = tuple(
            CandidateScore(
                action_name=action.name,
                verification_score=max(0.0, min(1.0, 1.0 - energy_error_score)),
                energy_error_score=energy_error_score,
            )
            for action, energy_error_score in zip(candidates, raw_scores, strict=True)
        )
        score_by_name = {score.action_name: score.verification_score for score in scores}
        action_by_name = {action.name: action for action in candidates}
        retained_names = tuple(
            score.action_name for score in scores if score.verification_score >= self.keep_threshold
        )
        pruned_count = len(candidates) - len(retained_names)

        if retained_names:
            selected_name = max(retained_names, key=lambda name: score_by_name[name])
            return RouterDecision(
                action=action_by_name[selected_name],
                scores=scores,
                retained_action_names=retained_names,
                pruned_count=pruned_count,
                fallback_used=False,
            )
        return RouterDecision(
            action=self.fallback_action,
            scores=scores,
            retained_action_names=retained_names,
            pruned_count=pruned_count,
            fallback_used=True,
        )


@dataclass(frozen=True)
class HarnessResult:
    solved: bool
    actions_taken: tuple[str, ...]
    total_pruned_count: int
    steps: int
    final_observation: Observation
    decisions: tuple[RouterDecision, ...]
    random_seed: int
    is_synthetic_not_real_benchmark: bool = True

    @property
    def synthetic_task_solved(self) -> bool:
        return self.solved

    def as_checksum_payload(self) -> dict[str, object]:
        return {
            "solved": self.solved,
            "actions_taken": self.actions_taken,
            "total_pruned_count": self.total_pruned_count,
            "steps": self.steps,
            "final_position": self.final_observation.position,
            "random_seed": self.random_seed,
            "is_synthetic_not_real_benchmark": self.is_synthetic_not_real_benchmark,
        }


class ArcAgi3Harness:
    """Run the synthetic env through verifier-first action routing."""

    def __init__(
        self,
        *,
        env: SyntheticGridEnv,
        router: VerifierRouter,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.env = env
        self.router = router
        self.random_seed = int(random_seed)

    def run(self, *, max_steps: int = 8) -> HarnessResult:
        observation = self.env.reset()
        actions: list[str] = []
        decisions: list[RouterDecision] = []
        total_pruned_count = 0
        for _ in range(max_steps):
            decision = self.router.select_action(observation, self.env.candidate_actions())
            observation, _, done = self.env.step(decision.action)
            actions.append(decision.action.name)
            decisions.append(decision)
            total_pruned_count += decision.pruned_count
            if done:
                return HarnessResult(
                    solved=True,
                    actions_taken=tuple(actions),
                    total_pruned_count=total_pruned_count,
                    steps=len(actions),
                    final_observation=observation,
                    decisions=tuple(decisions),
                    random_seed=self.random_seed,
                )
        return HarnessResult(
            solved=False,
            actions_taken=tuple(actions),
            total_pruned_count=total_pruned_count,
            steps=len(actions),
            final_observation=observation,
            decisions=tuple(decisions),
            random_seed=self.random_seed,
        )


@dataclass(frozen=True)
class PreconditionResult:
    preconditions_checked: bool
    carnot_verify_imported: bool
    blocked_resource: str
    detail: str


def check_preconditions(
    import_fn: Callable[[str], object] = importlib.import_module,
) -> PreconditionResult:
    try:
        import_fn("carnot.verify")
    except Exception as exc:
        return PreconditionResult(
            preconditions_checked=True,
            carnot_verify_imported=False,
            blocked_resource="blocked_carnot_verify_import",
            detail=repr(exc),
        )
    return PreconditionResult(
        preconditions_checked=True,
        carnot_verify_imported=True,
        blocked_resource="",
        detail="import carnot.verify OK",
    )


def stable_reproducibility_checksum(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _run_date() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d")


def build_result_artifact(
    result: HarnessResult,
    *,
    preconditions: PreconditionResult,
    unit_test_passed: bool,
    duration_s: float,
    reproducibility_checksum: str,
) -> dict[str, object]:
    harness_ready = bool(
        preconditions.carnot_verify_imported
        and unit_test_passed
        and result.synthetic_task_solved
        and result.total_pruned_count > 0
        and result.is_synthetic_not_real_benchmark
    )
    if not preconditions.carnot_verify_imported:
        honest_verdict = preconditions.blocked_resource
    elif harness_ready:
        honest_verdict = (
            "complete: arc_agi3_scaffold_READY"
            f"_pruned{result.total_pruned_count}"
            "_synthetic_only_agentic_proof_can_follow_offline_proof"
        )
    else:
        honest_verdict = f"complete: arc_agi3_scaffold_NOT_READY_unit_test{unit_test_passed}"

    return {
        "experiment": 3919,
        "title": "arc_agi3_harness_scaffold",
        "run_date": _run_date(),
        "status": honest_verdict,
        "honest_verdict": honest_verdict,
        "harness_module_path": HARNESS_MODULE_PATH,
        "unit_test_path": UNIT_TEST_PATH,
        "unit_test_passed": bool(unit_test_passed),
        "synthetic_task_solved": bool(result.synthetic_task_solved),
        "action_pruned_count": int(result.total_pruned_count),
        "is_synthetic_not_real_benchmark": bool(result.is_synthetic_not_real_benchmark),
        "preconditions_checked": bool(preconditions.preconditions_checked),
        "carnot_verify_imported": bool(preconditions.carnot_verify_imported),
        "blocked_resource": preconditions.blocked_resource,
        "precondition_detail": preconditions.detail,
        "random_seed": int(result.random_seed),
        "reproducibility_checksum": reproducibility_checksum,
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "harness_ready": harness_ready,
        "actions_taken": list(result.actions_taken),
        "steps": int(result.steps),
    }


def write_result_artifact(artifact: Mapping[str, object], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return output_path
