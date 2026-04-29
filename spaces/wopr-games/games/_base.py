"""Base interface for WOPR cartridges.

Every cartridge under `spaces/wopr-games/games/` must subclass
`WOPRGame` and implement the abstract methods. The shell calls these
methods uniformly so adding a new cartridge is a single new file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

State = TypeVar("State")
Action = TypeVar("Action")


@dataclass
class StepResult(Generic[State]):
    """One step of Carnot's energy descent on this game's state."""

    state: State
    energy: float
    iteration: int
    is_solved: bool
    annotation: str  # "WOPR" flavour text shown alongside the step


class WOPRGame(ABC, Generic[State, Action]):
    """Cartridge interface for the WOPR shell.

    Every cartridge implements:
      - `name` (short label for the game selector)
      - `description` (one-line subtitle in the WOPR aesthetic)
      - `initial_state()` (where Carnot starts solving from)
      - `energy(state)` (the function Carnot minimizes)
      - `available_actions(state)` (used by interactive variants)
      - `apply_action(state, action)` (state transition)
      - `visualize(state, energy)` (HTML/text representation)
      - `carnot_step(state, iteration)` (one MCMC step toward solution)

    The shell handles common UI: terminal aesthetic, typewriter
    streaming, energy bar, flavour text, easter eggs.
    """

    name: str = "UNKNOWN GAME"
    description: str = ""
    accent_color: str = "#39ff14"  # default WOPR green

    @abstractmethod
    def initial_state(self) -> State:
        """The starting state for a Carnot solve attempt."""

    @abstractmethod
    def energy(self, state: State) -> float:
        """Energy of a state. Carnot minimizes this. 0.0 = solved."""

    @abstractmethod
    def is_solved(self, state: State) -> bool:
        """Whether the state satisfies all constraints."""

    def available_actions(self, state: State) -> list[Action]:
        """Override for interactive games (default: no actions)."""
        return []

    def apply_action(self, state: State, action: Action) -> State:
        """Override for interactive games. Default is identity."""
        return state

    @abstractmethod
    def visualize(self, state: State, energy: float) -> str:
        """Render state as HTML. Will be wrapped by the WOPR shell."""

    @abstractmethod
    def carnot_step(self, state: State, iteration: int) -> StepResult[State]:
        """One MCMC step toward solution. Used by the shell to animate
        the energy descent. Should be deterministic given (state, iteration)
        when possible, so animations are reproducible.
        """

    def carnot_solve(
        self, max_iterations: int = 5000, energy_threshold: float = 0.0
    ) -> list[StepResult[State]]:
        """Run Carnot's sampler until solved or max_iterations.

        Default implementation iterates `carnot_step` and stops when
        energy reaches the threshold. Override for cartridges that
        need different control flow.
        """
        state = self.initial_state()
        steps: list[StepResult[State]] = []
        for iteration in range(max_iterations):
            step = self.carnot_step(state, iteration)
            steps.append(step)
            state = step.state
            if step.is_solved or step.energy <= energy_threshold:
                break
        return steps
