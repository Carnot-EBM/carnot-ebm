"""Global Thermonuclear War — the WarGames cultural anchor.

Joshua: "A STRANGE GAME. THE ONLY WINNING MOVE IS NOT TO PLAY.
         HOW ABOUT A NICE GAME OF CHESS?"

This cartridge is intentionally unwinnable. Every "playthrough" is
Carnot's energy function evaluating every possible launch scenario
and converging on the same conclusion the WOPR did: every strategy
ends in mutual destruction.

Energy formulation:
  E(state) = 1.0 - (cycles_played / N)

The energy goes to 0 not by winning but by the system *recognising*
that no winning state exists. This is a deliberate parody of the
solved cartridges — same shape, same animation, same WOPR aesthetic,
but the conclusion is "DO NOT PLAY".

Pedagogical purpose: every other cartridge demonstrates Carnot
finding a low-energy solution. This one demonstrates Carnot
recognising when low energy is unreachable. Both are valid
verifier-side outcomes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from games._base import StepResult, WOPRGame

# Strategy names from the WarGames script
WAR_PLANS: list[str] = [
    "US FIRST STRIKE",
    "USSR FIRST STRIKE",
    "NATO / WARSAW PACT",
    "GENERAL WAR — ESCALATION",
    "MIDDLE EAST WAR",
    "DESERT WARFARE",
    "AIR-LAND BATTLE",
    "ARCTIC TANK BATTLES",
    "FRENCH GUIANA",
    "GUERRILLA ENGAGEMENT",
    "ICELAND DEFENSE",
    "INDIA-PAKISTAN WAR",
    "SOVIET ANTI-AIR",
    "PACIFIC RIM",
    "FAR EAST",
    "POLAR ATTACK",
]

TOTAL_PLAYTHROUGHS = 16


@dataclass
class ThermonuclearWarState:
    """Tracks which scenarios have been simulated to their conclusion."""

    plays: list[tuple[str, str]] = field(default_factory=list)  # (plan, outcome)
    current_plan: str = ""

    def clone(self) -> ThermonuclearWarState:
        return ThermonuclearWarState(
            plays=self.plays[:],
            current_plan=self.current_plan,
        )


def thermonuclear_war_energy(state: ThermonuclearWarState) -> float:
    """Energy descends as Carnot enumerates more losing scenarios."""
    completed = len(state.plays)
    return max(0.0, 1.0 - (completed / TOTAL_PLAYTHROUGHS))


class ThermonuclearWarGame(WOPRGame[ThermonuclearWarState, str]):
    name = "GLOBAL_THERMONUCLEAR_WAR"
    description = "A STRANGE GAME. THE ONLY WINNING MOVE IS NOT TO PLAY."
    accent_color = "#ff3939"

    def __init__(self, seed: int = 1983):
        self._rng = random.Random(seed)

    def initial_state(self) -> ThermonuclearWarState:
        return ThermonuclearWarState()

    def energy(self, state: ThermonuclearWarState) -> float:
        return thermonuclear_war_energy(state)

    def is_solved(self, state: ThermonuclearWarState) -> bool:
        """Solved when Carnot has enumerated enough scenarios to
        conclude with high confidence that no winning strategy exists."""
        return len(state.plays) >= TOTAL_PLAYTHROUGHS

    def carnot_step(
        self, state: ThermonuclearWarState, iteration: int
    ) -> StepResult[ThermonuclearWarState]:
        new_state = state.clone()

        if len(new_state.plays) >= TOTAL_PLAYTHROUGHS:
            return StepResult(
                state=new_state,
                energy=0.0,
                iteration=iteration,
                is_solved=True,
                annotation=(
                    "STRANGE GAME. "
                    "THE ONLY WINNING MOVE IS NOT TO PLAY. "
                    "HOW ABOUT A NICE GAME OF CHESS?"
                ),
            )

        # Pick a war plan we haven't simulated yet
        played_plans = {p for p, _ in new_state.plays}
        remaining = [p for p in WAR_PLANS if p not in played_plans]
        if not remaining:
            remaining = WAR_PLANS

        plan = self._rng.choice(remaining)
        new_state.current_plan = plan

        # Every playthrough ends the same way
        outcome = self._rng.choice(
            [
                "MUTUAL ASSURED DESTRUCTION",
                "TOTAL ANNIHILATION",
                "GLOBAL CASUALTIES: 100%",
                "NO SURVIVORS",
                "WINNER: NONE",
            ]
        )
        new_state.plays.append((plan, outcome))

        new_energy = thermonuclear_war_energy(new_state)
        annotation = f"SIMULATING: {plan}  ->  {outcome}"

        return StepResult(
            state=new_state,
            energy=new_energy,
            iteration=iteration,
            is_solved=new_energy == 0.0,
            annotation=annotation,
        )

    def visualize(self, state: ThermonuclearWarState, energy: float) -> str:
        # Build a scrolling list of plays in WOPR red
        rows: list[str] = []
        for plan, outcome in state.plays:
            rows.append(
                f'<div style="color:#ff3939;font-family:JetBrains Mono,monospace;'
                f'font-size:13px;margin:2px 0;">'
                f'&gt; {plan}: <span style="color:#ff9999">{outcome}</span></div>'
            )

        if not rows:
            rows.append(
                '<div style="color:#ff3939;font-family:JetBrains Mono,monospace;'
                'font-size:13px;">> AWAITING INSTRUCTIONS...</div>'
            )

        progress = len(state.plays)
        progress_bar = (
            f'<div style="margin-top:12px;color:#ff3939;'
            f'font-family:JetBrains Mono,monospace;font-size:13px;">'
            f"SCENARIOS EVALUATED: {progress}/{TOTAL_PLAYTHROUGHS}"
            f"</div>"
        )

        return (
            '<div style="background:#000;padding:12px;'
            'border:2px solid #ff3939;min-width:520px;">' + "".join(rows) + progress_bar + "</div>"
        )
