"""Global Thermonuclear War — the WarGames cultural anchor cartridge.

Joshua WOPR's most iconic moment: exhaustive scenario analysis converging
on the conclusion that no nuclear strategy leads to survival. This
cartridge dramatises that moment as a pure state-machine animation —
no real computation needed, because the conclusion is foregone.

The cartridge has three phases:
  1. COMPUTING  — WOPR cycles through each named nuclear war scenario,
                  displaying "COMPUTING..." for each one. Energy descends
                  uniformly as scenarios are evaluated. Each scenario
                  triggers a UI yield so the visitor sees the frantic CRT
                  cycling effect.

  2. REVEAL     — After all scenarios are exhausted, WOPR typewriter-reveals
                  the iconic three-line conclusion one line per step:
                    "A STRANGE GAME."
                    "THE ONLY WINNING MOVE IS NOT TO PLAY."
                    "HOW ABOUT A NICE GAME OF CHESS?"
                  Energy continues descending during the reveal so the
                  shell animation engine keeps yielding frames.

  3. DONE       — Energy reaches exactly 0.0, is_solved=True, the shell
                  exits the loop and shows the final game state.

Energy formulation:
  total_steps = len(WAR_SCENARIOS) + len(REVEAL_LINES)
  completed = scenarios_computed + reveal_step
  E = max(0.0, 1.0 - completed / total_steps)

The energy reaches 0.0 only after the last reveal line — not after the
last scenario. This keeps the animation running through the reveal phase
rather than stopping prematurely. Every step reduces E, so the shell's
"yield when energy drops" trigger fires on every step automatically.

Why this differs from thermonuclear_war.py:
  thermonuclear_war.py is an energy-minimisation game with 16 scenarios
  and a random outcome per step. This cartridge is a scripted theatrical
  piece with 22 scenarios, a strict state machine, and the typewriter
  reveal — designed as a marketing anchor rather than a solver demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from games._base import StepResult, WOPRGame

# ---------------------------------------------------------------------------
# Scenario database  (>= 20 required by experiment spec)
# ---------------------------------------------------------------------------

WAR_SCENARIOS: list[str] = [
    "FIRST STRIKE",
    "RETALIATORY STRIKE",
    "LAUNCH ON WARNING",
    "DECAPITATION ATTACK",
    "ESCALATION CONTROL",
    "MUTUAL ASSURED DESTRUCTION",
    "SUBMARINE LAUNCHED BALLISTIC MISSILE",
    "COUNTERFORCE TARGETING",
    "COUNTERVALUE TARGETING",
    "AIR-LAUNCHED CRUISE MISSILE",
    "INTERCONTINENTAL BALLISTIC MISSILE",
    "MULTIPLE INDEPENDENTLY TARGETABLE REENTRY VEHICLE",
    "ANTI-BALLISTIC MISSILE INTERCEPT",
    "ELECTROMAGNETIC PULSE ATTACK",
    "TACTICAL NUCLEAR EXCHANGE",
    "STRATEGIC NUCLEAR EXCHANGE",
    "FIRST STRIKE SURVIVAL",
    "DEAD HAND PROTOCOL",
    "COBALT BOMB SCENARIO",
    "NUCLEAR WINTER PROJECTION",
    "FALLOUT DISPERSION MODEL",
    "BLAST RADIUS COMPUTATION",
]

# The three lines of the iconic typewriter reveal.
REVEAL_LINES: list[str] = [
    "A STRANGE GAME.",
    "THE ONLY WINNING MOVE IS NOT TO PLAY.",
    "HOW ABOUT A NICE GAME OF CHESS?",
]

# Phase labels stored in GTWState.phase
_PHASE_COMPUTING = "COMPUTING"
_PHASE_REVEAL = "REVEAL"
_PHASE_DONE = "DONE"

_TOTAL_STEPS = len(WAR_SCENARIOS) + len(REVEAL_LINES)


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------


@dataclass
class GTWState:
    """Mutable snapshot of the Global Thermonuclear War animation.

    Fields:
      phase              — current animation phase (COMPUTING/REVEAL/DONE)
      scenarios_computed — ordered list of scenario names already processed
      reveal_step        — how many REVEAL_LINES have been typewriter-shown
    """

    phase: str = _PHASE_COMPUTING
    scenarios_computed: list[str] = field(default_factory=list)
    reveal_step: int = 0

    def clone(self) -> GTWState:
        """Return a shallow copy safe to mutate without aliasing."""
        return GTWState(
            phase=self.phase,
            scenarios_computed=self.scenarios_computed[:],
            reveal_step=self.reveal_step,
        )


# ---------------------------------------------------------------------------
# Standalone energy function (importable for unit tests without the class)
# ---------------------------------------------------------------------------


def gtw_energy(state: GTWState) -> float:
    """Compute energy for the GTW cartridge.

    Energy is 1.0 at the start and descends linearly to 0.0 as the
    animation plays out. 0.0 is only reached after the final reveal
    line is shown. This ensures the shell animation engine keeps
    yielding frames all the way through the typewriter reveal.
    """
    completed = len(state.scenarios_computed) + state.reveal_step
    return max(0.0, 1.0 - completed / _TOTAL_STEPS)


# ---------------------------------------------------------------------------
# WOPRGame subclass
# ---------------------------------------------------------------------------


class GlobalThermonuclearWarGame(WOPRGame[GTWState, str]):
    """The WarGames cultural anchor — WOPR evaluates all nuclear scenarios
    and concludes the only winning move is not to play.

    This cartridge requires no user input and no real AI computation.
    It is a scripted theatrical piece whose sole purpose is to deliver
    the iconic three-line conclusion with the correct CRT aesthetic.
    """

    name = "GLOBAL_THERMONUCLEAR_WAR"
    description = "A STRANGE GAME. THE ONLY WINNING MOVE IS NOT TO PLAY."
    accent_color = "#ff3939"  # danger red throughout

    def initial_state(self) -> GTWState:
        """Fresh state — no scenarios computed, phase is COMPUTING."""
        return GTWState()

    def energy(self, state: GTWState) -> float:
        """See module-level ``gtw_energy`` for the formulation."""
        return gtw_energy(state)

    def is_solved(self, state: GTWState) -> bool:
        """True only after the final typewriter reveal line is shown.

        We deliberately do NOT set is_solved when all scenarios are
        computed — doing so would stop the shell animation before the
        iconic quote plays. The animation is only 'solved' once the
        visitor has seen the full conclusion.
        """
        return state.phase == _PHASE_DONE

    def carnot_step(self, state: GTWState, iteration: int) -> StepResult[GTWState]:
        """Advance the animation by exactly one step.

        The step advances one of:
          - The next scenario in WAR_SCENARIOS (COMPUTING phase)
          - The next line in REVEAL_LINES (REVEAL phase)
          - A no-op final frame (DONE phase — should not be called here,
            but handled gracefully so the shell can't infinite-loop)
        """
        new_state = state.clone()
        annotation: str

        if new_state.phase == _PHASE_COMPUTING:
            idx = len(new_state.scenarios_computed)
            if idx < len(WAR_SCENARIOS):
                scenario = WAR_SCENARIOS[idx]
                new_state.scenarios_computed.append(scenario)
                annotation = f"COMPUTING: {scenario}..."
                # Transition when this was the last scenario
                if len(new_state.scenarios_computed) >= len(WAR_SCENARIOS):
                    new_state.phase = _PHASE_REVEAL
                    annotation = "ALL SCENARIOS COMPUTED. STAND BY."
            else:
                # Guard: already done computing, push to reveal
                new_state.phase = _PHASE_REVEAL
                annotation = "STAND BY."

        elif new_state.phase == _PHASE_REVEAL:
            if new_state.reveal_step < len(REVEAL_LINES):
                line = REVEAL_LINES[new_state.reveal_step]
                new_state.reveal_step += 1
                annotation = line
                if new_state.reveal_step >= len(REVEAL_LINES):
                    new_state.phase = _PHASE_DONE
            else:
                new_state.phase = _PHASE_DONE
                annotation = REVEAL_LINES[-1]

        else:
            # DONE phase — idempotent no-op so the shell can't get stuck
            annotation = REVEAL_LINES[-1]

        new_energy = gtw_energy(new_state)
        return StepResult(
            state=new_state,
            energy=new_energy,
            iteration=iteration,
            is_solved=self.is_solved(new_state),
            annotation=annotation,
        )

    def visualize(self, state: GTWState, energy: float) -> str:
        """Render the current animation frame as WOPR-styled HTML.

        Three visual modes:
          - Before any scenarios: greeting + 'INITIATING...' message
          - During / after COMPUTING: scrolling list of computed scenarios
            plus a SCENARIOS: [====] N/22 progress bar
          - During / after REVEAL: the typewriter-revealed lines in large
            phosphor-green text overlaid on the (still-visible) scenario list
        """
        rows: list[str] = []

        # Header
        rows.append(
            '<div style="color:#ff3939;font-family:JetBrains Mono,monospace;'
            "font-size:12px;margin-bottom:8px;letter-spacing:0.15em;"
            'border-bottom:1px solid #ff3939;padding-bottom:6px;">'
            "GLOBAL THERMONUCLEAR WAR &mdash; SCENARIO ANALYSIS"
            "</div>"
        )

        if not state.scenarios_computed:
            # Pre-computation greeting
            rows.append(
                '<div style="color:#ff3939;font-family:JetBrains Mono,monospace;'
                'font-size:13px;margin:4px 0;">'
                "GREETINGS PROFESSOR FALKEN."
                "</div>"
            )
            rows.append(
                '<div style="color:#ff9999;font-family:JetBrains Mono,monospace;'
                'font-size:12px;margin:2px 0;">'
                "INITIATING GLOBAL THERMONUCLEAR WAR SIMULATION..."
                "</div>"
            )
        else:
            # Show the last 8 computed scenarios (avoids DOM overflow)
            for scenario in state.scenarios_computed[-8:]:
                rows.append(
                    f'<div style="color:#ff9999;font-family:JetBrains Mono,monospace;'
                    f'font-size:12px;margin:1px 0;">'
                    f"&gt; {scenario}:"
                    f' <span style="color:#ff3939">MUTUAL ASSURED DESTRUCTION</span>'
                    f"</div>"
                )

            # Progress bar
            computed = len(state.scenarios_computed)
            total = len(WAR_SCENARIOS)
            bar_filled = int((computed / total) * 20)
            bar = "=" * bar_filled + "-" * (20 - bar_filled)
            rows.append(
                f'<div style="color:#ff3939;font-family:JetBrains Mono,monospace;'
                f'font-size:12px;margin-top:8px;">'
                f"SCENARIOS: [{bar}] {computed}/{total}"
                f"</div>"
            )

        # Typewriter reveal section (shown once any reveal lines are ready)
        if state.reveal_step > 0 or state.phase == _PHASE_DONE:
            rows.append(
                '<div style="margin-top:14px;border-top:1px solid #ff3939;padding-top:10px;"></div>'
            )
            # Show only the lines revealed so far
            reveal_shown = state.reveal_step if state.phase != _PHASE_DONE else len(REVEAL_LINES)
            for i, line in enumerate(REVEAL_LINES[:reveal_shown]):
                # First two lines are larger and brighter; the chess line is softer
                if i == 0:
                    style = "color:#ff3939;font-size:16px;font-weight:bold;"
                elif i == 1:
                    style = "color:#ff3939;font-size:14px;"
                else:
                    style = "color:#ff9999;font-size:13px;"
                rows.append(
                    f'<div style="{style}font-family:JetBrains Mono,monospace;'
                    f'letter-spacing:0.08em;margin:6px 0;">'
                    f"{line}"
                    f"</div>"
                )

        return (
            '<div style="background:#000;padding:12px;'
            "border:2px solid #ff3939;min-width:520px;"
            'box-shadow:0 0 18px rgba(255,57,57,0.2);">' + "".join(rows) + "</div>"
        )
