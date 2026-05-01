"""Unit tests for the Global Thermonuclear War WOPR cartridge.

Covers:
  - REQ-GTW-001: >= 20 named war scenarios in WAR_SCENARIOS
  - REQ-GTW-002: energy strictly descends from 1.0 toward 0.0 each step
  - REQ-GTW-003: after all scenarios computed, phase transitions to REVEAL
  - REQ-GTW-004: typewriter reveal emits each REVEAL_LINES entry in order
  - REQ-GTW-005: is_solved returns True only after all reveal lines shown
  - REQ-GTW-006: full run completes in exactly len(WAR_SCENARIOS) + len(REVEAL_LINES) steps
  - REQ-GTW-007: visualize returns HTML with the progress bar and scenario names
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing from the spaces sub-package without installing it.
# The cartridge and its base class live under spaces/wopr-games/games/,
# which is not on the default Python path when running from the repo root.
_SPACES_GAMES = Path(__file__).resolve().parent.parent.parent / "spaces" / "wopr-games"
if str(_SPACES_GAMES) not in sys.path:
    sys.path.insert(0, str(_SPACES_GAMES))

from games.global_thermonuclear_war import (  # noqa: E402
    REVEAL_LINES,
    WAR_SCENARIOS,
    GTWState,
    GlobalThermonuclearWarGame,
    gtw_energy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_to_completion(game: GlobalThermonuclearWarGame) -> list:
    """Drive the game from initial_state to is_solved=True.

    Returns the list of StepResult objects produced, one per iteration.
    Raises AssertionError if the loop runs past twice the expected length
    (guard against infinite loops in the cartridge under test).
    """
    state = game.initial_state()
    steps = []
    max_iters = (len(WAR_SCENARIOS) + len(REVEAL_LINES)) * 2
    for i in range(max_iters):
        step = game.carnot_step(state, i)
        steps.append(step)
        state = step.state
        if step.is_solved:
            break
    return steps


# ---------------------------------------------------------------------------
# REQ-GTW-001: scenario count
# ---------------------------------------------------------------------------


def test_scenario_count_at_least_20() -> None:
    """WAR_SCENARIOS must contain at least 20 distinct named scenarios.

    The experiment spec requires >= 20; fewer would make the frantic CRT
    cycling effect unconvincing as a theatrical demo.
    """
    assert len(WAR_SCENARIOS) >= 20, f"Expected >= 20 scenarios, got {len(WAR_SCENARIOS)}"
    # All names must be non-empty strings
    for s in WAR_SCENARIOS:
        assert isinstance(s, str) and s.strip(), f"Blank/non-str scenario: {s!r}"


# ---------------------------------------------------------------------------
# REQ-GTW-002: energy strictly decreasing
# ---------------------------------------------------------------------------


def test_energy_strictly_decreasing() -> None:
    """Each carnot_step must reduce energy by exactly 1/total_steps.

    This ensures the shell animation engine fires a UI yield on every
    step (it yields whenever energy drops), giving the visitor the
    frantic CRT cycling effect.
    """
    game = GlobalThermonuclearWarGame()
    state = game.initial_state()
    prev_energy = game.energy(state)
    assert prev_energy == 1.0, "Initial energy must be 1.0"

    total_steps = len(WAR_SCENARIOS) + len(REVEAL_LINES)
    for i in range(total_steps):
        step = game.carnot_step(state, i)
        state = step.state
        assert step.energy < prev_energy, (
            f"Energy did not decrease at step {i}: {prev_energy} -> {step.energy}"
        )
        prev_energy = step.energy

    assert prev_energy == 0.0, f"Final energy must be 0.0, got {prev_energy}"


# ---------------------------------------------------------------------------
# REQ-GTW-003: phase transitions to REVEAL after all scenarios
# ---------------------------------------------------------------------------


def test_phase_transitions_computing_to_reveal() -> None:
    """After the last scenario is processed, phase must be REVEAL.

    We drive the game through exactly len(WAR_SCENARIOS) steps and
    inspect the resulting state — it should be in the REVEAL phase
    waiting to typewriter-reveal the first conclusion line.
    """
    game = GlobalThermonuclearWarGame()
    state = game.initial_state()
    for i in range(len(WAR_SCENARIOS)):
        step = game.carnot_step(state, i)
        state = step.state

    assert state.phase == "REVEAL", (
        f"Expected phase REVEAL after {len(WAR_SCENARIOS)} steps, got {state.phase!r}"
    )
    assert len(state.scenarios_computed) == len(WAR_SCENARIOS)


# ---------------------------------------------------------------------------
# REQ-GTW-004: typewriter reveal emits REVEAL_LINES in order
# ---------------------------------------------------------------------------


def test_typewriter_reveal_order() -> None:
    """The three reveal lines must appear in annotation in the correct order.

    Annotations during the REVEAL phase must match REVEAL_LINES[0],
    REVEAL_LINES[1], REVEAL_LINES[2] in sequence.
    """
    game = GlobalThermonuclearWarGame()
    state = game.initial_state()

    # Fast-forward through all COMPUTING steps
    for i in range(len(WAR_SCENARIOS)):
        step = game.carnot_step(state, i)
        state = step.state

    # Now collect the three REVEAL annotations
    reveal_annotations = []
    for i in range(len(REVEAL_LINES)):
        step = game.carnot_step(state, len(WAR_SCENARIOS) + i)
        reveal_annotations.append(step.annotation)
        state = step.state

    assert reveal_annotations == REVEAL_LINES, (
        f"Reveal annotations mismatch: {reveal_annotations!r} != {REVEAL_LINES!r}"
    )


# ---------------------------------------------------------------------------
# REQ-GTW-005: is_solved only True after all reveal lines
# ---------------------------------------------------------------------------


def test_is_solved_only_after_full_reveal() -> None:
    """is_solved must remain False until the final reveal line is emitted.

    Premature is_solved would cause the shell to stop animating before
    the visitor sees the iconic quote — the cartridge's entire purpose.
    """
    game = GlobalThermonuclearWarGame()
    state = game.initial_state()
    total_steps = len(WAR_SCENARIOS) + len(REVEAL_LINES)

    for i in range(total_steps - 1):
        step = game.carnot_step(state, i)
        assert not step.is_solved, (
            f"is_solved prematurely True at step {i} (expected only at step {total_steps - 1})"
        )
        state = step.state

    # Final step
    final_step = game.carnot_step(state, total_steps - 1)
    assert final_step.is_solved, "is_solved must be True on the final step"
    assert final_step.energy == 0.0, f"Final energy must be 0.0, got {final_step.energy}"


# ---------------------------------------------------------------------------
# REQ-GTW-006: exact step count
# ---------------------------------------------------------------------------


def test_completes_in_exact_step_count() -> None:
    """Full run must complete in exactly N_scenarios + N_reveal_lines steps.

    Fewer steps means the reveal was skipped; more means there is a
    state machine bug causing extra no-op steps.
    """
    game = GlobalThermonuclearWarGame()
    steps = _run_to_completion(game)
    expected = len(WAR_SCENARIOS) + len(REVEAL_LINES)
    assert len(steps) == expected, f"Expected {expected} steps, completed in {len(steps)}"


# ---------------------------------------------------------------------------
# REQ-GTW-007: visualize returns HTML with key elements
# ---------------------------------------------------------------------------


def test_visualize_contains_progress_bar() -> None:
    """visualize must include the SCENARIOS progress bar during COMPUTING phase."""
    game = GlobalThermonuclearWarGame()
    state = game.initial_state()

    # Run a few computing steps so there are scenarios to display
    for i in range(5):
        step = game.carnot_step(state, i)
        state = step.state

    html = game.visualize(state, game.energy(state))
    assert "SCENARIOS:" in html, "Progress bar label missing from visualize output"
    assert "MUTUAL ASSURED DESTRUCTION" in html, (
        "Computed scenario outcomes must appear in visualize output"
    )


def test_visualize_shows_reveal_lines() -> None:
    """visualize must display revealed lines once the REVEAL phase begins."""
    game = GlobalThermonuclearWarGame()
    state = game.initial_state()

    # Fast-forward through all scenarios + first reveal line
    for i in range(len(WAR_SCENARIOS) + 1):
        step = game.carnot_step(state, i)
        state = step.state

    html = game.visualize(state, game.energy(state))
    assert REVEAL_LINES[0] in html, (
        f"First reveal line {REVEAL_LINES[0]!r} missing from HTML after REVEAL starts"
    )


def test_gtw_energy_standalone() -> None:
    """The standalone gtw_energy function must agree with game.energy."""
    game = GlobalThermonuclearWarGame()
    state = game.initial_state()
    assert gtw_energy(state) == game.energy(state) == 1.0

    for i in range(len(WAR_SCENARIOS) + len(REVEAL_LINES)):
        step = game.carnot_step(state, i)
        state = step.state
        assert abs(gtw_energy(state) - game.energy(state)) < 1e-9
