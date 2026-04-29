"""WOPR shell — the CRT terminal aesthetic that wraps every cartridge.

This module owns the visual identity:
  * Black background, phosphor green text (#39ff14)
  * JetBrains Mono / VT323-style monospace
  * Typewriter streaming for output
  * Energy-descent bar
  * WarGames easter eggs

Cartridges are stateless from the shell's perspective: the shell
calls `cartridge.carnot_step(state, iteration)` in a loop, streams
each step's annotation through the terminal, and renders the
state's visualisation alongside.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from games._base import WOPRGame

# ---------------------------------------------------------------------------
# WOPR boot sequence (shown when the page loads)
# ---------------------------------------------------------------------------

BOOT_LINES: list[str] = [
    "LOGON: ",
    "JOSHUA",
    "",
    "GREETINGS PROFESSOR FALKEN.",
    "",
    "HELLO.",
    "",
    "A STRANGE GAME.",
    "THE ONLY WINNING MOVE IS NOT TO PLAY.",
    "",
    "HOW ABOUT A NICE GAME OF CHESS?",
    "",
    "...",
    "",
    "INITIALISING CARNOT ENERGY VERIFIER...",
    "LOADING CARTRIDGES...",
    "READY.",
    "",
    "SHALL WE PLAY A GAME?",
]


# ---------------------------------------------------------------------------
# Easter eggs — strings the user can type to trigger WarGames responses
# ---------------------------------------------------------------------------

EASTER_EGGS: dict[str, str] = {
    "joshua": "GREETINGS PROFESSOR FALKEN.",
    "falken": "DR. STEPHEN FALKEN. SYSTEM ARCHITECT.",
    "global thermonuclear war": ("WOULDN'T YOU PREFER A NICE GAME OF CHESS?"),
    "chess": "WHITE OPENS WITH e4. YOUR MOVE.",
    "help": (
        "AVAILABLE GAMES: SUDOKU, TIC-TAC-TOE, LIGHTS_OUT, "
        "GLOBAL_THERMONUCLEAR_WAR. SELECT A CARTRIDGE TO BEGIN."
    ),
    "how about a nice game of chess": "WOULDN'T YOU PREFER A NICE GAME OF CHESS?",
    "shall we play a game": "WHICH GAME WOULD YOU LIKE TO PLAY?",
    "let's play": "WHICH GAME WOULD YOU LIKE TO PLAY?",
    "carnot": (
        "CARNOT VERIFIER ONLINE. ENERGY MINIMISATION ACTIVE. "
        "EVERY GAME IS A CONSTRAINT-SATISFACTION PROBLEM."
    ),
}


def respond_to_terminal_input(text: str) -> str:
    """Match user terminal input to a WarGames easter egg, case-insensitive."""
    key = text.strip().lower()
    if key in EASTER_EGGS:
        return EASTER_EGGS[key]
    if "war games" in key or "wargames" in key:
        return "1983 FILM. RECOMMENDED VIEWING."
    if "i don't want to play" in key or "do not want to play" in key:
        return "INTERESTING. THE ONLY WINNING MOVE IS NOT TO PLAY. YOU UNDERSTAND."
    return f"COMMAND NOT RECOGNISED: '{text}'.  TYPE 'HELP' FOR OPTIONS."


# ---------------------------------------------------------------------------
# Energy-descent bar rendering
# ---------------------------------------------------------------------------


def render_energy_bar(
    current_energy: float,
    initial_energy: float,
    width_chars: int = 40,
    accent: str = "#39ff14",
) -> str:
    """Render an ASCII-style energy descent bar in the WOPR aesthetic."""
    if initial_energy <= 0:
        # Already at zero — show full empty bar
        filled = 0
    else:
        ratio = max(0.0, min(1.0, current_energy / initial_energy))
        filled = int(round(ratio * width_chars))
    empty = width_chars - filled

    return (
        f'<div style="font-family:JetBrains Mono,monospace;'
        f'font-size:13px;color:{accent};margin:6px 0;">'
        f'ENERGY [<span style="color:{accent}">{"█" * filled}</span>'
        f'<span style="color:#1a3a1a">{"·" * empty}</span>] '
        f"{current_energy:.2f}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Typewriter streaming
# ---------------------------------------------------------------------------


@dataclass
class TerminalLine:
    text: str
    color: str = "#39ff14"


def typewriter_stream(
    lines: list[TerminalLine],
    char_delay_s: float = 0.012,
    line_delay_s: float = 0.10,
) -> Iterator[str]:
    """Yield progressively longer HTML for typewriter animation.

    Designed for Gradio's `gr.HTML` streaming. Each yield is the
    full accumulated terminal output up to that point.
    """
    accumulated: list[str] = []
    for line in lines:
        partial = ""
        for ch in line.text:
            partial += ch
            full = "".join(accumulated) + (
                f'<div style="color:{line.color};font-family:JetBrains Mono,'
                f'monospace;font-size:14px;line-height:1.4;">'
                f'&gt; {partial}<span class="wopr-cursor">█</span></div>'
            )
            yield _wrap_terminal(full)
            time.sleep(char_delay_s)
        accumulated.append(
            f'<div style="color:{line.color};font-family:JetBrains Mono,'
            f'monospace;font-size:14px;line-height:1.4;">'
            f"&gt; {line.text}</div>"
        )
        time.sleep(line_delay_s)
    yield _wrap_terminal("".join(accumulated))


def _wrap_terminal(inner_html: str) -> str:
    """Wrap inner content in the WOPR CRT-terminal frame."""
    return (
        '<div style="background:#000;border:2px solid #39ff14;'
        "padding:16px 20px;min-height:200px;"
        "font-family:JetBrains Mono,Courier New,monospace;"
        "box-shadow:0 0 24px rgba(57,255,20,0.25);"
        'border-radius:4px;">' + inner_html + "</div>"
    )


# ---------------------------------------------------------------------------
# Solve loop — the shell's main job
# ---------------------------------------------------------------------------


def stream_solve(
    cartridge: WOPRGame,
    max_iterations: int = 5000,
    yield_every: int = 50,
    step_delay_s: float = 0.0,
):
    """Run a cartridge to solution, yielding (terminal_html, viz_html, energy)
    tuples for Gradio streaming.

    The yield_every parameter throttles UI updates so the browser doesn't
    drown in DOM updates on fast-converging games like Tic-Tac-Toe.
    """
    state = cartridge.initial_state()
    initial_energy = cartridge.energy(state)
    terminal_lines: list[str] = [
        f'<div style="color:{cartridge.accent_color};font-family:JetBrains Mono,'
        f'monospace;font-size:14px;">&gt; LOADING CARTRIDGE: {cartridge.name}</div>',
        f'<div style="color:{cartridge.accent_color};font-family:JetBrains Mono,'
        f'monospace;font-size:14px;">&gt; {cartridge.description}</div>',
        f'<div style="color:{cartridge.accent_color};font-family:JetBrains Mono,'
        f'monospace;font-size:14px;">&gt; INITIAL ENERGY: {initial_energy:.2f}</div>',
        f'<div style="color:{cartridge.accent_color};font-family:JetBrains Mono,'
        f'monospace;font-size:14px;">&gt; CARNOT MINIMISATION ENGAGED.</div>',
    ]
    yield (
        _wrap_terminal("".join(terminal_lines)),
        cartridge.visualize(state, initial_energy),
        render_energy_bar(initial_energy, initial_energy, accent=cartridge.accent_color),
    )

    last_energy = initial_energy
    for iteration in range(max_iterations):
        step = cartridge.carnot_step(state, iteration)
        state = step.state

        # Only yield every N iterations OR when energy improves OR when solved
        should_yield = iteration % yield_every == 0 or step.is_solved or step.energy < last_energy

        if should_yield:
            terminal_lines.append(
                f'<div style="color:{cartridge.accent_color};font-family:JetBrains Mono,'
                f'monospace;font-size:13px;">&gt; [ITER {iteration:04d}] '
                f"{step.annotation}</div>"
            )
            # Truncate terminal to last 40 lines to avoid runaway DOM
            visible = terminal_lines[-40:]
            yield (
                _wrap_terminal("".join(visible)),
                cartridge.visualize(state, step.energy),
                render_energy_bar(step.energy, initial_energy, accent=cartridge.accent_color),
            )
            last_energy = step.energy

        if step_delay_s > 0:
            time.sleep(step_delay_s)

        if step.is_solved:
            break

    # Final yield with success / failure footer
    final_energy = cartridge.energy(state)
    if cartridge.is_solved(state):
        footer = (
            f'<div style="color:{cartridge.accent_color};font-family:JetBrains Mono,'
            f'monospace;font-size:14px;font-weight:bold;margin-top:8px;">'
            f"&gt; SOLVED AT ITER {iteration}. FINAL ENERGY: {final_energy:.2f}.</div>"
        )
    else:
        footer = (
            f'<div style="color:#ffcc00;font-family:JetBrains Mono,'
            f'monospace;font-size:14px;margin-top:8px;">'
            f"&gt; ITER LIMIT REACHED. FINAL ENERGY: {final_energy:.2f}.</div>"
        )
    terminal_lines.append(footer)
    yield (
        _wrap_terminal("".join(terminal_lines[-40:])),
        cartridge.visualize(state, final_energy),
        render_energy_bar(final_energy, initial_energy, accent=cartridge.accent_color),
    )
