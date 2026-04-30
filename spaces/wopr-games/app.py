"""WOPR Games — Gradio Space entry point.

Single-input UX: the terminal is the only surface. Users type
commands like ``LOAD SUDOKU`` or ``PLAY GLOBAL THERMONUCLEAR WAR``;
easter eggs (``JOSHUA``, ``CHESS``, ``HELP``…) work in the same
input. The right panel renders the live game state. There is no
dropdown — selecting a cartridge IS the act of typing its name.

Run locally:
    python app.py

On HuggingFace Space: this file is the entry point. The rest of the
package ships alongside (``games/``, ``wopr_shell.py``, etc.).
"""

from __future__ import annotations

import gradio as gr
from games import ALL_GAMES
from wopr_shell import (
    BOOT_LINES,
    match_cartridge_load,
    render_energy_bar,
    respond_to_terminal_input,
)

# ---------------------------------------------------------------------------
# WOPR-themed CSS — phosphor green on black, blink, glow
# ---------------------------------------------------------------------------

WOPR_CSS = """
.gradio-container {
    background: #000 !important;
    color: #39ff14 !important;
    font-family: 'JetBrains Mono', 'VT323', 'Courier New', monospace !important;
}
body {
    background: #000 !important;
}
h1, h2, h3, h4, h5, p, span, label {
    color: #39ff14 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.wopr-cursor {
    color: #39ff14;
    animation: wopr-blink 0.8s infinite;
}
@keyframes wopr-blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
}
button {
    background: #001a00 !important;
    color: #39ff14 !important;
    border: 1px solid #39ff14 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
button:hover {
    background: #003a00 !important;
    box-shadow: 0 0 12px rgba(57, 255, 20, 0.5) !important;
}
input, textarea, select {
    background: #000 !important;
    color: #39ff14 !important;
    border: 1px solid #39ff14 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
"""


# ---------------------------------------------------------------------------
# Terminal rendering helpers
# ---------------------------------------------------------------------------


def _wrap_terminal(inner_html: str) -> str:
    """Wrap inner content in the WOPR CRT-terminal frame."""
    return (
        '<div style="background:#000;border:2px solid #39ff14;'
        "padding:16px 20px;min-height:340px;max-height:520px;overflow-y:auto;"
        "font-family:JetBrains Mono,Courier New,monospace;"
        "box-shadow:0 0 24px rgba(57,255,20,0.25);"
        'border-radius:4px;">' + inner_html + "</div>"
    )


def _render_history(history: list[dict]) -> str:
    """Render the running terminal transcript as HTML.

    Each entry is a dict {role: 'user'|'wopr'|'system', text: str}.
    Last 80 lines kept to avoid runaway DOM.
    """
    lines = []
    for entry in history[-80:]:
        role = entry.get("role", "wopr")
        text = entry.get("text", "")
        if role == "user":
            lines.append(
                f'<div style="color:#9aff9a;font-family:JetBrains Mono,monospace;'
                f'font-size:13px;line-height:1.4;">&gt; {text}</div>'
            )
        elif role == "system":
            lines.append(
                f'<div style="color:#5aff5a;font-family:JetBrains Mono,monospace;'
                f'font-size:12px;line-height:1.4;font-style:italic;">'
                f"[{text}]</div>"
            )
        else:  # wopr
            lines.append(
                f'<div style="color:#39ff14;font-family:JetBrains Mono,monospace;'
                f'font-size:13px;line-height:1.4;">{text}</div>'
            )
    return _wrap_terminal("".join(lines))


def _initial_history() -> list[dict]:
    """Boot-sequence transcript shown on page load."""
    history: list[dict] = []
    for line in BOOT_LINES:
        if not line:
            continue
        history.append({"role": "wopr", "text": line})
    history.append(
        {
            "role": "system",
            "text": (
                "TYPE A COMMAND. TRY: LIST GAMES  |  PLAY SUDOKU  |  "
                "JOSHUA  |  CHESS  |  GLOBAL THERMONUCLEAR WAR"
            ),
        }
    )
    return history


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------


def boot_html() -> str:
    """Static initial terminal contents."""
    return _render_history(_initial_history())


def handle_terminal(text: str, history: list[dict]):
    """Single input handler.

    - If the input matches a cartridge load command, stream the solve.
    - Otherwise treat as easter egg / help / unknown.

    Yields (terminal_html, viz_html, energy_bar_html, history_state, input_clear).
    Generator: streams multiple yields during cartridge solve, single yield
    for easter eggs.
    """
    text = text.strip()
    if not text:
        yield (_render_history(history), gr.update(), gr.update(), history, "")
        return

    # Append the user's typed line to the transcript right away.
    history = list(history) + [{"role": "user", "text": text}]

    # 1a. AGAIN / REPLAY / RETRY → re-run the last cartridge in history.
    if text.strip().lower() in {"again", "replay", "retry", "another", "more"}:
        last_cart = None
        for entry in reversed(history):
            stash = entry.get("_cartridge")
            if stash:
                last_cart = stash
                break
        if last_cart is None:
            history.append(
                {
                    "role": "wopr",
                    "text": "NO PRIOR CARTRIDGE LOADED. TYPE 'LIST GAMES' TO BEGIN.",
                }
            )
            yield (_render_history(history), gr.update(), gr.update(), history, "")
            return
        # Reuse the cartridge name as if the user typed it.
        text = last_cart

    # 1b. Try cartridge load
    cartridge = match_cartridge_load(text, ALL_GAMES)
    if cartridge is not None:
        # Stash the cartridge name so AGAIN / REPLAY can find it later.
        history.append(
            {
                "role": "wopr",
                "text": f"LOADING CARTRIDGE: {cartridge.name}.",
                "_cartridge": cartridge.name,
            }
        )
        history.append(
            {
                "role": "wopr",
                "text": f"&gt; {cartridge.description}",
            }
        )
        # WarGames immersion: David asked Joshua "is this a game or is it
        # real?" and Joshua replied "WHAT'S THE DIFFERENCE?" — but the
        # iconic line of cartridge selection in the film is when WOPR
        # asks "NUMBER OF PLAYERS:" and David types 0, signalling that
        # Joshua plays itself. Echoing that on every cartridge load.
        history.append({"role": "wopr", "text": "NUMBER OF PLAYERS: 0"})
        history.append({"role": "wopr", "text": "JOSHUA WILL PLAY ALONE."})
        history.append({"role": "system", "text": "CARNOT MINIMISATION ENGAGED."})

        # Stream the solve, appending each step's annotation to history.
        state = cartridge.initial_state()
        initial_energy = cartridge.energy(state)
        last_energy = initial_energy
        max_iterations = 5000
        yield_every = 25

        # First frame: show initial state
        yield (
            _render_history(history),
            cartridge.visualize(state, initial_energy),
            render_energy_bar(
                initial_energy,
                max(initial_energy, 1.0),
                accent=cartridge.accent_color,
            ),
            history,
            "",
        )

        for iteration in range(max_iterations):
            step = cartridge.carnot_step(state, iteration)
            state = step.state
            should_yield = (
                iteration % yield_every == 0 or step.is_solved or step.energy < last_energy
            )
            if should_yield:
                history.append(
                    {
                        "role": "wopr",
                        "text": f"[ITER {iteration:04d}] {step.annotation}",
                    }
                )
                yield (
                    _render_history(history),
                    cartridge.visualize(state, step.energy),
                    render_energy_bar(
                        step.energy,
                        max(initial_energy, 1.0),
                        accent=cartridge.accent_color,
                    ),
                    history,
                    "",
                )
                last_energy = step.energy
            if step.is_solved:
                break

        final_energy = cartridge.energy(state)
        if cartridge.is_solved(state):
            history.append(
                {
                    "role": "wopr",
                    "text": (f"SOLVED AT ITER {iteration}. FINAL ENERGY: {final_energy:.2f}."),
                }
            )
        else:
            history.append(
                {
                    "role": "wopr",
                    "text": (f"ITER LIMIT REACHED. FINAL ENERGY: {final_energy:.2f}."),
                }
            )
        history.append(
            {
                "role": "system",
                "text": (
                    "READY. TYPE 'AGAIN' TO REPLAY  |  "
                    "'LIST GAMES' FOR OTHER CARTRIDGES  |  "
                    "ANY OTHER COMMAND TO CONTINUE."
                ),
            }
        )
        yield (
            _render_history(history),
            cartridge.visualize(state, final_energy),
            render_energy_bar(
                final_energy,
                max(initial_energy, 1.0),
                accent=cartridge.accent_color,
            ),
            history,
            "",
        )
        return

    # 2. Easter egg / help / unknown
    response = respond_to_terminal_input(text)
    history.append({"role": "wopr", "text": response})
    yield (_render_history(history), gr.update(), gr.update(), history, "")


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="WOPR Games — Carnot Verifier Demos",
        css=WOPR_CSS,
        theme=gr.themes.Base(),
    ) as app:
        gr.HTML(
            '<h1 style="color:#39ff14;font-family:JetBrains Mono,monospace;'
            'text-align:center;letter-spacing:0.2em;">W.O.P.R.</h1>'
            '<p style="color:#9aff9a;font-family:JetBrains Mono,monospace;'
            'text-align:center;font-size:13px;">'
            "WAR OPERATION PLAN RESPONSE  |  CARNOT ENERGY-BASED VERIFIER  |  "
            "JOSHUA SUBSYSTEM ONLINE"
            "</p>"
        )

        history_state = gr.State(_initial_history())

        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML(
                    '<h3 style="color:#39ff14;font-family:JetBrains Mono,monospace;">TERMINAL</h3>'
                )
                terminal_display = gr.HTML(value=boot_html())

            with gr.Column(scale=1):
                gr.HTML(
                    '<h3 style="color:#39ff14;font-family:JetBrains Mono,monospace;">'
                    "GAME STATE"
                    "</h3>"
                )
                game_viz = gr.HTML(value="")
                energy_bar = gr.HTML(value="")

        with gr.Row():
            terminal_input = gr.Textbox(
                label="",
                placeholder="> _",
                lines=1,
                interactive=True,
                scale=5,
                show_label=False,
            )
            submit_button = gr.Button("ENTER", variant="primary", scale=1)

        outputs = [terminal_display, game_viz, energy_bar, history_state, terminal_input]
        submit_button.click(
            fn=handle_terminal,
            inputs=[terminal_input, history_state],
            outputs=outputs,
        )
        terminal_input.submit(
            fn=handle_terminal,
            inputs=[terminal_input, history_state],
            outputs=outputs,
        )

        gr.HTML(
            '<div style="color:#5aff5a;font-family:JetBrains Mono,monospace;'
            'font-size:11px;text-align:center;margin-top:24px;">'
            "CARNOT EBM PROJECT  |  ENERGY-BASED CONSTRAINT VERIFICATION  |  "
            '<a href="https://huggingface.co/Carnot-EBM" '
            'style="color:#39ff14;">huggingface.co/Carnot-EBM</a>'
            "</div>"
        )

    return app


if __name__ == "__main__":
    build_app().launch()
