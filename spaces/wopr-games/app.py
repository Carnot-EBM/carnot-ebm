"""WOPR Games — Gradio Space entry point.

Run locally:
    python -m spaces.wopr_games.app

Run on HuggingFace Space: this file is the entry point that
HuggingFace looks for. The rest of the package is shipped alongside
in `spaces/wopr-games/`.

The page presents a CRT-terminal aesthetic with one cartridge
selectable at a time. Each cartridge implements `WOPRGame`; the
shell handles all the WOPR flavour (boot sequence, terminal
streaming, energy bar, easter eggs).
"""

from __future__ import annotations

import gradio as gr
from games import ALL_GAMES
from wopr_shell import (
    BOOT_LINES,
    respond_to_terminal_input,
    stream_solve,
)

# ---------------------------------------------------------------------------
# WOPR-themed CSS — phosphor green on black, scanlines, glow
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
# Gradio handlers
# ---------------------------------------------------------------------------

GAMES_BY_NAME = {g.name: g for g in ALL_GAMES}


def boot_html() -> str:
    """Static boot-sequence HTML rendered on page load."""
    lines = []
    for line in BOOT_LINES:
        if not line:
            lines.append('<div style="height:6px;"></div>')
        else:
            lines.append(
                f'<div style="color:#39ff14;font-family:JetBrains Mono,monospace;'
                f'font-size:14px;line-height:1.4;">&gt; {line}</div>'
            )
    inner = "".join(lines)
    return (
        '<div style="background:#000;border:2px solid #39ff14;'
        "padding:16px 20px;min-height:200px;"
        "font-family:JetBrains Mono,Courier New,monospace;"
        "box-shadow:0 0 24px rgba(57,255,20,0.25);"
        'border-radius:4px;">' + inner + "</div>"
    )


def run_cartridge(game_name: str):
    """Generator: stream a cartridge solve to the UI."""
    cartridge = GAMES_BY_NAME.get(game_name)
    if cartridge is None:
        empty = f'<div style="color:#ff3939;">UNKNOWN CARTRIDGE: {game_name}</div>'
        yield empty, empty, ""
        return
    yield from stream_solve(cartridge, max_iterations=5000, yield_every=25)


def handle_terminal_input(text: str, history: list[tuple[str, str]]):
    """Respond to user terminal input with WarGames easter eggs."""
    response = respond_to_terminal_input(text)
    history = history + [(text, response)]
    transcript_html_lines = []
    for user_text, bot_text in history[-10:]:
        transcript_html_lines.append(
            f'<div style="color:#9aff9a;font-family:JetBrains Mono,monospace;'
            f'font-size:13px;">&gt; {user_text}</div>'
        )
        transcript_html_lines.append(
            f'<div style="color:#39ff14;font-family:JetBrains Mono,monospace;'
            f'font-size:13px;margin-bottom:6px;">{bot_text}</div>'
        )
    transcript_html = (
        '<div style="background:#000;border:1px solid #39ff14;padding:10px;'
        'min-height:80px;font-family:JetBrains Mono,monospace;">'
        + "".join(transcript_html_lines)
        + "</div>"
    )
    return transcript_html, history, ""


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
            cartridge_dropdown = gr.Dropdown(
                choices=[g.name for g in ALL_GAMES],
                value=ALL_GAMES[0].name,
                label="SELECT CARTRIDGE",
                interactive=True,
            )
            run_button = gr.Button("LOAD AND PLAY", variant="primary")

        with gr.Row(), gr.Column():
            gr.HTML(
                '<h3 style="color:#39ff14;font-family:JetBrains Mono,monospace;">'
                "TYPE A COMMAND  (try: JOSHUA, CHESS, GLOBAL THERMONUCLEAR WAR)"
                "</h3>"
            )
            terminal_history_state = gr.State([])
            with gr.Row():
                terminal_input = gr.Textbox(
                    label="",
                    placeholder="> ",
                    lines=1,
                    interactive=True,
                )
                submit_button = gr.Button("SEND", variant="secondary")
            terminal_response = gr.HTML(value="")

        # Wire it up
        run_button.click(
            fn=run_cartridge,
            inputs=[cartridge_dropdown],
            outputs=[terminal_display, game_viz, energy_bar],
        )

        submit_button.click(
            fn=handle_terminal_input,
            inputs=[terminal_input, terminal_history_state],
            outputs=[terminal_response, terminal_history_state, terminal_input],
        )
        terminal_input.submit(
            fn=handle_terminal_input,
            inputs=[terminal_input, terminal_history_state],
            outputs=[terminal_response, terminal_history_state, terminal_input],
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
