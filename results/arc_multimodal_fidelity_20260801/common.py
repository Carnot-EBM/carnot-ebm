"""Shared helpers for the ARC multimodal fidelity gate.

WHY THIS FILE EXISTS (verbose, for an engineer who is not an EBM/ARC specialist).
The ARC induce prompt hands the generator a 64x64 grid of small integers ("colour
indices", 0..15) serialised as run-length-encoded ASCII text. The shipped generator
gemma-4-31B-it is a MULTIMODAL model and the same repo publishes a vision projector,
so in principle we could hand it the grid as an IMAGE instead -- which is what the
grid natively is. Before doing any of that we have to answer one question, because
everything downstream depends on it:

    CAN THE VISION TOWER READ AN ARC COLOUR INDEX BACK EXACTLY?

ARC needs EXACT discrete values -- the induced engine is graded by exact-match on the
next grid, and it writes Python that indexes specific cells and compares to specific
integers. A vision encoder is lossy by construction. If it cannot tell colour index 4
from colour index 14, images would poison induction in a way that is invisible
downstream (you would see a slightly worse heldout accuracy, not a decode error).

That failure mode is NOT hypothetical in this codebase: `arc_executable_world_model
.to_ascii()` already collapses 4/14 and 5/15 today, because it renders each cell as
`str(int(v))[-1]` -- the last decimal digit only.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.request

import numpy as np

PORT = 18831
BASE = f"http://127.0.0.1:{PORT}"

# ---------------------------------------------------------------------------------
# THE PALETTE.
#
# The ARC-AGI-3 game environments in `environment_files/` work in colour INDICES only
# -- they never define RGB; the web UI does that. So this experiment must choose an
# index->RGB map, and that choice is load-bearing for how the result should be read.
#
# We deliberately choose a MAXIMALLY-SEPARATED, individually-nameable 16-colour
# palette. This makes the test CHARITABLE to the vision tower: these colours are far
# easier to tell apart than any real ARC palette, where several indices are close
# shades. Therefore the measured accuracy here is an UPPER BOUND on what the tower
# could achieve on the real palette. If it fails here, it fails a fortiori there --
# which is a sound one-directional argument and the reason this design is safe even
# though we could not recover the official RGB values.
#
# Every colour is given an unambiguous English name, and the name->index legend is
# put in the prompt. Without a legend the model cannot possibly answer, since nothing
# in an image says "this particular red means 14".
PALETTE: list[tuple[str, tuple[int, int, int]]] = [
    ("black", (0, 0, 0)),
    ("white", (255, 255, 255)),
    ("red", (230, 25, 75)),
    ("green", (60, 180, 75)),
    ("blue", (0, 90, 220)),
    ("yellow", (255, 225, 25)),
    ("orange", (245, 130, 48)),
    ("purple", (145, 30, 180)),
    ("cyan", (70, 240, 240)),
    ("magenta", (240, 50, 230)),
    ("lime", (190, 255, 0)),
    ("pink", (250, 190, 212)),
    ("teal", (0, 128, 128)),
    ("brown", (140, 70, 20)),
    ("navy", (0, 0, 128)),
    ("grey", (128, 128, 128)),
]
assert len(PALETTE) == 16
N_COLORS = 16

LEGEND = "\n".join(f"  {i} = {name}" for i, (name, _) in enumerate(PALETTE))


def render_png(grid: np.ndarray, px: int) -> bytes:
    """Render a colour-index grid to a PNG at `px` pixels per cell, nearest-neighbour.

    Nearest-neighbour (np.repeat) not interpolation: an ARC cell is a discrete symbol,
    so any smoothing at cell borders would invent colours that are not in the palette
    and would be OUR bug, not the tower's.
    """
    from PIL import Image

    g = np.asarray(grid)
    lut = np.array([rgb for _, rgb in PALETTE], dtype=np.uint8)
    rgb = lut[g]  # (h, w, 3)
    if px > 1:
        rgb = np.repeat(np.repeat(rgb, px, axis=0), px, axis=1)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def rle_grid(g: np.ndarray) -> str:
    """EXACTLY the shipped `_rle_grid` format from arc_executable_world_model.py.

    Copied verbatim (not imported) on purpose: that module is being edited by a
    concurrent session, and the negative control has to be pinned to a known-stable
    string format for the whole run. Format is
    'r<row>:<v0>x<n0>,<v1>x<n1>,...' with the column position IMPLICIT (running sum
    of prior run counts in that row).
    """
    g = np.asarray(g)
    h, w = g.shape
    lines = []
    for r in range(h):
        c = 0
        runs = []
        while c < w:
            v = g[r, c]
            c0 = c
            while c < w and g[r, c] == v:
                c += 1
            runs.append(f"{int(v)}x{c - c0}")
        lines.append(f"r{r}:" + ",".join(runs))
    return "\n".join(lines)


def chat(content_parts, max_tokens=3000, temperature=0.0, seed=20260801, timeout=2400):
    """One /v1/chat/completions call. temperature=0 so the measurement is deterministic.

    MAX_TOKENS MUST BE GENEROUS -- this is not a tuning knob, it is a correctness
    requirement, and getting it wrong silently returns "" for every question.
    gemma-4-31B-it is a THINKING model: llama.cpp routes its chain-of-thought into
    `reasoning_content` and only the post-CoT answer into `content`. With a small
    max_tokens the generation is cut off DURING the CoT, so `content` is "" and
    `finish_reason` is "length" -- which looks exactly like "the model had nothing to
    say about the image". The first run of this gate hit precisely that and produced
    0/4 on the liveness proof before the cause was found.

    Note there is NO `/no_think` escape here. That is a Qwen3 hybrid-thinking control
    token; the shipped code says so explicitly (`arc_executable_world_model.py`:
    `ARC_LIVE_GENERATOR_NO_THINK_PREFIX = ""  # /no_think is a Qwen3 token; inert on
    gemma-4`) and it was re-confirmed empirically here -- appending it made the CoT
    LONGER (189 tokens vs 46), not shorter. So the only fix is headroom.

    Returns (content, finish_reason, n_completion_tokens) so a caller can tell a real
    empty answer apart from a truncated one instead of scoring the truncation.
    """
    body = {
        "messages": [{"role": "user", "content": content_parts}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "cache_prompt": True,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    return (
        ch["message"].get("content") or "",
        ch.get("finish_reason", "?"),
        d.get("usage", {}).get("completion_tokens", -1),
    )


def text_part(s: str) -> dict:
    return {"type": "text", "text": s}


def image_part(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": data_uri(png)}}
