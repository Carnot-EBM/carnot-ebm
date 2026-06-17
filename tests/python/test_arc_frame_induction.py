"""Unit tests for FRAME-ONLY mechanic induction (scripts/arc3_frame_induction.py).

The detector must, from per-click frame-delta 'effects' alone (no internal state), (1) mask an
action-invariant HUD cell, (2) recognise a dense block of small local-toggle buttons as a
program-editor palette, and (3) NOT mis-classify a direct-control game (clicks that cause large
board changes) as an editor. These pin that logic on synthetic effects so it is game-independent.

Spec: REQ-PHASE4-081, SCENARIO-PHASE4-081 (the frame-only mechanic classifier that feeds the router).
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "arc3_frame_induction",
    Path(__file__).resolve().parents[2] / "scripts" / "arc3_frame_induction.py",
)
FI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FI)


def _editor_effects():
    """A 5-slot x 6-bit toggle-button palette (x in 19..43, y in 41..46) + a HUD cell at (61,1)
    that changes on EVERY click — the program-editor frame signature."""
    hud = (61, 1)
    effects = {}
    for cy in range(41, 47):
        for cx in range(19, 44):
            effects[(cx, cy)] = {hud, (cx, cy), (cx, cy - 1)}  # HUD + a small local toggle
    # a few inert cells elsewhere (only the HUD changes)
    for c in [(2, 2), (5, 50), (60, 30)]:
        effects[c] = {hud}
    return effects


def test_detects_program_editor_and_masks_hud():
    out = FI.induce(_editor_effects())
    assert out["mechanic"] == "program_editor"
    assert out["hud_cells"] == [(61, 1)]  # HUD masked (changed by ~every click)
    assert out["n_edit_buttons"] >= 100  # the dense palette
    bx0, by0, bx1, by1 = out["editor_bbox"]
    assert bx0 <= 19 and bx1 >= 43 and by0 <= 41 and by1 >= 46
    assert out["editor_density"] >= 0.5


def test_direct_control_game_is_not_an_editor():
    # A direct-control game: each click moves the avatar -> a LARGE board change (not a local
    # toggle). No dense local-toggle palette -> must NOT classify as program_editor.
    hud = (61, 1)
    big = {(x, y) for x in range(10, 40) for y in range(10, 30)}  # 600-cell change
    effects = {(cx, cy): {hud} | big for cy in range(0, 64, 8) for cx in range(0, 64, 8)}
    out = FI.induce(effects)
    assert out["mechanic"] == "unknown"


def test_empty_effects_is_unknown():
    out = FI.induce({})
    assert out["mechanic"] == "unknown"
    assert out["n_edit_buttons"] == 0


def test_resolve_toggles_finds_pattern_and_rejects_unreachable():
    # A slot codebook {toggle_pattern -> rendered glyph-bytes}: the blind winner-search resolves a
    # TARGET glyph back to the bit-row toggles that produce it, or None when the glyph is unreachable.
    book = {(0, 0): b"A", (1, 0): b"B", (0, 1): b"C", (1, 1): b"D"}
    assert FI._resolve_toggles(book, b"A") == (0, 0)  # reset glyph -> no toggles
    assert FI._resolve_toggles(book, b"D") == (1, 1)  # both bit-rows toggled
    assert FI._resolve_toggles(book, b"Z") is None  # unreachable glyph -> None (not a crash)
