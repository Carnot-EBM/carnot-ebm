"""tn36 L8 mechanic unit model: program-editor bits plus sxhtkytekm run button."""

from __future__ import annotations

from carnot.agentic.arc_program_editor_model import EditorGeometry, EditorState, simulate

BIT_DY = 3
DOWN = 3


def slot_bit_clicks(slot_top: tuple[int, int], code: int) -> tuple[tuple[int, int], ...]:
    """Return the bit-toggle coordinates needed to set a six-bit program code."""

    x, y0 = slot_top
    return tuple((x, y0 + BIT_DY * bit) for bit in range(6) if (code >> bit) & 1)


def sxhtkytekm_center(button_box: tuple[int, int, int, int]) -> tuple[int, int]:
    """Return the executable object's discovered sub-button center."""

    x, y, width, height = button_box
    return (x + width // 2, y + height // 2)


def transition_fixture() -> dict[str, object]:
    """Executable fixture for the L8 program-editor control mismatch."""

    before = EditorState(30, 13, 1, 0, 11)
    expected = EditorState(30, 33, 1, 0, 11)
    observed = simulate(before, [DOWN, DOWN, DOWN, DOWN, DOWN], EditorGeometry())
    clicks = slot_bit_clicks((24, 41), 33)
    button = sxhtkytekm_center((32, 51, 9, 9))
    return {
        "transition": "tn36:L8:sxhtkytekm_program_editor_run",
        "expected": expected,
        "observed": observed,
        "bit_clicks_for_up_33": clicks,
        "sxhtkytekm_center": button,
        "passed": observed == expected and clicks == ((24, 41), (24, 56)) and button == (36, 55),
    }
