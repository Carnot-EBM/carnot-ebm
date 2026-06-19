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


def adaptive_trace_fixture_4415() -> dict[str, object]:
    before = EditorState(30, 13, 1, 0, 11)
    expected = EditorState(30, 33, 1, 0, 11)
    observed = simulate(before, [DOWN, DOWN, DOWN, DOWN, DOWN], EditorGeometry())
    up_clicks = slot_bit_clicks((24, 41), 33)
    return {
        "adaptive_tests": [
            {
                "name": "tn36_adaptive_round1_run_button_executes_program",
                "round": 1,
                "source_failing_transition": "tn36:L8:rollout_sxhtkytekm_program_did_not_move",
                "derived_from_rollout_trace": True,
                "fresh_agent_state": True,
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
                "residual_behavior_after_test": "tn36_l8_palette_population_or_later_program_state_still_wrong",
            },
            {
                "name": "tn36_adaptive_round2_palette_population_residual",
                "round": 2,
                "source_failing_transition": "tn36:L8:fresh_agent_palette_population_still_wrong",
                "derived_from_rollout_trace": True,
                "fresh_agent_state": True,
                "expected": "palette_state_tracks_real_l8_rollout",
                "observed": {"bit_clicks_for_up_33": up_clicks, "stateful_palette_model": "not_encoded"},
                "passed": False,
                "residual_behavior_after_test": "tn36_l8_palette_population_or_later_program_state_still_wrong",
            },
        ]
    }
