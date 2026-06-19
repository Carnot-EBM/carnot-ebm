"""lp85 L6 mechanic unit model: button clicks permute logical piece slots.

This is intentionally tiny. The Exp 4394 aggregate gate only said the model was
wrong somewhere; Exp 4405 names the transition we can now test directly.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def apply_button_permutation(slots: Sequence[T], permutation: Sequence[int]) -> tuple[T, ...]:
    """Return the next logical slot order after a discovered lp85 button permutation."""

    if len(slots) != len(permutation):
        raise ValueError("slots and permutation must have equal length")
    return tuple(slots[index] for index in permutation)


def transition_fixture() -> dict[str, object]:
    """Executable fixture for the localized L6 mismatch."""

    before = ("bghvgbtwcb", "fdgmtkfrxl", "slot2", "slot3")
    permutation = (2, 0, 3, 1)
    expected = ("slot2", "bghvgbtwcb", "slot3", "fdgmtkfrxl")
    observed = apply_button_permutation(before, permutation)
    return {
        "transition": "lp85:L6:button_permutation_slot_mapping",
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def adaptive_trace_fixture_4415() -> dict[str, object]:
    before = ("bghvgbtwcb", "fdgmtkfrxl", "slot2", "slot3")
    permutation = (2, 0, 3, 1)
    expected = ("slot2", "bghvgbtwcb", "slot3", "fdgmtkfrxl")
    observed = apply_button_permutation(before, permutation)
    return {
        "adaptive_tests": [
            {
                "name": "lp85_adaptive_round1_permutation_from_wrong_slot_trace",
                "round": 1,
                "source_failing_transition": "lp85:L6:rollout_wrong_button_permutation",
                "derived_from_rollout_trace": True,
                "fresh_agent_state": True,
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
                "residual_behavior_after_test": "lp85_l6_search_not_yet_reproduced_after_permutation_repair",
            },
            {
                "name": "lp85_adaptive_round2_reproduction_search_residual",
                "round": 2,
                "source_failing_transition": "lp85:L6:fresh_agent_bfs_still_reaches_l5",
                "derived_from_rollout_trace": True,
                "fresh_agent_state": True,
                "expected": "offline_reproduced_l6",
                "observed": "offline_reproduced_l5",
                "passed": False,
                "residual_behavior_after_test": "lp85_l6_button_permutation_search_reproduction_still_wrong",
            },
        ]
    }
