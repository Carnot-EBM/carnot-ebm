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
