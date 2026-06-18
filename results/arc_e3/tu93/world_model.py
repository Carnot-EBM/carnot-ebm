"""tu93 L5 mechanic unit model: branch search must evaluate on fresh env parity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BranchState:
    player: tuple[int, int]
    goal: tuple[int, int]
    reset_parity: int = 0


MOVES = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}


def fresh_env_branch_state(
    state: BranchState,
    action: int,
    *,
    reused_reset_count: int = 0,
) -> BranchState:
    """Apply a direction while pinning branch parity to the fresh-env value."""

    dx, dy = MOVES[action]
    x, y = state.player
    return BranchState((x + dx, y + dy), state.goal, reset_parity=0)


def reused_env_branch_state(state: BranchState, action: int, *, reused_reset_count: int) -> BranchState:
    """The rejected transition: reset parity leaks into branch evaluation."""

    dx, dy = MOVES[action]
    x, y = state.player
    return BranchState((x + dx, y + dy), state.goal, reset_parity=reused_reset_count % 2)


def transition_fixture() -> dict[str, object]:
    """Executable fixture for the localized fresh-env branch-mode mismatch."""

    before = BranchState((2, 2), (4, 2), reset_parity=1)
    observed = fresh_env_branch_state(before, 4, reused_reset_count=5)
    rejected = reused_env_branch_state(before, 4, reused_reset_count=5)
    expected = BranchState((3, 2), (4, 2), reset_parity=0)
    return {
        "transition": "tu93:L5:fresh_env_branch_move",
        "expected": expected,
        "observed": observed,
        "rejected_reused_env_observed": rejected,
        "passed": observed == expected and rejected != expected,
    }
