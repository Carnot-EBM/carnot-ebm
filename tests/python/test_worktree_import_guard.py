"""REQ-INFRA-6810: a test run must not import `carnot` from a different checkout.

We adopted per-agent git worktrees on 2026-08-29 to stop concurrent agents destroying each
other's work. Worktrees close that class and open a quieter one: the venv resolves `import
carnot` to the MAIN checkout regardless of cwd, so a worktree run tests code the agent did not
edit and every mutation proof reads GREEN while measuring nothing.

Verified live before the guard was written: in a fresh worktree `carnot.__file__` pointed at the
main checkout, and an edit made in the worktree was invisible until `PYTHONPATH` was pinned.
"""

from __future__ import annotations

from pathlib import Path

from carnot.testing.worktree_import_guard import OVERRIDE_ENV, check, foreign_import_reason


def test_same_checkout_is_sound() -> None:
    assert foreign_import_reason(Path("/repo"), Path("/repo/python/carnot")) is None


def test_a_foreign_checkout_is_reported() -> None:
    """The incident shape: tests from the worktree, package from the main checkout."""
    reason = foreign_import_reason(Path("/wt"), Path("/repo/python/carnot"))
    assert reason is not None
    assert "/repo/python/carnot" in reason


def test_the_message_carries_the_fix() -> None:
    """A refusal an agent cannot act on just costs a run. Name the exact command."""
    reason = foreign_import_reason(Path("/wt"), Path("/repo/python/carnot"))
    assert reason is not None
    assert "PYTHONPATH=/wt/python" in reason


def test_a_sibling_prefix_is_not_mistaken_for_containment() -> None:
    """`/repo-other` must not read as inside `/repo`.

    A string-prefix comparison would call this sound and let the exact trap through.
    """
    assert foreign_import_reason(Path("/repo"), Path("/repo-other/python/carnot")) is not None


def test_check_raises_on_a_foreign_checkout() -> None:
    try:
        check(Path("/wt"), Path("/repo/python/carnot"))
    except RuntimeError as exc:
        assert "foreign checkout" in str(exc)
    else:  # pragma: no cover - the guard failing to fire is the whole bug
        raise AssertionError("check() accepted a foreign checkout")


def test_check_is_silent_when_the_trees_agree() -> None:
    check(Path("/repo"), Path("/repo/python/carnot"))


def test_the_override_is_honoured_and_is_exact(monkeypatch) -> None:
    """Deliberately testing an installed carnot is legitimate; drifting into it is not.

    Only the exact string "1" opts out, so a stray truthy value cannot silently disable the
    guard on the run that mattered.
    """
    monkeypatch.setenv(OVERRIDE_ENV, "1")
    check(Path("/wt"), Path("/repo/python/carnot"))
    monkeypatch.setenv(OVERRIDE_ENV, "yes")
    try:
        check(Path("/wt"), Path("/repo/python/carnot"))
    except RuntimeError:
        return
    raise AssertionError("a non-'1' value disabled the guard")


# --- The WIRING, not just the rule (added 2026-08-29 after an adversarial review) -------------
# The review replaced the `_check_worktree_import(...)` call in `tests/python/conftest.py` with
# `pass` and this file stayed 7/7 GREEN: the guard's rule was tested and its installation was
# not. A guard nothing calls is the bug class this repository keeps rediscovering.


def test_both_conftests_invoke_the_guard() -> None:
    """`testpaths` covers tests/python AND tests/integration; both must refuse.

    `tests/integration/test_install.py` imports carnot, so `pytest tests/integration` from an
    unpinned worktree was exactly the trap and was not refused -- the guard had been wired into
    one of the two places that needed it.
    """
    repo = Path(__file__).resolve().parents[2]
    for rel in ("tests/python/conftest.py", "tests/integration/conftest.py"):
        text = (repo / rel).read_text()
        assert "worktree_import_guard" in text, f"{rel} does not import the guard"
        assert "_check_worktree_import(" in text, f"{rel} imports the guard but never calls it"


def test_a_nested_worktree_or_clone_is_still_foreign() -> None:
    """Containment passed this; on this machine worktrees and clones live UNDER the main root.

    A worktree PYTHONPATH leaking into a main-checkout run is this trap with the trees swapped,
    and nested repo clones are not hypothetical here -- a scorer once swept two of them.
    """
    root = Path("/home/x/carnot")
    for nested in (
        root / ".claude/worktrees/agent-abc/python/carnot",
        root / "output/carnot-clone/python/carnot",
    ):
        assert foreign_import_reason(root, nested) is not None


def test_the_canonical_location_is_accepted() -> None:
    root = Path("/home/x/carnot")
    assert foreign_import_reason(root, root / "python" / "carnot") is None
