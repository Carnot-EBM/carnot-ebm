"""REQ-INFRA-6820: scope declarations are enforced, and conflicts resolve by wound-wait.

Until 2026-08-29 a declaration bound only its author. It said what THAT session could stage and
did nothing about a stranger staging the same path -- advisory in exactly the direction that
hurt. A shared index took one agent's staged work into another's commit three times in one day,
once taking the worktree guard while it was being written.

Enforcement was not simply switched on, because it had been tried and removed. Cross-session
judging deadlocked: A declares a.py, B declares b.py, B stages its own b.py, and A's record
refuses it. Wound-wait supplies the missing ordering -- oldest declaration wins, ties break on
run id -- so the order is total and a cycle cannot form.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from harness_integrity_lint import (  # noqa: E402
    CLAIM_STALE_AFTER,
    claim_owner,
    contested_paths,
    is_universal_pattern,
    overlapping_older_claims,
)

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def rec(run_id: str, hours_ago: float, *scope: str) -> dict:
    return {
        "run_id": run_id,
        "declared_at": (NOW - timedelta(hours=hours_ago)).isoformat(),
        "scope": list(scope),
    }


# --- the deadlock this replaces ---------------------------------------------------------------


def test_two_narrow_scopes_do_not_deadlock() -> None:
    """The 2026-08-29 incident, as a test.

    Each session staging only its OWN declared file must pass. Under the removed first-come
    rule each refused the other, which is why cross-session judging was switched off entirely.
    """
    records = [rec("a", 2, "a.py"), rec("b", 1, "b.py")]
    assert contested_paths(["b.py"], records, "b", NOW) == {}
    assert contested_paths(["a.py"], records, "a", NOW) == {}


def test_a_stranger_is_refused_and_the_owner_is_named() -> None:
    records = [rec("alpha", 2, "scripts/target.py")]
    assert contested_paths(["scripts/target.py"], records, "beta", NOW) == {
        "scripts/target.py": "alpha"
    }


def test_an_unclaimed_path_is_free() -> None:
    records = [rec("alpha", 2, "scripts/target.py")]
    assert contested_paths(["docs/other.md"], records, "beta", NOW) == {}


# --- wound-wait ordering ----------------------------------------------------------------------


def test_the_older_declaration_wins_regardless_of_arrival_order() -> None:
    older, younger = rec("older", 3, "x.py"), rec("young", 1, "x.py")
    assert claim_owner("x.py", [younger, older], NOW) == "older"
    assert claim_owner("x.py", [older, younger], NOW) == "older"


def test_ties_resolve_deterministically_so_both_sides_agree() -> None:
    """A rule where each side believes it won is worse than no rule."""
    one, two = rec("bbb", 2, "x.py"), rec("aaa", 2, "x.py")
    assert claim_owner("x.py", [one, two], NOW) == claim_owner("x.py", [two, one], NOW) == "aaa"


def test_an_undated_declaration_loses_to_a_dated_one() -> None:
    """A claim with no usable timestamp must not outrank one that has it."""
    undated = {"run_id": "undated", "scope": ["x.py"]}
    assert claim_owner("x.py", [undated, rec("dated", 0.1, "x.py")], NOW) == "dated"


# --- the two things that stop enforcement freezing the repository -------------------------------


def test_a_universal_pattern_never_blocks_a_stranger() -> None:
    """The conductor holds a STANDING `*` because it sweeps all uncommitted work.

    That states what it will COMMIT, not that nobody else may touch anything. Enforce `*`
    literally and no agent could stage a file again.
    """
    assert is_universal_pattern("*") and is_universal_pattern("**")
    assert not is_universal_pattern("scripts/*")
    assert contested_paths(["anything.py"], [rec("conductor", 5, "*")], "other", NOW) == {}


def test_a_universal_pattern_still_governs_its_own_author() -> None:
    """Advisory to strangers is not the same as void: the owner is still bound by its scope."""
    assert claim_owner("anything.py", [rec("conductor", 5, "*")], NOW) is None


def test_an_abandoned_claim_expires() -> None:
    """Agents die without releasing their scope. Without expiry the first crash locks a path.

    Fail toward NOT blocking: a wrongly-held claim stops real work with nobody to appeal to,
    a wrongly-released one costs attribution on one commit.
    """
    hours = CLAIM_STALE_AFTER.total_seconds() / 3600
    fresh = rec("ghost", hours - 0.5, "x.py")
    stale = rec("ghost", hours + 0.5, "x.py")
    assert contested_paths(["x.py"], [fresh], "live", NOW) == {"x.py": "ghost"}
    assert contested_paths(["x.py"], [stale], "live", NOW) == {}


def test_an_unidentified_committer_is_not_frozen_out() -> None:
    """Deliberately the opposite of the seal rule, which holds an unknown session to all of them.

    Applied here, a missing environment variable would become a repository-wide freeze: no
    commit could touch any claimed path while any claim stood.
    """
    assert contested_paths(["x.py"], [rec("alpha", 2, "x.py")], None, NOW) == {}


# --- the declaration-time warning ---------------------------------------------------------------


def test_a_younger_glob_is_warned_about_an_older_literal() -> None:
    """`scripts/*` overlaps an existing claim on `scripts/target.py`.

    A first version probed by substituting "x" for wildcards and found NOTHING: `scripts/*`
    became `scripts/x`, which the literal it collides with does not match. Probing runs both
    directions now.
    """
    mine = rec("beta", 1, "scripts/*")
    older = rec("alpha", 2, "scripts/target.py")
    assert overlapping_older_claims(["scripts/*"], mine, [older], NOW) == {"scripts/*": "alpha"}


def test_a_younger_literal_is_warned_about_an_older_glob() -> None:
    mine = rec("beta", 1, "scripts/target.py")
    older = rec("alpha", 2, "scripts/*")
    assert overlapping_older_claims(["scripts/target.py"], mine, [older], NOW) == {
        "scripts/target.py": "alpha"
    }


def test_an_older_declaration_is_not_warned_about_a_younger_one() -> None:
    """Only the side that will actually be refused is told."""
    mine = rec("alpha", 3, "scripts/*")
    younger = rec("beta", 1, "scripts/target.py")
    assert overlapping_older_claims(["scripts/*"], mine, [younger], NOW) == {}


def test_disjoint_scopes_produce_no_warning() -> None:
    mine = rec("beta", 1, "docs/*")
    older = rec("alpha", 2, "scripts/target.py")
    assert overlapping_older_claims(["docs/*"], mine, [older], NOW) == {}
