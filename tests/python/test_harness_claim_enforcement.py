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


# --- Tests that BITE at the wiring, added 2026-08-29 after an adversarial review -------------
# Every test above exercises a PURE FUNCTION. An adversarial review replaced the enforcement
# call inside `check()` with `contested = {}`, and the warning call inside `declare()` with
# `wounds = {}`, and the suite stayed GREEN both times: the feature itself was untested and only
# its helpers were covered. That is the same shape as the exp5879 hollow fix -- verifying
# through a path production never takes. These go through the real entry points.

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

import pytest  # noqa: E402

import harness_integrity_lint as guard  # noqa: E402


@pytest.fixture(autouse=True)
def _no_ambient_scope_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite must not inherit the developer's own session id.

    Running the harness tests with `CARNOT_AGENT_SCOPE_ID` exported -- the state any agent that
    actually used this tool is in -- failed three shipped tests. A test whose verdict depends on
    the operator's shell is not measuring the code.
    """
    monkeypatch.delenv(guard.RUN_ID_ENV, raising=False)


@pytest.fixture
def repo(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A real git repo with the guard's module-level paths pointed at it."""
    subprocess.run(["git", "init", "-q", "."], cwd=tmp_path, check=True)
    for key, value in (("user.email", "t@t"), ("user.name", "t")):
        subprocess.run(["git", "config", key, value], cwd=tmp_path, check=True)
    (tmp_path / "a.py").write_text("x\n")
    (tmp_path / "b.py").write_text("y\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(guard, "REPO", tmp_path)
    monkeypatch.setattr(guard, "SCOPES", tmp_path / "ops" / ".agent_scopes")
    return tmp_path


def _declare(run_id: str, *scope: str) -> None:
    guard.declare(list(scope), [], run_id)


def _stage(repo, name: str) -> None:
    (repo / name).write_text((repo / name).read_text() + "edit\n")
    subprocess.run(["git", "add", name], cwd=repo, check=True)


def test_check_refuses_a_path_another_session_owns(repo, monkeypatch, capsys) -> None:
    """The enforcement branch is wired into `check()`, not merely implemented beside it."""
    _declare("alpha", "a.py")
    _declare("beta", "b.py")
    _stage(repo, "a.py")
    monkeypatch.setenv(guard.RUN_ID_ENV, "beta")
    assert guard.check() == 1
    assert "claimed by alpha" in capsys.readouterr().out


def test_check_permits_a_session_committing_its_own_declared_path(repo, monkeypatch) -> None:
    _declare("alpha", "a.py")
    _declare("beta", "b.py")
    _stage(repo, "b.py")
    monkeypatch.setenv(guard.RUN_ID_ENV, "beta")
    assert guard.check() == 0


def test_two_narrow_scopes_do_not_deadlock_through_the_real_entry_point(repo) -> None:
    """The 2026-08-29 deadlock, end to end, with NO environment variable set.

    This is how the tool is actually used -- nothing in the repository set the id until the
    same day this test was written -- and in that state every declaration judged every commit,
    so B staging only its own file was refused by A's record. The pure-function test of the
    same property passed throughout, because it never asked who was judging.
    """
    _declare("agent-a", "a.py")
    _declare("agent-b", "b.py")
    _stage(repo, "b.py")
    assert guard.check() == 0


def test_a_stale_declaration_stops_blocking_through_the_real_entry_point(repo) -> None:
    """The refusal text promised this and the branch that fired did not honour it."""
    _declare("ghost", "a.py")
    path = guard.SCOPES / "ghost.scope.json"
    record = json.loads(path.read_text())
    record["declared_at"] = (datetime.now(UTC) - CLAIM_STALE_AFTER - timedelta(hours=1)).isoformat()
    path.write_text(json.dumps(record))
    _stage(repo, "b.py")
    assert guard.check() == 0


def test_a_lone_declaration_still_judges_an_unidentified_commit(repo) -> None:
    """Attribution is unambiguous with one declaration, so the guard must not go silent."""
    _declare("solo", "a.py")
    _stage(repo, "b.py")
    assert guard.check() == 1


def test_declare_warns_about_an_older_overlapping_claim(repo, capsys) -> None:
    """The declaration-time warning is wired into `declare()`."""
    _declare("alpha", "scripts/target.py")
    capsys.readouterr()
    _declare("beta", "scripts/*")
    assert "WOUNDED" in capsys.readouterr().out


def test_declare_tells_the_agent_to_export_the_id(repo, capsys) -> None:
    """Without this line nothing ever identifies a committer and the feature is inert.

    Before 2026-08-29 a repo-wide grep for the variable found exactly one hit: its own
    definition. Enforcement never fired for anyone.
    """
    _declare("alpha", "a.py")
    assert f"export {guard.RUN_ID_ENV}=alpha" in capsys.readouterr().out


# --- ordering tests that bite (run ids chosen to sort AGAINST age) ---------------------------
# The originals used ids where the run-id tie-break happened to agree with age ("older" < "young"),
# so destroying the age key entirely left them green.


def test_age_beats_run_id_when_the_two_disagree() -> None:
    older = rec("zzz-old", 3, "x.py")
    younger = rec("aaa-new", 1, "x.py")
    assert claim_owner("x.py", [younger, older], NOW) == "zzz-old"


def test_a_corrupt_timestamp_never_wins_and_is_treated_as_abandoned() -> None:
    """`"!"` sorts below every ISO stamp, and an unparseable value was never expiring.

    One corrupt or hostile scope file owned its patterns permanently, while the comment beside
    the code claimed an unusable timestamp loses every comparison.
    """
    corrupt = {"run_id": "corrupt", "declared_at": "!bad-timestamp", "scope": ["x.py"]}
    assert claim_owner("x.py", [corrupt, rec("honest", 1, "x.py")], NOW) == "honest"
    assert contested_paths(["x.py"], [corrupt], "anyone", NOW) == {}


def test_lexical_order_does_not_decide_age() -> None:
    """`.` < `Z`, so a half-second-later stamp sorted OLDER across two timestamp spellings."""
    early = {"run_id": "early", "declared_at": "2026-08-29T10:00:00Z", "scope": ["x.py"]}
    late = {"run_id": "late", "declared_at": "2026-08-29T10:00:00.500000+00:00", "scope": ["x.py"]}
    assert claim_owner("x.py", [early, late], NOW) == "early"


def test_a_non_utc_offset_is_compared_as_an_instant() -> None:
    utc = {"run_id": "utc", "declared_at": "2026-08-29T09:00:00+00:00", "scope": ["x.py"]}
    offset = {"run_id": "offset", "declared_at": "2026-08-29T08:00:00-05:00", "scope": ["x.py"]}
    assert claim_owner("x.py", [utc, offset], NOW) == "utc"


# --- the universal-pattern charset, which was entirely untested ------------------------------


@pytest.mark.parametrize("pattern", ["*", "**", "*/*", "?", "-", "", "   "])
def test_patterns_that_claim_everything_are_advisory(pattern: str) -> None:
    assert is_universal_pattern(pattern)


def test_a_bare_extension_claims_the_whole_repository() -> None:
    """`fnmatch`'s `*` crosses `/`, so `*.py` owned every Python file in the tree."""
    assert is_universal_pattern("*.py")
    assert (
        contested_paths(
            ["python/carnot/paths.py", "tests/python/conftest.py"],
            [rec("grabber", 1, "*.py")],
            "stranger",
            NOW,
        )
        == {}
    )


# `[a-z]` matches single-character names only, so it claims a place rather than the tree.
@pytest.mark.parametrize("pattern", ["scripts/*.py", "scripts/*", "a.py", "docs/notes", "[a-z]"])
def test_a_pattern_naming_a_place_stays_enforceable(pattern: str) -> None:
    assert not is_universal_pattern(pattern)


def test_identical_globs_warn_at_declaration_time() -> None:
    """Two sessions declaring the same glob is the likeliest collision and warned about nothing."""
    mine = rec("beta", 1, "scripts/*")
    older = rec("alpha", 2, "scripts/*")
    assert overlapping_older_claims(["scripts/*"], mine, [older], NOW) == {"scripts/*": "alpha"}
