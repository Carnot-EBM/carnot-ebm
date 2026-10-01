"""REQ-INFRA-7091: the conductor reconciles a non-fast-forward push instead of piling up commits.

INCIDENT 2026-09-30. An outer-loop session pushed merges that the live checkout's `main` did not
contain. The conductor's plain `git push origin main` was then rejected as non-fast-forward on
every commit, it logged one warning each time, and the unpushed backlog grew to 233 and then 75
commits with nothing raising an alarm.

These tests script `run_cmd` so no real git or network is touched. They check the exact git
calls made in each scenario, and that nothing destructive (stash, reset, rebase, force) is ever
among them: the conductor's standing rule is commit-first, never discard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import research_conductor as rc  # noqa: E402

NON_FF = (
    "To ssh://gitea/x.git\n ! [rejected]  main -> main (non-fast-forward)\n"
    "hint: Updates were rejected because the tip of your current branch is behind"
)
FETCH_FIRST = "! [rejected] main -> main (fetch first)\nhint: remote contains work you do not have"
DESTRUCTIVE = ("--force", "-f", "stash", "reset", "rebase", "--hard", "clean")


class FakeGit:
    """Scripted `run_cmd`: answers by git subcommand, in order, and records every call."""

    def __init__(self, answers: dict[str, list[tuple[int, str, str]]]) -> None:
        self.answers = {k: list(v) for k, v in answers.items()}
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], timeout: int = 600, input_text: str | None = None):
        self.calls.append(cmd)
        queue = self.answers.get(cmd[1], [])
        return queue.pop(0) if queue else (0, "", "")

    def subcommands(self) -> list[str]:
        return [c[1] for c in self.calls]


OK = (0, "", "")


@pytest.mark.parametrize(
    "stderr",
    [NON_FF, FETCH_FIRST, "error: failed to push\n! [rejected] main -> main (Non-Fast-Forward)"],
)
def test_the_two_real_non_fast_forward_phrasings_are_recognized(stderr: str) -> None:
    """SCENARIO-INFRA-7091-RECONCILE: only these phrasings mean "fetch and merge first"."""
    assert rc.push_rejected_non_fast_forward(stderr) is True


@pytest.mark.parametrize(
    "stderr",
    [
        "git@github.com: Permission denied (publickey).",
        "remote: error: File x.json is 613 MB; this exceeds GitHub's file size limit of 100 MB",
        "! [remote rejected] main -> main (pre-receive hook declined)",
        "Command timed out",
        "",
    ],
)
def test_other_push_failures_are_not_mistaken_for_non_fast_forward(stderr: str) -> None:
    """SCENARIO-INFRA-7091-OTHER: a merge cannot fix auth, network or a server file-size hook."""
    assert rc.push_rejected_non_fast_forward(stderr) is False


def test_a_clean_push_makes_exactly_one_git_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeGit({"push": [OK]})
    monkeypatch.setattr(rc, "run_cmd", fake)
    assert rc._push_origin_main() is True
    assert fake.subcommands() == ["push"]


def test_non_fast_forward_is_fetched_merged_and_pushed_once_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFRA-7091-RECONCILE."""
    fake = FakeGit({"push": [(1, "", NON_FF), OK], "fetch": [OK], "merge": [OK]})
    monkeypatch.setattr(rc, "run_cmd", fake)
    assert rc._push_origin_main() is True
    assert fake.subcommands() == ["push", "fetch", "merge", "push"]
    merge = fake.calls[2]
    assert "origin/main" in merge
    assert "--no-edit" in merge
    assert fake.calls[1] == ["git", "fetch", "origin", "main"]


def test_a_merge_conflict_is_aborted_and_the_commit_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFRA-7091-CONFLICT: abort restores the tree, and there is no second push."""
    fake = FakeGit(
        {"push": [(1, "", NON_FF)], "fetch": [OK], "merge": [(1, "", "CONFLICT (content)"), OK]}
    )
    monkeypatch.setattr(rc, "run_cmd", fake)
    assert rc._push_origin_main() is False
    assert fake.subcommands() == ["push", "fetch", "merge", "merge"]
    assert fake.calls[3] == ["git", "merge", "--abort"]


def test_a_failed_fetch_starts_no_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeGit({"push": [(1, "", NON_FF)], "fetch": [(128, "", "could not resolve host")]})
    monkeypatch.setattr(rc, "run_cmd", fake)
    assert rc._push_origin_main() is False
    assert fake.subcommands() == ["push", "fetch"]


def test_an_unrelated_push_failure_starts_no_fetch_or_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFRA-7091-OTHER."""
    fake = FakeGit({"push": [(1, "", "Permission denied (publickey).")]})
    monkeypatch.setattr(rc, "run_cmd", fake)
    assert rc._push_origin_main() is False
    assert fake.subcommands() == ["push"]


def test_a_second_rejection_after_the_reconcile_is_not_retried_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly one retry: another session may be pushing right now, and this must not loop."""
    fake = FakeGit({"push": [(1, "", NON_FF), (1, "", NON_FF)], "fetch": [OK], "merge": [OK]})
    monkeypatch.setattr(rc, "run_cmd", fake)
    assert rc._push_origin_main() is False
    assert fake.subcommands() == ["push", "fetch", "merge", "push"]


@pytest.mark.parametrize(
    "answers",
    [
        {"push": [(1, "", NON_FF), OK], "fetch": [OK], "merge": [OK]},
        {"push": [(1, "", NON_FF)], "fetch": [OK], "merge": [(1, "", "CONFLICT"), OK]},
        {"push": [(1, "", NON_FF)], "fetch": [(1, "", "boom")]},
        {"push": [(1, "", "Permission denied")]},
    ],
)
def test_no_path_ever_stashes_resets_rebases_or_forces(
    monkeypatch: pytest.MonkeyPatch, answers: dict[str, list[tuple[int, str, str]]]
) -> None:
    """The standing rule is commit-first, never discard: no scenario may run a destructive git."""
    fake = FakeGit(answers)
    monkeypatch.setattr(rc, "run_cmd", fake)
    rc._push_origin_main()
    flat = [part for call in fake.calls for part in call]
    assert not any(bad in flat for bad in DESTRUCTIVE)


def test_commit_still_reports_success_when_the_reconcile_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFRA-7091-CONFLICT: the work is committed locally, so the commit call is True."""
    fake = FakeGit(
        {
            "commit": [OK],
            "push": [(1, "", NON_FF)],
            "fetch": [OK],
            "merge": [(1, "", "CONFLICT"), OK],
        }
    )
    monkeypatch.setattr(rc, "run_cmd", fake)
    monkeypatch.setattr(rc, "_stage_all_except_claimed", lambda: None)
    monkeypatch.setattr(rc, "with_agent_signature", lambda m: m)
    assert rc.git_commit_and_push("[conductor] test", push=True) is True
    assert "commit" in fake.subcommands()


def test_push_false_never_touches_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeGit({"commit": [OK]})
    monkeypatch.setattr(rc, "run_cmd", fake)
    monkeypatch.setattr(rc, "_stage_all_except_claimed", lambda: None)
    monkeypatch.setattr(rc, "with_agent_signature", lambda m: m)
    assert rc.git_commit_and_push("[conductor] test", push=False) is True
    assert "push" not in fake.subcommands()
    assert "fetch" not in fake.subcommands()
