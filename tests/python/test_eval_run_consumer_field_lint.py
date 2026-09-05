"""The eval-run consumer/producer field join (REQ-ARC-WMTE-6642).

SCENARIO-ARC-WMTE-6642-DECLARE: a consumer without EVAL_RUN_FIELDS_READ fails.
SCENARIO-ARC-WMTE-6642-JOIN: observed passes; wired-unobserved notices; neither fails.
SCENARIO-ARC-WMTE-6642-FAIL-CLOSED: an explicit missing/empty runs dir fails; populations print;
    no consumer tree or an empty producer surface fails (amendment 2026-09-05).
SCENARIO-ARC-WMTE-6642-WORKTREE-CORPUS: a checkout with no corpus joins against the main
    checkout's, found through `git rev-parse --git-common-dir`, and says so.
SCENARIO-ARC-WMTE-6642-CORPUS-ABSENT-SKIP: no corpus anywhere skips the artifact half LOUDLY
    and still refuses a field absent from producer source; failures are a superset.

Origin: three consumers in three days ran against fields no artifact recorded
(trajectory_supervisor, generator_channels, induction_attempts provenance).
Second origin (2026-09-05): the corpus is gitignored, so every `scripts/*.py`
commit from a git worktree was refused with "runs directory missing".
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "eval_run_consumer_field_lint",
    Path(__file__).resolve().parents[2] / "scripts" / "eval_run_consumer_field_lint.py",
)
lint = importlib.util.module_from_spec(_SPEC)
sys.modules["eval_run_consumer_field_lint"] = lint
_SPEC.loader.exec_module(lint)

REPO = Path(__file__).resolve().parents[2]


def _fake_repo(tmp_path: Path, *, consumer_body: str, artifact: dict | None) -> Path:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "python" / "carnot" / "agentic").mkdir(parents=True)
    runs = root / "results" / "arc_leaderboard_eval_runs"
    runs.mkdir(parents=True)
    (root / "scripts" / "consumer.py").write_text(consumer_body)
    # The producer surface: the eval harness emits "wired_field" as a literal.
    (root / "scripts" / "arc_leaderboard_eval.py").write_text('ROW = {"wired_field": 1}\n')
    (root / "python" / "carnot" / "agentic" / "runtime.py").write_text("x = 1\n")
    if artifact is not None:
        (runs / "run.json").write_text(json.dumps(artifact))
    return root


def _run(root: Path):
    return lint.run_lint(
        root,
        root / "results" / "arc_leaderboard_eval_runs",
        root / "results" / "arc_leaderboard_eval.json",
        (root / "scripts" / "arc_leaderboard_eval.py", root / "python" / "carnot" / "agentic"),
    )


def test_consumer_without_declaration_fails(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6642-DECLARE."""

    root = _fake_repo(
        tmp_path,
        consumer_body='d = "arc_leaderboard_eval_runs"\n',
        artifact={"per_game": []},
    )
    failures, _ = _run(root)
    assert any("EVAL_RUN_FIELDS_READ" in f for f in failures)


def test_join_observed_wired_and_missing(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6642-JOIN: all three verdicts from one consumer."""

    body = (
        'd = "arc_leaderboard_eval_runs"\n'
        'EVAL_RUN_FIELDS_READ = ("observed_field", "wired_field", "ghost_field")\n'
    )
    root = _fake_repo(
        tmp_path,
        consumer_body=body,
        artifact={"per_game": [{"observed_field": 3}]},
    )
    failures, notices = _run(root)
    assert any("ghost_field" in f and "NO eval-run artifact" in f for f in failures)
    assert not any("observed_field" in f for f in failures)
    assert not any("wired_field" in f for f in failures)
    assert any("wired_field" in n and "wired" in n for n in notices)


def test_nested_keys_count_as_observed(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6642-JOIN: containment is any-depth, matching real rows."""

    body = 'd = "arc_leaderboard_eval_runs"\nEVAL_RUN_FIELDS_READ = ("deep_field",)\n'
    root = _fake_repo(
        tmp_path,
        consumer_body=body,
        artifact={"per_game": [{"diag": {"attempts": [{"deep_field": 1}]}}]},
    )
    failures, _ = _run(root)
    assert failures == []


def test_fails_closed_without_artifacts(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6642-FAIL-CLOSED, both halves."""

    body = 'd = "arc_leaderboard_eval_runs"\nEVAL_RUN_FIELDS_READ = ("f",)\n'
    root = _fake_repo(tmp_path, consumer_body=body, artifact=None)
    failures, _ = _run(root)
    assert any("join cannot run" in f for f in failures)

    import shutil

    shutil.rmtree(root / "results" / "arc_leaderboard_eval_runs")
    failures, _ = _run(root)
    assert any("runs directory missing" in f for f in failures)


def test_population_note_always_printed(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6642-FAIL-CLOSED: a zero-findings run names its populations."""

    body = 'd = "arc_leaderboard_eval_runs"\nEVAL_RUN_FIELDS_READ = ("per_game",)\n'
    root = _fake_repo(tmp_path, consumer_body=body, artifact={"per_game": []})
    failures, notices = _run(root)
    assert failures == []
    assert any("population:" in n for n in notices)


def test_a_heartbeat_key_does_not_count_as_observed(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7012-D (lint half): the REQ-ARC-WMTE-7010 heartbeat shares the runs
    directory. A key that exists only in `*.progress.json` is not a recorded field, so a
    consumer declaring it must still fail the join."""

    body = 'd = "arc_leaderboard_eval_runs"\nEVAL_RUN_FIELDS_READ = ("induction_in_flight",)\n'
    root = _fake_repo(tmp_path, consumer_body=body, artifact={"per_game": [{"game": "x"}]})
    runs = root / "results" / "arc_leaderboard_eval_runs"
    (runs / "x-1.progress.json").write_text(json.dumps({"induction_in_flight": None}))
    failures, _ = _run(root)
    assert failures, "a heartbeat-only key satisfied the observed join"
    assert any("induction_in_flight" in f for f in failures)


def test_real_repo_contract_holds() -> None:
    """The live checkout passes: every seeded declaration is emitted somewhere real.

    This is the wiring test — if a consumer is added without a declaration, or a
    declared field stops being emitted, this test goes RED with the suite.
    """

    failures, notices = lint.run_lint()
    assert failures == [], failures
    assert any("population:" in n for n in notices)


# --- The corpus exists once per machine (amendment 2026-09-05) ---------------------------------
# SCENARIO-ARC-WMTE-6642-WORKTREE-CORPUS and SCENARIO-ARC-WMTE-6642-CORPUS-ABSENT-SKIP.
# Before the amendment every `scripts/*.py` commit from a git worktree was refused with
# "runs directory missing": the corpus is gitignored and a worktree has none.

_GIT_ENV = {
    **os.environ,
    # No user or system config: a signing key or a hooks path on this machine must not
    # reach a fixture repository.
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_AUTHOR_NAME": "fixture",
    "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
    "GIT_COMMITTER_NAME": "fixture",
    "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
}


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=_GIT_ENV
    )


def _main_and_worktree(tmp_path: Path, *, body: str, artifact: dict) -> tuple[Path, Path]:
    """A real main checkout that holds a corpus, and a real worktree of it that holds none.

    `results/` is left untracked, exactly like the gitignored corpus in the live repository,
    so the worktree comes up without it.
    """
    main = _fake_repo(tmp_path, consumer_body=body, artifact=artifact)
    _git("init", "-q", "-b", "main", cwd=main)
    _git("add", "scripts", "python", cwd=main)
    _git("commit", "-q", "-m", "seed", cwd=main)
    wt = tmp_path / "wt"
    _git("worktree", "add", "-q", str(wt), "-b", "agent", cwd=main)
    assert not (wt / "results").exists(), "the worktree must start without a corpus"
    return main, wt


# Declares a field that exists ONLY in the corpus. Without the corpus it fails, so a green
# run proves the corpus was found — the fallback cannot be decorative.
_OBSERVED_ONLY = 'd = "arc_leaderboard_eval_runs"\nEVAL_RUN_FIELDS_READ = ("observed_field",)\n'


def test_a_worktree_uses_the_main_checkouts_corpus(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6642-WORKTREE-CORPUS: the incident end to end, on real git."""

    main, wt = _main_and_worktree(
        tmp_path, body=_OBSERVED_ONLY, artifact={"per_game": [{"observed_field": 3}]}
    )
    failures, notices = lint.run_lint(wt)
    assert failures == [], failures
    used = [n for n in notices if n.startswith("runs directory:")]
    assert used, notices
    assert "main checkout" in used[0] and str(main.resolve()) in used[0], used[0]
    assert not any(n.startswith(lint.SKIP_MARKER) for n in notices)


def test_main_checkout_root_is_none_outside_git_and_in_the_main_checkout(
    tmp_path: Path, monkeypatch
) -> None:
    """The resolver names the main checkout from a worktree, and nothing else."""

    # A ceiling stops git from climbing out of tmp_path into whatever contains it.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    nogit = tmp_path / "nogit"
    nogit.mkdir()
    assert lint.main_checkout_root(nogit) is None
    main, wt = _main_and_worktree(
        tmp_path, body=_OBSERVED_ONLY, artifact={"per_game": [{"observed_field": 3}]}
    )
    assert lint.main_checkout_root(main) is None
    assert lint.main_checkout_root(wt) == main.resolve()


def test_main_checkout_root_rejects_a_bare_repository_layout(monkeypatch) -> None:
    """A bare repository's common dir is `<name>.git` with no working tree beside it.

    Its parent is NOT a checkout, so the resolver must return None rather than point the
    join at a directory that never held a corpus. Mutation M13 (2026-09-05) survived until
    this test existed.
    """

    class _Proc:
        returncode = 0
        stdout = "/srv/git/carnot.git\n"

    monkeypatch.setattr(lint.subprocess, "run", lambda *a, **k: _Proc())
    assert lint.main_checkout_root(Path("/srv/git/anything")) is None


def test_no_corpus_anywhere_skips_loudly_and_still_refuses_an_unwired_field(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-6642-CORPUS-ABSENT-SKIP."""

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    body = (
        'd = "arc_leaderboard_eval_runs"\nEVAL_RUN_FIELDS_READ = ("wired_field", "ghost_field")\n'
    )
    root = _fake_repo(tmp_path, consumer_body=body, artifact=None)
    shutil.rmtree(root / "results")
    failures, notices = lint.run_lint(root)
    assert any(n.startswith(lint.SKIP_MARKER) for n in notices), notices
    assert any("0 artifact(s)" in n for n in notices), notices
    assert any("1 declared field(s) verified against producer source only" in n for n in notices)
    assert any("ghost_field" in f and "NO producer source" in f for f in failures), failures
    assert not any("wired_field" in f for f in failures), failures


def test_an_empty_local_runs_dir_still_falls_back_to_the_main_checkout(tmp_path: Path) -> None:
    """A `mkdir` of the runs directory is not a corpus. The fallback keys on artifacts."""

    main, wt = _main_and_worktree(
        tmp_path, body=_OBSERVED_ONLY, artifact={"per_game": [{"observed_field": 3}]}
    )
    (wt / "results" / "arc_leaderboard_eval_runs").mkdir(parents=True)
    failures, notices = lint.run_lint(wt)
    assert failures == [], failures
    assert any(n.startswith("runs directory:") and str(main.resolve()) in n for n in notices)


def test_a_worktree_of_a_main_without_a_corpus_skips_and_names_both(tmp_path: Path) -> None:
    """A fresh clone's shape: git names a main checkout, but it has no corpus either."""

    body = 'd = "arc_leaderboard_eval_runs"\nEVAL_RUN_FIELDS_READ = ("wired_field",)\n'
    main, wt = _main_and_worktree(tmp_path, body=body, artifact={"per_game": []})
    shutil.rmtree(main / "results")
    failures, notices = lint.run_lint(wt)
    assert failures == [], failures
    skip = [n for n in notices if n.startswith(lint.SKIP_MARKER)]
    assert skip, notices
    assert str(wt.resolve()) in skip[0] and str(main.resolve()) in skip[0], skip[0]


def test_missing_corpus_never_admits_more(tmp_path: Path, monkeypatch) -> None:
    """The artifact half can only widen passes: failures without it are a superset.

    This is the property that makes the loud skip fail-closed in substance.
    """

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    body = (
        'd = "arc_leaderboard_eval_runs"\n'
        'EVAL_RUN_FIELDS_READ = ("observed_field", "wired_field", "ghost_field")\n'
    )
    root = _fake_repo(tmp_path, consumer_body=body, artifact={"per_game": [{"observed_field": 3}]})
    with_corpus, _ = lint.run_lint(root, root / "results" / "arc_leaderboard_eval_runs")
    shutil.rmtree(root / "results")
    without_corpus, notices = lint.run_lint(root)
    assert any(n.startswith(lint.SKIP_MARKER) for n in notices), notices
    assert set(with_corpus) <= set(without_corpus)
    # The precision cost lands in the loud direction: observed-only fields FAIL.
    assert not any("observed_field" in f for f in with_corpus)
    assert any("observed_field" in f for f in without_corpus)


def test_fails_closed_when_there_is_no_consumer_tree(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-6642-FAIL-CLOSED (amendment): a wrong root scans nothing, so it fails."""

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    empty = tmp_path / "nothing"
    empty.mkdir()
    failures, _ = lint.run_lint(empty)
    assert any("no consumer was scanned" in f for f in failures), failures


def test_fails_closed_when_the_producer_surface_is_empty(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6642-FAIL-CLOSED (amendment): no producer file means it looked wrong."""

    body = 'd = "arc_leaderboard_eval_runs"\nEVAL_RUN_FIELDS_READ = ("per_game",)\n'
    root = _fake_repo(tmp_path, consumer_body=body, artifact={"per_game": []})
    failures, _ = lint.run_lint(
        root,
        root / "results" / "arc_leaderboard_eval_runs",
        root / "results" / "arc_leaderboard_eval.json",
        (root / "nowhere.py", root / "nowhere_dir"),
    )
    assert any("producer surface empty" in f for f in failures), failures


def test_the_ok_line_says_skipped_when_the_corpus_was_absent(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """SCENARIO-ARC-WMTE-6642-CORPUS-ABSENT-SKIP: the skip is on the LAST line, not only mid-output."""

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    body = 'd = "arc_leaderboard_eval_runs"\nEVAL_RUN_FIELDS_READ = ("wired_field",)\n'
    root = _fake_repo(tmp_path, consumer_body=body, artifact=None)
    shutil.rmtree(root / "results")
    rc = lint.main(["--repo-root", str(root)])
    out = capsys.readouterr().out
    assert rc == 0
    assert lint.SKIP_MARKER in out
    assert "SKIPPED" in out.strip().splitlines()[-1]


def test_real_repo_passes_on_producer_source_alone(monkeypatch) -> None:
    """Skip mode on the live checkout: no declared field needs the corpus to pass.

    Measured 2026-09-05: 7 consumers, 20 fields wired-only, 0 failures. If this goes RED,
    a consumer now reads a field that is observed in artifacts but absent from producer
    source, and commits from any checkout without a corpus will start failing. Decide
    then whether to wire the field, not whether to weaken the lint.
    """

    monkeypatch.setattr(lint, "resolve_runs_dir", lambda repo_root: (None, "forced absent"))
    failures, notices = lint.run_lint()
    assert failures == [], failures
    assert any(n.startswith(lint.SKIP_MARKER) for n in notices)
