"""Running the tests must not silently rewrite the research record -- and if it does, we must know.

REQ-ARC-WMTE-6041 / SCENARIOs: suite-mutation-is-detected, in-flight-work-is-not-blamed,
restore-returns-byte-identity, pending-marker-blocks-the-commit

WHY THIS EXISTS (2026-07-29). Running ``pytest tests/python/test_arc_*.py
tests/python/test_experiment_*.py`` left 39 tracked files modified that were clean before the
run. A class of ``test_experiment_*.py`` calls ``runpy.run_path`` on the REAL experiment script,
which writes its artifact as a side effect -- at the same fixed path the historical artifact
lives at. A green test run therefore OVERWRITES the research record, and anyone who then commits
with ``git add -A`` publishes the rewrite.

The detector under test does not try to be clever about which rewrites matter (that is
``determination_preservation_lint.py``'s job, and it is content-aware). This one answers one
factual question -- did the tracked tree move? -- and names the files. The tests below drive
REAL git in a throwaway repo rather than mocking it, for the same reason the sibling lint's
tests do: the bug class here is "which git question did we ask", and a mocked git would happily
reproduce the wrong question.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import test_suite_mutation_check as tsm  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, f"git {' '.join(args)} failed: {r.stderr}"
    return r.stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway git repo holding one committed artifact, with the detector pointed at it."""
    r = tmp_path / "repo"
    (r / "results").mkdir(parents=True)
    (r / "ops").mkdir(parents=True)
    _git(r.parent, "init", "-q", str(r))
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    art = r / "results" / "experiment_3946_r11l_first_solve.json"
    art.write_text(
        json.dumps(
            {
                "experiment": 3946,
                "duration_s": 4.21,
                "solve_provenance": "development_proxy",
                "inference_substrate_correction_note": "2026-07-27: original declaration illegal",
            },
            indent=2,
        )
        + "\n"
    )
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed")

    monkeypatch.setattr(tsm, "REPO", r)
    monkeypatch.setattr(tsm, "SNAPSHOT", r / "ops" / ".test_suite_mutation_snapshot.json")
    monkeypatch.setattr(tsm, "PENDING", r / "ops" / ".test_suite_mutation_pending.json")
    # Without this the tests would write pre-revert backups into the REAL repo's
    # ops/.test_suite_mutation_backup/ -- a test suite that litters the tree it is checking is
    # exactly the hazard this module exists to detect.
    monkeypatch.setattr(tsm, "BACKUP", r / "ops" / ".test_suite_mutation_backup")
    return r, art


def test_a_rewrite_of_a_committed_artifact_is_detected_and_named(repo):
    """THE ORIGIN INCIDENT, in miniature: the experiment script rewrites its own artifact.

    The rewritten artifact loses its corrigendum and its provenance record, exactly as
    ``results/experiment_3946_r11l_first_solve.json`` did on 2026-07-29. The detector's job is
    only to notice the file moved -- it deliberately forms no opinion about the content.
    """
    _, art = repo
    baseline = tsm.snapshot()
    assert baseline == {}, "the fixture repo starts clean"

    art.write_text(json.dumps({"experiment": 3946, "duration_s": 3.98}, indent=2) + "\n")

    muts = tsm.mutations(baseline)
    assert muts == ["results/experiment_3946_r11l_first_solve.json"]


def test_work_that_was_already_in_flight_is_not_blamed_on_the_run(repo):
    """A file dirty BEFORE the run is the operator's in-flight work, not a test side effect.

    Without this, the detector would fire on every real working tree and be switched off within
    a day -- which is the same outcome as not having it.
    """
    r, _ = repo
    inflight = r / "results" / "experiment_9_inflight.json"
    inflight.write_text(json.dumps({"experiment": 9}, indent=2) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed inflight")
    inflight.write_text(json.dumps({"experiment": 9, "edited": True}, indent=2) + "\n")

    baseline = tsm.snapshot()
    assert "results/experiment_9_inflight.json" in baseline

    assert tsm.mutations(baseline) == [], "pre-existing dirt is not attributed to the run"


def test_a_brand_new_untracked_artifact_is_not_a_rewrite(repo):
    """Writing a NEW artifact is normal work. Only an EXISTING committed file moving is harm."""
    r, _ = repo
    baseline = tsm.snapshot()
    (r / "results" / "experiment_9999_new.json").write_text("{}\n")
    assert tsm.mutations(baseline) == []


def test_restore_returns_the_artifact_byte_for_byte(repo):
    """Detection without a reliable undo would leave the record damaged after every survey."""
    _, art = repo
    before = art.read_bytes()
    baseline = tsm.snapshot()
    art.write_text(json.dumps({"experiment": 3946, "duration_s": 3.98}, indent=2) + "\n")
    assert art.read_bytes() != before

    recovered = tsm.restore(tsm.mutations(baseline))

    assert recovered == ["results/experiment_3946_r11l_first_solve.json"]
    assert art.read_bytes() == before, "restore must be byte-identical, not merely 'close'"
    assert tsm.mutations(baseline) == []


def test_restore_backs_a_file_up_before_reverting_it(repo):
    """A wrong revert must cost a ``cp``, not the work. Origin: it has happened twice.

    ``restore()`` is ``git checkout --``, which is unrecoverable for uncommitted content, and the
    detector CANNOT tell a test's write from a human's concurrent edit -- both are simply "modified
    since the snapshot". So editing a tracked file while a wrapped run is in flight gets that edit
    reverted, silently. Both real occurrences were the same file:
    ``docs/research-notes/test-suite-rewrites-the-record-survey-2026-07-29.md``, while it was being
    written (see its own section 3.6 for the first, and section 7b for the second, which happened
    while validating this very script).

    Authorship is not recoverable in principle, so the fix is not smarter attribution -- it is
    making the revert survivable. This pins that the PRE-REVERT content is what lands in the
    backup, which is the only version that is otherwise unrecoverable.
    """
    r, art = repo
    baseline = tsm.snapshot()
    edited = json.dumps({"experiment": 3946, "hand_written": "do not lose me"}, indent=2) + "\n"
    art.write_text(edited)

    tsm.restore(tsm.mutations(baseline))

    backed_up = r / "ops" / ".test_suite_mutation_backup" / "results" / art.name
    assert backed_up.exists(), "the pre-revert content must be recoverable somewhere"
    assert backed_up.read_text() == edited, (
        "the backup must hold what was REVERTED, not what was restored -- the committed version "
        "is already in git and was never at risk"
    )


def test_backup_reports_what_it_could_not_save(repo):
    """A partial backup must be visible, never silent.

    If a file cannot be copied aside, the revert still has to happen -- leaving the record dirty
    would be worse. But a shortfall that nobody sees turns a recoverable mistake back into an
    unrecoverable one, so ``backup()`` returns only the paths it actually saved and the caller
    reports the difference.
    """
    r, _ = repo
    baseline = tsm.snapshot()
    # A path git reports but that is not readable as a file: backup must skip it, not raise.
    assert tsm.backup(["results/does_not_exist.json"]) == []
    assert tsm.backup([]) == []
    del baseline, r


def test_the_wrapper_writes_a_pending_marker_when_it_does_not_restore(repo):
    """``--run`` without ``--restore`` must leave evidence that the record moved.

    The marker is the interlock: it is what stops a later ``git add -A && git commit`` from
    publishing the rewrite in an unattended session where nobody read the console output.
    """
    r, art = repo
    rc = tsm.cmd_run(
        [sys.executable, "-c", f"open({str(art)!r},'w').write('{{\"experiment\": 3946}}')"],
        do_restore=False,
    )
    assert rc == 1, "a detected mutation must be a non-zero exit"
    assert tsm.PENDING.exists()
    marker = json.loads(tsm.PENDING.read_text())
    assert marker["modified_tracked_files"] == ["results/experiment_3946_r11l_first_solve.json"]


def test_the_wrapper_clears_the_marker_when_it_does_restore(repo):
    """``--restore`` is the clean path: the record is put back, so nothing is left to block."""
    r, art = repo
    tsm.PENDING.write_text('{"modified_tracked_files": ["stale"]}\n')
    before = art.read_bytes()

    rc = tsm.cmd_run(
        [sys.executable, "-c", f"open({str(art)!r},'w').write('{{\"experiment\": 3946}}')"],
        do_restore=True,
    )

    assert rc == 0
    assert art.read_bytes() == before
    assert not tsm.PENDING.exists(), "a fully-restored run must not leave a commit block behind"


def test_a_clean_run_clears_a_stale_marker(repo):
    """A marker from an earlier run must not wedge every future commit forever."""
    tsm.PENDING.write_text('{"modified_tracked_files": ["stale"]}\n')
    rc = tsm.cmd_run([sys.executable, "-c", "pass"], do_restore=False)
    assert rc == 0
    assert not tsm.PENDING.exists()


def test_the_gate_refuses_a_commit_while_the_marker_exists(repo):
    """The pre-commit interlock. Exit 1 is what actually stops the unattended agent."""
    tsm.PENDING.write_text(
        json.dumps(
            {"command": ["pytest", "tests/python"], "modified_tracked_files": ["results/x.json"]}
        )
        + "\n"
    )
    assert tsm.cmd_gate() == 1


def test_the_gate_passes_when_there_is_nothing_pending(repo):
    assert not tsm.PENDING.exists()
    assert tsm.cmd_gate() == 0


def test_a_corrupt_marker_still_refuses_rather_than_crashing(repo):
    """Fail CLOSED on a damaged marker: its existence is the signal, its contents are detail."""
    tsm.PENDING.write_text("{not json")
    assert tsm.cmd_gate() == 1


def test_check_without_a_snapshot_is_a_no_op_not_a_failure(repo):
    """A pre-commit hook that hard-failed on a missing snapshot would block every fresh clone."""
    assert not tsm.SNAPSHOT.exists()
    assert tsm.main(["--check"]) == 0


def test_the_detector_never_edits_anything_unless_asked(repo):
    """``--check`` alone is read-only. Auto-restoring by default could destroy real output."""
    _, art = repo
    tsm.snapshot()
    art.write_text(json.dumps({"experiment": 3946, "duration_s": 3.98}, indent=2) + "\n")
    after_write = art.read_bytes()

    assert tsm.main(["--check"]) == 1
    assert art.read_bytes() == after_write, "--check must not silently revert the operator's tree"


def test_the_live_repo_has_no_unresolved_suite_mutation(repo, monkeypatch):
    """CI-visible: if a real run leaves an unrestored rewrite, this fails even if the hook was bypassed.

    Deliberately re-points the module back at the REAL repo (the ``repo`` fixture monkeypatched
    it away) so the assertion is about the live tree, not the throwaway one.
    """
    monkeypatch.setattr(tsm, "PENDING", REPO / "ops" / ".test_suite_mutation_pending.json")
    assert tsm.cmd_gate() == 0, (
        "a test run rewrote tracked files and they were not restored -- see the marker file"
    )


def test_a_renamed_tracked_file_is_parsed_as_one_entry_not_two(repo):
    """``git status --porcelain -z`` emits a rename as TWO records: ``R  <new>`` then a bare ``<old>``.

    An earlier draft sliced every record as ``code=line[:2], path=line[3:]``, which turned the
    unprefixed source record into a garbage entry -- ``results/x.json`` became code ``re`` at path
    ``ults/x.json``. That garbage would be reported to the operator as a modified file and handed
    to ``git checkout`` by ``--restore``, which cannot restore a path that does not exist. Pinned
    here so the parse cannot casually regress.
    """
    r, art = repo
    baseline = tsm.snapshot()
    _git(r, "mv", str(art.relative_to(r)), "results/experiment_3946_renamed.json")

    muts = tsm.mutations(baseline)

    assert muts == ["results/experiment_3946_renamed.json"], (
        f"a rename must yield exactly the destination path, got {muts}"
    )
    assert not any(p.startswith("ults/") or p.startswith("sults/") for p in muts), (
        "a mangled source path leaked out of the -z parse"
    )


# =========================================================================================
# 2026-07-29: the interlock's worst gap -- it was disarmed by the very invocation that caused
# both recorded incidents. `--gate` refuses while a pending marker exists, but only `--run`
# ever wrote that marker, so a bare `pytest tests/python/test_arc_*.py ...` left the gate
# green on a tree the suite had just rewritten. `arm_from_pytest` is called from
# tests/python/conftest.py so the marker is armed HOWEVER pytest was invoked.
# =========================================================================================


def test_arm_from_pytest_arms_the_gate_without_the_run_wrapper(repo):
    """The gap, closed: a rewrite detected outside `--run` must still block the commit."""
    r, art = repo
    baseline = tsm.dirty_tracked()
    assert baseline == {}, "fixture starts clean"

    art.write_text(json.dumps({"experiment": 3946, "duration_s": 9.99}, indent=2) + "\n")
    muts = tsm.arm_from_pytest(baseline, ["pytest", "tests/python/test_experiment_3946.py"])

    assert muts == ["results/experiment_3946_r11l_first_solve.json"]
    assert tsm.PENDING.exists(), "the marker is what the pre-commit gate keys off"
    assert tsm.cmd_gate() == 1, "and the gate must now refuse"

    recorded = json.loads(tsm.PENDING.read_text())
    assert recorded["modified_tracked_files"] == muts
    assert "pytest" in recorded["command"], "the marker names the run that caused it"


def test_arm_from_pytest_clears_a_stale_marker_on_a_clean_run(repo):
    """A clean run must DISARM, or one bad run would block commits forever."""
    r, _ = repo
    tsm.write_pending(["results/x.json"], ["pytest", "old"])
    assert tsm.cmd_gate() == 1

    assert tsm.arm_from_pytest(tsm.dirty_tracked(), ["pytest", "new"]) == []
    assert not tsm.PENDING.exists()
    assert tsm.cmd_gate() == 0


def test_arm_from_pytest_never_reverts_a_file(repo):
    """Arming is safe; auto-reverting is not, and has destroyed in-flight work twice.

    The detector cannot tell a test's write from a human's concurrent edit, so the hook that
    runs on EVERY pytest invocation must only record, never restore. If this ever starts
    reverting, a developer running the suite mid-edit loses their work.
    """
    r, art = repo
    baseline = tsm.dirty_tracked()
    rewritten = json.dumps({"experiment": 3946, "duration_s": 9.99}, indent=2) + "\n"
    art.write_text(rewritten)

    tsm.arm_from_pytest(baseline, ["pytest"])

    assert art.read_text() == rewritten, "arm_from_pytest must leave the working tree untouched"
    assert not tsm.BACKUP.exists(), "and must not have needed the backup path at all"


def test_a_file_already_dirty_before_the_run_is_not_blamed_on_it(repo):
    """In-flight human work must not arm the gate, or the interlock becomes unusable.

    This is the property that makes auto-arming safe to run on every invocation: the baseline
    is taken at session START, so a file the developer was already editing is excluded.
    """
    r, art = repo
    art.write_text(json.dumps({"experiment": 3946, "duration_s": 1.0}, indent=2) + "\n")
    baseline = tsm.dirty_tracked()
    assert "results/experiment_3946_r11l_first_solve.json" in baseline

    assert tsm.arm_from_pytest(baseline, ["pytest"]) == []
    assert not tsm.PENDING.exists()
