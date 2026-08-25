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
import os
import subprocess
import sys
import uuid
from datetime import timedelta
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
    # RUNS holds the per-run snapshots and pending markers that replaced the two single files.
    # It MUST be redirected for the same reason as BACKUP, and forgetting it is not hypothetical:
    # the first run of these tests against the per-run layout wrote 11 snapshot files into the
    # REAL repo's ops/.test_suite_mutation_runs/ from four xdist workers, because the fixture
    # had been written before the constant existed. A test suite that litters the tree it is
    # checking is precisely the hazard this module exists to detect, so
    # `test_no_module_state_path_escapes_the_throwaway_repo` now fails if a NEW state constant
    # is ever added without being redirected here.
    monkeypatch.setattr(tsm, "RUNS", r / "ops" / ".test_suite_mutation_runs")
    # Without this the tests would write pre-revert backups into the REAL repo's
    # ops/.test_suite_mutation_backup/ -- a test suite that litters the tree it is checking is
    # exactly the hazard this module exists to detect.
    monkeypatch.setattr(tsm, "BACKUP", r / "ops" / ".test_suite_mutation_backup")
    # The mutation-PROOF session lock. Without this redirect a proof test would take the REAL
    # lock and block a live agent's proof, or reclaim one mid-session.
    monkeypatch.setattr(tsm, "PROOF_LOCK", r / "ops" / ".test_suite_mutation_proof.lock")
    # The run id is otherwise derived from the parent PID, which every xdist worker shares -- so
    # without this, concurrent workers would collide on one snapshot name and the ambiguity
    # refusal would fire inside unrelated tests.
    monkeypatch.setenv(tsm.RUN_ID_ENV, f"test-{uuid.uuid4().hex[:12]}")
    # _stash_hidden_paths reads pre-commit's REAL cache by default; on the
    # operator's box that cache holds fresh patches from live commits, which
    # would leak nondeterminism into every test here. Point it at an empty
    # sandbox; the stash-window tests below fill it deliberately.
    monkeypatch.setenv("PRE_COMMIT_HOME", str(tmp_path / "pre-commit-cache"))
    return r, art


def _markers(tsm_mod=tsm) -> list[Path]:
    """Every pending marker currently blocking a commit, wherever the layout puts them."""
    return tsm_mod.marker_paths()


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
    markers = _markers()
    assert len(markers) == 1, "the run must leave exactly one marker"
    marker = json.loads(markers[0].read_text())
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
    """The pre-commit interlock. Exit 1 is what actually stops the unattended agent.

    FIXTURE CORRECTED 2026-07-29 (not the assertion). This named ``results/x.json``, a path that
    does not exist in the throwaway repo and is not dirty -- so it described a marker whose
    damage is already resolved, while its docstring describes the opposite ("a test run rewrote
    files and they were not restored"). Under the per-run layout a marker retires when every file
    it names is clean again, so the old fixture was asserting that a RESOLVED marker keeps
    refusing, which would wedge commits forever after a restore. The artifact is now genuinely
    rewritten, which is the situation the test is about.
    """
    _, art = repo
    art.write_text(json.dumps({"experiment": 3946, "duration_s": 0.01}) + "\n")
    tsm.write_pending(["results/experiment_3946_r11l_first_solve.json"], ["pytest", "tests/python"])
    assert tsm.cmd_gate() == 1


def test_the_gate_passes_when_there_is_nothing_pending(repo):
    assert not tsm.PENDING.exists()
    assert tsm.cmd_gate() == 0


def test_a_corrupt_marker_still_refuses_rather_than_crashing(repo):
    """Fail CLOSED on a damaged marker: its existence is the signal, its contents are detail."""
    tsm.PENDING.write_text("{not json")
    assert tsm.cmd_gate() == 1


def test_check_without_a_baseline_refuses_rather_than_reporting_a_clean_tree(repo):
    """EXPECTATION DELIBERATELY REVERSED 2026-07-29 -- recorded, not silently flipped.

    This test used to be ``test_check_without_a_snapshot_is_a_no_op_not_a_failure`` and asserted
    ``main(["--check"]) == 0``, reasoning that "a pre-commit hook that hard-failed on a missing
    snapshot would block every fresh clone".

    The reasoning does not apply to this command. ``--check`` is NOT a pre-commit hook -- ``--gate``
    is, and ``--gate`` keys off pending markers, not snapshots, so it still passes on a fresh clone
    (pinned by ``test_the_gate_passes_when_there_is_nothing_pending``). What exit 0 actually meant
    here was "the tracked tree is where you left it", which is this tool's documented contract for
    that code, asserted on the strength of never having looked. That is the same fail-open shape as
    the old ``_git`` swallowing errors, and it is the reason the ambiguous-baseline case can be
    handled coherently: with no baseline of its own the honest answer is "ask me again with a run
    id", never "fine".
    """
    assert not tsm.SNAPSHOT.exists()
    assert tsm.find_snapshots() == []
    assert tsm.main(["--check"]) == 1


def test_the_gate_still_passes_on_a_fresh_clone(repo):
    """The fresh-clone concern the reversed test above was protecting, pinned where it belongs.

    Making ``--check`` refuse without a baseline is only safe because the PRE-COMMIT command is a
    different one, and it does not need a baseline at all.
    """
    assert tsm.find_snapshots() == []
    assert not tsm.marker_paths()
    assert tsm.cmd_gate() == 0


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
    monkeypatch.setattr(tsm, "RUNS", REPO / "ops" / ".test_suite_mutation_runs")
    monkeypatch.setattr(tsm, "REPO", REPO)
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
    markers = _markers()
    assert markers, "the marker is what the pre-commit gate keys off"
    assert tsm.cmd_gate() == 1, "and the gate must now refuse"

    recorded = json.loads(markers[0].read_text())
    assert recorded["modified_tracked_files"] == muts
    assert "pytest" in recorded["command"], "the marker names the run that caused it"


def test_a_marker_retires_once_the_rewrite_it_names_is_undone(repo):
    """A clean run must DISARM, or one bad run would block commits forever.

    Was ``test_arm_from_pytest_clears_a_stale_marker_on_a_clean_run``. Same property, but the
    mechanism it pins changed for a reason: disarming used to be an unconditional
    ``PENDING.unlink()`` on any clean run, which let a CONCURRENT workflow's clean run switch off
    an interlock this one had just armed (Race 1, replayed in the module docstring). Retirement is
    now derived from the tree -- a marker retires exactly when every file it names is clean again
    -- so this test drives a REAL rewrite and a REAL restore rather than a marker naming a path
    that was never dirty.
    """
    _, art = repo
    before = art.read_bytes()
    baseline = tsm.dirty_tracked()
    art.write_text(json.dumps({"experiment": 3946, "duration_s": 9.99}) + "\n")

    assert tsm.arm_from_pytest(baseline, ["pytest", "old"]) != []
    assert tsm.cmd_gate() == 1, "armed while the rewrite is on disk"

    art.write_bytes(before)  # the operator restores it
    assert tsm.arm_from_pytest(tsm.dirty_tracked(), ["pytest", "new"]) == []
    assert not tsm.marker_paths(), "the marker must retire once its damage is gone"
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


# =========================================================================================
# CONCURRENCY (2026-07-29 review). Two or three workflows share this working tree routinely,
# and the first version kept ONE snapshot and ONE pending marker, both rewritten by every run.
# Each race below was reproduced against the committed code before it was fixed; these are the
# regression tests for the reproductions.
# =========================================================================================


def test_a_concurrent_clean_run_cannot_disarm_another_runs_interlock(repo):
    """RACE 1, the interlock switching itself off. The worst of the two by a distance.

    Reproduced against the committed version in a throwaway repo: workflow A detects a rewrite
    and arms the marker; workflow B then finishes an unrelated CLEAN run, whose unconditional
    ``PENDING.unlink()`` deletes A's marker. ``--gate`` afterwards printed OK and exited 0 with
    A's rewrite still sitting in the tree as ``M results/artA.json`` -- i.e. a passing test run
    turned off the only thing standing between a green suite and a published rewrite.
    """
    _, art = repo
    other = art.parent / "second_artifact.json"
    other.write_text('{"experiment": 1}\n')
    _git(art.parent.parent, "add", "-A")
    _git(art.parent.parent, "commit", "-q", "-m", "second")

    # Workflow A: a run that rewrites the record and does not restore it.
    baseline_a = tsm.dirty_tracked()
    art.write_text('{"experiment": 3946, "duration_s": 9.99}\n')
    assert tsm.arm_from_pytest(baseline_a, ["pytest", "-k", "A"]) != []
    assert tsm.cmd_gate() == 1

    # Workflow B: an unrelated run that changes nothing. It must not clear A's marker.
    assert tsm.arm_from_pytest(tsm.dirty_tracked(), ["pytest", "-k", "B"]) == []

    assert tsm.cmd_gate() == 1, "B's clean run must NOT disarm the interlock A armed"
    assert "duration_s" in art.read_text(), "and A's rewrite is still on disk, unpublished"


def test_two_runs_arming_at_once_both_keep_their_own_marker(repo):
    """One shared marker file loses information the moment two runs arm it.

    The old layout wrote a single file, so the second run's file list REPLACED the first's; the
    first run's rewrite then stopped being named by anything and became unguarded as soon as the
    second was resolved. One file per arming event makes that structurally impossible.
    """
    r, art = repo
    other = r / "results" / "other.json"
    other.write_text('{"experiment": 2}\n')
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "other")

    # EACH RUN TAKES ITS OWN BASELINE, which is what makes this a real test of the layout.
    # An earlier draft reused one baseline for both, so the second run's mutation list happened to
    # contain BOTH files -- and a single shared marker therefore still named both. Mutation
    # testing caught it: reverting `write_pending` to the one-file layout left the suite green.
    # With per-run baselines each marker names exactly one file, so a shared file loses one.
    base_one = tsm.dirty_tracked()
    art.write_text('{"experiment": 3946, "duration_s": 1.0}\n')
    tsm.arm_from_pytest(base_one, ["pytest", "run-one"])

    base_two = tsm.dirty_tracked()  # run two starts with run one's rewrite already in flight
    other.write_text('{"experiment": 2, "changed": true}\n')
    tsm.arm_from_pytest(base_two, ["pytest", "run-two"])

    markers = tsm.marker_paths()
    assert len(markers) == 2, "one marker per arming event; a shared file would show 1"
    named = set()
    for m in markers:
        named |= set(json.loads(m.read_text())["modified_tracked_files"])
    assert named == {"results/experiment_3946_r11l_first_solve.json", "results/other.json"}, (
        "a single shared marker would have lost run one's file when run two overwrote it"
    )
    assert tsm.cmd_gate() == 1


def test_one_workflows_baseline_cannot_be_overwritten_by_another(repo):
    """RACE 2, the false negative: A reads B's baseline and reports a real rewrite as OK.

    Reproduced against the committed version: A snapshots on a clean tree; B edits the same file
    in flight and snapshots (overwriting the single shared file); A's run then rewrites that file;
    A's ``--check`` reads B's baseline, finds the file listed as "already dirty", and prints
    "OK -- no tracked file was modified". Per-run baselines make each workflow's snapshot
    unreachable from the other.
    """
    _, art = repo
    tsm.snapshot("wfA")
    art.write_text('{"experiment": 3946, "in_flight": true}\n')  # B's concurrent edit
    tsm.snapshot("wfB")
    art.write_text('{"experiment": 3946, "duration_s": 9.99}\n')  # A's run rewrites it

    assert tsm.mutations(tsm._load_snapshot("wfA")) == [
        "results/experiment_3946_r11l_first_solve.json"
    ], "A must still see the rewrite its own baseline predates"
    assert tsm.mutations(tsm._load_snapshot("wfB")) == [], (
        "and B must still not be blamed for work it had already started"
    )


def test_check_refuses_when_several_baselines_could_be_meant(repo, monkeypatch):
    """Concurrent workflows in this project can share a PPID, so auto-matching can be ambiguous.

    Answering anyway is Race 2 with extra steps -- picking an arbitrary other run's baseline is
    exactly how a real rewrite gets reported as OK. So ambiguity is surfaced with the candidate
    run ids, never resolved by guessing.
    """
    monkeypatch.delenv(tsm.RUN_ID_ENV, raising=False)
    tsm.snapshot()
    tsm.snapshot()
    assert len(tsm.find_snapshots()) == 2

    with pytest.raises(tsm.AmbiguousSnapshot):
        tsm._load_snapshot()
    assert tsm.main(["--check"]) == 1


def test_check_honours_a_pinned_run_id_from_the_environment(repo, monkeypatch, capsys):
    """Found by replaying Race 2: `--check` consulted the env var when WRITING but not READING.

    `resolve_run_id` invents a fresh auto id when nothing is pinned, which is right for taking a
    new baseline and wrong for finding an existing one -- so every ``--check`` under a pinned
    ``$CARNOT_MUTATION_RUN_ID`` silently fell through to the session glob and reported OK.

    THE ASSERTION HAS TO BE ABOUT THE OUTPUT, not just the exit code. An earlier draft asserted
    only ``== 1``, which BOTH branches satisfy -- finding the baseline and reporting the rewrite
    exits 1, and failing to find any baseline also exits 1 (it refuses). Mutation testing caught
    that: reverting the read-side resolver left the suite green. So this pins WHICH refusal.
    """
    monkeypatch.setenv(tsm.RUN_ID_ENV, "pinned-workflow")
    tsm.snapshot()
    assert (tsm.RUNS / "pinned-workflow.snapshot.json").exists()

    _, art = repo
    art.write_text('{"experiment": 3946, "duration_s": 7.7}\n')
    assert tsm.main(["--check"]) == 1
    out = capsys.readouterr().out
    # Heading reworded 2026-07-30 with attribution: "MODIFIED BY THE RUN" was a claim the tool
    # could not support (this rewrite was made by the TEST process, not by an observed run), and
    # asserting it here would re-pin the overclaim the split exists to remove. The assertion's
    # intent is unchanged -- it still pins WHICH refusal this is.
    assert "MODIFIED SINCE THE BASELINE" in out, (
        f"must REPORT the rewrite, not refuse for lack of a baseline: {out}"
    )
    assert "results/experiment_3946_r11l_first_solve.json" in out
    assert "no baseline for this run" not in out


def test_a_git_failure_refuses_the_gate_instead_of_reporting_a_clean_tree(repo, monkeypatch):
    """FAIL CLOSED. ``_git`` used to return "" on a non-zero exit, so a broken git produced an
    empty dirty-set, which produced "OK -- no tracked file was modified", which is a confident
    all-clear over a tree nobody managed to look at. ``--gate`` is a pre-commit hook: the one
    moment its answer matters is the moment a wrong answer publishes the rewrite.
    """

    def boom(*_a, **_k):
        raise tsm.GitError("git status failed (exit 128): not a git repository")

    monkeypatch.setattr(tsm, "dirty_tracked", boom)
    assert tsm.cmd_gate() == 1, "a gate that cannot see the tree must refuse, not pass"
    assert tsm.main(["--check"]) == 1


def test_git_returning_nonzero_raises_rather_than_looking_like_a_clean_tree(repo):
    """The primitive itself, so the fail-open cannot be reintroduced one layer down."""
    with pytest.raises(tsm.GitError):
        tsm._git("cat-file", "-p", "definitely-not-an-object")


def test_an_unreadable_marker_is_never_retired(repo):
    """Unreadable is not the same as resolved.

    Retirement is evidence-based -- every file the marker names is clean again -- so a marker
    whose evidence cannot be read must keep refusing. Otherwise corrupting a marker would be a
    way to clear the gate.
    """
    (tsm.RUNS).mkdir(parents=True, exist_ok=True)
    bad = tsm.RUNS / "corrupt.pending.json"
    bad.write_text("{not json")
    empty = tsm.RUNS / "empty.pending.json"
    empty.write_text(json.dumps({"modified_tracked_files": []}) + "\n")

    assert set(tsm.prune_stale_markers()) == {bad, empty}
    assert bad.exists() and empty.exists()
    assert tsm.cmd_gate() == 1


def test_a_legacy_single_marker_still_blocks_after_the_upgrade(repo):
    """A tree that armed the OLD interlock must not silently unblock when this version lands."""
    _, art = repo
    art.write_text('{"experiment": 3946, "duration_s": 5.0}\n')
    tsm.PENDING.parent.mkdir(parents=True, exist_ok=True)
    tsm.PENDING.write_text(
        json.dumps(
            {
                "command": ["pytest"],
                "modified_tracked_files": ["results/experiment_3946_r11l_first_solve.json"],
            }
        )
        + "\n"
    )
    assert tsm.cmd_gate() == 1


def test_no_module_state_path_escapes_the_throwaway_repo(repo):
    """The fixture must redirect EVERY state path, and this fails if a new one is added.

    Not hypothetical: the first run of these tests against the per-run layout wrote 11 snapshot
    files into the REAL repo, from four xdist workers, because ``RUNS`` was added to the module
    without being added to the fixture. A test suite that writes into the tree it is guarding is
    the exact hazard this module exists to detect, so the guard is now mechanical.
    """
    r, _ = repo
    # DERIVED, not listed. The hardcoded five names this replaced were a pattern list narrower
    # than the concept it guards: PROOF_LOCK was added to the module on 2026-08-25 and the old
    # list would have stayed green while the proof tests took the REAL lock. Every module-level
    # Path is swept, so a new state constant is caught by existing.
    # A module Path outside the repo is allowed ONLY if it is read-only, and the reason has to
    # be written down here. Forcing someone to type the reason is the control -- the same shape
    # as ACKNOWLEDGED_NON_QA_LAYER. The sweep found OBSERVER_DIR on its first run, which the
    # hardcoded list had never covered.
    read_only_by_design = {
        "OBSERVER_DIR": "source of the sitecustomize shim; read and copied from, never written",
    }
    escapes = []
    for name, value in vars(tsm).items():
        if name.startswith("_") or not isinstance(value, Path):
            continue
        p = Path(value)
        if r in p.parents or p == r or name in read_only_by_design:
            continue
        escapes.append(f"tsm.{name} -> {p}")
    assert not escapes, (
        f"state path(s) outside the throwaway repo {r}: {escapes} -- redirect them in the "
        "fixture, or classify them in read_only_by_design with the reason"
    )
    # The sweep must actually see the known constants; an empty sweep would pass vacuously.
    swept = {n for n, v in vars(tsm).items() if isinstance(v, Path) and not n.startswith("_")}
    assert {"SNAPSHOT", "PENDING", "RUNS", "BACKUP", "REPO", "PROOF_LOCK"} <= swept


def test_the_legacy_shared_baseline_is_not_used_even_when_present(repo):
    """The asymmetry that makes the legacy compatibility story safe.

    A stale MARKER is still honoured, because honouring it can only make the gate refuse MORE.
    A stale BASELINE is not, because honouring it makes the gate refuse LESS: it is the shared
    mutable cell any other workflow may have overwritten, so reading it as if it were yours is
    Race 2 -- which is how a genuine rewrite gets reported as OK. An earlier draft of this fix
    kept the fallback "for in-flight compatibility" and thereby preserved the hole.
    """
    _, art = repo
    tsm.SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    tsm.SNAPSHOT.write_text(
        json.dumps(
            {
                "taken_at": "2026-07-29T14:54:02+00:00",
                # someone else's baseline, claiming this artifact was ALREADY dirty
                "already_dirty": {"results/experiment_3946_r11l_first_solve.json": " M"},
            }
        )
        + "\n"
    )
    art.write_text('{"experiment": 3946, "duration_s": 9.99}\n')

    assert tsm._load_snapshot() is None, "the shared baseline must not be adopted"
    assert tsm.main(["--check"]) == 1, "and with no baseline of its own, --check refuses"


# ---------------------------------------------------------------------------------------------
# ATTRIBUTION (2026-07-30, third destroyed-work incident)
# ---------------------------------------------------------------------------------------------
# The tool reported "changed since the baseline" and advised on "written by the run" -- two
# different claims -- and printed `git checkout -- <paths>` over the union. That advice is
# unrecoverable and has now been aimed at authored work three times. `classify()` splits the two
# using what the run was OBSERVED writing; only the attributed half is ever reverted or advised.

#: The concurrent lane's authored files in the 2026-07-30 incident. Three are the real paths that
#: lane had open (they were still modified in the tree when this was written); the other three
#: stand in for the rest of the six, which were never recorded individually. The set matters as a
#: CLASS -- prose and state that no test writes -- not as six specific names.
_CONCURRENT_LANE_FILES = (
    "docs/research-notes/arc-agi3-news-watch.md",
    "docs/research-notes/arc-agi3-leaderboard-technique-watch.md",
    "ops/.arc_leaderboard_technique_watch_state.json",
    "docs/research-notes/test-suite-rewrites-the-record-survey-2026-07-29.md",
    "ops/known-issues.md",
    "docs/research-notes/arc-agi3-generalization-floor.md",
)


def _seed_concurrent_lane(r: Path) -> dict[str, bytes]:
    """Commit the six files, then return the AUTHORED content each will carry mid-run."""
    for rel in _CONCURRENT_LANE_FILES:
        p = r / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# {rel}\n\ncommitted baseline prose\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed concurrent lane files")
    return {
        rel: (
            f"# {rel}\n\ncommitted baseline prose\n\n"
            f"## section authored DURING the test run -- not recoverable if reverted\n"
        ).encode()
        for rel in _CONCURRENT_LANE_FILES
    }


def test_authored_edits_are_not_blamed_on_the_run_that_ran_beside_them(repo, monkeypatch, capsys):
    """REPLAY OF THE 2026-07-30 INCIDENT. Six authored files, one real rewrite, one `--check`.

    What happened: a lane took a baseline, a test run rewrote an artifact, and a CONCURRENT agent
    authored six files in the same window. ``--check`` reported all seven as
    "TRACKED FILE(S) MODIFIED BY THE RUN" and printed ``git checkout -- <paths>``. Only a
    by-hand mtime+content audit stopped six files of authored work from being reverted; the same
    advice had already destroyed the survey document twice (its sections 3.6 and 7b.3).

    The survey concluded this was unfixable -- "at the file level they are identical events".
    They are, and they are NOT identical at the process level: the damage mechanism runs inside
    the interpreter, so the run can watch itself. This test pins that the split is real by
    driving both writers for real in a throwaway repo.
    """
    r, art = repo
    authored = _seed_concurrent_lane(r)
    committed = {rel: (r / rel).read_bytes() for rel in _CONCURRENT_LANE_FILES}
    monkeypatch.setenv(tsm.RUN_ID_ENV, "incident-replay")
    tsm.snapshot()

    # (1) THE TEST RUN. An observed Python child rewrites the artifact in place -- the real
    #     `runpy.run_path`-writes-its-artifact mechanism, reduced to its essentials.
    child = subprocess.run(
        [sys.executable, "-c", f"open({str(art)!r},'w').write('{{\"experiment\": 3946}}')"],
        env=tsm.observer_env("incident-replay"),
        cwd=r,
        capture_output=True,
    )
    assert child.returncode == 0, child.stderr

    # (2) THE CONCURRENT LANE, writing in the same window and NOT part of the run.
    for rel, content in authored.items():
        (r / rel).write_bytes(content)

    # Both are "modified since the baseline" -- the signal the old tool had, and all it had.
    muts = tsm.mutations(tsm._load_snapshot("incident-replay"))
    assert len(muts) == 7, muts

    observed = tsm.read_observed("incident-replay")
    attributed, unattributed = tsm.classify(muts, observed)

    assert attributed == ["results/experiment_3946_r11l_first_solve.json"], (
        f"only the file the run was seen writing may be attributed to it: {attributed}"
    )
    assert sorted(unattributed) == sorted(_CONCURRENT_LANE_FILES), (
        f"every authored file must land outside the run's blame: {unattributed}"
    )

    # (3) THE ADVICE. The authored files must not be offered up for reverting.
    assert tsm.main(["--check", "--run-id", "incident-replay"]) == 1
    out = capsys.readouterr().out
    checkout_advice = out.split("UNATTRIBUTED")[0]
    assert "git checkout --" in checkout_advice, "the real rewrite still gets its restore advice"
    for rel in _CONCURRENT_LANE_FILES:
        assert rel not in checkout_advice, (
            f"{rel} was authored, not written by the run -- it must never appear above the "
            f"UNATTRIBUTED heading, where the git checkout advice lives"
        )

    # (4) THE ACT. --restore reverts the rewrite and leaves every authored byte where it is.
    assert tsm.main(["--check", "--run-id", "incident-replay", "--restore"]) == 1
    assert (
        art.read_bytes()
        == json.dumps(
            {
                "experiment": 3946,
                "duration_s": 4.21,
                "solve_provenance": "development_proxy",
                "inference_substrate_correction_note": "2026-07-27: original declaration illegal",
            },
            indent=2,
        ).encode()
        + b"\n"
    ), "the attributed rewrite must be restored to its committed content"
    for rel, content in authored.items():
        assert (r / rel).read_bytes() == content, f"{rel} was reverted -- the incident recurred"
        assert (r / rel).read_bytes() != committed[rel]


def test_an_unobserved_run_attributes_nothing_rather_than_everything(repo, monkeypatch):
    """FAIL SAFE. No observation must mean "cannot attribute", never "attribute it all".

    A run that was killed before it flushed, or one whose writer was not a Python process, leaves
    no log. The tempting reading is "no evidence of another writer, so it must have been the
    run" -- which restores exactly the blanket behaviour that caused all three incidents. The
    honest reading is that nothing is known, so nothing is reverted. The interlock still arms, so
    the record stays protected; only the destructive suggestion is withheld.
    """
    r, art = repo
    authored = _seed_concurrent_lane(r)
    monkeypatch.setenv(tsm.RUN_ID_ENV, "never-observed")
    tsm.snapshot()

    art.write_text('{"experiment": 3946, "rewritten": true}\n')
    for rel, content in authored.items():
        (r / rel).write_bytes(content)

    assert tsm.read_observed("never-observed") is None, "no log was ever written"
    muts = tsm.mutations(tsm._load_snapshot("never-observed"))
    attributed, unattributed = tsm.classify(muts, tsm.read_observed("never-observed"))
    assert attributed == []
    assert len(unattributed) == 7

    assert tsm.main(["--check", "--run-id", "never-observed", "--restore"]) == 1
    for rel, content in authored.items():
        assert (r / rel).read_bytes() == content, f"{rel} was reverted without any evidence"
    assert art.read_text() == '{"experiment": 3946, "rewritten": true}\n', (
        "even the artifact must survive: without observation the tool cannot know the run wrote it"
    )


def test_the_marker_records_which_files_the_run_was_seen_writing(repo, monkeypatch, capsys):
    """--gate must not tell the operator to revert what the run was not seen writing."""
    r, art = repo
    authored = _seed_concurrent_lane(r)
    monkeypatch.setenv(tsm.RUN_ID_ENV, "marker-split")

    rc = tsm.cmd_run(
        [
            sys.executable,
            "-c",
            f"open({str(art)!r},'w').write('{{\"experiment\": 3946}}')",
        ],
        do_restore=False,
    )
    assert rc == 1
    for rel, content in authored.items():
        (r / rel).write_bytes(content)

    marker = json.loads(next(tsm.RUNS.glob("*.pending.json")).read_text())
    assert marker["write_observation"] == "recorded"
    assert marker["attributed_to_run"] == ["results/experiment_3946_r11l_first_solve.json"]

    capsys.readouterr()
    assert tsm.cmd_gate() == 1, "the interlock still refuses -- attribution narrows advice only"
    gate_out = capsys.readouterr().out
    assert "[run wrote it]   results/experiment_3946_r11l_first_solve.json" in gate_out


def test_observation_reaches_a_grandchild_process(repo, monkeypatch, tmp_path):
    """The observer must survive process boundaries, because that is where the writes happen.

    This suite runs under ``-n 4`` (pyproject addopts): the controller collects and the WORKERS
    execute, so an observer that lived only in the process that installs it would watch the one
    interpreter that never calls ``runpy.run_path``. Every real write would then come out
    UNATTRIBUTED -- attribution switched off precisely under the default invocation, which is the
    invocation behind all three incidents.

    ``observer_env`` closes that by putting ``scripts/_mutation_observer`` on PYTHONPATH, so every
    descendant interpreter installs the hook at startup. A python-spawning-python chain is exactly
    the controller/worker shape, one generation deeper for good measure.
    """
    r, _ = repo
    monkeypatch.setenv(tsm.RUN_ID_ENV, "grandchild")
    target = tmp_path / "written_by_the_grandchild.txt"
    grandchild = f"open({str(target)!r},'w').write('x')"
    child = (
        f"import subprocess,sys; subprocess.run([sys.executable,'-c',{grandchild!r}],check=True)"
    )

    proc = subprocess.run(
        [sys.executable, "-c", child],
        env=tsm.observer_env("grandchild"),
        cwd=r,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr

    log = (tsm.RUNS / "grandchild.writes.log").read_text()
    assert str(target) in log, (
        "a write two process generations down was not observed; under -n 4 that is every write "
        f"the suite makes: {log}"
    )


def test_a_new_baseline_discards_the_previous_windows_observations(repo, monkeypatch, tmp_path):
    """Observation must mean "in THIS window", or a stale write becomes a wrong revert.

    The write log is append-only so four xdist workers can share one file without locking. The
    documented agent workflow PINS one ``$CARNOT_MUTATION_RUN_ID`` for a whole session, so without
    a reset the log grows across every run under that id. Run 20's ``--check`` would then still
    see run 1's writes -- and ``--restore`` would revert a file on run 1's evidence, even after a
    human edited it by hand in between. That is the incident this whole change exists to stop,
    arriving by a slower route.
    """
    r, art = repo
    monkeypatch.setenv(tsm.RUN_ID_ENV, "windowed")

    # WINDOW 1: a run writes the artifact.
    tsm.snapshot("windowed")
    subprocess.run(
        [sys.executable, "-c", f"open({str(art)!r},'w').write('{{\"experiment\": 3946}}')"],
        env=tsm.observer_env("windowed"),
        cwd=r,
        check=True,
        capture_output=True,
    )
    assert tsm.read_observed("windowed") == {"results/experiment_3946_r11l_first_solve.json"}
    _git(r, "checkout", "--", ".")

    # WINDOW 2: a NEW baseline, and this time a human edits that same file by hand.
    tsm.snapshot("windowed")
    assert tsm.read_observed("windowed") is None, "the previous window's log must not survive"

    art.write_text('{"experiment": 3946, "edited_by_hand": true}\n')
    muts = tsm.mutations(tsm._load_snapshot("windowed"))
    attributed, unattributed = tsm.classify(muts, tsm.read_observed("windowed"))
    assert attributed == [], "a write from a previous window must not be attributed to this one"
    assert unattributed == ["results/experiment_3946_r11l_first_solve.json"]

    assert tsm.main(["--check", "--run-id", "windowed", "--restore"]) == 1
    assert art.read_text() == '{"experiment": 3946, "edited_by_hand": true}\n', (
        "the hand edit was reverted on a previous window's evidence"
    )


# ---------------------------------------------------------------------------
# Retention: the directory has to stop growing, without ever weakening the gate.
#
# REQ-OPS-MUTATION-RETENTION-6265. Measured 2026-08-13: 3,501 files and 464 MB, 3,441 of them
# `.writes.log` from runs that finished weeks ago. `prune_stale_markers` retires MARKERS; nothing
# ever removed the two debug files each run leaves behind. A guard that silently eats half a
# gigabyte of the operator's disk is a guard they eventually switch off.
#
# The whole risk of a cleanup routine living next to an interlock is that it deletes the
# interlock. So the load-bearing test here is the second one: a pending marker survives at ANY
# age. An old unresolved rewrite is more serious than a recent one, not less.
# ---------------------------------------------------------------------------


def _age(p: Path, days: float) -> None:
    """Backdate a file so the retention cutoff sees it as old."""
    import os
    import time

    t = time.time() - days * 86400
    os.utime(p, (t, t))


def test_prune_old_debris_removes_spent_files_past_the_cutoff(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(tsm, "RUNS", runs)

    old_log = runs / "abc.writes.log"
    old_log.write_text("x" * 100)
    _age(old_log, 30)
    old_snap = runs / "abc.snapshot.json"
    old_snap.write_text("{}")
    _age(old_snap, 30)
    fresh = runs / "def.writes.log"
    fresh.write_text("y")

    n, freed = tsm.prune_old_debris(days=7)

    assert n == 2, "both files older than the cutoff should go"
    assert freed >= 100
    assert not old_log.exists()
    assert not old_snap.exists()
    assert fresh.exists(), "a file inside the retention window must be kept"


def test_prune_old_debris_never_deletes_a_pending_marker_at_any_age(tmp_path, monkeypatch):
    """The one property that matters. A marker is the interlock; debris cleanup must not see it.

    Deliberately backdated a year. Age is not evidence a rewrite was resolved -- only the tree is,
    which is `prune_stale_markers`'s job and requires reading the marker's file list. If this test
    ever fails, the gate can be silenced by waiting.
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(tsm, "RUNS", runs)

    marker = runs / "ancient.pending.json"
    marker.write_text(json.dumps({"modified_tracked_files": ["results/x.json"]}))
    _age(marker, 365)

    n, _ = tsm.prune_old_debris(days=7)

    assert n == 0
    assert marker.exists(), "a pending marker must survive debris cleanup at any age"


def test_prune_old_debris_fails_open_when_a_file_cannot_be_removed(tmp_path, monkeypatch):
    """Cleanup must never break the gate it rides on.

    This is the OPPOSITE choice from `prune_stale_markers`, which fails closed. That one refuses
    when it cannot prove damage is resolved. This one only removes debris, so an unreadable or
    undeletable file costs disk and nothing else -- skip it and carry on.
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(tsm, "RUNS", runs)

    doomed = runs / "gone.writes.log"
    doomed.write_text("z")
    _age(doomed, 30)
    survivor = runs / "other.writes.log"
    survivor.write_text("zz")
    _age(survivor, 30)

    real_unlink = Path.unlink

    def flaky(self, *a, **k):
        if self.name == "gone.writes.log":
            raise OSError("permission denied")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", flaky)

    n, _ = tsm.prune_old_debris(days=7)  # must not raise

    assert n == 1, "the deletable file is still cleaned up"
    assert doomed.exists()
    assert not survivor.exists()


# ---------------------------------------------------------------------------
# The pre-commit stash window (QA-layer SILENT_NON_FIRING finding, 2026-08-23)
# ---------------------------------------------------------------------------


def _arm_marker_and_stash(repo_tuple, tmp_path, monkeypatch, *, patch_age_s=0.0):
    """Marker names a committed path; the tree is CLEAN for it; a pre-commit
    patch (the stash) names it. That is the hook-run window exactly."""
    r, _ = repo_tuple
    hidden_rel = "output/kanele_synth/post_synth.dcp"
    target = r / hidden_rel
    target.parent.mkdir(parents=True)
    target.write_text("synth checkpoint v1")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "add synth artifact")
    tsm.write_pending([hidden_rel], ["pytest", "tests/python"])
    cache = tmp_path / "pre-commit-cache"
    cache.mkdir(parents=True, exist_ok=True)
    patch = cache / "patch1787430000-424242"
    patch.write_text(
        f"diff --git a/{hidden_rel} b/{hidden_rel}\n"
        f"--- a/{hidden_rel}\n"
        f"+++ b/{hidden_rel}\n"
        "@@ -1 +1 @@\n"
        "-synth checkpoint v1\n"
        "+synth checkpoint REWRITTEN BY A TEST\n"
    )
    if patch_age_s:
        import os as _os
        import time as _time

        old = _time.time() - patch_age_s
        _os.utime(patch, (old, old))
    return hidden_rel


def test_unstaged_rewrite_hidden_by_precommit_stash_still_blocks(repo, tmp_path, monkeypatch):
    """Named for the reported input: an unresolved unstaged test rewrite of
    output/kanele_synth/post_synth.dcp temporarily hidden while pre-commit
    runs the gate. The tree looks clean because the stash holds the rewrite;
    the marker must NOT retire and the gate must refuse."""
    hidden_rel = _arm_marker_and_stash(repo, tmp_path, monkeypatch)
    assert hidden_rel not in tsm.dirty_tracked()  # the stash illusion, pinned
    assert hidden_rel in tsm._stash_hidden_paths()
    blocking = tsm.prune_stale_markers()
    assert blocking, "marker must survive the stash window"
    assert tsm.cmd_gate() == 1


def test_stash_hidden_gate_output_warns_against_checkout(repo, tmp_path, monkeypatch, capsys):
    """The refusal must say the rewrite is stash-hidden, not gone — a git
    checkout from inside the window is the destroy-authored-work class."""
    _arm_marker_and_stash(repo, tmp_path, monkeypatch)
    assert tsm.cmd_gate() == 1
    out = capsys.readouterr().out
    assert "stash-hidden" in out
    assert "output/kanele_synth/post_synth.dcp" in out


def test_stale_patch_does_not_wedge_the_gate(repo, tmp_path, monkeypatch):
    """A patch from a long-finished run must not keep a resolved marker
    blocking: freshness bounds the fail-closed direction."""
    _arm_marker_and_stash(repo, tmp_path, monkeypatch, patch_age_s=3600.0)
    assert tsm._stash_hidden_paths() == set()
    assert tsm.prune_stale_markers() == []
    assert tsm.cmd_gate() == 0


def test_no_patch_dir_means_no_hidden_paths(repo):
    """Missing cache dir (fresh machine) is simply 'nothing stashed'."""
    assert tsm._stash_hidden_paths() == set()


# --- REQ-OPS-MUTATION-PROOF-1: a mutation PROOF is exclusive, verified, and not inert ----------
#
# INCIDENT, 2026-08-25, observed not theorised. Two hand-run mutation proofs ran against this
# working tree at once. The tree carried, on the LIVE ARC scored path:
#
#     python/carnot/agentic/arc_executable_world_model.py:6466: pass  # MUTATED M6
#
# `pass  # MUTATED` is valid Python that clears every hook, and the conductor commits with
# `git add -A` and hooks skipped on its own schedule, so a checkpoint inside a mutation window
# publishes it silently. Readings were contaminated in BOTH directions -- a RED credited to your
# own deleted pattern can be the other run's mutation -- and an 11-mutation proof set was voided.
#
# Separately: a proof run in a `git worktree` copy is silently INERT, because
# `.venv/.../__editable__.carnot_ebm.pth` pins the absolute path of the MAIN checkout. The
# mutated file is never imported, every mutation reads GREEN, and the proof proves nothing.

_INCIDENT_LINE = "    pass  # MUTATED M6\n"


def _begin(target: Path, run_id: str | None = None) -> int:
    return tsm.cmd_mutation_begin(str(target), run_id)


def test_a_second_concurrent_proof_is_refused_and_names_the_holder(repo, capsys):
    """THE INCIDENT: two proofs at once. The second must refuse, not open."""
    r, _ = repo
    target = r / "victim.py"
    target.write_text("def f():\n    return 1\n")
    assert _begin(target, "run-A") == 0
    capsys.readouterr()

    other = r / "other.py"
    other.write_text("def g():\n    return 2\n")
    assert _begin(other, "run-B") == 1, "a second proof must not open while one is held"
    out = capsys.readouterr().out
    assert "already open" in out
    assert "run-A" in out, "the refusal must name which run holds the lock"
    assert str(target) in out, "and what that run is mutating"


def test_a_surviving_mutated_marker_fails_the_session_loudly(repo, capsys):
    """The exact incident line must not survive a closed session."""
    r, _ = repo
    target = r / "victim.py"
    target.write_text("def f():\n    return 1\n")
    assert _begin(target, "run-A") == 0
    capsys.readouterr()

    target.write_text("def f():\n" + _INCIDENT_LINE)
    assert tsm.cmd_mutation_end("run-A") == 1, "a surviving marker must fail the session"
    out = capsys.readouterr().out
    assert "SESSION NOT CLEAN" in out
    assert "victim.py:2" in out, "the surviving marker must be named with its line"
    assert tsm.PROOF_LOCK.exists(), "the lock is KEPT so the marker cannot be left behind"


def test_a_marker_in_a_file_other_than_the_target_is_still_caught(repo, capsys):
    """Scope is the blast radius, not the declared target. The 2026-08-25 marker sat in a file
    the concurrent session had never declared."""
    r, _ = repo
    target = r / "victim.py"
    target.write_text("def f():\n    return 1\n")
    assert _begin(target, "run-A") == 0
    capsys.readouterr()

    stray = r / "results" / "elsewhere.py"
    stray.write_text("x = 1  # MUTATED by the other harness\n")
    assert tsm.cmd_mutation_end("run-A") == 1
    assert "elsewhere.py:1" in capsys.readouterr().out


def test_an_unrestored_target_fails_even_with_no_marker(repo, capsys):
    """Byte-identity, not marker absence. A mutation stripped of its comment is still a
    mutation, and 'no marker' is not 'restored'."""
    r, _ = repo
    target = r / "victim.py"
    target.write_text("def f():\n    return 1\n")
    assert _begin(target, "run-A") == 0
    capsys.readouterr()

    target.write_text("def f():\n    pass\n")  # mutated, marker deliberately omitted
    assert tsm.cmd_mutation_end("run-A") == 1
    assert "NOT byte-identical" in capsys.readouterr().out


def test_a_clean_restore_closes_the_session_and_releases_the_lock(repo, capsys):
    """GREEN on restore: byte-identical target, no marker anywhere, lock released."""
    r, _ = repo
    target = r / "victim.py"
    original = "def f():\n    return 1\n"
    target.write_text(original)
    assert _begin(target, "run-A") == 0
    target.write_text("def f():\n" + _INCIDENT_LINE)
    target.write_text(original)  # restored byte-identical
    capsys.readouterr()
    assert tsm.cmd_mutation_end("run-A") == 0
    assert "CLOSED" in capsys.readouterr().out
    assert not tsm.PROOF_LOCK.exists()


def test_a_pre_existing_marker_refuses_the_begin(repo, capsys):
    """Opening over someone else's marker would attribute their mutation to your proof --
    the reading contamination that voided the 11-mutation set."""
    r, _ = repo
    (r / "stray.py").write_text(_INCIDENT_LINE.strip() + "\n")
    target = r / "victim.py"
    target.write_text("def f():\n    return 1\n")
    assert _begin(target, "run-A") == 1
    assert "already in the tree" in capsys.readouterr().out
    assert not tsm.PROOF_LOCK.exists()


def test_mutation_end_without_a_session_refuses(repo, capsys):
    """Fail closed: with no recorded target and no pre-mutation hash, 'restored' is a guess."""
    assert tsm.cmd_mutation_end("run-A") == 1
    assert "no mutation-proof session is open" in capsys.readouterr().out


def test_mutation_end_without_a_run_id_refuses_to_close_anyones_session(repo, capsys, monkeypatch):
    """THE SECOND HALF OF THE INCIDENT. An unpinned close used to skip the ownership check
    entirely, so a second agent in a fresh shell closed whoever's session was open -- and
    landing between two mutations, the tree looked clean, so it succeeded."""
    r, _ = repo
    target = r / "victim.py"
    target.write_text("x = 1\n")
    assert _begin(target, "run-A") == 0
    capsys.readouterr()
    # A fresh shell: no --run-id and no pinned env var.
    monkeypatch.delenv(tsm.RUN_ID_ENV, raising=False)
    assert tsm.cmd_mutation_end(None) == 1
    out = capsys.readouterr().out
    assert "no run id" in out and "run-A" in out
    assert tsm.PROOF_LOCK.exists(), "an unidentified close must not release the lock"


def test_an_unreadable_lock_refuses_rather_than_being_reclaimed(repo, capsys):
    """Unreadable is not unheld."""
    r, _ = repo
    tsm.PROOF_LOCK.parent.mkdir(parents=True, exist_ok=True)
    tsm.PROOF_LOCK.write_text("{ this is not json")
    target = r / "victim.py"
    target.write_text("x = 1\n")
    assert _begin(target, "run-A") == 1
    assert "unreadable" in capsys.readouterr().out


def test_a_fresh_lock_from_a_dead_pid_is_still_honoured(repo, capsys):
    """The defect this rule was rebuilt around, pinned as a test.

    A proof session spans two CLI calls, so the process that wrote the lock has ALWAYS exited by
    the time anyone reads it. The first build reclaimed on 'holder is dead' and therefore
    reclaimed every time -- it printed "session OPEN" for a second caller and locked nothing.
    Age decides; liveness may only ever add a refusal.
    """
    r, _ = repo
    tsm.PROOF_LOCK.parent.mkdir(parents=True, exist_ok=True)
    tsm.PROOF_LOCK.write_text(
        json.dumps(
            {
                "run_id": "run-DEAD",
                "pid": 999_999_999,  # certainly not running
                "host": "some-other-host",
                "target": str(r / "victim.py"),
                "started_at": tsm.datetime.now(tsm.UTC).isoformat(),
            }
        )
    )
    target = r / "victim.py"
    target.write_text("x = 1\n")
    assert _begin(target, "run-B") == 1, "a dead pid alone must not authorise a reclaim"
    assert "already open" in capsys.readouterr().out


def test_a_live_holder_is_never_reclaimed_however_old_the_lock(repo, capsys):
    """The liveness clause must BITE. Blanking it left the suite green in review, which made
    it decorative -- and age alone would reclaim a legitimately long proof."""
    r, _ = repo
    ancient = tsm.datetime.now(tsm.UTC) - timedelta(seconds=tsm.PROOF_LOCK_STALE_S * 10)
    tsm.PROOF_LOCK.parent.mkdir(parents=True, exist_ok=True)
    tsm.PROOF_LOCK.write_text(
        json.dumps(
            {
                "run_id": "run-LIVE",
                "pid": os.getpid(),  # this process is certainly running
                "host": os.uname().nodename,
                "target": str(r / "victim.py"),
                "started_at": ancient.isoformat(),
            }
        )
    )
    target = r / "victim.py"
    target.write_text("x = 1\n")
    assert _begin(target, "run-B") == 1, "a running holder must never be reclaimed on age"
    assert "still running" in capsys.readouterr().out


def test_unknowable_liveness_is_honoured_not_reclaimed(repo, monkeypatch, capsys):
    """_pid_alive's stated contract: None means REFUSE. The caller used to write `if alive:`,
    so None fell through to the age path -- a contract documented and not implemented."""
    r, _ = repo
    monkeypatch.setattr(tsm, "_pid_alive", lambda pid: None)
    old = tsm.datetime.now(tsm.UTC) - timedelta(seconds=tsm.PROOF_LOCK_STALE_S + 60)
    tsm.PROOF_LOCK.parent.mkdir(parents=True, exist_ok=True)
    tsm.PROOF_LOCK.write_text(
        json.dumps(
            {
                "run_id": "run-UNKNOWN",
                "pid": 4242,
                "host": os.uname().nodename,
                "target": str(r / "victim.py"),
                "started_at": old.isoformat(),
            }
        )
    )
    target = r / "victim.py"
    target.write_text("x = 1\n")
    assert _begin(target, "run-B") == 1
    assert "liveness could not be checked" in capsys.readouterr().out


def test_an_abandoned_lock_past_the_stale_window_is_reclaimed_loudly(repo, capsys):
    """A crashed proof must not wedge every future proof forever."""
    r, _ = repo
    old = tsm.datetime.now(tsm.UTC) - timedelta(seconds=tsm.PROOF_LOCK_STALE_S + 60)
    tsm.PROOF_LOCK.parent.mkdir(parents=True, exist_ok=True)
    tsm.PROOF_LOCK.write_text(
        json.dumps(
            {
                "run_id": "run-CRASHED",
                "pid": 999_999_999,
                "host": "some-other-host",
                "target": str(r / "victim.py"),
                "started_at": old.isoformat(),
            }
        )
    )
    target = r / "victim.py"
    target.write_text("x = 1\n")
    assert _begin(target, "run-B") == 0
    out = capsys.readouterr().out
    assert "reclaiming an abandoned lock" in out
    assert "run-CRASHED" in out
    assert "may still be in the tree" in out, "a reclaim must warn, never be silent"


def test_an_inert_worktree_run_refuses_rather_than_reporting_a_clean_pass(
    repo, capsys, monkeypatch
):
    """THE INERT-RUN TRAP, as the real editable install produces it.

    A file under `python/` whose dotted name resolves to a DIFFERENT file must refuse. This is
    the shape a `git worktree` proof takes here: the editable .pth pins the main checkout, so
    the worktree copy is never the file the tests import.
    """
    r, _ = repo
    pkgdir = r / "python" / "carnot" / "agentic"
    pkgdir.mkdir(parents=True)
    (r / "python" / "carnot" / "__init__.py").write_text("")
    (pkgdir / "__init__.py").write_text("")
    target = pkgdir / "arc_executable_world_model.py"
    target.write_text("x = 1\n")
    # find_spec resolves against the REAL installed carnot, which lives in the main checkout,
    # so this target can never be the file the interpreter imports.
    ok, why = tsm.check_target_is_live(target)
    assert not ok, "a worktree copy of an installed module must not pass as live"
    assert "INERT RUN" in why
    assert _begin(target, "run-A") == 1
    assert "INERT RUN" in capsys.readouterr().out
    assert not tsm.PROOF_LOCK.exists(), "an inert run must not hold the lock"

    monkeypatch.setattr(tsm, "_installed_package_root", lambda: None)
    ok, why = tsm.check_target_is_live(target)
    assert not ok
    assert "cannot tell which checkout" in why

    # Package __init__ modules resolve through the directory fallback rather than `carnot.py`.
    monkeypatch.setattr(tsm, "_installed_package_root", lambda: r / "python")
    ok, why = tsm.check_target_is_live(r / "python" / "carnot" / "__init__.py")
    assert ok
    assert "carnot resolves to the file being mutated" in why


def test_a_file_outside_the_package_is_declared_not_applicable_not_silently_passed(
    repo, monkeypatch
):
    """`scripts/*.py` is loaded by explicit path, so no import can be diverted. The check says
    so in words rather than returning a bare True nobody can audit."""
    r, _ = repo
    target = r / "some_script.py"
    target.write_text("x = 1\n")
    ok, why = tsm.check_target_is_live(target)
    assert ok, "a path-loaded file must not be refused; the refusal would cry wolf"
    assert "NOT APPLICABLE" in why
    assert "loaded by explicit path" in why
    # It must still SAY when carnot resolves elsewhere, rather than passing in silence.
    assert "CAUTION" in why or "checkout `carnot` resolves to" in why

    # Applicability is a property of the target, not of whether this interpreter happens to
    # expose its editable install as a plain sys.path entry. A hermetic runner may resolve
    # `carnot` through an import hook instead; that cannot turn an explicitly loaded script into
    # an import-resolution failure.
    monkeypatch.setattr(tsm, "_installed_package_root", lambda: None)
    ok, why = tsm.check_target_is_live(target)
    assert ok
    assert "NOT APPLICABLE" in why
    assert "loaded by explicit path" in why

    monkeypatch.setattr(tsm, "_installed_package_root", lambda: r / "python")
    ok, why = tsm.check_target_is_live(target)
    assert ok
    assert "NOT APPLICABLE" in why
    assert "checkout `carnot` resolves to" in why


def test_a_missing_target_refuses(repo):
    r, _ = repo
    ok, why = tsm.check_target_is_live(r / "nope.py")
    assert not ok and "does not exist" in why


def test_force_unlock_clears_deliberately_and_says_what_it_cleared(repo, capsys):
    r, _ = repo
    target = r / "victim.py"
    target.write_text("x = 1\n")
    assert _begin(target, "run-A") == 0
    capsys.readouterr()
    assert tsm.cmd_mutation_force_unlock() == 0
    out = capsys.readouterr().out
    assert "CLEARED by hand" in out and "run-A" in out
    assert "still in the tree" in out
    assert not tsm.PROOF_LOCK.exists()


def test_prose_that_quotes_the_marker_does_not_brick_the_session(repo, capsys):
    """THE SELF-INFLICTED BRICK. The first build scanned every changed file, so this project's
    own spec, changelog and research notes -- which quote `pass  # MUTATED M6` verbatim while
    documenting the incident -- refused every open. A marker only does damage in a file that
    RUNS; prose describing one is not a mutation."""
    r, _ = repo
    (r / "ops").mkdir(exist_ok=True)
    (r / "ops" / "changelog.md").write_text(
        "The tree carried `pass  # MUTATED M6` on the live scored path.\n"
    )
    (r / "spec.md").write_text("Given a tree carrying a surviving `MUTATED` marker, close...\n")
    target = r / "victim.py"
    target.write_text("x = 1\n")
    assert _begin(target, "run-A") == 0, "prose quoting the marker must not refuse the open"
    capsys.readouterr()
    assert tsm.cmd_mutation_end("run-A") == 0


def test_the_scanner_still_catches_a_marker_in_every_source_language_it_claims(repo, capsys):
    """The suffix set is the concept 'files that run', so each one must actually be scanned."""
    r, _ = repo
    target = r / "victim.py"
    target.write_text("x = 1\n")
    assert _begin(target, "run-A") == 0
    capsys.readouterr()
    (r / "helper.sh").write_text("true  # MUTATED M6\n")
    assert tsm.cmd_mutation_end("run-A") == 1
    assert "helper.sh:1" in capsys.readouterr().out


def test_mutation_end_refuses_a_lock_belonging_to_another_run(repo, capsys):
    r, _ = repo
    target = r / "victim.py"
    target.write_text("x = 1\n")
    assert _begin(target, "run-A") == 0
    capsys.readouterr()
    assert tsm.cmd_mutation_end("run-B") == 1
    assert "belongs to another run" in capsys.readouterr().out
    assert tsm.PROOF_LOCK.exists()


def test_a_marker_inside_a_new_untracked_directory_is_caught(repo, capsys):
    """F3. Plain `--porcelain` collapses a wholly-untracked directory to one `dir/` entry,
    which is not a file, so the entire subtree was skipped in silence. A scratch copy of the
    file under mutation lives in exactly such a directory."""
    r, _ = repo
    target = r / "victim.py"
    target.write_text("x = 1\n")
    assert _begin(target, "run-A") == 0
    capsys.readouterr()

    scratch = r / "scratch_copies" / "deep"
    scratch.mkdir(parents=True)
    (scratch / "probe.py").write_text("x = 1  # MUTATED M6\n")
    assert tsm.cmd_mutation_end("run-A") == 1
    assert "probe.py:1" in capsys.readouterr().out


def test_a_marker_committed_mid_session_is_caught(repo, capsys):
    """F4, the ORIGIN MECHANISM. The conductor commits on its own schedule, so a marker swept
    into a mid-session commit leaves the working tree CLEAN. Closing on `git status` alone
    reported no marker while the marker sat in HEAD -- published, which is worse."""
    r, _ = repo
    target = r / "victim.py"
    target.write_text("x = 1\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed victim")
    assert _begin(target, "run-A") == 0
    capsys.readouterr()

    (r / "swept.py").write_text("y = 2  # MUTATED M6\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "conductor checkpoint")
    assert _git(r, "status", "--porcelain").strip() == "", "the tree must look clean"
    assert tsm.cmd_mutation_end("run-A") == 1, "a committed marker must still fail the session"
    assert "swept.py:1" in capsys.readouterr().out


def test_an_unscannable_file_refuses_rather_than_being_noted(repo, capsys, monkeypatch):
    """F7. A file nobody looked at is not a clean file. The unscanned list used to be dropped
    entirely at open and merely printed at close."""
    r, _ = repo
    target = r / "victim.py"
    target.write_text("x = 1\n")
    monkeypatch.setattr(tsm, "_MARKER_SCAN_MAX_BYTES", 10)
    (r / "big.py").write_text("# " + "x" * 500 + "\n")
    assert _begin(target, "run-A") == 1
    out = capsys.readouterr().out
    assert "could not be scanned" in out and "big.py" in out
    assert not tsm.PROOF_LOCK.exists()


def test_the_gate_refuses_a_commit_while_a_proof_is_open(repo, capsys):
    """F13. A commit taken mid-proof is how a mutated line reaches the record."""
    r, _ = repo
    target = r / "victim.py"
    target.write_text("x = 1\n")
    assert _begin(target, "run-A") == 0
    capsys.readouterr()
    assert tsm.cmd_gate() == 1
    assert "a mutation proof is open" in capsys.readouterr().out


def test_the_cli_dispatches_every_mutation_flag(repo, capsys):
    """F12. The feature is ONLY reachable through the CLI, and blanking the dispatch left the
    suite green -- so the flags could stop working with nothing to show for it."""
    r, _ = repo
    target = r / "victim.py"
    target.write_text("x = 1\n")

    assert tsm.main(["--mutation-begin", "--mutation-target", str(target), "--run-id", "cli"]) == 0
    assert "session OPEN" in capsys.readouterr().out
    assert tsm.PROOF_LOCK.exists()

    assert tsm.main(["--mutation-end", "--run-id", "cli"]) == 0
    assert "CLOSED" in capsys.readouterr().out
    assert not tsm.PROOF_LOCK.exists()

    assert tsm.main(["--mutation-begin", "--mutation-target", str(target), "--run-id", "cli"]) == 0
    capsys.readouterr()
    assert tsm.main(["--mutation-force-unlock"]) == 0
    assert "CLEARED by hand" in capsys.readouterr().out


def test_the_cli_refuses_mutation_begin_without_a_target(repo, capsys):
    assert tsm.main(["--mutation-begin"]) == 1
    assert "needs --mutation-target" in capsys.readouterr().out


def test_the_proof_lock_is_shared_across_worktrees_of_one_repo(repo):
    """F5. This repo has ~10 live worktrees sharing one venv. A per-checkout lock is void in
    exactly the workflow the project uses for proofs, so the lock lives under the shared
    `--git-common-dir`, not under the checkout."""
    r, _ = repo
    resolved = tsm._proof_lock_path()
    common = Path(_git(r, "rev-parse", "--git-common-dir").strip())
    if not common.is_absolute():
        common = (r / common).resolve()
    # No `or ops/` fallback: that made this vacuous, and blanking the git-common-dir lookup
    # left the suite GREEN. The lock MUST live under the dir every worktree shares.
    assert resolved.parent == common, (
        f"lock at {resolved} is per-checkout; it must sit under the shared {common}"
    )
    assert resolved.name == "carnot_mutation_proof.lock"


def test_an_unscannable_file_fails_the_close_not_just_the_open(repo, capsys, monkeypatch):
    """M14. The open-side refusal was tested; the close-side was not, and blanking the close
    check left the suite GREEN. A file nobody could read is not a file with no marker."""
    r, _ = repo
    target = r / "victim.py"
    target.write_text("x = 1\n")
    assert _begin(target, "run-A") == 0
    capsys.readouterr()
    # Only now does the unscannable file appear, so the open could not have caught it.
    (r / "big.py").write_text("# " + "x" * 500 + "\n")
    monkeypatch.setattr(tsm, "_MARKER_SCAN_MAX_BYTES", 10)
    assert tsm.cmd_mutation_end("run-A") == 1
    out = capsys.readouterr().out
    assert "could not be scanned" in out and "big.py" in out
    assert tsm.PROOF_LOCK.exists(), "the lock is kept until the tree can actually be read"
