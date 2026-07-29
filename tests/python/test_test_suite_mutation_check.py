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
import uuid
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
    # The run id is otherwise derived from the parent PID, which every xdist worker shares -- so
    # without this, concurrent workers would collide on one snapshot name and the ambiguity
    # refusal would fire inside unrelated tests.
    monkeypatch.setenv(tsm.RUN_ID_ENV, f"test-{uuid.uuid4().hex[:12]}")
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
    assert "MODIFIED BY THE RUN" in out, (
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
    for name in ("SNAPSHOT", "PENDING", "RUNS", "BACKUP", "REPO"):
        p = Path(getattr(tsm, name))
        assert r in p.parents or p == r, (
            f"tsm.{name} points at {p}, outside the throwaway repo {r} -- add it to the fixture"
        )


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
