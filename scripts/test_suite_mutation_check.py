#!/usr/bin/env python3
"""Answer one question: did running the tests MODIFY TRACKED FILES, and which ones?

WHY THIS EXISTS (2026-07-29, found by the outer loop)
-----------------------------------------------------
Running ``pytest tests/python/test_arc_*.py tests/python/test_experiment_*.py`` left 39 tracked
files modified that were clean before the run: 36 ``results/*.json`` / ``.log`` artifacts, plus
``openspec/papers/paper-v6/section-6-limitations.md`` and ``output/kanele_synth/post_synth.dcp``.

The mechanism is not exotic. A class of ``tests/python/test_experiment_*.py`` calls
``runpy.run_path`` on the REAL ``scripts/experiments/experiment_NNNN_*.py`` and then asserts the
artifact exists. The script writes its artifact as a side effect, at the same fixed path the
historical artifact lives at. So the test does not merely READ the research record -- it
OVERWRITES it, and it does so on a green run, with no failure and no diff anyone reads.

That matters because ``results/*.json`` is not decoration. It is the input to the fabrication
gate (``scripts/adversarial_verify.py``), to every capstone aggregation, and to the paper.
Confirmed damage from one run:

  * ``experiment_3946_r11l_first_solve.json``  lost ``inference_substrate_correction_note``,
    ``inference_substrate_original_invalid_value``, ``solve_provenance`` and
    ``solve_provenance_note`` -- a hand-written corrigendum, gone.
  * ``experiment_307_jepa_real_training.json`` had ``inference_mode`` flipped
    ``live_gpu`` -> ``cpu_training``.
  * ``experiment_1035_dualgpu_rocm_v3.json``   had its run timestamps rewritten to today.

Anyone who runs the suite and then commits with ``git add -A`` publishes those rewrites.

WHY A DETECTOR AND NOT A FIX
----------------------------
Repairing the tests themselves is a design call with an operator-visible tradeoff -- redirect
each script's output directory, snapshot-and-restore around the ``runpy`` call, or stop
re-running the script and assert against the committed artifact instead (which stops exercising
the script at all). That choice belongs to the operator. Detection does not: knowing whether a
test run touched the record is a fact, needs no judgement, and is useful under every possible
repair.

This script is also deliberately NOT clever. It asks git what changed. It does not try to
classify a change as benign, it does not diff semantics, and it never edits an artifact unless
you explicitly pass ``--restore``.

USAGE
-----
    # Wrap a test run: snapshot, run, report, and refuse (exit 1) if the record moved.
    python3 scripts/test_suite_mutation_check.py --run -- pytest tests/python/test_arc_frame_induction.py

    # ...and put everything the run touched back the way it was. Every reverted file's previous
    # content is copied to ops/.test_suite_mutation_backup/ first -- see the warning below.
    python3 scripts/test_suite_mutation_check.py --restore --run -- pytest tests/python

DO NOT EDIT TRACKED FILES WHILE A ``--restore`` RUN IS IN FLIGHT
---------------------------------------------------------------
``--restore`` reverts everything that changed since the snapshot, and it CANNOT tell a test's
write from your concurrent edit -- both are "modified since the snapshot". Editing a tracked file
during a wrapped run therefore gets that edit reverted. This has happened twice, both times to
``docs/research-notes/test-suite-rewrites-the-record-survey-2026-07-29.md`` while it was being
written. Since authorship is not recoverable in principle, the revert is made SURVIVABLE instead:
every file is copied to ``ops/.test_suite_mutation_backup/<path>`` before it is reverted. If a
restore takes something it should not have, copy it back from there.

    # Or do it by hand, around anything at all. --snapshot prints the run id and the exact
    # --check line to pair with it; pass --run-id (or export CARNOT_MUTATION_RUN_ID for the whole
    # workflow) so a concurrent workflow's baseline can never be read instead of yours.
    python3 scripts/test_suite_mutation_check.py --snapshot
    <do the thing>
    python3 scripts/test_suite_mutation_check.py --check --run-id <id>

    # Pre-commit interlock (see below); no arguments.
    python3 scripts/test_suite_mutation_check.py --gate

Exit 0 = the tracked tree is where you left it. Exit 1 = it moved, and the names are printed.

THE PRE-COMMIT INTERLOCK
------------------------
``--run`` writes ``ops/.test_suite_mutation_pending.json`` whenever it detects mutations it did
not restore. ``--gate`` (wired as a pre-commit hook) refuses any commit while that file exists.
So an unattended agent that runs the suite through the wrapper, sees the record move, and then
tries to ``git add -A && git commit`` is stopped rather than publishing the rewrite. Clearing
the marker is a deliberate act: restore the files, or state in the commit that the rewrite is
intended, and delete the marker.

This is an interlock, not a sandbox. It cannot help a run that bypasses the wrapper entirely --
for that layer, ``scripts/determination_preservation_lint.py`` inspects the CONTENT of every
staged ``results/*.json`` and refuses the specific never-prune regressions regardless of how
they got there. The two are complementary: this one is broad and shallow (any tracked file, no
opinion about content), that one is narrow and deep (one directory, strong opinions).

CONCURRENCY: WHY THERE IS NO LONGER A SHARED MUTABLE CELL (2026-07-29, second review)
------------------------------------------------------------------------------------
The first version kept exactly two files -- ONE snapshot and ONE pending marker -- and both are
written by every run. With two or three workflows live in the same working tree (the normal
condition in this project) each is a shared mutable cell, and both races were reproduced
directly rather than argued about:

  RACE 1, the interlock disarms itself. ``arm_from_pytest`` and ``cmd_run`` both did an
  unconditional ``PENDING.unlink()`` on a clean run. So workflow A detects a rewrite and arms the
  marker; workflow B then finishes an unrelated CLEAN run and DELETES A's marker. Replayed in a
  throwaway repo: after B, ``--gate`` printed OK and exited 0 while A's rewrite was still sitting
  in the working tree as ``M results/artA.json``. The interlock is the only thing standing between
  a green test run and a published rewrite, and a passing test run switched it off.

  RACE 2, the baseline goes backwards and the check answers wrong. ``--snapshot`` overwrites the
  single snapshot file. A takes a baseline on a clean tree; B then edits a file in-flight and takes
  ITS baseline (now containing that file); A's run rewrites the same file; A's ``--check`` reads
  B's baseline, sees the file listed as "already dirty", and reports "OK -- no tracked file was
  modified". A FALSE NEGATIVE on exactly the damage the tool exists to find. This is the shape the
  operator observed live: a 14:52Z snapshot replaced by one stamped 11:43:01Z.

THE FIX -- and why it is not a lock, and not "compare against HEAD":

  * NOT "compare ``--check`` against git HEAD instead of a stored baseline". That removes the
    shared state but destroys the semantics, and the cost is measurable rather than theoretical:
    at the time of writing this tree carries 84 dirty files from a concurrent workflow's migration,
    every one of which HEAD-comparison would report as a test-run mutation. Excluding
    already-dirty work IS the mechanism (see ``mutations()``); without a baseline the detector
    cannot distinguish a test's rewrite from a human's in-flight edit and becomes unusable on any
    real working tree.
  * NOT an advisory lock. A lock cannot span the documented ``--snapshot`` / <do the thing> /
    ``--check`` sequence -- that is two processes with arbitrary work in between -- and serialising
    concurrent workflows to protect a diagnostic is a worse cure than the disease.
  * INSTEAD: there is no shared mutable cell left. Every run writes its OWN file under
    ``ops/.test_suite_mutation_runs/``, and no run ever overwrites or deletes another's.

    - Snapshots are ``<run-id>.snapshot.json``. The run id comes from ``--run-id``, else
      ``$CARNOT_MUTATION_RUN_ID``, else an auto id containing the parent PID and a timestamp, so
      two ``--snapshot`` calls can never collide even when both workflows are children of the same
      shell (they are, in this environment -- PPID alone does NOT discriminate here, which is why
      the timestamp is in the id). ``--check`` resolves the id explicitly, or auto-matches this
      session's snapshots: exactly one match is used, and ZERO or MORE THAN ONE is REFUSED with the
      candidates listed. It never silently borrows another run's baseline -- that borrowing is
      Race 2.
    - Pending markers are ``<uuid>.pending.json``, one per arming event, never rewritten.
      ``--gate`` refuses while ANY marker survives.
    - Clearing is derived from the TREE, not from ownership: a marker is stale exactly when every
      file it names is clean again, which is a checkable fact about the repo rather than a claim
      about who owns what. A clean run therefore retires its own marker (the "one bad run must not
      wedge commits forever" property) without being able to retire a marker whose damage is still
      on disk -- which is Race 1, closed. A marker that cannot be parsed is NEVER pruned: unreadable
      is not the same as resolved, so it keeps refusing.

  The legacy single ``ops/.test_suite_mutation_pending.json`` is still honoured on read, so a tree
  that armed it under the old code stays blocked rather than silently unblocking on upgrade.

FAIL CLOSED ON GIT ERRORS (2026-07-29, same review)
---------------------------------------------------
``_git`` used to swallow a non-zero git exit and return ``""``. Every caller then saw an empty
dirty-set, so ``mutations()`` returned nothing and the tool printed "OK -- no tracked file was
modified by the run". A broken git made the detector report a clean tree. That is the same
fail-open shape as the sibling lint's ``--diff-filter=M`` hole, and it is worse here because
``--gate`` is a pre-commit hook: the one moment the answer matters is the moment it would lie.
``_git`` now raises ``GitError`` and every command path turns that into a REFUSAL with the git
stderr attached. ``tests/python/conftest.py`` catches exceptions around its two calls by its own
deliberate design (a guard must not break the suite it guards), so raising does not change the
pytest path's behaviour -- it only stops the CLI from answering when it cannot know.

Cross-references:
- commit ``b3e31d341``                                -> the hazard as first diagnosed
- ``scripts/determination_preservation_lint.py``      -> the content-level sibling guard
- ``ops/known-issues.md`` 2026-07-29                  -> the survey and the repair options
- CLAUDE.md "Documentation Update Rules" (never-prune) -> what the rewrites violate
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# LEGACY single-file state. Nothing WRITES these any more -- see the CONCURRENCY section of the
# module docstring -- but `SNAPSHOT` is still readable and `PENDING` is still honoured by the gate,
# so a working tree that armed the interlock under the old code does not silently unblock when this
# version lands. New state is per-run, under RUNS/.
SNAPSHOT = REPO / "ops" / ".test_suite_mutation_snapshot.json"
PENDING = REPO / "ops" / ".test_suite_mutation_pending.json"
# One file per run. This is the whole concurrency fix: nothing in here is ever overwritten or
# deleted by a DIFFERENT run, so two workflows sharing a working tree cannot corrupt each other's
# baseline (Race 2) or switch off each other's interlock (Race 1).
RUNS = REPO / "ops" / ".test_suite_mutation_runs"
# Where `restore()` parks a file's pre-revert content. `git checkout --` cannot be undone, and the
# detector cannot tell a test's write from a human's concurrent edit, so every revert is made
# recoverable rather than trusted to be correct. See `backup()`.
BACKUP = REPO / "ops" / ".test_suite_mutation_backup"

#: Set this to pin one run id for a whole workflow, which makes `--check` unambiguous even when
#: several workflows share the tree AND the parent shell. Recommended for agents.
RUN_ID_ENV = "CARNOT_MUTATION_RUN_ID"


class GitError(RuntimeError):
    """git could not answer. The caller must REFUSE, never assume a clean tree.

    Named rather than generic so callers can distinguish "git said the tree is clean" from "git
    did not say anything" -- collapsing those two is precisely the fail-open this replaced.
    """


def _git(*args: str) -> str:
    """Run git in the repo and return stdout. RAISES ``GitError`` if git fails.

    THIS USED TO FAIL OPEN, and the comment justifying it had the principle backwards. The old
    body was ``return r.stdout if r.returncode == 0 else ""``, reasoning that "a detector that
    raises on an unexpected git state is worse than one that reports nothing". But this detector
    does not report nothing on an empty result -- it reports a CLEAN TREE. An empty dirty-set
    flows into `mutations()`, which returns [], which prints "OK -- no tracked file was modified
    by the run" and exits 0. So a git failure of any kind (corrupt index, a concurrent `git gc`
    holding a lock, the tool run outside a work tree) produced a confident all-clear over a tree
    that might have just been rewritten.

    That matters more here than in an ordinary diagnostic because `--gate` is a pre-commit hook.
    The single moment its answer is load-bearing is the moment a wrong answer publishes the
    rewrite. So it now raises, and each command path turns the failure into a refusal with git's
    own stderr attached. The "must not break the suite" concern is real but belongs one layer up,
    and is already handled there: `tests/python/conftest.py` wraps both of its calls into this
    module in `try/except Exception` by deliberate design.
    """
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed (exit {r.returncode}): {r.stderr.strip()}")
    return r.stdout


def dirty_tracked() -> dict[str, str]:
    """Map tracked-path -> git status code for every tracked file that differs from the index.

    ``-uno`` drops untracked files deliberately. An experiment script writing a BRAND NEW
    artifact is normal, expected work and is not a rewrite of the record; the harm this detects
    is specifically an EXISTING, committed file being changed underneath you.

    RENAME AND COPY ENTRIES NEED EXPLICIT HANDLING, and an earlier draft got this wrong in a way
    worth recording. Under ``-z``, git emits a rename as TWO NUL-terminated records: ``R  <new>``
    followed by a bare ``<old>`` with no status prefix. Naively slicing every record as
    ``code=line[:2], path=line[3:]`` therefore turns the second record into a garbage entry --
    ``results/x.json`` becomes code ``re`` at path ``ults/x.json``. That is not a harmless
    cosmetic bug: the garbage path is reported to the operator as a modified file, and
    ``restore()`` would then ``git checkout`` a path that does not exist. So the loop consumes the
    source record explicitly when the status code is R or C.
    """
    out: dict[str, str] = {}
    records = _git("status", "--porcelain", "-uno", "-z").split("\0")
    i = 0
    while i < len(records):
        line = records[i]
        i += 1
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:]
        if code[0] in ("R", "C") or code[1] in ("R", "C"):
            i += 1  # swallow the paired source path; the DESTINATION is what changed
        out[path] = code
    return out


def _session_prefix() -> str:
    """The part of an auto run id shared by every command in one shell session.

    PPID, deliberately: the documented workflow is ``--snapshot`` / <do the thing> / ``--check``,
    which is two processes, so the id has to survive across them -- the tool's own PID cannot.
    The parent shell is what they have in common.

    PPID ALONE IS NOT A UNIQUE WORKFLOW KEY IN THIS ENVIRONMENT, and that is not a detail. Several
    agent workflows can be children of the same long-lived process, so two concurrent workflows can
    genuinely share a PPID (measured: both of this tree's live workflows report PPID 126919). That
    is why `_auto_run_id` appends a timestamp, and why `--check`'s auto-resolution REFUSES on more
    than one match instead of picking. A colliding session degrades to "you must say which one",
    never to "silently use the other workflow's baseline".
    """
    return f"ppid-{os.getppid()}"


def _auto_run_id() -> str:
    """A fresh id for one ``--snapshot``. Unique even against a colliding PPID."""
    return f"{_session_prefix()}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}"


def resolve_run_id(explicit: str | None = None) -> str:
    """Run id for WRITING: explicit flag, else the pinned env var, else a fresh auto id."""
    return explicit or os.environ.get(RUN_ID_ENV) or _auto_run_id()


def read_run_id(explicit: str | None = None) -> str | None:
    """Run id for READING, or None to auto-match this session's snapshots.

    Deliberately NOT `resolve_run_id`: that one invents a fresh auto id when nothing is pinned,
    which is right for writing a new baseline and useless for finding an existing one (the new id
    matches no file, so every unpinned `--check` would refuse). Caught by replaying Race 2 --
    `--check` under a pinned ``$CARNOT_MUTATION_RUN_ID`` reported OK because it never looked at
    the env var at all and fell through to the session glob.
    """
    return explicit or os.environ.get(RUN_ID_ENV) or None


def _snapshot_path(run_id: str) -> Path:
    return RUNS / f"{run_id}.snapshot.json"


def snapshot(run_id: str | None = None) -> dict[str, str]:
    """Record which tracked files are ALREADY dirty, under this run's own id.

    Returns the baseline. The caller normally uses the RETURN VALUE and never re-reads the file --
    `cmd_run` and the pytest hook both hold their baseline in memory for the life of the run, so
    they were never exposed to Race 2. The file exists so that the two-process
    ``--snapshot`` / ``--check`` workflow has somewhere to put it.
    """
    rid = resolve_run_id(run_id)
    baseline = dirty_tracked()
    path = _snapshot_path(rid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": rid,
                "taken_at": datetime.now(UTC).isoformat(),
                "already_dirty": baseline,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    return baseline


def _read_baseline(path: Path) -> dict[str, str] | None:
    try:
        return json.loads(path.read_text()).get("already_dirty", {})
    except (json.JSONDecodeError, OSError, AttributeError):
        return None


def find_snapshots(run_id: str | None = None) -> list[Path]:
    """Snapshot files this invocation may legitimately read, newest last.

    With an explicit id (flag or env) that is exactly one file. Without one, only THIS SESSION's
    snapshots are candidates -- never every snapshot in the directory, because picking an arbitrary
    other workflow's baseline is Race 2 with extra steps.
    """
    if not RUNS.exists():
        return []
    if run_id:
        p = _snapshot_path(run_id)
        return [p] if p.exists() else []
    pref = _session_prefix()
    return sorted(RUNS.glob(f"{pref}-*.snapshot.json"))


class AmbiguousSnapshot(RuntimeError):
    """More than one snapshot could be meant, so none is used. Carries the candidates."""

    def __init__(self, candidates: list[Path]) -> None:
        self.candidates = candidates
        super().__init__(f"{len(candidates)} candidate snapshots")


def _load_snapshot(run_id: str | None = None) -> dict[str, str] | None:
    """The baseline for this run, or None if there is not exactly one to be sure about.

    Raises `AmbiguousSnapshot` when several could be meant. Answering with a guess is what made
    Race 2 a false negative rather than merely noise, so ambiguity is surfaced, not resolved.
    """
    candidates = find_snapshots(run_id)
    if len(candidates) > 1:
        raise AmbiguousSnapshot(candidates)
    if candidates:
        return _read_baseline(candidates[0])
    # THE LEGACY SHARED SNAPSHOT IS DELIBERATELY NOT READ. An earlier draft of this fix fell back
    # to it "so an in-flight workflow that snapshotted under the old code can still complete",
    # which quietly preserved the entire hole: that file is the shared mutable cell, so reading it
    # is Race 2 -- a baseline written by some other workflow, at some other time, silently used as
    # if it were yours, which is how a real rewrite gets reported as OK. (The one on disk when
    # this was written listed 7 dirty files while the tree had 84.) The legacy PENDING marker IS
    # still honoured, and the asymmetry is the point: honouring a stale marker can only make the
    # gate refuse MORE, while honouring a stale baseline makes it refuse LESS.
    return None


def mutations(baseline: dict[str, str]) -> list[str]:
    """Tracked files that moved since the baseline.

    A file already dirty BEFORE the run is not attributed to the run -- that is in-flight work,
    and flagging it would make the detector useless on any real working tree. A file that was
    already dirty and then changed AGAIN is indistinguishable from in-flight work at this layer,
    so it is not reported either; that is a stated blind spot, not an oversight.
    """
    now = dirty_tracked()
    return sorted(p for p in now if p not in baseline)


def backup(paths: list[str]) -> list[str]:
    """Copy each path's CURRENT content aside before anything reverts it.

    WHY THIS EXISTS -- it has now bitten twice, both times on the same file.

    ``restore()`` is ``git checkout --``, which is UNRECOVERABLE for uncommitted content. The
    detector attributes "modified since the snapshot" to the test run, but it cannot actually tell
    WHO wrote a file: a human editing a tracked file WHILE a wrapped run is in flight looks exactly
    like a test rewriting it. Both incidents were the same shape -- an in-progress edit to
    ``docs/research-notes/test-suite-rewrites-the-record-survey-2026-07-29.md`` was written after
    the snapshot was taken, so the restore dutifully reverted it and the work was gone with no
    diff, no warning, and nothing on disk to recover from. (The first is recorded in that
    document's own section 3.6; the second happened while validating this script.)

    Authorship is not recoverable in principle, so the fix is not smarter attribution -- it is
    making the mistake survivable. Every file is copied to ``ops/.test_suite_mutation_backup/``
    (gitignored) before it is reverted, and the destination is printed. A wrong revert then costs a
    ``cp`` instead of the work.
    """
    saved: list[str] = []
    for p in paths:
        src = REPO / p
        if not src.exists():
            continue
        dst = BACKUP / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.write_bytes(src.read_bytes())
            saved.append(p)
        except OSError:
            # A backup failure must never be the reason a restore does not happen; the caller
            # reports how many were saved, so a shortfall is visible rather than silent.
            continue
    return saved


def restore(paths: list[str]) -> list[str]:
    """Put the named paths back to their committed content. Returns the ones that came back.

    Backs every path up first -- see ``backup()`` for why that is not optional.
    """
    if not paths:
        return []
    saved = backup(paths)
    print(f"  backed up {len(saved)}/{len(paths)} file(s) to {BACKUP.relative_to(REPO)}/")
    if len(saved) < len(paths):
        print(f"  WARNING: {len(paths) - len(saved)} file(s) could NOT be backed up before revert")
    _git("checkout", "--", *paths)
    still = dirty_tracked()
    return [p for p in paths if p not in still]


def _report(muts: list[str]) -> None:
    print(f"test-suite-mutation-check: {len(muts)} TRACKED FILE(S) MODIFIED BY THE RUN")
    print(
        "  These were clean before the run. A test that re-executes a real experiment script\n"
        "  overwrites that script's artifact in place -- the historical measurement is replaced\n"
        "  by whatever the script produces today. Committing this publishes the rewrite.\n"
    )
    by_dir: dict[str, list[str]] = {}
    for p in muts:
        by_dir.setdefault(p.split("/")[0], []).append(p)
    for d, paths in sorted(by_dir.items()):
        print(f"  {d}/  ({len(paths)})")
        for p in paths:
            print(f"    {p}")
    print(
        "\n  To undo:   git checkout -- <paths>      (or re-run this with --restore)\n"
        "  To inspect what a rewrite actually cost:\n"
        "             python3 scripts/determination_preservation_lint.py"
    )


def write_pending(muts: list[str], command: list[str]) -> Path:
    """Arm the pre-commit interlock: record which tracked files a run rewrote.

    Factored out of ``cmd_run`` so the pytest auto-arm hook (``arm_from_pytest``) writes the
    SAME marker in the SAME shape. Two code paths writing two slightly different markers is how
    a gate ends up passing on one of them.

    ONE FILE PER ARMING EVENT, not one shared file. The old single-marker layout lost information
    the moment two runs armed it -- the second run's `write_text` replaced the first run's file
    list wholesale, so the first run's rewrite stopped being named by anything and became
    unguarded as soon as the second one was resolved. A uuid name means no marker can be
    clobbered; `prune_stale_markers` retires them individually on the evidence of the tree.
    """
    path = RUNS / f"{uuid.uuid4().hex}.pending.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "detected_at": datetime.now(UTC).isoformat(),
                "run_id": resolve_run_id(),
                "command": command,
                "modified_tracked_files": muts,
                "why_this_blocks_commits": (
                    "These tracked files were rewritten by a test run, not by an edit. "
                    "Committing them publishes a silent rewrite of the research record. "
                    "Restore them, or delete this marker deliberately if the rewrite is "
                    "intended and explained in the commit message."
                ),
            },
            indent=1,
        )
        + "\n"
    )
    return path


def marker_paths() -> list[Path]:
    """Every pending marker that currently blocks a commit, legacy file included."""
    found = sorted(RUNS.glob("*.pending.json")) if RUNS.exists() else []
    if PENDING.exists():
        found.append(PENDING)
    return found


def prune_stale_markers() -> list[Path]:
    """Delete markers whose damage is demonstrably undone. Returns the ones still blocking.

    WHY STALENESS IS DERIVED FROM THE TREE RATHER THAN FROM OWNERSHIP.

    A marker has to be retired somehow or one bad run wedges every future commit. The old code
    retired it the simplest possible way -- any clean run called ``PENDING.unlink()`` -- and that
    is Race 1: workflow B's clean run switched off the interlock workflow A had just armed, with
    A's rewrite still on disk. Replayed directly; ``--gate`` printed OK afterwards.

    Ownership ("a run may only clear its own marker") was the obvious alternative and does not
    work here, because there is no reliable per-workflow key: concurrent agent workflows in this
    tree share a PPID, so an ownership key built from it would let B clear A's marker again, while
    a per-process key would never let a re-run clear its OWN earlier marker and would wedge
    commits instead.

    So retirement is keyed to the thing that actually matters and that anyone can verify: a marker
    is stale exactly when every file it named is clean again. That is a fact about the repository,
    not a claim about who owns what, so any run -- or the gate itself -- can act on it safely. It
    also makes the tool self-healing after a ``git checkout --`` restore, which the old design
    only achieved as a side effect of the next clean run.

    FAIL CLOSED. A marker that cannot be read, or that names no files at all, is NEVER pruned.
    Unreadable is not the same as resolved, and a marker whose file list is missing tells us
    nothing about whether the damage is still there -- in both cases the existence of the marker
    remains the signal.
    """
    still_dirty = dirty_tracked()
    blocking: list[Path] = []
    for p in marker_paths():
        try:
            listed = json.loads(p.read_text()).get("modified_tracked_files", [])
        except (json.JSONDecodeError, OSError, AttributeError):
            blocking.append(p)  # unreadable => cannot prove it is resolved => keep refusing
            continue
        if not isinstance(listed, list) or not listed:
            blocking.append(p)
            continue
        if any(f in still_dirty for f in listed):
            blocking.append(p)
            continue
        p.unlink(missing_ok=True)
    return blocking


def arm_from_pytest(baseline: dict[str, str], command: list[str]) -> list[str]:
    """Called by ``tests/python/conftest.py`` at session end. Returns the mutations found.

    WHY THIS ENTRY POINT EXISTS (2026-07-29 review finding, the interlock's worst gap).
    ------------------------------------------------------------------------------------
    ``--gate`` refuses a commit while ``ops/.test_suite_mutation_pending.json`` exists, and until
    now only ``--run`` ever wrote that marker. So the interlock protected exactly the invocation
    nobody uses, and was silent for the one that caused BOTH recorded incidents:

        pytest tests/python/test_arc_*.py tests/python/test_experiment_*.py

    A bare ``pytest`` leaves no marker, so ``--gate`` exits 0 on a tree the suite has just
    rewritten. Demonstrated against the real tree: after simulating that rewrite class,
    ``README.md`` (operator-curated, and one of the original 39) and
    ``openspec/papers/paper-v6/section-6-limitations.md`` were both modified while ``--gate``
    exited 0 and the determination lint printed OK.

    The fix has to live where pytest is, not where the wrapper is -- an interlock that depends on
    the user choosing the safe wrapper is not an interlock. So ``conftest.py`` takes the baseline
    at session start and calls this at session end, and the marker gets armed HOWEVER pytest was
    invoked.

    Deliberately NOT the alternative fix of wiring ``--check`` into pre-commit: ``--check`` reads
    a snapshot file that a bare pytest run never wrote, so with no baseline it cannot distinguish
    a test's rewrite from a human's in-flight edit, and would refuse every commit made on a dirty
    tree. The baseline is the whole mechanism; the hook is just where it is read.

    This NEVER restores anything. Auto-reverting a file out from under a concurrent human edit is
    the failure this tool has already had twice (see ``backup()``); arming a marker is safe,
    reverting is not.

    STATED LIMITATION: a run that is KILLED never arms. ``pytest_sessionfinish`` does not run if
    the process is SIGTERM'd or SIGKILL'd, so a suite that times out (or that the operator
    interrupts) can leave rewrites on disk with no marker. Observed directly while validating
    this: a 2-minute timeout killed a full ``test_arc_*.py`` run mid-flight and no marker was
    written. That run happened to have moved nothing, but the gap is real. ``--run`` does not
    share it (the wrapper checks after the child exits however it exited), so wrap anything
    long-running. This is a reason to keep ``--run`` alive, not a reason to distrust the hook:
    the hook's job is to cover the *bare pytest* case that had no coverage at all.
    """
    muts = mutations(baseline)
    if not muts:
        # Retire markers the tree shows are resolved -- NOT every marker. An unconditional unlink
        # here is Race 1: this run finishing clean says nothing about whether ANOTHER run's
        # rewrite is still on disk. See `prune_stale_markers`.
        prune_stale_markers()
        return []
    write_pending(muts, command)
    return muts


def cmd_run(argv: list[str], do_restore: bool) -> int:
    baseline = snapshot()
    print(f"test-suite-mutation-check: baseline taken ({len(baseline)} file(s) already dirty)")
    proc = subprocess.run(argv, cwd=REPO)
    muts = mutations(baseline)
    if not muts:
        prune_stale_markers()
        print("test-suite-mutation-check: OK -- no tracked file was modified by the run")
        return proc.returncode
    _report(muts)
    if do_restore:
        recovered = restore(muts)
        print(f"\n  restored {len(recovered)}/{len(muts)} file(s) to their committed content")
        left = [p for p in muts if p not in recovered]
        if not left:
            prune_stale_markers()
            return proc.returncode or 0
        muts = left
        print(f"  COULD NOT RESTORE: {left}")
    write_pending(muts, argv)
    return 1


def cmd_gate() -> int:
    """The pre-commit interlock. Exit 1 refuses the commit.

    Fails CLOSED in every direction: an unreadable marker refuses, and git being unable to answer
    refuses rather than assuming the tree is clean.
    """
    try:
        blocking = prune_stale_markers()
    except GitError as exc:
        print("test-suite-mutation-check: REFUSING THE COMMIT -- git could not be queried.")
        print(f"  {exc}")
        print(
            "  This gate cannot confirm whether a test run rewrote tracked files, and a guard\n"
            "  that cannot confirm must refuse: reporting OK here is how an unreadable tree gets\n"
            "  a clean bill of health. Fix the git error and re-run."
        )
        return 1
    if not blocking:
        print("test-suite-mutation-check: OK")
        return 0

    print("test-suite-mutation-check: REFUSING THE COMMIT.")
    print(
        f"  {len(blocking)} test run(s) modified tracked files that were not restored.\n"
        f"  Committing now would publish a silent rewrite of the research record.\n"
    )
    for path in blocking:
        try:
            d = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            d = {}
        cmd = " ".join(d.get("command", ["?"]))
        print(f"  {path.relative_to(REPO)}  ({cmd})")
        for p in d.get("modified_tracked_files", []):
            print(f"    {p}")
    names = "  ".join(str(p.relative_to(REPO)) for p in blocking)
    print(
        f"\n  Fix:  git checkout -- <paths>          (the marker then retires itself)\n"
        f"  Or, if the rewrite is intended: explain it in the commit message and delete the\n"
        f"  marker deliberately:  rm {names}"
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--snapshot", action="store_true", help="record which tracked files are already dirty"
    )
    ap.add_argument(
        "--check", action="store_true", help="report tracked files modified since --snapshot"
    )
    ap.add_argument(
        "--restore", action="store_true", help="with --run/--check: git checkout the modified files"
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="pre-commit interlock: refuse while a pending marker exists",
    )
    ap.add_argument(
        "--run", action="store_true", help="snapshot, run the command after --, then check"
    )
    ap.add_argument(
        "--run-id",
        default=None,
        help=(
            "name this run's baseline explicitly, so --check cannot pick up another workflow's. "
            f"Defaults to ${RUN_ID_ENV} if set, else an id derived from the parent shell."
        ),
    )
    ap.add_argument("command", nargs=argparse.REMAINDER, help="the command to run (after --)")
    a = ap.parse_args(argv)

    try:
        if a.gate:
            return cmd_gate()

        if a.run:
            cmd = [c for c in a.command if c != "--"]
            if not cmd:
                ap.error("--run needs a command: --run -- pytest tests/python")
            return cmd_run(cmd, a.restore)

        if a.snapshot:
            rid = resolve_run_id(a.run_id)
            b = snapshot(rid)
            print(
                f"test-suite-mutation-check: baseline taken ({len(b)} file(s) already dirty)\n"
                f"  run id: {rid}\n"
                f"  check it with: python3 scripts/test_suite_mutation_check.py "
                f"--check --run-id {rid}"
            )
            return 0

        if a.check:
            return cmd_check(a.run_id, a.restore)
    except GitError as exc:
        # Every command path refuses on an unanswerable git rather than reporting a clean tree.
        print(f"test-suite-mutation-check: REFUSING -- git could not be queried.\n  {exc}")
        return 1

    ap.print_help()
    return 0


def cmd_check(run_id: str | None, do_restore: bool) -> int:
    """Report tracked files modified since this run's baseline.

    ``--check`` USED TO EXIT 0 WHEN THERE WAS NO SNAPSHOT, printing "no snapshot -- run --snapshot
    first". The documented contract for this tool is "Exit 0 = the tracked tree is where you left
    it", so answering 0 when no baseline was ever taken reports a clean tree on the strength of
    having no idea -- the same fail-open as the old `_git`. It now refuses, which is also what
    makes refusing on AMBIGUITY coherent: without a baseline of its own, the honest answer is "ask
    me again with a run id", never "fine".

    (The reversed test that pinned the old behaviour argued a hard failure "would block every
    fresh clone". That reasoning applies to a pre-commit hook; `--check` is not one. `--gate` is
    the hook, it keys off markers rather than snapshots, and it still passes on a fresh clone.)
    """
    try:
        baseline = _load_snapshot(read_run_id(run_id))
    except AmbiguousSnapshot as exc:
        print("test-suite-mutation-check: REFUSING -- more than one baseline could be meant.")
        print(
            "  Several snapshots match this session, so any answer would be a guess, and guessing\n"
            "  here is the concurrency bug this replaced: reading another run's baseline reports\n"
            "  OK over a rewrite that run had already marked as in-flight. Pick one:\n"
        )
        for p in exc.candidates:
            stamp = ""
            try:
                stamp = json.loads(p.read_text()).get("taken_at", "")
            except (json.JSONDecodeError, OSError):
                pass
            print(f"    --run-id {p.name.removesuffix('.snapshot.json')}   (taken {stamp})")
        return 1

    if baseline is None:
        print("test-suite-mutation-check: REFUSING -- no baseline for this run.")
        print(
            "  Nothing to compare against, so 'nothing changed' would be a guess rather than a\n"
            "  measurement. Take one first:\n"
            "    python3 scripts/test_suite_mutation_check.py --snapshot\n"
            f"  or pin one for the whole workflow with ${RUN_ID_ENV}."
        )
        if SNAPSHOT.exists():
            print(
                f"\n  ({SNAPSHOT.relative_to(REPO)} exists but is NOT used: it is the old shared\n"
                "   baseline, which any other workflow may have overwritten. Using it is the\n"
                "   concurrency bug this replaced. Delete it once no old-code run needs it.)"
            )
        return 1

    muts = mutations(baseline)
    if not muts:
        print("test-suite-mutation-check: OK -- no tracked file was modified since the snapshot")
        return 0
    _report(muts)
    if do_restore:
        recovered = restore(muts)
        print(f"\n  restored {len(recovered)}/{len(muts)} file(s)")
        return 0 if len(recovered) == len(muts) else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
