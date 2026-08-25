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

    # Or do it by hand, around anything at all. --snapshot prints the run id and the exact
    # --check line to pair with it; pass --run-id (or export CARNOT_MUTATION_RUN_ID for the whole
    # workflow) so a concurrent workflow's baseline can never be read instead of yours.
    python3 scripts/test_suite_mutation_check.py --snapshot
    <do the thing>
    python3 scripts/test_suite_mutation_check.py --check --run-id <id>

    # Pre-commit interlock (see below); no arguments.
    python3 scripts/test_suite_mutation_check.py --gate

Exit 0 = the tracked tree is where you left it. Exit 1 = it moved, and the names are printed.

ATTRIBUTION: WHICH MUTATIONS THE RUN ACTUALLY CAUSED (2026-07-30, third incident)
---------------------------------------------------------------------------------
"Changed since the baseline" and "written by the run" are different claims, and this tool used to
report the first while advising on the second -- printing ``git checkout -- <paths>`` over every
mutation. That advice is unrecoverable, and it has now been aimed at authored work three times:
twice at the survey document mid-edit (its sections 3.6 and 7b.3), and on 2026-07-30 at six files
a CONCURRENT AGENT had just written, which ``--check`` reported as test-run damage.

The survey's conclusion after the second incident was that this could not be fixed:

    "the detector cannot tell a test's write from a human's concurrent edit, because at the file
     level they are identical events. Anything cleverer would be a heuristic guessing at
     authorship."

The first clause is correct and the conclusion does not follow. The events are identical at the
FILE level and distinct at the PROCESS level -- and the documented damage mechanism,
``runpy.run_path`` executing an experiment script, runs INSIDE the pytest interpreter, where
``sys.addaudithook`` can watch it. So the run records what it writes
(``scripts/_mutation_observer/sitecustomize.py`` for subprocesses, ``install_write_observer`` for
the pytest process itself) and attribution becomes a lookup in that log rather than a guess.

Mutations are now reported in two groups (see ``classify``):

  * **ATTRIBUTED**   -- the run was observed writing the path. Gets the restore advice; this is
    what ``--restore`` acts on, and the only thing it acts on.
  * **UNATTRIBUTED** -- changed in the same window, but the run was not seen writing it (or
    nothing was observed at all). Reported, never reverted, never advised for reverting.

The safety property runs in the conservative direction: an unobserved write is UNATTRIBUTED, not
"clean". Unattributed paths still arm the interlock and still block commits via ``--gate``, so the
broad guard keeps its full scope -- only the narrow, destructive act now requires evidence.

``backup()`` is unchanged and still runs before every revert. Attribution makes a wrong revert
rare; the backup makes it survivable. Both, not either.

WHAT ATTRIBUTION DOES NOT COVER (state it, do not assume it away):

  * A writer that is not a Python process -- a shell redirect, a compiled tool -- is never
    observed, so its writes land in UNATTRIBUTED. Conservative, but it means ``--restore`` cannot
    clean up after a non-Python test command.
  * A run KILLED before it can flush loses its buffered observations. Same direction: everything
    becomes UNATTRIBUTED rather than wrongly attributed.
  * Observation says the run WROTE the path. It does not say the write was harmful, and it never
    claims the content changed meaningfully -- that is
    ``determination_preservation_lint.py``'s question, and it remains the content-aware layer.

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
import contextlib
import hashlib
import importlib.util
import json
import os
import time
import subprocess
import sys
import uuid
from collections.abc import Callable
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

# --- WRITE OBSERVATION: how a mutation gets ATTRIBUTED to the run rather than guessed at ---------
# `scripts/_mutation_observer/sitecustomize.py` installs an audit hook that appends every path the
# interpreter opens for writing to $CARNOT_MUTATION_WRITE_LOG. Putting that directory on PYTHONPATH
# extends it to every Python subprocess the run spawns. See `classify()` for what the log buys.
OBSERVER_DIR = REPO / "scripts" / "_mutation_observer"
WRITE_LOG_ENV = "CARNOT_MUTATION_WRITE_LOG"


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


def _stash_hidden_paths(max_age_s: float = 15 * 60) -> set[str]:
    """Repo-relative paths hidden by pre-commit's stash of unstaged changes.

    THE HOLE THIS CLOSES (QA-layer SILENT_NON_FIRING finding, 2026-08-23,
    reported input: an unresolved unstaged rewrite of
    output/kanele_synth/post_synth.dcp). pre-commit stashes unstaged changes
    into a patch file under its cache before running hooks and restores them
    after. `prune_stale_markers` keys staleness to the WORKING TREE, so during
    the hook run — the exact moment the gate executes — an unstaged rewrite a
    marker names looks clean, the marker retires itself, the gate prints OK,
    and pre-commit then restores the rewrite to a tree with no interlock left.
    The gate disarmed itself on every commit.

    "Hidden by the stash" is distinguished from "not there" by consulting the
    NEWEST patch in pre-commit's cache, and only when it is fresh (default 15
    minutes — hook runs take seconds to a few minutes; patches from restored
    earlier runs linger in the cache for days and must not count). A path in
    that patch exists as an unstaged change even though the tree looks clean.

    Fail direction: toward KEEPING a marker (refusing the commit). The
    residual false-refusal window — a user reverts a marker path and commits
    within minutes of a run whose newest patch named it, with no new patch
    created — costs one refusal with remediation text, never a lost rewrite.
    This function only ever READS the patch; nothing here reverts or restores
    anything.
    """
    home = os.environ.get("PRE_COMMIT_HOME")
    if not home:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg) if xdg else Path.home() / ".cache"
        home = str(base / "pre-commit")
    try:
        patches = sorted(Path(home).glob("patch*"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return set()
    if not patches:
        return set()
    newest = patches[-1]
    try:
        if time.time() - newest.stat().st_mtime > max_age_s:
            return set()
        text = newest.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    hidden: set[str] = set()
    for line in text.splitlines():
        # Patch files are git diffs; the b-side of the header names the path.
        if line.startswith("diff --git a/"):
            b_side = line.split(" b/", 1)
            if len(b_side) == 2 and b_side[1]:
                hidden.add(b_side[1])
    return hidden


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
    # A new baseline opens a new observation window, so the previous window's write log must not
    # survive into it. The docs recommend agents PIN one run id for a whole workflow, and the log
    # is append-only -- so without this, run 20's `--check` would still see run 1's writes and
    # could attribute (and with --restore, REVERT) a file that a human has since edited by hand.
    reset_writes(rid)
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


def _writes_path(run_id: str) -> Path:
    return RUNS / f"{run_id}.writes.log"


def reset_writes(run_id: str) -> None:
    """Start a fresh observation window for ``run_id``.

    Called wherever a new baseline is taken. The log is append-only so that four xdist workers can
    contribute to one file without locking; the cost of that choice is that it would otherwise
    accumulate forever under a pinned ``$CARNOT_MUTATION_RUN_ID`` (which is the RECOMMENDED agent
    workflow). Clearing it at baseline time keeps "observed" meaning "observed in THIS window",
    which is the only reading under which attribution is sound.
    """
    with contextlib.suppress(OSError):
        _writes_path(run_id).unlink(missing_ok=True)


def install_write_observer(log_path: Path) -> Callable[[], None]:
    """Record this process's writes to ``log_path``. Returns an explicit flush.

    The in-process twin of ``scripts/_mutation_observer/sitecustomize.py`` -- see that file for the
    full rationale. This one covers the pytest process itself, which is where ``runpy.run_path``
    executes the real experiment scripts, and is therefore where every recorded incident happened.
    The sitecustomize covers Python subprocesses.

    CALLERS MUST FLUSH EXPLICITLY when they need to READ the log in the same process.
    ``pytest_sessionfinish`` runs long before interpreter shutdown, so a conftest that relied on
    the ``atexit`` hook alone would read an empty log and attribute nothing -- silently degrading
    every mutation to UNATTRIBUTED. The ``atexit`` registration stays as a safety net for
    processes that just exit; the returned callable is idempotent, so using both is safe.
    """
    import atexit

    seen: set[str] = set()
    # The flush below opens the log for appending, which the hook would then record -- the observer
    # observing itself. Harmless for attribution (the log is gitignored, so it can never turn up in
    # `mutations()`) but it is noise in a record whose whole value is being trustworthy.
    self_path = str(log_path)

    def _hook(event: str, args: tuple) -> None:
        if event == "open":
            path, mode, _flags = args
            if isinstance(path, str) and mode and any(c in mode for c in "wxa+"):
                if path != self_path:
                    seen.add(path)
        elif event in ("os.rename", "os.replace"):
            dest = args[1]
            if isinstance(dest, (str, bytes)):
                seen.add(dest.decode() if isinstance(dest, bytes) else dest)

    sys.addaudithook(_hook)

    def _flush() -> None:
        # Idempotent: drains `seen`, so an explicit call followed by the atexit one writes once.
        if not seen:
            return
        batch = sorted(seen)
        seen.clear()
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write("".join(f"{p}\n" for p in batch))
        except OSError:
            pass

    atexit.register(_flush)
    return _flush


def observer_env(run_id: str, base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment that makes a child process (and its Python children) record their writes."""
    env = dict(os.environ if base is None else base)
    env[WRITE_LOG_ENV] = str(_writes_path(run_id))
    existing = env.get("PYTHONPATH", "")
    # PREPEND, so our sitecustomize wins if anything else on the path defines one.
    env["PYTHONPATH"] = f"{OBSERVER_DIR}{os.pathsep}{existing}" if existing else str(OBSERVER_DIR)
    return env


def read_observed(run_id: str) -> set[str] | None:
    """Repo-relative paths the run was OBSERVED writing, or None if it was never observed.

    ``None`` and ``set()`` mean different things and callers must not conflate them: ``None`` is
    "no observation was recorded, so nothing can be attributed", while an empty set is "the run was
    watched and wrote nothing". The first must withhold the destructive advice; the second is a
    positive finding that every mutation came from somewhere else.
    """
    path = _writes_path(run_id)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    # Resolved HERE, not at import: the audit hook reports real paths, this repo is reachable
    # through a symlink (`~/github.com/Carnot-EBM/carnot-ebm`) so unresolved comparison would miss
    # every write that arrived by the other name, AND `REPO` is monkeypatched by the test fixture
    # -- a module-level constant would keep pointing at the real tree and silently attribute
    # nothing, which is the quiet-failure shape this whole module exists to prevent.
    repo_real = Path(os.path.realpath(REPO))
    out: set[str] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            real = Path(os.path.realpath(line))
            out.add(str(real.relative_to(repo_real)))
        except (ValueError, OSError):
            # Outside the repo (/tmp, site-packages, ...) or unresolvable. Not our concern: only
            # tracked files inside the repo can ever appear in `mutations()`.
            continue
    return out


def classify(muts: list[str], observed: set[str] | None) -> tuple[list[str], list[str]]:
    """Split mutated paths into (ATTRIBUTED to the run, UNATTRIBUTED).

    THE POINT OF THIS FUNCTION -- read this before changing it.

    The survey that produced this tool concluded attribution was impossible: *"the detector cannot
    tell a test's write from a human's concurrent edit, because at the file level they are
    identical events. Anything cleverer would be a heuristic guessing at authorship."* The first
    sentence is right and the conclusion does not follow. They are identical at the FILE level and
    distinct at the PROCESS level, and the run can watch itself -- so this is a lookup in a
    recorded log, not a heuristic about who owns what.

    That distinction is load-bearing because the tool's advice is DESTRUCTIVE. ``git checkout --``
    is unrecoverable for uncommitted content, and the tool has now aimed it at authored work three
    times (survey sections 3.6 and 7b.3; then the 2026-07-30 ``--check`` that flagged six files a
    concurrent agent had just written). Every one of those files was a document no test writes --
    which is exactly what an observation log makes visible.

    Direction of failure is deliberate. A write that was NOT observed is UNATTRIBUTED, never
    "clean": an unobserved writer (a non-Python subprocess, a killed run that never flushed) must
    not be silently reverted, and must not be silently dismissed either. So:

      * UNATTRIBUTED still arms the interlock, so the record stays protected by ``--gate``.
      * UNATTRIBUTED is never auto-restored and never gets the ``git checkout --`` line.

    The broad, conservative guard keeps its scope; only the narrow, destructive act now requires
    evidence.
    """
    if observed is None:
        return [], list(muts)
    attributed = [p for p in muts if p in observed]
    return attributed, [p for p in muts if p not in observed]


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


def _print_group(paths: list[str]) -> None:
    by_dir: dict[str, list[str]] = {}
    for p in paths:
        by_dir.setdefault(p.split("/")[0], []).append(p)
    for d, group in sorted(by_dir.items()):
        print(f"    {d}/  ({len(group)})")
        for p in group:
            print(f"      {p}")


def _report(muts: list[str], observed: set[str] | None = None) -> None:
    """Report the mutations, SPLIT by whether the run was observed writing them.

    The split is the whole point: the two groups get different advice because only one of them has
    evidence behind it. Lumping them together under "MODIFIED BY THE RUN" plus a bare
    ``git checkout --`` line is what aimed an unrecoverable command at authored work three times.
    """
    attributed, unattributed = classify(muts, observed)
    print(f"test-suite-mutation-check: {len(muts)} TRACKED FILE(S) MODIFIED SINCE THE BASELINE")

    if attributed:
        print(
            f"\n  ATTRIBUTED TO THE RUN ({len(attributed)}) -- the run was OBSERVED writing these.\n"
            "  A test that re-executes a real experiment script overwrites that script's artifact\n"
            "  in place: the historical measurement is replaced by whatever the script produces\n"
            "  today. Committing this publishes the rewrite.\n"
        )
        _print_group(attributed)
        print(
            "\n    To undo:  git checkout -- <paths>     (or re-run this with --restore)\n"
            "    To see what a rewrite cost:\n"
            "              python3 scripts/determination_preservation_lint.py"
        )

    if unattributed:
        if observed is None:
            why = (
                "  NOT ATTRIBUTED -- this run recorded no write observations, so nothing here can\n"
                "  be pinned on it. That happens when the run was not started through --run, when\n"
                "  it was killed before it could flush, or when the writer was not a Python\n"
                "  process. Treat every path below as possibly someone else's in-flight work.\n"
            )
        else:
            why = (
                "  NOT ATTRIBUTED -- the run was watched and did NOT write these. They changed\n"
                "  during the same window for some other reason: most often a concurrent agent or\n"
                "  editor working in the same tree.\n"
            )
        print(f"\n  UNATTRIBUTED ({len(unattributed)})\n{why}")
        _print_group(unattributed)
        print(
            "\n    NOT reverted, and --restore will not touch these. Reverting an unattributed\n"
            "    path is how this tool destroyed authored work three times; git checkout -- is\n"
            "    unrecoverable for uncommitted content. Decide per file, by hand.\n"
            "    They still block commits via --gate until resolved, deliberately."
        )


def write_pending(muts: list[str], command: list[str], observed: set[str] | None = None) -> Path:
    """Arm the pre-commit interlock: record which tracked files a run rewrote.

    ``observed`` (when the run recorded its writes) splits the list into what the run was seen
    writing and what merely changed alongside it, so ``--gate`` can aim its restore suggestion at
    the first group only. Omitted -> everything is recorded as unattributed, which is the
    conservative direction: the gate still refuses, it just does not tell anyone to revert.

    Factored out of ``cmd_run`` so the pytest auto-arm hook (``arm_from_pytest``) writes the
    SAME marker in the SAME shape. Two code paths writing two slightly different markers is how
    a gate ends up passing on one of them.

    ONE FILE PER ARMING EVENT, not one shared file. The old single-marker layout lost information
    the moment two runs armed it -- the second run's `write_text` replaced the first run's file
    list wholesale, so the first run's rewrite stopped being named by anything and became
    unguarded as soon as the second one was resolved. A uuid name means no marker can be
    clobbered; `prune_stale_markers` retires them individually on the evidence of the tree.
    """
    attributed, unattributed = classify(muts, observed)
    path = RUNS / f"{uuid.uuid4().hex}.pending.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "detected_at": datetime.now(UTC).isoformat(),
                "run_id": resolve_run_id(),
                "command": command,
                "modified_tracked_files": muts,
                "attributed_to_run": attributed,
                "unattributed": unattributed,
                "write_observation": "recorded" if observed is not None else "none",
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
    # A clean-looking tree is NOT proof the damage is gone while pre-commit
    # has unstaged changes stashed: the stash hides exactly the unstaged
    # rewrite the marker names, and the restore happens after the hook. A
    # stash-hidden path counts as dirty for retirement purposes
    # (2026-08-23 SILENT_NON_FIRING fix; see _stash_hidden_paths).
    hidden = _stash_hidden_paths()
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
        if any(f in still_dirty or f in hidden for f in listed):
            blocking.append(p)
            continue
        p.unlink(missing_ok=True)
    return blocking


RETENTION_DAYS = 7


def prune_old_debris(days: int = RETENTION_DAYS) -> tuple[int, int]:
    """Delete spent snapshots and write-logs older than `days`. Returns (files, bytes).

    WHY THIS EXISTS. `prune_stale_markers` above retires MARKERS -- the interlock files. Nothing
    ever removed the two debug files each run leaves behind, so the directory grew without limit.
    Measured 2026-08-13: 3,501 files and 464 MB on the operator's disk, 3,441 of them `.writes.log`
    from runs that finished weeks ago. A guard that quietly consumes half a gigabyte of somebody's
    disk is a guard they will eventually turn off.

    WHAT IS SAFE TO DELETE, and why this cannot weaken the interlock. A `.pending.json` marker is
    the ONLY file the gate consults, and this function never touches one at any age -- an old
    marker means an old unresolved rewrite, which is more serious than a recent one, not less. A
    `.snapshot.json` is a baseline that only a matching `--check` reads, and a `--check` a week
    after its `--snapshot` is meaningless anyway: the tree has moved on and the diff would be
    mostly other people's work. A `.writes.log` is never read by any code path in this file; it is
    there for a human reading a fresh incident.

    FAIL OPEN, deliberately, and this is the opposite choice from `prune_stale_markers`. That
    function fails CLOSED because a marker it cannot read might be hiding real damage. This one
    fails OPEN because it only removes debris: if a file cannot be stat'ed or unlinked, skipping
    it costs some disk and nothing else. Never let a cleanup routine fail a gate.
    """
    cutoff = time.time() - days * 86400
    n = freed = 0
    for p in list(RUNS.glob("*.snapshot.json")) + list(RUNS.glob("*.writes.log")):
        try:
            st = p.stat()
            if st.st_mtime >= cutoff:
                continue
            size = st.st_size
            p.unlink()
        except OSError:
            continue  # fail open: debris cleanup must never break a gate
        n += 1
        freed += size
    return n, freed


def arm_from_pytest(
    baseline: dict[str, str], command: list[str], run_id: str | None = None
) -> list[str]:
    """Called by ``tests/python/conftest.py`` at session end. Returns the mutations found.

    ``run_id`` names this run's write-observation log, so the marker records WHICH mutations the
    run was actually seen writing. Omitted -> no attribution is recorded, and every mutation is
    marked unattributed, which still blocks the commit but withholds the restore advice.

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
    write_pending(muts, command, read_observed(run_id) if run_id else None)
    return muts


def attributed_from_pytest(baseline: dict[str, str], run_id: str | None) -> list[str]:
    """The subset of this pytest run's mutations the run was OBSERVED writing.

    ``conftest.py`` uses this to warn about attributed and unattributed rewrites differently. The
    hook still ARMS on everything (``arm_from_pytest``) -- the record stays protected either way;
    it is only the ``git checkout --`` suggestion that now requires evidence, because that
    suggestion is the part that has destroyed work.
    """
    if not run_id:
        return []
    attributed, _ = classify(mutations(baseline), read_observed(run_id))
    return attributed


def _restore_attributed(muts: list[str], observed: set[str] | None) -> list[str]:
    """Revert only what the run was observed writing. Returns what still needs a decision.

    ``--restore`` used to revert every path that changed since the baseline, which is how it ate
    an in-progress document twice. It now reverts only ATTRIBUTED paths; unattributed ones are
    reported and left alone, because for those the tool has no evidence and the action is
    unrecoverable.
    """
    attributed, unattributed = classify(muts, observed)
    if not attributed:
        print(
            "\n  --restore reverted NOTHING: no mutation is attributable to this run."
            + (
                "\n  (No write observations were recorded, so there is nothing to attribute with.)"
                if observed is None
                else ""
            )
        )
        return list(muts)
    recovered = restore(attributed)
    print(f"\n  restored {len(recovered)}/{len(attributed)} attributed file(s)")
    left = [p for p in attributed if p not in recovered]
    if left:
        print(f"  COULD NOT RESTORE: {left}")
    if unattributed:
        print(f"  left {len(unattributed)} unattributed file(s) alone -- decide those by hand")
    return left + unattributed


def cmd_run(argv: list[str], do_restore: bool) -> int:
    run_id = resolve_run_id()
    baseline = snapshot(run_id)
    print(f"test-suite-mutation-check: baseline taken ({len(baseline)} file(s) already dirty)")
    # The child records what it writes, so a mutation can be attributed rather than guessed at.
    proc = subprocess.run(argv, cwd=REPO, env=observer_env(run_id))
    observed = read_observed(run_id)
    muts = mutations(baseline)
    if not muts:
        prune_stale_markers()
        print("test-suite-mutation-check: OK -- no tracked file was modified by the run")
        return proc.returncode
    _report(muts, observed)
    if do_restore:
        muts = _restore_attributed(muts, observed)
        if not muts:
            prune_stale_markers()
            return proc.returncode or 0
    write_pending(muts, argv, observed)
    return 1


# --- MUTATION-PROOF SESSIONS: the OTHER thing this file is used for ------------------------------
# Everything above answers "did a test run rewrite tracked files". A mutation PROOF is a different
# activity that shares the word: an agent deletes a guard's pattern by hand, runs the suite,
# expects RED, then restores. CLAUDE.md requires that proof for every new guard. Nothing enforced
# it, and on 2026-08-25 two proofs ran against this working tree at once. The tree carried
#
#     python/carnot/agentic/arc_executable_world_model.py:6466: pass  # MUTATED M6
#
# on the LIVE ARC scored path. `pass  # MUTATED` is valid Python that clears every hook, and the
# conductor commits `git add -A --no-verify` on its own schedule, so a checkpoint inside a mutation
# window publishes it silently. Readings were contaminated both ways -- a RED credited to your own
# deleted pattern can be the other harness's mutation -- and an 11-mutation proof set was voided.
#
# These three checks are additive and touch none of the paths above. The module docstring rejects a
# lock for --snapshot/<work>/--check, and that reasoning stands: it spans two processes with
# arbitrary work between them, and serialising a diagnostic is worse than the race. A proof session
# is the opposite shape -- deliberately exclusive, explicitly begun and ended -- so it gets a lock.
def _proof_lock_path() -> Path:
    """One lock per REPOSITORY, not per checkout.

    `REPO` is this file's own checkout, and this repo has ~10 live worktrees
    sharing one venv whose editable .pth pins ONE checkout. A per-checkout lock
    is therefore void in exactly the workflow the project uses for proofs: agent
    A in the main tree and agent B in a worktree would each hold their own lock
    and never see each other. `--git-common-dir` is shared by every worktree, so
    a lock under it is shared too. Falls back to this checkout when git cannot
    answer -- degraded, but never worse than the per-checkout version.
    """
    try:
        common = _git("rev-parse", "--git-common-dir").strip()
    except (GitError, OSError):
        return REPO / "ops" / ".test_suite_mutation_proof.lock"
    root = Path(common)
    if not root.is_absolute():
        root = (REPO / root).resolve()
    return root / "carnot_mutation_proof.lock"


PROOF_LOCK = _proof_lock_path()

#: The marker convention agents write into a mutated line. Scanned for at session end.
MUTATION_MARKER = "MUTATED"

#: How long an unclosed session may sit before another may reclaim it. Age is the ONLY real
#: protection here (see _proof_lock_is_stale), so this must exceed the longest honest proof, not
#: the typical one: an 11-mutation set that re-runs the suite per mutation is hours, and the
#: 2 h first guess would have reclaimed a live session mid-set. Twelve hours is longer than any
#: proof and still shorter than forever, because a lock nobody can clear is its own outage.
PROOF_LOCK_STALE_S = 12 * 60 * 60

#: Only files that RUN can carry a damaging mutation. Prose that quotes the marker -- this
#: project's spec, changelog and research notes all describe the incident verbatim -- is not a
#: mutation, and scanning it bricked the tool against its own documentation on first use.
_MARKER_SCAN_SUFFIXES = frozenset({".py", ".pyi", ".rs", ".sh", ".bash", ".js", ".ts"})

#: Set after confirming by import which file the tests actually load, to proceed with a proof on
#: a path-loaded file in a checkout the installed package does not point at. Forcing the agent to
#: type it IS the control -- the same shape as ACKNOWLEDGED_NON_QA_LAYER.
PATH_LOADED_ACK_ENV = "CARNOT_MUTATION_PATH_LOADED_VERIFIED"

#: The two files that must contain the literal marker to DEFINE and TEST the scan. Everything else
#: is scanned. An allow-list, not an extension filter, so the scan stays as wide as its concept.
_MARKER_SCAN_EXEMPT = frozenset(
    {
        "scripts/test_suite_mutation_check.py",
        "tests/python/test_test_suite_mutation_check.py",
    }
)

#: Files larger than this are reported as UNSCANNED rather than skipped silently. A marker hides
#: just as well in a file nobody looked at as in one nobody checked.
_MARKER_SCAN_MAX_BYTES = 2_000_000


def _sha256(path: Path) -> str | None:
    """Content hash, or None when it cannot be read. None never compares
    equal to a recorded hash, so an unreadable target fails the restore
    check rather than passing it."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _pid_alive(pid: int) -> bool | None:
    """True / False / None when it cannot be known.

    None is not "probably dead". The caller REFUSES on None, because a lock
    whose holder cannot be checked is a lock that must be honoured.
    """
    try:
        return Path(f"/proc/{pid}").exists()
    except OSError:
        return None


def _module_dotted_name(target: Path) -> str | None:
    """The dotted name `target` would be imported as, or None if it has none.

    Only files under `python/` are importable as `carnot.*`. A script under
    `scripts/` is loaded by explicit path, so no resolution ambiguity exists
    for the inert check to find.
    """
    try:
        rel = target.resolve().relative_to((REPO / "python").resolve())
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else None


def _installed_package_root() -> Path | None:
    """The directory the interpreter will import `carnot` from.

    Replicates the finder's own search over `sys.path` rather than calling
    `importlib.util.find_spec`, which IMPORTS every parent package -- measured
    at 90 carnot modules, several of them experiment modules, inside a test.
    That is the write vector the Test-Run Record Integrity Discipline names, so
    the check must not use it.
    """
    for entry in sys.path:
        if not entry:
            continue
        if (Path(entry) / "carnot" / "__init__.py").exists():
            return Path(entry).resolve()
    return None


def check_target_is_live(target: Path) -> tuple[bool, str]:
    """(ok, explanation) -- would mutating this file change what the tests import?

    THE INERT-RUN TRAP. A proof run inside a `git worktree` copy is silently
    inert here: `.venv/.../__editable__.carnot_ebm.pth` holds the ABSOLUTE path
    of the MAIN checkout, so `carnot.*` resolves there from any worktree. The
    mutated file is never imported, every mutation reads GREEN, and the proof
    proves nothing while looking clean.

    Fails CLOSED: an unresolvable module refuses. A guard that passes while
    measuring nothing is the failure this whole file exists for.
    """
    target = target.resolve()
    if not target.exists():
        return False, f"target does not exist: {target}"
    root = _installed_package_root()
    if root is None:
        return False, "cannot tell which checkout `carnot` resolves to; sys.path has no carnot"
    dotted = _module_dotted_name(target)
    if dotted is not None:
        loaded = root.joinpath(*dotted.split(".")).with_suffix(".py")
        if not loaded.exists():
            loaded = root.joinpath(*dotted.split("."), "__init__.py")
        if loaded.resolve() != target:
            return False, (
                f"INERT RUN -- the interpreter imports {dotted} from\n"
                f"    {loaded}\n"
                f"  but you are about to mutate\n"
                f"    {target}\n"
                "  Every mutation would read GREEN while measuring nothing. This is the\n"
                "  worktree trap: the editable install pins an absolute path to the main\n"
                "  checkout, so a worktree copy is never the file under test. Run the proof\n"
                "  in the checkout the install points at, or repoint it."
            )
        return True, f"{dotted} resolves to the file being mutated"

    # Outside python/. Resolution depends on WHO loads the file, and that cannot be known from
    # here: pytest collecting in this rootdir reaches the local copy, but an installed carnot
    # module deriving its own repo root from __file__ adds the INSTALLED checkout's scripts/ to
    # sys.path and reaches that copy instead. WARN, do not refuse -- a hard refusal fires on
    # every legitimate proof of a scripts/ or tests/ file outside the installed checkout, and a
    # check that cries wolf gets bypassed, which is worse than the gap it closes.
    if not _is_under(target, root.parent):
        return True, (
            f"{target} is loaded by explicit path, so its route cannot be resolved here.\n"
            f"  CAUTION: `carnot` resolves to a DIFFERENT checkout ({root.parent}). A loader\n"
            "  that derives its own repo root will reach that copy, not this one. Confirm by\n"
            "  import which file your tests load before trusting a single reading."
        )
    return True, f"{target} is loaded by explicit path from the checkout `carnot` resolves to"


def _scan_paths_for_markers(paths: list[Path]) -> tuple[list[str], list[str]]:
    """(hits, unscanned). A hit is 'path:line: text'."""
    hits: list[str] = []
    unscanned: list[str] = []
    for path in paths:
        try:
            rel = str(path.resolve().relative_to(REPO))
        except ValueError:
            rel = str(path)
        if rel in _MARKER_SCAN_EXEMPT or not path.is_file():
            continue
        if path.suffix not in _MARKER_SCAN_SUFFIXES:
            continue
        try:
            if path.stat().st_size > _MARKER_SCAN_MAX_BYTES:
                unscanned.append(f"{rel} (too large)")
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            unscanned.append(f"{rel} ({type(exc).__name__})")
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if MUTATION_MARKER in line:
                hits.append(f"{rel}:{number}: {line.strip()[:120]}")
    return hits, unscanned


def _status_paths() -> list[Path]:
    """Every path git currently reports as changed or untracked.

    `-uall` lists files INSIDE a new directory. Plain `--porcelain` collapses a
    wholly-untracked directory to one `dir/` entry, which is not a file, so the
    whole subtree was skipped -- and a scratch copy of the file under mutation
    lives in exactly such a directory. `-z` avoids the C-quoting that mangles a
    non-ASCII path into one that does not exist, which the scan then drops.
    """
    out = _git("status", "--porcelain", "-uall", "-z")
    paths: list[Path] = []
    for record in out.split("\0"):
        if len(record) < 4:
            continue
        name = record[3:]
        if name:
            paths.append(REPO / name)
    return paths


def _paths_changed_since(commit: str | None) -> list[Path]:
    """Files changed since the commit HEAD pointed at when the session opened.

    THE ORIGIN MECHANISM, and `git status` cannot see it: the conductor commits
    on its own schedule, so a marker swept into a mid-session commit leaves the
    working tree CLEAN. Closing on `git status` alone would report no marker
    while the marker sits in HEAD -- published, which is worse than uncommitted.
    """
    if not commit:
        return []
    try:
        out = _git("diff", "--name-only", "-z", f"{commit}..HEAD")
    except GitError:
        # Fail closed at the caller: an unanswerable git is reported, never
        # silently treated as "nothing changed".
        raise
    return [REPO / n for n in out.split("\0") if n]


def surviving_markers(
    target: Path | None, since_commit: str | None = None
) -> tuple[list[str], list[str]]:
    """Markers left anywhere the session could have touched.

    Blast radius: the working tree's changed and untracked files, everything
    committed since the session opened, plus the declared target -- checked even
    when clean, because clean is what a restored file looks like.
    """
    candidates: list[Path] = []
    if target is not None:
        candidates.append(target)
    candidates.extend(_status_paths())
    candidates.extend(_paths_changed_since(since_commit))
    unique = {p.resolve(): p for p in candidates if p.exists()}
    return _scan_paths_for_markers(list(unique.values()))


def _read_proof_lock() -> dict | None:
    try:
        return json.loads(PROOF_LOCK.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        # Unreadable is NOT unheld. Signalled with a sentinel so the caller
        # refuses rather than reclaiming a lock it cannot read.
        return {"__unreadable__": True}


def _proof_lock_is_stale(held: dict) -> tuple[bool, str]:
    """(reclaimable, why) for a held lock.

    LIVENESS ALONE CANNOT DECIDE THIS, and assuming it could is a defect this
    function exists to correct. A proof session spans two CLI calls with the
    agent's hand-editing in between, so the process that wrote the lock has
    ALWAYS exited by the time anyone reads it. A dead-holder-means-reclaim rule
    therefore reclaims every time and locks nothing -- measured, not argued:
    the first build of this printed "mutation-proof session OPEN" for a second
    caller while the first session was still notionally open.

    So the rule is AGE, with liveness kept only as an extra refusal: a holder
    that IS still running refuses regardless of age. Fail direction is toward
    REFUSING -- an unreadable timestamp is not an expired one.
    """
    if held.get("host") == os.uname().nodename:
        alive = _pid_alive(int(held.get("pid", -1)))
        if alive:
            return False, f"pid {held.get('pid')} is still running"
        if alive is None:
            # Honours _pid_alive's stated contract. Unknowable liveness is a
            # lock that must be honoured, not one that falls through to age.
            return False, f"pid {held.get('pid')} liveness could not be checked"
    started = held.get("started_at")
    if not started:
        return False, "no start time recorded, so age cannot be checked"
    try:
        age = (datetime.now(UTC) - datetime.fromisoformat(started)).total_seconds()
    except (TypeError, ValueError):
        return False, f"unreadable start time {started!r}"
    if age < PROOF_LOCK_STALE_S:
        return False, f"open for {age / 60:.0f} min"
    return True, f"abandoned {age / 3600:.1f} h ago"


def cmd_mutation_force_unlock() -> int:
    """Deliberate override. Explicit so a wedged lock never needs `rm`, and
    auditable so nobody clears one by reflex."""
    held = _read_proof_lock()
    if held is None:
        print("test-suite-mutation-check: no mutation-proof lock to clear.")
        return 0
    PROOF_LOCK.unlink(missing_ok=True)
    print(f"test-suite-mutation-check: mutation-proof lock CLEARED by hand.\n  was: {held}")
    print("  Any mutation it left is still in the tree. Read `git status` now.")
    return 0


def cmd_mutation_begin(target_arg: str | None, run_id: str | None) -> int:
    """Open an exclusive mutation-proof session. Fails CLOSED throughout."""
    if not target_arg:
        print("test-suite-mutation-check: REFUSING -- --mutation-begin needs --mutation-target.")
        print("  The target is what the inert-run check resolves; without it nothing is proven.")
        return 1
    target = Path(target_arg)
    if not target.is_absolute():
        target = REPO / target

    held = _read_proof_lock()
    if held is not None:
        if held.get("__unreadable__"):
            print(f"test-suite-mutation-check: REFUSING -- {PROOF_LOCK.name} is unreadable.")
            print("  Unreadable is not unheld. Inspect it by hand and delete it deliberately.")
            return 1
        reclaim, why_stale = _proof_lock_is_stale(held)
        if not reclaim:
            print("test-suite-mutation-check: REFUSING -- a mutation proof is already open.")
            print(
                f"  held by pid {held.get('pid')}  run id {held.get('run_id')}\n"
                f"  target      {held.get('target')}\n"
                f"  started     {held.get('started_at')}  ({why_stale})\n"
                "  Two concurrent proofs contaminate each other's readings in BOTH directions:\n"
                "  a RED you credit to your own deleted pattern can be the other run's mutation.\n"
                "  Close it with --mutation-end, or override with --mutation-force-unlock."
            )
            return 1
        # Loud, never silent: the abandoned run's mutations may still be on disk.
        print(
            f"test-suite-mutation-check: reclaiming an abandoned lock ({why_stale})\n"
            f"  pid {held.get('pid')}, run id {held.get('run_id')}, target {held.get('target')}.\n"
            "  Its mutations may still be in the tree -- read `git status` before trusting\n"
            "  anything this session measures."
        )
        PROOF_LOCK.unlink(missing_ok=True)

    ok, why = check_target_is_live(target)
    if not ok:
        print(f"test-suite-mutation-check: REFUSING -- {why}")
        return 1

    leftovers, unscanned = surviving_markers(target)
    if leftovers:
        print("test-suite-mutation-check: REFUSING -- a MUTATION marker is already in the tree.")
        for hit in leftovers[:20]:
            print(f"    {hit}")
        print("  Starting here would attribute someone else's mutation to your proof.")
        return 1
    if unscanned:
        # An unlooked-at file is not a clean file. The constant's own comment says a marker
        # hides just as well in one, so the list is a refusal here, not an advisory.
        print("test-suite-mutation-check: REFUSING -- file(s) could not be scanned for markers.")
        for name in unscanned[:20]:
            print(f"    {name}")
        return 1

    rid = resolve_run_id(run_id)
    try:
        head = _git("rev-parse", "HEAD").strip()
    except GitError:
        head = ""  # reported at close; see _paths_changed_since
    payload = {
        "run_id": rid,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "host": os.uname().nodename,
        "target": str(target),
        "target_sha": _sha256(target),
        "head_at_begin": head,
        "started_at": datetime.now(UTC).isoformat(),
    }
    PROOF_LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        # O_EXCL: the acquire is atomic, so two simultaneous --mutation-begin calls cannot both win.
        fd = os.open(PROOF_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        print("test-suite-mutation-check: REFUSING -- another session took the lock just now.")
        return 1
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("test-suite-mutation-check: mutation-proof session OPEN")
    print(f"  run id  {rid}\n  target  {target}\n  live    {why}")
    print(f"  end it with: python3 scripts/test_suite_mutation_check.py --mutation-end")
    return 0


def cmd_mutation_end(run_id: str | None) -> int:
    """Close the session, verifying the tree carries nothing the proof left."""
    held = _read_proof_lock()
    if held is None:
        print("test-suite-mutation-check: REFUSING -- no mutation-proof session is open.")
        print(
            "  Nothing to verify: a session that was never begun recorded no target and no\n"
            "  pre-mutation hash, so 'restored' would be a guess."
        )
        return 1
    if held.get("__unreadable__"):
        print(f"test-suite-mutation-check: REFUSING -- {PROOF_LOCK.name} is unreadable.")
        return 1
    wanted = read_run_id(run_id)
    if not wanted:
        # Fail closed, the same shape as --check refusing without a baseline. An unpinned
        # --mutation-end used to skip the ownership test entirely, so any second agent in a
        # fresh shell closed whoever's session was open -- and if it landed between two
        # mutations the tree looked clean, so it succeeded. That IS the incident, rebuilt.
        print("test-suite-mutation-check: REFUSING -- no run id, so ownership cannot be checked.")
        print(
            "  Closing an unidentified session would let one agent end another's proof.\n"
            f"  Pass --run-id <id>, or export ${RUN_ID_ENV} for the whole workflow.\n"
            f"  The open session's id is: {held.get('run_id')}"
        )
        return 1
    if held.get("run_id") != wanted:
        print("test-suite-mutation-check: REFUSING -- that lock belongs to another run.")
        print(f"  held by run id {held.get('run_id')}, you asked for {wanted}.")
        return 1

    target = Path(held["target"]) if held.get("target") else None
    problems: list[str] = []

    hits, unscanned = surviving_markers(target, held.get("head_at_begin"))
    if hits:
        problems.append(f"{len(hits)} surviving {MUTATION_MARKER} marker(s)")
    if unscanned:
        problems.append(f"{len(unscanned)} file(s) that could not be scanned")

    if target is not None and held.get("target_sha"):
        now = _sha256(target)
        if now != held["target_sha"]:
            problems.append("target is NOT byte-identical to its pre-mutation content")

    if problems:
        print("test-suite-mutation-check: SESSION NOT CLEAN -- " + "; ".join(problems))
        for hit in hits[:40]:
            print(f"    {hit}")
        if target is not None and held.get("target_sha") and _sha256(target) != held["target_sha"]:
            print(f"    {target} differs from its pre-mutation bytes")
        if unscanned:
            print(f"  {len(unscanned)} file(s) not scanned: {', '.join(unscanned[:5])}")
        print(
            "\n  The lock is KEPT so this cannot be left behind. `pass  # "
            f"{MUTATION_MARKER}` is valid\n"
            "  Python that clears every hook, and the conductor commits on its own schedule.\n"
            "  Restore the file, then re-run --mutation-end."
        )
        return 1

    PROOF_LOCK.unlink(missing_ok=True)
    print("test-suite-mutation-check: mutation-proof session CLOSED -- tree carries no marker")
    return 0


def cmd_gate() -> int:
    """The pre-commit interlock. Exit 1 refuses the commit.

    Fails CLOSED in every direction: an unreadable marker refuses, and git being unable to answer
    refuses rather than assuming the tree is clean.
    """
    # Debris cleanup rides on the gate because the gate runs on every commit, so the directory
    # self-limits instead of needing anyone to remember. It touches no marker and cannot fail
    # this function -- see prune_old_debris for why it fails open where prune_stale_markers
    # fails closed.
    try:
        n, freed = prune_old_debris()
        if n:
            print(f"test-suite-mutation-check: pruned {n} spent file(s), {freed / 1048576:.0f} MB")
    except Exception:  # noqa: BLE001
        pass
    # A commit taken mid-proof is how a mutated line reaches the record. The hook cannot stop a
    # run that skips hooks, but it can stop the hook-respecting case, which is reachable.
    if PROOF_LOCK.exists():
        held = _read_proof_lock() or {}
        print("test-suite-mutation-check: REFUSING THE COMMIT -- a mutation proof is open.")
        print(
            f"  run id {held.get('run_id')}, target {held.get('target')}.\n"
            "  Committing now can publish a mutated line. Close it with --mutation-end."
        )
        return 1
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
    any_attributed = False
    any_unattributed = False
    for path in blocking:
        try:
            d = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            d = {}
        cmd = " ".join(d.get("command", ["?"]))
        print(f"  {path.relative_to(REPO)}  ({cmd})")
        # Markers written before attribution existed carry no split; show them as-is rather than
        # inventing an attribution the run never recorded.
        listed = d.get("modified_tracked_files", [])
        attributed = d.get("attributed_to_run")
        if attributed is None:
            for p in listed:
                print(f"    {p}")
        else:
            unattributed = d.get("unattributed", [])
            for p in attributed:
                print(f"    [run wrote it]   {p}")
            for p in unattributed:
                print(f"    [NOT this run]   {p}")
            any_attributed = any_attributed or bool(attributed)
            any_unattributed = any_unattributed or bool(unattributed)
    names = "  ".join(str(p.relative_to(REPO)) for p in blocking)
    print(
        f"\n  Fix:  git checkout -- <paths>          (the marker then retires itself)\n"
        f"  Or, if the rewrite is intended: explain it in the commit message and delete the\n"
        f"  marker deliberately:  rm {names}"
    )
    if any_unattributed:
        print(
            "\n  CAUTION: the paths marked [NOT this run] were NOT observed being written by the\n"
            "  run that armed the marker. They are more likely a concurrent agent's in-flight\n"
            "  work than test damage. Reverting those is how this guard destroyed authored work\n"
            "  three times -- check authorship before including them in any git checkout."
        )
    hidden = _stash_hidden_paths()
    stash_hidden = sorted(
        {
            f
            for path in blocking
            for f in _marker_listed_files(path)
            if f in hidden and f not in dirty_tracked()
        }
    )
    if stash_hidden:
        print(
            "\n  NOTE: these marker paths look clean ONLY because pre-commit has stashed the\n"
            "  unstaged changes for this hook run; the rewrite is unstaged, not gone, and will\n"
            "  be restored when the hook finishes. Do NOT git checkout them from inside this\n"
            "  window -- resolve them after the commit attempt returns:"
        )
        for f in stash_hidden:
            print(f"    [stash-hidden]   {f}")
    return 1


def _marker_listed_files(path: Path) -> list[str]:
    """The tracked files a marker names; [] when the marker is unreadable."""
    try:
        listed = json.loads(path.read_text()).get("modified_tracked_files", [])
    except (json.JSONDecodeError, OSError, AttributeError):
        return []
    return listed if isinstance(listed, list) else []


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
    ap.add_argument(
        "--mutation-begin",
        action="store_true",
        help="open an exclusive mutation-PROOF session (needs --mutation-target)",
    )
    ap.add_argument(
        "--mutation-end",
        action="store_true",
        help="close the proof session; refuses while a MUTATED marker survives",
    )
    ap.add_argument(
        "--mutation-target",
        default=None,
        help="the file the proof will mutate; resolved against what the interpreter imports",
    )
    ap.add_argument(
        "--mutation-force-unlock",
        action="store_true",
        help="clear a wedged proof lock deliberately (prints what it cleared)",
    )
    ap.add_argument("command", nargs=argparse.REMAINDER, help="the command to run (after --)")
    a = ap.parse_args(argv)

    try:
        if a.mutation_begin and a.mutation_end:
            ap.error("--mutation-begin and --mutation-end are separate calls")
        if a.mutation_force_unlock:
            return cmd_mutation_force_unlock()
        if a.mutation_begin:
            return cmd_mutation_begin(a.mutation_target, a.run_id)
        if a.mutation_end:
            return cmd_mutation_end(a.run_id)

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
    observed = read_observed(read_run_id(run_id) or "")
    _report(muts, observed)
    if do_restore:
        left = _restore_attributed(muts, observed)
        return 0 if not left else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
