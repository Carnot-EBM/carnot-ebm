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

    # Or do it by hand, around anything at all:
    python3 scripts/test_suite_mutation_check.py --snapshot
    <do the thing>
    python3 scripts/test_suite_mutation_check.py --check

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

Cross-references:
- commit ``b3e31d341``                                -> the hazard as first diagnosed
- ``scripts/determination_preservation_lint.py``      -> the content-level sibling guard
- ``ops/known-issues.md`` 2026-07-29                  -> the survey and the repair options
- CLAUDE.md "Documentation Update Rules" (never-prune) -> what the rewrites violate
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "ops" / ".test_suite_mutation_snapshot.json"
PENDING = REPO / "ops" / ".test_suite_mutation_pending.json"
# Where `restore()` parks a file's pre-revert content. `git checkout --` cannot be undone, and the
# detector cannot tell a test's write from a human's concurrent edit, so every revert is made
# recoverable rather than trusted to be correct. See `backup()`.
BACKUP = REPO / "ops" / ".test_suite_mutation_backup"


def _git(*args: str) -> str:
    """Run git in the repo and return stdout. Errors surface as empty output, never a crash.

    A detector that raises on an unexpected git state is worse than one that reports nothing:
    it turns a diagnostic into an outage in whatever wrapped it.
    """
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


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


def snapshot() -> dict[str, str]:
    baseline = dirty_tracked()
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(
        json.dumps(
            {"taken_at": datetime.now(UTC).isoformat(), "already_dirty": baseline},
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    return baseline


def _load_snapshot() -> dict[str, str] | None:
    if not SNAPSHOT.exists():
        return None
    try:
        return json.loads(SNAPSHOT.read_text()).get("already_dirty", {})
    except (json.JSONDecodeError, OSError):
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


def write_pending(muts: list[str], command: list[str]) -> None:
    """Arm the pre-commit interlock: record which tracked files a run rewrote.

    Factored out of ``cmd_run`` so the pytest auto-arm hook (``arm_from_pytest``) writes the
    SAME marker in the SAME shape. Two code paths writing two slightly different markers is how
    a gate ends up passing on one of them.
    """
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    PENDING.write_text(
        json.dumps(
            {
                "detected_at": datetime.now(UTC).isoformat(),
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
        PENDING.unlink(missing_ok=True)
        return []
    write_pending(muts, command)
    return muts


def cmd_run(argv: list[str], do_restore: bool) -> int:
    baseline = snapshot()
    print(f"test-suite-mutation-check: baseline taken ({len(baseline)} file(s) already dirty)")
    proc = subprocess.run(argv, cwd=REPO)
    muts = mutations(baseline)
    if not muts:
        PENDING.unlink(missing_ok=True)
        print("test-suite-mutation-check: OK -- no tracked file was modified by the run")
        return proc.returncode
    _report(muts)
    if do_restore:
        recovered = restore(muts)
        print(f"\n  restored {len(recovered)}/{len(muts)} file(s) to their committed content")
        left = [p for p in muts if p not in recovered]
        if not left:
            PENDING.unlink(missing_ok=True)
            return proc.returncode or 0
        muts = left
        print(f"  COULD NOT RESTORE: {left}")
    write_pending(muts, argv)
    return 1


def cmd_gate() -> int:
    if not PENDING.exists():
        print("test-suite-mutation-check: OK")
        return 0
    try:
        d = json.loads(PENDING.read_text())
    except (json.JSONDecodeError, OSError):
        d = {}
    print("test-suite-mutation-check: REFUSING THE COMMIT.")
    print(
        f"  A previous test run ({' '.join(d.get('command', ['?']))}) modified tracked files and\n"
        f"  they were not restored. Committing now would publish a silent rewrite of the\n"
        f"  research record.\n"
    )
    for p in d.get("modified_tracked_files", []):
        print(f"    {p}")
    print(
        f"\n  Fix:  git checkout -- <paths>   then   rm {PENDING.relative_to(REPO)}\n"
        f"  Or, if the rewrite is intended: explain it in the commit message and delete\n"
        f"  {PENDING.relative_to(REPO)} deliberately."
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
    ap.add_argument("command", nargs=argparse.REMAINDER, help="the command to run (after --)")
    a = ap.parse_args(argv)

    if a.gate:
        return cmd_gate()

    if a.run:
        cmd = [c for c in a.command if c != "--"]
        if not cmd:
            ap.error("--run needs a command: --run -- pytest tests/python")
        return cmd_run(cmd, a.restore)

    if a.snapshot:
        b = snapshot()
        print(f"test-suite-mutation-check: baseline taken ({len(b)} file(s) already dirty)")
        return 0

    if a.check:
        baseline = _load_snapshot()
        if baseline is None:
            print("test-suite-mutation-check: no snapshot -- run --snapshot first")
            return 0
        muts = mutations(baseline)
        if not muts:
            print(
                "test-suite-mutation-check: OK -- no tracked file was modified since the snapshot"
            )
            return 0
        _report(muts)
        if a.restore:
            recovered = restore(muts)
            print(f"\n  restored {len(recovered)}/{len(muts)} file(s)")
            return 0 if len(recovered) == len(muts) else 1
        return 1

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
