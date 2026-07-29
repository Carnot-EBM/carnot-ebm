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

    # ...and put everything the run touched back the way it was.
    python3 scripts/test_suite_mutation_check.py --restore --run -- pytest tests/python

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
    """
    out: dict[str, str] = {}
    for line in _git("status", "--porcelain", "-uno", "-z").split("\0"):
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:]
        # Rename entries carry "old\0new"; -z already split them, and the tail half arrives as
        # a bare path with no status code, which the length check above discards. Renames of
        # tracked artifacts are rare enough that missing one is preferable to mis-parsing.
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


def restore(paths: list[str]) -> list[str]:
    """Put the named paths back to their committed content. Returns the ones that came back."""
    if not paths:
        return []
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
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    PENDING.write_text(
        json.dumps(
            {
                "detected_at": datetime.now(UTC).isoformat(),
                "command": argv,
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
