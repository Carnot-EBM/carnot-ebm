#!/usr/bin/env python3
"""Refuse any commit that DROPS a fabrication-gate determination from a results artifact.

WHY THIS EXISTS (2026-07-27 incident, found by the outer loop).
-------------------------------------------------------------
The conductor re-runs experiments, and an experiment script writes its artifact to a FIXED
path (``results/experiment_<N>_<name>.json``). So a re-run OVERWRITES the previous artifact in
place. On 2026-07-27 that silently dropped ``flagged_adversarial: True`` from SEVEN artifacts
(exp1861, exp1938, exp2085, exp3734, exp4162, exp4170, exp696), six of which also lost their
``corrigendum_pending`` + ``corrigendum_note`` records.

That is far worse than an ordinary never-prune violation, and the reason is mechanical.
CLAUDE.md's fabrication gate says:

    "Capstone, evidence-table, paper-v6, and any headline-aggregation task MUST skip artifacts
     carrying ``flagged_adversarial: true`` -- never aggregate a flagged artifact's numbers
     into a milestone result or a forward-facing claim."

Every consumer of that rule keys off the FIELD BEING PRESENT. So an overwrite that strips the
field does not merely lose history -- it RE-ADMITS a quarantined artifact to headline
aggregation, silently, with no human-read diff anywhere in the loop. The quarantine is undone
by the very act of re-running the experiment, which is the one thing the loop does constantly.

Critically, the 7 artifacts were re-verified after the overwrite and ``adversarial_verify.py``
still reported ``1 flagged`` on ALL of them. The determinations were LIVE, not stale relics of
an older verifier. So this was not "an obsolete stamp got cleaned up" -- it was a live
quarantine being lifted by accident.

WHY A LINT AND NOT A CONDUCTOR-SIDE FIX
---------------------------------------
The obvious fix -- "make the conductor stop overwriting" -- cannot work at that layer: the
conductor does not write these files. The experiment SCRIPT does, and there are thousands of
them, written by many agents over many months, each choosing its own output path. Any guard
placed in the conductor's own write path would miss every one of them.

A commit-time diff check catches the entire class regardless of which script wrote the file,
which agent ran it, or whether the write was even intentional. It is the same reasoning that
put ``canonical_url_lint`` and ``verifier_authenticity_lint`` at Layer 1 rather than in the
code that happens to emit the violation.

WHAT THIS DOES *NOT* DO
-----------------------
It does not object to a re-run CHANGING NUMBERS. Fail-forward is the operator's standing
directive ("always committing and never reverting so that we fail forward and fix any problems
rather than lose transient assets"), and a re-run producing new measurements is normal, healthy
work. This lint is narrowly about the DETERMINATION fields -- the recorded judgement that an
artifact is quarantined, and the corrigendum trail explaining why. Those are review outputs,
not measurements, and a fresh measurement does not supersede them.

Nor does it require the stamp to be correct forever. If a determination genuinely no longer
applies, the correct action is to say so explicitly (see "HOW TO CLEAR A DETERMINATION" below)
rather than to let a re-run drop it on the floor.

HOW TO CLEAR A DETERMINATION LEGITIMATELY
-----------------------------------------
Keep the field and add an explicit retraction beside it, so the clearing is auditable:

    "flagged_adversarial": false,
    "flagged_adversarial_cleared_note": "Cleared 2026-07-27: the DURATION_TOO_SHORT flag was a
        false positive under the pre-2026-07-03 substrate taxonomy; this run declares
        inference_substrate: live_llm_embedding_extraction and adversarial_verify.py now
        reports 0 flagged. Verified by re-running the linter against THIS artifact."

The lint accepts ``flagged_adversarial: false`` when a ``*_cleared_note`` accompanies it. It
refuses a silent transition to absent/None, because that is indistinguishable from an accident
-- which is exactly what the origin incident was.

USAGE
-----
    python3 scripts/determination_preservation_lint.py              # staged changes (pre-commit)
    python3 scripts/determination_preservation_lint.py --ref HEAD~1 # audit a landed commit
    python3 scripts/determination_preservation_lint.py --all        # sweep the whole tree vs HEAD

Exit 0 = clean, 1 = a determination was dropped (refuse the commit).

Cross-references:
- CLAUDE.md "Adversarial Artifact Verification + Sample-Size Rigor" -> the fabrication gate
  whose consumers key off ``flagged_adversarial``
- CLAUDE.md "Documentation Update Rules" -> the never-prune rule this enforces mechanically for
  the one field class where losing it changes a GATE rather than just losing history
- ``scripts/adversarial_verify.py`` -> writes the determinations this protects
- ``ops/changelog.md`` 2026-07-27 -> the origin incident and the restoration
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The fields whose LOSS changes a gate rather than merely losing history.
# `flagged_adversarial` is the fabrication gate's own key. The corrigendum family is the
# documented reason a determination exists -- losing it strands the stamp without its evidence.
DETERMINATION_FIELD = "flagged_adversarial"
CORRIGENDUM_PREFIX = "corrigendum"


def _tracked_json_under_results(ref: str | None, all_files: bool) -> list[str]:
    """Which results/*.json files to check: modified-vs-HEAD, vs a ref, or every tracked one.

    THE DEFAULT COMPARES AGAINST ``HEAD`` WITHOUT ``--cached``, DELIBERATELY. A first draft used
    ``git diff --cached --name-only`` for the file list while reading the NEW side from the
    working tree -- so an UNSTAGED strip produced an empty file list and the lint printed OK on
    a tree that had just lost a determination. It failed to fire on a faithful replay of its own
    origin incident, which is the one bug a guard cannot have.

    Comparing against ``HEAD`` is correct in both contexts: pre-commit stashes unstaged changes
    before running hooks, so the working tree it sees IS the staged content; and a human running
    this on a dirty tree gets every modification, staged or not.
    """
    if all_files:
        cmd = ["git", "ls-files", "results/*.json"]
    elif ref:
        cmd = ["git", "diff", "--name-only", "--diff-filter=M", ref, "HEAD", "--", "results"]
    else:
        cmd = ["git", "diff", "--name-only", "--diff-filter=M", "HEAD", "--", "results"]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO).stdout
    return [p for p in out.splitlines() if p.endswith(".json")]


def _load_at(rev: str, path: str) -> dict | None:
    """Read a JSON artifact as of `rev`. None when absent or unparseable (not a violation)."""
    r = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        return None
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def _load_now(path: str, ref: str | None) -> dict | None:
    """Read the NEW side: the working tree (pre-commit) or `HEAD` (auditing a landed commit)."""
    if ref:
        return _load_at("HEAD", path)
    p = REPO / path
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return d if isinstance(d, dict) else None


def _corrigendum_keys(d: dict) -> set[str]:
    return {k for k in d if CORRIGENDUM_PREFIX in str(k).lower()}


def _cleared_deliberately(new: dict) -> bool:
    """A determination may be cleared to False IF an explicit note accompanies it.

    Absent/None is NOT a legitimate clearing -- it is indistinguishable from the accident this
    lint exists to catch. Requiring a note makes the clearing auditable and forces whoever
    clears it to state their reasoning where the next reader will find it.
    """
    if new.get(DETERMINATION_FIELD) is not True and DETERMINATION_FIELD in new:
        return any("cleared" in str(k).lower() and new.get(k) for k in new)
    return False


def check(ref: str | None = None, all_files: bool = False) -> list[str]:
    base = ref if ref else "HEAD"
    violations: list[str] = []
    for path in _tracked_json_under_results(ref, all_files):
        old = _load_at(base, path)
        if old is None:
            continue
        new = _load_now(path, ref)
        if new is None:
            continue

        # 1. The fabrication-gate stamp must not silently vanish.
        if old.get(DETERMINATION_FIELD) is True and new.get(DETERMINATION_FIELD) is not True:
            if _cleared_deliberately(new):
                pass  # explicit, auditable retraction -- allowed
            else:
                violations.append(
                    f"{path}: {DETERMINATION_FIELD} True -> "
                    f"{new.get(DETERMINATION_FIELD)!r} with no *_cleared_note. This LIFTS a "
                    f"quarantine: the fabrication gate keys off this field, so dropping it "
                    f"re-admits the artifact to headline aggregation."
                )

        # 2. The corrigendum trail explains WHY a determination exists; losing it strands the
        #    stamp without its evidence, and it is pure history that no re-run supersedes.
        lost = _corrigendum_keys(old) - _corrigendum_keys(new)
        if lost:
            violations.append(
                f"{path}: lost corrigendum record(s) {sorted(lost)}. These document why the "
                f"artifact was flagged; a re-run's fresh numbers do not supersede a review's "
                f"recorded judgement (CLAUDE.md never-prune)."
            )
    return violations


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", help="audit REF..HEAD instead of the staged diff (e.g. HEAD~1)")
    ap.add_argument("--all", action="store_true", help="sweep every tracked results/*.json vs HEAD")
    ap.add_argument("files", nargs="*", help="ignored; accepted so pre-commit may pass filenames")
    a = ap.parse_args(argv)

    violations = check(ref=a.ref, all_files=a.all)
    if not violations:
        print("determination-preservation-lint: OK")
        return 0

    print("determination-preservation-lint: REFUSING THE COMMIT.")
    print(
        "  A results artifact lost a FABRICATION-GATE determination. This is not a history\n"
        "  problem -- every consumer of CLAUDE.md's fabrication gate keys off the field being\n"
        "  PRESENT, so dropping it silently re-admits a quarantined artifact to capstone /\n"
        "  evidence-table / paper-v6 aggregation.\n"
    )
    for v in violations:
        print(f"  - {v}")
    print(
        "\n  Re-run measurements are fine and expected (fail-forward). Restore the determination\n"
        "  fields alongside the new numbers. To clear a determination on purpose, set it to\n"
        "  false AND add a `*_cleared_note` stating what you re-verified -- see this file's\n"
        "  docstring. Origin incident: ops/changelog.md 2026-07-27."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
