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

=====================================================================================
WIDENING, 2026-07-29: A GUARD THAT DID NOT FIRE IS WORSE THAN NO GUARD
=====================================================================================

The 2026-07-29 outer-loop session found a SECOND way the record gets rewritten, and found
that THIS lint sat directly in its path and stayed silent.

Running ``pytest tests/python/test_arc_*.py tests/python/test_experiment_*.py`` left 39
tracked files modified that were clean before the run. The mechanism is a sibling of the
origin incident rather than the same one: a class of ``test_experiment_*.py`` tests calls
``runpy.run_path`` on the REAL ``scripts/experiments/experiment_NNNN_*.py`` and then asserts
the artifact exists. The script writes its artifact as a side effect, so merely RUNNING THE
TEST SUITE overwrites the historical record. Anyone who then commits with ``git add -A``
publishes the rewrites.

Three confirmed instances, all of which this lint's pre-widening form let through:

  1. ``results/experiment_3946_r11l_first_solve.json`` lost FOUR fields --
     ``inference_substrate_correction_note`` and ``inference_substrate_original_invalid_value``
     (a hand-written 2026-07-27 corrigendum explaining why the original substrate string was
     illegal, and what it used to say), plus ``solve_provenance`` and
     ``solve_provenance_note``.
  2. ``results/experiment_307_jepa_real_training.json`` had ``inference_mode`` flipped
     ``live_gpu`` -> ``cpu_training``.
  3. ``results/experiment_1035_dualgpu_rocm_v3.json`` had its run timestamps rewritten.

Why the guard stayed silent on (1): ``inference_substrate_correction_note`` IS a corrigendum
in substance -- it is a dated, hand-written retraction of a field's previous value -- but its
NAME does not contain the string "corrigendum", and the only pattern the lint had was
``CORRIGENDUM_PREFIX = "corrigendum"``. ``solve_provenance`` is the ARC Live-Path
Reachability Discipline's own gate key (``outer_loop_re`` is CRITICAL-flagged,
``live_agent_self_discovery`` is headline-eligible), and it had no protection at all.

Why the guard stayed silent on (2): the lint only ever looked for fields DISAPPEARING. A
substrate/mode field that survives but FLIPS TO A WEAKER VALUE is the same class of harm --
``live_gpu -> cpu_training`` retroactively rewrites what hardware the measurement ran on --
and it defeats CLAUDE.md's Inference-Substrate Declaration Discipline, whose whole purpose is
that the declaration is the ground truth the duration floors are applied against.

THE THREE RULES AS OF THE WIDENING
----------------------------------
  Rule 1 (original)  ``flagged_adversarial: True`` must not silently vanish.
  Rule 2 (original)  the ``corrigendum*`` trail must not be dropped.
  Rule 3 (NEW)       no MARKER field may be dropped. A marker field is one whose NAME marks
                     it as a correction, a provenance declaration, a disclosure, or a review
                     note -- see ``MARKER_PATTERNS``. These are review OUTPUTS, not
                     measurements; a fresh measurement does not supersede them.
  Rule 4 (NEW)       a substrate / inference-mode field may not be WEAKENED (live -> cpu,
                     real -> simulated, anything -> blocked) without an accompanying note.

HOW THE MARKER LIST WAS DERIVED (empirically, not from a wish list)
-------------------------------------------------------------------
The patterns below were derived by censusing every top-level key in all 15,331
``results/**/*.json`` files in the tree (31,510 distinct key names), then keeping the
name-shapes that mark a REVIEW OUTPUT and rejecting the ones that mark a MEASUREMENT. That
distinction is the whole design, and it is why some obvious-looking patterns are absent:

  * ``correct`` is NOT a pattern. 601 artifacts carry ``energy_correct`` / ``sc_correct`` /
    ``judge_correct`` / ``n_correct`` -- these are accuracy MEASUREMENTS, and a re-run that
    changes them is exactly the fail-forward behaviour this lint must never obstruct. Only
    the longer, prose-shaped ``correction`` is matched.
  * ``corrected_`` is NOT a pattern, for the same reason: ``corrected_cdls_projection_mh``
    and ``corrected_cdls_acceptance_rate`` are numbers.
  * ``_note`` IS a pattern, but only fires when the OLD value was substantive (a non-empty
    string / dict / list). A field that was already null or empty carried no record, so
    losing it loses nothing.

SCOPE LIMIT, STATED PLAINLY
---------------------------
Rule 3 checks TOP-LEVEL keys only. Marker-shaped names also occur ~11,260 times NESTED
inside per-row records (``provenance`` 3,859x, ``note`` 2,185x), but those are per-row
bookkeeping that a legitimate re-run rewrites wholesale, and protecting them would refuse
every honest re-run. All three confirmed incidents were top-level. If a future incident is
nested, widen deliberately rather than pre-emptively -- a lint that cries wolf gets disabled,
which is the same outcome as having no lint.

Cross-references for the widening:
- ``ops/known-issues.md`` 2026-07-29 -> the test-suite-rewrites-the-record hazard
- commit ``b3e31d341`` -> the hazard as first diagnosed (not fixed there)
- ``scripts/test_suite_mutation_check.py`` -> the sibling detector that answers "did this
  test run modify tracked files?" BEFORE a commit is ever attempted
- CLAUDE.md "Inference-Substrate Declaration Discipline" -> what Rule 4 protects
- CLAUDE.md "ARC Live-Path Reachability Discipline" -> ``solve_provenance``, protected by
  Rule 3
- CLAUDE.md "QA-Layer Authenticity Discipline" -> why ``_unwrap_principle`` exists here: the
  project allows ANY artifact field to be written as ``{"principle": ..., "value": ...}``,
  and origin bug #2 of that discipline was ``adversarial_verify.py`` reading such a field as
  a bare string. This lint must not repeat it.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The fields whose LOSS changes a gate rather than merely losing history.
# `flagged_adversarial` is the fabrication gate's own key. The corrigendum family is the
# documented reason a determination exists -- losing it strands the stamp without its evidence.
DETERMINATION_FIELD = "flagged_adversarial"
CORRIGENDUM_PREFIX = "corrigendum"

# ---------------------------------------------------------------------------------------
# Rule 3: marker fields -- names that mark a REVIEW OUTPUT rather than a MEASUREMENT.
#
# Each entry is (compiled pattern, short human name used in the refusal message). Read the
# "HOW THE MARKER LIST WAS DERIVED" section of the module docstring before adding one: the
# cost of a false positive here is a refused honest commit, and the loop runs unattended.
# ---------------------------------------------------------------------------------------
MARKER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"corrigend", re.I), "corrigendum record"),
    (re.compile(r"correction", re.I), "correction record"),
    (re.compile(r"provenance", re.I), "provenance declaration"),
    (re.compile(r"^inference_substrate", re.I), "inference-substrate declaration"),
    (re.compile(r"acknowledg", re.I), "acknowledgment record"),
    (re.compile(r"retract", re.I), "retraction record"),
    (re.compile(r"errat", re.I), "erratum record"),
    (re.compile(r"disclos", re.I), "disclosure record"),
    (re.compile(r"caveat", re.I), "caveat record"),
    (re.compile(r"^(notes?|.+_notes?)$", re.I), "review note"),
    (re.compile(r"^verifier_is_oracle$", re.I), "circularity declaration"),
    (re.compile(r"^flagged_adversarial", re.I), "fabrication-gate stamp"),
    (re.compile(r"^preconditions_checked$", re.I), "precondition record"),
]

# ---------------------------------------------------------------------------------------
# Rule 4: substrate / mode weakening.
#
# Fields that DECLARE what compute actually happened. Flipping one of these to a weaker
# value rewrites the provenance of a measurement that already landed.
# ---------------------------------------------------------------------------------------
SUBSTRATE_FIELD = re.compile(
    r"^(inference_substrate|inference_mode|execution_mode|compute_mode|run_mode|substrate"
    r"|.+_substrate|.+_inference_mode)$",
    re.I,
)

# Strength bands, weakest first. A declared value's rank is the MINIMUM band of every token
# it matches, so `cpu_synthetic` ranks as SYNTHETIC (1) rather than as CPU (2), and
# `blocked_no_live_gpu` ranks as NOT-RUN (0) rather than as LIVE (3). Taking the minimum is
# the conservative reading: a string that admits any weak token is at most that strong.
STRENGTH_BANDS: list[tuple[int, tuple[str, ...]]] = [
    (0, ("blocked", "deferred", "not_run", "precondition_check_only", "no_run", "skipped")),
    (1, ("mock", "fake", "synthetic", "simulat", "stub", "dry_run", "placeholder")),
    (2, ("cpu", "offline", "aggregation", "cached", "replay", "analys", "reconciliation")),
    (
        3,
        (
            "live",
            "real",
            "gpu",
            "cuda",
            "hardware",
            "gguf",
            "llm_inference",
            "embedding_extraction",
        ),
    ),
]
_BAND_NAME = {0: "NOT-RUN", 1: "NOT-REAL-COMPUTE", 2: "REAL-BUT-CHEAP", 3: "LIVE/HARDWARE"}


def _unwrap_principle(value):
    """Return the real value of a possibly principle-annotated field.

    CLAUDE.md's "Principle-Annotated Artifact Fields" discipline permits ANY artifact field to
    be written as ``{"principle": "why this field matters", "value": <the real value>}``.
    162 artifacts write ``inference_substrate`` that way. A checker that reads such a field as
    a bare string silently stops recognising it -- that was literally origin bug #2 of the
    QA-Layer Authenticity Discipline, found in ``adversarial_verify.py``. Unwrap once, here,
    so no call site has to remember.
    """
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _is_substantive(value) -> bool:
    """Did this field actually CARRY a record, or was it an empty placeholder?

    Losing a field whose value was ``None`` / ``""`` / ``[]`` loses no information, and
    refusing a commit over it would be pure noise.
    """
    v = _unwrap_principle(value)
    if v is None:
        return False
    if isinstance(v, (str, list, dict, tuple)):
        return len(v) > 0
    return True


def _marker_kind(key: str) -> str | None:
    """The human name of the marker class `key` belongs to, or None if it is not a marker."""
    for pattern, kind in MARKER_PATTERNS:
        if pattern.search(str(key)):
            return kind
    return None


def _strength_rank(value) -> int | None:
    """Rank a declared substrate/mode value, or None when it matches no known vocabulary.

    None means "unrankable", NOT "weak". An unknown string must never be treated as a
    downgrade -- the project invents new substrate strings constantly (2,842 declarations
    across ~40 distinct vocabularies), and ranking an unrecognised one as 0 would refuse a
    large fraction of honest commits.
    """
    v = _unwrap_principle(value)
    if not isinstance(v, str):
        return None
    low = v.lower()
    matched = [band for band, tokens in STRENGTH_BANDS if any(t in low for t in tokens)]
    return min(matched) if matched else None


def _has_change_note(new: dict, field: str) -> bool:
    """Is there an explicit, auditable note beside a weakened/cleared declaration?

    Same escape hatch as the original ``*_cleared_note``: clearing or downgrading on PURPOSE
    is legitimate work, but it has to be stated where the next reader will find it, so it is
    distinguishable from the accident this lint exists to catch.

    The note must NAME THE FIELD it excuses. An earlier draft also accepted any key ending in
    ``_change_note`` regardless of what it referred to, which meant an unrelated
    ``corpus_change_note`` elsewhere in the artifact would silently excuse a substrate downgrade.
    That is the "guard that does not fire" failure mode in miniature, so the field-agnostic
    branch was removed rather than kept for convenience.
    """
    stem = str(field).lower()
    for k, v in new.items():
        kl = str(k).lower()
        if not v or kl == stem:
            continue
        if stem in kl and any(
            w in kl for w in ("note", "change", "downgrade", "rationale", "cleared")
        ):
            return True
    return False


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

        # 3. (2026-07-29 widening) ANY marker field -- a correction, a provenance declaration,
        #    a disclosure, a review note. The origin of this rule is that
        #    `inference_substrate_correction_note` IS a corrigendum in substance but does not
        #    contain the string "corrigendum", so rule 2 above sailed straight past its
        #    deletion. Reported separately from rule 2 so the two stay individually testable.
        lost_markers: dict[str, list[str]] = {}
        # `already_reported` keeps one deletion from producing two refusal lines. Rule 1 owns
        # `flagged_adversarial` when it was True (that message explains the quarantine-lifting
        # consequence, which is the important part); rule 2 owns the corrigendum family. Rule 3
        # still covers `flagged_adversarial: False` being dropped, which rule 1 ignores.
        already_reported = set(lost)
        if old.get(DETERMINATION_FIELD) is True:
            already_reported.add(DETERMINATION_FIELD)
        for key, old_value in old.items():
            if key in new or key in already_reported:
                continue
            if not _is_substantive(old_value):
                continue
            # A DROPPED substrate declaration is the loophole rule 4 cannot see, because rule 4
            # only compares a field that exists on BOTH sides. `inference_substrate*` is caught
            # by the marker patterns, but a bare `inference_mode` -- the exact field the exp307
            # incident touched -- matches no marker pattern, so deleting it outright would have
            # escaped every rule while flipping it merely weaker was refused. Absent is strictly
            # worse than weaker: the linter's strict default then reads the artifact as
            # `live_llm_inference` and applies the 60s floor to something that may have run in
            # milliseconds.
            kind = _marker_kind(key) or (
                "substrate declaration" if SUBSTRATE_FIELD.match(str(key)) else None
            )
            if kind:
                lost_markers.setdefault(kind, []).append(str(key))
        for kind, keys in sorted(lost_markers.items()):
            violations.append(
                f"{path}: lost {kind}(s) {sorted(keys)}. A field whose NAME marks it as a "
                f"review output is not superseded by a re-run's fresh measurements; if it no "
                f"longer applies, say so explicitly beside it (CLAUDE.md never-prune)."
            )

        # 4. (2026-07-29 widening) A substrate / mode declaration that survives but is
        #    WEAKENED. `live_gpu -> cpu_training` (exp307) retroactively rewrites what hardware
        #    a landed measurement ran on, and it defeats the duration floors that CLAUDE.md's
        #    Inference-Substrate Declaration Discipline applies per-substrate.
        for key, old_value in old.items():
            if not SUBSTRATE_FIELD.match(str(key)) or key not in new:
                continue
            old_rank = _strength_rank(old_value)
            new_rank = _strength_rank(new[key])
            if old_rank is None or new_rank is None or new_rank >= old_rank:
                continue
            if _has_change_note(new, key):
                continue
            ov = _unwrap_principle(old_value)
            nv = _unwrap_principle(new[key])
            violations.append(
                f"{path}: {key} WEAKENED {ov!r} ({_BAND_NAME[old_rank]}) -> {nv!r} "
                f"({_BAND_NAME[new_rank]}) with no accompanying note. This rewrites the "
                f"declared provenance of a measurement that already landed; the substrate "
                f"declaration is what the fabrication gate's duration floors key off."
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
        "  A results artifact lost a FABRICATION-GATE determination, lost a review record, or\n"
        "  had its declared compute substrate weakened. This is not a history problem -- every\n"
        "  consumer of CLAUDE.md's fabrication gate keys off these fields being PRESENT and\n"
        "  ACCURATE, so dropping or weakening one silently re-admits a quarantined artifact to\n"
        "  capstone / evidence-table / paper-v6 aggregation.\n"
        "\n"
        "  A COMMON CAUSE IS NOT A HUMAN EDIT AT ALL: running the test suite re-executes real\n"
        "  experiment scripts, which overwrite their own artifacts in place. Run\n"
        "  `python3 scripts/test_suite_mutation_check.py --check` to see whether that is what\n"
        "  happened; if so, `git checkout -- <paths>` rather than committing the rewrite.\n"
    )
    for v in violations:
        print(f"  - {v}")
    print(
        "\n  Re-run measurements are fine and expected (fail-forward). Restore the determination\n"
        "  fields alongside the new numbers. To clear a determination on purpose, set it to\n"
        "  false AND add a `*_cleared_note` stating what you re-verified -- see this file's\n"
        "  docstring. To weaken a substrate declaration on purpose, add a note beside it saying\n"
        "  why the cheaper substrate is now correct. Origin incidents: ops/changelog.md\n"
        "  2026-07-27 (dropped stamps) and 2026-07-29 (test suite rewrites the record)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
