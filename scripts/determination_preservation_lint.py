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

  * ``correct`` is NOT a pattern. 465 artifacts carry a top-level ``*correct*`` key that is
    not ``*correction*`` -- ``energy_correct`` (201), ``sc_correct`` (200),
    ``energy_pure_correct`` (200), ``judge_correct`` (200), ``n_correct`` (37) and others --
    and these are accuracy MEASUREMENTS. A re-run that changes them is exactly the
    fail-forward behaviour this lint must never obstruct. Only the longer, prose-shaped
    ``correction`` is matched.

    (The figure in the first draft of this docstring was 601, which does not reproduce. The
    predicate that yields 465 is: top-level keys only, case-insensitive ``correct`` in the
    name, excluding names containing ``correction``, counted once per artifact. The design
    decision is unaffected -- those are measurements either way -- but an unreproducible
    number is not evidence, so it is corrected rather than left standing.)
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


class GuardError(RuntimeError):
    """The guard could not complete its check, so it does not know whether the tree is clean.

    A GUARD MUST FAIL CLOSED. Every path that raises this used to return an empty result or a
    ``None`` instead, which made the lint print ``OK`` on a tree it had not actually examined --
    the single failure mode a guard cannot have, and the one this whole file exists to prevent
    in the artifacts it protects.

    ``main`` catches this and REFUSES the commit with the underlying reason. If the environment
    is genuinely broken, the correct outcome is a human looking at it, not a silent pass.
    """


# `git show <rev>:<path>` exits 128 both for "that path is not in that revision" (routine and
# benign -- a newly added artifact has no old side) and for "your repository is broken". Only
# the first is safe to swallow, so it is matched explicitly and EVERYTHING ELSE fails closed.
# Both spellings below are real git messages; the second appears when the path exists in the
# working tree but not in the given revision.
_GIT_PATH_ABSENT = re.compile(
    # `git show <rev>:<path>` -- path absent from that revision.
    r"does not exist in"
    # `git show :<path>` (the INDEX side) -- both real spellings, captured by running them
    # rather than guessed. The second is why this list is explicit: "does not exist (neither
    # on disk nor in the index)" does NOT contain the substring "does not exist in", so the
    # first alternative alone would have failed closed on a routine absent-from-index path.
    r"|does not exist \(neither on disk nor in the index\)"
    r"|exists on disk, but not in"
    r"|no such path"
    r"|不存在",
    re.I,
)


def _git(args: list[str]) -> str:
    """Run a git command, or raise GuardError. Never returns partial/empty output on failure.

    The pre-widening code did ``subprocess.run(...).stdout`` and ignored ``returncode``
    entirely. Any git failure -- broken repo, missing binary, bad ref, index lock contention
    with a concurrent workflow -- yielded an empty file list, which the loop below reads as
    "nothing changed" and reports as OK.
    """
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO)
    except OSError as exc:  # git absent / not executable / cwd gone
        raise GuardError(f"could not execute `git {' '.join(args)}`: {exc}") from exc
    if r.returncode != 0:
        raise GuardError(
            f"`git {' '.join(args)}` failed with exit {r.returncode}: "
            f"{r.stderr.strip() or '(no stderr)'}"
        )
    return r.stdout


# Sentinels for the NEW side of a comparison. `None` used to mean all three of "deleted",
# "unparseable" and "fine, nothing to do", and every one of them was skipped -- so deleting an
# artifact outright, or overwriting it with corrupt bytes, was indistinguishable from no change.
class _Missing:
    """The artifact is not present on this side at all (deleted, or never existed)."""


class _Unreadable:
    """The artifact IS present but could not be read as a JSON object (corrupt / truncated)."""


MISSING = _Missing()
UNREADABLE = _Unreadable()

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
    # Added 2026-07-29 after a review found these review-output names uncovered. Each is a
    # judgement a REVIEW produced, which a re-run's fresh numbers do not supersede:
    #   justif...        -> `n_samples_justification` (56), the sample-size disclosure that
    #                       CLAUDE.md's Adversarial Artifact Verification rule REQUIRES for
    #                       any distributional claim.
    #   false_negative.. -> `false_negative_risk_checked` (63), the FALSE_NEGATIVE_RISK
    #                       positive-control check.
    #   forbidden_claims -> `paper_v6_forbidden_claims` (22), the Paper-v6 Narrowing
    #                       Discipline's retraction list.
    # `honest_verdict` (5,245) is deliberately NOT here: a re-run legitimately produces a new
    # verdict, so protecting it would refuse the fail-forward behaviour this lint must allow.
    (re.compile(r"justif", re.I), "sample-size justification"),
    (re.compile(r"false_negative_risk", re.I), "false-negative-risk check"),
    (re.compile(r"forbidden_claims", re.I), "forbidden-claims record"),
    (re.compile(r"^adversarial_verify_flags$", re.I), "adversarial-verify flag record"),
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

# `inference_substrate` is the ONE field in this family governed by a documented fixed
# vocabulary (CLAUDE.md "Inference-Substrate Declaration Discipline"), enumerated in
# `adversarial_verify.py`'s three alias tuples. It is therefore ranked from that enum and
# NEVER by scanning prose -- see `_strength_rank` for why that distinction is load-bearing.
ENUM_GOVERNED_SUBSTRATE_FIELD = "inference_substrate"

# Strength bands, weakest first. A declared value's rank is the MINIMUM band of every token
# it matches, so `cpu_synthetic` ranks as SYNTHETIC (1) rather than as CPU (2), and
# `blocked_no_live_gpu` ranks as NOT-RUN (0) rather than as LIVE (3). Taking the minimum is
# the conservative reading: a string that admits any weak token is at most that strong.
#
# TOKENS ARE ANCHORED AT A TOKEN START, NOT MATCHED AS BARE SUBSTRINGS (2026-07-29). The
# first draft used `token in value`, so `real` matched inside `unreal`, `cpu` inside
# `cpu`-free prose, and so on. Values are normalised (non-alphanumeric runs -> `_`) and each
# band token must begin at a token boundary. Tokens stay PREFIXES rather than whole words on
# purpose -- `simulat` has to catch `simulated` and `simulation`, and `analys` has to catch
# `analysis` and `analyser`.
#
# Token anchoring does NOT resolve the `arc_live_path_patch_synthesis` collision, because
# that value genuinely contains `live` as a whole token. That case is handled structurally
# instead, by ranking `inference_substrate` from the canonical enum only -- see
# `_strength_rank`.
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

# Bands for the CANONICAL substrate enum, keyed by `adversarial_verify`'s own three kinds.
# Aggregation and no-LLM work is real but cheap (band 2); anything that loads a model is
# band 3. `hardware_smoke` is documented in CLAUDE.md's substrate table but is absent from
# `adversarial_verify`'s alias tuples, so it is supplemented here rather than silently
# becoming unrankable.
_ENUM_KIND_BAND = {"aggregation": 2, "no_llm": 2, "live_model": 3}
_SUPPLEMENTARY_ENUM_BANDS = {"hardware_smoke": 3}

# Separators after which a human note may follow the canonical value, per
# `adversarial_verify._inference_substrate_value_matches`. Kept only as the fallback for
# free-form (non-enum) fields; the enum path defers to that function so the two cannot drift.
_NOTE_SEPARATORS = " -;,:.("


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


_AV_MODULE = None


def _adversarial_verify():
    """Import ``adversarial_verify`` once, or raise GuardError.

    THIS USED TO BE A FAIL-OPEN, and the comment excusing it ("a guard must never die on an
    import") had the principle exactly backwards. What a guard must never do is claim a tree is
    clean when it could not check. Swallowing the ImportError silently disabled the CANONICAL
    ENUM half of rule 4 and fell through to the free-form token scan -- i.e. it silently
    substituted a DIFFERENT, weaker matcher, which is precisely the "drifted copy of a matcher"
    hazard ``_canonical_substrate_band``'s own docstring is written to prevent.

    Failing closed here cannot deadlock a repair of ``adversarial_verify.py`` itself: the
    pre-commit hook is scoped ``files: ^results/.*\\.json$``, so a commit that touches only
    ``scripts/adversarial_verify.py`` never invokes this lint at all.

    Cached in a module global so a sweep over 15k artifacts imports once, and so the repeated
    ``sys.path.insert`` of the previous implementation (which appended a duplicate entry to
    ``sys.path`` on EVERY call) is gone.
    """
    global _AV_MODULE
    if _AV_MODULE is None:
        scripts_dir = str(REPO / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        try:
            import adversarial_verify as _av
        except Exception as exc:
            raise GuardError(
                f"could not import scripts/adversarial_verify.py ({exc!r}). Rule 4 ranks the "
                f"enum-governed `inference_substrate` field from THAT module's alias tuples; "
                f"without it this lint would silently fall back to a weaker prose scan."
            ) from exc
        _AV_MODULE = _av
    return _AV_MODULE


def _marker_kind(key: str) -> str | None:
    """The human name of the marker class `key` belongs to, or None if it is not a marker."""
    for pattern, kind in MARKER_PATTERNS:
        if pattern.search(str(key)):
            return kind
    return None


def _canonical_substrate_band(raw: str) -> int | None:
    """Band of a value that declares a CANONICAL substrate, else None.

    Delegates the match to ``adversarial_verify._match_declared_substrate`` rather than
    re-deriving it. That function already implements the project's documented
    ``<canonical value><separator><human note>`` convention -- it strips the note and matches
    only the leading canonical token. Re-implementing that here is exactly how the two would
    drift, and a drifted copy of a matcher is how this rule got its worst bug (below).
    """
    _av = _adversarial_verify()
    for kind, aliases in (
        ("aggregation", _av.AGGREGATION_SUBSTRATE_ALIASES),
        ("no_llm", _av.NO_LLM_SUBSTRATE_ALIASES),
        ("live_model", _av.LIVE_MODEL_SUBSTRATE_ALIASES),
    ):
        if _av._match_declared_substrate(raw, tuple(aliases)) is not None:
            return _ENUM_KIND_BAND[kind]
    head = raw.split("--", 1)[0].strip()
    for sep in _NOTE_SEPARATORS:
        head = head.split(sep, 1)[0].strip()
    return _SUPPLEMENTARY_ENUM_BANDS.get(head.lower())


def _token_scan_band(raw: str) -> int | None:
    """Band of a FREE-FORM (non-enum) declaration, by anchored token scan, else None."""
    head = raw.split("--", 1)[0].strip()
    for sep in _NOTE_SEPARATORS:
        head = head.split(sep, 1)[0].strip()
    norm = "_" + re.sub(r"[^a-z0-9]+", "_", head.lower()).strip("_") + "_"
    matched = [band for band, tokens in STRENGTH_BANDS if any(("_" + t) in norm for t in tokens)]
    return min(matched) if matched else None


def _strength_rank(value, field: str | None = None) -> int | None:
    """Rank a declared substrate/mode value, or None when it matches no known vocabulary.

    None means "unrankable", NOT "weak". An unknown string must never be treated as a
    downgrade -- the project invents new substrate strings constantly (2,673 declarations
    across ~40 distinct vocabularies), and ranking an unrecognised one as 0 would refuse a
    large fraction of honest commits.

    TWO BUGS THIS SIGNATURE EXISTS TO FIX (2026-07-29 review, both found against real
    artifacts rather than fixtures):

    1. RANKING PROSE. The first draft scanned the WHOLE declaration string for band tokens.
       The project's documented convention is ``<canonical value><separator><human note>``
       (233+ live declarations use it), and the note is frequently a substrate CORRECTION
       that names the substrates it is correcting away from. exp5178 declares
       ``"live_llm_embedding_extraction; Substrate corrected 2026-07-03: ... "``, whose note
       contains the word ``cached``; the whole-string scan therefore took the MINIMUM band
       across the prose and ranked a real GGUF load (band 3) as REAL-BUT-CHEAP (band 2),
       refusing the commit and asserting the exact opposite of what the artifact says.
       exp5161 is the same shape. Both are CLAUDE.md's own named exemplars for their
       substrates, so the rule was refusing the convention the project documents.

    2. PHANTOM BAND-3 FROM A NON-SUBSTRATE NAME. exp5240 declared
       ``arc_live_path_patch_synthesis`` -- the ARC live CODE path, not a compute substrate,
       and never a legal value under the Inference-Substrate Declaration Discipline's fixed
       enum. A token scan reads its ``live`` token as LIVE/HARDWARE, so when a later commit
       honestly corrected it to the legal ``aggregation_from_upstream_artifacts``, the rule
       saw band 3 -> band 2 and refused a taxonomy REPAIR as a downgrade.

    The fix for (2) is structural, not another token-list patch: for the one field governed
    by a documented fixed vocabulary, rank ONLY from that vocabulary. An unrecognised
    ``inference_substrate`` is UNKNOWN, exactly as ``adversarial_verify`` itself classifies
    it (``unknown_top_level_inference_substrate``), and unknown is unrankable on BOTH sides.
    That symmetry is the point: the rule already refused to treat an unknown NEW value as
    weak, and it now likewise refuses to treat an unknown OLD value as strong. No protection
    is lost -- a genuine enum-to-enum downgrade is still caught -- and the collision cannot
    recur for any future ``arc_live_*`` name.

    Free-form mode fields (``inference_mode``, ``execution_mode``, ...) have no enum, so they
    keep the token scan. That is where the origin incident lives: exp307's
    ``inference_mode: live_gpu -> cpu_training``.
    """
    v = _unwrap_principle(value)
    if not isinstance(v, str):
        return None
    raw = v.strip()
    if not raw:
        return None
    enum_band = _canonical_substrate_band(raw)
    if enum_band is not None:
        return enum_band
    scan = _token_scan_band(raw)
    if field is not None and str(field).lower() == ENUM_GOVERNED_SUBSTRATE_FIELD:
        # ASYMMETRY, AND IT IS THE WHOLE POINT. For the enum-governed field, an unrecognised
        # name may rank WEAK but never STRONG.
        #
        # A first attempt at fix (2) above returned None for every non-enum value. That fixed
        # exp5240 but silently disarmed the rule against `sota_gguf_mock` -- CLAUDE.md's own
        # fabrication exemplar (exp3397: `duration_s=2.06` declaring a live 35B GGUF) -- which
        # is not in the enum either. Two existing tests caught it, which is what they are for.
        #
        # The distinction that survives both cases: a name asserting STRENGTH (`live`, `gpu`)
        # is a claim, and claims are cheap to make by accident -- `arc_live_path_patch_synthesis`
        # asserts nothing about compute, it just happens to contain the token. A name asserting
        # WEAKNESS (`mock`, `fake`, `blocked`) is an ADMISSION, and nobody accidentally admits
        # their run was mocked. So admissions are trusted from any string; claims are trusted
        # only from the documented vocabulary.
        return None if scan == 3 else scan
    return scan


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

    THE NOTE MAY BE CARRIED INLINE IN THE VALUE ITSELF (2026-07-29). An earlier draft only
    inspected SIBLING keys, which missed the project's own documented convention: the
    substrate field may be written ``<canonical value><separator><human note>``, and that
    trailing note is very often precisely the downgrade rationale this function is looking
    for. exp5161 corrects ``inference_substrate`` down to
    ``verifier_ensemble_against_cached_candidates`` and explains why in the same string --
    an explicit, auditable, dated justification sitting exactly where the next reader will
    find it, which is all this escape hatch ever asked for. Requiring it in a SEPARATE key
    would have refused the most carefully-documented corrections in the corpus while waving
    through terse ones that happened to have a sibling.
    """
    stem = str(field).lower()
    inline = _unwrap_principle(new.get(field))
    if isinstance(inline, str):
        body = inline.strip()
        head = body.split("--", 1)[0].strip()
        for sep in _NOTE_SEPARATORS:
            head = head.split(sep, 1)[0].strip()
        # A note is present when the declaration carries prose BEYOND the canonical token.
        # The 12-character floor keeps a bare `value.` or a one-word suffix from counting as
        # a rationale; the origin incidents' values (`cpu_training`) have no trailing prose
        # at all, so they remain refused.
        if len(body) - len(head) >= 12:
            return True
    for k, v in new.items():
        kl = str(k).lower()
        if not v or kl == stem:
            continue
        if stem in kl and any(
            w in kl for w in ("note", "change", "downgrade", "rationale", "cleared")
        ):
            return True
    return False


def _tracked_json_under_results(
    ref: str | None, all_files: bool, staged: bool = False
) -> list[tuple[str, str | None]]:
    """Which artifacts to check, as ``(old_path, new_path)`` pairs.

    ``new_path is None`` means the artifact was DELETED on the new side.

    THE DEFAULT COMPARES AGAINST ``HEAD`` WITHOUT ``--cached``, DELIBERATELY. A first draft used
    ``git diff --cached --name-only`` for the file list while reading the NEW side from the
    working tree -- so an UNSTAGED strip produced an empty file list and the lint printed OK on
    a tree that had just lost a determination. It failed to fire on a faithful replay of its own
    origin incident, which is the one bug a guard cannot have.

    Comparing against ``HEAD`` is correct in both contexts: pre-commit stashes unstaged changes
    before running hooks, so the working tree it sees IS the staged content; and a human running
    this on a dirty tree gets every modification, staged or not.

    ``--diff-filter=MDRT``, NOT ``=M`` (2026-07-29 fix). The pre-fix filter considered MODIFIED
    files only, so the two strictly WORSE ways to lose a determination were invisible:

      * DELETION. Removing ``results/experiment_3946_r11l_first_solve.json`` outright destroys
        every field the lint protects at once, and scored clean. Worse, ``pre_commit``'s own
        ``get_staged_files`` uses ``--diff-filter=ACMRTUXB`` -- "everything except for D" -- so
        a deletion-only commit did not even match this hook's ``files:`` pattern. That second
        layer is closed by ``always_run: true`` in ``.pre-commit-config.yaml``; this is the
        first.
      * RENAME. ``git mv``-ing an artifact and stripping its markers in the same commit showed
        up under neither the old nor the new path.

    ``T`` (type change) is included because a tracked artifact replaced by a symlink loses its
    content just as thoroughly; it is compared like a modification and caught as UNREADABLE.

    ``-M`` turns on rename detection. Note that an UNSTAGED rename is not detectable as one --
    the destination is an untracked file that ``git diff HEAD`` cannot see -- so it presents as a
    plain deletion and is refused as such. That is the conservative and correct reading: a move
    the guard cannot follow is indistinguishable from a removal.

    THE DIFF IS NOT CONFINED TO ``results/``, AND THAT IS THE WHOLE POINT OF RENAME DETECTION.
    A first cut of this fix kept the original ``-- results`` pathspec. Calibrating it against
    real history showed the cost immediately: commit ``bed0635b6`` ("[outer-loop] Retire
    fabricated exp2823 TruthfulQA artifact to legacy/fabricated/") MOVED a flagged artifact --
    ``flagged_adversarial: True`` plus its ``corrigendum_pending`` TAUTOLOGY record -- out of
    ``results/`` and into ``legacy/fabricated/``, alongside a README and an
    ``ops/exclusion_manifest.yaml`` entry. The record was preserved perfectly; that is the
    project's own documented way to retire a fabricated artifact.

    But a pathspec of ``-- results`` makes git report the source half of a cross-directory move
    as a plain DELETION, because the destination is outside the paths being diffed. So the
    pathspec version would have refused a careful, deliberate, fully-documented curation --
    and a guard that refuses honest work gets disabled, which is the same outcome as no guard.

    Diffing the whole tree and filtering on the OLD path instead lets git pair the move up as
    ``R100``, the content compares equal, and the retirement passes. A move that DROPS a marker
    on the way out is still refused, which is the behaviour that was actually wanted.
    """
    if all_files:
        out = _git(["ls-files", "results/*.json"])
        return [(p, p) for p in out.splitlines() if p.endswith(".json")]

    # NOTE: no `-- results` pathspec. See the docstring -- confining the diff makes a move OUT
    # of results/ (the project's documented `legacy/fabricated/` retirement path) look like a
    # deletion. The OLD path is filtered below instead, which keeps the scope identical while
    # letting git pair cross-directory moves up as renames.
    args = ["diff", "--name-status", "-M", "--diff-filter=MDRT"]
    if staged:
        args.append("--cached")
    args += [ref, "HEAD"] if ref else ["HEAD"]

    pairs: list[tuple[str, str | None]] = []
    for line in _git(args).splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") and len(parts) >= 3:
            old_path, new_path = parts[1], parts[2]
        elif len(parts) >= 2:
            old_path = parts[1]
            new_path = None if status.startswith("D") else parts[1]
        else:  # pragma: no cover - git does not emit this shape
            raise GuardError(f"could not parse `git diff --name-status` line: {line!r}")
        # Scope is defined by where the artifact WAS, not where it ended up. An artifact only
        # needs to have been JSON under results/ on the OLD side to be worth protecting --
        # renaming `x.json` to `x.json.bak` is still a loss of the determination it carried.
        if old_path.startswith("results/") and old_path.endswith(".json"):
            pairs.append((old_path, new_path))
    return pairs


def _load_at(rev: str, path: str) -> dict | None:
    """Read a JSON artifact as of `rev`. None when genuinely absent there, or unparseable.

    Used for the OLD side only, where "nothing readable was there" means there is nothing for
    this commit to have destroyed. A git failure that is NOT a plain missing-path is a
    GuardError: the pre-fix code treated every non-zero exit as "absent", so a broken repo or a
    bad ref silently emptied the old side of every comparison and the lint reported OK.
    """
    try:
        r = subprocess.run(
            ["git", "show", f"{rev}:{path}"], capture_output=True, text=True, cwd=REPO
        )
    except OSError as exc:
        raise GuardError(f"could not execute `git show {rev}:{path}`: {exc}") from exc
    if r.returncode != 0:
        if _GIT_PATH_ABSENT.search(r.stderr or ""):
            return None  # routine: the artifact did not exist at that revision
        raise GuardError(
            f"`git show {rev}:{path}` failed with exit {r.returncode}: "
            f"{r.stderr.strip() or '(no stderr)'}"
        )
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def _load_index(path: str) -> dict | _Missing | _Unreadable:
    """Read the STAGED content of `path` (``git show :path``) -- the bytes a commit would land.

    Separate from ``_load_at`` because the new side needs MISSING vs UNREADABLE kept apart,
    which ``_load_at`` deliberately collapses into ``None`` for the old side.
    """
    try:
        r = subprocess.run(["git", "show", f":{path}"], capture_output=True, text=True, cwd=REPO)
    except OSError as exc:
        raise GuardError(f"could not execute `git show :{path}`: {exc}") from exc
    if r.returncode != 0:
        if _GIT_PATH_ABSENT.search(r.stderr or ""):
            return MISSING
        raise GuardError(
            f"`git show :{path}` failed with exit {r.returncode}: "
            f"{r.stderr.strip() or '(no stderr)'}"
        )
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        return UNREADABLE
    return d if isinstance(d, dict) else UNREADABLE


def _load_now(
    path: str | None, ref: str | None, staged: bool = False
) -> dict | _Missing | _Unreadable:
    """Read the NEW side: the working tree (pre-commit) or `HEAD` (auditing a landed commit).

    Returns MISSING (gone) or UNREADABLE (present but not a JSON object) rather than collapsing
    both into ``None``. The pre-fix code returned ``None`` for both and the caller skipped it,
    so overwriting a flagged artifact with truncated or corrupt bytes destroyed its
    determination and scored clean -- the same outcome as deleting it.
    """
    if path is None:
        return MISSING
    if staged:
        return _load_index(path)
    if ref:
        r = _load_at("HEAD", path)
        return r if r is not None else MISSING
    p = REPO / path
    if not p.exists():
        return MISSING
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return UNREADABLE
    except OSError as exc:
        # The file is THERE (p.exists() passed) but cannot be read. That is an environment
        # fault, not a clean tree, so it must not be swallowed.
        raise GuardError(f"could not read {path}: {exc}") from exc
    return d if isinstance(d, dict) else UNREADABLE


def _corrigendum_keys(d: dict) -> set[str]:
    """Corrigendum keys that actually CARRY a record.

    SUBSTANTIVE-ONLY, WHICH IS WHAT CLOSES THE EMPTIED-IN-PLACE BYPASS (2026-07-29, second
    pass). This used to be every key whose name matched, so rule 2's ``old_keys - new_keys``
    compared NAMES and never values -- and a key is still a key when its value is gone:

        "corrigendum_pending": "TAUTOLOGY: ..."   ->   "corrigendum_pending": null

    destroys the record exactly as thoroughly as deleting the key, and scored CLEAN. Confirmed
    against this file as shipped, on the real ``experiment_1680_polarfire_smoke_v2.json``.

    Filtering to substantive values makes an emptied key drop out of the NEW side's set while
    remaining in the OLD side's, so the existing subtraction reports it with no further change.
    Empty-to-empty stays silent because such a key was never in either set.
    """
    return {k for k, v in d.items() if CORRIGENDUM_PREFIX in str(k).lower() and _is_substantive(v)}


def _cleared_deliberately(new: dict) -> bool:
    """A determination may be cleared to False IF an explicit written note accompanies it.

    Absent/None is NOT a legitimate clearing -- it is indistinguishable from the accident this
    lint exists to catch. Requiring a note makes the clearing auditable and forces whoever
    clears it to state their reasoning where the next reader will find it.

    THE EXEMPTION USED TO BE TRIVIALLY SATISFIABLE (2026-07-29 fix, flagged by the Layer-2 QA
    audit). The test was ``any("cleared" in key.lower() and value for key in new)`` -- ANY key
    whose name merely contains the substring "cleared", with ANY truthy value. Censusing the
    corpus shows exactly what that admits: ``cache_cleared: true``, ``step1_vram_cleared: true``,
    ``quota_gate_cleared: true``, ``game_fully_cleared: true`` (an ARC level clear!),
    ``zombie_already_cleared: true``, ``drc_ioplanning_errors_cleared: [...]``. Every one of
    those is real, live, in the tree today, and says nothing whatever about a fabrication
    determination -- yet each would have lifted a quarantine. Not one of the 14 distinct
    ``*cleared*`` key names in the corpus is a prose note, so tightening this cannot regress a
    single existing artifact.

    Two requirements now, both cheap to satisfy on purpose and near-impossible to trip by
    accident:

      1. THE NAME must be note-shaped -- either it contains "note" alongside "cleared", or it
         is prefixed with the determination field it excuses (``flagged_adversarial_cleared*``).
         This is the ``*_cleared_note`` convention the module docstring already documents.
      2. THE VALUE must be a non-empty STRING. A bare ``True`` is not a rationale; the whole
         point of the exemption is that a human wrote down what they re-verified.

    STILL NOT ENOUGH -- THE NOTE MUST NAME THE FIELD IT EXCUSES (2026-07-29, second pass).
    Requirement 1 above still accepted ANY key whose name paired "cleared" with "note",
    regardless of what it was about. Constructed and confirmed against this file as shipped:

        "flagged_adversarial": false,
        "cache_cleared_note": "VRAM cache cleared between runs to avoid OOM"

    scored CLEAN and lifted the quarantine. A GPU-housekeeping remark is not a retraction of a
    fabrication determination, and ``cache_cleared`` is a REAL key in this corpus.

    This is the same defect this file's own sibling ``_has_change_note`` had already diagnosed
    and removed -- its docstring says a field-agnostic note "would silently excuse a substrate
    downgrade ... the 'guard that does not fire' failure mode in miniature" -- so the two
    functions disagreed with each other about a rule they both implement. They now agree: the
    note's NAME must contain the determination field it excuses.

    A corpus census is what makes this safe to tighten: of the 15,284 artifacts in the tree,
    ZERO carry a key satisfying even the looser pre-fix name test, so no existing artifact
    changes verdict. The documented convention (``flagged_adversarial_cleared_note``) satisfies
    the tightened test unchanged.

    The 12-character floor matches ``_has_change_note``'s, for the same reason: a one-word
    string is not a statement of what was re-verified, and consistency between the two
    exemptions is worth more than either threshold individually.
    """
    # Unwrapped: a principle-annotated stamp is not `is True`, so a raw check here would
    # treat a still-flagged artifact as cleared. Same defect as rule 1's, same fix.
    if _unwrap_principle(new.get(DETERMINATION_FIELD)) is True or DETERMINATION_FIELD not in new:
        return False
    for key, value in new.items():
        kl = str(key).lower()
        if "cleared" not in kl:
            continue
        # The note must be ABOUT this determination. "cleared" + "note" in the name is not
        # enough on its own -- see the docstring's `cache_cleared_note` counterexample.
        if DETERMINATION_FIELD not in kl:
            continue
        if "note" not in kl and not kl.startswith(DETERMINATION_FIELD):
            continue
        v = _unwrap_principle(value)
        if isinstance(v, str) and len(v.strip()) >= 12:
            return True
    return False


def _protected_content(d: dict) -> list[str]:
    """The names of fields in `d` that this lint exists to preserve. Empty = nothing at stake.

    Used to decide whether a DELETION or a CORRUPTION is this lint's business. Scoping the
    refusal to artifacts that actually carry a determination, a corrigendum, a marker or a
    substrate declaration keeps the guard inside its charter: it is the determination-
    preservation lint, not a blanket ban on ever removing a file from ``results/``. An artifact
    carrying none of these loses no GATE when it goes -- that is an ordinary never-prune matter
    for human review, not a mechanical refusal this file can justify.
    """
    names: list[str] = []
    for key, value in d.items():
        k = str(key)
        # The live fabrication stamp counts even though a bare `True` is not substantive prose.
        # REDUNDANT, AND KEPT ONLY FOR EXPLICITNESS -- do not rely on it. Measured: the marker
        # branch below appends `flagged_adversarial` for a bare True, a wrapped True AND a bare
        # False (it matches `^flagged_adversarial`, and `_is_substantive` is True for all
        # three), so this branch cannot change the output of this function in any case. Its
        # comment used to justify itself with "a bare `True` is not substantive prose", which
        # is simply false -- `_is_substantive(True)` returns True.
        #
        # It is unwrapped anyway so the file has no raw `is True` check left on a field the
        # project permits to be principle-wrapped; a future edit that narrows the marker
        # patterns must not silently turn this into a live hole. Mutation testing correctly
        # reports this line as unkillable, which is the honest status: a defensive no-op.
        if k == DETERMINATION_FIELD and _unwrap_principle(value) is True:
            names.append(k)
            continue
        if not _is_substantive(value):
            continue
        if CORRIGENDUM_PREFIX in k.lower() or _marker_kind(k) or SUBSTRATE_FIELD.match(k):
            names.append(k)
    return sorted(set(names))


def check(ref: str | None = None, all_files: bool = False) -> list[str]:
    """Every violation this commit would land, from BOTH the working tree and the index.

    WHY BOTH (2026-07-29, second pass). The default path compares ``HEAD`` against the WORKING
    TREE, justified by pre-commit stashing unstaged changes so that the tree the hook sees is
    the staged content. That justification was verified rather than assumed -- a real
    ``git commit`` through a real installed pre-commit hook does stash, the guard does fire, and
    the stripping commit is refused. But it makes the guard's correctness depend on an EXTERNAL
    TOOL's behaviour, and only under one driver:

        strip the fields, `git add` the stripped copy, restore the working tree, then run
        `python3 scripts/determination_preservation_lint.py`

    reports OK, because the working tree matches HEAD. That invocation is the one this file's
    own USAGE section documents for auditing, and it is what any future CI job calling the
    script directly would do.

    The index is not a substitute for the working tree, so this takes the UNION rather than
    switching: the index is what a commit lands (so it must be checked), while the working tree
    is what an unstaged in-progress strip lives in (so it must stay checked, and that is what
    the pre-widening ``--cached`` bug got wrong in the other direction). Under pre-commit the
    two sides are identical and the union is a no-op with one extra git call.
    """
    if ref or all_files:
        return _check_side(ref, all_files, staged=False)
    tree = _check_side(None, False, staged=False)
    seen = set(tree)
    out = list(tree)
    for v in _check_side(None, False, staged=True):
        if v not in seen:
            seen.add(v)
            out.append(
                "[STAGED CONTENT, which is what a commit would land -- it differs from the "
                f"working tree] {v}"
            )
    return out


def _check_side(ref: str | None, all_files: bool, staged: bool) -> list[str]:
    base = ref if ref else "HEAD"
    violations: list[str] = []
    for path, new_path in _tracked_json_under_results(ref, all_files, staged):
        old = _load_at(base, path)
        if old is None:
            continue
        loaded = _load_now(new_path, ref, staged)

        # 0. (2026-07-29 fix) The artifact is GONE, or is no longer readable. Both destroy
        #    every protected field at once, which is strictly worse than editing one of them
        #    out -- and both used to be indistinguishable from "no change" and scored clean.
        if isinstance(loaded, (_Missing, _Unreadable)):
            stakes = _protected_content(old)
            if not stakes:
                continue
            if isinstance(loaded, _Missing):
                what = (
                    f"DELETED (was {path})"
                    if new_path is None
                    else f"MOVED to {new_path}, which does not exist"
                )
            else:
                what = (
                    f"present at {new_path} but NO LONGER READABLE as a JSON object "
                    f"(truncated or corrupt -- possibly an experiment still mid-write; if so "
                    f"wait for it, or `git checkout -- {new_path}`, rather than committing it)"
                )
            violations.append(
                f"{path}: {what}. It carried {stakes}, and destroying the whole artifact "
                f"destroys every one of them at once -- strictly worse than editing one field "
                f"out, and until 2026-07-29 it was the one way to do it that scored clean. "
                f"The evidence trail goes with the file (CLAUDE.md never-prune)."
            )
            continue
        new = loaded

        # A rename is not itself a violation -- but it must not smuggle a field drop through,
        # so the rules below compare old-path@base against new-path@working-tree. `label` keeps
        # the refusal message pointing at BOTH names so the reader can find either one.
        label = path if new_path == path else f"{path} -> {new_path}"

        # 1. The fabrication-gate stamp must not silently vanish.
        #
        # UNWRAPPED ON BOTH SIDES (2026-07-29, second pass). This was an `is True` identity
        # check against the RAW value, so a principle-annotated stamp --
        #     "flagged_adversarial": {"principle": "...", "value": true}
        # -- is not `is True`, the OLD side was never recognised as flagged, and flipping it to
        # a bare `false` WITHOUT any cleared-note scored a clean OK. Confirmed by construction.
        #
        # That is precisely origin bug #2 of CLAUDE.md's QA-Layer Authenticity Discipline
        # (`adversarial_verify.py` reading a wrappable field as a bare string), reproduced in
        # the guard whose own `_unwrap_principle` docstring cites that bug and says "This lint
        # must not repeat it". It did. The wrapper convention is pervasive -- 1,699 wrapped
        # top-level fields across 676 distinct names, including 44 artifacts that wrap
        # `preconditions_checked`, itself a protected marker -- so this is a live shape even
        # though no artifact wraps `flagged_adversarial` today.
        old_det = _unwrap_principle(old.get(DETERMINATION_FIELD))
        new_det = _unwrap_principle(new.get(DETERMINATION_FIELD))
        if old_det is True and new_det is not True:
            if _cleared_deliberately(new):
                pass  # explicit, auditable retraction -- allowed
            else:
                violations.append(
                    f"{label}: {DETERMINATION_FIELD} True -> "
                    f"{new_det!r} with no *_cleared_note. This LIFTS a "
                    f"quarantine: the fabrication gate keys off this field, so dropping it "
                    f"re-admits the artifact to headline aggregation."
                )

        # 2. The corrigendum trail explains WHY a determination exists; losing it strands the
        #    stamp without its evidence, and it is pure history that no re-run supersedes.
        lost = _corrigendum_keys(old) - _corrigendum_keys(new)
        # Split the message by HOW the record was lost. Both are refused identically, but the
        # repair differs: a dropped key is put back, whereas an emptied one usually means a
        # writer overwrote it with a variable that was unset -- and telling a reader to
        # "restore" a key they can plainly still see in the file is how a correct refusal gets
        # dismissed as a false positive.
        gone = sorted(k for k in lost if k not in new)
        emptied_corrigenda = sorted(k for k in lost if k in new)
        if gone:
            violations.append(
                f"{label}: lost corrigendum record(s) {gone}. These document why the "
                f"artifact was flagged; a re-run's fresh numbers do not supersede a review's "
                f"recorded judgement (CLAUDE.md never-prune)."
            )
        if emptied_corrigenda:
            violations.append(
                f"{label}: EMPTIED corrigendum record(s) {emptied_corrigenda} -- the key is "
                f"still present but its content is gone (null / empty), which destroys the "
                f"record as completely as deleting it. These document why the artifact was "
                f"flagged; a re-run's fresh numbers do not supersede a review's recorded "
                f"judgement (CLAUDE.md never-prune)."
            )

        # 3. (2026-07-29 widening) ANY marker field -- a correction, a provenance declaration,
        #    a disclosure, a review note. The origin of this rule is that
        #    `inference_substrate_correction_note` IS a corrigendum in substance but does not
        #    contain the string "corrigendum", so rule 2 above sailed straight past its
        #    deletion. Reported separately from rule 2 so the two stay individually testable.
        lost_markers: dict[str, list[str]] = {}
        emptied_markers: dict[str, list[str]] = {}
        # `already_reported` keeps one deletion from producing two refusal lines. Rule 1 owns
        # `flagged_adversarial` when it was True (that message explains the quarantine-lifting
        # consequence, which is the important part); rule 2 owns the corrigendum family. Rule 3
        # still covers `flagged_adversarial: False` being dropped, which rule 1 ignores.
        already_reported = set(lost)
        if old_det is True:
            already_reported.add(DETERMINATION_FIELD)
        for key, old_value in old.items():
            if key in already_reported:
                continue
            if not _is_substantive(old_value):
                continue
            # PRESENT-BUT-EMPTIED IS A LOSS TOO (2026-07-29, second pass). This condition used
            # to be a bare `key in new`, i.e. a pure NAME check, so setting the value to null /
            # "" / [] kept the key and sailed through while destroying the record just as
            # completely. Constructed against this file as shipped:
            #     "solve_provenance": "live_agent_self_discovery"  ->  ""
            #     "inference_substrate_correction_note": "<hand-written corrigendum>"  ->  ""
            # scored CLEAN. Rule 4 could not catch it either: an emptied value is unrankable,
            # and `_strength_rank` returns None for that, which it treats as "not a downgrade".
            #
            # Calibrated before shipping, not after: across the last 400 commits (1,804
            # artifact-pairs) protected fields were DELETED 130 times and EMPTIED exactly 0
            # times, so this refuses nothing the project has actually ever done while closing
            # the cheaper of the two ways to do the same damage.
            emptied = key in new
            if emptied and _is_substantive(new[key]):
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
                (emptied_markers if emptied else lost_markers).setdefault(kind, []).append(str(key))
        for kind, keys in sorted(lost_markers.items()):
            violations.append(
                f"{label}: lost {kind}(s) {sorted(keys)}. A field whose NAME marks it as a "
                f"review output is not superseded by a re-run's fresh measurements; if it no "
                f"longer applies, say so explicitly beside it (CLAUDE.md never-prune)."
            )
        for kind, keys in sorted(emptied_markers.items()):
            violations.append(
                f"{label}: EMPTIED {kind}(s) {sorted(keys)} -- the key is still there but its "
                f"content is gone (null / empty). That destroys the record as completely as "
                f"deleting it, so it is refused for the same reason: a review output is not "
                f"superseded by a re-run's fresh measurements (CLAUDE.md never-prune). If it "
                f"genuinely no longer applies, replace it with prose saying so."
            )

        # 4. (2026-07-29 widening) A substrate / mode declaration that survives but is
        #    WEAKENED. `live_gpu -> cpu_training` (exp307) retroactively rewrites what hardware
        #    a landed measurement ran on, and it defeats the duration floors that CLAUDE.md's
        #    Inference-Substrate Declaration Discipline applies per-substrate.
        for key, old_value in old.items():
            if not SUBSTRATE_FIELD.match(str(key)) or key not in new:
                continue
            old_rank = _strength_rank(old_value, key)
            new_rank = _strength_rank(new[key], key)
            if old_rank is None or new_rank is None or new_rank >= old_rank:
                continue
            if _has_change_note(new, key):
                continue
            ov = _unwrap_principle(old_value)
            nv = _unwrap_principle(new[key])
            violations.append(
                f"{label}: {key} WEAKENED {ov!r} ({_BAND_NAME[old_rank]}) -> {nv!r} "
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

    # FAIL CLOSED. If the check could not run to completion, the honest report is "I do not
    # know", and for a guard "I do not know" must read as REFUSE. Every one of these paths used
    # to degrade to an empty result and print OK.
    try:
        violations = check(ref=a.ref, all_files=a.all)
    except GuardError as exc:
        print("determination-preservation-lint: REFUSING THE COMMIT (the check could not run).")
        print(f"  {exc}")
        print(
            "\n  This is NOT a clean tree -- it is a guard that was unable to look. Fix the\n"
            "  environment and re-run. If a concurrent process holds the git index, wait for it\n"
            "  to finish rather than bypassing the hook."
        )
        return 1

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
