"""Which artifact fields are POST-HOC REVIEW ANNOTATIONS rather than part of the measured record.

WHY THIS DISTINCTION HAS TO EXIST.
Two mechanisms in this project both write to a landed results artifact, and they mean completely
different things:

1. The EXPERIMENT writes its measurement, then stamps a ``reproducibility_checksum`` over it. The
   checksum's job is to catch silent drift -- a corpus or model version changing underneath a
   number, or a later edit quietly altering a result.
2. The FABRICATION GATE (``scripts/adversarial_verify.py``) later re-reads that landed artifact
   and, if it trips a CRITICAL check, ADDS ``flagged_adversarial: True`` plus a
   ``corrigendum_pending`` list and a ``corrigendum_note`` explaining why. This is mandated
   behaviour, documented in CLAUDE.md, and deliberately non-destructive: it only ever adds.

Those two mechanisms collided. The checksum was computed over *every* key except the checksum
fields themselves, so the moment the gate added its stamp, the artifact's own recorded checksum
no longer matched a recomputation -- and the experiment's own validator then rejected its own
committed record. The project's mandated review process was invalidating the very artifacts it
reviewed.

The failure was proven, not inferred: for all four affected artifacts, recomputing the checksum
with ONLY the gate-applied keys removed reproduces the stored value EXACTLY. That is a stronger
result than "the mismatch goes away" -- byte-exact agreement with the checksum recorded at
authoring time demonstrates the measured record is untouched, and that the stamp was the sole
cause.

WHY EXCLUDING THEM IS CORRECT AND NOT A WEAKENING.
The question the checksum exists to answer is "has the MEASUREMENT drifted", not "has this file
been appended to at all". A review annotation is not a measurement -- it is a *statement about*
the measurement, produced by a different process, at a later time, from the artifact's own
contents. Hashing it folds the reviewer's output into the thing being reviewed, which is
circular, and it makes the artifact's integrity check fail for the one reason it should not: that
the artifact was correctly reviewed.

WHAT THIS DOES **NOT** WEAKEN. The excluded set is fixed, small, and enumerated here by name. Any
change to a measured field -- a metric, a duration, a seed, a verdict, a substrate declaration --
still changes the checksum, which is the tamper case the checksum is for. Two independent
protections cover the excluded fields themselves, so they are not unguarded:

  * ``scripts/determination_preservation_lint.py`` refuses any commit that DROPS or silently
    flips ``flagged_adversarial`` or a ``corrigendum*`` field. Loss of a determination is caught
    there, by design, and that lint is the right place for it -- a checksum can only say
    "something changed", whereas that lint knows *which* changes are illegitimate.
  * The gate itself never edits a measured field, so a stamp can only ever add these keys.

DELIBERATELY NARROW -- AND NARROWED AGAIN 2026-07-30 AFTER REVIEW. This is an exact-name allowlist,
not a prefix heuristic. It covers the three keys the fabrication gate actually writes (verified
against its only write site, ``adversarial_verify.py:6135-6137``) plus the one documented
hand-written convention for retiring a flag as a false positive.

The first version of this module matched the open prefixes ``corrigendum*`` and
``flagged_adversarial*``, borrowed from ``determination_preservation_lint.CORRIGENDUM_PREFIX``.
That was wrong, and the reason is worth stating because it is a genuinely easy mistake to repeat:
**THE DIRECTION OF SAFETY IS INVERTED BETWEEN THE TWO USE SITES.**

  * For the LINT, whose job is to catch a determination being DROPPED, a broad prefix is the safe
    error. Matching too much means the lint guards a few extra fields -- harmless.
  * For a CHECKSUM EXCLUSION, whose job is to decide what NOT to hash, a broad prefix is the
    dangerous error. Matching too much means a measured field silently stops being protected.

Borrowing the lint's pattern therefore imported its safety direction backwards. Concretely: 57
distinct keys in this repo begin with one of those two prefixes, and many are unambiguously
MEASURED values, not annotations -- ``corrigendum_pending_count`` (105 occurrences),
``flagged_adversarial_artifacts_excluded`` (32), ``corrigendum_kinds`` (25),
``flagged_adversarial_count`` (11), ``flagged_adversarial_artifact_count`` (3). Those are audit
tallies: exactly the numbers an audit artifact exists to report, and exactly the numbers a reader
would most want a checksum to protect. No artifact in the four adopting modules carries any of
them, so nothing was actually unprotected -- but the blind spot sat waiting for the first
audit-style experiment to call ``checksum_core``, at which point a tally would have gone unhashed
with no test failing.

TWO GUARDS, so this cannot silently regress:

  1. The allowlist is now exact names plus ONE bounded prefix, ``flagged_adversarial_cleared``,
     which exists solely for the ``*_cleared_note`` convention that
     ``determination_preservation_lint`` documents at its own line 813. Every other key with these
     prefixes is hashed.
  2. ``checksum_core`` additionally REFUSES to exclude any allowlisted key holding a numeric value
     (see ``_reject_measurement_shaped``). Every legitimate annotation is a bool, a list of flag
     strings, or a prose string; a number under one of these names means the allowlist has been
     widened to swallow a tally, and the loud failure is the point. Note the bool carve-out --
     ``flagged_adversarial`` is a bool and ``isinstance(True, int)`` is True in Python, so a naive
     numeric check would reject the single most important annotation there is.

If you are tempted to widen this again -- to a prefix, or to something like "any field containing
'note'" -- do not: the broader a checksum's blind spot, the less the checksum means, and an
unrelated ``methodology_note`` genuinely IS part of the measured record.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "CHECKSUM_FIELDS",
    "GATE_ANNOTATION_FIELDS",
    "GATE_ANNOTATION_PREFIXES",
    "checksum_core",
    "is_gate_annotation",
]

# The checksum fields themselves. A checksum cannot hash itself.
CHECKSUM_FIELDS = frozenset({"artifact_checksum", "reproducibility_checksum"})

# Exact keys written by the fabrication gate's stamp path
# (``adversarial_verify.py``'s backfill: ``flagged_adversarial``, ``corrigendum_pending``,
# ``corrigendum_note``).
GATE_ANNOTATION_FIELDS = frozenset(
    {
        "flagged_adversarial",
        "corrigendum_pending",
        "corrigendum_note",
    }
)

# The ONLY prefix, and it is bounded to a single documented convention: when a flag is retired as
# a false positive, ``flagged_adversarial`` is set back to false and a ``*_cleared_note`` records
# why (``determination_preservation_lint`` line 813 requires exactly this pairing). A prefix rather
# than the one exact name because the convention is "prefixed with the determination field it
# excuses", so a second cleared-family field is allowed to appear without a code change.
#
# Deliberately NOT ``corrigendum`` or ``flagged_adversarial``: see the module docstring. Those swallow
# 57 keys here, including audit tallies that must stay hashed.
GATE_ANNOTATION_PREFIXES = ("flagged_adversarial_cleared",)


def is_gate_annotation(key: str) -> bool:
    """True if ``key`` is a post-hoc review annotation rather than part of the measured record.

    Name-only, deliberately: whether a field is an annotation is a property of what wrote it, which
    the name records and the value cannot. The value is checked separately, in ``checksum_core``,
    and only as a tripwire against the allowlist being widened wrongly.
    """
    if key in GATE_ANNOTATION_FIELDS:
        return True
    return any(key.startswith(prefix) for prefix in GATE_ANNOTATION_PREFIXES)


def _reject_measurement_shaped(key: str, value: Any) -> None:
    """Refuse to drop an allowlisted key that is holding a number.

    A tripwire, not a classifier. Every legitimate annotation is a bool (``flagged_adversarial``), a
    list of flag strings (``corrigendum_pending``), or prose (``corrigendum_note``,
    ``*_cleared_note``). A NUMBER under one of these names means someone widened the allowlist until
    it caught a tally like ``flagged_adversarial_count``, and the failure mode of that mistake is
    silent: a measured number stops being hashed and no test notices. So it raises instead.

    ``bool`` is excluded from the check before ``int`` is, because ``isinstance(True, int)`` is True
    in Python and ``flagged_adversarial`` -- the single most important annotation here -- is a bool.
    """
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        raise ValueError(
            f"refusing to exclude {key!r} from the reproducibility checksum: it holds the numeric "
            f"value {value!r}, and a number under a gate-annotation name is a measured tally (e.g. "
            "flagged_adversarial_count, corrigendum_pending_count), not a review annotation. "
            "Excluding it would silently stop protecting a measurement. Narrow "
            "GATE_ANNOTATION_FIELDS / GATE_ANNOTATION_PREFIXES rather than relaxing this check."
        )


def checksum_core(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Return the subset of ``artifact`` a reproducibility checksum should be computed over.

    Drops the checksum fields (a checksum cannot hash itself) and the review annotations the
    fabrication gate may later add. Everything else -- every measurement, seed, duration,
    verdict, and substrate declaration -- is retained and therefore still protected.

    Raises ``ValueError`` if an excluded key holds a number; see ``_reject_measurement_shaped``.
    """
    core: dict[str, Any] = {}
    for key, value in artifact.items():
        if key in CHECKSUM_FIELDS:
            continue
        if is_gate_annotation(key):
            _reject_measurement_shaped(key, value)
            continue
        core[key] = value
    return core
