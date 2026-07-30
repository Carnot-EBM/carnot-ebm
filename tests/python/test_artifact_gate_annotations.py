"""The checksum's blind spot must stay exactly as wide as the fabrication gate's stamp.

Spec: REQ-ARC-FCP-5610 (the four v506-v509 artifacts whose self-validation this unblocked)

WHY THIS FILE EXISTS (2026-07-30 review of the A1 fix).

``artifact_gate_annotations.checksum_core`` exists because two mechanisms both write to a landed
artifact and meant different things: the experiment stamps a ``reproducibility_checksum`` over its
measurement, and the fabrication gate later ADDS ``flagged_adversarial`` / ``corrigendum_pending``
after re-reading it. Hashing the gate's stamp made every reviewed artifact fail its own validator.

The fix excludes the stamp from the hash. That is the right fix, but it creates a NEW risk that did
not exist before: this module is now the single place that decides what a checksum does NOT protect.
Every key it excludes is a key that can be edited without detection. So the module needs tests that
push in the direction of the *dangerous* error, not the convenient one.

THE DANGEROUS DIRECTION. The first version matched open prefixes ``corrigendum*`` and
``flagged_adversarial*``, borrowed from ``determination_preservation_lint``. For that lint -- whose
job is catching a DROPPED determination -- a broad prefix is the safe error. For a checksum
exclusion -- whose job is deciding what not to hash -- a broad prefix is the dangerous error. The
pattern was imported with its safety direction reversed. 57 keys in this repo carry those prefixes
and many are measured audit tallies: ``corrigendum_pending_count`` (105 occurrences),
``flagged_adversarial_artifacts_excluded`` (32), ``flagged_adversarial_count`` (11). Those are the
numbers an audit artifact exists to report -- precisely what a reader would want protected.

Nothing was actually unprotected: the four adopting artifacts carry only the two real gate keys
(asserted below against the real files). The blind spot was latent, waiting for the first
audit-style experiment to call ``checksum_core``. These tests are what makes it stay closed, and
they are written so that RE-WIDENING the allowlist fails rather than silently passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.artifact_gate_annotations import (
    GATE_ANNOTATION_FIELDS,
    GATE_ANNOTATION_PREFIXES,
    checksum_core,
    is_gate_annotation,
)

REPO = Path(__file__).resolve().parents[2]

# The gate's only write site is ``scripts/adversarial_verify.py:6135-6137``, which sets exactly
# these three. Anything beyond them is not something the gate can add.
GATE_WRITTEN_KEYS = ("flagged_adversarial", "corrigendum_pending", "corrigendum_note")

# Real measured tallies harvested from this repo's own artifacts. Each one begins with a gate-key
# prefix, which is what made the original open-prefix allowlist swallow them, and each is a number
# an audit reports -- the thing a checksum is FOR.
MEASURED_TALLIES_THAT_LOOK_LIKE_ANNOTATIONS = (
    "corrigendum_pending_count",
    "flagged_adversarial_count",
    "flagged_adversarial_artifact_count",
    "flagged_adversarial_artifacts_excluded",
    "flagged_adversarial_this_milestone",
    "corrigendum_kinds",
)


def test_measured_tallies_are_hashed_not_treated_as_annotations() -> None:
    """The regression this file was written for: a tally must not be mistaken for a stamp.

    Asserted two ways deliberately -- via the classifier AND via the checksum payload -- because
    the classifier being right is worthless if ``checksum_core`` does not consult it.
    """
    for key in MEASURED_TALLIES_THAT_LOOK_LIKE_ANNOTATIONS:
        assert not is_gate_annotation(key), (
            f"{key!r} is a MEASURED audit tally, not a review annotation. Classifying it as an "
            "annotation removes it from the reproducibility checksum, so it could then be edited "
            "with no test and no lint noticing. See this module's docstring on why the "
            "determination-lint's broad prefix must not be reused here."
        )

    artifact = {"experiment": 5610, **{k: 7 for k in MEASURED_TALLIES_THAT_LOOK_LIKE_ANNOTATIONS}}
    core = checksum_core(artifact)
    for key in MEASURED_TALLIES_THAT_LOOK_LIKE_ANNOTATIONS:
        assert key in core, f"{key!r} was dropped from the checksum payload"


def test_the_gates_own_three_keys_are_excluded() -> None:
    """The whole reason the module exists: the stamp must not invalidate the measurement."""
    for key in GATE_WRITTEN_KEYS:
        assert is_gate_annotation(key)
    artifact = {
        "experiment": 5610,
        "flagged_adversarial": True,
        "corrigendum_pending": ["TAUTOLOGY: levels_after == levels_before"],
        "corrigendum_note": "stamped by the fabrication gate",
    }
    assert checksum_core(artifact) == {"experiment": 5610}


def test_the_cleared_note_convention_is_excluded_but_nothing_wider() -> None:
    """``flagged_adversarial_cleared*`` is the one documented hand-written annotation
    (``determination_preservation_lint`` requires it when a flag is retired as a false positive).
    It is the ONLY prefix, and this pins that it did not quietly grow back into the open one."""
    assert is_gate_annotation("flagged_adversarial_cleared_note")
    assert GATE_ANNOTATION_PREFIXES == ("flagged_adversarial_cleared",), (
        "the annotation prefix list changed; a broader prefix here silently unhashes measured "
        "fields -- see this module's docstring"
    )
    # The open prefixes the first version used must no longer match on their own.
    assert not is_gate_annotation("corrigendum")
    assert not is_gate_annotation("corrigendum_summary")
    assert not is_gate_annotation("flagged_adversarial_stamped")


def test_a_number_under_an_annotation_name_is_refused_loudly() -> None:
    """The tripwire. If the allowlist is ever widened until it catches a tally, that must fail
    noisily rather than silently unhashing a measurement."""
    with pytest.raises(ValueError, match="numeric value"):
        checksum_core({"experiment": 1, "corrigendum_pending": 12})
    with pytest.raises(ValueError, match="numeric value"):
        checksum_core({"experiment": 1, "flagged_adversarial_cleared_count": 3.5})


def test_the_bool_carve_out_is_present_because_flagged_adversarial_is_a_bool() -> None:
    """``isinstance(True, int)`` is True in Python, so a naive numeric tripwire would reject the
    single most important annotation there is. This pins the carve-out."""
    assert checksum_core({"experiment": 1, "flagged_adversarial": True}) == {"experiment": 1}
    assert checksum_core({"experiment": 1, "flagged_adversarial": False}) == {"experiment": 1}


def test_measured_fields_are_still_protected() -> None:
    """The exclusion must not become a general-purpose escape hatch: every measured field, and the
    ones a fabricated artifact would most want to alter, must still change the payload."""
    base = {
        "experiment": 5610,
        "honest_verdict": "complete_live_self_discovery_levelup",
        "duration_s": 812.4,
        "random_seed": 5610,
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "levels_after": 4,
        "flagged_adversarial": True,
    }
    core = checksum_core(base)
    for field in (
        "honest_verdict",
        "duration_s",
        "random_seed",
        "inference_substrate",
        "levels_after",
    ):
        assert field in core, f"{field} must stay inside the checksum -- it is the measurement"
    assert "flagged_adversarial" not in core
    # ...and a tamper on any of them really does move the payload.
    for field in ("honest_verdict", "duration_s", "random_seed", "levels_after"):
        tampered = dict(base)
        tampered[field] = "TAMPERED" if isinstance(base[field], str) else 999999
        assert checksum_core(tampered) != core, f"tampering with {field} did not change the payload"


def test_checksum_fields_cannot_hash_themselves() -> None:
    """Mechanical, but it is the other half of why this function exists."""
    core = checksum_core(
        {"experiment": 1, "reproducibility_checksum": "sha256:abc", "artifact_checksum": "x"}
    )
    assert core == {"experiment": 1}


@pytest.mark.parametrize(
    ("experiment", "version"),
    [(5610, 506), (5621, 507), (5632, 508), (5643, 509)],
)
def test_the_real_stamped_artifacts_reproduce_their_stored_checksum(
    experiment: int, version: int
) -> None:
    """The A1 proof, captured so it cannot regress.

    This is stronger than "the mismatch went away": recomputing with only the gate keys removed
    reproduces the checksum recorded at AUTHORING time, byte-exactly. That demonstrates the measured
    record was never touched and the stamp was the sole cause of the mismatch -- which is what
    justifies excluding the stamp instead of rewriting the artifact (forbidden: never-prune).

    Reads the committed artifacts; never writes them.
    """
    import importlib

    module = importlib.import_module(
        f"carnot.experiment_{experiment}_arc_live_self_discovery_levelup_v{version}"
    )
    path = REPO / f"results/experiment_{experiment}_arc_live_self_discovery_levelup_v{version}.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))

    stored = artifact["reproducibility_checksum"]
    assert stored, "artifact carries no recorded checksum to reproduce"
    assert module.compute_artifact_checksum(artifact) == stored, (
        "the committed artifact no longer reproduces its own recorded checksum. If a gate "
        "annotation was added, exclude it in artifact_gate_annotations; do NOT edit the artifact."
    )

    # The claim that adoption is safely scoped: these artifacts carry ONLY real gate keys, so the
    # narrowed allowlist cannot be hiding a tally in them.
    suspicious = [
        k
        for k in artifact
        if k.startswith(("flagged_adversarial", "corrigendum")) and k not in GATE_ANNOTATION_FIELDS
    ]
    assert not suspicious, f"unexpected gate-prefixed keys need classifying: {suspicious}"


def test_gate_annotation_fields_matches_what_the_gate_actually_writes() -> None:
    """Pin the allowlist against the gate's real write site, so the two cannot drift.

    If ``adversarial_verify`` starts stamping a fourth key, this fails and whoever adds it has to
    decide -- deliberately -- whether it belongs in the checksum's blind spot.
    """
    verify_src = (REPO / "scripts/adversarial_verify.py").read_text(encoding="utf-8")
    assert 'd["flagged_adversarial"] = True' in verify_src, (
        "the fabrication gate's stamp site moved; re-derive GATE_ANNOTATION_FIELDS from it"
    )
    assert set(GATE_ANNOTATION_FIELDS) == set(GATE_WRITTEN_KEYS), (
        f"allowlist {sorted(GATE_ANNOTATION_FIELDS)} no longer equals the gate's written keys "
        f"{sorted(GATE_WRITTEN_KEYS)}"
    )
