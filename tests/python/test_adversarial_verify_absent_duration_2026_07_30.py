"""A compute-bound artifact that OMITS duration_s must not pass the duration family in silence.

REQ-HARNESS-6053 (SCENARIO-HARNESS-6053-1 .. SCENARIO-HARNESS-6053-4).

THE INCIDENT (2026-07-30). This session's composite ARC treatment-activation pre-flight artifact
declared ``inference_substrate: live_llm_inference`` -- whose DURATION_TOO_SHORT floor is 60s --
and reported ``duration_s: null``. ``adversarial_verify`` returned clean, 0 flagged, and that
clean result was then cited as evidence the artifact was sound. It was not evidence of anything
about the duration: ``check_duration_vs_claim`` returns immediately when ``duration_s`` is absent
or non-finite, so the entire DURATION_TOO_SHORT family never ran.

Verified directly against the real artifact before writing this test: injecting ``1.0`` or
``0.0001`` fires CRITICAL DURATION_TOO_SHORT, while ``null`` passes clean. That asymmetry is
strictly worse than a short duration -- an artifact that omits the field is INVISIBLE to the
check built to catch fabrication, while an honest one that records a real 35s gets flagged. It is
the same missing-vs-present error the QA-Layer Authenticity Discipline was filed for, in the one
check where CLAUDE.md names the field as THE load-bearing fabrication-detection signal.

THE FIX. ``check_methodology_present`` now lists ``duration_s`` alongside
``model_specs``/``random_seed``/``reproducibility_checksum``, at ``warn`` severity. It is the
right home: that function already gates on exactly the right population (compute-bound marker or
live-LLM substrate) and already carries every legitimate exemption (aggregation-only, ARC no-LLM,
deterministic verifier, precondition-blocked, offline-ARC), so the addition inherits all of them
rather than re-deriving a second, divergent set of carve-outs.

Corpus impact measured before shipping, over all 6311 result artifacts: 656 gain the string
``duration_s`` inside a METHODOLOGY_MISSING warn they ALREADY carried for another reason, and
ZERO artifacts acquire a warn they did not previously have. So the change adds detail to existing
findings and quarantines nothing retroactively.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "adversarial_verify_absent_duration",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "adversarial_verify.py",
)
assert _SPEC is not None and _SPEC.loader is not None
av = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = av
_SPEC.loader.exec_module(av)


def _kinds(flags: list) -> list[str]:
    return [f.kind for f in flags]


def _live_artifact(**over: object) -> dict:
    """A minimally-complete live-LLM artifact -- everything present EXCEPT what a test removes.

    Deliberately carries model_specs, random_seed and reproducibility_checksum so that any
    METHODOLOGY_MISSING raised by these tests can only be about duration_s. Without that, a test
    asserting "the flag fired" would pass for the wrong reason.
    """
    d: dict = {
        "experiment": "fixture_absent_duration",
        "schema": "carnot.fixture.v1",
        "honest_verdict": "complete: fixture",
        "inference_substrate": "live_llm_inference",
        "model_specs": {"generator": "unsloth/gemma-4-31B-it-GGUF"},
        "random_seed": 1,
        "reproducibility_checksum": "sha256:fixture",
        "duration_s": 1200.0,
    }
    d.update(over)
    return d


def test_absent_duration_on_live_llm_artifact_is_flagged() -> None:
    """The incident itself: duration_s missing entirely on a live_llm_inference artifact.

    REQ-HARNESS-6053 / SCENARIO-HARNESS-6053-1.
    """
    art = _live_artifact()
    del art["duration_s"]
    flags: list = []
    av.check_methodology_present(art, flags)
    assert "METHODOLOGY_MISSING" in _kinds(flags)
    assert "duration_s" in flags[0].detail
    # Nothing else may be reported missing -- otherwise this test would pass even if the
    # duration_s branch were deleted.
    assert "model_specs" not in flags[0].detail
    assert "random_seed" not in flags[0].detail


def test_null_duration_is_flagged_the_exact_shape_of_the_real_artifact() -> None:
    """``duration_s: null`` -- literally what the composite pre-flight emitted.

    REQ-HARNESS-6053 / SCENARIO-HARNESS-6053-1.
    """
    flags: list = []
    av.check_methodology_present(_live_artifact(duration_s=None), flags)
    assert "METHODOLOGY_MISSING" in _kinds(flags)
    assert "duration_s" in flags[0].detail


@pytest.mark.parametrize("bad", ["", "1200", float("nan"), float("inf"), [], {}])
def test_non_finite_or_non_numeric_duration_is_flagged(bad: object) -> None:
    """Anything the duration check would skip over must be caught here instead.

    The bug is not "null is missing" -- it is "``check_duration_vs_claim`` skips on any value it
    cannot treat as a finite number". So the methodology check has to cover that whole set, or a
    fabricator need only write ``duration_s: "1200"`` (a string) to stay invisible.

    REQ-HARNESS-6053 / SCENARIO-HARNESS-6053-2.
    """
    flags: list = []
    av.check_methodology_present(_live_artifact(duration_s=bad), flags)
    assert "METHODOLOGY_MISSING" in _kinds(flags)
    assert "duration_s" in flags[0].detail


def test_present_finite_duration_is_not_flagged() -> None:
    """The control. A complete compute-bound artifact stays clean.

    REQ-HARNESS-6053 / SCENARIO-HARNESS-6053-3.
    """
    flags: list = []
    av.check_methodology_present(_live_artifact(), flags)
    assert flags == []


def test_a_short_but_present_duration_is_left_to_the_duration_check() -> None:
    """Absence and implausibility are different findings and must not be conflated.

    A 1.0s live-LLM artifact is a DURATION_TOO_SHORT problem, not a methodology gap -- the field
    is there and was measured. If this test ever fails, the two checks have started
    double-reporting the same artifact and the counts downstream will be wrong.

    REQ-HARNESS-6053 / SCENARIO-HARNESS-6053-3.
    """
    flags: list = []
    av.check_methodology_present(_live_artifact(duration_s=1.0), flags)
    assert flags == []
    dur_flags: list = []
    av.check_duration_vs_claim(_live_artifact(duration_s=1.0), dur_flags)
    assert "DURATION_TOO_SHORT" in _kinds(dur_flags)


def test_aggregation_artifacts_are_still_exempt() -> None:
    """The addition must inherit the existing exemptions, not bypass them.

    An aggregation-only artifact is not a measurement, so requiring a duration from it would be
    the same category error the function already avoids for model_specs and random_seed.

    REQ-HARNESS-6053 / SCENARIO-HARNESS-6053-4.
    """
    art = {
        "experiment": "fixture_aggregation",
        "schema": "carnot.milestone_capstone.v1",
        "honest_verdict": "complete: capstone",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "cited_upstream_artifacts": [{"experiment_id": 1, "fields_imported": ["x"]}],
    }
    flags: list = []
    av.check_methodology_present(art, flags)
    assert "METHODOLOGY_MISSING" not in _kinds(flags)


def test_non_compute_bound_artifacts_are_untouched() -> None:
    """No compute-bound marker and no live substrate -> the whole check is a no-op, as before.

    REQ-HARNESS-6053 / SCENARIO-HARNESS-6053-4.
    """
    art = {
        "experiment": "fixture_plain",
        "schema": "carnot.notes.v1",
        "honest_verdict": "complete: notes",
    }
    flags: list = []
    av.check_methodology_present(art, flags)
    assert flags == []
