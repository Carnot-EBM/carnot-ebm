"""After the cutover the substrate class is REQUIRED, whatever the substrate name says.

REQ-SUBSTRATE-CLASS-1, severity ramp, operator-set 2026-09-07 ("cutover now").

Before the cutover the absent-class flag fired only when the substrate NAME matched no
allowlist, so an artifact with a RECOGNISED name was never nudged -- and a recognised name
is exactly where the hole is. Measured 2026-09-07: 254 artifacts sit under the 60s
live-model floor purely because of how their substrate is worded. A cutover that kept the
narrow scope would have changed nothing for the population it exists for.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import adversarial_verify as av  # noqa: E402

# A value the CLASSIFIER recognises (one of CLAUDE.md's legal six).
RECOGNISED = "aggregation_from_upstream_artifacts"
# A value the FLOOR recogniser accepts (leading "deterministic" -> 0.0001s) but the
# CLASSIFIER calls unknown. The two code paths disagree, which is why a name being
# "recognised" is not one fact: exp7091's real substrate has exactly this shape.
FLOORED_BUT_UNCLASSIFIED = "deterministic independent Markdown and YAML contract replay"


def _flags(artifact: dict) -> list[tuple[str, str]]:
    out: list[av.Flag] = []
    av.check_substrate_class(artifact, out)
    return [(f.kind, f.severity) for f in out]


def test_post_cutover_absence_is_critical_even_with_a_recognised_name() -> None:
    got = _flags({"inference_substrate": RECOGNISED, "run_date": "2026-09-07"})
    assert got == [(av.SUBSTRATE_CLASS_MISSING_KIND, "critical")]


def test_pre_cutover_a_recognised_name_still_draws_nothing() -> None:
    # Forward-only. Historical artifacts are neither rewritten nor re-judged.
    assert _flags({"inference_substrate": RECOGNISED, "run_date": "2026-09-06"}) == []


def test_a_string_the_FLOOR_accepts_but_the_classifier_does_not_still_ramps() -> None:
    """The floor recogniser and the classifier are different code paths and disagree.
    `deterministic ...` gets a 0.0001s floor while the classifier calls it unknown, so
    pre-cutover it drew only the narrow warn while its DURATION was already excused.
    Post-cutover it is critical, which is the point of the ramp."""
    pre = _flags({"inference_substrate": FLOORED_BUT_UNCLASSIFIED, "run_date": "2026-09-06"})
    assert pre == [(av.SUBSTRATE_CLASS_MISSING_KIND, "warn")]
    post = _flags({"inference_substrate": FLOORED_BUT_UNCLASSIFIED, "run_date": "2026-09-07"})
    assert post == [(av.SUBSTRATE_CLASS_MISSING_KIND, "critical")]


def test_pre_cutover_an_unrecognised_name_still_only_warns() -> None:
    got = _flags({"inference_substrate": "some entirely novel phrase", "run_date": "2026-09-06"})
    assert got == [(av.SUBSTRATE_CLASS_MISSING_KIND, "warn")]


def test_a_declared_class_post_cutover_is_clean() -> None:
    assert (
        _flags(
            {
                "inference_substrate": RECOGNISED,
                "inference_substrate_class": "no_model_load",
                "run_date": "2026-09-07",
                "duration_s": 0.04,
            }
        )
        == []
    )


def test_the_compact_date_form_is_parsed() -> None:
    """The corpus uses BOTH `20260907` and `2026-09-07`, sometimes in one milestone
    (exp7091 against exp7094). Parsing only the dashed form would treat every
    compact-dated artifact as pre-cutover and the cutover would cover almost nothing."""
    got = _flags({"inference_substrate": RECOGNISED, "run_date": "20260907"})
    assert got == [(av.SUBSTRATE_CLASS_MISSING_KIND, "critical")]
    assert _flags({"inference_substrate": RECOGNISED, "run_date": "20260906"}) == []


def test_a_principle_wrapped_run_date_is_read() -> None:
    # Any field in this project may be {"value": ..., "principle": ...}.
    got = _flags({"inference_substrate": RECOGNISED, "run_date": {"value": "20260907"}})
    assert got == [(av.SUBSTRATE_CLASS_MISSING_KIND, "critical")]


def test_an_undated_artifact_falls_back_to_the_narrow_warn() -> None:
    """A STATED gap: with no parseable run_date the cutover cannot apply, so the
    pre-existing behaviour holds. Dating the artifact is the producer's job."""
    assert _flags({"inference_substrate": RECOGNISED}) == []
    assert _flags({"inference_substrate": "novel phrase", "run_date": "garbage"}) == [
        (av.SUBSTRATE_CLASS_MISSING_KIND, "warn")
    ]
