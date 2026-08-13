"""REQ-OPS-MILESTONE-LEDGER-6262: make the substantive-work share visible.

Spec: REQ-OPS-MILESTONE-LEDGER-6262 /
SCENARIO-OPS-MILESTONE-LEDGER-6262-AMBIGUOUS-COUNTS-AS-SCAFFOLDING

`ops/north-star.md` §1 already states the test -- "a milestone that produces a new version of
an existing artifact without moving the headline is noise" -- and nothing computed it. These
tests pin the classifier, because the whole value of the ledger is that its number is not
flattering. A classifier that drifts generous turns the tool into reassurance.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from milestone_progress_ledger import classify  # noqa: E402


def test_a_real_measurement_is_substantive() -> None:
    assert classify("complete: bounded ASP energy compiler matches clingo oracle") == "substantive"


def test_blocked_beats_every_other_category() -> None:
    # A blocked readiness task is blocked FIRST; order matters in the classifier.
    assert classify("blocked: Exp6262 readiness controls did not all pass") == "blocked"
    assert classify("blocked_gate_check_failed") == "blocked"


def test_an_honest_null_is_not_counted_as_a_measurement() -> None:
    assert classify("complete_gate_not_met_no_reliable_signal") == "null"
    assert classify("complete_rex_gate_not_met_2_of_6_games") == "null"


def test_scaffolding_verdicts_are_recognised() -> None:
    for v in (
        "complete: active_goal_shadow_ready_default_off_no_solve_claim",
        # The FULL verdict, not a truncation. A first version of this test quoted only the
        # first clause and failed, because the marker that catches it ("licens") lives in the
        # part that was cut off -- a reminder to assert against real strings, not remembered ones.
        "complete_positive: family harnesses are frozen before held access; held licenses "
        "are not implied",
        "complete: V549 adversarial capstone reconciled",
    ):
        assert classify(v) == "scaffolding", v


def test_milestone_transition_tasks_are_scaffolding_not_measurements() -> None:
    """Found by spot-check: these read like results and inflated the substantive share.

    "V540 exact states and V541 roadmap contracts validated" moves a milestone forward; it
    measures nothing. Reclassifying them dropped the measured share from 19% to 15%.
    """
    v = "complete: V540 exact states and V541 roadmap contracts validated; broad-suite receipts"
    assert classify(v) == "scaffolding"


def test_an_unrecognised_verdict_is_UNCLASSIFIED_not_substantive() -> None:
    """The fail-safe direction. An unknown verdict must never be counted as a win -- that is
    how a visibility tool quietly becomes a flattering one."""
    assert classify("mysterious outcome nobody anticipated") == "unclassified"


def test_a_missing_verdict_is_its_own_category() -> None:
    assert classify("") == "no_verdict"
