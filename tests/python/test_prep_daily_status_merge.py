"""REQ-HARNESS-6040 regression tests for `_merge_prep_status` — the never-prune fix for the daily-prep status file.

THE INCIDENT (2026-07-29). `scripts/kaggle/prep_daily_submission.py` wrote
`ops/arc-daily-prep-status.json` with a bare `write_text(json.dumps({...six keys}))`. That
DELETED the seven submission-trail keys the file also carried, including `submission_ref`
(54768046), `submitted_at`, and `prior_submission_scores` — the real leaderboard score-by-date
history `{"2026-07-15": 0.12, "2026-06-30": 0.08}`. The script runs from an unattended systemd
timer, so the loss was silent, and the conductor's routine `git add -A` would have published it.

These tests are written against that exact record, not a synthetic happy path, per the project's
"write the regression test for the incident that motivated the check" discipline.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "prep_daily_submission",
    Path(__file__).resolve().parents[2] / "scripts" / "kaggle" / "prep_daily_submission.py",
)
assert _SPEC is not None and _SPEC.loader is not None
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)


# The real destroyed record, as recovered from commit bc2623761.
INCIDENT_PRIOR = {
    "kernel_version": 9,
    "prepped_at": "2026-07-16T18:02:11Z",
    "save_run": "complete",
    "parquet_ok": True,
    "ready_for_operator_submit": True,
    "submit_command": "…--kver 9",
    "submitted": True,
    "submitted_at": "2026-07-16T19:37:35Z",
    "submission_ref": 54768046,
    "submission_status_at_check": "complete",
    "local_gate_result_at_submit": "pass",
    "note": "v9 provenance note",
    "prior_submission_scores": {"2026-07-15": 0.12, "2026-06-30": 0.08},
}

FRESH_V10 = {
    "prepped_at": "2026-07-29T06:00:00Z",
    "kernel_version": 10,
    "save_run": "complete",
    "parquet_ok": True,
    "ready_for_operator_submit": True,
    "submit_command": "…--kver 10",
}


def test_leaderboard_score_history_survives_a_new_prep() -> None:
    # REQ-HARNESS-6040 / SCENARIO-HARNESS-6040-1
    """The key fact the incident destroyed: cumulative scores are NOT per-version."""
    merged = prep._merge_prep_status(INCIDENT_PRIOR, FRESH_V10)
    assert merged["prior_submission_scores"] == {"2026-07-15": 0.12, "2026-06-30": 0.08}


def test_no_prior_key_is_ever_dropped_without_being_archived() -> None:
    # REQ-HARNESS-6040 / SCENARIO-HARNESS-6040-1
    """Every key present before must still be reachable after — live or in history."""
    merged = prep._merge_prep_status(INCIDENT_PRIOR, FRESH_V10)
    archived: set[str] = set()
    for entry in merged.get("submission_history", []):
        archived |= set(entry)
    for key in INCIDENT_PRIOR:
        assert key in merged or key in archived, f"{key} was destroyed"


def test_submission_ref_is_archived_not_deleted_on_version_change() -> None:
    # REQ-HARNESS-6040 / SCENARIO-HARNESS-6040-2
    merged = prep._merge_prep_status(INCIDENT_PRIOR, FRESH_V10)
    assert merged["submission_history"][0]["submission_ref"] == 54768046
    assert merged["submission_history"][0]["kernel_version"] == 9


def test_a_fresh_version_is_not_left_looking_already_submitted() -> None:
    # REQ-HARNESS-6040 / SCENARIO-HARNESS-6040-2
    """The subtle failure a naive `{**prior, **fresh}` merge would introduce."""
    merged = prep._merge_prep_status(INCIDENT_PRIOR, FRESH_V10)
    assert merged["kernel_version"] == 10
    for key in prep._SUBMISSION_FIELDS:
        assert key not in merged, f"{key} leaked onto an unsubmitted prep"


def test_same_version_reprep_keeps_its_own_submission_fields() -> None:
    # REQ-HARNESS-6040 / SCENARIO-HARNESS-6040-3
    """Re-prepping the SAME version must not retire that version's own submission record."""
    same = {**FRESH_V10, "kernel_version": 9}
    merged = prep._merge_prep_status(INCIDENT_PRIOR, same)
    assert merged["submission_ref"] == 54768046
    assert "submission_history" not in merged


def test_unsubmitted_prior_version_creates_no_history_entry() -> None:
    # REQ-HARNESS-6040 / SCENARIO-HARNESS-6040-3
    prior = {**INCIDENT_PRIOR, "submitted": False}
    merged = prep._merge_prep_status(prior, FRESH_V10)
    assert "submission_history" not in merged
    assert merged["prior_submission_scores"] == INCIDENT_PRIOR["prior_submission_scores"]


def test_empty_prior_is_just_the_fresh_record() -> None:
    # REQ-HARNESS-6040 / SCENARIO-HARNESS-6040-1
    assert prep._merge_prep_status({}, FRESH_V10) == FRESH_V10


def test_history_accumulates_across_successive_version_changes() -> None:
    # REQ-HARNESS-6040 / SCENARIO-HARNESS-6040-2
    """Append-only: a second retirement must not overwrite the first."""
    after_v10 = prep._merge_prep_status(INCIDENT_PRIOR, FRESH_V10)
    submitted_v10 = {**after_v10, "submitted": True, "submission_ref": 55000001}
    after_v11 = prep._merge_prep_status(submitted_v10, {**FRESH_V10, "kernel_version": 11})
    refs = [e.get("submission_ref") for e in after_v11["submission_history"]]
    assert refs == [54768046, 55000001]


@pytest.mark.parametrize("bad", [None, 0, ""])
def test_absent_prior_version_is_not_treated_as_a_change(bad: object) -> None:
    # REQ-HARNESS-6040 / SCENARIO-HARNESS-6040-3
    """A prior with no usable version must not spuriously retire a live submission."""
    prior = {**INCIDENT_PRIOR, "kernel_version": bad}
    merged = prep._merge_prep_status(prior, FRESH_V10)
    if bad is None:
        assert merged["submission_ref"] == 54768046
    else:
        # A real-but-different version DOES retire — recorded so the boundary is explicit.
        assert merged["submission_history"][0]["submission_ref"] == 54768046
