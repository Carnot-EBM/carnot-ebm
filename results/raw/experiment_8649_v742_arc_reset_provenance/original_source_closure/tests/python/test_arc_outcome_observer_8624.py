"""REQ-REPORT-8624: horizon evidence cannot change policy or legacy credit."""

from copy import deepcopy

import pytest

from carnot.agentic.arc_outcome_observer import ArcOutcomeObserver, notify


def observer():
    """Use the live120-action contract, including the application outcome."""
    value = ArcOutcomeObserver()
    value.redirect("r1", "generic_arm", 0, applied=True)
    return value


def step(value, level=0, state="NOT_FINISHED", **extra):
    """Each real application earns at most one distinct next observation."""
    value.applied({"kind": 1, "data": None})
    value.observe(value.action_index, dict(level=level, state=state, **extra))


def test_complete_horizon_is_not_harm_or_legacy_resolution():
    """SCENARIO-REPORT-8624-PARITY:120 observed actions establish only no progress."""
    value = observer()
    for _ in range(120):
        step(value)
    row = value.rows()[0]
    assert row["horizon_status"] == "no_progress_within_horizon"
    assert row["observed_horizon"] == 120
    assert row["causal_attribution"] is None
    assert row["legacy_receipt"] is None
    value.finish("cutoff")
    assert value.rows()[0] == row


@pytest.mark.parametrize("reason", ["cutoff", "action_cap", "terminal", "reset"])
def test_incomplete_horizons_are_censored(reason):
    """REQ-REPORT-8624: closing a short stream cannot establish harm."""
    value = observer()
    step(value)
    value.finish(reason)
    assert value.rows()[0]["horizon_status"] == "censored"
    assert value.rows()[0]["end_reason"] == reason


def test_progress_overlap_and_legacy_are_separate():
    """SCENARIO-REPORT-8624-PARITY: one levelup cannot causally credit two arms."""
    value = observer()
    legacy = {"resolved_by_levelup": False, "actions_to_levelup": None}
    original = deepcopy(legacy)
    step(value)
    value.redirect("r2", "other_arm", 0, applied=True, legacy=legacy)
    step(value, level=1)
    rows = value.rows()
    assert [r["horizon_status"] for r in rows] == ["level_progress_observed"] * 2
    assert rows[0]["overlapping_redirect_ids"] == ["r2"]
    assert rows[1]["overlapping_redirect_ids"] == ["r1"]
    assert rows[1]["legacy_receipt"] == original == legacy
    assert all(r["causal_attribution"] is None for r in rows)
    rows[0]["horizon_status"] = "tampered"
    assert value.rows()[0]["horizon_status"] == "level_progress_observed"


@pytest.mark.parametrize(
    "observation,reason",
    [
        (None, "missing_observation"),
        ({"level": None}, "missing_level"),
        ({"level": 0, "state": "WIN"}, "terminal"),
        ({"level": 0, "full_reset": True}, "reset"),
        ({"level": -1}, "reset"),
    ],
)
def test_missing_reset_and_terminal_censor(observation, reason):
    """REQ-REPORT-8624: unavailable progress stays missing, even on terminal frames."""
    value = observer()
    value.applied({"kind": 1})
    value.observe(1, observation)
    assert value.rows()[0]["end_reason"] == reason


def test_duplicates_gaps_reset_actions_and_invalid_ids():
    """REQ-REPORT-8624: repeated delivery adds no support; holes censor."""
    value = observer()
    step(value)
    value.observe(1, dict(level=0, state="NOT_FINISHED"))
    assert value.rows()[0]["observed_horizon"] == 1
    value.observe(1, dict(level=2))
    assert value.rows()[0]["end_reason"] == "conflicting_duplicate"
    assert len(value.issues) == 2
    with pytest.raises(ValueError, match="duplicate_redirect"):
        value.redirect("r1", "arm", 0, applied=True)
    with pytest.raises(ValueError, match="action_index"):
        value.observe(99, {})
    value = observer()
    value.applied({"kind": 1})
    value.applied({"kind": 2})
    value.observe(2, {"level": 0})
    assert value.rows()[0]["end_reason"] == "missing_observation"
    value = observer()
    step(value)
    value.applied({"kind": "RESET"})
    assert value.rows()[0]["end_reason"] == "reset"


def test_boundary_opt_in_and_observer_failure_preserve_action_path():
    """REQ-REPORT-8624: missing or broken instrumentation cannot select actions."""
    from types import SimpleNamespace

    policy = SimpleNamespace()
    notify(policy, "application", {"kind": 1})
    policy._arc_outcome_observer = observer()
    notify(policy, "application", {"kind": 1})
    notify(policy, "observation", {"level": 0})
    assert policy._arc_outcome_observer.rows()[0]["observed_horizon"] == 1
    notify(policy, "invalid_phase", None)
    assert policy._arc_outcome_observer.issues[-1]["reason"] == "observer_error"


def test_unapplied_selection_and_reset_application_supply_no_support():
    """REQ-REPORT-8624: a selected shadow or reset action cannot open a horizon."""
    value = observer()
    value.redirect("shadow", "arm", 0, applied=False)
    step(value)
    assert value.rows()[1]["end_reason"] == "redirect_not_applied"
    assert value.rows()[0]["overlapping_redirect_ids"] == []
    value = observer()
    value.applied({"kind": "RESET"})
    assert value.rows()[0]["end_reason"] == "reset"
