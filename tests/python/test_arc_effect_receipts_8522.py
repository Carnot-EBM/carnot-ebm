"""REQ-REPORT-8522: direct receipts must preserve reachable policy behavior."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from carnot.agentic import arc_effect_receipts as receipts
from carnot.reporting import arc_effect_fixtures_8522 as fixtures


def test_real_wrapper_controls_and_parity(tmp_path):
    """SCENARIO-REPORT-8522-APPLICATION and SCENARIO-REPORT-8522-PARITY."""
    value = fixtures.collect(tmp_path)
    assert value["action_parity_rows"]
    assert all(row["passed"] for row in value["action_parity_rows"])
    assert all(
        row["solve_provenance"] == "development_proxy" for row in value["mutation_fixture_rows"]
    )
    by_case = {row["case"]: row for row in value["mutation_fixture_rows"]}
    for case in ["drop_bias", "reinduction", "diversity"]:
        event = by_case[case]["receipts"][0]
        assert event["application_disposition"] == "state_changed"
        assert event["before_state_sha256"] != event["after_state_sha256"]
    event = by_case["reinduction"]["receipts"][0]
    assert event["induction_attempt"]["skipped"] == "disabled_by_env"
    assert event["model_action_executed"] is False
    assert by_case["shadow"]["receipts"][0]["application_disposition"] == "selected_shadow"
    assert by_case["ineligible"]["receipts"][0]["selected"] is False
    assert (
        by_case["already_active"]["receipts"][0]["application_disposition"]
        == "already_active_no_op"
    )
    assert by_case["malformed"]["receipts"][0]["application_disposition"] == "state_refused"
    assert by_case["missing"]["receipts"][0]["application_disposition"] == "state_refused"
    assert fixtures.replay(value)
    changed = deepcopy(value)
    changed["mutation_fixture_rows"][0]["receipts"][0]["applied"] = True
    assert not fixtures.replay(changed)


def test_strict_relevant_state_and_effects():
    """REQ-REPORT-8522 refuses unknown fields instead of inventing false state."""
    with pytest.raises(ValueError, match="state"):
        receipts.relevant_state(SimpleNamespace())
    state = dict(
        goal_bias_present=False,
        induced=True,
        hybrid_diversity=True,
        steps_since_progress=151,
        stall_threshold=150,
    )
    assert receipts.action_policy(state) == dict(goal_bias_present=False, diversity_active=True)
    policy = SimpleNamespace(
        induced=True,
        explorer=SimpleNamespace(
            goal_bias=None, _hybrid_diversity=True, _steps_since_progress=151, _stall_threshold=150
        ),
    )
    assert receipts.relevant_state(policy) == state
    policy.explorer._steps_since_progress = "bad"
    with pytest.raises(ValueError, match="state"):
        receipts.relevant_state(policy)
    policy.explorer._steps_since_progress = 0
    policy.induced = "unknown"
    with pytest.raises(ValueError, match="state_boolean"):
        receipts.relevant_state(policy)


def test_disabled_and_observation_refusal(monkeypatch):
    """REQ-REPORT-8522: disabled collection adds no state; missing observations refuse evidence."""
    policy = SimpleNamespace()
    monkeypatch.delenv(receipts.FLAG, raising=False)
    assert receipts.begin(policy, None) is None
    receipts.finish(policy, None, None, [], None, "shadow", 1, None)
    receipts.bind_action(policy, 1, None)
    assert vars(policy) == {}
    monkeypatch.setenv(receipts.FLAG, "1")
    token = receipts.begin(policy, SimpleNamespace())
    policy.short = "private"
    receipts.finish(policy, token, None, [], None, "applied", 1, None)
    assert receipts.read(policy)[0]["application_disposition"] == "state_refused"


def test_non_disabled_attempt_is_not_claimed_as_model_action(monkeypatch):
    """REQ-REPORT-8522: observing an attempt does not prove a model ran."""
    row = dict(
        wrapper_action_index=1,
        induction_attempt_start_index=0,
        arm="allow_reinduction",
        mode="applied",
        induction_attempt=None,
    )
    policy = SimpleNamespace(
        induction_attempts=[dict(skipped="", planned=True)], _trajectory_effect_receipts=[row]
    )
    monkeypatch.setenv(receipts.FLAG, "1")
    receipts.bind_action(policy, 1, None)
    assert row["induction_outcome"] == "attempt_observed"
    assert row["model_action_executed"] is None


def test_fixture_tripwire_failure(monkeypatch):
    """REQ-VERIFY-8522: attempted model access disqualifies private controls."""
    from contextlib import contextmanager

    @contextmanager
    def denied():
        yield ["private attempted load"]

    monkeypatch.setattr(fixtures, "no_models", denied)
    with pytest.raises(ValueError, match="fixture_model_tripwire"):
        fixtures.run("ineligible", True)
