"""REQ-REPORT-8649, REQ-VERIFY-8649: causes are observed, never guessed."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from carnot.agentic import arc_reset_provenance as r


def test_duplicate_conflicting_and_repeated_requests():
    """SCENARIO-REPORT-8649-PARITY: an index cannot silently pick an owner."""
    recorder = r.ResetProvenance()
    recorder.begin(dict(level=0), 10, 2.0)
    recorder.request("explorer.bootstrap")
    recorder.request("induction.plan")
    recorder.selected(("RESET", None))
    assert recorder.receipt()["requester_ownership_rows"][0]["status"] == "conflicting"
    recorder.begin(None)
    recorder.request("explorer.replay")
    recorder.request("explorer.replay")
    assert recorder.receipt()["requester_ownership_rows"][1]["status"] == "duplicate"
    recorder.begin(None)
    recorder.request("explorer.replay")
    rows = recorder.receipt()["reset_request_rows"]
    assert len({row["reset_request_id"] for row in rows}) == 5
    assert rows[-1]["action_index"] == 3
    rows.clear()
    assert len(recorder.receipt()["reset_request_rows"]) == 5


def test_missing_terminal_harness_and_bounded_recording():
    """SCENARIO-REPORT-8649-PARITY: missing origin remains unknown."""
    recorder = r.ResetProvenance(limit=3)
    recorder.begin(None)
    recorder.selected(("RESET", None))
    recorder.returned(("RESET", None), dict(level=0, full_reset=True))
    recorder.begin(dict(level=0))
    recorder.selected((1, None))
    recorder.returned((1, None), dict(level=0, state="GAME_OVER"))
    recorder.begin(None)
    recorder.request("SDK.reset", "harness_reset")
    recorder.returned(("RESET", None), None)
    recorder.begin(None)
    recorder.request("excess")
    recorder.request("excess")
    receipt = recorder.receipt()
    assert receipt["missing_cause_rows"][0]["origin"] == "unknown"
    assert receipt["terminal_reason_rows"][0]["origin"] == "environment_terminal"
    assert receipt["reset_request_rows"][1]["origin"] == "harness_reset"
    assert receipt["issues"] == [dict(reason="record_limit")]
    assert receipt["reset_request_rows"][0]["post_observation_sha256"]
    assert receipt["reset_request_rows"][1]["sdk_status"]["observation_missing"]


def test_disabled_recorder_errors_and_sdk_exception():
    """SCENARIO-REPORT-8649-PARITY: instrumentation never swallows SDK failure."""
    policy = SimpleNamespace()
    r.request(policy, "ignored")
    r.prepare(policy, None)
    assert r.sdk_call(policy, (1, None), lambda: 7) == 7
    policy._arc_reset_provenance = SimpleNamespace(begin=lambda *a: 1 / 0)
    r.prepare(policy, None)
    assert policy._arc_reset_provenance_errors == ["ZeroDivisionError"]
    recorder = r.ResetProvenance()
    policy._arc_reset_provenance = recorder
    recorder.begin(None)
    recorder.request("request")

    def failure():
        raise TimeoutError("original SDK timeout")

    with pytest.raises(TimeoutError, match="original SDK timeout"):
        r.sdk_call(policy, ("RESET", None), failure)
    assert recorder.receipt()["reset_request_rows"][0]["sdk_status"]["error"] == "TimeoutError"


@pytest.mark.parametrize("mechanic", ["navigation", "toggle", "sequence"])
def test_real_scored_wrapper_and_sdk_parity(tmp_path, mechanic):
    """SCENARIO-REPORT-8649-PARITY: real decisions retain both observer windows."""
    from carnot.reporting.arc_reset_live_8649 import fixture_pair

    pair = fixture_pair(tmp_path, mechanic)
    assert pair["passed"], pair
    assert all(pair["equality"].values())
    assert pair["on"]["actual_wrapper"]["policy"] == "E3AgentPolicy"
    assert pair["on"]["numerical_head_calls"] > 0
    assert len(pair["on"]["torch_rng"]) == len(pair["on"]["actions"])
    receipt = pair["on"]["reset_provenance"]
    assert receipt["reset_request_rows"]
    assert all(row["status"] == "unique" for row in receipt["requester_ownership_rows"])
    assert not receipt["missing_cause_rows"]
    assert not receipt["issues"]
    assert pair["on"]["observer_rows"] == pair["off"]["observer_rows"]
    changed = deepcopy(pair["on"])
    changed["sdk_calls"] = []
    from carnot.reporting.arc_reset_live_8649 import parity

    assert not parity(pair["off"], changed)["passed"]


def test_direct_request_sites_and_actual_do_action_request(tmp_path, monkeypatch):
    """REQ-REPORT-8649: direct reset branches and the SDK method share one recorder."""
    import os
    from unittest.mock import patch
    from carnot.agentic import arc_competition_agent as live
    from carnot.agentic.arc_generalization_runtime import withheld_policy_inputs
    from carnot.agentic.arc_live_panel_runtime_8384 import no_models
    from carnot.reporting.arc_observer_fixtures_8624 import frame
    from carnot.agentic.arc_outcome_observer import ArcOutcomeObserver

    class Base:
        def __init__(self, game_id):
            self.game_id = game_id

        def do_action_request(self, action):
            return frame("navigation", 1, "terminal")

    flags = dict(CARNOT_ARC_DISABLE_INDUCTION="1", CARNOT_ARC_TRAJECTORY_SUPERVISOR="0")
    with (
        withheld_policy_inputs(),
        patch.dict(os.environ, flags),
        no_models(),
        patch.object(live, "_load_submitted_frame_change_scorer", return_value=None),
    ):
        agent = live.make_carnot_agent(Base)(game_id="fixture-unknown")
        policy = agent._policy
        recorder = r.ResetProvenance()
        policy._arc_reset_provenance = recorder
        policy.explorer._arc_reset_provenance = recorder
        action = agent.choose_action([], None)
        result = agent.do_action_request(action)
        assert result.levels_completed == 0
        policy.phase = "induce"
        policy.induced = False
        policy._induce_and_plan_timed = lambda: setattr(policy, "plan", [(1, None)])
        agent.choose_action([], None)
        policy.next_move = lambda *args: (None, None)
        agent.choose_action([], None)
        branches = {row["initiating_policy_branch"] for row in recorder.requests}
        assert "E3AgentPolicy.induce.plan_needs_reset" in branches
        assert "CarnotAgent.choose_action.none_fallback" in branches
        assert recorder.requests[0]["post_observation_sha256"]
        legacy = object.__new__(live.CarnotAgentPolicy)
        legacy.explorer = None
        legacy.reset_sent = False
        legacy._arc_reset_provenance = recorder
        assert legacy.next_move([], None) == ("RESET", None)
        assert recorder.requests[-1]["initiating_policy_branch"].startswith("CarnotAgentPolicy")
    observer = ArcOutcomeObserver()
    observer.redirect("a", "fixture", 0, applied=True)
    observer.applied((1, None))
    observer.observe(1, dict(level=0))
    observer.redirect("b", "fixture", 0, applied=True)
    observer.applied((2, None))
    observer.observe(2, dict(level=0))
    observer.finish("reset")
    assert observer.rows()[0]["overlapping_redirect_ids"] == ["b"]
    assert all(row["horizon_status"] == "censored" for row in observer.rows())
    assert r.public_hash(SimpleNamespace(frame=[[1]], levels_completed=0))
