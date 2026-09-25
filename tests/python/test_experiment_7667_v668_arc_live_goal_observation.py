"""REQ-REPORT-7667: live goal observation and terminal accounting."""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path

import pytest

from carnot import experiment_7667_v668_arc_live_goal_observation as exp
from carnot.agentic.arc_goal_confirmation import shadow_goal_only_decision


def test_selection_is_outcome_blind() -> None:
    """SCENARIO-REPORT-7667-SELECTION: order and outcome data do not select a game."""
    roster = ["aa11", "bb22", "cc33"]
    selected = exp.select_game(roster, salt="fixed")
    assert selected == exp.select_game(reversed(roster), salt="fixed")
    assert selected in roster
    with pytest.raises(ValueError, match="sdk_roster_empty"):
        exp.select_game([])


def test_new_attempt_keeps_old_raw_evidence(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7667-TERMINAL: request IDs belong to one run only."""
    first = exp.raw_for_run(tmp_path, "first")
    second = exp.raw_for_run(tmp_path, "second")
    assert first != second
    assert first.parent == second.parent == tmp_path
    with pytest.raises(ValueError, match="invalid_run_id"):
        exp.raw_for_run(tmp_path, "../escape")


def test_current_invocations_use_attempt_local_ledger(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7667-TERMINAL: inherited telemetry cannot absorb prior calls."""
    from carnot.agentic.arc_inference_boundary import BOUNDARY_LEDGER_ENV

    env = exp.isolated_session_environment({}, gpu_index=0, port=12345, raw_dir=tmp_path)
    assert env[BOUNDARY_LEDGER_ENV] == str(tmp_path / "current_invocation_events.jsonl")


def test_induction_has_one_wall_deadline() -> None:
    """SCENARIO-REPORT-7667-GOAL: several requests share one induction limit."""
    assert exp.bounded_induction_call(lambda: "done", 0.1) == "done"
    with pytest.raises(TimeoutError, match="induction_wall_limit"):
        exp.bounded_induction_call(lambda: time.sleep(0.1), 0.01)
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        assert exp.bounded_induction_call(lambda: "done", 0.1) == "done"
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@pytest.mark.parametrize(
    ("predicted", "status", "old", "guard"),
    [
        (True, "contradiction", "accept_goal", "reject_goal"),
        (True, "confirmed", "accept_goal", "accept_goal"),
        (True, "unknown", "accept_goal", "defer"),
        (False, "confirmed", "no_assertion", "accept_goal"),
    ],
)
def test_shadow_uses_same_sdk_observation(
    predicted: bool, status: str, old: str, guard: str
) -> None:
    """SCENARIO-REPORT-7667-GOAL: only the SDK receipt changes the guard decision."""
    assert shadow_goal_only_decision(predicted, status) == {
        "old_goal_only": old,
        "observed_guard": guard,
    }


def test_reduction_counts_opportunities_without_claiming_solve_rate() -> None:
    """SCENARIO-REPORT-7667-GOAL: zero accepted plans gives insufficient opportunity."""
    schedule = exp.make_schedule("aa11")
    row = {
        **schedule,
        "induction_events": [{"decision": "rejected"}],
        "goal_observations": [],
        "actions": [],
        "request_rows": [],
        "censored": False,
        "exclusion": None,
    }
    reduced = exp.reduce_rows([row], schedule)
    assert reduced["opportunity_counts"]["attempted_inductions"] == 1
    assert reduced["opportunity_counts"]["accepted_inductions"] == 0
    assert reduced["guard_opportunity"] == "insufficient"
    assert reduced["counterfactual_solve_rate_claim"] is False
    with pytest.raises(ValueError, match="one_episode_accounting"):
        exp.reduce_rows([], schedule)


def test_cold_reduction_rejects_forged_counts(tmp_path) -> None:
    """SCENARIO-REPORT-7667-TERMINAL: raw row, not producer summary, controls counts."""
    schedule = exp.make_schedule("aa11")
    row = {
        **schedule,
        "induction_events": [{"decision": "accepted"}],
        "goal_observations": [
            {"predicted_goal": True, "status": "contradiction", "contradiction": True}
        ],
        "actions": [{"action_index": 1}],
        "request_rows": [{"request_dispatched": True}],
        "censored": False,
        "exclusion": None,
    }
    reduced = exp.reduce_rows([row], schedule)
    assert reduced["opportunity_counts"]["confirmed_contradictions"] == 1
    assert reduced["decision_comparisons"][0]["old_goal_only"] == "accept_goal"
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"rows": [row], "schedule": schedule, "reduction": reduced}))
    assert exp.cold_reduce(candidate) == reduced
    row["goal_observations"] = []
    candidate.write_text(json.dumps({"rows": [row], "schedule": schedule, "reduction": reduced}))
    with pytest.raises(ValueError, match="reduction_mismatch"):
        exp.cold_reduce(candidate)


def test_missing_and_present_action_checkpoints(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7667-TERMINAL: only complete JSONL lines are replayed."""
    path = tmp_path / "actions.jsonl"
    assert exp._read_jsonl(path) == []
    path.write_text('{"action_index": 1}\n\n', encoding="utf-8")
    assert exp._read_jsonl(path) == [{"action_index": 1}]


def test_terminal_artifact_distinguishes_blocked_null_and_invalid(capsys) -> None:
    """SCENARIO-REPORT-7667-TERMINAL: scientific outcomes stay separate from bad receipts."""
    exp.progress(time.monotonic(), "test", "boundary")
    assert "phase=test" in capsys.readouterr().out
    schedule = exp.make_schedule("aa11")
    row = {
        **schedule,
        "actions": [{"action_index": 1}],
        "request_rows": [{"request_dispatched": True}],
        "current_output_tokens": 120,
        "induction_events": [],
        "goal_observations": [],
        "censored": False,
        "exclusion": None,
    }
    valid = exp._artifact(
        [], {}, schedule, [row], {"load_attempted": True, "model_loaded": True}, 61
    )
    assert valid["verdict_class"] == "null"
    assert valid["inference_substrate_class"] == "model_full_generation"
    assert valid["live_goal_observation_complete_score"] == 1
    assert valid["opportunity_counts"]["predicted_goals"] == 0
    blocked = exp._artifact(
        [
            exp.gate_check(
                "model_absent",
                upstream="model",
                path="/tmp/missing",
                field="exists",
                operator="eq",
                expected=True,
                observed=False,
            )
        ],
        {},
        schedule,
        [],
        {},
        1,
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["model_specs"] == []
    failed = exp._artifact(
        [], {}, schedule, [], {"load_attempted": True, "error": "load_failed"}, 2
    )
    assert failed["verdict_class"] == "disqualified"
    assert failed["inference_substrate_class"] == "model_load_no_generation"


def test_interrupted_episode_cannot_be_scientific_null() -> None:
    """SCENARIO-REPORT-7667-TERMINAL: an owned episode error disqualifies."""
    schedule = exp.make_schedule("aa11")
    row = {
        **schedule,
        "actions": [{"action_index": 1}],
        "request_rows": [{"request_dispatched": True}],
        "induction_events": [],
        "goal_observations": [],
        "censored": True,
        "exclusion": None,
        "error": "RuntimeError: induction handler crashed",
    }
    artifact = exp._artifact([], {}, schedule, [row], {"load_attempted": True}, 61)
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["live_goal_observation_complete_score"] == 0


def test_cold_reader_checks_block_operands(tmp_path: Path, capsys) -> None:
    """SCENARIO-REPORT-7667-TERMINAL: blocks require exact operands."""
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {"verdict_class": "blocked", "rows": [], "gate_check_summary": {"failed_checks": []}}
        )
    )
    with pytest.raises(ValueError, match="blocked_gate_operands_missing"):
        exp._reader(candidate)
    check = exp.gate_check(
        "missing",
        upstream="input",
        path="/tmp/x",
        field="exists",
        operator="eq",
        expected=True,
        observed=False,
    )
    candidate.write_text(
        json.dumps(
            {
                "verdict_class": "blocked",
                "rows": [],
                "gate_check_summary": {"failed_checks": [check]},
            }
        )
    )
    assert exp._reader(candidate) == 0
    assert "blocked_checks" in capsys.readouterr().out
    schedule = exp.make_schedule("aa11")
    row = {
        **schedule,
        "induction_events": [],
        "goal_observations": [],
        "censored": False,
        "exclusion": None,
    }
    candidate.write_text(
        json.dumps(
            {"rows": [row], "schedule": schedule, "reduction": exp.reduce_rows([row], schedule)}
        )
    )
    assert exp._reader(candidate) == 0


def test_cli_modes_use_reader_and_owned_child(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7667-TERMINAL: CLI modes do not cross into another path."""
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"schedule": exp.make_schedule("aa11")}))
    monkeypatch.setattr(exp, "_reader", lambda path: 7)

    def child(root, schedule, raw):
        assert raw == candidate.parent
        return 8

    monkeypatch.setattr(exp, "_child_measure", child)
    monkeypatch.setattr(exp, "run_experiment", lambda root, date, output: 9)
    assert exp.main(["--cold-reduce", str(candidate)]) == 7
    assert exp.main(["--independent-reduce", str(candidate)]) == 7
    assert exp.main(["--live-child", str(candidate)]) == 8
    assert exp.main(["--date", "20260925"]) == 9
