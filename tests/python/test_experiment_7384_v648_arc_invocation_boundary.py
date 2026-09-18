"""REQ-ARC-WMTE-7384: preserve and diagnose the scored invocation boundary."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from carnot import experiment_7384_v648_arc_invocation_boundary as subject


pytestmark = pytest.mark.memory_watchdog_skip
ROOT = Path(__file__).resolve().parents[2]


def _schedule(count: int = 6) -> list[dict[str, object]]:
    return [
        {"episode_id": f"game:seed-{index}", "game": "game", "seed": index}
        for index in range(count)
    ]


def _completed(index: int) -> dict[str, object]:
    return {
        **_schedule()[index],
        "disposition": "complete",
        "censored": False,
        "action_count": 12,
        "generation_calls_attempted": 2,
        "generation_calls_completed": 2,
    }


def test_timeout_recovery_keeps_completed_started_and_unstarted_units() -> None:
    """SCENARIO-ARC-WMTE-7384-TIMEOUT-RECOVERY: retain exact dispositions."""

    schedule = _schedule()
    reduced = subject.recover_timeout_accounting(
        schedule=schedule,
        terminal_session=None,
        durable_episode_rows=[_completed(index) for index in range(3)],
        progress_rows={
            "game:seed-3": {
                "last_event": "induction_started",
                "actions": 5,
                "induction_in_flight": {"attempt_index": 0},
            }
        },
        timed_out=True,
    )

    assert [row["disposition"] for row in reduced["rows"]] == [
        "complete",
        "complete",
        "complete",
        "started_censored_timeout",
        "unstarted_timeout",
        "unstarted_timeout",
    ]
    assert reduced["accounting"] == {
        "planned_units": 6,
        "attempted_units": 4,
        "completed_units": 3,
        "censored_units": 1,
        "unstarted_units": 2,
    }
    assert reduced["last_confirmed_event"] == "induction_started"
    assert reduced["first_missing_event"] == "child_terminal_session_write"

    errored = subject.recover_timeout_accounting(
        schedule=_schedule(1),
        terminal_session=None,
        durable_episode_rows=[],
        progress_rows={"game:seed-0": {"last_event": "policy_initialization", "error": "first"}},
        timed_out=False,
    )
    assert errored["earliest_exception"] == "first"
    assert errored["rows"][0]["disposition"] == "started_censored_timeout"


def test_terminal_session_wins_and_earliest_error_is_preserved() -> None:
    """REQ-ARC-WMTE-7384: a valid terminal session is authoritative."""

    schedule = _schedule(2)
    first = {**_completed(0), "error": "first failure"}
    second = {**_completed(1), "error": "later failure"}
    reduced = subject.recover_timeout_accounting(
        schedule=schedule,
        terminal_session={"episodes": [first, second], "error": "session failure"},
        durable_episode_rows=[],
        progress_rows={},
        timed_out=False,
    )

    assert reduced["source"] == "terminal_session"
    assert reduced["earliest_exception"] == "first failure"
    assert reduced["accounting"]["completed_units"] == 2
    assert reduced["accounting"]["unstarted_units"] == 0


@pytest.mark.parametrize(
    ("events", "state", "loads", "generations", "earliest"),
    [
        (
            [
                {"operation": "model_load", "call_id": "l", "state": "attempted"},
                {
                    "operation": "model_load",
                    "call_id": "l",
                    "state": "failed",
                    "error": "load failed",
                },
            ],
            "failed_load",
            (1, 0, 1),
            (0, 0, 0),
            "load failed",
        ),
        (
            [
                {"operation": "model_load", "call_id": "l", "state": "attempted"},
                {"operation": "model_load", "call_id": "l", "state": "completed"},
            ],
            "successful_load_no_generation",
            (1, 1, 0),
            (0, 0, 0),
            None,
        ),
        (
            [
                {"operation": "model_load", "call_id": "l", "state": "attempted"},
                {"operation": "model_load", "call_id": "l", "state": "completed"},
                {"operation": "generation", "call_id": "g", "state": "attempted"},
                {
                    "operation": "generation",
                    "call_id": "g",
                    "state": "failed",
                    "error": "generation failed",
                },
            ],
            "attempted_failed_generation",
            (1, 1, 0),
            (1, 0, 1),
            "generation failed",
        ),
        (
            [
                {"operation": "model_load", "call_id": "l", "state": "attempted"},
                {"operation": "model_load", "call_id": "l", "state": "completed"},
                {"operation": "generation", "call_id": "g", "state": "attempted"},
                {
                    "operation": "generation",
                    "call_id": "g",
                    "state": "completed",
                    "usable": True,
                },
            ],
            "successful_generation",
            (1, 1, 0),
            (1, 1, 0),
            None,
        ),
    ],
)
def test_invocation_receipt_states_stay_distinct(
    events: list[dict[str, object]],
    state: str,
    loads: tuple[int, int, int],
    generations: tuple[int, int, int],
    earliest: str | None,
) -> None:
    """SCENARIO-ARC-WMTE-7384-RECEIPT-STATES: do not conflate boundaries."""

    reduced = subject.reduce_invocation_events(events)
    counts = reduced["counts"]
    assert reduced["state"] == state
    assert (
        counts["model_loads_attempted"],
        counts["model_loads_completed"],
        counts["model_loads_failed"],
    ) == loads
    assert (
        counts["generation_calls_attempted"],
        counts["generation_calls_completed"],
        counts["generation_calls_failed"],
    ) == generations
    assert reduced["earliest_exception"] == earliest


def test_invocation_empty_and_inflight_states() -> None:
    """REQ-ARC-WMTE-7384: raw absence and an open load stay distinguishable."""

    assert subject.reduce_invocation_events([])["state"] == "no_invocation"
    in_flight = subject.reduce_invocation_events(
        [{"operation": "model_load", "call_id": "load", "state": "attempted"}]
    )
    assert in_flight["state"] == "load_in_flight_or_cancelled"
    assert in_flight["counts"]["model_loads_in_flight"] == 1


def test_exp7376_raw_bytes_reproduce_the_accounting_failure() -> None:
    """REQ-ARC-WMTE-7384: diagnose the immutable historical producer bytes."""

    diagnosis = subject.diagnose_historical_exp7376(ROOT)

    assert diagnosis["cause_reproduced"] is True
    assert diagnosis["terminal_summary_action_count"] == 0
    assert diagnosis["durable_completed_action_count"] == 624
    assert diagnosis["corrected_accounting"] == {
        "planned_units": 6,
        "attempted_units": 6,
        "completed_units": 5,
        "censored_units": 1,
        "unstarted_units": 0,
    }
    assert diagnosis["last_confirmed_event"] == "induction_started"
    assert diagnosis["first_missing_event"] == "child_terminal_session_write"
    assert diagnosis["historical_invocations"]["counts"]["generation_calls_attempted"] > 0


def test_real_factory_scripted_transport_reaches_a_later_action(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7384-SCORED-REACHABILITY: test the submitted factory."""

    receipt = subject.run_scripted_factory_path(tmp_path)

    assert receipt["factory"] == "make_carnot_agent"
    assert receipt["policy_class"] == "E3AgentPolicy"
    assert receipt["adapter_disabled"] is True
    assert receipt["http_request_count"] == 2
    assert receipt["tool_result_consumed_by_later_request"] is True
    assert receipt["subsequent_scored_policy_action"] is True
    assert receipt["model_invoked"] is False
    assert receipt["solve_credited"] is False


def test_scripted_factory_restores_preexisting_environment(monkeypatch, tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7384: the diagnostic leaves caller environment unchanged."""

    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "caller")
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "caller-disabled")
    assert subject.run_scripted_factory_path(tmp_path)["passed"] is True
    assert os.environ["CARNOT_ARC_INDUCE_THINK"] == "caller"
    assert os.environ["CARNOT_ARC_DISABLE_INDUCTION"] == "caller-disabled"


def test_validation_plan_comes_from_exp7358_and_propagates_coverage(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7384: use the scoped planner with command-local coverage."""

    commands = subject.build_validation_plan(ROOT, tmp_path)

    assert subject.validate_validation_plan(ROOT, commands) == []
    assert [command.name for command in commands] == list(subject.REQUIRED_CHECK_NAMES)
    report = next(
        command for command in commands if command.name == "changed_module_coverage_report"
    )
    environment = dict(getattr(report, "command_environment", ()))
    assert environment["COVERAGE_FILE"].endswith("/.coverage")
    assert Path(environment["COVERAGE_FILE"]).parent.is_dir()


def test_e2e_terminal_specs_and_preconditions_are_exact(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7384: command and producer inputs are authenticated before use."""

    e2e = subject.e2e_command_specs(ROOT, tmp_path / "e2e")
    assert [row.name for row in e2e] == list(subject.REQUIRED_E2E_NAMES)
    assert "--max-actions" in e2e[-1].argv and "12" in e2e[-1].argv
    candidate = tmp_path / "candidate.json"
    terminal = subject.terminal_command_specs(ROOT, candidate)
    assert [row.name for row in terminal] == [
        "independent_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    checks, hashes, historical = subject.collect_preconditions(ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert hashes[subject.HISTORICAL_ARTIFACT.as_posix()]["sha256"].startswith("sha256:")
    assert historical["verdict_class"] == "disqualified"
    assert historical["flagged_adversarial"] is True
    assert historical["authorizes_current_readiness"] is False


def test_artifact_builder_keeps_current_calls_zero_and_historical_labeled() -> None:
    """REQ-ARC-WMTE-7384: completion never promotes historical LLM work."""

    evidence = {
        "cause_reproduced": True,
        "corrected_accounting": {
            "planned_units": 1,
            "attempted_units": 1,
            "completed_units": 1,
            "censored_units": 0,
            "unstarted_units": 0,
        },
        "last_confirmed_event": "first_action",
        "first_missing_event": None,
        "boundary_event_rows": [],
        "root_cause_receipt": {"corrected": True},
        "historical_invocations": {"counts": {"generation_calls_attempted": 1}},
        "rows": [
            {"unit_id": "fixture", "disposition": "complete", "censored": False},
            None,
        ],
    }
    fixture = {
        "factory": "make_carnot_agent",
        "policy_class": "E3AgentPolicy",
        "subsequent_scored_policy_action": True,
        "tool_result_consumed_by_later_request": True,
        "passed": True,
    }
    receipts = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in (*subject.REQUIRED_CHECK_NAMES, *subject.REQUIRED_E2E_NAMES)
    ]
    artifact = subject.build_terminal_artifact(
        run_date="20260918",
        started_at_utc="2026-09-18T00:00:00+00:00",
        ended_at_utc="2026-09-18T00:00:01+00:00",
        duration_s=1.0,
        preconditions=[{"check": "fixture", "passed": True}],
        diagnosis=evidence,
        scripted_path=fixture,
        validation_receipts=receipts,
        source_hashes={"historical.json": {"sha256": "sha256:abc", "historical_only": True}},
        phase_spans=[{"phase": "evaluate", "duration_s": 1.0}],
        terminal_lints_passed=True,
    )

    assert subject.validate_artifact(artifact) == []
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == subject.ZERO_INVOCATION_COUNTS
    assert artifact["arc_invocation_ready_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["historical_model_receipts"]["historical_only"] is True
    assert "counts" not in artifact["historical_model_receipts"]
    assert all("raw_request_manifest" not in row for row in artifact["rows"])
    assert json.dumps(artifact["field_principles"])


def _artifact_fixture() -> dict[str, object]:
    evidence = {
        "cause_reproduced": True,
        "corrected_accounting": {
            "planned_units": 1,
            "attempted_units": 1,
            "completed_units": 1,
            "censored_units": 0,
            "unstarted_units": 0,
        },
        "last_confirmed_event": "first_action",
        "first_missing_event": None,
        "boundary_event_rows": [],
        "root_cause_receipt": {"corrected": True},
        "historical_invocations": {"counts": {}},
        "rows": [{"unit_id": "fixture", "disposition": "complete", "censored": False}],
    }
    fixture = {
        "passed": True,
        "tool_result_consumed_by_later_request": True,
        "subsequent_scored_policy_action": True,
        "http_request_count": 2,
    }
    receipts = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in (*subject.REQUIRED_CHECK_NAMES, *subject.REQUIRED_E2E_NAMES)
    ]
    return subject.build_terminal_artifact(
        run_date=subject.RUN_DATE,
        started_at_utc="2026-09-18T00:00:00+00:00",
        ended_at_utc="2026-09-18T00:00:01+00:00",
        duration_s=1.0,
        preconditions=[{"passed": True}],
        diagnosis=evidence,
        scripted_path=fixture,
        validation_receipts=receipts,
        source_hashes={},
        phase_spans=[],
        terminal_lints_passed=True,
    )


def test_helpers_and_cold_reducer_cover_failure_surfaces(capsys, tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7384: utility failures are explicit and cold reduction is stable."""

    assert subject.utc_now().endswith("+00:00")
    subject.progress(time.monotonic(), "test", "boundary", unit=1)
    assert "phase=test event=boundary" in capsys.readouterr().out
    payload = tmp_path / "payload.json"
    subject.atomic_json(payload, {"a": 1})
    assert subject.load_object(payload) == {"a": 1}
    assert subject.sha256_file(payload).startswith("sha256:")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    assert subject.load_object(malformed) == {}
    assert subject.load_object(tmp_path / "missing.json") == {}
    assert subject.gate_row(
        "floor", "test", 2, 3, upstream="fixture", artifact_field="n", operator=">="
    )["passed"]
    with pytest.raises(ValueError, match="unsupported gate operator"):
        subject.gate_row("bad", "test", 1, 1, upstream="fixture", artifact_field="n", operator="!=")
    phase = subject._phase("test", time.monotonic(), time.monotonic(), 1)
    assert phase["phase"] == "test" and phase["completed_units"] == 1
    assert subject.parse_args(["--date", subject.RUN_DATE]).date == subject.RUN_DATE

    artifact = _artifact_fixture()
    subject.atomic_json(payload, artifact)
    reduced = subject.independent_reduce_file(payload)
    assert reduced["matches_declared"] is True
    assert reduced["accounting_exact"] is True


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("schema", "bad", "schema or experiment identity mismatch"),
        ("milestone", "bad", "milestone or run date mismatch"),
        ("MODEL_SPECS", ["bad"], "current model declaration mismatch"),
        ("invocation_counts", {}, "current invocation counts mismatch"),
        ("inference_substrate_class", "bad", "inference substrate class mismatch"),
        ("execution_venue", "bad", "execution venue mismatch"),
        ("promotion_score", 1, "unsafe promotion or solve credit"),
        ("arc_invocation_ready_score", 0, "readiness reduction mismatch"),
        ("field_principles", {}, "field principles mismatch"),
        ("reproducibility_checksum", "bad", "checksum mismatch"),
    ],
)
def test_artifact_validator_rejects_each_contract_break(
    field: str, bad_value: object, message: str
) -> None:
    """REQ-ARC-WMTE-7384: terminal validation fails closed for schema drift."""

    artifact = _artifact_fixture()
    artifact[field] = bad_value
    assert message in subject.validate_artifact(artifact)


def test_builder_classifies_blocked_and_disqualified() -> None:
    """REQ-ARC-WMTE-7384: missing input blocks; failed own evidence disqualifies."""

    blocked = subject.build_terminal_artifact(
        run_date=subject.RUN_DATE,
        started_at_utc="2026-09-18T00:00:00+00:00",
        ended_at_utc="2026-09-18T00:00:01+00:00",
        duration_s=1.0,
        preconditions=[{"passed": False}],
        diagnosis={"corrected_accounting": {}},
        scripted_path={},
        validation_receipts=[],
        source_hashes={},
        phase_spans=[],
        terminal_lints_passed=False,
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["first_failure"]

    evidence = {
        "cause_reproduced": True,
        "root_cause_receipt": {"corrected": True},
        "corrected_accounting": {
            "planned_units": 1,
            "attempted_units": 1,
            "completed_units": 1,
            "censored_units": 0,
            "unstarted_units": 0,
        },
    }
    disqualified = subject.build_terminal_artifact(
        run_date=subject.RUN_DATE,
        started_at_utc="2026-09-18T00:00:00+00:00",
        ended_at_utc="2026-09-18T00:00:01+00:00",
        duration_s=1.0,
        preconditions=[{"passed": True}],
        diagnosis=evidence,
        scripted_path={"passed": False},
        validation_receipts=[],
        source_hashes={},
        phase_spans=[],
        terminal_lints_passed=True,
    )
    assert disqualified["verdict_class"] == "disqualified"
    disqualified["arc_invocation_ready_score"] = 1
    assert "unsafe readiness on non-ready artifact" in subject.validate_artifact(disqualified)


def test_malformed_historical_receipts_reduce_to_unresolved(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7384: missing external bytes stay unresolved, never invented."""

    raw = tmp_path / subject.HISTORICAL_RAW
    raw.mkdir(parents=True)
    (raw / "receipt_events.jsonl").write_text("not-json\n")
    diagnosis = subject.diagnose_historical_exp7376(tmp_path)
    assert diagnosis["cause_reproduced"] is False
    assert diagnosis["historical_invocations"]["state"] == "no_invocation"
