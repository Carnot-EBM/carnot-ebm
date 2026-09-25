"""REQ-REPORT-7653: live ARC paired-case accounting and custody."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_7653_v667_arc_live_generalization as exp


def test_identity_schedule_is_paired_and_outcome_blind() -> None:
    """SCENARIO-REPORT-7653-PAIRED: identities, not outcomes, choose games."""

    roster = ["r11l", "ls20", "ft09", "sc25", "tu93"]
    chosen = exp.select_games(roster, salt="v667-fixed")
    assert len(chosen) == 3
    assert len(set(chosen)) == 3
    assert chosen == exp.select_games(reversed(roster), salt="v667-fixed")
    schedule = exp.build_schedule(chosen, seed=667)
    assert len(schedule) == 6
    for game in chosen:
        pair = [row for row in schedule if row["game"] == game]
        assert [row["arm"] for row in pair] == ["baseline", "hud_dedup"]
        assert {row["seed"] for row in pair} == {667}
        assert {row["max_actions"] for row in pair} == {128}
        assert {row["max_inductions"] for row in pair} == {1}
        assert {row["max_output_tokens"] for row in pair} == {4096}


def test_blocked_gate_has_exact_operands_and_no_current_calls() -> None:
    """SCENARIO-REPORT-7653-BLOCKED: an absent input is terminal blocked."""

    check = exp.gate_check(
        "exclusive_cuda_capacity",
        upstream="current_gpu_inventory",
        path="/tmp/device",
        field="exclusive_device_available",
        operator="eq",
        expected=True,
        observed=False,
    )
    artifact = exp.blocked_artifact([check], source_hashes={}, duration_s=0.5)
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_checks"] == [check]
    assert artifact["MODEL_SPECS"] == []
    assert artifact["planned_MODEL_SPECS"] == [exp.MODEL_ID]
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == {
        "loads": 0,
        "forwards": 0,
        "generations": 0,
        "tokens": 0,
    }
    assert artifact["live_measurement_complete_score"] == 0


def test_cold_reduction_keeps_six_rows_and_reachability_null() -> None:
    """SCENARIO-REPORT-7653-REACHABILITY: absent executed engine has no gain."""

    schedule = exp.build_schedule(["a", "b", "c"], seed=1)
    rows = [
        {
            **unit,
            "start_level": 0,
            "peak_level": 0,
            "actions_used": 2,
            "engine_calls": 0,
            "induction_attempted": index == 0,
            "induction_completed": False,
            "engine_accepted": False,
            "plan_executed": False,
            "censored": index == 1,
            "exclusion": None,
            "raw_provenance": "live_agent_self_discovery",
        }
        for index, unit in enumerate(schedule)
    ]
    reduced = exp.reduce_rows(rows, schedule)
    assert reduced["intended_episodes"] == 6
    assert reduced["observed_episodes"] == 6
    assert reduced["censored_episodes"] == 1
    assert reduced["current_induction_counts"]["attempted"] == 1
    assert reduced["planner_lever_reachable"] is False
    assert reduced["paired_level_deltas"] == {"a": 0, "b": 0, "c": 0}
    rows.pop()
    with pytest.raises(ValueError, match="six_episode_accounting"):
        exp.reduce_rows(rows, schedule)


def test_cold_candidate_rejects_fabricated_reduction(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7653-TERMINAL: raw rows control the headline."""

    schedule = exp.build_schedule(["a", "b", "c"], seed=1)
    rows = [
        {
            **unit,
            "start_level": 0,
            "peak_level": 0,
            "actions_used": 0,
            "engine_calls": 0,
            "induction_attempted": False,
            "induction_completed": False,
            "engine_accepted": False,
            "plan_executed": False,
            "censored": False,
            "exclusion": None,
            "raw_provenance": "live_agent_self_discovery",
        }
        for unit in schedule
    ]
    artifact = {"rows": rows, "schedule": schedule, "reduction": exp.reduce_rows(rows, schedule)}
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.cold_reduce(candidate)["observed_episodes"] == 6
    artifact["reduction"]["observed_episodes"] = 7
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="reduction_mismatch"):
        exp.cold_reduce(candidate)


def test_schedule_and_gate_refuse_invalid_operands() -> None:
    """SCENARIO-REPORT-7653-PAIRED: short rosters and duplicate games fail."""

    with pytest.raises(ValueError, match="three_sdk_games_required"):
        exp.select_games(["a", "a", "b"])
    with pytest.raises(ValueError, match="three_distinct_games_required"):
        exp.build_schedule(["a", "a", "b"])
    assert exp.gate_check(
        "n", upstream="u", path="p", field="f", operator="ge", expected=3, observed=4
    )["passed"]
    assert exp.blocked_artifact([], source_hashes={}, duration_s=0)["honest_verdict"] == (
        "complete_blocked_unknown_input"
    )


def test_reducer_refuses_unpaired_arm() -> None:
    """SCENARIO-REPORT-7653-REACHABILITY: six rows need three exact arm pairs."""

    schedule = exp.build_schedule(["a", "b", "c"], seed=1)
    rows = [{**unit, "peak_level": 0} for unit in schedule]
    rows[1]["arm"] = "unexpected"
    with pytest.raises(ValueError, match="three_paired_games_required"):
        exp.reduce_rows(rows, schedule)


def test_live_artifact_preserves_current_and_planned_classes() -> None:
    """SCENARIO-REPORT-7653-REACHABILITY: calls, readiness and benefit stay separate."""

    schedule = exp.build_schedule(["a", "b", "c"], seed=1)
    rows = [
        {
            **unit,
            "start_level": 0,
            "peak_level": 0,
            "actions_used": 2,
            "engine_calls": 1,
            "induction_attempted": False,
            "induction_completed": False,
            "engine_accepted": False,
            "plan_executed": False,
            "censored": False,
            "exclusion": None,
            "raw_provenance": "live_agent_self_discovery",
            "request_rows": [],
            "current_output_tokens": 0,
        }
        for unit in schedule
    ]
    hashes = {
        "producer_files": {},
        "pre_gate_receipts": {},
        "missing_inputs": [],
        "planned_outputs": [],
    }
    runtime = {
        "load_attempted": True,
        "model_loaded": True,
        "owned_server_vram_mb": 18000,
        "gpu_uuid": "GPU-test",
        "error": None,
    }
    artifact = exp.build_live_artifact([], hashes, schedule, rows, runtime, 2.0)
    assert artifact["verdict_class"] == "null"
    assert artifact["inference_substrate_class"] == "model_load_no_generation"
    assert artifact["live_measurement_complete_score"] == 1
    assert artifact["acceptance_gate_results"]["probability_benefit"]["passed"] is False
    rows[0]["request_rows"] = [{"request_dispatched": True}]
    rows[0]["current_output_tokens"] = 5
    rows[0]["induction_attempted"] = True
    rows[0]["induction_completed"] = True
    rows[0]["engine_accepted"] = True
    rows[0]["plan_executed"] = True
    artifact = exp.build_live_artifact([], hashes, schedule, rows, runtime, 61.0)
    assert artifact["inference_substrate_class"] == "model_full_generation"
    assert artifact["invocation_counts"]["tokens"] == 5
    assert artifact["gate_check_summary"]["reachability"] is True
    incomplete = exp.build_live_artifact([], hashes, schedule, rows[:1], runtime, 1.0)
    assert incomplete["verdict_class"] == "disqualified"
    assert incomplete["live_measurement_complete_score"] == 0
    runtime["load_attempted"] = False
    blocked = exp.build_live_artifact([], hashes, schedule, [], runtime, 1.0)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["MODEL_SPECS"] == []


def test_input_checks_and_frozen_validation_manifest(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7653-TERMINAL: real files and private parents are named."""

    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    assert exp._input_check(tmp_path, Path("source.txt"))["passed"]
    assert not exp._input_check(tmp_path, Path("missing.txt"))["passed"]
    assert exp._hash_input(tmp_path, Path("source.txt"))["bytes"] == 6
    commands = exp.validation_commands(exp.ROOT, tmp_path / "private")
    assert (tmp_path / "private/pytest").is_dir()
    assert [command.name for command in commands][-1] == "full_python_suite"
    assert exp.TEST.as_posix() in commands[1].argv
    terminal = exp.terminal_commands(exp.ROOT, tmp_path / "candidate.json")
    assert [command.name for command in terminal] == [
        "cold_reduction",
        "independent_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]


def test_reader_handles_complete_block_and_rejects_missing_operands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7653-BLOCKED: terminal reader sees exact operands."""

    check = exp.gate_check(
        "model",
        upstream="cache",
        path="/missing",
        field="present",
        operator="eq",
        expected=True,
        observed=False,
    )
    artifact = exp.blocked_artifact([check], source_hashes={}, duration_s=1)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp._reader_main(candidate) == 0
    assert "blocked_checks" in capsys.readouterr().out
    assert exp.main(["--cold-reduce", str(candidate)]) == 0
    artifact["gate_check_summary"]["failed_checks"] = [{}]
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="blocked_gate_operands_missing"):
        exp._reader_main(candidate)


def test_main_cold_mode_and_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7653-TERMINAL: CLI selects cold work without CUDA."""

    schedule = exp.build_schedule(["a", "b", "c"], seed=1)
    rows = [{**unit, "peak_level": 0} for unit in schedule]
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {"rows": rows, "schedule": schedule, "reduction": exp.reduce_rows(rows, schedule)}
        ),
        encoding="utf-8",
    )
    assert exp.main(["--independent-reduce", str(candidate)]) == 0
    assert "observed_episodes" in capsys.readouterr().out
    called: list[tuple[Path, str, Path]] = []
    monkeypatch.setattr(
        exp, "run_experiment", lambda root, date, output: called.append((root, date, output)) or 0
    )
    assert exp.main(["--date", "20260925", "--output", "results/test.json"]) == 0
    assert called[0][0] == exp.ROOT.resolve()
    exp.progress(0, "test", "after", units=1)
    assert "phase=test" in capsys.readouterr().out
