"""Tests for REQ-ARC-WMTE-7511 and its evidence-recovery scenarios."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from carnot import experiment_7511_v657_arc_evidence_recovery as recovery


ROOT = Path(__file__).resolve().parents[2]


def _fixture() -> dict[str, object]:
    return recovery.build_artifact_for_test(ROOT)


def test_preconditions_hash_all_original_evidence() -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-CUSTODY."""

    checks, inventory, sources = recovery.collect_preconditions(ROOT)

    assert checks
    assert all(row["passed"] for row in checks)
    labels = {row["path"] for row in inventory}
    assert recovery.SESSION_PATH.as_posix() in labels
    assert recovery.CANDIDATE_PATH.as_posix() in labels
    assert recovery.SCHEDULE_PATH.as_posix() in labels
    assert recovery.CHECKPOINT_PATH.as_posix() in labels
    assert len([label for label in labels if "/validation/" in label]) == 16
    assert all(str(row["sha256"]).startswith("sha256:") for row in inventory)
    assert sources[recovery.CANDIDATE_PATH.as_posix()]["authoritative"] is False


def test_reducer_reconciles_schedule_calls_actions_and_intervals() -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-REDUCTION."""

    reduced = recovery.reduce_original_evidence(ROOT)

    assert reduced["schedule_matches"] is True
    assert reduced["completed_episode_count"] == 18
    assert reduced["action_count"] == 3_240
    assert reduced["generation_counts"] == {
        "attempted": 36,
        "completed": 36,
        "failed": 0,
        "cancelled": 0,
        "in_flight": 0,
    }
    assert reduced["model_load_counts"]["attempted"] == 1
    assert reduced["all_interval_bounds_valid"] is True
    assert reduced["all_disabled_controls_valid"] is True
    assert len(reduced["per_episode_results"]) == 18
    assert len(reduced["per_game_results"]) == 6


def test_schedule_and_disabled_control_mutations_fail_closed() -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-SCHEDULE."""

    schedule = recovery.load_object(ROOT / recovery.SCHEDULE_PATH)["rows"]
    episodes = recovery.load_object(ROOT / recovery.EPISODE_ROWS_PATH)["rows"]
    changed_schedule = deepcopy(schedule)
    changed_schedule[0]["seed"] = 9
    changed_episode = deepcopy(episodes)
    changed_episode[0]["game_source_disabled"] = False

    assert recovery.reduce_rows(changed_schedule, episodes)["schedule_matches"] is False
    bad = recovery.reduce_rows(schedule, changed_episode)
    assert bad["all_disabled_controls_valid"] is False
    assert "game_source_disabled" in bad["per_episode_results"][0]["failures"]


def test_historical_receipts_are_authenticated_but_not_current() -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-HISTORICAL-RECEIPTS."""

    candidate = recovery.load_object(ROOT / recovery.CANDIDATE_PATH)
    rows = recovery.authenticate_historical_receipts(ROOT, candidate)

    assert len(rows) == 12
    assert all(row["historical_only"] for row in rows)
    assert all(row["authenticated"] for row in rows)
    assert {row["name"] for row in rows} == set(recovery.HISTORICAL_RECEIPT_NAMES)

    changed = deepcopy(candidate)
    changed["validation_receipts"][0]["log_sha256"] = "sha256:" + "0" * 64
    changed_rows = recovery.authenticate_historical_receipts(ROOT, changed)
    assert changed_rows[0]["authenticated"] is False
    assert "log_hash_mismatch" in changed_rows[0]["failures"]


def test_model_runtime_and_process_custody_reconcile() -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-CUSTODY."""

    custody = recovery.authenticate_runtime_custody(ROOT)

    assert custody["qualified"] is True
    assert custody["child_pid"] == 152304
    assert custody["server_pid"] == 152330
    assert custody["gpu_uuid"] == "GPU-b52387a2-c625-de87-8d34-e6f64e684bab"
    assert custody["model"]["hf_id"] == "unsloth/Qwen3.8-27B-GGUF"
    assert custody["model"]["quantization"] == "Q4_K_M"
    assert custody["embedded_tokenizer"] is True


def test_terminal_fixture_separates_current_and_historical_work() -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-NONCLAIM."""

    artifact = _fixture()

    assert artifact["status"] == "complete_null_arc_panel_b_evidence_recovered"
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert not any(artifact["invocation_counts"].values())
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert '"model_loaded"' not in json.dumps(artifact)
    assert artifact["historical_model_specs"][0]["hf_id"] == ("unsloth/Qwen3.8-27B-GGUF")
    assert artifact["historical_duration_s"] == pytest.approx(3284.858228154)
    assert artifact["arc_panel_b_qualified_score"] == 1
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["new_level_credit"] == 0
    assert artifact["remote_submission"] is False


def test_artifact_reader_rejects_current_calls_and_row_mutation() -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-CURRENT-VALIDATION."""

    artifact = _fixture()
    assert recovery.validate_artifact(artifact, ROOT, require_terminal=False) == []

    current_call = deepcopy(artifact)
    current_call["model_invoked"] = True
    assert "current_model_invoked" in recovery.validate_artifact(
        current_call, ROOT, require_terminal=False
    )

    changed_row = deepcopy(artifact)
    changed_row["per_episode_results"][0]["action_count"] -= 1
    assert "independent_reduction_mismatch" in recovery.validate_artifact(
        changed_row, ROOT, require_terminal=False
    )


def test_blocked_and_disqualified_classification_stays_terminal() -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-HISTORICAL-RECEIPTS."""

    blocked = recovery.classify_qualification(
        external_available=False,
        evidence_valid=False,
        current_validation_passed=False,
        benefit_passed=False,
    )
    invalid = recovery.classify_qualification(
        external_available=True,
        evidence_valid=False,
        current_validation_passed=True,
        benefit_passed=True,
    )
    qualified_null = recovery.classify_qualification(
        external_available=True,
        evidence_valid=True,
        current_validation_passed=True,
        benefit_passed=False,
    )

    assert blocked == ("blocked", "complete_blocked_external_evidence_absent", 0)
    assert invalid == ("disqualified", "complete_disqualified_invalid_evidence", 0)
    assert qualified_null == (
        "null",
        "complete_null_arc_panel_b_evidence_recovered",
        1,
    )


def test_validation_plan_is_scoped_and_uses_private_coverage(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-CURRENT-VALIDATION."""

    plan = recovery.build_validation_plan(ROOT, tmp_path)

    assert recovery.validate_validation_plan(ROOT, plan) == []
    assert [row.name for row in plan] == list(recovery.CURRENT_VALIDATION_NAMES)
    assert all(
        argument.rstrip("/") not in {"tests", "tests/python"}
        for row in plan
        for argument in row.argv
    )
    coverage_report = next(row for row in plan if row.name == "changed_module_coverage_report")
    assert "COVERAGE_FILE" in dict(coverage_report.command_environment)


def test_terminal_commands_use_exact_candidate(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-CURRENT-VALIDATION."""

    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}\n", encoding="utf-8")
    commands = recovery.terminal_command_specs(ROOT, candidate)

    assert [row.name for row in commands] == list(recovery.TERMINAL_RECEIPT_NAMES)
    assert all(str(candidate) in row.argv for row in commands)
    assert "--strict" in commands[-1].argv


def test_replay_cli_accepts_exact_temp_candidate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-CURRENT-VALIDATION."""

    artifact = _fixture()
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")

    assert recovery.main(["--replay", str(candidate), "--reduce-only"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["matches_declared"] is True


def test_small_io_progress_and_gate_helpers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-CUSTODY."""

    assert recovery.utc_now().endswith("Z")
    recovery.progress(time.monotonic(), "test", "boundary", units=1)
    assert "phase=test event=boundary" in capsys.readouterr().out

    target = tmp_path / "value.json"
    recovery.atomic_json(target, {"value": 1})
    assert recovery.load_object(target) == {"value": 1}
    target.write_text("[]", encoding="utf-8")
    assert recovery.load_object(target) == {}
    target.write_text("{", encoding="utf-8")
    assert recovery.load_object(target) == {}
    assert recovery.load_object(tmp_path / "missing.json") == {}
    with pytest.raises(ValueError, match="unsupported_gate_operator"):
        recovery._gate(
            "bad",
            "validity",
            1,
            1,
            upstream="fixture",
            field="value",
            principle="Exercise the closed operator set.",
            op="!=",
        )
    span = recovery._phase("test", time.monotonic(), time.monotonic())
    assert span["phase"] == "test"


def test_reducer_names_missing_and_changed_episode_fields() -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-SCHEDULE."""

    schedule = recovery.load_object(ROOT / recovery.SCHEDULE_PATH)["rows"]
    episodes = recovery.load_object(ROOT / recovery.EPISODE_ROWS_PATH)["rows"]
    changed = deepcopy(episodes)
    changed[0]["seed"] = 7
    assert (
        "schedule_seed"
        in recovery.reduce_rows(schedule, changed)["per_episode_results"][0]["failures"]
    )
    missing = recovery.reduce_rows(schedule, episodes[1:])
    assert "episode_row_count" in missing["per_episode_results"][0]["failures"]


def test_historical_receipt_failure_matrix() -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-HISTORICAL-RECEIPTS."""

    candidate = recovery.load_object(ROOT / recovery.CANDIDATE_PATH)
    changed = deepcopy(candidate)
    first = changed["validation_receipts"][0]
    first.update(
        {
            "name": "unexpected",
            "log_path": "/tmp/exp7511-missing-log",
            "exit_code": 2,
            "passed": False,
            "timed_out": True,
            "command": "wrong scope",
        }
    )
    rows = recovery.authenticate_historical_receipts(ROOT, changed)
    failures = rows[0]["failures"]
    assert {
        "unexpected_receipt_name",
        "log_missing",
        "log_hash_mismatch",
        "exit_nonzero",
        "receipt_not_passed",
        "receipt_timed_out",
        "receipt_set_incomplete_or_duplicate",
    }.issubset(failures)

    scoped = deepcopy(candidate)
    scoped["validation_receipts"][0]["command"] = "wrong scope"
    scoped_rows = recovery.authenticate_historical_receipts(ROOT, scoped)
    assert "command_scope_missing_exp7499" in scoped_rows[0]["failures"]


def test_positive_class_and_validation_plan_failures(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-CURRENT-VALIDATION."""

    assert recovery.classify_qualification(
        external_available=True,
        evidence_valid=True,
        current_validation_passed=True,
        benefit_passed=True,
    ) == ("positive", "complete_positive_arc_panel_b_evidence_recovered", 1)

    plan = recovery.build_validation_plan(ROOT, tmp_path)
    broad = replace(plan[1], argv=(*plan[1].argv, "tests/python"))
    errors = recovery.validate_validation_plan(ROOT, [*plan, plan[0], broad])
    assert any(error.startswith("command_count:") for error in errors)
    assert any(error.startswith("broad_test_target:") for error in errors)


def test_artifact_defensive_failure_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-CURRENT-VALIDATION."""

    artifact = _fixture()
    monkeypatch.setattr(
        recovery,
        "independent_reduce",
        lambda _artifact, _root: {"matches_declared": True},
    )

    mutations = {
        "identity_mismatch:schema": ("schema", "wrong"),
        "current_model_specs_nonempty": ("MODEL_SPECS", ["model"]),
        "current_invocation_counts_nonzero": (
            "invocation_counts",
            {**recovery.ZERO_INVOCATION_COUNTS, "model_loads_attempted": 1},
        ),
        "new_live_work_claimed": ("new_game_episodes", 1),
        "raw_candidate_promoted_to_authority": ("raw_candidate_authoritative", True),
        "invalid_verdict_class": ("verdict_class", "unknown"),
        "honest_verdict_not_terminal": ("honest_verdict", "partial_retry"),
    }
    for expected_error, (field, value) in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected_error in recovery.validate_artifact(changed, ROOT, require_terminal=True)

    historical = deepcopy(artifact)
    historical["raw_validation_dispositions"][0]["authenticated"] = False
    assert "historical_validation_not_authenticated" in recovery.validate_artifact(
        historical, ROOT, require_terminal=True
    )
    current = deepcopy(artifact)
    current["validation_receipts"][0]["passed"] = False
    assert "current_scoped_validation_missing_or_failed" in recovery.validate_artifact(
        current, ROOT, require_terminal=True
    )
    terminal = deepcopy(artifact)
    terminal["validation_receipts"][-1]["passed"] = False
    assert "terminal_validation_missing_or_failed" in recovery.validate_artifact(
        terminal, ROOT, require_terminal=True
    )
    principle = deepcopy(artifact)
    principle["field_principles"]["schema"] = ""
    assert "field_principle_missing" in recovery.validate_artifact(
        principle, ROOT, require_terminal=True
    )
    gate = deepcopy(artifact)
    gate["acceptance_gate_results"][0]["principle"] = ""
    assert "gate_principle_missing" in recovery.validate_artifact(gate, ROOT, require_terminal=True)

    original_dumps = recovery.json.dumps
    monkeypatch.setattr(recovery.json, "dumps", lambda *_args, **_kwargs: "x" * (21 * 1024 * 1024))
    assert "artifact_exceeds_20_mib" in recovery.validate_artifact(
        artifact, ROOT, require_terminal=True
    )
    monkeypatch.setattr(recovery.json, "dumps", original_dumps)


def test_replay_validation_cli_and_missing_date(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-CURRENT-VALIDATION."""

    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(_fixture()), encoding="utf-8")
    assert recovery.main(["--replay", str(candidate)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    with pytest.raises(SystemExit, match="--date is required"):
        recovery.main([])
