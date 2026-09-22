"""Tests for REQ-HW-7528 and its service-boundary scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7528_v658_service_boundary as exp


def test_missing_prototype_blocks_only_service_branch() -> None:
    """SCENARIO-HW-7528-MISSING-PROTOTYPE keeps the board audit independent."""

    context = exp.collect_preconditions(exp.REPO_ROOT)

    assert context["service_ready"] is False
    assert context["board_ready"] is True
    assert context["service_blocker"] == {
        "check": "count_memory_module_available",
        "upstream": "Exp7523",
        "path": "python/carnot/experiment_7523_v658_count_memory.py",
        "field": "presence",
        "expected": True,
        "observed": False,
    }


def test_board_audit_preserves_three_distinct_scopes() -> None:
    """SCENARIO-HW-7528-BOARD-SCOPES keeps fabric, CPU, and blockers separate."""

    source = exp.load_object(exp.REPO_ROOT / exp.PLACEMENT_PATH)
    changed_state = {
        "exists": False,
        "accepted_receipt_count": 0,
        "latest_receipt_date": "20260823",
        "hardware_operations_issued": [],
    }
    rows, summary = exp.audit_board_rows(source, changed_state)

    assert [row["board"] for row in rows] == ["KV260", "PolarFire", "GateMate"]
    assert rows[0]["exact_claim_scope"] == "historical_kv260_fpga_fabric_sampling_only"
    assert rows[0]["future_access"] == "ssh kria only"
    assert rows[1]["exact_claim_scope"] == (
        "historical_hash_matched_cpu_dispatch_only_no_fpga_sampling"
    )
    assert rows[2]["current_disposition"] == "blocked_unchanged_physical_prerequisite"
    assert summary["board_continuity_complete_score"] == 1
    assert all(row["hardware_operations_issued"] == [] for row in rows)


def test_historical_context_refuses_kernel_substitution() -> None:
    """SCENARIO-HW-7528-NO-SUBSTITUTION leaves whole-service speedup null."""

    placement = exp.load_object(exp.REPO_ROOT / exp.PLACEMENT_PATH)
    trace = exp.load_object(exp.REPO_ROOT / exp.SERVICE_TRACE_PATH)
    context = exp.historical_service_context(placement, trace)

    assert context["exp7514_update_only_ceiling"] == pytest.approx(1.0000525608541164)
    assert context["exp7514_shared_native_forward_calls"] == 188
    assert context["exp7513_timed_parameter_count"] == 81
    assert context["composition_class"] == "hypothetical_bound_only"
    assert context["mismatch_checks"] == {
        "native_call_count_matched": False,
        "operation_context_matched": False,
        "durability_semantics_matched": False,
        "synchronization_semantics_matched": False,
    }
    assert context["target_100x_met"] is False
    assert context["whole_service_speedup"] is None


def test_blocked_fixture_reduces_and_validates() -> None:
    """REQ-HW-7528 requires a valid blocked artifact with a complete board audit."""

    artifact = exp.build_fixture_artifact()
    reduction = exp.independent_reduce(artifact)

    assert reduction["service_cost_complete_score"] == 0
    assert reduction["board_continuity_complete_score"] == 1
    assert reduction["required_validation_passed"] is True
    assert artifact["honest_verdict"] == "complete_blocked_missing_count_memory_prototype"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["whole_service_speedup"] is None
    assert artifact["sample_size_budget"]["batch_256"]["unstarted"] == 30
    assert artifact["sample_size_budget"]["batch_one_ack"]["unstarted"] == 30
    assert exp.validate_artifact(artifact, verify_sources=False) == []


def test_validator_rejects_mutated_boundary_fields() -> None:
    artifact = exp.build_fixture_artifact()

    mutated = deepcopy(artifact)
    mutated["whole_service_speedup"] = 100.0
    assert "whole_service_speedup_must_be_null" in exp.validate_artifact(
        mutated, verify_sources=False
    )

    mutated = deepcopy(artifact)
    mutated["invocation_counts"]["model_loads"]["attempted"] = 1
    assert "current_inference_declaration_invalid" in exp.validate_artifact(
        mutated, verify_sources=False
    )


def test_source_hash_and_checksum_mutations_fail_closed(tmp_path: Path) -> None:
    artifact = exp.build_fixture_artifact()
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    artifact["source_artifact_hashes"] = {
        str(source): exp.source_row(source, tmp_path),
    }
    artifact = exp.finalize_artifact(artifact)
    assert exp.validate_artifact(artifact, root=tmp_path) == []

    source.write_text('{"changed":true}\n', encoding="utf-8")
    assert any(
        error.startswith("source_hash_invalid:")
        for error in exp.validate_artifact(artifact, root=tmp_path)
    )


def test_terminal_command_set_is_exact(tmp_path: Path) -> None:
    commands = exp.terminal_commands(tmp_path / "candidate.json")

    assert [command.spec.name for command in commands] == list(exp.TERMINAL_CHECK_NAMES)
    assert commands[0].spec.argv[2] == exp.WRAPPER_PATH.as_posix()
    assert commands[-1].spec.argv[3:5] == ("--strict", str(tmp_path / "candidate.json"))


def test_cli_readers_reject_bad_date_and_accept_fixture(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--date must be 20260922"):
        exp.run_experiment(exp.REPO_ROOT, "20260921")

    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(exp.build_fixture_artifact()), encoding="utf-8")
    assert (
        exp.main(["--date", exp.RUN_DATE, "--cold-replay", str(candidate), "--no-source-check"])
        == 0
    )
    assert (
        exp.main(
            [
                "--date",
                exp.RUN_DATE,
                "--independent-reduce",
                str(candidate),
                "--no-source-check",
            ]
        )
        == 0
    )


def test_reader_helpers_cover_invalid_and_external_bytes(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    external = tmp_path / "external.json"
    external.write_text("{}\n", encoding="utf-8")

    assert exp.load_object(missing) == {}
    assert exp.load_object(invalid) == {}
    assert exp.load_object(sequence) == {}
    assert exp.source_row(external, tmp_path / "different")["path"] == str(external)


def test_ready_count_producer_is_read_only_after_both_files_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "count_memory.py"
    artifact = tmp_path / "count_memory.json"
    module.write_text("# producer marker\n", encoding="utf-8")
    artifact.write_text('{"count_memory_ready_score":1}\n', encoding="utf-8")
    monkeypatch.setattr(exp, "COUNT_MODULE_PATH", module)
    monkeypatch.setattr(exp, "COUNT_ARTIFACT_PATH", artifact)

    context = exp.collect_preconditions(exp.REPO_ROOT)

    assert context["service_ready"] is True
    assert context["service_blocker"] is None
    assert context["count_artifact"]["count_memory_ready_score"] == 1


def test_finalize_covers_all_terminal_classes() -> None:
    invalid = exp.build_fixture_artifact()
    invalid["validation_receipts"][0]["passed"] = False
    invalid = exp.finalize_artifact(invalid)
    assert invalid["verdict_class"] == "disqualified"
    assert invalid["honest_verdict"] == "complete_disqualified_required_validation_failed"

    incomplete = exp.build_fixture_artifact()
    incomplete["service_blocker"] = None
    incomplete = exp.finalize_artifact(incomplete)
    assert incomplete["honest_verdict"] == "complete_disqualified_service_measurement_incomplete"

    complete = exp.build_fixture_artifact()
    complete["service_blocker"] = None
    complete["service_branch_status"] = "complete_measured"
    complete["restart_parity"] = {"attempted": True, "exact_prediction_parity": True}
    components = {
        "lookup_ns": 1,
        "increment_ns": 1,
        "allocation_ns": 1,
        "serialization_ns": 1,
        "fsync_ns": 1,
        "acknowledgement_ns": 1,
    }
    complete["service_rows"] = [
        {"arm": arm, "unit_id": f"{arm}:{index}", **components}
        for arm in ("batch_256", "batch_one_ack")
        for index in range(30)
    ]
    complete["rows"] = [*complete["service_rows"], *complete["board_rows"]]
    complete = exp.finalize_artifact(complete)
    assert complete["service_cost_complete_score"] == 1
    assert complete["verdict_class"] == "null"


def test_validator_reports_all_defensive_mutations() -> None:
    artifact = exp.build_fixture_artifact()
    artifact["target_100x_met"] = True
    artifact["service_cost_complete_score"] = 9
    artifact["board_continuity_complete_score"] = 9
    artifact["service_blocker"] = {"check": "incomplete"}
    artifact["honest_verdict"] = "bad"
    artifact["verdict_class"] = "unknown"
    artifact["field_principles"] = {}
    artifact["acceptance_gate_results"] = [{}]

    errors = exp.validate_artifact(artifact, verify_sources=False)

    assert "target_100x_must_remain_unmet" in errors
    assert "service_cost_complete_score_mismatch" in errors
    assert "board_continuity_complete_score_mismatch" in errors
    assert "service_blocker_incomplete" in errors
    assert "blocked_verdict_invalid" in errors
    assert "blocked_class_invalid" in errors
    assert "terminal_verdict_prefix_invalid" in errors
    assert "verdict_class_invalid" in errors
    assert "field_principles_incomplete" in errors
    assert "gate_principle_missing" in errors

    broken_rows = exp.build_fixture_artifact()
    broken_rows["board_rows"] = [None]
    assert "independent_reduction_mismatch" in exp.validate_artifact(
        broken_rows, verify_sources=False
    )


def test_validation_plan_and_normal_cli_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = exp.build_validation_commands(tmp_path / "private")
    assert [command.name for command in commands] == list(exp.AFFECTED_CHECK_NAMES)

    called: list[tuple[Path, str, Path]] = []

    def fake_run(root: Path, date: str, *, output_path: Path) -> dict[str, object]:
        called.append((root, date, output_path))
        return {}

    monkeypatch.setattr(exp, "run_experiment", fake_run)
    assert exp.main(["--date", exp.RUN_DATE, "--root", str(exp.REPO_ROOT)]) == 0
    assert called == [(exp.REPO_ROOT, exp.RUN_DATE, exp.RESULT_PATH)]
