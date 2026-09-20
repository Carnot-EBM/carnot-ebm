"""Tests for REQ-REPORT-7445 and SCENARIO-REPORT-7445-*.

The fixtures contain only bounded archived values. They never contact a board
or rerun the sparse/int16 service benchmark.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7445_v652_hardware_envelope as exp


def _board_rows() -> list[dict]:
    return [
        {
            "unit_id": "board:KV260",
            "row_type": "board_disposition",
            "board": "KV260",
            "terminal_state": "graduated_preserved",
            "future_access": "ssh_only",
            "fpga_sampling_claimed": True,
            "hash_matched_cpu_dispatch": False,
            "last_authenticated_path": "results/experiment_7314_v642_board_continuity.json",
            "last_authenticated_hash": "sha256:" + "a" * 64,
            "metric": True,
            "error": None,
        },
        {
            "unit_id": "board:GateMate",
            "row_type": "board_disposition",
            "board": "GateMate",
            "terminal_state": "blocked_changed_physical_state",
            "future_access": None,
            "fpga_sampling_claimed": False,
            "hash_matched_cpu_dispatch": False,
            "last_authenticated_path": "results/raw/gatemate-search.json",
            "last_authenticated_hash": "sha256:" + "b" * 64,
            "metric": False,
            "error": exp.GATEMATE_MISSING_RECEIPT,
        },
        {
            "unit_id": "board:PolarFire",
            "row_type": "board_disposition",
            "board": "PolarFire",
            "terminal_state": "graduated_cpu_dispatch_preserved",
            "future_access": None,
            "fpga_sampling_claimed": False,
            "hash_matched_cpu_dispatch": True,
            "last_authenticated_path": "results/experiment_7314_v642_board_continuity.json",
            "last_authenticated_hash": "sha256:" + "a" * 64,
            "metric": True,
            "error": None,
        },
    ]


def _exp7432() -> dict:
    return {
        "experiment_id": "exp7432-v651-update-placement",
        "milestone": "2026.09.651",
        "status": "complete_null_sparse_update_no_registered_complete_service_benefit",
        "honest_verdict": "complete_null_sparse_update_no_registered_complete_service_benefit",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "amdahl_analysis": {
            "observed_unaccelerated_persistence_fraction": exp.PERSISTENCE_FRACTION,
        },
        "timing_summary": [
            {
                "arm": "float32_sparse",
                "baseline_arm": "float32_dense",
                "batch_size": 1,
                "paired_blocks": 30,
                "whole_service_time_ratio": 0.9937845708690535,
                "ci95_lower": 0.9435061459970505,
                "ci95_upper": 1.0470479326842994,
                "speed_gate_passed": False,
            }
        ],
        "board_rows": _board_rows(),
    }


def _exp7440(*, complete_stages: bool = False) -> dict:
    value = {
        "experiment_id": "exp7440-v652-mixture-learning",
        "milestone": "2026.09.652",
        "status": "complete_null_insufficient_online_benefit",
        "honest_verdict": "complete_null_insufficient_online_benefit",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "hardware_path": {
            "expert_count": 4,
            "adaptive_expert_count": 2,
            "bounded_state": True,
            "prediction_duration_ns": 61_465_757_696,
            "feedback_update_duration_ns": 8_945_762_241,
            "persistence_duration_ns": 748_946_870,
            "read_duration_ns": 14_064_423,
            "feedback_cost_included": True,
            "persistence_cost_included": True,
        },
        "small_ebm_training": {
            "current_training": False,
            "receipt_class": "small_ebm_training",
            "source": "Exp7439 hash-bound checkpoint receipts",
        },
    }
    if complete_stages:
        value["complete_service_stage_rows"] = [
            {"stage": stage, "duration_ns": index + 1, "complete_service": True}
            for index, stage in enumerate(exp.REQUIRED_STAGES)
        ]
    return value


def test_authentication_preserves_original_flags_and_fails_each_source_independently() -> None:
    """REQ-REPORT-7445: identity, class, and flags remain authenticated."""

    checks = exp.authenticate_upstreams(_exp7432(), _exp7440())
    assert all(row["passed"] for row in checks)
    assert {row["upstream"] for row in checks} == {"Exp7432", "Exp7440"}

    broken_7432 = deepcopy(_exp7432())
    broken_7432["flagged_adversarial"] = True
    failed = exp.authenticate_upstreams(broken_7432, _exp7440())
    assert any(
        row["check"] == "exp7432_flagged_adversarial" and not row["passed"] for row in failed
    )
    assert all(row["passed"] for row in failed if row["upstream"] == "Exp7440")


def test_amdahl_ceiling_and_strict_100x_condition_are_bounds() -> None:
    """SCENARIO-REPORT-7445-AMDahl: independently compute 1/f and f<0.01."""

    rows = exp.compute_amdahl_rows(exp.PERSISTENCE_FRACTION)
    ceiling = next(row for row in rows if row["bound"] == "infinite_acceleration_ceiling")
    target = next(row for row in rows if row["bound"] == "one_hundred_x_condition")
    assert ceiling["formula"] == "1 / f"
    assert ceiling["service_speed_limit_x"] == pytest.approx(1 / 0.37696053267914603)
    assert ceiling["is_new_speed_measurement"] is False
    assert target["operator"] == "<"
    assert target["required_unaccelerated_fraction"] == 0.01
    assert target["condition_met"] is False
    with pytest.raises(ValueError, match="persistence_fraction_out_of_range"):
        exp.compute_amdahl_rows(0.0)


def test_incomplete_exp7440_stages_retain_v651_envelope_and_name_missing_costs() -> None:
    """SCENARIO-REPORT-7445-STAGES: aggregates cannot become a decomposition."""

    reduced = exp.reduce_stage_envelope(_exp7432(), _exp7440())
    assert reduced["authority"] == "Exp7432_V651_complete_service_envelope"
    assert reduced["complete_stage_decomposition_available"] is False
    assert reduced["complete_service_timing_rows"] == _exp7432()["timing_summary"]
    assert reduced["unchanged_sparse_or_int16_benchmark_rerun"] is False
    assert reduced["unavailable_stages"] == list(exp.REQUIRED_STAGES)
    assert reduced["expert_accounting"]["expert_count"] == 4
    assert reduced["expert_accounting"]["four_expert_prediction_cost_ns"] == 61_465_757_696
    assert reduced["expert_accounting"]["weight_normalization_cost_ns"] is None
    assert "not_isolated" in reduced["expert_accounting"]["weight_normalization_status"]


def test_complete_stage_rows_are_accepted_only_as_a_full_exact_set() -> None:
    """REQ-REPORT-7445: all five complete-service stages are required."""

    complete = exp.reduce_stage_envelope(_exp7432(), _exp7440(complete_stages=True))
    assert complete["authority"] == "Exp7440_complete_service_stage_rows"
    assert complete["complete_stage_decomposition_available"] is True
    assert complete["unavailable_stages"] == []

    duplicate = _exp7440(complete_stages=True)
    duplicate["complete_service_stage_rows"][1]["stage"] = exp.REQUIRED_STAGES[0]
    incomplete = exp.reduce_stage_envelope(_exp7432(), duplicate)
    assert incomplete["complete_stage_decomposition_available"] is False

    wrong_shape = _exp7440()
    wrong_shape["complete_service_stage_rows"] = [
        *[
            {"stage": stage, "duration_ns": 1, "complete_service": True}
            for stage in exp.REQUIRED_STAGES[:-1]
        ],
        "not-a-row",
    ]
    assert (
        exp.reduce_stage_envelope(_exp7432(), wrong_shape)["complete_stage_decomposition_available"]
        is False
    )


def test_board_rows_preserve_graduations_and_block_gatemate_without_probe() -> None:
    """SCENARIO-REPORT-7445-BOARDS: board dispositions remain independent."""

    changed_state = {
        "exists": False,
        "accepted_receipt_count": 0,
        "search_receipt_path": "results/raw/exp7445/gatemate.json",
        "search_receipt_hash": "sha256:" + "c" * 64,
        "eligibility_contract": {"receipt_date": ">20260823"},
    }
    rows = exp.build_board_rows(_board_rows(), changed_state)
    assert [row["board"] for row in rows] == ["KV260", "GateMate", "PolarFire"]
    kv260, gate, polarfire = rows
    assert kv260["future_access"] == "ssh_only"
    assert kv260["terminal_state"] == "graduated_preserved"
    assert gate["terminal_state"] == "blocked_changed_physical_state"
    assert gate["gate_check_summary"]["observed"] == 0
    assert gate["hardware_operations_issued"] == []
    assert polarfire["terminal_state"] == "graduated_cpu_dispatch_preserved"
    assert polarfire["fpga_sampling_claimed"] is False
    assert all(row["hardware_ready_score"] == 0 for row in rows)
    assert all(row["hardware_value_score"] == 0 for row in rows)


def test_changed_gatemate_receipt_only_marks_future_eligibility() -> None:
    """REQ-REPORT-7445: a changed state is not a hardware execution."""

    changed = {
        "exists": True,
        "accepted_receipt_count": 1,
        "search_receipt_path": "receipt.json",
        "search_receipt_hash": "sha256:" + "d" * 64,
        "eligibility_contract": {"receipt_date": ">20260823"},
    }
    gate = exp.build_board_rows(_board_rows(), changed)[1]
    assert gate["terminal_state"] == "changed_state_future_task_eligible"
    assert gate["new_hardware_execution_claimed"] is False
    assert gate["hardware_ready_score"] == 0


def test_missing_board_identity_fails_closed() -> None:
    """REQ-REPORT-7445: no board row may be fabricated."""

    with pytest.raises(ValueError, match="board_rows_invalid"):
        exp.build_board_rows(_board_rows()[:-1], {"exists": False})


def test_future_options_name_exact_access_or_bottleneck() -> None:
    """SCENARIO-REPORT-7445-FUTURE: external options remain unqualified."""

    rows = exp.external_future_rows()
    assert [row["option"] for row in rows] == [
        "Extropic Z1T/TSU",
        "photonic",
        "D-Wave",
        "NPU",
        "larger FPGA",
    ]
    assert all(row["status"] == "external_future_work" for row in rows)
    assert all(row["justification_required"] for row in rows)
    assert all(row["hardware_value_score"] == 0 for row in rows)


def test_learning_route_keeps_durable_semantics_on_cpu() -> None:
    """REQ-REPORT-7445: accelerable numeric state excludes durable service claims."""

    route = exp.learning_hardware_route(exp.PERSISTENCE_FRACTION)
    assert route["accelerator_candidates"] == ["GPU/NPU", "FPGA"]
    assert "log_weight_updates" in route["accelerator_scope"]
    assert "durable_state" in route["cpu_scope"]
    assert route["measured_residual_service_bottleneck_fraction"] == exp.PERSISTENCE_FRACTION
    assert route["requires_changed_persistence_orchestration_design_for_100x"] is True
    assert route["equivalent_acknowledgement_and_crash_semantics_required"] is True
    assert route["durability_trade_allowed"] is False


def test_fixture_artifact_validates_and_reduces_independently(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7445-ARTIFACT: raw rows reproduce the null envelope."""

    artifact = exp.build_fixture_artifact()
    assert exp.validate_artifact(artifact, root=tmp_path, require_terminal=False) == []
    reduction = exp.independent_reduce(artifact)
    assert reduction["matches_declared"] is True
    assert reduction["amdahl_ceiling_x"] == pytest.approx(1 / exp.PERSISTENCE_FRACTION)
    assert reduction["board_names"] == ["GateMate", "KV260", "PolarFire"]
    assert artifact["honest_verdict"].startswith("complete_null")
    assert artifact["hardware_ready_score"] == 0
    assert artifact["hardware_value_score"] == 0
    assert artifact["promotion_score"] == 0


def test_disqualified_fixture_and_terminal_gate_branch_are_explicit() -> None:
    """REQ-REPORT-7445: validation defects are disqualified, never benefit nulls."""

    source7432, source7440 = exp._fixture_sources()
    preconditions = exp.authenticate_upstreams(source7432, source7440)
    amdahl = exp.compute_amdahl_rows(exp.PERSISTENCE_FRACTION)
    stages = exp.reduce_stage_envelope(source7432, source7440)
    boards = exp.build_board_rows(source7432["board_rows"], {"exists": False})
    artifact = exp.assemble_artifact(
        preconditions=preconditions,
        source_hashes={},
        amdahl_rows=amdahl,
        stage_envelope=stages,
        board_rows=boards,
        future_rows=exp.external_future_rows(),
        validation_receipts=[],
        phase_spans=[],
        sidecar_references=[],
        started_at_utc="2026-09-20T00:00:00+00:00",
        completed_at_utc="2026-09-20T00:00:01+00:00",
        started_monotonic_ns=1,
        ended_monotonic_ns=2,
        validation_required=False,
        require_terminal=True,
        flagged_adversarial=True,
    )
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["honest_verdict"].startswith("complete_disqualified")


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (
            lambda value: value.__setitem__("MODEL_SPECS", ["unsloth/Qwen3.8-27B-GGUF"]),
            "current_model_boundary_invalid",
        ),
        (lambda value: value.__setitem__("hardware_value_score", 1), "hardware_score_nonzero"),
        (
            lambda value: value["amdahl_rows"][0].__setitem__("service_speed_limit_x", 100.0),
            "independent_reduction_mismatch",
        ),
        (
            lambda value: value["board_rows"][0].__setitem__("future_access", "sd_card"),
            "board_boundary_invalid",
        ),
        (
            lambda value: value.__setitem__("reproducibility_checksum", "sha256:wrong"),
            "reproducibility_checksum_mismatch",
        ),
    ],
)
def test_artifact_mutations_fail_closed(tmp_path: Path, mutator: object, expected: str) -> None:
    """SCENARIO-REPORT-7445-ARTIFACT: claim and checksum drift is rejected."""

    artifact = exp.build_fixture_artifact()
    mutator(artifact)  # type: ignore[operator]
    if expected != "reproducibility_checksum_mismatch":
        artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert expected in exp.validate_artifact(artifact, root=tmp_path, require_terminal=False)


def test_identity_receipt_source_and_durability_defects_are_named(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7445-ARTIFACT: every structural boundary fails closed."""

    artifact = exp.build_fixture_artifact()
    artifact.update(
        {
            "schema": "wrong",
            "experiment_id": "wrong",
            "milestone": "wrong",
            "run_date": "19000101",
            "invocation_counts": {},
            "inference_substrate_class": "live_model",
            "execution_venue": "device",
            "learning_hardware_route": {},
            "validation_required": True,
            "source_artifact_hashes": [],
        }
    )
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    errors = exp.validate_artifact(artifact, root=tmp_path, require_terminal=False)
    assert {
        "schema_mismatch",
        "experiment_id_mismatch",
        "run_identity_mismatch",
        "current_invocation_counts_nonzero",
        "substrate_class_invalid",
        "execution_venue_invalid",
        "source_artifact_hashes_invalid",
        "durability_boundary_invalid",
        "affected_receipts_invalid",
    } <= set(errors)

    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    artifact = exp.build_fixture_artifact()
    artifact["source_artifact_hashes"] = {
        "ignored": "not-a-record",
        "verified": {
            "path": "source.json",
            "sha256": "sha256:" + "f" * 64,
            "verify_on_replay": True,
        },
    }
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert "source_hash_mismatch:verified" in exp.validate_artifact(
        artifact, root=tmp_path, require_terminal=False
    )


def test_gate_summary_names_exact_first_failure() -> None:
    """REQ-REPORT-7445: blocked or failed gates retain all exact operands."""

    gates = exp.authenticate_upstreams({}, _exp7440())
    summary = exp.gate_check_summary(gates)
    assert summary["all_passed"] is False
    assert summary["first_failed_check"] == "exp7432_experiment_id"
    assert summary["first_failed_upstream"] == "Exp7432"
    assert summary["first_failed_path"] == exp.EXP7432_PATH.as_posix()
    assert summary["first_failed_field"] == "experiment_id"
    assert summary["first_failed_expected"] == "exp7432-v651-update-placement"
    assert summary["first_failed_observed"] is None
    assert exp.gate_check_summary([])["all_passed"] is True


def test_helpers_and_cold_replay_cli_are_read_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7445-ARTIFACT: the fresh reader does not publish."""

    assert "+00:00" in exp.utc_now()
    exp.progress(0.0, "test", "boundary", completed_units=1)
    assert "phase=test event=boundary" in capsys.readouterr().out

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    assert exp.load_json(malformed) == {}
    assert exp.load_json(sequence) == {}
    assert exp.load_json(tmp_path / "missing.json") == {}

    artifact = exp.build_fixture_artifact()
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.main(["--cold-replay", str(candidate)]) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["validation_errors"] == []
    assert replay["independent_reduction"]["matches_declared"] is True

    assert exp.main(["--independent-reduce", str(candidate)]) == 0
    reduced = json.loads(capsys.readouterr().out)
    assert reduced["matches_declared"] is True

    artifact["amdahl_rows"][0]["service_speed_limit_x"] = 100.0
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.main(["--independent-reduce", str(candidate)]) == 1


def test_validation_plan_is_scoped_and_uses_private_parents(tmp_path: Path) -> None:
    """REQ-REPORT-7445: affected validation cannot expand to the full suite."""

    commands = exp.build_validation_plan(exp.REPO_ROOT, tmp_path)
    assert exp.validate_validation_plan(exp.REPO_ROOT, commands) == []
    assert [command.name for command in commands] == list(exp.validation_scope.REQUIRED_CHECK_NAMES)
    assert all("tests/python" not in command.argv for command in commands)
    widened = list(commands)
    widened[1] = exp.validation_scope.CommandSpec(
        widened[1].name,
        (*widened[1].argv[:-2], "tests/python", "-q"),
        widened[1].scope,
    )
    assert any(
        "unscoped_test_target" in error
        for error in exp.validate_validation_plan(exp.REPO_ROOT, widened)
    )


def test_terminal_receipts_are_required_only_for_terminal_validation(tmp_path: Path) -> None:
    """REQ-REPORT-7445: publication requires every exact fresh-process reader."""

    artifact = exp.build_fixture_artifact()
    assert "terminal_receipts_invalid" in exp.validate_artifact(
        artifact, root=tmp_path, require_terminal=True
    )
    artifact["validation_receipts"] = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in exp.TERMINAL_CHECK_NAMES
    ]
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert exp.validate_artifact(artifact, root=tmp_path, require_terminal=True) == []
