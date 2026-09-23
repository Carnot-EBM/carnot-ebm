"""Tests for REQ-REPORT-7571 and SCENARIO-REPORT-7571-* contracts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7571_v661_portable_calibration as exp
from carnot.reporting import experiment_7303_validation_scope as validation_scope


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Represent one private check without claiming that a child process ran."""

    return {
        "name": name,
        "command": f"private {name}",
        "command_argv": ["private", name],
        "scope": "private_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "a" * 64,
        "output_tail": "private fixture",
        "passed": passed,
        "timed_out": False,
    }


def _receipts() -> list[dict[str, Any]]:
    """Provide every fixed receipt required by a pure artifact fixture."""

    names = [*validation_scope.REQUIRED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES]
    return [_receipt(name) for name in names]


def test_preconditions_stop_on_disqualified_exp7561() -> None:
    """REQ-REPORT-7571; SCENARIO-REPORT-7571-BLOCKED."""

    context = exp.collect_preconditions(ROOT)

    assert context["prototype"]["verdict_class"] == "disqualified"
    assert context["prototype"]["recalibration_ready_score"] == 0
    assert context["blocker"] == {
        "check": "exp7561_recalibration_ready_score",
        "upstream": "Exp7561",
        "path": exp.PROTOTYPE_PATH.as_posix(),
        "field": "recalibration_ready_score",
        "expected": 1,
        "observed": 0,
        "op": "eq",
    }
    assert context["kernel_branch_ready"] is False
    assert all(
        row["passed"] for row in context["rows"] if row["check"] != context["blocker"]["check"]
    )


def test_board_rows_preserve_three_distinct_scopes(tmp_path: Path) -> None:
    """REQ-REPORT-7571; SCENARIO-REPORT-7571-BOARDS."""

    context = exp.collect_preconditions(ROOT)
    receipt = exp.scan_gatemate_receipts(ROOT, tmp_path / "gatemate.json")
    rows = exp.build_board_rows(context["service_artifact"], receipt)

    assert [row["board"] for row in rows] == ["KV260", "PolarFire", "GateMate"]
    assert rows[0]["exact_claim_scope"] == "historical_kv260_fpga_fabric_sampling_only"
    assert rows[0]["future_access"] == "ssh kria only"
    assert rows[0]["architecture_limit"] == "k_max<=5"
    assert rows[1]["exact_claim_scope"] == (
        "historical_hash_matched_cpu_dispatch_only_no_fpga_sampling"
    )
    assert rows[1]["fpga_sampling_claimed"] is False
    assert rows[2]["current_disposition"] == "blocked_unchanged_physical_prerequisite"
    assert rows[2]["gate_check_summary"]["observed"] == 0
    assert all(row["hardware_operations_issued"] == [] for row in rows)


def test_blocked_artifact_uses_unstarted_not_zero_measurements(tmp_path: Path) -> None:
    """REQ-REPORT-7571; SCENARIO-REPORT-7571-BLOCKED and -COSTS."""

    artifact = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())

    assert artifact["honest_verdict"] == ("complete_blocked_exp7561_recalibration_ready_score")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["portable_kernel_ready_score"] == 0
    assert artifact["service_measurement_complete_score"] == 0
    assert artifact["board_continuity_complete_score"] == 1
    assert artifact["sample_size_budget"]["parity_cases"]["unstarted"] == 10_000
    assert artifact["sample_size_budget"]["paired_service_trials"]["unstarted"] == 30
    assert artifact["kernel_and_service_costs"]["kernel_speedup_ratio"] is None
    assert artifact["kernel_and_service_costs"]["complete_service_speedup_ratio"] is None
    assert artifact["hardware_acceleration_bound"]["ideal_kernel_only_speedup"] is None
    assert artifact["positive_portability"] is False
    assert artifact["predictive_benefit_claimed"] is False
    assert artifact["hardware_speed_claimed"] is False
    assert artifact["e2e_004"]["applicable"] is False
    assert exp.validate_artifact(artifact, root=tmp_path, verify_sources=False) == []


def test_reducer_rejects_promoted_or_missing_blocked_evidence(tmp_path: Path) -> None:
    """REQ-REPORT-7571 keeps readiness, service, and board scores independent."""

    artifact = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    reduction = exp.independent_reduce(artifact)
    assert reduction["portable_kernel_ready_score"] == 0
    assert reduction["service_measurement_complete_score"] == 0
    assert reduction["board_continuity_complete_score"] == 1
    assert reduction["required_validation_passed"] is True

    changed = deepcopy(artifact)
    changed["portable_kernel_ready_score"] = 1
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "portable_kernel_ready_score_mismatch" in exp.validate_artifact(
        changed, root=tmp_path, verify_sources=False
    )

    changed = deepcopy(artifact)
    changed["board_rows"].pop()
    changed["rows"] = deepcopy(changed["board_rows"])
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "board_continuity_complete_score_mismatch" in exp.validate_artifact(
        changed, root=tmp_path, verify_sources=False
    )


def test_reader_rejects_cost_call_receipt_and_checksum_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7571-COSTS and -E2E reject fabricated completion."""

    artifact = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    mutations = [
        (
            "blocked_cost_must_be_unmeasured",
            lambda value: value["kernel_and_service_costs"].update(kernel_speedup_ratio=10.0),
        ),
        (
            "current_invocation_claim_invalid",
            lambda value: value["invocation_counts"]["forward_calls"].update(attempted=1),
        ),
        (
            "required_validation_failed",
            lambda value: value["validation_receipts"][0].update(passed=False),
        ),
        (
            "field_principles_incomplete",
            lambda value: value["field_principles"].pop("rows"),
        ),
    ]
    for expected, mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert expected in exp.validate_artifact(changed, root=tmp_path, verify_sources=False)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        changed, root=tmp_path, verify_sources=False
    )


def test_source_custody_detects_changed_bytes(tmp_path: Path) -> None:
    """REQ-REPORT-7571 binds every current input by exact byte hash."""

    source = tmp_path / "source.json"
    source.write_text('{"verdict_class":"blocked"}\n', encoding="utf-8")
    reference = exp.source_row(source, tmp_path)
    artifact = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    artifact["source_artifact_hashes"] = {"source.json": reference}
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert exp.validate_artifact(artifact, root=tmp_path) == []

    source.write_text('{"verdict_class":"null"}\n', encoding="utf-8")
    assert "source_hash_invalid:source.json" in exp.validate_artifact(artifact, root=tmp_path)


def test_cold_replay_and_independent_cli_modes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7571-E2E runs both fresh read-only modes."""

    artifact = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    candidate = tmp_path / "candidate.json"
    exp.atomic_json(candidate, artifact)

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
    assert '"valid": true' in capsys.readouterr().out.lower()

    candidate.write_text("[]\n", encoding="utf-8")
    assert exp.main(["--date", exp.RUN_DATE, "--cold-replay", str(candidate)]) == 1


def test_validation_manifest_and_terminal_commands_are_scoped(tmp_path: Path) -> None:
    """REQ-REPORT-7571; SCENARIO-REPORT-7571-E2E."""

    commands = exp.build_validation_commands(ROOT, tmp_path / "private")
    assert [row.name for row in commands] == list(validation_scope.REQUIRED_CHECK_NAMES)
    assert exp.validate_command_plan(ROOT, exp.AFFECTED_MANIFEST, commands) == []
    by_name = {row.name: row for row in commands}
    assert exp.TEST_PATH.as_posix() in by_name["focused_pytest"].argv
    assert "--fail-under=100" in by_name["changed_module_coverage_report"].argv
    assert dict(by_name["changed_module_coverage_report"].command_environment)["COVERAGE_FILE"]

    terminal = exp.terminal_commands(tmp_path / "candidate.json")
    assert [row.spec.name for row in terminal] == list(exp.TERMINAL_CHECK_NAMES)
    assert "--strict" in terminal[-1].spec.argv


def test_parser_binds_date_and_paths() -> None:
    """REQ-REPORT-7571 binds the declared run date and thin CLI modes."""

    args = exp.parse_args(["--date", exp.RUN_DATE, "--root", str(ROOT)])
    assert args.root == ROOT
    assert args.output == exp.RESULT_PATH
    with pytest.raises(ValueError, match="run_date_must_equal_20260923"):
        exp.parse_args(["--date", "20260922"])


def test_checksum_ignores_only_runtime_noise() -> None:
    """REQ-REPORT-7571 binds evidence while allowing replay timing to differ."""

    value = {"alpha": 1, "duration_s": 1.0, "reproducibility_checksum": "old"}
    first = exp.reproducibility_checksum(value)
    value["duration_s"] = 2.0
    assert exp.reproducibility_checksum(value) == first
    value["alpha"] = 2
    assert exp.reproducibility_checksum(value) != first


def test_missing_or_malformed_sources_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7571-BLOCKED never fabricates fallback source data."""

    missing = tmp_path / "missing.json"
    assert exp.load_object(missing) == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("[", encoding="utf-8")
    assert exp.load_object(malformed) == {}
    with pytest.raises(FileNotFoundError):
        exp.source_row(missing, tmp_path)


def test_validator_names_remaining_defensive_boundaries(tmp_path: Path) -> None:
    """REQ-REPORT-7571 rejects identity, declaration, gate, and claim drift."""

    artifact = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    mutations = [
        ("artifact_identity_mismatch", lambda value: value.update(schema="wrong")),
        ("task_binding_mismatch", lambda value: value.update(run_date="20260922")),
        ("model_specs_not_empty", lambda value: value.update(model_specs=[{"name": "x"}])),
        (
            "inference_declaration_invalid",
            lambda value: value.update(execution_venue="host_cpu"),
        ),
        ("external_blocker_incomplete", lambda value: value.update(external_blocker={})),
        ("terminal_verdict_mismatch", lambda value: value.update(verdict_class="null")),
        (
            "score_not_bare_numeric:portable_kernel_ready_score",
            lambda value: value.update(portable_kernel_ready_score=True),
        ),
        (
            "independent_reduction_failed",
            lambda value: value.update(board_rows=[1], rows=[1]),
        ),
        (
            "blocked_hardware_bound_must_be_unmeasured",
            lambda value: value["hardware_acceleration_bound"].update(
                ideal_kernel_only_speedup=2.0
            ),
        ),
        (
            "blocked_claim_scope_invalid",
            lambda value: value.update(positive_portability=True),
        ),
        (
            "gate_principle_missing",
            lambda value: value["acceptance_gate_results"][0].update(principle=""),
        ),
    ]
    for expected, mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert expected in exp.validate_artifact(changed, root=tmp_path, verify_sources=False)


def test_source_labels_size_and_independent_malformed_reader(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7571-E2E checks external labels and exact source sizes."""

    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    reference = exp.source_row(source, tmp_path / "other-root")
    assert reference["path"] == str(source.resolve())

    artifact = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    artifact["source_artifact_hashes"] = {"source": {**reference, "bytes": 99}}
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert "source_size_invalid:source" in exp.validate_artifact(artifact, root=tmp_path)

    malformed = tmp_path / "not-an-object.json"
    malformed.write_text("[]\n", encoding="utf-8")
    assert exp.independent_replay(malformed) == ["artifact_not_object"]
