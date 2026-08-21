"""Tests for Exp6481 monotonic phase and concurrency receipts.

Spec refs: REQ-INFRA-6481, SCENARIO-INFRA-6481-MONOTONIC-PHASES,
SCENARIO-INFRA-6481-DEPENDENCY-BINDING,
SCENARIO-INFRA-6481-RESOURCE-OWNERSHIP,
SCENARIO-INFRA-6481-CONCURRENCY-OVERLAP,
SCENARIO-INFRA-6481-FAIL-CLOSED-VALIDATION,
SCENARIO-INFRA-6481-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6481_monotonic_phase_concurrency_receipt_contract as mod
from carnot import phase_concurrency_receipts as receipts


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _fixture(tmp_path: Path) -> dict:
    return mod.build_positive_fixture_rows(root=REPO, fixture_root=tmp_path)


def _validate(rows: list[dict], fixture: dict) -> dict:
    return receipts.validate_receipt_rows(
        rows,
        expected_attempts=fixture["expected_attempts"],
        verify_dependency_files=True,
    )


def _with_checksum(artifact: dict) -> dict:
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def test_req_infra_6481_spec_declares_contract_and_fields() -> None:
    """REQ-INFRA-6481: OpenSpec owns the receipt contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6481") : text.index("REQ-INFRA-6351")]
    for marker in (
        "SCENARIO-INFRA-6481-MONOTONIC-PHASES",
        "SCENARIO-INFRA-6481-DEPENDENCY-BINDING",
        "SCENARIO-INFRA-6481-RESOURCE-OWNERSHIP",
        "SCENARIO-INFRA-6481-CONCURRENCY-OVERLAP",
        "SCENARIO-INFRA-6481-FAIL-CLOSED-VALIDATION",
        "SCENARIO-INFRA-6481-ARTIFACT",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.API_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_infra_6481_monotonic_phases_and_dependency_binding(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6481-MONOTONIC-PHASES and DEPENDENCY-BINDING."""

    fixture = _fixture(tmp_path)
    rows = fixture["rows"]
    report = _validate(rows, fixture)

    assert report["accepted"] is True
    assert report["reasons"] == []
    assert report["phase_count_by_attempt"] == {
        attempt: len(receipts.REQUIRED_PHASES) for attempt in fixture["expected_attempts"]
    }
    assert receipts.receipt_schema_and_hash()["schema_version"] == receipts.SCHEMA_VERSION

    bad = deepcopy(rows)
    first_phase = next(row for row in bad if row.get("row_type") == "phase")
    first_phase["monotonic_end_ns"] = first_phase["monotonic_start_ns"] - 1
    first_phase = receipts.refresh_row_hash(first_phase)
    bad[bad.index(first_phase)] = first_phase
    assert "negative_interval" in _validate(bad, fixture)["reasons"]

    bad = deepcopy(rows)
    execution = next(row for row in bad if row.get("phase") == "execution")
    execution["monotonic_start_ns"] = execution["monotonic_start_ns"] - 1_000_000
    execution["monotonic_end_ns"] = execution["monotonic_start_ns"] + 100
    refreshed = receipts.refresh_row_hash(execution)
    bad[bad.index(execution)] = refreshed
    assert "phase_inversion" in _validate(bad, fixture)["reasons"]

    bad = deepcopy(rows)
    first_phase = next(row for row in bad if row.get("row_type") == "phase")
    first_phase["wall_clock_end"] = "2026-08-21T00:00:00Z"
    first_phase["wall_clock_start"] = "2026-08-21T00:00:01Z"
    first_phase = receipts.refresh_row_hash(first_phase)
    bad[bad.index(first_phase)] = first_phase
    assert "wall_clock_inversion" in _validate(bad, fixture)["reasons"]

    fixture["dependency_path"].write_text("changed dependency bytes\n", encoding="utf-8")
    assert "dependency_hash_changed" in _validate(rows, fixture)["reasons"]


def test_scenario_infra_6481_resource_ownership_and_concurrency_overlap(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6481-RESOURCE-OWNERSHIP and CONCURRENCY-OVERLAP."""

    fixture = _fixture(tmp_path)
    rows = fixture["rows"]
    report = _validate(rows, fixture)
    decisions = receipts.build_concurrency_decision_rows(rows)
    decisions_by_kind = {row["decision"] for row in decisions}

    assert report["accepted"] is True
    assert "safe_overlap" in decisions_by_kind
    assert "serialized_exclusive" in decisions_by_kind
    assert any(row["resource_key"] == "CPU:shared" for row in decisions)
    assert any(row["resource_key"] == "GPU:0" for row in decisions)

    bad = deepcopy(rows)
    gpu_rows = [
        row
        for row in bad
        if row.get("row_type") == "resource_interval" and row.get("resource_key") == "GPU:0"
    ]
    gpu_rows[1]["monotonic_start_ns"] = gpu_rows[0]["monotonic_start_ns"]
    gpu_rows[1]["monotonic_end_ns"] = gpu_rows[0]["monotonic_end_ns"]
    gpu_rows[1] = receipts.refresh_row_hash(gpu_rows[1])
    bad[bad.index(gpu_rows[1])] = gpu_rows[1]
    assert "overlapping_exclusive_resource_claim" in _validate(bad, fixture)["reasons"]

    bad = deepcopy(rows)
    resource = next(row for row in bad if row.get("row_type") == "resource_interval")
    resource["release_present"] = False
    resource = receipts.refresh_row_hash(resource)
    bad[bad.index(next(row for row in bad if row.get("row_hash") == resource["row_hash"]))] = resource

    report = _validate(bad, fixture)
    assert "missing_release" in report["reasons"]


def test_scenario_infra_6481_fail_closed_attack_matrix(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6481-FAIL-CLOSED-VALIDATION: attacks are rejected."""

    fixture = _fixture(tmp_path)
    rows = fixture["rows"]
    matrix = receipts.mutation_attack_matrix(
        rows,
        expected_attempts=fixture["expected_attempts"],
        verify_dependency_files=True,
    )
    by_id = {row["attack_id"]: row for row in matrix["rows"]}

    assert set(by_id) == set(receipts.ATTACK_IDS)
    assert matrix["all_critical_fail_closed"] is True
    assert matrix["false_accept_count"] == 0
    expected_reason = {
        "borrowed_nvidia_smi_activity": "borrowed_global_activity",
        "stale_dependency_artifact": "dependency_hash_changed",
        "duplicated_attempt_id": "duplicated_attempt_id",
        "forged_clocks": "negative_interval",
        "cross_task_output_path": "cross_task_output_path",
        "parent_child_pid_confusion": "parent_child_pid_confusion",
        "pid_reuse": "pid_reuse",
        "output_before_execution": "output_write_before_execution",
        "copied_receipt": "row_hash_mismatch",
    }
    for attack_id, reason in expected_reason.items():
        assert by_id[attack_id]["fail_closed"] is True
        assert reason in by_id[attack_id]["reasons"]

    with pytest.raises(ValueError, match="unknown attack_id"):
        receipts.mutate_rows_for_attack("unknown", rows)


def test_req_infra_6481_defensive_validator_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-INFRA-6481: defensive branches stay explicit and fail closed."""

    fixture = _fixture(tmp_path)
    rows = fixture["rows"]

    assert receipts._int_value(None) is None
    assert receipts._int_value(True) is None
    assert receipts._int_value("bad") is None
    assert receipts._parse_wall_clock("") is None
    assert receipts._parse_wall_clock("not-a-clock") is None
    assert receipts._proc_start_identity(-1) == "pid_start_unavailable:-1"
    assert receipts._overlap({"monotonic_start_ns": None}, {"monotonic_start_ns": 1}) is False

    decisions = receipts.build_concurrency_decision_rows(rows)
    assert receipts.validate_receipt_rows(
        [*rows, *decisions, {"row_type": "attack", "row_hash": "bad"}],
        expected_attempts=fixture["expected_attempts"],
    )["accepted"] is False

    cpu_resources = [
        row
        for row in deepcopy(rows)
        if row.get("row_type") == "resource_interval" and row.get("resource_key") == "CPU:shared"
    ]
    cpu_resources[1]["monotonic_start_ns"] = cpu_resources[0]["monotonic_end_ns"] + 1
    cpu_resources[1]["monotonic_end_ns"] = cpu_resources[1]["monotonic_start_ns"] + 1
    receipts.refresh_row_hash(cpu_resources[1])
    assert {
        row["decision"] for row in receipts.build_concurrency_decision_rows(cpu_resources)
    } == {"independent_serial"}

    bad = deepcopy(rows)
    process = next(row for row in bad if row.get("row_type") == "process_identity")
    process["parent_pid"] = process["pid"]
    receipts.refresh_row_hash(process)
    assert "parent_child_pid_confusion" in _validate(bad, fixture)["reasons"]

    bad = [
        row
        for row in deepcopy(rows)
        if not (
            row.get("row_type") == "process_identity"
            and row.get("attempt_id") == "attempt-cpu-a"
        )
    ]
    assert "process_identity_missing" in _validate(bad, fixture)["reasons"]

    bad = deepcopy(rows)
    phase = next(row for row in bad if row.get("row_type") == "phase")
    phase["monotonic_start_ns"] = None
    receipts.refresh_row_hash(phase)
    report = _validate(bad, fixture)
    assert "missing_monotonic_interval" in report["reasons"]

    bad = deepcopy(rows)
    phase = next(row for row in bad if row.get("row_type") == "phase")
    phase["wall_clock_start"] = ""
    receipts.refresh_row_hash(phase)
    assert "wall_clock_interval_missing" in _validate(bad, fixture)["reasons"]

    bad = [
        row
        for row in deepcopy(rows)
        if not (
            row.get("row_type") == "phase"
            and row.get("attempt_id") == "attempt-cpu-a"
            and row.get("phase") == "resource_release"
        )
    ]
    assert "missing_phase:attempt-cpu-a:resource_release" in _validate(bad, fixture)[
        "reasons"
    ]

    bad = deepcopy(rows)
    dependency = next(row for row in bad if row.get("row_type") == "dependency")
    dependency["sha256"] = "bad"
    receipts.refresh_row_hash(dependency)
    assert "dependency_hash_missing" in _validate(bad, fixture)["reasons"]

    bad = deepcopy(rows)
    dependency = next(row for row in bad if row.get("row_type") == "dependency")
    duplicate = deepcopy(dependency)
    duplicate["sha256"] = "sha256:" + "2" * 64
    receipts.refresh_row_hash(duplicate)
    assert "dependency_hash_changed" in receipts.validate_receipt_rows(
        [*bad, duplicate],
        expected_attempts=fixture["expected_attempts"],
        verify_dependency_files=False,
    )["reasons"]

    bad = deepcopy(rows)
    resource = next(row for row in bad if row.get("row_type") == "resource_interval")
    resource["monotonic_start_ns"] = None
    receipts.refresh_row_hash(resource)
    assert "missing_resource_interval" in _validate(bad, fixture)["reasons"]

    bad = deepcopy(rows)
    resource = next(row for row in bad if row.get("row_type") == "resource_interval")
    resource["monotonic_end_ns"] = resource["monotonic_start_ns"] - 1
    receipts.refresh_row_hash(resource)
    assert "negative_interval" in _validate(bad, fixture)["reasons"]

    bad = deepcopy(rows)
    same_attempt_resource = deepcopy(
        next(row for row in bad if row.get("row_type") == "resource_interval")
    )
    same_attempt_resource["resource_key"] = "CPU:shared"
    receipts.refresh_row_hash(same_attempt_resource)
    assert receipts.validate_receipt_rows(
        [*bad, same_attempt_resource],
        expected_attempts=fixture["expected_attempts"],
    )["accepted"] is True

    bad = deepcopy(rows)
    output = next(row for row in bad if row.get("row_type") == "output")
    output["sha256"] = "bad"
    output["write_monotonic_ns"] = None
    receipts.refresh_row_hash(output)
    report = _validate(bad, fixture)
    assert {"output_hash_missing", "output_write_time_missing"} <= set(report["reasons"])

    assert mod._status(0.0, {"all_gates_passed": False}) == (
        "blocked_phase_concurrency_receipt_contract"
    )
    assert mod._honest_verdict("blocked_phase_concurrency_receipt_contract").startswith(
        "complete_blocked:"
    )

    blocked = mod.build_artifact(
        root=REPO,
        result_path=tmp_path / "blocked.json",
        fixture_root=tmp_path / "blocked-fixtures",
        write=False,
        duration_s=1.0,
        tests_run=[],
    )
    with monkeypatch.context() as mp:
        mp.setattr(
            mod,
            "_protected_unchanged",
            lambda root, before: {  # noqa: ARG005
                "protected_files_unchanged": False,
                "files": {},
            },
        )
        blocked = mod.build_artifact(
            root=REPO,
            result_path=tmp_path / "blocked.json",
            fixture_root=tmp_path / "blocked-fixtures-2",
            write=False,
            duration_s=1.0,
            tests_run=[],
        )
    assert blocked["status"] == "blocked_phase_concurrency_receipt_contract"

    written = mod.build_artifact(
        root=REPO,
        result_path=tmp_path / "build-write.json",
        fixture_root=tmp_path / "build-write-fixtures",
        write=True,
        duration_s=1.0,
        tests_run=[],
    )
    assert json.loads((tmp_path / "build-write.json").read_text(encoding="utf-8")) == written


def test_scenario_infra_6481_artifact_recomputes_and_validates(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6481-ARTIFACT: terminal artifact is row-recomputed."""

    artifact = mod.build_artifact(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        fixture_root=tmp_path / "fixtures",
        write=False,
        duration_s=1.0,
        tests_run=[{"command": "focused", "exit_code": 0}],
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_phase_concurrency_receipt_contract"
    assert artifact["phase_concurrency_receipt_ready_score"] == 1.0
    assert artifact["aggregate_row_recomputation"] == mod.recompute_aggregates_from_rows(
        artifact["per_unit_rows"]
    )
    assert artifact["protected_files_unchanged"]["protected_files_unchanged"] is True
    assert artifact["protected_files_unchanged"]["files"]["scripts/research_conductor.py"][
        "unchanged"
    ] is True
    assert artifact["protected_files_unchanged"]["files"]["research-roadmap.yaml"][
        "unchanged"
    ] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["honest_verdict"].startswith("complete:")

    bad = _with_checksum({**artifact, "phase_concurrency_receipt_ready_score": 0.0})
    assert "phase_concurrency_receipt_ready_score mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["per_unit_rows"] = bad["per_unit_rows"][:-1]
    bad = _with_checksum(bad)
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(bad)

    bad = _with_checksum({**artifact, "inference_substrate": "live_llm_inference"})
    assert "inference_substrate mismatch" in mod.validate_artifact(bad)

    bad = _with_checksum({**artifact, "verifier_is_oracle": False})
    assert "verifier_is_oracle must be true" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["protected_files_unchanged"]["protected_files_unchanged"] = False
    bad = _with_checksum(bad)
    assert "protected_files_unchanged must be true" in mod.validate_artifact(bad)

    bad = _with_checksum({**artifact, "field_provenance": {}})
    assert "field_provenance must cover exactly required fields" in mod.validate_artifact(bad)

    bad = _with_checksum({**artifact, "field_principles": {}})
    assert "missing field_principles entry: status" in mod.validate_artifact(bad)

    bad = _with_checksum({**artifact, "honest_verdict": "done"})
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(bad)

    bad = {**artifact, "reproducibility_checksum": "sha256:bad"}
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    del bad["status"]
    assert "missing required field: status" in mod.validate_artifact(bad)


def test_req_infra_6481_run_write_and_cli_validate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-INFRA-6481: run writes the deliverable and validates it."""

    result = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.run(
        date="20260821",
        result_path=result,
        fixture_root=tmp_path / "run-fixtures",
        write=True,
        tests_run=[{"command": "focused", "exit_code": 0}],
    )

    assert json.loads(result.read_text(encoding="utf-8")) == artifact
    assert artifact["phase_concurrency_receipt_ready_score"] == 1.0

    result_cli = tmp_path / "cli.json"
    assert (
        mod.main(
            [
                "--date",
                "20260821",
                "--result-path",
                str(result_cli),
                "--fixture-root",
                str(tmp_path / "cli-fixtures"),
            ]
        )
        == 0
    )
    written = json.loads(result_cli.read_text(encoding="utf-8"))
    assert written["status"] == "complete_phase_concurrency_receipt_contract"

    assert mod.main(["--validate", "--result-path", str(result_cli)]) == 0
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out["ok"] is True

    missing = tmp_path / "missing.json"
    assert mod.main(["--validate", "--result-path", str(missing)]) == 1
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out == {"errors": ["artifact missing"], "ok": False}
