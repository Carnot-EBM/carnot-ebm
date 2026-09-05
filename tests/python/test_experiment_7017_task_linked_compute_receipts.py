"""Tests for the task-linked compute receipt contract.

Spec refs: REQ-REPORT-7017 and SCENARIO-REPORT-7017-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from carnot import experiment_7017_task_linked_compute_receipts as mod
from carnot import gpu_lease_phase_journal as lease_api
from carnot import task_runtime_receipts as receipts
from carnot.pipeline.dual_gpu_assigner import DualGPUAssigner
from carnot.pipeline.dual_gpu_monitor import DualGPUMonitor


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"
TASK_ID = "exp7017-test-task"


def _runner(*, simultaneous: int, available_gpus: int = 2) -> dict[str, object]:
    dual = simultaneous >= 2 and available_gpus >= 2
    return {
        "runner_selected": "DualGPURunner" if dual else "SequentialRunner",
        "selection_rule": (
            "select DualGPURunner only when at least two models are simultaneous, "
            "two GPUs are available, and live execution is requested"
        ),
        "dual_gpu_runner_eligible": dual,
        "simultaneous_model_count": simultaneous,
        "available_gpu_count": available_gpus,
        "live_execution_requested": True,
        "selected": True,
    }


def _receipt(*, overlap: bool = False, model_count: int = 1) -> dict[str, object]:
    task_process = {
        "pid": 4100,
        "parent_pid": 4000,
        "start_time_ticks": 41000,
        "boot_id": "fixture-boot",
        "cmdline_hash": receipts.sha256_text("fixture-task"),
    }
    boundaries = (0, 10, 20, 60, 70, 80)
    phase_clocks = [
        {
            "phase": phase,
            "monotonic_start_ns": boundaries[index],
            "monotonic_end_ns": boundaries[index + 1],
            "wall_clock_start": f"2026-09-05T00:00:0{index}Z",
            "wall_clock_end": f"2026-09-05T00:00:0{index + 1}Z",
        }
        for index, phase in enumerate(receipts.TASK_COMPUTE_REQUIRED_PHASES)
    ]
    intervals = [(22, 58)]
    if model_count == 2:
        intervals = [(22, 38), (42, 58)]
        if overlap:
            intervals = [(22, 55), (25, 58)]
    model_rows = []
    lease_rows = []
    samples = []
    cleanup_rows = []
    for index, (start, stop) in enumerate(intervals):
        pid = 4200 + index
        model_id = f"fixture-model-{index}"
        model_hash = receipts.sha256_text(f"fixture-model-file-{index}")
        lease_id = f"lease-{index}"
        gpu_uuid = f"GPU-fixture-{index}"
        device = f"cuda:{index}"
        lease_rows.append(
            {
                "task_id": TASK_ID,
                "lease_id": lease_id,
                "gpu_uuid": gpu_uuid,
                "device": device,
                "released": True,
                "released_monotonic_ns": 75 + index,
            }
        )
        model_rows.append(
            {
                "task_id": TASK_ID,
                "lease_id": lease_id,
                "pid": pid,
                "process_start_identity": f"linux_proc_stat_starttime:{pid * 10}",
                "model_id": model_id,
                "model_file_hash": model_hash,
                "gpu_uuid": gpu_uuid,
                "device": device,
                "inference_start_ns": start,
                "inference_end_ns": stop,
            }
        )
        samples.append(
            DualGPUMonitor.build_task_linked_sample(
                task_id=TASK_ID,
                lease_id=lease_id,
                gpu_uuid=gpu_uuid,
                device=device,
                utilization_pct=70 + index,
                memory_used_mb=2048 + index,
                power_w=None if index == 0 else 310.5,
                sample_time=f"2026-09-05T00:00:02.{index}Z",
                monotonic_ns=(start + stop) // 2,
                sample_age_s=0.0,
                pid=pid,
                model_id=model_id,
                model_file_hash=model_hash,
            )
        )
        cleanup_rows.append(
            {
                "kind": "model",
                "task_id": TASK_ID,
                "lease_id": lease_id,
                "pid": pid,
                "model_id": model_id,
                "cleanup_monotonic_ns": 72 + index,
                "process_exit_confirmed": True,
                "process_reaped": True,
                "model_unloaded": True,
            }
        )
        cleanup_rows.append(
            {
                "kind": "lease",
                "task_id": TASK_ID,
                "lease_id": lease_id,
                "cleanup_monotonic_ns": 75 + index,
                "lease_released": True,
            }
        )
    simultaneous = 2 if overlap and model_count == 2 else 1
    return receipts.build_task_compute_receipt(
        task_id=TASK_ID,
        task_process_identity=task_process,
        lease_link_rows=lease_rows,
        phase_clocks=phase_clocks,
        gpu_sample_rows=samples,
        model_process_rows=model_rows,
        runner_decision=_runner(simultaneous=simultaneous),
        cleanup_rows=cleanup_rows,
        command=[sys.executable, "fixture.py"],
        config={"fixture": True},
    )


def _refresh_rows(receipt: dict[str, object]) -> None:
    rows = receipt["rows"]
    assert isinstance(rows, list)
    receipt["rows"] = [receipts.seal_adoption_row(row) for row in rows]
    receipt["receipt_sha256"] = receipts.sha256_json(receipt["rows"])


def test_req_report_7017_spec_declares_contract_and_fields() -> None:
    """REQ-REPORT-7017: OpenSpec owns all receipt behavior and artifact fields."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-7017") :]
    for scenario in (
        "SCENARIO-REPORT-7017-PHASE-CLOCKS",
        "SCENARIO-REPORT-7017-LINKAGE",
        "SCENARIO-REPORT-7017-GPU-SAMPLES",
        "SCENARIO-REPORT-7017-RUNNER",
        "SCENARIO-REPORT-7017-CLEANUP",
        "SCENARIO-REPORT-7017-MALFORMED",
        "SCENARIO-REPORT-7017-CONSUMER",
    ):
        assert scenario in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_report_7017_phase_clocks_are_monotonic() -> None:
    """SCENARIO-REPORT-7017-PHASE-CLOCKS: durations derive from ordered clocks."""

    receipt = _receipt()
    report = receipts.validate_task_compute_receipt(receipt)
    assert report["accepted"] is True
    assert report["phase_duration_s"] == 0.00000008
    assert [row["phase"] for row in receipt["rows"]] == list(receipts.TASK_COMPUTE_REQUIRED_PHASES)
    assert all(
        row["duration_s"] == (row["monotonic_end_ns"] - row["monotonic_start_ns"]) / 1_000_000_000
        for row in receipt["rows"]
    )

    changed = deepcopy(receipt)
    changed["rows"][2]["monotonic_start_ns"] = 15
    changed["rows"][2]["duration_s"] = -0.000000005
    _refresh_rows(changed)
    rejected = receipts.validate_task_compute_receipt(changed)
    assert rejected["accepted"] is False
    assert {"phase_overlap", "phase_duration_mismatch"} <= set(rejected["reasons"])


def test_scenario_report_7017_task_and_lease_linkage_fail_closed() -> None:
    """SCENARIO-REPORT-7017-LINKAGE: every sample retains task and lease ownership."""

    receipt = _receipt()
    changed = deepcopy(receipt)
    changed["gpu_sample_rows"][0]["lease_id"] = "foreign-lease"
    report = receipts.validate_task_compute_receipt(changed)
    assert report["accepted"] is False
    assert "gpu_sample_lease_mismatch" in report["reasons"]

    changed = deepcopy(receipt)
    changed["model_process_rows"][0]["task_id"] = "foreign-task"
    report = receipts.validate_task_compute_receipt(changed)
    assert "model_process_task_mismatch" in report["reasons"]


def test_scenario_report_7017_gpu_samples_cover_overlap_and_non_overlap() -> None:
    """SCENARIO-REPORT-7017-GPU-SAMPLES: owned sample intervals determine concurrency."""

    one = receipts.validate_task_compute_receipt(_receipt(model_count=1))
    serial_two = receipts.validate_task_compute_receipt(_receipt(model_count=2))
    overlapping_two = receipts.validate_task_compute_receipt(_receipt(model_count=2, overlap=True))
    assert one["accepted"] is True
    assert serial_two["accepted"] is True
    assert serial_two["peak_simultaneous_model_count"] == 1
    assert overlapping_two["accepted"] is True
    assert overlapping_two["peak_simultaneous_model_count"] == 2


def test_scenario_report_7017_stale_and_malformed_gpu_samples_reject() -> None:
    """SCENARIO-REPORT-7017-GPU-SAMPLES: stale or incomplete telemetry fails closed."""

    stale = _receipt()
    stale["gpu_sample_rows"][0]["sample_age_s"] = 5.1
    report = receipts.validate_task_compute_receipt(stale)
    assert "stale_gpu_sample" in report["reasons"]

    malformed = _receipt()
    del malformed["gpu_sample_rows"][0]["memory_used_mb"]
    report = receipts.validate_task_compute_receipt(malformed)
    assert "gpu_sample_field_missing" in report["reasons"]


def test_scenario_report_7017_power_support_is_explicit() -> None:
    """SCENARIO-REPORT-7017-GPU-SAMPLES: absent power is not_applicable, never zero."""

    unavailable = _receipt()["gpu_sample_rows"][0]
    available = _receipt(model_count=2)["gpu_sample_rows"][1]
    assert unavailable["power_support"] == "not_applicable"
    assert unavailable["power_w"] == "not_applicable"
    assert available["power_support"] == "available"
    assert available["power_w"] == 310.5


def test_scenario_report_7017_runner_choice_is_concurrency_bound(
    monkeypatch,
) -> None:
    """SCENARIO-REPORT-7017-RUNNER: one model is sequential and two can be dual."""

    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    one = DualGPUAssigner([{"name": "a"}], n_gpus=2).runner_decision(simultaneous_model_count=1)
    two = DualGPUAssigner([{"name": "a"}, {"name": "b"}], n_gpus=2).runner_decision(
        simultaneous_model_count=2
    )
    assert one["runner_selected"] == "SequentialRunner"
    assert one["dual_gpu_runner_eligible"] is False
    assert one["simultaneous_model_count"] == 1
    assert two["runner_selected"] == "DualGPURunner"
    assert two["dual_gpu_runner_eligible"] is True
    assert two["selection_rule"]

    false_claim = _receipt()
    false_claim["runner_decision"].update(
        {
            "runner_selected": "DualGPURunner",
            "dual_gpu_runner_eligible": True,
            "simultaneous_model_count": 2,
        }
    )
    report = receipts.validate_task_compute_receipt(false_claim)
    assert "runner_concurrency_mismatch" in report["reasons"]


def test_scenario_report_7017_cleanup_is_complete() -> None:
    """SCENARIO-REPORT-7017-CLEANUP: all model processes and leases terminate."""

    receipt = _receipt(model_count=2, overlap=True)
    assert receipts.validate_task_compute_receipt(receipt)["cleanup_complete"] is True

    missing = deepcopy(receipt)
    missing["cleanup_rows"] = [row for row in missing["cleanup_rows"] if row.get("pid") != 4201]
    report = receipts.validate_task_compute_receipt(missing)
    assert report["accepted"] is False
    assert "model_cleanup_missing" in report["reasons"]

    early = deepcopy(receipt)
    lease_cleanup = next(row for row in early["cleanup_rows"] if row["kind"] == "lease")
    lease_cleanup["cleanup_monotonic_ns"] = 65
    report = receipts.validate_task_compute_receipt(early)
    assert "lease_release_before_cleanup" in report["reasons"]


def test_scenario_report_7017_malformed_receipts_fail_closed() -> None:
    """SCENARIO-REPORT-7017-MALFORMED: truncation and row changes are rejected."""

    assert "receipt_not_mapping" in receipts.validate_task_compute_receipt([])["reasons"]
    receipt = _receipt()
    del receipt["rows"][0]["receipt_hash"]
    report = receipts.validate_task_compute_receipt(receipt)
    assert report["accepted"] is False
    assert "receipt_hash_mismatch" in report["reasons"]

    truncated = _receipt()
    del truncated["runner_decision"]
    report = receipts.validate_task_compute_receipt(truncated)
    assert "task_compute_field_missing" in report["reasons"]


def test_scenario_report_7017_rejection_branches_name_the_failed_rule(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7017-MALFORMED: every malformed field names its rule."""

    base = _receipt(model_count=2, overlap=True)

    def reject(reason: str, mutate, *, refresh_rows: bool = False) -> None:
        changed = deepcopy(base)
        mutate(changed)
        if refresh_rows:
            _refresh_rows(changed)
        report = receipts.validate_task_compute_receipt(changed)
        assert report["accepted"] is False
        assert reason in report["reasons"]

    reject("task_compute_schema_mismatch", lambda value: value.update(schema_version="wrong"))
    reject("task_compute_kind_mismatch", lambda value: value.update(receipt_kind="wrong"))
    reject(
        "task_process_identity_invalid",
        lambda value: value["task_process_identity"].update(pid=1),
    )
    reject(
        "phase_task_identity_mismatch",
        lambda value: value["rows"][0].update(task_process_identity={}),
        refresh_rows=True,
    )
    reject(
        "phase_task_mismatch",
        lambda value: value["rows"][0].update(task_id="foreign-task"),
        refresh_rows=True,
    )
    reject(
        "phase_order_invalid",
        lambda value: (
            value["rows"].reverse(),
            value.update(receipt_sha256=receipts.sha256_json(value["rows"])),
        ),
    )
    reject(
        "phase_clock_invalid",
        lambda value: value["rows"][2].update(monotonic_end_ns=10),
        refresh_rows=True,
    )
    reject(
        "lease_link_field_missing",
        lambda value: value["lease_link_rows"][0].pop("device"),
    )
    reject(
        "lease_task_mismatch",
        lambda value: value["lease_link_rows"][0].update(task_id="foreign-task"),
    )
    reject("lease_id_invalid", lambda value: value["lease_link_rows"].append({**value["lease_link_rows"][0]}))
    reject(
        "phase_lease_mismatch",
        lambda value: value["rows"][0].update(lease_ids=["foreign-lease"]),
        refresh_rows=True,
    )
    reject(
        "model_process_field_missing",
        lambda value: value["model_process_rows"][0].pop("process_start_identity"),
    )
    reject(
        "model_process_lease_mismatch",
        lambda value: value["model_process_rows"][0].update(lease_id="foreign-lease"),
    )
    reject(
        "model_process_gpu_mismatch",
        lambda value: value["model_process_rows"][0].update(device="cuda:99"),
    )
    reject(
        "model_process_identity_invalid",
        lambda value: value["model_process_rows"][0].update(pid=1),
    )
    reject(
        "model_process_duplicate",
        lambda value: value["model_process_rows"].append({**value["model_process_rows"][0]}),
    )
    reject(
        "model_file_hash_invalid",
        lambda value: value["model_process_rows"][0].update(model_file_hash="not-a-hash"),
    )
    reject(
        "model_process_interval_invalid",
        lambda value: value["model_process_rows"][0].update(inference_end_ns=0),
    )
    reject(
        "gpu_sample_task_mismatch",
        lambda value: value["gpu_sample_rows"][0].update(task_id="foreign-task"),
    )
    reject(
        "gpu_sample_time_invalid",
        lambda value: value["gpu_sample_rows"][0].update(sample_time=""),
    )
    reject(
        "gpu_sample_gpu_mismatch",
        lambda value: value["gpu_sample_rows"][0].update(gpu_uuid="GPU-foreign"),
    )
    reject(
        "gpu_sample_process_mismatch",
        lambda value: value["gpu_sample_rows"][0].update(pid=9999),
    )
    reject(
        "gpu_sample_model_hash_mismatch",
        lambda value: value["gpu_sample_rows"][0].update(
            model_file_hash=receipts.sha256_text("foreign-model")
        ),
    )
    reject(
        "gpu_sample_outside_inference",
        lambda value: value["gpu_sample_rows"][0].update(monotonic_ns=10),
    )
    reject(
        "gpu_utilization_invalid",
        lambda value: value["gpu_sample_rows"][0].update(utilization_pct=101),
    )
    reject(
        "gpu_memory_invalid",
        lambda value: value["gpu_sample_rows"][0].update(memory_used_mb=-1),
    )
    reject(
        "gpu_power_support_invalid",
        lambda value: value["gpu_sample_rows"][0].update(power_w=0),
    )
    reject(
        "gpu_power_invalid",
        lambda value: value["gpu_sample_rows"][1].update(power_w=-1),
    )
    reject(
        "gpu_power_support_invalid",
        lambda value: value["gpu_sample_rows"][1].update(power_support="unknown"),
    )
    reject(
        "cleanup_identity_mismatch",
        lambda value: value["cleanup_rows"][0].update(task_id="foreign-task"),
    )
    reject(
        "cleanup_incomplete",
        lambda value: value["cleanup_rows"][0].update(process_reaped=False),
    )
    reject(
        "lease_cleanup_missing",
        lambda value: value["cleanup_rows"].__setitem__(
            slice(None), [row for row in value["cleanup_rows"] if row.get("kind") != "lease"]
        ),
    )
    reject(
        "cleanup_identity_mismatch",
        lambda value: next(
            row for row in value["cleanup_rows"] if row.get("kind") == "lease"
        ).update(task_id="foreign-task"),
    )
    reject(
        "cleanup_incomplete",
        lambda value: next(
            row for row in value["cleanup_rows"] if row.get("kind") == "lease"
        ).update(lease_released=False),
    )
    reject(
        "lease_link_missing",
        lambda value: value.update(
            lease_link_rows=[],
            gpu_sample_rows=[],
            model_process_rows=[],
            cleanup_rows=[],
        ),
    )
    empty = deepcopy(base)
    empty.update(
        lease_link_rows=[],
        gpu_sample_rows=[],
        model_process_rows=[],
        cleanup_rows=[],
    )
    empty["rows"] = [
        receipts.seal_adoption_row({**row, "lease_ids": []}) for row in empty["rows"]
    ]
    empty["receipt_sha256"] = receipts.sha256_json(empty["rows"])
    empty_report = receipts.validate_task_compute_receipt(empty)
    assert {"lease_link_missing", "model_process_missing", "gpu_sample_missing"} <= set(
        empty_report["reasons"]
    )

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert "receipt_not_mapping" in receipts.load_and_validate_task_compute_receipt(malformed)[
        "reasons"
    ]
    valid = tmp_path / "valid.json"
    receipts.write_json_atomic(valid, base)
    assert receipts.load_and_validate_task_compute_receipt(valid)["accepted"] is True


def test_req_report_7017_gpu_lease_exposes_stable_link_id(tmp_path: Path) -> None:
    """REQ-REPORT-7017: the existing lease journal exports its task link."""

    lease = lease_api.GpuLease.acquire(
        runtime_dir=tmp_path,
        task_id=TASK_ID,
        device_uuid="GPU-fixture-lease",
        expected_model="fixture.gguf",
        vram_before_mb=0,
    )
    owner = lease.owner_receipt()
    assert owner["lease_id"] == lease.document["lease_id"]
    malformed = deepcopy(lease.document)
    malformed["lease_id"] = "not-a-lease-id"
    malformed["checksum"] = lease_api.journal_checksum(malformed)
    assert "lease_id_invalid" in lease_api.validate_journal_document(malformed)
    lease.transition("terminal_blocked")
    released = lease.release()
    assert released["lease_id"] == owner["lease_id"]


def test_scenario_report_7017_consumer_calls_shared_builder(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-REPORT-7017-CONSUMER: template wiring calls the shared builder."""

    calls = 0
    original = receipts.build_task_compute_receipt

    def tracked(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(receipts, "build_task_compute_receipt", tracked)
    receipt, validation = mod.run_consumer_fixture(tmp_path, fixture_id="one-model")
    assert calls == 1
    assert receipt["validation"]["accepted"] is True
    assert validation["accepted"] is True


def test_req_report_7017_emits_ready_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-7017: all acceptance and rejection fixtures produce readiness one."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.run(date="20260905", output_path=output, repo_root=REPO)
    assert artifact == json.loads(output.read_text(encoding="utf-8"))
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= artifact.keys()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= artifact["field_principles"].keys()
    assert artifact["inference_substrate"] == ("deterministic_task_compute_receipt_fixtures_no_llm")
    assert artifact["task_compute_receipt_ready_score"] == 1
    assert all(row["accepted"] for row in artifact["acceptance_fixture_rows"])
    assert all(row["rejected"] for row in artifact["rejection_fixture_rows"])
    assert artifact["consumer_wiring_rows"][0]["calls_shared_builder"] is True
    assert artifact["gate_check_summary"] == []
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("positive_")
    assert mod.validate_artifact(artifact) == []


def test_req_report_7017_blocked_artifact_is_schema_complete(tmp_path: Path) -> None:
    """REQ-REPORT-7017: a failed precondition writes exact blocked diagnostics."""

    checks = mod.check_preconditions(tmp_path)
    artifact = mod.blocked_artifact(date="20260905", preconditions=checks)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= artifact.keys()
    assert artifact["task_compute_receipt_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_task_compute_receipt_contract"
    assert artifact["gate_check_summary"]
    assert {"failed_check", "expected_value", "observed_value"} <= set(
        artifact["gate_check_summary"][0]
    )
    output = tmp_path / "blocked.json"
    persisted = mod.run(date="20260905", output_path=output, repo_root=tmp_path)
    assert persisted == json.loads(output.read_text(encoding="utf-8"))
    assert persisted["honest_verdict"] == "blocked_task_compute_receipt_contract"


def test_req_report_7017_artifact_validator_rejects_forged_score() -> None:
    """REQ-REPORT-7017: the artifact score must recompute from its fixture rows."""

    artifact = mod._empty_artifact(date="20260905", preconditions=[])
    artifact["task_compute_receipt_ready_score"] = 1
    artifact["verdict_class"] = "positive"
    artifact["honest_verdict"] = "positive_forged"
    errors = mod.validate_artifact(artifact)
    assert "ready_score_mismatch" in errors


def test_req_report_7017_artifact_validator_rejects_malformed_metadata() -> None:
    """REQ-REPORT-7017: required principles, substrate, checksum, and prefix are checked."""

    assert mod.validate_artifact([]) == ["artifact_not_mapping"]
    artifact = mod._empty_artifact(date="20260905", preconditions=[])
    artifact.pop("rows")
    artifact["field_principles"] = {}
    artifact["inference_substrate"] = "wrong"
    artifact["verifier_is_oracle"] = True
    artifact["honest_verdict"] = "wrong"
    errors = mod.validate_artifact(artifact)
    assert {
        "required_artifact_field_missing",
        "field_principle_missing",
        "inference_substrate_mismatch",
        "verifier_oracle_mismatch",
        "verdict_prefix_mismatch",
        "reproducibility_checksum_mismatch",
        "blocked_gate_summary_missing",
    } <= set(errors)

    ready = mod._empty_artifact(date="20260905", preconditions=[])
    ready.update(
        {
            "acceptance_fixture_rows": [{"accepted": True}],
            "rejection_fixture_rows": [{"rejected": True}],
            "consumer_wiring_rows": [
                {"calls_shared_builder": True, "serialized_receipt_valid": True}
            ],
            "task_compute_receipt_ready_score": 1,
            "verdict_class": "null",
            "honest_verdict": "null_ready_but_wrong_class",
        }
    )
    ready["reproducibility_checksum"] = mod._checksum(ready)
    assert "ready_verdict_class_mismatch" in mod.validate_artifact(ready)


def test_scenario_report_7017_consumer_requires_task_identity(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-REPORT-7017-LINKAGE: an unavailable task identity stops emission."""

    monkeypatch.setattr(receipts, "read_process_identity", lambda _pid: None)
    try:
        mod.run_consumer_fixture(tmp_path, fixture_id="one-model")
    except RuntimeError as exc:
        assert str(exc) == "task process identity is unavailable"
    else:
        raise AssertionError("missing task identity did not stop receipt emission")


def test_req_report_7017_internal_artifact_failure_stays_partial(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-REPORT-7017: a producer-side validation error cannot retain readiness."""

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced_error"])
    artifact = mod.run(date="20260905", output_path=tmp_path / "forced.json", repo_root=REPO)
    assert artifact["task_compute_receipt_ready_score"] == 0
    assert artifact["honest_verdict"] == "partial_task_compute_receipt_artifact_invalid"
    assert artifact["gate_check_summary"][0]["observed_value"] == ["forced_error"]


def test_req_report_7017_cli_generates_and_validates(tmp_path: Path, capsys) -> None:
    """REQ-REPORT-7017: the required command path emits and validates the artifact."""

    output = tmp_path / "cli.json"
    assert mod.main(["--date", "20260905", "--output", str(output)]) == 0
    generated = json.loads(capsys.readouterr().out)
    assert generated["task_compute_receipt_ready_score"] == 1
    assert mod.main(["--date", "20260905", "--validate", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["accepted"] is True
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["task_compute_receipt_ready_score"] = 0
    receipts.write_json_atomic(output, payload)
    assert mod.main(["--date", "20260905", "--validate", str(output)]) == 1
    assert json.loads(capsys.readouterr().out)["accepted"] is False


def test_scenario_report_7017_consumer_wrapper_runs_from_repo_root(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7017-CONSUMER: the required script path runs start to stop."""

    output = tmp_path / "wrapper.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/experiments/experiment_7017_task_linked_compute_receipts.py"),
            "--date",
            "20260905",
            "--output",
            str(output),
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["task_compute_receipt_ready_score"] == 1
