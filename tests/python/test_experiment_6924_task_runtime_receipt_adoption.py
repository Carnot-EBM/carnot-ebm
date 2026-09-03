"""Tests for the V606 task-owned runtime receipt adoption path.

Spec refs: REQ-REPORT-6924 and SCENARIO-REPORT-6924-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

from carnot import experiment_6924_task_runtime_receipt_adoption as mod
from carnot import task_runtime_receipts as receipts
from scripts.experiment_template import ExperimentTemplate


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"


def _identity(pid: int, parent_pid: int) -> dict[str, object]:
    return {
        "pid": pid,
        "parent_pid": parent_pid,
        "start_time_ticks": pid * 10,
        "boot_id": "fixture-boot-id",
        "cmdline_hash": receipts.sha256_text(f"fixture-process-{pid}"),
    }


def _runner(substrate: str) -> dict[str, object]:
    return {
        "runner_id": f"fixture-{substrate}",
        "binary_path": sys.executable,
        "binary_sha256": receipts.sha256_file(sys.executable),
        "substrate": substrate,
        "selected": True,
    }


def _row(
    *,
    start: int,
    end: int,
    phase: str = "generation",
    model_id: str = "fixture-model-a",
    model_count: int = 1,
    concurrency_mode: str = "sequential",
    concurrency_group: str = "fixture-group",
    task_pid: int = 4100,
    child_pid: int | None = None,
    device_ids: tuple[str, ...] = ("CPU",),
    gpu_samples: list[dict[str, object]] | None = None,
    server_lifecycle: dict[str, object] | None = None,
    overlap_explained: bool = False,
    sample_gap_limit_s: float = 5.0,
) -> dict[str, object]:
    task_identity = _identity(task_pid, 4000)
    child_pids = [] if child_pid is None else [child_pid]
    lineages = []
    if child_pid is not None:
        lineages.append(
            {
                "child_pid": child_pid,
                "owned": True,
                "chain": [_identity(child_pid, task_pid), task_identity],
            }
        )
    row = receipts.build_phase_row(
        task_id=mod.TASK_ID,
        control_id="cpu-fixture" if device_ids == ("CPU",) else "gpu-fixture",
        phase=phase,
        monotonic_start_ns=start,
        monotonic_end_ns=end,
        wall_clock_start="2026-09-03T00:00:00Z",
        wall_clock_end="2026-09-03T00:00:01Z",
        parent_pid=task_pid,
        child_pids=child_pids,
        command=[sys.executable, "-c", "fixture"],
        config={"fixture": True},
        model_identity={
            "model_id": model_id,
            "model_sha256": receipts.sha256_text(model_id),
            "model_identity_bound": True,
        },
        runner_selection=_runner("cpu" if device_ids == ("CPU",) else "cuda_gguf"),
        device_ids=list(device_ids),
        concurrency_group=concurrency_group,
        raw_output_bytes=f"{model_id}:{phase}:{start}".encode(),
        exit_status={"returncode": 0, "timed_out": False, "signal": None},
        attribution_confidence=1.0,
        gpu_samples=gpu_samples or [],
        cpu_fallback=False,
        extra={
            "task_process_identity": task_identity,
            "process_lineage": lineages,
            "model_lifecycle": {
                "model_id": model_id,
                "model_count": model_count,
                "concurrency_mode": concurrency_mode,
            },
            "server_lifecycle": server_lifecycle or {},
            "overlap_explained": overlap_explained,
            "telemetry_sample_gap_limit_s": sample_gap_limit_s,
        },
    )
    return receipts.seal_adoption_row(row)


def _validate(rows: list[dict[str, object]], task_pid: int = 4100) -> dict[str, object]:
    return receipts.validate_adoption_rows(
        rows,
        expected_task_id=mod.TASK_ID,
        expected_task_pid=task_pid,
    )


def test_req_report_6924_spec_declares_contract_and_fields() -> None:
    """REQ-REPORT-6924: OpenSpec owns the adoption behavior and artifact fields."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6924") :]
    for scenario in (
        "SCENARIO-REPORT-6924-PHASES",
        "SCENARIO-REPORT-6924-CPU",
        "SCENARIO-REPORT-6924-OWNERSHIP",
        "SCENARIO-REPORT-6924-GPU",
        "SCENARIO-REPORT-6924-CONCURRENCY",
        "SCENARIO-REPORT-6924-TEARDOWN",
        "SCENARIO-REPORT-6924-SERIALIZATION",
    ):
        assert scenario in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_report_6924_phases_are_monotonic_and_non_overlapping() -> None:
    """SCENARIO-REPORT-6924-PHASES: ordering and duration come from row clocks."""

    rows = [
        _row(start=100, end=200, phase="model_load"),
        _row(start=200, end=500, phase="generation"),
        _row(start=500, end=600, phase="teardown"),
    ]
    report = _validate(rows)
    assert report["accepted"] is True
    assert report["phase_order_valid"] is True
    assert report["recomputed_duration_s"] == 0.0000005

    overlapping = deepcopy(rows)
    overlapping[1]["monotonic_start_ns"] = 150
    overlapping[1] = receipts.seal_adoption_row(overlapping[1])
    rejected = _validate(overlapping)
    assert rejected["accepted"] is False
    assert "overlap_unexplained" in rejected["reasons"]


def test_scenario_report_6924_cpu_context_and_fresh_process(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6924-CPU: CPU receipts survive fresh-process validation."""

    receipt_path = tmp_path / "cpu-receipt.json"
    template = ExperimentTemplate(
        6924,
        "fixture",
        "results/unused.json",
        repo_root=tmp_path,
    )
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.05); print('cpu fixture')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    with template.task_runtime_receipts(
        receipt_path,
        task_id=mod.TASK_ID,
        control_id="cpu-fixture",
        runner_selection=_runner("cpu"),
        model_identity={
            "model_id": "cpu-fixture-model",
            "model_sha256": receipts.sha256_text("cpu-fixture-model"),
            "model_identity_bound": True,
        },
        device_ids=["CPU"],
        model_count=1,
    ) as runtime:
        with runtime.phase("queue_wait"):
            pass
        with runtime.phase(
            "model_load",
            model_id="cpu-fixture-model",
            child_pids=[process.pid],
            server_lifecycle={
                "event": "started",
                "server_id": "cpu-fixture-server",
                "pid": process.pid,
            },
        ):
            pass
        with runtime.phase(
            "generation", model_id="cpu-fixture-model", child_pids=[process.pid]
        ) as state:
            stdout, stderr = process.communicate(timeout=2)
            state["raw_output_bytes"] = stdout
            state["exit_status"] = {
                "returncode": process.returncode,
                "timed_out": False,
                "signal": None,
                "stderr_sha256": receipts.sha256_bytes(stderr),
            }
        with runtime.phase("exact_verification", model_id="cpu-fixture-model"):
            pass
        with runtime.phase(
            "teardown",
            model_id="cpu-fixture-model",
            server_lifecycle={
                "event": "teardown",
                "server_id": "cpu-fixture-server",
                "pid": process.pid,
                "process_exit_confirmed": True,
                "process_reaped": True,
                "vram_after_teardown_mb": 0,
            },
        ):
            pass
        with runtime.phase("artifact_write"):
            pass

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["validation"]["accepted"] is True
    assert len(template._phase_timings) == 6
    first_bytes = receipt_path.read_bytes()
    receipts.write_adoption_receipt(receipt_path, payload)
    assert receipt_path.read_bytes() == first_bytes

    command = [
        sys.executable,
        "-c",
        (
            "import json,sys; "
            "from carnot.task_runtime_receipts import load_and_validate_adoption_receipt as check; "
            "print(json.dumps(check(sys.argv[1], expected_task_id=sys.argv[2], "
            "expected_task_pid=int(sys.argv[3])), sort_keys=True))"
        ),
        str(receipt_path),
        mod.TASK_ID,
        str(os.getpid()),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    fresh = json.loads(completed.stdout)
    assert fresh["accepted"] is True
    assert fresh["ownership_valid"] is True
    assert fresh["teardown_complete"] is True
    assert fresh["peak_model_concurrency"] == 1


def test_scenario_report_6924_rejects_pid_and_lineage_mismatch() -> None:
    """SCENARIO-REPORT-6924-OWNERSHIP: task and child ownership fail closed."""

    row = _row(start=100, end=200, child_pid=4200)
    wrong_parent = deepcopy(row)
    wrong_parent["parent_pid"] = 4999
    wrong_parent = receipts.seal_adoption_row(wrong_parent)
    assert "task_pid_mismatch" in _validate([wrong_parent])["reasons"]

    wrong_lineage = deepcopy(row)
    wrong_lineage["process_lineage"][0]["chain"][-1] = _identity(4300, 4000)
    wrong_lineage = receipts.seal_adoption_row(wrong_lineage)
    assert "cross_process_child" in _validate([wrong_lineage])["reasons"]

    forged_hash = deepcopy(row)
    forged_hash["phase"] = "forged"
    assert "receipt_hash_mismatch" in _validate([forged_hash])["reasons"]


def test_scenario_report_6924_rejects_gpu_uuid_and_sample_gaps() -> None:
    """SCENARIO-REPORT-6924-GPU: telemetry binds PID, UUID, fields, and time gaps."""

    sample = {
        "pid": 4200,
        "device_uuid": "GPU-fixture-0",
        "pid_memory_mb": 2048,
        "device_memory_used_mb": 4096,
        "utilization_pct": 75,
        "offload_layers": 32,
        "monotonic_ns": 5_000_000_000,
        "sample_age_s": 0.0,
    }
    row = _row(
        start=4_900_000_000,
        end=5_100_000_000,
        child_pid=4200,
        device_ids=("GPU-fixture-0",),
        gpu_samples=[sample],
        sample_gap_limit_s=0.2,
    )
    assert _validate([row])["accepted"] is True

    wrong_uuid = deepcopy(row)
    wrong_uuid["gpu_samples"][0]["device_uuid"] = "GPU-forged"
    wrong_uuid = receipts.seal_adoption_row(wrong_uuid)
    assert "gpu_uuid_mismatch" in _validate([wrong_uuid])["reasons"]

    missing_utilization = deepcopy(row)
    del missing_utilization["gpu_samples"][0]["utilization_pct"]
    missing_utilization = receipts.seal_adoption_row(missing_utilization)
    assert "gpu_sample_field_missing" in _validate([missing_utilization])["reasons"]

    gap = deepcopy(row)
    gap["telemetry_sample_gap_limit_s"] = 0.01
    gap = receipts.seal_adoption_row(gap)
    assert "gpu_telemetry_gap" in _validate([gap])["reasons"]


def test_scenario_report_6924_sequential_and_concurrent_models() -> None:
    """SCENARIO-REPORT-6924-CONCURRENCY: only declared model overlap is valid."""

    sequential = [
        _row(start=100, end=300, model_id="model-a", model_count=2),
        _row(start=200, end=400, model_id="model-b", model_count=2),
    ]
    report = _validate(sequential)
    assert report["accepted"] is False
    assert report["peak_model_concurrency"] == 2
    assert "sequential_model_overlap" in report["reasons"]

    concurrent = [
        _row(
            start=100,
            end=300,
            model_id="model-a",
            model_count=2,
            concurrency_mode="concurrent",
            overlap_explained=True,
        ),
        _row(
            start=200,
            end=400,
            model_id="model-b",
            model_count=2,
            concurrency_mode="concurrent",
            overlap_explained=True,
        ),
    ]
    accepted = _validate(concurrent)
    assert accepted["accepted"] is True
    assert accepted["peak_model_concurrency"] == 2


def test_scenario_report_6924_requires_matching_teardown() -> None:
    """SCENARIO-REPORT-6924-TEARDOWN: every started server must be reaped."""

    started = _row(
        start=100,
        end=200,
        phase="model_load",
        child_pid=4200,
        server_lifecycle={"event": "started", "server_id": "server-a", "pid": 4200},
    )
    assert "missing_server_teardown" in _validate([started])["reasons"]

    teardown = _row(
        start=200,
        end=300,
        phase="teardown",
        server_lifecycle={
            "event": "teardown",
            "server_id": "server-a",
            "pid": 4200,
            "process_exit_confirmed": True,
            "process_reaped": True,
            "vram_after_teardown_mb": 0,
        },
    )
    assert _validate([started, teardown])["teardown_complete"] is True


def test_scenario_report_6924_serialization_is_deterministic(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6924-SERIALIZATION: stable rows produce stable JSON bytes."""

    rows = [_row(start=100, end=200)]
    payload = receipts.build_adoption_receipt(
        task_id=mod.TASK_ID,
        task_process_identity=_identity(4100, 4000),
        rows=rows,
        validation=_validate(rows),
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    receipts.write_adoption_receipt(first, payload)
    receipts.write_adoption_receipt(second, deepcopy(payload))
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text())["receipt_sha256"] == receipts.sha256_json(rows)


def test_req_report_6924_experiment_emits_ready_advisory_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-6924: the CPU fixture is sufficient for the advisory artifact."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.run(date="20260903", output_path=output)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert persisted == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= artifact.keys()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= artifact["field_principles"].keys()
    assert artifact["inference_substrate"] == "deterministic_runtime_receipt_fixture_no_llm"
    assert artifact["task_runtime_receipt_adoption_ready_score"] == 1
    assert artifact["fresh_process_recheck_rows"][0]["accepted"] is True
    assert artifact["forged_receipt_rejection_rows"]
    assert all(row["rejected"] for row in artifact["forged_receipt_rejection_rows"])
    assert artifact["optional_gpu_fixture_rows"][0]["status"] in {
        "not_run_no_task_owned_gpu",
        "complete",
    }
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["gate_check_summary"] == []
    assert "with tmpl.task_runtime_receipts(" in artifact["adoption_api_rows"][0]["call_pattern"]


def test_req_report_6924_precondition_failure_is_complete_blocked(tmp_path: Path) -> None:
    """REQ-REPORT-6924: a missing prerequisite records expected and observed values."""

    checks = mod.check_preconditions(tmp_path)
    artifact = mod.blocked_artifact(date="20260903", preconditions=checks)
    assert artifact["task_runtime_receipt_adoption_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "complete_blocked_task_runtime_receipt_adoption"
    assert artifact["gate_check_summary"]
    assert {
        "failed_check",
        "expected_value",
        "observed_value",
    } <= artifact["gate_check_summary"][0].keys()
