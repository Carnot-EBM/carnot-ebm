"""Tests for the V607 SOTA runtime receipt qualification.

Spec refs: REQ-REPORT-6928 and SCENARIO-REPORT-6928-*.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import types
from typing import Mapping

import pytest

from carnot import experiment_6928_sota_runtime_receipt_qualification as mod
from carnot import task_runtime_receipts as receipts


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"
GPU_UUIDS = (
    "GPU-11111111-1111-1111-1111-111111111111",
    "GPU-22222222-2222-2222-2222-222222222222",
)


def _identity(pid: int, parent_pid: int) -> dict[str, object]:
    return {
        "pid": pid,
        "parent_pid": parent_pid,
        "start_time_ticks": pid * 10,
        "boot_id": "fixture-boot",
        "cmdline_hash": receipts.sha256_text(f"process-{pid}"),
    }


def _runner() -> dict[str, object]:
    return {
        "runner_id": "llama-cpp-python-worker",
        "binary_path": sys.executable,
        "binary_sha256": receipts.sha256_file(sys.executable),
        "llama_cpp_version": "fixture",
        "supports_gpu_offload": True,
        "substrate": "cuda_gguf",
        "selected": True,
    }


def _phase_row(
    *,
    model_index: int,
    phase: str,
    start: int,
    end: int,
    task_pid: int = 7000,
    task_identity: dict[str, object] | None = None,
    gpu_uuids: tuple[str, ...] = GPU_UUIDS,
) -> dict[str, object]:
    spec = mod.MODEL_SPECS[model_index]
    child_pid = 7100 + model_index
    identity = task_identity or _identity(task_pid, 6900)
    samples = []
    if phase == "generation":
        samples = [
            {
                "pid": child_pid,
                "device_uuid": uuid,
                "pid_memory_mb": 8192,
                "device_memory_used_mb": 8200,
                "utilization_pct": 60,
                "offload_layers": -1,
                "monotonic_ns": start + (end - start) // 2,
                "sample_age_s": 0.0,
            }
            for uuid in gpu_uuids
        ]
    lifecycle: dict[str, object] = {}
    if phase == "model_load":
        lifecycle = {
            "event": "started",
            "server_id": f"worker-{model_index}",
            "pid": child_pid,
        }
    elif phase == "teardown":
        lifecycle = {
            "event": "teardown",
            "server_id": f"worker-{model_index}",
            "pid": child_pid,
            "process_exit_confirmed": True,
            "process_reaped": True,
            "vram_after_teardown_mb": 8,
        }
    row = receipts.build_phase_row(
        task_id=mod.TASK_ID,
        control_id=f"model-{model_index}",
        phase=phase,
        monotonic_start_ns=start,
        monotonic_end_ns=end,
        wall_clock_start="2026-09-03T00:00:00Z",
        wall_clock_end="2026-09-03T00:00:01Z",
        parent_pid=task_pid,
        child_pids=[child_pid],
        command=[sys.executable, "qualification-worker", spec["hf_id"]],
        config={"n_ctx": mod.CONTEXT_SIZE, "n_gpu_layers": mod.OFFLOAD_LAYERS},
        model_identity={
            **spec,
            "model_path": f"/cache/{spec['name']}-{spec['quantization']}.gguf",
            "model_sha256": receipts.sha256_text(spec["hf_id"]),
            "model_identity_bound": True,
            "cache_state": "hit",
        },
        runner_selection=_runner(),
        device_ids=list(gpu_uuids),
        concurrency_group="exp6928-sequential-models",
        raw_output_bytes=f"{spec['hf_id']}:{phase}".encode(),
        exit_status={"returncode": 0, "timed_out": False, "signal": None},
        attribution_confidence=1.0,
        gpu_samples=samples,
        extra={
            "task_process_identity": identity,
            "process_lineage": [
                {
                    "child_pid": child_pid,
                    "owned": True,
                    "chain": [_identity(child_pid, task_pid), identity],
                }
            ],
            "model_lifecycle": {
                "model_id": spec["hf_id"],
                "model_count": len(mod.MODEL_SPECS),
                "concurrency_mode": "sequential",
            },
            "server_lifecycle": lifecycle,
            "overlap_explained": False,
            "telemetry_sample_gap_limit_s": 5.0,
            "phase_metadata": {"child_command": ["qualification-worker"]},
        },
    )
    return receipts.seal_adoption_row(row)


def _valid_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    clock = 1_000_000_000
    for index in range(3):
        rows.extend(
            [
                _phase_row(
                    model_index=index,
                    phase="model_load",
                    start=clock,
                    end=clock + 100_000_000,
                ),
                _phase_row(
                    model_index=index,
                    phase="generation",
                    start=clock + 100_000_000,
                    end=clock + 300_000_000,
                ),
                _phase_row(
                    model_index=index,
                    phase="teardown",
                    start=clock + 300_000_000,
                    end=clock + 400_000_000,
                ),
            ]
        )
        clock += 500_000_000
    return rows


def _write_receipt(path: Path, rows: list[dict[str, object]]) -> None:
    validation = receipts.validate_adoption_rows(
        rows,
        expected_task_id=mod.TASK_ID,
        expected_task_pid=7000,
    )
    payload = receipts.build_adoption_receipt(
        task_id=mod.TASK_ID,
        task_process_identity=_identity(7000, 6900),
        rows=rows,
        validation=validation,
    )
    receipts.write_adoption_receipt(path, payload)


def test_req_report_6928_spec_models_and_required_fields() -> None:
    """REQ-REPORT-6928: OpenSpec fixes the model set and artifact contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6928") :]
    assert [row["hf_id"] for row in mod.MODEL_SPECS] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]
    for scenario in (
        "SCENARIO-REPORT-6928-OWNERSHIP",
        "SCENARIO-REPORT-6928-PHASES",
        "SCENARIO-REPORT-6928-TEARDOWN",
        "SCENARIO-REPORT-6928-CACHE",
        "SCENARIO-REPORT-6928-REPLAY",
        "SCENARIO-REPORT-6928-ARTIFACT",
    ):
        assert scenario in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "AutoTokenizer" not in source


def test_scenario_report_6928_cache_hit_and_miss_are_explicit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6928-CACHE: paths and misses remain auditable."""

    paths: dict[str, str | None] = {}
    for index, spec in enumerate(mod.MODEL_SPECS):
        path = tmp_path / f"model-{index}-{spec['quantization']}.gguf"
        path.write_bytes(f"model-{index}".encode())
        paths[spec["hf_id"]] = str(path)

    files, checks = mod.resolve_model_files(resolver=lambda hf_id, _quant: paths[hf_id])
    assert [row["cache_state"] for row in files] == ["hit", "hit", "hit"]
    assert all(row["model_sha256"].startswith("sha256:") for row in files)
    assert all(row["size_bytes"] > 0 for row in files)
    assert [row["model_file"] for row in files] == [
        f"model-{index}-{spec['quantization']}.gguf" for index, spec in enumerate(mod.MODEL_SPECS)
    ]
    assert all(row["passed"] for row in checks)

    paths[mod.MODEL_SPECS[1]["hf_id"]] = None
    files, checks = mod.resolve_model_files(resolver=lambda hf_id, _quant: paths[hf_id])
    assert files[1]["cache_state"] == "miss"
    assert files[1]["model_path"] is None
    assert checks[1]["passed"] is False
    assert checks[1]["expected_value"] == "cached_or_resolvable_gguf"


def test_scenario_report_6928_valid_rows_prove_sequential_dual_gpu_work() -> None:
    """SCENARIO-REPORT-6928-PHASES: three valid lifecycles peak at one model."""

    report = mod.qualify_receipt_rows(
        _valid_rows(), expected_task_pid=7000, expected_gpu_uuids=GPU_UUIDS
    )
    assert report["accepted"] is True
    assert report["phase_order_valid"] is True
    assert report["ownership_valid"] is True
    assert report["device_ownership_valid"] is True
    assert report["sequential_model_lifecycle_valid"] is True
    assert report["peak_model_concurrency"] == 1
    assert report["teardown_complete"] is True
    assert report["qualified_model_count"] == 3
    assert report["recomputed_duration_s"] == 1.2


def test_scenario_report_6928_rejects_pid_lineage_and_copied_rows() -> None:
    """SCENARIO-REPORT-6928-OWNERSHIP: task, lineage, and task IDs are bound."""

    rows = _valid_rows()
    wrong_pid = deepcopy(rows)
    wrong_pid[0]["parent_pid"] = 7999
    wrong_pid[0] = receipts.seal_adoption_row(wrong_pid[0])
    assert (
        "task_pid_mismatch"
        in mod.qualify_receipt_rows(
            wrong_pid, expected_task_pid=7000, expected_gpu_uuids=GPU_UUIDS
        )["reasons"]
    )

    wrong_lineage = deepcopy(rows)
    wrong_lineage[1]["process_lineage"][0]["chain"][-1] = _identity(7999, 6900)
    wrong_lineage[1] = receipts.seal_adoption_row(wrong_lineage[1])
    assert (
        "cross_process_child"
        in mod.qualify_receipt_rows(
            wrong_lineage, expected_task_pid=7000, expected_gpu_uuids=GPU_UUIDS
        )["reasons"]
    )

    copied = deepcopy(rows)
    copied[0]["task_id"] = "exp6924-task-runtime-receipt-adoption"
    copied[0] = receipts.seal_adoption_row(copied[0])
    assert (
        "task_id_mismatch"
        in mod.qualify_receipt_rows(copied, expected_task_pid=7000, expected_gpu_uuids=GPU_UUIDS)[
            "reasons"
        ]
    )

    cross_process = deepcopy(rows)
    cross_process[4]["task_process_identity"] = _identity(7001, 6900)
    cross_process[4] = receipts.seal_adoption_row(cross_process[4])
    reasons = mod.qualify_receipt_rows(
        cross_process, expected_task_pid=7000, expected_gpu_uuids=GPU_UUIDS
    )["reasons"]
    assert "task_identity_mismatch" in reasons
    assert "cross_process_receipt" in reasons


def test_scenario_report_6928_rejects_uuid_and_residency_mismatch() -> None:
    """SCENARIO-REPORT-6928-OWNERSHIP: both GPU identities need PID residency."""

    wrong_uuid = deepcopy(_valid_rows())
    wrong_uuid[1]["gpu_samples"][0]["device_uuid"] = "GPU-foreign"
    wrong_uuid[1] = receipts.seal_adoption_row(wrong_uuid[1])
    reasons = mod.qualify_receipt_rows(
        wrong_uuid, expected_task_pid=7000, expected_gpu_uuids=GPU_UUIDS
    )["reasons"]
    assert "gpu_uuid_mismatch" in reasons
    assert "gpu_device_ownership_incomplete" in reasons

    missing_residency = deepcopy(_valid_rows())
    missing_residency[1]["gpu_samples"][0]["pid_memory_mb"] = 0
    missing_residency[1] = receipts.seal_adoption_row(missing_residency[1])
    assert (
        "gpu_device_ownership_incomplete"
        in mod.qualify_receipt_rows(
            missing_residency,
            expected_task_pid=7000,
            expected_gpu_uuids=GPU_UUIDS,
        )["reasons"]
    )

    unresolved = mod.qualify_receipt_rows(
        _valid_rows(), expected_task_pid=7000, expected_gpu_uuids=("", GPU_UUIDS[1])
    )
    assert "unresolved_gpu_identity" in unresolved["reasons"]


def test_scenario_report_6928_rejects_overlap_rollback_and_missing_teardown() -> None:
    """SCENARIO-REPORT-6928-TEARDOWN: clocks and exit evidence fail closed."""

    missing = [
        row
        for row in _valid_rows()
        if not (
            row["model_identity"]["hf_id"] == mod.MODEL_SPECS[2]["hf_id"]
            and row["phase"] == "teardown"
        )
    ]
    assert (
        "missing_server_teardown"
        in mod.qualify_receipt_rows(missing, expected_task_pid=7000, expected_gpu_uuids=GPU_UUIDS)[
            "reasons"
        ]
    )

    overlap = deepcopy(_valid_rows())
    overlap[3]["monotonic_start_ns"] = overlap[2]["monotonic_start_ns"]
    overlap[3] = receipts.seal_adoption_row(overlap[3])
    reasons = mod.qualify_receipt_rows(
        overlap, expected_task_pid=7000, expected_gpu_uuids=GPU_UUIDS
    )["reasons"]
    assert "overlap_unexplained" in reasons
    assert "sequential_model_overlap" in reasons

    rollback = deepcopy(_valid_rows())
    rollback[0]["monotonic_end_ns"] = rollback[0]["monotonic_start_ns"] - 1
    rollback[0] = receipts.seal_adoption_row(rollback[0])
    assert (
        "invalid_monotonic_interval"
        in mod.qualify_receipt_rows(rollback, expected_task_pid=7000, expected_gpu_uuids=GPU_UUIDS)[
            "reasons"
        ]
    )


def test_scenario_report_6928_fresh_process_replays_serialized_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6928-REPLAY: a new interpreter recomputes every gate."""

    path = tmp_path / "task-receipt.json"
    _write_receipt(path, _valid_rows())
    fresh = mod.fresh_process_recheck(path, task_pid=7000, gpu_uuids=GPU_UUIDS)
    assert fresh["accepted"] is True
    assert fresh["returncode"] == 0
    assert fresh["device_ownership_valid"] is True
    assert fresh["sequential_model_lifecycle_valid"] is True
    assert fresh["teardown_complete"] is True
    assert fresh["recomputed_duration_s"] == 1.2


def test_scenario_report_6928_forgery_matrix_rejects_critical_attacks() -> None:
    """SCENARIO-REPORT-6928-REPLAY: copied and changed evidence fails closed."""

    attacks = mod.forged_receipt_rejection_rows(_valid_rows(), task_pid=7000, gpu_uuids=GPU_UUIDS)
    assert {row["attack_id"] for row in attacks} == {
        "copied_foreign_task_receipt",
        "cross_process_receipt",
        "gpu_uuid_mismatch",
        "missing_teardown",
    }
    assert all(row["rejected"] for row in attacks)
    assert all(row["reasons"] for row in attacks)


def test_req_report_6928_blocked_artifact_is_schema_complete() -> None:
    """REQ-REPORT-6928: any failed precondition emits the exact blocked verdict."""

    preconditions = [
        {
            "check": "dual_authenticated_rtx3090_cuda_devices",
            "expected_value": 2,
            "observed_value": 1,
            "passed": False,
        }
    ]
    artifact = mod.blocked_artifact(date="20260903", preconditions=preconditions)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= artifact.keys()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= artifact["field_principles"].keys()
    assert artifact["inference_substrate"] == "task_owned_local_gguf_cuda_inference"
    assert artifact["sota_runtime_receipt_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_sota_runtime_receipt_qualification"
    assert artifact["gate_check_summary"] == [
        {
            "failed_check": "dual_authenticated_rtx3090_cuda_devices",
            "expected_value": 2,
            "observed_value": 1,
        }
    ]


def test_req_report_6928_run_stops_at_failed_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6928: blocked preflight writes once and starts no model."""

    output = tmp_path / "blocked.json"
    preflight = {
        "checks": [
            {
                "check": "model_cache:missing",
                "expected_value": "cached_or_resolvable_gguf",
                "observed_value": "missing",
                "passed": False,
            }
        ],
        "model_files": [],
        "gpus": [],
    }
    monkeypatch.setattr(mod, "check_preconditions", lambda **_kwargs: preflight)
    monkeypatch.setattr(
        mod,
        "execute_live_models",
        lambda *_args, **_kwargs: pytest.fail("models must not start"),
    )
    artifact = mod.run(date="20260903", output_path=output, repo_root=tmp_path)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "blocked_sota_runtime_receipt_qualification"


def test_host_probe_helpers_parse_success_and_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6928: bounded host probes retain honest terminal states."""

    ok = mod._run_command([sys.executable, "-c", "print('ok')"])
    assert ok["ok"] is True
    assert ok["stdout"].strip() == "ok"
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing")),
    )
    assert mod._run_command(["missing"])["ok"] is False

    gpu_stdout = (
        f"0, NVIDIA GeForce RTX 3090, {GPU_UUIDS[0]}, 24576, 4, 0\n"
        f"1, NVIDIA GeForce RTX 3090, {GPU_UUIDS[1]}, 24576, 5, 1\n"
        "bad,row\n2, broken, GPU-x, nope, 2, 3\n"
    )
    monkeypatch.setattr(
        mod,
        "_run_command",
        lambda *_args, **_kwargs: {
            "ok": True,
            "returncode": 0,
            "stdout": gpu_stdout,
            "stderr": "",
        },
    )
    assert [row["index"] for row in mod.query_gpus()] == [0, 1]
    monkeypatch.setattr(
        mod,
        "_run_command",
        lambda *_args, **_kwargs: {"ok": False, "stdout": ""},
    )
    assert mod.query_gpus() == []

    backend = types.SimpleNamespace(llama_supports_gpu_offload=lambda: True)
    package = types.ModuleType("llama_cpp")
    package.__file__ = str(tmp_path / "llama_cpp.py")
    package.__version__ = "fixture-version"
    package.llama_cpp = backend
    monkeypatch.setitem(sys.modules, "llama_cpp", package)
    assert mod.llama_cpp_status()["supports_gpu_offload"] is True


def test_resolver_exception_and_quantization_fallback_are_explicit(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6928-CACHE: resolver errors and exact tokens are stable."""

    quantized = tmp_path / "fixture-UD-Q5_K_M.gguf"
    quantized.write_bytes(b"weights")
    assert mod._exact_quantization(quantized, "Q4_K_M") == "UD-Q5_K_M"
    assert mod._exact_quantization(tmp_path / "unknown.gguf", "Q4_K_M") == "Q4_K_M"

    def broken(_hf_id: str, _quant: str) -> str | None:
        raise RuntimeError("resolver unavailable")

    files, checks = mod.resolve_model_files(resolver=broken)
    assert all(row["cache_state"] == "miss" for row in files)
    assert all("resolver unavailable" in str(row["resolver_error"]) for row in files)
    assert all(row["passed"] is False for row in checks)


def test_check_preconditions_combines_models_gpus_cuda_and_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6928: the preflight combines each required live resource."""

    model_files = [{**spec, "cache_state": "hit"} for spec in mod.MODEL_SPECS]
    model_checks = [mod._check(f"model:{index}", "hit", "hit", True) for index in range(3)]
    gpus = [
        {
            "index": index,
            "name": "NVIDIA GeForce RTX 3090",
            "uuid": GPU_UUIDS[index],
            "memory_total_mb": 24576,
        }
        for index in range(2)
    ]
    monkeypatch.setattr(mod, "resolve_model_files", lambda: (model_files, model_checks))
    monkeypatch.setattr(mod, "query_gpus", lambda: gpus)
    monkeypatch.setattr(
        mod,
        "llama_cpp_status",
        lambda: {"importable": True, "supports_gpu_offload": True},
    )
    monkeypatch.setattr(
        mod.shutil,
        "disk_usage",
        lambda _path: shutil_usage(total=10, used=2, free=8 * 1024**3),
    )
    helper = tmp_path / "python/carnot/task_runtime_receipts.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("fixture", encoding="utf-8")
    output = tmp_path / "results/out.json"
    output.parent.mkdir()
    result = mod.check_preconditions(repo_root=tmp_path, output_path=output)
    assert all(row["passed"] for row in result["checks"])
    assert [row["uuid"] for row in result["gpus"]] == list(GPU_UUIDS)


def shutil_usage(*, total: int, used: int, free: int) -> object:
    """Return the named fields used from a disk-usage result."""

    return types.SimpleNamespace(total=total, used=used, free=free)


def test_payload_hash_and_duplicate_gpu_identity_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6928-REPLAY: top-level hashes and UUID uniqueness are checked."""

    path = tmp_path / "receipt.json"
    _write_receipt(path, _valid_rows())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["receipt_sha256"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = mod.replay_task_runtime_receipt(path, task_pid=7000, gpu_uuids=GPU_UUIDS)
    assert "receipt_payload_hash_mismatch" in report["reasons"]
    duplicate = mod.qualify_receipt_rows(
        _valid_rows(), expected_task_pid=7000, expected_gpu_uuids=(GPU_UUIDS[0],) * 2
    )
    assert "unresolved_gpu_identity" in duplicate["reasons"]

    unbound = deepcopy(_valid_rows())
    unbound[0]["model_identity"]["cache_state"] = "miss"
    unbound[0] = receipts.seal_adoption_row(unbound[0])
    reasons = mod.qualify_receipt_rows(
        unbound, expected_task_pid=7000, expected_gpu_uuids=GPU_UUIDS
    )["reasons"]
    assert any(reason.startswith("model_cache_not_bound:") for reason in reasons)

    incomplete_order = [
        row
        for row in _valid_rows()
        if row["model_identity"]["hf_id"] != mod.MODEL_SPECS[0]["hf_id"]
    ]
    assert (
        "model_load_order_mismatch"
        in mod.qualify_receipt_rows(
            incomplete_order,
            expected_task_pid=7000,
            expected_gpu_uuids=GPU_UUIDS,
        )["reasons"]
    )


def test_fresh_process_failure_is_a_terminal_replay_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6928-REPLAY: an interpreter failure cannot pass replay."""

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=2, stdout="", stderr="boom"),
    )
    report = mod.fresh_process_recheck(tmp_path / "missing.json", task_pid=1, gpu_uuids=GPU_UUIDS)
    assert report["accepted"] is False
    assert report["reasons"] == ["fresh_process_replay_failed"]


def test_worker_protocol_success_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-6928: the child exposes deterministic load, generation, and close boundaries."""

    closed: list[bool] = []

    class FakeLlama:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["tensor_split"] == [0.5, 0.5]

        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"choices": [{"text": "4"}], "usage": {"completion_tokens": 1}}

        def close(self) -> None:
            closed.append(True)

    package = types.ModuleType("llama_cpp")
    package.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", package)
    monkeypatch.setattr(sys, "stdin", io.StringIO("load\ngenerate\nclose\n"))
    assert mod.worker_main(model_path="fixture.gguf", context=32, offload_layers=-1, seed=7) == 0
    events = [json.loads(line)["event"] for line in capsys.readouterr().out.splitlines()]
    assert events == ["started", "loaded", "generated", "closed"]
    assert closed == [True]

    monkeypatch.setattr(sys, "stdin", io.StringIO("wrong\n"))
    assert mod.worker_main(model_path="fixture.gguf", context=32, offload_layers=-1, seed=7) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["event"] == "error"

    class FailingLlama(FakeLlama):
        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            raise RuntimeError("generation failed")

    package.Llama = FailingLlama
    monkeypatch.setattr(sys, "stdin", io.StringIO("load\ngenerate\n"))
    assert mod.worker_main(model_path="fixture.gguf", context=32, offload_layers=-1, seed=7) == 1
    assert closed[-1] is True
    capsys.readouterr()

    package.Llama = FakeLlama
    monkeypatch.setattr(sys, "stdin", io.StringIO("load\nwrong\n"))
    assert mod.worker_main(model_path="fixture.gguf", context=32, offload_layers=-1, seed=7) == 1
    monkeypatch.setattr(sys, "stdin", io.StringIO("load\ngenerate\nwrong\n"))
    assert mod.worker_main(model_path="fixture.gguf", context=32, offload_layers=-1, seed=7) == 1
    capsys.readouterr()


class FakeProcess:
    """Small pipe-compatible process used to test parent protocol failures."""

    def __init__(self, *, pid: int = 4242, stdin: io.StringIO | None = None) -> None:
        self.pid = pid
        self.stdin = stdin if stdin is not None else io.StringIO()
        self.stdout: io.StringIO | None = io.StringIO()
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def test_parent_worker_pipe_helpers_cover_terminal_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6928: malformed, missing, and failed child events stay visible."""

    process = FakeProcess()
    mod._send_worker_command(process, "load")  # type: ignore[arg-type]
    assert process.stdin.getvalue() == "load\n"
    process.stdin = None
    with pytest.raises(RuntimeError, match="stdin"):
        mod._send_worker_command(process, "load")  # type: ignore[arg-type]

    process.stdout = io.StringIO('{"event":"loaded"}\n')
    monkeypatch.setattr(mod.select, "select", lambda *_args: ([process.stdout], [], []))
    assert mod._read_worker_event(process, timeout_s=0.1) == {  # type: ignore[arg-type]
        "event": "loaded"
    }
    process.stdout = io.StringIO("not-json\n")
    with pytest.raises(RuntimeError, match="non-JSON"):
        mod._read_worker_event(process, timeout_s=0.1)  # type: ignore[arg-type]
    process.stdout = io.StringIO("[]\n")
    with pytest.raises(RuntimeError, match="not an object"):
        mod._read_worker_event(process, timeout_s=0.1)  # type: ignore[arg-type]
    process.stdout = io.StringIO()
    monkeypatch.setattr(mod.select, "select", lambda *_args: ([], [], []))
    assert mod._read_worker_event(process, timeout_s=0.1) is None  # type: ignore[arg-type]
    monkeypatch.setattr(mod.select, "select", lambda *_args: ([process.stdout], [], []))
    assert mod._read_worker_event(process, timeout_s=0.1) is None  # type: ignore[arg-type]
    process.stdout = None
    with pytest.raises(RuntimeError, match="stdout"):
        mod._read_worker_event(process, timeout_s=0.1)  # type: ignore[arg-type]


def test_wait_helpers_accept_expected_and_reject_error_or_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6928: parent waits are bounded and child errors fail closed."""

    process = FakeProcess()
    events = iter([None, {"event": "loaded", "pid": process.pid}])
    monkeypatch.setattr(mod, "_read_worker_event", lambda *_args, **_kwargs: next(events))
    assert mod._wait_worker_event(process, "loaded", timeout_s=1)["pid"] == process.pid  # type: ignore[arg-type]

    monkeypatch.setattr(
        mod,
        "_read_worker_event",
        lambda *_args, **_kwargs: {"event": "error", "stage": "load", "error": "x"},
    )
    with pytest.raises(RuntimeError, match="worker error"):
        mod._wait_worker_event(process, "loaded", timeout_s=1)  # type: ignore[arg-type]

    process.returncode = 3
    monkeypatch.setattr(mod, "_read_worker_event", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="exited before"):
        mod._wait_worker_event(process, "loaded", timeout_s=1)  # type: ignore[arg-type]

    process.returncode = None
    clocks = iter([0.0, 0.0, 0.0, 2.0])
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(clocks))
    with pytest.raises(TimeoutError, match="waiting"):
        mod._wait_worker_event(process, "loaded", timeout_s=1)  # type: ignore[arg-type]


def test_gpu_sampler_and_generation_wait_join_pid_uuid_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-6928-OWNERSHIP: telemetry joins task PID to both UUIDs."""

    outputs = iter(
        [
            {
                "ok": True,
                "stdout": f"{GPU_UUIDS[0]}, 9000, 40\n{GPU_UUIDS[1]}, 8000, 50\nbad\nGPU-x, nope, bad\n",
            },
            {
                "ok": True,
                "stdout": f"4242, {GPU_UUIDS[0]}, 8990\n4242, {GPU_UUIDS[1]}, 7990\nother, bad, row\n4242, GPU-x, nope\n",
            },
        ]
    )
    monkeypatch.setattr(mod, "_run_command", lambda *_args, **_kwargs: next(outputs))
    gpus = [{"uuid": GPU_UUIDS[0]}, {"uuid": GPU_UUIDS[1]}]
    samples = mod.sample_gpu_telemetry(4242, gpus)
    assert [row["pid_memory_mb"] for row in samples] == [8990, 7990]
    assert [row["utilization_pct"] for row in samples] == [40, 50]

    process = FakeProcess()
    monkeypatch.setattr(mod, "sample_gpu_telemetry", lambda *_args: samples)
    events = iter([None, {"event": "generated", "text": "4"}])
    monkeypatch.setattr(mod, "_read_worker_event", lambda *_args, **_kwargs: next(events))
    event, collected = mod._wait_generation(process, gpus, timeout_s=1)  # type: ignore[arg-type]
    assert event["text"] == "4"
    assert len(collected) == 4

    post = mod._post_teardown_gpu_state(4242, gpus)
    assert post["child_pid_absent_from_compute_apps"] is False
    assert post["vram_after_teardown_mb"] == 17000

    process.returncode = 2
    monkeypatch.setattr(mod, "_read_worker_event", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="exited during generation"):
        mod._wait_generation(process, gpus, timeout_s=1)  # type: ignore[arg-type]

    process.returncode = None
    monkeypatch.setattr(
        mod,
        "_read_worker_event",
        lambda *_args, **_kwargs: {
            "event": "error",
            "stage": "generation",
            "error": "failed",
        },
    )
    with pytest.raises(RuntimeError, match="worker error"):
        mod._wait_generation(process, gpus, timeout_s=1)  # type: ignore[arg-type]

    clocks = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(clocks))
    monkeypatch.setattr(mod, "_read_worker_event", lambda *_args, **_kwargs: None)
    with pytest.raises(TimeoutError, match="generation timed out"):
        mod._wait_generation(process, gpus, timeout_s=1)  # type: ignore[arg-type]


class FakeRuntime:
    """Context that writes a minimal receipt after the parent phases complete."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> "FakeRuntime":
        return self

    def __exit__(self, *_args: object) -> None:
        self.path.write_text(json.dumps({"rows": [{"phase": "fixture"}]}), encoding="utf-8")

    @contextmanager
    def phase(self, _name: str, **_kwargs: object):
        yield {}


class FakeTemplate:
    """Template surface needed by the one-model orchestrator."""

    def __init__(self) -> None:
        self._phase_timings = [{"name": "fixture", "elapsed_s": 0.1}]

    def task_runtime_receipts(self, path: Path, **_kwargs: object) -> FakeRuntime:
        return FakeRuntime(path)


class KillRequiredProcess(FakeProcess):
    """Process that ignores terminate until the parent uses kill."""

    def __init__(self) -> None:
        super().__init__(pid=6262)
        self.wait_count = 0

    def terminate(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        self.wait_count += 1
        if self.wait_count == 1:
            raise subprocess.TimeoutExpired("worker", timeout or 0)
        self.returncode = -9
        return self.returncode


def _resolved_models(tmp_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(mod.MODEL_SPECS):
        path = tmp_path / f"model-{index}-Q4_K_M.gguf"
        path.write_bytes(b"x")
        rows.append(
            {
                **spec,
                "model_path": str(path),
                "model_file": path.name,
                "model_sha256": receipts.sha256_file(path),
                "cache_state": "hit",
            }
        )
    return rows


def test_one_model_orchestrator_records_success_and_start_pid_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6928: one worker records command, output, teardown, and errors."""

    process = FakeProcess()
    monkeypatch.setattr(mod, "_new_template", lambda _path: FakeTemplate())
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *_args, **_kwargs: process)

    def event(_process: object, expected: str, **_kwargs: object) -> dict[str, object]:
        return {"event": expected, "pid": process.pid}

    monkeypatch.setattr(mod, "_wait_worker_event", event)
    monkeypatch.setattr(
        mod,
        "_wait_generation",
        lambda *_args, **_kwargs: (
            {"event": "generated", "text": "4", "usage": {"completion_tokens": 1}},
            [],
        ),
    )
    monkeypatch.setattr(
        mod,
        "_post_teardown_gpu_state",
        lambda *_args: {
            "child_pid_absent_from_compute_apps": True,
            "per_device_memory_used_mb": {uuid: 4 for uuid in GPU_UUIDS},
            "vram_after_teardown_mb": 8,
        },
    )
    models = _resolved_models(tmp_path)
    gpus = [{"uuid": uuid} for uuid in GPU_UUIDS]
    row = mod._execute_one_model(
        models[0],
        gpus,
        {"version": "x", "supports_gpu_offload": True, "module_path": __file__},
        tmp_path,
        0,
    )
    assert row["status"] == "complete"
    assert row["generated_text"] == "4"
    assert row["rows"] == [{"phase": "fixture"}]
    assert row["command"][0] == sys.executable

    failed_process = FakeProcess(pid=5252)
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *_args, **_kwargs: failed_process)
    monkeypatch.setattr(
        mod,
        "_wait_worker_event",
        lambda *_args, **_kwargs: {"event": "started", "pid": 9999},
    )
    failed = mod._execute_one_model(
        models[1],
        gpus,
        {"version": "x", "supports_gpu_offload": True, "module_path": __file__},
        tmp_path,
        1,
    )
    assert failed["status"] == "failed"
    assert "start PID" in failed["error"]

    kill_process = KillRequiredProcess()
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *_args, **_kwargs: kill_process)
    killed = mod._execute_one_model(
        models[2],
        gpus,
        {"version": "x", "supports_gpu_offload": True, "module_path": __file__},
        tmp_path,
        2,
    )
    assert killed["returncode"] == -9


def test_execute_live_models_combines_rows_and_writes_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6928-PHASES: three one-model runs combine in order."""

    rows = _valid_rows()

    def execute(
        model: Mapping[str, object],
        _gpus: object,
        _llama: object,
        _work_dir: object,
        index: int,
    ) -> dict[str, object]:
        return {
            "hf_id": model["hf_id"],
            "status": "complete",
            "rows": rows[index * 3 : index * 3 + 3],
        }

    monkeypatch.setattr(mod, "_execute_one_model", execute)
    monkeypatch.setattr(mod.os, "getpid", lambda: 7000)
    monkeypatch.setattr(mod.receipts, "read_process_identity", lambda _pid: _identity(7000, 6900))
    result = mod.execute_live_models(
        _resolved_models(tmp_path),
        [{"uuid": uuid} for uuid in GPU_UUIDS],
        {},
        tmp_path,
    )
    assert result["validation"]["accepted"] is True
    assert result["receipt_path"].is_file()
    assert len(result["model_rows"]) == 3

    monkeypatch.setattr(mod.receipts, "read_process_identity", lambda _pid: None)
    with pytest.raises(RuntimeError, match="identity became unavailable"):
        mod.execute_live_models(
            _resolved_models(tmp_path),
            [{"uuid": uuid} for uuid in GPU_UUIDS],
            {},
            tmp_path,
        )


def test_runtime_gate_rows_report_each_failed_condition() -> None:
    """SCENARIO-REPORT-6928-ARTIFACT: runtime blockers retain expected and observed values."""

    gates = mod._runtime_gate_rows(
        [{"hf_id": "failed-model", "status": "failed"}],
        {"accepted": False, "reasons": ["bad-receipt"]},
        {"accepted": False, "reasons": ["bad-replay"]},
        [{"attack_id": "copy", "rejected": False}],
    )
    assert {row["failed_check"] for row in gates} == {
        "all_three_model_runs_complete",
        "task_owned_cuda_receipt_validation",
        "fresh_process_receipt_replay",
        "forged_receipts_fail_closed",
    }
    assert (
        mod._runtime_gate_rows(
            [{"status": "complete"}],
            {"accepted": True},
            {"accepted": True},
            [{"rejected": True}],
        )
        == []
    )


def test_run_builds_ready_artifact_from_live_execution_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6928-ARTIFACT: successful rows alone produce advisory readiness."""

    models = _resolved_models(tmp_path)
    rows = _valid_rows()
    gpus = [{"uuid": uuid} for uuid in GPU_UUIDS]
    checks = [mod._check("all", True, True, True)]
    monkeypatch.setattr(
        mod,
        "check_preconditions",
        lambda **_kwargs: {
            "checks": checks,
            "model_files": models,
            "gpus": gpus,
            "llama_cpp": {"supports_gpu_offload": True},
        },
    )
    model_rows = [
        {
            "hf_id": spec["hf_id"],
            "status": "complete",
            "phase_timings": [{"name": "generation", "elapsed_s": 0.2}],
            "runner_selection": _runner(),
        }
        for spec in mod.MODEL_SPECS
    ]
    validation = mod.qualify_receipt_rows(
        rows, expected_task_pid=7000, expected_gpu_uuids=GPU_UUIDS
    )
    monkeypatch.setattr(
        mod,
        "execute_live_models",
        lambda *_args, **_kwargs: {
            "task_pid": 7000,
            "model_rows": model_rows,
            "rows": rows,
            "validation": validation,
            "task_runtime_receipt": {"rows": rows},
            "receipt_path": tmp_path / "receipt.json",
        },
    )
    monkeypatch.setattr(
        mod,
        "fresh_process_recheck",
        lambda *_args, **_kwargs: {
            "accepted": True,
            "peak_model_concurrency": 1,
            "sequential_model_lifecycle_valid": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "forged_receipt_rejection_rows",
        lambda *_args, **_kwargs: [{"attack_id": "all", "rejected": True}],
    )
    monkeypatch.setattr(mod, "_source_hashes", lambda _root: {"source": "sha256:x"})
    output = tmp_path / "ready.json"
    artifact = mod.run(date="20260903", output_path=output, repo_root=tmp_path)
    assert artifact["sota_runtime_receipt_ready_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"] == "complete_sota_runtime_receipt_qualified"
    assert artifact["gate_check_summary"] == []
    assert len(artifact["task_gpu_telemetry_rows"]) == 6
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_main_routes_worker_errors_and_dated_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-6928: the CLI separates internal workers from dated runs."""

    monkeypatch.setattr(mod, "worker_main", lambda **_kwargs: 7)
    assert mod.main(["--worker", "--model-path", "fixture.gguf"]) == 7
    with pytest.raises(SystemExit):
        mod.main(["--worker"])
    with pytest.raises(SystemExit):
        mod.main([])
    monkeypatch.setattr(
        mod,
        "run",
        lambda **_kwargs: {
            "honest_verdict": "complete_fixture",
            "sota_runtime_receipt_ready_score": 1,
        },
    )
    assert mod.main(["--date", "20260903", "--output", str(tmp_path / "x")]) == 0
    assert json.loads(capsys.readouterr().out)["sota_runtime_receipt_ready_score"] == 1
