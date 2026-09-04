"""Tests for the lease-aware three-family GGUF runtime handoff.

Spec refs: REQ-INFRA-6973, SCENARIO-INFRA-6973-LEASE,
SCENARIO-INFRA-6973-STALE, SCENARIO-INFRA-6973-SERVER,
SCENARIO-INFRA-6973-TEARDOWN, and SCENARIO-INFRA-6973-BARE-READINESS.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from carnot import experiment_6973_lease_aware_gguf_runtime as exp


REPO = Path(__file__).resolve().parents[2]


def _resolved_specs(tmp_path: Path) -> list[dict]:
    rows = []
    for index, model_id in enumerate(exp.REQUIRED_MODEL_IDS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(b"GGUF" + bytes([index]))
        rows.append(
            {
                "name": model_id.rsplit("/", 1)[-1].removesuffix("-GGUF"),
                "hf_id": model_id,
                "model_path": str(path),
                "gpu_indices": [0, 1],
                "headline_eligible": True,
                "preferred_quant": "Q4_K_M",
                "resolution_method": "test",
            }
        )
    return rows


def _gpu_probe(*, foreign: bool = False) -> dict:
    devices = [
        {
            "index": index,
            "uuid": f"GPU-{index}",
            "name": "NVIDIA GeForce RTX 3090",
            "memory_total_mb": 24576,
            "memory_used_mb": 4,
            "memory_free_mb": 24572,
            "utilization_gpu_pct": 0,
            "temperature_c": 40 + index,
        }
        for index in range(2)
    ]
    processes = (
        [{"pid": 991, "gpu_uuid": "GPU-0", "used_memory_mb": 9000, "process_name": "foreign"}]
        if foreign
        else []
    )
    return {"query_ok": True, "devices": devices, "processes": processes, "raw_receipts": []}


def _released_ledger_rows() -> list[dict]:
    return [
        {
            "device_uuid": f"GPU-{index}",
            "journal_path": f"/tmp/device-{index}.journal.json",
            "readable": True,
            "classification": "released",
            "owner_live": False,
            "foreign_live": False,
            "signals_sent": [],
            "document": {"released": True, "phase": "terminal_complete"},
        }
        for index in range(2)
    ]


def _passing_preflight(tmp_path: Path) -> dict:
    return exp.collect_preconditions(
        model_specs=_resolved_specs(tmp_path),
        checkpoint_path=tmp_path / "checkpoint.json",
        gpu_probe=lambda: _gpu_probe(),
        llama_probe=lambda: {"importable": True, "gpu_offload": True, "version": "test"},
        ledger_probe=lambda _devices: _released_ledger_rows(),
        metadata_probe=lambda model: {
            "model_id": model["hf_id"],
            "passed": True,
            "tokenizer_source": "embedded_gguf",
            "metadata": {"general.architecture": "test"},
            "metadata_hash": exp.sha256_text("metadata"),
        },
    )


def _lease_row(model_id: str, device_uuid: str) -> dict:
    return {
        "model_id": model_id,
        "device_uuid": device_uuid,
        "task_id": exp.EXPERIMENT_ID,
        "owner_pid": 444,
        "owner_pid_start_ticks": 555,
        "owner_verified": True,
        "expected_model": f"/{model_id}.gguf",
        "journal_before_acquisition": {"released": True},
        "journal_after_acquisition": {"released": False, "phase": "loading"},
        "journal_after_release": {"released": True, "phase": "terminal_complete"},
        "journal_validation_errors": [],
        "recovery": {"performed": False, "signals_sent": []},
        "release_receipt": {"released": True, "signals_sent": []},
        "signals_sent": [],
        "consistent": True,
    }


def _generation_row(model_id: str, *, terminal: str = "complete") -> dict:
    good = terminal == "complete"
    return {
        "model_id": model_id,
        "terminal_state": terminal,
        "config_id": exp.LOAD_CONFIG["config_id"],
        "load_config": deepcopy(exp.LOAD_CONFIG),
        "live_cuda": good,
        "gpu_uuids_used": ["GPU-0", "GPU-1"] if good else [],
        "output": "CANARY" if good else "",
        "output_hash": exp.sha256_text("CANARY" if good else ""),
        "process_exit_code": 0 if good else 1,
        "owned_process_absent": True,
        "process_owned": True,
        "port_release_confirmed": True,
        "teardown_complete": True,
        "vram_release_passed": True,
        "lease_consistent": True,
        "offloaded_layers": 49 if good else 0,
        "live_duration_s": 1.0 if good else 0.0,
        "tokens_per_second": 4.0 if good else 0.0,
        "lease_rows": [_lease_row(model_id, "GPU-0"), _lease_row(model_id, "GPU-1")],
        "process_ownership": {"model_id": model_id, "owned": True},
        "gpu_runtime": {"model_id": model_id, "live_cuda": good},
        "teardown": {"model_id": model_id, "passed": True},
        "vram_release": {"model_id": model_id, "passed": True},
    }


def test_req_infra_6973_spec_anchors_the_contract() -> None:
    """REQ-INFRA-6973 exists before implementation code."""

    text = (REPO / "openspec/capabilities/llm-ebm-inference/spec.md").read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6973") :]
    for anchor in (
        "SCENARIO-INFRA-6973-LEASE",
        "SCENARIO-INFRA-6973-STALE",
        "SCENARIO-INFRA-6973-SERVER",
        "SCENARIO-INFRA-6973-TEARDOWN",
        "SCENARIO-INFRA-6973-BARE-READINESS",
        "cached_sota_pair(gpu_indices=(0, 1))",
        "n_ctx >= 16384",
        "512 MiB",
        "runtime_handoff_complete_score",
        "lease_aware_runtime_ready_score",
    ):
        assert anchor in section


def test_req_infra_6973_resolves_exact_cached_three_family_manifest(tmp_path: Path) -> None:
    """REQ-INFRA-6973 fixes the cached pair call and exact family order."""

    expected = _resolved_specs(tmp_path)
    calls: list[dict] = []

    def pair(**kwargs: object) -> list[dict]:
        calls.append(dict(kwargs))
        return [
            {**expected[0], "gpu": 0},
            {**expected[2], "gpu": 1},
        ]

    rows = exp.resolve_model_specs(
        cached_pair_func=pair,
        resolver=lambda model_id, _quant: (
            expected[1]["model_path"] if model_id == exp.REQUIRED_MODEL_IDS[1] else None
        ),
    )
    assert calls == [{"gpu_indices": (0, 1)}]
    assert [row["hf_id"] for row in rows] == list(exp.REQUIRED_MODEL_IDS)
    assert all(row["gpu_indices"] == [0, 1] for row in rows)
    assert exp.model_spec_errors(rows) == []

    changed = deepcopy(rows)
    changed[0]["hf_id"] = "legacy/smoke-GGUF"
    changed[1]["model_path"] = str(tmp_path / "mmproj.gguf")
    changed[2]["gpu_indices"] = [0]
    changed[2]["headline_eligible"] = False
    errors = exp.model_spec_errors(changed)
    assert "model_ids_mismatch" in errors
    assert any(error.startswith("model_path_not_primary_gguf:") for error in errors)
    assert any(error.startswith("dual_gpu_indices_missing:") for error in errors)
    assert any(error.startswith("headline_eligibility_missing:") for error in errors)
    missing = deepcopy(rows)
    missing[0]["model_path"] = ""
    assert any(error.startswith("model_path_missing:") for error in exp.model_spec_errors(missing))


def test_req_infra_6973_load_config_uses_dual_cuda_and_32_token_budget() -> None:
    """REQ-INFRA-6973 freezes the promoted live generation controls."""

    assert exp.LOAD_CONFIG["n_ctx"] >= 16_384
    assert exp.LOAD_CONFIG["n_gpu_layers"] == -1
    assert exp.LOAD_CONFIG["visible_devices"] == [0, 1]
    assert exp.LOAD_CONFIG["tensor_split"] == [0.5, 0.5]
    assert exp.MAX_OUTPUT_TOKENS == 32
    assert "AutoTokenizer" not in exp.worker_execute.__code__.co_names


def test_scenario_infra_6973_stale_discriminates_released_dead_and_live() -> None:
    """SCENARIO-INFRA-6973-STALE keeps dead recovery separate from live ownership."""

    base = {
        "released": False,
        "task_id": "other-task",
        "owner": {"pid": 22, "pid_start_ticks": 33},
        "phase": "loading",
    }
    assert exp.classify_lease_document({**base, "released": True}, lambda *_: True) == {
        "classification": "released",
        "owner_live": False,
        "foreign_live": False,
    }
    assert exp.classify_lease_document(base, lambda *_: False) == {
        "classification": "stale_recoverable",
        "owner_live": False,
        "foreign_live": False,
    }
    assert exp.classify_lease_document(base, lambda *_: True) == {
        "classification": "live_foreign",
        "owner_live": True,
        "foreign_live": True,
    }
    assert exp.classify_lease_document({}, lambda *_: True)["classification"] == "invalid"
    assert (
        exp.classify_lease_document(
            {**base, "owner": {"pid": "22", "pid_start_ticks": 33}}, lambda *_: True
        )["classification"]
        == "invalid"
    )


def test_scenario_infra_6973_lease_snapshot_fails_closed_on_unreadable(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6973-LEASE requires one readable journal per device."""

    devices = _gpu_probe()["devices"]
    paths = {
        exp.lease_api.journal_path_for(tmp_path, "GPU-0"): {"released": True},
        exp.lease_api.journal_path_for(tmp_path, "GPU-1"): RuntimeError("bad journal"),
    }
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    def reader(path: Path) -> dict:
        value = paths[path]
        if isinstance(value, Exception):
            raise value
        return value

    rows = exp.lease_ledger_snapshot(devices, tmp_path, reader=reader)
    assert rows[0]["classification"] == "released"
    assert rows[0]["signals_sent"] == []
    assert rows[1]["readable"] is False
    assert rows[1]["classification"] == "unreadable"


def test_scenario_infra_6973_lease_preflight_blocks_live_foreign_owner(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6973-LEASE emits the exact blocked gate without a worker."""

    ledgers = _released_ledger_rows()
    ledgers[0].update({"classification": "live_foreign", "owner_live": True, "foreign_live": True})
    preflight = exp.collect_preconditions(
        model_specs=_resolved_specs(tmp_path),
        checkpoint_path=tmp_path / "checkpoint.json",
        gpu_probe=lambda: _gpu_probe(),
        llama_probe=lambda: {"importable": True, "gpu_offload": True},
        ledger_probe=lambda _devices: ledgers,
        metadata_probe=lambda model: {"model_id": model["hf_id"], "passed": True},
    )
    assert preflight["all_passed"] is False
    failed = [row for row in preflight["checks"] if row["passed"] is False]
    assert failed[0]["check"] == "lease_ledger_available_without_live_foreign_owner"

    calls: list[dict] = []
    artifact = exp.run(
        result_path=tmp_path / "blocked.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        model_specs=_resolved_specs(tmp_path),
        preflight_fn=lambda _specs, _path: preflight,
        attempt_runner=lambda **kwargs: calls.append(kwargs) or {},
        source_hash_fn=lambda: {},
        model_hash_fn=lambda _specs: {},
    )
    assert calls == []
    assert artifact["honest_verdict"] == "blocked_lease_aware_gguf_runtime"
    assert artifact["gate_check_summary"]["failed_check"] == failed[0]["check"]


def test_scenario_infra_6973_preflight_checks_every_required_resource(tmp_path: Path) -> None:
    """REQ-INFRA-6973 records exact expected and observed precondition values."""

    specs = _resolved_specs(tmp_path)
    specs[1]["model_path"] = str(tmp_path / "missing.gguf")
    specs[1]["headline_eligible"] = False
    preflight = exp.collect_preconditions(
        model_specs=specs,
        checkpoint_path=tmp_path / "missing" / "checkpoint.json",
        gpu_probe=lambda: {"query_ok": False, "devices": [], "processes": []},
        llama_probe=lambda: {"importable": False, "gpu_offload": False},
        writable_probe=lambda _path: False,
        ledger_probe=lambda _devices: [],
        metadata_probe=lambda model: {"model_id": model["hf_id"], "passed": False},
    )
    failures = {row["check"] for row in preflight["checks"] if row["passed"] is False}
    assert {
        "nvidia_device_count",
        "exact_model_specs",
        "all_three_cached_gguf_files",
        "llama_cpp_cuda_bindings",
        "lease_ledger_available_without_live_foreign_owner",
        "checkpoint_writable",
        "embedded_gguf_tokenizers",
    } <= failures
    summary = exp.gate_summary(preflight["checks"])
    assert summary["failed_check"] == "nvidia_device_count"
    assert summary["expected_value"] == 2


def test_scenario_infra_6973_acquisition_uses_shipped_owner_api() -> None:
    """SCENARIO-INFRA-6973-LEASE acquires both UUIDs with one task identity."""

    calls: list[dict] = []

    class FakeLease:
        def __init__(self, kwargs: dict) -> None:
            self.kwargs = kwargs
            self.document = {"phase": "preflight", "released": False}
            self.journal_path = Path(f"/tmp/{kwargs['device_uuid']}.json")

        def owner_receipt(self) -> dict:
            return {
                "task_id": self.kwargs["task_id"],
                "device_uuid": self.kwargs["device_uuid"],
                "pid": 44,
                "pid_start_ticks": 55,
                "expected_model": self.kwargs["expected_model"],
                "recovery": {"performed": False, "signals_sent": []},
                "signals_sent": [],
            }

        def transition(self, phase: str, **_kwargs: object) -> dict:
            self.document["phase"] = phase
            return {"accepted": True, "to_phase": phase}

        def release(self) -> dict:
            self.document["released"] = True
            return {"released": True}

        def close(self) -> None:
            self.document["closed"] = True

    def factory(**kwargs: object) -> FakeLease:
        calls.append(dict(kwargs))
        return FakeLease(dict(kwargs))

    devices = _gpu_probe()["devices"]
    leases, rows = exp.acquire_owned_leases(
        model={"hf_id": exp.REQUIRED_MODEL_IDS[0], "model_path": "/model.gguf"},
        devices=devices,
        runtime_dir=Path("/tmp/test-leases"),
        journal_before={"GPU-0": {"released": True}, "GPU-1": {"released": True}},
        lease_factory=factory,
    )
    assert len(leases) == len(rows) == 2
    assert [call["device_uuid"] for call in calls] == ["GPU-0", "GPU-1"]
    assert all(call["task_id"] == exp.EXPERIMENT_ID for call in calls)
    assert all(row["owner_verified"] is True for row in rows)
    assert all(row["signals_sent"] == [] for row in rows)

    created: list[FakeLease] = []

    def racing_factory(**kwargs: object) -> FakeLease:
        if created:
            raise exp.lease_api.LeaseBusy("race")
        lease = FakeLease(dict(kwargs))
        created.append(lease)
        return lease

    with pytest.raises(exp.lease_api.LeaseBusy, match="race"):
        exp.acquire_owned_leases(
            model={"hf_id": exp.REQUIRED_MODEL_IDS[0], "model_path": "/model.gguf"},
            devices=devices,
            runtime_dir=Path("/tmp/test-leases"),
            journal_before={},
            lease_factory=racing_factory,
        )
    assert created[0].document == {
        "phase": "terminal_blocked",
        "released": True,
    }

    class BadCleanupLease(FakeLease):
        def transition(self, phase: str, **kwargs: object) -> dict:
            if phase == "terminal_blocked":
                raise exp.lease_api.TransitionError("cleanup failed")
            return super().transition(phase, **kwargs)

    bad_created: list[BadCleanupLease] = []

    def bad_cleanup_factory(**kwargs: object) -> BadCleanupLease:
        if bad_created:
            raise exp.lease_api.LeaseBusy("second race")
        lease = BadCleanupLease(dict(kwargs))
        bad_created.append(lease)
        return lease

    with pytest.raises(exp.lease_api.LeaseBusy, match="second race"):
        exp.acquire_owned_leases(
            model={"hf_id": exp.REQUIRED_MODEL_IDS[0], "model_path": "/model.gguf"},
            devices=devices,
            runtime_dir=Path("/tmp/test-leases"),
            journal_before={},
            lease_factory=bad_cleanup_factory,
        )
    assert bad_created[0].document["closed"] is True

    class ResidentLease(FakeLease):
        def transition(self, phase: str, **kwargs: object) -> dict:
            result = super().transition(phase, **kwargs)
            if phase == "loading":
                self.document["phase"] = "resident"
            return result

    resident_created: list[ResidentLease] = []

    def resident_factory(**kwargs: object) -> ResidentLease:
        if resident_created:
            raise exp.lease_api.LeaseBusy("resident race")
        lease = ResidentLease(dict(kwargs))
        resident_created.append(lease)
        return lease

    with pytest.raises(exp.lease_api.LeaseBusy, match="resident race"):
        exp.acquire_owned_leases(
            model={"hf_id": exp.REQUIRED_MODEL_IDS[0], "model_path": "/model.gguf"},
            devices=devices,
            runtime_dir=Path("/tmp/test-leases"),
            journal_before={},
            lease_factory=resident_factory,
        )
    assert resident_created[0].document["closed"] is True


def test_scenario_infra_6973_server_ownership_refuses_foreign_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFRA-6973-SERVER signals only the exact spawned identity."""

    monkeypatch.setattr(exp, "_proc_stat", lambda _pid: (111, 222))
    owned = exp.capture_server_ownership(
        7,
        111,
        [sys.executable, "-m", "worker"],
        8899,
        port_owner_probe=lambda _port: [7],
        lineage_probe=lambda _pid, _task: {"task_owned": True, "chain": [7, 111]},
        task_identity={"pid": 111, "pid_start_ticks": 1},
    )
    assert owned["owned"] is True
    assert owned["port_owned_by_child"] is True
    assert owned["process_tree"]["task_owned"] is True

    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(exp.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    assert exp.terminate_owned_process(7, 111, 999) is False
    assert exp.terminate_owned_process(7, 111, 222) is True
    assert signals == [(7, exp.signal.SIGTERM)]
    monkeypatch.setattr(exp, "_proc_stat", lambda _pid: None)
    assert exp.owned_process_absent(7, 222) is True


def test_scenario_infra_6973_owner_filter_keeps_task_process_out_of_foreign_set() -> None:
    """SCENARIO-INFRA-6973-SERVER does not relabel the controller as foreign."""

    rows = [
        {"pid": 100, "process_name": "task-controller"},
        {"pid": 200, "process_name": "foreign-server"},
    ]
    assert exp.foreign_process_rows(rows, owner_pid=100) == [rows[1]]

    verified = {"owned": True, "port_owner_pids": [100]}
    late_empty = {"owned": False, "port_owner_pids": []}
    assert exp.retain_verified_ownership(verified, late_empty) == verified
    assert exp.retain_verified_ownership({"owned": False}, verified) == verified


def test_scenario_infra_6973_worker_uses_embedded_llama_tokenizer_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFRA-6973-SERVER generates 32-token-bounded output and closes."""

    class FakeLlama:
        closed = False

        def __init__(self, **kwargs: object) -> None:
            assert kwargs["n_ctx"] == 16_384
            assert kwargs["n_gpu_layers"] == -1
            assert kwargs["tensor_split"] == [0.5, 0.5]

        def create_completion(self, prompt: str, **kwargs: object) -> dict:
            assert prompt == exp.FIXED_PROMPT
            assert kwargs["max_tokens"] == 32
            return {
                "choices": [{"text": "owned CUDA handoff", "finish_reason": "length"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4},
            }

        def close(self) -> None:
            self.closed = True

    collected: list[bool] = []
    monkeypatch.setattr(exp.gc, "collect", lambda: collected.append(True) or 0)
    row = exp.worker_execute(
        {"model_id": exp.REQUIRED_MODEL_IDS[0], "model_path": "/model.gguf"},
        llama_factory=FakeLlama,
        clock=iter([1_000_000_000, 2_000_000_000, 4_000_000_000]).__next__,
    )
    assert row["terminal_state"] == "complete"
    assert row["output"] == "owned CUDA handoff"
    assert row["output_hash"] == exp.sha256_text(row["output"])
    assert row["live_duration_s"] == 2.0
    assert row["tokens_per_second"] == 2.0
    assert row["model_close_called"] is True
    assert collected == [True]

    class BrokenLlama:
        def __init__(self, **_kwargs: object) -> None:
            raise ValueError("no_memory")

    failed = exp.worker_execute(
        {"model_id": exp.REQUIRED_MODEL_IDS[0], "model_path": "/model.gguf"},
        llama_factory=BrokenLlama,
    )
    assert failed["terminal_state"] == "failed"
    assert failed["exception_type"] == "ValueError"
    assert failed["exception_message"] == "no_memory"


def test_scenario_infra_6973_teardown_requires_both_gpu_baselines() -> None:
    """SCENARIO-INFRA-6973-TEARDOWN rejects missing or excessive VRAM residue."""

    baseline = [
        {"index": 0, "memory_used_mb": 4},
        {"index": 1, "memory_used_mb": 8},
    ]
    passed = exp.build_vram_release_row(
        model_id=exp.REQUIRED_MODEL_IDS[0],
        baseline_rows=baseline,
        after_rows=[
            {"index": 0, "memory_used_mb": 516},
            {"index": 1, "memory_used_mb": 1},
        ],
    )
    assert passed["passed"] is True
    assert passed["max_increase_mb"] == 512
    failed = exp.build_vram_release_row(
        model_id=exp.REQUIRED_MODEL_IDS[0],
        baseline_rows=baseline,
        after_rows=[{"index": 0, "memory_used_mb": 517}],
    )
    assert failed["passed"] is False
    assert failed["missing_device_indices"] == [1]


def test_scenario_infra_6973_bare_readiness_recomputes_exact_rows() -> None:
    """SCENARIO-INFRA-6973-BARE-READINESS rejects every weak handoff."""

    rows = [_generation_row(model_id) for model_id in exp.REQUIRED_MODEL_IDS]
    assert exp.reduce_scores(rows) == (1, 1)
    for field, value in (
        ("output", ""),
        ("live_cuda", False),
        ("process_owned", False),
        ("port_release_confirmed", False),
        ("vram_release_passed", False),
        ("lease_consistent", False),
    ):
        weak = deepcopy(rows)
        weak[1][field] = value
        assert exp.reduce_scores(weak) == (1, 0)
    legacy = deepcopy(rows)
    legacy[2]["model_id"] = "legacy/smoke-GGUF"
    assert exp.reduce_scores(legacy) == (0, 0)
    assert exp.lease_row_is_consistent(rows[0]["lease_rows"][0]) is True
    mutated_lease = deepcopy(rows[0]["lease_rows"][0])
    mutated_lease["signals_sent"] = ["SIGTERM"]
    assert exp.lease_row_is_consistent(mutated_lease) is False


def test_req_infra_6973_artifact_has_bare_scores_and_cold_validation(tmp_path: Path) -> None:
    """REQ-INFRA-6973 binds all required fields to independently reduced rows."""

    specs = _resolved_specs(tmp_path)
    rows = [_generation_row(model_id) for model_id in exp.REQUIRED_MODEL_IDS]
    artifact = exp.build_artifact(
        run_date=exp.RUN_DATE,
        duration_s=4.0,
        model_specs=specs,
        preconditions=_passing_preflight(tmp_path),
        live_generation_rows=rows,
        source_artifact_hashes={},
        model_file_hashes={},
        checkpoint_rows=[{"model_id": row["model_id"], "written": True} for row in rows],
    )
    assert artifact["runtime_handoff_complete_score"] == 1
    assert artifact["lease_aware_runtime_ready_score"] == 1
    assert type(artifact["lease_aware_runtime_ready_score"]) is int
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete:")
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert exp.validate_artifact(artifact) == []

    mutated = deepcopy(artifact)
    mutated["lease_aware_runtime_ready_score"] = {"value": 1}
    assert "gate_score_not_bare_int:lease_aware_runtime_ready_score" in exp.validate_artifact(
        mutated
    )


def test_req_infra_6973_validator_rejects_mutated_contract_fields(tmp_path: Path) -> None:
    """REQ-INFRA-6973 rejects headline, row projection, and checksum mutations."""

    specs = _resolved_specs(tmp_path)
    rows = [_generation_row(model_id) for model_id in exp.REQUIRED_MODEL_IDS]
    artifact = exp.build_artifact(
        run_date=exp.RUN_DATE,
        duration_s=3.0,
        model_specs=specs,
        preconditions=_passing_preflight(tmp_path),
        live_generation_rows=rows,
    )
    assert exp.validate_artifact({}) == [
        f"missing_field:{field}" for field in exp.REQUIRED_ARTIFACT_FIELDS
    ]
    changed = deepcopy(artifact)
    changed.update(
        {
            "field_principles": {},
            "inference_substrate": "cpu",
            "models_used": [],
            "MODEL_SPECS": [],
            "runtime_handoff_complete_score": 0,
            "lease_aware_runtime_ready_score": 0,
            "random_seed": 0,
            "verifier_is_oracle": True,
            "duration_s": -1,
            "verdict_class": "null",
            "honest_verdict": "wrong",
            "lease_rows": [],
        }
    )
    errors = set(exp.validate_artifact(changed))
    assert {
        "field_principles_mismatch",
        "inference_substrate_mismatch",
        "models_used_mismatch",
        "model_specs_mismatch",
        "lease_rows_projection_mismatch",
        "gate_score_mismatch:runtime_handoff_complete_score",
        "gate_score_mismatch:lease_aware_runtime_ready_score",
        "random_seed_mismatch",
        "verifier_is_oracle_mismatch",
        "duration_invalid",
        "positive_verdict_mismatch",
        "reproducibility_checksum_mismatch",
    } <= errors

    projection = deepcopy(artifact)
    projection["gpu_runtime_rows"] = []
    assert "gpu_runtime_rows_projection_mismatch" in exp.validate_artifact(projection)

    identity = deepcopy(artifact)
    identity["schema"] = "wrong"
    identity["run_date"] = "19000101"
    assert {"schema_mismatch", "run_date_mismatch"} <= set(exp.validate_artifact(identity))

    invalid_duration = deepcopy(artifact)
    invalid_duration["duration_s"] = "not-a-number"
    assert "duration_invalid" in exp.validate_artifact(invalid_duration)

    failed_rows = [
        _generation_row(model_id, terminal="failed") for model_id in exp.REQUIRED_MODEL_IDS
    ]
    null = exp.build_artifact(
        run_date=exp.RUN_DATE,
        duration_s=1.0,
        model_specs=specs,
        preconditions=_passing_preflight(tmp_path),
        live_generation_rows=failed_rows,
    )
    assert null["verdict_class"] == "null"
    null["honest_verdict"] = "wrong"
    assert "null_verdict_mismatch" in exp.validate_artifact(null)

    partial = exp.build_artifact(
        run_date=exp.RUN_DATE,
        duration_s=1.0,
        model_specs=specs,
        preconditions=_passing_preflight(tmp_path),
    )
    partial["honest_verdict"] = "wrong"
    assert "partial_verdict_mismatch" in exp.validate_artifact(partial)

    blocked_preflight = deepcopy(_passing_preflight(tmp_path))
    blocked_preflight["all_passed"] = False
    blocked = exp.build_artifact(
        run_date=exp.RUN_DATE,
        duration_s=1.0,
        model_specs=specs,
        preconditions=blocked_preflight,
    )
    blocked["verdict_class"] = "null"
    blocked["honest_verdict"] = "wrong"
    blocked["gate_check_summary"] = None
    assert {"blocked_verdict_mismatch", "blocked_gate_summary_incomplete"} <= set(
        exp.validate_artifact(blocked)
    )


def test_req_infra_6973_controller_checkpoints_each_terminal_model(tmp_path: Path) -> None:
    """REQ-INFRA-6973 serializes each safe handoff before admitting the next model."""

    specs = _resolved_specs(tmp_path)
    calls: list[str] = []

    def attempt(**kwargs: object) -> dict:
        model = dict(kwargs["model"])
        calls.append(model["hf_id"])
        return _generation_row(model["hf_id"])

    artifact = exp.run(
        result_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        model_specs=specs,
        preflight_fn=lambda _specs, _path: _passing_preflight(tmp_path),
        attempt_runner=attempt,
        source_hash_fn=lambda: {},
        model_hash_fn=lambda _specs: {},
    )
    assert calls == list(exp.REQUIRED_MODEL_IDS)
    assert len(artifact["checkpoint_rows"]) == 3
    assert artifact["runtime_handoff_complete_score"] == 1
    assert artifact["lease_aware_runtime_ready_score"] == 1
    assert exp.validate_artifact(artifact) == []

    calls.clear()

    def unsafe(**kwargs: object) -> dict:
        model = dict(kwargs["model"])
        row = _generation_row(model["hf_id"], terminal="failed")
        row["handoff_safe"] = False
        calls.append(model["hf_id"])
        return row

    partial = exp.run(
        result_path=tmp_path / "partial.json",
        checkpoint_path=tmp_path / "partial-checkpoint.json",
        model_specs=specs,
        preflight_fn=lambda _specs, _path: _passing_preflight(tmp_path),
        attempt_runner=unsafe,
        source_hash_fn=lambda: {},
        model_hash_fn=lambda _specs: {},
    )
    assert calls == [exp.REQUIRED_MODEL_IDS[0]]
    assert partial["verdict_class"] == "partial"


def test_req_infra_6973_checkpoint_and_host_helpers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6973 preserves atomic checkpoints and exact process evidence."""

    row = _generation_row(exp.REQUIRED_MODEL_IDS[0])
    first = exp.checkpoint_model_row(tmp_path / "checkpoint.json", "sha256:manifest", row)
    assert first["written"] is True
    same = exp.checkpoint_model_row(tmp_path / "checkpoint.json", "sha256:manifest", row)
    assert same["written"] is False
    with pytest.raises(ValueError, match="checkpoint_manifest_mismatch"):
        exp.checkpoint_model_row(tmp_path / "checkpoint.json", "sha256:other", row)
    changed = deepcopy(row)
    changed["output"] = "changed"
    with pytest.raises(ValueError, match="checkpoint_model_row_mismatch"):
        exp.checkpoint_model_row(tmp_path / "checkpoint.json", "sha256:manifest", changed)

    success = exp._run_command([sys.executable, "-c", "print('ok')"])
    assert success["passed"] is True
    assert exp._run_command([str(tmp_path / "absent")])["passed"] is False
    monkeypatch.setattr(
        exp.tempfile, "mkstemp", lambda **_kwargs: (_ for _ in ()).throw(OSError("no"))
    )
    assert exp.checkpoint_is_writable(tmp_path / "other.json") is False


def test_req_infra_6973_capture_live_child_identity(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6973-SERVER binds a real child PID and start time."""

    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.2)"])
    try:
        row = exp.capture_server_ownership(
            process.pid,
            os.getpid(),
            [sys.executable, "-c"],
            9999,
            port_owner_probe=lambda _port: [process.pid],
            lineage_probe=lambda _pid, _task: {"task_owned": True},
            task_identity={"pid": os.getpid(), "pid_start_ticks": 1},
        )
        assert row["owned"] is True
        assert row["pid_start_ticks"] > 0
        assert row["command_hash"].startswith("sha256:")
    finally:
        process.terminate()
        process.wait(timeout=5)
    assert exp._proc_stat(999_999_999) is None


def test_req_infra_6973_worker_import_and_token_count_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-INFRA-6973 covers the native binding import and embedded token fallback."""

    class FakeLlama:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def create_completion(self, *_args: object, **_kwargs: object) -> dict:
            return {"choices": [{"text": "TOKENS"}], "usage": {}}

        def tokenize(self, _value: bytes, **_kwargs: object) -> list[int]:
            return [1, 2, 3]

        def close(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    row = exp.worker_execute(
        {"model_id": exp.REQUIRED_MODEL_IDS[0], "model_path": "/model.gguf"},
        clock=iter([0, 1_000_000_000, 2_000_000_000]).__next__,
    )
    assert row["completion_tokens"] == 3


def test_req_infra_6973_blocked_attempt_and_invalid_run_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6973 retains acquisition failures and refuses an invalid artifact."""

    model = _resolved_specs(tmp_path)[0]
    blocked = exp._empty_attempt(model, "lease busy")
    assert blocked["terminal_state"] == "blocked"
    assert blocked["exception_message"] == "lease busy"

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced-invalid"])
    with pytest.raises(RuntimeError, match="artifact_validation_failed"):
        exp.run(
            result_path=tmp_path / "invalid.json",
            checkpoint_path=tmp_path / "invalid-checkpoint.json",
            model_specs=_resolved_specs(tmp_path),
            preflight_fn=lambda _specs, _path: _passing_preflight(tmp_path),
            attempt_runner=lambda **kwargs: _generation_row(dict(kwargs["model"])["hf_id"]),
            source_hash_fn=lambda: {},
            model_hash_fn=lambda _specs: {},
        )
