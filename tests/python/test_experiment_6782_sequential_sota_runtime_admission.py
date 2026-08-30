"""Tests for sequential receipt-scoped SOTA CUDA admission.

Spec refs: REQ-INFER-SOTA-6782, SCENARIO-INFER-SOTA-6782-IDENTITY,
SCENARIO-INFER-SOTA-6782-FIRST-TOKEN,
SCENARIO-INFER-SOTA-6782-SEQUENTIAL-RECOVERY, REQ-REPORT-6782,
SCENARIO-REPORT-6782-BOUNDED-WAIT, SCENARIO-REPORT-6782-CHECKPOINT, and
SCENARIO-REPORT-6782-COLD-CONSISTENCY.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

from carnot import experiment_6782_sequential_sota_runtime_admission as exp


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _models(tmp_path: Path) -> list[dict]:
    rows = []
    for index, planned in enumerate(exp.MODEL_SPECS):
        path = tmp_path / planned["filename"]
        path.write_bytes(f"model-{index}".encode())
        rows.append(
            {
                **deepcopy(planned),
                "revision": f"revision-{index}",
                "model_path": str(path),
                "model_sha256": planned["expected_sha256"],
                "model_size_bytes": path.stat().st_size,
                "tokenizer": {
                    "source": "llama.cpp_embedded_gguf",
                    "loadable": True,
                    "detail": "fixture tokenizer",
                },
                "context_tokens": exp.CANARY_CONTEXT_TOKENS,
                "max_output_tokens": exp.CANARY_MAX_OUTPUT_TOKENS,
            }
        )
    return rows


def _device(index: int = 0, *, active: list[dict] | None = None) -> dict:
    return {
        "index": index,
        "uuid": exp.EXPECTED_GPU_UUIDS[index],
        "name": "NVIDIA GeForce RTX 3090",
        "memory_total_mb": 24576,
        "memory_used_mb": 4,
        "memory_free_mb": 24120,
        "temperature_c": 45,
        "utilization_pct": 0,
        "active_compute_processes": list(active or []),
    }


def _receipt(model: dict, device: dict, *, ready: bool = True) -> dict:
    phases = list(exp.COMPLETE_PHASE_SEQUENCE if ready else ("preflight", "terminal_blocked"))
    worker = {"pid": 501, "pid_start_ticks": 9001, "exit_code": 0, "absent_after_exit": True}
    owner = {
        "task_id": "exp6782-fixture",
        "device_uuid": device["uuid"],
        "pid": worker["pid"],
        "pid_start_ticks": worker["pid_start_ticks"],
        "expected_model": model["model_path"],
        "signals_sent": [],
    }
    row = {
        "model_id": model["hf_id"],
        "model_record": deepcopy(model),
        "device": deepcopy(device),
        "worker_process": worker,
        "lease_owner": owner,
        "phase_history": [
            {
                "phase": phase,
                "previous_phase": phases[index - 1] if index else None,
                "monotonic_ns": 1_000 + index,
                "event_checksum": _sha(f"{model['family_id']}:{phase}"),
            }
            for index, phase in enumerate(phases)
        ],
        "lease_release": {
            "released": ready,
            "phase": "terminal_complete" if ready else "terminal_blocked",
            "device_uuid": device["uuid"],
            "pid": worker["pid"],
            "signals_sent": [],
        },
        "cuda_offload": {
            "requested_gpu_layers": -1,
            "supports_gpu_offload": ready,
            "owned_cuda_resident": ready,
        },
        "resident_owned_vram_mb": 18_000 if ready else 0,
        "peak_owned_vram_mb": 18_500 if ready else 0,
        "first_token_canary": {
            "prompt_sha256": exp.CANARY_PROMPT_SHA256,
            "first_token_observed": ready,
            "completion_tokens": 1 if ready else 0,
            "first_token_sha256": _sha(" blue") if ready else "missing",
            "non_fixture_token": ready,
            "bounded": True,
        },
        "backend_teardown": {"close_called": ready, "close_error": None},
        "vram_recovery": {
            "before_used_mb": 4,
            "after_used_mb": 4,
            "absolute_delta_mb": 0,
            "tolerance_mb": exp.VRAM_RECOVERY_TOLERANCE_MB,
            "owned_pid_present": False,
            "passed": ready,
        },
        "protected_process_actions": [],
        "unrelated_processes_signaled": [],
        "duration_s": 2.0,
        "errors": [] if ready else ["lease_wait_deadline_expired"],
    }
    row["receipt_sha256"] = exp.gpu_receipt_checksum(row)
    return row


def _preflight(models: list[dict], *, passed: bool = True) -> dict:
    checks = [
        exp.check_row("models_resolved", True, passed, passed),
        exp.check_row("llama_cpp_cuda", True, passed, passed),
    ]
    return {
        "all_passed": passed,
        "checks": checks,
        "models": models,
        "ports": [45_100, 45_101, 45_102],
        "device_inventory_before": [_device(0), _device(1)],
        "llama_cpp": {"cuda_linked": passed, "python_cuda_offload": passed},
        "resources": {"ram_available_bytes": 96 * 1024**3, "disk_free_bytes": 2 * 1024**3},
    }


def test_resolver_keeps_exact_three_family_identity(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-6782-IDENTITY fixes paths, hashes, and tokenizers."""

    paths = {}
    for planned in exp.MODEL_SPECS:
        path = tmp_path / planned["filename"]
        path.write_bytes(planned["family_id"].encode())
        paths[planned["hf_id"]] = path

    def pair_resolver(**_: object) -> list[dict]:
        return [
            {
                "hf_id": exp.MODEL_SPECS[0]["hf_id"],
                "model_path": str(paths[exp.MODEL_SPECS[0]["hf_id"]]),
            },
            {
                "hf_id": exp.MODEL_SPECS[1]["hf_id"],
                "model_path": str(paths[exp.MODEL_SPECS[1]["hf_id"]]),
            },
        ]

    rows = exp.resolve_model_specs(
        pair_resolver=pair_resolver,
        single_resolver=lambda hf_id, _quant: str(paths[hf_id]),
        tokenizer_probe=lambda _path: (True, "embedded fixture"),
        file_hasher=lambda path: next(
            planned["expected_sha256"]
            for planned in exp.MODEL_SPECS
            if planned["filename"] == Path(path).name
        ),
    )

    assert [row["hf_id"] for row in rows] == [row["hf_id"] for row in exp.MODEL_SPECS]
    assert all(
        not exp.model_record_errors(row, planned)
        for row, planned in zip(rows, exp.MODEL_SPECS, strict=True)
    )
    assert all(row["tokenizer"]["loadable"] is True for row in rows)


def test_bounded_wait_records_every_poll_and_expires() -> None:
    """SCENARIO-REPORT-6782-BOUNDED-WAIT uses one monotonic deadline."""

    class Clock:
        now = 0

        def read(self) -> int:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += int(seconds * 1_000_000_000)

    clock = Clock()
    busy = _device(0, active=[{"pid": 91, "process_name": "llama-server", "used_memory_mb": 20000}])
    selected, rows = exp.wait_for_eligible_device(
        deadline_ns=2_000_000_000,
        inventory_fn=lambda: {"devices": [busy, _device(1, active=[{"pid": 92}])]},
        clock=clock.read,
        sleep_fn=clock.sleep,
        poll_interval_s=1.0,
        protected_classifier=lambda pid: "serving" if pid == 91 else None,
    )

    assert selected is None
    assert [row["row_kind"] for row in rows] == ["lease_poll", "lease_poll", "lease_poll"]
    assert (
        rows[0]["selection"]["evaluated_devices"][0]["protected_processes"][0]["kind"] == "serving"
    )
    assert clock.now == 2_000_000_000


def test_active_or_protected_process_makes_device_ineligible_without_actions() -> None:
    """REQ-REPORT-6782 refuses protected and unowned process reuse."""

    devices = [
        _device(0, active=[{"pid": 17, "process_name": "train.py", "used_memory_mb": 1024}]),
        _device(1, active=[{"pid": 18, "process_name": "python", "used_memory_mb": 256}]),
    ]
    selection = exp.rank_eligible_devices(
        devices, protected_classifier=lambda pid: "training" if pid == 17 else None
    )

    assert selection["selected_device"] is None
    assert selection["protected_process_actions"] == []
    assert selection["evaluated_devices"][0]["ineligibility_reasons"] == [
        "active_unowned_compute_process"
    ]
    assert selection["evaluated_devices"][1]["ineligibility_reasons"] == [
        "active_unowned_compute_process"
    ]


def test_live_worker_proves_owned_token_teardown_and_recovery(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-6782-FIRST-TOKEN proves the full owned lifecycle."""

    model = _models(tmp_path)[0]
    device = _device()

    class Lease:
        def __init__(self) -> None:
            self.document = {"phase": "preflight", "phase_history": []}
            self.released = False

        def owner_receipt(self) -> dict:
            return {
                "task_id": "exp6782-fixture",
                "device_uuid": device["uuid"],
                "pid": 77,
                "pid_start_ticks": 88,
                "expected_model": model["model_path"],
                "signals_sent": [],
            }

        def transition(self, phase: str, **_: object) -> dict:
            previous = self.document["phase"] if self.document["phase_history"] else None
            self.document["phase"] = phase
            self.document["phase_history"].append(
                {
                    "phase": phase,
                    "previous_phase": previous,
                    "monotonic_ns": len(self.document["phase_history"]),
                    "event_checksum": _sha(phase),
                }
            )
            return {"accepted": True}

        def release(self) -> dict:
            self.released = True
            return {
                "released": True,
                "phase": self.document["phase"],
                "device_uuid": device["uuid"],
                "pid": 77,
                "signals_sent": [],
            }

        def close(self) -> None:
            pass

    lease = Lease()
    lease.transition("preflight")

    class Llama:
        closed = False

        def create_completion(self, **_: object) -> dict:
            return {"choices": [{"text": " blue"}], "usage": {"completion_tokens": 1}}

        def close(self) -> None:
            self.closed = True

    llm = Llama()
    samples = iter(
        [
            {**device, "owned_pid_present": False, "owned_pid_vram_mb": 0},
            {
                **device,
                "memory_used_mb": 18004,
                "owned_pid_present": True,
                "owned_pid_vram_mb": 18000,
            },
            {
                **device,
                "memory_used_mb": 18504,
                "owned_pid_present": True,
                "owned_pid_vram_mb": 18500,
            },
            {**device, "memory_used_mb": 4, "owned_pid_present": False, "owned_pid_vram_mb": 0},
        ]
    )

    row = exp.run_live_model_worker(
        model,
        device,
        prompt=exp.CANARY_PROMPT,
        lease_runtime_dir=tmp_path / "leases",
        llama_factory=lambda **_: llm,
        lease_factory=lambda **_: lease,
        snapshot_fn=lambda _uuid, _pid: next(samples),
        process_identity_fn=lambda: {
            "pid": 77,
            "pid_start_ticks": 88,
            "executable": "python",
            "exit_code": None,
            "absent_after_exit": False,
        },
        supports_gpu_offload_fn=lambda: True,
        sleep_fn=lambda _seconds: None,
    )

    assert exp.gpu_receipt_errors(row, model, require_worker_exit=False) == []
    assert row["first_token_canary"]["non_fixture_token"] is True
    assert row["backend_teardown"]["close_called"] is True
    assert row["vram_recovery"]["passed"] is True
    assert lease.released is True and llm.closed is True


def test_blocked_precondition_still_writes_complete_schema(tmp_path: Path) -> None:
    """REQ-REPORT-6782 keeps a complete blocked artifact."""

    models = _models(tmp_path)
    preflight = _preflight(models, passed=False)
    artifact = exp.build_artifact(
        date=exp.RUN_DATE,
        preconditions=preflight,
        gpu_receipts=[],
        poll_rows=[],
        code_receipts={"module": _sha("module")},
        started_ns=100,
        finished_ns=200,
    )

    assert artifact["status"] == "complete_blocked_sequential_sota_runtime"
    assert artifact["honest_verdict"].startswith("complete_blocked_sequential_sota_runtime")
    assert artifact["models_used"] == []
    assert artifact["gate_check_summary"]["failed_check"] == "models_resolved"
    assert exp.validate_artifact(artifact) == []


def test_checkpoint_preserves_earlier_model_when_later_model_blocks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6782-CHECKPOINT keeps earlier model-local readiness."""

    models = _models(tmp_path)
    devices = [_device(0), _device(1)]
    waits = iter(
        [
            (
                devices[0],
                [
                    {
                        "row_kind": "lease_poll",
                        "model_id": models[0]["hf_id"],
                        "poll_index": 0,
                        "observed_monotonic_ns": 10,
                        "inventory": devices,
                        "selection": {"selected_device": devices[0]},
                        "passed": True,
                    }
                ],
            ),
            (
                None,
                [
                    {
                        "row_kind": "lease_poll",
                        "model_id": models[1]["hf_id"],
                        "poll_index": 0,
                        "observed_monotonic_ns": 20,
                        "inventory": devices,
                        "selection": {"selected_device": None},
                        "passed": False,
                    }
                ],
            ),
        ]
    )
    result_path = tmp_path / "experiment_6782.json"
    artifact = exp.run(
        result_path=result_path,
        date=exp.RUN_DATE,
        preflight_fn=lambda: _preflight(models),
        device_waiter=lambda _deadline, _model_id: next(waits),
        worker_runner=lambda model, device, _prompt, _runtime: _receipt(model, device),
        code_receipt_fn=lambda: {"module": _sha("module")},
        clock=iter([1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000]).__next__,
    )

    assert result_path.is_file()
    assert artifact["qwen36_runtime_ready"] is True
    assert artifact["gemma31_runtime_ready"] is False
    assert artifact["gemma26_runtime_ready"] is False
    assert artifact["all_mandated_runtime_ready"] is False
    assert artifact["models_used"] == [models[0]]
    assert any(row["row_kind"] == "recovery" for row in artifact["rows"])
    assert exp.validate_artifact(artifact) == []


def test_cold_validator_rejects_copied_readiness_and_bad_token(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6782-COLD-CONSISTENCY recomputes readiness."""

    models = _models(tmp_path)
    receipt = _receipt(models[0], _device())
    artifact = exp.build_artifact(
        date=exp.RUN_DATE,
        preconditions=_preflight(models),
        gpu_receipts=[receipt],
        poll_rows=[],
        code_receipts={"module": _sha("module")},
        started_ns=0,
        finished_ns=1_000_000_000,
    )
    copied = deepcopy(artifact)
    copied["all_mandated_runtime_ready"] = True
    copied["reproducibility_checksum"] = exp.artifact_checksum(copied)
    assert "all_mandated_runtime_ready" in exp.validate_artifact(copied)

    bad_receipt = deepcopy(receipt)
    bad_receipt["first_token_canary"]["non_fixture_token"] = False
    bad_receipt["receipt_sha256"] = exp.gpu_receipt_checksum(bad_receipt)
    assert "first_token_canary" in exp.gpu_receipt_errors(bad_receipt, models[0])


def test_resolver_failures_and_preconditions_are_explicit(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-6782 and REQ-REPORT-6782 fail closed at preflight."""

    probe = tmp_path / "probe.bin"
    probe.write_bytes(b"probe")
    assert exp.sha256_file(probe) == _sha("probe")
    assert exp.sha256_file(tmp_path / "missing") == "missing"
    assert exp._revision_from_path(Path("cache/snapshots/revision/model.gguf")) == "revision"
    assert exp._revision_from_path(Path("cache/model.gguf")) == "local-unversioned"

    missing = exp.resolve_model_specs(
        pair_resolver=lambda **_: None,
        single_resolver=lambda _hf_id, _quant: None,
        tokenizer_probe=lambda _path: (False, "missing"),
    )
    assert all(row["model_sha256"] == "missing" for row in missing)
    damaged = deepcopy(_models(tmp_path)[0])
    damaged.pop("role")
    damaged.update(
        {
            "hf_id": "substituted/model",
            "model_sha256": "missing",
            "revision": "",
            "model_path": "wrong.gguf",
            "model_size_bytes": 0,
            "tokenizer": None,
            "context_tokens": 1,
            "max_output_tokens": 2,
        }
    )
    assert set(exp.model_record_errors(damaged, exp.MODEL_SPECS[0])) >= {
        "field_set",
        "hf_id",
        "model_sha256",
        "path_or_revision",
        "model_size_bytes",
        "tokenizer",
        "context_tokens",
        "max_output_tokens",
    }

    models = _models(tmp_path)
    good_llama = {
        "exists": True,
        "executable": True,
        "cuda_linked": True,
        "python_cuda_offload": True,
    }
    good_resources = {
        "ram_available_bytes": exp.RAM_AVAILABLE_FLOOR_BYTES,
        "disk_free_bytes": exp.DISK_FREE_FLOOR_BYTES,
    }
    preflight = exp.collect_preconditions(
        date=exp.RUN_DATE,
        model_resolver=lambda: models,
        inventory_fn=lambda: {"devices": [_device(0), _device(1)]},
        llama_receipt_fn=lambda: good_llama,
        port_picker=lambda count: list(range(45_200, 45_200 + count)),
        port_probe=lambda _port: True,
        resource_fn=lambda _root: good_resources,
    )
    assert preflight["all_passed"] is True
    blocked = exp.collect_preconditions(
        date="bad-date",
        model_resolver=lambda: missing,
        inventory_fn=lambda: {"devices": []},
        llama_receipt_fn=lambda: {},
        port_picker=lambda _count: [45_200],
        port_probe=lambda _port: False,
        resource_fn=lambda _root: {},
    )
    assert blocked["all_passed"] is False


def test_device_ranking_covers_identity_capacity_and_immediate_success() -> None:
    """SCENARIO-REPORT-6782-BOUNDED-WAIT records success and refusal reasons."""

    wrong = _device(0, active=["malformed"])
    wrong.update(uuid="GPU-wrong", name="Strix", memory_free_mb=100)
    selection = exp.rank_eligible_devices([wrong], protected_classifier=lambda _pid: None)
    assert selection["evaluated_devices"][0]["ineligibility_reasons"] == [
        "unexpected_device_identity",
        "free_vram_below_floor",
        "active_unowned_compute_process",
    ]
    selected, rows = exp.wait_for_eligible_device(
        deadline_ns=0,
        model_id="fixture/model",
        inventory_fn=lambda: {"devices": [_device()]},
        clock=lambda: 0,
        sleep_fn=lambda _seconds: None,
        protected_classifier=lambda _pid: None,
    )
    assert selected["uuid"] == exp.EXPECTED_GPU_UUIDS[0]
    assert rows[0]["model_id"] == "fixture/model" and rows[0]["passed"] is True


def test_terminalization_and_worker_failures_remain_blocked(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-6782-SEQUENTIAL-RECOVERY rejects each live failure stage."""

    class Lease:
        def __init__(self, phase: str = "preflight") -> None:
            self.document = {"phase": phase, "phase_history": []}
            self.closed = False

        def owner_receipt(self) -> dict:
            return {
                "task_id": "fixture",
                "device_uuid": exp.EXPECTED_GPU_UUIDS[0],
                "pid": 77,
                "pid_start_ticks": 88,
                "expected_model": str(tmp_path / exp.MODEL_SPECS[0]["filename"]),
                "signals_sent": [],
            }

        def transition(self, phase: str, **_: object) -> dict:
            previous = self.document["phase"]
            self.document["phase"] = phase
            self.document["phase_history"].append(
                {
                    "phase": phase,
                    "previous_phase": previous,
                    "monotonic_ns": 1,
                    "event_checksum": _sha(phase),
                }
            )
            return {}

        def release(self) -> dict:
            return {"released": True, "phase": self.document["phase"]}

        def close(self) -> None:
            self.closed = True

    for phase, complete, terminal in (
        ("resident", True, "terminal_complete"),
        ("preflight", False, "terminal_blocked"),
        ("validating", False, "terminal_blocked"),
    ):
        lease = Lease(phase)
        release = exp._terminalize_lease(lease, complete, {"memory_used_mb": 4})
        assert release["phase"] == terminal
    unknown = Lease("unknown")
    assert exp._terminalize_lease(unknown, False, {}) == {} and unknown.closed is True

    model = _models(tmp_path)[0]
    device = _device()

    def run_failure(
        *,
        before: dict,
        supports: bool = True,
        resident: dict | None = None,
        result: dict | None = None,
    ) -> dict:
        lease = Lease()
        lease.document["phase_history"] = [
            {
                "phase": "preflight",
                "previous_phase": None,
                "monotonic_ns": 0,
                "event_checksum": _sha("preflight"),
            }
        ]
        samples = [before]
        if resident is not None:
            samples.append(resident)
            if resident.get("owned_pid_present"):
                samples.append(resident)
        samples.append(
            {**device, "memory_used_mb": 4, "owned_pid_present": False, "owned_pid_vram_mb": 0}
        )

        class Llama:
            def create_completion(self, **_: object) -> dict:
                return result or {"choices": [], "usage": {}}

            def close(self) -> None:
                pass

        return exp.run_live_model_worker(
            model,
            device,
            prompt=exp.CANARY_PROMPT,
            lease_runtime_dir=tmp_path,
            llama_factory=lambda **_: Llama(),
            lease_factory=lambda **_: lease,
            snapshot_fn=lambda _uuid, _pid: samples.pop(0),
            process_identity_fn=lambda: {
                "pid": 77,
                "pid_start_ticks": 88,
                "exit_code": None,
                "absent_after_exit": False,
            },
            supports_gpu_offload_fn=lambda: supports,
            sleep_fn=lambda _seconds: None,
        )

    bad_selection = run_failure(
        before={**device, "memory_free_mb": 1, "active_compute_processes": []}
    )
    no_offload = run_failure(
        before={**device, "owned_pid_present": False, "active_compute_processes": []},
        supports=False,
    )
    no_residency = run_failure(
        before={**device, "owned_pid_present": False, "active_compute_processes": []},
        resident={**device, "owned_pid_present": False, "owned_pid_vram_mb": 0},
    )
    no_token = run_failure(
        before={**device, "owned_pid_present": False, "active_compute_processes": []},
        resident={**device, "owned_pid_present": True, "owned_pid_vram_mb": 100},
        result={"choices": [{"text": ""}], "usage": {"completion_tokens": 0}},
    )
    assert all(row["errors"] for row in (bad_selection, no_offload, no_residency, no_token))


def test_all_receipt_and_artifact_error_branches_are_cold_checked(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6782-COLD-CONSISTENCY rejects every copied claim class."""

    models = _models(tmp_path)
    receipts = [_receipt(model, _device(index % 2)) for index, model in enumerate(models)]
    artifact = exp.build_artifact(
        date=exp.RUN_DATE,
        preconditions=_preflight(models),
        gpu_receipts=receipts,
        poll_rows=[],
        code_receipts={"module": _sha("module")},
        started_ns=0,
        finished_ns=1,
    )
    assert artifact["all_mandated_runtime_ready"] is True
    assert artifact["verdict_class"] == "positive"
    assert exp.validate_artifact(artifact) == []

    bad = deepcopy(receipts[0])
    bad.update(
        {
            "model_id": "wrong",
            "model_record": {},
            "device": {},
            "lease_owner": {},
            "phase_history": [],
            "lease_release": {},
            "cuda_offload": {},
            "resident_owned_vram_mb": 0,
            "peak_owned_vram_mb": 0,
            "first_token_canary": {},
            "backend_teardown": {},
            "worker_process": {},
            "vram_recovery": {"passed": False},
            "protected_process_actions": ["forbidden"],
            "unrelated_processes_signaled": [99],
            "errors": ["failure"],
            "receipt_sha256": "wrong",
        }
    )
    assert set(exp.gpu_receipt_errors(bad, models[0])) >= {
        "receipt_sha256",
        "model_record",
        "lease_owner",
        "phase_sequence",
        "lease_release",
        "cuda_offload",
        "first_token_canary",
        "backend_teardown",
        "worker_process",
        "vram_recovery",
        "protected_process_actions",
        "unrelated_processes_signaled",
        "errors",
    }
    unknown_model = deepcopy(models[0])
    unknown_model["hf_id"] = "unknown/model"
    assert "model_identity" in exp.gpu_receipt_errors(bad, unknown_model)
    blocked_receipt = exp._blocked_worker_receipt(models[0], _device(), "blocked")
    blocked_artifact = exp.build_artifact(
        date=exp.RUN_DATE,
        preconditions=_preflight(models),
        gpu_receipts=[blocked_receipt],
        poll_rows=[],
        code_receipts={},
        started_ns=0,
        finished_ns=1,
    )
    assert blocked_artifact["gate_check_summary"]["observed"] != "lease_wait_deadline_expired"

    mutations = [
        lambda row: row.pop("schema"),
        lambda row: row.update(field_principles={}),
        lambda row: row.update(schema="wrong"),
        lambda row: row.update(run_date="wrong"),
        lambda row: row.update(MODEL_SPECS=[]),
        lambda row: row.update(inference_substrate="cpu"),
        lambda row: row.update(duration_s=float("nan")),
        lambda row: row.update(random_seed=0),
        lambda row: row.update(verifier_is_oracle=True),
        lambda row: row.update(verdict_class="unknown"),
        lambda row: row.update(honest_verdict="blocked"),
        lambda row: row.update(model_specs=[]),
        lambda row: row["rows"].append({"row_kind": "extra"}),
        lambda row: row.update(protected_process_actions=["forbidden"]),
        lambda row: row.update(status="wrong"),
        lambda row: row.update(models_used=[]),
        lambda row: row.update(live_model_invoked=False),
        lambda row: row.update(gate_check_summary={}),
        lambda row: row.update(reproducibility_checksum="wrong"),
    ]
    for mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        assert exp.validate_artifact(changed)

    invalid_path = tmp_path / "invalid.json"
    try:
        exp.write_artifact(invalid_path, {**artifact, "schema": "wrong"})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid artifact was written")


def test_run_returns_atomic_blocked_preflight(tmp_path: Path) -> None:
    """REQ-REPORT-6782 checkpoints even when live work cannot start."""

    result = exp.run(
        result_path=tmp_path / "blocked.json",
        date=exp.RUN_DATE,
        preflight_fn=lambda: _preflight(_models(tmp_path), passed=False),
        device_waiter=lambda _deadline, _model: (_device(), []),
        worker_runner=lambda *_args: raise_unexpected(),
        code_receipt_fn=lambda: {},
        clock=iter([1, 2]).__next__,
    )
    assert result["status"] == "complete_blocked_sequential_sota_runtime"


def raise_unexpected() -> None:
    raise AssertionError("worker must not run")
