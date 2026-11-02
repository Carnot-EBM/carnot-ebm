"""REQ-VERIFY-8277: test the leased native backend without generating tokens.

GPU enumeration is only a selection input. Health, model identity, CUDA
offload and process-bound residency must agree before execution is qualified.
Historical PyTorch failure remains evidence, not a native backend verdict.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.experiment_7969_v691_qwen_calibration_capture import loaded_libraries
from carnot.gpu_lease_phase_journal import GpuLease, LeaseError, proc_start_ticks
from carnot.inference.gguf_metadata import read_gguf_metadata
from carnot.inference.llama_cpp_process import OwnedLlamaCppProcess
from carnot.inference.qwen_sufficiency_7920 import bounded, get_json, offload_layers
from carnot.inference.sota_models import cached_current_model
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.reporting.v709_execution import child
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8277_v715_lease_backend_qualification"
TASK = "exp8277-lease-backend-qualification"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = "python/carnot/verify/lease_backend_8277.py"
TEST = "tests/python/test_lease_backend_8277.py"
OWNED = [MODULE, CLI]
MODEL_PIN = "sha256:7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
MODEL_SPECS = [
    dict(hf_id="unsloth/Qwen3.8-27B-GGUF", quantization="Q4_K_M", model_path_sha256=MODEL_PIN)
]
MASKS = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries so real work and pending children stay visible."""
    print(f"[exp8277] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Bind an operand to actual bytes; absent evidence has no invented hash."""
    return dict(path=str(path), sha256=sha256_file(path) if path.is_file() else None)


def gate(work: Json, path: Path, field: str, expected: Any, observed: Any, upstream: str) -> None:
    """Retain exact failed values instead of replacing absence with zero."""
    work["checks"].append(
        dict(
            reference(path),
            upstream=upstream,
            artifact_field=field,
            op="==",
            expected=expected,
            observed=observed,
            passed=expected == observed,
        )
    )


def empty_work() -> Json:
    """Describe unattempted work explicitly so blocked runs retain every field."""
    return dict(
        checks=[],
        refs=[],
        counts=dict(ZERO_INVOCATION_COUNTS),
        device_inventory=[],
        inherited_mask={k: os.environ[k] for k in MASKS if k in os.environ},
        child_mask={},
        uuid_to_local_index={},
        gpu_lease_receipt={},
        backend_identity={},
        offloaded_layers=None,
        resident_gpu_receipt={},
        load_spans=[],
        cleanup_receipt={},
        cuda_failure_comparison={},
        phase_spans=[],
        duration_s=0,
        invocation_argv=list(sys.argv),
    )


def permitted(rows: list[Json], env: Json) -> list[Json]:
    """Intersect resource masks without assuming an unavailable physical ordinal."""
    selected = rows
    for key in MASKS[:2]:
        mask = env.get(key)
        if mask is None or (key == "NVIDIA_VISIBLE_DEVICES" and mask == "all"):
            continue
        tokens = mask.split(",")
        selected = [r for r in selected if str(r["index"]) in tokens or r["uuid"] in tokens]
    return [r for r in selected if r["memory_used_mb"] < 1000 and r["memory_free_mb"] >= 20000]


def device_flags(help_text: str, enumeration: str, local: int) -> list[str]:
    """Use observed CLI capabilities and require the no-warmup safety flag."""
    if "--no-warmup" not in help_text:
        raise ValueError("no_warmup_flag_unavailable")
    if not re.search(rf"\bCUDA{local}\s*:", enumeration):
        raise ValueError("native_cuda_device_unavailable:" + enumeration)
    if "--device" not in help_text or "--main-gpu" not in help_text:
        raise ValueError("native_device_flags_unavailable")
    return ["--device", f"CUDA{local}", "--main-gpu", str(local), "--no-warmup"]


def load_errors(props: Json, model: Path, offload: list[int]) -> list[str]:
    """A healthy wrong-model or CPU server cannot qualify the declared backend."""
    return [
        name
        for name, ok in [
            ("model_path", props.get("model_path") == str(model)),
            ("embedded_chat_template", bool(props.get("chat_template"))),
            ("gpu_offload", offload[0] > 0 and offload[0] == offload[1]),
        ]
        if not ok
    ]


def residency(text: str, uuid: str, owner: Json) -> Json:
    """Join GPU compute residency to the launched PID, never aggregate VRAM."""
    rows = [line.split(",") for line in text.splitlines() if len(line.split(",")) == 3]
    matching = [r for r in rows if r[0].strip() == uuid and r[1].strip() == str(owner["pid"])]
    memory = sum(int(r[2].strip()) for r in matching)
    if memory < 10000:
        raise ValueError("pid_bound_residency:" + text)
    return dict(
        uuid=uuid,
        pid=owner["pid"],
        start_time_ticks=owner["start_time_ticks"],
        resident_memory_mb=memory,
    )


def load_child(plan: Json, raw: Path, private: Path) -> Json:
    """Bind only this fresh process to the lease UUID and retain exact load evidence.

    The parent retains its environment and the lease lock. Only GET metadata is
    allowed; --no-warmup prevents llama.cpp's usual startup forward pass.
    """
    result = empty_work()
    model, binary = Path(plan["model"]), Path(plan["binary"])
    lease = plan["lease"]
    uuid = lease["device_uuid"]
    result.update(
        child_mask={"CUDA_VISIBLE_DEVICES": uuid, "CARNOT_FORCE_LIVE": "1"},
        uuid_to_local_index={uuid: 0},
        gpu_lease_receipt=lease,
    )
    os.environ.update(result["child_mask"])
    server: Any = None
    start = time.monotonic_ns()
    try:
        if proc_start_ticks(lease["pid"]) != lease["pid_start_ticks"]:
            raise ValueError("lease_owner_identity")
        journal = json.loads(Path(plan["journal"]).read_bytes())
        if (
            journal["released"]
            or journal["lease_id"] != lease["lease_id"]
            or journal["expected_model"] != str(model)
            or time.monotonic_ns() > journal["expires_monotonic_ns"]
        ):
            raise ValueError("lease_custody")
        if sha256_file(model) != MODEL_PIN or sha256_file(binary) != plan["binary_sha256"]:
            raise ValueError("model_or_backend_hash")
        help_receipt = child("native_help", [str(binary), "--help"], raw, deadline=20)
        enum_receipt = child("native_devices", [str(binary), "--list-devices"], raw, deadline=20)
        result["backend_identity"] = dict(
            reference(binary), help_receipt=help_receipt, enumeration_receipt=enum_receipt
        )
        result["backend_identity"]["installed_library_hashes"] = [
            reference(p) for p in sorted(binary.parent.glob("lib*.so*")) if p.is_file()
        ]
        flags = device_flags(
            Path(help_receipt["stdout_path"]).read_text(),
            Path(enum_receipt["stdout_path"]).read_text()
            + Path(enum_receipt["stderr_path"]).read_text(),
            0,
        )
        if not help_receipt["passed"] or not enum_receipt["passed"]:
            raise ValueError("native_enumeration_exit")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = [
            str(binary),
            "-m",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-c",
            "512",
            "-ngl",
            "99",
            "--parallel",
            "1",
            "--jinja",
            *flags,
        ]
        server = OwnedLlamaCppProcess(
            command=command,
            port=port,
            env=dict(os.environ),
            log_path=raw / "server.log",
            state_path=private / "owner.json",
        )
        progress("before_model_load", 0, 1)
        result["counts"]["model_loads_attempted"] = 1
        owner = server.launch()
        result["backend_identity"].update(
            owner=owner,
            command=command,
            launch_mask=result["child_mask"],
            launch_environment_sha256=canonical_hash(dict(os.environ)),
        )
        health = bounded(
            lambda: server.wait_for_health(max(0.01, 600 - (time.monotonic_ns() - start) / 1e9)),
            600,
        )
        result["backend_identity"]["health"] = health
        if not health["ok"]:
            raise ValueError("native_health:" + json.dumps(health))
        result["counts"]["model_loads_completed"] = 1
        props = get_json(f"http://127.0.0.1:{port}/props")
        result["backend_identity"]["props"] = props
        atomic_json(raw / "native_props.json", props)
        offload = offload_layers((raw / "server.log").read_text())
        result["offloaded_layers"] = offload
        errors = load_errors(props, model, offload)
        if errors:
            raise ValueError("native_load_identity:" + ",".join(errors))
        result["backend_identity"]["loaded_libraries"] = loaded_libraries(owner, raw)
        resident = child(
            "resident_gpu",
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            raw,
            deadline=10,
        )
        if not resident["passed"] or proc_start_ticks(owner["pid"]) != owner["start_time_ticks"]:
            raise ValueError("resident_identity_or_query")
        result["resident_gpu_receipt"] = dict(
            residency(Path(resident["stdout_path"]).read_text(), uuid, owner),
            command_receipt=resident,
        )
        result["counts"]["gpu_model_loads_completed"] = 1
        gate(result, raw / "server.log", "gguf_backend", True, True, "native_backend")
    except (OSError, ValueError, RuntimeError, TimeoutError, KeyError) as error:
        diagnostic = raw / "server.log"
        if not diagnostic.is_file() and (raw / "native_devices.stderr").is_file():
            diagnostic = raw / "native_devices.stderr"
        gate(
            result,
            diagnostic,
            "gguf_backend",
            True,
            f"{type(error).__name__}:{error}",
            "native_backend",
        )
    finally:
        end = time.monotonic_ns()
        result["load_spans"] = [
            dict(
                phase="cold_load",
                started_monotonic_ns=start,
                ended_monotonic_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ]
        progress("after_model_load", result["counts"]["model_loads_completed"], 0)
        progress("before_model_shutdown", 0, int(server is not None))
        result["cleanup_receipt"] = (
            server.cleanup()
            if server
            else dict(leak_free=True, action="not_started", unrelated_process_kill_count_delta=0)
        )
        result["load_spans"].append(
            dict(
                phase="shutdown",
                started_monotonic_ns=end,
                ended_monotonic_ns=time.monotonic_ns(),
                duration_s=(time.monotonic_ns() - end) / 1e9,
            )
        )
        result["counts"]["model_loads_failed"] = (
            result["counts"]["model_loads_attempted"] - result["counts"]["model_loads_completed"]
        )
        result["duration_s"] = (time.monotonic_ns() - start) / 1e9
        gate(
            result,
            raw / "server.log",
            "owned_cleanup",
            True,
            result["cleanup_receipt"]["leak_free"],
            "owned",
        )
        atomic_json(raw / "load_result.json", result)
        progress("after_model_shutdown", int(server is not None), 0)
    return result


def authenticate(root: Path, raw: Path, work: Json) -> None:
    """Authenticate imported identities and preserve the old failed preflight bytes."""
    for name in [
        "experiment_8264_v714_evidence_view_canary",
        "experiment_8276_v715_current_contract_readiness",
    ]:
        path = root / "results" / (name + ".json")
        gate(work, path, "exists", True, path.is_file(), name)
        if not path.is_file():
            continue
        value = json.loads(path.read_bytes())
        terminal = Path(value["terminal_validation_sidecar_path"])
        report = json.loads(terminal.read_bytes())
        sidecar = Path(report["publication"]["sidecar_path"])
        bound = read_bound_sidecar(path, sidecar)
        gate(
            work,
            terminal,
            "publication.primary_sha256",
            sha256_file(path),
            report["publication"]["primary_sha256"],
            name,
        )
        gate(work, sidecar, "report.passed", True, bound["report"]["passed"], name)
        work["refs"].extend(
            dict(reference(p), fields_imported=fields)
            for p, fields in [
                (path, ["experiment_id", "task_id", "cuda_preflight_receipt"]),
                (terminal, ["publication"]),
                (sidecar, ["report.passed"]),
            ]
        )
        if value["experiment_id"] == 8264:
            receipt = value["cuda_preflight_receipt"]
            comparison = dict(
                historical_receipt=receipt,
                observed_facts=["Exp8264 PyTorch preflight exited before llama-server launch."],
                hypotheses=[
                    "Device ordinal/mask mismatch or CUDA runtime/driver state; Error101 cause is unproved."
                ],
                causality_proved=False,
            )
            for stream in ["stdout", "stderr"]:
                source = Path(receipt[stream + "_path"])
                gate(
                    work,
                    source,
                    stream + "_sha256",
                    receipt[stream + "_sha256"],
                    sha256_file(source),
                    name,
                )
                target = raw / ("exp8264_preflight." + stream)
                target.write_bytes(source.read_bytes())
                comparison[stream + "_verbatim"] = target.read_text()
                work["refs"].append(dict(reference(target), fields_imported=["verbatim bytes"]))
            atomic_json(raw / "exp8264_preflight.receipt.json", receipt)
            work["cuda_failure_comparison"] = comparison


def measure(root: Path, raw: Path) -> Json:
    """Hold existing lease custody while a fresh, bounded child tests the model."""
    work = empty_work()
    raw.mkdir(parents=True, exist_ok=True)
    start, wall = time.monotonic_ns(), time.time_ns()
    progress("before_input_authentication")
    lease: Any = None
    try:
        authenticate(root, raw, work)
        model_spec = cached_current_model() or {}
        model = Path(model_spec.get("model_path", "/missing-qwen-cache"))
        binary = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
        gate(work, model, "gguf_cache", True, model.is_file(), "qwen_cache")
        gate(work, binary, "backend_binary", True, binary.is_file(), "native_backend")
        if all(c["passed"] for c in work["checks"]):
            progress("before_model_hash", 0, 1)
            digest = bounded(lambda: sha256_file(model), 120)
            metadata = read_gguf_metadata(model)
            gate(work, model, "gguf_sha256", MODEL_PIN, digest, "qwen_cache")
            gate(work, model, "quantization", "Q4_K_M", metadata["quantization"], "qwen_cache")
            work["refs"].append(
                dict(
                    reference(model),
                    fields_imported=["frozen weights", "embedded tokenizer/chat template"],
                )
            )
            progress("after_model_hash", 1, 0)
        progress("after_input_authentication", len(work["checks"]), 0)
        inventory = child(
            "device_inventory",
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            raw,
            deadline=10,
            scope="external",
        )
        gate(
            work,
            Path(inventory["stdout_path"]),
            "device_inventory_exit",
            0,
            inventory["exit_code"],
            "nvidia-smi",
        )
        work["device_inventory_receipt"] = inventory
        rows = [
            r.split(",")
            for r in Path(inventory["stdout_path"]).read_text().splitlines()
            if len(r.split(",")) == 5
        ]
        work["device_inventory"] = [
            dict(
                index=int(r[0]),
                uuid=r[1].strip(),
                name=r[2].strip(),
                memory_used_mb=int(r[3]),
                memory_free_mb=int(r[4]),
            )
            for r in rows
        ]
        choices = permitted(work["device_inventory"], work["inherited_mask"])
        refusals = []
        if all(c["passed"] for c in work["checks"]):
            for index, device in enumerate(choices):
                progress("lease_selection", index, len(choices) - index)
                try:
                    lease = GpuLease.acquire(
                        runtime_dir="/tmp/carnot-gpu-leases",
                        task_id=TASK,
                        device_uuid=device["uuid"],
                        expected_model=str(model),
                        vram_before_mb=device["memory_used_mb"],
                        ttl_s=900,
                    )
                    break
                except LeaseError as error:
                    refusals.append(str(error))
            gate(
                work,
                Path(inventory["stdout_path"]),
                "permitted_gpu_lease",
                True,
                True
                if lease
                else dict(
                    permitted_uuids=[r["uuid"] for r in choices],
                    refusals=refusals,
                    inherited_mask=work["inherited_mask"],
                ),
                "gpu_lease",
            )
        if lease:
            work["gpu_lease_receipt"] = lease.owner_receipt()
            lease.transition("admitted")
            lease.transition("loading")
            plan = dict(
                model=str(model),
                binary=str(binary),
                binary_sha256=sha256_file(binary),
                lease=lease.owner_receipt(),
                journal=str(lease.journal_path),
            )
            atomic_json(raw / "load_plan.json", plan)
            with TemporaryDirectory(prefix="carnot8277-load-") as directory:
                receipt = child(
                    "leased_load_child",
                    [
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        str(ROOT / CLI),
                        "--load-child",
                        str(raw / "load_plan.json"),
                        "--private",
                        directory,
                    ],
                    raw,
                    deadline=650,
                )
            work["load_child_receipt"] = receipt
            if (raw / "load_result.json").is_file():
                result = json.loads((raw / "load_result.json").read_bytes())
                work["checks"].extend(result["checks"])
                for key in [
                    "counts",
                    "child_mask",
                    "uuid_to_local_index",
                    "backend_identity",
                    "offloaded_layers",
                    "resident_gpu_receipt",
                    "load_spans",
                    "cleanup_receipt",
                ]:
                    work[key] = result[key]
            gate(
                work,
                Path(receipt["stdout_path"]),
                "owned_child_normal_exit",
                True,
                receipt["passed"],
                "owned",
            )
    except (OSError, ValueError, RuntimeError, TimeoutError, KeyError) as error:
        gate(
            work, root, "authenticated_operand", True, f"{type(error).__name__}:{error}", "external"
        )
    finally:
        if lease:
            if work["counts"].get("gpu_model_loads_completed") and work["cleanup_receipt"].get(
                "leak_free"
            ):
                lease.transition(
                    "resident", vram_mb=work["resident_gpu_receipt"]["resident_memory_mb"]
                )
                lease.transition("unloading")
                after = child(
                    "after_gpu",
                    ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                    raw,
                    deadline=10,
                    scope="external",
                )
                after_rows = [
                    r.split(",")
                    for r in Path(after["stdout_path"]).read_text().splitlines()
                    if len(r.split(",")) == 2 and r.split(",")[0].strip() == lease.device_uuid
                ]
                measured_after = int(after_rows[0][1]) if after_rows and after["passed"] else None
                work["cleanup_receipt"].update(
                    after_gpu_receipt=after, device_memory_after_mb=measured_after
                )
                gate(
                    work,
                    Path(after["stdout_path"]),
                    "shutdown_inventory",
                    True,
                    measured_after is not None,
                    "native_backend",
                )
                lease.transition(
                    "validating",
                    vram_mb=measured_after if measured_after is not None else -1,
                    exit_code=receipt["exit_code"],
                    unload_observed=True,
                )
                # -1 marks missing device memory; readiness is blocked by the exact gate above.
                lease.transition(
                    "terminal_complete"
                    if all(c["passed"] for c in work["checks"])
                    else "terminal_blocked"
                )
            else:
                lease.transition("terminal_blocked")
            work["gpu_lease_receipt"]["release"] = lease.release()
            atomic_json(raw / "gpu_lease_journal.json", lease.document)
        end = time.monotonic_ns()
        work["duration_s"] = (end - start) / 1e9
        work["phase_spans"] = [
            dict(
                phase="measurement",
                started_monotonic_ns=start,
                ended_monotonic_ns=end,
                started_wall_ns=wall,
                duration_s=work["duration_s"],
            )
        ]
        work["cuda_failure_comparison"]["native_observation"] = work["checks"]
        atomic_json(raw / "measurement.json", work)
        binding = {
            k: work[k]
            for k in [
                "child_mask",
                "uuid_to_local_index",
                "gpu_lease_receipt",
                "backend_identity",
                "resident_gpu_receipt",
                "cleanup_receipt",
            ]
        }
        binding.update(
            model_specs=MODEL_SPECS,
            generation_permitted=False,
            future_capacity_promised=False,
            required_each_run=[
                "new permitted lease",
                "rehash model/backend",
                "PID/start identity",
                "offload/residency/health",
            ],
        )
        atomic_json(raw / "backend_binding.json", binding)
        progress("after_measurement", work["counts"]["model_loads_completed"], 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Execution readiness requires every owned check; it implies no benefit."""
    owned = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and all(c["passed"] for c in work["checks"] if c["upstream"] == "owned")
    )
    failed = [c for c in work["checks"] if not c["passed"]]
    native = work["counts"].get("gpu_model_loads_completed", 0) == 1
    ready = int(
        owned
        and not failed
        and native
        and work["cleanup_receipt"].get("leak_free") is True
        and work["gpu_lease_receipt"].get("release", {}).get("released") is True
        and work["duration_s"] >= 2
        and work["counts"]["generation_calls_attempted"] == 0
    )
    verdict = (
        "disqualified" if not owned else "blocked" if failed or not ready else "circular_positive"
    )
    suffix = failed[0]["artifact_field"] if failed else "gguf_backend"
    row = dict(
        unit_id="leased_qwen_backend",
        source_cluster_id="host_gpu",
        condition="cold_load_no_generation",
        arm="native_llama_server",
        status="completed"
        if ready
        else "failed"
        if work["counts"]["model_loads_attempted"]
        else "censored",
        numerator=ready if work["counts"]["model_loads_attempted"] else None,
        denominator=1,
        metric="backend_ready",
    )
    value = dict(
        experiment_id=8277,
        task_id=TASK,
        milestone="2026.10.715",
        run_date="20261008",
        honest_verdict="complete_" + verdict + "_" + suffix,
        verdict_class=verdict,
        gate_check_summary=work["checks"],
        inference_substrate="live_llm_inference"
        if work["counts"]["model_loads_attempted"]
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="model_load_no_generation" if native else "blocked_no_run",
        load_only_contract=dict(
            class_="model_load_no_generation", minimum_duration_s=2, generation_permitted=False
        ),
        inference_mode="live_gpu_gguf" if native else "no_gpu_load_confirmed",
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=work["counts"],
        rows=[row],
        intended_count=1,
        completed_count=int(ready),
        failed_count=int(row["status"] == "failed"),
        censored_count=int(row["status"] == "censored"),
        excluded_count=0,
        independent_count=0,
        verifier_is_oracle=True,
        exposure_scope="exposed_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=any(
            r["name"] == "adversarial_verify" and not r["passed"] for r in receipts
        ),
        acceptance_gates=dict(
            owned_checks=owned,
            native_gpu_load=native,
            complete_lease_qualification=bool(ready),
            generation_attempts_zero=work["counts"]["generation_calls_attempted"] == 0,
            scientific_benefit=False,
        ),
        validation_receipts=receipts,
        repository_health=work.get("repository_health", {}),
        repository_health_reused=work.get("repository_health_reused", False),
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["checks"],
        duration_s=work["duration_s"] + sum(r.get("duration_s", 0) for r in receipts),
        random_seed=7158277,
        source_artifact_hashes=work["refs"],
        code_config_hashes=[
            reference(ROOT / p)
            for p in OWNED
            + [
                TEST,
                "python/carnot/inference/llama_cpp_process.py",
                "python/carnot/gpu_lease_phase_journal.py",
                "python/carnot/verify/evidence_view_execution_8264.py",
            ]
        ],
        raw_shard_hashes=[
            reference(p)
            for p in sorted(raw.rglob("*"))
            if p.is_file()
            and p.name not in {"terminal_validation.json", "terminal_candidate.json"}
            and "validation" not in p.parts
            and "health" not in p.parts
            and "terminal" not in p.parts
        ],
        phase_spans=work["phase_spans"] + work["load_spans"],
        cited_upstream_artifacts=work["refs"],
        gguf_backend_ready_score=ready,
        backend_binding_path=str(raw / "backend_binding.json"),
        backend_binding_sha256=reference(raw / "backend_binding.json")["sha256"],
        methodology_note="One leased host GPU; native load-only qualification with no warm-up, completion or embedding. Real backend CUDA and PID-bound residency required. Error101 cause unproved. No scientific benefit or hardware-board claim.",
        invocation_argv=work["invocation_argv"],
    )
    value.update(
        {
            k: work[k]
            for k in [
                "device_inventory",
                "inherited_mask",
                "child_mask",
                "uuid_to_local_index",
                "gpu_lease_receipt",
                "backend_identity",
                "offloaded_layers",
                "resident_gpu_receipt",
                "load_spans",
                "cleanup_receipt",
                "cuda_failure_comparison",
            ]
        }
    )
    value["load_only_contract"]["class"] = value["load_only_contract"].pop("class_")
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Bind actual bytes, invocation and observed evidence; missing operands are not measured zero."
        for k in value
    }
    value["field_principles"]["gguf_backend_ready_score"] = (
        "Current execution qualification only; recheck lease and identity each run. No inference benefit or future capacity is promised."
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return dict(value)


def replay(path: Path) -> bool:
    """Rebuild the projection before rehashing operands to reject rehashed edits."""
    try:
        value = json.loads(path.read_bytes())
        raw = Path(value["backend_binding_path"]).parent
        work = json.loads((raw / "measurement.json").read_bytes())
        if (raw / "load_result.json").is_file():
            loaded = json.loads((raw / "load_result.json").read_bytes())
            if any(
                work[key] != loaded[key]
                for key in [
                    "counts",
                    "child_mask",
                    "uuid_to_local_index",
                    "backend_identity",
                    "offloaded_layers",
                    "resident_gpu_receipt",
                    "load_spans",
                ]
            ):
                return False
            if any(
                work["cleanup_receipt"].get(key) != val
                for key, val in loaded["cleanup_receipt"].items()
            ):
                return False
            if work["counts"].get("gpu_model_loads_completed"):
                if (
                    offload_layers((raw / "server.log").read_text()) != work["offloaded_layers"]
                    or json.loads((raw / "native_props.json").read_bytes())
                    != work["backend_identity"]["props"]
                ):
                    return False
        if build(work, raw, value["validation_receipts"]) != value:
            return False
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if (
                    stream + "_path" in receipt
                    and sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]
                ):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path, raw: Path) -> list[Json]:
    """Freeze owned commands before measurement, including child and CLI coverage."""
    config = private / "coverage.ini"
    config.write_text(
        f"[run]\ndata_file={private / 'coverage.data'}\nparallel=True\npatch=subprocess,_exit\ninclude=\n    */lease_backend_8277.py\n    */{NAME}.py\n[report]\nexclude_lines=\n"
    )
    py = str(ROOT / ".venv/bin/python")
    cov = str(ROOT / ".venv/bin/coverage")
    tests = [str(ROOT / ".venv/bin/pytest"), "-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    commands = [
        (
            "owned_unit_coverage",
            [cov, "run", "--rcfile", str(config), "-m", "pytest", *tests[1:], TEST],
        ),
        ("coverage_combine", [cov, "combine", "--rcfile", str(config)]),
        ("coverage_json", [cov, "json", "--rcfile", str(config), "-o", str(raw / "coverage.json")]),
        (
            "changed_statement_coverage",
            [cov, "report", "--rcfile", str(config), "--fail-under=100", "-m"],
        ),
        (
            "consumer_E2E015_019",
            [
                *tests,
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_source_boundary_7852.py",
                "tests/python/test_experiment_7942_v689_sentence_labels.py",
            ],
        ),
        ("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST]),
        ("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST]),
        (
            "strict_mypy",
            [
                str(ROOT / ".venv/bin/mypy"),
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=skip",
                "--ignore-missing-imports",
                *OWNED,
            ],
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", "--files", TEST]),
    ]
    return [dict(name=name, argv=argv, deadline=300) for name, argv in commands]


def terminal(candidate: Path, raw: Path, private: Path) -> Json:
    """Validate identical private bytes and real cold-replay negative controls."""
    receipts = []
    py = str(ROOT / ".venv/bin/python")
    specs = [
        ("adversarial_verify", [py, "scripts/adversarial_verify.py", "--json", str(candidate)], 0),
        (
            "strict_rows",
            [py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
            0,
        ),
        ("cold_replay", [py, "-u", CLI, "--cold-replay", str(candidate)], 0),
    ]
    for name, argv, expected in specs:
        receipts.append(child(name, argv, raw / "terminal", deadline=180, expected=expected))
    value = json.loads(candidate.read_bytes())
    for name in ["rehashed_tamper", "negative_control"]:
        bad = dict(value)
        bad["gguf_backend_ready_score"] = 1 - value["gguf_backend_ready_score"]
        bad["model_invocation_counts"] = dict(
            value["model_invocation_counts"], generation_calls_attempted=1
        )
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        path = private / (name + ".json")
        atomic_json(path, bad if name == "rehashed_tamper" else {})
        receipts.append(
            child(
                name,
                [py, "-u", CLI, "--cold-replay", str(path)],
                raw / "terminal",
                deadline=180,
                expected=1,
            )
        )
    return dict(
        passed=all(r["passed"] for r in receipts),
        candidate_sha256=sha256_file(candidate),
        receipts=receipts,
    )


def main(argv: list[str] | None = None) -> int:
    """Run bounded owned work, validate privately, and publish checked bytes only."""
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261008"], default="20261008")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--load-child", type=Path)
    parser.add_argument("--private", type=Path)
    parser.add_argument("--repository-health-receipt", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        progress("replay_passed" if passed else "reduction_drift")
        return 0 if passed else 1
    if args.load_child:
        load_child(json.loads(args.load_child.read_bytes()), args.load_child.parent, args.private)
        return 0
    output = args.output.absolute()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="carnot8277-validation-") as directory:
        private = Path(directory)
        candidate = private / (NAME + ".json")
        specs = manifest(private, candidate, raw)
        atomic_json(
            raw / "validation_commands.json",
            dict(
                commands=specs,
                repository_health=dict(
                    argv=[str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"], deadline_s=180
                ),
                terminal_candidate=str(candidate),
            ),
        )
        work = measure(args.root, raw)
        if output.exists():
            previous = raw / "previous_primary.json"
            previous.write_bytes(output.read_bytes())
            work["refs"].append(
                dict(reference(previous), fields_imported=["prior attempt bytes; historical only"])
            )
        receipts = []
        for index, spec in enumerate(specs):
            progress("owned_validation", index, len(specs) - index)
            receipts.append(
                child(spec["name"], spec["argv"], raw / "validation", deadline=spec["deadline"])
            )
        if args.repository_health_receipt:
            health = json.loads(args.repository_health_receipt.read_bytes())
            work["refs"].append(
                dict(
                    reference(args.repository_health_receipt),
                    fields_imported=["argv", "actual_exit", "clocks", "stream hashes", "passed"],
                )
            )
            work["repository_health_reused"] = True
        else:
            health = child(
                "full_python_suite",
                [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
                raw / "health",
                deadline=180,
                scope="repository_health",
            )
        work["repository_health"] = health
        atomic_json(raw / "measurement.json", work)
        atomic_json(raw / "repository_health.json", health)
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
        value = build(work, raw, receipts)
        atomic_json(candidate, value)
        report = terminal(candidate, raw, private)
        if not report["passed"]:
            progress("terminal_validation_failed")
            atomic_json(raw / "failed_terminal.json", report)
            return 1
        publication = publish_primary(
            output,
            value,
            lambda path: dict(report, passed=sha256_file(path) == report["candidate_sha256"]),
        )
        atomic_json(raw / "terminal_validation.json", dict(publication=publication, report=report))
    progress("complete", 1, 0)
    return 0
