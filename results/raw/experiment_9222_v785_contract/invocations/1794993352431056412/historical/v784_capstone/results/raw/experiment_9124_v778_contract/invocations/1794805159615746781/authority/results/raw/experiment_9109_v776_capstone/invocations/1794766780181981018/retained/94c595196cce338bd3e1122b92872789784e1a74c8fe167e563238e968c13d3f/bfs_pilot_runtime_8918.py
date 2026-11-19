"""REQ-REPORT-8918: own one GPU-0 server and preserve every bounded call."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

from carnot.gpu_lease_phase_journal import GpuLease, LeaseError
from carnot.inference.qwen_sufficiency_7920 import QwenRuntime, bounded
from carnot.inference.sota_models import cached_current_model
from carnot.reporting import bfs_error_pilot_8918 as p
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify.qwen_development_capture_7995 import Ledger

Json = dict[str, Any]


def gpu_state() -> Json:
    """Inspect only device zero; the detached second device is outside this task."""
    import subprocess

    output = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=index,uuid,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    index, uuid, used, free = output.stdout.strip().split(",")
    return dict(
        index=int(index),
        uuid=uuid.strip(),
        used_mb=int(used),
        free_mb=int(free),
        stdout=output.stdout,
    )


def preflight(raw: Path) -> Json:
    """Hash the mandated cache and authenticate support without loading weights."""
    import llama_cpp

    spec = cached_current_model(gpu_index=0, preferred_quant="Q4_K_M")
    if spec is None:
        raise RuntimeError("cache_missing")
    path = Path(spec["model_path"])
    if (
        spec["hf_id"] != p.SETTINGS["model"]
        or not path.is_file()
        or not path.name.endswith("Q4_K_M.gguf")
    ):
        raise RuntimeError("mandated_Q4_K_M_missing")
    p.progress("before_gguf_hash")
    spec["sha256"] = sha256_file(path)
    p.progress("after_gguf_hash")
    supported = bool(llama_cpp.llama_supports_gpu_offload())
    result = dict(
        model=spec,
        gpu=gpu_state(),
        gpu_offload_supported=supported,
        llama_cpp=p.reference(Path(llama_cpp.__file__)),
    )
    atomic_json(raw / "preflight.json", result)
    if not supported:
        raise RuntimeError("gpu_offload_unavailable")
    return result


def run(graphs: list[Json], raw: Path) -> Json:
    """Fail closed without repair, retries, CPU fallback or a second device."""
    os.environ.update(CARNOT_FORCE_LIVE="1", CUDA_VISIBLE_DEVICES="0")
    raw.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(raw / "invocation_ledger.json")
    ledger.save()
    result: Json = dict(
        responses=[],
        checks=[],
        live=False,
        blocker=None,
        model_receipt={},
        resource_lease={},
        phase_spans=[],
        settings=dict(p.SETTINGS),
        owned_failure=False,
    )
    lease: Any = None
    runtime: Any = None
    stage = "preflight"
    path = raw / "preflight.json"
    try:
        began = time.monotonic()
        p.progress("before_preflight")
        resources = bounded(lambda: preflight(raw), 240)
        result["preconditions_checked"] = resources
        result["phase_spans"].append(dict(phase=stage, duration_s=time.monotonic() - began))
        p.progress("after_preflight")
        stage = "gpu_lease"
        spec, gpu = resources["model"], resources["gpu"]
        path = Path(spec["model_path"])
        lease = GpuLease.acquire(
            runtime_dir=os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"),
            task_id=p.TASK,
            device_uuid=gpu["uuid"],
            expected_model=str(path),
            vram_before_mb=gpu["used_mb"],
            ttl_s=2100,
        )
        result["resource_lease"] = lease.owner_receipt()
        locked = gpu_state()
        result["resource_lease"]["locked_gpu_state"] = locked
        if (
            locked["uuid"] != gpu["uuid"]
            or locked["index"] != 0
            or locked["used_mb"] >= 1000
            or locked["free_mb"] < 20000
        ):
            raise RuntimeError("gpu_0_not_free")
        lease.transition("admitted")
        lease.transition("loading")
        stage = "model_load"
        runtime = QwenRuntime(path, raw, 0)
        ledger.start("model_load", "qwen-load", spec)
        p.progress("before_model_load")
        began = time.monotonic()
        try:
            loaded = bounded(runtime.load, 240)
        except (RuntimeError, OSError, ValueError, TimeoutError) as error:
            ledger.finish("qwen-load", "failed", dict(error=str(error)))
            raise
        ledger.finish("qwen-load", "completed", loaded)
        result["phase_spans"].append(dict(phase=stage, duration_s=time.monotonic() - began))
        result["model_receipt"] = dict(loaded, preflight_gguf_sha256=spec["sha256"])
        p.progress("after_model_load")
        if (
            loaded["gguf_sha256"] != spec["sha256"]
            or not loaded["authenticated"]
            or loaded["offload_layers"][0] <= 0
            or loaded["offload_layers"][0] != loaded["offload_layers"][1]
        ):
            raise RuntimeError("actual_offload_or_hash_rejected")
        result["settings"].update(
            chat_template=loaded["props"]["chat_template"],
            chat_template_sha256=canonical_hash(loaded["props"]["chat_template"]),
            parser="fullmatch whitespace uppercase two letters optional . or ! whitespace",
            per_question_timeout_s=30,
            repairs=0,
            grammar=None,
        )
        atomic_json(raw / "generator_settings.json", result["settings"])
        result["settings_ref"] = p.reference(raw / "generator_settings.json")
        lease.transition("resident", vram_mb=gpu_state()["used_mb"])
        lease.transition("inferencing")
        generation_started = time.monotonic()
        for index, graph in enumerate(graphs):
            if time.monotonic() - generation_started >= 1500:
                raise TimeoutError("pilot_budget")
            stage = "generation"
            payload = dict(p.SETTINGS, messages=[dict(role="user", content=graph["prompt"])])
            atomic_json(raw / "requests" / (graph["unit_id"] + ".json"), payload)
            ledger.start("generation", graph["unit_id"], payload)
            p.progress("before_generation", index, 50 - index)
            began = time.monotonic()
            try:
                response = bounded(
                    lambda: runtime.worker.post_json("/v1/chat/completions", payload, 30),
                    30,
                    interval_s=10,
                )
                answer = response["choices"][0]["message"].get("content") or ""
                if not isinstance(answer, str) or not isinstance(
                    response["usage"]["completion_tokens"], int
                ):
                    raise ValueError("malformed_response")
            except (RuntimeError, OSError, ValueError, KeyError, IndexError, TimeoutError) as error:
                ledger.finish(graph["unit_id"], "failed", dict(error=str(error)))
                result["responses"].append(
                    dict(
                        unit_id=graph["unit_id"],
                        status="runtime_failed",
                        missing_reason=str(error),
                        duration_s=time.monotonic() - began,
                    )
                )
                raise RuntimeError("generation_censored") from error
            ledger.finish(graph["unit_id"], "completed", response)
            row = dict(
                unit_id=graph["unit_id"],
                status="completed",
                answer=answer,
                usage=response["usage"],
                duration_s=time.monotonic() - began,
                raw_response=response,
                request_sha256=canonical_hash(payload),
            )
            result["responses"].append(row)
            atomic_json(raw / "responses" / (graph["unit_id"] + ".json"), row)
            result["phase_spans"].append(
                dict(phase="generation", unit_id=graph["unit_id"], duration_s=row["duration_s"])
            )
            result["live"] = True
            lease.heartbeat()
            p.progress("after_generation", index + 1, 49 - index)
            if (index + 1) % 10 == 0:
                p.progress("pilot_checkpoint", index + 1, 49 - index)
    except (RuntimeError, OSError, ValueError, KeyError, TimeoutError, LeaseError) as error:
        result["blocker"] = str(error)
        result["checks"].append(
            dict(
                check=stage,
                upstream=p.TASK,
                artifact_field=stage,
                expected="authenticated available GPU-0 Qwen runtime",
                observed=str(error),
                disposition="missing_file" if "missing" in str(error) else "observed_zero",
                path=str(path),
                hash=sha256_file(path) if path.is_file() else None,
            )
        )
        p.progress("runtime_blocked_" + stage)
    finally:
        if runtime is not None:
            p.progress("before_model_shutdown")
            cleanup = runtime.close()
            result["cleanup"] = cleanup
            result["owned_failure"] = not cleanup["leak_free"]
            p.progress("after_model_shutdown")
        if lease is not None:
            if lease.document["phase"] == "inferencing":
                lease.transition("unloading")
                lease.transition(
                    "validating",
                    vram_mb=gpu_state()["used_mb"],
                    exit_code=0,
                    unload_observed=not result["owned_failure"],
                )
            lease.transition("terminal_blocked" if result["blocker"] else "terminal_complete")
            result["resource_lease"]["release"] = lease.release()
        result["counts"] = ledger.counts()
        result["ledger"] = ledger.rows
        atomic_json(raw / "runtime.json", result)
    return result
