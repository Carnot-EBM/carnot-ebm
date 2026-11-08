"""REQ-REPORT-8924: own one bounded Qwen extraction per frozen natural case."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
from typing import Any

from carnot.gpu_lease_phase_journal import GpuLease, LeaseError
from carnot.inference.bfs_pilot_runtime_8918 import gpu_state, preflight
from carnot.inference.qwen_sufficiency_7920 import QwenRuntime, bounded
from carnot.reporting import qwen_natural_canary_8924 as c
from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify.qwen_development_capture_7995 import Ledger

Json = dict[str, Any]


def run(evidence: Json, raw: Path) -> Json:
    """Reacquire device zero; stop on resource drift without repair or fallback."""
    os.environ.update(CARNOT_FORCE_LIVE="1", CUDA_VISIBLE_DEVICES="0")
    raw.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(raw / "invocation_ledger.json")
    ledger.save()
    result: Json = dict(
        outputs=[],
        model_receipt={},
        resource_lease={},
        phase_spans=[],
        checks=[],
        blocker=None,
        live=False,
        owned_failure=False,
        generation_duration_s=0.0,
        settings=c.SETTINGS,
    )
    runtime: Any = None
    lease: Any = None
    stage, path = "preflight", raw
    generation_started: float | None = None
    try:
        c.progress("before_preflight", 0, len(evidence["cases"]))
        resources = bounded(lambda: preflight(raw), 240)
        result["preconditions_checked"] = resources
        prior = evidence["pilot"]
        path = Path(resources["model"]["model_path"])
        c.require(
            "GGUF_hash",
            prior["model_receipt"]["gguf_sha256"],
            resources["model"]["sha256"],
            path,
            "exp8918-bfs-error-pilot",
        )
        c.require(
            "binding_hash",
            prior["preconditions_checked"]["llama_cpp"]["sha256"],
            resources["llama_cpp"]["sha256"],
            path,
            "exp8918-bfs-error-pilot",
        )
        backend = Path(prior["model_receipt"]["command"][0])
        c.progress("before_backend_version")
        version = subprocess.run(
            [str(backend), "--version"], capture_output=True, text=True, check=True, timeout=15
        )
        c.progress("after_backend_version")
        build = prior["model_receipt"]["props"]["build_info"]
        c.require(
            "backend_build",
            True,
            build.split("-", 1)[-1] in version.stdout + version.stderr,
            backend,
            "exp8918-bfs-error-pilot",
        )
        result["backend"] = dict(
            **c.reference(backend), version_stdout=version.stdout, version_stderr=version.stderr
        )
        c.progress("after_preflight")
        stage = "gpu_lease"
        gpu = resources["gpu"]
        lease = GpuLease.acquire(
            runtime_dir=os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"),
            task_id=c.TASK,
            device_uuid=gpu["uuid"],
            expected_model=str(path),
            vram_before_mb=gpu["used_mb"],
            ttl_s=2400,
        )
        result["resource_lease"] = lease.owner_receipt()
        locked = gpu_state()
        result["resource_lease"]["locked_gpu_state"] = locked
        c.require(
            "current_gpu0_free",
            True,
            locked["uuid"] == gpu["uuid"]
            and locked["index"] == 0
            and locked["used_mb"] < 1000
            and locked["free_mb"] >= 20000,
            path,
            c.TASK,
        )
        lease.transition("admitted")
        lease.transition("loading")
        stage = "model_load"
        runtime = QwenRuntime(path, raw, 0)
        ledger.start("model_load", "qwen-load", resources["model"])
        c.progress("before_model_load")
        began = time.monotonic()
        try:
            loaded = bounded(runtime.load, 240)
        except (RuntimeError, OSError, ValueError, TimeoutError) as error:
            ledger.finish("qwen-load", "failed", dict(error=str(error)))
            raise
        ledger.finish("qwen-load", "completed", loaded)
        result["phase_spans"].append(dict(phase=stage, duration_s=time.monotonic() - began))
        result["model_receipt"] = dict(loaded, preflight_gguf_sha256=resources["model"]["sha256"])
        c.progress("after_model_load")
        c.require(
            "loaded_identity",
            True,
            loaded["authenticated"]
            and loaded["gguf_sha256"] == resources["model"]["sha256"]
            and loaded["offload_layers"][0] > 0
            and loaded["offload_layers"][0] == loaded["offload_layers"][1],
            path,
            c.TASK,
        )
        c.require(
            "served_template",
            prior["model_receipt"]["props"]["chat_template"],
            loaded["props"]["chat_template"],
            path,
            "exp8918-bfs-error-pilot",
        )
        c.require(
            "served_build", build, loaded["props"]["build_info"], path, "exp8918-bfs-error-pilot"
        )
        lease.transition("resident", vram_mb=gpu_state()["used_mb"])
        lease.transition("inferencing")
        began = time.monotonic()
        generation_started = began
        for index, case in enumerate(evidence["cases"]):
            stage = "generation"
            if time.monotonic() - began >= 1680:
                raise TimeoutError("inference_budget")
            payload = dict(c.SETTINGS, messages=[dict(role="user", content=c.prompt(case))])
            atomic_json(raw / "requests" / (case["unit_id"] + ".json"), payload)
            ledger.start("generation", case["unit_id"], payload)
            c.progress("before_generation", index, len(evidence["cases"]) - index)
            tick = time.monotonic()
            try:
                response = runtime.generate(payload)
                answer, usage = (
                    response["choices"][0]["message"].get("content") or "",
                    response["usage"],
                )
                if (
                    not isinstance(answer, str)
                    or type(usage["completion_tokens"]) is not int
                    or not 0 <= usage["completion_tokens"] <= 128
                ):
                    raise ValueError("transport_token_contract")
            except (RuntimeError, OSError, ValueError, KeyError, IndexError, TimeoutError) as error:
                ledger.finish(case["unit_id"], "failed", dict(error=str(error)))
                result["outputs"].append(
                    dict(unit_id=case["unit_id"], status="failed", missing_reason=str(error))
                )
                raise
            ledger.finish(case["unit_id"], "completed", response)
            row = dict(
                unit_id=case["unit_id"],
                answer=answer,
                usage=usage,
                raw_response=response,
                request_sha256=c.canonical_hash(payload),
                duration_s=time.monotonic() - tick,
            )
            result["outputs"].append(row)
            result["phase_spans"].append(
                dict(phase=stage, unit_id=case["unit_id"], duration_s=row["duration_s"])
            )
            atomic_json(raw / "responses" / (case["unit_id"] + ".json"), row)
            lease.heartbeat()
            c.progress("after_generation", index + 1, len(evidence["cases"]) - index - 1)
        result["generation_duration_s"] = time.monotonic() - began
        result["live"] = True
        if result["generation_duration_s"] < 10:
            result["blocker"] = "generation_floor_not_met"
            result["owned_failure"] = True
            result["checks"].append(
                dict(
                    check="generation_floor_s",
                    upstream=c.TASK,
                    artifact_field="generation_duration_s",
                    expected=">=10",
                    observed=result["generation_duration_s"],
                    path=str(raw),
                    hash=None,
                )
            )
    except (
        RuntimeError,
        OSError,
        ValueError,
        KeyError,
        TimeoutError,
        LeaseError,
        c.ExternalGate,
    ) as error:
        result["blocker"] = str(error)
        result["checks"].append(
            error.row
            if isinstance(error, c.ExternalGate)
            else dict(
                check=stage,
                upstream=c.TASK,
                artifact_field=stage,
                expected="authenticated available GPU-0 Qwen",
                observed=str(error),
                path=str(path),
                hash=c.reference(path)["sha256"],
            )
        )
        c.progress("blocked_" + stage)
    finally:
        if generation_started is not None:
            result["generation_duration_s"] = time.monotonic() - generation_started
        if runtime is not None:
            c.progress("before_model_shutdown")
            try:
                result["cleanup"] = runtime.close()
            except (RuntimeError, OSError, ValueError) as error:
                result["cleanup"] = dict(leak_free=False, error=str(error))
                result["checks"].append(
                    dict(
                        check="model_cleanup",
                        upstream=c.TASK,
                        artifact_field="leak_free",
                        expected=True,
                        observed=str(error),
                        path=str(raw),
                        hash=None,
                    )
                )
            result["owned_failure"] = result["owned_failure"] or not result["cleanup"]["leak_free"]
            c.progress("after_model_shutdown")
        if lease is not None:
            try:
                if lease.document["phase"] == "inferencing":
                    lease.transition("unloading")
                    lease.transition(
                        "validating",
                        vram_mb=gpu_state()["used_mb"],
                        exit_code=0,
                        unload_observed=result["cleanup"]["leak_free"],
                    )
                lease.transition(
                    "terminal_blocked"
                    if result["blocker"] or result["owned_failure"]
                    else "terminal_complete"
                )
            except (RuntimeError, OSError, ValueError, LeaseError) as error:
                result["owned_failure"] = True
                result["checks"].append(
                    dict(
                        check="lease_terminal_transition",
                        upstream=c.TASK,
                        artifact_field="phase",
                        expected="validated terminal phase",
                        observed=str(error),
                        path=str(raw),
                        hash=None,
                    )
                )
            try:
                result["resource_lease"]["release"] = lease.release()
            except (RuntimeError, OSError, ValueError, LeaseError) as error:
                result["owned_failure"] = True
                result["checks"].append(
                    dict(
                        check="lease_release",
                        upstream=c.TASK,
                        artifact_field="release",
                        expected=True,
                        observed=str(error),
                        path=str(raw),
                        hash=None,
                    )
                )
        result["counts"], result["ledger"] = ledger.counts(), ledger.rows
        atomic_json(raw / "runtime.json", result)
    return result
