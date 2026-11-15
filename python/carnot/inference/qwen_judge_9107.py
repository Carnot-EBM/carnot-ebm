"""REQ-REPORT-9107-V776: use only the cached frozen GGUF and actual GPU0 forwards."""

from __future__ import annotations

import os
from pathlib import Path
import re
import time
from typing import Any

from carnot.gpu_lease_phase_journal import GpuLease, LeaseError
from carnot.inference.bfs_pilot_runtime_8918 import gpu_state, preflight
from carnot.inference.embedding_runtime_9098 import FrozenRuntime
from carnot.inference.gguf_metadata import build_gguf_admission_record
from carnot.reporting import qwen_judge_9107 as d
from carnot.reporting.current_work_receipt import atomic_json, build_current_work_receipt
from carnot.reporting.natural_label_runner_8975 import read_ref
from carnot.reporting.spline_basis_methods_8986 import gate

Json = dict[str, Any]


class JudgeRuntime(FrozenRuntime):
    """Template control tokens must retain their embedded tokenizer identities."""

    def tokenize(self, text: str) -> list[int]:
        """Literal continuations and the assistant prefix use the same tokenizer."""
        return list(self.model.tokenize(text.encode(), add_bos=False, special=True))


def worker(raw: Path) -> Json:
    """Retain native failures and release only our lease; never load another model."""
    import llama_cpp

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    start, owner = time.monotonic_ns(), os.getpid()
    job = read_ref(d.ref(raw / "job.json"))
    result: Json = dict(
        rows=[],
        checks=[],
        blocker=None,
        events=[],
        runtime_reference={},
        model_loads=0,
        generated_tokens=0,
    )
    lease: Any = None
    model: Any = None
    stage = "cached_runtime"
    try:
        d.progress("preflight_before")
        resources = preflight(raw)
        spec, gpu = resources["model"], resources["gpu"]
        admission = build_gguf_admission_record(
            spec["model_path"],
            repository_id=d.MODEL_SPECS[0]["hf_id"],
            trusted_sha256=spec["sha256"],
            expected_architectures={"qwen35"},
        )
        if not admission["admitted"] or admission["content_metadata"]["quantization"] != "Q4_K_M":
            raise ValueError("gguf_admission")
        libraries = sorted(Path(llama_cpp.__file__).parent.rglob("*.so"))
        if not libraries:
            raise ValueError("native_build_missing")
        result["runtime_reference"] = dict(
            model=spec,
            admission=admission,
            native_libraries=[d.ref(p) for p in libraries],
            python_binding=d.ref(Path(llama_cpp.__file__)),
            system_info=llama_cpp.llama_print_system_info().decode(),
        )
        d.progress("preflight_after")
        stage = "gpu0_lease_and_memory"
        lease = GpuLease.acquire(
            runtime_dir=os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"),
            task_id=d.TASK,
            device_uuid=gpu["uuid"],
            expected_model=spec["model_path"],
            vram_before_mb=gpu["used_mb"],
            ttl_s=3400,
        )
        result["gpu_lease"] = lease.owner_receipt()
        locked = gpu_state()
        result["memory_before"] = locked
        if locked["index"] != 0 or locked["uuid"] != gpu["uuid"] or locked["free_mb"] < 20000:
            raise ValueError("gpu0_memory_or_identity")
        lease.transition("admitted")
        lease.transition("loading")
        stage = "model_load"
        event = dict(
            call_id="qwen-load",
            operation="model_load",
            scope="current",
            transport="owned_runtime",
            run_id=raw.name,
            owner_pid=owner,
        )
        result["events"].append(dict(event, state="attempted", monotonic_ns=time.monotonic_ns()))
        atomic_json(raw / "events.json", dict(events=result["events"]))
        result["model_loads"] = 1
        d.progress("model_load_before")
        try:
            model = llama_cpp.Llama(
                model_path=spec["model_path"],
                n_ctx=2048,
                n_batch=1536,
                n_ubatch=1536,
                n_gpu_layers=-1,
                main_gpu=0,
                split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,
                tensor_split=[1.0],
                logits_all=True,
                embedding=False,
                seed=7769107,
                verbose=True,
            )
        except (OSError, ValueError, RuntimeError):
            result["events"].append(dict(event, state="failed", monotonic_ns=time.monotonic_ns()))
            raise
        result["events"].append(dict(event, state="completed", monotonic_ns=time.monotonic_ns()))
        d.progress("model_load_after")
        stage = "actual_cuda_offload"
        resident = gpu_state()
        log = Path(os.readlink("/proc/self/fd/2")).read_text(errors="replace")
        offload = re.findall(r"offloaded (\d+)/(\d+) layers", log)
        result.update(
            memory_resident=resident,
            offload_layers=offload,
            embedded_chat_template=model.metadata.get("tokenizer.chat_template"),
        )
        if (
            not offload
            or int(offload[-1][0]) <= 0
            or offload[-1][0] != offload[-1][1]
            or resident["used_mb"] - locked["used_mb"] < 1000
        ):
            raise ValueError("missing_actual_cuda_offload")
        if not result["embedded_chat_template"]:
            raise ValueError("missing_embedded_chat_template")
        lease.transition("resident", vram_mb=resident["used_mb"])
        lease.transition("inferencing")
        runtime = JudgeRuntime(model, raw)
        stage = "raw_likelihoods"
        for i, (row, view) in enumerate(zip(job["rows"], job["views"], strict=True)):
            if time.monotonic_ns() - start >= 3200e9:
                raise TimeoutError("measurement_deadline")
            d.progress("request_before", i, len(job["rows"]) - i)
            result["rows"].append(d.score(runtime, row, view, raw))
            atomic_json(raw / "scores.json", dict(rows=result["rows"]))
            lease.heartbeat()
            d.progress("request_after", i + 1, len(job["rows"]) - i - 1)
    except (OSError, ValueError, RuntimeError, TimeoutError, KeyError, LeaseError) as error:
        result["blocker"] = stage
        result["checks"].append(
            gate(raw, stage, "authenticated frozen GPU0 scoring", str(error), stage, d.TASK)
        )
        d.progress("blocked_" + stage)
    finally:
        d.progress("model_cleanup_before")
        if model is not None:
            model.close()
        if lease is not None:
            if lease.document["phase"] == "inferencing":
                lease.transition("unloading")
                lease.transition(
                    "validating", vram_mb=gpu_state()["used_mb"], exit_code=0, unload_observed=True
                )
            lease.transition("terminal_blocked" if result["blocker"] else "terminal_complete")
            result["lease_release"] = lease.release()
            lease.close()
        result["receipt"] = build_current_work_receipt(
            run_id=raw.name,
            owner_pid=owner,
            events=result["events"],
            inference_substrate="frozen_gguf_forward_scoring",
            inference_substrate_details=dict(generated_tokens=0, requests=len(result["rows"])),
            inference_substrate_class="model_load_no_generation"
            if result["model_loads"]
            else "blocked_no_run",
            execution_venue="host",
            started_monotonic_ns=start,
            ended_monotonic_ns=time.monotonic_ns(),
        )
        atomic_json(raw / "runtime.json", result)
        d.progress("model_cleanup_after")
    return result
