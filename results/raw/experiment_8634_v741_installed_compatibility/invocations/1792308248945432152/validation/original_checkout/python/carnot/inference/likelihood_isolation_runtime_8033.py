"""REQ-REPORT-8033: reuse CUDA admission and cleanup with changed execution.

Only this owned scoring child patches the earlier runtime. Global inference and
live ARC defaults keep their existing settings.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from carnot.gpu_lease_phase_journal import GpuLease
from carnot.inference import fixed_answer_likelihood_8022 as base
from carnot.inference import likelihood_runtime_8022 as runtime
from carnot.inference.scoring_isolation_8033 import NativeController, capture, reduce, METHODS
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

Json = dict[str, Any]
original_acquire = GpuLease.acquire


def verify_panel(panel: list[Json], tokenizer: Any) -> list[Json]:
    """Original bytes must reproduce the frozen full-answer token view."""
    for item in panel:
        current = base.prepare(
            {k: item[k] for k in ("family_id", "source_bytes", "answer_bytes")}, tokenizer
        )
        if any(current[k] != item[k] for k in ("views", "target_tokens", "response_token_offsets")):
            raise ValueError("frozen_panel_token_drift")
    return panel


def worker(plan_path: Path, output: Path) -> None:
    """Keep failed cached-model gates durable and bind each pass to producer bytes."""
    import llama_cpp
    from llama_cpp import _internals

    started = time.monotonic()
    plan = json.loads(plan_path.read_text())
    cached = runtime.cached_current_model()
    observed: Any = "missing_mandated_cache"
    if cached is not None:
        base.progress("8033_before_model_hash", started)
        observed = runtime.bounded(lambda: sha256_file(Path(cached["model_path"])), 120)
        base.progress("8033_after_model_hash", started)
    if observed != plan["gguf_sha256"]:
        atomic_json(
            output,
            dict(
                checks=[
                    dict(
                        check_name="cached_model_identity",
                        upstream_id="exp8022",
                        path=str(plan_path),
                        sha256=sha256_file(plan_path),
                        artifact_field="gguf_sha256",
                        expected=plan["gguf_sha256"],
                        observed=observed,
                        passed=False,
                    )
                ]
            ),
        )
        return
    atomic_json(
        output.parent / "reset_semantics.json",
        dict(
            runtime_version=llama_cpp.__version__,
            python_binding=inspect.getsource(llama_cpp.Llama.reset),
            eval_binding=inspect.getsource(llama_cpp.Llama.eval),
            kv_removal_binding=inspect.getsource(_internals.LlamaContext.kv_cache_seq_rm),
            kv_clear_binding=inspect.getsource(_internals.LlamaContext.kv_cache_clear),
            native_library=runtime.sha256_file(Path(llama_cpp.llama_cpp._lib._name)),
            bindings=runtime.sha256_file(Path(inspect.getfile(llama_cpp.Llama))),
        ),
    )
    bound_original = runtime.bounded
    deadline = started + METHODS["model_work_seconds"]
    original_progress = base.progress
    completed = 0

    def progress(phase: str, began: float, units: int = 0, pending: int = 0) -> None:
        if phase in {"before_benchmark", "after_benchmark"}:
            units, pending = completed, 132 - completed
        original_progress(phase, began, units, pending)

    def bound(call: Any, timeout: float) -> Any:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("total_model_work_cap")
        return bound_original(call, min(timeout, 120, remaining))

    def scoring(model: Any, checkpoints: Path) -> Json:
        nonlocal completed
        gpu = int(model.model.model_params.main_gpu)
        reserve = runtime.run_commands(
            runtime.ROOT,
            [
                runtime.CommandSpec(
                    "memory_reserve",
                    (
                        "nvidia-smi",
                        f"--id={gpu}",
                        "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits",
                    ),
                    "runtime",
                    10,
                )
            ],
            log_dir=output.parent / "memory_reserve",
            heartbeat_s=5,
        )[0]
        atomic_json(output.parent / "memory_reserve.json", reserve)
        if (
            not reserve["passed"]
            or int(reserve["output_tail"].strip()) < METHODS["memory_reserve_mb"]
        ):
            raise ValueError("model_memory_reserve_unavailable")
        rows = capture(
            plan["panel"], NativeController(model, deadline), checkpoints, deadline=deadline
        )
        completed = sum(r["status"] == "completed" for r in rows)
        return dict(
            reduce(plan["panel"], rows),
            rows=rows,
            passed=len(rows) == 132
            and reduce(plan["panel"], rows)["selected_condition"] is not None,
            generated_tokens=0,
            scoring_config_hash=plan["scoring_config_hash"],
        )

    class Lease:
        @staticmethod
        def acquire(**kwargs: Any) -> Any:
            return original_acquire(**dict(kwargs, ttl_s=1100))

    with (
        patch.object(runtime, "TASK", "exp8033-scoring-isolation"),
        patch.object(runtime, "GpuLease", Lease),
        patch.object(runtime, "bounded", bound),
        patch.object(runtime, "sha256_file", lambda _: observed),
        patch.object(base, "freeze_panel", lambda _, tok: verify_panel(plan["panel"], tok)),
        patch.object(base, "qualify", scoring),
        patch.object(base, "progress", progress),
    ):
        runtime.worker(plan_path, output)
    result = json.loads(output.read_text())
    result["scoring_config_hash"] = plan["scoring_config_hash"]
    result["kernel_configuration"] = dict(
        n_batch=256,
        n_ubatch=256,
        flash_attn=False,
        n_ctx=6384,
        n_gpu_layers=-1,
        weights_reloaded=False,
        shape_policy=METHODS["chunk_tokens"],
        padding=False,
    )
    result["reset_semantics_reference"] = dict(
        path=str(output.parent / "reset_semantics.json"),
        sha256=sha256_file(output.parent / "reset_semantics.json"),
    )
    result["producer_identity_hash"] = canonical_hash(
        dict(methods=METHODS, config=plan["scoring_config_hash"])
    )
    atomic_json(output, result)
