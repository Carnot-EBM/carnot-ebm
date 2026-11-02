"""REQ-REPORT-8023: reuse the qualified loader with a frozen fit/tune scorer.

Adapters change the scheduled work and its limits, while the existing loader
continues to own CUDA admission, model identity and cleanup evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from carnot.gpu_lease_phase_journal import GpuLease
from carnot.inference import fixed_answer_likelihood_8022 as s
from carnot.inference import likelihood_runtime_8022 as runtime
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import likelihood_calibration_8023 as c

Json = dict[str, Any]


def verify_panel(panel: Json, tokenizer: Any) -> Json:
    """Re-tokenize original bytes before the weight load to reject stale views."""
    seen: set[str] = set()
    counts = dict(fit=0, tune=0)
    for item in panel["rows"]:
        current = s.prepare(
            {k: item[k] for k in ("family_id", "source_bytes", "answer_bytes")}, tokenizer
        )
        if any(current[k] != item[k] for k in current) or item["source_normalized_hash"] in seen:
            raise ValueError("frozen_panel_drift_or_overlap")
        seen.add(item["source_normalized_hash"])
        counts[item["role"]] += 1
    if counts != dict(fit=64, tune=32):
        raise ValueError("fit_tune_roster")
    return panel


def worker(plan_path: Path, output: Path) -> None:
    """One adapter invocation enforces the current model hash and bounded work."""
    started = time.monotonic()
    plan = json.loads(plan_path.read_text())
    cached = runtime.cached_current_model()
    if cached is None:
        atomic_json(
            output,
            dict(
                checks=[
                    dict(
                        upstream_id="cached_qwen",
                        path=str(plan_path),
                        hash=None,
                        artifact_field="model_path",
                        expected="cached mandated GGUF",
                        observed=None,
                        passed=False,
                    )
                ]
            ),
        )
        return
    path = Path(cached["model_path"])
    original_bounded = runtime.bounded
    s.progress("8023_before_model_hash", started)
    digest = original_bounded(lambda: sha256_file(path), 120)
    s.progress("8023_after_model_hash", started)
    if digest != plan["gguf_sha256"]:
        atomic_json(
            output,
            dict(
                checks=[
                    dict(
                        upstream_id="exp8022",
                        path=str(path),
                        hash=digest,
                        artifact_field="gguf_sha256",
                        expected=plan["gguf_sha256"],
                        observed=digest,
                        passed=False,
                    )
                ]
            ),
        )
        return
    original_acquire = GpuLease.acquire
    original_progress = s.progress
    completed = 0

    class Lease:
        @staticmethod
        def acquire(**kwargs: Any) -> Any:
            return original_acquire(**dict(kwargs, ttl_s=2520))

    def bound(fn: Any, timeout: float) -> Any:
        remaining = 2400 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("total_model_work_cap")
        return original_bounded(fn, min(timeout, 120, remaining))

    def scoring(model: Any, checkpoints: Path) -> Json:
        nonlocal completed
        rows = c.capture(plan["panel"], model, checkpoints, deadline=started + 2400)
        completed = sum(r["status"] == "completed" for r in rows)
        reduced = c.reduce(plan["panel"], rows)
        return dict(reduced, rows=rows, generated_tokens=0)

    def progress(phase: str, began: float, units: int = 0, pending: int = 0) -> None:
        if phase in {"before_benchmark", "after_benchmark"}:
            units, pending = completed, 384 - completed
        original_progress(phase, began, units, pending)

    with (
        patch.object(runtime, "TASK", "exp8023-likelihood-calibration"),
        patch.object(runtime, "GpuLease", Lease),
        patch.object(runtime, "bounded", bound),
        patch.object(runtime, "sha256_file", lambda _: digest),
        patch.object(s, "freeze_panel", lambda _, tok: verify_panel(plan["panel"], tok)),
        patch.object(s, "qualify", scoring),
        patch.object(s, "progress", progress),
    ):
        runtime.worker(plan_path, output)
