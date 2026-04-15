#!/usr/bin/env python3
"""Experiment 328 — Live GPU Benchmark Wrapper.

**Researcher summary:**
    Experiment 316 executed the full-scale benchmark script (Exp 315) in
    ``inference_mode="simulated"`` because live GPU was unavailable at run time.
    The simulation produced implausible results (Qwen3.5-0.8B at 34%, Gemma4-E4B-it
    at 30%) compared to published model-card baselines (25% and 80% respectively).

    This script re-runs the Exp 315 benchmark with ``CARNOT_FORCE_LIVE=1`` to obtain
    authoritative live GPU inference numbers.  It:
    1. Runs DualGPUMonitor pre-check and documents the GPU state.
    2. Invokes ``experiment_315_fullscale_benchmark.py`` without ``--simulated``.
    3. Validates that the result has ``inference_mode="live_gpu"``.
    4. Computes simulation divergence (vs Exp 316) and baseline deviation.
    5. Emits a wrapper artifact with schema ``carnot.live_fullscale_benchmark.v1``.

    If the GPU pre-warm fails (OOM, driver error, insufficient VRAM), this script
    emits an honest ``status="blocked"`` artifact with ``gpu_diagnostics`` rather
    than silently falling back to simulation.

**Why a wrapper artifact instead of modifying Exp 315:**
    Exp 315/316 are part of the research record.  The wrapper pattern preserves the
    original script and execution history while adding the live-GPU layer on top.

Spec: REQ-BENCH-002, SCENARIO-BENCH-003, SCENARIO-BENCH-004
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT: int = 328
TITLE: str = "Live GPU Full-Scale Benchmark (Exp 315 re-run)"
SCHEMA: str = "carnot.live_fullscale_benchmark.v1"

WRAPPER_OUTPUT: str = "results/experiment_328_live_fullscale_results.json"
LIVE_OUTPUT: str = "results/experiment_316_fullscale_results_live.json"
SIMULATED_INPUT: str = "results/experiment_316_fullscale_results.json"

# Published baselines — model-card accuracy for GSM8K (main split, 5-shot or 0-shot).
# Qwen3.5-0.8B: ~25% (0-shot CoT on GSM8K main).
# Gemma4-E4B-it: ~80% (instruction-tuned, 0-shot CoT on GSM8K).
PUBLISHED_BASELINES: dict[str, float] = {
    "Qwen3.5-0.8B": 0.25,
    "Gemma4-E4B-it": 0.80,
}

TOLERANCE: float = 0.15  # Acceptable absolute deviation from published baseline.


# ---------------------------------------------------------------------------
# Helper functions (tested independently in test_experiment_328_live_benchmark.py)
# ---------------------------------------------------------------------------


def load_live_benchmark_results(path: str | Path) -> dict[str, Any]:
    """Load a live GPU benchmark result JSON and validate top-level schema.

    **Why this function exists:**
        Downstream callers need a single validated entry point so they detect
        schema drift early rather than propagating stale data.

    Args:
        path: Path to the JSON artifact.

    Returns:
        Parsed and minimally validated artifact dict.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ValueError: if any required top-level key is absent.

    Spec: REQ-BENCH-002
    """
    from tests.python.test_experiment_328_live_benchmark import (  # type: ignore[import]
        REQUIRED_WRAPPER_KEYS,
        load_live_benchmark_results as _load,
    )

    return _load(path)


def validate_live_result(result: dict[str, Any]) -> None:
    """Raise ValueError if inference_mode != 'live_gpu'.

    Spec: REQ-BENCH-002, SCENARIO-BENCH-003
    """
    mode = result.get("inference_mode", "<missing>")
    if mode != "live_gpu":
        raise ValueError(
            f"Result inference_mode={mode!r} does not qualify as a live GPU result. "
            "Only inference_mode='live_gpu' results may be promoted to headline claims. "
            "Re-run with CARNOT_FORCE_LIVE=1 when GPUs are available."
        )


def compare_to_simulated(
    live: dict[str, Any],
    simulated: dict[str, Any],
) -> dict[str, Any]:
    """Compute per-model, per-mode accuracy delta (live - simulated).

    Spec: REQ-BENCH-002, SCENARIO-BENCH-004
    """
    divergence: dict[str, Any] = {}
    live_pmr = live.get("per_model_results", {})
    sim_pmr = simulated.get("per_model_results", {})

    for model_name, live_modes in live_pmr.items():
        sim_modes = sim_pmr.get(model_name, {})
        divergence[model_name] = {}
        for mode, live_variants in live_modes.items():
            sim_variants = sim_modes.get(mode, {})
            divergence[model_name][mode] = {}
            for variant, live_cell in live_variants.items():
                sim_cell = sim_variants.get(variant)
                if sim_cell is None:
                    continue
                live_acc = live_cell.get("accuracy", 0.0)
                sim_acc = sim_cell.get("accuracy", 0.0)
                divergence[model_name][mode][variant] = {
                    "live": live_acc,
                    "simulated": sim_acc,
                    "delta": round(live_acc - sim_acc, 6),
                }
    return divergence


def compare_to_published_baseline(
    result: dict[str, Any],
    baselines: dict[str, float],
) -> dict[str, Any]:
    """Compute per-model deviation of live accuracy from published model-card baseline.

    Spec: REQ-BENCH-002, SCENARIO-BENCH-004
    """
    deviations: dict[str, Any] = {}
    pmr = result.get("per_model_results", {})

    for model_name, modes in pmr.items():
        published = baselines.get(model_name)
        if published is None:
            for base_key, base_val in baselines.items():
                if base_key in model_name or model_name in base_key:
                    published = base_val
                    break
        if published is None:
            continue

        baseline_mode = modes.get("baseline", {})
        all_cell = baseline_mode.get("all")
        if all_cell is None:
            continue

        live_acc = all_cell.get("accuracy", 0.0)
        deviation = round(live_acc - published, 6)
        within = abs(live_acc - published) <= TOLERANCE
        deviations[model_name] = {
            "baseline_accuracy": live_acc,
            "published_baseline": published,
            "deviation": deviation,
            "within_tolerance": within,
        }
    return deviations


def _utc_now() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_date() -> str:
    """Return today's date as 8-digit string."""
    import datetime

    return datetime.date.today().strftime("%Y%m%d")


def _get_gpu_diagnostics() -> dict[str, Any]:
    """Collect GPU diagnostics via nvidia-smi and DualGPUMonitor.

    **Why this function exists:**
        Blocked artifacts must document exactly WHY the GPU was unavailable so
        future operators can diagnose whether the issue was VRAM pressure, driver
        error, zombie processes, or something else.  This function collects all
        available signals without raising.
    """
    diagnostics: dict[str, Any] = {}

    # DualGPUMonitor health check
    try:
        sys.path.insert(0, str(_REPO_ROOT / "python"))
        from carnot.pipeline.dual_gpu_monitor import DualGPUMonitor  # noqa: PLC0415

        monitor = DualGPUMonitor()
        diagnostics["dual_gpu_monitor"] = monitor.to_dict()
    except Exception as exc:
        diagnostics["dual_gpu_monitor"] = {"error": str(exc)}

    # Raw nvidia-smi summary
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        diagnostics["nvidia_smi_raw"] = result.stdout.strip()
        diagnostics["nvidia_smi_returncode"] = result.returncode
    except Exception as exc:
        diagnostics["nvidia_smi_raw"] = None
        diagnostics["nvidia_smi_error"] = str(exc)

    # Current processes (using gpu_uuid since gpu_index is not supported on this driver)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        diagnostics["compute_apps"] = result.stdout.strip()
    except Exception as exc:
        diagnostics["compute_apps_error"] = str(exc)

    return diagnostics


def _build_blocked_artifact(
    t0: float,
    reason: str,
    gpu_diagnostics: dict[str, Any],
    stderr_excerpt: str = "",
) -> dict[str, Any]:
    """Build an honest blocked artifact for emission when GPU is unavailable.

    **Why 'blocked' and not 'failure':**
        A blocked artifact means the infrastructure condition (GPU VRAM, driver,
        pre-warm timeout) prevented execution.  This is distinct from a code error.
        Blocked artifacts are expected in the research record when GPU resources
        are contested; they document the constraint without fabricating results.
    """
    duration = round(time.perf_counter() - t0, 3)
    now = _utc_now()
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "inference_mode": "blocked",
        "status": "blocked",
        "run_date": _run_date(),
        "started_at": now,
        "finished_at": now,
        "duration_s": duration,
        "primary_result_path": LIVE_OUTPUT,
        "blocked_reason": reason,
        "stderr_excerpt": stderr_excerpt[:2000],
        "gpu_diagnostics": gpu_diagnostics,
        "simulation_divergence": {},
        "baseline_deviation": {},
    }


def _load_simulated_result() -> dict[str, Any] | None:
    """Load the Exp 316 simulated result for comparison, or None if absent."""
    sim_path = _REPO_ROOT / SIMULATED_INPUT
    if not sim_path.exists():
        return None
    with sim_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def run_experiment_328() -> dict[str, Any]:
    """Execute the full Exp 328 lifecycle and return the wrapper artifact.

    Steps:
    1. GPU pre-check (DualGPUMonitor).
    2. Run experiment_315_fullscale_benchmark.py with CARNOT_FORCE_LIVE=1.
    3. Validate inference_mode="live_gpu".
    4. Compute simulation_divergence and baseline_deviation.
    5. Build and return the wrapper artifact.

    Returns:
        The wrapper artifact dict (status="success" or "blocked").
    """
    t0 = time.perf_counter()
    started_at = _utc_now()

    print(f"[Exp 328] {TITLE}")
    print(f"[Exp 328] Started: {started_at}")

    # ------------------------------------------------------------------ #
    # 1. GPU pre-check                                                    #
    # ------------------------------------------------------------------ #
    print("[Exp 328] Step 1: GPU pre-check via DualGPUMonitor ...")
    gpu_diagnostics = _get_gpu_diagnostics()
    monitor_result = gpu_diagnostics.get("dual_gpu_monitor", {}).get("health", {})
    print(f"[Exp 328] DualGPUMonitor: {monitor_result}")

    # Document the DualGPUMonitor bug: nvidia-smi 'gpu_index' field is not
    # supported on this driver, so the monitor falsely reports idle_gpus=[0,1].
    # We log this but do not abort — the actual VRAM state (from raw nvidia-smi)
    # shows sufficient free memory on both GPUs.
    if not monitor_result.get("all_healthy", False):
        raw = gpu_diagnostics.get("nvidia_smi_raw", "")
        print("[Exp 328] WARNING: DualGPUMonitor reports all_healthy=False.")
        print("[Exp 328] NOTE: This may be a driver compatibility issue with 'gpu_index' field.")
        print(f"[Exp 328] Raw GPU state:\n{raw}")

    # ------------------------------------------------------------------ #
    # 2. Obtain benchmark result (reuse existing or run fresh)            #
    # ------------------------------------------------------------------ #
    live_output_path = _REPO_ROOT / LIVE_OUTPUT
    live_output_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = ""
    stderr = ""

    # Reuse an existing live_gpu result rather than re-running inference.
    # Re-running would OOM if the previous run's models still hold VRAM.
    need_run = True
    if live_output_path.exists():
        print(f"[Exp 328] Step 2: Checking existing result at {live_output_path} ...")
        try:
            with live_output_path.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
            if existing.get("inference_mode") == "live_gpu":
                print("[Exp 328] REUSE: Valid live_gpu result exists — skipping re-run.")
                qwen_acc = existing.get("per_model_results", {}).get("Qwen3.5-0.8B", {}).get("baseline", {}).get("all", {}).get("accuracy", "N/A")
                gem_acc = existing.get("per_model_results", {}).get("Gemma4-E4B-it", {}).get("baseline", {}).get("all", {}).get("accuracy", "N/A")
                print(f"[Exp 328]   Qwen3.5-0.8B GSM8K baseline={qwen_acc}")
                print(f"[Exp 328]   Gemma4-E4B-it GSM8K baseline={gem_acc}")
                need_run = False
            else:
                print(f"[Exp 328] Existing inference_mode={existing.get('inference_mode')!r} — re-running.")
        except Exception as exc:
            print(f"[Exp 328] Could not read existing result ({exc}) — re-running.")
    else:
        print(f"[Exp 328] No existing result — will run benchmark.")

    if need_run:
        benchmark_script = _REPO_ROOT / "scripts" / "experiment_315_fullscale_benchmark.py"
        env = {**os.environ, "CARNOT_FORCE_LIVE": "1", "JAX_PLATFORMS": "cpu"}
        cmd = [
            sys.executable,
            str(benchmark_script),
            "--output_path",
            str(live_output_path),
        ]
        print(f"[Exp 328] Step 2: Running benchmark ...")
        print(f"[Exp 328] Command: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                cwd=str(_REPO_ROOT),
                timeout=7200,
            )
            bench_duration = round(time.perf_counter() - t0, 1)
            stdout = proc.stdout
            stderr = proc.stderr
            print(f"[Exp 328] Benchmark returned in {bench_duration}s (rc={proc.returncode})")
            for line in (stdout or "").strip().splitlines()[-30:]:
                print(f"  {line}")
            if proc.returncode != 0:
                reason = f"Benchmark exited rc={proc.returncode}"
                artifact = _build_blocked_artifact(t0, reason, gpu_diagnostics, (stderr or "")[-2000:])
                artifact["stdout_excerpt"] = (stdout or "")[-2000:]
                return artifact
        except subprocess.TimeoutExpired:
            reason = "Benchmark timed out after 7200s"
            return _build_blocked_artifact(t0, reason, gpu_diagnostics)

    # ------------------------------------------------------------------ #
    # 3. Validate result has inference_mode="live_gpu"                    #
    # ------------------------------------------------------------------ #
    print("[Exp 328] Step 3: Validating live GPU result ...")
    if not live_output_path.exists():
        reason = f"Benchmark completed but output file not found: {live_output_path}"
        print(f"[Exp 328] BLOCKED: {reason}")
        return _build_blocked_artifact(t0, reason, gpu_diagnostics)

    with live_output_path.open("r", encoding="utf-8") as fh:
        live_result = json.load(fh)

    inference_mode = live_result.get("inference_mode", "unknown")
    print(f"[Exp 328] inference_mode={inference_mode!r}")

    if inference_mode != "live_gpu":
        reason = (
            f"Benchmark fell back to inference_mode={inference_mode!r}. "
            "GPU pre-warm likely failed. See stdout/stderr above for root cause."
        )
        print(f"[Exp 328] BLOCKED: {reason}")
        artifact = _build_blocked_artifact(t0, reason, gpu_diagnostics, stderr[-2000:] if stderr else "")
        artifact["fallback_result_path"] = LIVE_OUTPUT
        artifact["fallback_inference_mode"] = inference_mode
        # Preserve stdout so the operator can see what went wrong
        artifact["stdout_excerpt"] = stdout[-3000:] if stdout else ""
        return artifact

    print("[Exp 328] VALIDATED: inference_mode='live_gpu' confirmed.")

    # ------------------------------------------------------------------ #
    # 4. Compute simulation_divergence and baseline_deviation             #
    # ------------------------------------------------------------------ #
    print("[Exp 328] Step 4: Computing divergence metrics ...")
    simulated_result = _load_simulated_result()
    if simulated_result is not None:
        simulation_divergence = compare_to_simulated(live_result, simulated_result)
        print("[Exp 328] Simulation divergence (live - simulated):")
        for model, modes in simulation_divergence.items():
            for mode, variants in modes.items():
                for variant, cell in variants.items():
                    if variant == "all":
                        print(
                            f"  {model}/{mode}/all: live={cell['live']:.3f} "
                            f"sim={cell['simulated']:.3f} delta={cell['delta']:+.3f}"
                        )
    else:
        simulation_divergence = {}
        print("[Exp 328] WARNING: Exp 316 simulated result not found; skipping divergence.")

    baseline_deviation = compare_to_published_baseline(live_result, PUBLISHED_BASELINES)
    print("[Exp 328] Baseline deviation (live - published):")
    for model, info in baseline_deviation.items():
        within = info["within_tolerance"]
        marker = "OK" if within else "OUT-OF-RANGE"
        print(
            f"  {model}: live={info['baseline_accuracy']:.3f} "
            f"published={info['published_baseline']:.3f} "
            f"deviation={info['deviation']:+.3f} [{marker}]"
        )

    # ------------------------------------------------------------------ #
    # 5. Build wrapper artifact                                           #
    # ------------------------------------------------------------------ #
    print("[Exp 328] Step 5: Building wrapper artifact ...")
    finished_at = _utc_now()
    duration = round(time.perf_counter() - t0, 3)

    wrapper: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "inference_mode": "live_gpu",
        "status": "success",
        "run_date": _run_date(),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration,
        "primary_result_path": LIVE_OUTPUT,
        "gpu_diagnostics": gpu_diagnostics,
        "simulation_divergence": simulation_divergence,
        "baseline_deviation": baseline_deviation,
        "benchmark_script": "scripts/experiment_315_fullscale_benchmark.py",
        "benchmark_experiment_id": live_result.get("experiment", 315),
        "benchmark_n_gsm8k": live_result.get("n_gsm8k", 0),
        "benchmark_n_humaneval": live_result.get("n_humaneval", 0),
        "benchmark_modes": live_result.get("modes_run", []),
        "benchmark_duration_s": live_result.get("duration_s", 0.0),
    }

    return wrapper


def main() -> None:
    """CLI entry point for Experiment 328."""
    t0 = time.perf_counter()
    wrapper = run_experiment_328()

    output_path = _REPO_ROOT / WRAPPER_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(wrapper, indent=2))

    print(f"\n[Exp 328] Wrapper artifact written to: {output_path}")
    print(f"[Exp 328] status={wrapper['status']!r}, inference_mode={wrapper['inference_mode']!r}")
    if wrapper["status"] == "blocked":
        print(f"[Exp 328] BLOCKED REASON: {wrapper.get('blocked_reason', 'unknown')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
