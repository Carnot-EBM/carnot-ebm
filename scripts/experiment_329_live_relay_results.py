#!/usr/bin/env python3
"""Experiment 329 — Live GPU Four-Tier Relay Benchmark Wrapper.

**Researcher summary:**
    Experiment 318 ran the four-tier continuous self-learning relay benchmark in
    ``inference_mode="simulated"`` because live GPU was unavailable at run time.
    The simulation produced improvement_1to3 = -0.0606 (honest regression under
    synthetic JEPA energies and Z3 decisions from a fixed random seed, not from
    real model outputs).

    This script re-runs the Exp 318 four-tier relay with ``CARNOT_FORCE_LIVE=1``
    to determine whether the relay stack produces improvement_1to3 > 0 on real
    model outputs (Qwen3.5-0.8B on GPU 0, Gemma4-E4B-it on GPU 1).  It:
    1. Runs DualGPUMonitor pre-check and documents the GPU state.
    2. Invokes ``experiment_318_self_learning_relay.py`` with CARNOT_FORCE_LIVE=1.
    3. Validates that the result has ``inference_mode="live_gpu"``.
    4. Computes simulation_comparison against Exp 318 simulated baseline.
    5. Emits a wrapper artifact with schema ``carnot.live_relay_benchmark.v1``.

    If the GPU pre-warm fails (OOM, driver error, insufficient VRAM), this script
    emits an honest ``status="blocked"`` artifact with ``gpu_diagnostics`` rather
    than silently falling back to simulation.

**Why a wrapper artifact instead of modifying Exp 318:**
    Exp 318 is part of the research record.  The wrapper pattern preserves the
    original script and execution history while adding the live-GPU layer on top.
    This mirrors the Exp 328/316 pattern used for the full-scale benchmark.

**Honest reporting policy:**
    improvement_1to3 is signed: negative values are valid and must appear in the
    artifact unchanged.  NEVER clamp, abs(), or hide regressions.
    (REQ-LEARN-014-3, SCENARIO-LEARN-022)

Spec: REQ-LEARN-014, SCENARIO-LEARN-023, SCENARIO-LEARN-024
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

EXPERIMENT: int = 329
TITLE: str = "Live GPU Four-Tier Relay Benchmark (Exp 318 re-run)"
SCHEMA: str = "carnot.live_relay_benchmark.v1"

WRAPPER_OUTPUT: str = "results/experiment_329_live_relay_results.json"
LIVE_RELAY_OUTPUT: str = "results/experiment_318_live_relay.json"
SIMULATED_INPUT: str = "results/experiment_318_self_learning_relay.json"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """Return current UTC time in ISO-8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run_date() -> str:
    """Return today's date as an 8-digit string (e.g. '20260415')."""
    return time.strftime("%Y%m%d", time.gmtime())


def _write_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON artifact to *path* with parent directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# GPU diagnostics
# ---------------------------------------------------------------------------


def _get_gpu_diagnostics() -> dict[str, Any]:
    """Collect GPU diagnostics via nvidia-smi and DualGPUMonitor.

    **Why this function exists:**
        Blocked artifacts must document exactly WHY the GPU was unavailable
        so future operators can diagnose VRAM pressure, driver errors, or zombie
        processes.  This function collects all available signals without raising.

    Returns:
        Dict with keys: dual_gpu_monitor, nvidia_smi_raw, nvidia_smi_returncode,
        compute_apps.  Each field degrades gracefully to an error string on failure.
    """
    diagnostics: dict[str, Any] = {}

    # DualGPUMonitor health check (from Exp 326)
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
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        diagnostics["nvidia_smi_raw"] = result.stdout.strip()
        diagnostics["nvidia_smi_returncode"] = result.returncode
    except Exception as exc:
        diagnostics["nvidia_smi_raw"] = None
        diagnostics["nvidia_smi_error"] = str(exc)

    # Active compute processes
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


# ---------------------------------------------------------------------------
# Comparison helpers (importable from test module too)
# ---------------------------------------------------------------------------


def compare_relay_to_simulated(
    live: dict[str, Any],
    simulated: dict[str, Any],
) -> dict[str, Any]:
    """Compute per-batch accuracy deltas between live and simulated relay results.

    **Why these deltas matter:**
        The Exp 318 simulated baseline (improvement_1to3 = -0.0606) used
        synthetic JEPA energies and Z3 decisions from a fixed random seed, not
        from real model outputs.  This function quantifies how far the live
        inference diverges from the simulation on each batch so we can detect
        and document simulation drift in the research record.

        Delta = live_value - simulated_value.  Negative deltas are preserved
        without clamping (REQ-LEARN-014-3, SCENARIO-LEARN-024).

    Args:
        live:      Live wrapper artifact dict (Exp 329).
        simulated: Simulated baseline artifact dict (Exp 318).

    Returns:
        Dict with signed float deltas:
          - batch1_accuracy_delta
          - batch3_accuracy_delta
          - improvement_delta

    Spec: REQ-LEARN-014, SCENARIO-LEARN-024
    """
    # Live values may be nested under "live_result" in the wrapper
    live_payload = live.get("live_result", live)

    live_b1 = float(live_payload.get("batch1_accuracy", live.get("batch1_accuracy", 0.0)))
    live_b3 = float(live_payload.get("batch3_accuracy", live.get("batch3_accuracy", 0.0)))
    live_imp = float(live_payload.get("improvement_1to3", live.get("improvement_1to3", 0.0)))

    sim_b1 = float(simulated.get("batch1_accuracy", 0.0))
    sim_b3 = float(simulated.get("batch3_accuracy", 0.0))
    sim_imp = float(simulated.get("improvement_1to3", 0.0))

    return {
        "batch1_accuracy_delta": round(live_b1 - sim_b1, 6),
        "batch3_accuracy_delta": round(live_b3 - sim_b3, 6),
        "improvement_delta": round(live_imp - sim_imp, 6),
    }


# ---------------------------------------------------------------------------
# Blocked artifact builder
# ---------------------------------------------------------------------------


def _build_blocked_artifact(
    t0: float,
    started_at: str,
    reason: str,
    gpu_diagnostics: dict[str, Any],
    stderr_excerpt: str = "",
    stdout_excerpt: str = "",
) -> dict[str, Any]:
    """Build an honest blocked artifact when GPU inference is unavailable.

    **Why 'blocked' and not 'failure':**
        A blocked artifact means the infrastructure condition (GPU VRAM, driver,
        pre-warm timeout) prevented execution — distinct from a code error.
        Blocked artifacts are expected in the research record when GPU resources
        are contested; they document the constraint without fabricating results.

    Args:
        t0:            perf_counter() timestamp from when the experiment started.
        started_at:    ISO-8601 UTC string recorded at start.
        reason:        Human-readable explanation of why the run was blocked.
        gpu_diagnostics: Dict from _get_gpu_diagnostics().
        stderr_excerpt: Last ~2000 chars of subprocess stderr (for diagnosis).
        stdout_excerpt: Last ~3000 chars of subprocess stdout (for diagnosis).

    Returns:
        Artifact dict with status="blocked" and inference_mode="blocked".
    """
    duration = round(time.perf_counter() - t0, 3)
    now = _utc_now()
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "title": TITLE,
        "inference_mode": "blocked",
        "status": "blocked",
        "run_date": _run_date(),
        "started_at": started_at,
        "finished_at": now,
        "duration_s": duration,
        "primary_result_path": LIVE_RELAY_OUTPUT,
        "improvement_1to3": None,
        "jepa_skip_rate_live": None,
        "simulation_comparison": {},
        "blocked_reason": reason,
        "stderr_excerpt": stderr_excerpt[:2000],
        "stdout_excerpt": stdout_excerpt[:3000],
        "gpu_diagnostics": gpu_diagnostics,
    }


# ---------------------------------------------------------------------------
# Load simulated baseline
# ---------------------------------------------------------------------------


def _load_simulated_result() -> dict[str, Any] | None:
    """Load the Exp 318 simulated relay result for comparison, or None if absent."""
    sim_path = _REPO_ROOT / SIMULATED_INPUT
    if not sim_path.exists():
        return None
    with sim_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_experiment_329() -> dict[str, Any]:
    """Execute the full Exp 329 lifecycle and return the wrapper artifact.

    **Steps:**
    1. GPU pre-check (DualGPUMonitor).
    2. Run experiment_318_self_learning_relay.py with CARNOT_FORCE_LIVE=1.
       Reuse existing live_gpu result if one already exists to avoid OOM.
    3. Validate inference_mode="live_gpu".
    4. Compute simulation_comparison against Exp 318 simulated baseline.
    5. Build and write wrapper artifact.

    Returns:
        Wrapper artifact dict (status="success" or "blocked").

    Spec: REQ-LEARN-014, SCENARIO-LEARN-023, SCENARIO-LEARN-024
    """
    t0 = time.perf_counter()
    started_at = _utc_now()

    print(f"[Exp 329] {TITLE}")
    print(f"[Exp 329] Started: {started_at}")

    # ------------------------------------------------------------------ #
    # Step 1: GPU pre-check                                               #
    # ------------------------------------------------------------------ #
    print("[Exp 329] Step 1: GPU pre-check via DualGPUMonitor ...")
    gpu_diagnostics = _get_gpu_diagnostics()
    monitor_result = gpu_diagnostics.get("dual_gpu_monitor", {})
    all_healthy = monitor_result.get("health", {}).get("all_healthy", False)
    print(f"[Exp 329] DualGPUMonitor all_healthy={all_healthy}")
    if not all_healthy:
        raw = gpu_diagnostics.get("nvidia_smi_raw", "")
        print("[Exp 329] NOTE: DualGPUMonitor reports all_healthy=False.")
        print("[Exp 329] NOTE: May be driver compatibility issue with 'gpu_index' field.")
        print(f"[Exp 329] Raw GPU state:\n{raw}")

    # ------------------------------------------------------------------ #
    # Step 2: Obtain relay result (reuse existing or run fresh)           #
    # ------------------------------------------------------------------ #
    live_output_path = _REPO_ROOT / LIVE_RELAY_OUTPUT
    live_output_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = ""
    stderr = ""

    need_run = True
    if live_output_path.exists():
        print(f"[Exp 329] Step 2: Checking existing result at {live_output_path} ...")
        try:
            with live_output_path.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
            if existing.get("inference_mode") == "live_gpu":
                print("[Exp 329] REUSE: Valid live_gpu result exists — skipping re-run.")
                print(f"[Exp 329]   batch1_accuracy={existing.get('batch1_accuracy', 'N/A')}")
                print(f"[Exp 329]   batch3_accuracy={existing.get('batch3_accuracy', 'N/A')}")
                print(f"[Exp 329]   improvement_1to3={existing.get('improvement_1to3', 'N/A')}")
                need_run = False
            else:
                print(f"[Exp 329] Existing inference_mode={existing.get('inference_mode')!r} — re-running.")
        except Exception as exc:
            print(f"[Exp 329] Could not read existing result ({exc}) — re-running.")
    else:
        print("[Exp 329] No existing live result — will run Exp 318 with CARNOT_FORCE_LIVE=1.")

    if need_run:
        relay_script = _REPO_ROOT / "scripts" / "experiment_318_self_learning_relay.py"
        env = {**os.environ, "CARNOT_FORCE_LIVE": "1", "JAX_PLATFORMS": "cpu"}
        cmd = [
            sys.executable,
            str(relay_script),
            "--output",
            str(live_output_path),
        ]
        print(f"[Exp 329] Step 2: Running relay benchmark ...")
        print(f"[Exp 329] Command: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                cwd=str(_REPO_ROOT),
                timeout=1800,  # 30-minute hard limit (3 batches × 5 min/batch + overhead)
            )
            run_duration = round(time.perf_counter() - t0, 1)
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            print(f"[Exp 329] Relay returned in {run_duration}s (rc={proc.returncode})")
            for line in stdout.strip().splitlines()[-30:]:
                print(f"  {line}")
            if proc.returncode != 0:
                reason = f"Relay script exited with rc={proc.returncode}"
                return _build_blocked_artifact(
                    t0, started_at, reason, gpu_diagnostics,
                    stderr[-2000:], stdout[-3000:]
                )
        except subprocess.TimeoutExpired:
            reason = "Relay script timed out after 1800s"
            return _build_blocked_artifact(t0, started_at, reason, gpu_diagnostics)

    # ------------------------------------------------------------------ #
    # Step 3: Validate inference_mode="live_gpu"                          #
    # ------------------------------------------------------------------ #
    print("[Exp 329] Step 3: Validating live GPU relay result ...")
    if not live_output_path.exists():
        reason = f"Relay completed but output file not found: {live_output_path}"
        print(f"[Exp 329] BLOCKED: {reason}")
        return _build_blocked_artifact(t0, started_at, reason, gpu_diagnostics)

    with live_output_path.open("r", encoding="utf-8") as fh:
        live_result = json.load(fh)

    inference_mode = live_result.get("inference_mode", "unknown")
    print(f"[Exp 329] inference_mode={inference_mode!r}")

    if inference_mode != "live_gpu":
        reason = (
            f"Relay fell back to inference_mode={inference_mode!r}. "
            "GPU pre-warm likely failed. See stdout/stderr above for root cause."
        )
        print(f"[Exp 329] BLOCKED: {reason}")
        artifact = _build_blocked_artifact(
            t0, started_at, reason, gpu_diagnostics,
            stderr[-2000:], stdout[-3000:]
        )
        artifact["fallback_result_path"] = LIVE_RELAY_OUTPUT
        artifact["fallback_inference_mode"] = inference_mode
        return artifact

    print("[Exp 329] VALIDATED: inference_mode='live_gpu' confirmed.")

    # Extract key metrics from live result
    b1_acc = float(live_result.get("batch1_accuracy", 0.0))
    b3_acc = float(live_result.get("batch3_accuracy", 0.0))
    imp_1to3 = float(live_result.get("improvement_1to3", 0.0))
    jepa_skip_rate_live = float(live_result.get("jepa_skip_rate", 0.0))

    print(f"[Exp 329] batch1_accuracy={b1_acc:.4f}")
    print(f"[Exp 329] batch3_accuracy={b3_acc:.4f}")
    print(f"[Exp 329] improvement_1to3={imp_1to3:+.4f}")
    print(f"[Exp 329] jepa_skip_rate_live={jepa_skip_rate_live:.4f}")

    # ------------------------------------------------------------------ #
    # Step 4: Compute simulation_comparison                               #
    # ------------------------------------------------------------------ #
    print("[Exp 329] Step 4: Computing simulation comparison ...")
    simulated_result = _load_simulated_result()
    if simulated_result is not None:
        simulation_comparison = compare_relay_to_simulated(live_result, simulated_result)
        sim_imp = simulated_result.get("improvement_1to3", 0.0)
        sim_jepa = simulated_result.get("jepa_skip_rate", 0.0)
        print(f"[Exp 329]   simulated improvement_1to3={sim_imp:+.4f}  live={imp_1to3:+.4f}")
        print(f"[Exp 329]   improvement_delta={simulation_comparison['improvement_delta']:+.4f}")
        print(f"[Exp 329]   simulated jepa_skip_rate={sim_jepa:.4f}  live={jepa_skip_rate_live:.4f}")
    else:
        simulation_comparison = {}
        sim_jepa = None
        print("[Exp 329] WARNING: Exp 318 simulated result not found; skipping comparison.")

    # ------------------------------------------------------------------ #
    # Step 5: Build wrapper artifact                                      #
    # ------------------------------------------------------------------ #
    print("[Exp 329] Step 5: Building wrapper artifact ...")
    finished_at = _utc_now()
    duration = round(time.perf_counter() - t0, 3)

    wrapper: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "title": TITLE,
        "inference_mode": "live_gpu",
        "status": "success",
        "run_date": _run_date(),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration,
        # Primary metric — signed float, never clamped (SCENARIO-LEARN-022)
        "improvement_1to3": imp_1to3,
        # Gate efficiency metrics
        "jepa_skip_rate_live": jepa_skip_rate_live,
        "jepa_skip_rate_simulated": sim_jepa,
        # Simulation comparison (SCENARIO-LEARN-024)
        "simulation_comparison": simulation_comparison,
        # Paths
        "primary_result_path": LIVE_RELAY_OUTPUT,
        "simulated_result_path": SIMULATED_INPUT,
        # Embedded live result for convenience
        "live_result": {
            "batch1_accuracy": b1_acc,
            "batch2_accuracy": float(live_result.get("batch2_accuracy", 0.0)),
            "batch3_accuracy": b3_acc,
            "improvement_1to2": float(live_result.get("improvement_1to2", 0.0)),
            "improvement_1to3": imp_1to3,
            "jepa_skip_rate": jepa_skip_rate_live,
            "z3_sat_rate": float(live_result.get("z3_sat_rate", 0.0)),
        },
        # GPU diagnostics for provenance
        "gpu_diagnostics": gpu_diagnostics,
    }

    # Write wrapper artifact
    wrapper_path = _REPO_ROOT / WRAPPER_OUTPUT
    _write_artifact(wrapper_path, wrapper)
    print(f"[Exp 329] Wrapper artifact written to {wrapper_path}")
    print(f"[Exp 329] Done. improvement_1to3={imp_1to3:+.4f} ({duration:.1f}s)")

    return wrapper


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point for Exp 329."""
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 329 — Live GPU Relay Wrapper")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Override wrapper artifact path (default: results/experiment_329_live_relay_results.json)",
    )
    args = parser.parse_args()

    artifact = run_experiment_329()

    if args.output_path is not None:
        # Re-write to the requested path
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        print(f"[Exp 329] Artifact also written to {args.output_path}")

    status = artifact.get("status", "unknown")
    if status == "blocked":
        print(f"[Exp 329] BLOCKED: {artifact.get('blocked_reason', 'unknown reason')}")
        sys.exit(1)
    print(f"[Exp 329] SUCCESS: improvement_1to3={artifact.get('improvement_1to3', 'N/A')}")


if __name__ == "__main__":
    main()
