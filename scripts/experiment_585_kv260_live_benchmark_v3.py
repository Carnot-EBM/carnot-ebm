#!/usr/bin/env python3
"""Experiment 585: KV260 Live Benchmark V3 — Hardware Ising Sampler vs CPU Baseline.

**Researcher summary:**
    GATED on Exp 584 (Vivado synthesis).  If bitfile_built=False in
    results/experiment_584_kv260_synthesis.json, this experiment writes a
    blocked artifact immediately and exits without consuming any resources.

    When a bitfile is available, this experiment benchmarks the KV260 FPGA
    hardware Ising sampler against the CPU baseline of 289608 µs (Exp 568).
    The target is mean hardware latency < 100 µs — a 2900x speedup that would
    validate the FPGA as a practical accelerator for Carnot's sampling pipeline.

**What this experiment does (when not blocked):**
    1. apply_env_autofix() — adjust JAX/ROCm environment before any heavy import.
    2. ExperimentTimeoutWatchdog(585, 60 min) — hard wall-clock cap.
    3. Gate check: load Exp 584 result.  If bitfile_built != True → blocked artifact.
    4. Load FpgaBackend with the bitfile path from Exp 584.
    5. Run 1000 trials of 100-spin Ising sampling; measure latency.
    6. Compute speedup_ratio and fpga_target_met (target: < 100 µs).
    7. Write artifact with schema='carnot.kv260_benchmark.v3'.
    8. tmpl.assert_deliverable_written() as FINAL LINE.

**Honest reporting:**
    honest_verdict is one of:
      'blocked_no_bitfile'   — Exp 584 did not produce a bitfile; benchmark skipped
      'hardware_working'     — mean latency < 100 µs; target met
      'hardware_too_slow'    — mean latency >= 100 µs; FPGA too slow
      'hardware_failed'      — FpgaBackend raised an exception during sampling

Spec: REQ-SAMPLE-033, SCENARIO-SAMPLE-055, SCENARIO-SAMPLE-056, SCENARIO-SAMPLE-057
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# apply_env_autofix() MUST come before any JAX/CUDA import.
# Why: ROCm plugin load order and JAX_PLATFORMS must be set before JAX init.
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID: int = 585
TITLE: str = "KV260 Live Benchmark V3"
DELIVERABLE: str = "results/experiment_585_kv260_live_benchmark_v3.json"
SCHEMA: str = "carnot.kv260_benchmark.v3"

# CPU baseline from Exp 568 — 100 trials of 100-spin Ising sampling on CPU.
CPU_BASELINE_LATENCY_US: float = 289608.0

# Hardware target: < 100 µs mean latency (2900× faster than CPU).
FPGA_LATENCY_TARGET_US: float = 100.0

# Benchmark parameters.
N_SPINS: int = 100
N_TRIALS: int = 1000

# Path to the Exp 584 gate artifact.
GATE_RESULT_PATH: str = "results/experiment_584_kv260_synthesis.json"


# ---------------------------------------------------------------------------
# Gate check helper — testable standalone
# ---------------------------------------------------------------------------

def load_gate_result(repo_root: Path) -> dict[str, Any]:
    """Load Exp 584 synthesis result and return the parsed dict.

    **Detailed explanation for engineers:**
        This is the gate check for the benchmark.  If the file is missing or
        bitfile_built is not True, the caller should write a blocked artifact
        and exit rather than attempting hardware sampling.

    Args:
        repo_root: Repository root directory.

    Returns:
        Parsed JSON dict, or a synthetic dict with bitfile_built=False if the
        file is absent.

    Spec: REQ-SAMPLE-033-1
    """
    gate_path = repo_root / GATE_RESULT_PATH
    if not gate_path.exists():
        logger.warning("Gate file %s not found; treating as bitfile_built=False.", gate_path)
        return {"bitfile_built": False, "bitfile_path": None, "honest_verdict": "missing"}
    with gate_path.open() as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def check_bitfile_env_match(bitfile_path: str) -> bool:
    """Return True if CARNOT_KV260_BITFILE matches bitfile_path.

    **Detailed explanation for engineers:**
        When both the env var and the Exp 584 path are set but differ, this is
        a configuration smell — the user may have overridden the env var to
        point at a different bitfile.  We warn but do not abort.

    Args:
        bitfile_path: Path from Exp 584 result artifact.

    Returns:
        True if the env var matches or is unset, False if they differ.

    Spec: REQ-SAMPLE-033-2
    """
    env_val = os.environ.get("CARNOT_KV260_BITFILE")
    if env_val is None:
        return True  # Env var not set; FpgaBackend will read from explicit kwarg.
    match = env_val == bitfile_path
    if not match:
        logger.warning(
            "CARNOT_KV260_BITFILE=%r does not match Exp 584 bitfile_path=%r. "
            "Using Exp 584 path for the benchmark.",
            env_val,
            bitfile_path,
        )
    return match


def run_hardware_benchmark(
    bitfile_path: str,
    n_spins: int = N_SPINS,
    n_trials: int = N_TRIALS,
) -> dict[str, Any]:
    """Run the FPGA Ising benchmark and return raw timing results.

    **Detailed explanation for engineers:**
        Instantiates FpgaBackend with the given bitfile path, then calls
        sample() n_trials times for a 100-spin fully-connected ferromagnetic
        Ising problem.  Each call is timed individually so we can compute mean
        and std latency.

        We use a ferromagnetic problem (all-negative couplings, zero biases)
        because it has a known ground state (all-up or all-down) that lets
        us sanity-check whether the hardware is returning physically
        reasonable results.

        If FpgaBackend raises an exception on any trial, we record
        hardware_failed=True and return immediately with the samples collected
        so far (may be an empty list).

    Args:
        bitfile_path: Path to the .bit file for the KV260 overlay.
        n_spins: Number of Ising spins (default 100).
        n_trials: Number of independent sampling calls to time (default 1000).

    Returns:
        Dict with keys: latencies_us (list[float]), hardware_failed (bool),
        error_message (str | None), n_completed (int).

    Spec: REQ-SAMPLE-033-2
    """
    from carnot.samplers.fpga_backend import FpgaBackend  # local import; FPGA deps optional

    backend = FpgaBackend(bitfile_path=bitfile_path)

    # Simple ferromagnetic Ising: zero biases, all couplings = -1.
    biases = np.zeros(n_spins, dtype=np.float32)
    couplings = -np.ones((n_spins, n_spins), dtype=np.float32)
    np.fill_diagonal(couplings, 0.0)  # no self-coupling

    config: dict[str, Any] = {"beta": 2.0, "n_steps": 100}

    latencies_us: list[float] = []
    for _ in range(n_trials):
        try:
            t0 = time.perf_counter()
            backend.sample(biases, couplings, n_samples=1, config=config)
            elapsed_us = (time.perf_counter() - t0) * 1e6
            latencies_us.append(elapsed_us)
        except Exception as exc:  # noqa: BLE001
            logger.error("FpgaBackend.sample() raised %s: %s", type(exc).__name__, exc)
            return {
                "latencies_us": latencies_us,
                "hardware_failed": True,
                "error_message": str(exc),
                "n_completed": len(latencies_us),
            }

    return {
        "latencies_us": latencies_us,
        "hardware_failed": False,
        "error_message": None,
        "n_completed": len(latencies_us),
    }


def compute_benchmark_stats(
    latencies_us: list[float],
    cpu_baseline_us: float = CPU_BASELINE_LATENCY_US,
    target_us: float = FPGA_LATENCY_TARGET_US,
) -> dict[str, Any]:
    """Compute summary statistics from raw latency measurements.

    **Detailed explanation for engineers:**
        Converts the raw per-trial latency list to mean, std, and derived
        metrics.  All latency values are in microseconds for consistency with
        Exp 568 and the hardware target.

    Args:
        latencies_us: Per-trial latency in microseconds.
        cpu_baseline_us: CPU baseline from Exp 568.
        target_us: Hardware latency target (default 100 µs).

    Returns:
        Dict with mean_hardware_latency_us, std_hardware_latency_us,
        speedup_ratio, fpga_target_met.

    Spec: REQ-SAMPLE-033-3, REQ-SAMPLE-033-4
    """
    if not latencies_us:
        return {
            "mean_hardware_latency_us": None,
            "std_hardware_latency_us": None,
            "speedup_ratio": None,
            "fpga_target_met": False,
        }
    arr = np.array(latencies_us, dtype=np.float64)
    mean_us = float(arr.mean())
    std_us = float(arr.std())
    speedup = cpu_baseline_us / mean_us if mean_us > 0 else None
    return {
        "mean_hardware_latency_us": mean_us,
        "std_hardware_latency_us": std_us,
        "speedup_ratio": speedup,
        "fpga_target_met": mean_us < target_us,
    }


def choose_verdict(
    hardware_failed: bool,
    mean_latency_us: float | None,
    target_us: float = FPGA_LATENCY_TARGET_US,
) -> str:
    """Return honest_verdict string from benchmark outcome.

    **Detailed explanation for engineers:**
        Three outcomes:
        - hardware_failed: FpgaBackend raised an exception; hardware is broken.
        - hardware_too_slow: Benchmark completed but latency >= target.
        - hardware_working: Benchmark completed and latency < target.

    Spec: REQ-SAMPLE-033-5
    """
    if hardware_failed or mean_latency_us is None:
        return "hardware_failed"
    if mean_latency_us < target_us:
        return "hardware_working"
    return "hardware_too_slow"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entrypoint for Exp 585."""
    with ExperimentTimeoutWatchdog(EXPERIMENT_ID, timeout_minutes=60):
        tmpl = ExperimentTemplate(
            EXPERIMENT_ID,
            TITLE,
            DELIVERABLE,
            requires_gpu=False,
        )
        tmpl.setup()

        result_path = _REPO_ROOT / DELIVERABLE

        # --- GATE CHECK ---------------------------------------------------
        gate = load_gate_result(_REPO_ROOT)
        if not gate.get("bitfile_built"):
            blocked: dict[str, Any] = {
                "schema": SCHEMA,
                "experiment": EXPERIMENT_ID,
                "status": "blocked",
                "bitfile_built": False,
                "bitfile_path": None,
                "hardware_latency_us": None,
                "std_latency_us": None,
                "cpu_baseline_latency_us": CPU_BASELINE_LATENCY_US,
                "speedup_ratio": None,
                "fpga_target_met": False,
                "n_trials": 0,
                "honest_verdict": "blocked_no_bitfile",
                "upstream_exp": 584,
            }
            result_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = result_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(blocked, indent=2))
            os.replace(str(tmp), str(result_path))
            logger.info("Exp 584 bitfile_built=False — blocked artifact written.")
            tmpl.assert_deliverable_written()
            return

        # --- BITFILE ENV CHECK --------------------------------------------
        bitfile_path: str = gate["bitfile_path"]
        check_bitfile_env_match(bitfile_path)

        # --- RUN BENCHMARK ------------------------------------------------
        raw = run_hardware_benchmark(bitfile_path, n_spins=N_SPINS, n_trials=N_TRIALS)
        stats = compute_benchmark_stats(raw["latencies_us"])
        verdict = choose_verdict(
            raw["hardware_failed"],
            stats["mean_hardware_latency_us"],
        )

        # --- BUILD ARTIFACT -----------------------------------------------
        artifact: dict[str, Any] = {
            "schema": SCHEMA,
            "experiment": EXPERIMENT_ID,
            "status": "success" if not raw["hardware_failed"] else "error",
            "bitfile_path": bitfile_path,
            "hardware_latency_us": stats["mean_hardware_latency_us"],
            "std_latency_us": stats["std_hardware_latency_us"],
            "cpu_baseline_latency_us": CPU_BASELINE_LATENCY_US,
            "speedup_ratio": stats["speedup_ratio"],
            "fpga_target_met": stats["fpga_target_met"],
            "n_trials": raw["n_completed"],
            "honest_verdict": verdict,
        }
        if raw["error_message"]:
            artifact["error_message"] = raw["error_message"]

        result_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = result_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(artifact, indent=2))
        os.replace(str(tmp), str(result_path))

        logger.info(
            "Exp 585 complete: verdict=%s mean_latency=%.1f µs speedup=%.1f×",
            verdict,
            stats["mean_hardware_latency_us"] or 0,
            stats["speedup_ratio"] or 0,
        )

        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
