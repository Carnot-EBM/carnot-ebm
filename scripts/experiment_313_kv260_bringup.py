#!/usr/bin/env python3
"""Experiment 313: KV260 FPGA hardware bring-up with honest latency measurement.

**Researcher summary:**
    Attempts actual KV260 FPGA hardware bring-up following the honest_verdict
    pattern from Exp 303. The experiment checks each prerequisite in sequence:
    CARNOT_KV260_BITFILE env var, pynq importability, overlay load, AXI register
    round-trip, and spin validity. The first failing check sets honest_verdict
    and the experiment continues to measure CPU fallback latency regardless.

    Target from arXiv 2602.15985: 77.5μs convergence. KV260 goal: <100μs for
    100-spin Ising problems when the overlay is loaded correctly.

**Honest labeling (honest_verdict):**
    - "hardware_working"    — overlay loaded, AXI round-trip passed, spins valid,
                              mean_latency_us ≤ 100μs
    - "blocked_no_bitfile"  — CARNOT_KV260_BITFILE env var not set
    - "blocked_pynq"        — pynq not importable (ImportError from overlay factory)
    - "blocked_overlay"     — overlay load returned None or raised non-import error
    - "blocked_timeout"     — STATUS_DONE never asserted within timeout

**CPU fallback:**
    Always measured (100 trials, 100 spins) for comparison, regardless of
    hardware status. Reported as cpu_fallback_mean_latency_us / p99_latency_us.

**Bringup steps:**
    Each step increments bringup_steps_passed:
    0: env var check passed (bitfile path set)
    1: pynq importable + overlay loaded successfully
    2: AXI register round-trip completed
    3: spin validity check passed
    4: latency measurement completed within target

Writes:
    results/experiment_313_kv260_bringup.json

Spec: REQ-SAMPLE-012,
      SCENARIO-SAMPLE-025, SCENARIO-SAMPLE-026

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_313_kv260_bringup.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID: int = 313
RUN_DATE: str = "20260414"
BENCHMARK_NAME: str = "kv260_bringup_313"
DEFAULT_OUTPUT: Path = Path("results/experiment_313_kv260_bringup.json")

BITFILE_ENV: str = "CARNOT_KV260_BITFILE"

# Number of spins for the hardware target problem.
# The KV260 goal from arXiv 2602.15985 targets 77.5μs for small problems.
N_SPINS: int = 100

# Number of latency measurement trials (mean + p99 requires statistical depth).
LATENCY_TRIALS: int = 100

# Timeout for the AXI register round-trip (STATUS_DONE must assert within this).
DEFAULT_ROUNDTRIP_TIMEOUT: float = 10.0

# Hardware latency target (µs) — from arXiv 2602.15985 / KV260 spec.
HARDWARE_LATENCY_TARGET_US: float = 100.0

SPEC_REFS: list[str] = [
    "REQ-SAMPLE-012",
    "SCENARIO-SAMPLE-025",
    "SCENARIO-SAMPLE-026",
]


# ---------------------------------------------------------------------------
# Internal helpers (exported for test introspection)
# ---------------------------------------------------------------------------


def _check_bitfile_env() -> str | None:
    """Return the value of CARNOT_KV260_BITFILE, or None if unset.

    **Why exported:**
        Tests use this to decide whether to skip hardware-path tests.
        It reads only the environment variable — no filesystem access.
    """
    return os.environ.get(BITFILE_ENV)


def _try_import_pynq() -> bool:
    """Return True if pynq can be imported (without actually loading an overlay).

    **Why this matters:**
        PYNQ is only available on Xilinx/Kria platforms. On development machines
        it will be absent. We check importability here so tests can gate on it
        independently of the overlay load step.
    """
    try:
        import importlib

        importlib.import_module("pynq")
        return True
    except ImportError:
        return False


def _utc_now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_overlay_factory(bitfile_path: Any) -> Any:
    """Load the real PYNQ overlay and return the MMIO transport.

    **How it works:**
        Calls pynq.Overlay(bitfile_path, download=True) which programs the
        KV260 PL (programmable logic) with the Carnot Ising sampler bitstream.
        Then retrieves the carnot_ising_0 IP core's MMIO handle, which maps
        the AXI-Lite register space into the ARM processor's virtual address space.

    Raises:
        ImportError: if pynq is not installed.
        Any exception from pynq.Overlay: propagated to caller for error labeling.
    """
    import importlib

    pynq = importlib.import_module("pynq")
    overlay = pynq.Overlay(str(bitfile_path), download=True)
    mmio = getattr(getattr(overlay, "carnot_ising_0", None), "mmio", None)
    if mmio is None:
        return None

    # Wrap in a bound object so the overlay reference is kept alive.
    # If we don't hold onto `overlay`, Python GC may free the PYNQ driver
    # and invalidate the MMIO mapping underneath us.
    class _BoundMMIO:
        def __init__(self, _overlay: Any, _mmio: Any) -> None:
            self._overlay = _overlay
            self._mmio = _mmio

        def write(self, offset: int, value: int) -> None:
            self._mmio.write(offset, value)

        def read(self, offset: int) -> int:
            return int(self._mmio.read(offset))

    return _BoundMMIO(overlay, mmio)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


def detect_kv260_hardware(
    *,
    overlay_factory: Any = None,
) -> dict[str, Any]:
    """Check hardware availability in sequence; return detection result dict.

    **What "detection" means here:**
        We do not simply ask "is the FPGA present?" — we walk through the three
        prerequisite checks that must all pass before we can exercise real hardware:
        1. CARNOT_KV260_BITFILE env var is set (gives us a path to the bitstream).
        2. The overlay factory succeeds (pynq is installed AND the bitfile loads).
        3. The factory returns a non-None transport object (IP core found in overlay).

        The first check that fails sets honest_verdict and ends the detection phase.
        We return a transport=None for all blocked outcomes so callers don't need
        to check both verdict and transport.

    Returns:
        dict with keys:
            honest_verdict: str — one of the APPROVED_VERDICTS or None (if hardware ok)
            kv260_detected: bool
            transport: the MMIO transport object, or None
            bitfile_path: str | None

    Spec: REQ-SAMPLE-012, SCENARIO-SAMPLE-026
    """
    factory = overlay_factory or _default_overlay_factory

    # Step 1: env var check
    bitfile_path = _check_bitfile_env()
    if bitfile_path is None:
        return {
            "honest_verdict": "blocked_no_bitfile",
            "kv260_detected": False,
            "transport": None,
            "bitfile_path": None,
        }

    # Steps 2+3: overlay load
    try:
        transport = factory(bitfile_path)
    except ImportError:
        # pynq not installed on this machine
        return {
            "honest_verdict": "blocked_pynq",
            "kv260_detected": False,
            "transport": None,
            "bitfile_path": bitfile_path,
        }
    except Exception:
        # bitfile bad, pynq version mismatch, or other overlay load error
        return {
            "honest_verdict": "blocked_overlay",
            "kv260_detected": False,
            "transport": None,
            "bitfile_path": bitfile_path,
        }

    if transport is None:
        # Overlay loaded but the carnot_ising_0 IP core was not found.
        return {
            "honest_verdict": "blocked_overlay",
            "kv260_detected": False,
            "transport": None,
            "bitfile_path": bitfile_path,
        }

    # All checks passed — transport is ready.
    return {
        "honest_verdict": None,  # not yet determined — need AXI round-trip
        "kv260_detected": True,
        "transport": transport,
        "bitfile_path": bitfile_path,
    }


# ---------------------------------------------------------------------------
# Spin validity
# ---------------------------------------------------------------------------


def spin_validity_check(
    spins: np.ndarray,
    *,
    expected_n: int,
) -> tuple[bool, int, bool]:
    """Validate that all spins are exactly ±1 and the array has the expected length.

    **Why ±1 matters:**
        The Ising model uses spin variables s_i ∈ {-1, +1}. Any value outside
        this set indicates a bug in the hardware readback path (e.g. uninitialized
        register, bit-packing error, or AXI read timeout returning 0x00000000).
        We check this deterministically so experiments never silently accept
        corrupted spin states.

    Args:
        spins: 1-D array of spin values (any numeric dtype).
        expected_n: Expected number of spins (used for shape validation).

    Returns:
        Tuple (valid, n_spins, shape_ok) where:
            valid: True if all values are exactly ±1.
            n_spins: Actual length of the array.
            shape_ok: True if len(spins) == expected_n.

    Spec: REQ-SAMPLE-012, SCENARIO-SAMPLE-025
    """
    flat = spins.ravel()
    n_spins = int(flat.shape[0])
    shape_ok = n_spins == expected_n
    valid = bool(np.all((flat == 1) | (flat == -1)))
    return valid, n_spins, shape_ok


# ---------------------------------------------------------------------------
# AXI register round-trip + latency measurement
# ---------------------------------------------------------------------------


def _run_hardware_roundtrip(
    transport: Any,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Exercise the AXI-Lite register map and return the round-trip result.

    **What this measures:**
        We build a 100-spin ring Ising problem, upload it via AXI writes, trigger
        sampling via CONTROL.START, and poll STATUS_DONE. The latency is the wall
        time from CONTROL.START to STATUS_DONE asserting.

        This is the same protocol as Exp 288 but extended to measure 100 independent
        trials so we can report mean_latency_us and p99_latency_us.

    Returns:
        dict with mean_latency_us, p99_latency_us, spin_state_valid, sample_shape.

    Raises:
        RuntimeError: if STATUS_DONE does not assert within timeout_seconds.

    Spec: REQ-SAMPLE-012, SCENARIO-SAMPLE-025
    """
    from carnot.samplers.fpga_ising import (
        AXILiteRegisterMap,
        FPGAIsingSampler,
        _quantize_word,
        unpack_sample_words,
    )

    # Build the problem once; re-use it across all latency trials.
    biases = np.full(N_SPINS, 0.2, dtype=np.float32)
    couplings = np.zeros((N_SPINS, N_SPINS), dtype=np.float32)
    for i in range(N_SPINS):
        for d in range(1, 5):
            j = (i + d) % N_SPINS
            couplings[i, j] = 0.35
            couplings[j, i] = 0.35

    sampler = FPGAIsingSampler(
        mode="hardware",
        allow_cpu_fallback=False,
        overlay_factory=lambda _bitfile: transport,
    )
    compiled = sampler.upload_problem(biases, couplings)
    regmap = AXILiteRegisterMap()

    # Upload problem parameters once.
    beta = 6.0
    transport.write(regmap.SPIN_COUNT, compiled.n_spins)
    transport.write(regmap.SAMPLE_COUNT, 1)
    transport.write(regmap.WARMUP_STEPS, 40)
    transport.write(regmap.STEPS_PER_SAMPLE, 10)
    transport.write(regmap.BETA_INIT, _quantize_word(beta, sampler.architecture.frac_bits))
    transport.write(regmap.BETA_FINAL, _quantize_word(beta, sampler.architecture.frac_bits))
    transport.write(regmap.RUN_FLAGS, 0)

    words_per_sample = max(1, (compiled.n_spins + 31) // 32)
    latencies_us: list[float] = []
    last_sample: np.ndarray | None = None

    for _trial in range(LATENCY_TRIALS):
        t0 = time.perf_counter()
        transport.write(regmap.CONTROL, regmap.CONTROL_CLEAR_RESULTS | regmap.CONTROL_START)
        deadline = t0 + timeout_seconds
        status = transport.read(regmap.STATUS)
        while not (status & regmap.STATUS_DONE):
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    f"KV260 STATUS_DONE not asserted within {timeout_seconds:.2f}s "
                    f"(trial {_trial + 1}/{LATENCY_TRIALS})"
                )
            status = transport.read(regmap.STATUS)
        latency_us = (time.perf_counter() - t0) * 1e6
        latencies_us.append(latency_us)

        # Read back spin state from the last trial.
        words = [
            transport.read(regmap.sample_offset(0, w, words_per_sample))
            for w in range(words_per_sample)
        ]
        last_sample = unpack_sample_words(words, n_spins=compiled.n_spins)

    arr = np.array(latencies_us)
    # Convert boolean spin array to ±1 for validity check.
    pm1 = np.where(last_sample, np.int8(1), np.int8(-1))  # type: ignore[arg-type]
    valid, n_spins, shape_ok = spin_validity_check(pm1, expected_n=N_SPINS)

    return {
        "mean_latency_us": float(np.mean(arr)),
        "p99_latency_us": float(np.percentile(arr, 99)),
        "min_latency_us": float(np.min(arr)),
        "max_latency_us": float(np.max(arr)),
        "n_trials": LATENCY_TRIALS,
        "spin_state_valid": valid,
        "sample_shape": [1, int(n_spins)],
    }


def _measure_cpu_fallback_latency(n_trials: int = LATENCY_TRIALS) -> dict[str, Any]:
    """Measure 100-spin Ising sampling latency via the CPU backend.

    **Why we always measure this:**
        Even when the KV260 hardware is unavailable, the CPU fallback latency
        gives us a reference point: how fast is the software implementation on
        this machine? This makes the blocked artifacts still informative rather
        than empty. It also lets us quantify the hardware speedup when the KV260
        eventually works.

    Args:
        n_trials: Number of timing samples to collect. Defaults to LATENCY_TRIALS
            (100). Tests may pass a smaller value to avoid JIT compilation overhead.

    Returns:
        dict with mean_latency_us, p99_latency_us, and cpu_fallback_latency_us
        (the mean, kept as a flat top-level alias for convenience).

    Spec: REQ-SAMPLE-012, SCENARIO-SAMPLE-026
    """
    from carnot.samplers.fpga_backend import FpgaBackend

    biases = np.full(N_SPINS, 0.2, dtype=np.float32)
    couplings = np.zeros((N_SPINS, N_SPINS), dtype=np.float32)
    for i in range(N_SPINS):
        j = (i + 1) % N_SPINS
        couplings[i, j] = 0.35
        couplings[j, i] = 0.35

    backend = FpgaBackend()
    config = {"beta": 6.0, "steps_per_sample": 10, "n_warmup": 40}

    latencies_us: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        backend.sample(biases, couplings, 1, config)
        latencies_us.append((time.perf_counter() - t0) * 1e6)

    arr = np.array(latencies_us)
    mean_us = float(np.mean(arr))
    return {
        "cpu_fallback_mean_latency_us": mean_us,
        "cpu_fallback_p99_latency_us": float(np.percentile(arr, 99)),
        "cpu_fallback_latency_us": mean_us,
    }


# ---------------------------------------------------------------------------
# Main experiment flow
# ---------------------------------------------------------------------------


def run_experiment(
    *,
    output_path: Path | None = None,
    overlay_factory: Any = None,
    roundtrip_timeout_seconds: float = DEFAULT_ROUNDTRIP_TIMEOUT,
    write_output: bool = False,
    _cpu_trials: int = LATENCY_TRIALS,
) -> dict[str, Any]:
    """Run the Exp 313 KV260 bring-up flow and return the artifact payload.

    **Flow:**
    1. [DETECT] Call detect_kv260_hardware(); on any blocked verdict proceed to
       CPU fallback immediately.
    2. [AXI ROUND-TRIP] If transport available, call _run_hardware_roundtrip().
       RuntimeError → honest_verdict="blocked_timeout".
    3. [VERDICT] Determine honest_verdict:
       "hardware_working" only if all steps pass AND mean_latency_us ≤ 100μs.
    4. [CPU FALLBACK] Always measured regardless of hardware status.
    5. [ARTIFACT] Assemble and optionally write JSON.

    Args:
        _cpu_trials: Number of CPU fallback latency trials. Defaults to
            LATENCY_TRIALS (100). Tests may pass a smaller value (e.g. 2) to
            avoid JAX JIT compilation overhead causing timeout failures.

    Spec: REQ-SAMPLE-012, SCENARIO-SAMPLE-025, SCENARIO-SAMPLE-026
    """
    started_at = _utc_now()
    wall_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: hardware detection
    # ------------------------------------------------------------------
    detection = detect_kv260_hardware(overlay_factory=overlay_factory)
    honest_verdict: str = detection["honest_verdict"] or ""
    kv260_detected: bool = detection["kv260_detected"]
    transport = detection["transport"]
    bringup_steps_passed: int = 0
    hardware_latency_us: dict[str, Any] | None = None

    # Count env-var check as step 0 (passed if bitfile_path was set).
    if detection["bitfile_path"] is not None:
        bringup_steps_passed = 1  # step 0: env var set

    # If transport is available, overlay loaded successfully.
    if transport is not None:
        bringup_steps_passed = 2  # step 1: overlay loaded

    # ------------------------------------------------------------------
    # Step 2: AXI register round-trip + latency measurement (hardware path)
    # ------------------------------------------------------------------
    # Detect whether the transport is the software model — if so, we can
    # test AXI register contract but we must NOT report hardware latency.
    # Software model timing is meaningless CPU simulation, not FPGA latency.
    from carnot.samplers.fpga_ising import SoftwareFPGAOverlay

    is_real_hardware = transport is not None and not isinstance(transport, SoftwareFPGAOverlay)
    roundtrip_error: str | None = None

    if transport is not None and not honest_verdict:
        if is_real_hardware:
            # Real hardware path: run full round-trip with latency measurement.
            try:
                hw_result = _run_hardware_roundtrip(
                    transport,
                    timeout_seconds=roundtrip_timeout_seconds,
                )
                bringup_steps_passed = 3  # step 2: AXI round-trip completed

                spin_valid = hw_result["spin_state_valid"]
                if spin_valid:
                    bringup_steps_passed = 4  # step 3: spins valid

                mean_us = hw_result["mean_latency_us"]
                if spin_valid and mean_us <= HARDWARE_LATENCY_TARGET_US:
                    bringup_steps_passed = 5  # step 4: latency target met
                    honest_verdict = "hardware_working"
                elif spin_valid:
                    honest_verdict = "blocked_timeout"
                else:
                    honest_verdict = "blocked_overlay"

                hardware_latency_us = hw_result

            except RuntimeError as exc:
                roundtrip_error = str(exc)
                honest_verdict = "blocked_timeout"
        else:
            # Software model path: AXI register contract is testable, but the
            # latency numbers reflect CPU simulation speed, not FPGA performance.
            # We do NOT populate hardware_latency_us to avoid fabricating data.
            # bringup_steps_passed stays at 2 (overlay loaded, but not real HW).
            honest_verdict = "blocked_overlay"

    # ------------------------------------------------------------------
    # Step 3: CPU fallback latency (always measured)
    # ------------------------------------------------------------------
    cpu_latency = _measure_cpu_fallback_latency(n_trials=_cpu_trials)

    # ------------------------------------------------------------------
    # Step 4: Assemble artifact
    # ------------------------------------------------------------------
    finished_at = _utc_now()
    runtime_seconds = round(time.perf_counter() - wall_start, 6)

    payload: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "schema": {"artifact": "carnot.kv260_bringup_313.v1"},
        "run_date": RUN_DATE,
        "benchmark": BENCHMARK_NAME,
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_seconds": runtime_seconds,
        "honest_verdict": honest_verdict,
        "kv260_detected": kv260_detected,
        "bringup_steps_passed": bringup_steps_passed,
        "bitfile_path": detection["bitfile_path"],
        "hardware_latency_us": hardware_latency_us,
        "cpu_fallback_latency_us": cpu_latency["cpu_fallback_latency_us"],
        "cpu_fallback_mean_latency_us": cpu_latency["cpu_fallback_mean_latency_us"],
        "cpu_fallback_p99_latency_us": cpu_latency["cpu_fallback_p99_latency_us"],
        "spec_requirements": SPEC_REFS,
    }
    if roundtrip_error:
        payload["error"] = roundtrip_error

    if write_output and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exp 313: Attempt KV260 FPGA hardware bring-up with honest latency measurement."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output path for results/experiment_313_kv260_bringup.json",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_ROUNDTRIP_TIMEOUT,
        help="Hard timeout for AXI STATUS_DONE per trial (default: 10 s).",
    )
    return parser


def _get_repo_root() -> Path:
    """Return the repository root, preferring the CARNOT_REPO_ROOT override."""
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = _get_repo_root()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    payload = run_experiment(
        output_path=output_path,
        roundtrip_timeout_seconds=float(args.timeout_seconds),
        write_output=True,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
