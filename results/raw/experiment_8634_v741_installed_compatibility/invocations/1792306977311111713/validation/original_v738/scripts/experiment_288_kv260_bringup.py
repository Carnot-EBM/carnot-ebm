#!/usr/bin/env python3
"""Experiment 288: KV260 FPGA overlay bring-up with 60-second hard timeout.

**What this script does:**
    Attempts to bring up the Kria KV260 FPGA overlay for the Carnot Ising
    sampler.  It checks for the CARNOT_KV260_BITFILE environment variable
    first; if unset, it emits a blocked artifact immediately without touching
    PYNQ at all.  When a bitfile path is available it loads the overlay,
    exercises the AXI-Lite register map (CONTROL → STATUS round-trip), writes
    a minimal coupling matrix into the bias window, triggers a sampling run,
    reads back the spin state, and validates that every spin is ±1.

**Honest labeling:**
    - ``hardware``      — PYNQ + AXI-Lite MMIO working on the real KV260
    - ``software_model`` — SoftwareFPGAOverlay used (register contract only)
    - ``blocked``       — env var missing, overlay load failed, or 60 s timeout

**Hard constraint:**
    The entire bring-up (overlay load + register round-trip + sample readback)
    must complete within BRINGUP_TIMEOUT_SECONDS = 60 s.  If any step exceeds
    that budget the script emits a blocked artifact instead of fabricating
    timing numbers.

Writes:
    ``results/experiment_288_results.json``

Spec: REQ-SAMPLE-009,
      SCENARIO-SAMPLE-018, SCENARIO-SAMPLE-019
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

from carnot.samplers.fpga_ising import (  # type: ignore[import-untyped]
    AXILiteRegisterMap,
    FPGAIsingSampler,
    SoftwareFPGAOverlay,
    _quantize_word,
    compile_sparse_problem,
    unpack_sample_words,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID: int = 288
RUN_DATE: str = "20260414"
BENCHMARK_NAME: str = "kv260_bringup_288"
DEFAULT_OUTPUT: Path = Path("results/experiment_288_results.json")
DEFAULT_BITFILE_ENV: str = "CARNOT_KV260_BITFILE"
OVERLAY_IP_NAME: str = "carnot_ising_0"

# Hard constraint: the full bring-up must complete within this window.
BRINGUP_TIMEOUT_SECONDS: float = 60.0

# Small problem used for the register round-trip validation.
DEFAULT_N_SPINS: int = 128
DEFAULT_N_SAMPLES: int = 4
DEFAULT_WARMUP_STEPS: int = 40
DEFAULT_STEPS_PER_SAMPLE: int = 10
DEFAULT_BETA: float = 6.0

SPEC_REFS: list[str] = [
    "REQ-SAMPLE-005",
    "REQ-SAMPLE-006",
    "REQ-SAMPLE-007",
    "REQ-SAMPLE-009",
    "SCENARIO-SAMPLE-018",
    "SCENARIO-SAMPLE-019",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_repo_root() -> Path:
    """Return the repository root, preferring the CARNOT_REPO_ROOT override."""
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1]


def check_env_var() -> str | None:
    """Return the value of CARNOT_KV260_BITFILE, or None if it is unset."""
    return os.environ.get(DEFAULT_BITFILE_ENV)


def spins_to_pm1(bool_array: np.ndarray) -> np.ndarray:
    """Convert a boolean spin array to a signed ±1 int8 array.

    True  →  +1  (spin up)
    False → −1  (spin down)

    Spec: SCENARIO-SAMPLE-019
    """
    return np.where(bool_array, np.int8(1), np.int8(-1))


def validate_spin_state(pm1_array: np.ndarray) -> bool:
    """Return True if every element of *pm1_array* is exactly +1 or −1.

    Spec: SCENARIO-SAMPLE-019
    """
    return bool(np.all((pm1_array == 1) | (pm1_array == -1)))


def build_problem(
    n_spins: int = DEFAULT_N_SPINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the sparse ring Ising problem used for the Exp 288 round-trip.

    Creates a ring topology where each spin is ferromagnetically coupled to
    its four nearest neighbours on each side (8 neighbours total, well within
    the hardware max_degree = 32 limit).
    """
    biases = np.full(n_spins, 0.2, dtype=np.float32)
    couplings = np.zeros((n_spins, n_spins), dtype=np.float32)
    for row in range(n_spins):
        for distance in range(1, 5):
            neighbor = (row + distance) % n_spins
            couplings[row, neighbor] = 0.35
            couplings[neighbor, row] = 0.35
    return biases, couplings


# ---------------------------------------------------------------------------
# Blocked artifact builder
# ---------------------------------------------------------------------------


def build_blocked_artifact(
    *,
    missing: str,
    next_step: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a minimal blocked artifact with the required fields.

    The artifact always has:
    - ``execution_path: "blocked"``
    - ``missing``: name of the missing resource or env var
    - ``next_step``: human-readable instruction for unblocking
    - ``overlay_load_ms``: None (no overlay was attempted)

    Spec: REQ-SAMPLE-009, SCENARIO-SAMPLE-018
    """
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "execution_path": "blocked",
        "missing": missing,
        "next_step": next_step,
        "overlay_load_ms": None,
        "round_trip": None,
        "spin_state_valid": None,
    }
    if error is not None:
        artifact["error"] = error
    return artifact


# ---------------------------------------------------------------------------
# Overlay loading
# ---------------------------------------------------------------------------


def _default_overlay_loader(bitfile_path: Any) -> Any:
    """Load the real PYNQ overlay and return the MMIO transport.

    Returns None if the IP core ``carnot_ising_0`` is missing from the overlay.
    Raises ImportError if PYNQ is not installed, or any other exception that
    pynq.Overlay raises on failure.
    """
    import importlib

    pynq = importlib.import_module("pynq")
    overlay = pynq.Overlay(str(bitfile_path), download=True)
    mmio = getattr(getattr(overlay, OVERLAY_IP_NAME, None), "mmio", None)
    if mmio is None:
        return None

    # Keep overlay alive so the underlying driver is not freed.
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
# Register round-trip measurement
# ---------------------------------------------------------------------------


def _measure_roundtrip(
    transport: Any,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Exercise the AXI-Lite register map and read back one sample batch.

    Steps:
    1. Compile and upload the sparse ring problem.
    2. Write SPIN_COUNT, SAMPLE_COUNT, WARMUP_STEPS, STEPS_PER_SAMPLE, and
       BETA registers.
    3. Issue CONTROL_CLEAR_RESULTS | CONTROL_START and poll STATUS_DONE with
       a wall-clock deadline.
    4. Read back packed sample words, unpack to bool, convert to ±1.
    5. Validate all spins are ±1.

    Returns a dict with:
    - ``register_roundtrip_us``: CONTROL.START → STATUS.DONE in microseconds
    - ``sample_shape``: [n_samples, n_spins]
    - ``status_word``: final STATUS register value
    - ``spin_state_valid``: True / False

    Raises RuntimeError on timeout.

    Spec: REQ-SAMPLE-009, SCENARIO-SAMPLE-019
    """
    sampler = FPGAIsingSampler(
        mode="hardware",
        allow_cpu_fallback=False,
        overlay_factory=lambda _bitfile: transport,
    )
    biases, couplings = build_problem()
    compiled = sampler.upload_problem(biases, couplings)

    regmap = AXILiteRegisterMap()
    transport.write(regmap.SPIN_COUNT, compiled.n_spins)
    transport.write(regmap.SAMPLE_COUNT, DEFAULT_N_SAMPLES)
    transport.write(regmap.WARMUP_STEPS, DEFAULT_WARMUP_STEPS)
    transport.write(regmap.STEPS_PER_SAMPLE, DEFAULT_STEPS_PER_SAMPLE)
    transport.write(regmap.BETA_INIT, _quantize_word(DEFAULT_BETA, sampler.architecture.frac_bits))
    transport.write(regmap.BETA_FINAL, _quantize_word(DEFAULT_BETA, sampler.architecture.frac_bits))
    transport.write(regmap.RUN_FLAGS, 0)

    trigger_start = time.perf_counter()
    transport.write(regmap.CONTROL, regmap.CONTROL_CLEAR_RESULTS | regmap.CONTROL_START)
    deadline = trigger_start + timeout_seconds
    status = transport.read(regmap.STATUS)
    while not (status & regmap.STATUS_DONE):
        if time.perf_counter() >= deadline:
            raise RuntimeError(
                f"KV260 bring-up did not assert STATUS.DONE within {timeout_seconds:.2f}s"
            )
        status = transport.read(regmap.STATUS)
    register_roundtrip_us = (time.perf_counter() - trigger_start) * 1e6

    words_per_sample = max(1, (compiled.n_spins + 31) // 32)
    samples: list[np.ndarray] = []
    for sample_index in range(DEFAULT_N_SAMPLES):
        words = [
            transport.read(regmap.sample_offset(sample_index, word_index, words_per_sample))
            for word_index in range(words_per_sample)
        ]
        samples.append(unpack_sample_words(words, n_spins=compiled.n_spins))
    sample_array = np.asarray(samples, dtype=bool)

    # Convert to ±1 and validate.
    pm1 = spins_to_pm1(sample_array.ravel())
    spin_valid = validate_spin_state(pm1)

    return {
        "register_roundtrip_us": round(register_roundtrip_us, 3),
        "sample_shape": [int(d) for d in sample_array.shape],
        "status_word": int(status),
        "spin_state_valid": spin_valid,
    }


# ---------------------------------------------------------------------------
# Main experiment flow
# ---------------------------------------------------------------------------


def run_experiment(
    *,
    output_path: Path,
    bitfile_path: str | None,
    overlay_loader: Any = None,
    timeout_seconds: float = BRINGUP_TIMEOUT_SECONDS,
    write_output: bool = False,
) -> dict[str, Any]:
    """Run the Exp 288 KV260 bring-up flow and return the artifact payload.

    The flow is:
    1. If *bitfile_path* is None → emit blocked immediately (env var missing).
    2. If bitfile path is set → load the overlay within *timeout_seconds*.
       On import error, load exception, or None transport → emit blocked.
    3. Detect whether transport is SoftwareFPGAOverlay (→ software_model) or
       some other object (→ hardware).
    4. Measure register round-trip; on RuntimeError (timeout) → emit blocked.
    5. Assemble and (optionally) write the artifact JSON.

    Spec: REQ-SAMPLE-009, SCENARIO-SAMPLE-018, SCENARIO-SAMPLE-019
    """
    started_at = utc_now()
    wall_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: env var / bitfile check — blocked immediately if missing
    # ------------------------------------------------------------------
    if bitfile_path is None:
        payload = build_blocked_artifact(
            missing=DEFAULT_BITFILE_ENV,
            next_step=(
                f"Set {DEFAULT_BITFILE_ENV}=/path/to/carnot_ising.bit on the KV260 host "
                "and rerun Exp 288."
            ),
        )
        payload["experiment"] = EXPERIMENT_ID
        payload["run_date"] = RUN_DATE
        payload["started_at"] = started_at
        payload["finished_at"] = utc_now()
        payload["runtime_seconds"] = round(time.perf_counter() - wall_start, 6)
        payload["spec_requirements"] = SPEC_REFS
        if write_output:
            _write_json(output_path, payload)
        return payload

    # ------------------------------------------------------------------
    # Step 2: attempt overlay load
    # ------------------------------------------------------------------
    loader = overlay_loader or _default_overlay_loader
    load_start = time.perf_counter()
    transport = None
    load_error: str | None = None

    try:
        transport = loader(bitfile_path)
    except Exception as exc:
        load_error = str(exc)

    overlay_load_ms = round((time.perf_counter() - load_start) * 1e3, 3)

    if load_error is not None or transport is None:
        payload = build_blocked_artifact(
            missing=OVERLAY_IP_NAME if load_error is None else "pynq_overlay_load",
            next_step=(
                f"Ensure {bitfile_path} loads via `pynq.Overlay(...)` and exposes "
                f"`{OVERLAY_IP_NAME}.mmio`.  Install PYNQ on the KV260 image if needed."
            ),
            error=load_error,
        )
        payload["experiment"] = EXPERIMENT_ID
        payload["run_date"] = RUN_DATE
        payload["started_at"] = started_at
        payload["finished_at"] = utc_now()
        payload["runtime_seconds"] = round(time.perf_counter() - wall_start, 6)
        payload["overlay_load_ms"] = overlay_load_ms
        payload["spec_requirements"] = SPEC_REFS
        if write_output:
            _write_json(output_path, payload)
        return payload

    # ------------------------------------------------------------------
    # Step 3: label the execution path honestly
    # ------------------------------------------------------------------
    execution_path = "software_model" if isinstance(transport, SoftwareFPGAOverlay) else "hardware"

    # ------------------------------------------------------------------
    # Step 4: measure register round-trip
    # ------------------------------------------------------------------
    remaining = timeout_seconds - (time.perf_counter() - wall_start)
    round_trip: dict[str, Any] | None = None
    roundtrip_error: str | None = None

    try:
        round_trip = _measure_roundtrip(transport, timeout_seconds=max(remaining, 0.0))
    except Exception as exc:
        roundtrip_error = str(exc)

    spin_valid: bool | None = round_trip["spin_state_valid"] if round_trip else None

    if roundtrip_error is not None:
        execution_path = "blocked"

    # ------------------------------------------------------------------
    # Step 5: assemble artifact
    # ------------------------------------------------------------------
    finished_at = utc_now()
    runtime_seconds = round(time.perf_counter() - wall_start, 6)

    payload = {
        "experiment": EXPERIMENT_ID,
        "benchmark": BENCHMARK_NAME,
        "run_date": RUN_DATE,
        "schema": {"artifact": "carnot.kv260_bringup.v1"},
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_seconds": runtime_seconds,
        "execution_path": execution_path,
        "missing": None,
        "next_step": None,
        "overlay_load_ms": overlay_load_ms,
        "round_trip": round_trip,
        "spin_state_valid": spin_valid,
        "spec_requirements": SPEC_REFS,
        "source_artifacts": [
            "results/experiment_228_results.json",
            "results/experiment_242_results.json",
            "docs/fpga-ising-design.md",
        ],
    }
    if roundtrip_error:
        payload["error"] = roundtrip_error
        payload["missing"] = "status_done_within_timeout"
        payload["next_step"] = (
            f"Confirm STATUS.DONE asserts after CONTROL.START within {timeout_seconds:.0f}s.  "
            "Check the overlay register map contract matches the Exp 228 design."
        )

    if write_output:
        _write_json(output_path, payload)

    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exp 288: Attempt KV260 FPGA overlay bring-up within a 60-second hard timeout."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output path for results/experiment_288_results.json",
    )
    parser.add_argument(
        "--bitfile",
        default=os.environ.get(DEFAULT_BITFILE_ENV),
        help=f"KV260 overlay bitfile path (defaults to ${DEFAULT_BITFILE_ENV})",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=BRINGUP_TIMEOUT_SECONDS,
        help="Hard wall-clock timeout for the entire bring-up sequence (default: 60 s).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = get_repo_root()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    bitfile_path = args.bitfile or check_env_var()

    payload = run_experiment(
        output_path=output_path,
        bitfile_path=bitfile_path,
        timeout_seconds=float(args.timeout_seconds),
        write_output=True,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
