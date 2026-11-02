#!/usr/bin/env python3
"""Experiment 242: KV260 host / overlay round-trip validation.

Writes:
- ``results/experiment_242_results.json``

Spec: REQ-SAMPLE-007,
SCENARIO-SAMPLE-012, SCENARIO-SAMPLE-013, SCENARIO-SAMPLE-014
"""

from __future__ import annotations

import argparse
import importlib
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
    RegisterIO,
    SoftwareFPGAOverlay,
    _quantize_word,
    unpack_sample_words,
)

EXPERIMENT_ID = 242
RUN_DATE = "20260413"
BENCHMARK_NAME = "kv260_host_overlay_roundtrip"
DEFAULT_OUTPUT = Path("results/experiment_242_results.json")
DEFAULT_BITFILE_ENV = "CARNOT_KV260_BITFILE"
OVERLAY_IP_NAME = "carnot_ising_0"
DEFAULT_N_SPINS = 128
DEFAULT_SAMPLE_COUNT = 4
DEFAULT_WARMUP_STEPS = 40
DEFAULT_STEPS_PER_SAMPLE = 10
DEFAULT_BETA = 6.0
DEFAULT_TIMEOUT_SECONDS = 5.0

SPEC_REFS = [
    "REQ-SAMPLE-005",
    "REQ-SAMPLE-006",
    "REQ-SAMPLE-007",
    "SCENARIO-SAMPLE-012",
    "SCENARIO-SAMPLE-013",
    "SCENARIO-SAMPLE-014",
]


class BoundMMIO:
    """Keep the parent overlay alive while exposing the MMIO interface."""

    def __init__(self, overlay: Any, mmio: RegisterIO) -> None:
        self._overlay = overlay
        self._mmio = mmio

    def write(self, offset: int, value: int) -> None:
        self._mmio.write(offset, value)

    def read(self, offset: int) -> int:
        return int(self._mmio.read(offset))


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_repo_root() -> Path:
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attempt a KV260 FPGA host/overlay round trip and record honest timings.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Relative or absolute path for results/experiment_242_results.json",
    )
    parser.add_argument(
        "--bitfile",
        default=os.environ.get(DEFAULT_BITFILE_ENV),
        help=f"KV260 overlay bitfile path (defaults to ${DEFAULT_BITFILE_ENV})",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Maximum time to wait for STATUS.DONE before blocking the run.",
    )
    return parser


def default_overlay_loader(bitfile_path: Path) -> RegisterIO | None:
    """Load the real PYNQ overlay transport for the KV260 bitfile."""
    pynq = importlib.import_module("pynq")
    overlay = pynq.Overlay(str(bitfile_path), download=True)
    mmio = getattr(getattr(overlay, OVERLAY_IP_NAME, None), "mmio", None)
    if mmio is None:
        return None
    return BoundMMIO(overlay, mmio)


def build_problem(n_spins: int = DEFAULT_N_SPINS) -> tuple[np.ndarray, np.ndarray]:
    """Build the sparse ring problem used for the Exp 242 round trip."""
    biases = np.full(n_spins, 0.2, dtype=np.float32)
    couplings = np.zeros((n_spins, n_spins), dtype=np.float32)
    for row in range(n_spins):
        for distance in range(1, 5):
            neighbor = (row + distance) % n_spins
            couplings[row, neighbor] = 0.35
            couplings[neighbor, row] = 0.35
    return biases, couplings


def build_blocker(
    *,
    code: str,
    stage: str,
    message: str,
    setup_step: str,
    bitfile_path: Path | None,
    error: str | None = None,
) -> dict[str, Any]:
    blocker = {
        "code": code,
        "stage": stage,
        "message": message,
        "setup_step": setup_step,
        "overlay_path": str(bitfile_path) if bitfile_path is not None else None,
    }
    if error is not None:
        blocker["error"] = error
    return blocker


def load_transport(
    bitfile_path: Path | None,
    overlay_loader: Any | None = None,
) -> tuple[str, RegisterIO | None, list[dict[str, Any]]]:
    """Resolve the requested hardware transport or emit a blocker."""
    if bitfile_path is None:
        return (
            "blocked",
            None,
            [
                build_blocker(
                    code="missing_bitfile_config",
                    stage="overlay_config",
                    message="No KV260 overlay bitfile path was configured.",
                    setup_step=(
                        f"Set {DEFAULT_BITFILE_ENV}=/path/to/carnot_ising.bit on the KV260 host "
                        "before rerunning Exp 242."
                    ),
                    bitfile_path=None,
                )
            ],
        )

    if not bitfile_path.exists():
        return (
            "blocked",
            None,
            [
                build_blocker(
                    code="bitfile_not_found",
                    stage="overlay_config",
                    message=f"Configured KV260 bitfile was not found: {bitfile_path}",
                    setup_step=(
                        f"Copy or build the KV260 bitfile at {bitfile_path} and rerun Exp 242."
                    ),
                    bitfile_path=bitfile_path,
                )
            ],
        )

    loader = overlay_loader or default_overlay_loader
    try:
        transport = loader(bitfile_path)
    except Exception as error:
        return (
            "blocked",
            None,
            [
                build_blocker(
                    code="overlay_load_failed",
                    stage="overlay_load",
                    message="Loading the KV260 overlay failed.",
                    setup_step=(
                        "Install PYNQ on the KV260 image and verify the bitfile loads through "
                        "`pynq.Overlay(...)`."
                    ),
                    bitfile_path=bitfile_path,
                    error=str(error),
                )
            ],
        )

    if transport is None:
        return (
            "blocked",
            None,
            [
                build_blocker(
                    code="missing_mmio_endpoint",
                    stage="overlay_endpoint",
                    message=f"Overlay does not expose {OVERLAY_IP_NAME}.mmio.",
                    setup_step=(
                        f"Expose `{OVERLAY_IP_NAME}.mmio` from the KV260 overlay built from "
                        "the Exp 228 register-map contract."
                    ),
                    bitfile_path=bitfile_path,
                )
            ],
        )

    execution_path = "software_model" if isinstance(transport, SoftwareFPGAOverlay) else "hardware"
    return execution_path, transport, []


def probe_auto_backend(
    bitfile_path: Path | None,
    auto_overlay_factory: Any | None = None,
) -> dict[str, Any]:
    """Record how FPGAIsingSampler(mode=\"auto\") behaves in this environment."""
    sampler = FPGAIsingSampler(
        mode="auto",
        bitfile_path=str(bitfile_path) if bitfile_path is not None else None,
        overlay_factory=auto_overlay_factory,
    )
    return {
        "backend_name": sampler.backend_name,
        "using_cpu_fallback": sampler.using_cpu_fallback,
    }


def measure_roundtrip(
    transport: RegisterIO,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Measure upload, trigger, and readback latencies on the active transport."""
    sampler = FPGAIsingSampler(
        mode="hardware",
        allow_cpu_fallback=False,
        overlay_factory=lambda _bitfile: transport,
    )
    biases, couplings = build_problem()

    upload_start = time.perf_counter()
    compiled = sampler.upload_problem(biases, couplings)
    upload_seconds = time.perf_counter() - upload_start

    regmap = AXILiteRegisterMap()
    transport.write(regmap.SPIN_COUNT, compiled.n_spins)
    transport.write(regmap.SAMPLE_COUNT, DEFAULT_SAMPLE_COUNT)
    transport.write(regmap.WARMUP_STEPS, DEFAULT_WARMUP_STEPS)
    transport.write(regmap.STEPS_PER_SAMPLE, DEFAULT_STEPS_PER_SAMPLE)
    transport.write(regmap.BETA_INIT, _quantize_word(DEFAULT_BETA, sampler.architecture.frac_bits))
    transport.write(regmap.BETA_FINAL, _quantize_word(DEFAULT_BETA, sampler.architecture.frac_bits))
    transport.write(regmap.RUN_FLAGS, 0)

    trigger_start = time.perf_counter()
    transport.write(regmap.CONTROL, regmap.CONTROL_CLEAR_RESULTS | regmap.CONTROL_START)
    deadline = time.perf_counter() + timeout_seconds
    status = transport.read(regmap.STATUS)
    while not status & regmap.STATUS_DONE:
        if time.perf_counter() >= deadline:
            raise RuntimeError(f"KV260 round-trip did not complete within {timeout_seconds:.2f}s")
        status = transport.read(regmap.STATUS)
    trigger_seconds = time.perf_counter() - trigger_start

    words_per_sample = max(1, (compiled.n_spins + 31) // 32)
    readback_start = time.perf_counter()
    samples: list[np.ndarray] = []
    first_sample_words: list[int] = []
    for sample_index in range(DEFAULT_SAMPLE_COUNT):
        words = [
            transport.read(regmap.sample_offset(sample_index, word_index, words_per_sample))
            for word_index in range(words_per_sample)
        ]
        if sample_index == 0:
            first_sample_words = [int(word) for word in words]
        samples.append(unpack_sample_words(words, n_spins=compiled.n_spins))
    readback_seconds = time.perf_counter() - readback_start
    sample_array = np.asarray(samples, dtype=bool)

    return {
        "latency_seconds": {
            "upload": upload_seconds,
            "trigger": trigger_seconds,
            "readback": readback_seconds,
        },
        "sample_shape": [int(dim) for dim in sample_array.shape],
        "status_word": int(status),
        "first_sample_words": first_sample_words,
    }


def build_notes(execution_path: str, blockers: list[dict[str, Any]]) -> list[str]:
    if execution_path == "hardware":
        return [
            "Measured on the active KV260 overlay/MMIO endpoint using the Exp 228 register map.",
            "Upload, trigger, and readback timings are host-to-overlay round-trip measurements.",
        ]
    if execution_path == "software_model":
        return [
            "This run completed against the software-model overlay path, not real KV260 hardware.",
            "The software-model timings validate contract fidelity only; they are not "
            "live throughput numbers.",
        ]
    return [
        "Hardware validation remained blocked, so no latency numbers were fabricated.",
        blockers[0]["message"] if blockers else "KV260 bring-up blockers were encountered.",
    ]


def build_artifact_payload(
    *,
    output_path: Path,
    bitfile_path: Path | None,
    execution_path: str,
    auto_backend_probe: dict[str, Any],
    round_trip: dict[str, Any] | None,
    blockers: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_ID,
        "benchmark": BENCHMARK_NAME,
        "title": "KV260 host / overlay round-trip benchmark",
        "run_date": RUN_DATE,
        "schema": {
            "artifact": "carnot.kv260_roundtrip.v1",
        },
        "metadata": {
            "hardware_target": "AMD/Xilinx Kria KV260",
            "execution_path": execution_path,
            "hardware_detected": execution_path == "hardware",
            "started_at": started_at,
            "finished_at": finished_at,
            "runtime_seconds": round(runtime_seconds, 6),
            "output_path": str(output_path),
            "bitfile_path": str(bitfile_path) if bitfile_path is not None else None,
            "overlay_ip_name": OVERLAY_IP_NAME,
            "module": "python/carnot/samplers/fpga_ising.py",
            "script": "scripts/experiment_242_kv260_roundtrip.py",
            "source_artifacts": [
                "results/experiment_228_results.json",
                "docs/fpga-ising-design.md",
            ],
            "spec_requirements": list(SPEC_REFS),
            "auto_backend_probe": dict(auto_backend_probe),
            "notes": build_notes(execution_path, blockers),
        },
        "problem": {
            "n_spins": DEFAULT_N_SPINS,
            "topology": "symmetric sparse ring with 4 neighbors on each side",
            "n_samples": DEFAULT_SAMPLE_COUNT,
            "warmup_steps": DEFAULT_WARMUP_STEPS,
            "steps_per_sample": DEFAULT_STEPS_PER_SAMPLE,
            "beta": DEFAULT_BETA,
            "seed": EXPERIMENT_ID,
        },
        "bring_up": {
            "required_env_var": DEFAULT_BITFILE_ENV,
            "overlay_path_checked": str(bitfile_path) if bitfile_path is not None else None,
            "checklist": [
                f"Set {DEFAULT_BITFILE_ENV} to the KV260 bitfile path on the board host.",
                f"Load the overlay and expose `{OVERLAY_IP_NAME}.mmio` through PYNQ.",
                "Verify the STATUS register raises DONE after CONTROL.START.",
            ],
        },
        "round_trip": round_trip,
        "blockers": [dict(blocker) for blocker in blockers],
        "run_status": "complete" if not blockers and round_trip is not None else "blocked",
    }


def run_experiment(
    *,
    output_path: Path,
    bitfile_path: Path | str | None,
    overlay_loader: Any | None = None,
    auto_overlay_factory: Any | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run the Exp 242 bring-up flow and return the artifact payload."""
    started_at = utc_now()
    start = time.perf_counter()
    resolved_bitfile = Path(bitfile_path) if bitfile_path is not None else None
    auto_backend_probe = probe_auto_backend(resolved_bitfile, auto_overlay_factory)
    execution_path, transport, blockers = load_transport(resolved_bitfile, overlay_loader)
    round_trip = None

    if transport is not None and not blockers:
        try:
            round_trip = measure_roundtrip(transport, timeout_seconds)
        except Exception as error:
            execution_path = "blocked"
            blockers = [
                build_blocker(
                    code="roundtrip_failed",
                    stage="round_trip",
                    message="The KV260 overlay failed during upload/trigger/readback.",
                    setup_step=(
                        "Confirm the overlay honors the Exp 228 register map and that STATUS.DONE "
                        "asserts after CONTROL.START."
                    ),
                    bitfile_path=resolved_bitfile,
                    error=str(error),
                )
            ]

    return build_artifact_payload(
        output_path=output_path,
        bitfile_path=resolved_bitfile,
        execution_path=execution_path,
        auto_backend_probe=auto_backend_probe,
        round_trip=round_trip,
        blockers=blockers,
        started_at=started_at,
        finished_at=utc_now(),
        runtime_seconds=time.perf_counter() - start,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = get_repo_root()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    payload = run_experiment(
        output_path=output_path,
        bitfile_path=args.bitfile,
        timeout_seconds=float(args.timeout_seconds),
    )
    write_json(output_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
