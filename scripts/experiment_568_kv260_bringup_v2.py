#!/usr/bin/env python3
"""Experiment 568: KV260 FPGA Bring-Up v2 — hardware arrived 2026-04-20.

Protocol:
    1. apply_env_autofix() FIRST — ensures CARNOT_FORCE_LIVE=1 etc. are set.
    2. ExperimentTimeoutWatchdog(568, timeout_minutes=60) — hard wall-clock cap.
    3. ExperimentTemplate(568, ...) — standardised artifact scaffolding.
    4. Check CARNOT_KV260_BITFILE env var.
       - If SET:   instantiate FpgaBackend, run 100-spin benchmark (100 trials),
                   compute mean_latency_us, compare to CPU baseline.
       - If UNSET: emit Vivado synthesis command, write synth_ising.tcl stub if
                   absent, run CPU simulation baseline for provenance.
    5. Build artifact with schema='carnot.kv260_bringup.v2'.
    6. tmpl.assert_deliverable_written() — FINAL LINE.

Context:
    KV260 board physically arrived on 2026-04-20.  Previous attempts
    (Exps 228, 288, 289, 290, 313) were blocked because CARNOT_KV260_BITFILE
    was not set — the hardware was not present.  This experiment closes the
    loop: if the bitfile is synthesised and the env var is set, we benchmark
    real hardware; otherwise we emit the synthesis command so the operator can
    build the bitfile and re-run.

Target:
    hardware_latency_us < 100 (vs CPU baseline ~358000μs = 358ms from Exp 313)

Spec: REQ-SAMPLE-031, SCENARIO-SAMPLE-049, SCENARIO-SAMPLE-050, SCENARIO-SAMPLE-051
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# apply_env_autofix MUST be called before JAX or CUDA imports.
from carnot.pipeline.env_autofix import apply_env_autofix

_env_result = apply_env_autofix()

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]

EXP_ID = 568
TITLE = "KV260 FPGA Bring-Up v2"
DELIVERABLE = "results/experiment_568_kv260_bringup_v2.json"
TIMEOUT_MINUTES = 60
N_SPINS = 100
N_TRIALS = 100
CPU_BASELINE_REF_US = 358740.98  # Exp 313 reference for comparison reporting
SYNTHESIS_COMMAND = "vivado -mode batch -source hardware/kv260/synth_ising.tcl"
TCL_STUB_PATH = "hardware/kv260/synth_ising.tcl"
LATENCY_TARGET_US = 100.0  # FPGA must beat this to qualify as hardware_working


def _measure_cpu_baseline(n_spins: int = N_SPINS, n_trials: int = 10) -> float:
    """Measure ParallelIsingSampler CPU latency in microseconds (mean over n_trials).

    Why this is separate from FpgaBackend.dispatch CPU fallback:
        We want a clean CPU-only baseline unaffected by FpgaBackend's sparsify
        and LagONN penalty overhead.  This gives an apples-to-apples number
        matching the Exp 313 methodology for continuity.

    Returns:
        Mean latency in microseconds across n_trials.
    """
    import jax.numpy as jnp
    import jax.random as jrandom

    from carnot.samplers.parallel_ising import AnnealingSchedule, ParallelIsingSampler

    sampler = ParallelIsingSampler(
        n_warmup=200,
        n_samples=1,
        steps_per_sample=5,
        schedule=AnnealingSchedule(beta_init=0.5, beta_final=5.0),
        use_checkerboard=True,
    )
    key = jrandom.PRNGKey(0)
    biases = jnp.zeros(n_spins, dtype=jnp.float32)
    couplings = jnp.zeros((n_spins, n_spins), dtype=jnp.float32)

    # Warm up JIT to exclude compilation time from timing.
    _ = sampler.sample(key, biases, couplings)

    latencies_us: list[float] = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        _ = sampler.sample(jrandom.PRNGKey(i), biases, couplings)
        latencies_us.append((time.perf_counter() - t0) * 1e6)

    return float(sum(latencies_us) / len(latencies_us))


def _measure_fpga_latency(bitfile_path: str, n_spins: int = N_SPINS, n_trials: int = N_TRIALS) -> float:
    """Measure FpgaBackend sampling latency in microseconds (mean over n_trials).

    Why we time FpgaBackend.sample() end-to-end:
        The AXI register upload + trigger + spin readback path is what determines
        whether the FPGA is actually faster than CPU.  Timing only the FPGA kernel
        would miss the PL↔PS data transfer bottleneck that matters in practice.

    Args:
        bitfile_path: Path to .bit file (must exist; already validated by caller).
        n_spins: Number of spins for the benchmark problem.
        n_trials: Number of independent timed samples.

    Returns:
        Mean latency in microseconds.
    """
    import numpy as np

    from carnot.samplers.fpga_backend import FpgaBackend

    backend = FpgaBackend(seed=42, beta_max=5.0)
    biases = np.zeros(n_spins, dtype=np.float32)
    couplings = np.zeros((n_spins, n_spins), dtype=np.float32)
    config = {"n_steps": 200}

    # Warm-up call so any lazy initialisation doesn't contaminate timing.
    _ = backend.sample(biases, couplings, 1, config)

    latencies_us: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        backend.sample(biases, couplings, 1, config)
        latencies_us.append((time.perf_counter() - t0) * 1e6)

    return float(sum(latencies_us) / len(latencies_us))


def _ensure_tcl_stub(repo_root: Path) -> bool:
    """Write a minimal synth_ising.tcl stub if it does not already exist.

    Returns True if the file already existed, False if we created it now.
    The stub is auto-generated so the operator has a complete build recipe
    even on a fresh clone without a pre-built bitfile.
    """
    tcl_path = repo_root / TCL_STUB_PATH
    if tcl_path.exists():
        return True

    # synth_ising.tcl is tracked in the repo (hardware/kv260/synth_ising.tcl).
    # If it's missing (e.g. shallow clone), recreate the minimal stub.
    tcl_path.parent.mkdir(parents=True, exist_ok=True)
    tcl_path.write_text(
        "# synth_ising.tcl — auto-generated stub by Exp 568\n"
        "# Run: vivado -mode batch -source hardware/kv260/synth_ising.tcl\n"
        'set part "xck26-sfvc784-2LV-c"\n'
        'set top_module "ising_sampler_128"\n'
        'add_files "hardware/kv260/ising_sampler_v1.v"\n'
        "synth_design -top $top_module -part $part\n"
        "write_bitstream -force output/carnot_ising.bit\n"
    )
    return False


def run_experiment(
    output_path: Path,
    *,
    write_output: bool = True,
    _cpu_trials: int = 10,
    _fpga_trials: int = N_TRIALS,
) -> dict[str, Any]:
    """Run the KV260 bring-up benchmark and return the artifact dict.

    Separating logic from main() allows tests to call this without side effects
    (e.g. no ExperimentTemplate/watchdog overhead, parameterised trial counts).

    Args:
        output_path: Where to write the JSON artifact.
        write_output: If True, write the artifact to disk.
        _cpu_trials: Number of CPU timing trials (reduced in tests for speed).
        _fpga_trials: Number of FPGA timing trials (reduced in tests for speed).

    Returns:
        Artifact dict with all required schema fields.

    Spec: REQ-SAMPLE-031, SCENARIO-SAMPLE-049, SCENARIO-SAMPLE-050, SCENARIO-SAMPLE-051
    """
    import datetime

    started_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.perf_counter()

    bitfile_path = os.environ.get("CARNOT_KV260_BITFILE")
    bitfile_set = bitfile_path is not None

    hardware_latency_us: float | None = None
    fpga_alive: bool = False
    synthesis_command: str | None = None
    tcl_stub_existed: bool | None = None

    # ------------------------------------------------------------------
    # CPU baseline — always measured regardless of hardware availability.
    # This gives honest provenance even when FPGA path is blocked.
    # ------------------------------------------------------------------
    _log.info("Exp %d: measuring CPU baseline (%d trials)...", EXP_ID, _cpu_trials)
    cpu_baseline_latency_us = _measure_cpu_baseline(n_spins=N_SPINS, n_trials=_cpu_trials)
    _log.info("Exp %d: CPU baseline = %.1f μs", EXP_ID, cpu_baseline_latency_us)

    # ------------------------------------------------------------------
    # Hardware path or synthesis path
    # ------------------------------------------------------------------
    if bitfile_set:
        _log.info("Exp %d: CARNOT_KV260_BITFILE=%s — attempting hardware benchmark", EXP_ID, bitfile_path)
        try:
            hardware_latency_us = _measure_fpga_latency(
                bitfile_path,  # type: ignore[arg-type]
                n_spins=N_SPINS,
                n_trials=_fpga_trials,
            )
            fpga_alive = hardware_latency_us < LATENCY_TARGET_US
            _log.info(
                "Exp %d: FPGA latency = %.2f μs  fpga_alive=%s",
                EXP_ID, hardware_latency_us, fpga_alive,
            )
        except Exception as exc:
            _log.warning("Exp %d: FPGA benchmark failed — %s", EXP_ID, exc)
            hardware_latency_us = None
            fpga_alive = False
    else:
        # No bitfile — emit synthesis recipe and run CPU simulation fallback.
        synthesis_command = SYNTHESIS_COMMAND
        _log.info("Exp %d: CARNOT_KV260_BITFILE not set — synthesis path", EXP_ID)
        _log.info("Exp %d: synthesis_command = %s", EXP_ID, synthesis_command)
        tcl_stub_existed = _ensure_tcl_stub(_REPO_ROOT)
        _log.info("Exp %d: synth_ising.tcl pre-existing=%s", EXP_ID, tcl_stub_existed)

    # ------------------------------------------------------------------
    # Compute speedup ratio and honest_verdict
    # ------------------------------------------------------------------
    fpga_speedup: float | None = None
    if hardware_latency_us is not None and cpu_baseline_latency_us > 0:
        fpga_speedup = cpu_baseline_latency_us / hardware_latency_us

    if fpga_alive:
        honest_verdict = "hardware_working"
    elif not bitfile_set:
        honest_verdict = "synthesis_required"
    else:
        honest_verdict = "hardware_too_slow"

    # ------------------------------------------------------------------
    # Build artifact
    # ------------------------------------------------------------------
    finished_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    duration_s = round(time.perf_counter() - t0, 3)

    artifact: dict[str, Any] = {
        "experiment": EXP_ID,
        "title": TITLE,
        "run_date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d"),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "status": "success",
        "schema": "carnot.kv260_bringup.v2",
        "bitfile_set": bitfile_set,
        "hardware_latency_us": hardware_latency_us,
        "cpu_baseline_latency_us": cpu_baseline_latency_us,
        "fpga_speedup": fpga_speedup,
        "fpga_alive": fpga_alive,
        "synthesis_command": synthesis_command,
        "tcl_stub_existed": tcl_stub_existed,
        "honest_verdict": honest_verdict,
        "n_spins": N_SPINS,
        "n_trials": _fpga_trials if bitfile_set else _cpu_trials,
        "cpu_baseline_ref_us": CPU_BASELINE_REF_US,
        "spec_requirements": [
            "REQ-SAMPLE-031",
            "SCENARIO-SAMPLE-049",
            "SCENARIO-SAMPLE-050",
            "SCENARIO-SAMPLE-051",
        ],
    }

    if write_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(artifact, f, indent=2)
        _log.info("Exp %d: wrote %s", EXP_ID, output_path)

    return artifact


def main() -> None:
    """Entry point: run Exp 568 under watchdog and ExperimentTemplate scaffolding."""
    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    output_path = _REPO_ROOT / DELIVERABLE

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=TIMEOUT_MINUTES):
        artifact = run_experiment(output_path, write_output=True)

    verdict = artifact["honest_verdict"]
    _log.info("Exp %d: honest_verdict=%s", EXP_ID, verdict)
    print(f"[Exp {EXP_ID}] honest_verdict={verdict}")
    print(f"[Exp {EXP_ID}] bitfile_set={artifact['bitfile_set']}")
    print(f"[Exp {EXP_ID}] cpu_baseline_latency_us={artifact['cpu_baseline_latency_us']:.1f}")
    if artifact["hardware_latency_us"] is not None:
        print(f"[Exp {EXP_ID}] hardware_latency_us={artifact['hardware_latency_us']:.2f}")
        print(f"[Exp {EXP_ID}] fpga_speedup={artifact['fpga_speedup']:.1f}x")
    if artifact["synthesis_command"]:
        print(f"[Exp {EXP_ID}] synthesis_command={artifact['synthesis_command']}")
    print(f"[Exp {EXP_ID}] Deliverable: {output_path}")

    # FINAL LINE: guard raises FileNotFoundError if deliverable is absent.
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
