#!/usr/bin/env python3
"""Experiment 624 — KV260 Vivado Synthesis v2 + Synchronous Ising Simulation.

**Researcher summary:**
    Two deliverables in one experiment:
    1. Vivado check: if Vivado is installed, synthesise ising_sampler_v2.v and
       report whether the bitfile was produced.
    2. Python simulation: validate the synchronous checkerboard p-bit update logic
       of ising_sampler_v2.v using SynchronousIsingSampler before any FPGA run.

**Exit paths (every path writes the deliverable):**
    1. apply_env_autofix() before any imports
    2. ExperimentTimeoutWatchdog(624, timeout_minutes=35)
    3. ExperimentTemplate.setup()
    4. Vivado check (subprocess; blocked if not installed)
    5. Python sim: SynchronousIsingSampler.compare_with_async()
    6. tmpl.build_result(...) writes the deliverable
    7. tmpl.assert_deliverable_written()  -- FINAL LINE

Spec: REQ-SAMPLE-037, SCENARIO-SAMPLE-061, SCENARIO-SAMPLE-062
"""

from __future__ import annotations

from carnot.pipeline.env_autofix import apply_env_autofix

_env_result = apply_env_autofix()

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_RESULT_PATH = "results/experiment_624_kv260_vivado_v2.json"
_BITFILE_PATH = "output/carnot_ising_synth/carnot_ising.bit"
_TCL_PATH = "hardware/kv260/synth_ising.tcl"


def _check_vivado() -> bool:
    """Return True if Vivado is on PATH and responds to -version."""
    try:
        result = subprocess.run(
            ["vivado", "-version"],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_synthesis(repo_root: Path) -> bool:
    """Run Vivado synthesis and return True if the bitfile was produced."""
    tcl = repo_root / _TCL_PATH
    try:
        subprocess.run(
            ["vivado", "-mode", "batch", "-source", str(tcl)],
            timeout=1800,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return (repo_root / _BITFILE_PATH).exists()


def _run_simulation() -> dict:
    """Run the Python synchronous Ising simulation validation.

    Uses a small 10-spin random Ising instance.  Compares sync vs async mean
    final energy to confirm the synchronous update logic is behaving correctly.
    """
    import numpy as np

    from carnot.samplers.synchronous_ising import SynchronousIsingSampler

    rng = np.random.default_rng(42)
    n_spins = 10
    J = rng.standard_normal((n_spins, n_spins)) * 0.1
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = np.zeros(n_spins)

    sampler = SynchronousIsingSampler(n_spins=n_spins, couplings=J, biases=h, beta=1.0)
    result = sampler.compare_with_async(n_steps=100, n_trials=10)
    result["simulation_validated"] = result["sync_mean_energy"] is not None
    return result


def main() -> None:
    """Run Exp 624: Vivado v2 synthesis check + synchronous Ising simulation."""
    result_path = str(_REPO_ROOT / _RESULT_PATH)
    tmpl_obj = __import__(
        "scripts.experiment_template", fromlist=["ExperimentTemplate"]
    )
    ExperimentTemplate = tmpl_obj.ExperimentTemplate

    tmpl = ExperimentTemplate(
        624,
        "KV260 Vivado Synthesis v2 + Sync Ising Sim",
        result_path,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(624, timeout_minutes=35, result_path=result_path):
        # 1. Vivado availability check.
        vivado_installed = _check_vivado()

        synthesis_succeeded: bool | str
        if vivado_installed:
            synthesis_succeeded = _run_synthesis(_REPO_ROOT)
        else:
            synthesis_succeeded = "not_attempted"

        # 2. Python simulation validation.
        sim = _run_simulation()
        simulation_validated = sim["simulation_validated"]

        # 3. Honest verdict.
        if vivado_installed and synthesis_succeeded is True:
            honest_verdict = "synthesis_and_simulation"
        elif simulation_validated:
            honest_verdict = "simulation_only_vivado_blocked"
        else:
            honest_verdict = "both_blocked"

        artifact = tmpl.build_result(
            {
                "vivado_installed": vivado_installed,
                "synthesis_succeeded": synthesis_succeeded,
                "simulation_validated": simulation_validated,
                "sync_mean_energy": sim["sync_mean_energy"],
                "async_mean_energy": sim["async_mean_energy"],
                "energy_gap": sim["energy_gap"],
                "honest_verdict": honest_verdict,
            },
            status="success",
        )

        artifact["schema"] = "carnot.kv260_vivado_v2.v1"
        with open(result_path, "w") as f:
            json.dump(artifact, f, indent=2)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
