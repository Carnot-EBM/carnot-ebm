#!/usr/bin/env python3
"""Experiment 1098: q=3 Potts sampler Python simulation plus KV260 RTL.

The deliverable for this experiment is not a verifier-accuracy result.  It is
the implementation artifact that Exp 534 did not produce: a CPU reference
sampler, a q=3 RTL module, and a JSON result saying whether the simulator and
RTL are present and validated.

Spec: REQ-POTTS-001, REQ-POTTS-002, REQ-POTTS-003, REQ-POTTS-004,
      REQ-POTTS-005
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from python.carnot.samplers.potts_sampler import PottsSampler  # noqa: E402

RESULT_PATH = _REPO_ROOT / "results" / "experiment_1098_potts_machine_q3_verilog.json"
RTL_PATH = _REPO_ROOT / "hardware" / "kv260" / "potts_sampler_v1.v"
PYTEST_PATH = _REPO_ROOT / "tests" / "python" / "test_potts_sampler.py"
KV260_LUT_BUDGET = 117_000


def _complete_ferromagnetic_j(n_spins: int, weight: float = 1.0) -> np.ndarray:
    j_matrix = np.full((n_spins, n_spins), weight, dtype=np.float64)
    np.fill_diagonal(j_matrix, 0.0)
    return j_matrix


def validate_python_simulation() -> dict[str, Any]:
    """Validate the two requested simulator properties on 16 spins."""
    np.random.seed(1098)
    n_spins = 16
    sampler = PottsSampler(n_spins=n_spins, q=3, beta=3.0)
    j_matrix = _complete_ferromagnetic_j(n_spins)

    initial_energies: list[float] = []
    final_energies: list[float] = []
    for _ in range(30):
        init = np.random.randint(0, 3, size=n_spins)
        initial_energies.append(sampler.energy(j_matrix, init))
        final = sampler.sample(j_matrix, n_steps=80, init_state=init)
        final_energies.append(sampler.energy(j_matrix, final))

    mean_initial = float(np.mean(initial_energies))
    mean_final = float(np.mean(final_energies))
    energy_nonincreasing = mean_final <= mean_initial

    zero_j = np.zeros((n_spins, n_spins), dtype=np.float64)
    states_seen: set[int] = set()
    for _ in range(12):
        final = sampler.sample(zero_j, n_steps=5)
        states_seen.update(int(v) for v in np.unique(final))

    distribution_has_three_states = states_seen == {0, 1, 2}
    return {
        "energy_nonincreasing_in_expectation": energy_nonincreasing,
        "mean_initial_energy": mean_initial,
        "mean_final_energy": mean_final,
        "distribution_has_three_states": distribution_has_three_states,
        "states_seen": sorted(states_seen),
    }


def inspect_verilog() -> dict[str, Any]:
    """Check that the RTL file contains the requested structural elements."""
    if not RTL_PATH.exists():
        return {"exists": False, "complete": False, "reasons": ["RTL file missing"]}

    rtl = RTL_PATH.read_text()
    checks = {
        "module": "module potts_sampler_v1" in rtl,
        "n_spins": bool(re.search(r"parameter\s+integer\s+N_SPINS\s*=\s*64", rtl)),
        "q_states": bool(re.search(r"parameter\s+integer\s+Q_STATES\s*=\s*3", rtl)),
        "beta": bool(re.search(r"parameter\s+\[7:0\]\s+BETA_FIXED\s*=\s*8'h40", rtl)),
        "two_bit_spins": "2 bits per spin" in rtl,
        "softmax": "softmax" in rtl.lower(),
        "lfsr2": "lfsr2" in rtl.lower(),
        "axi_lite": "S_AXI_AWADDR" in rtl and "ADDR_CONTROL" in rtl,
    }
    return {
        "exists": True,
        "complete": all(checks.values()),
        "checks": checks,
    }


def estimate_lut_area(n_spins: int = 64, q_states: int = 3, max_degree: int = 32) -> int:
    """Static LUT estimate for the q=3 RTL before Vivado synthesis is available.

    The estimate is intentionally conservative: AXI/control overhead plus one
    replicated local-energy and softmax lane per spin.  It is not a synthesis
    report, but it is enough to answer whether the design is plausibly inside
    the 117K LUT KV260 budget before running Vivado.
    """
    axi_and_control = 2_200
    per_spin_energy = max_degree * q_states * 6
    per_spin_softmax = q_states * 96
    per_spin_state_rng = 48
    return int(
        axi_and_control + n_spins * (per_spin_energy + per_spin_softmax + per_spin_state_rng)
    )


def run_focused_tests() -> dict[str, Any]:
    """Run the requested focused pytest file and count passing tests."""
    pytest_cmd = shutil.which("pytest")
    if pytest_cmd is None:
        venv_pytest = _REPO_ROOT / ".venv" / "bin" / "pytest"
        pytest_cmd = str(venv_pytest) if venv_pytest.exists() else None
    if pytest_cmd is None:
        return {
            "returncode": 127,
            "tests_passing": 0,
            "summary": "pytest not found on PATH and .venv/bin/pytest is missing",
        }

    proc = subprocess.run(
        [pytest_cmd, str(PYTEST_PATH), "-q", "--no-cov"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = proc.stdout + "\n" + proc.stderr
    match = re.search(r"(\d+)\s+passed", output)
    tests_passing = int(match.group(1)) if match else 0
    return {
        "returncode": proc.returncode,
        "tests_passing": tests_passing,
        "summary": "\n".join(output.strip().splitlines()[-8:]),
    }


def run_experiment() -> dict[str, Any]:
    started = datetime.now(UTC)
    t0 = time.perf_counter()

    sim_validation = validate_python_simulation()
    verilog = inspect_verilog()
    lut_estimate = estimate_lut_area()
    test_result = run_focused_tests()

    python_sim_written = (
        _REPO_ROOT / "python" / "carnot" / "samplers" / "potts_sampler.py"
    ).exists()
    python_sim_validated = bool(
        sim_validation["energy_nonincreasing_in_expectation"]
        and sim_validation["distribution_has_three_states"]
    )
    verilog_file_written = bool(verilog["exists"] and verilog["complete"])
    verilog_fits = lut_estimate < KV260_LUT_BUDGET
    tests_passing = int(test_result["tests_passing"])

    if python_sim_validated and verilog_file_written and tests_passing >= 5:
        honest_verdict = "potts_sim_and_rtl_complete"
        status = "success"
    elif python_sim_validated and verilog["exists"]:
        honest_verdict = "potts_sim_only_rtl_stub"
        status = "blocked"
    else:
        honest_verdict = "failed"
        status = "failed"

    finished = datetime.now(UTC)
    artifact: dict[str, Any] = {
        "experiment": 1098,
        "schema": "potts_machine_q3_verilog_v1",
        "run_date": started.date().isoformat(),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_s": round(time.perf_counter() - t0, 3),
        "status": status,
        "title": "Potts Machine q=3 Verilog + Python Simulation",
        "python_sim_written": python_sim_written,
        "python_sim_validated": python_sim_validated,
        "verilog_file_written": verilog_file_written,
        "verilog_synthesis_area_estimate_lut": lut_estimate,
        "verilog_fits_kv260_budget": verilog_fits,
        "tests_passing": tests_passing,
        "honest_verdict": honest_verdict,
        "kv260_lut_budget": KV260_LUT_BUDGET,
        "python_validation": sim_validation,
        "verilog_inspection": verilog,
        "pytest": test_result,
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def main() -> int:
    artifact = run_experiment()
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "honest_verdict": artifact["honest_verdict"],
                "python_sim_validated": artifact["python_sim_validated"],
                "verilog_file_written": artifact["verilog_file_written"],
                "tests_passing": artifact["tests_passing"],
            },
            indent=2,
        )
    )
    return 0 if artifact["honest_verdict"] == "potts_sim_and_rtl_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
