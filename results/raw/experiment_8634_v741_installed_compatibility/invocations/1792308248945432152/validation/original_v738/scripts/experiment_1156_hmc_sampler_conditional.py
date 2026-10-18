"""Experiment 1156: deploy the sampler recommended by Exp 1155.

Spec coverage: REQ-KONA-010, SCENARIO-KONA-009
"""

from __future__ import annotations

import datetime as _datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.phase3.continuous_ebm import ContinuousEBM  # noqa: E402
from carnot.samplers.phase4_sampler import (  # noqa: E402
    Phase4Sampler,
    continuous_ebm_energy,
)

EXP1155_PATH = REPO_ROOT / "results" / "experiment_1155_hmc_compatibility_diagnostics.json"
RESULT_PATH = REPO_ROOT / "results" / "experiment_1156_hmc_sampler_conditional.json"
SAMPLER_MODULE = "python/carnot/samplers/phase4_sampler.py"
N_VALIDATION_EXAMPLES = 100
N_STEPS = 1000
SEED = 1156


def _load_exp1155(path: Path = EXP1155_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validation_model(latent_dim: int, discrete_indices: tuple[int, ...]) -> ContinuousEBM:
    """Build a small ContinuousEBM landscape for the 100-example validation run."""
    coupling = -0.12 * np.eye(latent_dim, dtype=np.float64)
    for idx in range(latent_dim - 1):
        coupling[idx, idx + 1] = -0.01
        coupling[idx + 1, idx] = -0.01

    bias = np.linspace(-0.08, 0.08, latent_dim, dtype=np.float64)
    for idx in discrete_indices:
        coupling[idx, :] = 0.0
        coupling[:, idx] = 0.0
        bias[idx] = 0.0

    return ContinuousEBM(variables=latent_dim, coupling=coupling, bias=bias)


def _run_validation(
    sampler: Phase4Sampler,
    energy_fn: Any,
    latent_dim: int,
) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    starts = rng.uniform(-1.0, 1.0, size=(N_VALIDATION_EXAMPLES, latent_dim))
    retained: list[np.ndarray] = []
    for start in starts:
        chain = sampler.sample(energy_fn, start, N_STEPS)
        retained.append(chain[N_STEPS // 2 :])
    return np.vstack(retained)


def _boltzmann_reference(
    energy_fn: Any,
    latent_dim: int,
    *,
    n_reference: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED + 1)
    points = rng.uniform(-1.0, 1.0, size=(n_reference, latent_dim))
    energies = np.asarray([energy_fn(point) for point in points], dtype=np.float64)
    log_weights = -energies - float(np.max(-energies))
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    return energies, weights


def _energy_histogram_kl(
    sample_points: np.ndarray,
    energy_fn: Any,
    reference_energies: np.ndarray,
    reference_weights: np.ndarray,
) -> float:
    sample_energies = np.asarray([energy_fn(point) for point in sample_points], dtype=np.float64)
    lo = float(min(np.min(sample_energies), np.min(reference_energies)))
    hi = float(max(np.max(sample_energies), np.max(reference_energies)))
    bins = np.linspace(lo, hi + 1e-9, 33)
    sample_hist, _ = np.histogram(sample_energies, bins=bins)
    ref_hist, _ = np.histogram(reference_energies, bins=bins, weights=reference_weights)

    eps = 1e-12
    p = sample_hist.astype(np.float64) + eps
    q = ref_hist.astype(np.float64) + eps
    p /= np.sum(p)
    q /= np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def _phase4_tests_passing() -> int:
    python_bin = REPO_ROOT / ".venv" / "bin" / "python"
    runner = str(python_bin if python_bin.exists() else Path(sys.executable))
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PYTHON_ROOT}:{REPO_ROOT}:{env.get('PYTHONPATH', '')}"

    subprocess.run(
        [runner, "-m", "coverage", "erase"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    test_cmd = [
        runner,
        "-m",
        "coverage",
        "run",
        "-m",
        "pytest",
        "tests/python/test_phase4_sampler.py",
        "-q",
        "-o",
        "addopts=",
    ]
    test_run = subprocess.run(
        test_cmd,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    coverage_run = subprocess.run(
        [
            runner,
            "-m",
            "coverage",
            "report",
            "--include=*/phase4_sampler.py",
            "--fail-under=100",
            "-m",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if test_run.returncode != 0 or coverage_run.returncode != 0:
        return 0
    match = re.search(r"(\d+) passed", test_run.stdout)
    return int(match.group(1)) if match else 0


def _blocked_artifact(reason: str) -> dict[str, Any]:
    return {
        "schema": "carnot.phase4_sampler_conditional.v1",
        "experiment": 1156,
        "run_date": _datetime.date.today().isoformat(),
        "hmc_regime_used": "",
        "sampler_algorithm": "",
        "sampler_module": SAMPLER_MODULE,
        "sampler_written": (REPO_ROOT / SAMPLER_MODULE).exists(),
        "n_validation_examples": 0,
        "acceptance_rate": None,
        "kl_divergence_vs_boltzmann": float("inf"),
        "n_tests_passing": 0,
        "active_inference_sampler_ready": False,
        "hmc_sampler_honest_result": True,
        "honest_verdict": "pipeline_not_found_blocked",
        "blocked_reason": reason,
    }


def main() -> dict[str, Any]:
    if not EXP1155_PATH.exists():
        artifact = _blocked_artifact(f"Missing {EXP1155_PATH}")
        RESULT_PATH.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
        return artifact

    exp1155 = _load_exp1155()
    sampler = Phase4Sampler.from_exp1155(
        EXP1155_PATH,
        seed=SEED,
        step_size=0.01,
        temperature=1.0,
    )
    latent_dim = int(exp1155.get("latent_dim", 10))
    model = _validation_model(latent_dim, sampler.discrete_indices)
    energy_fn = continuous_ebm_energy(model)

    samples = _run_validation(sampler, energy_fn, latent_dim)
    reference_energies, reference_weights = _boltzmann_reference(energy_fn, latent_dim)
    kl_divergence = _energy_histogram_kl(samples, energy_fn, reference_energies, reference_weights)
    n_tests_passing = _phase4_tests_passing()
    sampler_written = (REPO_ROOT / SAMPLER_MODULE).exists()
    ready = bool(kl_divergence < 0.5 and sampler_written)
    verdict = "sampler_kl_below_05_viable" if ready else "sampler_kl_above_05_needs_tuning"

    artifact: dict[str, Any] = {
        "schema": "carnot.phase4_sampler_conditional.v1",
        "experiment": 1156,
        "run_date": _datetime.date.today().isoformat(),
        "hmc_regime_used": str(exp1155.get("hmc_regime", "")),
        "sampler_algorithm": sampler.algorithm,
        "sampler_module": SAMPLER_MODULE,
        "sampler_written": sampler_written,
        "n_validation_examples": N_VALIDATION_EXAMPLES,
        "acceptance_rate": sampler.last_diagnostics.get("acceptance_rate"),
        "kl_divergence_vs_boltzmann": kl_divergence,
        "n_tests_passing": n_tests_passing,
        "active_inference_sampler_ready": ready,
        "hmc_sampler_honest_result": True,
        "honest_verdict": verdict,
        "recommended_sampler_from_exp1155": str(exp1155.get("recommended_sampler", "")),
        "d4_discrete_components_bottleneck": bool(
            exp1155.get("d4_discrete_components_bottleneck", False)
        ),
        "n_sampler_steps_per_example": N_STEPS,
    }
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    return artifact


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, allow_nan=False))
