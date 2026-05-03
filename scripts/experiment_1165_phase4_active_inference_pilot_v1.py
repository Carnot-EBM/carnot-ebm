"""Experiment 1165: Phase 4 active-inference pilot.

Spec coverage: REQ-KONA-012, SCENARIO-KONA-012
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.phase3.active_inference_pilot import (  # noqa: E402
    ARC3PuzzleEnv,
    ActiveInferencePilot,
    build_default_k5_ensemble_energies,
    build_experiment_artifact,
    run_phase4_vs_baseline,
    write_experiment_artifact,
)
from carnot.phase3.snap_validity import snap_to_action  # noqa: E402
from carnot.samplers.phase4_sampler import Phase4Sampler  # noqa: E402

EXP1154_PATH = REPO_ROOT / "results" / "experiment_1154_snap_validity_sweep.json"
EXP1155_PATH = REPO_ROOT / "results" / "experiment_1155_hmc_compatibility_diagnostics.json"
EXP1156_PATH = REPO_ROOT / "results" / "experiment_1156_hmc_sampler_conditional.json"
RESULT_PATH = REPO_ROOT / "results" / "experiment_1165_phase4_active_inference_pilot_v1.json"
SEED = 1165


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _blocked_params(exp1154: dict[str, Any], exp1156: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_sweeps": int(exp1156.get("n_sampler_steps_per_example", 1000)),
        "n_blocks": int(exp1154.get("latent_dim", 10)),
        "step_size": 0.01,
    }


def main() -> dict[str, Any]:
    exp1154 = _load_json(EXP1154_PATH)
    exp1155 = _load_json(EXP1155_PATH)
    exp1156 = _load_json(EXP1156_PATH)
    params = _blocked_params(exp1154, exp1156)
    latent_dim = int(params["n_blocks"])

    sampler = Phase4Sampler(
        algorithm=str(exp1156.get("sampler_algorithm", "blocked_gibbs")),
        seed=SEED,
        step_size=float(params["step_size"]),
        temperature=0.25,
        discrete_indices=tuple(range(latent_dim)),
        continuous_indices=(),
        hmc_regime_used=str(exp1155.get("hmc_regime", "C")),
    )
    env = ARC3PuzzleEnv()
    pilot = ActiveInferencePilot(
        build_default_k5_ensemble_energies(),
        snap_to_action,
        sampler,
        latent_dim=latent_dim,
        rng_seed=SEED,
    )
    summary = run_phase4_vs_baseline(
        pilot,
        env,
        n_episodes=5,
        max_actions=50,
        n_gibbs_sweeps=int(params["n_sweeps"]),
        baseline_seed=SEED,
    )
    artifact = build_experiment_artifact(summary, blocked_gibbs_params=params)
    artifact.update(
        {
            "snap_validity_verdict": str(exp1154.get("honest_verdict", "")),
            "hmc_regime_used": str(exp1155.get("hmc_regime", "")),
            "sampler_honest_verdict": str(exp1156.get("honest_verdict", "")),
        }
    )
    write_experiment_artifact(artifact, RESULT_PATH)
    return artifact


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, allow_nan=False))
