#!/usr/bin/env python3
"""Experiment 1155: HMC compatibility diagnostics for the k=5 verifier gradient.

Run:
    JAX_PLATFORMS=cpu python scripts/experiment_1155_hmc_compatibility_diagnostics.py

Outputs:
    results/experiment_1155_hmc_compatibility_diagnostics.json

Spec: REQ-KONA-009, SCENARIO-KONA-008
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT))

from carnot.phase3.hmc_compatibility import (  # noqa: E402
    HMCCompatibilityConfig,
    build_default_continuous_subspace_components,
    build_default_latent_components,
    load_latent_dim_from_exp1154,
    run_hmc_compatibility_diagnostics,
)

EXP1154_PATH = _REPO_ROOT / "results" / "experiment_1154_snap_validity_sweep.json"
RESULT_PATH = _REPO_ROOT / "results" / "experiment_1155_hmc_compatibility_diagnostics.json"
DEFAULT_CONFIG = HMCCompatibilityConfig()


def main() -> dict[str, object]:
    """Run D1-D4 diagnostics and write the required JSON artifact."""
    latent_dim = load_latent_dim_from_exp1154(EXP1154_PATH)
    components = build_default_latent_components(latent_dim)
    continuous_components = build_default_continuous_subspace_components(latent_dim)
    artifact = run_hmc_compatibility_diagnostics(
        latent_dim=latent_dim,
        components=components,
        continuous_components=continuous_components,
        config=DEFAULT_CONFIG,
    )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print("=== Exp 1155: HMC compatibility diagnostics ===")
    print(f"latent_dim={artifact['latent_dim']}")
    print(f"d1_error_mean={artifact['d1_symplectic_reversibility_error_mean']:.6g}")
    print(f"d2_hamiltonian_variance={artifact['d2_hamiltonian_variance']:.6g}")
    print(f"d3_gradient_disparity_ratio={artifact['d3_gradient_disparity_ratio']:.6g}")
    print(f"d4_subspace_delta_h_variance={artifact['d4_subspace_delta_h_variance']:.6g}")
    print(f"d4_full_delta_h_variance={artifact['d4_full_delta_h_variance']:.6g}")
    print(f"gradient_method={artifact['gradient_method']}")
    print(f"hmc_regime={artifact['hmc_regime']}")
    print(f"recommended_sampler={artifact['recommended_sampler']}")
    print(f"honest_verdict={artifact['honest_verdict']}")
    print(f"written={RESULT_PATH}")
    return artifact


if __name__ == "__main__":  # pragma: no cover
    main()
