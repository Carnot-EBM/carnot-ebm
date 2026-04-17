"""Phase 3 seed modules — bridging discrete Ising to continuous energy landscapes.

**Phase 3 North Star:** functional parity with Kona (Logical Intelligence) —
continuous-latent, non-autoregressive reasoning.  Nothing in this package is
production-ready; these are concrete seeds for that long-horizon goal.

Spec: REQ-KONA-001, REQ-KONA-002, REQ-KONA-003
"""

from carnot.phase3.continuous_ebm import (
    ContinuousEBM,
    build_kona_artifact,
    compare_minima,
    compare_samplers,
    fit_continuous_ebm,
    sample_continuous,
    sample_energy_matching,
    sample_langevin,
)

__all__ = [
    "ContinuousEBM",
    "build_kona_artifact",
    "compare_minima",
    "compare_samplers",
    "fit_continuous_ebm",
    "sample_continuous",
    "sample_energy_matching",
    "sample_langevin",
]
