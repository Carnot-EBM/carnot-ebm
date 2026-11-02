"""carnot.hardware.transpiler — Phase 2 continuous-to-Ising transpiler.

Compiles a trained continuous energy function into a deployable
``IsingSpec`` for sparse or dense Ising hardware (KV260, ECP5/Nexus,
future Extropic XTR-0, future photonic SLM). This is the canonical
artifact bridging Phase 1 (continuous parametric verifiers) and Phase
2 (Ising-native sampling hardware).

The transpiler scopes three approaches sharing a single output schema:

- **Approach 1 (execution-trace embedding)**: small verifiers, formal
  KL bound. Produces sparse coupling and an m-bit fractional-binary
  encoder with a static QUBO penalty graph. Not yet implemented in this
  module — the load-bearing production path for transformer-class
  verifiers is Approach 3 below.
- **Approach 2 (arc-cosine RKHS lift)**: dense optical hardware.
  Anchored in Cho-Saul (2009) + Kar-Karnick (2012) for the wave-
  interference free-lunch result; planned but not implemented here
  pending dense-hardware availability.
- **Approach 3 (native thermodynamic distillation)**: any verifier
  size, sparse hardware. Train a Boltzmann Machine to match the
  verifier's MCMC samples via PT-PCD with Gray-code encoding. This
  module *implements* Approach 3 — the production path that decouples
  spin count from verifier parameter count.

Public surface::

    from carnot.hardware.transpiler import (
        HardwareSpec, IsingSpec,                  # api.py
        encode_2d, decode_2d,                     # gray_code.py
        CarnotNativeDistiller, DistillerConfig,   # distill.py
        kde_overlap_2d, energy_histogram_overlap,
        swap_acceptance_health, all_diagnostics_pass,  # diagnostics.py
    )

Origin: Phase 2 mathematical foundation closed across four Deep Think
rounds in dialogue with the user, 2026-04-27. Continuous Ising-Rank
Theorem; Split-Verifier + Native Thermodynamic Distillation; Arc-
Cosine Quadratic Lemma; PT-PCD recipe with five guardrails. See
``openspec/change-proposals/continuous-to-ising-transpiler.md`` and
``memory/project_continuous_ising_rank.md`` for the full record.

Spec: REQ-PHASE2-001 through REQ-PHASE2-005.
"""

from carnot.hardware.transpiler.api import (
    HardwareKind,
    HardwareSpec,
    IsingSpec,
    VendorTarget,
)
from carnot.hardware.transpiler.diagnostics import (
    DiagnosticResult,
    all_diagnostics_pass,
    energy_histogram_overlap,
    kde_overlap_2d,
    swap_acceptance_health,
)
from carnot.hardware.transpiler.distill import (
    CarnotNativeDistiller,
    DistillerConfig,
)
from carnot.hardware.transpiler.gray_code import (
    decode_2d,
    decode_axis,
    encode_2d,
    encode_axis,
)

__all__ = [
    # api
    "HardwareKind",
    "HardwareSpec",
    "IsingSpec",
    "VendorTarget",
    # gray code
    "decode_2d",
    "decode_axis",
    "encode_2d",
    "encode_axis",
    # distillation
    "CarnotNativeDistiller",
    "DistillerConfig",
    # diagnostics
    "DiagnosticResult",
    "all_diagnostics_pass",
    "energy_histogram_overlap",
    "kde_overlap_2d",
    "swap_acceptance_health",
]
