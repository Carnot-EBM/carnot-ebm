"""Phase 2 transpiler API: HardwareSpec + IsingSpec.

The transpiler consumes a trained energy function and a `HardwareSpec`
descriptor, and emits an `IsingSpec` containing the static `(J, h)` plus
the encoder/decoder pair `(phi, psi)` needed to deploy to physical Ising
hardware (KV260, ECP5/Nexus, future XTR-0, future photonic SLM).

These two dataclasses are the canonical Phase 2 surface. New hardware
backends extend them via the `kind` and `vendor_target` fields rather
than writing parallel one-off paths — that's the rule-7 (no vendor
abstractions in the core) compliance for hardware portability.

**Why three approaches share the same output schema.** Approach 1
(execution-trace, sparse hardware, small verifiers), Approach 2
(arc-cosine RKHS lift, dense optical hardware), and Approach 3 (native
thermodynamic distillation, sparse hardware, *any verifier size*) all
produce the same `(J, h, phi, psi)` artifact. The downstream
`SamplerBackend` integration (REQ-KONA-006) is approach-agnostic — it
just wants an Ising spec and an encoder/decoder pair. This is what
makes the transpiler a single artifact rather than three parallel
pipelines.

**Why provenance tracking matters.** A trained Ising payload is
useless without knowing which `state_dict` it was distilled from,
which hardware it targeted, and which transpilation approach
produced it. Mismatches between the three are the #1 cause of
phase-2 deployment confusion (you flash a payload trained for a
different verifier and the hardware "works" but produces garbage
verdicts). The provenance field is mandatory and recorded on every
`IsingSpec`.

Spec: REQ-PHASE2-001 (HardwareSpec descriptor), REQ-PHASE2-002 (IsingSpec
output schema).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np


HardwareKind = Literal["sparse", "dense"]
"""Hardware topology kind.

- ``sparse``: per-coupling routed hardware (KV260 LUT-based, Extropic Z1,
  ECP5/Nexus). Local couplings only; the J matrix is banded or sparse.
  Approach 1 (trace-based) and Approach 3 (native distillation) target
  this.
- ``dense``: wave-interference / matrix-vector hardware (photonic SLM,
  optical Ising, fully-connected analog crossbars). All N×N couplings
  are computed natively. Approach 2 (arc-cosine RKHS lift) targets this.
"""


VendorTarget = Literal[
    "xilinx-kv260",
    "extropic-z1",
    "lattice-ecp5",
    "photonic-slm",
    "synthetic-cpu",
]
"""Specific hardware target. ``synthetic-cpu`` runs the Ising sampler in
software for prototyping without real hardware — the conductor and tests
use this path; production deployments select one of the others.
"""


@dataclass
class HardwareSpec:
    """Hardware-capability descriptor consumed by the transpiler.

    The transpiler dispatches across Approaches 1/2/3 based on
    ``kind``, ``max_spins``, and the verifier's parameter count. The
    full selector logic is:

    - ``kind=="dense"`` → Approach 2 (arc-cosine RKHS lift)
    - ``kind=="sparse"`` and verifier W small (``W·log²(1/eps) <= max_spins``)
      → Approach 1 (execution-trace embedding) preferred for the formal
      KL bound
    - ``kind=="sparse"`` and verifier W large → Approach 3 (native
      thermodynamic distillation), the production path

    Fields:

    Parameters
    ----------
    kind
        Hardware topology — ``sparse`` or ``dense``. See ``HardwareKind``.
    max_spins
        Spin budget. Z1 is 16384; KV260 LUT-implementable is ~16k; ECP5
        depends on bitstream size; photonic SLM depends on vendor.
    beta_range
        ``(beta_min, beta_max)`` over which the deployed sampler must
        produce the target distribution. Approach 1's KL bound is taken
        over this range; Approach 3's PT ladder is initialized to span it.
    locality
        Optional. For ``sparse`` hardware, the maximum interaction order
        (2-local or 3-local) that the routing fabric can implement. KV260
        is 2-local for native LUTs, 3-local with auxiliary spin gadgets.
    vendor_target
        Specific hardware target string. Used for routing-mask generation
        and provenance only — no vendor SDK is imported on this path.
    """

    kind: HardwareKind
    max_spins: int
    beta_range: tuple[float, float]
    locality: Literal[2, 3] | None = None
    vendor_target: VendorTarget = "synthetic-cpu"

    def __post_init__(self) -> None:
        if self.max_spins <= 0:
            raise ValueError(f"max_spins must be positive, got {self.max_spins}")
        beta_min, beta_max = self.beta_range
        if not (beta_min > 0 and beta_max > beta_min):
            raise ValueError(
                f"beta_range must satisfy 0 < beta_min < beta_max, got {self.beta_range}"
            )
        if self.kind == "sparse" and self.locality is None:
            # Default sparse hardware to 2-local couplings (the conservative
            # KV260 / ECP5 / Z1 default; 3-local needs explicit auxiliary spins).
            object.__setattr__(self, "locality", 2)


@dataclass
class IsingSpec:
    """Compiled Ising payload deployable to physical hardware.

    The downstream sampler reads ``J``, ``h`` to produce spin samples;
    the encoder ``phi`` maps continuous inputs to visible spins; the
    decoder ``psi`` maps sampled spins back to the continuous output
    space. ``provenance`` records *which* verifier was distilled and
    *how*, so a deployed payload can never be mistaken for a different
    verifier's payload.

    Parameters
    ----------
    J
        Symmetric coupling matrix. Shape ``(N, N)``. Diagonal is zero
        (no self-loops, standard Ising convention).
    h
        Local-field vector. Shape ``(N,)``.
    phi
        Encoder ``continuous_input -> visible_spins``. Maps a real-valued
        latent ``z`` to its ``{-1, +1}`` visible-spin representation. For
        Approach 3 this is the Gray-code encoder; for Approach 1 it's the
        m-bit fractional binary encoder.
    psi
        Decoder ``visible_spins -> continuous_output``. Inverse of ``phi``,
        adds uniform spatial noise within the cell to keep the decoded
        distribution absolutely continuous (which matters for the formal
        KL bound in Approach 1).
    provenance
        Free-form dict recording: source ``state_dict`` hash, hardware
        spec, transpilation approach, training-corpus checksum, training
        date, key hyperparameters. Mandatory for deployment auditing.
    """

    J: np.ndarray
    h: np.ndarray
    phi: Callable[[np.ndarray], np.ndarray]
    psi: Callable[[np.ndarray], np.ndarray]
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.J.ndim != 2 or self.J.shape[0] != self.J.shape[1]:
            raise ValueError(f"J must be square 2D, got shape {self.J.shape}")
        n = self.J.shape[0]
        if self.h.shape != (n,):
            raise ValueError(f"h shape {self.h.shape} doesn't match J ({n}-spin)")
        if not np.allclose(self.J, self.J.T, atol=1e-8):
            raise ValueError("J must be symmetric (within 1e-8)")
        if np.any(np.abs(np.diag(self.J)) > 1e-8):
            raise ValueError("J diagonal must be zero (no self-loops)")

    @property
    def n_spins(self) -> int:
        """Total spin count (visible + hidden)."""
        return self.J.shape[0]

    def energy(self, s: np.ndarray) -> np.ndarray:
        """Compute Ising energy ``E(s) = -s^T J s - h^T s`` for one or more
        spin configurations. ``s`` shape ``(N,)`` returns a scalar; shape
        ``(B, N)`` returns a length-``B`` vector.
        """
        if s.ndim == 1:
            return -float(s @ self.J @ s) - float(self.h @ s)
        # batched: (B, N)
        return -np.einsum("bi,ij,bj->b", s, self.J, s) - s @ self.h
