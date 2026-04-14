"""FpgaBackend: quantum-inspired sparse Ising sampler for KV260 FPGA.

**Researcher summary:**
    Full SamplerBackend implementation targeting the KV260 AXI-Lite control
    plane (Exp 228 register map). Uses a quantum-inspired log-linear β-schedule
    from arXiv 2604.04606 achieving a 6× SA speedup, sparse coupling (max_degree
    ≤ 32) matching Carnot's clause-graph masking from Exp 61, and optional
    LagONN augmented Lagrangian penalty (arXiv 2505.07179) to escape infeasible
    local minima.

    When ``CARNOT_KV260_BITFILE`` is set, dispatch routes samples through the
    PYNQ AXI-Lite overlay via ``FPGAIsingSampler``.  When the variable is
    absent, dispatch falls back to ``ParallelIsingSampler`` using the same
    geometric annealing schedule so the quantum-inspired speedup is preserved
    even on CPU.

    Future: KANELÉ (arXiv 2512.12850) for KAN LUT evaluation directly on FPGA
    without a host-side JAX forward pass.

**Detailed explanation for engineers:**
    Pipeline per call to ``minimize_energy`` or ``sample``:

    1. ``sparsify_coupling(coupling, max_degree)``  — zero out below-threshold
       couplings so each spin has at most *max_degree* neighbours.  This
       matches the KV260 AXI-Lite contract (docs/fpga-ising-design.md).

    2. ``_apply_lagrangian_penalty(coupling, h, strength)``  (optional) — augment
       biases with a frustration-weighted Lagrangian term (arXiv 2505.07179)
       to help the annealer escape locally infeasible states.

    3. ``dispatch(coupling, h, n_samples, n_steps)``  — if
       ``CARNOT_KV260_BITFILE`` is set, hand off to ``FPGAIsingSampler`` (Exp
       228 AXI path + readback); otherwise run ``ParallelIsingSampler`` with
       ``schedule_type="geometric"`` for the log-linear warmup from
       arXiv 2604.04606.

    The Q8.8 quantization helper (``quantize_to_q88``) and AXI serializer
    (``serialize_to_axi``) are also exposed as standalone functions so tests
    and future tooling can exercise the register-map path independently.

Spec: REQ-SAMPLE-009, SCENARIO-SAMPLE-018, SCENARIO-SAMPLE-019
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from carnot.samplers.parallel_ising import AnnealingSchedule, ParallelIsingSampler

logger = logging.getLogger(__name__)

# Q8.8 fixed-point: 8 integer bits + 8 fractional bits = 16-bit signed integer.
# Representable range: [-128.0, 127.99609375]
_Q88_FRAC_BITS: int = 8
_Q88_SCALE: float = float(1 << _Q88_FRAC_BITS)  # 256.0
_Q88_MIN: int = -(1 << 15)  # -32768
_Q88_MAX: int = (1 << 15) - 1  # 32767


def quantize_to_q88(matrix: np.ndarray) -> np.ndarray:
    """Convert a float array to Q8.8 fixed-point 16-bit signed integers.

    **Researcher summary:**
        Q8.8 is the format used by the KV260 Ising overlay (Exp 228).
        8 integer bits + 8 fractional bits gives a representable range of
        [-128.0, ~127.996] with a resolution of 1/256 ≈ 0.0039.

    **Detailed explanation for engineers:**
        Each float value is multiplied by 256 (the fractional-bit scale),
        rounded to the nearest integer, clipped to the 16-bit signed range
        [-32768, 32767], and cast to int16.  The inverse operation is to
        divide by 256.

        Example: 1.5 → round(1.5 * 256) = 384 fits in 16-bit range.
        Dequantized back: 384 / 256 = 1.5 exactly (no rounding error for
        multiples of 1/256).

        Values outside [-128.0, ~127.996] are clipped, introducing a
        quantization error equal to the excess above/below the range.

    Args:
        matrix: Float array of any shape.

    Returns:
        int16 numpy array of the same shape, Q8.8 encoded.

    Spec: REQ-SAMPLE-009
    """
    scaled = np.round(np.asarray(matrix, dtype=np.float64) * _Q88_SCALE)
    clipped = np.clip(scaled, _Q88_MIN, _Q88_MAX)
    return clipped.astype(np.int16)


def sparsify_coupling(coupling: np.ndarray, max_degree: int = 32) -> np.ndarray:
    """Prune each row of a coupling matrix to at most *max_degree* neighbours.

    **Researcher summary:**
        Implements the sparse-connectivity strategy from arXiv 2604.04606
        (quantum-inspired sparse Ising, 6× SA speedup).  Per spin, only the
        *max_degree* largest-magnitude couplings are retained; the rest are
        zeroed.  This matches Carnot's clause-graph masking from Exp 61 and the
        KV260 hardware contract of max_degree=32 (docs/fpga-ising-design.md).

    **Detailed explanation for engineers:**
        For each row *i* of the coupling matrix, the function:
        1. Identifies all non-zero entries (by index).
        2. If the count exceeds *max_degree*, sorts those entries by descending
           absolute value.
        3. Zeros out the excess (lower-magnitude) entries in the output array.

        The diagonal is always forced to zero (no self-coupling in Ising
        models).  The resulting matrix may be asymmetric: spin i may keep
        coupling to spin j while spin j prunes coupling to spin i.  The KV260
        hardware handles this correctly because each spin's edge list is
        independent (CSR sparse upload).

    Args:
        coupling: Dense symmetric coupling matrix, shape ``(n, n)``.
        max_degree: Maximum non-zero couplings to retain per spin (row).
            Must be ≥ 0.  Default 32 matches the KV260 AXI-Lite contract.

    Returns:
        float32 numpy array, same shape as *coupling*, with excess entries
        zeroed.

    Spec: REQ-SAMPLE-009
    """
    out = np.asarray(coupling, dtype=np.float32).copy()
    np.fill_diagonal(out, 0.0)
    n = out.shape[0]
    for i in range(n):
        row_magnitudes = np.abs(out[i])
        nnz_indices = np.flatnonzero(row_magnitudes)
        if len(nnz_indices) > max_degree:
            # Sort the non-zero indices by descending magnitude.
            sorted_nnz = nnz_indices[np.argsort(row_magnitudes[nnz_indices])[::-1]]
            # Zero out entries ranked beyond max_degree.
            out[i, sorted_nnz[max_degree:]] = 0.0
    return out


def quantum_annealing_schedule(
    n_steps: int, beta_min: float, beta_max: float
) -> list[float]:
    """Compute a log-linear (geometric) β-schedule from arXiv 2604.04606.

    **Researcher summary:**
        β(t) = β_min × (β_max / β_min)^(t / n_steps), for t = 0, 1, …,
        n_steps.

        This geometric interpolation in inverse temperature (log-linear in β)
        achieves the 6× simulated-annealing speedup reported in
        arXiv 2604.04606 by allocating proportionally more sweeps near the
        low-temperature ground-state phase compared to a linear schedule.

    **Detailed explanation for engineers:**
        The schedule is strictly monotone increasing from β_min to β_max over
        n_steps+1 steps.  At t=0, β=β_min (hot: broad exploration).  At
        t=n_steps, β=β_max (cold: ground-state focus).  The midpoint satisfies
        β(n_steps/2) = sqrt(β_min × β_max) — the geometric mean — which is
        exactly the property that distinguishes a log-linear from a linear
        schedule.

        When n_steps=0 (no annealing), the function returns [β_max] (a
        single-element list at the target temperature).

    Args:
        n_steps: Number of annealing steps.  The returned list has length
            n_steps+1.
        beta_min: Starting inverse temperature (high temperature).
        beta_max: Ending inverse temperature (low temperature).

    Returns:
        List of n_steps+1 float values, monotone increasing, β_min to β_max.

    Spec: REQ-SAMPLE-009
    """
    if n_steps == 0:
        return [beta_max]
    ratio = beta_max / beta_min
    return [beta_min * (ratio ** (t / n_steps)) for t in range(n_steps + 1)]


def serialize_to_axi(
    j_sparse: np.ndarray,
    h: np.ndarray,
    beta: float,
) -> dict[str, Any]:
    """Serialise a sparse Ising problem to Exp 228 AXI-Lite register-map values.

    **Researcher summary:**
        Converts a sparsified coupling matrix and bias vector into the integer
        words that the KV260 AXI-Lite Ising overlay expects (see
        ``carnot.samplers.fpga_ising`` and ``docs/fpga-ising-design.md``).
        Builds the CSR sparse encoding via ``compile_sparse_problem`` and
        encodes β at the BETA_FINAL register using Q8.8 fixed-point.

    **Detailed explanation for engineers:**
        Register layout (Exp 228 design):

        - BETA_FINAL  (0x001C): Q8.8-encoded beta value
        - SPIN_COUNT  (0x0008): total number of spins
        - bias_words [0x1000+]: Q8.8 bias per spin (one 32-bit word each)
        - row_ptr    [0x2000+]: CSR row-pointer array (n+1 uint32 words)
        - edge_words [0x4000+]: packed neighbour+weight word per edge

        The returned dict can be iterated to perform a register upload:

            for reg, val in result.items():
                transport.write(regmap[reg], val)

    Args:
        j_sparse: Sparse coupling matrix (dense float32 with zeroed entries),
            shape ``(n, n)``.  Must have max ≤ 32 non-zeros per row, zero
            diagonal, and be compatible with ``FPGAArchitecture.max_spins``.
        h: Bias vector, shape ``(n,)``.
        beta: Inverse temperature to encode at BETA_FINAL.

    Returns:
        Dict with integer-valued entries:

        - ``"SPIN_COUNT"``: int
        - ``"BETA_FINAL"``: int (Q8.8 encoded)
        - ``"bias_words"``: list[int]
        - ``"row_ptr"``: list[int]
        - ``"edge_words"``: list[int]

    Spec: REQ-SAMPLE-009
    """
    from carnot.samplers.fpga_ising import (
        FPGAArchitecture,
        _quantize_word,
        compile_sparse_problem,
    )

    arch = FPGAArchitecture()
    compiled = compile_sparse_problem(h, j_sparse, architecture=arch)
    beta_word = _quantize_word(beta, arch.frac_bits)

    return {
        "SPIN_COUNT": compiled.n_spins,
        "BETA_FINAL": int(beta_word),
        "bias_words": [int(w) for w in compiled.bias_words],
        "row_ptr": [int(p) for p in compiled.row_ptr],
        "edge_words": [int(w) for w in compiled.edge_words],
    }


def _apply_lagrangian_penalty(
    coupling: np.ndarray,
    h: np.ndarray,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a LagONN-inspired augmented Lagrangian penalty to the bias vector.

    **Researcher summary:**
        LagONN (arXiv 2505.07179) adds augmented Lagrangian penalty terms to
        help the annealer escape infeasible local minima — states that satisfy
        no clause or constraint even at low temperature.  This implementation
        approximates the penalty by augmenting each spin's bias with a term
        proportional to the total frustrated (antiferromagnetic, negative)
        coupling incident on it.  Increasing the penalty strength *λ* makes
        infeasible configurations energetically less favourable, pushing the
        annealing trajectory away from frustrated attractors.

    **Detailed explanation for engineers:**
        For each spin *i* the penalty contribution is:

            penalty_i = −λ × Σ_j  min(coupling[i,j], 0)

        The sum picks up only the antiferromagnetic (negative) coupling
        strengths incident on *i*.  Subtracting this from the bias:

            h_penalized[i] = h[i] + λ × |Σ_j  min(coupling[i,j], 0)|

        effectively increases the bias for spins embedded in frustrated
        neighbourhoods, raising their probability of flipping and escaping
        the infeasible attractor.

        The coupling matrix is returned unchanged; only the bias vector is
        modified.

    Args:
        coupling: Coupling matrix, shape ``(n, n)``.
        h: Bias vector, shape ``(n,)``.
        strength: Lagrange multiplier scale *λ*.  Larger values impose a
            stronger penalty on frustrated configurations.

    Returns:
        Tuple ``(coupling, h_penalized)`` where *coupling* is the original
        matrix and *h_penalized* has frustration penalties added to the biases.

    Spec: REQ-SAMPLE-009
    """
    # sum_j min(coupling[i,j], 0) is ≤ 0 for each spin i.
    # Negating and multiplying by strength gives a positive bias augmentation
    # for spins that are heavily coupled to antiferromagnetic neighbours.
    frustration = np.sum(np.minimum(coupling, 0.0), axis=1).astype(np.float32)
    h_penalized = np.asarray(h, dtype=np.float32) - strength * frustration
    return coupling, h_penalized


@dataclass
class FpgaBackend:
    """FPGA sampler backend with quantum-inspired sparse Ising annealing.

    **Researcher summary:**
        Full ``SamplerBackend`` implementation for the KV260 Ising overlay.
        Combines the Exp 228 AXI-Lite register-map path with:

        - sparse connectivity (max_degree ≤ 32) from arXiv 2604.04606
        - log-linear β-schedule achieving 6× SA speedup (arXiv 2604.04606)
        - optional LagONN penalty for infeasible local minima (arXiv 2505.07179)

        When ``CARNOT_KV260_BITFILE`` is set the backend routes to the PYNQ
        AXI overlay; otherwise it falls back to ``ParallelIsingSampler`` with
        the same geometric annealing schedule.

        Future: KANELÉ (arXiv 2512.12850) — KAN LUT evaluation on FPGA
        without host-side JAX forward pass.

    **Detailed explanation for engineers:**
        ``minimize_energy`` and ``sample`` both call the same three-stage
        pipeline: sparsify → (optional) penalise → dispatch.  The dispatch
        checks ``CARNOT_KV260_BITFILE`` at call time (not construction), so the
        object can be created before the environment is configured.

        The ``backend_name`` property reflects the active execution path:
        ``"fpga"`` when the bitfile variable is set, ``"fpga_cpu_fallback"``
        otherwise.  This mirrors the honest labelling contract from
        REQ-SAMPLE-009.

    Attributes:
        seed: PRNG seed for the CPU-fallback ``ParallelIsingSampler``.
        max_degree: Max neighbours per spin kept by ``sparsify_coupling``.
            Default 32 matches the KV260 AXI-Lite hardware contract.
        beta_min: Starting inverse temperature for the annealing schedule.
        beta_max: Final inverse temperature (used as the annealing endpoint
            and passed as *beta* to ``FPGAIsingSampler.minimize_energy``).
        use_lagrangian_penalty: If True, apply the LagONN penalty augmentation
            (arXiv 2505.07179) before dispatching.
        lagrangian_penalty_strength: Lagrange multiplier scale *λ*.

    Spec: REQ-SAMPLE-009, SCENARIO-SAMPLE-018, SCENARIO-SAMPLE-019
    """

    seed: int = 42
    max_degree: int = 32
    beta_min: float = 0.1
    beta_max: float = 10.0
    use_lagrangian_penalty: bool = False
    lagrangian_penalty_strength: float = 1.0

    @property
    def backend_name(self) -> str:
        """Return execution-path label consistent with REQ-SAMPLE-009.

        Returns ``"fpga"`` when ``CARNOT_KV260_BITFILE`` is set (hardware or
        software-model path via ``FPGAIsingSampler``), or
        ``"fpga_cpu_fallback"`` when the env var is absent.
        """
        if os.environ.get("CARNOT_KV260_BITFILE"):
            return "fpga"
        return "fpga_cpu_fallback"

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        """Run quantum-inspired annealing to find low-energy spin configurations.

        **Detailed explanation for engineers:**
            Sparsifies couplings to max_degree neighbours per spin, optionally
            applies LagONN penalty to biases, then calls ``dispatch``.  The
            *beta* argument is accepted for protocol compliance but the backend
            uses its own ``beta_min`` / ``beta_max`` schedule from the
            dataclass attributes.

        Args:
            biases: Bias vector, shape ``(n_spins,)``.
            couplings: Symmetric coupling matrix, shape ``(n_spins, n_spins)``.
            n_samples: Number of independent samples to return.
            n_steps: Number of annealing / sweep steps.
            beta: Ignored; the backend uses its ``beta_max`` attribute.

        Returns:
            Boolean array of shape ``(n_samples, n_spins)``.

        Spec: REQ-SAMPLE-009
        """
        j_sparse = sparsify_coupling(np.asarray(couplings), self.max_degree)
        h = np.asarray(biases, dtype=np.float32)
        if self.use_lagrangian_penalty:
            j_sparse, h = _apply_lagrangian_penalty(
                j_sparse, h, self.lagrangian_penalty_strength
            )
        return self.dispatch(j_sparse, h, n_samples, n_steps)

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Draw samples using the quantum-inspired dispatch pipeline.

        **Detailed explanation for engineers:**
            Reads ``n_steps`` from *config* (default 500), sparsifies,
            optionally penalises, then dispatches.  The ``"beta"`` key in
            *config* is accepted for protocol compliance but the backend uses
            its own ``beta_max``.

        Args:
            biases: Bias vector, shape ``(n_spins,)``.
            couplings: Symmetric coupling matrix, shape ``(n_spins, n_spins)``.
            n_samples: Number of samples to draw.
            config: Backend-specific configuration.  Reads ``"n_steps"`` (int,
                default 500).

        Returns:
            Boolean array of shape ``(n_samples, n_spins)``.

        Spec: REQ-SAMPLE-009
        """
        steps = int(config.get("n_steps", 500))
        j_sparse = sparsify_coupling(np.asarray(couplings), self.max_degree)
        h = np.asarray(biases, dtype=np.float32)
        if self.use_lagrangian_penalty:
            j_sparse, h = _apply_lagrangian_penalty(
                j_sparse, h, self.lagrangian_penalty_strength
            )
        return self.dispatch(j_sparse, h, n_samples, steps)

    def dispatch(
        self,
        coupling: np.ndarray,
        h: np.ndarray,
        n_samples: int,
        n_steps: int,
    ) -> np.ndarray:
        """Route sampling to the KV260 FPGA overlay or CPU fallback.

        **Detailed explanation for engineers:**
            Checks ``CARNOT_KV260_BITFILE`` at *call time* (not at object
            construction).  This allows the backend to be instantiated before
            the hardware environment is known, and lets tests toggle the env
            var between calls.

            **FPGA path** (``CARNOT_KV260_BITFILE`` is set):
                Uploads the problem via ``FPGAIsingSampler`` (Exp 228
                AXI-Lite register map: Q8.8 biases, CSR edges, BETA_FINAL,
                CONTROL_START, sample readback).  ``FPGAIsingSampler`` will
                fall back to its own CPU path if the bitfile does not load —
                ensuring the call always succeeds.

            **CPU fallback path** (env var absent):
                Runs ``ParallelIsingSampler`` with
                ``AnnealingSchedule(schedule_type="geometric")`` — the same
                log-linear warmup used by the FPGA path, preserving the
                arXiv 2604.04606 speedup benefit on CPU.

        Args:
            coupling: Sparsified coupling matrix (dense float32, zeroed pruned
                entries), shape ``(n, n)``.
            h: Bias vector, shape ``(n,)``.
            n_samples: Number of samples to return.
            n_steps: Number of annealing steps (warmup sweeps).

        Returns:
            Boolean array of shape ``(n_samples, n_spins)``.

        Spec: REQ-SAMPLE-009, SCENARIO-SAMPLE-018
        """
        bitfile = os.environ.get("CARNOT_KV260_BITFILE")

        if bitfile:
            # Hardware (or software-model) path: PYNQ AXI upload via Exp 228.
            # FPGAIsingSampler handles the PYNQ overlay load and falls back to
            # its own CPU path if the bitfile is missing or PYNQ is unavailable.
            logger.info(
                "FpgaBackend.dispatch: routing to KV260 via CARNOT_KV260_BITFILE=%s",
                bitfile,
            )
            from carnot.samplers.fpga_ising import FPGAIsingSampler

            fpga = FPGAIsingSampler(bitfile_path=bitfile, seed=self.seed)
            return fpga.minimize_energy(h, coupling, n_samples, n_steps, self.beta_max)

        # CPU fallback: quantum-inspired geometric (log-linear) β-schedule.
        logger.debug(
            "FpgaBackend.dispatch: CARNOT_KV260_BITFILE unset — CPU fallback "
            "with geometric annealing schedule (arXiv 2604.04606)"
        )
        sampler = ParallelIsingSampler(
            n_warmup=n_steps,
            n_samples=n_samples,
            steps_per_sample=20,
            schedule=AnnealingSchedule(
                beta_init=self.beta_min,
                beta_final=self.beta_max,
                schedule_type="geometric",
            ),
            use_checkerboard=True,
        )
        key = jrandom.PRNGKey(self.seed)
        b_jax = jnp.asarray(h, dtype=jnp.float32)
        j_jax = jnp.asarray(coupling, dtype=jnp.float32)
        samples = sampler.sample(key, b_jax, j_jax, beta=self.beta_max)
        return np.asarray(samples)
