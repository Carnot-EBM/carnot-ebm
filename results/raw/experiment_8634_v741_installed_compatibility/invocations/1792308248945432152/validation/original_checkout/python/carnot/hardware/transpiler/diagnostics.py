"""Convergence diagnostics for Native Thermodynamic Distillation.

The three falsifiable checks Round 4 specified. **No payload is flashed
to physical hardware until all three pass** — that's the discipline
that protects us against the empirical-not-formal nature of
PT-PCD's KL guarantee.

Each diagnostic targets a specific failure mode of the training loop:

- ``kde_overlap_2d`` (Diagnostic A) — catches *mode collapse*. If the
  teacher MCMC corpus has 4 valid clusters and the student cold-chain
  samples populate only 1, the 2D earth-mover-style distance between
  the kernel-density estimates is large.
- ``energy_histogram_overlap`` (Diagnostic B) — catches *spurious
  minima*. If the BM hallucinated a deep well far from any teacher
  data, the student-energy histogram has a secondary spike below the
  teacher's minimum energy.
- ``swap_acceptance_health`` (Diagnostic C) — catches a *broken PT
  ladder*. If swap acceptance between any pair of adjacent rungs falls
  below the threshold (default 15%), hot-chain exploration is
  blockaded from reaching cold readouts and the negative phase is
  effectively a vanilla-PCD chain again.

These are *necessary, not sufficient* — small modes can still be
missed at finite sample resolution. They catch the major failure modes
that have caused real EBM-distillation failures in the literature.

Spec: REQ-PHASE2-005 (PT-PCD convergence diagnostics).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DiagnosticResult:
    """Outcome of one diagnostic. Pass/fail plus the underlying scalar
    score so the caller can plot trajectories across training epochs.
    """

    passed: bool
    score: float
    threshold: float
    name: str
    detail: str = ""

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return f"[{verdict}] {self.name}: score={self.score:.4f} thr={self.threshold:.4f}"


def kde_overlap_2d(
    teacher_samples: np.ndarray,
    student_samples: np.ndarray,
    threshold: float = 0.10,
    grid_size: int = 32,
) -> DiagnosticResult:
    """Diagnostic A: 2D kernel-density overlap between teacher MCMC
    samples and student decoded cold-chain samples. Both inputs are
    ``(B, 2)`` continuous arrays.

    Score is a discretized total-variation distance between two KDE-
    smoothed histograms on a common ``grid_size x grid_size`` grid:
    ``score = 0.5 * sum(|p_teacher - p_student|)``. Range ``[0, 1]``.
    Identical distributions give 0; disjoint give 1. The default
    ``threshold=0.10`` is the Round-4-specified hard pass criterion.

    Why TV not earth-mover: TV is cheaper to compute (no transport
    plan), and at the grid resolution we use the two are
    monotonically related for small differences. For the
    mode-collapse failure mode — where one mode has zero student
    mass — TV distance is ``2 * (mass_of_missing_mode)`` and clearly
    above 0.10 for any non-trivial collapse.
    """
    t = np.atleast_2d(np.asarray(teacher_samples, dtype=np.float64))
    s = np.atleast_2d(np.asarray(student_samples, dtype=np.float64))
    if t.shape[-1] != 2 or s.shape[-1] != 2:
        raise ValueError("expected 2D samples")

    # Common grid spanning union of supports + 5% margin
    lo = min(t.min(), s.min())
    hi = max(t.max(), s.max())
    margin = 0.05 * (hi - lo + 1e-12)
    edges = np.linspace(lo - margin, hi + margin, grid_size + 1)

    p_t, _, _ = np.histogram2d(t[:, 0], t[:, 1], bins=[edges, edges])
    p_s, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=[edges, edges])

    # Normalize and smooth (3x3 box filter — coarse but cheap KDE)
    p_t = p_t / max(p_t.sum(), 1.0)
    p_s = p_s / max(p_s.sum(), 1.0)

    kernel = np.ones((3, 3)) / 9.0
    p_t_smooth = _conv2d_same(p_t, kernel)
    p_s_smooth = _conv2d_same(p_s, kernel)

    # Renormalize after smoothing (boundary effects)
    p_t_smooth = p_t_smooth / p_t_smooth.sum()
    p_s_smooth = p_s_smooth / p_s_smooth.sum()

    tv = 0.5 * float(np.abs(p_t_smooth - p_s_smooth).sum())

    return DiagnosticResult(
        passed=tv <= threshold,
        score=tv,
        threshold=threshold,
        name="kde_overlap_2d",
        detail=f"TV distance over {grid_size}x{grid_size} grid",
    )


def _conv2d_same(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Plain numpy 2D 'same'-mode convolution. Used for KDE smoothing.
    We avoid scipy.signal.convolve2d to keep the dependency surface
    minimal; the kernel is small (3x3) so the overhead is trivial.
    """
    h, w = arr.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)))
    out = np.zeros_like(arr)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i : i + h, j : j + w]
    return out


def energy_histogram_overlap(
    teacher_energies: np.ndarray,
    student_energies: np.ndarray,
    n_bins: int = 64,
    secondary_spike_floor: float | None = None,
) -> DiagnosticResult:
    """Diagnostic B: detect *spurious-minima* hallucinations by
    comparing the teacher and student energy histograms.

    The pass criterion is: the student histogram has no secondary
    spike at energy lower than the teacher minimum. We define
    "secondary spike" as a histogram bin with mass > 1% of the total
    student samples, located strictly below ``min(teacher_energies)
    - secondary_spike_floor`` (default ``= 0.5 * std(teacher_energies)``,
    which is a reasonable margin for noise tolerance).

    The score returned is the *fraction of student samples* whose
    energy is below the teacher minimum (after subtracting the
    margin). Pass when score < 0.01.

    Parameters
    ----------
    teacher_energies, student_energies
        1D arrays of Ising energies ``E(s) = -s^T J s - h^T s``,
        evaluated at teacher-encoded samples (visible+hidden) and
        cold-chain student samples respectively.
    n_bins
        Histogram bins. Diagnostic doesn't actually use the histogram
        for the pass criterion; ``n_bins`` is reserved for caller-side
        plotting and is recorded in the detail string.
    secondary_spike_floor
        Energy below ``teacher_min - this`` counts as a hallucinated
        well. Default ``0.5 * std(teacher_energies)``.
    """
    te = np.asarray(teacher_energies, dtype=np.float64).ravel()
    se = np.asarray(student_energies, dtype=np.float64).ravel()
    if te.size == 0 or se.size == 0:
        raise ValueError("non-empty energy arrays required")
    if secondary_spike_floor is None:
        secondary_spike_floor = 0.5 * float(te.std())

    teacher_min = float(te.min())
    threshold_energy = teacher_min - secondary_spike_floor
    # Fraction of student samples below the threshold
    below = float((se < threshold_energy).mean())

    return DiagnosticResult(
        passed=below < 0.01,
        score=below,
        threshold=0.01,
        name="energy_histogram_overlap",
        detail=(
            f"teacher_min={teacher_min:.3f} thr={threshold_energy:.3f} "
            f"floor={secondary_spike_floor:.3f} bins={n_bins}"
        ),
    )


def swap_acceptance_health(
    swap_accept_per_pair: np.ndarray, threshold: float = 0.15
) -> DiagnosticResult:
    """Diagnostic C: PT replica-exchange health.

    ``swap_accept_per_pair`` is the per-rung-pair acceptance rate
    averaged over recent training epochs. Pass when the *minimum*
    acceptance across all adjacent pairs is at least ``threshold``
    (default 15%). Below that, hot-chain exploration is blockaded from
    reaching the cold chain — the temperature ladder is broken and
    Approach 3 degrades to vanilla PCD.

    Caller is expected to average over the last K epochs (e.g., 20)
    rather than passing a single noisy snapshot. The
    ``swap_accept_history`` list maintained by ``CarnotNativeDistiller``
    is the canonical source.
    """
    sa = np.asarray(swap_accept_per_pair, dtype=np.float64).ravel()
    if sa.size == 0:
        raise ValueError("non-empty swap-acceptance array required")
    min_acc = float(sa.min())

    return DiagnosticResult(
        passed=min_acc >= threshold,
        score=min_acc,
        threshold=threshold,
        name="swap_acceptance_health",
        detail=f"per-pair rates: min={min_acc:.3f} mean={sa.mean():.3f}",
    )


def all_diagnostics_pass(
    teacher_samples: np.ndarray,
    student_samples: np.ndarray,
    teacher_energies: np.ndarray,
    student_energies: np.ndarray,
    swap_accept_per_pair: np.ndarray,
) -> tuple[bool, list[DiagnosticResult]]:
    """Run all three diagnostics. Return ``(passed, results)``. The
    transpiler's caller should not flash a payload to physical
    hardware unless ``passed`` is True.
    """
    results = [
        kde_overlap_2d(teacher_samples, student_samples),
        energy_histogram_overlap(teacher_energies, student_energies),
        swap_acceptance_health(swap_accept_per_pair),
    ]
    return all(r.passed for r in results), results
