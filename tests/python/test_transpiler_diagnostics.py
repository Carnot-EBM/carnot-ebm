"""Tests for PT-PCD convergence diagnostics.

Each diagnostic targets a specific failure mode. The tests construct
synthetic *known-failing* and *known-passing* inputs to verify the
diagnostic detects what it claims to detect.

Spec: REQ-PHASE2-005 (PT-PCD convergence diagnostics).
"""

from __future__ import annotations

import numpy as np

from carnot.hardware.transpiler import (
    all_diagnostics_pass,
    energy_histogram_overlap,
    kde_overlap_2d,
    swap_acceptance_health,
)


# REQ-PHASE2-005
def test_kde_overlap_passes_when_distributions_match() -> None:
    """Two samples drawn from the same Gaussian should have low TV
    distance and the diagnostic should pass.
    """
    rng = np.random.default_rng(0)
    teacher = rng.normal(size=(2000, 2))
    student = rng.normal(size=(2000, 2))
    res = kde_overlap_2d(teacher, student)
    assert res.passed, str(res)
    assert res.score < 0.10


# REQ-PHASE2-005
def test_kde_overlap_detects_mode_collapse() -> None:
    """Teacher has 2 well-separated modes, student only populates one.
    This is the canonical mode-collapse failure; the diagnostic must
    detect it.
    """
    rng = np.random.default_rng(0)
    # Teacher: bimodal at (-2, 0) and (+2, 0)
    teacher = np.concatenate(
        [
            rng.normal(loc=[-2.0, 0.0], scale=0.3, size=(1000, 2)),
            rng.normal(loc=[2.0, 0.0], scale=0.3, size=(1000, 2)),
        ]
    )
    # Student: only the right mode
    student = rng.normal(loc=[2.0, 0.0], scale=0.3, size=(2000, 2))
    res = kde_overlap_2d(teacher, student)
    assert not res.passed, str(res)
    # Mode collapse should give TV ~ 0.5 (half the mass missing)
    assert res.score > 0.30


# REQ-PHASE2-005
def test_energy_histogram_passes_when_aligned() -> None:
    """Teacher and student energies drawn from the same distribution —
    no spurious-minimum spike, diagnostic passes.
    """
    rng = np.random.default_rng(0)
    teacher_e = rng.normal(loc=-5.0, scale=1.0, size=2000)
    student_e = rng.normal(loc=-5.0, scale=1.0, size=2000)
    res = energy_histogram_overlap(teacher_e, student_e)
    assert res.passed, str(res)


# REQ-PHASE2-005
def test_energy_histogram_detects_spurious_well() -> None:
    """Student has a 5% mass spike at energy way below teacher minimum.
    That's a hallucinated black hole that hardware would relax into;
    the diagnostic must catch it.
    """
    rng = np.random.default_rng(0)
    teacher_e = rng.normal(loc=-5.0, scale=1.0, size=2000)
    spurious_well = np.full(100, -20.0)  # 5% of 2000 at very low energy
    student_e = np.concatenate(
        [
            rng.normal(loc=-5.0, scale=1.0, size=1900),
            spurious_well,
        ]
    )
    res = energy_histogram_overlap(teacher_e, student_e)
    assert not res.passed, str(res)
    assert res.score > 0.01


# REQ-PHASE2-005
def test_swap_acceptance_health_pass_and_fail() -> None:
    """A healthy ladder has all acceptance rates above 15%. A broken
    one has at least one rung pair below.
    """
    healthy = np.array([0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.18])
    res = swap_acceptance_health(healthy)
    assert res.passed
    assert res.score == healthy.min()

    broken = np.array([0.45, 0.40, 0.05, 0.30, 0.25])  # one bad pair
    res = swap_acceptance_health(broken)
    assert not res.passed
    assert res.score == 0.05


# REQ-PHASE2-005
def test_all_diagnostics_pass_aggregator() -> None:
    """The aggregator runs all three and returns ``(all_passed, results)``."""
    rng = np.random.default_rng(0)
    teacher_z = rng.normal(size=(2000, 2))
    student_z = rng.normal(size=(2000, 2))
    teacher_e = rng.normal(loc=-5.0, size=2000)
    student_e = rng.normal(loc=-5.0, size=2000)
    swap_accept = np.array([0.4, 0.35, 0.3, 0.25])

    passed, results = all_diagnostics_pass(teacher_z, student_z, teacher_e, student_e, swap_accept)
    assert passed
    assert len(results) == 3
    assert {r.name for r in results} == {
        "kde_overlap_2d",
        "energy_histogram_overlap",
        "swap_acceptance_health",
    }
