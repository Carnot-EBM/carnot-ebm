"""Tests for Phase-2a sampler correctness audit (Exp 1094).

These tests verify that:
  - The sequential single-site Gibbs reference satisfies detailed
    balance on a simple 2-spin Ising model with a known closed-form
    Boltzmann distribution.
  - KLDivergenceEstimator runs without error on the reference Gibbs
    samples and returns a non-negative number.
  - The experiment script's build_artifact() emits a fully-populated
    schema with the verdict mapping correctly tied to the measured
    KL value.

Spec: REQ-DIAG-002, REQ-SAMPLE-003, SCENARIO-PHASE2A-001
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from carnot.eval.diagnostics import KLDivergenceEstimator  # noqa: E402

import experiment_1094_phase2a_sampler_correctness_audit as exp1094  # noqa: E402


# ---------------------------------------------------------------------------
# Test 1: Gibbs sampler obeys detailed balance on a 2-spin model
# ---------------------------------------------------------------------------


def test_gibbs_sampler_satisfies_detailed_balance_on_simple_case() -> None:
    """Sequential Gibbs on J=[[0,1],[1,0]] must converge to the correct
    Boltzmann distribution.

    For two spins with ferromagnetic coupling J=1 the Hamiltonian is
    H(s) = -J * s1 * s2, so the agree/disagree weight ratio is
    exp(2*beta*J) and at beta=1.0 the closed-form agreement
    probability is exp(beta*J) / (exp(beta*J) + exp(-beta*J)) =
    sigmoid(2*beta*J) ≈ 0.881. We sample long enough that the
    empirical agreement-fraction has small sampling error and then
    check it within a generous 5%-absolute band of the closed form.

    A sampler that violates detailed balance — for instance, a
    fully-parallel Glauber on this same J — does NOT converge to the
    Boltzmann distribution at strong coupling, and would fail this
    test. So the test doubles as a regression guard against future
    changes silently swapping the reference sampler for an
    approximate one.

    Spec: REQ-SAMPLE-003
    """
    J = np.array([[0.0, 1.0], [1.0, 0.0]])
    beta = 1.0
    samples = exp1094.gibbs_single_site(J, beta=beta, n_samples=20000, burnin=2000, seed=42)
    agree = float(np.mean(samples[:, 0] == samples[:, 1]))
    expected = math.exp(beta) / (math.exp(beta) + math.exp(-beta))
    assert abs(agree - expected) < 0.05, (
        f"agree fraction {agree:.3f} deviates from Boltzmann "
        f"{expected:.3f} by more than 5% — detailed balance suspect"
    )


# ---------------------------------------------------------------------------
# Test 2: KL estimator runs on reference samples
# ---------------------------------------------------------------------------


def test_kl_estimator_runs_on_reference_samples() -> None:
    """KLDivergenceEstimator must produce a non-negative finite value
    on the popcount distributions of two independently-drawn Gibbs runs.

    Two independent Gibbs runs on the same J should produce a small
    KL — they are sampling the same distribution. A "small" KL is
    bounded by the plug-in estimator's noise floor, which we
    approximate as the asymptotic confidence-interval half-width.
    Anything above 5x that bound suggests the estimator (or one of
    the runs) is broken, so we assert a generous upper bound rather
    than KL == 0.

    Spec: REQ-DIAG-002
    """
    J = exp1094.make_frustrated_ring(8)
    a = exp1094.gibbs_single_site(J, n_samples=2000, burnin=200, seed=1)
    b = exp1094.gibbs_single_site(J, n_samples=2000, burnin=200, seed=2)
    pop_a = exp1094.samples_to_popcount(a)
    pop_b = exp1094.samples_to_popcount(b)
    estimator = KLDivergenceEstimator()
    kl = estimator.estimate(pop_a, pop_b, n_bins=9)
    assert math.isfinite(kl), f"KL produced non-finite value {kl!r}"
    assert kl >= 0.0, f"KL must be non-negative, got {kl}"
    ci = estimator.kl_confidence_interval(min(len(pop_a), len(pop_b)))
    # Two independent Gibbs runs should agree well within 5x the
    # asymptotic noise floor; a much larger value would indicate a
    # statistical bug in either the sampler or the estimator.
    assert kl < 5.0 * ci + 0.1, (
        f"KL between two independent Gibbs runs {kl:.4f} exceeds "
        f"5x noise floor + slack ({5.0 * ci + 0.1:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 3: build_artifact emits required schema and verdict mapping
# ---------------------------------------------------------------------------


def test_sampler_audit_artifact_written(tmp_path, monkeypatch) -> None:
    """build_artifact() must produce all schema fields and choose the
    correct verdict from the measured KL value.

    We drive build_artifact() directly with synthetic inputs so the
    test is fast and deterministic — the full main() takes minutes
    to run the 5,000-sample sweeps. Three branches matter:
      1. Board unreachable + KL=None  -> board_unreachable_theoretical_bound_only.
      2. KL above threshold           -> fpga_sampler_distribution_mismatch_confirmed.
      3. KL below threshold           -> fpga_poc_validated_kl_within_bounds.

    The test also writes the artifact to disk in a tmp_path to
    confirm it is JSON-serialisable end-to-end.

    Spec: SCENARIO-PHASE2A-001
    """
    rng = np.random.default_rng(0)
    fake_gibbs = rng.choice([-1, 1], size=(100, exp1094.N_SPINS)).astype(np.int8)
    fake_parallel = rng.choice([-1, 1], size=(100, exp1094.N_SPINS)).astype(np.int8)

    # Branch 1: board unreachable with no KL measurement.
    art1 = exp1094.build_artifact(
        board_reachable=False,
        fpga_stats=None,
        kl_fpga_gibbs=None,
        kl_measurement_mode="theoretical_analytical_bound",
        gibbs_samples=fake_gibbs,
        parallel_samples=fake_parallel,
        cpu_gibbs_latency_ms=1.23,
        gpu_ising_available=False,
        gpu_ising_latency_ms=None,
        gpu_backend="torch_unavailable",
        duration_s=10.0,
    )
    assert art1["honest_verdict"] == "board_unreachable_theoretical_bound_only"

    # Branch 2: KL above threshold confirms Finding #2.
    art2 = exp1094.build_artifact(
        board_reachable=True,
        fpga_stats={"latency_us_mean": 25.0},
        kl_fpga_gibbs=0.20,  # > 0.05 threshold
        kl_measurement_mode="software_parallel_glauber_proxy",
        gibbs_samples=fake_gibbs,
        parallel_samples=fake_parallel,
        cpu_gibbs_latency_ms=1.23,
        gpu_ising_available=True,
        gpu_ising_latency_ms=0.05,
        gpu_backend="torch_cuda",
        duration_s=10.0,
    )
    assert art2["honest_verdict"] == "fpga_sampler_distribution_mismatch_confirmed"
    assert art2["phase2a_finding2_confirmed"] is True

    # Branch 3: KL below threshold validates the POC.
    art3 = exp1094.build_artifact(
        board_reachable=True,
        fpga_stats={"latency_us_mean": 25.0},
        kl_fpga_gibbs=0.001,  # << 0.05 threshold
        kl_measurement_mode="software_parallel_glauber_proxy",
        gibbs_samples=fake_gibbs,
        parallel_samples=fake_parallel,
        cpu_gibbs_latency_ms=1.23,
        gpu_ising_available=True,
        gpu_ising_latency_ms=0.05,
        gpu_backend="torch_cuda",
        duration_s=10.0,
    )
    assert art3["honest_verdict"] == "fpga_poc_validated_kl_within_bounds"
    assert art3["phase2a_finding2_confirmed"] is False

    # Required fields all present.
    required = {
        "experiment",
        "title",
        "run_date",
        "schema",
        "duration_s",
        "board_ip",
        "board_reachable",
        "kl_fpga_gibbs",
        "kl_measurement_mode",
        "kl_acceptance_threshold",
        "phase2a_finding2_confirmed",
        "gpu_ising_available",
        "gpu_ising_latency_ms",
        "cpu_gibbs_latency_ms",
        "honest_verdict",
    }
    for art in (art1, art2, art3):
        missing = required - art.keys()
        assert not missing, f"artifact missing required fields: {missing}"

    # End-to-end JSON serialisation must succeed.
    import json

    out = tmp_path / "artifact.json"
    out.write_text(json.dumps(art3, indent=2))
    reloaded = json.loads(out.read_text())
    assert reloaded["honest_verdict"] == "fpga_poc_validated_kl_within_bounds"


# ---------------------------------------------------------------------------
# Test 4: parallel-vs-sequential KL is observable on frustrated J
# ---------------------------------------------------------------------------


def test_parallel_vs_sequential_kl_distinguishable_on_frustrated_ring() -> None:
    """On the 12-spin frustrated antiferromagnetic ring, parallel
    Glauber and sequential Gibbs must produce empirically
    distinguishable popcount distributions, OR the sampler under
    audit happens to be safe on this J.

    This is a "no-op crash" guard rather than a strict numerical
    check: the experiment's value comes from REPORTING the KL number
    to the artifact, not from asserting a specific value. We only
    require that the run completes and produces a finite KL with the
    correct sign — the audit verdict itself is data, not a test
    assertion.

    Spec: REQ-DIAG-002, REQ-SAMPLE-003
    """
    J = exp1094.make_frustrated_ring(exp1094.N_SPINS)
    g = exp1094.gibbs_single_site(J, n_samples=1000, burnin=200, seed=1)
    p = exp1094.parallel_glauber(J, n_samples=1000, burnin=200, seed=2)
    kl = KLDivergenceEstimator().estimate(
        exp1094.samples_to_popcount(p),
        exp1094.samples_to_popcount(g),
        n_bins=exp1094.N_SPINS + 1,
    )
    assert math.isfinite(kl), "KL must be finite"
    assert kl >= 0.0, f"KL must be non-negative, got {kl}"
