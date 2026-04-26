"""Tests for Exp 918: Lagrange Forgetting Multi-Constraint (RETRO-LAGRANGE-ENTROPY-DEGENERATE).

Verifies:
1. 8-constraint heterogeneous corpus produces non-degenerate entropy at step 20 (> 0.1).
2. Decay and no-decay updaters diverge by step 100 (different weight distributions).
3. Decay updater achieves weight_entropy > 0.5 nats at step 100 (SCENARIO-SELF-007).
4. signed_entropy_improvement is finite and real (verdict not degenerate_again_retire).

Spec traces: REQ-SELF-007, SCENARIO-SELF-007
"""

import math
import random
import sys
from pathlib import Path

import pytest

# Allow import from scripts/ and python/ when running from repo root.
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_918_lagrange_forgetting_multi_constraint import (  # noqa: E402
    CONSTRAINTS,
    FORGETTING_LAMBDA,
    TOTAL_STEPS,
    PHASE_BOUNDARY,
    compute_verdict,
    run_simulation,
    _make_updater,
)


# ---------------------------------------------------------------------------
# REQ-SELF-007-1: corpus is non-degenerate at first measurement interval
# ---------------------------------------------------------------------------

def test_initial_entropy_is_non_degenerate():
    """8-constraint heterogeneous corpus must have entropy > 0.1 at step 20.

    Root cause of Exp 909 failure was entropy = 0 (single constraint, p=1.0 always).
    This assertion is the primary guard ensuring that fix holds.

    Spec: REQ-SELF-007, SCENARIO-SELF-007
    """
    rng = random.Random(42)
    sim = run_simulation(rng)
    entropy_step20 = sim["entropy_at_first_interval_baseline"]
    assert entropy_step20 > 0.1, (
        f"Corpus still degenerate: entropy at step 20 = {entropy_step20:.4f} <= 0.1. "
        "Root cause of Exp 909 is not fixed."
    )


def test_initial_entropy_near_maximum():
    """With 8 equal-weight constraints, initial entropy should be close to log(8) ≈ 2.08.

    Not required to be exactly log(8) because updates shift weights slightly, but
    it must be substantially above 0 to confirm the corpus is genuinely diverse.

    Spec: REQ-SELF-007
    """
    rng = random.Random(918)
    sim = run_simulation(rng)
    # log(8) = 2.079; allow generous margin since 20 update steps already shift weights.
    assert sim["entropy_at_first_interval_baseline"] > 1.0, (
        "Initial entropy is far below log(8)=2.08 — constraint diversity insufficient."
    )


# ---------------------------------------------------------------------------
# REQ-SELF-007-3: decay and no-decay diverge by step 100
# ---------------------------------------------------------------------------

def test_decay_and_baseline_weights_differ_at_step_100():
    """Forgetting lambda > 0 must produce different weight distributions than lambda=0.

    If both updaters had the same weights, there would be no point to the forgetting curve.
    We verify they diverge by comparing weight_entropy at step 100.

    Spec: REQ-SELF-007
    """
    rng = random.Random(1234)
    sim = run_simulation(rng)
    # Find step-100 entry in interval_log (step 100 = index 4 since we record every 20).
    log_at_100 = next(e for e in sim["interval_log"] if e["step"] == 100)
    assert log_at_100["entropy_baseline"] != log_at_100["entropy_decay"], (
        "Baseline and decay updaters have identical entropy at step 100 — "
        "forgetting curve has no effect."
    )


# ---------------------------------------------------------------------------
# SCENARIO-SELF-007: decay updater entropy > 0.5 nats at step 100
# ---------------------------------------------------------------------------

def test_decay_entropy_above_floor_at_step_100():
    """Decay updater weight_entropy must be > 0.5 nats at step 100.

    SCENARIO-SELF-007 spec: 'weight_entropy at step 100 (with decay) > 0.5 nats'.

    Spec: SCENARIO-SELF-007
    """
    rng = random.Random(918)
    sim = run_simulation(rng)
    log_at_100 = next(e for e in sim["interval_log"] if e["step"] == 100)
    entropy_decay_100 = log_at_100["entropy_decay"]
    assert entropy_decay_100 > 0.5, (
        f"Decay updater entropy at step 100 = {entropy_decay_100:.4f} <= 0.5 nats. "
        "SCENARIO-SELF-007 not satisfied."
    )


# ---------------------------------------------------------------------------
# Verdict computation: non-degenerate corpus -> verdict not 'degenerate_again_retire'
# ---------------------------------------------------------------------------

def test_verdict_is_not_degenerate():
    """With a heterogeneous 8-constraint corpus, verdict must not be 'degenerate_again_retire'.

    If it is, the root cause identified in Exp 909 is still not fixed.

    Spec: REQ-SELF-007
    """
    rng = random.Random(918)
    sim = run_simulation(rng)
    verdict = compute_verdict(sim)
    assert verdict != "degenerate_again_retire", (
        f"Corpus is still degenerate despite heterogeneous design. Verdict: {verdict}. "
        f"Entropy at step 20: {sim['entropy_at_first_interval_baseline']:.4f}"
    )


def test_verdict_is_improvement():
    """signed_entropy_improvement > 0 -> verdict is 'marginal_improvement' or better.

    Spec: REQ-SELF-007
    """
    rng = random.Random(918)
    sim = run_simulation(rng)
    verdict = compute_verdict(sim)
    assert verdict in ("marginal_improvement", "forgetting_curve_improves_entropy"), (
        f"Expected improvement verdict, got: {verdict}. "
        f"signed_entropy_improvement = {sim['signed_entropy_improvement']:.4f}"
    )


# ---------------------------------------------------------------------------
# _make_updater: sanity check that pre-registration gives equal weights
# ---------------------------------------------------------------------------

def test_make_updater_registers_all_constraints():
    """_make_updater must register all 8 constraints from CONSTRAINTS dict.

    Pre-registration ensures entropy is defined from step 0 (equal weights = max entropy).

    Spec: REQ-SELF-007
    """
    updater = _make_updater(forgetting_lambda=0.0)
    assert updater.n_constraints == len(CONSTRAINTS), (
        f"Expected {len(CONSTRAINTS)} constraints, got {updater.n_constraints}."
    )


def test_make_updater_initial_entropy_non_zero():
    """Pre-registered equal weights must produce non-zero Shannon entropy.

    Spec: REQ-SELF-007-1
    """
    updater = _make_updater(forgetting_lambda=0.0)
    assert updater.weight_entropy > 0.0, (
        "Initial weight_entropy is 0.0 — equal-weight initialisation failed."
    )


# ---------------------------------------------------------------------------
# Corpus constants sanity checks
# ---------------------------------------------------------------------------

def test_eight_constraint_types():
    """CONSTRAINTS dict must have exactly 8 entries.

    The entire premise of Exp 918 vs Exp 909 is using 8 heterogeneous constraints.

    Spec: REQ-SELF-007
    """
    assert len(CONSTRAINTS) == 8, f"Expected 8 constraint types, got {len(CONSTRAINTS)}."


def test_violation_probabilities_are_distinct():
    """High violation probabilities must span at least 3 distinct values.

    A single shared violation probability would recreate the Exp 909 degenerate condition.

    Spec: REQ-SELF-007
    """
    high_probs = {v[0] for v in CONSTRAINTS.values()}
    assert len(high_probs) >= 3, (
        f"High violation probs are not diverse enough: {sorted(high_probs)}"
    )


def test_forgetting_lambda_corresponds_to_decay_rate():
    """FORGETTING_LAMBDA must equal -ln(0.95) within float tolerance.

    Spec: REQ-SELF-007
    """
    expected = -math.log(0.95)
    assert abs(FORGETTING_LAMBDA - expected) < 1e-9, (
        f"FORGETTING_LAMBDA={FORGETTING_LAMBDA} != -ln(0.95)={expected}"
    )
