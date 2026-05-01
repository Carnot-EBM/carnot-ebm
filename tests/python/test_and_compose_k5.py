"""Tests for AndCompositionVerifier k=5 ensemble and VerifyRepairPipeline wiring.

Spec: REQ-VERIFY-1121, SCENARIO-PHASE1D-001
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: stub carnot package __init__ files that transitively import JAX.
# We only need individual submodule files, not the full package init chain.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

for _pkg in ["carnot", "carnot.eval", "carnot.verify", "carnot.models", "carnot.pipeline"]:
    if _pkg not in sys.modules:
        _m = types.ModuleType(_pkg)
        _m.__path__ = [str(_PYTHON_DIR / _pkg.replace(".", "/"))]  # type: ignore[attr-defined]
        _m.__package__ = _pkg
        sys.modules[_pkg] = _m


from carnot.verify.and_composition_verifier import (  # noqa: E402
    AndCompositionResult,
    AndCompositionVerifier,
    ASTStructureAdapter,
    SemanticConsistencyAdapter,
    SemEnergyProbeAdapter,
    SOSKANEnergyV3Adapter,
    Z3MathAdapter,
    build_default_verifier_ensemble,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLEAN_TEXT = "The total cost is $47 + $28 = $75. The answer is 75."
_QUESTION = "What is 47 + 28?"


# ---------------------------------------------------------------------------
# test_k5_and_compose_default
# REQ-VERIFY-1121: pipeline uses k=5 ensemble when no verifier specified
# ---------------------------------------------------------------------------


def test_k5_and_compose_default():
    """build_default_verifier_ensemble() returns an AndCompositionVerifier with k=5.

    Spec: REQ-VERIFY-1121
    """
    ensemble = build_default_verifier_ensemble()
    assert isinstance(ensemble, AndCompositionVerifier)
    assert ensemble.k == 5


def test_k5_verifier_names():
    """The k=5 default ensemble contains exactly the 5 expected verifier names.

    Spec: REQ-VERIFY-1121
    """
    ensemble = build_default_verifier_ensemble()
    names = ensemble.verifier_names
    assert "SOSKANEnergyV3" in names
    assert "SemEnergyProbe" in names
    assert "ASTStructureVerifier" in names
    assert "SemanticConsistencyVerifier" in names
    assert "Z3MathVerifier" in names
    assert len(names) == 5


# ---------------------------------------------------------------------------
# test_thinkprm_not_in_and_compose
# REQ-VERIFY-1121: ThinkPRM must NOT appear in the AND-compose set
# ---------------------------------------------------------------------------


def test_thinkprm_not_in_and_compose():
    """ThinkPRM is excluded from the k=5 AND-compose ensemble.

    Reason: ThinkPRMProbe x Z3MathVerifier pairwise r=0.507 exceeds the 0.5
    viability threshold for exponential null-space shrinkage. Including it
    would degrade kernel-orthogonality.

    Spec: REQ-VERIFY-1121, SCENARIO-PHASE1D-001
    """
    ensemble = build_default_verifier_ensemble()
    names = ensemble.verifier_names
    assert "ThinkPRM" not in names
    assert "ThinkPRMProbe" not in names
    for name in names:
        assert "thinkprm" not in name.lower()


# ---------------------------------------------------------------------------
# test_and_compose_all_must_agree
# REQ-VERIFY-1121: AND requires all verifiers to agree (all below threshold)
# ---------------------------------------------------------------------------


def test_and_compose_all_must_agree():
    """AND-composition is True only when ALL verifiers return verified=True.

    We construct a minimal 2-verifier ensemble where one always passes and
    one always fails, and verify that AND=False. Then both passing → AND=True.

    Spec: REQ-VERIFY-1121
    """

    class AlwaysPassAdapter:
        @property
        def name(self) -> str:
            return "AlwaysPass"

        def score(self, text: str) -> float:
            return 0.0  # energy=0 < threshold=0.5 → verified

    class AlwaysFailAdapter:
        @property
        def name(self) -> str:
            return "AlwaysFail"

        def score(self, text: str) -> float:
            return 1.0  # energy=1 >= threshold=0.5 → not verified

    mixed = AndCompositionVerifier(verifiers=[AlwaysPassAdapter(), AlwaysFailAdapter()])
    result = mixed.verify("q", "r")
    assert result.verified is False, "AND must be False when any verifier fails"
    assert result.per_verifier_verified["AlwaysPass"] is True
    assert result.per_verifier_verified["AlwaysFail"] is False

    both_pass = AndCompositionVerifier(verifiers=[AlwaysPassAdapter(), AlwaysPassAdapter()])
    result2 = both_pass.verify("q", "r")
    assert result2.verified is True, "AND must be True when all verifiers pass"


def test_and_compose_result_records_per_verifier_scores():
    """AndCompositionResult.per_verifier_scores contains one entry per verifier.

    Spec: REQ-VERIFY-1121
    """
    ensemble = build_default_verifier_ensemble()
    result = ensemble.verify(_QUESTION, _CLEAN_TEXT)
    assert isinstance(result, AndCompositionResult)
    assert len(result.per_verifier_scores) == 5
    for name in ensemble.verifier_names:
        assert name in result.per_verifier_scores
        score = result.per_verifier_scores[name]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# test_benchmark_k5_vs_individual
# REQ-VERIFY-1121: benchmark runs without errors and produces expected fields
# ---------------------------------------------------------------------------


def test_benchmark_k5_vs_individual():
    """Benchmark scaffold runs without raising and returns numeric AUROC proxies.

    We use a tiny synthetic dataset (4 pairs) because the full FoVer holdout
    requires disk access. The test verifies that the benchmark code paths all
    execute without error and return float values in [0, 1].

    Spec: REQ-VERIFY-1121, SCENARIO-PHASE1D-001
    """
    import numpy as np

    from carnot.verify.and_composition_verifier import build_default_verifier_ensemble

    examples = [
        ("What is 2+2?", "The answer is 4."),
        ("What is 3+3?", "The answer is 6."),
        ("What is 5+5?", "The answer is 11. Wait no, 10."),  # inconsistent but fixable
        ("What is 7+7?", "7 + 7 = 15"),  # arithmetic error
    ]
    labels = np.array([1, 1, 0, 0], dtype=float)

    ensemble = build_default_verifier_ensemble()
    k5_scores = np.array([1.0 if ensemble.verify(q, r).verified else 0.0 for q, r in examples])

    # Individual best: Z3MathVerifier alone
    from carnot.verify.z3_math_verifier import Z3MathVerifier

    z3v = Z3MathVerifier()
    individual_scores = np.array([1.0 - z3v.score(r) for _, r in examples])

    # No-compose: random (use uniform)
    nocompose_scores = np.full(len(examples), 0.5)

    # Compute proxy AUROC via simple rank correlation (Wilcoxon statistic)
    def _proxy_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        count = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(
            1 for p in pos for n in neg if p == n
        )
        return float(count / (len(pos) * len(neg)))

    auroc_k5 = _proxy_auroc(k5_scores, labels)
    auroc_individual = _proxy_auroc(individual_scores, labels)
    auroc_nocompose = _proxy_auroc(nocompose_scores, labels)

    assert 0.0 <= auroc_k5 <= 1.0
    assert 0.0 <= auroc_individual <= 1.0
    assert 0.0 <= auroc_nocompose <= 1.0
