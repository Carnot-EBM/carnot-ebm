"""Tests for Experiment 1132: Goodfire Exemplar Cascade TP Rate.

Spec: REQ-VERIFY-1132, SCENARIO-VERIFY-1132 — measure per-tier TP rate of the
Carnot cascade against the named LLM-failure exemplar corpus produced by Exp
1112, and quantify the Z3MathVerifier engineering-tier differentiation.

Tests cover only the code added for this experiment:
- _tp_rate handles empty + non-empty inputs without ZeroDivisionError
- _aggregate computes per-tier and per-category rates from a stub corpus
- The honest_verdict classifier returns one of the four allowed strings
- The on-disk deliverable conforms to the goodfire_exemplar_cascade_tp_v1
  schema and contains every REQUIRED artifact field promised in the task
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import experiment_1132_goodfire_exemplar_cascade_tp as exp  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DELIVERABLE = _REPO_ROOT / "results" / "experiment_1132_goodfire_exemplar_cascade_tp.json"


_REQUIRED_FIELDS = [
    "experiment",
    "schema",
    "run_date",
    "n_exemplars_tested",
    "n_categories",
    "per_tier_tp_rate",
    "per_category_tp_rate",
    "z3_arithmetic_tp_rate",
    "semenergy_tp_rate",
    "goodfire_exemplar_tp_rate_measured",
    "per_tier_results_logged",
    "honest_verdict",
]

_REQUIRED_PER_TIER_KEYS = {
    "tier_0a_thinkprm",
    "tier_0c_semenergy",
    "tier_25_symcode",
    "tier_27_causal",
    "tier_3_k5",
}

_VALID_VERDICTS = {
    "z3_dominates_arithmetic",
    "learned_tiers_dominant",
    "mixed_results",
    "corpus_too_small",
}


def test_tp_rate_empty_returns_zero():
    """Empty input is degenerate — must return 0.0 instead of raising."""
    assert exp._tp_rate([]) == 0.0


def test_tp_rate_basic():
    """Three of four flags True ⟹ rate = 0.75."""
    assert exp._tp_rate([True, True, True, False]) == 0.75


def test_aggregate_corpus_too_small_under_10():
    """Verdict is 'corpus_too_small' when fewer than 10 exemplars are evaluated."""
    stub = [
        {
            "id": f"e{i}",
            "category": "arithmetic_comparison",
            "tier_results": {
                "tier_0a_thinkprm": False,
                "tier_0b_spilled": False,
                "tier_0c_semenergy": False,
                "tier_25_symcode": False,
                "tier_27_causal": False,
                "tier_3_k5": True,
                "z3_math_standalone": True,
            },
        }
        for i in range(3)
    ]
    agg = exp._aggregate(stub)
    assert agg["honest_verdict"] == "corpus_too_small"
    # All 3 are arithmetic ⟹ z3 fires on 100% of arithmetic.
    assert agg["z3_arithmetic_tp_rate"] == 1.0
    assert agg["per_tier_tp_rate"]["tier_3_k5"] == 1.0


def _make_stub(n: int, *, category: str, z3: bool, semenergy: bool):
    """Build n stub per-exemplar dicts with controlled flag patterns."""
    return [
        {
            "id": f"e{i}",
            "category": category,
            "tier_results": {
                "tier_0a_thinkprm": False,
                "tier_0b_spilled": False,
                "tier_0c_semenergy": semenergy,
                "tier_25_symcode": False,
                "tier_27_causal": False,
                "tier_3_k5": False,
                "z3_math_standalone": z3,
            },
        }
        for i in range(n)
    ]


def test_aggregate_z3_dominates_arithmetic():
    """When z3 fires on >70% of arithmetic and beats learned tiers ⟹ z3_dominates_arithmetic."""
    arith = _make_stub(15, category="arithmetic_comparison", z3=True, semenergy=False)
    agg = exp._aggregate(arith)
    assert agg["z3_arithmetic_tp_rate"] == 1.0
    assert agg["honest_verdict"] == "z3_dominates_arithmetic"


def test_aggregate_learned_tiers_dominant():
    """When SemEnergy fires more than z3 on arithmetic ⟹ learned_tiers_dominant."""
    arith = _make_stub(15, category="arithmetic_comparison", z3=False, semenergy=True)
    agg = exp._aggregate(arith)
    # SemEnergy beats z3 here (1.0 > 0.0).
    assert agg["honest_verdict"] == "learned_tiers_dominant"


def test_aggregate_mixed_results():
    """Equal but moderate rates ⟹ mixed_results (not dominance, not too small)."""
    # All flags False — z3 = max_learned = 0.0, ≥ 10 exemplars.
    flat = _make_stub(12, category="arithmetic_comparison", z3=False, semenergy=False)
    agg = exp._aggregate(flat)
    assert agg["honest_verdict"] == "mixed_results"


def test_deliverable_exists_and_validates():
    """The on-disk artifact must exist and contain every required field."""
    assert _DELIVERABLE.exists(), f"Missing deliverable: {_DELIVERABLE}"
    with _DELIVERABLE.open() as f:
        artifact = json.load(f)

    for field in _REQUIRED_FIELDS:
        assert field in artifact, f"Missing required field: {field}"

    assert artifact["experiment"] == 1132
    assert artifact["schema"] == "goodfire_exemplar_cascade_tp_v1"
    assert artifact["goodfire_exemplar_tp_rate_measured"] is True
    assert artifact["per_tier_results_logged"] is True
    assert artifact["honest_verdict"] in _VALID_VERDICTS

    per_tier = artifact["per_tier_tp_rate"]
    missing_tiers = _REQUIRED_PER_TIER_KEYS - set(per_tier.keys())
    assert not missing_tiers, f"Missing required tier keys: {missing_tiers}"

    # Every TP rate must be a float in [0, 1].
    for tier, rate in per_tier.items():
        assert isinstance(rate, float), f"Tier {tier} rate not float: {rate!r}"
        assert 0.0 <= rate <= 1.0, f"Tier {tier} rate out of range: {rate}"

    assert artifact["n_exemplars_tested"] >= 1
    assert artifact["n_categories"] >= 1
    assert 0.0 <= artifact["z3_arithmetic_tp_rate"] <= 1.0
    assert 0.0 <= artifact["semenergy_tp_rate"] <= 1.0


def test_deliverable_per_category_keys_subset_of_known():
    """Every per-category key must come from the known category enums."""
    with _DELIVERABLE.open() as f:
        artifact = json.load(f)
    known = exp.ARITHMETIC_CATEGORIES | exp.SEMANTIC_CATEGORIES | exp.CAUSAL_CATEGORIES
    for cat in artifact["per_category_tp_rate"]:
        assert cat in known, f"Unknown category in artifact: {cat}"
