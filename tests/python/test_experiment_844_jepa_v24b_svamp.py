"""Tests for Experiment 844: JEPA v24b SVAMP corpus coverage fix.

Traces to: REQ-LEARN-010, REQ-LEARN-020, SCENARIO-LEARN-015, SCENARIO-LEARN-020

REQ-LEARN-010: Constraint addition from CaseMemory patterns.
REQ-LEARN-020: JEPA training MUST assert coverage >= 15 pairs per domain.
SCENARIO-LEARN-015: extract_patterns groups CaseMemory by violation family.
SCENARIO-LEARN-020: Training with 0 SVAMP pairs fires assertion with diagnostic message.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure repo root is on PYTHONPATH
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_844_jepa_v24b_svamp import (
    DOMAIN_NAMES,
    DREAM_PRM_WEIGHTS_V24B,
    MIN_PAIRS_PER_DOMAIN,
    SVAMP_TRIPLETS_RAW,
    _eval_op,
    assert_domain_coverage,
    build_corpus_v24b,
    compute_honest_verdict_v24b,
    verify_and_build_svamp_triplets,
)


# ---------------------------------------------------------------------------
# REQ-LEARN-020 / SCENARIO-LEARN-020: domain coverage assertion
# ---------------------------------------------------------------------------


class TestDomainCoverageAssertion:
    """REQ-LEARN-020: assert_domain_coverage fires when any domain is below threshold."""

    def test_svamp_assertion_fires_if_zero_triplets(self) -> None:
        """SCENARIO-LEARN-020: 0 SVAMP pairs triggers AssertionError with diagnostic."""
        with pytest.raises(AssertionError, match="SVAMP coverage insufficient"):
            assert_domain_coverage(
                n_gsm8k=40,
                n_humaneval=40,
                n_arc=40,
                n_svamp=0,
            )

    def test_arc_assertion_fires(self) -> None:
        """SCENARIO-LEARN-020: 0 ARC pairs triggers AssertionError naming ARC."""
        with pytest.raises(AssertionError, match="ARC coverage insufficient"):
            assert_domain_coverage(
                n_gsm8k=40,
                n_humaneval=40,
                n_arc=0,
                n_svamp=40,
            )

    def test_humaneval_assertion_fires(self) -> None:
        """SCENARIO-LEARN-020: 0 HumanEval pairs triggers AssertionError naming HumanEval."""
        with pytest.raises(AssertionError, match="HumanEval coverage insufficient"):
            assert_domain_coverage(
                n_gsm8k=40,
                n_humaneval=0,
                n_arc=40,
                n_svamp=40,
            )

    def test_gsm8k_assertion_fires(self) -> None:
        """SCENARIO-LEARN-020: 0 GSM8K pairs triggers AssertionError naming GSM8K."""
        with pytest.raises(AssertionError, match="GSM8K coverage insufficient"):
            assert_domain_coverage(
                n_gsm8k=0,
                n_humaneval=40,
                n_arc=40,
                n_svamp=40,
            )

    def test_all_domains_covered_passes(self) -> None:
        """SCENARIO-LEARN-020: When all domains >= MIN_PAIRS, assertion passes silently."""
        # Should not raise
        assert_domain_coverage(
            n_gsm8k=MIN_PAIRS_PER_DOMAIN,
            n_humaneval=MIN_PAIRS_PER_DOMAIN,
            n_arc=MIN_PAIRS_PER_DOMAIN,
            n_svamp=MIN_PAIRS_PER_DOMAIN,
        )

    def test_below_minimum_fires(self) -> None:
        """SCENARIO-LEARN-020: n_svamp = MIN_PAIRS - 1 fires assertion."""
        with pytest.raises(AssertionError, match="SVAMP coverage insufficient"):
            assert_domain_coverage(
                n_gsm8k=40,
                n_humaneval=40,
                n_arc=40,
                n_svamp=MIN_PAIRS_PER_DOMAIN - 1,
            )

    def test_assertion_message_includes_count(self) -> None:
        """Diagnostic message should include the actual count to help debugging."""
        with pytest.raises(AssertionError, match="5 pairs"):
            assert_domain_coverage(
                n_gsm8k=40,
                n_humaneval=40,
                n_arc=40,
                n_svamp=5,
            )


# ---------------------------------------------------------------------------
# SVAMP triplet symbolic verification
# ---------------------------------------------------------------------------


class TestSvampTripletVerification:
    """Symbolic arithmetic verification for SVAMP triplets."""

    def test_all_20_triplets_verified(self) -> None:
        """All 20 raw SVAMP triplets pass symbolic arithmetic check."""
        validated = verify_and_build_svamp_triplets(SVAMP_TRIPLETS_RAW)
        assert len(validated) == 20

    def test_triplet_has_required_keys(self) -> None:
        """Each validated triplet has anchor, positive, negative, domain."""
        validated = verify_and_build_svamp_triplets(SVAMP_TRIPLETS_RAW)
        for t in validated:
            assert set(t.keys()) == {"anchor", "positive", "negative", "domain"}

    def test_triplet_domain_is_svamp(self) -> None:
        """All validated triplets are labelled domain='svamp'."""
        validated = verify_and_build_svamp_triplets(SVAMP_TRIPLETS_RAW)
        for t in validated:
            assert t["domain"] == "svamp"

    def test_bad_correct_answer_raises(self) -> None:
        """A triplet with wrong 'correct' value triggers AssertionError."""
        bad = [{"op": "add", "a": 3, "b": 4, "correct": 999, "wrong": 8,
                "anchor": "x", "positive": "y", "negative": "z"}]
        with pytest.raises(AssertionError, match=r"anchor\+positive inconsistent"):
            verify_and_build_svamp_triplets(bad)

    def test_bad_wrong_answer_matches_correct_raises(self) -> None:
        """A triplet where 'wrong' accidentally equals the correct result raises."""
        bad = [{"op": "add", "a": 3, "b": 4, "correct": 7, "wrong": 7,
                "anchor": "x", "positive": "y", "negative": "z"}]
        with pytest.raises(AssertionError, match=r"anchor\+negative is actually correct"):
            verify_and_build_svamp_triplets(bad)

    def test_eval_op_add(self) -> None:
        """_eval_op('add', a, b) == a + b."""
        assert _eval_op("add", 10, 5) == 15

    def test_eval_op_sub(self) -> None:
        """_eval_op('sub', a, b) == a - b."""
        assert _eval_op("sub", 10, 3) == 7

    def test_eval_op_mul(self) -> None:
        """_eval_op('mul', a, b) == a * b."""
        assert _eval_op("mul", 4, 5) == 20

    def test_eval_op_div(self) -> None:
        """_eval_op('div', a, b) == a // b for exact integer division."""
        assert _eval_op("div", 20, 4) == 5

    def test_eval_op_div_remainder_raises(self) -> None:
        """_eval_op('div', a, b) raises when b does not divide a evenly."""
        with pytest.raises(ValueError, match="not evenly divisible"):
            _eval_op("div", 7, 3)

    def test_eval_op_unknown_raises(self) -> None:
        """_eval_op with unknown op string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown op"):
            _eval_op("pow", 2, 3)


# ---------------------------------------------------------------------------
# Domain weights applied correctly
# ---------------------------------------------------------------------------


class TestDomainWeightsApplied:
    """REQ-LEARN-020: DreamPRM weights reflect domain deficit (SVAMP=8.0)."""

    def test_svamp_weight_is_maximum(self) -> None:
        """SVAMP weight must be 8.0 — maximum-deficit domain from Exp 834."""
        assert DREAM_PRM_WEIGHTS_V24B["svamp"] == 8.0

    def test_gsm8k_weight_is_baseline(self) -> None:
        """GSM8K is baseline in-distribution domain, weight == 1.0."""
        assert DREAM_PRM_WEIGHTS_V24B["gsm8k"] == 1.0

    def test_arc_weight_reduced_from_v24(self) -> None:
        """ARC weight must be <= 1.5; v24 used 5.0 but ARC recovered (auc=0.72)."""
        assert DREAM_PRM_WEIGHTS_V24B["arc"] <= 1.5

    def test_all_domains_have_weights(self) -> None:
        """All four domain names have an entry in DREAM_PRM_WEIGHTS_V24B."""
        for d in DOMAIN_NAMES:
            assert d in DREAM_PRM_WEIGHTS_V24B
            assert DREAM_PRM_WEIGHTS_V24B[d] > 0


# ---------------------------------------------------------------------------
# All domains covered in built corpus
# ---------------------------------------------------------------------------


class TestAllDomainsCovered:
    """After building corpus v24b, all four domains must have >= MIN_PAIRS_PER_DOMAIN pairs."""

    def test_corpus_contains_svamp_pairs(self) -> None:
        """Corpus must have SVAMP pairs from the 20 verified triplets."""
        validated = verify_and_build_svamp_triplets(SVAMP_TRIPLETS_RAW)
        corpus = build_corpus_v24b(validated)
        svamp_pairs = [p for p in corpus if p["domain"] == "svamp"]
        assert len(svamp_pairs) >= MIN_PAIRS_PER_DOMAIN

    def test_corpus_svamp_count_is_40(self) -> None:
        """20 triplets × 2 (positive + negative) = 40 SVAMP pairs."""
        validated = verify_and_build_svamp_triplets(SVAMP_TRIPLETS_RAW)
        corpus = build_corpus_v24b(validated)
        svamp_pairs = [p for p in corpus if p["domain"] == "svamp"]
        assert len(svamp_pairs) == 40

    def test_corpus_all_domains_present(self) -> None:
        """All four domains appear in the corpus."""
        validated = verify_and_build_svamp_triplets(SVAMP_TRIPLETS_RAW)
        corpus = build_corpus_v24b(validated)
        domains_in_corpus = {p["domain"] for p in corpus}
        assert domains_in_corpus == set(DOMAIN_NAMES)

    def test_corpus_labels_are_binary(self) -> None:
        """All corpus labels are 0 or 1."""
        validated = verify_and_build_svamp_triplets(SVAMP_TRIPLETS_RAW)
        corpus = build_corpus_v24b(validated)
        for p in corpus:
            assert p["label"] in (0, 1)

    def test_empty_svamp_triplets_triggers_coverage_assertion(self) -> None:
        """Passing zero SVAMP triplets to build_corpus_v24b causes coverage assertion."""
        with pytest.raises(AssertionError, match="SVAMP coverage insufficient"):
            build_corpus_v24b([])


# ---------------------------------------------------------------------------
# Honest verdict logic
# ---------------------------------------------------------------------------


class TestHonestVerdictLogic:
    """Verdict function maps per-domain AUC to the correct string."""

    def test_all_domains_viable_when_min_high(self) -> None:
        """All domains >= 0.50 AND ood_auc >= 0.65 → jepa_v24b_all_domains_viable."""
        verdict = compute_honest_verdict_v24b(
            auc_gsm8k=0.80,
            auc_humaneval=0.75,
            auc_arc=0.72,
            auc_svamp=0.65,
        )
        assert verdict == "jepa_v24b_all_domains_viable"

    def test_svamp_fixed_when_svamp_above_40(self) -> None:
        """auc_svamp >= 0.40 but min_domain_auc < 0.50 → jepa_v24b_svamp_fixed."""
        verdict = compute_honest_verdict_v24b(
            auc_gsm8k=0.80,
            auc_humaneval=0.75,
            auc_arc=0.72,
            auc_svamp=0.45,  # above 0.40 but below 0.50
        )
        assert verdict == "jepa_v24b_svamp_fixed"

    def test_svamp_still_collapsed_when_below_40(self) -> None:
        """auc_svamp < 0.40 → jepa_v24b_svamp_still_collapsed."""
        verdict = compute_honest_verdict_v24b(
            auc_gsm8k=0.80,
            auc_humaneval=0.75,
            auc_arc=0.72,
            auc_svamp=0.0,
        )
        assert verdict == "jepa_v24b_svamp_still_collapsed"

    def test_svamp_at_exactly_040_is_fixed(self) -> None:
        """auc_svamp == 0.40 is treated as 'fixed' (boundary inclusive)."""
        verdict = compute_honest_verdict_v24b(
            auc_gsm8k=0.80,
            auc_humaneval=0.75,
            auc_arc=0.72,
            auc_svamp=0.40,
        )
        assert verdict == "jepa_v24b_svamp_fixed"

    def test_ood_auc_excludes_gsm8k(self) -> None:
        """OOD average is mean(humaneval, arc, svamp) — not including gsm8k.

        When humaneval=0.50, arc=0.50, svamp=0.50, ood_auc=0.50 < 0.65
        even though gsm8k=0.99.  The verdict should NOT be all_domains_viable
        because min_domain_auc < 0.50 (all three ood domains at border).
        """
        verdict = compute_honest_verdict_v24b(
            auc_gsm8k=0.99,
            auc_humaneval=0.50,
            auc_arc=0.50,
            auc_svamp=0.50,
        )
        # min_domain_auc = 0.50 (== threshold), ood_auc = 0.50 (< 0.65)
        # svamp = 0.50 >= 0.40 → svamp_fixed
        assert verdict == "jepa_v24b_svamp_fixed"

    def test_result_json_exists_and_has_required_fields(self) -> None:
        """The deliverable JSON (if present) must contain all required schema fields."""
        result_path = _REPO_ROOT / "results" / "experiment_844_jepa_v24b_svamp.json"
        if not result_path.exists():
            pytest.skip("Deliverable not yet generated — run the script first.")
        data = json.loads(result_path.read_text())
        required = [
            "experiment",
            "auc_gsm8k",
            "auc_humaneval",
            "auc_arc",
            "auc_svamp",
            "overall_ood_auc",
            "min_domain_auc",
            "all_domains_coverage",
            "honest_verdict",
            "checkpoint_path",
        ]
        for field in required:
            assert field in data, f"Missing required field: {field}"
        assert data["all_domains_coverage"] is True
        assert data["honest_verdict"] in {
            "jepa_v24b_all_domains_viable",
            "jepa_v24b_svamp_fixed",
            "jepa_v24b_svamp_still_collapsed",
        }
