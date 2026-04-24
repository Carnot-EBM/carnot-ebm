"""Tests for Exp 797: JEPA v21 Multi-Source FOVER Data Collection.

Covers all pure helper functions — no GPU, no LLM, no filesystem side-effects.
Each test traces to REQ-LEARN-093 or REQ-LEARN-094.

Spec: REQ-LEARN-093, REQ-LEARN-094, SCENARIO-LEARN-144
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root and scripts dir are on sys.path for direct module import.
_REPO = Path(__file__).resolve().parents[2]
for _d in [str(_REPO / "python"), str(_REPO / "scripts")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import pytest

from scripts.experiment_797_jepa_v21_data_collection import (
    build_gsm8k_responses,
    build_humaneval_responses,
    build_math_responses,
    compute_honest_verdict,
    count_sources_with_data,
    merge_corpus_with_domain,
)


# ---------------------------------------------------------------------------
# compute_honest_verdict — REQ-LEARN-093
# ---------------------------------------------------------------------------


class TestComputeHonestVerdict:
    """Verdict logic covers all honest_verdict branches (REQ-LEARN-093)."""

    def test_adequate_when_80_labeled_and_2_sources(self) -> None:
        """REQ-LEARN-093: >= 80 pairs and >= 2 sources → multi_source_corpus_adequate."""
        assert compute_honest_verdict(80, 2) == "multi_source_corpus_adequate"

    def test_adequate_when_above_threshold_and_3_sources(self) -> None:
        """REQ-LEARN-093: 150 pairs and 3 sources → adequate."""
        assert compute_honest_verdict(150, 3) == "multi_source_corpus_adequate"

    def test_insufficient_when_below_80_but_2_sources(self) -> None:
        """REQ-LEARN-093: < 80 pairs but >= 2 sources → multi_source_insufficient."""
        assert compute_honest_verdict(50, 2) == "multi_source_insufficient"

    def test_insufficient_when_1_pair_and_2_sources(self) -> None:
        """REQ-LEARN-093: 1 pair from 2 sources → insufficient (not adequate)."""
        assert compute_honest_verdict(1, 2) == "multi_source_insufficient"

    def test_single_source_when_only_1_domain(self) -> None:
        """REQ-LEARN-093: exactly 1 source with data → single_source_only."""
        assert compute_honest_verdict(100, 1) == "single_source_only"

    def test_single_source_when_zero_labeled(self) -> None:
        """REQ-LEARN-093: 0 pairs and 0 sources → single_source_only (no data)."""
        assert compute_honest_verdict(0, 0) == "single_source_only"

    def test_exact_threshold_boundary_79_pairs(self) -> None:
        """REQ-LEARN-093: 79 pairs with 2 sources → insufficient (just below threshold)."""
        assert compute_honest_verdict(79, 2) == "multi_source_insufficient"


# ---------------------------------------------------------------------------
# merge_corpus_with_domain — REQ-LEARN-094
# ---------------------------------------------------------------------------


class TestMergeCorpusWithDomain:
    """Merged corpus must add source_domain per pair (REQ-LEARN-094)."""

    def test_source_domain_added_to_gsm8k_pairs(self) -> None:
        """REQ-LEARN-094: pairs from source A get source_domain='gsm8k'."""
        pairs_a = [{"question_id": "q1", "step_text": "1 + 2 = 3", "label": "correct", "confidence": 1.0}]
        merged = merge_corpus_with_domain(pairs_a, [], [])
        assert merged[0]["source_domain"] == "gsm8k"

    def test_source_domain_added_to_math500_pairs(self) -> None:
        """REQ-LEARN-094: pairs from source B get source_domain='math500'."""
        pairs_b = [{"question_id": "m1", "step_text": "3 * 4 = 12", "label": "correct", "confidence": 1.0}]
        merged = merge_corpus_with_domain([], pairs_b, [])
        assert merged[0]["source_domain"] == "math500"

    def test_source_domain_added_to_humaneval_pairs(self) -> None:
        """REQ-LEARN-094: pairs from source C get source_domain='humaneval'."""
        pairs_c = [{"question_id": "h1", "step_text": "5 + 5 = 10", "label": "correct", "confidence": 1.0}]
        merged = merge_corpus_with_domain([], [], pairs_c)
        assert merged[0]["source_domain"] == "humaneval"

    def test_all_original_fields_preserved(self) -> None:
        """REQ-LEARN-094: merging preserves all original pair fields."""
        pair = {"question_id": "q1", "step_text": "2 + 2 = 4", "label": "correct", "confidence": 1.0}
        merged = merge_corpus_with_domain([pair], [], [])
        assert merged[0]["question_id"] == "q1"
        assert merged[0]["step_text"] == "2 + 2 = 4"
        assert merged[0]["label"] == "correct"
        assert merged[0]["confidence"] == 1.0

    def test_total_length_is_sum_of_all_sources(self) -> None:
        """REQ-LEARN-094: merged list length = len(A) + len(B) + len(C)."""
        pairs_a = [{"question_id": f"a{i}", "step_text": "", "label": "correct", "confidence": 1.0} for i in range(3)]
        pairs_b = [{"question_id": f"b{i}", "step_text": "", "label": "correct", "confidence": 1.0} for i in range(2)]
        pairs_c = [{"question_id": f"c{i}", "step_text": "", "label": "correct", "confidence": 1.0} for i in range(4)]
        merged = merge_corpus_with_domain(pairs_a, pairs_b, pairs_c)
        assert len(merged) == 9

    def test_ordering_is_a_then_b_then_c(self) -> None:
        """REQ-LEARN-094: merged order is A pairs, then B pairs, then C pairs."""
        a = [{"question_id": "a0", "step_text": "", "label": "correct", "confidence": 1.0}]
        b = [{"question_id": "b0", "step_text": "", "label": "correct", "confidence": 1.0}]
        c = [{"question_id": "c0", "step_text": "", "label": "correct", "confidence": 1.0}]
        merged = merge_corpus_with_domain(a, b, c)
        domains = [p["source_domain"] for p in merged]
        assert domains == ["gsm8k", "math500", "humaneval"]

    def test_empty_inputs_produce_empty_output(self) -> None:
        """REQ-LEARN-094: merging three empty sources returns empty list."""
        assert merge_corpus_with_domain([], [], []) == []


# ---------------------------------------------------------------------------
# count_sources_with_data — REQ-LEARN-093
# ---------------------------------------------------------------------------


class TestCountSourcesWithData:
    """Source count logic (REQ-LEARN-093)."""

    def test_all_three_sources_have_data(self) -> None:
        """REQ-LEARN-093: all three non-zero → 3."""
        assert count_sources_with_data(10, 5, 20) == 3

    def test_two_sources_have_data(self) -> None:
        """REQ-LEARN-093: two non-zero → 2."""
        assert count_sources_with_data(10, 0, 20) == 2

    def test_one_source_has_data(self) -> None:
        """REQ-LEARN-093: one non-zero → 1."""
        assert count_sources_with_data(5, 0, 0) == 1

    def test_no_sources_have_data(self) -> None:
        """REQ-LEARN-093: all zero → 0."""
        assert count_sources_with_data(0, 0, 0) == 0


# ---------------------------------------------------------------------------
# build_gsm8k_responses — REQ-LEARN-093
# ---------------------------------------------------------------------------


class TestBuildGSM8KResponses:
    """GSM8K response builder (REQ-LEARN-093)."""

    def test_returns_correct_count(self) -> None:
        """REQ-LEARN-093: build_gsm8k_responses returns end-start dicts."""
        responses = build_gsm8k_responses(301, 361)
        assert len(responses) == 60

    def test_question_ids_are_gsm8k_prefixed(self) -> None:
        """REQ-LEARN-093: each response has question_id with gsm8k_ prefix."""
        responses = build_gsm8k_responses(301, 304)
        for r in responses:
            assert r["question_id"].startswith("gsm8k_")

    def test_responses_contain_arithmetic(self) -> None:
        """REQ-LEARN-093: each response contains an arithmetic equation."""
        responses = build_gsm8k_responses(301, 302)
        assert "=" in responses[0]["response"]

    def test_empty_range_returns_empty(self) -> None:
        """REQ-LEARN-093: start==end returns empty list."""
        assert build_gsm8k_responses(301, 301) == []


# ---------------------------------------------------------------------------
# build_math_responses — REQ-LEARN-093
# ---------------------------------------------------------------------------


class TestBuildMathResponses:
    """MATH-500 response builder (REQ-LEARN-093)."""

    def test_returns_same_count_as_problems(self) -> None:
        """REQ-LEARN-093: one response per problem."""
        problems = [{"question_id": f"m{i}", "question": "q", "answer": "a"} for i in range(5)]
        responses = build_math_responses(problems)
        assert len(responses) == 5

    def test_question_ids_are_preserved(self) -> None:
        """REQ-LEARN-093: question_id from problem is used in response."""
        problems = [{"question_id": "math500_007", "question": "q", "answer": "a"}]
        responses = build_math_responses(problems)
        assert responses[0]["question_id"] == "math500_007"

    def test_responses_contain_arithmetic(self) -> None:
        """REQ-LEARN-093: each math response has an inline equation."""
        problems = [{"question_id": "m0", "question": "q", "answer": "a"}]
        responses = build_math_responses(problems)
        assert "=" in responses[0]["response"]


# ---------------------------------------------------------------------------
# build_humaneval_responses — REQ-LEARN-093
# ---------------------------------------------------------------------------


class TestBuildHumanEvalResponses:
    """HumanEval response builder (REQ-LEARN-093)."""

    def test_returns_same_count_as_problems(self) -> None:
        """REQ-LEARN-093: one response per problem."""
        problems = [{"question_id": f"h{i}", "question": "q"} for i in range(4)]
        responses = build_humaneval_responses(problems)
        assert len(responses) == 4

    def test_question_ids_are_preserved(self) -> None:
        """REQ-LEARN-093: question_id from problem is used in response."""
        problems = [{"question_id": "humaneval_005", "question": "q"}]
        responses = build_humaneval_responses(problems)
        assert responses[0]["question_id"] == "humaneval_005"

    def test_responses_contain_arithmetic(self) -> None:
        """REQ-LEARN-093: each humaneval response has an inline equation."""
        problems = [{"question_id": "h0", "question": "q"}]
        responses = build_humaneval_responses(problems)
        assert "=" in responses[0]["response"]
