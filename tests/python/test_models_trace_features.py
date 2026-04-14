"""Tests for feature extraction from verification traces.

Covers:
- TraceRecord: NamedTuple with features and label
- _majority: majority voting over constraint satisfaction flags
- extract_constraint_ir_features: feature vector extraction from case dict
- load_constraint_ir_traces: JSON deserialization
- auroc_score: AUROC via Wilcoxon-Mann-Whitney U

Spec: REQ-CORE-001, REQ-CORE-002
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from python.carnot.models.trace_features import (
    FEATURE_DIM,
    TraceRecord,
    _majority,
    auroc_score,
    extract_constraint_ir_features,
    load_constraint_ir_traces,
)


# ---------------------------------------------------------------------------
# TraceRecord
# ---------------------------------------------------------------------------


class TestTraceRecord:
    """TraceRecord NamedTuple must hold features and label."""

    def test_trace_record_has_features_and_label(self) -> None:
        """TraceRecord can be instantiated with features and label."""
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        label = 1.0
        record = TraceRecord(features=features, label=label)
        assert record.features.shape == (FEATURE_DIM,)
        assert record.label == 1.0

    def test_trace_record_label_is_float(self) -> None:
        """TraceRecord label can be 0.0 or 1.0."""
        features = np.ones(FEATURE_DIM, dtype=np.float32)
        record_correct = TraceRecord(features=features, label=1.0)
        record_wrong = TraceRecord(features=features, label=0.0)
        assert record_correct.label == 1.0
        assert record_wrong.label == 0.0


# ---------------------------------------------------------------------------
# _majority
# ---------------------------------------------------------------------------


class TestMajority:
    """_majority must vote on constraint satisfaction flags."""

    def test_majority_empty_list_returns_zero(self) -> None:
        """_majority([]) returns 0.0 (no evidence)."""
        assert _majority([]) == 0.0

    def test_majority_single_true_returns_one(self) -> None:
        """_majority([True]) returns 1.0."""
        assert _majority([True]) == 1.0

    def test_majority_single_false_returns_zero(self) -> None:
        """_majority([False]) returns 0.0."""
        assert _majority([False]) == 0.0

    def test_majority_two_true_one_false_returns_one(self) -> None:
        """_majority with 2 True, 1 False returns 1.0 (majority True)."""
        assert _majority([True, True, False]) == 1.0

    def test_majority_one_true_two_false_returns_zero(self) -> None:
        """_majority with 1 True, 2 False returns 0.0 (majority False)."""
        assert _majority([True, False, False]) == 0.0

    def test_majority_equal_split_returns_zero(self) -> None:
        """_majority with equal split [True, False] returns 0.0 (not strictly > 0.5)."""
        assert _majority([True, False]) == 0.0

    def test_majority_returns_float(self) -> None:
        """_majority always returns a float."""
        assert isinstance(_majority([True]), float)
        assert isinstance(_majority([]), float)


# ---------------------------------------------------------------------------
# extract_constraint_ir_features
# ---------------------------------------------------------------------------


class TestExtractConstraintIrFeatures:
    """extract_constraint_ir_features must extract 13 binary features."""

    def test_empty_case_returns_zero_vector(self) -> None:
        """Empty case dict returns all-zero feature vector."""
        case: dict[str, Any] = {}
        features = extract_constraint_ir_features(case)
        assert features.shape == (FEATURE_DIM,)
        assert np.array_equal(features, np.zeros(FEATURE_DIM, dtype=np.float32))

    def test_all_constraints_satisfied(self) -> None:
        """Case with all constraints satisfied has features with majority = 1.0."""
        case: dict[str, Any] = {
            "evaluation": {
                "constraint_results": [
                    {"family": "literal", "status": "satisfied"},
                    {"family": "literal", "status": "satisfied"},
                    {"family": "search_optimization_limited", "status": "satisfied"},
                    {"family": "semantic", "status": "satisfied"},
                ]
            },
            "constraint_extraction_coverage": 1.0,
            "partial_satisfaction": 1.0,
            "semantic_violation_count": 0,
            "mode": "verify_repair",
            "output_style": "code_only",
        }
        features = extract_constraint_ir_features(case, model_is_gemma=False)
        # Features: [literal_maj, search_opt_maj, semantic_maj, cov_75, cov_perf, part_50, part_75, no_viol, verify_only, verify_repair, code_only, prose, gemma]
        # = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        assert features[0] == 1.0  # literal majority satisfied
        assert features[1] == 1.0  # search_opt majority satisfied
        assert features[2] == 1.0  # semantic majority satisfied
        assert features[3] == 1.0  # coverage > 0.75
        assert features[4] == 1.0  # coverage == 1.0
        assert features[5] == 1.0  # partial > 0.50
        assert features[6] == 1.0  # partial > 0.75
        assert features[7] == 1.0  # no semantic violations
        assert features[8] == 0.0  # not verify_only
        assert features[9] == 1.0  # verify_repair mode
        assert features[10] == 1.0 # code_only style
        assert features[11] == 0.0 # not text_prose style
        assert features[12] == 0.0 # not gemma

    def test_coverage_boundary_75(self) -> None:
        """Coverage exactly at 0.75 is not > 0.75."""
        case: dict[str, Any] = {"constraint_extraction_coverage": 0.75}
        features = extract_constraint_ir_features(case)
        assert features[3] == 0.0  # coverage_above_75

    def test_coverage_above_75(self) -> None:
        """Coverage > 0.75 sets feature to 1.0."""
        case: dict[str, Any] = {"constraint_extraction_coverage": 0.76}
        features = extract_constraint_ir_features(case)
        assert features[3] == 1.0

    def test_partial_satisfaction_boundary_50(self) -> None:
        """Partial satisfaction exactly at 0.50 is not > 0.50."""
        case: dict[str, Any] = {"partial_satisfaction": 0.50}
        features = extract_constraint_ir_features(case)
        assert features[5] == 0.0

    def test_partial_satisfaction_above_50(self) -> None:
        """Partial satisfaction > 0.50 sets feature to 1.0."""
        case: dict[str, Any] = {"partial_satisfaction": 0.51}
        features = extract_constraint_ir_features(case)
        assert features[5] == 1.0

    def test_mode_verify_only(self) -> None:
        """Mode 'verify_only' sets feature 8 to 1.0."""
        case: dict[str, Any] = {"mode": "verify_only"}
        features = extract_constraint_ir_features(case)
        assert features[8] == 1.0
        assert features[9] == 0.0

    def test_mode_verify_repair(self) -> None:
        """Mode 'verify_repair' sets feature 9 to 1.0."""
        case: dict[str, Any] = {"mode": "verify_repair"}
        features = extract_constraint_ir_features(case)
        assert features[8] == 0.0
        assert features[9] == 1.0

    def test_style_code_only(self) -> None:
        """Output style 'code_only' sets feature 10 to 1.0."""
        case: dict[str, Any] = {"output_style": "code_only"}
        features = extract_constraint_ir_features(case)
        assert features[10] == 1.0
        assert features[11] == 0.0

    def test_style_text_prose(self) -> None:
        """Output style 'text_prose' sets feature 11 to 1.0."""
        case: dict[str, Any] = {"output_style": "text_prose"}
        features = extract_constraint_ir_features(case)
        assert features[10] == 0.0
        assert features[11] == 1.0

    def test_model_is_gemma_true(self) -> None:
        """model_is_gemma=True sets feature 12 to 1.0."""
        case: dict[str, Any] = {}
        features = extract_constraint_ir_features(case, model_is_gemma=True)
        assert features[12] == 1.0

    def test_model_is_gemma_false(self) -> None:
        """model_is_gemma=False sets feature 12 to 0.0."""
        case: dict[str, Any] = {}
        features = extract_constraint_ir_features(case, model_is_gemma=False)
        assert features[12] == 0.0

    def test_semantic_violations_count(self) -> None:
        """semantic_violation_count == 0 sets feature 7 to 1.0."""
        case_no_violations: dict[str, Any] = {"semantic_violation_count": 0}
        case_with_violations: dict[str, Any] = {"semantic_violation_count": 1}
        features_no = extract_constraint_ir_features(case_no_violations)
        features_with = extract_constraint_ir_features(case_with_violations)
        assert features_no[7] == 1.0
        assert features_with[7] == 0.0

    def test_features_dtype_is_float32(self) -> None:
        """Returned features array is float32."""
        case: dict[str, Any] = {}
        features = extract_constraint_ir_features(case)
        assert features.dtype == np.float32


# ---------------------------------------------------------------------------
# load_constraint_ir_traces
# ---------------------------------------------------------------------------


class TestLoadConstraintIrTraces:
    """load_constraint_ir_traces must deserialize JSON and yield TraceRecords."""

    def test_empty_paired_runs_returns_empty_list(self) -> None:
        """JSON with empty paired_runs returns empty list."""
        data = {"paired_runs": []}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            fpath = f.name
        try:
            records = load_constraint_ir_traces(fpath)
            assert records == []
        finally:
            Path(fpath).unlink()

    def test_single_run_single_case(self) -> None:
        """Single run with one case yields one TraceRecord."""
        data = {
            "paired_runs": [
                {
                    "model_name": "qwen",
                    "cases": [
                        {
                            "evaluation": {"constraint_results": []},
                            "constraint_extraction_coverage": 0.5,
                            "exact_satisfaction": True,
                        }
                    ],
                }
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            fpath = f.name
        try:
            records = load_constraint_ir_traces(fpath)
            assert len(records) == 1
            assert records[0].label == 1.0
        finally:
            Path(fpath).unlink()

    def test_gemma_model_sets_feature_12(self) -> None:
        """Run with model_name containing 'gemma' sets feature 12 to 1.0."""
        data = {
            "paired_runs": [
                {
                    "model_name": "Gemma4-E4B-IT",
                    "cases": [
                        {
                            "evaluation": {"constraint_results": []},
                            "exact_satisfaction": False,
                        }
                    ],
                }
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            fpath = f.name
        try:
            records = load_constraint_ir_traces(fpath)
            assert records[0].features[12] == 1.0
        finally:
            Path(fpath).unlink()

    def test_multiple_runs_preserves_order(self) -> None:
        """Multiple runs appear in order: all cases from run 0, then run 1."""
        data = {
            "paired_runs": [
                {
                    "model_name": "qwen",
                    "cases": [
                        {"evaluation": {"constraint_results": []}, "exact_satisfaction": True},
                        {"evaluation": {"constraint_results": []}, "exact_satisfaction": False},
                    ],
                },
                {
                    "model_name": "gemma",
                    "cases": [
                        {"evaluation": {"constraint_results": []}, "exact_satisfaction": True},
                    ],
                },
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            fpath = f.name
        try:
            records = load_constraint_ir_traces(fpath)
            assert len(records) == 3
            assert records[0].label == 1.0
            assert records[1].label == 0.0
            assert records[2].label == 1.0
            assert records[2].features[12] == 1.0  # Third record from gemma run
        finally:
            Path(fpath).unlink()


# ---------------------------------------------------------------------------
# auroc_score
# ---------------------------------------------------------------------------


class TestAurocScore:
    """auroc_score must compute AUROC via Wilcoxon-Mann-Whitney U."""

    def test_empty_arrays_returns_half(self) -> None:
        """Empty arrays return 0.5 (undefined)."""
        assert auroc_score(np.array([]), np.array([])) == 0.5

    def test_empty_correct_returns_half(self) -> None:
        """Empty energies_correct returns 0.5."""
        assert auroc_score(np.array([]), np.array([1.0, 2.0])) == 0.5

    def test_empty_wrong_returns_half(self) -> None:
        """Empty energies_wrong returns 0.5."""
        assert auroc_score(np.array([1.0, 2.0]), np.array([])) == 0.5

    def test_perfect_separation(self) -> None:
        """All correct energies < all wrong energies yields 1.0 AUROC."""
        correct = np.array([1.0, 2.0, 3.0])
        wrong = np.array([4.0, 5.0, 6.0])
        score = auroc_score(correct, wrong)
        assert score == 1.0

    def test_complete_reversal(self) -> None:
        """All correct energies > all wrong energies yields 0.0 AUROC."""
        correct = np.array([4.0, 5.0, 6.0])
        wrong = np.array([1.0, 2.0, 3.0])
        score = auroc_score(correct, wrong)
        assert score == 0.0

    def test_random_chance(self) -> None:
        """Identical distributions yield ~0.5 AUROC."""
        correct = np.array([1.0, 2.0, 3.0])
        wrong = np.array([1.0, 2.0, 3.0])
        score = auroc_score(correct, wrong)
        assert score == 0.5

    def test_ties_award_half_point(self) -> None:
        """When energies are equal, award 0.5 per tie."""
        correct = np.array([2.0])
        wrong = np.array([2.0])
        score = auroc_score(correct, wrong)
        # 1 pair: 1 tie → 0.5 wins / 1 pair = 0.5
        assert score == 0.5

    def test_mixed_ordering(self) -> None:
        """Partial overlap yields AUROC between 0 and 1."""
        correct = np.array([2.0, 3.0])
        wrong = np.array([1.0, 4.0])
        # Pairs: (2,1) → win, (2,4) → loss, (3,1) → win, (3,4) → loss
        # 2 wins / 4 pairs = 0.5
        score = auroc_score(correct, wrong)
        assert score == 0.5

    def test_auroc_returns_float(self) -> None:
        """auroc_score returns a float."""
        result = auroc_score(np.array([1.0]), np.array([2.0]))
        assert isinstance(result, (float, np.floating))
