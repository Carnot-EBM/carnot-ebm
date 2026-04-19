"""Tests for PPSConstraintLearner (PPSEBM, arXiv 2512.15658).

Spec: REQ-SELFLEARN-016, REQ-SELFLEARN-017, REQ-SELFLEARN-018,
      SCENARIO-SELFLEARN-016, SCENARIO-SELFLEARN-017, SCENARIO-SELFLEARN-018
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.pipeline.lsebm_replayer import LSEBMConstraintReplayer
from carnot.pipeline.pps_constraint_learner import (
    ConstraintDomain,
    DomainParameterPartition,
    PartitionIsolationScore,
    PPSConstraintLearner,
)


# ---------------------------------------------------------------------------
# ConstraintDomain tests
# ---------------------------------------------------------------------------


class TestConstraintDomain:
    """REQ-SELFLEARN-016: domain enum has ARITHMETIC, CODE, LOGICAL."""

    def test_enum_values(self):
        assert ConstraintDomain.ARITHMETIC.value == "arithmetic"
        assert ConstraintDomain.CODE.value == "code"
        assert ConstraintDomain.LOGICAL.value == "logical"

    def test_all_three_domains_exist(self):
        domains = list(ConstraintDomain)
        assert len(domains) == 3


# ---------------------------------------------------------------------------
# DomainParameterPartition tests
# ---------------------------------------------------------------------------


class TestDomainParameterPartition:
    """REQ-SELFLEARN-016, REQ-SELFLEARN-018."""

    def _make_partition(self, domain=ConstraintDomain.ARITHMETIC, dim=8):
        weights = np.zeros(dim, dtype=np.float64)
        return DomainParameterPartition(domain=domain, weights=weights)

    def test_initial_cumulative_gradient_is_zero(self):
        p = self._make_partition()
        assert np.allclose(p.gradient_direction(), np.zeros(8))

    def test_update_modifies_weights(self):
        # REQ-SELFLEARN-016: update() changes this partition's weights.
        p = self._make_partition()
        grad = np.ones(8)
        p.update(grad)
        assert not np.allclose(p.weights, np.zeros(8))

    def test_update_accumulates_gradient(self):
        p = self._make_partition()
        p.update(np.ones(8))
        p.update(np.ones(8))
        # cumulative gradient should be 2 * ones
        assert np.allclose(p._cumulative_gradient, np.full(8, 2.0))

    def test_gradient_direction_is_unit_vector(self):
        p = self._make_partition()
        p.update(np.array([3.0, 4.0] + [0.0] * 6))
        d = p.gradient_direction()
        assert abs(np.linalg.norm(d) - 1.0) < 1e-10

    def test_gradient_direction_zero_before_update(self):
        p = self._make_partition()
        d = p.gradient_direction()
        assert np.allclose(d, np.zeros(8))

    def test_weights_copy_on_init(self):
        # The partition must copy weights; external mutation must not affect it.
        w = np.ones(4)
        p = DomainParameterPartition(domain=ConstraintDomain.CODE, weights=w)
        w[0] = 999.0
        assert p.weights[0] == 1.0

    def test_domain_attribute(self):
        p = DomainParameterPartition(domain=ConstraintDomain.LOGICAL, weights=np.zeros(4))
        assert p.domain == ConstraintDomain.LOGICAL


# ---------------------------------------------------------------------------
# PartitionIsolationScore tests
# ---------------------------------------------------------------------------


class TestPartitionIsolationScore:
    """REQ-SELFLEARN-018: cosine-distance isolation score."""

    def _make_updated_partition(self, domain, direction):
        p = DomainParameterPartition(domain=domain, weights=np.zeros(len(direction)))
        p.update(np.array(direction, dtype=np.float64))
        return p

    def test_score_one_partition(self):
        # Only one non-zero gradient — no pairs to compare, so score = 1.0.
        p = self._make_updated_partition(ConstraintDomain.ARITHMETIC, [1.0, 0.0, 0.0, 0.0])
        pis = PartitionIsolationScore([p])
        assert pis.score() == 1.0

    def test_score_orthogonal_partitions(self):
        # REQ-SELFLEARN-018: orthogonal gradients → cosine distance = 1.0.
        p1 = self._make_updated_partition(ConstraintDomain.ARITHMETIC, [1.0, 0.0, 0.0, 0.0])
        p2 = self._make_updated_partition(ConstraintDomain.CODE, [0.0, 1.0, 0.0, 0.0])
        pis = PartitionIsolationScore([p1, p2])
        assert abs(pis.score() - 1.0) < 1e-10

    def test_score_identical_partitions(self):
        # Same gradient direction → cosine distance = 0.0 (no isolation).
        p1 = self._make_updated_partition(ConstraintDomain.ARITHMETIC, [1.0, 0.0, 0.0, 0.0])
        p2 = self._make_updated_partition(ConstraintDomain.CODE, [1.0, 0.0, 0.0, 0.0])
        pis = PartitionIsolationScore([p1, p2])
        assert pis.score() < 1e-10

    def test_is_isolated_true_when_above_threshold(self):
        p1 = self._make_updated_partition(ConstraintDomain.ARITHMETIC, [1.0, 0.0, 0.0, 0.0])
        p2 = self._make_updated_partition(ConstraintDomain.CODE, [0.0, 1.0, 0.0, 0.0])
        pis = PartitionIsolationScore([p1, p2])
        assert pis.is_isolated(threshold=0.8) is True

    def test_is_isolated_false_when_below_threshold(self):
        p1 = self._make_updated_partition(ConstraintDomain.ARITHMETIC, [1.0, 0.0, 0.0, 0.0])
        p2 = self._make_updated_partition(ConstraintDomain.CODE, [1.0, 0.0, 0.0, 0.0])
        pis = PartitionIsolationScore([p1, p2])
        assert pis.is_isolated(threshold=0.8) is False

    def test_score_three_partitions_min_distance(self):
        # Three orthogonal vectors: min pairwise cosine distance should be 1.0.
        p1 = self._make_updated_partition(ConstraintDomain.ARITHMETIC, [1.0, 0.0, 0.0])
        p2 = self._make_updated_partition(ConstraintDomain.CODE, [0.0, 1.0, 0.0])
        p3 = self._make_updated_partition(ConstraintDomain.LOGICAL, [0.0, 0.0, 1.0])
        pis = PartitionIsolationScore([p1, p2, p3])
        assert abs(pis.score() - 1.0) < 1e-10

    def test_score_zero_gradient_partitions_excluded(self):
        # Partitions with zero gradient are excluded from scoring.
        p1 = DomainParameterPartition(domain=ConstraintDomain.ARITHMETIC, weights=np.zeros(4))
        p2 = self._make_updated_partition(ConstraintDomain.CODE, [1.0, 0.0, 0.0, 0.0])
        pis = PartitionIsolationScore([p1, p2])
        # Only one non-zero gradient — returns 1.0.
        assert pis.score() == 1.0

    def test_is_isolated_default_threshold(self):
        p1 = self._make_updated_partition(ConstraintDomain.ARITHMETIC, [1.0, 0.0, 0.0, 0.0])
        p2 = self._make_updated_partition(ConstraintDomain.CODE, [0.0, 1.0, 0.0, 0.0])
        pis = PartitionIsolationScore([p1, p2])
        # Default threshold is 0.8; orthogonal vectors have distance 1.0.
        assert pis.is_isolated() is True


# ---------------------------------------------------------------------------
# PPSConstraintLearner tests
# ---------------------------------------------------------------------------


class TestPPSConstraintLearner:
    """REQ-SELFLEARN-016, REQ-SELFLEARN-017, SCENARIO-SELFLEARN-016/017."""

    def _make_learner(self):
        replayer = LSEBMConstraintReplayer(n_replay=10, ebm_n_iter=50)
        domains = [ConstraintDomain.ARITHMETIC, ConstraintDomain.CODE, ConstraintDomain.LOGICAL]
        return PPSConstraintLearner(domains=domains, replayer=replayer)

    def test_fit_domain_arithmetic_does_not_change_code_partition(self):
        """SCENARIO-SELFLEARN-016: arithmetic training must not touch code partition."""
        learner = self._make_learner()
        # Record initial weights for CODE and LOGICAL.
        code_before = learner._partitions[ConstraintDomain.CODE].weights.copy()
        logical_before = learner._partitions[ConstraintDomain.LOGICAL].weights.copy()

        # Fit ARITHMETIC only.
        learner.fit_domain(ConstraintDomain.ARITHMETIC, ["carry", "carry", "sign", "carry"])

        # CODE and LOGICAL must be bit-for-bit unchanged.
        assert np.array_equal(
            learner._partitions[ConstraintDomain.CODE].weights, code_before
        ), "CODE partition must not change when ARITHMETIC is trained"
        assert np.array_equal(
            learner._partitions[ConstraintDomain.LOGICAL].weights, logical_before
        ), "LOGICAL partition must not change when ARITHMETIC is trained"

    def test_fit_domain_arithmetic_changes_arithmetic_partition(self):
        learner = self._make_learner()
        arith_before = learner._partitions[ConstraintDomain.ARITHMETIC].weights.copy()
        learner.fit_domain(ConstraintDomain.ARITHMETIC, ["carry", "carry", "sign"])
        assert not np.array_equal(
            learner._partitions[ConstraintDomain.ARITHMETIC].weights, arith_before
        )

    def test_fit_domain_code_does_not_change_arithmetic_or_logical(self):
        learner = self._make_learner()
        arith_before = learner._partitions[ConstraintDomain.ARITHMETIC].weights.copy()
        logical_before = learner._partitions[ConstraintDomain.LOGICAL].weights.copy()

        learner.fit_domain(ConstraintDomain.CODE, ["type_error", "off_by_one"])

        assert np.array_equal(
            learner._partitions[ConstraintDomain.ARITHMETIC].weights, arith_before
        )
        assert np.array_equal(
            learner._partitions[ConstraintDomain.LOGICAL].weights, logical_before
        )

    def test_generate_boundary_violations_returns_n_strings(self):
        """SCENARIO-SELFLEARN-017: generate_boundary_violations returns n strings."""
        learner = self._make_learner()
        learner.fit_domain(ConstraintDomain.ARITHMETIC, ["carry", "carry", "sign", "carry"])
        violations = learner.generate_boundary_violations(ConstraintDomain.ARITHMETIC, 10)
        assert len(violations) == 10

    def test_generate_boundary_violations_from_domain_vocab(self):
        """SCENARIO-SELFLEARN-017: all returned strings are from domain vocabulary."""
        learner = self._make_learner()
        learner.fit_domain(ConstraintDomain.ARITHMETIC, ["carry", "carry", "sign"])
        violations = learner.generate_boundary_violations(ConstraintDomain.ARITHMETIC, 10)
        for v in violations:
            assert v in {"carry", "sign"}, f"Unexpected violation type: {v}"

    def test_generate_boundary_violations_empty_before_fit(self):
        learner = self._make_learner()
        # No fit called — should return empty list.
        violations = learner.generate_boundary_violations(ConstraintDomain.CODE, 5)
        assert violations == []

    def test_partition_isolation_score_after_independent_training(self):
        """SCENARIO-SELFLEARN-018: isolation score > 0.8 after 3 domains trained independently."""
        replayer = LSEBMConstraintReplayer(n_replay=10, ebm_n_iter=50)
        learner = PPSConstraintLearner(
            domains=[ConstraintDomain.ARITHMETIC, ConstraintDomain.CODE, ConstraintDomain.LOGICAL],
            replayer=replayer,
        )
        # Train each domain on DIFFERENT violation types to ensure orthogonal gradients.
        learner.fit_domain(ConstraintDomain.ARITHMETIC, ["carry", "carry", "sign"])
        learner.fit_domain(ConstraintDomain.CODE, ["type_error", "type_error", "off_by_one"])
        learner.fit_domain(ConstraintDomain.LOGICAL, ["scope_error", "contradiction"])

        pis = PartitionIsolationScore(learner.partitions)
        score = pis.score()
        assert score >= 0.8, f"Expected isolation score >= 0.8, got {score}"
        assert pis.is_isolated(threshold=0.8)

    def test_session_fp_rate_decreases_when_correct_domain_trained(self):
        """REQ-SELFLEARN-016: training the correct domain reduces FP rate for that domain."""
        learner = self._make_learner()

        # Test questions: (question_text, expected_violation_type)
        test_questions = [("q1", "carry"), ("q2", "sign"), ("q3", "carry")]

        # Before training: no vocabulary → all failures.
        fp_before = learner.session_fp_rate(ConstraintDomain.ARITHMETIC, test_questions)
        assert fp_before == 1.0

        # After training: vocabulary includes 'carry' and 'sign' → no failures.
        learner.fit_domain(ConstraintDomain.ARITHMETIC, ["carry", "carry", "sign"])
        fp_after = learner.session_fp_rate(ConstraintDomain.ARITHMETIC, test_questions)
        assert fp_after < fp_before, "FP rate should decrease after correct domain training"
        assert fp_after == 0.0

    def test_session_fp_rate_empty_questions(self):
        learner = self._make_learner()
        assert learner.session_fp_rate(ConstraintDomain.ARITHMETIC, []) == 0.0

    def test_session_fp_rate_unknown_format_counted_as_failure(self):
        learner = self._make_learner()
        learner.fit_domain(ConstraintDomain.ARITHMETIC, ["carry"])
        # Plain strings (not tuples) are counted as not-detected.
        fp = learner.session_fp_rate(ConstraintDomain.ARITHMETIC, ["some_question"])
        assert fp == 1.0

    def test_partitions_property_returns_all_domains(self):
        learner = self._make_learner()
        ps = learner.partitions
        assert len(ps) == 3

    def test_fit_domain_empty_violations_is_noop(self):
        learner = self._make_learner()
        before = learner._partitions[ConstraintDomain.ARITHMETIC].weights.copy()
        learner.fit_domain(ConstraintDomain.ARITHMETIC, [])
        assert np.array_equal(learner._partitions[ConstraintDomain.ARITHMETIC].weights, before)

    def test_replayer_config_inherited(self):
        replayer = LSEBMConstraintReplayer(n_replay=7, ebm_n_iter=33)
        learner = PPSConstraintLearner(
            domains=[ConstraintDomain.ARITHMETIC],
            replayer=replayer,
        )
        assert learner._n_replay == 7
        assert learner._ebm_n_iter == 33
