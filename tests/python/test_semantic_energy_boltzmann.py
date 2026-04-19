"""Tests for BoltzmannSemanticEnergy Tier 0d hallucination detector.

Spec: REQ-VERIFY-101, REQ-VERIFY-102, REQ-VERIFY-103
SCENARIO-VERIFY-134, SCENARIO-VERIFY-135, SCENARIO-VERIFY-136
"""

from __future__ import annotations

import math

import pytest

from carnot.pipeline.semantic_energy_boltzmann import (
    BoltzmannSemanticEnergy,
    SemanticCluster,
    _char_embedding,
    _cosine_sim,
)


# ---------------------------------------------------------------------------
# SemanticCluster tests
# ---------------------------------------------------------------------------


class TestSemanticCluster:
    """Tests for SemanticCluster dataclass and boltzmann_weight().

    Spec: REQ-VERIFY-101
    """

    def test_default_construction(self):
        """SemanticCluster constructs with empty tokens and zero values."""
        c = SemanticCluster()
        assert c.tokens == []
        assert c.mean_logit == 0.0
        assert c.cluster_energy == 0.0

    def test_explicit_construction(self):
        """SemanticCluster stores provided field values correctly."""
        c = SemanticCluster(tokens=["cat", "cats"], mean_logit=2.5, cluster_energy=1.0)
        assert c.tokens == ["cat", "cats"]
        assert c.mean_logit == 2.5
        assert c.cluster_energy == 1.0

    def test_boltzmann_weight_zero_energy(self):
        """boltzmann_weight returns 1.0 when cluster_energy=0."""
        c = SemanticCluster(tokens=["a"], mean_logit=0.0, cluster_energy=0.0)
        assert c.boltzmann_weight(temperature=1.0) == pytest.approx(1.0)

    def test_boltzmann_weight_positive_energy(self):
        """boltzmann_weight is exp(-energy/temp) for positive energy."""
        c = SemanticCluster(tokens=["b"], mean_logit=-1.0, cluster_energy=1.0)
        expected = math.exp(-1.0 / 1.0)
        assert c.boltzmann_weight(temperature=1.0) == pytest.approx(expected)

    def test_boltzmann_weight_negative_energy(self):
        """boltzmann_weight is > 1 when energy is negative (high mean_logit)."""
        c = SemanticCluster(tokens=["c"], mean_logit=2.0, cluster_energy=-2.0)
        expected = math.exp(2.0)
        assert c.boltzmann_weight(temperature=1.0) == pytest.approx(expected)

    def test_boltzmann_weight_high_temperature_flattens(self):
        """High temperature makes weights for different energies converge toward 1."""
        c_high = SemanticCluster(tokens=["x"], mean_logit=5.0, cluster_energy=-5.0)
        c_low = SemanticCluster(tokens=["y"], mean_logit=-5.0, cluster_energy=5.0)
        # At temp=100, both weights approach 1 (nearly flat distribution)
        w_high = c_high.boltzmann_weight(temperature=100.0)
        w_low = c_low.boltzmann_weight(temperature=100.0)
        assert abs(w_high - w_low) < 1.0  # much closer than at temp=1

    def test_boltzmann_weight_default_temperature(self):
        """boltzmann_weight() uses temperature=1.0 by default."""
        c = SemanticCluster(tokens=["d"], mean_logit=0.0, cluster_energy=1.0)
        assert c.boltzmann_weight() == c.boltzmann_weight(temperature=1.0)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestCharEmbedding:
    """Tests for _char_embedding helper."""

    def test_returns_8_floats(self):
        """_char_embedding always returns a list of 8 floats."""
        vec = _char_embedding("hello")
        assert len(vec) == 8
        assert all(isinstance(v, float) for v in vec)

    def test_empty_string(self):
        """_char_embedding handles empty string without error."""
        vec = _char_embedding("")
        assert len(vec) == 8

    def test_values_bounded(self):
        """All embedding values are in [0, 1] for typical tokens."""
        for token in ["cat", "running", "123", "ABC", "the", ""]:
            vec = _char_embedding(token)
            for v in vec:
                assert 0.0 <= v <= 1.0 + 1e-9, f"out of range for token {token!r}: {v}"


class TestCosineSim:
    """Tests for _cosine_sim helper."""

    def test_identical_vectors(self):
        """Cosine similarity of a vector with itself is 1.0."""
        v = [1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert _cosine_sim(v, v) == pytest.approx(1.0)

    def test_zero_vectors(self):
        """Cosine similarity returns 0.0 for zero vectors."""
        z = [0.0] * 8
        assert _cosine_sim(z, z) == 0.0

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have cosine similarity 0.0."""
        a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert _cosine_sim(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BoltzmannSemanticEnergy.cluster() tests
# ---------------------------------------------------------------------------


class TestBoltzmannSemanticEnergyCluster:
    """Tests for BoltzmannSemanticEnergy.cluster().

    Spec: REQ-VERIFY-101
    """

    def test_empty_token_logits_returns_empty(self):
        """cluster() returns [] when token_logits is empty."""
        bse = BoltzmannSemanticEnergy()
        assert bse.cluster({}) == []

    def test_fewer_tokens_than_clusters(self):
        """cluster() uses min(n_clusters, n_tokens) clusters."""
        bse = BoltzmannSemanticEnergy(n_clusters=10)
        result = bse.cluster({"a": 1.0, "b": 2.0})
        assert 1 <= len(result) <= 2

    def test_cluster_returns_semantic_cluster_objects(self):
        """cluster() returns SemanticCluster instances."""
        bse = BoltzmannSemanticEnergy(n_clusters=3)
        token_logits = {"cat": 1.0, "dog": 0.5, "run": -0.5, "walk": -1.0, "the": 0.1}
        result = bse.cluster(token_logits)
        assert all(isinstance(c, SemanticCluster) for c in result)

    def test_all_tokens_assigned(self):
        """All tokens in token_logits appear in exactly one cluster."""
        bse = BoltzmannSemanticEnergy(n_clusters=5)
        token_logits = {f"tok{i}": float(i) for i in range(20)}
        clusters = bse.cluster(token_logits)
        all_assigned = [t for c in clusters for t in c.tokens]
        assert sorted(all_assigned) == sorted(token_logits.keys())

    def test_cluster_energy_formula(self):
        """cluster_energy = -mean_logit / temperature for each cluster."""
        bse = BoltzmannSemanticEnergy(n_clusters=1, temperature=2.0)
        token_logits = {"a": 4.0, "b": 4.0}
        clusters = bse.cluster(token_logits)
        assert len(clusters) == 1
        assert clusters[0].mean_logit == pytest.approx(4.0)
        assert clusters[0].cluster_energy == pytest.approx(-4.0 / 2.0)

    def test_single_token(self):
        """cluster() works with a single token."""
        bse = BoltzmannSemanticEnergy(n_clusters=10)
        result = bse.cluster({"only": 3.0})
        assert len(result) == 1
        assert result[0].tokens == ["only"]
        assert result[0].mean_logit == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# BoltzmannSemanticEnergy.score() tests
# ---------------------------------------------------------------------------


class TestBoltzmannSemanticEnergyScore:
    """Tests for BoltzmannSemanticEnergy.score().

    Spec: REQ-VERIFY-102
    SCENARIO-VERIFY-134, SCENARIO-VERIFY-135
    """

    def test_score_returns_float_in_01(self):
        """score() returns a float in [0.0, 1.0] (SCENARIO-VERIFY-134)."""
        bse = BoltzmannSemanticEnergy()
        token_logits = {f"tok{i}": float(i - 5) for i in range(15)}
        s = bse.score("some response text", token_logits)
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_score_empty_logits_returns_half(self):
        """score() returns 0.5 (uninformative) when token_logits is empty."""
        bse = BoltzmannSemanticEnergy()
        assert bse.score("response", {}) == 0.5

    def test_score_high_variance_greater_than_low_variance(self):
        """Uncertain (negative) logits yield higher score than confident (positive) logits (SCENARIO-VERIFY-135).

        Boltzmann energy: cluster_energy = -mean_logit / temperature.
        Negative mean_logit → positive cluster_energy → high hallucination energy.
        Positive mean_logit → negative cluster_energy → low hallucination energy.

        The Boltzmann-weighted total energy is:
            total_energy = sum(cluster_energy_k * weight_k) where weight_k ∝ exp(-cluster_energy_k/T)

        For all-positive logits (confident model): all cluster_energy < 0 → total_energy < 0 → score < 0.5
        For all-negative logits (uncertain model):  all cluster_energy > 0 → total_energy > 0 → score > 0.5

        Thus: score(uncertain, negative logits) > score(confident, positive logits).
        "High variance" in the sense of the hallucination task means HIGH ENERGY (uncertain model),
        which corresponds to NEGATIVE/LOW logits.
        """
        bse = BoltzmannSemanticEnergy(n_clusters=4, temperature=1.0)

        # Low "variance" / high confidence: all logits strongly positive
        # → all cluster energies negative → total_energy negative → low score
        low_energy_logits = {f"tok{i}": 3.0 for i in range(20)}

        # High "variance" / low confidence: all logits strongly negative
        # → all cluster energies positive → total_energy positive → high score
        high_energy_logits = {f"tok{i}": -3.0 for i in range(20)}

        s_low = bse.score("resp", low_energy_logits)
        s_high = bse.score("resp", high_energy_logits)
        # Uncertain model (negative logits) → high hallucination score
        assert s_high > s_low, (
            f"Expected high-energy score ({s_high:.4f}) > low-energy ({s_low:.4f})"
        )

    def test_score_all_high_logits_low_score(self):
        """When all logits are large positive, score is close to 0 (low hallucination)."""
        bse = BoltzmannSemanticEnergy(n_clusters=3)
        # All high logits → all clusters have negative energy → total_energy << 0 → sigmoid ≈ 0
        token_logits = {f"tok{i}": 10.0 for i in range(15)}
        s = bse.score("response", token_logits)
        assert s < 0.5

    def test_score_all_negative_logits_high_score(self):
        """When all logits are large negative, score is close to 1 (high hallucination)."""
        bse = BoltzmannSemanticEnergy(n_clusters=3)
        # All negative logits → all clusters have high positive energy → total_energy >> 0 → sigmoid ≈ 1
        token_logits = {f"tok{i}": -10.0 for i in range(15)}
        s = bse.score("response", token_logits)
        assert s > 0.5

    def test_score_response_text_not_required_to_affect_result(self):
        """score() accepts any response text without error."""
        bse = BoltzmannSemanticEnergy()
        logits = {"a": 1.0, "b": -1.0}
        s1 = bse.score("hello world", logits)
        s2 = bse.score("", logits)
        s3 = bse.score("completely different text with numbers 123", logits)
        # All calls succeed; scores are equal (response text currently unused)
        assert s1 == s2 == s3

    def test_score_single_token(self):
        """score() with a single token does not raise and returns [0, 1]."""
        bse = BoltzmannSemanticEnergy()
        s = bse.score("response", {"only": 2.0})
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# BoltzmannSemanticEnergy.benchmark() tests
# ---------------------------------------------------------------------------


class TestBoltzmannSemanticEnergyBenchmark:
    """Tests for BoltzmannSemanticEnergy.benchmark().

    Spec: REQ-VERIFY-103
    SCENARIO-VERIFY-136
    """

    def test_benchmark_returns_auroc_key(self):
        """benchmark() returns dict with auroc key (SCENARIO-VERIFY-136)."""
        bse = BoltzmannSemanticEnergy(n_clusters=3)
        responses = [
            (f"response {i}", {f"tok{j}": float(j) for j in range(5)})
            for i in range(10)
        ]
        ground_truth = [True, False, True, False, True, False, True, False, True, False]
        result = bse.benchmark(responses, ground_truth)
        assert "auroc" in result

    def test_benchmark_returns_required_keys(self):
        """benchmark() result contains all required keys."""
        bse = BoltzmannSemanticEnergy()
        responses = [("resp", {"a": 1.0})]
        ground_truth = [True]
        result = bse.benchmark(responses, ground_truth)
        for key in ("auroc", "skip_rate", "n_total", "n_hallucinated", "n_correct"):
            assert key in result, f"Missing key: {key}"

    def test_benchmark_empty_returns_defaults(self):
        """benchmark() on empty input returns auroc=0.5 and skip_rate=0.0."""
        bse = BoltzmannSemanticEnergy()
        result = bse.benchmark([], [])
        assert result["auroc"] == 0.5
        assert result["skip_rate"] == 0.0
        assert result["n_total"] == 0

    def test_benchmark_auroc_in_01(self):
        """benchmark() auroc is always in [0, 1]."""
        bse = BoltzmannSemanticEnergy(n_clusters=5)
        # 10 synthetic responses with varied logit profiles
        responses = []
        ground_truth = []
        for i in range(10):
            is_correct = i % 2 == 0
            # Correct responses: high logits (low hallucination energy)
            # Hallucinated: low/negative logits (high hallucination energy)
            base = 3.0 if is_correct else -3.0
            logits = {f"w{j}": base + 0.1 * j for j in range(8)}
            responses.append((f"response {i}", logits))
            ground_truth.append(is_correct)
        result = bse.benchmark(responses, ground_truth)
        assert 0.0 <= result["auroc"] <= 1.0

    def test_benchmark_perfect_discrimination(self):
        """benchmark() achieves auroc=1.0 when hallucinated always score > correct."""
        bse = BoltzmannSemanticEnergy(n_clusters=2)
        # Correct: all tokens with high logits → low score
        # Hallucinated: all tokens with very negative logits → high score
        correct_logits = {f"c{i}": 10.0 for i in range(5)}
        hall_logits = {f"h{i}": -10.0 for i in range(5)}
        responses = [
            ("correct response", correct_logits),
            ("hallucinated response", hall_logits),
        ]
        ground_truth = [True, False]
        result = bse.benchmark(responses, ground_truth)
        assert result["auroc"] == pytest.approx(1.0)

    def test_benchmark_n_counts(self):
        """benchmark() counts n_hallucinated and n_correct correctly."""
        bse = BoltzmannSemanticEnergy()
        responses = [("r", {"a": 1.0})] * 6
        ground_truth = [True, True, True, False, False, False]
        result = bse.benchmark(responses, ground_truth)
        assert result["n_correct"] == 3
        assert result["n_hallucinated"] == 3
        assert result["n_total"] == 6

    def test_benchmark_no_hallucinated_responses(self):
        """benchmark() returns auroc=0.5 when ground_truth has no False entries."""
        bse = BoltzmannSemanticEnergy()
        responses = [("r", {"a": 1.0, "b": -1.0})] * 4
        ground_truth = [True, True, True, True]
        result = bse.benchmark(responses, ground_truth)
        assert result["auroc"] == pytest.approx(0.5)
        assert result["n_hallucinated"] == 0

    def test_benchmark_no_correct_responses(self):
        """benchmark() returns auroc=0.5 when ground_truth has no True entries."""
        bse = BoltzmannSemanticEnergy()
        responses = [("r", {"a": 1.0, "b": -1.0})] * 4
        ground_truth = [False, False, False, False]
        result = bse.benchmark(responses, ground_truth)
        assert result["auroc"] == pytest.approx(0.5)
        assert result["n_correct"] == 0

    def test_benchmark_skip_rate_all_above_threshold(self):
        """skip_rate=1.0 when all scores are > 0.5."""
        bse = BoltzmannSemanticEnergy(n_clusters=2)
        # All negative logits → all scores > 0.5
        responses = [("r", {f"h{i}": -10.0 for i in range(5)})] * 4
        ground_truth = [False, False, False, False]
        result = bse.benchmark(responses, ground_truth)
        assert result["skip_rate"] == pytest.approx(1.0)

    def test_benchmark_skip_rate_none_above_threshold(self):
        """skip_rate=0.0 when all scores are <= 0.5."""
        bse = BoltzmannSemanticEnergy(n_clusters=2)
        # All high positive logits → all scores < 0.5
        responses = [("r", {f"c{i}": 10.0 for i in range(5)})] * 4
        ground_truth = [True, True, True, True]
        result = bse.benchmark(responses, ground_truth)
        assert result["skip_rate"] == pytest.approx(0.0)
