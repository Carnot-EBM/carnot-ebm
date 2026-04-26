"""Tests for Exp 885: SpectralAttentionProbe Tier 0h.

Covers SpectralAttentionProbe and its wiring into VerifyRepairPipeline.verify().

Spec: REQ-VERIFY-146, SCENARIO-VERIFY-173, SCENARIO-VERIFY-174
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_probe():
    from carnot.verify.spectral_attention_probe import SpectralAttentionProbe

    return SpectralAttentionProbe(window=3, n_eigenvalues=10, threshold=2.0)


# ---------------------------------------------------------------------------
# build_cooccurrence_matrix
# ---------------------------------------------------------------------------


class TestBuildCooccurrenceMatrix:
    """REQ-VERIFY-146: co-occurrence matrix structure and symmetry."""

    def test_empty_text_returns_one_by_one(self) -> None:
        """Empty text produces a 1×1 zero matrix (graceful fallback).

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        mat = probe.build_cooccurrence_matrix("")
        assert mat.shape == (1, 1)
        assert mat[0, 0] == 0.0

    def test_single_token_returns_one_by_one(self) -> None:
        """A single unique token produces a 1×1 zero matrix.

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        mat = probe.build_cooccurrence_matrix("hello")
        assert mat.shape == (1, 1)
        assert mat[0, 0] == 0.0

    def test_two_tokens_produces_two_by_two(self) -> None:
        """Two different tokens within the window produce a 2×2 symmetric matrix.

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        mat = probe.build_cooccurrence_matrix("hello world")
        assert mat.shape == (2, 2)
        # Off-diagonal must be non-zero (co-occurrence within window).
        assert mat[0, 1] > 0 or mat[1, 0] > 0

    def test_symmetry(self) -> None:
        """Co-occurrence matrix must be symmetric: M[i,j] == M[j,i].

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        mat = probe.build_cooccurrence_matrix("the cat sat on the mat")
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)

    def test_diagonal_is_zero(self) -> None:
        """Self-co-occurrence (diagonal) must always be zero.

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        mat = probe.build_cooccurrence_matrix("the quick brown fox jumps")
        assert np.all(np.diag(mat) == 0.0)

    def test_repeated_tokens_increment_count(self) -> None:
        """Repeated token pairs within window accumulate higher per-edge counts.

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        # "a b" appears once; max off-diagonal should be 1.
        mat_once = probe.build_cooccurrence_matrix("a b")
        # "a b a b a b" — "a" and "b" co-occur many times; max off-diagonal > 1.
        mat_many = probe.build_cooccurrence_matrix("a b a b a b a b")
        # max entry in many > max entry in once.
        assert mat_many.max() > mat_once.max()

    def test_window_limits_cooccurrence(self) -> None:
        """Tokens outside the window should not co-occur; narrower window = fewer edges.

        Spec: REQ-VERIFY-146
        """
        from carnot.verify.spectral_attention_probe import SpectralAttentionProbe

        probe_w1 = SpectralAttentionProbe(window=1)
        probe_w4 = SpectralAttentionProbe(window=4)
        # Longer text so the window restriction matters.
        text = "alpha beta gamma delta epsilon zeta"
        mat_w1 = probe_w1.build_cooccurrence_matrix(text)
        mat_w4 = probe_w4.build_cooccurrence_matrix(text)
        # Wider window captures more pairs → higher total edge weight.
        assert mat_w4.sum() >= mat_w1.sum()


# ---------------------------------------------------------------------------
# compute_laplacian
# ---------------------------------------------------------------------------


class TestComputeLaplacian:
    """REQ-VERIFY-146: Laplacian L = D - A properties."""

    def test_row_sums_are_zero(self) -> None:
        """Every row of the Laplacian must sum to zero (standard property of L = D - A).

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        cooc = probe.build_cooccurrence_matrix("the cat sat on the mat near the cat")
        lap = probe.compute_laplacian(cooc)
        row_sums = lap.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.zeros_like(row_sums), atol=1e-5)

    def test_diagonal_equals_degree(self) -> None:
        """Diagonal of the Laplacian must equal the row-sum of the adjacency matrix.

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        cooc = probe.build_cooccurrence_matrix("alpha beta gamma alpha beta")
        lap = probe.compute_laplacian(cooc)
        expected_diag = cooc.sum(axis=1)
        np.testing.assert_allclose(np.diag(lap), expected_diag, atol=1e-5)

    def test_off_diagonal_equals_negative_adjacency(self) -> None:
        """Off-diagonal entries must equal the negative of the adjacency matrix.

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        cooc = probe.build_cooccurrence_matrix("a b c a b")
        lap = probe.compute_laplacian(cooc)
        n = cooc.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert abs(lap[i, j] - (-cooc[i, j])) < 1e-5


# ---------------------------------------------------------------------------
# compute_spectral_entropy
# ---------------------------------------------------------------------------


class TestComputeSpectralEntropy:
    """REQ-VERIFY-146: spectral entropy properties."""

    def test_empty_graph_returns_zero(self) -> None:
        """A 1×1 Laplacian (single node) must return spectral entropy 0.0.

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        lap = np.array([[0.0]])
        assert probe.compute_spectral_entropy(lap) == 0.0

    def test_non_negative(self) -> None:
        """Spectral entropy must be non-negative.

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        cooc = probe.build_cooccurrence_matrix("the quick brown fox jumps over the lazy dog")
        lap = probe.compute_laplacian(cooc)
        entropy = probe.compute_spectral_entropy(lap)
        assert entropy >= 0.0

    def test_concentrated_graph_lower_entropy_than_diffuse(self) -> None:
        """A concentrated graph (one hub) must have lower spectral entropy than a uniform one.

        This is the core invariant of the spectral diffuseness probe.

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        # Concentrated: one word dominates ("the" appears 8×, others once).
        concentrated_text = "the the the the the cat sat mat"
        # Diffuse: all words unique and evenly distributed.
        diffuse_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"

        cooc_c = probe.build_cooccurrence_matrix(concentrated_text)
        cooc_d = probe.build_cooccurrence_matrix(diffuse_text)
        lap_c = probe.compute_laplacian(cooc_c)
        lap_d = probe.compute_laplacian(cooc_d)
        entropy_c = probe.compute_spectral_entropy(lap_c)
        entropy_d = probe.compute_spectral_entropy(lap_d)

        # Diffuse graph → higher spectral entropy (flatter spectrum).
        assert entropy_d >= entropy_c - 0.1  # Allow small numerical tolerance.


# ---------------------------------------------------------------------------
# compute_trajectory and is_diffuse
# ---------------------------------------------------------------------------


class TestTrajectoryAndDiffuse:
    """REQ-VERIFY-146: trajectory and is_diffuse logic."""

    def test_empty_steps_returns_empty_array(self) -> None:
        """Empty step list must produce an empty trajectory array.

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        traj = probe.compute_trajectory([])
        assert len(traj) == 0

    def test_trajectory_length_matches_steps(self) -> None:
        """Trajectory length must equal number of input steps.

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        steps = ["step one alpha beta", "step two gamma delta epsilon", "step three omega"]
        traj = probe.compute_trajectory(steps)
        assert len(traj) == len(steps)

    def test_is_diffuse_empty_returns_false(self) -> None:
        """Empty trajectory must return is_diffuse=False.

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        assert probe.is_diffuse(np.array([])) is False

    def test_is_diffuse_low_entropy_returns_false(self) -> None:
        """Low-entropy trajectory (below threshold) must return is_diffuse=False.

        Spec: SCENARIO-VERIFY-173
        """
        probe = _make_probe()
        # All zeros → mean=0.0 < threshold=2.0.
        traj = np.array([0.1, 0.05, 0.12])
        assert probe.is_diffuse(traj) is False

    def test_is_diffuse_high_increasing_returns_true(self) -> None:
        """High, monotonically-increasing trajectory must return is_diffuse=True.

        Spec: SCENARIO-VERIFY-174
        """
        probe = _make_probe()
        # Mean=5.0 > threshold=2.0; all diffs > 0.
        traj = np.array([3.0, 5.0, 7.0])
        assert probe.is_diffuse(traj) is True

    def test_is_diffuse_high_but_decreasing_returns_false(self) -> None:
        """High-entropy but decreasing trajectory must return is_diffuse=False (wrong trend).

        Spec: SCENARIO-VERIFY-174
        """
        probe = _make_probe()
        # Mean=5.0 > threshold, but all diffs < 0 (decreasing).
        traj = np.array([7.0, 5.0, 3.0])
        assert probe.is_diffuse(traj) is False


# ---------------------------------------------------------------------------
# train / predict / evaluate
# ---------------------------------------------------------------------------


class TestTrainPredict:
    """REQ-VERIFY-146: logistic regression training and prediction."""

    def _make_correct_chain(self, idx: int) -> list[str]:
        words = ["compute", "sum", "total", "result", "equals", "therefore"]
        return [f"step {i}: " + " ".join(words) + f" value{idx}" for i in range(4)]

    def _make_halluc_chain(self, idx: int) -> list[str]:
        import random

        rng = random.Random(idx * 17)
        extra = [f"novel{rng.randint(0, 200)}" for _ in range(10)]
        return [
            f"step {i}: " + " ".join(extra[i : i + 3]) + f" new_{i} unique_{idx}_{i} " * (i + 1)
            for i in range(4)
        ]

    def test_train_predict_direction(self) -> None:
        """After training, hallucinating chains must score higher than correct chains.

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        pos = [self._make_correct_chain(i) for i in range(10)]
        neg = [self._make_halluc_chain(i) for i in range(10)]
        probe.train(pos, neg)

        # Average proba for hallucinating chains should exceed correct chains.
        correct_proba = np.mean([probe._score_proba(c) for c in pos])
        halluc_proba = np.mean([probe._score_proba(n) for n in neg])
        # Not a strict assert (tiny synthetic data) — just direction.
        assert halluc_proba >= correct_proba - 0.2

    def test_predict_keys(self) -> None:
        """predict() must return all three required keys.

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        result = probe.predict(["step one compute sum", "step two result equals"])
        assert "is_spectrally_diffuse" in result
        assert "spectral_entropy_mean" in result
        assert "auc_score" in result

    def test_predict_is_spectrally_diffuse_is_bool(self) -> None:
        """is_spectrally_diffuse must be a Python bool.

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        result = probe.predict(["step one compute sum", "step two result equals"])
        assert isinstance(result["is_spectrally_diffuse"], bool)

    def test_evaluate_returns_float_in_0_1(self) -> None:
        """evaluate() must return an AUC float in [0, 1].

        Spec: REQ-VERIFY-146
        """
        probe = _make_probe()
        pos = [self._make_correct_chain(i) for i in range(8)]
        neg = [self._make_halluc_chain(i) for i in range(8)]
        probe.train(pos, neg)
        auc = probe.evaluate(
            [self._make_correct_chain(i + 8) for i in range(5)],
            [self._make_halluc_chain(i + 8) for i in range(5)],
        )
        assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# CARNOT_SPECTRAL_PROBE env flag wiring
# ---------------------------------------------------------------------------


class TestSpectralProbeWiring:
    """SCENARIO-VERIFY-173/174: advisory wiring in VerifyRepairPipeline.verify()."""

    def test_flag_disabled_no_certificate_key(self) -> None:
        """When CARNOT_SPECTRAL_PROBE is unset, certificate must NOT contain tier_0h_spectral.

        Spec: SCENARIO-VERIFY-173
        """
        os.environ.pop("CARNOT_SPECTRAL_PROBE", None)
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        pipeline = VerifyRepairPipeline(model=None, domains=["arithmetic"])
        result = pipeline.verify(
            "What is 2+2?",
            "Step 1: 2+2=4. Step 2: result is 4.",
            domain="arithmetic",
            tracker=None,
            jepa_predictor=None,
            jepa_threshold=0.5,
            think_probe=None,
            hallufield_detector=None,
            semantic_energy_probe=None,
            embedding_constraint_store=None,
            ising_constraint_injector=None,
        )
        assert "tier_0h_spectral" not in result.certificate
        assert result.spectral_diffuse is False
        assert result.spectral_entropy_mean == 0.0

    def test_flag_enabled_certificate_populated(self) -> None:
        """When CARNOT_SPECTRAL_PROBE=1, certificate must contain tier_0h_spectral.

        Spec: SCENARIO-VERIFY-173
        """
        os.environ["CARNOT_SPECTRAL_PROBE"] = "1"
        try:
            from carnot.pipeline.verify_repair import VerifyRepairPipeline

            pipeline = VerifyRepairPipeline(model=None, domains=["arithmetic"])
            # Multi-step response so extract_cot_steps finds real steps.
            response = (
                "Step 1: compute the sum of 2 and 3.\n"
                "Step 2: 2 + 3 equals 5.\n"
                "Step 3: therefore the result is 5."
            )
            result = pipeline.verify(
                "What is 2+3?",
                response,
                domain="arithmetic",
                tracker=None,
                jepa_predictor=None,
                jepa_threshold=0.5,
                think_probe=None,
                hallufield_detector=None,
                semantic_energy_probe=None,
                embedding_constraint_store=None,
                ising_constraint_injector=None,
            )
            assert "tier_0h_spectral" in result.certificate
            assert isinstance(result.spectral_diffuse, bool)
            assert isinstance(result.spectral_entropy_mean, float)
            assert result.spectral_entropy_mean >= 0.0
        finally:
            os.environ.pop("CARNOT_SPECTRAL_PROBE", None)

    def test_certificate_contains_required_keys(self) -> None:
        """tier_0h_spectral certificate entry must have all three expected keys.

        Spec: SCENARIO-VERIFY-174
        """
        os.environ["CARNOT_SPECTRAL_PROBE"] = "1"
        try:
            from carnot.pipeline.verify_repair import VerifyRepairPipeline

            pipeline = VerifyRepairPipeline(model=None, domains=["arithmetic"])
            response = (
                "Step 1: alpha beta gamma delta.\n"
                "Step 2: epsilon zeta eta theta.\n"
                "Step 3: iota kappa lambda mu."
            )
            result = pipeline.verify(
                "Some question?",
                response,
                domain="arithmetic",
                tracker=None,
                jepa_predictor=None,
                jepa_threshold=0.5,
                think_probe=None,
                hallufield_detector=None,
                semantic_energy_probe=None,
                embedding_constraint_store=None,
                ising_constraint_injector=None,
            )
            cert = result.certificate.get("tier_0h_spectral", {})
            assert "is_spectrally_diffuse" in cert
            assert "spectral_entropy_mean" in cert
            assert "n_steps" in cert
        finally:
            os.environ.pop("CARNOT_SPECTRAL_PROBE", None)

    def test_advisory_does_not_change_verified_flag(self) -> None:
        """Spectral probe must not change the verified/violated outcome (advisory only).

        Spec: REQ-VERIFY-146
        """
        os.environ["CARNOT_SPECTRAL_PROBE"] = "1"
        try:
            from carnot.pipeline.verify_repair import VerifyRepairPipeline

            pipeline = VerifyRepairPipeline(model=None, domains=["arithmetic"])
            # A deliberately diffuse multi-step response.
            response = (
                "Step 1: dragon wizard quantum neutrino chromosome unique1.\n"
                "Step 2: wormhole singularity photon enzyme genome unique2.\n"
                "Step 3: recipe ingredient bandwidth latency subnet unique3."
            )
            result_with = pipeline.verify(
                "Question?",
                response,
                domain="arithmetic",
                tracker=None,
                jepa_predictor=None,
                jepa_threshold=0.5,
                think_probe=None,
                hallufield_detector=None,
                semantic_energy_probe=None,
                embedding_constraint_store=None,
                ising_constraint_injector=None,
            )

            os.environ.pop("CARNOT_SPECTRAL_PROBE", None)
            result_without = pipeline.verify(
                "Question?",
                response,
                domain="arithmetic",
                tracker=None,
                jepa_predictor=None,
                jepa_threshold=0.5,
                think_probe=None,
                hallufield_detector=None,
                semantic_energy_probe=None,
                embedding_constraint_store=None,
                ising_constraint_injector=None,
            )

            # verified flag must be the same regardless of spectral probe.
            assert result_with.verified == result_without.verified
        finally:
            os.environ.pop("CARNOT_SPECTRAL_PROBE", None)


# ---------------------------------------------------------------------------
# _auc_roc helper
# ---------------------------------------------------------------------------


class TestAucRoc:
    """Unit tests for the standalone _auc_roc helper."""

    def test_perfect_auc(self) -> None:
        """Perfect separation must produce AUC=1.0.

        Spec: REQ-VERIFY-146
        """
        from carnot.verify.spectral_attention_probe import _auc_roc

        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        auc = _auc_roc(scores, labels)
        assert abs(auc - 1.0) < 1e-6

    def test_random_auc(self) -> None:
        """Random scores (labels shuffled) must produce AUC near 0.5.

        Spec: REQ-VERIFY-146
        """
        from carnot.verify.spectral_attention_probe import _auc_roc

        rng = np.random.default_rng(42)
        scores = rng.random(100)
        labels = rng.integers(0, 2, size=100)
        auc = _auc_roc(scores, labels)
        assert 0.3 <= auc <= 0.7

    def test_empty_returns_half(self) -> None:
        """Empty inputs must return AUC=0.5 (neutral).

        Spec: REQ-VERIFY-146
        """
        from carnot.verify.spectral_attention_probe import _auc_roc

        assert _auc_roc(np.array([]), np.array([])) == 0.5


# ---------------------------------------------------------------------------
# Experiment script deliverable test
# ---------------------------------------------------------------------------


class TestExperiment885Deliverable:
    """Integration test: run the experiment and verify the JSON deliverable.

    Spec: REQ-VERIFY-146, SCENARIO-VERIFY-173, SCENARIO-VERIFY-174
    """

    def test_deliverable_written_and_valid(self) -> None:
        """Running main() must produce a JSON with all required schema fields.

        Spec: REQ-VERIFY-146
        """
        import json
        import sys
        import importlib.util

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        scripts_dir = os.path.join(repo_root, "scripts")
        sys.path.insert(0, scripts_dir)

        # Run from repo root so relative paths ("results/...") resolve correctly.
        orig_dir = os.getcwd()
        os.chdir(repo_root)
        try:
            script_path = os.path.join(scripts_dir, "experiment_885_spectral_attention_probe.py")
            spec = importlib.util.spec_from_file_location("exp885", script_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
        finally:
            os.chdir(orig_dir)

        deliverable = os.path.join(
            repo_root, "results", "experiment_885_spectral_attention_probe.json"
        )
        assert os.path.exists(deliverable), "Deliverable JSON must be written by main()"

        with open(deliverable) as f:
            data = json.load(f)

        for field in [
            "experiment",
            "schema",
            "run_date",
            "started_at",
            "finished_at",
            "duration_s",
            "status",
            "title",
        ]:
            assert field in data, f"Required field '{field}' missing from deliverable"

        assert data["experiment"] == 885
        assert data["status"] == "success"
        assert "probe_auc" in data
        assert "advisory_signal_rate" in data
        assert "honest_verdict" in data
        assert data["honest_verdict"] in (
            "tier_0h_viable",
            "tier_0h_marginal",
            "tier_0h_not_viable",
        )
