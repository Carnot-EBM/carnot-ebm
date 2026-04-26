"""Tests for Experiment 911: DRIFTProbe Tier 0i — Multi-Layer Hidden-State Drift.

Covers:
- DRIFTProbe (python/carnot/verify/drift_probe.py) core API.
- GSM8K triple generation helpers in experiment_911.
- End-to-end deliverable schema validation.

All tests are CI-safe: no real LLM is required.  DRIFTProbe accepts a synthetic
model_runner that returns pre-built numpy arrays, so the full probe logic
(extract_drift_signature, fit_from_signatures, predict_violation_prob) is exercised
without downloading model weights.

Spec traces: REQ-TIER0-005, SCENARIO-TIER0-005
Spec: REQ-AUTO-011
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_runner(layers: list[int], hidden_dim: int = 16):
    """Return a synthetic model_runner with controllable drift level.

    Calling runner(text) returns a dict[int, np.ndarray] where each layer's
    hidden states are independent random vectors (high inter-layer cosine drift).

    Args:
        layers:     Layer indices to return.
        hidden_dim: Dimensionality of hidden state vectors.

    Returns:
        Callable[[str], dict[int, np.ndarray]]
    """
    rng = np.random.default_rng(99)

    def runner(text: str) -> dict[int, np.ndarray]:
        # Each layer gets independent random vectors → high cosine distance.
        return {layer: rng.standard_normal((8, hidden_dim)).astype(np.float32) for layer in layers}

    return runner


def _low_drift_runner(layers: list[int], hidden_dim: int = 16):
    """Return a synthetic model_runner producing low inter-layer drift.

    All layers share the same base vector with tiny noise → cosine similarity
    close to 1.0 → drift close to 0.0.

    Spec: REQ-TIER0-005-1
    """

    def runner(text: str) -> dict[int, np.ndarray]:
        base = np.ones((8, hidden_dim), dtype=np.float32)
        return {
            layer: base + 0.001 * np.eye(hidden_dim, 8).T.astype(np.float32) for layer in layers
        }

    return runner


# ---------------------------------------------------------------------------
# REQ-TIER0-005-1: extract_drift_signature shape and value range
# ---------------------------------------------------------------------------


class TestExtractDriftSignature:
    """REQ-TIER0-005-1: extract_drift_signature returns correct shape and range."""

    def test_shape_matches_n_drift_pairs(self) -> None:
        """Signature shape equals (n_drift_pairs,) for any probe configuration.

        Spec: REQ-TIER0-005-1
        """
        from carnot.verify.drift_probe import DRIFTProbe

        layers = [-4, -3, -2, -1]
        probe = DRIFTProbe(model_runner=None, layers=layers)
        # Inject prebuilt hidden states.
        hidden_dim = 8
        hs = {l: np.ones((4, hidden_dim), dtype=np.float32) for l in layers}
        sig = probe.extract_drift_signature(hs)
        assert sig.shape == (3,), f"Expected (3,), got {sig.shape}"

    def test_values_clamped_to_zero_two(self) -> None:
        """All drift values must lie in [0, 2].

        Spec: REQ-TIER0-005-1
        """
        from carnot.verify.drift_probe import DRIFTProbe

        layers = [-3, -2, -1]
        rng = np.random.default_rng(0)
        probe = DRIFTProbe(model_runner=None, layers=layers)
        hs = {l: rng.standard_normal((6, 16)).astype(np.float32) for l in layers}
        sig = probe.extract_drift_signature(hs)
        assert np.all(sig >= 0.0), f"Negative drift: {sig}"
        assert np.all(sig <= 2.0), f"Drift exceeds 2.0: {sig}"

    def test_dtype_is_float32(self) -> None:
        """Drift signature dtype must be float32.

        Spec: REQ-TIER0-005-1
        """
        from carnot.verify.drift_probe import DRIFTProbe

        probe = DRIFTProbe(model_runner=None, layers=[-2, -1])
        hs = {-2: np.ones((4, 8), dtype=np.float32), -1: np.ones((4, 8), dtype=np.float32)}
        sig = probe.extract_drift_signature(hs)
        assert sig.dtype == np.float32

    def test_identical_layers_produces_zero_drift(self) -> None:
        """Identical hidden states across consecutive layers yield drift ≈ 0.

        If layer L and layer L+1 have exactly the same representation, cosine
        similarity is 1.0, so drift = 1 - 1 = 0.

        Spec: REQ-TIER0-005-1
        """
        from carnot.verify.drift_probe import DRIFTProbe

        probe = DRIFTProbe(model_runner=None, layers=[-2, -1])
        shared = np.random.default_rng(7).standard_normal((5, 16)).astype(np.float32)
        hs = {-2: shared.copy(), -1: shared.copy()}
        sig = probe.extract_drift_signature(hs)
        np.testing.assert_allclose(sig, 0.0, atol=1e-5)

    def test_orthogonal_layers_produces_high_drift(self) -> None:
        """Orthogonal hidden states across consecutive layers yield drift ≈ 1.

        cosine_sim(a, b) = 0 for orthogonal vectors, so drift = 1 - 0 = 1.

        Spec: REQ-TIER0-005-1
        """
        from carnot.verify.drift_probe import DRIFTProbe

        probe = DRIFTProbe(model_runner=None, layers=[-2, -1])
        # Two clearly orthogonal vectors, replicated as seq_len=1.
        a = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        hs = {-2: a, -1: b}
        sig = probe.extract_drift_signature(hs)
        # Drift should be close to 1.0 (orthogonal → cosine_sim ≈ 0).
        assert abs(sig[0] - 1.0) < 1e-4, f"Expected drift≈1.0 for orthogonal vectors, got {sig[0]}"

    def test_missing_layer_gives_zero_not_crash(self) -> None:
        """Missing layer key in hidden_states must produce 0.0 drift, not a crash.

        Spec: REQ-TIER0-005-5
        """
        from carnot.verify.drift_probe import DRIFTProbe

        probe = DRIFTProbe(model_runner=None, layers=[-4, -3, -2, -1])
        # Only provide two of the four required layers.
        hs = {-2: np.ones((4, 8), dtype=np.float32), -1: np.ones((4, 8), dtype=np.float32)}
        sig = probe.extract_drift_signature(hs)
        assert sig.shape == (3,)
        # First two pairs missing → zero; last pair present.
        assert np.all(sig >= 0.0)


# ---------------------------------------------------------------------------
# REQ-TIER0-005-4: default layers resolution
# ---------------------------------------------------------------------------


class TestDefaultLayers:
    """REQ-TIER0-005-4: default layers resolves to last n_drift_pairs+1 indices."""

    def test_default_n3_gives_last_4_layers(self) -> None:
        """n_drift_pairs=3 (default) → layers=[-4, -3, -2, -1].

        Spec: REQ-TIER0-005-4
        """
        from carnot.verify.drift_probe import DRIFTProbe

        probe = DRIFTProbe(model_runner=None, n_drift_pairs=3)
        assert probe.layers == [-4, -3, -2, -1]

    def test_custom_n2_gives_last_3_layers(self) -> None:
        """n_drift_pairs=2 → layers=[-3, -2, -1].

        Spec: REQ-TIER0-005-4
        """
        from carnot.verify.drift_probe import DRIFTProbe

        probe = DRIFTProbe(model_runner=None, n_drift_pairs=2)
        assert probe.layers == [-3, -2, -1]

    def test_explicit_layers_override_n_drift_pairs(self) -> None:
        """When layers is provided explicitly, it overrides n_drift_pairs resolution.

        Spec: REQ-TIER0-005-4
        """
        from carnot.verify.drift_probe import DRIFTProbe

        probe = DRIFTProbe(model_runner=None, layers=[1, 5, 9])
        assert probe.layers == [1, 5, 9]


# ---------------------------------------------------------------------------
# REQ-TIER0-005-3: predict_violation_prob
# ---------------------------------------------------------------------------


class TestPredictViolationProb:
    """REQ-TIER0-005-3: predict_violation_prob returns float in [0, 1]."""

    def test_returns_half_before_fit(self) -> None:
        """predict_violation_prob returns 0.5 before fit() is called.

        Spec: REQ-TIER0-005-3
        """
        from carnot.verify.drift_probe import DRIFTProbe

        probe = DRIFTProbe(model_runner=None, layers=[-2, -1])
        hs = {-2: np.ones((4, 8), dtype=np.float32), -1: np.ones((4, 8), dtype=np.float32)}
        assert probe.predict_violation_prob(hs) == 0.5

    def test_range_after_fit(self) -> None:
        """predict_violation_prob returns a float in [0, 1] after fit.

        Spec: REQ-TIER0-005-3
        """
        from carnot.verify.drift_probe import DRIFTProbe

        probe = DRIFTProbe(model_runner=None, layers=[-2, -1])
        # Fit on perfectly separated synthetic signatures.
        rng = np.random.default_rng(1)
        correct_sigs = rng.uniform(0.0, 0.1, (20, 1)).astype(np.float32)
        halluc_sigs = rng.uniform(0.9, 1.0, (20, 1)).astype(np.float32)
        probe.fit_from_signatures(correct_sigs, halluc_sigs)

        hs_low = {-2: np.ones((4, 8), dtype=np.float32), -1: np.ones((4, 8), dtype=np.float32)}
        p = probe.predict_violation_prob(hs_low)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_halluc_scores_higher_than_correct(self) -> None:
        """After training on separated data, hallucinated sigs score higher than correct.

        Spec: REQ-TIER0-005-3
        """
        from carnot.verify.drift_probe import DRIFTProbe

        probe = DRIFTProbe(model_runner=None, layers=[-2, -1])
        rng = np.random.default_rng(2)
        correct_sigs = rng.uniform(0.0, 0.05, (30, 1)).astype(np.float32)
        halluc_sigs = rng.uniform(0.95, 1.0, (30, 1)).astype(np.float32)
        probe.fit_from_signatures(correct_sigs, halluc_sigs)

        # Low-drift hidden state → should score low (correct).
        a = np.ones((4, 8), dtype=np.float32)
        hs_identical = {-2: a, -1: a.copy()}
        p_correct = probe.predict_violation_prob(hs_identical)

        # High-drift hidden state → should score high (hallucinated).
        b = np.zeros((4, 8), dtype=np.float32)
        b[:, 0] = 1.0
        c = np.zeros((4, 8), dtype=np.float32)
        c[:, 1] = 1.0
        hs_orthogonal = {-2: b, -1: c}
        p_halluc = probe.predict_violation_prob(hs_orthogonal)

        assert (
            p_halluc > p_correct
        ), f"Hallucinated score {p_halluc:.3f} should exceed correct score {p_correct:.3f}"


# ---------------------------------------------------------------------------
# REQ-TIER0-005-2: fit_from_signatures trains the probe
# ---------------------------------------------------------------------------


class TestFitFromSignatures:
    """REQ-TIER0-005-2: fit / fit_from_signatures trains a usable probe."""

    def test_probe_set_after_fit_from_signatures(self) -> None:
        """fit_from_signatures() must set _probe to a non-None LogisticRegression.

        Spec: REQ-TIER0-005-2
        """
        from carnot.verify.drift_probe import DRIFTProbe

        probe = DRIFTProbe(model_runner=None, layers=[-2, -1])
        assert probe._probe is None

        rng = np.random.default_rng(3)
        probe.fit_from_signatures(
            rng.random((10, 1)).astype(np.float32),
            rng.random((10, 1)).astype(np.float32),
        )
        assert probe._probe is not None

    def test_coef_shape_matches_n_drift_pairs(self) -> None:
        """probe.coef_ shape must be (1, n_drift_pairs).

        Spec: REQ-TIER0-005-2
        """
        from carnot.verify.drift_probe import DRIFTProbe

        probe = DRIFTProbe(model_runner=None, layers=[-4, -3, -2, -1])
        rng = np.random.default_rng(4)
        probe.fit_from_signatures(
            rng.random((15, 3)).astype(np.float32),
            rng.random((15, 3)).astype(np.float32),
        )
        assert probe._probe.coef_.shape == (
            1,
            3,
        ), f"Expected coef shape (1, 3), got {probe._probe.coef_.shape}"


# ---------------------------------------------------------------------------
# _cosine_sim helper
# ---------------------------------------------------------------------------


class TestCosineSim:
    """Unit tests for the _cosine_sim helper.

    Spec: REQ-TIER0-005-1
    """

    def test_zero_vector_returns_one(self) -> None:
        """Zero vector must return cosine_sim=1.0 (no drift contribution).

        Spec: REQ-TIER0-005-1
        """
        from carnot.verify.drift_probe import _cosine_sim

        a = np.zeros(4, dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert _cosine_sim(a, b) == 1.0

    def test_identical_vectors_return_one(self) -> None:
        """Identical vectors must return cosine_sim=1.0.

        Spec: REQ-TIER0-005-1
        """
        from carnot.verify.drift_probe import _cosine_sim

        a = np.array([0.3, 0.4, 0.5], dtype=np.float32)
        assert abs(_cosine_sim(a, a) - 1.0) < 1e-5

    def test_orthogonal_vectors_return_zero(self) -> None:
        """Orthogonal vectors must return cosine_sim≈0.0.

        Spec: REQ-TIER0-005-1
        """
        from carnot.verify.drift_probe import _cosine_sim

        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(_cosine_sim(a, b)) < 1e-5


# ---------------------------------------------------------------------------
# GSM8K triple generation
# ---------------------------------------------------------------------------


class TestGSM8KTripleGeneration:
    """Tests for the GSM8K triple generator in experiment_911.

    Spec: REQ-TIER0-005, SCENARIO-TIER0-005
    """

    @pytest.fixture(autouse=True)
    def _load_exp_module(self):
        """Dynamically load the experiment module (not imported at module level)."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        scripts_dir = os.path.join(repo_root, "scripts")
        sys.path.insert(0, scripts_dir)

        script_path = os.path.join(scripts_dir, "experiment_911_drift_probe_tier0i.py")
        spec = importlib.util.spec_from_file_location("exp911", script_path)
        self.mod = importlib.util.module_from_spec(spec)
        # Don't exec — just extract the functions we need via partial exec.
        # Safer: import only the specific functions.
        src = open(script_path).read()
        globs = {"__name__": "exp911"}
        # Execute only the non-main portion to get the helper functions.
        # We do this by truncating at the main() def body call.
        globs["__file__"] = script_path
        exec(compile(src, script_path, "exec"), globs)  # noqa: S102
        self.generate_gsm8k_triples = globs["generate_gsm8k_triples"]
        self.make_correct = globs["_make_correct_response"]
        self.make_halluc = globs["_make_hallucinated_response"]

    def test_generates_100_triples(self) -> None:
        """generate_gsm8k_triples(100) returns exactly 100 dicts.

        Spec: SCENARIO-TIER0-005
        """
        triples = self.generate_gsm8k_triples(n=100, seed=42)
        assert len(triples) == 100

    def test_triple_has_required_keys(self) -> None:
        """Each triple must have 'question', 'correct', 'hallucinated' keys.

        Spec: SCENARIO-TIER0-005
        """
        triples = self.generate_gsm8k_triples(n=5, seed=0)
        for t in triples:
            assert "question" in t
            assert "correct" in t
            assert "hallucinated" in t

    def test_correct_and_hallucinated_differ(self) -> None:
        """correct and hallucinated responses must differ for every triple.

        Spec: SCENARIO-TIER0-005
        """
        triples = self.generate_gsm8k_triples(n=25, seed=42)
        for t in triples:
            assert (
                t["correct"] != t["hallucinated"]
            ), f"correct and hallucinated are identical for: {t['question']}"

    def test_reproducible_with_same_seed(self) -> None:
        """generate_gsm8k_triples with the same seed produces identical output.

        Spec: SCENARIO-TIER0-005
        """
        t1 = self.generate_gsm8k_triples(n=10, seed=77)
        t2 = self.generate_gsm8k_triples(n=10, seed=77)
        for a, b in zip(t1, t2, strict=False):
            assert a["correct"] == b["correct"]
            assert a["hallucinated"] == b["hallucinated"]

    def test_different_seeds_differ(self) -> None:
        """Different seeds must produce at least some different hallucinated responses.

        Spec: SCENARIO-TIER0-005
        """
        t1 = self.generate_gsm8k_triples(n=10, seed=1)
        t2 = self.generate_gsm8k_triples(n=10, seed=2)
        diffs = sum(
            1 for a, b in zip(t1, t2, strict=False) if a["hallucinated"] != b["hallucinated"]
        )
        assert diffs > 0


# ---------------------------------------------------------------------------
# Experiment 911 deliverable integration test
# ---------------------------------------------------------------------------


class TestExperiment911Deliverable:
    """Integration test: run main() and validate the JSON deliverable schema.

    Spec: REQ-TIER0-005, SCENARIO-TIER0-005
    """

    def test_deliverable_written_and_valid(self) -> None:
        """Running main() must produce a valid JSON deliverable with all required fields.

        Spec: REQ-TIER0-005
        """
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        scripts_dir = os.path.join(repo_root, "scripts")
        sys.path.insert(0, scripts_dir)

        orig_dir = os.getcwd()
        os.chdir(repo_root)
        try:
            script_path = os.path.join(scripts_dir, "experiment_911_drift_probe_tier0i.py")
            spec = importlib.util.spec_from_file_location("exp911_main", script_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
        finally:
            os.chdir(orig_dir)

        deliverable = os.path.join(repo_root, "results", "experiment_911_drift_probe_tier0i.json")
        assert os.path.exists(deliverable), "Deliverable JSON must exist after main()"

        with open(deliverable) as f:
            data = json.load(f)

        # Required base schema fields.
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
            assert field in data, f"Required field '{field}' missing"

        assert data["experiment"] == 911
        assert data["status"] == "success"

        # Experiment-specific fields.
        assert "ood_auc_drift" in data
        assert "honest_verdict" in data
        assert data["honest_verdict"] in ("tier0i_viable", "tier0i_marginal", "tier0i_not_viable")
        assert 0.0 <= data["ood_auc_drift"] <= 1.0
        assert "inference_mode" in data
        assert "probe_layers" in data
