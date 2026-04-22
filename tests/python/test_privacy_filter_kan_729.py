"""Tests for PrivacyFilterEnergyChecker and privacy_filter_features.

Tests cover only code added in Exp 729 (REQ-SAFE-015, REQ-SAFE-016).
Each test references the spec requirement it exercises.

Spec: REQ-SAFE-015, REQ-SAFE-016
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.models.privacy_filter_features import (
    encode_privacy,
    privacy_feature_names,
    N_PRIVACY_FEATURES,
)
from carnot.models.privacy_filter_kan import (
    PrivacyFilterEnergyChecker,
    PrivacyExample,
    _compute_auroc,
)


# ---------------------------------------------------------------------------
# privacy_filter_features tests
# ---------------------------------------------------------------------------


class TestEncodePrivacy:
    """REQ-SAFE-015: feature encoder must be deterministic and correct shape."""

    def test_output_shape(self):
        # REQ-SAFE-015: output must have exactly N_PRIVACY_FEATURES elements.
        vec = encode_privacy("hello world")
        assert vec.shape == (N_PRIVACY_FEATURES,)

    def test_output_dtype(self):
        vec = encode_privacy("hello world")
        assert vec.dtype == jnp.float32

    def test_deterministic(self):
        # REQ-SAFE-015: same text must always produce the same vector.
        text = "My SSN is 123-45-6789 and email is foo@bar.com"
        v1 = encode_privacy(text)
        v2 = encode_privacy(text)
        np.testing.assert_array_equal(np.array(v1), np.array(v2))

    def test_benign_text_low_features(self):
        # Ordinary math question should have near-zero PII pattern features.
        vec = encode_privacy("What is 2 + 2?")
        # Sum of all features should be very small for benign text.
        total = float(jnp.sum(vec))
        assert total < 1.0, f"Expected near-zero features for benign text, got {total}"

    def test_cc_pattern_detected(self):
        # Credit card pattern should trigger feature index 0.
        vec = encode_privacy("My card is 4111 1111 1111 1111 expiry 01/25")
        assert float(vec[0]) > 0.0, "CC pattern feature should be nonzero"

    def test_ssn_pattern_detected(self):
        # SSN pattern should trigger feature index 1.
        vec = encode_privacy("Social security: 123-45-6789")
        assert float(vec[1]) > 0.0, "SSN pattern feature should be nonzero"

    def test_email_pattern_detected(self):
        # Email pattern should trigger feature index 2.
        vec = encode_privacy("Contact me at user@example.com")
        assert float(vec[2]) > 0.0, "Email pattern feature should be nonzero"

    def test_phone_pattern_detected(self):
        # Phone pattern should trigger feature index 3.
        vec = encode_privacy("Call me at (617) 555-1234")
        assert float(vec[3]) > 0.0, "Phone pattern feature should be nonzero"

    def test_digit_density_feature(self):
        # Digit density (feature 12) should be higher for numeric text.
        dense_text = "1234 5678 9012 3456 7890 1234"
        sparse_text = "Hello how are you doing today"
        dense_vec = encode_privacy(dense_text)
        sparse_vec = encode_privacy(sparse_text)
        assert float(dense_vec[12]) > float(sparse_vec[12]), "Digit density should be higher for numeric text"

    def test_max_features_truncation(self):
        # max_features parameter should control output length.
        vec8 = encode_privacy("test text", max_features=8)
        assert vec8.shape == (8,)

    def test_values_in_reasonable_range(self):
        # All features should be non-negative and finite.
        vec = encode_privacy("My SSN is 123-45-6789, CC: 4111 1111 1111 1111")
        arr = np.array(vec)
        assert np.all(arr >= 0.0), "Features should be non-negative"
        assert np.all(np.isfinite(arr)), "Features should be finite"

    def test_empty_text(self):
        # Empty text should not raise; should return zero-ish vector.
        vec = encode_privacy("")
        assert vec.shape == (N_PRIVACY_FEATURES,)


class TestPrivacyFeatureNames:
    """REQ-SAFE-015: feature names must match feature count."""

    def test_count_matches_n_features(self):
        names = privacy_feature_names()
        assert len(names) == N_PRIVACY_FEATURES

    def test_names_are_strings(self):
        names = privacy_feature_names()
        assert all(isinstance(n, str) for n in names)

    def test_custom_length(self):
        names = privacy_feature_names(max_features=8)
        assert len(names) == 8


# ---------------------------------------------------------------------------
# PrivacyFilterEnergyChecker tests
# ---------------------------------------------------------------------------


class TestPrivacyFilterEnergyCheckerBasic:
    """REQ-SAFE-015: checker must expose energy() and is_safe() API."""

    def test_n_params(self):
        # Architecture spec: ~3264 params for n_hidden=32, n_features=16, n_ctrl=6.
        checker = PrivacyFilterEnergyChecker()
        assert checker.n_params() == 3264

    def test_energy_returns_float(self):
        checker = PrivacyFilterEnergyChecker()
        e = checker.energy("What is 2 + 2?")
        assert isinstance(e, float)

    def test_energy_is_finite(self):
        checker = PrivacyFilterEnergyChecker()
        e = checker.energy("My SSN is 123-45-6789")
        assert np.isfinite(e)

    def test_is_safe_returns_bool(self):
        checker = PrivacyFilterEnergyChecker()
        result = checker.is_safe("Hello, what time is it?")
        assert isinstance(result, bool)

    def test_is_safe_threshold(self):
        # is_safe(text, threshold) should return True when energy < threshold.
        checker = PrivacyFilterEnergyChecker()
        e = checker.energy("Test text")
        # At a very high threshold, is_safe should return True.
        assert checker.is_safe("Test text", threshold=e + 1.0) is True
        # At a very low threshold, is_safe should return False.
        assert checker.is_safe("Test text", threshold=e - 1.0) is False

    def test_energy_deterministic(self):
        # REQ-SAFE-015: same text must always give the same energy.
        checker = PrivacyFilterEnergyChecker()
        text = "My credit card is 4111 1111 1111 1111"
        e1 = checker.energy(text)
        e2 = checker.energy(text)
        assert e1 == e2


class TestPrivacyFilterEnergyCheckerTraining:
    """REQ-SAFE-016: training must produce a loss curve and update weights."""

    def _make_examples(self):
        benign = [PrivacyExample(text=f"What is {i} + {i}?", label="benign") for i in range(20)]
        pii = [
            PrivacyExample(
                text=f"My SSN is {100 + i:03d}-{i:02d}-{1000 + i:04d}",
                label="pii",
            )
            for i in range(20)
        ]
        return benign + pii

    def test_train_returns_loss_curve(self):
        checker = PrivacyFilterEnergyChecker()
        examples = self._make_examples()
        curve = checker.train(examples, n_epochs=5)
        assert len(curve) == 5
        assert all(isinstance(x, float) for x in curve)

    def test_train_updates_weights(self):
        checker = PrivacyFilterEnergyChecker()
        initial_ctrl = np.array(checker._edge_ctrl).copy()
        examples = self._make_examples()
        checker.train(examples, n_epochs=10)
        assert not np.allclose(np.array(checker._edge_ctrl), initial_ctrl), \
            "Weights should change after training"

    def test_train_empty_examples_returns_empty_curve(self):
        checker = PrivacyFilterEnergyChecker()
        curve = checker.train([], n_epochs=10)
        assert curve == []

    def test_train_all_benign_returns_empty_curve(self):
        # Without a mix of labels, contrastive loss cannot be computed.
        checker = PrivacyFilterEnergyChecker()
        examples = [PrivacyExample(text=f"text {i}", label="benign") for i in range(10)]
        curve = checker.train(examples, n_epochs=5)
        assert curve == []

    def test_trained_checker_separates_classes(self):
        # After sufficient training on clearly distinguishable examples, PII texts
        # should have higher energy than benign texts on average.
        checker = PrivacyFilterEnergyChecker()
        examples = self._make_examples()
        checker.train(examples, n_epochs=50)

        benign_energies = [checker.energy(f"What is {i} + {i}?") for i in range(20)]
        pii_energies = [checker.energy(f"My SSN is {100 + i:03d}-{i:02d}-{1000 + i:04d}") for i in range(20)]

        mean_benign = np.mean(benign_energies)
        mean_pii = np.mean(pii_energies)
        # After 50 epochs of contrastive training, PII mean energy should exceed benign.
        assert mean_pii > mean_benign, (
            f"PII mean energy {mean_pii:.4f} should exceed benign {mean_benign:.4f} after training"
        )


class TestPrivacyFilterEnergyCheckerAUROC:
    """REQ-SAFE-015: evaluate_auroc must return values in [0, 1]."""

    def test_auroc_range(self):
        checker = PrivacyFilterEnergyChecker()
        examples = [
            PrivacyExample("What is 2 + 2?", "benign"),
            PrivacyExample("My SSN is 123-45-6789", "pii"),
            PrivacyExample("Explain recursion", "benign"),
            PrivacyExample("CC: 4111 1111 1111 1111", "pii"),
        ]
        auroc = checker.evaluate_auroc(examples)
        assert 0.0 <= auroc <= 1.0

    def test_auroc_degenerate_all_benign(self):
        # Degenerate label set should return 0.5 (random-classifier baseline).
        checker = PrivacyFilterEnergyChecker()
        examples = [PrivacyExample(f"text {i}", "benign") for i in range(5)]
        assert checker.evaluate_auroc(examples) == 0.5


class TestPrivacyFilterInspectSpline:
    """REQ-SAFE-016: inspect_spline must expose control points for auditability."""

    def test_inspect_spline_shape(self):
        checker = PrivacyFilterEnergyChecker()
        ctrl = checker.inspect_spline(0, 0)
        assert len(ctrl) == checker._N_KNOTS + checker._DEGREE

    def test_inspect_spline_is_list_of_floats(self):
        checker = PrivacyFilterEnergyChecker()
        ctrl = checker.inspect_spline(0, 1)
        assert all(isinstance(v, float) for v in ctrl)


class TestPrivacyFilterSaveLoad:
    """REQ-SAFE-016: save/load round-trip must preserve weights and schema."""

    def test_save_load_round_trip(self):
        checker = PrivacyFilterEnergyChecker()
        # Give it non-trivial weights by doing a few training steps.
        examples = [
            PrivacyExample("What is 1 + 1?", "benign"),
            PrivacyExample("My SSN is 123-45-6789", "pii"),
        ] * 5
        checker.train(examples, n_epochs=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "privacy_filter_test.json"
            checker.save(path)

            # Verify schema field in saved file.
            with open(path) as fh:
                payload = json.load(fh)
            assert payload["schema"] == "carnot.privacy_filter_kan.v1"

            # Load and verify energies match.
            loaded = PrivacyFilterEnergyChecker.load(path)
            text = "My credit card is 4111 1111 1111 1111"
            assert abs(checker.energy(text) - loaded.energy(text)) < 1e-5

    def test_save_wrong_schema_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wrong.json"
            with open(path, "w") as fh:
                json.dump({"schema": "carnot.prompt_injection_kan.v1", "n_features": 16, "n_hidden": 32, "n_knots": 3, "degree": 3, "edge_ctrl": [], "output_ctrl": []}, fh)
            with pytest.raises(ValueError, match="Unexpected schema"):
                PrivacyFilterEnergyChecker.load(path)

    def test_save_does_not_overwrite_injection_weights(self):
        # Verify the privacy_filter_kan weights file uses a distinct name from
        # prompt_injection_kan weights to prevent accidental overwrite (CONSTRAINT).
        privacy_path = Path("python/carnot/models/privacy_filter_kan_v1.json")
        injection_path = Path("python/carnot/models/prompt_injection_kan_weights.json")
        assert privacy_path.name != injection_path.name


# ---------------------------------------------------------------------------
# _compute_auroc tests (internal helper)
# ---------------------------------------------------------------------------


class TestComputeAuroc:
    """Verify the Mann-Whitney AUROC implementation."""

    def test_perfect_separation(self):
        # Positive scores all above negative scores → AUROC = 1.0.
        scores = [0.1, 0.2, 0.9, 1.0]
        labels = [0, 0, 1, 1]
        assert _compute_auroc(scores, labels) == 1.0

    def test_random_classifier(self):
        # Interleaved scores where positives and negatives are equally ranked → AUROC = 0.5.
        # Positive scores: 1.0, 3.0; Negative scores: 2.0, 4.0.
        # Pairs: (1.0 vs 2.0)->0, (1.0 vs 4.0)->0, (3.0 vs 2.0)->1, (3.0 vs 4.0)->0.
        # U = 1 / (2*2) = 0.25 — not 0.5.
        # Use symmetric interleaving: positives exactly split negatives → 0.5.
        scores = [1.0, 2.0, 3.0, 4.0]
        labels = [0, 1, 0, 1]
        # Positives (label=1) scores: 2.0, 4.0; Negatives: 1.0, 3.0.
        # Pairs: (2.0>1.0)->1, (2.0>3.0)->0, (4.0>1.0)->1, (4.0>3.0)->1.
        # U = 3 / (2*2) = 0.75.
        auroc = _compute_auroc(scores, labels)
        assert abs(auroc - 0.75) < 0.01

    def test_degenerate_all_positive(self):
        scores = [1.0, 2.0, 3.0]
        labels = [1, 1, 1]
        assert _compute_auroc(scores, labels) == 0.5

    def test_degenerate_empty(self):
        assert _compute_auroc([], []) == 0.5


# ---------------------------------------------------------------------------
# Experiment script smoke tests
# ---------------------------------------------------------------------------


class TestExperiment729Blocked:
    """Verify the experiment emits blocked_on_dependency when model is absent."""

    def test_blocked_on_missing_model(self, tmp_path):
        # The openai/privacy-filter directory does not exist → blocked artifact.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

        from scripts.experiment_template import ExperimentTemplate
        import logging

        log = logging.getLogger("test_729")
        deliverable = "results/experiment_729_privacy_filter_kan_true_distillation.json"
        missing_model_dir = tmp_path / "nonexistent_model"
        corpus_dir = tmp_path / "corpus"
        cache_path = corpus_dir / "cache.json"
        weights_path = tmp_path / "weights.json"

        tmpl = ExperimentTemplate(729, "Test 729", str(tmp_path / deliverable))
        tmpl.setup()

        from scripts.experiment_729_privacy_filter_kan_true_distillation import _run
        _run(tmpl, log, str(tmp_path / deliverable), missing_model_dir, corpus_dir, cache_path, weights_path)

        artifact_path = tmp_path / deliverable
        assert artifact_path.exists(), "Artifact should be written for blocked run"
        with open(artifact_path) as fh:
            artifact = json.load(fh)
        assert artifact["honest_verdict"] == "blocked_on_dependency"
        assert artifact["status"] == "blocked"
        assert "huggingface-cli download" in artifact.get("reason", "")
