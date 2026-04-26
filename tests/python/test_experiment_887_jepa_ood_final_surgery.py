"""Tests for Experiment 887: RETRO-JEPA-OOD Final Surgery.

Spec traces: REQ-LEARN-050

**Coverage targets (code added in Exp 887):**
    - load_encoder_from_safetensors: loads weights, correct shape, raises on missing file
    - VJEPAPretrainedJEPA._get_mu: returns shape (latent_dim,) for a single input
    - VJEPAPretrainedJEPA.predict: returns float in [0, 1]
    - VJEPAPretrainedJEPA.get_cls_params / set_cls_params: round-trip
    - VJEPAPretrainedJEPA.train: only classifier params change (encoder frozen)
    - evaluate_on_split: returns 0.5 on empty corpus, float on non-empty
    - retire_discriminative_jepa: writes entry to manifest (idempotent)
    - run_experiment (via mock): blocked artifact when no safetensors present
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from experiment_887_jepa_ood_final_surgery import (
    VJEPAPretrainedJEPA,
    evaluate_on_split,
    generate_arc_synthetic,
    generate_gsm8k_synthetic,
    generate_svamp_synthetic,
    load_encoder_from_safetensors,
    retire_discriminative_jepa,
    run_experiment,
    split_by_question_id,
)
from python.carnot.models.vjepa_predictor import (
    VariationalEncoder,
    build_tfidf_features,
    prepare_corpus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_corpus(n: int = 8) -> list[dict[str, Any]]:
    """Build a minimal feature/label corpus for training smoke-tests."""
    feats = [[float(i % 3)] * 50 for i in range(n)]
    labels = [i % 2 for i in range(n)]
    return [{"feature": f, "context": f, "label": lbl} for f, lbl in zip(feats, labels)]


def _make_safetensors(path: Path, in_dim: int = 50, latent_dim: int = 32) -> None:
    """Write a minimal safetensors file with encoder keys (enc_*) for testing."""
    from safetensors.numpy import save_file

    enc = VariationalEncoder(in_dim=in_dim, latent_dim=latent_dim)
    enc_params = enc.get_params()
    tensors = {f"enc_{k}": np.array(v) for k, v in enc_params.items()}
    # Also include classifier weights so the file looks like a full VJEPA save
    tensors["w_cls"] = np.zeros((latent_dim, 1), dtype=np.float32)
    tensors["b_cls"] = np.zeros(1, dtype=np.float32)
    save_file(tensors, str(path))


# ---------------------------------------------------------------------------
# REQ-LEARN-050: Encoder loading
# ---------------------------------------------------------------------------


class TestLoadEncoderFromSafetensors:
    """REQ-LEARN-050 — encoder loading from safetensors."""

    def test_loads_weights_from_valid_file(self, tmp_path: Path) -> None:
        """Encoder weights loaded from file match expected shapes."""
        sf = tmp_path / "test.safetensors"
        _make_safetensors(sf, in_dim=50, latent_dim=32)
        enc = load_encoder_from_safetensors(sf)
        assert enc.w1.shape == (50, 128)
        assert enc.w_mu.shape == (64, 32)

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """FileNotFoundError raised when safetensors file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_encoder_from_safetensors(tmp_path / "nonexistent.safetensors")

    def test_raises_on_missing_enc_keys(self, tmp_path: Path) -> None:
        """KeyError raised when file has no enc_* keys."""
        from safetensors.numpy import save_file

        sf = tmp_path / "bad.safetensors"
        save_file({"some_key": np.zeros((4,), dtype=np.float32)}, str(sf))
        with pytest.raises(KeyError):
            load_encoder_from_safetensors(sf)

    def test_in_dim_inferred_correctly(self, tmp_path: Path) -> None:
        """in_dim is inferred from enc_w1 shape, not hardcoded."""
        sf = tmp_path / "test_30.safetensors"
        _make_safetensors(sf, in_dim=30, latent_dim=16)
        enc = load_encoder_from_safetensors(sf)
        assert enc.in_dim == 30
        assert enc.latent_dim == 16


# ---------------------------------------------------------------------------
# REQ-LEARN-050: VJEPAPretrainedJEPA — frozen encoder, trainable head
# ---------------------------------------------------------------------------


class TestVJEPAPretrainedJEPA:
    """REQ-LEARN-050 — classifier-only training with frozen VJEPA encoder."""

    def setup_method(self) -> None:
        self.enc = VariationalEncoder(in_dim=50, latent_dim=32)
        self.model = VJEPAPretrainedJEPA(encoder=self.enc, latent_dim=32)

    def test_get_mu_shape(self) -> None:
        """_get_mu returns (latent_dim,) for a single feature vector."""
        x = jnp.zeros(50)
        mu = self.model._get_mu(x)
        assert mu.shape == (32,)

    def test_predict_returns_float_in_01(self) -> None:
        """predict() returns a float in [0, 1] for any input."""
        x = jnp.ones(50)
        p = self.model.predict(x)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_predict_batch_shape(self) -> None:
        """_get_mu works for a batch of inputs (batch, in_dim)."""
        xs = jnp.ones((4, 50))
        mus = self.model._get_mu(xs)
        assert mus.shape == (4, 32)

    def test_cls_params_round_trip(self) -> None:
        """get_cls_params / set_cls_params round-trip preserves values."""
        params = self.model.get_cls_params()
        params["w_cls"] = params["w_cls"] + 1.0
        self.model.set_cls_params(params)
        params2 = self.model.get_cls_params()
        np.testing.assert_allclose(np.array(params2["w_cls"]), np.array(params["w_cls"]))

    def test_encoder_weights_unchanged_after_training(self) -> None:
        """Encoder parameters must not change after training (frozen check)."""
        enc_w1_before = np.array(self.enc.w1).copy()
        corpus = _tiny_corpus(8)
        self.model.train(corpus, n_epochs=5, lr=1e-2)
        enc_w1_after = np.array(self.enc.w1)
        np.testing.assert_array_equal(enc_w1_before, enc_w1_after)

    def test_classifier_weights_change_after_training(self) -> None:
        """Classifier parameters MUST change after training (not frozen)."""
        w_cls_before = np.array(self.model.w_cls).copy()
        corpus = _tiny_corpus(8)
        self.model.train(corpus, n_epochs=5, lr=1e-2)
        w_cls_after = np.array(self.model.w_cls)
        assert not np.allclose(w_cls_before, w_cls_after), "w_cls must change during training"

    def test_train_returns_epoch_losses(self) -> None:
        """train() returns a list of per-epoch losses of length n_epochs."""
        corpus = _tiny_corpus(8)
        losses = self.model.train(corpus, n_epochs=10)
        assert len(losses) == 10
        assert all(isinstance(v, float) for v in losses)

    def test_train_empty_corpus_returns_empty(self) -> None:
        """train() on an empty corpus returns an empty list without error."""
        losses = self.model.train([], n_epochs=10)
        assert losses == []


# ---------------------------------------------------------------------------
# REQ-LEARN-050: evaluate_on_split
# ---------------------------------------------------------------------------


class TestEvaluateOnSplit:
    """REQ-LEARN-050 — AUC evaluation helper."""

    def setup_method(self) -> None:
        enc = VariationalEncoder(in_dim=50, latent_dim=32)
        self.model = VJEPAPretrainedJEPA(encoder=enc, latent_dim=32)

    def test_returns_half_on_empty_corpus(self) -> None:
        """evaluate_on_split returns 0.5 for an empty corpus (degenerate case)."""
        auc = evaluate_on_split(self.model, [])
        assert auc == 0.5

    def test_returns_float_on_non_empty(self) -> None:
        """evaluate_on_split returns a float in [0, 1] for a non-empty corpus."""
        corpus = _tiny_corpus(8)
        auc = evaluate_on_split(self.model, corpus)
        assert isinstance(auc, float)
        assert 0.0 <= auc <= 1.0

    def test_all_same_label_returns_half(self) -> None:
        """All-same-label corpus returns 0.5 (no discrimination possible)."""
        corpus = [{"feature": [0.0] * 50, "context": [0.0] * 50, "label": 1}] * 4
        auc = evaluate_on_split(self.model, corpus)
        assert auc == 0.5


# ---------------------------------------------------------------------------
# REQ-LEARN-050: retire_discriminative_jepa
# ---------------------------------------------------------------------------


class TestRetireDiscriminativeJepa:
    """REQ-LEARN-050 — exclusion manifest update."""

    def test_appends_entry_to_manifest(self, tmp_path: Path) -> None:
        """retire_discriminative_jepa appends exp887 entry to manifest."""
        manifest = tmp_path / "exclusion_manifest.yaml"
        manifest.write_text("retired:\n")
        retire_discriminative_jepa(manifest)
        content = manifest.read_text()
        assert "experiment_id: 887" in content

    def test_idempotent_no_duplicate(self, tmp_path: Path) -> None:
        """Calling retire_discriminative_jepa twice does not create a duplicate entry."""
        manifest = tmp_path / "exclusion_manifest.yaml"
        manifest.write_text("retired:\n")
        retire_discriminative_jepa(manifest)
        retire_discriminative_jepa(manifest)
        count = manifest.read_text().count("experiment_id: 887")
        assert count == 1

    def test_includes_prior_exp_ids(self, tmp_path: Path) -> None:
        """Manifest entry mentions prior discriminative JEPA experiment IDs."""
        manifest = tmp_path / "exclusion_manifest.yaml"
        manifest.write_text("retired:\n")
        retire_discriminative_jepa(manifest)
        content = manifest.read_text()
        assert "experiment_id: 783" in content
        assert "experiment_id: 834" in content


# ---------------------------------------------------------------------------
# REQ-LEARN-050: run_experiment — blocked artifact path
# ---------------------------------------------------------------------------


class TestRunExperimentBlocked:
    """REQ-LEARN-050 — run_experiment writes blocked artifact when no safetensors."""

    def test_blocked_when_no_safetensors(self, tmp_path: Path) -> None:
        """run_experiment writes a blocked artifact when vjepa safetensors are absent."""
        result_path = tmp_path / "experiment_887_jepa_ood_final_surgery.json"
        with (
            patch(
                "experiment_887_jepa_ood_final_surgery.SAFETENSORS_V2",
                tmp_path / "nonexistent_v2.safetensors",
            ),
            patch(
                "experiment_887_jepa_ood_final_surgery.SAFETENSORS_V1",
                tmp_path / "nonexistent_v1.safetensors",
            ),
            patch(
                "experiment_887_jepa_ood_final_surgery.RESULT_PATH",
                result_path,
            ),
        ):
            from experiment_887_jepa_ood_final_surgery import run_experiment as _run

            result = _run()

        assert result["honest_verdict"] == "blocked"
        assert result["blocked_by"] == "vjepa_model_not_found"
        assert result_path.exists()
        artifact = json.loads(result_path.read_text())
        assert artifact["honest_verdict"] == "blocked"


# ---------------------------------------------------------------------------
# REQ-LEARN-050: integration — train from real safetensors (if present)
# ---------------------------------------------------------------------------


class TestRunExperimentIntegration:
    """REQ-LEARN-050 — full run_experiment with real VJEPA v2 safetensors."""

    def test_produces_valid_artifact(self, tmp_path: Path) -> None:
        """run_experiment writes a valid artifact JSON with all required fields."""
        real_sf_v2 = _ROOT / "results" / "vjepa_predictor_v2.safetensors"
        real_sf_v1 = _ROOT / "results" / "vjepa_predictor_v1.safetensors"
        if not real_sf_v2.exists() and not real_sf_v1.exists():
            pytest.skip("No vjepa safetensors available — skipping integration test")

        result_path = tmp_path / "experiment_887_jepa_ood_final_surgery.json"

        # Use a temp manifest so we don't mutate the real one
        tmp_manifest = tmp_path / "exclusion_manifest.yaml"
        tmp_manifest.write_text("retired:\n")

        with (
            patch("experiment_887_jepa_ood_final_surgery.RESULT_PATH", result_path),
            patch(
                "experiment_887_jepa_ood_final_surgery._ROOT",
                _ROOT,
            ),
        ):
            # Patch retire to use tmp manifest, avoiding real-file side-effects
            import experiment_887_jepa_ood_final_surgery as mod887

            orig_retire = mod887.retire_discriminative_jepa

            def _safe_retire(manifest_path: Path) -> None:
                orig_retire(tmp_manifest)

            with patch.object(mod887, "retire_discriminative_jepa", _safe_retire):
                result = mod887.run_experiment()

        required_fields = [
            "experiment",
            "schema",
            "run_date",
            "honest_verdict",
            "in_dist_auc",
            "ood_auc",
            "svamp_auc",
            "encoder_frozen",
            "discriminative_jepa_retired",
            "n_epochs",
            "n_training_pairs",
            "spec",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        assert result["experiment"] == 887
        assert result["encoder_frozen"] is True
        assert result["honest_verdict"] in (
            "retro_jepa_ood_closed",
            "marginal",
            "jepa_discriminative_retired",
        )
        assert 0.0 <= result["in_dist_auc"] <= 1.0
        assert 0.0 <= result["ood_auc"] <= 1.0
        assert 0.0 <= result["svamp_auc"] <= 1.0
        assert result_path.exists()
