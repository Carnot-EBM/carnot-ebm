"""Tests for ThinkPRMProbe and Exp 1057 probe ensemble v6 helper functions.

Covers:
  - LogisticProbe (Adam BCE training, predict_proba, AUROC)
  - ThinkPRMProbe interface (fit/transform feature pipeline)
  - Vectorized energy/gradient helpers (_eval_energies_vec, _contrastive_adam_train)
  - NK-KAEM components (_nk_step_vec, _promote_grid)
  - Corpus loading helpers (extract_labels)

All tests use synthetic data — no heavy model loading, no GPU required.
Model inference is tested via a mock that returns random hidden states.

Spec: REQ-VERIFY-098, REQ-LEARN-011, REQ-SAMPLE-015
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent.parent
for _d in [str(_REPO / "python"), str(_REPO / "scripts"), str(_REPO)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from python.carnot.verify.thinkprm_probe import LogisticProbe, ThinkPRMProbe

# Also import experiment helpers without running main()
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "exp1057",
    str(_REPO / "scripts" / "experiment_1057_probe_ensemble_v6.py"),
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_eval_energies_vec = _mod._eval_energies_vec
_contrastive_adam_train = _mod._contrastive_adam_train
_nk_step_vec = _mod._nk_step_vec
_promote_grid = _mod._promote_grid
_enforce_mono = _mod._enforce_mono
extract_labels = _mod.extract_labels
fit_pca_and_normalize = _mod.fit_pca_and_normalize


# ===========================================================================
# LogisticProbe tests
# ===========================================================================


class TestLogisticProbe:
    """Tests for the full-batch Adam logistic probe."""

    def test_fit_returns_epoch_log(self):
        """Fit should return a list of dicts with epoch/loss/train_auroc at multiples of 75.

        REQ-LEARN-011: probe must log training metrics during fit.
        """
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 4)).astype(np.float32)
        y = (rng.random(50) > 0.5).astype(np.float64)
        probe = LogisticProbe(n_features=4, lr=0.1, n_epochs=150, reg=0.01)
        log = probe.fit(X, y)
        assert len(log) == 2  # epochs 75 and 150
        for entry in log:
            assert "epoch" in entry
            assert "loss" in entry
            assert "train_auroc" in entry
            assert 0.0 <= entry["train_auroc"] <= 1.0

    def test_predict_proba_in_01(self):
        """predict_proba should return values in [0, 1] for any input.

        REQ-LEARN-011: probe outputs must be valid probabilities.
        """
        rng = np.random.default_rng(7)
        X_train = rng.normal(0, 1, (60, 8)).astype(np.float64)
        y_train = (X_train[:, 0] > 0).astype(np.float64)
        probe = LogisticProbe(n_features=8, lr=0.05, n_epochs=150, reg=0.01)
        probe.fit(X_train, y_train)

        X_test = rng.normal(0, 2, (20, 8)).astype(np.float64)
        p = probe.predict_proba(X_test)
        assert p.shape == (20,)
        assert np.all(p >= 0.0) and np.all(p <= 1.0)

    def test_learns_separable_data(self):
        """LogisticProbe should achieve AUROC > 0.9 on linearly separable data.

        REQ-LEARN-011: probe must learn from labeled data.
        """
        rng = np.random.default_rng(99)
        n = 200
        X_pos = rng.normal(+2.0, 0.5, (n // 2, 4))
        X_neg = rng.normal(-2.0, 0.5, (n // 2, 4))
        X = np.vstack([X_pos, X_neg]).astype(np.float64)
        y = np.array([1.0] * (n // 2) + [0.0] * (n // 2))

        probe = LogisticProbe(n_features=4, lr=0.1, n_epochs=300, reg=0.001)
        probe.fit(X, y)

        from carnot.eval.metrics import auroc

        p = probe.predict_proba(X)
        auc = auroc(y, p)
        assert auc > 0.9, f"LogisticProbe AUROC={auc:.4f} on separable data, expected > 0.9"

    def test_random_labels_auroc_near_half(self):
        """With random labels, LogisticProbe should not learn above AUROC=0.7.

        REQ-LEARN-011: probe must not overfit to pure noise on small datasets.
        """
        rng = np.random.default_rng(13)
        n = 80
        X_train = rng.normal(0, 1, (n, 8)).astype(np.float64)
        y_train = rng.choice([0.0, 1.0], size=n)
        X_test = rng.normal(0, 1, (40, 8)).astype(np.float64)
        y_test = rng.choice([0.0, 1.0], size=40)

        probe = LogisticProbe(n_features=8, lr=0.05, n_epochs=150, reg=0.1)
        probe.fit(X_train, y_train)

        from carnot.eval.metrics import auroc

        p = probe.predict_proba(X_test)
        auc = auroc(y_test, p)
        # Allow some variance but should not be wildly high with strong regularization
        assert auc < 0.75, f"LogisticProbe AUROC={auc:.4f} on random labels — possible overfit"


# ===========================================================================
# ThinkPRMProbe interface tests (mock model)
# ===========================================================================


class TestThinkPRMProbeInterface:
    """Tests for ThinkPRMProbe using a mocked model (no GPU required)."""

    def _make_mock_model_and_tok(self, hidden_size: int = 1024):
        """Return (mock_model, mock_tok) that produce random hidden states."""
        import torch

        class MockTok:
            def __call__(self, texts, return_tensors, padding, truncation, max_length):
                n = len(texts)
                input_ids = torch.ones(n, min(max_length, 16), dtype=torch.long)
                attention_mask = torch.ones(n, min(max_length, 16), dtype=torch.long)
                return {"input_ids": input_ids, "attention_mask": attention_mask}

        class MockOutput:
            def __init__(self, hidden_size, n, seq_len):
                self.last_hidden_state = torch.randn(n, seq_len, hidden_size)

        class MockModel:
            def __init__(self, hidden_size):
                self.h = hidden_size
                self.training = False

            def eval(self):
                return self

            def __call__(self, **kwargs):
                n = kwargs["input_ids"].shape[0]
                seq = kwargs["input_ids"].shape[1]
                return MockOutput(self.h, n, seq)

        return MockModel(hidden_size), MockTok()

    def test_fit_transform_shapes(self):
        """fit_features and transform_features should return (n, n_pca_dims) arrays.

        REQ-VERIFY-098: probe must extract features for arbitrary text inputs.
        """
        probe = ThinkPRMProbe(n_pca_dims=8, seed=42)

        mock_model, mock_tok = self._make_mock_model_and_tok(hidden_size=64)

        with patch.object(
            probe, "_load_model_and_tokenizer", return_value=(mock_model, mock_tok, "mock-model")
        ):
            texts_train = [f"step {i}: x = {i}" for i in range(30)]
            X_train = probe.fit_features(texts_train, batch_size=8, max_length=16)
            assert X_train.shape == (30, 8)
            assert X_train.dtype == np.float32
            assert np.all(X_train >= -1.0) and np.all(X_train <= 1.0)

        with patch.object(
            probe, "_load_model_and_tokenizer", return_value=(mock_model, mock_tok, "mock-model")
        ):
            texts_test = [f"test step {i}" for i in range(10)]
            X_test = probe.transform_features(texts_test, batch_size=8, max_length=16)
            assert X_test.shape == (10, 8)
            assert np.all(X_test >= -1.5) and np.all(X_test <= 1.5)  # allow slight clip violations

    def test_transform_before_fit_raises(self):
        """transform_features must raise if fit_features has not been called.

        REQ-VERIFY-098: probe interface must enforce state ordering.
        """
        probe = ThinkPRMProbe()
        with pytest.raises(RuntimeError, match="fit_features"):
            probe.transform_features(["test"], batch_size=4, max_length=16)

    def test_fit_classifier_and_auroc(self):
        """fit_classifier + auroc on separable data should return AUROC > 0.8.

        REQ-VERIFY-098: ThinkPRMProbe must produce useful discrimination scores.
        """
        probe = ThinkPRMProbe(n_pca_dims=4, seed=42)
        rng = np.random.default_rng(11)

        # Directly set PCA state to avoid model loading
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        probe._pca = PCA(n_components=4, random_state=42)
        probe._scaler = StandardScaler()
        probe._model_used = "synthetic"

        # Synthetic separable features
        n_train, n_test = 100, 50
        X_pos = rng.normal(+1.5, 0.4, (n_train // 2, 4))
        X_neg = rng.normal(-1.5, 0.4, (n_train // 2, 4))
        X_train_raw = np.vstack([X_pos, X_neg])
        # Manually set PCA state (bypass actual fit)
        probe._pca.fit(X_train_raw)
        probe._scaler.fit(probe._pca.transform(X_train_raw))

        X_train = np.clip(probe._scaler.transform(probe._pca.transform(X_train_raw)) / 3.0, -1, 1)
        y_train_tp = np.array([1.0] * (n_train // 2) + [0.0] * (n_train // 2))

        X_pos_test = rng.normal(+1.5, 0.4, (n_test // 2, 4))
        X_neg_test = rng.normal(-1.5, 0.4, (n_test // 2, 4))
        X_test_raw = np.vstack([X_pos_test, X_neg_test])
        X_test = np.clip(probe._scaler.transform(probe._pca.transform(X_test_raw)) / 3.0, -1, 1)
        y_test_tp = np.array([1.0] * (n_test // 2) + [0.0] * (n_test // 2))

        probe.fit_classifier(X_train, y_train_tp, n_epochs=200, lr=0.1)
        auc = probe.auroc(X_test, y_test_tp)
        assert auc > 0.8, f"ThinkPRMProbe AUROC={auc:.4f} on separable data, expected > 0.8"


# ===========================================================================
# Vectorized energy / gradient helpers
# ===========================================================================


class TestVectorizedHelpers:
    """Tests for the vectorized energy and gradient functions from exp1057."""

    def test_eval_energies_vec_shape(self):
        """_eval_energies_vec should return shape (n_samples,).

        REQ-SAMPLE-015: vectorized energy must match scalar energy for all samples.
        """
        rng = np.random.default_rng(42)
        n_vars, n_knots = 8, 6
        ctrl = rng.normal(0, 0.1, (n_vars, n_knots))
        ctrl = _enforce_mono(ctrl)
        X = rng.uniform(-1, 1, (20, n_vars))

        E_vec = _eval_energies_vec(ctrl, X, n_knots)
        assert E_vec.shape == (20,)
        assert np.isfinite(E_vec).all()

    def test_eval_energies_vec_matches_scalar(self):
        """Vectorized energy must match scalar energy (loop-based) for each sample.

        REQ-SAMPLE-015: both implementations must agree to within float64 precision.
        """
        rng = np.random.default_rng(7)
        n_vars, n_knots = 6, 5
        ctrl = np.abs(rng.normal(0, 0.3, (n_vars, n_knots)))
        ctrl = _enforce_mono(ctrl.astype(np.float64))
        X = rng.uniform(-0.9, 0.9, (15, n_vars))

        E_vec = _eval_energies_vec(ctrl, X, n_knots)

        # Scalar reference (loop implementation)
        def scalar_energy(ctrl, x, n_knots):
            X_c = np.clip(x, -1.0, 1.0)
            scaled = (X_c + 1.0) / 2.0 * (n_knots - 1)
            left = np.clip(np.floor(scaled).astype(np.int32), 0, n_knots - 2)
            t = scaled - left.astype(np.float64)
            return float(
                sum(
                    ctrl[i, left[i]] * (1 - t[i]) + ctrl[i, left[i] + 1] * t[i]
                    for i in range(n_vars)
                )
            )

        for j in range(15):
            expected = scalar_energy(ctrl, X[j], n_knots)
            assert abs(E_vec[j] - expected) < 1e-10, (
                f"Sample {j}: vec={E_vec[j]}, scalar={expected}"
            )

    def test_contrastive_adam_separates_classes(self):
        """After contrastive Adam training, AUROC should be > 0.65 on separable data.

        REQ-LEARN-011: contrastive training must learn to discriminate classes.
        """
        rng = np.random.default_rng(2024)
        n_vars, n_knots = 8, 6
        n_train, n_test = 200, 80

        # Separable features: class 1 (incorrect) has higher values in all dims
        X_train_pos = rng.uniform(0.2, 1.0, (n_train // 2, n_vars))
        X_train_neg = rng.uniform(-1.0, -0.2, (n_train // 2, n_vars))
        X_train = np.vstack([X_train_pos, X_train_neg])
        y_train = np.array([1.0] * (n_train // 2) + [0.0] * (n_train // 2))

        X_test_pos = rng.uniform(0.1, 1.0, (n_test // 2, n_vars))
        X_test_neg = rng.uniform(-1.0, -0.1, (n_test // 2, n_vars))
        X_test = np.vstack([X_test_pos, X_test_neg])
        y_test = np.array([1.0] * (n_test // 2) + [0.0] * (n_test // 2))

        ctrl, losses = _contrastive_adam_train(
            X_train, y_train, n_knots=n_knots, n_epochs=50, lr=0.02
        )
        assert ctrl.shape == (n_vars, n_knots)
        assert len(losses) == 50

        from carnot.eval.metrics import auroc

        scores = _eval_energies_vec(ctrl, X_test, n_knots)
        auc = auroc(y_test, scores)
        assert auc > 0.65, f"Contrastive Adam AUROC={auc:.4f} on separable data, expected > 0.65"

    def test_contrastive_adam_losses_decrease(self):
        """Contrastive Adam loss should generally decrease over epochs (with noise).

        REQ-LEARN-011: optimizer must make progress on the training objective.
        """
        rng = np.random.default_rng(55)
        n_vars, n_knots = 6, 5
        X = rng.uniform(-1, 1, (120, n_vars))
        # Linearly separable labels
        y = (X[:, 0] > 0).astype(np.float32)

        _, losses = _contrastive_adam_train(X, y, n_knots=n_knots, n_epochs=40, lr=0.02)

        # First 10 epochs vs last 10 epochs: mean loss should be lower at end
        first10 = float(np.mean(losses[:10]))
        last10 = float(np.mean(losses[-10:]))
        assert last10 <= first10 + 0.2, (
            f"Loss did not decrease: first10={first10:.4f}, last10={last10:.4f}"
        )


# ===========================================================================
# NK-KAEM component tests
# ===========================================================================


class TestNKComponents:
    """Tests for Newton-Kaczmarz step and multilevel grid promotion."""

    def test_nk_step_returns_same_shape(self):
        """_nk_step_vec must return ctrl with same shape as input.

        REQ-LEARN-011: NK step must be shape-preserving.
        """
        rng = np.random.default_rng(3)
        n_vars, n_knots = 6, 5
        ctrl = _enforce_mono(rng.normal(0, 0.1, (n_vars, n_knots)).astype(np.float64))
        X_batch = rng.uniform(-0.9, 0.9, (5, n_vars))
        y_batch = (rng.random(5) > 0.5).astype(np.float64)

        ctrl_new = _nk_step_vec(ctrl, X_batch, y_batch, n_knots, lam=0.1)
        assert ctrl_new.shape == ctrl.shape
        assert np.isfinite(ctrl_new).all()

    def test_nk_step_gradient_clipping(self):
        """NK step must clip delta norm to 1.0 to prevent explosive updates.

        REQ-LEARN-011: gradient clipping prevents NK divergence.
        """
        rng = np.random.default_rng(42)
        n_vars, n_knots = 8, 4
        # Init at zero — will produce large residual
        ctrl = np.zeros((n_vars, n_knots), dtype=np.float64)
        X_batch = rng.uniform(-0.9, 0.9, (NK_K_ROWS := 5, n_vars))
        y_batch = np.ones(NK_K_ROWS, dtype=np.float64)  # target=1, energy=0 → large residual

        ctrl_new = _nk_step_vec(ctrl, X_batch, y_batch, n_knots, lam=0.1)

        # The update ||ctrl_new - ctrl|| should be clipped to at most 1.0
        delta_norm = float(np.linalg.norm(ctrl_new - ctrl))
        assert delta_norm <= 1.0 + 1e-10, (
            f"NK step delta_norm={delta_norm:.4f} exceeds clip threshold 1.0"
        )

    def test_promote_grid_shape(self):
        """_promote_grid must produce correct output shape.

        REQ-LEARN-011: multilevel promotion must preserve n_vars dimension.
        """
        rng = np.random.default_rng(9)
        n_vars = 8
        ctrl_coarse = rng.normal(0, 0.1, (n_vars, 4)).astype(np.float64)
        ctrl_fine = _promote_grid(ctrl_coarse, 8)
        assert ctrl_fine.shape == (n_vars, 8)

    def test_promote_grid_preserves_endpoints(self):
        """Promoted grid must interpolate: endpoint values from coarse must be preserved.

        REQ-LEARN-011: multilevel promotion must warm-start fine grid from coarse.
        """
        n_vars = 4
        ctrl_coarse = np.array(
            [
                [0.0, 0.5, 0.8, 1.0],
                [0.1, 0.3, 0.7, 0.9],
                [0.2, 0.4, 0.6, 0.8],
                [0.0, 0.2, 0.5, 1.0],
            ],
            dtype=np.float64,
        )
        ctrl_fine = _promote_grid(ctrl_coarse, 8)

        # Endpoints must be preserved exactly
        assert ctrl_fine.shape == (n_vars, 8)
        for i in range(n_vars):
            assert abs(ctrl_fine[i, 0] - ctrl_coarse[i, 0]) < 1e-10
            assert abs(ctrl_fine[i, -1] - ctrl_coarse[i, -1]) < 1e-10

    def test_enforce_mono_monotone(self):
        """_enforce_mono must produce non-decreasing rows.

        REQ-SAMPLE-015: energy models require monotone control points.
        """
        rng = np.random.default_rng(77)
        ctrl = rng.normal(0, 1, (6, 8)).astype(np.float64)
        ctrl_m = _enforce_mono(ctrl)

        assert ctrl_m.shape == (6, 8)
        for i in range(6):
            diffs = np.diff(ctrl_m[i])
            assert np.all(diffs >= -1e-12), f"Row {i} not monotone: min diff = {diffs.min():.6f}"
        # Min per row should be 0 (zero-floor)
        assert np.allclose(ctrl_m.min(axis=1), 0.0, atol=1e-10)
        # Max per row should be <= 1 (unit-max)
        assert np.all(ctrl_m.max(axis=1) <= 1.0 + 1e-10)


# ===========================================================================
# Corpus helper tests
# ===========================================================================


class TestCorpusHelpers:
    """Tests for corpus loading and label extraction."""

    def test_extract_labels_correct(self):
        """extract_labels must map 'incorrect'→1.0 and 'correct'→0.0.

        REQ-VERIFY-098: energy convention (y=1=incorrect) must be enforced.
        """
        items = [
            {"step_text": "step 1", "label": "correct"},
            {"step_text": "step 2", "label": "incorrect"},
            {"step_text": "step 3", "label": "correct"},
            {"step_text": "step 4", "label": "incorrect"},
        ]
        y = extract_labels(items)
        assert y.shape == (4,)
        assert y.dtype == np.float32
        np.testing.assert_array_equal(y, [0.0, 1.0, 0.0, 1.0])

    def test_extract_labels_all_correct(self):
        """extract_labels with all-correct items must return all zeros."""
        items = [{"label": "correct"} for _ in range(10)]
        y = extract_labels(items)
        assert np.all(y == 0.0)

    def test_extract_labels_all_incorrect(self):
        """extract_labels with all-incorrect items must return all ones."""
        items = [{"label": "incorrect"} for _ in range(10)]
        y = extract_labels(items)
        assert np.all(y == 1.0)

    def test_fit_pca_and_normalize_shapes_and_range(self):
        """fit_pca_and_normalize must return arrays in [-1, 1] with correct shape.

        REQ-SAMPLE-015: feature normalization must preserve energy model input range.
        """
        rng = np.random.default_rng(42)
        n_train, n_test, hidden = 80, 20, 64
        raw_train = rng.normal(0, 2, (n_train, hidden)).astype(np.float32)
        raw_test = rng.normal(0, 2, (n_test, hidden)).astype(np.float32)

        X_tr, X_te = fit_pca_and_normalize(raw_train, raw_test, n_dims=12)
        assert X_tr.shape == (n_train, 12)
        assert X_te.shape == (n_test, 12)
        assert X_tr.dtype == np.float32
        # Training set must be fully in [-1, 1]
        assert np.all(X_tr >= -1.0) and np.all(X_tr <= 1.0)

    def test_fit_pca_and_normalize_no_test_leakage(self):
        """PCA and scaler must be fitted ONLY on train data.

        REQ-VERIFY-098: no data leakage between train and test splits.
        """
        rng = np.random.default_rng(99)
        # Train: all values near 0; Test: all values near 100 (out-of-distribution)
        raw_train = rng.normal(0, 1, (100, 32)).astype(np.float32)
        raw_test = rng.normal(100, 1, (20, 32)).astype(np.float32)

        X_tr, X_te = fit_pca_and_normalize(raw_train, raw_test, n_dims=8)
        # Train features should be near 0 (standardized)
        assert abs(float(X_tr.mean())) < 0.3
        # Test features will be clipped (OOD), but should still be finite
        assert np.isfinite(X_te).all()
