"""Tests for carnot.inference.constraint_models.

Tests cover:
- IsingConstraintModel construction and validation (REQ-VERIFY-002)
- energy() and score() computations on known inputs (SCENARIO-VERIFY-001)
- energy_batch() vectorized computation (REQ-VERIFY-002)
- save_pretrained() / from_pretrained() round-trip (REQ-VERIFY-002)
- Hub loading path via mock (REQ-VERIFY-002)
- ImportError branches for missing safetensors/huggingface_hub (REQ-VERIFY-002)
- ConstraintPropagationModel.from_pretrained() factory API (REQ-VERIFY-002)
- Loading the exported domain models (arithmetic, logic, code) (FR-11)
- Verifying exported models separate correct from wrong features (REQ-VERIFY-003)

Spec: REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-001, FR-11
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from carnot.inference.constraint_models import (
    ConstraintPropagationModel,
    IsingConstraintModel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURE_DIM = 10  # small dim for unit tests


def make_identity_model(dim: int = FEATURE_DIM) -> IsingConstraintModel:
    """Create a deterministic Ising model for unit tests.

    The coupling matrix J = I (identity) and bias b = ones. With these
    parameters:
    - E(ones) = -(d + d) = -2d  (low energy — all spins aligned with bias
      and self-coupling encourages all-positive)
    - E(zeros) = -(−d − d) = +2d (high energy — all spins anti-aligned)

    This gives a maximally discriminative model for unit tests.

    Spec: SCENARIO-VERIFY-001
    """
    J = np.eye(dim, dtype=np.float32)
    b = np.ones(dim, dtype=np.float32)
    config = {
        "domain": "test",
        "feature_dim": dim,
        "carnot_version": "test",
    }
    return IsingConstraintModel(coupling=J, bias=b, config=config)


def make_random_model(dim: int = FEATURE_DIM, seed: int = 0) -> IsingConstraintModel:
    """Create a reproducible random Ising model for property tests.

    Uses Xavier-like initialization with enforced symmetry and zero diagonal —
    the same pattern as the Exp 62/89 training code.

    Spec: REQ-VERIFY-002
    """
    rng = np.random.default_rng(seed)
    limit = float(np.sqrt(6.0 / (dim + dim)))
    J = rng.uniform(-limit, limit, (dim, dim)).astype(np.float32)
    J = (J + J.T) / 2.0          # enforce symmetry
    np.fill_diagonal(J, 0.0)     # zero diagonal (no self-interaction)
    b = rng.uniform(-0.1, 0.1, dim).astype(np.float32)
    config = {
        "domain": "test_random",
        "feature_dim": dim,
        "auroc": 0.75,
    }
    return IsingConstraintModel(coupling=J, bias=b, config=config)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

class TestIsingConstraintModelConstruction:
    """Tests for IsingConstraintModel.__init__.

    Spec: REQ-VERIFY-002, SCENARIO-VERIFY-001
    """

    def test_construction_succeeds_with_valid_inputs(self):
        """Model constructs cleanly from valid numpy arrays.

        Spec: REQ-VERIFY-002
        """
        J = np.eye(5, dtype=np.float32)
        b = np.zeros(5, dtype=np.float32)
        model = IsingConstraintModel(coupling=J, bias=b, config={"domain": "test"})
        assert model.feature_dim == 5
        assert model.domain == "test"

    def test_coupling_must_be_square(self):
        """Non-square coupling raises ValueError.

        Spec: REQ-VERIFY-002
        """
        J = np.ones((4, 5), dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        with pytest.raises(ValueError, match="square"):
            IsingConstraintModel(coupling=J, bias=b, config={})

    def test_coupling_must_be_2d(self):
        """1D coupling raises ValueError.

        Spec: REQ-VERIFY-002
        """
        J = np.ones(5, dtype=np.float32)
        b = np.zeros(5, dtype=np.float32)
        with pytest.raises(ValueError, match="square"):
            IsingConstraintModel(coupling=J, bias=b, config={})

    def test_bias_dim_must_match_coupling(self):
        """Bias length != coupling dim raises ValueError.

        Spec: REQ-VERIFY-002
        """
        J = np.eye(5, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)  # wrong dim
        with pytest.raises(ValueError, match="bias shape"):
            IsingConstraintModel(coupling=J, bias=b, config={})

    def test_arrays_are_cast_to_float32(self):
        """float64 inputs are silently cast to float32.

        Spec: REQ-VERIFY-002
        """
        J = np.eye(3, dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        model = IsingConstraintModel(coupling=J, bias=b, config={})
        assert model.coupling.dtype == np.float32
        assert model.bias.dtype == np.float32

    def test_repr_includes_domain_and_dim(self):
        """repr shows domain and feature_dim.

        Spec: REQ-VERIFY-002
        """
        model = make_identity_model()
        r = repr(model)
        assert "test" in r
        assert str(FEATURE_DIM) in r

    def test_domain_property_default(self):
        """domain property falls back to 'unknown' when not in config.

        Spec: REQ-VERIFY-002
        """
        J = np.eye(3, dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)
        model = IsingConstraintModel(coupling=J, bias=b, config={})
        assert model.domain == "unknown"


# ---------------------------------------------------------------------------
# Energy computation
# ---------------------------------------------------------------------------

class TestEnergy:
    """Tests for IsingConstraintModel.energy().

    The identity model (J=I, b=ones) has exact analytical energies:
    - all-ones input x=1: spins s=+1, E = -(b^T s + s^T J s) = -(d + d) = -2d
    - all-zeros input x=0: spins s=-1, E = -(-d + d) = 0   (I@(-1)=-1, b@(-1)=-d, coupling: (-1)^T I (-1) = d)

    Wait, let me recalculate:
    - x = ones: s = +1 (all +1 spins)
      bias_term = b^T s = 1^T 1 = d
      coupling_term = s^T J s = 1^T I 1 = d (sum of diagonal = d)
      E = -(d + d) = -2d

    - x = zeros: s = -1 (all -1 spins)
      bias_term = b^T s = 1^T (-1) = -d
      coupling_term = s^T J s = (-1)^T I (-1) = d (sum of squared -1's = d)
      E = -(-d + d) = 0

    So all-ones has energy -2d (lower), all-zeros has energy 0 (higher).

    Spec: REQ-CORE-002, SCENARIO-VERIFY-001
    """

    def test_energy_all_ones_is_negative(self):
        """All-ones input produces negative energy (correct pattern).

        Spec: SCENARIO-VERIFY-001
        """
        model = make_identity_model(dim=10)
        x = np.ones(10, dtype=np.float32)
        e = model.energy(x)
        assert e < 0.0, f"Expected negative energy for all-ones, got {e}"

    def test_energy_all_zeros_is_zero(self):
        """All-zeros input produces zero energy with J=I, b=ones.

        With J=I and b=ones:
        - s = -1 (all spins = -1)
        - bias_term = 1^T (-1) = -d
        - coupling_term = (-1)^T I (-1) = d
        - E = -(-d + d) = 0

        Spec: SCENARIO-VERIFY-001
        """
        model = make_identity_model(dim=10)
        x = np.zeros(10, dtype=np.float32)
        e = model.energy(x)
        assert abs(e - 0.0) < 1e-5, f"Expected ~0 energy for all-zeros, got {e}"

    def test_energy_ones_lower_than_zeros(self):
        """All-ones has lower energy than all-zeros with identity model.

        Spec: SCENARIO-VERIFY-001
        """
        model = make_identity_model(dim=10)
        e_ones = model.energy(np.ones(10, dtype=np.float32))
        e_zeros = model.energy(np.zeros(10, dtype=np.float32))
        assert e_ones < e_zeros, (
            f"Expected E(ones)={e_ones:.4f} < E(zeros)={e_zeros:.4f}"
        )

    def test_energy_exact_value_all_ones(self):
        """Energy for all-ones matches analytical formula -2d.

        Spec: SCENARIO-VERIFY-001
        """
        dim = 10
        model = make_identity_model(dim=dim)
        x = np.ones(dim, dtype=np.float32)
        e = model.energy(x)
        expected = -2.0 * dim  # -2d with J=I, b=ones
        assert abs(e - expected) < 1e-4, f"Expected {expected}, got {e}"

    def test_energy_returns_scalar(self):
        """energy() returns a Python float, not an array.

        Spec: REQ-VERIFY-002
        """
        model = make_identity_model()
        x = np.zeros(FEATURE_DIM, dtype=np.float32)
        result = model.energy(x)
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_energy_accepts_integer_input(self):
        """energy() works with integer {0,1} arrays (common in tests).

        Spec: REQ-VERIFY-002
        """
        model = make_identity_model()
        x = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
        e = model.energy(x)
        assert isinstance(e, float)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

class TestScore:
    """Tests for IsingConstraintModel.score().

    Spec: REQ-VERIFY-003
    """

    def test_score_in_open_unit_interval(self):
        """score() output is strictly in (0, 1).

        Spec: REQ-VERIFY-003
        """
        model = make_random_model()
        for _ in range(20):
            x = (np.random.default_rng(42).random(FEATURE_DIM) > 0.5).astype(np.float32)
            s = model.score(x)
            assert 0.0 < s < 1.0, f"score={s} out of (0,1)"

    def test_score_correct_higher_than_wrong(self):
        """For identity model, all-ones scores higher than all-zeros.

        Spec: REQ-VERIFY-003
        """
        model = make_identity_model()
        s_ones = model.score(np.ones(FEATURE_DIM, dtype=np.float32))
        s_zeros = model.score(np.zeros(FEATURE_DIM, dtype=np.float32))
        assert s_ones > s_zeros, (
            f"Expected score(ones)={s_ones:.4f} > score(zeros)={s_zeros:.4f}"
        )

    def test_score_monotone_with_energy(self):
        """score = sigmoid(-energy) — higher energy → lower score.

        Spec: REQ-VERIFY-003
        """
        model = make_random_model(seed=7)
        rng = np.random.default_rng(7)
        energies = []
        scores = []
        for _ in range(10):
            x = (rng.random(FEATURE_DIM) > 0.5).astype(np.float32)
            energies.append(model.energy(x))
            scores.append(model.score(x))
        # Check monotone inverse relationship: higher energy → lower score
        for i in range(len(energies)):
            for j in range(len(energies)):
                if energies[i] < energies[j] - 1e-6:
                    assert scores[i] > scores[j] - 1e-6, (
                        f"E[i]={energies[i]:.4f} < E[j]={energies[j]:.4f} "
                        f"but score[i]={scores[i]:.4f} <= score[j]={scores[j]:.4f}"
                    )

    def test_score_near_half_for_zero_energy(self):
        """score ≈ 0.5 when energy ≈ 0.

        Spec: REQ-VERIFY-003
        """
        # Build a model with J=0, b=0 → energy always 0 → score always 0.5
        J = np.zeros((FEATURE_DIM, FEATURE_DIM), dtype=np.float32)
        b = np.zeros(FEATURE_DIM, dtype=np.float32)
        model = IsingConstraintModel(coupling=J, bias=b, config={})
        x = np.ones(FEATURE_DIM, dtype=np.float32)
        s = model.score(x)
        assert abs(s - 0.5) < 1e-5, f"Expected score≈0.5 for zero model, got {s}"

    def test_score_clamps_extreme_energies(self):
        """Very large energies are clamped to prevent overflow.

        Spec: REQ-VERIFY-003
        """
        # Build a model with very large bias to produce extreme energies
        dim = 5
        J = np.zeros((dim, dim), dtype=np.float32)
        b = np.full(dim, 1e10, dtype=np.float32)
        model = IsingConstraintModel(coupling=J, bias=b, config={})
        # x=zeros: s=-1, E = -(b^T (-1)) = +5e10 → score near 0
        x_zeros = np.zeros(dim, dtype=np.float32)
        s = model.score(x_zeros)
        assert 0.0 <= s <= 1.0, f"score={s} overflowed"


# ---------------------------------------------------------------------------
# Batch energy
# ---------------------------------------------------------------------------

class TestEnergyBatch:
    """Tests for IsingConstraintModel.energy_batch().

    Spec: REQ-VERIFY-002
    """

    def test_batch_matches_individual(self):
        """energy_batch results match element-wise energy() calls.

        Spec: REQ-VERIFY-002
        """
        model = make_random_model(seed=1)
        rng = np.random.default_rng(1)
        X = (rng.random((8, FEATURE_DIM)) > 0.5).astype(np.float32)
        batch = model.energy_batch(X)
        individual = np.array([model.energy(X[i]) for i in range(8)])
        np.testing.assert_allclose(batch, individual, atol=1e-4)

    def test_batch_output_shape(self):
        """energy_batch returns (n,) shaped array.

        Spec: REQ-VERIFY-002
        """
        model = make_random_model()
        X = np.zeros((5, FEATURE_DIM), dtype=np.float32)
        result = model.energy_batch(X)
        assert result.shape == (5,), f"Expected shape (5,), got {result.shape}"

    def test_batch_correctly_ranks_candidates(self):
        """For identity model, batch correctly ranks all-ones above all-zeros.

        Spec: REQ-VERIFY-002
        """
        model = make_identity_model(dim=10)
        X = np.stack([
            np.ones(10, dtype=np.float32),   # correct
            np.zeros(10, dtype=np.float32),  # wrong
        ])
        energies = model.energy_batch(X)
        # Correct (index 0) should have lower energy than wrong (index 1)
        assert energies[0] < energies[1], (
            f"Expected E(correct)={energies[0]:.4f} < E(wrong)={energies[1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    """Tests for save_pretrained() / from_pretrained() round-trip.

    Spec: REQ-VERIFY-002
    """

    def test_save_creates_expected_files(self, tmp_path):
        """save_pretrained creates model.safetensors and config.json.

        Spec: REQ-VERIFY-002
        """
        model = make_random_model()
        model.save_pretrained(str(tmp_path / "model"))
        assert (tmp_path / "model" / "model.safetensors").exists()
        assert (tmp_path / "model" / "config.json").exists()

    def test_load_restores_coupling(self, tmp_path):
        """Loaded model has coupling matrix identical to saved model.

        Spec: REQ-VERIFY-002
        """
        model = make_random_model(seed=5)
        path = str(tmp_path / "ckpt")
        model.save_pretrained(path)
        loaded = IsingConstraintModel.from_pretrained(path)
        np.testing.assert_array_equal(model.coupling, loaded.coupling)

    def test_load_restores_bias(self, tmp_path):
        """Loaded model has bias vector identical to saved model.

        Spec: REQ-VERIFY-002
        """
        model = make_random_model(seed=6)
        path = str(tmp_path / "ckpt")
        model.save_pretrained(path)
        loaded = IsingConstraintModel.from_pretrained(path)
        np.testing.assert_array_equal(model.bias, loaded.bias)

    def test_load_restores_config(self, tmp_path):
        """Loaded model config matches saved config.

        Spec: REQ-VERIFY-002
        """
        model = make_random_model(seed=7)
        model.config["auroc"] = 0.987
        path = str(tmp_path / "ckpt")
        model.save_pretrained(path)
        loaded = IsingConstraintModel.from_pretrained(path)
        assert loaded.config["auroc"] == 0.987
        assert loaded.config["domain"] == "test_random"

    def test_load_produces_same_energies(self, tmp_path):
        """Round-trip produces identical energy computations.

        Spec: REQ-VERIFY-002, SCENARIO-VERIFY-001
        """
        model = make_random_model(seed=3)
        path = str(tmp_path / "ckpt")
        model.save_pretrained(path)
        loaded = IsingConstraintModel.from_pretrained(path)

        rng = np.random.default_rng(3)
        X = (rng.random((5, FEATURE_DIM)) > 0.5).astype(np.float32)
        for i in range(5):
            e_orig = model.energy(X[i])
            e_load = loaded.energy(X[i])
            assert abs(e_orig - e_load) < 1e-5, (
                f"Energy mismatch after round-trip: {e_orig} vs {e_load}"
            )

    def test_load_local_path_with_slash(self, tmp_path):
        """from_pretrained recognizes paths starting with '/'.

        Spec: REQ-VERIFY-002
        """
        model = make_identity_model()
        path = str(tmp_path / "explicit_path")
        model.save_pretrained(path)
        # Absolute path starts with '/'
        loaded = IsingConstraintModel.from_pretrained(path)
        assert loaded.feature_dim == FEATURE_DIM

    def test_load_missing_config_raises(self, tmp_path):
        """from_pretrained raises FileNotFoundError for missing config.json.

        Spec: REQ-VERIFY-002
        """
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="config.json"):
            IsingConstraintModel.from_pretrained(str(empty_dir))

    def test_load_missing_weights_raises(self, tmp_path):
        """from_pretrained raises FileNotFoundError for missing model.safetensors.

        Spec: REQ-VERIFY-002
        """
        d = tmp_path / "no_weights"
        d.mkdir()
        # Write config but no safetensors
        with open(d / "config.json", "w") as f:
            json.dump({"domain": "test"}, f)
        with pytest.raises(FileNotFoundError, match="model.safetensors"):
            IsingConstraintModel.from_pretrained(str(d))

    def test_save_creates_parent_dirs(self, tmp_path):
        """save_pretrained creates nested parent directories if absent.

        Spec: REQ-VERIFY-002
        """
        model = make_identity_model()
        nested = tmp_path / "a" / "b" / "c"
        model.save_pretrained(str(nested))
        assert (nested / "model.safetensors").exists()

    def test_hub_load_path_triggers_for_non_local_repo_id(self, tmp_path):
        """from_pretrained routes 'Org/repo' style IDs to Hub loader.

        Uses a mock hf_hub_download that returns local file paths so no
        network access is needed.

        Spec: REQ-VERIFY-002
        """
        # Save a real model to tmp_path so the Hub mock can return its files.
        model = make_identity_model()
        model.save_pretrained(str(tmp_path))

        config_path = str(tmp_path / "config.json")
        weights_path = str(tmp_path / "model.safetensors")

        # Mock huggingface_hub.hf_hub_download to return local paths.
        mock_hf = types.ModuleType("huggingface_hub")
        call_count = {"n": 0}

        def fake_download(repo_id, filename):
            call_count["n"] += 1
            if filename == "config.json":
                return config_path
            return weights_path

        mock_hf.hf_hub_download = fake_download

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            # "Carnot-EBM/constraint-propagation-arithmetic" is not a local path
            loaded = IsingConstraintModel.from_pretrained(
                "Carnot-EBM/constraint-propagation-arithmetic"
            )

        assert isinstance(loaded, IsingConstraintModel)
        assert call_count["n"] == 2, "Expected 2 hf_hub_download calls (config + weights)"
        np.testing.assert_array_equal(model.coupling, loaded.coupling)

    def test_hub_load_raises_import_error_without_huggingface_hub(self, tmp_path):
        """_load_hub raises ImportError when huggingface_hub is not installed.

        Spec: REQ-VERIFY-002
        """
        # Remove huggingface_hub from sys.modules to simulate it being absent.
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub"):
                # A non-local repo ID (no '.' or '/' prefix, doesn't exist)
                IsingConstraintModel.from_pretrained("Org/missing-repo")

    def test_save_raises_import_error_without_safetensors(self, tmp_path):
        """save_pretrained raises ImportError when safetensors is not installed.

        Spec: REQ-VERIFY-002
        """
        model = make_identity_model()

        # Temporarily hide safetensors by making the import fail.
        orig = sys.modules.get("safetensors.numpy")
        try:
            sys.modules["safetensors.numpy"] = None  # type: ignore[assignment]
            with pytest.raises(ImportError, match="safetensors"):
                model.save_pretrained(str(tmp_path / "out"))
        finally:
            if orig is None:
                del sys.modules["safetensors.numpy"]
            else:
                sys.modules["safetensors.numpy"] = orig

    def test_load_raises_import_error_without_safetensors(self, tmp_path):
        """from_pretrained raises ImportError when safetensors is missing.

        Spec: REQ-VERIFY-002
        """
        # Create the files manually so from_pretrained reaches the load_file call.
        d = tmp_path / "model_dir"
        d.mkdir()
        with open(d / "config.json", "w") as f:
            json.dump({"domain": "test"}, f)
        # Create a dummy safetensors file (will be intercepted by mocked import).
        (d / "model.safetensors").write_bytes(b"")

        orig = sys.modules.get("safetensors.numpy")
        try:
            sys.modules["safetensors.numpy"] = None  # type: ignore[assignment]
            with pytest.raises(ImportError, match="safetensors"):
                IsingConstraintModel.from_pretrained(str(d))
        finally:
            if orig is None:
                sys.modules.pop("safetensors.numpy", None)
            else:
                sys.modules["safetensors.numpy"] = orig


# ---------------------------------------------------------------------------
# ConstraintPropagationModel factory
# ---------------------------------------------------------------------------

class TestConstraintPropagationModel:
    """Tests for ConstraintPropagationModel.from_pretrained().

    Spec: REQ-VERIFY-002
    """

    def test_from_pretrained_returns_ising_model(self, tmp_path):
        """ConstraintPropagationModel.from_pretrained returns IsingConstraintModel.

        Spec: REQ-VERIFY-002
        """
        orig = make_identity_model()
        path = str(tmp_path / "m")
        orig.save_pretrained(path)
        loaded = ConstraintPropagationModel.from_pretrained(path)
        assert isinstance(loaded, IsingConstraintModel)

    def test_factory_preserves_weights(self, tmp_path):
        """ConstraintPropagationModel.from_pretrained loads correct weights.

        Spec: REQ-VERIFY-002
        """
        orig = make_random_model(seed=99)
        path = str(tmp_path / "m")
        orig.save_pretrained(path)
        loaded = ConstraintPropagationModel.from_pretrained(path)
        np.testing.assert_array_equal(orig.coupling, loaded.coupling)
        np.testing.assert_array_equal(orig.bias, loaded.bias)


# ---------------------------------------------------------------------------
# Exported domain models
# ---------------------------------------------------------------------------

EXPORTS_ROOT = (
    Path(__file__).parent.parent.parent
    / "exports"
    / "constraint-propagation-models"
)


def _domain_model_available(domain: str) -> bool:
    """Check whether the exported model files exist on disk."""
    p = EXPORTS_ROOT / domain
    return (p / "model.safetensors").exists() and (p / "config.json").exists()


@pytest.mark.skipif(
    not _domain_model_available("arithmetic"),
    reason="arithmetic export not found — run scripts/export_constraint_models.py",
)
class TestArithmeticModel:
    """Tests for the exported arithmetic constraint model.

    Spec: REQ-VERIFY-002, REQ-VERIFY-003, FR-11
    """

    @pytest.fixture(scope="class")
    def model(self):
        return ConstraintPropagationModel.from_pretrained(
            str(EXPORTS_ROOT / "arithmetic")
        )

    def test_loads_successfully(self, model):
        """Arithmetic model loads without error.

        Spec: REQ-VERIFY-002
        """
        assert isinstance(model, IsingConstraintModel)

    def test_domain_is_arithmetic(self, model):
        """Arithmetic model config has domain='arithmetic'.

        Spec: REQ-VERIFY-002
        """
        assert model.domain == "arithmetic"

    def test_feature_dim_is_200(self, model):
        """Arithmetic model has 200 input features.

        Spec: REQ-VERIFY-002
        """
        assert model.feature_dim == 200

    def test_coupling_is_symmetric(self, model):
        """Coupling matrix is symmetric (required by Ising energy formula).

        Spec: REQ-VERIFY-002
        """
        diff = np.abs(model.coupling - model.coupling.T).max()
        assert diff < 1e-5, f"Coupling is not symmetric: max|J - J^T| = {diff}"

    def test_coupling_diagonal_is_zero(self, model):
        """Coupling matrix has zero diagonal (no self-interaction).

        Spec: REQ-VERIFY-002
        """
        diag_max = np.abs(np.diag(model.coupling)).max()
        assert diag_max < 1e-5, f"Diagonal not zero: max|diag(J)| = {diag_max}"

    def test_energy_on_correct_answer(self, model):
        """Correct arithmetic answer has lower energy than wrong answer.

        This is the core verification test: the model must rank correct above
        wrong for a simple addition question.

        Spec: REQ-VERIFY-003, SCENARIO-VERIFY-001
        """
        # Import the encoder from the export script.
        import sys, os
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from export_constraint_models import encode_answer

        x_correct = encode_answer("What is 47 + 28?", "The answer is 75.")
        x_wrong   = encode_answer("What is 47 + 28?", "The answer is 74.")

        e_correct = model.energy(x_correct)
        e_wrong   = model.energy(x_wrong)

        # Lower energy = more likely correct.
        assert e_correct < e_wrong, (
            f"Arithmetic model should assign lower energy to correct answer.\n"
            f"E(correct)={e_correct:.4f}, E(wrong)={e_wrong:.4f}"
        )

    def test_auroc_in_config(self, model):
        """AUROC benchmark result is present in config.

        Spec: FR-11
        """
        auroc = model.config.get("benchmark", {}).get("auroc_reproduced")
        assert auroc is not None, "auroc_reproduced not found in config.benchmark"
        assert 0.5 < auroc <= 1.0, f"AUROC={auroc} out of expected range (0.5, 1.0]"

    def test_batch_energy_over_multiple_pairs(self, model):
        """Batch energy ranks 3 correct answers above 3 wrong answers.

        Spec: REQ-VERIFY-002
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from export_constraint_models import encode_answer, generate_arithmetic_pairs

        rng = np.random.default_rng(0)
        pairs = generate_arithmetic_pairs(20, rng)
        correct_vecs = np.array(
            [encode_answer(q, c) for (q, c, w) in pairs], dtype=np.float32
        )
        wrong_vecs = np.array(
            [encode_answer(q, w) for (q, c, w) in pairs], dtype=np.float32
        )

        e_correct = model.energy_batch(correct_vecs)
        e_wrong   = model.energy_batch(wrong_vecs)

        # Most correct should have lower energy than wrong
        pct_correct = float(np.mean(e_correct < e_wrong))
        assert pct_correct >= 0.7, (
            f"Expected ≥70% of pairs to have E(correct) < E(wrong), "
            f"got {pct_correct:.1%}"
        )


@pytest.mark.skipif(
    not _domain_model_available("logic"),
    reason="logic export not found — run scripts/export_constraint_models.py",
)
class TestLogicModel:
    """Tests for the exported logic constraint model.

    Spec: REQ-VERIFY-002, REQ-VERIFY-003, FR-11
    """

    @pytest.fixture(scope="class")
    def model(self):
        return ConstraintPropagationModel.from_pretrained(
            str(EXPORTS_ROOT / "logic")
        )

    def test_loads_successfully(self, model):
        """Logic model loads without error.

        Spec: REQ-VERIFY-002
        """
        assert isinstance(model, IsingConstraintModel)

    def test_domain_is_logic(self, model):
        """Logic model config has domain='logic'.

        Spec: REQ-VERIFY-002
        """
        assert model.domain == "logic"

    def test_separates_valid_from_invalid_syllogism(self, model):
        """Valid modus ponens answer scores higher than invalid affirming-consequent.

        Spec: REQ-VERIFY-003, SCENARIO-VERIFY-001
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from export_constraint_models import encode_answer

        q = "If all cats are mortal, and all robots are cats, what follows?"
        correct = "All robots are mortal. This follows by modus ponens."
        wrong   = "Some robots are not mortal. The premises do not guarantee this."

        x_correct = encode_answer(q, correct)
        x_wrong   = encode_answer(q, wrong)

        e_correct = model.energy(x_correct)
        e_wrong   = model.energy(x_wrong)

        assert e_correct < e_wrong, (
            f"Logic model should rank valid syllogism lower energy.\n"
            f"E(correct)={e_correct:.4f}, E(wrong)={e_wrong:.4f}"
        )

    def test_auroc_at_least_0_9(self, model):
        """Logic model AUROC should be ≥ 0.9 (Exp 89 achieved 1.0).

        Spec: FR-11
        """
        auroc = model.config.get("benchmark", {}).get("auroc_reproduced", 0.0)
        assert auroc >= 0.9, f"Expected AUROC ≥ 0.9, got {auroc}"


@pytest.mark.skipif(
    not _domain_model_available("code"),
    reason="code export not found — run scripts/export_constraint_models.py",
)
class TestCodeModel:
    """Tests for the exported code constraint model.

    Spec: REQ-VERIFY-002, REQ-VERIFY-003, FR-11
    """

    @pytest.fixture(scope="class")
    def model(self):
        return ConstraintPropagationModel.from_pretrained(
            str(EXPORTS_ROOT / "code")
        )

    def test_loads_successfully(self, model):
        """Code model loads without error.

        Spec: REQ-VERIFY-002
        """
        assert isinstance(model, IsingConstraintModel)

    def test_domain_is_code(self, model):
        """Code model config has domain='code'.

        Spec: REQ-VERIFY-002
        """
        assert model.domain == "code"

    def test_separates_correct_from_buggy_code(self, model):
        """Correct sum_range implementation ranks below off-by-one bug.

        Spec: REQ-VERIFY-003, SCENARIO-VERIFY-001
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from export_constraint_models import encode_answer

        question = "Write a function that returns the sum of integers from 1 to n."
        correct_code = (
            "def sum_range(n):\n"
            "    total = 0\n"
            "    for i in range(1, n + 1):\n"
            "        total += i\n"
            "    return total"
        )
        buggy_code = (
            "def sum_range(n):\n"
            "    total = 0\n"
            "    for i in range(1, n):\n"  # off-by-one
            "        total += i\n"
            "    return total"
        )

        x_correct = encode_answer(question, correct_code)
        x_buggy   = encode_answer(question, buggy_code)

        e_correct = model.energy(x_correct)
        e_buggy   = model.energy(x_buggy)

        assert e_correct < e_buggy, (
            f"Code model should rank correct implementation lower energy.\n"
            f"E(correct)={e_correct:.4f}, E(buggy)={e_buggy:.4f}"
        )

    def test_auroc_at_least_0_7(self, model):
        """Code model AUROC ≥ 0.7 (Exp 89 reference: 0.91).

        Spec: FR-11
        """
        auroc = model.config.get("benchmark", {}).get("auroc_reproduced", 0.0)
        assert auroc >= 0.7, f"Expected AUROC ≥ 0.7, got {auroc}"
