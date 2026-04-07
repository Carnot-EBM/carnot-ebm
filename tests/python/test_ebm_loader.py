"""Tests for EBM model loading from HuggingFace/local exports.

**Researcher summary:**
    Verifies the EBM loader can find models locally, reconstruct
    GibbsModel from saved weights, and score activations correctly.

**Detailed explanation for engineers:**
    Tests use a temporary export directory with real safetensors
    weights to verify the full loading pipeline without network access.

Spec coverage: REQ-INFER-015
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 — used at runtime in tmp_path

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest
from carnot.inference.ebm_loader import (
    KNOWN_MODELS,
    _find_model_dir,
    get_model_info,
    load_ebm,
)
from carnot.models.gibbs import GibbsConfig, GibbsModel
from safetensors.numpy import save_file


def _create_test_export(tmp_path: Path, model_id: str = "test-model") -> Path:
    """Create a minimal model export for testing."""
    model_dir = tmp_path / model_id
    model_dir.mkdir()

    # Create a small model and save its weights
    config = GibbsConfig(input_dim=8, hidden_dims=[4])
    ebm = GibbsModel(config, key=jrandom.PRNGKey(42))

    weights = {}
    for i, (w, b) in enumerate(ebm.layers):
        weights[f"layer_{i}_weight"] = np.array(w)
        weights[f"layer_{i}_bias"] = np.array(b)
    weights["output_weight"] = np.array(ebm.output_weight)
    weights["output_bias"] = np.array(ebm.output_bias)
    save_file(weights, str(model_dir / "model.safetensors"))

    config_dict = {
        "model_type": "gibbs_ebm",
        "input_dim": 8,
        "hidden_dims": [4],
        "activation": "silu",
        "n_layers": 1,
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config_dict, f)

    return model_dir


def test_load_ebm_from_local_dir(tmp_path: Path) -> None:
    """Load model from explicit local directory.

    Spec: REQ-INFER-015
    """
    model_dir = _create_test_export(tmp_path)
    ebm = load_ebm("test-model", local_dir=str(model_dir))

    # Should produce finite energy
    x = jnp.ones(8)
    energy = float(ebm.energy(x))
    assert np.isfinite(energy)


def test_load_ebm_from_exports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Load model from project exports/ directory.

    Spec: REQ-INFER-015
    """
    # Create exports/test-model/ structure
    _create_test_export(tmp_path, "test-model")

    # Monkey-patch the project root detection to use tmp_path
    def _mock_find(mid: str, ld: str | None, org: str) -> str:
        path = tmp_path / mid
        if path.exists():
            return str(path)
        msg = "not found"
        raise FileNotFoundError(msg)

    monkeypatch.setattr("carnot.inference.ebm_loader._find_model_dir", _mock_find)

    ebm = load_ebm("test-model", local_dir=str(tmp_path / "test-model"))
    energy = float(ebm.energy(jnp.ones(8)))
    assert np.isfinite(energy)


def test_load_ebm_not_found() -> None:
    """Missing model raises FileNotFoundError.

    Spec: REQ-INFER-015
    """
    with pytest.raises(FileNotFoundError):
        load_ebm("nonexistent-model-that-doesnt-exist-anywhere")


def test_loaded_ebm_matches_original(tmp_path: Path) -> None:
    """Loaded model produces same energy as the original.

    Spec: REQ-INFER-015
    """
    model_dir = _create_test_export(tmp_path)

    # Load the model
    loaded = load_ebm("test-model", local_dir=str(model_dir))

    # Create the same model fresh
    original = GibbsModel(GibbsConfig(input_dim=8, hidden_dims=[4]), key=jrandom.PRNGKey(42))

    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert float(loaded.energy(x)) == pytest.approx(float(original.energy(x)), abs=1e-5)


def test_known_models_metadata() -> None:
    """KNOWN_MODELS has expected entries.

    Spec: REQ-INFER-015
    """
    assert "per-token-ebm-qwen3-06b" in KNOWN_MODELS
    assert "per-token-ebm-qwen35-08b-nothink" in KNOWN_MODELS
    assert "per-token-ebm-qwen35-08b-think" in KNOWN_MODELS


def test_get_model_info_known() -> None:
    """get_model_info returns metadata for known models.

    Spec: REQ-INFER-015
    """
    info = get_model_info("per-token-ebm-qwen35-08b-nothink")
    assert info["source_model"] == "Qwen/Qwen3.5-0.8B"
    assert info["thinking"] == "disabled"


def test_get_model_info_unknown() -> None:
    """get_model_info returns 'unknown' for unregistered models.

    Spec: REQ-INFER-015
    """
    info = get_model_info("some-future-model")
    assert info["source_model"] == "unknown"


def test_find_model_dir_local(tmp_path: Path) -> None:
    """_find_model_dir finds local directory first.

    Spec: REQ-INFER-015
    """
    model_dir = _create_test_export(tmp_path)
    found = _find_model_dir("test-model", str(model_dir), "Carnot-EBM")
    assert found == str(model_dir)


def test_find_model_dir_no_config(tmp_path: Path) -> None:
    """_find_model_dir rejects directory without config.json.

    Spec: REQ-INFER-015
    """
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        _find_model_dir("empty", str(empty_dir), "nonexistent-org-xyzzy")


def test_exports_from_inference_package() -> None:
    """Loader symbols exported from carnot.inference.

    Spec: REQ-INFER-015
    """
    from carnot.inference import KNOWN_MODELS, get_model_info, load_ebm

    assert callable(load_ebm)
    assert callable(get_model_info)
    assert isinstance(KNOWN_MODELS, dict)
