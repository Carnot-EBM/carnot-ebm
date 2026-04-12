"""Tests for publish_v12_models.py — KAN artifact serialization and model card validation.

Verifies that the KAN constraint verifier artifact can be saved, loaded, and
that all model card files contain required sections.

Spec coverage: REQ-CORE-001, REQ-CORE-003, REQ-CORE-004
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from carnot.models.kan import BSplineParams, KANConfig, KANEnergyFunction

# Add scripts directory to path so we can import helpers from publish script
_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from publish_v12_models import (  # noqa: E402
    KAN_CONFIG,
    OUTPUT_DIR,
    build_kan_model,
    extract_kan_parameters,
    write_config,
)

# Path to the model card directory checked into the repo
_MODEL_CARD_DIR = Path(__file__).parent.parent.parent / "models" / "constraint-verifier-v2"

# Required sections in model cards (checked as substrings)
_README_REQUIRED_SECTIONS = [
    "Research Prototype",
    "KAN",
    "Ising",
    "Gibbs",
    "AUROC",
    "Limitations",
    "Installation",
    "config.json",
    "model.safetensors",
    "REQ-CORE-001" if False else "Spec",  # README uses plain English, not spec IDs
    "Phase",
    "Disclaimer",
]

_GUIDED_README_REQUIRED_SECTIONS = [
    "Research Prototype",
    "alpha",
    "check_every_k",
    "Quick Start",
    "Limitations",
    "Performance",
]

_CONFIG_REQUIRED_KEYS = [
    "model_type",
    "version",
    "input_dim",
    "num_knots",
    "degree",
    "sparse",
    "edge_density",
    "disclaimer",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_kan_config() -> KANConfig:
    """REQ-CORE-001: Minimal KAN config for fast test execution."""
    return KANConfig(
        input_dim=5,
        num_knots=5,
        degree=2,
        sparse=True,
        edge_density=1.0,
    )


@pytest.fixture(scope="module")
def small_kan_model(small_kan_config: KANConfig) -> KANEnergyFunction:
    """REQ-CORE-001: Small KAN model for serialization tests."""
    return KANEnergyFunction(small_kan_config, key=jrandom.PRNGKey(0))


# ---------------------------------------------------------------------------
# Safetensors roundtrip tests
# ---------------------------------------------------------------------------


class TestKANSafetensorsRoundtrip:
    """REQ-CORE-003, REQ-CORE-004: KAN weight serialization via safetensors."""

    def test_kan_params_extract_not_empty(self, small_kan_model: KANEnergyFunction) -> None:
        """REQ-CORE-001: Parameter extraction yields at least one array."""
        params = extract_kan_parameters(small_kan_model)
        assert len(params) > 0, "Expected at least one parameter array"

    def test_kan_params_extract_names(self, small_kan_model: KANEnergyFunction) -> None:
        """REQ-CORE-001: Extracted parameters have expected naming conventions."""
        params = extract_kan_parameters(small_kan_model)
        edge_keys = [k for k in params if k.startswith("edge_") and k.endswith("_cp")]
        bias_keys = [k for k in params if k.startswith("bias_") and k.endswith("_cp")]
        assert len(edge_keys) == len(small_kan_model.edges), (
            f"Expected {len(small_kan_model.edges)} edge keys, got {len(edge_keys)}"
        )
        assert len(bias_keys) == small_kan_model.input_dim, (
            f"Expected {small_kan_model.input_dim} bias keys, got {len(bias_keys)}"
        )

    def test_kan_params_are_numpy(self, small_kan_model: KANEnergyFunction) -> None:
        """REQ-CORE-004: Extracted params are numpy arrays (safetensors requirement)."""
        params = extract_kan_parameters(small_kan_model)
        for key, arr in params.items():
            assert isinstance(arr, np.ndarray), (
                f"Key {key!r}: expected np.ndarray, got {type(arr)}"
            )

    def test_kan_params_finite(self, small_kan_model: KANEnergyFunction) -> None:
        """REQ-CORE-001: All extracted parameters are finite (no NaN/Inf)."""
        params = extract_kan_parameters(small_kan_model)
        for key, arr in params.items():
            assert np.all(np.isfinite(arr)), f"Key {key!r}: contains non-finite values"

    def test_safetensors_save_load_roundtrip(self, small_kan_model: KANEnergyFunction) -> None:
        """REQ-CORE-004: Weights survive save -> load roundtrip with no data loss."""
        params = extract_kan_parameters(small_kan_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            safetensors_path = Path(tmpdir) / "model.safetensors"
            save_file(params, str(safetensors_path))

            loaded = load_file(str(safetensors_path))

        assert set(loaded.keys()) == set(params.keys()), "Key sets must match"

        for key in params:
            original = params[key].astype(np.float32)
            reloaded = loaded[key].astype(np.float32)
            max_diff = float(np.max(np.abs(original - reloaded)))
            assert max_diff < 1e-6, (
                f"Key {key!r}: max roundtrip diff {max_diff:.2e} exceeds 1e-6 tolerance"
            )

    def test_safetensors_shapes_preserved(self, small_kan_model: KANEnergyFunction) -> None:
        """REQ-CORE-004: Tensor shapes are preserved through safetensors roundtrip."""
        params = extract_kan_parameters(small_kan_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            safetensors_path = Path(tmpdir) / "model.safetensors"
            save_file(params, str(safetensors_path))
            loaded = load_file(str(safetensors_path))

        for key in params:
            assert loaded[key].shape == params[key].shape, (
                f"Key {key!r}: shape {params[key].shape} -> {loaded[key].shape}"
            )

    def test_energy_reproducible_after_reload(self, small_kan_config: KANConfig) -> None:
        """REQ-CORE-001, REQ-CORE-004: Energy is identical before and after weight reload."""
        model_a = KANEnergyFunction(small_kan_config, key=jrandom.PRNGKey(99))
        params = extract_kan_parameters(model_a)

        x_test = jnp.array([0.3, -0.1, 0.5, 0.0, -0.7])
        energy_before = float(model_a.energy(x_test))

        with tempfile.TemporaryDirectory() as tmpdir:
            safetensors_path = Path(tmpdir) / "model.safetensors"
            save_file(params, str(safetensors_path))
            loaded_params = load_file(str(safetensors_path))

        # Reconstruct model_b with same config, then restore weights
        model_b = KANEnergyFunction(small_kan_config, key=jrandom.PRNGKey(99))
        for idx, (edge, spline) in enumerate(model_b.edge_splines.items()):
            cp_key = f"edge_{idx}_cp"
            spline.params = BSplineParams(
                control_points=jnp.array(loaded_params[cp_key])
            )
        for idx, spline in enumerate(model_b.bias_splines):
            cp_key = f"bias_{idx}_cp"
            spline.params = BSplineParams(
                control_points=jnp.array(loaded_params[cp_key])
            )

        energy_after = float(model_b.energy(x_test))
        assert abs(energy_before - energy_after) < 1e-5, (
            f"Energy changed after reload: {energy_before:.6f} -> {energy_after:.6f}"
        )


# ---------------------------------------------------------------------------
# Config JSON schema tests
# ---------------------------------------------------------------------------


class TestConfigJsonSchema:
    """REQ-CORE-001, REQ-CORE-003: Config JSON schema validity."""

    def test_config_has_required_keys(self) -> None:
        """REQ-CORE-003: Config JSON contains all required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            write_config(KAN_CONFIG, Path(tmpdir), n_params=100, n_edges=10)
            with open(Path(tmpdir) / "config.json") as f:
                cfg = json.load(f)

        for key in _CONFIG_REQUIRED_KEYS:
            assert key in cfg, f"Config missing required key: {key!r}"

    def test_config_input_dim_matches(self) -> None:
        """REQ-CORE-001: Config input_dim matches KAN_CONFIG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            write_config(KAN_CONFIG, Path(tmpdir), n_params=100, n_edges=10)
            with open(Path(tmpdir) / "config.json") as f:
                cfg = json.load(f)

        assert cfg["input_dim"] == KAN_CONFIG.input_dim

    def test_config_num_knots_matches(self) -> None:
        """REQ-CORE-001: Config num_knots matches KAN_CONFIG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            write_config(KAN_CONFIG, Path(tmpdir), n_params=100, n_edges=10)
            with open(Path(tmpdir) / "config.json") as f:
                cfg = json.load(f)

        assert cfg["num_knots"] == KAN_CONFIG.num_knots

    def test_config_has_disclaimer(self) -> None:
        """REQ-CORE-001: Config JSON contains research prototype disclaimer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            write_config(KAN_CONFIG, Path(tmpdir), n_params=100, n_edges=10)
            with open(Path(tmpdir) / "config.json") as f:
                cfg = json.load(f)

        assert "disclaimer" in cfg
        disclaimer = cfg["disclaimer"].lower()
        assert "research" in disclaimer or "prototype" in disclaimer, (
            f"Disclaimer should mention 'research' or 'prototype': {cfg['disclaimer']!r}"
        )

    def test_config_is_valid_json(self) -> None:
        """REQ-CORE-003: Config JSON is parseable (no syntax errors)."""
        config_path = _MODEL_CARD_DIR / "config.json"
        assert config_path.exists(), f"config.json not found at {config_path}"
        with open(config_path) as f:
            cfg = json.load(f)  # raises if invalid JSON
        assert isinstance(cfg, dict)


# ---------------------------------------------------------------------------
# Adapter module loader (Python 3.14 compatible)
# ---------------------------------------------------------------------------


def _load_adapter_module() -> Any:
    """Load guided_decoding_adapter.py as a module, Python-3.14-safe.

    Python 3.14's dataclass mechanism requires the module to be registered
    in sys.modules before exec_module is called, otherwise the @dataclass
    decorator cannot resolve the module namespace for annotation processing.

    Returns:
        Loaded module object with EnergyGuidedSampler and GuidedDecodingResult.
    """
    import importlib.util
    from types import ModuleType

    module_name = "guided_decoding_adapter"
    # Reuse cached module if already loaded (avoids re-exec across tests)
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        module_name,
        _MODEL_CARD_DIR / "guided_decoding_adapter.py",
    )
    assert spec is not None and spec.loader is not None, (
        "Could not create module spec for guided_decoding_adapter.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so @dataclass can resolve
    # the module namespace. Required for Python 3.14.
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception:
        del sys.modules[module_name]
        raise
    return mod


# ---------------------------------------------------------------------------
# Model card content tests
# ---------------------------------------------------------------------------


class TestModelCardContent:
    """REQ-CORE-001: Model cards contain required sections for discoverability."""

    def test_readme_exists(self) -> None:
        """REQ-CORE-001: README.md exists in models/constraint-verifier-v2/."""
        readme = _MODEL_CARD_DIR / "README.md"
        assert readme.exists(), f"README.md not found at {readme}"

    def test_readme_has_disclaimer(self) -> None:
        """REQ-CORE-001: README.md contains research prototype disclaimer."""
        readme = _MODEL_CARD_DIR / "README.md"
        content = readme.read_text()
        assert "Research Prototype" in content or "research prototype" in content.lower(), (
            "README.md must contain a 'Research Prototype' disclaimer"
        )

    def test_readme_has_auroc_table(self) -> None:
        """REQ-CORE-001: README.md contains AUROC performance table."""
        readme = _MODEL_CARD_DIR / "README.md"
        content = readme.read_text()
        assert "AUROC" in content, "README.md must contain an AUROC performance table"

    def test_readme_has_phase_comparison(self) -> None:
        """REQ-CORE-001: README.md distinguishes Phase 1 vs Phase 5+ artifacts."""
        readme = _MODEL_CARD_DIR / "README.md"
        content = readme.read_text()
        assert "Phase 1" in content and "Phase 5" in content, (
            "README.md must contain Phase 1 vs Phase 5+ comparison"
        )

    def test_readme_has_limitations(self) -> None:
        """REQ-CORE-001: README.md contains Limitations section."""
        readme = _MODEL_CARD_DIR / "README.md"
        content = readme.read_text()
        assert "Limitations" in content or "limitations" in content, (
            "README.md must contain a Limitations section"
        )

    def test_readme_has_installation(self) -> None:
        """REQ-CORE-001: README.md contains installation instructions."""
        readme = _MODEL_CARD_DIR / "README.md"
        content = readme.read_text()
        assert "pip install carnot" in content or "Installation" in content, (
            "README.md must contain installation instructions"
        )

    def test_readme_guided_exists(self) -> None:
        """REQ-CORE-001: README_guided.md exists."""
        readme_guided = _MODEL_CARD_DIR / "README_guided.md"
        assert readme_guided.exists(), f"README_guided.md not found at {readme_guided}"

    def test_readme_guided_has_required_sections(self) -> None:
        """REQ-CORE-001: README_guided.md contains required sections."""
        readme_guided = _MODEL_CARD_DIR / "README_guided.md"
        content = readme_guided.read_text()
        for section in _GUIDED_README_REQUIRED_SECTIONS:
            assert section.lower() in content.lower(), (
                f"README_guided.md missing required section: {section!r}"
            )

    def test_guided_decoding_adapter_exists(self) -> None:
        """REQ-CORE-001: guided_decoding_adapter.py exists as standalone file."""
        adapter = _MODEL_CARD_DIR / "guided_decoding_adapter.py"
        assert adapter.exists(), f"guided_decoding_adapter.py not found at {adapter}"

    def test_guided_decoding_adapter_importable(self) -> None:
        """REQ-CORE-001: guided_decoding_adapter.py can be imported standalone."""
        mod = _load_adapter_module()
        assert hasattr(mod, "EnergyGuidedSampler"), (
            "guided_decoding_adapter must export EnergyGuidedSampler"
        )
        assert hasattr(mod, "GuidedDecodingResult"), (
            "guided_decoding_adapter must export GuidedDecodingResult"
        )

    def test_adapter_sampler_init(self) -> None:
        """REQ-CORE-001: EnergyGuidedSampler in adapter initializes with valid params."""
        mod = _load_adapter_module()
        sampler = mod.EnergyGuidedSampler(alpha=0.5, check_every_k=5)
        assert sampler.alpha == 0.5
        assert sampler.check_every_k == 5
        assert sampler.energy_threshold == 0.0

    def test_adapter_sampler_rejects_bad_alpha(self) -> None:
        """REQ-CORE-001: EnergyGuidedSampler raises ValueError for negative alpha."""
        mod = _load_adapter_module()
        with pytest.raises(ValueError, match="alpha must be >= 0"):
            mod.EnergyGuidedSampler(alpha=-1.0)

    def test_adapter_compute_energy_penalty_short_text(self) -> None:
        """REQ-CORE-001: EnergyGuidedSampler returns 0 for very short text."""
        mod = _load_adapter_module()
        sampler = mod.EnergyGuidedSampler(alpha=0.5)
        energy = sampler.compute_energy_penalty("Hi")
        assert energy == 0.0, f"Expected 0.0 for short text, got {energy}"

    def test_adapter_modify_logits_no_penalty_below_threshold(self) -> None:
        """REQ-CORE-001: modify_logits returns logits unchanged when energy <= threshold."""
        mod = _load_adapter_module()
        sampler = mod.EnergyGuidedSampler(alpha=0.5, energy_threshold=1.0)
        logits = np.array([1.0, 2.0, 3.0])
        modified = sampler.modify_logits(logits, "test text", energy=0.5)
        # energy=0.5 <= threshold=1.0 → no penalty
        np.testing.assert_array_equal(modified, logits)

    def test_adapter_modify_logits_applies_penalty_above_threshold(self) -> None:
        """REQ-CORE-001: modify_logits subtracts alpha*energy when energy > threshold."""
        mod = _load_adapter_module()
        sampler = mod.EnergyGuidedSampler(alpha=0.5, energy_threshold=0.0)
        logits = np.array([1.0, 2.0, 3.0])
        modified = sampler.modify_logits(logits, "test text", energy=2.0)
        # energy=2.0 > threshold=0.0 → penalty = 0.5 * 2.0 = 1.0
        expected = logits - 1.0
        np.testing.assert_allclose(modified, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Build pipeline integration test
# ---------------------------------------------------------------------------


class TestBuildKANModel:
    """REQ-CORE-001: build_kan_model function produces a valid KAN model."""

    def test_build_kan_model_produces_valid_model(self) -> None:
        """REQ-CORE-001: build_kan_model returns a KANEnergyFunction."""
        model = build_kan_model(KAN_CONFIG, seed=0)
        assert isinstance(model, KANEnergyFunction)

    def test_build_kan_model_energy_is_finite(self) -> None:
        """REQ-CORE-001: KAN energy is finite for a ones input."""
        model = build_kan_model(KAN_CONFIG, seed=0)
        x = jnp.ones(KAN_CONFIG.input_dim)
        energy = model.energy(x)
        assert jnp.isfinite(energy), f"Energy must be finite, got {energy}"

    def test_build_kan_model_reproducible(self) -> None:
        """REQ-CORE-001: Same seed produces same energy."""
        model_a = build_kan_model(KAN_CONFIG, seed=7)
        model_b = build_kan_model(KAN_CONFIG, seed=7)
        x = jnp.array([0.1] * KAN_CONFIG.input_dim)
        e_a = float(model_a.energy(x))
        e_b = float(model_b.energy(x))
        assert abs(e_a - e_b) < 1e-6, (
            f"Same seed should give same energy: {e_a:.6f} vs {e_b:.6f}"
        )
