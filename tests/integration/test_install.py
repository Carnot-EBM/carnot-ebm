"""Integration tests for Carnot package installation and imports.

Verifies that the package is importable, exposes the expected version,
has a working console_scripts entrypoint, and that all public modules
are accessible. These are smoke tests for packaging correctness.

Spec: REQ-CODE-006
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# Import and version checks
# ---------------------------------------------------------------------------


class TestPackageImport:
    """Verify that carnot is importable and has correct metadata.

    Spec: REQ-CODE-006
    """

    def test_import_carnot(self) -> None:
        """REQ-CODE-006: Top-level import works."""
        import carnot

        assert hasattr(carnot, "__version__")

    def test_version_format(self) -> None:
        """REQ-CODE-006: Version is a non-empty string."""
        from carnot._version import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0
        # Should contain at least one dot (semver-ish).
        assert "." in __version__

    def test_version_matches_init(self) -> None:
        """REQ-CODE-006: __init__ re-exports the same version."""
        import carnot
        from carnot._version import __version__

        assert carnot.__version__ == __version__


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Verify that key public symbols are importable.

    Spec: REQ-CODE-006
    """

    def test_core_classes(self) -> None:
        """REQ-CODE-006: Core abstractions are importable."""
        from carnot import EnergyFunction, ModelConfig, ModelState

        assert EnergyFunction is not None
        assert ModelConfig is not None
        assert ModelState is not None

    def test_model_tiers(self) -> None:
        """REQ-CODE-006: All three model tiers importable."""
        from carnot import (
            BoltzmannConfig,
            BoltzmannModel,
            GibbsConfig,
            GibbsModel,
            IsingConfig,
            IsingModel,
        )

        assert IsingModel is not None
        assert GibbsModel is not None
        assert BoltzmannModel is not None

    def test_samplers(self) -> None:
        """REQ-CODE-006: Samplers importable."""
        from carnot import HMCSampler, LangevinSampler

        assert LangevinSampler is not None
        assert HMCSampler is not None

    def test_training_losses(self) -> None:
        """REQ-CODE-006: Training losses importable."""
        from carnot import dsm_loss, nce_loss, snl_loss

        assert callable(dsm_loss)
        assert callable(nce_loss)
        assert callable(snl_loss)

    def test_pipeline_classes(self) -> None:
        """REQ-CODE-006: Pipeline classes importable."""
        from carnot.pipeline.verify_repair import (
            RepairResult,
            VerificationResult,
            VerifyRepairPipeline,
        )

        assert VerifyRepairPipeline is not None
        assert VerificationResult is not None
        assert RepairResult is not None

    def test_pipeline_errors(self) -> None:
        """REQ-CODE-006: Error hierarchy importable."""
        from carnot.pipeline.errors import (
            CarnotError,
            ExtractionError,
            ModelLoadError,
            PipelineTimeoutError,
            RepairError,
            VerificationError,
        )

        assert issubclass(ExtractionError, CarnotError)
        assert issubclass(VerificationError, CarnotError)
        assert issubclass(RepairError, CarnotError)
        assert issubclass(ModelLoadError, CarnotError)
        assert issubclass(PipelineTimeoutError, CarnotError)

    def test_rust_compat_flag(self) -> None:
        """REQ-CODE-006: RUST_AVAILABLE flag is exposed."""
        from carnot import RUST_AVAILABLE

        assert isinstance(RUST_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# Submodule imports
# ---------------------------------------------------------------------------


class TestSubmoduleImports:
    """Verify key submodules are importable without errors.

    Spec: REQ-CODE-006
    """

    @pytest.mark.parametrize(
        "module_path",
        [
            "carnot.core",
            "carnot.models",
            "carnot.samplers",
            "carnot.training",
            "carnot.verify",
            "carnot.pipeline",
            "carnot.pipeline.extract",
            "carnot.pipeline.verify_repair",
            "carnot.pipeline.errors",
            "carnot.cli",
        ],
    )
    def test_submodule_importable(self, module_path: str) -> None:
        """REQ-CODE-006: Submodule imports without error."""
        mod = importlib.import_module(module_path)
        assert mod is not None


# ---------------------------------------------------------------------------
# Entrypoint check
# ---------------------------------------------------------------------------


class TestEntrypoint:
    """Verify the carnot console_scripts entrypoint works.

    Spec: REQ-CODE-006
    """

    def test_carnot_m_cli_runs(self) -> None:
        """REQ-CODE-006: python -m carnot.cli runs without crash."""
        result = subprocess.run(
            [sys.executable, "-m", "carnot.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            env={**__import__("os").environ, "JAX_PLATFORMS": "cpu"},
        )
        assert result.returncode == 0
        assert "carnot" in result.stdout.lower()

    def test_carnot_entrypoint_exists(self) -> None:
        """REQ-CODE-006: 'carnot' entrypoint resolves to cli:main."""
        from carnot.cli import main

        assert callable(main)
