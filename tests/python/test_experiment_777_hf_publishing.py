"""Tests for Experiment 777 — HuggingFace Publishing.

Spec: REQ-PUBLISH-010, REQ-PUBLISH-011,
      SCENARIO-PUBLISH-010, SCENARIO-PUBLISH-011
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import experiment_777_hf_publishing as mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exp752_json(artifact_paths: dict | None = None) -> str:
    """Build a minimal Exp 752 artifact JSON for test injection."""
    return json.dumps({
        "experiment": 752,
        "artifact_paths": artifact_paths or {},
        "honest_verdict": "hf_artifacts_ready",
    })


def _write_exp752(tmp_path: Path, artifact_paths: dict | None = None) -> None:
    """Write a fake Exp 752 result to tmp_path/results/."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "experiment_752_hf_model_preparation.json").write_text(
        _make_exp752_json(artifact_paths)
    )


def _make_fake_tmpl(tmp_path: Path) -> MagicMock:
    """Build a fake ExperimentTemplate-like object pointing at tmp_path."""
    fake = MagicMock()
    fake._repo_root = str(tmp_path)
    return fake


# ---------------------------------------------------------------------------
# REQ-PUBLISH-010: blocked when HF not authenticated
# ---------------------------------------------------------------------------


class TestHfNotAuthenticated:
    """REQ-PUBLISH-010: upload MUST NOT be attempted when HF_TOKEN is not set / whoami fails."""

    def test_blocked_when_not_authenticated(self, tmp_path):
        """REQ-PUBLISH-010: check_hf_authentication()=False → blocked_hf_not_authenticated."""
        _write_exp752(tmp_path)
        fake_tmpl = _make_fake_tmpl(tmp_path)

        with patch.object(mod, "check_hf_authentication", return_value=(False, "")):
            result = mod.run_experiment(fake_tmpl)

        assert result["honest_verdict"] == "blocked_hf_not_authenticated"
        assert result["hf_authenticated"] is False

    def test_no_upload_attempted_when_not_authenticated(self, tmp_path):
        """REQ-PUBLISH-010: upload_artifact is NEVER called when not authenticated."""
        _write_exp752(tmp_path)
        fake_tmpl = _make_fake_tmpl(tmp_path)

        with patch.object(mod, "check_hf_authentication", return_value=(False, "")), \
             patch.object(mod, "upload_artifact") as mock_upload:
            mod.run_experiment(fake_tmpl)

        mock_upload.assert_not_called()

    def test_no_readme_update_when_not_authenticated(self, tmp_path):
        """REQ-PUBLISH-010: update_readme_with_production_section never called when not authed."""
        _write_exp752(tmp_path)
        fake_tmpl = _make_fake_tmpl(tmp_path)

        with patch.object(mod, "check_hf_authentication", return_value=(False, "")), \
             patch.object(mod, "update_readme_with_production_section") as mock_readme:
            mod.run_experiment(fake_tmpl)

        mock_readme.assert_not_called()

    def test_check_hf_authentication_returns_false_on_cli_failure(self):
        """REQ-PUBLISH-010: whoami rc=1 → (False, '') from check_hf_authentication."""
        with patch.object(mod, "_run", return_value=(1, "", "Not logged in")):
            ok, username = mod.check_hf_authentication()
        assert ok is False
        assert username == ""

    def test_check_hf_authentication_returns_true_on_success(self):
        """REQ-PUBLISH-010: whoami rc=0 → (True, username)."""
        with patch.object(mod, "_run", return_value=(0, "testuser\n", "")):
            ok, username = mod.check_hf_authentication()
        assert ok is True
        assert username == "testuser"


# ---------------------------------------------------------------------------
# REQ-PUBLISH-010: n_models_published increments per successful upload
# ---------------------------------------------------------------------------


class TestModelsPublishedCount:
    """REQ-PUBLISH-010: n_models_published increments for each successfully uploaded model."""

    def _make_artifacts(self, tmp_path: Path) -> dict:
        """Write placeholder artifact files and return artifact_paths dict."""
        models_dir = tmp_path / "models"
        models_dir.mkdir(exist_ok=True)
        paths = {}
        for name in [
            "carnot_step_jepa_probe_v1.safetensors",
            "carnot_step_jepa_probe_v1_config.json",
            "MODELCARD_carnot_step_jepa_probe_v1.md",
            "carnot_kan_tier0b_v3.safetensors",
            "carnot_kan_tier0b_v3_config.json",
            "MODELCARD_carnot_kan_tier0b_v3.md",
        ]:
            (models_dir / name).write_text("placeholder")
        paths = {
            "jepa_weights": str(models_dir / "carnot_step_jepa_probe_v1.safetensors"),
            "jepa_config": str(models_dir / "carnot_step_jepa_probe_v1_config.json"),
            "jepa_model_card": str(models_dir / "MODELCARD_carnot_step_jepa_probe_v1.md"),
            "kan_weights": str(models_dir / "carnot_kan_tier0b_v3.safetensors"),
            "kan_config": str(models_dir / "carnot_kan_tier0b_v3_config.json"),
            "kan_model_card": str(models_dir / "MODELCARD_carnot_kan_tier0b_v3.md"),
        }
        return paths

    def test_n_models_published_two_when_both_succeed(self, tmp_path):
        """REQ-PUBLISH-010: both JEPA and KAN upload ok → n_models_published=2."""
        paths = self._make_artifacts(tmp_path)
        _write_exp752(tmp_path, paths)
        fake_tmpl = _make_fake_tmpl(tmp_path)

        with patch.object(mod, "check_hf_authentication", return_value=(True, "testuser")), \
             patch.object(mod, "upload_artifact", return_value=(True, "https://hf.co/test")), \
             patch.object(mod, "get_existing_org_models", return_value=[]):
            result = mod.run_experiment(fake_tmpl)

        assert result["n_models_published"] == 2
        assert len(result["published_urls"]) == 2

    def test_n_models_published_zero_when_artifacts_missing(self, tmp_path):
        """REQ-PUBLISH-010: missing artifact files → n_models_published=0."""
        _write_exp752(tmp_path, {})
        fake_tmpl = _make_fake_tmpl(tmp_path)

        with patch.object(mod, "check_hf_authentication", return_value=(True, "testuser")), \
             patch.object(mod, "get_existing_org_models", return_value=[]):
            result = mod.run_experiment(fake_tmpl)

        assert result["n_models_published"] == 0

    def test_n_models_published_one_when_jepa_fails(self, tmp_path):
        """REQ-PUBLISH-010: JEPA upload fails, KAN succeeds → n_models_published=1."""
        paths = self._make_artifacts(tmp_path)
        _write_exp752(tmp_path, paths)
        fake_tmpl = _make_fake_tmpl(tmp_path)

        call_count = [0]

        def _upload_side_effect(repo_id, local_path):
            call_count[0] += 1
            # Fail all JEPA uploads (first 3 calls)
            if "jepa" in repo_id:
                return False, "upload failed"
            return True, f"https://hf.co/{repo_id}"

        with patch.object(mod, "check_hf_authentication", return_value=(True, "testuser")), \
             patch.object(mod, "upload_artifact", side_effect=_upload_side_effect), \
             patch.object(mod, "get_existing_org_models", return_value=[]):
            result = mod.run_experiment(fake_tmpl)

        assert result["n_models_published"] == 1


# ---------------------------------------------------------------------------
# REQ-PUBLISH-011: README update adds "pip install carnot" section
# ---------------------------------------------------------------------------


class TestReadmeUpdate:
    """REQ-PUBLISH-011: all 16 existing Carnot-EBM model READMEs MUST include pip install carnot."""

    def test_readme_update_adds_production_section(self, tmp_path):
        """REQ-PUBLISH-011: production section with 'pip install carnot' injected into README."""
        assert "pip install carnot" in mod._PRODUCTION_USE_SECTION

    def test_readme_update_section_contains_github_url(self):
        """REQ-PUBLISH-011: production section references carnot github URL."""
        assert "github.com/ianblenke/carnot" in mod._PRODUCTION_USE_SECTION

    def test_readme_update_section_mentions_phase1(self):
        """REQ-PUBLISH-011: section clarifies models are Phase 1 research artifacts."""
        assert "Phase 1" in mod._PRODUCTION_USE_SECTION

    def test_n_readmes_updated_increments_per_model(self, tmp_path):
        """REQ-PUBLISH-011: n_readmes_updated counts each successfully updated README."""
        paths: dict = {}
        _write_exp752(tmp_path, paths)
        fake_tmpl = _make_fake_tmpl(tmp_path)
        existing = ["Carnot-EBM/model-a", "Carnot-EBM/model-b", "Carnot-EBM/model-c"]

        with patch.object(mod, "check_hf_authentication", return_value=(True, "testuser")), \
             patch.object(mod, "upload_artifact", return_value=(False, "no artifacts")), \
             patch.object(mod, "get_existing_org_models", return_value=existing), \
             patch.object(mod, "update_readme_with_production_section", return_value=(True, "")):
            result = mod.run_experiment(fake_tmpl)

        assert result["n_readmes_updated"] == 3

    def _write_full_artifacts(self, tmp_path: Path) -> dict:
        """Write placeholder artifact files with real paths for verdict-logic tests."""
        models_dir = tmp_path / "models"
        models_dir.mkdir(exist_ok=True)
        for name in [
            "carnot_step_jepa_probe_v1.safetensors",
            "carnot_step_jepa_probe_v1_config.json",
            "MODELCARD_carnot_step_jepa_probe_v1.md",
            "carnot_kan_tier0b_v3.safetensors",
            "carnot_kan_tier0b_v3_config.json",
            "MODELCARD_carnot_kan_tier0b_v3.md",
        ]:
            (models_dir / name).write_text("placeholder")
        return {
            "jepa_weights": str(models_dir / "carnot_step_jepa_probe_v1.safetensors"),
            "jepa_config": str(models_dir / "carnot_step_jepa_probe_v1_config.json"),
            "jepa_model_card": str(models_dir / "MODELCARD_carnot_step_jepa_probe_v1.md"),
            "kan_weights": str(models_dir / "carnot_kan_tier0b_v3.safetensors"),
            "kan_config": str(models_dir / "carnot_kan_tier0b_v3_config.json"),
            "kan_model_card": str(models_dir / "MODELCARD_carnot_kan_tier0b_v3.md"),
        }

    def test_honest_verdict_readmes_updated_when_both_succeed(self, tmp_path):
        """REQ-PUBLISH-011: hf_published_readmes_updated when n_models > 0 AND n_readmes > 0."""
        paths_dict = self._write_full_artifacts(tmp_path)
        _write_exp752(tmp_path, paths_dict)
        fake_tmpl = _make_fake_tmpl(tmp_path)
        existing = ["Carnot-EBM/ebm-v1"]

        with patch.object(mod, "check_hf_authentication", return_value=(True, "u")), \
             patch.object(mod, "get_existing_org_models", return_value=existing), \
             patch.object(mod, "update_readme_with_production_section", return_value=(True, "")), \
             patch.object(mod, "upload_artifact", return_value=(True, "https://hf.co/test")):
            result = mod.run_experiment(fake_tmpl)

        assert result["honest_verdict"] == "hf_published_readmes_updated"
        assert result["n_readmes_updated"] >= 1

    def test_honest_verdict_artifacts_only_when_no_existing_models(self, tmp_path):
        """REQ-PUBLISH-011: hf_artifacts_uploaded_only when no existing org models found."""
        paths_dict = self._write_full_artifacts(tmp_path)
        _write_exp752(tmp_path, paths_dict)
        fake_tmpl = _make_fake_tmpl(tmp_path)

        with patch.object(mod, "check_hf_authentication", return_value=(True, "u")), \
             patch.object(mod, "get_existing_org_models", return_value=[]), \
             patch.object(mod, "upload_artifact", return_value=(True, "https://hf.co/test")):
            result = mod.run_experiment(fake_tmpl)

        assert result["honest_verdict"] == "hf_artifacts_uploaded_only"

    def test_honest_verdict_readmes_blocked_when_readme_update_fails(self, tmp_path):
        """REQ-PUBLISH-011: hf_published_readmes_blocked when readme updates all fail."""
        paths_dict = self._write_full_artifacts(tmp_path)
        _write_exp752(tmp_path, paths_dict)
        fake_tmpl = _make_fake_tmpl(tmp_path)
        existing = ["Carnot-EBM/ebm-v1"]

        with patch.object(mod, "check_hf_authentication", return_value=(True, "u")), \
             patch.object(mod, "get_existing_org_models", return_value=existing), \
             patch.object(mod, "update_readme_with_production_section", return_value=(False, "err")), \
             patch.object(mod, "upload_artifact", return_value=(True, "https://hf.co/test")):
            result = mod.run_experiment(fake_tmpl)

        assert result["honest_verdict"] == "hf_published_readmes_blocked"

    def test_update_readme_idempotent_when_section_present(self, tmp_path):
        """REQ-PUBLISH-011: update_readme_with_production_section returns (True, 'already_present')
        when the section already exists — does not re-upload."""
        existing_readme = "# MyModel\n\n## Production Use\n\npip install carnot\n"

        with patch.object(mod, "_run", side_effect=[
            # download succeeds
            (0, "", ""),
        ]):
            readme_path = tmp_path / "README.md"
            readme_path.write_text(existing_readme)

            # Patch the tempfile to use our tmp_path so README is found.
            import tempfile
            real_tempdir = tempfile.TemporaryDirectory

            class FakeTmpDir:
                def __init__(self): self.name = str(tmp_path)
                def __enter__(self): return self.name
                def __exit__(self, *_): pass

            with patch("tempfile.TemporaryDirectory", FakeTmpDir), \
                 patch.object(mod, "_run", return_value=(0, "", "")):
                ok, msg = mod.update_readme_with_production_section("Carnot-EBM/test-model")

        assert ok is True

    def test_get_existing_org_models_parses_cli_output(self):
        """REQ-PUBLISH-011: get_existing_org_models parses huggingface-cli list output."""
        fake_output = (
            "NAME                        TYPE\n"
            "---                         ---\n"
            "Carnot-EBM/ebm-v1           model\n"
            "Carnot-EBM/ebm-v2           model\n"
        )
        with patch.object(mod, "_run", return_value=(0, fake_output, "")):
            models = mod.get_existing_org_models("Carnot-EBM")
        assert "Carnot-EBM/ebm-v1" in models
        assert "Carnot-EBM/ebm-v2" in models

    def test_get_existing_org_models_returns_empty_on_failure(self):
        """REQ-PUBLISH-011: get_existing_org_models returns [] when CLI fails."""
        with patch.object(mod, "_run", return_value=(1, "", "error")):
            models = mod.get_existing_org_models("Carnot-EBM")
        assert models == []
