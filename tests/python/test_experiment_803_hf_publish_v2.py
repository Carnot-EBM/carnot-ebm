"""Tests for Experiment 803 — HuggingFace Publish v2: SOPS HF_TOKEN + authenticated upload.

Spec: REQ-PUBLISH-005, REQ-PUBLISH-006, SCENARIO-PUBLISH-009
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import experiment_803_hf_publish_v2 as mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_tmpl(tmp_path: Path) -> MagicMock:
    """Build a minimal ExperimentTemplate-like stub pointing at tmp_path."""
    fake = MagicMock()
    fake._repo_root = str(tmp_path)
    # Mimic build_result so tests can inspect the returned dict.
    fake.build_result.side_effect = lambda data, **kw: {**data, **kw}
    return fake


def _setup_sops_doc(tmp_path: Path) -> None:
    """Create the SOPS doc and upload script stubs under tmp_path."""
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "sops-hf-token-setup.md").write_text("# SOPS setup")
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    (models / "hf_upload_commands.sh").write_text("#!/bin/bash")


# ---------------------------------------------------------------------------
# REQ-PUBLISH-005: get_hf_token reads both env var names
# ---------------------------------------------------------------------------


class TestGetHfToken:
    """REQ-PUBLISH-005: token check reads HF_TOKEN and HUGGING_FACE_HUB_TOKEN."""

    def test_reads_hf_token(self, monkeypatch):
        """REQ-PUBLISH-005: get_hf_token() returns value from HF_TOKEN."""
        monkeypatch.setenv("HF_TOKEN", "hf_abc123")
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        assert mod.get_hf_token() == "hf_abc123"

    def test_reads_legacy_hub_token(self, monkeypatch):
        """REQ-PUBLISH-005: get_hf_token() falls back to HUGGING_FACE_HUB_TOKEN."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy456")
        assert mod.get_hf_token() == "hf_legacy456"

    def test_returns_none_when_absent(self, monkeypatch):
        """REQ-PUBLISH-005: get_hf_token() returns None when neither var is set."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        assert mod.get_hf_token() is None

    def test_hf_token_takes_precedence(self, monkeypatch):
        """REQ-PUBLISH-005: HF_TOKEN wins when both vars are set."""
        monkeypatch.setenv("HF_TOKEN", "hf_primary")
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy")
        assert mod.get_hf_token() == "hf_primary"


# ---------------------------------------------------------------------------
# REQ-PUBLISH-006: honest_verdict maps correctly to auth state
# ---------------------------------------------------------------------------


class TestHonestVerdictMapping:
    """REQ-PUBLISH-006: honest_verdict reflects CLI/token/auth state correctly."""

    def test_hf_cli_not_installed_when_no_cli(self, tmp_path):
        """REQ-PUBLISH-006: honest_verdict=hf_cli_not_installed when no CLI in PATH."""
        _setup_sops_doc(tmp_path)
        fake = _make_fake_tmpl(tmp_path)

        with patch.object(mod, "get_hf_token", return_value="hf_tok"), \
             patch.object(mod, "find_hf_cli", return_value=None):
            result = mod.run_experiment(fake)

        assert result["honest_verdict"] == "hf_cli_not_installed"
        assert result["hf_authenticated"] is False

    def test_hf_auth_documented_when_token_absent(self, tmp_path):
        """REQ-PUBLISH-006: honest_verdict=hf_auth_documented when HF_TOKEN not set."""
        _setup_sops_doc(tmp_path)
        fake = _make_fake_tmpl(tmp_path)

        with patch.object(mod, "get_hf_token", return_value=None), \
             patch.object(mod, "find_hf_cli", return_value="hf"):
            result = mod.run_experiment(fake)

        assert result["honest_verdict"] == "hf_auth_documented"
        assert result["hf_token_present"] is False
        assert result["hf_authenticated"] is False

    def test_hf_auth_documented_when_whoami_fails(self, tmp_path):
        """REQ-PUBLISH-006: honest_verdict=hf_auth_documented when whoami returns False."""
        _setup_sops_doc(tmp_path)
        fake = _make_fake_tmpl(tmp_path)

        with patch.object(mod, "get_hf_token", return_value="hf_tok"), \
             patch.object(mod, "find_hf_cli", return_value="hf"), \
             patch.object(mod, "check_hf_auth", return_value=(False, "")):
            result = mod.run_experiment(fake)

        assert result["honest_verdict"] == "hf_auth_documented"
        assert result["hf_token_present"] is True
        assert result["hf_authenticated"] is False

    def test_hf_models_published_when_upload_succeeds(self, tmp_path):
        """SCENARIO-PUBLISH-009: honest_verdict=hf_models_published when upload OK."""
        _setup_sops_doc(tmp_path)
        fake = _make_fake_tmpl(tmp_path)

        with patch.object(mod, "get_hf_token", return_value="hf_tok"), \
             patch.object(mod, "find_hf_cli", return_value="hf"), \
             patch.object(mod, "check_hf_auth", return_value=(True, "testuser")), \
             patch.object(mod, "attempt_readme_update",
                          return_value=(True, "https://huggingface.co/Carnot-EBM/carnot-ising-sampler-v1")):
            result = mod.run_experiment(fake)

        assert result["honest_verdict"] == "hf_models_published"
        assert result["hf_authenticated"] is True
        assert "Carnot-EBM/carnot-ising-sampler-v1" in result["models_published"]

    def test_hf_auth_documented_when_upload_fails(self, tmp_path):
        """REQ-PUBLISH-006: honest_verdict=hf_auth_documented when authenticated but upload fails."""
        _setup_sops_doc(tmp_path)
        fake = _make_fake_tmpl(tmp_path)

        with patch.object(mod, "get_hf_token", return_value="hf_tok"), \
             patch.object(mod, "find_hf_cli", return_value="hf"), \
             patch.object(mod, "check_hf_auth", return_value=(True, "testuser")), \
             patch.object(mod, "attempt_readme_update", return_value=(False, "repo not found")):
            result = mod.run_experiment(fake)

        assert result["honest_verdict"] == "hf_auth_documented"
        assert result["hf_authenticated"] is True
        assert result["models_published"] == []

    def test_sops_doc_and_script_presence_reported(self, tmp_path):
        """REQ-PUBLISH-005/006: result reports whether SOPS doc and upload script exist."""
        _setup_sops_doc(tmp_path)
        fake = _make_fake_tmpl(tmp_path)

        with patch.object(mod, "get_hf_token", return_value=None), \
             patch.object(mod, "find_hf_cli", return_value="hf"):
            result = mod.run_experiment(fake)

        assert result["sops_doc_written"] is True
        assert result["upload_script_written"] is True
