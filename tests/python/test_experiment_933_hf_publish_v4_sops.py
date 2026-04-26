"""Tests for Exp 933: HuggingFace Publish v4 — SOPS credential injection + upload.

Covers the logic added by this experiment:
  (a) auth_required_sops_missing verdict when no token is available
  (b) hf_auth_failed verdict when token present but login rejected
  (c) hf_published verdict when both models upload successfully
  (d) hf_published_partial verdict when one model fails
  (e) _repo_create idempotency (409 / already-exists is treated as success)
  (f) _hf_upload delegates to subprocess correctly
  (g) _publish_vjepa_v2 handles missing card / missing weights
  (h) _publish_estimation_verifier handles missing card / missing pipeline dir

All network calls (subprocess.run / huggingface-cli) are mocked; no real uploads occur.

Spec: REQ-VERIFY-145, REQ-VERIFY-175, REQ-VER-085
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import experiment_933_hf_publish_v4_sops as _mod  # noqa: E402


# ---------------------------------------------------------------------------
# (a) auth_required_sops_missing — no token
# ---------------------------------------------------------------------------


class TestNoToken:
    """When decrypt_secret returns None the artifact must be auth_required_sops_missing."""

    def test_verdict_and_fields(self, tmp_path):
        with (
            patch.object(_mod, "_RESULT_PATH", tmp_path / "result.json"),
            patch("experiment_933_hf_publish_v4_sops.decrypt_secret", return_value=None),
        ):
            artifact = _mod.run_experiment()

        assert artifact["honest_verdict"] == "auth_required_sops_missing"
        assert artifact["hf_authenticated"] is False
        assert artifact["vjepa_v2_published"] is False
        assert artifact["estimation_verifier_published"] is False
        assert "action_required" in artifact
        # Must document the expected SOPS file path so operator knows what to create
        assert "secrets.enc.yaml" in artifact["action_required"]

    def test_schema_fields_present(self):
        with patch("experiment_933_hf_publish_v4_sops.decrypt_secret", return_value=None):
            artifact = _mod.run_experiment()

        required = {
            "experiment",
            "schema",
            "title",
            "run_date",
            "status",
            "honest_verdict",
            "duration_s",
            "spec",
        }
        assert required.issubset(artifact.keys())

    def test_experiment_number(self):
        with patch("experiment_933_hf_publish_v4_sops.decrypt_secret", return_value=None):
            artifact = _mod.run_experiment()
        assert artifact["experiment"] == 933


# ---------------------------------------------------------------------------
# (b) hf_auth_failed — token present but login rejected
# ---------------------------------------------------------------------------


class TestAuthFailed:
    """When huggingface-cli login returns non-zero the verdict must be hf_auth_failed."""

    def test_verdict(self):
        with (
            patch("experiment_933_hf_publish_v4_sops.decrypt_secret", return_value="bad_token"),
            patch.object(_mod, "_hf_login", return_value=False),
        ):
            artifact = _mod.run_experiment()

        assert artifact["honest_verdict"] == "hf_auth_failed"
        assert artifact["hf_authenticated"] is False
        assert artifact["vjepa_v2_published"] is False
        assert artifact["estimation_verifier_published"] is False

    def test_status_blocked(self):
        with (
            patch("experiment_933_hf_publish_v4_sops.decrypt_secret", return_value="bad_token"),
            patch.object(_mod, "_hf_login", return_value=False),
        ):
            artifact = _mod.run_experiment()
        assert artifact["status"] == "blocked"


# ---------------------------------------------------------------------------
# (c) hf_published — both models succeed
# ---------------------------------------------------------------------------


class TestBothPublished:
    """When both _publish_vjepa_v2 and _publish_estimation_verifier succeed."""

    _VJEPA_OK = {
        "repo_created": True,
        "card_uploaded": True,
        "weights_uploaded": True,
        "error": None,
    }
    _EST_OK = {
        "repo_created": True,
        "card_uploaded": True,
        "pipeline_uploaded": True,
        "error": None,
    }

    def test_verdict_hf_published(self):
        with (
            patch("experiment_933_hf_publish_v4_sops.decrypt_secret", return_value="tok"),
            patch.object(_mod, "_hf_login", return_value=True),
            patch.object(_mod, "_publish_vjepa_v2", return_value=self._VJEPA_OK),
            patch.object(_mod, "_publish_estimation_verifier", return_value=self._EST_OK),
        ):
            artifact = _mod.run_experiment()

        assert artifact["honest_verdict"] == "hf_published"
        assert artifact["status"] == "success"
        assert artifact["hf_authenticated"] is True
        assert artifact["vjepa_v2_published"] is True
        assert artifact["estimation_verifier_published"] is True


# ---------------------------------------------------------------------------
# (d) hf_published_partial — one fails
# ---------------------------------------------------------------------------


class TestPartialPublish:
    """When only one model publishes the verdict must be hf_published_partial."""

    _VJEPA_OK = {
        "repo_created": True,
        "card_uploaded": True,
        "weights_uploaded": True,
        "error": None,
    }
    _EST_FAIL = {
        "repo_created": True,
        "card_uploaded": False,
        "pipeline_uploaded": False,
        "error": "card upload failed",
    }

    def test_verdict_partial(self):
        with (
            patch("experiment_933_hf_publish_v4_sops.decrypt_secret", return_value="tok"),
            patch.object(_mod, "_hf_login", return_value=True),
            patch.object(_mod, "_publish_vjepa_v2", return_value=self._VJEPA_OK),
            patch.object(_mod, "_publish_estimation_verifier", return_value=self._EST_FAIL),
        ):
            artifact = _mod.run_experiment()

        assert artifact["honest_verdict"] == "hf_published_partial"
        assert artifact["vjepa_v2_published"] is True
        assert artifact["estimation_verifier_published"] is False

    def test_both_fail_still_partial(self):
        fail = {
            "repo_created": False,
            "card_uploaded": False,
            "weights_uploaded": False,
            "error": "network",
        }
        with (
            patch("experiment_933_hf_publish_v4_sops.decrypt_secret", return_value="tok"),
            patch.object(_mod, "_hf_login", return_value=True),
            patch.object(_mod, "_publish_vjepa_v2", return_value=fail),
            patch.object(_mod, "_publish_estimation_verifier", return_value=fail),
        ):
            artifact = _mod.run_experiment()
        assert artifact["honest_verdict"] == "hf_published_partial"


# ---------------------------------------------------------------------------
# (e) _repo_create idempotency
# ---------------------------------------------------------------------------


class TestRepoCreate:
    """409 / already-exists from huggingface-cli must be treated as success."""

    def test_rc1_already_exists(self):
        mock_result = MagicMock(returncode=1, stdout="", stderr="Repository already exists")
        with patch("subprocess.run", return_value=mock_result):
            ok, err = _mod._repo_create("Carnot-EBM/vjepa-v2")
        assert ok is True

    def test_rc0_new_repo(self):
        mock_result = MagicMock(returncode=0, stdout="Created", stderr="")
        with patch("subprocess.run", return_value=mock_result):
            ok, _ = _mod._repo_create("Carnot-EBM/vjepa-v2")
        assert ok is True

    def test_rc1_other_error(self):
        mock_result = MagicMock(returncode=1, stdout="", stderr="Permission denied")
        with patch("subprocess.run", return_value=mock_result):
            ok, err = _mod._repo_create("Carnot-EBM/vjepa-v2")
        assert ok is False


# ---------------------------------------------------------------------------
# (f) _hf_upload delegates to subprocess
# ---------------------------------------------------------------------------


class TestHfUpload:
    def test_success(self):
        mock_result = MagicMock(returncode=0, stdout="uploaded", stderr="")
        with patch("subprocess.run", return_value=mock_result):
            ok, err = _mod._hf_upload("Carnot-EBM/vjepa-v2", "/tmp/card.md", "README.md")
        assert ok is True

    def test_failure(self):
        mock_result = MagicMock(returncode=1, stdout="", stderr="Forbidden")
        with patch("subprocess.run", return_value=mock_result):
            ok, err = _mod._hf_upload("Carnot-EBM/vjepa-v2", "/tmp/card.md", "README.md")
        assert ok is False
        assert "Forbidden" in err


# ---------------------------------------------------------------------------
# (g) _publish_vjepa_v2 — missing card / missing weights
# ---------------------------------------------------------------------------


class TestPublishVjepaV2:
    def test_missing_card_returns_false(self, tmp_path):
        with (
            patch.object(_mod, "_VJEPA_CARD", tmp_path / "missing.md"),
            patch.object(_mod, "_repo_create", return_value=(True, "")),
        ):
            result = _mod._publish_vjepa_v2()
        assert result["card_uploaded"] is False
        assert result["weights_uploaded"] is False
        assert result["error"] is not None

    def test_card_present_weights_missing(self, tmp_path):
        card = tmp_path / "card.md"
        card.write_text("# VJEPA v2")
        with (
            patch.object(_mod, "_VJEPA_CARD", card),
            patch.object(_mod, "_VJEPA_WEIGHTS", tmp_path / "missing.safetensors"),
            patch.object(_mod, "_repo_create", return_value=(True, "")),
            patch.object(_mod, "_hf_upload", return_value=(True, "")),
        ):
            result = _mod._publish_vjepa_v2()
        assert result["card_uploaded"] is True
        assert result["weights_uploaded"] is False

    def test_repo_create_fails(self, tmp_path):
        with patch.object(_mod, "_repo_create", return_value=(False, "Permission denied")):
            result = _mod._publish_vjepa_v2()
        assert result["repo_created"] is False
        assert "repo create failed" in result["error"]

    def test_card_upload_fails(self, tmp_path):
        card = tmp_path / "card.md"
        card.write_text("# VJEPA v2")
        with (
            patch.object(_mod, "_VJEPA_CARD", card),
            patch.object(_mod, "_repo_create", return_value=(True, "")),
            patch.object(_mod, "_hf_upload", return_value=(False, "Forbidden")),
        ):
            result = _mod._publish_vjepa_v2()
        assert result["card_uploaded"] is False
        assert "card upload failed" in result["error"]


# ---------------------------------------------------------------------------
# (h) _publish_estimation_verifier — missing card / missing pipeline
# ---------------------------------------------------------------------------


class TestPublishEstimationVerifier:
    def test_missing_card(self, tmp_path):
        with (
            patch.object(_mod, "_EST_CARD", tmp_path / "missing.md"),
            patch.object(_mod, "_repo_create", return_value=(True, "")),
        ):
            result = _mod._publish_estimation_verifier()
        assert result["card_uploaded"] is False
        assert result["error"] is not None

    def test_card_present_pipeline_missing(self, tmp_path):
        card = tmp_path / "card.md"
        card.write_text("# EstimationVerifier")
        with (
            patch.object(_mod, "_EST_CARD", card),
            patch.object(_mod, "_PIPELINE_DIR", tmp_path / "missing_dir"),
            patch.object(_mod, "_repo_create", return_value=(True, "")),
            patch.object(_mod, "_hf_upload", return_value=(True, "")),
        ):
            result = _mod._publish_estimation_verifier()
        assert result["card_uploaded"] is True
        assert result["pipeline_uploaded"] is False

    def test_repo_create_fails(self):
        with patch.object(_mod, "_repo_create", return_value=(False, "error")):
            result = _mod._publish_estimation_verifier()
        assert result["repo_created"] is False
        assert result["error"] is not None

    def test_full_success(self, tmp_path):
        card = tmp_path / "card.md"
        card.write_text("# EstimationVerifier")
        pipeline = tmp_path / "pipeline"
        pipeline.mkdir()
        with (
            patch.object(_mod, "_EST_CARD", card),
            patch.object(_mod, "_PIPELINE_DIR", pipeline),
            patch.object(_mod, "_repo_create", return_value=(True, "")),
            patch.object(_mod, "_hf_upload", return_value=(True, "")),
        ):
            result = _mod._publish_estimation_verifier()
        assert result["card_uploaded"] is True
        assert result["pipeline_uploaded"] is True
        assert result["error"] is None
