"""Tests for Experiment 304: HuggingFace publish attempt — credential check + upload retry.

Carry-forward from Exp 293 (blocked by missing CLI credentials).
Exp 304 adds Python API fallback for credential checking when huggingface-cli
is not in PATH, allowing upload when HF_TOKEN or cached credentials are present.

Publishes to:
  - Carnot-EBM/carnot-joint-constraint-v1  (Exp 66)
  - Carnot-EBM/carnot-formal-claim-verifier-v1  (FCV)

Spec: REQ-VERIFY-058, REQ-VERIFY-059
Run date: 20260414
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Credential check tests — REQ-VERIFY-058
# ---------------------------------------------------------------------------


class TestCredentialCheck304:
    """Verify credential checking with CLI + Python API fallback."""

    def test_check_credentials_returns_true_when_cli_succeeds(self) -> None:
        """check_hf_credentials_304() returns True when huggingface-cli whoami succeeds."""
        from scripts.experiment_304_hf_publish import check_hf_credentials_304

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Carnot-EBM\n", stderr="")
            ok, username = check_hf_credentials_304()
        assert ok is True
        assert username  # non-empty username

    def test_check_credentials_falls_back_to_python_api_when_cli_missing(self) -> None:
        """When CLI not found, fall back to Python API; return True if API works."""
        from scripts.experiment_304_hf_publish import check_hf_credentials_304

        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "ianblenke", "type": "user"}
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("huggingface-cli not found")),
            patch("scripts.experiment_304_hf_publish._make_hf_api", return_value=mock_api),
        ):
            ok, username = check_hf_credentials_304()
        assert ok is True
        assert "ianblenke" in username

    def test_check_credentials_false_when_cli_fails_and_api_fails(self) -> None:
        """When CLI fails AND Python API raises, return (False, instructions)."""
        from scripts.experiment_304_hf_publish import check_hf_credentials_304

        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("Not authenticated")
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("not found")),
            patch("scripts.experiment_304_hf_publish._make_hf_api", return_value=mock_api),
        ):
            ok, msg = check_hf_credentials_304()
        assert ok is False
        assert "huggingface-cli login" in msg

    def test_check_credentials_false_when_cli_returns_nonzero_and_api_fails(self) -> None:
        """When CLI returns non-zero AND API fails, return (False, instructions)."""
        from scripts.experiment_304_hf_publish import check_hf_credentials_304

        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("Not authenticated")
        with (
            patch("subprocess.run") as mock_run,
            patch("scripts.experiment_304_hf_publish._make_hf_api", return_value=mock_api),
        ):
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Not logged in")
            ok, msg = check_hf_credentials_304()
        assert ok is False
        assert "huggingface-cli login" in msg

    def test_exp304_next_action_in_blocked_artifact(self, tmp_path: Path) -> None:
        """Blocked artifact must include exp_304_next_action with login command."""
        from scripts.experiment_304_hf_publish import run_experiment_304

        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("Not authenticated")
        results_file = tmp_path / "experiment_304_hf_results.json"
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("not found")),
            patch("scripts.experiment_304_hf_publish._make_hf_api", return_value=mock_api),
        ):
            result = run_experiment_304(
                out_dir=tmp_path / "staging",
                dry_run=True,
                results_path=results_file,
            )
        assert result.get("blocked") is True
        action = result.get("exp_304_next_action", "")
        assert "huggingface-cli login" in action
        assert "--token" in action


# ---------------------------------------------------------------------------
# Blocked artifact schema — REQ-VERIFY-058
# ---------------------------------------------------------------------------


class TestBlockedArtifact304:
    """Verify that a blocked 304 artifact matches the expected schema."""

    def _get_blocked_result(self, tmp_path: Path) -> dict:
        from scripts.experiment_304_hf_publish import run_experiment_304

        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("Not authenticated")
        results_file = tmp_path / "experiment_304_hf_results.json"
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("not found")),
            patch("scripts.experiment_304_hf_publish._make_hf_api", return_value=mock_api),
        ):
            return run_experiment_304(
                out_dir=tmp_path / "staging",
                dry_run=True,
                results_path=results_file,
            )

    def test_blocked_has_experiment_304(self, tmp_path: Path) -> None:
        """Blocked artifact must have experiment == 304."""
        result = self._get_blocked_result(tmp_path)
        assert result.get("experiment") == 304

    def test_blocked_has_run_date(self, tmp_path: Path) -> None:
        """Blocked artifact must have run_date == '20260414'."""
        result = self._get_blocked_result(tmp_path)
        assert result.get("run_date") == "20260414"

    def test_blocked_has_blocked_true(self, tmp_path: Path) -> None:
        """Blocked artifact must have blocked == True."""
        result = self._get_blocked_result(tmp_path)
        assert result.get("blocked") is True

    def test_blocked_has_repo_ids(self, tmp_path: Path) -> None:
        """Blocked artifact must include repo_ids block."""
        result = self._get_blocked_result(tmp_path)
        assert "repo_ids" in result
        assert "carnot-joint-constraint-v1" in result["repo_ids"]["exp66"]
        assert "carnot-formal-claim-verifier-v1" in result["repo_ids"]["fcv"]

    def test_blocked_written_to_disk(self, tmp_path: Path) -> None:
        """Blocked artifact must be written to results_path."""
        from scripts.experiment_304_hf_publish import run_experiment_304

        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("Not authenticated")
        results_file = tmp_path / "experiment_304_hf_results.json"
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("not found")),
            patch("scripts.experiment_304_hf_publish._make_hf_api", return_value=mock_api),
        ):
            run_experiment_304(
                out_dir=tmp_path / "staging",
                dry_run=True,
                results_path=results_file,
            )
        assert results_file.exists()
        with open(results_file) as f:
            data = json.load(f)
        assert data["blocked"] is True


# ---------------------------------------------------------------------------
# Successful credential path (dry_run) — REQ-VERIFY-059
# ---------------------------------------------------------------------------


class TestSuccessfulCredentialPath304:
    """Verify behavior when credentials are valid (dry_run to avoid actual upload)."""

    def _get_dry_run_result(self, tmp_path: Path) -> dict:
        from scripts.experiment_304_hf_publish import run_experiment_304

        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "ianblenke", "type": "user",
                                        "orgs": [{"name": "Carnot-EBM"}]}
        results_file = tmp_path / "experiment_304_hf_results.json"
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("not found")),
            patch("scripts.experiment_304_hf_publish._make_hf_api", return_value=mock_api),
            patch("scripts.experiment_293_huggingface_publish._EXP66_SAFETENSORS_PATH",
                  tmp_path / "nonexistent.safetensors"),
        ):
            return run_experiment_304(
                out_dir=tmp_path / "staging",
                dry_run=True,
                results_path=results_file,
            )

    def test_not_blocked_when_credentials_ok(self, tmp_path: Path) -> None:
        """With valid credentials, result must NOT have blocked=True."""
        result = self._get_dry_run_result(tmp_path)
        assert result.get("blocked") is not True

    def test_has_experiment_304(self, tmp_path: Path) -> None:
        """Result must have experiment == 304."""
        result = self._get_dry_run_result(tmp_path)
        assert result.get("experiment") == 304

    def test_has_run_date_20260414(self, tmp_path: Path) -> None:
        """Result must have run_date == '20260414'."""
        result = self._get_dry_run_result(tmp_path)
        assert result.get("run_date") == "20260414"

    def test_fcv_upload_status_dry_run_or_uploaded(self, tmp_path: Path) -> None:
        """FCV artifact upload_status must be 'dry_run' or 'uploaded'."""
        result = self._get_dry_run_result(tmp_path)
        fcv_status = result["artifacts"]["fcv"]["upload_status"]
        assert fcv_status in ("dry_run", "uploaded"), (
            f"Expected dry_run or uploaded, got: {fcv_status}"
        )

    def test_fcv_hf_url_present(self, tmp_path: Path) -> None:
        """FCV artifact must include hf_url."""
        result = self._get_dry_run_result(tmp_path)
        assert result["artifacts"]["fcv"].get("hf_url"), "FCV must have hf_url"

    def test_repo_urls_in_result(self, tmp_path: Path) -> None:
        """Result must contain repo_urls list with FCV URL (exp66 skipped when safetensors absent)."""
        result = self._get_dry_run_result(tmp_path)
        assert "repo_urls" in result, "Must have repo_urls key"
        urls = result["repo_urls"]
        # FCV artifact is always built; exp66 may be skipped when safetensors absent
        assert any("carnot-formal-claim-verifier-v1" in u for u in urls)

    def test_credentials_available_true(self, tmp_path: Path) -> None:
        """Result must record credentials_available == True when login succeeded."""
        result = self._get_dry_run_result(tmp_path)
        assert result.get("credentials_available") is True

    def test_result_written_to_disk(self, tmp_path: Path) -> None:
        """Results JSON must be written to disk even in dry_run mode."""
        from scripts.experiment_304_hf_publish import run_experiment_304

        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "ianblenke", "type": "user",
                                        "orgs": [{"name": "Carnot-EBM"}]}
        results_file = tmp_path / "experiment_304_hf_results.json"
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("not found")),
            patch("scripts.experiment_304_hf_publish._make_hf_api", return_value=mock_api),
            patch("scripts.experiment_293_huggingface_publish._EXP66_SAFETENSORS_PATH",
                  tmp_path / "nonexistent.safetensors"),
        ):
            run_experiment_304(
                out_dir=tmp_path / "staging",
                dry_run=True,
                results_path=results_file,
            )
        assert results_file.exists()
        with open(results_file) as f:
            data = json.load(f)
        assert data.get("experiment") == 304


# ---------------------------------------------------------------------------
# Results JSON schema — on-disk file (if it exists)
# ---------------------------------------------------------------------------


class TestResultsJsonSchema304:
    """Validate results/experiment_304_hf_results.json schema when file exists."""

    @pytest.fixture
    def results(self) -> dict:
        results_path = (
            Path(__file__).parent.parent.parent / "results" / "experiment_304_hf_results.json"
        )
        if not results_path.exists():
            pytest.skip("experiment_304_hf_results.json not yet generated")
        with open(results_path) as f:
            return json.load(f)

    def test_has_experiment_304(self, results: dict) -> None:
        """Must have experiment == 304."""
        assert results.get("experiment") == 304

    def test_has_run_date_20260414(self, results: dict) -> None:
        """Must have run_date == '20260414'."""
        assert results.get("run_date") == "20260414"

    def test_has_credentials_available(self, results: dict) -> None:
        """Must record credentials_available (bool)."""
        assert "credentials_available" in results

    def test_has_artifacts_block(self, results: dict) -> None:
        """Must have top-level artifacts dict with exp66 and fcv keys."""
        assert "artifacts" in results
        assert "exp66" in results["artifacts"]
        assert "fcv" in results["artifacts"]

    def test_no_fabricated_uploaded_without_url(self, results: dict) -> None:
        """If upload_status == 'uploaded', hf_url must be non-None."""
        for name, art in results.get("artifacts", {}).items():
            if art.get("upload_status") == "uploaded":
                assert art.get("hf_url"), f"{name}: claimed 'uploaded' but no hf_url"

    def test_blocked_has_next_action(self, results: dict) -> None:
        """Blocked result must have exp_304_next_action with login command."""
        if not results.get("blocked"):
            pytest.skip("Not blocked — skipping next_action check")
        action = results.get("exp_304_next_action", "")
        assert "huggingface-cli login" in action

    def test_uploaded_has_repo_urls(self, results: dict) -> None:
        """If not blocked, must have repo_urls list."""
        if results.get("blocked"):
            pytest.skip("Blocked — no repo_urls expected")
        assert "repo_urls" in results
        assert len(results["repo_urls"]) >= 1
