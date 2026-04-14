"""Tests for Experiment 317: HuggingFace README accuracy audit and update.

Updates all 16 per-token activation EBM model READMEs with Phase 1
research artifact framing (finding from Exp 184/203: these models detect
model confidence, not factual correctness).  Also updates FCV README with
Exp 316 benchmark results and creates honest placeholder for joint-constraint.

Spec: REQ-PUBLISH-003, SCENARIO-PUBLISH-005, SCENARIO-PUBLISH-006
Run date: 20260414
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_api(
    existing_readme: str = "",
    whoami_result: dict | None = None,
    upload_raises: Exception | None = None,
) -> MagicMock:
    """Build a mock HfApi for injection into experiment_317 functions."""
    api = MagicMock()
    api.whoami.return_value = whoami_result or {"name": "ianblenke", "type": "user"}

    if existing_readme:
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.__str__ = lambda self: "mock_readme.md"
        # hf_hub_download returns a path; we simulate by returning the string directly
        api.hf_hub_download.return_value = existing_readme
    else:
        api.hf_hub_download.side_effect = Exception("404 README not found")

    if upload_raises is not None:
        api.upload_file.side_effect = upload_raises
    return api


# ---------------------------------------------------------------------------
# build_phase1_readme_patch — REQ-PUBLISH-003
# ---------------------------------------------------------------------------


class TestBuildPhase1ReadmePatch:
    """Verify the Phase 1 patch block content and structure."""

    def test_returns_string(self) -> None:
        """build_phase1_readme_patch() must return a non-empty string."""
        from scripts.experiment_317_hf_publish import build_phase1_readme_patch

        result = build_phase1_readme_patch()
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_phase1_sentinel(self) -> None:
        """Patch must contain the idempotency sentinel comment."""
        from scripts.experiment_317_hf_publish import (
            _PHASE1_SENTINEL,
            build_phase1_readme_patch,
        )

        result = build_phase1_readme_patch()
        assert _PHASE1_SENTINEL in result

    def test_contains_confidence_not_correctness_disclaimer(self) -> None:
        """Patch must state 'detects model confidence, not factual correctness'."""
        from scripts.experiment_317_hf_publish import build_phase1_readme_patch

        result = build_phase1_readme_patch()
        # The exact phrasing required by the spec
        assert "confidence" in result.lower()
        assert "correctness" in result.lower()
        # The disclaimer must be explicit
        lower = result.lower()
        assert "not factual correctness" in lower or "not correctness" in lower

    def test_contains_pip_install_carnot(self) -> None:
        """Patch must include 'pip install carnot' install instructions."""
        from scripts.experiment_317_hf_publish import build_phase1_readme_patch

        result = build_phase1_readme_patch()
        assert "pip install carnot" in result

    def test_contains_exp316_benchmark_when_results_provided(self) -> None:
        """When exp316_results are provided, patch includes benchmark summary."""
        from scripts.experiment_317_hf_publish import build_phase1_readme_patch

        exp316 = {
            "per_variant_results": {
                "all": {
                    "Qwen3.5-0.8B": {
                        "accuracy": 0.34,
                        "ci_lower": 0.254,
                        "ci_upper": 0.437,
                        "n_total": 100,
                    }
                }
            },
            "n_gsm8k": 100,
            "n_humaneval": 20,
        }
        result = build_phase1_readme_patch(exp316_results=exp316)
        assert "316" in result
        assert "Qwen3.5-0.8B" in result
        assert "34.0%" in result or "0.340" in result or "34%" in result

    def test_no_benchmark_section_when_results_none(self) -> None:
        """When exp316_results is None, patch must not include a benchmark table."""
        from scripts.experiment_317_hf_publish import build_phase1_readme_patch

        result = build_phase1_readme_patch(exp316_results=None)
        # Should not contain a benchmark table
        assert "| Model |" not in result
        assert "GSM8K" not in result

    def test_no_benchmark_section_when_no_all_variant(self) -> None:
        """When all-variant data is missing from results, no table is included."""
        from scripts.experiment_317_hf_publish import build_phase1_readme_patch

        exp316 = {"per_variant_results": {}, "n_gsm8k": 100, "n_humaneval": 20}
        result = build_phase1_readme_patch(exp316_results=exp316)
        assert "| Model |" not in result


# ---------------------------------------------------------------------------
# placeholder_card — REQ-PUBLISH-003
# ---------------------------------------------------------------------------


class TestPlaceholderCard:
    """Verify the honest placeholder model card for repos with no published weights."""

    def test_returns_string(self) -> None:
        """placeholder_card() must return a non-empty markdown string."""
        from scripts.experiment_317_hf_publish import placeholder_card

        result = placeholder_card("Carnot-EBM/carnot-joint-constraint-v1")
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_weights_not_published(self) -> None:
        """Placeholder card must include 'weights not published' label."""
        from scripts.experiment_317_hf_publish import placeholder_card

        result = placeholder_card("Carnot-EBM/carnot-joint-constraint-v1")
        assert "weights not published" in result.lower()

    def test_contains_research_prototype_label(self) -> None:
        """Placeholder card must include 'RESEARCH PROTOTYPE' label."""
        from scripts.experiment_317_hf_publish import placeholder_card

        result = placeholder_card("Carnot-EBM/carnot-joint-constraint-v1")
        assert "RESEARCH PROTOTYPE" in result

    def test_contains_pip_install_carnot(self) -> None:
        """Placeholder card must point to pip install carnot."""
        from scripts.experiment_317_hf_publish import placeholder_card

        result = placeholder_card("Carnot-EBM/carnot-joint-constraint-v1")
        assert "pip install carnot" in result

    def test_contains_auroc_methodology(self) -> None:
        """Placeholder card must mention the 1.0 AUROC held-out validation methodology."""
        from scripts.experiment_317_hf_publish import placeholder_card

        result = placeholder_card("Carnot-EBM/carnot-joint-constraint-v1")
        assert "AUROC" in result
        assert "1.0" in result

    def test_short_name_in_title(self) -> None:
        """Placeholder card title must use the short repo name (not org prefix)."""
        from scripts.experiment_317_hf_publish import placeholder_card

        result = placeholder_card("Carnot-EBM/carnot-joint-constraint-v1")
        assert "carnot-joint-constraint-v1" in result
        # Title line should start with '# carnot-...'
        assert "# carnot-joint-constraint-v1" in result


# ---------------------------------------------------------------------------
# model_card_update idempotency — SCENARIO-PUBLISH-005
# ---------------------------------------------------------------------------


class TestModelCardUpdateIdempotent:
    """Verify that model_card_update is idempotent on already-patched READMEs."""

    def test_skips_when_sentinel_present(self, tmp_path: Path) -> None:
        """If README already contains the sentinel, status must be 'skipped'."""
        from scripts.experiment_317_hf_publish import (
            _PHASE1_SENTINEL,
            model_card_update,
        )

        existing = f"{_PHASE1_SENTINEL}\nExisting content."
        api = _make_mock_api(existing_readme=existing)

        result = model_card_update(
            repo_id="Carnot-EBM/per-token-ebm-qwen3-06b",
            patch="Some patch",
            hf_api=api,
        )
        assert result["status"] == "skipped"
        # No upload should have been made
        api.upload_file.assert_not_called()

    def test_updates_when_sentinel_absent(self, tmp_path: Path) -> None:
        """If README does not contain sentinel, status must be 'updated'."""
        from scripts.experiment_317_hf_publish import model_card_update

        existing = "Existing content without the sentinel."
        api = _make_mock_api(existing_readme=existing)

        result = model_card_update(
            repo_id="Carnot-EBM/per-token-ebm-qwen3-06b",
            patch="<!-- carnot-exp317-phase1-patch -->\nPhase 1 disclaimer.",
            hf_api=api,
        )
        assert result["status"] == "updated"
        api.upload_file.assert_called_once()

    def test_dry_run_does_not_upload(self) -> None:
        """In dry_run mode, no upload must be made even when update is needed."""
        from scripts.experiment_317_hf_publish import model_card_update

        api = _make_mock_api(existing_readme="No sentinel here.")

        result = model_card_update(
            repo_id="Carnot-EBM/per-token-ebm-qwen3-06b",
            patch="<!-- carnot-exp317-phase1-patch -->\nPhase 1.",
            hf_api=api,
            dry_run=True,
        )
        assert result["status"] == "updated"
        api.upload_file.assert_not_called()

    def test_returns_hf_url(self) -> None:
        """model_card_update result must include hf_url for the repo."""
        from scripts.experiment_317_hf_publish import model_card_update

        api = _make_mock_api(existing_readme="")

        result = model_card_update(
            repo_id="Carnot-EBM/per-token-ebm-qwen3-06b",
            patch="<!-- carnot-exp317-phase1-patch -->\nPhase 1.",
            hf_api=api,
            dry_run=True,
        )
        assert "hf_url" in result
        assert "Carnot-EBM/per-token-ebm-qwen3-06b" in result["hf_url"]

    def test_error_status_on_upload_failure(self) -> None:
        """When upload raises, status must be 'error' with error message."""
        from scripts.experiment_317_hf_publish import model_card_update

        api = _make_mock_api(
            existing_readme="No sentinel.",
            upload_raises=Exception("Connection refused"),
        )
        result = model_card_update(
            repo_id="Carnot-EBM/per-token-ebm-qwen3-06b",
            patch="<!-- carnot-exp317-phase1-patch -->\nPhase 1.",
            hf_api=api,
            dry_run=False,
        )
        assert result["status"] == "error"
        assert "Connection refused" in result.get("error", "")


# ---------------------------------------------------------------------------
# build_fcv_readme_with_exp316
# ---------------------------------------------------------------------------


class TestBuildFcvReadmeWithExp316:
    """Verify FCV README patching with Exp 316 results."""

    def test_appends_exp316_section(self) -> None:
        """Exp 316 section is appended when not already present."""
        from scripts.experiment_317_hf_publish import build_fcv_readme_with_exp316

        existing = "# FCV Model\n\nExisting content."
        exp316 = {
            "per_variant_results": {
                "all": {
                    "Qwen3.5-0.8B": {
                        "accuracy": 0.34,
                        "ci_lower": 0.25,
                        "ci_upper": 0.44,
                        "n_total": 100,
                    }
                }
            },
            "n_gsm8k": 100,
            "n_humaneval": 20,
            "inference_mode": "simulated",
        }
        result = build_fcv_readme_with_exp316(existing, exp316)
        assert "316" in result
        assert "Qwen3.5-0.8B" in result
        assert "simulated" in result
        # Original content preserved
        assert "Existing content." in result

    def test_idempotent_when_sentinel_present(self) -> None:
        """If sentinel already in README, content must not be duplicated."""
        from scripts.experiment_317_hf_publish import build_fcv_readme_with_exp316

        exp316 = {
            "per_variant_results": {
                "all": {"Qwen3.5-0.8B": {"accuracy": 0.34, "ci_lower": 0.25, "ci_upper": 0.44, "n_total": 100}}
            },
            "n_gsm8k": 100,
            "n_humaneval": 20,
            "inference_mode": "simulated",
        }
        # First application
        existing = "# FCV\n\nContent."
        patched_once = build_fcv_readme_with_exp316(existing, exp316)
        # Second application
        patched_twice = build_fcv_readme_with_exp316(patched_once, exp316)
        # Should not have been changed
        assert patched_twice == patched_once

    def test_returns_existing_when_exp316_none(self) -> None:
        """When exp316_results is None, existing README is returned unchanged."""
        from scripts.experiment_317_hf_publish import build_fcv_readme_with_exp316

        existing = "# FCV\n\nContent."
        result = build_fcv_readme_with_exp316(existing, None)
        assert result == existing

    def test_returns_existing_when_no_all_variant(self) -> None:
        """When per_variant_results has no 'all' key, README is returned unchanged."""
        from scripts.experiment_317_hf_publish import build_fcv_readme_with_exp316

        existing = "# FCV\n\nContent."
        result = build_fcv_readme_with_exp316(existing, {"per_variant_results": {}})
        assert result == existing


# ---------------------------------------------------------------------------
# Credential check — SCENARIO-PUBLISH-006
# ---------------------------------------------------------------------------


class TestCredentialCheck317:
    """Verify credential checking with CLI + Python API fallback."""

    def test_true_when_cli_succeeds(self) -> None:
        """Returns True when huggingface-cli whoami succeeds."""
        from scripts.experiment_317_hf_publish import check_hf_credentials_317

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ianblenke\n", stderr="")
            ok, msg = check_hf_credentials_317()
        assert ok is True
        assert "ianblenke" in msg

    def test_true_when_cli_missing_but_api_works(self) -> None:
        """Falls back to Python API when CLI not found."""
        from scripts.experiment_317_hf_publish import check_hf_credentials_317

        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "ianblenke"}
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("scripts.experiment_317_hf_publish._make_hf_api", return_value=mock_api),
        ):
            ok, msg = check_hf_credentials_317()
        assert ok is True

    def test_false_when_both_fail(self) -> None:
        """Returns False with login instructions when CLI and API both fail."""
        from scripts.experiment_317_hf_publish import check_hf_credentials_317

        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("Not authenticated")
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("scripts.experiment_317_hf_publish._make_hf_api", return_value=mock_api),
        ):
            ok, msg = check_hf_credentials_317()
        assert ok is False
        assert "huggingface-cli login" in msg

    def test_false_when_cli_nonzero_and_api_fails(self) -> None:
        """Returns False when CLI non-zero exit and API also fails."""
        from scripts.experiment_317_hf_publish import check_hf_credentials_317

        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("Not authenticated")
        with (
            patch("subprocess.run") as mock_run,
            patch("scripts.experiment_317_hf_publish._make_hf_api", return_value=mock_api),
        ):
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="not logged in")
            ok, msg = check_hf_credentials_317()
        assert ok is False
        assert "huggingface-cli login" in msg


# ---------------------------------------------------------------------------
# Blocked artifact schema — SCENARIO-PUBLISH-006
# ---------------------------------------------------------------------------


class TestBlockedArtifact317:
    """Verify that a blocked 317 artifact matches the expected schema."""

    def _get_blocked(self, tmp_path: Path) -> dict:
        from scripts.experiment_317_hf_publish import run_experiment_317

        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("Not authenticated")
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("scripts.experiment_317_hf_publish._make_hf_api", return_value=mock_api),
        ):
            return run_experiment_317(
                dry_run=True,
                results_path=tmp_path / "experiment_317_hf_publish.json",
            )

    def test_blocked_has_experiment_317(self, tmp_path: Path) -> None:
        """Blocked artifact must have experiment == 317."""
        result = self._get_blocked(tmp_path)
        assert result.get("experiment") == 317

    def test_blocked_has_run_date(self, tmp_path: Path) -> None:
        """Blocked artifact must have run_date == '20260414'."""
        result = self._get_blocked(tmp_path)
        assert result.get("run_date") == "20260414"

    def test_blocked_is_true(self, tmp_path: Path) -> None:
        """Blocked artifact must have blocked == True."""
        result = self._get_blocked(tmp_path)
        assert result.get("blocked") is True

    def test_blocked_has_next_action(self, tmp_path: Path) -> None:
        """Blocked artifact must include exp_317_next_action with login command."""
        result = self._get_blocked(tmp_path)
        action = result.get("exp_317_next_action", "")
        assert "huggingface-cli login" in action
        assert "--token" in action

    def test_blocked_has_empty_models_updated(self, tmp_path: Path) -> None:
        """Blocked artifact must have models_updated == []."""
        result = self._get_blocked(tmp_path)
        assert result.get("models_updated") == []

    def test_blocked_written_to_disk(self, tmp_path: Path) -> None:
        """Blocked artifact must be written to results_path."""
        results_file = tmp_path / "experiment_317_hf_publish.json"
        from scripts.experiment_317_hf_publish import run_experiment_317

        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("Not authenticated")
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("scripts.experiment_317_hf_publish._make_hf_api", return_value=mock_api),
        ):
            run_experiment_317(dry_run=True, results_path=results_file)
        assert results_file.exists()
        data = json.loads(results_file.read_text())
        assert data["blocked"] is True


# ---------------------------------------------------------------------------
# Full pipeline — artifact schema — REQ-PUBLISH-003
# ---------------------------------------------------------------------------


class TestRunExperiment317Schema:
    """Verify the shape of a successful (dry_run) Exp 317 results artifact."""

    def _get_dry_run_result(self, tmp_path: Path) -> dict:
        from scripts.experiment_317_hf_publish import run_experiment_317

        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "ianblenke", "type": "user"}
        # Simulate fresh READMEs (no sentinel) for all repos
        mock_api.hf_hub_download.side_effect = Exception("404 not found")

        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("scripts.experiment_317_hf_publish._make_hf_api", return_value=mock_api),
        ):
            return run_experiment_317(
                dry_run=True,
                results_path=tmp_path / "exp317.json",
                hf_api=mock_api,
            )

    def test_has_experiment_317(self, tmp_path: Path) -> None:
        """Result must have experiment == 317."""
        result = self._get_dry_run_result(tmp_path)
        assert result.get("experiment") == 317

    def test_has_run_date_20260414(self, tmp_path: Path) -> None:
        """Result must have run_date == '20260414'."""
        result = self._get_dry_run_result(tmp_path)
        assert result.get("run_date") == "20260414"

    def test_not_blocked(self, tmp_path: Path) -> None:
        """Result must NOT have blocked == True when credentials OK."""
        result = self._get_dry_run_result(tmp_path)
        assert result.get("blocked") is not True

    def test_has_models_updated_list(self, tmp_path: Path) -> None:
        """Result must have models_updated as a list."""
        result = self._get_dry_run_result(tmp_path)
        assert isinstance(result.get("models_updated"), list)

    def test_has_models_skipped_list(self, tmp_path: Path) -> None:
        """Result must have models_skipped as a list."""
        result = self._get_dry_run_result(tmp_path)
        assert isinstance(result.get("models_skipped"), list)

    def test_has_errors_list(self, tmp_path: Path) -> None:
        """Result must have errors as a list."""
        result = self._get_dry_run_result(tmp_path)
        assert isinstance(result.get("errors"), list)

    def test_has_honest_verdict(self, tmp_path: Path) -> None:
        """Result must have honest_verdict dict."""
        result = self._get_dry_run_result(tmp_path)
        assert "honest_verdict" in result
        assert "status" in result["honest_verdict"]

    def test_dry_run_flag_in_result(self, tmp_path: Path) -> None:
        """Result must record dry_run == True in dry-run mode."""
        result = self._get_dry_run_result(tmp_path)
        assert result.get("dry_run") is True

    def test_result_written_to_disk(self, tmp_path: Path) -> None:
        """Results JSON must be written to the specified results_path."""
        results_file = tmp_path / "exp317_disk.json"
        from scripts.experiment_317_hf_publish import run_experiment_317

        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "ianblenke"}
        mock_api.hf_hub_download.side_effect = Exception("404")
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("scripts.experiment_317_hf_publish._make_hf_api", return_value=mock_api),
        ):
            run_experiment_317(dry_run=True, results_path=results_file, hf_api=mock_api)
        assert results_file.exists()
        data = json.loads(results_file.read_text())
        assert data.get("experiment") == 317


# ---------------------------------------------------------------------------
# No fake uploads — REQ-PUBLISH-003
# ---------------------------------------------------------------------------


class TestNoFakeUploads:
    """models_updated must only include repos with confirmed status from the API."""

    def test_models_updated_empty_when_all_skipped(self, tmp_path: Path) -> None:
        """When all READMEs already have the sentinel, models_updated must be empty."""
        from scripts.experiment_317_hf_publish import _PHASE1_SENTINEL, run_experiment_317

        already_patched = f"{_PHASE1_SENTINEL}\nAlready patched content."
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "ianblenke"}
        mock_api.hf_hub_download.return_value = already_patched

        # FCV and joint-constraint also already patched
        # (FCV has exp316 sentinel, joint has "RESEARCH PROTOTYPE — weights not published")
        def _side_effect(repo_id: str, filename: str, repo_type: str) -> str:
            if repo_id == "Carnot-EBM/carnot-formal-claim-verifier-v1":
                return "<!-- carnot-exp317-exp316-results -->\nAlready patched."
            if repo_id == "Carnot-EBM/carnot-joint-constraint-v1":
                return "RESEARCH PROTOTYPE — weights not published\nAlready done."
            return already_patched

        mock_api.hf_hub_download.side_effect = _side_effect

        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("scripts.experiment_317_hf_publish._make_hf_api", return_value=mock_api),
        ):
            result = run_experiment_317(
                dry_run=False,
                results_path=tmp_path / "exp317.json",
                hf_api=mock_api,
            )

        assert result["models_updated"] == []
        mock_api.upload_file.assert_not_called()

    def test_no_repo_counted_twice(self, tmp_path: Path) -> None:
        """A repo must not appear in both models_updated and models_skipped."""
        from scripts.experiment_317_hf_publish import run_experiment_317

        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "ianblenke"}
        mock_api.hf_hub_download.side_effect = Exception("404")

        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("scripts.experiment_317_hf_publish._make_hf_api", return_value=mock_api),
        ):
            result = run_experiment_317(
                dry_run=True,
                results_path=tmp_path / "exp317.json",
                hf_api=mock_api,
            )

        updated_set = set(result["models_updated"])
        skipped_set = set(result["models_skipped"])
        overlap = updated_set & skipped_set
        assert overlap == set(), f"Repos appear in both updated and skipped: {overlap}"


# ---------------------------------------------------------------------------
# Per-token EBM repo list completeness
# ---------------------------------------------------------------------------


class TestPerTokenEbmRepoList:
    """Verify the 16 per-token EBM repos are all enumerated."""

    def test_has_16_per_token_repos(self) -> None:
        """_PER_TOKEN_EBM_REPOS must contain exactly 16 entries."""
        from scripts.experiment_317_hf_publish import _PER_TOKEN_EBM_REPOS

        assert len(_PER_TOKEN_EBM_REPOS) == 16

    def test_all_repos_under_carnot_ebm_org(self) -> None:
        """All per-token EBM repos must be under the Carnot-EBM org."""
        from scripts.experiment_317_hf_publish import _PER_TOKEN_EBM_REPOS

        for repo in _PER_TOKEN_EBM_REPOS:
            assert repo.startswith("Carnot-EBM/"), f"Unexpected org in repo: {repo}"

    def test_all_repos_are_per_token_ebm(self) -> None:
        """All per-token EBM repo names must contain 'per-token-ebm'."""
        from scripts.experiment_317_hf_publish import _PER_TOKEN_EBM_REPOS

        for repo in _PER_TOKEN_EBM_REPOS:
            assert "per-token-ebm" in repo, f"Unexpected repo name: {repo}"


# ---------------------------------------------------------------------------
# Results JSON on-disk schema validation (if file exists)
# ---------------------------------------------------------------------------


class TestResultsJsonSchema317:
    """Validate results/experiment_317_hf_publish.json schema when file exists."""

    @pytest.fixture
    def results(self) -> dict:
        results_path = (
            Path(__file__).parent.parent.parent
            / "results"
            / "experiment_317_hf_publish.json"
        )
        if not results_path.exists():
            pytest.skip("experiment_317_hf_publish.json not yet generated")
        return json.loads(results_path.read_text())

    def test_has_experiment_317(self, results: dict) -> None:
        assert results.get("experiment") == 317

    def test_has_run_date_20260414(self, results: dict) -> None:
        assert results.get("run_date") == "20260414"

    def test_has_models_updated(self, results: dict) -> None:
        assert "models_updated" in results

    def test_has_models_skipped(self, results: dict) -> None:
        assert "models_skipped" in results

    def test_has_errors(self, results: dict) -> None:
        assert "errors" in results

    def test_no_repo_in_both_updated_and_skipped(self, results: dict) -> None:
        """No repo should appear in both models_updated and models_skipped."""
        overlap = set(results["models_updated"]) & set(results["models_skipped"])
        assert overlap == set()

    def test_blocked_has_next_action(self, results: dict) -> None:
        if not results.get("blocked"):
            pytest.skip("Not blocked")
        assert "huggingface-cli login" in results.get("exp_317_next_action", "")
