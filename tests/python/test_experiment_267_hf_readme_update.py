"""Tests for Experiment 267: HuggingFace per-token EBM README batch update.

Verifies that the script:
1. Enumerates exactly 16 per-token EBM model repos under Carnot-EBM.
2. Prepends the Phase 1 status banner (without overwriting existing content).
3. Appends a "What's Proven to Work (2026)" section.
4. Logs repo name, HF URL, and push status per model.
5. Emits a clear blocker artifact when HF credentials are absent.
6. Uses huggingface_hub Python API (not subprocess) for push.

Spec: REQ-HF-267-A (banner present), REQ-HF-267-B (append structure),
      REQ-HF-267-C (repo enumeration), REQ-HF-267-D (mock HF push),
      REQ-HF-267-E (credential blocker)
SCENARIO-EXP267-A (banner prepend preserves existing content)
SCENARIO-EXP267-B (append section appended after existing content)
SCENARIO-EXP267-C (all 16 repos enumerated in MODEL_REPOS constant)
SCENARIO-EXP267-D (mock HF push logs success status)
SCENARIO-EXP267-E (unauthenticated path emits blocker artifact not exception)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def load_module() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_267_hf_readme_update.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_267_hf_readme_update",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Constants tests (SCENARIO-EXP267-C)
# ---------------------------------------------------------------------------


class TestRepoEnumeration:
    """REQ-HF-267-C: The 16 per-token EBM model repos are statically enumerated."""

    def test_model_repos_constant_has_16_entries(self) -> None:
        mod = load_module()
        assert len(mod.MODEL_REPOS) == 16

    def test_model_repos_all_strings(self) -> None:
        mod = load_module()
        for repo in mod.MODEL_REPOS:
            assert isinstance(repo, str), f"Expected str, got {type(repo)}: {repo}"

    def test_model_repos_all_under_carnot_ebm_org(self) -> None:
        mod = load_module()
        for repo in mod.MODEL_REPOS:
            assert repo.startswith("Carnot-EBM/"), (
                f"Repo {repo!r} must be under Carnot-EBM org"
            )

    def test_model_repos_include_known_models(self) -> None:
        mod = load_module()
        repos = set(mod.MODEL_REPOS)
        expected_samples = {
            "Carnot-EBM/per-token-ebm-qwen35-08b-nothink",
            "Carnot-EBM/per-token-ebm-gemma4-e2b-it-nothink",
            "Carnot-EBM/per-token-ebm-bonsai-17b-nothink",
        }
        for repo in expected_samples:
            assert repo in repos, f"{repo!r} missing from MODEL_REPOS"

    def test_no_duplicate_repos(self) -> None:
        mod = load_module()
        assert len(mod.MODEL_REPOS) == len(set(mod.MODEL_REPOS))


# ---------------------------------------------------------------------------
# Banner content tests (SCENARIO-EXP267-A)
# ---------------------------------------------------------------------------


class TestBannerContent:
    """REQ-HF-267-A: The status banner communicates Phase 1 research artifact status."""

    def test_banner_constant_exists(self) -> None:
        mod = load_module()
        assert hasattr(mod, "STATUS_BANNER")
        assert isinstance(mod.STATUS_BANNER, str)
        assert len(mod.STATUS_BANNER) > 50

    def test_banner_mentions_phase_1(self) -> None:
        mod = load_module()
        assert "PHASE 1" in mod.STATUS_BANNER.upper() or "phase 1" in mod.STATUS_BANNER.lower()

    def test_banner_mentions_confidence_not_correctness(self) -> None:
        mod = load_module()
        banner_lower = mod.STATUS_BANNER.lower()
        assert "confidence" in banner_lower or "detects" in banner_lower
        assert "correctness" in banner_lower or "correct" in banner_lower

    def test_banner_mentions_pip_install_carnot(self) -> None:
        mod = load_module()
        assert "pip install carnot" in mod.STATUS_BANNER

    def test_banner_mentions_mcp_server(self) -> None:
        mod = load_module()
        banner_lower = mod.STATUS_BANNER.lower()
        assert "mcp" in banner_lower

    def test_banner_mentions_formal_claim_verifier(self) -> None:
        mod = load_module()
        banner_lower = mod.STATUS_BANNER.lower()
        assert "formalclaimverifier" in banner_lower or "formal" in banner_lower


# ---------------------------------------------------------------------------
# Append section tests (SCENARIO-EXP267-B)
# ---------------------------------------------------------------------------


class TestAppendSection:
    """REQ-HF-267-B: The append section exists and mentions proven capabilities."""

    def test_proven_section_constant_exists(self) -> None:
        mod = load_module()
        assert hasattr(mod, "PROVEN_SECTION")
        assert isinstance(mod.PROVEN_SECTION, str)

    def test_proven_section_has_2026_header(self) -> None:
        mod = load_module()
        assert "2026" in mod.PROVEN_SECTION

    def test_proven_section_mentions_formal_claim_verifier(self) -> None:
        mod = load_module()
        assert "FormalClaimVerifier" in mod.PROVEN_SECTION or "formal" in mod.PROVEN_SECTION.lower()

    def test_proven_section_mentions_pbt(self) -> None:
        mod = load_module()
        section_lower = mod.PROVEN_SECTION.lower()
        assert "pbt" in section_lower or "property" in section_lower

    def test_proven_section_mentions_humaneval(self) -> None:
        mod = load_module()
        assert "HumanEval" in mod.PROVEN_SECTION or "164" in mod.PROVEN_SECTION

    def test_proven_section_mentions_process_integrity(self) -> None:
        mod = load_module()
        section_lower = mod.PROVEN_SECTION.lower()
        assert "process" in section_lower or "integrity" in section_lower

    def test_proven_section_mentions_mcp(self) -> None:
        mod = load_module()
        section_lower = mod.PROVEN_SECTION.lower()
        assert "mcp" in section_lower


# ---------------------------------------------------------------------------
# README mutation helpers (SCENARIO-EXP267-A, SCENARIO-EXP267-B)
# ---------------------------------------------------------------------------


class TestReadmeMutation:
    """REQ-HF-267-A/B: build_updated_readme prepends banner and appends section."""

    def test_banner_prepended_before_existing_content(self) -> None:
        mod = load_module()
        existing = "# My Model\n\nThis model does stuff.\n"
        result = mod.build_updated_readme(existing)
        assert result.startswith(mod.STATUS_BANNER) or mod.STATUS_BANNER in result[:len(mod.STATUS_BANNER) + 200]
        assert "# My Model" in result

    def test_existing_content_preserved(self) -> None:
        mod = load_module()
        existing = "# My Model\n\nThis model does stuff.\n"
        result = mod.build_updated_readme(existing)
        assert "# My Model" in result
        assert "This model does stuff." in result

    def test_proven_section_appended_after_existing(self) -> None:
        mod = load_module()
        existing = "# My Model\n\nExisting content.\n"
        result = mod.build_updated_readme(existing)
        banner_pos = result.find(mod.STATUS_BANNER[:30])
        existing_pos = result.find("Existing content.")
        proven_pos = result.find(mod.PROVEN_SECTION[:30])
        assert banner_pos < existing_pos, "Banner should appear before existing content"
        assert existing_pos < proven_pos, "Existing content should appear before proven section"

    def test_idempotent_already_has_banner(self) -> None:
        """If the README already contains the banner, status is skipped_already_current."""
        mod = load_module()
        existing = mod.STATUS_BANNER + "# My Model\n\n" + mod.PROVEN_SECTION
        result = mod.build_updated_readme(existing)
        # Should not double-insert
        assert result.count(mod.STATUS_BANNER[:40]) == 1

    def test_yaml_frontmatter_preserved(self) -> None:
        """YAML front-matter block (--- ... ---) must survive the mutation."""
        mod = load_module()
        existing = "---\nlicense: mit\ntags:\n- ebm\n---\n# My Model\nContent.\n"
        result = mod.build_updated_readme(existing)
        assert "license: mit" in result
        assert "tags:" in result


# ---------------------------------------------------------------------------
# is_already_current helper
# ---------------------------------------------------------------------------


class TestIsAlreadyCurrent:
    """Idempotency guard: detect when a README already has both additions."""

    def test_returns_true_when_both_present(self) -> None:
        mod = load_module()
        text = mod.STATUS_BANNER + "\n\n# Model\n\n" + mod.PROVEN_SECTION
        assert mod.is_already_current(text) is True

    def test_returns_false_when_banner_missing(self) -> None:
        mod = load_module()
        text = "# Model\n\n" + mod.PROVEN_SECTION
        assert mod.is_already_current(text) is False

    def test_returns_false_when_section_missing(self) -> None:
        mod = load_module()
        text = mod.STATUS_BANNER + "\n\n# Model\n\n"
        assert mod.is_already_current(text) is False

    def test_returns_false_for_empty_readme(self) -> None:
        mod = load_module()
        assert mod.is_already_current("") is False


# ---------------------------------------------------------------------------
# Mock HF push tests (SCENARIO-EXP267-D)
# ---------------------------------------------------------------------------


class TestMockHfPush:
    """REQ-HF-267-D: update_model_readme uses huggingface_hub API, logs status."""

    def test_success_path_returns_success_status(self) -> None:
        mod = load_module()
        mock_api = MagicMock()
        mock_api.model_info.return_value = MagicMock(modelId="Carnot-EBM/per-token-ebm-qwen35-08b-nothink")
        mock_api.hf_hub_download.return_value = None

        # Simulate fetching the README content via the API
        existing_readme = "# Existing Model Card\n\nContent here.\n"
        with patch.object(mod, "_fetch_readme", return_value=existing_readme):
            with patch.object(mod, "_push_readme", return_value=None):
                result = mod.update_model_readme(
                    mock_api,
                    "Carnot-EBM/per-token-ebm-qwen35-08b-nothink",
                )
        assert result["status"] == "success"
        assert result["repo_id"] == "Carnot-EBM/per-token-ebm-qwen35-08b-nothink"
        assert "hf_url" in result

    def test_already_current_returns_skipped(self) -> None:
        mod = load_module()
        mock_api = MagicMock()
        existing_readme = mod.STATUS_BANNER + "\n\n# Model\n\n" + mod.PROVEN_SECTION
        with patch.object(mod, "_fetch_readme", return_value=existing_readme):
            result = mod.update_model_readme(
                mock_api,
                "Carnot-EBM/per-token-ebm-qwen35-08b-nothink",
            )
        assert result["status"] == "skipped_already_current"

    def test_fetch_failure_returns_failed(self) -> None:
        mod = load_module()
        mock_api = MagicMock()
        with patch.object(mod, "_fetch_readme", side_effect=Exception("404 not found")):
            result = mod.update_model_readme(
                mock_api,
                "Carnot-EBM/per-token-ebm-qwen35-08b-nothink",
            )
        assert result["status"] == "failed"
        assert "404" in result.get("error", "")

    def test_push_failure_returns_failed(self) -> None:
        mod = load_module()
        mock_api = MagicMock()
        existing_readme = "# Model\n\nContent.\n"
        with patch.object(mod, "_fetch_readme", return_value=existing_readme):
            with patch.object(mod, "_push_readme", side_effect=Exception("auth error")):
                result = mod.update_model_readme(
                    mock_api,
                    "Carnot-EBM/per-token-ebm-qwen35-08b-nothink",
                )
        assert result["status"] == "failed"


# ---------------------------------------------------------------------------
# Credential blocker tests (SCENARIO-EXP267-E)
# ---------------------------------------------------------------------------


class TestCredentialBlocker:
    """REQ-HF-267-E: Missing HF credentials emit a clear blocker artifact."""

    def test_check_auth_returns_false_when_no_token(self) -> None:
        mod = load_module()
        with patch("huggingface_hub.whoami", side_effect=Exception("not logged in")):
            ok, reason = mod.check_authenticated()
        assert ok is False
        assert "login" in reason.lower() or "token" in reason.lower() or "not" in reason.lower()

    def test_check_auth_returns_true_when_authenticated(self) -> None:
        mod = load_module()
        with patch("huggingface_hub.whoami", return_value={"name": "test-user"}):
            ok, reason = mod.check_authenticated()
        assert ok is True
        assert reason == "test-user"

    def test_blocker_artifact_has_required_fields(self, tmp_path: Path) -> None:
        mod = load_module()
        results_path = tmp_path / "results.json"
        mod.write_blocker_artifact(
            results_path,
            "No HuggingFace token found (run `huggingface-cli login`)",
        )
        assert results_path.exists()
        data = json.loads(results_path.read_text(encoding="utf-8"))
        assert data["status"] == "blocked"
        assert "huggingface-cli login" in data["blocker_message"]
        assert data["experiment"] == 267
        assert data["run_date"] == "20260413"

    def test_blocker_artifact_lists_repos(self, tmp_path: Path) -> None:
        mod = load_module()
        results_path = tmp_path / "results.json"
        mod.write_blocker_artifact(results_path, "No token")
        data = json.loads(results_path.read_text(encoding="utf-8"))
        assert "repos_to_update" in data
        assert len(data["repos_to_update"]) == 16


# ---------------------------------------------------------------------------
# Results artifact schema tests
# ---------------------------------------------------------------------------


class TestResultsArtifact:
    """The results JSON has required fields and per-repo log entries."""

    def test_build_results_artifact_schema(self) -> None:
        mod = load_module()
        repo_logs = [
            {
                "repo_id": "Carnot-EBM/per-token-ebm-qwen35-08b-nothink",
                "hf_url": "https://huggingface.co/Carnot-EBM/per-token-ebm-qwen35-08b-nothink",
                "status": "success",
            },
            {
                "repo_id": "Carnot-EBM/per-token-ebm-gemma4-e2b-nothink",
                "hf_url": "https://huggingface.co/Carnot-EBM/per-token-ebm-gemma4-e2b-nothink",
                "status": "skipped_already_current",
            },
        ]
        artifact = mod.build_results_artifact(repo_logs)
        assert artifact["experiment"] == 267
        assert artifact["run_date"] == "20260413"
        assert artifact["status"] in ("complete", "partial", "blocked")
        assert "repo_logs" in artifact
        assert len(artifact["repo_logs"]) == 2
        assert "summary" in artifact

    def test_summary_counts_statuses(self) -> None:
        mod = load_module()
        repo_logs = [
            {"repo_id": "Carnot-EBM/a", "hf_url": "https://huggingface.co/Carnot-EBM/a", "status": "success"},
            {"repo_id": "Carnot-EBM/b", "hf_url": "https://huggingface.co/Carnot-EBM/b", "status": "success"},
            {"repo_id": "Carnot-EBM/c", "hf_url": "https://huggingface.co/Carnot-EBM/c", "status": "skipped_already_current"},
            {"repo_id": "Carnot-EBM/d", "hf_url": "https://huggingface.co/Carnot-EBM/d", "status": "failed"},
        ]
        artifact = mod.build_results_artifact(repo_logs)
        summary = artifact["summary"]
        assert summary["success"] == 2
        assert summary["skipped_already_current"] == 1
        assert summary["failed"] == 1
        assert summary["total"] == 4
