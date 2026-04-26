"""Tests for Experiment 829 — HuggingFace v3 Publish.

These tests cover:
  - disclaimer detection logic (_disclaimer_present / _prepend_disclaimer)
  - SOPS auth failure path producing hf_auth_blocked artifact
  - jepa_published=True / False conditioned on tier35_deployed from mocked Exp 825 JSON
  - injection_published=True conditioned on retro_injection_closed from mocked Exp 819 JSON

All tests mock huggingface_hub and sops_helper so no network access is required.

Spec traces: REQ-INFRA-062, SCENARIO-INFRA-070
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# Import the module under test.  We import individual helpers directly so
# tests run without triggering the apply_env_autofix / huggingface_hub import
# at module level.
# ---------------------------------------------------------------------------

import scripts.experiment_829_huggingface_v3_publish as exp829  # noqa: E402


# ---------------------------------------------------------------------------
# REQ-INFRA-062 — disclaimer detection
# ---------------------------------------------------------------------------


class TestDisclaimerPresent:
    """Covers _disclaimer_present() — the guard that avoids double-inserting."""

    def test_returns_false_when_disclaimer_absent(self) -> None:
        """A fresh README with no disclaimer should return False.

        REQ-INFRA-062: every card must eventually contain the disclaimer string.
        This test verifies the check correctly identifies cards that need updating.
        """
        readme = "# My Model\n\nThis is a cool model.\n"
        assert exp829._disclaimer_present(readme) is False

    def test_returns_true_when_disclaimer_present(self) -> None:
        """A README that already has the disclaimer should return True.

        REQ-INFRA-062: we must not prepend the disclaimer twice if it already exists.
        """
        readme = "Phase 1 research artifact. Some content.\n"
        assert exp829._disclaimer_present(readme) is True

    def test_prepend_adds_disclaimer_to_front(self) -> None:
        """_prepend_disclaimer must add the Phase 1 block before existing content.

        REQ-INFRA-062: the disclaimer must appear before any usage section so it
        is visible without scrolling.
        """
        original = "# Existing Model Card\n\nSome content.\n"
        updated = exp829._prepend_disclaimer(original)
        assert updated.startswith(exp829._PHASE1_DISCLAIMER)
        assert "# Existing Model Card" in updated
        assert "Some content." in updated

    def test_prepend_preserves_all_original_content(self) -> None:
        """_prepend_disclaimer must never delete existing README lines.

        CLAUDE.md: never remove existing content from docs when updating.
        """
        original = "# Old Title\n\nLine A.\nLine B.\nLine C.\n"
        updated = exp829._prepend_disclaimer(original)
        for line in ["# Old Title", "Line A.", "Line B.", "Line C."]:
            assert line in updated


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-070 — SOPS auth failure
# ---------------------------------------------------------------------------


class TestSopsAuthFailure:
    """Covers the hf_auth_blocked path when SOPS / env var both absent."""

    def test_hf_auth_blocked_when_token_unavailable(self, tmp_path: Path) -> None:
        """When decrypt_secret returns None the artifact must have hf_auth_blocked=True.

        SCENARIO-INFRA-070: experiment must write a deterministic blocked artifact
        rather than crash so the conductor can detect the auth failure and re-queue
        after the operator provisions the SOPS secret.
        """
        deliverable_path = tmp_path / "results" / "experiment_829_huggingface_v3_publish.json"

        # Build a minimal ExperimentTemplate stand-in
        mock_tmpl = MagicMock()
        mock_tmpl._repo_root = tmp_path

        # Simulate build_result producing a dict with the expected fields
        def fake_build_result(data: dict, **kwargs) -> dict:
            return {**data, "experiment": 829, "status": kwargs.get("status", "success")}

        mock_tmpl.build_result.side_effect = fake_build_result

        with patch(
            "scripts.experiment_829_huggingface_v3_publish.decrypt_secret", return_value=None
        ):
            artifact = exp829.run(mock_tmpl)

        assert artifact["hf_auth_blocked"] is True
        assert artifact["honest_verdict"] == "hf_auth_blocked"
        assert artifact["n_cards_updated"] == 0
        assert artifact["jepa_published"] is False
        assert artifact["injection_published"] is False

        # Verify the deliverable JSON was written to disk
        assert deliverable_path.exists(), "Blocked artifact must be written to results/"
        written = json.loads(deliverable_path.read_text())
        assert written["hf_auth_blocked"] is True


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-070 — JEPA v23 publish gating
# ---------------------------------------------------------------------------


class TestJepaPublishGating:
    """Covers jepa_published conditioned on tier35_deployed from Exp 825."""

    def _make_tmpl(self, tmp_path: Path) -> MagicMock:
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        mock_tmpl = MagicMock()
        mock_tmpl._repo_root = tmp_path

        def fake_build_result(data: dict, **kwargs) -> dict:
            return {**data, "experiment": 829, "status": kwargs.get("status", "success")}

        mock_tmpl.build_result.side_effect = fake_build_result
        return mock_tmpl

    def _write_exp825(self, tmp_path: Path, tier35_deployed: bool) -> None:
        result_file = tmp_path / "results" / "experiment_825_jepa_v23_eval_fr11_tier3.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(json.dumps({"tier35_deployed": tier35_deployed}))

    def _write_exp826(self, tmp_path: Path) -> None:
        result_file = tmp_path / "results" / "experiment_826_prm_cross_domain_benchmark.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(
            json.dumps(
                {
                    "in_dist_auc": 0.87,
                    "auc_gsm8k": 0.36,
                    "auc_humaneval": 0.76,
                    "auc_arc": 0.04,
                    "overall_ood_auc": 0.4,
                    "worst_domain": "arc",
                }
            )
        )

    def _write_exp819(self, tmp_path: Path, retro_injection_closed: bool) -> None:
        result_file = tmp_path / "results" / "experiment_819_injection_field_fix.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(
            json.dumps(
                {
                    "retro_injection_closed": retro_injection_closed,
                    "discrimination_rate": 1.0,
                    "n_pairs": 10,
                    "n_spins": 16,
                }
            )
        )

    def test_jepa_not_published_when_tier35_false(self, tmp_path: Path) -> None:
        """When tier35_deployed=False, jepa_published must be False.

        SCENARIO-INFRA-070: the Tier 3.5 gate (Exp 825) must control JEPA v23 publish.
        Publishing a model that failed its quality gate would violate the honesty principle.
        """
        self._write_exp825(tmp_path, tier35_deployed=False)
        self._write_exp826(tmp_path)
        self._write_exp819(tmp_path, retro_injection_closed=False)
        mock_tmpl = self._make_tmpl(tmp_path)

        # Mock hf_hub so no network calls are made
        mock_hf = MagicMock()
        mock_hf.list_models.return_value = iter(
            [
                SimpleNamespace(id="Carnot-EBM/fake-model-1"),
            ]
        )
        mock_hf.hf_hub_download.return_value = str(tmp_path / "fake_readme.md")
        (tmp_path / "fake_readme.md").write_text("Phase 1 research artifact. Already present.")

        with (
            patch(
                "scripts.experiment_829_huggingface_v3_publish.decrypt_secret",
                return_value="hf_fake_token",
            ),
            patch("scripts.experiment_829_huggingface_v3_publish.huggingface_hub", mock_hf),
            patch.object(
                exp829,
                "EXP_825_RESULT",
                tmp_path / "results/experiment_825_jepa_v23_eval_fr11_tier3.json",
            ),
            patch.object(
                exp829,
                "EXP_826_RESULT",
                tmp_path / "results/experiment_826_prm_cross_domain_benchmark.json",
            ),
            patch.object(
                exp829,
                "EXP_819_RESULT",
                tmp_path / "results/experiment_819_injection_field_fix.json",
            ),
        ):
            artifact = exp829.run(mock_tmpl)

        assert artifact["jepa_published"] is False

    def test_jepa_published_when_tier35_true(self, tmp_path: Path) -> None:
        """When tier35_deployed=True, jepa_published must be True.

        SCENARIO-INFRA-070: a JEPA model that passed the Tier 3.5 gate should
        be published so the community can use and validate it.
        """
        self._write_exp825(tmp_path, tier35_deployed=True)
        self._write_exp826(tmp_path)
        self._write_exp819(tmp_path, retro_injection_closed=False)
        mock_tmpl = self._make_tmpl(tmp_path)

        mock_hf = MagicMock()
        # list_models called twice: before and after publish
        mock_hf.list_models.side_effect = [
            iter([SimpleNamespace(id="Carnot-EBM/existing-1")]),  # n_existing
            iter(
                [
                    SimpleNamespace(id="Carnot-EBM/existing-1"),
                    SimpleNamespace(id="Carnot-EBM/jepa-v23-limo"),
                ]
            ),  # n_after
        ]
        (tmp_path / "fake_readme.md").write_text("Phase 1 research artifact. Already present.")
        mock_hf.hf_hub_download.return_value = str(tmp_path / "fake_readme.md")

        with (
            patch(
                "scripts.experiment_829_huggingface_v3_publish.decrypt_secret",
                return_value="hf_fake_token",
            ),
            patch("scripts.experiment_829_huggingface_v3_publish.huggingface_hub", mock_hf),
            patch.object(
                exp829,
                "EXP_825_RESULT",
                tmp_path / "results/experiment_825_jepa_v23_eval_fr11_tier3.json",
            ),
            patch.object(
                exp829,
                "EXP_826_RESULT",
                tmp_path / "results/experiment_826_prm_cross_domain_benchmark.json",
            ),
            patch.object(
                exp829,
                "EXP_819_RESULT",
                tmp_path / "results/experiment_819_injection_field_fix.json",
            ),
        ):
            artifact = exp829.run(mock_tmpl)

        assert artifact["jepa_published"] is True

    def test_injection_published_when_retro_closed(self, tmp_path: Path) -> None:
        """When retro_injection_closed=True, injection_published must be True.

        SCENARIO-INFRA-070: the external field fix (Exp 819) should be published
        once the retro ticket is closed so downstream users get the corrected injector.
        """
        self._write_exp825(tmp_path, tier35_deployed=False)
        self._write_exp826(tmp_path)
        self._write_exp819(tmp_path, retro_injection_closed=True)
        mock_tmpl = self._make_tmpl(tmp_path)

        mock_hf = MagicMock()
        mock_hf.list_models.side_effect = [
            iter([SimpleNamespace(id="Carnot-EBM/existing-1")]),
            iter(
                [
                    SimpleNamespace(id="Carnot-EBM/existing-1"),
                    SimpleNamespace(id="Carnot-EBM/ising-constraint-injector-v2"),
                ]
            ),
        ]
        (tmp_path / "fake_readme.md").write_text("Phase 1 research artifact. Already present.")
        mock_hf.hf_hub_download.return_value = str(tmp_path / "fake_readme.md")

        with (
            patch(
                "scripts.experiment_829_huggingface_v3_publish.decrypt_secret",
                return_value="hf_fake_token",
            ),
            patch("scripts.experiment_829_huggingface_v3_publish.huggingface_hub", mock_hf),
            patch.object(
                exp829,
                "EXP_825_RESULT",
                tmp_path / "results/experiment_825_jepa_v23_eval_fr11_tier3.json",
            ),
            patch.object(
                exp829,
                "EXP_826_RESULT",
                tmp_path / "results/experiment_826_prm_cross_domain_benchmark.json",
            ),
            patch.object(
                exp829,
                "EXP_819_RESULT",
                tmp_path / "results/experiment_819_injection_field_fix.json",
            ),
        ):
            artifact = exp829.run(mock_tmpl)

        assert artifact["injection_published"] is True
