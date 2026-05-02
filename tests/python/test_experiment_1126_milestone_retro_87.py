"""Tests for experiment_1126_milestone_retro_87.

Covers only code added in this experiment — the retro evaluator and
artifact schema — not pre-existing code.  Every test has at least one
assertion (per CLAUDE.md mandate: no assertion-free tests).

Spec trace: REQ-RETRO-1126, SCENARIO-MILESTONE-RETRO-87-001.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the scripts directory importable without installing the package.
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import experiment_1126_milestone_retro_87 as retro


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_deliverables(tmp_path):
    """Return a helper that patches _load() with fixture data."""
    data = {
        1116: {"arxiv_bundle_created": True, "arxiv_submitted": False},
        1117: {"honest_verdict": "all_four_fixes_deployed"},
        1118: {
            "grpo_energy_prm_honest_result": True,
            "honest_verdict": "positive_improvement",
            "baseline_fraction_correct": 0.24,
            "trained_fraction_correct": 0.28,
            "improvement_over_baseline": 0.04,
        },
        1119: {"fover_sota_pairs_added_above_7000": True},
        1120: {
            "energy_inversion_measured_post_retrain": True,
            "energy_inversion_fixed": True,
            "mean_correct_energy_before": 0.689,
            "mean_incorrect_energy_before": 0.621,
            "mean_correct_energy_after": 1.647,
            "mean_incorrect_energy_after": 2.096,
            "retrained_auroc_val": 0.9774,
        },
        1121: {"k5_and_compose_production_deployed": True},
        1122: {"kv260_v4_kl_measured": True},
        1123: {"adaptive_cascade_savings_measured": True},
        1124: {"hashi_cartridge_shipped": True},
        1125: {"gallery_updated": True},
    }

    def _fake_load(exp_id: int) -> dict:
        return data.get(exp_id, {})

    with patch.object(retro, "_load", side_effect=_fake_load):
        yield data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvaluateCriteria:
    """REQ-RETRO-1126: all 11 criteria must be evaluable from deliverable JSONs."""

    def test_all_criteria_met_when_deliverables_complete(self, mock_deliverables):
        """All 11 criteria return True when all deliverables report success."""
        results = retro.evaluate_criteria()
        assert len(results) == 11, "Expected exactly 11 criteria"
        assert all(results.values()), f"Expected all True, got: {results}"

    def test_criteria_met_count_is_11(self, mock_deliverables):
        results = retro.evaluate_criteria()
        assert sum(results.values()) == 11

    def test_arxiv_criterion_uses_or_logic(self, mock_deliverables):
        """Criterion 1 is satisfied by bundle_created even when submitted=False."""
        results = retro.evaluate_criteria()
        assert results["arxiv_submitted_or_bundle_uploaded"] is True

    def test_infrastructure_criterion_accepts_all_four_verdict(self, mock_deliverables):
        results = retro.evaluate_criteria()
        assert results["infrastructure_3_bottlenecks_fixed"] is True

    def test_missing_deliverable_counts_as_false(self):
        """_load() returns {} for a missing file; criteria map that to False."""
        with patch.object(retro, "_load", return_value={}):
            results = retro.evaluate_criteria()
            # retro_complete is hardcoded True — all others should be False
            false_criteria = {k: v for k, v in results.items() if k != "retro_complete"}
            assert not any(false_criteria.values()), (
                f"Expected all non-retro criteria False on empty deliverables: {false_criteria}"
            )

    def test_retro_complete_always_true(self):
        """retro_complete is a sentinel that is always True regardless of deliverables."""
        with patch.object(retro, "_load", return_value={}):
            results = retro.evaluate_criteria()
        assert results["retro_complete"] is True


class TestBuildArtifact:
    """REQ-RETRO-1126: the artifact must satisfy the required schema fields."""

    REQUIRED_FIELDS = [
        "criteria_results",
        "criteria_met",
        "criteria_total",
        "wall_time_minutes",
        "experiments_completed",
        "slowest_experiments",
        "notable_successes",
        "bottlenecks_identified",
        "improvements_suggested",
        "energy_inversion_status",
        "grpo_result",
        "retro_complete",
        "honest_verdict",
    ]

    def test_all_required_fields_present(self, mock_deliverables):
        artifact = retro.build_artifact()
        missing = [f for f in self.REQUIRED_FIELDS if f not in artifact]
        assert not missing, f"Missing required fields: {missing}"

    def test_criteria_total_is_11(self, mock_deliverables):
        artifact = retro.build_artifact()
        assert artifact["criteria_total"] == 11

    def test_honest_verdict_when_all_met(self, mock_deliverables):
        artifact = retro.build_artifact()
        assert artifact["honest_verdict"] == "11_of_11_criteria_met"

    def test_retro_complete_sentinel(self, mock_deliverables):
        artifact = retro.build_artifact()
        assert artifact["retro_complete"] is True

    def test_slowest_experiments_is_list(self, mock_deliverables):
        artifact = retro.build_artifact()
        assert isinstance(artifact["slowest_experiments"], list)
        assert len(artifact["slowest_experiments"]) >= 1

    def test_slowest_experiment_has_required_keys(self, mock_deliverables):
        artifact = retro.build_artifact()
        for entry in artifact["slowest_experiments"]:
            for key in ("rank", "id", "title", "duration_min", "diagnosis"):
                assert key in entry, f"slowest_experiments entry missing key '{key}': {entry}"

    def test_wall_time_positive(self, mock_deliverables):
        artifact = retro.build_artifact()
        assert artifact["wall_time_minutes"] > 0


class TestEnergyInversionStatus:
    """REQ-RETRO-1126: energy inversion narrative must reflect exp1120 facts."""

    def test_reports_fixed_when_inversion_resolved(self, mock_deliverables):
        status = retro._derive_energy_inversion_status()
        assert "FIXED" in status

    def test_reports_before_and_after_values(self, mock_deliverables):
        status = retro._derive_energy_inversion_status()
        # The before values from the fixture are 0.689 and 0.621.
        assert "0.689" in status or "0.621" in status


class TestGrpoResult:
    """REQ-RETRO-1126: GRPO narrative must reflect exp1118 outcome."""

    def test_reports_positive_outcome(self, mock_deliverables):
        result = retro._derive_grpo_result()
        assert "positive" in result.lower()

    def test_contains_baseline_and_trained_fractions(self, mock_deliverables):
        result = retro._derive_grpo_result()
        # The fixture has baseline=0.24 and trained=0.28.
        assert "24%" in result or "0.24" in result or "28%" in result or "0.28" in result


class TestMainWritesArtifact:
    """Integration check: main() produces a valid JSON file."""

    def test_artifact_file_written(self, mock_deliverables, tmp_path, monkeypatch):
        result_path = tmp_path / "experiment_1126_milestone_retro_87.json"
        monkeypatch.setattr(retro, "RESULT_PATH", result_path)
        monkeypatch.setattr(retro, "RESULTS_DIR", tmp_path)
        retro.main()
        assert result_path.exists(), "Artifact file was not written"
        with open(result_path) as fh:
            data = json.load(fh)
        assert data["retro_complete"] is True
        assert data["criteria_total"] == 11
