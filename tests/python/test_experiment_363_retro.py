"""Tests for scripts/experiment_363_retro.py — Milestone 2026.05.20 retrospective.

Coverage targets
----------------
- load_result_file: missing path (None), file not found, valid JSON, invalid JSON
- load_all_results: multiple experiments, partial availability
- compute_statistics: normal case, skipped exclusion, edge cases
- evaluate_success_criteria: all six criteria with positive and negative fixture data
- compute_retro: integration test against fixture directory
- NEW_RETRO_ITEMS: structural validation (id, priority, description present)
- TOP_IMPROVEMENTS: structural validation (rank, action present)

Spec: REQ-INFRA-014 (live GPU gating), REQ-BENCH-006/007 (adversarial),
      REQ-EXTRACT-021 (LLMExtractor), REQ-LEARN-025 (EORM retrain)
SCENARIO: RETRO-2026.05.20
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure the repo root is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_363_retro import (
    MILESTONE,
    MILESTONE_EXPERIMENTS,
    NEW_RETRO_ITEMS,
    TOP_IMPROVEMENTS,
    compute_retro,
    compute_statistics,
    evaluate_success_criteria,
    load_all_results,
    load_result_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo_root(tmp_path: Path) -> Path:
    """Create a minimal fake repo structure for testing compute_retro."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return tmp_path


def _write_result(directory: Path, filename: str, data: dict[str, Any]) -> Path:
    """Write a JSON result file to a directory and return the path."""
    path = directory / filename
    path.write_text(json.dumps(data))
    return path


@pytest.fixture()
def exp352_data() -> dict[str, Any]:
    return {
        "experiment": 352,
        "status": "success",
        "is_live_capable": True,
        "inference_mode": "diagnostic_only",
        "checks_passed": ["cuda_visible", "torch_cuda", "model_loadable"],
        "checks_failed": [],
        "failure_reason": "",
    }


@pytest.fixture()
def exp355_simulated_data() -> dict[str, Any]:
    return {
        "experiment": 355,
        "status": "simulated",
        "inference_mode": "simulated",
        "honest_verdict": "blocked_simulated",
        "headline_result": {
            "honest_verdict": "blocked_simulated",
            "inference_mode": "simulated",
            "improvement_positive": False,
            "avg_accuracy_drop": 0.15,
            "avg_repair_improvement": 0.03,
        },
    }


@pytest.fixture()
def exp355_live_data() -> dict[str, Any]:
    """Fixture for a hypothetical live-GPU Exp 355 result."""
    return {
        "experiment": 355,
        "status": "success",
        "inference_mode": "live_gpu",
        "honest_verdict": "improvement_positive",
        "headline_result": {
            "honest_verdict": "improvement_positive",
            "inference_mode": "live_gpu",
            "improvement_positive": True,
            "avg_accuracy_drop": 0.10,
            "avg_repair_improvement": 0.08,
        },
    }


@pytest.fixture()
def exp359_synthetic_data() -> dict[str, Any]:
    return {
        "experiment": 359,
        "status": "success",
        "retrain_mode": "synthetic_only",
        "n_real_pairs": 5,
        "n_synthetic_pairs": 100,
        "before_auc": 0.5,
        "after_auc": 0.5,
        "auc_improvement": 0.0,
        "honest_verdict": "synthetic_only",
    }


@pytest.fixture()
def exp361_improved_data() -> dict[str, Any]:
    return {
        "experiment": 361,
        "status": "success",
        "inference_mode": "cpu_synthetic",
        "improved": True,
        "honest_verdict": "synthetic_only",
        "batch1_accuracy": 0.6,
        "batch4_accuracy": 0.72,
    }


# ---------------------------------------------------------------------------
# Tests for load_result_file
# ---------------------------------------------------------------------------


class TestLoadResultFile:
    def test_returns_none_for_none_path(self, tmp_path: Path) -> None:
        """None path means no JSON deliverable — must return None gracefully."""
        assert load_result_file(tmp_path, None) is None

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        """Missing result file returns None (not FileNotFoundError)."""
        assert load_result_file(tmp_path, "results/nonexistent.json") is None

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        """Valid JSON file is loaded and returned as dict."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        data = {"experiment": 352, "status": "success"}
        (results_dir / "exp_352.json").write_text(json.dumps(data))
        result = load_result_file(tmp_path, "results/exp_352.json")
        assert result == data

    def test_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        """Corrupted JSON file returns None (not JSONDecodeError propagated)."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "corrupt.json").write_text("{not valid json")
        assert load_result_file(tmp_path, "results/corrupt.json") is None


# ---------------------------------------------------------------------------
# Tests for load_all_results
# ---------------------------------------------------------------------------


class TestLoadAllResults:
    def test_returns_dict_keyed_by_experiment_id(self, tmp_path: Path) -> None:
        """Results dict is keyed by experiment ID integer."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "exp_352.json").write_text(json.dumps({"experiment": 352}))
        experiments = [
            {"id": 352, "result_file": "results/exp_352.json"},
            {"id": 351, "result_file": None},
        ]
        loaded = load_all_results(tmp_path, experiments)
        assert 352 in loaded
        assert 351 in loaded
        assert loaded[351] is None
        assert loaded[352] == {"experiment": 352}

    def test_handles_all_missing_files(self, tmp_path: Path) -> None:
        """All missing files produces all-None dict — no exceptions."""
        experiments = [
            {"id": 999, "result_file": "results/missing.json"},
        ]
        loaded = load_all_results(tmp_path, experiments)
        assert loaded[999] is None

    def test_handles_empty_experiment_list(self, tmp_path: Path) -> None:
        """Empty list produces empty dict."""
        assert load_all_results(tmp_path, []) == {}


# ---------------------------------------------------------------------------
# Tests for compute_statistics
# ---------------------------------------------------------------------------


class TestComputeStatistics:
    def test_skipped_experiments_excluded_from_mean(self) -> None:
        """Skipped experiment (Exp 356, wall_time_min=0) must not enter the mean."""
        experiments = [
            {"id": 355, "title": "A", "wall_time_min": 40, "status": "completed"},
            {"id": 356, "title": "B", "wall_time_min": 0, "status": "skipped"},
        ]
        stats = compute_statistics(experiments)
        # Only Exp 355 ran; mean should be 40, not 20
        assert stats["mean_time_per_exp_min"] == 40.0
        assert stats["n_experiments_ran"] == 1

    def test_total_wall_time_excludes_skipped(self) -> None:
        """Total wall time must not include the skipped experiment's 0 min."""
        experiments = [
            {"id": 352, "title": "A", "wall_time_min": 20, "status": "completed"},
            {"id": 353, "title": "B", "wall_time_min": 38, "status": "partial"},
            {"id": 356, "title": "C", "wall_time_min": 0, "status": "skipped"},
        ]
        stats = compute_statistics(experiments)
        assert stats["total_wall_time_min"] == 58

    def test_slowest_is_highest_wall_time(self) -> None:
        experiments = [
            {"id": 359, "title": "Slow", "wall_time_min": 51, "status": "completed", "note": ""},
            {"id": 352, "title": "Fast", "wall_time_min": 20, "status": "completed"},
        ]
        stats = compute_statistics(experiments)
        assert stats["slowest_experiment"]["id"] == 359
        assert stats["fastest_experiment"]["id"] == 352

    def test_counts_are_correct(self) -> None:
        experiments = [
            {"id": 351, "title": "A", "wall_time_min": 28, "status": "completed"},
            {"id": 353, "title": "B", "wall_time_min": 38, "status": "partial"},
            {"id": 356, "title": "C", "wall_time_min": 0, "status": "skipped"},
        ]
        stats = compute_statistics(experiments)
        assert stats["n_experiments_planned"] == 3
        assert stats["n_experiments_completed"] == 1
        assert stats["n_experiments_partial"] == 1
        assert stats["n_experiments_skipped"] == 1

    def test_total_hours_derived_from_minutes(self) -> None:
        experiments = [
            {"id": 352, "title": "A", "wall_time_min": 60, "status": "completed"},
        ]
        stats = compute_statistics(experiments)
        assert stats["total_wall_time_hours"] == 1.0

    def test_single_experiment_mean_equals_its_wall_time(self) -> None:
        experiments = [
            {"id": 359, "title": "A", "wall_time_min": 51, "status": "completed", "note": ""},
        ]
        stats = compute_statistics(experiments)
        assert stats["mean_time_per_exp_min"] == 51.0


# ---------------------------------------------------------------------------
# Tests for evaluate_success_criteria
# ---------------------------------------------------------------------------


class TestEvaluateSuccessCriteria:
    def _make_results(self, overrides: dict[int, Any]) -> dict[int, Any]:
        """Build a results dict from MILESTONE_EXPERIMENTS with optional overrides."""
        base: dict[int, Any] = {exp["id"]: None for exp in MILESTONE_EXPERIMENTS}
        base.update(overrides)
        return base

    def test_live_gpu_false_when_no_live_inference_mode(
        self, exp352_data: dict[str, Any]
    ) -> None:
        """live_gpu_confirmed=False when no experiment has inference_mode='live_gpu'."""
        results = self._make_results({352: exp352_data})
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        assert criteria["live_gpu_confirmed"]["value"] is False

    def test_live_gpu_true_when_any_experiment_has_live_mode(self) -> None:
        """live_gpu_confirmed=True when any result has inference_mode='live_gpu'."""
        results = self._make_results({355: {"inference_mode": "live_gpu"}})
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        assert criteria["live_gpu_confirmed"]["value"] is True

    def test_live_gpu_captures_is_live_capable_from_exp352(
        self, exp352_data: dict[str, Any]
    ) -> None:
        """Diagnostic is_live_capable from Exp 352 is reported even when live_gpu=False."""
        results = self._make_results({352: exp352_data})
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        assert criteria["live_gpu_confirmed"]["is_live_capable_diagnostic"] is True

    def test_adversarial_false_when_simulated(
        self, exp355_simulated_data: dict[str, Any]
    ) -> None:
        """adversarial_result_credible=False when honest_verdict=blocked_simulated."""
        results = self._make_results({355: exp355_simulated_data})
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        assert criteria["adversarial_result_credible"]["value"] is False

    def test_adversarial_true_when_improvement_positive(
        self, exp355_live_data: dict[str, Any]
    ) -> None:
        """adversarial_result_credible=True when headline_result.improvement_positive=True."""
        results = self._make_results({355: exp355_live_data})
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        assert criteria["adversarial_result_credible"]["value"] is True

    def test_adversarial_false_when_no_result(self) -> None:
        """adversarial_result_credible=False when Exp 355 result is missing."""
        results = self._make_results({355: None})
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        assert criteria["adversarial_result_credible"]["value"] is False

    def test_llm_extractor_false_when_exp356_skipped(self) -> None:
        """llm_extractor_beats_regex=False when Exp 356 is skipped (no script exists)."""
        results = self._make_results({356: None, 358: None})
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        assert criteria["llm_extractor_beats_regex"]["value"] is False
        assert criteria["llm_extractor_beats_regex"]["exp356_completed"] is False

    def test_llm_extractor_false_when_exp358_missing(self) -> None:
        """llm_extractor_beats_regex=False when Exp 358 result file is missing."""
        results = self._make_results({358: None})
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        assert criteria["llm_extractor_beats_regex"]["exp358_result_available"] is False

    def test_llm_extractor_true_when_detection_rate_positive(self) -> None:
        """llm_extractor_beats_regex=True when exp358 has llm_detection_rate > 0."""
        results = self._make_results({358: {"llm_detection_rate": 0.65}})
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        assert criteria["llm_extractor_beats_regex"]["value"] is True
        assert criteria["llm_extractor_beats_regex"]["detection_rate"] == 0.65

    def test_eorm_false_when_synthetic_only(
        self, exp359_synthetic_data: dict[str, Any]
    ) -> None:
        """eorm_retrained_on_real=False when retrain_mode=synthetic_only."""
        results = self._make_results({359: exp359_synthetic_data})
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        assert criteria["eorm_retrained_on_real"]["value"] is False
        assert criteria["eorm_retrained_on_real"]["retrain_mode"] == "synthetic_only"

    def test_eorm_true_when_real_data_mode(self) -> None:
        """eorm_retrained_on_real=True when retrain_mode=real_data_improvement."""
        results = self._make_results(
            {359: {"retrain_mode": "real_data_improvement", "n_real_pairs": 150}}
        )
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        assert criteria["eorm_retrained_on_real"]["value"] is True

    def test_eorm_false_when_missing(self) -> None:
        """eorm_retrained_on_real=False when Exp 359 result is missing."""
        results = self._make_results({359: None})
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        assert criteria["eorm_retrained_on_real"]["value"] is False

    def test_self_learning_true_when_improved(
        self, exp361_improved_data: dict[str, Any]
    ) -> None:
        """self_learning_improved=True when exp361 improved=True."""
        results = self._make_results({361: exp361_improved_data})
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        assert criteria["self_learning_improved"]["value"] is True
        assert criteria["self_learning_improved"]["batch1_accuracy"] == 0.6
        assert criteria["self_learning_improved"]["batch4_accuracy"] == 0.72

    def test_self_learning_false_when_missing(self) -> None:
        """self_learning_improved=False when Exp 361 result is missing."""
        results = self._make_results({361: None})
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        assert criteria["self_learning_improved"]["value"] is False

    def test_all_retros_false_when_exp351_missing(self) -> None:
        """all_retros_closed=False when Exp 351 result JSON is absent."""
        results = self._make_results({351: None})
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        assert criteria["all_retros_closed"]["value"] is False
        assert criteria["all_retros_closed"]["exp351_result_available"] is False

    def test_all_retros_true_when_exp351_confirms(self) -> None:
        """all_retros_closed=True when Exp 351 JSON reports all_closed=True."""
        results = self._make_results({351: {"all_closed": True}})
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        assert criteria["all_retros_closed"]["value"] is True

    def test_all_six_criteria_present(self) -> None:
        """All six success criteria keys are present in the output."""
        results = {exp["id"]: None for exp in MILESTONE_EXPERIMENTS}
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        expected_keys = {
            "live_gpu_confirmed",
            "adversarial_result_credible",
            "llm_extractor_beats_regex",
            "eorm_retrained_on_real",
            "self_learning_improved",
            "all_retros_closed",
        }
        assert set(criteria.keys()) == expected_keys

    def test_each_criterion_has_value_and_explanation(self) -> None:
        """Every criterion dict has 'value' (bool) and 'explanation' (str) fields."""
        results = {exp["id"]: None for exp in MILESTONE_EXPERIMENTS}
        criteria = evaluate_success_criteria(results, MILESTONE_EXPERIMENTS)
        for key, val in criteria.items():
            assert "value" in val, f"Criterion '{key}' missing 'value' key"
            assert "explanation" in val, f"Criterion '{key}' missing 'explanation' key"
            assert isinstance(val["value"], bool), f"Criterion '{key}' value must be bool"


# ---------------------------------------------------------------------------
# Tests for compute_retro (integration)
# ---------------------------------------------------------------------------


class TestComputeRetro:
    def test_returns_dict_with_required_schema(self, repo_root: Path) -> None:
        """compute_retro always returns a dict with the canonical schema field."""
        retro = compute_retro(repo_root)
        assert retro["schema"] == "carnot.operational_retro.v1"

    def test_milestone_matches_constant(self, repo_root: Path) -> None:
        assert compute_retro(repo_root)["milestone"] == MILESTONE

    def test_summary_has_required_keys(self, repo_root: Path) -> None:
        summary = compute_retro(repo_root)["summary"]
        for key in (
            "n_experiments_planned",
            "n_experiments_ran",
            "total_wall_time_min",
            "mean_time_per_exp_min",
        ):
            assert key in summary, f"Summary missing key: {key}"

    def test_success_criteria_present(self, repo_root: Path) -> None:
        retro = compute_retro(repo_root)
        assert "milestone_success_criteria" in retro

    def test_new_retro_items_present(self, repo_root: Path) -> None:
        retro = compute_retro(repo_root)
        assert "new_retro_items" in retro
        assert len(retro["new_retro_items"]) >= 3

    def test_top_improvements_present(self, repo_root: Path) -> None:
        retro = compute_retro(repo_root)
        assert "top_improvements" in retro
        assert len(retro["top_improvements"]) == 3

    def test_estimated_savings_pct_is_positive(self, repo_root: Path) -> None:
        retro = compute_retro(repo_root)
        assert retro["estimated_savings_next_milestone_pct"] > 0

    def test_key_findings_present(self, repo_root: Path) -> None:
        """Four key finding fields must be present and non-empty."""
        retro = compute_retro(repo_root)
        for key in (
            "key_finding_live_gpu",
            "key_finding_adversarial",
            "key_finding_llm_extractor",
            "key_finding_self_learning",
        ):
            assert key in retro
            assert len(retro[key]) > 20, f"Key finding '{key}' suspiciously short"

    def test_carry_forward_retro_status_has_expected_ids(self, repo_root: Path) -> None:
        """Carry-forward RETRO status should cover items from prior milestone."""
        retro = compute_retro(repo_root)
        carry = retro["carry_forward_retro_status"]
        for retro_id in ("RETRO-003", "RETRO-005", "RETRO-009", "RETRO-010", "RETRO-011"):
            assert retro_id in carry

    def test_cumulative_experiments_exceeds_prior_milestone(self, repo_root: Path) -> None:
        """Cumulative experiment count must be > 399 (prior milestone total)."""
        retro = compute_retro(repo_root)
        assert retro["cumulative_through_this_milestone"]["total_experiments"] > 399

    def test_with_live_result_files(
        self,
        repo_root: Path,
        exp352_data: dict[str, Any],
        exp355_simulated_data: dict[str, Any],
        exp359_synthetic_data: dict[str, Any],
        exp361_improved_data: dict[str, Any],
    ) -> None:
        """compute_retro correctly loads real-looking result files from a fixture dir."""
        results_dir = repo_root / "results"
        results_dir.mkdir(exist_ok=True)
        _write_result(results_dir, "experiment_352_live_gpu_diagnostic.json", exp352_data)
        _write_result(results_dir, "experiment_355_adversarial_gsm8k_benchmark.json", exp355_simulated_data)
        _write_result(results_dir, "experiment_359_eorm_real_retrain.json", exp359_synthetic_data)
        _write_result(results_dir, "experiment_361_self_learning_relay.json", exp361_improved_data)

        retro = compute_retro(repo_root)
        criteria = retro["milestone_success_criteria"]

        # With these fixtures: live_gpu=False, adversarial=False, eorm=False, self_learning=True
        assert criteria["live_gpu_confirmed"]["value"] is False
        assert criteria["live_gpu_confirmed"]["is_live_capable_diagnostic"] is True
        assert criteria["adversarial_result_credible"]["value"] is False
        assert criteria["eorm_retrained_on_real"]["value"] is False
        assert criteria["self_learning_improved"]["value"] is True


# ---------------------------------------------------------------------------
# Tests for RETRO items and improvements structural validation
# ---------------------------------------------------------------------------


class TestRetroItemsStructure:
    def test_new_retro_items_have_required_fields(self) -> None:
        """Every new RETRO item must have id, title, status, priority, description, fix."""
        for item in NEW_RETRO_ITEMS:
            for field in ("id", "title", "status", "priority", "description", "fix"):
                assert field in item, f"RETRO item missing field: {field}"

    def test_new_retro_ids_are_sequential_from_012(self) -> None:
        """New RETRO IDs must start at RETRO-012 (prior milestone opened RETRO-011)."""
        ids = [item["id"] for item in NEW_RETRO_ITEMS]
        assert "RETRO-012" in ids

    def test_new_retro_ids_are_unique(self) -> None:
        ids = [item["id"] for item in NEW_RETRO_ITEMS]
        assert len(ids) == len(set(ids))

    def test_retro_012_addresses_carnot_force_live(self) -> None:
        """RETRO-012 must address the CARNOT_FORCE_LIVE conductor gap."""
        retro_012 = next(i for i in NEW_RETRO_ITEMS if i["id"] == "RETRO-012")
        assert "CARNOT_FORCE_LIVE" in retro_012["description"]

    def test_retro_013_addresses_exp356(self) -> None:
        """RETRO-013 must address the missing Exp 356 (LLMExtractor)."""
        retro_013 = next(i for i in NEW_RETRO_ITEMS if i["id"] == "RETRO-013")
        assert "356" in retro_013["description"]

    def test_retro_014_addresses_missing_json(self) -> None:
        """RETRO-014 must address missing result JSON files."""
        retro_014 = next(i for i in NEW_RETRO_ITEMS if i["id"] == "RETRO-014")
        assert "JSON" in retro_014["description"] or "json" in retro_014["description"]


class TestTopImprovementsStructure:
    def test_exactly_three_improvements(self) -> None:
        assert len(TOP_IMPROVEMENTS) == 3

    def test_improvements_have_required_fields(self) -> None:
        for imp in TOP_IMPROVEMENTS:
            for field in ("rank", "action", "effort", "rationale"):
                assert field in imp, f"Improvement missing field: {field}"

    def test_improvements_ranked_1_2_3(self) -> None:
        ranks = [imp["rank"] for imp in TOP_IMPROVEMENTS]
        assert sorted(ranks) == [1, 2, 3]

    def test_rank_1_addresses_carnot_force_live(self) -> None:
        """The highest-priority improvement must address the CARNOT_FORCE_LIVE gap."""
        rank1 = next(i for i in TOP_IMPROVEMENTS if i["rank"] == 1)
        assert "CARNOT_FORCE_LIVE" in rank1["action"] or "RETRO-012" in rank1["action"]


# ---------------------------------------------------------------------------
# Tests for MILESTONE_EXPERIMENTS constant
# ---------------------------------------------------------------------------


class TestMilestoneExperimentsConstant:
    def test_covers_exps_351_to_362(self) -> None:
        """All 12 planned experiments (351–362) must be in the list."""
        ids = {exp["id"] for exp in MILESTONE_EXPERIMENTS}
        assert ids == set(range(351, 363))

    def test_exp356_has_status_skipped(self) -> None:
        exp356 = next(e for e in MILESTONE_EXPERIMENTS if e["id"] == 356)
        assert exp356["status"] == "skipped"

    def test_exp356_has_zero_wall_time(self) -> None:
        exp356 = next(e for e in MILESTONE_EXPERIMENTS if e["id"] == 356)
        assert exp356["wall_time_min"] == 0

    def test_all_experiments_have_id_title_status(self) -> None:
        for exp in MILESTONE_EXPERIMENTS:
            assert "id" in exp
            assert "title" in exp
            assert "status" in exp
            assert isinstance(exp["id"], int)

    def test_all_statuses_are_valid(self) -> None:
        valid = {"completed", "partial", "skipped"}
        for exp in MILESTONE_EXPERIMENTS:
            assert exp["status"] in valid, f"Exp {exp['id']} has invalid status: {exp['status']}"
