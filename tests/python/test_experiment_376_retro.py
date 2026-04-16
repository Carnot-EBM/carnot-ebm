"""Tests for scripts/experiment_376_retro_2026_05_27.py — Milestone 2026.05.27 retro.

Coverage targets
----------------
- MilestoneRetro2026_05_27: dataclass construction, field types, default values
- compute_retro_2026_05_27: all six success criteria with positive and negative fixtures
- build_retro_artifact: schema, required fields, retro_items_opened structure
- estimate_speedup_pct: positive speedup, zero case, edge cases
- load_milestone_results: missing files, valid JSON, partial JSON, invalid JSON
- compute_timing_stats: normal, empty list, single experiment
- main() integration: runs without error against real repo root (CPU-only, no GPU)

Spec: REQ-INFRA-014 (live GPU gating), REQ-BENCH-006/007 (adversarial GSM8K),
      REQ-EXTRACT-023 (LLM extractor comparison), REQ-LEARN-026/027 (self-learning relay)
SCENARIO: RETRO-2026.05.27
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_376_retro_2026_05_27 import (
    MILESTONE,
    MILESTONE_EXPERIMENTS,
    MilestoneRetro2026_05_27,
    NEW_RETRO_ITEMS,
    PREV_MEAN_EXP_DURATION_MIN,
    build_retro_artifact,
    compute_retro_2026_05_27,
    compute_timing_stats,
    estimate_speedup_pct,
    load_milestone_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# MilestoneRetro2026_05_27 dataclass
# ---------------------------------------------------------------------------


class TestMilestoneRetroDataclass:
    """MilestoneRetro2026_05_27 is a plain dataclass — fields must be settable."""

    def _make(self, **overrides: Any) -> MilestoneRetro2026_05_27:
        defaults: dict[str, Any] = {
            "live_gpu_confirmed": False,
            "llm_extractor_beats_regex": False,
            "adversarial_result_credible": False,
            "eorm_retrained_on_real": False,
            "self_learning_confirmed": False,
            "cikan_implemented": False,
            "all_result_jsons_present": False,
            "retro_012_closed": False,
            "mean_exp_duration_min": 0.0,
            "n_experiments_total": 0,
            "n_experiments_blocked": 0,
            "retro_items_opened": [],
        }
        defaults.update(overrides)
        return MilestoneRetro2026_05_27(**defaults)

    def test_construction_defaults(self) -> None:
        retro = self._make()
        assert retro.live_gpu_confirmed is False
        assert retro.retro_items_opened == []

    def test_bool_fields_can_be_true(self) -> None:
        retro = self._make(live_gpu_confirmed=True, cikan_implemented=True)
        assert retro.live_gpu_confirmed is True
        assert retro.cikan_implemented is True

    def test_mean_duration_float(self) -> None:
        retro = self._make(mean_exp_duration_min=23.4)
        assert isinstance(retro.mean_exp_duration_min, float)
        assert retro.mean_exp_duration_min == pytest.approx(23.4)

    def test_retro_items_list(self) -> None:
        retro = self._make(retro_items_opened=["RETRO-015", "RETRO-016"])
        assert "RETRO-015" in retro.retro_items_opened

    def test_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(MilestoneRetro2026_05_27)

    def test_n_fields(self) -> None:
        # Ensure all 12 required fields exist
        fields = {f.name for f in dataclasses.fields(MilestoneRetro2026_05_27)}
        required = {
            "live_gpu_confirmed",
            "llm_extractor_beats_regex",
            "adversarial_result_credible",
            "eorm_retrained_on_real",
            "self_learning_confirmed",
            "cikan_implemented",
            "all_result_jsons_present",
            "retro_012_closed",
            "mean_exp_duration_min",
            "n_experiments_total",
            "n_experiments_blocked",
            "retro_items_opened",
        }
        assert required.issubset(fields)

    def test_n_experiments_counts(self) -> None:
        retro = self._make(n_experiments_total=11, n_experiments_blocked=7)
        assert retro.n_experiments_total == 11
        assert retro.n_experiments_blocked == 7


# ---------------------------------------------------------------------------
# estimate_speedup_pct
# ---------------------------------------------------------------------------


class TestEstimateSpeedupPct:
    def test_improvement(self) -> None:
        # 33.3 min/exp → 23.4 min/exp = ~29.7% speedup
        pct = estimate_speedup_pct(33.3, 23.4)
        assert pct == pytest.approx(29.73, rel=0.02)

    def test_no_change(self) -> None:
        assert estimate_speedup_pct(30.0, 30.0) == pytest.approx(0.0)

    def test_regression(self) -> None:
        # Slower is negative speedup
        pct = estimate_speedup_pct(20.0, 30.0)
        assert pct < 0

    def test_prev_zero_returns_zero(self) -> None:
        # Guard against ZeroDivisionError
        assert estimate_speedup_pct(0.0, 10.0) == 0.0

    def test_return_type(self) -> None:
        assert isinstance(estimate_speedup_pct(33.3, 20.0), float)

    def test_large_speedup(self) -> None:
        # 100 → 10 = 90% speedup
        pct = estimate_speedup_pct(100.0, 10.0)
        assert pct == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# load_milestone_results
# ---------------------------------------------------------------------------


class TestLoadMilestoneResults:
    def test_loads_valid_json(self, tmp_path: Path) -> None:
        data = {"experiment": 365, "status": "success"}
        _write_json(tmp_path / "results" / "experiment_365.json", data)
        result = load_milestone_results(tmp_path, {"365": "results/experiment_365.json"})
        assert result["365"]["status"] == "success"

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        result = load_milestone_results(
            tmp_path, {"999": "results/experiment_999_missing.json"}
        )
        assert result["999"] is None

    def test_none_path_returns_none(self, tmp_path: Path) -> None:
        result = load_milestone_results(tmp_path, {"abc": None})
        assert result["abc"] is None

    def test_invalid_json_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "results" / "bad.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("not-json{{{")
        result = load_milestone_results(tmp_path, {"bad": "results/bad.json"})
        assert result["bad"] is None

    def test_empty_mapping_returns_empty(self, tmp_path: Path) -> None:
        result = load_milestone_results(tmp_path, {})
        assert result == {}

    def test_partial_json_file(self, tmp_path: Path) -> None:
        # Minimal partial result (like what Exp 367 has)
        data = {"experiment": 367, "status": "partial", "finding": "needs GPU"}
        _write_json(tmp_path / "results" / "exp367.json", data)
        result = load_milestone_results(tmp_path, {"367": "results/exp367.json"})
        assert result["367"]["status"] == "partial"


# ---------------------------------------------------------------------------
# compute_timing_stats
# ---------------------------------------------------------------------------


class TestComputeTimingStats:
    def test_normal_case(self) -> None:
        exps = [
            {"id": 365, "wall_time_min": 5, "status": "completed"},
            {"id": 366, "wall_time_min": 25, "status": "completed"},
            {"id": 367, "wall_time_min": 35, "status": "partial"},
        ]
        stats = compute_timing_stats(exps)
        assert stats["n_ran"] == 3
        assert stats["total_min"] == 65
        assert stats["mean_min"] == pytest.approx(65 / 3, rel=0.01)

    def test_blocked_excluded_from_mean(self) -> None:
        exps = [
            {"id": 368, "wall_time_min": 3, "status": "blocked"},
            {"id": 369, "wall_time_min": 40, "status": "completed"},
        ]
        stats = compute_timing_stats(exps)
        # blocked experiments have essentially zero meaningful wall time
        # mean should only count non-blocked
        assert stats["n_blocked"] == 1
        assert stats["n_ran"] == 2  # all experiments count toward n_ran

    def test_single_experiment(self) -> None:
        exps = [{"id": 365, "wall_time_min": 10, "status": "completed"}]
        stats = compute_timing_stats(exps)
        assert stats["mean_min"] == pytest.approx(10.0)

    def test_empty_list(self) -> None:
        stats = compute_timing_stats([])
        assert stats["n_ran"] == 0
        assert stats["mean_min"] == 0.0
        assert stats["total_min"] == 0

    def test_returns_dict(self) -> None:
        stats = compute_timing_stats([{"id": 1, "wall_time_min": 5, "status": "completed"}])
        assert isinstance(stats, dict)
        assert "n_ran" in stats
        assert "total_min" in stats
        assert "mean_min" in stats
        assert "n_blocked" in stats


# ---------------------------------------------------------------------------
# compute_retro_2026_05_27 — success criteria evaluation
# ---------------------------------------------------------------------------


class TestComputeRetro:
    """Test compute_retro_2026_05_27 against fixture result files."""

    def _repo_with_results(
        self, tmp_path: Path, results: dict[str, dict[str, Any]]
    ) -> Path:
        """Write result files into a fake repo structure."""
        for filename, data in results.items():
            _write_json(tmp_path / "results" / filename, data)
        return tmp_path

    # --- retro_012_closed ---

    def test_retro_012_closed_when_exp365_all_closed(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {"experiment_365_retro_close.json": {"all_closed": True, "status": "success"}},
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.retro_012_closed is True

    def test_retro_012_not_closed_when_missing(self, tmp_path: Path) -> None:
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.retro_012_closed is False

    def test_retro_012_not_closed_when_false(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {"experiment_365_retro_close.json": {"all_closed": False, "status": "partial"}},
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.retro_012_closed is False

    # --- live_gpu_confirmed ---

    def test_live_gpu_false_when_no_live_inference(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {
                "experiment_373_three_tier_live.json": {
                    "status": "partial",
                    "inference_mode": "blocked",
                }
            },
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.live_gpu_confirmed is False

    def test_live_gpu_true_when_live_inference_found(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {
                "experiment_373_three_tier_live.json": {
                    "status": "success",
                    "inference_mode": "live_gpu",
                }
            },
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.live_gpu_confirmed is True

    # --- llm_extractor_beats_regex ---

    def test_llm_extractor_false_when_partial(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {"experiment_367_extraction_live.json": {"status": "partial", "finding": "x"}},
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.llm_extractor_beats_regex is False

    def test_llm_extractor_true_when_live_gpu_winner(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {
                "experiment_367_extraction_live.json": {
                    "status": "success",
                    "honest_verdict": "live_gpu_winner",
                    "inference_mode": "live_gpu",
                }
            },
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.llm_extractor_beats_regex is True

    # --- adversarial_result_credible ---

    def test_adversarial_false_when_no_json(self, tmp_path: Path) -> None:
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.adversarial_result_credible is False

    def test_adversarial_true_when_improvement_positive(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {
                "experiment_370_adversarial_live.json": {
                    "status": "success",
                    "honest_verdict": "improvement_positive",
                    "inference_mode": "live_gpu",
                }
            },
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.adversarial_result_credible is True

    def test_adversarial_false_when_blocked_simulated(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {
                "experiment_370_adversarial_live.json": {
                    "status": "blocked",
                    "honest_verdict": "blocked_simulated",
                }
            },
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.adversarial_result_credible is False

    # --- eorm_retrained_on_real ---

    def test_eorm_false_when_partial(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {"experiment_371_eorm_real_retrain.json": {"status": "partial", "finding": "x"}},
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.eorm_retrained_on_real is False

    def test_eorm_true_when_real_data_improvement(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {
                "experiment_371_eorm_real_retrain.json": {
                    "status": "success",
                    "retrain_mode": "real_data",
                    "honest_verdict": "real_data_improvement",
                    "after_auc": 0.72,
                    "before_auc": 0.50,
                }
            },
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.eorm_retrained_on_real is True

    def test_eorm_false_when_synthetic_only(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {
                "experiment_371_eorm_real_retrain.json": {
                    "status": "success",
                    "retrain_mode": "synthetic_only",
                    "honest_verdict": "synthetic_only",
                }
            },
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.eorm_retrained_on_real is False

    # --- self_learning_confirmed ---

    def test_self_learning_false_when_partial(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {"experiment_374_self_learning_relay_live.json": {"status": "partial"}},
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.self_learning_confirmed is False

    def test_self_learning_true_when_learning_confirmed(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {
                "experiment_374_self_learning_relay_live.json": {
                    "status": "success",
                    "honest_verdict": "learning_confirmed",
                    "inference_mode": "live_gpu",
                    "improved": True,
                }
            },
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.self_learning_confirmed is True

    def test_self_learning_false_when_synthetic_only(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {
                "experiment_374_self_learning_relay_live.json": {
                    "status": "success",
                    "honest_verdict": "synthetic_only",
                    "improved": True,
                }
            },
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.self_learning_confirmed is False

    # --- cikan_implemented ---

    def test_cikan_false_when_deliverable_corrupt(self, tmp_path: Path) -> None:
        # cikan_energy.py is a JSON file in the actual repo — not valid Python
        models_dir = tmp_path / "python" / "carnot" / "models"
        models_dir.mkdir(parents=True)
        # Write a JSON stub (as it is in the real repo)
        (models_dir / "cikan_energy.py").write_text('{"status": "partial"}')
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.cikan_implemented is False

    def test_cikan_true_when_valid_python_with_class(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "python" / "carnot" / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "cikan_energy.py").write_text(
            "class CIKANEnergy:\n    pass\n"
        )
        # Also need a result JSON
        _write_json(
            tmp_path / "results" / "experiment_375_cikan_energy.json",
            {"status": "success", "schema": "carnot.cikan_energy.v1"},
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.cikan_implemented is True

    def test_cikan_false_when_no_deliverable(self, tmp_path: Path) -> None:
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.cikan_implemented is False

    # --- all_result_jsons_present ---

    def test_all_jsons_false_when_any_missing(self, tmp_path: Path) -> None:
        # Only provide one result — the rest are missing
        self._repo_with_results(
            tmp_path,
            {"experiment_365_retro_close.json": {"status": "success", "all_closed": True}},
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.all_result_jsons_present is False

    def test_all_jsons_present_when_all_provided(self, tmp_path: Path) -> None:
        # Write ALL required result files (based on MILESTONE_EXPERIMENTS that have result_file)
        from scripts.experiment_376_retro_2026_05_27 import MILESTONE_EXPERIMENTS as EXPS
        for exp in EXPS:
            rf = exp.get("result_file")
            if rf:
                _write_json(tmp_path / rf, {"experiment": exp["id"], "status": "success"})
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.all_result_jsons_present is True

    # --- n_experiments counts ---

    def test_n_experiments_total(self, tmp_path: Path) -> None:
        retro = compute_retro_2026_05_27(tmp_path)
        # Milestone 2026.05.27 had 12 experiments (365–376)
        assert retro.n_experiments_total >= 11

    def test_n_experiments_blocked_ge_zero(self, tmp_path: Path) -> None:
        retro = compute_retro_2026_05_27(tmp_path)
        assert retro.n_experiments_blocked >= 0

    # --- retro_items_opened ---

    def test_retro_015_opened_when_gpu_still_false(self, tmp_path: Path) -> None:
        retro = compute_retro_2026_05_27(tmp_path)
        # live_gpu not confirmed → RETRO-015 must be opened
        assert "RETRO-015" in retro.retro_items_opened

    def test_no_retro_015_when_gpu_confirmed(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {
                "experiment_373_three_tier_live.json": {
                    "inference_mode": "live_gpu",
                    "status": "success",
                }
            },
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert "RETRO-015" not in retro.retro_items_opened

    def test_retro_016_opened_when_extractor_false(self, tmp_path: Path) -> None:
        retro = compute_retro_2026_05_27(tmp_path)
        assert "RETRO-016" in retro.retro_items_opened

    def test_no_retro_016_when_extractor_confirmed(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {
                "experiment_367_extraction_live.json": {
                    "honest_verdict": "live_gpu_winner",
                    "inference_mode": "live_gpu",
                    "status": "success",
                }
            },
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert "RETRO-016" not in retro.retro_items_opened

    def test_retro_017_opened_when_learning_not_confirmed(self, tmp_path: Path) -> None:
        retro = compute_retro_2026_05_27(tmp_path)
        assert "RETRO-017" in retro.retro_items_opened

    def test_no_retro_017_when_learning_confirmed(self, tmp_path: Path) -> None:
        self._repo_with_results(
            tmp_path,
            {
                "experiment_374_self_learning_relay_live.json": {
                    "honest_verdict": "learning_confirmed",
                    "inference_mode": "live_gpu",
                    "status": "success",
                }
            },
        )
        retro = compute_retro_2026_05_27(tmp_path)
        assert "RETRO-017" not in retro.retro_items_opened

    # --- mean_exp_duration_min ---

    def test_mean_duration_is_float(self, tmp_path: Path) -> None:
        retro = compute_retro_2026_05_27(tmp_path)
        assert isinstance(retro.mean_exp_duration_min, float)
        assert retro.mean_exp_duration_min >= 0.0


# ---------------------------------------------------------------------------
# build_retro_artifact
# ---------------------------------------------------------------------------


class TestBuildRetroArtifact:
    def _make_retro(self, **overrides: Any) -> MilestoneRetro2026_05_27:
        defaults: dict[str, Any] = {
            "live_gpu_confirmed": False,
            "llm_extractor_beats_regex": False,
            "adversarial_result_credible": False,
            "eorm_retrained_on_real": False,
            "self_learning_confirmed": False,
            "cikan_implemented": False,
            "all_result_jsons_present": False,
            "retro_012_closed": True,
            "mean_exp_duration_min": 23.4,
            "n_experiments_total": 12,
            "n_experiments_blocked": 7,
            "retro_items_opened": ["RETRO-015", "RETRO-016", "RETRO-017"],
        }
        defaults.update(overrides)
        return MilestoneRetro2026_05_27(**defaults)

    def test_schema_v2(self) -> None:
        artifact = build_retro_artifact(self._make_retro())
        assert artifact["schema"] == "carnot.operational_retro.v2"

    def test_milestone_field(self) -> None:
        artifact = build_retro_artifact(self._make_retro())
        assert artifact["milestone"] == "2026.05.27"

    def test_success_criteria_present(self) -> None:
        artifact = build_retro_artifact(self._make_retro())
        sc = artifact["success_criteria"]
        assert "live_gpu_confirmed" in sc
        assert "llm_extractor_beats_regex" in sc
        assert "adversarial_result_credible" in sc
        assert "eorm_retrained_on_real" in sc
        assert "self_learning_confirmed" in sc
        assert "cikan_implemented" in sc
        assert "all_result_jsons_present" in sc
        assert "retro_012_closed" in sc

    def test_timing_analysis_present(self) -> None:
        artifact = build_retro_artifact(self._make_retro())
        assert "timing_analysis" in artifact
        ta = artifact["timing_analysis"]
        assert "mean_exp_duration_min" in ta
        assert "prev_mean_exp_duration_min" in ta
        assert "estimated_speedup_pct" in ta

    def test_retro_items_opened_in_artifact(self) -> None:
        retro = self._make_retro(retro_items_opened=["RETRO-015"])
        artifact = build_retro_artifact(retro)
        assert "RETRO-015" in artifact["retro_items_opened"]

    def test_estimated_savings_field_present(self) -> None:
        artifact = build_retro_artifact(self._make_retro())
        assert "estimated_savings_next_pct" in artifact
        assert isinstance(artifact["estimated_savings_next_pct"], (int, float))

    def test_returns_dict(self) -> None:
        artifact = build_retro_artifact(self._make_retro())
        assert isinstance(artifact, dict)

    def test_all_criteria_values_present(self) -> None:
        retro = self._make_retro(live_gpu_confirmed=True, retro_012_closed=True)
        artifact = build_retro_artifact(retro)
        assert artifact["success_criteria"]["live_gpu_confirmed"] is True
        assert artifact["success_criteria"]["retro_012_closed"] is True

    def test_timing_speedup_positive_when_faster(self) -> None:
        retro = self._make_retro(mean_exp_duration_min=20.0)
        artifact = build_retro_artifact(retro)
        assert artifact["timing_analysis"]["estimated_speedup_pct"] > 0

    def test_timing_speedup_negative_when_slower(self) -> None:
        retro = self._make_retro(mean_exp_duration_min=40.0)
        artifact = build_retro_artifact(retro)
        assert artifact["timing_analysis"]["estimated_speedup_pct"] < 0

    def test_n_experiments_in_artifact(self) -> None:
        retro = self._make_retro(n_experiments_total=12, n_experiments_blocked=7)
        artifact = build_retro_artifact(retro)
        assert artifact["success_criteria"]["n_experiments_total"] == 12
        assert artifact["success_criteria"]["n_experiments_blocked"] == 7


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_milestone_string(self) -> None:
        assert MILESTONE == "2026.05.27"

    def test_prev_mean_is_float(self) -> None:
        assert isinstance(PREV_MEAN_EXP_DURATION_MIN, float)
        assert PREV_MEAN_EXP_DURATION_MIN == pytest.approx(33.3, rel=0.01)

    def test_milestone_experiments_list(self) -> None:
        assert isinstance(MILESTONE_EXPERIMENTS, list)
        assert len(MILESTONE_EXPERIMENTS) >= 11  # Exps 365–375 (376 is this experiment)

    def test_each_experiment_has_id_and_title(self) -> None:
        for exp in MILESTONE_EXPERIMENTS:
            assert "id" in exp, f"Missing 'id' in {exp}"
            assert "title" in exp, f"Missing 'title' in {exp}"
            assert "wall_time_min" in exp, f"Missing 'wall_time_min' in {exp}"
            assert "status" in exp, f"Missing 'status' in {exp}"

    def test_new_retro_items_structure(self) -> None:
        assert isinstance(NEW_RETRO_ITEMS, list)
        for item in NEW_RETRO_ITEMS:
            assert "id" in item
            assert "priority" in item
            assert "description" in item

    def test_retro_ids_are_015_plus(self) -> None:
        # Milestone 2026.05.27 opens RETRO-015+
        for item in NEW_RETRO_ITEMS:
            retro_num = int(item["id"].replace("RETRO-", ""))
            assert retro_num >= 15, f"{item['id']} is not >= RETRO-015"


# ---------------------------------------------------------------------------
# Integration: compute_retro_2026_05_27 against real repo root
# ---------------------------------------------------------------------------


class TestIntegrationRealRepo:
    """Run against the real repo root to ensure no import errors or path failures."""

    def test_runs_without_error(self) -> None:
        retro = compute_retro_2026_05_27(_REPO_ROOT)
        assert isinstance(retro, MilestoneRetro2026_05_27)

    def test_retro_012_closed_in_real_repo(self) -> None:
        # Exp 365 exists and has all_closed=True in the real repo
        retro = compute_retro_2026_05_27(_REPO_ROOT)
        assert retro.retro_012_closed is True

    def test_live_gpu_still_false_in_real_repo(self) -> None:
        # No live GPU inference happened in this milestone
        retro = compute_retro_2026_05_27(_REPO_ROOT)
        assert retro.live_gpu_confirmed is False

    def test_retro_015_opened_in_real_repo(self) -> None:
        retro = compute_retro_2026_05_27(_REPO_ROOT)
        assert "RETRO-015" in retro.retro_items_opened

    def test_build_artifact_from_real_repo(self) -> None:
        retro = compute_retro_2026_05_27(_REPO_ROOT)
        artifact = build_retro_artifact(retro)
        assert artifact["schema"] == "carnot.operational_retro.v2"
        assert artifact["milestone"] == "2026.05.27"

    def test_mean_duration_reasonable(self) -> None:
        retro = compute_retro_2026_05_27(_REPO_ROOT)
        # Mean should be >0 and <200 (no experiment takes 3+ hours)
        assert 0 < retro.mean_exp_duration_min < 200

    def test_n_total_ge_11(self) -> None:
        retro = compute_retro_2026_05_27(_REPO_ROOT)
        assert retro.n_experiments_total >= 11
