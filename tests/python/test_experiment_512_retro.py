"""Tests for Exp 512 — Milestone 2026.04.38 Retrospective.

Covers the helper functions added in scripts/experiment_512_retro_2026_04_38.py:
- _load_results: loads JSON files from disk, handles missing/corrupt files
- _count_deferred_to_gpu: classifies GPU-blocked experiments
- _assess_credibility_milestones: reads the five credibility booleans
- _assess_retro_closures: reads all RETRO closure booleans
- _compute_wall_time_stats: computes total and average wall time
- _build_headline_results: extracts Exp 502/503/504 benchmark results
- _build_open_retro_items: enumerates carry-forward open items
- _build_new_retro_items: generates new RETRO items for .39
- _build_meta_reflection: composes meta-reflection section

Spec: REQ-RETRO-038, SCENARIO-RETRO-038
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so scripts/ and carnot/ are importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_512_retro_2026_04_38 as retro


# ---------------------------------------------------------------------------
# _load_results
# ---------------------------------------------------------------------------

class TestLoadResults:
    """_load_results handles present, missing, and corrupt files correctly."""

    def test_loads_present_file(self, tmp_path):
        # Write a minimal experiment result JSON and confirm it loads
        exp_dir = tmp_path
        data = {"experiment": 500, "status": "success", "is_within_budget": True}
        p = exp_dir / "results" / "experiment_500_gemma4_int4_quantized.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))

        # Patch EXP_RESULT_PATHS to only include Exp 500 for this test
        orig = retro.EXP_RESULT_PATHS
        retro.EXP_RESULT_PATHS = {500: "results/experiment_500_gemma4_int4_quantized.json"}
        try:
            loaded = retro._load_results(exp_dir)
        finally:
            retro.EXP_RESULT_PATHS = orig

        assert 500 in loaded
        assert loaded[500]["is_within_budget"] is True

    def test_missing_file_excluded_with_warning(self, tmp_path, caplog):
        import logging
        orig = retro.EXP_RESULT_PATHS
        retro.EXP_RESULT_PATHS = {500: "results/experiment_500_MISSING.json"}
        try:
            with caplog.at_level(logging.WARNING):
                loaded = retro._load_results(tmp_path)
        finally:
            retro.EXP_RESULT_PATHS = orig
        assert 500 not in loaded
        assert any("missing" in r.message.lower() for r in caplog.records)

    def test_corrupt_json_excluded_with_warning(self, tmp_path, caplog):
        import logging
        p = tmp_path / "results" / "experiment_500_bad.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not valid json")
        orig = retro.EXP_RESULT_PATHS
        retro.EXP_RESULT_PATHS = {500: "results/experiment_500_bad.json"}
        try:
            with caplog.at_level(logging.WARNING):
                loaded = retro._load_results(tmp_path)
        finally:
            retro.EXP_RESULT_PATHS = orig
        assert 500 not in loaded

    def test_empty_path_mapping_returns_empty(self, tmp_path):
        orig = retro.EXP_RESULT_PATHS
        retro.EXP_RESULT_PATHS = {}
        try:
            loaded = retro._load_results(tmp_path)
        finally:
            retro.EXP_RESULT_PATHS = orig
        assert loaded == {}


# ---------------------------------------------------------------------------
# _count_deferred_to_gpu
# ---------------------------------------------------------------------------

class TestCountDeferredToGpu:
    """_count_deferred_to_gpu identifies GPU-blocked experiments."""

    def test_deferred_retro_033_v6_verdict_detected(self):
        results = {502: {"honest_verdict": "deferred_retro_033_v6", "status": "gpu_required"}}
        n, ids = retro._count_deferred_to_gpu(results)
        assert n == 1
        assert 502 in ids

    def test_gpu_required_status_with_cuda_oom_blocked_reason(self):
        results = {
            503: {
                "honest_verdict": "gpu_required",
                "status": "blocked",
                "blocked_reason": "Qwen load failed: CUDA error: out of memory",
            }
        }
        n, ids = retro._count_deferred_to_gpu(results)
        assert n == 1
        assert 503 in ids

    def test_successful_experiment_not_counted(self):
        results = {
            500: {"honest_verdict": "retro_048_unblocked", "status": "success"},
            508: {"honest_verdict": "kaem_advantage_found", "status": "success"},
        }
        n, ids = retro._count_deferred_to_gpu(results)
        assert n == 0
        assert ids == []

    def test_multiple_deferred_sorted(self):
        results = {
            504: {"honest_verdict": "gpu_required", "status": "gpu_required"},
            502: {"honest_verdict": "deferred_retro_033_v6", "status": "gpu_required"},
        }
        n, ids = retro._count_deferred_to_gpu(results)
        assert n == 2
        assert ids == [502, 504]

    def test_empty_results_returns_zero(self):
        n, ids = retro._count_deferred_to_gpu({})
        assert n == 0
        assert ids == []


# ---------------------------------------------------------------------------
# _assess_credibility_milestones
# ---------------------------------------------------------------------------

class TestAssessCredibilityMilestones:
    """_assess_credibility_milestones reads fields from the right experiment JSONs."""

    def test_all_false_when_results_empty(self):
        cred = retro._assess_credibility_milestones({})
        assert cred["retro_048_resolved"] is False
        assert cred["retro_033_closed"] is False
        assert cred["retro_038_closed"] is False
        assert cred["retro_039_confirmed"] is False
        assert cred["gpu1_utilization_improved"] is False
        assert cred["fr11_live_relay"] is False
        assert cred["npu_status"] == "unknown"

    def test_retro_048_from_exp_500_is_within_budget(self):
        results = {500: {"is_within_budget": True, "honest_verdict": "retro_048_unblocked"}}
        cred = retro._assess_credibility_milestones(results)
        assert cred["retro_048_resolved"] is True

    def test_retro_033_from_exp_502(self):
        results = {502: {"retro_033_closed": False, "status": "gpu_required"}}
        cred = retro._assess_credibility_milestones(results)
        assert cred["retro_033_closed"] is False

    def test_gpu1_utilization_improved_requires_patched_gt_0(self):
        results = {505: {"n_scripts_patched": 3, "honest_verdict": "sweep_complete"}}
        cred = retro._assess_credibility_milestones(results)
        assert cred["gpu1_utilization_improved"] is True

    def test_gpu1_utilization_not_improved_when_patched_eq_0(self):
        results = {505: {"n_scripts_patched": 0, "honest_verdict": "sweep_complete"}}
        cred = retro._assess_credibility_milestones(results)
        assert cred["gpu1_utilization_improved"] is False

    def test_npu_status_from_exp_511_verdict(self):
        results = {511: {"honest_verdict": "npu_not_available", "npu_available": False}}
        cred = retro._assess_credibility_milestones(results)
        assert cred["npu_status"] == "npu_not_available"

    def test_fr11_live_relay_from_exp_510(self):
        results = {510: {"fr11_relay_confirmed": True}}
        cred = retro._assess_credibility_milestones(results)
        assert cred["fr11_live_relay"] is True


# ---------------------------------------------------------------------------
# _assess_retro_closures
# ---------------------------------------------------------------------------

class TestAssessRetroClosure:
    """_assess_retro_closures reads correct fields from the right experiments."""

    def test_all_false_when_results_empty(self):
        closures = retro._assess_retro_closures({})
        assert all(v is False for v in closures.values())

    def test_retro_031_from_exp_508(self):
        closures = retro._assess_retro_closures({508: {"retro_031_closed": True}})
        assert closures["retro_031_closed"] is True

    def test_retro_050_from_exp_509(self):
        closures = retro._assess_retro_closures({509: {"retro_050_closed": True}})
        assert closures["retro_050_closed"] is True

    def test_retro_048_from_exp_500_is_within_budget(self):
        closures = retro._assess_retro_closures({500: {"is_within_budget": True}})
        assert closures["retro_048_resolved"] is True

    def test_all_closed(self):
        results = {
            500: {"is_within_budget": True},
            502: {"retro_033_closed": True},
            503: {"retro_038_closed": True},
            504: {"retro_039_confirmed": True},
            507: {"retro_049_closed": True},
            508: {"retro_031_closed": True},
            509: {"retro_050_closed": True},
        }
        closures = retro._assess_retro_closures(results)
        assert all(v is True for v in closures.values())


# ---------------------------------------------------------------------------
# _compute_wall_time_stats
# ---------------------------------------------------------------------------

class TestComputeWallTimeStats:
    """_compute_wall_time_stats correctly aggregates duration_s from results."""

    def test_total_and_average(self):
        results = {
            500: {"duration_s": 60.0},  # 1 min
            501: {"duration_s": 120.0},  # 2 min
        }
        stats = retro._compute_wall_time_stats(results)
        assert stats["total_wall_time_minutes"] == pytest.approx(3.0, abs=0.01)
        assert stats["average_minutes_per_experiment"] == pytest.approx(1.5, abs=0.01)

    def test_empty_results_gives_zeros(self):
        stats = retro._compute_wall_time_stats({})
        assert stats["total_wall_time_minutes"] == 0.0
        assert stats["average_minutes_per_experiment"] == 0.0

    def test_missing_duration_s_defaults_to_zero(self):
        results = {500: {"status": "success"}}  # no duration_s key
        stats = retro._compute_wall_time_stats(results)
        assert stats["total_wall_time_minutes"] == 0.0

    def test_per_exp_duration_minutes_keys_are_strings(self):
        results = {502: {"duration_s": 5.658}}
        stats = retro._compute_wall_time_stats(results)
        assert "502" in stats["per_exp_duration_minutes"]


# ---------------------------------------------------------------------------
# _build_headline_results
# ---------------------------------------------------------------------------

class TestBuildHeadlineResults:
    """_build_headline_results extracts structured data from Exps 502/503/504."""

    def test_missing_results_give_missing_status(self):
        hr = retro._build_headline_results({})
        assert hr["live_100q_v6"]["status"] == "missing"
        assert hr["live_200q_v4"]["status"] == "missing"
        assert hr["adversarial_v4"]["status"] == "missing"

    def test_reads_exp_502_fields(self):
        results = {
            502: {
                "status": "gpu_required",
                "honest_verdict": "deferred_retro_033_v6",
                "retro_033_closed": False,
                "gemma4_quantized": True,
                "vram_forecasts": [{"is_feasible": True}, {"is_feasible": True}],
            }
        }
        hr = retro._build_headline_results(results)
        assert hr["live_100q_v6"]["status"] == "gpu_required"
        assert hr["live_100q_v6"]["gemma4_quantized"] is True
        assert hr["live_100q_v6"]["vram_forecast_feasible"] is True

    def test_vram_forecast_false_when_any_infeasible(self):
        results = {
            502: {
                "status": "gpu_required",
                "vram_forecasts": [{"is_feasible": True}, {"is_feasible": False}],
            }
        }
        hr = retro._build_headline_results(results)
        assert hr["live_100q_v6"]["vram_forecast_feasible"] is False

    def test_vram_forecast_none_when_no_forecasts(self):
        results = {502: {"status": "gpu_required", "vram_forecasts": []}}
        hr = retro._build_headline_results(results)
        assert hr["live_100q_v6"]["vram_forecast_feasible"] is None

    def test_blocked_reason_truncated_to_120_chars(self):
        long_reason = "A" * 200
        results = {503: {"blocked_reason": long_reason}}
        hr = retro._build_headline_results(results)
        assert len(hr["live_200q_v4"]["blocked_reason_summary"]) == 120


# ---------------------------------------------------------------------------
# _build_open_retro_items
# ---------------------------------------------------------------------------

class TestBuildOpenRetroItems:
    """_build_open_retro_items enumerates carry-forward open items."""

    def _all_closed(self) -> dict:
        return {
            "retro_031_closed": True,
            "retro_033_closed": True,
            "retro_038_closed": True,
            "retro_039_confirmed": True,
            "retro_048_resolved": True,
            "retro_049_closed": True,
            "retro_050_closed": True,
        }

    def _all_open(self) -> dict:
        return {k: False for k in self._all_closed()}

    def test_no_items_when_all_closed(self):
        items = retro._build_open_retro_items(self._all_closed(), {})
        assert items == []

    def test_retro_033_item_present_when_open(self):
        closures = self._all_closed()
        closures["retro_033_closed"] = False
        items = retro._build_open_retro_items(closures, {})
        assert any("RETRO-033" in item for item in items)

    def test_retro_049_item_includes_auroc_when_available(self):
        closures = self._all_open()
        closures["retro_033_closed"] = True
        closures["retro_038_closed"] = True
        closures["retro_039_confirmed"] = True
        items = retro._build_open_retro_items(
            closures, {507: {"auroc": 0.4, "retro_049_closed": False}}
        )
        assert any("0.4" in item for item in items)

    def test_all_open_produces_four_items(self):
        # retro_033, retro_038, retro_039, retro_049 should all produce items
        items = retro._build_open_retro_items(
            self._all_open(),
            {507: {"auroc": 0.4, "retro_049_closed": False}},
        )
        assert len(items) == 4


# ---------------------------------------------------------------------------
# _build_new_retro_items
# ---------------------------------------------------------------------------

class TestBuildNewRetroItems:
    """_build_new_retro_items generates correct new RETRO items for .39."""

    def _closures_with_retro_048_resolved_033_open(self) -> dict:
        return {
            "retro_031_closed": True,
            "retro_033_closed": False,
            "retro_038_closed": False,
            "retro_039_confirmed": False,
            "retro_048_resolved": True,
            "retro_049_closed": False,
            "retro_050_closed": True,
        }

    def test_retro_051_generated_when_forecast_feasible_and_033_open(self):
        closures = self._closures_with_retro_048_resolved_033_open()
        results = {
            502: {
                "vram_forecasts": [{"is_feasible": True}, {"is_feasible": True}],
                "retro_033_closed": False,
            }
        }
        items = retro._build_new_retro_items(closures, results, n_deferred=3)
        ids = [r["id"] for r in items]
        assert "RETRO-051" in ids
        r051 = next(r for r in items if r["id"] == "RETRO-051")
        assert r051["priority"] == "CRITICAL"
        assert "2026.04.39" == r051["target_milestone"]

    def test_retro_051_not_generated_when_033_closed(self):
        closures = self._closures_with_retro_048_resolved_033_open()
        closures["retro_033_closed"] = True
        items = retro._build_new_retro_items(closures, {}, n_deferred=0)
        ids = [r["id"] for r in items]
        assert "RETRO-051" not in ids

    def test_retro_052_generated_when_n_scripts_patched_eq_0(self):
        closures = self._closures_with_retro_048_resolved_033_open()
        results = {505: {"n_scripts_patched": 0, "n_scripts_found": 0}}
        items = retro._build_new_retro_items(closures, results, n_deferred=0)
        ids = [r["id"] for r in items]
        assert "RETRO-052" in ids

    def test_retro_052_not_generated_when_scripts_were_patched(self):
        closures = self._closures_with_retro_048_resolved_033_open()
        results = {505: {"n_scripts_patched": 5, "n_scripts_found": 5}}
        items = retro._build_new_retro_items(closures, results, n_deferred=0)
        ids = [r["id"] for r in items]
        assert "RETRO-052" not in ids

    def test_retro_049_carry_generated_when_still_open(self):
        closures = self._closures_with_retro_048_resolved_033_open()
        results = {507: {"auroc": 0.4, "retro_049_closed": False}}
        items = retro._build_new_retro_items(closures, results, n_deferred=0)
        ids = [r["id"] for r in items]
        assert "RETRO-049" in ids

    def test_no_duplicate_ids(self):
        closures = self._closures_with_retro_048_resolved_033_open()
        results = {
            502: {"vram_forecasts": [{"is_feasible": True}], "retro_033_closed": False},
            505: {"n_scripts_patched": 0, "n_scripts_found": 0},
            507: {"auroc": 0.4, "retro_049_closed": False},
        }
        items = retro._build_new_retro_items(closures, results, n_deferred=3)
        ids = [r["id"] for r in items]
        assert len(ids) == len(set(ids)), f"Duplicate RETRO IDs: {ids}"

    def test_all_items_have_required_fields(self):
        closures = self._closures_with_retro_048_resolved_033_open()
        results = {
            502: {"vram_forecasts": [{"is_feasible": True}], "retro_033_closed": False},
            505: {"n_scripts_patched": 0, "n_scripts_found": 0},
            507: {"auroc": 0.4},
        }
        items = retro._build_new_retro_items(closures, results, n_deferred=3)
        for item in items:
            assert "id" in item
            assert "description" in item
            assert "priority" in item
            assert "target_milestone" in item


# ---------------------------------------------------------------------------
# _build_meta_reflection
# ---------------------------------------------------------------------------

class TestBuildMetaReflection:
    """_build_meta_reflection composes a structured reflection dict."""

    def _base_closures(self) -> dict:
        return {
            "retro_031_closed": True,
            "retro_033_closed": False,
            "retro_038_closed": False,
            "retro_039_confirmed": False,
            "retro_048_resolved": True,
            "retro_049_closed": False,
            "retro_050_closed": True,
        }

    def test_has_required_keys(self):
        meta = retro._build_meta_reflection(
            self._base_closures(), {}, {}, n_deferred=3
        )
        assert "vram_status" in meta
        assert "vram_note" in meta
        assert "credibility_verdict" in meta
        assert "retro_033_miss_count" in meta
        assert "closures_achieved_in_38" in meta
        assert "wall_time_note" in meta

    def test_vram_status_partially_resolved_when_048_resolved_and_033_open(self):
        meta = retro._build_meta_reflection(
            self._base_closures(), {}, {}, n_deferred=3
        )
        assert meta["vram_status"] == "PARTIALLY_RESOLVED"

    def test_vram_status_fully_resolved_when_both_resolved_and_closed(self):
        closures = self._base_closures()
        closures["retro_033_closed"] = True
        meta = retro._build_meta_reflection(closures, {}, {}, n_deferred=0)
        assert meta["vram_status"] == "FULLY_RESOLVED"

    def test_miss_count_is_6(self):
        meta = retro._build_meta_reflection(
            self._base_closures(), {}, {}, n_deferred=3
        )
        assert meta["retro_033_miss_count"] == 6

    def test_closures_list_includes_031_when_closed(self):
        results = {508: {"best_family": "gaussian_mixture", "retro_031_closed": True}}
        meta = retro._build_meta_reflection(
            self._base_closures(), {}, results, n_deferred=0
        )
        assert any("RETRO-031" in c for c in meta["closures_achieved_in_38"])

    def test_closures_list_includes_050_when_closed(self):
        results = {509: {"retro_050_closed": True, "energy_magnitude_better": True}}
        meta = retro._build_meta_reflection(
            self._base_closures(), {}, results, n_deferred=0
        )
        assert any("RETRO-050" in c for c in meta["closures_achieved_in_38"])
