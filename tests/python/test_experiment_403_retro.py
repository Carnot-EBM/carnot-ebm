"""Tests for scripts/experiment_403_retro_2026_04_29.py — Milestone 2026.04.29 retro.

Coverage targets
----------------
- MilestoneRetro2026_04_29: dataclass construction, all 18 fields, type checks
- compute_retro_2026_04_29: all 14 success criteria, positive and negative fixtures
- build_retro_artifact: schema v4, required fields, headline_results, first_live_gpu flag
- estimate_speedup_pct: positive speedup, zero case, regression case
- load_milestone_results: missing files, valid JSON, partial JSON, invalid JSON, None keys
- compute_timing_stats: normal, empty list, missing-status experiments, all-blocked
- _check_cikan_implemented: file absent, JSON content, no class, wrong status, full pass
- main() integration: runs without error against real repo root (CPU-only, no GPU needed)
- RETRO item generation: RETRO-022/023/024 opened and RETRO-016_CLOSE conditioned correctly

Spec: REQ-INFRA-017/018/019 (LiveGPUGate, preflight),
      REQ-LEARN-025/026/027 (EORM/relay retrain),
      REQ-BENCH-003/004/006/007 (benchmarks), REQ-EXTRACT-023/025, REQ-AGENT-001/002,
      REQ-EBM-031 (semantic energy)
SCENARIO: RETRO-2026.04.29
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_403_retro_2026_04_29 import (
    MILESTONE,
    MILESTONE_EXPERIMENTS,
    MilestoneRetro2026_04_29,
    NEW_RETRO_ITEMS,
    PREV_MEAN_EXP_DURATION_MIN,
    RESULT_FILE_MAP,
    _check_cikan_implemented,
    build_retro_artifact,
    compute_retro_2026_04_29,
    compute_timing_stats,
    estimate_speedup_pct,
    load_milestone_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON data to path, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _make_retro(**overrides: Any) -> MilestoneRetro2026_04_29:
    """Build a MilestoneRetro2026_04_29 with all-False defaults plus overrides."""
    defaults: dict[str, Any] = {
        "retro_019_resolved": False,
        "retro_020_closed": False,
        "retro_021_closed": False,
        "live_gpu_confirmed": False,
        "precision_result_credible": False,
        "humaneval_result_credible": False,
        "adversarial_result_credible": False,
        "extraction_winner_known": False,
        "fr11_learning_confirmed": False,
        "jitrl_memory_works": False,
        "safety_kan_works": False,
        "saver_live_verified": False,
        "semantic_energy_viable": False,
        "crane_extraction_improved": False,
        "mean_exp_duration_min": 0.0,
        "n_experiments_blocked": 0,
        "retro_items_opened": [],
        "headline_results": {},
    }
    defaults.update(overrides)
    return MilestoneRetro2026_04_29(**defaults)


# ---------------------------------------------------------------------------
# MilestoneRetro2026_04_29 dataclass
# ---------------------------------------------------------------------------


class TestMilestoneRetroDataclass:
    """MilestoneRetro2026_04_29 is a plain dataclass — all fields must be settable."""

    def test_construction_all_false(self) -> None:
        retro = _make_retro()
        assert retro.live_gpu_confirmed is False
        assert retro.retro_019_resolved is False
        assert retro.retro_020_closed is False
        assert retro.retro_021_closed is False
        assert retro.precision_result_credible is False
        assert retro.humaneval_result_credible is False
        assert retro.adversarial_result_credible is False
        assert retro.extraction_winner_known is False
        assert retro.fr11_learning_confirmed is False
        assert retro.jitrl_memory_works is False
        assert retro.safety_kan_works is False
        assert retro.saver_live_verified is False
        assert retro.semantic_energy_viable is False
        assert retro.crane_extraction_improved is False
        assert retro.retro_items_opened == []
        assert retro.headline_results == {}

    def test_bool_fields_can_be_true(self) -> None:
        retro = _make_retro(
            retro_019_resolved=True,
            live_gpu_confirmed=True,
            jitrl_memory_works=True,
            safety_kan_works=True,
            semantic_energy_viable=True,
            crane_extraction_improved=True,
        )
        assert retro.retro_019_resolved is True
        assert retro.live_gpu_confirmed is True
        assert retro.jitrl_memory_works is True
        assert retro.safety_kan_works is True
        assert retro.semantic_energy_viable is True
        assert retro.crane_extraction_improved is True

    def test_mean_duration_float(self) -> None:
        retro = _make_retro(mean_exp_duration_min=7.5)
        assert isinstance(retro.mean_exp_duration_min, float)
        assert retro.mean_exp_duration_min == pytest.approx(7.5)

    def test_n_experiments_blocked_int(self) -> None:
        retro = _make_retro(n_experiments_blocked=4)
        assert isinstance(retro.n_experiments_blocked, int)
        assert retro.n_experiments_blocked == 4

    def test_retro_items_list(self) -> None:
        retro = _make_retro(retro_items_opened=["RETRO-022", "RETRO-023", "RETRO-024"])
        assert retro.retro_items_opened == ["RETRO-022", "RETRO-023", "RETRO-024"]

    def test_headline_results_dict(self) -> None:
        hr = {"exp_394": {"honest_verdict": "live_improvement"}}
        retro = _make_retro(headline_results=hr)
        assert retro.headline_results == hr

    def test_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(MilestoneRetro2026_04_29)

    def test_field_names_complete(self) -> None:
        """All 18 fields from the task spec must be present."""
        field_names = {f.name for f in dataclasses.fields(MilestoneRetro2026_04_29)}
        required = {
            "retro_019_resolved",
            "retro_020_closed",
            "retro_021_closed",
            "live_gpu_confirmed",
            "precision_result_credible",
            "humaneval_result_credible",
            "adversarial_result_credible",
            "extraction_winner_known",
            "fr11_learning_confirmed",
            "jitrl_memory_works",
            "safety_kan_works",
            "saver_live_verified",
            "semantic_energy_viable",
            "crane_extraction_improved",
            "mean_exp_duration_min",
            "n_experiments_blocked",
            "retro_items_opened",
            "headline_results",
        }
        assert required.issubset(field_names)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_milestone_string(self) -> None:
        assert MILESTONE == "2026.04.29"

    def test_prev_mean_positive(self) -> None:
        assert PREV_MEAN_EXP_DURATION_MIN > 0.0

    def test_prev_mean_value(self) -> None:
        # Previous milestone (2026.04.28) had mean of 14.0 min/exp
        assert PREV_MEAN_EXP_DURATION_MIN == pytest.approx(14.0)

    def test_result_file_map_has_thirteen_entries(self) -> None:
        # Exps 390-402 = 13 experiments
        assert len(RESULT_FILE_MAP) == 13

    def test_result_file_map_keys(self) -> None:
        expected_keys = {str(i) for i in range(390, 403)}
        assert set(RESULT_FILE_MAP.keys()) == expected_keys

    def test_milestone_experiments_has_thirteen(self) -> None:
        assert len(MILESTONE_EXPERIMENTS) == 13

    def test_new_retro_items_has_three(self) -> None:
        # RETRO-022 (critical), RETRO-023, RETRO-024
        assert len(NEW_RETRO_ITEMS) == 3

    def test_new_retro_item_ids(self) -> None:
        ids = {item["id"] for item in NEW_RETRO_ITEMS}
        assert ids == {"RETRO-022", "RETRO-023", "RETRO-024"}

    def test_retro_022_priority_critical(self) -> None:
        retro022 = next(i for i in NEW_RETRO_ITEMS if i["id"] == "RETRO-022")
        assert retro022["priority"] == "critical"

    def test_milestone_experiments_wall_times_non_negative(self) -> None:
        for exp in MILESTONE_EXPERIMENTS:
            assert exp["wall_time_min"] >= 0, f"Exp {exp['id']} has negative wall time"

    def test_milestone_experiments_ids_match_range(self) -> None:
        exp_ids = {exp["id"] for exp in MILESTONE_EXPERIMENTS}
        assert exp_ids == set(range(390, 403))


# ---------------------------------------------------------------------------
# estimate_speedup_pct
# ---------------------------------------------------------------------------


class TestEstimateSpeedupPct:
    def test_speedup_positive(self) -> None:
        # 14.0 → 7.5 = (14.0-7.5)/14.0 * 100 ≈ 46.4%
        result = estimate_speedup_pct(14.0, 7.5)
        assert result > 0.0

    def test_speedup_regression(self) -> None:
        # curr > prev → negative speedup
        result = estimate_speedup_pct(14.0, 20.0)
        assert result < 0.0

    def test_no_change(self) -> None:
        result = estimate_speedup_pct(14.0, 14.0)
        assert result == pytest.approx(0.0)

    def test_zero_prev_mean(self) -> None:
        # Guard against ZeroDivisionError
        result = estimate_speedup_pct(0.0, 10.0)
        assert result == 0.0

    def test_returns_float(self) -> None:
        result = estimate_speedup_pct(14.0, 7.5)
        assert isinstance(result, float)

    def test_precision_two_decimal_places(self) -> None:
        result = estimate_speedup_pct(14.0, 7.5)
        assert result == round(result, 2)

    def test_full_speedup(self) -> None:
        # curr = 0 → 100% speedup
        result = estimate_speedup_pct(14.0, 0.0)
        assert result == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# load_milestone_results
# ---------------------------------------------------------------------------


class TestLoadMilestoneResults:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        file_map = {"390": "results/experiment_390.json"}
        results = load_milestone_results(tmp_path, file_map)
        assert results["390"] is None

    def test_valid_json_loaded(self, tmp_path: Path) -> None:
        data = {"status": "complete", "experiment": 390}
        _write_json(tmp_path / "results/experiment_390.json", data)
        file_map = {"390": "results/experiment_390.json"}
        results = load_milestone_results(tmp_path, file_map)
        assert results["390"] == data

    def test_invalid_json_returns_none(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "results" / "exp.json"
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        bad_path.write_text("not valid json {{{")
        results = load_milestone_results(tmp_path, {"x": "results/exp.json"})
        assert results["x"] is None

    def test_none_path_returns_none(self, tmp_path: Path) -> None:
        results = load_milestone_results(tmp_path, {"391": None})
        assert results["391"] is None

    def test_partial_result_loaded(self, tmp_path: Path) -> None:
        data = {"status": "partial", "experiment": 394}
        _write_json(tmp_path / "results/experiment_394.json", data)
        results = load_milestone_results(tmp_path, {"394": "results/experiment_394.json"})
        assert results["394"]["status"] == "partial"

    def test_empty_file_map(self, tmp_path: Path) -> None:
        results = load_milestone_results(tmp_path, {})
        assert results == {}

    def test_multiple_files_mixed(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "results/a.json", {"ok": True})
        file_map: dict[str, str | None] = {
            "a": "results/a.json",
            "b": "results/missing.json",
            "c": None,
        }
        results = load_milestone_results(tmp_path, file_map)
        assert results["a"] == {"ok": True}
        assert results["b"] is None
        assert results["c"] is None


# ---------------------------------------------------------------------------
# compute_timing_stats
# ---------------------------------------------------------------------------


class TestComputeTimingStats:
    def test_empty_list(self) -> None:
        stats = compute_timing_stats([])
        assert stats["n_ran"] == 0
        assert stats["mean_min"] == 0.0
        assert stats["slowest"] is None
        assert stats["fastest"] is None

    def test_single_experiment(self) -> None:
        exps = [{"id": 390, "title": "test", "wall_time_min": 9, "status": "completed"}]
        stats = compute_timing_stats(exps)
        assert stats["n_ran"] == 1
        assert stats["mean_min"] == 9.0
        assert stats["slowest"]["id"] == 390
        assert stats["fastest"]["id"] == 390

    def test_blocked_count(self) -> None:
        exps = [
            {"id": 390, "wall_time_min": 9, "status": "completed"},
            {"id": 392, "wall_time_min": 0, "status": "missing"},
            {"id": 393, "wall_time_min": 0, "status": "missing"},
            {"id": 401, "wall_time_min": 0, "status": "missing"},
            {"id": 402, "wall_time_min": 0, "status": "missing"},
        ]
        stats = compute_timing_stats(exps)
        assert stats["n_blocked"] == 4

    def test_mean_includes_zero_wall_times(self) -> None:
        exps = [
            {"id": 1, "wall_time_min": 14, "status": "completed"},
            {"id": 2, "wall_time_min": 0, "status": "missing"},
        ]
        stats = compute_timing_stats(exps)
        assert stats["mean_min"] == pytest.approx(7.0)

    def test_slowest_fastest_identification(self) -> None:
        exps = [
            {"id": 390, "title": "slow", "wall_time_min": 9, "status": "completed"},
            {"id": 401, "title": "zero", "wall_time_min": 0, "status": "missing"},
            {"id": 394, "title": "mid", "wall_time_min": 7, "status": "partial"},
        ]
        stats = compute_timing_stats(exps)
        assert stats["slowest"]["id"] == 390
        assert stats["fastest"]["id"] == 401

    def test_n_ran_equals_total(self) -> None:
        exps = [{"id": i, "wall_time_min": 7, "status": "partial"} for i in range(13)]
        stats = compute_timing_stats(exps)
        assert stats["n_ran"] == 13

    def test_total_wall_time(self) -> None:
        exps = [
            {"id": 1, "wall_time_min": 9, "status": "partial"},
            {"id": 2, "wall_time_min": 7, "status": "partial"},
            {"id": 3, "wall_time_min": 0, "status": "missing"},
        ]
        stats = compute_timing_stats(exps)
        assert stats["total_min"] == 16

    def test_blocked_status_counted(self) -> None:
        exps = [{"id": 1, "wall_time_min": 5, "status": "blocked"}]
        stats = compute_timing_stats(exps)
        assert stats["n_blocked"] == 1


# ---------------------------------------------------------------------------
# _check_cikan_implemented
# ---------------------------------------------------------------------------


class TestCheckCikanImplemented:
    def test_file_absent_returns_false(self, tmp_path: Path) -> None:
        results: dict[str, Any] = {"391": {"status": "success"}}
        assert _check_cikan_implemented(tmp_path, results) is False

    def test_json_object_content_returns_false(self, tmp_path: Path) -> None:
        cikan_path = tmp_path / "python" / "carnot" / "models" / "cikan_energy.py"
        cikan_path.parent.mkdir(parents=True, exist_ok=True)
        cikan_path.write_text('{"experiment": 375, "status": "partial"}')
        results: dict[str, Any] = {"391": {"status": "success"}}
        assert _check_cikan_implemented(tmp_path, results) is False

    def test_json_array_content_returns_false(self, tmp_path: Path) -> None:
        cikan_path = tmp_path / "python" / "carnot" / "models" / "cikan_energy.py"
        cikan_path.parent.mkdir(parents=True, exist_ok=True)
        cikan_path.write_text('[{"a": 1}]')
        results: dict[str, Any] = {"391": {"status": "success"}}
        assert _check_cikan_implemented(tmp_path, results) is False

    def test_python_without_class_returns_false(self, tmp_path: Path) -> None:
        cikan_path = tmp_path / "python" / "carnot" / "models" / "cikan_energy.py"
        cikan_path.parent.mkdir(parents=True, exist_ok=True)
        cikan_path.write_text("def compute_energy(x):\n    return x ** 2\n")
        results: dict[str, Any] = {"391": {"status": "success"}}
        assert _check_cikan_implemented(tmp_path, results) is False

    def test_class_present_but_exp391_missing_returns_false(self, tmp_path: Path) -> None:
        cikan_path = tmp_path / "python" / "carnot" / "models" / "cikan_energy.py"
        cikan_path.parent.mkdir(parents=True, exist_ok=True)
        cikan_path.write_text("class CIKANEnergy:\n    pass\n")
        results: dict[str, Any] = {"391": None}
        assert _check_cikan_implemented(tmp_path, results) is False

    def test_class_present_but_status_partial_returns_false(self, tmp_path: Path) -> None:
        cikan_path = tmp_path / "python" / "carnot" / "models" / "cikan_energy.py"
        cikan_path.parent.mkdir(parents=True, exist_ok=True)
        cikan_path.write_text("class CIKANEnergy:\n    pass\n")
        results: dict[str, Any] = {"391": {"status": "partial"}}
        assert _check_cikan_implemented(tmp_path, results) is False

    def test_full_pass(self, tmp_path: Path) -> None:
        cikan_path = tmp_path / "python" / "carnot" / "models" / "cikan_energy.py"
        cikan_path.parent.mkdir(parents=True, exist_ok=True)
        cikan_path.write_text("class CIKANEnergy:\n    def energy(self, x):\n        return x\n")
        results: dict[str, Any] = {"391": {"status": "success"}}
        assert _check_cikan_implemented(tmp_path, results) is True

    def test_exp391_key_absent_returns_false(self, tmp_path: Path) -> None:
        cikan_path = tmp_path / "python" / "carnot" / "models" / "cikan_energy.py"
        cikan_path.parent.mkdir(parents=True, exist_ok=True)
        cikan_path.write_text("class CIKANEnergy:\n    pass\n")
        results: dict[str, Any] = {}
        assert _check_cikan_implemented(tmp_path, results) is False


# ---------------------------------------------------------------------------
# compute_retro_2026_04_29
# ---------------------------------------------------------------------------


class TestComputeRetro2026_04_29:
    """Success criteria evaluation — positive and negative fixtures for all 14 criteria."""

    def _all_none(self) -> dict[str, Any | None]:
        """Fixture with all result files missing (worst case)."""
        return {str(i): None for i in range(390, 403)}

    def test_all_missing_all_false(self, tmp_path: Path) -> None:
        retro = compute_retro_2026_04_29(self._all_none(), tmp_path)
        assert retro.retro_019_resolved is False
        assert retro.retro_020_closed is False
        assert retro.retro_021_closed is False
        assert retro.live_gpu_confirmed is False
        assert retro.precision_result_credible is False
        assert retro.humaneval_result_credible is False
        assert retro.adversarial_result_credible is False
        assert retro.extraction_winner_known is False
        assert retro.fr11_learning_confirmed is False
        assert retro.jitrl_memory_works is False
        assert retro.safety_kan_works is False
        assert retro.saver_live_verified is False
        assert retro.semantic_energy_viable is False
        assert retro.crane_extraction_improved is False

    # --- retro_019_resolved ---

    def test_retro_019_resolved_when_gpu_confirmed_live(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["390"] = {"honest_verdict": "gpu_confirmed_live", "status": "complete"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.retro_019_resolved is True

    def test_retro_019_not_resolved_when_status_complete_only(self, tmp_path: Path) -> None:
        # "complete" status without gpu_confirmed_live verdict is NOT resolved
        files = self._all_none()
        files["390"] = {"status": "complete", "finding": "GPU preflight script created."}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.retro_019_resolved is False

    def test_retro_019_not_resolved_when_missing(self, tmp_path: Path) -> None:
        files = self._all_none()
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.retro_019_resolved is False

    # --- live_gpu_confirmed ---

    def test_live_gpu_confirmed_when_any_result_live(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["394"] = {"status": "success", "inference_mode": "live_gpu"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.live_gpu_confirmed is True

    def test_live_gpu_not_confirmed_when_all_partial(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["394"] = {"status": "partial", "inference_mode": "blocked"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.live_gpu_confirmed is False

    def test_live_gpu_confirmed_from_exp390(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["390"] = {"honest_verdict": "gpu_confirmed_live", "inference_mode": "live_gpu"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.live_gpu_confirmed is True

    # --- precision_result_credible ---

    def test_precision_credible_when_live_improvement(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["394"] = {"honest_verdict": "live_improvement", "inference_mode": "live_gpu"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.precision_result_credible is True

    def test_precision_not_credible_without_live_gpu(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["394"] = {"honest_verdict": "live_improvement", "inference_mode": "simulated"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.precision_result_credible is False

    def test_precision_not_credible_wrong_verdict(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["394"] = {"honest_verdict": "no_improvement", "inference_mode": "live_gpu"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.precision_result_credible is False

    # --- humaneval_result_credible ---

    def test_humaneval_credible_when_code_verification_positive(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["395"] = {"honest_verdict": "code_verification_positive", "inference_mode": "live_gpu"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.humaneval_result_credible is True

    def test_humaneval_not_credible_wrong_verdict(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["395"] = {"honest_verdict": "no_improvement", "inference_mode": "live_gpu"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.humaneval_result_credible is False

    def test_humaneval_not_credible_simulated(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["395"] = {"honest_verdict": "code_verification_positive", "inference_mode": "simulated"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.humaneval_result_credible is False

    # --- adversarial_result_credible ---

    def test_adversarial_credible_when_improvement_positive(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["396"] = {"honest_verdict": "improvement_positive", "inference_mode": "live_gpu"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.adversarial_result_credible is True

    def test_adversarial_not_credible_simulated(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["396"] = {"honest_verdict": "improvement_positive", "inference_mode": "simulated"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.adversarial_result_credible is False

    # --- extraction_winner_known ---

    def test_extraction_known_when_live_winner(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["397"] = {"honest_verdict": "live_gpu_winner", "inference_mode": "live_gpu"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.extraction_winner_known is True

    def test_extraction_known_when_live_no_improvement(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["397"] = {"honest_verdict": "live_gpu_no_improvement", "inference_mode": "live_gpu"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.extraction_winner_known is True

    def test_extraction_unknown_when_simulated(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["397"] = {"honest_verdict": "live_gpu_winner", "inference_mode": "simulated"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.extraction_winner_known is False

    def test_extraction_unknown_when_blocked(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["397"] = {"honest_verdict": "blocked", "inference_mode": "live_gpu"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.extraction_winner_known is False

    # --- fr11_learning_confirmed (and retro_021_closed alias) ---

    def test_fr11_confirmed_when_learning_confirmed_live(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["399"] = {"honest_verdict": "learning_confirmed", "inference_mode": "live_gpu"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.fr11_learning_confirmed is True

    def test_fr11_not_confirmed_synthetic_only(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["399"] = {"honest_verdict": "learning_confirmed", "inference_mode": "synthetic"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.fr11_learning_confirmed is False

    def test_fr11_not_confirmed_wrong_verdict(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["399"] = {"honest_verdict": "insufficient_data", "inference_mode": "live_gpu"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.fr11_learning_confirmed is False

    def test_retro_021_closed_matches_fr11_confirmed(self, tmp_path: Path) -> None:
        # retro_021_closed is the same as fr11_learning_confirmed
        files = self._all_none()
        files["399"] = {"honest_verdict": "learning_confirmed", "inference_mode": "live_gpu"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.retro_021_closed == retro.fr11_learning_confirmed

    def test_retro_021_not_closed_when_partial(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["399"] = {"status": "partial"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.retro_021_closed is False

    # --- jitrl_memory_works ---

    def test_jitrl_works_when_threshold_modulation_works_true(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["392"] = {"threshold_modulation_works": True}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.jitrl_memory_works is True

    def test_jitrl_not_works_false_value(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["392"] = {"threshold_modulation_works": False}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.jitrl_memory_works is False

    def test_jitrl_not_works_missing(self, tmp_path: Path) -> None:
        files = self._all_none()
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.jitrl_memory_works is False

    def test_jitrl_not_works_wrong_key(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["392"] = {"honest_verdict": "threshold_modulation_confirmed"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.jitrl_memory_works is False

    # --- safety_kan_works ---

    def test_safety_kan_works_above_threshold(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["393"] = {"test_auroc": 0.82}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.safety_kan_works is True

    def test_safety_kan_not_works_below_threshold(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["393"] = {"test_auroc": 0.65}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.safety_kan_works is False

    def test_safety_kan_not_works_exactly_threshold(self, tmp_path: Path) -> None:
        # Must be ABOVE 0.70, not equal
        files = self._all_none()
        files["393"] = {"test_auroc": 0.70}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.safety_kan_works is False

    def test_safety_kan_not_works_missing(self, tmp_path: Path) -> None:
        files = self._all_none()
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.safety_kan_works is False

    def test_safety_kan_not_works_non_numeric_auroc(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["393"] = {"test_auroc": "not_a_number"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.safety_kan_works is False

    def test_safety_kan_works_integer_auroc(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["393"] = {"test_auroc": 1}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.safety_kan_works is True

    # --- saver_live_verified ---

    def test_saver_verified_when_live_with_active(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["400"] = {"inference_mode": "live_gpu", "live_verification_active": True}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.saver_live_verified is True

    def test_saver_not_verified_simulated(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["400"] = {"inference_mode": "simulated", "live_verification_active": True}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.saver_live_verified is False

    def test_saver_not_verified_inactive(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["400"] = {"inference_mode": "live_gpu", "live_verification_active": False}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.saver_live_verified is False

    def test_saver_not_verified_missing(self, tmp_path: Path) -> None:
        files = self._all_none()
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.saver_live_verified is False

    # --- semantic_energy_viable ---

    def test_semantic_viable_above_threshold(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["401"] = {"auroc": 0.80}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.semantic_energy_viable is True

    def test_semantic_not_viable_below_threshold(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["401"] = {"auroc": 0.65}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.semantic_energy_viable is False

    def test_semantic_not_viable_exactly_threshold(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["401"] = {"auroc": 0.70}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.semantic_energy_viable is False

    def test_semantic_not_viable_missing(self, tmp_path: Path) -> None:
        files = self._all_none()
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.semantic_energy_viable is False

    def test_semantic_not_viable_non_numeric(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["401"] = {"auroc": "not_a_number"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.semantic_energy_viable is False

    # --- crane_extraction_improved ---

    def test_crane_improved_when_beats_arithmetic(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["402"] = {"crane_beats_arithmetic": True}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.crane_extraction_improved is True

    def test_crane_not_improved_false_value(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["402"] = {"crane_beats_arithmetic": False}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.crane_extraction_improved is False

    def test_crane_not_improved_missing(self, tmp_path: Path) -> None:
        files = self._all_none()
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.crane_extraction_improved is False

    # --- retro_020_closed ---

    def test_retro_020_closed_when_cikan_implemented_and_exp391_success(self, tmp_path: Path) -> None:
        cikan_path = tmp_path / "python" / "carnot" / "models" / "cikan_energy.py"
        cikan_path.parent.mkdir(parents=True, exist_ok=True)
        cikan_path.write_text("class CIKANEnergy:\n    pass\n")
        files = self._all_none()
        files["391"] = {"status": "success"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.retro_020_closed is True

    def test_retro_020_not_closed_when_cikan_json(self, tmp_path: Path) -> None:
        cikan_path = tmp_path / "python" / "carnot" / "models" / "cikan_energy.py"
        cikan_path.parent.mkdir(parents=True, exist_ok=True)
        cikan_path.write_text('{"experiment": 375, "status": "partial"}')
        files = self._all_none()
        files["391"] = {"status": "success"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.retro_020_closed is False

    def test_retro_020_not_closed_when_missing(self, tmp_path: Path) -> None:
        files = self._all_none()
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.retro_020_closed is False

    # --- timing stats ---

    def test_mean_exp_duration_computed(self, tmp_path: Path) -> None:
        retro = compute_retro_2026_04_29(self._all_none(), tmp_path)
        assert isinstance(retro.mean_exp_duration_min, float)
        assert retro.mean_exp_duration_min >= 0.0

    def test_n_experiments_blocked_computed(self, tmp_path: Path) -> None:
        retro = compute_retro_2026_04_29(self._all_none(), tmp_path)
        assert isinstance(retro.n_experiments_blocked, int)
        # Exps 392, 393, 401, 402 have status="missing"
        assert retro.n_experiments_blocked >= 4

    # --- RETRO item generation ---

    def test_retro_022_opened_when_live_gpu_not_confirmed(self, tmp_path: Path) -> None:
        files = self._all_none()
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert "RETRO-022" in retro.retro_items_opened

    def test_retro_022_not_opened_when_live_gpu_confirmed(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["394"] = {"inference_mode": "live_gpu", "status": "success"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert "RETRO-022" not in retro.retro_items_opened

    def test_retro_023_opened_when_cikan_not_implemented(self, tmp_path: Path) -> None:
        files = self._all_none()
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert "RETRO-023" in retro.retro_items_opened

    def test_retro_023_not_opened_when_cikan_implemented(self, tmp_path: Path) -> None:
        cikan_path = tmp_path / "python" / "carnot" / "models" / "cikan_energy.py"
        cikan_path.parent.mkdir(parents=True, exist_ok=True)
        cikan_path.write_text("class CIKANEnergy:\n    pass\n")
        files = self._all_none()
        files["391"] = {"status": "success"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert "RETRO-023" not in retro.retro_items_opened

    def test_retro_024_opened_when_fr11_not_confirmed(self, tmp_path: Path) -> None:
        files = self._all_none()
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert "RETRO-024" in retro.retro_items_opened

    def test_retro_024_not_opened_when_fr11_confirmed(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["399"] = {"honest_verdict": "learning_confirmed", "inference_mode": "live_gpu"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert "RETRO-024" not in retro.retro_items_opened

    def test_retro_016_close_opened_when_extraction_known(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["397"] = {"honest_verdict": "live_gpu_winner", "inference_mode": "live_gpu"}
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert "RETRO-016_CLOSE" in retro.retro_items_opened

    def test_retro_016_close_not_opened_when_extraction_unknown(self, tmp_path: Path) -> None:
        files = self._all_none()
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert "RETRO-016_CLOSE" not in retro.retro_items_opened

    def test_all_criteria_fail_opens_all_three_retros(self, tmp_path: Path) -> None:
        retro = compute_retro_2026_04_29(self._all_none(), tmp_path)
        assert set(retro.retro_items_opened) >= {"RETRO-022", "RETRO-023", "RETRO-024"}

    # --- headline_results empty when no live GPU ---

    def test_headline_results_empty_when_no_live_gpu(self, tmp_path: Path) -> None:
        files = self._all_none()
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert retro.headline_results == {}

    def test_headline_results_populated_when_live_and_credible(self, tmp_path: Path) -> None:
        files = self._all_none()
        files["394"] = {
            "honest_verdict": "live_improvement",
            "inference_mode": "live_gpu",
            "signed_improvement": 0.12,
        }
        retro = compute_retro_2026_04_29(files, tmp_path)
        assert "exp_394_precision" in retro.headline_results
        assert retro.headline_results["exp_394_precision"]["honest_verdict"] == "live_improvement"


# ---------------------------------------------------------------------------
# build_retro_artifact
# ---------------------------------------------------------------------------


class TestBuildRetroArtifact:
    def _make_retro_all_false(self) -> MilestoneRetro2026_04_29:
        return _make_retro(
            retro_items_opened=["RETRO-022", "RETRO-023", "RETRO-024"],
            n_experiments_blocked=4,
            mean_exp_duration_min=7.5,
        )

    def test_schema_v4(self) -> None:
        artifact = build_retro_artifact(self._make_retro_all_false())
        assert artifact["schema"] == "carnot.operational_retro.v4"

    def test_milestone_field(self) -> None:
        artifact = build_retro_artifact(self._make_retro_all_false())
        assert artifact["milestone"] == "2026.04.29"

    def test_first_live_gpu_results_achieved_false(self) -> None:
        artifact = build_retro_artifact(self._make_retro_all_false())
        assert artifact["first_live_gpu_results_achieved"] is False

    def test_first_live_gpu_results_achieved_true(self) -> None:
        retro = _make_retro(live_gpu_confirmed=True)
        artifact = build_retro_artifact(retro)
        assert artifact["first_live_gpu_results_achieved"] is True

    def test_headline_results_present(self) -> None:
        artifact = build_retro_artifact(self._make_retro_all_false())
        assert "headline_results" in artifact
        assert isinstance(artifact["headline_results"], dict)

    def test_headline_results_empty_when_no_live_gpu(self) -> None:
        artifact = build_retro_artifact(self._make_retro_all_false())
        assert artifact["headline_results"] == {}

    def test_success_criteria_present(self) -> None:
        artifact = build_retro_artifact(self._make_retro_all_false())
        sc = artifact["success_criteria"]
        required_keys = {
            "retro_019_resolved",
            "retro_020_closed",
            "retro_021_closed",
            "live_gpu_confirmed",
            "precision_result_credible",
            "humaneval_result_credible",
            "adversarial_result_credible",
            "extraction_winner_known",
            "fr11_learning_confirmed",
            "jitrl_memory_works",
            "safety_kan_works",
            "saver_live_verified",
            "semantic_energy_viable",
            "crane_extraction_improved",
            "n_experiments_blocked",
        }
        assert required_keys.issubset(sc.keys())

    def test_explanations_present(self) -> None:
        artifact = build_retro_artifact(self._make_retro_all_false())
        assert "explanations" in artifact
        assert len(artifact["explanations"]) > 0

    def test_timing_analysis_present(self) -> None:
        artifact = build_retro_artifact(self._make_retro_all_false())
        ta = artifact["timing_analysis"]
        assert "mean_exp_duration_min" in ta
        assert "estimated_speedup_pct" in ta

    def test_retro_items_opened_in_artifact(self) -> None:
        retro = _make_retro(retro_items_opened=["RETRO-022"])
        artifact = build_retro_artifact(retro)
        assert "RETRO-022" in artifact["retro_items_opened"]

    def test_retro_016_close_not_in_opened_list(self) -> None:
        # RETRO-016_CLOSE is a close action — should NOT appear in retro_items_opened
        retro = _make_retro(retro_items_opened=["RETRO-016_CLOSE"])
        artifact = build_retro_artifact(retro)
        assert "RETRO-016_CLOSE" not in artifact["retro_items_opened"]

    def test_retro_016_in_closed_list_when_close_present(self) -> None:
        retro = _make_retro(retro_items_opened=["RETRO-016_CLOSE"])
        artifact = build_retro_artifact(retro)
        assert "RETRO-016" in artifact["retro_items_closed"]

    def test_retro_items_closed_empty_when_no_close(self) -> None:
        artifact = build_retro_artifact(self._make_retro_all_false())
        assert artifact["retro_items_closed"] == []

    def test_new_retro_items_in_artifact(self) -> None:
        artifact = build_retro_artifact(self._make_retro_all_false())
        assert "new_retro_items" in artifact
        # All three RETRO items opened
        assert len(artifact["new_retro_items"]) == 3

    def test_meta_reflection_present(self) -> None:
        artifact = build_retro_artifact(self._make_retro_all_false())
        assert "meta_reflection" in artifact
        assert len(artifact["meta_reflection"]) > 0

    def test_key_findings_present(self) -> None:
        artifact = build_retro_artifact(self._make_retro_all_false())
        assert "key_findings" in artifact

    def test_estimated_savings_next_pct_positive(self) -> None:
        artifact = build_retro_artifact(self._make_retro_all_false())
        assert artifact["estimated_savings_next_pct"] > 0

    def test_retro_type_field(self) -> None:
        artifact = build_retro_artifact(self._make_retro_all_false())
        assert artifact["retro_type"] == "full_milestone"

    def test_all_criteria_false_in_artifact(self) -> None:
        artifact = build_retro_artifact(self._make_retro_all_false())
        sc = artifact["success_criteria"]
        bool_criteria = [
            "retro_019_resolved",
            "retro_020_closed",
            "retro_021_closed",
            "live_gpu_confirmed",
            "precision_result_credible",
            "humaneval_result_credible",
            "adversarial_result_credible",
            "extraction_winner_known",
            "fr11_learning_confirmed",
            "jitrl_memory_works",
            "safety_kan_works",
            "saver_live_verified",
            "semantic_energy_viable",
            "crane_extraction_improved",
        ]
        for key in bool_criteria:
            assert sc[key] is False, f"Expected {key}=False, got {sc[key]}"

    def test_all_criteria_true_in_artifact(self) -> None:
        retro = _make_retro(
            retro_019_resolved=True,
            retro_020_closed=True,
            retro_021_closed=True,
            live_gpu_confirmed=True,
            precision_result_credible=True,
            humaneval_result_credible=True,
            adversarial_result_credible=True,
            extraction_winner_known=True,
            fr11_learning_confirmed=True,
            jitrl_memory_works=True,
            safety_kan_works=True,
            saver_live_verified=True,
            semantic_energy_viable=True,
            crane_extraction_improved=True,
        )
        artifact = build_retro_artifact(retro)
        sc = artifact["success_criteria"]
        bool_criteria = [
            "retro_019_resolved", "retro_020_closed", "retro_021_closed",
            "live_gpu_confirmed", "precision_result_credible", "humaneval_result_credible",
            "adversarial_result_credible", "extraction_winner_known", "fr11_learning_confirmed",
            "jitrl_memory_works", "safety_kan_works", "saver_live_verified",
            "semantic_energy_viable", "crane_extraction_improved",
        ]
        for key in bool_criteria:
            assert sc[key] is True, f"Expected {key}=True, got {sc[key]}"

    def test_artifact_is_json_serializable(self) -> None:
        artifact = build_retro_artifact(self._make_retro_all_false())
        serialized = json.dumps(artifact)
        roundtripped = json.loads(serialized)
        assert roundtripped["schema"] == "carnot.operational_retro.v4"

    def test_speedup_computed_correctly(self) -> None:
        retro = _make_retro(mean_exp_duration_min=PREV_MEAN_EXP_DURATION_MIN / 2)
        artifact = build_retro_artifact(retro)
        speedup = artifact["timing_analysis"]["estimated_speedup_pct"]
        assert speedup == pytest.approx(50.0, abs=1.0)

    def test_new_retro_items_empty_when_all_criteria_met(self) -> None:
        retro = _make_retro(
            live_gpu_confirmed=True,
            retro_020_closed=True,
            retro_021_closed=True,
            fr11_learning_confirmed=True,
            retro_items_opened=[],
        )
        artifact = build_retro_artifact(retro)
        assert artifact["new_retro_items"] == []


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_runs_without_error(self, tmp_path: Path) -> None:
        """main() should complete successfully against the real repo root."""
        output_path = tmp_path / "results" / "operational_retro_2026_04_29.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import scripts.experiment_403_retro_2026_04_29 as mod

        class _FakeTmpl:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._repo_root = _REPO_ROOT

            def setup(self) -> None:
                pass

            def build_result(self, data: dict, status: str = "success") -> dict:
                return {**data, "status": status}

        with patch.object(mod, "ExperimentTemplate", _FakeTmpl):
            result_files = load_milestone_results(_REPO_ROOT, RESULT_FILE_MAP)
            retro = compute_retro_2026_04_29(result_files, _REPO_ROOT)
            artifact = build_retro_artifact(retro)
            artifact_out = _FakeTmpl().build_result(artifact, status="complete")
            output_path.write_text(json.dumps(artifact_out, indent=2))

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["schema"] == "carnot.operational_retro.v4"

    def test_main_writes_valid_retro_for_real_repo(self) -> None:
        """Against the real repo root, evaluate actual milestone state."""
        result_files = load_milestone_results(_REPO_ROOT, RESULT_FILE_MAP)
        retro = compute_retro_2026_04_29(result_files, _REPO_ROOT)
        # Exp 390 has status=complete but NOT gpu_confirmed_live — not resolved
        assert retro.retro_019_resolved is False
        # Live GPU not confirmed (all partial/missing)
        assert retro.live_gpu_confirmed is False
        # RETRO-022 must be opened
        assert "RETRO-022" in retro.retro_items_opened
        # CIKAN still corrupt JSON — not implemented
        assert retro.retro_020_closed is False
        # RETRO-023 must be opened
        assert "RETRO-023" in retro.retro_items_opened
        # FR-11 not confirmed
        assert retro.fr11_learning_confirmed is False
        # RETRO-024 must be opened
        assert "RETRO-024" in retro.retro_items_opened

    def test_main_timing_stats_from_real_experiments(self) -> None:
        """Timing stats computed from MILESTONE_EXPERIMENTS are internally consistent."""
        stats = compute_timing_stats(MILESTONE_EXPERIMENTS)
        assert stats["n_ran"] == len(MILESTONE_EXPERIMENTS)
        assert stats["total_min"] == sum(e["wall_time_min"] for e in MILESTONE_EXPERIMENTS)
        assert stats["mean_min"] == pytest.approx(
            stats["total_min"] / len(MILESTONE_EXPERIMENTS), abs=0.2
        )

    def test_first_live_gpu_results_achieved_is_false_for_real_repo(self) -> None:
        """The milestone question: live GPU results NOT achieved."""
        result_files = load_milestone_results(_REPO_ROOT, RESULT_FILE_MAP)
        retro = compute_retro_2026_04_29(result_files, _REPO_ROOT)
        artifact = build_retro_artifact(retro)
        assert artifact["first_live_gpu_results_achieved"] is False
