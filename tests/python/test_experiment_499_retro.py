"""Tests for Exp 499 — Milestone 2026.04.37 Retrospective.

Tests cover the helper functions in scripts/experiment_499_retro_2026_04_37.py:
- _count_deferred_to_gpu: correctly identifies blocked/deferred experiments
- _assess_retro_closures: reads closure booleans from result JSONs
- _compute_adoption_rate: counts enforcement items installed
- _build_new_retro_items: generates correct new RETRO items

Spec: REQ-INFRA-057, SCENARIO-INFRA-058
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so scripts/ is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import the module under test
import scripts.experiment_499_retro_2026_04_37 as retro


class TestCountDeferredToGpu:
    """_count_deferred_to_gpu classifies GPU-blocked experiments correctly."""

    def test_deferred_retro_verdict_detected(self):
        # Any verdict in _GPU_DEFERRED_VERDICTS must count as deferred
        results = {
            100: {"honest_verdict": "deferred_retro_033", "status": "blocked"},
            101: {"honest_verdict": "gpu_required", "status": "gpu_required"},
        }
        n, ids = retro._count_deferred_to_gpu(results)
        assert n == 2
        assert 100 in ids
        assert 101 in ids

    def test_successful_experiment_not_deferred(self):
        results = {
            200: {"honest_verdict": "vram_gate_v2_operational", "status": "success"},
            201: {"honest_verdict": "retro_040_closed", "status": "success"},
        }
        n, ids = retro._count_deferred_to_gpu(results)
        assert n == 0
        assert ids == []

    def test_cuda_oom_in_blocked_reason_counts(self):
        # A blocked status with CUDA OOM message counts even without a matching verdict
        results = {
            300: {
                "honest_verdict": "some_other_verdict",
                "status": "blocked",
                "blocked_reason": "CUDA out of memory. Tried to allocate 14 GiB.",
            }
        }
        n, ids = retro._count_deferred_to_gpu(results)
        assert n == 1

    def test_empty_results(self):
        n, ids = retro._count_deferred_to_gpu({})
        assert n == 0
        assert ids == []


class TestAssessRetroClosure:
    """_assess_retro_closures reads correct fields from the right experiment JSONs."""

    def test_all_open_when_results_empty(self):
        closures = retro._assess_retro_closures({})
        assert all(v is False for v in closures.values())
        assert "retro_031_closed" in closures
        assert "retro_047_closed" in closures

    def test_reads_from_correct_exp_id(self):
        # retro_031_closed comes from Exp 498; retro_033_closed from Exp 488
        results = {
            498: {"retro_031_closed": True},
            488: {"retro_033_closed": False},
        }
        closures = retro._assess_retro_closures(results)
        assert closures["retro_031_closed"] is True
        assert closures["retro_033_closed"] is False

    def test_missing_key_defaults_false(self):
        # Exp 490 (retro_039) may not have the key; should default False
        results = {490: {"status": "gpu_required"}}
        closures = retro._assess_retro_closures(results)
        assert closures["retro_039_closed"] is False

    def test_all_closed(self):
        results = {
            498: {"retro_031_closed": True},
            488: {"retro_033_closed": True},
            489: {"retro_038_closed": True},
            490: {"retro_039_closed": True},
            492: {"retro_040_closed": True},
            493: {"retro_045_closed": True},
            494: {"retro_046_closed": True},
            496: {"retro_047_closed": True},
        }
        closures = retro._assess_retro_closures(results)
        assert all(v is True for v in closures.values())


class TestComputeAdoptionRate:
    """_compute_adoption_rate counts enforcement items correctly."""

    def test_all_installed(self):
        results = {
            493: {"retro_045_closed": True},
            494: {"retro_046_closed": True},
            495: {"honest_verdict": "all_patched"},
        }
        rate, detail = retro._compute_adoption_rate(results)
        assert rate == pytest.approx(1.0)
        assert detail["batching_hook"] is True
        assert detail["thermal_gate"] is True
        assert detail["harness_patch"] is True

    def test_none_installed(self):
        results = {}
        rate, detail = retro._compute_adoption_rate(results)
        assert rate == pytest.approx(0.0)
        assert all(v is False for v in detail.values())

    def test_partial_installation(self):
        results = {
            493: {"retro_045_closed": True},
            494: {"retro_046_closed": False},
            495: {"honest_verdict": "partial"},
        }
        rate, detail = retro._compute_adoption_rate(results)
        # Only batching_hook installed
        assert rate == pytest.approx(1 / 3)
        assert detail["batching_hook"] is True
        assert detail["thermal_gate"] is False
        assert detail["harness_patch"] is False

    def test_harness_patch_requires_exact_verdict(self):
        # Only 'all_patched' counts, not similar strings
        results = {495: {"honest_verdict": "partially_patched"}}
        _, detail = retro._compute_adoption_rate(results)
        assert detail["harness_patch"] is False


class TestBuildNewRetroItems:
    """_build_new_retro_items generates the correct set of RETRO items."""

    def _all_open_closures(self) -> dict:
        return {
            "retro_031_closed": False,
            "retro_033_closed": False,
            "retro_038_closed": False,
            "retro_039_closed": False,
            "retro_040_closed": False,
            "retro_045_closed": False,
            "retro_046_closed": False,
            "retro_047_closed": False,
        }

    def test_retro_048_generated_when_deferred_gt_0(self):
        items = retro._build_new_retro_items(3, self._all_open_closures(), {})
        ids = [r["id"] for r in items]
        assert "RETRO-048" in ids
        # RETRO-048 must be CRITICAL priority because it blocks live benchmarks
        r048 = next(r for r in items if r["id"] == "RETRO-048")
        assert r048["priority"] == "CRITICAL"

    def test_retro_048_not_generated_when_deferred_eq_0(self):
        closures = self._all_open_closures()
        closures["retro_033_closed"] = True  # avoid duplicate RETRO-033 item
        closures["retro_038_closed"] = True
        closures["retro_039_closed"] = True
        items = retro._build_new_retro_items(0, closures, {})
        ids = [r["id"] for r in items]
        assert "RETRO-048" not in ids

    def test_retro_049_generated_when_nup_not_closed(self):
        results = {496: {"auc_v2": 0.6, "is_viable_tier_0c": False}}
        closures = self._all_open_closures()
        items = retro._build_new_retro_items(0, closures, results)
        ids = [r["id"] for r in items]
        assert "RETRO-049" in ids

    def test_retro_050_generated_when_sure_not_better(self):
        results = {497: {"sure_better": False, "isolation_improvement": -0.117}}
        closures = self._all_open_closures()
        items = retro._build_new_retro_items(0, closures, results)
        ids = [r["id"] for r in items]
        assert "RETRO-050" in ids

    def test_no_duplicate_ids(self):
        closures = self._all_open_closures()
        results = {
            496: {"auc_v2": 0.6},
            497: {"sure_better": False, "isolation_improvement": -0.1},
        }
        items = retro._build_new_retro_items(3, closures, results)
        ids = [r["id"] for r in items]
        assert len(ids) == len(set(ids)), f"Duplicate RETRO IDs found: {ids}"

    def test_all_items_have_required_fields(self):
        closures = self._all_open_closures()
        results = {
            496: {"auc_v2": 0.6},
            497: {"sure_better": False, "isolation_improvement": -0.1},
        }
        items = retro._build_new_retro_items(3, closures, results)
        for item in items:
            assert "id" in item, f"Missing 'id' in {item}"
            assert "description" in item, f"Missing 'description' in {item}"
            assert "priority" in item, f"Missing 'priority' in {item}"
            assert "target_milestone" in item, f"Missing 'target_milestone' in {item}"
