"""Tests for DualGPUAssigner.

Spec: REQ-INFRA-034, SCENARIO-INFRA-042
"""

from __future__ import annotations

import os

import pytest

from carnot.pipeline.dual_gpu_assigner import DualGPUAssigner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _specs(n: int) -> list[dict]:
    return [{"name": f"model_{i}", "hf_id": f"org/model-{i}"} for i in range(n)]


# ---------------------------------------------------------------------------
# is_dual_gpu_eligible()
# ---------------------------------------------------------------------------


class TestDualGPUAssignerEligibility:
    """REQ-INFRA-034: eligibility checks."""

    def test_not_eligible_in_ci_no_force_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SCENARIO-INFRA-042: CI mode — CARNOT_FORCE_LIVE not set, eligible=False."""
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        assigner = DualGPUAssigner(_specs(2), n_gpus=2)
        assert assigner.is_dual_gpu_eligible() is False

    def test_not_eligible_when_only_one_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        assigner = DualGPUAssigner(_specs(2), n_gpus=1)
        assert assigner.is_dual_gpu_eligible() is False

    def test_not_eligible_when_fewer_than_two_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        assigner = DualGPUAssigner(_specs(1), n_gpus=2)
        assert assigner.is_dual_gpu_eligible() is False

    def test_eligible_with_two_models_two_gpus_and_force_live(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        assigner = DualGPUAssigner(_specs(2), n_gpus=2)
        assert assigner.is_dual_gpu_eligible() is True

    def test_not_eligible_zero_gpus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        assigner = DualGPUAssigner(_specs(2), n_gpus=0)
        assert assigner.is_dual_gpu_eligible() is False


# ---------------------------------------------------------------------------
# assign()
# ---------------------------------------------------------------------------


class TestDualGPUAssignerAssign:
    """REQ-INFRA-034: assignment sets cuda:0 for first model, cuda:1 for second."""

    def test_assign_sets_gpu_and_device_map(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SCENARIO-INFRA-042: first model gets cuda:0, second gets cuda:1."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        specs = _specs(2)
        assigner = DualGPUAssigner(specs, n_gpus=2)
        result = assigner.assign()
        assert result[0]["gpu"] == 0
        assert result[0]["device_map"] == {"": "cuda:0"}
        assert result[1]["gpu"] == 1
        assert result[1]["device_map"] == {"": "cuda:1"}

    def test_assign_returns_same_list_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """assign() mutates and returns the original list, not a copy."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        specs = _specs(2)
        assigner = DualGPUAssigner(specs, n_gpus=2)
        result = assigner.assign()
        assert result is specs

    def test_assign_noop_when_not_eligible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When not eligible, assign() returns specs unchanged."""
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        specs = _specs(2)
        assigner = DualGPUAssigner(specs, n_gpus=2)
        result = assigner.assign()
        assert "gpu" not in result[0]
        assert "device_map" not in result[0]

    def test_assign_three_models_two_gpus_caps_at_last_gpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Third model is capped to GPU 1 (last available) with a warning."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        specs = _specs(3)
        assigner = DualGPUAssigner(specs, n_gpus=2)
        result = assigner.assign()
        assert result[2]["gpu"] == 1
        assert result[2]["device_map"] == {"": "cuda:1"}
