"""Tests for carnot.pipeline.gpu1_routing_fix.

Spec: REQ-INFRA-071, REQ-INFRA-072, SCENARIO-INFRA-081, SCENARIO-INFRA-082
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from carnot.pipeline.gpu1_routing_fix import (
    GPU1RoutingResult,
    force_cuda1_device_map,
    verify_model_on_device,
)


# ---------------------------------------------------------------------------
# GPU1RoutingResult — SCENARIO-INFRA-081
# ---------------------------------------------------------------------------


class TestGPU1RoutingResult:
    """REQ-INFRA-071: GPU1RoutingResult captures routing outcome faithfully."""

    def test_gpu1_active_verdict(self):
        # SCENARIO-INFRA-081: routing_verified=True → honest_verdict='gpu1_active'
        result = GPU1RoutingResult(
            device_used="cuda:1",
            gpu1_compute_pct_during_inference=45.0,
            routing_verified=True,
            honest_verdict="gpu1_active",
        )
        assert result.routing_verified is True
        assert result.honest_verdict == "gpu1_active"
        assert result.device_used == "cuda:1"
        assert result.gpu1_compute_pct_during_inference == 45.0

    def test_gpu1_still_idle_verdict(self):
        # SCENARIO-INFRA-081: live GPU present but compute=0 → 'gpu1_still_idle'
        result = GPU1RoutingResult(
            device_used="cuda:1",
            gpu1_compute_pct_during_inference=0.0,
            routing_verified=False,
            honest_verdict="gpu1_still_idle",
        )
        assert result.routing_verified is False
        assert result.honest_verdict == "gpu1_still_idle"

    def test_gpu_required_verdict(self):
        # SCENARIO-INFRA-081: no live GPU → 'gpu_required', pct=0.0
        result = GPU1RoutingResult(
            device_used="unknown",
            gpu1_compute_pct_during_inference=0.0,
            routing_verified=False,
            honest_verdict="gpu_required",
        )
        assert result.honest_verdict == "gpu_required"
        assert result.device_used == "unknown"

    def test_routing_verified_threshold(self):
        # routing_verified=True only when caller sets it based on pct > 10.0
        result_above = GPU1RoutingResult(
            device_used="cuda:1",
            gpu1_compute_pct_during_inference=10.1,
            routing_verified=True,
            honest_verdict="gpu1_active",
        )
        result_at = GPU1RoutingResult(
            device_used="cuda:1",
            gpu1_compute_pct_during_inference=10.0,
            routing_verified=False,
            honest_verdict="gpu1_still_idle",
        )
        assert result_above.routing_verified is True
        assert result_at.routing_verified is False


# ---------------------------------------------------------------------------
# force_cuda1_device_map — SCENARIO-INFRA-082
# ---------------------------------------------------------------------------


class TestForceCuda1DeviceMap:
    """REQ-INFRA-072: force_cuda1_device_map pins all layers to cuda:1."""

    def test_empty_string_key_is_cuda1(self):
        # SCENARIO-INFRA-082: the canonical '' sentinel must map to 'cuda:1'
        dm = force_cuda1_device_map("Qwen/Qwen2.5-0.5B")
        assert dm[""] == "cuda:1"

    def test_model_id_embedded_for_traceability(self):
        # The _model_id key must survive for log traceability
        model_id = "Qwen/Qwen2.5-0.5B"
        dm = force_cuda1_device_map(model_id)
        assert dm["_model_id"] == model_id

    def test_never_returns_auto(self):
        # 'auto' is the root cause of RETRO-025; it must never appear
        dm = force_cuda1_device_map("any/model")
        assert "auto" not in dm.values()

    def test_different_model_ids(self):
        # Returns correct map for any model_id string
        dm1 = force_cuda1_device_map("meta-llama/Llama-3-8B")
        dm2 = force_cuda1_device_map("google/gemma-2b")
        assert dm1[""] == "cuda:1"
        assert dm2[""] == "cuda:1"
        assert dm1["_model_id"] != dm2["_model_id"]


# ---------------------------------------------------------------------------
# verify_model_on_device — SCENARIO-INFRA-082
# ---------------------------------------------------------------------------


class TestVerifyModelOnDevice:
    """REQ-INFRA-072: verify_model_on_device checks first-parameter device."""

    def _make_model(self, device_type: str, device_index: int | None) -> MagicMock:
        """Build a MagicMock model whose first parameter has the given device."""
        param = MagicMock()
        param.device = MagicMock()
        param.device.type = device_type
        param.device.index = device_index
        model = MagicMock()
        model.parameters.return_value = iter([param])
        return model

    def test_returns_true_when_param_on_cuda1(self):
        # SCENARIO-INFRA-082: first param on cuda:1 → True
        model = self._make_model("cuda", 1)
        assert verify_model_on_device(model, expected_device_id=1) is True

    def test_returns_false_when_param_on_cuda0(self):
        # First param on cuda:0 when cuda:1 expected → False
        model = self._make_model("cuda", 0)
        assert verify_model_on_device(model, expected_device_id=1) is False

    def test_returns_false_when_param_on_cpu(self):
        # CPU model (type='cpu', index=None) is never on a CUDA device
        model = self._make_model("cpu", None)
        assert verify_model_on_device(model, expected_device_id=1) is False

    def test_returns_false_when_no_parameters(self):
        # StopIteration from empty iterator → False (safe fallback)
        model = MagicMock()
        model.parameters.return_value = iter([])
        assert verify_model_on_device(model, expected_device_id=1) is False

    def test_returns_false_when_parameters_raises(self):
        # AttributeError from a non-standard model object → False
        model = MagicMock()
        model.parameters.side_effect = AttributeError("not a module")
        assert verify_model_on_device(model, expected_device_id=1) is False

    def test_device_index_mismatch_higher_id(self):
        # cuda:2 when expecting cuda:1 → False
        model = self._make_model("cuda", 2)
        assert verify_model_on_device(model, expected_device_id=1) is False

    def test_expected_device_id_zero(self):
        # Works correctly for GPU 0 as well (not only GPU 1)
        model = self._make_model("cuda", 0)
        assert verify_model_on_device(model, expected_device_id=0) is True
