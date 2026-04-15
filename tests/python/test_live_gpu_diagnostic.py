"""Tests for python/carnot/pipeline/live_gpu_diagnostic.py.

Spec: REQ-INFRA-014, SCENARIO-INFRA-014, SCENARIO-INFRA-015

These tests are fully CI-safe: they never require a real GPU, never import torch
or transformers at the module level, and never invoke nvidia-smi without mocking.
Every branch of live_gpu_diagnostic.py must be exercised here (100% coverage).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.live_gpu_diagnostic import (
    LiveGPUDiagnostic,
    _load_tokenizer,
    check_carnot_force_live,
    check_cuda_visible,
    check_model_loadable,
    check_torch_cuda,
    diagnose_live_gpu,
)


# ---------------------------------------------------------------------------
# LiveGPUDiagnostic dataclass
# ---------------------------------------------------------------------------


class TestLiveGPUDiagnostic:
    """REQ-INFRA-014: LiveGPUDiagnostic dataclass contract."""

    def test_fields_present(self):
        # All required fields must be present and correctly typed.
        d = LiveGPUDiagnostic(
            cuda_visible=True,
            torch_available=True,
            model_loadable=True,
            carnot_force_live_set=True,
            failure_reason="",
            is_live_capable=True,
        )
        assert d.cuda_visible is True
        assert d.torch_available is True
        assert d.model_loadable is True
        assert d.carnot_force_live_set is True
        assert d.failure_reason == ""
        assert d.is_live_capable is True

    def test_defaults_for_failure_case(self):
        # Failure case: not all checks pass.
        d = LiveGPUDiagnostic(
            cuda_visible=False,
            torch_available=False,
            model_loadable=False,
            carnot_force_live_set=True,
            failure_reason="cuda_visible: nvidia-smi returned no GPUs",
            is_live_capable=False,
        )
        assert d.is_live_capable is False
        assert "cuda_visible" in d.failure_reason

    def test_checks_passed_and_failed_lists(self):
        # Verify the dataclass is a proper dataclass (fields accessible via __dataclass_fields__).
        import dataclasses

        fields = {f.name for f in dataclasses.fields(LiveGPUDiagnostic)}
        assert "cuda_visible" in fields
        assert "torch_available" in fields
        assert "model_loadable" in fields
        assert "carnot_force_live_set" in fields
        assert "failure_reason" in fields
        assert "is_live_capable" in fields


# ---------------------------------------------------------------------------
# check_cuda_visible
# ---------------------------------------------------------------------------


class TestCheckCudaVisible:
    """REQ-INFRA-014: check_cuda_visible() — subprocess nvidia-smi, returns bool."""

    def test_returns_true_when_nvidia_smi_succeeds_with_gpu(self):
        # nvidia-smi exits 0 and reports at least one GPU line.
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "GPU 00000000:01:00.0\n"
        with patch("subprocess.run", return_value=mock_result):
            assert check_cuda_visible() is True

    def test_returns_false_when_nvidia_smi_nonzero(self):
        # nvidia-smi exits non-zero (driver not loaded or no GPU).
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            assert check_cuda_visible() is False

    def test_returns_false_when_nvidia_smi_not_found(self):
        # FileNotFoundError when nvidia-smi is not installed.
        with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
            assert check_cuda_visible() is False

    def test_returns_false_on_subprocess_timeout(self):
        # Timeout should not propagate — must return False gracefully.
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5),
        ):
            assert check_cuda_visible() is False

    def test_returns_false_on_unexpected_exception(self):
        # Any unexpected OS error must be caught and return False.
        with patch("subprocess.run", side_effect=OSError("unexpected")):
            assert check_cuda_visible() is False


# ---------------------------------------------------------------------------
# check_torch_cuda
# ---------------------------------------------------------------------------


class TestCheckTorchCuda:
    """REQ-INFRA-014: check_torch_cuda() — import torch; torch.cuda.is_available()."""

    def test_returns_true_when_torch_cuda_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert check_torch_cuda() is True

    def test_returns_false_when_torch_cuda_not_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert check_torch_cuda() is False

    def test_returns_false_when_torch_not_installed(self):
        # ImportError when torch is not installed.
        with patch.dict("sys.modules", {"torch": None}):
            # When the module is None in sys.modules, import raises ImportError.
            assert check_torch_cuda() is False

    def test_returns_false_on_unexpected_exception_in_torch(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA init failed")
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert check_torch_cuda() is False


# ---------------------------------------------------------------------------
# check_carnot_force_live
# ---------------------------------------------------------------------------


class TestCheckCarnotForceLive:
    """REQ-INFRA-014: check_carnot_force_live() — os.environ check."""

    def test_returns_true_when_set_to_1(self, monkeypatch):
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        assert check_carnot_force_live() is True

    def test_returns_false_when_set_to_0(self, monkeypatch):
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
        assert check_carnot_force_live() is False

    def test_returns_false_when_not_set(self, monkeypatch):
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        assert check_carnot_force_live() is False

    def test_returns_false_when_set_to_other_value(self, monkeypatch):
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "true")
        assert check_carnot_force_live() is False


# ---------------------------------------------------------------------------
# check_model_loadable
# ---------------------------------------------------------------------------


class TestCheckModelLoadable:
    """REQ-INFRA-014: check_model_loadable(model_id) — attempts AutoTokenizer.from_pretrained."""

    def test_returns_true_when_model_loads(self):
        mock_tokenizer_cls = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        with patch(
            "carnot.pipeline.live_gpu_diagnostic._load_tokenizer",
            return_value=(True, ""),
        ):
            loadable, err = check_model_loadable("Qwen/Qwen3.5-0.8B")
        assert loadable is True
        assert err == ""

    def test_returns_false_when_model_not_found(self):
        with patch(
            "carnot.pipeline.live_gpu_diagnostic._load_tokenizer",
            return_value=(False, "OSError: model not found"),
        ):
            loadable, err = check_model_loadable("nonexistent/model-xyz")
        assert loadable is False
        assert "OSError" in err or err != ""

    def test_returns_false_on_timeout(self):
        # Simulate timeout returning False with an error message.
        with patch(
            "carnot.pipeline.live_gpu_diagnostic._load_tokenizer",
            return_value=(False, "timeout after 30s"),
        ):
            loadable, err = check_model_loadable("slow/model", timeout_s=30)
        assert loadable is False
        assert "timeout" in err

    def test_accepts_timeout_parameter(self):
        # timeout_s parameter is accepted without error.
        with patch(
            "carnot.pipeline.live_gpu_diagnostic._load_tokenizer",
            return_value=(True, ""),
        ):
            loadable, err = check_model_loadable("some/model", timeout_s=10)
        assert loadable is True

    def test_check_model_loadable_handles_exception_from_load_tokenizer(self):
        # check_model_loadable must catch any exception from _load_tokenizer.
        with patch(
            "carnot.pipeline.live_gpu_diagnostic._load_tokenizer",
            side_effect=RuntimeError("unexpected internal error"),
        ):
            loadable, err = check_model_loadable("some/model")
        assert loadable is False
        assert "RuntimeError" in err


# ---------------------------------------------------------------------------
# _load_tokenizer direct tests (covers the real code paths inside the function)
# ---------------------------------------------------------------------------


class TestLoadTokenizer:
    """Cover _load_tokenizer's real execution paths (lines 164-178)."""

    def test_success_path_returns_true(self):
        # Patch transformers.AutoTokenizer so we don't need the package installed.
        mock_tokenizer = MagicMock()
        mock_auto = MagicMock()
        mock_auto.from_pretrained.return_value = mock_tokenizer
        with patch.dict("sys.modules", {"transformers": MagicMock(AutoTokenizer=mock_auto)}):
            loadable, err = _load_tokenizer("some/model", timeout_s=5.0)
        assert loadable is True
        assert err == ""

    def test_exception_in_attempt_returns_false(self):
        # When AutoTokenizer.from_pretrained raises, _attempt catches and returns False.
        mock_auto = MagicMock()
        mock_auto.from_pretrained.side_effect = OSError("model not found")
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer = mock_auto
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            loadable, err = _load_tokenizer("bad/model", timeout_s=5.0)
        assert loadable is False
        assert "OSError" in err

    def test_timeout_returns_false_with_timeout_message(self):
        # Simulate a slow tokenizer load that exceeds the timeout.
        import time

        def _slow(*_args, **_kwargs):
            time.sleep(10)  # will be interrupted by timeout

        mock_auto = MagicMock()
        mock_auto.from_pretrained.side_effect = _slow
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer = mock_auto
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            loadable, err = _load_tokenizer("slow/model", timeout_s=0.05)
        assert loadable is False
        assert "timeout" in err


# ---------------------------------------------------------------------------
# diagnose_live_gpu
# ---------------------------------------------------------------------------


class TestDiagnoseGpu:
    """REQ-INFRA-014: diagnose_live_gpu() — full layer-by-layer diagnostic."""

    def _patch_all_ok(self, monkeypatch):
        """Helper: patch all checks to return success."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        with (
            patch("carnot.pipeline.live_gpu_diagnostic.check_cuda_visible", return_value=True),
            patch("carnot.pipeline.live_gpu_diagnostic.check_torch_cuda", return_value=True),
            patch(
                "carnot.pipeline.live_gpu_diagnostic.check_model_loadable",
                return_value=(True, ""),
            ),
        ):
            yield

    def test_all_checks_pass_returns_live_capable(self, monkeypatch):
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        with (
            patch("carnot.pipeline.live_gpu_diagnostic.check_cuda_visible", return_value=True),
            patch("carnot.pipeline.live_gpu_diagnostic.check_torch_cuda", return_value=True),
            patch(
                "carnot.pipeline.live_gpu_diagnostic.check_model_loadable",
                return_value=(True, ""),
            ),
        ):
            result = diagnose_live_gpu(["Qwen/Qwen3.5-0.8B"])
        assert result.is_live_capable is True
        assert result.cuda_visible is True
        assert result.torch_available is True
        assert result.model_loadable is True
        assert result.carnot_force_live_set is True
        assert result.failure_reason == ""

    def test_cuda_not_visible_stops_early(self, monkeypatch):
        # SCENARIO-INFRA-014: cuda_visible=False → failure_reason contains "cuda_visible"
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        with (
            patch("carnot.pipeline.live_gpu_diagnostic.check_cuda_visible", return_value=False),
            patch("carnot.pipeline.live_gpu_diagnostic.check_torch_cuda", return_value=True),
            patch(
                "carnot.pipeline.live_gpu_diagnostic.check_model_loadable",
                return_value=(True, ""),
            ),
        ):
            result = diagnose_live_gpu(["Qwen/Qwen3.5-0.8B"])
        assert result.is_live_capable is False
        assert result.cuda_visible is False
        assert "cuda_visible" in result.failure_reason

    def test_torch_cuda_unavailable_reported(self, monkeypatch):
        # SCENARIO-INFRA-014: torch_available=False
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        with (
            patch("carnot.pipeline.live_gpu_diagnostic.check_cuda_visible", return_value=True),
            patch("carnot.pipeline.live_gpu_diagnostic.check_torch_cuda", return_value=False),
            patch(
                "carnot.pipeline.live_gpu_diagnostic.check_model_loadable",
                return_value=(True, ""),
            ),
        ):
            result = diagnose_live_gpu(["Qwen/Qwen3.5-0.8B"])
        assert result.is_live_capable is False
        assert result.torch_available is False
        assert "torch_cuda" in result.failure_reason

    def test_model_not_loadable_reported(self, monkeypatch):
        # SCENARIO-INFRA-014: model_loadable=False
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        with (
            patch("carnot.pipeline.live_gpu_diagnostic.check_cuda_visible", return_value=True),
            patch("carnot.pipeline.live_gpu_diagnostic.check_torch_cuda", return_value=True),
            patch(
                "carnot.pipeline.live_gpu_diagnostic.check_model_loadable",
                return_value=(False, "OSError: repo not found"),
            ),
        ):
            result = diagnose_live_gpu(["nonexistent/model-xyz"])
        assert result.is_live_capable is False
        assert result.model_loadable is False
        assert "model_loadable" in result.failure_reason

    def test_carnot_force_live_not_set_still_runs_but_noted(self, monkeypatch):
        # When CARNOT_FORCE_LIVE is not set, carnot_force_live_set=False but no error.
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        with (
            patch("carnot.pipeline.live_gpu_diagnostic.check_cuda_visible", return_value=True),
            patch("carnot.pipeline.live_gpu_diagnostic.check_torch_cuda", return_value=True),
            patch(
                "carnot.pipeline.live_gpu_diagnostic.check_model_loadable",
                return_value=(True, ""),
            ),
        ):
            result = diagnose_live_gpu(["Qwen/Qwen3.5-0.8B"])
        assert result.carnot_force_live_set is False
        # is_live_capable requires force_live to be unambiguous — depends on impl;
        # what matters is the field is populated correctly.

    def test_no_model_ids_skips_model_check(self, monkeypatch):
        # When model_ids=[] or None, model_loadable should default True (no models to check).
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        with (
            patch("carnot.pipeline.live_gpu_diagnostic.check_cuda_visible", return_value=True),
            patch("carnot.pipeline.live_gpu_diagnostic.check_torch_cuda", return_value=True),
        ):
            result = diagnose_live_gpu([])
        assert result.model_loadable is True

    def test_multiple_models_first_failure_reported(self, monkeypatch):
        # If there are 2 models and the first loads but the second does not,
        # model_loadable=False and failure_reason mentions the failing model.
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        with (
            patch("carnot.pipeline.live_gpu_diagnostic.check_cuda_visible", return_value=True),
            patch("carnot.pipeline.live_gpu_diagnostic.check_torch_cuda", return_value=True),
            patch(
                "carnot.pipeline.live_gpu_diagnostic.check_model_loadable",
                side_effect=[(True, ""), (False, "not found")],
            ),
        ):
            result = diagnose_live_gpu(["good/model", "bad/model"])
        assert result.model_loadable is False
        assert result.is_live_capable is False

    def test_never_raises_on_unexpected_exception(self, monkeypatch):
        # CI-safe: even if an internal check raises, diagnose_live_gpu must not propagate.
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        with patch(
            "carnot.pipeline.live_gpu_diagnostic.check_cuda_visible",
            side_effect=RuntimeError("boom"),
        ):
            result = diagnose_live_gpu(["Qwen/Qwen3.5-0.8B"])
        assert isinstance(result, LiveGPUDiagnostic)
        assert result.is_live_capable is False
        assert result.failure_reason != ""

    def test_default_model_ids_none_uses_empty_list(self, monkeypatch):
        # diagnose_live_gpu() with no argument (model_ids=None) should work.
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        with (
            patch("carnot.pipeline.live_gpu_diagnostic.check_cuda_visible", return_value=True),
            patch("carnot.pipeline.live_gpu_diagnostic.check_torch_cuda", return_value=True),
        ):
            result = diagnose_live_gpu()
        assert isinstance(result, LiveGPUDiagnostic)


# ---------------------------------------------------------------------------
# ExperimentTemplate.setup_gpu() integration — SCENARIO-INFRA-015
# ---------------------------------------------------------------------------


class TestSetupGpuRaisesOnForceLive:
    """SCENARIO-INFRA-015: setup_gpu() raises RuntimeError when CARNOT_FORCE_LIVE=1 and unhealthy."""

    def _make_template(self, tmp_path):
        from scripts.experiment_template import ExperimentTemplate

        tmpl = ExperimentTemplate(
            exp_id=9999,
            title="test",
            deliverable="results/test_9999.json",
            repo_root=tmp_path,
        )
        tmpl.setup()
        return tmpl

    def _failing_prewarm(self, name, hf_id, gpu):
        """Always returns an unhealthy prewarm result."""
        result = MagicMock()
        result.health_ok = False
        result.load_time_s = 0.1
        result.stall_root_cause = "mock failure"
        return result

    def _healthy_prewarm(self, name, hf_id, gpu):
        result = MagicMock()
        result.health_ok = True
        result.load_time_s = 0.1
        result.stall_root_cause = None
        return result

    def test_raises_when_force_live_and_all_unhealthy(self, tmp_path, monkeypatch):
        """SCENARIO-INFRA-015: RuntimeError raised when CARNOT_FORCE_LIVE=1 and all models fail."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        tmpl = self._make_template(tmp_path)

        mock_diagnostic = LiveGPUDiagnostic(
            cuda_visible=False,
            torch_available=False,
            model_loadable=False,
            carnot_force_live_set=True,
            failure_reason="cuda_visible: no GPUs detected",
            is_live_capable=False,
        )
        with patch(
            "carnot.pipeline.live_gpu_diagnostic.diagnose_live_gpu", return_value=mock_diagnostic
        ):
            with pytest.raises(RuntimeError, match="Live GPU required but unavailable"):
                tmpl.setup_gpu(
                    [{"name": "TestModel", "hf_id": "test/model", "gpu": 0}],
                    prewarm_fn=self._failing_prewarm,
                )

    def test_no_raise_when_force_live_0_and_unhealthy(self, tmp_path, monkeypatch):
        """SCENARIO-INFRA-015: No exception when CARNOT_FORCE_LIVE=0, even if models fail."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
        tmpl = self._make_template(tmp_path)

        result = tmpl.setup_gpu(
            [{"name": "TestModel", "hf_id": "test/model", "gpu": 0}],
            prewarm_fn=self._failing_prewarm,
        )
        assert result["all_healthy"] is False

    def test_no_raise_when_force_live_and_all_healthy(self, tmp_path, monkeypatch):
        """SCENARIO-INFRA-015: No exception when CARNOT_FORCE_LIVE=1 but all models healthy."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        tmpl = self._make_template(tmp_path)

        mock_diagnostic = LiveGPUDiagnostic(
            cuda_visible=True,
            torch_available=True,
            model_loadable=True,
            carnot_force_live_set=True,
            failure_reason="",
            is_live_capable=True,
        )
        with patch(
            "carnot.pipeline.dual_gpu_monitor.DualGPUMonitor"
        ) as mock_monitor_cls, patch(
            "carnot.pipeline.live_gpu_diagnostic.diagnose_live_gpu", return_value=mock_diagnostic
        ):
            mock_monitor = MagicMock()
            mock_monitor._get_gpu_count.return_value = 2
            mock_monitor.check_dual_gpu_health.return_value = {
                "all_healthy": True,
                "n_gpus_detected": 2,
                "n_zombies": 0,
                "idle_gpus": [],
            }
            mock_monitor_cls.return_value = mock_monitor

            result = tmpl.setup_gpu(
                [{"name": "TestModel", "hf_id": "test/model", "gpu": 0}],
                prewarm_fn=self._healthy_prewarm,
            )
        assert result["all_healthy"] is True

    def test_error_message_contains_failure_reason(self, tmp_path, monkeypatch):
        """Error message must contain the failure_reason from the diagnostic."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        tmpl = self._make_template(tmp_path)

        mock_diagnostic = LiveGPUDiagnostic(
            cuda_visible=True,
            torch_available=False,
            model_loadable=False,
            carnot_force_live_set=True,
            failure_reason="torch_cuda: torch.cuda.is_available() returned False",
            is_live_capable=False,
        )
        with patch(
            "carnot.pipeline.live_gpu_diagnostic.diagnose_live_gpu", return_value=mock_diagnostic
        ):
            with pytest.raises(RuntimeError) as exc_info:
                tmpl.setup_gpu(
                    [{"name": "TestModel", "hf_id": "test/model", "gpu": 0}],
                    prewarm_fn=self._failing_prewarm,
                )
        assert "torch_cuda" in str(exc_info.value)
