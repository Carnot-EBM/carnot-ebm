"""Tests for Exp 632: DualGPU 13B Forward Pass Proof.

100% targeted coverage on functions added in
scripts/experiment_632_dualgpu_13b_proof.py:
  - detect_gpus()
  - load_model()
  - sample_utilization()
  - run_forward_passes()
  - _llama_cpp_forward_passes()
  - run_experiment()

Tests run without GPU hardware by mocking torch and transformers.

Spec: REQ-INFRA-089, SCENARIO-INFRA-094, SCENARIO-INFRA-095
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# Suppress GPU assertion in CI.
os.environ["CARNOT_IS_CI"] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import scripts.experiment_632_dualgpu_13b_proof as exp632  # noqa: E402


# ---------------------------------------------------------------------------
# detect_gpus
# ---------------------------------------------------------------------------


class TestDetectGpus:
    """REQ-INFRA-089-1: GPU count and VRAM detected before model loading."""

    def test_torch_not_importable(self) -> None:
        # SCENARIO: torch is unavailable — returns (0, 0.0, 0.0).
        with patch.dict(sys.modules, {"torch": None}):
            n, v0, v1 = exp632.detect_gpus()
        assert n == 0
        assert v0 == 0.0
        assert v1 == 0.0

    def test_cuda_not_available(self) -> None:
        # torch present but CUDA not available.
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": torch_mock}):
            n, v0, v1 = exp632.detect_gpus()
        assert n == 0

    def test_one_gpu(self) -> None:
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.device_count.return_value = 1
        props = MagicMock()
        props.total_memory = 24 * (1024 ** 3)
        torch_mock.cuda.get_device_properties.return_value = props
        with patch.dict(sys.modules, {"torch": torch_mock}):
            n, v0, v1 = exp632.detect_gpus()
        assert n == 1
        assert v0 == pytest.approx(24.0, abs=0.1)
        assert v1 == 0.0

    def test_two_gpus(self) -> None:
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.device_count.return_value = 2
        props = MagicMock()
        props.total_memory = 24 * (1024 ** 3)
        torch_mock.cuda.get_device_properties.return_value = props
        with patch.dict(sys.modules, {"torch": torch_mock}):
            n, v0, v1 = exp632.detect_gpus()
        assert n == 2
        assert v0 == pytest.approx(24.0, abs=0.1)
        assert v1 == pytest.approx(24.0, abs=0.1)

    def test_zero_gpus_is_available_returns_false(self) -> None:
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.device_count.return_value = 0
        with patch.dict(sys.modules, {"torch": torch_mock}):
            n, v0, v1 = exp632.detect_gpus()
        assert n == 0


# ---------------------------------------------------------------------------
# sample_utilization
# ---------------------------------------------------------------------------


class TestSampleUtilization:
    """REQ-INFRA-089-5: utilization sampled after each forward pass."""

    def test_pynvml_path_success(self) -> None:
        # pynvml available and returns 75%.
        pynvml_mock = MagicMock()
        handle = MagicMock()
        rates = MagicMock()
        rates.gpu = 75
        pynvml_mock.nvmlDeviceGetHandleByIndex.return_value = handle
        pynvml_mock.nvmlDeviceGetUtilizationRates.return_value = rates
        with patch.dict(sys.modules, {"pynvml": pynvml_mock}):
            val = exp632.sample_utilization(0)
        assert val == 75.0

    def test_pynvml_fails_torch_fallback(self) -> None:
        # pynvml raises, fall back to torch.cuda.utilization.
        pynvml_mock = MagicMock()
        pynvml_mock.nvmlDeviceGetHandleByIndex.side_effect = RuntimeError("nvml error")
        torch_mock = MagicMock()
        torch_mock.cuda.utilization.return_value = 50
        with patch.dict(sys.modules, {"pynvml": pynvml_mock, "torch": torch_mock}):
            val = exp632.sample_utilization(1)
        assert val == 50.0

    def test_both_fail_returns_sentinel(self) -> None:
        # Both pynvml and torch fail — returns -1.0.
        pynvml_mock = MagicMock()
        pynvml_mock.nvmlInit.side_effect = Exception("no nvml")
        torch_mock = MagicMock()
        torch_mock.cuda.utilization.side_effect = Exception("no torch")
        with patch.dict(sys.modules, {"pynvml": pynvml_mock, "torch": torch_mock}):
            val = exp632.sample_utilization(0)
        assert val == -1.0

    def test_pynvml_not_installed_torch_used(self) -> None:
        # pynvml not importable at all — torch path used.
        torch_mock = MagicMock()
        torch_mock.cuda.utilization.return_value = 30
        with patch.dict(sys.modules, {"pynvml": None, "torch": torch_mock}):
            val = exp632.sample_utilization(0)
        assert val == 30.0


# ---------------------------------------------------------------------------
# run_forward_passes
# ---------------------------------------------------------------------------


class TestRunForwardPasses:
    """REQ-INFRA-089-4: 10 forward passes run with utilization sampled after each."""

    def test_tokenizer_fails_returns_empty(self) -> None:
        # If tokenizer setup fails, returns empty lists — no crash.
        model_mock = MagicMock()
        transformers_mock = MagicMock()
        transformers_mock.AutoTokenizer.from_pretrained.side_effect = RuntimeError("no tok")
        with patch.dict(sys.modules, {"transformers": transformers_mock}):
            u0, u1 = exp632.run_forward_passes(model_mock, "fake/model", n_passes=3)
        assert u0 == []
        assert u1 == []

    def test_successful_passes_returns_lists(self) -> None:
        # Happy path: model.generate() succeeds, utilization sampled.
        model_mock = MagicMock()
        model_mock.generate.return_value = MagicMock()

        # MagicMock as the inputs dict — __getitem__ auto-returns a MagicMock
        # whose .to() also returns a MagicMock, which is what the code needs.
        inputs_mock = MagicMock()
        tok_mock = MagicMock()
        tok_mock.return_value = inputs_mock

        torch_mock = MagicMock()
        transformers_mock = MagicMock()
        transformers_mock.AutoTokenizer.from_pretrained.return_value = tok_mock

        with (
            patch.dict(sys.modules, {"transformers": transformers_mock, "torch": torch_mock}),
            patch.object(exp632, "sample_utilization", side_effect=[60.0, 70.0] * 10),
        ):
            u0, u1 = exp632.run_forward_passes(model_mock, "fake/model", n_passes=3)

        # Should have 3 entries each (one per pass).
        assert len(u0) == 3
        assert len(u1) == 3

    def test_generate_exception_records_sentinel(self) -> None:
        # generate() raises — pass records -1.0 for both GPUs, doesn't crash.
        model_mock = MagicMock()
        model_mock.generate.side_effect = RuntimeError("CUDA OOM")

        tok_mock = MagicMock()
        tensor_mock = MagicMock()
        tensor_mock.to.return_value = tensor_mock
        tok_mock.return_value = {"input_ids": tensor_mock, "attention_mask": tensor_mock}

        torch_mock = MagicMock()
        # Make the context manager no_grad() work
        torch_mock.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        torch_mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        transformers_mock = MagicMock()
        transformers_mock.AutoTokenizer.from_pretrained.return_value = tok_mock

        with patch.dict(sys.modules, {"transformers": transformers_mock, "torch": torch_mock}):
            u0, u1 = exp632.run_forward_passes(model_mock, "fake/model", n_passes=2)

        assert len(u0) == 2
        assert all(v == -1.0 for v in u0)


# ---------------------------------------------------------------------------
# _llama_cpp_forward_passes
# ---------------------------------------------------------------------------


class TestLlamaCppForwardPasses:
    """REQ-INFRA-089-3: llama-cpp fallback records utilization correctly."""

    def test_success_records_utilization(self) -> None:
        model_mock = MagicMock()
        model_mock.return_value = {"choices": [{"text": "Paris"}]}

        with patch.object(exp632, "sample_utilization", side_effect=[55.0, 65.0] * 5):
            u0, u1 = exp632._llama_cpp_forward_passes(model_mock, n_passes=5)

        assert len(u0) == 5
        assert len(u1) == 5

    def test_failure_records_sentinel(self) -> None:
        model_mock = MagicMock()
        model_mock.side_effect = RuntimeError("llama error")

        u0, u1 = exp632._llama_cpp_forward_passes(model_mock, n_passes=3)
        assert all(v == -1.0 for v in u0)
        assert all(v == -1.0 for v in u1)


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    """REQ-INFRA-089-2: load_model tries options in order and stops at first success."""

    def test_14b_loaded_when_vram_sufficient(self) -> None:
        # 50 GB VRAM -> attempts 14B first; if success returns 14B model.
        fake_model = MagicMock()
        with patch.object(exp632, "_try_transformers_auto", return_value=(fake_model, None)) as m:
            model, name, size, blocked = exp632.load_model(50.0)
        assert model is fake_model
        assert "14B" in name or "14b" in name.lower()
        assert size == 14
        assert blocked == []

    def test_7b_fallback_when_14b_fails(self) -> None:
        # 14B fails, 7B auto succeeds.
        fake_model = MagicMock()

        def fake_auto(model_name: str):  # type: ignore[return]
            if "14B" in model_name:
                return None, "transformers_auto_failed:OOM"
            return fake_model, None

        with patch.object(exp632, "_try_transformers_auto", side_effect=fake_auto):
            model, name, size, blocked = exp632.load_model(50.0)

        assert model is fake_model
        assert "7B" in name
        assert size == 7
        assert len(blocked) == 1  # 14B failure recorded

    def test_explicit_fallback_when_auto_fails(self) -> None:
        # Both auto attempts fail, explicit split succeeds.
        fake_model = MagicMock()
        with (
            patch.object(exp632, "_try_transformers_auto", return_value=(None, "auto_fail")),
            patch.object(exp632, "_try_transformers_explicit", return_value=(fake_model, None)),
        ):
            model, name, size, blocked = exp632.load_model(50.0)

        assert model is fake_model
        assert size == 7

    def test_all_fail_returns_none(self) -> None:
        # All methods fail — no GGUF on disk.
        # Patch Path.glob at the class level so both _REPO_ROOT.glob and
        # Path("/tmp").glob return empty lists without touching PosixPath slots.
        with (
            patch.object(exp632, "_try_transformers_auto", return_value=(None, "auto_fail")),
            patch.object(exp632, "_try_transformers_explicit", return_value=(None, "explicit_fail")),
            patch("pathlib.Path.glob", return_value=iter([])),
        ):
            model, name, size, blocked = exp632.load_model(50.0)

        assert model is None
        assert name == ""
        assert size is None
        assert len(blocked) >= 2

    def test_low_vram_skips_14b(self) -> None:
        # < 30 GB VRAM -> skips 14B and goes straight to 7B.
        fake_model = MagicMock()
        with patch.object(exp632, "_try_transformers_auto", return_value=(fake_model, None)) as m:
            model, name, size, blocked = exp632.load_model(20.0)

        # First call should be for 7B (not 14B)
        first_call_name = m.call_args_list[0][0][0]
        assert "7B" in first_call_name
        assert size == 7


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------


class TestRunExperiment:
    """REQ-INFRA-089: run_experiment produces all required schema fields."""

    def test_no_cuda_gpus(self) -> None:
        # SCENARIO: no CUDA GPUs -> blocked with no_cuda_gpus.
        with patch.object(exp632, "detect_gpus", return_value=(0, 0.0, 0.0)):
            result = exp632.run_experiment()

        assert result["n_gpus"] == 0
        assert result["model_loaded"] is False
        assert result["dualgpu_proven"] is False
        assert result["retro_071_resolved"] is False
        assert result["honest_verdict"] == "dualgpu_model_load_failed"

    def test_only_one_gpu(self) -> None:
        with patch.object(exp632, "detect_gpus", return_value=(1, 24.0, 0.0)):
            result = exp632.run_experiment()

        assert result["blocked_reason"] == "only_one_gpu"
        assert result["model_loaded"] is False

    def test_model_load_failed(self) -> None:
        # 2 GPUs but all model loading options fail.
        with (
            patch.object(exp632, "detect_gpus", return_value=(2, 24.0, 24.0)),
            patch.object(exp632, "load_model", return_value=(None, "", None, ["auto_fail", "explicit_fail", "llama_cpp_no_gguf_found"])),
        ):
            result = exp632.run_experiment()

        assert result["model_loaded"] is False
        assert result["honest_verdict"] == "dualgpu_model_load_failed"
        assert result["dualgpu_proven"] is False

    def test_model_loaded_high_util(self) -> None:
        # Model loaded, GPU-1 peaks at 75% -> dualgpu_proven=True.
        fake_model = MagicMock()
        fake_model.generate = MagicMock()  # has generate -> not llama_cpp

        with (
            patch.object(exp632, "detect_gpus", return_value=(2, 24.0, 24.0)),
            patch.object(exp632, "load_model", return_value=(fake_model, "Qwen/Qwen2.5-7B-Instruct", 7, [])),
            patch.object(exp632, "sample_utilization", return_value=75.0),
            patch.object(exp632, "run_forward_passes", return_value=([80.0] * 10, [75.0] * 10)),
        ):
            result = exp632.run_experiment()

        assert result["model_loaded"] is True
        assert result["peak_gpu1_util"] == 75.0
        assert result["dualgpu_proven"] is True
        assert result["retro_071_resolved"] is True
        assert result["honest_verdict"] == "dualgpu_proven"

    def test_model_loaded_low_util(self) -> None:
        # Model loaded but GPU-1 stays below threshold -> low_util verdict.
        fake_model = MagicMock()
        fake_model.generate = MagicMock()

        with (
            patch.object(exp632, "detect_gpus", return_value=(2, 24.0, 24.0)),
            patch.object(exp632, "load_model", return_value=(fake_model, "Qwen/Qwen2.5-7B-Instruct", 7, [])),
            patch.object(exp632, "sample_utilization", return_value=0.0),
            patch.object(exp632, "run_forward_passes", return_value=([30.0] * 10, [5.0] * 10)),
        ):
            result = exp632.run_experiment()

        assert result["model_loaded"] is True
        assert result["dualgpu_proven"] is False
        assert result["honest_verdict"] == "dualgpu_loaded_low_util"
        assert result["retro_071_resolved"] is False

    def test_llama_cpp_path_used_when_no_generate(self) -> None:
        # Model without .generate attribute -> llama-cpp path.
        fake_model = MagicMock(spec=[])  # no generate attribute
        assert not hasattr(fake_model, "generate")

        with (
            patch.object(exp632, "detect_gpus", return_value=(2, 24.0, 24.0)),
            patch.object(exp632, "load_model", return_value=(fake_model, "local.gguf", None, [])),
            patch.object(exp632, "sample_utilization", return_value=0.0),
            patch.object(exp632, "_llama_cpp_forward_passes", return_value=([60.0] * 10, [80.0] * 10)),
        ):
            result = exp632.run_experiment()

        assert result["model_loaded"] is True
        assert result["peak_gpu1_util"] == 80.0
        assert result["dualgpu_proven"] is True

    def test_sustained_fraction_path(self) -> None:
        # peak_gpu1_util stays at 40% but 8/10 passes > 10% -> sustained_gpu1_fraction=0.8 -> proven.
        fake_model = MagicMock()
        fake_model.generate = MagicMock()

        util_1 = [40.0] * 8 + [5.0] * 2  # 8 passes above 10%, 2 below

        with (
            patch.object(exp632, "detect_gpus", return_value=(2, 24.0, 24.0)),
            patch.object(exp632, "load_model", return_value=(fake_model, "Qwen/Qwen2.5-7B-Instruct", 7, [])),
            patch.object(exp632, "sample_utilization", return_value=0.0),
            patch.object(exp632, "run_forward_passes", return_value=([50.0] * 10, util_1)),
        ):
            result = exp632.run_experiment()

        assert result["sustained_gpu1_fraction"] == pytest.approx(0.8)
        assert result["dualgpu_proven"] is True

    def test_all_required_fields_present(self) -> None:
        # REQ-INFRA-089-9: all required fields must be present in artifact.
        with patch.object(exp632, "detect_gpus", return_value=(0, 0.0, 0.0)):
            result = exp632.run_experiment()

        required = [
            "n_gpus", "vram_0_gb", "vram_1_gb",
            "model_loaded", "model_name", "model_size_B",
            "peak_gpu0_util", "peak_gpu1_util",
            "sustained_gpu1_fraction", "dualgpu_proven",
            "retro_071_resolved", "honest_verdict",
        ]
        for field in required:
            assert field in result, f"Missing required field: {field}"

    def test_cleanup_called_after_passes(self) -> None:
        # del model + torch.cuda.empty_cache() must be called after passes — no leak.
        fake_model = MagicMock()
        fake_model.generate = MagicMock()

        torch_mock = MagicMock()

        with (
            patch.object(exp632, "detect_gpus", return_value=(2, 24.0, 24.0)),
            patch.object(exp632, "load_model", return_value=(fake_model, "Qwen/Qwen2.5-7B-Instruct", 7, [])),
            patch.object(exp632, "sample_utilization", return_value=0.0),
            patch.object(exp632, "run_forward_passes", return_value=([0.0] * 10, [0.0] * 10)),
            patch.dict(sys.modules, {"torch": torch_mock}),
        ):
            exp632.run_experiment()

        torch_mock.cuda.empty_cache.assert_called()
