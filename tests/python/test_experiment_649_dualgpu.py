"""Tests for Exp 649: DualGPU 13B Proof v2 — Pre-verify HF cache.

100% targeted coverage on functions added in
scripts/experiment_649_dualgpu_13b_v2.py:
  - check_hf_cache()
  - detect_gpus()
  - build_device_map()
  - load_model_split()
  - run_forward_passes()
  - sample_util()
  - run_experiment()

Tests run without GPU hardware by mocking torch, pynvml, and transformers.

Spec: REQ-INFRA-092, SCENARIO-INFRA-099, SCENARIO-INFRA-100
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

os.environ["CARNOT_IS_CI"] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import scripts.experiment_649_dualgpu_13b_v2 as exp649  # noqa: E402


# ---------------------------------------------------------------------------
# check_hf_cache
# ---------------------------------------------------------------------------


class TestCheckHfCache:
    """REQ-INFRA-092-1: HF cache check identifies present vs absent weights."""

    def test_no_dir_returns_empty(self, tmp_path: Path) -> None:
        # Cache dir does not exist at all — no models found.
        with patch.object(exp649, "_hf_home", return_value=str(tmp_path)):
            result = exp649.check_hf_cache(["Qwen/Qwen2.5-7B-Instruct"])
        assert result == []

    def test_dir_exists_no_weights(self, tmp_path: Path) -> None:
        # Cache dir exists but only has config files — not counted as cached.
        cache_dir = tmp_path / "hub" / "models--Qwen--Qwen2.5-7B-Instruct"
        cache_dir.mkdir(parents=True)
        (cache_dir / "config.json").write_text("{}")
        with patch.object(exp649, "_hf_home", return_value=str(tmp_path)):
            result = exp649.check_hf_cache(["Qwen/Qwen2.5-7B-Instruct"])
        assert result == []

    def test_dir_exists_with_safetensors(self, tmp_path: Path) -> None:
        # .safetensors shard present — model counts as cached.
        cache_dir = tmp_path / "hub" / "models--Qwen--Qwen2.5-7B-Instruct"
        cache_dir.mkdir(parents=True)
        (cache_dir / "model.safetensors").write_bytes(b"\x00" * 16)
        with patch.object(exp649, "_hf_home", return_value=str(tmp_path)):
            result = exp649.check_hf_cache(["Qwen/Qwen2.5-7B-Instruct"])
        assert result == ["Qwen/Qwen2.5-7B-Instruct"]

    def test_dir_exists_with_bin(self, tmp_path: Path) -> None:
        # .bin shard present — model counts as cached.
        cache_dir = tmp_path / "hub" / "models--Qwen--Qwen2.5-7B-Instruct"
        cache_dir.mkdir(parents=True)
        (cache_dir / "pytorch_model.bin").write_bytes(b"\x00" * 16)
        with patch.object(exp649, "_hf_home", return_value=str(tmp_path)):
            result = exp649.check_hf_cache(["Qwen/Qwen2.5-7B-Instruct"])
        assert result == ["Qwen/Qwen2.5-7B-Instruct"]

    def test_returns_first_found_model(self, tmp_path: Path) -> None:
        # Both candidates cached — returns both in candidate order.
        for model_id in ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]:
            slug = "models--" + model_id.replace("/", "--")
            cache_dir = tmp_path / "hub" / slug
            cache_dir.mkdir(parents=True)
            (cache_dir / "model.safetensors").write_bytes(b"\x00")
        with patch.object(exp649, "_hf_home", return_value=str(tmp_path)):
            result = exp649.check_hf_cache(["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"])
        assert result[0] == "Qwen/Qwen2.5-7B-Instruct"
        assert len(result) == 2

    def test_hf_home_env_var_respected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # HF_HOME env var overrides ~/.cache/huggingface.
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        cache_dir = tmp_path / "hub" / "models--Qwen--Qwen2.5-7B-Instruct"
        cache_dir.mkdir(parents=True)
        (cache_dir / "model.safetensors").write_bytes(b"\x00")
        result = exp649.check_hf_cache(["Qwen/Qwen2.5-7B-Instruct"])
        assert result == ["Qwen/Qwen2.5-7B-Instruct"]


# ---------------------------------------------------------------------------
# detect_gpus
# ---------------------------------------------------------------------------


class TestDetectGpus:
    """REQ-INFRA-092-2: GPU count and VRAM reported before model loading."""

    def test_torch_not_importable(self) -> None:
        with patch.dict(sys.modules, {"torch": None}):
            n, v0, v1 = exp649.detect_gpus()
        assert n == 0
        assert v0 == 0.0
        assert v1 == 0.0

    def test_cuda_not_available(self) -> None:
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": torch_mock}):
            n, v0, v1 = exp649.detect_gpus()
        assert n == 0

    def test_zero_device_count(self) -> None:
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.device_count.return_value = 0
        with patch.dict(sys.modules, {"torch": torch_mock}):
            n, v0, v1 = exp649.detect_gpus()
        assert n == 0

    def test_two_gpus_24gb_each(self) -> None:
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.device_count.return_value = 2
        props = MagicMock()
        props.total_memory = 24 * (1024 ** 3)
        torch_mock.cuda.get_device_properties.return_value = props
        with patch.dict(sys.modules, {"torch": torch_mock}):
            n, v0, v1 = exp649.detect_gpus()
        assert n == 2
        assert v0 == pytest.approx(24.0, abs=0.1)
        assert v1 == pytest.approx(24.0, abs=0.1)

    def test_one_gpu_returns_zero_for_gpu1(self) -> None:
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.device_count.return_value = 1
        props = MagicMock()
        props.total_memory = 24 * (1024 ** 3)
        torch_mock.cuda.get_device_properties.return_value = props
        with patch.dict(sys.modules, {"torch": torch_mock}):
            n, v0, v1 = exp649.detect_gpus()
        assert n == 1
        assert v0 == pytest.approx(24.0, abs=0.1)
        assert v1 == 0.0


# ---------------------------------------------------------------------------
# build_device_map
# ---------------------------------------------------------------------------


class TestBuildDeviceMap:
    """REQ-INFRA-092-3: Layer split device_map assigns layers to correct GPUs."""

    def test_28_layer_split_14(self) -> None:
        dm = exp649.build_device_map(28, 14)
        # Layers 0-13 on cuda:0, 14-27 on cuda:1.
        for i in range(14):
            assert dm[f"model.layers.{i}"] == "cuda:0"
        for i in range(14, 28):
            assert dm[f"model.layers.{i}"] == "cuda:1"
        assert dm["model.embed_tokens"] == "cuda:0"
        assert dm["model.norm"] == "cuda:1"
        assert dm["lm_head"] == "cuda:1"

    def test_all_layers_assigned(self) -> None:
        n_layers = 28
        dm = exp649.build_device_map(n_layers, 14)
        layer_keys = [f"model.layers.{i}" for i in range(n_layers)]
        for key in layer_keys:
            assert key in dm

    def test_asymmetric_split(self) -> None:
        # Split at 10 (7B variant with different architecture).
        dm = exp649.build_device_map(20, 10)
        assert dm["model.layers.9"] == "cuda:0"
        assert dm["model.layers.10"] == "cuda:1"


# ---------------------------------------------------------------------------
# sample_util
# ---------------------------------------------------------------------------


class TestSampleUtil:
    """REQ-INFRA-092-4: GPU utilization sampled via pynvml with torch fallback."""

    def test_pynvml_success(self) -> None:
        pynvml_mock = MagicMock()
        handle = MagicMock()
        rates = MagicMock()
        rates.gpu = 80
        pynvml_mock.nvmlDeviceGetHandleByIndex.return_value = handle
        pynvml_mock.nvmlDeviceGetUtilizationRates.return_value = rates
        with patch.dict(sys.modules, {"pynvml": pynvml_mock}):
            val = exp649.sample_util(1)
        assert val == 80.0

    def test_pynvml_fails_torch_fallback(self) -> None:
        pynvml_mock = MagicMock()
        pynvml_mock.nvmlDeviceGetHandleByIndex.side_effect = RuntimeError("nvml fail")
        torch_mock = MagicMock()
        torch_mock.cuda.utilization.return_value = 55
        with patch.dict(sys.modules, {"pynvml": pynvml_mock, "torch": torch_mock}):
            val = exp649.sample_util(0)
        assert val == 55.0

    def test_both_fail_returns_sentinel(self) -> None:
        pynvml_mock = MagicMock()
        pynvml_mock.nvmlDeviceGetHandleByIndex.side_effect = Exception("no nvml")
        torch_mock = MagicMock()
        torch_mock.cuda.utilization.side_effect = Exception("no torch")
        with patch.dict(sys.modules, {"pynvml": pynvml_mock, "torch": torch_mock}):
            val = exp649.sample_util(0)
        assert val == -1.0

    def test_pynvml_not_installed(self) -> None:
        torch_mock = MagicMock()
        torch_mock.cuda.utilization.return_value = 30
        with patch.dict(sys.modules, {"pynvml": None, "torch": torch_mock}):
            val = exp649.sample_util(0)
        assert val == 30.0


# ---------------------------------------------------------------------------
# load_model_split
# ---------------------------------------------------------------------------


class TestLoadModelSplit:
    """REQ-INFRA-092-5: Model loaded with explicit split device_map."""

    def test_success_returns_model(self) -> None:
        fake_model = MagicMock()
        transformers_mock = MagicMock()
        transformers_mock.AutoModelForCausalLM.from_pretrained.return_value = fake_model
        torch_mock = MagicMock()
        torch_mock.float16 = "float16"
        with patch.dict(sys.modules, {"transformers": transformers_mock, "torch": torch_mock}):
            model = exp649.load_model_split("Qwen/Qwen2.5-7B-Instruct")
        assert model is fake_model

    def test_from_pretrained_raises_returns_none(self) -> None:
        transformers_mock = MagicMock()
        transformers_mock.AutoModelForCausalLM.from_pretrained.side_effect = OSError("no weights")
        torch_mock = MagicMock()
        torch_mock.float16 = "float16"
        with patch.dict(sys.modules, {"transformers": transformers_mock, "torch": torch_mock}):
            model = exp649.load_model_split("Qwen/Qwen2.5-7B-Instruct")
        assert model is None

    def test_passes_device_map_to_from_pretrained(self) -> None:
        fake_model = MagicMock()
        transformers_mock = MagicMock()
        transformers_mock.AutoModelForCausalLM.from_pretrained.return_value = fake_model
        torch_mock = MagicMock()
        torch_mock.float16 = "float16"
        with patch.dict(sys.modules, {"transformers": transformers_mock, "torch": torch_mock}):
            exp649.load_model_split("Qwen/Qwen2.5-7B-Instruct")
        call_kwargs = transformers_mock.AutoModelForCausalLM.from_pretrained.call_args[1]
        assert "device_map" in call_kwargs
        dm = call_kwargs["device_map"]
        assert "lm_head" in dm
        assert dm["lm_head"] == "cuda:1"


# ---------------------------------------------------------------------------
# run_forward_passes
# ---------------------------------------------------------------------------


class TestRunForwardPasses:
    """REQ-INFRA-092-6: 10 forward passes run, utilization sampled after each."""

    def test_tokenizer_fail_returns_empty(self) -> None:
        model_mock = MagicMock()
        transformers_mock = MagicMock()
        transformers_mock.AutoTokenizer.from_pretrained.side_effect = RuntimeError("no tok")
        with patch.dict(sys.modules, {"transformers": transformers_mock}):
            u0, u1 = exp649.run_forward_passes(model_mock, "fake/model", n_passes=3)
        assert u0 == []
        assert u1 == []

    def test_successful_passes_return_correct_length(self) -> None:
        model_mock = MagicMock()
        model_mock.generate.return_value = MagicMock()
        inputs_mock = MagicMock()
        tok_mock = MagicMock()
        tok_mock.return_value = inputs_mock
        transformers_mock = MagicMock()
        transformers_mock.AutoTokenizer.from_pretrained.return_value = tok_mock
        torch_mock = MagicMock()
        with (
            patch.dict(sys.modules, {"transformers": transformers_mock, "torch": torch_mock}),
            patch.object(exp649, "sample_util", side_effect=[70.0, 80.0] * 10),
        ):
            u0, u1 = exp649.run_forward_passes(model_mock, "fake/model", n_passes=5)
        assert len(u0) == 5
        assert len(u1) == 5

    def test_generate_exception_records_sentinel(self) -> None:
        model_mock = MagicMock()
        model_mock.generate.side_effect = RuntimeError("CUDA OOM")
        tok_mock = MagicMock()
        tensor_mock = MagicMock()
        tensor_mock.to.return_value = tensor_mock
        tok_mock.return_value = {"input_ids": tensor_mock}
        transformers_mock = MagicMock()
        transformers_mock.AutoTokenizer.from_pretrained.return_value = tok_mock
        torch_mock = MagicMock()
        torch_mock.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        torch_mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        with patch.dict(sys.modules, {"transformers": transformers_mock, "torch": torch_mock}):
            u0, u1 = exp649.run_forward_passes(model_mock, "fake/model", n_passes=2)
        assert len(u0) == 2
        assert all(v == -1.0 for v in u0)


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------


class TestRunExperiment:
    """REQ-INFRA-092: run_experiment produces all required schema fields."""

    def test_only_one_gpu_blocked(self) -> None:
        # SCENARIO-INFRA-099: only 1 GPU → blocked immediately.
        with patch.object(exp649, "detect_gpus", return_value=(1, 24.0, 0.0)):
            result = exp649.run_experiment()
        assert result["n_gpus"] == 1
        assert result["model_loaded"] is False
        assert result["blocked_reason"] == "only_one_gpu"
        assert result["dualgpu_proven"] is False

    def test_no_gpus_blocked(self) -> None:
        with patch.object(exp649, "detect_gpus", return_value=(0, 0.0, 0.0)):
            result = exp649.run_experiment()
        assert result["n_gpus"] == 0
        assert result["blocked_reason"] == "only_one_gpu"

    def test_model_not_cached_returns_action_required(self, tmp_path: Path) -> None:
        # SCENARIO-INFRA-099: 2 GPUs but no cached weights → action_required.
        with (
            patch.object(exp649, "detect_gpus", return_value=(2, 24.0, 24.0)),
            patch.object(exp649, "check_hf_cache", return_value=[]),
        ):
            result = exp649.run_experiment()
        assert result["model_loaded"] is False
        assert result["blocked_reason"] == "model_not_cached_HF_weights_required"
        assert "action_required" in result
        assert "huggingface-cli download" in result["action_required"]
        assert result["honest_verdict"] == "model_not_cached"

    def test_model_load_fails_after_cache_found(self) -> None:
        # Cache check passes but transformers load fails.
        with (
            patch.object(exp649, "detect_gpus", return_value=(2, 24.0, 24.0)),
            patch.object(exp649, "check_hf_cache", return_value=["Qwen/Qwen2.5-7B-Instruct"]),
            patch.object(exp649, "load_model_split", return_value=None),
        ):
            result = exp649.run_experiment()
        assert result["model_loaded"] is False
        assert result["blocked_reason"] == "model_load_failed"

    def test_dualgpu_proven_high_util(self) -> None:
        # SCENARIO-INFRA-100: model loaded, GPU-1 peaks > 50% → proven.
        fake_model = MagicMock()
        with (
            patch.object(exp649, "detect_gpus", return_value=(2, 24.0, 24.0)),
            patch.object(exp649, "check_hf_cache", return_value=["Qwen/Qwen2.5-7B-Instruct"]),
            patch.object(exp649, "load_model_split", return_value=fake_model),
            patch.object(exp649, "run_forward_passes", return_value=([80.0] * 10, [75.0] * 10)),
        ):
            result = exp649.run_experiment()
        assert result["model_loaded"] is True
        assert result["peak_gpu1_util"] == 75.0
        assert result["dualgpu_proven"] is True
        assert result["retro_071_resolved"] is True
        assert result["honest_verdict"] == "dualgpu_proven"

    def test_dualgpu_loaded_low_util(self) -> None:
        # Model loaded but GPU-1 stays below threshold.
        fake_model = MagicMock()
        with (
            patch.object(exp649, "detect_gpus", return_value=(2, 24.0, 24.0)),
            patch.object(exp649, "check_hf_cache", return_value=["Qwen/Qwen2.5-7B-Instruct"]),
            patch.object(exp649, "load_model_split", return_value=fake_model),
            patch.object(exp649, "run_forward_passes", return_value=([50.0] * 10, [5.0] * 10)),
        ):
            result = exp649.run_experiment()
        assert result["model_loaded"] is True
        assert result["dualgpu_proven"] is False
        assert result["honest_verdict"] == "dualgpu_loaded_low_util"
        assert result["retro_071_resolved"] is False

    def test_sustained_fraction_proves_dualgpu(self) -> None:
        # Peak stays at 40% but 8/10 passes > 10% → sustained_gpu1_fraction=0.8 → proven.
        fake_model = MagicMock()
        util_1 = [40.0] * 8 + [5.0] * 2
        with (
            patch.object(exp649, "detect_gpus", return_value=(2, 24.0, 24.0)),
            patch.object(exp649, "check_hf_cache", return_value=["Qwen/Qwen2.5-7B-Instruct"]),
            patch.object(exp649, "load_model_split", return_value=fake_model),
            patch.object(exp649, "run_forward_passes", return_value=([50.0] * 10, util_1)),
        ):
            result = exp649.run_experiment()
        assert result["sustained_gpu1_fraction"] == pytest.approx(0.8)
        assert result["dualgpu_proven"] is True

    def test_all_required_fields_present(self) -> None:
        # REQ-INFRA-092: all required fields in artifact.
        with patch.object(exp649, "detect_gpus", return_value=(0, 0.0, 0.0)):
            result = exp649.run_experiment()
        required = [
            "n_gpus", "vram_0_gb", "vram_1_gb",
            "model_loaded", "model_name",
            "peak_gpu1_util", "sustained_gpu1_fraction",
            "dualgpu_proven", "retro_071_resolved", "honest_verdict",
        ]
        for field in required:
            assert field in result, f"Missing required field: {field}"

    def test_gpu_cleanup_called_after_passes(self) -> None:
        # torch.cuda.empty_cache() must be called after forward passes.
        fake_model = MagicMock()
        torch_mock = MagicMock()
        with (
            patch.object(exp649, "detect_gpus", return_value=(2, 24.0, 24.0)),
            patch.object(exp649, "check_hf_cache", return_value=["Qwen/Qwen2.5-7B-Instruct"]),
            patch.object(exp649, "load_model_split", return_value=fake_model),
            patch.object(exp649, "run_forward_passes", return_value=([0.0] * 10, [0.0] * 10)),
            patch.dict(sys.modules, {"torch": torch_mock}),
        ):
            exp649.run_experiment()
        torch_mock.cuda.empty_cache.assert_called()

    def test_sentinel_utils_excluded_from_peak(self) -> None:
        # -1.0 sentinel values (measurement failures) must not inflate peak.
        fake_model = MagicMock()
        util_1 = [-1.0] * 5 + [30.0] * 5
        with (
            patch.object(exp649, "detect_gpus", return_value=(2, 24.0, 24.0)),
            patch.object(exp649, "check_hf_cache", return_value=["Qwen/Qwen2.5-7B-Instruct"]),
            patch.object(exp649, "load_model_split", return_value=fake_model),
            patch.object(exp649, "run_forward_passes", return_value=([0.0] * 10, util_1)),
        ):
            result = exp649.run_experiment()
        # Peak should be 30.0, not -1.0.
        assert result["peak_gpu1_util"] == pytest.approx(30.0)
