"""Tests for Exp 3726 tiny EBT GSM8K CUDA train-step smoke.

Spec: REQ-EBT-3726, SCENARIO-EBT-3726.
"""

from __future__ import annotations

import json
import sys
import builtins
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carnot.phase3 import tiny_ebt_train_smoke as exp3726


class _FakeCudaAvailable:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 0
        return "NVIDIA GeForce RTX 3090"


def test_byte_tokenizer_builds_reasoning_blocks_without_p01_corpus() -> None:
    """REQ-EBT-3726: GSM8K train rows are tokenized into training blocks."""
    texts = [
        "Question: Ada has 3 marbles. She buys 4.\nAnswer: 3 + 4 = 7. #### 7",
        "Question: Ben splits 12 cards across 3 piles.\nAnswer: 12 / 3 = 4. #### 4",
    ]

    blocks = exp3726.tokenize_texts_to_blocks(texts, block_size=16)

    assert blocks.shape[1] == 17
    assert blocks.dtype.name == "int64"
    assert int(blocks.min()) >= 1
    assert int(blocks.max()) < exp3726.BYTE_TOKENIZER_VOCAB_SIZE


def test_format_and_tokenizer_reject_invalid_inputs() -> None:
    """REQ-EBT-3726: corpus text must be structured and large enough."""
    assert exp3726.format_gsm8k_row({"question": "What is 2+2?", "answer": "#### 4"}) == (
        "Question: What is 2+2?\nAnswer: #### 4"
    )

    with pytest.raises(ValueError, match="block_size"):
        exp3726.tokenize_texts_to_blocks(["abc"], block_size=1)
    with pytest.raises(ValueError, match="not enough"):
        exp3726.tokenize_texts_to_blocks(["a"], block_size=128)


def test_load_gsm8k_train_texts_uses_huggingface_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-EBT-3726: corpus acquisition uses GSM8K train rows."""
    import datasets

    def _fake_load_dataset(name: str, subset: str, split: str) -> list[dict[str, str]]:
        assert (name, subset, split) == ("gsm8k", "main", "train[:2]")
        return [
            {"question": "A?", "answer": "#### 1"},
            {"question": "B?", "answer": "#### 2"},
        ]

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)

    assert exp3726.load_gsm8k_train_texts(2) == [
        "Question: A?\nAnswer: #### 1",
        "Question: B?\nAnswer: #### 2",
    ]
    with pytest.raises(ValueError, match="n_train"):
        exp3726.load_gsm8k_train_texts(0)


def test_reproducibility_checksum_changes_with_corpus_and_config() -> None:
    """SCENARIO-EBT-3726: corpus/config drift changes the checksum."""
    config = exp3726.TinyEBTSmokeConfig(n_train=2, train_steps=10)
    checksum_a = exp3726.reproducibility_checksum(["gsm8k a", "gsm8k b"], config)
    checksum_b = exp3726.reproducibility_checksum(["gsm8k a", "gsm8k c"], config)
    checksum_c = exp3726.reproducibility_checksum(
        ["gsm8k a", "gsm8k b"],
        exp3726.TinyEBTSmokeConfig(n_train=3, train_steps=10),
    )

    assert len(checksum_a) == 64
    assert checksum_a != checksum_b
    assert checksum_a != checksum_c


@pytest.mark.parametrize(
    ("losses", "expected"),
    [
        ([2.0, 1.8, 1.5], True),
        ([2.0, float("nan"), 1.5], False),
        ([2.0, 2.1, 2.05], False),
        ([2.0], False),
    ],
)
def test_loss_smoke_requires_finite_and_decreasing(losses: list[float], expected: bool) -> None:
    """REQ-EBT-3726: finite decreasing losses are required for success."""
    assert exp3726.loss_smoke_passed(losses) is expected


def test_success_artifact_has_required_terminal_schema() -> None:
    """SCENARIO-EBT-3726: successful train smoke writes required fields."""
    config = exp3726.TinyEBTSmokeConfig(
        n_train=2048,
        dim=512,
        n_layers=4,
        n_heads=8,
        ffn_dim_multiplier=4.0,
        block_size=64,
        batch_size=4,
        train_steps=10,
    )
    preconditions = {
        "cuda": {"available": True, "device_count": 1, "device_name": "NVIDIA GeForce RTX 3090"},
        "ebt_vendored_importable": True,
        "dataset": {"available": True, "source": "huggingface_api"},
    }
    artifact = exp3726.build_success_artifact(
        n_train=2048,
        param_count=16_910_337,
        peak_vram_mb=900,
        losses=[2.0, 1.7, 1.1],
        preconditions=preconditions,
        model_specs=exp3726.model_specs(config, param_count=16_910_337),
        random_seed=3726,
        checksum="a" * 64,
        duration_s=61.0,
    )

    errors = exp3726.validate_success_artifact(artifact)

    assert errors == []
    assert artifact["honest_verdict"] == (
        "complete: tiny_ebt_17M_fits_3090_900mb_single_train_step_"
        "loss_finite_and_decreasing_corpus_gsm8k_n2048"
    )
    assert artifact["inference_substrate"].startswith("live_llm_inference")
    assert artifact["model_specs"]["from_scratch_not_pretrained_llm"] is True
    assert json.loads(json.dumps(artifact)) == artifact


def test_success_artifact_validation_rejects_out_of_band_model_and_losses() -> None:
    """REQ-EBT-3726: tiny-EBT band and decreasing-loss evidence are enforced."""
    artifact = exp3726.build_success_artifact(
        n_train=2048,
        param_count=7_000_000,
        peak_vram_mb=900,
        losses=[1.0, 1.1],
        preconditions={"cuda": {"available": True}},
        model_specs={"from_scratch_not_pretrained_llm": False},
        random_seed=3726,
        checksum="b" * 64,
        duration_s=1.0,
    )

    errors = exp3726.validate_success_artifact(artifact)

    assert "ebt_param_count must be within 10M-50M" in errors
    assert "first_step_losses must be finite and decreasing" in errors
    assert "model_specs must identify a from-scratch non-pretrained EBT" in errors


def test_success_artifact_validation_reports_all_required_field_failures() -> None:
    """REQ-EBT-3726: invalid success artifacts are not accepted silently."""
    artifact = {
        "honest_verdict": "blocked_cuda",
        "inference_substrate": "cpu_fixture",
        "n_train": 0,
        "ebt_param_count": 60_000_000,
        "peak_vram_mb": 0,
        "first_step_losses": [1.0, 0.8],
        "loss_finite": False,
        "preconditions_checked": {},
        "model_specs": {"from_scratch_not_pretrained_llm": True},
        "random_seed": 3726,
        "reproducibility_checksum": "",
        "duration_s": 0.0,
    }

    errors = exp3726.validate_success_artifact(artifact)

    assert "honest_verdict must start with complete:" in errors
    assert "inference_substrate must identify live_llm_inference" in errors
    assert "ebt_param_count must be within 10M-50M" in errors
    assert "n_train must be positive" in errors
    assert "peak_vram_mb must be positive" in errors
    assert "loss_finite must be true" in errors
    assert "reproducibility_checksum must be present" in errors
    assert "duration_s must be positive" in errors


def test_success_artifact_validation_reports_missing_fields() -> None:
    """SCENARIO-EBT-3726: required artifact fields are explicit."""
    errors = exp3726.validate_success_artifact({})

    assert any(error.startswith("missing required fields:") for error in errors)


def test_blocked_artifact_uses_terminal_blocked_prefix() -> None:
    """SCENARIO-EBT-3726: missing preconditions produce blocked verdicts."""
    artifact = exp3726.build_blocked_artifact(
        "blocked_cuda",
        preconditions={"cuda": {"available": False}},
        random_seed=3726,
        duration_s=0.2,
    )

    assert artifact["honest_verdict"] == "blocked_cuda"
    assert artifact["preconditions_checked"]["cuda"]["available"] is False


def test_write_artifact_serializes_numpy_scalars(tmp_path: Path) -> None:
    """REQ-EBT-3726: artifact JSON stores bare values, not numpy wrappers."""
    path = tmp_path / "artifact.json"

    exp3726.write_artifact(path, {"value": np.int64(7)})

    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 7}


def test_json_default_rejects_unknown_objects() -> None:
    """REQ-EBT-3726: JSON fallback only unwraps known scalar values."""
    with pytest.raises(TypeError, match="not JSON serializable"):
        exp3726._json_default(object())


def test_dataset_api_and_cache_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-EBT-3726: dataset precondition accepts network or local cache."""
    monkeypatch.setattr(
        exp3726.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    assert exp3726._dataset_api_available() is True

    def _raise_timeout(*args: object, **kwargs: object) -> None:
        raise exp3726.subprocess.TimeoutExpired("curl", timeout=10)

    monkeypatch.setattr(exp3726.subprocess, "run", _raise_timeout)
    assert exp3726._dataset_api_available() is False

    monkeypatch.setattr(exp3726.Path, "home", lambda: tmp_path)
    assert exp3726._local_gsm8k_cache_exists() is False
    (tmp_path / ".cache" / "huggingface" / "datasets" / "gsm8k").mkdir(parents=True)
    assert exp3726._local_gsm8k_cache_exists() is True


def test_check_preconditions_accepts_local_cache_when_network_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-EBT-3726: local GSM8K cache satisfies the dataset precondition."""
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_FakeCudaAvailable))
    monkeypatch.setitem(sys.modules, "carnot.phase3.ebt_upstream", SimpleNamespace())
    monkeypatch.setattr(exp3726, "_dataset_api_available", lambda: False)
    monkeypatch.setattr(exp3726, "_local_gsm8k_cache_exists", lambda: True)

    preconditions = exp3726.check_preconditions()

    assert preconditions["dataset"]["source"] == "local_gsm8k_cache"


def test_check_preconditions_accepts_dataset_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-EBT-3726: HuggingFace dataset API satisfies the corpus precondition."""
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_FakeCudaAvailable))
    monkeypatch.setitem(sys.modules, "carnot.phase3.ebt_upstream", SimpleNamespace())
    monkeypatch.setattr(exp3726, "_dataset_api_available", lambda: True)

    preconditions = exp3726.check_preconditions()

    assert preconditions["dataset"]["source"] == "huggingface_api"


def test_check_preconditions_blocks_missing_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-EBT-3726: CUDA absence produces a terminal blocked resource."""

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda))

    with pytest.raises(exp3726.PreconditionError) as exc:
        exp3726.check_preconditions()

    assert exc.value.verdict == "blocked_cuda"
    assert exc.value.preconditions["cuda"]["available"] is False


def test_check_preconditions_records_torch_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-EBT-3726: broken torch imports block before training."""
    original_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise RuntimeError("fixture torch missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(exp3726.PreconditionError) as exc:
        exp3726.check_preconditions()

    assert exc.value.verdict == "blocked_cuda"
    assert "fixture torch missing" in exc.value.preconditions["cuda"]["error"]


def test_check_preconditions_blocks_missing_ebt(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-EBT-3726: missing vendored EBT reports the required verdict."""
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_FakeCudaAvailable))
    original_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "carnot.phase3.ebt_upstream":
            raise ImportError("fixture missing EBT")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(exp3726.PreconditionError) as exc:
        exp3726.check_preconditions()

    assert exc.value.verdict == "blocked_ebt_not_vendored"
    assert "fixture missing EBT" in exc.value.preconditions["ebt_vendored_error"]


def test_check_preconditions_blocks_missing_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-EBT-3726: no network/cache blocks GSM8K corpus acquisition."""
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_FakeCudaAvailable))
    monkeypatch.setitem(sys.modules, "carnot.phase3.ebt_upstream", SimpleNamespace())
    monkeypatch.setattr(exp3726, "_dataset_api_available", lambda: False)
    monkeypatch.setattr(exp3726, "_local_gsm8k_cache_exists", lambda: False)

    with pytest.raises(exp3726.PreconditionError) as exc:
        exp3726.check_preconditions()

    assert exc.value.verdict == "blocked_gsm8k_dataset"


def test_run_cuda_training_smoke_rejects_empty_batches() -> None:
    """REQ-EBT-3726: live train smoke needs at least one full batch."""
    config = exp3726.TinyEBTSmokeConfig(batch_size=2)

    with pytest.raises(ValueError, match="not enough token blocks"):
        exp3726.run_cuda_training_smoke(np.zeros((1, 4), dtype=np.int64), config)


def test_run_experiment_writes_blocked_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-EBT-3726: failed preconditions write a blocked artifact."""

    def _blocked() -> dict[str, object]:
        raise exp3726.PreconditionError("blocked_cuda", {"cuda": {"available": False}})

    monkeypatch.setattr(exp3726, "check_preconditions", _blocked)
    artifact = exp3726.run_experiment(result_path=tmp_path / "blocked.json")

    assert artifact["honest_verdict"] == "blocked_cuda"
    assert json.loads((tmp_path / "blocked.json").read_text())["honest_verdict"] == "blocked_cuda"


def test_run_experiment_success_with_injected_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-EBT-3726: orchestration writes complete artifact from train evidence."""
    config = exp3726.TinyEBTSmokeConfig(n_train=2, block_size=8, batch_size=1, train_steps=2)
    ticks = iter([10.0, 11.0])
    monkeypatch.setattr(exp3726.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        exp3726,
        "check_preconditions",
        lambda: {
            "cuda": {"available": True, "device_count": 1, "device_name": "NVIDIA GeForce RTX 3090"},
            "ebt_vendored_importable": True,
            "dataset": {"available": True, "source": "fixture"},
        },
    )
    monkeypatch.setattr(exp3726, "set_reproducibility", lambda seed: None)
    monkeypatch.setattr(
        exp3726,
        "load_gsm8k_train_texts",
        lambda n_train: [
            "Question: fixture A\nAnswer: #### 1",
            "Question: fixture B\nAnswer: #### 2",
        ],
    )
    monkeypatch.setattr(
        exp3726,
        "run_cuda_training_smoke",
        lambda token_blocks, cfg: {
            "losses": [2.0, 1.0],
            "param_count": 16_000_000,
            "peak_vram_mb": 512,
        },
    )

    artifact = exp3726.run_experiment(config=config, result_path=tmp_path / "success.json")

    assert artifact["honest_verdict"].startswith("complete: tiny_ebt_16M")
    assert artifact["n_train"] == 2


def test_run_experiment_marks_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-EBT-3726: orchestration does not report success on invalid evidence."""
    config = exp3726.TinyEBTSmokeConfig(n_train=2, block_size=8, batch_size=1, train_steps=2)
    ticks = iter([20.0, 21.0])
    monkeypatch.setattr(exp3726.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        exp3726,
        "check_preconditions",
        lambda: {
            "cuda": {"available": True, "device_count": 1, "device_name": "NVIDIA GeForce RTX 3090"},
            "ebt_vendored_importable": True,
            "dataset": {"available": True, "source": "fixture"},
        },
    )
    monkeypatch.setattr(exp3726, "set_reproducibility", lambda seed: None)
    monkeypatch.setattr(
        exp3726,
        "load_gsm8k_train_texts",
        lambda n_train: [
            "Question: fixture A\nAnswer: #### 1",
            "Question: fixture B\nAnswer: #### 2",
        ],
    )
    monkeypatch.setattr(
        exp3726,
        "run_cuda_training_smoke",
        lambda token_blocks, cfg: {
            "losses": [1.0, 1.2],
            "param_count": 16_000_000,
            "peak_vram_mb": 512,
        },
    )

    artifact = exp3726.run_experiment(config=config, result_path=tmp_path / "invalid.json")

    assert artifact["honest_verdict"] == "blocked_training_smoke_validation"
    assert "validation_errors" in artifact


def test_main_status_reflects_terminal_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-EBT-3726: CLI exits nonzero for blocked artifacts."""
    monkeypatch.setattr(exp3726, "run_experiment", lambda: {"honest_verdict": "complete: ok"})
    assert exp3726.main([]) == 0

    monkeypatch.setattr(exp3726, "run_experiment", lambda: {"honest_verdict": "blocked_cuda"})
    assert exp3726.main([]) == 1
