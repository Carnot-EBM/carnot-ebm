"""Tests for carnot.pipeline.dualgpu_retrain.

Spec: REQ-INFRA-091, SCENARIO-INFRA-097, SCENARIO-INFRA-098
"""

from __future__ import annotations

import pytest

from carnot.pipeline.dualgpu_retrain import DualGPURetrain, DualGPURetrainConfig


# ---------------------------------------------------------------------------
# DualGPURetrainConfig
# ---------------------------------------------------------------------------


def test_config_stores_devices() -> None:
    """Config dataclass stores eorm_device and jepa_device verbatim."""
    cfg = DualGPURetrainConfig(eorm_device="cuda:0", jepa_device="cuda:1")
    assert cfg.eorm_device == "cuda:0"
    assert cfg.jepa_device == "cuda:1"


def test_config_cpu_fallback() -> None:
    """Config accepts cpu strings when GPUs are unavailable (SCENARIO-INFRA-098)."""
    cfg = DualGPURetrainConfig(eorm_device="cpu", jepa_device="cpu")
    assert cfg.eorm_device == "cpu"
    assert cfg.jepa_device == "cpu"


# ---------------------------------------------------------------------------
# DualGPURetrain.run_parallel
# ---------------------------------------------------------------------------


def test_run_parallel_returns_both_results() -> None:
    """run_parallel returns dict with both 'eorm' and 'jepa' keys (SCENARIO-INFRA-097)."""
    cfg = DualGPURetrainConfig(eorm_device="cpu", jepa_device="cpu")
    retrain = DualGPURetrain(cfg)
    result = retrain.run_parallel(lambda: "eorm_done", lambda: "jepa_done")
    assert result["eorm"] == "eorm_done"
    assert result["jepa"] == "jepa_done"


def test_run_parallel_correct_values() -> None:
    """run_parallel maps each callable's return value to its respective key."""
    cfg = DualGPURetrainConfig(eorm_device="cpu", jepa_device="cpu")
    retrain = DualGPURetrain(cfg)
    result = retrain.run_parallel(lambda: 42, lambda: 99)
    assert result["eorm"] == 42
    assert result["jepa"] == 99


def test_run_parallel_returns_only_two_keys() -> None:
    """run_parallel result dict has exactly the 'eorm' and 'jepa' keys."""
    cfg = DualGPURetrainConfig(eorm_device="cpu", jepa_device="cpu")
    retrain = DualGPURetrain(cfg)
    result = retrain.run_parallel(lambda: None, lambda: None)
    assert set(result.keys()) == {"eorm", "jepa"}


def test_run_parallel_propagates_exception() -> None:
    """An exception raised by either callable propagates out of run_parallel."""
    cfg = DualGPURetrainConfig(eorm_device="cpu", jepa_device="cpu")
    retrain = DualGPURetrain(cfg)

    def bad_eorm() -> None:
        raise ValueError("eorm exploded")

    with pytest.raises(ValueError, match="eorm exploded"):
        retrain.run_parallel(bad_eorm, lambda: "ok")


def test_run_parallel_jepa_exception_propagates() -> None:
    """An exception from the jepa callable also propagates correctly."""
    cfg = DualGPURetrainConfig(eorm_device="cpu", jepa_device="cpu")
    retrain = DualGPURetrain(cfg)

    def bad_jepa() -> None:
        raise RuntimeError("jepa exploded")

    with pytest.raises(RuntimeError, match="jepa exploded"):
        retrain.run_parallel(lambda: "ok", bad_jepa)


def test_config_stored_on_instance() -> None:
    """DualGPURetrain.config is the config object passed at construction."""
    cfg = DualGPURetrainConfig(eorm_device="cuda:0", jepa_device="cuda:1")
    retrain = DualGPURetrain(cfg)
    assert retrain.config is cfg
