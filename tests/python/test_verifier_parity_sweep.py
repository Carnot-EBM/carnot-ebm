"""Tests for VerifierParitySweep — k=16 verifier ensemble parity sweep.

Spec: REQ-VERIFY-2152, SCENARIO-VERIFY-2152
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from carnot.pipeline.verifier_parity_sweep import (
    VerifierParitySweep,
    VerifierParitySweepConfig,
    VerifierSweepResult,
    check_preconditions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

QWEN_SPEC = {"name": "Qwen3.6-35B-A3B", "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}
GEMMA_SPEC = {"name": "Gemma4-31B", "hf_id": "unsloth/gemma-4-31B-it-GGUF"}
FAKE_SPEC = {"name": "FakeModel", "hf_id": "fake-org/fake-model-GGUF"}


# ---------------------------------------------------------------------------
# REQ-VERIFY-2152-1: VerifierParitySweep construction and config validation
# ---------------------------------------------------------------------------


def test_verifier_parity_sweep_requires_positive_k():
    """REQ-VERIFY-2152-1: k_verifiers must be >= 1."""
    with pytest.raises(ValueError, match="k_verifiers"):
        VerifierParitySweep(
            VerifierParitySweepConfig(model_specs=[QWEN_SPEC], k_verifiers=0)
        )


def test_verifier_parity_sweep_requires_positive_n():
    """REQ-VERIFY-2152-1: n_test_cases must be >= 1."""
    with pytest.raises(ValueError, match="n_test_cases"):
        VerifierParitySweep(
            VerifierParitySweepConfig(model_specs=[QWEN_SPEC], n_test_cases=0)
        )


def test_verifier_parity_sweep_construction():
    """REQ-VERIFY-2152-1: Successful construction with valid config."""
    cfg = VerifierParitySweepConfig(model_specs=[QWEN_SPEC], n_test_cases=10, k_verifiers=16)
    sweep = VerifierParitySweep(cfg)
    assert sweep.config.k_verifiers == 16
    assert sweep.config.n_test_cases == 10


# ---------------------------------------------------------------------------
# REQ-VERIFY-2152-2: check_preconditions returns availability per model
# ---------------------------------------------------------------------------


def test_check_preconditions_missing_model(tmp_path):
    """REQ-VERIFY-2152-2: Model with no GGUF cache returns available=False."""
    with patch(
        "carnot.pipeline.verifier_parity_sweep._gguf_files_present",
        return_value=False,
    ):
        cfg = VerifierParitySweepConfig(model_specs=[QWEN_SPEC, GEMMA_SPEC])
        sweep = VerifierParitySweep(cfg)
        results = sweep.check_preconditions()

    assert len(results) == 2
    for r in results:
        assert "resource" in r
        assert "available" in r
        assert r["available"] is False


def test_check_preconditions_present_model():
    """REQ-VERIFY-2152-2: Model with cached GGUF returns available=True."""
    with patch(
        "carnot.pipeline.verifier_parity_sweep._gguf_files_present",
        return_value=True,
    ):
        cfg = VerifierParitySweepConfig(model_specs=[QWEN_SPEC])
        sweep = VerifierParitySweep(cfg)
        results = sweep.check_preconditions()

    assert len(results) == 1
    assert results[0]["available"] is True
    assert results[0]["resource"] == QWEN_SPEC["hf_id"]


def test_check_preconditions_convenience_wrapper():
    """REQ-VERIFY-2152-2: Module-level check_preconditions() is a thin wrapper."""
    with patch(
        "carnot.pipeline.verifier_parity_sweep._gguf_files_present",
        return_value=False,
    ):
        results = check_preconditions([FAKE_SPEC])

    assert len(results) == 1
    assert results[0]["available"] is False


# ---------------------------------------------------------------------------
# Internal helpers: _gguf_cache_dir and _gguf_files_present
# ---------------------------------------------------------------------------


def test_gguf_cache_dir_format():
    """_gguf_cache_dir converts hf_id separators correctly."""
    from carnot.pipeline.verifier_parity_sweep import _gguf_cache_dir

    result = _gguf_cache_dir("unsloth/Qwen3.6-35B-A3B-GGUF")
    assert "models--unsloth--Qwen3.6-35B-A3B-GGUF" in result


def test_gguf_files_present_no_dir():
    """_gguf_files_present returns False when cache dir does not exist."""
    from carnot.pipeline.verifier_parity_sweep import _gguf_files_present

    assert _gguf_files_present("definitely-nonexistent-org/nonexistent-model-GGUF") is False


def test_gguf_files_present_with_gguf_file(tmp_path):
    """_gguf_files_present returns True when a .gguf file is present."""
    from carnot.pipeline.verifier_parity_sweep import _gguf_files_present

    # Create a fake cache structure
    model_dir = tmp_path / "models--fake-org--fake-model-GGUF"
    model_dir.mkdir()
    gguf_file = model_dir / "model.gguf"
    gguf_file.write_bytes(b"fake")

    with patch(
        "carnot.pipeline.verifier_parity_sweep._gguf_cache_dir",
        return_value=str(model_dir),
    ):
        assert _gguf_files_present("fake-org/fake-model-GGUF") is True


def test_gguf_files_present_dir_exists_no_gguf(tmp_path):
    """_gguf_files_present returns False when dir exists but no .gguf files."""
    from carnot.pipeline.verifier_parity_sweep import _gguf_files_present

    model_dir = tmp_path / "models--fake-org--no-gguf-GGUF"
    model_dir.mkdir()
    (model_dir / "some_other_file.json").write_text("{}")

    with patch(
        "carnot.pipeline.verifier_parity_sweep._gguf_cache_dir",
        return_value=str(model_dir),
    ):
        assert _gguf_files_present("fake-org/no-gguf-GGUF") is False


# ---------------------------------------------------------------------------
# REQ-VERIFY-2152-3: run_sweep_for_model produces VerifierSweepResult
# ---------------------------------------------------------------------------


def test_run_sweep_for_model_basic():
    """REQ-VERIFY-2152-3: Synthetic sweep returns well-formed VerifierSweepResult."""
    cfg = VerifierParitySweepConfig(
        model_specs=[QWEN_SPEC], n_test_cases=20, k_verifiers=16
    )
    sweep = VerifierParitySweep(cfg)
    result = sweep.run_sweep_for_model(QWEN_SPEC)

    assert isinstance(result, VerifierSweepResult)
    assert result.model_name == QWEN_SPEC["name"]
    assert result.model_hf_id == QWEN_SPEC["hf_id"]
    assert result.n_test_cases == 20
    assert 0.0 <= result.acceptance_rate <= 1.0
    assert 0.0 <= result.false_accept_rate <= 1.0
    assert result.projection_tax_ms >= 0.0


def test_run_sweep_for_model_per_verifier_rates_k16():
    """REQ-VERIFY-2152-3: per_verifier_pass_rates contains exactly k entries."""
    cfg = VerifierParitySweepConfig(
        model_specs=[GEMMA_SPEC], n_test_cases=30, k_verifiers=16
    )
    sweep = VerifierParitySweep(cfg)
    result = sweep.run_sweep_for_model(GEMMA_SPEC)

    assert len(result.per_verifier_pass_rates) == 16
    for rate in result.per_verifier_pass_rates.values():
        assert 0.0 <= rate <= 1.0


def test_run_sweep_for_model_k8():
    """REQ-VERIFY-2152-3: Works with k=8 (non-default ensemble size)."""
    cfg = VerifierParitySweepConfig(
        model_specs=[FAKE_SPEC], n_test_cases=10, k_verifiers=8
    )
    sweep = VerifierParitySweep(cfg)
    result = sweep.run_sweep_for_model(FAKE_SPEC)

    assert len(result.per_verifier_pass_rates) == 8
    assert result.n_test_cases == 10


def test_run_sweep_for_model_with_ground_truth():
    """REQ-VERIFY-2152-3: false_accept_rate is 0 when ground_truth is all True."""
    cfg = VerifierParitySweepConfig(
        model_specs=[FAKE_SPEC], n_test_cases=20, k_verifiers=16
    )
    sweep = VerifierParitySweep(cfg)
    result = sweep.run_sweep_for_model(
        FAKE_SPEC, ground_truth_labels=[True] * 20
    )
    # When all ground truths are True, no accepted case is a false accept.
    assert result.false_accept_rate == 0.0


def test_run_sweep_for_model_with_provided_features():
    """REQ-VERIFY-2152-3: Provided feature_vectors are used for NLA verifier."""
    cfg = VerifierParitySweepConfig(
        model_specs=[FAKE_SPEC], n_test_cases=10, k_verifiers=16
    )
    sweep = VerifierParitySweep(cfg)
    fake_features = np.zeros((10, 256), dtype=np.float32)
    result = sweep.run_sweep_for_model(FAKE_SPEC, feature_vectors=fake_features)

    assert isinstance(result, VerifierSweepResult)
    assert result.n_test_cases == 10


def test_run_sweep_for_model_n_accepted_consistent():
    """REQ-VERIFY-2152-3: n_accepted equals acceptance_rate * n_test_cases."""
    cfg = VerifierParitySweepConfig(
        model_specs=[FAKE_SPEC], n_test_cases=50, k_verifiers=16
    )
    sweep = VerifierParitySweep(cfg)
    result = sweep.run_sweep_for_model(FAKE_SPEC)

    expected_rate = result.n_accepted / result.n_test_cases
    assert abs(result.acceptance_rate - expected_rate) < 1e-9


# ---------------------------------------------------------------------------
# VerifierParitySweep.run() orchestration
# ---------------------------------------------------------------------------


def test_run_returns_result_per_model():
    """run() returns one VerifierSweepResult per configured model spec."""
    cfg = VerifierParitySweepConfig(
        model_specs=[QWEN_SPEC, GEMMA_SPEC], n_test_cases=10, k_verifiers=16
    )
    sweep = VerifierParitySweep(cfg)
    results = sweep.run()

    assert len(results) == 2
    names = {r.model_name for r in results}
    assert names == {QWEN_SPEC["name"], GEMMA_SPEC["name"]}


def test_run_sweep_for_model_k_greater_than_15_base_rates():
    """REQ-VERIFY-2152-3: k > 16 pads base_pass_rates with default 0.70 entries (line 196)."""
    cfg = VerifierParitySweepConfig(
        model_specs=[FAKE_SPEC], n_test_cases=5, k_verifiers=20
    )
    sweep = VerifierParitySweep(cfg)
    # k=20 > 16-entry _VERIFIER_NAMES; per_verifier_pass_rates is capped at
    # min(k, len(_VERIFIER_NAMES)) but the padding branch for base_pass_rates
    # must execute without error.
    result = sweep.run_sweep_for_model(FAKE_SPEC)

    assert isinstance(result, VerifierSweepResult)
    assert result.n_test_cases == 5


def test_run_dual_gpu_runner_unused_path():
    """run() accepts an optional dual_gpu_runner and falls back to sequential when given None."""
    cfg = VerifierParitySweepConfig(
        model_specs=[FAKE_SPEC], n_test_cases=5, k_verifiers=8
    )
    sweep = VerifierParitySweep(cfg)
    results = sweep.run(dual_gpu_runner=None)

    assert len(results) == 1
    assert results[0].model_name == FAKE_SPEC["name"]
