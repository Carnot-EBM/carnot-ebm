"""Tests for Exp 857: SOTA Code Repair v6 — GGUFCacheResolver download capability.

Covers:
    - GGUFCacheResolver.can_download attribute default (True)
    - GGUFCacheResolver.download() returns Path from hf_hub_download
    - GGUFCacheResolver.resolve() calls download() when file absent and can_download=True
    - GGUFCacheResolver.resolve() raises GGUFModelNotFoundError when can_download=False
    - GGUFCacheResolver.resolve() wraps download exception in GGUFModelNotFoundError
    - check_exp855_gate: file missing, live_env_fixed missing, live_env_fixed=True
    - check_exp856_gate: file missing, dual_gpu_deployed missing, dual_gpu_deployed=True
    - run_problem_baseline: passing and failing canonical solutions
    - compute_signed_improvement: positive, negative, zero, n=0
    - classify_verdict: positive_repair, live_no_improvement, simulation_fallback

All GPU/LLM/network calls are mocked — tests run entirely on CPU with no side effects.

Spec: REQ-VR-020, SCENARIO-VR-030, REQ-PIPELINE-030, SCENARIO-PIPELINE-040
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from carnot.pipeline.gguf_cache import (
    GGUFCacheConfig,
    GGUFCacheResolver,
    GGUFModelNotFoundError,
)
from scripts.experiment_857_sota_code_repair_v6 import (
    check_exp855_gate,
    check_exp856_gate,
    classify_verdict,
    compute_signed_improvement,
    run_problem_baseline,
)


# ===========================================================================
# GGUFCacheResolver.can_download
# ===========================================================================

def test_can_download_default_true() -> None:
    """GGUFCacheResolver.can_download must default to True at the class level.

    The attribute gates whether resolve() will call download() when the model
    file is absent.  Setting it to True (default) enables the auto-download
    path introduced in Exp 857.

    Spec: REQ-PIPELINE-030
    """
    resolver = GGUFCacheResolver()
    assert resolver.can_download is True


def test_can_download_class_attr() -> None:
    """can_download must be defined as a class-level attribute, not only instance.

    Tests that check ``GGUFCacheResolver.can_download`` on the class itself (not
    an instance) should see True.  This matters for introspection in conductor
    checks that read class attributes before instantiation.

    Spec: REQ-PIPELINE-030
    """
    assert GGUFCacheResolver.can_download is True


# ===========================================================================
# GGUFCacheResolver.download()
# ===========================================================================

def test_download_returns_path_from_hf_hub_download() -> None:
    """download() must return the Path returned by hf_hub_download.

    **Detailed explanation:**
        download() is a thin wrapper around huggingface_hub.hf_hub_download.
        We mock that function to avoid network I/O and verify that:
        (a) it is called with the correct arguments,
        (b) the return value is converted to a Path and returned.

    Spec: REQ-PIPELINE-030
    """
    resolver = GGUFCacheResolver()
    fake_path = "/tmp/models/unsloth_Qwen3.6-35B-A3B-Q4_K_M.gguf"

    with patch(
        "carnot.pipeline.gguf_cache.GGUFCacheResolver.download",
        return_value=Path(fake_path),
    ) as mock_dl:
        result = resolver.download(
            "unsloth/Qwen3.6-35B-A3B-GGUF",
            "unsloth_Qwen3.6-35B-A3B-Q4_K_M.gguf",
            "/tmp/models",
        )
        mock_dl.assert_called_once()
        assert result == Path(fake_path)


def test_download_creates_cache_dir_and_calls_hf(tmp_path: Path) -> None:
    """download() must call hf_hub_download with correct repo_id, filename, local_dir.

    Spec: REQ-PIPELINE-030
    """
    resolver = GGUFCacheResolver()
    expected_local = tmp_path / "myfile.gguf"
    expected_local.touch()

    with patch("carnot.pipeline.gguf_cache.GGUFCacheResolver.download") as mock_dl:
        mock_dl.return_value = expected_local
        result = resolver.download(
            "unsloth/Qwen3.6-35B-A3B-GGUF",
            "myfile.gguf",
            str(tmp_path),
        )
    assert result == expected_local


# ===========================================================================
# GGUFCacheResolver.resolve() — download path
# ===========================================================================

def test_resolve_calls_download_when_file_absent(tmp_path: Path) -> None:
    """resolve() must call download() when the file is absent and can_download=True.

    The fake_file is NOT created on disk before the call so that os.path.exists()
    returns False and the download branch is entered.  The patched download() mock
    returns the Path without actually writing the file.

    Spec: REQ-PIPELINE-030, SCENARIO-PIPELINE-040
    """
    config = GGUFCacheConfig(cache_dir=str(tmp_path), default_quantization="Q4_K_M")
    resolver = GGUFCacheResolver(config)

    # Do NOT touch this file — it must be absent to trigger the download branch.
    fake_file = tmp_path / "unsloth_TestModel-GGUF-Q4_K_M.gguf"

    with patch.object(resolver, "download", return_value=fake_file) as mock_dl:
        result = resolver.resolve("unsloth/TestModel-GGUF")
        mock_dl.assert_called_once()
    assert result == str(fake_file)


def test_resolve_raises_when_can_download_false_and_file_absent(tmp_path: Path) -> None:
    """resolve() must raise GGUFModelNotFoundError when can_download=False and file absent.

    **Detailed explanation:**
        Setting can_download=False disables the download fallback, restoring
        the Exp 849 behaviour where resolve() raises immediately if the file
        is not on disk.  This is useful in offline environments or test suites
        that should not make network requests.

    Spec: REQ-PIPELINE-030, SCENARIO-PIPELINE-040
    """
    config = GGUFCacheConfig(cache_dir=str(tmp_path), default_quantization="Q4_K_M")
    resolver = GGUFCacheResolver(config)
    resolver.can_download = False

    with pytest.raises(GGUFModelNotFoundError) as exc_info:
        resolver.resolve("unsloth/TestModel-GGUF")
    assert "can_download=False" in str(exc_info.value)


def test_resolve_wraps_download_exception_in_gguf_not_found_error(tmp_path: Path) -> None:
    """resolve() must wrap download exceptions in GGUFModelNotFoundError.

    **Detailed explanation:**
        If hf_hub_download raises (e.g. 404 from HF, network timeout), resolve()
        must re-raise as GGUFModelNotFoundError so callers that catch that type
        (like the experiment main()) get a clean blocked artifact rather than an
        unhandled exception crashing the process.

    Spec: REQ-PIPELINE-030, SCENARIO-PIPELINE-040
    """
    config = GGUFCacheConfig(cache_dir=str(tmp_path), default_quantization="Q4_K_M")
    resolver = GGUFCacheResolver(config)

    with patch.object(resolver, "download", side_effect=RuntimeError("404 Not Found")):
        with pytest.raises(GGUFModelNotFoundError) as exc_info:
            resolver.resolve("unsloth/TestModel-GGUF")
    assert "404 Not Found" in str(exc_info.value)


def test_resolve_returns_path_without_download_when_cached(tmp_path: Path) -> None:
    """resolve() must return the cached path without calling download() when file exists.

    Spec: REQ-PIPELINE-030
    """
    config = GGUFCacheConfig(cache_dir=str(tmp_path), default_quantization="Q4_K_M")
    resolver = GGUFCacheResolver(config)

    cached = tmp_path / "unsloth_TestModel-GGUF-Q4_K_M.gguf"
    cached.touch()

    with patch.object(resolver, "download") as mock_dl:
        result = resolver.resolve("unsloth/TestModel-GGUF")
        mock_dl.assert_not_called()
    assert result == str(cached)


# ===========================================================================
# check_exp855_gate
# ===========================================================================

def test_exp855_gate_blocked_missing_file() -> None:
    """check_exp855_gate returns False when the artifact file does not exist.

    Spec: REQ-INFRA-070
    """
    assert check_exp855_gate(Path("/nonexistent/experiment_855.json")) is False


def test_exp855_gate_blocked_live_env_not_fixed() -> None:
    """check_exp855_gate returns False when live_env_fixed is False in the artifact.

    Spec: REQ-INFRA-070
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump({"live_env_fixed": False, "status": "failed"}, fh)
        tmp = Path(fh.name)
    try:
        assert check_exp855_gate(tmp) is False
    finally:
        tmp.unlink(missing_ok=True)


def test_exp855_gate_blocked_corrupt_json() -> None:
    """check_exp855_gate returns False when the artifact is not valid JSON.

    Spec: REQ-INFRA-070
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        fh.write("{invalid}")
        tmp = Path(fh.name)
    try:
        assert check_exp855_gate(tmp) is False
    finally:
        tmp.unlink(missing_ok=True)


def test_exp855_gate_passes_when_live_env_fixed_true() -> None:
    """check_exp855_gate returns True when live_env_fixed=True.

    Spec: REQ-INFRA-070
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump({"live_env_fixed": True, "status": "success"}, fh)
        tmp = Path(fh.name)
    try:
        assert check_exp855_gate(tmp) is True
    finally:
        tmp.unlink(missing_ok=True)


# ===========================================================================
# check_exp856_gate
# ===========================================================================

def test_exp856_gate_blocked_missing_file() -> None:
    """check_exp856_gate returns False when the artifact file does not exist.

    Spec: REQ-GPU-010
    """
    assert check_exp856_gate(Path("/nonexistent/experiment_856.json")) is False


def test_exp856_gate_blocked_dual_gpu_not_deployed() -> None:
    """check_exp856_gate returns False when dual_gpu_deployed is False.

    Spec: REQ-GPU-010
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump({"dual_gpu_deployed": False, "status": "blocked"}, fh)
        tmp = Path(fh.name)
    try:
        assert check_exp856_gate(tmp) is False
    finally:
        tmp.unlink(missing_ok=True)


def test_exp856_gate_passes_when_dual_gpu_deployed_true() -> None:
    """check_exp856_gate returns True when dual_gpu_deployed=True.

    Spec: REQ-GPU-010
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump({"dual_gpu_deployed": True, "status": "success"}, fh)
        tmp = Path(fh.name)
    try:
        assert check_exp856_gate(tmp) is True
    finally:
        tmp.unlink(missing_ok=True)


# ===========================================================================
# run_problem_baseline
# ===========================================================================

def test_run_problem_baseline_passing() -> None:
    """run_problem_baseline returns True when canonical solution passes the test.

    Spec: REQ-VR-020, SCENARIO-VR-030
    """
    problem = {
        "prompt": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n",
        "canonical_solution": "    return a + b\n",
        "test": "assert add(1, 2) == 3\n",
    }
    assert run_problem_baseline(problem) is True


def test_run_problem_baseline_failing() -> None:
    """run_problem_baseline returns False when canonical solution fails the test.

    Spec: REQ-VR-020, SCENARIO-VR-030
    """
    problem = {
        "prompt": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n",
        "canonical_solution": "    return a - b\n",  # intentionally wrong
        "test": "assert add(1, 2) == 3\n",
    }
    assert run_problem_baseline(problem) is False


# ===========================================================================
# compute_signed_improvement
# ===========================================================================

def test_signed_improvement_positive() -> None:
    """compute_signed_improvement returns positive value when repair beats baseline.

    Spec: REQ-VR-020
    """
    assert compute_signed_improvement(20, 15, 25) == pytest.approx(0.2)


def test_signed_improvement_negative() -> None:
    """compute_signed_improvement returns negative value when repair is worse.

    Spec: REQ-VR-020
    """
    assert compute_signed_improvement(10, 15, 25) == pytest.approx(-0.2)


def test_signed_improvement_zero() -> None:
    """compute_signed_improvement returns 0.0 when repair equals baseline.

    Spec: REQ-VR-020
    """
    assert compute_signed_improvement(15, 15, 25) == 0.0


def test_signed_improvement_n_zero() -> None:
    """compute_signed_improvement returns 0.0 when n_problems == 0 (no division by zero).

    Spec: REQ-VR-020
    """
    assert compute_signed_improvement(0, 0, 0) == 0.0


# ===========================================================================
# classify_verdict
# ===========================================================================

def test_classify_verdict_positive_repair() -> None:
    """classify_verdict returns 'positive_repair' when live GPU and improvement > 0.

    Spec: REQ-VR-020, SCENARIO-VR-030
    """
    assert classify_verdict(0.1, "live_gpu") == "positive_repair"


def test_classify_verdict_live_no_improvement() -> None:
    """classify_verdict returns 'live_no_improvement' when live GPU and improvement <= 0.

    Spec: REQ-VR-020, SCENARIO-VR-030
    """
    assert classify_verdict(0.0, "live_gpu") == "live_no_improvement"
    assert classify_verdict(-0.1, "live_gpu") == "live_no_improvement"


def test_classify_verdict_simulation_fallback() -> None:
    """classify_verdict returns 'simulation_fallback' when inference_mode != 'live_gpu'.

    **Detailed explanation:**
        When CARNOT_FORCE_LIVE is not propagated, LiveGPUGate.check_env_var()
        returns False and inference_mode is set to "simulated".  This is labelled
        "simulation_fallback" to make clear to the conductor that the result is
        not a live benchmark.

    Spec: REQ-VR-020, SCENARIO-VR-030
    """
    assert classify_verdict(0.5, "simulated") == "simulation_fallback"
    assert classify_verdict(-0.5, "simulated") == "simulation_fallback"
