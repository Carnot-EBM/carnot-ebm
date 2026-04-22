"""Tests for scripts/experiment_714_npu_iron_unblock.py.

Covers:
- IRON availability detection (iron_install_ok, iron_available) (REQ-HW-039)
- VitisAI availability detection (vitis_available) (REQ-HW-039)
- honest_verdict classification for all branches (REQ-HW-039, SCENARIO-HW-039)
- cpu_gemm_benchmark returns a positive elapsed time (REQ-HW-039)
- classify_verdict correctly maps each outcome combination (REQ-HW-039)

Spec: REQ-HW-039, SCENARIO-HW-039
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_714_npu_iron_unblock import (
    MIN_SPEEDUP,
    check_iron_importable,
    check_vitisai_available,
    classify_verdict,
    cpu_gemm_benchmark,
)


# ---------------------------------------------------------------------------
# classify_verdict — pure function tests (REQ-HW-039)
# ---------------------------------------------------------------------------


def test_classify_verdict_npu_iron_working():
    """IRON installed + compiled + ran + speedup >= 2.0 → npu_iron_working.

    Spec: REQ-HW-039, SCENARIO-HW-039
    """
    result = classify_verdict(
        iron_install_ok=True,
        iron_available=True,
        iron_compile_ok=True,
        iron_run_ok=True,
        gemm_speedup=3.0,
        vitis_available=False,
    )
    assert result == "npu_iron_working"


def test_classify_verdict_npu_iron_installed_no_speedup():
    """IRON installed + ran but speedup < 2.0 → npu_iron_installed_no_speedup.

    Spec: REQ-HW-039
    """
    result = classify_verdict(
        iron_install_ok=True,
        iron_available=True,
        iron_compile_ok=True,
        iron_run_ok=True,
        gemm_speedup=1.1,
        vitis_available=False,
    )
    assert result == "npu_iron_installed_no_speedup"


def test_classify_verdict_npu_iron_speedup_exactly_at_threshold():
    """gemm_speedup == MIN_SPEEDUP should be classified as npu_iron_working.

    Spec: REQ-HW-039
    """
    result = classify_verdict(
        iron_install_ok=True,
        iron_available=True,
        iron_compile_ok=True,
        iron_run_ok=True,
        gemm_speedup=MIN_SPEEDUP,
        vitis_available=False,
    )
    assert result == "npu_iron_working"


def test_classify_verdict_npu_iron_install_failed():
    """pip install returns non-zero → npu_iron_install_failed.

    Spec: REQ-HW-039
    """
    result = classify_verdict(
        iron_install_ok=False,
        iron_available=False,
        iron_compile_ok=False,
        iron_run_ok=False,
        gemm_speedup=None,
        vitis_available=False,
    )
    assert result == "npu_iron_install_failed"


def test_classify_verdict_npu_vitisai_working():
    """IRON install failed but VitisAI is available → npu_vitisai_working.

    Why install_ok=True but run_ok=False:
        IRON package may install and even import but fail to compile/run on
        hardware that lacks AIE runtime support.  VitisAI then provides a
        fallback path.

    Spec: REQ-HW-039
    """
    result = classify_verdict(
        iron_install_ok=True,
        iron_available=True,
        iron_compile_ok=False,
        iron_run_ok=False,
        gemm_speedup=None,
        vitis_available=True,
    )
    assert result == "npu_vitisai_working"


def test_classify_verdict_npu_still_blocked():
    """IRON installs but neither IRON run nor VitisAI works → npu_still_blocked.

    Spec: REQ-HW-039
    """
    result = classify_verdict(
        iron_install_ok=True,
        iron_available=False,
        iron_compile_ok=False,
        iron_run_ok=False,
        gemm_speedup=None,
        vitis_available=False,
    )
    assert result == "npu_still_blocked"


def test_classify_verdict_iron_run_ok_none_speedup_no_speedup():
    """run_ok=True but gemm_speedup=None → treated as no_speedup (defensive).

    Spec: REQ-HW-039
    """
    result = classify_verdict(
        iron_install_ok=True,
        iron_available=True,
        iron_compile_ok=True,
        iron_run_ok=True,
        gemm_speedup=None,
        vitis_available=False,
    )
    assert result == "npu_iron_installed_no_speedup"


# ---------------------------------------------------------------------------
# cpu_gemm_benchmark — sanity checks (REQ-HW-039)
# ---------------------------------------------------------------------------


def test_cpu_gemm_benchmark_returns_positive():
    """cpu_gemm_benchmark(8, 5) should complete and return a positive float.

    Spec: REQ-HW-039
    """
    elapsed = cpu_gemm_benchmark(8, 5)
    assert isinstance(elapsed, float)
    assert elapsed > 0.0


def test_cpu_gemm_benchmark_scales_with_iterations():
    """100 iterations must take at least as long as 1 iteration (timing sanity).

    WHY 1 vs 100 (not 1 vs 50):
        On fast hardware with numpy, a single 4x4 GEMM is ~1 µs; 50 iterations
        may be indistinguishable from 1 within perf_counter resolution.  Using 1
        vs 100 iterations provides enough separation to be stable on any host.

    Spec: REQ-HW-039
    """
    t_one = cpu_gemm_benchmark(16, 1)
    t_hundred = cpu_gemm_benchmark(16, 100)
    # 100 iterations must always consume at least as much wall-time as 1.
    assert t_hundred >= t_one


# ---------------------------------------------------------------------------
# check_iron_importable — mock import (REQ-HW-039)
# ---------------------------------------------------------------------------


def test_check_iron_importable_returns_true_when_module_present():
    """check_iron_importable() returns True when mlir_aie can be imported.

    Spec: REQ-HW-039
    """
    mock_module = MagicMock()
    with patch("importlib.import_module", return_value=mock_module) as mock_import:
        result = check_iron_importable()
        mock_import.assert_called_once_with("mlir_aie")
    assert result is True


def test_check_iron_importable_returns_false_on_import_error():
    """check_iron_importable() returns False when mlir_aie raises ImportError.

    Spec: REQ-HW-039
    """
    with patch("importlib.import_module", side_effect=ImportError("no module")):
        result = check_iron_importable()
    assert result is False


# ---------------------------------------------------------------------------
# check_vitisai_available — mock subprocess + import (REQ-HW-039)
# ---------------------------------------------------------------------------


def test_check_vitisai_available_already_installed():
    """Returns True when VitisAIExecutionProvider is already in onnxruntime providers.

    Spec: REQ-HW-039
    """
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CPUExecutionProvider",
        "VitisAIExecutionProvider",
    ]
    with patch("importlib.import_module", return_value=mock_ort):
        available, detail = check_vitisai_available("/fake/pip")
    assert available is True
    assert "VitisAIExecutionProvider" in detail


def test_check_vitisai_available_not_installed_pip_fails():
    """Returns False when onnxruntime is absent and pip install fails.

    WHY we manipulate sys.modules directly:
        Patching importlib.import_module globally breaks the import machinery for
        other modules (subprocess, importlib.util).  Inserting a sentinel into
        sys.modules is the narrowest way to control what the code sees when it
        executes ``importlib.import_module("onnxruntime")``.

    Spec: REQ-HW-039
    """
    import sys  # noqa: PLC0415

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "ERROR: Could not find package onnxruntime-vitisai"

    # Remove onnxruntime from sys.modules so the first import attempt raises ImportError.
    saved = sys.modules.pop("onnxruntime", None)
    try:
        with patch("subprocess.run", return_value=mock_proc):
            available, detail = check_vitisai_available("/fake/pip")
    finally:
        if saved is not None:
            sys.modules["onnxruntime"] = saved

    assert available is False
    assert "failed" in detail.lower() or "error" in detail.lower() or "pip" in detail.lower()


def test_check_vitisai_available_pip_succeeds_but_provider_absent():
    """Returns False when pip install succeeds but VitisAI provider not in list.

    WHY we use sys.modules injection:
        After pip install succeeds, the code re-imports onnxruntime.  By placing a
        mock with only CPUExecutionProvider in sys.modules, we simulate a host where
        the VitisAI EP is not compiled in — without touching importlib globally.

    Spec: REQ-HW-039
    """
    import sys  # noqa: PLC0415

    mock_proc = MagicMock()
    mock_proc.returncode = 0

    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    # Remove real onnxruntime (if installed) and inject mock.
    saved = sys.modules.pop("onnxruntime", None)
    sys.modules["onnxruntime"] = mock_ort

    # importlib.util.find_spec must see the mocked module.
    import importlib.util  # noqa: PLC0415

    mock_spec = MagicMock()
    try:
        with (
            patch("subprocess.run", return_value=mock_proc),
            patch("importlib.util.find_spec", return_value=mock_spec),
        ):
            available, detail = check_vitisai_available("/fake/pip")
    finally:
        del sys.modules["onnxruntime"]
        if saved is not None:
            sys.modules["onnxruntime"] = saved

    assert available is False
    assert "VitisAI" in detail or "absent" in detail or "provider" in detail.lower()
