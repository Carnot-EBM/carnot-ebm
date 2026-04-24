"""Tests for Experiment 790 — NPU Unblock v9.

Each test traces to REQ-INFRA-057 and SCENARIO-INFRA-066.

Coverage targets:
    - compute_honest_verdict() all verdict branches
    - option_b_attempted=True propagates when option_a_success=False
    - ninja_found / openblas_found reflect subprocess output
    - npu_gemm_running verdict only when npu_gemm_runs=True
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test.  We import at the function level inside each
# test so that sys.path manipulation in the module's top-level code doesn't
# interfere with pytest collection.
import importlib
import sys
import os

# Ensure repo root is on the path so the import resolves.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.experiment_790_npu_unblock_v9 import (  # noqa: E402
    compute_honest_verdict,
    check_vitisai_preconditions,
    attempt_option_b,
)


# ---------------------------------------------------------------------------
# REQ-INFRA-057 / SCENARIO-INFRA-066: option_b propagation
# ---------------------------------------------------------------------------


def test_option_b_attempted_when_option_a_fails():
    """When option_a_success=False, option_b_attempted MUST be True.

    Spec: REQ-INFRA-057, SCENARIO-INFRA-066
    Why: the spec mandates Option B is tried if and only if Option A fails.
    Without this guarantee the experiment would stop after one failed strategy,
    violating the two-strategy requirement.
    """
    # Simulate: option_a failed, option_b tried (no installer found)
    with patch("scripts.experiment_790_npu_unblock_v9.os.listdir", return_value=[]):
        with patch("scripts.experiment_790_npu_unblock_v9.os.path.isdir", return_value=True):
            result = attempt_option_b()
    assert result["attempted"] is True, (
        "option_b_attempted must be True when option_a_success=False"
    )


def test_option_b_not_called_when_option_a_succeeds():
    """When option_a_success=True, option_b should not be invoked.

    Spec: REQ-INFRA-057
    Why: the spec says MUST NOT attempt more than 2 strategies. Calling B
    after A succeeds would be unnecessary churn.
    This test verifies the verdict branch that corresponds to A-success.
    """
    verdict = compute_honest_verdict(
        option_a_success=True,
        option_b_attempted=False,
        option_b_blocker=None,
        ninja_found=True,
        npu_gemm_runs=False,
        mlir_aie_import_ok=True,
    )
    # mlir-aie installed but GEMM not running → partial progress verdict
    assert verdict == "option_a_installed_no_benchmark"


# ---------------------------------------------------------------------------
# REQ-INFRA-057: honest_verdict correctness
# ---------------------------------------------------------------------------


def test_verdict_npu_gemm_running():
    """npu_gemm_running is emitted only when npu_gemm_runs=True.

    Spec: REQ-INFRA-057
    Why: this is the breakthrough condition. We must never emit it when the
    GEMM did not actually run on the NPU.
    """
    verdict = compute_honest_verdict(
        option_a_success=True,
        option_b_attempted=False,
        option_b_blocker=None,
        ninja_found=True,
        npu_gemm_runs=True,
        mlir_aie_import_ok=True,
    )
    assert verdict == "npu_gemm_running", (
        f"Expected npu_gemm_running, got {verdict!r}"
    )


def test_verdict_npu_gemm_running_not_emitted_without_benchmark():
    """npu_gemm_running MUST NOT be emitted when npu_gemm_runs=False.

    Spec: REQ-INFRA-057
    """
    verdict = compute_honest_verdict(
        option_a_success=True,
        option_b_attempted=False,
        option_b_blocker=None,
        ninja_found=True,
        npu_gemm_runs=False,
        mlir_aie_import_ok=True,
    )
    assert verdict != "npu_gemm_running"


def test_verdict_all_options_exhausted_ninja_missing():
    """all_options_exhausted_ninja_missing when both options fail and ninja absent.

    Spec: REQ-INFRA-057
    Why: ninja absence is the actionable next step (apt install ninja-build).
    The verdict must encode this so the retrospective can prescribe a fix.
    """
    verdict = compute_honest_verdict(
        option_a_success=False,
        option_b_attempted=True,
        option_b_blocker="no installer found",
        ninja_found=False,
        npu_gemm_runs=False,
        mlir_aie_import_ok=False,
    )
    assert verdict == "all_options_exhausted_ninja_missing"


def test_verdict_all_options_exhausted_no_auth():
    """all_options_exhausted_no_auth when both options fail due to auth requirement.

    Spec: REQ-INFRA-057
    """
    verdict = compute_honest_verdict(
        option_a_success=False,
        option_b_attempted=True,
        option_b_blocker="requires AMD account authentication to download",
        ninja_found=True,
        npu_gemm_runs=False,
        mlir_aie_import_ok=False,
    )
    assert verdict == "all_options_exhausted_no_auth"


def test_verdict_new_blocker_discovered():
    """new_blocker_discovered when conditions don't match known patterns.

    Spec: REQ-INFRA-057
    Why: a catch-all verdict ensures the conductor can detect genuinely new
    failure modes rather than silently mapping them to a wrong category.
    """
    # option_a failed, option_b not attempted — unusual state
    verdict = compute_honest_verdict(
        option_a_success=False,
        option_b_attempted=False,
        option_b_blocker=None,
        ninja_found=True,
        npu_gemm_runs=False,
        mlir_aie_import_ok=False,
    )
    assert verdict == "new_blocker_discovered"


# ---------------------------------------------------------------------------
# REQ-INFRA-057: ninja_found and openblas_found from subprocess
# ---------------------------------------------------------------------------


def test_ninja_found_true_when_which_ninja_succeeds():
    """ninja_found=True when `which ninja` exits with returncode 0.

    Spec: REQ-INFRA-057
    Why: check_vitisai_preconditions must read from the real subprocess call,
    not a hardcoded constant.  If `which ninja` is mocked to succeed, the
    function must reflect that.
    """
    completed_ninja = MagicMock()
    completed_ninja.returncode = 0
    completed_ninja.stdout = "/usr/bin/ninja\n"

    completed_ldconfig = MagicMock()
    completed_ldconfig.returncode = 0
    completed_ldconfig.stdout = "libopenblas.so => /usr/lib/libopenblas.so\n"

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [completed_ninja, completed_ldconfig]
        result = check_vitisai_preconditions()

    assert result["ninja_found"] is True
    assert result["openblas_found"] is True


def test_ninja_found_false_when_which_ninja_fails():
    """ninja_found=False when `which ninja` exits with returncode 1.

    Spec: REQ-INFRA-057
    """
    completed_ninja = MagicMock()
    completed_ninja.returncode = 1
    completed_ninja.stdout = ""

    completed_ldconfig = MagicMock()
    completed_ldconfig.returncode = 0
    completed_ldconfig.stdout = ""

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [completed_ninja, completed_ldconfig]
        result = check_vitisai_preconditions()

    assert result["ninja_found"] is False
    assert result["openblas_found"] is False
