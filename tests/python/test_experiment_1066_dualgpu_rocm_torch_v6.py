"""Tests for Experiment 1066 — DualGPU live confirmation v6 (post-respawn).

These tests cover the helpers in
``scripts/experiment_1066_dualgpu_rocm_torch_v6.py`` and confirm the
artifact at ``results/experiment_1066_dualgpu_rocm_torch_v6.json``
satisfies the required schema fields named in the conductor's task spec.

We deliberately do **not** re-install torch in the test environment or
exercise GPU hardware; the live tensor smoke test is run by the
experiment script itself, and re-running it from the test suite would
make the suite host-dependent.

Spec: REQ-INFRA-007, SCENARIO-INFRA-011, RETRO-DUALGPU-2026-04.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import experiment_1066_dualgpu_rocm_torch_v6 as exp1066  # noqa: E402


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------


def test_pytest_expected_sum_for_32x32_add():
    """Reference-sum constant matches 32 * 32 * (1 + 1)."""
    assert exp1066.pytest_expected_sum_for_32x32_add() == 2048.0


def test_diagnose_torch_returns_expected_keys():
    """``diagnose_torch`` always returns the documented dict shape.

    Why: the artifact builder reads these keys unconditionally; a missing
    key would surface as a KeyError that masks the actual diagnostic.
    """
    info = exp1066.diagnose_torch()
    expected = {
        "torch_importable",
        "torch_version",
        "torch_cuda_available",
        "torch_cuda_device_count",
        "torch_hip_version",
        "import_error",
    }
    assert expected.issubset(info.keys())
    assert isinstance(info["torch_cuda_device_count"], int)


def test_nvidia_smi_gpu_count_returns_non_negative_int():
    """``nvidia_smi_gpu_count`` swallows missing-binary errors."""
    n = exp1066.nvidia_smi_gpu_count()
    assert isinstance(n, int)
    assert n >= 0


def test_run_tensor_smoke_test_no_gpu_returns_not_attempted():
    """When ``gpu_count == 0`` the smoke test reports no attempt rather than
    silently passing.

    Why: we must not let ``all_passed=True`` leak through on a CPU host —
    that would falsely raise ``honest_verdict`` to ``dualgpu_live_confirmed``.
    """
    result = exp1066.run_tensor_smoke_test(0)
    assert result["smoke_test_attempted"] is False
    assert result["all_passed"] is False
    assert result["per_gpu_passed"] == []


# ---------------------------------------------------------------------------
# Install-path decision logic
# ---------------------------------------------------------------------------


def test_determine_install_path_already_installed():
    """When CUDA is already up, the script must NOT re-install torch.

    Why: re-installing a working torch wheel risks regressing a verified-
    good environment for zero benefit; the verdict still credits Path A.
    """
    before = {"torch_cuda_available": True, "torch_cuda_device_count": 2}
    path, ok = exp1066.determine_install_path(before)
    assert path == "cuda12_already_installed"
    assert ok is True


def test_determine_install_path_when_not_installed():
    """When CUDA is missing, the planned install path is plain ``cuda12``."""
    before = {"torch_cuda_available": False, "torch_cuda_device_count": 0}
    path, ok = exp1066.determine_install_path(before)
    assert path == "cuda12"
    assert ok is False


def test_determine_install_path_handles_missing_keys():
    """Robust to a partial diagnostic dict — defaults to "cuda12 not done"."""
    path, ok = exp1066.determine_install_path({})
    assert path == "cuda12"
    assert ok is False


# ---------------------------------------------------------------------------
# Verdict-derivation table — every documented branch
# ---------------------------------------------------------------------------


def test_derive_honest_verdict_dualgpu_confirmed():
    v = exp1066.derive_honest_verdict(
        install_succeeded=True,
        gpu_count_detected=2,
        dualgpu_live=True,
        smoke_all_passed=True,
    )
    assert v == "dualgpu_live_confirmed"


def test_derive_honest_verdict_single_gpu_smoke_pass():
    v = exp1066.derive_honest_verdict(
        install_succeeded=True,
        gpu_count_detected=1,
        dualgpu_live=False,
        smoke_all_passed=True,
    )
    assert v == "torch_installed_smoke_passed"


def test_derive_honest_verdict_install_only_no_smoke():
    """Install OK but smoke test absent — still credit the install."""
    v = exp1066.derive_honest_verdict(
        install_succeeded=True,
        gpu_count_detected=1,
        dualgpu_live=False,
        smoke_all_passed=False,
    )
    assert v == "torch_installed_smoke_passed"


def test_derive_honest_verdict_all_paths_failed():
    v = exp1066.derive_honest_verdict(
        install_succeeded=False,
        gpu_count_detected=0,
        dualgpu_live=False,
        smoke_all_passed=False,
    )
    assert v == "all_paths_failed"


# ---------------------------------------------------------------------------
# Artifact contract — fields the conductor's task spec demands
# ---------------------------------------------------------------------------


def test_artifact_present_and_schema_complete():
    """The deliverable JSON exists and contains every required field."""
    deliverable = (
        Path(__file__).parent.parent.parent
        / "results"
        / "experiment_1066_dualgpu_rocm_torch_v6.json"
    )
    assert deliverable.is_file(), f"missing deliverable {deliverable}"
    payload = json.loads(deliverable.read_text())
    required = [
        "torch_version_before",
        "torch_cuda_before",
        "install_path_tried",
        "install_path_succeeded",
        "torch_version_after",
        "gpu_count_detected",
        "dualgpu_live",
        "honest_verdict",
        # ExperimentTemplate.REQUIRED_RESULT_FIELDS
        "experiment",
        "run_date",
        "started_at",
        "finished_at",
        "duration_s",
        "status",
        "title",
    ]
    missing = [field for field in required if field not in payload]
    assert not missing, f"deliverable missing fields: {missing}"
    assert payload["experiment"] == 1066
    assert payload["honest_verdict"] in {
        "dualgpu_live_confirmed",
        "torch_installed_smoke_passed",
        "llamacpp_path_only",
        "all_paths_failed",
        "failed",
    }
