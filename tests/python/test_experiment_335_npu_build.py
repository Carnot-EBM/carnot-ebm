"""Tests for Experiment 335: AMD XDNA NPU build — Exp 335 prereq retry with ORT source build.

Exp 335 is the third retry of the VitisAI ORT source build that has been blocked
since Exp 292 by missing `ninja` and `openblas` system packages.  This test file
validates:
  1. The pure-logic helper functions (check_ninja_available, check_openblas_available,
     check_xrt_available, check_amdxdna_module_loaded, prereq_status) using mocks so
     the test suite does not depend on the host system's package state.
  2. The attempt_ort_source_build() return-schema contract (not the long build itself).
  3. The prereq_changes_vs_exp314() comparison logic.
  4. The honest_verdict assignment rules and artifact schema.
  5. The blocked-path invariants (SCENARIO-EXP303-D): no fabricated latency.

All tests are parametric-safe and work regardless of whether ninja/openblas are
actually installed on the CI host.

Spec:
  REQ-PRED-003 (honest labeling — no fabricated latency on blocked paths)
  SCENARIO-EXP303-A (prereq check — detection with install_command)
  SCENARIO-EXP303-B (source build path — timeout_s, log tail on failure)
  SCENARIO-EXP303-C (inference benchmark — npu vs cpu latency when working)
  SCENARIO-EXP303-D (honest labeling — null inference_result on blocked paths)
  SCENARIO-EXP303-E (Exp 335 still blocked — prereq state unchanged from Exp 314)
  SCENARIO-EXP303-F (Exp 335 build attempted — prereqs now met)

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/pytest tests/python/test_experiment_335_npu_build.py -v
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Path setup — import the script under test
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# We import the module at the bottom of each test so mocks can be applied;
# however, we lazily import once for unit tests that do not require subprocess.
import importlib

import experiment_335_npu_build as _mod


# ---------------------------------------------------------------------------
# Schema constants (mirrors what the script must produce)
# ---------------------------------------------------------------------------

REQUIRED_TOP_LEVEL_KEYS = {
    "experiment",
    "description",
    "run_date",
    "honest_verdict",
    "prereq_status",
    "prereq_changes_vs_exp314",
    "build_attempt_result",
    "npu_inference_result",
    "next_steps",
}

VALID_VERDICTS = {
    "blocked_prereq",
    "build_failed",
    "inference_success",
    "timeout",
}

REQUIRED_PREREQ_STATUS_KEYS = {
    "ninja_available",
    "openblas_available",
    "xrt_available",
    "amdxdna_module_loaded",
    "all_met",
}

REQUIRED_PREREQ_CHANGE_KEYS = {"ninja", "openblas"}

VALID_PREREQ_CHANGE_VALUES = {"now_available", "still_missing"}

REQUIRED_BUILD_ATTEMPT_KEYS = {"success", "duration_seconds"}

REQUIRED_INFERENCE_KEYS = {
    "npu_latency_us",
    "cpu_latency_us",
    "speedup_factor",
    "provider_used",
}

BLOCKED_VERDICTS = {"blocked_prereq", "build_failed", "timeout"}


# ---------------------------------------------------------------------------
# Unit tests: check_ninja_available()
# ---------------------------------------------------------------------------


class TestCheckNinjaAvailable:
    """check_ninja_available() runs `ninja --version` and returns bool.

    WHY: The function must not raise even when ninja is absent; it must
    return False cleanly so the caller can make a blocked_prereq decision.

    Spec: SCENARIO-EXP303-A
    """

    def test_returns_true_when_ninja_present(self) -> None:
        """Returns True when `ninja --version` exits 0."""
        result = mock.MagicMock()
        result.returncode = 0
        with mock.patch("subprocess.run", return_value=result):
            assert _mod.check_ninja_available() is True

    def test_returns_false_when_ninja_absent(self) -> None:
        """Returns False when `ninja --version` exits non-zero."""
        result = mock.MagicMock()
        result.returncode = 1
        with mock.patch("subprocess.run", return_value=result):
            assert _mod.check_ninja_available() is False

    def test_returns_false_on_file_not_found(self) -> None:
        """Returns False when ninja binary does not exist at all (FileNotFoundError)."""
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            assert _mod.check_ninja_available() is False

    def test_returns_false_on_timeout(self) -> None:
        """Returns False when the subprocess times out rather than crashing."""
        with mock.patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("ninja", 5)
        ):
            assert _mod.check_ninja_available() is False

    def test_calls_ninja_version(self) -> None:
        """Must invoke `ninja --version`, not some other command."""
        captured: list[Any] = []
        result = mock.MagicMock()
        result.returncode = 0

        def _fake_run(cmd, **kwargs):  # noqa: ANN001
            captured.append(cmd)
            return result

        with mock.patch("subprocess.run", side_effect=_fake_run):
            _mod.check_ninja_available()

        assert captured, "subprocess.run was never called"
        assert "ninja" in captured[0], f"Expected 'ninja' in cmd, got: {captured[0]}"


# ---------------------------------------------------------------------------
# Unit tests: check_openblas_available()
# ---------------------------------------------------------------------------


class TestCheckOpenblasAvailable:
    """check_openblas_available() detects openblas via pkg-config or ldconfig.

    WHY: openblas absence was the second blocker in Exps 292/303/314.
    We test both detection paths and the clean-False fallback.

    Spec: SCENARIO-EXP303-A
    """

    def test_returns_true_when_pkg_config_finds_openblas(self) -> None:
        """Returns True when `pkg-config --exists openblas` exits 0."""
        result = mock.MagicMock()
        result.returncode = 0
        with mock.patch("subprocess.run", return_value=result):
            assert _mod.check_openblas_available() is True

    def test_falls_back_to_ldconfig_when_pkg_config_fails(self) -> None:
        """Falls back to ldconfig search when pkg-config not found/fails."""
        fail = mock.MagicMock()
        fail.returncode = 1
        fail.stdout = ""

        success = mock.MagicMock()
        success.returncode = 0
        success.stdout = "libopenblas.so.0 -> /usr/lib/libopenblas.so.0"

        call_count = [0]

        def _side_effect(cmd, **kwargs):  # noqa: ANN001
            call_count[0] += 1
            if call_count[0] == 1:
                return fail  # pkg-config fails
            return success  # ldconfig succeeds

        with mock.patch("subprocess.run", side_effect=_side_effect):
            assert _mod.check_openblas_available() is True

    def test_returns_false_when_both_checks_fail(self) -> None:
        """Returns False when both pkg-config and ldconfig find nothing."""
        fail = mock.MagicMock()
        fail.returncode = 1
        fail.stdout = ""
        with mock.patch("subprocess.run", return_value=fail):
            assert _mod.check_openblas_available() is False

    def test_returns_false_on_file_not_found(self) -> None:
        """Returns False when no detection tool is available (FileNotFoundError)."""
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            assert _mod.check_openblas_available() is False

    def test_return_type_is_bool(self) -> None:
        """Return value must be exactly bool, not int or truthy/falsy object."""
        result = mock.MagicMock()
        result.returncode = 0
        with mock.patch("subprocess.run", return_value=result):
            v = _mod.check_openblas_available()
        assert type(v) is bool, f"Expected bool, got {type(v)}"


# ---------------------------------------------------------------------------
# Unit tests: check_xrt_available()
# ---------------------------------------------------------------------------


class TestCheckXrtAvailable:
    """check_xrt_available() checks /opt/xilinx/xrt/ directory existence.

    WHY: XRT (Xilinx Runtime) is a hard dependency for the VitisAI EP.
    Checking a filesystem path is faster and more reliable than subprocess.

    Spec: SCENARIO-EXP303-A
    """

    def test_returns_true_when_xrt_dir_exists(self, tmp_path: Path) -> None:
        """Returns True when the XRT directory is present."""
        xrt_dir = tmp_path / "xrt"
        xrt_dir.mkdir()
        with mock.patch.object(_mod, "_XRT_DIR", xrt_dir):
            assert _mod.check_xrt_available() is True

    def test_returns_false_when_xrt_dir_absent(self, tmp_path: Path) -> None:
        """Returns False when the XRT directory does not exist."""
        xrt_dir = tmp_path / "xrt_missing"
        # Intentionally not created
        with mock.patch.object(_mod, "_XRT_DIR", xrt_dir):
            assert _mod.check_xrt_available() is False

    def test_return_type_is_bool(self, tmp_path: Path) -> None:
        """Return type must be bool."""
        xrt_dir = tmp_path / "xrt"
        xrt_dir.mkdir()
        with mock.patch.object(_mod, "_XRT_DIR", xrt_dir):
            v = _mod.check_xrt_available()
        assert type(v) is bool


# ---------------------------------------------------------------------------
# Unit tests: check_amdxdna_module_loaded()
# ---------------------------------------------------------------------------


class TestCheckAmdxdnaModuleLoaded:
    """check_amdxdna_module_loaded() inspects lsmod output for amdxdna.

    WHY: Even when XRT is installed, the NPU kernel module may not be loaded.
    This check tells the researcher whether the driver is active without
    requiring root access (lsmod is world-readable).

    Spec: SCENARIO-EXP303-A
    """

    def test_returns_true_when_module_present(self) -> None:
        """Returns True when lsmod output contains 'amdxdna'."""
        result = mock.MagicMock()
        result.returncode = 0
        result.stdout = "amdxdna               122880  0\n"
        with mock.patch("subprocess.run", return_value=result):
            assert _mod.check_amdxdna_module_loaded() is True

    def test_returns_false_when_module_absent(self) -> None:
        """Returns False when lsmod output does not mention amdxdna."""
        result = mock.MagicMock()
        result.returncode = 0
        result.stdout = "snd_hda_intel         110592  4\nnvme                   77824  0\n"
        with mock.patch("subprocess.run", return_value=result):
            assert _mod.check_amdxdna_module_loaded() is False

    def test_returns_false_when_lsmod_fails(self) -> None:
        """Returns False (not raises) when lsmod exits non-zero."""
        result = mock.MagicMock()
        result.returncode = 1
        result.stdout = ""
        with mock.patch("subprocess.run", return_value=result):
            assert _mod.check_amdxdna_module_loaded() is False

    def test_returns_false_on_file_not_found(self) -> None:
        """Returns False when lsmod is not available on the system."""
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            assert _mod.check_amdxdna_module_loaded() is False

    def test_return_type_is_bool(self) -> None:
        """Return type must be bool, not a truthy string."""
        result = mock.MagicMock()
        result.returncode = 0
        result.stdout = "amdxdna 122880 0\n"
        with mock.patch("subprocess.run", return_value=result):
            v = _mod.check_amdxdna_module_loaded()
        assert type(v) is bool


# ---------------------------------------------------------------------------
# Unit tests: prereq_status()
# ---------------------------------------------------------------------------


class TestPrereqStatus:
    """prereq_status() aggregates all four individual checks into one dict.

    WHY: The main() function only calls prereq_status() once; it does not call
    the individual checks directly.  This simplifies the decision logic and
    makes the aggregate state easily serializable.

    Spec: SCENARIO-EXP303-A
    """

    def _patch_all(
        self,
        ninja: bool = True,
        openblas: bool = True,
        xrt: bool = True,
        amdxdna: bool = True,
    ):
        """Return a context manager that patches all four check functions."""
        return mock.patch.multiple(
            _mod,
            check_ninja_available=mock.MagicMock(return_value=ninja),
            check_openblas_available=mock.MagicMock(return_value=openblas),
            check_xrt_available=mock.MagicMock(return_value=xrt),
            check_amdxdna_module_loaded=mock.MagicMock(return_value=amdxdna),
        )

    def test_returns_dict_with_required_keys(self) -> None:
        """Returns a dict containing all REQUIRED_PREREQ_STATUS_KEYS."""
        with self._patch_all():
            result = _mod.prereq_status()
        missing = REQUIRED_PREREQ_STATUS_KEYS - set(result.keys())
        assert not missing, f"prereq_status() missing keys: {missing}"

    def test_all_met_true_when_all_checks_pass(self) -> None:
        """all_met is True when all four checks return True."""
        with self._patch_all(ninja=True, openblas=True, xrt=True, amdxdna=True):
            result = _mod.prereq_status()
        assert result["all_met"] is True

    def test_all_met_false_when_ninja_missing(self) -> None:
        """all_met is False when ninja is absent (SCENARIO-EXP303-E)."""
        with self._patch_all(ninja=False):
            result = _mod.prereq_status()
        assert result["all_met"] is False

    def test_all_met_false_when_openblas_missing(self) -> None:
        """all_met is False when openblas is absent (SCENARIO-EXP303-E)."""
        with self._patch_all(openblas=False):
            result = _mod.prereq_status()
        assert result["all_met"] is False

    def test_all_met_false_when_both_missing(self) -> None:
        """all_met is False when both ninja and openblas are absent (Exp 314 state)."""
        with self._patch_all(ninja=False, openblas=False):
            result = _mod.prereq_status()
        assert result["all_met"] is False

    def test_individual_flags_reflect_checks(self) -> None:
        """Each individual key reflects the corresponding check result."""
        with self._patch_all(ninja=True, openblas=False, xrt=True, amdxdna=False):
            result = _mod.prereq_status()
        assert result["ninja_available"] is True
        assert result["openblas_available"] is False
        assert result["xrt_available"] is True
        assert result["amdxdna_module_loaded"] is False


# ---------------------------------------------------------------------------
# Unit tests: prereq_changes_vs_exp314()
# ---------------------------------------------------------------------------


class TestPrereqChangesVsExp314:
    """prereq_changes_vs_exp314() computes delta from Exp 314's blocked state.

    WHY: Exp 314 had both ninja_installed=False and openblas_installed=False.
    This function tells the researcher which packages changed since then without
    requiring them to diff two JSON files.

    Spec: SCENARIO-EXP303-E, SCENARIO-EXP303-F
    """

    def test_both_still_missing(self) -> None:
        """Both 'still_missing' when neither package changed (matches Exp 314 state)."""
        status = {
            "ninja_available": False,
            "openblas_available": False,
            "xrt_available": True,
            "amdxdna_module_loaded": False,
            "all_met": False,
        }
        changes = _mod.prereq_changes_vs_exp314(status)
        assert changes["ninja"] == "still_missing"
        assert changes["openblas"] == "still_missing"

    def test_both_now_available(self) -> None:
        """Both 'now_available' when both packages are now installed."""
        status = {
            "ninja_available": True,
            "openblas_available": True,
            "xrt_available": True,
            "amdxdna_module_loaded": True,
            "all_met": True,
        }
        changes = _mod.prereq_changes_vs_exp314(status)
        assert changes["ninja"] == "now_available"
        assert changes["openblas"] == "now_available"

    def test_ninja_now_available_openblas_still_missing(self) -> None:
        """Partial install: ninja installed but openblas still missing."""
        status = {
            "ninja_available": True,
            "openblas_available": False,
            "xrt_available": True,
            "amdxdna_module_loaded": False,
            "all_met": False,
        }
        changes = _mod.prereq_changes_vs_exp314(status)
        assert changes["ninja"] == "now_available"
        assert changes["openblas"] == "still_missing"

    def test_returns_only_ninja_and_openblas_keys(self) -> None:
        """Returns a dict with exactly the ninja and openblas keys."""
        status = {
            "ninja_available": True,
            "openblas_available": True,
            "xrt_available": True,
            "amdxdna_module_loaded": True,
            "all_met": True,
        }
        changes = _mod.prereq_changes_vs_exp314(status)
        assert set(changes.keys()) == REQUIRED_PREREQ_CHANGE_KEYS

    def test_values_in_valid_vocabulary(self) -> None:
        """Values are strictly 'now_available' or 'still_missing' (controlled vocabulary)."""
        for ninja in (True, False):
            for openblas in (True, False):
                status = {
                    "ninja_available": ninja,
                    "openblas_available": openblas,
                    "xrt_available": True,
                    "amdxdna_module_loaded": True,
                    "all_met": ninja and openblas,
                }
                changes = _mod.prereq_changes_vs_exp314(status)
                assert changes["ninja"] in VALID_PREREQ_CHANGE_VALUES
                assert changes["openblas"] in VALID_PREREQ_CHANGE_VALUES


# ---------------------------------------------------------------------------
# Unit tests: attempt_ort_source_build() — schema contract only
# ---------------------------------------------------------------------------


class TestAttemptOrtSourceBuildSchema:
    """attempt_ort_source_build() must always return a correctly shaped dict.

    We do NOT actually run cmake here (that would take 10+ minutes).  Instead we
    mock subprocess.run to return controlled outcomes and verify the return schema.

    Spec: SCENARIO-EXP303-B, SCENARIO-EXP303-F
    """

    def _make_ok_proc(self, stdout: str = "", returncode: int = 0) -> mock.MagicMock:
        p = mock.MagicMock()
        p.returncode = returncode
        p.stdout = stdout
        p.stderr = ""
        return p

    def _make_fail_proc(
        self, stdout: str = "", stderr: str = "error", returncode: int = 1
    ) -> mock.MagicMock:
        p = mock.MagicMock()
        p.returncode = returncode
        p.stdout = stdout
        p.stderr = stderr
        return p

    def test_clone_failure_returns_schema(self, tmp_path: Path) -> None:
        """Clone failure returns a dict with all required build keys."""
        fail = self._make_fail_proc(stderr="repository not found")
        with mock.patch("subprocess.run", return_value=fail):
            result = _mod.attempt_ort_source_build(tmp_path / "ort_build", timeout_s=60)
        assert "success" in result
        assert result["success"] is False
        assert "duration_seconds" in result
        assert isinstance(result["duration_seconds"], float)
        assert "error_summary" in result
        assert "build_log_tail" in result
        assert isinstance(result["build_log_tail"], list)
        assert "timeout_exceeded" in result

    def test_cmake_configure_failure_returns_schema(self, tmp_path: Path) -> None:
        """cmake configure failure returns schema with error_summary."""
        # Clone succeeds, cmake fails
        ok = self._make_ok_proc()
        fail = self._make_fail_proc(stderr="CMake Error: could not find ninja")
        call_count = [0]

        def _side_effect(cmd, **kwargs):  # noqa: ANN001
            call_count[0] += 1
            if call_count[0] == 1:
                return ok  # git clone succeeds
            return fail  # cmake configure fails

        build_dir = tmp_path / "ort_build"
        build_dir.mkdir()  # Simulate existing clone to skip the clone step

        with mock.patch("subprocess.run", side_effect=_side_effect):
            result = _mod.attempt_ort_source_build(build_dir, timeout_s=60)

        assert "success" in result
        assert "duration_seconds" in result
        assert "timeout_exceeded" in result

    def test_timeout_sets_timeout_exceeded(self, tmp_path: Path) -> None:
        """When subprocess.TimeoutExpired is raised, timeout_exceeded is True."""
        build_dir = tmp_path / "ort_build"
        build_dir.mkdir()

        ok = self._make_ok_proc()

        call_count = [0]

        def _side_effect(cmd, **kwargs):  # noqa: ANN001
            call_count[0] += 1
            if call_count[0] == 1:
                return ok  # cmake configure ok
            raise subprocess.TimeoutExpired("cmake", 60)

        with mock.patch("subprocess.run", side_effect=_side_effect):
            result = _mod.attempt_ort_source_build(build_dir, timeout_s=60)

        assert result["timeout_exceeded"] is True
        assert result["success"] is False

    def test_success_returns_whl_path(self, tmp_path: Path) -> None:
        """On success, whl_path is a string pointing to the built wheel."""
        build_dir = tmp_path / "ort_build"
        build_dir.mkdir()

        # Create a fake wheel file
        dist_dir = build_dir / "build_vitisai" / "dist"
        dist_dir.mkdir(parents=True)
        fake_whl = dist_dir / "onnxruntime_vitisai-1.20.1-cp311-cp311-linux_x86_64.whl"
        fake_whl.touch()

        ok = self._make_ok_proc()
        with mock.patch("subprocess.run", return_value=ok):
            result = _mod.attempt_ort_source_build(build_dir, timeout_s=60)

        assert result["success"] is True
        assert result.get("whl_path") is not None
        assert isinstance(result["whl_path"], str)


# ---------------------------------------------------------------------------
# Integration tests: artifact schema from results file (if generated)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results_json() -> dict[str, Any]:
    """Load the Exp 335 results JSON artifact — skip if not generated yet."""
    path = _REPO_ROOT / "results" / "experiment_335_npu_build.json"
    if not path.exists():
        pytest.skip(f"Results artifact not yet generated: {path}")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def exp314_json() -> dict[str, Any]:
    """Load the Exp 314 reference artifact — skip if not present."""
    path = _REPO_ROOT / "results" / "experiment_314_npu_prereq_install.json"
    if not path.exists():
        pytest.skip("Exp 314 artifact not found — cannot compare prereq states")
    return json.loads(path.read_text())


class TestExp335ArtifactSchema:
    """Top-level artifact schema — valid for all execution paths.

    Spec: REQ-PRED-003 — artifacts must be schema-valid for downstream consumers.
    """

    def test_top_level_keys_present(self, results_json: dict[str, Any]) -> None:
        """All required top-level keys must be present."""
        missing = REQUIRED_TOP_LEVEL_KEYS - set(results_json.keys())
        assert not missing, f"Top-level keys missing: {missing}"

    def test_experiment_number_is_335(self, results_json: dict[str, Any]) -> None:
        """experiment field must be 335."""
        assert results_json["experiment"] == 335

    def test_run_date_format(self, results_json: dict[str, Any]) -> None:
        """run_date must be an 8-digit YYYYMMDD string."""
        run_date = results_json["run_date"]
        assert isinstance(run_date, str)
        assert re.fullmatch(r"\d{8}", run_date), f"run_date must be YYYYMMDD, got: {run_date!r}"

    def test_honest_verdict_is_valid(self, results_json: dict[str, Any]) -> None:
        """honest_verdict must be one of the four defined values."""
        v = results_json["honest_verdict"]
        assert v in VALID_VERDICTS, f"honest_verdict must be in {VALID_VERDICTS}, got: {v!r}"

    def test_prereq_status_is_dict(self, results_json: dict[str, Any]) -> None:
        """prereq_status must be a dict."""
        assert isinstance(results_json["prereq_status"], dict)

    def test_prereq_status_has_required_keys(self, results_json: dict[str, Any]) -> None:
        """prereq_status must contain all required keys."""
        ps = results_json["prereq_status"]
        missing = REQUIRED_PREREQ_STATUS_KEYS - set(ps.keys())
        assert not missing, f"prereq_status missing keys: {missing}"

    def test_prereq_changes_is_dict(self, results_json: dict[str, Any]) -> None:
        """prereq_changes_vs_exp314 must be a dict."""
        assert isinstance(results_json["prereq_changes_vs_exp314"], dict)

    def test_prereq_changes_has_required_keys(self, results_json: dict[str, Any]) -> None:
        """prereq_changes_vs_exp314 must have ninja and openblas keys."""
        missing = REQUIRED_PREREQ_CHANGE_KEYS - set(results_json["prereq_changes_vs_exp314"])
        assert not missing

    def test_prereq_change_values_are_valid(self, results_json: dict[str, Any]) -> None:
        """All prereq_changes_vs_exp314 values must be in the controlled vocabulary."""
        for key, val in results_json["prereq_changes_vs_exp314"].items():
            assert val in VALID_PREREQ_CHANGE_VALUES, (
                f"prereq_changes_vs_exp314[{key!r}] = {val!r} not in {VALID_PREREQ_CHANGE_VALUES}"
            )

    def test_next_steps_is_list(self, results_json: dict[str, Any]) -> None:
        """next_steps must be a list."""
        assert isinstance(results_json["next_steps"], list)

    def test_build_attempt_result_type(self, results_json: dict[str, Any]) -> None:
        """build_attempt_result must be None or a dict."""
        bar = results_json["build_attempt_result"]
        assert bar is None or isinstance(bar, dict)

    def test_npu_inference_result_type(self, results_json: dict[str, Any]) -> None:
        """npu_inference_result must be None or a dict."""
        nir = results_json["npu_inference_result"]
        assert nir is None or isinstance(nir, dict)


class TestExp335BlockedPrereq:
    """SCENARIO-EXP303-E: Exp 335 still blocked — prereq state matches Exp 314.

    When ninja or openblas is still missing, the artifact must follow the
    blocked_prereq contract: no build attempted, no fabricated latency.
    """

    @pytest.fixture(autouse=True)
    def _require_blocked_prereq(self, results_json: dict[str, Any]) -> None:
        if results_json.get("honest_verdict") != "blocked_prereq":
            pytest.skip("honest_verdict != blocked_prereq — skipping blocked tests")

    def test_build_attempt_result_is_none(self, results_json: dict[str, Any]) -> None:
        """build_attempt_result must be None when prereqs are still missing.

        No build was attempted, so no build artifact should appear.
        Spec: SCENARIO-EXP303-D
        """
        assert results_json["build_attempt_result"] is None

    def test_npu_inference_result_is_none(self, results_json: dict[str, Any]) -> None:
        """npu_inference_result must be None on blocked_prereq path.

        Fabricating latency values would corrupt the research record.
        Spec: REQ-PRED-003, SCENARIO-EXP303-D
        """
        assert results_json["npu_inference_result"] is None

    def test_prereq_status_all_met_false(self, results_json: dict[str, Any]) -> None:
        """all_met must be False on blocked_prereq path."""
        assert results_json["prereq_status"]["all_met"] is False

    def test_prereq_changes_consistent_with_exp314(
        self,
        results_json: dict[str, Any],
        exp314_json: dict[str, Any],
    ) -> None:
        """prereq_changes_vs_exp314 must correctly reflect delta from Exp 314.

        Exp 314 had ninja_installed=False, openblas_installed=False.
        If still missing in Exp 335, both must be 'still_missing'.
        Spec: SCENARIO-EXP303-E
        """
        exp314_pc = exp314_json["prereq_check"]
        exp335_ps = results_json["prereq_status"]
        exp335_chg = results_json["prereq_changes_vs_exp314"]

        ninja_now = exp335_ps["ninja_available"]
        expected_ninja = "now_available" if ninja_now else "still_missing"
        assert exp335_chg["ninja"] == expected_ninja

        openblas_now = exp335_ps["openblas_available"]
        expected_openblas = "now_available" if openblas_now else "still_missing"
        assert exp335_chg["openblas"] == expected_openblas

    def test_next_steps_names_install_command(self, results_json: dict[str, Any]) -> None:
        """next_steps must include install commands for the missing packages.

        The researcher should not need to look up install commands separately.
        Spec: SCENARIO-EXP303-E
        """
        steps = results_json["next_steps"]
        assert any("pacman" in s or "apt" in s for s in steps), (
            f"next_steps should contain pacman or apt install commands, got: {steps}"
        )


class TestExp335BuildAttempted:
    """SCENARIO-EXP303-F: Exp 335 build attempted — prereqs now met.

    When all prereqs are satisfied, honest_verdict must not be 'blocked_prereq'.
    """

    @pytest.fixture(autouse=True)
    def _require_build_attempted(self, results_json: dict[str, Any]) -> None:
        if results_json.get("honest_verdict") == "blocked_prereq":
            pytest.skip("honest_verdict == blocked_prereq — build was not attempted")

    def test_prereq_status_all_met_true(self, results_json: dict[str, Any]) -> None:
        """all_met must be True when build was attempted.

        Spec: SCENARIO-EXP303-F
        """
        assert results_json["prereq_status"]["all_met"] is True

    def test_build_attempt_result_present(self, results_json: dict[str, Any]) -> None:
        """build_attempt_result must be a dict when build was attempted.

        Spec: SCENARIO-EXP303-F
        """
        bar = results_json["build_attempt_result"]
        assert isinstance(bar, dict), f"Expected dict, got: {type(bar)}"

    def test_build_attempt_result_has_required_keys(
        self, results_json: dict[str, Any]
    ) -> None:
        """build_attempt_result must have success and duration_seconds."""
        bar = results_json["build_attempt_result"]
        missing = REQUIRED_BUILD_ATTEMPT_KEYS - set(bar.keys())
        assert not missing, f"build_attempt_result missing keys: {missing}"

    def test_npu_inference_result_null_unless_success(
        self, results_json: dict[str, Any]
    ) -> None:
        """npu_inference_result must be None unless honest_verdict == 'inference_success'.

        Spec: REQ-PRED-003, SCENARIO-EXP303-D, SCENARIO-EXP303-F
        """
        verdict = results_json["honest_verdict"]
        nir = results_json["npu_inference_result"]
        if verdict != "inference_success":
            assert nir is None, (
                f"Verdict {verdict!r} must not have fabricated npu_inference_result, "
                f"got: {nir!r}"
            )

    def test_prereq_changes_shows_now_available(
        self, results_json: dict[str, Any]
    ) -> None:
        """When build was attempted, ninja and openblas must be 'now_available'.

        Spec: SCENARIO-EXP303-F
        """
        chg = results_json["prereq_changes_vs_exp314"]
        assert chg["ninja"] == "now_available", (
            f"ninja should be 'now_available' when build was attempted, got: {chg['ninja']!r}"
        )
        assert chg["openblas"] == "now_available", (
            f"openblas should be 'now_available' when build was attempted, "
            f"got: {chg['openblas']!r}"
        )


class TestExp335InferenceSuccess:
    """Validate npu_inference_result when honest_verdict == 'inference_success'.

    Spec: SCENARIO-EXP303-C, SCENARIO-EXP303-F
    """

    @pytest.fixture(autouse=True)
    def _require_inference_success(self, results_json: dict[str, Any]) -> None:
        if results_json.get("honest_verdict") != "inference_success":
            pytest.skip("honest_verdict != inference_success — no inference result to validate")

    def test_npu_inference_result_is_dict(self, results_json: dict[str, Any]) -> None:
        """npu_inference_result must be a dict on inference_success path."""
        assert isinstance(results_json["npu_inference_result"], dict)

    def test_inference_result_has_required_keys(
        self, results_json: dict[str, Any]
    ) -> None:
        """npu_inference_result must contain all required latency fields."""
        nir = results_json["npu_inference_result"]
        missing = REQUIRED_INFERENCE_KEYS - set(nir.keys())
        assert not missing, f"npu_inference_result missing keys: {missing}"

    def test_npu_latency_us_is_positive(self, results_json: dict[str, Any]) -> None:
        """npu_latency_us must be a positive float — not null, not fabricated zero."""
        lat = results_json["npu_inference_result"]["npu_latency_us"]
        assert isinstance(lat, (int, float)) and lat > 0

    def test_cpu_latency_us_is_positive(self, results_json: dict[str, Any]) -> None:
        """cpu_latency_us must be a positive float."""
        lat = results_json["npu_inference_result"]["cpu_latency_us"]
        assert isinstance(lat, (int, float)) and lat > 0

    def test_speedup_factor_consistent(self, results_json: dict[str, Any]) -> None:
        """speedup_factor must equal cpu_latency_us / npu_latency_us within 5%."""
        nir = results_json["npu_inference_result"]
        expected = nir["cpu_latency_us"] / nir["npu_latency_us"]
        assert abs(nir["speedup_factor"] - expected) < 0.05

    def test_provider_used_contains_vitisai(self, results_json: dict[str, Any]) -> None:
        """provider_used must reference VitisAI when NPU is working."""
        provider = results_json["npu_inference_result"]["provider_used"]
        assert isinstance(provider, str) and "VitisAI" in provider
