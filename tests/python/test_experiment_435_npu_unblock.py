"""Tests for Experiment 435: AMD XDNA NPU Unblock — 5th attempt + IRON toolchain.

Validates all pure-logic helpers, dataclass contract, and artifact schema without
depending on the host's package state.  All subprocess calls are mocked so the
test suite passes in CI (where ninja, openblas, and mlir_aie are absent).

Coverage targets:
  - check_ninja_available()           — subprocess which ninja/ninja-build
  - check_openblas_available()        — ldconfig, pkg-config, fallback .so paths
  - check_iron_toolchain_available()  — mlir_aie / aie importlib check
  - check_xdna_driver_loaded()        — grep /proc/modules
  - NPUPrereqResult dataclass         — all four fields + asdict
  - build_npu_result()                — all three verdict branches
  - _build_install_commands()         — all combinations of missing prereqs
  - _attempt_iron_gemm_dispatch()     — import fail, no runtime, runtime present
  - _attempt_vitisai_build()          — import fail, missing prereqs, model missing, success, failure

Spec:
  REQ-PRED-005 (NPU unblock status audit — IRON toolchain viability)
  SCENARIO-EXP303-G (IRON toolchain check as VitisAI alternative when prereqs missing)

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/pytest tests/python/test_experiment_435_npu_unblock.py -v
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import types
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import experiment_435_npu_unblock as _mod


# ---------------------------------------------------------------------------
# check_ninja_available
# ---------------------------------------------------------------------------


class TestCheckNinjaAvailable:
    """Spec: REQ-PRED-005, SCENARIO-EXP303-G"""

    def test_returns_true_when_ninja_found(self):
        """ninja on PATH → True."""
        def fake_run(cmd, **kw):
            if cmd == ["which", "ninja"]:
                r = mock.MagicMock()
                r.returncode = 0
                r.stdout = "/usr/bin/ninja\n"
                return r
            r = mock.MagicMock()
            r.returncode = 1
            r.stdout = ""
            return r

        with mock.patch("subprocess.run", side_effect=fake_run):
            assert _mod.check_ninja_available() is True

    def test_returns_true_when_ninja_build_found(self):
        """ninja not found but ninja-build is → True."""
        def fake_run(cmd, **kw):
            if cmd == ["which", "ninja-build"]:
                r = mock.MagicMock()
                r.returncode = 0
                r.stdout = "/usr/bin/ninja-build\n"
                return r
            r = mock.MagicMock()
            r.returncode = 1
            r.stdout = ""
            return r

        with mock.patch("subprocess.run", side_effect=fake_run):
            assert _mod.check_ninja_available() is True

    def test_returns_false_when_neither_found(self):
        """Neither ninja nor ninja-build on PATH → False."""
        def fake_run(cmd, **kw):
            r = mock.MagicMock()
            r.returncode = 1
            r.stdout = ""
            return r

        with mock.patch("subprocess.run", side_effect=fake_run):
            assert _mod.check_ninja_available() is False

    def test_returns_false_on_file_not_found(self):
        """FileNotFoundError (which not on PATH) → False."""
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            assert _mod.check_ninja_available() is False

    def test_returns_false_on_timeout(self):
        """TimeoutExpired → False."""
        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="which", timeout=10),
        ):
            assert _mod.check_ninja_available() is False

    def test_returns_false_when_stdout_empty(self):
        """which exits 0 but stdout empty → treated as not found."""
        def fake_run(cmd, **kw):
            r = mock.MagicMock()
            r.returncode = 0
            r.stdout = ""
            return r

        with mock.patch("subprocess.run", side_effect=fake_run):
            assert _mod.check_ninja_available() is False


# ---------------------------------------------------------------------------
# check_openblas_available
# ---------------------------------------------------------------------------


class TestCheckOpenblasAvailable:
    """Spec: REQ-PRED-005, SCENARIO-EXP303-G"""

    def test_returns_true_via_ldconfig(self):
        """ldconfig -p output contains 'openblas' → True."""
        def fake_run(cmd, **kw):
            if cmd == ["ldconfig", "-p"]:
                r = mock.MagicMock()
                r.returncode = 0
                r.stdout = "libopenblas.so.0 (libc6,x86-64) => /usr/lib/libopenblas.so.0\n"
                return r
            r = mock.MagicMock()
            r.returncode = 1
            r.stdout = ""
            return r

        with mock.patch("subprocess.run", side_effect=fake_run):
            assert _mod.check_openblas_available() is True

    def test_returns_true_via_pkg_config(self):
        """ldconfig fails, but pkg-config finds openblas → True."""
        call_count = {"n": 0}

        def fake_run(cmd, **kw):
            call_count["n"] += 1
            if cmd == ["ldconfig", "-p"]:
                r = mock.MagicMock()
                r.returncode = 0
                r.stdout = "no openblas here\n"
                return r
            if cmd == ["pkg-config", "--modversion", "openblas"]:
                r = mock.MagicMock()
                r.returncode = 0
                r.stdout = "0.3.21\n"
                return r
            r = mock.MagicMock()
            r.returncode = 1
            r.stdout = ""
            return r

        with mock.patch("subprocess.run", side_effect=fake_run):
            assert _mod.check_openblas_available() is True

    def test_returns_true_via_so_file(self, tmp_path):
        """All subprocess checks fail, but .so file exists → True."""
        so_path = tmp_path / "libopenblas.so"
        so_path.write_text("fake so")

        def fake_run(cmd, **kw):
            r = mock.MagicMock()
            r.returncode = 1
            r.stdout = ""
            return r

        # Patch Path.exists to return True only for our fake path
        original_exists = Path.exists

        def patched_exists(self):
            if str(self) == str(so_path):
                return True
            # Redirect the well-known paths to our tmp
            for candidate in (
                "/usr/lib/libopenblas.so",
                "/usr/lib/x86_64-linux-gnu/libopenblas.so",
                "/usr/local/lib/libopenblas.so",
            ):
                if str(self) == candidate:
                    return str(so_path) == candidate  # won't match
            return original_exists(self)

        # Patch the module's check directly via the candidate list
        with mock.patch("subprocess.run", side_effect=fake_run):
            with mock.patch.object(
                Path,
                "exists",
                lambda self: str(self) in (
                    "/usr/lib/libopenblas.so",
                    "/usr/lib/x86_64-linux-gnu/libopenblas.so",
                    "/usr/local/lib/libopenblas.so",
                ) and str(self) == "/usr/lib/libopenblas.so",
            ):
                assert _mod.check_openblas_available() is True

    def test_returns_false_when_nothing_found(self):
        """All checks fail → False."""
        def fake_run(cmd, **kw):
            r = mock.MagicMock()
            r.returncode = 1
            r.stdout = ""
            return r

        with mock.patch("subprocess.run", side_effect=fake_run):
            with mock.patch.object(Path, "exists", return_value=False):
                assert _mod.check_openblas_available() is False

    def test_returns_false_on_file_not_found(self):
        """FileNotFoundError on ldconfig → falls through, returns False."""
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            with mock.patch.object(Path, "exists", return_value=False):
                assert _mod.check_openblas_available() is False

    def test_returns_false_on_timeout(self):
        """TimeoutExpired → falls through, returns False."""
        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ldconfig", timeout=10),
        ):
            with mock.patch.object(Path, "exists", return_value=False):
                assert _mod.check_openblas_available() is False


# ---------------------------------------------------------------------------
# check_iron_toolchain_available
# ---------------------------------------------------------------------------


class TestCheckIronToolchainAvailable:
    """Spec: REQ-PRED-005, SCENARIO-EXP303-G"""

    def test_returns_true_when_mlir_aie_importable(self):
        """mlir_aie importable → True."""
        fake_mod = types.ModuleType("mlir_aie")
        with mock.patch.dict(sys.modules, {"mlir_aie": fake_mod}):
            # Remove aie from modules to avoid interference
            with mock.patch("importlib.import_module", side_effect=lambda name: {
                "mlir_aie": fake_mod,
            }[name]):
                result = _mod.check_iron_toolchain_available()
        assert result is True

    def test_returns_true_when_aie_importable(self):
        """mlir_aie fails but aie importable → True."""
        fake_aie = types.ModuleType("aie")

        def fake_import(name):
            if name == "mlir_aie":
                raise ImportError("no mlir_aie")
            if name == "aie":
                return fake_aie
            raise ImportError(name)

        with mock.patch("importlib.import_module", side_effect=fake_import):
            assert _mod.check_iron_toolchain_available() is True

    def test_returns_false_when_neither_importable(self):
        """Both mlir_aie and aie fail → False."""
        def fake_import(name):
            raise ImportError(name)

        with mock.patch("importlib.import_module", side_effect=fake_import):
            assert _mod.check_iron_toolchain_available() is False


# ---------------------------------------------------------------------------
# check_xdna_driver_loaded
# ---------------------------------------------------------------------------


class TestCheckXdnaDriverLoaded:
    """Spec: REQ-PRED-005, SCENARIO-EXP303-G"""

    def test_returns_true_when_amdxdna_in_proc_modules(self):
        """grep finds amdxdna → True."""
        r = mock.MagicMock()
        r.returncode = 0
        r.stdout = "amdxdna 12345 0\n"
        with mock.patch("subprocess.run", return_value=r):
            assert _mod.check_xdna_driver_loaded() is True

    def test_returns_false_when_not_found(self):
        """grep returns 1 (not found) → False."""
        r = mock.MagicMock()
        r.returncode = 1
        r.stdout = ""
        with mock.patch("subprocess.run", return_value=r):
            assert _mod.check_xdna_driver_loaded() is False

    def test_returns_false_on_file_not_found(self):
        """FileNotFoundError → False."""
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            assert _mod.check_xdna_driver_loaded() is False

    def test_returns_false_on_timeout(self):
        """TimeoutExpired → False."""
        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="grep", timeout=10),
        ):
            assert _mod.check_xdna_driver_loaded() is False


# ---------------------------------------------------------------------------
# NPUPrereqResult dataclass
# ---------------------------------------------------------------------------


class TestNPUPrereqResult:
    """Spec: REQ-PRED-005"""

    def test_all_true(self):
        p = _mod.NPUPrereqResult(ninja=True, openblas=True, iron_toolchain=True, xdna_driver=True)
        assert p.ninja is True
        assert p.openblas is True
        assert p.iron_toolchain is True
        assert p.xdna_driver is True

    def test_all_false(self):
        p = _mod.NPUPrereqResult(ninja=False, openblas=False, iron_toolchain=False, xdna_driver=False)
        assert p.ninja is False
        assert p.openblas is False

    def test_asdict(self):
        p = _mod.NPUPrereqResult(ninja=True, openblas=False, iron_toolchain=True, xdna_driver=False)
        d = asdict(p)
        assert d == {
            "ninja": True,
            "openblas": False,
            "iron_toolchain": True,
            "xdna_driver": False,
        }


# ---------------------------------------------------------------------------
# build_npu_result
# ---------------------------------------------------------------------------


class TestBuildNpuResult:
    """Spec: REQ-PRED-005, SCENARIO-EXP303-G"""

    def _prereqs(self, ninja=True, openblas=True, iron=True, xdna=True):
        return _mod.NPUPrereqResult(
            ninja=ninja, openblas=openblas, iron_toolchain=iron, xdna_driver=xdna
        )

    def test_verdict_npu_ready_iron_path(self):
        """iron_viable=True → 'npu_ready_iron_path'."""
        prereqs = self._prereqs()
        result = _mod.build_npu_result(prereqs, vitisai_path_blocked=True, iron_viable=True)
        assert result["honest_verdict"] == "npu_ready_iron_path"
        assert result["schema"] == "carnot.npu_unblock.v1"

    def test_verdict_npu_ready_vitisai_path(self):
        """iron_viable=False, vitisai_path_blocked=False → 'npu_ready_vitisai_path'."""
        prereqs = self._prereqs()
        result = _mod.build_npu_result(prereqs, vitisai_path_blocked=False, iron_viable=False)
        assert result["honest_verdict"] == "npu_ready_vitisai_path"

    def test_verdict_blocked_prereq(self):
        """iron_viable=False, vitisai_path_blocked=True → 'blocked_prereq'."""
        prereqs = self._prereqs(ninja=False, openblas=False)
        result = _mod.build_npu_result(prereqs, vitisai_path_blocked=True, iron_viable=False)
        assert result["honest_verdict"] == "blocked_prereq"

    def test_iron_viable_wins_over_vitisai_unblocked(self):
        """iron_viable=True takes precedence even when VitisAI path is also unblocked."""
        prereqs = self._prereqs()
        result = _mod.build_npu_result(prereqs, vitisai_path_blocked=False, iron_viable=True)
        assert result["honest_verdict"] == "npu_ready_iron_path"

    def test_result_contains_required_fields(self):
        prereqs = self._prereqs()
        result = _mod.build_npu_result(prereqs, vitisai_path_blocked=True, iron_viable=False)
        required = {"schema", "experiment", "run_date", "honest_verdict", "prereqs",
                    "vitisai_path_blocked", "iron_viable"}
        assert required.issubset(result.keys())

    def test_prereqs_embedded_as_dict(self):
        prereqs = self._prereqs(ninja=False)
        result = _mod.build_npu_result(prereqs, vitisai_path_blocked=True, iron_viable=False)
        assert result["prereqs"]["ninja"] is False
        assert isinstance(result["prereqs"], dict)


# ---------------------------------------------------------------------------
# _build_install_commands
# ---------------------------------------------------------------------------


class TestBuildInstallCommands:
    """Spec: REQ-PRED-005"""

    def _prereqs(self, ninja=True, openblas=True, iron=True, xdna=True):
        return _mod.NPUPrereqResult(
            ninja=ninja, openblas=openblas, iron_toolchain=iron, xdna_driver=xdna
        )

    def test_all_missing(self):
        p = self._prereqs(ninja=False, openblas=False, iron=False)
        cmds = _mod._build_install_commands(p)
        assert any("ninja" in c for c in cmds["arch_linux"])
        assert any("openblas" in c for c in cmds["arch_linux"])
        assert any("mlir-aie" in c for c in cmds["arch_linux"])
        assert any("ninja" in c for c in cmds["ubuntu"])
        assert any("openblas" in c for c in cmds["ubuntu"])
        assert any("mlir-aie" in c for c in cmds["ubuntu"])

    def test_only_ninja_missing(self):
        p = self._prereqs(ninja=False)
        cmds = _mod._build_install_commands(p)
        assert any("ninja" in c for c in cmds["arch_linux"])
        assert not any("openblas" in c for c in cmds["arch_linux"])

    def test_only_openblas_missing(self):
        p = self._prereqs(openblas=False)
        cmds = _mod._build_install_commands(p)
        assert any("openblas" in c for c in cmds["arch_linux"])
        assert not any("ninja" in c for c in cmds["arch_linux"])

    def test_nothing_missing_returns_empty_lists(self):
        p = self._prereqs()
        cmds = _mod._build_install_commands(p)
        assert cmds["arch_linux"] == []
        assert cmds["ubuntu"] == []

    def test_returns_note_field(self):
        p = self._prereqs()
        cmds = _mod._build_install_commands(p)
        assert "note" in cmds
        assert isinstance(cmds["note"], str)


# ---------------------------------------------------------------------------
# _attempt_iron_gemm_dispatch
# ---------------------------------------------------------------------------


class TestAttemptIronGemmDispatch:
    """Spec: REQ-PRED-005, SCENARIO-EXP303-G"""

    def test_returns_error_when_import_fails(self):
        """mlir_aie not importable → ok=False with error."""
        def fake_import(name, *a, **kw):
            raise ImportError("no mlir_aie")

        # Remove mlir_aie from sys.modules to ensure import is attempted
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("mlir_aie", None)
            with mock.patch.object(_mod.importlib, "import_module", side_effect=fake_import):
                # Patch the try block's direct import
                with mock.patch.dict(sys.modules, {"mlir_aie": None}):
                    result = _mod._attempt_iron_gemm_dispatch()
        # mlir_aie=None in sys.modules causes ImportError on 'import mlir_aie'
        assert result["ok"] is False
        assert result["latency_us"] is None

    def test_returns_error_when_module_has_no_runtime(self):
        """mlir_aie importable but no runtime attr → ok=False."""
        fake_mod = types.ModuleType("mlir_aie")
        # No 'runtime' or 'ipu_runner' attribute
        with mock.patch.dict(sys.modules, {"mlir_aie": fake_mod}):
            result = _mod._attempt_iron_gemm_dispatch()
        assert result["ok"] is False
        assert result["approach"] == "iron_import_only"

    def test_returns_ok_when_runtime_present(self):
        """mlir_aie with runtime attr → ok=True."""
        fake_mod = types.ModuleType("mlir_aie")
        fake_mod.runtime = mock.MagicMock()  # runtime attribute present
        with mock.patch.dict(sys.modules, {"mlir_aie": fake_mod}):
            result = _mod._attempt_iron_gemm_dispatch()
        assert result["ok"] is True
        assert result["latency_us"] is not None
        assert result["approach"] == "iron_runtime_api_present"

    def test_returns_ok_when_ipu_runner_present(self):
        """mlir_aie with ipu_runner attr (alternate API) → ok=True."""
        fake_mod = types.ModuleType("mlir_aie")
        fake_mod.ipu_runner = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"mlir_aie": fake_mod}):
            result = _mod._attempt_iron_gemm_dispatch()
        assert result["ok"] is True

    def test_returns_error_on_exception(self):
        """Unexpected exception in dispatch → ok=False with error string."""
        fake_mod = types.ModuleType("mlir_aie")
        fake_mod.runtime = mock.MagicMock(side_effect=RuntimeError("NPU fault"))
        with mock.patch.dict(sys.modules, {"mlir_aie": fake_mod}):
            # The runtime attribute access itself won't raise; we need to trigger
            # an error path by making hasattr raise
            with mock.patch("builtins.hasattr", side_effect=RuntimeError("hasattr broken")):
                result = _mod._attempt_iron_gemm_dispatch()
        assert result["ok"] is False
        assert "exception" in result["error"].lower()

    def test_speedup_is_none_without_real_dispatch(self):
        """speedup_vs_cpu must be None when no real NPU timing exists."""
        fake_mod = types.ModuleType("mlir_aie")
        fake_mod.runtime = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"mlir_aie": fake_mod}):
            result = _mod._attempt_iron_gemm_dispatch()
        assert result["speedup_vs_cpu"] is None


# ---------------------------------------------------------------------------
# _attempt_vitisai_build
# ---------------------------------------------------------------------------


class TestAttemptVitisaiBuild:
    """Spec: REQ-PRED-005"""

    def test_returns_not_attempted_when_import_fails(self):
        """experiment_292 not importable → attempted=False."""
        with mock.patch.dict(sys.modules, {"experiment_292_amd_xdna_npu": None}):
            result = _mod._attempt_vitisai_build()
        assert result["attempted"] is False
        assert result["succeeded"] is False
        assert "import_failed" in result["build_step"] or "import" in result["error_summary"].lower()

    def test_returns_not_attempted_when_prereqs_missing(self, tmp_path):
        """_check_source_build_prereqs returns non-empty list → attempted=False."""
        fake_292 = types.ModuleType("experiment_292_amd_xdna_npu")
        fake_292._check_source_build_prereqs = lambda: ["ninja: not found"]
        fake_292._select_onnx_model = lambda: None
        fake_292._attempt_source_build = lambda *a, **kw: {}

        with mock.patch.dict(sys.modules, {"experiment_292_amd_xdna_npu": fake_292}):
            result = _mod._attempt_vitisai_build()

        assert result["attempted"] is False
        assert "ninja" in result["error_summary"]

    def test_returns_model_missing_when_no_onnx(self, tmp_path):
        """Prereqs met but no ONNX model → attempted=True, succeeded=False."""
        fake_292 = types.ModuleType("experiment_292_amd_xdna_npu")
        fake_292._check_source_build_prereqs = lambda: []
        fake_292._select_onnx_model = lambda: None
        fake_292._attempt_source_build = lambda *a, **kw: {}

        with mock.patch.dict(sys.modules, {"experiment_292_amd_xdna_npu": fake_292}):
            result = _mod._attempt_vitisai_build()

        assert result["attempted"] is True
        assert result["succeeded"] is False
        assert result["build_step"] == "model_select"

    def test_returns_succeeded_true_on_ok_result(self, tmp_path):
        """_attempt_source_build returns ok=True → succeeded=True."""
        fake_onnx = tmp_path / "model.onnx"
        fake_onnx.write_text("fake")

        fake_292 = types.ModuleType("experiment_292_amd_xdna_npu")
        fake_292._check_source_build_prereqs = lambda: []
        fake_292._select_onnx_model = lambda: fake_onnx
        fake_292._attempt_source_build = lambda onnx, build_dir: {
            "ok": True,
            "latency_us": 2.5,
            "providers_used": ["VitisAIExecutionProvider"],
        }

        with mock.patch.dict(sys.modules, {"experiment_292_amd_xdna_npu": fake_292}):
            with mock.patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                mock_tmpdir.return_value.__enter__ = lambda s: str(tmp_path)
                mock_tmpdir.return_value.__exit__ = lambda s, *a: False
                result = _mod._attempt_vitisai_build()

        assert result["attempted"] is True
        assert result["succeeded"] is True
        assert result["latency_us"] == 2.5

    def test_returns_failed_on_build_error(self, tmp_path):
        """_attempt_source_build returns build_failed → succeeded=False."""
        fake_onnx = tmp_path / "model.onnx"
        fake_onnx.write_text("fake")

        fake_292 = types.ModuleType("experiment_292_amd_xdna_npu")
        fake_292._check_source_build_prereqs = lambda: []
        fake_292._select_onnx_model = lambda: fake_onnx
        fake_292._attempt_source_build = lambda onnx, build_dir: {
            "build_failed": True,
            "build_step": "ninja_build",
            "next_action": "Fix the build.",
        }

        with mock.patch.dict(sys.modules, {"experiment_292_amd_xdna_npu": fake_292}):
            with mock.patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                mock_tmpdir.return_value.__enter__ = lambda s: str(tmp_path)
                mock_tmpdir.return_value.__exit__ = lambda s, *a: False
                result = _mod._attempt_vitisai_build()

        assert result["attempted"] is True
        assert result["succeeded"] is False
        assert result["build_step"] == "ninja_build"


# ---------------------------------------------------------------------------
# Integration: full main() smoke test
# ---------------------------------------------------------------------------


class TestMainSmoke:
    """Spec: REQ-PRED-005, SCENARIO-EXP303-G — smoke test with all paths mocked."""

    def test_main_blocked_prereq(self, tmp_path):
        """main() runs without error when ninja + openblas both missing."""
        result_file = tmp_path / "experiment_435_npu_unblock.json"

        with mock.patch.object(_mod, "RESULT_PATH", result_file):
            with mock.patch.object(_mod, "check_ninja_available", return_value=False):
                with mock.patch.object(_mod, "check_openblas_available", return_value=False):
                    with mock.patch.object(_mod, "check_iron_toolchain_available", return_value=False):
                        with mock.patch.object(_mod, "check_xdna_driver_loaded", return_value=True):
                            with mock.patch("python.carnot.pipeline.env_autofix.apply_env_autofix") as mock_fix:
                                mock_fix.return_value = mock.MagicMock(
                                    gpu_detected=False, auto_fix_applied=False
                                )
                                _mod.main()

        assert result_file.exists()
        artifact = json.loads(result_file.read_text())
        assert artifact["honest_verdict"] == "blocked_prereq"
        assert artifact["schema"] == "carnot.npu_unblock.v1"
        assert artifact["prereqs"]["ninja"] is False
        assert artifact["prereqs"]["openblas"] is False
        assert artifact["vitisai_build_attempted"] is False
        assert artifact["iron_path_tested"] is False

    def test_main_iron_path_success(self, tmp_path):
        """main() reports npu_ready_iron_path when IRON dispatch succeeds."""
        result_file = tmp_path / "experiment_435_npu_unblock.json"

        iron_result = {
            "ok": True,
            "latency_us": 1.5,
            "speedup_vs_cpu": None,
            "error": None,
            "approach": "iron_runtime_api_present",
        }

        with mock.patch.object(_mod, "RESULT_PATH", result_file):
            with mock.patch.object(_mod, "check_ninja_available", return_value=False):
                with mock.patch.object(_mod, "check_openblas_available", return_value=False):
                    with mock.patch.object(_mod, "check_iron_toolchain_available", return_value=True):
                        with mock.patch.object(_mod, "check_xdna_driver_loaded", return_value=True):
                            with mock.patch.object(_mod, "_attempt_iron_gemm_dispatch", return_value=iron_result):
                                with mock.patch("python.carnot.pipeline.env_autofix.apply_env_autofix") as mock_fix:
                                    mock_fix.return_value = mock.MagicMock(
                                        gpu_detected=False, auto_fix_applied=False
                                    )
                                    _mod.main()

        artifact = json.loads(result_file.read_text())
        assert artifact["honest_verdict"] == "npu_ready_iron_path"
        assert artifact["iron_path_tested"] is True
        assert artifact["iron_path_succeeded"] is True

    def test_main_vitisai_success(self, tmp_path):
        """main() reports npu_ready_vitisai_path when VitisAI build succeeds."""
        result_file = tmp_path / "experiment_435_npu_unblock.json"

        vitisai_result = {
            "attempted": True,
            "succeeded": True,
            "error_summary": None,
            "build_step": "complete",
            "latency_us": 3.0,
        }

        with mock.patch.object(_mod, "RESULT_PATH", result_file):
            with mock.patch.object(_mod, "check_ninja_available", return_value=True):
                with mock.patch.object(_mod, "check_openblas_available", return_value=True):
                    with mock.patch.object(_mod, "check_iron_toolchain_available", return_value=False):
                        with mock.patch.object(_mod, "check_xdna_driver_loaded", return_value=True):
                            with mock.patch.object(_mod, "_attempt_vitisai_build", return_value=vitisai_result):
                                with mock.patch.object(_mod, "_attempt_iron_gemm_dispatch", return_value={"ok": False, "latency_us": None, "speedup_vs_cpu": None, "error": "no mlir_aie", "approach": "iron_import_failed"}):
                                    with mock.patch("python.carnot.pipeline.env_autofix.apply_env_autofix") as mock_fix:
                                        mock_fix.return_value = mock.MagicMock(
                                            gpu_detected=False, auto_fix_applied=False
                                        )
                                        _mod.main()

        artifact = json.loads(result_file.read_text())
        assert artifact["honest_verdict"] == "npu_ready_vitisai_path"
        assert artifact["vitisai_build_succeeded"] is True

    def test_main_artifact_has_all_required_fields(self, tmp_path):
        """Artifact contains all fields required by schema 'carnot.npu_unblock.v1'."""
        result_file = tmp_path / "experiment_435_npu_unblock.json"

        with mock.patch.object(_mod, "RESULT_PATH", result_file):
            with mock.patch.object(_mod, "check_ninja_available", return_value=False):
                with mock.patch.object(_mod, "check_openblas_available", return_value=False):
                    with mock.patch.object(_mod, "check_iron_toolchain_available", return_value=False):
                        with mock.patch.object(_mod, "check_xdna_driver_loaded", return_value=False):
                            with mock.patch("python.carnot.pipeline.env_autofix.apply_env_autofix") as mock_fix:
                                mock_fix.return_value = mock.MagicMock(
                                    gpu_detected=False, auto_fix_applied=False
                                )
                                _mod.main()

        artifact = json.loads(result_file.read_text())
        required_keys = {
            "schema",
            "experiment",
            "run_date",
            "honest_verdict",
            "prereqs",
            "vitisai_path_blocked",
            "iron_viable",
            "vitisai_build_attempted",
            "vitisai_build_succeeded",
            "iron_path_tested",
            "iron_path_succeeded",
            "iron_speedup_vs_cpu",
            "cpu_ort_baseline_us",
            "milestone_block_count",
            "install_commands",
        }
        assert required_keys.issubset(artifact.keys())

    def test_main_install_commands_present_when_blocked(self, tmp_path):
        """install_commands provides arch_linux and ubuntu entries when blocked."""
        result_file = tmp_path / "experiment_435_npu_unblock.json"

        with mock.patch.object(_mod, "RESULT_PATH", result_file):
            with mock.patch.object(_mod, "check_ninja_available", return_value=False):
                with mock.patch.object(_mod, "check_openblas_available", return_value=False):
                    with mock.patch.object(_mod, "check_iron_toolchain_available", return_value=False):
                        with mock.patch.object(_mod, "check_xdna_driver_loaded", return_value=False):
                            with mock.patch("python.carnot.pipeline.env_autofix.apply_env_autofix") as mock_fix:
                                mock_fix.return_value = mock.MagicMock(
                                    gpu_detected=False, auto_fix_applied=False
                                )
                                _mod.main()

        artifact = json.loads(result_file.read_text())
        install = artifact["install_commands"]
        assert "arch_linux" in install
        assert "ubuntu" in install
        assert any("ninja" in c for c in install["arch_linux"])
        assert any("ninja" in c for c in install["ubuntu"])

    def test_main_milestone_block_count_is_5(self, tmp_path):
        """milestone_block_count in artifact is 5 (honest reporting of history)."""
        result_file = tmp_path / "experiment_435_npu_unblock.json"

        with mock.patch.object(_mod, "RESULT_PATH", result_file):
            with mock.patch.object(_mod, "check_ninja_available", return_value=False):
                with mock.patch.object(_mod, "check_openblas_available", return_value=False):
                    with mock.patch.object(_mod, "check_iron_toolchain_available", return_value=False):
                        with mock.patch.object(_mod, "check_xdna_driver_loaded", return_value=False):
                            with mock.patch("python.carnot.pipeline.env_autofix.apply_env_autofix") as mock_fix:
                                mock_fix.return_value = mock.MagicMock(
                                    gpu_detected=False, auto_fix_applied=False
                                )
                                _mod.main()

        artifact = json.loads(result_file.read_text())
        assert artifact["milestone_block_count"] == 5
