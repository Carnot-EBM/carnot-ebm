"""Tests for gpu_preflight module and experiment_390 script — 100% coverage.

Coverage targets
----------------
- GPUPreflightResult: dataclass construction + field defaults
- run_gpu_preflight(): all six layers, short-circuit when GPU not live
- _compute_honest_verdict(): all four verdict branches
- build_preflight_artifact(): schema and field mapping
- experiment_390 main(): gpu_confirmed_live path (exit 0) + all blocked paths (exit 1)

Spec: REQ-INFRA-017, REQ-INFRA-018,
      SCENARIO-INFRA-019, SCENARIO-INFRA-020, SCENARIO-INFRA-021
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.gpu_preflight import (
    GPUPreflightResult,
    _compute_honest_verdict,
    build_preflight_artifact,
    run_gpu_preflight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_live_diagnostic() -> MagicMock:
    """Return a mock LiveGPUDiagnostic with is_live_capable=True."""
    diag = MagicMock()
    diag.is_live_capable = True
    return diag


def _make_dead_diagnostic(reason: str = "cuda_visible: no GPU") -> MagicMock:
    """Return a mock LiveGPUDiagnostic with is_live_capable=False."""
    diag = MagicMock()
    diag.is_live_capable = False
    diag.failure_reason = reason
    return diag


def _make_smoke_result(is_live: bool, model_id: str = "Qwen/Qwen3.5-0.8B") -> MagicMock:
    """Return a mock SmokeTestResult."""
    r = MagicMock()
    r.is_live = is_live
    r.model_id = model_id
    return r


# ---------------------------------------------------------------------------
# GPUPreflightResult dataclass
# ---------------------------------------------------------------------------


class TestGPUPreflightResult:
    """Dataclass construction and defaults."""

    def test_required_fields(self) -> None:
        """GPUPreflightResult can be constructed with all required fields."""
        r = GPUPreflightResult(
            env_var_set=True,
            subprocess_inherits_env=True,
            session_startup_exists=True,
            conductor_gpu_env_exists=True,
            is_live_capable=True,
            smoke_test_passed=True,
        )
        assert r.env_var_set is True
        assert r.subprocess_inherits_env is True
        assert r.session_startup_exists is True
        assert r.conductor_gpu_env_exists is True
        assert r.is_live_capable is True
        assert r.smoke_test_passed is True

    def test_defaults(self) -> None:
        """model_ids_loadable defaults to [], retro_019_resolved/honest_verdict default to falsy."""
        r = GPUPreflightResult(
            env_var_set=False,
            subprocess_inherits_env=False,
            session_startup_exists=False,
            conductor_gpu_env_exists=False,
            is_live_capable=False,
            smoke_test_passed=False,
        )
        assert r.model_ids_loadable == []
        assert r.retro_019_resolved is False
        assert r.honest_verdict == ""

    def test_model_ids_loadable_not_shared(self) -> None:
        """Each instance gets its own model_ids_loadable list (no mutable default sharing)."""
        r1 = GPUPreflightResult(False, False, False, False, False, False)
        r2 = GPUPreflightResult(False, False, False, False, False, False)
        r1.model_ids_loadable.append("x")
        assert r2.model_ids_loadable == []


# ---------------------------------------------------------------------------
# _compute_honest_verdict
# ---------------------------------------------------------------------------


class TestComputeHonestVerdict:
    """All four verdict branches."""

    def test_scripts_missing_startup(self) -> None:
        """Returns 'scripts_missing' when session_startup is absent."""
        verdict = _compute_honest_verdict(
            session_startup_exists=False,
            conductor_gpu_env_exists=True,
            subprocess_inherits_env=True,
            is_live_capable=True,
            retro_019_resolved=True,
        )
        assert verdict == "scripts_missing"

    def test_scripts_missing_conductor_env(self) -> None:
        """Returns 'scripts_missing' when conductor_gpu_env is absent."""
        verdict = _compute_honest_verdict(
            session_startup_exists=True,
            conductor_gpu_env_exists=False,
            subprocess_inherits_env=True,
            is_live_capable=True,
            retro_019_resolved=True,
        )
        assert verdict == "scripts_missing"

    def test_scripts_missing_both_absent(self) -> None:
        """Returns 'scripts_missing' when both startup scripts are absent."""
        verdict = _compute_honest_verdict(
            session_startup_exists=False,
            conductor_gpu_env_exists=False,
            subprocess_inherits_env=False,
            is_live_capable=False,
            retro_019_resolved=False,
        )
        assert verdict == "scripts_missing"

    def test_env_not_propagating(self) -> None:
        """Returns 'env_not_propagating' when subprocess does not inherit env."""
        verdict = _compute_honest_verdict(
            session_startup_exists=True,
            conductor_gpu_env_exists=True,
            subprocess_inherits_env=False,
            is_live_capable=True,
            retro_019_resolved=False,
        )
        assert verdict == "env_not_propagating"

    def test_gpu_hardware_not_live(self) -> None:
        """Returns 'gpu_hardware_not_live' when GPU diagnostic fails."""
        verdict = _compute_honest_verdict(
            session_startup_exists=True,
            conductor_gpu_env_exists=True,
            subprocess_inherits_env=True,
            is_live_capable=False,
            retro_019_resolved=False,
        )
        assert verdict == "gpu_hardware_not_live"

    def test_gpu_confirmed_live(self) -> None:
        """Returns 'gpu_confirmed_live' when all checks pass."""
        verdict = _compute_honest_verdict(
            session_startup_exists=True,
            conductor_gpu_env_exists=True,
            subprocess_inherits_env=True,
            is_live_capable=True,
            retro_019_resolved=True,
        )
        assert verdict == "gpu_confirmed_live"

    def test_live_capable_but_smoke_failed(self) -> None:
        """Returns 'gpu_hardware_not_live' when GPU capable but smoke test failed."""
        # is_live_capable=True but retro_019_resolved=False (smoke_test_passed=False)
        verdict = _compute_honest_verdict(
            session_startup_exists=True,
            conductor_gpu_env_exists=True,
            subprocess_inherits_env=True,
            is_live_capable=True,
            retro_019_resolved=False,
        )
        assert verdict == "gpu_hardware_not_live"


# ---------------------------------------------------------------------------
# build_preflight_artifact
# ---------------------------------------------------------------------------


class TestBuildPreflightArtifact:
    """Schema and field mapping."""

    def _live_result(self) -> GPUPreflightResult:
        return GPUPreflightResult(
            env_var_set=True,
            subprocess_inherits_env=True,
            session_startup_exists=True,
            conductor_gpu_env_exists=True,
            is_live_capable=True,
            smoke_test_passed=True,
            model_ids_loadable=["google/gemma-4-E4B-it"],
            retro_019_resolved=True,
            honest_verdict="gpu_confirmed_live",
        )

    def test_schema(self) -> None:
        """Artifact has schema=carnot.gpu_preflight.v1."""
        artifact = build_preflight_artifact(self._live_result())
        assert artifact["schema"] == "carnot.gpu_preflight.v1"

    def test_all_fields_present(self) -> None:
        """Artifact contains all GPUPreflightResult fields."""
        artifact = build_preflight_artifact(self._live_result())
        expected_keys = {
            "schema",
            "honest_verdict",
            "env_var_set",
            "subprocess_inherits_env",
            "session_startup_exists",
            "conductor_gpu_env_exists",
            "is_live_capable",
            "smoke_test_passed",
            "model_ids_loadable",
            "retro_019_resolved",
        }
        assert set(artifact.keys()) == expected_keys

    def test_live_verdict(self) -> None:
        """Artifact has correct honest_verdict for live GPU."""
        artifact = build_preflight_artifact(self._live_result())
        assert artifact["honest_verdict"] == "gpu_confirmed_live"
        assert artifact["retro_019_resolved"] is True
        assert artifact["model_ids_loadable"] == ["google/gemma-4-E4B-it"]

    def test_blocked_verdict(self) -> None:
        """Artifact has correct fields when GPU is offline."""
        result = GPUPreflightResult(
            env_var_set=True,
            subprocess_inherits_env=True,
            session_startup_exists=True,
            conductor_gpu_env_exists=True,
            is_live_capable=False,
            smoke_test_passed=False,
            model_ids_loadable=[],
            retro_019_resolved=False,
            honest_verdict="gpu_hardware_not_live",
        )
        artifact = build_preflight_artifact(result)
        assert artifact["honest_verdict"] == "gpu_hardware_not_live"
        assert artifact["retro_019_resolved"] is False
        assert artifact["model_ids_loadable"] == []

    def test_json_serializable(self) -> None:
        """Artifact is JSON-serializable (no non-JSON types)."""
        artifact = build_preflight_artifact(self._live_result())
        serialized = json.dumps(artifact)
        restored = json.loads(serialized)
        assert restored["schema"] == "carnot.gpu_preflight.v1"


# ---------------------------------------------------------------------------
# run_gpu_preflight
# ---------------------------------------------------------------------------


_MODULE = "carnot.pipeline.gpu_preflight"


class TestRunGpuPreflight:
    """Six-layer preflight — all paths."""

    def _patch_all_live(
        self,
        tmp_path: Path,
        *,
        env_var: bool = True,
        subprocess_env: bool = True,
        session_startup: bool = True,
        conductor_env: bool = True,
        is_live: bool = True,
        smoke_live: bool = True,
    ):
        """Patch all external calls and create script stubs in tmp_path."""
        if session_startup:
            (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
            (tmp_path / "scripts" / "session_startup.sh").touch()
        if conductor_env:
            (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
            (tmp_path / "scripts" / "conductor_gpu_env.sh").touch()

        diag = _make_live_diagnostic() if is_live else _make_dead_diagnostic()
        smoke_result = _make_smoke_result(is_live=smoke_live)

        patches = [
            patch(f"{_MODULE}.LiveGPUGate.check_env_var", return_value=env_var),
            patch(
                f"{_MODULE}.LiveGPUGate.verify_subprocess_env_propagation",
                return_value=subprocess_env,
            ),
            patch(f"{_MODULE}.diagnose_live_gpu", return_value=diag),
        ]
        return patches, smoke_result

    def test_all_live_two_models_loadable(self, tmp_path: Path) -> None:
        """Returns retro_019_resolved=True when all checks pass."""
        model_ids = ["google/gemma-4-E4B-it", "Qwen/Qwen3.5-0.8B"]
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "session_startup.sh").touch()
        (tmp_path / "scripts" / "conductor_gpu_env.sh").touch()
        diag = _make_live_diagnostic()

        smoke_results = {
            "google/gemma-4-E4B-it": _make_smoke_result(True, "google/gemma-4-E4B-it"),
            "Qwen/Qwen3.5-0.8B": _make_smoke_result(True, "Qwen/Qwen3.5-0.8B"),
        }

        def _smoke_side_effect(mid: str) -> MagicMock:
            return smoke_results[mid]

        with (
            patch(f"{_MODULE}.LiveGPUGate.check_env_var", return_value=True),
            patch(
                f"{_MODULE}.LiveGPUGate.verify_subprocess_env_propagation",
                return_value=True,
            ),
            patch(f"{_MODULE}.diagnose_live_gpu", return_value=diag),
            patch(
                "carnot.pipeline.gpu_preflight.run_smoke_test",
                side_effect=_smoke_side_effect,
            ),
        ):
            result = run_gpu_preflight(tmp_path, model_ids=model_ids)

        assert result.retro_019_resolved is True
        assert result.honest_verdict == "gpu_confirmed_live"
        assert result.smoke_test_passed is True
        assert sorted(result.model_ids_loadable) == sorted(model_ids)

    def test_gpu_offline_skips_smoke_test(self, tmp_path: Path) -> None:
        """Smoke test is NOT called when GPU is not live-capable."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "session_startup.sh").touch()
        (tmp_path / "scripts" / "conductor_gpu_env.sh").touch()
        diag = _make_dead_diagnostic()

        with (
            patch(f"{_MODULE}.LiveGPUGate.check_env_var", return_value=True),
            patch(
                f"{_MODULE}.LiveGPUGate.verify_subprocess_env_propagation",
                return_value=True,
            ),
            patch(f"{_MODULE}.diagnose_live_gpu", return_value=diag),
            patch(
                "carnot.pipeline.gpu_preflight.run_smoke_test"
            ) as mock_smoke,
        ):
            result = run_gpu_preflight(tmp_path)

        mock_smoke.assert_not_called()
        assert result.is_live_capable is False
        assert result.smoke_test_passed is False
        assert result.retro_019_resolved is False
        assert result.honest_verdict == "gpu_hardware_not_live"

    def test_scripts_missing_session_startup(self, tmp_path: Path) -> None:
        """Returns scripts_missing when session_startup.sh is absent."""
        (tmp_path / "scripts").mkdir()
        # Only conductor_gpu_env.sh present — session_startup.sh absent
        (tmp_path / "scripts" / "conductor_gpu_env.sh").touch()
        diag = _make_live_diagnostic()

        with (
            patch(f"{_MODULE}.LiveGPUGate.check_env_var", return_value=True),
            patch(
                f"{_MODULE}.LiveGPUGate.verify_subprocess_env_propagation",
                return_value=True,
            ),
            patch(f"{_MODULE}.diagnose_live_gpu", return_value=diag),
        ):
            result = run_gpu_preflight(tmp_path)

        assert result.session_startup_exists is False
        assert result.honest_verdict == "scripts_missing"

    def test_scripts_missing_conductor_gpu_env(self, tmp_path: Path) -> None:
        """Returns scripts_missing when conductor_gpu_env.sh is absent."""
        (tmp_path / "scripts").mkdir()
        # Only session_startup.sh present — conductor_gpu_env.sh absent
        (tmp_path / "scripts" / "session_startup.sh").touch()
        diag = _make_live_diagnostic()

        with (
            patch(f"{_MODULE}.LiveGPUGate.check_env_var", return_value=True),
            patch(
                f"{_MODULE}.LiveGPUGate.verify_subprocess_env_propagation",
                return_value=True,
            ),
            patch(f"{_MODULE}.diagnose_live_gpu", return_value=diag),
        ):
            result = run_gpu_preflight(tmp_path)

        assert result.conductor_gpu_env_exists is False
        assert result.honest_verdict == "scripts_missing"

    def test_env_not_propagating(self, tmp_path: Path) -> None:
        """Returns env_not_propagating when subprocess cannot inherit env."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "session_startup.sh").touch()
        (tmp_path / "scripts" / "conductor_gpu_env.sh").touch()
        diag = _make_live_diagnostic()

        with (
            patch(f"{_MODULE}.LiveGPUGate.check_env_var", return_value=True),
            patch(
                f"{_MODULE}.LiveGPUGate.verify_subprocess_env_propagation",
                return_value=False,
            ),
            patch(f"{_MODULE}.diagnose_live_gpu", return_value=diag),
        ):
            result = run_gpu_preflight(tmp_path)

        assert result.subprocess_inherits_env is False
        assert result.honest_verdict == "env_not_propagating"

    def test_env_var_not_set(self, tmp_path: Path) -> None:
        """env_var_set=False still allows preflight to run but retro_019_resolved=False."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "session_startup.sh").touch()
        (tmp_path / "scripts" / "conductor_gpu_env.sh").touch()
        diag = _make_live_diagnostic()
        smoke_result = _make_smoke_result(True)

        with (
            patch(f"{_MODULE}.LiveGPUGate.check_env_var", return_value=False),
            patch(
                f"{_MODULE}.LiveGPUGate.verify_subprocess_env_propagation",
                return_value=True,
            ),
            patch(f"{_MODULE}.diagnose_live_gpu", return_value=diag),
            patch(
                "carnot.pipeline.gpu_preflight.run_smoke_test",
                return_value=smoke_result,
            ),
        ):
            result = run_gpu_preflight(tmp_path, model_ids=["Qwen/Qwen3.5-0.8B"])

        # env_var_set=False means retro_019_resolved=False even if GPU is live
        assert result.env_var_set is False
        assert result.retro_019_resolved is False

    def test_smoke_test_raises_is_handled(self, tmp_path: Path) -> None:
        """Smoke test that raises an exception is treated as failed (no crash)."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "session_startup.sh").touch()
        (tmp_path / "scripts" / "conductor_gpu_env.sh").touch()
        diag = _make_live_diagnostic()

        with (
            patch(f"{_MODULE}.LiveGPUGate.check_env_var", return_value=True),
            patch(
                f"{_MODULE}.LiveGPUGate.verify_subprocess_env_propagation",
                return_value=True,
            ),
            patch(f"{_MODULE}.diagnose_live_gpu", return_value=diag),
            patch(
                "carnot.pipeline.gpu_preflight.run_smoke_test",
                side_effect=RuntimeError("GPU OOM"),
            ),
        ):
            result = run_gpu_preflight(tmp_path, model_ids=["Qwen/Qwen3.5-0.8B"])

        assert result.smoke_test_passed is False
        assert result.model_ids_loadable == []
        assert result.retro_019_resolved is False

    def test_smoke_test_ci_skip_not_counted_as_live(self, tmp_path: Path) -> None:
        """Smoke test returning is_live=False (ci_skip) does not count as loadable."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "session_startup.sh").touch()
        (tmp_path / "scripts" / "conductor_gpu_env.sh").touch()
        diag = _make_live_diagnostic()
        smoke_result = _make_smoke_result(is_live=False)

        with (
            patch(f"{_MODULE}.LiveGPUGate.check_env_var", return_value=True),
            patch(
                f"{_MODULE}.LiveGPUGate.verify_subprocess_env_propagation",
                return_value=True,
            ),
            patch(f"{_MODULE}.diagnose_live_gpu", return_value=diag),
            patch(
                "carnot.pipeline.gpu_preflight.run_smoke_test",
                return_value=smoke_result,
            ),
        ):
            result = run_gpu_preflight(tmp_path, model_ids=["Qwen/Qwen3.5-0.8B"])

        assert result.smoke_test_passed is False
        assert result.model_ids_loadable == []

    def test_default_model_ids_used_when_none(self, tmp_path: Path) -> None:
        """Default model IDs are used when model_ids=None."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "session_startup.sh").touch()
        (tmp_path / "scripts" / "conductor_gpu_env.sh").touch()
        diag = _make_dead_diagnostic()

        with (
            patch(f"{_MODULE}.LiveGPUGate.check_env_var", return_value=False),
            patch(
                f"{_MODULE}.LiveGPUGate.verify_subprocess_env_propagation",
                return_value=False,
            ),
            patch(f"{_MODULE}.diagnose_live_gpu", return_value=diag) as mock_diag,
        ):
            result = run_gpu_preflight(tmp_path, model_ids=None)

        # diagnose_live_gpu should have been called with the default model list
        call_args = mock_diag.call_args[0][0]
        assert "google/gemma-4-E4B-it" in call_args
        assert "Qwen/Qwen3.5-0.8B" in call_args
        assert result.is_live_capable is False


# ---------------------------------------------------------------------------
# experiment_390 main() — integration tests
# ---------------------------------------------------------------------------


class TestExperiment390Main:
    """main() function paths: confirmed live (exit 0) and all blocked verdicts (exit 1)."""

    def _run_main(
        self,
        tmp_path: Path,
        preflight_result: GPUPreflightResult,
    ) -> int:
        """Run main() with patched run_gpu_preflight and return exit code."""
        import scripts.experiment_390_gpu_preflight as exp390

        with (
            patch.object(
                exp390,
                "run_gpu_preflight",
                return_value=preflight_result,
            ),
            patch.object(
                exp390,
                "_REPO_ROOT",
                tmp_path,
            ),
            patch.object(
                exp390,
                "_OUTPUT_PATH",
                "results/experiment_390_gpu_preflight.json",
            ),
        ):
            return exp390.main()

    def test_exit_0_when_confirmed_live(self, tmp_path: Path) -> None:
        """main() returns 0 when honest_verdict=gpu_confirmed_live."""
        result = GPUPreflightResult(
            env_var_set=True,
            subprocess_inherits_env=True,
            session_startup_exists=True,
            conductor_gpu_env_exists=True,
            is_live_capable=True,
            smoke_test_passed=True,
            model_ids_loadable=["Qwen/Qwen3.5-0.8B"],
            retro_019_resolved=True,
            honest_verdict="gpu_confirmed_live",
        )
        exit_code = self._run_main(tmp_path, result)
        assert exit_code == 0

    def test_exit_1_when_scripts_missing(self, tmp_path: Path) -> None:
        """main() returns 1 when honest_verdict=scripts_missing."""
        result = GPUPreflightResult(
            env_var_set=True,
            subprocess_inherits_env=True,
            session_startup_exists=False,
            conductor_gpu_env_exists=False,
            is_live_capable=False,
            smoke_test_passed=False,
            retro_019_resolved=False,
            honest_verdict="scripts_missing",
        )
        exit_code = self._run_main(tmp_path, result)
        assert exit_code == 1

    def test_exit_1_when_env_not_propagating(self, tmp_path: Path) -> None:
        """main() returns 1 when honest_verdict=env_not_propagating."""
        result = GPUPreflightResult(
            env_var_set=True,
            subprocess_inherits_env=False,
            session_startup_exists=True,
            conductor_gpu_env_exists=True,
            is_live_capable=True,
            smoke_test_passed=False,
            retro_019_resolved=False,
            honest_verdict="env_not_propagating",
        )
        exit_code = self._run_main(tmp_path, result)
        assert exit_code == 1

    def test_exit_1_when_gpu_hardware_not_live(self, tmp_path: Path) -> None:
        """main() returns 1 when honest_verdict=gpu_hardware_not_live (RETRO-019)."""
        result = GPUPreflightResult(
            env_var_set=True,
            subprocess_inherits_env=True,
            session_startup_exists=True,
            conductor_gpu_env_exists=True,
            is_live_capable=False,
            smoke_test_passed=False,
            retro_019_resolved=False,
            honest_verdict="gpu_hardware_not_live",
        )
        exit_code = self._run_main(tmp_path, result)
        assert exit_code == 1

    def test_exit_1_when_unknown_verdict(self, tmp_path: Path) -> None:
        """main() returns 1 on any unknown verdict (defensive branch)."""
        result = GPUPreflightResult(
            env_var_set=False,
            subprocess_inherits_env=False,
            session_startup_exists=True,
            conductor_gpu_env_exists=True,
            is_live_capable=False,
            smoke_test_passed=False,
            retro_019_resolved=False,
            honest_verdict="some_unexpected_verdict",
        )
        exit_code = self._run_main(tmp_path, result)
        assert exit_code == 1

    def test_artifact_written_to_disk(self, tmp_path: Path) -> None:
        """main() writes a JSON artifact to the output path."""
        result = GPUPreflightResult(
            env_var_set=True,
            subprocess_inherits_env=True,
            session_startup_exists=True,
            conductor_gpu_env_exists=True,
            is_live_capable=True,
            smoke_test_passed=True,
            model_ids_loadable=["Qwen/Qwen3.5-0.8B"],
            retro_019_resolved=True,
            honest_verdict="gpu_confirmed_live",
        )

        import scripts.experiment_390_gpu_preflight as exp390

        with (
            patch.object(exp390, "run_gpu_preflight", return_value=result),
            patch.object(exp390, "_REPO_ROOT", tmp_path),
            patch.object(
                exp390,
                "_OUTPUT_PATH",
                "results/experiment_390_gpu_preflight.json",
            ),
        ):
            exit_code = exp390.main()

        assert exit_code == 0
        artifact_path = tmp_path / "results" / "experiment_390_gpu_preflight.json"
        assert artifact_path.exists()
        artifact = json.loads(artifact_path.read_text())
        # ExperimentTemplate.build_result replaces schema with sorted key list;
        # check the preflight-specific fields directly.
        assert artifact["honest_verdict"] == "gpu_confirmed_live"
        assert "env_var_set" in artifact

    def test_artifact_written_even_when_blocked(self, tmp_path: Path) -> None:
        """main() writes artifact even when blocked (so conductor can read it)."""
        result = GPUPreflightResult(
            env_var_set=True,
            subprocess_inherits_env=True,
            session_startup_exists=True,
            conductor_gpu_env_exists=True,
            is_live_capable=False,
            smoke_test_passed=False,
            retro_019_resolved=False,
            honest_verdict="gpu_hardware_not_live",
        )

        import scripts.experiment_390_gpu_preflight as exp390

        with (
            patch.object(exp390, "run_gpu_preflight", return_value=result),
            patch.object(exp390, "_REPO_ROOT", tmp_path),
            patch.object(
                exp390,
                "_OUTPUT_PATH",
                "results/experiment_390_gpu_preflight.json",
            ),
        ):
            exit_code = exp390.main()

        assert exit_code == 1
        artifact_path = tmp_path / "results" / "experiment_390_gpu_preflight.json"
        assert artifact_path.exists()
        artifact = json.loads(artifact_path.read_text())
        assert artifact["honest_verdict"] == "gpu_hardware_not_live"
        assert artifact["status"] == "blocked"
