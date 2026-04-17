"""Tests for experiment 438: GPU1 Zombie Fix (RETRO-025 root-cause fix).

Coverage targets
----------------
ZombieFixResult dataclass:
  - Instantiation with all required fields
  - honest_verdict values: fix_applied_and_verified / fix_applied_unverified / ci_mode

build_zombie_fix_strategy():
  - n_gpus=2, 2 model_ids  → explicit {'': 'cuda:0'} and {'': 'cuda:1'}
  - n_gpus=1, 1 model_id   → 'auto' (fallback)
  - n_gpus=0, 2 model_ids  → 'auto' for all (CI mode)
  - n_gpus=2, 3 model_ids  → first 2 explicit, third 'auto'
  - n_gpus=2, 1 model_id   → 'auto' (not enough models to trigger dual-GPU path)
  - empty model_ids         → empty dict

build_zombie_fix_artifact():
  - schema = 'carnot.gpu1_zombie_fix.v1'
  - honest_verdict forwarded from ZombieFixResult
  - retro_025_status mirrors honest_verdict
  - all fields present in output dict
  - fix_applied_and_verified path
  - fix_applied_unverified path (post_fix_gpu1_util_pct=0)
  - ci_mode path (post_fix_gpu1_util_pct=None)

run_experiment() (scripts/experiment_438_gpu1_zombie_fix.py):
  - CI mode (no GPU): honest_verdict='ci_mode', artifact has all REQUIRED_RESULT_FIELDS
  - force_live=0 always yields ci_mode
  - Artifact written to disk
  - env_autofix block present
  - zombie_fix_strategy embedded
  - baseline_gpu1_* fields present

main():
  - Calls run_experiment() inside ExperimentTimeoutWatchdog
  - Logs headline results

_detect_n_gpus():
  - pynvml success path
  - pynvml unavailable + nvidia-smi success path
  - nvidia-smi returns non-zero (failure)
  - both unavailable → 0

Spec: REQ-INFRA-029, REQ-INFRA-030,
      SCENARIO-INFRA-037, SCENARIO-INFRA-038 (Exp 438)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.gpu_zombie_fix import (  # noqa: E402
    ZombieFixResult,
    build_zombie_fix_artifact,
    build_zombie_fix_strategy,
)
from carnot.pipeline.dual_gpu_health import DualGPUHealthResult  # noqa: E402
from scripts.experiment_template import REQUIRED_RESULT_FIELDS  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CI_HEALTH = DualGPUHealthResult(
    gpu0_util_pct=0.0,
    gpu1_util_pct=0.0,
    gpu0_temp_c=0.0,
    gpu1_temp_c=0.0,
    gpu0_vram_mb=0.0,
    gpu1_vram_mb=0.0,
    gpu1_is_zombie=False,
    temperature_warning=False,
    recommended_batch_size_factor=1.0,
)

_MODEL_A = "Vendor/ModelA-0.5B"
_MODEL_B = "Vendor/ModelB-0.8B"
_MODEL_C = "Vendor/ModelC-1B"


def _make_autofix(gpu_detected=False, auto_fix_applied=False):
    from carnot.pipeline.env_autofix import EnvironmentAutoFix

    return EnvironmentAutoFix(
        gpu_detected=gpu_detected,
        carnot_force_live_was_set=False,
        auto_fix_applied=auto_fix_applied,
        final_env_value=None,
    )


# ---------------------------------------------------------------------------
# ZombieFixResult tests
# ---------------------------------------------------------------------------


class TestZombieFixResult:
    """ZombieFixResult dataclass instantiation and fields."""

    def test_ci_mode_result(self):
        """ci_mode: fix_applied=False, post_fix_gpu1_util_pct=None."""
        r = ZombieFixResult(
            gpu0_model_id=_MODEL_A,
            gpu1_model_id=_MODEL_B,
            gpu0_device_map="auto",
            gpu1_device_map="auto",
            fix_applied=False,
            post_fix_gpu1_util_pct=None,
            honest_verdict="ci_mode",
        )
        assert r.honest_verdict == "ci_mode"
        assert r.fix_applied is False
        assert r.post_fix_gpu1_util_pct is None

    def test_fix_applied_and_verified_result(self):
        """fix_applied_and_verified: explicit device_map + gpu1 utilization > 0."""
        r = ZombieFixResult(
            gpu0_model_id=_MODEL_A,
            gpu1_model_id=_MODEL_B,
            gpu0_device_map="{'': 'cuda:0'}",
            gpu1_device_map="{'': 'cuda:1'}",
            fix_applied=True,
            post_fix_gpu1_util_pct=45.0,
            honest_verdict="fix_applied_and_verified",
        )
        assert r.fix_applied is True
        assert r.post_fix_gpu1_util_pct == pytest.approx(45.0)
        assert r.honest_verdict == "fix_applied_and_verified"

    def test_fix_applied_unverified_result(self):
        """fix_applied_unverified: explicit device_map but gpu1 util still 0."""
        r = ZombieFixResult(
            gpu0_model_id=_MODEL_A,
            gpu1_model_id=_MODEL_B,
            gpu0_device_map="{'': 'cuda:0'}",
            gpu1_device_map="{'': 'cuda:1'}",
            fix_applied=True,
            post_fix_gpu1_util_pct=0.0,
            honest_verdict="fix_applied_unverified",
        )
        assert r.fix_applied is True
        assert r.honest_verdict == "fix_applied_unverified"


# ---------------------------------------------------------------------------
# build_zombie_fix_strategy tests
# ---------------------------------------------------------------------------


class TestBuildZombieFixStrategy:
    """build_zombie_fix_strategy() — SCENARIO-INFRA-037/038."""

    def test_dual_gpu_explicit_assignment(self):
        """SCENARIO-INFRA-037: n_gpus=2, 2 models → explicit cuda:0 / cuda:1."""
        s = build_zombie_fix_strategy(n_gpus=2, model_ids=[_MODEL_A, _MODEL_B])
        assert s[_MODEL_A] == {"": "cuda:0"}
        assert s[_MODEL_B] == {"": "cuda:1"}
        # Ensure 'auto' is NOT used — that is the whole point of the RETRO-025 fix
        assert s[_MODEL_A] != "auto"
        assert s[_MODEL_B] != "auto"

    def test_single_gpu_auto_fallback(self):
        """SCENARIO-INFRA-038: n_gpus=1, 1 model → 'auto' (safe fallback)."""
        s = build_zombie_fix_strategy(n_gpus=1, model_ids=[_MODEL_A])
        assert s[_MODEL_A] == "auto"

    def test_zero_gpus_ci_mode(self):
        """n_gpus=0 (CI): all models get 'auto'."""
        s = build_zombie_fix_strategy(n_gpus=0, model_ids=[_MODEL_A, _MODEL_B])
        assert s[_MODEL_A] == "auto"
        assert s[_MODEL_B] == "auto"

    def test_three_models_dual_gpu_third_gets_auto(self):
        """n_gpus=2, 3 models: first 2 explicit, third falls back to 'auto'."""
        s = build_zombie_fix_strategy(n_gpus=2, model_ids=[_MODEL_A, _MODEL_B, _MODEL_C])
        assert s[_MODEL_A] == {"": "cuda:0"}
        assert s[_MODEL_B] == {"": "cuda:1"}
        assert s[_MODEL_C] == "auto"

    def test_dual_gpu_single_model_gets_auto(self):
        """n_gpus=2 but only 1 model_id: not enough to trigger dual-GPU path → 'auto'."""
        s = build_zombie_fix_strategy(n_gpus=2, model_ids=[_MODEL_A])
        assert s[_MODEL_A] == "auto"

    def test_empty_model_ids(self):
        """Empty model_ids list → empty strategy dict."""
        s = build_zombie_fix_strategy(n_gpus=2, model_ids=[])
        assert s == {}


# ---------------------------------------------------------------------------
# build_zombie_fix_artifact tests
# ---------------------------------------------------------------------------


class TestBuildZombieFixArtifact:
    """build_zombie_fix_artifact() — schema + field presence + all verdict paths."""

    def _make_result(self, verdict, fix_applied=True, gpu1_util=None):
        return ZombieFixResult(
            gpu0_model_id=_MODEL_A,
            gpu1_model_id=_MODEL_B,
            gpu0_device_map="{'': 'cuda:0'}" if fix_applied else "auto",
            gpu1_device_map="{'': 'cuda:1'}" if fix_applied else "auto",
            fix_applied=fix_applied,
            post_fix_gpu1_util_pct=gpu1_util,
            honest_verdict=verdict,
        )

    def test_schema(self):
        """Artifact schema is 'carnot.gpu1_zombie_fix.v1'."""
        a = build_zombie_fix_artifact(self._make_result("ci_mode", fix_applied=False))
        assert a["schema"] == "carnot.gpu1_zombie_fix.v1"

    def test_honest_verdict_forwarded(self):
        """honest_verdict from ZombieFixResult is present in artifact."""
        for verdict in ("ci_mode", "fix_applied_and_verified", "fix_applied_unverified"):
            a = build_zombie_fix_artifact(self._make_result(verdict))
            assert a["honest_verdict"] == verdict

    def test_retro_025_status_mirrors_honest_verdict(self):
        """retro_025_status equals honest_verdict for traceability."""
        for verdict in ("ci_mode", "fix_applied_and_verified", "fix_applied_unverified"):
            a = build_zombie_fix_artifact(self._make_result(verdict))
            assert a["retro_025_status"] == a["honest_verdict"]

    def test_all_required_fields_present(self):
        """All expected top-level keys are present in the artifact."""
        a = build_zombie_fix_artifact(self._make_result("ci_mode", fix_applied=False))
        for key in (
            "schema",
            "honest_verdict",
            "retro_025_status",
            "gpu0_model_id",
            "gpu1_model_id",
            "gpu0_device_map",
            "gpu1_device_map",
            "fix_applied",
            "post_fix_gpu1_util_pct",
        ):
            assert key in a, f"Missing key: {key}"

    def test_fix_applied_and_verified_path(self):
        """fix_applied_and_verified: fix_applied=True, post_fix_gpu1_util_pct > 0."""
        r = self._make_result("fix_applied_and_verified", fix_applied=True, gpu1_util=50.0)
        a = build_zombie_fix_artifact(r)
        assert a["fix_applied"] is True
        assert a["post_fix_gpu1_util_pct"] == pytest.approx(50.0)

    def test_fix_applied_unverified_path(self):
        """fix_applied_unverified: fix_applied=True, post_fix_gpu1_util_pct=0."""
        r = self._make_result("fix_applied_unverified", fix_applied=True, gpu1_util=0.0)
        a = build_zombie_fix_artifact(r)
        assert a["fix_applied"] is True
        assert a["post_fix_gpu1_util_pct"] == pytest.approx(0.0)

    def test_ci_mode_path(self):
        """ci_mode: fix_applied=False, post_fix_gpu1_util_pct=None."""
        r = self._make_result("ci_mode", fix_applied=False, gpu1_util=None)
        a = build_zombie_fix_artifact(r)
        assert a["fix_applied"] is False
        assert a["post_fix_gpu1_util_pct"] is None


# ---------------------------------------------------------------------------
# run_experiment() tests
# ---------------------------------------------------------------------------


class TestRunExperiment:
    """run_experiment() — all code paths."""

    def _run(self, force_live="0", n_gpus=0):
        import scripts.experiment_438_gpu1_zombie_fix as exp438  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            with (
                patch.object(exp438, "_REPO_ROOT", tmp_root),
                patch.object(exp438, "_autofix_result", _make_autofix()),
                patch(
                    "scripts.experiment_438_gpu1_zombie_fix.check_dual_gpu_health",
                    return_value=_CI_HEALTH,
                ),
                patch(
                    "scripts.experiment_438_gpu1_zombie_fix._detect_n_gpus",
                    return_value=n_gpus,
                ),
                patch.dict("os.environ", {"CARNOT_FORCE_LIVE": force_live}),
            ):
                artifact = exp438.run_experiment()

        return artifact

    def test_ci_mode_honest_verdict(self):
        """CI mode (no GPU): honest_verdict='ci_mode'."""
        artifact = self._run(force_live="0", n_gpus=0)
        assert artifact["honest_verdict"] == "ci_mode"

    def test_ci_mode_required_fields(self):
        """CI mode: all REQUIRED_RESULT_FIELDS present in artifact."""
        artifact = self._run(force_live="0", n_gpus=0)
        for field in REQUIRED_RESULT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_ci_mode_schema_is_list(self):
        """schema field is a sorted list of key names."""
        artifact = self._run(force_live="0", n_gpus=0)
        assert isinstance(artifact["schema"], list)

    def test_env_autofix_block_present(self):
        """env_autofix dict always embedded in artifact."""
        artifact = self._run()
        assert "env_autofix" in artifact
        assert "gpu_detected" in artifact["env_autofix"]
        assert "auto_fix_applied" in artifact["env_autofix"]

    def test_zombie_fix_strategy_embedded(self):
        """zombie_fix_strategy dict embedded in artifact."""
        artifact = self._run()
        assert "zombie_fix_strategy" in artifact
        assert isinstance(artifact["zombie_fix_strategy"], dict)

    def test_baseline_gpu1_fields_present(self):
        """Baseline GPU1 metrics present in artifact."""
        artifact = self._run()
        assert "baseline_gpu1_vram_mb" in artifact
        assert "baseline_gpu1_util_pct" in artifact
        assert "baseline_gpu1_is_zombie" in artifact

    def test_n_gpus_detected_in_artifact(self):
        """n_gpus_detected field present and correct."""
        artifact = self._run(n_gpus=0)
        assert artifact["n_gpus_detected"] == 0

    def test_force_live_false_yields_ci_mode(self):
        """CARNOT_FORCE_LIVE=0 always yields ci_mode regardless of n_gpus."""
        artifact = self._run(force_live="0", n_gpus=2)
        assert artifact["honest_verdict"] == "ci_mode"

    def test_artifact_written_to_disk(self):
        """Output JSON file is written to deliverable path."""
        import scripts.experiment_438_gpu1_zombie_fix as exp438  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            with (
                patch.object(exp438, "_REPO_ROOT", tmp_root),
                patch.object(exp438, "_autofix_result", _make_autofix()),
                patch(
                    "scripts.experiment_438_gpu1_zombie_fix.check_dual_gpu_health",
                    return_value=_CI_HEALTH,
                ),
                patch(
                    "scripts.experiment_438_gpu1_zombie_fix._detect_n_gpus",
                    return_value=0,
                ),
                patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "0"}),
            ):
                exp438.run_experiment()

            output_path = tmp_root / "results" / "experiment_438_gpu1_zombie_fix.json"
            assert output_path.exists(), "Output JSON was not written"
            data = json.loads(output_path.read_text())
            assert data["experiment"] == 438

    def test_fix_applied_unverified_when_live_gpu_util_zero(self):
        """live GPU, n_gpus=2: fix_applied=True but gpu1 still 0 → fix_applied_unverified."""
        import scripts.experiment_438_gpu1_zombie_fix as exp438  # noqa: PLC0415

        post_load_health = DualGPUHealthResult(
            gpu0_util_pct=50.0,
            gpu1_util_pct=0.0,  # still 0 after load
            gpu0_temp_c=70.0,
            gpu1_temp_c=55.0,
            gpu0_vram_mb=10000.0,
            gpu1_vram_mb=800.0,
            gpu1_is_zombie=True,
            temperature_warning=False,
            recommended_batch_size_factor=1.0,
        )

        mock_model = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            with (
                patch.object(exp438, "_REPO_ROOT", tmp_root),
                patch.object(exp438, "_autofix_result", _make_autofix(gpu_detected=True)),
                patch(
                    "scripts.experiment_438_gpu1_zombie_fix.check_dual_gpu_health",
                    side_effect=[_CI_HEALTH, post_load_health],
                ),
                patch(
                    "scripts.experiment_438_gpu1_zombie_fix._detect_n_gpus",
                    return_value=2,
                ),
                patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "1"}),
                patch(
                    "transformers.AutoModelForCausalLM.from_pretrained",
                    return_value=mock_model,
                ),
            ):
                artifact = exp438.run_experiment()

        assert artifact["honest_verdict"] == "fix_applied_unverified"
        assert artifact["fix_applied"] is True

    def test_fix_applied_and_verified_when_live_gpu_util_positive(self):
        """live GPU, n_gpus=2, gpu1_util > 0 after load → fix_applied_and_verified."""
        import scripts.experiment_438_gpu1_zombie_fix as exp438  # noqa: PLC0415

        post_load_health = DualGPUHealthResult(
            gpu0_util_pct=50.0,
            gpu1_util_pct=42.0,  # GPU1 now computing — fix confirmed
            gpu0_temp_c=70.0,
            gpu1_temp_c=60.0,
            gpu0_vram_mb=10000.0,
            gpu1_vram_mb=900.0,
            gpu1_is_zombie=False,
            temperature_warning=False,
            recommended_batch_size_factor=1.0,
        )

        mock_model = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            with (
                patch.object(exp438, "_REPO_ROOT", tmp_root),
                patch.object(exp438, "_autofix_result", _make_autofix(gpu_detected=True)),
                patch(
                    "scripts.experiment_438_gpu1_zombie_fix.check_dual_gpu_health",
                    side_effect=[_CI_HEALTH, post_load_health],
                ),
                patch(
                    "scripts.experiment_438_gpu1_zombie_fix._detect_n_gpus",
                    return_value=2,
                ),
                patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "1"}),
                patch(
                    "transformers.AutoModelForCausalLM.from_pretrained",
                    return_value=mock_model,
                ),
            ):
                artifact = exp438.run_experiment()

        assert artifact["honest_verdict"] == "fix_applied_and_verified"

    def test_live_load_failure_falls_back_to_ci_mode(self):
        """When live model load throws, honest_verdict falls back to ci_mode."""
        import scripts.experiment_438_gpu1_zombie_fix as exp438  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            with (
                patch.object(exp438, "_REPO_ROOT", tmp_root),
                patch.object(exp438, "_autofix_result", _make_autofix(gpu_detected=True)),
                patch(
                    "scripts.experiment_438_gpu1_zombie_fix.check_dual_gpu_health",
                    return_value=_CI_HEALTH,
                ),
                patch(
                    "scripts.experiment_438_gpu1_zombie_fix._detect_n_gpus",
                    return_value=2,
                ),
                patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "1"}),
                patch(
                    "transformers.AutoModelForCausalLM.from_pretrained",
                    side_effect=RuntimeError("CUDA OOM"),
                ),
            ):
                artifact = exp438.run_experiment()

        assert artifact["honest_verdict"] == "ci_mode"


# ---------------------------------------------------------------------------
# _detect_n_gpus tests
# ---------------------------------------------------------------------------


class TestDetectNGpus:
    """_detect_n_gpus() — pynvml + nvidia-smi + fallback paths."""

    def test_pynvml_success(self):
        """pynvml path: returns GPU count from nvmlDeviceGetCount."""
        import scripts.experiment_438_gpu1_zombie_fix as exp438  # noqa: PLC0415

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 2

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            count = exp438._detect_n_gpus()

        assert count == 2
        mock_pynvml.nvmlInit.assert_called_once()
        mock_pynvml.nvmlShutdown.assert_called_once()

    def test_pynvml_unavailable_nvidia_smi_success(self):
        """pynvml unavailable: falls back to nvidia-smi CSV line count."""
        import scripts.experiment_438_gpu1_zombie_fix as exp438  # noqa: PLC0415

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception("pynvml unavailable")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 3090\nNVIDIA GeForce RTX 3090\n"

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch("subprocess.run", return_value=mock_result),
        ):
            count = exp438._detect_n_gpus()

        assert count == 2

    def test_nvidia_smi_nonzero_returncode(self):
        """nvidia-smi non-zero return code: fallback continues."""
        import scripts.experiment_438_gpu1_zombie_fix as exp438  # noqa: PLC0415

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception("pynvml unavailable")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch("subprocess.run", return_value=mock_result),
        ):
            count = exp438._detect_n_gpus()

        assert count == 0

    def test_both_unavailable_returns_zero(self):
        """Both pynvml and nvidia-smi unavailable → 0 (CI safe)."""
        import scripts.experiment_438_gpu1_zombie_fix as exp438  # noqa: PLC0415

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception("pynvml unavailable")

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")),
        ):
            count = exp438._detect_n_gpus()

        assert count == 0


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------


class TestMain:
    """main() — calls run_experiment() inside watchdog, logs headline."""

    def test_main_runs_without_error(self):
        """main() completes without raising when run_experiment() succeeds."""
        import scripts.experiment_438_gpu1_zombie_fix as exp438  # noqa: PLC0415

        mock_artifact = {
            "honest_verdict": "ci_mode",
            "fix_applied": False,
            "n_gpus_detected": 0,
            "post_fix_gpu1_util_pct": None,
            "experiment": 438,
        }

        with (
            patch("scripts.experiment_438_gpu1_zombie_fix.run_experiment", return_value=mock_artifact),
            patch("scripts.experiment_438_gpu1_zombie_fix.get_timeout_minutes", return_value=1),
            patch("scripts.experiment_438_gpu1_zombie_fix.ExperimentTimeoutWatchdog") as mock_watchdog,
        ):
            mock_watchdog.return_value.__enter__ = lambda s: s
            mock_watchdog.return_value.__exit__ = MagicMock(return_value=False)
            exp438.main()

        mock_watchdog.assert_called_once_with(
            experiment_id=438,
            timeout_minutes=1,
            result_path=str(exp438._REPO_ROOT / exp438.DELIVERABLE),
        )

    def test_main_calls_run_experiment_once(self):
        """main() calls run_experiment() exactly once."""
        import scripts.experiment_438_gpu1_zombie_fix as exp438  # noqa: PLC0415

        mock_artifact = {
            "honest_verdict": "ci_mode",
            "fix_applied": False,
            "n_gpus_detected": 0,
            "post_fix_gpu1_util_pct": None,
        }

        with (
            patch(
                "scripts.experiment_438_gpu1_zombie_fix.run_experiment",
                return_value=mock_artifact,
            ) as mock_run,
            patch("scripts.experiment_438_gpu1_zombie_fix.get_timeout_minutes", return_value=1),
            patch("scripts.experiment_438_gpu1_zombie_fix.ExperimentTimeoutWatchdog") as mock_watchdog,
        ):
            mock_watchdog.return_value.__enter__ = lambda s: s
            mock_watchdog.return_value.__exit__ = MagicMock(return_value=False)
            exp438.main()

        mock_run.assert_called_once()
