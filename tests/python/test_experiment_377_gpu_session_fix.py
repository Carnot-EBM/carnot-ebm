"""Tests for scripts/experiment_377_gpu_session_fix.py — RETRO-015 close.

Coverage targets (100% required)
---------------------------------
- run_experiment: happy path — all scripts exist, subprocess inherits env var
- run_experiment: env_var_set=True/False reflected correctly
- run_experiment: subprocess_inherits_env=True propagates to retro_015_infrastructure_fixed
- run_experiment: session_startup_exists=False → retro_015_infrastructure_fixed=False
- run_experiment: conductor_gpu_env_exists=False → retro_015_infrastructure_fixed=False
- run_experiment: session_startup does not source conductor_gpu_env.sh → fixed=False
- run_experiment: honest_verdict values (all four branches)
- Artifact schema: 'carnot.gpu_session_fix.v1', all required fields present
- main(): writes JSON, prints summary

Spec: REQ-INFRA-017, REQ-INFRA-018,
      SCENARIO-INFRA-019, SCENARIO-INFRA-020, SCENARIO-INFRA-021
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_377_gpu_session_fix import (
    DELIVERABLE,
    EXP_ID,
    TITLE,
    run_experiment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo(
    tmp_path: Path,
    *,
    add_session_startup: bool = True,
    add_conductor_gpu_env: bool = True,
    session_startup_sources_conductor: bool = True,
) -> Path:
    """Create a minimal fake repo structure for testing run_experiment."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "checkpoints").mkdir(parents=True)

    if add_conductor_gpu_env:
        (tmp_path / "scripts" / "conductor_gpu_env.sh").write_text(
            "#!/usr/bin/env bash\nexport CARNOT_FORCE_LIVE=1\n"
        )

    if add_session_startup:
        if session_startup_sources_conductor:
            content = (
                "#!/usr/bin/env bash\n"
                "source conductor_gpu_env.sh\n"
                "export CARNOT_FORCE_LIVE=1\n"
                'echo "[session_startup] CARNOT_FORCE_LIVE=1 exported"\n'
            )
        else:
            content = (
                "#!/usr/bin/env bash\n"
                "export CARNOT_FORCE_LIVE=1\n"
            )
        (tmp_path / "scripts" / "session_startup.sh").write_text(content)

    return tmp_path


def _mock_diag(is_live_capable: bool = False, failure_reason: str = "") -> Any:
    """Return a minimal mock LiveGPUDiagnostic."""
    mock = MagicMock()
    mock.is_live_capable = is_live_capable
    mock.failure_reason = failure_reason
    return mock


def _run_with_subprocess_mock(
    tmp_path: Path,
    *,
    subprocess_inherits: bool = True,
    env_var_set: bool = True,
    is_live_capable: bool = False,
    add_session_startup: bool = True,
    add_conductor_gpu_env: bool = True,
    session_startup_sources_conductor: bool = True,
) -> dict:
    """Helper: run_experiment with subprocess + diagnose_live_gpu mocked."""
    repo = _make_repo(
        tmp_path,
        add_session_startup=add_session_startup,
        add_conductor_gpu_env=add_conductor_gpu_env,
        session_startup_sources_conductor=session_startup_sources_conductor,
    )
    env = dict(os.environ)
    if env_var_set:
        env["CARNOT_FORCE_LIVE"] = "1"
    else:
        env.pop("CARNOT_FORCE_LIVE", None)

    with patch.dict(os.environ, env, clear=True):
        with patch(
            "scripts.experiment_377_gpu_session_fix.LiveGPUGate.verify_subprocess_env_propagation",
            return_value=subprocess_inherits,
        ):
            with patch(
                "scripts.experiment_377_gpu_session_fix.diagnose_live_gpu",
                return_value=_mock_diag(is_live_capable=is_live_capable),
            ):
                return run_experiment(repo)


# ---------------------------------------------------------------------------
# Artifact schema
# ---------------------------------------------------------------------------


class TestArtifactSchema:
    """Validate the artifact shape produced by run_experiment."""

    def test_required_result_fields_present(self, tmp_path: Path) -> None:
        """All REQUIRED_RESULT_FIELDS must be in the artifact."""
        artifact = _run_with_subprocess_mock(tmp_path)
        for field in ("experiment", "run_date", "started_at",
                      "finished_at", "duration_s", "status", "title"):
            assert field in artifact, f"Missing field: {field}"

    def test_schema_is_gpu_session_fix_v1(self, tmp_path: Path) -> None:
        """fix_schema field must be 'carnot.gpu_session_fix.v1'.

        ExperimentTemplate.build_result() overwrites 'schema' with sorted key list;
        the experiment-specific version identifier is stored as 'fix_schema'.
        """
        artifact = _run_with_subprocess_mock(tmp_path)
        assert artifact["fix_schema"] == "carnot.gpu_session_fix.v1"

    def test_experiment_id_is_377(self, tmp_path: Path) -> None:
        """experiment field must be 377."""
        artifact = _run_with_subprocess_mock(tmp_path)
        assert artifact["experiment"] == 377

    def test_title_is_correct(self, tmp_path: Path) -> None:
        """title must match the TITLE constant."""
        artifact = _run_with_subprocess_mock(tmp_path)
        assert artifact["title"] == TITLE

    def test_all_retro015_fields_present(self, tmp_path: Path) -> None:
        """All RETRO-015 specific fields must be present in the artifact."""
        artifact = _run_with_subprocess_mock(tmp_path)
        for field in (
            "fix_schema",
            "env_var_set",
            "subprocess_inherits_env",
            "session_startup_exists",
            "conductor_gpu_env_exists",
            "session_startup_sources_conductor",
            "is_live_capable",
            "retro_015_infrastructure_fixed",
            "honest_verdict",
            "diagnostic_failure_reason",
        ):
            assert field in artifact, f"Missing RETRO-015 field: {field}"


# ---------------------------------------------------------------------------
# Happy path — infrastructure_fixed
# ---------------------------------------------------------------------------


class TestHappyPath:
    """retro_015_infrastructure_fixed=True when all checks pass."""

    def test_infrastructure_fixed_when_all_pass(self, tmp_path: Path) -> None:
        """infrastructure_fixed=True when subprocess inherits env and scripts exist."""
        artifact = _run_with_subprocess_mock(
            tmp_path,
            subprocess_inherits=True,
            add_session_startup=True,
            add_conductor_gpu_env=True,
            session_startup_sources_conductor=True,
        )
        assert artifact["retro_015_infrastructure_fixed"] is True

    def test_honest_verdict_infrastructure_fixed(self, tmp_path: Path) -> None:
        """honest_verdict='infrastructure_fixed' in happy path."""
        artifact = _run_with_subprocess_mock(
            tmp_path,
            subprocess_inherits=True,
            add_session_startup=True,
            add_conductor_gpu_env=True,
            session_startup_sources_conductor=True,
        )
        assert artifact["honest_verdict"] == "infrastructure_fixed"

    def test_status_success_when_fixed(self, tmp_path: Path) -> None:
        """status='success' when retro_015_infrastructure_fixed=True."""
        artifact = _run_with_subprocess_mock(
            tmp_path,
            subprocess_inherits=True,
            add_session_startup=True,
            add_conductor_gpu_env=True,
            session_startup_sources_conductor=True,
        )
        assert artifact["status"] == "success"


# ---------------------------------------------------------------------------
# env_var_set field reflects current env
# ---------------------------------------------------------------------------


class TestEnvVarSet:
    """env_var_set field is True/False based on current env, not fix verdict."""

    def test_env_var_set_true_when_set(self, tmp_path: Path) -> None:
        """env_var_set=True when CARNOT_FORCE_LIVE=1 in current env."""
        artifact = _run_with_subprocess_mock(tmp_path, env_var_set=True)
        assert artifact["env_var_set"] is True

    def test_env_var_set_false_when_absent(self, tmp_path: Path) -> None:
        """env_var_set=False when CARNOT_FORCE_LIVE not in current env."""
        artifact = _run_with_subprocess_mock(tmp_path, env_var_set=False)
        assert artifact["env_var_set"] is False


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    """Each failure path sets honest_verdict and retro_015_infrastructure_fixed=False."""

    def test_env_propagation_failed(self, tmp_path: Path) -> None:
        """honest_verdict='env_propagation_failed' when subprocess cannot inherit var."""
        artifact = _run_with_subprocess_mock(
            tmp_path,
            subprocess_inherits=False,
            add_session_startup=True,
            add_conductor_gpu_env=True,
        )
        assert artifact["retro_015_infrastructure_fixed"] is False
        assert artifact["honest_verdict"] == "env_propagation_failed"
        assert artifact["status"] == "blocked"

    def test_scripts_missing_no_session_startup(self, tmp_path: Path) -> None:
        """honest_verdict='scripts_missing' when session_startup.sh absent."""
        artifact = _run_with_subprocess_mock(
            tmp_path,
            subprocess_inherits=True,
            add_session_startup=False,
            add_conductor_gpu_env=True,
        )
        assert artifact["retro_015_infrastructure_fixed"] is False
        assert artifact["honest_verdict"] == "scripts_missing"

    def test_scripts_missing_no_conductor_env(self, tmp_path: Path) -> None:
        """honest_verdict='scripts_missing' when conductor_gpu_env.sh absent."""
        artifact = _run_with_subprocess_mock(
            tmp_path,
            subprocess_inherits=True,
            add_session_startup=True,
            add_conductor_gpu_env=False,
            session_startup_sources_conductor=False,
        )
        assert artifact["retro_015_infrastructure_fixed"] is False
        assert artifact["honest_verdict"] == "scripts_missing"

    def test_session_startup_does_not_source_conductor(self, tmp_path: Path) -> None:
        """honest_verdict='session_startup_does_not_source_conductor_env' when not sourcing."""
        artifact = _run_with_subprocess_mock(
            tmp_path,
            subprocess_inherits=True,
            add_session_startup=True,
            add_conductor_gpu_env=True,
            session_startup_sources_conductor=False,
        )
        assert artifact["retro_015_infrastructure_fixed"] is False
        assert artifact["honest_verdict"] == "session_startup_does_not_source_conductor_env"


# ---------------------------------------------------------------------------
# is_live_capable is informational, does not gate retro_015_infrastructure_fixed
# ---------------------------------------------------------------------------


class TestIsLiveCapable:
    """is_live_capable is recorded but does not affect retro_015_infrastructure_fixed."""

    def test_is_live_capable_false_does_not_block_fix(self, tmp_path: Path) -> None:
        """is_live_capable=False (no GPU in CI) does not prevent fixed=True."""
        artifact = _run_with_subprocess_mock(
            tmp_path,
            subprocess_inherits=True,
            is_live_capable=False,
        )
        # Fix is about infrastructure, not hardware; is_live_capable=False is OK.
        assert artifact["retro_015_infrastructure_fixed"] is True
        assert artifact["is_live_capable"] is False

    def test_is_live_capable_true_recorded(self, tmp_path: Path) -> None:
        """is_live_capable=True is recorded when GPU stack is live."""
        artifact = _run_with_subprocess_mock(
            tmp_path,
            subprocess_inherits=True,
            is_live_capable=True,
        )
        assert artifact["is_live_capable"] is True


# ---------------------------------------------------------------------------
# Scripts existence fields
# ---------------------------------------------------------------------------


class TestScriptFields:
    """session_startup_exists and conductor_gpu_env_exists reflect filesystem state."""

    def test_session_startup_exists_true(self, tmp_path: Path) -> None:
        """session_startup_exists=True when file is present."""
        artifact = _run_with_subprocess_mock(tmp_path, add_session_startup=True)
        assert artifact["session_startup_exists"] is True

    def test_session_startup_exists_false(self, tmp_path: Path) -> None:
        """session_startup_exists=False when file is absent."""
        artifact = _run_with_subprocess_mock(tmp_path, add_session_startup=False)
        assert artifact["session_startup_exists"] is False

    def test_conductor_gpu_env_exists_true(self, tmp_path: Path) -> None:
        """conductor_gpu_env_exists=True when file is present."""
        artifact = _run_with_subprocess_mock(tmp_path, add_conductor_gpu_env=True)
        assert artifact["conductor_gpu_env_exists"] is True

    def test_conductor_gpu_env_exists_false(self, tmp_path: Path) -> None:
        """conductor_gpu_env_exists=False when file is absent."""
        artifact = _run_with_subprocess_mock(
            tmp_path,
            add_conductor_gpu_env=False,
            add_session_startup=True,
            session_startup_sources_conductor=False,
        )
        assert artifact["conductor_gpu_env_exists"] is False


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------


class TestMain:
    """main() writes JSON and prints summary."""

    def test_main_writes_json(self, tmp_path: Path, capsys: Any) -> None:
        """main() writes the artifact JSON to the deliverable path."""
        import scripts.experiment_377_gpu_session_fix as exp_mod

        orig_repo_root = exp_mod._REPO_ROOT
        try:
            exp_mod._REPO_ROOT = tmp_path
            # Create minimal structure
            (tmp_path / "scripts").mkdir(exist_ok=True)
            (tmp_path / "results").mkdir(exist_ok=True)
            (tmp_path / "results" / "checkpoints").mkdir(parents=True, exist_ok=True)
            (tmp_path / "scripts" / "session_startup.sh").write_text(
                "source conductor_gpu_env.sh\nexport CARNOT_FORCE_LIVE=1\n"
            )
            (tmp_path / "scripts" / "conductor_gpu_env.sh").write_text(
                "export CARNOT_FORCE_LIVE=1\n"
            )
            with patch(
                "scripts.experiment_377_gpu_session_fix.LiveGPUGate.verify_subprocess_env_propagation",
                return_value=True,
            ):
                with patch(
                    "scripts.experiment_377_gpu_session_fix.diagnose_live_gpu",
                    return_value=_mock_diag(is_live_capable=False),
                ):
                    exp_mod.main()

            output_path = tmp_path / DELIVERABLE
            assert output_path.exists(), f"Expected {output_path} to exist"
            artifact = json.loads(output_path.read_text())
            assert artifact["experiment"] == EXP_ID
        finally:
            exp_mod._REPO_ROOT = orig_repo_root

    def test_main_prints_verdict(self, tmp_path: Path, capsys: Any) -> None:
        """main() prints the honest_verdict to stdout."""
        import scripts.experiment_377_gpu_session_fix as exp_mod

        orig_repo_root = exp_mod._REPO_ROOT
        try:
            exp_mod._REPO_ROOT = tmp_path
            (tmp_path / "scripts").mkdir(exist_ok=True)
            (tmp_path / "results").mkdir(exist_ok=True)
            (tmp_path / "results" / "checkpoints").mkdir(parents=True, exist_ok=True)
            (tmp_path / "scripts" / "session_startup.sh").write_text(
                "source conductor_gpu_env.sh\nexport CARNOT_FORCE_LIVE=1\n"
            )
            (tmp_path / "scripts" / "conductor_gpu_env.sh").write_text(
                "export CARNOT_FORCE_LIVE=1\n"
            )
            with patch(
                "scripts.experiment_377_gpu_session_fix.LiveGPUGate.verify_subprocess_env_propagation",
                return_value=True,
            ):
                with patch(
                    "scripts.experiment_377_gpu_session_fix.diagnose_live_gpu",
                    return_value=_mock_diag(is_live_capable=False),
                ):
                    exp_mod.main()

            captured = capsys.readouterr()
            assert "honest_verdict" in captured.out
        finally:
            exp_mod._REPO_ROOT = orig_repo_root
