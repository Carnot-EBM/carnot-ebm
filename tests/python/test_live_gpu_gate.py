"""Tests for python/carnot/pipeline/live_gpu_gate.py — RETRO-015 hard gate.

Coverage targets (100% required)
---------------------------------
- LiveGPUGate.check_env_var() → True when CARNOT_FORCE_LIVE=1, False otherwise
- LiveGPUGate.check_gpu_live() → delegates to diagnose_live_gpu().is_live_capable
- LiveGPUGate.require_live() → raises RuntimeError when env var missing
- LiveGPUGate.require_live() → raises RuntimeError when GPU not live
- LiveGPUGate.require_live() → returns None when both checks pass
- LiveGPUGate.require_live_or_blocked() → returns blocked artifact on failure
- LiveGPUGate.require_live_or_blocked() → returns None on success
- LiveGPUGate.verify_subprocess_env_propagation() → True when var set, False when absent
- build_session_startup_script() → returns valid shell script content
- check_session_startup_exists() → reflects filesystem state

Spec: REQ-INFRA-017, REQ-INFRA-018,
      SCENARIO-INFRA-019, SCENARIO-INFRA-020, SCENARIO-INFRA-021
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.live_gpu_gate import (
    LiveGPUGate,
    build_session_startup_script,
    check_session_startup_exists,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_tmpl(tmp_path: Path) -> Any:
    """Return a minimal mock ExperimentTemplate that has a working build_result()."""
    tmpl = MagicMock()
    tmpl.build_result.side_effect = lambda data, status, **kwargs: {
        "status": status,
        **kwargs,
        **data,
    }
    return tmpl


# ---------------------------------------------------------------------------
# check_env_var
# ---------------------------------------------------------------------------


class TestCheckEnvVar:
    """SCENARIO-INFRA-019: env var check."""

    def test_returns_true_when_set(self) -> None:
        """check_env_var() returns True when CARNOT_FORCE_LIVE=1."""
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
            assert LiveGPUGate.check_env_var() is True

    def test_returns_false_when_not_set(self) -> None:
        """check_env_var() returns False when CARNOT_FORCE_LIVE is absent."""
        env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
        with patch.dict(os.environ, env, clear=True):
            assert LiveGPUGate.check_env_var() is False

    def test_returns_false_when_set_to_zero(self) -> None:
        """check_env_var() returns False when CARNOT_FORCE_LIVE=0 (not '1')."""
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            assert LiveGPUGate.check_env_var() is False

    def test_returns_false_when_set_to_other_value(self) -> None:
        """check_env_var() returns False for values other than '1'."""
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "true"}):
            assert LiveGPUGate.check_env_var() is False


# ---------------------------------------------------------------------------
# check_gpu_live
# ---------------------------------------------------------------------------


class TestCheckGpuLive:
    """check_gpu_live() delegates to diagnose_live_gpu()."""

    def test_returns_true_when_live_capable(self) -> None:
        """check_gpu_live() returns True when diagnose_live_gpu says is_live_capable."""
        mock_diag = MagicMock(is_live_capable=True)
        with patch(
            "carnot.pipeline.live_gpu_gate.diagnose_live_gpu",
            return_value=mock_diag,
        ):
            assert LiveGPUGate.check_gpu_live() is True

    def test_returns_false_when_not_live_capable(self) -> None:
        """check_gpu_live() returns False when diagnose_live_gpu says not capable."""
        mock_diag = MagicMock(is_live_capable=False)
        with patch(
            "carnot.pipeline.live_gpu_gate.diagnose_live_gpu",
            return_value=mock_diag,
        ):
            assert LiveGPUGate.check_gpu_live() is False

    def test_passes_model_ids(self) -> None:
        """check_gpu_live() passes model_ids to diagnose_live_gpu."""
        mock_diag = MagicMock(is_live_capable=True)
        with patch(
            "carnot.pipeline.live_gpu_gate.diagnose_live_gpu",
            return_value=mock_diag,
        ) as mock_fn:
            LiveGPUGate.check_gpu_live(model_ids=["Qwen/Qwen3.5-0.8B"])
            mock_fn.assert_called_once_with(["Qwen/Qwen3.5-0.8B"])


# ---------------------------------------------------------------------------
# require_live
# ---------------------------------------------------------------------------


class TestRequireLive:
    """SCENARIO-INFRA-019/020: require_live() raises or returns None."""

    def test_raises_when_env_var_missing(self) -> None:
        """require_live() raises RuntimeError when CARNOT_FORCE_LIVE not set.

        SCENARIO-INFRA-019: env var missing → RuntimeError with useful message.
        """
        env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="CARNOT_FORCE_LIVE not set"):
                LiveGPUGate.require_live()

    def test_raises_when_gpu_not_live(self) -> None:
        """require_live() raises RuntimeError when GPU is not live.

        SCENARIO-INFRA-020: env var set but GPU not capable → RuntimeError.
        """
        mock_diag = MagicMock(is_live_capable=False)
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
            with patch(
                "carnot.pipeline.live_gpu_gate.diagnose_live_gpu",
                return_value=mock_diag,
            ):
                with pytest.raises(RuntimeError, match="is_live_capable=False"):
                    LiveGPUGate.require_live()

    def test_returns_none_when_all_pass(self) -> None:
        """require_live() returns None when env var set and GPU is live."""
        mock_diag = MagicMock(is_live_capable=True)
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
            with patch(
                "carnot.pipeline.live_gpu_gate.diagnose_live_gpu",
                return_value=mock_diag,
            ):
                result = LiveGPUGate.require_live()
                assert result is None

    def test_returns_none_with_model_ids(self) -> None:
        """require_live(model_ids=[...]) passes IDs to check_gpu_live."""
        mock_diag = MagicMock(is_live_capable=True)
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
            with patch(
                "carnot.pipeline.live_gpu_gate.diagnose_live_gpu",
                return_value=mock_diag,
            ) as mock_fn:
                LiveGPUGate.require_live(model_ids=["m1"])
                mock_fn.assert_called_once_with(["m1"])


# ---------------------------------------------------------------------------
# require_live_or_blocked
# ---------------------------------------------------------------------------


class TestRequireLiveOrBlocked:
    """require_live_or_blocked() returns blocked dict on failure, None on success."""

    def test_returns_blocked_when_env_var_missing(self, tmp_path: Path) -> None:
        """Returns a blocked artifact dict when env var is missing."""
        tmpl = _make_mock_tmpl(tmp_path)
        env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
        with patch.dict(os.environ, env, clear=True):
            result = LiveGPUGate.require_live_or_blocked(tmpl, [])
        assert result is not None
        assert result["status"] == "blocked"
        assert "CARNOT_FORCE_LIVE not set" in result["blocked_reason"]

    def test_returns_blocked_when_gpu_not_live(self, tmp_path: Path) -> None:
        """Returns a blocked artifact dict when GPU is not live."""
        tmpl = _make_mock_tmpl(tmp_path)
        mock_diag = MagicMock(is_live_capable=False)
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
            with patch(
                "carnot.pipeline.live_gpu_gate.diagnose_live_gpu",
                return_value=mock_diag,
            ):
                result = LiveGPUGate.require_live_or_blocked(tmpl, [])
        assert result is not None
        assert result["status"] == "blocked"
        assert "is_live_capable=False" in result["blocked_reason"]

    def test_returns_none_when_live(self, tmp_path: Path) -> None:
        """Returns None when env var is set and GPU is live."""
        tmpl = _make_mock_tmpl(tmp_path)
        mock_diag = MagicMock(is_live_capable=True)
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
            with patch(
                "carnot.pipeline.live_gpu_gate.diagnose_live_gpu",
                return_value=mock_diag,
            ):
                result = LiveGPUGate.require_live_or_blocked(tmpl, [])
        assert result is None


# ---------------------------------------------------------------------------
# verify_subprocess_env_propagation
# ---------------------------------------------------------------------------


class TestVerifySubprocessEnvPropagation:
    """SCENARIO-INFRA-021: subprocess inherits env var."""

    def test_returns_true_when_var_set(self) -> None:
        """verify_subprocess_env_propagation returns True when var is in env.

        SCENARIO-INFRA-021: parent sets var → subprocess inherits it.
        """
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
            result = LiveGPUGate.verify_subprocess_env_propagation("CARNOT_FORCE_LIVE")
        assert result is True

    def test_returns_false_when_var_absent(self) -> None:
        """verify_subprocess_env_propagation returns False when var not in env."""
        env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
        with patch.dict(os.environ, env, clear=True):
            result = LiveGPUGate.verify_subprocess_env_propagation("CARNOT_FORCE_LIVE")
        assert result is False

    def test_default_var_name_is_carnot_force_live(self) -> None:
        """Default env_var parameter is CARNOT_FORCE_LIVE."""
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
            result = LiveGPUGate.verify_subprocess_env_propagation()
        assert result is True

    def test_works_for_arbitrary_var(self) -> None:
        """verify_subprocess_env_propagation works for any env var name."""
        with patch.dict(os.environ, {"MY_TEST_VAR": "1"}):
            result = LiveGPUGate.verify_subprocess_env_propagation("MY_TEST_VAR")
        assert result is True


# ---------------------------------------------------------------------------
# build_session_startup_script
# ---------------------------------------------------------------------------


class TestBuildSessionStartupScript:
    """build_session_startup_script() returns valid shell script content."""

    def test_contains_shebang(self, tmp_path: Path) -> None:
        """Script starts with bash shebang."""
        content = build_session_startup_script(tmp_path)
        assert content.startswith("#!/usr/bin/env bash")

    def test_contains_export_line(self, tmp_path: Path) -> None:
        """Script contains 'export CARNOT_FORCE_LIVE=1'."""
        content = build_session_startup_script(tmp_path)
        assert "export CARNOT_FORCE_LIVE=1" in content

    def test_sources_conductor_gpu_env(self, tmp_path: Path) -> None:
        """Script sources conductor_gpu_env.sh."""
        content = build_session_startup_script(tmp_path)
        assert "conductor_gpu_env.sh" in content

    def test_contains_confirmation_echo(self, tmp_path: Path) -> None:
        """Script echoes a confirmation line with [session_startup] prefix."""
        content = build_session_startup_script(tmp_path)
        assert "[session_startup]" in content
        assert "CARNOT_FORCE_LIVE=1 exported" in content

    def test_contains_set_euo_pipefail(self, tmp_path: Path) -> None:
        """Script uses set -euo pipefail for safety."""
        content = build_session_startup_script(tmp_path)
        assert "set -euo pipefail" in content


# ---------------------------------------------------------------------------
# check_session_startup_exists
# ---------------------------------------------------------------------------


class TestCheckSessionStartupExists:
    """check_session_startup_exists() reflects filesystem state."""

    def test_returns_false_when_missing(self, tmp_path: Path) -> None:
        """Returns False when scripts/session_startup.sh does not exist."""
        assert check_session_startup_exists(tmp_path) is False

    def test_returns_true_when_exists(self, tmp_path: Path) -> None:
        """Returns True when scripts/session_startup.sh exists."""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "session_startup.sh").write_text("#!/usr/bin/env bash\n")
        assert check_session_startup_exists(tmp_path) is True
