"""Tests for Experiment 855 — EnvPropagationGuard (REQ-INFRA-070, SCENARIO-INFRA-080).

Every test must assert.  No skips.  100% coverage of the code added in Exp 855.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure scripts/ is on the path so experiment_template is importable.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from experiment_template import EnvPropagationGuard, ExperimentTemplate  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patched_guard(tmp_path: Path):
    """Context manager: redirect EnvPropagationGuard to a temp file."""
    return patch.object(EnvPropagationGuard, "_path", tmp_path / "session_env")


# ---------------------------------------------------------------------------
# EnvPropagationGuard.write_session_env
# ---------------------------------------------------------------------------


class TestWriteSessionEnv:
    """REQ-INFRA-070: write_session_env() writes KEY=VALUE lines."""

    def test_creates_file_with_entry(self, tmp_path):
        """write_session_env creates ~/.carnot_session_env with the given key."""
        with _patched_guard(tmp_path):
            EnvPropagationGuard.write_session_env({"CARNOT_FORCE_LIVE": "1"})
            content = EnvPropagationGuard._path.read_text()
        assert "CARNOT_FORCE_LIVE=1" in content

    def test_overwrites_existing_key(self, tmp_path):
        """write_session_env overwrites a key that already exists."""
        with _patched_guard(tmp_path):
            EnvPropagationGuard.write_session_env({"FOO": "old"})
            EnvPropagationGuard.write_session_env({"FOO": "new"})
            content = EnvPropagationGuard._path.read_text()
        assert "FOO=new" in content
        assert "FOO=old" not in content

    def test_preserves_other_keys(self, tmp_path):
        """write_session_env preserves keys it was not asked to change."""
        with _patched_guard(tmp_path):
            EnvPropagationGuard.write_session_env({"KEY_A": "1"})
            EnvPropagationGuard.write_session_env({"KEY_B": "2"})
            content = EnvPropagationGuard._path.read_text()
        assert "KEY_A=1" in content
        assert "KEY_B=2" in content

    def test_multiple_vars_in_one_call(self, tmp_path):
        """write_session_env handles multiple vars passed at once."""
        with _patched_guard(tmp_path):
            EnvPropagationGuard.write_session_env({"A": "1", "B": "2"})
            content = EnvPropagationGuard._path.read_text()
        assert "A=1" in content
        assert "B=2" in content


# ---------------------------------------------------------------------------
# EnvPropagationGuard.load_session_env
# ---------------------------------------------------------------------------


class TestLoadSessionEnv:
    """REQ-INFRA-070: load_session_env() applies vars to os.environ."""

    def test_applies_missing_var(self, tmp_path):
        """load_session_env sets a var that is not already in os.environ."""
        with _patched_guard(tmp_path):
            EnvPropagationGuard.write_session_env({"_TEST_CARNOT_EXP855": "loaded"})
            saved = os.environ.pop("_TEST_CARNOT_EXP855", None)
            try:
                applied = EnvPropagationGuard.load_session_env()
                assert os.environ.get("_TEST_CARNOT_EXP855") == "loaded"
                assert "_TEST_CARNOT_EXP855" in applied
            finally:
                os.environ.pop("_TEST_CARNOT_EXP855", None)
                if saved is not None:
                    os.environ["_TEST_CARNOT_EXP855"] = saved

    def test_does_not_override_existing_var(self, tmp_path):
        """load_session_env does NOT overwrite a var that is already set."""
        with _patched_guard(tmp_path):
            EnvPropagationGuard.write_session_env({"_TEST_CARNOT_EXP855": "file_val"})
            os.environ["_TEST_CARNOT_EXP855"] = "existing_val"
            try:
                applied = EnvPropagationGuard.load_session_env()
                assert os.environ["_TEST_CARNOT_EXP855"] == "existing_val"
                assert "_TEST_CARNOT_EXP855" not in applied
            finally:
                os.environ.pop("_TEST_CARNOT_EXP855", None)

    def test_returns_empty_when_file_absent(self, tmp_path):
        """load_session_env returns {} when ~/.carnot_session_env does not exist."""
        with _patched_guard(tmp_path):
            # file was never created
            result = EnvPropagationGuard.load_session_env()
        assert result == {}

    def test_ignores_comment_lines(self, tmp_path):
        """load_session_env skips lines starting with #."""
        with _patched_guard(tmp_path):
            EnvPropagationGuard._path.write_text("# this is a comment\n_TEST_COMMENT_KEY=val\n")
            saved = os.environ.pop("_TEST_COMMENT_KEY", None)
            try:
                applied = EnvPropagationGuard.load_session_env()
                assert "_TEST_COMMENT_KEY" in applied
            finally:
                os.environ.pop("_TEST_COMMENT_KEY", None)
                if saved is not None:
                    os.environ["_TEST_COMMENT_KEY"] = saved


# ---------------------------------------------------------------------------
# ExperimentTemplate.assert_live_env_if_gpu
# ---------------------------------------------------------------------------


class TestAssertLiveEnvIfGpu:
    """REQ-INFRA-070: assert_live_env_if_gpu() raises for GPU experiments without env var."""

    def test_raises_when_gpu_required_and_env_missing(self, tmp_path):
        """Raises RuntimeError when requires_gpu=True and CARNOT_FORCE_LIVE is absent."""
        saved = os.environ.pop("CARNOT_FORCE_LIVE", None)
        try:
            tmpl = ExperimentTemplate(
                855,
                "test",
                "results/test_855.json",
                requires_gpu=True,
                repo_root=tmp_path,
            )
            # Override the session file path so load_session_env finds nothing
            with _patched_guard(tmp_path):
                # Reset state that __init__ may have loaded
                os.environ.pop("CARNOT_FORCE_LIVE", None)
                with pytest.raises(RuntimeError, match="LIVE-ENV not propagated"):
                    tmpl.assert_live_env_if_gpu()
        finally:
            if saved is not None:
                os.environ["CARNOT_FORCE_LIVE"] = saved
            else:
                os.environ.pop("CARNOT_FORCE_LIVE", None)

    def test_no_raise_when_env_set(self, tmp_path):
        """Does NOT raise when CARNOT_FORCE_LIVE=1 is set, even for GPU experiment."""
        os.environ["CARNOT_FORCE_LIVE"] = "1"
        try:
            tmpl = ExperimentTemplate(
                855,
                "test",
                "results/test_855.json",
                requires_gpu=True,
                repo_root=tmp_path,
            )
            # Should not raise
            tmpl.assert_live_env_if_gpu()
        finally:
            os.environ.pop("CARNOT_FORCE_LIVE", None)

    def test_no_raise_for_cpu_experiment(self, tmp_path):
        """Does NOT raise for CPU-only experiments regardless of env var."""
        saved = os.environ.pop("CARNOT_FORCE_LIVE", None)
        try:
            tmpl = ExperimentTemplate(
                855,
                "test",
                "results/test_855.json",
                requires_gpu=False,
                repo_root=tmp_path,
            )
            # Must not raise even without the env var
            tmpl.assert_live_env_if_gpu()
        finally:
            if saved is not None:
                os.environ["CARNOT_FORCE_LIVE"] = saved


# ---------------------------------------------------------------------------
# ExperimentTemplate.apply_env_autofix
# ---------------------------------------------------------------------------


class TestApplyEnvAutofix:
    """REQ-INFRA-070: apply_env_autofix() sets env AND writes to session file."""

    def test_sets_os_environ(self, tmp_path):
        """apply_env_autofix sets CARNOT_FORCE_LIVE in os.environ."""
        saved = os.environ.pop("CARNOT_FORCE_LIVE", None)
        try:
            with _patched_guard(tmp_path):
                tmpl = ExperimentTemplate(
                    855,
                    "test",
                    "results/test_855.json",
                    requires_gpu=False,
                    repo_root=tmp_path,
                )
                os.environ.pop("CARNOT_FORCE_LIVE", None)
                tmpl.apply_env_autofix()
                assert os.environ.get("CARNOT_FORCE_LIVE") == "1"
        finally:
            if saved is not None:
                os.environ["CARNOT_FORCE_LIVE"] = saved
            else:
                os.environ.pop("CARNOT_FORCE_LIVE", None)

    def test_writes_session_file(self, tmp_path):
        """apply_env_autofix writes CARNOT_FORCE_LIVE to the session env file."""
        with _patched_guard(tmp_path):
            tmpl = ExperimentTemplate(
                855,
                "test",
                "results/test_855.json",
                requires_gpu=False,
                repo_root=tmp_path,
            )
            tmpl.apply_env_autofix()
            content = EnvPropagationGuard._path.read_text()
        assert "CARNOT_FORCE_LIVE=1" in content


# ---------------------------------------------------------------------------
# ExperimentTemplate.__init__ sources session env
# ---------------------------------------------------------------------------


class TestInitSourcesSessionEnv:
    """SCENARIO-INFRA-080: __init__ loads session env before anything else."""

    def test_init_loads_session_env(self, tmp_path):
        """__init__ calls load_session_env() so session vars are present immediately."""
        with _patched_guard(tmp_path):
            EnvPropagationGuard.write_session_env({"_TEST_INIT_LOAD": "yes"})
            saved = os.environ.pop("_TEST_INIT_LOAD", None)
            try:
                # Construction must load the var
                ExperimentTemplate(
                    855,
                    "test",
                    "results/test_855.json",
                    requires_gpu=False,
                    repo_root=tmp_path,
                )
                assert os.environ.get("_TEST_INIT_LOAD") == "yes"
            finally:
                os.environ.pop("_TEST_INIT_LOAD", None)
                if saved is not None:
                    os.environ["_TEST_INIT_LOAD"] = saved
