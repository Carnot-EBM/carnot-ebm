"""Tests for python/carnot/pipeline/env_autofix.py — 100% coverage.

Coverage targets
----------------
- EnvironmentAutoFix dataclass field assignment
- apply_env_autofix():
  - No GPU (torch.cuda returns False)           → SCENARIO-INFRA-025
  - torch not importable                        → SCENARIO-INFRA-025
  - GPU present, var already set                → SCENARIO-INFRA-027
  - GPU present, var absent → auto_fix_applied  → SCENARIO-INFRA-026
  - Warning log emitted iff auto_fix_applied
- build_env_autofix_artifact():
  - gpu_not_detected verdict
  - gpu_detected_env_was_correct verdict
  - auto_fix_applied verdict
  - gpu_confirmed_live fallback verdict
  - retro_022_resolved True/False
  - preflight dict merged under 'preflight' key

Spec: REQ-INFRA-021, REQ-INFRA-022,
      SCENARIO-INFRA-025, SCENARIO-INFRA-026, SCENARIO-INFRA-027
"""

from __future__ import annotations

import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import (  # noqa: E402
    EnvironmentAutoFix,
    apply_env_autofix,
    build_env_autofix_artifact,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_env():
    """Remove CARNOT_FORCE_LIVE from os.environ (restore after test)."""
    return patch.dict(os.environ, {}, clear=False)


# ---------------------------------------------------------------------------
# EnvironmentAutoFix dataclass
# ---------------------------------------------------------------------------


class TestEnvironmentAutoFixDataclass:
    """REQ-INFRA-021: dataclass field assignment."""

    def test_fields_set_correctly(self):
        r = EnvironmentAutoFix(
            gpu_detected=True,
            carnot_force_live_was_set=False,
            auto_fix_applied=True,
            final_env_value="1",
        )
        assert r.gpu_detected is True
        assert r.carnot_force_live_was_set is False
        assert r.auto_fix_applied is True
        assert r.final_env_value == "1"

    def test_none_final_env_value(self):
        r = EnvironmentAutoFix(
            gpu_detected=False,
            carnot_force_live_was_set=False,
            auto_fix_applied=False,
            final_env_value=None,
        )
        assert r.final_env_value is None


# ---------------------------------------------------------------------------
# apply_env_autofix — SCENARIO-INFRA-025: no GPU
# ---------------------------------------------------------------------------


class TestApplyEnvAutofixNoGPU:
    """SCENARIO-INFRA-025: no GPU → no mutation, auto_fix_applied=False."""

    def test_no_gpu_torch_returns_false(self):
        # torch importable but CUDA unavailable
        with patch.dict(os.environ, {}, clear=False):
            # ensure CARNOT_FORCE_LIVE is absent
            os.environ.pop("CARNOT_FORCE_LIVE", None)
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False
            with patch.dict(sys.modules, {"torch": mock_torch}):
                result = apply_env_autofix()
        assert result.gpu_detected is False
        assert result.auto_fix_applied is False
        assert result.carnot_force_live_was_set is False
        assert result.final_env_value is None

    def test_no_gpu_does_not_mutate_env(self):
        """CARNOT_FORCE_LIVE must NOT appear in env when no GPU."""
        env_copy = dict(os.environ)
        env_copy.pop("CARNOT_FORCE_LIVE", None)
        with patch.dict(os.environ, env_copy, clear=True):
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False
            with patch.dict(sys.modules, {"torch": mock_torch}):
                apply_env_autofix()
            assert "CARNOT_FORCE_LIVE" not in os.environ

    def test_torch_import_error_treated_as_no_gpu(self):
        """If torch is not importable, gpu_detected=False."""
        env_copy = dict(os.environ)
        env_copy.pop("CARNOT_FORCE_LIVE", None)
        with patch.dict(os.environ, env_copy, clear=True):
            # Remove torch from sys.modules and block re-import
            with patch.dict(sys.modules, {"torch": None}):
                result = apply_env_autofix()
        assert result.gpu_detected is False
        assert result.auto_fix_applied is False
        assert result.final_env_value is None

    def test_no_warning_when_no_gpu(self, caplog):
        env_copy = dict(os.environ)
        env_copy.pop("CARNOT_FORCE_LIVE", None)
        with patch.dict(os.environ, env_copy, clear=True):
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False
            with patch.dict(sys.modules, {"torch": mock_torch}):
                with caplog.at_level(logging.WARNING, logger="carnot.pipeline.env_autofix"):
                    apply_env_autofix()
        assert "EnvironmentAutoFix" not in caplog.text


# ---------------------------------------------------------------------------
# apply_env_autofix — SCENARIO-INFRA-027: GPU present, var already set
# ---------------------------------------------------------------------------


class TestApplyEnvAutofixVarAlreadySet:
    """SCENARIO-INFRA-027: CARNOT_FORCE_LIVE already in env."""

    def test_var_already_set_no_fix(self):
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}, clear=False):
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            with patch.dict(sys.modules, {"torch": mock_torch}):
                result = apply_env_autofix()
        assert result.gpu_detected is True
        assert result.carnot_force_live_was_set is True
        assert result.auto_fix_applied is False
        assert result.final_env_value == "1"

    def test_no_warning_when_var_already_set(self, caplog):
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}, clear=False):
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            with patch.dict(sys.modules, {"torch": mock_torch}):
                with caplog.at_level(logging.WARNING, logger="carnot.pipeline.env_autofix"):
                    apply_env_autofix()
        assert "EnvironmentAutoFix" not in caplog.text


# ---------------------------------------------------------------------------
# apply_env_autofix — SCENARIO-INFRA-026: GPU present, var absent → auto-fix
# ---------------------------------------------------------------------------


class TestApplyEnvAutofixAutoFix:
    """SCENARIO-INFRA-026: GPU present and var absent → inject and warn."""

    def test_auto_fix_applied(self):
        env_copy = dict(os.environ)
        env_copy.pop("CARNOT_FORCE_LIVE", None)
        with patch.dict(os.environ, env_copy, clear=True):
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            with patch.dict(sys.modules, {"torch": mock_torch}):
                result = apply_env_autofix()
            assert result.gpu_detected is True
            assert result.carnot_force_live_was_set is False
            assert result.auto_fix_applied is True
            assert result.final_env_value == "1"
            # env mutation persists inside the patch context
            assert os.environ.get("CARNOT_FORCE_LIVE") == "1"

    def test_warning_emitted_on_auto_fix(self, caplog):
        env_copy = dict(os.environ)
        env_copy.pop("CARNOT_FORCE_LIVE", None)
        with patch.dict(os.environ, env_copy, clear=True):
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            with patch.dict(sys.modules, {"torch": mock_torch}):
                with caplog.at_level(logging.WARNING, logger="carnot.pipeline.env_autofix"):
                    apply_env_autofix()
        assert "EnvironmentAutoFix applied CARNOT_FORCE_LIVE=1" in caplog.text

    def test_warning_is_at_warning_level(self, caplog):
        env_copy = dict(os.environ)
        env_copy.pop("CARNOT_FORCE_LIVE", None)
        with patch.dict(os.environ, env_copy, clear=True):
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            with patch.dict(sys.modules, {"torch": mock_torch}):
                with caplog.at_level(logging.WARNING, logger="carnot.pipeline.env_autofix"):
                    apply_env_autofix()
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("EnvironmentAutoFix applied CARNOT_FORCE_LIVE=1" in r.message for r in warning_records)


# ---------------------------------------------------------------------------
# build_env_autofix_artifact
# ---------------------------------------------------------------------------


class TestBuildEnvAutofixArtifact:
    """Tests for build_env_autofix_artifact — all verdict branches."""

    _SAMPLE_PREFLIGHT = {"honest_verdict": "env_not_propagating", "env_var_set": False}

    def _make_result(self, gpu_detected, was_set, auto_fix, final_val):
        return EnvironmentAutoFix(
            gpu_detected=gpu_detected,
            carnot_force_live_was_set=was_set,
            auto_fix_applied=auto_fix,
            final_env_value=final_val,
        )

    def test_schema_key(self):
        r = self._make_result(False, False, False, None)
        art = build_env_autofix_artifact(r, self._SAMPLE_PREFLIGHT)
        assert art["schema"] == "carnot.env_autofix.v1"

    def test_preflight_merged(self):
        r = self._make_result(False, False, False, None)
        art = build_env_autofix_artifact(r, self._SAMPLE_PREFLIGHT)
        assert art["preflight"] == self._SAMPLE_PREFLIGHT

    # --- gpu_not_detected ---

    def test_verdict_gpu_not_detected(self):
        r = self._make_result(False, False, False, None)
        art = build_env_autofix_artifact(r, {})
        assert art["honest_verdict"] == "gpu_not_detected"

    def test_retro_022_not_resolved_when_no_gpu(self):
        r = self._make_result(False, False, False, None)
        art = build_env_autofix_artifact(r, {})
        assert art["retro_022_resolved"] is False

    # --- gpu_detected_env_was_correct ---

    def test_verdict_env_was_correct(self):
        r = self._make_result(True, True, False, "1")
        art = build_env_autofix_artifact(r, {})
        assert art["honest_verdict"] == "gpu_detected_env_was_correct"

    def test_retro_022_resolved_env_was_correct(self):
        r = self._make_result(True, True, False, "1")
        art = build_env_autofix_artifact(r, {})
        assert art["retro_022_resolved"] is True

    # --- auto_fix_applied ---

    def test_verdict_auto_fix_applied(self):
        r = self._make_result(True, False, True, "1")
        art = build_env_autofix_artifact(r, {})
        assert art["honest_verdict"] == "auto_fix_applied"

    def test_retro_022_resolved_auto_fix(self):
        r = self._make_result(True, False, True, "1")
        art = build_env_autofix_artifact(r, {})
        assert art["retro_022_resolved"] is True

    # --- gpu_confirmed_live fallback ---

    def test_verdict_gpu_confirmed_live_fallback(self):
        # gpu_detected=True, was_set=False, auto_fix_applied=False, final_val='1'
        # This is the edge case where var is '1' but neither was preset nor auto-fixed
        r = self._make_result(True, False, False, "1")
        art = build_env_autofix_artifact(r, {})
        assert art["honest_verdict"] == "gpu_confirmed_live"

    def test_retro_022_resolved_gpu_confirmed_live(self):
        r = self._make_result(True, False, False, "1")
        art = build_env_autofix_artifact(r, {})
        assert art["retro_022_resolved"] is True

    # --- field completeness ---

    def test_all_fields_present(self):
        r = self._make_result(True, False, True, "1")
        art = build_env_autofix_artifact(r, self._SAMPLE_PREFLIGHT)
        required = {
            "schema", "honest_verdict", "retro_022_resolved",
            "gpu_detected", "carnot_force_live_was_set", "auto_fix_applied",
            "final_env_value", "preflight",
        }
        assert required.issubset(art.keys())

    def test_fields_echo_result(self):
        r = self._make_result(True, False, True, "1")
        art = build_env_autofix_artifact(r, {})
        assert art["gpu_detected"] is True
        assert art["carnot_force_live_was_set"] is False
        assert art["auto_fix_applied"] is True
        assert art["final_env_value"] == "1"
