"""Tests for carnot.pipeline.experiment_watchdog.

100% coverage targets:
  - ExperimentTimeoutResult dataclass
  - ExperimentTimeoutWatchdog: start, stop, is_active, elapsed_minutes,
    _on_timeout (with and without result_path), context manager
  - get_timeout_minutes (default and env var)
  - build_timeout_artifact

Spec: REQ-INFRA-023, REQ-INFRA-024,
      SCENARIO-INFRA-028, SCENARIO-INFRA-029, SCENARIO-INFRA-030
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.experiment_watchdog import (
    ExperimentTimeoutResult,
    ExperimentTimeoutWatchdog,
    build_timeout_artifact,
    get_timeout_minutes,
)


# ---------------------------------------------------------------------------
# ExperimentTimeoutResult
# ---------------------------------------------------------------------------


class TestExperimentTimeoutResult:
    """REQ-INFRA-023: dataclass has correct fields."""

    def test_fields(self):
        r = ExperimentTimeoutResult(
            experiment_id=425,
            timeout_minutes=45,
            elapsed_minutes=3.14,
            timed_out=False,
            partial_result_path=None,
        )
        assert r.experiment_id == 425
        assert r.timeout_minutes == 45
        assert r.elapsed_minutes == 3.14
        assert r.timed_out is False
        assert r.partial_result_path is None

    def test_fields_with_path(self):
        r = ExperimentTimeoutResult(
            experiment_id=1,
            timeout_minutes=10,
            elapsed_minutes=9.9,
            timed_out=True,
            partial_result_path="/tmp/partial.json",
        )
        assert r.timed_out is True
        assert r.partial_result_path == "/tmp/partial.json"


# ---------------------------------------------------------------------------
# get_timeout_minutes
# ---------------------------------------------------------------------------


class TestGetTimeoutMinutes:
    """REQ-INFRA-024, SCENARIO-INFRA-030."""

    def test_default_when_unset(self, monkeypatch):
        # SCENARIO-INFRA-030: default is 45
        monkeypatch.delenv("CARNOT_CONDUCTOR_TIMEOUT_MINUTES", raising=False)
        assert get_timeout_minutes() == 45

    def test_reads_env_var(self, monkeypatch):
        # SCENARIO-INFRA-030: reads the env var
        monkeypatch.setenv("CARNOT_CONDUCTOR_TIMEOUT_MINUTES", "30")
        assert get_timeout_minutes() == 30

    def test_env_var_empty_string_defaults(self, monkeypatch):
        # Empty string behaves like unset
        monkeypatch.setenv("CARNOT_CONDUCTOR_TIMEOUT_MINUTES", "")
        assert get_timeout_minutes() == 45

    def test_env_var_other_value(self, monkeypatch):
        monkeypatch.setenv("CARNOT_CONDUCTOR_TIMEOUT_MINUTES", "120")
        assert get_timeout_minutes() == 120


# ---------------------------------------------------------------------------
# ExperimentTimeoutWatchdog — basic properties
# ---------------------------------------------------------------------------


class TestWatchdogProperties:
    """REQ-INFRA-023: start, stop, is_active, elapsed_minutes."""

    def test_default_timeout_is_45(self):
        w = ExperimentTimeoutWatchdog(experiment_id=1)
        assert w.timeout_minutes == 45

    def test_explicit_timeout(self):
        w = ExperimentTimeoutWatchdog(experiment_id=1, timeout_minutes=60)
        assert w.timeout_minutes == 60

    def test_elapsed_before_start_is_zero(self):
        w = ExperimentTimeoutWatchdog(experiment_id=1, timeout_minutes=100)
        assert w.elapsed_minutes() == 0.0

    def test_is_active_before_start(self):
        w = ExperimentTimeoutWatchdog(experiment_id=1, timeout_minutes=100)
        assert w.is_active() is False

    def test_is_active_after_start(self):
        w = ExperimentTimeoutWatchdog(experiment_id=1, timeout_minutes=100)
        w.start()
        try:
            assert w.is_active() is True
        finally:
            w.stop()

    def test_elapsed_after_start_is_positive(self):
        w = ExperimentTimeoutWatchdog(experiment_id=1, timeout_minutes=100)
        w.start()
        try:
            time.sleep(0.05)
            assert w.elapsed_minutes() > 0.0
        finally:
            w.stop()

    def test_is_active_false_after_stop(self):
        w = ExperimentTimeoutWatchdog(experiment_id=1, timeout_minutes=100)
        w.start()
        w.stop()
        assert w.is_active() is False

    def test_stop_before_start_is_safe(self):
        # stop() before start() must not raise
        w = ExperimentTimeoutWatchdog(experiment_id=1, timeout_minutes=100)
        w.stop()  # should not raise

    def test_stop_twice_is_idempotent(self):
        w = ExperimentTimeoutWatchdog(experiment_id=1, timeout_minutes=100)
        w.start()
        w.stop()
        w.stop()  # second stop must not raise


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-029: stop before timeout prevents firing
# ---------------------------------------------------------------------------


class TestStopPreventsTimeout:
    """SCENARIO-INFRA-029."""

    def test_stop_before_timeout(self, monkeypatch):
        # Use a short timeout; stop() before it fires
        fired = []

        w = ExperimentTimeoutWatchdog(experiment_id=1, timeout_minutes=1.0)

        original_on_timeout = w._on_timeout

        def patched_on_timeout():
            fired.append(True)
            original_on_timeout()

        w._on_timeout = patched_on_timeout
        w.start()
        w.stop()

        # Give the timer time to fire if it wasn't cancelled
        time.sleep(0.1)
        assert fired == [], "_on_timeout must not fire after stop()"

    def test_start_twice_logs_warning(self, caplog):
        import logging

        w = ExperimentTimeoutWatchdog(experiment_id=1, timeout_minutes=100)
        w.start()
        try:
            with caplog.at_level(logging.WARNING, logger="carnot.pipeline.experiment_watchdog"):
                w.start()  # second call should warn
            assert any("called twice" in r.message for r in caplog.records)
        finally:
            w.stop()


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-028: watchdog fires after timeout
# ---------------------------------------------------------------------------


class TestWatchdogFires:
    """SCENARIO-INFRA-028: watchdog fires sys.exit(1) after timeout elapses."""

    def test_fires_sys_exit_without_result_path(self):
        # Use 0.01 min = 0.6 s; patch sys.exit to capture the call
        w = ExperimentTimeoutWatchdog(experiment_id=99, timeout_minutes=0.01)

        with patch("sys.exit") as mock_exit:
            w.start()
            # Wait long enough for the timer to fire (0.6 s + margin)
            time.sleep(1.0)
            mock_exit.assert_called_once_with(1)

    def test_fires_sys_exit_with_result_path(self, tmp_path):
        result_file = str(tmp_path / "partial.json")
        w = ExperimentTimeoutWatchdog(
            experiment_id=99,
            timeout_minutes=0.01,
            result_path=result_file,
        )

        with patch("sys.exit") as mock_exit:
            w.start()
            time.sleep(1.0)
            mock_exit.assert_called_once_with(1)

        # Partial result JSON must exist
        assert Path(result_file).exists()
        data = json.loads(Path(result_file).read_text())
        assert data["timed_out"] is True
        assert data["experiment"] == 99
        assert data["schema"] == "carnot.timeout_watchdog.partial.v1"

    def test_on_timeout_sets_timed_out_flag(self):
        w = ExperimentTimeoutWatchdog(experiment_id=5, timeout_minutes=100)
        w.start()

        with patch("sys.exit"):
            w._on_timeout()

        assert w._timed_out is True

    def test_on_timeout_logs_error(self, caplog):
        import logging

        w = ExperimentTimeoutWatchdog(experiment_id=5, timeout_minutes=1)
        w.start()

        with caplog.at_level(logging.ERROR, logger="carnot.pipeline.experiment_watchdog"):
            with patch("sys.exit"):
                w._on_timeout()

        assert any("FIRED" in r.message for r in caplog.records)
        w.stop()

    def test_is_active_false_after_timeout(self):
        w = ExperimentTimeoutWatchdog(experiment_id=5, timeout_minutes=100)
        w.start()
        with patch("sys.exit"):
            w._on_timeout()
        assert w.is_active() is False


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    """REQ-INFRA-023: __enter__ calls start(), __exit__ calls stop()."""

    def test_context_manager_normal_flow(self):
        w = ExperimentTimeoutWatchdog(experiment_id=1, timeout_minutes=100)
        with w:
            assert w.is_active() is True
        assert w.is_active() is False  # stop() was called in __exit__

    def test_context_manager_does_not_suppress_exceptions(self):
        w = ExperimentTimeoutWatchdog(experiment_id=1, timeout_minutes=100)
        with pytest.raises(ValueError):
            with w:
                raise ValueError("test error")

    def test_context_manager_stops_on_exception(self):
        w = ExperimentTimeoutWatchdog(experiment_id=1, timeout_minutes=100)
        try:
            with w:
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert w.is_active() is False


# ---------------------------------------------------------------------------
# build_timeout_artifact
# ---------------------------------------------------------------------------


class TestBuildTimeoutArtifact:
    """REQ-INFRA-023: artifact schema and fields."""

    def test_artifact_schema(self):
        r = ExperimentTimeoutResult(
            experiment_id=425,
            timeout_minutes=45,
            elapsed_minutes=5.0,
            timed_out=False,
            partial_result_path=None,
        )
        artifact = build_timeout_artifact(r)
        assert artifact["schema"] == "carnot.timeout_watchdog.v1"
        assert artifact["honest_verdict"] == "watchdog_implemented"
        assert artifact["retro_003_resolved"] is True
        assert artifact["estimated_savings_minutes_per_runaway"] == 99

    def test_artifact_fields_match_result(self):
        r = ExperimentTimeoutResult(
            experiment_id=7,
            timeout_minutes=20,
            elapsed_minutes=1.5,
            timed_out=True,
            partial_result_path="/tmp/x.json",
        )
        artifact = build_timeout_artifact(r)
        assert artifact["experiment_id"] == 7
        assert artifact["timeout_minutes"] == 20
        assert artifact["elapsed_minutes"] == 1.5
        assert artifact["timed_out"] is True
        assert artifact["partial_result_path"] == "/tmp/x.json"

    def test_artifact_is_json_serializable(self):
        r = ExperimentTimeoutResult(425, 45, 0.0, False, None)
        artifact = build_timeout_artifact(r)
        # Must not raise
        json.dumps(artifact)
