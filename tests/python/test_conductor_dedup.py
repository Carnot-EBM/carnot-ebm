"""Tests for ConductorDedupCheck and PartialResultHandoff.

Spec: REQ-INFRA-042, REQ-INFRA-043, REQ-INFRA-044,
      SCENARIO-INFRA-050, SCENARIO-INFRA-051, SCENARIO-INFRA-052
"""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.conductor_dedup import ConductorDedupCheck, PartialResultHandoff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_result(results_dir: Path, exp_id: int, verdict: str, suffix: str = "") -> Path:
    """Write a minimal experiment result JSON for testing."""
    path = results_dir / f"experiment_{exp_id}{suffix}.json"
    path.write_text(json.dumps({"honest_verdict": verdict, "experiment": exp_id}), encoding="utf-8")
    return path


def _make_template(exp_id: int, tmp_path: Path) -> MagicMock:
    """Build a mock ExperimentTemplate with the minimum required attributes."""
    tmpl = MagicMock()
    tmpl.exp_id = exp_id
    return tmpl


# ---------------------------------------------------------------------------
# ConductorDedupCheck.is_blocked_verdict — SCENARIO-INFRA-051
# ---------------------------------------------------------------------------


class TestIsBlockedVerdict:
    """REQ-INFRA-042: blocked verdicts are correctly identified."""

    def test_blocked(self):
        assert ConductorDedupCheck.is_blocked_verdict("blocked") is True

    def test_deferred_to_gpu(self):
        assert ConductorDedupCheck.is_blocked_verdict("deferred_to_gpu") is True

    def test_gpu_required(self):
        assert ConductorDedupCheck.is_blocked_verdict("gpu_required") is True

    def test_scaffolding_only(self):
        assert ConductorDedupCheck.is_blocked_verdict("scaffolding_only") is True

    def test_partial_prefix(self):
        # Any verdict starting with 'partial_' is blocked — SCENARIO-INFRA-051
        assert ConductorDedupCheck.is_blocked_verdict("partial_308") is True
        assert ConductorDedupCheck.is_blocked_verdict("partial_anything") is True

    def test_success_not_blocked(self):
        assert ConductorDedupCheck.is_blocked_verdict("throughput_improved") is False

    def test_infrastructure_hardened_not_blocked(self):
        assert ConductorDedupCheck.is_blocked_verdict("infrastructure_hardened") is False

    def test_empty_string_not_blocked(self):
        # Empty string is not in the blocked set and doesn't start with 'partial_'
        assert ConductorDedupCheck.is_blocked_verdict("") is False


# ---------------------------------------------------------------------------
# ConductorDedupCheck.is_complete — SCENARIO-INFRA-050
# ---------------------------------------------------------------------------


class TestIsComplete:
    """REQ-INFRA-042: is_complete returns True for genuine results, False otherwise."""

    def test_no_file_returns_false(self, tmp_path):
        # SCENARIO-INFRA-050: no result file → not complete
        check = ConductorDedupCheck(results_dir=str(tmp_path))
        assert check.is_complete(999) is False

    def test_valid_verdict_returns_true(self, tmp_path):
        # SCENARIO-INFRA-050: file with non-blocked verdict → complete
        _write_result(tmp_path, 447, "throughput_improved")
        check = ConductorDedupCheck(results_dir=str(tmp_path))
        assert check.is_complete(447) is True

    def test_deferred_verdict_returns_false(self, tmp_path):
        # SCENARIO-INFRA-050: deferred_to_gpu verdict → not complete
        _write_result(tmp_path, 447, "deferred_to_gpu")
        check = ConductorDedupCheck(results_dir=str(tmp_path))
        assert check.is_complete(447) is False

    def test_blocked_verdict_returns_false(self, tmp_path):
        _write_result(tmp_path, 100, "blocked")
        check = ConductorDedupCheck(results_dir=str(tmp_path))
        assert check.is_complete(100) is False

    def test_partial_verdict_returns_false(self, tmp_path):
        _write_result(tmp_path, 200, "partial_200")
        check = ConductorDedupCheck(results_dir=str(tmp_path))
        assert check.is_complete(200) is False

    def test_partial_json_file_excluded(self, tmp_path):
        # _partial.json files should not count as complete results
        partial = tmp_path / "experiment_300_partial.json"
        partial.write_text(json.dumps({"honest_verdict": "throughput_improved"}), encoding="utf-8")
        check = ConductorDedupCheck(results_dir=str(tmp_path))
        assert check.is_complete(300) is False

    def test_corrupt_file_returns_false(self, tmp_path):
        bad = tmp_path / "experiment_400.json"
        bad.write_text("not json", encoding="utf-8")
        check = ConductorDedupCheck(results_dir=str(tmp_path))
        assert check.is_complete(400) is False

    def test_missing_verdict_field_returns_false(self, tmp_path):
        no_verdict = tmp_path / "experiment_500.json"
        no_verdict.write_text(json.dumps({"experiment": 500}), encoding="utf-8")
        check = ConductorDedupCheck(results_dir=str(tmp_path))
        assert check.is_complete(500) is False

    def test_multiple_files_one_valid(self, tmp_path):
        # If multiple matches exist and at least one is valid, return True
        _write_result(tmp_path, 600, "blocked", suffix="_v1")
        _write_result(tmp_path, 600, "throughput_improved", suffix="_v2")
        check = ConductorDedupCheck(results_dir=str(tmp_path))
        assert check.is_complete(600) is True


# ---------------------------------------------------------------------------
# ConductorDedupCheck.should_skip
# ---------------------------------------------------------------------------


class TestShouldSkip:
    """REQ-INFRA-042: should_skip delegates to is_complete."""

    def test_skip_when_complete(self, tmp_path):
        _write_result(tmp_path, 447, "vram_gate_operational")
        check = ConductorDedupCheck(results_dir=str(tmp_path))
        assert check.should_skip(447) is True

    def test_no_skip_when_absent(self, tmp_path):
        check = ConductorDedupCheck(results_dir=str(tmp_path))
        assert check.should_skip(447) is False

    def test_no_skip_when_blocked(self, tmp_path):
        _write_result(tmp_path, 447, "deferred_to_gpu")
        check = ConductorDedupCheck(results_dir=str(tmp_path))
        assert check.should_skip(447) is False


# ---------------------------------------------------------------------------
# PartialResultHandoff.save — SCENARIO-INFRA-051
# ---------------------------------------------------------------------------


class TestPartialResultHandoffSave:
    """REQ-INFRA-043: save() writes atomic partial JSON."""

    def test_save_writes_file(self, tmp_path):
        handoff = PartialResultHandoff(results_dir=str(tmp_path))
        tmpl = _make_template(308, tmp_path)
        handoff.save(tmpl, {"done_count": 42})
        path = tmp_path / "experiment_308_partial.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["experiment"] == 308
        assert data["partial"] is True
        assert data["done_count"] == 42

    def test_save_sets_partial_verdict(self, tmp_path):
        handoff = PartialResultHandoff(results_dir=str(tmp_path))
        tmpl = _make_template(308, tmp_path)
        handoff.save(tmpl, {})
        data = json.loads((tmp_path / "experiment_308_partial.json").read_text())
        assert data["honest_verdict"] == "partial_308"

    def test_save_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "deep" / "results"
        handoff = PartialResultHandoff(results_dir=str(nested))
        tmpl = _make_template(1, tmp_path)
        handoff.save(tmpl, {"x": 1})
        assert (nested / "experiment_1_partial.json").exists()

    def test_save_overwrites_existing(self, tmp_path):
        handoff = PartialResultHandoff(results_dir=str(tmp_path))
        tmpl = _make_template(308, tmp_path)
        handoff.save(tmpl, {"step": 1})
        handoff.save(tmpl, {"step": 2})
        data = json.loads((tmp_path / "experiment_308_partial.json").read_text())
        assert data["step"] == 2


# ---------------------------------------------------------------------------
# PartialResultHandoff.resume_if_available — SCENARIO-INFRA-052
# ---------------------------------------------------------------------------


class TestPartialResultHandoffResume:
    """REQ-INFRA-044: resume_if_available returns saved state or None."""

    def test_returns_none_when_no_partial(self, tmp_path):
        # SCENARIO-INFRA-052: no partial file → None
        handoff = PartialResultHandoff(results_dir=str(tmp_path))
        tmpl = _make_template(308, tmp_path)
        assert handoff.resume_if_available(tmpl) is None

    def test_returns_partial_state_when_present(self, tmp_path):
        handoff = PartialResultHandoff(results_dir=str(tmp_path))
        tmpl = _make_template(308, tmp_path)
        handoff.save(tmpl, {"done_count": 55})
        result = handoff.resume_if_available(tmpl)
        assert result is not None
        assert result["done_count"] == 55
        assert result["experiment"] == 308

    def test_returns_none_on_corrupt_partial(self, tmp_path):
        bad = tmp_path / "experiment_308_partial.json"
        bad.write_text("corrupt{", encoding="utf-8")
        handoff = PartialResultHandoff(results_dir=str(tmp_path))
        tmpl = _make_template(308, tmp_path)
        assert handoff.resume_if_available(tmpl) is None


# ---------------------------------------------------------------------------
# PartialResultHandoff.install — SCENARIO-INFRA-051
# ---------------------------------------------------------------------------


class TestPartialResultHandoffInstall:
    """REQ-INFRA-043: install() registers atexit and SIGTERM handlers."""

    def test_install_registers_atexit(self, tmp_path):
        import atexit as atexit_mod

        handoff = PartialResultHandoff(results_dir=str(tmp_path))
        tmpl = _make_template(308, tmp_path)

        with patch.object(atexit_mod, "register") as mock_reg, \
             patch("signal.signal"):
            handoff.install(tmpl)
            mock_reg.assert_called_once_with(handoff._atexit_handler)

    def test_install_registers_sigterm(self, tmp_path):
        handoff = PartialResultHandoff(results_dir=str(tmp_path))
        tmpl = _make_template(308, tmp_path)

        with patch("atexit.register"), \
             patch("signal.signal") as mock_sig:
            handoff.install(tmpl)
            mock_sig.assert_called_once_with(signal.SIGTERM, handoff._sigterm_handler)

    def test_atexit_handler_fires_save(self, tmp_path):
        handoff = PartialResultHandoff(results_dir=str(tmp_path))
        tmpl = _make_template(99, tmp_path)
        handoff._active_template = tmpl
        handoff._active_partial_state = {"key": "value"}
        handoff._atexit_handler()
        assert (tmp_path / "experiment_99_partial.json").exists()

    def test_atexit_handler_noop_when_no_template(self, tmp_path):
        # Should not raise when no template is installed
        handoff = PartialResultHandoff(results_dir=str(tmp_path))
        handoff._atexit_handler()  # Should not raise

    def test_atexit_handler_noop_when_no_state(self, tmp_path):
        # Should not raise when no partial state is set
        handoff = PartialResultHandoff(results_dir=str(tmp_path))
        tmpl = _make_template(99, tmp_path)
        handoff._active_template = tmpl
        # _active_partial_state is None
        handoff._atexit_handler()  # Should not raise

    def test_sigterm_handler_fires_save_and_re_raises(self, tmp_path):
        handoff = PartialResultHandoff(results_dir=str(tmp_path))
        tmpl = _make_template(99, tmp_path)
        handoff._active_template = tmpl
        handoff._active_partial_state = {"key": "value"}

        with patch("signal.signal"), patch("os.kill") as mock_kill:
            handoff._sigterm_handler(signal.SIGTERM, None)
            mock_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)
        assert (tmp_path / "experiment_99_partial.json").exists()

    def test_update_partial_state(self, tmp_path):
        handoff = PartialResultHandoff(results_dir=str(tmp_path))
        handoff.update_partial_state({"step": 5})
        assert handoff._active_partial_state == {"step": 5}
