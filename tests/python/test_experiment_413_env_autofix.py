"""Tests for scripts/experiment_413_env_autofix.py — 100% targeted coverage.

Coverage targets
----------------
- main():
  - apply_env_autofix() called before ExperimentTemplate
  - run_gpu_preflight() result incorporated
  - artifact written to results/experiment_413_env_autofix.json
  - honest_verdict and retro_022_resolved present in artifact
  - ACTION REQUIRED printed when verdict not in ('gpu_confirmed_live', 'auto_fix_applied')
  - ACTION REQUIRED NOT printed when verdict is 'gpu_confirmed_live' or 'auto_fix_applied'
  - Prior preflight missing handled gracefully
- _load_prior_preflight():
  - file exists → returns dict
  - file missing → returns note dict
  - file unreadable/corrupt → returns note dict
- _count_corrupt_files():
  - delegates to DeliverableContentValidator.audit_known_corrupt_files
- _print_action_required():
  - prints cloud GPU options (Lambda, vast.ai, RunPod)

Spec: REQ-INFRA-021, REQ-INFRA-022,
      SCENARIO-INFRA-025, SCENARIO-INFRA-026, SCENARIO-INFRA-027
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_413_env_autofix as _mod  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_autofix_result(
    gpu_detected: bool = False,
    carnot_force_live_was_set: bool = False,
    auto_fix_applied: bool = False,
    final_env_value: str | None = None,
) -> MagicMock:
    r = MagicMock()
    r.gpu_detected = gpu_detected
    r.carnot_force_live_was_set = carnot_force_live_was_set
    r.auto_fix_applied = auto_fix_applied
    r.final_env_value = final_env_value
    return r


def _make_preflight_result(
    honest_verdict: str = "env_not_propagating",
    env_var_set: bool = False,
    subprocess_inherits_env: bool = False,
    is_live_capable: bool = False,
    smoke_test_passed: bool = False,
    retro_019_resolved: bool = False,
) -> MagicMock:
    r = MagicMock()
    r.honest_verdict = honest_verdict
    r.env_var_set = env_var_set
    r.subprocess_inherits_env = subprocess_inherits_env
    r.is_live_capable = is_live_capable
    r.smoke_test_passed = smoke_test_passed
    r.retro_019_resolved = retro_019_resolved
    return r


def _make_env_autofix_art(
    honest_verdict: str = "gpu_not_detected",
    retro_022_resolved: bool = False,
) -> dict:
    return {
        "schema": "carnot.env_autofix.v1",
        "honest_verdict": honest_verdict,
        "retro_022_resolved": retro_022_resolved,
        "gpu_detected": False,
        "carnot_force_live_was_set": False,
        "auto_fix_applied": False,
        "final_env_value": None,
        "preflight": {},
    }


# ---------------------------------------------------------------------------
# _load_prior_preflight tests
# ---------------------------------------------------------------------------


class TestLoadPriorPreflight:
    def test_file_exists_returns_dict(self, tmp_path):
        p = tmp_path / "preflight.json"
        p.write_text(json.dumps({"honest_verdict": "env_not_propagating"}))
        result = _mod._load_prior_preflight(p)
        assert result["honest_verdict"] == "env_not_propagating"

    def test_file_missing_returns_note(self, tmp_path):
        p = tmp_path / "missing.json"
        result = _mod._load_prior_preflight(p)
        assert "note" in result
        assert "missing" in result["note"]

    def test_file_corrupt_returns_note(self, tmp_path):
        p = tmp_path / "corrupt.json"
        p.write_text("not json {{{{")
        result = _mod._load_prior_preflight(p)
        assert "note" in result
        assert "unreadable" in result["note"]


# ---------------------------------------------------------------------------
# _count_corrupt_files tests
# ---------------------------------------------------------------------------


class TestCountCorruptFiles:
    def test_counts_corrupt_json_entries(self, tmp_path):
        audit = {
            "a.py": "corrupt_json",
            "b.py": "corrupt_json",
            "c.py": "valid_python",
            "d.py": "missing",
        }
        with patch.object(
            _mod.DeliverableContentValidator,
            "audit_known_corrupt_files",
            return_value=audit,
        ):
            count = _mod._count_corrupt_files(tmp_path)
        assert count == 2

    def test_zero_when_all_valid(self, tmp_path):
        audit = {"a.py": "valid_python", "b.py": "valid_python"}
        with patch.object(
            _mod.DeliverableContentValidator,
            "audit_known_corrupt_files",
            return_value=audit,
        ):
            count = _mod._count_corrupt_files(tmp_path)
        assert count == 0


# ---------------------------------------------------------------------------
# _print_action_required tests
# ---------------------------------------------------------------------------


class TestPrintActionRequired:
    def test_prints_cloud_options(self, capsys):
        _mod._print_action_required("gpu_not_detected")
        out = capsys.readouterr().out
        assert "Lambda" in out
        assert "vast.ai" in out
        assert "RunPod" in out
        assert "ACTION REQUIRED" in out

    def test_includes_verdict_in_output(self, capsys):
        _mod._print_action_required("my_verdict")
        out = capsys.readouterr().out
        assert "my_verdict" in out


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for main() covering all branch paths."""

    def _patch_all(self, tmp_path, autofix_result, preflight_result, env_autofix_art):
        """Return a dict of patches for clean main() execution."""
        return {
            "apply_env_autofix": patch.object(
                _mod, "apply_env_autofix", return_value=autofix_result
            ),
            "run_gpu_preflight": patch.object(
                _mod, "run_gpu_preflight", return_value=preflight_result
            ),
            "build_env_autofix_artifact": patch.object(
                _mod, "build_env_autofix_artifact", return_value=env_autofix_art
            ),
            "repo_root": patch.object(_mod, "_REPO_ROOT", tmp_path),
            "prior_path": patch.object(
                _mod,
                "_PRIOR_PREFLIGHT_PATH",
                tmp_path / "results" / "experiment_404_preflight_v2.json",
            ),
        }

    def _run_main(self, patches: dict) -> dict:
        ctx_managers = list(patches.values())
        # Enter all context managers
        mocks = []
        for cm in ctx_managers:
            mocks.append(cm.__enter__(cm))
        try:
            result = _mod.main()
        finally:
            for cm in ctx_managers:
                cm.__exit__(cm, None, None, None)
        return result

    def test_main_gpu_not_detected_writes_artifact(self, tmp_path, capsys):
        autofix = _make_autofix_result()
        preflight = _make_preflight_result()
        art = _make_env_autofix_art("gpu_not_detected", False)
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(_mod, "apply_env_autofix", return_value=autofix),
            patch.object(_mod, "run_gpu_preflight", return_value=preflight),
            patch.object(_mod, "build_env_autofix_artifact", return_value=art),
            patch.object(_mod, "_REPO_ROOT", tmp_path),
            patch.object(_mod, "_PRIOR_PREFLIGHT_PATH", tmp_path / "results" / "x.json"),
            patch.object(_mod, "_count_corrupt_files", return_value=5),
        ):
            result = _mod.main()

        assert result["honest_verdict"] == "gpu_not_detected"
        assert result["retro_022_resolved"] is False
        # artifact file written
        out_file = tmp_path / "results" / "experiment_413_env_autofix.json"
        assert out_file.exists()
        on_disk = json.loads(out_file.read_text())
        assert on_disk["honest_verdict"] == "gpu_not_detected"

    def test_main_action_required_printed_when_not_live(self, tmp_path, capsys):
        autofix = _make_autofix_result()
        preflight = _make_preflight_result()
        art = _make_env_autofix_art("gpu_not_detected", False)
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(_mod, "apply_env_autofix", return_value=autofix),
            patch.object(_mod, "run_gpu_preflight", return_value=preflight),
            patch.object(_mod, "build_env_autofix_artifact", return_value=art),
            patch.object(_mod, "_REPO_ROOT", tmp_path),
            patch.object(_mod, "_PRIOR_PREFLIGHT_PATH", tmp_path / "results" / "x.json"),
            patch.object(_mod, "_count_corrupt_files", return_value=5),
        ):
            _mod.main()
        out = capsys.readouterr().out
        assert "ACTION REQUIRED" in out

    def test_main_no_action_required_when_auto_fix_applied(self, tmp_path, capsys):
        autofix = _make_autofix_result(gpu_detected=True, auto_fix_applied=True, final_env_value="1")
        preflight = _make_preflight_result(honest_verdict="env_not_propagating")
        art = _make_env_autofix_art("auto_fix_applied", True)
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(_mod, "apply_env_autofix", return_value=autofix),
            patch.object(_mod, "run_gpu_preflight", return_value=preflight),
            patch.object(_mod, "build_env_autofix_artifact", return_value=art),
            patch.object(_mod, "_REPO_ROOT", tmp_path),
            patch.object(_mod, "_PRIOR_PREFLIGHT_PATH", tmp_path / "results" / "x.json"),
            patch.object(_mod, "_count_corrupt_files", return_value=5),
        ):
            _mod.main()
        out = capsys.readouterr().out
        assert "ACTION REQUIRED" not in out

    def test_main_no_action_required_when_gpu_confirmed_live(self, tmp_path, capsys):
        autofix = _make_autofix_result(
            gpu_detected=True, carnot_force_live_was_set=True, final_env_value="1"
        )
        preflight = _make_preflight_result(honest_verdict="gpu_confirmed_live")
        art = _make_env_autofix_art("gpu_confirmed_live", True)
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(_mod, "apply_env_autofix", return_value=autofix),
            patch.object(_mod, "run_gpu_preflight", return_value=preflight),
            patch.object(_mod, "build_env_autofix_artifact", return_value=art),
            patch.object(_mod, "_REPO_ROOT", tmp_path),
            patch.object(_mod, "_PRIOR_PREFLIGHT_PATH", tmp_path / "results" / "x.json"),
            patch.object(_mod, "_count_corrupt_files", return_value=0),
        ):
            _mod.main()
        out = capsys.readouterr().out
        assert "ACTION REQUIRED" not in out

    def test_main_retro_022_resolved_in_artifact(self, tmp_path, capsys):
        autofix = _make_autofix_result(gpu_detected=True, auto_fix_applied=True, final_env_value="1")
        preflight = _make_preflight_result()
        art = _make_env_autofix_art("auto_fix_applied", True)
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(_mod, "apply_env_autofix", return_value=autofix),
            patch.object(_mod, "run_gpu_preflight", return_value=preflight),
            patch.object(_mod, "build_env_autofix_artifact", return_value=art),
            patch.object(_mod, "_REPO_ROOT", tmp_path),
            patch.object(_mod, "_PRIOR_PREFLIGHT_PATH", tmp_path / "results" / "x.json"),
            patch.object(_mod, "_count_corrupt_files", return_value=3),
        ):
            result = _mod.main()
        assert result["retro_022_resolved"] is True
        assert result["auto_fix_applied"] is True
        assert result["n_corrupt_files_remaining"] == 3

    def test_main_prior_preflight_missing_handled(self, tmp_path, capsys):
        autofix = _make_autofix_result()
        preflight = _make_preflight_result()
        art = _make_env_autofix_art("gpu_not_detected", False)
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)
        # do NOT create prior preflight file

        with (
            patch.object(_mod, "apply_env_autofix", return_value=autofix),
            patch.object(_mod, "run_gpu_preflight", return_value=preflight),
            patch.object(_mod, "build_env_autofix_artifact", return_value=art),
            patch.object(_mod, "_REPO_ROOT", tmp_path),
            patch.object(
                _mod,
                "_PRIOR_PREFLIGHT_PATH",
                tmp_path / "results" / "nonexistent.json",
            ),
            patch.object(_mod, "_count_corrupt_files", return_value=5),
        ):
            result = _mod.main()
        # Should not crash; artifact should still be written
        assert result["honest_verdict"] == "gpu_not_detected"

    def test_main_artifact_has_required_fields(self, tmp_path):
        autofix = _make_autofix_result()
        preflight = _make_preflight_result()
        art = _make_env_autofix_art("gpu_not_detected", False)
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(_mod, "apply_env_autofix", return_value=autofix),
            patch.object(_mod, "run_gpu_preflight", return_value=preflight),
            patch.object(_mod, "build_env_autofix_artifact", return_value=art),
            patch.object(_mod, "_REPO_ROOT", tmp_path),
            patch.object(_mod, "_PRIOR_PREFLIGHT_PATH", tmp_path / "results" / "x.json"),
            patch.object(_mod, "_count_corrupt_files", return_value=5),
        ):
            result = _mod.main()

        required = {
            "experiment", "title", "run_date", "started_at", "finished_at",
            "duration_s", "status", "honest_verdict", "retro_022_resolved",
            "auto_fix_applied", "gpu_detected", "n_corrupt_files_remaining",
        }
        assert required.issubset(result.keys())
        assert result["experiment"] == 413

    def test_main_apply_env_autofix_called_first(self, tmp_path):
        """apply_env_autofix must be called (mocked call order tracked)."""
        call_order = []
        autofix = _make_autofix_result()
        preflight = _make_preflight_result()
        art = _make_env_autofix_art()
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)

        def _track_autofix():
            call_order.append("autofix")
            return autofix

        def _track_preflight(*a, **kw):
            call_order.append("preflight")
            return preflight

        with (
            patch.object(_mod, "apply_env_autofix", side_effect=_track_autofix),
            patch.object(_mod, "run_gpu_preflight", side_effect=_track_preflight),
            patch.object(_mod, "build_env_autofix_artifact", return_value=art),
            patch.object(_mod, "_REPO_ROOT", tmp_path),
            patch.object(_mod, "_PRIOR_PREFLIGHT_PATH", tmp_path / "results" / "x.json"),
            patch.object(_mod, "_count_corrupt_files", return_value=0),
        ):
            _mod.main()

        assert call_order[0] == "autofix", "apply_env_autofix must be called first"
