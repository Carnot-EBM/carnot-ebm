"""Tests for scripts/experiment_425_conductor_timeout.py.

100% coverage for the experiment script's logic: synthetic workload,
watchdog integration, artifact schema.

Spec: REQ-INFRA-023, REQ-INFRA-024,
      SCENARIO-INFRA-028, SCENARIO-INFRA-029, SCENARIO-INFRA-030
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import helpers — resolve the scripts package
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import the module under test (after path setup)
import importlib
import types


def _load_exp425():
    """Load the experiment_425 module without executing main()."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "experiment_425",
        _REPO_ROOT / "scripts" / "experiment_425_conductor_timeout.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # The module calls apply_env_autofix() at import time (top-level).
    # Patch it to a no-op so import doesn't mutate os.environ in tests.
    with patch("python.carnot.pipeline.env_autofix.apply_env_autofix", return_value=None):
        spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic workload
# ---------------------------------------------------------------------------


class TestSyntheticConstraintCheck:
    """Tests for _run_synthetic_constraint_check()."""

    def test_returns_dict_with_required_keys(self):
        mod = _load_exp425()
        result = mod._run_synthetic_constraint_check.__wrapped__(0) if hasattr(
            mod._run_synthetic_constraint_check, "__wrapped__"
        ) else None

        # Call directly with a patched sleep so tests are fast
        with patch("time.sleep"):
            result = mod._run_synthetic_constraint_check(0)

        assert "check_id" in result
        assert "energy" in result
        assert "satisfied" in result

    def test_energy_minimum_at_five(self):
        mod = _load_exp425()
        with patch("time.sleep"):
            r = mod._run_synthetic_constraint_check(5)
        assert r["energy"] == 0
        assert r["satisfied"] is True

    def test_nonzero_energy_at_other_values(self):
        mod = _load_exp425()
        with patch("time.sleep"):
            r = mod._run_synthetic_constraint_check(0)
        assert r["energy"] == 5
        assert r["satisfied"] is False

    def test_check_id_matches_input(self):
        mod = _load_exp425()
        with patch("time.sleep"):
            r = mod._run_synthetic_constraint_check(7)
        assert r["check_id"] == 7


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------


class TestMain:
    """Test that main() produces a valid result artifact."""

    def test_main_writes_artifact(self, tmp_path, monkeypatch):
        """main() must write a JSON artifact with all required fields."""
        mod = _load_exp425()

        # Redirect the deliverable to a temp directory
        deliverable = "results/experiment_425_conductor_timeout.json"
        output_file = tmp_path / "experiment_425_conductor_timeout.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Patch repo root so output goes to tmp_path
        monkeypatch.setattr(mod, "_REPO_ROOT", tmp_path)

        # Patch ExperimentTemplate to avoid creating dirs in the real repo
        real_tmpl_cls = mod.ExperimentTemplate

        class FastTemplate(real_tmpl_cls):
            def setup(self):
                # create the output directory in tmp_path
                (tmp_path / "results").mkdir(parents=True, exist_ok=True)
                (tmp_path / "results" / "checkpoints" / "experiment_425").mkdir(
                    parents=True, exist_ok=True
                )
                self.checkpoint = None
                import time as _t
                self._t0 = _t.perf_counter()

        monkeypatch.setattr(mod, "ExperimentTemplate", FastTemplate)

        # Patch sleep so the test is fast (no actual 10s delay)
        with patch("time.sleep"):
            mod.main()

        # Check the artifact exists
        result_path = tmp_path / deliverable
        assert result_path.exists(), f"Artifact not written at {result_path}"

        data = json.loads(result_path.read_text())

        # Required fields
        assert data["honest_verdict"] == "watchdog_implemented"
        assert data["retro_003_resolved"] is True
        assert data["demo_timed_out"] is False
        assert data["estimated_savings_minutes_per_runaway"] == 99
        assert data["status"] == "success"
        assert len(data["demo_checks"]) == 10
        assert data["artifact_schema"] == "carnot.timeout_watchdog.v1"

    def test_main_demo_checks_have_correct_structure(self, tmp_path, monkeypatch):
        """All 10 synthetic checks must have check_id, energy, satisfied."""
        mod = _load_exp425()
        monkeypatch.setattr(mod, "_REPO_ROOT", tmp_path)

        real_tmpl_cls = mod.ExperimentTemplate

        class FastTemplate(real_tmpl_cls):
            def setup(self):
                (tmp_path / "results").mkdir(parents=True, exist_ok=True)
                (tmp_path / "results" / "checkpoints" / "experiment_425").mkdir(
                    parents=True, exist_ok=True
                )
                self.checkpoint = None
                import time as _t
                self._t0 = _t.perf_counter()

        monkeypatch.setattr(mod, "ExperimentTemplate", FastTemplate)

        with patch("time.sleep"):
            mod.main()

        result_path = tmp_path / "results" / "experiment_425_conductor_timeout.json"
        data = json.loads(result_path.read_text())

        for check in data["demo_checks"]:
            assert "check_id" in check
            assert "energy" in check
            assert "satisfied" in check

    def test_main_watchdog_artifact_embedded(self, tmp_path, monkeypatch):
        """Artifact must embed a watchdog_artifact sub-dict."""
        mod = _load_exp425()
        monkeypatch.setattr(mod, "_REPO_ROOT", tmp_path)

        real_tmpl_cls = mod.ExperimentTemplate

        class FastTemplate(real_tmpl_cls):
            def setup(self):
                (tmp_path / "results").mkdir(parents=True, exist_ok=True)
                (tmp_path / "results" / "checkpoints" / "experiment_425").mkdir(
                    parents=True, exist_ok=True
                )
                self.checkpoint = None
                import time as _t
                self._t0 = _t.perf_counter()

        monkeypatch.setattr(mod, "ExperimentTemplate", FastTemplate)

        with patch("time.sleep"):
            mod.main()

        result_path = tmp_path / "results" / "experiment_425_conductor_timeout.json"
        data = json.loads(result_path.read_text())

        wa = data["watchdog_artifact"]
        assert wa["schema"] == "carnot.timeout_watchdog.v1"
        assert wa["retro_003_resolved"] is True
        assert wa["estimated_savings_minutes_per_runaway"] == 99
