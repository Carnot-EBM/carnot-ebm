"""Tests for scripts/experiment_547_legacy_modernization.py.

Covers:
- audit_script: returns correct markers for a real script, missing script
- classify_modernization: all tiers (fully_modern, mostly_modern, partial, legacy, missing)
- estimate_savings_pct: correct fraction, zero when no watchdog/env_autofix
- main(): writes artifact with all required schema fields and correct honest_verdict

Spec: REQ-INFRA-007, REQ-INFRA-023, REQ-INFRA-073
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_547_legacy_modernization import (
    audit_script,
    classify_modernization,
    estimate_savings_pct,
    _AUDIT_TARGETS,
)


# ---------------------------------------------------------------------------
# audit_script
# ---------------------------------------------------------------------------


class TestAuditScript:
    """audit_script must detect presence/absence of modernization markers."""

    def test_missing_script_returns_all_false(self, tmp_path: Path) -> None:
        result = audit_script(tmp_path / "nonexistent.py")
        assert result["script_exists"] is False
        assert result["env_autofix"] is False
        assert result["watchdog"] is False
        assert result["template"] is False
        assert result["batched_runner"] is False
        assert result["assert_deliverable"] is False

    def test_fully_modern_script_detected(self, tmp_path: Path) -> None:
        script = tmp_path / "modern.py"
        script.write_text(
            "apply_env_autofix()\n"
            "ExperimentTimeoutWatchdog(1, 2, 'x')\n"
            "ExperimentTemplate(1, 'x', 'y')\n"
            "BatchedInferenceRunner(fn, batch_size=8)\n"
            "tmpl.assert_deliverable_written()\n"
        )
        result = audit_script(script)
        assert result["script_exists"] is True
        assert result["env_autofix"] is True
        assert result["watchdog"] is True
        assert result["template"] is True
        assert result["batched_runner"] is True
        assert result["assert_deliverable"] is True

    def test_legacy_script_all_false(self, tmp_path: Path) -> None:
        script = tmp_path / "legacy.py"
        script.write_text("import json\nresult = {}\n")
        result = audit_script(script)
        assert result["script_exists"] is True
        assert result["env_autofix"] is False
        assert result["watchdog"] is False
        assert result["template"] is False
        assert result["batched_runner"] is False
        assert result["assert_deliverable"] is False


# ---------------------------------------------------------------------------
# classify_modernization
# ---------------------------------------------------------------------------


class TestClassifyModernization:
    """classify_modernization must return correct tier based on marker count."""

    def test_missing_script(self) -> None:
        audit = {"script_exists": False, "env_autofix": False, "watchdog": False,
                 "template": False, "batched_runner": False, "assert_deliverable": False}
        assert classify_modernization(audit) == "missing"

    def test_fully_modern_all_four(self) -> None:
        audit = {"script_exists": True, "env_autofix": True, "watchdog": True,
                 "template": True, "batched_runner": False, "assert_deliverable": True}
        assert classify_modernization(audit) == "fully_modern"

    def test_mostly_modern_three(self) -> None:
        audit = {"script_exists": True, "env_autofix": True, "watchdog": True,
                 "template": False, "batched_runner": False, "assert_deliverable": True}
        assert classify_modernization(audit) == "mostly_modern"

    def test_partial_two(self) -> None:
        audit = {"script_exists": True, "env_autofix": True, "watchdog": True,
                 "template": False, "batched_runner": False, "assert_deliverable": False}
        assert classify_modernization(audit) == "partial"

    def test_legacy_zero(self) -> None:
        audit = {"script_exists": True, "env_autofix": False, "watchdog": False,
                 "template": False, "batched_runner": False, "assert_deliverable": False}
        assert classify_modernization(audit) == "legacy"


# ---------------------------------------------------------------------------
# estimate_savings_pct
# ---------------------------------------------------------------------------


class TestEstimateSavingsPct:
    """estimate_savings_pct must compute correct fraction from overhead data."""

    def test_all_watchdog_saves_20pct(self) -> None:
        targets = [{"overhead_min": 100}, {"overhead_min": 100}]
        audits = [
            {"env_autofix": True, "watchdog": True},
            {"env_autofix": True, "watchdog": True},
        ]
        pct = estimate_savings_pct(targets, audits)
        assert abs(pct - 0.20) < 1e-6

    def test_none_watchdog_zero(self) -> None:
        targets = [{"overhead_min": 100}, {"overhead_min": 100}]
        audits = [
            {"env_autofix": False, "watchdog": False},
            {"env_autofix": False, "watchdog": False},
        ]
        pct = estimate_savings_pct(targets, audits)
        assert pct == 0.0

    def test_empty_targets_zero(self) -> None:
        pct = estimate_savings_pct([], [])
        assert pct == 0.0


# ---------------------------------------------------------------------------
# main() integration — artifact schema
# ---------------------------------------------------------------------------


class TestMain:
    """main() must write a valid artifact with all required schema fields."""

    def test_main_writes_valid_artifact(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from scripts.experiment_547_legacy_modernization import main
        from scripts.experiment_template import ExperimentTemplate

        monkeypatch.chdir(tmp_path)

        real_cls = ExperimentTemplate

        class FastTemplate(real_cls):
            def setup(self) -> None:
                (tmp_path / "results").mkdir(parents=True, exist_ok=True)
                (tmp_path / "results" / "checkpoints" / "experiment_547").mkdir(
                    parents=True, exist_ok=True
                )
                self.checkpoint = None
                import time as _t
                self._t0 = _t.perf_counter()

        monkeypatch.setattr(
            "scripts.experiment_547_legacy_modernization.ExperimentTemplate", FastTemplate
        )
        monkeypatch.setattr(
            "scripts.experiment_547_legacy_modernization._REPO_ROOT", tmp_path
        )

        # Copy the real scripts into tmp_path so audit_script can find them
        import shutil
        (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
        for target in _AUDIT_TARGETS:
            real_path = _REPO_ROOT / target["script"]
            if real_path.exists():
                shutil.copy(real_path, tmp_path / target["script"])

        with patch("scripts.experiment_547_legacy_modernization.ExperimentTimeoutWatchdog") as MockWD:
            MockWD.return_value.start.return_value = None
            MockWD.return_value.stop.return_value = None
            main()

        out = tmp_path / "results" / "experiment_547_legacy_modernization.json"
        assert out.exists(), "Artifact not written"
        data = json.loads(out.read_text())

        # Required fields from REQUIRED_RESULT_FIELDS
        for field in ["experiment", "schema", "run_date", "started_at", "finished_at",
                      "duration_s", "status", "title"]:
            assert field in data, f"Missing required field: {field}"

        # Exp 547-specific fields
        assert data["artifact_schema"] == "carnot.legacy_modernization.v1"
        assert data["scripts_audited"] == ["exp308", "exp260", "exp309", "exp425", "exp410"]
        assert "honest_verdict" in data
        assert data["honest_verdict"] in ("sprint_complete", "partial_sprint", "audit_only")
        assert "estimated_savings_pct" in data
        assert isinstance(data["estimated_savings_pct"], float)
        assert "batching_added" in data
        assert "teardown_added" in data
        assert data["status"] == "success"

    def test_artifact_honest_verdict_sprint_complete_when_all_modernized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When all 5 scripts have watchdog+template+env_autofix+assert, verdict is sprint_complete."""
        from scripts.experiment_547_legacy_modernization import main
        from scripts.experiment_template import ExperimentTemplate

        class FastTemplate(ExperimentTemplate):
            def setup(self) -> None:
                (tmp_path / "results").mkdir(parents=True, exist_ok=True)
                (tmp_path / "results" / "checkpoints" / "experiment_547").mkdir(
                    parents=True, exist_ok=True
                )
                self.checkpoint = None
                import time as _t
                self._t0 = _t.perf_counter()

        monkeypatch.setattr(
            "scripts.experiment_547_legacy_modernization.ExperimentTemplate", FastTemplate
        )
        monkeypatch.setattr(
            "scripts.experiment_547_legacy_modernization._REPO_ROOT", tmp_path
        )

        # Create fake "fully modern" scripts in tmp_path
        (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
        modern_content = (
            "apply_env_autofix()\n"
            "ExperimentTimeoutWatchdog(1, 2, 'x')\n"
            "ExperimentTemplate(1, 'x', 'y')\n"
            "tmpl.assert_deliverable_written()\n"
        )
        for target in _AUDIT_TARGETS:
            (tmp_path / target["script"]).write_text(modern_content)

        with patch("scripts.experiment_547_legacy_modernization.ExperimentTimeoutWatchdog") as MockWD:
            MockWD.return_value.start.return_value = None
            MockWD.return_value.stop.return_value = None
            main()

        out = tmp_path / "results" / "experiment_547_legacy_modernization.json"
        data = json.loads(out.read_text())
        assert data["honest_verdict"] == "sprint_complete"
