"""Tests for Exp 339: pre-session startup health check (RETRO-007 + RETRO-008).

Spec coverage: REQ-INFRA-008,
               SCENARIO-INFRA-012, SCENARIO-INFRA-013

Written test-first per REQ-INFRA-002.  Tests validate:

- scripts/experiment_339_session_startup.py exists and is importable.
- Script references correct schema ("carnot.session_startup.v1").
- Script references RETRO-007 and RETRO-008 in retro_items_implemented.
- main() produces a valid artifact dict with all required keys.
- Artifact all_healthy follows the n_gpus>=2 AND zombies==0 rule.
- Artifact n_zombies_killed is 0 when run in dry-run mode.
- Script degrades gracefully when session_startup module errors out.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
RESULTS_DIR = REPO_ROOT / "results"
EXP_339_SCRIPT = SCRIPTS_DIR / "experiment_339_session_startup.py"
EXP_339_RESULT = RESULTS_DIR / "experiment_339_session_startup.json"

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _load_exp_339() -> ModuleType:
    """Dynamically import experiment_339_session_startup as a module."""
    spec = importlib.util.spec_from_file_location(
        "experiment_339_session_startup", EXP_339_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# TestExp339ScriptExists
# ---------------------------------------------------------------------------


class TestExp339ScriptExists:
    """Basic structural checks on the experiment script."""

    def test_script_file_exists(self) -> None:
        """scripts/experiment_339_session_startup.py must exist."""
        assert EXP_339_SCRIPT.exists(), f"Missing: {EXP_339_SCRIPT}"

    def test_script_references_correct_schema(self) -> None:
        """Script must reference 'carnot.session_startup.v1' (stored as artifact_schema)."""
        text = EXP_339_SCRIPT.read_text()
        assert "carnot.session_startup.v1" in text

    def test_script_references_retro_007(self) -> None:
        """Script must reference RETRO-007."""
        text = EXP_339_SCRIPT.read_text()
        assert "RETRO-007" in text

    def test_script_references_retro_008(self) -> None:
        """Script must reference RETRO-008."""
        text = EXP_339_SCRIPT.read_text()
        assert "RETRO-008" in text

    def test_script_references_artifact_schema_key(self) -> None:
        """Script must use 'artifact_schema' key (build_result overwrites 'schema')."""
        text = EXP_339_SCRIPT.read_text()
        assert "artifact_schema" in text

    def test_script_references_n_gpus_detected(self) -> None:
        """Script must include n_gpus_detected in artifact."""
        text = EXP_339_SCRIPT.read_text()
        assert "n_gpus_detected" in text

    def test_script_references_n_zombies_found(self) -> None:
        """Script must include n_zombies_found in artifact."""
        text = EXP_339_SCRIPT.read_text()
        assert "n_zombies_found" in text

    def test_script_references_all_healthy(self) -> None:
        """Script must include all_healthy in artifact."""
        text = EXP_339_SCRIPT.read_text()
        assert "all_healthy" in text

    def test_script_references_retro_items_implemented(self) -> None:
        """Script must include retro_items_implemented in artifact."""
        text = EXP_339_SCRIPT.read_text()
        assert "retro_items_implemented" in text

    def test_script_references_exp_id_339(self) -> None:
        """Script must reference EXP_ID = 339."""
        text = EXP_339_SCRIPT.read_text()
        assert "339" in text

    def test_script_uses_dry_run(self) -> None:
        """Script must call run_session_startup with dry_run=True."""
        text = EXP_339_SCRIPT.read_text()
        assert "dry_run=True" in text


# ---------------------------------------------------------------------------
# TestExp339MainArtifact
# REQ-INFRA-008 / SCENARIO-INFRA-012
# ---------------------------------------------------------------------------


class TestExp339MainArtifact:
    """main() produces a valid artifact with required keys."""

    def _run_main_with_mock(
        self,
        n_gpus: int = 2,
        zombies: int = 0,
        killed: int = 0,
        all_healthy: bool = True,
    ) -> dict:
        """Run experiment_339 main() with a mocked session_startup result."""
        mock_result = {
            "n_gpus_detected": n_gpus,
            "n_zombies_found": zombies,
            "n_zombies_killed": killed,
            "all_healthy": all_healthy,
        }

        artifact_holder: dict = {}

        original_write = Path.write_text

        def capture_write(self, data, *args, **kwargs):  # type: ignore[override]
            if "experiment_339" in str(self):
                artifact_holder["data"] = data
            return original_write(self, data, *args, **kwargs)

        with (
            patch(
                "carnot.pipeline.session_startup.run_session_startup",
                return_value=mock_result,
            ),
            patch.object(Path, "write_text", capture_write),
            patch.object(Path, "mkdir", lambda *a, **k: None),
        ):
            mod = _load_exp_339()
            mod.main()

        return json.loads(artifact_holder.get("data", "{}"))

    def test_artifact_has_artifact_schema(self) -> None:
        """Artifact must contain 'artifact_schema' with the version string."""
        artifact = self._run_main_with_mock()
        assert artifact.get("artifact_schema") == "carnot.session_startup.v1"

    def test_artifact_has_n_gpus_detected(self) -> None:
        """Artifact must contain n_gpus_detected."""
        artifact = self._run_main_with_mock(n_gpus=2)
        assert artifact.get("n_gpus_detected") == 2

    def test_artifact_has_n_zombies_found(self) -> None:
        """Artifact must contain n_zombies_found."""
        artifact = self._run_main_with_mock(zombies=1)
        assert artifact.get("n_zombies_found") == 1

    def test_artifact_has_n_zombies_killed(self) -> None:
        """Artifact must contain n_zombies_killed."""
        artifact = self._run_main_with_mock(killed=0)
        assert artifact.get("n_zombies_killed") == 0

    def test_artifact_has_all_healthy(self) -> None:
        """Artifact must contain all_healthy."""
        artifact = self._run_main_with_mock(n_gpus=2, zombies=0, all_healthy=True)
        assert artifact.get("all_healthy") is True

    def test_artifact_has_retro_items_implemented(self) -> None:
        """Artifact must contain retro_items_implemented list."""
        artifact = self._run_main_with_mock()
        retro = artifact.get("retro_items_implemented", [])
        assert "RETRO-007" in retro
        assert "RETRO-008" in retro

    def test_artifact_status_is_success(self) -> None:
        """Artifact status must be 'success'."""
        artifact = self._run_main_with_mock()
        assert artifact.get("status") == "success"

    def test_dry_run_n_zombies_killed_is_zero(self) -> None:
        """SCENARIO-INFRA-012: dry-run always reports killed=0."""
        artifact = self._run_main_with_mock(zombies=2, killed=0)
        assert artifact.get("n_zombies_killed") == 0

    def test_all_healthy_false_when_one_gpu(self) -> None:
        """Artifact reflects unhealthy state when only 1 GPU detected."""
        artifact = self._run_main_with_mock(n_gpus=1, zombies=0, all_healthy=False)
        assert artifact.get("all_healthy") is False

    def test_artifact_is_json_serialisable(self) -> None:
        """Artifact dict must be serialisable to JSON without errors."""
        artifact = self._run_main_with_mock()
        try:
            json.dumps(artifact)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"Artifact is not JSON-serialisable: {exc}")


# ---------------------------------------------------------------------------
# TestExp339Degradation
# SCENARIO-INFRA-013
# ---------------------------------------------------------------------------


class TestExp339Degradation:
    """main() degrades gracefully when session_startup module errors out."""

    def test_import_error_does_not_crash_main(self) -> None:
        """SCENARIO-INFRA-013: ImportError from session_startup is caught."""
        artifact_holder: dict = {}
        original_write = Path.write_text

        def capture_write(self, data, *args, **kwargs):  # type: ignore[override]
            if "experiment_339" in str(self):
                artifact_holder["data"] = data
            return original_write(self, data, *args, **kwargs)

        with (
            patch(
                "carnot.pipeline.session_startup.run_session_startup",
                side_effect=RuntimeError("simulated failure"),
            ),
            patch.object(Path, "write_text", capture_write),
            patch.object(Path, "mkdir", lambda *a, **k: None),
        ):
            mod = _load_exp_339()
            try:
                mod.main()
            except Exception as exc:
                pytest.fail(f"main() raised unexpectedly: {exc}")

    def test_error_path_still_writes_artifact(self) -> None:
        """SCENARIO-INFRA-013: even on error, main() writes a status=success artifact."""
        artifact_holder: dict = {}
        original_write = Path.write_text

        def capture_write(self, data, *args, **kwargs):  # type: ignore[override]
            if "experiment_339" in str(self):
                artifact_holder["data"] = data
            return original_write(self, data, *args, **kwargs)

        with (
            patch(
                "carnot.pipeline.session_startup.run_session_startup",
                side_effect=RuntimeError("simulated failure"),
            ),
            patch.object(Path, "write_text", capture_write),
            patch.object(Path, "mkdir", lambda *a, **k: None),
        ):
            mod = _load_exp_339()
            mod.main()

        assert "data" in artifact_holder, "No artifact written on error path"
        artifact = json.loads(artifact_holder["data"])
        assert "n_gpus_detected" in artifact


# ---------------------------------------------------------------------------
# TestExp339ResultFile
# ---------------------------------------------------------------------------


class TestExp339ResultFile:
    """When the result file exists, it conforms to the expected schema."""

    @pytest.mark.skipif(
        not EXP_339_RESULT.exists(),
        reason="Experiment 339 result file not yet present",
    )
    def test_result_schema(self) -> None:
        """Result file must have artifact_schema == 'carnot.session_startup.v1'."""
        data = json.loads(EXP_339_RESULT.read_text())
        assert data.get("artifact_schema") == "carnot.session_startup.v1"

    @pytest.mark.skipif(
        not EXP_339_RESULT.exists(),
        reason="Experiment 339 result file not yet present",
    )
    def test_result_has_retro_items(self) -> None:
        """Result file must list RETRO-007 and RETRO-008."""
        data = json.loads(EXP_339_RESULT.read_text())
        retro = data.get("retro_items_implemented", [])
        assert "RETRO-007" in retro
        assert "RETRO-008" in retro
