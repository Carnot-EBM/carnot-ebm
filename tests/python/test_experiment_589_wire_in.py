"""Tests for scripts/conductor_session_wrapper.py (Exp 589 wire-in).

Spec: REQ-INFRA-080, SCENARIO-INFRA-085, SCENARIO-INFRA-086
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure repo root and scripts/ are on the path so we can import the wrapper.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _p in (str(_REPO_ROOT), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import conductor_session_wrapper as csw  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_manifest(tmp_path: Path) -> Path:
    """Write a minimal exclusion manifest to a temp file and return its path."""
    manifest = {
        "excluded": [
            {
                "experiment_id": 308,
                "completed_milestone": "2026.04.37",
                "reason": "slowest-5 seven consecutive milestones, legacy checkpoint-failure state",
            },
            {
                "experiment_id": 260,
                "completed_milestone": "2026.04.37",
                "reason": "slowest-5 seven consecutive milestones, sequential inference loop",
            },
        ]
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    return p


# ---------------------------------------------------------------------------
# check_experiment() — unit tests
# ---------------------------------------------------------------------------


def test_check_experiment_excluded(tmp_manifest: Path) -> None:
    """SCENARIO-INFRA-085: check_experiment returns (True, reason) for excluded ID."""
    is_excl, reason = csw.check_experiment(308, manifest_path=str(tmp_manifest))
    assert is_excl is True
    # Reason must reference the experiment and the milestone.
    assert "308" in reason
    assert "2026.04.37" in reason


def test_check_experiment_not_excluded(tmp_manifest: Path) -> None:
    """SCENARIO-INFRA-086: check_experiment returns (False, '') for non-excluded ID."""
    is_excl, reason = csw.check_experiment(589, manifest_path=str(tmp_manifest))
    assert is_excl is False
    assert reason == ""


def test_check_experiment_second_excluded(tmp_manifest: Path) -> None:
    """check_experiment handles multiple excluded entries correctly."""
    is_excl, reason = csw.check_experiment(260, manifest_path=str(tmp_manifest))
    assert is_excl is True
    assert "260" in reason


def test_check_experiment_missing_manifest(tmp_path: Path) -> None:
    """Missing manifest file is treated as empty (no exclusions) — safe default."""
    missing = str(tmp_path / "nonexistent.json")
    is_excl, reason = csw.check_experiment(308, manifest_path=missing)
    assert is_excl is False
    assert reason == ""


# ---------------------------------------------------------------------------
# main() — CLI exit code tests
# ---------------------------------------------------------------------------


def test_main_exits_1_for_excluded(tmp_manifest: Path) -> None:
    """SCENARIO-INFRA-085: main() exits 1 when experiment_id is excluded."""
    with (
        patch.object(sys, "argv", ["conductor_session_wrapper.py", "308"]),
        patch("conductor_session_wrapper.check_experiment", return_value=(True, "Experiment 308 excluded")),
        pytest.raises(SystemExit) as exc_info,
    ):
        csw.main()
    assert exc_info.value.code == 1


def test_main_exits_0_for_not_excluded(tmp_manifest: Path) -> None:
    """SCENARIO-INFRA-086: main() exits 0 when experiment_id is not excluded."""
    with (
        patch.object(sys, "argv", ["conductor_session_wrapper.py", "589"]),
        patch("conductor_session_wrapper.check_experiment", return_value=(False, "")),
        pytest.raises(SystemExit) as exc_info,
    ):
        csw.main()
    assert exc_info.value.code == 0


def test_main_exits_2_no_args() -> None:
    """main() exits 2 when no experiment_id argument is provided."""
    with (
        patch.object(sys, "argv", ["conductor_session_wrapper.py"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        csw.main()
    assert exc_info.value.code == 2


def test_main_exits_2_non_integer() -> None:
    """main() exits 2 when experiment_id is not an integer."""
    with (
        patch.object(sys, "argv", ["conductor_session_wrapper.py", "notanint"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        csw.main()
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# print_usage() — smoke test
# ---------------------------------------------------------------------------


def test_print_usage(capsys: pytest.CaptureFixture) -> None:
    """print_usage() prints the canonical USAGE line."""
    csw.print_usage()
    out = capsys.readouterr().out
    assert "USAGE" in out
    assert "conductor_session_wrapper.py" in out
    assert "exp_id" in out


# ---------------------------------------------------------------------------
# Integration: wrapper uses real manifest file
# ---------------------------------------------------------------------------


def test_real_manifest_excludes_308() -> None:
    """The real conductor_exclusion_manifest.json must exclude experiment 308."""
    real_manifest = str(_REPO_ROOT / "scripts" / "conductor_exclusion_manifest.json")
    is_excl, reason = csw.check_experiment(308, manifest_path=real_manifest)
    assert is_excl is True, "Experiment 308 must be in the real exclusion manifest"


def test_real_manifest_excludes_all_five() -> None:
    """The real manifest excludes all five historically wasted experiments."""
    real_manifest = str(_REPO_ROOT / "scripts" / "conductor_exclusion_manifest.json")
    for exp_id in (308, 260, 309, 425, 410):
        is_excl, _ = csw.check_experiment(exp_id, manifest_path=real_manifest)
        assert is_excl is True, f"Experiment {exp_id} must be excluded in real manifest"


def test_real_manifest_allows_589() -> None:
    """The real manifest must NOT exclude experiment 589 (this experiment)."""
    real_manifest = str(_REPO_ROOT / "scripts" / "conductor_exclusion_manifest.json")
    is_excl, _ = csw.check_experiment(589, manifest_path=real_manifest)
    assert is_excl is False, "Experiment 589 must NOT be in the exclusion manifest"
