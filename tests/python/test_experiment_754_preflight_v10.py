"""Tests for experiment_754_preflight_v10 helpers.

Spec: REQ-INFRA-051, REQ-INFRA-052, SCENARIO-INFRA-060, SCENARIO-INFRA-061

WHY THESE TESTS:
    Four consecutive milestones closed without the manifest patch being applied.
    These tests provide machine-checkable evidence that:
    (a) check_patch_applied() correctly detects the guard clause in
        scripts/research_conductor.py (REQ-INFRA-051 / REQ-INFRA-052).
    (b) check_exp527_excluded() correctly reads the exclusion manifest and
        detects Exp 527 (REQ-INFRA-048).
    (c) determine_honest_verdict() maps the three boolean conditions to the
        correct canonical verdict string (REQ-INFRA-052).
"""

import json
import sys
from pathlib import Path

import pytest

# Make repo root importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_754_preflight_v10 import (
    check_exp527_excluded,
    check_patch_applied,
    determine_honest_verdict,
)


# ---------------------------------------------------------------------------
# Tests for check_patch_applied — REQ-INFRA-051, REQ-INFRA-052
# ---------------------------------------------------------------------------


class TestCheckPatchApplied:
    """Verify that check_patch_applied() correctly detects the guard clause."""

    def test_returns_true_when_guard_present(self, tmp_path: Path) -> None:
        """patch_applied=True when 'validate_manifest_at_dequeue' is in file.

        Spec: REQ-INFRA-051, SCENARIO-INFRA-061
        """
        conductor = tmp_path / "research_conductor.py"
        conductor.write_text(
            "logger.info('RESEARCH STEP')\n"
            "    task_id = task.get('id', '')\n"
            "    if not validate_manifest_at_dequeue(task_id):\n"
            "        return True\n"
        )
        assert check_patch_applied(conductor) is True

    def test_returns_false_when_guard_absent(self, tmp_path: Path) -> None:
        """patch_applied=False when guard clause is missing from file.

        Spec: REQ-INFRA-051
        """
        conductor = tmp_path / "research_conductor.py"
        conductor.write_text(
            "logger.info('RESEARCH STEP')\n"
            "    if dry_run:\n"
            "        return True\n"
        )
        assert check_patch_applied(conductor) is False

    def test_returns_false_when_file_missing(self, tmp_path: Path) -> None:
        """patch_applied=False when the conductor file does not exist.

        Spec: REQ-INFRA-052 (patch application verified via code-level search)
        """
        missing = tmp_path / "nonexistent.py"
        assert check_patch_applied(missing) is False

    def test_live_conductor_has_patch_applied(self) -> None:
        """The actual scripts/research_conductor.py contains the guard clause.

        This is the machine-checkable evidence that the patch was applied in
        milestone .58 (closes the 4-milestone enforcement gap, REQ-INFRA-051).

        Spec: REQ-INFRA-051, SCENARIO-INFRA-061
        """
        conductor_path = _REPO_ROOT / "scripts" / "research_conductor.py"
        assert conductor_path.exists(), "research_conductor.py must exist"
        assert check_patch_applied(conductor_path) is True, (
            "validate_manifest_at_dequeue guard clause not found in "
            "scripts/research_conductor.py — the manifest fix patch has NOT "
            "been applied (REQ-INFRA-051 violated)"
        )


# ---------------------------------------------------------------------------
# Tests for check_exp527_excluded — REQ-INFRA-048
# ---------------------------------------------------------------------------


class TestCheckExp527Excluded:
    """Verify that check_exp527_excluded() reads the manifest correctly."""

    def _write_manifest(self, tmp_path: Path, entries: list) -> Path:
        """Write a manifest JSON file and return its path."""
        p = tmp_path / "conductor_exclusion_manifest.json"
        p.write_text(json.dumps({"excluded": entries}))
        return p

    def test_returns_true_when_527_present(self, tmp_path: Path) -> None:
        """exp527_excluded=True when experiment_id 527 is in the manifest.

        Spec: REQ-INFRA-048, SCENARIO-INFRA-060
        """
        manifest = self._write_manifest(
            tmp_path,
            [
                {"experiment_id": 527, "reason": "3-consecutive mandatory"},
                {"experiment_id": 308, "reason": "legacy"},
            ],
        )
        excluded, count = check_exp527_excluded(manifest)
        assert excluded is True
        assert count == 2

    def test_returns_false_when_527_absent(self, tmp_path: Path) -> None:
        """exp527_excluded=False when 527 is not in the manifest.

        Spec: REQ-INFRA-048
        """
        manifest = self._write_manifest(
            tmp_path,
            [{"experiment_id": 308, "reason": "legacy"}],
        )
        excluded, count = check_exp527_excluded(manifest)
        assert excluded is False
        assert count == 1

    def test_returns_false_when_manifest_missing(self, tmp_path: Path) -> None:
        """exp527_excluded=False, count=0 when the manifest file is missing.

        Non-fatal: the conductor must never be blocked by a missing manifest.
        Spec: REQ-INFRA-048
        """
        missing = tmp_path / "missing.json"
        excluded, count = check_exp527_excluded(missing)
        assert excluded is False
        assert count == 0

    def test_live_manifest_has_exp527(self) -> None:
        """The actual conductor_exclusion_manifest.json lists Exp 527.

        Spec: REQ-INFRA-048, SCENARIO-INFRA-057
        """
        manifest_path = _REPO_ROOT / "scripts" / "conductor_exclusion_manifest.json"
        assert manifest_path.exists(), "conductor_exclusion_manifest.json must exist"
        excluded, _ = check_exp527_excluded(manifest_path)
        assert excluded is True, (
            "Exp 527 is NOT in the exclusion manifest — it will re-enter "
            "the conductor queue for milestone .58 (REQ-INFRA-048 violated)"
        )


# ---------------------------------------------------------------------------
# Tests for determine_honest_verdict — REQ-INFRA-052
# ---------------------------------------------------------------------------


class TestDetermineHonestVerdict:
    """Verify that the four honest_verdict branches map correctly.

    Spec: REQ-INFRA-052
    """

    def test_patch_applied_gpu_clean_all_ok(self) -> None:
        """Golden path: patch applied, GPUs clean, 527 excluded.

        Spec: REQ-INFRA-052, SCENARIO-INFRA-061
        """
        verdict = determine_honest_verdict(
            patch_applied=True,
            gpu_clean=True,
            exp527_excluded=True,
        )
        assert verdict == "preflight_v10_patch_applied_gpu_clean"

    def test_patch_applied_gpu_dirty(self) -> None:
        """Patch applied but GPUs have residual VRAM (>= 100 MB).

        Spec: REQ-INFRA-052
        """
        verdict = determine_honest_verdict(
            patch_applied=True,
            gpu_clean=False,
            exp527_excluded=True,
        )
        assert verdict == "preflight_v10_patch_applied_gpu_dirty"

    def test_patch_failed(self) -> None:
        """Patch NOT applied — highest priority failure mode.

        Spec: REQ-INFRA-052, SCENARIO-INFRA-060
        """
        verdict = determine_honest_verdict(
            patch_applied=False,
            gpu_clean=True,
            exp527_excluded=True,
        )
        assert verdict == "preflight_v10_patch_failed"

    def test_patch_failed_overrides_other_conditions(self) -> None:
        """patch_failed verdict regardless of gpu/exp527 state when patch missing.

        Spec: REQ-INFRA-052
        """
        verdict = determine_honest_verdict(
            patch_applied=False,
            gpu_clean=False,
            exp527_excluded=False,
        )
        assert verdict == "preflight_v10_patch_failed"

    def test_exp527_leak(self) -> None:
        """Patch applied but Exp 527 is NOT in the manifest — enforcement gap.

        Spec: REQ-INFRA-052
        """
        verdict = determine_honest_verdict(
            patch_applied=True,
            gpu_clean=True,
            exp527_excluded=False,
        )
        assert verdict == "preflight_v10_exp527_leak"

    def test_exp527_leak_gpu_dirty(self) -> None:
        """exp527_leak takes precedence over gpu_dirty when 527 is missing.

        Spec: REQ-INFRA-052
        """
        verdict = determine_honest_verdict(
            patch_applied=True,
            gpu_clean=False,
            exp527_excluded=False,
        )
        # gpu_dirty check only fires when exp527 IS excluded; otherwise exp527_leak
        assert verdict == "preflight_v10_exp527_leak"
