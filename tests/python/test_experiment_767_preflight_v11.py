"""Tests for experiment_767_preflight_v11 helpers.

Spec: REQ-INFRA-053, REQ-INFRA-054, SCENARIO-INFRA-062, SCENARIO-INFRA-063

WHY THESE TESTS:
    Exp 425 appeared for the 22nd consecutive full-milestone slowest-5 despite being
    in the exclusion manifest.  This test suite provides machine-checkable evidence
    that the pre-flight v11 helpers correctly:
    (a) audit_dequeue_sites() detects all dequeue sites and their guard status
        (REQ-INFRA-053 / SCENARIO-INFRA-062).
    (b) add_new_exclusions() adds Exps 425, 491, 603, 627 to the manifest and
        reports n_excluded_total (REQ-INFRA-054 / SCENARIO-INFRA-063).
    (c) compute_honest_verdict() maps coverage + n_excluded to the correct verdict
        string (REQ-INFRA-053).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make repo root importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_767_preflight_v11 import (
    _NEW_EXCLUSIONS,
    add_new_exclusions,
    audit_dequeue_sites,
    compute_honest_verdict,
)


# ---------------------------------------------------------------------------
# Tests for audit_dequeue_sites — REQ-INFRA-053, SCENARIO-INFRA-062
# ---------------------------------------------------------------------------


class TestAuditDequeueSites:
    """Verify that audit_dequeue_sites() correctly counts and classifies dequeue sites."""

    def test_primary_site_guarded_when_both_present(self, tmp_path: Path) -> None:
        """coverage_pct=100 when pick_next_task contains _task_is_excluded.

        Spec: REQ-INFRA-053, SCENARIO-INFRA-062
        """
        conductor = tmp_path / "research_conductor.py"
        conductor.write_text(
            "def pick_next_task(completed_log):\n"
            "    for task in RESEARCH_TASKS:\n"
            "        excluded, reason = _task_is_excluded(task)\n"
            "        if excluded:\n"
            "            continue\n"
            "        return task\n"
            "\n"
            "def other_fn():\n"
            "    pass\n"
        )
        result = audit_dequeue_sites(conductor)
        assert result["total_dequeue_sites"] == 1
        assert result["primary_site_guarded"] is True
        assert result["guarded_sites_before_patch"] == 1
        assert result["guarded_sites_after_patch"] == 1
        assert result["coverage_pct"] == 100.0
        assert result["full_coverage"] is True

    def test_primary_site_unguarded_when_guard_missing(self, tmp_path: Path) -> None:
        """coverage_pct=0 when pick_next_task lacks _task_is_excluded.

        Spec: REQ-INFRA-053
        """
        conductor = tmp_path / "research_conductor.py"
        conductor.write_text(
            "def pick_next_task(completed_log):\n"
            "    for task in RESEARCH_TASKS:\n"
            "        return task\n"
        )
        result = audit_dequeue_sites(conductor)
        assert result["total_dequeue_sites"] == 1
        assert result["primary_site_guarded"] is False
        assert result["guarded_sites_before_patch"] == 0
        assert result["coverage_pct"] == 0.0
        assert result["full_coverage"] is False

    def test_returns_zeros_for_missing_file(self, tmp_path: Path) -> None:
        """All counts are 0 when the conductor file does not exist.

        Spec: REQ-INFRA-053
        """
        missing = tmp_path / "nonexistent_conductor.py"
        result = audit_dequeue_sites(missing)
        assert result["total_dequeue_sites"] == 0
        assert result["coverage_pct"] == 0.0
        assert result["full_coverage"] is False

    def test_live_conductor_has_primary_site_guarded(self) -> None:
        """The actual scripts/research_conductor.py has pick_next_task with manifest guard.

        This is machine-checkable evidence that the primary dequeue site (pick_next_task)
        has the _task_is_excluded guard applied (REQ-INFRA-053 / SCENARIO-INFRA-062).

        Spec: REQ-INFRA-053, SCENARIO-INFRA-062
        """
        conductor_path = _REPO_ROOT / "scripts" / "research_conductor.py"
        assert conductor_path.exists(), "scripts/research_conductor.py must exist"
        result = audit_dequeue_sites(conductor_path)
        assert result["total_dequeue_sites"] >= 1, "At least one dequeue site must be found"
        assert result["primary_site_guarded"] is True, (
            "pick_next_task() in scripts/research_conductor.py does not contain "
            "_task_is_excluded — the manifest guard is missing (REQ-INFRA-053 violated)"
        )
        assert result["coverage_pct"] == 100.0, (
            f"Not all dequeue sites are guarded: coverage_pct={result['coverage_pct']:.1f}% "
            f"({result['guarded_sites_after_patch']}/{result['total_dequeue_sites']} guarded)"
        )


# ---------------------------------------------------------------------------
# Tests for add_new_exclusions — REQ-INFRA-054, SCENARIO-INFRA-063
# ---------------------------------------------------------------------------


class TestAddNewExclusions:
    """Verify that add_new_exclusions() correctly adds 425/491/603/627 to the manifest."""

    def _make_manifest(self, path: Path, entries: list[dict]) -> None:
        """Write a minimal manifest JSON to path."""
        path.write_text(json.dumps({"excluded": entries}, indent=2) + "\n")

    def test_adds_all_four_when_absent(self, tmp_path: Path) -> None:
        """All four IDs (425, 491, 603, 627) are added when manifest has no .58 entries.

        Spec: REQ-INFRA-054, SCENARIO-INFRA-063
        """
        manifest = tmp_path / "conductor_exclusion_manifest.json"
        self._make_manifest(manifest, [
            {"experiment_id": 308, "completed_milestone": "2026.04.37", "reason": "legacy"},
        ])
        added, n_total = add_new_exclusions(manifest)
        assert set(added) == {425, 491, 603, 627}
        assert n_total == 5  # 1 existing + 4 new
        # Verify they are in the written file
        raw = json.loads(manifest.read_text())
        written_ids = [e["experiment_id"] for e in raw["excluded"]]
        for eid in (425, 491, 603, 627):
            assert eid in written_ids

    def test_skips_already_added_at_58(self, tmp_path: Path) -> None:
        """No duplicate entries created when .58 entries already exist.

        Spec: REQ-INFRA-054
        """
        manifest = tmp_path / "conductor_exclusion_manifest.json"
        self._make_manifest(manifest, [
            {"experiment_id": 425, "completed_milestone": "2026.04.58", "reason": "already added"},
            {"experiment_id": 491, "completed_milestone": "2026.04.58", "reason": "already added"},
            {"experiment_id": 603, "completed_milestone": "2026.04.58", "reason": "already added"},
            {"experiment_id": 627, "completed_milestone": "2026.04.58", "reason": "already added"},
        ])
        added, n_total = add_new_exclusions(manifest)
        assert added == []
        assert n_total == 4  # No duplicates added

    def test_partial_add_when_some_already_present(self, tmp_path: Path) -> None:
        """Only missing IDs are added when some .58 entries already exist.

        Spec: REQ-INFRA-054
        """
        manifest = tmp_path / "conductor_exclusion_manifest.json"
        self._make_manifest(manifest, [
            {"experiment_id": 425, "completed_milestone": "2026.04.58", "reason": "already added"},
        ])
        added, n_total = add_new_exclusions(manifest)
        assert 425 not in added
        assert set(added) == {491, 603, 627}
        assert n_total == 4  # 1 + 3

    def test_returns_empty_for_missing_manifest(self, tmp_path: Path) -> None:
        """add_new_exclusions returns ([], 0) when manifest file does not exist.

        Spec: REQ-INFRA-054
        """
        missing = tmp_path / "nonexistent.json"
        added, n_total = add_new_exclusions(missing)
        assert added == []
        assert n_total == 0

    def test_returns_empty_for_invalid_json(self, tmp_path: Path) -> None:
        """add_new_exclusions returns ([], 0) for corrupt manifest JSON.

        Spec: REQ-INFRA-054
        """
        bad_manifest = tmp_path / "bad.json"
        bad_manifest.write_text("NOT VALID JSON {{{")
        added, n_total = add_new_exclusions(bad_manifest)
        assert added == []
        assert n_total == 0

    def test_live_manifest_has_all_four_after_patch(self) -> None:
        """The actual conductor_exclusion_manifest.json has entries for 425, 491, 603, 627.

        This verifies REQ-INFRA-054: all four IDs are present with completed_milestone
        set to at least '2026.04.58'.

        Spec: REQ-INFRA-054, SCENARIO-INFRA-063
        """
        manifest_path = _REPO_ROOT / "scripts" / "conductor_exclusion_manifest.json"
        assert manifest_path.exists(), "conductor_exclusion_manifest.json must exist"
        raw = json.loads(manifest_path.read_text())
        entries = raw.get("excluded", [])
        excluded_ids = {e["experiment_id"] for e in entries}
        for eid in (425, 491, 603, 627):
            assert eid in excluded_ids, (
                f"Exp {eid} not found in conductor_exclusion_manifest.json — "
                f"REQ-INFRA-054 violated (Exp {eid} must be in manifest after v11 patch)"
            )

    def test_new_exclusions_constant_has_four_entries(self) -> None:
        """_NEW_EXCLUSIONS constant defines exactly 4 entries with IDs 425/491/603/627.

        Spec: REQ-INFRA-054
        """
        ids = [e["experiment_id"] for e in _NEW_EXCLUSIONS]
        assert sorted(ids) == [425, 491, 603, 627]
        for entry in _NEW_EXCLUSIONS:
            assert "completed_milestone" in entry
            assert "reason" in entry
            assert entry["completed_milestone"] == "2026.04.58"


# ---------------------------------------------------------------------------
# Tests for compute_honest_verdict — REQ-INFRA-053
# ---------------------------------------------------------------------------


class TestComputeHonestVerdict:
    """Verify that compute_honest_verdict() maps conditions to correct verdict strings."""

    def test_full_coverage_achieved_when_all_conditions_met(self) -> None:
        """Verdict is 'full_manifest_coverage_achieved' when coverage=100% and n>=27.

        Spec: REQ-INFRA-053, SCENARIO-INFRA-062
        """
        verdict = compute_honest_verdict(
            full_coverage=True,
            n_excluded_total=27,
            gpu_clean=True,
            tests_passed=True,
        )
        assert verdict == "full_manifest_coverage_achieved"

    def test_full_coverage_achieved_with_extra_exclusions(self) -> None:
        """Verdict 'full_manifest_coverage_achieved' holds with n > 27.

        Spec: REQ-INFRA-053
        """
        verdict = compute_honest_verdict(
            full_coverage=True,
            n_excluded_total=30,
            gpu_clean=False,
            tests_passed=False,
        )
        assert verdict == "full_manifest_coverage_achieved"

    def test_partial_coverage_when_not_full(self) -> None:
        """Verdict is 'partial_coverage_remaining_sites' when full_coverage=False.

        Spec: REQ-INFRA-053
        """
        verdict = compute_honest_verdict(
            full_coverage=False,
            n_excluded_total=30,
            gpu_clean=True,
            tests_passed=True,
        )
        assert verdict == "partial_coverage_remaining_sites"

    def test_manifest_updated_unknown_when_coverage_met_but_n_low(self) -> None:
        """Verdict is 'manifest_updated_coverage_unknown' when coverage=100% but n<27.

        Spec: REQ-INFRA-053
        """
        verdict = compute_honest_verdict(
            full_coverage=True,
            n_excluded_total=10,
            gpu_clean=True,
            tests_passed=True,
        )
        assert verdict == "manifest_updated_coverage_unknown"

    def test_partial_coverage_takes_precedence_over_low_n(self) -> None:
        """Partial-coverage verdict wins even if n is also low.

        Spec: REQ-INFRA-053
        """
        verdict = compute_honest_verdict(
            full_coverage=False,
            n_excluded_total=5,
            gpu_clean=False,
            tests_passed=False,
        )
        assert verdict == "partial_coverage_remaining_sites"
