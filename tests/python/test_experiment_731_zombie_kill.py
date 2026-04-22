"""Tests for Exp 731: conductor_manifest_validator and incremental test selector.

Spec traces: REQ-INFRA-046b, REQ-INFRA-047b,
             SCENARIO-INFRA-055b, SCENARIO-INFRA-056b
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.conductor_manifest_validator import validate_manifest_at_dequeue


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def manifest_with_308(tmp_path: Path) -> Path:
    """A manifest JSON that excludes experiment_id=308 (integer) and jepa_v15_cascade (string)."""
    data = {
        "excluded": [
            {"experiment_id": 308, "completed_milestone": "2026.04.37", "reason": "legacy"},
            {"experiment_id": "jepa_v15_cascade", "completed_milestone": "2026.04.53", "reason": "ood_auc_below_random"},
        ]
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture()
def empty_manifest(tmp_path: Path) -> Path:
    """A manifest with no excluded experiments."""
    data = {"excluded": []}
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# Tests: validate_manifest_at_dequeue
# ---------------------------------------------------------------------------

class TestValidateManifestAtDequeue:
    """REQ-INFRA-046b: conductor MUST block excluded tasks at dequeue."""

    def test_excluded_integer_id_returns_false(self, manifest_with_308: Path) -> None:
        """SCENARIO-INFRA-055b: exp308-legacy is in manifest → returns False (blocked).

        Spec: REQ-INFRA-046b, SCENARIO-INFRA-055b
        """
        result = validate_manifest_at_dequeue("exp308-legacy", manifest_path=manifest_with_308)
        assert result is False, "exp308 is in manifest — must be blocked"

    def test_allowed_unknown_id_returns_true(self, manifest_with_308: Path) -> None:
        """SCENARIO-INFRA-056b: exp999-new is NOT in manifest → returns True (allowed).

        Spec: REQ-INFRA-046b, SCENARIO-INFRA-056b
        """
        result = validate_manifest_at_dequeue("exp999-new-task", manifest_path=manifest_with_308)
        assert result is True, "exp999 is not in manifest — must be allowed"

    def test_string_id_excluded(self, manifest_with_308: Path) -> None:
        """String experiment IDs like 'jepa_v15_cascade' are matched and blocked.

        This is the root cause of the .55 787-minute gap: _task_is_excluded's regex
        returned (False, 'no id parsed') for string IDs, admitting them back into
        the queue.  validate_manifest_at_dequeue handles string IDs directly.

        Spec: REQ-INFRA-046b
        """
        result = validate_manifest_at_dequeue("jepa_v15_cascade", manifest_path=manifest_with_308)
        assert result is False, "jepa_v15_cascade is in manifest as string — must be blocked"

    def test_missing_manifest_allows_all(self, tmp_path: Path) -> None:
        """Missing manifest file → allow everything (safe default, never block conductor).

        Spec: REQ-INFRA-046b (exclusion is a performance optimisation, not a safety gate)
        """
        missing = tmp_path / "no_such_manifest.json"
        result = validate_manifest_at_dequeue("exp308-legacy", manifest_path=missing)
        assert result is True, "missing manifest must default to allowing tasks"

    def test_empty_manifest_allows_all(self, empty_manifest: Path) -> None:
        """Empty excluded list → allow everything.

        Spec: REQ-INFRA-046b
        """
        result = validate_manifest_at_dequeue("exp308-legacy", manifest_path=empty_manifest)
        assert result is True, "empty manifest must allow all tasks"

    def test_bare_numeric_string_id_excluded(self, manifest_with_308: Path) -> None:
        """Bare numeric string '308' is treated as excluded via integer normalisation.

        Spec: REQ-INFRA-046b
        """
        result = validate_manifest_at_dequeue("308", manifest_path=manifest_with_308)
        assert result is False, "'308' should map to excluded experiment_id=308"


# ---------------------------------------------------------------------------
# Tests: incremental test selector returns 0 on clean repo
# ---------------------------------------------------------------------------

class TestIncrementalTestSelector:
    """REQ-INFRA-047b: validate clean-repo behaviour of incremental test selector."""

    def test_clean_repo_selects_zero_tests(self) -> None:
        """On a clean git diff, incremental selector must return 0 test files.

        This confirms that the incremental mode is operational and the
        pre-flight step will not run unnecessary tests on a clean repo.

        Spec: REQ-INFRA-047b (variant: GPU VRAM clean / milestone pre-flight clean)
        """
        import subprocess
        repo_root = Path(__file__).resolve().parents[2]

        # Check if the working tree is clean before trusting the selector result.
        diff_result = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        untracked_result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--", "python/", "tests/"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        changed_py = [
            f for f in diff_result.stdout.splitlines()
            if f.strip().endswith(".py")
        ]
        new_py = [
            f for f in untracked_result.stdout.splitlines()
            if f.strip().endswith(".py")
        ]

        from carnot.pipeline.incremental_test_selector import IncrementalTestSelector
        sel = IncrementalTestSelector(repo_root=repo_root)
        stats = sel.get_stats()
        selected = sel.select()

        if changed_py or new_py:
            # Repo has uncommitted Python changes — selector may return > 0.
            # Don't fail the test; just assert incremental_mode is active.
            assert stats["incremental_mode"] is True, "incremental_mode must be True"
        else:
            # Truly clean repo — must select 0 files.
            n = len(selected) if selected is not None else stats["tests_selected"]
            assert stats["incremental_mode"] is True, "incremental_mode must be True on clean repo"
            assert n == 0, f"Expected 0 tests selected on clean repo, got {n}"
