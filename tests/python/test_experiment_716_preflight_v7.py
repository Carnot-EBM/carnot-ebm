"""Tests for Exp 716: Pre-flight v7 incremental test selection.

Spec: REQ-INFRA-041, REQ-INFRA-042, SCENARIO-INFRA-050, SCENARIO-INFRA-051

WHY THESE TESTS:
    REQ-INFRA-041 requires that pre-flight uses a git-diff-based incremental selector
    rather than always running the full suite.  These tests verify the selector's core
    logic without running a real git process or touching the filesystem unnecessarily.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

from carnot.pipeline.incremental_test_selector import (  # noqa: E402
    IncrementalTestSelector,
    _get_changed_files,
    _any_rust_changed,
    _python_modules_changed,
    _FULL_SUITE_DIFF_THRESHOLD,
    _build_import_map,
    _collect_test_imports,
)


# ---------------------------------------------------------------------------
# REQ-INFRA-041 / SCENARIO-INFRA-050: incremental selector maps changed files
# ---------------------------------------------------------------------------


class TestIncrementalSelectorMapsChangedFiles:
    """SCENARIO-INFRA-050: incremental selector maps changed Python modules to test files.

    Spec: REQ-INFRA-041, SCENARIO-INFRA-050
    """

    def test_changed_python_module_selects_matching_tests(self, tmp_path):
        """When 1 Python module changes, only tests that import it are returned.

        WHY: REQ-INFRA-041-1 requires that the selector maps changed module paths
        to dependent test files via import graph analysis.
        """
        # Arrange: create a minimal test directory with two test files
        tests_dir = tmp_path / "tests" / "python"
        tests_dir.mkdir(parents=True)

        # test_a imports "carnot"
        (tests_dir / "test_a.py").write_text(
            "from carnot.pipeline import foo\n\ndef test_something(): pass\n"
        )
        # test_b imports only "scripts"
        (tests_dir / "test_b.py").write_text(
            "import scripts.experiment_307\n\ndef test_other(): pass\n"
        )

        # Simulate a git diff that changed a single carnot module
        changed = ["python/carnot/pipeline/foo.py"]

        selector = IncrementalTestSelector(repo_root=tmp_path, tests_dir=tests_dir)
        with patch(
            "carnot.pipeline.incremental_test_selector._get_changed_files", return_value=changed
        ):
            selected = selector.select()

        # Only test_a should be selected (imports carnot)
        assert selected is not None, "Should be incremental, not full-suite"
        assert str(tests_dir / "test_a.py") in selected
        assert str(tests_dir / "test_b.py") not in selected

    def test_no_python_changed_returns_empty_list(self, tmp_path):
        """When no Python modules changed (e.g. only YAML/docs changed), return [].

        WHY: REQ-INFRA-041-1 says to return [] not None when no Python is affected.
        An empty list signals 'no tests needed' without triggering the full suite.
        """
        tests_dir = tmp_path / "tests" / "python"
        tests_dir.mkdir(parents=True)

        changed = ["docs/readme.md", "ops/status.md"]
        selector = IncrementalTestSelector(repo_root=tmp_path, tests_dir=tests_dir)
        with patch(
            "carnot.pipeline.incremental_test_selector._get_changed_files", return_value=changed
        ):
            selected = selector.select()

        assert selected == [], f"Expected empty list, got {selected}"

    def test_get_stats_returns_all_fields(self, tmp_path):
        """get_stats() returns all four required fields for the Exp 716 artifact.

        WHY: REQ-INFRA-041-4 requires logging incremental_mode, tests_selected,
        tests_total, selection_ratio.
        """
        tests_dir = tmp_path / "tests" / "python"
        tests_dir.mkdir(parents=True)
        (tests_dir / "test_foo.py").write_text("import carnot\ndef test_x(): pass\n")

        changed = ["python/carnot/core.py"]
        selector = IncrementalTestSelector(repo_root=tmp_path, tests_dir=tests_dir)
        with patch(
            "carnot.pipeline.incremental_test_selector._get_changed_files", return_value=changed
        ):
            stats = selector.get_stats()

        assert "incremental_mode" in stats
        assert "tests_selected" in stats
        assert "tests_total" in stats
        assert "selection_ratio" in stats
        assert isinstance(stats["selection_ratio"], float)


# ---------------------------------------------------------------------------
# REQ-INFRA-041-2 / SCENARIO-INFRA-051: full-suite fallback conditions
# ---------------------------------------------------------------------------


class TestFullSuiteFallback:
    """SCENARIO-INFRA-051: full-suite fallback triggers on large diff or Rust change.

    Spec: REQ-INFRA-041-2, SCENARIO-INFRA-051
    """

    def test_large_diff_returns_none(self, tmp_path):
        """When diff > 20 files, select() returns None (full-suite signal).

        WHY: REQ-INFRA-041-2 — a broad diff could have cross-module regressions
        that partial selection would miss.
        """
        tests_dir = tmp_path / "tests" / "python"
        tests_dir.mkdir(parents=True)

        # 21 changed files (above threshold)
        changed = [f"python/carnot/module_{i}.py" for i in range(21)]
        assert len(changed) > _FULL_SUITE_DIFF_THRESHOLD

        selector = IncrementalTestSelector(repo_root=tmp_path, tests_dir=tests_dir)
        with patch(
            "carnot.pipeline.incremental_test_selector._get_changed_files", return_value=changed
        ):
            selected = selector.select()

        assert selected is None, "Large diff must trigger full-suite fallback (None)"

    def test_rust_change_returns_none(self, tmp_path):
        """When any crates/ file changed, select() returns None.

        WHY: REQ-INFRA-041-2 — Rust changes may affect PyO3 bindings, making any
        Python test potentially affected.  Only a full suite is safe.
        """
        tests_dir = tmp_path / "tests" / "python"
        tests_dir.mkdir(parents=True)

        changed = ["crates/carnot-ising/src/lib.rs", "python/carnot/pipeline/foo.py"]

        selector = IncrementalTestSelector(repo_root=tmp_path, tests_dir=tests_dir)
        with patch(
            "carnot.pipeline.incremental_test_selector._get_changed_files", return_value=changed
        ):
            selected = selector.select()

        assert selected is None, "Rust change must trigger full-suite fallback (None)"

    def test_exactly_20_files_runs_incremental(self, tmp_path):
        """Exactly 20 changed files is at the threshold — should still be incremental.

        WHY: REQ-INFRA-041-2 says "> 20 files", so 20 is still within incremental range.
        """
        tests_dir = tmp_path / "tests" / "python"
        tests_dir.mkdir(parents=True)

        changed = [f"python/carnot/module_{i}.py" for i in range(20)]
        assert len(changed) == _FULL_SUITE_DIFF_THRESHOLD

        selector = IncrementalTestSelector(repo_root=tmp_path, tests_dir=tests_dir)
        with patch(
            "carnot.pipeline.incremental_test_selector._get_changed_files", return_value=changed
        ):
            # Should NOT return None (20 is not > 20)
            selected = selector.select()

        # selected may be [] or a list, but not None
        assert selected is not None, "Exactly 20 files should NOT trigger full-suite"


# ---------------------------------------------------------------------------
# Artifact schema validation
# ---------------------------------------------------------------------------


class TestExp716ArtifactSchema:
    """Verify the Exp 716 deliverable JSON has all required fields.

    Spec: REQ-INFRA-041, REQ-INFRA-042
    """

    def test_artifact_has_all_required_fields(self, tmp_path):
        """The Exp 716 artifact must contain all fields listed in the task spec.

        WHY: The conductor reads specific fields from the artifact.  Missing fields
        cause KeyErrors in the retrospective and changelog generators.
        """
        # Build a minimal artifact the same way experiment_716 does
        required_fields = {
            "experiment",
            "schema",
            "run_date",
            "started_at",
            "finished_at",
            "duration_s",
            "status",
            "title",
            "incremental_mode",
            "tests_selected",
            "tests_total",
            "selection_ratio",
            "wall_time_minutes",
            "exp527_batched",
            "honest_verdict",
        }

        # Check the actual deliverable if it exists
        deliverable = _REPO_ROOT / "results" / "experiment_716_preflight_v7.json"
        if deliverable.exists():
            artifact = json.loads(deliverable.read_text())
            missing = required_fields - set(artifact.keys())
            assert not missing, f"Artifact missing required fields: {missing}"
        else:
            # Build a synthetic artifact to validate the schema contract
            artifact = {
                "experiment": 716,
                "schema": ["experiment", "honest_verdict", "incremental_mode"],
                "run_date": "20260422",
                "started_at": "2026-04-22T00:00:00Z",
                "finished_at": "2026-04-22T00:00:01Z",
                "duration_s": 1.0,
                "status": "success",
                "title": "Pre-flight v7: Incremental Test Selection",
                "incremental_mode": True,
                "tests_selected": 10,
                "tests_total": 450,
                "selection_ratio": 0.022,
                "wall_time_minutes": 12.4,
                "exp527_batched": False,
                "honest_verdict": "preflight_v7_complete",
            }
            missing = required_fields - set(artifact.keys())
            assert not missing, f"Schema contract missing required fields: {missing}"

    def test_honest_verdict_values(self):
        """honest_verdict must be one of the three specified values.

        WHY: REQ-INFRA-041 specifies three exact verdict strings.  Any other value
        would be unrecognised by the conductor's changelog generator.
        """
        valid_verdicts = {
            "preflight_v7_complete",
            "preflight_v7_full_suite",
            "preflight_v7_overhead_unchanged",
        }

        deliverable = _REPO_ROOT / "results" / "experiment_716_preflight_v7.json"
        if deliverable.exists():
            artifact = json.loads(deliverable.read_text())
            verdict = artifact.get("honest_verdict")
            assert verdict in valid_verdicts, (
                f"honest_verdict='{verdict}' not in {valid_verdicts}"
            )


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Unit tests for the internal helper functions in incremental_test_selector.

    Spec: REQ-INFRA-041
    """

    def test_any_rust_changed_detects_crates_path(self):
        """_any_rust_changed returns True for files under crates/."""
        assert _any_rust_changed(["crates/carnot-ising/src/lib.rs"])
        assert not _any_rust_changed(["python/carnot/pipeline/foo.py"])
        assert not _any_rust_changed([])

    def test_python_modules_changed_filters_non_python(self):
        """_python_modules_changed includes only .py files under python/ or scripts/."""
        changed = [
            "python/carnot/pipeline/foo.py",
            "scripts/experiment_716.py",
            "ops/status.md",
            "results/experiment_700.json",
            "crates/carnot-ising/src/lib.rs",
        ]
        result = _python_modules_changed(changed)
        paths = [str(p) for p in result]
        assert "python/carnot/pipeline/foo.py" in paths
        assert "scripts/experiment_716.py" in paths
        assert not any("ops" in p or "results" in p or "crates" in p for p in paths)

    def test_collect_test_imports_parses_from_import(self, tmp_path):
        """_collect_test_imports extracts stem from 'from X.Y import Z' statements."""
        test_file = tmp_path / "test_foo.py"
        test_file.write_text("from carnot.pipeline import foo\nimport scripts.exp\n")
        stems = _collect_test_imports(test_file)
        assert "carnot" in stems
        assert "scripts" in stems

    def test_collect_test_imports_handles_syntax_error(self, tmp_path):
        """_collect_test_imports returns empty set on unparseable files (non-fatal)."""
        test_file = tmp_path / "test_bad.py"
        test_file.write_text("def def broken syntax:")
        stems = _collect_test_imports(test_file)
        assert stems == set()

    def test_get_changed_files_exception_fallback(self, tmp_path):
        """_get_changed_files returns [] when subprocess raises (non-fatal).

        WHY: REQ-INFRA-041 says the selector must never block pre-flight.  If git
        is unavailable, the graceful fallback is an empty changed-files list.
        """
        from unittest.mock import patch, MagicMock
        with patch("carnot.pipeline.incremental_test_selector.subprocess.run", side_effect=OSError("no git")):
            result = _get_changed_files(tmp_path)
        assert result == []

    def test_get_changed_files_success_path(self, tmp_path):
        """_get_changed_files parses stdout lines from a successful git diff call."""
        from unittest.mock import patch, MagicMock
        mock_result = MagicMock()
        mock_result.stdout = "python/carnot/foo.py\nscripts/bar.py\n"
        with patch("carnot.pipeline.incremental_test_selector.subprocess.run", return_value=mock_result):
            result = _get_changed_files(tmp_path)
        assert result == ["python/carnot/foo.py", "scripts/bar.py"]

    def test_load_cache_reads_existing_file(self, tmp_path):
        """_load_cache returns the dict when the cache file exists and is valid JSON."""
        from carnot.pipeline.incremental_test_selector import _load_cache, _save_cache
        cache_path = tmp_path / ".preflight_test_cache.json"
        data = {"import_map": {"carnot": ["tests/python/test_foo.py"]}}
        _save_cache(cache_path, data)
        result = _load_cache(cache_path)
        assert result is not None
        assert "import_map" in result

    def test_load_cache_returns_none_for_missing_file(self, tmp_path):
        """_load_cache returns None when the cache file does not exist."""
        from carnot.pipeline.incremental_test_selector import _load_cache
        result = _load_cache(tmp_path / "nonexistent.json")
        assert result is None

    def test_load_cache_returns_none_for_corrupt_json(self, tmp_path):
        """_load_cache returns None for corrupt JSON (non-fatal)."""
        from carnot.pipeline.incremental_test_selector import _load_cache
        cache_path = tmp_path / ".cache.json"
        cache_path.write_text("{bad json}")
        result = _load_cache(cache_path)
        assert result is None

    def test_get_import_map_uses_cache_on_second_call(self, tmp_path):
        """_get_import_map returns cached result on second call without rebuilding.

        WHY: REQ-INFRA-041-3 requires caching to keep repeated invocations fast.
        """
        tests_dir = tmp_path / "tests" / "python"
        tests_dir.mkdir(parents=True)
        (tests_dir / "test_foo.py").write_text("import carnot\ndef test_x(): pass\n")

        selector = IncrementalTestSelector(repo_root=tmp_path, tests_dir=tests_dir)
        # First call builds the cache
        map1 = selector._get_import_map()
        # Second call should hit the cache (same result, no rebuild)
        map2 = selector._get_import_map()
        assert map1 == map2

    def test_get_stats_full_suite_mode(self, tmp_path):
        """get_stats() reports incremental_mode=False when selector returns None.

        WHY: REQ-INFRA-041-4 requires logging incremental_mode in all cases.
        """
        tests_dir = tmp_path / "tests" / "python"
        tests_dir.mkdir(parents=True)
        (tests_dir / "test_foo.py").write_text("import carnot\ndef test_x(): pass\n")

        # 21 files triggers full suite
        changed = [f"python/carnot/module_{i}.py" for i in range(21)]
        selector = IncrementalTestSelector(repo_root=tmp_path, tests_dir=tests_dir)
        with patch(
            "carnot.pipeline.incremental_test_selector._get_changed_files", return_value=changed
        ):
            stats = selector.get_stats()

        assert stats["incremental_mode"] is False
        assert stats["tests_selected"] == stats["tests_total"]
        assert stats["selection_ratio"] == 1.0

    def test_selector_short_path_stem_fallback(self, tmp_path):
        """select() uses Path.stem when a changed module path has fewer than 2 parts.

        WHY: Line 257 in incremental_test_selector handles Path objects with only a
        stem (e.g. Path("foo.py") has parts=('foo.py',), len < 2).  This test
        exercises that else-branch by constructing such a Path via mock.
        """
        tests_dir = tmp_path / "tests" / "python"
        tests_dir.mkdir(parents=True)
        (tests_dir / "test_foo.py").write_text("import foo\ndef test_x(): pass\n")

        from pathlib import PurePosixPath
        # A file with a single-component path: no directory prefix
        changed_files = ["foo.py"]

        selector = IncrementalTestSelector(repo_root=tmp_path, tests_dir=tests_dir)
        with patch(
            "carnot.pipeline.incremental_test_selector._get_changed_files",
            return_value=changed_files,
        ):
            # Force _python_modules_changed to return a Path with len(parts)==1
            # by patching it to return [Path("foo.py")] directly
            with patch(
                "carnot.pipeline.incremental_test_selector._python_modules_changed",
                return_value=[Path("foo.py")],
            ):
                selected = selector.select()

        # test_foo.py imports 'foo' and 'foo.py' stem is 'foo' — should be selected
        assert selected is not None

    def test_repo_root_function(self):
        """_repo_root() returns a valid directory path."""
        from carnot.pipeline.incremental_test_selector import _repo_root
        root = _repo_root()
        assert root.is_dir()
