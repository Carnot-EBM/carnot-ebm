"""Tests for scripts/conductor_manifest_precheck.py — 100% coverage.

Spec coverage: REQ-INFRA-085 (conductor session pre-check emits conductor_consulted=True),
               SCENARIO-INFRA-090 (precheck exits 1 and prints [EXCLUDED] for manifest IDs),
               SCENARIO-INFRA-091 (precheck exits 0 and writes sentinel for non-manifest IDs).

These tests exercise every branch in conductor_manifest_precheck.py:
- run_precheck(): excluded IDs, non-excluded IDs, mixed lists
- write_sentinel(): creates the sentinel file with a timestamp
- main(): all exit codes (0, 1, 2), no-args usage message, bad-int argument
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Module loader — loads the script without pip-installing it
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(tmp_manifest: Path | None = None):
    """Load scripts/conductor_manifest_precheck.py as a module.

    Parameters
    ----------
    tmp_manifest : Path, optional
        If given, patch _MANIFEST_PATH inside the module to point to this file.
    """
    module_path = _REPO_ROOT / "scripts" / "conductor_manifest_precheck.py"
    spec = importlib.util.spec_from_file_location("conductor_manifest_precheck", module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["conductor_manifest_precheck"] = mod
    spec.loader.exec_module(mod)
    if tmp_manifest is not None:
        mod._MANIFEST_PATH = str(tmp_manifest)
    return mod


# Load module once at module level for tests that don't need a custom manifest.
MODULE = _load_module()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manifest(tmp_path: Path, excluded_ids: list[int]) -> Path:
    """Write a minimal exclusion manifest JSON to tmp_path and return its path."""
    manifest_path = tmp_path / "test_manifest.json"
    payload = {
        "excluded": [
            {
                "experiment_id": eid,
                "completed_milestone": "2026.04.37",
                "reason": f"test exclusion for {eid}",
            }
            for eid in excluded_ids
        ]
    }
    manifest_path.write_text(json.dumps(payload))
    return manifest_path


# ---------------------------------------------------------------------------
# Tests for run_precheck()
# ---------------------------------------------------------------------------

class TestRunPrecheck:
    """REQ-INFRA-085: run_precheck() returns (False, [id]) when id is excluded."""

    def test_excluded_id_returns_false(self, tmp_path: Path, capsys) -> None:
        """SCENARIO-INFRA-090: excluded ID causes all_safe=False and prints [EXCLUDED]."""
        manifest_path = _make_manifest(tmp_path, [308, 260])
        mod = _load_module(manifest_path)

        all_safe, excluded = mod.run_precheck([308], manifest_path=str(manifest_path))

        assert all_safe is False
        assert 308 in excluded
        captured = capsys.readouterr()
        assert "[EXCLUDED]" in captured.out
        assert "308" in captured.out

    def test_non_excluded_id_returns_true(self, tmp_path: Path, capsys) -> None:
        """SCENARIO-INFRA-091: non-excluded ID causes all_safe=True and empty excluded list."""
        manifest_path = _make_manifest(tmp_path, [308])
        mod = _load_module(manifest_path)

        all_safe, excluded = mod.run_precheck([999], manifest_path=str(manifest_path))

        assert all_safe is True
        assert excluded == []
        captured = capsys.readouterr()
        assert "[EXCLUDED]" not in captured.out

    def test_mixed_list_returns_false_with_all_excluded(self, tmp_path: Path, capsys) -> None:
        """Mixed list: all_safe=False when even one ID is excluded."""
        manifest_path = _make_manifest(tmp_path, [308])
        mod = _load_module(manifest_path)

        all_safe, excluded = mod.run_precheck([308, 999], manifest_path=str(manifest_path))

        assert all_safe is False
        assert 308 in excluded
        assert 999 not in excluded

    def test_empty_manifest_allows_all(self, tmp_path: Path) -> None:
        """Empty manifest: every ID is safe."""
        manifest_path = _make_manifest(tmp_path, [])
        mod = _load_module(manifest_path)

        all_safe, excluded = mod.run_precheck([308, 260, 309], manifest_path=str(manifest_path))

        assert all_safe is True
        assert excluded == []

    def test_missing_manifest_allows_all(self, tmp_path: Path) -> None:
        """Missing manifest file: no exclusions (safe default)."""
        missing = tmp_path / "nonexistent.json"
        mod = _load_module(missing)

        all_safe, excluded = mod.run_precheck([308], manifest_path=str(missing))

        assert all_safe is True
        assert excluded == []

    def test_multiple_excluded_ids_all_printed(self, tmp_path: Path, capsys) -> None:
        """All excluded IDs in the input list are printed."""
        manifest_path = _make_manifest(tmp_path, [308, 260, 309])
        mod = _load_module(manifest_path)

        all_safe, excluded = mod.run_precheck([308, 260, 999], manifest_path=str(manifest_path))

        assert all_safe is False
        assert sorted(excluded) == [260, 308]
        captured = capsys.readouterr()
        assert "308" in captured.out
        assert "260" in captured.out


# ---------------------------------------------------------------------------
# Tests for write_sentinel()
# ---------------------------------------------------------------------------

class TestWriteSentinel:
    """REQ-INFRA-085: write_sentinel() creates the sentinel file with a timestamp."""

    def test_sentinel_file_created(self, tmp_path: Path) -> None:
        """Sentinel file is created and contains a non-empty timestamp."""
        mod = _load_module()
        # Redirect sentinel to a temp path.
        orig = mod._SENTINEL_PATH
        try:
            mod._SENTINEL_PATH = tmp_path / "conductor_consulted_at.txt"
            ts = mod.write_sentinel()
            assert mod._SENTINEL_PATH.exists()
            content = mod._SENTINEL_PATH.read_text().strip()
            assert content == ts
            assert "T" in ts  # ISO-8601 timestamp
        finally:
            mod._SENTINEL_PATH = orig

    def test_sentinel_overwritten_on_second_call(self, tmp_path: Path) -> None:
        """A second call to write_sentinel() overwrites the previous sentinel."""
        mod = _load_module()
        orig = mod._SENTINEL_PATH
        try:
            mod._SENTINEL_PATH = tmp_path / "sentinel.txt"
            ts1 = mod.write_sentinel()
            ts2 = mod.write_sentinel()
            # File exists and contains the latest timestamp.
            content = mod._SENTINEL_PATH.read_text().strip()
            assert content == ts2
            # Timestamps are strings so we can compare inequality only if time passed.
            # Just assert both are non-empty ISO-8601-like strings.
            assert "T" in ts1
            assert "T" in ts2
        finally:
            mod._SENTINEL_PATH = orig


# ---------------------------------------------------------------------------
# Tests for main()
# ---------------------------------------------------------------------------

class TestMain:
    """Tests for the CLI entry point main()."""

    def _run_main(self, args: list[str], tmp_path: Path, manifest_path: Path | None = None):
        """Call main() with patched sys.argv and _SENTINEL_PATH, return exit code."""
        mod = _load_module(manifest_path)
        orig_sentinel = mod._SENTINEL_PATH
        mod._SENTINEL_PATH = tmp_path / "sentinel.txt"

        try:
            with patch.object(sys, "argv", ["conductor_manifest_precheck.py"] + args):
                with pytest.raises(SystemExit) as exc_info:
                    mod.main()
            return exc_info.value.code, mod._SENTINEL_PATH
        finally:
            mod._SENTINEL_PATH = orig_sentinel

    def test_no_args_exits_2(self, tmp_path: Path) -> None:
        """main() exits 2 when called with no arguments."""
        exit_code, _ = self._run_main([], tmp_path)
        assert exit_code == 2

    def test_bad_int_exits_2(self, tmp_path: Path) -> None:
        """main() exits 2 when experiment_id is not an integer."""
        exit_code, _ = self._run_main(["not_an_int"], tmp_path)
        assert exit_code == 2

    def test_excluded_id_exits_1(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-090: main() exits 1 when experiment is excluded."""
        manifest_path = _make_manifest(tmp_path, [308])
        exit_code, sentinel_path = self._run_main(["308"], tmp_path, manifest_path)
        assert exit_code == 1
        # Sentinel must NOT be written when excluded.
        assert not sentinel_path.exists()

    def test_safe_id_exits_0(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-091: main() exits 0 and writes sentinel for non-excluded ID."""
        manifest_path = _make_manifest(tmp_path, [308])
        exit_code, sentinel_path = self._run_main(["601"], tmp_path, manifest_path)
        assert exit_code == 0
        assert sentinel_path.exists()

    def test_safe_id_prints_precheck_ok(self, tmp_path: Path, capsys) -> None:
        """main() prints [PRECHECK OK] when all IDs are safe."""
        manifest_path = _make_manifest(tmp_path, [308])
        mod = _load_module(manifest_path)
        mod._SENTINEL_PATH = tmp_path / "sentinel.txt"

        with patch.object(sys, "argv", ["conductor_manifest_precheck.py", "601"]):
            with pytest.raises(SystemExit):
                mod.main()

        captured = capsys.readouterr()
        assert "[PRECHECK OK]" in captured.out
        assert "conductor_consulted=True" in captured.out

    def test_excluded_id_prints_excluded(self, tmp_path: Path, capsys) -> None:
        """main() prints [EXCLUDED] for excluded experiment."""
        manifest_path = _make_manifest(tmp_path, [308])
        mod = _load_module(manifest_path)
        mod._SENTINEL_PATH = tmp_path / "sentinel.txt"

        with patch.object(sys, "argv", ["conductor_manifest_precheck.py", "308"]):
            with pytest.raises(SystemExit):
                mod.main()

        captured = capsys.readouterr()
        assert "[EXCLUDED]" in captured.out

    def test_multiple_safe_ids_exits_0(self, tmp_path: Path) -> None:
        """main() accepts multiple safe IDs and exits 0."""
        manifest_path = _make_manifest(tmp_path, [308])
        exit_code, _ = self._run_main(["601", "602", "603"], tmp_path, manifest_path)
        assert exit_code == 0
