"""Tests for AtomicResultWriter (RETRO-030 fix).

Root cause of RETRO-030: Exp 446 exited with status 0 but produced no result file.
An exception during json.dump() left no file on disk; the watchdog missed it because
it only checked the exit code.  AtomicResultWriter prevents this by writing to a .tmp
file first, then os.rename() to the final path — the final path is either absent or
contains a complete JSON document, never a partial write.

Spec: REQ-INFRA-031, REQ-INFRA-032,
      SCENARIO-INFRA-039, SCENARIO-INFRA-040
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from carnot.pipeline.atomic_writer import AtomicResultWriter


class TestAtomicResultWriterWriteAndVerify:
    """SCENARIO-INFRA-039: write() creates file; verify_exists() returns True."""

    def test_write_creates_file(self, tmp_path: Path) -> None:
        """write() produces a file at the configured path."""
        # REQ-INFRA-031
        out = tmp_path / "result.json"
        writer = AtomicResultWriter(str(out))
        writer.write({"key": "value"})
        assert out.exists()

    def test_write_content_is_valid_json(self, tmp_path: Path) -> None:
        """write() produces valid JSON content."""
        out = tmp_path / "result.json"
        writer = AtomicResultWriter(str(out))
        data = {"experiment": 452, "status": "success"}
        writer.write(data)
        loaded = json.loads(out.read_text())
        assert loaded == data

    def test_verify_exists_true_after_write(self, tmp_path: Path) -> None:
        """verify_exists() returns True when the file was written."""
        # SCENARIO-INFRA-039
        out = tmp_path / "result.json"
        writer = AtomicResultWriter(str(out))
        writer.write({"ok": True})
        assert writer.verify_exists() is True

    def test_verify_exists_false_before_write(self, tmp_path: Path) -> None:
        """verify_exists() returns False when no file exists yet."""
        out = tmp_path / "missing.json"
        writer = AtomicResultWriter(str(out))
        assert writer.verify_exists() is False

    def test_tmp_file_absent_after_successful_write(self, tmp_path: Path) -> None:
        """The .tmp file is cleaned up by rename on a successful write."""
        out = tmp_path / "result.json"
        writer = AtomicResultWriter(str(out))
        writer.write({"x": 1})
        tmp = Path(str(out) + ".tmp")
        assert not tmp.exists()

    def test_write_creates_parent_directory(self, tmp_path: Path) -> None:
        """write() creates missing parent directories rather than raising FileNotFoundError."""
        out = tmp_path / "nested" / "dir" / "result.json"
        writer = AtomicResultWriter(str(out))
        writer.write({"nested": True})
        assert out.exists()


class TestAtomicResultWriterPartialWriteSafety:
    """SCENARIO-INFRA-040: partial write does not corrupt existing file."""

    def test_exception_mid_write_preserves_existing_file(self, tmp_path: Path) -> None:
        """If json.dumps() raises, the existing file at the final path is unchanged.

        Why this matters (RETRO-030): a plain open()+write() leaves the file partially
        written or absent if an exception occurs.  AtomicResultWriter writes to .tmp
        first; if the write fails, os.rename() never runs, so the original is intact.
        """
        # SCENARIO-INFRA-040
        out = tmp_path / "result.json"
        original_data = {"original": "content"}
        out.write_text(json.dumps(original_data))

        writer = AtomicResultWriter(str(out))

        # Simulate an exception during JSON serialisation.
        with patch("json.dumps", side_effect=ValueError("serialisation error")):
            with pytest.raises(ValueError, match="serialisation error"):
                writer.write({"new": "data"})

        # The original file must be intact.
        assert out.exists()
        assert json.loads(out.read_text()) == original_data

    def test_exception_mid_rename_leaves_tmp(self, tmp_path: Path) -> None:
        """If os.rename() raises, the .tmp file is the only partial artifact.

        The original file at the final path (if any) must be unchanged.
        """
        out = tmp_path / "result.json"
        original_data = {"safe": True}
        out.write_text(json.dumps(original_data))

        writer = AtomicResultWriter(str(out))

        with patch("os.rename", side_effect=OSError("rename failed")):
            with pytest.raises(OSError, match="rename failed"):
                writer.write({"new": True})

        # Original final path unchanged.
        assert json.loads(out.read_text()) == original_data
        # The .tmp file should exist (was written before the failed rename).
        tmp = Path(str(out) + ".tmp")
        assert tmp.exists()

    def test_no_existing_file_exception_leaves_no_final_file(self, tmp_path: Path) -> None:
        """When no prior result exists and write() fails, no file is left at the final path."""
        out = tmp_path / "result.json"
        writer = AtomicResultWriter(str(out))

        with patch("json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                writer.write({"x": 1})

        # No file at the final path (the .tmp rename never happened).
        assert not out.exists()


class TestAtomicResultWriterPath:
    """Unit tests for path attribute and string/Path interop."""

    def test_path_attribute_is_string(self, tmp_path: Path) -> None:
        """path attribute exposes the configured output path as a string."""
        out = tmp_path / "r.json"
        writer = AtomicResultWriter(str(out))
        assert writer.path == str(out)

    def test_write_overwrites_existing_file(self, tmp_path: Path) -> None:
        """A second write() replaces the first result atomically."""
        out = tmp_path / "result.json"
        writer = AtomicResultWriter(str(out))
        writer.write({"v": 1})
        writer.write({"v": 2})
        assert json.loads(out.read_text()) == {"v": 2}
