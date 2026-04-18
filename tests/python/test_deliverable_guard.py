"""Tests for DeliverableGuard and DocOnlyClassifier.

Spec: REQ-INFRA-033, REQ-INFRA-035,
      SCENARIO-INFRA-041, SCENARIO-INFRA-042, SCENARIO-INFRA-043
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from carnot.pipeline.deliverable_guard import DeliverableGuard, DocOnlyClassifier


# ---------------------------------------------------------------------------
# DeliverableGuard
# ---------------------------------------------------------------------------


class TestDeliverableGuardAssertWritten:
    """REQ-INFRA-033: assert_written() raises when file absent."""

    def test_passes_when_file_exists(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-041: guard passes when deliverable is present on disk."""
        deliverable = tmp_path / "result.json"
        deliverable.write_text('{"status": "success"}')
        guard = DeliverableGuard(str(deliverable))
        guard.assert_written()  # must not raise

    def test_raises_when_file_absent(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-041: guard raises FileNotFoundError when file is missing."""
        deliverable = tmp_path / "missing_result.json"
        guard = DeliverableGuard(str(deliverable))
        with pytest.raises(FileNotFoundError, match="RETRO-032"):
            guard.assert_written()

    def test_error_message_names_path(self, tmp_path: Path) -> None:
        """Error message must include the path so the researcher knows what is missing."""
        deliverable = tmp_path / "experiment_999.json"
        guard = DeliverableGuard(str(deliverable))
        with pytest.raises(FileNotFoundError, match="experiment_999.json"):
            guard.assert_written()


class TestDeliverableGuardAssertWrittenOrPartial:
    """SCENARIO-INFRA-042: assert_written_or_partial() passes if either file exists."""

    def test_passes_when_deliverable_exists(self, tmp_path: Path) -> None:
        deliverable = tmp_path / "result.json"
        partial = tmp_path / "partial.json"
        deliverable.write_text("{}")
        guard = DeliverableGuard(str(deliverable))
        guard.assert_written_or_partial(str(partial))  # must not raise

    def test_passes_when_only_partial_exists(self, tmp_path: Path) -> None:
        deliverable = tmp_path / "result.json"
        partial = tmp_path / "partial.json"
        partial.write_text("{}")
        guard = DeliverableGuard(str(deliverable))
        guard.assert_written_or_partial(str(partial))  # must not raise

    def test_raises_when_neither_exists(self, tmp_path: Path) -> None:
        deliverable = tmp_path / "result.json"
        partial = tmp_path / "partial.json"
        guard = DeliverableGuard(str(deliverable))
        with pytest.raises(FileNotFoundError):
            guard.assert_written_or_partial(str(partial))


# ---------------------------------------------------------------------------
# DocOnlyClassifier
# ---------------------------------------------------------------------------


class TestDocOnlyClassifier:
    """REQ-INFRA-035: doc-only classifier gates full test suite."""

    def test_doc_only_ops_and_bmad(self) -> None:
        """SCENARIO-INFRA-043: ops/ and _bmad/ files are doc-only."""
        clf = DocOnlyClassifier()
        assert clf.is_doc_only_diff(["ops/status.md", "_bmad/prd.md"]) is True

    def test_doc_only_markdown_at_root(self) -> None:
        """Plain .md files anywhere are doc-only."""
        clf = DocOnlyClassifier()
        assert clf.is_doc_only_diff(["README.md", "CHANGELOG.md"]) is True

    def test_doc_only_openspec_md(self) -> None:
        """openspec/**/*.md files are doc-only."""
        clf = DocOnlyClassifier()
        assert clf.is_doc_only_diff(["openspec/capabilities/verifiable-reasoning/spec.md"]) is True

    def test_code_file_makes_it_not_doc_only(self) -> None:
        """SCENARIO-INFRA-043: presence of any Python file returns False."""
        clf = DocOnlyClassifier()
        assert clf.is_doc_only_diff(["python/carnot/models/ising.py"]) is False

    def test_mixed_diff_not_doc_only(self) -> None:
        """A mix of docs and code is not doc-only — code wins."""
        clf = DocOnlyClassifier()
        changed = ["ops/status.md", "python/carnot/pipeline/deliverable_guard.py"]
        assert clf.is_doc_only_diff(changed) is False

    def test_rust_file_not_doc_only(self) -> None:
        """Rust source files are not doc-only."""
        clf = DocOnlyClassifier()
        assert clf.is_doc_only_diff(["crates/carnot-ising/src/lib.rs"]) is False

    def test_empty_list_returns_false(self) -> None:
        """Empty diff must return False to avoid accidentally skipping tests."""
        clf = DocOnlyClassifier()
        assert clf.is_doc_only_diff([]) is False

    def test_docs_prefix_is_doc_only(self) -> None:
        """Files under docs/ are doc-only even without .md extension."""
        clf = DocOnlyClassifier()
        assert clf.is_doc_only_diff(["docs/architecture.rst"]) is True
