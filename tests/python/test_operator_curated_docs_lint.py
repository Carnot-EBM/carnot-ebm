"""Tests for scripts/operator_curated_docs_lint.py.

Origin: 2026-05-26 exp3166 codex agent reverted README.md from the
project-level intro to a HuggingFace model card, violating CLAUDE.md
"Public Documentation Discipline". This lint is the Layer 1 mechanical
backstop. These tests pin the contract:

  - [conductor] subject + operator-curated path -> refuse
  - [outer-loop] subject + operator-curated path -> allow
  - [conductor] subject + non-curated path     -> allow
  - bare subject + operator-curated path        -> allow
  - empty / malformed inputs                    -> fail-open (warn but
                                                    don't block)

Spec coverage: CLAUDE.md "Public Documentation Discipline".
Tests trace to REQ-ARC-WMTE-6042 (the commit-time half of the
operator-curated defense; REQ-ARC-WMTE-6043 is the runtime guard).
The rename-source-side tests below are the named regression for the
2026-08-23 QA-layer SILENT_NON_FIRING finding on this lint.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _load():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "operator_curated_docs_lint.py"
    spec = importlib.util.spec_from_file_location("operator_curated_docs_lint", module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["operator_curated_docs_lint"] = mod
    spec.loader.exec_module(mod)
    return mod


_MOD = _load()


# -----------------------------------------------------------------------------
# Pattern matching
# -----------------------------------------------------------------------------


class TestMatchesOperatorCurated:
    """Path-pattern recognition for the operator-curated set."""

    @pytest.mark.parametrize(
        "path",
        [
            "README.md",
            "NOTICE",
            "LICENSE",
            "docs/index.html",
            "docs/roadmap.md",
            "docs/research-log.md",
            "docs/getting-started.md",
            "docs/cli-usage.md",
            "docs/mcp-server.md",
            "docs/tutorial.md",
            "docs/concepts.md",
            "docs/api-reference.md",
            "docs/CNAME",
            "docs/arxiv-paper/main.tex",
        ],
    )
    def test_canonical_protected_paths(self, path: str) -> None:
        assert _MOD._matches_operator_curated(path), f"{path} must be protected"

    @pytest.mark.parametrize(
        "path",
        [
            "docs/blog/announcement.html",
            "docs/blog/2026/may-26-release.html",
        ],
    )
    def test_blog_glob_patterns(self, path: str) -> None:
        assert _MOD._matches_operator_curated(path), f"{path} should match docs/blog/**/*.html glob"

    @pytest.mark.parametrize(
        "path",
        [
            "scripts/research_conductor.py",
            "python/carnot/pipeline/verify_repair.py",
            "results/experiment_3170_counterexample.json",
            "ops/conductor-log.md",
            "docs/research-notes/phase3-deep-think.md",  # NOT in protected set
            "docs/technical-report.md",  # has its own lint scope
            "docs/model_card_carnot_thinkprm_v3.md",  # not protected
            "ops/known-issues.md",
            "openspec/capabilities/safety/spec.md",
        ],
    )
    def test_non_protected_paths(self, path: str) -> None:
        assert not _MOD._matches_operator_curated(path), (
            f"{path} must NOT be in the operator-curated set"
        )


# -----------------------------------------------------------------------------
# Subject classification
# -----------------------------------------------------------------------------


class TestIsConductorCommit:
    """Recognition of the [conductor] subject marker."""

    @pytest.mark.parametrize(
        "subject",
        [
            "[conductor] In-process docs",
            "[Conductor] In-process docs",  # case-insensitive
            "[CONDUCTOR] anything",
        ],
    )
    def test_conductor_subjects(self, subject: str) -> None:
        assert _MOD._is_conductor_commit(subject)

    @pytest.mark.parametrize(
        "subject",
        [
            "[outer-loop] operator-authorized README update",
            "[operator] manual edit",
            "fix: typo in README",
            "",
            "[outer-loop] [conductor]",  # outer-loop prefix wins (it's first)
        ],
    )
    def test_non_conductor_subjects(self, subject: str) -> None:
        assert not _MOD._is_conductor_commit(subject)


# -----------------------------------------------------------------------------
# Full main() integration
# -----------------------------------------------------------------------------


class TestMainIntegration:
    """End-to-end main() with mocked git diff output."""

    def _run(
        self,
        tmp_path: Path,
        subject: str,
        staged: list[str],
    ) -> int:
        msg_file = tmp_path / "COMMIT_EDITMSG"
        msg_file.write_text(subject + "\n")
        with patch.object(_MOD, "_staged_files", return_value=staged):
            with patch.object(sys, "argv", ["operator_curated_docs_lint.py", str(msg_file)]):
                return _MOD.main()

    def test_blocks_conductor_readme_edit(self, tmp_path: Path, capsys) -> None:
        rc = self._run(tmp_path, "[conductor] Some research step", ["README.md"])
        assert rc == 1, "must refuse [conductor] commit touching README.md"
        captured = capsys.readouterr()
        assert "README.md" in captured.err

    def test_blocks_conductor_tutorial_edit(self, tmp_path: Path) -> None:
        rc = self._run(tmp_path, "[conductor] auto-doc-update", ["docs/tutorial.md"])
        assert rc == 1

    def test_blocks_conductor_blog_glob(self, tmp_path: Path) -> None:
        rc = self._run(
            tmp_path,
            "[conductor] auto-doc-update",
            ["docs/blog/2026/announcement.html"],
        )
        assert rc == 1

    def test_allows_outer_loop_readme_edit(self, tmp_path: Path) -> None:
        rc = self._run(
            tmp_path,
            "[outer-loop] README: replace ThinkPRM model card (operator-authorized)",
            ["README.md"],
        )
        assert rc == 0, "outer-loop is operator-authorized — must allow"

    def test_allows_operator_readme_edit(self, tmp_path: Path) -> None:
        rc = self._run(tmp_path, "fix: typo in README", ["README.md"])
        assert rc == 0, "operator commits (no marker) — must allow"

    def test_allows_conductor_non_protected_path(self, tmp_path: Path) -> None:
        rc = self._run(
            tmp_path,
            "[conductor] Cross-corpus matrix v28",
            ["results/experiment_3175_cross_corpus_matrix_v28.json"],
        )
        assert rc == 0, "conductor edits to non-curated paths must pass"

    def test_blocks_conductor_with_mixed_paths(self, tmp_path: Path) -> None:
        """If even ONE staged path is protected, refuse the whole commit."""
        rc = self._run(
            tmp_path,
            "[conductor] Some step",
            [
                "results/experiment_3175.json",
                "ops/conductor-log.md",
                "docs/tutorial.md",  # ← protected, this is what should block
            ],
        )
        assert rc == 1

    def test_fails_open_on_missing_message_file(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist.txt"
        with patch.object(_MOD, "_staged_files", return_value=["README.md"]):
            with patch.object(sys, "argv", ["operator_curated_docs_lint.py", str(nonexistent)]):
                rc = _MOD.main()
        assert rc == 0, "missing message file must fail open, not block"

    def test_fails_open_on_no_arg(self) -> None:
        """Called without a message-file arg, exit 0 (fail-open)."""
        with patch.object(sys, "argv", ["operator_curated_docs_lint.py"]):
            assert _MOD.main() == 0

    def test_skips_comment_lines_in_message(self, tmp_path: Path) -> None:
        """Git's commit message helpers prepend comment lines starting #;
        the subject is the first non-comment, non-empty line."""
        msg_file = tmp_path / "COMMIT_EDITMSG"
        msg_file.write_text("# Please enter the commit message\n#\n[conductor] real subject\n")
        with patch.object(_MOD, "_staged_files", return_value=["README.md"]):
            with patch.object(sys, "argv", ["operator_curated_docs_lint.py", str(msg_file)]):
                rc = _MOD.main()
        assert rc == 1, "must find subject past comment lines"


# -----------------------------------------------------------------------------
# Rename source side (QA-layer SILENT_NON_FIRING finding, 2026-08-23)
# -----------------------------------------------------------------------------


class TestRenameSourceSide:
    """A rename moves a protected doc out from under the guard; the guard
    must read the SOURCE side of an R-status entry, not only the
    destination."""

    def test_rename_of_readme_source_side_is_caught(self) -> None:
        """Named for the reported input: the protected source path
        `README.md` in the staged rename
        `R100 README.md docs/archive/project-intro.md`."""
        paths = _MOD._paths_from_name_status("R100\tREADME.md\tdocs/archive/project-intro.md\n")
        assert "README.md" in paths
        assert "docs/archive/project-intro.md" in paths
        assert any(_MOD._matches_operator_curated(p) for p in paths)

    def test_plain_statuses_still_parse(self) -> None:
        text = "M\tscripts/foo.py\nA\tdocs/new.md\nD\tops/old.md\n"
        assert _MOD._paths_from_name_status(text) == [
            "scripts/foo.py",
            "docs/new.md",
            "ops/old.md",
        ]

    def test_conductor_rename_of_readme_is_blocked_end_to_end(self, tmp_path: Path) -> None:
        msg_file = tmp_path / "COMMIT_EDITMSG"
        msg_file.write_text("[conductor] tidy docs\n")
        staged = _MOD._paths_from_name_status("R100\tREADME.md\tdocs/archive/project-intro.md\n")
        with patch.object(_MOD, "_staged_files", return_value=staged):
            with patch.object(sys, "argv", ["operator_curated_docs_lint.py", str(msg_file)]):
                assert _MOD.main() == 1
