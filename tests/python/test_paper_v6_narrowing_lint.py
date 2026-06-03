"""Tests for scripts/paper_v6_narrowing_lint.py.

Spec traces: REQ-PUBLISH-3716, SCENARIO-PUBLISH-3716,
REQ-PUBLISH-3768, SCENARIO-PUBLISH-3768.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import paper_v6_narrowing_lint as lint


@pytest.mark.parametrize(
    ("case_name", "text", "should_pass", "expected_fragment"),
    [
        (
            "clean_doc_passes",
            "The paper reports the frozen FoVer AUROC as 0.9131 under a 5-seed dual-condition protocol.",
            True,
            None,
        ),
        (
            "forbidden_phrasing_fails",
            "The KV260 hardware speedup demonstrates the current verifier deployment.",
            False,
            "KV260 hardware speedup",
        ),
        (
            "retracted_number_fails",
            "The prior table still lists FoVer AUROC 0.9857 as the headline value.",
            False,
            "0.9857",
        ),
        (
            "energy_as_generator_retraction_fails",
            "The paper now claims energy-as-generator works at scale.",
            False,
            "energy-as-generator works at scale",
        ),
    ],
)
def test_synthetic_doc_narrowing_lint_cases(
    tmp_path: Path,
    case_name: str,
    text: str,
    should_pass: bool,
    expected_fragment: str | None,
) -> None:
    """SCENARIO-PUBLISH-3716/3768: clean prose passes; retracted prose fails."""
    doc = tmp_path / f"{case_name}.md"
    doc.write_text(text, encoding="utf-8")

    hits = lint.scan_paths([doc])

    assert (not hits) is should_pass
    if expected_fragment is not None:
        assert any(expected_fragment in hit.matched_text for hit in hits)


def test_cli_nonzero_and_reports_file_line_for_req_publish_3716(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-PUBLISH-3716: CLI violations include file:line and return non-zero."""
    paper = tmp_path / "docs" / "technical-report.md"
    paper.parent.mkdir(parents=True)
    paper.write_text("Claim: Carnot's verifier ensemble runs faster on KV260.\n", encoding="utf-8")

    rc = lint.main(["--path", str(paper)])

    out = capsys.readouterr().out
    assert rc == 1
    assert f"{paper}:1" in out
    assert "Carnot's verifier ensemble runs faster on KV260" in out


def test_allowlist_and_missing_files_are_ignored_for_req_publish_3716(tmp_path: Path) -> None:
    """REQ-PUBLISH-3716: rule prose and unreadable/missing files are out of scope."""
    claude = tmp_path / "CLAUDE.md"
    claude.write_text("The forbidden phrase 0.9857 is documented here.\n", encoding="utf-8")

    assert lint.should_skip(claude, root=tmp_path) is True
    assert lint.scan_file(claude, root=tmp_path) == []
    assert lint.scan_file(tmp_path / "missing.md", root=tmp_path) == []


def test_retraction_context_does_not_fail_current_g3_semantics(tmp_path: Path) -> None:
    """REQ-PUBLISH-3716: explicit retraction disclosures are not live claims."""
    doc = tmp_path / "paper.md"
    doc.write_text(
        "Repinned downward from the v2 0.9857 number after the narrowing audit.\n",
        encoding="utf-8",
    )

    assert lint.scan_file(doc, root=tmp_path) == []


def test_discover_targets_uses_paper_targets_and_tracked_paper_v6_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-PUBLISH-3716: default scan covers paper targets plus tracked paper_v6 JSON."""
    paper = tmp_path / "docs" / "technical-report.md"
    tex = tmp_path / "docs" / "arxiv-paper" / "main.tex"
    artifact = tmp_path / "results" / "paper_v6_table.json"
    ignored = tmp_path / "results" / "experiment_1_paper_v6_table.json"
    for path in (paper, tex, artifact, ignored):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("clean\n", encoding="utf-8")
    monkeypatch.setattr(
        lint,
        "list_tracked_files",
        lambda root: [
            "results/paper_v6_table.json",
            "results/experiment_1_paper_v6_table.json",
            "results/paper_v6_missing.json",
        ],
    )

    targets = lint.discover_targets(tmp_path)

    assert targets == [tex, paper, artifact]


def test_list_tracked_files_handles_git_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-PUBLISH-3716: tracked-file discovery is tolerant of git failures."""
    calls: list[list[str]] = []

    def fake_success(cmd, cwd, capture_output, text, check):  # noqa: ANN001
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="a\n\nb\n")

    monkeypatch.setattr(lint.subprocess, "run", fake_success)
    assert lint.list_tracked_files(tmp_path) == ["a", "b"]
    assert calls == [["git", "ls-files"]]

    monkeypatch.setattr(
        lint.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stdout="ignored\n"),
    )
    assert lint.list_tracked_files(tmp_path) == []


def test_cli_verbose_clean_for_req_publish_3716(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-PUBLISH-3716: clean CLI runs can report scanned-file count."""
    doc = tmp_path / "clean.md"
    doc.write_text("FoVer AUROC is 0.9131.\n", encoding="utf-8")

    rc = lint.main(["--path", str(doc), "--verbose"])

    assert rc == 0
    assert "clean (1 files scanned)" in capsys.readouterr().out
