"""Tests for the Exp 1270 gated arXiv bundle-v10 runner.

Spec traces: REQ-PUBLISH-014, SCENARIO-PUBLISH-014.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

from carnot.reporting import arxiv_bundle_v10 as exp1270


def _write_gate(results_dir: Path, *, fixed: int = 5) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "experiment_1269_paper_v6_critical_fixes_v2.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "critical_issues_fixed": fixed,
                "honest_verdict": "paper_v6_critical_fixes_v2_complete",
            }
        ),
        encoding="utf-8",
    )


def _write_paper(arxiv_dir: Path) -> None:
    figures = arxiv_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (arxiv_dir / "main.tex").write_text(
        r"\documentclass{article}\begin{document}Carnot\end{document}",
        encoding="utf-8",
    )
    (arxiv_dir / "carnot.bib").write_text("@misc{carnot,title={Carnot}}\n", encoding="utf-8")
    (figures / "fig1.pdf").write_bytes(b"%PDF-1.4\n")


def test_in_progress_artifact_is_written_before_gate_result(tmp_path: Path) -> None:
    """REQ-PUBLISH-014: every run starts with a durable in-progress artifact."""
    out_path = tmp_path / "results" / "experiment_1270_arxiv_bundle_v10_gated.json"

    artifact = exp1270.write_in_progress_artifact(out_path)

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["run_date"] == "20260504"
    assert written["arxiv_submitted"] is False


def test_gate_blocks_when_exp1269_fixed_count_is_too_low(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-014: exp1269 must fix at least five critical issues."""
    results = tmp_path / "results"
    _write_gate(results, fixed=4)

    artifact = exp1270.run(
        project_root=tmp_path,
        results_dir=results,
        arxiv_dir=tmp_path / "docs" / "arxiv-paper",
        out_path=results / "experiment_1270_arxiv_bundle_v10_gated.json",
        which=lambda _name: None,
    )

    assert artifact["status"] == "blocked"
    assert artifact["pdf_compiled"] is False
    assert artifact["bundle_path"] is None
    assert artifact["honest_verdict"] == "blocked_exp1269_gate_not_satisfied"


def test_missing_exp1269_gate_loads_as_empty_dict(tmp_path: Path) -> None:
    """REQ-PUBLISH-014: absent prerequisite artifacts cannot satisfy the gate."""
    assert exp1270.load_critical_gate(tmp_path) == {}
    assert exp1270.critical_gate_satisfied({}) is False


def test_missing_tool_artifact_names_exact_compile_commands(tmp_path: Path) -> None:
    """REQ-PUBLISH-014: local TeX-tooling absence is reported honestly."""
    results = tmp_path / "results"
    arxiv_dir = tmp_path / "docs" / "arxiv-paper"
    _write_gate(results)
    _write_paper(arxiv_dir)

    artifact = exp1270.run(
        project_root=tmp_path,
        results_dir=results,
        arxiv_dir=arxiv_dir,
        out_path=results / "experiment_1270_arxiv_bundle_v10_gated.json",
        which=lambda _name: None,
    )

    assert artifact["status"] == "blocked"
    assert artifact["pdf_compiled"] is False
    assert artifact["missing_tool"] == ["tectonic", "latexmk", "make"]
    assert artifact["honest_verdict"] == "blocked_missing_local_tex_tooling"


def test_make_without_targets_does_not_create_a_build_command(tmp_path: Path) -> None:
    """REQ-PUBLISH-014: make is only used when the paper directory exposes a target."""
    command, tools, targets = exp1270.discover_build_command(
        tmp_path,
        which=lambda name: "/usr/bin/make" if name == "make" else None,
    )

    assert command == []
    assert tools == {"tectonic": False, "latexmk": False, "make": True}
    assert targets == []


def test_make_targets_are_parsed_from_arxiv_makefile(tmp_path: Path) -> None:
    """REQ-PUBLISH-014: Makefile targets under docs/arxiv-paper are discoverable."""
    arxiv_dir = tmp_path / "docs" / "arxiv-paper"
    arxiv_dir.mkdir(parents=True)
    (arxiv_dir / "Makefile").write_text(
        "all: main.pdf\nbundle: main.pdf\n\t@tar -czf bundle.tar.gz main.tex\nclean:\n",
        encoding="utf-8",
    )

    assert exp1270.discover_make_targets(arxiv_dir) == ["all", "bundle", "clean"]


def test_default_command_runner_captures_stdout(tmp_path: Path) -> None:
    """REQ-PUBLISH-014: command output is available for honest artifact logging."""
    result = exp1270.default_command_runner(
        ["python3", "-c", "print('ok')"],
        tmp_path,
        10,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_tectonic_compile_creates_complete_bundle_artifact(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-014: a passing compile produces a verified v10 bundle."""
    results = tmp_path / "results"
    arxiv_dir = tmp_path / "docs" / "arxiv-paper"
    _write_gate(results)
    _write_paper(arxiv_dir)
    commands: list[tuple[tuple[str, ...], Path]] = []

    def fake_run(cmd: list[str], cwd: Path, timeout: int) -> SimpleNamespace:
        commands.append((tuple(cmd), cwd))
        (cwd / "main.pdf").write_bytes(b"%PDF-1.4 compiled\n")
        return SimpleNamespace(returncode=0, stdout="compiled", stderr="")

    artifact = exp1270.run(
        project_root=tmp_path,
        results_dir=results,
        arxiv_dir=arxiv_dir,
        out_path=results / "experiment_1270_arxiv_bundle_v10_gated.json",
        which=lambda name: f"/usr/bin/{name}" if name == "tectonic" else None,
        command_runner=fake_run,
    )

    assert commands == [(("tectonic", "main.tex"), arxiv_dir)]
    assert artifact["status"] == "complete"
    assert artifact["pdf_compiled"] is True
    assert artifact["bundle_path"] == "results/carnot-arxiv-v10-20260504.tar.gz"
    assert artifact["arxiv_submitted"] is False
    assert artifact["honest_verdict"] == "arxiv_bundle_v10_compiled_upload_pending"
    with tarfile.open(tmp_path / artifact["bundle_path"], "r:gz") as tf:
        names = set(tf.getnames())
    assert {"main.tex", "main.pdf", "carnot.bib", "figures/fig1.pdf"} <= names


def test_latexmk_is_second_priority_and_compile_failures_are_blocked(tmp_path: Path) -> None:
    """REQ-PUBLISH-014: compile failures do not masquerade as bundle success."""
    results = tmp_path / "results"
    arxiv_dir = tmp_path / "docs" / "arxiv-paper"
    _write_gate(results)
    _write_paper(arxiv_dir)

    artifact = exp1270.run(
        project_root=tmp_path,
        results_dir=results,
        arxiv_dir=arxiv_dir,
        out_path=results / "experiment_1270_arxiv_bundle_v10_gated.json",
        which=lambda name: f"/usr/bin/{name}" if name == "latexmk" else None,
        command_runner=lambda *_args: SimpleNamespace(returncode=12, stdout="", stderr="no pdf"),
    )

    assert artifact["status"] == "blocked"
    assert artifact["compile_command"] == ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"]
    assert artifact["pdf_compiled"] is False
    assert artifact["honest_verdict"] == "blocked_compile_failed"


def test_make_package_target_is_used_when_no_tex_engine_exists(tmp_path: Path) -> None:
    """REQ-PUBLISH-014: Makefile package targets are the final narrow build path."""
    results = tmp_path / "results"
    arxiv_dir = tmp_path / "docs" / "arxiv-paper"
    _write_gate(results)
    _write_paper(arxiv_dir)
    (arxiv_dir / "Makefile").write_text("package:\n\t@echo package\n", encoding="utf-8")

    def fake_run(cmd: list[str], cwd: Path, timeout: int) -> SimpleNamespace:
        assert cmd == ["make", "package"]
        (cwd / "main.pdf").write_bytes(b"%PDF-1.4 packaged\n")
        return SimpleNamespace(returncode=0, stdout="package", stderr="")

    artifact = exp1270.run(
        project_root=tmp_path,
        results_dir=results,
        arxiv_dir=arxiv_dir,
        out_path=results / "experiment_1270_arxiv_bundle_v10_gated.json",
        which=lambda name: "/usr/bin/make" if name == "make" else None,
        command_runner=fake_run,
    )

    assert artifact["status"] == "complete"
    assert artifact["compile_command"] == ["make", "package"]
    assert artifact["pdf_compiled"] is True


def test_submission_receipt_is_never_invented(tmp_path: Path) -> None:
    """REQ-PUBLISH-014: arxiv_submitted is true only with a local receipt."""
    receipt = tmp_path / "results" / "arxiv_submission_receipt_1270.json"
    assert exp1270.local_submission_receipt_exists(tmp_path / "results") is False
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"submitted": true}', encoding="utf-8")
    assert exp1270.local_submission_receipt_exists(tmp_path / "results") is True
