"""Tests for the Exp 1380 audited arXiv bundle-v11 runner.

Spec traces: REQ-PUBLISH-018, SCENARIO-PUBLISH-019.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

from carnot.reporting import arxiv_bundle_v11_submission as exp1380


def _write_audit(results_dir: Path, *, ready: bool = True) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "experiment_1379_paper_integrity_audit_v2.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "paper_file_found": True,
                "paper_file_path": "docs/arxiv-paper/main.tex",
                "arxiv_submission_ready": ready,
                "remaining_blockers": [] if ready else ["paper integrity gate failed"],
                "figures_with_live_provenance": [
                    {"figure": "fig4.pdf / fig:alpha", "provenance": "live_gpu"},
                    {"figure": "fig5.pdf / fig:humaneval", "provenance": "live_gpu"},
                ],
                "figures_needing_verification": [
                    {"figure": "fig3.pdf", "reason": "placeholder file exists but is unused"}
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_paper(arxiv_dir: Path, *, include_placeholder: bool = False) -> None:
    figures = arxiv_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure_line = r"\includegraphics{figures/fig3.pdf}" if include_placeholder else ""
    (arxiv_dir / "main.tex").write_text(
        "\n".join(
            [
                r"\documentclass{article}",
                r"\usepackage{graphicx}",
                r"\begin{document}",
                r"\includegraphics{figures/fig1.pdf}",
                r"\includegraphics{figures/fig4.pdf}",
                r"\includegraphics{figures/fig5.pdf}",
                figure_line,
                r"\bibliographystyle{plain}",
                r"\bibliography{carnot}",
                r"\end{document}",
            ]
        ),
        encoding="utf-8",
    )
    (arxiv_dir / "carnot.bib").write_text("@misc{carnot,title={Carnot}}\n", encoding="utf-8")
    for name in ("fig1.pdf", "fig3.pdf", "fig4.pdf", "fig5.pdf"):
        (figures / name).write_bytes(b"%PDF-1.4\n")


def test_in_progress_artifact_contains_required_fields(tmp_path: Path) -> None:
    """REQ-PUBLISH-018: the runner writes a durable in-progress artifact first."""
    out_path = tmp_path / "results" / "experiment_1380_arxiv_bundle_v11_submission.json"

    artifact = exp1380.write_in_progress_artifact(out_path)

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    for field in (
        "status",
        "paper_file_found",
        "latex_compile_success",
        "bundle_file_path",
        "bundle_size_bytes",
        "figures_included",
        "submission_attempted",
        "submission_result",
        "arxiv_id_if_submitted",
        "remaining_blocker",
        "honest_verdict",
    ):
        assert field in written


def test_exp1379_gate_false_blocks_before_compile(tmp_path: Path) -> None:
    """REQ-PUBLISH-018: a false Exp 1379 arXiv gate prevents packaging."""
    results = tmp_path / "results"
    _write_audit(results, ready=False)

    artifact = exp1380.run(
        project_root=tmp_path,
        results_dir=results,
        out_path=results / "experiment_1380_arxiv_bundle_v11_submission.json",
        which=lambda _name: None,
    )

    assert artifact["status"] == "blocked"
    assert artifact["latex_compile_success"] is False
    assert artifact["remaining_blocker"] == "exp1379_arxiv_submission_ready_false"
    assert artifact["bundle_file_path"] is None


def test_success_creates_bundle_with_active_non_placeholder_figures(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-019: successful compile packages active paper sources only."""
    results = tmp_path / "results"
    arxiv_dir = tmp_path / "docs" / "arxiv-paper"
    bundle_path = results / "arxiv_bundle_v11.tar.gz"
    _write_audit(results)
    _write_paper(arxiv_dir)

    def fake_run(cmd: list[str], cwd: Path, timeout: int) -> SimpleNamespace:
        assert cmd == ["tectonic", "--keep-intermediates", "main.tex"]
        (cwd / "main.pdf").write_bytes(b"%PDF-1.4 compiled\n")
        (cwd / "main.bbl").write_text("\\begin{thebibliography}{1}\\end{thebibliography}\n")
        return SimpleNamespace(returncode=0, stdout="compiled", stderr="")

    artifact = exp1380.run(
        project_root=tmp_path,
        results_dir=results,
        out_path=results / "experiment_1380_arxiv_bundle_v11_submission.json",
        bundle_path=bundle_path,
        which=lambda name: f"/usr/bin/{name}" if name == "tectonic" else None,
        command_runner=fake_run,
    )

    assert artifact["status"] == "complete"
    assert artifact["latex_compile_success"] is True
    assert artifact["bundle_file_path"] == "results/arxiv_bundle_v11.tar.gz"
    assert artifact["bundle_size_bytes"] > 0
    assert artifact["figures_included"] == ["fig1.pdf", "fig4.pdf", "fig5.pdf"]
    assert artifact["submission_attempted"] is False
    assert artifact["honest_verdict"] == "submission_ready_archive_created_manual_upload_required"
    with tarfile.open(tmp_path / artifact["bundle_file_path"], "r:gz") as tf:
        names = set(tf.getnames())
    assert {"main.tex", "carnot.bib", "main.bbl"} <= names
    assert {"figures/fig1.pdf", "figures/fig4.pdf", "figures/fig5.pdf"} <= names
    assert "figures/fig3.pdf" not in names


def test_compile_failure_blocks_with_specific_blocker(tmp_path: Path) -> None:
    """REQ-PUBLISH-018: LaTeX failures are terminal and recorded honestly."""
    results = tmp_path / "results"
    arxiv_dir = tmp_path / "docs" / "arxiv-paper"
    _write_audit(results)
    _write_paper(arxiv_dir)

    artifact = exp1380.run(
        project_root=tmp_path,
        results_dir=results,
        out_path=results / "experiment_1380_arxiv_bundle_v11_submission.json",
        which=lambda name: f"/usr/bin/{name}" if name == "tectonic" else None,
        command_runner=lambda *_args: SimpleNamespace(returncode=17, stdout="", stderr="bad tex"),
    )

    assert artifact["status"] == "blocked"
    assert artifact["paper_file_found"] is True
    assert artifact["latex_compile_success"] is False
    assert artifact["remaining_blocker"] == "latex_compile_failed: tectonic returned 17"
    assert artifact["submission_attempted"] is False


def test_active_placeholder_figure_blocks_bundle(tmp_path: Path) -> None:
    """REQ-PUBLISH-018: audit-identified placeholder figures cannot enter the archive."""
    results = tmp_path / "results"
    arxiv_dir = tmp_path / "docs" / "arxiv-paper"
    _write_audit(results)
    _write_paper(arxiv_dir, include_placeholder=True)

    artifact = exp1380.run(
        project_root=tmp_path,
        results_dir=results,
        out_path=results / "experiment_1380_arxiv_bundle_v11_submission.json",
        which=lambda name: f"/usr/bin/{name}" if name == "tectonic" else None,
    )

    assert artifact["status"] == "blocked"
    assert "figure_provenance_blocker" in artifact["remaining_blocker"]
    assert "fig3.pdf" in artifact["remaining_blocker"]


def test_submission_attempt_extracts_arxiv_id(tmp_path: Path) -> None:
    """REQ-PUBLISH-018: successful upload output records the arXiv identifier."""
    bundle = tmp_path / "arxiv_bundle_v11.tar.gz"
    bundle.write_bytes(b"bundle")

    submission = exp1380.attempt_submission(
        bundle,
        which=lambda name: f"/usr/bin/{name}" if name == "arxiv-upload" else None,
        command_runner=lambda *_args: SimpleNamespace(
            returncode=0,
            stdout="submitted as arXiv:2605.01234v1",
            stderr="",
        ),
    )

    assert submission["submission_attempted"] is True
    assert submission["submission_result"] == "submitted"
    assert submission["arxiv_id_if_submitted"] == "2605.01234"
