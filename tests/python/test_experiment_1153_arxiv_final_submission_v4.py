"""Tests for the Exp 1153 arXiv final-submission v4 artifact.

Spec traces: REQ-PUBLISH-005, SCENARIO-PUBLISH-005.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import experiment_1153_arxiv_final_submission_v4 as exp1153


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _paper_text(*, include_projection: bool = False, include_metacluster: bool = False) -> str:
    additions = []
    if include_projection:
        additions.append(
            "In milestone .89, HardNet++-style arithmetic projection repair "
            "corrected 20/20 synthetic violations at 100\\% accuracy and ran "
            "76{,}130$\\times$ faster than prompt repair (exp1147)."
        )
    if include_metacluster:
        additions.append(
            "MetaCluster-style centroid compression made SOSKANEnergyV3 "
            "5.03$\\times$ smaller with AUROC drop 0.018 (0.9902 to 0.9718), "
            "keeping the compressed verifier within the 0.02 degradation target (exp1148)."
        )
    return (
        "\\section{Empirical Realities \\& Anomalies}\n"
        "\\subsection{Milestone .87-.88 positive updates}\n"
        "GRPO with Energy Reward achieved +4 pp to +8.51 pp improvement.\n"
        + "\n".join(additions)
        + "\n\\subsection{$D_{\\mathrm{int}} = 1.6$ motivates the Welch bound (exp1093)}\n"
    )


def _source_artifacts(project_root: Path) -> None:
    _write_json(
        project_root / "results" / "experiment_1147_hardnet_projection_repair.json",
        {
            "speedup_factor": 76130.4127,
            "projection_repair_accuracy": 1.0,
            "n_violations_tested": 20,
        },
    )
    _write_json(
        project_root / "results" / "experiment_1148_metacluster_sos_kan_compression.json",
        {
            "size_reduction_factor": 5.026627,
            "auroc_drop": 0.018447,
            "auroc_original": 0.9902,
            "auroc_compressed": 0.971753,
        },
    )


def test_ensure_paper_mentions_adds_missing_req_publish_005_sentences(tmp_path: Path) -> None:
    """REQ-PUBLISH-005: missing exp1147/exp1148 summaries are inserted once."""
    main_tex = tmp_path / "docs" / "arxiv-paper" / "main.tex"
    main_tex.parent.mkdir(parents=True)
    main_tex.write_text(_paper_text(), encoding="utf-8")
    _source_artifacts(tmp_path)

    projection = json.loads(
        (tmp_path / "results" / "experiment_1147_hardnet_projection_repair.json").read_text()
    )
    metacluster = json.loads(
        (tmp_path / "results" / "experiment_1148_metacluster_sos_kan_compression.json").read_text()
    )

    flags, paper_updated = exp1153.ensure_paper_mentions(main_tex, projection, metacluster)
    updated = main_tex.read_text(encoding="utf-8")

    assert paper_updated is True
    assert flags["grpo_v2_result_in_paper"] is True
    assert flags["projection_repair_in_paper"] is True
    assert flags["metacluster_in_paper"] is True
    assert updated.count("HardNet++-style arithmetic projection repair") == 1
    assert updated.count("MetaCluster-style centroid compression") == 1
    assert "76{,}130$\\times$" in updated
    assert "5.03$\\times$" in updated

    second_flags, second_update = exp1153.ensure_paper_mentions(main_tex, projection, metacluster)
    assert second_flags == flags
    assert second_update is False


def test_run_experiment_writes_final_bundle_artifact_for_scenario_publish_005(
    tmp_path: Path,
) -> None:
    """SCENARIO-PUBLISH-005: exp1153 writes the upload-ready artifact schema."""
    arxiv_dir = tmp_path / "docs" / "arxiv-paper"
    arxiv_dir.mkdir(parents=True)
    (arxiv_dir / "main.tex").write_text(
        _paper_text(include_projection=True, include_metacluster=True),
        encoding="utf-8",
    )
    _source_artifacts(tmp_path)
    calls: list[tuple[tuple[str, ...], Path]] = []

    def fake_runner(cmd, cwd: Path, timeout: int):
        calls.append((tuple(cmd), cwd))
        if cmd[0] == "tectonic":
            (cwd / "main.pdf").write_bytes(b"0" * (328 * 1024))
        if cmd[0] == "tar":
            (tmp_path / "results" / "carnot-arxiv-v4.tar.gz").write_bytes(b"bundle")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    output_path = tmp_path / "results" / "experiment_1153_arxiv_final_submission_v4.json"
    artifact = exp1153.run_experiment(
        project_root=tmp_path,
        output_path=output_path,
        command_runner=fake_runner,
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact == written
    assert calls == [
        (("tectonic", "main.tex"), arxiv_dir),
        (("tar", "-czf", "results/carnot-arxiv-v4.tar.gz", "docs/arxiv-paper/"), tmp_path),
    ]
    assert written["grpo_v2_result_in_paper"] is True
    assert written["projection_repair_in_paper"] is True
    assert written["metacluster_in_paper"] is True
    assert written["pdf_recompiled"] is True
    assert written["pdf_path"] == "docs/arxiv-paper/main.pdf"
    assert written["pdf_size_kb"] == 328.0
    assert written["bundle_path"] == "results/carnot-arxiv-v4.tar.gz"
    assert written["bundle_verified"] is True
    assert written["arxiv_submitted"] is False
    assert written["arxiv_submission_id"] is None
    assert written["submission_deadline"] == "2026-05-15"
    assert written["honest_verdict"] == "pdf_recompiled_bundle_ready_upload_pending"
    assert len(written["manual_upload_steps"]) >= 8
    assert written["manual_upload_steps"][0].startswith("1. Open https://arxiv.org/login")


def test_verdict_and_command_failure_helpers_cover_req_publish_005(tmp_path: Path) -> None:
    """REQ-PUBLISH-005: helper branches expose deterministic final states."""
    assert exp1153.classify_verdict(True, True, True, False) == "paper_updated_recompiled"
    assert (
        exp1153.classify_verdict(False, False, False, False) == "paper_verified_no_recompile_needed"
    )
    assert exp1153.classify_verdict(False, True, True, True) == "submitted"

    def failing_runner(cmd, cwd: Path, timeout: int):
        return SimpleNamespace(returncode=2, stdout="bad stdout", stderr="bad stderr")

    try:
        exp1153.compile_pdf(tmp_path, command_runner=failing_runner)
    except RuntimeError as exc:
        assert "tectonic failed" in str(exc)
        assert "bad stderr" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("compile_pdf should have raised")
