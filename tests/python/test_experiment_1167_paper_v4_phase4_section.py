"""Tests for Exp 1167 Phase 4 arXiv Section 7 revision.

Spec traces: REQ-PUBLISH-006, SCENARIO-PUBLISH-006.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import experiment_1167_paper_v4_phase4_section as exp1167  # noqa: E402


def _exp1165_payload() -> dict[str, object]:
    return {
        "action_count_ratio": 0.25341914722445696,
        "phase4_solved_rate": 1.0,
        "energy_trace_monotone_fraction": 1.0,
        "phase4_mean_action_count": 6.3,
        "baseline_mean_action_count": 24.86,
    }


def _exp1166_payload() -> dict[str, object]:
    return {
        "seed_iq_score": 1.0,
        "seed_iq_score_confirmed": False,
        "seed_iq_action_efficiency": "115% of human baseline (2674 vs human 7534-8073 actions)",
        "leaderboard_comparison_table": [
            {"system_name": "Seed IQ (Active Inference)", "score": 1.0},
            {"system_name": "Carnot Phase 4 pilot", "score": "solved_rate=1.000"},
            {"system_name": "Frontier LLMs (autoregressive)", "score": "<1%"},
        ],
    }


def _phase4_tex() -> str:
    return r"""
\section{Phase 4: Carnot as Active Inference (Empirical Comparison)}
\label{sec:phase4-active-inference}
\subsection{Theoretical equivalence: free energy and Carnot $k=N$}
Carnot's $F(z)=\sum_k w_k E_k(z)$ is a variational-free-energy approximation.
\subsection{Phase 4 pilot results}
The pilot reports action_count_ratio = 0.253419 and solved_rate = 1.000.
\subsection{ARC-AGI-3 leaderboard context}
Seed IQ reports score 1.00 at 115\% human action-efficiency while frontier LLMs remain below 1\%.
\subsection{Gap analysis and future work}
Carnot is still a 5x5 synthetic prototype and Seed IQ targets full 30x30 ARC-AGI-3.
\section{Decentralization \& Deployment Sovereignty}
"""


def test_detect_phase4_section_verifies_req_publish_006_fields() -> None:
    """REQ-PUBLISH-006: Section 7 detection proves the Phase 4 revision content."""

    flags = exp1167.detect_phase4_section(
        _phase4_tex(),
        _exp1165_payload(),
        _exp1166_payload(),
    )

    assert flags == {
        "section7_expanded": True,
        "n_subsections_added": 4,
        "phase4_results_in_paper": True,
        "leaderboard_comparison_in_paper": True,
        "theoretical_equivalence_stated": True,
    }


def test_run_experiment_recompiles_repacks_and_writes_ready_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-PUBLISH-006: runner writes the hold-lift review artifact."""

    arxiv_dir = tmp_path / "docs" / "arxiv-paper"
    arxiv_dir.mkdir(parents=True)
    (arxiv_dir / "main.tex").write_text(_phase4_tex(), encoding="utf-8")
    (arxiv_dir / "main.pdf").write_bytes(b"old" * 1024)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "experiment_1165_phase4_active_inference_pilot_v1.json").write_text(
        json.dumps(_exp1165_payload()),
        encoding="utf-8",
    )
    (results_dir / "experiment_1166_arc_agi3_leaderboard_themesis_outreach.json").write_text(
        json.dumps(_exp1166_payload()),
        encoding="utf-8",
    )
    calls: list[tuple[tuple[str, ...], Path, int]] = []

    def fake_runner(cmd, cwd: Path, timeout: int):
        calls.append((tuple(cmd), cwd, timeout))
        if cmd[0] == "tectonic":
            (cwd / "main.pdf").write_bytes(b"new" * (200 * 1024))
        if cmd[0] == "tar":
            (tmp_path / "results" / "carnot-arxiv-v5.tar.gz").write_bytes(b"bundle")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    output_path = results_dir / "experiment_1167_paper_v4_phase4_section.json"
    artifact = exp1167.run_experiment(
        project_root=tmp_path,
        output_path=output_path,
        command_runner=fake_runner,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert calls == [
        (("tectonic", "main.tex"), arxiv_dir, 180),
        (("tar", "-czf", "results/carnot-arxiv-v5.tar.gz", "docs/arxiv-paper/"), tmp_path, 180),
    ]
    assert artifact["section7_expanded"] is True
    assert artifact["n_subsections_added"] == 4
    assert artifact["phase4_results_in_paper"] is True
    assert artifact["leaderboard_comparison_in_paper"] is True
    assert artifact["theoretical_equivalence_stated"] is True
    assert artifact["pdf_recompiled"] is True
    assert artifact["pdf_size_kb"] == 600.0
    assert artifact["bundle_path"] == "results/carnot-arxiv-v5.tar.gz"
    assert artifact["paper_ready_for_arxiv_hold_lift"] is True
    assert artifact["honest_verdict"] == "paper_v4_phase4_complete_arxiv_ready"


def test_failure_paths_are_explicit_for_req_publish_006(tmp_path: Path) -> None:
    """REQ-PUBLISH-006: failed compile or incomplete paper yields explicit verdicts."""

    partial_flags = exp1167.detect_phase4_section(
        "\\section{Related Work}\nSeed IQ only.\n\\section{Next}",
        _exp1165_payload(),
        _exp1166_payload(),
    )
    partial = exp1167.build_artifact(partial_flags, True, 400.0, True)
    assert partial["paper_ready_for_arxiv_hold_lift"] is False
    assert partial["honest_verdict"] == "partial_expansion_only"

    failed_pdf = exp1167.build_artifact(
        exp1167.detect_phase4_section(_phase4_tex(), _exp1165_payload(), _exp1166_payload()),
        False,
        0.0,
        False,
    )
    assert failed_pdf["honest_verdict"] == "section_expanded_pdf_recompile_failed"

    def failing_runner(cmd, cwd: Path, timeout: int):
        return SimpleNamespace(returncode=1, stdout="bad stdout", stderr="bad stderr")

    with pytest.raises(RuntimeError, match="tectonic failed: bad stderr"):
        exp1167.compile_pdf(tmp_path, command_runner=failing_runner)

    with pytest.raises(RuntimeError, match="tar failed: bad stderr"):
        exp1167.repack_bundle(tmp_path, tmp_path / "results" / "bundle.tar.gz", failing_runner)

    missing_pdf = tmp_path / "missing.pdf"
    with pytest.raises(RuntimeError, match="missing PDF"):
        exp1167.verify_pdf(missing_pdf, minimum_size_bytes=1)

    small_pdf = tmp_path / "small.pdf"
    small_pdf.write_bytes(b"tiny")
    with pytest.raises(RuntimeError, match="smaller than previous build"):
        exp1167.verify_pdf(small_pdf, minimum_size_bytes=10)
