import json
from pathlib import Path

from carnot.reporting.paper_v6_writethrough_2552 import (
    REQUIRED_CITATION_IDS,
    build_artifact,
    paper_update_status,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_exp2552_build_artifact_records_required_fields(tmp_path):
    """REQ-PUBLISH-003: Exp 2552 records a terminal paper update artifact."""

    paper = tmp_path / "docs" / "arxiv-paper" / "main.tex"
    bib = tmp_path / "docs" / "arxiv-paper" / "carnot.bib"
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text(
        r"""
Best .245 Ensemble v7b & 0.9857 & $\sigma=0.0175$ \\
Tier 0r (real corpus) & 0.9414 & $n=6{,}548$ \\
Tier 0s (real corpus) & 0.3758 & $n=6{,}548$ \\
Tier 0u (real corpus) & 0.5360 & $n=6{,}548$ \\
ARM-EBM~\cite{wu2025arm-ebm,tan2025rltunedebm}
hardware precedent~\cite{zhu2026parallelisingmachine}
free-energy framing~\cite{bouchaffra2026gamefep}
\paragraph{Honest negative result.}
exp2486 exp2508 exp2519 exp2532.
Phase~4 remains a theoretical hypothesis.
step-level granularity was not achieved.
""",
        encoding="utf-8",
    )
    bib.write_text(
        "\n".join(
            [
                "arXiv:2512.18730 tan2025rltunedebm",
                "arXiv:2604.17109 zhu2026parallelisingmachine",
                "arXiv:2605.09515 bouchaffra2026gamefep",
            ]
        ),
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "results" / "experiment_2546_ensemble_v7b.json",
        {
            "ensemble_v7b_auroc": 0.9857142857142858,
            "ensemble_v7b_auroc_std": 0.01749635530559415,
        },
    )
    _write_json(
        tmp_path / "results" / "experiment_2548_real_corpus_validation.json",
        {
            "n_real": 6548,
            "paper_citable": {"tier0r": True, "tier0s": True, "tier0u": True},
            "tier0r_real_auroc": 0.9413750415828194,
            "tier0s_real_auroc": 0.3758077973921437,
            "tier0u_real_auroc": 0.535990952669208,
        },
    )
    _write_json(
        tmp_path / "results" / "experiment_2544_phase4_option_b.json",
        {"phase4_honest_negative_documented": True},
    )

    artifact = build_artifact(tmp_path, started_epoch=100.0, now_epoch=112.5)

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["paper_updated"] is True
    assert artifact["ensemble_v7b_incorporated"] is True
    assert artifact["phase4_section_intact"] is True
    assert artifact["citations_added"] == list(REQUIRED_CITATION_IDS)
    assert artifact["duration_s"] == 12.5
    assert artifact["acceptance_gates"]["paper_updated == true"] is True


def test_exp2552_checked_in_paper_and_artifact_are_consistent():
    """REQ-PUBLISH-021: checked-in paper text carries the Exp 2552 write-through."""

    paper_text = Path("docs/arxiv-paper/main.tex").read_text(encoding="utf-8")
    bib_text = Path("docs/arxiv-paper/carnot.bib").read_text(encoding="utf-8")
    exp2546 = json.loads(Path("results/experiment_2546_ensemble_v7b.json").read_text())
    exp2548 = json.loads(Path("results/experiment_2548_real_corpus_validation.json").read_text())
    artifact = json.loads(Path("results/experiment_2552_paper_writethrough.json").read_text())

    status = paper_update_status(paper_text, bib_text, exp2546, exp2548)

    assert status["ensemble_v7b_incorporated"] is True
    assert status["real_corpus_aurocs_incorporated"] is True
    assert status["citations_present"] is True
    assert status["phase4_section_intact"] is True
    assert artifact["paper_updated"] is True
    assert artifact["citations_added"] == list(REQUIRED_CITATION_IDS)
