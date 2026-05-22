"""Tests for Exp 2841 paper-v6 multi-corpus table v3.

Spec: REQ-PUBLISH-034, SCENARIO-PUBLISH-034, SCENARIO-PUBLISH-034B.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.reporting import paper_v6_multicorpus_table_v3 as exp2841


OLD_BLOCK = r"""
\begin{table}[h]
\centering
\caption{Multi-Corpus Dual-Condition Evaluation (exp2828--exp2832)}
\label{tab:multi_corpus}
\begin{tabular}{l c c c c l}
\toprule
Corpus & N & Architecture-only AUROC & Production AUROC & Learning $\Delta$ & Peer baseline \\
\midrule
FoVer & 1000 & \emph{unmeasured (blocked CUDA)} & \emph{unmeasured (blocked CUDA)} & \emph{unmeasured (blocked CUDA)} & HIVE 0.924 \\
MBPP & 100 & \emph{unmeasured (blocked CUDA)} & \emph{unmeasured (blocked CUDA)} & \emph{unmeasured (blocked CUDA)} & peer baseline not established \\
HumanEval & 164 & \emph{unmeasured (blocked CUDA)} & \emph{unmeasured (blocked CUDA)} & \emph{unmeasured (blocked CUDA)} & peer baseline not established \\
TruthfulQA & 200 & \emph{unmeasured (blocked CUDA)} & \emph{unmeasured (blocked CUDA)} & \emph{unmeasured (blocked CUDA)} & GPT-3 MC1 $\sim$28\% \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Self-Learning Contribution Disclosure}
\label{sec:self_learning_disclosure}
Exp 2828 reports blocked CUDA data.

\subsection{Per-verifier Breakdown}
\label{sec:per_verifier_breakdown}
Exp 2832 provides an empty matrix.
"""


def _write_json(root: Path, relative_path: str, payload: dict[str, object]) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _paper(root: Path) -> Path:
    paper = root / "docs" / "arxiv-paper" / "main.tex"
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text(
        "\n".join(
            [
                r"\documentclass{article}",
                r"\usepackage{booktabs}",
                r"\begin{document}",
                OLD_BLOCK,
                r"Section~\ref{sec:bounds} corrects the old text.",
                r"\end{document}",
            ]
        ),
        encoding="utf-8",
    )
    return paper


def _measured_corpus(
    *,
    production: float,
    architecture_only: float,
    production_std: float | None = 0.01,
    architecture_std: float | None = 0.02,
    learning: float | None = None,
) -> dict[str, object]:
    return {
        "honest_verdict": "complete: measured",
        "condition_a_production_auroc_mean": production,
        "condition_a_production_auroc_std": production_std,
        "condition_b_architecture_only_auroc_mean": architecture_only,
        "condition_b_architecture_only_auroc_std": architecture_std,
        "learning_contribution": production - architecture_only if learning is None else learning,
        "per_verifier_learning_contribution": {"tier0r": 0.04, "nexus": 0.10},
        "per_verifier_condition_a_auroc": {"tier0r": [production]},
        "per_verifier_condition_b_auroc": {"tier0r": [architecture_only]},
    }


def _measured_inputs() -> dict[str, dict[str, object]]:
    return {
        "Runtime": {
            "honest_verdict": "success: runtime ready",
            "sota_runtime_ready": True,
            "selected_python": ".venv/bin/python",
        },
        "FoVer": _measured_corpus(
            production=0.9131336,
            architecture_only=0.8946624,
            production_std=0.007494,
            architecture_std=0.007539,
            learning=0.0184712,
        ),
        "MBPP": _measured_corpus(production=0.82, architecture_only=0.80),
        "HumanEval": _measured_corpus(production=0.78, architecture_only=0.76, production_std=None),
        "TruthfulQA": _measured_corpus(production=0.66, architecture_only=0.68),
        "Matrix": {
            "honest_verdict": "complete: matrix",
            "verifier_corpus_dual_matrix": {"tier0r": {"FoVer": {"production": 0.91}}},
            "architecture_transfer_verifiers": ["tier0r"],
            "memory_augmented_verifiers": ["nexus"],
            "corpus_specific_verifiers": ["truth_probe"],
            "low_signal_verifiers": ["low_probe"],
            "diversity_gap_on_non_fover": True,
            "methodology_note": "real matrix note",
        },
    }


def _blocked_inputs() -> dict[str, dict[str, object]]:
    blocked = {
        "honest_verdict": "blocked_dataset",
        "condition_a_production_auroc_mean": None,
        "condition_a_production_auroc_std": None,
        "condition_b_architecture_only_auroc_mean": None,
        "condition_b_architecture_only_auroc_std": None,
        "learning_contribution": None,
        "per_verifier_learning_contribution": {},
        "per_verifier_condition_a_auroc": {},
        "per_verifier_condition_b_auroc": {},
    }
    return {
        "Runtime": {
            "honest_verdict": "success: runtime ready",
            "sota_runtime_ready": True,
        },
        "FoVer": _measured_inputs()["FoVer"],
        "MBPP": {**blocked, "honest_verdict": "blocked_mbpp_dataset"},
        "HumanEval": {**blocked, "honest_verdict": "blocked_humaneval_dataset"},
        "TruthfulQA": {**blocked, "honest_verdict": "blocked_truthfulqa_generation_split"},
        "Matrix": {
            "honest_verdict": "complete: real upstream per-verifier AUROC matrix v3 built",
            "verifier_corpus_dual_matrix": {
                "tier0r": {
                    "FoVer": {
                        "production": 0.8947408,
                        "architecture_only": 0.8947408,
                        "delta": 0.0,
                    }
                }
            },
            "architecture_transfer_verifiers": [],
            "memory_augmented_verifiers": ["fr11_session_memory"],
            "corpus_specific_verifiers": ["tier0r"],
            "low_signal_verifiers": ["tier0s", "tier0u"],
            "diversity_gap_on_non_fover": True,
            "methodology_note": "Missing verifier/corpus cells remain null.",
        },
    }


def test_req_publish_2841_spec_anchor_exists() -> None:
    """REQ-PUBLISH-034: the Exp 2841 publication contract is in OpenSpec."""

    spec = (exp2841.REPO_ROOT / "openspec/capabilities/publication/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-PUBLISH-034" in spec
    assert "SCENARIO-PUBLISH-034" in spec
    assert "SCENARIO-PUBLISH-034B" in spec
    assert "experiment_2841_paper_v6_multicorpus_table_v3.json" in spec


def test_scenario_publish_2841_measured_artifacts_render_numeric_rows() -> None:
    """SCENARIO-PUBLISH-034: numeric source AUROCs render into the paper block."""

    block = exp2841.render_section_block(_measured_inputs())

    assert "Multi-Corpus Dual-Condition Evaluation (exp2836--exp2840)" in block
    assert "FoVer & 1000 & 0.895 $\\pm$ 0.008 & 0.913 $\\pm$ 0.007 & +0.018" in block
    assert "TruthfulQA & 200 & 0.680 $\\pm$ 0.020 & 0.660 $\\pm$ 0.010 & -0.020" in block
    assert "learning\\_contribution = 0.018" in block
    assert "tier0r (+0.040)" in block
    assert "Exp 2840 provides the per-verifier cross-corpus matrix" in block
    assert "$<$peer$>$" not in block
    assert "exp2828" not in block


def test_scenario_publish_2841_blocked_upstream_blocks_readiness() -> None:
    """SCENARIO-PUBLISH-034B: blocked artifacts do not become fake AUROC."""

    artifact = exp2841.build_artifact(
        _blocked_inputs(),
        paper_v6_compile_success=True,
        compile_result={"success": True},
        duration_s=11.0,
    )
    block = exp2841.render_section_block(_blocked_inputs())

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["corpora_in_table"] == ["FoVer", "MBPP", "HumanEval", "TruthfulQA"]
    assert artifact["submission_package_ready"] is False
    assert artifact["arxiv_ready_v8"] is False
    assert artifact["all_dual_condition_auroc_measured"] is False
    assert artifact["duration_s"] == pytest.approx(11.0)
    assert artifact["source_artifacts"]["Runtime"]["sota_runtime_ready"] is True
    assert "unmeasured" in block
    assert "blocked mbpp dataset" in block
    assert "blocked humaneval dataset" in block
    assert "blocked truthfulqa generation split" in block
    assert "$<$peer$>$" not in block
    assert "exp2833" not in block


def test_req_publish_2841_unmeasured_fover_disclosure_variants() -> None:
    """REQ-PUBLISH-034: missing FoVer learning data stays visibly unmeasured."""

    inputs = _blocked_inputs()
    fover_blocked = {
        **inputs["FoVer"],
        "honest_verdict": "blocked: manual review",
        "learning_contribution": None,
        "per_verifier_learning_contribution": {},
    }
    inputs["FoVer"] = fover_blocked
    blocked_block = exp2841.render_section_block(inputs)

    assert "learning\\_contribution is unmeasured" in blocked_block
    assert "ended at blocked." in blocked_block
    assert "No per-verifier learning-contribution rows were measured" in blocked_block

    inputs["FoVer"] = {**fover_blocked, "honest_verdict": None}
    missing_block = exp2841.render_section_block(inputs)

    assert "ended at not reported." in missing_block


def test_scenario_publish_2841_run_updates_paper_and_writes_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-PUBLISH-034: run writes paper, artifact, and submission guards."""

    _paper(tmp_path)
    for name, relative_path in exp2841.ARTIFACT_FILES.items():
        _write_json(tmp_path, relative_path, _measured_inputs()[name])

    calls: list[tuple[list[str], Path, int]] = []

    def fake_runner(cmd: list[str], cwd: Path, timeout: int) -> SimpleNamespace:
        calls.append((cmd, cwd, timeout))
        return SimpleNamespace(returncode=0, stdout="compiled", stderr="")

    clock_values = iter([100.0, 106.5])
    artifact = exp2841.run(
        root=tmp_path,
        command_runner=fake_runner,
        clock=lambda: next(clock_values),
    )

    written = json.loads(
        (tmp_path / "results" / exp2841.OUTPUT_FILENAME).read_text(encoding="utf-8")
    )
    paper_text = (tmp_path / "docs/arxiv-paper/main.tex").read_text(encoding="utf-8")
    assert artifact == written
    assert artifact["paper_v6_compile_success"] is True
    assert artifact["submission_package_ready"] is True
    assert artifact["arxiv_ready_v8"] is True
    assert artifact["submission_attempted"] is False
    assert artifact["credentialed_submission_attempted"] is False
    assert artifact["operator_only_external_publication"] is True
    assert artifact["duration_s"] == pytest.approx(6.5)
    assert "Exp 2828" not in paper_text
    assert "0.895 $\\pm$ 0.008" in paper_text
    assert calls == [
        (
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            tmp_path / "docs" / "arxiv-paper",
            300,
        )
    ]


def test_req_publish_2841_compile_failure_keeps_package_not_ready(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-034: pdflatex failure is captured without submission attempts."""

    _paper(tmp_path)
    for name, relative_path in exp2841.ARTIFACT_FILES.items():
        _write_json(tmp_path, relative_path, _measured_inputs()[name])

    artifact = exp2841.run(
        root=tmp_path,
        command_runner=lambda *_args: SimpleNamespace(
            returncode=17, stdout="bad", stderr="tex error"
        ),
        clock=lambda: 1.0,
    )

    assert artifact["paper_v6_compile_success"] is False
    assert artifact["submission_package_ready"] is False
    assert artifact["arxiv_ready_v8"] is False
    assert artifact["compile_result"]["returncode"] == 17
    assert artifact["compile_result"]["stderr_tail"] == "tex error"
    assert artifact["submission_attempted"] is False
