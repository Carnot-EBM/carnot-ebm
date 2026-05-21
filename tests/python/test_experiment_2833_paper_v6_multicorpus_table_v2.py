"""Tests for Exp 2833 paper-v6 multi-corpus table v2.

Spec: REQ-PUBLISH-033, SCENARIO-PUBLISH-033, SCENARIO-PUBLISH-033B.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.reporting import paper_v6_multicorpus_table_v2 as exp2833


OLD_BLOCK = r"""
\begin{table}[h]
\centering
\caption{Multi-Corpus Dual-Condition Evaluation (exp2820--exp2823)}
\label{tab:multi_corpus}
\begin{tabular}{l c c c c l}
\toprule
Corpus & N & Architecture-only AUROC & Production AUROC & Learning $\Delta$ & Peer baseline \\
\midrule
FoVer & 1000 & 0.60 $\pm$ 0.05 & 0.85 $\pm$ 0.05 & +0.25 & HIVE 0.924 \\
MBPP & 100 & 0.80 $\pm$ 0.02 & 0.80 $\pm$ 0.02 & 0.00 & $<$peer$>$ \\
HumanEval & 164 & 0.80 $\pm$ 0.02 & 0.80 $\pm$ 0.02 & 0.00 & $<$peer$>$ \\
TruthfulQA & 200 & 0.69 $\pm$ 0.02 & 0.68 $\pm$ 0.02 & -0.01 & GPT-3 MC1 $\sim$28\% \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Self-Learning Contribution Disclosure}
\label{sec:self_learning_disclosure}
Per exp2820, the Production AUROC for FoVer includes placeholder state.

\subsection{Per-verifier Breakdown}
\label{sec:per_verifier_breakdown}
The exp2824 matrix analysis used placeholders.
"""


def _write_json(root: Path, relative_path: str, payload: dict[str, object]) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
        "learning_contribution": production - architecture_only
        if learning is None
        else learning,
        "per_verifier_learning_contribution": {"tier0r": 0.04, "nexus": 0.10},
        "per_verifier_condition_a_auroc": {"tier0r": production},
        "per_verifier_condition_b_auroc": {"tier0r": architecture_only},
    }


def _measured_inputs() -> dict[str, dict[str, object]]:
    return {
        "FoVer": _measured_corpus(
            production=0.91,
            architecture_only=0.84,
            production_std=0.02,
            architecture_std=0.03,
            learning=0.07,
        ),
        "MBPP": _measured_corpus(production=0.82, architecture_only=0.80),
        "HumanEval": _measured_corpus(
            production=0.78, architecture_only=0.76, production_std=None
        ),
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
        "honest_verdict": "blocked_cuda_unavailable",
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
        "FoVer": {**blocked, "n_examples": 1000},
        "MBPP": {
            **blocked,
            "honest_verdict": "blocked_model_cache",
            "corpus": "MBPP-sanitized-test",
            "n_problems": 100,
        },
        "HumanEval": {
            **blocked,
            "honest_verdict": None,
            "corpus": "HumanEval-full",
            "n_problems": 164,
        },
        "TruthfulQA": {**blocked, "corpus": "TruthfulQA-generation", "n_questions": 200},
        "Matrix": {
            "honest_verdict": "complete: upstream artifacts loaded but empty",
            "verifier_corpus_dual_matrix": {},
            "architecture_transfer_verifiers": [],
            "memory_augmented_verifiers": [],
            "corpus_specific_verifiers": [],
            "low_signal_verifiers": [],
            "diversity_gap_on_non_fover": True,
            "methodology_note": "No synthetic verifier rows were inferred.",
        },
    }


def test_req_publish_2833_spec_anchor_exists() -> None:
    """REQ-PUBLISH-033: the Exp 2833 publication contract is in OpenSpec."""

    spec = (exp2833.REPO_ROOT / "openspec/capabilities/publication/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-PUBLISH-033" in spec
    assert "SCENARIO-PUBLISH-033" in spec
    assert "SCENARIO-PUBLISH-033B" in spec
    assert "experiment_2833_paper_v6_multicorpus_table_v2.json" in spec


def test_scenario_publish_2833_measured_artifacts_render_numeric_rows() -> None:
    """SCENARIO-PUBLISH-033: numeric source AUROCs render into the paper block."""

    block = exp2833.render_section_block(_measured_inputs())

    assert "Multi-Corpus Dual-Condition Evaluation (exp2828--exp2832)" in block
    assert "FoVer & 1000 & 0.840 $\\pm$ 0.030 & 0.910 $\\pm$ 0.020 & +0.070" in block
    assert "TruthfulQA & 200 & 0.680 $\\pm$ 0.020 & 0.660 $\\pm$ 0.010 & -0.020" in block
    assert "learning\\_contribution = 0.070" in block
    assert "tier0r (+0.040)" in block
    assert "architecture-transfer: tier0r" in block
    assert "$<$peer$>$" not in block


def test_scenario_publish_2833_blocked_upstream_stays_unmeasured() -> None:
    """SCENARIO-PUBLISH-033B: blocked artifacts are not converted to fake AUROC."""

    artifact = exp2833.build_artifact(
        _blocked_inputs(),
        paper_v6_compile_success=True,
        compile_result={"success": True},
        duration_s=11.0,
    )
    block = exp2833.render_section_block(_blocked_inputs())

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["corpora_in_table"] == ["FoVer", "MBPP", "HumanEval", "TruthfulQA"]
    assert artifact["submission_package_ready"] is False
    assert artifact["arxiv_ready_v7"] is False
    assert artifact["all_dual_condition_auroc_measured"] is False
    assert artifact["duration_s"] == pytest.approx(11.0)
    assert "unmeasured" in block
    assert "blocked CUDA" in block
    assert "blocked model cache" in block
    assert "not reported" in block
    assert "$<$peer$>$" not in block
    assert "learning\\_contribution is unmeasured" in block


def test_scenario_publish_2833_run_updates_paper_and_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-033: run writes paper, artifact, and submission guards."""

    _paper(tmp_path)
    for corpus, relative_path in exp2833.ARTIFACT_FILES.items():
        _write_json(tmp_path, relative_path, _measured_inputs()[corpus])

    calls: list[tuple[list[str], Path, int]] = []

    def fake_runner(cmd: list[str], cwd: Path, timeout: int) -> SimpleNamespace:
        calls.append((cmd, cwd, timeout))
        return SimpleNamespace(returncode=0, stdout="compiled", stderr="")

    clock_values = iter([100.0, 106.5])
    artifact = exp2833.run(
        root=tmp_path,
        command_runner=fake_runner,
        clock=lambda: next(clock_values),
    )

    written = json.loads(
        (tmp_path / "results" / exp2833.OUTPUT_FILENAME).read_text(encoding="utf-8")
    )
    paper_text = (tmp_path / "docs/arxiv-paper/main.tex").read_text(encoding="utf-8")
    assert artifact == written
    assert artifact["paper_v6_compile_success"] is True
    assert artifact["submission_package_ready"] is True
    assert artifact["arxiv_ready_v7"] is True
    assert artifact["submission_attempted"] is False
    assert artifact["credentialed_submission_attempted"] is False
    assert artifact["duration_s"] == pytest.approx(6.5)
    assert "exp2820" not in paper_text
    assert "0.840 $\\pm$ 0.030" in paper_text
    assert calls == [
        (
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            tmp_path / "docs" / "arxiv-paper",
            300,
        )
    ]


def test_req_publish_2833_compile_failure_keeps_package_not_ready(tmp_path: Path) -> None:
    """REQ-PUBLISH-033: pdflatex failure is captured without submission attempts."""

    _paper(tmp_path)
    for corpus, relative_path in exp2833.ARTIFACT_FILES.items():
        _write_json(tmp_path, relative_path, _measured_inputs()[corpus])

    artifact = exp2833.run(
        root=tmp_path,
        command_runner=lambda *_args: SimpleNamespace(
            returncode=17, stdout="bad", stderr="tex error"
        ),
        clock=lambda: 1.0,
    )

    assert artifact["paper_v6_compile_success"] is False
    assert artifact["submission_package_ready"] is False
    assert artifact["arxiv_ready_v7"] is False
    assert artifact["compile_result"]["returncode"] == 17
    assert artifact["compile_result"]["stderr_tail"] == "tex error"
    assert artifact["submission_attempted"] is False
