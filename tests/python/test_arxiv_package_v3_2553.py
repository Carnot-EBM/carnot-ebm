"""Tests for the Exp 2553 arXiv package v3 readiness artifact.

Spec traces: REQ-PUBLISH-030, SCENARIO-PUBLISH-030.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from carnot.reporting import arxiv_package_v3_2553 as exp2553


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_ready_inputs(root: Path) -> None:
    paper = root / "docs" / "arxiv-paper" / "main.tex"
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text(
        r"""
\documentclass{article}
\begin{document}
\begin{abstract}
One two three four five.
\end{abstract}
\section{Results}
Best AUROC is 0.9750.
\end{document}
""",
        encoding="utf-8",
    )
    _write_json(
        root / "results" / "experiment_2544_phase4_option_b.json",
        {"phase4_honest_negative_documented": True},
    )
    _write_json(
        root / "results" / "experiment_2536_latex_compile_fix.json",
        {"latex_compile_success": True, "abstract_word_count": 205},
    )
    _write_json(
        root / "results" / "experiment_2552_paper_writethrough.json",
        {"paper_updated": True},
    )
    _write_json(
        root / "results" / "experiment_2441_phase1_ship_gate_completion_v5.json",
        {"phase1_ship_gate_met": True},
    )
    _write_json(
        root / "results" / "experiment_2479_paper_integrity_fix.json",
        {"audit_passed_after_fix": True},
    )
    _write_json(
        root / "results" / "experiment_2498_auroc_adversarial_v2_group_cond.json",
        {
            "auroc_adversarially_verified": True,
            "group_conditional_auroc_replicated": 0.975,
        },
    )


def test_count_abstract_words_uses_requested_latex_split_req_publish_030() -> None:
    tex = r"\begin{abstract}Alpha beta \LaTeX{} token.\end{abstract}"

    assert exp2553.count_abstract_words(tex) == 4
    assert exp2553.count_abstract_words("no abstract") == "not_found"


def test_build_artifact_marks_ready_with_honest_negative_scenario_publish_030(
    tmp_path: Path,
) -> None:
    _write_ready_inputs(tmp_path)
    commands: list[tuple[list[str], Path, int]] = []

    def fake_which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name == "tectonic" else None

    def fake_runner(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
        commands.append((cmd, cwd, timeout))
        return subprocess.CompletedProcess(cmd, 0, stdout="compiled", stderr="")

    artifact = exp2553.build_artifact(
        tmp_path,
        started_epoch=100.0,
        now_epoch=106.25,
        which=fake_which,
        command_runner=fake_runner,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["arxiv_ready"] is True
    assert artifact["submission_package_ready"] is True
    assert artifact["gate_3_phase4_resolved"] is True
    assert artifact["phase4_honest_negative_documented"] is True
    assert artifact["latex_compile_success"] is True
    assert artifact["abstract_word_count"] == 5
    assert artifact["duration_s"] == 6.25
    assert artifact["arxiv_gates"] == {
        "gate_1_phase1_ship": True,
        "gate_2_audit": True,
        "gate_3_phase4_resolved": True,
        "gate_4_auroc_adversarially_verified": True,
    }
    assert artifact["operator_submission_checklist"] == exp2553.OPERATOR_SUBMISSION_CHECKLIST
    assert artifact["submission_attempted"] is False
    assert artifact["credentialed_submission_attempted"] is False
    assert commands == [(["tectonic", "main.tex"], tmp_path / "docs" / "arxiv-paper", 300)]


def test_missing_paper_blocks_before_compile_req_publish_030(tmp_path: Path) -> None:
    def fake_runner(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
        raise AssertionError("compile should not run when main.tex is missing")

    artifact = exp2553.build_artifact(
        tmp_path,
        started_epoch=1.0,
        now_epoch=2.0,
        which=lambda name: "/usr/bin/tectonic" if name == "tectonic" else None,
        command_runner=fake_runner,
    )

    assert artifact["honest_verdict"] == "blocked_paper_not_found"
    assert artifact["arxiv_ready"] is False
    assert artifact["submission_package_ready"] is False
    assert artifact["latex_compile_success"] is False
    assert artifact["duration_s"] == 1.0


def test_checked_in_exp2553_artifact_schema_and_gate_req_publish_030() -> None:
    artifact_path = Path("results/experiment_2553_arxiv_package_v3.json")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    required_fields = {
        "honest_verdict",
        "arxiv_ready",
        "submission_package_ready",
        "gate_3_phase4_resolved",
        "latex_compile_success",
        "abstract_word_count",
        "operator_submission_checklist",
        "preconditions_checked",
        "duration_s",
    }
    assert required_fields <= set(artifact)
    assert artifact["honest_verdict"].startswith(exp2553.TERMINAL_PREFIXES)
    assert artifact["arxiv_ready"] is True
    assert artifact["submission_package_ready"] is True
    assert artifact["gate_3_phase4_resolved"] is True
    assert artifact["latex_compile_success"] is True
    assert isinstance(artifact["abstract_word_count"], int)
    assert artifact["abstract_word_count"] <= exp2553.ABSTRACT_LIMIT_WORDS
    assert artifact["operator_submission_checklist"] == exp2553.OPERATOR_SUBMISSION_CHECKLIST
    assert artifact["submission_attempted"] is False
    assert artifact["credentialed_submission_attempted"] is False
