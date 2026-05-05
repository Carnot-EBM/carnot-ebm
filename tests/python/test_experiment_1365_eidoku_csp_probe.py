"""Tests for Exp 1365 Eidoku CSP neuro-symbolic verification probe.

Spec: REQ-VERIFY-1365, SCENARIO-VERIFY-1365
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval.eidoku_csp_probe import (
    REQUIRED_ARTIFACT_FIELDS,
    FoVerCSPCase,
    build_artifact,
    extract_reasoning_steps,
    load_fover_cases,
    run_experiment,
    score_case,
    tie_aware_auroc,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_load_fover_cases_balances_jsonl_labels(tmp_path: Path) -> None:
    """REQ-VERIFY-1365: local FoVer rows with labels become balanced cases."""

    corpus = tmp_path / "fover.jsonl"
    _write_jsonl(
        corpus,
        [
            {"question_id": "q1", "step_text": "1 + 1 = 2", "label": "correct"},
            {"question_id": "q2", "step_text": "2 + 2 = 5", "label": "incorrect"},
            {"question_id": "q3", "step_text": "3 + 3 = 6", "label": "correct"},
            {"question_id": "q4", "step_text": "4 + 4 = 9", "label": "incorrect"},
        ],
    )

    cases = load_fover_cases(corpus, limit=4)

    assert len(cases) == 4
    assert {case.label for case in cases} == {0, 1}
    assert all(case.steps for case in cases)


def test_score_case_detects_symbolic_arithmetic_violation() -> None:
    """SCENARIO-VERIFY-1365: symbolic proxy rejects false arithmetic claims."""

    text = "Step 1: Compute the total. 2 + 2 = 5. Therefore, answer = 5."
    case = FoVerCSPCase(
        case_id="bad",
        question="",
        response=text,
        steps=extract_reasoning_steps(text),
        label=1,
    )

    scored = score_case(case)

    assert scored.symbolic_cost == pytest.approx(1.0)
    assert scored.symbolic_entailed is False
    assert scored.csp_feasible is False


def test_score_case_accepts_connected_consistent_chain() -> None:
    """REQ-VERIFY-1365: connected, geometrically close, entailed steps are feasible."""

    text = "Step 1: total = 4. Step 2: total = 4 because 2 + 2 = 4."
    case = FoVerCSPCase(
        case_id="good",
        question="",
        response=text,
        steps=["Step 1: total = 4.", "Step 2: total = 4 because 2 + 2 = 4."],
        label=0,
    )

    scored = score_case(case)

    assert scored.structural_cost < 0.35
    assert scored.geometric_consistent is True
    assert scored.symbolic_entailed is True
    assert scored.csp_feasible is True


def test_build_artifact_required_fields_and_viability_gate() -> None:
    """REQ-VERIFY-1365: artifact schema and viability gate are deterministic."""

    cases = [
        FoVerCSPCase("good-1", "", "2 + 2 = 4", ["2 + 2 = 4"], 0),
        FoVerCSPCase("good-2", "", "3 + 3 = 6", ["3 + 3 = 6"], 0),
        FoVerCSPCase("bad", "", "2 + 2 = 5", ["2 + 2 = 5"], 1),
    ]
    scores = [score_case(case) for case in cases]

    artifact = build_artifact(
        cases,
        scores,
        corpus_path=Path("local_fover.jsonl"),
        ising_scores=[0.0, 0.0, 1.0],
        kan_scores=[0.0, 0.0, 1.0],
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["corpus_cases_used"] == 3
    assert artifact["eidoku_csp_viable"] is True
    assert artifact["eidoku_auroc_proxy"] == pytest.approx(1.0)


def test_run_experiment_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1365: run_experiment persists the complete JSON artifact."""

    corpus = tmp_path / "fover.jsonl"
    out = tmp_path / "experiment_1365.json"
    _write_jsonl(
        corpus,
        [
            {"question_id": "q1", "step_text": "1 + 1 = 2", "label": "correct"},
            {"question_id": "q2", "step_text": "2 + 2 = 5", "label": "incorrect"},
        ],
    )

    artifact = run_experiment(
        corpus_path=corpus,
        output_path=out,
        limit=2,
        use_kan_adapter=False,
    )

    persisted = json.loads(out.read_text(encoding="utf-8"))
    assert persisted == artifact
    assert persisted["status"] == "complete"
    assert persisted["corpus_cases_used"] == 2


def test_tie_aware_auroc_handles_ties_and_single_class() -> None:
    """REQ-VERIFY-1365: AUROC proxy is stable for tiny or tied inputs."""

    assert tie_aware_auroc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert tie_aware_auroc([1, 1], [0.2, 0.8]) == pytest.approx(0.5)
