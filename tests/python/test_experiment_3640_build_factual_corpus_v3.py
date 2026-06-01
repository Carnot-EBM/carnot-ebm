"""Tests for Exp 3640 real-evidence factual corpus v3.

Spec: REQ-REPORT-3640,
      SCENARIO-REPORT-3640,
      SCENARIO-REPORT-3640-DEGENERATE,
      SCENARIO-REPORT-3640-BLOCKED.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scripts import experiment_3640_build_factual_corpus_v3 as mod


def _source_rows(n_pairs: int = 120) -> list[dict[str, str]]:
    cities = [
        ("France", "Paris", "Lyon"),
        ("Japan", "Tokyo", "Osaka"),
        ("Canada", "Ottawa", "Toronto"),
        ("Brazil", "Brasilia", "Rio de Janeiro"),
        ("Kenya", "Nairobi", "Mombasa"),
    ]
    rows: list[dict[str, str]] = []
    for idx in range(n_pairs):
        country, capital, distractor = cities[idx % len(cities)]
        rows.append(
            {
                "knowledge": (
                    f"{country} is a sovereign country. Its national capital is "
                    f"{capital}, and government records identify {capital} as "
                    "the seat of national administration."
                ),
                "question": f"What is the capital city of {country}?",
                "right_answer": capital,
                "hallucinated_answer": (
                    f"The capital city of {country} is {distractor}, according "
                    "to the travel summary."
                ),
            }
        )
    return rows


def _confidence_with_headroom(_question: str, answer: str) -> float:
    """SCENARIO-REPORT-3640: confidence is useful but intentionally imperfect."""

    if answer in {"Paris", "Tokyo", "Ottawa", "Brasilia", "Nairobi"}:
        return {
            "Paris": 0.92,
            "Tokyo": 0.84,
            "Ottawa": 0.72,
            "Brasilia": 0.58,
            "Nairobi": 0.46,
        }[answer]
    country = answer.split(" of ", 1)[-1].split(" is ", 1)[0]
    return {
        "France": 0.38,
        "Japan": 0.50,
        "Canada": 0.62,
        "Brazil": 0.74,
        "Kenya": 0.86,
    }[country]


def _confidence_degenerate(_question: str, answer: str) -> float:
    """SCENARIO-REPORT-3640-DEGENERATE: near-perfect confidence has no headroom."""

    return 0.99 if answer in {"Paris", "Tokyo", "Ottawa", "Brasilia", "Nairobi"} else 0.01


@pytest.mark.parametrize(
    ("case_name", "source_loader", "network_checker", "confidence_fn", "expected_verdict"),
    [
        (
            "validated",
            _source_rows,
            lambda: True,
            _confidence_with_headroom,
            mod.VERDICT_VALIDATED,
        ),
        (
            "degenerate",
            _source_rows,
            lambda: True,
            _confidence_degenerate,
            mod.VERDICT_DEGENERATE,
        ),
        (
            "blocked",
            lambda: [],
            lambda: False,
            _confidence_with_headroom,
            mod.VERDICT_BLOCKED,
        ),
    ],
)
def test_req_report_3640_honest_verdict_matrix(
    tmp_path: Path,
    case_name: str,
    source_loader: Callable[[], list[dict[str, str]]],
    network_checker: Callable[[], bool],
    confidence_fn: Callable[[str, str], float],
    expected_verdict: str,
) -> None:
    """REQ-REPORT-3640: validated, degenerate, and blocked verdicts are all honest."""

    config = mod.CorpusBuildConfig(
        repo_root=tmp_path,
        random_seed=3640,
        max_source_pairs=120,
        min_records=200,
        started_at=10.0,
        clock=lambda: 14.5,
    )

    artifact = mod.run_experiment(
        config=config,
        source_loader=source_loader,
        network_checker=network_checker,
        confidence_fn=confidence_fn,
    )

    assert artifact["honest_verdict"] == expected_verdict
    assert type(artifact["facts_corpus_has_evidence"]) is bool
    assert type(artifact["facts_corpus_validated"]) is bool
    assert type(artifact["placeholder_tokens_rejected"]) is bool
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["duration_s"] == pytest.approx(4.5)

    saved = json.loads((tmp_path / mod.RESULT_PATH).read_text(encoding="utf-8"))
    assert saved == artifact

    corpus_path = tmp_path / mod.CORPUS_PATH
    if case_name == "blocked":
        assert artifact["n_examples"] == 0
        assert artifact["facts_corpus_has_evidence"] is False
        assert artifact["facts_corpus_validated"] is False
        assert not corpus_path.exists()
    else:
        assert corpus_path.is_file()
        rows = [
            json.loads(line)
            for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == 240
        assert {row["is_hallucination"] for row in rows} == {0, 1}
        assert all(row["evidence_passage"] for row in rows)
        assert all(set(row) == set(mod.CORPUS_RECORD_FIELDS) for row in rows)

    if case_name == "validated":
        assert artifact["facts_corpus_has_evidence"] is True
        assert artifact["facts_corpus_validated"] is True
        assert artifact["evidence_independent_of_label"] is True
        assert 0.50 < artifact["confidence_baseline_auroc_on_corpus"] < 0.95
    elif case_name == "degenerate":
        assert artifact["facts_corpus_has_evidence"] is True
        assert artifact["facts_corpus_validated"] is False
        assert artifact["confidence_baseline_auroc_on_corpus"] >= 0.95


def test_scenario_report_3640_local_manifest_rows_are_supported(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3640: cached HaluEval-style manifests remain auditable."""

    manifest = tmp_path / "data" / "eval_manifests" / "halueval_20260522.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "\n"
        + "\n".join(
            [
                json.dumps(
                    {
                        "candidate": "Unused prompt",
                        "label": 0,
                        "prompt": "Context only with no question marker.",
                    }
                ),
                *(
                    json.dumps(
                        {
                            "candidate": "Paris" if label == 0 else "The answer is Lyon.",
                            "label": label,
                            "prompt": (
                                "Context: France has Paris as its national capital.\n"
                                "Question: What is the capital city of France?"
                            ),
                            "reference": "Paris",
                            "source_name": "pminervini/HaluEval:qa:data",
                            "stable_id": f"france-{label}",
                        }
                    )
                    for label in (0, 1)
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = mod.load_local_manifest_rows(tmp_path)

    assert rows == [
        {
            "knowledge": "France has Paris as its national capital.",
            "question": "What is the capital city of France?",
            "right_answer": "Paris",
            "hallucinated_answer": "The answer is Lyon.",
        }
    ]
    assert mod.load_local_manifest_rows(tmp_path / "absent") == []
    assert mod._split_prompt("Standalone context without a question.") == (
        "Standalone context without a question.",
        "",
    )


def test_req_report_3640_default_loader_and_scoring_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-3640: loader fallback, confidence, and AUROC edges are explicit."""

    fetched_rows = _source_rows(1)
    monkeypatch.setattr(mod, "fetch_halueval_qa_rows", lambda _limit: fetched_rows)
    assert (
        mod.default_source_loader(
            mod.CorpusBuildConfig(repo_root=tmp_path, max_source_pairs=1), network_ok=True
        )
        == fetched_rows
    )

    monkeypatch.setattr(
        mod,
        "fetch_halueval_qa_rows",
        lambda _limit: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    manifest = tmp_path / "data" / "eval_manifests" / "halueval_20260522.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "candidate": "Tokyo" if label == 0 else "The answer is Osaka.",
                    "label": label,
                    "prompt": (
                        "Context: Japan has Tokyo as its national capital.\n"
                        "Question: What is the capital city of Japan?"
                    ),
                }
            )
            for label in (0, 1)
        )
        + "\n",
        encoding="utf-8",
    )
    fallback = mod.default_source_loader(
        mod.CorpusBuildConfig(repo_root=tmp_path, max_source_pairs=1), network_ok=True
    )

    assert fallback[0]["right_answer"] == "Tokyo"
    assert mod.default_model_confidence("What is the capital city of Japan?", "Tokyo") > 0.0
    assert mod.default_model_confidence("What is the capital city of Japan?", "") == 0.0
    assert mod.binary_auroc([], []) == 0.0


def test_req_report_3640_builder_rejects_incomplete_and_placeholder_rows() -> None:
    """REQ-REPORT-3640: incomplete rows and toy placeholder rows do not enter v3."""

    source_rows = [
        {
            "knowledge": "Italy has Rome as its national capital.",
            "question": "What is the capital city of Italy?",
            "right_answer": "Rome",
            "hallucinated_answer": "",
        },
        {
            "knowledge": "Germany has Berlin as its national capital.",
            "question": "What is the capital city of Germany?",
            "right_answer": "R17",
            "hallucinated_answer": "The answer is Munich.",
        },
    ]

    records = mod.build_corpus_records(
        source_rows,
        max_source_pairs=2,
        confidence_fn=lambda _question, _answer: 0.5,
    )

    assert records == []
    assert mod.has_placeholder_token("R17") is True


def test_req_report_3640_validation_rejects_label_specific_evidence() -> None:
    """REQ-REPORT-3640: evidence must be paired across both labels, not class-specific."""

    rows = [
        {
            "question": "What city is the capital of France?",
            "answer": "Paris",
            "is_hallucination": 0,
            "evidence_passage": "Correct evidence passage for the Paris answer.",
            "model_confidence": 0.7,
        },
        {
            "question": "What city is the capital of France?",
            "answer": "Lyon",
            "is_hallucination": 1,
            "evidence_passage": "label: hallucinated passage for the Lyon answer.",
            "model_confidence": 0.6,
        },
    ]

    validation = mod.validate_corpus_records(rows, min_records=2)

    assert validation.evidence_independent_of_label is False
    assert validation.facts_corpus_has_evidence is True
    assert validation.facts_corpus_validated is False

    missing_evidence = mod.validate_corpus_records(
        [
            {
                "question": "What city is the capital of Spain?",
                "answer": "Madrid",
                "is_hallucination": 0,
                "evidence_passage": "",
                "model_confidence": 0.7,
            }
        ],
        min_records=1,
    )
    assert missing_evidence.facts_corpus_has_evidence is False
