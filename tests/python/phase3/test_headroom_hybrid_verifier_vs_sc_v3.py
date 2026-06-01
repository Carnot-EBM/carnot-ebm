"""Tests for Exp 3645 headroom hybrid verifier-vs-SC positive control.

Spec: REQ-AR-052, SCENARIO-AR-052-01, SCENARIO-AR-052-02.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.phase3 import headroom_hybrid_verifier_vs_sc_v3 as mod


def _sample(answer: str | None, correct: bool, score: float) -> dict:
    return {
        "text": f"candidate {answer} score={score}",
        "extracted_answer_norm": answer,
        "correct": correct,
        "verifier_fixture_score": score,
        "reasoning_steps": [f"step for {answer}"],
    }


def _record(pid: str, gold: str, rows: list[tuple[str | None, bool, float]]) -> dict:
    return {
        "problem_id": pid,
        "gold_answer_norm": gold,
        "samples": [_sample(answer, correct, score) for answer, correct, score in rows],
    }


def _fixture_scorer(candidate: mod.Candidate) -> float:
    text = candidate.text
    return float(text.rsplit("score=", 1)[1])


def test_req_ar_052_load_multicandidate_records_normalizes_jsonl(tmp_path: Path) -> None:
    """REQ-AR-052: cached JSONL rows become labeled multi-candidate records."""

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(_record("p1", "g", [("x", False, 2.0), ("g", True, 0.1)]))
        + "\n",
        encoding="utf-8",
    )

    records = mod.load_multicandidate_records([corpus], min_candidates=2)

    assert len(records) == 1
    assert records[0].problem_id == "p1"
    assert records[0].gold == "g"
    assert records[0].source_path.endswith("corpus.jsonl")
    assert [candidate.answer for candidate in records[0].candidates] == ["x", "g"]
    assert [candidate.correct for candidate in records[0].candidates] == [False, True]


def test_req_ar_052_normalise_record_edge_cases() -> None:
    """REQ-AR-052: schema normalization handles missing and variant fields."""

    assert mod._pick_present({}, "missing") is None
    assert mod._coerce_bool(0) is False
    assert mod._coerce_bool("correct") is True
    assert mod._coerce_bool("incorrect") is False
    assert mod._coerce_bool("unknown", default=True) is True
    assert mod.normalise_record({"samples": []}, source_path="synthetic", min_candidates=1) is None

    record = mod.normalise_record(
        {
            "gold_answer_norm": "g",
            "samples": [
                "not a sample",
                {"answer": "g", "correct": "yes", "steps": [1]},
                {"answer": "x", "correct": 0},
            ],
        },
        source_path="synthetic",
        min_candidates=2,
    )

    assert record is not None
    assert record.problem_id.startswith("synthetic:")
    assert record.candidates[0].reasoning_steps == ("1",)
    assert [candidate.correct for candidate in record.candidates] == [True, False]
    assert (
        mod.normalise_record(
            {"gold_answer_norm": "g", "samples": [{"answer": None}, {"answer": None}]},
            source_path="synthetic",
            min_candidates=2,
        )
        is None
    )
    assert (
        mod.normalise_record(
            {"gold_answer_norm": "g", "samples": [{"answer": "g"}]},
            source_path="synthetic",
            min_candidates=2,
        )
        is None
    )


def test_req_ar_052_load_records_skips_unusable_lines(tmp_path: Path) -> None:
    """REQ-AR-052: loader ignores missing files, blanks, bad JSON, and short rows."""

    corpus = tmp_path / "mixed.jsonl"
    corpus.write_text(
        "\n"
        "{bad json}\n"
        + json.dumps(_record("short", "g", [("g", True, 0.1)]))
        + "\n"
        + json.dumps(_record("valid", "g", [("x", False, 1.0), ("g", True, 0.1)]))
        + "\n",
        encoding="utf-8",
    )

    records = mod.load_multicandidate_records(
        [tmp_path / "missing.jsonl", corpus],
        min_candidates=2,
    )

    assert [record.problem_id for record in records] == ["valid"]


def test_req_ar_052_empty_majority_and_stats() -> None:
    """REQ-AR-052: empty answer/stat helpers are total functions."""

    empty_record = mod.ProblemRecord(
        problem_id="empty",
        source_path="synthetic",
        gold="g",
        candidates=(mod.Candidate(None, False, "", ()),),
    )

    assert mod.majority_vote_with_support(empty_record) == (None, 0.0)
    assert mod.compute_corpus_stats([]) == mod.CorpusStats(0, 0.0, 0.0, 0.0, 0.0)


def test_req_ar_052_select_stratum_falls_back_when_no_contested() -> None:
    """REQ-AR-052: when no rows are contested, all rows are measured honestly."""

    records = [
        mod.normalise_record(
            _record("p1", "g", [("g", True, 0.1), ("g", True, 0.2)]),
            source_path="synthetic",
            min_candidates=2,
        ),
        mod.normalise_record(
            _record("p2", "z", [("x", False, 0.1), ("x", False, 0.2)]),
            source_path="synthetic",
            min_candidates=2,
        ),
    ]

    selected = mod.select_contested_headroom_stratum(
        [record for record in records if record is not None],
        max_majority_support=0.5,
    )

    assert selected.status == "no_headroom"
    assert len(selected.records) == 2


def test_scenario_ar_052_select_contested_stratum_has_oracle_gt_sc() -> None:
    """SCENARIO-AR-052-01: selected contested rows expose oracle > SC headroom."""

    records = [
        mod.normalise_record(
            _record("p1", "g", [("x", False, 3.0), ("x", False, 2.0), ("g", True, 0.1)]),
            source_path="synthetic",
            min_candidates=2,
        ),
        mod.normalise_record(
            _record("p2", "a", [("a", True, 0.1), ("a", True, 0.2), ("b", False, 3.0)]),
            source_path="synthetic",
            min_candidates=2,
        ),
        mod.normalise_record(
            _record("p3", "z", [("z", True, 0.1), ("z", True, 0.2), ("z", True, 0.3)]),
            source_path="synthetic",
            min_candidates=2,
        ),
    ]

    selected = mod.select_contested_headroom_stratum(
        [record for record in records if record is not None],
        max_majority_support=2 / 3,
    )

    assert selected.status == "headroom"
    assert [record.problem_id for record in selected.records] == ["p1", "p2"]
    assert selected.stats.oracle_accuracy == pytest.approx(1.0)
    assert selected.stats.sc_accuracy == pytest.approx(0.5)
    assert selected.stats.oracle_minus_sc_headroom == pytest.approx(0.5)


def test_scenario_ar_052_evaluate_reports_verifier_and_hybrid_lift() -> None:
    """SCENARIO-AR-052-02: verifier and hybrid lift include paired CIs."""

    records = [
        mod.normalise_record(
            _record("p1", "g", [("x", False, 4.0), ("x", False, 3.0), ("g", True, 0.1)]),
            source_path="synthetic",
            min_candidates=2,
        ),
        mod.normalise_record(
            _record("p2", "a", [("a", True, 0.1), ("a", True, 0.2), ("b", False, 4.0)]),
            source_path="synthetic",
            min_candidates=2,
        ),
        mod.normalise_record(
            _record("p3", "d", [("c", False, 4.0), ("d", True, 0.1), ("c", False, 3.0)]),
            source_path="synthetic",
            min_candidates=2,
        ),
    ]
    clean = [record for record in records if record is not None]

    result = mod.evaluate_headroom_stratum(
        clean,
        scorer=_fixture_scorer,
        seed=123,
        n_boot=200,
        verifier_temperature=0.5,
    )

    assert result.oracle_minus_sc_headroom > 0.0
    assert result.verifier_reranked_accuracy > result.sc_accuracy
    assert result.hybrid_accuracy >= result.verifier_reranked_accuracy
    assert result.verifier_beats_sc_where_headroom_exists is True
    assert "ci95" in result.verifier_over_sc_lift
    assert len(result.verifier_over_sc_lift["ci95"]) == 2


def test_req_ar_052_artifact_positive_terminal_fields() -> None:
    """REQ-AR-052: positive artifacts contain every required value field."""

    records = [
        mod.normalise_record(
            _record("p1", "g", [("x", False, 4.0), ("x", False, 3.0), ("g", True, 0.1)]),
            source_path="synthetic",
            min_candidates=2,
        ),
        mod.normalise_record(
            _record("p2", "a", [("a", True, 0.55), ("a", True, 0.55), ("b", False, 0.1)]),
            source_path="synthetic",
            min_candidates=2,
        ),
        mod.normalise_record(
            _record("p3", "d", [("c", False, 4.0), ("d", True, 0.1), ("c", False, 3.0)]),
            source_path="synthetic",
            min_candidates=2,
        ),
    ]
    result = mod.evaluate_headroom_stratum(
        [record for record in records if record is not None],
        scorer=_fixture_scorer,
        seed=456,
        n_boot=100,
        verifier_temperature=0.5,
    )

    artifact = mod.build_result_artifact(
        result=result,
        selected_status="headroom",
        corpus_paths=["synthetic"],
        duration_s=1.25,
    )

    required = {
        "honest_verdict",
        "inference_substrate",
        "oracle_minus_sc_headroom",
        "sc_accuracy",
        "verifier_reranked_accuracy",
        "verifier_over_sc_lift",
        "hybrid_accuracy",
        "hybrid_beats_both",
        "verifier_beats_sc_where_headroom_exists",
        "n_examples",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    }
    assert required <= artifact.keys()
    assert artifact["honest_verdict"] == (
        "complete: verifier_beats_sc_on_headroom_corpus_hybrid_wins_under_budget"
    )
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["acceptance_gate"]["required_fields_present"] is True


def test_req_ar_052_blocked_artifact_when_no_multicandidate_corpus() -> None:
    """REQ-AR-052: missing usable candidates returns the blocked terminal verdict."""

    artifact = mod.build_blocked_artifact(
        verdict="complete: blocked_no_multicandidate_corpus",
        corpus_paths=["missing.jsonl"],
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "complete: blocked_no_multicandidate_corpus"
    assert artifact["n_examples"] == 0
    assert artifact["oracle_minus_sc_headroom"] is None
    assert artifact["acceptance_gate"]["required_fields_present"] is True


def test_req_ar_052_no_headroom_artifact_is_uninformative() -> None:
    """REQ-AR-052: oracle <= SC maps to no-headroom, not verifier failure."""

    no_headroom = mod.CorpusStats(
        n_examples=2,
        oracle_accuracy=1.0,
        sc_accuracy=1.0,
        oracle_minus_sc_headroom=0.0,
        mean_candidates_per_example=3.0,
    )

    artifact = mod.build_no_headroom_artifact(
        stats=no_headroom,
        corpus_paths=["synthetic"],
        duration_s=0.2,
    )

    assert artifact["honest_verdict"] == (
        "complete: no_headroom_corpus_found_verifier_study_uninformative"
    )
    assert artifact["oracle_minus_sc_headroom"] == 0.0
    assert artifact["verifier_over_sc_lift"] is None


def test_req_ar_052_run_experiment_writes_positive_artifact(tmp_path: Path) -> None:
    """REQ-AR-052: run_experiment writes the scored artifact when headroom exists."""

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _record("p1", "g", [("x", False, 4.0), ("x", False, 3.0), ("g", True, 0.1)]),
                _record("p2", "a", [("a", True, 0.55), ("a", True, 0.55), ("b", False, 0.1)]),
                _record("p3", "d", [("c", False, 4.0), ("d", True, 0.1), ("c", False, 3.0)]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "results" / "artifact.json"

    artifact = mod.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        corpus_paths=[Path("corpus.jsonl")],
        min_candidates=2,
        n_boot=100,
        scorer=_fixture_scorer,
    )

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == (
        "complete: verifier_beats_sc_on_headroom_corpus_hybrid_wins_under_budget"
    )


def test_req_ar_052_run_experiment_blocked_and_no_headroom(tmp_path: Path) -> None:
    """REQ-AR-052: run_experiment covers blocked and no-headroom terminal paths."""

    blocked = mod.run_experiment(
        repo_root=tmp_path,
        corpus_paths=[Path("missing.jsonl")],
        min_candidates=2,
        scorer=_fixture_scorer,
    )
    assert blocked["honest_verdict"] == "complete: blocked_no_multicandidate_corpus"

    corpus = tmp_path / "no_headroom.jsonl"
    corpus.write_text(
        json.dumps(_record("p1", "g", [("g", True, 0.1), ("g", True, 0.2)])) + "\n",
        encoding="utf-8",
    )
    no_headroom = mod.run_experiment(
        repo_root=tmp_path,
        corpus_paths=[Path("no_headroom.jsonl")],
        min_candidates=2,
        max_majority_support=1.0,
        scorer=_fixture_scorer,
    )
    assert no_headroom["honest_verdict"] == (
        "complete: no_headroom_corpus_found_verifier_study_uninformative"
    )


def test_req_ar_052_default_scorer_uses_fover_energy(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-AR-052: default scorer wraps the FoVer candidate energy once."""

    import carnot.phase3.p01_trained_energy_reranker as reranker

    marker = object()
    monkeypatch.setattr(reranker, "_Verifiers", lambda: marker)
    monkeypatch.setattr(
        reranker,
        "fover_candidate_energy",
        lambda text, verifiers: 7.0 if text == "target" and verifiers is marker else -1.0,
    )

    scorer = mod.make_default_scorer()

    assert scorer(mod.Candidate("a", True, "target", ())) == 7.0
