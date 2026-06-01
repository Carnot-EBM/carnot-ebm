"""Tests for Exp 3672 ensemble selection where SC is weak.

Spec: REQ-VERIFY-3672, SCENARIO-VERIFY-3672.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verify import ensemble_selection_sc_weak_3672 as mod


def _sample(answer: str | None, correct: bool, score: float, confidence: float | None) -> dict:
    sample = {
        "answer": answer,
        "correct": correct,
        "text": f"candidate answer={answer} score={score}",
    }
    if confidence is None:
        sample["token_logprobs"] = [-1.0, -3.0]
    else:
        sample["mean_token_logprob"] = confidence
    return sample


def _record(
    problem_id: str,
    gold: str,
    rows: list[tuple[str | None, bool, float, float | None]],
) -> dict:
    return {
        "problem_id": problem_id,
        "gold": gold,
        "samples": [_sample(*row) for row in rows],
    }


def _fixture_scorer(candidate: mod.Candidate) -> float:
    return float(candidate.text.rsplit("score=", 1)[1])


def _positive_rows() -> list[dict]:
    return [
        _record(
            "p1",
            "g",
            [
                ("x", False, 4.0, 0.9),
                ("x", False, 3.0, 0.8),
                ("g", True, 0.1, 0.1),
            ],
        ),
        _record(
            "p2",
            "a",
            [
                ("b", False, 4.0, 0.9),
                ("b", False, 3.0, 0.8),
                ("a", True, 0.1, 0.1),
            ],
        ),
        _record(
            "p3",
            "c",
            [
                ("z", False, 4.0, 0.9),
                ("z", False, 4.0, 0.9),
                ("c", True, 0.1, 0.1),
            ],
        ),
    ]


@pytest.mark.parametrize(
    (
        "blocked",
        "positive_control_valid",
        "ensemble_adds_value",
        "expected_category",
        "expected_verdict",
    ),
    [
        pytest.param(
            False,
            True,
            True,
            "ensemble_adds_selection_value",
            mod.POSITIVE_VERDICT,
            id="ensemble_adds_selection_value",
        ),
        pytest.param(
            False,
            True,
            False,
            "no_value_even_with_headroom",
            mod.NEGATIVE_VERDICT,
            id="no_value_even_with_headroom",
        ),
        pytest.param(
            False,
            False,
            False,
            "no_selectable_headroom",
            mod.NO_HEADROOM_VERDICT,
            id="no_selectable_headroom",
        ),
        pytest.param(
            True,
            False,
            False,
            "blocked",
            mod.BLOCKED_VERDICT,
            id="blocked",
        ),
    ],
)
def test_scenario_verify_3672_parametrizes_honest_outcomes(
    blocked: bool,
    positive_control_valid: bool,
    ensemble_adds_value: bool,
    expected_category: str,
    expected_verdict: str,
) -> None:
    """SCENARIO-VERIFY-3672: outcome classification is not one success string."""

    classification = mod.classify_outcome(
        blocked=blocked,
        positive_control_valid=positive_control_valid,
        ensemble_adds_selection_value=ensemble_adds_value,
    )

    assert classification.category == expected_category
    assert classification.terminal_verdict == expected_verdict
    assert type(classification.ensemble_adds_selection_value_sc_weak) is bool


def test_req_verify_3672_normalizes_records_and_selects_sc_weak_headroom() -> None:
    """REQ-VERIFY-3672: cached rows become an SC-weak headroom stratum."""

    records = [
        mod.normalise_record(row, source_path="synthetic.jsonl", min_candidates=3)
        for row in _positive_rows()
    ]
    clean = [record for record in records if record is not None]

    selected = mod.select_sc_weak_regime(
        clean,
        min_examples=3,
        max_sc_accuracy=0.40,
        max_majority_supports=(2 / 3,),
    )

    assert selected.status == "sc_weak_headroom"
    assert selected.stats.n_examples == 3
    assert selected.stats.sc_accuracy == pytest.approx(0.0)
    assert selected.stats.oracle_bestofn_accuracy == pytest.approx(1.0)
    assert selected.max_majority_support == pytest.approx(2 / 3)
    assert clean[0].candidates[0].confidence == pytest.approx(0.9)
    assert clean[0].candidates[2].answer == "g"


def test_req_verify_3672_loader_skips_unusable_rows_and_derives_confidence(tmp_path: Path) -> None:
    """REQ-VERIFY-3672: loader handles bad JSON, missing files, and confidence fallback."""

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n"
        "{bad json}\n"
        + json.dumps({"samples": []})
        + "\n"
        + json.dumps(_record("p1", "g", [(None, False, 1.0, None), (None, False, 2.0, None)]))
        + "\n"
        + json.dumps(_record("p2", "g", [("x", False, 2.0, None), ("g", True, 0.1, None)]))
        + "\n",
        encoding="utf-8",
    )

    records = mod.load_multicandidate_records(
        [tmp_path / "missing.jsonl", corpus],
        min_candidates=2,
    )

    assert [record.problem_id for record in records] == ["p2"]
    assert records[0].candidates[0].confidence == pytest.approx(-2.0)
    assert mod.normalise_record("not a dict", source_path="x", min_candidates=1) is None
    assert mod._pick_present({}, "missing") is None
    assert mod._coerce_bool("correct") is True
    assert mod._coerce_bool("incorrect") is False
    assert mod._coerce_bool("unknown", default=True) is True


def test_req_verify_3672_edge_helpers_and_validation_failures() -> None:
    """REQ-VERIFY-3672: defensive helpers and schema guards are covered."""

    assert mod._coerce_bool(0) is False
    assert mod._candidate_confidence({}) is None
    assert (
        mod.normalise_record(
            {
                "gold": "g",
                "samples": [
                    "not a sample",
                    {"answer": "g", "text": "candidate answer=g score=0.1"},
                    {"answer": "x", "text": "candidate answer=x score=2.0"},
                ],
            },
            source_path="synthetic.jsonl",
            min_candidates=2,
        )
        is not None
    )
    assert (
        mod.normalise_record(
            {"gold": "g", "samples": [{"answer": "g"}]},
            source_path="synthetic.jsonl",
            min_candidates=3,
        )
        is None
    )

    empty_answer_record = mod.ProblemRecord(
        problem_id="empty",
        source_path="synthetic",
        gold="g",
        candidates=(mod.Candidate(None, False, "", None),),
    )
    assert mod.majority_vote_with_support(empty_answer_record) == (None, 0.0)
    assert mod.compute_regime_stats([]) == mod.RegimeStats(0, 0.0, 0.0, 0.0, 0.0)

    one_record = mod.normalise_record(
        _positive_rows()[0], source_path="synthetic", min_candidates=3
    )
    assert one_record is not None
    no_selectable = mod.select_sc_weak_regime(
        [one_record],
        min_examples=2,
        max_sc_accuracy=0.0,
        max_majority_supports=(0.0, 1.0),
    )
    assert no_selectable.status == "no_selectable_headroom"

    assert (
        mod._select_best_answer(
            [
                mod.Candidate(None, False, "", 1.0),
                mod.Candidate("a", True, "", None),
            ],
            lambda candidate: candidate.confidence,
            higher_is_better=True,
        )
        is None
    )
    assert mod._minmax_good([], low_is_good=True) == []
    assert mod._minmax_good([2.0, 2.0], low_is_good=False) == [0.0, 0.0]
    assert (
        mod.fusion_confidence_energy_answer([mod.Candidate(None, False, "", None)], [1.0]) is None
    )

    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact({})

    valid = mod.build_blocked_artifact(corpus_paths=["missing.jsonl"], duration_s=0.1)
    missing_principle = dict(valid)
    missing_principle["field_principles"] = {}
    with pytest.raises(AssertionError, match="missing field principles"):
        mod.validate_artifact(missing_principle)

    bad_verdict = dict(valid)
    bad_verdict["honest_verdict"] = "complete: unexpected"
    with pytest.raises(AssertionError, match="unknown terminal verdict"):
        mod.validate_artifact(bad_verdict)

    bad_core_bool = dict(valid)
    bad_core_bool["ensemble_adds_selection_value_sc_weak"] = "false"
    with pytest.raises(AssertionError, match="bare bool"):
        mod.validate_artifact(bad_core_bool)

    bad_pc_bool = dict(valid)
    bad_pc_bool["positive_control_valid"] = "false"
    with pytest.raises(AssertionError, match="positive_control_valid"):
        mod.validate_artifact(bad_pc_bool)

    bad_gate = dict(valid)
    bad_gate["acceptance_gate"] = {"required_fields_present": False}
    with pytest.raises(AssertionError, match="acceptance gate"):
        mod.validate_artifact(bad_gate)


def test_scenario_verify_3672_evaluates_positive_selection_value() -> None:
    """SCENARIO-VERIFY-3672: ensemble selection can beat SC and confidence."""

    records = [
        mod.normalise_record(row, source_path="synthetic.jsonl", min_candidates=3)
        for row in _positive_rows()
    ]
    selected = mod.select_sc_weak_regime(
        [record for record in records if record is not None],
        min_examples=3,
        max_sc_accuracy=0.40,
        max_majority_supports=(2 / 3,),
    )

    result = mod.evaluate_selection_regime(
        selected.records,
        energy_scorer=_fixture_scorer,
        seed=123,
        n_boot=200,
    )

    assert result.sc_accuracy == pytest.approx(0.0)
    assert result.oracle_bestofn_accuracy == pytest.approx(1.0)
    assert result.ensemble_selection_accuracy == pytest.approx(1.0)
    assert result.confidence_selection_accuracy == pytest.approx(0.0)
    assert result.flip_count == 3
    assert result.positive_control_valid is True
    assert result.ensemble_adds_selection_value_sc_weak is True
    assert result.ensemble_vs_sc_delta_ci["ci95"][0] > 0.0
    assert 0.0 <= result.fusion_selection_accuracy <= 1.0


def test_req_verify_3672_artifacts_cover_positive_negative_no_headroom_and_blocked(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3672: terminal artifacts preserve all required fields."""

    records = [
        mod.normalise_record(row, source_path="synthetic.jsonl", min_candidates=3)
        for row in _positive_rows()
    ]
    selected = mod.select_sc_weak_regime(
        [record for record in records if record is not None],
        min_examples=3,
        max_sc_accuracy=0.40,
        max_majority_supports=(2 / 3,),
    )
    positive_result = mod.evaluate_selection_regime(
        selected.records,
        energy_scorer=_fixture_scorer,
        seed=456,
        n_boot=100,
    )
    positive = mod.build_measured_artifact(
        result=positive_result,
        selected=selected,
        corpus_paths=["synthetic.jsonl"],
        duration_s=1.25,
    )

    mod.validate_artifact(positive)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(positive)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(positive["field_principles"])
    assert positive["honest_verdict"] == mod.POSITIVE_VERDICT
    assert type(positive["ensemble_adds_selection_value_sc_weak"]) is bool
    assert positive["acceptance_gate"]["positive_control_valid"] is True

    negative_result = mod.EvaluationResult(
        **{
            **positive_result.__dict__,
            "ensemble_selection_accuracy": 0.0,
            "ensemble_vs_sc_delta_ci": {
                "comparison": "ensemble_energy_vs_self_consistency",
                "delta": -1 / 3,
                "ci95": [-2 / 3, 0.0],
                "mcnemar_exact_p": 1.0,
            },
            "ensemble_vs_confidence_delta_ci": {
                "comparison": "ensemble_energy_vs_confidence",
                "delta": 0.0,
                "ci95": [-1 / 3, 1 / 3],
                "mcnemar_exact_p": 1.0,
            },
            "ensemble_adds_selection_value_sc_weak": False,
        }
    )
    negative = mod.build_measured_artifact(
        result=negative_result,
        selected=selected,
        corpus_paths=["synthetic.jsonl"],
        duration_s=1.25,
    )
    mod.validate_artifact(negative)
    assert negative["honest_verdict"] == mod.NEGATIVE_VERDICT
    assert negative["positive_control_valid"] is True

    no_headroom = mod.build_no_headroom_artifact(
        stats=mod.RegimeStats(
            n_examples=3,
            sc_accuracy=1.0,
            oracle_bestofn_accuracy=1.0,
            oracle_minus_sc_headroom=0.0,
            mean_candidates_per_example=3.0,
        ),
        selected_status="no_selectable_headroom",
        corpus_paths=["synthetic.jsonl"],
        duration_s=0.5,
    )
    mod.validate_artifact(no_headroom)
    assert no_headroom["honest_verdict"] == mod.NO_HEADROOM_VERDICT
    assert no_headroom["positive_control_valid"] is False

    blocked = mod.build_blocked_artifact(corpus_paths=["missing.jsonl"], duration_s=0.1)
    mod.validate_artifact(blocked)
    assert blocked["honest_verdict"] == mod.BLOCKED_VERDICT
    assert blocked["n_examples"] == 0
    assert blocked["acceptance_gate"]["required_fields_present"] is True
    assert tmp_path.exists()


def test_req_verify_3672_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-3672: run_experiment writes the selected SC-weak artifact."""

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _positive_rows()),
        encoding="utf-8",
    )
    output = tmp_path / "results" / "artifact.json"

    artifact = mod.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        corpus_paths=[Path("corpus.jsonl")],
        min_candidates=3,
        min_examples=3,
        max_sc_accuracy=0.40,
        max_majority_supports=(2 / 3,),
        energy_scorer=_fixture_scorer,
        n_boot=100,
    )

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == mod.POSITIVE_VERDICT


def test_req_verify_3672_run_experiment_blocks_and_reports_no_headroom(tmp_path: Path) -> None:
    """REQ-VERIFY-3672: run_experiment covers blocked and no-headroom paths."""

    blocked = mod.run_experiment(
        repo_root=tmp_path,
        corpus_paths=[Path("missing.jsonl")],
        min_candidates=2,
        energy_scorer=_fixture_scorer,
    )
    assert blocked["honest_verdict"] == mod.BLOCKED_VERDICT

    no_headroom_corpus = tmp_path / "no_headroom.jsonl"
    no_headroom_corpus.write_text(
        json.dumps(
            _record(
                "p1",
                "g",
                [
                    ("g", True, 0.1, 0.9),
                    ("g", True, 0.2, 0.8),
                    ("x", False, 1.0, 0.1),
                ],
            )
        )
        + "\n",
        encoding="utf-8",
    )
    no_headroom = mod.run_experiment(
        repo_root=tmp_path,
        corpus_paths=[Path("no_headroom.jsonl")],
        min_candidates=3,
        min_examples=1,
        max_sc_accuracy=1.0,
        max_majority_supports=(1.0,),
        energy_scorer=_fixture_scorer,
    )
    assert no_headroom["honest_verdict"] == mod.NO_HEADROOM_VERDICT


def test_req_verify_3672_default_scorer_uses_fover_energy(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-3672: default scorer wraps the FoVer candidate energy."""

    import carnot.phase3.p01_trained_energy_reranker as reranker

    marker = object()
    monkeypatch.setattr(reranker, "_Verifiers", lambda: marker)
    monkeypatch.setattr(
        reranker,
        "fover_candidate_energy",
        lambda text, verifiers: 7.0 if text == "target" and verifiers is marker else -1.0,
    )

    scorer = mod.make_default_energy_scorer()

    assert scorer(mod.Candidate("a", True, "target", 0.0)) == 7.0
