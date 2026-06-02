"""Tests for Exp 3694 proper selection-gap rediagnosis.

Spec: REQ-VERIFY-3694, SCENARIO-VERIFY-3694.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verify import selection_gap_proper_rediagnosis_3694 as mod


def _candidate(
    answer: str,
    gold: str,
    *,
    energy: float,
    components: tuple[float, float],
    uncertainty: float,
    confidence: float,
) -> mod.CandidateEvidence:
    return mod.CandidateEvidence(
        answer=answer,
        correct=answer == gold,
        text=f"candidate answer={answer}",
        confidence=confidence,
        energy=energy,
        uncertainty=uncertainty,
        components=components,
    )


def _record(
    idx: int,
    *,
    low_energy_wrong: bool = False,
    high_uncertainty_correct: bool = False,
    high_confidence_wrong: bool = False,
) -> mod.ProblemEvidence:
    gold = f"g{idx}"
    wrong_energy = -0.1 if low_energy_wrong else (0.1 if high_confidence_wrong else 1.0)
    correct_uncertainty = 10.0 if high_uncertainty_correct else 0.0
    correct_confidence = 0.05 if high_confidence_wrong else 0.40
    wrong_confidence = 0.99 if high_confidence_wrong else 0.20
    wrong_components = (-10.0, 10.0) if low_energy_wrong else (1.0, 1.0)
    return mod.ProblemEvidence(
        problem_id=f"p{idx}",
        source_path="synthetic.jsonl",
        gold=gold,
        candidates=(
            _candidate(
                "m",
                gold,
                energy=wrong_energy,
                components=wrong_components,
                uncertainty=0.0,
                confidence=wrong_confidence,
            ),
            _candidate(
                "m",
                gold,
                energy=1.1,
                components=(1.2, 1.1),
                uncertainty=0.0,
                confidence=0.18,
            ),
            _candidate(
                gold,
                gold,
                energy=0.0,
                components=(0.0, 0.0),
                uncertainty=correct_uncertainty,
                confidence=correct_confidence,
            ),
            _candidate(
                "d",
                gold,
                energy=1.4,
                components=(1.4, 1.4),
                uncertainty=0.0,
                confidence=0.10,
            ),
        ),
    )


def _recovering_records() -> list[mod.ProblemEvidence]:
    records = []
    for idx in range(20):
        records.append(
            _record(
                idx,
                low_energy_wrong=idx < 4,
                high_uncertainty_correct=4 <= idx < 12,
                high_confidence_wrong=9 <= idx < 15,
            )
        )
    return records


def _to_raw_record(record: mod.ProblemEvidence) -> dict:
    return {
        "problem_id": record.problem_id,
        "gold": record.gold,
        "samples": [
            {
                "answer": candidate.answer,
                "correct": candidate.correct,
                "text": candidate.text,
                "mean_token_logprob": candidate.confidence,
                "verifier_energy": candidate.energy,
                "energy_uncertainty": candidate.uncertainty,
                "energy_components": list(candidate.components),
            }
            for candidate in record.candidates
        ],
    }


@pytest.mark.parametrize(
    (
        "blocked",
        "positive_control_valid",
        "non_degeneracy_assert",
        "selection_gap_closed",
        "expected_category",
        "expected_verdict",
    ),
    [
        pytest.param(
            False,
            True,
            True,
            True,
            "fix_recovers_selection_value",
            mod.CLOSED_VERDICT,
            id="fix_recovers_selection_value",
        ),
        pytest.param(
            False,
            True,
            True,
            False,
            "decoupling_fundamental_no_fix_helps",
            mod.FUNDAMENTAL_VERDICT,
            id="decoupling_fundamental_no_fix_helps",
        ),
        pytest.param(
            True,
            False,
            False,
            False,
            "blocked",
            mod.BLOCKED_VERDICT,
            id="blocked",
        ),
    ],
)
def test_scenario_verify_3694_parametrizes_honest_outcomes(
    blocked: bool,
    positive_control_valid: bool,
    non_degeneracy_assert: bool,
    selection_gap_closed: bool,
    expected_category: str,
    expected_verdict: str,
) -> None:
    """SCENARIO-VERIFY-3694: honest outcomes are not one success string."""

    classification = mod.classify_outcome(
        blocked=blocked,
        positive_control_valid=positive_control_valid,
        non_degeneracy_assert=non_degeneracy_assert,
        selection_gap_closed=selection_gap_closed,
    )

    assert classification.category == expected_category
    assert classification.terminal_verdict == expected_verdict
    assert type(classification.selection_gap_closed) is bool


def test_req_verify_3694_reproduces_discrimination_before_gap_closure() -> None:
    """REQ-VERIFY-3694: fixes run only after AUROC and non-degeneracy pass."""

    result = mod.evaluate_gap(
        _recovering_records(),
        seed=123,
        n_boot=300,
        bootstrap_rounds=81,
    )

    assert result.per_candidate_auroc is not None
    assert result.per_candidate_auroc >= 0.85
    assert result.sc_selection_accuracy == pytest.approx(0.0)
    assert result.oracle_bestofn_accuracy == pytest.approx(1.0)
    assert result.selection_accuracy_per_question_normalized == pytest.approx(1.0)
    assert result.selection_accuracy_pessimistic_lcb == pytest.approx(0.40)
    assert result.selection_accuracy_bootstrapped == pytest.approx(0.75)
    assert result.selection_accuracy_self_certainty_fusion == pytest.approx(0.70)
    assert result.per_fix_flip_counts == {
        "per_question_normalized": 20,
        "pessimistic_lcb": 8,
        "bootstrapped": 15,
        "self_certainty_fusion": 14,
    }
    assert result.non_degeneracy_assert is True
    assert result.positive_control_valid is True
    assert result.best_fix_vs_sc_delta_ci["method"] == "per_question_normalized"
    assert result.best_fix_vs_sc_delta_ci["ci95"][0] > 0.0
    assert result.selection_gap_closed is True


def test_req_verify_3694_artifacts_validate_required_fields_and_bare_bools() -> None:
    """REQ-VERIFY-3694: terminal artifacts preserve the required schema."""

    result = mod.evaluate_gap(
        _recovering_records(),
        seed=456,
        n_boot=200,
        bootstrap_rounds=81,
    )
    artifact = mod.build_measured_artifact(
        result=result,
        corpus_paths=["synthetic.jsonl"],
        duration_s=1.25,
    )

    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"] == mod.CLOSED_VERDICT
    assert type(artifact["selection_gap_closed"]) is bool
    assert type(artifact["positive_control_valid"]) is bool
    assert type(artifact["non_degeneracy_assert"]) is bool
    assert type(artifact["adversarial_verify_clean"]) is bool
    assert artifact["selection_gap_closed"] is True

    blocked = mod.build_blocked_artifact(
        corpus_paths=["missing.jsonl"],
        duration_s=0.1,
        block_reason="cached per-candidate energy unavailable",
    )
    mod.validate_artifact(blocked)
    assert blocked["honest_verdict"] == mod.BLOCKED_VERDICT
    assert blocked["selection_gap_closed"] is False
    assert blocked["n_examples"] == 0

    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact({})

    bad_principles = dict(blocked)
    bad_principles["field_principles"] = {}
    with pytest.raises(AssertionError, match="missing field principles"):
        mod.validate_artifact(bad_principles)

    bad_bool = dict(blocked)
    bad_bool["selection_gap_closed"] = "false"
    with pytest.raises(AssertionError, match="selection_gap_closed"):
        mod.validate_artifact(bad_bool)

    bad_substrate = dict(blocked)
    bad_substrate["inference_substrate"] = "GGUF live model"
    with pytest.raises(AssertionError, match="compute-bound marker"):
        mod.validate_artifact(bad_substrate)

    bad_gate = dict(blocked)
    bad_gate["acceptance_gate"] = {"required_fields_present": False}
    with pytest.raises(AssertionError, match="acceptance gate"):
        mod.validate_artifact(bad_gate)


def test_req_verify_3694_loader_requires_cached_energy_and_run_blocks(tmp_path: Path) -> None:
    """REQ-VERIFY-3694: missing cached energy blocks instead of rescoring text."""

    raw_without_energy = {
        "problem_id": "no-energy",
        "gold": "g",
        "samples": [
            {"answer": "m", "text": "wrong", "mean_token_logprob": -0.1},
            {"answer": "m", "text": "wrong again", "mean_token_logprob": -0.2},
            {"answer": "g", "text": "correct", "mean_token_logprob": -0.3},
            {"answer": "d", "text": "decoy", "mean_token_logprob": -0.4},
        ],
    }
    assert (
        mod.normalise_cached_record(
            raw_without_energy,
            source_path="synthetic.jsonl",
            min_candidates=4,
        )
        is None
    )

    raw_with_energy = dict(raw_without_energy)
    raw_with_energy["samples"] = [
        {**sample, "verifier_energy": float(idx), "energy_components": [idx, idx + 1]}
        for idx, sample in enumerate(raw_without_energy["samples"])
    ]
    record = mod.normalise_cached_record(
        raw_with_energy,
        source_path="synthetic.jsonl",
        min_candidates=4,
    )
    assert record is not None
    assert record.candidates[0].energy == pytest.approx(0.0)
    assert record.candidates[0].components == (0.0, 1.0)
    assert record.candidates[0].uncertainty == pytest.approx(0.5)

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps(raw_without_energy) + "\n", encoding="utf-8")
    output = tmp_path / "result.json"

    artifact = mod.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        corpus_paths=[Path("corpus.jsonl")],
        min_candidates=4,
        min_examples=1,
        n_boot=50,
    )

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert "cached per-candidate energy" in artifact["block_reason"]


def test_req_verify_3694_run_experiment_writes_measured_and_low_auroc_blocks(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3694: run_experiment covers measured and failed-AUROC paths."""

    corpus = tmp_path / "measured.jsonl"
    corpus.write_text(
        "".join(json.dumps(_to_raw_record(record), sort_keys=True) + "\n" for record in _recovering_records()),
        encoding="utf-8",
    )
    measured = mod.run_experiment(
        repo_root=tmp_path,
        corpus_paths=[Path("measured.jsonl")],
        min_candidates=4,
        min_examples=20,
        seed=123,
        n_boot=50,
    )
    assert measured["honest_verdict"] == mod.CLOSED_VERDICT
    assert measured["acceptance_gate"]["passed"] is True

    bad_record = _record(99)
    bad_candidates = tuple(
        mod.CandidateEvidence(
            answer=candidate.answer,
            correct=candidate.correct,
            text=candidate.text,
            confidence=candidate.confidence,
            energy=0.0 if not candidate.correct else 10.0,
            uncertainty=candidate.uncertainty,
            components=candidate.components,
        )
        for candidate in bad_record.candidates
    )
    low_auroc = tmp_path / "low_auroc.jsonl"
    low_auroc.write_text(
        json.dumps(_to_raw_record(mod.ProblemEvidence("bad", "synthetic", "g99", bad_candidates)))
        + "\n",
        encoding="utf-8",
    )
    blocked = mod.run_experiment(
        repo_root=tmp_path,
        corpus_paths=[Path("low_auroc.jsonl")],
        min_candidates=4,
        min_examples=1,
        seed=123,
        n_boot=10,
    )
    assert blocked["honest_verdict"] == mod.BLOCKED_VERDICT
    assert blocked["per_candidate_auroc"] < mod.MIN_REPRODUCED_AUROC
    assert "failed discrimination" in blocked["block_reason"]


def test_req_verify_3694_defensive_helpers(tmp_path: Path) -> None:
    """REQ-VERIFY-3694: helper edge cases do not fabricate evidence."""

    assert mod._accuracy([]) == 0.0
    assert mod._coerce_bool(True) is True
    assert mod._coerce_bool(0) is False
    assert mod._majority_answer(()) is None
    assert mod._normalise_high_good([]) == []
    assert mod._normalise_high_good([3.0, 3.0]) == [0.0, 0.0]
    assert mod._normalise_low_good([1.0, 3.0]) == [1.0, 0.0]
    assert mod._select_answer((), []) is None
    assert (
        mod._select_answer(
            [mod.CandidateEvidence(None, False, "", None, 0.0, 0.0, (0.0,))],
            [1.0],
        )
        is None
    )
    assert mod._coerce_float(None) is None
    assert mod._coerce_float(True) is None
    assert mod._coerce_float("not numeric") is None
    assert mod._coerce_bool("correct") is True
    assert mod._coerce_bool("incorrect") is False
    assert mod._coerce_bool("unknown", default=True) is True
    assert mod._sequence_signature(["a", None, "b"]) == ("a", None, "b")
    assert mod._candidate_confidence({"token_logprobs": [-1.0, None, "-3"]}) == pytest.approx(-2.0)
    assert mod._candidate_confidence({"token_logprobs": []}) is None
    assert mod._numeric_components({"b": 2, "a": 1}) == (1.0, 2.0)
    assert mod._numeric_components("not components") == ()
    assert mod._cached_energy({"energy_components": [1.0, 2.0]}) == (3.0, (1.0, 2.0))
    assert mod._cached_uncertainty({"uncertainty": 7.0}, [1.0, 2.0]) == pytest.approx(7.0)
    assert mod._cached_uncertainty({}, [1.0]) == pytest.approx(0.0)
    assert mod.normalise_cached_record("not a row", source_path="x", min_candidates=1) is None
    assert mod.normalise_cached_record({"samples": []}, source_path="x", min_candidates=1) is None
    assert (
        mod.normalise_cached_record(
            {
                "gold": "g",
                "samples": [
                    "not a sample",
                    {"answer": None, "verifier_energy": 1.0},
                ],
            },
            source_path="x",
            min_candidates=1,
        )
        is None
    )
    assert mod.load_cached_energy_records([Path("/definitely/missing")]) == []
    loader_corpus = tmp_path / "loader_edges.jsonl"
    loader_corpus.write_text(
        "\n{bad json}\n"
        + json.dumps(
            {
                "problem_id": "ok",
                "gold": "g",
                "samples": [
                    {"answer": "g", "verifier_energy": 0.0},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = mod.load_cached_energy_records([loader_corpus], min_candidates=1)
    assert [record.problem_id for record in loaded] == ["ok"]
    assert mod._component_normalized_scores(()) == []
    assert mod._confidence_fusion_scores(()) == []
    empty_record = mod.ProblemEvidence("empty", "synthetic", "g", ())
    assert mod._bootstrapped_answer(empty_record, seed=1, rounds=1) is None
    no_pick_record = mod.ProblemEvidence(
        "none",
        "synthetic",
        "g",
        (mod.CandidateEvidence(None, False, "", None, 0.0, 0.0, (0.0,)),),
    )
    assert mod._bootstrapped_answer(no_pick_record, seed=1, rounds=0) is None
    assert mod._path_label(Path("/outside/file.jsonl"), Path("/repo")) == "/outside/file.jsonl"

    bad_verdict = mod.build_blocked_artifact(
        corpus_paths=["x"],
        duration_s=0.1,
        block_reason="blocked",
    )
    bad_verdict["honest_verdict"] = "complete: invalid"
    with pytest.raises(AssertionError, match="unknown terminal verdict"):
        mod.validate_artifact(bad_verdict)

    bad_closed = mod.build_blocked_artifact(
        corpus_paths=["x"],
        duration_s=0.1,
        block_reason="blocked",
    )
    bad_closed["selection_gap_closed"] = True
    with pytest.raises(AssertionError, match="positive CI"):
        mod.validate_artifact(bad_closed)
