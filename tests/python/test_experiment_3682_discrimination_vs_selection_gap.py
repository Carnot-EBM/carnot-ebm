"""Tests for Exp 3682 discrimination-vs-selection gap diagnosis.

Spec: REQ-VERIFY-3682, SCENARIO-VERIFY-3682.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verify import discrimination_vs_selection_gap_3682 as mod


def _candidate(answer: str, gold: str, energy: float, confidence: float) -> mod.Candidate:
    return mod.Candidate(
        answer=answer,
        correct=answer == gold,
        text=f"candidate energy={energy}",
        confidence=confidence,
    )


def _record(
    problem_id: str,
    gold: str,
    rows: list[tuple[str, float, float]],
) -> mod.ProblemRecord:
    return mod.ProblemRecord(
        problem_id=problem_id,
        source_path="synthetic.jsonl",
        gold=gold,
        candidates=tuple(
            _candidate(answer, gold, energy, confidence) for answer, energy, confidence in rows
        ),
    )


def _energy_scorer(candidate: mod.Candidate) -> float:
    return float(candidate.text.rsplit("energy=", 1)[1])


def _fundamental_records(n: int = 6) -> list[mod.ProblemRecord]:
    records = []
    for idx in range(n):
        gold = f"g{idx}"
        records.append(
            _record(
                f"fundamental-{idx}",
                gold,
                [
                    ("majority", 10.0, 0.80),
                    ("majority", 10.5, 0.75),
                    ("decoy", 0.0, 0.90),
                    (gold, 1.0, 0.10),
                    (f"other-{idx}", 12.0, 0.05),
                ],
            )
        )
    return records


def _calibration_recovers_records(n: int = 8) -> list[mod.ProblemRecord]:
    records = []
    for idx in range(n):
        gold = f"g{idx}"
        records.append(
            _record(
                f"recovers-{idx}",
                gold,
                [
                    ("majority", 0.0, 0.90),
                    ("majority", 0.2, 0.85),
                    ("decoy", 0.1, 0.80),
                    (gold, 10.0, 0.10),
                ],
            )
        )
    return records


@pytest.mark.parametrize(
    (
        "blocked",
        "positive_control_valid",
        "selection_gap_closed",
        "expected_category",
        "expected_verdict",
    ),
    [
        pytest.param(
            False,
            True,
            True,
            "fix_recovers_selection_value",
            mod.CLOSED_VERDICT,
            id="fix_recovers_selection_value",
        ),
        pytest.param(
            False,
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
            "blocked",
            mod.BLOCKED_VERDICT,
            id="blocked",
        ),
    ],
)
def test_scenario_verify_3682_parametrizes_honest_outcomes(
    blocked: bool,
    positive_control_valid: bool,
    selection_gap_closed: bool,
    expected_category: str,
    expected_verdict: str,
) -> None:
    """SCENARIO-VERIFY-3682: honest outcomes are not one hard-coded success."""

    classification = mod.classify_outcome(
        blocked=blocked,
        positive_control_valid=positive_control_valid,
        selection_gap_closed=selection_gap_closed,
    )

    assert classification.category == expected_category
    assert classification.terminal_verdict == expected_verdict
    assert type(classification.selection_gap_closed) is bool


def test_req_verify_3682_diagnoses_decoupled_discrimination_without_gap_closure() -> None:
    """REQ-VERIFY-3682: AUROC can be useful while within-question selection still fails."""

    result = mod.evaluate_gap(
        _fundamental_records(),
        energy_scorer=_energy_scorer,
        seed=123,
        n_boot=200,
        train_fraction=0.5,
    )

    assert result.per_candidate_auroc == pytest.approx(0.75)
    assert result.within_question_rank_corr == pytest.approx(0.5)
    assert result.sc_selection_accuracy == pytest.approx(0.0)
    assert result.oracle_bestofn_accuracy == pytest.approx(1.0)
    assert result.selection_accuracy_per_question_normalized == pytest.approx(0.0)
    assert result.selection_accuracy_ranking_calibrated == pytest.approx(0.0)
    assert result.self_certainty_selection_accuracy == pytest.approx(0.0)
    assert result.positive_control_valid is True
    assert result.flip_count == 6
    assert result.selection_gap_closed is False
    assert result.best_fix_vs_sc_delta_ci["delta"] == pytest.approx(0.0)


def test_req_verify_3682_pairwise_calibration_can_recover_selection_value() -> None:
    """REQ-VERIFY-3682: ranking calibration is a real attempted fix on held-out rows."""

    result = mod.evaluate_gap(
        _calibration_recovers_records(),
        energy_scorer=_energy_scorer,
        seed=456,
        n_boot=200,
        train_fraction=0.5,
    )

    assert result.sc_selection_accuracy == pytest.approx(0.0)
    assert result.oracle_bestofn_accuracy == pytest.approx(1.0)
    assert result.ranking_calibration["orientation"] == "higher_energy_better"
    assert result.ranking_calibration["heldout_n"] == 4
    assert result.selection_accuracy_ranking_calibrated == pytest.approx(1.0)
    assert result.best_fix_vs_sc_delta_ci["method"] == "ranking_calibrated"
    assert result.best_fix_vs_sc_delta_ci["ci95"][0] > 0.0
    assert result.positive_control_valid is True
    assert result.selection_gap_closed is True


def test_req_verify_3682_artifacts_validate_required_fields_and_bare_bool() -> None:
    """REQ-VERIFY-3682: terminal artifacts preserve the required schema."""

    result = mod.evaluate_gap(
        _calibration_recovers_records(),
        energy_scorer=_energy_scorer,
        seed=789,
        n_boot=100,
        train_fraction=0.5,
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
    assert artifact["acceptance_gate"]["positive_control_valid"] is True

    blocked = mod.build_blocked_artifact(corpus_paths=["missing.jsonl"], duration_s=0.1)
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
    with pytest.raises(AssertionError, match="bare bool"):
        mod.validate_artifact(bad_bool)

    bad_positive_control_bool = dict(blocked)
    bad_positive_control_bool["positive_control_valid"] = "false"
    with pytest.raises(AssertionError, match="positive_control_valid"):
        mod.validate_artifact(bad_positive_control_bool)

    bad_verdict = dict(blocked)
    bad_verdict["honest_verdict"] = "complete: invalid"
    with pytest.raises(AssertionError, match="unknown terminal verdict"):
        mod.validate_artifact(bad_verdict)

    bad_gate = dict(blocked)
    bad_gate["acceptance_gate"] = {"required_fields_present": False}
    with pytest.raises(AssertionError, match="acceptance gate"):
        mod.validate_artifact(bad_gate)


def test_req_verify_3682_run_experiment_writes_artifact_and_blocks(tmp_path: Path) -> None:
    """REQ-VERIFY-3682: run_experiment writes measured and blocked terminal artifacts."""

    corpus = tmp_path / "corpus.jsonl"
    rows = []
    for record in _calibration_recovers_records():
        rows.append(
            {
                "problem_id": record.problem_id,
                "gold": record.gold,
                "samples": [
                    {
                        "answer": candidate.answer,
                        "correct": candidate.correct,
                        "text": candidate.text,
                        "mean_token_logprob": candidate.confidence,
                    }
                    for candidate in record.candidates
                ],
            }
        )
    corpus.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"

    artifact = mod.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        corpus_paths=[Path("corpus.jsonl")],
        min_candidates=4,
        min_examples=4,
        max_sc_accuracy=0.25,
        max_majority_supports=(0.50,),
        energy_scorer=_energy_scorer,
        seed=321,
        n_boot=100,
    )

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == mod.CLOSED_VERDICT

    blocked = mod.run_experiment(
        repo_root=tmp_path,
        corpus_paths=[Path("missing.jsonl")],
        energy_scorer=_energy_scorer,
    )
    assert blocked["honest_verdict"] == mod.BLOCKED_VERDICT


def test_req_verify_3682_edge_metrics_are_defensive() -> None:
    """REQ-VERIFY-3682: helpers handle degenerate rows without fabricating evidence."""

    assert mod.tie_aware_auroc([], []) is None
    assert mod.tie_aware_auroc([1, 1], [0.2, 0.3]) is None
    assert mod.tie_aware_auroc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert mod.mean_within_question_rank_corr([], []) == {
        "mean_tau": None,
        "weighted_tau": None,
        "n_questions": 0,
        "n_comparable_pairs": 0,
    }
    assert mod.normalized_energy_scores([], method="zscore") == []
    assert mod.normalized_energy_scores([2.0, 2.0], method="zscore") == [0.0, 0.0]
    assert mod.normalized_energy_scores([2.0, 2.0], method="minmax") == [0.0, 0.0]
    assert mod.normalized_energy_scores([2.0, 4.0], method="minmax") == [1.0, 0.0]
    with pytest.raises(ValueError, match="unknown normalization"):
        mod.normalized_energy_scores([1.0], method="bad")

    assert mod._minmax_high_good([]) == []
    assert mod._minmax_high_good([1.0, 1.0]) == [0.0, 0.0]
    assert mod._select_by_scores([mod.Candidate(None, False, "", None)], [1.0]) is None
    assert mod._split_indices(1, seed=1, train_fraction=0.5) == ([0], [0])
    assert mod._path_label(Path("/outside/file.jsonl"), Path("/repo")) == "/outside/file.jsonl"

    tied = mod._fit_pairwise_orientation(
        [
            mod.ProblemRecord(
                problem_id="tie",
                source_path="synthetic",
                gold="g",
                candidates=(
                    mod.Candidate("g", True, "", None),
                    mod.Candidate("x", False, "", None),
                ),
            )
        ],
        [[1.0, 1.0]],
        [0],
    )
    assert tied["pairwise_train_ties"] == 1
