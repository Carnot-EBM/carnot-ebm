"""CPU-only tests for the JevBench decision readout evaluator.

Spec: REQ-AUTO-028, SCENARIO-AUTO-028-A, SCENARIO-AUTO-028-B.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.jevbench_readout_eval import (
    _uniform_fraction,
    readout_positive_control,
    DecisionTask,
    Option,
    brier_score,
    cross_validated_temperature_scale,
    jevbench_calibration_score,
    kfold_indices,
    load_public_tasks,
    main,
    run_evaluation,
    run_option_order_probe,
    top_label_ece,
    total_variation_distance,
)


def _row(task_id: str = "synthetic-1") -> dict[str, object]:
    return {
        "id": task_id,
        "family": "synthetic",
        "state": "A synthetic state.",
        "question": {
            "type": "choice",
            "instructions": "Choose one option.",
            "criteria": {"a": "Option A", "b": "Option B"},
        },
        "labels": ["a", "b"],
        "expected": "a",
        "split": "public",
        "group": None,
        "provenance": {"license": "synthetic"},
    }


def _write_fixture(tasks_dir: Path, rows: list[dict[str, object]]) -> Path:
    tasks_dir.mkdir(parents=True)
    path = tasks_dir / "easy.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _task() -> DecisionTask:
    return DecisionTask(
        task_id="synthetic-1",
        source_split="easy",
        family="synthetic",
        state="A synthetic state.",
        question="Choose one option.",
        options=(Option("a", "Option A"), Option("b", "Option B")),
        expected="a",
        gold_probs=None,
    )


def test_top_label_ece_matches_hand_computation() -> None:
    probabilities = [{"a": 0.8, "b": 0.2}, {"a": 0.6, "b": 0.4}]
    expected = ["a", "b"]

    assert top_label_ece(probabilities, expected) == pytest.approx(0.4)


def test_total_variation_fidelity_matches_hand_computation() -> None:
    predicted = {"a": 0.7, "b": 0.3}
    gold = {"a": 0.4, "b": 0.6}

    distance = total_variation_distance(predicted, gold, ("a", "b"))

    assert distance == pytest.approx(0.3)
    assert 1.0 - distance == pytest.approx(0.7)


def test_brier_score_matches_multiclass_sum() -> None:
    score = brier_score({"a": 0.7, "b": 0.3}, "a", ("a", "b"))

    assert score == pytest.approx(0.18)


def test_jevbench_calibration_score_matches_formula() -> None:
    score = jevbench_calibration_score(ece=0.1, mean_tvd=0.2)

    assert score == pytest.approx(80.0)


def test_temperature_scaling_improves_overconfident_ece() -> None:
    probabilities = [{"a": 0.99, "b": 0.01} for _ in range(10)]
    expected = ["a"] * 6 + ["b"] * 4
    labels = [("a", "b") for _ in probabilities]

    calibrated, records = cross_validated_temperature_scale(
        probabilities,
        expected,
        labels,
        folds=5,
        seed=19,
    )

    assert top_label_ece(calibrated, expected) < top_label_ece(probabilities, expected)
    assert all(record.temperature > 1.0 for record in records)


def test_temperature_scaling_folds_are_disjoint() -> None:
    folds = kfold_indices(n_rows=11, folds=5, seed=7)

    assert set().union(*(set(test) for _, test in folds)) == set(range(11))
    assert all(set(train).isdisjoint(test) for train, test in folds)
    assert all(len(test) > 0 for _, test in folds)


class _OrderSensitiveScorer:
    def score(
        self,
        state: object,
        question: str,
        options: tuple[Option, ...],
    ) -> dict[str, float]:
        del state, question
        return {
            option.option_id: 0.9 if index == 0 else 0.1 for index, option in enumerate(options)
        }


class _OrderInvariantScorer:
    def score(
        self,
        state: object,
        question: str,
        options: tuple[Option, ...],
    ) -> dict[str, float]:
        del state, question
        return {option.option_id: 0.8 if option.option_id == "a" else 0.2 for option in options}


def test_option_order_probe_detects_order_sensitivity() -> None:
    result = run_option_order_probe((_task(),), _OrderSensitiveScorer(), progress_every=0)

    assert result.accuracy_original == pytest.approx(1.0)
    assert result.accuracy_reversed == pytest.approx(0.0)
    assert result.mean_absolute_probability_shift == pytest.approx(0.8)


def test_option_order_probe_reports_zero_for_invariant_scorer() -> None:
    result = run_option_order_probe((_task(),), _OrderInvariantScorer(), progress_every=0)

    assert result.accuracy_original == pytest.approx(1.0)
    assert result.accuracy_reversed == pytest.approx(1.0)
    assert result.mean_absolute_probability_shift == pytest.approx(0.0)


def test_loader_reports_a_malformed_row(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    path = _write_fixture(tasks_dir, [_row()])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not-json}\n")

    report = load_public_tasks(tasks_dir, ("easy",))

    assert len(report.tasks) == 1
    assert len(report.errors) == 1
    assert report.errors[0].line_number == 2
    assert "JSON" in report.errors[0].message


def test_stub_artifact_has_required_fields_and_is_not_live(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    _write_fixture(tasks_dir, [_row()])

    artifact = run_evaluation(
        tasks_dir=tasks_dir,
        split="easy",
        limit=None,
        dry_run=True,
        model_gguf=None,
        seed=11,
        folds=2,
        bootstrap_resamples=20,
    )

    required = {
        "inference_substrate",
        "model_specs",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
        "n_per_split",
        "verifier_is_oracle",
        "honest_verdict",
        "preconditions_checked",
        "methodology_note",
        "source",
        "metrics",
        "option_order_probe",
    }
    assert required <= artifact.keys()
    assert artifact["inference_substrate"] == "stub_deterministic_no_live_inference"
    assert artifact["inference_substrate"] != "live_llm_inference"
    assert str(artifact["honest_verdict"]).startswith("complete_")
    assert artifact["verifier_is_oracle"] is False


def test_cli_without_output_writes_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tasks_dir = tmp_path / "tasks"
    _write_fixture(tasks_dir, [_row()])
    monkeypatch.chdir(tmp_path)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}

    exit_code = main(
        [
            "--tasks-dir",
            str(tasks_dir),
            "--split",
            "easy",
            "--dry-run",
            "--seed",
            "3",
            "--folds",
            "2",
        ]
    )
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}

    assert exit_code == 0
    assert after == before
    assert not (tmp_path / "results").exists()


def test_positive_control_rejects_a_uniform_readout() -> None:
    """A readout that cannot tell options apart must be refused, not benchmarked."""

    class Uniform:
        def score(self, state, question, options):  # noqa: ANN001, ANN201
            return {option.option_id: 1.0 / len(options) for option in options}

    with pytest.raises(RuntimeError, match="positive control"):
        readout_positive_control(Uniform())


def test_positive_control_accepts_a_readout_that_knows_the_answers() -> None:
    class Knows:
        answers = {"apple", "four", "paris", "water"}

        def score(self, state, question, options):  # noqa: ANN001, ANN201
            return {
                option.option_id: 0.9 if option.option_id in self.answers else 0.1
                for option in options
            }

    result = readout_positive_control(Knows())
    assert result["correct"] == result["cases"] == 8


def test_uniform_fraction_counts_uniform_distributions() -> None:
    assert _uniform_fraction([{"a": 0.5, "b": 0.5}, {"a": 0.9, "b": 0.1}]) == 0.5
    assert _uniform_fraction([]) == 0.0
