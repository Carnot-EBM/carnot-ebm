"""Tests for Exp 3727 matched-compute FLOP accounting harness.

Spec refs: REQ-AR-053, SCENARIO-AR-053-01, SCENARIO-AR-053-02,
SCENARIO-AR-053-03.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.phase3 import matched_compute_eval_harness as h


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "autoresearch" / "spec.md"


def _heldout_examples() -> list[h.ReasoningExample]:
    return [
        h.ReasoningExample("p1", "question 1", "A", prompt_tokens=2, generated_tokens=3),
        h.ReasoningExample("p2", "question 2", "B", prompt_tokens=2, generated_tokens=3),
        h.ReasoningExample("p3", "question 3", "C", prompt_tokens=2, generated_tokens=3),
        h.ReasoningExample("p4", "question 4", "D", prompt_tokens=2, generated_tokens=3),
    ]


def test_req_ar_053_spec_anchor_exists() -> None:
    """REQ-AR-053: the matched-compute harness is OpenSpec anchored."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-AR-053" in spec
    assert "SCENARIO-AR-053-01" in spec
    assert "SCENARIO-AR-053-02" in spec
    assert "SCENARIO-AR-053-03" in spec
    assert "parameter_count * sequence_tokens * forward_passes" in spec


def test_scenario_ar_053_01_flop_accounting_hand_computed_symmetry() -> None:
    """SCENARIO-AR-053-01: EBT and AR totals match a hand-computed toy case."""
    # 10 parameters * (2 prompt + 3 generated tokens) * (1 initial + 4 descent passes)
    ebt_flops = h.ebt_generation_flops(
        parameter_count=10,
        prompt_tokens=2,
        generated_tokens=3,
        energy_descent_steps=4,
    )
    # 10 parameters * 5 sequence tokens * 5 independent AR samples.
    ar_flops = h.ar_generation_flops(
        parameter_count=10,
        prompt_tokens=2,
        generated_tokens=3,
        best_of_m=5,
    )

    assert ebt_flops == 250
    assert ar_flops == 250
    assert h.sequence_forward_flops(
        parameter_count=10,
        sequence_tokens=5,
        forward_passes=5,
    ) == 250
    assert "parameter_count * sequence_tokens * forward_passes" in h.FLOP_MODEL_DESCRIPTION

    with pytest.raises(ValueError, match="parameter_count"):
        h.sequence_forward_flops(parameter_count=0, sequence_tokens=5, forward_passes=1)
    with pytest.raises(ValueError, match="energy_descent_steps"):
        h.ebt_generation_flops(
            parameter_count=10,
            prompt_tokens=2,
            generated_tokens=3,
            energy_descent_steps=-1,
        )


def test_scenario_ar_053_02_budget_matcher_equalizes_within_tolerance() -> None:
    """SCENARIO-AR-053-02: the matcher selects AR M that matches EBT FLOPs."""
    examples = _heldout_examples()
    target = h.total_ebt_flops(
        examples,
        parameter_count=100,
        energy_descent_steps=4,
    )
    ar_single = h.total_ar_flops(examples, parameter_count=100, best_of_m=1)

    match = h.match_ar_best_of_m(
        target_total_flops=target,
        ar_single_sample_total_flops=ar_single,
        tolerance=0.001,
    )

    assert target == 10_000
    assert ar_single == 2_000
    assert match.ar_best_of_m == 5
    assert match.ar_total_flops == target
    assert match.relative_error == 0.0
    assert match.within_tolerance is True

    with pytest.raises(ValueError, match="tolerance"):
        h.match_ar_best_of_m(
            target_total_flops=target,
            ar_single_sample_total_flops=ar_single,
            tolerance=-0.1,
        )


def test_scenario_ar_053_03_synthetic_driver_returns_expected_verdict() -> None:
    """SCENARIO-AR-053-03: known synthetic labels produce the expected verdict."""
    examples = _heldout_examples()

    def ebt_predict(example: h.ReasoningExample) -> h.Prediction:
        answers = {"p1": "A", "p2": "B", "p3": "C", "p4": "wrong"}
        return h.Prediction(answers[example.example_id], score=0.0)

    def ar_sample(example: h.ReasoningExample, best_of_m: int, seed: int) -> list[h.Prediction]:
        assert best_of_m == 5
        assert seed >= h.RANDOM_SEED
        selected_answers = {"p1": "A", "p2": "wrong", "p3": "wrong", "p4": "D"}
        return [
            h.Prediction(selected_answers[example.example_id], score=1.0 - index * 0.01)
            for index in range(best_of_m)
        ]

    result = h.compare_matched_compute(
        examples,
        ebt_predictor=ebt_predict,
        ar_sampler=ar_sample,
        ebt_parameter_count=100,
        ar_parameter_count=100,
        energy_descent_steps=4,
        tolerance=0.001,
        random_seed=h.RANDOM_SEED,
    )

    assert result.budget_match.ar_best_of_m == 5
    assert result.budget_match.within_tolerance is True
    assert result.ebt_correct == 3
    assert result.ar_correct == 2
    assert result.ebt_accuracy == 0.75
    assert result.ar_accuracy == 0.5
    assert result.verdict == "ebt_higher_at_equal_flops"
    assert result.rows[0]["ar_answer"] == "A"

    def short_ar_sample(
        example: h.ReasoningExample,
        best_of_m: int,
        seed: int,
    ) -> list[h.Prediction]:
        return [h.Prediction("A", score=0.0)]

    with pytest.raises(ValueError, match="exactly best_of_m"):
        h.compare_matched_compute(
            examples,
            ebt_predictor=ebt_predict,
            ar_sampler=short_ar_sample,
            ebt_parameter_count=100,
            ar_parameter_count=100,
            energy_descent_steps=4,
            tolerance=0.001,
            random_seed=h.RANDOM_SEED,
        )


def test_req_ar_053_artifact_fields_and_run_experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-AR-053: the runner writes required bare-value artifact fields."""
    class Completed:
        returncode = 0
        stdout = "...... [100%]\n6 passed in 0.65s\n"
        stderr = ""

    def fake_run(command: list[str], check: bool, capture_output: bool, text: bool) -> Completed:
        assert command[0]
        assert check is False
        assert capture_output is True
        assert text is True
        return Completed()

    monkeypatch.setattr(h.subprocess, "run", fake_run)
    summary = h.run_unit_tests()
    assert summary.count_string() == "6_of_6_pass"

    output_path = tmp_path / "results" / "experiment_3727_matched_compute_eval_harness.json"
    artifact = h.run_experiment(result_path=output_path)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["honest_verdict"] == (
        "complete: matched_compute_eval_harness_built_flop_accounting_"
        "documented_unit_tests_6_of_6_pass"
    )
    assert artifact["inference_substrate"] == h.INFERENCE_SUBSTRATE
    assert artifact["flop_model_description"] == h.FLOP_MODEL_DESCRIPTION
    assert artifact["unit_tests_added"] == "tests/python/test_matched_compute_eval_harness.py"
    assert artifact["unit_tests_passed"] == "6_of_6_pass"
    assert artifact["budget_matcher_tolerance"] == h.DEFAULT_TOLERANCE
    assert artifact["random_seed"] == h.RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["duration_s"] > 0.0
    assert set(h.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert h.validate_artifact(artifact) == []

    bad = dict(artifact, unit_tests_passed="4_of_5_pass")
    assert "unit_tests_passed must match the terminal verdict count" in h.validate_artifact(bad)

    class Unparseable:
        returncode = 0
        stdout = "pytest completed but summary is absent"
        stderr = ""

    monkeypatch.setattr(h.subprocess, "run", lambda *args, **kwargs: Unparseable())
    with pytest.raises(RuntimeError, match="could not parse"):
        h.run_unit_tests()

    monkeypatch.setattr(h.subprocess, "run", fake_run)
    monkeypatch.setattr(h, "validate_artifact", lambda artifact: ["forced validation error"])
    with pytest.raises(ValueError, match="forced validation error"):
        h.run_experiment(result_path=tmp_path / "bad.json")
