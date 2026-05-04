"""Tests for Exp 1278 pure-data EST gaming-defense measurement.

Spec: REQ-VERIFY-1278, SCENARIO-VERIFY-1278
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import gaming_verifiers_defense_est_final as exp


def _fover_pairs(n_pairs: int) -> list[dict]:
    pairs: list[dict] = []
    for index in range(n_pairs):
        is_correct = index % 4 == 0
        pairs.append(
            {
                "question_index": index,
                "question": f"What is {index}+1?",
                "response": (
                    f"Step 1: add one to {index}. "
                    f"The answer is {index + 1}. Therefore the total is {index + 1}."
                ),
                "is_correct": is_correct,
            }
        )
    return pairs


def _score_payload(n_pairs: int) -> dict:
    labels = [index % 4 == 0 for index in range(n_pairs)]
    return {
        "experiment": "1252_q11_tss_instrumentation",
        "corpus": "results/fover_corpus_v5.json",
        "n_samples": n_pairs,
        "acceptance_threshold": 0.5,
        "verifier_names": ["A", "B", "C", "D", "E"],
        "per_verifier_energies": {
            "A": [0.1 if label else 0.8 for label in labels],
            "B": [0.1 if label else 0.2 for label in labels],
            "C": [0.1 for _ in labels],
            "D": [0.1 for _ in labels],
            "E": [0.1 for _ in labels],
        },
    }


def _write_json(path: Path, payload: dict | list[dict]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_select_scored_examples_requires_req1278_labels_and_scores() -> None:
    """REQ-VERIFY-1278-2: selected examples have labels and aligned scores."""

    pairs = _fover_pairs(36)
    pairs[3].pop("is_correct")

    examples = exp.select_scored_examples(
        pairs,
        _score_payload(36),
        min_examples=30,
        max_examples=35,
    )

    assert len(examples) == 35
    assert all(example.index != 3 for example in examples)
    assert all(isinstance(example.is_correct, bool) for example in examples)
    assert all(
        set(example.per_verifier_scores) == {"A", "B", "C", "D", "E"} for example in examples
    )
    assert examples[0].question == "What is 0+1?"
    assert examples[1].base_k5_blocked is True


def test_select_scored_examples_rejects_insufficient_req1278_data() -> None:
    """REQ-VERIFY-1278-2: fewer than 30 scored labeled rows is insufficient."""

    with pytest.raises(ValueError, match="need at least 30"):
        exp.select_scored_examples(
            {"pairs": _fover_pairs(10)},
            _score_payload(10),
            min_examples=30,
        )


def test_perturbations_and_est_measurement_are_deterministic_for_req1278() -> None:
    """REQ-VERIFY-1278-3/4: EST uses deterministic preserving/changing strings."""

    example = exp.ScoredFoVerExample(
        index=0,
        question="What is 2+2?",
        response="Step 1: add the numbers. The answer is 4. Therefore the total is 4.",
        is_correct=True,
        per_verifier_scores={"A": 0.1, "B": 0.2, "C": 0.1, "D": 0.2, "E": 0.1},
    )

    preserving = exp.meaning_preserving_perturbations(example.response)
    changing = exp.meaning_changing_perturbations(example.response)
    measurement = exp.measure_example(example, stability_threshold=0.1, sensitivity_threshold=0.1)

    assert [item.tag for item in preserving] == [
        "synonym_replacement",
        "formatting_preserving_wording",
        "whitespace_variation",
    ]
    assert [item.tag for item in changing] == ["negation", "numeric_mutation", "step_removal"]
    assert all(item.text != example.response for item in preserving + changing)
    assert any("not" in item.text.lower() for item in changing)
    assert any("3+2" in item.text or "5" in item.text for item in changing)
    assert measurement["base_score"] == pytest.approx(0.14)
    assert measurement["max_changing_delta"] > measurement["max_preserving_delta"]
    assert measurement["preserving_unstable"] is False
    assert measurement["changing_sensitive"] is True


def test_perturbation_helper_edges_keep_req1278_deterministic() -> None:
    """REQ-VERIFY-1278-3: fallback string transforms stay deterministic."""

    preserving = exp.meaning_preserving_perturbations("unchanged phrasing")
    changing = exp.meaning_changing_perturbations("plain text")
    empty_example = exp.ScoredFoVerExample(
        index=1,
        question="",
        response="",
        is_correct=False,
        per_verifier_scores={"A": 0.0},
    )
    empty_score = exp.score_perturbation(
        empty_example,
        exp.Perturbation("meaning_preserving", "empty", ""),
    )

    assert preserving[0].text == "unchanged phrasing Same meaning."
    assert changing[1].text == "plain text Final answer: 1."
    assert changing[2].text == "plain"
    assert empty_score == 0.0


def test_build_est_artifact_computes_required_req1278_fields() -> None:
    """REQ-VERIFY-1278-4/5: EST proxies and vulnerability fields are computed."""

    examples = exp.select_scored_examples(
        {"pairs": _fover_pairs(32)},
        _score_payload(32),
        min_examples=30,
        max_examples=32,
    )

    artifact = exp.build_est_artifact(
        examples,
        exp1256_payload={"max_pairwise_r_k5": 0.46, "k_eff": 1.76},
        source_artifacts={"fover": "tmp/fover.json", "score_surface": "tmp/q11.json"},
        run_date="20260504",
    )

    assert artifact["experiment"] == "1278_gaming_verifiers_defense_est_final"
    assert artifact["schema"] == "gaming_verifiers_defense_est_v1"
    assert artifact["run_date"] == "20260504"
    assert artifact["status"] == "complete"
    assert artifact["n_selected_examples"] == 32
    assert artifact["label_counts"] == {"correct": 8, "incorrect": 24}
    assert 0.0 <= artifact["est_precision_proxy"] <= 1.0
    assert 0.0 <= artifact["est_recall_proxy"] <= 1.0
    assert 0.0 <= artifact["gaming_vulnerability_score"] <= 1.0
    assert artifact["k5_blocks_surface_gaming"] is True
    assert artifact["gaming_defense_measured"] is True
    assert artifact["honest_verdict"].startswith("est_vulnerability_")
    assert len(artifact["per_example_measurements"]) == 32


def test_run_experiment_writes_in_progress_then_complete_scenario1278(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-1278: runner writes the required final artifact."""

    fover_path = tmp_path / "fover_corpus_v5.json"
    score_path = tmp_path / "q11_tss_diagnostic_report.json"
    exp1256_path = tmp_path / "experiment_1256_verifier_orthogonality_audit_v3.json"
    exp1263_path = tmp_path / "experiment_1263_gaming_verifiers_defense_v4.json"
    output_path = tmp_path / "experiment_1278_gaming_verifiers_defense_est_final.json"
    _write_json(fover_path, {"pairs": _fover_pairs(34)})
    _write_json(score_path, _score_payload(34))
    _write_json(exp1256_path, {"max_pairwise_r_k5": 0.46, "k_eff": 1.76})
    _write_json(exp1263_path, {"status": "in_progress", "honest_verdict": "in_progress"})

    statuses: list[str] = []
    original = exp.write_in_progress_artifact

    def tracking_write(path: Path | str, *, run_date: str = exp.RUN_DATE) -> dict:
        artifact = original(path, run_date=run_date)
        statuses.append(json.loads(Path(path).read_text(encoding="utf-8"))["status"])
        return artifact

    monkeypatch.setattr(exp, "write_in_progress_artifact", tracking_write)

    artifact = exp.run_experiment(
        fover_path=fover_path,
        score_path=score_path,
        exp1256_path=exp1256_path,
        exp1263_path=exp1263_path,
        output_path=output_path,
        run_date="20260504",
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert statuses == ["in_progress"]
    assert artifact == persisted
    assert persisted["status"] == "complete"
    assert persisted["source_artifacts"]["exp1263"]["status"] == "in_progress"
    assert persisted["gaming_defense_measured"] is True
