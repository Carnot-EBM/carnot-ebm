"""Tests for Exp 1172 NRGPT per-token energy inference.

Spec refs: REQ-KONA-014, SCENARIO-KONA-014.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.phase3 import nrgpt_energy as nrgpt  # noqa: E402
from scripts import experiment_1172_nrgpt_per_token_energy_inference as exp1172  # noqa: E402


def test_energy_per_token_sums_to_batch_energy() -> None:
    """REQ-KONA-014: per-token readout returns one energy and batch sums them."""
    model = nrgpt.NRGPTEnergyModel(d_hidden=8, seed=12)
    tokens = nrgpt.tokenize_response("Compute 2 + 2. The answer is 4.")

    per_token = model.energy_per_token(tokens)
    batch = model.batch_energy("Compute 2 + 2. The answer is 4.")

    assert len(per_token) == len(tokens)
    assert per_token == pytest.approx([float(value) for value in per_token])
    assert batch == pytest.approx(sum(per_token))
    assert model.energy_per_token([]) == []


def test_per_token_energy_spikes_at_arithmetic_error_token() -> None:
    """REQ-KONA-014: incorrect arithmetic has its largest energy at the error token."""
    model = nrgpt.NRGPTEnergyModel(d_hidden=8, seed=3)
    wrong_tokens = nrgpt.tokenize_response("Compute 2 + 2. The answer is 5.")
    correct_tokens = nrgpt.tokenize_response("Compute 2 + 2. The answer is 4.")

    wrong_energies = model.energy_per_token(wrong_tokens)
    correct_energies = model.energy_per_token(correct_tokens)
    error_idx = nrgpt.locate_arithmetic_error_token(wrong_tokens)

    assert error_idx is not None
    assert wrong_tokens[error_idx] == "5"
    assert abs(int(np.argmax(wrong_energies)) - error_idx) <= 2
    assert max(wrong_energies) > max(correct_energies) + 0.5
    assert max(correct_energies) - min(correct_energies) <= 0.1


def test_per_token_evaluation_reports_auc_and_localization() -> None:
    """SCENARIO-KONA-014: token AUROC and spike localization are measured."""
    rows = [
        {"step_text": "Compute 2 + 2. The answer is 4.", "label": "correct"},
        {"step_text": "Compute 3 + 4. The answer is 7.", "label": "correct"},
        {"step_text": "Compute 2 + 2. The answer is 5.", "label": "incorrect"},
        {"step_text": "Compute 3 + 4. The answer is 9.", "label": "incorrect"},
    ]

    result = nrgpt.evaluate_per_token_fover(rows, model=nrgpt.NRGPTEnergyModel(d_hidden=8))

    assert result.per_token_auroc >= 0.99
    assert result.energy_spike_localization_rate == pytest.approx(1.0)
    assert result.n_error_tokens == 2
    assert result.n_token_scores > result.n_error_tokens


def test_exp1172_artifact_and_script_main_write_required_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-KONA-014: Exp 1172 writes honest per-token comparison fields."""
    corpus = tmp_path / "fover_corpus.jsonl"
    baseline = tmp_path / "experiment_1163.json"
    deliverable = tmp_path / "experiment_1172.json"
    corpus.write_text(
        "\n".join(json.dumps(row) for row in nrgpt.build_synthetic_fover_rows(40, seed=5)) + "\n",
        encoding="utf-8",
    )
    baseline.write_text(json.dumps({"baseline_auroc": 0.75}), encoding="utf-8")

    artifact = nrgpt.run_per_token_experiment(
        dataset_path=tmp_path / "missing.jsonl",
        corpus_path=corpus,
        baseline_artifact_path=baseline,
        deliverable_path=deliverable,
        n_train=20,
        n_eval=10,
    )
    payload = json.loads(deliverable.read_text(encoding="utf-8"))

    assert payload == artifact
    assert nrgpt.PER_TOKEN_REQUIRED_ARTIFACT_FIELDS <= set(payload)
    assert payload["batch_auroc_baseline"] == 0.75
    assert payload["per_token_above_batch"] is True
    assert payload["nrgpt_per_token_energy_above_batch"] is True
    assert payload["honest_verdict"] == "per_token_improves_auroc"

    assert nrgpt.classify_per_token_honest_verdict(0.75, 0.75) == "per_token_tied_with_batch"
    assert nrgpt.classify_per_token_honest_verdict(0.74, 0.75) == "per_token_worse_than_batch"

    monkeypatch.setattr(exp1172, "DATASET_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(exp1172, "CORPUS_PATH", corpus)
    monkeypatch.setattr(exp1172, "BASELINE_ARTIFACT", baseline)
    monkeypatch.setattr(exp1172, "DELIVERABLE", tmp_path / "script_result.json")
    monkeypatch.setattr(exp1172, "N_TRAIN", 20)
    monkeypatch.setattr(exp1172, "N_EVAL", 10)

    assert exp1172.main() == 0
    script_payload = json.loads((tmp_path / "script_result.json").read_text(encoding="utf-8"))
    assert nrgpt.PER_TOKEN_REQUIRED_ARTIFACT_FIELDS <= set(script_payload)


def test_per_token_parser_and_artifact_defensive_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-KONA-014: parser edge cases and artifact validation stay explicit."""
    with pytest.raises(ValueError, match="d_hidden"):
        nrgpt.NRGPTEnergyModel(d_hidden=0)

    assert nrgpt.locate_arithmetic_error_token(nrgpt.tokenize_response("2 + 2 = 5")) == 4
    assert nrgpt.locate_arithmetic_error_token(nrgpt.tokenize_response("2 = 3")) == 2
    assert nrgpt.locate_arithmetic_error_token(nrgpt.tokenize_response("2 times 3 = 7")) == 4
    assert nrgpt.locate_arithmetic_error_token(nrgpt.tokenize_response("8 / 2 = 5")) == 4
    assert (
        nrgpt.locate_arithmetic_error_token(nrgpt.tokenize_response("foo 2 + 2 = $ 5 apples")) == 6
    )
    assert nrgpt.locate_arithmetic_error_token(nrgpt.tokenize_response("foo = 2")) is None
    assert (
        nrgpt.locate_arithmetic_error_token(nrgpt.tokenize_response("No arithmetic here")) is None
    )
    assert (
        nrgpt.locate_arithmetic_error_token(nrgpt.tokenize_response("The answer is five")) is None
    )
    assert nrgpt.locate_arithmetic_error_token(nrgpt.tokenize_response("2 + answer is 4")) is None

    result = nrgpt.evaluate_per_token_fover(
        [
            {"step_text": "", "label": "correct"},
            {"step_text": "No arithmetic here", "label": "incorrect"},
        ]
    )
    assert result.n_eval_rows == 1
    assert result.n_error_tokens == 0
    assert result.energy_spike_localization_rate == 0.0

    assert nrgpt.load_batch_auroc_baseline(tmp_path / "missing.json") == 0.5
    assert nrgpt._expression_value([]) is None
    assert nrgpt._expression_value([(0, "bad")]) is None
    assert nrgpt._first_number_position([(0, "+")]) is None
    assert nrgpt._first_number_after(["answer", "soon"], 0) is None
    assert nrgpt._trim_expression_pairs([(0, "*"), (1, "2")]) == [(1, "2")]
    assert nrgpt._trim_expression_pairs([(0, "+")]) == []
    assert nrgpt._find_first_arithmetic_expression(["2", "$", "+", "2"], stop_idx=4) == 4.0
    assert nrgpt._find_first_arithmetic_expression(["foo", "2", "bar"], stop_idx=3) is None
    assert nrgpt._safe_eval_arithmetic("1/0") is None
    assert nrgpt._safe_eval_arithmetic("-3+5") == pytest.approx(2.0)
    assert nrgpt._safe_eval_arithmetic("5-3") == pytest.approx(2.0)
    assert nrgpt._safe_eval_arithmetic("2*3") == pytest.approx(6.0)
    assert nrgpt._safe_eval_arithmetic("8/2") == pytest.approx(4.0)
    assert nrgpt._safe_eval_arithmetic("abs(1)") is None

    corpus = tmp_path / "fover_corpus.jsonl"
    baseline = tmp_path / "experiment_1163.json"
    corpus.write_text(
        "\n".join(json.dumps(row) for row in nrgpt.build_synthetic_fover_rows(20, seed=9)) + "\n",
        encoding="utf-8",
    )
    baseline.write_text(json.dumps({"baseline_auroc": 0.75}), encoding="utf-8")

    monkeypatch.setattr(nrgpt, "PER_TOKEN_REQUIRED_ARTIFACT_FIELDS", {"missing_field"})
    with pytest.raises(AssertionError, match="missing Exp 1172 artifact fields"):
        nrgpt.run_per_token_experiment(
            dataset_path=tmp_path / "missing.jsonl",
            corpus_path=corpus,
            baseline_artifact_path=baseline,
            deliverable_path=tmp_path / "bad_missing.json",
            n_train=10,
            n_eval=6,
        )

    monkeypatch.setattr(
        nrgpt,
        "PER_TOKEN_REQUIRED_ARTIFACT_FIELDS",
        {
            "per_token_auroc",
            "batch_auroc_baseline",
            "per_token_above_batch",
            "nrgpt_per_token_energy_above_batch",
            "energy_spike_localization_rate",
            "honest_verdict",
        },
    )
    monkeypatch.setattr(nrgpt, "PER_TOKEN_HONEST_VERDICTS", {"not_the_verdict"})
    with pytest.raises(AssertionError, match="unsupported honest_verdict"):
        nrgpt.run_per_token_experiment(
            dataset_path=tmp_path / "missing.jsonl",
            corpus_path=corpus,
            baseline_artifact_path=baseline,
            deliverable_path=tmp_path / "bad_verdict.json",
            n_train=10,
            n_eval=6,
        )
