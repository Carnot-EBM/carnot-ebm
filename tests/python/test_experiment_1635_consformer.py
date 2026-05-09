"""Tests for Exp 1635 ConsFormer-style FoVer refiner.

Spec: REQ-LEARN-1635, SCENARIO-LEARN-1635.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import experiment_1635_consformer as mod


def _write_jsonl(path: Path, rows: list[dict[str, object] | str]) -> Path:
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_req_learn_1635_loads_valid_fover_rows_in_order(tmp_path: Path) -> None:
    """REQ-LEARN-1635-2: loader skips malformed rows and preserves order."""

    corpus = _write_jsonl(
        tmp_path / "fover.jsonl",
        [
            {"row_id": "a", "step_text": "2 + 3 = 5", "label": "correct"},
            "not-json",
            {"row_id": "missing-step", "label": "correct"},
            {"row_id": "bad-label", "step_text": "2 + 3 = 5", "label": "maybe"},
            {"question_id": "q2", "step_text": "2 + 3 = 8", "label": "incorrect"},
        ],
    )

    rows = mod.load_fover_rows(corpus)

    assert [row.row_id for row in rows] == ["a", "q2:5"]
    assert [row.label for row in rows] == [True, False]
    assert rows[0].step_text == "2 + 3 = 5"


def test_req_learn_1635_refiner_scores_consistent_constraint_higher() -> None:
    """REQ-LEARN-1635-3: self-attention refiner learns from CSP features only."""

    correct = mod.FoVerRow("ok", "2 + 3 = 5. Therefore the total is 5.", True)
    incorrect = mod.FoVerRow("bad", "2 + 3 = 8. Therefore the total is 8.", False)
    refiner = mod.train_label_free_refiner([correct, incorrect])

    correct_score = refiner.score(mod.extract_csp_features(correct.step_text))
    incorrect_score = refiner.score(mod.extract_csp_features(incorrect.step_text))

    assert correct_score > incorrect_score
    assert refiner.label_free_training is True


def test_req_learn_1635_training_is_label_free() -> None:
    """REQ-LEARN-1635-3: changing labels does not change the trained refiner."""

    rows = [
        mod.FoVerRow("a", "10 - 4 = 6. The remainder is 6.", True),
        mod.FoVerRow("b", "10 - 4 = 9. The remainder is 9.", False),
        mod.FoVerRow("c", "3 * 7 = 21. The product is 21.", True),
    ]
    flipped = [replace(row, label=not row.label) for row in rows]

    trained = mod.train_label_free_refiner(rows).to_json()
    trained_flipped = mod.train_label_free_refiner(flipped).to_json()

    assert trained == trained_flipped


def test_scenario_learn_1635_run_writes_refiner_accuracy(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1635: run writes a complete artifact with accuracy fields."""

    corpus = _write_jsonl(
        tmp_path / "fover.jsonl",
        [
            {"row_id": "r1", "step_text": "2 + 2 = 4. Therefore 4.", "label": "correct"},
            {"row_id": "r2", "step_text": "2 + 2 = 7. Therefore 7.", "label": "incorrect"},
            {"row_id": "r3", "step_text": "8 / 2 = 4. Therefore 4.", "label": "correct"},
            {"row_id": "r4", "step_text": "8 / 2 = 5. Therefore 5.", "label": "incorrect"},
            {"row_id": "r5", "step_text": "9 - 3 = 6. Therefore 6.", "label": "correct"},
            {"row_id": "r6", "step_text": "9 - 3 = 1. Therefore 1.", "label": "incorrect"},
        ],
    )
    output_path = tmp_path / "results" / "experiment_1635_consformer.json"

    artifact = mod.run_experiment(
        corpus_path=corpus,
        output_path=output_path,
        tests_run=[".venv/bin/pytest tests/python/test_experiment_1635_consformer.py -q"],
        eval_fraction=0.5,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["spec_refs"] == mod.SPEC_REFS
    assert artifact["dataset_rows"] == 6
    assert artifact["train_rows"] == 3
    assert artifact["eval_rows"] == 3
    assert artifact["label_free_training"] is True
    assert 0.0 <= artifact["refiner_accuracy"] <= 1.0
    assert 0.0 <= artifact["baseline_accuracy"] <= 1.0


def test_req_learn_1635_validation_rejects_bad_artifacts() -> None:
    """REQ-LEARN-1635-4/5: artifact validation enforces required fields."""

    artifact = mod.build_artifact(
        rows=[
            mod.FoVerRow("a", "1 + 1 = 2", True),
            mod.FoVerRow("b", "1 + 1 = 3", False),
            mod.FoVerRow("c", "2 + 2 = 4", True),
            mod.FoVerRow("d", "2 + 2 = 5", False),
        ],
        tests_run=(),
        eval_fraction=0.5,
    )

    missing = dict(artifact)
    del missing["refiner_accuracy"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)

    with pytest.raises(AssertionError, match="refiner_accuracy"):
        mod.validate_artifact(dict(artifact, refiner_accuracy=1.1))

    with pytest.raises(AssertionError, match="label_free_training"):
        mod.validate_artifact(dict(artifact, label_free_training=False))

    with pytest.raises(AssertionError, match="train_rows"):
        mod.validate_artifact(dict(artifact, train_rows=0))


def test_req_learn_1635_edge_paths_remain_deterministic(tmp_path: Path) -> None:
    """REQ-LEARN-1635-2/3/5: edge cases fail deterministically."""

    corpus = _write_jsonl(
        tmp_path / "fover.jsonl",
        [
            {"row_id": "a", "step_text": "-2 + 3 = 1", "label": "correct"},
            [{"not": "a dict"}],
            {"row_id": "b", "step_text": "1 + 1 = 2", "label": "correct"},
        ],
    )

    assert [row.row_id for row in mod.load_fover_rows(corpus, limit=1)] == ["a"]
    assert [row.row_id for row in mod.load_fover_rows(corpus)] == ["a", "b"]
    assert mod._safe_arithmetic_value("-2") == -2.0
    assert mod._safe_arithmetic_value("name") is None

    bad_features = mod.extract_csp_features("1 / 0 = 0 and no final punctuation")
    no_equation = mod.extract_csp_features("")

    assert bad_features.equation_presence == 0.0
    assert no_equation.equality_consistency == pytest.approx(0.56)
    assert no_equation.repetition_absence == pytest.approx(0.5)

    with pytest.raises(ValueError, match="train row"):
        mod.train_label_free_refiner([])

    with pytest.raises(ValueError, match="train/eval split"):
        mod.split_rows([mod.FoVerRow("solo", "1 + 1 = 2", True)])
