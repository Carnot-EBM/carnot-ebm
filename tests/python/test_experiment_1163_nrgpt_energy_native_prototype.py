"""Tests for Exp 1163 NRGPT energy recurrence Phase 3 seed.

Spec refs: REQ-KONA-011, SCENARIO-KONA-010, SCENARIO-KONA-011.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.phase3 import nrgpt_energy as nrgpt  # noqa: E402
from scripts import experiment_1163_nrgpt_energy_native_prototype as exp1163  # noqa: E402


def _tiny_rows() -> list[dict[str, str]]:
    return [
        {"step_text": "2 + 2 = 4 and the arithmetic is correct", "label": "correct"},
        {"step_text": "3 + 3 = 6 and the arithmetic is correct", "label": "correct"},
        {"step_text": "4 + 4 = 8 and the arithmetic is correct", "label": "correct"},
        {"step_text": "2 + 2 = 5 and the arithmetic is incorrect", "label": "incorrect"},
        {"step_text": "3 + 3 = 7 and the arithmetic is incorrect", "label": "incorrect"},
        {"step_text": "4 + 4 = 9 and the arithmetic is incorrect", "label": "incorrect"},
    ]


def test_nrgpt_energy_block_preserves_shape_and_iteration_count() -> None:
    """REQ-KONA-011: NRGPTEnergyBlock applies bounded recurrent energy updates."""
    block = nrgpt.NRGPTEnergyBlock(d_emb=8, d_energy=4, n_iters=3, seed=7)
    z = np.ones((5, 8), dtype=np.float32) * 0.25

    refined = block(z)
    one_step = block.copy_with_n_iters(1)(z)

    assert refined.shape == z.shape
    assert block.energy(z).shape == (5,)
    assert block.energy_grad(z).shape == z.shape
    assert np.isfinite(refined).all()
    assert not np.allclose(refined, one_step)
    assert block.n_linear_layers == 3


def test_energy_block_bounds_large_updates() -> None:
    """REQ-KONA-011: recurrent updates stay inside the configured latent bound."""
    block = nrgpt.NRGPTEnergyBlock(
        d_emb=3,
        d_energy=2,
        n_iters=2,
        step_size=10.0,
        grad_clip=0.1,
        state_clip=0.2,
        seed=3,
    )
    z = np.full((2, 3), 5.0, dtype=np.float32)

    refined = block.forward(z)

    assert float(np.max(np.abs(refined))) <= 0.200001


def test_hashing_embeddings_and_labels_are_deterministic() -> None:
    """SCENARIO-KONA-010: FoVer rows become fixed embeddings plus binary labels."""
    embedder = nrgpt.HashingFoVerEmbedder(dim=16)
    rows = _tiny_rows()

    first = nrgpt.fover_rows_to_arrays(rows, embedder)
    second = nrgpt.fover_rows_to_arrays(rows, embedder)

    assert first.embeddings.shape == (6, 16)
    assert np.allclose(first.embeddings, second.embeddings)
    assert first.labels.tolist() == [1, 1, 1, 0, 0, 0]
    assert np.linalg.norm(first.embeddings, axis=1).min() > 0.0


def test_label_and_row_validation_errors_are_clear() -> None:
    """REQ-KONA-011: labels must be binary correct/incorrect values."""
    probe = nrgpt.LogisticProbe(d_emb=2, seed=1)
    probs = probe.probabilities(np.zeros((1, 2), dtype=np.float32))

    assert probs.shape == (1,)
    assert nrgpt.label_to_int("correct") == 1
    assert nrgpt.label_to_int("incorrect") == 0
    assert nrgpt.label_to_int(True) == 1
    assert nrgpt.label_to_int(0) == 0
    assert nrgpt.response_text({"response": "fallback"}) == "fallback"
    assert nrgpt.response_text({"text": "plain"}) == "plain"
    assert nrgpt._balanced_weights(np.array([1.0, 1.0])).tolist() == [1.0, 1.0]

    with pytest.raises(ValueError, match="unsupported FoVer label"):
        nrgpt.label_to_int("maybe")
    with pytest.raises(ValueError, match="embedding dim"):
        nrgpt.HashingFoVerEmbedder(dim=0)
    with pytest.raises(ValueError, match="d_emb"):
        nrgpt.LogisticProbe(d_emb=0)
    with pytest.raises(ValueError, match="d_emb"):
        nrgpt.NRGPTEnergyBlock(d_emb=0)
    with pytest.raises(ValueError, match="d_energy"):
        nrgpt.NRGPTEnergyBlock(d_emb=2, d_energy=0)
    with pytest.raises(ValueError, match="n_iters"):
        nrgpt.NRGPTEnergyBlock(d_emb=2, n_iters=0)
    with pytest.raises(ValueError, match="missing response text"):
        nrgpt.fover_rows_to_arrays([{"label": "correct"}], nrgpt.HashingFoVerEmbedder(dim=4))
    with pytest.raises(ValueError, match="expected 1D or 2D"):
        nrgpt._as_2d(np.zeros((1, 1, 1), dtype=np.float32))


def test_split_is_stratified_and_normalization_reuses_train_stats() -> None:
    """SCENARIO-KONA-010: train/eval splits preserve both label classes."""
    rows = nrgpt.build_synthetic_fover_rows(n_pairs=40, seed=11)

    train, eval_rows = nrgpt.split_fover_rows(rows, n_train=20, n_eval=10, seed=5)
    labels_train = [nrgpt.label_to_int(row["label"]) for row in train]
    labels_eval = [nrgpt.label_to_int(row["label"]) for row in eval_rows]
    X_train = np.array([[1.0, 1.0], [2.0, 1.0]], dtype=np.float32)
    X_eval = np.array([[3.0, 1.0]], dtype=np.float32)

    normalized_train, normalized_eval, stats = nrgpt.normalize_train_eval(X_train, X_eval)

    assert len(train) == 20
    assert len(eval_rows) == 10
    assert set(labels_train) == {0, 1}
    assert set(labels_eval) == {0, 1}
    assert normalized_train[:, 1].tolist() == [0.0, 0.0]
    assert normalized_eval.shape == (1, 2)
    assert stats["mean"].shape == (2,)


def test_split_and_row_reader_cover_small_and_empty_inputs(tmp_path: Path) -> None:
    """SCENARIO-KONA-010: small corpora and JSON-list files stay deterministic."""
    rows = nrgpt.build_synthetic_fover_rows(n_pairs=12, seed=19)
    train, eval_rows = nrgpt.split_fover_rows(rows, n_train=20, n_eval=10, seed=19)
    json_path = tmp_path / "rows.json"
    empty_path = tmp_path / "empty.jsonl"
    json_path.write_text(json.dumps(rows[:3]), encoding="utf-8")
    empty_path.write_text("", encoding="utf-8")

    vector, was_vector = nrgpt._as_2d(np.zeros(3, dtype=np.float32))

    assert len(train) + len(eval_rows) == 12
    assert len(eval_rows) == 2
    assert nrgpt._read_rows(json_path) == rows[:3]
    assert nrgpt._read_rows(empty_path) == []
    assert vector.shape == (1, 3)
    assert was_vector is True

    with pytest.raises(ValueError, match="both correct and incorrect"):
        nrgpt.split_fover_rows(
            [{"step_text": "all good", "label": "correct"}],
            n_train=1,
            n_eval=1,
        )


def test_training_pipeline_reports_baseline_and_nrgpt_auc() -> None:
    """REQ-KONA-011: train/evaluate returns baseline, n=1, and n=3 AUROC values."""
    rows = nrgpt.build_synthetic_fover_rows(n_pairs=80, seed=17)
    train, eval_rows = nrgpt.split_fover_rows(rows, n_train=50, n_eval=20, seed=17)
    result = nrgpt.train_and_compare(
        train,
        eval_rows,
        d_emb=32,
        d_energy=8,
        energy_epochs=2,
        head_epochs=3,
        seed=17,
    )

    assert 0.0 <= result.baseline_auroc <= 1.0
    assert 0.0 <= result.nrgpt_auroc_n1 <= 1.0
    assert 0.0 <= result.nrgpt_auroc_n3 <= 1.0
    assert result.energy_block.n_iters == 3
    assert result.baseline.continuous_ebm.variables == 32


def test_artifact_schema_and_honest_verdict_are_stable() -> None:
    """SCENARIO-KONA-011: artifact reports iteration monotonicity without bias."""
    artifact = nrgpt.build_artifact(
        n_training_pairs=5000,
        n_eval_pairs=500,
        baseline_auroc=0.6,
        nrgpt_auroc_n1=0.62,
        nrgpt_auroc_n3=0.61,
        fover_data_source="pipeline_generated",
        duration_s=1.25,
    )

    assert nrgpt.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["nrgpt_above_baseline"] is True
    assert artifact["n_iters_monotone"] is False
    assert artifact["energy_block_module_written"] is True
    assert artifact["nrgpt_phase3_prototype_honest_result"] is True
    assert artifact["honest_verdict"] == "nrgpt_above_baseline_energy_recurrence_helps"

    tie = nrgpt.classify_honest_verdict(0.5, 0.5, "pipeline_generated")
    below = nrgpt.classify_honest_verdict(0.7, 0.6, "pipeline_generated")
    synthetic = nrgpt.classify_honest_verdict(0.7, 0.8, "synthetic")
    assert tie == "nrgpt_ties_baseline"
    assert below == "nrgpt_below_baseline"
    assert synthetic == "fover_data_not_found_synthetic_used"


def test_load_fover_rows_prefers_dataset_then_pipeline_then_synthetic(tmp_path: Path) -> None:
    """SCENARIO-KONA-010: FoVer source labels match the selected data path."""
    dataset = tmp_path / "fover_dataset.jsonl"
    corpus = tmp_path / "fover_corpus.jsonl"
    dataset.write_text(json.dumps(_tiny_rows()[0]) + "\n", encoding="utf-8")
    corpus.write_text(json.dumps(_tiny_rows()[1]) + "\n", encoding="utf-8")

    rows, source = nrgpt.load_fover_rows(dataset_path=dataset, corpus_path=corpus)
    assert source == "data/fover_dataset.jsonl"
    assert rows[0]["label"] == "correct"

    dataset.unlink()
    rows, source = nrgpt.load_fover_rows(dataset_path=dataset, corpus_path=corpus)
    assert source == "pipeline_generated"
    assert rows[0]["label"] == "correct"

    corpus.unlink()
    rows, source = nrgpt.load_fover_rows(
        dataset_path=dataset,
        corpus_path=corpus,
        synthetic_pairs=6,
    )
    assert source == "synthetic"
    assert len(rows) == 6


def test_run_experiment_and_script_main_write_deliverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-KONA-010: runner writes the required Exp 1163 JSON fields."""
    corpus = tmp_path / "fover_corpus.jsonl"
    deliverable = tmp_path / "experiment_1163.json"
    corpus.write_text(
        "\n".join(json.dumps(row) for row in nrgpt.build_synthetic_fover_rows(80)) + "\n",
        encoding="utf-8",
    )

    artifact = nrgpt.run_experiment(
        dataset_path=tmp_path / "missing.jsonl",
        corpus_path=corpus,
        deliverable_path=deliverable,
        n_train=50,
        n_eval=20,
        d_emb=32,
        d_energy=8,
        energy_epochs=2,
        head_epochs=3,
        seed=23,
    )
    payload = json.loads(deliverable.read_text(encoding="utf-8"))

    assert payload == artifact
    assert nrgpt.REQUIRED_ARTIFACT_FIELDS <= set(payload)
    assert payload["n_training_pairs"] == 50
    assert payload["n_eval_pairs"] == 20
    assert payload["fover_data_source"] == "pipeline_generated"
    assert payload["honest_verdict"] in nrgpt.HONEST_VERDICTS

    monkeypatch.setattr(exp1163, "DATASET_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(exp1163, "CORPUS_PATH", corpus)
    monkeypatch.setattr(exp1163, "DELIVERABLE", tmp_path / "script_result.json")
    monkeypatch.setattr(exp1163, "N_TRAIN", 50)
    monkeypatch.setattr(exp1163, "N_EVAL", 20)
    monkeypatch.setattr(exp1163, "D_EMB", 32)
    monkeypatch.setattr(exp1163, "D_ENERGY", 8)
    monkeypatch.setattr(exp1163, "ENERGY_EPOCHS", 2)
    monkeypatch.setattr(exp1163, "HEAD_EPOCHS", 3)

    assert exp1163.main() == 0
    script_payload = json.loads((tmp_path / "script_result.json").read_text(encoding="utf-8"))
    assert nrgpt.REQUIRED_ARTIFACT_FIELDS <= set(script_payload)
