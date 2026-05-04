"""Tests for Exp 1237 Boltzmann-GPT contrastive training.

Spec refs: REQ-KONA-019, SCENARIO-KONA-019.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.data.fover import FoVerDataset  # noqa: E402
from carnot.phase3 import boltzmann_gpt as bgpt  # noqa: E402


def _toy_rows(n_pairs: int = 8) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for i in range(n_pairs):
        rows.append(
            {
                "step_text": f"valid proof clean arithmetic balance exact total {i} good",
                "label": "correct",
            }
        )
        rows.append(
            {
                "step_text": f"invalid proof wrong arithmetic mismatch error total {i} bad",
                "label": "incorrect",
            }
        )
    return rows


def test_fover_dataset_loads_binary_labels_from_rows() -> None:
    """REQ-KONA-019: FoVerDataset exposes text with 1=correct and 0=incorrect."""
    dataset = FoVerDataset(rows=_toy_rows(2))

    assert len(dataset) == 4
    assert dataset[0].label == 1
    assert dataset[1].label == 0
    assert dataset.labels == [1, 0, 1, 0]
    assert dataset.texts[0].startswith("valid proof")

    with pytest.raises(ValueError, match="both correct and incorrect"):
        FoVerDataset(rows=[{"step_text": "only one class", "label": "correct"}])


def test_layer_forward_returns_finite_energy_and_state_dict() -> None:
    """REQ-KONA-019: BoltzmannGPTLayer is trainable and checkpointable."""
    model = bgpt.BoltzmannGPTLayer(visible_dim=4, hidden_dim=3, seed=7)
    visible = torch.eye(4, dtype=torch.float32)[:3]

    energies = model(visible)

    assert energies.shape == (3,)
    assert torch.isfinite(energies).all()
    assert set(model.state_dict()) == {"W", "b", "c"}


def test_text_embedding_and_split_are_deterministic() -> None:
    """SCENARIO-KONA-019: FoVer rows use a deterministic 80/20 held-out split."""
    dataset = FoVerDataset(rows=_toy_rows(10))

    train_a, test_a = bgpt.split_dataset(dataset, test_fraction=0.2, seed=1237)
    train_b, test_b = bgpt.split_dataset(dataset, test_fraction=0.2, seed=1237)
    visible_a = bgpt.embed_texts(["same text", "same text"], visible_dim=8)
    visible_b = bgpt.embed_texts(["same text", "same text"], visible_dim=8)

    assert len(train_a) == 16
    assert len(test_a) == 4
    assert [item.label for item in train_a] == [item.label for item in train_b]
    assert [item.text for item in test_a] == [item.text for item in test_b]
    assert torch.allclose(visible_a, visible_b)
    assert torch.linalg.vector_norm(visible_a, dim=1).tolist() == pytest.approx([1.0, 1.0])


def test_contrastive_training_increases_held_out_energy_gap() -> None:
    """REQ-KONA-019: CD training pushes correct traces below incorrect traces."""
    dataset = FoVerDataset(rows=_toy_rows(12))
    train, test = bgpt.split_dataset(dataset, test_fraction=0.25, seed=5)
    model = bgpt.BoltzmannGPTLayer(visible_dim=8, hidden_dim=4, seed=5)

    before = bgpt.evaluate_energy_gap(model, test, visible_dim=8)
    history = bgpt.train_contrastive(
        model,
        train,
        n_epochs=6,
        lr=1e-2,
        batch_size=4,
        visible_dim=8,
        seed=5,
    )
    after = bgpt.evaluate_energy_gap(model, test, visible_dim=8)

    assert len(history) == 6
    assert all(torch.isfinite(torch.tensor(value)) for value in history)
    assert after > before


def test_run_experiment_writes_required_artifact_and_checkpoint(tmp_path: Path) -> None:
    """SCENARIO-KONA-019: Exp 1237 writes the JSON schema and checkpoint."""
    artifact_path = tmp_path / "experiment_1237.json"
    checkpoint_path = tmp_path / "boltzmann_gpt_cd_v1.pt"

    artifact = bgpt.run_experiment(
        rows=_toy_rows(20),
        artifact_path=artifact_path,
        checkpoint_path=checkpoint_path,
        n_epochs=3,
        lr=1e-2,
        batch_size=4,
        visible_dim=8,
        hidden_dim=4,
        seed=11,
        device="cpu",
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    assert checkpoint_path.exists()
    assert bgpt.REQUIRED_ARTIFACT_FIELDS <= set(payload)
    assert payload["experiment"] == "1237_boltzmann_gpt_contrastive_training"
    assert payload["n_training_epochs"] == 3
    assert payload["n_train_samples"] == 32
    assert payload["n_test_samples"] == 8
    assert 0.0 <= payload["boltzmann_gpt_contrastive_auroc"] <= 1.0
    assert payload["nrgpt_auroc_baseline"] == pytest.approx(0.921)
    assert payload["checkpoint_path"] == str(checkpoint_path)
    assert payload["honest_verdict"] in bgpt.HONEST_VERDICTS
