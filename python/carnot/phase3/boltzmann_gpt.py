"""Boltzmann-GPT contrastive training helpers for REQ-KONA-019."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch

from carnot.data.fover import FoVerDataset, FoVerItem

N_EPOCHS = 10
NRGPT_AUROC_BASELINE = 0.921
SEED_AUROC_BASELINE = 0.65
HONEST_VERDICTS = {
    "contrastive_auroc_above_0p80",
    "contrastive_auroc_improved_below_threshold",
    "training_diverged",
    "blocked",
}
REQUIRED_ARTIFACT_FIELDS = {
    "n_training_epochs",
    "n_train_samples",
    "n_test_samples",
    "boltzmann_gpt_contrastive_auroc",
    "nrgpt_auroc_baseline",
    "boltzmann_gpt_beats_seed",
    "boltzmann_gpt_above_0p80",
    "checkpoint_path",
    "honest_verdict",
}


class BoltzmannGPTLayer(torch.nn.Module):
    """Visible-hidden Boltzmann energy layer with trainable torch parameters."""

    def __init__(self, *, visible_dim: int = 16, hidden_dim: int = 16, seed: int = 42) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.W = torch.nn.Parameter(
            torch.randn(visible_dim, hidden_dim, generator=generator) * 0.01
        )
        self.b = torch.nn.Parameter(torch.zeros(visible_dim))
        self.c = torch.nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, visible: torch.Tensor) -> torch.Tensor:
        hidden = torch.sigmoid(visible @ self.W + self.c)
        coupling = torch.sum((visible @ self.W) * hidden, dim=1)
        visible_bias = visible @ self.b
        hidden_bias = hidden @ self.c
        return -(coupling + visible_bias + hidden_bias)


def embed_texts(texts: list[str], *, visible_dim: int = 16) -> torch.Tensor:
    """Project text into deterministic L2-normalized character-bigram features."""

    rows: list[torch.Tensor] = []
    for text in texts:
        counts = torch.zeros(visible_dim, dtype=torch.float32)
        for token in text.split():
            for index in range(len(token) - 1):
                bucket = (ord(token[index]) * 31 + ord(token[index + 1])) % visible_dim
                counts[bucket] += 1.0
        norm = torch.linalg.vector_norm(counts)
        rows.append(counts / norm if norm > 0 else torch.full_like(counts, 1.0 / visible_dim))
    return torch.stack(rows)


def split_dataset(
    dataset: FoVerDataset,
    *,
    test_fraction: float = 0.2,
    seed: int = 1237,
) -> tuple[list[FoVerItem], list[FoVerItem]]:
    """Return a deterministic stratified train/test split."""

    rng = random.Random(seed)
    positives = [item for item in dataset if item.label == 1]
    negatives = [item for item in dataset if item.label == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    n_pos_test = max(1, round(len(positives) * test_fraction))
    n_neg_test = max(1, round(len(negatives) * test_fraction))
    test = positives[:n_pos_test] + negatives[:n_neg_test]
    train = positives[n_pos_test:] + negatives[n_neg_test:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def train_contrastive(
    model: BoltzmannGPTLayer,
    items: list[FoVerItem],
    *,
    n_epochs: int = N_EPOCHS,
    lr: float = 1e-3,
    batch_size: int = 16,
    visible_dim: int = 16,
    seed: int = 1237,
) -> list[float]:
    """Minimize ``E_correct.mean() - E_incorrect.mean()`` on FoVer traces."""

    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    positives = embed_texts(
        [item.text for item in items if item.label == 1], visible_dim=visible_dim
    )
    negatives = embed_texts(
        [item.text for item in items if item.label == 0], visible_dim=visible_dim
    )
    history: list[float] = []
    del batch_size
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = model(positives).mean() - model(negatives).mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def evaluate_energy_gap(
    model: BoltzmannGPTLayer,
    items: list[FoVerItem],
    *,
    visible_dim: int = 16,
) -> float:
    """Return mean incorrect energy minus mean correct energy."""

    positives = embed_texts(
        [item.text for item in items if item.label == 1], visible_dim=visible_dim
    )
    negatives = embed_texts(
        [item.text for item in items if item.label == 0], visible_dim=visible_dim
    )
    with torch.no_grad():
        return float(model(negatives).mean() - model(positives).mean())


def derive_honest_verdict(auroc: float) -> str:
    """Map measured held-out AUROC to the REQ-KONA-019 verdict vocabulary."""

    if math.isnan(auroc) or auroc <= SEED_AUROC_BASELINE:
        return "training_diverged"
    if auroc > 0.80:
        return "contrastive_auroc_above_0p80"
    return "contrastive_auroc_improved_below_threshold"


def run_experiment(
    *,
    rows: list[dict[str, str]] | None = None,
    artifact_path: str | Path = "results/experiment_1237_boltzmann_gpt_contrastive_training.json",
    checkpoint_path: str | Path = "python/carnot/phase3/boltzmann_gpt_cd_v1.pt",
    n_epochs: int = N_EPOCHS,
    lr: float = 1e-3,
    batch_size: int = 16,
    visible_dim: int = 16,
    hidden_dim: int = 16,
    seed: int = 1237,
    device: str = "cpu",
) -> dict[str, object]:
    """Train Boltzmann-GPT on FoVer and write the Exp 1237 artifact."""

    dataset = FoVerDataset(rows=rows) if rows is not None else FoVerDataset()
    train, test = split_dataset(dataset, test_fraction=0.2, seed=seed)
    model = BoltzmannGPTLayer(visible_dim=visible_dim, hidden_dim=hidden_dim, seed=seed).to(device)
    train_contrastive(
        model,
        train,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        visible_dim=visible_dim,
        seed=seed,
    )
    test_visible = embed_texts([item.text for item in test], visible_dim=visible_dim).to(device)
    with torch.no_grad():
        scores = (-model(test_visible)).detach().cpu().tolist()
    labels = [item.label for item in test]
    auroc = _binary_auroc(labels, scores)
    artifact = {
        "experiment": "1237_boltzmann_gpt_contrastive_training",
        "n_training_epochs": n_epochs,
        "n_train_samples": len(train),
        "n_test_samples": len(test),
        "boltzmann_gpt_contrastive_auroc": auroc,
        "nrgpt_auroc_baseline": NRGPT_AUROC_BASELINE,
        "boltzmann_gpt_beats_seed": auroc > SEED_AUROC_BASELINE,
        "boltzmann_gpt_above_0p80": auroc > 0.80,
        "checkpoint_path": str(checkpoint_path),
        "honest_verdict": derive_honest_verdict(auroc),
    }
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    artifact_file = Path(artifact_path)
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact


def _binary_auroc(labels: list[int], scores: list[float]) -> float:
    positives = [(score, label) for score, label in zip(scores, labels) if label == 1]
    negatives = [(score, label) for score, label in zip(scores, labels) if label == 0]
    wins = 0.0
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            wins += float(pos_score > neg_score) + (0.5 * float(pos_score == neg_score))
    return wins / (len(positives) * len(negatives))
