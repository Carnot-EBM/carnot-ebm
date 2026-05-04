"""Exp 1265 DiffuTruth semantic energy comparison.

Spec: REQ-VERIFY-1265, SCENARIO-VERIFY-1265
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FOVER_PATH = REPO_ROOT / "results" / "fover_corpus_v5.json"
DEFAULT_CARNOT_BASELINE_PATH = (
    REPO_ROOT / "results" / "experiment_1096_semenergy_probe_v1.json"
)
DEFAULT_RESULT_PATH = (
    REPO_ROOT / "results" / "experiment_1265_diffutruth_vs_carnot_baseline.json"
)
EXPERIMENT_NAME = "1265_diffutruth_vs_carnot_baseline"
DIFFUTRUTH_FEVER_PAPER_AUROC = 0.725
REQUIRED_ARTIFACT_FIELDS: set[str] = {
    "diffutruth_semantic_energy_auroc",
    "carnot_semenergy_probe_auroc",
    "diffutruth_fever_paper_auroc",
    "carnot_beats_diffutruth_paper",
    "diffutruth_comparison_measured",
    "honest_verdict",
}

@dataclass(frozen=True)
class FoVerExample:
    """One FoVer response with label 1 for hallucination and 0 for correct."""

    response: str
    label: int


def reconstruct_response(response: str) -> str:
    """Remove the final sentence to simulate DiffuTruth reconstruction."""

    sentences = response.strip().split(". ")
    if len(sentences) <= 1:
        return response.strip()
    return ". ".join(sentences[:-1])


def semantic_energy(response: str) -> float:
    """Return TF-IDF cosine divergence between original and reconstructed text."""

    sentences = response.strip().split(". ")
    if len(sentences) <= 1:
        return 0.0
    original = response
    reconstructed = ". ".join(sentences[:-1])
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(max_features=200)
        matrix = vectorizer.fit_transform([original, reconstructed])
        similarity = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
        return 1.0 - similarity
    except Exception:  # pragma: no cover - mirrors the roadmap fallback.
        return float(len(sentences[-1])) / max(len(response), 1)


def tie_aware_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC with 0.5 tie credit and 0.5 for single-class inputs."""

    if len(labels) != len(scores):  # pragma: no cover - defensive contract check.
        raise ValueError(f"labels and scores must have same length: {len(labels)} vs {len(scores)}")

    positives = [
        float(score)
        for label, score in zip(labels, scores, strict=True)
        if int(label) == 1
    ]
    negatives = [
        float(score)
        for label, score in zip(labels, scores, strict=True)
        if int(label) == 0
    ]
    if not positives or not negatives:
        return 0.5

    wins = 0.0
    ties = 0.0
    for positive_score in positives:
        for negative_score in negatives:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                ties += 1.0
    return float((wins + 0.5 * ties) / (len(positives) * len(negatives)))


def load_fover_pairs(
    path: Path | str = DEFAULT_FOVER_PATH,
    *,
    limit: int = 100,
) -> list[FoVerExample]:
    """Load the first ``limit`` FoVer pairs and map incorrect rows to label 1."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        FoVerExample(
            response=str(pair.get("response", "")),
            label=int(not bool(pair.get("is_correct", True))),
        )
        for pair in payload["pairs"][:limit]
    ]


def load_carnot_baseline_auroc(path: Path | str = DEFAULT_CARNOT_BASELINE_PATH) -> float:
    """Read Carnot's SemEnergyProbe AUROC from the known baseline artifact keys."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("auroc", "auc", "semenergy_auroc"):
        value = payload.get(key)
        if value is not None:
            return float(value)
    raise KeyError(  # pragma: no cover
        "baseline artifact does not contain auroc, auc, or semenergy_auroc"
    )


def compute_diffutruth_auroc(examples: Sequence[FoVerExample]) -> tuple[float, list[float]]:
    """Score FoVer responses with the DiffuTruth proxy and return AUROC plus scores."""

    labels = [example.label for example in examples]
    scores = [semantic_energy(example.response) for example in examples]
    return tie_aware_auroc(labels, scores), scores


def build_artifact(
    examples: Sequence[FoVerExample],
    scores: Sequence[float],
    *,
    diffutruth_auroc: float,
    carnot_auroc: float,
) -> dict[str, Any]:
    """Build the required Exp 1265 comparison artifact."""

    score_values = [float(score) for score in scores]
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "status": "complete",
        "n_pairs": len(examples),
        "n_correct": sum(1 for example in examples if example.label == 0),
        "n_hallucination": sum(1 for example in examples if example.label == 1),
        "diffutruth_score_min": round(min(score_values), 6) if score_values else 0.0,
        "diffutruth_score_max": round(max(score_values), 6) if score_values else 0.0,
        "diffutruth_semantic_energy_auroc": round(float(diffutruth_auroc), 4),
        "carnot_semenergy_probe_auroc": float(carnot_auroc),
        "diffutruth_fever_paper_auroc": DIFFUTRUTH_FEVER_PAPER_AUROC,
        "carnot_beats_diffutruth_paper": float(carnot_auroc) > DIFFUTRUTH_FEVER_PAPER_AUROC,
        "carnot_beats_diffutruth_fover": float(carnot_auroc) > float(diffutruth_auroc),
        "comparison_note": (
            f"DiffuTruth semantic energy AUROC on FoVer: {diffutruth_auroc:.4f}; "
            f"Carnot SemEnergyProbe: {carnot_auroc:.4f}; "
            f"DiffuTruth FEVER paper: {DIFFUTRUTH_FEVER_PAPER_AUROC}"
        ),
        "diffutruth_comparison_measured": True,
        "honest_verdict": f"diffutruth_fover_{diffutruth_auroc:.3f}_carnot_{carnot_auroc:.3f}",
    }
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:  # pragma: no cover - protects future schema edits.
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    return artifact


def run_experiment(
    *,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    carnot_baseline_path: Path | str = DEFAULT_CARNOT_BASELINE_PATH,
    output_path: Path | str = DEFAULT_RESULT_PATH,
    limit: int = 100,
) -> dict[str, Any]:
    """Run Exp 1265 and persist the DiffuTruth-vs-Carnot artifact."""

    examples = load_fover_pairs(fover_path, limit=limit)
    diffutruth_auroc, scores = compute_diffutruth_auroc(examples)
    carnot_auroc = load_carnot_baseline_auroc(carnot_baseline_path)
    artifact = build_artifact(
        examples,
        scores,
        diffutruth_auroc=diffutruth_auroc,
        carnot_auroc=carnot_auroc,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":  # pragma: no cover
    run_experiment()
