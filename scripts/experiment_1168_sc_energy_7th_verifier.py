#!/usr/bin/env python3
"""Exp 1168: SC-Energy as k=6 verifier candidate.

Spec: REQ-VERIFY-1168, SCENARIO-VERIFY-1168
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.ast_structure_verifier import ASTStructureVerifier  # noqa: E402
from carnot.verify.sc_energy_verifier import SCEnergyVerifier  # noqa: E402
from carnot.verify.semantic_consistency_verifier import SemanticConsistencyVerifier  # noqa: E402
from carnot.verify.semenergy_probe import SemEnergyProbe  # noqa: E402
from carnot.verify.z3_math_verifier import Z3MathVerifier  # noqa: E402


EXPERIMENT_ID = 1168
RANDOM_SEED = 1168
SC_AUROC_THRESHOLD = 0.65
R_CORR_THRESHOLD = 0.5
OUTPUT_PATH = PROJECT_ROOT / "results" / "experiment_1168_sc_energy_7th_verifier.json"
CORRELATION_CORPUS_PATH = PROJECT_ROOT / "data" / "fover_corpus_v4.json"
VERIFIER_MODULE_PATH = "python/carnot/verify/sc_energy_verifier.py"


def latest_labeled_pairs_path(results_dir: Path) -> Path:
    """Return the latest named FoVer labeled-pairs JSON in results/."""
    candidates = sorted(results_dir.glob("fover_labeled*.json"))
    if not candidates:
        raise FileNotFoundError("no results/fover_labeled*.json corpus found")
    return candidates[-1]


def load_rows(path: Path) -> list[dict]:
    """Load list-style or {pairs: [...]} FoVer JSON rows."""
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        rows = data.get("pairs") or data.get("items") or data.get("examples") or []
    else:
        rows = data
    if not isinstance(rows, list):
        raise ValueError(f"unsupported corpus shape in {path}")
    return [row for row in rows if isinstance(row, dict)]


def build_contrastive_pairs(rows: list[dict]) -> list[dict]:
    """Build coherent/incoherent context-response pairs grouped by question id."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        qid = str(row.get("question_id") or row.get("question") or row.get("id") or "")
        if qid:
            grouped[qid].append(row)

    pairs: list[dict] = []
    for qid, items in sorted(grouped.items()):
        correct = [_step_text(row) for row in items if _is_correct(row)]
        incorrect = [_step_text(row) for row in items if not _is_correct(row)]
        correct = [step for step in correct if step]
        incorrect = [step for step in incorrect if step]
        if len(correct) < 2 or not incorrect:
            continue
        pairs.append(
            {
                "qid": qid,
                "coherent": (correct[-1], " ".join(correct[:-1])),
                "incoherent": (incorrect[0], " ".join(correct)),
            }
        )
    return pairs


def split_pairs(pairs: list[dict], seed: int = RANDOM_SEED) -> tuple[list[dict], list[dict]]:
    """Deterministically split pairs 80/20."""
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    split_at = max(1, int(0.8 * len(shuffled)))
    return shuffled[:split_at], shuffled[split_at:]


def evaluate_sc_auroc(
    verifier: SCEnergyVerifier, eval_pairs: list[dict]
) -> tuple[float, list[int], list[float]]:
    """Compute AUROC where label 1 means incoherent."""
    labels: list[int] = []
    scores: list[float] = []
    for pair in eval_pairs:
        coherent_response, coherent_context = pair["coherent"]
        incoherent_response, incoherent_context = pair["incoherent"]
        labels.extend([0, 1])
        scores.append(verifier.energy(coherent_response, coherent_context))
        scores.append(verifier.energy(incoherent_response, incoherent_context))
    return mann_whitney_auroc(labels, scores), labels, scores


def load_correlation_corpus(path: Path, seed: int = RANDOM_SEED) -> list[dict]:
    """Load the 500-row FoVer holdout protocol used by Exp 1108/1121."""
    rows = load_rows(path)
    correct = [row for row in rows if _is_correct(row)]
    wrong = [row for row in rows if not _is_correct(row)]
    rng = random.Random(seed)
    rng.shuffle(correct)
    rng.shuffle(wrong)
    sample = wrong[:114] + correct[:386]
    rng.shuffle(sample)
    return sample


def score_existing_verifiers(texts: list[str]) -> dict[str, list[float]]:
    """Score texts with the existing k=5 verifier mechanisms."""
    sem_energy = SemEnergyProbe()
    ast = ASTStructureVerifier()
    semantic_consistency = SemanticConsistencyVerifier()
    z3_math = Z3MathVerifier()
    return {
        "sos_kan": [_sos_kan_text_feature_proxy(text) for text in texts],
        "sem_energy": [float(sem_energy.score_response_proxy(text)) for text in texts],
        "ast": [float(ast.score(text)) for text in texts],
        "sem_consistency": [float(semantic_consistency.score(text)) for text in texts],
        "z3_math": [float(z3_math.score(text)) for text in texts],
    }


def score_sc_for_correlation(
    verifier: SCEnergyVerifier, rows: list[dict]
) -> tuple[list[str], list[float]]:
    """Score FoVer rows with SC-Energy for pairwise r measurement."""
    texts = [_step_text(row) for row in rows]
    scores = [verifier.energy(text, str(row.get("question", ""))) for row, text in zip(rows, texts)]
    return texts, scores


def mann_whitney_auroc(labels: list[int], scores: list[float]) -> float:
    """Compute AUROC without sklearn."""
    positive = [score for label, score in zip(labels, scores) if label == 1]
    negative = [score for label, score in zip(labels, scores) if label == 0]
    if not positive or not negative:
        return 0.5
    wins = 0.0
    for pos in positive:
        for neg in negative:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return float(wins / (len(positive) * len(negative)))


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson r with zero-variance guard."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) == 0 or len(x) != len(y) or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def honest_verdict(sc_auroc: float, all_r_below: bool, training_failed: bool = False) -> str:
    """Map measured gate conditions to the required verdict enum."""
    if training_failed:
        return "sc_energy_training_failed"
    if sc_auroc <= SC_AUROC_THRESHOLD:
        return "sc_energy_below_auroc_threshold"
    if not all_r_below:
        return "sc_energy_viable_corr_too_high"
    return "sc_energy_viable_k6_ready"


def run() -> dict:
    """Train, evaluate, correlate, and write the Exp 1168 artifact."""
    started_at = datetime.now(tz=UTC).isoformat()
    t0 = time.time()
    training_failed = False

    corpus_path = latest_labeled_pairs_path(PROJECT_ROOT / "results")
    rows = load_rows(corpus_path)
    pairs = build_contrastive_pairs(rows)
    train_pairs, eval_pairs = split_pairs(pairs)

    verifier = SCEnergyVerifier(model_name="roberta-base", hidden_dim=128)
    try:
        verifier.train(
            [pair["coherent"] for pair in train_pairs],
            [pair["incoherent"] for pair in train_pairs],
            n_epochs=10,
        )
    except Exception:
        training_failed = True

    sc_auroc, _, _ = evaluate_sc_auroc(verifier, eval_pairs)

    correlation_rows = load_correlation_corpus(CORRELATION_CORPUS_PATH)
    texts, sc_corr_scores = score_sc_for_correlation(verifier, correlation_rows)
    existing_scores = score_existing_verifiers(texts)
    r_corr = {name: pearson_r(sc_corr_scores, scores) for name, scores in existing_scores.items()}

    sc_above = bool(sc_auroc > SC_AUROC_THRESHOLD)
    all_r_below = bool(all(abs(value) < R_CORR_THRESHOLD for value in r_corr.values()))
    k6_viable = bool(sc_above and all_r_below and not training_failed)
    finished_at = datetime.now(tz=UTC).isoformat()

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": "sc_energy_7th_verifier",
        "run_date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(time.time() - t0, 2),
        "status": "success" if not training_failed else "training_failed",
        "fover_labeled_pairs_path": str(corpus_path.relative_to(PROJECT_ROOT)),
        "fover_correlation_path": str(CORRELATION_CORPUS_PATH.relative_to(PROJECT_ROOT)),
        "n_contrastive_pairs": len(pairs),
        "n_train_pairs": len(train_pairs),
        "n_eval_pairs": len(eval_pairs),
        "encoder_backend": verifier.encoder_backend,
        "model_name": verifier.model_name,
        "hidden_dim": verifier.hidden_dim,
        "sc_energy_auroc": round(float(sc_auroc), 6),
        "sc_energy_7th_verifier_auroc": round(float(sc_auroc), 6),
        "sc_energy_auroc_above_threshold": sc_above,
        "r_corr_with_sos_kan": round(float(r_corr["sos_kan"]), 6),
        "r_corr_with_sem_energy": round(float(r_corr["sem_energy"]), 6),
        "r_corr_with_ast": round(float(r_corr["ast"]), 6),
        "r_corr_with_sem_consistency": round(float(r_corr["sem_consistency"]), 6),
        "r_corr_with_z3_math": round(float(r_corr["z3_math"]), 6),
        "all_r_below_0.5": all_r_below,
        "k6_viable": k6_viable,
        "verifier_module_path": VERIFIER_MODULE_PATH,
        "honest_verdict": honest_verdict(sc_auroc, all_r_below, training_failed),
        "sos_kan_scoring_backend": "text_feature_proxy_no_jax",
    }
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def _step_text(row: dict) -> str:
    return str(row.get("step_text") or row.get("step") or row.get("response") or "")


def _is_correct(row: dict) -> bool:
    if "label" in row:
        return row["label"] == "correct"
    if "step_correct" in row:
        return bool(row["step_correct"])
    if "is_correct" in row:
        return bool(row["is_correct"])
    return False


def _sos_kan_text_feature_proxy(text: str) -> float:
    """CPU-only SOS-KAN score proxy for correlation when JAX is unavailable."""
    words = text.split()
    n_words = max(len(words), 1)
    numeric_density = sum(1 for word in words if any(ch.isdigit() for ch in word)) / n_words
    vocabulary_richness = len(set(words)) / n_words
    length_feature = min(math.log(len(text) + 1) / 8.0, 1.0)
    energy = (
        0.45 * length_feature + 0.35 * (1.0 - numeric_density) + 0.20 * (1.0 - vocabulary_richness)
    )
    return float(max(0.0, min(1.0, energy)))


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True))
