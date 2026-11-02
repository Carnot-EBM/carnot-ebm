"""Pure helpers for Exp 1176 k=6 AND-composition validation.

Spec: REQ-VERIFY-1176, SCENARIO-VERIFY-1176
"""

from __future__ import annotations

import json
import random
import sys
import time
import types
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENT_ID = 1176
SCHEMA = "k6_and_compose_validation"
K5_AUROC_BASELINE = 0.9402
RANDOM_SEED = 1176
EXP1168_RANDOM_SEED = 1168
FOVER_EVAL_N = 200

K5_VERIFIER_NAMES = [
    "SOSKANEnergyV3",
    "SemEnergyProbe",
    "ASTStructureVerifier",
    "SemanticConsistencyVerifier",
    "Z3MathVerifier",
]

REQUIRED_ARTIFACT_FIELDS = {
    "k5_auroc_baseline",
    "k6_auroc",
    "k6_above_k5",
    "k6_and_compose_auroc_measured",
    "sc_energy_r_corr_on_eval",
    "sc_energy_marginal_gain",
    "honest_verdict",
}

ALLOWED_VERDICTS = {
    "k6_improves_over_k5",
    "k6_no_improvement",
    "k6_degrades_due_to_correlation",
}


@dataclass(frozen=True)
class ValidationScores:
    """Raw per-row energies used for the Exp1176 metrics."""

    labels: list[int]
    sc_scores: list[float]
    existing_scores: dict[str, list[float]]
    k5_scores: list[float]
    k6_scores: list[float]


@dataclass(frozen=True)
class ValidationMetrics:
    """Schema-ready Exp1176 metric summary."""

    k5_auroc_baseline: float
    k5_auroc_on_eval: float
    k6_auroc: float
    k6_above_k5: bool
    k6_and_compose_auroc_measured: bool
    sc_energy_r_corr_on_eval: dict[str, float]
    sc_energy_marginal_gain: float
    honest_verdict: str
    largest_sc_energy_overlap: str
    max_abs_sc_energy_r_corr: float
    n_eval_examples: int
    n_correct: int
    n_incorrect: int


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load JSON or JSONL FoVer rows from ``path``."""
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [row for row in rows if isinstance(row, dict)]

    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("pairs", "items", "examples", "data", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError(f"unsupported row payload in {path}")


def is_incorrect(row: dict[str, Any]) -> bool:
    """Return whether a FoVer row is labeled incorrect."""
    if "is_correct" in row:
        return not bool(row["is_correct"])
    if "step_correct" in row:
        return not bool(row["step_correct"])
    label = row.get("label")
    if label is None:
        label = row.get("sc_energy_label")
    if label is None:
        label = row.get("coherence_label")
    if isinstance(label, str):
        return label.lower() in {"incorrect", "incoherent", "wrong", "false", "0"}
    if isinstance(label, bool):
        return not label
    return False


def row_text(row: dict[str, Any]) -> str:
    """Return the response text field used by the verifiers."""
    return str(row.get("response") or row.get("step_text") or row.get("step") or "")


def row_context(row: dict[str, Any]) -> str:
    """Return the prompt/question context field when present."""
    return str(row.get("question") or row.get("prompt") or row.get("context") or "")


def select_heldout_eval_rows(
    rows: list[dict[str, Any]],
    n_examples: int = FOVER_EVAL_N,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Select exactly ``n_examples`` held-out rows while preserving rare errors."""
    if len(rows) < n_examples:
        raise ValueError(f"held-out corpus must contain at least {n_examples} rows")

    incorrect = [row for row in rows if is_incorrect(row)]
    correct = [row for row in rows if not is_incorrect(row)]
    if not incorrect or not correct:
        raise ValueError("held-out AUROC evaluation requires both correct and incorrect rows")

    rng = random.Random(seed)
    rng.shuffle(incorrect)
    rng.shuffle(correct)

    target_incorrect = min(len(incorrect), max(1, n_examples // 2))
    target_correct = n_examples - target_incorrect
    if target_correct > len(correct):
        target_correct = len(correct)
        target_incorrect = n_examples - target_correct

    selected = incorrect[:target_incorrect] + correct[:target_correct]
    rng.shuffle(selected)
    return selected


def build_contrastive_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild Exp1168 coherent/incoherent SC-Energy pairs from labeled rows."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        qid = str(row.get("question_id") or row.get("question") or row.get("id") or "")
        if qid:
            grouped.setdefault(qid, []).append(row)

    pairs: list[dict[str, Any]] = []
    for qid, items in sorted(grouped.items()):
        correct = [row_text(row) for row in items if not is_incorrect(row)]
        incorrect = [row_text(row) for row in items if is_incorrect(row)]
        correct = [item for item in correct if item]
        incorrect = [item for item in incorrect if item]
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


def split_pairs(
    pairs: list[dict[str, Any]],
    seed: int = EXP1168_RANDOM_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split contrastive pairs with the Exp1168 protocol."""
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    split_at = max(1, int(0.8 * len(shuffled)))
    return shuffled[:split_at], shuffled[split_at:]


def load_sc_energy_from_exp1168(
    exp1168_path: Path,
    *,
    project_root: Path,
    n_epochs: int = 10,
) -> tuple[Any, dict[str, Any]]:
    """Instantiate SC-Energy from Exp1168 metadata and deterministic training rows."""
    payload = json.loads(exp1168_path.read_text())
    if (
        payload.get("sc_energy_auroc_above_threshold") is not True
        or payload.get("k6_viable") is not True
    ):
        raise RuntimeError("Exp 1168 gate is not satisfied; k=6 validation is not authorized")

    from carnot.verify.sc_energy_verifier import SCEnergyVerifier

    model_name = str(payload.get("model_name") or "roberta-base")
    hidden_dim = int(payload.get("hidden_dim") or 128)
    train_path = _resolve_project_path(
        project_root, str(payload.get("fover_labeled_pairs_path", ""))
    )
    rows = load_rows(train_path)
    pairs = build_contrastive_pairs(rows)
    train_pairs, eval_pairs = split_pairs(pairs)

    verifier = SCEnergyVerifier(model_name=model_name, hidden_dim=hidden_dim)
    verifier.train(
        [pair["coherent"] for pair in train_pairs],
        [pair["incoherent"] for pair in train_pairs],
        n_epochs=n_epochs,
    )

    source = {
        "exp1168_artifact": str(exp1168_path),
        "sc_energy_loading_mode": "deterministic_retrain_from_exp1168_artifact",
        "checkpoint_loaded": False,
        "checkpoint_note": "Exp1168 artifact records model metadata but no serialized SC-Energy weights.",
        "model_name": model_name,
        "hidden_dim": hidden_dim,
        "fover_labeled_pairs_path": str(train_path),
        "n_contrastive_pairs": len(pairs),
        "n_train_pairs": len(train_pairs),
        "n_exp1168_eval_pairs": len(eval_pairs),
    }
    return verifier, source


def build_fixed_k5_verifiers(
    corpus_path: Path,
    *,
    n_correct: int = 386,
    n_wrong: int = 114,
    seed: int = 1128,
    n_epochs: int = 100,
) -> list[Any]:
    """Build the repaired Exp1128 k=5 verifier list, including trained SOS-KAN."""
    install_lightweight_carnot_import_stubs(Path(__file__).resolve().parents[3])

    from carnot.verify.and_composition_verifier import (
        ASTStructureAdapter,
        SOSKANEnergyV3Adapter,
        SemEnergyProbeAdapter,
        SemanticConsistencyAdapter,
        Z3MathAdapter,
    )

    rows = load_rows(corpus_path)
    correct = [row for row in rows if not is_incorrect(row)]
    wrong = [row for row in rows if is_incorrect(row)]
    rng = random.Random(seed)
    rng.shuffle(correct)
    rng.shuffle(wrong)
    training_rows = wrong[:n_wrong] + correct[:n_correct]
    rng.shuffle(training_rows)

    soskan = SOSKANEnergyV3Adapter()
    soskan.fit_from_corpus(training_rows, n_epochs=n_epochs, lr=3e-3)
    return [
        soskan,
        SemEnergyProbeAdapter(),
        ASTStructureAdapter(),
        SemanticConsistencyAdapter(),
        Z3MathAdapter(),
    ]


def install_lightweight_carnot_import_stubs(project_root: Path) -> None:
    """Install namespace-package stubs for verifier/model submodule imports."""
    python_dir = project_root / "python"
    if str(python_dir) not in sys.path:  # pragma: no cover - process bootstrap branch.
        sys.path.insert(0, str(python_dir))

    for package in ("carnot.verify", "carnot.models", "carnot.pipeline"):
        if package in sys.modules:
            continue
        module = types.ModuleType(package)  # pragma: no cover - process bootstrap branch.
        module.__path__ = [str(python_dir / package.replace(".", "/"))]  # type: ignore[attr-defined]  # pragma: no cover
        module.__package__ = package  # pragma: no cover - process bootstrap branch.
        sys.modules[package] = module  # pragma: no cover - process bootstrap branch.

        parent_name, attr = package.rsplit(".", 1)  # pragma: no cover - process bootstrap branch.
        parent = sys.modules.get(parent_name)  # pragma: no cover - process bootstrap branch.
        if parent is not None:  # pragma: no cover - process bootstrap branch.
            setattr(parent, attr, module)


def score_eval_rows(
    rows: list[dict[str, Any]],
    sc_verifier: Any,
    k5_verifiers: list[Any],
) -> ValidationScores:
    """Score FoVer rows with k=5 and SC-Energy and compose via max energy."""
    labels: list[int] = []
    sc_scores: list[float] = []
    existing_scores: dict[str, list[float]] = {verifier.name: [] for verifier in k5_verifiers}
    k5_scores: list[float] = []
    k6_scores: list[float] = []

    for row in rows:
        response = row_text(row)
        context = row_context(row)
        combined = f"{context}\n{response}" if context.strip() else response
        labels.append(1 if is_incorrect(row) else 0)

        row_existing: list[float] = []
        for verifier in k5_verifiers:
            try:
                energy = float(verifier.score(combined))
            except Exception:  # pragma: no cover - defensive parity with AndCompositionVerifier.
                energy = 0.0
            existing_scores[verifier.name].append(energy)
            row_existing.append(energy)

        sc_energy = float(sc_verifier.score(response, context))
        k5_energy = max(row_existing) if row_existing else 0.0
        labels[-1] = int(labels[-1])
        sc_scores.append(sc_energy)
        k5_scores.append(k5_energy)
        k6_scores.append(max(k5_energy, sc_energy))

    return ValidationScores(
        labels=labels,
        sc_scores=sc_scores,
        existing_scores=existing_scores,
        k5_scores=k5_scores,
        k6_scores=k6_scores,
    )


def tie_aware_auroc(labels: list[int], scores: list[float]) -> float:
    """Return AUROC with 0.5 credit for tied positive/negative scores."""
    positive = [score for label, score in zip(labels, scores, strict=False) if label == 1]
    negative = [score for label, score in zip(labels, scores, strict=False) if label == 0]
    if not positive or not negative:
        return 0.5
    wins = 0.0
    for pos_score in positive:
        for neg_score in negative:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return float(wins / (len(positive) * len(negative)))


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Return Pearson r, guarded for shape mismatch and zero variance."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) == 0 or len(x) != len(y) or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def honest_verdict(k6_auroc: float, r_corr: dict[str, float]) -> str:
    """Map the measured Exp1176 outcome to the required verdict enum."""
    if k6_auroc > K5_AUROC_BASELINE:
        return "k6_improves_over_k5"
    if any(abs(value) >= 0.5 for value in r_corr.values()):
        return "k6_degrades_due_to_correlation"
    return "k6_no_improvement"


def compute_validation_metrics(scores: ValidationScores) -> ValidationMetrics:
    """Compute AUROC, marginal gain, and SC-Energy pairwise correlations."""
    r_corr = {
        name: round(pearson_r(scores.sc_scores, verifier_scores), 6)
        for name, verifier_scores in scores.existing_scores.items()
    }
    largest_overlap = max(r_corr, key=lambda name: abs(r_corr[name]))
    k6_auroc = round(tie_aware_auroc(scores.labels, scores.k6_scores), 6)
    k5_eval_auroc = round(tie_aware_auroc(scores.labels, scores.k5_scores), 6)
    verdict = honest_verdict(k6_auroc, r_corr)
    n_incorrect = int(sum(scores.labels))
    n_eval = len(scores.labels)
    return ValidationMetrics(
        k5_auroc_baseline=K5_AUROC_BASELINE,
        k5_auroc_on_eval=k5_eval_auroc,
        k6_auroc=k6_auroc,
        k6_above_k5=bool(k6_auroc > K5_AUROC_BASELINE),
        k6_and_compose_auroc_measured=True,
        sc_energy_r_corr_on_eval=r_corr,
        sc_energy_marginal_gain=round(k6_auroc - K5_AUROC_BASELINE, 6),
        honest_verdict=verdict,
        largest_sc_energy_overlap=largest_overlap,
        max_abs_sc_energy_r_corr=round(abs(r_corr[largest_overlap]), 6),
        n_eval_examples=n_eval,
        n_correct=n_eval - n_incorrect,
        n_incorrect=n_incorrect,
    )


def build_artifact(
    metrics: ValidationMetrics,
    *,
    sc_source: dict[str, Any],
    eval_corpus_path: Path,
    started_at: str,
    duration_s: float,
) -> dict[str, Any]:
    """Build the schema-complete Exp1176 result artifact."""
    now = datetime.now(tz=UTC)
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": now.date().isoformat(),
        "started_at": started_at,
        "finished_at": now.isoformat(),
        "duration_s": round(float(duration_s), 3),
        "status": "success",
        "title": "k=6 AND-Composition Validation with SC-Energy",
        "spec": ["REQ-VERIFY-1176", "SCENARIO-VERIFY-1176"],
        "k5_verifiers": K5_VERIFIER_NAMES,
        "k6_added_verifier": "SCEnergyVerifier",
        "and_composition_rule": "final_energy=max(E_1,...,E_6)",
        "eval_corpus_path": str(eval_corpus_path),
        "n_eval_examples": metrics.n_eval_examples,
        "n_correct": metrics.n_correct,
        "n_incorrect": metrics.n_incorrect,
        "k5_auroc_baseline": metrics.k5_auroc_baseline,
        "k5_auroc_on_eval": metrics.k5_auroc_on_eval,
        "k6_auroc": metrics.k6_auroc,
        "k6_above_k5": metrics.k6_above_k5,
        "k6_and_compose_auroc_measured": metrics.k6_and_compose_auroc_measured,
        "sc_energy_r_corr_on_eval": metrics.sc_energy_r_corr_on_eval,
        "sc_energy_marginal_gain": metrics.sc_energy_marginal_gain,
        "largest_sc_energy_overlap": metrics.largest_sc_energy_overlap,
        "max_abs_sc_energy_r_corr": metrics.max_abs_sc_energy_r_corr,
        "honest_verdict": metrics.honest_verdict,
        "decision_note": _decision_note(metrics),
        "follow_up_task": _follow_up_task(metrics),
        "pipeline_modified": False,
        "sc_energy_source": sc_source,
    }
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:  # pragma: no cover - schema constant regression guard.
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact["honest_verdict"] not in ALLOWED_VERDICTS:  # pragma: no cover - enum guard.
        raise ValueError(f"unexpected honest_verdict: {artifact['honest_verdict']}")
    return artifact


def write_artifact(artifact: dict[str, Any], output_path: Path) -> None:
    """Write the Exp1176 artifact as stable, readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def run_experiment(
    *,
    project_root: Path,
    exp1168_path: Path | None = None,
    eval_corpus_path: Path | None = None,
    soskan_training_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run Exp1176 end-to-end and write the validation artifact."""
    root = Path(project_root)
    exp1168 = exp1168_path or root / "results" / "experiment_1168_sc_energy_7th_verifier.json"
    eval_path = eval_corpus_path or root / "data" / "fover_test_v4.json"
    soskan_path = soskan_training_path or root / "data" / "fover_corpus_v4.json"
    out_path = output_path or root / "results" / "experiment_1176_k6_and_compose_validation.json"

    started_at = datetime.now(tz=UTC).isoformat()
    t0 = time.perf_counter()
    sc_verifier, sc_source = load_sc_energy_from_exp1168(exp1168, project_root=root)
    eval_rows = select_heldout_eval_rows(load_rows(eval_path), n_examples=FOVER_EVAL_N)
    k5_verifiers = build_fixed_k5_verifiers(soskan_path)
    scores = score_eval_rows(eval_rows, sc_verifier, k5_verifiers)
    metrics = compute_validation_metrics(scores)
    artifact = build_artifact(
        metrics,
        sc_source=sc_source,
        eval_corpus_path=eval_path,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
    )
    write_artifact(artifact, out_path)
    return artifact


def _resolve_project_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def _decision_note(metrics: ValidationMetrics) -> str:
    if metrics.honest_verdict == "k6_improves_over_k5":
        return (
            "k=6 improves over the Exp1128 k=5 baseline; create a follow-up task "
            "to wire SC-Energy as the default sixth verifier."
        )
    if metrics.honest_verdict == "k6_degrades_due_to_correlation":
        return (
            "k=6 does not beat k=5 and the largest SC-Energy overlap is "
            f"{metrics.largest_sc_energy_overlap} at |r|={metrics.max_abs_sc_energy_r_corr}."
        )
    return (
        "k=6 does not beat k=5 even though SC-Energy correlations remain below the "
        "0.5 viability threshold on this evaluation set."
    )


def _follow_up_task(metrics: ValidationMetrics) -> str:
    if metrics.honest_verdict == "k6_improves_over_k5":
        return "Open a follow-up implementation task to add SCEnergyVerifier to default k=6 wiring."
    return ""
