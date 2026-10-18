"""Exp 4209 oracle-distinct V-STaR-style ARC verifier.

Spec refs: REQ-VERIFY-4209, SCENARIO-VERIFY-4209, SCENARIO-VERIFY-4209-LABELED.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.reporting import vstar_learned_selector_4176 as vstar_4176


RANDOM_SEED = 4209
POOL_REL = Path("results/arc3_trm_verifier_rerank.json")
OUTPUT_REL = Path("results/experiment_4209_oracle_distinct_arc_verifier_build.json")
VERIFIER_REL = Path("results/experiment_4209_oracle_distinct_arc_verifier_model.json")
SPEC_REFS = [
    "REQ-VERIFY-4209",
    "SCENARIO-VERIFY-4209",
    "SCENARIO-VERIFY-4209-LABELED",
]
FEATURE_NAMES = (
    "candidate_index",
    "vote_weight",
    "self_consistency_margin",
    "output_height",
    "output_width",
    "output_cells",
    "output_color_count",
    "output_nonzero_fraction",
    "output_mean",
    "output_entropy",
    "program_length",
    "program_digit_fraction",
    "region_confidence_mean",
    "region_confidence_min",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A trained off-fold verifier OR an honest 'no learnable "
        "oracle-distinct signal (AUROC~0.5)' is COMPLETE -- both feed A3."
    ),
    "selector_trained": (
        "BARE bool: A3's gate compares this raw value (gated-fields-must-be-bare); "
        "true iff a learned ARC verifier artifact was persisted out-of-fold."
    ),
    "oracle_distinct_auroc": (
        "BARE float: off-fold detection AUROC of the LEARNED verifier vs is_correct "
        "-- the oracle-distinct discrimination the GAP-3 content energies lacked; "
        ">0.5 CI95-excl is the precondition for an A3 beats-vote win."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the learned verifier scores WITHOUT executing the demos "
        "(Circularity Discipline); this is what makes an A3 win headline/gate-eligible, "
        "unlike the circular execution verifier."
    ),
    "learned_verifier_path": (
        "The persisted artifact A3 loads to rerank held-out ARC candidates; the build deliverable."
    ),
    "model_specs": (
        "The V-STaR probe architecture + feature set + any base used; required methodology."
    ),
    "random_seed": (
        "Determinism precondition; the fold split + probe init seeded so the AUROC is reproducible."
    ),
    "reproducibility_checksum": (
        "Hash of the ARC pool + fold split + features; catches silent pool/feature drift "
        "before A3 measures."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "selector_trained",
    "oracle_distinct_auroc",
    "oracle_distinct_auroc_ci95",
    "verifier_is_oracle",
    "learned_verifier_path",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
)


class BlockedRun(RuntimeError):
    """Expected precondition failure that still writes a terminal artifact."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ArcCandidateRow:
    task_id: str
    candidate_id: str
    candidate_index: int
    correct: bool
    features: dict[str, float]


@dataclass(frozen=True)
class ArcCandidateCorpus:
    source_path: Path
    source_sha256: str
    rows: list[ArcCandidateRow]


@dataclass(frozen=True)
class OOFRow:
    task_id: str
    candidate_id: str
    correct: bool
    score: float
    fold: int
    train_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class OOFReport:
    oracle_distinct_auroc: float
    oracle_distinct_auroc_ci95: tuple[float, float]
    fold_task_ids: list[list[str]]
    oof_rows: list[OOFRow]
    final_verifier: dict[str, Any]


def _model_specs(status: str) -> dict[str, Any]:
    return {
        "architecture": "logistic_regression_binary_selector",
        "base": "Exp4176 V-STaR logistic selector",
        "feature_set": list(FEATURE_NAMES),
        "training_recipe": "accepted_and_rejected_arc_candidates_task_held_out",
        "status": status,
    }


def _pool_path(repo_root_or_pool: Path | str) -> Path:
    path = Path(repo_root_or_pool)
    return path if path.is_file() else path / POOL_REL


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BlockedRun("blocked_arc_pool_missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BlockedRun("blocked_malformed_arc_pool")
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _flatten_grid(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    flat: list[float] = []
    for row in value:
        if isinstance(row, list):
            flat.extend(_as_float(cell) for cell in row)
        else:
            flat.append(_as_float(row))
    return flat


def _grid_shape(value: Any) -> tuple[int, int]:
    if not isinstance(value, list):
        return (0, 0)
    height = len(value)
    width = max((len(row) if isinstance(row, list) else 1 for row in value), default=0)
    return (height, width)


def _grid_stats(value: Any) -> dict[str, float]:
    height, width = _grid_shape(value)
    flat = _flatten_grid(value)
    cells = len(flat)
    if cells == 0:
        return {
            "output_height": 0.0,
            "output_width": 0.0,
            "output_cells": 0.0,
            "output_color_count": 0.0,
            "output_nonzero_fraction": 0.0,
            "output_mean": 0.0,
            "output_entropy": 0.0,
        }
    counts: dict[float, int] = {}
    for cell in flat:
        counts[cell] = counts.get(cell, 0) + 1
    entropy = -sum((count / cells) * math.log2(count / cells) for count in counts.values())
    return {
        "output_height": float(height),
        "output_width": float(width),
        "output_cells": float(cells),
        "output_color_count": float(len(counts)),
        "output_nonzero_fraction": sum(1 for cell in flat if cell != 0.0) / float(cells),
        "output_mean": sum(flat) / float(cells),
        "output_entropy": entropy,
    }


def _candidate_output(candidate: dict[str, Any]) -> Any:
    for key in ("output", "candidate_output", "grid", "answer"):
        if key in candidate:
            return candidate[key]
    return None


def _candidate_program(candidate: dict[str, Any]) -> str:
    for key in ("program", "candidate_program", "trace", "candidate_trace"):
        value = candidate.get(key)
        if value is not None:
            return value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return ""


def _confidence_values(candidate: dict[str, Any]) -> list[float]:
    for key in ("region_confidence", "per_region_confidence", "confidences"):
        value = candidate.get(key)
        if isinstance(value, list):
            return [_as_float(item) for item in value]
    return []


def _has_candidate_content(candidate: dict[str, Any]) -> bool:
    return _candidate_output(candidate) is not None or bool(_candidate_program(candidate))


def _feature_map(candidate: dict[str, Any], candidate_index: int) -> dict[str, float]:
    output = _candidate_output(candidate)
    program = _candidate_program(candidate)
    confidences = _confidence_values(candidate)
    digit_count = sum(1 for char in program if char.isdigit())
    program_length = len(program)
    features = {
        "candidate_index": float(candidate_index),
        "vote_weight": _as_float(candidate.get("vote_weight", candidate.get("vote_count"))),
        "self_consistency_margin": _as_float(candidate.get("self_consistency_margin")),
        "program_length": float(program_length),
        "program_digit_fraction": digit_count / float(program_length) if program_length else 0.0,
        "region_confidence_mean": (
            sum(confidences) / float(len(confidences)) if confidences else 0.0
        ),
        "region_confidence_min": min(confidences) if confidences else 0.0,
    }
    features.update(_grid_stats(output))
    return {name: float(features[name]) for name in FEATURE_NAMES}


def load_arc_candidate_pool(repo_root_or_pool: Path | str = Path(".")) -> ArcCandidateCorpus:
    """SCENARIO-VERIFY-4209: load ARC candidate rows or block on task summaries."""

    source_path = _pool_path(repo_root_or_pool)
    payload = _read_json_object(source_path)
    per_task = payload.get("per_task")
    if not isinstance(per_task, list):
        raise BlockedRun("blocked_arc_pool_no_candidate_labels")
    rows: list[ArcCandidateRow] = []
    for task_index, task in enumerate(per_task):
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task") or task.get("task_id") or f"task-{task_index}")
        candidates = task.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            correct = candidate.get("is_correct")
            if not isinstance(correct, bool) or not _has_candidate_content(candidate):
                continue
            index = int(_as_float(candidate.get("candidate_index", candidate_index)))
            candidate_id = str(candidate.get("candidate_id") or f"{task_id}::{index}")
            rows.append(
                ArcCandidateRow(
                    task_id=task_id,
                    candidate_id=candidate_id,
                    candidate_index=index,
                    correct=correct,
                    features=_feature_map(candidate, index),
                )
            )
    if not rows:
        raise BlockedRun("blocked_arc_pool_no_candidate_labels")
    return ArcCandidateCorpus(source_path=source_path.resolve(), source_sha256=_sha256_file(source_path), rows=rows)


def accepted_rejected_counts(rows: list[ArcCandidateRow]) -> dict[str, int]:
    accepted = sum(row.correct for row in rows)
    rejected = len(rows) - accepted
    return {"accepted": int(accepted), "rejected": int(rejected), "total": len(rows)}


def _feature_vector(features: dict[str, float]) -> list[float]:
    return [float(features[name]) for name in FEATURE_NAMES]


def _split_task_folds(rows: list[ArcCandidateRow], random_seed: int, n_folds: int) -> list[set[str]]:
    task_ids = sorted({row.task_id for row in rows})
    if len(task_ids) < 2:
        raise BlockedRun("blocked_arc_pool_needs_two_tasks")
    fold_count = max(2, min(int(n_folds), len(task_ids)))
    shuffled = task_ids[:]
    random.Random(random_seed).shuffle(shuffled)
    return [set(shuffled[index::fold_count]) for index in range(fold_count)]


def _train_verifier(rows: list[ArcCandidateRow], random_seed: int) -> dict[str, Any]:
    model = vstar_4176.LogisticRegression(
        random_state=random_seed,
        solver="liblinear",
        max_iter=1000,
    )
    model.fit([_feature_vector(row.features) for row in rows], [int(row.correct) for row in rows])
    return {
        "model_type": "logistic_regression",
        "feature_names": list(FEATURE_NAMES),
        "intercept": float(model.intercept_[0]),
        "coefficients": [float(value) for value in model.coef_[0]],
    }


def score_with_verifier(verifier: dict[str, Any], features: dict[str, float]) -> float:
    values = [float(features[name]) for name in verifier["feature_names"]]
    logit = float(verifier["intercept"]) + sum(
        float(weight) * value for weight, value in zip(verifier["coefficients"], values, strict=True)
    )
    return 1.0 / (1.0 + math.exp(-logit))


def _bootstrap_auroc_ci95(labels: list[bool], scores: list[float], random_seed: int) -> tuple[float, float]:
    rng = random.Random(random_seed)
    n = len(labels)
    samples: list[float] = []
    for _ in range(200):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_labels = [labels[index] for index in indices]
        if all(sample_labels) or not any(sample_labels):
            continue
        sample_scores = [scores[index] for index in indices]
        samples.append(vstar_4176._auroc(sample_labels, sample_scores))
    if not samples:
        point = vstar_4176._auroc(labels, scores)
        return (point, point)
    samples.sort()
    low_index = int(0.025 * (len(samples) - 1))
    high_index = int(0.975 * (len(samples) - 1))
    return (samples[low_index], samples[high_index])


def train_oof_verifier(
    rows: list[ArcCandidateRow], *, random_seed: int = RANDOM_SEED, n_folds: int = 5
) -> OOFReport:
    """SCENARIO-VERIFY-4209-LABELED: train and score candidates by task-held-out folds."""

    counts = accepted_rejected_counts(rows)
    if counts["accepted"] == 0 or counts["rejected"] == 0:
        raise BlockedRun("blocked_arc_pool_lacks_accepted_rejected")
    folds = _split_task_folds(rows, random_seed, n_folds)
    oof_scores_by_id: dict[str, float] = {}
    oof_rows: list[OOFRow] = []
    for fold, heldout_task_ids in enumerate(folds):
        train_rows = [row for row in rows if row.task_id not in heldout_task_ids]
        test_rows = [row for row in rows if row.task_id in heldout_task_ids]
        train_counts = accepted_rejected_counts(train_rows)
        if train_counts["accepted"] == 0 or train_counts["rejected"] == 0:
            raise BlockedRun("blocked_arc_fold_lacks_label_contrast")
        verifier = _train_verifier(train_rows, random_seed + fold)
        train_task_ids = tuple(sorted({row.task_id for row in train_rows}))
        for row in test_rows:
            score = score_with_verifier(verifier, row.features)
            oof_scores_by_id[row.candidate_id] = score
            oof_rows.append(
                OOFRow(
                    task_id=row.task_id,
                    candidate_id=row.candidate_id,
                    correct=row.correct,
                    score=score,
                    fold=fold,
                    train_task_ids=train_task_ids,
                )
            )
    oof_scores = [oof_scores_by_id[row.candidate_id] for row in rows]
    labels = [row.correct for row in rows]
    auroc = vstar_4176._auroc(labels, oof_scores)
    return OOFReport(
        oracle_distinct_auroc=auroc,
        oracle_distinct_auroc_ci95=_bootstrap_auroc_ci95(labels, oof_scores, random_seed),
        fold_task_ids=[sorted(fold) for fold in folds],
        oof_rows=oof_rows,
        final_verifier=_train_verifier(rows, random_seed),
    )


def reproducibility_checksum(
    corpus: ArcCandidateCorpus | None,
    *,
    fold_task_ids: list[list[str]] | None = None,
    rows: list[ArcCandidateRow] | None = None,
) -> str:
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "fold_task_ids": fold_task_ids or [],
        "pool_sha256": corpus.source_sha256 if corpus is not None else "",
        "rows": [
            {
                "candidate_id": row.candidate_id,
                "correct": row.correct,
                "features": row.features,
                "task_id": row.task_id,
            }
            for row in (rows or [])
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def persist_verifier(
    path: Path,
    verifier: dict[str, Any],
    *,
    checksum: str,
    counts: dict[str, int],
    report: OOFReport,
    random_seed: int,
) -> None:
    payload = {
        **verifier,
        "accepted_rejected_n": counts,
        "fold_task_ids": report.fold_task_ids,
        "model_specs": _model_specs("trained"),
        "oof_rows": [
            {
                "candidate_id": row.candidate_id,
                "correct": row.correct,
                "fold": row.fold,
                "score": _round_metric(row.score),
                "task_id": row.task_id,
                "train_task_ids": list(row.train_task_ids),
            }
            for row in report.oof_rows
        ],
        "random_seed": random_seed,
        "reproducibility_checksum": checksum,
        "spec_refs": SPEC_REFS,
        "verifier_is_oracle": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_verifier(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("verifier artifact must be a JSON object")
    return payload


def _blocked_artifact(
    reason: str,
    *,
    candidate_pool_source: str,
    random_seed: int,
    checksum: str,
) -> dict[str, Any]:
    return {
        "honest_verdict": reason,
        "selector_trained": False,
        "oracle_distinct_auroc": 0.0,
        "oracle_distinct_auroc_ci95": [0.0, 0.0],
        "verifier_is_oracle": False,
        "learned_verifier_path": "",
        "model_specs": _model_specs("blocked_no_candidate_labels"),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "accepted_rejected_n": {"accepted": 0, "rejected": 0, "total": 0},
        "candidate_pool_source": candidate_pool_source,
    }


def _complete_artifact(
    corpus: ArcCandidateCorpus,
    report: OOFReport,
    *,
    checksum: str,
    counts: dict[str, int],
    verifier_path: Path,
    random_seed: int,
) -> dict[str, Any]:
    auroc = _round_metric(report.oracle_distinct_auroc)
    ci95 = [_round_metric(value) for value in report.oracle_distinct_auroc_ci95]
    verdict = (
        f"complete: oracle_distinct_arc_verifier_trained_auroc_{auroc:.4f}"
        if not (0.45 <= auroc <= 0.55)
        else f"complete_oracle_distinct_arc_verifier_no_learnable_signal_auroc{auroc:.4f}"
    )
    return {
        "honest_verdict": verdict,
        "selector_trained": True,
        "oracle_distinct_auroc": auroc,
        "oracle_distinct_auroc_ci95": ci95,
        "verifier_is_oracle": False,
        "learned_verifier_path": str(verifier_path),
        "model_specs": _model_specs("trained"),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "accepted_rejected_n": counts,
        "candidate_pool_source": str(corpus.source_path),
        "feature_names": list(FEATURE_NAMES),
        "oof_folds": len(report.fold_task_ids),
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("complete_") or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if not isinstance(artifact["selector_trained"], bool):
        raise ValueError("selector_trained must be a bare bool")
    if not isinstance(artifact["oracle_distinct_auroc"], float):
        raise ValueError("oracle_distinct_auroc must be a bare float")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4209")
    if artifact["selector_trained"] and not Path(artifact["learned_verifier_path"]).exists():
        raise ValueError("trained artifacts require a persisted verifier")


def run(
    repo_root: Path | str = Path("."), *, random_seed: int = RANDOM_SEED, n_folds: int = 5
) -> dict[str, Any]:
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    candidate_pool_source = str(_pool_path(root))
    try:
        corpus = load_arc_candidate_pool(root)
        counts = accepted_rejected_counts(corpus.rows)
        report = train_oof_verifier(corpus.rows, random_seed=random_seed, n_folds=n_folds)
        checksum = reproducibility_checksum(
            corpus,
            fold_task_ids=report.fold_task_ids,
            rows=corpus.rows,
        )
        verifier_path = (root / VERIFIER_REL).resolve()
        persist_verifier(
            verifier_path,
            report.final_verifier,
            checksum=checksum,
            counts=counts,
            report=report,
            random_seed=random_seed,
        )
        artifact = _complete_artifact(
            corpus,
            report,
            checksum=checksum,
            counts=counts,
            verifier_path=verifier_path,
            random_seed=random_seed,
        )
    except BlockedRun as blocked:
        pool_path = _pool_path(root)
        source_hash = _sha256_file(pool_path) if pool_path.exists() else ""
        checksum = hashlib.sha256(
            json.dumps(
                {
                    "feature_names": list(FEATURE_NAMES),
                    "fold_task_ids": [],
                    "pool_sha256": source_hash,
                    "rows": [],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        artifact = _blocked_artifact(
            blocked.reason,
            candidate_pool_source=candidate_pool_source,
            random_seed=random_seed,
            checksum=checksum,
        )
    validate_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
