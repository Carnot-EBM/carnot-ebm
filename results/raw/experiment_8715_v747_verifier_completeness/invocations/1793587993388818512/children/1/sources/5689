"""Exp 4176 V-STaR-style learned selector over cached accepted/rejected traces.

Spec refs: REQ-VERIFY-4176, SCENARIO-VERIFY-4176.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression


RANDOM_SEED = 4176
HEADROOM_REL = Path("results/experiment_4175_headroom_gate_executable_census.json")
OUTPUT_REL = Path("results/experiment_4176_vstar_learned_selector.json")
SELECTOR_REL = Path("results/experiment_4176_vstar_selector_model.json")
INFERENCE_SUBSTRATE = "cached_artifact_oof_vstar_selector"
FEATURE_NAMES = ("role_repair", "vote_weight", "candidate_index", "extracted_constraints")

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A trained selector (even one that ties vote) is a COMPLETE verdict."
    ),
    "selector_auroc_oof": (
        "Out-of-fold gold-vs-non-gold AUROC; the learned selector's discrimination, "
        "measured without oracle leakage."
    ),
    "selector_pass1_vs_vote": (
        "pass@1 as a ranker minus SC-vote pass@1 (OOF) -- the learned-selector's "
        "standalone lift, input to A3."
    ),
    "accepted_rejected_n": (
        "Counts of accepted and rejected traces trained on; V-STaR's value is using BOTH "
        "(rejected traces define the boundary)."
    ),
    "random_seed": (
        "Determinism is the precondition for reproducibility; the selector must be "
        "re-trainable to the same result."
    ),
    "reproducibility_checksum": (
        "Content hash of the trace corpus + features; catches silent corpus drift between "
        "this run and any replication."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "selector_auroc_oof",
    "selector_pass1_vs_vote",
    "accepted_rejected_n",
    "random_seed",
    "reproducibility_checksum",
    "selector_path",
    "field_principles",
    "spec_refs",
    "inference_substrate",
)


class BlockedRun(RuntimeError):
    """Expected precondition failure that still writes a terminal artifact."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class TraceRow:
    task_id: str
    candidate_id: str
    role: str
    candidate_index: int
    vote_weight: float
    correct: bool
    features: dict[str, float]


@dataclass(frozen=True)
class TraceCorpus:
    domain: str
    source_path: Path
    rows: list[TraceRow]


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
    selector_auroc_oof: float
    selector_pass1: float
    sc_vote_pass1: float
    selector_pass1_vs_vote: float
    oof_scores: list[float]
    oof_rows: list[OOFRow]
    final_selector: dict[str, Any]


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BlockedRun("blocked_malformed_json_artifact")
    return payload


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _resolve_source_path(repo_root: Path, source: Any) -> Path:
    if not isinstance(source, str) or not source:
        raise BlockedRun("blocked_missing_candidate_pool_source")
    path = Path(source)
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.exists():
        raise BlockedRun("blocked_candidate_pool_missing")
    return resolved


def _feature_map(row: dict[str, Any], role: str, candidate_index: int, vote_weight: float) -> dict[str, float]:
    return {
        "role_repair": 1.0 if role == "repair" else 0.0,
        "vote_weight": float(vote_weight),
        "candidate_index": float(candidate_index),
        "extracted_constraints": _as_float(row.get("extracted_constraints")),
    }


def _code_trace_rows(source_path: Path) -> list[TraceRow]:
    payload = _read_json_object(source_path)
    raw_rows = payload.get("results")
    if not isinstance(raw_rows, list):
        raise BlockedRun("blocked_candidate_pool_missing_rows")
    rows: list[TraceRow] = []
    for task_index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            continue
        task_id = str(raw.get("task_id") or f"task-{task_index}")
        specs = (
            ("baseline", 0, 1.0, raw.get("baseline_passed")),
            ("repair", 1, 0.0, raw.get("repair_passed")),
        )
        for role, candidate_index, vote_weight, correct in specs:
            if not isinstance(correct, bool):
                continue
            rows.append(
                TraceRow(
                    task_id=task_id,
                    candidate_id=f"{task_id}::{role}",
                    role=role,
                    candidate_index=candidate_index,
                    vote_weight=vote_weight,
                    correct=correct,
                    features=_feature_map(raw, role, candidate_index, vote_weight),
                )
            )
    if not rows:
        raise BlockedRun("blocked_no_labeled_candidate_traces")
    return rows


def load_trace_corpus(repo_root: Path | str = Path(".")) -> TraceCorpus:
    """SCENARIO-VERIFY-4176: load traces selected by Exp 4175 without live inference."""

    root = Path(repo_root)
    headroom_path = root / HEADROOM_REL
    if not headroom_path.exists():
        raise BlockedRun("blocked_missing_headroom_gate")
    headroom = _read_json_object(headroom_path)
    domain = str(headroom.get("headroom_present_domain") or "")
    if not domain:
        raise BlockedRun("blocked_no_headroom_present_domain")
    per_domain = headroom.get("per_domain_headroom")
    domain_stats = per_domain.get(domain) if isinstance(per_domain, dict) else None
    if not isinstance(domain_stats, dict):
        raise BlockedRun("blocked_headroom_domain_missing_stats")
    artifact_flags = domain_stats.get("artifact_flags")
    source_path = _resolve_source_path(
        root, artifact_flags.get("source") if isinstance(artifact_flags, dict) else None
    )
    if domain != "code":
        raise BlockedRun(f"blocked_unsupported_headroom_domain_{domain}")
    return TraceCorpus(domain=domain, source_path=source_path, rows=_code_trace_rows(source_path))


def accepted_rejected_counts(rows: list[TraceRow]) -> dict[str, int]:
    accepted = sum(row.correct for row in rows)
    rejected = len(rows) - accepted
    return {"accepted": int(accepted), "rejected": int(rejected), "total": len(rows)}


def _feature_vector(features: dict[str, float]) -> list[float]:
    return [float(features[name]) for name in FEATURE_NAMES]


def reproducibility_checksum(rows: list[TraceRow]) -> str:
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "rows": [
            {
                "candidate_id": row.candidate_id,
                "correct": row.correct,
                "features": row.features,
                "task_id": row.task_id,
            }
            for row in rows
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _split_task_folds(rows: list[TraceRow], random_seed: int, n_folds: int) -> list[set[str]]:
    task_ids = sorted({row.task_id for row in rows})
    fold_count = max(2, min(int(n_folds), len(task_ids)))
    shuffled = task_ids[:]
    random.Random(random_seed).shuffle(shuffled)
    return [set(shuffled[index::fold_count]) for index in range(fold_count)]


def _train_selector(rows: list[TraceRow], random_seed: int) -> dict[str, Any]:
    model = LogisticRegression(random_state=random_seed, solver="liblinear", max_iter=1000)
    model.fit([_feature_vector(row.features) for row in rows], [int(row.correct) for row in rows])
    return {
        "model_type": "logistic_regression",
        "feature_names": list(FEATURE_NAMES),
        "intercept": float(model.intercept_[0]),
        "coefficients": [float(value) for value in model.coef_[0]],
    }


def score_with_selector(selector: dict[str, Any], features: dict[str, float]) -> float:
    values = [float(features[name]) for name in selector["feature_names"]]
    logit = float(selector["intercept"]) + sum(
        float(weight) * value for weight, value in zip(selector["coefficients"], values, strict=True)
    )
    return 1.0 / (1.0 + math.exp(-logit))


def _auroc(labels: list[bool], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positives or not negatives:
        raise BlockedRun("blocked_lacks_accepted_rejected_traces")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / float(len(positives) * len(negatives))


def _pass1(rows: list[TraceRow], scores: dict[str, float]) -> float:
    grouped: dict[str, list[TraceRow]] = defaultdict(list)
    for row in rows:
        grouped[row.task_id].append(row)
    correct = 0
    for task_rows in grouped.values():
        selected = max(
            task_rows,
            key=lambda row: (scores[row.candidate_id], row.vote_weight, -row.candidate_index),
        )
        correct += int(selected.correct)
    return correct / float(len(grouped))


def train_oof_selector(
    rows: list[TraceRow], *, random_seed: int = RANDOM_SEED, n_folds: int = 5
) -> OOFReport:
    """REQ-VERIFY-4176: train folds on accepted and rejected traces, score held-out rows."""

    counts = accepted_rejected_counts(rows)
    if counts["accepted"] == 0 or counts["rejected"] == 0:
        raise BlockedRun("blocked_lacks_accepted_rejected_traces")
    folds = _split_task_folds(rows, random_seed, n_folds)
    oof_scores_by_id: dict[str, float] = {}
    oof_rows: list[OOFRow] = []
    for fold, heldout_task_ids in enumerate(folds):
        train_rows = [row for row in rows if row.task_id not in heldout_task_ids]
        test_rows = [row for row in rows if row.task_id in heldout_task_ids]
        selector = _train_selector(train_rows, random_seed + fold)
        train_task_ids = tuple(sorted({row.task_id for row in train_rows}))
        for row in test_rows:
            score = score_with_selector(selector, row.features)
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
    selector_pass1 = _pass1(rows, oof_scores_by_id)
    vote_scores = {row.candidate_id: row.vote_weight for row in rows}
    sc_vote_pass1 = _pass1(rows, vote_scores)
    return OOFReport(
        selector_auroc_oof=_auroc([row.correct for row in rows], oof_scores),
        selector_pass1=selector_pass1,
        sc_vote_pass1=sc_vote_pass1,
        selector_pass1_vs_vote=selector_pass1 - sc_vote_pass1,
        oof_scores=oof_scores,
        oof_rows=oof_rows,
        final_selector=_train_selector(rows, random_seed),
    )


def persist_selector(
    path: Path,
    selector: dict[str, Any],
    *,
    checksum: str,
    counts: dict[str, int],
    random_seed: int,
) -> None:
    payload = {
        **selector,
        "accepted_rejected_n": counts,
        "random_seed": random_seed,
        "reproducibility_checksum": checksum,
        "spec_refs": ["REQ-VERIFY-4176", "SCENARIO-VERIFY-4176"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_selector(path: Path | str) -> dict[str, Any]:
    return _read_json_object(Path(path))


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _blocked_artifact(reason: str, random_seed: int, duration_s: float) -> dict[str, Any]:
    return {
        "honest_verdict": reason,
        "selector_auroc_oof": 0.0,
        "selector_pass1_vs_vote": 0.0,
        "accepted_rejected_n": {"accepted": 0, "rejected": 0, "total": 0},
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum([]),
        "selector_path": "",
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4176", "SCENARIO-VERIFY-4176"],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "acceptance_gate": True,
    }


def _complete_artifact(
    corpus: TraceCorpus,
    report: OOFReport,
    *,
    checksum: str,
    counts: dict[str, int],
    selector_path: Path,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    delta = _round_metric(report.selector_pass1_vs_vote)
    verdict = (
        f"complete: vstar_selector_trained_delta_{delta:.4f}_"
        f"auroc_{report.selector_auroc_oof:.4f}"
    )
    return {
        "honest_verdict": verdict,
        "selector_auroc_oof": _round_metric(report.selector_auroc_oof),
        "selector_pass1_vs_vote": delta,
        "accepted_rejected_n": counts,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "selector_path": str(selector_path),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4176", "SCENARIO-VERIFY-4176"],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "domain": corpus.domain,
        "candidate_pool_source": str(corpus.source_path),
        "feature_names": list(FEATURE_NAMES),
        "selector_pass1_oof": _round_metric(report.selector_pass1),
        "sc_vote_pass1_oof": _round_metric(report.sc_vote_pass1),
        "oof_folds": len({row.fold for row in report.oof_rows}),
        "duration_s": round(duration_s, 6),
        "acceptance_gate": True,
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if not isinstance(artifact["selector_auroc_oof"], float):
        raise ValueError("selector_auroc_oof must be a bare float")
    if not isinstance(artifact["selector_pass1_vs_vote"], float):
        raise ValueError("selector_pass1_vs_vote must be a bare float")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4176")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("unexpected inference_substrate")
    counts = artifact["accepted_rejected_n"]
    if not isinstance(counts, dict) or sorted(counts) != ["accepted", "rejected", "total"]:
        raise ValueError("accepted_rejected_n must contain accepted/rejected/total")
    if verdict.startswith("complete:") and (
        counts["accepted"] <= 0 or counts["rejected"] <= 0 or not Path(artifact["selector_path"]).exists()
    ):
        raise ValueError("complete selector artifacts require both classes and a persisted selector")


def run(
    repo_root: Path | str = Path("."), *, random_seed: int = RANDOM_SEED, n_folds: int = 5
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    try:
        corpus = load_trace_corpus(root)
        counts = accepted_rejected_counts(corpus.rows)
        checksum = reproducibility_checksum(corpus.rows)
        report = train_oof_selector(corpus.rows, random_seed=random_seed, n_folds=n_folds)
        selector_path = (root / SELECTOR_REL).resolve()
        persist_selector(
            selector_path,
            report.final_selector,
            checksum=checksum,
            counts=counts,
            random_seed=random_seed,
        )
        artifact = _complete_artifact(
            corpus,
            report,
            checksum=checksum,
            counts=counts,
            selector_path=selector_path,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    except BlockedRun as blocked:
        artifact = _blocked_artifact(blocked.reason, random_seed, time.perf_counter() - start)
    validate_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
