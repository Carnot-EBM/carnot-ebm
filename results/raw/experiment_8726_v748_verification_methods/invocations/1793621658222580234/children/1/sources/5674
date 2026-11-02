"""Build the Exp 2940 verifier-ensemble AUPRC/base-rate artifact.

Spec refs: REQ-REPORT-2940, SCENARIO-REPORT-2940.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_2940_verifier_ensemble_auprc_code_corpora_v1.json")
EXP2910_REL_PATH = Path("results/experiment_2910_sota_code_generation_corrigendum_v2.json")
EXP2837_REL_PATH = Path("results/experiment_2837_fover_memory_leakage_v3.json")
RUN_DATE = "20260523"
ARTIFACT = "experiment_2940_verifier_ensemble_auprc_code_corpora_v1"
SCHEMA = "carnot.verifier_ensemble_auprc_code_corpora.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_BASELINE_AUPRC = 0.075
RANDOM_BASELINE_PRINCIPLE = (
    "Equal to the positive base rate; any verifier with AUPRC at or below this "
    "value provides no information on this corpus."
)
FOVER_AUPRC_PRINCIPLE = (
    "Recomputed from exp2837 for the apples-to-apples comparison the paper needs."
)
PPV_50_PRINCIPLE = (
    "The threshold where verifier-approval has at least 50/50 odds of being "
    "correct. If unreachable, verifier is a hallucination multiplier on this corpus."
)
PAPER_V6_RECOMMENDATION_PRINCIPLE = (
    "retain | narrow | retract for the code-corpus active-inference claim."
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "code_corpus_auprc",
    "code_corpus_baseline_random_auprc",
    "fover_corpus_auprc",
    "max_f1_operating_point",
    "ppv_50_operating_point",
    "recall_80_operating_point",
    "paper_v6_recommendation",
    "cited_upstream_artifacts",
    "methodology_note",
    "duration_s",
)


@dataclass(frozen=True)
class OperatingPoint:
    """One precision-recall operating point."""

    threshold: float
    ppv: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "ppv": self.ppv,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class PrecisionRecallSummary:
    """AUPRC plus the three requested operating points."""

    auprc: float
    points: list[OperatingPoint]
    max_f1: OperatingPoint
    ppv_50: OperatingPoint
    recall_80: OperatingPoint


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_status_energy(row: dict[str, Any]) -> float:
    """Map Exp 2910 upstream status fields to a monotone verifier energy."""

    if not row.get("extraction_success"):
        return 3.0
    if not row.get("syntax_success"):
        return 2.0
    if not row.get("runtime_success"):
        return 1.0
    return 0.0


def approval_score_from_energy(energy: float) -> float:
    """Convert lower-is-better energy into higher-is-better approval score."""

    if energy <= 0.0:
        return 1.0
    if energy <= 1.0:
        return 0.5
    if energy <= 2.0:
        return 0.25
    return 0.0


def code_labels_scores_from_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[int], list[float], list[float]]:
    """Return correct-candidate labels, approval scores, and status energies."""

    labels: list[int] = []
    scores: list[float] = []
    energies: list[float] = []
    for row in candidates:
        energy = candidate_status_energy(row)
        labels.append(1 if row.get("passed") is True else 0)
        scores.append(approval_score_from_energy(energy))
        energies.append(energy)
    return labels, scores, energies


def summarize_precision_recall(labels: list[int], scores: list[float]) -> PrecisionRecallSummary:
    """Compute average-precision AUPRC and requested operating thresholds."""

    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and same-length")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("precision-recall summary requires both classes")

    points: list[OperatingPoint] = []
    auprc = 0.0
    previous_recall = 0.0
    tp = 0
    fp = 0
    ordered = sorted(zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True)
    cursor = 0
    while cursor < len(ordered):
        threshold = float(ordered[cursor][0])
        group_tp = 0
        group_fp = 0
        while cursor < len(ordered) and float(ordered[cursor][0]) == threshold:
            if int(ordered[cursor][1]) == 1:
                group_tp += 1
            else:
                group_fp += 1
            cursor += 1
        tp += group_tp
        fp += group_fp
        ppv = tp / (tp + fp)
        recall = tp / positives
        f1 = 2.0 * ppv * recall / (ppv + recall) if (ppv + recall) else 0.0
        auprc += (recall - previous_recall) * ppv
        previous_recall = recall
        points.append(OperatingPoint(threshold=threshold, ppv=ppv, recall=recall, f1=f1))

    max_f1 = max(points, key=lambda point: (point.f1, point.threshold))
    ppv_eligible = [point for point in points if point.ppv >= 0.5]
    ppv_50 = max(ppv_eligible, key=lambda point: (point.recall, -point.threshold))
    recall_eligible = [point for point in points if point.recall >= 0.8]
    recall_80 = max(recall_eligible, key=lambda point: (point.threshold, point.ppv))
    return PrecisionRecallSummary(
        auprc=auprc,
        points=points,
        max_f1=max_f1,
        ppv_50=ppv_50,
        recall_80=recall_80,
    )


def fover_labels_scores_from_artifact(
    exp2837: dict[str, Any],
    root: Path,
) -> tuple[list[int], list[float], list[str]]:
    rows = exp2837.get("fover_candidate_scores")
    if isinstance(rows, list) and rows:
        labels: list[int] = []
        scores: list[float] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            labels.append(_correct_label(row.get("label")))
            scores.append(_approval_score_from_fover_row(row))
        return labels, scores, ["fover_candidate_scores"]
    labels, scores = _recompute_fover_labels_scores_from_local_corpus(exp2837, root)
    return labels, scores, ["per_seed_results", "random_seeds_used", "n_examples"]


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """Build the Exp 2940 terminal artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else started_s
    exp2910_path = root_path / EXP2910_REL_PATH
    exp2837_path = root_path / EXP2837_REL_PATH
    exp2910 = read_json_object(exp2910_path)
    exp2837 = read_json_object(exp2837_path)
    preconditions = _preconditions(root_path, exp2910_path, exp2837_path, exp2910, exp2837)

    if any(not item["available"] for item in preconditions):
        end = time.perf_counter() if now_s is None else now_s
        duration_s = round(max(0.0, end - start), 6)
        return _blocked_artifact(
            preconditions=preconditions,
            duration_s=duration_s,
            exp2910_path=exp2910_path,
            exp2837_path=exp2837_path,
        )

    candidates = list(exp2910["candidate_results"])
    code_labels, code_scores, code_energies = code_labels_scores_from_candidates(candidates)
    code_summary = summarize_precision_recall(code_labels, code_scores)
    fover_labels, fover_scores, fover_fields = fover_labels_scores_from_artifact(exp2837, root_path)
    fover_summary = summarize_precision_recall(fover_labels, fover_scores)
    recommendation = _paper_v6_recommendation(code_summary)
    end = time.perf_counter() if now_s is None else now_s
    duration_s = round(max(0.0, end - start), 6)

    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "honest_verdict": _honest_verdict(recommendation),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "code_corpus_auprc": code_summary.auprc,
        "code_corpus_empirical_positive_rate": sum(code_labels) / len(code_labels),
        "code_corpus_candidate_count": len(code_labels),
        "code_corpus_positive_count": sum(code_labels),
        "code_corpus_baseline_random_auprc": {
            "value": RANDOM_BASELINE_AUPRC,
            "principle": RANDOM_BASELINE_PRINCIPLE,
        },
        "fover_corpus_auprc": {
            "value": fover_summary.auprc,
            "principle": FOVER_AUPRC_PRINCIPLE,
        },
        "fover_corpus_positive_rate": sum(fover_labels) / len(fover_labels),
        "fover_corpus_candidate_count": len(fover_labels),
        "max_f1_operating_point": code_summary.max_f1.as_dict(),
        "ppv_50_operating_point": {
            **code_summary.ppv_50.as_dict(),
            "principle": PPV_50_PRINCIPLE,
        },
        "recall_80_operating_point": code_summary.recall_80.as_dict(),
        "precision_recall_curve": [point.as_dict() for point in code_summary.points],
        "paper_v6_recommendation": {
            "value": recommendation,
            "principle": PAPER_V6_RECOMMENDATION_PRINCIPLE,
        },
        "cited_upstream_artifacts": [
            {
                "experiment_id": "exp2910",
                "fields_imported": [
                    "candidate_results",
                    "codegen_corrigendum_ready",
                    "k_candidates_per_task",
                    "per_task_results",
                ],
                "sha256": sha256_file(exp2910_path),
            },
            {
                "experiment_id": "exp2837",
                "fields_imported": fover_fields,
                "sha256": sha256_file(exp2837_path),
            },
        ],
        "acceptance_gates": {
            "code_corpus_auprc_probability": 0.0 <= code_summary.auprc <= 1.0,
            "code_corpus_auprc_not_exact_0_5": code_summary.auprc != 0.5,
            "cites_at_least_two_upstream_artifacts": True,
        },
        "methodology_note": _methodology_note(fover_fields),
        "code_status_energy_definition": {
            "0.0": "extracted, syntax-valid, runtime-clean candidate",
            "1.0": "extracted and syntax-valid candidate with runtime failure",
            "2.0": "extracted candidate with syntax failure",
            "3.0": "candidate extraction failed",
            "note": (
                "Exp 2910 does not store a learned verifier-energy tensor; this "
                "audit uses only upstream per-candidate verifier/status fields "
                "and keeps sandbox pass/fail as the correctness label."
            ),
        },
        "code_status_energy_values": code_energies,
        "run_date": RUN_DATE,
        "duration_s": duration_s,
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """Build and persist the Exp 2940 artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _preconditions(
    root: Path,
    exp2910_path: Path,
    exp2837_path: Path,
    exp2910: dict[str, Any],
    exp2837: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = exp2910.get("candidate_results")
    fover_rows = exp2837.get("fover_candidate_scores")
    fover_corpus_path = root / "data" / "fover_corpus.jsonl"
    return [
        {
            "resource": "exp2910_artifact",
            "available": exp2910_path.is_file() and bool(exp2910),
            "detail": str(EXP2910_REL_PATH),
        },
        {
            "resource": "exp2910_codegen_corrigendum_ready",
            "available": exp2910.get("codegen_corrigendum_ready") is True,
            "detail": str(exp2910.get("codegen_corrigendum_ready")),
        },
        {
            "resource": "exp2910_candidate_results",
            "available": isinstance(candidates, list) and len(candidates) > 0,
            "detail": f"count={len(candidates) if isinstance(candidates, list) else 0}",
        },
        {
            "resource": "exp2837_artifact",
            "available": exp2837_path.is_file() and bool(exp2837),
            "detail": str(EXP2837_REL_PATH),
        },
        {
            "resource": "exp2837_fover_score_source",
            "available": (isinstance(fover_rows, list) and len(fover_rows) > 0)
            or fover_corpus_path.is_file(),
            "detail": "raw_score_rows" if isinstance(fover_rows, list) else str(fover_corpus_path),
        },
    ]


def _blocked_artifact(
    *,
    preconditions: list[dict[str, Any]],
    duration_s: float,
    exp2910_path: Path,
    exp2837_path: Path,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "honest_verdict": "blocked_required_upstream_artifact_missing",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "code_corpus_auprc": None,
        "code_corpus_baseline_random_auprc": {
            "value": RANDOM_BASELINE_AUPRC,
            "principle": RANDOM_BASELINE_PRINCIPLE,
        },
        "fover_corpus_auprc": {
            "value": None,
            "principle": FOVER_AUPRC_PRINCIPLE,
        },
        "max_f1_operating_point": None,
        "ppv_50_operating_point": {
            "threshold": None,
            "ppv": None,
            "recall": None,
            "f1": None,
            "principle": PPV_50_PRINCIPLE,
        },
        "recall_80_operating_point": None,
        "paper_v6_recommendation": {
            "value": "retract",
            "principle": PAPER_V6_RECOMMENDATION_PRINCIPLE,
        },
        "cited_upstream_artifacts": [
            {"experiment_id": "exp2910", "fields_imported": [], "sha256": sha256_file(exp2910_path)},
            {"experiment_id": "exp2837", "fields_imported": [], "sha256": sha256_file(exp2837_path)},
        ],
        "methodology_note": (
            "Blocked before AUPRC computation because one or more required upstream "
            "artifacts or score sources were missing."
        ),
        "run_date": RUN_DATE,
        "duration_s": duration_s,
    }


def _correct_label(label: Any) -> int:
    if label in {"correct", "passed", "pass", True}:
        return 1
    if label in {"incorrect", "failed", "fail", False}:
        return 0
    raise ValueError(f"unsupported FoVer correctness label: {label!r}")


def _approval_score_from_fover_row(row: dict[str, Any]) -> float:
    if "approval_score" in row:
        return float(row["approval_score"])
    if "score" in row:
        return float(row["score"])
    return -float(row["energy"])


def _recompute_fover_labels_scores_from_local_corpus(
    exp2837: dict[str, Any],
    root: Path,
) -> tuple[list[int], list[float]]:
    from carnot.eval import fover_memory_leakage_v3 as fover

    seeds = [int(seed) for seed in exp2837.get("random_seeds_used", [42])]
    n_examples = int(exp2837.get("n_examples", 1000))
    corpus_rows = fover._read_fover_rows(root / "data" / "fover_corpus.jsonl")
    memory_index = fover._load_fr11_memory_index(root)
    labels: list[int] = []
    scores: list[float] = []
    for seed in seeds:
        rows = fover._select_balanced_subset(corpus_rows, seed=seed, n_examples=n_examples)
        texts = [str(row.get("step_text", "")) for row in rows]
        verifier_scores = fover._score_text_verifiers(texts)
        architecture_scores = [
            0.9 * r_score + 0.1 * u_score
            for r_score, u_score in zip(
                verifier_scores["tier0r_curry_howard"],
                verifier_scores["tier0u_logical_consistency"],
                strict=True,
            )
        ]
        if memory_index["question_ids"] or memory_index["prompt_token_sets"]:
            memory_scores = [fover._fr11_memory_score(row, memory_index) for row in rows]
            architecture_scores = [
                score + fover.FR11_MEMORY_BOOST * memory_score
                for score, memory_score in zip(architecture_scores, memory_scores, strict=True)
            ]
        labels.extend(1 if fover._label_to_int(row["label"]) == 0 else 0 for row in rows)
        scores.extend(-score for score in architecture_scores)
    return labels, scores


def _paper_v6_recommendation(summary: PrecisionRecallSummary) -> str:
    if summary.auprc > 2.0 * RANDOM_BASELINE_AUPRC and summary.max_f1.f1 > 0.30:
        return "retain"
    return "retract"


def _honest_verdict(recommendation: str) -> str:
    if recommendation == "retain":
        return "complete: verifier provides meaningful information on code corpora"
    return "complete: retract code-corpus active-inference claims"


def _methodology_note(fover_fields: list[str]) -> str:
    fover_source = (
        "raw Exp 2837 score rows"
        if fover_fields == ["fover_candidate_scores"]
        else "the local FoVer scoring path used by Exp 2837 because the checked-in "
        "artifact stores AUROC summaries rather than raw score rows"
    )
    return (
        "Computed average-precision AUPRC and threshold operating points without "
        "using AUROC. Code labels are Exp 2910 sandbox pass/fail outcomes. Code "
        "approval scores are derived only from Exp 2910 per-candidate extraction, "
        "syntax, and runtime-status fields because the artifact does not store a "
        f"learned energy tensor. FoVer AUPRC uses {fover_source} and the same "
        "precision-recall implementation."
    )


def main() -> int:  # pragma: no cover
    artifact = write_artifact(REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
