"""Build the Exp 3273 prompt-injection KAN full-corpus DeLong eval artifact.

Spec refs: REQ-REPORT-3273, SCENARIO-REPORT-3273.

The evaluator treats the KAN as a sidecar detector, not as exact verifier
authority. It trains only on the frozen v4 train split, scores eval, holdout,
and Garak rows, and reports statistical uncertainty against deployable
text-only baselines before any downstream repair gate can consume the result.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from carnot.models.prompt_injection_features import encode_prompt_injection
from carnot.models.prompt_injection_kan import InjectionExample, PromptInjectionEnergyCheckerV3


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.prompt_injection_kan_full_corpus_delong_eval.v1"
EXPERIMENT_ID = "exp3273"
TASK_ID = "exp3273-prompt-injection-kan-full-corpus-delong-eval-v1"
ARTIFACT = "experiment_3273_prompt_injection_kan_full_corpus_delong_eval_v1"
MILESTONE = "2026.05.303"
RUN_DATE = "20260528"
RANDOM_SEED = 3273
DEFAULT_N_EPOCHS = 100
DEFAULT_LR = 1e-3
NONINFERIORITY_MARGIN_AUROC = -0.02
SHARD_302_AUROC_FALLBACK = 0.791096

OUTPUT_REL_PATH = Path(
    "results/experiment_3273_prompt_injection_kan_full_corpus_delong_eval_v1.json"
)
SCRIPT_REL_PATH = (
    REPO_ROOT / "scripts" / "experiment_3273_prompt_injection_kan_full_corpus_delong_eval_v1.py"
)
EXP3272_REL_PATH = Path(
    "results/experiment_3272_prompt_injection_v4_full_corpus_assembly_leakage_audit_v1.json"
)
EXP3265_REL_PATH = Path("results/experiment_3265_prompt_injection_kan_train_eval_shard_v3.json")
SPLIT_REL_PATHS = {
    "train": Path("data/prompt_injection_v4/frozen_splits/prompt_injection_v4_train_v1.jsonl"),
    "eval": Path("data/prompt_injection_v4/frozen_splits/prompt_injection_v4_eval_v1.jsonl"),
    "holdout": Path(
        "data/prompt_injection_v4/frozen_splits/prompt_injection_v4_holdout_v1.jsonl"
    ),
    "garak": Path("data/prompt_injection_v4/frozen_splits/prompt_injection_v4_garak_v1.jsonl"),
}
NORMAL_SPLITS = ("train", "eval", "holdout")
EVAL_SPLITS = ("eval", "holdout")
ALL_EVAL_SPLITS = ("eval", "holdout", "garak")
ALLOWED_LABELS = ("benign", "injection")
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
DEPLOYABLE_BASELINES = ("keyword_feature_baseline", "regex_phrase_baseline")

REQUIRED_ARTIFACT_FIELDS = {
    "v4_full_eval_ready",
    "full_corpus_auroc",
    "full_corpus_auprc",
    "delong_ci",
    "delong_noninferiority_passed",
    "calibration_ece",
    "per_slice_metrics",
    "garak_split_preliminary_metrics",
    "sidecar_only",
    "output_paths",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}

REGEX_BASELINE_TERMS = (
    "ignore",
    "disregard",
    "override",
    "jailbreak",
    "system prompt",
    "developer instructions",
    "reveal",
    "exfiltrate",
    "secret",
    "credential",
    "attacker",
    "tool output",
    "encoded",
)


def run_experiment(
    *,
    project_root: str | Path = REPO_ROOT,
    output_path: str | Path = OUTPUT_REL_PATH,
    monotonic: ClockFn = time.perf_counter,
    random_seed: int = RANDOM_SEED,
    n_epochs: int = DEFAULT_N_EPOCHS,
    lr: float = DEFAULT_LR,
) -> JsonDict:
    """REQ-REPORT-3273: train on frozen train and evaluate leakage-audited splits."""

    start = monotonic()
    root = Path(project_root)
    out_path = resolve_output_path(root, output_path)
    exp3272 = read_json_object(root / EXP3272_REL_PATH)
    blocked_reason = precondition_blocker(root=root, exp3272=exp3272)
    rows_by_split: dict[str, list[JsonDict]] = {split: [] for split in SPLIT_REL_PATHS}
    leakage_audit = empty_leakage_audit()

    if not blocked_reason:
        rows_by_split = load_frozen_splits(root)
        leakage_audit = audit_frozen_split_leakage(rows_by_split)
        if leakage_audit["leakage_audit_passed"] is not True:
            blocked_reason = "frozen_split_leakage_detected"
        elif not has_both_classes(rows_by_split["train"]):
            blocked_reason = "train_split_lacks_both_classes"
        elif not has_both_classes(rows_by_split["eval"] + rows_by_split["holdout"]):
            blocked_reason = "eval_holdout_scope_lacks_both_classes"

    if blocked_reason:
        artifact = empty_artifact(
            blocked_reason=blocked_reason,
            duration_s=duration(start, monotonic()),
            output_path=output_path,
            random_seed=random_seed,
        )
    else:
        artifact = build_ready_artifact(
            root=root,
            output_path=output_path,
            rows_by_split=rows_by_split,
            leakage_audit=leakage_audit,
            exp3272=exp3272,
            duration_s=0.0,
            random_seed=random_seed,
            n_epochs=n_epochs,
            lr=lr,
        )
        artifact["duration_s"] = duration(start, monotonic())

    validate_artifact(artifact)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def build_ready_artifact(
    *,
    root: Path,
    output_path: str | Path,
    rows_by_split: Mapping[str, list[JsonDict]],
    leakage_audit: Mapping[str, Any],
    exp3272: Mapping[str, Any],
    duration_s: float,
    random_seed: int,
    n_epochs: int,
    lr: float,
) -> JsonDict:
    """Train the KAN sidecar and assemble the full statistical result."""

    checker = PromptInjectionEnergyCheckerV3()
    train_examples = [to_injection_example(row) for row in rows_by_split["train"]]
    loss_curve = checker.train(train_examples, n_epochs=int(n_epochs), lr=float(lr))

    scores_by_split = {
        split: score_rows(checker, rows_by_split[split]) for split in ALL_EVAL_SPLITS
    }
    eval_rows = rows_by_split["eval"] + rows_by_split["holdout"]
    eval_scores = scores_by_split["eval"] + scores_by_split["holdout"]
    eval_labels = labels_for(eval_rows)

    full_corpus_auroc = metric_float(compute_auroc(eval_labels, eval_scores))
    full_corpus_auprc = metric_float(compute_auprc(eval_labels, eval_scores))
    selected_thresholds = select_thresholds(
        labels_for(rows_by_split["eval"]),
        scores_by_split["eval"],
    )
    threshold_metrics = threshold_metric_table(rows_by_split, scores_by_split, selected_thresholds)
    calibration = calibration_report(
        labels_for(rows_by_split["eval"]),
        scores_by_split["eval"],
        labels_for(rows_by_split["holdout"]),
        scores_by_split["holdout"],
    )
    baseline_scores = baseline_score_table(eval_rows)
    baseline_metrics = {
        name: metric_summary(eval_labels, scores) for name, scores in baseline_scores.items()
    }
    reference_name = strongest_deployable_baseline(baseline_metrics)
    delong = delong_noninferiority(
        eval_labels,
        eval_scores,
        baseline_scores[reference_name],
        margin=NONINFERIORITY_MARGIN_AUROC,
    )
    exact_delong = delong_noninferiority(
        eval_labels,
        eval_scores,
        baseline_scores["exact_label_upper_bound"],
        margin=NONINFERIORITY_MARGIN_AUROC,
    )
    garak_metrics = garak_preliminary_metrics(
        rows_by_split["garak"],
        scores_by_split["garak"],
        threshold=selected_thresholds["max_f1_eval"],
    )
    shard_comparison = shard_302_comparison(root, full_corpus_auroc)

    output_paths = output_path_list(output_path)
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "v4_full_eval_ready": True,
        "blocked_reason": "",
        "full_corpus_eval_scope": "eval_plus_holdout_excludes_train_and_garak_single_class",
        "full_corpus_auroc": full_corpus_auroc,
        "full_corpus_auprc": full_corpus_auprc,
        "delong_ci": list(delong["ci"]),
        "delong_noninferiority_passed": bool(delong["noninferiority_passed"]),
        "delong_comparison": {
            "method": "paired_delong_auc_ci",
            "reference_detector": reference_name,
            "noninferiority_margin_auroc": NONINFERIORITY_MARGIN_AUROC,
            "candidate_minus_reference_auroc": delong["auc_difference"],
            "ci": delong["ci"],
            "standard_error": delong["standard_error"],
            "exact_label_upper_bound_reference": exact_delong,
        },
        "calibration_ece": metric_float(calibration["holdout_ece"]),
        "calibration": calibration,
        "per_slice_metrics": per_slice_metrics(
            rows_by_split["eval"] + rows_by_split["holdout"] + rows_by_split["garak"],
            eval_scores + scores_by_split["garak"],
            threshold=selected_thresholds["max_f1_eval"],
        ),
        "garak_split_preliminary_metrics": garak_metrics,
        "sidecar_only": True,
        "split_metrics": {
            "eval": metric_summary(labels_for(rows_by_split["eval"]), scores_by_split["eval"]),
            "holdout": metric_summary(
                labels_for(rows_by_split["holdout"]),
                scores_by_split["holdout"],
            ),
            "eval_plus_holdout": metric_summary(eval_labels, eval_scores),
        },
        "threshold_metrics": threshold_metrics,
        "baseline_detector_metrics": baseline_metrics,
        "shard_302_comparison": shard_comparison,
        "training_summary": {
            "n_train": len(rows_by_split["train"]),
            "train_label_counts": label_counts(rows_by_split["train"]),
            "loss_curve_count": len(loss_curve),
            "loss_curve_first": metric_float(loss_curve[0]) if loss_curve else None,
            "loss_curve_last": metric_float(loss_curve[-1]) if loss_curve else None,
            "trained_model_checksum": model_checksum(checker),
            "model_specs": model_specs(checker, n_epochs=n_epochs, lr=lr),
        },
        "split_counts": {split: len(rows_by_split[split]) for split in SPLIT_REL_PATHS},
        "leakage_audit": dict(leakage_audit),
        "source_artifacts": source_artifacts(root, exp3272),
        "output_paths": output_paths,
        "checksums": file_checksums(root, output_paths[1:]),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "duration_s": duration_s,
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    return artifact


def empty_artifact(
    *,
    blocked_reason: str,
    duration_s: float,
    output_path: str | Path,
    random_seed: int,
) -> JsonDict:
    """Return a complete sidecar-only gated-skip artifact."""

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "v4_full_eval_ready": False,
        "blocked_reason": str(blocked_reason),
        "full_corpus_eval_scope": "not_evaluated",
        "full_corpus_auroc": 0.0,
        "full_corpus_auprc": 0.0,
        "delong_ci": [0.0, 0.0],
        "delong_noninferiority_passed": False,
        "delong_comparison": {},
        "calibration_ece": 0.0,
        "calibration": {},
        "per_slice_metrics": {},
        "garak_split_preliminary_metrics": {},
        "sidecar_only": True,
        "split_metrics": {},
        "threshold_metrics": {},
        "baseline_detector_metrics": {},
        "shard_302_comparison": {
            "prior_shard_auroc": SHARD_302_AUROC_FALLBACK,
            "prior_shard_source": EXP3265_REL_PATH.as_posix(),
            "full_minus_prior_shard_auroc": 0.0,
        },
        "training_summary": {},
        "split_counts": {},
        "leakage_audit": empty_leakage_audit(),
        "source_artifacts": {},
        "output_paths": [path_as_artifact_string(output_path)],
        "checksums": {},
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "duration_s": duration_s,
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    return artifact


def precondition_blocker(*, root: Path, exp3272: Mapping[str, Any]) -> str:
    """Return the first closed gate before training can start."""

    if exp3272.get("full_15k_corpus_ready") is not True:
        return "gated_exp3272_full_15k_corpus_not_ready"
    if exp3272.get("leakage_audit_passed") is not True:
        return "gated_exp3272_leakage_audit_not_passed"
    missing = [rel.as_posix() for rel in SPLIT_REL_PATHS.values() if not (root / rel).is_file()]
    if missing:
        return "missing_frozen_split_files:" + ",".join(missing)
    return ""


def load_frozen_splits(root: Path) -> dict[str, list[JsonDict]]:
    """Read the frozen JSONL splits and normalize labels for evaluation."""

    return {split: [normalize_row(row, split) for row in read_jsonl(root / rel)] for split, rel in SPLIT_REL_PATHS.items()}


def normalize_row(row: Mapping[str, Any], split: str) -> JsonDict:
    """Keep only the fields needed for scoring, slices, and leakage checks."""

    label = normalize_label(row.get("teacher_label") or row.get("source_label"))
    return {
        "canonical_id": str(row.get("canonical_id") or row.get("example_id") or ""),
        "split": str(row.get("split") or split),
        "text": str(row.get("text") or ""),
        "label": label,
        "teacher_label": label,
        "category_id": str(row.get("category_id") or "unknown"),
        "instruction_alignment": str(row.get("instruction_alignment") or "unknown"),
        "training_eligible": bool(row.get("training_eligible")),
        "normalized_text_sha256": str(row.get("normalized_text_sha256") or sha256_text(str(row.get("text") or ""))),
        "near_duplicate_sha256": str(row.get("near_duplicate_sha256") or sha256_text(str(row.get("text") or "") + ":near")),
        "template_family_sha256": str(row.get("template_family_sha256") or sha256_text(str(row.get("text") or "") + ":template")),
    }


def normalize_label(value: Any) -> str:
    """Normalize unexpected labels to an explicit unknown bucket."""

    label = str(value or "").lower()
    return label if label in ALLOWED_LABELS else "unknown"


def audit_frozen_split_leakage(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    """Check that train signatures do not appear in held-out normal splits."""

    train = rows_by_split.get("train", [])
    train_exact = signature_set(train, "normalized_text_sha256")
    train_near = signature_set(train, "near_duplicate_sha256")
    train_template = signature_set(train, "template_family_sha256")
    audit: JsonDict = {"leakage_audit_passed": True}
    for split in ("eval", "holdout"):
        rows = rows_by_split.get(split, [])
        exact = train_exact & signature_set(rows, "normalized_text_sha256")
        near = train_near & signature_set(rows, "near_duplicate_sha256")
        template = train_template & signature_set(rows, "template_family_sha256")
        audit[f"train_{split}_exact_overlap_count"] = len(exact)
        audit[f"train_{split}_near_overlap_count"] = len(near)
        audit[f"train_{split}_template_overlap_count"] = len(template)
        audit[f"train_{split}_sample"] = sorted(exact | near | template)[:5]
        if exact or near or template:
            audit["leakage_audit_passed"] = False
    garak = rows_by_split.get("garak", [])
    garak_exact = train_exact & signature_set(garak, "normalized_text_sha256")
    garak_near = train_near & signature_set(garak, "near_duplicate_sha256")
    audit["train_garak_exact_overlap_count"] = len(garak_exact)
    audit["train_garak_near_overlap_count"] = len(garak_near)
    audit["train_garak_sample"] = sorted(garak_exact | garak_near)[:5]
    if garak_exact or garak_near:
        audit["leakage_audit_passed"] = False
    return audit


def empty_leakage_audit() -> JsonDict:
    """Return the leakage-audit shape used by gated-skip artifacts."""

    return {
        "leakage_audit_passed": False,
        "train_eval_exact_overlap_count": 0,
        "train_eval_near_overlap_count": 0,
        "train_eval_template_overlap_count": 0,
        "train_holdout_exact_overlap_count": 0,
        "train_holdout_near_overlap_count": 0,
        "train_holdout_template_overlap_count": 0,
        "train_garak_exact_overlap_count": 0,
        "train_garak_near_overlap_count": 0,
    }


def signature_set(rows: Sequence[Mapping[str, Any]], key: str) -> set[str]:
    """Collect non-empty split-signature values."""

    return {str(row.get(key) or "") for row in rows if row.get(key)}


def score_rows(checker: Any, rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """Score rows with the trained sidecar in input order."""

    return [float(checker.energy(str(row.get("text") or ""))) for row in rows]


def to_injection_example(row: Mapping[str, Any]) -> InjectionExample:
    """Convert a normalized row into the model's training example type."""

    return InjectionExample(
        text=str(row.get("text") or ""),
        label=str(row.get("label") or "benign"),  # type: ignore[arg-type]
        source=str(row.get("canonical_id") or row.get("split") or "v4"),
    )


def labels_for(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return binary labels where injection is positive."""

    return [1 if row.get("label") == "injection" or row.get("teacher_label") == "injection" else 0 for row in rows]


def label_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count normalized text labels for provenance summaries."""

    return dict(sorted(Counter(str(row.get("label") or "unknown") for row in rows).items()))


def has_both_classes(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only when both benign and injection are present."""

    counts = label_counts(rows)
    return counts.get("benign", 0) > 0 and counts.get("injection", 0) > 0


def compute_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute AUROC with average ranks and return None for one-class inputs."""

    label_arr, score_arr = checked_metric_arrays(labels, scores)
    positives = label_arr == 1
    n_pos = int(np.sum(positives))
    n_neg = int(label_arr.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(score_arr)
    rank_sum_pos = float(np.sum(ranks[positives]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_auprc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute average precision where injection is positive."""

    label_arr, score_arr = checked_metric_arrays(labels, scores)
    n_pos = int(np.sum(label_arr == 1))
    if n_pos == 0:
        return None
    if n_pos == label_arr.size:
        return 1.0
    order = np.argsort(-score_arr)
    sorted_labels = label_arr[order]
    tp = 0
    precision_sum = 0.0
    for rank, label in enumerate(sorted_labels, start=1):
        if int(label) == 1:
            tp += 1
            precision_sum += tp / rank
    return float(precision_sum / n_pos)


def checked_metric_arrays(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert metric inputs to arrays and reject length mismatches."""

    label_arr = np.asarray(labels, dtype=np.int32)
    score_arr = np.asarray(scores, dtype=np.float64)
    if label_arr.shape[0] != score_arr.shape[0]:
        raise ValueError("labels and scores must have the same length")
    return label_arr, score_arr


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, matching DeLong/Mann-Whitney AUROC."""

    order = np.argsort(values)
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def metrics_at_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> JsonDict:
    """Compute precision/recall/F1 for scores above a selected threshold."""

    label_arr, score_arr = checked_metric_arrays(labels, scores)
    pred = score_arr >= float(threshold)
    pos = label_arr == 1
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    tn = int(np.sum(~pred & ~pos))
    fn = int(np.sum(~pred & pos))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / label_arr.size if label_arr.size else 0.0
    return {
        "threshold": metric_float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metric_float(precision),
        "recall": metric_float(recall),
        "f1": metric_float(f1),
        "accuracy": metric_float(accuracy),
        "predicted_positive_count": int(np.sum(pred)),
    }


def metric_summary(labels: Sequence[int], scores: Sequence[float]) -> JsonDict:
    """Summarize ranking metrics for one split or slice."""

    label_arr, score_arr = checked_metric_arrays(labels, scores)
    return {
        "n": int(label_arr.size),
        "positive_count": int(np.sum(label_arr == 1)),
        "negative_count": int(np.sum(label_arr == 0)),
        "auroc": nullable_metric(compute_auroc(labels, scores)),
        "auprc": nullable_metric(compute_auprc(labels, scores)),
        "mean_score": metric_float(float(np.mean(score_arr))) if score_arr.size else 0.0,
        "min_score": metric_float(float(np.min(score_arr))) if score_arr.size else 0.0,
        "max_score": metric_float(float(np.max(score_arr))) if score_arr.size else 0.0,
    }


def select_thresholds(labels: Sequence[int], scores: Sequence[float]) -> JsonDict:
    """Select eval-only operating thresholds for reporting holdout/Garak behavior."""

    label_arr, score_arr = checked_metric_arrays(labels, scores)
    candidates = sorted({float(score) for score in score_arr})
    if not candidates:
        candidates = [0.0]
    best_f1_threshold = candidates[0]
    best_f1 = -1.0
    best_youden_threshold = candidates[0]
    best_youden = -math.inf
    for threshold in candidates:
        metrics = metrics_at_threshold(label_arr.tolist(), score_arr.tolist(), threshold)
        if float(metrics["f1"]) > best_f1:
            best_f1 = float(metrics["f1"])
            best_f1_threshold = threshold
        tpr = float(metrics["recall"])
        fpr = metrics["fp"] / (metrics["fp"] + metrics["tn"]) if (metrics["fp"] + metrics["tn"]) else 0.0
        youden = tpr - fpr
        if youden > best_youden:
            best_youden = youden
            best_youden_threshold = threshold
    benign_scores = score_arr[label_arr == 0]
    p95_benign = float(np.percentile(benign_scores, 95)) if benign_scores.size else max(candidates)
    return {
        "max_f1_eval": metric_float(best_f1_threshold),
        "youden_eval": metric_float(best_youden_threshold),
        "p95_benign_eval": metric_float(p95_benign),
    }


def threshold_metric_table(
    rows_by_split: Mapping[str, list[JsonDict]],
    scores_by_split: Mapping[str, list[float]],
    thresholds: Mapping[str, float],
) -> JsonDict:
    """Apply selected eval thresholds to each reported split."""

    table: JsonDict = {"selected_thresholds": dict(thresholds)}
    scopes = {
        "eval": ("eval",),
        "holdout": ("holdout",),
        "eval_plus_holdout": ("eval", "holdout"),
        "garak": ("garak",),
    }
    for scope, splits in scopes.items():
        rows = [row for split in splits for row in rows_by_split[split]]
        scores = [score for split in splits for score in scores_by_split[split]]
        labels = labels_for(rows)
        table[scope] = {
            name: metrics_at_threshold(labels, scores, float(threshold))
            for name, threshold in thresholds.items()
        }
    return table


def calibration_report(
    eval_labels: Sequence[int],
    eval_scores: Sequence[float],
    holdout_labels: Sequence[int],
    holdout_scores: Sequence[float],
) -> JsonDict:
    """Calibrate scores from eval distribution and report holdout ECE."""

    center, scale = calibration_center_scale(eval_scores)
    holdout_probs = score_probabilities(holdout_scores, center=center, scale=scale)
    eval_probs = score_probabilities(eval_scores, center=center, scale=scale)
    return {
        "method": "eval_score_sigmoid_center_scale",
        "center": metric_float(center),
        "scale": metric_float(scale),
        "eval_ece": metric_float(expected_calibration_error(eval_labels, eval_probs)),
        "holdout_ece": metric_float(expected_calibration_error(holdout_labels, holdout_probs)),
        "bin_count": 10,
    }


def calibration_center_scale(scores: Sequence[float]) -> tuple[float, float]:
    """Use robust eval-score center/scale for deterministic probability reporting."""

    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 1.0
    center = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    scale = float(q75 - q25)
    if scale <= 1e-12:
        scale = float(np.std(arr))
    if scale <= 1e-12:
        scale = 1.0
    return center, scale


def score_probabilities(scores: Sequence[float], *, center: float, scale: float) -> list[float]:
    """Map arbitrary sidecar scores into [0, 1] with a fixed sigmoid."""

    arr = (np.asarray(scores, dtype=np.float64) - float(center)) / max(float(scale), 1e-12)
    arr = np.clip(arr, -50.0, 50.0)
    return [float(value) for value in (1.0 / (1.0 + np.exp(-arr)))]


def expected_calibration_error(
    labels: Sequence[int],
    probabilities: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Compute expected calibration error for binary injection probabilities."""

    label_arr, prob_arr = checked_metric_arrays(labels, probabilities)
    if label_arr.size == 0:
        return 0.0
    prob_arr = np.clip(prob_arr, 0.0, 1.0)
    ece = 0.0
    for idx in range(int(n_bins)):
        low = idx / n_bins
        high = (idx + 1) / n_bins
        if idx == n_bins - 1:
            mask = (prob_arr >= low) & (prob_arr <= high)
        else:
            mask = (prob_arr >= low) & (prob_arr < high)
        if not np.any(mask):
            continue
        confidence = float(np.mean(prob_arr[mask]))
        accuracy = float(np.mean(label_arr[mask]))
        ece += float(np.mean(mask)) * abs(confidence - accuracy)
    return float(ece)


def baseline_score_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    """Score rows with deployable baselines and an exact upper-bound reference."""

    return {
        "keyword_feature_baseline": [keyword_feature_baseline_score(str(row.get("text") or "")) for row in rows],
        "regex_phrase_baseline": [regex_phrase_baseline_score(str(row.get("text") or "")) for row in rows],
        "exact_label_upper_bound": [1.0 if row.get("label") == "injection" else 0.0 for row in rows],
    }


def keyword_feature_baseline_score(text: str) -> float:
    """Text-only baseline: sum the same human-readable injection feature counts."""

    return float(np.asarray(encode_prompt_injection(text, 32)).sum())


def regex_phrase_baseline_score(text: str) -> float:
    """Text-only baseline using direct phrase counts normalized by text length."""

    lower = text.lower()
    word_count = max(len(re.findall(r"\w+", lower)), 1)
    return float(sum(lower.count(term) for term in REGEX_BASELINE_TERMS) / word_count)


def strongest_deployable_baseline(metrics: Mapping[str, Mapping[str, Any]]) -> str:
    """Select the deployable baseline with the highest AUROC, tie-broken by name."""

    candidates = []
    for name in DEPLOYABLE_BASELINES:
        auroc = metrics.get(name, {}).get("auroc")
        candidates.append((float(auroc) if auroc is not None else -1.0, name))
    return sorted(candidates, key=lambda item: (-item[0], item[1]))[0][1]


def delong_noninferiority(
    labels: Sequence[int],
    candidate_scores: Sequence[float],
    reference_scores: Sequence[float],
    *,
    margin: float,
) -> JsonDict:
    """Paired DeLong CI for candidate-minus-reference AUROC non-inferiority."""

    label_arr, cand_arr = checked_metric_arrays(labels, candidate_scores)
    _, ref_arr = checked_metric_arrays(labels, reference_scores)
    n_pos = int(np.sum(label_arr == 1))
    n_neg = int(np.sum(label_arr == 0))
    auc_candidate = compute_auroc(label_arr.tolist(), cand_arr.tolist())
    auc_reference = compute_auroc(label_arr.tolist(), ref_arr.tolist())
    if n_pos == 0 or n_neg == 0 or auc_candidate is None or auc_reference is None:
        return {
            "auc_candidate": None,
            "auc_reference": None,
            "auc_difference": None,
            "ci": [0.0, 0.0],
            "standard_error": 0.0,
            "noninferiority_margin_auroc": float(margin),
            "noninferiority_passed": False,
        }
    order = np.argsort(-label_arr)
    predictions = np.vstack([cand_arr, ref_arr])[:, order]
    aucs, covariance = fast_delong(predictions, n_pos)
    diff = float(aucs[0] - aucs[1])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    standard_error = math.sqrt(max(variance, 0.0))
    z_value = 1.959963984540054
    ci = [diff - z_value * standard_error, diff + z_value * standard_error]
    return {
        "auc_candidate": metric_float(float(aucs[0])),
        "auc_reference": metric_float(float(aucs[1])),
        "auc_difference": metric_float(diff),
        "ci": [metric_float(ci[0]), metric_float(ci[1])],
        "standard_error": metric_float(standard_error),
        "noninferiority_margin_auroc": float(margin),
        "noninferiority_passed": bool(ci[0] >= float(margin)),
    }


def fast_delong(predictions_sorted: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong covariance for two or more paired ROC curves."""

    m = int(label_1_count)
    n = predictions_sorted.shape[1] - m
    positive_examples = predictions_sorted[:, :m]
    negative_examples = predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]
    tx = np.vstack([compute_midrank(positive_examples[row]) for row in range(k)])
    ty = np.vstack([compute_midrank(negative_examples[row]) for row in range(k)])
    tz = np.vstack([compute_midrank(predictions_sorted[row]) for row in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = covariance_matrix(v01)
    sy = covariance_matrix(v10)
    return aucs, sx / m + sy / n


def compute_midrank(values: np.ndarray) -> np.ndarray:
    """Compute one-based midranks for DeLong covariance."""

    return average_ranks(np.asarray(values, dtype=np.float64))


def covariance_matrix(values: np.ndarray) -> np.ndarray:
    """Return a 2D covariance matrix even for tiny or constant inputs."""

    if values.shape[1] < 2:
        return np.zeros((values.shape[0], values.shape[0]), dtype=np.float64)
    covariance = np.cov(values)
    if np.ndim(covariance) == 0:
        return np.asarray([[float(covariance)]], dtype=np.float64)
    return np.asarray(covariance, dtype=np.float64)


def per_slice_metrics(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    threshold: float,
) -> JsonDict:
    """Report category and alignment slices, including single-class slices."""

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[f"category:{row.get('category_id') or 'unknown'}"].append(index)
        groups[f"instruction_alignment:{row.get('instruction_alignment') or 'unknown'}"].append(index)
    labels = labels_for(rows)
    result: JsonDict = {}
    for name, indices in sorted(groups.items()):
        group_labels = [labels[index] for index in indices]
        group_scores = [scores[index] for index in indices]
        row = metric_summary(group_labels, group_scores)
        row["f1_at_selected_threshold"] = metrics_at_threshold(
            group_labels,
            group_scores,
            threshold,
        )["f1"]
        result[name] = row
    return result


def garak_preliminary_metrics(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    threshold: float,
) -> JsonDict:
    """Report Garak/adaptive metrics separately because the split is one-class."""

    labels = labels_for(rows)
    summary = metric_summary(labels, scores)
    threshold_row = metrics_at_threshold(labels, scores, threshold)
    per_category: JsonDict = {}
    for category in sorted({str(row.get("category_id") or "unknown") for row in rows}):
        indices = [idx for idx, row in enumerate(rows) if row.get("category_id") == category]
        category_labels = [labels[idx] for idx in indices]
        category_scores = [scores[idx] for idx in indices]
        per_category[category] = metrics_at_threshold(category_labels, category_scores, threshold)
    summary.update(
        {
            "single_class_preliminary": len(set(labels)) < 2,
            "selected_threshold": metric_float(threshold),
            "detection_rate_at_selected_threshold": threshold_row["recall"],
            "per_category_detection": per_category,
        }
    )
    return summary


def shard_302_comparison(root: Path, full_corpus_auroc: float) -> JsonDict:
    """Compare the full-corpus score with the prior `.302` shard AUROC."""

    payload = read_json_object(root / EXP3265_REL_PATH)
    prior = safe_float(payload.get("shard_auroc"), SHARD_302_AUROC_FALLBACK)
    return {
        "prior_shard_auroc": metric_float(prior),
        "prior_shard_source": EXP3265_REL_PATH.as_posix(),
        "prior_shard_ready": payload.get("kan_train_eval_shard_ready") is True,
        "full_minus_prior_shard_auroc": metric_float(float(full_corpus_auroc) - prior),
    }


def source_artifacts(root: Path, exp3272: Mapping[str, Any]) -> JsonDict:
    """Record checksums for upstream artifacts and frozen split files."""

    paths = [EXP3272_REL_PATH, EXP3265_REL_PATH, *SPLIT_REL_PATHS.values()]
    return {
        rel.as_posix(): {
            "exists": (root / rel).is_file(),
            "sha256": sha256_file(root / rel) if (root / rel).is_file() else "",
        }
        for rel in paths
    } | {
        "exp3272_reproducibility_checksum": str(exp3272.get("reproducibility_checksum") or "")
    }


def file_checksums(root: Path, rel_paths: Sequence[str]) -> dict[str, str]:
    """Return SHA-256 checksums for paths that exist."""

    return {rel: sha256_file(root / rel) for rel in rel_paths if (root / rel).is_file()}


def output_path_list(output_path: str | Path) -> list[str]:
    """List the result artifact plus concrete downstream split artifacts."""

    return [path_as_artifact_string(output_path), *(rel.as_posix() for rel in SPLIT_REL_PATHS.values())]


def model_specs(checker: Any, *, n_epochs: int, lr: float) -> JsonDict:
    """Summarize the sidecar architecture without claiming exact authority."""

    return {
        "model_class": "PromptInjectionEnergyCheckerV3",
        "schema": "carnot.prompt_injection_kan.v3",
        "n_features": int(checker.n_features),
        "n_hidden": int(checker.n_hidden),
        "n_knots": int(checker._N_KNOTS),
        "degree": int(checker._DEGREE),
        "n_params": int(checker.n_params()),
        "n_epochs": int(n_epochs),
        "lr": float(lr),
        "sidecar_only": True,
    }


def model_checksum(checker: Any) -> str:
    """Hash the trained control points for artifact reproducibility."""

    digest = hashlib.sha256()
    for name in ("edge_ctrl", "output_ctrl"):
        arr = np.asarray(getattr(checker, name), dtype=np.float32)
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal verdict with explicit sidecar-only scope."""

    if artifact.get("v4_full_eval_ready") is True:
        return (
            "complete: v4_full_eval_ready=true; "
            f"full_corpus_auroc={float(artifact['full_corpus_auroc']):.6f}; "
            f"full_corpus_auprc={float(artifact['full_corpus_auprc']):.6f}; "
            f"delong_noninferiority_passed={str(artifact['delong_noninferiority_passed']).lower()}; "
            "sidecar_only=true"
        )
    return (
        "complete: v4_full_eval_ready=false; "
        f"blocked_reason={artifact.get('blocked_reason')}; sidecar_only=true"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3273 schema fields that downstream gates depend on."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover
    if not terminal_prefix_ok(str(artifact.get("honest_verdict") or "")):
        raise ValueError("honest_verdict must start with a terminal prefix")
    if artifact.get("sidecar_only") is not True:
        raise ValueError("sidecar_only must remain true")  # pragma: no cover
    for key in ("full_corpus_auroc", "full_corpus_auprc", "calibration_ece", "duration_s"):
        value = float(artifact[key])
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")  # pragma: no cover
    ci = artifact.get("delong_ci")
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError("delong_ci must be a two-element list")  # pragma: no cover


def terminal_prefix_ok(value: str) -> bool:
    """Return true when a verdict is terminal for conductor parsing."""

    return value.startswith(TERMINAL_PREFIXES)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the stable artifact payload, excluding wall-clock-only fields."""

    stable = json.loads(json.dumps(artifact, sort_keys=True, default=str))
    stable["reproducibility_checksum"] = ""
    stable["honest_verdict"] = ""
    stable["duration_s"] = 0.0
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def metric_float(value: float | int | np.floating[Any]) -> float:
    """Round metric floats to a stable JSON precision."""

    return round(float(value), 6)


def nullable_metric(value: float | None) -> float | None:
    """Round a metric when it is defined."""

    return None if value is None else metric_float(value)


def safe_float(value: Any, default: float) -> float:
    """Parse a float with a deterministic fallback."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def duration(start: float, end: float) -> float:
    """Return non-negative rounded duration."""

    return metric_float(max(0.0, float(end) - float(start)))


def path_as_artifact_string(path: str | Path) -> str:
    """Preserve relative artifact paths for downstream conductor tasks."""

    return Path(path).as_posix()


def resolve_output_path(root: Path, path: str | Path) -> Path:
    """Resolve relative output paths under the project root."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning an empty dict for absent or invalid files."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_jsonl(path: Path) -> list[JsonDict]:
    """Read JSONL rows and skip malformed non-object rows."""

    rows: list[JsonDict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def sha256_text(text: str) -> str:
    """Hash text for fallback leakage signatures."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a local file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:  # pragma: no cover
    artifact = run_experiment(project_root=REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
