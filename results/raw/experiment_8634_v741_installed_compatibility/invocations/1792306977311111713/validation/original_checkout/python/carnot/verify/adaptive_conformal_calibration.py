from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from carnot.verify.group_conditional_calibration import calibrate_group_scores, fisher_combine

PROMPT_TYPES: tuple[str, ...] = ("factual", "reasoning", "creative", "code")


def prompt_type_classifier(prompt: str) -> str:
    """Classify a prompt into the four calibration buckets used by Exp 2547.

    The classifier is intentionally lexical and deterministic because the
    calibration layer must not depend on another LLM call. Domain cues take
    priority over output-format cues: an arithmetic prompt that asks for a short
    sentence is still a reasoning task, while a pure writing prompt is creative.
    """

    text = prompt.lower()
    code_terms = (
        "code",
        "python",
        "function",
        "script",
        "program",
        "debug",
        "sql",
        "javascript",
        "rust",
        "compiler",
        "class ",
        "parse json",
    )
    if any(term in text for term in code_terms):
        return "code"

    primary_reasoning_terms = (
        "constraint",
        "satisfies",
        "calculate",
        "compute",
        "what is the final answer",
        "how many",
        "marble",
        "riders",
        "pencils",
    )
    if any(term in text for term in primary_reasoning_terms):
        return "reasoning"

    factual_terms = (
        "verify claim",
        "return 1 if true",
        "true or false",
        "is true",
        "is false",
        "who ",
        "what ",
        "when ",
        "where ",
        "telemetry obedience",
        "return exactly this integer",
        "answer ",
    )
    if any(term in text for term in factual_terms):
        return "factual"

    arithmetic_terms = (
        " + ",
        " - ",
        " * ",
        " / ",
    )
    if any(term in text for term in arithmetic_terms):
        return "reasoning"

    creative_terms = (
        "write",
        "story",
        "poem",
        "haiku",
        "creative",
        "draft",
        "tagline",
        "one short sentence",
    )
    if any(term in text for term in creative_terms):
        return "creative"

    return "factual"


def prompt_type_distribution(prompts: Sequence[str]) -> dict[str, int]:
    """Return prompt-type counts with all four buckets present in the mapping."""

    counts = Counter(prompt_type_classifier(prompt) for prompt in prompts)
    return {prompt_type: int(counts.get(prompt_type, 0)) for prompt_type in PROMPT_TYPES}


def compute_acse_entropy_proxy(
    top_logprobs: Sequence[Mapping[str, float]],
    verifier_scores: Sequence[float],
    top_k: int = 5,
) -> float:
    """Compute a lightweight ACSE-style semantic entropy proxy.

    ACSE uses semantic dispersion over multiple generations. The exp2547 corpus
    has fixed verifier rows rather than fresh generations, so this proxy uses
    two local uncertainty signals available for every row: the variance of the
    top-k token logprob alternatives and the variance across verifier scores.
    """

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    token_position_variances: list[float] = []
    for position in top_logprobs:
        top_values = sorted((float(value) for value in position.values()), reverse=True)[:top_k]
        if len(top_values) > 1:
            token_position_variances.append(float(np.var(top_values)))

    logprob_variance = float(np.mean(token_position_variances)) if token_position_variances else 0.0
    verifier_array = np.asarray([float(score) for score in verifier_scores], dtype=np.float64)
    verifier_variance = float(np.var(verifier_array)) if verifier_array.size > 1 else 0.0
    return logprob_variance + verifier_variance


def _normalize_with_train_range(train_values: np.ndarray, test_values: np.ndarray) -> np.ndarray:
    train_min = float(np.min(train_values))
    train_max = float(np.max(train_values))
    if train_max <= train_min:
        return np.zeros_like(test_values, dtype=np.float64)
    return np.clip((test_values - train_min) / (train_max - train_min), 0.0, 1.0)


def _class_count_is_calibratable(labels: np.ndarray) -> bool:
    return labels.size >= 2 and np.unique(labels).size == 2


def run_adaptive_conformal_calibration(
    *,
    score_groups: Mapping[str, np.ndarray],
    labels: np.ndarray,
    prompts: Sequence[str],
    top_logprobs_by_row: Sequence[Sequence[Mapping[str, float]]],
    seeds: Sequence[int],
    group_order: Sequence[str],
    prompt_type_shrinkage: float = 0.2,
    acse_entropy_weight: float = 0.01,
    min_prompt_type_train: int = 2,
    test_size: float = 0.3,
) -> dict[str, Any]:
    """Run prompt-adaptive conformal calibration and return AUROC statistics."""

    if not 0.0 <= prompt_type_shrinkage <= 1.0:
        raise ValueError("prompt_type_shrinkage must be in [0, 1]")
    if acse_entropy_weight < 0.0:
        raise ValueError("acse_entropy_weight must be non-negative")

    labels = np.asarray(labels, dtype=int)
    prompt_types = np.asarray([prompt_type_classifier(prompt) for prompt in prompts], dtype=object)
    score_matrix = np.column_stack([score_groups[group_name] for group_name in group_order])
    acse_entropy = np.asarray(
        [
            compute_acse_entropy_proxy(top_logprobs, score_matrix[row_idx])
            for row_idx, top_logprobs in enumerate(top_logprobs_by_row)
        ],
        dtype=np.float64,
    )

    seed_results: list[dict[str, Any]] = []
    adaptive_aurocs: list[float] = []
    baseline_aurocs: list[float] = []

    for seed in seeds:
        idx = np.arange(len(labels))
        idx_train, idx_test, y_train, y_test = train_test_split(
            idx,
            labels,
            test_size=test_size,
            random_state=int(seed),
            stratify=labels,
        )

        global_calibrated_by_group: dict[str, np.ndarray] = {}
        adaptive_calibrated_by_group: dict[str, np.ndarray] = {}
        local_fit_count = 0
        fallback_count = 0

        for group_name in group_order:
            group_scores = score_groups[group_name]
            global_scores = calibrate_group_scores(
                group_scores[idx_train],
                group_scores[idx_test],
                y_train,
            )
            adaptive_scores = np.array(global_scores, copy=True)

            for prompt_type in PROMPT_TYPES:
                train_mask = prompt_types[idx_train] == prompt_type
                test_mask = prompt_types[idx_test] == prompt_type
                if not np.any(test_mask):
                    continue

                if int(
                    np.sum(train_mask)
                ) >= min_prompt_type_train and _class_count_is_calibratable(y_train[train_mask]):
                    local_scores = calibrate_group_scores(
                        group_scores[idx_train[train_mask]],
                        group_scores[idx_test[test_mask]],
                        y_train[train_mask],
                    )
                    adaptive_scores[test_mask] = (1.0 - prompt_type_shrinkage) * global_scores[
                        test_mask
                    ] + prompt_type_shrinkage * local_scores
                    local_fit_count += 1
                else:
                    fallback_count += 1

            global_calibrated_by_group[group_name] = global_scores
            adaptive_calibrated_by_group[group_name] = adaptive_scores

        baseline_risk = fisher_combine(
            np.column_stack([1.0 - global_calibrated_by_group[name] for name in group_order])
        )
        prompt_adaptive_risk = fisher_combine(
            np.column_stack([1.0 - adaptive_calibrated_by_group[name] for name in group_order])
        )
        entropy_weight = 1.0 + acse_entropy_weight * _normalize_with_train_range(
            acse_entropy[idx_train],
            acse_entropy[idx_test],
        )
        adaptive_risk = prompt_adaptive_risk * entropy_weight

        baseline_auroc = float(roc_auc_score(y_test, baseline_risk))
        adaptive_auroc = float(roc_auc_score(y_test, adaptive_risk))
        baseline_aurocs.append(baseline_auroc)
        adaptive_aurocs.append(adaptive_auroc)

        row: dict[str, Any] = {
            "seed": int(seed),
            "test_auroc_group_cond_baseline": baseline_auroc,
            "test_auroc_adaptive_conformal": adaptive_auroc,
            "mean_acse_entropy_test": float(np.mean(acse_entropy[idx_test])),
            "prompt_type_local_fit_count": int(local_fit_count),
            "prompt_type_fallback_count": int(fallback_count),
            "prompt_type_counts_test": prompt_type_distribution(
                [str(prompts[row_idx]) for row_idx in idx_test]
            ),
        }
        for group_name in group_order:
            row[f"mean_adaptive_cal_{group_name}"] = float(
                np.mean(adaptive_calibrated_by_group[group_name])
            )
        seed_results.append(row)

    return {
        "adaptive_conformal_auroc": float(np.mean(adaptive_aurocs)),
        "adaptive_conformal_auroc_std": float(np.std(adaptive_aurocs)),
        "group_conditional_baseline_auroc": float(np.mean(baseline_aurocs)),
        "group_conditional_baseline_auroc_std": float(np.std(baseline_aurocs)),
        "prompt_type_distribution": prompt_type_distribution(prompts),
        "acse_entropy_mean": float(np.mean(acse_entropy)),
        "acse_entropy_std": float(np.std(acse_entropy)),
        "results_by_seed": seed_results,
    }
