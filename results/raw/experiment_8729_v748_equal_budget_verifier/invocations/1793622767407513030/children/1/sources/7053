from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import experiment_2546_ensemble_v7b as exp2546
from carnot.verify.adaptive_conformal_calibration import run_adaptive_conformal_calibration


SEEDS = (42, 123, 456, 789, 1337)
RANDOM_SEED = SEEDS[0]
MIN_BASELINE_AUROC = 0.970
OUTPUT_FILENAME = "experiment_2547_adaptive_conformal_v2.json"
BASELINE_FILENAME = "experiment_2546_ensemble_v7b.json"
PROMPT_TYPE_SHRINKAGE = 0.2
ACSE_ENTROPY_WEIGHT = 0.000000001

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix required.",
    "adaptive_conformal_auroc": (
        "Primary metric - documents whether prompt-adaptation improves beyond fixed calibration."
    ),
    "ensemble_v7b_baseline": "Comparison baseline from exp2546 for regression detection.",
    "prompt_type_distribution": (
        "Shows how prompts were classified - validates classifier is not degenerate "
        "(not all in one class)."
    ),
    "acse_component_used": "True if semantic entropy component was incorporated.",
    "n_seeds": "Must be >= 3.",
    "preconditions_checked": "Records which resources were verified.",
    "duration_s": "Wall-clock measurement.",
    "random_seed": "Set to 42.",
}


def _blocked_artifact(
    *,
    reason: str,
    duration_s: float,
    preconditions_checked: list[str],
    ensemble_v7b_baseline: float | None,
) -> dict[str, Any]:
    return {
        "honest_verdict": reason,
        "adaptive_conformal_auroc": None,
        "ensemble_v7b_baseline": ensemble_v7b_baseline,
        "prompt_type_distribution": {},
        "acse_component_used": False,
        "n_seeds": len(SEEDS),
        "preconditions_checked": preconditions_checked,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "acceptance_gates": {
            "adaptive_conformal_auroc >= ensemble_v7b_baseline": False,
        },
        "field_principles": FIELD_PRINCIPLES,
    }


def _write_artifact(results_dir: Path, deliverable: dict[str, Any]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / OUTPUT_FILENAME
    out_path.write_text(json.dumps(deliverable, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


def _load_exp2546_baseline(results_dir: Path) -> float | None:
    baseline_path = results_dir / BASELINE_FILENAME
    if not baseline_path.exists():
        return None
    data = exp2546.robust_load_json(baseline_path)
    value = data.get("ensemble_v7b_auroc")
    if value is None:
        return None
    return float(value)


def run_experiment(results_dir: Path = Path("results"), write: bool = True) -> dict[str, Any]:
    start_time = time.time()
    results_dir = Path(results_dir)
    preconditions_checked: list[str] = []

    ensemble_v7b_baseline = _load_exp2546_baseline(results_dir)
    if ensemble_v7b_baseline is None:
        preconditions_checked.append("exp2546_baseline_missing")
        duration_s = time.time() - start_time
        deliverable = _blocked_artifact(
            reason="blocked_ensemble_v7b_below_threshold",
            duration_s=duration_s,
            preconditions_checked=preconditions_checked,
            ensemble_v7b_baseline=None,
        )
        if write:
            _write_artifact(results_dir, deliverable)
        return deliverable

    preconditions_checked.append("exp2546_baseline_loaded")
    if ensemble_v7b_baseline < MIN_BASELINE_AUROC:
        preconditions_checked.append("ensemble_v7b_auroc_below_0.970")
        duration_s = time.time() - start_time
        deliverable = _blocked_artifact(
            reason="blocked_ensemble_v7b_below_threshold",
            duration_s=duration_s,
            preconditions_checked=preconditions_checked,
            ensemble_v7b_baseline=ensemble_v7b_baseline,
        )
        if write:
            _write_artifact(results_dir, deliverable)
        return deliverable
    preconditions_checked.append("ensemble_v7b_auroc>=0.970")

    missing_files = [
        path.name for path in exp2546._required_score_files(results_dir) if not path.exists()
    ]
    if missing_files:
        preconditions_checked.append("score_files_missing")
        duration_s = time.time() - start_time
        deliverable = _blocked_artifact(
            reason=f"blocked_missing_score_files: {','.join(missing_files)}",
            duration_s=duration_s,
            preconditions_checked=preconditions_checked,
            ensemble_v7b_baseline=ensemble_v7b_baseline,
        )
        if write:
            _write_artifact(results_dir, deliverable)
        return deliverable
    preconditions_checked.append("score_files_present")

    rows = exp2546.load_manifest_rows(results_dir)
    labels = np.asarray(
        [exp2546.normalize_label(row["correctness_label"]) for row in rows], dtype=int
    )
    score_groups = exp2546.build_score_groups(results_dir, rows)
    preconditions_checked.append("exp2546_score_groups_loaded")

    prompts = [str(row.get("prompt", "")) for row in rows]
    top_logprobs_by_row = [row.get("top_logprobs", []) for row in rows]
    calibration = run_adaptive_conformal_calibration(
        score_groups=score_groups,
        labels=labels,
        prompts=prompts,
        top_logprobs_by_row=top_logprobs_by_row,
        seeds=SEEDS,
        group_order=exp2546.GROUP_ORDER,
        prompt_type_shrinkage=PROMPT_TYPE_SHRINKAGE,
        acse_entropy_weight=ACSE_ENTROPY_WEIGHT,
    )
    preconditions_checked.append("prompt_type_classifier_applied")
    preconditions_checked.append("acse_entropy_proxy_applied")

    adaptive_conformal_auroc = calibration["adaptive_conformal_auroc"]
    non_regression_passed = bool(adaptive_conformal_auroc >= ensemble_v7b_baseline)
    if non_regression_passed:
        honest_verdict = f"complete: {adaptive_conformal_auroc:.4f} no regression vs v7b"
    else:
        honest_verdict = (
            f"terminal: {adaptive_conformal_auroc:.4f} below v7b baseline "
            f"{ensemble_v7b_baseline:.4f}"
        )

    duration_s = time.time() - start_time
    deliverable = {
        "honest_verdict": honest_verdict,
        "adaptive_conformal_auroc": adaptive_conformal_auroc,
        "adaptive_conformal_auroc_std": calibration["adaptive_conformal_auroc_std"],
        "ensemble_v7b_baseline": ensemble_v7b_baseline,
        "adaptive_vs_ensemble_v7b_delta": float(adaptive_conformal_auroc - ensemble_v7b_baseline),
        "prompt_type_distribution": calibration["prompt_type_distribution"],
        "acse_component_used": True,
        "n_seeds": len(SEEDS),
        "preconditions_checked": preconditions_checked,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "prompt_type_shrinkage": PROMPT_TYPE_SHRINKAGE,
        "acse_entropy_weight": ACSE_ENTROPY_WEIGHT,
        "acse_entropy_mean": calibration["acse_entropy_mean"],
        "acse_entropy_std": calibration["acse_entropy_std"],
        "group_conditional_baseline_auroc": calibration["group_conditional_baseline_auroc"],
        "group_conditional_baseline_auroc_std": calibration["group_conditional_baseline_auroc_std"],
        "acceptance_gates": {
            "adaptive_conformal_auroc >= ensemble_v7b_baseline": non_regression_passed,
        },
        "field_principles": FIELD_PRINCIPLES,
        "source_experiment": BASELINE_FILENAME,
        "results_by_seed": calibration["results_by_seed"],
    }

    if write:
        _write_artifact(results_dir, deliverable)
    return deliverable


if __name__ == "__main__":
    run_experiment()
