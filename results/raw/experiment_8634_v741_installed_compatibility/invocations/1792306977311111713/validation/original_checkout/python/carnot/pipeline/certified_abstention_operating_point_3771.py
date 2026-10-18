"""Exp 3771 Certified abstention operating point.

Spec: REQ-SPOE-3771, SCENARIO-SPOE-3771.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from carnot.pipeline import second_pair_detector as spd
from carnot.pipeline.risk_coverage_abstention_3718 import (
    load_fover_abstention_examples,
    AbstentionExample,
    risk_coverage_summary,
)

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3771_certified_abstention_operating_point.json")
RANDOM_SEED = 3771
TARGET_RISK = 0.05
DELTA = 0.05
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: reuses cached discriminator scores, no live model)."
)

VERDICT_SUCCESS = "complete: certified_abstention_point"
VERDICT_FAILURE = "complete: no_usable_certified_abstention_point_found"
VERDICT_BLOCKED = "complete: blocked_fover_perstep_scores_unavailable"
TERMINAL_VERDICTS = (
    VERDICT_SUCCESS,
    VERDICT_FAILURE,
    VERDICT_BLOCKED,
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "aurc",
    "selected_threshold",
    "risk_target",
    "coverage_at_operating_point",
    "certified_risk_bound",
    "certification_method",
    "n_calibration",
    "n_test",
    "usable_operating_point_exists",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; blocked_<resource> if a precondition failed.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "aurc": "Area under the risk-coverage curve -- the deployment-facing summary of the abstention gate's quality.",
    "selected_threshold": "The chosen abstention threshold on the discriminator score -- the deployable operating point.",
    "risk_target": "The fixed selective-risk target the threshold achieves -- the product spec the operating point meets.",
    "coverage_at_operating_point": "Fraction the verifier confidently judges (vs abstains) at the selected threshold -- the coverage half of the trade-off.",
    "certified_risk_bound": "The certified upper bound on selective risk (with delta, n) -- the GUARANTEE that distinguishes this from a mere characterization.",
    "certification_method": "split-conformal / PAC-Bayes + its assumptions stated -- so a reviewer can audit the certificate's validity.",
    "n_calibration": "Calibration sample size -- the certificate's strength scales with it; sample-size rigor.",
    "n_test": "Held-out test size for the risk estimate (n>=100 for a CLT-valid claim).",
    "usable_operating_point_exists": "BARE bool -- whether a deployable point at the target risk + non-trivial coverage exists; an honest false is a valid finding, not a failure to manufacture.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3771 artifact from cached FoVer per-step rows."""

    root_path = Path(root)
    examples, corpus_status = load_fover_abstention_examples(root_path)
    preconditions = [
        {
            "resource": "fover_perstep_scores",
            "available": bool(len(examples) >= 200),
            "detail": corpus_status.get("math", {}).get("path"),
            "n_examples": len(examples),
        },
    ]
    artifact = build_artifact_from_examples(
        examples,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
        min_examples=200,
        extra={
            "corpus_status": corpus_status,
            "preconditions_checked": preconditions,
            "output_path": str(_repo_path(root_path, OUTPUT_REL_PATH)),
        },
    )
    return artifact


def build_artifact_from_examples(
    examples: Sequence[AbstentionExample],
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    min_examples: int = 200,
    adversarial_verify_clean: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Assemble the certified abstention artifact from synthetic or cached examples."""

    start = time.perf_counter() if started_s is None else float(started_s)
    clean = _clean_examples(examples)
    blocked = len(clean) < int(min_examples)
    if blocked:
        finished = time.perf_counter() if now_s is None else float(now_s)
        artifact = _base_artifact(
            verdict=VERDICT_BLOCKED,
            duration_s=_round(max(0.0, finished - start)),
            tests_run=tests_run,
        )
        artifact.update(_empty_measurements())
        artifact.update(dict(extra or {}))
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    labels = [example.label for example in clean]
    energy_scores = [example.energy_score for example in clean]

    result = split_conformal_certification(labels, energy_scores)

    finished = time.perf_counter() if now_s is None else float(now_s)

    usable = result["usable_operating_point_exists"]
    if usable:
        verdict = f"complete: certified_abstention_point_threshold_{result['selected_threshold']}_coverage_{result['coverage_at_operating_point']}_at_risk_{result['risk_target']}_certified_split_conformal_delta_{DELTA}_n{result['n_calibration']}"
    else:
        verdict = VERDICT_FAILURE

    artifact = _base_artifact(
        verdict=verdict,
        duration_s=_round(max(0.0, finished - start)),
        tests_run=tests_run,
    )
    artifact.update(result)
    artifact.update(dict(extra or {}))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def split_conformal_certification(
    labels: Sequence[int],
    scores: Sequence[float],
) -> JsonDict:
    """Split data, find threshold on calibration, evaluate on test."""
    
    label_arr = np.asarray(labels, dtype=np.int64)
    score_arr = np.asarray(scores, dtype=np.float64)
    n = len(label_arr)
    
    # 40/60 random split for exchangeability and to avoid exact n_cal == n_test
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(n)
    mid = int(0.4 * n)
    cal_idx = indices[:mid]
    test_idx = indices[mid:]
    
    cal_labels = label_arr[cal_idx]
    cal_scores = score_arr[cal_idx]
    test_labels = label_arr[test_idx]
    test_scores = score_arr[test_idx]
    
    n_cal = len(cal_labels)
    n_test = len(test_labels)
    
    # 1. Compute full AURC on all data
    summary = risk_coverage_summary(labels, scores)
    aurc = summary["aurc"]
    
    # 2. Find threshold on calibration
    cal_order = np.argsort(cal_scores, kind="mergesort")
    ordered_cal_scores = cal_scores[cal_order]
    ordered_cal_labels = cal_labels[cal_order]
    
    selected_threshold = None
    
    prior_errors = 0
    best_threshold = ordered_cal_scores[0]
    found = False
    
    # Sweep over calibration
    for i in range(n_cal):
        prior_errors += ordered_cal_labels[i]
        count = i + 1
        emp_risk = prior_errors / count
        # Anticipate Hoeffding bound for selection: empirical risk + Hoeffding margin <= TARGET_RISK
        # This is a heuristic to pick a threshold that is likely to pass the test bound.
        bound = emp_risk + math.sqrt(-math.log(DELTA) / (2 * count))
        
        # we want the largest coverage (count) where bound <= TARGET_RISK
        # since risk generally increases with coverage, we update our best as long as it's valid
        if bound <= TARGET_RISK:
            best_threshold = ordered_cal_scores[i]
            found = True
            
    # If no threshold satisfies the heuristic bound, we still need to pick one to evaluate.
    selected_threshold = float(best_threshold) if found else float(ordered_cal_scores[0])
    
    # 3. Evaluate on test set
    test_kept_mask = test_scores <= selected_threshold
    test_kept_count = int(np.sum(test_kept_mask))
    test_errors = int(np.sum(test_labels[test_kept_mask]))
    
    if test_kept_count > 0:
        test_emp_risk = test_errors / test_kept_count
        # Hoeffding upper bound on the true risk for the test set
        test_bound = test_emp_risk + math.sqrt(-math.log(DELTA) / (2 * test_kept_count))
    else:
        test_emp_risk = 1.0
        test_bound = 1.0
        
    usable = found and (test_kept_count > 0) and (test_bound <= TARGET_RISK)
    
    coverage = float(test_kept_count) / n_test

    return {
        "aurc": float(aurc) if aurc is not None else 0.0,
        "selected_threshold": _round(selected_threshold),
        "risk_target": TARGET_RISK,
        "coverage_at_operating_point": _round(coverage),
        "certified_risk_bound": _round(test_bound),
        "certification_method": f"split-conformal (Hoeffding upper bound, assumes exchangeability, delta={DELTA})",
        "n_calibration": n_cal,
        "n_test": n_test,
        "usable_operating_point_exists": bool(usable),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3771 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("honest_verdict") not in TERMINAL_VERDICTS and not str(artifact.get("honest_verdict", "")).startswith("complete: certified_abstention_point"):
        raise ValueError("honest_verdict is not an accepted Exp 3771 terminal verdict")
    if type(artifact.get("usable_operating_point_exists")) is not bool:
        raise ValueError("usable_operating_point_exists must be a bare top-level bool")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")
    n_cal = artifact.get("n_calibration")
    n_test = artifact.get("n_test")
    if artifact.get("honest_verdict") != VERDICT_BLOCKED:
        if not isinstance(n_cal, int) or n_cal < 100:
            raise ValueError("n_calibration must be an integer >= 100")
        if not isinstance(n_test, int) or n_test < 100:
            raise ValueError("n_test must be an integer >= 100")


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic Exp 3771 artifact fields."""

    payload = {
        "aurc": artifact.get("aurc"),
        "selected_threshold": artifact.get("selected_threshold"),
        "coverage_at_operating_point": artifact.get("coverage_at_operating_point"),
        "certified_risk_bound": artifact.get("certified_risk_bound"),
        "usable_operating_point_exists": artifact.get("usable_operating_point_exists"),
        "n_calibration": artifact.get("n_calibration"),
        "n_test": artifact.get("n_test"),
        "random_seed": RANDOM_SEED,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3771 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(root_path, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    
    # We can run adversarial_verify here if needed, but it will be invoked from the script.
    validate_artifact(artifact)
    return output


def _base_artifact(
    *,
    verdict: str,
    duration_s: float,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    return {
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": list(tests_run or []),
    }


def _empty_measurements() -> JsonDict:
    return {
        "aurc": 0.0,
        "selected_threshold": 0.0,
        "risk_target": TARGET_RISK,
        "coverage_at_operating_point": 0.0,
        "certified_risk_bound": 0.0,
        "certification_method": "none",
        "n_calibration": 0,
        "n_test": 0,
        "usable_operating_point_exists": False,
    }


def _clean_examples(examples: Sequence[AbstentionExample]) -> list[AbstentionExample]:
    clean = []
    for example in examples:
        energy = float(example.energy_score)
        if math.isfinite(energy):
            clean.append(
                AbstentionExample(
                    label=1 if int(example.label) else 0,
                    energy_score=energy,
                    baseline_score=float(example.baseline_score) if math.isfinite(float(example.baseline_score)) else 0.0,
                    example_id=str(example.example_id),
                )
            )
    return clean


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _round(value: float) -> float:
    if not math.isfinite(float(value)):
        return float(value)
    return round(float(value), 6)


__all__ = [
    "OUTPUT_REL_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_artifact",
    "build_artifact_from_examples",
    "split_conformal_certification",
    "validate_artifact",
    "write_artifact",
]
