"""Exp 3684 product value rebaseline against self-certainty.

Spec: REQ-SPOE-3684, SCENARIO-SPOE-3684.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
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


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3684_product_value_vs_self_certainty.json")
RANDOM_SEED = 3684
BOOTSTRAP_SEEDS = (3684, 3685, 3686)
DEFAULT_N_BOOTSTRAP = 200
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates (principle: scores cached corpora; no LLM load)."
)

VERDICT_ROBUST = "complete: ensemble_adds_value_over_self_certainty_product_value_robust"
VERDICT_COLLAPSED = "complete: product_value_collapses_vs_self_certainty_claim_narrowed"
VERDICT_BLOCKED = "complete: blocked_no_labeled_corpus_for_rebaseline"
TERMINAL_VERDICTS = (VERDICT_ROBUST, VERDICT_COLLAPSED, VERDICT_BLOCKED)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "self_certainty_auroc_per_domain",
    "plain_confidence_auroc_per_domain",
    "fused_ensemble_self_certainty_auroc_per_domain",
    "ensemble_minus_self_certainty_delta_ci_per_domain",
    "self_certainty_implementation",
    "ensemble_adds_value_over_self_certainty",
    "n_examples_per_domain",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "self_certainty_auroc_per_domain": (
        "The stronger free baseline AUROC per domain (math, code) -- the new bar."
    ),
    "plain_confidence_auroc_per_domain": (
        "The original (weaker) baseline -- shows whether the prior claim leaned on a weak comparator."
    ),
    "fused_ensemble_self_certainty_auroc_per_domain": (
        "Ensemble + self-certainty fused AUROC per domain -- the product number under the stronger baseline."
    ),
    "ensemble_minus_self_certainty_delta_ci_per_domain": (
        "Paired delta + CI of fused vs self-certainty-alone per domain -- the additive value over the stronger baseline."
    ),
    "self_certainty_implementation": (
        "Honest description of the self-certainty implementation or the disclosed proxy + its gap (verifier authenticity)."
    ),
    "ensemble_adds_value_over_self_certainty": (
        "BARE bool. True iff on >=1 domain the fused detector beats self-certainty-alone with the delta CI excluding 0 -- the robust-product-value verdict. STORE AS BARE true/false."
    ),
    "n_examples_per_domain": "Sample-size rigor per domain.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}

_SPD_MODULE_NAME = "_carnot_exp3684_second_pair_detector"


def _load_second_pair_detector() -> Any:
    module = sys.modules.get(_SPD_MODULE_NAME)
    if module is not None:
        return module
    path = Path(__file__).with_name("second_pair_detector.py")
    spec = importlib.util.spec_from_file_location(_SPD_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load second_pair_detector from {path}")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[_SPD_MODULE_NAME] = loaded
    spec.loader.exec_module(loaded)
    return loaded


spd = _load_second_pair_detector()


@dataclass(frozen=True)
class RebaselineExample:
    """One labeled cached candidate row for the self-certainty rebaseline."""

    domain: str
    label: int
    ensemble_energy: float
    confidence_error: float
    self_certainty_error: float
    example_id: str = ""


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3684 artifact from cached FoVer and balanced code rows."""

    root_path = Path(root)
    examples, corpus_status = load_rebaseline_examples(root_path)
    artifact = build_artifact_from_examples(
        examples,
        started_s=started_s,
        now_s=now_s,
        seeds=seeds,
        n_bootstrap=n_bootstrap,
        tests_run=tests_run,
    )
    artifact["corpus_status"] = corpus_status
    artifact["output_path"] = str(_repo_path(root_path, OUTPUT_REL_PATH))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def build_artifact_from_examples(
    examples: Sequence[RebaselineExample],
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Evaluate whether ensemble fusion adds value beyond self-certainty."""

    start = time.perf_counter() if started_s is None else float(started_s)
    clean = _clean_examples(examples)
    train, holdout = spd.stratified_train_holdout(clean, seed=RANDOM_SEED)
    if not _has_both_classes(train) or not any(
        _has_both_classes(group) for group in _by_domain(holdout).values()
    ):
        finished = time.perf_counter() if now_s is None else float(now_s)
        artifact = _base_artifact(
            verdict=VERDICT_BLOCKED,
            ensemble_adds_value=False,
            duration_s=_round(max(0.0, finished - start)),
            tests_run=tests_run,
        )
        artifact.update(
            {
                "self_certainty_auroc_per_domain": {},
                "plain_confidence_auroc_per_domain": {},
                "fused_ensemble_self_certainty_auroc_per_domain": {},
                "ensemble_alone_auroc_per_domain": {},
                "ensemble_minus_self_certainty_delta_ci_per_domain": {},
                "recall_at_fixed_fpr_table": {},
                "prior_confidence_claim_assessment_per_domain": {},
                "n_examples_per_domain": _n_examples_per_domain(clean),
                "heldout_examples_per_domain": _n_examples_per_domain(holdout),
            }
        )
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        validate_artifact(artifact)
        return artifact

    detector = spd.CalibratedFusedDetector().fit(_as_spd_examples(train, use_self_certainty=True))
    self_auroc: JsonDict = {}
    confidence_auroc: JsonDict = {}
    fused_auroc: JsonDict = {}
    ensemble_auroc: JsonDict = {}
    delta_ci: JsonDict = {}
    recall_table: JsonDict = {}
    prior_assessment: JsonDict = {}
    material_wins: dict[str, bool] = {}
    for domain, domain_examples in sorted(_by_domain(holdout).items()):
        if not _has_both_classes(domain_examples):
            continue
        labels = [example.label for example in domain_examples]
        fused_scores = detector.predict_proba(_as_spd_examples(domain_examples, use_self_certainty=True))
        self_scores = [example.self_certainty_error for example in domain_examples]
        confidence_scores = [example.confidence_error for example in domain_examples]
        ensemble_scores = [example.ensemble_energy for example in domain_examples]
        self_auroc[domain] = auroc_metric(
            labels,
            self_scores,
            seeds=seeds,
            n_bootstrap=n_bootstrap,
        )
        confidence_auroc[domain] = auroc_metric(
            labels,
            confidence_scores,
            seeds=seeds,
            n_bootstrap=n_bootstrap,
        )
        fused_auroc[domain] = auroc_metric(
            labels,
            fused_scores,
            seeds=seeds,
            n_bootstrap=n_bootstrap,
        )
        ensemble_auroc[domain] = auroc_metric(
            labels,
            ensemble_scores,
            seeds=seeds,
            n_bootstrap=n_bootstrap,
        )
        delta_ci[domain] = paired_auroc_delta_metric(
            labels,
            fused_scores,
            self_scores,
            seeds=seeds,
            n_bootstrap=n_bootstrap,
        )
        material_wins[domain] = delta_ci_excludes_zero_positive(delta_ci[domain])
        recall_table[domain] = recall_at_fixed_fpr_table(
            labels,
            fused_scores=fused_scores,
            self_certainty_scores=self_scores,
            ensemble_scores=ensemble_scores,
            confidence_scores=confidence_scores,
        )
        fused_point = _metric_point(fused_auroc[domain])
        confidence_point = _metric_point(confidence_auroc[domain])
        prior_assessment[domain] = {
            "beats_plain_confidence": bool(
                fused_point is not None
                and confidence_point is not None
                and fused_point > confidence_point
            ),
            "adds_over_self_certainty": material_wins[domain],
            "inflated_by_weak_baseline": bool(
                fused_point is not None
                and confidence_point is not None
                and fused_point > confidence_point
                and not material_wins[domain]
            ),
        }

    ensemble_adds_value = any(material_wins.values())
    verdict = VERDICT_ROBUST if ensemble_adds_value else VERDICT_COLLAPSED
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact = _base_artifact(
        verdict=verdict,
        ensemble_adds_value=ensemble_adds_value,
        duration_s=_round(max(0.0, finished - start)),
        tests_run=tests_run,
    )
    artifact.update(
        {
            "self_certainty_auroc_per_domain": self_auroc,
            "plain_confidence_auroc_per_domain": confidence_auroc,
            "fused_ensemble_self_certainty_auroc_per_domain": fused_auroc,
            "ensemble_alone_auroc_per_domain": ensemble_auroc,
            "ensemble_minus_self_certainty_delta_ci_per_domain": delta_ci,
            "recall_at_fixed_fpr_table": recall_table,
            "prior_confidence_claim_assessment_per_domain": prior_assessment,
            "material_win_per_domain": material_wins,
            "n_examples_per_domain": _n_examples_per_domain(clean),
            "heldout_examples_per_domain": _n_examples_per_domain(holdout),
            "calibrator": {
                "method": "logistic",
                "feature_names": ["ensemble_energy", "self_certainty_error"],
                "coef": [_round(value) for value in detector.coef_ or []],
                "intercept": _round(detector.intercept_ or 0.0),
            },
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def load_rebaseline_examples(
    root: Path | str,
    *,
    score_overrides: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
) -> tuple[list[RebaselineExample], JsonDict]:
    """Load detector rows and derive disclosed self-certainty proxy scores."""

    detector_examples, corpus_status = spd.load_cached_labeled_examples(
        Path(root),
        score_overrides=score_overrides,
        use_balanced_code_corpus=True,
    )
    proxy_scores = self_certainty_error_proxy_from_confidence_errors(
        [example.confidence_error for example in detector_examples]
    )
    examples = [
        RebaselineExample(
            domain=example.domain,
            label=example.label,
            ensemble_energy=example.ensemble_energy,
            confidence_error=example.confidence_error,
            self_certainty_error=proxy,
            example_id=example.example_id,
        )
        for example, proxy in zip(detector_examples, proxy_scores, strict=True)
    ]
    return examples, corpus_status


def token_distribution_self_certainty(
    probability_distributions: Sequence[Sequence[float]],
) -> float:
    """Return mean KL divergence from uniform for token distributions."""

    values: list[float] = []
    for distribution in probability_distributions:
        probs = np.asarray([float(value) for value in distribution], dtype=np.float64)
        probs = probs[np.isfinite(probs) & (probs > 0.0)]
        if len(probs) == 0:
            continue
        probs = probs / float(np.sum(probs))
        uniform = 1.0 / len(probs)
        values.append(float(np.sum(probs * np.log(probs / uniform))))
    return _round(float(np.mean(values)) if values else 0.0)


def self_certainty_error_proxy_from_confidence_errors(
    confidence_errors: Sequence[float],
) -> list[float]:
    """Return signed Bernoulli-uniform KL proxy scores oriented as error risk."""

    scores: list[float] = []
    for confidence_error in confidence_errors:
        p_error = min(1.0 - 1e-12, max(1e-12, float(confidence_error)))
        certainty = _bernoulli_uniform_kl(p_error)
        scores.append(_round((2.0 * p_error - 1.0) * certainty))
    return scores


def auroc_metric(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> JsonDict:
    """Return tie-aware AUROC plus deterministic bootstrap CI."""

    clean_labels, clean_scores = spd.finite_label_scores(labels, scores)
    if not clean_labels or len(set(clean_labels)) < 2:
        return empty_metric(seeds)
    label_arr = np.asarray(clean_labels, dtype=np.int64)
    score_arr = np.asarray(clean_scores, dtype=np.float64)
    point = spd.tie_aware_auroc(label_arr, score_arr)
    boot_values: list[float] = []
    seed_means: list[float] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        values: list[float] = []
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, len(label_arr), size=len(label_arr))
            if len(set(label_arr[idx].tolist())) < 2:
                continue
            value = spd.tie_aware_auroc(label_arr[idx], score_arr[idx])
            values.append(float(value))
            boot_values.append(float(value))
        seed_means.append(_round(float(np.mean(values))) if values else _round(point))
    if boot_values:
        ci_low, ci_high = np.percentile(np.asarray(boot_values, dtype=np.float64), [2.5, 97.5])
    else:
        ci_low = ci_high = point
    positives = int(np.sum(label_arr == 1))
    return {
        "point": _round(point),
        "ci95": [_round(float(ci_low)), _round(float(ci_high))],
        "n": len(clean_labels),
        "n_positive_errors": positives,
        "n_negative_correct": len(clean_labels) - positives,
        "bootstrap_seeds": [int(seed) for seed in seeds],
        "seed_mean_aurocs": seed_means,
    }


def paired_auroc_delta_metric(
    labels: Sequence[int],
    fused_scores: Sequence[float],
    baseline_scores: Sequence[float],
    *,
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> JsonDict:
    """Return paired AUROC delta for fused minus baseline scores."""

    clean_labels, clean_fused, clean_baseline = _finite_triplets(
        labels,
        fused_scores,
        baseline_scores,
    )
    if not clean_labels or len(set(clean_labels)) < 2:
        metric = empty_metric(seeds)
        metric["delta_ci_excludes_zero"] = False
        return metric
    label_arr = np.asarray(clean_labels, dtype=np.int64)
    fused_arr = np.asarray(clean_fused, dtype=np.float64)
    baseline_arr = np.asarray(clean_baseline, dtype=np.float64)
    point = spd.tie_aware_auroc(label_arr, fused_arr) - spd.tie_aware_auroc(
        label_arr,
        baseline_arr,
    )
    boot_values: list[float] = []
    seed_means: list[float] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        values: list[float] = []
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, len(label_arr), size=len(label_arr))
            if len(set(label_arr[idx].tolist())) < 2:
                continue
            delta = spd.tie_aware_auroc(label_arr[idx], fused_arr[idx]) - spd.tie_aware_auroc(
                label_arr[idx],
                baseline_arr[idx],
            )
            values.append(float(delta))
            boot_values.append(float(delta))
        seed_means.append(_round(float(np.mean(values))) if values else _round(point))
    if boot_values:
        ci_low, ci_high = np.percentile(np.asarray(boot_values, dtype=np.float64), [2.5, 97.5])
    else:
        ci_low = ci_high = point
    positives = int(np.sum(label_arr == 1))
    result = {
        "point": _round(point),
        "ci95": [_round(float(ci_low)), _round(float(ci_high))],
        "n": len(clean_labels),
        "n_positive_errors": positives,
        "n_negative_correct": len(clean_labels) - positives,
        "bootstrap_seeds": [int(seed) for seed in seeds],
        "seed_mean_deltas": seed_means,
    }
    result["delta_ci_excludes_zero"] = delta_ci_excludes_zero_positive(result)
    return result


def delta_ci_excludes_zero_positive(metric: Mapping[str, Any]) -> bool:
    """Return true only when the paired delta CI is strictly positive."""

    ci = metric.get("ci95")
    point = metric.get("point")
    if point is None or not isinstance(ci, Sequence) or len(ci) != 2:
        return False
    return bool(float(point) > 0.0 and float(ci[0]) > 0.0)


def recall_at_fixed_fpr_table(
    labels: Sequence[int],
    *,
    fused_scores: Sequence[float],
    self_certainty_scores: Sequence[float],
    ensemble_scores: Sequence[float],
    confidence_scores: Sequence[float],
) -> JsonDict:
    """Return fused, self-certainty, ensemble, and confidence fixed-FPR recall."""

    fused = spd.operating_points_at_fixed_fpr(labels, fused_scores)
    self_certainty = spd.operating_points_at_fixed_fpr(labels, self_certainty_scores)
    ensemble = spd.operating_points_at_fixed_fpr(labels, ensemble_scores)
    confidence = spd.operating_points_at_fixed_fpr(labels, confidence_scores)
    table: JsonDict = {}
    for key in fused:
        table[key] = {
            "fused_recall": fused[key]["recall"],
            "self_certainty_recall": self_certainty[key]["recall"],
            "ensemble_recall": ensemble[key]["recall"],
            "plain_confidence_recall": confidence[key]["recall"],
            "fused_actual_fpr": fused[key]["actual_fpr"],
            "self_certainty_actual_fpr": self_certainty[key]["actual_fpr"],
            "ensemble_actual_fpr": ensemble[key]["actual_fpr"],
            "plain_confidence_actual_fpr": confidence[key]["actual_fpr"],
            "fused_threshold": fused[key]["threshold"],
            "self_certainty_threshold": self_certainty[key]["threshold"],
            "ensemble_threshold": ensemble[key]["threshold"],
            "plain_confidence_threshold": confidence[key]["threshold"],
        }
    return table


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3684 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("honest_verdict") not in TERMINAL_VERDICTS:
        raise ValueError("honest_verdict is not an accepted Exp 3684 terminal verdict")
    if type(artifact.get("ensemble_adds_value_over_self_certainty")) is not bool:
        raise ValueError("ensemble_adds_value_over_self_certainty must be a bare top-level bool")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic Exp 3684 artifact fields."""

    payload = {
        "self_certainty_auroc_per_domain": artifact.get("self_certainty_auroc_per_domain"),
        "plain_confidence_auroc_per_domain": artifact.get("plain_confidence_auroc_per_domain"),
        "fused_ensemble_self_certainty_auroc_per_domain": artifact.get(
            "fused_ensemble_self_certainty_auroc_per_domain"
        ),
        "ensemble_alone_auroc_per_domain": artifact.get("ensemble_alone_auroc_per_domain"),
        "ensemble_minus_self_certainty_delta_ci_per_domain": artifact.get(
            "ensemble_minus_self_certainty_delta_ci_per_domain"
        ),
        "recall_at_fixed_fpr_table": artifact.get("recall_at_fixed_fpr_table"),
        "ensemble_adds_value_over_self_certainty": artifact.get(
            "ensemble_adds_value_over_self_certainty"
        ),
        "n_examples_per_domain": artifact.get("n_examples_per_domain"),
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
    """Build, validate, and persist the Exp 3684 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(root_path, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_artifact_from_examples(
    root: Path | str,
    *,
    output_path: Path | str,
    examples: Sequence[RebaselineExample],
    **kwargs: Any,
) -> Path:
    """Persist a synthetic or pre-measured Exp 3684 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact_from_examples(examples, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _base_artifact(
    *,
    verdict: str,
    ensemble_adds_value: bool,
    duration_s: float,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    return {
        "artifact": "experiment_3684_product_value_vs_self_certainty",
        "schema": "carnot.product_value_vs_self_certainty.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "self_certainty_implementation": _self_certainty_implementation(),
        "ensemble_adds_value_over_self_certainty": bool(ensemble_adds_value),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "self_certainty_auroc_per_domain present AND "
                "fused_ensemble_self_certainty_auroc_per_domain present AND "
                "ensemble_minus_self_certainty_delta_ci_per_domain present"
            ),
            "passed": bool(ensemble_adds_value),
            "principle": (
                "A robust-product verdict requires the stronger baseline, the "
                "fused number and the paired delta CI per domain -- comparing only "
                "against plain confidence would repeat the weak-baseline inflation this task checks for."
            ),
        },
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }


def _self_certainty_implementation() -> JsonDict:
    return {
        "paper_definition": (
            "Mean KL divergence between each predicted token distribution and the "
            "uniform distribution over that token's support; higher means a more peaked model distribution."
        ),
        "implemented_exact_helper": "token_distribution_self_certainty(probability_distributions)",
        "cached_corpus_signal_available": (
            "FoVer v4 exposes scalar confidence only; balanced Exp3658 code rows "
            "expose no token logits/probabilities and reuse the cached confidence/self-consistency baseline."
        ),
        "proxy_used_for_rebaseline": "signed_bernoulli_uniform_kl_from_confidence_error",
        "proxy_formula": (
            "p_error=confidence_error; KL=p_error*log(2*p_error)+(1-p_error)*log(2*(1-p_error)); "
            "self_certainty_error=(2*p_error-1)*KL so larger scores mean higher error risk."
        ),
        "proxy_disclosure_required": True,
        "gap": (
            "This is a scalar-confidence proxy, not true arXiv:2502.18581 token-distribution self-certainty; "
            "it is disclosed and kept separate from plain_confidence_auroc_per_domain."
        ),
    }


def empty_metric(seeds: Sequence[int]) -> JsonDict:
    """Return an empty metric bundle for blocked or one-class inputs."""

    return {
        "point": None,
        "ci95": None,
        "n": 0,
        "n_positive_errors": 0,
        "n_negative_correct": 0,
        "bootstrap_seeds": [int(seed) for seed in seeds],
        "seed_mean_aurocs": [],
    }


def _as_spd_examples(
    examples: Sequence[RebaselineExample],
    *,
    use_self_certainty: bool,
) -> list[spd.LabeledDetectorExample]:
    return [
        spd.LabeledDetectorExample(
            domain=example.domain,
            label=example.label,
            ensemble_energy=example.ensemble_energy,
            confidence_error=(
                example.self_certainty_error if use_self_certainty else example.confidence_error
            ),
            example_id=example.example_id,
        )
        for example in examples
    ]


def _clean_examples(examples: Sequence[RebaselineExample]) -> list[RebaselineExample]:
    clean: list[RebaselineExample] = []
    for example in examples:
        if all(
            math.isfinite(float(value))
            for value in (
                example.ensemble_energy,
                example.confidence_error,
                example.self_certainty_error,
            )
        ):
            clean.append(example)
    return clean


def _by_domain(examples: Sequence[RebaselineExample]) -> dict[str, list[RebaselineExample]]:
    groups: dict[str, list[RebaselineExample]] = defaultdict(list)
    for example in examples:
        groups[example.domain].append(example)
    return dict(groups)


def _has_both_classes(examples: Sequence[RebaselineExample]) -> bool:
    return len({example.label for example in examples}) == 2


def _n_examples_per_domain(examples: Sequence[RebaselineExample]) -> JsonDict:
    return {domain: len(group) for domain, group in sorted(_by_domain(examples).items())}


def _finite_triplets(
    labels: Sequence[int],
    first_scores: Sequence[float],
    second_scores: Sequence[float],
) -> tuple[list[int], list[float], list[float]]:
    clean_labels: list[int] = []
    clean_first: list[float] = []
    clean_second: list[float] = []
    for label, first, second in zip(labels, first_scores, second_scores, strict=False):
        first_f = float(first)
        second_f = float(second)
        if math.isfinite(first_f) and math.isfinite(second_f):
            clean_labels.append(int(label))
            clean_first.append(first_f)
            clean_second.append(second_f)
    return clean_labels, clean_first, clean_second


def _bernoulli_uniform_kl(p_error: float) -> float:
    p_correct = 1.0 - p_error
    return p_error * math.log(2.0 * p_error) + p_correct * math.log(2.0 * p_correct)


def _metric_point(metric: Mapping[str, Any]) -> float | None:
    point = metric.get("point")
    return None if point is None else float(point)


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _round(value: float) -> float:
    if not math.isfinite(float(value)):
        return float(value)
    return round(float(value), 6)


__all__ = [
    "OUTPUT_REL_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "RebaselineExample",
    "auroc_metric",
    "build_artifact",
    "build_artifact_from_examples",
    "delta_ci_excludes_zero_positive",
    "load_rebaseline_examples",
    "paired_auroc_delta_metric",
    "recall_at_fixed_fpr_table",
    "self_certainty_error_proxy_from_confidence_errors",
    "token_distribution_self_certainty",
    "validate_artifact",
    "write_artifact",
    "write_artifact_from_examples",
]
