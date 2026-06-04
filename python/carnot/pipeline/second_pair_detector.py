"""Deployable calibrated second-pair-of-eyes detector.

The detector fuses two runtime signals: verifier ensemble energy and model
confidence-derived error score.  It fits calibration on a deterministic train
split, then reports held-out discrimination, calibration quality, and fixed-FPR
operating points for each domain.  Math uses the original verifier ensemble
energy.  Code is currently an explicit abstention: Exp 3705 found the Exp 3695
in-corpus code AUROC=1.0 was separable-by-construction leakage and did not pass
the held-out survival gate, so the shipped surface returns no code verdict
instead of publishing a noisy or leaked code score.

Spec: REQ-SPOE-3657, REQ-SPOE-3657-ARTIFACT,
      REQ-SPOE-3671, REQ-SPOE-3696,
      REQ-SPOE-3706, REQ-SPOE-3779, SCENARIO-SPOE-3657,
      SCENARIO-SPOE-3658, SCENARIO-SPOE-3696, SCENARIO-SPOE-3706,
      SCENARIO-SPOE-3779
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np


JsonDict = dict[str, Any]

OUTPUT_REL_PATH = Path("results/experiment_3657_deployable_second_pair_of_eyes_detector.json")
SHIP_OUTPUT_REL_PATH = Path("results/experiment_3671_ship_second_pair_of_eyes_detector.json")
RANDOM_SEED = 3657
SHIP_RANDOM_SEED = 3671
CODE_OPERATING_POINT_SCOPE = (
    "math_only_with_code_abstain; math AUROC 0.98/ECE 0.009 preserved; "
    "Exp 3705 found the Exp 3695 in-corpus code AUROC 1.0 leaked and "
    "code_signal_survives_heldout=false, so code-domain candidates return no "
    "code verdict until a non-leaky held-out code operating point is earned"
)
CODE_NATIVE_CODE_PATH_ENABLED = False
CODE_ABSTAIN_ON_CODE = True
CODE_ABSTAIN_REASON = (
    "Exp 3705 leak audit set code_signal_survives_heldout=false after finding "
    "the Exp 3695 in-corpus 1.0 code AUROC was separable-by-construction; the "
    "shipped detector is narrowed to math-only for product verdicts."
)
CODE_NATIVE_OPERATING_POINT = {
    "source": "results/experiment_3705_code_native_leak_audit_heldout.json",
    "auroc": None,
    "auroc_ci": None,
    "fpr_budget": None,
    "threshold": None,
    "calibration": {},
    "abstain_on_code": True,
    "reason": CODE_ABSTAIN_REASON,
}
FPR_BUDGETS = (0.05, 0.10, 0.20)
MATERIAL_AUROC_LIFT = 0.01
MATERIAL_RECALL_LIFT = 0.01
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates (principle: scores cached corpora; no LLM load)."
)
DETECTOR_MODULE_PATH = "python/carnot/pipeline/second_pair_detector.py"
WIRED_SURFACE = "score_candidates MCP tool and carnot score-candidates CLI"


class _LazyCodeNativeVerifierModule:
    """Expose Exp 3695 lazily so this shipped module avoids import recursion."""

    def __getattr__(self, name: str) -> Any:
        from carnot.pipeline import code_native_verifier_3695 as module

        return getattr(module, name)


code_native_verifier_3695 = _LazyCodeNativeVerifierModule()

VERDICT_FUSION_WINS = (
    "complete: deployable_second_pair_of_eyes_detector_built_fusion_wins_calibrated"
)
VERDICT_FUSION_REDUNDANT = (
    "complete: deployable_detector_built_fusion_redundant_with_confidence_product_value_weak"
)
VERDICT_BLOCKED = "complete: blocked_no_labeled_corpus_for_fusion"
SHIP_VERDICT_MATH_CODE = (
    "complete: second_pair_of_eyes_detector_shipped_math_strong_code_honest_e2e_green"
)
SHIP_VERDICT_MATH_ONLY = (
    "complete: second_pair_of_eyes_detector_shipped_math_only_code_weak_documented_e2e_green"
)
SHIP_VERDICT_BLOCKED = "complete: blocked_no_labeled_corpus_for_detector"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "detector_module_path",
    "fused_detector_auroc",
    "confidence_alone_auroc",
    "recall_at_fixed_fpr_table",
    "calibration_brier_ece",
    "fusion_beats_confidence_alone",
    "n_examples_per_domain",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)
REQUIRED_SHIP_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "detector_module_path",
    "wired_surface",
    "fused_detector_auroc_per_domain",
    "confidence_alone_auroc_per_domain",
    "ensemble_alone_auroc_per_domain",
    "recall_at_fixed_fpr_table",
    "calibration_brier_ece_per_domain",
    "e2e_test_passed",
    "detector_shipped",
    "n_examples_per_domain",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates "
        "(principle: scores cached corpora; no LLM load)."
    ),
    "detector_module_path": (
        "Where the deployable fused detector lives -- the Phase-1 product surface."
    ),
    "fused_detector_auroc": "The deployable detector's headline number (per domain).",
    "confidence_alone_auroc": (
        "The bar -- the fused detector must materially beat confidence alone."
    ),
    "recall_at_fixed_fpr_table": (
        "domain -> {fpr -> recall} for the fused detector vs confidence -- the "
        "operating-point table a deployer reads."
    ),
    "calibration_brier_ece": (
        "Calibration quality of the fused score -- a deployable detector must be "
        "calibrated, not just discriminative."
    ),
    "fusion_beats_confidence_alone": (
        "BARE bool. True iff the fused detector materially beats confidence alone on "
        ">=1 headroom-bearing domain -- the product-value gate. STORE AS BARE true/false."
    ),
    "n_examples_per_domain": "Sample-size rigor per domain.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}
SHIP_FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "detector_module_path": (
        "Where the deployable fused detector lives -- the Phase-1 product surface."
    ),
    "wired_surface": (
        "Which caller surface (score_candidates MCP tool / CLI) the detector is wired to -- proves it is deployable, not just an experiment script."
    ),
    "fused_detector_auroc_per_domain": (
        "Held-out fused AUROC per domain (math, code) -- the deployable headline numbers."
    ),
    "confidence_alone_auroc_per_domain": (
        "The per-domain bar the fused detector must beat."
    ),
    "ensemble_alone_auroc_per_domain": (
        "The per-domain energy-only baseline for null discipline."
    ),
    "recall_at_fixed_fpr_table": (
        "domain -> {fpr -> recall} fused vs confidence -- the operating-point table a deployer reads."
    ),
    "calibration_brier_ece_per_domain": (
        "Calibration per domain -- a deployable detector must be calibrated and honest where it is not."
    ),
    "e2e_test_passed": (
        "True iff the shipped surface was called end-to-end and returned a calibrated score -- Phase-1 software-operational ship evidence."
    ),
    "detector_shipped": (
        "BARE bool. True iff the detector is wired to a real caller surface AND the E2E test passed AND it beats confidence on >=1 domain. STORE AS BARE true/false."
    ),
    "n_examples_per_domain": "Sample-size rigor per domain.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


@dataclass(frozen=True)
class LabeledDetectorExample:
    """One labeled row for fused detector calibration.

    ``label`` is 1 for an error/hallucination/bug and 0 for a correct example.
    Both scores are oriented so larger values mean higher error risk.
    """

    domain: str
    label: int
    ensemble_energy: float
    confidence_error: float
    example_id: str = ""


@dataclass(frozen=True)
class CandidateScoreInput:
    """One runtime row for the shipped score_candidates surface.

    ``confidence`` is model confidence that the candidate is correct.  The
    derived ``confidence_error`` feature is ``1 - confidence``.  Callers that
    already computed the oriented error score may pass ``confidence_error``
    directly.
    """

    candidate_id: str
    domain: str
    text: str
    confidence: float | None = None
    confidence_error: float | None = None
    ensemble_energy: float | None = None


class CalibratedFusedDetector:
    """Logistic calibrator over ensemble energy and confidence error."""

    feature_names = ("ensemble_energy", "confidence_error")

    def __init__(
        self,
        *,
        learning_rate: float = 0.2,
        max_iter: int = 800,
        l2: float = 1e-4,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.l2 = float(l2)
        self.coef_: list[float] | None = None
        self.intercept_: float | None = None
        self.feature_mean_: list[float] | None = None
        self.feature_scale_: list[float] | None = None

    def fit(self, examples: Sequence[LabeledDetectorExample]) -> CalibratedFusedDetector:
        """Fit logistic calibration on labeled examples."""

        clean = _clean_examples(examples)
        labels = np.asarray([example.label for example in clean], dtype=np.float64)
        if len(set(labels.tolist())) < 2:
            raise ValueError("calibration requires both positive and negative labels")
        features = _feature_array(clean)
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale == 0.0] = 1.0
        standardized = (features - mean) / scale
        design = np.column_stack([np.ones(len(standardized)), standardized])
        weights = np.zeros(design.shape[1], dtype=np.float64)
        for _ in range(self.max_iter):
            probs = _sigmoid(design @ weights)
            grad = (design.T @ (probs - labels)) / len(labels)
            grad[1:] += self.l2 * weights[1:]
            weights -= self.learning_rate * grad
        self.intercept_ = float(weights[0])
        self.coef_ = [float(value) for value in weights[1:]]
        self.feature_mean_ = [float(value) for value in mean]
        self.feature_scale_ = [float(value) for value in scale]
        return self

    def predict_proba(self, examples: Sequence[LabeledDetectorExample]) -> list[float]:
        """Return calibrated error probabilities for examples."""

        if (
            self.coef_ is None
            or self.intercept_ is None
            or self.feature_mean_ is None
            or self.feature_scale_ is None
        ):
            raise ValueError("detector must be fitted before predict_proba")
        clean = _clean_examples(examples)
        if not clean:
            return []
        features = _feature_array(clean)
        mean = np.asarray(self.feature_mean_, dtype=np.float64)
        scale = np.asarray(self.feature_scale_, dtype=np.float64)
        coef = np.asarray(self.coef_, dtype=np.float64)
        logits = self.intercept_ + ((features - mean) / scale) @ coef
        return [round(float(value), 12) for value in _sigmoid(logits)]

    def evaluate_domains(
        self,
        examples: Sequence[LabeledDetectorExample],
    ) -> JsonDict:
        """Evaluate held-out discrimination, calibration, and operating points."""

        return evaluate_domains_with_detector(self, examples)


def build_artifact(
    root: Path | str,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    examples: Sequence[LabeledDetectorExample] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3657 artifact from cached corpora or supplied examples."""

    root_path = Path(root)
    if examples is None:
        examples, corpus_status = load_cached_labeled_examples(root_path)
    else:
        corpus_status = {"synthetic": {"status": "provided", "n_examples": len(examples)}}
    artifact = build_artifact_from_examples(
        examples,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    artifact["corpus_status"] = corpus_status
    artifact["output_path"] = str(_repo_path(root_path, Path(output_path)))
    validate_artifact(artifact)
    return artifact


def build_artifact_from_examples(
    examples: Sequence[LabeledDetectorExample],
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Evaluate held-out domains and return the deployable detector artifact."""

    start = time.perf_counter() if started_s is None else float(started_s)
    clean = _clean_examples(examples)
    train, holdout = stratified_train_holdout(clean, seed=RANDOM_SEED)
    if not _has_both_classes(train) or not any(
        _has_both_classes(group) for group in _by_domain(holdout).values()
    ):
        finished = time.perf_counter() if now_s is None else float(now_s)
        artifact = _base_artifact(
            verdict=VERDICT_BLOCKED,
            fusion_beats=False,
            duration_s=round(max(0.0, finished - start), 6),
            tests_run=tests_run,
        )
        artifact.update(
            {
                "fused_detector_auroc": {},
                "confidence_alone_auroc": {},
                "ensemble_alone_auroc": {},
                "recall_at_fixed_fpr_table": {},
                "calibration_brier_ece": {},
                "operating_points": {},
                "n_examples_per_domain": _n_examples_per_domain(clean),
            }
        )
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        validate_artifact(artifact)
        return artifact

    detector = CalibratedFusedDetector().fit(train)
    fused_auroc: JsonDict = {}
    confidence_auroc: JsonDict = {}
    ensemble_auroc: JsonDict = {}
    recall_table: JsonDict = {}
    calibration: JsonDict = {}
    operating_points: JsonDict = {}
    material_wins: dict[str, bool] = {}

    for domain, domain_examples in sorted(_by_domain(holdout).items()):
        if not _has_both_classes(domain_examples):
            continue
        labels = [example.label for example in domain_examples]
        fused_scores = detector.predict_proba(domain_examples)
        confidence_scores = [example.confidence_error for example in domain_examples]
        ensemble_scores = [example.ensemble_energy for example in domain_examples]
        fused_auroc[domain] = _round(tie_aware_auroc(labels, fused_scores))
        confidence_auroc[domain] = _round(tie_aware_auroc(labels, confidence_scores))
        ensemble_auroc[domain] = _round(tie_aware_auroc(labels, ensemble_scores))
        recall_table[domain] = recall_at_fixed_fpr_table(
            labels,
            fused_scores,
            confidence_scores,
            ensemble_scores,
        )
        calibration[domain] = {
            "brier": _round(brier_score(labels, fused_scores)),
            "ece": _round(expected_calibration_error(labels, fused_scores)),
        }
        operating_points[domain] = recommended_operating_point(recall_table[domain])
        recall_delta = (
            recall_table[domain]["0.10"]["fused_recall"]
            - recall_table[domain]["0.10"]["confidence_recall"]
        )
        material_wins[domain] = bool(
            confidence_auroc[domain] < 0.95
            and fused_auroc[domain] - confidence_auroc[domain] >= MATERIAL_AUROC_LIFT
            and recall_delta >= MATERIAL_RECALL_LIFT
        )

    fusion_beats = any(material_wins.values())
    verdict = VERDICT_FUSION_WINS if fusion_beats else VERDICT_FUSION_REDUNDANT
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact = _base_artifact(
        verdict=verdict,
        fusion_beats=fusion_beats,
        duration_s=round(max(0.0, finished - start), 6),
        tests_run=tests_run,
    )
    artifact.update(
        {
            "fused_detector_auroc": fused_auroc,
            "confidence_alone_auroc": confidence_auroc,
            "ensemble_alone_auroc": ensemble_auroc,
            "recall_at_fixed_fpr_table": recall_table,
            "calibration_brier_ece": calibration,
            "operating_points": operating_points,
            "n_examples_per_domain": _n_examples_per_domain(clean),
            "heldout_examples_per_domain": _n_examples_per_domain(holdout),
            "material_win_per_domain": material_wins,
            "calibrator": {
                "method": "logistic",
                "feature_names": list(detector.feature_names),
                "coef": [_round(value) for value in detector.coef_ or []],
                "intercept": _round(detector.intercept_ or 0.0),
            },
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    examples: Sequence[LabeledDetectorExample] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3657 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(
        root_path,
        output_path=output_path,
        examples=examples,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_ship_artifact(
    root: Path | str,
    *,
    output_path: Path | str = SHIP_OUTPUT_REL_PATH,
    examples: Sequence[LabeledDetectorExample] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    e2e_override: bool | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3671 shipped detector artifact."""

    root_path = Path(root)
    if examples is None:
        examples, corpus_status = load_cached_labeled_examples(
            root_path,
            use_balanced_code_corpus=True,
        )
    else:
        corpus_status = {"synthetic": {"status": "provided", "n_examples": len(examples)}}
    artifact = build_ship_artifact_from_examples(
        examples,
        root=root_path,
        started_s=started_s,
        now_s=now_s,
        e2e_override=e2e_override,
        tests_run=tests_run,
    )
    artifact["corpus_status"] = corpus_status
    artifact["output_path"] = str(_repo_path(root_path, Path(output_path)))
    validate_ship_artifact(artifact)
    return artifact


def build_ship_artifact_from_examples(
    examples: Sequence[LabeledDetectorExample],
    *,
    root: Path | str | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    e2e_override: bool | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Evaluate domains and return the Exp 3671 ship artifact."""

    start = time.perf_counter() if started_s is None else float(started_s)
    clean = _clean_examples(examples)
    ship_clean = _shipped_detector_examples(clean)
    train, holdout = stratified_train_holdout(ship_clean, seed=SHIP_RANDOM_SEED)
    if not _has_both_classes(train) or not any(
        _has_both_classes(group) for group in _by_domain(holdout).values()
    ):
        finished = time.perf_counter() if now_s is None else float(now_s)
        artifact = _base_ship_artifact(
            verdict=SHIP_VERDICT_BLOCKED,
            detector_shipped=False,
            e2e_passed=False,
            duration_s=round(max(0.0, finished - start), 6),
            tests_run=tests_run,
        )
        artifact.update(
            {
                "fused_detector_auroc_per_domain": {},
                "confidence_alone_auroc_per_domain": {},
                "ensemble_alone_auroc_per_domain": {},
                "recall_at_fixed_fpr_table": {},
                "calibration_brier_ece_per_domain": {},
                "operating_points": {},
                "material_win_per_domain": {},
                "n_examples_per_domain": _n_examples_per_domain(clean),
                "heldout_examples_per_domain": _n_examples_per_domain(holdout),
                "code_abstention": _code_abstention_payload(clean),
            }
        )
        artifact["reproducibility_checksum"] = reproducibility_checksum_for_ship(artifact)
        validate_ship_artifact(artifact)
        return artifact

    detector = CalibratedFusedDetector().fit(train)
    metrics = detector.evaluate_domains(holdout)
    e2e_passed = (
        bool(e2e_override)
        if e2e_override is not None
        else _run_surface_e2e_smoke(clean, holdout, root=Path(root or "."))
    )
    material_wins = metrics["material_win_per_domain"]
    beats_confidence = any(bool(value) for value in material_wins.values())
    detector_shipped = bool(WIRED_SURFACE and e2e_passed and beats_confidence)
    code_strong = bool(material_wins.get("code"))
    if not detector_shipped:
        verdict = SHIP_VERDICT_BLOCKED
    elif code_strong:
        verdict = SHIP_VERDICT_MATH_CODE
    else:
        verdict = SHIP_VERDICT_MATH_ONLY

    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact = _base_ship_artifact(
        verdict=verdict,
        detector_shipped=detector_shipped,
        e2e_passed=e2e_passed,
        duration_s=round(max(0.0, finished - start), 6),
        tests_run=tests_run,
    )
    artifact.update(
        {
            "fused_detector_auroc_per_domain": metrics["fused_detector_auroc_per_domain"],
            "confidence_alone_auroc_per_domain": metrics[
                "confidence_alone_auroc_per_domain"
            ],
            "ensemble_alone_auroc_per_domain": metrics["ensemble_alone_auroc_per_domain"],
            "recall_at_fixed_fpr_table": metrics["recall_at_fixed_fpr_table"],
            "calibration_brier_ece_per_domain": metrics[
                "calibration_brier_ece_per_domain"
            ],
            "operating_points": metrics["operating_points"],
            "material_win_per_domain": material_wins,
            "n_examples_per_domain": _n_examples_per_domain(clean),
            "heldout_examples_per_domain": _n_examples_per_domain(holdout),
            "code_abstention": _code_abstention_payload(clean),
            "calibrator": {
                "method": "logistic",
                "feature_names": list(detector.feature_names),
                "coef": [_round(value) for value in detector.coef_ or []],
                "intercept": _round(detector.intercept_ or 0.0),
            },
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum_for_ship(artifact)
    validate_ship_artifact(artifact)
    return artifact


def write_ship_artifact(
    root: Path | str,
    *,
    output_path: Path | str = SHIP_OUTPUT_REL_PATH,
    examples: Sequence[LabeledDetectorExample] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    e2e_override: bool | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3671 ship artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_ship_artifact(
        root_path,
        output_path=output_path,
        examples=examples,
        started_s=started_s,
        now_s=now_s,
        e2e_override=e2e_override,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def score_candidates(
    candidates: Sequence[CandidateScoreInput | Mapping[str, Any]],
    *,
    root: Path | str = ".",
    examples: Sequence[LabeledDetectorExample] | None = None,
    default_domain: str | None = None,
    abstention_mode: bool = False,
    abstention_threshold: float | None = None,
) -> JsonDict:
    """Score candidates with the shipped calibrated fused detector surface."""

    root_path = Path(root)
    if examples is None:
        examples, corpus_status = load_cached_labeled_examples(
            root_path,
            use_balanced_code_corpus=True,
        )
    else:
        corpus_status = {"synthetic": {"status": "provided", "n_examples": len(examples)}}
    clean = _clean_examples(examples)
    ship_clean = _shipped_detector_examples(clean)
    train, holdout = stratified_train_holdout(ship_clean, seed=SHIP_RANDOM_SEED)
    if not _has_both_classes(train) or not any(
        _has_both_classes(group) for group in _by_domain(holdout).values()
    ):
        raise ValueError("score_candidates requires a labeled corpus with evaluable holdout rows")
    detector = CalibratedFusedDetector().fit(train)
    metrics = detector.evaluate_domains(holdout)
    normalized = [
        _coerce_candidate_input(candidate, default_domain=default_domain)
        for candidate in candidates
    ]
    scoreable: list[tuple[int, CandidateScoreInput, LabeledDetectorExample]] = []
    rows_by_index: dict[int, JsonDict] = {}
    for idx, candidate in enumerate(normalized):
        confidence_error = _candidate_confidence_error(candidate)
        if _domain_abstains(candidate.domain):
            rows_by_index[idx] = {
                "candidate_id": candidate.candidate_id,
                "domain": candidate.domain,
                "calibrated_error_score": None,
                "ensemble_energy": None,
                "confidence_error": _round(confidence_error),
                "operating_point": None,
                "code_verdict": "no_code_verdict",
                "abstained": True,
                "abstain_reason": CODE_ABSTAIN_REASON,
            }
            continue
        scoreable.append(
            (
                idx,
                candidate,
                LabeledDetectorExample(
                    domain=candidate.domain,
                    label=0,
                    ensemble_energy=_candidate_ensemble_energy(candidate, root_path),
                    confidence_error=confidence_error,
                    example_id=candidate.candidate_id,
                ),
            )
        )
    probabilities = detector.predict_proba([example for _, _, example in scoreable])
    for (idx, candidate, example), probability in zip(scoreable, probabilities, strict=True):
        operating_point = metrics["operating_points"].get(candidate.domain)
        rows_by_index[idx] = {
            "candidate_id": candidate.candidate_id,
            "domain": candidate.domain,
            "calibrated_error_score": _round(probability),
            "ensemble_energy": _round(example.ensemble_energy),
            "confidence_error": _round(example.confidence_error),
            "operating_point": operating_point,
            "abstained": False,
        }
    rows = [rows_by_index[idx] for idx in range(len(normalized))]
    abstention_summary = None
    if abstention_mode:
        from carnot.pipeline import certified_abstention_surface as abstention

        config = abstention.load_certified_abstention_config()
        operator_override = abstention_threshold is not None
        if operator_override:
            config = config.with_threshold(float(abstention_threshold))
        rows = [
            (
                abstention.apply_certified_abstention(row, config)
                if row.get("calibrated_error_score") is not None
                else row
            )
            for row in rows
        ]
        abstention_summary = abstention.abstention_mode_summary(
            config,
            operator_threshold_override=operator_override,
        )
    result = {
        "surface": "score_candidates",
        "wired_surface": WIRED_SURFACE,
        "detector_module_path": DETECTOR_MODULE_PATH,
        "code_operating_point_scope": CODE_OPERATING_POINT_SCOPE,
        "calibration_source": {
            "random_seed": SHIP_RANDOM_SEED,
            "n_examples_per_domain": _n_examples_per_domain(clean),
            "corpus_status": corpus_status,
            "code_abstention": _code_abstention_payload(clean),
        },
        "domain_metrics": {
            "fused_detector_auroc_per_domain": metrics[
                "fused_detector_auroc_per_domain"
            ],
            "confidence_alone_auroc_per_domain": metrics[
                "confidence_alone_auroc_per_domain"
            ],
            "ensemble_alone_auroc_per_domain": metrics["ensemble_alone_auroc_per_domain"],
            "calibration_brier_ece_per_domain": metrics[
                "calibration_brier_ece_per_domain"
            ],
        },
        "scores": rows,
    }
    if abstention_summary is not None:
        result["abstention_mode"] = abstention_summary
    return result


def load_cached_labeled_examples(
    root: Path | str,
    *,
    score_overrides: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    use_balanced_code_corpus: bool = False,
) -> tuple[list[LabeledDetectorExample], JsonDict]:
    """Load labeled FoVer math and Exp 3641 code rows with computed scores."""

    root_path = Path(root)
    overrides = score_overrides or {}
    examples: list[LabeledDetectorExample] = []
    status: JsonDict = {}
    math_examples, math_status = _load_math_examples(root_path, overrides.get("math", {}))
    code_examples, code_status = _load_code_examples(
        root_path,
        overrides.get("code", {}),
        use_balanced_code_corpus=use_balanced_code_corpus,
    )
    examples.extend(math_examples)
    examples.extend(code_examples)
    status["math"] = math_status
    status["code"] = code_status
    return examples, status


def operating_points_at_fixed_fpr(
    labels: Sequence[int],
    scores: Sequence[float],
    fpr_budgets: Sequence[float] = FPR_BUDGETS,
) -> JsonDict:
    """Return best recall thresholds under each fixed-FPR budget."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    clean_labels, clean_scores = finite_label_scores(labels, scores)
    positives = sum(1 for label in clean_labels if label == 1)
    negatives = len(clean_labels) - positives
    if positives == 0 or negatives == 0:
        return {
            _fpr_key(budget): {"threshold": None, "actual_fpr": 0.0, "recall": 0.0}
            for budget in fpr_budgets
        }
    thresholds = sorted(set(clean_scores), reverse=True) + [math.inf]
    table: JsonDict = {}
    for budget in fpr_budgets:
        best = {"threshold": None, "actual_fpr": 0.0, "recall": 0.0}
        for threshold in thresholds:
            predictions = [score >= threshold for score in clean_scores]
            fp = sum(
                1
                for label, pred in zip(clean_labels, predictions, strict=True)
                if label == 0 and pred
            )
            tp = sum(
                1
                for label, pred in zip(clean_labels, predictions, strict=True)
                if label == 1 and pred
            )
            actual_fpr = fp / negatives
            recall = tp / positives
            if actual_fpr <= float(budget) + 1e-12 and (
                recall > best["recall"]
                or (recall == best["recall"] and actual_fpr > best["actual_fpr"])
            ):
                best = {
                    "threshold": None if math.isinf(threshold) else _round(threshold),
                    "actual_fpr": _round(actual_fpr),
                    "recall": _round(recall),
                }
        table[_fpr_key(budget)] = best
    return table


def recall_at_fixed_fpr_table(
    labels: Sequence[int],
    fused_scores: Sequence[float],
    confidence_scores: Sequence[float],
    ensemble_scores: Sequence[float],
) -> JsonDict:
    """Combine fused, confidence, and ensemble fixed-FPR operating rows."""

    fused = operating_points_at_fixed_fpr(labels, fused_scores)
    confidence = operating_points_at_fixed_fpr(labels, confidence_scores)
    ensemble = operating_points_at_fixed_fpr(labels, ensemble_scores)
    table: JsonDict = {}
    for key in fused:
        table[key] = {
            "fused_recall": fused[key]["recall"],
            "confidence_recall": confidence[key]["recall"],
            "ensemble_recall": ensemble[key]["recall"],
            "fused_actual_fpr": fused[key]["actual_fpr"],
            "confidence_actual_fpr": confidence[key]["actual_fpr"],
            "ensemble_actual_fpr": ensemble[key]["actual_fpr"],
            "fused_threshold": fused[key]["threshold"],
            "confidence_threshold": confidence[key]["threshold"],
            "ensemble_threshold": ensemble[key]["threshold"],
        }
    return table


def evaluate_domains_with_detector(
    detector: CalibratedFusedDetector,
    examples: Sequence[LabeledDetectorExample],
) -> JsonDict:
    """Evaluate a fitted detector on per-domain held-out examples."""

    fused_auroc: JsonDict = {}
    confidence_auroc: JsonDict = {}
    ensemble_auroc: JsonDict = {}
    recall_table: JsonDict = {}
    calibration: JsonDict = {}
    operating_points: JsonDict = {}
    material_wins: dict[str, bool] = {}
    for domain, domain_examples in sorted(_by_domain(_clean_examples(examples)).items()):
        if not _has_both_classes(domain_examples):
            continue
        labels = [example.label for example in domain_examples]
        fused_scores = detector.predict_proba(domain_examples)
        confidence_scores = [example.confidence_error for example in domain_examples]
        ensemble_scores = [example.ensemble_energy for example in domain_examples]
        fused_auroc[domain] = _round(tie_aware_auroc(labels, fused_scores))
        confidence_auroc[domain] = _round(tie_aware_auroc(labels, confidence_scores))
        ensemble_auroc[domain] = _round(tie_aware_auroc(labels, ensemble_scores))
        recall_table[domain] = recall_at_fixed_fpr_table(
            labels,
            fused_scores,
            confidence_scores,
            ensemble_scores,
        )
        calibration[domain] = {
            "brier": _round(brier_score(labels, fused_scores)),
            "ece": _round(expected_calibration_error(labels, fused_scores)),
        }
        operating_points[domain] = recommended_operating_point(recall_table[domain])
        material_wins[domain] = bool(
            confidence_auroc[domain] < 0.95
            and fused_auroc[domain] - confidence_auroc[domain] >= MATERIAL_AUROC_LIFT
        )
    return {
        "fused_detector_auroc_per_domain": fused_auroc,
        "confidence_alone_auroc_per_domain": confidence_auroc,
        "ensemble_alone_auroc_per_domain": ensemble_auroc,
        "recall_at_fixed_fpr_table": recall_table,
        "calibration_brier_ece_per_domain": calibration,
        "operating_points": operating_points,
        "material_win_per_domain": material_wins,
    }


def recommended_operating_point(table: Mapping[str, Mapping[str, float | None]]) -> JsonDict:
    """Pick the default deployer operating point from the fixed-FPR table."""

    row = table["0.10"]
    return {
        "fpr_budget": 0.10,
        "threshold": row["fused_threshold"],
        "expected_recall": row["fused_recall"],
        "actual_fpr": row["fused_actual_fpr"],
    }


def tie_aware_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC with half credit for tied positive/negative pairs."""

    clean_labels, clean_scores = finite_label_scores(labels, scores)
    positives = [
        score for label, score in zip(clean_labels, clean_scores, strict=True) if label == 1
    ]
    negatives = [
        score for label, score in zip(clean_labels, clean_scores, strict=True) if label == 0
    ]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def brier_score(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Return mean squared calibration error."""

    clean_labels, clean_probs = finite_label_scores(labels, probabilities)
    if not clean_labels:
        return 0.0
    return sum(
        (prob - label) ** 2 for label, prob in zip(clean_labels, clean_probs, strict=True)
    ) / len(clean_labels)


def expected_calibration_error(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    """Return equal-width-bin expected calibration error."""

    clean_labels, clean_probs = finite_label_scores(labels, probabilities)
    if not clean_labels:
        return 0.0
    total = len(clean_labels)
    ece = 0.0
    for bin_idx in range(n_bins):
        lo = bin_idx / n_bins
        hi = (bin_idx + 1) / n_bins
        in_bin = [
            (label, prob)
            for label, prob in zip(clean_labels, clean_probs, strict=True)
            if (lo <= prob <= hi if bin_idx == 0 else lo < prob <= hi)
        ]
        if in_bin:
            accuracy = sum(label for label, _ in in_bin) / len(in_bin)
            confidence = sum(prob for _, prob in in_bin) / len(in_bin)
            ece += (len(in_bin) / total) * abs(accuracy - confidence)
    return ece


def stratified_train_holdout(
    examples: Sequence[LabeledDetectorExample],
    *,
    seed: int,
    train_fraction: float = 0.7,
) -> tuple[list[LabeledDetectorExample], list[LabeledDetectorExample]]:
    """Split each domain/label bucket so held-out metrics keep both classes."""

    rng = random.Random(seed)
    train: list[LabeledDetectorExample] = []
    holdout: list[LabeledDetectorExample] = []
    buckets: dict[tuple[str, int], list[LabeledDetectorExample]] = defaultdict(list)
    for example in examples:
        buckets[(example.domain, example.label)].append(example)
    for bucket in buckets.values():
        ordered = sorted(bucket, key=lambda item: item.example_id)
        rng.shuffle(ordered)
        if len(ordered) == 1:
            train.extend(ordered)
            continue
        n_train = max(1, min(len(ordered) - 1, round(len(ordered) * train_fraction)))
        train.extend(ordered[:n_train])
        holdout.extend(ordered[n_train:])
    return train, holdout


def finite_label_scores(
    labels: Sequence[int],
    scores: Sequence[float],
) -> tuple[list[int], list[float]]:
    """Drop non-finite scores and align labels to scores."""

    clean_labels: list[int] = []
    clean_scores: list[float] = []
    for label, score in zip(labels, scores, strict=False):
        score_f = float(score)
        if math.isfinite(score_f):
            clean_labels.append(int(label))
            clean_scores.append(score_f)
    return clean_labels, clean_scores


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3657 terminal artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("honest_verdict") not in {
        VERDICT_FUSION_WINS,
        VERDICT_FUSION_REDUNDANT,
        VERDICT_BLOCKED,
    }:
        raise ValueError("honest_verdict is not an accepted Exp 3657 terminal verdict")
    if type(artifact.get("fusion_beats_confidence_alone")) is not bool:
        raise ValueError("fusion_beats_confidence_alone must be a bare top-level bool")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")


def validate_ship_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3671 terminal ship artifact contract."""

    missing = [field for field in REQUIRED_SHIP_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required ship artifact fields: {missing}")
    if artifact.get("honest_verdict") not in {
        SHIP_VERDICT_MATH_CODE,
        SHIP_VERDICT_MATH_ONLY,
        SHIP_VERDICT_BLOCKED,
    }:
        raise ValueError("honest_verdict is not an accepted Exp 3671 terminal verdict")
    if type(artifact.get("detector_shipped")) is not bool:
        raise ValueError("detector_shipped must be a bare top-level bool")
    if type(artifact.get("e2e_test_passed")) is not bool:
        raise ValueError("e2e_test_passed must be a bare top-level bool")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic artifact fields for drift detection."""

    payload = {
        "fused_detector_auroc": artifact.get("fused_detector_auroc"),
        "confidence_alone_auroc": artifact.get("confidence_alone_auroc"),
        "ensemble_alone_auroc": artifact.get("ensemble_alone_auroc"),
        "recall_at_fixed_fpr_table": artifact.get("recall_at_fixed_fpr_table"),
        "calibration_brier_ece": artifact.get("calibration_brier_ece"),
        "n_examples_per_domain": artifact.get("n_examples_per_domain"),
        "random_seed": RANDOM_SEED,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def reproducibility_checksum_for_ship(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic Exp 3671 ship fields for drift detection."""

    payload = {
        "fused_detector_auroc_per_domain": artifact.get("fused_detector_auroc_per_domain"),
        "confidence_alone_auroc_per_domain": artifact.get(
            "confidence_alone_auroc_per_domain"
        ),
        "ensemble_alone_auroc_per_domain": artifact.get("ensemble_alone_auroc_per_domain"),
        "recall_at_fixed_fpr_table": artifact.get("recall_at_fixed_fpr_table"),
        "calibration_brier_ece_per_domain": artifact.get(
            "calibration_brier_ece_per_domain"
        ),
        "detector_shipped": artifact.get("detector_shipped"),
        "e2e_test_passed": artifact.get("e2e_test_passed"),
        "n_examples_per_domain": artifact.get("n_examples_per_domain"),
        "random_seed": SHIP_RANDOM_SEED,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _base_artifact(
    *,
    verdict: str,
    fusion_beats: bool,
    duration_s: float,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    return {
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "detector_module_path": DETECTOR_MODULE_PATH,
        "fusion_beats_confidence_alone": bool(fusion_beats),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "fused_detector_auroc present AND confidence_alone_auroc present "
                "AND calibration_brier_ece present"
            ),
            "passed": True,
            "principle": (
                "A deployable detector claim requires a measured fused AUROC vs "
                "confidence AND a calibration metric -- discrimination without "
                "calibration is not deployable."
            ),
        },
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
    }


def _base_ship_artifact(
    *,
    verdict: str,
    detector_shipped: bool,
    e2e_passed: bool,
    duration_s: float,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    return {
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "detector_module_path": DETECTOR_MODULE_PATH,
        "wired_surface": WIRED_SURFACE,
        "e2e_test_passed": bool(e2e_passed),
        "detector_shipped": bool(detector_shipped),
        "random_seed": SHIP_RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": duration_s,
        "field_principles": dict(SHIP_FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "detector_module_path present AND e2e_test_passed == true AND "
                "fused_detector_auroc_per_domain present AND "
                "calibration_brier_ece_per_domain present"
            ),
            "passed": bool(detector_shipped),
            "principle": (
                "A shipped product surface requires a real module, a passing E2E "
                "test, measured per-domain discrimination AND calibration -- an "
                "experiment artifact alone is not a shipped product."
            ),
        },
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }


def _run_surface_e2e_smoke(
    examples: Sequence[LabeledDetectorExample],
    holdout: Sequence[LabeledDetectorExample],
    *,
    root: Path,
) -> bool:
    """Call the shipped score surface on one held-out example."""

    for example in holdout:
        if _has_both_classes(_by_domain(holdout).get(example.domain, [])):
            try:
                result = score_candidates(
                    [
                        CandidateScoreInput(
                            candidate_id=f"e2e-{example.example_id}",
                            domain=example.domain,
                            text=example.example_id,
                            confidence_error=example.confidence_error,
                            ensemble_energy=example.ensemble_energy,
                        )
                    ],
                    root=root,
                    examples=examples,
                )
            except Exception:
                return False
            scores = result.get("scores")
            if isinstance(scores, list) and scores:
                value = scores[0].get("calibrated_error_score")
                return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
    return False


def _coerce_candidate_input(
    candidate: CandidateScoreInput | Mapping[str, Any],
    *,
    default_domain: str | None,
) -> CandidateScoreInput:
    if isinstance(candidate, CandidateScoreInput):
        if not candidate.domain and default_domain is not None:
            return CandidateScoreInput(
                candidate_id=candidate.candidate_id,
                domain=default_domain,
                text=candidate.text,
                confidence=candidate.confidence,
                confidence_error=candidate.confidence_error,
                ensemble_energy=candidate.ensemble_energy,
            )
        return candidate
    domain = str(candidate.get("domain") or default_domain or "math")
    text = str(
        candidate.get("text")
        or candidate.get("answer")
        or candidate.get("response")
        or candidate.get("candidate_code")
        or ""
    )
    candidate_id = str(candidate.get("candidate_id") or candidate.get("id") or len(text))
    confidence_raw = candidate.get("confidence")
    confidence_error_raw = candidate.get("confidence_error")
    ensemble_raw = candidate.get("ensemble_energy")
    return CandidateScoreInput(
        candidate_id=candidate_id,
        domain=domain,
        text=text,
        confidence=None if confidence_raw is None else float(confidence_raw),
        confidence_error=None if confidence_error_raw is None else float(confidence_error_raw),
        ensemble_energy=None if ensemble_raw is None else float(ensemble_raw),
    )


def _candidate_confidence_error(candidate: CandidateScoreInput) -> float:
    if candidate.confidence_error is not None:
        return max(0.0, min(1.0, float(candidate.confidence_error)))
    if candidate.confidence is not None:
        return max(0.0, min(1.0, 1.0 - float(candidate.confidence)))
    return 0.5


def _candidate_ensemble_energy(candidate: CandidateScoreInput, root: Path) -> float:
    if candidate.ensemble_energy is not None:
        return float(candidate.ensemble_energy)
    domain = candidate.domain.lower()
    if _domain_abstains(domain):
        raise ValueError("code-domain detector is narrowed to no-code-verdict abstention")
    if domain == "code":
        rows = [
            {
                "candidate_code": candidate.text,
                "task_id": candidate.candidate_id,
                "metadata": {
                    "corpus": "surface",
                    "stable_id": candidate.candidate_id,
                    "candidate_index": 0,
                },
            }
        ]
        return float(_score_code_native_rows(rows)[0])
    rows = [{"step_text": candidate.text}]
    return float(_score_math_rows(rows)[0])


def _load_math_examples(
    root: Path,
    overrides: Mapping[str, Sequence[float]],
) -> tuple[list[LabeledDetectorExample], JsonDict]:
    path = root / "data/fover_corpus_v4.json"
    if not path.exists():
        return [], {"status": "missing", "path": str(path)}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        return [], {
            "status": "blocked",
            "reason": "empty_or_invalid_fover_corpus",
            "path": str(path),
        }
    if "ensemble_scores" in overrides:
        ensemble_scores = [float(score) for score in overrides["ensemble_scores"]]
    else:
        ensemble_scores = _score_math_rows(rows)
    confidence_scores = [
        float(score)
        for score in overrides.get(
            "confidence_scores",
            [
                1.0 - _coerce_float(row.get("confidence"), 0.5)
                for row in rows
                if isinstance(row, Mapping)
            ],
        )
    ]
    examples = []
    for idx, (row, ensemble, confidence) in enumerate(
        zip(rows, ensemble_scores, confidence_scores, strict=False)
    ):
        if not isinstance(row, Mapping):
            continue
        label = 1 if str(row.get("label") or "").lower() == "incorrect" else 0
        examples.append(
            LabeledDetectorExample(
                domain="math",
                label=label,
                ensemble_energy=float(ensemble),
                confidence_error=float(confidence),
                example_id=f"math-{row.get('question_id', idx)}-{idx}",
            )
        )
    return examples, {"status": "loaded", "path": str(path), "n_examples": len(examples)}


def _load_code_examples(
    root: Path,
    overrides: Mapping[str, Sequence[float]],
    *,
    use_balanced_code_corpus: bool,
) -> tuple[list[LabeledDetectorExample], JsonDict]:
    if use_balanced_code_corpus:
        artifact_path = root / "results/experiment_3658_code_generalization_second_corpus.json"
        corpus_path = root / "data/code_verification_corpus_v2.jsonl"
        if artifact_path.exists():
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            if isinstance(artifact, Mapping) and artifact.get("second_code_corpus_path"):
                corpus_path = _repo_path(root, Path(str(artifact["second_code_corpus_path"])))
    else:
        artifact_path = root / "results/experiment_3641_code_corpus_verifiers_fire_transfer_v3.json"
        corpus_path = root / "data/code_verification_corpus_v1.jsonl"
    if not use_balanced_code_corpus and artifact_path.exists():
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if isinstance(artifact, Mapping) and artifact.get("code_corpus_path"):
            corpus_path = _repo_path(root, Path(str(artifact["code_corpus_path"])))
    if not corpus_path.exists():
        return [], {
            "status": "missing",
            "path": str(corpus_path),
            "balanced_exp3658": bool(use_balanced_code_corpus),
        }
    rows = _read_jsonl(corpus_path)
    if not rows:
        return [], {
            "status": "blocked",
            "reason": "empty_code_corpus",
            "path": str(corpus_path),
            "balanced_exp3658": bool(use_balanced_code_corpus),
        }
    if use_balanced_code_corpus and CODE_ABSTAIN_ON_CODE and "ensemble_scores" not in overrides:
        return [], {
            "status": "abstained",
            "reason": CODE_ABSTAIN_REASON,
            "path": str(corpus_path),
            "n_examples": len(rows),
            "balanced_exp3658": True,
            "code_native_exp3695": False,
        }
    if "ensemble_scores" in overrides:
        ensemble_scores = [float(score) for score in overrides["ensemble_scores"]]
        code_native_scored = False
    elif use_balanced_code_corpus:
        ensemble_scores = _score_code_native_rows(rows)
        code_native_scored = True
    else:
        from carnot.verify import corrected_cross_domain_remeasurement_v4 as exp3642

        ensemble_scores = exp3642.score_code_rows(rows, root)
        code_native_scored = False
    if "confidence_scores" in overrides:
        confidence_scores = [float(score) for score in overrides["confidence_scores"]]
    else:
        from carnot.verify import corrected_cross_domain_remeasurement_v4 as exp3642

        confidence_scores = exp3642.score_code_confidence(rows)
    examples = [
        LabeledDetectorExample(
            domain="code",
            label=0 if bool(row.get("label")) else 1,
            ensemble_energy=float(ensemble),
            confidence_error=float(confidence),
            example_id=f"code-{row.get('candidate_sha256', idx)}",
        )
        for idx, (row, ensemble, confidence) in enumerate(
            zip(rows, ensemble_scores, confidence_scores, strict=False)
        )
    ]
    return examples, {
        "status": "loaded",
        "path": str(corpus_path),
        "n_examples": len(examples),
        "balanced_exp3658": bool(use_balanced_code_corpus),
        "code_native_exp3695": bool(code_native_scored),
    }


def _score_math_rows(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    from carnot.verify import code_corpus_verifiers_fire_transfer_v3 as exp3641

    wrapped = [{"candidate_code": str(row.get("step_text") or "")} for row in rows]
    return [float(score) for score in exp3641.score_math_signal(wrapped, score_overrides={})]


def _score_code_native_rows(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    scored = code_native_verifier_3695.CodeNativeVerifier().score_rows(rows)
    scores = [float(item.score) for item in scored]
    if len(scores) != len(rows):
        raise ValueError("CodeNativeVerifier returned a mismatched number of scores")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("CodeNativeVerifier scores must be finite")
    return scores


def _read_jsonl(path: Path) -> list[JsonDict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def _clean_examples(examples: Sequence[LabeledDetectorExample]) -> list[LabeledDetectorExample]:
    clean = []
    for example in examples:
        ensemble = float(example.ensemble_energy)
        confidence = float(example.confidence_error)
        if math.isfinite(ensemble) and math.isfinite(confidence):
            clean.append(
                LabeledDetectorExample(
                    domain=str(example.domain),
                    label=1 if int(example.label) else 0,
                    ensemble_energy=ensemble,
                    confidence_error=confidence,
                    example_id=str(example.example_id),
                )
            )
    return clean


def _domain_abstains(domain: str) -> bool:
    return CODE_ABSTAIN_ON_CODE and str(domain).lower() == "code"


def _shipped_detector_examples(
    examples: Sequence[LabeledDetectorExample],
) -> list[LabeledDetectorExample]:
    return [example for example in examples if not _domain_abstains(example.domain)]


def _code_abstention_payload(examples: Sequence[LabeledDetectorExample]) -> JsonDict:
    n_code = sum(1 for example in examples if str(example.domain).lower() == "code")
    return {
        "abstain_on_code": bool(CODE_ABSTAIN_ON_CODE),
        "code_verdict": "no_code_verdict" if CODE_ABSTAIN_ON_CODE else "scored",
        "reason": CODE_ABSTAIN_REASON if CODE_ABSTAIN_ON_CODE else None,
        "n_code_examples_seen": n_code,
    }


def _feature_array(examples: Sequence[LabeledDetectorExample]) -> np.ndarray:
    return np.asarray(
        [[example.ensemble_energy, example.confidence_error] for example in examples],
        dtype=np.float64,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _has_both_classes(examples: Sequence[LabeledDetectorExample]) -> bool:
    return len({example.label for example in examples}) == 2


def _by_domain(
    examples: Sequence[LabeledDetectorExample],
) -> dict[str, list[LabeledDetectorExample]]:
    domains: dict[str, list[LabeledDetectorExample]] = defaultdict(list)
    for example in examples:
        domains[example.domain].append(example)
    return dict(domains)


def _n_examples_per_domain(examples: Sequence[LabeledDetectorExample]) -> dict[str, int]:
    return {domain: len(items) for domain, items in sorted(_by_domain(examples).items())}


def _fpr_key(value: float) -> str:
    return f"{float(value):.2f}"


def _round(value: float) -> float:
    return round(float(value), 6)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path
