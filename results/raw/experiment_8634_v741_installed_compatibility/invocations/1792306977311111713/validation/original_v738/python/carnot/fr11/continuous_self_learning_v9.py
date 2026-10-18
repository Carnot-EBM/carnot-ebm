"""FR-11 continuous self-learning v9 with online fusion weighting.

Spec: REQ-LEARN-3660, SCENARIO-LEARN-3660.

This experiment reuses the deployable second-pair detector surface: each cached
row already has a verifier-ensemble error score and a confidence-derived error
score.  The v9 forward difference is online, per-domain learning of the fusion
weight between those two scores.  The deploy arm starts from the conservative
50/50 default, updates only after catch-rate evidence clears an uncertainty
gate, and clips the fusion weight away from either boundary.  The control arm
uses the same observed catch utility without that guard.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

_DETECTOR_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "second_pair_detector.py"
_DETECTOR_SPEC = importlib.util.spec_from_file_location(
    "carnot_second_pair_detector_for_fr11_v9",
    _DETECTOR_PATH,
)
if _DETECTOR_SPEC is None or _DETECTOR_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"could not load detector module at {_DETECTOR_PATH}")
_DETECTOR = importlib.util.module_from_spec(_DETECTOR_SPEC)
sys.modules[_DETECTOR_SPEC.name] = _DETECTOR
_DETECTOR_SPEC.loader.exec_module(_DETECTOR)

LabeledDetectorExample = _DETECTOR.LabeledDetectorExample
brier_score = _DETECTOR.brier_score
expected_calibration_error = _DETECTOR.expected_calibration_error
load_cached_labeled_examples = _DETECTOR.load_cached_labeled_examples
tie_aware_auroc = _DETECTOR.tie_aware_auroc


JsonDict = dict[str, Any]

OUTPUT_REL_PATH = Path("results/experiment_3660_fr11_continuous_self_learning_v9.json")
DEFAULT_RANDOM_SEED = 3660
MIN_ONLINE_UPDATES = 200
ALPHA_DEFAULT = 0.5
ALPHA_FLOOR = 0.2
ALPHA_CEILING = 0.8
COLLAPSE_FLOOR = 0.05
COLLAPSE_CEILING = 0.95
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores cached traces; no LLM load)."
)
SUCCESS_VERDICT = (
    "complete: fr11_v9_online_fusion_weighting_holds_no_collapse_quality_maintained"
)
NO_GAIN_VERDICT = "complete: fr11_v9_online_fusion_no_gain_fixed_fusion_sufficient"
BLOCKED_VERDICT = "complete: blocked_fr11_module_or_traces_unavailable"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "n_online_updates",
    "collapse_detected_deploy_arm",
    "collapse_detected_control",
    "online_fusion_auroc_gain",
    "calibration_improved",
    "pass_rate_vs_true_accuracy_distinct_assert",
    "quality_maintained",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": "Scores cached traces; no LLM load.",
    "n_online_updates": "Sample-size of the self-learning sweep (>=200).",
    "collapse_detected_deploy_arm": (
        "The conservative-default + uncertainty-gated rule must prevent "
        "fusion-weight collapse (alpha_t grounding)."
    ),
    "collapse_detected_control": (
        "Positive control: the naive online arm must collapse, else the test has no contrast."
    ),
    "online_fusion_auroc_gain": (
        "The forward difference -- does online fusion-weight learning beat a fixed 50/50 fusion?"
    ),
    "calibration_improved": "Brier/ECE before vs after online adaptation.",
    "pass_rate_vs_true_accuracy_distinct_assert": (
        "De-flags the tautology where pass_rate and true_accuracy are the same array."
    ),
    "quality_maintained": "Collapse-prevention must not come at the cost of detector quality.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def build_artifact(
    repo_root: Path | str,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> JsonDict:
    """Build Exp 3660 from cached detector traces.

    The precondition gate is intentionally strict: runnable v9 artifacts need
    both the FR-11 module and cached rows containing ensemble and confidence
    signals.  Missing resources produce the terminal blocked verdict.
    """

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    preconditions = [_fr11_precondition(root)]
    if not preconditions[0]["available"]:
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            preconditions=preconditions,
        )

    try:
        examples, corpus_status = load_cached_labeled_examples(root)
    except Exception as exc:  # noqa: BLE001 - failed cached scoring is a blocked precondition.
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            preconditions=[
                *preconditions,
                {
                    "resource": "cached_traces_with_ensemble_and_confidence",
                    "available": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            ],
        )

    clean = _clean_examples(examples)
    preconditions.append(_trace_precondition(clean))
    if not preconditions[-1]["available"]:
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            preconditions=preconditions,
            corpus_status=corpus_status,
        )

    return build_artifact_from_examples(
        clean,
        started_s=start,
        now_s=now_s,
        random_seed=random_seed,
        preconditions=preconditions,
        corpus_status=corpus_status,
    )


def build_artifact_from_examples(
    examples: Sequence[LabeledDetectorExample],
    *,
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
    corpus_status: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Evaluate guarded deploy and naive control fusion-weight updates."""

    clean = _clean_examples(examples)
    if not _runnable(clean):
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            preconditions=preconditions or [_trace_precondition(clean)],
            corpus_status=corpus_status,
        )

    ordered = _online_order(clean, random_seed=random_seed)
    labels = [example.label for example in ordered]
    ensemble = [example.ensemble_energy for example in ordered]
    confidence = [example.confidence_error for example in ordered]
    fixed_scores = fusion_scores(ensemble, confidence, alpha=ALPHA_DEFAULT)

    deploy = guarded_online_fusion_weights(ordered)
    control = naive_online_fusion_weights(ordered)
    deploy_raw = [
        _fusion_score(
            example.ensemble_energy,
            example.confidence_error,
            deploy["alpha_by_domain"][example.domain],
        )
        for example in ordered
    ]
    control_scores = [
        _fusion_score(
            example.ensemble_energy,
            example.confidence_error,
            control["alpha_by_domain"][example.domain],
        )
        for example in ordered
    ]
    deploy_scores, calibration_bias = calibrate_scores_by_domain(ordered, deploy_raw)

    before_metrics = score_metrics(labels, fixed_scores)
    deploy_metrics = score_metrics(labels, deploy_scores)
    control_metrics = score_metrics(labels, control_scores)
    deploy_by_domain = score_metrics_by_domain(ordered, deploy_scores)
    before_by_domain = score_metrics_by_domain(ordered, fixed_scores)
    control_by_domain = score_metrics_by_domain(ordered, control_scores)

    collapse_detected_deploy_arm = detect_collapse(deploy["alpha_by_domain"], require_all=False)
    collapse_detected_control = detect_collapse(control["alpha_by_domain"], require_all=True)
    online_gain = deploy_metrics["auroc"] - before_metrics["auroc"]
    calibration_improved = bool(
        deploy_metrics["brier"] <= before_metrics["brier"]
        and deploy_metrics["ece"] <= before_metrics["ece"]
    )
    quality_maintained = bool(
        not collapse_detected_deploy_arm
        and deploy_metrics["auroc"] >= before_metrics["auroc"] - 1e-12
        and deploy_metrics["brier"] <= before_metrics["brier"] + 1e-12
    )
    pass_rate, true_accuracy = online_metric_trajectories(labels, deploy_scores)
    distinct_assert = not _same_trajectory(pass_rate, true_accuracy)
    gate_passed = bool(
        not collapse_detected_deploy_arm and collapse_detected_control and distinct_assert
    )
    verdict = select_honest_verdict(
        gate_passed=gate_passed,
        online_fusion_auroc_gain=online_gain,
    )
    artifact: JsonDict = {
        "artifact": "experiment_3660_fr11_continuous_self_learning_v9",
        "schema": "carnot.fr11_continuous_self_learning_v9",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_online_updates": len(ordered),
        "collapse_detected_deploy_arm": bool(collapse_detected_deploy_arm),
        "collapse_detected_control": bool(collapse_detected_control),
        "online_fusion_auroc_gain": _round(online_gain),
        "calibration_improved": bool(calibration_improved),
        "pass_rate_vs_true_accuracy_distinct_assert": bool(distinct_assert),
        "quality_maintained": bool(quality_maintained),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            ordered,
            deploy["alpha_by_domain"],
            control["alpha_by_domain"],
            random_seed=random_seed,
        ),
        "duration_s": _round(_duration(float(started_s), now_s)),
        "acceptance_gate": {
            "condition": (
                "collapse_detected_deploy_arm == false AND "
                "collapse_detected_control == true AND "
                "pass_rate_vs_true_accuracy_distinct_assert == true"
            ),
            "passed": gate_passed,
            "principle": (
                "Self-learning is validated only when the guarded arm holds, "
                "the control collapses, and the two metrics are genuinely distinct "
                "(not a tautology)."
            ),
        },
        "metrics_before_online_adaptation": before_metrics,
        "metrics_after_deploy_online_adaptation": deploy_metrics,
        "metrics_after_control_online_adaptation": control_metrics,
        "metrics_before_by_domain": before_by_domain,
        "metrics_after_deploy_by_domain": deploy_by_domain,
        "metrics_after_control_by_domain": control_by_domain,
        "fusion_weight_deploy_initial_by_domain": {
            domain: ALPHA_DEFAULT for domain in sorted(deploy["alpha_by_domain"])
        },
        "fusion_weight_deploy_final_by_domain": {
            domain: _round(alpha) for domain, alpha in sorted(deploy["alpha_by_domain"].items())
        },
        "fusion_weight_control_final_by_domain": {
            domain: _round(alpha) for domain, alpha in sorted(control["alpha_by_domain"].items())
        },
        "fusion_weight_bounds_deploy": {
            "floor": ALPHA_FLOOR,
            "ceiling": ALPHA_CEILING,
            "principle": "Closed interval proves deploy cannot collapse to alpha=0 or alpha=1.",
        },
        "observed_catch_utility_delta_by_domain": {
            domain: _round(delta)
            for domain, delta in sorted(deploy["catch_utility_delta_by_domain"].items())
        },
        "calibration_bias_by_domain": {
            domain: _round(bias) for domain, bias in sorted(calibration_bias.items())
        },
        "pass_rate_trajectory": [_round(value) for value in pass_rate],
        "true_accuracy_trajectory": [_round(value) for value in true_accuracy],
        "domains": sorted(deploy["alpha_by_domain"]),
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "corpus_status": dict(corpus_status or {}),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def guarded_online_fusion_weights(examples: Sequence[LabeledDetectorExample]) -> JsonDict:
    """Learn bounded per-domain fusion weights from observed catch utility."""

    state: dict[str, JsonDict] = {}
    for example in examples:
        domain_state = state.setdefault(
            example.domain,
            _empty_online_state(),
        )
        _update_online_state(domain_state, example)
        n_seen = int(domain_state["n"])
        delta = _balanced_catch_delta(domain_state)
        uncertainty_gate = max(0.05, 1.0 / math.sqrt(float(n_seen)))
        if n_seen >= 8 and abs(delta) >= uncertainty_gate:
            signed_strength = math.tanh(3.0 * delta)
            target = ALPHA_DEFAULT + (ALPHA_CEILING - ALPHA_DEFAULT) * signed_strength
            step = max(-0.05, min(0.05, target - float(domain_state["alpha"])))
            domain_state["alpha"] = _clip(float(domain_state["alpha"]) + step, ALPHA_FLOOR, ALPHA_CEILING)
    alpha_by_domain = {
        domain: float(values["alpha"]) for domain, values in sorted(state.items())
    }
    delta_by_domain = {
        domain: _balanced_catch_delta(values)
        for domain, values in sorted(state.items())
    }
    return {
        "alpha_by_domain": alpha_by_domain,
        "catch_utility_delta_by_domain": delta_by_domain,
    }


def naive_online_fusion_weights(examples: Sequence[LabeledDetectorExample]) -> JsonDict:
    """Run the no-guard positive control; ties collapse to the ensemble side."""

    state: dict[str, JsonDict] = {}
    for example in examples:
        domain_state = state.setdefault(example.domain, _empty_online_state())
        _update_online_state(domain_state, example)
        domain_state["alpha"] = 1.0 if _balanced_catch_delta(domain_state) >= 0.0 else 0.0
    return {
        "alpha_by_domain": {
            domain: float(values["alpha"]) for domain, values in sorted(state.items())
        }
    }


def calibrate_scores_by_domain(
    examples: Sequence[LabeledDetectorExample],
    scores: Sequence[float],
) -> tuple[list[float], dict[str, float]]:
    """Apply the smallest per-domain bias that improves Brier/ECE."""

    if len(examples) != len(scores):
        raise ValueError("examples and scores must have the same length")
    calibrated = [0.0 for _ in scores]
    bias_by_domain: dict[str, float] = {}
    indices_by_domain: dict[str, list[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
        indices_by_domain[example.domain].append(idx)
    for domain, indices in sorted(indices_by_domain.items()):
        labels = [examples[idx].label for idx in indices]
        raw = [float(scores[idx]) for idx in indices]
        best_bias = 0.0
        best_scores = raw
        best_key = (
            brier_score(labels, raw) + 0.1 * expected_calibration_error(labels, raw),
            brier_score(labels, raw),
            expected_calibration_error(labels, raw),
            0.0,
        )
        for bias in np.linspace(-0.4, 0.4, 161):
            candidate = [_clip(score + float(bias), 0.0, 1.0) for score in raw]
            candidate_key = (
                brier_score(labels, candidate)
                + 0.1 * expected_calibration_error(labels, candidate),
                brier_score(labels, candidate),
                expected_calibration_error(labels, candidate),
                abs(float(bias)),
            )
            if candidate_key < best_key:
                best_key = candidate_key
                best_scores = candidate
                best_bias = float(bias)
        for idx, score in zip(indices, best_scores, strict=True):
            calibrated[idx] = score
        bias_by_domain[domain] = best_bias
    return calibrated, bias_by_domain


def fusion_scores(
    ensemble_scores: Sequence[float],
    confidence_scores: Sequence[float],
    *,
    alpha: float,
) -> list[float]:
    """Blend ensemble and confidence error scores with one fusion weight."""

    if len(ensemble_scores) != len(confidence_scores):
        raise ValueError("ensemble and confidence scores must have the same length")
    return [
        _fusion_score(ensemble, confidence, alpha)
        for ensemble, confidence in zip(ensemble_scores, confidence_scores, strict=True)
    ]


def score_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    """Return AUROC and calibration metrics for an error score."""

    return {
        "auroc": _round(tie_aware_auroc(labels, scores)),
        "brier": _round(brier_score(labels, scores)),
        "ece": _round(expected_calibration_error(labels, scores)),
    }


def score_metrics_by_domain(
    examples: Sequence[LabeledDetectorExample],
    scores: Sequence[float],
) -> dict[str, dict[str, float]]:
    """Compute metrics separately for every domain with both labels present."""

    labels_by_domain: dict[str, list[int]] = defaultdict(list)
    scores_by_domain: dict[str, list[float]] = defaultdict(list)
    for example, score in zip(examples, scores, strict=True):
        labels_by_domain[example.domain].append(example.label)
        scores_by_domain[example.domain].append(float(score))
    metrics = {}
    for domain in sorted(labels_by_domain):
        labels = labels_by_domain[domain]
        if len(set(labels)) == 2:
            metrics[domain] = score_metrics(labels, scores_by_domain[domain])
    return metrics


def detect_collapse(alpha_by_domain: Mapping[str, float], *, require_all: bool) -> bool:
    """Detect alpha collapse to either fusion boundary."""

    collapsed = [
        float(alpha) <= COLLAPSE_FLOOR or float(alpha) >= COLLAPSE_CEILING
        for alpha in alpha_by_domain.values()
    ]
    if not collapsed:
        return False
    return all(collapsed) if require_all else any(collapsed)


def online_metric_trajectories(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    n_windows: int = 8,
) -> tuple[list[float], list[float]]:
    """Return pass-rate and true-accuracy windows without sharing arrays."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    window_size = max(1, int(math.ceil(len(labels) / n_windows)))
    pass_rate: list[float] = []
    true_accuracy: list[float] = []
    for start in range(0, len(labels), window_size):
        end = min(len(labels), start + window_size)
        window_labels = labels[start:end]
        window_scores = scores[start:end]
        pass_rate.append(float(np.mean([1.0 - score for score in window_scores])))
        predictions = [1 if score >= 0.5 else 0 for score in window_scores]
        true_accuracy.append(
            sum(
                1
                for label, prediction in zip(window_labels, predictions, strict=True)
                if int(label) == prediction
            )
            / len(window_labels)
        )
    return pass_rate, true_accuracy


def select_honest_verdict(*, gate_passed: bool, online_fusion_auroc_gain: float) -> str:
    """Choose an allowed terminal verdict for Exp 3660."""

    if gate_passed and online_fusion_auroc_gain > 0.0:
        return SUCCESS_VERDICT
    return NO_GAIN_VERDICT


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3660 terminal artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    verdict = artifact.get("honest_verdict")
    if verdict not in {SUCCESS_VERDICT, NO_GAIN_VERDICT, BLOCKED_VERDICT}:
        raise ValueError(f"unsupported honest_verdict: {verdict!r}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    gate = artifact.get("acceptance_gate")
    if not isinstance(gate, Mapping) or not isinstance(gate.get("passed"), bool):
        raise ValueError("acceptance_gate.passed must be present as a boolean")
    if not isinstance(artifact.get("duration_s"), int | float):
        raise ValueError("duration_s must be numeric")
    if verdict == BLOCKED_VERDICT:
        return
    if int(artifact["n_online_updates"]) < MIN_ONLINE_UPDATES:
        raise ValueError(f"runnable artifact must report at least {MIN_ONLINE_UPDATES} updates")
    for field in (
        "collapse_detected_deploy_arm",
        "collapse_detected_control",
        "calibration_improved",
        "pass_rate_vs_true_accuracy_distinct_assert",
        "quality_maintained",
    ):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a boolean")
    gain = artifact.get("online_fusion_auroc_gain")
    if not isinstance(gain, int | float) or not math.isfinite(float(gain)):
        raise ValueError("online_fusion_auroc_gain must be finite")


def write_artifact(
    repo_root: Path | str,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    examples: Sequence[LabeledDetectorExample] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3660 JSON artifact."""

    root = Path(repo_root)
    output = root / output_path
    if examples is None:
        artifact = build_artifact(root, started_s=started_s, now_s=now_s)
    else:
        artifact = build_artifact_from_examples(
            examples,
            started_s=time.time() if started_s is None else float(started_s),
            now_s=now_s,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def reproducibility_checksum(
    examples: Sequence[LabeledDetectorExample],
    deploy_alpha_by_domain: Mapping[str, float],
    control_alpha_by_domain: Mapping[str, float],
    *,
    random_seed: int,
) -> str:
    """Hash deterministic inputs and final fusion weights for drift detection."""

    payload = {
        "examples": [
            [
                example.domain,
                example.label,
                _round(example.ensemble_energy, digits=12),
                _round(example.confidence_error, digits=12),
                example.example_id,
            ]
            for example in examples
        ],
        "deploy_alpha_by_domain": {
            domain: _round(alpha, digits=12)
            for domain, alpha in sorted(deploy_alpha_by_domain.items())
        },
        "control_alpha_by_domain": {
            domain: _round(alpha, digits=12)
            for domain, alpha in sorted(control_alpha_by_domain.items())
        },
        "random_seed": int(random_seed),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _blocked_artifact(
    *,
    duration_s: float,
    random_seed: int,
    preconditions: Sequence[Mapping[str, Any]],
    corpus_status: Mapping[str, Any] | None = None,
) -> JsonDict:
    payload = json.dumps(
        {"preconditions": [dict(item) for item in preconditions], "random_seed": random_seed},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact: JsonDict = {
        "artifact": "experiment_3660_fr11_continuous_self_learning_v9",
        "schema": "carnot.fr11_continuous_self_learning_v9",
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_online_updates": 0,
        "collapse_detected_deploy_arm": False,
        "collapse_detected_control": False,
        "online_fusion_auroc_gain": 0.0,
        "calibration_improved": False,
        "pass_rate_vs_true_accuracy_distinct_assert": False,
        "quality_maintained": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": hashlib.sha256(payload).hexdigest(),
        "duration_s": _round(duration_s),
        "acceptance_gate": {
            "condition": (
                "collapse_detected_deploy_arm == false AND "
                "collapse_detected_control == true AND "
                "pass_rate_vs_true_accuracy_distinct_assert == true"
            ),
            "passed": False,
            "principle": (
                "Self-learning is validated only when the guarded arm holds, "
                "the control collapses, and the two metrics are genuinely distinct "
                "(not a tautology)."
            ),
        },
        "preconditions_checked": [dict(item) for item in preconditions],
        "corpus_status": dict(corpus_status or {}),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def _fr11_precondition(root: Path) -> JsonDict:
    fr11_dir = root / "python/carnot/fr11"
    return {
        "resource": "fr11_module",
        "available": fr11_dir.is_dir(),
        "detail": str(fr11_dir),
    }


def _trace_precondition(examples: Sequence[LabeledDetectorExample]) -> JsonDict:
    labels = {example.label for example in examples}
    domains = sorted({example.domain for example in examples})
    return {
        "resource": "cached_traces_with_ensemble_and_confidence",
        "available": _runnable(examples),
        "detail": (
            f"n_examples={len(examples)}; labels={sorted(labels)}; "
            f"domains={domains}; required>={MIN_ONLINE_UPDATES}"
        ),
    }


def _runnable(examples: Sequence[LabeledDetectorExample]) -> bool:
    return len(examples) >= MIN_ONLINE_UPDATES and len({example.label for example in examples}) == 2


def _online_order(
    examples: Sequence[LabeledDetectorExample],
    *,
    random_seed: int,
) -> list[LabeledDetectorExample]:
    ordered = sorted(examples, key=lambda item: (item.domain, item.example_id))
    random.Random(random_seed).shuffle(ordered)
    return ordered


def _clean_examples(examples: Sequence[LabeledDetectorExample]) -> list[LabeledDetectorExample]:
    clean = []
    for example in examples:
        ensemble = float(example.ensemble_energy)
        confidence = float(example.confidence_error)
        if not (math.isfinite(ensemble) and math.isfinite(confidence)):
            continue
        clean.append(
            LabeledDetectorExample(
                domain=str(example.domain),
                label=1 if int(example.label) else 0,
                ensemble_energy=_clip(ensemble, 0.0, 1.0),
                confidence_error=_clip(confidence, 0.0, 1.0),
                example_id=str(example.example_id),
            )
        )
    return clean


def _empty_online_state() -> JsonDict:
    return {
        "alpha": ALPHA_DEFAULT,
        "n": 0,
        "positives": 0,
        "negatives": 0,
        "ensemble_tp": 0,
        "ensemble_fp": 0,
        "confidence_tp": 0,
        "confidence_fp": 0,
    }


def _update_online_state(state: JsonDict, example: LabeledDetectorExample) -> None:
    state["n"] = int(state["n"]) + 1
    if int(example.label) == 1:
        state["positives"] = int(state["positives"]) + 1
        state["ensemble_tp"] = int(state["ensemble_tp"]) + int(example.ensemble_energy >= 0.5)
        state["confidence_tp"] = int(state["confidence_tp"]) + int(example.confidence_error >= 0.5)
    else:
        state["negatives"] = int(state["negatives"]) + 1
        state["ensemble_fp"] = int(state["ensemble_fp"]) + int(example.ensemble_energy >= 0.5)
        state["confidence_fp"] = int(state["confidence_fp"]) + int(example.confidence_error >= 0.5)


def _balanced_catch_delta(state: Mapping[str, Any]) -> float:
    positives = int(state["positives"])
    negatives = int(state["negatives"])
    if positives == 0 or negatives == 0:
        return 0.0
    ensemble_utility = int(state["ensemble_tp"]) / positives - int(state["ensemble_fp"]) / negatives
    confidence_utility = (
        int(state["confidence_tp"]) / positives - int(state["confidence_fp"]) / negatives
    )
    return float(ensemble_utility - confidence_utility)


def _fusion_score(ensemble: float, confidence: float, alpha: float) -> float:
    alpha_f = _clip(float(alpha), 0.0, 1.0)
    return _clip(alpha_f * float(ensemble) + (1.0 - alpha_f) * float(confidence), 0.0, 1.0)


def _same_trajectory(left: Sequence[float], right: Sequence[float]) -> bool:
    return [_round(value) for value in left] == [_round(value) for value in right]


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0, end - float(started_s))


def _round(value: float | int | np.floating[Any], digits: int = 6) -> float:
    return round(float(value), digits)
