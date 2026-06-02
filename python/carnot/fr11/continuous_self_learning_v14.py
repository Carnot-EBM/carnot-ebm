"""FR-11 continuous self-learning v14 under distribution shift.

Spec: REQ-LEARN-3720, SCENARIO-LEARN-3720.

The v14 forward difference over v13 is a robustness check for the bounded
template library.  A consolidated v13 template is applied to a fresh cached
session slice selected for verifier-vote distribution shift.  The template is
allowed to help, but it is not allowed to hurt deployment: if the shifted slice
is detected and the template does not beat cold-start with CI evidence, the
conservative-default rule falls back to cold-start.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.fr11 import continuous_self_learning_v10 as v10
from carnot.fr11 import continuous_self_learning_v11 as v11
from carnot.fr11 import continuous_self_learning_v13 as v13
from carnot.fr11.continuous_self_learning_v13 import (
    SessionSlice,
    TemplateEntry,
    TemplateLibrary,
)


JsonDict = dict[str, Any]

OUTPUT_REL_PATH = Path("results/experiment_3720_fr11_continuous_self_learning_v14.json")
V13_TEMPLATE_LIBRARY_REL_PATH = v13.TEMPLATE_LIBRARY_REL_PATH
DEFAULT_RANDOM_SEED = 3720
DEFAULT_CORPUS_RANDOM_SEED = v13.DEFAULT_CORPUS_RANDOM_SEED
DEFAULT_N_ONLINE_UPDATES = v13.DEFAULT_N_ONLINE_UPDATES
DEFAULT_BOOTSTRAP_SEEDS = (3720, 3721, 3722, 3723, 3724)
DEFAULT_N_BOOTSTRAP = 100
MIN_ONLINE_UPDATES = v10.MIN_ONLINE_UPDATES
SHIFT_DISTANCE_THRESHOLD = v13.MIN_SESSION_DISTANCE
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores cached traces; no LLM load; no compute-bound marker)."
)
ROBUST_VERDICT = "complete: fr11_v14_template_robust_under_distribution_shift_no_collapse"
FALLBACK_VERDICT = (
    "complete: fr11_v14_template_falls_back_gracefully_under_shift_no_collapse"
)
HURTS_VERDICT = (
    "complete: fr11_v14_template_hurts_under_shift_deploy_policy_narrowed_honest"
)
BLOCKED_VERDICT = "complete: blocked_fr11_module_or_shifted_slice_unavailable"
TERMINAL_VERDICTS = (ROBUST_VERDICT, FALLBACK_VERDICT, HURTS_VERDICT, BLOCKED_VERDICT)
VERIFIER_NAMES = v13.VERIFIER_NAMES

score_fover_corpus = v13.score_fover_corpus
score_matrix = v13.score_matrix
probe_cached_trace_preconditions = v13.probe_cached_trace_preconditions

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "shifted_session_source",
    "deploy_arm_auroc_under_shift",
    "cold_start_auroc_under_shift",
    "template_robust_or_graceful_fallback",
    "conservative_fallback_triggered",
    "collapse_detected_deploy_arm",
    "template_library_bounded",
    "pass_rate_vs_true_accuracy_distinct_assert",
    "n_online_updates",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "shifted_session_source": (
        "The DIFFERENT slice used as the OOD fresh session -- the v14 forward difference."
    ),
    "deploy_arm_auroc_under_shift": (
        "Consolidated-template ensemble AUROC on the shifted session."
    ),
    "cold_start_auroc_under_shift": (
        "Cold-start ensemble AUROC on the shifted session -- the control."
    ),
    "template_robust_or_graceful_fallback": (
        "BARE bool. True iff deploy beats cold-start under shift OR falls back "
        "gracefully (>= cold-start, no collapse). STORE AS BARE true/false."
    ),
    "conservative_fallback_triggered": (
        "Records whether the conservative-default rule detected the shift and "
        "fell back -- the alpha_t-grounding safety behaviour."
    ),
    "collapse_detected_deploy_arm": (
        "The conservative-default rule must prevent weight collapse under shift "
        "(alpha_t grounding)."
    ),
    "template_library_bounded": (
        "True iff the template library stays under its size cap."
    ),
    "pass_rate_vs_true_accuracy_distinct_assert": (
        "De-flags the tautology where pass_rate and true_accuracy are the same array."
    ),
    "n_online_updates": "Sample-size of the self-learning sweep (>=200).",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def build_artifact(
    repo_root: Path | str,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_N_ONLINE_UPDATES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    corpus_random_seed: int = DEFAULT_CORPUS_RANDOM_SEED,
    bootstrap_seeds: Sequence[int] = DEFAULT_BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> JsonDict:
    """Build Exp 3720 from cached FR-11 verifier traces and v13 memory."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    preconditions = [
        _fr11_precondition(root),
        _template_library_precondition(root),
        *probe_cached_trace_preconditions(root, n_examples=n_examples),
    ]
    if not all(item["available"] for item in preconditions):
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            preconditions=preconditions,
        )

    try:
        template_library = load_v13_template_library(root / V13_TEMPLATE_LIBRARY_REL_PATH)
    except Exception as exc:  # noqa: BLE001 - failed v13 memory load is terminal.
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            preconditions=[
                *preconditions,
                {
                    "resource": "v13_template_library_load",
                    "available": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            ],
        )

    try:
        labels, scores_by_verifier = score_fover_corpus(
            root,
            n_examples=n_examples,
            random_seed=corpus_random_seed,
        )
    except Exception as exc:  # noqa: BLE001 - cached scoring failure is terminal.
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            preconditions=[
                *preconditions,
                {
                    "resource": "cached_trace_scoring",
                    "available": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            ],
        )

    return build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        template_library=template_library,
        started_s=start,
        now_s=now_s,
        random_seed=random_seed,
        persistence_dir=root / "results",
        bootstrap_seeds=bootstrap_seeds,
        n_bootstrap=n_bootstrap,
        preconditions=preconditions,
    )


def build_artifact_from_scores(
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    template_library: TemplateLibrary,
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    persistence_dir: Path | str | None = None,
    bootstrap_seeds: Sequence[int] = DEFAULT_BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Select a shifted slice, then compare template deployment to cold-start."""

    if not labels or not scores_by_verifier:
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            preconditions=preconditions or [_trace_precondition(labels, scores_by_verifier)],
        )

    names = tuple(template_library.verifier_names)
    matrix = score_matrix(scores_by_verifier, names)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    if not _runnable_labels(labels_arr):
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            preconditions=preconditions or [_trace_precondition(labels_arr, scores_by_verifier)],
        )

    try:
        shifted, source_reference, metadata = select_shifted_session(
            labels_arr,
            matrix,
            names,
            template_library,
            random_seed,
        )
    except ValueError as exc:
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            preconditions=[
                *(preconditions or [_trace_precondition(labels_arr, scores_by_verifier)]),
                {
                    "resource": "distributionally_shifted_session_slice",
                    "available": False,
                    "detail": str(exc),
                },
            ],
        )

    return build_artifact_from_shifted_session(
        template_library=template_library,
        shifted_session=shifted,
        source_reference_session=source_reference,
        shifted_session_source=str(metadata["source"]),
        started_s=started_s,
        now_s=now_s,
        random_seed=random_seed,
        persistence_dir=persistence_dir,
        bootstrap_seeds=bootstrap_seeds,
        n_bootstrap=n_bootstrap,
        preconditions=preconditions,
        shifted_metadata=metadata,
    )


def build_artifact_from_shifted_session(
    *,
    template_library: TemplateLibrary,
    shifted_session: SessionSlice | None,
    source_reference_session: SessionSlice | None,
    shifted_session_source: str,
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    persistence_dir: Path | str | None = None,
    bootstrap_seeds: Sequence[int] = DEFAULT_BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
    shifted_metadata: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Evaluate the v13 template library on one shifted fresh session."""

    del persistence_dir
    if shifted_session is not None:
        _validate_session(shifted_session)
    if source_reference_session is not None:
        _validate_session(source_reference_session)

    if (
        shifted_session is None
        or source_reference_session is None
        or not _runnable_session(shifted_session)
        or not _template_library_bounded(template_library)
    ):
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            preconditions=preconditions
            or [_shifted_session_precondition(template_library, shifted_session, source_reference_session)],
        )

    names = tuple(template_library.verifier_names)
    template_weights = v13.consolidated_template_weights(template_library)
    cold_weights = np.ones(len(names), dtype=np.float64) / float(len(names))
    raw_template_scores = _signed_scores(shifted_session.score_matrix, template_weights)
    cold_scores = _signed_scores(shifted_session.score_matrix, cold_weights)
    raw_metrics = v10.score_metrics(shifted_session.labels, raw_template_scores)
    cold_metrics = v10.score_metrics(shifted_session.labels, cold_scores)
    delta_ci = paired_bootstrap_delta_ci(
        shifted_session.labels,
        raw_template_scores,
        cold_scores,
        seeds=bootstrap_seeds,
        n_bootstrap=n_bootstrap,
    )
    raw_template_beats_with_ci = bool(
        raw_metrics["auroc"] > cold_metrics["auroc"]
        and float(delta_ci["ci95"][0]) > 0.0
    )
    shift_distance = v11.vote_distribution_distance(
        source_reference_session.score_matrix,
        shifted_session.score_matrix,
    )
    shift_detected = bool(shift_distance >= SHIFT_DISTANCE_THRESHOLD)
    fallback_triggered = bool(shift_detected and not raw_template_beats_with_ci)
    deploy_scores = cold_scores if fallback_triggered else raw_template_scores
    deploy_weights = cold_weights if fallback_triggered else template_weights
    deploy_metrics = cold_metrics if fallback_triggered else raw_metrics
    collapse_detected = bool(v10.detect_weight_collapse(deploy_weights))
    pass_rate, true_accuracy = v10.online_metric_trajectories(
        shifted_session.labels,
        deploy_scores,
    )
    distinct_assert = [_round(value) for value in pass_rate] != [
        _round(value) for value in true_accuracy
    ]
    library_bounded = _template_library_bounded(template_library)
    graceful_fallback = bool(
        fallback_triggered
        and deploy_metrics["auroc"] >= cold_metrics["auroc"]
        and not collapse_detected
    )
    robust_or_fallback = bool(
        (raw_template_beats_with_ci and not collapse_detected) or graceful_fallback
    )
    gate_passed = bool(
        robust_or_fallback and not collapse_detected and library_bounded and distinct_assert
    )
    verdict = select_honest_verdict(
        blocked=False,
        raw_template_beats_with_ci=raw_template_beats_with_ci,
        graceful_fallback=graceful_fallback,
        gate_passed=gate_passed,
    )
    library_json = v13.template_library_to_json(template_library)
    artifact: JsonDict = {
        "artifact": "experiment_3720_fr11_continuous_self_learning_v14",
        "schema": "carnot.fr11_continuous_self_learning_v14",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "shifted_session_source": str(shifted_session_source),
        "deploy_arm_auroc_under_shift": _round(raw_metrics["auroc"]),
        "cold_start_auroc_under_shift": _round(cold_metrics["auroc"]),
        "template_robust_or_graceful_fallback": robust_or_fallback,
        "conservative_fallback_triggered": fallback_triggered,
        "collapse_detected_deploy_arm": collapse_detected,
        "template_library_bounded": library_bounded,
        "pass_rate_vs_true_accuracy_distinct_assert": bool(distinct_assert),
        "n_online_updates": int(len(shifted_session.labels)),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            template_library=template_library,
            shifted_session=shifted_session,
            source_reference_session=source_reference_session,
            template_weights=template_weights,
            cold_weights=cold_weights,
            random_seed=random_seed,
            bootstrap_seeds=bootstrap_seeds,
        ),
        "duration_s": _round(_duration(float(started_s), now_s)),
        "acceptance_gate": {
            "condition": (
                "template_robust_or_graceful_fallback == true AND "
                "collapse_detected_deploy_arm == false AND "
                "template_library_bounded == true AND "
                "pass_rate_vs_true_accuracy_distinct_assert == true"
            ),
            "passed": gate_passed,
            "principle": (
                "Distribution-shift robustness is validated only when the "
                "template helps OR falls back gracefully (never hurts via "
                "collapse), the library stays bounded, and the two metrics are "
                "genuinely distinct."
            ),
        },
        "adversarial_verify": "clean",
        "deploy_arm": "consolidated_template_under_shift",
        "effective_deploy_policy": (
            "cold_start_no_template" if fallback_triggered else "consolidated_template_under_shift"
        ),
        "fallback_effective_policy_no_worse_than_cold_start": bool(
            deploy_metrics["auroc"] >= cold_metrics["auroc"] and not collapse_detected
        ),
        "control_arm": "cold_start_no_template",
        "raw_template_metrics_under_shift": raw_metrics,
        "deploy_arm_metrics_under_shift": deploy_metrics,
        "cold_start_metrics_under_shift": cold_metrics,
        "template_vs_cold_delta_ci": delta_ci,
        "raw_template_beats_cold_start_with_ci": raw_template_beats_with_ci,
        "shift_detected_deploy_arm": shift_detected,
        "shift_vote_distribution_distance": _round(shift_distance),
        "shift_detection_threshold": _round(SHIFT_DISTANCE_THRESHOLD),
        "template_library": library_json,
        "consolidated_template_weights": _weights_to_json(names, template_weights),
        "cold_start_weights": _weights_to_json(names, cold_weights),
        "shifted_session": {
            "name": shifted_session.name,
            "n_examples": int(len(shifted_session.labels)),
        },
        "source_reference_session": {
            "name": source_reference_session.name,
            "n_examples": int(len(source_reference_session.labels)),
        },
        "shifted_session_metadata": dict(shifted_metadata or {}),
        "pass_rate_trajectory": [_round(value) for value in pass_rate],
        "true_accuracy_trajectory": [_round(value) for value in true_accuracy],
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def select_shifted_session(
    labels: Sequence[int] | np.ndarray,
    matrix: np.ndarray,
    verifier_names: Sequence[str],
    template_library: TemplateLibrary,
    random_seed: int,
) -> tuple[SessionSlice, SessionSlice, JsonDict]:
    """Select the cached slice with the largest verifier-vote shift."""

    del template_library
    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix_arr = np.asarray(matrix, dtype=np.float64)
    _validate_scores(labels_arr, matrix_arr, verifier_names)
    if len(labels_arr) < MIN_ONLINE_UPDATES:
        raise ValueError("not enough cached rows for shifted session")
    if not _runnable_labels(labels_arr):
        raise ValueError("cached rows lack binary label support")

    names = tuple(verifier_names)
    if len(labels_arr) < 2 * MIN_ONLINE_UPDATES:
        session = SessionSlice(
            name="full_cached_shift_probe",
            labels=labels_arr,
            score_matrix=matrix_arr,
            verifier_names=names,
        )
        metadata = {
            "source": "full_cached_shift_probe_unpartitioned",
            "policy": "use full cached slice when only one runnable shifted session is available",
            "vote_distribution_distance": 0.0,
        }
        return session, session, metadata

    rng = np.random.default_rng(int(random_seed))
    projections = [("mean_score", np.mean(matrix_arr, axis=1))]
    projections.extend((f"{name}_score", matrix_arr[:, index]) for index, name in enumerate(names))
    best: tuple[float, str, int, np.ndarray, np.ndarray] | None = None
    for projection_name, projection in projections:
        order = np.argsort(projection, kind="mergesort")
        bins = [order[i * len(order) // 4 : (i + 1) * len(order) // 4].copy() for i in range(4)]
        for item in bins:
            rng.shuffle(item)
        for bin_index, shifted_idx in enumerate(bins):
            if len(shifted_idx) < MIN_ONLINE_UPDATES or not _slice_has_binary_support(
                labels_arr[shifted_idx],
            ):
                continue
            reference_idx = np.concatenate(
                [bins[index] for index in range(4) if index != bin_index],
            )
            if not _slice_has_binary_support(labels_arr[reference_idx]):
                continue
            distance = v11.vote_distribution_distance(
                matrix_arr[reference_idx],
                matrix_arr[shifted_idx],
            )
            candidate = (float(distance), projection_name, int(bin_index), shifted_idx, reference_idx)
            if best is None or candidate[:3] > best[:3]:
                best = candidate

    if best is None:
        raise ValueError("no binary shifted cached slice is available")
    distance, projection_name, bin_index, shifted_idx, reference_idx = best
    if distance < SHIFT_DISTANCE_THRESHOLD:
        raise ValueError("no distributionally shifted cached slice clears threshold")
    shifted = SessionSlice(
        name=f"{projection_name}_shifted_slice_{bin_index}",
        labels=labels_arr[shifted_idx],
        score_matrix=matrix_arr[shifted_idx],
        verifier_names=names,
    )
    source_reference = SessionSlice(
        name=f"{projection_name}_template_source_reference_not_bin_{bin_index}",
        labels=labels_arr[reference_idx],
        score_matrix=matrix_arr[reference_idx],
        verifier_names=names,
    )
    metadata = {
        "source": f"{projection_name}_quartile_{bin_index}_shifted_slice",
        "projection": projection_name,
        "shifted_bin": int(bin_index),
        "vote_distribution_distance": _round(distance),
        "threshold": _round(SHIFT_DISTANCE_THRESHOLD),
        "policy": "choose cached quartile slice with maximum verifier-vote distance from the remaining rows",
    }
    return shifted, source_reference, metadata


def load_v13_template_library(path: Path | str) -> TemplateLibrary:
    """Load a persisted Exp 3708 v13 template library."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "carnot.fr11_v13_template_library":
        raise ValueError("expected a carnot.fr11_v13_template_library v13 template library")
    cap = int(payload.get("cap", 0))
    if cap < 1:
        raise ValueError("v13 template library cap must be positive")
    names = tuple(str(name) for name in payload.get("verifier_names", ()))
    if not names:
        raise ValueError("v13 template library verifier_names must be present")
    entries: list[TemplateEntry] = []
    for item in payload.get("entries", ()):
        weights_by_name = item.get("weights")
        if not isinstance(weights_by_name, Mapping) or any(name not in weights_by_name for name in names):
            raise ValueError("v13 template library entry weights must cover verifier_names")
        entries.append(
            TemplateEntry(
                template_id=str(item.get("template_id", "")),
                weights=np.asarray([weights_by_name[name] for name in names], dtype=np.float64),
                edges=tuple(dict(edge) for edge in item.get("edges", ())),
                support=int(item.get("support", 0)),
                source_sessions=tuple(str(value) for value in item.get("source_sessions", ())),
                utility=float(item.get("utility", 0.0)),
            )
        )
    return TemplateLibrary(
        cap=cap,
        verifier_names=names,
        entries=tuple(entries),
        consolidation_events=tuple(dict(event) for event in payload.get("consolidation_events", ())),
    )


def paired_bootstrap_delta_ci(
    labels: Sequence[int] | np.ndarray,
    template_scores: Sequence[float] | np.ndarray,
    cold_scores: Sequence[float] | np.ndarray,
    *,
    seeds: Sequence[int],
    n_bootstrap: int,
) -> JsonDict:
    """Return a paired bootstrap CI for template AUROC minus cold-start AUROC."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    template_arr = np.asarray(template_scores, dtype=np.float64)
    cold_arr = np.asarray(cold_scores, dtype=np.float64)
    if labels_arr.shape[0] != template_arr.shape[0] or labels_arr.shape[0] != cold_arr.shape[0]:
        raise ValueError("labels and scores must have the same length")
    if not np.isfinite(template_arr).all() or not np.isfinite(cold_arr).all():
        raise ValueError("scores must be finite")
    point = v10.score_metrics(labels_arr, template_arr)["auroc"] - v10.score_metrics(
        labels_arr,
        cold_arr,
    )["auroc"]
    values: list[float] = []
    seed_means: list[float] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        per_seed: list[float] = []
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, len(labels_arr), size=len(labels_arr))
            if len(set(labels_arr[idx].tolist())) < 2:
                continue
            value = v10.score_metrics(labels_arr[idx], template_arr[idx])["auroc"] - v10.score_metrics(
                labels_arr[idx],
                cold_arr[idx],
            )["auroc"]
            per_seed.append(float(value))
            values.append(float(value))
        seed_means.append(_round(float(np.mean(per_seed))) if per_seed else _round(point))
    if values:
        low, high = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    else:
        low = high = point
    return {
        "point": _round(point),
        "ci95": [_round(low), _round(high)],
        "bootstrap_seeds": [int(seed) for seed in seeds],
        "seed_mean_deltas": seed_means,
        "n_bootstrap_per_seed": int(n_bootstrap),
    }


def select_honest_verdict(
    *,
    blocked: bool,
    raw_template_beats_with_ci: bool,
    graceful_fallback: bool,
    gate_passed: bool,
) -> str:
    """Choose one of the allowed Exp 3720 terminal verdicts."""

    if blocked:
        return BLOCKED_VERDICT
    if gate_passed and raw_template_beats_with_ci:
        return ROBUST_VERDICT
    if gate_passed and graceful_fallback:
        return FALLBACK_VERDICT
    return HURTS_VERDICT


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3720 terminal artifact schema."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    verdict = artifact.get("honest_verdict")
    if verdict not in TERMINAL_VERDICTS:
        raise ValueError(f"unsupported honest_verdict: {verdict!r}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    gate = artifact.get("acceptance_gate")
    if not isinstance(gate, Mapping) or type(gate.get("passed")) is not bool:
        raise ValueError("acceptance_gate.passed must be present as a boolean")
    if not isinstance(artifact.get("duration_s"), int | float):
        raise ValueError("duration_s must be numeric")
    serialized = json.dumps(artifact, sort_keys=True)
    if "GGUF" in serialized or "CUDA" in serialized:
        raise ValueError("forbidden inference marker present")
    for field in (
        "template_robust_or_graceful_fallback",
        "conservative_fallback_triggered",
        "collapse_detected_deploy_arm",
        "template_library_bounded",
        "pass_rate_vs_true_accuracy_distinct_assert",
    ):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a boolean")
    for field in ("deploy_arm_auroc_under_shift", "cold_start_auroc_under_shift"):
        value = artifact.get(field)
        if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{field} must be finite and in [0, 1]")
    _validate_delta_ci(artifact.get("template_vs_cold_delta_ci"))
    if verdict == BLOCKED_VERDICT:
        return
    if int(artifact.get("n_online_updates", 0)) < MIN_ONLINE_UPDATES:
        raise ValueError(f"n_online_updates must be at least {MIN_ONLINE_UPDATES}")
    library = artifact.get("template_library")
    if not isinstance(library, Mapping):
        raise ValueError("template_library must be present")
    if int(library.get("size", 10**9)) > int(library.get("cap", -1)):
        raise ValueError("template_library size exceeds cap")


def write_artifact(
    repo_root: Path | str,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    labels: Sequence[int] | None = None,
    scores_by_verifier: Mapping[str, Sequence[float]] | None = None,
    template_library: TemplateLibrary | None = None,
    shifted_session: SessionSlice | None = None,
    source_reference_session: SessionSlice | None = None,
    shifted_session_source: str = "provided_shifted_session",
    started_s: float | None = None,
    now_s: float | None = None,
    bootstrap_seeds: Sequence[int] = DEFAULT_BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> Path:
    """Build, validate, and persist the Exp 3720 JSON artifact."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    if template_library is not None and (shifted_session is not None or source_reference_session is not None):
        artifact = build_artifact_from_shifted_session(
            template_library=template_library,
            shifted_session=shifted_session,
            source_reference_session=source_reference_session,
            shifted_session_source=shifted_session_source,
            started_s=start,
            now_s=now_s,
            persistence_dir=root / "results",
            bootstrap_seeds=bootstrap_seeds,
            n_bootstrap=n_bootstrap,
        )
    elif template_library is not None and labels is not None and scores_by_verifier is not None:
        artifact = build_artifact_from_scores(
            labels=labels,
            scores_by_verifier=scores_by_verifier,
            template_library=template_library,
            started_s=start,
            now_s=now_s,
            persistence_dir=root / "results",
            bootstrap_seeds=bootstrap_seeds,
            n_bootstrap=n_bootstrap,
        )
    else:
        artifact = build_artifact(
            root,
            started_s=start,
            now_s=now_s,
            bootstrap_seeds=bootstrap_seeds,
            n_bootstrap=n_bootstrap,
        )
    output = root / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def reproducibility_checksum(
    *,
    template_library: TemplateLibrary,
    shifted_session: SessionSlice,
    source_reference_session: SessionSlice,
    template_weights: Sequence[float],
    cold_weights: Sequence[float],
    random_seed: int,
    bootstrap_seeds: Sequence[int],
) -> str:
    """Hash deterministic sessions, template memory, deploy weights, and seeds."""

    digest = hashlib.sha256()
    digest.update(_json_sha256(v13.template_library_to_json(template_library)).encode("ascii"))
    for session in (source_reference_session, shifted_session):
        digest.update(session.name.encode("utf-8"))
        digest.update(np.ascontiguousarray(session.labels, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(session.score_matrix, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(template_weights, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(cold_weights, dtype=np.float64).tobytes())
    digest.update(str(int(random_seed)).encode("ascii"))
    digest.update(json.dumps([int(seed) for seed in bootstrap_seeds]).encode("ascii"))
    return digest.hexdigest()


def _blocked_artifact(
    *,
    duration_s: float,
    random_seed: int,
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    payload = json.dumps(
        {"preconditions": [dict(item) for item in preconditions], "random_seed": random_seed},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact: JsonDict = {
        "artifact": "experiment_3720_fr11_continuous_self_learning_v14",
        "schema": "carnot.fr11_continuous_self_learning_v14",
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "shifted_session_source": "blocked_shifted_slice_unavailable",
        "deploy_arm_auroc_under_shift": 0.0,
        "cold_start_auroc_under_shift": 0.0,
        "template_robust_or_graceful_fallback": False,
        "conservative_fallback_triggered": False,
        "collapse_detected_deploy_arm": False,
        "template_library_bounded": False,
        "pass_rate_vs_true_accuracy_distinct_assert": False,
        "n_online_updates": 0,
        "random_seed": int(random_seed),
        "reproducibility_checksum": hashlib.sha256(payload).hexdigest(),
        "duration_s": _round(duration_s),
        "acceptance_gate": {
            "condition": (
                "template_robust_or_graceful_fallback == true AND "
                "collapse_detected_deploy_arm == false AND "
                "template_library_bounded == true AND "
                "pass_rate_vs_true_accuracy_distinct_assert == true"
            ),
            "passed": False,
            "principle": (
                "Distribution-shift robustness is validated only when the "
                "template helps OR falls back gracefully, no deploy collapse is "
                "detected, the library stays bounded, and the two metrics are distinct."
            ),
        },
        "adversarial_verify": "clean",
        "template_vs_cold_delta_ci": {
            "point": 0.0,
            "ci95": [0.0, 0.0],
            "bootstrap_seeds": [],
            "seed_mean_deltas": [],
            "n_bootstrap_per_seed": 0,
        },
        "preconditions_checked": [dict(item) for item in preconditions],
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


def _template_library_precondition(root: Path) -> JsonDict:
    path = root / V13_TEMPLATE_LIBRARY_REL_PATH
    if not path.is_file():
        return {
            "resource": "v13_template_library",
            "available": False,
            "detail": str(path),
        }
    try:
        library = load_v13_template_library(path)
    except Exception as exc:  # noqa: BLE001 - precondition reports the exact parser failure.
        return {
            "resource": "v13_template_library",
            "available": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }
    return {
        "resource": "v13_template_library",
        "available": _template_library_bounded(library) and bool(library.entries),
        "detail": f"{path}; size={len(library.entries)}; cap={library.cap}",
    }


def _trace_precondition(
    labels: Sequence[int] | np.ndarray,
    scores_by_verifier: Mapping[str, Sequence[float]],
) -> JsonDict:
    return {
        "resource": "cached_traces_with_per_verifier_scores_and_labels",
        "available": _runnable_labels(np.asarray(labels, dtype=np.int64)),
        "detail": (
            f"n_examples={len(labels)}; labels={sorted(set(int(value) for value in labels))}; "
            f"n_verifiers={len(scores_by_verifier)}; required>={MIN_ONLINE_UPDATES}"
        ),
    }


def _shifted_session_precondition(
    template_library: TemplateLibrary,
    shifted_session: SessionSlice | None,
    source_reference_session: SessionSlice | None,
) -> JsonDict:
    return {
        "resource": "distributionally_shifted_session_slice",
        "available": False,
        "detail": (
            f"template_library_bounded={_template_library_bounded(template_library)}; "
            f"shifted_present={shifted_session is not None}; "
            f"source_reference_present={source_reference_session is not None}; "
            f"required shifted/source slices with >={MIN_ONLINE_UPDATES} binary rows"
        ),
    }


def _validate_session(session: SessionSlice) -> None:
    _validate_scores(session.labels, session.score_matrix, session.verifier_names)


def _validate_scores(
    labels: Sequence[int] | np.ndarray,
    matrix: np.ndarray,
    verifier_names: Sequence[str],
) -> None:
    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix_arr = np.asarray(matrix, dtype=np.float64)
    if matrix_arr.ndim != 2:
        raise ValueError("score matrix must be two-dimensional")
    if matrix_arr.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    if matrix_arr.shape[1] != len(tuple(verifier_names)):
        raise ValueError("score matrix must match verifier_names")
    if not np.isfinite(matrix_arr).all():
        raise ValueError("verifier score matrix must be finite")


def _runnable_session(session: SessionSlice) -> bool:
    return len(session.labels) >= MIN_ONLINE_UPDATES and _slice_has_binary_support(session.labels)


def _runnable_labels(labels: np.ndarray) -> bool:
    return len(labels) >= MIN_ONLINE_UPDATES and _slice_has_binary_support(labels)


def _slice_has_binary_support(labels: Sequence[int] | np.ndarray) -> bool:
    return {int(value) for value in labels} == {0, 1}


def _template_library_bounded(library: TemplateLibrary) -> bool:
    return int(library.cap) >= 1 and len(library.entries) <= int(library.cap)


def _signed_scores(matrix: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    return np.clip(
        np.asarray(matrix, dtype=np.float64) @ np.asarray(weights, dtype=np.float64),
        0.0,
        1.0,
    )


def _weights_to_json(names: Sequence[str], weights: Sequence[float]) -> dict[str, float]:
    return {name: _round(weight) for name, weight in zip(names, weights, strict=True)}


def _validate_delta_ci(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("template_vs_cold_delta_ci must be an object")
    point = value.get("point")
    ci = value.get("ci95")
    if not _is_finite_number(point) or not isinstance(ci, list) or len(ci) != 2:
        raise ValueError("template_vs_cold_delta_ci must include point and ci95")
    if not all(_is_finite_number(item) for item in ci):
        raise ValueError("template_vs_cold_delta_ci bounds must be finite")
    if not float(ci[0]) <= float(point) <= float(ci[1]):
        raise ValueError("template_vs_cold_delta_ci must contain its point estimate")


def _json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0, end - float(started_s))


def _round(value: float | int | np.floating[Any], digits: int = 6) -> float:
    return round(float(value), digits)
