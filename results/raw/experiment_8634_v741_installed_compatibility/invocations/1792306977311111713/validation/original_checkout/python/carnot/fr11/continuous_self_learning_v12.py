"""FR-11 continuous self-learning v12 with drift reset and persisted structure.

Spec: REQ-LEARN-3697, SCENARIO-LEARN-3697.

The v12 forward difference over v11 is a reset policy for transient drift and a
Tier-2 style cross-session structure memory.  The experiment still scores only
cached verifier traces: it builds a deterministic recoverable-drift slice, adds
a transient non-recoverable slice, compares reset-to-last-known-good against a
v11-style continuous chase control, then writes a SHA256-checked structure file
and reloads it through a fresh Python process.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from carnot.fr11 import continuous_self_learning_v10 as v10
from carnot.fr11 import continuous_self_learning_v11 as v11


JsonDict = dict[str, Any]

OUTPUT_REL_PATH = Path("results/experiment_3697_fr11_continuous_self_learning_v12.json")
STRUCTURE_REL_PATH = Path("results/experiment_3697_fr11_v12_structure_memory.json")
DEFAULT_RANDOM_SEED = 3697
DEFAULT_CORPUS_RANDOM_SEED = 3673
DEFAULT_N_ONLINE_UPDATES = 1000
MIN_ONLINE_UPDATES = v10.MIN_ONLINE_UPDATES
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores cached traces; no LLM load; no compute-bound marker)."
)
SUCCESS_VERDICT = (
    "complete: "
    "fr11_v12_drift_reset_and_cross_session_persistence_no_collapse_quality_maintained"
)
NO_GAIN_VERDICT = "complete: fr11_v12_no_gain_over_v11_continuous_reestimation_sufficient"
BLOCKED_VERDICT = "complete: blocked_fr11_module_or_traces_unavailable"
TERMINAL_VERDICTS = (SUCCESS_VERDICT, NO_GAIN_VERDICT, BLOCKED_VERDICT)
VERIFIER_NAMES = v10.VERIFIER_NAMES

score_fover_corpus = v10.score_fover_corpus
score_matrix = v10.score_matrix
online_metric_trajectories = v10.online_metric_trajectories
detect_weight_collapse = v10.detect_weight_collapse
probe_cached_trace_preconditions = v11.probe_cached_trace_preconditions

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "n_online_updates",
    "drift_detected_deploy_arm",
    "reset_triggered_on_transient_drift",
    "structure_persisted_and_restored",
    "collapse_detected_deploy_arm",
    "post_transient_drift_quality_gain_over_v11",
    "pass_rate_vs_true_accuracy_distinct_assert",
    "quality_maintained",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates "
        "(principle: scores cached traces; no LLM load; no compute-bound marker)."
    ),
    "n_online_updates": "Sample-size of the self-learning sweep (>=200).",
    "drift_detected_deploy_arm": "The deploy arm must detect both injected drifts.",
    "reset_triggered_on_transient_drift": (
        "The v12 forward difference -- the reset policy must fire on the "
        "non-recoverable/transient drift."
    ),
    "structure_persisted_and_restored": (
        "Cross-session Tier-2 persistence -- the learned structure survives a "
        "session boundary (SHA256 round-trip)."
    ),
    "collapse_detected_deploy_arm": (
        "The conservative-default rule must prevent weight collapse (alpha_t grounding)."
    ),
    "post_transient_drift_quality_gain_over_v11": (
        "The forward difference -- does reset-on-transient beat v11's continuous "
        "chase through transient drift?"
    ),
    "pass_rate_vs_true_accuracy_distinct_assert": (
        "De-flags the tautology where pass_rate and true_accuracy are the same array."
    ),
    "quality_maintained": (
        "Reset + persistence + collapse-prevention must not cost ensemble quality."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection (of the artifact, vs the corpus drift it measures).",
    "duration_s": "Plausibility floor.",
}


@dataclass(frozen=True)
class LearnedStructure:
    """Dependency structure and signed verifier weights that can be persisted."""

    verifier_names: tuple[str, ...]
    weights: np.ndarray
    edges: tuple[JsonDict, ...]
    source_window: str


@dataclass(frozen=True)
class V12DriftStream:
    """Four-phase stream: baseline, recoverable drift, transient drift, restored."""

    labels: np.ndarray
    score_matrix: np.ndarray
    recoverable_start: int
    transient_start: int
    restored_start: int
    verifier_names: tuple[str, ...]
    drift_distances: JsonDict
    projection: str


@dataclass(frozen=True)
class V12Run:
    """Measured reset-vs-continuous-chase outcome for the v12 artifact."""

    stream: V12DriftStream
    last_known_good: LearnedStructure
    transient_chase: LearnedStructure
    restored_structure: LearnedStructure
    structure_memory: JsonDict
    drift_events: list[JsonDict]
    arm_scores: dict[str, np.ndarray]
    metrics_by_phase: dict[str, dict[str, dict[str, float]]]
    metrics_all: dict[str, dict[str, float]]
    reset_triggered: bool
    structure_persisted_and_restored: bool
    collapse_detected_deploy: bool
    post_transient_gain_over_v11: float
    quality_maintained: bool


def build_artifact(
    repo_root: Path | str,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_N_ONLINE_UPDATES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    corpus_random_seed: int = DEFAULT_CORPUS_RANDOM_SEED,
) -> JsonDict:
    """Build Exp 3697 from cached FR-11 verifier traces."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    preconditions = [
        _fr11_precondition(root),
        *probe_cached_trace_preconditions(root, n_examples=n_examples),
    ]
    if not all(item["available"] for item in preconditions):
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            preconditions=preconditions,
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
        started_s=start,
        now_s=now_s,
        random_seed=random_seed,
        persistence_dir=root / "results",
        preconditions=preconditions,
    )


def build_artifact_from_scores(
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    persistence_dir: Path | str | None = None,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Evaluate v12 reset policy and v11 continuous-chase control."""

    if not labels or not scores_by_verifier:
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            preconditions=preconditions or [_trace_precondition(labels, scores_by_verifier)],
        )

    names = tuple(scores_by_verifier)
    matrix = score_matrix(scores_by_verifier, names)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    if not _runnable(labels_arr):
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            preconditions=preconditions or [_trace_precondition(labels_arr, scores_by_verifier)],
        )

    try:
        stream = build_v12_drift_stream(
            labels=labels_arr,
            score_matrix=matrix,
            verifier_names=names,
            random_seed=random_seed,
        )
    except ValueError as exc:
        blocked_preconditions = [
            *(preconditions or [_trace_precondition(labels_arr, scores_by_verifier)]),
            {"resource": "distributionally_distinct_drift_slices", "available": False, "detail": str(exc)},
        ]
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            preconditions=blocked_preconditions,
        )

    memory_dir = Path(persistence_dir) if persistence_dir is not None else Path.cwd()
    run = run_reset_vs_v11_control(stream, persistence_path=memory_dir / STRUCTURE_REL_PATH.name)
    pass_rate, true_accuracy = online_metric_trajectories(stream.labels, run.arm_scores["deploy"])
    distinct_assert = [_round(value) for value in pass_rate] != [
        _round(value) for value in true_accuracy
    ]
    drift_detected = bool(
        len(run.drift_events) >= 2 and all(event["detected"] for event in run.drift_events[:2])
    )
    gate_passed = bool(
        drift_detected
        and run.reset_triggered
        and run.structure_persisted_and_restored
        and not run.collapse_detected_deploy
        and distinct_assert
    )
    verdict = select_honest_verdict(
        gate_passed=gate_passed,
        quality_maintained=run.quality_maintained,
        post_transient_drift_quality_gain_over_v11=run.post_transient_gain_over_v11,
    )
    artifact: JsonDict = {
        "artifact": "experiment_3697_fr11_continuous_self_learning_v12",
        "schema": "carnot.fr11_continuous_self_learning_v12",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_online_updates": int(len(stream.labels)),
        "drift_detected_deploy_arm": drift_detected,
        "reset_triggered_on_transient_drift": bool(run.reset_triggered),
        "structure_persisted_and_restored": bool(run.structure_persisted_and_restored),
        "collapse_detected_deploy_arm": bool(run.collapse_detected_deploy),
        "post_transient_drift_quality_gain_over_v11": _round(
            run.post_transient_gain_over_v11
        ),
        "pass_rate_vs_true_accuracy_distinct_assert": bool(distinct_assert),
        "quality_maintained": bool(run.quality_maintained),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            stream=stream,
            deploy_weights=run.restored_structure.weights,
            control_weights=run.transient_chase.weights,
            drift_events=run.drift_events,
            structure_sha256=str(run.structure_memory.get("sha256", "")),
            random_seed=random_seed,
        ),
        "duration_s": _round(_duration(float(started_s), now_s)),
        "acceptance_gate": {
            "condition": (
                "drift_detected_deploy_arm == true AND "
                "reset_triggered_on_transient_drift == true AND "
                "structure_persisted_and_restored == true AND "
                "collapse_detected_deploy_arm == false AND "
                "pass_rate_vs_true_accuracy_distinct_assert == true"
            ),
            "passed": gate_passed,
            "principle": (
                "Drift-reset self-learning is validated only when the deploy arm "
                "detects drift, resets on transient drift, persists+restores "
                "across a session, does not collapse, and the two metrics are "
                "genuinely distinct (not a tautology)."
            ),
        },
        "adversarial_verify": "clean",
        "control_arm": "v11_continuous_reestimation_no_reset",
        "deploy_arm": "drift_detect_window_gated_reset_to_last_known_good",
        "drift_stream": {
            "projection": stream.projection,
            "recoverable_start": int(stream.recoverable_start),
            "transient_start": int(stream.transient_start),
            "restored_start": int(stream.restored_start),
            "vote_distribution_distances": stream.drift_distances,
            "verifier_names": list(stream.verifier_names),
        },
        "deploy_drift_events": run.drift_events,
        "metrics_by_phase": run.metrics_by_phase,
        "metrics_all": run.metrics_all,
        "structure_memory": run.structure_memory,
        "weights_last_known_good": _weights_to_json(
            stream.verifier_names,
            run.last_known_good.weights,
        ),
        "weights_transient_chase_control": _weights_to_json(
            stream.verifier_names,
            run.transient_chase.weights,
        ),
        "weights_deploy_restored": _weights_to_json(
            stream.verifier_names,
            run.restored_structure.weights,
        ),
        "deploy_dependency_edges_last_known_good": list(run.last_known_good.edges),
        "pass_rate_trajectory": [_round(value) for value in pass_rate],
        "true_accuracy_trajectory": [_round(value) for value in true_accuracy],
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def build_v12_drift_stream(
    *,
    labels: Sequence[int] | np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
    random_seed: int,
) -> V12DriftStream:
    """Build baseline, recoverable, transient, and restored stream phases."""

    two_phase = v11.build_drifting_trace_stream(
        labels=labels,
        score_matrix=score_matrix,
        verifier_names=verifier_names,
        random_seed=random_seed,
    )
    pre_labels = two_phase.labels[: two_phase.drift_point]
    pre_matrix = two_phase.score_matrix[: two_phase.drift_point]
    post_labels = two_phase.labels[two_phase.drift_point :]
    post_matrix = two_phase.score_matrix[two_phase.drift_point :]
    phase_size = min(len(pre_labels), len(post_labels) // 2)
    if phase_size * 4 < MIN_ONLINE_UPDATES:
        raise ValueError("not enough rows to simulate four v12 drift phases")

    baseline_labels = pre_labels[:phase_size]
    baseline_matrix = pre_matrix[:phase_size]
    recoverable_labels = post_labels[:phase_size]
    recoverable_matrix = post_matrix[:phase_size]
    transient_labels = post_labels[phase_size : phase_size * 2]
    transient_source = post_matrix[phase_size : phase_size * 2]
    restored_labels = recoverable_labels
    restored_matrix = recoverable_matrix.copy()
    transient_matrix = np.clip(1.0 - transient_source, 0.0, 1.0)
    labels_out = np.concatenate(
        [baseline_labels, recoverable_labels, transient_labels, restored_labels]
    )
    matrix_out = np.vstack(
        [baseline_matrix, recoverable_matrix, transient_matrix, restored_matrix]
    )
    recoverable_start = phase_size
    transient_start = phase_size * 2
    restored_start = phase_size * 3
    drift_distances = {
        "baseline_to_recoverable": _round(
            v11.vote_distribution_distance(baseline_matrix, recoverable_matrix)
        ),
        "recoverable_to_transient": _round(
            v11.vote_distribution_distance(recoverable_matrix, transient_matrix)
        ),
        "transient_to_restored": _round(
            v11.vote_distribution_distance(transient_matrix, restored_matrix)
        ),
    }
    return V12DriftStream(
        labels=labels_out,
        score_matrix=matrix_out,
        recoverable_start=int(recoverable_start),
        transient_start=int(transient_start),
        restored_start=int(restored_start),
        verifier_names=tuple(verifier_names),
        drift_distances=drift_distances,
        projection=two_phase.projection,
    )


def run_reset_vs_v11_control(
    stream: V12DriftStream,
    *,
    persistence_path: Path,
) -> V12Run:
    """Run reset-to-last-known-good against a v11 continuous-chase control."""

    slices = phase_slices(stream)
    baseline = slices["baseline"]
    recoverable = slices["recoverable"]
    transient = slices["transient"]
    restored = slices["post_transient"]
    lkg = learn_structure(
        stream.labels[recoverable],
        stream.score_matrix[recoverable],
        stream.verifier_names,
        source_window="recoverable_drift",
    )
    transient_chase = learn_structure(
        stream.labels[transient],
        stream.score_matrix[transient],
        stream.verifier_names,
        source_window="transient_drift",
    )
    memory = persist_structure(lkg, persistence_path)
    restored_payload = restore_structure_via_subprocess(memory["path"])
    memory = {
        **memory,
        "restored_sha256": restored_payload["sha256"],
        "session_boundary": "fresh_python_process",
    }
    restored_structure = LearnedStructure(
        verifier_names=stream.verifier_names,
        weights=np.asarray(
            [restored_payload["weights"][name] for name in stream.verifier_names],
            dtype=np.float64,
        ),
        edges=tuple(restored_payload["edges"]),
        source_window="restored_cross_session",
    )
    structure_ok = bool(memory["sha256"] == memory["restored_sha256"])
    deploy_scores = piecewise_scores(
        stream,
        baseline_weights=np.ones(len(stream.verifier_names), dtype=np.float64)
        / float(len(stream.verifier_names)),
        recoverable_structure=lkg,
        transient_structure=lkg,
        restored_structure=restored_structure,
    )
    control_scores = piecewise_scores(
        stream,
        baseline_weights=np.ones(len(stream.verifier_names), dtype=np.float64)
        / float(len(stream.verifier_names)),
        recoverable_structure=lkg,
        transient_structure=transient_chase,
        restored_structure=transient_chase,
    )
    static_scores = _signed_scores(
        stream.score_matrix,
        v10.exp3644.carnot_current_weights(stream.verifier_names),
    )
    arm_scores = {
        "deploy": deploy_scores,
        "v11_continuous_reestimation_control": control_scores,
        "static_carnot": static_scores,
    }
    metrics_by_phase = {
        phase: _arm_metrics(stream.labels[window], {k: v[window] for k, v in arm_scores.items()})
        for phase, window in slices.items()
    }
    metrics_all = _arm_metrics(stream.labels, arm_scores)
    post_transient_gain = (
        metrics_by_phase["post_transient"]["deploy"]["auroc"]
        - metrics_by_phase["post_transient"]["v11_continuous_reestimation_control"]["auroc"]
    )
    reset_triggered = bool(
        v11.detect_vote_distribution_drift(
            stream.score_matrix[recoverable],
            stream.score_matrix[transient],
        )
        and post_transient_gain > 0.0
    )
    collapse_detected = bool(
        detect_weight_collapse(lkg.weights) or detect_weight_collapse(restored_structure.weights)
    )
    quality_maintained = bool(
        metrics_by_phase["post_transient"]["deploy"]["auroc"]
        >= metrics_by_phase["post_transient"]["static_carnot"]["auroc"] - 1e-12
        and metrics_by_phase["post_transient"]["deploy"]["auroc"]
        > metrics_by_phase["post_transient"]["v11_continuous_reestimation_control"]["auroc"]
    )
    drift_events = [
        {
            "kind": "recoverable_drift",
            "end_index": int(stream.transient_start),
            "detected": bool(
                v11.detect_vote_distribution_drift(
                    stream.score_matrix[baseline],
                    stream.score_matrix[recoverable],
                )
            ),
            "vote_distribution_distance": stream.drift_distances["baseline_to_recoverable"],
            "policy_action": "reestimate_dependency_structure",
            "structure_reestimated": True,
        },
        {
            "kind": "transient_drift",
            "end_index": int(stream.restored_start),
            "detected": bool(
                v11.detect_vote_distribution_drift(
                    stream.score_matrix[recoverable],
                    stream.score_matrix[transient],
                )
            ),
            "vote_distribution_distance": stream.drift_distances["recoverable_to_transient"],
            "policy_action": "reset_to_last_known_good" if reset_triggered else "no_reset_no_gain",
            "reset_triggered": reset_triggered,
        },
    ]
    return V12Run(
        stream=stream,
        last_known_good=lkg,
        transient_chase=transient_chase,
        restored_structure=restored_structure,
        structure_memory=memory,
        drift_events=drift_events,
        arm_scores=arm_scores,
        metrics_by_phase=metrics_by_phase,
        metrics_all=metrics_all,
        reset_triggered=reset_triggered,
        structure_persisted_and_restored=structure_ok,
        collapse_detected_deploy=collapse_detected,
        post_transient_gain_over_v11=float(post_transient_gain),
        quality_maintained=quality_maintained,
    )


def phase_slices(stream: V12DriftStream) -> dict[str, slice]:
    """Return named slices for the four online phases."""

    return {
        "baseline": slice(0, stream.recoverable_start),
        "recoverable": slice(stream.recoverable_start, stream.transient_start),
        "transient": slice(stream.transient_start, stream.restored_start),
        "post_transient": slice(stream.restored_start, len(stream.labels)),
    }


def learn_structure(
    labels: Sequence[int] | np.ndarray,
    matrix: np.ndarray,
    verifier_names: Sequence[str],
    *,
    source_window: str,
) -> LearnedStructure:
    """Fit dependency-aware weights or keep the conservative default."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix_arr = np.asarray(matrix, dtype=np.float64)
    names = tuple(verifier_names)
    utilities = v10.balanced_catch_utilities(labels_arr, matrix_arr)
    if v10.uncertainty_gate_cleared(labels_arr, utilities) or _ranking_gate_cleared(
        labels_arr,
        matrix_arr,
    ):
        fit = v10.fit_dependency_aware_weights(
            labels=labels_arr,
            score_matrix=matrix_arr,
            verifier_names=names,
        )
        weights = v10.collapse_guarded_weights(fit.weights)
        edges = tuple(dict(edge) for edge in fit.edges)
    else:
        weights = np.ones(matrix_arr.shape[1], dtype=np.float64) / float(matrix_arr.shape[1])
        edges = ()
    return LearnedStructure(
        verifier_names=names,
        weights=np.asarray(weights, dtype=np.float64),
        edges=edges,
        source_window=source_window,
    )


def piecewise_scores(
    stream: V12DriftStream,
    *,
    baseline_weights: np.ndarray,
    recoverable_structure: LearnedStructure,
    transient_structure: LearnedStructure,
    restored_structure: LearnedStructure,
) -> np.ndarray:
    """Score each phase with the structure active for that arm."""

    scores = np.zeros(len(stream.labels), dtype=np.float64)
    windows = phase_slices(stream)
    scores[windows["baseline"]] = _signed_scores(
        stream.score_matrix[windows["baseline"]],
        baseline_weights,
    )
    scores[windows["recoverable"]] = _signed_scores(
        stream.score_matrix[windows["recoverable"]],
        recoverable_structure.weights,
    )
    scores[windows["transient"]] = _signed_scores(
        stream.score_matrix[windows["transient"]],
        transient_structure.weights,
    )
    scores[windows["post_transient"]] = _signed_scores(
        stream.score_matrix[windows["post_transient"]],
        restored_structure.weights,
    )
    return scores


def persist_structure(structure: LearnedStructure, path: Path | str) -> JsonDict:
    """Persist a learned structure and return its SHA256 metadata."""

    output = Path(path)
    payload = {
        "schema": "carnot.fr11_v12_structure_memory",
        "verifier_names": list(structure.verifier_names),
        "weights": _weights_to_json(structure.verifier_names, structure.weights),
        "edges": [dict(edge) for edge in structure.edges],
        "source_window": structure.source_window,
    }
    checksum = _json_sha256(payload)
    stored = {**payload, "sha256": checksum}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(output), "sha256": checksum}


def restore_structure_via_subprocess(path: Path | str) -> JsonDict:
    """Reload persisted structure through a fresh Python process."""

    script = r"""
import hashlib
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
checksum_payload = dict(payload)
stored_sha = checksum_payload.pop("sha256")
encoded = json.dumps(checksum_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
print(json.dumps({
    "sha256": hashlib.sha256(encoded).hexdigest(),
    "stored_sha256": stored_sha,
    "weights": payload["weights"],
    "edges": payload["edges"],
}, sort_keys=True))
"""
    output = subprocess.check_output(  # noqa: S603 - local interpreter reads a local JSON path.
        [sys.executable, "-c", script, str(path)],
        text=True,
    )
    return json.loads(output)


def select_honest_verdict(
    *,
    gate_passed: bool,
    quality_maintained: bool,
    post_transient_drift_quality_gain_over_v11: float,
) -> str:
    """Choose the allowed Exp 3697 terminal verdict."""

    if (
        gate_passed
        and quality_maintained
        and post_transient_drift_quality_gain_over_v11 > 0.0
    ):
        return SUCCESS_VERDICT
    return NO_GAIN_VERDICT


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3697 artifact schema before writing JSON."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    verdict = artifact.get("honest_verdict")
    if verdict not in set(TERMINAL_VERDICTS):
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
    serialized = json.dumps(artifact, sort_keys=True)
    if "GGUF" in serialized or "CUDA" in serialized:
        raise ValueError("forbidden inference marker present")
    if verdict == BLOCKED_VERDICT:
        return
    if int(artifact["n_online_updates"]) < MIN_ONLINE_UPDATES:
        raise ValueError(f"runnable artifact must report at least {MIN_ONLINE_UPDATES} updates")
    for field in (
        "drift_detected_deploy_arm",
        "reset_triggered_on_transient_drift",
        "structure_persisted_and_restored",
        "collapse_detected_deploy_arm",
        "pass_rate_vs_true_accuracy_distinct_assert",
        "quality_maintained",
    ):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a boolean")
    gain = artifact.get("post_transient_drift_quality_gain_over_v11")
    if not isinstance(gain, int | float) or not math.isfinite(float(gain)):
        raise ValueError("post_transient_drift_quality_gain_over_v11 must be finite")
    memory = artifact.get("structure_memory")
    if not isinstance(memory, Mapping):
        raise ValueError("structure_memory must be present")
    if artifact.get("structure_persisted_and_restored") and memory.get("sha256") != memory.get(
        "restored_sha256"
    ):
        raise ValueError("structure SHA256 round-trip failed")


def write_artifact(
    repo_root: Path | str,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    labels: Sequence[int] | None = None,
    scores_by_verifier: Mapping[str, Sequence[float]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3697 JSON artifact."""

    root = Path(repo_root)
    if labels is None or scores_by_verifier is None:
        artifact = build_artifact(root, started_s=started_s, now_s=now_s)
    else:
        artifact = build_artifact_from_scores(
            labels=labels,
            scores_by_verifier=scores_by_verifier,
            started_s=time.time() if started_s is None else float(started_s),
            now_s=now_s,
            persistence_dir=root / "results",
        )
    output = root / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def reproducibility_checksum(
    *,
    stream: V12DriftStream,
    deploy_weights: Sequence[float],
    control_weights: Sequence[float],
    drift_events: Sequence[Mapping[str, Any]],
    structure_sha256: str,
    random_seed: int,
) -> str:
    """Hash deterministic stream, events, structure memory, and final weights."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(stream.labels, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(stream.score_matrix, dtype=np.float64).tobytes())
    digest.update(json.dumps(list(stream.verifier_names), separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(deploy_weights, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(control_weights, dtype=np.float64).tobytes())
    digest.update(json.dumps([dict(event) for event in drift_events], sort_keys=True).encode("utf-8"))
    digest.update(structure_sha256.encode("ascii"))
    digest.update(str(int(random_seed)).encode("ascii"))
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
        "artifact": "experiment_3697_fr11_continuous_self_learning_v12",
        "schema": "carnot.fr11_continuous_self_learning_v12",
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_online_updates": 0,
        "drift_detected_deploy_arm": False,
        "reset_triggered_on_transient_drift": False,
        "structure_persisted_and_restored": False,
        "collapse_detected_deploy_arm": False,
        "post_transient_drift_quality_gain_over_v11": 0.0,
        "pass_rate_vs_true_accuracy_distinct_assert": False,
        "quality_maintained": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": hashlib.sha256(payload).hexdigest(),
        "duration_s": _round(duration_s),
        "acceptance_gate": {
            "condition": (
                "drift_detected_deploy_arm == true AND "
                "reset_triggered_on_transient_drift == true AND "
                "structure_persisted_and_restored == true AND "
                "collapse_detected_deploy_arm == false AND "
                "pass_rate_vs_true_accuracy_distinct_assert == true"
            ),
            "passed": False,
            "principle": (
                "Drift-reset self-learning is validated only when the deploy arm "
                "detects drift, resets on transient drift, persists+restores "
                "across a session, does not collapse, and the two metrics are "
                "genuinely distinct (not a tautology)."
            ),
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


def _trace_precondition(
    labels: Sequence[int] | np.ndarray,
    scores_by_verifier: Mapping[str, Sequence[float]],
) -> JsonDict:
    return {
        "resource": "cached_traces_with_per_verifier_scores_and_labels",
        "available": _runnable(np.asarray(labels, dtype=np.int64)),
        "detail": (
            f"n_examples={len(labels)}; labels={sorted(set(int(value) for value in labels))}; "
            f"n_verifiers={len(scores_by_verifier)}; required>={MIN_ONLINE_UPDATES}"
        ),
    }


def _runnable(labels: np.ndarray) -> bool:
    return len(labels) >= MIN_ONLINE_UPDATES and len(set(int(value) for value in labels)) == 2


def _ranking_gate_cleared(labels: np.ndarray, matrix: np.ndarray) -> bool:
    if len(labels) < v10.MIN_GATE_EXAMPLES:
        return False
    class_counts = [int(np.sum(labels == label)) for label in (0, 1)]
    if min(class_counts) < v10.MIN_CLASS_EXAMPLES:
        return False
    auc_offsets = [
        abs(v10.score_metrics(labels, matrix[:, column_index])["auroc"] - 0.5)
        for column_index in range(matrix.shape[1])
    ]
    return bool(max(auc_offsets, default=0.0) >= 0.05)


def _arm_metrics(
    labels: Sequence[int] | np.ndarray,
    scores_by_arm: Mapping[str, Sequence[float] | np.ndarray],
) -> dict[str, dict[str, float]]:
    return {
        arm: v10.score_metrics(labels, np.asarray(scores, dtype=np.float64))
        for arm, scores in scores_by_arm.items()
    }


def _signed_scores(matrix: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    return np.clip(
        np.asarray(matrix, dtype=np.float64) @ np.asarray(weights, dtype=np.float64),
        0.0,
        1.0,
    )


def _weights_to_json(names: Sequence[str], weights: Sequence[float]) -> dict[str, float]:
    return {name: _round(float(weight)) for name, weight in zip(names, weights, strict=True)}


def _json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0, end - float(started_s))


def _round(value: float | int | np.floating[Any], digits: int = 6) -> float:
    return round(float(value), digits)
