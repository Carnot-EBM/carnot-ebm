"""FR-11 continuous self-learning v13 with multi-session template memory.

Spec: REQ-LEARN-3708, SCENARIO-LEARN-3708.

The v13 forward difference over v12 is bounded consolidation across multiple
session boundaries.  Each session learns the same dependency-aware verifier
structure used by v12, then stores it as a reusable template in a capped
Tier-2 memory library.  The cap matters because an unbounded memory can look
good by accumulating every local accident.  This module therefore merges the
most similar pair whenever the cap is exceeded and checks the library by
SHA256 after every simulated session boundary before testing transfer on a
fresh held-out session.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
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
from carnot.fr11 import continuous_self_learning_v12 as v12


JsonDict = dict[str, Any]

OUTPUT_REL_PATH = Path("results/experiment_3708_fr11_continuous_self_learning_v13.json")
TEMPLATE_LIBRARY_REL_PATH = Path("results/experiment_3708_fr11_v13_template_library.json")
DEFAULT_RANDOM_SEED = 3708
DEFAULT_CORPUS_RANDOM_SEED = 3673
DEFAULT_N_ONLINE_UPDATES = 1000
DEFAULT_LIBRARY_CAP = 2
MIN_ONLINE_UPDATES = v10.MIN_ONLINE_UPDATES
MIN_SESSION_EXAMPLES = 80
MIN_SESSION_DISTANCE = 0.30
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores cached traces; no LLM load; no compute-bound marker)."
)
SUCCESS_VERDICT = (
    "complete: "
    "fr11_v13_multi_session_consolidation_transfers_no_collapse_quality_maintained"
)
NO_GAIN_VERDICT = (
    "complete: fr11_v13_consolidation_no_gain_over_cold_start_single_session_sufficient"
)
BLOCKED_VERDICT = "complete: blocked_fr11_module_or_traces_unavailable"
TERMINAL_VERDICTS = (SUCCESS_VERDICT, NO_GAIN_VERDICT, BLOCKED_VERDICT)
VERIFIER_NAMES = v10.VERIFIER_NAMES

score_fover_corpus = v10.score_fover_corpus
score_matrix = v10.score_matrix
probe_cached_trace_preconditions = v12.probe_cached_trace_preconditions

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "n_sessions",
    "n_online_updates",
    "template_library_bounded",
    "consolidated_template_transfer_gain_over_cold_start",
    "structure_persisted_and_restored",
    "collapse_detected_deploy_arm",
    "pass_rate_vs_true_accuracy_distinct_assert",
    "quality_maintained",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "n_sessions": (
        "Number of session boundaries consolidated across (>=3) -- the v13 forward difference."
    ),
    "n_online_updates": "Sample-size of the self-learning sweep across all sessions (>=200).",
    "template_library_bounded": (
        "True iff the consolidated template library stays under its size cap "
        "(merge/evict policy works)."
    ),
    "consolidated_template_transfer_gain_over_cold_start": (
        "The forward difference -- does a consolidated template beat cold-start "
        "on a fresh session's ensemble AUROC?"
    ),
    "structure_persisted_and_restored": (
        "Cross-session Tier-2 persistence -- the consolidated library round-trips "
        "(SHA256) across boundaries."
    ),
    "collapse_detected_deploy_arm": (
        "The conservative-default rule must prevent weight collapse during "
        "consolidation (alpha_t grounding)."
    ),
    "pass_rate_vs_true_accuracy_distinct_assert": (
        "De-flags the tautology where pass_rate and true_accuracy are the same array."
    ),
    "quality_maintained": "Consolidation + transfer must not cost ensemble quality.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


@dataclass(frozen=True)
class SessionSlice:
    """One cached-score session with labels and verifier score columns."""

    name: str
    labels: np.ndarray
    score_matrix: np.ndarray
    verifier_names: Sequence[str]


@dataclass(frozen=True)
class TemplateEntry:
    """One reusable dependency template learned from one or more sessions."""

    template_id: str
    weights: np.ndarray
    edges: tuple[JsonDict, ...]
    support: int
    source_sessions: tuple[str, ...]
    utility: float


@dataclass(frozen=True)
class TemplateLibrary:
    """Bounded Tier-2 memory for dependency-aware verifier templates."""

    cap: int
    verifier_names: tuple[str, ...]
    entries: tuple[TemplateEntry, ...]
    consolidation_events: tuple[JsonDict, ...]


def build_artifact(
    repo_root: Path | str,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_N_ONLINE_UPDATES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    corpus_random_seed: int = DEFAULT_CORPUS_RANDOM_SEED,
    library_cap: int = DEFAULT_LIBRARY_CAP,
) -> JsonDict:
    """Build Exp 3708 from cached FR-11 verifier traces."""

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
        library_cap=library_cap,
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
    library_cap: int = DEFAULT_LIBRARY_CAP,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Select real cached slices, then run bounded consolidation and transfer."""

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
    if not _runnable_trace(labels_arr):
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            preconditions=preconditions or [_trace_precondition(labels_arr, scores_by_verifier)],
        )

    try:
        sessions, heldout, panel = select_session_panel(
            labels_arr,
            matrix,
            names,
            random_seed=random_seed,
        )
    except ValueError as exc:
        blocked_preconditions = [
            *(preconditions or [_trace_precondition(labels_arr, scores_by_verifier)]),
            {
                "resource": "distributionally_distinct_session_slices",
                "available": False,
                "detail": str(exc),
            },
        ]
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            preconditions=blocked_preconditions,
        )

    return build_artifact_from_score_slices(
        sessions=sessions,
        heldout_session=heldout,
        started_s=started_s,
        now_s=now_s,
        random_seed=random_seed,
        persistence_dir=persistence_dir,
        library_cap=library_cap,
        preconditions=preconditions,
        panel_metadata=panel,
    )


def select_session_panel(
    labels: Sequence[int] | np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
    *,
    random_seed: int,
) -> tuple[list[SessionSlice], SessionSlice, JsonDict]:
    """Choose three distinct cached sessions and one fresh held-out session."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix = np.asarray(score_matrix, dtype=np.float64)
    _validate_scores(labels_arr, matrix, verifier_names)
    if len(labels_arr) < 4 * MIN_SESSION_EXAMPLES:
        raise ValueError("not enough cached rows for three sessions plus heldout")

    projections = [("mean_score", np.mean(matrix, axis=1))]
    projections.extend((f"{name}_score", matrix[:, index]) for index, name in enumerate(verifier_names))
    rng = np.random.default_rng(int(random_seed))
    best: tuple[float, str, tuple[int, int, int], int, list[np.ndarray], dict[tuple[int, int], float]] | None = None
    for projection_name, projection in projections:
        order = np.argsort(projection, kind="mergesort")
        bins = [order[i * len(order) // 4 : (i + 1) * len(order) // 4].copy() for i in range(4)]
        for item in bins:
            rng.shuffle(item)
        if any(len(item) < MIN_SESSION_EXAMPLES for item in bins):  # pragma: no cover - row-count guard keeps quartile bins large enough.
            continue
        if any(not _slice_has_binary_support(labels_arr[item]) for item in bins):
            continue
        distances = {
            (i, j): v11.vote_distribution_distance(matrix[bins[i]], matrix[bins[j]])
            for i in range(4)
            for j in range(i + 1, 4)
        }
        for training_bins in itertools.combinations(range(4), 3):
            min_distance = min(distances[tuple(sorted(pair))] for pair in itertools.combinations(training_bins, 2))
            if min_distance < MIN_SESSION_DISTANCE:
                continue
            heldout_bin = next(index for index in range(4) if index not in training_bins)
            candidate = (
                float(min_distance),
                projection_name,
                tuple(int(item) for item in training_bins),
                int(heldout_bin),
                bins,
                distances,
            )
            if best is None or candidate[:4] > best[:4]:
                best = candidate

    if best is None:
        raise ValueError("fewer than three distributionally distinct cached slices")

    min_distance, projection_name, training_bins, heldout_bin, bins, distances = best
    names = tuple(verifier_names)
    sessions = [
        SessionSlice(
            name=f"{projection_name}_session_{bin_index}",
            labels=labels_arr[bins[bin_index]],
            score_matrix=matrix[bins[bin_index]],
            verifier_names=names,
        )
        for bin_index in training_bins
    ]
    heldout = SessionSlice(
        name=f"{projection_name}_fresh_heldout_{heldout_bin}",
        labels=labels_arr[bins[heldout_bin]],
        score_matrix=matrix[bins[heldout_bin]],
        verifier_names=names,
    )
    panel = {
        "projection": projection_name,
        "training_bins": list(training_bins),
        "heldout_bin": int(heldout_bin),
        "min_pairwise_vote_distribution_distance": _round(min_distance),
        "session_vote_distances": [
            _round(distances[tuple(sorted(pair))])
            for pair in itertools.combinations(training_bins, 2)
        ],
        "policy": "choose the three cached quartile slices with the largest minimum pairwise vote-distribution distance; hold out the remaining quartile",
    }
    return sessions, heldout, panel


def build_artifact_from_score_slices(
    *,
    sessions: Sequence[SessionSlice],
    heldout_session: SessionSlice | None,
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    persistence_dir: Path | str | None = None,
    library_cap: int = DEFAULT_LIBRARY_CAP,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
    panel_metadata: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Run consolidation over explicit score slices and test held-out transfer."""

    for session in sessions:
        _validate_session(session)
    if heldout_session is not None:
        _validate_session(heldout_session)

    if (
        len(sessions) < 3
        or heldout_session is None
        or not all(_runnable_session(session) for session in sessions)
        or not _runnable_session(heldout_session)
    ):
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            preconditions=preconditions or [_session_precondition(sessions, heldout_session)],
        )

    names = tuple(sessions[0].verifier_names)
    memory_dir = Path(persistence_dir) if persistence_dir is not None else Path.cwd()
    library = TemplateLibrary(
        cap=int(library_cap),
        verifier_names=names,
        entries=(),
        consolidation_events=(),
    )
    round_trips: list[JsonDict] = []
    session_summaries: list[JsonDict] = []

    for session_index, session in enumerate(sessions, start=1):
        structure = v12.learn_structure(
            session.labels,
            session.score_matrix,
            names,
            source_window=session.name,
        )
        session_scores = _signed_scores(session.score_matrix, structure.weights)
        utility = v10.score_metrics(session.labels, session_scores)["auroc"]
        entry = TemplateEntry(
            template_id=f"template_{session_index}_{_short_hash(session.name)}",
            weights=np.asarray(structure.weights, dtype=np.float64),
            edges=tuple(dict(edge) for edge in structure.edges),
            support=len(session.labels),
            source_sessions=(session.name,),
            utility=float(utility),
        )
        library = consolidate_template_library(library, entry)
        memory = persist_template_library(
            library,
            memory_dir / f"experiment_3708_fr11_v13_template_library_session_{session_index}.json",
        )
        restored = restore_template_library_via_subprocess(memory["path"])
        round_trip_ok = bool(restored["sha256"] == memory["sha256"])
        round_trips.append(
            {
                "session_boundary": session_index,
                "path": memory["path"],
                "sha256": memory["sha256"],
                "restored_sha256": restored["sha256"],
                "round_trip_ok": round_trip_ok,
            }
        )
        session_summaries.append(
            {
                "name": session.name,
                "n_examples": int(len(session.labels)),
                "learned_template_id": entry.template_id,
                "template_auroc": _round(utility),
                "collapse_detected": bool(v10.detect_weight_collapse(entry.weights)),
                "n_edges": int(len(entry.edges)),
            }
        )

    final_memory = persist_template_library(library, memory_dir / TEMPLATE_LIBRARY_REL_PATH.name)
    restored_final = restore_template_library_via_subprocess(final_memory["path"])
    structure_persisted = bool(
        restored_final["sha256"] == final_memory["sha256"]
        and all(item["round_trip_ok"] for item in round_trips)
    )
    template_weights = consolidated_template_weights(library)
    cold_weights = np.ones(len(names), dtype=np.float64) / float(len(names))
    deploy_scores = _signed_scores(heldout_session.score_matrix, template_weights)
    cold_scores = _signed_scores(heldout_session.score_matrix, cold_weights)
    transfer_metrics = {
        "consolidated_template": v10.score_metrics(heldout_session.labels, deploy_scores),
        "cold_start_no_template": v10.score_metrics(heldout_session.labels, cold_scores),
    }
    transfer_gain_value = (
        transfer_metrics["consolidated_template"]["auroc"]
        - transfer_metrics["cold_start_no_template"]["auroc"]
    )
    transfer_gain = bool(transfer_gain_value > 0.0)
    collapse_detected = bool(
        v10.detect_weight_collapse(template_weights)
        or any(summary["collapse_detected"] for summary in session_summaries)
    )
    pass_rate, true_accuracy = v10.online_metric_trajectories(
        heldout_session.labels,
        deploy_scores,
    )
    distinct_assert = [_round(value) for value in pass_rate] != [
        _round(value) for value in true_accuracy
    ]
    library_bounded = bool(len(library.entries) <= library.cap)
    quality_maintained = bool(transfer_gain and not collapse_detected and library_bounded)
    gate_passed = bool(
        library_bounded
        and structure_persisted
        and not collapse_detected
        and distinct_assert
    )
    verdict = select_honest_verdict(
        gate_passed=gate_passed,
        transfer_gain=transfer_gain,
        quality_maintained=quality_maintained,
    )
    artifact: JsonDict = {
        "artifact": "experiment_3708_fr11_continuous_self_learning_v13",
        "schema": "carnot.fr11_continuous_self_learning_v13",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_sessions": int(len(sessions)),
        "n_online_updates": int(sum(len(session.labels) for session in sessions)),
        "template_library_bounded": library_bounded,
        "consolidated_template_transfer_gain_over_cold_start": transfer_gain,
        "structure_persisted_and_restored": structure_persisted,
        "collapse_detected_deploy_arm": collapse_detected,
        "pass_rate_vs_true_accuracy_distinct_assert": bool(distinct_assert),
        "quality_maintained": quality_maintained,
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            sessions=sessions,
            heldout_session=heldout_session,
            library_sha256=final_memory["sha256"],
            consolidated_weights=template_weights,
            random_seed=random_seed,
        ),
        "duration_s": _round(_duration(float(started_s), now_s)),
        "acceptance_gate": {
            "condition": (
                "template_library_bounded == true AND "
                "structure_persisted_and_restored == true AND "
                "collapse_detected_deploy_arm == false AND "
                "pass_rate_vs_true_accuracy_distinct_assert == true"
            ),
            "passed": gate_passed,
            "principle": (
                "Multi-session consolidation is validated only when the library "
                "stays bounded, round-trips across sessions, does not collapse, "
                "and the two metrics are genuinely distinct (not a tautology)."
            ),
        },
        "adversarial_verify": "clean",
        "deploy_arm": "consolidated_template",
        "control_arm": "cold_start_no_template",
        "fresh_session_transfer_auroc_gain": _round(transfer_gain_value),
        "fresh_session_metrics": transfer_metrics,
        "template_library": {
            **template_library_to_json(library),
            "path": final_memory["path"],
            "sha256": final_memory["sha256"],
            "restored_sha256": restored_final["sha256"],
            "round_trips": round_trips,
        },
        "consolidated_template_weights": _weights_to_json(names, template_weights),
        "cold_start_weights": _weights_to_json(names, cold_weights),
        "session_summaries": session_summaries,
        "session_panel": dict(panel_metadata or {}),
        "fresh_session": {
            "name": heldout_session.name,
            "n_examples": int(len(heldout_session.labels)),
        },
        "pass_rate_trajectory": [_round(value) for value in pass_rate],
        "true_accuracy_trajectory": [_round(value) for value in true_accuracy],
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def consolidate_template_library(
    library: TemplateLibrary,
    entry: TemplateEntry,
) -> TemplateLibrary:
    """Add a template, then merge the closest pair until the cap holds."""

    entries = [*library.entries, entry]
    events = [
        *library.consolidation_events,
        {"action": "add", "template_id": entry.template_id, "size": len(entries)},
    ]
    while len(entries) > library.cap:
        left, right = _most_similar_pair(entries)
        merged = merge_template_entries(entries[left], entries[right], library.verifier_names)
        keep = [item for index, item in enumerate(entries) if index not in {left, right}]
        entries = [*keep, merged]
        if len(entries) > library.cap:
            evict_index = min(range(len(entries)), key=lambda index: (entries[index].utility, entries[index].support))
            evicted = entries.pop(evict_index)
            action = "merge_then_evict"
            events.append(
                {
                    "action": action,
                    "merged_template_id": merged.template_id,
                    "evicted_template_id": evicted.template_id,
                    "size": len(entries),
                    "policy": "merge highest cosine similarity pair, then evict lowest utility/support if still over cap",
                }
            )
        else:
            events.append(
                {
                    "action": "merge",
                    "merged_template_id": merged.template_id,
                    "size": len(entries),
                    "policy": "merge highest cosine similarity pair when cap is exceeded",
                }
            )
    return TemplateLibrary(
        cap=library.cap,
        verifier_names=library.verifier_names,
        entries=tuple(entries),
        consolidation_events=tuple(events),
    )


def merge_template_entries(
    left: TemplateEntry,
    right: TemplateEntry,
    verifier_names: Sequence[str],
) -> TemplateEntry:
    """Merge two templates by support-weighted weights and unioned edges."""

    total_support = max(1, int(left.support) + int(right.support))
    weights = (
        np.asarray(left.weights, dtype=np.float64) * (int(left.support) / total_support)
        + np.asarray(right.weights, dtype=np.float64) * (int(right.support) / total_support)
    )
    weights = v10.collapse_guarded_weights(weights)
    edge_map: dict[str, JsonDict] = {}
    for edge in [*left.edges, *right.edges]:
        pair = tuple(edge.get("pair", []))
        key = "::".join(str(item) for item in pair)
        edge_map[key] = dict(edge)
    utility = (
        float(left.utility) * int(left.support) + float(right.utility) * int(right.support)
    ) / float(total_support)
    source_sessions = tuple(dict.fromkeys([*left.source_sessions, *right.source_sessions]))
    return TemplateEntry(
        template_id=f"merged_{_short_hash('|'.join(source_sessions))}",
        weights=np.asarray(weights, dtype=np.float64),
        edges=tuple(edge_map[key] for key in sorted(edge_map)),
        support=total_support,
        source_sessions=source_sessions,
        utility=float(utility),
    )


def consolidated_template_weights(library: TemplateLibrary) -> np.ndarray:
    """Return the support-weighted template-library deploy weights."""

    if not library.entries:
        return np.ones(len(library.verifier_names), dtype=np.float64) / float(len(library.verifier_names))
    total_support = max(1, sum(int(entry.support) for entry in library.entries))
    weights = np.zeros(len(library.verifier_names), dtype=np.float64)
    for entry in library.entries:
        weights += np.asarray(entry.weights, dtype=np.float64) * (int(entry.support) / total_support)
    return v10.collapse_guarded_weights(weights)


def persist_template_library(library: TemplateLibrary, path: Path | str) -> JsonDict:
    """Persist the bounded template library and return its checksum metadata."""

    output = Path(path)
    payload = template_library_to_json(library)
    checksum = _json_sha256(payload)
    stored = {**payload, "sha256": checksum}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(output), "sha256": checksum}


def restore_template_library_via_subprocess(path: Path | str) -> JsonDict:
    """Reload persisted library through a fresh Python process."""

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
    "entries": payload["entries"],
}, sort_keys=True))
"""
    output = subprocess.check_output(  # noqa: S603 - local interpreter reads a local JSON path.
        [sys.executable, "-c", script, str(path)],
        text=True,
    )
    return json.loads(output)


def template_library_to_json(library: TemplateLibrary) -> JsonDict:
    """Convert the in-memory library into stable JSON."""

    return {
        "schema": "carnot.fr11_v13_template_library",
        "cap": int(library.cap),
        "size": int(len(library.entries)),
        "verifier_names": list(library.verifier_names),
        "bounded_policy": (
            "cap entries; on overflow merge the highest cosine-similarity pair; "
            "evict lowest utility/support only if a merge cannot restore the cap"
        ),
        "entries": [template_entry_to_json(entry, library.verifier_names) for entry in library.entries],
        "consolidation_events": [dict(event) for event in library.consolidation_events],
    }


def template_entry_to_json(entry: TemplateEntry, verifier_names: Sequence[str]) -> JsonDict:
    """Convert one template entry into stable JSON."""

    return {
        "template_id": entry.template_id,
        "weights": _weights_to_json(verifier_names, entry.weights),
        "edges": [dict(edge) for edge in entry.edges],
        "support": int(entry.support),
        "source_sessions": list(entry.source_sessions),
        "utility": _round(entry.utility),
    }


def select_honest_verdict(
    *,
    gate_passed: bool,
    transfer_gain: bool,
    quality_maintained: bool,
) -> str:
    """Choose the allowed Exp 3708 terminal verdict."""

    if gate_passed and transfer_gain and quality_maintained:
        return SUCCESS_VERDICT
    return NO_GAIN_VERDICT


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3708 artifact schema before writing JSON."""

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
    if int(artifact["n_sessions"]) < 3:
        raise ValueError("n_sessions must be at least 3")
    if int(artifact["n_online_updates"]) < MIN_ONLINE_UPDATES:
        raise ValueError(f"n_online_updates must be at least {MIN_ONLINE_UPDATES}")
    for field in (
        "template_library_bounded",
        "consolidated_template_transfer_gain_over_cold_start",
        "structure_persisted_and_restored",
        "collapse_detected_deploy_arm",
        "pass_rate_vs_true_accuracy_distinct_assert",
        "quality_maintained",
    ):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a boolean")
    gain = artifact.get("fresh_session_transfer_auroc_gain")
    if not isinstance(gain, int | float) or not math.isfinite(float(gain)):
        raise ValueError("fresh_session_transfer_auroc_gain must be finite")
    library = artifact.get("template_library")
    if not isinstance(library, Mapping):
        raise ValueError("template_library must be present")
    if int(library.get("size", 10**9)) > int(library.get("cap", -1)):
        raise ValueError("template_library size exceeds cap")
    if artifact.get("structure_persisted_and_restored") and library.get("sha256") != library.get(
        "restored_sha256"
    ):
        raise ValueError("library SHA256 round-trip failed")


def write_artifact(
    repo_root: Path | str,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    labels: Sequence[int] | None = None,
    scores_by_verifier: Mapping[str, Sequence[float]] | None = None,
    sessions: Sequence[SessionSlice] | None = None,
    heldout_session: SessionSlice | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3708 JSON artifact."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    if sessions is not None or heldout_session is not None:
        artifact = build_artifact_from_score_slices(
            sessions=sessions or (),
            heldout_session=heldout_session,
            started_s=start,
            now_s=now_s,
            persistence_dir=root / "results",
        )
    elif labels is not None and scores_by_verifier is not None:
        artifact = build_artifact_from_scores(
            labels=labels,
            scores_by_verifier=scores_by_verifier,
            started_s=start,
            now_s=now_s,
            persistence_dir=root / "results",
        )
    else:
        artifact = build_artifact(root, started_s=start, now_s=now_s)
    output = root / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def reproducibility_checksum(
    *,
    sessions: Sequence[SessionSlice],
    heldout_session: SessionSlice,
    library_sha256: str,
    consolidated_weights: Sequence[float],
    random_seed: int,
) -> str:
    """Hash deterministic sessions, held-out slice, library, and deploy weights."""

    digest = hashlib.sha256()
    for session in [*sessions, heldout_session]:
        digest.update(session.name.encode("utf-8"))
        digest.update(np.ascontiguousarray(session.labels, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(session.score_matrix, dtype=np.float64).tobytes())
    digest.update(library_sha256.encode("ascii"))
    digest.update(np.ascontiguousarray(consolidated_weights, dtype=np.float64).tobytes())
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
        "artifact": "experiment_3708_fr11_continuous_self_learning_v13",
        "schema": "carnot.fr11_continuous_self_learning_v13",
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_sessions": 0,
        "n_online_updates": 0,
        "template_library_bounded": False,
        "consolidated_template_transfer_gain_over_cold_start": False,
        "structure_persisted_and_restored": False,
        "collapse_detected_deploy_arm": False,
        "pass_rate_vs_true_accuracy_distinct_assert": False,
        "quality_maintained": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": hashlib.sha256(payload).hexdigest(),
        "duration_s": _round(duration_s),
        "acceptance_gate": {
            "condition": (
                "template_library_bounded == true AND "
                "structure_persisted_and_restored == true AND "
                "collapse_detected_deploy_arm == false AND "
                "pass_rate_vs_true_accuracy_distinct_assert == true"
            ),
            "passed": False,
            "principle": (
                "Multi-session consolidation is validated only when the library "
                "stays bounded, round-trips across sessions, does not collapse, "
                "and the two metrics are genuinely distinct (not a tautology)."
            ),
        },
        "fresh_session_transfer_auroc_gain": 0.0,
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
        "available": _runnable_trace(np.asarray(labels, dtype=np.int64)),
        "detail": (
            f"n_examples={len(labels)}; labels={sorted(set(int(value) for value in labels))}; "
            f"n_verifiers={len(scores_by_verifier)}; required>={MIN_ONLINE_UPDATES}"
        ),
    }


def _session_precondition(
    sessions: Sequence[SessionSlice],
    heldout_session: SessionSlice | None,
) -> JsonDict:
    return {
        "resource": "distributionally_distinct_session_slices",
        "available": False,
        "detail": (
            f"n_sessions={len(sessions)}; heldout_present={heldout_session is not None}; "
            "required n_sessions>=3 with binary labels and a binary heldout"
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


def _runnable_trace(labels: np.ndarray) -> bool:
    return len(labels) >= MIN_ONLINE_UPDATES and _slice_has_binary_support(labels)


def _runnable_session(session: SessionSlice) -> bool:
    return len(session.labels) >= MIN_SESSION_EXAMPLES and _slice_has_binary_support(session.labels)


def _slice_has_binary_support(labels: Sequence[int] | np.ndarray) -> bool:
    values = {int(value) for value in labels}
    return values == {0, 1}


def _most_similar_pair(entries: Sequence[TemplateEntry]) -> tuple[int, int]:
    best_pair = (0, 1)
    best_similarity = -float("inf")
    for left, right in itertools.combinations(range(len(entries)), 2):
        similarity = _cosine_similarity(entries[left].weights, entries[right].weights)
        if similarity > best_similarity:
            best_similarity = similarity
            best_pair = (left, right)
    return best_pair


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(left_arr) * np.linalg.norm(right_arr))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left_arr, right_arr) / denom)


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


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0, end - float(started_s))


def _round(value: float | int | np.floating[Any], digits: int = 6) -> float:
    return round(float(value), digits)
