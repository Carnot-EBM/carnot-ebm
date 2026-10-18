"""Exp 2881 FR-11 RecMem-style recurrence-triggered consolidation.

This prototype tests the RecMem timing idea on Carnot's existing offline replay
evidence: keep verifier-feedback events in a cheap subconscious buffer, detect
when the same failure motif recurs, and consolidate only those recurring
clusters. The implementation deliberately uses deterministic metadata features
instead of a live LLM summarizer, so the artifact can account for token-cost
avoidance without pretending to perform semantic extraction with a model.

Spec: REQ-LEARN-2881,
      SCENARIO-LEARN-2881,
      SCENARIO-LEARN-2881-GUARD.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.eval.fr11_continuous_self_learning_replay_v3 import (
    BACKEND_MODULE_PATH,
    ExperimentConfig as ReplayExperimentConfig,
    build_backend_rows,
    load_clean_replay_corpus,
)

OUTPUT_FILENAME = "experiment_2881_fr11_recmem_recurrence_trigger_v1.json"
MEMORY_STATE_FILENAME = "fr11_recmem_recurrence_trigger_2881_memory.json"
EXP2868_FILENAME = "experiment_2868_offline_recurrence_backend_adapter_v2.json"
EXP2869_FILENAME = "experiment_2869_fr11_continuous_self_learning_replay_v3.json"
EXP2869_MEMORY_FILENAME = "fr11_continuous_self_learning_replay_2869_memory.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260522"
RANDOM_SEED = 2881
RECURRENCE_THRESHOLD = 0.6
MIN_SUPPORT = 2
MAX_LOOPS = 3
REPLAY_N_EXAMPLES = 9

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "continuous_self_learning_task",
    "recmem_trigger_ready",
    "source_artifacts",
    "recurrence_threshold",
    "n_events_ingested",
    "n_recurrence_clusters",
    "n_consolidations_triggered",
    "eager_consolidations_avoided",
    "token_reduction_proxy_pct",
    "memory_hash_before",
    "memory_hash_after",
    "contradiction_rate",
    "duplicate_rate",
    "forgetting_regression_count",
    "live_llm_called",
    "tests_run",
    "field_principles",
    "run_date",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal complete:/blocked_ verdict; no inferred live success.",
    "continuous_self_learning_task": "True because this is the FR-11 memory task.",
    "recmem_trigger_ready": (
        "True only when recurrence support exists and drift guards report no "
        "contradiction or forgetting regression."
    ),
    "source_artifacts": "Existing offline replay artifacts loaded before consolidation.",
    "recurrence_threshold": "Feature-similarity threshold used by the recurrence detector.",
    "n_events_ingested": "Events accepted into the subconscious buffer before any consolidation.",
    "n_recurrence_clusters": "Sustained failure-motif clusters detected before drift gating.",
    "n_consolidations_triggered": "Clusters actually written into the consolidated memory state.",
    "eager_consolidations_avoided": "Per-event eager consolidation calls not made.",
    "token_reduction_proxy_pct": "Deterministic byte/token-like proxy reduction vs eager writes.",
    "memory_hash_before": "Hash of prior replay memory plus the Exp 2881 memory file before write.",
    "memory_hash_after": "Hash of prior replay memory plus the Exp 2881 memory file after write.",
    "contradiction_rate": "Fraction of motif groups with conflicting final correctness labels.",
    "duplicate_rate": "Fraction of buffered events repeating an already-seen motif key.",
    "forgetting_regression_count": "Rows whose final energy or correctness regressed.",
    "live_llm_called": "False because summaries are deterministic metadata strings.",
    "duration_s": "Real wall-clock duration; no sleep padding.",
}

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the Exp 2881 RecMem recurrence-trigger prototype."""

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    run_date: str = RUN_DATE
    random_seed: int = RANDOM_SEED
    recurrence_threshold: float = RECURRENCE_THRESHOLD
    min_support: int = MIN_SUPPORT
    max_loops: int = MAX_LOOPS
    replay_n_examples: int = REPLAY_N_EXAMPLES
    memory_state_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        if not 0.0 <= self.recurrence_threshold <= 1.0:
            raise ValueError("recurrence_threshold must be in [0.0, 1.0]")
        if self.min_support < 2:
            raise ValueError("min_support must be >= 2")
        if self.max_loops < 1:
            raise ValueError("max_loops must be >= 1")
        if self.replay_n_examples < 1:
            raise ValueError("replay_n_examples must be >= 1")

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def output_path(self) -> Path:
        return self.output_dir() / OUTPUT_FILENAME

    def exp2868_path(self) -> Path:
        return self.output_dir() / EXP2868_FILENAME

    def exp2869_path(self) -> Path:
        return self.output_dir() / EXP2869_FILENAME

    def exp2869_memory_path(self) -> Path:
        return self.output_dir() / EXP2869_MEMORY_FILENAME

    def memory_path(self) -> Path:
        return self.memory_state_path or self.output_dir() / MEMORY_STATE_FILENAME

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


@dataclass(frozen=True)
class ReplayEvent:
    """One verifier-feedback event held in the subconscious RecMem buffer."""

    event_id: str
    source: str
    energy_before: float
    energy_after: float
    correctness_before: bool
    correctness_after: bool
    localized_violations: tuple[str, ...] = ()
    early_exit_reason: str = ""

    @property
    def energy_delta(self) -> float:
        return _round_float(float(self.energy_before) - float(self.energy_after))

    @property
    def motif_key(self) -> str:
        if self.localized_violations:
            return "+".join(sorted(self.localized_violations))
        if not self.correctness_before:
            return "unlocalized_failure"
        return "clean"

    @property
    def is_failure(self) -> bool:
        return bool(self.localized_violations) or not self.correctness_before

    @property
    def feature_tokens(self) -> frozenset[str]:
        tokens = {
            f"source_{_slug(self.source)}",
            "initial_correct" if self.correctness_before else "initial_incorrect",
            "final_correct" if self.correctness_after else "final_incorrect",
            "energy_improved" if self.energy_delta > 0.0 else "energy_not_improved",
        }
        if self.localized_violations:
            tokens.add("verifier_failure")
            for violation in self.localized_violations:
                tokens.add(f"violation_{_slug(violation)}")
                tokens.update(_tokenize(violation))
        else:
            tokens.add("clean" if self.correctness_before else "unlocalized_failure")
        tokens.update(_tokenize(self.early_exit_reason))
        return frozenset(token for token in tokens if token)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source": self.source,
            "energy_before": _round_float(self.energy_before),
            "energy_after": _round_float(self.energy_after),
            "energy_delta": self.energy_delta,
            "correctness_before": self.correctness_before,
            "correctness_after": self.correctness_after,
            "localized_violations": list(self.localized_violations),
            "early_exit_reason": self.early_exit_reason,
            "motif_key": self.motif_key,
        }


@dataclass(frozen=True)
class RecurrenceCluster:
    """A group of buffered events whose local feature sets recur."""

    cluster_id: str
    events: tuple[ReplayEvent, ...]
    recurrence_threshold: float
    min_support: int

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self.events)

    @property
    def support(self) -> int:
        return len(self.events)

    @property
    def motif_key(self) -> str:
        counts = Counter(event.motif_key for event in self.events if event.is_failure)
        if counts:
            return counts.most_common(1)[0][0]
        return self.events[0].motif_key if self.events else "empty"

    @property
    def consolidation_ready(self) -> bool:
        return self.support >= self.min_support and any(event.is_failure for event in self.events)

    @property
    def similarity_floor(self) -> float:
        if len(self.events) < 2:
            return 1.0
        similarities: list[float] = []
        for left_index, left_event in enumerate(self.events):
            for right_event in self.events[left_index + 1 :]:
                similarities.append(_jaccard(left_event.feature_tokens, right_event.feature_tokens))
        return _round_float(min(similarities) if similarities else 1.0)


class SubconsciousRecurrenceBuffer:
    """Cheap event buffer that intentionally does not consolidate on ingest.

    RecMem's operational distinction is timing: raw interactions first go into
    a low-cost store, and expensive consolidation happens only after recurrence.
    Here the low-cost store is just deterministic metadata plus local features,
    which is enough for the offline prototype and avoids any live LLM call.
    """

    def __init__(self) -> None:
        self._events: list[ReplayEvent] = []

    def ingest(self, event: ReplayEvent) -> None:
        self._events.append(event)

    def ingest_many(self, events: Iterable[ReplayEvent]) -> None:
        for event in events:
            self.ingest(event)

    @property
    def events(self) -> tuple[ReplayEvent, ...]:
        return tuple(self._events)

    @property
    def n_events(self) -> int:
        return len(self._events)


class RecurrenceDetector:
    """Greedy recurrence detector over deterministic verifier-event features."""

    def __init__(self, *, recurrence_threshold: float, min_support: int = MIN_SUPPORT) -> None:
        if not 0.0 <= recurrence_threshold <= 1.0:
            raise ValueError("recurrence_threshold must be in [0.0, 1.0]")
        if min_support < 2:
            raise ValueError("min_support must be >= 2")
        self.recurrence_threshold = float(recurrence_threshold)
        self.min_support = int(min_support)

    def detect(self, events: Sequence[ReplayEvent]) -> list[RecurrenceCluster]:
        clusters: list[list[ReplayEvent]] = []
        for event in events:
            best_index: int | None = None
            best_score = -1.0
            for index, cluster in enumerate(clusters):
                if any(member.is_failure != event.is_failure for member in cluster):
                    continue
                score = max(
                    _jaccard(event.feature_tokens, member.feature_tokens) for member in cluster
                )
                if score >= self.recurrence_threshold and score > best_score:
                    best_index = index
                    best_score = score
            if best_index is None:
                clusters.append([event])
            else:
                clusters[best_index].append(event)
        return [
            RecurrenceCluster(
                cluster_id=_cluster_id(cluster),
                events=tuple(cluster),
                recurrence_threshold=self.recurrence_threshold,
                min_support=self.min_support,
            )
            for cluster in clusters
        ]


def evaluate_recmem_events(
    events: Sequence[ReplayEvent],
    *,
    recurrence_threshold: float = RECURRENCE_THRESHOLD,
    min_support: int = MIN_SUPPORT,
) -> dict[str, Any]:
    """Evaluate recurrence triggers and drift guards for buffered events.

    Spec: REQ-LEARN-2881-1, REQ-LEARN-2881-2, REQ-LEARN-2881-4
    """

    detector = RecurrenceDetector(
        recurrence_threshold=recurrence_threshold,
        min_support=min_support,
    )
    clusters = detector.detect(events)
    recurrence_clusters = [cluster for cluster in clusters if cluster.consolidation_ready]
    duplicate_rate = _duplicate_rate(events)
    contradiction_rate = _contradiction_rate(events)
    forgetting_regression_count = _forgetting_regression_count(events)
    guard_passed = contradiction_rate == 0.0 and forgetting_regression_count == 0
    consolidations = (
        [_consolidation_for_cluster(cluster) for cluster in recurrence_clusters]
        if guard_passed
        else []
    )
    eager_token_proxy = sum(_event_token_proxy(event) for event in events)
    triggered_token_proxy = sum(_consolidation_token_proxy(item) for item in consolidations)
    token_reduction_proxy_pct = _token_reduction(eager_token_proxy, triggered_token_proxy)
    n_consolidations = len(consolidations)
    return {
        "recmem_trigger_ready": bool(recurrence_clusters) and guard_passed,
        "recurrence_threshold": _round_float(recurrence_threshold),
        "n_events_ingested": len(events),
        "n_recurrence_clusters": len(recurrence_clusters),
        "n_consolidations_triggered": n_consolidations,
        "eager_consolidations_avoided": max(0, len(events) - n_consolidations),
        "token_reduction_proxy_pct": token_reduction_proxy_pct,
        "token_proxy_before": eager_token_proxy,
        "token_proxy_after": triggered_token_proxy,
        "duplicate_rate": duplicate_rate,
        "contradiction_rate": contradiction_rate,
        "forgetting_regression_count": forgetting_regression_count,
        "consolidations": consolidations,
        "threshold_sensitivity": _threshold_sensitivity(events, recurrence_threshold, min_support),
        "live_llm_called": False,
    }


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    tests_run: Sequence[str] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Run the Exp 2881 RecMem recurrence-triggered consolidation workflow."""

    active_config = config or ExperimentConfig()
    started_at = active_config.start_time()
    events, preconditions, blocker = _load_offline_events(active_config)
    if blocker is not None:
        artifact = _blocked_artifact(
            config=active_config,
            honest_verdict=blocker,
            preconditions=preconditions,
            duration_s=active_config.clock() - started_at,
            tests_run=tests_run or [],
        )
        if write:
            _write_json(active_config.output_path(), artifact)
        return artifact

    memory_paths = [active_config.exp2869_memory_path(), active_config.memory_path()]
    memory_hash_before, before_manifest = _hash_memory_state(memory_paths, active_config.repo_root)
    evaluation = evaluate_recmem_events(
        events,
        recurrence_threshold=active_config.recurrence_threshold,
        min_support=active_config.min_support,
    )
    memory_state = _build_memory_state(active_config, events, evaluation)
    _write_json(active_config.memory_path(), memory_state)
    memory_hash_after, after_manifest = _hash_memory_state(memory_paths, active_config.repo_root)
    changed_paths = _changed_memory_paths(before_manifest, after_manifest)
    permitted_path = _relative_path(active_config.memory_path(), active_config.repo_root)
    permitted_changes = all(path == permitted_path for path in changed_paths)
    preconditions.append(
        {
            "check": "permitted_memory_changes",
            "passed": permitted_changes,
            "observed": changed_paths,
        }
    )
    ready = (
        bool(evaluation["recmem_trigger_ready"])
        and permitted_changes
        and memory_hash_before != memory_hash_after
    )
    artifact: dict[str, Any] = {
        "artifact": "experiment_2881_fr11_recmem_recurrence_trigger_v1",
        "schema": "carnot.fr11.recmem_recurrence_trigger.v1",
        "honest_verdict": (
            "complete: recurrence-triggered consolidation ready without live LLM"
            if ready
            else "blocked_recmem_drift_guard_or_no_recurrence"
        ),
        "continuous_self_learning_task": True,
        "recmem_trigger_ready": ready,
        "source_artifacts": _source_artifacts(active_config),
        "recurrence_threshold": evaluation["recurrence_threshold"],
        "n_events_ingested": evaluation["n_events_ingested"],
        "n_recurrence_clusters": evaluation["n_recurrence_clusters"],
        "n_consolidations_triggered": (evaluation["n_consolidations_triggered"] if ready else 0),
        "eager_consolidations_avoided": evaluation["eager_consolidations_avoided"],
        "token_reduction_proxy_pct": evaluation["token_reduction_proxy_pct"],
        "memory_hash_before": memory_hash_before,
        "memory_hash_after": memory_hash_after,
        "contradiction_rate": evaluation["contradiction_rate"],
        "duplicate_rate": evaluation["duplicate_rate"],
        "forgetting_regression_count": evaluation["forgetting_regression_count"],
        "live_llm_called": False,
        "tests_run": list(tests_run or []),
        "field_principles": FIELD_PRINCIPLES,
        "run_date": active_config.run_date,
        "duration_s": _round_float(active_config.clock() - started_at),
        "preconditions_checked": preconditions,
        "token_proxy_before": evaluation["token_proxy_before"],
        "token_proxy_after": evaluation["token_proxy_after"],
        "threshold_sensitivity": evaluation["threshold_sensitivity"],
        "consolidated_memory_path": _relative_path(
            active_config.memory_path(),
            active_config.repo_root,
        ),
        "methodology_note": (
            "Offline RecMem timing prototype over deterministic verifier metadata. "
            "No live LLM, generator, model weights, or eager replay writes are used."
        ),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    if write:
        _write_json(active_config.output_path(), artifact)
    return artifact


def _round_float(value: float) -> float:
    return round(float(value), 12)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _slug(value: object) -> str:
    tokens = _tokenize(str(value))
    return "_".join(tokens) if tokens else "unknown"


def _tokenize(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(value).lower()))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _cluster_id(events: Sequence[ReplayEvent]) -> str:
    encoded = "|".join(sorted(event.event_id for event in events)).encode("utf-8")
    return f"cluster-{hashlib.sha256(encoded).hexdigest()[:12]}"


def _event_token_proxy(event: ReplayEvent) -> int:
    # The fixed overhead approximates one eager summarizer call envelope.
    return len(json.dumps(event.to_dict(), sort_keys=True)) + 96


def _consolidation_token_proxy(consolidation: Mapping[str, Any]) -> int:
    return len(str(consolidation.get("deterministic_summary", ""))) + 96


def _token_reduction(before: int, after: int) -> float:
    if before <= 0:
        return 0.0
    return _round_float(max(0.0, (1.0 - (after / before)) * 100.0))


def _duplicate_rate(events: Sequence[ReplayEvent]) -> float:
    if not events:
        return 0.0
    counts = Counter(event.motif_key for event in events)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return _round_float(duplicates / len(events))


def _contradiction_rate(events: Sequence[ReplayEvent]) -> float:
    groups: dict[str, set[bool]] = {}
    for event in events:
        if event.motif_key == "clean":
            continue
        groups.setdefault(event.motif_key, set()).add(event.correctness_after)
    if not groups:
        return 0.0
    contradictions = sum(1 for labels in groups.values() if len(labels) > 1)
    return _round_float(contradictions / len(groups))


def _forgetting_regression_count(events: Sequence[ReplayEvent]) -> int:
    return sum(
        1
        for event in events
        if event.energy_after > event.energy_before
        or (event.correctness_before and not event.correctness_after)
    )


def _consolidation_for_cluster(cluster: RecurrenceCluster) -> dict[str, Any]:
    energy_deltas = [event.energy_delta for event in cluster.events]
    sources = sorted({event.source for event in cluster.events})
    violations = sorted(
        {violation for event in cluster.events for violation in event.localized_violations}
    )
    summary = (
        f"Recurring {cluster.motif_key} motif across {len(sources)} source(s); "
        f"support={cluster.support}; mean_energy_delta={_round_float(_mean(energy_deltas))}."
    )
    return {
        "cluster_id": cluster.cluster_id,
        "motif_key": cluster.motif_key,
        "support": cluster.support,
        "event_ids": list(cluster.event_ids),
        "sources": sources,
        "localized_violations": violations,
        "mean_energy_delta": _round_float(_mean(energy_deltas)),
        "similarity_floor": cluster.similarity_floor,
        "deterministic_summary": summary,
    }


def _threshold_sensitivity(
    events: Sequence[ReplayEvent],
    recurrence_threshold: float,
    min_support: int,
) -> list[dict[str, Any]]:
    thresholds = sorted(
        {
            _round_float(max(0.0, min(1.0, recurrence_threshold - 0.1))),
            _round_float(recurrence_threshold),
            _round_float(max(0.0, min(1.0, recurrence_threshold + 0.1))),
        }
    )
    sensitivity: list[dict[str, Any]] = []
    for threshold in thresholds:
        clusters = RecurrenceDetector(
            recurrence_threshold=threshold,
            min_support=min_support,
        ).detect(events)
        ready_clusters = [cluster for cluster in clusters if cluster.consolidation_ready]
        sensitivity.append(
            {
                "threshold": threshold,
                "n_recurrence_clusters": len(ready_clusters),
                "n_consolidations_if_guards_pass": len(ready_clusters),
            }
        )
    return sensitivity


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_artifacts(config: ExperimentConfig) -> list[str]:
    return [
        _relative_path(config.exp2868_path(), config.repo_root),
        _relative_path(config.exp2869_path(), config.repo_root),
        _relative_path(config.exp2869_memory_path(), config.repo_root),
    ]


def _load_backend(config: ExperimentConfig) -> tuple[type[Any] | None, list[dict[str, Any]]]:
    exp2868 = _read_json(config.exp2868_path())
    module_path = str(exp2868.get("backend_module_path") or "")
    checks = [
        {
            "check": "repo_root",
            "passed": config.repo_root.is_dir(),
            "observed": str(config.repo_root),
        },
        {
            "check": "exp2868_artifact",
            "passed": bool(exp2868),
            "observed": str(config.exp2868_path()) if exp2868 else "missing",
        },
        {
            "check": "exp2868_backend_module_path",
            "passed": module_path == BACKEND_MODULE_PATH,
            "observed": module_path or "missing",
        },
    ]
    backend_cls: type[Any] | None = None
    imported = False
    if module_path:
        try:
            module = importlib.import_module(module_path)
            backend_cls = getattr(module, "OfflineRecurrenceReplayBackend")
            imported = True
        except (AttributeError, ImportError, ModuleNotFoundError):
            imported = False
    checks.append(
        {"check": "offline_backend_imported", "passed": imported, "observed": module_path}
    )
    return backend_cls, checks


def _load_offline_events(
    config: ExperimentConfig,
) -> tuple[list[ReplayEvent], list[dict[str, Any]], str | None]:
    backend_cls, checks = _load_backend(config)
    exp2868 = _read_json(config.exp2868_path())
    exp2869 = _read_json(config.exp2869_path())
    exp2869_memory = _read_json(config.exp2869_memory_path())
    checks.extend(
        [
            {
                "check": "exp2869_artifact",
                "passed": bool(exp2869),
                "observed": str(config.exp2869_path()) if exp2869 else "missing",
            },
            {
                "check": "exp2869_replay_ready",
                "passed": bool(exp2869.get("fr11_self_learning_ready")),
                "observed": exp2869.get("honest_verdict", "missing"),
            },
            {
                "check": "exp2869_memory_artifact",
                "passed": bool(exp2869_memory),
                "observed": str(config.exp2869_memory_path()) if exp2869_memory else "missing",
            },
        ]
    )
    failed = {str(check["check"]) for check in checks if not check.get("passed")}
    if "exp2868_artifact" in failed:
        return [], checks, "blocked_missing_exp2868_artifact"
    if "exp2868_backend_module_path" in failed:
        return [], checks, "blocked_offline_backend_module_path"
    if backend_cls is None or "offline_backend_imported" in failed:
        return [], checks, "blocked_offline_backend_import"
    if "exp2869_artifact" in failed:
        return [], checks, "blocked_missing_exp2869_artifact"
    if "exp2869_replay_ready" in failed:
        return [], checks, "blocked_exp2869_replay_not_ready"
    if "exp2869_memory_artifact" in failed:
        return [], checks, "blocked_missing_exp2869_memory"

    events = _events_from_exp2868(exp2868)
    replay_events, replay_checks = _events_from_exp2869_replay(config, backend_cls)
    checks.extend(replay_checks)
    events.extend(replay_events)
    checks.append(
        {
            "check": "offline_events_loaded",
            "passed": bool(events),
            "observed": len(events),
        }
    )
    if not events:
        return [], checks, "blocked_no_offline_events"
    return events, checks, None


def _events_from_exp2868(exp2868: Mapping[str, Any]) -> list[ReplayEvent]:
    events: list[ReplayEvent] = []
    traces = exp2868.get("per_example_trace", [])
    if not isinstance(traces, list):
        return events
    for trace in traces:
        if isinstance(trace, Mapping):
            events.append(_event_from_trace("exp2868", trace))
    return events


def _events_from_exp2869_replay(
    config: ExperimentConfig,
    backend_cls: type[Any],
) -> tuple[list[ReplayEvent], list[dict[str, Any]]]:
    replay_config = ReplayExperimentConfig(
        repo_root=config.repo_root,
        results_dir=config.results_dir,
        random_seed=config.random_seed,
        max_loops=config.max_loops,
        replay_n_examples=config.replay_n_examples,
    )
    corpus, corpus_checks = load_clean_replay_corpus(replay_config)
    if not corpus:
        return [], corpus_checks
    backend = backend_cls(max_loops=config.max_loops)
    replay = backend.replay(build_backend_rows(corpus, config.max_loops))
    events = [
        _event_from_trace("exp2869", trace)
        for trace in replay.get("per_example_trace", [])
        if isinstance(trace, Mapping)
    ]
    return events, corpus_checks


def _event_from_trace(prefix: str, trace: Mapping[str, Any]) -> ReplayEvent:
    loops = trace.get("energy_after_each_loop", [])
    loop_values = [float(value) for value in loops] if isinstance(loops, list) else []
    energy_before = float(trace.get("energy_before", 0.0))
    energy_after = loop_values[-1] if loop_values else energy_before
    localized = trace.get("localized_violations", [])
    violations = tuple(str(item) for item in localized) if isinstance(localized, list) else ()
    return ReplayEvent(
        event_id=f"{prefix}::{trace.get('example_id', len(violations))}",
        source=str(trace.get("source") or "verifier_trace"),
        energy_before=_round_float(energy_before),
        energy_after=_round_float(energy_after),
        correctness_before=bool(trace.get("correctness_before")),
        correctness_after=bool(trace.get("correctness_after")),
        localized_violations=violations,
        early_exit_reason=str(trace.get("early_exit_reason") or ""),
    )


def _build_memory_state(
    config: ExperimentConfig,
    events: Sequence[ReplayEvent],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact": "fr11_recmem_recurrence_trigger_2881_memory",
        "schema": "carnot.fr11.recmem_recurrence_memory.v1",
        "run_date": config.run_date,
        "random_seed": config.random_seed,
        "recurrence_threshold": evaluation["recurrence_threshold"],
        "n_events_ingested": len(events),
        "consolidations": list(evaluation["consolidations"]),
        "event_motif_counts": dict(sorted(Counter(event.motif_key for event in events).items())),
        "threshold_sensitivity": list(evaluation["threshold_sensitivity"]),
        "live_llm_called": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:  # pragma: no cover - caller-supplied external memory paths only.
        return path.resolve().as_posix()


def _memory_manifest(paths: Sequence[Path], repo_root: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(paths, key=lambda item: _relative_path(item, repo_root)):
        rel_path = _relative_path(path, repo_root)
        if rel_path in seen:
            continue
        seen.add(rel_path)
        if path.is_file():
            manifest.append(
                {
                    "path": rel_path,
                    "present": True,
                    "n_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        else:
            manifest.append({"path": rel_path, "present": False, "n_bytes": 0, "sha256": None})
    return manifest


def _hash_memory_state(paths: Sequence[Path], repo_root: Path) -> tuple[str, list[dict[str, Any]]]:
    manifest = _memory_manifest(paths, repo_root)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), manifest


def _changed_memory_paths(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> list[str]:
    before_by_path = {str(item["path"]): dict(item) for item in before}
    after_by_path = {str(item["path"]): dict(item) for item in after}
    return [
        path
        for path in sorted(set(before_by_path) | set(after_by_path))
        if before_by_path.get(path) != after_by_path.get(path)
    ]


def _checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        key: artifact[key]
        for key in sorted(artifact)
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    honest_verdict: str,
    preconditions: Sequence[Mapping[str, Any]],
    duration_s: float,
    tests_run: Sequence[str],
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "artifact": "experiment_2881_fr11_recmem_recurrence_trigger_v1",
        "schema": "carnot.fr11.recmem_recurrence_trigger.v1",
        "honest_verdict": honest_verdict,
        "continuous_self_learning_task": True,
        "recmem_trigger_ready": False,
        "source_artifacts": _source_artifacts(config),
        "recurrence_threshold": _round_float(config.recurrence_threshold),
        "n_events_ingested": 0,
        "n_recurrence_clusters": 0,
        "n_consolidations_triggered": 0,
        "eager_consolidations_avoided": 0,
        "token_reduction_proxy_pct": 0.0,
        "memory_hash_before": "not_checked_precondition_failed",
        "memory_hash_after": "not_checked_precondition_failed",
        "contradiction_rate": 0.0,
        "duplicate_rate": 0.0,
        "forgetting_regression_count": 0,
        "live_llm_called": False,
        "tests_run": list(tests_run),
        "field_principles": FIELD_PRINCIPLES,
        "run_date": config.run_date,
        "duration_s": _round_float(duration_s),
        "preconditions_checked": list(preconditions),
        "methodology_note": "Blocked before RecMem consolidation; no metrics were inferred.",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact
