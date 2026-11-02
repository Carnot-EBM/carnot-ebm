"""Exp 2882 FR-11 RecMem replay scale-up comparison.

This runner compares eager offline verifier replay against RecMem-style
recurrence-triggered replay over the same bounded local labeled examples.  It
does not ask a model to generate or repair answers.  The only replay effect is
the deterministic energy summary already exposed by the Exp 2868 offline
backend, and the RecMem path only applies that effect to examples whose failure
motifs recur strongly enough to pass the Exp 2881 trigger.

Spec: REQ-LEARN-2882,
      SCENARIO-LEARN-2882,
      SCENARIO-LEARN-2882-BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.eval.fr11_continuous_self_learning_replay_v3 import (
    ExperimentConfig as ReplayExperimentConfig,
    ReplayExample,
    build_backend_rows,
    load_clean_replay_corpus,
)
from carnot.eval.fr11_recmem_recurrence_trigger_v1 import (
    MIN_SUPPORT,
    RECURRENCE_THRESHOLD,
    ReplayEvent,
    evaluate_recmem_events,
)
from carnot.eval.offline_recurrence_backend_adapter_v2 import OfflineRecurrenceReplayBackend


OUTPUT_FILENAME = "experiment_2882_fr11_recmem_replay_scaleup_v1.json"
EXP2869_FILENAME = "experiment_2869_fr11_continuous_self_learning_replay_v3.json"
EXP2881_FILENAME = "experiment_2881_fr11_recmem_recurrence_trigger_v1.json"
EXP2865_FILENAME = "experiment_2865_cross_corpus_matrix_v5.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260522"
RANDOM_SEED = 2882
TARGET_EXAMPLES = 50
MAX_LOOPS = 3

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "continuous_self_learning_task",
    "recmem_replay_scaleup_ready",
    "source_artifacts",
    "n_examples",
    "target_examples_met",
    "energy_delta_mean",
    "correctness_delta",
    "auroc_delta",
    "token_reduction_pct",
    "memory_drift_score",
    "forgetting_regression_count",
    "model_weights_mutated",
    "live_llm_called",
    "random_seed",
    "field_principles",
    "run_date",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal complete:/blocked_ verdict; no inferred live model success.",
    "continuous_self_learning_task": "True because this is an FR-11 replay-scale task.",
    "recmem_replay_scaleup_ready": (
        "True only when source artifacts are ready, the 50-example target is met, "
        "the RecMem trigger fires, token cost falls, and no forgetting/drift guard fails."
    ),
    "source_artifacts": "Prior Exp 2869/2881 artifacts plus the local clean-corpus manifest.",
    "n_examples": "Deterministic local labeled examples selected under the bounded budget.",
    "target_examples_met": "Whether at least the configured 50-example target was selected.",
    "energy_delta_mean": "Mean initial energy minus RecMem-triggered final energy.",
    "correctness_delta": "RecMem-triggered correctness rate minus eager correctness rate.",
    "auroc_delta": "RecMem-triggered error-detection AUROC minus eager AUROC where valid.",
    "token_reduction_pct": "RecMem triggered consolidation token proxy reduction versus eager writes.",
    "memory_drift_score": "Contradiction rate plus normalized forgetting, clipped to [0, 1].",
    "forgetting_regression_count": "Rows whose RecMem final energy or correctness regressed.",
    "model_weights_mutated": "False because no model object or weight tensor is loaded or updated.",
    "live_llm_called": "False because all rows are local verifier-label replay metadata.",
    "random_seed": "Controls deterministic corpus selection for both replay modes.",
    "field_principles": "Per-field accounting rules for the artifact.",
    "run_date": "Pinned conductor run date.",
    "duration_s": "Real wall-clock duration; no sleep padding.",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the Exp 2882 offline scale-up comparison."""

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    run_date: str = RUN_DATE
    random_seed: int = RANDOM_SEED
    target_examples: int = TARGET_EXAMPLES
    max_loops: int = MAX_LOOPS
    recurrence_threshold: float = RECURRENCE_THRESHOLD
    min_support: int = MIN_SUPPORT
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        if self.target_examples < 1:
            raise ValueError("target_examples must be >= 1")
        if self.max_loops < 1:
            raise ValueError("max_loops must be >= 1")
        if not 0.0 <= self.recurrence_threshold <= 1.0:
            raise ValueError("recurrence_threshold must be in [0.0, 1.0]")
        if self.min_support < 2:
            raise ValueError("min_support must be >= 2")

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def output_path(self) -> Path:
        return self.output_dir() / OUTPUT_FILENAME

    def exp2869_path(self) -> Path:
        return self.output_dir() / EXP2869_FILENAME

    def exp2881_path(self) -> Path:
        return self.output_dir() / EXP2881_FILENAME

    def exp2865_path(self) -> Path:
        return self.output_dir() / EXP2865_FILENAME

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, Any]:
    """Run the bounded eager-vs-RecMem replay comparison.

    Spec: REQ-LEARN-2882-1 through REQ-LEARN-2882-5.
    """

    active_config = config or ExperimentConfig()
    started_at = active_config.start_time()
    source_checks, blocker = _source_preconditions(active_config)
    if blocker is not None:
        artifact = _blocked_artifact(active_config, blocker, source_checks, started_at)
        if write:
            _write_json(active_config.output_path(), artifact)
        return artifact

    replay_config = ReplayExperimentConfig(
        repo_root=active_config.repo_root,
        results_dir=active_config.results_dir,
        random_seed=active_config.random_seed,
        max_loops=active_config.max_loops,
        replay_n_examples=active_config.target_examples,
    )
    examples, corpus_checks = load_clean_replay_corpus(replay_config)
    comparison = compare_replay_modes(
        examples,
        max_loops=active_config.max_loops,
        recurrence_threshold=active_config.recurrence_threshold,
        min_support=active_config.min_support,
    )
    n_examples = int(comparison["n_examples"])
    target_examples_met = n_examples >= active_config.target_examples
    forgetting_regression_count = int(comparison["forgetting_regression_count"])
    memory_drift_score = _memory_drift_score(
        float(comparison["contradiction_rate"]),
        forgetting_regression_count,
        n_examples,
    )
    ready = (
        target_examples_met
        and bool(comparison["recmem_trigger_ready"])
        and float(comparison["token_reduction_pct"]) > 0.0
        and memory_drift_score == 0.0
        and forgetting_regression_count == 0
        and float(comparison["correctness_delta"]) >= 0.0
        and (not bool(comparison["auroc_valid"]) or float(comparison["auroc_delta"]) >= 0.0)
    )
    if ready:
        honest_verdict = "complete: RecMem-triggered replay matched eager replay with lower token cost and no forgetting"
    elif not target_examples_met:
        honest_verdict = "blocked_target_examples_not_met"
    else:
        honest_verdict = "blocked_recmem_scaleup_guard_failed"

    artifact: dict[str, Any] = {
        "artifact": "experiment_2882_fr11_recmem_replay_scaleup_v1",
        "schema": "carnot.fr11.recmem_replay_scaleup.v1",
        "honest_verdict": honest_verdict,
        "continuous_self_learning_task": True,
        "recmem_replay_scaleup_ready": ready,
        "source_artifacts": _source_artifacts(active_config),
        "n_examples": n_examples,
        "target_examples_met": target_examples_met,
        "energy_delta_mean": comparison["energy_delta_mean"],
        "correctness_delta": comparison["correctness_delta"],
        "auroc_delta": comparison["auroc_delta"],
        "token_reduction_pct": comparison["token_reduction_pct"],
        "memory_drift_score": memory_drift_score,
        "forgetting_regression_count": forgetting_regression_count,
        "model_weights_mutated": False,
        "live_llm_called": False,
        "random_seed": active_config.random_seed,
        "field_principles": FIELD_PRINCIPLES,
        "run_date": active_config.run_date,
        "duration_s": _round_float(active_config.clock() - started_at),
        "preconditions_checked": [*source_checks, *corpus_checks],
        "target_examples": active_config.target_examples,
        "bounded_wall_clock_budget_note": (
            "Offline CPU replay is bounded by target_examples; no live model or LLM call is made."
        ),
        **comparison,
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    if write:
        _write_json(active_config.output_path(), artifact)
    return artifact


def compare_replay_modes(
    examples: Sequence[ReplayExample],
    *,
    max_loops: int,
    recurrence_threshold: float,
    min_support: int,
) -> dict[str, Any]:
    """Compare eager replay with recurrence-triggered replay on the same rows."""

    backend = OfflineRecurrenceReplayBackend(max_loops=max_loops)
    eager_replay = backend.replay(build_backend_rows(examples, max_loops))
    traces = list(eager_replay["per_example_trace"])
    events = [_event_from_trace(trace) for trace in traces]
    recmem_eval = evaluate_recmem_events(
        events,
        recurrence_threshold=recurrence_threshold,
        min_support=min_support,
    )
    triggered_ids = {
        str(event_id)
        for consolidation in recmem_eval["consolidations"]
        for event_id in consolidation["event_ids"]
    }
    rows = [_comparison_row(trace, triggered_ids) for trace in traces]
    labels = [0 if row["initial_correct"] else 1 for row in rows]
    eager_scores = [row["eager_final_energy"] for row in rows]
    recmem_scores = [row["recmem_final_energy"] for row in rows]
    eager_auroc = _roc_auc(labels, eager_scores)
    recmem_auroc = _roc_auc(labels, recmem_scores)
    auroc_valid = eager_auroc is not None and recmem_auroc is not None
    eager_energy_delta = _mean([row["initial_energy"] - row["eager_final_energy"] for row in rows])
    recmem_energy_delta = _mean(
        [row["initial_energy"] - row["recmem_final_energy"] for row in rows]
    )
    eager_correctness = _mean([1.0 if row["eager_correct"] else 0.0 for row in rows])
    recmem_correctness = _mean([1.0 if row["recmem_correct"] else 0.0 for row in rows])
    forgetting_regression_count = sum(1 for row in rows if _row_forgot(row))
    source_counts = Counter(str(trace.get("source", "unknown")) for trace in traces)
    return {
        "n_examples": len(rows),
        "energy_delta_mean": _round_float(recmem_energy_delta),
        "eager_energy_delta_mean": _round_float(eager_energy_delta),
        "energy_delta_vs_eager": _round_float(recmem_energy_delta - eager_energy_delta),
        "correctness_delta": _round_float(recmem_correctness - eager_correctness),
        "recmem_correctness_rate": _round_float(recmem_correctness),
        "eager_correctness_rate": _round_float(eager_correctness),
        "auroc_delta": _round_float((recmem_auroc - eager_auroc) if auroc_valid else 0.0),
        "auroc_valid": auroc_valid,
        "recmem_auroc": _round_float(recmem_auroc) if recmem_auroc is not None else None,
        "eager_auroc": _round_float(eager_auroc) if eager_auroc is not None else None,
        "token_reduction_pct": recmem_eval["token_reduction_proxy_pct"],
        "token_proxy_before": recmem_eval["token_proxy_before"],
        "token_proxy_after": recmem_eval["token_proxy_after"],
        "contradiction_rate": recmem_eval["contradiction_rate"],
        "duplicate_rate": recmem_eval["duplicate_rate"],
        "forgetting_regression_count": max(
            forgetting_regression_count,
            int(recmem_eval["forgetting_regression_count"]),
        ),
        "recmem_trigger_ready": recmem_eval["recmem_trigger_ready"],
        "n_recurrence_clusters": recmem_eval["n_recurrence_clusters"],
        "n_consolidations_triggered": recmem_eval["n_consolidations_triggered"],
        "recmem_triggered_example_count": len(triggered_ids),
        "source_counts": dict(sorted(source_counts.items())),
        "selected_example_ids": [str(trace["example_id"]) for trace in traces],
        "eager_example_ids": [str(trace["example_id"]) for trace in traces],
    }


def _source_preconditions(config: ExperimentConfig) -> tuple[list[dict[str, Any]], str | None]:
    exp2869 = _read_json(config.exp2869_path())
    exp2881 = _read_json(config.exp2881_path())
    checks = [
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
            "check": "exp2881_artifact",
            "passed": bool(exp2881),
            "observed": str(config.exp2881_path()) if exp2881 else "missing",
        },
        {
            "check": "exp2881_recmem_ready",
            "passed": bool(exp2881.get("recmem_trigger_ready")),
            "observed": exp2881.get("honest_verdict", "missing"),
        },
    ]
    failed = {str(check["check"]) for check in checks if not check["passed"]}
    if "exp2869_artifact" in failed:
        return checks, "blocked_missing_exp2869_artifact"
    if "exp2869_replay_ready" in failed:
        return checks, "blocked_exp2869_not_ready"
    if "exp2881_artifact" in failed:
        return checks, "blocked_missing_exp2881_artifact"
    if "exp2881_recmem_ready" in failed:
        return checks, "blocked_exp2881_not_ready"
    return checks, None


def _blocked_artifact(
    config: ExperimentConfig,
    honest_verdict: str,
    preconditions: Sequence[Mapping[str, Any]],
    started_at: float,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "artifact": "experiment_2882_fr11_recmem_replay_scaleup_v1",
        "schema": "carnot.fr11.recmem_replay_scaleup.v1",
        "honest_verdict": honest_verdict,
        "continuous_self_learning_task": True,
        "recmem_replay_scaleup_ready": False,
        "source_artifacts": _source_artifacts(config),
        "n_examples": 0,
        "target_examples_met": False,
        "energy_delta_mean": 0.0,
        "correctness_delta": 0.0,
        "auroc_delta": 0.0,
        "token_reduction_pct": 0.0,
        "memory_drift_score": 0.0,
        "forgetting_regression_count": 0,
        "model_weights_mutated": False,
        "live_llm_called": False,
        "random_seed": config.random_seed,
        "field_principles": FIELD_PRINCIPLES,
        "run_date": config.run_date,
        "duration_s": _round_float(config.clock() - started_at),
        "preconditions_checked": list(preconditions),
        "target_examples": config.target_examples,
        "auroc_valid": False,
        "methodology_note": "Blocked before replay comparison; no metrics were inferred.",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _event_from_trace(trace: Mapping[str, Any]) -> ReplayEvent:
    loops = trace.get("energy_after_each_loop", [])
    loop_values = [float(value) for value in loops] if isinstance(loops, list) else []
    energy_before = float(trace.get("energy_before", 0.0))
    return ReplayEvent(
        event_id=_event_id(trace),
        source=str(trace.get("source") or "verifier_trace"),
        energy_before=_round_float(energy_before),
        energy_after=_round_float(loop_values[-1] if loop_values else energy_before),
        correctness_before=bool(trace.get("correctness_before")),
        correctness_after=bool(trace.get("correctness_after")),
        localized_violations=tuple(str(item) for item in trace.get("localized_violations", [])),
        early_exit_reason=str(trace.get("early_exit_reason") or ""),
    )


def _comparison_row(trace: Mapping[str, Any], triggered_ids: set[str]) -> dict[str, Any]:
    event_id = _event_id(trace)
    initial_energy = float(trace.get("energy_before", 0.0))
    eager_final = _final_energy(trace)
    triggered = event_id in triggered_ids
    initial_correct = bool(trace.get("correctness_before"))
    eager_correct = bool(trace.get("correctness_after", initial_correct))
    return {
        "event_id": event_id,
        "initial_energy": initial_energy,
        "eager_final_energy": eager_final,
        "recmem_final_energy": eager_final if triggered else initial_energy,
        "initial_correct": initial_correct,
        "eager_correct": eager_correct,
        "recmem_correct": eager_correct if triggered else initial_correct,
    }


def _event_id(trace: Mapping[str, Any]) -> str:
    return f"exp2882::{trace.get('source', 'unknown')}::{trace.get('example_id', 'unknown')}"


def _final_energy(trace: Mapping[str, Any]) -> float:
    loops = trace.get("energy_after_each_loop", [])
    if isinstance(loops, list) and loops:
        return float(loops[-1])
    return float(trace.get("energy_before", 0.0))


def _row_forgot(row: Mapping[str, Any]) -> bool:
    return bool(
        float(row["recmem_final_energy"]) > float(row["initial_energy"])
        or (bool(row["initial_correct"]) and not bool(row["recmem_correct"]))
    )


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [float(score) for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [float(score) for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _memory_drift_score(
    contradiction_rate: float,
    forgetting_regression_count: int,
    n_examples: int,
) -> float:
    if n_examples <= 0:
        return 0.0
    normalized_forgetting = forgetting_regression_count / n_examples
    return _round_float(min(1.0, max(0.0, contradiction_rate + normalized_forgetting)))


def _source_artifacts(config: ExperimentConfig) -> list[str]:
    return [
        _relative_path(config.exp2869_path(), config.repo_root),
        _relative_path(config.exp2881_path(), config.repo_root),
        _relative_path(config.exp2865_path(), config.repo_root),
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:  # pragma: no cover - only used for caller-supplied external paths.
        return path.resolve().as_posix()


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round_float(value: float) -> float:
    return round(float(value), 12)


def _checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        key: artifact[key]
        for key in sorted(artifact)
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
