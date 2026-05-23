"""Exp 2947 FR-11 continuation replay curriculum scheduler.

This module does not run a generator, verifier, or hardware job. The point is
to replace flat-uniform replay selection with a deterministic curriculum that
uses the already-written FR-11 and continuation artifacts as evidence. That
keeps the pilot honest: it can choose which replay buckets deserve more budget,
but it cannot invent new model quality from a scheduler-only aggregation.

Spec: REQ-LEARN-2947,
      SCENARIO-LEARN-2947,
      SCENARIO-LEARN-2947-BLOCKED.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2947_fr11_continuation_replay_curriculum_v1.json"
EXP2918_FILENAME = "experiment_2918_fr11_verifiable_process_rewards_self_learning_v1.json"
EXP2933_FILENAME = "experiment_2933_kan_cl_per_knot_self_learning_v1.json"
EXP2942_FILENAME = "experiment_2942_kv260_continuation_n_scaling_v1.json"
RUN_DATE = "20260523"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
CURRICULUM_SCHEDULE = "fr11_continuation_curriculum_v1"
DEFAULT_REPLAY_BUDGET = 64

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "curriculum_schedule_used",
    "replay_count_distribution",
    "cited_upstream_artifacts",
    "duration_s",
}

EXP2918_FIELDS = [
    "honest_verdict",
    "online_self_learning_ready",
    "replay_scheduler_updated",
    "delta_energy_proxy",
    "pdi_proxy",
    "forgetting_rate",
    "replay_corpus_summary",
]
EXP2933_FIELDS = [
    "honest_verdict",
    "kan_cl_self_learning_ready",
    "utility_delta_vs_replay_only",
    "energy_proxy_delta",
    "forgetting_rate",
    "updated_knot_or_rbf_count",
    "dataset_manifest",
]
EXP2942_FIELDS = [
    "honest_verdict",
    "inference_substrate",
    "measured_n_values",
    "unsupported_n_values",
    "per_n_results",
    "bitstream_supports_variable_n",
]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and clock for the artifact-only curriculum pilot."""

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    replay_budget: int = DEFAULT_REPLAY_BUDGET
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def output_path(self) -> Path:
        return self.output_dir() / OUTPUT_FILENAME

    def artifact_path(self, filename: str) -> Path:
        return self.output_dir() / filename

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


@dataclass(frozen=True)
class UpstreamArtifact:
    """A loaded upstream artifact plus the citation metadata needed downstream."""

    experiment_id: str
    path: Path
    fields_imported: Sequence[str]
    payload: Mapping[str, Any]
    sha256: str
    missing: bool = False
    malformed: bool = False


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, Any]:
    """Build the Exp 2947 curriculum artifact from upstream result files."""

    active_config = config or ExperimentConfig()
    started_at = active_config.start_time()
    upstreams = _load_upstreams(active_config)
    gate_results = _gate_results(upstreams)
    failed_gates = [gate["name"] for gate in gate_results if not gate["passed"]]
    duration_s = _round_float(active_config.clock() - started_at)
    if failed_gates:
        artifact = _blocked_artifact(active_config, upstreams, gate_results, failed_gates, duration_s)
    else:
        scores = curriculum_signal_scores(upstreams)
        distribution = allocate_replay_counts(scores, replay_budget=active_config.replay_budget)
        artifact = {
            "artifact": "experiment_2947_fr11_continuation_replay_curriculum_v1",
            "schema": "carnot.fr11.continuation_replay_curriculum.v1",
            "honest_verdict": "complete: nonuniform_continuation_replay_curriculum_piloted",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "curriculum_schedule_used": CURRICULUM_SCHEDULE,
            "replay_count_distribution": distribution,
            "cited_upstream_artifacts": _citations(active_config, upstreams),
            "duration_s": duration_s,
            "run_date": RUN_DATE,
            "replay_budget": active_config.replay_budget,
            "flat_uniform_sampling_used": False,
            "model_weights_mutated": False,
            "live_model_invoked": False,
            "failed_gates": [],
            "gate_results": gate_results,
            "curriculum_signal_scores": scores,
        }
    if write:
        _write_json(active_config.output_path(), artifact)
    return artifact


def allocate_replay_counts(
    scores: Mapping[str, float],
    *,
    replay_budget: int,
) -> dict[str, int]:
    """Turn positive curriculum scores into deterministic integer replay counts."""

    positive_scores = {
        bucket: float(score) for bucket, score in sorted(scores.items()) if float(score) > 0.0
    }
    if not positive_scores:
        raise ValueError("at least one positive curriculum score is required")
    if replay_budget < len(positive_scores):
        raise ValueError("replay_budget must cover every positive curriculum bucket")

    counts = {bucket: 1 for bucket in positive_scores}
    remaining = replay_budget - len(positive_scores)
    total_score = sum(positive_scores.values())
    fractional: list[tuple[float, str]] = []
    for bucket, score in positive_scores.items():
        exact_extra = remaining * score / total_score
        whole_extra = int(exact_extra)
        counts[bucket] += whole_extra
        fractional.append((exact_extra - whole_extra, bucket))

    leftover = replay_budget - sum(counts.values())
    for _, bucket in sorted(fractional, key=lambda item: (-item[0], item[1]))[:leftover]:
        counts[bucket] += 1
    return counts


def curriculum_signal_scores(upstreams: Mapping[str, UpstreamArtifact]) -> dict[str, float]:
    """Compute schedule scores from upstream evidence without rereading artifacts."""

    exp2918 = upstreams["exp2918"].payload
    exp2933 = upstreams["exp2933"].payload
    exp2942 = upstreams["exp2942"].payload
    structural = (
        _positive_float(exp2933.get("utility_delta_vs_replay_only"))
        + 0.5 * _positive_float(exp2933.get("energy_proxy_delta"))
        + 0.25 * min(1.0, _positive_float(exp2933.get("updated_knot_or_rbf_count")) / 12.0)
    )
    process_reward = _positive_float(exp2918.get("pdi_proxy")) + _positive_float(
        exp2918.get("delta_energy_proxy")
    )
    continuation_boundary = (
        0.20
        + 0.05 * len(exp2942.get("measured_n_values", []))
        + 0.03 * len(exp2942.get("unsupported_n_values", []))
    )
    max_forgetting = max(
        _positive_float(exp2918.get("forgetting_rate")),
        _positive_float(exp2933.get("forgetting_rate")),
    )
    retention_guard = 0.25 + 0.25 * max(0.0, 1.0 - max_forgetting)
    return {
        "structural_memory_bootstrap": _round_float(structural),
        "process_reward_replay": _round_float(process_reward),
        "continuation_boundary_replay": _round_float(continuation_boundary),
        "retention_guard_replay": _round_float(retention_guard),
    }


def _blocked_artifact(
    config: ExperimentConfig,
    upstreams: Mapping[str, UpstreamArtifact],
    gate_results: Sequence[Mapping[str, Any]],
    failed_gates: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "artifact": "experiment_2947_fr11_continuation_replay_curriculum_v1",
        "schema": "carnot.fr11.continuation_replay_curriculum.v1",
        "honest_verdict": _blocked_verdict(failed_gates),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "curriculum_schedule_used": "blocked",
        "replay_count_distribution": {},
        "cited_upstream_artifacts": _citations(config, upstreams),
        "duration_s": duration_s,
        "run_date": RUN_DATE,
        "replay_budget": config.replay_budget,
        "flat_uniform_sampling_used": False,
        "model_weights_mutated": False,
        "live_model_invoked": False,
        "failed_gates": list(failed_gates),
        "gate_results": list(gate_results),
        "curriculum_signal_scores": {},
    }


def _load_upstreams(config: ExperimentConfig) -> dict[str, UpstreamArtifact]:
    return {
        "exp2918": _read_upstream("exp2918", config.artifact_path(EXP2918_FILENAME), EXP2918_FIELDS),
        "exp2933": _read_upstream("exp2933", config.artifact_path(EXP2933_FILENAME), EXP2933_FIELDS),
        "exp2942": _read_upstream("exp2942", config.artifact_path(EXP2942_FILENAME), EXP2942_FIELDS),
    }


def _read_upstream(
    experiment_id: str,
    path: Path,
    fields_imported: Sequence[str],
) -> UpstreamArtifact:
    if not path.exists():
        return UpstreamArtifact(experiment_id, path, fields_imported, {}, "", missing=True)
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return UpstreamArtifact(experiment_id, path, fields_imported, {}, sha256, malformed=True)
    if not isinstance(payload, Mapping):
        return UpstreamArtifact(experiment_id, path, fields_imported, {}, sha256, malformed=True)
    return UpstreamArtifact(experiment_id, path, fields_imported, payload, sha256)


def _gate_results(upstreams: Mapping[str, UpstreamArtifact]) -> list[dict[str, Any]]:
    validators = {
        "exp2918": _exp2918_ready,
        "exp2933": _exp2933_ready,
        "exp2942": _exp2942_ready,
    }
    results: list[dict[str, Any]] = []
    for experiment_id, upstream in upstreams.items():
        if upstream.missing:
            results.append({"name": f"missing_{experiment_id}_artifact", "passed": False})
            continue
        if upstream.malformed:
            results.append({"name": f"{experiment_id}_artifact_malformed", "passed": False})
            continue
        ready = validators[experiment_id](upstream.payload)
        gate_name = f"{experiment_id}_ready" if ready else f"{experiment_id}_not_ready"
        results.append({"name": gate_name, "passed": ready})
    return results


def _exp2918_ready(payload: Mapping[str, Any]) -> bool:
    return payload.get("online_self_learning_ready") is True and payload.get(
        "replay_scheduler_updated"
    ) is True


def _exp2933_ready(payload: Mapping[str, Any]) -> bool:
    return payload.get("kan_cl_self_learning_ready") is True


def _exp2942_ready(payload: Mapping[str, Any]) -> bool:
    return (
        str(payload.get("honest_verdict", "")).startswith("complete:")
        and payload.get("inference_substrate") == "hardware_smoke"
        and bool(payload.get("per_n_results"))
    )


def _blocked_verdict(failed_gates: Sequence[str]) -> str:
    first = failed_gates[0]
    return {
        "missing_exp2918_artifact": "blocked_missing_exp2918_artifact",
        "exp2918_artifact_malformed": "blocked_malformed_exp2918_artifact",
        "exp2918_not_ready": "blocked_exp2918_not_ready",
        "missing_exp2933_artifact": "blocked_missing_exp2933_artifact",
        "exp2933_artifact_malformed": "blocked_malformed_exp2933_artifact",
        "exp2933_not_ready": "blocked_exp2933_not_ready",
        "missing_exp2942_artifact": "blocked_missing_exp2942_artifact",
        "exp2942_artifact_malformed": "blocked_malformed_exp2942_artifact",
        "exp2942_not_ready": "blocked_exp2942_not_ready",
    }.get(first, f"blocked_{first}")


def _citations(
    config: ExperimentConfig,
    upstreams: Mapping[str, UpstreamArtifact],
) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for upstream in upstreams.values():
        if upstream.missing or upstream.malformed:
            continue
        citations.append(
            {
                "experiment_id": upstream.experiment_id,
                "path": upstream.path.resolve().relative_to(config.repo_root.resolve()).as_posix(),
                "fields_imported": list(upstream.fields_imported),
                "sha256": upstream.sha256,
            }
        )
    return citations


def _positive_float(value: object) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _round_float(value: float) -> float:
    return round(float(value), 12)
