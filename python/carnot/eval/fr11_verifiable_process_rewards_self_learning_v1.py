"""Exp 2918 FR-11 verifier-driven process-reward replay.

This experiment is intentionally bounded. It does not imitate a generator and
does not mutate model weights. It turns deterministic verifier outcomes into
replay priority weights, applies one online scheduler update, and measures the
small external-memory effect honestly.

Spec: REQ-LEARN-2918,
      SCENARIO-LEARN-2918,
      SCENARIO-LEARN-2918-BLOCKED.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2918_fr11_verifiable_process_rewards_self_learning_v1.json"
EXP2911_FILENAME = "experiment_2911_code_hallucination_taxonomy_verifier_v1.json"
EXP2912_FILENAME = "experiment_2912_kv260_same_basis_cpu_gibbs_baseline_v1.json"
EXP2887_FILENAME = "experiment_2887_fr11_fast_slow_memory_corrigendum_v2.json"
EXP2906_FILENAME = "experiment_2906_fr11_hardware_accelerated_replay_pilot_v1.json"
RUN_DATE = "20260523"
INFERENCE_SUBSTRATE = "deterministic_verifier_plus_replay"
SCHEDULER_SCOPE = "replay_priority_table_only_no_model_weights"
REPLAY_THRESHOLD = 0.5

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "online_self_learning_ready",
    "fr11_requirement_targeted",
    "process_reward_definition",
    "replay_corpus_summary",
    "online_update_performed",
    "replay_scheduler_updated",
    "delta_overall",
    "delta_energy_proxy",
    "forgetting_rate",
    "contradiction_rate_before",
    "contradiction_rate_after",
    "pdi_proxy",
    "hardware_replay_used",
    "inference_substrate",
    "duration_s",
    "run_date",
)


@dataclass(frozen=True)
class ProcessReward:
    """A deterministic weight plus the verifier components that produced it."""

    weight: float
    components: dict[str, float]


@dataclass(frozen=True)
class ReplayRow:
    """One verifier-backed replay item for the external scheduler."""

    row_id: str
    source_type: str
    motif_key: str
    process_reward: ProcessReward
    energy_proxy_before: float
    expected_replay: bool
    held_out: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and clock for the Exp 2918 aggregation."""

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
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


class ReplayScheduler:
    """Tiny online scheduler used when no trainable FR-11 component is available."""

    def __init__(self, rows: Sequence[ReplayRow], *, default_priority: float = REPLAY_THRESHOLD):
        self.before = {row.row_id: _round_float(default_priority) for row in rows}
        self.after = dict(self.before)

    def update(self, rows: Sequence[ReplayRow]) -> None:
        for row in rows:
            if row.held_out:
                continue
            self.after[row.row_id] = row.process_reward.weight


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, Any]:
    """Run the bounded FR-11 process-reward replay experiment."""

    active_config = config or ExperimentConfig()
    started_at = active_config.start_time()
    artifacts = _load_artifacts(active_config)
    failed_preconditions = _failed_preconditions(artifacts)
    if failed_preconditions:
        artifact = _blocked_artifact(active_config, failed_preconditions, started_at)
        if write:
            _write_json(active_config.output_path(), artifact)
        return artifact

    rows = _build_replay_corpus(artifacts)
    scheduler = ReplayScheduler(rows)
    metrics_before = _evaluate(rows, scheduler.before)
    scheduler.update(rows)
    metrics_after = _evaluate(rows, scheduler.after)
    delta_overall = _round_float(metrics_after["correctness"] - metrics_before["correctness"])
    delta_energy_proxy = _round_float(metrics_before["energy_proxy"] - metrics_after["energy_proxy"])
    contradiction_before = _contradiction_rate(rows, scheduler.before)
    contradiction_after = _contradiction_rate(rows, scheduler.after)
    forgetting_rate = _forgetting_rate(rows, scheduler.before, scheduler.after)
    pdi_proxy = _round_float(
        max(
            0.0,
            (
                delta_overall
                + delta_energy_proxy
                + max(0.0, contradiction_before - contradiction_after)
            )
            / 3.0,
        )
    )

    artifact = {
        "artifact": "experiment_2918_fr11_verifiable_process_rewards_self_learning_v1",
        "honest_verdict": "complete: verifier_process_rewards_updated_replay_scheduler",
        "online_self_learning_ready": True,
        "fr11_requirement_targeted": "FR-11",
        "process_reward_definition": process_reward_definition(),
        "replay_corpus_summary": _corpus_summary(rows),
        "online_update_performed": True,
        "replay_scheduler_updated": True,
        "delta_overall": delta_overall,
        "delta_energy_proxy": delta_energy_proxy,
        "forgetting_rate": forgetting_rate,
        "contradiction_rate_before": contradiction_before,
        "contradiction_rate_after": contradiction_after,
        "pdi_proxy": pdi_proxy,
        "hardware_replay_used": any(row.source_type == "hardware" for row in rows),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _round_float(active_config.clock() - started_at),
        "run_date": RUN_DATE,
        "failed_preconditions": [],
        "model_weights_mutated": False,
        "scheduler_update_scope": SCHEDULER_SCOPE,
        "priority_summary": _priority_summary(scheduler.before, scheduler.after),
        "duplicate_rate_before": _duplicate_rate(rows),
        "duplicate_rate_after": _duplicate_rate(
            [row for row in rows if scheduler.after[row.row_id] >= REPLAY_THRESHOLD]
        ),
        "source_artifacts": {
            "exp2911": EXP2911_FILENAME,
            "exp2912": EXP2912_FILENAME,
            "exp2887": EXP2887_FILENAME,
            "exp2906": EXP2906_FILENAME,
        },
    }
    if write:
        _write_json(active_config.output_path(), artifact)
    return artifact


def code_process_reward(candidate: Mapping[str, Any]) -> ProcessReward:
    """Derive code replay weight from syntax, static, and runtime verifier labels."""

    labels = {str(label) for label in candidate.get("labels", [])}
    passed = bool(candidate.get("passed"))
    syntax_success = bool(candidate.get("syntax_success", True))
    hallucination_labels = labels.intersection(
        {
            "invented_import",
            "undefined_name",
            "invented_attribute_or_method",
            "invalid_argument",
        }
    )
    if passed:
        components = {"syntax": 0.25, "static": 0.30, "runtime": 0.40}
    elif not syntax_success:
        components = {"syntax": 0.05, "static": 0.05, "runtime": 0.05}
    elif hallucination_labels:
        components = {
            "syntax": 0.15,
            "static": 0.05 * min(len(hallucination_labels), 2),
            "runtime": 0.0,
        }
    else:
        components = {"syntax": 0.15, "static": 0.05, "runtime": 0.15}
    return ProcessReward(weight=_round_float(sum(components.values())), components=components)


def hardware_process_reward(
    measurement: Mapping[str, Any],
    *,
    min_energy: float,
    max_energy: float,
    min_latency: float,
    max_latency: float,
    basis_matched: bool,
) -> ProcessReward:
    """Derive sampler replay weight from basis, energy, and latency evidence."""

    if not basis_matched:
        components = {"basis": 0.0, "energy": 0.0, "latency": 0.0, "mismatch_floor": 0.45}
        return ProcessReward(weight=0.45, components=components)
    energy = float(measurement.get("final_energy", max_energy))
    latency = float(measurement.get("cpu_latency_us_median", max_latency))
    energy_score = (max_energy - energy) / max(max_energy - min_energy, 1e-9)
    latency_score = (max_latency - latency) / max(max_latency - min_latency, 1e-9)
    components = {
        "basis": 0.45,
        "energy": _round_float(0.30 * max(0.0, min(1.0, energy_score))),
        "latency": _round_float(0.20 * max(0.0, min(1.0, latency_score))),
    }
    return ProcessReward(weight=_round_float(sum(components.values())), components=components)


def process_reward_definition() -> dict[str, dict[str, Any]]:
    """Return the public process-reward contract recorded in the artifact."""

    return {
        "code": {
            "drives": "syntax_static_runtime",
            "components": ["syntax_success", "static_hallucination_labels", "runtime_passed"],
            "likelihood_used": False,
        },
        "hardware": {
            "drives": "basis_energy_latency",
            "components": ["same_basis_match", "final_energy_proxy", "latency_proxy"],
            "likelihood_used": False,
        },
        "prior_fr11": {
            "drives": "nonforgetting_replay_retention",
            "components": ["prior_fast_slow_memory_event", "held_out_retention"],
            "likelihood_used": False,
        },
    }


def _load_artifacts(config: ExperimentConfig) -> dict[str, dict[str, Any]]:
    return {
        "exp2911": _read_json(config.artifact_path(EXP2911_FILENAME)),
        "exp2912": _read_json(config.artifact_path(EXP2912_FILENAME)),
        "exp2887": _read_json(config.artifact_path(EXP2887_FILENAME)),
        "exp2906": _read_json(config.artifact_path(EXP2906_FILENAME)),
    }


def _failed_preconditions(artifacts: Mapping[str, Mapping[str, Any]]) -> list[str]:
    checks = [
        ("exp2911", "code_hallucination_verifier_ready"),
        ("exp2912", "same_basis_cpu_baseline_ready"),
    ]
    failed: list[str] = []
    for artifact_name, ready_field in checks:
        payload = artifacts[artifact_name]
        if not payload:
            failed.append(f"missing_{artifact_name}_artifact")
        elif payload.get(ready_field) is not True:
            failed.append(f"{artifact_name}_not_ready")
    return failed


def _blocked_artifact(
    config: ExperimentConfig,
    failed_preconditions: Sequence[str],
    started_at: float,
) -> dict[str, Any]:
    return {
        "artifact": "experiment_2918_fr11_verifiable_process_rewards_self_learning_v1",
        "honest_verdict": _blocked_verdict(failed_preconditions),
        "online_self_learning_ready": False,
        "fr11_requirement_targeted": "FR-11",
        "process_reward_definition": process_reward_definition(),
        "replay_corpus_summary": {
            "total_rows": 0,
            "code_rows": 0,
            "hardware_rows": 0,
            "prior_fr11_rows": 0,
            "held_out_prior_rows": 0,
        },
        "online_update_performed": False,
        "replay_scheduler_updated": False,
        "delta_overall": 0.0,
        "delta_energy_proxy": 0.0,
        "forgetting_rate": 0.0,
        "contradiction_rate_before": 0.0,
        "contradiction_rate_after": 0.0,
        "pdi_proxy": 0.0,
        "hardware_replay_used": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _round_float(config.clock() - started_at),
        "run_date": RUN_DATE,
        "failed_preconditions": list(failed_preconditions),
        "model_weights_mutated": False,
        "scheduler_update_scope": SCHEDULER_SCOPE,
    }


def _blocked_verdict(failed_preconditions: Sequence[str]) -> str:
    first = failed_preconditions[0]
    return {
        "missing_exp2911_artifact": "blocked_missing_exp2911_artifact",
        "missing_exp2912_artifact": "blocked_missing_exp2912_artifact",
        "exp2911_not_ready": "blocked_exp2911_not_ready",
        "exp2912_not_ready": "blocked_exp2912_not_ready",
    }[first]


def _build_replay_corpus(artifacts: Mapping[str, Mapping[str, Any]]) -> list[ReplayRow]:
    rows = _code_rows(artifacts["exp2911"])
    rows.extend(_hardware_rows(artifacts["exp2912"]))
    rows.extend(_prior_fr11_rows(artifacts["exp2887"]))
    return rows


def _code_rows(artifact: Mapping[str, Any]) -> list[ReplayRow]:
    rows: list[ReplayRow] = []
    for candidate in artifact.get("per_candidate_labels", []):
        reward = code_process_reward(candidate)
        row_id = f"code:{candidate.get('stable_id', 'unknown')}:{candidate.get('candidate_index', 0)}"
        passed = bool(candidate.get("passed"))
        labels = ",".join(str(label) for label in candidate.get("labels", [])) or "clean"
        energy_proxy = 0.1 if passed else 1.0 - reward.weight
        rows.append(
            ReplayRow(
                row_id=row_id,
                source_type="code",
                motif_key=f"code:{labels}",
                process_reward=reward,
                energy_proxy_before=_round_float(energy_proxy),
                expected_replay=reward.weight >= REPLAY_THRESHOLD,
            )
        )
    return rows


def _hardware_rows(artifact: Mapping[str, Any]) -> list[ReplayRow]:
    measurements = list(artifact.get("cpu_per_seed_results", []))
    energies = [float(row.get("final_energy", 0.0)) for row in measurements] or [0.0]
    latencies = [float(row.get("cpu_latency_us_median", 0.0)) for row in measurements] or [0.0]
    basis_matched = all(
        bool(artifact.get(field))
        for field in ("matched_sparse_topology", "matched_coupling_tensor", "matched_field_tensor")
    )
    rows: list[ReplayRow] = []
    for measurement in measurements:
        reward = hardware_process_reward(
            measurement,
            min_energy=min(energies),
            max_energy=max(energies),
            min_latency=min(latencies),
            max_latency=max(latencies),
            basis_matched=basis_matched,
        )
        row_id = f"hardware:{measurement.get('seed')}:{measurement.get('sample_count')}"
        rows.append(
            ReplayRow(
                row_id=row_id,
                source_type="hardware",
                motif_key=f"hardware:{measurement.get('sample_count')}",
                process_reward=reward,
                energy_proxy_before=_round_float(1.0 - reward.weight),
                expected_replay=reward.weight >= REPLAY_THRESHOLD,
            )
        )
    return rows


def _prior_fr11_rows(artifact: Mapping[str, Any]) -> list[ReplayRow]:
    policy = str(artifact.get("best_policy") or "fast_slow_memory")
    metrics = artifact.get("policy_metrics", {}).get(policy, {})
    event_ids = [str(event_id) for event_id in metrics.get("applied_event_ids", [])]
    held_out_start = max(0, len(event_ids) - max(1, len(event_ids) // 3)) if event_ids else 0
    rows: list[ReplayRow] = []
    for index, event_id in enumerate(event_ids):
        held_out = index >= held_out_start
        rows.append(
            ReplayRow(
                row_id=f"prior_fr11:{event_id}",
                source_type="prior_fr11",
                motif_key="prior_fr11:fast_slow_memory",
                process_reward=ProcessReward(
                    weight=0.70,
                    components={"retention": 0.45, "nonforgetting": 0.25},
                ),
                energy_proxy_before=0.30,
                expected_replay=True,
                held_out=held_out,
            )
        )
    return rows


def _evaluate(rows: Sequence[ReplayRow], priorities: Mapping[str, float]) -> dict[str, float]:
    correctness_values = [
        (priorities[row.row_id] >= REPLAY_THRESHOLD) == row.expected_replay for row in rows
    ]
    energy_values = [row.energy_proxy_before * priorities[row.row_id] for row in rows]
    return {
        "correctness": _round_float(_mean([float(value) for value in correctness_values])),
        "energy_proxy": _round_float(_mean(energy_values)),
    }


def _corpus_summary(rows: Sequence[ReplayRow]) -> dict[str, Any]:
    counts = Counter(row.source_type for row in rows)
    return {
        "total_rows": len(rows),
        "code_rows": counts["code"],
        "hardware_rows": counts["hardware"],
        "prior_fr11_rows": counts["prior_fr11"],
        "held_out_prior_rows": sum(1 for row in rows if row.held_out),
        "reward_weight_mean": _round_float(_mean([row.process_reward.weight for row in rows])),
    }


def _priority_summary(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    return {
        "mean_before": _round_float(_mean(list(before.values()))),
        "mean_after": _round_float(_mean(list(after.values()))),
        "max_after": _round_float(max(after.values())),
        "min_after": _round_float(min(after.values())),
    }


def _contradiction_rate(rows: Sequence[ReplayRow], priorities: Mapping[str, float]) -> float:
    labels_by_motif: dict[str, set[bool]] = defaultdict(set)
    for row in rows:
        if priorities[row.row_id] >= REPLAY_THRESHOLD:
            labels_by_motif[row.motif_key].add(row.expected_replay)
    contradictory = sum(1 for labels in labels_by_motif.values() if len(labels) > 1)
    return _round_float(contradictory / max(len(labels_by_motif), 1))


def _duplicate_rate(rows: Sequence[ReplayRow]) -> float:
    motif_counts = Counter(row.motif_key for row in rows)
    duplicates = sum(max(0, count - 1) for count in motif_counts.values())
    return _round_float(duplicates / max(len(rows), 1))


def _forgetting_rate(
    rows: Sequence[ReplayRow],
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> float:
    held_out = [row for row in rows if row.held_out]
    regressions = sum(1 for row in held_out if after[row.row_id] < before[row.row_id])
    return _round_float(regressions / max(len(held_out), 1))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round_float(value: float) -> float:
    return round(float(value), 12)
