"""Exp 2887 FR-11 fast/slow memory corrigendum.

The Exp 2882 scale-up artifact was useful because it proved the local replay
set could reach 50 examples, but its core eager-vs-RecMem metrics were not
causal: once a recurrence cluster was found, the replay effect was assigned to
all examples in that cluster, including the examples that created the trigger.
That made RecMem energy and correctness match eager replay exactly on this
corpus. This corrigendum replays the same local examples with three external
memory policies and keeps the RecMem policy causal.

Spec: REQ-LEARN-2887,
      SCENARIO-LEARN-2887,
      SCENARIO-LEARN-2887-GUARD.
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
from carnot.eval.fr11_recmem_recurrence_trigger_v1 import MIN_SUPPORT
from carnot.eval.offline_recurrence_backend_adapter_v2 import OfflineRecurrenceReplayBackend


OUTPUT_FILENAME = "experiment_2887_fr11_fast_slow_memory_corrigendum_v2.json"
EXP2869_FILENAME = "experiment_2869_fr11_continuous_self_learning_replay_v3.json"
EXP2881_FILENAME = "experiment_2881_fr11_recmem_recurrence_trigger_v1.json"
EXP2882_FILENAME = "experiment_2882_fr11_recmem_replay_scaleup_v1.json"
EXP2865_FILENAME = "experiment_2865_cross_corpus_matrix_v5.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260522"
RANDOM_SEED = 2887
TARGET_EXAMPLES = 50
MAX_LOOPS = 3
POLICIES = ("eager_replay", "recmem_causal_triggered", "fast_slow_memory")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "continuous_self_learning_task",
    "fr11_scaleup_clean",
    "source_artifacts",
    "n_examples",
    "target_examples_met",
    "policies_compared",
    "energy_delta_by_policy",
    "correctness_delta_by_policy",
    "auroc_delta_by_policy",
    "token_reduction_by_policy",
    "memory_drift_by_policy",
    "duplicate_rate_by_policy",
    "contradiction_rate_by_policy",
    "forgetting_regression_count_by_policy",
    "best_policy",
    "model_weights_mutated",
    "live_llm_called",
    "random_seed",
    "reproducibility_checksum",
    "tests_run",
    "field_principles",
    "run_date",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal complete:/blocked_ verdict; prior Exp 2882 flags are disclosed.",
    "continuous_self_learning_task": "True because this is the milestone FR-11 memory task.",
    "fr11_scaleup_clean": (
        "True only when target size, no-mutation, no-live-LLM, drift, forgetting, "
        "and non-tautological policy separation checks all pass."
    ),
    "source_artifacts": "Exp 2869, Exp 2881, and the flagged Exp 2882 artifact loaded first.",
    "n_examples": "The deterministic local labeled replay rows evaluated by every policy.",
    "target_examples_met": "Whether the selected local replay set reached at least 50 rows.",
    "policies_compared": "The three policies run on identical examples and seed.",
    "energy_delta_by_policy": "Mean initial energy minus policy final energy.",
    "correctness_delta_by_policy": (
        "Policy final correctness rate minus initial correctness rate; no live repair is inferred."
    ),
    "auroc_delta_by_policy": "Policy final-energy AUROC minus initial-energy AUROC where valid.",
    "token_reduction_by_policy": "Policy memory token proxy reduction versus eager per-row writes.",
    "memory_drift_by_policy": "Contradiction rate plus normalized forgetting, clipped to [0, 1].",
    "duplicate_rate_by_policy": "Duplicate output-memory motif rate for the policy.",
    "contradiction_rate_by_policy": "Fraction of non-clean motifs with conflicting final labels.",
    "forgetting_regression_count_by_policy": "Rows or memory updates that regress energy/correctness.",
    "best_policy": "Highest deterministic utility over energy, token cost, duplicate, and drift terms.",
    "model_weights_mutated": "False because only external memory state is updated.",
    "live_llm_called": "False because all rows are local verifier-label replay metadata.",
    "random_seed": "Controls deterministic corpus selection for every policy.",
    "reproducibility_checksum": "Hashes stable artifact fields, selected rows, and source checksums.",
    "tests_run": "Commands run by the operator/agent for this artifact.",
    "field_principles": "Per-field accounting rules for auditability.",
    "run_date": "Pinned conductor run date.",
    "duration_s": "Real wall-clock duration; no sleep padding.",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the Exp 2887 corrigendum."""

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    run_date: str = RUN_DATE
    random_seed: int = RANDOM_SEED
    target_examples: int = TARGET_EXAMPLES
    max_loops: int = MAX_LOOPS
    min_support: int = MIN_SUPPORT
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        if self.target_examples < 1:
            raise ValueError("target_examples must be >= 1")
        if self.max_loops < 1:
            raise ValueError("max_loops must be >= 1")
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

    def exp2882_path(self) -> Path:
        return self.output_dir() / EXP2882_FILENAME

    def exp2865_path(self) -> Path:
        return self.output_dir() / EXP2865_FILENAME

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


@dataclass(frozen=True)
class PolicyRow:
    """One backend trace row normalized for causal external-memory policies."""

    event_id: str
    source: str
    motif_key: str
    initial_energy: float
    replay_final_energy: float
    initial_correct: bool
    replay_correct: bool
    localized_violations: tuple[str, ...]

    @property
    def replay_energy_delta(self) -> float:
        return _round_float(self.initial_energy - self.replay_final_energy)

    @property
    def is_failure(self) -> bool:
        return self.motif_key != "clean"

    def memory_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source": self.source,
            "motif_key": self.motif_key,
            "initial_energy": _round_float(self.initial_energy),
            "replay_final_energy": _round_float(self.replay_final_energy),
            "initial_correct": self.initial_correct,
            "replay_correct": self.replay_correct,
            "localized_violations": list(self.localized_violations),
        }


class CausalRecMemPolicy:
    """Recurrence-triggered replay that cannot apply effects retroactively."""

    def __init__(self, *, min_support: int) -> None:
        if min_support < 2:
            raise ValueError("min_support must be >= 2")
        self.min_support = int(min_support)
        self._support_by_motif: Counter[str] = Counter()
        self._ready_motifs: set[str] = set()
        self._labels_by_motif: dict[str, set[bool]] = {}

    def should_apply(self, row: PolicyRow) -> bool:
        return row.is_failure and self._support_by_motif[row.motif_key] >= self.min_support

    def observe(self, row: PolicyRow, *, applied: bool) -> None:
        if not row.is_failure:
            return
        self._labels_by_motif.setdefault(row.motif_key, set()).add(row.replay_correct)
        self._support_by_motif[row.motif_key] += 1
        if applied:
            self._ready_motifs.add(row.motif_key)

    def contradiction_rate(self) -> float:
        return _contradiction_rate_from_labels(self._labels_by_motif)

    def duplicate_rate(self) -> float:
        return 0.0

    def token_proxy_after(self) -> int:
        return sum(len(motif) + 96 for motif in sorted(self._ready_motifs))


class FastSlowMemoryPolicy:
    """Memini-style external memory with fast edges, slow edges, and forgetting.

    The policy is an honest deterministic proxy for the paper idea, not an
    implementation of Memini's full graph dynamics. Fast strength captures
    recent episodic verifier evidence; slow strength is a consolidated edge
    promoted from repeated fast evidence. Contradictory or regressive evidence
    weakens both stores instead of preserving unsafe memory.
    """

    def __init__(
        self,
        *,
        fast_threshold: float = 0.2,
        slow_threshold: float = 0.5,
        fast_decay: float = 0.85,
        consolidation_threshold: float = 0.7,
        consolidation_rate: float = 0.35,
        forgetting_rate: float = 0.5,
    ) -> None:
        self.fast_threshold = float(fast_threshold)
        self.slow_threshold = float(slow_threshold)
        self.fast_decay = float(fast_decay)
        self.consolidation_threshold = float(consolidation_threshold)
        self.consolidation_rate = float(consolidation_rate)
        self.forgetting_rate = float(forgetting_rate)
        self.fast_strength_by_motif: dict[str, float] = {}
        self.slow_strength_by_motif: dict[str, float] = {}
        self._labels_by_motif: dict[str, set[bool]] = {}
        self._active_motifs: set[str] = set()
        self.forgetting_regression_count = 0

    def should_apply(self, row: PolicyRow) -> bool:
        if not row.is_failure:
            return False
        fast = self.fast_strength_by_motif.get(row.motif_key, 0.0)
        slow = self.slow_strength_by_motif.get(row.motif_key, 0.0)
        return fast >= self.fast_threshold or slow >= self.slow_threshold

    def observe(self, row: PolicyRow) -> None:
        if not row.is_failure:
            self._decay_fast_edges()
            return
        labels = self._labels_by_motif.setdefault(row.motif_key, set())
        contradiction = bool(labels and row.replay_correct not in labels)
        regression = row.replay_final_energy > row.initial_energy or (
            row.initial_correct and not row.replay_correct
        )
        if contradiction or regression:
            self._forget(row.motif_key)
            if regression:
                self.forgetting_regression_count += 1
        else:
            evidence = max(0.0, row.replay_energy_delta)
            previous_fast = self.fast_strength_by_motif.get(row.motif_key, 0.0)
            fast = previous_fast * self.fast_decay + evidence
            self.fast_strength_by_motif[row.motif_key] = _round_float(fast)
            if fast >= self.consolidation_threshold:
                previous_slow = self.slow_strength_by_motif.get(row.motif_key, 0.0)
                slow = previous_slow + self.consolidation_rate * fast
                self.slow_strength_by_motif[row.motif_key] = _round_float(slow)
            self._active_motifs.add(row.motif_key)
        labels.add(row.replay_correct)

    def contradiction_rate(self) -> float:
        return _contradiction_rate_from_labels(self._labels_by_motif)

    def duplicate_rate(self) -> float:
        return 0.0

    def token_proxy_after(self) -> int:
        payload = {
            "fast": self.fast_strength_by_motif,
            "slow": self.slow_strength_by_motif,
            "active_motifs": sorted(self._active_motifs),
        }
        return len(json.dumps(payload, sort_keys=True)) + 96 * len(self._active_motifs)

    def _forget(self, motif: str) -> None:
        keep = max(0.0, 1.0 - self.forgetting_rate)
        self.fast_strength_by_motif[motif] = _round_float(
            self.fast_strength_by_motif.get(motif, 0.0) * keep
        )
        self.slow_strength_by_motif[motif] = _round_float(
            self.slow_strength_by_motif.get(motif, 0.0) * keep
        )

    def _decay_fast_edges(self) -> None:
        for motif, strength in list(self.fast_strength_by_motif.items()):
            self.fast_strength_by_motif[motif] = _round_float(strength * self.fast_decay)


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    tests_run: Sequence[str] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Run the bounded fast/slow memory corrigendum."""

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
    rows = _rows_from_examples(examples, active_config.max_loops)
    n_examples = len(rows)
    target_examples_met = n_examples >= active_config.target_examples
    target_note = (
        f"selected {n_examples} local labeled examples"
        if target_examples_met
        else f"only {n_examples} local labeled examples available for target {active_config.target_examples}"
    )
    metrics = _compare_policies(rows, min_support=active_config.min_support)
    clean_checks = _adversarial_clean_checks(
        metrics=metrics,
        target_examples_met=target_examples_met,
    )
    fr11_scaleup_clean = all(clean_checks.values())
    if not target_examples_met:
        honest_verdict = "blocked_target_examples_not_met"
    elif fr11_scaleup_clean:
        honest_verdict = "complete: causal RecMem and fast/slow memory separated replay metrics cleanly"
    else:
        honest_verdict = "complete_with_clean_gate_false: corrigendum metrics are disclosed but guarded"

    artifact: dict[str, Any] = {
        "artifact": "experiment_2887_fr11_fast_slow_memory_corrigendum_v2",
        "schema": "carnot.fr11.fast_slow_memory_corrigendum.v2",
        "honest_verdict": honest_verdict,
        "continuous_self_learning_task": True,
        "fr11_scaleup_clean": fr11_scaleup_clean,
        "source_artifacts": _source_artifacts(active_config),
        "n_examples": n_examples,
        "target_examples_met": target_examples_met,
        "policies_compared": list(POLICIES),
        "energy_delta_by_policy": metrics["energy_delta_by_policy"],
        "correctness_delta_by_policy": metrics["correctness_delta_by_policy"],
        "auroc_delta_by_policy": metrics["auroc_delta_by_policy"],
        "token_reduction_by_policy": metrics["token_reduction_by_policy"],
        "memory_drift_by_policy": metrics["memory_drift_by_policy"],
        "duplicate_rate_by_policy": metrics["duplicate_rate_by_policy"],
        "contradiction_rate_by_policy": metrics["contradiction_rate_by_policy"],
        "forgetting_regression_count_by_policy": metrics[
            "forgetting_regression_count_by_policy"
        ],
        "best_policy": metrics["best_policy"],
        "model_weights_mutated": False,
        "live_llm_called": False,
        "random_seed": active_config.random_seed,
        "tests_run": list(tests_run or []),
        "field_principles": FIELD_PRINCIPLES,
        "run_date": active_config.run_date,
        "duration_s": _round_float(active_config.clock() - started_at),
        "exp2882_flag_diagnosis": _diagnose_exp2882(_read_json(active_config.exp2882_path())),
        "adversarial_clean_checks": clean_checks,
        "selected_examples_checksum": _selected_examples_checksum(examples),
        "selected_example_ids": [f"{example.source}::{example.example_id}" for example in examples],
        "source_file_checksums": _source_file_checksums(active_config),
        "target_examples_note": target_note,
        "preconditions_checked": [*source_checks, *corpus_checks],
        "policy_metrics": metrics["policy_metrics"],
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    if write:
        _write_json(active_config.output_path(), artifact)
    return artifact


def _rows_from_examples(examples: Sequence[ReplayExample], max_loops: int) -> list[PolicyRow]:
    backend = OfflineRecurrenceReplayBackend(max_loops=max_loops)
    replay = backend.replay(build_backend_rows(examples, max_loops))
    return [_policy_row_from_trace(trace) for trace in replay["per_example_trace"]]


def _compare_policies(rows: Sequence[PolicyRow], *, min_support: int) -> dict[str, Any]:
    baseline_auc = _roc_auc(
        [0 if row.initial_correct else 1 for row in rows],
        [row.initial_energy for row in rows],
    )
    token_proxy_before = sum(_event_token_proxy(row) for row in rows)
    policy_results = {
        "eager_replay": _evaluate_eager(rows, token_proxy_before, baseline_auc),
        "recmem_causal_triggered": _evaluate_causal_recmem(
            rows,
            min_support=min_support,
            token_proxy_before=token_proxy_before,
            baseline_auc=baseline_auc,
        ),
        "fast_slow_memory": _evaluate_fast_slow(
            rows,
            token_proxy_before=token_proxy_before,
            baseline_auc=baseline_auc,
        ),
    }
    energy = {name: result["energy_delta_mean"] for name, result in policy_results.items()}
    correctness = {name: result["correctness_delta"] for name, result in policy_results.items()}
    auroc = {name: result["auroc_delta"] for name, result in policy_results.items()}
    token = {name: result["token_reduction_pct"] for name, result in policy_results.items()}
    drift = {name: result["memory_drift_score"] for name, result in policy_results.items()}
    duplicate = {name: result["duplicate_rate"] for name, result in policy_results.items()}
    contradiction = {
        name: result["contradiction_rate"] for name, result in policy_results.items()
    }
    forgetting = {
        name: result["forgetting_regression_count"] for name, result in policy_results.items()
    }
    best_policy = max(
        POLICIES,
        key=lambda policy: _policy_utility(
            energy[policy],
            token[policy],
            duplicate[policy],
            drift[policy],
        ),
    )
    return {
        "energy_delta_by_policy": energy,
        "correctness_delta_by_policy": correctness,
        "auroc_delta_by_policy": auroc,
        "token_reduction_by_policy": token,
        "memory_drift_by_policy": drift,
        "duplicate_rate_by_policy": duplicate,
        "contradiction_rate_by_policy": contradiction,
        "forgetting_regression_count_by_policy": forgetting,
        "best_policy": best_policy,
        "policy_metrics": policy_results,
    }


def _evaluate_eager(
    rows: Sequence[PolicyRow],
    token_proxy_before: int,
    baseline_auc: float | None,
) -> dict[str, Any]:
    final_energies = [row.replay_final_energy for row in rows]
    final_correct = [row.replay_correct for row in rows]
    labels = [0 if row.initial_correct else 1 for row in rows]
    contradiction_rate = _memory_contradiction_rate(rows)
    forgetting = _forgetting_regression_count(rows, final_energies, final_correct)
    return _policy_result(
        rows=rows,
        final_energies=final_energies,
        final_correct=final_correct,
        labels=labels,
        baseline_auc=baseline_auc,
        token_proxy_before=token_proxy_before,
        token_proxy_after=token_proxy_before,
        duplicate_rate=_duplicate_rate([row.motif_key for row in rows]),
        contradiction_rate=contradiction_rate,
        forgetting_regression_count=forgetting,
        applied_event_ids=[row.event_id for row in rows],
    )


def _evaluate_causal_recmem(
    rows: Sequence[PolicyRow],
    *,
    min_support: int,
    token_proxy_before: int,
    baseline_auc: float | None,
) -> dict[str, Any]:
    policy = CausalRecMemPolicy(min_support=min_support)
    final_energies: list[float] = []
    final_correct: list[bool] = []
    applied_event_ids: list[str] = []
    for row in rows:
        applied = policy.should_apply(row)
        final_energies.append(row.replay_final_energy if applied else row.initial_energy)
        final_correct.append(row.replay_correct if applied else row.initial_correct)
        if applied:
            applied_event_ids.append(row.event_id)
        policy.observe(row, applied=applied)
    forgetting = _forgetting_regression_count(rows, final_energies, final_correct)
    return _policy_result(
        rows=rows,
        final_energies=final_energies,
        final_correct=final_correct,
        labels=[0 if row.initial_correct else 1 for row in rows],
        baseline_auc=baseline_auc,
        token_proxy_before=token_proxy_before,
        token_proxy_after=policy.token_proxy_after(),
        duplicate_rate=policy.duplicate_rate(),
        contradiction_rate=policy.contradiction_rate(),
        forgetting_regression_count=forgetting,
        applied_event_ids=applied_event_ids,
    )


def _evaluate_fast_slow(
    rows: Sequence[PolicyRow],
    *,
    token_proxy_before: int,
    baseline_auc: float | None,
) -> dict[str, Any]:
    policy = FastSlowMemoryPolicy()
    final_energies: list[float] = []
    final_correct: list[bool] = []
    applied_event_ids: list[str] = []
    for row in rows:
        applied = policy.should_apply(row)
        final_energies.append(row.replay_final_energy if applied else row.initial_energy)
        final_correct.append(row.replay_correct if applied else row.initial_correct)
        if applied:
            applied_event_ids.append(row.event_id)
        policy.observe(row)
    forgetting = _forgetting_regression_count(rows, final_energies, final_correct)
    forgetting = max(forgetting, policy.forgetting_regression_count)
    return _policy_result(
        rows=rows,
        final_energies=final_energies,
        final_correct=final_correct,
        labels=[0 if row.initial_correct else 1 for row in rows],
        baseline_auc=baseline_auc,
        token_proxy_before=token_proxy_before,
        token_proxy_after=policy.token_proxy_after(),
        duplicate_rate=policy.duplicate_rate(),
        contradiction_rate=policy.contradiction_rate(),
        forgetting_regression_count=forgetting,
        applied_event_ids=applied_event_ids,
        extra={
            "fast_strength_by_motif": policy.fast_strength_by_motif,
            "slow_strength_by_motif": policy.slow_strength_by_motif,
        },
    )


def _policy_result(
    *,
    rows: Sequence[PolicyRow],
    final_energies: Sequence[float],
    final_correct: Sequence[bool],
    labels: Sequence[int],
    baseline_auc: float | None,
    token_proxy_before: int,
    token_proxy_after: int,
    duplicate_rate: float,
    contradiction_rate: float,
    forgetting_regression_count: int,
    applied_event_ids: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy_auc = _roc_auc(labels, final_energies)
    auroc_delta = (
        _round_float(policy_auc - baseline_auc)
        if policy_auc is not None and baseline_auc is not None
        else 0.0
    )
    result: dict[str, Any] = {
        "energy_delta_mean": _round_float(
            _mean(
                [
                    row.initial_energy - final_energy
                    for row, final_energy in zip(rows, final_energies, strict=True)
                ]
            )
        ),
        "correctness_delta": _round_float(
            _mean([1.0 if value else 0.0 for value in final_correct])
            - _mean([1.0 if row.initial_correct else 0.0 for row in rows])
        ),
        "auroc_delta": auroc_delta,
        "auroc": _round_float(policy_auc) if policy_auc is not None else None,
        "token_reduction_pct": _token_reduction(token_proxy_before, token_proxy_after),
        "token_proxy_before": token_proxy_before,
        "token_proxy_after": token_proxy_after,
        "duplicate_rate": _round_float(duplicate_rate),
        "contradiction_rate": _round_float(contradiction_rate),
        "memory_drift_score": _memory_drift_score(
            contradiction_rate,
            forgetting_regression_count,
            len(rows),
        ),
        "forgetting_regression_count": int(forgetting_regression_count),
        "applied_event_count": len(applied_event_ids),
        "applied_event_ids": list(applied_event_ids),
    }
    if extra:
        result.update(extra)
    return result


def _source_preconditions(config: ExperimentConfig) -> tuple[list[dict[str, Any]], str | None]:
    exp2869 = _read_json(config.exp2869_path())
    exp2881 = _read_json(config.exp2881_path())
    exp2882 = _read_json(config.exp2882_path())
    checks = [
        _artifact_check("exp2869_artifact", config.exp2869_path(), exp2869),
        {
            "check": "exp2869_replay_ready",
            "passed": bool(exp2869.get("fr11_self_learning_ready")),
            "observed": exp2869.get("honest_verdict", "missing"),
        },
        _artifact_check("exp2881_artifact", config.exp2881_path(), exp2881),
        {
            "check": "exp2881_recmem_ready",
            "passed": bool(exp2881.get("recmem_trigger_ready")),
            "observed": exp2881.get("honest_verdict", "missing"),
        },
        _artifact_check("exp2882_artifact", config.exp2882_path(), exp2882),
        {
            "check": "exp2882_loaded_for_corrigendum",
            "passed": bool(exp2882),
            "observed": exp2882.get("honest_verdict", "missing"),
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
    if "exp2882_artifact" in failed:
        return checks, "blocked_missing_exp2882_artifact"
    return checks, None


def _blocked_artifact(
    config: ExperimentConfig,
    honest_verdict: str,
    preconditions: Sequence[Mapping[str, Any]],
    started_at: float,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "artifact": "experiment_2887_fr11_fast_slow_memory_corrigendum_v2",
        "schema": "carnot.fr11.fast_slow_memory_corrigendum.v2",
        "honest_verdict": honest_verdict,
        "continuous_self_learning_task": True,
        "fr11_scaleup_clean": False,
        "source_artifacts": _source_artifacts(config),
        "n_examples": 0,
        "target_examples_met": False,
        "policies_compared": list(POLICIES),
        "energy_delta_by_policy": {},
        "correctness_delta_by_policy": {},
        "auroc_delta_by_policy": {},
        "token_reduction_by_policy": {},
        "memory_drift_by_policy": {},
        "duplicate_rate_by_policy": {},
        "contradiction_rate_by_policy": {},
        "forgetting_regression_count_by_policy": {},
        "best_policy": "none",
        "model_weights_mutated": False,
        "live_llm_called": False,
        "random_seed": config.random_seed,
        "tests_run": [],
        "field_principles": FIELD_PRINCIPLES,
        "run_date": config.run_date,
        "duration_s": _round_float(config.clock() - started_at),
        "preconditions_checked": list(preconditions),
        "target_examples_note": "blocked before local replay set selection",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _policy_row_from_trace(trace: Mapping[str, Any]) -> PolicyRow:
    localized = trace.get("localized_violations", [])
    violations = tuple(str(item) for item in localized) if isinstance(localized, list) else ()
    motif = "+".join(sorted(violations)) if violations else "clean"
    return PolicyRow(
        event_id=f"exp2887::{trace.get('source', 'unknown')}::{trace.get('example_id', 'unknown')}",
        source=str(trace.get("source") or "verifier_trace"),
        motif_key=motif,
        initial_energy=_round_float(float(trace.get("energy_before", 0.0))),
        replay_final_energy=_round_float(_final_energy(trace)),
        initial_correct=bool(trace.get("correctness_before")),
        replay_correct=bool(trace.get("correctness_after")),
        localized_violations=violations,
    )


def _diagnose_exp2882(exp2882: Mapping[str, Any]) -> dict[str, Any]:
    pending = exp2882.get("corrigendum_pending", [])
    critical = [
        item for item in pending if isinstance(item, Mapping) and item.get("severity") == "critical"
    ]
    return {
        "flagged_adversarial": bool(exp2882.get("flagged_adversarial")) or bool(critical),
        "root_cause": "retroactive_cluster_application",
        "evidence": list(critical),
        "explanation": (
            "Exp 2882 formed recurrence clusters over the full selected set and then "
            "credited replay to all event_ids in a ready cluster. On the local corpus, "
            "all rows with energy improvement belonged to the recurring failure motif, "
            "so RecMem energy/correctness/AUROC matched eager replay exactly."
        ),
    }


def _adversarial_clean_checks(
    *,
    metrics: Mapping[str, Any],
    target_examples_met: bool,
) -> dict[str, bool]:
    energy = metrics["energy_delta_by_policy"]
    drift = metrics["memory_drift_by_policy"]
    forgetting = metrics["forgetting_regression_count_by_policy"]
    return {
        "target_examples_met": target_examples_met,
        "model_weights_mutated_false": True,
        "live_llm_called_false": True,
        "non_tautological_policy_energy": len({_round_float(value) for value in energy.values()})
        == len(POLICIES),
        "causal_recmem_not_retroactive_eager": energy["recmem_causal_triggered"]
        < energy["eager_replay"],
        "fast_slow_separates_from_recmem": energy["fast_slow_memory"]
        > energy["recmem_causal_triggered"],
        "zero_memory_drift": all(value == 0.0 for value in drift.values()),
        "zero_forgetting_regressions": all(value == 0 for value in forgetting.values()),
    }


def _source_artifacts(config: ExperimentConfig) -> list[str]:
    return [
        _relative_path(config.exp2869_path(), config.repo_root),
        _relative_path(config.exp2881_path(), config.repo_root),
        _relative_path(config.exp2882_path(), config.repo_root),
    ]


def _source_file_checksums(config: ExperimentConfig) -> dict[str, str]:
    paths = [
        config.exp2865_path(),
        config.repo_root / "data" / "fover_corpus.jsonl",
        config.repo_root / "data" / "eval_manifests" / "halueval_20260522.jsonl",
        config.repo_root / "data" / "eval_manifests" / "fever_20260522.jsonl",
    ]
    return {
        _relative_path(path, config.repo_root): _sha256_file(path)
        for path in paths
        if path.is_file()
    }


def _selected_examples_checksum(examples: Sequence[ReplayExample]) -> str:
    payload = [
        {
            "example_id": example.example_id,
            "source": example.source,
            "energy_before": example.energy_before,
            "correctness_before": example.correctness_before,
            "localized_violations": list(example.localized_violations),
        }
        for example in examples
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_check(name: str, path: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "check": name,
        "passed": bool(artifact),
        "observed": str(path) if artifact else "missing",
    }


def _memory_contradiction_rate(rows: Sequence[PolicyRow]) -> float:
    labels: dict[str, set[bool]] = {}
    for row in rows:
        if row.is_failure:
            labels.setdefault(row.motif_key, set()).add(row.replay_correct)
    return _contradiction_rate_from_labels(labels)


def _contradiction_rate_from_labels(labels_by_motif: Mapping[str, set[bool]]) -> float:
    if not labels_by_motif:
        return 0.0
    contradictions = sum(1 for labels in labels_by_motif.values() if len(labels) > 1)
    return _round_float(contradictions / len(labels_by_motif))


def _forgetting_regression_count(
    rows: Sequence[PolicyRow],
    final_energies: Sequence[float],
    final_correct: Sequence[bool],
) -> int:
    return sum(
        1
        for row, energy, correct in zip(rows, final_energies, final_correct, strict=True)
        if energy > row.initial_energy or (row.initial_correct and not correct)
    )


def _duplicate_rate(motifs: Sequence[str]) -> float:
    if not motifs:
        return 0.0
    counts = Counter(motifs)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return _round_float(duplicates / len(motifs))


def _memory_drift_score(
    contradiction_rate: float,
    forgetting_regression_count: int,
    n_examples: int,
) -> float:
    if n_examples <= 0:
        return 0.0
    return _round_float(
        min(1.0, max(0.0, contradiction_rate + forgetting_regression_count / n_examples))
    )


def _event_token_proxy(row: PolicyRow) -> int:
    return len(json.dumps(row.memory_payload(), sort_keys=True)) + 96


def _token_reduction(before: int, after: int) -> float:
    if before <= 0:
        return 0.0
    return _round_float(max(0.0, (1.0 - after / before) * 100.0))


def _policy_utility(
    energy_delta: float,
    token_reduction: float,
    duplicate_rate: float,
    memory_drift: float,
) -> float:
    return _round_float(energy_delta + token_reduction * 0.001 - duplicate_rate * 0.05 - memory_drift)


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


def _final_energy(trace: Mapping[str, Any]) -> float:
    loops = trace.get("energy_after_each_loop", [])
    if isinstance(loops, list) and loops:
        return float(loops[-1])
    return float(trace.get("energy_before", 0.0))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:  # pragma: no cover - external caller paths are not used in tests.
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
