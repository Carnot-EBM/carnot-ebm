"""Exp 4273 ARC cross-family Tier-1 online adaptation.

Spec refs: REQ-VERIFY-4273, SCENARIO-VERIFY-4273.

This module keeps the learning mechanism deliberately small: every selector arm
is just a counter-backed precision estimate. The static Set-Encoder and each raw
candidate feature nominate candidates; after a family is scored, the verifier
labels update those arm counters for future families. No model weights are
trained or fine-tuned here, which is the point of the Tier-1 self-learning test.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

from carnot.reporting import arc_cross_family_transfer_existing_pool_4271 as exp4271
from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244


RANDOM_SEED = 4273
BOOTSTRAP_RESAMPLES = 2000
OUTPUT_REL = Path("results/experiment_4273_arc_cross_family_online_adaptation.json")
PROVENANCE_REL = exp4271.PROVENANCE_REL
PRIOR_CROSS_FAMILY_REL = exp4271.OUTPUT_REL
SET_ENCODER_BUILD_REL = exp4271.SET_ENCODER_BUILD_REL
SET_ENCODER_MODEL_REL = exp4271.SET_ENCODER_MODEL_REL
DEFERRED_TO_FRESH_POOL_VERDICT = "complete_self_learning_deferred_to_fresh_pool"
BLOCKED_INPUTS_VERDICT = "blocked_arc_cross_family_online_adaptation_inputs_missing"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
TIER1_COUNTER_UPDATE = "cpu_counter_update_only_lt_1us_no_model_retrain"
STATIC_ARM = "subverifier:set_encoder_static_4271"
VOTE_ARM = "subverifier:vote_weight"
FEATURE_PREFIX = "feature:"
PRECISION_PRIOR = 0.5
SPEC_REFS = ["REQ-VERIFY-4273", "SCENARIO-VERIFY-4273"]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An online-adaptation gain AND an honest static-is-the-ceiling "
        "null are BOTH COMPLETE -- both are valid self-learning findings."
    ),
    "online_adaptation_helps": (
        "BARE bool: true iff online-reweighted cross-family delta exceeds static AND the "
        "(online-static) CI95 excludes 0 -- the self-learning value verdict."
    ),
    "static_cross_family_delta": (
        "BARE float: the static set-encoder's held-out-family delta -- the "
        "no-adaptation baseline."
    ),
    "online_cross_family_delta": (
        "BARE float: the online-reweighted selector's held-out-family delta -- the "
        "Tier-1 self-learning result."
    ),
    "adaptation_curve": (
        "Per-family (online - static) gain as families stream in -- shows whether "
        "adaptation compounds (the continuous-self-learning signature) or is flat."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- learned selector with online weight updates, no demo "
        "execution and no fine-tuning."
    ),
    "random_seed": (
        "Determinism precondition; the family stream order + reweighting reproducible."
    ),
    "reproducibility_checksum": (
        "Hash of the folds + reweighting trace; lets a third party re-run."
    ),
    "model_specs": (
        "The Tier-1 online-reweighting rule (per-feature precision update) + the "
        "family stream protocol; required methodology."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "online_adaptation_helps",
    "static_cross_family_delta",
    "online_cross_family_delta",
    "online_minus_static_ci95",
    "adaptation_curve",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
    "adversarial_verify",
)


class DeferredToFreshPool(RuntimeError):
    """Expected A1 infeasibility path for the fresh-pool fork."""


class BlockedRun(RuntimeError):
    """Expected input failure that still writes an honest terminal artifact."""


@dataclass(frozen=True)
class ExperimentInputs:
    corpus: exp4271.FamilyAnnotatedCorpus
    prior_cross_family_artifact: dict[str, Any]
    build_artifact: dict[str, Any]
    model_artifact: dict[str, Any]
    provenance_artifact: dict[str, Any]


@dataclass
class ArmStats:
    correct: int = 0
    total: int = 0

    def record(self, hit: bool) -> None:
        self.correct += int(bool(hit))
        self.total += 1

    @property
    def precision(self) -> float:
        return (self.correct + PRECISION_PRIOR) / float(self.total + 2.0 * PRECISION_PRIOR)

    def to_json(self) -> dict[str, Any]:
        return {
            "correct": int(self.correct),
            "total": int(self.total),
            "precision": _round_metric(self.precision),
        }


@dataclass
class PrecisionTracker:
    global_stats: dict[str, ArmStats] = field(default_factory=lambda: defaultdict(ArmStats))
    family_stats: dict[str, dict[str, ArmStats]] = field(default_factory=dict)

    def weight_for(self, arm: str, nearest_family: str | None) -> float:
        global_precision = self.global_stats[arm].precision
        if nearest_family is None or nearest_family not in self.family_stats:
            return global_precision
        local_precision = self.family_stats[nearest_family].get(arm, ArmStats()).precision
        return 0.5 * global_precision + 0.5 * local_precision

    def record_family(self, family_id: str, arm_hits: Mapping[str, Sequence[bool]]) -> None:
        local: dict[str, ArmStats] = defaultdict(ArmStats)
        for arm, hits in arm_hits.items():
            for hit in hits:
                self.global_stats[arm].record(bool(hit))
                local[arm].record(bool(hit))
        self.family_stats[family_id] = dict(local)

    def precision_table(self) -> dict[str, dict[str, Any]]:
        return {
            arm: stats.to_json()
            for arm, stats in sorted(self.global_stats.items())
            if stats.total > 0
        }


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _safe_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _read_json_object(path: Path) -> dict[str, Any]:  # pragma: no cover - exercised by live run.
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _resolve_required_path(
    repo_root: Path,
    value: Any,
    fallback: Path | None = None,
) -> Path:  # pragma: no cover - exercised by live run.
    candidate = value if isinstance(value, str) and value else str(fallback or "")
    if not candidate:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    path = Path(candidate)
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.exists():
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    return resolved


def load_inputs(repo_root: Path | str = Path(".")) -> ExperimentInputs:  # pragma: no cover
    root = Path(repo_root)
    try:
        provenance = _read_json_object(root / PROVENANCE_REL)
    except Exception as exc:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT) from exc
    if provenance.get("family_split_feasible") is False:
        raise DeferredToFreshPool(DEFERRED_TO_FRESH_POOL_VERDICT)
    if provenance.get("family_split_feasible") is not True:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    if provenance.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)

    try:
        corpus = exp4271.load_family_annotated_corpus(root)
        prior = _read_json_object(root / PRIOR_CROSS_FAMILY_REL)
        build = _read_json_object(root / SET_ENCODER_BUILD_REL)
        model_path = _resolve_required_path(root, build.get("learned_verifier_path"), SET_ENCODER_MODEL_REL)
        model = exp4244.load_set_encoder(model_path)
    except Exception as exc:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT) from exc
    if prior.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    if build.get("aggregator_trained") is not True or build.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    if model.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    if not isinstance(prior.get("task_rows"), list):
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    return ExperimentInputs(
        corpus=corpus,
        prior_cross_family_artifact=prior,
        build_artifact=build,
        model_artifact=model,
        provenance_artifact=provenance,
    )


def _group_rows_by_task(
    rows: Iterable[exp4271.FamilyAnnotatedRow],
) -> dict[str, list[exp4271.FamilyAnnotatedRow]]:
    grouped: dict[str, list[exp4271.FamilyAnnotatedRow]] = defaultdict(list)
    for row in rows:
        grouped[row.task_id].append(row)
    return {
        task_id: sorted(task_rows, key=lambda item: item.candidate_index)
        for task_id, task_rows in sorted(grouped.items())
    }


def _feature_arms(feature_names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    arms: list[str] = []
    for name in feature_names:
        feature = str(name)
        if feature and feature not in seen:
            seen.add(feature)
            arms.append(FEATURE_PREFIX + feature)
    return arms


def _arm_names(feature_names: Sequence[str]) -> list[str]:
    return [STATIC_ARM, VOTE_ARM, *_feature_arms(feature_names)]


def _static_rows_by_task(static_task_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for item in static_task_rows:
        task_id = str(item.get("task_id") or "")
        if task_id:
            rows[task_id] = item
    return rows


def _pick_by_scores(
    task_rows: Sequence[exp4271.FamilyAnnotatedRow],
    scores: Mapping[str, float],
) -> exp4271.FamilyAnnotatedRow:
    return max(
        task_rows,
        key=lambda row: (
            _safe_float(scores.get(row.candidate_id)),
            row.vote_weight,
            -row.candidate_index,
        ),
    )


def _arm_nominees(
    task_rows: Sequence[exp4271.FamilyAnnotatedRow],
    static_row: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, exp4271.FamilyAnnotatedRow]:
    static_candidate_id = str(static_row.get("set_encoder_candidate_id") or "")
    static_scores = {
        row.candidate_id: 1.0 if row.candidate_id == static_candidate_id else 0.0
        for row in task_rows
    }
    nominees = {
        STATIC_ARM: _pick_by_scores(task_rows, static_scores),
        VOTE_ARM: _pick_by_scores(task_rows, {row.candidate_id: row.vote_weight for row in task_rows}),
    }
    for feature_name in feature_names:
        arm = FEATURE_PREFIX + str(feature_name)
        nominees[arm] = _pick_by_scores(
            task_rows,
            {
                row.candidate_id: _safe_float(row.features.get(str(feature_name)))
                for row in task_rows
            },
        )
    return nominees


def _online_pick(
    task_rows: Sequence[exp4271.FamilyAnnotatedRow],
    nominees: Mapping[str, exp4271.FamilyAnnotatedRow],
    arm_weights: Mapping[str, float],
    static_row: Mapping[str, Any],
) -> exp4271.FamilyAnnotatedRow:
    static_candidate_id = str(static_row.get("set_encoder_candidate_id") or "")
    best_weight_by_candidate: dict[str, float] = defaultdict(float)
    for arm, nominee in nominees.items():
        best_weight_by_candidate[nominee.candidate_id] = max(
            best_weight_by_candidate[nominee.candidate_id],
            _safe_float(arm_weights.get(arm)),
        )
    return max(
        task_rows,
        key=lambda row: (
            best_weight_by_candidate[row.candidate_id],
            1.0 if row.candidate_id == static_candidate_id else 0.0,
            row.vote_weight,
            -row.candidate_index,
        ),
    )


def _family_profiles(
    corpus: exp4271.FamilyAnnotatedCorpus,
    feature_names: Sequence[str],
) -> dict[str, tuple[float, ...]]:
    by_family: dict[str, list[exp4271.FamilyAnnotatedRow]] = defaultdict(list)
    for row in corpus.rows:
        by_family[row.family_id].append(row)
    profiles: dict[str, tuple[float, ...]] = {}
    for family_id, rows in by_family.items():
        values = []
        for feature_name in feature_names:
            feature_values = [_safe_float(row.features.get(str(feature_name))) for row in rows]
            values.append(sum(feature_values) / float(len(feature_values)) if feature_values else 0.0)
        profiles[family_id] = tuple(values)
    return profiles


def _nearest_seen_family(
    family_id: str,
    profiles: Mapping[str, tuple[float, ...]],
    seen_families: Sequence[str],
) -> str | None:
    if not seen_families:
        return None
    current = profiles.get(family_id, ())
    return min(
        seen_families,
        key=lambda other: (
            sum((left - right) ** 2 for left, right in zip(current, profiles.get(other, ()), strict=False)),
            other,
        ),
    )


def _rate(values: Sequence[bool]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _bootstrap_ci95(deltas: Sequence[float], *, random_seed: int, resamples: int) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    if len(deltas) == 1:
        point = _round_metric(deltas[0])
        return [point, point]
    rng = random.Random(random_seed)
    n = len(deltas)
    samples = [
        sum(deltas[rng.randrange(n)] for _ in range(n)) / float(n)
        for _ in range(int(resamples))
    ]
    if not samples:
        point = sum(deltas) / float(len(deltas))
        return [_round_metric(point), _round_metric(point)]
    samples.sort()
    return [
        _round_metric(samples[int(0.025 * (len(samples) - 1))]),
        _round_metric(samples[int(0.975 * (len(samples) - 1))]),
    ]


def _ci_excludes_zero(ci95: Sequence[float]) -> bool:
    return len(ci95) == 2 and (ci95[0] > 0.0 or ci95[1] < 0.0)


def _top_arm_weights(weights: Mapping[str, float], limit: int = 6) -> dict[str, float]:
    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return {arm: _round_metric(weight) for arm, weight in ranked}


def measure_online_adaptation(
    corpus: exp4271.FamilyAnnotatedCorpus,
    static_task_rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """SCENARIO-VERIFY-4273: stream families and update precision counters only."""

    grouped_tasks = _group_rows_by_task(corpus.rows)
    static_by_task = _static_rows_by_task(static_task_rows)
    filtered_features = [name for name in feature_names if any(name in row.features for row in corpus.rows)]
    arms = _arm_names(filtered_features)
    profiles = _family_profiles(corpus, filtered_features)
    family_to_tasks: dict[str, list[str]] = defaultdict(list)
    for task_id, family_id in sorted(corpus.task_family_ids.items()):
        if task_id in grouped_tasks and task_id in static_by_task:
            family_to_tasks[family_id].append(task_id)
    stream_order = sorted(family_to_tasks)

    tracker = PrecisionTracker()
    seen_families: list[str] = []
    vote_hits: list[bool] = []
    static_hits: list[bool] = []
    online_hits: list[bool] = []
    online_minus_static: list[float] = []
    adaptation_curve: list[dict[str, Any]] = []
    task_rows_payload: list[dict[str, Any]] = []
    cumulative_gain_values: list[float] = []

    for stream_index, family_id in enumerate(stream_order):
        nearest = _nearest_seen_family(family_id, profiles, seen_families)
        arm_weights = {arm: tracker.weight_for(arm, nearest) for arm in arms}
        family_vote_hits: list[bool] = []
        family_static_hits: list[bool] = []
        family_online_hits: list[bool] = []
        family_gain_values: list[float] = []
        family_arm_hits: dict[str, list[bool]] = {arm: [] for arm in arms}

        for task_id in family_to_tasks[family_id]:
            task_rows = grouped_tasks[task_id]
            static_row = static_by_task[task_id]
            nominees = _arm_nominees(task_rows, static_row, filtered_features)
            online_pick = _online_pick(task_rows, nominees, arm_weights, static_row)
            vote_hit = bool(static_row.get("vote_correct"))
            static_hit = bool(static_row.get("set_encoder_correct"))
            online_hit = bool(online_pick.correct)
            vote_hits.append(vote_hit)
            static_hits.append(static_hit)
            online_hits.append(online_hit)
            delta = float(online_hit) - float(static_hit)
            online_minus_static.append(delta)
            family_vote_hits.append(vote_hit)
            family_static_hits.append(static_hit)
            family_online_hits.append(online_hit)
            family_gain_values.append(delta)
            for arm, nominee in nominees.items():
                family_arm_hits[arm].append(bool(nominee.correct))
            task_rows_payload.append(
                {
                    "task_id": task_id,
                    "family_id": family_id,
                    "static_candidate_id": str(static_row.get("set_encoder_candidate_id") or ""),
                    "static_correct": static_hit,
                    "vote_candidate_id": str(static_row.get("vote_candidate_id") or ""),
                    "vote_correct": vote_hit,
                    "online_candidate_id": online_pick.candidate_id,
                    "online_correct": online_hit,
                    "nearest_seen_family": nearest,
                }
            )

        tracker.record_family(family_id, family_arm_hits)
        seen_families.append(family_id)
        cumulative_gain_values.extend(family_gain_values)
        family_gain = sum(family_gain_values) / float(len(family_gain_values)) if family_gain_values else 0.0
        cumulative_gain = (
            sum(cumulative_gain_values) / float(len(cumulative_gain_values))
            if cumulative_gain_values
            else 0.0
        )
        adaptation_curve.append(
            {
                "stream_index": stream_index,
                "family_id": family_id,
                "nearest_seen_family": nearest,
                "task_n": len(family_gain_values),
                "static_at_1": _round_metric(_rate(family_static_hits)),
                "online_at_1": _round_metric(_rate(family_online_hits)),
                "vote_at_1": _round_metric(_rate(family_vote_hits)),
                "online_minus_static_gain": _round_metric(family_gain),
                "cumulative_online_minus_static_gain": _round_metric(cumulative_gain),
                "top_arm_weights_before": _top_arm_weights(arm_weights),
            }
        )

    static_delta = _round_metric(_rate(static_hits) - _rate(vote_hits))
    online_delta = _round_metric(_rate(online_hits) - _rate(vote_hits))
    online_static_delta = _round_metric(_rate(online_hits) - _rate(static_hits))
    ci95 = _bootstrap_ci95(
        online_minus_static,
        random_seed=random_seed,
        resamples=bootstrap_resamples,
    )
    helps = bool(online_delta > static_delta and _ci_excludes_zero(ci95))
    return {
        "online_adaptation_helps": helps,
        "static_cross_family_delta": static_delta,
        "online_cross_family_delta": online_delta,
        "online_minus_static_delta": online_static_delta,
        "online_minus_static_ci95": ci95,
        "adaptation_curve": adaptation_curve,
        "pass_rates": {
            "vote_at_1": _round_metric(_rate(vote_hits)),
            "static_set_encoder_at_1": _round_metric(_rate(static_hits)),
            "online_reweighted_at_1": _round_metric(_rate(online_hits)),
        },
        "held_out_family_n": len(stream_order),
        "held_out_task_n": len(static_hits),
        "bootstrap_resamples": int(bootstrap_resamples),
        "tier1_counter_update": TIER1_COUNTER_UPDATE,
        "precision_table": tracker.precision_table(),
        "task_rows": task_rows_payload,
        "feature_arm_n": len(filtered_features),
        "selector_arm_n": len(arms),
        "family_stream_order": stream_order,
    }


def _model_specs(inputs: ExperimentInputs, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family_stream_protocol": {
            "split_unit": "family_id",
            "stream_order_policy": "deterministic_sorted_family_id",
            "family_stream_order": metrics.get("family_stream_order", []),
            "held_out_family_n": metrics.get("held_out_family_n", 0),
            "held_out_task_n": metrics.get("held_out_task_n", 0),
        },
        "tier1_online_reweighting_rule": {
            "arms": "static_set_encoder + vote_weight + per-feature candidate selectors",
            "counter_update": TIER1_COUNTER_UPDATE,
            "precision_prior": PRECISION_PRIOR,
            "nearest_family_weighting": "0.5 * global_precision + 0.5 * nearest_seen_family_precision",
            "current_family_feedback": "not used until after every task in that family is scored",
            "fine_tuning": False,
            "model_training": False,
            "hardware_path": "pure_cpu_counter_updates_lt_1us_per_arm_update",
        },
        "upstream_artifacts": {
            "exp4271_reproducibility_checksum": inputs.prior_cross_family_artifact.get(
                "reproducibility_checksum"
            ),
            "exp4244_build_checksum": inputs.build_artifact.get("reproducibility_checksum"),
            "exp4244_model_checksum": inputs.model_artifact.get("reproducibility_checksum"),
            "exp4270_checksum": inputs.provenance_artifact.get("reproducibility_checksum"),
        },
        "set_encoder_config": inputs.build_artifact.get("model_specs", {}),
    }


def reproducibility_checksum(
    *,
    inputs: ExperimentInputs,
    metrics: Mapping[str, Any],
    random_seed: int,
) -> str:
    payload = {
        "adaptation_curve": metrics.get("adaptation_curve"),
        "family_stream_order": metrics.get("family_stream_order"),
        "manifest_sha256": inputs.corpus.manifest_sha256,
        "model_checksum": inputs.model_artifact.get("reproducibility_checksum"),
        "online_cross_family_delta": metrics.get("online_cross_family_delta"),
        "pool_artifact_sha256": inputs.corpus.pool_artifact_sha256,
        "prior_cross_family_checksum": inputs.prior_cross_family_artifact.get("reproducibility_checksum"),
        "random_seed": int(random_seed),
        "static_cross_family_delta": metrics.get("static_cross_family_delta"),
        "task_family_ids": sorted(inputs.corpus.task_family_ids.items()),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_checksum(reason: str, random_seed: int) -> str:
    raw = json.dumps({"random_seed": int(random_seed), "reason": reason}, sort_keys=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _terminal_artifact(
    reason: str,
    *,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4273_arc_cross_family_online_adaptation",
        "schema": "carnot.arc_cross_family_online_adaptation_4273.v1",
        "status": "complete",
        "honest_verdict": reason,
        "online_adaptation_helps": False,
        "static_cross_family_delta": 0.0,
        "online_cross_family_delta": 0.0,
        "online_minus_static_delta": 0.0,
        "online_minus_static_ci95": [0.0, 0.0],
        "adaptation_curve": [],
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": _blocked_checksum(reason, random_seed),
        "model_specs": {
            "status": "deferred" if reason == DEFERRED_TO_FRESH_POOL_VERDICT else "blocked",
            "blocked_reason": reason,
            "tier1_online_reweighting_rule": {
                "fine_tuning": False,
                "model_training": False,
            },
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": reason == DEFERRED_TO_FRESH_POOL_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "pass_rates": {
            "vote_at_1": 0.0,
            "static_set_encoder_at_1": 0.0,
            "online_reweighted_at_1": 0.0,
        },
        "held_out_family_n": 0,
        "held_out_task_n": 0,
        "bootstrap_resamples": 0,
        "tier1_counter_update": TIER1_COUNTER_UPDATE,
        "precision_table": {},
        "task_rows": [],
        "duration_s": round(duration_s, 6),
        "adversarial_verify": {"status": "pending"},
    }


def _complete_artifact(
    *,
    inputs: ExperimentInputs,
    metrics: dict[str, Any],
    checksum: str,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    honest_read = (
        "online_adaptation_improves_generalization"
        if metrics["online_adaptation_helps"]
        else "static_is_the_ceiling_for_online_adaptation"
    )
    return {
        "experiment": "experiment_4273_arc_cross_family_online_adaptation",
        "schema": "carnot.arc_cross_family_online_adaptation_4273.v1",
        "status": "complete",
        "honest_verdict": f"complete: {honest_read}",
        **metrics,
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "model_specs": _model_specs(inputs, metrics),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "candidate_count": inputs.corpus.candidate_n,
        "candidate_pool_path": str(inputs.corpus.pool_artifact_path),
        "candidate_pool_sha256": inputs.corpus.pool_artifact_sha256,
        "family_manifest_path": str(inputs.corpus.manifest_path),
        "family_manifest_sha256": inputs.corpus.manifest_sha256,
        "duration_s": round(duration_s, 6),
        "adversarial_verify": {"status": "pending"},
    }


def _run_adversarial_verify(repo_root: Path, artifact_path: Path) -> dict[str, Any]:  # pragma: no cover
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "adversarial_verify.py"), "--json", str(artifact_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    payload["returncode"] = proc.returncode
    return payload


def _clean_adversarial_report(report: Mapping[str, Any]) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []
    for item in report.get("reports", []):
        if isinstance(item, dict):
            flags.extend(flag for flag in item.get("flags", []) if isinstance(flag, dict))
    circular_clean = not any(flag.get("kind") == "CIRCULAR_MOAT_OVERCLAIM" for flag in flags)
    return {
        "status": "clean" if not flags else "flagged",
        "circular_moat_overclaim_clean": circular_clean,
        "flag_count": len(flags),
        "flags": flags,
        "returncode": int(report.get("returncode", 0) or 0),
    }


def _bare_float(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a bare float")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("complete_") or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if type(artifact["online_adaptation_helps"]) is not bool:
        raise ValueError("online_adaptation_helps must be a bare bool")
    for field_name in ("static_cross_family_delta", "online_cross_family_delta"):
        _bare_float(artifact[field_name], field_name)
    ci95 = artifact["online_minus_static_ci95"]
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ci95)
    ):
        raise ValueError("online_minus_static_ci95 must be a two-number ci95")
    if not isinstance(artifact["adaptation_curve"], list):
        raise ValueError("adaptation_curve must be a list")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4273")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4273")


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    try:
        inputs = load_inputs(root)
        feature_names = [
            str(name)
            for name in inputs.model_artifact.get("feature_names", exp4244.FEATURE_NAMES)
            if str(name)
        ]
        metrics = measure_online_adaptation(
            inputs.corpus,
            inputs.prior_cross_family_artifact["task_rows"],
            feature_names=feature_names,
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
        )
        checksum = reproducibility_checksum(inputs=inputs, metrics=metrics, random_seed=random_seed)
        artifact = _complete_artifact(
            inputs=inputs,
            metrics=metrics,
            checksum=checksum,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    except DeferredToFreshPool:
        artifact = _terminal_artifact(
            DEFERRED_TO_FRESH_POOL_VERDICT,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    except BlockedRun as exc:
        artifact = _terminal_artifact(
            str(exc) or BLOCKED_INPUTS_VERDICT,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    validate_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw_report = (
        adversarial_runner(output_path)
        if adversarial_runner is not None
        else _run_adversarial_verify(root, output_path)
    )
    artifact["adversarial_verify"] = _clean_adversarial_report(raw_report)
    validate_artifact(artifact)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover - exercised by result entrypoint.
    repo_root = Path(__file__).resolve().parents[3]
    print(json.dumps(run(repo_root), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
