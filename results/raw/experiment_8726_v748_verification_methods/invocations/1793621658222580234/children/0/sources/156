"""Measure whether earlier ARC beliefs improve later action rankings.

The comparison is an offline chronological audit. It uses only logged actions
and immediate observations. It does not call a model or inspect a current
outcome before ranking that decision.

Spec refs: REQ-CSL-7021 and SCENARIO-CSL-7021-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import time
from typing import Any

from carnot.agentic import arc_belief_ledger as ledger_mod


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7021
SCHEMA = "carnot.exp7021.prospective_belief_utility.v1"
SIDECAR_SCHEMA = "carnot.arc.sealed_held_future.v1"
RUN_DATE = "20260905"
RANDOM_SEED = 70_212_026_090_5
BOOTSTRAP_RESAMPLES = 10_000
CAPACITY = 8
INFERENCE_SUBSTRATE = "deterministic_prospective_arc_belief_comparison_no_llm"
ARMS = ("frozen", "recency_only", "counterexample_belief")
CONTROLS = ("frozen", "recency_only")

REPO_ROOT = Path(__file__).resolve().parents[3]
CSL_SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_PATH = Path("python/carnot/agentic/arc_prospective_belief_utility.py")
TEST_PATH = Path("tests/python/test_experiment_7021_prospective_belief_utility.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7021_prospective_belief_utility.py")
OUTPUT_PATH = Path("results/experiment_7021_prospective_belief_utility.json")
EXP7019_ARTIFACT_PATH = Path("results/experiment_7019_arc_belief_stream_fixture.json")
EXP7020_ARTIFACT_PATH = Path("results/experiment_7020_counterexample_belief_ledger.json")
EXP6873_ARTIFACT_PATH = Path("results/experiment_6873_prospective_sealed_self_learning_audit.json")
EXP6978_ARTIFACT_PATH = Path("results/experiment_6978_transactional_constraint_self_learning.json")
FIXTURE_PATH = ledger_mod.EXP7019_FIXTURE_PATH
SIDECAR_PATH = ledger_mod.EXP7019_SIDECAR_PATH
LEDGER_MODULE_PATH = ledger_mod.MODULE_PATH

EXPECTED_EXP7020_HASH = "sha256:dc253079e62f68f2577b03d2b29ff6e89f47e4046917c0c89d92ed09da27ec25"
EXPECTED_FIXTURE_HASH = "sha256:0f7f0e75c6fc8612f2685bb3ff6d1a014795f15aa0cb361b33415343987d98ea"
EXPECTED_SIDECAR_HASH = "sha256:44b19333cd2c617c7eaee469521ca3425f1659313773cd98da66f9249020a3e7"

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "cited_upstream_artifacts",
    "source_artifact_hashes",
    "frozen_split",
    "arm_definition_rows",
    "capacity_match_rows",
    "rows",
    "per_decision_results",
    "per_mechanic_results",
    "no_headroom_rows",
    "unsupported_rows",
    "action_ranking_rows",
    "contradiction_avoidance_rows",
    "retention_rows",
    "query_cost_rows",
    "paired_delta_rows",
    "confidence_intervals",
    "aggregate_row_recomputation",
    "leakage_check_rows",
    "belief_utility_comparison_complete_score",
    "belief_future_utility_positive_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "A principle for each field makes the evidence contract reviewable.",
    "preconditions_checked": "Exact gates stop changed or insufficient evidence before scoring.",
    "inference_substrate": "The substrate states that deterministic rows, not an LLM, make rankings.",
    "duration_s": "Measured wall time proves that the comparison command ran.",
    "cited_upstream_artifacts": "Field citations identify which prior claims the comparison imports.",
    "source_artifact_hashes": "Content hashes bind every imported claim to exact bytes.",
    "frozen_split": "A pre-sidecar protocol prevents target-driven split or metric changes.",
    "arm_definition_rows": "Explicit policies prevent controls from changing after outcomes are seen.",
    "capacity_match_rows": "Equal slot limits make recency a fair bounded-memory control.",
    "rows": "Arm aggregates expose the primary and secondary results without hiding units.",
    "per_decision_results": "Decision rows prove that each ranking used only earlier evidence.",
    "per_mechanic_results": "Mechanic rows show whether one group alone drives a pooled result.",
    "no_headroom_rows": "No-headroom rows prevent a null target contrast from counting as failure.",
    "unsupported_rows": "Unsupported rows expose decisions that lack two logged candidates.",
    "action_ranking_rows": "Pairwise rows give half credit to ties and preserve the primary statistic.",
    "contradiction_avoidance_rows": "Selection rows show how often an arm chooses contradicted evidence.",
    "retention_rows": "Retention rows test whether protected construction facts survive later traffic.",
    "query_cost_rows": "Per-unit bytes and time expose the cost of each memory policy.",
    "paired_delta_rows": "Matched deltas compare belief with each control on the same decisions.",
    "confidence_intervals": "Decision-cluster intervals avoid treating candidate pairs as independent.",
    "aggregate_row_recomputation": "An independent row reduction detects aggregate drift.",
    "leakage_check_rows": "Leakage receipts prove current and sealed outcomes do not enter rankings.",
    "belief_utility_comparison_complete_score": "One means all planned comparison rows are present and consistent.",
    "belief_future_utility_positive_score": "One requires prospective ranking value beyond both controls.",
    "random_seed": "A fixed seed makes the cluster bootstrap reproducible.",
    "reproducibility_checksum": "A timing-free digest detects deterministic artifact drift.",
    "gate_check_summary": "The first failed check gives downstream readers an exact terminal reason.",
    "verifier_is_oracle": "False states that no outcome oracle creates a policy ranking.",
    "verdict_class": "A closed class separates a positive result from a complete null.",
    "honest_verdict": "A class-consistent prefix prevents storage safety from becoming a utility claim.",
}

ROW_TABLES = (
    "arm_definition_rows",
    "capacity_match_rows",
    "rows",
    "per_decision_results",
    "per_mechanic_results",
    "no_headroom_rows",
    "unsupported_rows",
    "action_ranking_rows",
    "contradiction_avoidance_rows",
    "retention_rows",
    "query_cost_rows",
    "paired_delta_rows",
    "confidence_intervals",
    "leakage_check_rows",
)


def _mean(values: Sequence[float]) -> float | None:
    """Return a stable mean while preserving an empty metric as unsupported."""

    if not values:
        return None
    return round(sum(values) / len(values), 12)


def _action_sort_key(action: Mapping[str, Any]) -> bytes:
    return ledger_mod.canonical_bytes(dict(action))


def action_id(event_or_action: Mapping[str, Any]) -> str:
    """Name a candidate from visible action bytes only."""

    raw = event_or_action.get("action", event_or_action)
    if not isinstance(raw, Mapping):
        raise ValueError("action mapping is required")
    return ledger_mod.sha256_bytes(ledger_mod.canonical_bytes(dict(raw)))


def _hypothesis(event: Mapping[str, Any]) -> str:
    signature = event.get("mechanic_signature")
    if not isinstance(signature, Mapping) or not isinstance(signature.get("hypothesis_key"), str):
        raise ValueError("observable hypothesis key is required")
    return str(signature["hypothesis_key"])


def target_class(event: Mapping[str, Any]) -> str:
    """Classify one observed action after ranking has already completed."""

    observation = event.get("next_observation")
    delta = observation.get("state_delta") if isinstance(observation, Mapping) else None
    if not isinstance(delta, Mapping):
        return "invalid"
    changed = delta.get("changed_cell_count", 0)
    changed_count = int(changed) if isinstance(changed, int | float) and not isinstance(changed, bool) else 0
    boundary = bool(observation.get("level_boundary"))
    touches = delta.get("touches_action_coordinate") is True
    if changed_count > 1 or boundary or touches:
        return "progress_producing"
    contradiction = event.get("contradiction")
    if isinstance(contradiction, Mapping) and contradiction.get("is_contradiction") is True:
        return "contradicted"
    return "valid" if changed_count > 0 else "invalid"


def _decision_projection(event: Mapping[str, Any]) -> JsonDict:
    """Keep fields available before the current observation is revealed."""

    action = event.get("action")
    pre_action = event.get("pre_action")
    return {
        "event_id": event.get("event_id"),
        "stream_index": event.get("stream_index"),
        "source_attempt_time": event.get("source_attempt_time"),
        "pre_grid_hash": pre_action.get("grid_hash") if isinstance(pre_action, Mapping) else None,
        "hypothesis_key": _hypothesis(event),
        "action": deepcopy(dict(action)) if isinstance(action, Mapping) else None,
    }


def freeze_protocol(
    events: Sequence[Mapping[str, Any]],
    *,
    capacity: int = CAPACITY,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """Freeze every analysis choice without reading held target fields."""

    if capacity < 1:
        raise ValueError("capacity must be positive")
    indices = [event.get("stream_index") for event in events]
    if indices != list(range(len(events))):
        raise ValueError("events must form a complete chronological stream")
    attempts = [str(event.get("source_attempt_time", "")) for event in events]
    if not attempts or any(not attempt for attempt in attempts):
        raise ValueError("source attempt time is required")
    construction_attempt = attempts[0]
    construction = [event for event in events if event.get("source_attempt_time") == construction_attempt]
    held = [event for event in events if event.get("source_attempt_time") != construction_attempt]
    if not held:
        raise ValueError("at least one later attempt is required")
    if construction + held != list(events):
        raise ValueError("construction attempt must precede every later attempt")
    unit_rows = []
    for event in held:
        projection = _decision_projection(event)
        unit_rows.append(
            {
                "unit_id": ledger_mod._sha256_json({"future_decision": projection}),
                **projection,
            }
        )
    frozen = {
        "split_rule": "first_source_attempt_construction_later_attempts_held",
        "construction_attempt": construction_attempt,
        "construction_stream_indices": [int(event["stream_index"]) for event in construction],
        "held_stream_indices": [int(event["stream_index"]) for event in held],
        "held_unit_rows": unit_rows,
        "arm_names": list(ARMS),
        "capacity": capacity,
        "primary_metric": "mean_pairwise_action_ranking_accuracy",
        "secondary_metrics": [
            "contradicted_action_rate",
            "useful_query_coverage",
            "protected_retention_rate",
            "memory_bytes",
            "query_time_us",
        ],
        "candidate_rule": "distinct_logged_actions_for_observable_hypothesis_at_or_before_decision",
        "tie_credit": 0.5,
        "bootstrap_unit": "future_decision_unit_id",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "random_seed": random_seed,
        "positive_gate": {
            "belief_strictly_beats": list(CONTROLS),
            "paired_interval_lower_bound_minimum": 0.0,
            "non_negative_mechanic_groups_minimum": 2,
            "protected_retention_regression_allowed": False,
        },
        "protocol_frozen_before_sidecar_open": True,
    }
    frozen["protocol_checksum"] = ledger_mod._sha256_json(frozen)
    return frozen


def load_sidecar(path: Path | str) -> list[JsonDict]:
    """Open the held sidecar only after the caller freezes the protocol."""

    try:
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"held sidecar is unreadable: {exc}") from exc
    if any(not isinstance(row, dict) or row.get("schema") != SIDECAR_SCHEMA for row in rows):
        raise ValueError("held sidecar schema is invalid")
    return rows


def classify_candidate_support(candidate_rows: Sequence[Mapping[str, Any]]) -> str:
    """Separate absent candidates from absent useful-versus-contradicted contrast."""

    if len(candidate_rows) < 2:
        return "unsupported"
    classes = {row.get("target_class") for row in candidate_rows}
    useful = bool(classes & {"progress_producing", "valid"})
    return "supported" if useful and "contradicted" in classes else "no_headroom"


def pairwise_ranking_accuracy(
    candidate_rows: Sequence[Mapping[str, Any]], scores: Mapping[str, float]
) -> JsonDict:
    """Compare every useful candidate with every contradicted candidate."""

    useful = [row for row in candidate_rows if row.get("target_class") in {"progress_producing", "valid"}]
    contradicted = [row for row in candidate_rows if row.get("target_class") == "contradicted"]
    pair_credits: list[float] = []
    for good in useful:
        for bad in contradicted:
            good_score = float(scores[str(good["action_id"])])
            bad_score = float(scores[str(bad["action_id"])])
            pair_credits.append(
                1.0 if good_score > bad_score else (0.5 if good_score == bad_score else 0.0)
            )
    return {
        "accuracy": _mean(pair_credits),
        "pair_count": len(pair_credits),
        "credit_sum": round(sum(pair_credits), 12),
    }


def cluster_bootstrap(
    values_by_unit: Mapping[str, float], *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> JsonDict:
    """Bootstrap decision groups so nested candidate pairs never inflate support."""

    if not isinstance(values_by_unit, Mapping):
        raise TypeError("cluster bootstrap requires a mapping of decision IDs")
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    ordered = [float(values_by_unit[key]) for key in sorted(values_by_unit)]
    if not ordered:
        return {
            "point_estimate": None,
            "lower": None,
            "upper": None,
            "cluster_count": 0,
            "resamples": resamples,
            "seed": seed,
        }
    generator = random.Random(seed)
    means = sorted(
        sum(generator.choice(ordered) for _ in ordered) / len(ordered) for _ in range(resamples)
    )
    lower_index = int(0.025 * (resamples - 1))
    upper_index = int(0.975 * (resamples - 1))
    return {
        "point_estimate": _mean(ordered),
        "lower": round(means[lower_index], 12),
        "upper": round(means[upper_index], 12),
        "cluster_count": len(ordered),
        "resamples": resamples,
        "seed": seed,
    }


class RecencyMemory:
    """Keep raw observation rows under the same item limit as the belief arm."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.entries: list[JsonDict] = []

    def observe(self, event: Mapping[str, Any]) -> None:
        self.entries.append(
            {
                "event_id": event["event_id"],
                "stream_index": int(event["stream_index"]),
                "hypothesis_key": _hypothesis(event),
                "action_id": action_id(event),
                "target_class": target_class(event),
                "observed_outcome_key": event["mechanic_signature"]["observed_outcome_key"],
            }
        )
        self.entries = self.entries[-self.capacity :]

    def evidence(self, hypothesis_key: str, candidate_id: str) -> JsonDict | None:
        for row in reversed(self.entries):
            if row["hypothesis_key"] == hypothesis_key and row["action_id"] == candidate_id:
                return row
        return None

    @property
    def memory_bytes(self) -> int:
        return len(ledger_mod.canonical_bytes(self.entries))

    @property
    def max_stream_index(self) -> int:
        return max((int(row["stream_index"]) for row in self.entries), default=-1)


def _evidence_score(label: str, state: str) -> float:
    if label == "contradicted" or state == "contradicted":
        return -3.0
    if label == "invalid":
        return -2.0
    state_bonus = {"known": 3.0, "possible": 2.0, "uncertain": 1.0}.get(state, 0.0)
    return state_bonus if label == "progress_producing" else state_bonus - 0.5


def _candidate_views(history: Sequence[Mapping[str, Any]], decision: Mapping[str, Any]) -> list[JsonDict]:
    hypothesis_key = _hypothesis(decision)
    by_id: dict[str, Mapping[str, Any]] = {}
    for event in [*history, decision]:
        if _hypothesis(event) == hypothesis_key:
            by_id[action_id(event)] = event
    rows = [
        {
            "action_id": candidate_id,
            "action": deepcopy(dict(event["action"])),
            "sort_key": _action_sort_key(event["action"]).decode("utf-8"),
        }
        for candidate_id, event in by_id.items()
    ]
    return sorted(rows, key=lambda row: row["sort_key"])


def _candidate_targets(
    history: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    current_id = action_id(decision)
    rows = []
    for candidate in candidates:
        candidate_id = str(candidate["action_id"])
        if candidate_id == current_id:
            source = decision
            source_kind = "held_current_observation"
        else:
            source = next(
                event
                for event in reversed(history)
                if _hypothesis(event) == _hypothesis(decision) and action_id(event) == candidate_id
            )
            source_kind = "strictly_earlier_logged_observation"
        rows.append(
            {
                "action_id": candidate_id,
                "target_class": target_class(source),
                "target_event_id": source["event_id"],
                "target_stream_index": source["stream_index"],
                "target_source_kind": source_kind,
            }
        )
    return rows


def _rank_recency(
    memory: RecencyMemory, hypothesis_key: str, candidates: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, float], int, int]:
    scores = {}
    for candidate in candidates:
        candidate_id = str(candidate["action_id"])
        evidence = memory.evidence(hypothesis_key, candidate_id)
        scores[candidate_id] = (
            _evidence_score(str(evidence["target_class"]), "possible") if evidence else 0.0
        )
    return scores, memory.max_stream_index, len(memory.entries)


def _rank_belief(
    ledger: ledger_mod.BeliefLedger,
    history: Sequence[Mapping[str, Any]],
    hypothesis_key: str,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], int, int]:
    scores: dict[str, float] = {}
    for candidate in candidates:
        candidate_id = str(candidate["action_id"])
        evidence_rows = [
            event
            for event in reversed(history)
            if _hypothesis(event) == hypothesis_key and action_id(event) == candidate_id
        ]
        score = 0.0
        for evidence in evidence_rows:
            key = ledger_mod.mechanic_key_from_event(evidence)
            outcome_key = str(evidence["mechanic_signature"]["observed_outcome_key"])
            query = ledger.query(key, outcome_key=outcome_key)
            if query["fact_id"] is not None:
                score = _evidence_score(target_class(evidence), str(query["state"]))
                break
        scores[candidate_id] = score
    return scores, int(ledger._state["last_stream_index"]), len(ledger.facts())


def _selected(candidates: Sequence[Mapping[str, Any]], scores: Mapping[str, float]) -> str:
    ordered = sorted(candidates, key=lambda row: row["sort_key"])
    return str(max(ordered, key=lambda row: float(scores[str(row["action_id"])]))["action_id"])


def _arm_definitions(capacity: int) -> list[JsonDict]:
    return [
        {
            "arm": "frozen",
            "memory_read": False,
            "memory_write": False,
            "memory_item_capacity": capacity,
            "ranking_rule": "fixed_zero_score_then_canonical_action_order",
            "terminal": True,
        },
        {
            "arm": "recency_only",
            "memory_read": True,
            "memory_write": True,
            "memory_item_capacity": capacity,
            "ranking_rule": "newest_matching_raw_observation_with_oldest_row_eviction",
            "terminal": True,
        },
        {
            "arm": "counterexample_belief",
            "memory_read": True,
            "memory_write": True,
            "memory_item_capacity": capacity,
            "ranking_rule": "exp7020_tombstone_support_and_protected_fact_query",
            "terminal": True,
        },
    ]


def _retention_rows(
    unit_id: str,
    stream_index: int,
    anchors: Sequence[Mapping[str, Any]],
    recency: RecencyMemory,
    belief: ledger_mod.BeliefLedger,
) -> list[JsonDict]:
    anchor_ids = {str(event_id) for row in anchors for event_id in row["event_ids"]}
    belief_ids = {str(row["fact_id"]) for row in belief.facts() if row["state"] != "contradicted"}
    retained = {
        "frozen": 0,
        "recency_only": sum(
            1
            for anchor in anchors
            if set(map(str, anchor["event_ids"])) & {str(row["event_id"]) for row in recency.entries}
        ),
        "counterexample_belief": sum(1 for anchor in anchors if str(anchor["fact_id"]) in belief_ids),
    }
    return [
        {
            "unit_id": unit_id,
            "stream_index": stream_index,
            "arm": arm,
            "protected_case_count": len(anchors),
            "retained_protected_case_count": retained[arm],
            "protected_retention_rate": (
                round(retained[arm] / len(anchors), 12) if anchors else 1.0
            ),
            "anchor_event_count": len(anchor_ids),
            "stage": "after_observation",
            "terminal": True,
        }
        for arm in ARMS
    ]


def _validate_sidecar_receipts(
    events: Sequence[Mapping[str, Any]], sidecar_rows: Sequence[Mapping[str, Any]]
) -> None:
    observed = [(row.get("after_stream_index"), row.get("event_id")) for row in sidecar_rows]
    expected = [(event.get("stream_index"), event.get("event_id")) for event in events]
    if observed != expected:
        raise ValueError("held sidecar does not match the chronological fixture")


def run_comparison(
    events: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    work_root: Path | str,
) -> JsonDict:
    """Rank held decisions, then update writable memories from each observation."""

    _validate_sidecar_receipts(events, sidecar_rows)
    held_indices = list(protocol["held_stream_indices"])
    construction_indices = list(protocol["construction_stream_indices"])
    capacity = int(protocol["capacity"])
    construction = [events[index] for index in construction_indices]
    held = [events[index] for index in held_indices]
    unit_by_index = {
        int(row["stream_index"]): str(row["unit_id"]) for row in protocol["held_unit_rows"]
    }
    recency = RecencyMemory(capacity)
    belief = ledger_mod.BeliefLedger(Path(work_root) / "belief", capacity=capacity)
    for event in construction:
        recency.observe(event)
        belief.observe(event)
    anchors = [deepcopy(row) for row in belief.facts() if row.get("protected") is True]
    history: list[Mapping[str, Any]] = list(construction)

    per_decision: list[JsonDict] = []
    action_rows: list[JsonDict] = []
    contradiction_rows: list[JsonDict] = []
    query_rows: list[JsonDict] = []
    retention_rows: list[JsonDict] = []
    unsupported_rows: list[JsonDict] = []
    no_headroom_rows: list[JsonDict] = []

    for decision in held:
        stream_index = int(decision["stream_index"])
        unit_id = unit_by_index[stream_index]
        hypothesis_key = _hypothesis(decision)
        mechanic_group = str(decision["mechanic_signature"]["mechanic_group"])
        candidates = _candidate_views(history, decision)
        rankings: dict[str, JsonDict] = {}
        for arm in ARMS:
            started = time.perf_counter_ns()
            if arm == "frozen":
                scores = {str(row["action_id"]): 0.0 for row in candidates}
                max_index, item_count, memory_bytes = -1, 0, len(ledger_mod.canonical_bytes([]))
            elif arm == "recency_only":
                scores, max_index, item_count = _rank_recency(recency, hypothesis_key, candidates)
                memory_bytes = recency.memory_bytes
            else:
                scores, max_index, item_count = _rank_belief(
                    belief, history, hypothesis_key, candidates
                )
                memory_bytes = len(belief.state_bytes)
            elapsed_us = round((time.perf_counter_ns() - started) / 1_000, 3)
            rankings[arm] = {
                "scores": scores,
                "selected": _selected(candidates, scores),
                "max_index": max_index,
                "item_count": item_count,
                "memory_bytes": memory_bytes,
                "query_time_us": elapsed_us,
            }

        targets = _candidate_targets(history, decision, candidates)
        support_class = classify_candidate_support(targets)
        if support_class == "unsupported":
            unsupported_rows.append(
                {
                    "unit_id": unit_id,
                    "stream_index": stream_index,
                    "mechanic_group": mechanic_group,
                    "distinct_candidate_count": len(candidates),
                    "reason": "fewer_than_two_distinct_logged_actions",
                    "terminal": True,
                }
            )
        elif support_class == "no_headroom":
            no_headroom_rows.append(
                {
                    "unit_id": unit_id,
                    "stream_index": stream_index,
                    "mechanic_group": mechanic_group,
                    "candidate_target_classes": sorted(
                        str(row["target_class"]) for row in targets
                    ),
                    "reason": "missing_useful_or_contradicted_target_class",
                    "terminal": True,
                }
            )

        target_by_id = {str(row["action_id"]): str(row["target_class"]) for row in targets}
        for arm in ARMS:
            rank = rankings[arm]
            pairwise = pairwise_ranking_accuracy(targets, rank["scores"])
            selected_id = str(rank["selected"])
            selected_target = target_by_id[selected_id]
            useful_query = arm != "frozen" and any(
                float(value) != 0.0 for value in rank["scores"].values()
            )
            per_decision.append(
                {
                    "unit_id": unit_id,
                    "event_id": decision["event_id"],
                    "stream_index": stream_index,
                    "mechanic_group": mechanic_group,
                    "arm": arm,
                    "support_class": support_class,
                    "candidate_count": len(candidates),
                    "candidate_scores": rank["scores"],
                    "selected_action_id": selected_id,
                    "selected_target_class": selected_target,
                    "action_ranking_accuracy": (
                        pairwise["accuracy"] if support_class == "supported" else None
                    ),
                    "useful_query": useful_query,
                    "declared_capacity": capacity,
                    "memory_item_count": rank["item_count"],
                    "memory_bytes": rank["memory_bytes"],
                    "query_time_us": rank["query_time_us"],
                    "max_evidence_stream_index": rank["max_index"],
                    "strict_chronology": int(rank["max_index"]) < stream_index,
                    "current_outcome_used_by_policy": False,
                    "candidate_targets_used_for_scoring_only": targets,
                    "terminal": True,
                }
            )
            query_rows.append(
                {
                    "unit_id": unit_id,
                    "stream_index": stream_index,
                    "arm": arm,
                    "memory_item_count": rank["item_count"],
                    "memory_bytes": rank["memory_bytes"],
                    "query_time_us": rank["query_time_us"],
                    "terminal": True,
                }
            )
            if support_class == "supported":
                action_rows.append(
                    {
                        "unit_id": unit_id,
                        "stream_index": stream_index,
                        "mechanic_group": mechanic_group,
                        "arm": arm,
                        **pairwise,
                        "terminal": True,
                    }
                )
                contradiction_rows.append(
                    {
                        "unit_id": unit_id,
                        "stream_index": stream_index,
                        "mechanic_group": mechanic_group,
                        "arm": arm,
                        "selected_action_id": selected_id,
                        "selected_target_class": selected_target,
                        "selected_contradicted": selected_target == "contradicted",
                        "terminal": True,
                    }
                )

        recency.observe(decision)
        belief.observe(decision)
        retention_rows.extend(_retention_rows(unit_id, stream_index, anchors, recency, belief))
        history.append(decision)

    comparison: JsonDict = {
        "arm_definition_rows": _arm_definitions(capacity),
        "per_decision_results": per_decision,
        "no_headroom_rows": no_headroom_rows,
        "unsupported_rows": unsupported_rows,
        "action_ranking_rows": action_rows,
        "contradiction_avoidance_rows": contradiction_rows,
        "retention_rows": retention_rows,
        "query_cost_rows": query_rows,
    }
    comparison["rows"] = recompute_aggregate_rows(comparison)
    comparison["capacity_match_rows"] = [
        {
            "arm": arm,
            "declared_capacity": capacity,
            "maximum_observed_item_count": max(
                row["memory_item_count"] for row in query_rows if row["arm"] == arm
            ),
            "same_capacity": True,
            "within_capacity": all(
                row["memory_item_count"] <= capacity for row in query_rows if row["arm"] == arm
            ),
            "terminal": True,
        }
        for arm in ARMS
    ]
    comparison["per_mechanic_results"] = _per_mechanic(action_rows)
    intervals, paired = _interval_rows(action_rows, int(protocol["random_seed"]))
    comparison["confidence_intervals"] = intervals
    comparison["paired_delta_rows"] = paired
    return comparison


def recompute_aggregate_rows(comparison: Mapping[str, Any]) -> list[JsonDict]:
    """Reduce arm metrics from decision-level sufficient statistics."""

    decision_rows = list(comparison.get("per_decision_results", []))
    action_rows = list(comparison.get("action_ranking_rows", []))
    contradiction_rows = list(comparison.get("contradiction_avoidance_rows", []))
    retention_rows = list(comparison.get("retention_rows", []))
    query_rows = list(comparison.get("query_cost_rows", []))
    aggregates = []
    for arm in ARMS:
        arm_decisions = [row for row in decision_rows if row["arm"] == arm]
        arm_actions = [row for row in action_rows if row["arm"] == arm]
        arm_contradictions = [row for row in contradiction_rows if row["arm"] == arm]
        arm_queries = [row for row in query_rows if row["arm"] == arm]
        arm_retention = [row for row in retention_rows if row["arm"] == arm]
        final_retention = max(arm_retention, key=lambda row: row["stream_index"], default=None)
        aggregates.append(
            {
                "arm": arm,
                "planned_decision_count": len(arm_decisions),
                "supported_decision_count": len(arm_actions),
                "action_ranking_accuracy": _mean(
                    [float(row["accuracy"]) for row in arm_actions]
                ),
                "contradicted_action_rate": _mean(
                    [float(bool(row["selected_contradicted"])) for row in arm_contradictions]
                ),
                "useful_query_coverage": _mean(
                    [float(bool(row["useful_query"])) for row in arm_decisions]
                ),
                "final_protected_retention_rate": (
                    final_retention["protected_retention_rate"] if final_retention else None
                ),
                "mean_memory_bytes": _mean(
                    [float(row["memory_bytes"]) for row in arm_queries]
                ),
                "mean_query_time_us": _mean(
                    [float(row["query_time_us"]) for row in arm_queries]
                ),
                "terminal": True,
            }
        )
    return aggregates


def audit_aggregate_recomputation(comparison: Mapping[str, Any]) -> JsonDict:
    """Compare published aggregate rows with an independent fresh reduction."""

    expected = recompute_aggregate_rows(comparison)
    observed = deepcopy(list(comparison.get("rows", [])))
    return {
        "expected_rows": expected,
        "observed_rows": observed,
        "matches": expected == observed,
    }


def _per_mechanic(action_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in action_rows:
        grouped[(str(row["mechanic_group"]), str(row["arm"]))].append(float(row["accuracy"]))
    return [
        {
            "mechanic_group": mechanic,
            "arm": arm,
            "decision_count": len(values),
            "action_ranking_accuracy": _mean(values),
            "terminal": True,
        }
        for (mechanic, arm), values in sorted(grouped.items())
    ]


def _interval_rows(
    action_rows: Sequence[Mapping[str, Any]], seed: int
) -> tuple[list[JsonDict], list[JsonDict]]:
    values = {
        arm: {str(row["unit_id"]): float(row["accuracy"]) for row in action_rows if row["arm"] == arm}
        for arm in ARMS
    }
    intervals = []
    for offset, arm in enumerate(ARMS):
        intervals.append(
            {
                "metric": "action_ranking_accuracy",
                "arm": arm,
                **cluster_bootstrap(values[arm], seed=seed + offset),
                "terminal": True,
            }
        )
    paired = []
    for offset, control in enumerate(CONTROLS, start=len(ARMS)):
        common = sorted(set(values["counterexample_belief"]) & set(values[control]))
        deltas = {
            unit_id: values["counterexample_belief"][unit_id] - values[control][unit_id]
            for unit_id in common
        }
        interval = cluster_bootstrap(deltas, seed=seed + offset)
        row = {
            "metric": "action_ranking_accuracy",
            "comparison": f"counterexample_belief_minus_{control}",
            **interval,
            "terminal": True,
        }
        paired.append(row)
        intervals.append(deepcopy(row))
    return intervals, paired


def _gate(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(expected == observed if passed is None else passed),
        "terminal": True,
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def _read_json(path: Path) -> JsonDict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def collect_preconditions(repo_root: Path, output_path: Path) -> tuple[list[JsonDict], JsonDict, list[JsonDict] | None, JsonDict | None]:
    """Check exact upstream bytes without parsing the held sidecar."""

    exp7020_path = repo_root / EXP7020_ARTIFACT_PATH
    fixture_path = repo_root / FIXTURE_PATH
    sidecar_path = repo_root / SIDECAR_PATH
    exp7020 = _read_json(exp7020_path)
    events: list[JsonDict] | None = None
    fixture_error: str | None = None
    try:
        events = ledger_mod.load_updater_events(fixture_path)
    except ValueError as exc:
        fixture_error = str(exc)
    protocol: JsonDict | None = None
    split_error: str | None = fixture_error
    if events is not None:
        try:
            protocol = freeze_protocol(events)
        except ValueError as exc:
            split_error = str(exc)
    held_groups = (
        sorted(
            {
                str(events[index]["mechanic_signature"]["mechanic_group"])
                for index in protocol["held_stream_indices"]
            }
        )
        if events is not None and protocol is not None
        else []
    )
    ready = exp7020.get("belief_ledger_ready_score") if exp7020 else None
    declared_fixture = (
        exp7020.get("source_artifact_hashes", {}).get(str(FIXTURE_PATH)) if exp7020 else None
    )
    checks = [
        _gate("exp7020_artifact_readable", True, exp7020 is not None),
        _gate(
            "exp7020_belief_ledger_ready_score",
            1,
            ready,
            passed=type(ready) is int and ready == 1,
        ),
        _gate("fixture_loadable", None, split_error),
        _gate("exp7020_artifact_hash", EXPECTED_EXP7020_HASH, ledger_mod.sha256_path(exp7020_path)),
        _gate("fixture_hash", EXPECTED_FIXTURE_HASH, ledger_mod.sha256_path(fixture_path)),
        _gate("fixture_hash_declared_by_exp7020", EXPECTED_FIXTURE_HASH, declared_fixture),
        _gate("held_sidecar_hash", EXPECTED_SIDECAR_HASH, ledger_mod.sha256_path(sidecar_path)),
        _gate(
            "held_mechanic_group_count",
            ">=3",
            len(held_groups),
            passed=len(held_groups) >= 3,
        ),
        _gate("artifact_path_writable", True, ledger_mod._path_is_writable(output_path)),
    ]
    source_hashes = {
        str(EXP7019_ARTIFACT_PATH): ledger_mod.sha256_path(repo_root / EXP7019_ARTIFACT_PATH),
        str(EXP7020_ARTIFACT_PATH): ledger_mod.sha256_path(exp7020_path),
        str(EXP6873_ARTIFACT_PATH): ledger_mod.sha256_path(repo_root / EXP6873_ARTIFACT_PATH),
        str(EXP6978_ARTIFACT_PATH): ledger_mod.sha256_path(repo_root / EXP6978_ARTIFACT_PATH),
        str(FIXTURE_PATH): ledger_mod.sha256_path(fixture_path),
        str(SIDECAR_PATH): ledger_mod.sha256_path(sidecar_path),
        str(LEDGER_MODULE_PATH): ledger_mod.sha256_path(repo_root / LEDGER_MODULE_PATH),
    }
    return checks, source_hashes, events, protocol


def _empty_artifact(run_date: str, source_hashes: Mapping[str, Any]) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "cited_upstream_artifacts": [],
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "frozen_split": {},
        "aggregate_row_recomputation": {"expected_rows": [], "observed_rows": [], "matches": False},
        "belief_utility_comparison_complete_score": 0,
        "belief_future_utility_positive_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_prospective_belief_utility",
    }
    for field in ROW_TABLES:
        artifact[field] = []
    return artifact


def _citations(source_hashes: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "experiment_id": 7019,
            "fields_imported": ["fixture_hash", "held_future_sidecar_hash", "mechanic_signature_rows"],
            "sha256": source_hashes[str(EXP7019_ARTIFACT_PATH)],
        },
        {
            "experiment_id": 7020,
            "fields_imported": ["belief_ledger_ready_score", "capacity_rows", "retention_rows"],
            "sha256": source_hashes[str(EXP7020_ARTIFACT_PATH)],
        },
        {
            "experiment_id": 6873,
            "fields_imported": ["honest_verdict", "held_future_utility_by_arm"],
            "sha256": source_hashes[str(EXP6873_ARTIFACT_PATH)],
        },
        {
            "experiment_id": 6978,
            "fields_imported": ["honest_verdict", "arm_config_rows", "held_future_rows"],
            "sha256": source_hashes[str(EXP6978_ARTIFACT_PATH)],
        },
    ]


def _positive_gate(artifact: Mapping[str, Any]) -> tuple[bool, JsonDict]:
    by_arm = {row["arm"]: row for row in artifact["rows"]}
    belief_value = by_arm.get("counterexample_belief", {}).get("action_ranking_accuracy")
    strict_beats = bool(
        belief_value is not None
        and all(
            by_arm.get(control, {}).get("action_ranking_accuracy") is not None
            and belief_value > by_arm[control]["action_ranking_accuracy"]
            for control in CONTROLS
        )
    )
    paired = {row["comparison"]: row for row in artifact["paired_delta_rows"]}
    interval_lower_nonnegative = all(
        paired.get(f"counterexample_belief_minus_{control}", {}).get("lower") is not None
        and paired[f"counterexample_belief_minus_{control}"]["lower"] >= 0.0
        for control in CONTROLS
    )
    mechanics: dict[str, dict[str, float]] = defaultdict(dict)
    for row in artifact["per_mechanic_results"]:
        mechanics[str(row["mechanic_group"])][str(row["arm"])] = float(
            row["action_ranking_accuracy"]
        )
    nonnegative_groups = sum(
        1
        for values in mechanics.values()
        if "counterexample_belief" in values
        and "recency_only" in values
        and values["counterexample_belief"] - values["recency_only"] >= 0.0
    )
    retention_ok = bool(
        by_arm.get("counterexample_belief", {}).get("final_protected_retention_rate") is not None
        and by_arm.get("recency_only", {}).get("final_protected_retention_rate") is not None
        and by_arm["counterexample_belief"]["final_protected_retention_rate"]
        >= by_arm["recency_only"]["final_protected_retention_rate"]
    )
    details = {
        "strictly_beats_both_controls": strict_beats,
        "paired_interval_lower_bounds_nonnegative": interval_lower_nonnegative,
        "nonnegative_belief_minus_recency_mechanic_group_count": nonnegative_groups,
        "minimum_nonnegative_mechanic_groups": 2,
        "protected_retention_does_not_regress": retention_ok,
    }
    return bool(
        artifact["belief_utility_comparison_complete_score"] == 1
        and strict_beats
        and interval_lower_nonnegative
        and nonnegative_groups >= 2
        and retention_ok
    ), details


def build_artifact(
    *,
    repo_root: Path | str = REPO_ROOT,
    output_path: Path | str | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Build a blocked, null, or positive artifact from exact chronological rows."""

    started = time.perf_counter()
    root = Path(repo_root).resolve()
    output = Path(output_path) if output_path is not None else root / OUTPUT_PATH
    checks, source_hashes, events, protocol = collect_preconditions(root, output)
    artifact = _empty_artifact(run_date, source_hashes)
    artifact["preconditions_checked"] = checks
    artifact["cited_upstream_artifacts"] = _citations(source_hashes)
    if any(row["passed"] is not True for row in checks):
        artifact["duration_s"] = round(time.perf_counter() - started, 6)
        artifact["gate_check_summary"] = _gate_summary(checks)
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return artifact

    assert events is not None and protocol is not None
    sidecar = load_sidecar(root / SIDECAR_PATH)
    frozen_split = deepcopy(protocol)
    frozen_split["sidecar_opened_after_freeze"] = True
    artifact["frozen_split"] = frozen_split
    with tempfile.TemporaryDirectory(prefix="carnot-exp7021-") as directory:
        comparison = run_comparison(events, sidecar, protocol, Path(directory))
    for field, value in comparison.items():
        artifact[field] = value
    aggregate_audit = audit_aggregate_recomputation(artifact)
    artifact["aggregate_row_recomputation"] = aggregate_audit
    held_count = len(protocol["held_stream_indices"])
    strict_rows = all(
        row["strict_chronology"] and row["current_outcome_used_by_policy"] is False
        for row in artifact["per_decision_results"]
    )
    complete_checks = [
        _gate("planned_per_decision_rows", held_count * len(ARMS), len(artifact["per_decision_results"])),
        _gate("planned_query_cost_rows", held_count * len(ARMS), len(artifact["query_cost_rows"])),
        _gate("planned_retention_rows", held_count * len(ARMS), len(artifact["retention_rows"])),
        _gate(
            "all_held_units_classified",
            held_count,
            len(artifact["unsupported_rows"])
            + len(artifact["no_headroom_rows"])
            + len({row["unit_id"] for row in artifact["action_ranking_rows"]}),
        ),
        _gate(
            "equal_capacity_arms",
            True,
            all(row["same_capacity"] and row["within_capacity"] for row in artifact["capacity_match_rows"]),
        ),
        _gate("strictly_earlier_memory", True, strict_rows),
        _gate("aggregate_rows_recompute", True, aggregate_audit["matches"]),
    ]
    leakage_rows = [
        _gate("protocol_frozen_before_sidecar_open", True, protocol["protocol_frozen_before_sidecar_open"]),
        _gate("sidecar_opened_after_protocol_freeze", True, frozen_split["sidecar_opened_after_freeze"]),
        _gate("current_outcome_used_by_policy", False, any(row["current_outcome_used_by_policy"] for row in artifact["per_decision_results"])),
        _gate("future_fields_in_policy_rows", [], ledger_mod.find_forbidden_paths([
            {
                "candidate_scores": row["candidate_scores"],
                "selected_action_id": row["selected_action_id"],
                "max_evidence_stream_index": row["max_evidence_stream_index"],
            }
            for row in artifact["per_decision_results"]
        ])),
    ]
    artifact["leakage_check_rows"] = leakage_rows
    complete = all(row["passed"] for row in [*complete_checks, *leakage_rows])
    artifact["belief_utility_comparison_complete_score"] = int(complete)
    positive, positive_details = _positive_gate(artifact)
    artifact["belief_future_utility_positive_score"] = int(positive)
    value_check = _gate("belief_future_utility_positive_score", 1, int(positive))
    value_check["condition_details"] = positive_details
    artifact["gate_check_summary"] = _gate_summary([*checks, *complete_checks, *leakage_rows, value_check])
    if positive:
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = "complete_positive_prospective_counterexample_belief_utility"
    elif complete:
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = "complete_null_prospective_belief_utility_not_demonstrated"
    else:
        artifact["verdict_class"] = "partial"
        artifact["honest_verdict"] = "partial_prospective_belief_utility_comparison"
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _without_timing(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_timing(nested)
            for key, nested in value.items()
            if key not in {"duration_s", "query_time_us", "mean_query_time_us", "reproducibility_checksum"}
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return deepcopy(value)


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic fields while excluding measured wall-clock values."""

    return ledger_mod._sha256_json(_without_timing(artifact))


def validate_artifact(artifact: Any) -> list[str]:
    """Return structural, gate, and row-recomputation errors without repair."""

    if not isinstance(artifact, Mapping):
        return ["artifact_object_required"]
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"required_fields_missing:{','.join(missing)}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    for field in ROW_TABLES:
        rows = artifact.get(field)
        if not isinstance(rows, list):
            errors.append(f"{field}_not_list")
        elif any(not isinstance(row, Mapping) or row.get("terminal") is not True for row in rows):
            errors.append(f"nonterminal_row:{field}")
    for field, marker in (
        ("belief_utility_comparison_complete_score", "complete_score"),
        ("belief_future_utility_positive_score", "positive_score"),
    ):
        value = artifact.get(field)
        if type(value) is not int or value not in {0, 1}:
            errors.append(f"{marker}_must_be_bare_integer")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    verdict_class = artifact.get("verdict_class")
    prefixes = {
        "positive": "complete_positive_",
        "circular_positive": "complete_circular_positive_",
        "null": "complete_null_",
        "blocked": "blocked_",
        "disqualified": "disqualified_",
        "partial": "partial_",
    }
    if verdict_class not in prefixes:
        errors.append("verdict_class_invalid")
    elif not str(artifact.get("honest_verdict", "")).startswith(prefixes[verdict_class]):
        errors.append("verdict_prefix_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    gate = artifact.get("gate_check_summary")
    if not isinstance(gate, Mapping) or "failed_check" not in gate:
        errors.append("gate_check_summary_invalid")
    if artifact.get("belief_utility_comparison_complete_score") == 1:
        audit = audit_aggregate_recomputation(artifact)
        if not audit["matches"] or artifact.get("aggregate_row_recomputation") != audit:
            errors.append("aggregate_row_recomputation_mismatch")
        held = artifact.get("frozen_split", {}).get("held_stream_indices", [])
        if len(artifact.get("per_decision_results", [])) != len(held) * len(ARMS):
            errors.append("complete_decision_count_mismatch")
        if any(row.get("passed") is not True for row in artifact.get("leakage_check_rows", [])):
            errors.append("complete_leakage_check_failed")
    if artifact.get("belief_future_utility_positive_score") == 1:
        positive, _details = _positive_gate(artifact)
        if not positive or verdict_class != "positive":
            errors.append("positive_gate_inconsistent")
    return errors


def write_validated_artifact(path: Path | str, artifact: Mapping[str, Any]) -> None:
    """Publish one complete JSON document only after all validations pass."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"Exp7021 artifact validation failed:{errors}")
    payload = json.dumps(artifact, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    ledger_mod._atomic_write(Path(path), payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sealed comparison and write its requested artifact path."""

    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    output = args.output or root / OUTPUT_PATH
    artifact = build_artifact(repo_root=root, output_path=output, run_date=args.date)
    write_validated_artifact(output, artifact)
    print(
        f"wrote {output} belief_utility_comparison_complete_score="
        f"{artifact['belief_utility_comparison_complete_score']} "
        f"belief_future_utility_positive_score={artifact['belief_future_utility_positive_score']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the command wrapper.
    raise SystemExit(main())
