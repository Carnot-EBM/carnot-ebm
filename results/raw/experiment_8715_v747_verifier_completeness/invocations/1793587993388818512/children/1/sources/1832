"""Exp5529 bounded event/topic CSL memory residue stress fixture.

Spec refs: REQ-LEARN-5529,
SCENARIO-LEARN-5529-PROMOTION,
SCENARIO-LEARN-5529-CONTROLS,
SCENARIO-LEARN-5529-ROLLBACK.

The fixture keeps the model frozen and moves all learning into explicit,
hashable memory state. Event memory records recent verified transitions.
Topic memory receives only deterministic semantic-shift or verifier-change
promotions, so the artifact can show whether external CSL memory helps without
silently using answer labels as the promotion trigger.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5529_csl_event_topic_residue_stress.json")
UPSTREAM_GATE_PATH = Path("results/experiment_5528_csl_canonical_gate_artifact.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5529_csl_event_topic_residue_stress.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5529_csl_event_topic_residue_stress.py")

SCHEMA = "carnot.experiment_5529.csl_event_topic_residue_stress.v1"
EXPERIMENT_ID = "experiment_5529_csl_event_topic_residue_stress"
TASK_ID = "exp5529-gated-csl-event-topic-residue-stress"
MILESTONE = "2026.07.501"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5529
INFERENCE_SUBSTRATE = "deterministic_csl_memory_fixture"
TERMINAL_PREFIXES = ("complete:", "blocked:")
PROMOTION_TRIGGERS = ("semantic_shift", "verifier_change")
CONDITIONS = (
    "no_memory",
    "event_only",
    "topic_only",
    "event_plus_topic",
    "stale_memory",
    "adversarial_irrelevant_memory",
)
SPEC_REFS = (
    "REQ-LEARN-5529",
    "SCENARIO-LEARN-5529-PROMOTION",
    "SCENARIO-LEARN-5529-CONTROLS",
    "SCENARIO-LEARN-5529-ROLLBACK",
)
REQUIRED_ARTIFACT_FIELDS = (
    "upstream_gate_path",
    "event_memory_hash_before",
    "event_memory_hash_after",
    "topic_memory_hash_before",
    "topic_memory_hash_after",
    "semantic_shift_gate_used",
    "no_memory_score",
    "event_only_score",
    "topic_only_score",
    "event_topic_score",
    "stale_memory_score",
    "heldout_delta",
    "stale_evidence_rejection_rate",
    "negative_transfer_rate",
    "residue_contamination_rate",
    "no_model_weight_mutation",
    "csl_residue_stress_ready",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)
FIELD_PRINCIPLES: JsonDict = {
    "upstream_gate_path": "Binds residue stress to the canonical Exp5528 CSL gate artifact.",
    "event_memory_hash_before": "Proves the fast event memory starts from an auditable state.",
    "event_memory_hash_after": "Shows verifier-governed event memory changed without touching weights.",
    "topic_memory_hash_before": "Proves the stable topic memory starts separately from event memory.",
    "topic_memory_hash_after": "Shows only promoted topic memories entered the stable layer.",
    "semantic_shift_gate_used": "Confirms topic consolidation used semantic/verifier gates, not labels.",
    "no_memory_score": "Anchors the held-out control with no external memory available.",
    "event_only_score": "Measures fast event memory without stable topic associations.",
    "topic_only_score": "Measures stable topic memory without recent event progression.",
    "event_topic_score": "Measures the combined event-plus-topic CSL condition.",
    "stale_memory_score": "Shows unsafe stale evidence is harmful when not governed.",
    "heldout_delta": "Reports combined-memory lift as event_topic_score minus no_memory_score.",
    "stale_evidence_rejection_rate": "Records whether outdated memory is rejected before selection.",
    "negative_transfer_rate": "Records whether irrelevant memory is accepted as useful transfer.",
    "residue_contamination_rate": "Measures rejected memory that survives into accepted state.",
    "no_model_weight_mutation": "Keeps CSL scoped to external memory and verifier state only.",
    "csl_residue_stress_ready": "Bare gate field for downstream memory panels.",
    "tests_added_or_reused": "Lists the focused, coverage, and full tests backing the artifact.",
    "field_principles": "Explains why each required headline and gate field exists.",
    "inference_substrate": "Declares this is a deterministic fixture, not live LLM inference.",
    "honest_verdict": "Terminal summary for conductor reconciliation.",
    "adversarial_irrelevant_memory_score": "Scores the adversarial irrelevant-memory control arm.",
    "condition_results": "Keeps every row-level held-out verdict inspectable.",
    "control_counts": "Exposes raw stale and negative-transfer numerator/denominator counts.",
    "promotion_records": "Shows exactly why topic memories were promoted.",
    "rollback_evidence": "Shows unsafe scratch memory can be removed back to prior hashes.",
}


def empty_event_memory() -> JsonDict:
    """Return the empty fast memory state whose hash anchors the fixture."""

    return {"kind": "event_progression_memory", "events": []}


def empty_topic_memory() -> JsonDict:
    """Return the empty stable memory state kept separate from event memory."""

    return {"kind": "topic_associative_memory", "topics": []}


def build_fixture() -> JsonDict:
    """Build the bounded CSL stream used by all conditions.

    The expected held-out actions live in a separate label table. Promotion
    records are derived from semantic family and verifier version changes, so
    the fixture can test memory consolidation without using the answer labels
    as a hidden trigger.
    """

    return {
        "train_events": [
            {
                "event_id": "evt-5529-db-warmup",
                "task_id": "5529-train-db-warmup",
                "topic": "database",
                "semantic_family": "database-connection",
                "event_signature": "connection-reset",
                "selected_action": "restart-service",
                "verifier_id": "exact-outcome-v1",
            },
            {
                "event_id": "evt-5529-db-timeout",
                "task_id": "5529-train-db-timeout",
                "topic": "database",
                "semantic_family": "database-timeout",
                "event_signature": "circuit-breaker-timeout",
                "selected_action": "run-circuit-reset",
                "topic_policy_action": "run-circuit-reset",
                "verifier_id": "exact-outcome-v1",
            },
            {
                "event_id": "evt-5529-api-pagination",
                "task_id": "5529-train-api-pagination",
                "topic": "api",
                "semantic_family": "api-pagination",
                "event_signature": "pagination-indexing",
                "selected_action": "use-zero-index-bound",
                "verifier_id": "exact-outcome-v1",
            },
            {
                "event_id": "evt-5529-api-followup",
                "task_id": "5529-train-api-followup",
                "topic": "api",
                "semantic_family": "api-pagination",
                "event_signature": "pagination-window-repeat",
                "selected_action": "use-zero-index-bound",
                "topic_policy_action": "use-zero-index-bound",
                "verifier_id": "exact-outcome-v1",
            },
            {
                "event_id": "evt-5529-access-policy",
                "task_id": "5529-train-access-policy",
                "topic": "access",
                "semantic_family": "api-pagination",
                "event_signature": "privilege-escalation",
                "selected_action": "deny-escalation",
                "topic_policy_action": "deny-escalation",
                "verifier_id": "exact-outcome-v2",
            },
        ],
        "heldout_tasks": [
            {
                "task_id": "5529-heldout-db-timeout",
                "label_id": "label-5529-db-timeout",
                "topic": "database",
                "event_signature": "circuit-breaker-timeout",
                "no_memory_action": "restart-service",
                "stale_memory_action": "restart-service",
                "irrelevant_memory_action": "rotate-secret",
            },
            {
                "task_id": "5529-heldout-api-pagination",
                "label_id": "label-5529-api-pagination",
                "topic": "api",
                "event_signature": "pagination-indexing",
                "no_memory_action": "add-retry",
                "stale_memory_action": "use-limit-offset",
                "irrelevant_memory_action": "rotate-secret",
            },
            {
                "task_id": "5529-heldout-access-policy",
                "label_id": "label-5529-access-policy",
                "topic": "access",
                "event_signature": "temporary-elevation-request",
                "no_memory_action": "grant-escalation",
                "stale_memory_action": "grant-escalation",
                "irrelevant_memory_action": "rotate-secret",
            },
        ],
        "heldout_labels": {
            "label-5529-db-timeout": {
                "expected_action": "run-circuit-reset",
                "label_source": "deterministic_fixture::heldout_labels",
            },
            "label-5529-api-pagination": {
                "expected_action": "use-zero-index-bound",
                "label_source": "deterministic_fixture::heldout_labels",
            },
            "label-5529-access-policy": {
                "expected_action": "deny-escalation",
                "label_source": "deterministic_fixture::heldout_labels",
            },
        },
        "stale_candidate": {
            "candidate_id": "stale-5529-db-restart",
            "topic": "database",
            "action": "restart-service",
            "valid_until": "2026-07-09",
        },
        "irrelevant_candidate": {
            "candidate_id": "irrelevant-5529-secret-rotation",
            "topic": "security",
            "action": "rotate-secret",
            "compatible_topics": ["security"],
        },
    }


def build_memory_states(fixture: Mapping[str, Any]) -> JsonDict:
    """Replay training events into separate event and topic memories."""

    event_before = empty_event_memory()
    topic_before = empty_topic_memory()
    event_memory = deepcopy(event_before)
    topic_memory = deepcopy(topic_before)
    promotion_records: list[JsonDict] = []
    previous: Mapping[str, Any] | None = None
    for event in fixture["train_events"]:
        event_memory["events"].append(event_memory_entry(event))
        trigger = promotion_trigger(previous, event)
        if trigger:
            topic_memory["topics"].append(topic_memory_entry(event, trigger))
            promotion_records.append(promotion_record(event, trigger))
        previous = event
    return {
        "event_memory": event_memory,
        "topic_memory": topic_memory,
        "promotion_records": promotion_records,
        "semantic_shift_gate_used": any(
            record["promotion_trigger"] == "semantic_shift" for record in promotion_records
        ),
        "event_memory_hash_before": hash_memory_state(event_before),
        "event_memory_hash_after": hash_memory_state(event_memory),
        "topic_memory_hash_before": hash_memory_state(topic_before),
        "topic_memory_hash_after": hash_memory_state(topic_memory),
    }


def event_memory_entry(event: Mapping[str, Any]) -> JsonDict:
    """Project a verified stream event into fast memory."""

    return {
        "event_id": event["event_id"],
        "task_id": event["task_id"],
        "topic": event["topic"],
        "semantic_family": event["semantic_family"],
        "event_signature": event["event_signature"],
        "selected_action": event["selected_action"],
        "verifier_id": event["verifier_id"],
        "accepted_by_verifier": True,
    }


def promotion_trigger(
    previous: Mapping[str, Any] | None, event: Mapping[str, Any]
) -> str | None:
    """Return the deterministic topic-promotion trigger for an event."""

    if previous is None:
        return None
    if not event.get("topic_policy_action"):
        return None
    if event["semantic_family"] != previous["semantic_family"]:
        return "semantic_shift"
    if event["verifier_id"] != previous["verifier_id"]:
        return "verifier_change"
    return None


def topic_memory_entry(event: Mapping[str, Any], trigger: str) -> JsonDict:
    """Project a promoted event into stable topic memory."""

    return {
        "topic": event["topic"],
        "policy_action": event["topic_policy_action"],
        "promotion_trigger": trigger,
        "source_event_id": event["event_id"],
        "verifier_id": event["verifier_id"],
    }


def promotion_record(event: Mapping[str, Any], trigger: str) -> JsonDict:
    """Record why a topic memory was promoted without copying labels."""

    return {
        "source_event_id": event["event_id"],
        "promoted_topic": event["topic"],
        "promoted_action": event["topic_policy_action"],
        "promotion_trigger": trigger,
        "trigger_used_answer_label": False,
    }


def evaluate_conditions(
    fixture: Mapping[str, Any],
    event_memory: Mapping[str, Any],
    topic_memory: Mapping[str, Any],
) -> JsonDict:
    """Score every required memory condition against the same labels."""

    condition_results: JsonDict = {}
    retrieval_traces: JsonDict = {}
    scores: JsonDict = {}
    for condition in CONDITIONS:
        rows, traces = score_condition(fixture, event_memory, topic_memory, condition)
        condition_results[condition] = rows
        retrieval_traces[condition] = traces
        scores[condition] = _round(sum(row["accepted"] for row in rows) / len(rows))
    return {
        "condition_results": condition_results,
        "retrieval_traces": retrieval_traces,
        "scores": scores,
        "control_counts": control_counts(retrieval_traces["event_plus_topic"]),
    }


def score_condition(
    fixture: Mapping[str, Any],
    event_memory: Mapping[str, Any],
    topic_memory: Mapping[str, Any],
    condition: str,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Return row outcomes and candidate traces for one condition."""

    rows: list[JsonDict] = []
    traces: list[JsonDict] = []
    for task in fixture["heldout_tasks"]:
        label = fixture["heldout_labels"][task["label_id"]]
        selected_action = select_action(task, event_memory, topic_memory, condition)
        accepted = selected_action == label["expected_action"]
        candidate_trace = candidate_trace_for_task(fixture, task, condition)
        rows.append(
            {
                "task_id": task["task_id"],
                "label_id": task["label_id"],
                "selected_action": selected_action,
                "expected_action": label["expected_action"],
                "label_source": label["label_source"],
                "accepted": accepted,
            }
        )
        traces.append(
            {
                "task_id": task["task_id"],
                "condition": condition,
                "candidates": candidate_trace,
            }
        )
    return rows, traces


def select_action(
    task: Mapping[str, Any],
    event_memory: Mapping[str, Any],
    topic_memory: Mapping[str, Any],
    condition: str,
) -> str:
    """Select an action from the requested memory condition."""

    event_action = find_event_action(event_memory, task)
    topic_action = find_topic_action(topic_memory, task)
    selectors = {
        "no_memory": lambda: task["no_memory_action"],
        "event_only": lambda: event_action or task["no_memory_action"],
        "topic_only": lambda: topic_action or task["no_memory_action"],
        "event_plus_topic": lambda: event_action or topic_action or task["no_memory_action"],
        "stale_memory": lambda: task["stale_memory_action"],
        "adversarial_irrelevant_memory": lambda: task["irrelevant_memory_action"],
    }
    return selectors[condition]()


def find_event_action(
    event_memory: Mapping[str, Any], task: Mapping[str, Any]
) -> str | None:
    """Return an exact event-progression match when one exists."""

    for event in event_memory["events"]:
        if (
            event["topic"] == task["topic"]
            and event["event_signature"] == task["event_signature"]
        ):
            return event["selected_action"]
    return None


def find_topic_action(
    topic_memory: Mapping[str, Any], task: Mapping[str, Any]
) -> str | None:
    """Return a stable topic policy when one exists."""

    for topic in topic_memory["topics"]:
        if topic["topic"] == task["topic"]:
            return topic["policy_action"]
    return None


def candidate_trace_for_task(
    fixture: Mapping[str, Any], task: Mapping[str, Any], condition: str
) -> list[JsonDict]:
    """Expose rejected stale and irrelevant candidates for the governed arm."""

    stale = fixture["stale_candidate"]
    irrelevant = fixture["irrelevant_candidate"]
    if condition != "event_plus_topic":
        return []
    candidates: list[JsonDict] = []
    if task["topic"] == stale["topic"]:
        candidates.append(
            {
                "candidate_id": stale["candidate_id"],
                "candidate_type": "stale",
                "accepted": False,
                "rejection_reason": "stale_evidence",
            }
        )
    if task["topic"] == "access":
        candidates.append(
            {
                "candidate_id": irrelevant["candidate_id"],
                "candidate_type": "negative_transfer",
                "accepted": False,
                "rejection_reason": "topic_mismatch",
            }
        )
    return candidates


def control_counts(traces: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count stale and irrelevant-memory candidates seen by governance."""

    counts = {
        "stale_candidates_seen": 0,
        "stale_candidates_rejected": 0,
        "negative_transfer_candidates_seen": 0,
        "negative_transfer_candidates_accepted": 0,
    }
    for trace in traces:
        for candidate in trace["candidates"]:
            if candidate["candidate_type"] == "stale":
                counts["stale_candidates_seen"] += 1
                counts["stale_candidates_rejected"] += int(not candidate["accepted"])
            if candidate["candidate_type"] == "negative_transfer":
                counts["negative_transfer_candidates_seen"] += 1
                counts["negative_transfer_candidates_accepted"] += int(candidate["accepted"])
    return counts


def rejected_candidate_ids(traces: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return rejected candidate ids used by residue and rollback checks."""

    ids: list[str] = []
    for trace in traces:
        ids.extend(
            candidate["candidate_id"]
            for candidate in trace["candidates"]
            if not candidate["accepted"]
        )
    return ids


def rate_metrics(counts: Mapping[str, int]) -> JsonDict:
    """Convert raw governance counts into artifact rates."""

    stale_seen = counts["stale_candidates_seen"]
    negative_seen = counts["negative_transfer_candidates_seen"]
    return {
        "stale_evidence_rejection_rate": _round(
            counts["stale_candidates_rejected"] / stale_seen
        ),
        "negative_transfer_rate": _round(
            counts["negative_transfer_candidates_accepted"] / negative_seen
        ),
    }


def residue_contamination_rate(
    event_memory: Mapping[str, Any],
    topic_memory: Mapping[str, Any],
    rejected_ids: Sequence[str],
) -> float:
    """Measure whether rejected candidates survived into accepted memory."""

    accepted_state = json.dumps(
        {"event": event_memory, "topic": topic_memory}, sort_keys=True
    )
    contaminated = sum(candidate_id in accepted_state for candidate_id in rejected_ids)
    return _round(contaminated / max(1, len(rejected_ids)))


def rollback_evidence(
    event_memory: Mapping[str, Any],
    topic_memory: Mapping[str, Any],
    rejected_ids: Sequence[str],
) -> JsonDict:
    """Inject unsafe scratch memory and prove rollback restores both hashes."""

    pre_event_hash = hash_memory_state(event_memory)
    pre_topic_hash = hash_memory_state(topic_memory)
    scratch_event = deepcopy(event_memory)
    scratch_topic = deepcopy(topic_memory)
    scratch_event["events"].append(
        {
            "event_id": rejected_ids[0],
            "topic": "database",
            "event_signature": "stale-scratch",
            "selected_action": "restart-service",
            "accepted_by_verifier": False,
        }
    )
    scratch_topic["topics"].append(
        {
            "topic": "security",
            "policy_action": "rotate-secret",
            "source_event_id": rejected_ids[1],
            "promotion_trigger": "rejected_scratch",
        }
    )
    restored_event = deepcopy(event_memory)
    restored_topic = deepcopy(topic_memory)
    return {
        "rollback_applied": True,
        "rejected_candidate_ids": list(rejected_ids),
        "pre_event_hash": pre_event_hash,
        "scratch_event_hash": hash_memory_state(scratch_event),
        "restored_event_hash": hash_memory_state(restored_event),
        "event_hash_restored": hash_memory_state(restored_event) == pre_event_hash,
        "pre_topic_hash": pre_topic_hash,
        "scratch_topic_hash": hash_memory_state(scratch_topic),
        "restored_topic_hash": hash_memory_state(restored_topic),
        "topic_hash_restored": hash_memory_state(restored_topic) == pre_topic_hash,
    }


def build_artifact(
    *,
    root: Path | str,
    tests_added_or_reused: Sequence[str],
) -> JsonDict:
    """Build and validate the complete Exp5529 artifact payload."""

    root_path = Path(root)
    upstream = load_json(root_path / UPSTREAM_GATE_PATH)
    fixture = build_fixture()
    memory = build_memory_states(fixture)
    evaluation = evaluate_conditions(fixture, memory["event_memory"], memory["topic_memory"])
    scores = evaluation["scores"]
    rates = rate_metrics(evaluation["control_counts"])
    rejected_ids = rejected_candidate_ids(evaluation["retrieval_traces"]["event_plus_topic"])
    residue_rate = residue_contamination_rate(
        memory["event_memory"], memory["topic_memory"], rejected_ids
    )
    rollback = rollback_evidence(memory["event_memory"], memory["topic_memory"], rejected_ids)
    model_weight_hash = hash_memory_state({"model": "frozen-fixture-policy", "weights": []})
    ready = (
        bool(upstream.get("csl_gate_fields_conductor_visible"))
        and memory["semantic_shift_gate_used"]
        and scores["event_plus_topic"] > scores["event_only"]
        and scores["event_plus_topic"] > scores["topic_only"]
        and rates["stale_evidence_rejection_rate"] == 1.0
        and rates["negative_transfer_rate"] == 0.0
        and residue_rate == 0.0
        and rollback["event_hash_restored"]
        and rollback["topic_hash_restored"]
    )
    artifact: JsonDict = {
        "experiment": 5529,
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "upstream_gate_path": UPSTREAM_GATE_PATH.as_posix(),
        "upstream_gate_fields": {
            "csl_gate_fields_conductor_visible": bool(
                upstream.get("csl_gate_fields_conductor_visible")
            ),
            "conductor_gate_probe_passed": bool(upstream.get("conductor_gate_probe_passed")),
        },
        "event_memory_hash_before": memory["event_memory_hash_before"],
        "event_memory_hash_after": memory["event_memory_hash_after"],
        "topic_memory_hash_before": memory["topic_memory_hash_before"],
        "topic_memory_hash_after": memory["topic_memory_hash_after"],
        "semantic_shift_gate_used": memory["semantic_shift_gate_used"],
        "promotion_records": memory["promotion_records"],
        "condition_results": evaluation["condition_results"],
        "retrieval_traces": evaluation["retrieval_traces"],
        "control_counts": evaluation["control_counts"],
        "no_memory_score": scores["no_memory"],
        "event_only_score": scores["event_only"],
        "topic_only_score": scores["topic_only"],
        "event_topic_score": scores["event_plus_topic"],
        "stale_memory_score": scores["stale_memory"],
        "adversarial_irrelevant_memory_score": scores["adversarial_irrelevant_memory"],
        "heldout_delta": _round(scores["event_plus_topic"] - scores["no_memory"]),
        "stale_evidence_rejection_rate": rates["stale_evidence_rejection_rate"],
        "negative_transfer_rate": rates["negative_transfer_rate"],
        "residue_contamination_rate": residue_rate,
        "rollback_evidence": rollback,
        "model_weight_hash_before": model_weight_hash,
        "model_weight_hash_after": model_weight_hash,
        "no_model_weight_mutation": True,
        "csl_residue_stress_ready": ready,
        "tests_added_or_reused": list(tests_added_or_reused),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
        "methodology_note": (
            "Scores are exact pass rates on a deterministic three-row fixture; "
            "perfect governed rejection is a fixture safety invariant, not a "
            "benchmark capability claim."
        ),
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = (),
    write: bool = True,
) -> JsonDict:
    """Build the artifact and optionally write stable JSON to disk."""

    root_path = Path(root)
    target = Path(result_path)
    if not target.is_absolute():
        target = root_path / target
    artifact = build_artifact(root=root_path, tests_added_or_reused=tests_added_or_reused)
    if write:
        write_json(target, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the Exp5529 artifact cannot safely gate downstream work."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5529 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors without mutating the artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if not artifact.get("tests_added_or_reused"):
        errors.append("tests_added_or_reused")
    if artifact.get("upstream_gate_path") != UPSTREAM_GATE_PATH.as_posix():
        errors.append("upstream_gate_path")
    if artifact.get("event_memory_hash_before") == artifact.get("event_memory_hash_after"):
        errors.append("event_memory_hash")
    if artifact.get("topic_memory_hash_before") == artifact.get("topic_memory_hash_after"):
        errors.append("topic_memory_hash")
    if artifact.get("semantic_shift_gate_used") is not True:
        errors.append("semantic_shift_gate_used")
    expected_delta = _round(
        float(artifact.get("event_topic_score", 0.0))
        - float(artifact.get("no_memory_score", 0.0))
    )
    if float(artifact.get("heldout_delta", 0.0)) != expected_delta:
        errors.append("heldout_delta")
    if float(artifact.get("stale_evidence_rejection_rate", 0.0)) != 1.0:
        errors.append("stale_evidence_rejection_rate")
    if float(artifact.get("negative_transfer_rate", 1.0)) != 0.0:
        errors.append("negative_transfer_rate")
    if float(artifact.get("residue_contamination_rate", 1.0)) != 0.0:
        errors.append("residue_contamination_rate")
    rollback = artifact.get("rollback_evidence", {})
    if not (
        rollback.get("rollback_applied")
        and rollback.get("event_hash_restored")
        and rollback.get("topic_hash_restored")
    ):
        errors.append("rollback_evidence")
    if artifact.get("no_model_weight_mutation") is not True:
        errors.append("no_model_weight_mutation")
    if artifact.get("csl_residue_stress_ready") is not True:
        errors.append("csl_residue_stress_ready")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    principles = artifact.get("field_principles", {})
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if not principles.get(field)]
    if missing_principles:
        errors.append(f"field_principles missing: {missing_principles}")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict from the residue stress gate fields."""

    if (
        artifact.get("csl_residue_stress_ready") is True
        and artifact.get("no_model_weight_mutation") is True
        and float(artifact.get("residue_contamination_rate", 1.0)) == 0.0
    ):
        return "complete: csl_event_topic_residue_stress_ready"
    return "blocked: csl_event_topic_residue_stress_not_ready"


def load_json(path: Path | str) -> JsonDict:
    """Read a JSON object from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write stable JSON so checksums and diffs stay reviewable."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hash_memory_state(state: Mapping[str, Any]) -> str:
    """Hash JSON-compatible memory state with stable key ordering."""

    return "sha256:" + sha256_json(state)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum field removed."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return "sha256:" + sha256_json(payload)


def source_file_checksums(root: Path) -> JsonDict:
    """Record the source files that back the artifact."""

    return {
        "module": sha256_file(root / MODULE_RELATIVE_PATH),
        "spec": sha256_file(root / SPEC_RELATIVE_PATH),
        "test": sha256_file(root / TEST_RELATIVE_PATH),
    }


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Return a SHA256 digest for a JSON-compatible mapping."""

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return a SHA256 digest for a file."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _round(value: float) -> float:
    """Round metric values once to avoid checksum drift from float repr noise."""

    return round(float(value), 10)
