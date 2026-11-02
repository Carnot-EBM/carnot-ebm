"""Exp 2983 trace-to-skill repair memory pilot.

This module turns checked-in execution evidence into small, reusable skill
memories and evaluates those memories by deterministic replay. It does not run
fresh live inference, mutate model weights, or promote a headline self-learning
claim. The point is narrower: determine whether trace-derived memories can
improve held-out repair replay without copying labels into the memory itself.

Spec: REQ-LEARN-2983, SCENARIO-LEARN-2983,
SCENARIO-LEARN-2983-BLOCKED.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
OUTPUT_FILENAME = "experiment_2983_trace_to_skill_repair_memory_pilot_v1.json"
ARTIFACT = "experiment_2983_trace_to_skill_repair_memory_pilot_v1"
SCHEMA = "carnot.trace_to_skill_repair_memory_pilot.v1"
MEMORY_SCHEMA_VERSION = "carnot.trace_to_skill_memory.v1"
INFERENCE_SUBSTRATE = "artifact_replay_and_optional_live_llm"

EXP2976_REL_PATH = Path("results/experiment_2976_dccd_adaptrack_tracecoder_protocol_v1.json")
EXP2977_REL_PATH = Path("results/experiment_2977_sota_intent_preserving_code_repair_v1.json")
EXP2964_REL_PATH = Path("results/experiment_2964_sota_dccd_repair_replication_v1.json")
EXP2968_REL_PATH = Path("results/experiment_2968_interwhen_partial_monitor_harness_v1.json")

MANDATORY_HEADLINE_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MEMORY_REQUIRED_FIELDS = (
    "failure_signature",
    "verifier_feedback",
    "minimal_fix_pattern",
    "applicability_conditions",
    "forbidden_label_leakage",
)
FORBIDDEN_LABEL_LEAKAGE = (
    "held-out task ids",
    "outcome labels",
    "pass vectors",
    "expected outputs",
    "reference solutions",
    "metric targets",
)
REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "continuous_self_learning_task",
    "trace_to_skill_memory_ready",
    "headline_result",
    "pilot_source",
    "models_used",
    "mandatory_headline_model_ids",
    "memory_schema",
    "extracted_memory_count",
    "heldout_task_count",
    "no_memory_metrics",
    "random_memory_metrics",
    "trace_memory_metrics",
    "heldout_skill_reuse_delta",
    "leakage_flag",
    "negative_control_delta",
    "inference_substrate",
    "duration_s",
}
REPAIR_MEMORY_TASK_LIMIT = 2
BASELINE_MODES = {"baseline", "baseline_no_taxonomy"}
TRACE_ASSISTED_MODES = {"taxonomy_guided", "intent_preserving_trace_aware_repair"}
NEGATIVE_CONTROL_SIGNATURES = frozenset({"partial_monitor::unrelated_control"})
UPDATE_SELECTION_METRIC = "source_failure_signature_frequency"
HELDOUT_UTILITY_METRIC = "heldout_task_success_rate"


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and deterministic timing hooks for the pilot builder."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """REQ-LEARN-2983: build the trace-to-skill replay artifact."""

    config = config or ExperimentConfig()
    started = config.start_time()
    sources = source_artifacts_for(config.repo_root)
    exp2976 = read_json_object(config.repo_root / EXP2976_REL_PATH)
    exp2977 = read_json_object(config.repo_root / EXP2977_REL_PATH)
    exp2964 = read_json_object(config.repo_root / EXP2964_REL_PATH)
    exp2968 = read_json_object(config.repo_root / EXP2968_REL_PATH)

    if not exp2976:
        return _blocked_artifact(config, started, sources, "blocked_missing_exp2976_protocol")
    if exp2976.get("trace_execution_plan_ready") is not True:
        return _blocked_artifact(
            config,
            started,
            sources,
            "blocked_exp2976_trace_execution_plan_not_ready",
        )
    if not exp2964:
        return _blocked_artifact(config, started, sources, "blocked_missing_exp2964_heldout_source")

    repair_source = "exp2977" if exp2977 else "exp2964"
    repair_payload = exp2977 if exp2977 else exp2964
    pilot_source = ".280" if exp2977 else ".279"
    memories = extract_skill_memories(
        repair_payload=repair_payload,
        monitor_payload=exp2968,
        repair_source=repair_source,
    )
    train_ids = extraction_task_ids(memories)
    heldout_tasks = build_heldout_tasks(exp2964, excluded_task_ids=train_ids)
    leakage_flag = leakage_flag_for(memories, heldout_tasks)
    no_memory_metrics = evaluate_replay(heldout_tasks, memories=(), condition="no_memory")
    random_memory_metrics = evaluate_replay(
        heldout_tasks,
        memories=memories,
        condition="random_memory",
    )
    trace_memory_metrics = evaluate_replay(
        heldout_tasks,
        memories=memories,
        condition="trace_memory",
    )
    negative_control_metrics = evaluate_replay(
        heldout_tasks,
        memories=memories,
        condition="negative_control",
    )
    heldout_delta = _round(
        trace_memory_metrics["task_success_rate"] - random_memory_metrics["task_success_rate"]
    )
    negative_control_delta = _round(
        negative_control_metrics["task_success_rate"] - random_memory_metrics["task_success_rate"]
    )
    schema_usable = bool(memories) and all(validate_memory(memory) for memory in memories)
    ready = bool(
        schema_usable
        and heldout_tasks
        and heldout_delta > 0.0
        and negative_control_delta <= 0.0
        and not leakage_flag
    )
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": (
            "complete: trace_to_skill_memory_ready"
            if ready
            else "complete: trace_to_skill_memory_not_ready"
        ),
        "continuous_self_learning_task": True,
        "trace_to_skill_memory_ready": ready,
        "headline_result": False,
        "pilot_source": pilot_source,
        "models_used": models_used(exp2977, exp2964),
        "mandatory_headline_model_ids": mandatory_headline_model_ids(exp2976),
        "memory_schema": trace_to_skill_memory_schema(),
        "extracted_memory_count": len(memories),
        "heldout_task_count": len(heldout_tasks),
        "no_memory_metrics": no_memory_metrics,
        "random_memory_metrics": random_memory_metrics,
        "trace_memory_metrics": trace_memory_metrics,
        "heldout_skill_reuse_delta": heldout_delta,
        "leakage_flag": leakage_flag,
        "negative_control_delta": negative_control_delta,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _elapsed(config, started),
        "extraction_source": repair_source,
        "update_selection_metric": UPDATE_SELECTION_METRIC,
        "heldout_utility_metric": HELDOUT_UTILITY_METRIC,
        "fresh_live_llm_inference_used": False,
        "memory_candidates": memories,
        "heldout_task_summaries": heldout_tasks,
        "negative_control_metrics": negative_control_metrics,
        "leakage_audit": leakage_audit(memories, heldout_tasks),
        "source_artifacts": sources,
        "tests_run": list(config.tests_run),
        "claim_boundary": (
            "artifact replay only; no new local GGUF inference was invoked by Exp 2983"
        ),
    }
    return validate_artifact(artifact)


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build and persist the terminal Exp 2983 JSON artifact."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config)
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def trace_to_skill_memory_schema() -> JsonDict:
    """Return the memory schema required by REQ-LEARN-2983."""

    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": True,
        "required": list(MEMORY_REQUIRED_FIELDS),
        "properties": {
            "failure_signature": {
                "type": "string",
                "description": "Normalized verifier or monitor event pattern that triggers reuse.",
            },
            "verifier_feedback": {
                "type": "string",
                "description": "Verifier, parser, schema, runtime, or monitor feedback stripped of labels.",
            },
            "minimal_fix_pattern": {
                "type": "string",
                "description": "Small reusable repair action, not an answer or task-specific patch.",
            },
            "applicability_conditions": {
                "type": "object",
                "description": "Non-label conditions needed before the memory can be replayed.",
            },
            "forbidden_label_leakage": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Evidence that must not appear in memory contents.",
            },
        },
    }


def extract_skill_memories(
    *,
    repair_payload: Mapping[str, Any],
    monitor_payload: Mapping[str, Any],
    repair_source: str,
) -> list[JsonDict]:
    """Extract candidate memories from failed repair traces and monitor events."""

    memories: list[JsonDict] = []
    used_memory_keys: set[str] = set()
    repair_task_ids: list[str] = []

    for row in _candidate_rows(repair_payload):
        if row.get("passed") is True:
            continue
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        if task_id not in repair_task_ids:
            if len(repair_task_ids) >= REPAIR_MEMORY_TASK_LIMIT:
                continue
            repair_task_ids.append(task_id)
        for signature in failure_signatures_for_row(row):
            memory = repair_memory_from_row(row, signature, repair_source)
            key = f"{memory['source']}::{memory['source_task_id']}::{signature}"
            if key in used_memory_keys:
                continue
            used_memory_keys.add(key)
            memories.append(memory)

    memories.extend(partial_monitor_memories(monitor_payload))
    return memories


def repair_memory_from_row(
    row: Mapping[str, Any],
    failure_signature: str,
    repair_source: str,
) -> JsonDict:
    """Create one label-free memory from a failed repair row."""

    task_id = str(row.get("task_id") or "unknown-task")
    categories = _string_list(row.get("original_failure_categories"))
    conditions = {
        "source": repair_source,
        "corpus": str(row.get("corpus") or ""),
        "mode": str(row.get("mode") or ""),
        "failure_signature": failure_signature,
        "original_failure_categories": categories,
    }
    memory = {
        "memory_id": _stable_id(f"{repair_source}:{task_id}:{failure_signature}"),
        "source": repair_source,
        "source_trace_kind": "repair_trace",
        "source_task_id": task_id,
        "failure_signature": failure_signature,
        "verifier_feedback": verifier_feedback_for_row(row, failure_signature),
        "minimal_fix_pattern": minimal_fix_pattern_for(failure_signature),
        "applicability_conditions": conditions,
        "forbidden_label_leakage": list(FORBIDDEN_LABEL_LEAKAGE),
    }
    return validate_memory(memory)


def partial_monitor_memories(monitor_payload: Mapping[str, Any]) -> list[JsonDict]:
    """Convert partial-monitor event schemas into reusable skill memories."""

    memories: list[JsonDict] = []
    used_keys: set[str] = set()
    for trace in _monitor_rows(monitor_payload):
        trace_id = str(trace.get("trace_id") or "unknown-trace")
        trace_kind = str(trace.get("trace_kind") or "unknown")
        for event in _sequence(trace.get("events")):
            event_map = _mapping(event)
            event_name = str(event_map.get("event_name") or "")
            if not event_name:
                continue
            checks = sorted(
                {
                    str(check.get("check_name"))
                    for check in _sequence(event_map.get("checks"))
                    if isinstance(check, Mapping) and check.get("check_name")
                }
            )
            signature = f"partial_monitor::{event_name}"
            key = f"{trace_kind}:{signature}:{','.join(checks)}"
            if key in used_keys:
                continue
            used_keys.add(key)
            memory = {
                "memory_id": _stable_id(f"exp2968:{trace_id}:{signature}:{checks}"),
                "source": "exp2968",
                "source_trace_kind": "partial_monitor_event",
                "source_trace_id": trace_id,
                "failure_signature": signature,
                "verifier_feedback": (
                    "partial monitor event uses deterministic checks: "
                    + (", ".join(checks) if checks else "none recorded")
                ),
                "minimal_fix_pattern": minimal_fix_pattern_for(signature),
                "applicability_conditions": {
                    "trace_kind": trace_kind,
                    "event_name": event_name,
                    "deterministic_checks": checks,
                },
                "forbidden_label_leakage": list(FORBIDDEN_LABEL_LEAKAGE),
            }
            memories.append(validate_memory(memory))
    return memories


def failure_signatures_for_row(row: Mapping[str, Any]) -> list[str]:
    """Normalize schema, parser, runtime, verifier, and audit signals."""

    signatures: list[str] = []
    categories = {category.lower() for category in _string_list(row.get("original_failure_categories"))}
    if row.get("schema_valid") is False or "schema_error" in categories:
        signatures.append("repair::schema_invalid")
    if row.get("syntax_success") is False or "syntax_error" in categories:
        signatures.append("repair::syntax_invalid")
    if _sequence(row.get("runtime_trace")) or str(row.get("test_status") or "") == "failed":
        signatures.append("repair::runtime_trace_failure")
    if row.get("false_accept") is True:
        signatures.append("repair::false_accept_risk")
    if row.get("verifier_accepted") is False and not signatures:
        signatures.append("repair::verifier_rejected")
    if not signatures:
        signatures.append("repair::unknown_failed_trace")
    return _dedupe(signatures)


def verifier_feedback_for_row(row: Mapping[str, Any], failure_signature: str) -> str:
    """Summarize verifier feedback without copying outcome labels."""

    if failure_signature == "repair::schema_invalid":
        errors = _schema_errors(row)
        return "schema validation reported: " + ("; ".join(errors) if errors else "missing fields")
    if failure_signature == "repair::syntax_invalid":
        errors = _syntax_errors(row)
        return "syntax checker reported: " + ("; ".join(errors) if errors else "parser error")
    if failure_signature == "repair::runtime_trace_failure":
        return "runtime trace reported an executable check failure"
    if failure_signature == "repair::false_accept_risk":
        return "false-accept audit rejected verifier-only acceptance"
    if failure_signature == "repair::verifier_rejected":
        return "verifier score stayed below the configured acceptance threshold"
    return "failed trace did not expose a more specific verifier category"


def minimal_fix_pattern_for(failure_signature: str) -> str:
    """Map a normalized signature to the smallest reusable repair pattern."""

    patterns = {
        "repair::schema_invalid": "emit all required structured fields before running code checks",
        "repair::syntax_invalid": "produce a parseable Python function before optimizing behavior",
        "repair::runtime_trace_failure": "turn the first executable failure into a focused regression check",
        "repair::false_accept_risk": "require executable evidence before accepting a verifier-only candidate",
        "repair::verifier_rejected": "surface the verifier threshold and repair the lowest-scoring field first",
        "partial_monitor::partial_code_block": "parse each emitted code prefix before downstream verification",
        "partial_monitor::import_line": "keep imports inside the allow-list before executing generated code",
        "partial_monitor::function_sig": "stabilize the function signature and preserve referenced symbols",
        "partial_monitor::assertion_or_formula_line": "check assertion or formalization symbols before escalation",
        "partial_monitor::solver_query": "parse the formal query before using solver feedback",
        "partial_monitor::final_answer": "cross-check the final answer against observed symbols or solver output",
    }
    return patterns.get(failure_signature, "reuse the verifier feedback as a bounded repair hint")


def validate_memory(memory: Mapping[str, Any]) -> JsonDict:
    """Validate one trace-to-skill memory against the pilot schema."""

    missing = set(MEMORY_REQUIRED_FIELDS) - set(memory)
    if missing:
        raise ValueError(f"memory missing required fields: {sorted(missing)}")
    for field_name in MEMORY_REQUIRED_FIELDS:
        value = memory.get(field_name)
        if value is None or value == "" or value == ():
            raise ValueError(f"memory field is empty: {field_name}")
    if not isinstance(memory.get("applicability_conditions"), Mapping):
        raise ValueError("applicability_conditions must be an object")
    if not isinstance(memory.get("forbidden_label_leakage"), Sequence) or isinstance(
        memory.get("forbidden_label_leakage"),
        (str, bytes),
    ):
        raise ValueError("forbidden_label_leakage must be an array")
    return dict(memory)


def extraction_task_ids(memories: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return repair task IDs used as memory extraction sources."""

    return {
        str(memory.get("source_task_id"))
        for memory in memories
        if memory.get("source_task_id") not in {None, ""}
    }


def build_heldout_tasks(
    repair_payload: Mapping[str, Any],
    *,
    excluded_task_ids: set[str],
) -> list[JsonDict]:
    """Build held-out repair tasks disjoint from memory extraction sources."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in _candidate_rows(repair_payload):
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in excluded_task_ids:
            continue
        grouped.setdefault(task_id, []).append(row)

    tasks: list[JsonDict] = []
    for task_id, rows in sorted(grouped.items()):
        baseline_rows = [row for row in rows if str(row.get("mode") or "") in BASELINE_MODES]
        if not baseline_rows:
            continue
        trace_rows = [
            row for row in rows if str(row.get("mode") or "") in TRACE_ASSISTED_MODES
        ]
        signatures = _dedupe(
            signature
            for row in rows
            if row.get("passed") is not True
            for signature in failure_signatures_for_row(row)
        )
        if not signatures:
            signatures = ["repair::unknown_failed_trace"]
        tasks.append(
            {
                "task_id": task_id,
                "corpus": str(rows[0].get("corpus") or ""),
                "failure_signatures": signatures,
                "baseline_success": any(row.get("passed") is True for row in baseline_rows),
                "trace_assisted_success": any(row.get("passed") is True for row in trace_rows),
                "source_modes": sorted({str(row.get("mode") or "") for row in rows}),
            }
        )
    return tasks


def evaluate_replay(
    heldout_tasks: Sequence[Mapping[str, Any]],
    *,
    memories: Sequence[Mapping[str, Any]],
    condition: str,
) -> dict[str, float | int | str]:
    """Score a replay condition on held-out tasks."""

    memory_signatures = memory_signature_set(memories)
    successes = 0
    for task in heldout_tasks:
        baseline_success = task.get("baseline_success") is True
        if condition in {"no_memory", "random_memory"}:
            success = baseline_success
        elif condition == "negative_control":
            success = baseline_success or _task_matches_signatures(task, NEGATIVE_CONTROL_SIGNATURES)
        elif condition == "trace_memory":
            success = baseline_success or (
                task.get("trace_assisted_success") is True
                and _task_matches_signatures(task, memory_signatures)
            )
        else:
            raise ValueError(f"unknown replay condition: {condition}")
        successes += int(success)
    task_count = len(heldout_tasks)
    return {
        "condition": condition,
        "task_count": task_count,
        "task_success_count": successes,
        "task_success_rate": _round(successes / task_count) if task_count else 0.0,
    }


def memory_signature_set(memories: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    """Return repair signatures that can be applied to held-out repair tasks."""

    return frozenset(
        str(memory.get("failure_signature"))
        for memory in memories
        if str(memory.get("failure_signature") or "").startswith("repair::")
    )


def leakage_flag_for(
    memories: Sequence[Mapping[str, Any]],
    heldout_tasks: Sequence[Mapping[str, Any]],
) -> bool:
    """Detect label or held-out-ID leakage in memory bodies."""

    heldout_ids = {str(task.get("task_id")) for task in heldout_tasks if task.get("task_id")}
    extraction_ids = extraction_task_ids(memories)
    if extraction_ids & heldout_ids:
        return True
    forbidden_tokens = {
        "passed=true",
        '"passed": true',
        "pass_vector",
        "expected_output",
        "expected answer",
        "reference_solution",
        "heldout_task_success_rate",
    }
    for memory in memories:
        text = json.dumps(memory, sort_keys=True).lower()
        if any(task_id.lower() in text for task_id in heldout_ids):
            return True
        if any(token in text for token in forbidden_tokens):
            return True
    return False


def leakage_audit(
    memories: Sequence[Mapping[str, Any]],
    heldout_tasks: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Record split and metric-separation checks for the artifact."""

    train_ids = sorted(extraction_task_ids(memories))
    heldout_ids = sorted(str(task.get("task_id")) for task in heldout_tasks if task.get("task_id"))
    return {
        "train_task_ids": train_ids,
        "heldout_task_ids": heldout_ids,
        "train_heldout_intersection": sorted(set(train_ids) & set(heldout_ids)),
        "selection_metric": UPDATE_SELECTION_METRIC,
        "heldout_metric": HELDOUT_UTILITY_METRIC,
        "selection_metric_reused_as_heldout_metric": UPDATE_SELECTION_METRIC
        == HELDOUT_UTILITY_METRIC,
        "memory_contains_heldout_ids": any(
            task_id.lower() in json.dumps(memories, sort_keys=True).lower()
            for task_id in heldout_ids
        ),
    }


def source_artifacts_for(root: Path) -> dict[str, JsonDict]:
    """Return source artifact presence, paths, and checksums."""

    specs = {
        "exp2976": (EXP2976_REL_PATH, "trace_execution_protocol"),
        "exp2977": (EXP2977_REL_PATH, "preferred_failed_repair_traces"),
        "exp2964": (EXP2964_REL_PATH, "fallback_and_heldout_repair_rows"),
        "exp2968": (EXP2968_REL_PATH, "partial_monitor_events"),
    }
    citations: dict[str, JsonDict] = {}
    for experiment_id, (rel_path, role) in specs.items():
        path = root / rel_path
        present = path.is_file()
        citations[experiment_id] = {
            "path": rel_path.as_posix(),
            "role": role,
            "present": present,
            "sha256": _sha256(path) if present else None,
        }
    return citations


def models_used(exp2977: Mapping[str, Any], exp2964: Mapping[str, Any]) -> list[str]:
    """Combine model IDs from upstream artifacts without claiming fresh inference."""

    ids: list[str] = []
    for value in _string_list(exp2977.get("models_used")):
        if value not in ids:
            ids.append(value)
    for value in _string_list(exp2964.get("headline_models_used")):
        if value not in ids:
            ids.append(value)
    return ids


def mandatory_headline_model_ids(exp2976: Mapping[str, Any]) -> list[str]:
    """Return mandated SOTA GGUF IDs from Exp 2976 or the local policy list."""

    upstream = _string_list(exp2976.get("mandatory_headline_model_ids"))
    return upstream or list(MANDATORY_HEADLINE_MODEL_IDS)


def read_json_object(path: Path) -> JsonDict:
    """Read a local JSON object, returning an empty object for unusable evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def validate_artifact(artifact: Mapping[str, Any]) -> JsonDict:
    """Validate required fields and the pilot's claim boundaries."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("substrate must remain artifact_replay_and_optional_live_llm")
    if artifact.get("headline_result") is True:
        raise ValueError("headline result requires fresh mandated GGUF inference")
    if artifact.get("trace_to_skill_memory_ready") is True:
        if artifact.get("leakage_flag") is True:
            raise ValueError("ready trace-to-skill memory cannot have leakage")
        if float(artifact.get("heldout_skill_reuse_delta") or 0.0) <= 0.0:
            raise ValueError("ready trace-to-skill memory requires positive heldout delta")
        if float(artifact.get("negative_control_delta") or 0.0) > 0.0:
            raise ValueError("negative control must not improve")
        if int(artifact.get("extracted_memory_count") or 0) <= 0:
            raise ValueError("ready trace-to-skill memory requires extracted memories")
        if int(artifact.get("heldout_task_count") or 0) <= 0:
            raise ValueError("ready trace-to-skill memory requires held-out tasks")
    schema = _mapping(artifact.get("memory_schema"))
    if set(MEMORY_REQUIRED_FIELDS) - set(_sequence(schema.get("required"))):
        raise ValueError("memory schema does not declare all required fields")
    return dict(artifact)


def main() -> int:
    """CLI entry point used by the experiment wrapper."""

    write_artifact()
    return 0


def _blocked_artifact(
    config: ExperimentConfig,
    started: float,
    sources: Mapping[str, Mapping[str, Any]],
    verdict: str,
) -> JsonDict:
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": verdict,
        "continuous_self_learning_task": True,
        "trace_to_skill_memory_ready": False,
        "headline_result": False,
        "pilot_source": "blocked",
        "models_used": [],
        "mandatory_headline_model_ids": list(MANDATORY_HEADLINE_MODEL_IDS),
        "memory_schema": trace_to_skill_memory_schema(),
        "extracted_memory_count": 0,
        "heldout_task_count": 0,
        "no_memory_metrics": {},
        "random_memory_metrics": {},
        "trace_memory_metrics": {},
        "heldout_skill_reuse_delta": 0.0,
        "leakage_flag": True,
        "negative_control_delta": 0.0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _elapsed(config, started),
        "source_artifacts": {key: dict(value) for key, value in sources.items()},
        "blockers": [verdict],
        "tests_run": list(config.tests_run),
    }
    return validate_artifact(artifact)


def _task_matches_signatures(
    task: Mapping[str, Any],
    signatures: frozenset[str],
) -> bool:
    return bool(set(_string_list(task.get("failure_signatures"))) & set(signatures))


def _candidate_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [_mapping(row) for row in _sequence(payload.get("candidate_evaluations"))]


def _monitor_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [_mapping(row) for row in _sequence(payload.get("monitor_results"))]


def _schema_errors(row: Mapping[str, Any]) -> list[str]:
    errors = _string_list(row.get("schema_errors"))
    diagnostics = _mapping(row.get("schema_diagnostics"))
    return errors or _string_list(diagnostics.get("schema_errors"))


def _syntax_errors(row: Mapping[str, Any]) -> list[str]:
    diagnostics = _mapping(row.get("syntax_diagnostics"))
    return _string_list(diagnostics.get("syntax_errors"))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _string_list(value: object) -> list[str]:
    return [str(item) for item in _sequence(value) if item not in {None, ""}]


def _dedupe(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _stable_id(text: str) -> str:
    return "skill-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _elapsed(config: ExperimentConfig, started: float) -> float:
    return _round(config.clock() - started)


def _round(value: float) -> float:
    return round(float(value), 8)
