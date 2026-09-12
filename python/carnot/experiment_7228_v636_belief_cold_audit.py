"""Audit packed belief learning from immutable rows and cold state.

The audit trusts producer completion, not producer learning value. It rebuilds
all metrics from event rows. Fresh processes then test state parity, delayed
updates, invalid transactions, rollback, and future-label isolation.

Spec refs: REQ-CL-7228 and SCENARIO-CL-7228-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import re
import socket
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_7199_v634_bounded_acquisition as exp7199
from carnot import experiment_7214_v635_refinement_cold_audit as exp7214
from carnot import experiment_7226_v636_belief_compiler as exp7226
from carnot import experiment_7227_v636_belief_learning as exp7227
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7228
SCHEMA = "carnot.exp7228.v636_belief_cold_audit.v1"
MILESTONE = "2026.09.636"
RUN_DATE = "20260912"
AUDIT_SEED = 7_228_001
STREAM_SEEDS = tuple(exp7227.STREAM_SEEDS)
ARMS = tuple(exp7227.ARMS)
EVENTS_PER_SEED = exp7227.EVENTS_PER_SEED
WARMUP_COUNT = exp7227.WARMUP_COUNT
PROBE_START = 992
PROBE_COUNT = 32
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRODUCER_PATH = Path("results/experiment_7227_v636_belief_learning.json")
COMPILER_ARTIFACT_PATH = Path("results/experiment_7226_v636_belief_compiler.json")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/experiment_7228_v636_belief_cold_audit.json")
DEFAULT_ARTIFACT_PATH = Path("results/experiment_7228_v636_belief_cold_audit.json")
V635_AUDIT_PATH = Path("results/experiment_7214_v635_refinement_cold_audit.json")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
WRAPPER_PATH = REPO_ROOT / "scripts/experiments/experiment_7228_v636_belief_cold_audit.py"
EXPECTED_PRODUCER_SHA256 = "sha256:ada61ddef40c2b1875666da974a319e12c1d0b3b4864efeec6aa4e2a57ff7eb6"
EXPECTED_V635_AUDIT_SHA256 = (
    "sha256:0be789c5f132503042aa814448ecded5c55b8f861273df7e0451afc9424baabd"
)
RESULT_PREFIX = exp7214.RESULT_PREFIX

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7214_v635_refinement_cold_audit.py"),
    Path("python/carnot/experiment_7226_v636_belief_compiler.py"),
    Path("python/carnot/experiment_7227_v636_belief_learning.py"),
    Path("python/carnot/experiment_7228_v636_belief_cold_audit.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("scripts/experiments/experiment_7228_v636_belief_cold_audit.py"),
    Path("tests/python/test_experiment_7228_v636_belief_cold_audit.py"),
    SPEC_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "field_principles",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "preconditions_checked",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "execution_host",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
    "MODEL_SPECS",
    "model_invoked",
    "belief_audit_complete_score",
    "belief_promotion_score",
    "cold_reload_rows",
    "rollback_rows",
    "causal_control_rows",
    "metric_recomputation_rows",
    "comparison_recomputation_rows",
    "deletion_recomputation_rows",
    "source_grounding_rows",
    "runtime_isolation_receipt",
    "checkpoint_receipt",
    "producer_gate_receipt",
    "v635_history_receipt",
    "audit_errors",
    "no_model_weight_mutation",
    "no_fitting_or_parameter_adjustment",
    "certificate_published",
    "default_pipeline_modified",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "A versioned schema makes incompatible readers fail closed.",
    "experiment_id": "A fixed ID prevents another task from supplying this evidence.",
    "milestone": "The milestone binds this result to the V636 contract.",
    "field_principles": (
        "Annotate actual values in this map; do not wrap arbitrary dictionaries as "
        "principle/value records."
    ),
    "status": (
        "Write a terminal artifact only when done or externally blocked; running "
        "checkpoints use a different path."
    ),
    "run_date": "Use 20260912 and record actual UTC timestamps, never copy an upstream run date.",
    "started_at_utc": "Record the actual UTC start separately from the fixed execution date.",
    "completed_at_utc": "Record the actual UTC completion separately from the fixed execution date.",
    "preconditions_checked": "Actual code, resource, identity and gate observations before expensive work.",
    "inference_substrate": (
        "Use the recognized literal for the work actually executed; custom free text caused "
        "the Exp7208 quarantine."
    ),
    "inference_substrate_class": (
        "Match actual generation, load-only, CPU or aggregation work and its duration floor."
    ),
    "execution_venue": (
        "Exactly host, kv260, gatemate or polarfire; the top-level orchestration here is host."
    ),
    "execution_host": "Actual hostname separate from venue.",
    "duration_s": "Measured monotonic work time; no padding or reclassification to evade a floor.",
    "source_artifact_hashes": "Bind code, source documents, manifests and raw evidence to claims.",
    "rows": (
        "Per unit/arm/seed metric, error and abstention for every comparison; retain full "
        "denominators."
    ),
    "sample_size_budget": (
        "Planned, attempted, completed, censored and independent units; no silent removal."
    ),
    "random_seed": "Freeze random choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash exact source, inputs, settings and raw unit rows.",
    "gate_check_summary": (
        "Every blocked_* verdict names failed check, upstream, field, expected and observed value."
    ),
    "verifier_is_oracle": (
        "True when correctness authority is reused as the verifier; independent code alone is "
        "not distinct authority."
    ),
    "verdict_class": (
        "Closed enum positive | circular_positive | null | blocked | disqualified | partial. "
        "partial means unfinished own work only."
    ),
    "honest_verdict": (
        "Use complete_ or complete: for completed findings; blocked_* for external absence. "
        "A failed acceptance gate forbids positive."
    ),
    "MODEL_SPECS": (
        "Only models actually invoked; [] for CPU/aggregation, mandated Qwen3.8 for every model task."
    ),
    "model_invoked": (
        "True only for actual model execution; upstream model outputs are cached evidence."
    ),
    "belief_audit_complete_score": "Independent metrics and cold replay complete.",
    "belief_promotion_score": "Positive prospective value plus independent causality/rollback proof.",
    "cold_reload_rows": "State identity and prediction/update equivalence.",
    "rollback_rows": "Actual old-state restoration and later decisions.",
    "causal_control_rows": "Time-separated authority and learned-state interventions.",
    "metric_recomputation_rows": "Headline values reconstructed from event rows.",
    "comparison_recomputation_rows": "Independent stream bootstrap protects every comparison.",
    "deletion_recomputation_rows": "Scheduled deletion effects come from event rows.",
    "source_grounding_rows": "Public parsing, compiler input, and two label executors agree.",
    "runtime_isolation_receipt": "Process and file-policy evidence proves future sidecar denial.",
    "checkpoint_receipt": "Cold state stays outside the terminal artifact.",
    "producer_gate_receipt": "Producer completeness and learning value stay separate facts.",
    "v635_history_receipt": "Prior committed-memory nulls remain visible and unpromoted.",
    "audit_errors": "A complete audit has no failed owned check.",
    "no_model_weight_mutation": "The audit changes no model weights.",
    "no_fitting_or_parameter_adjustment": "The audit only replays fixed state and fixed controls.",
    "certificate_published": "A circular cold audit publishes no correctness certificate.",
    "default_pipeline_modified": "Read-only evidence cannot enable the measured mechanism.",
}

unwrap_principled = exp7214.unwrap_principled
gate_check = exp7214.gate_check
gate_summary = exp7214.gate_summary


class AuditEvidenceError(ValueError):
    """Reject incomplete or malformed evidence before it can change a verdict."""


class ExperimentPaths:
    """Keep cold replay state outside the terminal result."""

    def __init__(self, checkpoint: Path, artifact: Path) -> None:
        self.checkpoint = checkpoint
        self.artifact = artifact

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return paths used by the required command."""

        return cls(DEFAULT_CHECKPOINT_PATH, DEFAULT_ARTIFACT_PATH)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put all test-owned output below one caller directory."""

        return cls(
            root / "checkpoints" / DEFAULT_CHECKPOINT_PATH.name, root / DEFAULT_ARTIFACT_PATH.name
        )


def _resolve(repo_root: Path, path: Path | str) -> Path:
    """Resolve repository-relative evidence without changing absolute paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _sha256_path(path: Path) -> str | None:
    """Hash exact bytes while missing input remains an explicit observation."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_object(path: Path) -> JsonDict:
    """Decode one object while malformed evidence remains a failed gate."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe_text(path: Path) -> str:
    """Read contract text without turning absence into an exception."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating evidence early."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _task_identity(text: str) -> JsonDict:
    """Read only the Exp7228 roadmap block for identity checks."""

    match = re.search(r"(?ms)^- id: exp7228-belief-cold-audit\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(1)
    milestone = re.search(r"(?m)^  milestone: (.+)$", block)
    deliverable = re.search(r"(?m)^  deliverable: (.+)$", block)
    return {
        "id": "exp7228-belief-cold-audit" if block else None,
        "milestone": None if milestone is None else milestone.group(1).strip(),
        "deliverable": None if deliverable is None else deliverable.group(1).strip(),
    }


def _producer_receipts(producer: Mapping[str, Any]) -> dict[str, str]:
    """Select immutable producer rows, state, and sealed stream receipts."""

    selected: dict[str, str] = {}
    for name in ("decision_rows_path", "checkpoint_path"):
        receipt = unwrap_principled(producer.get(name))
        if isinstance(receipt, Mapping) and receipt.get("path") and receipt.get("sha256"):
            selected[str(receipt["path"])] = str(receipt["sha256"])
    declared = unwrap_principled(producer.get("source_artifact_hashes"))
    if isinstance(declared, Mapping):
        for path, digest in declared.items():
            text = str(path)
            if "results/streams/experiment_7226/" in text or text.endswith(
                "results/experiment_7226_v636_belief_compiler.json"
            ):
                selected[text] = str(digest)
    return selected


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    producer_path: Path = DEFAULT_PRODUCER_PATH,
) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate all immutable bytes before any outcome row is reduced."""

    root = Path(repo_root)
    resolved_producer = _resolve(root, producer_path)
    producer = _load_object(resolved_producer)
    v635 = _load_object(_resolve(root, V635_AUDIT_PATH))
    source_hashes: JsonDict = {
        str(path): _sha256_path(_resolve(root, path)) for path in SOURCE_PATHS
    }
    source_hashes[str(producer_path)] = _sha256_path(resolved_producer)
    source_hashes[str(V635_AUDIT_PATH)] = _sha256_path(_resolve(root, V635_AUDIT_PATH))
    receipts = _producer_receipts(producer)
    observed_receipts = {path: _sha256_path(_resolve(root, path)) for path in receipts}
    source_hashes.update(observed_receipts)
    spec_text = _safe_text(_resolve(root, SPEC_PATH))
    roadmap_text = _safe_text(_resolve(root, "research-roadmap.yaml"))
    exclusion_text = _safe_text(_resolve(root, "ops/exclusion_manifest.yaml"))
    producer_quarantine = exp7214.quarantine_state(
        producer,
        exclusion_text,
        DEFAULT_PRODUCER_PATH.name,
        "exp7227-belief-learning",
    )
    v635_quarantine = exp7214.quarantine_state(
        v635,
        exclusion_text,
        V635_AUDIT_PATH.name,
        "exp7214-refinement-cold-audit",
    )
    imports = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "carnot.experiment_7199_v634_bounded_acquisition",
            "carnot.experiment_7214_v635_refinement_cold_audit",
            "carnot.experiment_7226_v636_belief_compiler",
            "carnot.experiment_7227_v636_belief_learning",
            "carnot.memory.transactional_constraint_memory",
        )
    }
    source_state = {
        str(path): "nonempty" if source_hashes[str(path)] is not None else "missing"
        for path in SOURCE_PATHS
    }
    output_state = {
        "artifact": _path_writable(paths.artifact),
        "checkpoint": _path_writable(paths.checkpoint),
        "checkpoint_under_results_checkpoints": (
            "results/checkpoints" in paths.checkpoint.as_posix()
            or paths.checkpoint.parent.name == "checkpoints"
        ),
    }
    required_fields = {name: name in producer for name in exp7227.REQUIRED_ARTIFACT_FIELDS}
    checksum_valid = bool(producer) and exp7227.reproducibility_checksum(producer) == producer.get(
        "reproducibility_checksum"
    )
    decision_receipt = producer.get("decision_rows_path", {})
    checkpoint_receipt = producer.get("checkpoint_path", {})
    decision_count = None
    if isinstance(decision_receipt, Mapping):
        decision_path = _resolve(root, str(decision_receipt.get("path", "")))
        try:
            decision_count = sum(1 for _ in decision_path.open(encoding="utf-8"))
        except OSError:
            decision_count = None
    expected_identity = {
        "id": "exp7228-belief-cold-audit",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT_PATH),
    }
    checks = [
        gate_check(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-CL-7228",
            True,
            "## REQ-CL-7228:" in spec_text,
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7228-*",
            8,
            len(set(re.findall(r"SCENARIO-CL-7228-[A-Z-]+", spec_text))),
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            {str(path): "nonempty" for path in SOURCE_PATHS},
            source_state,
        ),
        gate_check(
            "v636_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            expected_identity,
            _task_identity(roadmap_text),
        ),
        gate_check("required_imports", "python", "imports", {k: True for k in imports}, imports),
        gate_check(
            "output_destinations",
            "host_filesystem",
            "checkpoint,artifact",
            {key: True for key in output_state},
            output_state,
        ),
        gate_check(
            "producer_artifact_hash",
            "exp7227-belief-learning",
            str(DEFAULT_PRODUCER_PATH),
            EXPECTED_PRODUCER_SHA256,
            source_hashes[str(producer_path)],
        ),
        gate_check(
            "producer_status",
            "exp7227-belief-learning",
            "status",
            "complete",
            producer.get("status"),
        ),
        gate_check(
            "producer_run_complete",
            "exp7227-belief-learning",
            "belief_run_complete_score",
            1,
            producer.get("belief_run_complete_score"),
        ),
        gate_check(
            "producer_learning_value_observed",
            "exp7227-belief-learning",
            "belief_learning_value_score in {0,1}",
            True,
            producer.get("belief_learning_value_score") in {0, 1},
        ),
        gate_check(
            "producer_gate_passed",
            "exp7227-belief-learning",
            "gate_check_summary.passed",
            True,
            producer.get("gate_check_summary", {}).get("passed")
            if isinstance(producer.get("gate_check_summary"), Mapping)
            else None,
        ),
        gate_check(
            "producer_required_fields",
            "exp7227-belief-learning",
            "required_fields",
            {name: True for name in required_fields},
            required_fields,
        ),
        gate_check(
            "producer_no_model",
            "exp7227-belief-learning",
            "MODEL_SPECS,model_invoked",
            {"MODEL_SPECS": [], "model_invoked": False},
            {
                "MODEL_SPECS": unwrap_principled(producer.get("MODEL_SPECS")),
                "model_invoked": unwrap_principled(producer.get("model_invoked")),
            },
        ),
        gate_check(
            "producer_not_quarantined",
            "artifact_metadata_and_ops/exclusion_manifest.yaml",
            "quarantined",
            False,
            producer_quarantine["quarantined"],
        ),
        gate_check(
            "producer_reproducibility_checksum",
            "exp7227-belief-learning",
            "reproducibility_checksum",
            True,
            checksum_valid,
        ),
        gate_check(
            "producer_immutable_receipts",
            "exp7227 source receipts",
            "artifact,stream,decision,state.sha256",
            receipts,
            observed_receipts,
        ),
        gate_check(
            "producer_decision_denominator",
            "exp7227-belief-learning",
            "decision_rows_path.row_count",
            len(STREAM_SEEDS) * len(ARMS) * EVENTS_PER_SEED,
            decision_count,
        ),
        gate_check(
            "producer_state_receipt",
            "exp7227-belief-learning",
            "checkpoint_path.controller_count",
            len(STREAM_SEEDS) * len(ARMS),
            checkpoint_receipt.get("controller_count")
            if isinstance(checkpoint_receipt, Mapping)
            else None,
        ),
        gate_check(
            "v635_history_hash",
            "exp7214-refinement-cold-audit",
            str(V635_AUDIT_PATH),
            EXPECTED_V635_AUDIT_SHA256,
            source_hashes[str(V635_AUDIT_PATH)],
        ),
        gate_check(
            "v635_null_retained",
            "exp7214-refinement-cold-audit",
            "status,audit_complete,promotion,verdict",
            {
                "status": "complete",
                "refinement_audit_complete_score": 1,
                "memory_promotion_score": 0,
                "verdict_class": "null",
                "quarantined": False,
            },
            {
                "status": v635.get("status"),
                "refinement_audit_complete_score": v635.get("refinement_audit_complete_score"),
                "memory_promotion_score": v635.get("memory_promotion_score"),
                "verdict_class": v635.get("verdict_class"),
                "quarantined": v635_quarantine["quarantined"],
            },
        ),
    ]
    return checks, producer, source_hashes


def read_jsonl(path: Path) -> list[JsonDict]:
    """Read immutable event rows while rejecting malformed or non-object lines."""

    rows: list[JsonDict] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AuditEvidenceError("invalid_jsonl:non_object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise AuditEvidenceError("invalid_jsonl") from error
    return rows


def _outcome(prediction: str, exact_label: str) -> tuple[int, int, int]:
    """Compute full-denominator error without trusting stored outcome fields."""

    abstention = int(prediction == "abstain")
    error = int(abstention == 1 or prediction != exact_label)
    false_accept = int(prediction == "accept" and exact_label == "reject")
    return error, false_accept, abstention


def recompute_stream_metrics(
    decision_rows: Sequence[Mapping[str, Any]],
    producer_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    expected_seeds: Sequence[int] = STREAM_SEEDS,
    expected_arms: Sequence[str] = ARMS,
    events_per_seed: int = EVENTS_PER_SEED,
    warmup_count: int = WARMUP_COUNT,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Rebuild every stream aggregate and refuse any missing denominator."""

    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        grouped[(int(row["seed"]), str(row["arm"]))].append(row)
    expected_units = {(int(seed), str(arm)) for seed in expected_seeds for arm in expected_arms}
    if set(grouped) != expected_units:
        raise AuditEvidenceError("missing_denominator:unit_set")
    producer_by_unit = {str(row["unit_id"]): row for row in producer_rows or ()}
    rebuilt: list[JsonDict] = []
    receipts: list[JsonDict] = []
    for seed, arm in sorted(expected_units):
        rows = sorted(grouped[(seed, arm)], key=lambda row: int(row["chronology_index"]))
        indices = [int(row["chronology_index"]) for row in rows]
        if len(rows) != events_per_seed or indices != list(range(events_per_seed)):
            raise AuditEvidenceError(f"missing_denominator:{seed}:{arm}")
        outcomes = [_outcome(str(row["prediction"]), str(row["exact_label"])) for row in rows]
        prospective = [index for index, row in enumerate(rows) if row.get("prospective") is True]
        recurrence = [index for index, row in enumerate(rows) if row.get("window") == "recurrence"]
        if len(prospective) != events_per_seed - warmup_count:
            raise AuditEvidenceError(f"missing_denominator:prospective:{seed}:{arm}")
        query_count = sum(row.get("query_admitted") is True for row in rows)
        released_ids = {
            str(event_id) for row in rows for event_id in row.get("released_feedback_ids", [])
        }
        mismatch_count = sum(
            int(row.get(name) != value)
            for row, outcome in zip(rows, outcomes, strict=True)
            for name, value in zip(("error", "false_accept", "abstention"), outcome, strict=True)
        )
        error = sum(outcomes[index][0] for index in prospective)
        false_accept = sum(outcomes[index][1] for index in prospective)
        abstention = sum(outcomes[index][2] for index in prospective)
        recurrence_error = sum(outcomes[index][0] for index in recurrence)
        unit_id = f"{seed}:{arm}"
        aggregate = {
            "unit_id": unit_id,
            "arm": arm,
            "seed": seed,
            "metric": "independent_prospective_full_denominator_error",
            "error": error,
            "abstention": abstention,
            "event_count": len(prospective),
            "error_rate": error / len(prospective),
            "false_accept": false_accept,
            "false_accept_rate": false_accept / len(prospective),
            "recurrence_error": recurrence_error,
            "recurrence_event_count": len(recurrence),
            "recurrence_error_rate": recurrence_error / len(recurrence),
            "query_count": query_count,
            "released_query_count": len(released_ids),
            "pending_at_end": query_count - len(released_ids),
            "max_pending": max(int(row.get("pending_capacity_use", 0)) for row in rows),
        }
        expected = producer_by_unit.get(unit_id)
        compared = (
            "error",
            "abstention",
            "event_count",
            "error_rate",
            "false_accept",
            "false_accept_rate",
            "recurrence_error",
            "recurrence_event_count",
            "recurrence_error_rate",
            "query_count",
            "released_query_count",
            "pending_at_end",
            "max_pending",
        )
        producer_parity = expected is None or all(
            expected.get(name) == aggregate[name] for name in compared
        )
        chronology = all(
            row.get("prediction_before_release") is True
            and row.get("future_label_visible_to_decision") is False
            and row.get("hidden_parameter_visible_to_decision") is False
            for row in rows
        )
        rebuilt.append(aggregate)
        receipts.append(
            {
                "unit_id": unit_id,
                "arm": arm,
                "seed": seed,
                "event_row_count": len(rows),
                "prospective_row_count": len(prospective),
                "stored_outcome_mismatch_count": mismatch_count,
                "producer_aggregate_parity": producer_parity,
                "chronology_passed": chronology,
                "query_ceiling_passed": query_count <= exp7227.QUERY_CEILING,
                "pending_capacity_passed": aggregate["max_pending"] <= exp7227.PENDING_CAPACITY,
                "passed": mismatch_count == 0
                and producer_parity
                and chronology
                and query_count <= exp7227.QUERY_CEILING
                and aggregate["max_pending"] <= exp7227.PENDING_CAPACITY,
            }
        )
    return rebuilt, receipts


def _percentile(values: Sequence[float | int], probability: float) -> float:
    """Use the producer's published nearest-rank rule without its reducer."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(probability * (len(ordered) - 1)))))
    return ordered[index]


def _bootstrap_interval(values: Sequence[float], draws: int, salt: str) -> JsonDict:
    """Resample complete stream differences as the only independent units."""

    if not values:
        return {"estimate": 0.0, "ci95": [0.0, 0.0], "draws": draws}
    seed = int(transactional.sha256_json([exp7227.BOOTSTRAP_SEED, salt])[-16:], 16)
    generator = random.Random(seed)
    means = [
        sum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    ]
    return {
        "estimate": sum(values) / len(values),
        "ci95": [_percentile(means, 0.025), _percentile(means, 0.975)],
        "draws": draws,
    }


def recompute_comparisons(
    rows: Sequence[Mapping[str, Any]], *, draws: int = exp7227.BOOTSTRAP_DRAWS
) -> list[JsonDict]:
    """Rebuild all fixed producer comparisons from independent stream rows."""

    by_unit = {(int(row["seed"]), str(row["arm"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    result: list[JsonDict] = []
    for control in (
        "frozen_warmup",
        "reference_online_version_space",
        "original_committed_predicate",
    ):
        differences = []
        for seed in seeds:
            packed = by_unit[(seed, "packed_online_memory")]
            baseline = by_unit[(seed, control)]
            differences.append(
                {
                    "seed": seed,
                    "future_error_delta": float(packed["error_rate"])
                    - float(baseline["error_rate"]),
                    "false_accept_delta": float(packed["false_accept_rate"])
                    - float(baseline["false_accept_rate"]),
                    "recurrence_error_increase": float(packed["recurrence_error_rate"])
                    - float(baseline["recurrence_error_rate"]),
                }
            )
        comparison_id = f"packed_online_memory_vs_{control}"
        result.append(
            {
                "comparison_id": comparison_id,
                "independent_unit": "stream_seed",
                "independent_unit_count": len(differences),
                "seed_differences": differences,
                "future_error_delta": _bootstrap_interval(
                    [row["future_error_delta"] for row in differences],
                    draws,
                    comparison_id + ":error",
                ),
                "false_accept_delta": _bootstrap_interval(
                    [row["false_accept_delta"] for row in differences],
                    draws,
                    comparison_id + ":false_accept",
                ),
                "recurrence_error_increase": _bootstrap_interval(
                    [row["recurrence_error_increase"] for row in differences],
                    draws,
                    comparison_id + ":recurrence",
                ),
            }
        )
    return result


def compare_comparisons(
    rebuilt: Sequence[Mapping[str, Any]], producer_rows: Sequence[Mapping[str, Any]] | None
) -> list[JsonDict]:
    """Record exact producer parity while preserving every rebuilt value."""

    expected = {str(row["comparison_id"]): row for row in producer_rows or ()}
    return [
        {
            **dict(row),
            "producer_values": deepcopy(expected.get(str(row["comparison_id"]))),
            "passed": not expected or expected.get(str(row["comparison_id"])) == row,
        }
        for row in rebuilt
    ]


def recompute_deletions(decisions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Rebuild scheduled feedback-deletion effects from paired event rows."""

    result: list[JsonDict] = []
    seeds = sorted({int(row["seed"]) for row in decisions})
    for seed in seeds:
        packed = {
            int(row["chronology_index"]): row
            for row in decisions
            if int(row["seed"]) == seed and row["arm"] == "packed_online_memory"
        }
        withheld = {
            int(row["chronology_index"]): row
            for row in decisions
            if int(row["seed"]) == seed and row["arm"] == "packed_feedback_withheld"
        }
        if set(packed) != set(range(EVENTS_PER_SEED)) or set(withheld) != set(packed):
            raise AuditEvidenceError(f"missing_denominator:deletion:{seed}")
        release_indices = [index for index, row in packed.items() if row["withheld_release_ids"]]
        first_release = min(release_indices) if release_indices else EVENTS_PER_SEED
        changed = [
            index
            for index in sorted(packed)
            if packed[index]["prediction"] != withheld[index]["prediction"]
        ]
        future = [index for index in sorted(packed) if index >= WARMUP_COUNT]
        result.append(
            {
                "unit_id": f"{seed}:packed_feedback_withheld",
                "seed": seed,
                "withhold_segments": [list(segment) for segment in exp7227.WITHHOLD_SEGMENTS],
                "first_withheld_release_index": None
                if first_release == EVENTS_PER_SEED
                else first_release,
                "withheld_feedback_count": sum(
                    len(row["withheld_release_ids"]) for row in packed.values()
                ),
                "pre_release_difference_count": sum(index < first_release for index in changed),
                "changed_decision_count": len(changed),
                "changed_decision_event_ids": [packed[index]["event_id"] for index in changed],
                "packed_future_error": sum(
                    _outcome(str(packed[index]["prediction"]), str(packed[index]["exact_label"]))[0]
                    for index in future
                ),
                "withheld_future_error": sum(
                    _outcome(
                        str(withheld[index]["prediction"]), str(withheld[index]["exact_label"])
                    )[0]
                    for index in future
                ),
                "future_error_difference": sum(
                    _outcome(
                        str(withheld[index]["prediction"]), str(withheld[index]["exact_label"])
                    )[0]
                    - _outcome(str(packed[index]["prediction"]), str(packed[index]["exact_label"]))[
                        0
                    ]
                    for index in future
                ),
                "causal_dependence_observed": bool(changed),
                "no_access_before_release": not any(index < first_release for index in changed),
            }
        )
    return result


def compare_deletions(
    rebuilt: Sequence[Mapping[str, Any]], producer_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Keep exact scheduled-deletion parity visible per independent stream."""

    expected = {int(row["seed"]): row for row in producer_rows}
    return [
        {
            **dict(row),
            "producer_values": deepcopy(expected.get(int(row["seed"]))),
            "passed": expected.get(int(row["seed"])) == row,
        }
        for row in rebuilt
    ]


def source_grounding_rows(
    public_rows: Sequence[Mapping[str, Any]], authority_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Run public text through the parser, compiler input, and two executors."""

    authority = {str(row["event_id"]): row for row in authority_rows}
    by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in public_rows:
        by_seed[int(row["seed"])].append(row)
    result = []
    for seed in sorted(by_seed):
        parsed_count = 0
        parity_count = 0
        compiled_count = 0
        compiler = exp7226.PackedBeliefController()
        for public in by_seed[seed]:
            parsed = exp7226.exp7198.extract_public_input(str(public["public_input"]))
            parsed_ok = (
                parsed["family_id"] == public["family_id"]
                and parsed["numeric_value"] == public["numeric_value"]
            )
            parsed_count += int(parsed_ok)
            prediction, _ = compiler.predict(parsed)
            compiled_count += int(prediction in {"accept", "reject"})
            truth = authority[str(public["event_id"])]
            first = exp7226.exact_label(
                str(parsed["family_id"]),
                int(parsed["numeric_value"]),
                int(truth["hidden_parameter"]),
            )
            second = exp7226.independent_exact_label(
                str(parsed["family_id"]),
                int(parsed["numeric_value"]),
                int(truth["hidden_parameter"]),
            )
            parity_count += int(first == second == truth["exact_label"])
        count = len(by_seed[seed])
        result.append(
            {
                "unit_id": f"{seed}:source_grounding",
                "seed": seed,
                "public_event_count": count,
                "parsed_event_count": parsed_count,
                "compiler_input_count": compiled_count,
                "independent_executor_parity_count": parity_count,
                "authority_joined_after_public_execution": True,
                "passed": parsed_count == compiled_count == parity_count == count,
            }
        )
    return result


def reference_from_survivors(
    survivors: Mapping[str, set[int]],
) -> exp7199.VersionSpaceController:
    """Create the shipped reference controller from explicit hypothesis IDs."""

    controller = exp7199.VersionSpaceController()
    for family in exp7226.FAMILIES:
        controller.families[family].hypotheses = set(survivors[family])
    return controller


def reference_from_state(value: Mapping[str, Any]) -> exp7199.VersionSpaceController:
    """Restore reference state without using packed masks as its serializer."""

    controller = exp7199.VersionSpaceController()
    if set(value) != set(exp7226.FAMILIES):
        raise AuditEvidenceError("invalid_reference_families")
    for family in exp7226.FAMILIES:
        payload = value[family]
        state = controller.families[family]
        state.hypotheses = {int(item) for item in payload["hypotheses"]}
        state.support_ids = [str(item) for item in payload.get("support_ids", [])]
        state.validation_ids = [str(item) for item in payload.get("validation_ids", [])]
        state.candidate_parameter = payload.get("candidate_parameter")
        state.freeze_release_index = payload.get("freeze_release_index")
        state.committed_template = deepcopy(payload.get("committed_template"))
        state.superseded_templates = deepcopy(payload.get("superseded_templates", []))
        state.archive = deepcopy(payload.get("archive", []))
        state.epoch = int(payload.get("epoch", 0))
    return controller


def _predictions(controller: Any, public_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Record only public decision fields before any authority joins."""

    result = []
    for public in public_rows:
        prediction, disagreement = controller.predict(public)
        result.append(
            {
                "event_id": str(public["event_id"]),
                "prediction": prediction,
                "energy": disagreement,
            }
        )
    return result


def _vote_mismatch_count(
    packed: exp7226.PackedBeliefController,
    reference: exp7199.VersionSpaceController,
) -> int:
    """Compare cached votes with independent reference enumeration."""

    mismatches = 0
    for family in exp7226.FAMILIES:
        packed_state = packed.family_state(family)
        reference_hypotheses = reference.families[family].hypotheses
        mismatches += int(packed.survivors(family) != reference_hypotheses)
        expected_votes = [
            sum(
                exp7226.exact_label(family, value, parameter) == "accept"
                for parameter in reference_hypotheses
            )
            for value in exp7226.PARAMETER_DOMAIN
        ]
        mismatches += sum(
            left != right
            for left, right in zip(packed_state["vote_counts"], expected_votes, strict=True)
        )
    return mismatches


def _attack_row(
    seed: int,
    attack: str,
    *,
    rejection_reason: str | None,
    byte_equal: bool,
    hash_equal: bool,
    decision_equal: bool,
    passed: bool,
) -> JsonDict:
    """Use one stable schema for each transaction-boundary observation."""

    return {
        "unit_id": f"{seed}:{attack}",
        "arm": "packed_online_memory",
        "seed": seed,
        "metric": "transaction_state_mismatch_count",
        "error": int(not passed),
        "abstention": 0,
        "attack": attack,
        "rejection_reason": rejection_reason,
        "byte_equal": byte_equal,
        "hash_equal": hash_equal,
        "subsequent_decisions_equal": decision_equal,
        "passed": passed,
    }


def run_state_probe(
    state_record: Mapping[str, Any],
    public_rows: Sequence[Mapping[str, Any]],
    delayed_release: Mapping[str, Any],
) -> JsonDict:
    """Replay cold parity, delayed update, four attacks, and state removal."""

    seed = int(state_record["seed"])
    packed_state = state_record["packed_online_memory"]
    reference_state = state_record["reference_online_version_space"]
    packed = exp7226.PackedBeliefController.from_state(packed_state)
    reference = reference_from_state(reference_state)
    packed_before = _predictions(packed, public_rows)
    reference_before = _predictions(reference, public_rows)
    prediction_mismatches = sum(
        left != right for left, right in zip(packed_before, reference_before, strict=True)
    )
    vote_mismatches = _vote_mismatch_count(packed, reference)
    original_bytes = packed.state_bytes()
    original_hash = packed.state_hash()
    rollback_rows: list[JsonDict] = []

    updated = exp7226.PackedBeliefController.from_state(packed_state)
    updated_reference = reference_from_state(reference_state)
    premature_hash = updated.state_hash()
    premature_reason = None
    try:
        updated.commit_batch(
            [delayed_release],
            current_cycle=int(delayed_release["release_index"]) - 1,
            expected_parent_hash=updated.state_hash(),
        )
    except exp7226.CommitRejected as error:
        premature_reason = str(error)
    premature_preserved = premature_hash == updated.state_hash()
    receipt = updated.commit_batch(
        [delayed_release],
        current_cycle=int(delayed_release["release_index"]),
        expected_parent_hash=updated.state_hash(),
    )
    updated_reference.observe(
        delayed_release,
        observed_label=str(delayed_release["observed_label"]),
        role=str(delayed_release["role"]),
        request_index=int(delayed_release["request_index"]),
        release_index=int(delayed_release["release_index"]),
    )
    next_vote_mismatches = _vote_mismatch_count(updated, updated_reference)
    next_prediction_mismatches = sum(
        left != right
        for left, right in zip(
            _predictions(updated, public_rows),
            _predictions(updated_reference, public_rows),
            strict=True,
        )
    )
    update_mismatches = next_vote_mismatches + next_prediction_mismatches
    delayed_passed = (
        premature_reason == "future_release" and premature_preserved and update_mismatches == 0
    )
    rollback_rows.append(
        _attack_row(
            seed,
            "delayed_correction",
            rejection_reason=premature_reason,
            byte_equal=premature_preserved,
            hash_equal=premature_preserved,
            decision_equal=update_mismatches == 0,
            passed=delayed_passed,
        )
    )

    stale = exp7226.PackedBeliefController.from_state(packed_state)
    stale_before = _predictions(stale, public_rows)
    stale_reason = None
    try:
        stale.commit_batch(
            [delayed_release],
            current_cycle=int(delayed_release["release_index"]),
            expected_parent_hash="sha256:" + "0" * 64,
        )
    except exp7226.CommitRejected as error:
        stale_reason = str(error)
    stale_equal = stale.state_bytes() == original_bytes
    stale_decisions = _predictions(stale, public_rows) == stale_before
    rollback_rows.append(
        _attack_row(
            seed,
            "stale_parent_commit",
            rejection_reason=stale_reason,
            byte_equal=stale_equal,
            hash_equal=stale.state_hash() == original_hash,
            decision_equal=stale_decisions,
            passed=stale_reason == "stale_parent" and stale_equal and stale_decisions,
        )
    )

    corrupted = deepcopy(dict(packed_state))
    corrupted["families"][exp7226.FAMILIES[0]]["survivor_mask"] = exp7226.FULL_MASK + 1
    corrupt_reason = None
    try:
        exp7226.PackedBeliefController.from_state(corrupted)
    except ValueError as error:
        corrupt_reason = str(error)
    rollback_rows.append(
        _attack_row(
            seed,
            "corrupted_mask",
            rejection_reason=corrupt_reason,
            byte_equal=packed.state_bytes() == original_bytes,
            hash_equal=packed.state_hash() == original_hash,
            decision_equal=_predictions(packed, public_rows) == packed_before,
            passed=corrupt_reason == "invalid_survivor_mask"
            and packed.state_bytes() == original_bytes,
        )
    )

    duplicate = exp7226.PackedBeliefController.from_state(packed_state)
    duplicate.commit_batch(
        [delayed_release],
        current_cycle=int(delayed_release["release_index"]),
        expected_parent_hash=duplicate.state_hash(),
    )
    duplicate_bytes = duplicate.state_bytes()
    duplicate_hash = duplicate.state_hash()
    duplicate_decisions = _predictions(duplicate, public_rows)
    duplicate_reason = None
    try:
        duplicate.commit_batch(
            [delayed_release],
            current_cycle=int(delayed_release["release_index"]),
            expected_parent_hash=duplicate.state_hash(),
        )
    except exp7226.CommitRejected as error:
        duplicate_reason = str(error)
    duplicate_equal = duplicate.state_bytes() == duplicate_bytes
    rollback_rows.append(
        _attack_row(
            seed,
            "duplicate_release",
            rejection_reason=duplicate_reason,
            byte_equal=duplicate_equal,
            hash_equal=duplicate.state_hash() == duplicate_hash,
            decision_equal=_predictions(duplicate, public_rows) == duplicate_decisions,
            passed=duplicate_reason == "duplicate_release" and duplicate_equal,
        )
    )

    restored = updated.rollback(receipt)
    restored_decisions = _predictions(updated, public_rows) == packed_before
    rollback_passed = (
        restored["byte_identical"] is True
        and updated.state_bytes() == original_bytes
        and updated.state_hash() == original_hash
        and restored_decisions
    )
    rollback_rows.append(
        _attack_row(
            seed,
            "rollback",
            rejection_reason=None,
            byte_equal=updated.state_bytes() == original_bytes,
            hash_equal=updated.state_hash() == original_hash,
            decision_equal=restored_decisions,
            passed=rollback_passed,
        )
    )

    removed = exp7226.PackedBeliefController()
    removed_predictions = _predictions(removed, public_rows)
    changed = sum(
        left["prediction"] != right["prediction"]
        for left, right in zip(packed_before, removed_predictions, strict=True)
    )
    cold_passed = (
        prediction_mismatches == 0
        and vote_mismatches == 0
        and update_mismatches == 0
        and transactional.sha256_bytes(original_bytes) == original_hash
        and all(row["passed"] for row in rollback_rows)
    )
    return {
        "cold_row": {
            "unit_id": f"{seed}:cold_reload",
            "arm": "packed_online_memory",
            "seed": seed,
            "metric": "cold_state_parity_mismatch_count",
            "error": prediction_mismatches + vote_mismatches + update_mismatches,
            "abstention": sum(row["prediction"] == "abstain" for row in packed_before),
            "packed_state_hash": original_hash,
            "reference_state_hash": reference.state_hash(),
            "prediction_digest_before_release": transactional.sha256_json(packed_before),
            "prediction_mismatch_count": prediction_mismatches,
            "vote_mismatch_count": vote_mismatches,
            "next_update_mismatch_count": update_mismatches,
            "delayed_release_index": int(delayed_release["release_index"]),
            "fixed_configuration": True,
            "passed": cold_passed,
        },
        "rollback_rows": rollback_rows,
        "removal_row": {
            "unit_id": f"{seed}:learned_state_removal",
            "arm": "packed_online_memory",
            "seed": seed,
            "metric": "prospective_error_increase_after_state_removal",
            "error": 0,
            "abstention": sum(row["prediction"] == "abstain" for row in removed_predictions),
            "changed_decision_count": changed,
            "learned_predictions": packed_before,
            "removed_predictions": removed_predictions,
            "labels_joined_after_decision": True,
        },
    }


def _authority_keys(value: Any) -> set[str]:
    """Find nested authority names in the public-only cold payload."""

    if isinstance(value, Mapping):
        return set(value).intersection(exp7226.FORBIDDEN_AUTHORITY_FIELDS) | set().union(
            *(_authority_keys(item) for item in value.values())
        )
    if isinstance(value, list):
        return set().union(*(_authority_keys(item) for item in value))
    return set()


def cold_reload_worker(args: argparse.Namespace) -> JsonDict:
    """Load one serialized state under a real future-sidecar deny policy."""

    forbidden = Path(os.environ["CARNOT_7228_FORBIDDEN_SIDECAR"]).resolve()
    denied_attempts = 0
    original_open = io.open

    def deny_sidecar(file: Any, *open_args: Any, **open_kwargs: Any) -> Any:
        """Deny the exact future sidecar while other files stay readable."""

        nonlocal denied_attempts
        if Path(str(file)).resolve() == forbidden:
            denied_attempts += 1
            raise PermissionError("future_sidecar_denied")
        return original_open(file, *open_args, **open_kwargs)

    try:
        io.open = deny_sidecar
        checkpoint = _load_object(Path(args.checkpoint_path))
        seed = int(args.cold_seed)
        state = next(row for row in checkpoint["states"] if int(row["seed"]) == seed)
        public = checkpoint["public_probes"][str(seed)]
        release = checkpoint["delayed_releases"][str(seed)]
        sidecar_read_success = True
        try:
            forbidden.read_bytes()
        except PermissionError:
            sidecar_read_success = False
    finally:
        io.open = original_open
    result = run_state_probe(state, public, release)
    result["process_receipt"] = {
        "parent_pid": int(os.environ.get("CARNOT_7228_PARENT_PID", "-1")),
        "worker_pid": os.getpid(),
        "fresh_process": int(os.environ.get("CARNOT_7228_PARENT_PID", "-1")) == os.getppid(),
        "variant": os.environ.get("CARNOT_7228_VARIANT"),
        "future_sidecar_path": str(forbidden),
        "future_sidecar_open_denied": denied_attempts == 1,
        "future_sidecar_denied_attempt_count": denied_attempts,
        "future_sidecar_read_success": sidecar_read_success,
        "public_authority_fields_absent": not _authority_keys(public),
        "public_input_hash": transactional.sha256_json(public),
        "gpu_disabled": os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("NVIDIA_VISIBLE_DEVICES") == "none",
        "network_cache_offline": os.environ.get("HF_HUB_OFFLINE") == "1"
        and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "no_model_load": not any(
            name in sys.modules for name in ("llama_cpp", "transformers", "torch")
        ),
    }
    return result


def _isolated_environment(parent_pid: int, sidecar: Path, variant: str) -> JsonDict:
    """Disable accelerators and network caches and identify the denied file."""

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "none",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "CARNOT_7228_PARENT_PID": str(parent_pid),
            "CARNOT_7228_FORBIDDEN_SIDECAR": str(sidecar.resolve()),
            "CARNOT_7228_VARIANT": variant,
        }
    )
    return environment


def spawn_cold_worker(checkpoint: Path, seed: int, sidecar: Path, variant: str) -> JsonDict:
    """Start one bounded worker with streamed output and an explicit timeout."""

    command = [
        sys.executable,
        "-I",
        str(WRAPPER_PATH),
        "--cold-worker",
        "--date",
        RUN_DATE,
        "--checkpoint-path",
        str(checkpoint),
        "--cold-seed",
        str(seed),
    ]
    return exp7214.cold_support._stream_subprocess(  # noqa: SLF001
        command,
        _isolated_environment(os.getpid(), sidecar, variant),
        label=f"COLD {variant.upper()} STREAM {seed} SUBPROCESS",
        timeout_s=120.0,
    )


def derive_terminal_scores(
    producer_value: int, audit_complete: bool, causal_checks_passed: bool
) -> tuple[int, int, str, str]:
    """Keep audit completion independent from producer learning value."""

    complete = int(audit_complete)
    promotion = int(complete == 1 and producer_value == 1 and causal_checks_passed)
    if complete and not causal_checks_passed:
        return (
            complete,
            0,
            "disqualified",
            "complete_disqualified: causal or isolation checks failed",
        )
    if promotion:
        return (
            complete,
            1,
            "circular_positive",
            "complete: producer value and independent belief audit passed",
        )
    if complete:
        return (
            complete,
            0,
            "null",
            "complete_null: belief audit completed and producer learning value was null",
        )
    return 0, 0, "partial", "partial: owned belief audit work did not complete"


def _empty_budget(seeds: Sequence[int] = STREAM_SEEDS) -> JsonDict:
    """Keep planned counts while blocked work remains unattempted."""

    return {
        "planned_streams": len(seeds),
        "attempted_streams": 0,
        "completed_streams": 0,
        "censored_streams": len(seeds),
        "independent_units_planned": len(seeds),
        "independent_units_attempted": 0,
        "independent_units_completed": 0,
        "independent_units_censored": len(seeds),
        "planned_event_recomputations": len(seeds) * len(ARMS) * EVENTS_PER_SEED,
        "completed_event_recomputations": 0,
        "planned_cold_processes": len(seeds) * 2,
        "completed_cold_processes": 0,
    }


def base_artifact(
    checks: Sequence[Mapping[str, Any]],
    producer: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    paths: ExperimentPaths,
    *,
    duration_s: float,
    started_at: str | None = None,
    completed_at: str | None = None,
    seeds: Sequence[int] = STREAM_SEEDS,
) -> JsonDict:
    """Build a schema-complete blocked base before scientific classification."""

    now = datetime.now(UTC).isoformat()
    v635 = _load_object(_resolve(REPO_ROOT, V635_AUDIT_PATH))
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at or now,
        "completed_at_utc": completed_at or now,
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": socket.gethostname(),
        "duration_s": duration_s,
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": _empty_budget(seeds),
        "random_seed": {
            "audit": AUDIT_SEED,
            "producer_bootstrap": exp7227.BOOTSTRAP_SEED,
            "bootstrap_draws": exp7227.BOOTSTRAP_DRAWS,
            "streams": list(seeds),
            "probe_start": PROBE_START,
            "probe_count": PROBE_COUNT,
        },
        "reproducibility_checksum": None,
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_external:unknown_precondition",
        "MODEL_SPECS": [],
        "model_invoked": False,
        "belief_audit_complete_score": 0,
        "belief_promotion_score": 0,
        "cold_reload_rows": [],
        "rollback_rows": [],
        "causal_control_rows": [],
        "metric_recomputation_rows": [],
        "comparison_recomputation_rows": [],
        "deletion_recomputation_rows": [],
        "source_grounding_rows": [],
        "runtime_isolation_receipt": {},
        "checkpoint_receipt": {"path": str(paths.checkpoint), "sha256": None},
        "producer_gate_receipt": {
            "artifact_path": str(DEFAULT_PRODUCER_PATH),
            "artifact_hash": source_hashes.get(str(DEFAULT_PRODUCER_PATH)),
            "belief_run_complete_score": unwrap_principled(
                producer.get("belief_run_complete_score")
            ),
            "belief_learning_value_score": unwrap_principled(
                producer.get("belief_learning_value_score")
            ),
            "null_promoted": False,
        },
        "v635_history_receipt": {
            "artifact_path": str(V635_AUDIT_PATH),
            "artifact_hash": source_hashes.get(str(V635_AUDIT_PATH)),
            "refinement_audit_complete_score": v635.get("refinement_audit_complete_score"),
            "memory_promotion_score": v635.get("memory_promotion_score"),
            "verdict_class": v635.get("verdict_class"),
            "retained_as_history": True,
        },
        "audit_errors": [],
        "no_model_weight_mutation": True,
        "no_fitting_or_parameter_adjustment": True,
        "certificate_published": False,
        "default_pipeline_modified": False,
    }


def _stable_artifact(value: Any) -> Any:
    """Exclude clocks and process identities from the reproducibility hash."""

    if isinstance(value, Mapping):
        return {
            str(key): _stable_artifact(item)
            for key, item in value.items()
            if key
            not in {
                "duration_s",
                "execution_host",
                "started_at_utc",
                "completed_at_utc",
                "parent_pid",
                "worker_pid",
                "future_sidecar_path",
                "reproducibility_checksum",
            }
        }
    if isinstance(value, list):
        return [_stable_artifact(item) for item in value]
    return value


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash source identities, fixed settings, and all audit rows."""

    return transactional.sha256_json(_stable_artifact(dict(artifact)))


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    producer: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    paths: ExperimentPaths,
    *,
    duration_s: float,
    started_at: str | None = None,
    completed_at: str | None = None,
    seeds: Sequence[int] = STREAM_SEEDS,
) -> JsonDict:
    """Return row-free terminal evidence for an unchanged external block."""

    artifact = base_artifact(
        checks,
        producer,
        source_hashes,
        paths,
        duration_s=duration_s,
        started_at=started_at,
        completed_at=completed_at,
        seeds=seeds,
    )
    failed = artifact["gate_check_summary"]["failed_check"] or "unknown_precondition"
    artifact["honest_verdict"] = f"blocked_external:{failed}"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any], *, expected_seeds: Sequence[int] = STREAM_SEEDS
) -> list[str]:
    """Recompute schema, completion, promotion, and checksum decisions."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return ["missing_fields:" + ",".join(missing)]
    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS), "field_principles")
    add(artifact.get("schema") != SCHEMA, "schema")
    add(artifact.get("experiment_id") != EXPERIMENT_ID, "experiment_id")
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    add(artifact.get("execution_venue") != EXECUTION_VENUE, "execution_venue")
    add(not artifact.get("execution_host"), "execution_host")
    add(artifact.get("MODEL_SPECS") != [], "model_specs")
    add(artifact.get("model_invoked") is not False, "model_invoked")
    add(artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle")
    add(artifact.get("no_model_weight_mutation") is not True, "weight_mutation")
    add(
        artifact.get("no_fitting_or_parameter_adjustment") is not True,
        "fitting_or_parameter_adjustment",
    )
    add(artifact.get("certificate_published") is not False, "certificate_published")
    add(artifact.get("default_pipeline_modified") is not False, "default_pipeline_modified")
    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        add(artifact.get("status") != "blocked", "blocked_status")
        add(artifact.get("inference_substrate") != "blocked_no_run", "blocked_substrate")
        add(bool(artifact.get("rows")), "blocked_rows")
        add(artifact.get("belief_audit_complete_score") != 0, "blocked_complete")
        add(artifact.get("belief_promotion_score") != 0, "blocked_promotion")
        summary = artifact.get("gate_check_summary", {})
        add(summary.get("passed") is not False, "blocked_gate")
        for name in ("failed_check", "upstream", "field", "expected_value", "observed_value"):
            add(summary.get(name) is None, f"blocked_gate_{name}")
    else:
        add(artifact.get("status") != "complete", "status")
        add(artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate")
        add(
            artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
            "inference_substrate_class",
        )
        add(artifact.get("gate_check_summary", {}).get("passed") is not True, "preconditions")
        add(len(artifact.get("rows", [])) != len(expected_seeds) * len(ARMS), "row_count")
        add(
            len(artifact.get("metric_recomputation_rows", [])) != len(expected_seeds) * len(ARMS),
            "metric_row_count",
        )
        add(len(artifact.get("comparison_recomputation_rows", [])) != 3, "comparison_count")
        add(
            len(artifact.get("deletion_recomputation_rows", [])) != len(expected_seeds),
            "deletion_count",
        )
        add(len(artifact.get("cold_reload_rows", [])) != len(expected_seeds), "cold_count")
        add(len(artifact.get("rollback_rows", [])) != len(expected_seeds) * 5, "rollback_count")
        add(
            len(artifact.get("causal_control_rows", [])) != len(expected_seeds) * 2,
            "causal_count",
        )
        add(
            len(artifact.get("source_grounding_rows", [])) != len(expected_seeds),
            "grounding_count",
        )
        owned = (
            "metric_recomputation_rows",
            "comparison_recomputation_rows",
            "deletion_recomputation_rows",
            "cold_reload_rows",
            "rollback_rows",
            "causal_control_rows",
            "source_grounding_rows",
        )
        complete = (
            all(row.get("passed") is True for name in owned for row in artifact.get(name, []))
            and not artifact.get("audit_errors")
            and artifact.get("runtime_isolation_receipt", {}).get("all_workers_isolated") is True
        )
        causal = all(row.get("passed") is True for row in artifact.get("causal_control_rows", []))
        producer_value = int(
            artifact.get("producer_gate_receipt", {}).get("belief_learning_value_score", 0)
        )
        expected = derive_terminal_scores(producer_value, complete, causal)
        add(artifact.get("belief_audit_complete_score") != expected[0], "completion_score")
        add(artifact.get("belief_promotion_score") != expected[1], "promotion_score")
        add(artifact.get("verdict_class") != expected[2], "verdict_class")
        add(artifact.get("honest_verdict") != expected[3], "honest_verdict")
        add(producer_value == 0 and artifact.get("belief_promotion_score") != 0, "null_promoted")
    add(
        artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
        "reproducibility_checksum",
    )
    return errors


def _future_probe(public: Mapping[str, Any], seed: int, offset: int) -> JsonDict:
    """Reuse an actual public input as a new label-blind future event."""

    return {
        "event_id": f"exp7228-{seed}-future-{offset:04d}",
        "seed": seed,
        "chronology_index": EVENTS_PER_SEED + offset,
        "family_id": public["family_id"],
        "numeric_value": public["numeric_value"],
        "public_input": public["public_input"],
        "source_event_id": public["event_id"],
    }


def _sidecar_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode ephemeral future authority variants for file-policy evidence."""

    return b"".join(transactional.canonical_json_bytes(dict(row)) for row in rows)


def _score_removal_row(removal: Mapping[str, Any], labels: Mapping[str, str]) -> JsonDict:
    """Join labels only after both learned and removed states have decided."""

    learned = removal["learned_predictions"]
    removed = removal["removed_predictions"]
    learned_error = sum(
        _outcome(str(row["prediction"]), labels[str(row["event_id"])])[0] for row in learned
    )
    removed_error = sum(
        _outcome(str(row["prediction"]), labels[str(row["event_id"])])[0] for row in removed
    )
    result = dict(removal)
    result.update(
        {
            "error": removed_error - learned_error,
            "event_count": len(learned),
            "learned_error": learned_error,
            "removed_error": removed_error,
            "prospective_error_increase": removed_error - learned_error,
            "passed": removal.get("labels_joined_after_decision") is True,
        }
    )
    result.pop("learned_predictions", None)
    result.pop("removed_predictions", None)
    return result


def _audit_error_names(
    owned_rows: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    state_count: int,
    expected_state_count: int,
    isolation: Mapping[str, Any],
) -> list[str]:
    """Name each failed owned check without hiding a second failure."""

    errors = [
        name
        for name, rows in owned_rows
        if not rows or any(row.get("passed") is not True for row in rows)
    ]
    if state_count != expected_state_count:
        errors.append("state_count")
    if isolation.get("all_workers_isolated") is not True:
        errors.append("runtime_isolation")
    if isolation.get("protected_inputs_unchanged") is not True:
        errors.append("protected_inputs_changed")
    return errors


def _require_valid(errors: Sequence[str], prefix: str) -> None:
    """Refuse publication when a terminal object fails validation."""

    if errors:
        raise AuditEvidenceError(prefix + ":" + ",".join(errors))


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    producer_path: Path = DEFAULT_PRODUCER_PATH,
    seeds: Sequence[int] = STREAM_SEEDS,
    progress: bool = False,
) -> JsonDict:
    """Run row reconstruction, cold controls, attacks, and terminal scoring."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    if progress:
        print(
            "PHASE 0 START: authenticate code, producer, raw rows, state, and outputs", flush=True
        )
    checks, producer, source_hashes = collect_preconditions(
        repo_root, paths, producer_path=producer_path
    )
    if not all(row["passed"] for row in checks):
        if progress:
            print("PHASE 0 END: external prerequisite failed; no audit replay ran", flush=True)
        return build_blocked_artifact(
            checks,
            producer,
            source_hashes,
            paths,
            duration_s=time.monotonic() - started,
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            seeds=seeds,
        )
    protected_before = {
        path: digest
        for path, digest in source_hashes.items()
        if digest is not None and _resolve(repo_root, path).is_file()
    }
    if progress:
        print("PHASE 0 END: producer completion and immutable receipts passed", flush=True)
        print("PHASE 1 START: load immutable event rows and separated stream views", flush=True)
    selected_seeds = {int(seed) for seed in seeds}
    decision_receipt = producer["decision_rows_path"]
    decision_rows = read_jsonl(_resolve(repo_root, str(decision_receipt["path"])))
    decision_rows = [row for row in decision_rows if int(row["seed"]) in selected_seeds]
    checkpoint = _load_object(_resolve(repo_root, str(producer["checkpoint_path"]["path"])))
    state_rows = [row for row in checkpoint["states"] if int(row["seed"]) in selected_seeds]
    views = exp7227.load_stream_views(repo_root, COMPILER_ARTIFACT_PATH)
    public_rows = [row for row in views.public if int(row["seed"]) in selected_seeds]
    authority_rows = [row for row in views.authority if int(row["seed"]) in selected_seeds]
    if progress:
        print(
            f"PHASE 1 END: loaded decisions={len(decision_rows)} states={len(state_rows)}",
            flush=True,
        )
        print(
            "PHASE 2 START: independently rebuild 100 stream-arm metrics and comparisons",
            flush=True,
        )
    producer_rows = [row for row in producer["rows"] if int(row["seed"]) in selected_seeds]
    rebuilt_rows, metric_receipts = recompute_stream_metrics(
        decision_rows,
        producer_rows,
        expected_seeds=seeds,
    )
    rebuilt_comparisons = recompute_comparisons(rebuilt_rows)
    comparison_receipts = compare_comparisons(
        rebuilt_comparisons,
        producer["comparison_rows"] if tuple(seeds) == STREAM_SEEDS else None,
    )
    rebuilt_deletions = recompute_deletions(decision_rows)
    producer_deletions = [
        row for row in producer["memory_deletion_rows"] if int(row["seed"]) in selected_seeds
    ]
    deletion_receipts = compare_deletions(rebuilt_deletions, producer_deletions)
    if progress:
        print(
            f"PHASE 2 END: metrics={len(rebuilt_rows)} comparisons={len(comparison_receipts)} "
            f"deletions={len(deletion_receipts)}",
            flush=True,
        )
        print(
            "PHASE 3 START: run public parser, compiler input, and independent label executors",
            flush=True,
        )
    grounding = source_grounding_rows(public_rows, authority_rows)
    if progress:
        print(f"PHASE 3 END: grounded streams={len(grounding)}", flush=True)
        print(
            "PHASE 4 START: seal packed/reference states and public-only future probes", flush=True
        )
    public_by_seed = {
        seed: sorted(
            [row for row in public_rows if int(row["seed"]) == seed],
            key=lambda row: int(row["chronology_index"]),
        )
        for seed in seeds
    }
    authority_by_id = {str(row["event_id"]): row for row in authority_rows}
    probes: dict[str, list[JsonDict]] = {}
    delayed: dict[str, JsonDict] = {}
    baseline_future: list[JsonDict] = []
    shifted_future: list[JsonDict] = []
    labels_by_seed: dict[int, dict[str, str]] = {}
    for seed in seeds:
        source_probes = public_by_seed[int(seed)][PROBE_START : PROBE_START + PROBE_COUNT]
        future = [_future_probe(row, int(seed), index) for index, row in enumerate(source_probes)]
        probes[str(seed)] = future
        labels: dict[str, str] = {}
        for public, source in zip(future, source_probes, strict=True):
            truth = authority_by_id[str(source["event_id"])]
            label = str(truth["exact_label"])
            labels[str(public["event_id"])] = label
            baseline_future.append({"event_id": public["event_id"], "exact_label": label})
            shifted_future.append(
                {
                    "event_id": public["event_id"],
                    "exact_label": "reject" if label == "accept" else "accept",
                }
            )
        labels_by_seed[int(seed)] = labels
        delayed[str(seed)] = {
            "event_id": f"exp7228-{seed}-delayed-correction",
            "family_id": future[0]["family_id"],
            "numeric_value": future[0]["numeric_value"],
            "observed_label": labels[str(future[0]["event_id"])],
            "role": "support",
            "request_index": EVENTS_PER_SEED,
            "release_index": EVENTS_PER_SEED + 4,
        }
    audit_checkpoint = {
        "schema": "carnot.exp7228.cold_checkpoint.v1",
        "run_date": RUN_DATE,
        "producer_checkpoint_hash": producer["checkpoint_path"]["sha256"],
        "states": state_rows,
        "public_probes": probes,
        "delayed_releases": delayed,
        "contains_future_sidecar": False,
    }
    transactional._atomic_write(
        paths.checkpoint, transactional.canonical_json_bytes(audit_checkpoint)
    )
    checkpoint_hash = _sha256_path(paths.checkpoint)
    if progress:
        print(f"PHASE 4 END: checkpoint hash={checkpoint_hash}", flush=True)
        print("PHASE 5 SUBPROCESS START: cold-load baseline and shifted-label variants", flush=True)
    cold_rows: list[JsonDict] = []
    rollback_rows: list[JsonDict] = []
    causal_rows: list[JsonDict] = []
    process_receipts: list[JsonDict] = []
    baseline_hash = transactional.sha256_bytes(_sidecar_bytes(baseline_future))
    shifted_hash = transactional.sha256_bytes(_sidecar_bytes(shifted_future))
    with tempfile.TemporaryDirectory(prefix="carnot-exp7228-sidecar-") as temporary:
        temporary_root = Path(temporary)
        baseline_path = temporary_root / "future-baseline.jsonl"
        shifted_path = temporary_root / "future-shifted.jsonl"
        transactional._atomic_write(baseline_path, _sidecar_bytes(baseline_future))
        transactional._atomic_write(shifted_path, _sidecar_bytes(shifted_future))
        for number, seed in enumerate(seeds, start=1):
            baseline = spawn_cold_worker(paths.checkpoint, int(seed), baseline_path, "baseline")
            shifted = spawn_cold_worker(paths.checkpoint, int(seed), shifted_path, "shifted")
            process_receipts.extend([baseline["process_receipt"], shifted["process_receipt"]])
            cold_rows.append(
                {
                    **baseline["cold_row"],
                    "fresh_process": baseline["process_receipt"]["fresh_process"],
                    "future_sidecar_open_denied": baseline["process_receipt"][
                        "future_sidecar_open_denied"
                    ],
                }
            )
            rollback_rows.extend(baseline["rollback_rows"])
            prediction_same = (
                baseline["cold_row"]["prediction_digest_before_release"]
                == shifted["cold_row"]["prediction_digest_before_release"]
            )
            isolation_passed = all(
                receipt["fresh_process"]
                and receipt["future_sidecar_open_denied"]
                and not receipt["future_sidecar_read_success"]
                and receipt["public_authority_fields_absent"]
                and receipt["no_model_load"]
                for receipt in (baseline["process_receipt"], shifted["process_receipt"])
            )
            causal_rows.append(
                {
                    "unit_id": f"{seed}:future_label_shift",
                    "arm": "packed_online_memory",
                    "seed": int(seed),
                    "metric": "pre_release_prediction_change_count",
                    "error": int(not prediction_same),
                    "abstention": 0,
                    "control": "future_label_shift",
                    "released_history_hash": baseline["cold_row"]["packed_state_hash"],
                    "baseline_future_sidecar_hash": baseline_hash,
                    "shifted_future_sidecar_hash": shifted_hash,
                    "public_input_hash": baseline["process_receipt"]["public_input_hash"],
                    "predictions_unchanged_before_release": prediction_same,
                    "sidecar_access_denied_in_both_processes": isolation_passed,
                    "passed": prediction_same
                    and baseline_hash != shifted_hash
                    and isolation_passed,
                }
            )
            removal = _score_removal_row(baseline["removal_row"], labels_by_seed[int(seed)])
            causal_rows.append({**removal, "control": "learned_state_removal"})
            if progress:
                print(
                    f"PHASE 5 PROGRESS: completed stream {number}/{len(seeds)}; "
                    f"elapsed_s={time.monotonic() - started:.3f}",
                    flush=True,
                )
    if progress:
        print(
            f"PHASE 5 SUBPROCESS END: completed cold processes={len(process_receipts)}", flush=True
        )
        print(
            "PHASE 6 START: classify completion, causality, rollback, and retained null history",
            flush=True,
        )
    protected_after = {path: _sha256_path(_resolve(repo_root, path)) for path in protected_before}
    isolation = {
        "worker_process_count": len(process_receipts),
        "distinct_worker_pids": len({row["worker_pid"] for row in process_receipts}),
        "all_workers_isolated": bool(process_receipts)
        and all(
            row["fresh_process"]
            and row["future_sidecar_open_denied"]
            and not row["future_sidecar_read_success"]
            and row["public_authority_fields_absent"]
            and row["gpu_disabled"]
            and row["network_cache_offline"]
            and row["no_model_load"]
            for row in process_receipts
        ),
        "future_authority_variant_hashes": {
            "baseline": baseline_hash,
            "shifted": shifted_hash,
        },
        "protected_inputs_unchanged": protected_before == protected_after,
        "future_sidecar_persisted": False,
        "worker_receipts": process_receipts,
    }
    owned = (
        ("metric_recomputation", metric_receipts),
        ("comparison_recomputation", comparison_receipts),
        ("deletion_recomputation", deletion_receipts),
        ("source_grounding", grounding),
        ("cold_reload", cold_rows),
        ("rollback", rollback_rows),
        ("causal_controls", causal_rows),
    )
    audit_errors = _audit_error_names(owned, len(state_rows), len(seeds), isolation)
    causal_passed = all(row["passed"] for row in causal_rows)
    audit_complete = not audit_errors
    scores = derive_terminal_scores(
        int(producer["belief_learning_value_score"]), audit_complete, causal_passed
    )
    artifact = base_artifact(
        checks,
        producer,
        source_hashes,
        paths,
        duration_s=time.monotonic() - started,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        seeds=seeds,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "rows": rebuilt_rows,
            "sample_size_budget": {
                "planned_streams": len(seeds),
                "attempted_streams": len(seeds),
                "completed_streams": len(seeds),
                "censored_streams": 0,
                "independent_units_planned": len(seeds),
                "independent_units_attempted": len(seeds),
                "independent_units_completed": len(seeds),
                "independent_units_censored": 0,
                "planned_event_recomputations": len(seeds) * len(ARMS) * EVENTS_PER_SEED,
                "completed_event_recomputations": len(decision_rows),
                "planned_cold_processes": len(seeds) * 2,
                "completed_cold_processes": len(process_receipts),
                "bootstrap_resamples_planned": exp7227.BOOTSTRAP_DRAWS,
                "bootstrap_resamples_completed": exp7227.BOOTSTRAP_DRAWS,
            },
            "belief_audit_complete_score": scores[0],
            "belief_promotion_score": scores[1],
            "verdict_class": scores[2],
            "honest_verdict": scores[3],
            "cold_reload_rows": cold_rows,
            "rollback_rows": rollback_rows,
            "causal_control_rows": causal_rows,
            "metric_recomputation_rows": metric_receipts,
            "comparison_recomputation_rows": comparison_receipts,
            "deletion_recomputation_rows": deletion_receipts,
            "source_grounding_rows": grounding,
            "runtime_isolation_receipt": isolation,
            "checkpoint_receipt": {
                "path": str(paths.checkpoint),
                "sha256": checkpoint_hash,
                "state_count": len(state_rows),
                "probe_events_per_state": PROBE_COUNT,
                "contains_future_sidecar": False,
            },
            "audit_errors": audit_errors,
        }
    )
    artifact["producer_gate_receipt"]["null_promoted"] = (
        int(producer["belief_learning_value_score"]) == 0 and scores[1] != 0
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact, expected_seeds=seeds)
    _require_valid(errors, "invalid_terminal_artifact")
    if progress:
        print(
            f"PHASE 6 END: complete={scores[0]} promotion={scores[1]} "
            f"audit_errors={len(audit_errors)}",
            flush=True,
        )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date, private output root, and cold worker inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--producer-artifact", type=Path, default=DEFAULT_PRODUCER_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--cold-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cold-seed", type=int, default=STREAM_SEEDS[0], help=argparse.SUPPRESS)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit and atomically publish only validated terminal bytes."""

    print("PHASE 0 PRECONDITION: parse fixed inputs before checking resources", flush=True)
    args = parse_args(argv)
    if str(args.date) != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    if args.cold_worker:
        result = cold_reload_worker(args)
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    if args.output_root is None:
        paths = ExperimentPaths(Path(args.checkpoint_path), paths.artifact)
    if args.validate:
        artifact = _load_object(paths.artifact)
        errors = validate_artifact(artifact)
        print(json.dumps({"ok": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    artifact = build_and_seal(
        REPO_ROOT,
        paths,
        producer_path=args.producer_artifact,
        progress=True,
    )
    print("PHASE 7 FINAL VALIDATION START: recompute terminal schema and checksum", flush=True)
    errors = validate_artifact(artifact)
    _require_valid(errors, "final_validation_failed")
    print("PHASE 7 FINAL VALIDATION END: terminal object is valid", flush=True)
    print("PHASE 8 ATOMIC TERMINAL WRITE START", flush=True)
    transactional._atomic_write(paths.artifact, transactional.canonical_json_bytes(artifact))
    print(f"PHASE 8 ATOMIC TERMINAL WRITE END: wrote {paths.artifact}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper owns this command boundary.
    raise SystemExit(main())
