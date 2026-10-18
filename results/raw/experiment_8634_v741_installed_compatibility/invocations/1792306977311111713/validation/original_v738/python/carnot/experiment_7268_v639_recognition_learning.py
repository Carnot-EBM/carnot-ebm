"""Measure prospective learning from the sealed active-recognition fixture.

The run reuses the exact Exp7267 controllers and sealed streams. It keeps
completion separate from scientific value, so a complete null remains useful.

Spec refs: REQ-CL-7268 and SCENARIO-CL-7268-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time
from typing import Any

from carnot import experiment_7267_v639_recognition_prototype as prototype
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7268
SCHEMA = "carnot.exp7268.v639_recognition_learning.v1"
MILESTONE = "2026.09.639"
RUN_DATE = "20260913"
RANDOM_SEED = 7_268_000
BOOTSTRAP_SEED = 7_268_901
BOOTSTRAP_RESAMPLES = 10_000
MEASUREMENT_LIMIT_S = 1_800
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INVOCATION_COUNTS = {
    "attempted_model_loads": 0,
    "completed_model_loads": 0,
    "attempted_generation_calls": 0,
    "completed_generation_calls": 0,
    "usable_answers": 0,
}
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
REDUCER_SUBSTRATE = "aggregation_from_upstream_artifacts"
REDUCER_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"

ARMS = prototype.ARMS
STREAM_COUNT = prototype.STREAM_COUNT
EVENTS_PER_STREAM = prototype.EVENTS_PER_STREAM
QUERY_CEILING = prototype.QUERY_CEILING
MEMORY_CAP_BYTES = prototype.MEMORY_CAPS["total_bytes"]
EXPECTED_EVENT_ROWS = STREAM_COUNT * EVENTS_PER_STREAM * len(ARMS)
RECOGNITION_ARMS = set(prototype.RECOGNITION_ARMS)
RECEIPT_KINDS = (
    "query",
    "addition_deactivation",
    "nomination",
    "validation",
    "commit",
    "reactivation",
)
COST_OPERATIONS = (
    "lookup",
    "query",
    "update",
    "validation",
    "memory_bytes",
    "durable_commit",
    "full_event",
)
COMPARISON_SPECS = (
    ("future_error_vs_reset", "future_error_rate", "active_recognition", "reset"),
    ("false_accept_vs_reset", "false_accept_rate", "active_recognition", "reset"),
    (
        "recurrence_error_vs_random",
        "recurrence_error_rate",
        "active_recognition",
        "random_query_recognition",
    ),
    (
        "recurrence_error_vs_shuffle",
        "recurrence_error_rate",
        "active_recognition",
        "shuffled_archive_association",
    ),
    (
        "recurrence_degradation_vs_frozen",
        "recurrence_error_rate",
        "active_recognition",
        "frozen",
    ),
    (
        "future_error_vs_full_memory",
        "future_error_rate",
        "active_recognition",
        "full_version_space_memory",
    ),
    (
        "full_memory_vs_reset",
        "future_error_rate",
        "full_version_space_memory",
        "reset",
    ),
)
SCIENTIFIC_GATE_NAMES = (
    "future_error_vs_reset",
    "false_accept_vs_reset",
    "recurrence_error_vs_random",
    "recurrence_error_vs_shuffle",
    "recurrence_degradation_vs_frozen",
    "prospective_causal_change",
    "pre_release_isolation",
    "query_memory_limits",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
WRAPPER_PATH = Path("scripts/experiments/experiment_7268_v639_recognition_learning.py")
TEST_PATH = Path("tests/python/test_experiment_7268_v639_recognition_learning.py")
DEFAULT_UPSTREAM_ARTIFACT = Path("results/experiment_7267_v639_recognition_prototype.json")
DEFAULT_ARTIFACT = Path("results/experiment_7268_v639_recognition_learning.json")
UPSTREAM_RAW_ROWS = REPO_ROOT / "results/raw/experiment_7267/event_rows.jsonl"
EXPECTED_UPSTREAM_SHA256 = "sha256:f456dac10abed3ebec7dfeaf323ee54599b031b7be694f15b64f4aa375fd606c"
SCENARIO_PATTERN = re.compile(r"SCENARIO-CL-7268-[A-Z-]+")
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7226_v636_belief_compiler.py"),
    Path("python/carnot/experiment_7254_v638_coverage_learning.py"),
    Path("python/carnot/experiment_7255_v638_coverage_audit.py"),
    Path("python/carnot/experiment_7267_v639_recognition_prototype.py"),
    Path("python/carnot/experiment_7268_v639_recognition_learning.py"),
    WRAPPER_PATH,
    TEST_PATH,
    SPEC_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "field_principles",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "verifier_is_oracle",
    "honest_verdict",
    "verdict_class",
    "validation_receipts",
    "recognition_run_complete_score",
    "recognition_value_score",
    "continuous_self_learning_task",
    "prequential_rows_path",
    "comparison_rows",
    "cost_rows",
    "no_model_weight_mutation",
)
FIELD_PRINCIPLES = {
    "schema": "Version the result; keep ordinary experiment and milestone fields.",
    "experiment_id": "Bind this evidence to the active Exp7268 task.",
    "milestone": "Bind this evidence to milestone 2026.09.639.",
    "status": "Use complete or blocked only; keep unfinished work in checkpoints.",
    "run_date": "Use 20260913 with actual UTC start and end timestamps.",
    "started_at_utc": "Record the actual UTC invocation start.",
    "completed_at_utc": "Record the actual UTC measurement end.",
    "field_principles": "Store explanations here and ordinary values at top level.",
    "preconditions_checked": "Retain exact hashes, ownership, and failures before replay.",
    "MODEL_SPECS": "Declare models executable now; this task has none.",
    "model_invoked": "Derive model use from actual calls; this task makes none.",
    "invocation_counts": "Separate attempted and completed model work from usable answers.",
    "inference_substrate": "Describe the exact CPU controller replay.",
    "inference_substrate_class": "Declare actual compute and do not pad elapsed time.",
    "execution_venue": "Use host for host orchestration.",
    "duration_s": "Measure monotonic time and disjoint phase spans.",
    "random_seed": "Freeze stream and bootstrap seeds before outcomes.",
    "reproducibility_checksum": "Bind code, inputs, configuration, and raw evidence.",
    "source_artifact_hashes": "Authenticate the prototype, streams, and source bytes.",
    "rows": "Retain every stream-arm metric with censoring state.",
    "sample_size_budget": "Record planned, attempted, completed, and censored units.",
    "acceptance_gate_results": "Keep expected, observed, passed, and principle per gate.",
    "gate_check_summary": "Name the upstream and exact failed check for a block.",
    "verifier_is_oracle": "Expose that the exact evaluator defines correctness.",
    "honest_verdict": "Use complete_ for measurements and blocked_ for external absence.",
    "verdict_class": "Use the closed class set; oracle authority forbids positive.",
    "validation_receipts": "Retain actual commands, exit codes, and log hashes.",
    "recognition_run_complete_score": "Account for each sealed stream-arm unit.",
    "recognition_value_score": "Require every frozen causal, safety, and error gate.",
    "continuous_self_learning_task": "True marks real online constraint-state updates.",
    "prequential_rows_path": "Every prediction precedes feedback and resulting commits.",
    "comparison_rows": "Keep stream-paired intervals for both strata and full memory.",
    "cost_rows": "Report lookup, query, update, validation, memory, and durable costs.",
    "no_model_weight_mutation": "Only constraint memory changes; model weights stay fixed.",
}


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw rows, checkpoints, candidate bytes, and terminal bytes separate."""

    prequential_rows: Path
    lifecycle_receipts: Path
    e2e_sidecar: Path
    evidence_sidecar: Path
    provisional: Path
    stream_shards: Path
    terminal_candidate: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the task-owned paths below the repository results directory."""

        return cls.from_results_root(REPO_ROOT / "results")

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put all test output below a caller-owned temporary directory."""

        return cls.from_results_root(root)

    @classmethod
    def from_results_root(cls, root: Path) -> ExperimentPaths:
        """Derive raw, checkpoint, candidate, and terminal output paths."""

        raw = root / "raw" / "experiment_7268"
        checkpoints = root / "checkpoints"
        return cls(
            raw / "prequential_rows.jsonl",
            raw / "lifecycle_receipts.jsonl",
            checkpoints / "experiment_7268_v639_e2e.json",
            checkpoints / "experiment_7268_v639_evidence.json",
            checkpoints / "experiment_7268_v639_in_progress.json",
            raw / "streams",
            raw / "terminal_candidate.json",
            root / DEFAULT_ARTIFACT.name,
        )


@dataclass(frozen=True)
class LearningPanel:
    """Keep event rows, summaries, lifecycle receipts, and completion counts."""

    event_rows: list[JsonDict]
    rows: list[JsonDict]
    receipts: list[JsonDict]
    completed_stream_ids: list[str]
    censored_stream_ids: list[str]
    maximum_memory_bytes: int


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Print one flushed phase boundary for the external watchdog."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _resolve(repo_root: Path, path: str | Path) -> Path:
    """Resolve repository evidence while preserving absolute test paths."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _task_identity(text: str) -> JsonDict:
    """Extract only the active Exp7268 block from the executable roadmap."""

    match = re.search(r"(?ms)^- id: exp7268-recognition-learning\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(1)
    milestone = re.search(r"(?m)^  milestone: (.+)$", block)
    deliverable = re.search(r"(?m)^  deliverable: (.+)$", block)
    return {
        "id": "exp7268-recognition-learning" if block else None,
        "milestone": None if milestone is None else milestone.group(1).strip(),
        "deliverable": None if deliverable is None else deliverable.group(1).strip(),
    }


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    upstream_path: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict]:
    """Authenticate the ready prototype, source bytes, streams, and output owners."""

    upstream_file = _resolve(repo_root, upstream_path or DEFAULT_UPSTREAM_ARTIFACT)
    upstream = prototype._load_object(upstream_file)
    spec = _resolve(repo_root, SPEC_PATH).read_text(encoding="utf-8")
    roadmap = _resolve(repo_root, "research-roadmap.yaml").read_text(encoding="utf-8")
    exclusions = _resolve(repo_root, "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    source_hashes = {
        str(_resolve(repo_root, path)): prototype._sha256_path(_resolve(repo_root, path))
        for path in SOURCE_PATHS
    }
    source_hashes[str(upstream_file)] = prototype._sha256_path(upstream_file)
    stream_receipts = upstream.get("stream_manifest", {}).get("receipts", {})
    receipt_paths = [
        upstream.get("stream_manifest", {}).get("manifest_receipt", {}),
        *stream_receipts.values(),
    ]
    for receipt in receipt_paths:
        if isinstance(receipt, Mapping) and receipt.get("path"):
            receipt_path = _resolve(repo_root, str(receipt["path"]))
            source_hashes[str(receipt_path)] = prototype._sha256_path(receipt_path)
    prototype_sources = upstream.get("source_artifact_hashes", {})
    immutable_paths = (
        repo_root / "python/carnot/experiment_7267_v639_recognition_prototype.py",
        repo_root / "scripts/experiments/experiment_7267_v639_recognition_prototype.py",
        repo_root / "tests/python/test_experiment_7267_v639_recognition_prototype.py",
    )
    immutable_observed = {str(path): prototype._sha256_path(path) for path in immutable_paths}
    immutable_expected = {str(path): prototype_sources.get(str(path)) for path in immutable_paths}
    stream_expected = {
        str(_resolve(repo_root, str(receipt.get("path")))): receipt.get("sha256")
        for receipt in receipt_paths
        if isinstance(receipt, Mapping) and receipt.get("path")
    }
    stream_observed = {
        path: prototype._sha256_path(_resolve(repo_root, path)) for path in stream_expected
    }
    writable = {
        field: prototype._path_writable(getattr(paths, field))
        for field in (
            "prequential_rows",
            "lifecycle_receipts",
            "e2e_sidecar",
            "evidence_sidecar",
            "provisional",
            "terminal_candidate",
            "artifact",
        )
    }
    current_uid = os.getuid()
    owner_observed = {}
    for field in writable:
        parent = getattr(paths, field).parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        owner_observed[field] = parent.stat().st_uid == current_uid
    checks = [
        prototype.gate_check(
            "driving_capability_spec", str(SPEC_PATH), "REQ-CL-7268", True, "REQ-CL-7268" in spec
        ),
        prototype.gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7268-*",
            8,
            len(set(SCENARIO_PATTERN.findall(spec))),
        ),
        prototype.gate_check(
            "v639_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            {
                "id": "exp7268-recognition-learning",
                "milestone": MILESTONE,
                "deliverable": str(DEFAULT_ARTIFACT),
            },
            _task_identity(roadmap),
        ),
        prototype.gate_check(
            "prototype_artifact_hash",
            str(upstream_file),
            "sha256",
            EXPECTED_UPSTREAM_SHA256,
            prototype._sha256_path(upstream_file),
        ),
        prototype.gate_check(
            "recognition_fixture_ready",
            "exp7267-recognition-prototype",
            "recognition_fixture_ready_score",
            1,
            upstream.get("recognition_fixture_ready_score"),
        ),
        prototype.gate_check(
            "prototype_terminal_state",
            "exp7267-recognition-prototype",
            "status,verdict_class",
            ["complete", "circular_positive"],
            [upstream.get("status"), upstream.get("verdict_class")],
        ),
        prototype.gate_check(
            "immutable_prototype_code",
            "exp7267-recognition-prototype",
            "source_artifact_hashes",
            immutable_expected,
            immutable_observed,
        ),
        prototype.gate_check(
            "immutable_sealed_streams",
            "exp7267-recognition-prototype.stream_manifest",
            "receipts",
            stream_expected,
            stream_observed,
        ),
        prototype.gate_check(
            "recognition_contract",
            "exp7267-recognition-prototype",
            "stream_count,arms,query_ceiling,memory_cap",
            [24, 8, 128, 69_632],
            [
                upstream.get("stream_manifest", {}).get("prospective", {}).get("stream_count"),
                len(ARMS),
                upstream.get("recognition_contract", {}).get("query_ceiling_per_arm"),
                upstream.get("recognition_contract", {}).get("memory_caps", {}).get("total_bytes"),
            ],
        ),
        prototype.gate_check(
            "not_quarantined",
            "ops/exclusion_manifest.yaml",
            "exp7267,exp7268",
            False,
            "exp7267" in exclusions or "exp7268" in exclusions,
        ),
        prototype.gate_check(
            "writable_output_paths", "host", "paths", dict.fromkeys(writable, True), writable
        ),
        prototype.gate_check(
            "output_parent_ownership",
            "host",
            "uid",
            dict.fromkeys(owner_observed, True),
            owner_observed,
        ),
    ]
    return checks, source_hashes, upstream


def load_sealed_views(repo_root: Path, paths: ExperimentPaths) -> prototype.StreamViews:
    """Load only authenticated prospective public, release, and authority bytes."""

    del paths
    root = repo_root / "results/streams/experiment_7267"
    manifest = prototype._load_object(root / "stream_manifest.json")["prospective"]
    views = prototype.StreamViews(
        prototype._read_jsonl(root / "prospective_public.jsonl"),
        prototype._read_jsonl(root / "prospective_private_authority.jsonl"),
        prototype._read_jsonl(root / "prospective_releases.jsonl"),
        manifest,
    )
    errors = prototype.stream_conformance_errors(views, "prospective")
    if errors:
        raise ValueError("sealed_stream_conformance:" + ",".join(errors))
    return views


def _retained_constraint_count(row: Mapping[str, Any]) -> int:
    """Report retained released constraints under each shipped memory policy."""

    released = int(row["released_query_count_after"])
    arm = str(row["arm"])
    if arm == "feedback_withheld":
        return 0
    if arm in RECOGNITION_ARMS or arm == "previous_coverage":
        return min(released, prototype.LEDGER_CAPACITY)
    return released


def _lifecycle_rows(row: Mapping[str, Any], seal: str) -> list[JsonDict]:
    """Bind lifecycle receipts to one sealed event-arm execution row."""

    base = {
        "stream_id": row["stream_id"],
        "arm": row["arm"],
        "event_id": row["event_id"],
        "chronology_index": row["chronology_index"],
        "prediction_seal_sha256": seal,
    }
    receipts: list[JsonDict] = []
    if row["query_selected"]:
        receipts.append(
            {
                **base,
                "kind": "query",
                "charged_label_count": 1,
                "release_index": row["query_release_index"],
                "prediction_frozen_before_release": True,
            }
        )
    release_count = int(row["released_query_count_after"]) - int(row["released_query_count_before"])
    if release_count > 0:
        receipts.append(
            {
                **base,
                "kind": "commit",
                "release_count": release_count,
                "same_event_correction": False,
            }
        )
        if row["arm"] in RECOGNITION_ARMS:
            receipts.extend(
                [
                    {
                        **base,
                        "kind": "nomination",
                        "release_count": release_count,
                        "selection_change_count": row["selection_change_count"],
                    },
                    {
                        **base,
                        "kind": "validation",
                        "release_count": release_count,
                        "fresh_feedback_required": True,
                        "zero_contradictions_required": True,
                    },
                ]
            )
    if int(row["archive_admission_count"]) > 0:
        receipts.append(
            {
                **base,
                "kind": "addition_deactivation",
                "addition_count": int(row["archive_admission_count"]),
                "deactivation_count": int(row["archive_admission_count"]),
            }
        )
    if int(row["archive_reactivation_count"]) > 0:
        receipts.append(
            {
                **base,
                "kind": "reactivation",
                "reactivation_count": int(row["archive_reactivation_count"]),
                "fresh_validation_required": True,
            }
        )
    for index, receipt in enumerate(receipts):
        receipt["receipt_id"] = transactional.sha256_json([base, receipt["kind"], index])
    return receipts


def _enrich_stream_rows(
    rows: Sequence[Mapping[str, Any]],
    releases: Mapping[str, Mapping[str, Any]],
    controller_replay_ns: int,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Measure reducer costs and retain receipts for each executed event-arm row."""

    enriched: list[JsonDict] = []
    receipts: list[JsonDict] = []
    amortized = controller_replay_ns // max(1, len(rows))
    for source in rows:
        full_start = time.perf_counter_ns()
        lookup_start = time.perf_counter_ns()
        schedule = releases[str(source["event_id"])]
        lookup_ns = time.perf_counter_ns() - lookup_start
        validation_start = time.perf_counter_ns()
        valid = int(source["full_denominator_error"]) == int(
            source["prediction"] != source["later_released_label"]
        )
        validation_ns = time.perf_counter_ns() - validation_start
        if not valid:
            raise ValueError("sealed_error_mismatch")
        query_start = time.perf_counter_ns()
        query_release_index = (
            int(source["chronology_index"]) + int(schedule["delay"])
            if source["query_selected"]
            else None
        )
        query_ns = time.perf_counter_ns() - query_start if source["query_selected"] else 0
        row = {
            **dict(source),
            "query_release_index": query_release_index,
            "retained_constraint_count": _retained_constraint_count(source),
            "lookup_cost_ns": lookup_ns,
            "query_cost_ns": query_ns,
            "update_cost_ns": 0,
            "validation_cost_ns": validation_ns,
            "controller_replay_amortized_ns": amortized,
        }
        seal = transactional.sha256_json(
            [
                row["stream_id"],
                row["arm"],
                row["event_id"],
                row["prediction"],
                row["released_query_count_before"],
            ]
        )
        update_start = time.perf_counter_ns()
        bound_receipts = _lifecycle_rows(row, seal)
        update_ns = time.perf_counter_ns() - update_start if bound_receipts else 0
        row["update_cost_ns"] = update_ns
        row["prediction_seal_sha256"] = seal
        row["full_event_cost_ns"] = amortized + lookup_ns + query_ns + update_ns + validation_ns
        row["cost_measurement_scope"] = "controller_replay_amortized_plus_event_receipt_reducer"
        enriched.append(row)
        receipts.extend(bound_receipts)
    return enriched, receipts


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return the deterministic nearest-rank percentile used by the frozen reducer."""

    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(probability * len(ordered))))
    return float(ordered[index])


def _reduce_group(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce one stream-arm unit without dropping abstentions or recurrence."""

    ordered = sorted(rows, key=lambda row: int(row["chronology_index"]))
    base = prototype._reduce_group(ordered)
    future = [row for row in ordered if int(row["chronology_index"]) >= prototype.WARMUP_COUNT]
    recurrence = [row for row in future if row["recurrence_eligible"] is True]
    correct = [
        int(row["chronology_index"])
        for row in recurrence
        if int(row["full_denominator_error"]) == 0
    ]
    delay = (correct[0] - 768) if correct else len(recurrence)
    future_count = len(future)
    recurrence_count = len(recurrence)
    base.update(
        {
            "metric": "prospective_full_denominator_error",
            "future_error_rate": int(base["future_error"]) / future_count,
            "false_accept_rate": int(base["false_accept"]) / future_count,
            "coverage_rate": 1.0 - int(base["abstention"]) / future_count,
            "recurrence_error_rate": int(base["recurrence_error"]) / recurrence_count,
            "recurrence_recovery_delay_events": delay,
            "retained_constraint_count": max(
                int(row.get("retained_constraint_count", row["released_query_count_after"]))
                for row in ordered
            ),
            "full_event_cost_p50_ns": _percentile(
                [float(row.get("full_event_cost_ns", 0)) for row in ordered], 0.50
            ),
            "full_event_cost_p95_ns": _percentile(
                [float(row.get("full_event_cost_ns", 0)) for row in ordered], 0.95
            ),
        }
    )
    return base


def reduce_prequential_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep each stream as an independent unit and preserve the frozen arm order."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["stream_id"]), str(row["arm"]))].append(row)
    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    return [
        _reduce_group(group)
        for _, group in sorted(groups.items(), key=lambda item: (item[0][0], arm_order[item[0][1]]))
    ]


def prequential_row_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check chronology, prediction seals, costs, limits, and authority isolation."""

    errors = list(prototype.event_row_errors(rows))

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(
        any(
            int(row.get(key, 0)) < 0
            for row in rows
            for key in (
                "lookup_cost_ns",
                "query_cost_ns",
                "update_cost_ns",
                "validation_cost_ns",
                "full_event_cost_ns",
            )
        ),
        "negative_cost",
    )
    add(
        any(
            row.get("query_selected") is True
            and row.get("query_release_index") is not None
            and int(row["query_release_index"]) < int(row["chronology_index"])
            for row in rows
        ),
        "release_before_query",
    )
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["stream_id"]), str(row["arm"]))].append(row)
    add(
        any(
            sum(int(row["query_selected"]) for row in group) > QUERY_CEILING
            for group in groups.values()
        ),
        "query_cap",
    )
    add(any(int(row["memory_total_bytes"]) > MEMORY_CAP_BYTES for row in rows), "memory_cap")
    return errors


def independent_reduce(path: Path) -> list[JsonDict]:
    """Cold-reduce one stream at a time so large evidence stays memory-bounded."""

    reduced: list[JsonDict] = []
    current_stream: str | None = None
    stream_rows: list[JsonDict] = []

    def finish_stream() -> None:
        if not stream_rows:
            return
        errors = prequential_row_errors(stream_rows)
        if errors:
            raise ValueError("raw_row_validation_failed:" + ",".join(errors))
        reduced.extend(reduce_prequential_rows(stream_rows))

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("invalid_raw_row")
            stream_id = str(value.get("stream_id"))
            if current_stream is not None and stream_id != current_stream:
                finish_stream()
                stream_rows.clear()
            current_stream = stream_id
            stream_rows.append(value)
    finish_stream()
    if not reduced:
        raise ValueError("raw_rows_unavailable")
    return reduced


def independent_causal_summary(path: Path) -> JsonDict:
    """Cold-reduce causal counts per stream without retaining the full raw file."""

    totals = {
        "prospective_selection_change_count": 0,
        "later_changed_prediction_count": 0,
        "pre_release_difference_count": 0,
        "cap_violation_count": 0,
        "constraint_addition_count": 0,
        "constraint_deactivation_count": 0,
        "reactivation_count": 0,
    }
    current_stream: str | None = None
    stream_rows: list[JsonDict] = []

    def finish_stream() -> None:
        if not stream_rows:
            return
        for key, value in causal_summary_from_rows(stream_rows).items():
            totals[key] += int(value)

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("invalid_raw_row")
            stream_id = str(value.get("stream_id"))
            if current_stream is not None and stream_id != current_stream:
                finish_stream()
                stream_rows.clear()
            current_stream = stream_id
            stream_rows.append(value)
    finish_stream()
    if current_stream is None:
        raise ValueError("raw_rows_unavailable")
    return totals


def run_learning_panel(
    views: prototype.StreamViews,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    progress: bool = False,
) -> LearningPanel:
    """Replay each stream through all eight arms and checkpoint every unit."""

    selected = tuple(
        stream_ids or (f"prospective-{index + 1:02d}" for index in range(STREAM_COUNT))
    )
    releases = {str(row["event_id"]): row for row in views.releases}
    all_rows: list[JsonDict] = []
    all_receipts: list[JsonDict] = []
    completed: list[str] = []
    maximum_memory = 0
    started = time.monotonic()
    last_heartbeat = started
    for index, stream_id in enumerate(selected):
        before = time.perf_counter_ns()
        panel = prototype.run_recognition_panel(views, stream_ids=(stream_id,), progress=False)
        replay_ns = time.perf_counter_ns() - before
        enriched, receipts = _enrich_stream_rows(panel.event_rows, releases, replay_ns)
        errors = prequential_row_errors(enriched)
        if errors:
            raise ValueError("prequential_conformance:" + ",".join(errors))
        shard = paths.stream_shards / f"{stream_id}.jsonl"
        shard_receipt = prototype._atomic_write(shard, prototype.jsonl_bytes(enriched))
        all_rows.extend(enriched)
        all_receipts.extend(receipts)
        completed.append(stream_id)
        maximum_memory = max(maximum_memory, panel.maximum_memory_bytes)
        checkpoint = {
            "schema": SCHEMA,
            "status": "in_progress",
            "completed_stream_ids": completed,
            "planned_stream_ids": list(selected),
            "completed_event_arm_rows": len(all_rows),
            "last_stream_shard": shard_receipt,
            "elapsed_s": time.monotonic() - started,
        }
        prototype._atomic_write(paths.provisional, transactional.canonical_json_bytes(checkpoint))
        now = time.monotonic()
        if progress:
            print(
                f"phase 4 benchmark unit {index + 1}/{len(selected)} "
                f"completed_rows={len(all_rows)} elapsed_s={now - started:.3f}",
                flush=True,
            )
        if now - last_heartbeat >= 60:
            print(
                f"phase 4 benchmark heartbeat completed_units={len(completed)} "
                f"completed_rows={len(all_rows)} elapsed_s={now - started:.3f}",
                flush=True,
            )
            last_heartbeat = now
        if now - started > MEASUREMENT_LIMIT_S:
            raise TimeoutError("measurement_limit_exceeded_checkpoint_preserved")
    return LearningPanel(
        all_rows,
        reduce_prequential_rows(all_rows),
        all_receipts,
        completed,
        [],
        maximum_memory,
    )


def _bootstrap_interval(values: Sequence[float], draws: int, salt: str) -> JsonDict:
    """Resample whole paired streams with the fixed deterministic seed."""

    if not values:
        return {"estimate": 0.0, "ci95": [0.0, 0.0]}
    seed = int(transactional.sha256_json([BOOTSTRAP_SEED, salt])[-16:], 16)
    generator = random.Random(seed)
    means = [
        sum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    ]
    return {
        "estimate": sum(values) / len(values),
        "ci95": [_percentile(means, 0.025), _percentile(means, 0.975)],
    }


def build_comparison_rows(
    rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP_RESAMPLES
) -> list[JsonDict]:
    """Build stream-paired overall and stratum intervals for every comparator."""

    strata: tuple[str | None, ...] = (
        None,
        "separated_recurrence",
        "overlapping_recurrence",
    )
    comparisons: list[JsonDict] = []
    for comparison_id, metric, treatment_arm, control_arm in COMPARISON_SPECS:
        for stratum in strata:
            selected = [row for row in rows if stratum is None or row["stratum"] == stratum]
            by_unit = {(str(row["stream_id"]), str(row["arm"])): row for row in selected}
            stream_ids = sorted({str(row["stream_id"]) for row in selected})
            differences = [
                float(by_unit[(stream_id, treatment_arm)][metric])
                - float(by_unit[(stream_id, control_arm)][metric])
                for stream_id in stream_ids
            ]
            interval = _bootstrap_interval(
                differences,
                draws,
                f"{comparison_id}:{stratum or 'overall'}",
            )
            comparisons.append(
                {
                    "comparison_id": comparison_id,
                    "metric": metric,
                    "treatment_arm": treatment_arm,
                    "control_arm": control_arm,
                    "stratum": stratum or "overall",
                    "independent_unit": "stream",
                    "independent_unit_count": len(stream_ids),
                    "bootstrap_resamples": draws,
                    "paired_differences": differences,
                    "estimate": interval["estimate"],
                    "ci95": interval["ci95"],
                    "ci95_lower": interval["ci95"][0],
                    "ci95_upper": interval["ci95"][1],
                }
            )
    return comparisons


def causal_summary_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Measure changed public selections, later decisions, chronology, and caps."""

    by_key = {
        (str(row["stream_id"]), str(row["arm"]), int(row["chronology_index"])): row for row in rows
    }
    stream_ids = sorted({str(row["stream_id"]) for row in rows})
    selection_changes = 0
    later_changes = 0
    for stream_id in stream_ids:
        change_boundaries = []
        for block in range(0, EVENTS_PER_STREAM, prototype.QUERY_BLOCK_SIZE):
            active = [
                index
                for index in range(block, block + prototype.QUERY_BLOCK_SIZE)
                if by_key[(stream_id, "active_recognition", index)]["query_selected"]
            ]
            random_query = [
                index
                for index in range(block, block + prototype.QUERY_BLOCK_SIZE)
                if by_key[(stream_id, "random_query_recognition", index)]["query_selected"]
            ]
            if active != random_query:
                selection_changes += 1
                change_boundaries.append(block + prototype.QUERY_BLOCK_SIZE - 1)
        if change_boundaries:
            first_later = min(change_boundaries) + 1
            later_changes += sum(
                int(
                    by_key[(stream_id, "active_recognition", index)]["prediction"]
                    != by_key[(stream_id, "random_query_recognition", index)]["prediction"]
                )
                for index in range(first_later, EVENTS_PER_STREAM)
            )
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["stream_id"]), str(row["arm"]))].append(row)
    pre_release = sum(
        int(row.get("same_event_correction") is True)
        + int(row.get("prediction_frozen_before_release") is not True)
        + int(row.get("held_out_label_visible_to_controller") is not False)
        for row in rows
    )
    cap_violations = sum(
        int(sum(int(row["query_selected"]) for row in group) > QUERY_CEILING)
        for group in groups.values()
    ) + sum(int(int(row["memory_total_bytes"]) > MEMORY_CAP_BYTES) for row in rows)
    return {
        "prospective_selection_change_count": selection_changes,
        "later_changed_prediction_count": later_changes,
        "pre_release_difference_count": pre_release,
        "cap_violation_count": cap_violations,
        "constraint_addition_count": sum(int(row["archive_admission_count"]) for row in rows),
        "constraint_deactivation_count": sum(int(row["archive_admission_count"]) for row in rows),
        "reactivation_count": sum(int(row["archive_reactivation_count"]) for row in rows),
    }


def _overall_comparison(
    comparisons: Sequence[Mapping[str, Any]], comparison_id: str
) -> Mapping[str, Any]:
    """Return the overall row for one frozen comparison identifier."""

    return next(
        row
        for row in comparisons
        if row["comparison_id"] == comparison_id and row["stratum"] == "overall"
    )


def score_acceptance_gates(
    comparisons: Sequence[Mapping[str, Any]], causal: Mapping[str, Any]
) -> dict[str, JsonDict]:
    """Score every frozen scientific gate without mixing in run completion."""

    future = _overall_comparison(comparisons, "future_error_vs_reset")
    false_accept = _overall_comparison(comparisons, "false_accept_vs_reset")
    recurrence_random = _overall_comparison(comparisons, "recurrence_error_vs_random")
    recurrence_shuffle = _overall_comparison(comparisons, "recurrence_error_vs_shuffle")
    recurrence_frozen = _overall_comparison(comparisons, "recurrence_degradation_vs_frozen")
    definitions = (
        (
            "future_error_vs_reset",
            "ci95_upper<0",
            future["ci95_upper"],
            float(future["ci95_upper"]) < 0,
        ),
        (
            "false_accept_vs_reset",
            "ci95_upper<=0",
            false_accept["ci95_upper"],
            float(false_accept["ci95_upper"]) <= 0,
        ),
        (
            "recurrence_error_vs_random",
            "ci95_upper<0",
            recurrence_random["ci95_upper"],
            float(recurrence_random["ci95_upper"]) < 0,
        ),
        (
            "recurrence_error_vs_shuffle",
            "ci95_upper<0",
            recurrence_shuffle["ci95_upper"],
            float(recurrence_shuffle["ci95_upper"]) < 0,
        ),
        (
            "recurrence_degradation_vs_frozen",
            "estimate<=0.02",
            recurrence_frozen["estimate"],
            float(recurrence_frozen["estimate"]) <= 0.02,
        ),
        (
            "prospective_causal_change",
            "selection_changes>0 and later_prediction_changes>0",
            [
                causal["prospective_selection_change_count"],
                causal["later_changed_prediction_count"],
            ],
            int(causal["prospective_selection_change_count"]) > 0
            and int(causal["later_changed_prediction_count"]) > 0,
        ),
        (
            "pre_release_isolation",
            "==0",
            causal["pre_release_difference_count"],
            int(causal["pre_release_difference_count"]) == 0,
        ),
        (
            "query_memory_limits",
            "==0",
            causal["cap_violation_count"],
            int(causal["cap_violation_count"]) == 0,
        ),
    )
    return {
        name: {
            "principle": "Keep this frozen value criterion separate from run completion.",
            "expected": expected,
            "observed": observed,
            "pass": passed,
            "passed": passed,
        }
        for name, expected, observed, passed in definitions
    }


def classify_result(gates: Mapping[str, Mapping[str, Any]]) -> tuple[int, str]:
    """Return circular value evidence or an honest completed scientific null."""

    value = int(all(gates[name]["passed"] is True for name in SCIENTIFIC_GATE_NAMES))
    return value, "circular_positive" if value else "null"


def run_e2e_controls(root: Path) -> tuple[list[JsonDict], int]:
    """Measure durable delayed commit, cold restart, rejection, and rollback."""

    root.mkdir(parents=True, exist_ok=True)
    masks = dict.fromkeys(prototype.FAMILIES, 1 << 0)
    controller = prototype.RecognitionController.from_masks(masks)
    controller.add_archive(dict.fromkeys(prototype.FAMILIES, 1 << 8))
    controller.add_archive(dict.fromkeys(prototype.FAMILIES, 1 << 16))
    event = {"event_id": "exp7268-e2e", "family_id": "lower_bound", "numeric_value": 12}
    before = controller.predict(event)
    selected, query = controller.select_request([event], [0])
    controller.record_query(selected, query, request_index=1, release_index=2)
    state_path = root / "controller.json"
    controller.save(state_path)
    parent_bytes = controller.state_bytes()
    release = prototype._control_release("exp7268-e2e", 12, "reject", 1)
    release["release_index"] = 2
    premature = False
    try:
        controller.commit_batch(
            [release],
            current_cycle=1,
            expected_parent_hash=controller.state_hash(),
            state_path=state_path,
        )
    except prototype.RecognitionCommitRejected as error:
        premature = str(error) == "future_release"
    durable_start = time.perf_counter_ns()
    receipt = controller.commit_batch(
        [release],
        current_cycle=2,
        expected_parent_hash=controller.state_hash(),
        state_path=state_path,
    )
    durable_ns = time.perf_counter_ns() - durable_start
    later = controller.predict(event)
    restored = prototype.RecognitionController.load(state_path)
    reload_ok = (
        restored.state_hash() == controller.state_hash() and restored.predict(event) == later
    )
    rollback = restored.rollback(receipt, state_path=state_path)
    rollback_ok = (
        rollback["byte_identical"] is True
        and restored.state_bytes() == parent_bytes == state_path.read_bytes()
    )
    stale = False
    try:
        restored.commit_batch(
            [prototype._control_release("stale", 0, "accept", 3)],
            current_cycle=3,
            expected_parent_hash="sha256:" + "0" * 64,
            state_path=state_path,
        )
    except prototype.RecognitionCommitRejected as error:
        stale = str(error) == "stale_parent"
    rows = [
        {"stage": "prediction_sealed", "passed": before == ("accept", 0.0)},
        {"stage": "delayed_feedback", "passed": premature},
        {"stage": "durable_commit", "passed": receipt["new_state_hash"] == controller.state_hash()},
        {"stage": "later_prediction", "passed": later != before},
        {"stage": "cold_restart", "passed": reload_ok},
        {"stage": "rollback", "passed": rollback_ok},
        {"stage": "stale_rejection", "passed": stale and state_path.read_bytes() == parent_bytes},
    ]
    return rows, durable_ns


def cost_summary(rows: Sequence[Mapping[str, Any]], durable_commit_ns: int) -> list[JsonDict]:
    """Reduce event costs and one measured durable commit into operation rows."""

    fields = {
        "lookup": "lookup_cost_ns",
        "query": "query_cost_ns",
        "update": "update_cost_ns",
        "validation": "validation_cost_ns",
        "full_event": "full_event_cost_ns",
    }
    result = []
    for operation, field in fields.items():
        values = [int(row.get(field, 0)) for row in rows]
        result.append(
            {
                "operation": operation,
                "unit": "ns",
                "count": len(values),
                "p50_ns": _percentile(values, 0.50),
                "p95_ns": _percentile(values, 0.95),
            }
        )
    memory = [int(row["memory_total_bytes"]) for row in rows]
    result.append(
        {
            "operation": "memory_bytes",
            "unit": "bytes",
            "count": len(memory),
            "p50": _percentile(memory, 0.50),
            "p95": _percentile(memory, 0.95),
            "maximum": max(memory),
        }
    )
    result.append(
        {
            "operation": "durable_commit",
            "unit": "ns",
            "count": 1,
            "p50_ns": durable_commit_ns,
            "p95_ns": durable_commit_ns,
        }
    )
    order = {operation: index for index, operation in enumerate(COST_OPERATIONS)}
    return sorted(result, key=lambda row: order[str(row["operation"])])


def _sample_budget(stream_ids: Sequence[str], complete: bool) -> JsonDict:
    """Declare the fixed units, label ceiling, censoring, and stopping rule."""

    return {
        "fixed_global_stream_count": STREAM_COUNT,
        "planned_stream_count": len(stream_ids),
        "attempted_stream_count": len(stream_ids),
        "completed_stream_count": len(stream_ids) if complete else 0,
        "censored_stream_count": 0 if complete else len(stream_ids),
        "arms_per_stream": len(ARMS),
        "events_per_stream": EVENTS_PER_STREAM,
        "planned_event_arm_rows": len(stream_ids) * len(ARMS) * EVENTS_PER_STREAM,
        "completed_event_arm_rows": len(stream_ids) * len(ARMS) * EVENTS_PER_STREAM
        if complete
        else 0,
        "query_ceiling_per_stream_arm": QUERY_CEILING,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "stopping_rule": "all sealed streams once; no outcome-based extension",
    }


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    stream_ids: Sequence[str],
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Create all required fields before blocked or measured classification."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": list(checks),
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_invoked": MODEL_INVOKED,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "current_inference_count": 0,
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "reducer_inference_substrate": REDUCER_SUBSTRATE,
        "reducer_inference_substrate_class": REDUCER_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "random_seed": {
            "experiment": RANDOM_SEED,
            "stream_seeds": list(prototype.STREAM_SEEDS[: len(stream_ids)]),
            "bootstrap": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": _sample_budget(stream_ids, False),
        "acceptance_gate_results": {},
        "gate_check_summary": _gate_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_prerequisite",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "recognition_run_complete_score": 0,
        "recognition_value_score": 0,
        "continuous_self_learning_task": True,
        "prequential_rows_path": None,
        "comparison_rows": [],
        "cost_rows": [],
        "no_model_weight_mutation": True,
        "default_pipeline_modified": False,
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every failed precondition and the first exact failure."""

    summary = prototype.gate_summary(checks)
    failed = [dict(row) for row in checks if row.get("passed") is not True]
    summary["failed_checks"] = failed
    return summary


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    stream_ids: Sequence[str],
    *,
    started_at: str | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    """Return row-free terminal evidence for an external prerequisite failure."""

    now = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks,
        source_hashes,
        stream_ids,
        started_at=started_at or now,
        completed_at=now,
        duration_s=duration_s,
    )
    failure = artifact["gate_check_summary"].get("first_failure") or {}
    artifact["honest_verdict"] = (
        "blocked_external_prerequisite: "
        f"{failure.get('upstream', 'unknown')}.{failure.get('field', 'unknown')} "
        f"observed={failure.get('observed_value')} expected={failure.get('expected_value')}"
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable code, input, configuration, raw evidence, gates, and receipts."""

    stable = deepcopy(dict(artifact))
    for field in ("started_at_utc", "completed_at_utc", "duration_s", "phase_spans_s"):
        stable.pop(field, None)
    stable["reproducibility_checksum"] = ""
    return transactional.sha256_json(stable)


def _receipt_matches(repo_root: Path, receipt: Mapping[str, Any]) -> bool:
    """Check that one declared sidecar hash matches current exact bytes."""

    path = receipt.get("path")
    expected = receipt.get("sha256")
    return bool(
        path and expected and prototype._sha256_path(_resolve(repo_root, str(path))) == expected
    )


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    progress: bool = False,
    bootstrap_draws: int = BOOTSTRAP_RESAMPLES,
    upstream_path: Path | None = None,
) -> JsonDict:
    """Authenticate, replay, reduce, run controls, and seal one terminal candidate."""

    selected = tuple(
        stream_ids or (f"prospective-{index + 1:02d}" for index in range(STREAM_COUNT))
    )
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: JsonDict = {}
    if progress:
        _progress(0, "start", "authenticate prototype, streams, ownership, and output paths")
    phase_start = time.monotonic()
    checks, source_hashes, upstream = collect_preconditions(
        repo_root, paths, upstream_path=upstream_path
    )
    spans["phase_0_preconditions"] = time.monotonic() - phase_start
    prototype._atomic_write(
        paths.provisional,
        transactional.canonical_json_bytes(
            {"schema": SCHEMA, "status": "in_progress", "phase": 0, "checks": checks}
        ),
    )
    if prototype.gate_summary(checks)["passed"] is not True:
        if progress:
            _progress(0, "end", "external precondition failed; measurement did not start")
        return build_blocked_artifact(
            checks,
            source_hashes,
            selected,
            started_at=started_at,
            duration_s=time.monotonic() - started,
        )
    if progress:
        _progress(0, "end", "all external preconditions passed")

    phase_start = time.monotonic()
    if progress:
        _progress(1, "start", "confirm that no model work is scheduled")
        print("phase 1 BEFORE model load: no model load scheduled", flush=True)
        print("phase 1 AFTER model load: attempted and completed loads remain zero", flush=True)
        print("phase 1 BEFORE generation: no generation call scheduled", flush=True)
        print("phase 1 AFTER generation: generation counters remain zero", flush=True)
    spans["phase_1_no_llm"] = time.monotonic() - phase_start
    if progress:
        _progress(1, "end", "MODEL_SPECS is empty and every current counter is zero")

    phase_start = time.monotonic()
    if progress:
        _progress(2, "start", "load the authenticated sealed stream views")
    views = load_sealed_views(repo_root, paths)
    spans["phase_2_stream_load"] = time.monotonic() - phase_start
    if progress:
        _progress(2, "end", "public, release, and authority bytes loaded separately")

    phase_start = time.monotonic()
    if progress:
        _progress(3, "start", "freeze arms, budgets, seeds, and recognition contract")
    configuration = {
        "stream_ids": list(selected),
        "arms": list(ARMS),
        "events_per_stream": EVENTS_PER_STREAM,
        "query_ceiling": QUERY_CEILING,
        "memory_cap_bytes": MEMORY_CAP_BYTES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": bootstrap_draws,
        "recognition_contract": upstream["recognition_contract"],
        "frozen_before_first_prospective_label": True,
    }
    configuration["sha256"] = transactional.sha256_json(configuration)
    spans["phase_3_contract_freeze"] = time.monotonic() - phase_start
    if progress:
        _progress(3, "end", f"configuration_sha256={configuration['sha256']}")

    phase_start = time.monotonic()
    if progress:
        _progress(4, "start", "BEFORE 24-stream eight-arm CPU benchmark")
        print("phase 4 BEFORE benchmark: exact sealed-controller replay", flush=True)
    panel = run_learning_panel(views, paths, stream_ids=selected, progress=progress)
    spans["phase_4_prequential_replay"] = time.monotonic() - phase_start
    if progress:
        print("phase 4 AFTER benchmark: all selected event-arm rows completed", flush=True)
        _progress(4, "end", f"completed_event_rows={len(panel.event_rows)}")

    phase_start = time.monotonic()
    if progress:
        _progress(5, "start", f"run {bootstrap_draws} whole-stream bootstrap resamples")
    comparisons = build_comparison_rows(panel.rows, draws=bootstrap_draws)
    causal = causal_summary_from_rows(panel.event_rows)
    scientific_gates = score_acceptance_gates(comparisons, causal)
    value_score, verdict_class = classify_result(scientific_gates)
    spans["phase_5_bootstrap_and_gates"] = time.monotonic() - phase_start
    if progress:
        _progress(5, "end", f"recognition_value_score={value_score}")

    phase_start = time.monotonic()
    if progress:
        _progress(6, "start", "run durable E2E and independent raw-row reduction")
    e2e_rows, durable_ns = run_e2e_controls(paths.e2e_sidecar.parent / "experiment_7268_e2e")
    if any(row["passed"] is not True for row in e2e_rows):
        raise ValueError("e2e_control_failed")
    prequential_receipt = prototype._atomic_write(
        paths.prequential_rows, prototype.jsonl_bytes(panel.event_rows)
    )
    lifecycle_receipt = prototype._atomic_write(
        paths.lifecycle_receipts, prototype.jsonl_bytes(panel.receipts)
    )
    e2e_receipt = prototype._atomic_write(
        paths.e2e_sidecar,
        transactional.canonical_json_bytes(
            {"schema": "carnot.exp7268.e2e.v1", "rows": e2e_rows, "durable_commit_ns": durable_ns}
        ),
    )
    evidence_receipt = prototype._atomic_write(
        paths.evidence_sidecar,
        transactional.canonical_json_bytes(
            {
                "schema": "carnot.exp7268.evidence.v1",
                "historical_model_receipts": {},
                "upstream_artifact": {
                    "path": str(DEFAULT_UPSTREAM_ARTIFACT),
                    "sha256": EXPECTED_UPSTREAM_SHA256,
                    "recognition_fixture_ready_score": upstream["recognition_fixture_ready_score"],
                },
                "supplied_change_oracle_rows": [],
            }
        ),
    )
    for receipt in (prequential_receipt, lifecycle_receipt, e2e_receipt, evidence_receipt):
        source_hashes[str(receipt["path"])] = str(receipt["sha256"])
    reduced = independent_reduce(paths.prequential_rows)
    if reduced != panel.rows:
        raise ValueError("independent_reducer_mismatch")
    costs = cost_summary(panel.event_rows, durable_ns)
    expected_rows = len(selected) * EVENTS_PER_STREAM * len(ARMS)
    run_complete = (
        len(panel.event_rows) == expected_rows
        and panel.completed_stream_ids == list(selected)
        and not panel.censored_stream_ids
    )
    receipt_kinds = {str(row["kind"]) for row in panel.receipts}
    completion_gates = {
        "complete_event_arm_matrix": {
            "principle": "Account for every fixed stream, arm, and event.",
            "expected": expected_rows,
            "observed": len(panel.event_rows),
            "pass": run_complete,
            "passed": run_complete,
        },
        "lifecycle_receipts": {
            "principle": "Retain every required online update receipt kind.",
            "expected": list(RECEIPT_KINDS),
            "observed": sorted(receipt_kinds),
            "pass": receipt_kinds == set(RECEIPT_KINDS),
            "passed": receipt_kinds == set(RECEIPT_KINDS),
        },
        "independent_raw_reducer": {
            "principle": "Cold JSONL reduction must reproduce every stream-arm row.",
            "expected": transactional.sha256_json(panel.rows),
            "observed": transactional.sha256_json(reduced),
            "pass": reduced == panel.rows,
            "passed": reduced == panel.rows,
        },
        "durable_e2e": {
            "principle": "Delayed commit, restart, rejection, and rollback must pass.",
            "expected": len(e2e_rows),
            "observed": sum(int(row["passed"] is True) for row in e2e_rows),
            "pass": all(row["passed"] is True for row in e2e_rows),
            "passed": all(row["passed"] is True for row in e2e_rows),
        },
    }
    run_complete_score = int(all(row["passed"] is True for row in completion_gates.values()))
    spans["phase_6_e2e_reduction_and_seal"] = time.monotonic() - phase_start
    if progress:
        _progress(6, "end", "durable controls, sidecar seals, and cold reduction passed")

    overall_full = _overall_comparison(comparisons, "future_error_vs_full_memory")
    overlap = next(
        row
        for row in comparisons
        if row["comparison_id"] == "future_error_vs_reset"
        and row["stratum"] == "overlapping_recurrence"
    )
    acceptance = {**scientific_gates, **completion_gates}
    completed_at = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks,
        source_hashes,
        selected,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=time.monotonic() - started,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "phase_spans_s": spans,
            "random_seed": {
                **artifact["random_seed"],
                "bootstrap_resamples": bootstrap_draws,
            },
            "rows": panel.rows,
            "sample_size_budget": {
                **_sample_budget(selected, run_complete),
                "bootstrap_resamples": bootstrap_draws,
            },
            "acceptance_gate_results": acceptance,
            "verdict_class": verdict_class if run_complete_score else "partial",
            "honest_verdict": (
                "complete_circular_positive: every frozen recognition value gate passed"
                if value_score and run_complete_score
                else "complete_null: autonomous recognition completed but one or more frozen value gates failed"
                if run_complete_score
                else "partial: task-owned measurement is incomplete and remains checkpointed"
            ),
            "recognition_run_complete_score": run_complete_score,
            "recognition_value_score": value_score if run_complete_score else 0,
            "prequential_rows_path": str(paths.prequential_rows),
            "prequential_rows_receipt": {
                **prequential_receipt,
                "row_count": len(panel.event_rows),
                "format": "jsonl",
            },
            "lifecycle_receipts_path": str(paths.lifecycle_receipts),
            "lifecycle_receipts_receipt": {
                **lifecycle_receipt,
                "row_count": len(panel.receipts),
                "receipt_kinds": sorted(receipt_kinds),
            },
            "comparison_rows": comparisons,
            "causal_summary": causal,
            "cost_rows": costs,
            "cost_methodology": "Measured event reducer costs plus amortized exact controller replay; durable commit is timed directly.",
            "e2e_sidecar_receipt": {**e2e_receipt, "row_count": len(e2e_rows)},
            "evidence_sidecar_receipt": evidence_receipt,
            "e2e_summary": {
                "row_count": len(e2e_rows),
                "passed_count": sum(int(row["passed"] is True) for row in e2e_rows),
                "failed_count": sum(int(row["passed"] is not True) for row in e2e_rows),
            },
            "frozen_configuration_receipt": configuration,
            "recognition_contract": upstream["recognition_contract"],
            "stream_manifest": upstream["stream_manifest"],
            "supplied_change_oracle_rows": [],
            "overall_headline_includes_strata": [
                "separated_recurrence",
                "overlapping_recurrence",
            ],
            "overlap_reuse_effectiveness": {
                "effective": float(overlap["ci95_upper"]) < 0,
                "comparison": overlap,
                "principle": "A nonnegative upper bound means overlap makes reuse ineffective.",
            },
            "full_memory_performance": {
                "rows": [row for row in panel.rows if row["arm"] == "full_version_space_memory"],
                "active_vs_full_memory": overall_full,
                "active_superiority_claimed": float(overall_full["ci95_upper"]) < 0,
            },
            "validation_receipts": [
                {
                    "command": f"independent_reduce {paths.prequential_rows}",
                    "exit_code": 0,
                    "classification": "passed",
                    "log_sha256": transactional.sha256_json(reduced),
                },
                {
                    "command": "run_e2e_controls",
                    "exit_code": 0,
                    "classification": "passed",
                    "log_sha256": transactional.sha256_json(e2e_rows),
                },
            ],
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(
        artifact,
        repo_root=repo_root,
        expected_stream_ids=selected,
        check_files=True,
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
    check_files: bool = False,
) -> list[str]:
    """Cold-check schema, rows, gates, receipts, costs, files, and classification."""

    selected = tuple(
        expected_stream_ids or (f"prospective-{index + 1:02d}" for index in range(STREAM_COUNT))
    )
    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(any(field not in artifact for field in REQUIRED_ARTIFACT_FIELDS), "required_fields")
    add(
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE,
        "identity",
    )
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    principles = artifact.get("field_principles", {})
    add(
        not isinstance(principles, Mapping)
        or any(field not in principles for field in REQUIRED_ARTIFACT_FIELDS),
        "field_principles",
    )
    add(
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or any(
            artifact.get(field) != 0
            for field in (
                "current_model_load_count",
                "current_generation_count",
                "current_inference_count",
            )
        ),
        "model_invocation",
    )
    add(artifact.get("execution_venue") != EXECUTION_VENUE, "execution_venue")
    add(artifact.get("verifier_is_oracle") is not True, "oracle_declaration")
    add(artifact.get("no_model_weight_mutation") is not True, "weight_mutation")
    add(artifact.get("default_pipeline_modified") is not False, "production_default")
    add(
        artifact.get("verdict_class")
        not in {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"},
        "verdict_class",
    )
    add(
        artifact.get("verdict_class") == "positive" and artifact.get("verifier_is_oracle") is True,
        "oracle_positive_forbidden",
    )
    add(
        artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
        "reproducibility_checksum",
    )
    if artifact.get("status") == "blocked":
        add(
            artifact.get("rows") != []
            or artifact.get("recognition_run_complete_score") != 0
            or artifact.get("recognition_value_score") != 0,
            "blocked_rows",
        )
        add(
            artifact.get("inference_substrate") != "blocked_no_run"
            or artifact.get("inference_substrate_class") != "blocked_no_run"
            or artifact.get("verdict_class") != "blocked"
            or not str(artifact.get("honest_verdict", "")).startswith("blocked_")
            or artifact.get("gate_check_summary", {}).get("passed") is not False,
            "blocked_contract",
        )
        return errors
    add(artifact.get("status") != "complete", "status")
    if artifact.get("status") != "complete":
        return errors
    add(
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
        "substrate",
    )
    add(artifact.get("gate_check_summary", {}).get("passed") is not True, "preconditions")
    rows = artifact.get("rows", [])
    expected_units = {(stream_id, arm) for stream_id in selected for arm in ARMS}
    add(
        not isinstance(rows, list)
        or {(row.get("stream_id"), row.get("arm")) for row in rows} != expected_units,
        "rows",
    )
    expected_events = len(selected) * EVENTS_PER_STREAM * len(ARMS)
    budget = artifact.get("sample_size_budget", {})
    raw_receipt = artifact.get("prequential_rows_receipt", {})
    add(
        budget.get("completed_event_arm_rows") != expected_events
        or raw_receipt.get("row_count") != expected_events
        or artifact.get("recognition_run_complete_score") != 1,
        "completion_score",
    )
    draws = int(artifact.get("random_seed", {}).get("bootstrap_resamples", BOOTSTRAP_RESAMPLES))
    comparisons = build_comparison_rows(rows, draws=draws)
    add(artifact.get("comparison_rows") != comparisons, "comparison_rows")
    scientific = score_acceptance_gates(comparisons, artifact.get("causal_summary", {}))
    recorded = artifact.get("acceptance_gate_results", {})
    add(
        any(recorded.get(name) != scientific.get(name) for name in SCIENTIFIC_GATE_NAMES),
        "acceptance_gate_results",
    )
    value, verdict = classify_result(scientific)
    add(
        artifact.get("recognition_value_score") != value
        or artifact.get("verdict_class") != verdict
        or not str(artifact.get("honest_verdict", "")).startswith(
            "complete_circular_positive:" if value else "complete_null:"
        ),
        "terminal_classification",
    )
    e2e = artifact.get("e2e_summary", {})
    add(
        e2e.get("row_count") != 7 or e2e.get("passed_count") != 7 or e2e.get("failed_count") != 0,
        "e2e_controls",
    )
    add(
        {row.get("operation") for row in artifact.get("cost_rows", [])} != set(COST_OPERATIONS),
        "cost_rows",
    )
    if check_files:
        receipts = (
            artifact.get("prequential_rows_receipt", {}),
            artifact.get("lifecycle_receipts_receipt", {}),
            artifact.get("e2e_sidecar_receipt", {}),
            artifact.get("evidence_sidecar_receipt", {}),
        )
        add(any(not _receipt_matches(repo_root, receipt) for receipt in receipts), "sidecar_hashes")
        raw_path = _resolve(repo_root, str(artifact["prequential_rows_path"]))
        raw_rows = prototype._read_jsonl(raw_path)
        add(prequential_row_errors(raw_rows) != [], "prequential_rows")
        add(reduce_prequential_rows(raw_rows) != rows, "independent_reducer")
        add(causal_summary_from_rows(raw_rows) != artifact.get("causal_summary"), "causal_summary")
        e2e_object = prototype._load_object(
            _resolve(repo_root, artifact["e2e_sidecar_receipt"]["path"])
        )
        expected_costs = cost_summary(raw_rows, int(e2e_object["durable_commit_ns"]))
        add(expected_costs != artifact.get("cost_rows"), "cost_summary")
        add(
            any(
                expected is None or prototype._sha256_path(_resolve(repo_root, path)) != expected
                for path, expected in artifact.get("source_artifact_hashes", {}).items()
            ),
            "source_artifact_hashes",
        )
    return errors


def attach_validation_receipts(
    artifact: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Attach actual validation commands and refresh the stable checksum."""

    updated = deepcopy(dict(artifact))
    updated["validation_receipts"] = [dict(row) for row in receipts]
    updated["reproducibility_checksum"] = reproducibility_checksum(updated)
    return updated


def write_artifact(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
) -> None:
    """Cold-validate and atomically publish complete terminal bytes."""

    errors = validate_artifact(
        artifact,
        repo_root=repo_root,
        expected_stream_ids=expected_stream_ids,
        check_files=True,
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    prototype._atomic_write(path, transactional.canonical_json_bytes(dict(artifact)))


def _command_receipt(command: Sequence[str]) -> JsonDict:
    """Stream one validation command and keep its exact combined log hash."""

    command_text = " ".join(command)
    print(f"validation BEFORE subprocess: {command_text}", flush=True)
    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{REPO_ROOT / 'python'}:{REPO_ROOT}",
            "COVERAGE_FILE": "/tmp/.coverage-carnot-exp7268",
        },
    )
    output = []
    if process.stdout is not None:
        for line in process.stdout:
            output.append(line)
            print(line, end="", flush=True)
    exit_code = process.wait()
    combined = "".join(output)
    print(
        f"validation AFTER subprocess: exit_code={exit_code} "
        f"elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )
    return {
        "command": command_text,
        "exit_code": exit_code,
        "classification": "passed" if exit_code == 0 else "failed",
        "log_sha256": transactional.sha256_bytes(combined.encode()),
    }


def _validation_commands(candidate: Path) -> list[list[str]]:
    """Return focused coverage, style, type, spec, adversarial, and row checks."""

    python = str(REPO_ROOT / ".venv/bin/python")
    coverage = str(REPO_ROOT / ".venv/bin/coverage")
    ruff = str(REPO_ROOT / ".venv/bin/ruff")
    mypy = str(REPO_ROOT / ".venv/bin/mypy")
    module = "python/carnot/experiment_7268_v639_recognition_learning.py"
    test = str(TEST_PATH)
    wrapper = str(WRAPPER_PATH)
    return [
        [coverage, "erase"],
        [
            coverage,
            "run",
            f"--include={REPO_ROOT / module}",
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-n",
            "0",
            "--basetemp=/tmp/carnot-exp7268-final",
            test,
            "-q",
        ],
        [
            coverage,
            "report",
            f"--include={REPO_ROOT / module}",
            "--show-missing",
            "--fail-under=100",
        ],
        [ruff, "check", module, test, wrapper],
        [ruff, "format", "--check", module, test, wrapper],
        [mypy, module, wrapper],
        [python, "scripts/check_spec_coverage.py", test],
        [python, "scripts/adversarial_verify.py", "--json", str(candidate)],
        [python, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
    ]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed execution date and optional private result root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure, validate, and atomically publish the recognition result."""

    print("phase 0 immediate: Exp7268 recognition learning started", flush=True)
    args = _parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"run_date_must_be_{RUN_DATE}")
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    artifact = build_and_seal(REPO_ROOT, paths, progress=True)
    if artifact["status"] == "blocked":
        write_artifact(paths.artifact, artifact, repo_root=REPO_ROOT)
        _progress(7, "end", f"wrote blocked terminal artifact {paths.artifact}")
        return 0
    _progress(7, "start", "write measured terminal candidate under raw evidence")
    prototype._atomic_write(paths.terminal_candidate, transactional.canonical_json_bytes(artifact))
    _progress(7, "end", f"candidate={paths.terminal_candidate}")
    _progress(8, "start", "BEFORE focused tests, coverage, lint, type, and artifact checks")
    receipts = list(artifact["validation_receipts"])
    receipts.extend(
        _command_receipt(command) for command in _validation_commands(paths.terminal_candidate)
    )
    artifact = attach_validation_receipts(artifact, receipts)
    prototype._atomic_write(
        paths.provisional,
        transactional.canonical_json_bytes(
            {"schema": SCHEMA, "status": "in_progress", "phase": 8, "validation_receipts": receipts}
        ),
    )
    failed = [row for row in receipts if row["exit_code"] != 0]
    if failed:
        raise RuntimeError(
            "focused_validation_failed:" + ",".join(row["command"] for row in failed)
        )
    _progress(8, "end", "AFTER all focused validations passed")
    _progress(9, "start", "BEFORE final cold validation and atomic terminal write")
    write_artifact(paths.artifact, artifact, repo_root=REPO_ROOT)
    _progress(9, "end", f"AFTER atomic terminal write {paths.artifact}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin wrapper owns execution.
    raise SystemExit(main())
