"""Audit active-recognition learning from authenticated raw evidence.

The audit accepts a complete learning run even when its value score is zero.
It reconstructs evidence in a cold CPU process and keeps audit completion
separate from scientific promotion.

Spec refs: REQ-CL-7269 and SCENARIO-CL-7269-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import socket
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_7267_v639_recognition_prototype as prototype
from carnot import experiment_7268_v639_recognition_learning as learning
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7269
SCHEMA = "carnot.exp7269.v639_recognition_audit.v1"
MILESTONE = "2026.09.639"
RUN_DATE = "20260913"
AUDIT_SEED = 7_269_000
BOOTSTRAP_SEED = learning.BOOTSTRAP_SEED
BOOTSTRAP_DRAWS = learning.BOOTSTRAP_RESAMPLES
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
RESULT_PREFIX = "CARNOT_RESULT_JSON="

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
WRAPPER_PATH = Path("scripts/experiments/experiment_7269_v639_recognition_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7269_v639_recognition_audit.py")
DEFAULT_UPSTREAM_ARTIFACT = Path("results/experiment_7268_v639_recognition_learning.json")
DEFAULT_RAW_ROWS = Path("results/raw/experiment_7268/prequential_rows.jsonl")
DEFAULT_LIFECYCLE_ROWS = Path("results/raw/experiment_7268/lifecycle_receipts.jsonl")
DEFAULT_ARTIFACT = Path("results/experiment_7269_v639_recognition_audit.json")
EXPECTED_UPSTREAM_SHA256 = "sha256:5ec52b42298a73e3454fce47e03f34118a1c348f72f81a1ee768673c4cef8d16"
SCENARIO_PATTERN = re.compile(r"SCENARIO-CL-7269-[A-Z-]+")
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7255_v638_coverage_audit.py"),
    Path("python/carnot/experiment_7267_v639_recognition_prototype.py"),
    Path("python/carnot/experiment_7268_v639_recognition_learning.py"),
    Path("python/carnot/experiment_7269_v639_recognition_audit.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    WRAPPER_PATH,
    TEST_PATH,
    SPEC_PATH,
)
REQUIRED_RAW_FIELDS = {
    "unit_id",
    "stream_id",
    "seed",
    "stratum",
    "arm",
    "event_id",
    "chronology_index",
    "prediction",
    "prediction_energy",
    "later_released_label",
    "full_denominator_error",
    "false_accept",
    "abstention",
    "recurrence_eligible",
    "query_selected",
    "query_release_index",
    "released_query_count_before",
    "released_query_count_after",
    "archive_admission_count",
    "archive_reactivation_count",
    "selection_change_count",
    "memory_total_bytes",
    "retained_constraint_count",
    "prediction_frozen_before_release",
    "held_out_label_visible_to_controller",
    "same_event_correction",
    "controller_input_fields",
    "prediction_seal_sha256",
    "full_event_cost_ns",
}
REPLAY_FIELDS = (
    "prediction",
    "prediction_energy",
    "full_denominator_error",
    "false_accept",
    "abstention",
    "recurrence_eligible",
    "query_selected",
    "released_query_count_before",
    "released_query_count_after",
    "archive_admission_count",
    "archive_reactivation_count",
    "selection_change_count",
    "memory_total_bytes",
)
AUDIT_SAFETY_GATE_NAMES = (
    "complete_stream_arm_matrix",
    "independent_raw_checks",
    "released_only_replay",
    "prospective_intervention",
    "label_memory_bounds",
    "mutation_rejections",
    "e2e_lifecycle",
    "cold_process",
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
    "recognition_audit_complete_score",
    "recognition_promotion_score",
    "intervention_rows",
    "rollback_rows",
)
FIELD_PRINCIPLES = {
    "schema": "Version the result; retain ordinary top-level experiment_id and milestone.",
    "experiment_id": "Bind the evidence to Exp7269.",
    "milestone": "Bind the evidence to milestone 2026.09.639.",
    "status": "Use complete or blocked only for terminal work; checkpoints stay separate.",
    "run_date": "Use 20260913 with actual UTC start and end timestamps.",
    "started_at_utc": "Record the actual UTC invocation start.",
    "completed_at_utc": "Record the actual UTC audit end.",
    "field_principles": "Store explanations here; top-level fields keep ordinary values.",
    "preconditions_checked": "Retain input hashes, ownership, and failures before work.",
    "MODEL_SPECS": "Declare current executable models; this invocation has none.",
    "model_invoked": "Derive model use from actual calls, including failed calls.",
    "invocation_counts": "Separate attempted and completed loads and generations.",
    "inference_substrate": "Use the recognized literal for the CPU audit.",
    "inference_substrate_class": "Declare actual compute without time padding.",
    "execution_venue": "Use host for host orchestration.",
    "duration_s": "Measure monotonic time and disjoint phase spans.",
    "random_seed": "Freeze stream and bootstrap seeds before outcomes.",
    "reproducibility_checksum": "Bind code, inputs, configuration, and raw findings.",
    "source_artifact_hashes": "Authenticate exact producer, raw, stream, and source bytes.",
    "rows": "Retain every independent stream-arm metric and censoring state.",
    "sample_size_budget": "Record planned, attempted, completed, and censored units.",
    "acceptance_gate_results": "Retain expected, observed, passed, and principle per gate.",
    "gate_check_summary": "For a block, name the exact upstream check and values.",
    "verifier_is_oracle": "Expose exact evaluator authority; conformance is not learned truth.",
    "honest_verdict": "Use complete_ for findings and blocked_ for external absence.",
    "verdict_class": "Use the closed class; oracle authority forbids positive.",
    "validation_receipts": "Record actual command, exit code, classification, and log hash.",
    "recognition_audit_complete_score": "One means the independent audit completed, even for a null.",
    "recognition_promotion_score": "One requires every reconstructed science and safety gate.",
    "intervention_rows": "Join changed nomination identity to later changed predictions.",
    "rollback_rows": "Retain byte-preserving rejection and cold rollback evidence.",
}

gate_check = prototype.gate_check
gate_summary = prototype.gate_summary


class AuditEvidenceError(ValueError):
    """Reject malformed evidence before it can change an audit verdict."""


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep provisional, raw, control, candidate, and terminal bytes separate."""

    checkpoint: Path
    raw_summary: Path
    mutation_sidecar: Path
    e2e_sidecar: Path
    terminal_candidate: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Use the fixed task-owned paths below the repository results directory."""

        return cls.from_results_root(REPO_ROOT / "results")

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put test-owned outputs below one caller-owned result directory."""

        return cls.from_results_root(root)

    @classmethod
    def from_results_root(cls, root: Path) -> ExperimentPaths:
        """Derive all output paths without creating success-shaped bytes."""

        raw = root / "raw" / "experiment_7269"
        checkpoints = root / "checkpoints"
        return cls(
            checkpoints / "experiment_7269_v639_in_progress.json",
            raw / "cold_reduction.json",
            raw / "mutation_receipts.json",
            raw / "e2e_receipts.json",
            raw / "terminal_candidate.json",
            root / DEFAULT_ARTIFACT.name,
        )


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Print one flushed watchdog line at a phase boundary."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _resolve(repo_root: Path, path: str | Path) -> Path:
    """Resolve repository-relative evidence while preserving absolute paths."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _sha256_path(path: Path) -> str | None:
    """Hash exact bytes while keeping an absent file distinct from empty content."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object; malformed or absent evidence stays unavailable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating evidence early."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _task_identity(text: str) -> JsonDict:
    """Extract only the fixed Exp7269 block from the executable roadmap."""

    match = re.search(r"(?ms)^- id: exp7269-recognition-audit\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(1)
    milestone = re.search(r"(?m)^  milestone: (.+)$", block)
    deliverable = re.search(r"(?m)^  deliverable: (.+)$", block)
    return {
        "id": "exp7269-recognition-audit" if block else None,
        "milestone": None if milestone is None else milestone.group(1).strip(),
        "deliverable": None if deliverable is None else deliverable.group(1).strip(),
    }


def _receipt_matches(repo_root: Path, receipt: Any) -> bool:
    """Require one declared path and hash to match its current exact bytes."""

    if not isinstance(receipt, Mapping) or not isinstance(receipt.get("path"), str):
        return False
    return _sha256_path(_resolve(repo_root, str(receipt["path"]))) == receipt.get("sha256")


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    upstream_path: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict]:
    """Authenticate completion, exact raw inputs, code, streams, and output owners."""

    upstream_file = _resolve(repo_root, upstream_path or DEFAULT_UPSTREAM_ARTIFACT)
    upstream = _load_object(upstream_file)
    source_hashes = {
        str(_resolve(repo_root, path)): _sha256_path(_resolve(repo_root, path))
        for path in SOURCE_PATHS
    }
    source_hashes[str(upstream_file)] = _sha256_path(upstream_file)
    evidence_receipts = {
        "prequential_rows": upstream.get("prequential_rows_receipt", {}),
        "lifecycle_receipts": upstream.get("lifecycle_receipts_receipt", {}),
        "e2e_sidecar": upstream.get("e2e_sidecar_receipt", {}),
        "evidence_sidecar": upstream.get("evidence_sidecar_receipt", {}),
    }
    stream_receipts = upstream.get("stream_manifest", {}).get("receipts", {})
    for receipt in (*evidence_receipts.values(), *stream_receipts.values()):
        if isinstance(receipt, Mapping) and isinstance(receipt.get("path"), str):
            source_hashes[str(_resolve(repo_root, str(receipt["path"])))] = _sha256_path(
                _resolve(repo_root, str(receipt["path"]))
            )
    try:
        specification = (repo_root / SPEC_PATH).read_text(encoding="utf-8")
        roadmap = (repo_root / "research-roadmap.yaml").read_text(encoding="utf-8")
        exclusions = (repo_root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    except OSError:
        specification = roadmap = exclusions = ""
    producer_paths = (
        repo_root / "python/carnot/experiment_7267_v639_recognition_prototype.py",
        repo_root / "python/carnot/experiment_7268_v639_recognition_learning.py",
        repo_root / "scripts/experiments/experiment_7268_v639_recognition_learning.py",
    )
    producer_expected = {
        str(path): upstream.get("source_artifact_hashes", {}).get(str(path))
        for path in producer_paths
    }
    producer_observed = {str(path): _sha256_path(path) for path in producer_paths}
    writable = {
        field: _path_writable(getattr(paths, field))
        for field in (
            "checkpoint",
            "raw_summary",
            "mutation_sidecar",
            "e2e_sidecar",
            "terminal_candidate",
            "artifact",
        )
    }
    current_uid = os.getuid()
    owned: dict[str, bool] = {}
    for field in writable:
        parent = getattr(paths, field).parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        owned[field] = parent.is_dir() and parent.stat().st_uid == current_uid
    checksum_valid = False
    try:
        checksum_valid = learning.reproducibility_checksum(upstream) == upstream.get(
            "reproducibility_checksum"
        )
    except (KeyError, TypeError, ValueError):
        pass
    quarantine = prototype.exp7213.quarantine_state(
        upstream,
        exclusions,
        upstream_file.name,
        "exp7268-recognition-learning",
    )
    checks = [
        gate_check(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-CL-7269",
            True,
            "REQ-CL-7269" in specification,
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7269-*",
            7,
            len(set(SCENARIO_PATTERN.findall(specification))),
        ),
        gate_check(
            "v639_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            {
                "id": "exp7269-recognition-audit",
                "milestone": MILESTONE,
                "deliverable": str(DEFAULT_ARTIFACT),
            },
            _task_identity(roadmap),
        ),
        gate_check(
            "exp7268_artifact_hash",
            str(upstream_file),
            "sha256",
            EXPECTED_UPSTREAM_SHA256,
            _sha256_path(upstream_file),
        ),
        gate_check(
            "exp7268_terminal_state",
            "exp7268-recognition-learning",
            "status",
            "complete",
            upstream.get("status"),
        ),
        gate_check(
            "exp7268_run_complete",
            "exp7268-recognition-learning",
            "recognition_run_complete_score",
            1,
            upstream.get("recognition_run_complete_score"),
        ),
        gate_check(
            "exp7268_checksum",
            "exp7268-recognition-learning",
            "reproducibility_checksum",
            True,
            checksum_valid,
        ),
        gate_check(
            "exp7268_not_quarantined",
            "artifact_and_ops/exclusion_manifest.yaml",
            "quarantined",
            False,
            quarantine["quarantined"],
        ),
        gate_check(
            "producer_code_hashes",
            "exp7268.source_artifact_hashes",
            "controller_and_learning_code",
            producer_expected,
            producer_observed,
        ),
        gate_check(
            "raw_and_sidecar_receipts",
            "exp7268",
            "prequential,lifecycle,e2e,evidence",
            dict.fromkeys(evidence_receipts, True),
            {
                name: _receipt_matches(repo_root, receipt)
                for name, receipt in evidence_receipts.items()
            },
        ),
        gate_check(
            "sealed_stream_receipts",
            "exp7268.stream_manifest",
            "receipts",
            dict.fromkeys(stream_receipts, True),
            {
                name: _receipt_matches(repo_root, receipt)
                for name, receipt in stream_receipts.items()
            },
        ),
        gate_check(
            "writable_output_paths", "host", "paths", dict.fromkeys(writable, True), writable
        ),
        gate_check("output_parent_ownership", "host", "uid", dict.fromkeys(owned, True), owned),
    ]
    return checks, source_hashes, upstream


def _selected_raw_rows(path: Path, stream_ids: Sequence[str]) -> list[JsonDict]:
    """Read selected strict rows while missing fields remain errors, not zeros."""

    wanted = set(stream_ids)
    rows: list[JsonDict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise AuditEvidenceError(f"invalid_jsonl:{line_number}") from error
                if not isinstance(value, dict):
                    raise AuditEvidenceError(f"non_object_jsonl:{line_number}")
                if str(value.get("stream_id")) not in wanted:
                    continue
                missing = sorted(REQUIRED_RAW_FIELDS - set(value))
                if missing:
                    raise AuditEvidenceError("missing_raw_fields:" + ",".join(missing))
                rows.append(value)
    except OSError as error:
        raise AuditEvidenceError(f"raw_rows_unavailable:{path}") from error
    if not rows:
        raise AuditEvidenceError("raw_rows_unavailable")
    return rows


def _percentile(values: Sequence[float], probability: float) -> float:
    """Use the frozen deterministic nearest-rank percentile."""

    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(probability * len(ordered))))
    return float(ordered[index])


def _reduce_stream_arm(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Rebuild one independent unit directly from chronological event evidence."""

    ordered = sorted(rows, key=lambda row: int(row["chronology_index"]))
    future = [row for row in ordered if int(row["chronology_index"]) >= prototype.WARMUP_COUNT]
    recurrence = [row for row in future if row["recurrence_eligible"] is True]
    correct = [
        int(row["chronology_index"])
        for row in recurrence
        if int(row["full_denominator_error"]) == 0
    ]
    future_count = len(future)
    recurrence_count = len(recurrence)
    return {
        "unit_id": ordered[0]["unit_id"],
        "stream_id": ordered[0]["stream_id"],
        "seed": ordered[0]["seed"],
        "stratum": ordered[0]["stratum"],
        "arm": ordered[0]["arm"],
        "metric": "prospective_full_denominator_error",
        "event_count": len(ordered),
        "future_event_count": future_count,
        "future_error": sum(int(row["full_denominator_error"]) for row in future),
        "false_accept": sum(int(row["false_accept"]) for row in future),
        "abstention": sum(int(row["abstention"]) for row in future),
        "recurrence_event_count": recurrence_count,
        "recurrence_error": sum(int(row["full_denominator_error"]) for row in recurrence),
        "query_count": sum(int(row["query_selected"]) for row in ordered),
        "release_count": int(ordered[-1]["released_query_count_after"]),
        "archive_admission_count": sum(int(row["archive_admission_count"]) for row in ordered),
        "reactivation_count": sum(int(row["archive_reactivation_count"]) for row in ordered),
        "selection_change_count": sum(int(row["selection_change_count"]) for row in ordered),
        "maximum_memory_bytes": max(int(row["memory_total_bytes"]) for row in ordered),
        "censored": False,
        "future_error_rate": sum(int(row["full_denominator_error"]) for row in future)
        / future_count,
        "false_accept_rate": sum(int(row["false_accept"]) for row in future) / future_count,
        "coverage_rate": 1.0 - sum(int(row["abstention"]) for row in future) / future_count,
        "recurrence_error_rate": sum(int(row["full_denominator_error"]) for row in recurrence)
        / recurrence_count,
        "recurrence_recovery_delay_events": correct[0] - 768 if correct else recurrence_count,
        "retained_constraint_count": max(int(row["retained_constraint_count"]) for row in ordered),
        "full_event_cost_p50_ns": _percentile(
            [float(row["full_event_cost_ns"]) for row in ordered], 0.50
        ),
        "full_event_cost_p95_ns": _percentile(
            [float(row["full_event_cost_ns"]) for row in ordered], 0.95
        ),
    }


def _reduce_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Preserve each stream as an independent unit and keep frozen arm order."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["stream_id"]), str(row["arm"]))].append(row)
    arm_order = {arm: index for index, arm in enumerate(learning.ARMS)}
    return [
        _reduce_stream_arm(group)
        for _, group in sorted(
            grouped.items(), key=lambda item: (item[0][0], arm_order[item[0][1]])
        )
    ]


def _raw_check_rows(rows: Sequence[Mapping[str, Any]], stream_id: str) -> list[JsonDict]:
    """Check matrix completeness, prediction seals, authority, and fixed bounds."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm"])].append(row)
    complete = set(grouped) == set(learning.ARMS) and all(
        sorted(int(row["chronology_index"]) for row in group)
        == list(range(prototype.EVENTS_PER_STREAM))
        for group in grouped.values()
    )
    authority = all(
        row["controller_input_fields"] == ["event_id", "family_id", "numeric_value"]
        and row["held_out_label_visible_to_controller"] is False
        and row["prediction_frozen_before_release"] is True
        and row["same_event_correction"] is False
        for row in rows
    )
    seals = all(
        row["prediction_seal_sha256"]
        == transactional.sha256_json(
            [
                row["stream_id"],
                row["arm"],
                row["event_id"],
                row["prediction"],
                row["released_query_count_before"],
            ]
        )
        for row in rows
    )
    bounds = all(
        int(row["memory_total_bytes"]) <= learning.MEMORY_CAP_BYTES for row in rows
    ) and all(
        sum(int(row["query_selected"]) for row in group) <= learning.QUERY_CEILING
        for group in grouped.values()
    )
    return [
        {"stream_id": stream_id, "check": "complete_event_arm_matrix", "passed": complete},
        {"stream_id": stream_id, "check": "authority_and_release_isolation", "passed": authority},
        {"stream_id": stream_id, "check": "prediction_seals", "passed": seals},
        {"stream_id": stream_id, "check": "label_and_memory_bounds", "passed": bounds},
    ]


def build_intervention_rows(
    rows: Sequence[Mapping[str, Any]], *, block_size: int = prototype.QUERY_BLOCK_SIZE
) -> list[JsonDict]:
    """Join changed query or association identity to predictions after release."""

    by_stream: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["arm"] in {
            "active_recognition",
            "random_query_recognition",
            "shuffled_archive_association",
        }:
            by_stream[str(row["stream_id"])].append(row)
    result: list[JsonDict] = []
    for stream_id, stream_rows in sorted(by_stream.items()):
        by_key = {(str(row["arm"]), int(row["chronology_index"])): row for row in stream_rows}
        maximum_index = max(int(row["chronology_index"]) for row in stream_rows)
        stratum = str(stream_rows[0]["stratum"])
        for offset in range(0, maximum_index + 1, block_size):
            for comparison, control in (
                ("active_vs_random", "random_query_recognition"),
                ("active_vs_shuffled", "shuffled_archive_association"),
            ):
                active_queries = [
                    str(by_key[("active_recognition", index)]["event_id"])
                    for index in range(offset, min(offset + block_size, maximum_index + 1))
                    if by_key[("active_recognition", index)]["query_selected"] is True
                ]
                control_queries = [
                    str(by_key[(control, index)]["event_id"])
                    for index in range(offset, min(offset + block_size, maximum_index + 1))
                    if by_key[(control, index)]["query_selected"] is True
                ]
                release_values = [
                    int(row["query_release_index"])
                    for row in (
                        by_key[(arm, index)]
                        for arm in ("active_recognition", control)
                        for index in range(offset, min(offset + block_size, maximum_index + 1))
                    )
                    if row["query_selected"] is True and row["query_release_index"] is not None
                ]
                release_boundary = max(release_values, default=offset + block_size - 1)
                later_indices = range(release_boundary + 1, maximum_index + 1)
                changed_indices = [
                    index
                    for index in later_indices
                    if by_key[("active_recognition", index)]["prediction"]
                    != by_key[(control, index)]["prediction"]
                ]
                association_changes = (
                    sum(
                        int(by_key[(control, index)]["selection_change_count"] > 0)
                        for index in range(release_boundary + 1, maximum_index + 1)
                    )
                    if control == "shuffled_archive_association"
                    else 0
                )
                changed_identity = active_queries != control_queries
                if changed_identity or association_changes or changed_indices:
                    result.append(
                        {
                            "unit_id": f"{stream_id}:{offset}:{comparison}",
                            "stream_id": stream_id,
                            "stratum": stratum,
                            "block_start": offset,
                            "comparison": comparison,
                            "active_query_event_ids": active_queries,
                            "control_query_event_ids": control_queries,
                            "changed_nomination_identity": changed_identity,
                            "changed_association_count": association_changes,
                            "release_boundary": release_boundary,
                            "later_changed_prediction_count": len(changed_indices),
                            "first_later_changed_prediction_index": (
                                changed_indices[0] if changed_indices else None
                            ),
                            "release_precedes_effect": all(
                                index > release_boundary for index in changed_indices
                            ),
                            "censored": False,
                        }
                    )
    return result


def build_bound_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Retain label, memory, constraint, false-accept, and overlap-recall values."""

    bounds = []
    overlap = []
    for row in rows:
        bound = {
            "unit_id": row["unit_id"],
            "stream_id": row["stream_id"],
            "arm": row["arm"],
            "stratum": row["stratum"],
            "label_budget_used": int(row["query_count"]),
            "label_budget_cap": learning.QUERY_CEILING,
            "memory_bytes": int(row["maximum_memory_bytes"]),
            "memory_byte_cap": learning.MEMORY_CAP_BYTES,
            "retained_constraint_count": int(row["retained_constraint_count"]),
            "false_accept_count": int(row["false_accept"]),
            "false_accept_rate": float(row["false_accept_rate"]),
            "passed": int(row["query_count"]) <= learning.QUERY_CEILING
            and int(row["maximum_memory_bytes"]) <= learning.MEMORY_CAP_BYTES,
        }
        bounds.append(bound)
        if row["stratum"] == "overlapping_recurrence":
            overlap.append(
                {
                    **bound,
                    "recurrence_event_count": int(row["recurrence_event_count"]),
                    "recurrence_error_count": int(row["recurrence_error"]),
                    "recurrence_recall": 1.0 - float(row["recurrence_error_rate"]),
                }
            )
    return bounds, overlap


def _replay_stream(
    views: prototype.StreamViews,
    raw_rows: Sequence[Mapping[str, Any]],
    stream_id: str,
) -> JsonDict:
    """Replay shipped controllers and compare only learner-visible decisions."""

    replay = prototype.run_recognition_panel(views, stream_ids=(stream_id,), progress=False)
    expected = {(str(row["arm"]), int(row["chronology_index"])): row for row in raw_rows}
    mismatch_counts = dict.fromkeys(REPLAY_FIELDS, 0)
    for row in replay.event_rows:
        source = expected[(str(row["arm"]), int(row["chronology_index"]))]
        for field in REPLAY_FIELDS:
            mismatch_counts[field] += int(row[field] != source[field])
    return {
        "stream_id": stream_id,
        "event_arm_count": len(replay.event_rows),
        "controller_input_fields": ["event_id", "family_id", "numeric_value"],
        "future_label_used": False,
        "private_regime_used": False,
        "mismatch_counts": mismatch_counts,
        "passed": len(replay.event_rows) == len(raw_rows) and not any(mismatch_counts.values()),
    }


def _load_views(repo_root: Path) -> prototype.StreamViews:
    """Cold-load the authenticated public, release, and authority stream views."""

    root = repo_root / "results/streams/experiment_7267"
    manifest = _load_object(root / "stream_manifest.json")["prospective"]
    views = prototype.StreamViews(
        prototype._read_jsonl(root / "prospective_public.jsonl"),
        prototype._read_jsonl(root / "prospective_private_authority.jsonl"),
        prototype._read_jsonl(root / "prospective_releases.jsonl"),
        manifest,
    )
    errors = prototype.stream_conformance_errors(views, "prospective")
    if errors:
        raise AuditEvidenceError("sealed_stream_conformance:" + ",".join(errors))
    return views


def _lifecycle_check(
    repo_root: Path,
    stream_ids: Sequence[str],
    expected: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Rebuild lifecycle receipts from rows and compare their exact ordered bytes."""

    wanted = set(stream_ids)
    observed = [
        row
        for row in prototype._read_jsonl(repo_root / DEFAULT_LIFECYCLE_ROWS)
        if str(row.get("stream_id")) in wanted
    ]
    expected_hash = transactional.sha256_json(list(expected))
    observed_hash = transactional.sha256_json(observed)
    return {
        "check": "lifecycle_receipt_reconstruction",
        "expected_count": len(expected),
        "observed_count": len(observed),
        "expected_sha256": expected_hash,
        "observed_sha256": observed_hash,
        "receipt_kinds": sorted({str(row.get("kind")) for row in observed}),
        "passed": len(expected) == len(observed) and expected_hash == observed_hash,
    }


def _audit_raw_evidence_impl(
    repo_root: Path,
    *,
    stream_ids: Sequence[str],
    bootstrap_draws: int,
    progress: bool,
) -> JsonDict:
    """Perform one bounded cold reconstruction without producer aggregates."""

    wanted = tuple(stream_ids)
    rows = _selected_raw_rows(repo_root / DEFAULT_RAW_ROWS, wanted)
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["stream_id"])].append(row)
    if set(grouped) != set(wanted):
        raise AuditEvidenceError("selected_streams_incomplete")
    views = _load_views(repo_root)
    reduced: list[JsonDict] = []
    interventions: list[JsonDict] = []
    raw_checks: list[JsonDict] = []
    replay_rows: list[JsonDict] = []
    expected_lifecycle: list[JsonDict] = []
    causal = {
        "prospective_selection_change_count": 0,
        "later_changed_prediction_count": 0,
        "pre_release_difference_count": 0,
        "cap_violation_count": 0,
        "constraint_addition_count": 0,
        "constraint_deactivation_count": 0,
        "reactivation_count": 0,
    }
    started = time.monotonic()
    for index, stream_id in enumerate(wanted, start=1):
        stream_rows = grouped[stream_id]
        checks = _raw_check_rows(stream_rows, stream_id)
        if any(row["passed"] is not True for row in checks):
            raise AuditEvidenceError(f"raw_check_failed:{stream_id}")
        stream_reduced = _reduce_rows(stream_rows)
        if len(stream_reduced) != len(learning.ARMS):
            raise AuditEvidenceError(f"stream_arm_matrix:{stream_id}")
        reduced.extend(stream_reduced)
        interventions.extend(build_intervention_rows(stream_rows))
        raw_checks.extend(checks)
        replay_rows.append(_replay_stream(views, stream_rows, stream_id))
        for row in stream_rows:
            expected_lifecycle.extend(
                learning._lifecycle_rows(row, str(row["prediction_seal_sha256"]))
            )
        for key, value in learning.causal_summary_from_rows(stream_rows).items():
            causal[key] += int(value)
        if progress:
            print(
                f"cold reducer completed_units={index}/{len(wanted)} "
                f"elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
    lifecycle = _lifecycle_check(repo_root, wanted, expected_lifecycle)
    raw_checks.append(lifecycle)
    producer = _load_object(repo_root / DEFAULT_UPSTREAM_ARTIFACT)
    producer_rows = [row for row in producer.get("rows", []) if row.get("stream_id") in set(wanted)]
    producer_match = reduced == producer_rows
    raw_checks.append(
        {
            "check": "producer_summary_reconstruction",
            "expected_sha256": transactional.sha256_json(producer_rows),
            "observed_sha256": transactional.sha256_json(reduced),
            "passed": producer_match,
        }
    )
    if not producer_match or any(row["passed"] is not True for row in replay_rows):
        raise AuditEvidenceError("cold_reconstruction_mismatch")
    comparisons = learning.build_comparison_rows(reduced, draws=bootstrap_draws)
    science_gates = learning.score_acceptance_gates(comparisons, causal)
    bound_rows, overlap_rows = build_bound_rows(reduced)
    return {
        "rows": reduced,
        "comparison_rows": comparisons,
        "science_gates": science_gates,
        "causal_summary": causal,
        "intervention_rows": interventions,
        "bound_rows": bound_rows,
        "overlap_rows": overlap_rows,
        "raw_check_rows": raw_checks,
        "replay_rows": replay_rows,
        "lifecycle_summary": lifecycle,
        "process_receipt": {
            "fresh_process": os.environ.get("CARNOT_EXP7269_COLD_WORKER") == "1",
            "gpu_disabled": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "network_cache_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
            "no_model_load": MODEL_SPECS == [] and MODEL_INVOKED is False,
            "inference_substrate": REDUCER_SUBSTRATE,
            "inference_substrate_class": REDUCER_SUBSTRATE_CLASS,
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
        },
    }


def _spawn_worker(command: Sequence[str], label: str, timeout_s: float = 1200.0) -> JsonDict:
    """Stream a bounded subprocess and emit truthful heartbeats while it runs."""

    print(f"BEFORE subprocess: {label}", flush=True)
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
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CARNOT_EXP7269_COLD_WORKER": "1",
        },
    )
    if process.stdout is None:  # pragma: no cover - Popen guarantees the requested pipe.
        raise RuntimeError("subprocess_stdout_unavailable")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    result: JsonDict | None = None
    while process.poll() is None:
        if time.monotonic() - started > timeout_s:
            process.kill()
            raise TimeoutError(f"subprocess_timeout:{label}")
        ready = selector.select(timeout=30.0)
        if not ready:
            print(
                f"subprocess heartbeat: {label} elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )
            continue
        line = process.stdout.readline()
        if line.startswith(RESULT_PREFIX):
            value = json.loads(line[len(RESULT_PREFIX) :])
            if not isinstance(value, dict):
                raise RuntimeError(f"subprocess_non_object:{label}")
            result = value
        elif line:
            print(line, end="", flush=True)
    for line in process.stdout:
        if line.startswith(RESULT_PREFIX):
            value = json.loads(line[len(RESULT_PREFIX) :])
            result = value if isinstance(value, dict) else None
        else:
            print(line, end="", flush=True)
    exit_code = process.wait()
    print(
        f"AFTER subprocess: {label} exit_code={exit_code} "
        f"elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )
    if exit_code != 0 or result is None:
        raise RuntimeError(f"subprocess_failed:{label}:exit_code={exit_code}")
    return result


def _spawn_audit_worker(stream_ids: Sequence[str], bootstrap_draws: int) -> JsonDict:
    """Start the raw reducer in an isolated interpreter process."""

    return _spawn_worker(
        [
            sys.executable,
            "-I",
            str(REPO_ROOT / WRAPPER_PATH),
            "--date",
            RUN_DATE,
            "--audit-worker",
            "--stream-ids",
            ",".join(stream_ids),
            "--bootstrap-draws",
            str(bootstrap_draws),
        ],
        "Exp7269 cold raw-row reduction and released-only replay",
    )


def audit_raw_evidence(
    repo_root: Path,
    *,
    stream_ids: Sequence[str] | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    progress: bool = False,
) -> JsonDict:
    """Run the public raw audit API in a cold process unless already isolated."""

    selected = tuple(
        stream_ids or (f"prospective-{index + 1:02d}" for index in range(learning.STREAM_COUNT))
    )
    if os.environ.get("CARNOT_EXP7269_COLD_WORKER") == "1":
        return _audit_raw_evidence_impl(
            repo_root,
            stream_ids=selected,
            bootstrap_draws=bootstrap_draws,
            progress=progress,
        )
    if repo_root != REPO_ROOT:
        raise AuditEvidenceError("cold_worker_requires_repository_root")
    return _spawn_audit_worker(selected, bootstrap_draws)


def _visibility_mutation(root: Path, field: str) -> JsonDict:
    """Prove that one private query field is rejected without state changes."""

    controller = prototype.RecognitionController()
    controller.add_archive(dict.fromkeys(prototype.FAMILIES, 1 << 0))
    controller.add_archive(dict.fromkeys(prototype.FAMILIES, 1 << 8))
    path = root / f"{field}.json"
    controller.save(path)
    parent = controller.state_bytes()
    observed = None
    try:
        controller.select_request(
            [
                {
                    "event_id": field,
                    "family_id": "lower_bound",
                    "numeric_value": 7,
                    field: "private",
                }
            ],
            [0],
        )
    except ValueError as error:
        observed = str(error)
    preserved = controller.state_bytes() == parent == path.read_bytes()
    return {
        "mutation": "future_label" if field == "exact_label" else "regime_id",
        "expected_rejection": "private_authority_in_query",
        "observed_rejection": observed,
        "rejected": observed == "private_authority_in_query",
        "prior_bytes_preserved": preserved,
        "passed": observed == "private_authority_in_query" and preserved,
    }


def run_mutation_controls(root: Path) -> list[JsonDict]:
    """Reject five isolated recognition attacks and preserve exact parent bytes."""

    root.mkdir(parents=True, exist_ok=True)
    rows = [_visibility_mutation(root, "exact_label"), _visibility_mutation(root, "regime_id")]
    controller = prototype.RecognitionController.from_masks(
        dict.fromkeys(prototype.FAMILIES, 1 << 0)
    )
    path = root / "transaction.json"
    controller.save(path)
    release = prototype._control_release("duplicate", 0, "accept", 1)
    controller.commit_batch(
        [release], current_cycle=1, expected_parent_hash=controller.state_hash(), state_path=path
    )
    child = controller.state_bytes()
    for name, candidate, expected_parent, expected_error in (
        ("duplicate_release", release, controller.state_hash(), "duplicate_release"),
        (
            "stale_parent",
            prototype._control_release("stale", 0, "reject", 2),
            "sha256:" + "0" * 64,
            "stale_parent",
        ),
    ):
        observed = None
        try:
            controller.commit_batch(
                [candidate],
                current_cycle=2,
                expected_parent_hash=expected_parent,
                state_path=path,
            )
        except prototype.RecognitionCommitRejected as error:
            observed = str(error)
        preserved = controller.state_bytes() == child == path.read_bytes()
        rows.append(
            {
                "mutation": name,
                "expected_rejection": expected_error,
                "observed_rejection": observed,
                "rejected": observed == expected_error,
                "prior_bytes_preserved": preserved,
                "passed": observed == expected_error and preserved,
            }
        )
    archive_controller = prototype.RecognitionController()
    archive_controller.add_archive(dict.fromkeys(prototype.FAMILIES, 1 << 8))
    archive_path = root / "corrupted_archive.json"
    archive_controller.save(archive_path)
    archive_parent = archive_controller.state_bytes()
    corrupted = archive_controller.state_dict()
    corrupted["archives"][0]["state_hash"] = "sha256:" + "0" * 64
    observed = None
    try:
        prototype.RecognitionController.from_state(corrupted)
    except ValueError as error:
        observed = str(error)
    preserved = archive_controller.state_bytes() == archive_parent == archive_path.read_bytes()
    rows.append(
        {
            "mutation": "corrupted_archive",
            "expected_rejection": "archive_identity",
            "observed_rejection": observed,
            "rejected": observed == "archive_identity",
            "prior_bytes_preserved": preserved,
            "passed": observed == "archive_identity" and preserved,
        }
    )
    return rows


def run_e2e_controls(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Audit prediction, accepted change, rejection, reload, use, and rollback."""

    root.mkdir(parents=True, exist_ok=True)
    controller = prototype.RecognitionController.from_masks(
        dict.fromkeys(prototype.FAMILIES, 1 << 0)
    )
    controller.add_archive(dict.fromkeys(prototype.FAMILIES, 1 << 8))
    controller.add_archive(dict.fromkeys(prototype.FAMILIES, 1 << 16))
    event = {"event_id": "exp7269-e2e", "family_id": "lower_bound", "numeric_value": 12}
    before = controller.predict(event)
    selected, query = controller.select_request([event], [0])
    controller.record_query(selected, query, request_index=1, release_index=2)
    state_path = root / "controller.json"
    controller.save(state_path)
    parent = controller.state_bytes()
    parent_hash = controller.state_hash()
    release = prototype._control_release("exp7269-e2e", 12, "reject", 1)
    release["release_index"] = 2
    receipt = controller.commit_batch(
        [release], current_cycle=2, expected_parent_hash=parent_hash, state_path=state_path
    )
    child = controller.state_bytes()
    later = controller.predict(event)
    observed = None
    try:
        controller.commit_batch(
            [release],
            current_cycle=2,
            expected_parent_hash=controller.state_hash(),
            state_path=state_path,
        )
    except prototype.RecognitionCommitRejected as error:
        observed = str(error)
    rejection_preserved = controller.state_bytes() == child == state_path.read_bytes()
    restored = prototype.RecognitionController.load(state_path)
    reload_parity = (
        restored.state_hash() == controller.state_hash() and restored.predict(event) == later
    )
    rollback = restored.rollback(receipt, state_path=state_path)
    rollback_ok = (
        rollback["byte_identical"] is True
        and restored.state_bytes() == parent == state_path.read_bytes()
    )
    rows = [
        {
            "control": "prediction_before_release",
            "passed": before == ("accept", 0.0) and query["future_label_used"] is False,
        },
        {
            "control": "accepted_commit_changes_state",
            "parent_hash": parent_hash,
            "child_hash": receipt["new_state_hash"],
            "passed": child != parent and receipt["parent_hash"] == parent_hash,
        },
        {
            "control": "invalid_commit_preserves_state",
            "observed_rejection": observed,
            "passed": observed == "duplicate_release" and rejection_preserved,
        },
        {"control": "cold_reload_parity", "passed": reload_parity},
        {"control": "later_prediction_observes_commit", "passed": later != before},
        {"control": "rollback_restores_parent", "passed": rollback_ok},
    ]
    rollback_rows = [
        {
            "control": "rollback_restores_parent",
            "parent_bytes_preserved": rollback_ok,
            "cold_restore_exercised": True,
            "passed": rollback_ok,
        }
    ]
    return rows, rollback_rows


def derive_terminal_scores(
    audit_complete: bool,
    safety_passed: bool,
    science_gates: Mapping[str, Mapping[str, Any]],
) -> tuple[int, int, str, str]:
    """Keep completed review separate from favorable scientific value."""

    complete = int(audit_complete)
    science_passed = all(
        science_gates.get(name, {}).get("passed") is True for name in learning.SCIENTIFIC_GATE_NAMES
    )
    promotion = int(bool(complete) and safety_passed and science_passed)
    if not complete:
        return 0, 0, "partial", "partial: task-owned audit work is incomplete"
    if promotion:
        return (
            1,
            1,
            "circular_positive",
            "complete_circular_positive: recognition audit and every promotion gate passed",
        )
    return (
        1,
        0,
        "null",
        "complete_null: recognition audit completed but promotion gates did not all pass",
    )


def _sample_budget(stream_ids: Sequence[str], complete: bool) -> JsonDict:
    """Declare the fixed all-stream stopping rule and completed units."""

    count = len(stream_ids)
    event_rows = count * len(learning.ARMS) * learning.EVENTS_PER_STREAM
    return {
        "fixed_global_stream_count": learning.STREAM_COUNT,
        "planned_stream_count": count,
        "attempted_stream_count": count,
        "completed_stream_count": count if complete else 0,
        "censored_stream_count": 0 if complete else count,
        "arms_per_stream": len(learning.ARMS),
        "events_per_stream": learning.EVENTS_PER_STREAM,
        "planned_event_arm_rows": event_rows,
        "completed_event_arm_rows": event_rows if complete else 0,
        "query_ceiling_per_stream_arm": learning.QUERY_CEILING,
        "bootstrap_resamples": BOOTSTRAP_DRAWS,
        "stopping_rule": "all sealed streams once; no outcome-based extension",
    }


def _summary_with_failures(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every precondition and the first exact failed observation."""

    summary = gate_summary(checks)
    failures = [dict(row) for row in checks if row.get("passed") is not True]
    summary["failed_checks"] = failures
    summary["first_failure"] = failures[0] if failures else None
    return summary


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    stream_ids: Sequence[str],
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Create every required top-level field before terminal classification."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in checks],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "current_inference_count": 0,
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "reducer_inference_substrate": REDUCER_SUBSTRATE,
        "reducer_inference_substrate_class": REDUCER_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "execution_host": socket.gethostname(),
        "duration_s": duration_s,
        "phase_spans_s": {},
        "random_seed": {
            "audit": AUDIT_SEED,
            "bootstrap": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_DRAWS,
            "stream_seeds": list(prototype.STREAM_SEEDS[: len(stream_ids)]),
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": _sample_budget(stream_ids, False),
        "acceptance_gate_results": {},
        "gate_check_summary": _summary_with_failures(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_prerequisite",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "recognition_audit_complete_score": 0,
        "recognition_promotion_score": 0,
        "intervention_rows": [],
        "rollback_rows": [],
        "comparison_rows": [],
        "causal_summary": {},
        "raw_check_rows": [],
        "replay_rows": [],
        "bound_rows": [],
        "overlap_rows": [],
        "mutation_rows": [],
        "e2e_rows": [],
        "continuous_self_learning_task": True,
        "no_model_weight_mutation": True,
        "default_pipeline_modified": False,
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    upstream: Mapping[str, Any],
    stream_ids: Sequence[str],
    *,
    started_at: str | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    """Build row-free terminal evidence for an external prerequisite failure."""

    del upstream
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
    """Bind stable code, inputs, configuration, raw evidence, gates, and controls."""

    stable = deepcopy(dict(artifact))
    for field in (
        "started_at_utc",
        "completed_at_utc",
        "duration_s",
        "phase_spans_s",
        "execution_host",
    ):
        stable.pop(field, None)
    stable["reproducibility_checksum"] = ""
    return transactional.sha256_json(stable)


def _write_evidence(path: Path, value: Mapping[str, Any]) -> JsonDict:
    """Write task-owned checkpoint or raw evidence through one atomic rename."""

    return prototype._atomic_write(path, transactional.canonical_json_bytes(dict(value)))


def _audit_gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Use one explicit shape for every completion or safety criterion."""

    return {
        "expected": expected,
        "observed": observed,
        "pass": passed,
        "passed": passed,
        "principle": principle,
    }


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    progress: bool = False,
    upstream_path: Path | None = None,
) -> JsonDict:
    """Authenticate, cold-reduce, attack, score, and seal one terminal object."""

    selected = tuple(
        stream_ids or (f"prospective-{index + 1:02d}" for index in range(learning.STREAM_COUNT))
    )
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: JsonDict = {}
    phase_start = time.monotonic()
    if progress:
        _progress(0, "start", "authenticate producer, raw evidence, streams, code, and outputs")
    checks, source_hashes, upstream = collect_preconditions(
        repo_root, paths, upstream_path=upstream_path
    )
    spans["phase_0_preconditions"] = time.monotonic() - phase_start
    _write_evidence(
        paths.checkpoint,
        {
            "schema": "carnot.exp7269.checkpoint.v1",
            "status": "in_progress",
            "phase": 0,
            "started_at_utc": started_at,
            "completed_units": 0,
            "planned_units": len(selected),
            "checks": checks,
        },
    )
    if gate_summary(checks)["passed"] is not True:
        if progress:
            _progress(0, "end", "external prerequisite failed; audit did not start")
        return build_blocked_artifact(
            checks,
            source_hashes,
            upstream,
            selected,
            started_at=started_at,
            duration_s=time.monotonic() - started,
        )
    if progress:
        _progress(0, "end", "all external preconditions passed")

    phase_start = time.monotonic()
    if progress:
        _progress(1, "start", "confirm no current model work")
        print("phase 1 BEFORE model load: no model load scheduled", flush=True)
        print("phase 1 AFTER model load: all load counters remain zero", flush=True)
        print("phase 1 BEFORE generation: no generation call scheduled", flush=True)
        print("phase 1 AFTER generation: all generation counters remain zero", flush=True)
    spans["phase_1_no_llm"] = time.monotonic() - phase_start
    if progress:
        _progress(1, "end", "MODEL_SPECS=[] and current invocation counters are zero")

    phase_start = time.monotonic()
    if progress:
        _progress(2, "start", "BEFORE cold 24-stream eight-arm raw benchmark")
        print("phase 2 BEFORE benchmark: cold raw reduction and controller replay", flush=True)
    worker = _spawn_audit_worker(selected, bootstrap_draws)
    spans["phase_2_cold_reduction"] = time.monotonic() - phase_start
    if progress:
        print("phase 2 AFTER benchmark: cold raw reduction returned", flush=True)
        _progress(2, "end", f"completed_units={len(worker['rows'])} stream_arm_rows")
    raw_receipt = _write_evidence(paths.raw_summary, worker)
    source_hashes[str(paths.raw_summary)] = str(raw_receipt["sha256"])

    phase_start = time.monotonic()
    if progress:
        _progress(3, "start", "recompute science gates, bounds, overlap recall, and interventions")
    science_gates = worker["science_gates"]
    interventions = worker["intervention_rows"]
    effective_interventions = [
        row
        for row in interventions
        if row["changed_nomination_identity"] is True
        and int(row["later_changed_prediction_count"]) > 0
        and row["release_precedes_effect"] is True
    ]
    bounds_passed = all(row["passed"] is True for row in worker["bound_rows"])
    spans["phase_3_science_and_intervention"] = time.monotonic() - phase_start
    if progress:
        _progress(
            3,
            "end",
            f"effective_interventions={len(effective_interventions)} overlap_rows={len(worker['overlap_rows'])}",
        )

    phase_start = time.monotonic()
    if progress:
        _progress(4, "start", "run five byte-preserving mutations")
    mutation_rows = run_mutation_controls(paths.mutation_sidecar.parent / "mutation_state")
    mutation_receipt = _write_evidence(
        paths.mutation_sidecar,
        {
            "schema": "carnot.exp7269.mutations.v1",
            "historical_model_receipts": {},
            "current_model_fixture_count": 0,
            "rows": mutation_rows,
        },
    )
    source_hashes[str(paths.mutation_sidecar)] = str(mutation_receipt["sha256"])
    spans["phase_4_mutations"] = time.monotonic() - phase_start
    if progress:
        _progress(4, "end", f"completed_mutations={len(mutation_rows)}")

    phase_start = time.monotonic()
    if progress:
        _progress(5, "start", "apply E2E-007 lifecycle principles to recognition memory")
    e2e_rows, rollback_rows = run_e2e_controls(paths.e2e_sidecar.parent / "e2e_state")
    e2e_receipt = _write_evidence(
        paths.e2e_sidecar,
        {
            "schema": "carnot.exp7269.e2e.v1",
            "scope": "new finite recognition memory; Exp1659 was not rerun",
            "rows": e2e_rows,
            "rollback_rows": rollback_rows,
        },
    )
    source_hashes[str(paths.e2e_sidecar)] = str(e2e_receipt["sha256"])
    spans["phase_5_e2e"] = time.monotonic() - phase_start
    if progress:
        _progress(5, "end", f"completed_controls={len(e2e_rows)}")

    phase_start = time.monotonic()
    if progress:
        _progress(6, "start", "separate independent audit completion from promotion")
    raw_passed = all(row["passed"] is True for row in worker["raw_check_rows"])
    replay_passed = all(row["passed"] is True for row in worker["replay_rows"])
    mutations_passed = all(row["passed"] is True for row in mutation_rows)
    e2e_passed = all(row["passed"] is True for row in e2e_rows)
    cold_passed = all(
        worker["process_receipt"].get(field) is True
        for field in ("fresh_process", "gpu_disabled", "network_cache_offline", "no_model_load")
    )
    expected_units = len(selected) * len(learning.ARMS)
    matrix_passed = len(worker["rows"]) == expected_units
    intervention_passed = bool(effective_interventions)
    safety_gates = {
        "complete_stream_arm_matrix": _audit_gate(
            expected_units,
            len(worker["rows"]),
            matrix_passed,
            "Reconstruct every fixed stream-arm unit from raw rows.",
        ),
        "independent_raw_checks": _audit_gate(
            len(worker["raw_check_rows"]),
            sum(int(row["passed"] is True) for row in worker["raw_check_rows"]),
            raw_passed,
            "Missing fields, chronology, lifecycle, and seals must fail closed.",
        ),
        "released_only_replay": _audit_gate(
            len(selected),
            sum(int(row["passed"] is True) for row in worker["replay_rows"]),
            replay_passed,
            "Replay choices with public events and then-released feedback only.",
        ),
        "prospective_intervention": _audit_gate(
            ">0 joined changed nominations and later predictions",
            len(effective_interventions),
            intervention_passed,
            "Require prospective intervention changes, not development-only changes.",
        ),
        "label_memory_bounds": _audit_gate(
            "zero cap violations",
            sum(int(row["passed"] is not True) for row in worker["bound_rows"]),
            bounds_passed,
            "Recompute label and byte limits while retaining zero-effect rows.",
        ),
        "mutation_rejections": _audit_gate(
            5,
            sum(int(row["passed"] is True) for row in mutation_rows),
            mutations_passed and len(mutation_rows) == 5,
            "All five invalid changes must preserve prior bytes.",
        ),
        "e2e_lifecycle": _audit_gate(
            6,
            sum(int(row["passed"] is True) for row in e2e_rows),
            e2e_passed and len(e2e_rows) == 6,
            "Accepted state, rejection, reload, later use, and rollback must pass.",
        ),
        "cold_process": _audit_gate(
            True,
            cold_passed,
            cold_passed,
            "The reducer must run in a fresh offline CPU process without a model load.",
        ),
    }
    audit_complete = matrix_passed and raw_passed and len(worker["replay_rows"]) == len(selected)
    safety_passed = all(row["passed"] is True for row in safety_gates.values())
    scores = derive_terminal_scores(audit_complete, safety_passed, science_gates)
    spans["phase_6_terminal_scoring"] = time.monotonic() - phase_start
    if progress:
        _progress(6, "end", f"audit_complete={scores[0]} promotion={scores[1]}")
    if not audit_complete:
        raise AuditEvidenceError("task_owned_audit_incomplete")

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
            "rows": worker["rows"],
            "sample_size_budget": {
                **_sample_budget(selected, True),
                "bootstrap_resamples": bootstrap_draws,
            },
            "acceptance_gate_results": {**science_gates, **safety_gates},
            "gate_check_summary": _summary_with_failures(checks),
            "honest_verdict": scores[3],
            "verdict_class": scores[2],
            "recognition_audit_complete_score": scores[0],
            "recognition_promotion_score": scores[1],
            "intervention_rows": interventions,
            "rollback_rows": rollback_rows,
            "comparison_rows": worker["comparison_rows"],
            "causal_summary": worker["causal_summary"],
            "raw_check_rows": worker["raw_check_rows"],
            "replay_rows": worker["replay_rows"],
            "bound_rows": worker["bound_rows"],
            "overlap_rows": worker["overlap_rows"],
            "mutation_rows": mutation_rows,
            "e2e_rows": e2e_rows,
            "raw_reducer_process_receipt": worker["process_receipt"],
            "raw_summary_receipt": raw_receipt,
            "mutation_sidecar_receipt": mutation_receipt,
            "e2e_sidecar_receipt": e2e_receipt,
            "upstream_recognition_run_complete_score": upstream["recognition_run_complete_score"],
            "upstream_recognition_value_score": upstream.get("recognition_value_score"),
            "historical_e2e_scope": "E2E-007 principles only; Exp1659 was not rerun",
            "validation_receipts": [
                {
                    "command": "cold Exp7269 raw-row reducer and released-only replay",
                    "exit_code": 0,
                    "classification": "passed",
                    "log_sha256": str(raw_receipt["sha256"]),
                },
                {
                    "command": "run_e2e_controls for finite recognition memory",
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
        raise AuditEvidenceError("artifact_validation_failed:" + ",".join(errors))
    return artifact


def _receipt_error(receipt: Mapping[str, Any]) -> bool:
    """Reject validation receipts that do not contain actual command evidence."""

    return not (
        isinstance(receipt.get("command"), str)
        and bool(receipt.get("command"))
        and isinstance(receipt.get("exit_code"), int)
        and isinstance(receipt.get("classification"), str)
        and bool(receipt.get("classification"))
        and isinstance(receipt.get("log_sha256"), str)
        and str(receipt.get("log_sha256")).startswith("sha256:")
    )


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
    check_files: bool = False,
) -> list[str]:
    """Cold-check terminal schema, reconstruction, scores, controls, and files."""

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
        or artifact.get("invocation_counts") != INVOCATION_COUNTS
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
    add(artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact), "checksum")
    receipts = artifact.get("validation_receipts", [])
    add(
        not isinstance(receipts, list)
        or any(not isinstance(row, Mapping) or _receipt_error(row) for row in receipts),
        "validation_receipt_schema",
    )
    status = artifact.get("status")
    add(status not in {"complete", "blocked"}, "status")
    if status not in {"complete", "blocked"}:
        return errors
    if status == "blocked":
        add(
            artifact.get("verdict_class") != "blocked"
            or artifact.get("rows") != []
            or artifact.get("recognition_audit_complete_score") != 0
            or artifact.get("recognition_promotion_score") != 0
            or artifact.get("gate_check_summary", {}).get("passed") is not False
            or not str(artifact.get("honest_verdict", "")).startswith("blocked_"),
            "blocked_contract",
        )
        return errors
    add(
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
        "substrate",
    )
    add(artifact.get("gate_check_summary", {}).get("passed") is not True, "preconditions")
    rows = artifact.get("rows", [])
    selected = tuple(
        expected_stream_ids
        or sorted({str(row.get("stream_id")) for row in rows if isinstance(row, Mapping)})
    )
    expected_units = {(stream_id, arm) for stream_id in selected for arm in learning.ARMS}
    add(
        not isinstance(rows, list)
        or {(row.get("stream_id"), row.get("arm")) for row in rows} != expected_units,
        "rows",
    )
    draws = int(artifact.get("random_seed", {}).get("bootstrap_resamples", BOOTSTRAP_DRAWS))
    comparisons = learning.build_comparison_rows(rows, draws=draws)
    add(artifact.get("comparison_rows") != comparisons, "comparison_rows")
    science = learning.score_acceptance_gates(comparisons, artifact.get("causal_summary", {}))
    recorded = artifact.get("acceptance_gate_results", {})
    add(
        any(recorded.get(name) != science.get(name) for name in learning.SCIENTIFIC_GATE_NAMES),
        "science_gates",
    )
    safety_passed = all(
        recorded.get(name, {}).get("passed") is True for name in AUDIT_SAFETY_GATE_NAMES
    )
    scores = derive_terminal_scores(True, safety_passed, science)
    add(artifact.get("recognition_audit_complete_score") != scores[0], "audit_score")
    add(artifact.get("recognition_promotion_score") != scores[1], "promotion_score")
    add(artifact.get("verdict_class") != scores[2], "verdict_class")
    add(artifact.get("honest_verdict") != scores[3], "honest_verdict")
    add(artifact.get("verdict_class") == "positive", "oracle_positive_forbidden")
    add(len(artifact.get("mutation_rows", [])) != 5, "mutation_rows")
    add(len(artifact.get("e2e_rows", [])) != 6, "e2e_rows")
    add(len(artifact.get("rollback_rows", [])) != 1, "rollback_rows")
    add(
        len(artifact.get("bound_rows", [])) != len(selected) * len(learning.ARMS),
        "bound_rows",
    )
    expected_overlap = sum(int(index > 12) for index in range(1, len(selected) + 1)) * len(
        learning.ARMS
    )
    if selected == tuple(f"prospective-{index + 1:02d}" for index in range(len(selected))):
        add(len(artifact.get("overlap_rows", [])) != expected_overlap, "overlap_rows")
    add(not artifact.get("intervention_rows"), "intervention_rows")
    if check_files:
        for field in ("raw_summary_receipt", "mutation_sidecar_receipt", "e2e_sidecar_receipt"):
            add(not _receipt_matches(repo_root, artifact.get(field, {})), "sidecar_hashes")
        add(
            any(
                expected is None or _sha256_path(_resolve(repo_root, path)) != expected
                for path, expected in artifact.get("source_artifact_hashes", {}).items()
            ),
            "source_artifact_hashes",
        )
        raw_summary = _load_object(
            _resolve(repo_root, str(artifact.get("raw_summary_receipt", {}).get("path", "")))
        )
        add(raw_summary.get("rows") != rows, "raw_summary_rows")
        add(
            raw_summary.get("intervention_rows") != artifact.get("intervention_rows"),
            "raw_summary_interventions",
        )
    return errors


def attach_validation_receipts(
    artifact: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Attach actual validation commands and refresh the stable checksum."""

    if any(_receipt_error(row) for row in receipts):
        raise ValueError("validation_receipt_schema")
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
) -> JsonDict:
    """Cold-validate and publish terminal bytes through one atomic rename."""

    errors = validate_artifact(
        artifact,
        repo_root=repo_root,
        expected_stream_ids=expected_stream_ids,
        check_files=artifact.get("status") == "complete",
    )
    if errors:
        raise AuditEvidenceError("artifact_validation_failed:" + ",".join(errors))
    return prototype._atomic_write(path, transactional.canonical_json_bytes(dict(artifact)))


def _command_receipt(command: Sequence[str]) -> JsonDict:
    """Run one unbuffered validation subprocess and hash its combined log."""

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
            "COVERAGE_FILE": "/tmp/.coverage-carnot-exp7269",
        },
    )
    if process.stdout is None:  # pragma: no cover - Popen guarantees the requested pipe.
        raise RuntimeError("validation_stdout_unavailable")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    output: list[str] = []
    while process.poll() is None:
        ready = selector.select(timeout=30.0)
        if not ready:
            print(
                f"validation heartbeat: elapsed_s={time.monotonic() - started:.1f} "
                f"command={command_text}",
                flush=True,
            )
            continue
        line = process.stdout.readline()
        if line:
            output.append(line)
            print(line, end="", flush=True)
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
    """Return only focused coverage, affected suites, static checks, and artifact checks."""

    python = str(REPO_ROOT / ".venv/bin/python")
    coverage = str(REPO_ROOT / ".venv/bin/coverage")
    ruff = str(REPO_ROOT / ".venv/bin/ruff")
    mypy = str(REPO_ROOT / ".venv/bin/mypy")
    module = "python/carnot/experiment_7269_v639_recognition_audit.py"
    wrapper = str(WRAPPER_PATH)
    test = str(TEST_PATH)
    affected = [
        "tests/python/test_experiment_7267_v639_recognition_prototype.py",
        "tests/python/test_experiment_7268_v639_recognition_learning.py",
    ]
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
            "--basetemp=/tmp/carnot-exp7269-coverage",
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
        [
            python,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-n",
            "0",
            "--basetemp=/tmp/carnot-exp7269-affected",
            *affected,
            "-q",
        ],
        [ruff, "check", module, test, wrapper],
        [ruff, "format", "--check", module, test, wrapper],
        [mypy, module, wrapper],
        [python, "scripts/check_spec_coverage.py", test, *affected],
        [
            python,
            "-u",
            wrapper,
            "--date",
            RUN_DATE,
            "--e2e-worker",
            "--output-root",
            "/tmp/carnot-exp7269-e2e-validation",
        ],
        [
            python,
            "-u",
            wrapper,
            "--date",
            RUN_DATE,
            "--validate",
            "--artifact-path",
            str(candidate),
        ],
        [python, "scripts/adversarial_verify.py", "--json", str(candidate)],
        [python, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
    ]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed date plus private worker and validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stream-ids", default="")
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--audit-worker", action="store_true")
    parser.add_argument("--e2e-worker", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--artifact-path", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the no-LLM audit and publish only validated terminal evidence."""

    print("phase 0 immediate: Exp7269 recognition audit started", flush=True)
    args = _parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"run_date_must_be_{RUN_DATE}")
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    selected = tuple(filter(None, args.stream_ids.split(","))) or tuple(
        f"prospective-{index + 1:02d}" for index in range(learning.STREAM_COUNT)
    )
    if args.audit_worker:
        result = _audit_raw_evidence_impl(
            REPO_ROOT,
            stream_ids=selected,
            bootstrap_draws=args.bootstrap_draws,
            progress=True,
        )
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
        return 0
    if args.e2e_worker:
        rows, rollback_rows = run_e2e_controls(paths.e2e_sidecar.parent / "validation_e2e")
        print(
            RESULT_PREFIX
            + json.dumps({"rows": rows, "rollback_rows": rollback_rows}, sort_keys=True),
            flush=True,
        )
        return int(not all(row["passed"] is True for row in rows))
    if args.validate:
        candidate = args.artifact_path or paths.artifact
        artifact = _load_object(candidate)
        expected = (
            tuple(sorted({str(row.get("stream_id")) for row in artifact.get("rows", [])})) or None
        )
        errors = validate_artifact(
            artifact,
            repo_root=REPO_ROOT,
            expected_stream_ids=expected,
            check_files=artifact.get("status") == "complete",
        )
        if errors:
            raise AuditEvidenceError("artifact_validation_failed:" + ",".join(errors))
        _progress(7, "end", f"validated terminal candidate {candidate}")
        return 0
    artifact = build_and_seal(
        REPO_ROOT,
        paths,
        stream_ids=selected,
        bootstrap_draws=args.bootstrap_draws,
        progress=True,
    )
    if artifact["status"] == "blocked":
        receipt = write_artifact(
            paths.artifact, artifact, repo_root=REPO_ROOT, expected_stream_ids=selected
        )
        _progress(7, "end", f"wrote blocked terminal artifact sha256={receipt['sha256']}")
        return 0
    _progress(7, "start", "write complete/null measured candidate under raw evidence")
    _write_evidence(paths.terminal_candidate, artifact)
    _progress(7, "end", f"candidate={paths.terminal_candidate}")
    _progress(8, "start", "BEFORE focused tests, coverage, E2E, and artifact validators")
    receipts = list(artifact["validation_receipts"])
    receipts.extend(
        _command_receipt(command) for command in _validation_commands(paths.terminal_candidate)
    )
    artifact = attach_validation_receipts(artifact, receipts)
    _write_evidence(
        paths.checkpoint,
        {
            "schema": "carnot.exp7269.checkpoint.v1",
            "status": "in_progress",
            "phase": 8,
            "validation_receipts": receipts,
        },
    )
    failed = [row for row in receipts if row["exit_code"] != 0]
    if failed:
        raise RuntimeError(
            "focused_validation_failed:" + ",".join(row["command"] for row in failed)
        )
    _progress(8, "end", "AFTER focused tests, coverage, E2E, and validators passed")
    _progress(9, "start", "BEFORE final cold validation and atomic terminal publication")
    receipt = write_artifact(
        paths.artifact,
        artifact,
        repo_root=REPO_ROOT,
        expected_stream_ids=selected,
    )
    _progress(9, "end", f"AFTER terminal write sha256={receipt['sha256']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin wrapper owns execution.
    raise SystemExit(main())
