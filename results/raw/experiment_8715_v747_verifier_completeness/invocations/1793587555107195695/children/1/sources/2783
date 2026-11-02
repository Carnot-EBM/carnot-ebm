"""Measure prospective learning value from validated archive reuse.

This experiment replays the sealed Exp7240 streams. It keeps evaluator labels
outside controller inputs until each prediction and query receipt exists. The
result can be a complete null: run completion and learning value are separate.

Spec refs: REQ-CL-7241 and SCENARIO-CL-7241-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import socket
import time
from typing import Any

from carnot import experiment_7199_v634_bounded_acquisition as exp7199
from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7226_v636_belief_compiler as exp7226
from carnot import experiment_7227_v636_belief_learning as exp7227
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7241
SCHEMA = "carnot.exp7241.v637_recurrence_learning.v1"
MILESTONE = "2026.09.637"
RUN_DATE = "20260912"
RANDOM_SEED = 7_241_000
BOOTSTRAP_SEED = 7_241_951
BOOTSTRAP_DRAWS = 10_000
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"
EXPECTED_UPSTREAM_SHA256 = "sha256:0f12abc839f3d70006698ebaa5169f12ab1cae9dc7526c6d124078ec7a6baf74"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM_ARTIFACT = Path("results/experiment_7240_v637_recurrence_fixture.json")
DEFAULT_STREAM_ROOT = Path("results/streams/experiment_7240")
DEFAULT_ARTIFACT = Path("results/experiment_7241_v637_recurrence_learning.json")
DEFAULT_DECISION_ROWS = Path("results/raw/experiment_7241/decision_rows.jsonl")
DEFAULT_OPERATION_ROWS = Path("results/raw/experiment_7241/operation_receipts.jsonl")
DEFAULT_STATE_MANIFEST = Path("results/checkpoints/experiment_7241_v637_memory_state_manifest.json")
DEFAULT_EVIDENCE_MANIFEST = Path("results/checkpoints/experiment_7241_v637_evidence_manifest.json")
DEFAULT_PROVISIONAL_CHECKPOINT = Path(
    "results/checkpoints/experiment_7241_v637_recurrence_learning.json"
)
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7226_v636_belief_compiler.py"),
    Path("python/carnot/experiment_7227_v636_belief_learning.py"),
    Path("python/carnot/experiment_7240_v637_recurrence_fixture.py"),
    Path("python/carnot/experiment_7241_v637_recurrence_learning.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("scripts/experiments/experiment_7240_v637_recurrence_fixture.py"),
    Path("scripts/experiments/experiment_7241_v637_recurrence_learning.py"),
    Path("tests/python/test_experiment_7240_v637_recurrence_fixture.py"),
    Path("tests/python/test_experiment_7241_v637_recurrence_learning.py"),
    SPEC_PATH,
)

TOP_LEVEL_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "execution_host",
    "duration_s",
    "phase_spans_s",
    "MODEL_SPECS",
    "model_invoked",
    "current_model_load_count",
    "current_generation_count",
    "current_inference_count",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
    "acceptance_gate_results",
    "recurrence_run_complete_score",
    "recurrence_learning_value_score",
    "continuous_self_learning_task",
    "continuous_learning_counts",
    "decision_rows_path",
    "paired_seed_rows",
    "memory_state_manifest",
    "operation_receipts_path",
    "evidence_manifest_path",
    "comparison_rows",
    "causal_summary",
    "latency_summary",
    "stream_contract",
    "arm_contract",
    "e2e_control_summary",
    "sidecar_write_receipts",
    "family_retired",
    "no_model_weight_mutation",
    "methodology",
    "default_pipeline_modified",
    "publication_performed",
)
REQUIRED_ARTIFACT_FIELDS = TOP_LEVEL_FIELDS

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Version the artifact and bind experiment_id and milestone to this task.",
    "experiment_id": "A fixed identifier prevents another task from supplying this result.",
    "milestone": "Bind the result to the V637 recurrence-learning roadmap item.",
    "status": "Terminal complete or blocked only; unfinished work uses a separate checkpoint path.",
    "run_date": "Use 20260912; retain actual UTC start and end timestamps.",
    "started_at_utc": "Keep the actual UTC start separate from the fixed execution date.",
    "completed_at_utc": "Keep the actual UTC end separate from the fixed execution date.",
    "field_principles": "Keep ordinary values at top level; put their explanations in this map.",
    "preconditions_checked": "Observe paths, resources, identity, hashes, and quarantine before replay.",
    "inference_substrate": "Use the recognized literal for the operation actually executed.",
    "inference_substrate_class": "The actual class determines the duration floor; never pad duration.",
    "execution_venue": "Top-level orchestration is host; board rows name a board venue.",
    "execution_host": "Record the actual hostname separately from the execution venue.",
    "duration_s": "Measure monotonic elapsed work and record phase spans separately.",
    "phase_spans_s": "Retain measured monotonic time for each numbered work phase.",
    "MODEL_SPECS": "Models actually invoked; use an empty list when no LLM runs.",
    "model_invoked": "Describe only current execution; historical sources are separate.",
    "current_model_load_count": "Count only model loads performed by this run.",
    "current_generation_count": "Count only generations performed by this run.",
    "current_inference_count": "Count only inference calls performed by this run.",
    "source_artifact_hashes": "Hash source code, public inputs, private inputs, and raw outputs.",
    "rows": "Retain one comparison row for every independent stream and arm.",
    "sample_size_budget": "Predeclare units, attempts, completions, censoring, and stopping.",
    "random_seed": "Freeze stream and bootstrap schedules before evaluation labels are observed.",
    "reproducibility_checksum": "Hash exact settings, inputs, gates, and raw-row receipts.",
    "gate_check_summary": "Name failed check, upstream, field, expected value, and observed value.",
    "verifier_is_oracle": "True because the exact evaluator also defines correctness.",
    "verdict_class": "Use exactly positive, circular_positive, null, blocked, disqualified, or partial.",
    "honest_verdict": "A completed finding starts complete_; an external absence starts blocked_.",
    "acceptance_gate_results": "Keep each frozen criterion, actual value, and pass state separate.",
    "recurrence_run_complete_score": "One means all scheduled event-and-arm outcomes are accounted for.",
    "recurrence_learning_value_score": "One means all causal, recurrence, safety, and error gates pass.",
    "continuous_self_learning_task": "True; constraints can be added, deactivated, and reactivated.",
    "continuous_learning_counts": "Report additions, deactivations, and validated reactivations.",
    "decision_rows_path": "The raw JSONL keeps each pre-release prediction and later label.",
    "paired_seed_rows": "Keep 32 independent stream units per arm with efficacy and cost metrics.",
    "memory_state_manifest": "Hash before and after active masks, archives, certificates, and bytes.",
    "operation_receipts_path": "Retain measured query, delivery, validation, switch, and write receipts.",
    "evidence_manifest_path": "Keep historical sources and synthetic negative controls in a sidecar.",
    "comparison_rows": "Use paired seed differences and frozen 10,000-draw bootstrap intervals.",
    "causal_summary": "Separate reactivation, later-change, pre-release, and headroom counts.",
    "latency_summary": "Report lookup and update tails, bytes, CPU time, and the 100-times target gap.",
    "stream_contract": "Bind separated labels, delays, warmup, recurrence, and controller inputs.",
    "arm_contract": "Bind the six frozen arms and their shared observable feedback budget.",
    "e2e_control_summary": "Require rejected writes, fresh restore, and byte-identical rollback.",
    "sidecar_write_receipts": "Measure and hash each durable raw or state sidecar write.",
    "family_retired": "An inconclusive null does not retire the constraint family.",
    "no_model_weight_mutation": "The opt-in memory experiment must not alter model weights.",
    "methodology": "State the CPU replay, authority boundary, and independent-unit method.",
    "default_pipeline_modified": "Keep production defaults unchanged.",
    "publication_performed": "This task does not publish, upload, or submit results.",
}

unwrap_principled = exp7213.unwrap_principled
gate_check = exp7213.gate_check
gate_summary = exp7213.gate_summary
quarantine_state = exp7213.quarantine_state


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw rows, state, evidence, provisional work, and terminal bytes separate."""

    decision_rows: Path
    operation_rows: Path
    state_manifest: Path
    evidence_manifest: Path
    provisional_checkpoint: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Use the roadmap paths below this checkout's result directory."""

        return cls.from_results_root(REPO_ROOT / "results")

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put test-owned outputs below one caller-owned result root."""

        return cls.from_results_root(root)

    @classmethod
    def from_results_root(cls, root: Path) -> ExperimentPaths:
        """Derive every output without changing the authenticated input paths."""

        raw = root / "raw" / "experiment_7241"
        checkpoints = root / "checkpoints"
        return cls(
            raw / "decision_rows.jsonl",
            raw / "operation_receipts.jsonl",
            checkpoints / DEFAULT_STATE_MANIFEST.name,
            checkpoints / DEFAULT_EVIDENCE_MANIFEST.name,
            checkpoints / DEFAULT_PROVISIONAL_CHECKPOINT.name,
            root / DEFAULT_ARTIFACT.name,
        )


@dataclass(frozen=True)
class LearningPanel:
    """Retain prospective rows, independent summaries, costs, receipts, and exact state."""

    decision_rows: list[JsonDict]
    paired_seed_rows: list[JsonDict]
    operation_receipts: list[JsonDict]
    state_entries: list[JsonDict]
    costs: dict[str, list[int]]
    causal_summary: JsonDict
    maximum_state_bytes: int


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Emit a truthful flushed phase boundary for the external task watchdog."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _resolve(repo_root: Path, path: str | Path) -> Path:
    """Resolve repository-relative evidence while preserving absolute output roots."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _sha256_path(path: Path) -> str | None:
    """Hash exact file bytes and preserve absence as a failed observation."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object; malformed or absent evidence cannot pass a gate."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read object rows; any malformed file becomes unavailable evidence."""

    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError):
        return []
    return rows if all(isinstance(row, dict) for row in rows) else []


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating output evidence early."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _atomic_write(path: Path, payload: bytes) -> JsonDict:
    """Write complete bytes through one flushed rename and measure the real operation."""

    started = time.perf_counter_ns()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path),
        "sha256": _sha256_path(path),
        "bytes": len(payload),
        "cost_ns": time.perf_counter_ns() - started,
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode one canonical JSON object per immutable audit line."""

    return b"".join(transactional.canonical_json_bytes(dict(row)) for row in rows)


def _task_identity(text: str) -> JsonDict:
    """Read only the Exp7241 roadmap block for the identity gate."""

    match = re.search(r"(?ms)^- id: exp7241-recurrence-learning\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(1)
    milestone = re.search(r"(?m)^  milestone: (.+)$", block)
    deliverable = re.search(r"(?m)^  deliverable: (.+)$", block)
    return {
        "id": "exp7241-recurrence-learning" if block else None,
        "milestone": None if milestone is None else milestone.group(1).strip(),
        "deliverable": None if deliverable is None else deliverable.group(1).strip(),
    }


def _receipt_matches(repo_root: Path, receipt: Mapping[str, Any]) -> bool:
    """Authenticate one declared path against its exact current bytes."""

    path = receipt.get("path")
    expected = receipt.get("sha256")
    return isinstance(path, str) and _sha256_path(_resolve(repo_root, path)) == expected


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    upstream_artifact: Path = DEFAULT_UPSTREAM_ARTIFACT,
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict]:
    """Authenticate the exact Exp7240 artifact, streams, contract, and local resources."""

    upstream_path = _resolve(repo_root, upstream_artifact)
    upstream = _load_object(upstream_path)
    hashes = {str(path): _sha256_path(_resolve(repo_root, path)) for path in SOURCE_PATHS}
    hashes[str(upstream_path)] = _sha256_path(upstream_path)
    spec_text = _resolve(repo_root, SPEC_PATH).read_text(encoding="utf-8")
    roadmap = _resolve(repo_root, "research-roadmap.yaml").read_text(encoding="utf-8")
    exclusions = _resolve(repo_root, "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    identity = _task_identity(roadmap)
    imports = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "carnot.experiment_7226_v636_belief_compiler",
            "carnot.experiment_7227_v636_belief_learning",
            "carnot.experiment_7240_v637_recurrence_fixture",
            "carnot.memory.transactional_constraint_memory",
        )
    }
    outputs = {
        field: _path_writable(getattr(paths, field))
        for field in (
            "decision_rows",
            "operation_rows",
            "state_manifest",
            "evidence_manifest",
            "provisional_checkpoint",
            "artifact",
        )
    }
    quarantine = quarantine_state(
        upstream,
        exclusions,
        upstream_path.name,
        "exp7240-recurrence-fixture",
    )
    try:
        checksum_valid = bool(upstream) and exp7240.reproducibility_checksum(
            upstream
        ) == upstream.get("reproducibility_checksum")
    except (AttributeError, KeyError, TypeError, ValueError):
        checksum_valid = False
    receipts = upstream.get("stream_receipts", {})
    receipt_matches = (
        {
            name: isinstance(receipt, Mapping) and _receipt_matches(repo_root, receipt)
            for name, receipt in receipts.items()
        }
        if isinstance(receipts, Mapping)
        else {}
    )
    expected_receipt_names = {
        "public_stream",
        "private_authority",
        "release_schedule",
        "public_manifest",
    }
    raw_receipts = {
        name: upstream.get(name, {})
        for name in ("raw_rows_receipt", "checkpoint_receipt", "control_receipts_path")
    }
    raw_matches = {
        name: isinstance(receipt, Mapping) and _receipt_matches(repo_root, receipt)
        for name, receipt in raw_receipts.items()
    }
    source_state = {
        str(path): "nonempty" if hashes[str(path)] is not None else "missing"
        for path in SOURCE_PATHS
    }
    expected_identity = {
        "id": "exp7241-recurrence-learning",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT),
    }
    arm_contract = upstream.get("arm_contract", {})
    manifest = _load_object(_resolve(repo_root, DEFAULT_STREAM_ROOT / "public_manifest.json"))
    checks = [
        gate_check(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-CL-7241",
            True,
            "REQ-CL-7241" in spec_text,
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7241-*",
            9,
            len(set(re.findall(r"SCENARIO-CL-7241-[A-Z-]+", spec_text))),
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            {str(path): "nonempty" for path in SOURCE_PATHS},
            source_state,
        ),
        gate_check(
            "v637_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            expected_identity,
            identity,
        ),
        gate_check("required_imports", "python", "imports", dict.fromkeys(imports, True), imports),
        gate_check(
            "writable_output_paths",
            "host_filesystem",
            "raw,checkpoint,artifact",
            dict.fromkeys(outputs, True),
            outputs,
        ),
        gate_check(
            "exp7240_artifact_hash",
            "exp7240-recurrence-fixture",
            str(upstream_artifact),
            EXPECTED_UPSTREAM_SHA256,
            hashes[str(upstream_path)],
        ),
        gate_check(
            "exp7240_status",
            "exp7240-recurrence-fixture",
            "status",
            "complete",
            upstream.get("status"),
        ),
        gate_check(
            "exp7240_fixture_ready",
            "exp7240-recurrence-fixture",
            "recurrence_fixture_ready_score",
            1,
            unwrap_principled(upstream.get("recurrence_fixture_ready_score")),
        ),
        gate_check(
            "exp7240_checksum",
            "exp7240-recurrence-fixture",
            "reproducibility_checksum",
            True,
            checksum_valid,
        ),
        gate_check(
            "exp7240_stream_receipts",
            "exp7240-recurrence-fixture",
            "stream_receipts",
            dict.fromkeys(sorted(expected_receipt_names), True),
            {name: receipt_matches.get(name, False) for name in sorted(expected_receipt_names)},
        ),
        gate_check(
            "exp7240_raw_receipts",
            "exp7240-recurrence-fixture",
            "raw_rows,checkpoint,controls",
            dict.fromkeys(raw_matches, True),
            raw_matches,
        ),
        gate_check(
            "exp7240_six_arm_contract",
            "exp7240-recurrence-fixture",
            "arm_contract.arms",
            list(exp7240.ARMS),
            arm_contract.get("arms") if isinstance(arm_contract, Mapping) else None,
        ),
        gate_check(
            "exp7240_stream_contract",
            str(DEFAULT_STREAM_ROOT / "public_manifest.json"),
            "stream_count,events_per_stream",
            [exp7240.STREAM_COUNT, exp7240.EVENTS_PER_STREAM],
            [manifest.get("stream_count"), manifest.get("events_per_stream")],
        ),
        gate_check(
            "exp7240_not_quarantined",
            "artifact_metadata_and_ops/exclusion_manifest.yaml",
            "quarantined",
            False,
            quarantine["quarantined"],
        ),
    ]
    for receipt in list(receipts.values()) + list(raw_receipts.values()):
        if isinstance(receipt, Mapping) and isinstance(receipt.get("path"), str):
            receipt_path = _resolve(repo_root, str(receipt["path"]))
            hashes[str(receipt_path)] = _sha256_path(receipt_path)
    return checks, hashes, upstream


def load_stream_views(repo_root: Path, upstream: Mapping[str, Any]) -> exp7240.StreamViews:
    """Load four authenticated views without copying private fields into public rows."""

    receipts = upstream.get("stream_receipts", {})
    if not isinstance(receipts, Mapping):
        return exp7240.StreamViews([], [], [], {})
    paths = {
        name: _resolve(repo_root, str(receipts.get(name, {}).get("path", "")))
        for name in (
            "public_stream",
            "private_authority",
            "release_schedule",
            "public_manifest",
        )
    }
    return exp7240.StreamViews(
        _read_jsonl(paths["public_stream"]),
        _read_jsonl(paths["private_authority"]),
        _read_jsonl(paths["release_schedule"]),
        _load_object(paths["public_manifest"]),
    )


def _controller_input(event: Mapping[str, Any]) -> JsonDict:
    """Expose only the three public fields accepted by the controller contract."""

    return {
        "event_id": event["event_id"],
        "family_id": event["family_id"],
        "numeric_value": event["numeric_value"],
    }


def _support_release(pending: Mapping[str, Any]) -> JsonDict:
    """Release one evaluator label only after its scheduled delay has elapsed."""

    public = pending["public"]
    return {
        "event_id": public["event_id"],
        "family_id": public["family_id"],
        "numeric_value": public["numeric_value"],
        "observed_label": pending["observed_label"],
        "role": "support",
        "request_index": pending["request_index"],
        "release_index": pending["release_index"],
    }


def _state_entry(
    stream_id: str,
    arm: str,
    initial_bytes: bytes,
    controller: Any,
    certificates: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Retain exact initial and final bytes plus active and archive mask hashes."""

    final_bytes = controller.state_bytes()
    state = controller.state_dict()
    if isinstance(controller, exp7240.ArchivedBeliefController):
        active = state["active"]
        archives = state["archives"]
        release_ids = state["release_ids"]
    else:
        active = state
        archives = []
        release_ids = []
    active_masks = {
        family: int(active["families"][family]["survivor_mask"]) for family in exp7240.FAMILIES
    }
    archive_masks = [
        {
            "archive_id": row["archive_id"],
            "state_hash": row["state_hash"],
            "survivor_masks": deepcopy(row["survivor_masks"]),
        }
        for row in archives
    ]
    return {
        "unit_id": f"{stream_id}:{arm}",
        "stream_id": stream_id,
        "arm": arm,
        "initial_state_sha256": transactional.sha256_bytes(initial_bytes),
        "initial_state_bytes_b64": transactional.encode_bytes(initial_bytes),
        "final_state_sha256": transactional.sha256_bytes(final_bytes),
        "final_state_bytes_b64": transactional.encode_bytes(final_bytes),
        "final_serialized_bytes": len(final_bytes),
        "active_survivor_masks": active_masks,
        "active_masks_sha256": transactional.sha256_json(active_masks),
        "archived_masks": archive_masks,
        "archived_masks_sha256": transactional.sha256_json(archive_masks),
        "release_certificate_count": len(release_ids),
        "operation_certificate_count": len(certificates),
        "operation_certificates_sha256": transactional.sha256_json(list(certificates)),
    }


def _reduce_seed_arm(
    stream_id: str,
    seed: int,
    arm: str,
    decision_rows: Sequence[Mapping[str, Any]],
    operation_rows: Sequence[Mapping[str, Any]],
    state_bytes: int,
) -> JsonDict:
    """Reduce one independent stream and arm without treating events as independent."""

    future = [row for row in decision_rows if int(row["chronology_index"]) >= exp7240.WARMUP_COUNT]
    recurrence = [row for row in future if row["recurrence_eligible"] is True]
    operations = [
        row for row in operation_rows if row.get("stream_id") == stream_id and row.get("arm") == arm
    ]
    hits = sum(int(row.get("archive_reactivation_count", 0)) for row in operations)
    valid_hits = sum(int(row.get("valid_reactivation_count", 0)) for row in operations)
    update_costs = [
        int(row["cost_ns"])
        for row in operations
        if row["operation"] == "validation_and_state_write"
    ]
    lookup_costs = [int(row["lookup_cost_ns"]) for row in decision_rows]
    return {
        "unit_id": f"{stream_id}:{arm}",
        "stream_id": stream_id,
        "seed": seed,
        "arm": arm,
        "metric": "prospective_full_denominator_error",
        "future_event_count": len(future),
        "future_error": sum(int(row["full_denominator_error"]) for row in future),
        "future_error_rate": (
            sum(int(row["full_denominator_error"]) for row in future) / len(future)
        ),
        "false_accept": sum(int(row["false_accept"]) for row in future),
        "false_accept_rate": sum(int(row["false_accept"]) for row in future) / len(future),
        "abstention": sum(int(row["abstention"]) for row in future),
        "abstention_rate": sum(int(row["abstention"]) for row in future) / len(future),
        "recurrence_event_count": len(recurrence),
        "recurrence_error": sum(int(row["full_denominator_error"]) for row in recurrence),
        "recurrence_error_rate": (
            None
            if not recurrence
            else sum(int(row["full_denominator_error"]) for row in recurrence) / len(recurrence)
        ),
        "archive_hit_count": hits,
        "archive_hit_valid_count": valid_hits,
        "archive_hit_validity": None if hits == 0 else valid_hits / hits,
        "valid_reactivation_count": valid_hits,
        "later_changed_decision_count": sum(
            int(row["later_changed_decision"]) for row in decision_rows
        ),
        "pre_release_difference_count": sum(
            int(row["pre_release_difference"]) for row in decision_rows
        ),
        "query_count": sum(int(row["query_selected"]) for row in decision_rows),
        "lookup_p50_ns": _percentile(lookup_costs, 0.50),
        "lookup_p95_ns": _percentile(lookup_costs, 0.95),
        "update_p50_ns": _percentile(update_costs, 0.50),
        "update_p95_ns": _percentile(update_costs, 0.95),
        "serialized_state_bytes": state_bytes,
    }


def run_learning_panel(
    views: exp7240.StreamViews,
    *,
    stream_ids: Sequence[str] | None = None,
    progress: bool = False,
) -> LearningPanel:
    """Replay all arms while receipts remain earlier than evaluator-label access."""

    selected_streams = tuple(
        stream_ids
        if stream_ids is not None
        else (f"stream-{index + 1:02d}" for index in range(exp7240.STREAM_COUNT))
    )
    public_by_stream: dict[str, list[JsonDict]] = {stream_id: [] for stream_id in selected_streams}
    for row in views.public:
        stream_id = str(row["stream_id"])
        if stream_id in public_by_stream:
            public_by_stream[stream_id].append(row)
    authority = {str(row["event_id"]): row for row in views.authority}
    schedules = {str(row["event_id"]): row for row in views.releases}
    decision_rows: list[JsonDict] = []
    operation_rows: list[JsonDict] = []
    state_entries: list[JsonDict] = []
    costs = {
        "lookup_ns": [],
        "query_ns": [],
        "delayed_delivery_ns": [],
        "update_ns": [],
        "validation_ns": [],
        "archive_switch_ns": [],
        "state_write_ns": [],
    }
    causal = {
        "valid_reactivation_count": 0,
        "later_changed_decision_count": 0,
        "pre_release_difference_count": 0,
        "positive_control_changed_decision_count": 0,
    }
    maximum_state_bytes = 0
    started = time.monotonic()

    for stream_offset, stream_id in enumerate(selected_streams):
        events = public_by_stream[stream_id]
        stream_seed = int(authority[str(events[0]["event_id"])]["stream_seed"])
        frozen = exp7226.PackedBeliefController()
        destructive = exp7226.PackedBeliefController()
        reset = exp7240.ArchivedBeliefController(archive_cap=0, nomination_mode="none")
        stale = exp7240.ArchivedBeliefController(nomination_mode="stale")
        validated = exp7240.ArchivedBeliefController(nomination_mode="validated")
        shuffled = exp7240.ArchivedBeliefController(nomination_mode="shuffled_validated")
        controllers: dict[str, Any] = {
            "frozen_warmup": frozen,
            "destructive_packed_learner": destructive,
            "reset_relearn_no_archive": reset,
            "unvalidated_stale_archive_reuse": stale,
            "validation_selected_archive": validated,
            "shuffled_nomination_validated": shuffled,
        }
        initial_bytes = {arm: controller.state_bytes() for arm, controller in controllers.items()}
        certificates: dict[str, list[JsonDict]] = {arm: [] for arm in controllers}
        stream_decision_start = len(decision_rows)
        pending: list[JsonDict] = []
        actual_queries = 0
        released_queries = 0

        for block_index, offset in enumerate(
            range(0, exp7240.EVENTS_PER_STREAM, exp7240.QUERY_BLOCK_SIZE)
        ):
            block = events[offset : offset + exp7240.QUERY_BLOCK_SIZE]
            public_block = [_controller_input(row) for row in block]
            tie_ranks = exp7199.seeded_tie_ranks(stream_seed, block_index, public_block)
            query_started = time.perf_counter_ns()
            selected = destructive.select_request(public_block, tie_ranks)
            query_completed = time.perf_counter_ns()
            query_cost = query_completed - query_started
            costs["query_ns"].append(query_cost)
            selected_id = str(selected["event_id"])
            query_source = transactional.sha256_json(
                {"block": public_block, "tie_ranks": tie_ranks, "selected_event_id": selected_id}
            )

            for event in block:
                event_id = str(event["event_id"])
                chronology_index = int(event["chronology_index"])
                controller_event = _controller_input(event)
                predictions: dict[str, JsonDict] = {}
                for arm, controller in controllers.items():
                    state_hash = controller.state_hash()
                    lookup_started = time.perf_counter_ns()
                    prediction, energy = controller.predict(controller_event)
                    lookup_completed = time.perf_counter_ns()
                    lookup_cost = lookup_completed - lookup_started
                    costs["lookup_ns"].append(lookup_cost)
                    archive_count = (
                        len(controller.archives())
                        if isinstance(controller, exp7240.ArchivedBeliefController)
                        else 0
                    )
                    receipt = {
                        "event_id": event_id,
                        "stream_id": stream_id,
                        "arm": arm,
                        "controller_input": controller_event,
                        "state_hash_before_release": state_hash,
                        "prediction": prediction,
                        "prediction_energy": energy,
                    }
                    predictions[arm] = {
                        "prediction": prediction,
                        "energy": energy,
                        "state_hash": state_hash,
                        "archive_count": archive_count,
                        "lookup_cost_ns": lookup_cost,
                        "completed_ns": lookup_completed,
                        "receipt_sha256": transactional.sha256_json(receipt),
                    }
                    maximum_state_bytes = max(maximum_state_bytes, len(controller.state_bytes()))

                will_query = (
                    event_id == selected_id
                    and actual_queries < exp7240.QUERY_CEILING
                    and len(pending) < exp7240.PENDING_CAPACITY
                )
                query_receipt = {
                    "event_id": event_id,
                    "stream_id": stream_id,
                    "selected_event_id": selected_id,
                    "admitted": will_query,
                    "pending_before": len(pending),
                    "query_count_before": actual_queries,
                    "source_sha256": query_source,
                }
                query_receipt_sha = transactional.sha256_json(query_receipt)
                query_receipt_completed = time.perf_counter_ns()
                if event_id == selected_id:
                    operation_rows.append(
                        {
                            "operation": "query",
                            "stream_id": stream_id,
                            "arm": "shared_adaptive_query",
                            "event_id": event_id,
                            "cost_ns": query_cost,
                            "source_sha256": query_source,
                            "receipt_sha256": query_receipt_sha,
                            "admitted": will_query,
                        }
                    )

                truth = authority[event_id]
                schedule = schedules[event_id]
                label_accessed_ns = time.perf_counter_ns()
                label_receipt = {
                    "event_id": event_id,
                    "exact_label": truth["exact_label"],
                    "drift_pattern": truth["drift_pattern"],
                    "regime_id": truth["regime_id"],
                    "release_delay": schedule["delay"],
                }
                label_receipt_sha = transactional.sha256_json(label_receipt)
                frozen_prediction = str(predictions["frozen_warmup"]["prediction"])
                validated_prediction = str(predictions["validation_selected_archive"]["prediction"])
                stale_prediction = str(predictions["unvalidated_stale_archive_reuse"]["prediction"])
                for arm, prediction_row in predictions.items():
                    prediction = str(prediction_row["prediction"])
                    abstention = int(prediction == "abstain")
                    classification_error = int(prediction != truth["exact_label"])
                    later_changed = int(released_queries > 0 and prediction != frozen_prediction)
                    pre_release_difference = int(
                        released_queries == 0 and prediction != frozen_prediction
                    )
                    decision_rows.append(
                        {
                            "unit_id": f"{stream_id}:{arm}",
                            "stream_id": stream_id,
                            "seed": stream_seed,
                            "arm": arm,
                            "event_id": event_id,
                            "chronology_index": chronology_index,
                            "controller_input_fields": [
                                "event_id",
                                "family_id",
                                "numeric_value",
                            ],
                            "held_out_label_visible_to_controller": False,
                            "prediction": prediction,
                            "prediction_energy": prediction_row["energy"],
                            "prediction_receipt_sha256": prediction_row["receipt_sha256"],
                            "prediction_receipt_completed_ns": prediction_row["completed_ns"],
                            "query_receipt_sha256": query_receipt_sha,
                            "query_receipt_completed_ns": query_receipt_completed,
                            "label_accessed_ns": label_accessed_ns,
                            "label_release_receipt_sha256": label_receipt_sha,
                            "later_released_label": truth["exact_label"],
                            "drift_pattern": truth["drift_pattern"],
                            "regime_id": truth["regime_id"],
                            "recurrence_eligible": (
                                truth["drift_pattern"] == "aba_recurrence"
                                and chronology_index >= 768
                            ),
                            "state_hash_before_release": prediction_row["state_hash"],
                            "archive_count_before_release": prediction_row["archive_count"],
                            "released_query_count_before": released_queries,
                            "query_selected": will_query,
                            "classification_error": classification_error,
                            "abstention": abstention,
                            "full_denominator_error": max(classification_error, abstention),
                            "false_accept": int(
                                prediction == "accept" and truth["exact_label"] == "reject"
                            ),
                            "later_changed_decision": later_changed,
                            "pre_release_difference": pre_release_difference,
                            "lookup_cost_ns": prediction_row["lookup_cost_ns"],
                            "prediction_frozen_before_release": True,
                        }
                    )
                causal["later_changed_decision_count"] += int(
                    released_queries > 0 and validated_prediction != frozen_prediction
                )
                causal["pre_release_difference_count"] += int(
                    released_queries == 0 and validated_prediction != frozen_prediction
                )
                causal["positive_control_changed_decision_count"] += int(
                    released_queries > 0 and stale_prediction != validated_prediction
                )

                if will_query:
                    actual_queries += 1
                    pending.append(
                        {
                            "public": controller_event,
                            "observed_label": truth["exact_label"],
                            "request_index": chronology_index,
                            "release_index": chronology_index + int(schedule["delay"]),
                            "authority_receipt_sha256": label_receipt_sha,
                        }
                    )
                due = sorted(
                    [row for row in pending if int(row["release_index"]) <= chronology_index],
                    key=lambda row: (int(row["release_index"]), int(row["request_index"])),
                )
                if due:
                    payload: list[JsonDict] = []
                    for pending_row in due:
                        delivery_started = time.perf_counter_ns()
                        release = _support_release(pending_row)
                        delivery_cost = time.perf_counter_ns() - delivery_started
                        costs["delayed_delivery_ns"].append(delivery_cost)
                        release_source = transactional.sha256_json(release)
                        payload.append(release)
                        operation_rows.append(
                            {
                                "operation": "delayed_delivery",
                                "stream_id": stream_id,
                                "arm": "shared_adaptive_feedback",
                                "event_id": release["event_id"],
                                "cost_ns": delivery_cost,
                                "source_sha256": release_source,
                                "authority_receipt_sha256": pending_row["authority_receipt_sha256"],
                            }
                        )
                    release_source = transactional.sha256_json(payload)
                    update_controllers = dict(controllers)
                    if chronology_index >= exp7240.WARMUP_COUNT:
                        update_controllers.pop("frozen_warmup")
                    for arm, controller in update_controllers.items():
                        update_started = time.perf_counter_ns()
                        receipt = controller.commit_batch(
                            payload,
                            current_cycle=chronology_index,
                            expected_parent_hash=controller.state_hash(),
                        )
                        update_cost = time.perf_counter_ns() - update_started
                        costs["update_ns"].append(update_cost)
                        validation_started = time.perf_counter_ns()
                        receipt_operations = receipt.get("operations", [])
                        certificate_sha = transactional.sha256_json(receipt_operations)
                        valid_reactivations = 0
                        reactivations = 0
                        archive_creations = 0
                        for operation in receipt_operations:
                            archive_creations += int(operation.get("archived_before_reset") is True)
                            reactivated = operation.get("reactivated_archive_id") is not None
                            reactivations += int(reactivated)
                            nomination = operation.get("nomination_receipt", {})
                            valid_reactivations += int(
                                reactivated
                                and nomination.get("final_gate")
                                == "at_least_8_released_witnesses_and_zero_contradictions"
                                and any(
                                    candidate.get("archive_id")
                                    == operation.get("reactivated_archive_id")
                                    and candidate.get("gate_passed") is True
                                    for candidate in nomination.get("candidates", [])
                                )
                            )
                        validation_cost = time.perf_counter_ns() - validation_started
                        costs["validation_ns"].append(validation_cost)
                        switch_started = time.perf_counter_ns()
                        switch_receipt = transactional.sha256_json(
                            {
                                "archive_creations": archive_creations,
                                "reactivations": reactivations,
                                "valid_reactivations": valid_reactivations,
                            }
                        )
                        switch_cost = time.perf_counter_ns() - switch_started
                        costs["archive_switch_ns"].append(switch_cost)
                        state_write_started = time.perf_counter_ns()
                        state_bytes = controller.state_bytes()
                        new_state_hash = transactional.sha256_bytes(state_bytes)
                        state_write_cost = time.perf_counter_ns() - state_write_started
                        costs["state_write_ns"].append(state_write_cost)
                        maximum_state_bytes = max(maximum_state_bytes, len(state_bytes))
                        operation_row = {
                            "operation": "validation_and_state_write",
                            "stream_id": stream_id,
                            "arm": arm,
                            "event_id": event_id,
                            "release_count": len(payload),
                            "cost_ns": update_cost,
                            "validation_cost_ns": validation_cost,
                            "archive_switch_cost_ns": switch_cost,
                            "state_write_cost_ns": state_write_cost,
                            "source_sha256": release_source,
                            "parent_state_sha256": receipt.get("parent_hash"),
                            "new_state_sha256": new_state_hash,
                            "certificate_sha256": certificate_sha,
                            "archive_switch_receipt_sha256": switch_receipt,
                            "archive_creation_count": archive_creations,
                            "archive_reactivation_count": reactivations,
                            "valid_reactivation_count": valid_reactivations,
                        }
                        operation_rows.append(operation_row)
                        certificates[arm].append(
                            {
                                "event_id": event_id,
                                "certificate_sha256": certificate_sha,
                                "parent_state_sha256": receipt.get("parent_hash"),
                                "new_state_sha256": new_state_hash,
                            }
                        )
                        causal["valid_reactivation_count"] += (
                            valid_reactivations if arm == "validation_selected_archive" else 0
                        )
                    released_queries += len(due)
                    pending = [row for row in pending if row not in due]

        stream_decisions = decision_rows[stream_decision_start:]
        for arm, controller in controllers.items():
            entry = _state_entry(
                stream_id,
                arm,
                initial_bytes[arm],
                controller,
                certificates[arm],
            )
            state_entries.append(entry)
        if progress:
            print(
                f"phase 5 benchmark unit {stream_offset + 1}/{len(selected_streams)} "
                f"elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )

    state_size_by_unit = {
        (entry["stream_id"], entry["arm"]): int(entry["final_serialized_bytes"])
        for entry in state_entries
    }
    paired_rows: list[JsonDict] = []
    for stream_id in selected_streams:
        stream_rows = [row for row in decision_rows if row["stream_id"] == stream_id]
        stream_seed = int(stream_rows[0]["seed"])
        for arm in exp7240.ARMS:
            arm_rows = [row for row in stream_rows if row["arm"] == arm]
            paired_rows.append(
                _reduce_seed_arm(
                    stream_id,
                    stream_seed,
                    arm,
                    arm_rows,
                    operation_rows,
                    state_size_by_unit[(stream_id, arm)],
                )
            )
    return LearningPanel(
        decision_rows,
        paired_rows,
        operation_rows,
        state_entries,
        costs,
        causal,
        maximum_state_bytes,
    )


def panel_conformance_errors(
    panel: LearningPanel,
    *,
    expected_stream_ids: Sequence[str],
) -> list[str]:
    """Check counts, chronology, denominators, source hashes, costs, and state bytes."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    expected_events = len(expected_stream_ids) * exp7240.EVENTS_PER_STREAM * len(exp7240.ARMS)
    expected_units = len(expected_stream_ids) * len(exp7240.ARMS)
    add(len(panel.decision_rows) != expected_events, "decision_row_count")
    add(len(panel.paired_seed_rows) != expected_units, "paired_seed_rows")
    add(len(panel.state_entries) != expected_units, "state_entry_count")
    add(
        any(
            row["prediction_receipt_completed_ns"] > row["label_accessed_ns"]
            or row["query_receipt_completed_ns"] > row["label_accessed_ns"]
            or row["prediction_frozen_before_release"] is not True
            for row in panel.decision_rows
        ),
        "prediction_release_chronology",
    )
    add(
        any(
            row["held_out_label_visible_to_controller"] is not False
            or row["controller_input_fields"] != ["event_id", "family_id", "numeric_value"]
            for row in panel.decision_rows
        ),
        "authority_isolation",
    )
    add(
        any(row["full_denominator_error"] < row["abstention"] for row in panel.decision_rows),
        "abstention_denominator",
    )
    add(
        any(
            int(row["query_count"]) > exp7240.QUERY_CEILING
            or int(row["future_event_count"]) != exp7240.EVENTS_PER_STREAM - exp7240.WARMUP_COUNT
            for row in panel.paired_seed_rows
        ),
        "row_contract",
    )
    add(
        any(
            int(row["cost_ns"]) < 0 or not str(row["source_sha256"]).startswith("sha256:")
            for row in panel.operation_receipts
        ),
        "operation_receipts",
    )
    add(
        any(
            not entry["initial_state_bytes_b64"] or not entry["final_state_bytes_b64"]
            for entry in panel.state_entries
        ),
        "state_bytes",
    )
    add(panel.causal_summary["pre_release_difference_count"] != 0, "pre_release_difference")
    return errors


def _percentile(values: Sequence[int | float], probability: float) -> float:
    """Return one deterministic nearest-rank percentile for measured samples."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(probability * (len(ordered) - 1)))))
    return ordered[index]


def _bootstrap_interval(values: Sequence[float], *, draws: int, salt: str) -> JsonDict:
    """Resample complete paired stream differences as the independent units."""

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


COMPARISON_SPECS = (
    ("future_error_vs_frozen", "future_error_rate", "frozen_warmup"),
    ("future_error_vs_reset", "future_error_rate", "reset_relearn_no_archive"),
    (
        "recurrence_error_vs_destructive",
        "recurrence_error_rate",
        "destructive_packed_learner",
    ),
    (
        "recurrence_error_vs_shuffled",
        "recurrence_error_rate",
        "shuffled_nomination_validated",
    ),
    ("recurrence_error_vs_frozen", "recurrence_error_rate", "frozen_warmup"),
    ("false_accept_vs_frozen", "false_accept_rate", "frozen_warmup"),
    ("false_accept_vs_reset", "false_accept_rate", "reset_relearn_no_archive"),
)


def build_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> list[JsonDict]:
    """Build every frozen paired comparison without pooling correlated events."""

    by_unit = {(int(row["seed"]), str(row["arm"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    comparisons: list[JsonDict] = []
    for comparison_id, metric, control in COMPARISON_SPECS:
        differences = []
        for seed in seeds:
            target_value = by_unit[(seed, "validation_selected_archive")].get(metric)
            control_value = by_unit[(seed, control)].get(metric)
            if isinstance(target_value, (int, float)) and isinstance(control_value, (int, float)):
                differences.append(
                    {
                        "seed": seed,
                        "difference": float(target_value) - float(control_value),
                    }
                )
        interval = _bootstrap_interval(
            [row["difference"] for row in differences],
            draws=draws,
            salt=comparison_id,
        )
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "metric": metric,
                "target_arm": "validation_selected_archive",
                "control_arm": control,
                "independent_unit": "stream_seed",
                "independent_unit_count": len(differences),
                "bootstrap_draws": draws,
                "seed_differences": differences,
                **interval,
            }
        )
    return comparisons


def score_acceptance_gates(
    comparisons: Sequence[Mapping[str, Any]],
    causal_summary: Mapping[str, Any],
) -> dict[str, JsonDict]:
    """Score every frozen efficacy and chronology criterion independently."""

    by_id = {str(row["comparison_id"]): row for row in comparisons}

    def upper(comparison_id: str) -> float:
        return float(by_id[comparison_id]["ci95"][1])

    def estimate(comparison_id: str) -> float:
        return float(by_id[comparison_id]["estimate"])

    definitions = (
        (
            "future_error_vs_frozen_upper_ci95_lt_zero",
            "<0",
            upper("future_error_vs_frozen"),
            upper("future_error_vs_frozen") < 0,
        ),
        (
            "future_error_vs_reset_upper_ci95_lt_zero",
            "<0",
            upper("future_error_vs_reset"),
            upper("future_error_vs_reset") < 0,
        ),
        (
            "recurrence_error_vs_destructive_upper_ci95_lt_zero",
            "<0",
            upper("recurrence_error_vs_destructive"),
            upper("recurrence_error_vs_destructive") < 0,
        ),
        (
            "recurrence_error_vs_shuffled_upper_ci95_lt_zero",
            "<0",
            upper("recurrence_error_vs_shuffled"),
            upper("recurrence_error_vs_shuffled") < 0,
        ),
        (
            "recurrence_error_increase_vs_frozen_lte_0_02",
            "<=0.02",
            estimate("recurrence_error_vs_frozen"),
            estimate("recurrence_error_vs_frozen") <= 0.02,
        ),
        (
            "false_accept_vs_frozen_upper_ci95_lte_zero",
            "<=0",
            upper("false_accept_vs_frozen"),
            upper("false_accept_vs_frozen") <= 0,
        ),
        (
            "false_accept_vs_reset_upper_ci95_lte_zero",
            "<=0",
            upper("false_accept_vs_reset"),
            upper("false_accept_vs_reset") <= 0,
        ),
        (
            "valid_reactivation_gt_zero",
            ">0",
            int(causal_summary["valid_reactivation_count"]),
            int(causal_summary["valid_reactivation_count"]) > 0,
        ),
        (
            "later_changed_decision_gt_zero",
            ">0",
            int(causal_summary["later_changed_decision_count"]),
            int(causal_summary["later_changed_decision_count"]) > 0,
        ),
        (
            "pre_release_difference_eq_zero",
            "==0",
            int(causal_summary["pre_release_difference_count"]),
            int(causal_summary["pre_release_difference_count"]) == 0,
        ),
        (
            "positive_control_headroom_gt_zero",
            ">0",
            int(causal_summary["positive_control_changed_decision_count"]),
            int(causal_summary["positive_control_changed_decision_count"]) > 0,
        ),
    )
    return {
        name: {"criterion": name, "expected": expected, "actual": actual, "pass": passed}
        for name, expected, actual, passed in definitions
    }


def classify_result(
    gates: Mapping[str, Mapping[str, Any]],
    *,
    run_complete: bool,
) -> JsonDict:
    """Keep completed nulls separate from externally blocked or incomplete runs."""

    learning_value = int(run_complete and all(row.get("pass") is True for row in gates.values()))
    no_headroom = (
        gates.get("later_changed_decision_gt_zero", {}).get("pass") is not True
        or gates.get("positive_control_headroom_gt_zero", {}).get("pass") is not True
    )
    if learning_value:
        verdict = "complete_circular_positive: validated archive reuse passed every frozen gate"
        verdict_class = "circular_positive"
    elif no_headroom:
        verdict = (
            "complete_null: inconclusive recurrence learning because changed output or "
            "positive-control headroom was absent"
        )
        verdict_class = "null"
    else:
        verdict = "complete_null: validated archive reuse did not pass every frozen learning gate"
        verdict_class = "null"
    return {
        "recurrence_learning_value_score": learning_value,
        "verdict_class": verdict_class,
        "honest_verdict": verdict,
        "family_retired": False,
    }


def latency_summary(panel: LearningPanel, *, total_cpu_time_s: float) -> JsonDict:
    """Reduce measured costs while keeping the 100-times hardware goal unclaimed."""

    operations = {
        name: {
            "count": len(values),
            "p50_ns": _percentile(values, 0.50),
            "p95_ns": _percentile(values, 0.95),
            "total_ns": sum(values),
        }
        for name, values in panel.costs.items()
    }
    return {
        "clock": "time.perf_counter_ns",
        "operations": operations,
        "lookup_p50_ns": operations["lookup_ns"]["p50_ns"],
        "lookup_p95_ns": operations["lookup_ns"]["p95_ns"],
        "update_p50_ns": operations["update_ns"]["p50_ns"],
        "update_p95_ns": operations["update_ns"]["p95_ns"],
        "maximum_serialized_state_bytes": panel.maximum_state_bytes,
        "total_cpu_time_s": total_cpu_time_s,
        "hardware_target_x": 100.0,
        "measured_hardware_acceleration_x": 1.0,
        "hardware_target_gap_x": 99.0,
        "hardware_target_met": False,
        "performance_target_required_for_efficacy": False,
    }


def _control_release(event_id: str, label: str, index: int) -> JsonDict:
    """Build one finite released witness for transaction boundary controls."""

    return {
        "event_id": event_id,
        "family_id": "lower_bound",
        "numeric_value": 0,
        "observed_label": label,
        "role": "support",
        "request_index": index,
        "release_index": index,
    }


def run_e2e_controls(root: Path) -> list[JsonDict]:
    """Exercise E2E-007 rejection, durable restore, and byte-identical rollback."""

    active = exp7226.PackedBeliefController.from_survivors(
        {family: {0} for family in exp7226.FAMILIES}
    )
    controller = exp7240.ArchivedBeliefController.from_active(active)
    state_path = root / "e2e" / "controller.json"
    controller.save(state_path)
    initial_bytes = controller.state_bytes()
    rejected = False
    try:
        controller.commit_batch(
            [_control_release("rejected", "reject", 1)],
            current_cycle=1,
            expected_parent_hash="sha256:" + "0" * 64,
            state_path=state_path,
        )
    except exp7240.ArchiveCommitRejected:
        rejected = True
    rejected_row = {
        "control": "rejected_stale_parent",
        "passed": rejected and controller.state_bytes() == initial_bytes,
        "parent_sha256": transactional.sha256_bytes(initial_bytes),
        "unchanged_sha256": controller.state_hash(),
        "source_sha256": transactional.sha256_json(_control_release("rejected", "reject", 1)),
    }
    receipt = controller.commit_batch(
        [_control_release("commit", "reject", 2)],
        current_cycle=2,
        expected_parent_hash=controller.state_hash(),
        state_path=state_path,
    )
    restored = exp7240.ArchivedBeliefController.load(state_path)
    restore_row = {
        "control": "fresh_process_restore",
        "passed": restored.state_bytes() == controller.state_bytes(),
        "state_sha256": restored.state_hash(),
        "source_sha256": transactional.sha256_json(receipt["release_order"]),
    }
    rollback = controller.rollback(receipt, state_path=state_path)
    rollback_row = {
        "control": "byte_identical_rollback",
        "passed": rollback["byte_identical"] is True and controller.state_bytes() == initial_bytes,
        "byte_identical": rollback["byte_identical"],
        "restored_state_sha256": rollback["restored_state_hash"],
        "source_sha256": transactional.sha256_json(
            {"parent_hash": receipt["parent_hash"], "new_state_hash": receipt["new_state_hash"]}
        ),
    }
    return [rejected_row, rollback_row, restore_row]


def _sample_budget(stream_ids: Sequence[str], *, complete: bool) -> JsonDict:
    """Return the frozen attempt, completion, censoring, and stopping declaration."""

    units = len(stream_ids)
    events = units * exp7240.EVENTS_PER_STREAM
    arm_rows = events * len(exp7240.ARMS)
    return {
        "independent_units_planned": units,
        "independent_units_attempted": units if complete else 0,
        "independent_units_completed": units if complete else 0,
        "independent_units_censored": 0 if complete else units,
        "planned_events": events,
        "attempted_events": events if complete else 0,
        "completed_events": events if complete else 0,
        "planned_arm_event_rows": arm_rows,
        "completed_arm_event_rows": arm_rows if complete else 0,
        "censored_arm_event_rows": 0 if complete else arm_rows,
        "stopping_rule": "all predeclared streams once; no outcome-based extension",
    }


def _empty_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    paths: ExperimentPaths,
    stream_ids: Sequence[str],
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Create every required field before a block or completed result is classified."""

    summary = gate_summary(checks)
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
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": socket.gethostname(),
        "duration_s": duration_s,
        "phase_spans_s": {},
        "MODEL_SPECS": [],
        "model_invoked": False,
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "current_inference_count": 0,
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": _sample_budget(stream_ids, complete=False),
        "random_seed": {
            "root": RANDOM_SEED,
            "streams": [
                exp7240.STREAM_SEEDS[int(stream_id.split("-")[1]) - 1] for stream_id in stream_ids
            ],
            "bootstrap": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "schedule_frozen_before_labels": True,
        },
        "reproducibility_checksum": "",
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": (
            "blocked_external_precondition:"
            + str(summary.get("failed_check") or "unknown_external_gate")
        ),
        "acceptance_gate_results": {},
        "recurrence_run_complete_score": 0,
        "recurrence_learning_value_score": 0,
        "continuous_self_learning_task": True,
        "continuous_learning_counts": {
            "addition_count": 0,
            "deactivation_count": 0,
            "validated_reactivation_count": 0,
        },
        "decision_rows_path": {
            "path": str(paths.decision_rows),
            "sha256": None,
            "row_count": 0,
            "format": "jsonl",
        },
        "paired_seed_rows": [],
        "memory_state_manifest": {"path": str(paths.state_manifest), "sha256": None},
        "operation_receipts_path": {"path": str(paths.operation_rows), "sha256": None},
        "evidence_manifest_path": {"path": str(paths.evidence_manifest), "sha256": None},
        "comparison_rows": [],
        "causal_summary": {},
        "latency_summary": {"operations": {}},
        "stream_contract": {},
        "arm_contract": {},
        "e2e_control_summary": {},
        "sidecar_write_receipts": [],
        "family_retired": False,
        "no_model_weight_mutation": True,
        "methodology": {
            "method": "prospective sealed-stream CPU exact-solver replay",
            "independent_unit": "stream_seed",
            "held_out_authority": "private evaluator joined after prediction and query receipts",
            "natural_language_transfer_evidence": False,
            "live_llm_evidence": False,
        },
        "default_pipeline_modified": False,
        "publication_performed": False,
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    upstream: Mapping[str, Any],
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    duration_s: float = 0.0,
    started_at: str | None = None,
) -> JsonDict:
    """Return a row-free terminal result when an external gate prevents CPU replay."""

    del upstream
    selected = tuple(
        stream_ids
        if stream_ids is not None
        else (f"stream-{index + 1:02d}" for index in range(exp7240.STREAM_COUNT))
    )
    now = datetime.now(UTC).isoformat()
    artifact = _empty_artifact(
        checks,
        source_hashes,
        paths,
        selected,
        started_at=started_at or now,
        completed_at=now,
        duration_s=duration_s,
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash frozen settings, source bytes, raw receipts, comparisons, gates, and verdict."""

    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "status",
        "run_date",
        "preconditions_checked",
        "inference_substrate",
        "inference_substrate_class",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "random_seed",
        "gate_check_summary",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
        "acceptance_gate_results",
        "recurrence_run_complete_score",
        "recurrence_learning_value_score",
        "decision_rows_path",
        "paired_seed_rows",
        "memory_state_manifest",
        "operation_receipts_path",
        "evidence_manifest_path",
        "comparison_rows",
        "causal_summary",
        "family_retired",
    )
    return transactional.sha256_json({key: deepcopy(artifact.get(key)) for key in keys})


def _continuous_counts(operation_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count finite-state additions, contradiction deactivations, and valid returns."""

    validated = [row for row in operation_rows if row.get("arm") == "validation_selected_archive"]
    additions = sum(int(row.get("archive_creation_count", 0)) for row in validated)
    reactivations = sum(int(row.get("valid_reactivation_count", 0)) for row in validated)
    return {
        "addition_count": additions,
        "deactivation_count": additions,
        "validated_reactivation_count": reactivations,
    }


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    progress: bool = False,
) -> JsonDict:
    """Authenticate, replay, reduce, retain sidecars, and cold-check one terminal object."""

    selected = tuple(
        stream_ids
        if stream_ids is not None
        else (f"stream-{index + 1:02d}" for index in range(exp7240.STREAM_COUNT))
    )
    monotonic_start = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: JsonDict = {}
    if progress:
        _progress(0, "start", "authenticate requirements, sources, outputs, and Exp7240")
    phase_start = time.monotonic()
    checks, source_hashes, upstream = collect_preconditions(repo_root, paths)
    spans["phase_0_preconditions"] = time.monotonic() - phase_start
    _atomic_write(
        paths.provisional_checkpoint,
        transactional.canonical_json_bytes(
            {"schema": SCHEMA, "status": "in_progress", "phase": 0, "checks": checks}
        ),
    )
    if gate_summary(checks)["passed"] is not True:
        artifact = build_blocked_artifact(
            checks,
            source_hashes,
            upstream,
            paths,
            stream_ids=selected,
            duration_s=time.monotonic() - monotonic_start,
            started_at=started_at,
        )
        if progress:
            _progress(0, "end", "external precondition failed; no CPU replay ran")
            for phase in range(1, 8):
                _progress(phase, "start", "skipped after blocking precondition")
                _progress(phase, "end", "skipped after blocking precondition")
        return artifact
    if progress:
        _progress(0, "end", "all external preconditions passed")

    phase_start = time.monotonic()
    if progress:
        _progress(1, "start", "activate flushed progress and monotonic operation clocks")
    spans["phase_1_progress_contract"] = time.monotonic() - phase_start
    if progress:
        _progress(1, "end", "progress contract active")

    phase_start = time.monotonic()
    if progress:
        _progress(2, "start", "confirm zero model loads, generations, and inference calls")
        print("phase 2 BEFORE model load: no model load is scheduled", flush=True)
        print("phase 2 AFTER model load: model load count remains zero", flush=True)
        print("phase 2 BEFORE generation: no generation is scheduled", flush=True)
        print("phase 2 AFTER generation: generation count remains zero", flush=True)
    spans["phase_2_no_llm"] = time.monotonic() - phase_start
    if progress:
        _progress(2, "end", "MODEL_SPECS is empty and current invocation counts are zero")

    phase_start = time.monotonic()
    if progress:
        _progress(3, "start", "load authenticated public, release, and private stream views")
    views = load_stream_views(repo_root, upstream)
    stream_errors = exp7240.stream_conformance_errors(views)
    if stream_errors:
        raise ValueError("exp7240_stream_conformance:" + ",".join(stream_errors))
    spans["phase_3_stream_load"] = time.monotonic() - phase_start
    if progress:
        _progress(3, "end", "32 sealed streams conform and labels remain separately loaded")

    phase_start = time.monotonic()
    if progress:
        _progress(4, "start", "freeze independent seeds and 10000-draw bootstrap schedule")
    frozen_seed_receipt = transactional.sha256_json(
        {
            "stream_ids": selected,
            "stream_seeds": [
                exp7240.STREAM_SEEDS[int(stream_id.split("-")[1]) - 1] for stream_id in selected
            ],
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": bootstrap_draws,
        }
    )
    spans["phase_4_frozen_schedule"] = time.monotonic() - phase_start
    if progress:
        _progress(4, "end", f"schedule_sha256={frozen_seed_receipt}")

    phase_start = time.monotonic()
    if progress:
        _progress(5, "start", "BEFORE six-arm CPU benchmark")
        print("phase 5 BEFORE benchmark: prospective controller replay", flush=True)
    panel = run_learning_panel(views, stream_ids=selected, progress=progress)
    spans["phase_5_cpu_benchmark"] = time.monotonic() - phase_start
    if progress:
        print("phase 5 AFTER benchmark: every selected event-and-arm row completed", flush=True)
        _progress(5, "end", f"decision_rows={len(panel.decision_rows)}")

    phase_start = time.monotonic()
    if progress:
        _progress(6, "start", "reduce seed rows, bootstrap comparisons, and frozen gates")
    conformance_errors = panel_conformance_errors(panel, expected_stream_ids=selected)
    if conformance_errors:
        raise ValueError("learning_panel_conformance:" + ",".join(conformance_errors))
    comparisons = build_comparison_rows(panel.paired_seed_rows, draws=bootstrap_draws)
    gates = score_acceptance_gates(comparisons, panel.causal_summary)
    planned_rows = len(selected) * exp7240.EVENTS_PER_STREAM * len(exp7240.ARMS)
    run_complete = len(panel.decision_rows) == planned_rows
    classification = classify_result(gates, run_complete=run_complete)
    spans["phase_6_reduction_and_gates"] = time.monotonic() - phase_start
    if progress:
        _progress(6, "end", f"learning_value={classification['recurrence_learning_value_score']}")

    phase_start = time.monotonic()
    if progress:
        _progress(7, "start", "run E2E-007 controls and write cold-audit sidecars")
    e2e_rows = run_e2e_controls(paths.provisional_checkpoint.parent / "experiment_7241_e2e")
    if any(row["passed"] is not True for row in e2e_rows):
        raise ValueError("e2e_control_failed")
    decision_write = _atomic_write(paths.decision_rows, _jsonl_bytes(panel.decision_rows))
    operation_write = _atomic_write(
        paths.operation_rows,
        _jsonl_bytes(panel.operation_receipts),
    )
    state_manifest_value = {
        "schema": "carnot.exp7241.memory_state_manifest.v1",
        "entry_count": len(panel.state_entries),
        "entries": panel.state_entries,
    }
    state_write = _atomic_write(
        paths.state_manifest,
        transactional.canonical_json_bytes(state_manifest_value),
    )
    evidence_value = {
        "schema": "carnot.exp7241.evidence_manifest.v1",
        "historical_source_details": {
            "experiment_id": upstream.get("experiment_id"),
            "path": str(DEFAULT_UPSTREAM_ARTIFACT),
            "sha256": source_hashes[str(_resolve(repo_root, DEFAULT_UPSTREAM_ARTIFACT))],
            "status": upstream.get("status"),
            "honest_verdict": upstream.get("honest_verdict"),
            "recurrence_fixture_ready_score": upstream.get("recurrence_fixture_ready_score"),
        },
        "synthetic_negative_receipts": [
            row for row in e2e_rows if row["control"] == "rejected_stale_parent"
        ],
        "e2e_control_rows": e2e_rows,
    }
    evidence_write = _atomic_write(
        paths.evidence_manifest,
        transactional.canonical_json_bytes(evidence_value),
    )
    sidecar_writes = [decision_write, operation_write, state_write, evidence_write]
    for receipt in sidecar_writes:
        source_hashes[str(receipt["path"])] = str(receipt["sha256"])
    spans["phase_7_e2e_and_sidecars"] = time.monotonic() - phase_start
    if progress:
        _progress(7, "end", "rejection, restore, rollback, raw rows, and state bytes retained")

    completed_at = datetime.now(UTC).isoformat()
    duration = time.monotonic() - monotonic_start
    artifact = _empty_artifact(
        checks,
        source_hashes,
        paths,
        selected,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=duration,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "phase_spans_s": spans,
            "rows": panel.paired_seed_rows,
            "sample_size_budget": _sample_budget(selected, complete=run_complete),
            "random_seed": {
                "root": RANDOM_SEED,
                "streams": [
                    exp7240.STREAM_SEEDS[int(stream_id.split("-")[1]) - 1] for stream_id in selected
                ],
                "bootstrap": BOOTSTRAP_SEED,
                "bootstrap_draws": bootstrap_draws,
                "schedule_frozen_before_labels": True,
                "schedule_sha256": frozen_seed_receipt,
            },
            "verdict_class": classification["verdict_class"],
            "honest_verdict": classification["honest_verdict"],
            "acceptance_gate_results": gates,
            "recurrence_run_complete_score": int(run_complete),
            "recurrence_learning_value_score": classification["recurrence_learning_value_score"],
            "continuous_learning_counts": _continuous_counts(panel.operation_receipts),
            "decision_rows_path": {
                **decision_write,
                "row_count": len(panel.decision_rows),
                "format": "jsonl",
            },
            "paired_seed_rows": panel.paired_seed_rows,
            "memory_state_manifest": {
                **state_write,
                "entry_count": len(panel.state_entries),
            },
            "operation_receipts_path": {
                **operation_write,
                "row_count": len(panel.operation_receipts),
                "format": "jsonl",
            },
            "evidence_manifest_path": {
                **evidence_write,
                "e2e_control_count": len(e2e_rows),
            },
            "comparison_rows": comparisons,
            "causal_summary": panel.causal_summary,
            "latency_summary": latency_summary(panel, total_cpu_time_s=duration),
            "stream_contract": {
                **deepcopy(views.manifest),
                "private_authority_sha256": source_hashes[
                    str(_resolve(repo_root, DEFAULT_STREAM_ROOT / "private_evaluator.jsonl"))
                ],
                "release_schedule_sha256": source_hashes[
                    str(_resolve(repo_root, DEFAULT_STREAM_ROOT / "release_schedule.jsonl"))
                ],
            },
            "arm_contract": deepcopy(upstream["arm_contract"]),
            "e2e_control_summary": {
                "row_count": len(e2e_rows),
                "passed_count": sum(int(row["passed"] is True) for row in e2e_rows),
                "failed_count": sum(int(row["passed"] is not True) for row in e2e_rows),
            },
            "sidecar_write_receipts": sidecar_writes,
            "family_retired": classification["family_retired"],
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
    _atomic_write(paths.provisional_checkpoint, transactional.canonical_json_bytes(artifact))
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
    check_files: bool = False,
) -> list[str]:
    """Cold-check identity, exact rows, recomputed gates, receipts, and classification."""

    selected = tuple(
        expected_stream_ids
        if expected_stream_ids is not None
        else (f"stream-{index + 1:02d}" for index in range(exp7240.STREAM_COUNT))
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
    add(artifact.get("execution_venue") != "host", "execution_venue")
    add(artifact.get("verifier_is_oracle") is not True, "oracle_declaration")
    add(artifact.get("no_model_weight_mutation") is not True, "weight_mutation")
    add(artifact.get("default_pipeline_modified") is not False, "production_default")
    add(
        artifact.get("verdict_class")
        not in {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"},
        "verdict_class",
    )
    add(
        artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
        "reproducibility_checksum",
    )
    if artifact.get("status") == "blocked":
        add(
            artifact.get("rows") != []
            or artifact.get("paired_seed_rows") != []
            or artifact.get("recurrence_run_complete_score") != 0,
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
    paired = artifact.get("paired_seed_rows", [])
    expected_units = len(selected) * len(exp7240.ARMS)
    add(
        len(paired) != expected_units
        or artifact.get("rows") != paired
        or {(row.get("stream_id"), row.get("arm")) for row in paired}
        != {(stream_id, arm) for stream_id in selected for arm in exp7240.ARMS},
        "paired_seed_rows",
    )
    budget = artifact.get("sample_size_budget", {})
    expected_decisions = len(selected) * exp7240.EVENTS_PER_STREAM * len(exp7240.ARMS)
    add(
        budget.get("completed_arm_event_rows") != expected_decisions
        or artifact.get("recurrence_run_complete_score") != 1,
        "completion_score",
    )
    draws = int(artifact.get("random_seed", {}).get("bootstrap_draws", BOOTSTRAP_DRAWS))
    expected_comparisons = build_comparison_rows(paired, draws=draws)
    add(artifact.get("comparison_rows") != expected_comparisons, "comparison_rows")
    causal = artifact.get("causal_summary", {})
    expected_gates = score_acceptance_gates(expected_comparisons, causal)
    add(artifact.get("acceptance_gate_results") != expected_gates, "acceptance_gate_results")
    expected_classification = classify_result(expected_gates, run_complete=True)
    add(
        artifact.get("recurrence_learning_value_score")
        != expected_classification["recurrence_learning_value_score"]
        or artifact.get("verdict_class") != expected_classification["verdict_class"]
        or artifact.get("honest_verdict") != expected_classification["honest_verdict"]
        or artifact.get("family_retired") != expected_classification["family_retired"],
        "terminal_classification",
    )
    e2e = artifact.get("e2e_control_summary", {})
    add(
        e2e.get("row_count") != 3 or e2e.get("passed_count") != 3 or e2e.get("failed_count") != 0,
        "e2e_controls",
    )
    if check_files:
        file_receipts = (
            artifact.get("decision_rows_path", {}),
            artifact.get("memory_state_manifest", {}),
            artifact.get("operation_receipts_path", {}),
            artifact.get("evidence_manifest_path", {}),
        )
        add(
            any(
                not isinstance(receipt, Mapping) or not _receipt_matches(repo_root, receipt)
                for receipt in file_receipts
            ),
            "sidecar_hashes",
        )
        decision_receipt = artifact.get("decision_rows_path", {})
        add(decision_receipt.get("row_count") != expected_decisions, "decision_row_count")
        for path, expected in artifact.get("source_artifact_hashes", {}).items():
            add(
                expected is None or _sha256_path(_resolve(repo_root, path)) != expected,
                "source_hashes",
            )
    return errors


def write_artifact(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
) -> None:
    """Cold-validate and publish terminal bytes through one atomic rename."""

    errors = validate_artifact(
        artifact,
        repo_root=repo_root,
        expected_stream_ids=expected_stream_ids,
        check_files=True,
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    _atomic_write(path, transactional.canonical_json_bytes(dict(artifact)))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed execution date and an optional private output root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CPU panel and atomically publish its validated terminal artifact."""

    args = _parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"run_date_must_be_{RUN_DATE}")
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    artifact = build_and_seal(REPO_ROOT, paths, progress=True)
    _progress(8, "start", "BEFORE final cold validation and atomic terminal write")
    print("phase 8 BEFORE final validation", flush=True)
    errors = validate_artifact(artifact, repo_root=REPO_ROOT, check_files=True)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    print("phase 8 AFTER final validation", flush=True)
    print("phase 8 BEFORE atomic terminal write", flush=True)
    write_artifact(paths.artifact, artifact, repo_root=REPO_ROOT)
    print("phase 8 AFTER atomic terminal write", flush=True)
    _progress(8, "end", f"wrote {paths.artifact}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin wrapper owns execution.
    raise SystemExit(main())
