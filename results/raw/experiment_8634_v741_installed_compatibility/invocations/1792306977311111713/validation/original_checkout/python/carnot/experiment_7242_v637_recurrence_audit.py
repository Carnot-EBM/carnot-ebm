"""Cold-audit recurrence learning from raw pre-release decisions.

This audit reads the V637 fixture and learner deliverables. A fresh process
rebuilds every seed-and-arm metric from raw rows. It also restores final state,
denies private authority during state probes, and runs transaction mutations.

Spec refs: REQ-CL-7242 and SCENARIO-CL-7242-*.
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
import time
from typing import Any

from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7226_v636_belief_compiler as exp7226
from carnot import experiment_7228_v636_belief_cold_audit as exp7228
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot import experiment_7241_v637_recurrence_learning as exp7241
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7242
SCHEMA = "carnot.exp7242.v637_recurrence_audit.v1"
MILESTONE = "2026.09.637"
RUN_DATE = "20260912"
AUDIT_SEED = 7_242_000
BOOTSTRAP_SEED = 7_242_951
BOOTSTRAP_DRAWS = 10_000
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_ARTIFACT = Path("results/experiment_7240_v637_recurrence_fixture.json")
DEFAULT_LEARNER_ARTIFACT = Path("results/experiment_7241_v637_recurrence_learning.json")
DEFAULT_OLD_AUDIT = Path("results/experiment_7228_v636_belief_cold_audit.json")
DEFAULT_PUBLIC_STREAM = Path("results/streams/experiment_7240/public_stream.jsonl")
DEFAULT_PRIVATE_AUTHORITY = Path("results/streams/experiment_7240/private_evaluator.jsonl")
DEFAULT_RELEASE_SCHEDULE = Path("results/streams/experiment_7240/release_schedule.jsonl")
DEFAULT_PUBLIC_MANIFEST = Path("results/streams/experiment_7240/public_manifest.json")
DEFAULT_DECISION_ROWS = Path("results/raw/experiment_7241/decision_rows.jsonl")
DEFAULT_OPERATION_ROWS = Path("results/raw/experiment_7241/operation_receipts.jsonl")
DEFAULT_STATE_MANIFEST = Path("results/checkpoints/experiment_7241_v637_memory_state_manifest.json")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/experiment_7242_v637_recurrence_audit.json")
DEFAULT_MUTATION_RECEIPT_PATH = Path(
    "results/checkpoints/experiment_7242_v637_mutation_receipts.json"
)
DEFAULT_ARTIFACT_PATH = Path("results/experiment_7242_v637_recurrence_audit.json")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
WRAPPER_PATH = REPO_ROOT / "scripts/experiments/experiment_7242_v637_recurrence_audit.py"
EXPECTED_FIXTURE_SHA256 = "sha256:0f12abc839f3d70006698ebaa5169f12ab1cae9dc7526c6d124078ec7a6baf74"
EXPECTED_LEARNER_SHA256 = "sha256:ebf779e3b79a07c82300758efb4392fd9c90cda833609bfd7c2cf105ba4710ce"
EXPECTED_OLD_AUDIT_SHA256 = (
    "sha256:9ba66b80dcd221f34bd452737c4ac151b0e4c11cf16b73ccfe407aaaa17988e3"
)
RESULT_PREFIX = exp7228.RESULT_PREFIX

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7228_v636_belief_cold_audit.py"),
    Path("python/carnot/experiment_7227_v636_belief_learning.py"),
    Path("python/carnot/experiment_7240_v637_recurrence_fixture.py"),
    Path("python/carnot/experiment_7241_v637_recurrence_learning.py"),
    Path("python/carnot/experiment_7242_v637_recurrence_audit.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/experiments/experiment_7242_v637_recurrence_audit.py"),
    Path("tests/python/test_experiment_7242_v637_recurrence_audit.py"),
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
    "recurrence_audit_complete_score",
    "recurrence_promotion_score",
    "recomputed_seed_rows",
    "mutation_receipt_path",
    "continuous_self_learning_task",
    "comparison_rows",
    "raw_check_rows",
    "restore_rows",
    "causal_control_rows",
    "causal_summary",
    "control_summary",
    "reducer_process_receipt",
    "upstream_receipts",
    "no_model_weight_mutation",
    "cross_domain_transfer_claimed",
    "hardware_speedup_claimed",
    "default_pipeline_modified",
    "publication_performed",
    "methodology",
    "validation_command_receipts",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Version the artifact and bind experiment_id and milestone to this task.",
    "experiment_id": "A fixed identifier prevents another task from supplying this audit.",
    "milestone": "Bind this result to the V637 recurrence-memory audit.",
    "status": "Terminal complete or blocked only; unfinished work uses a separate checkpoint path.",
    "run_date": "Use 20260912; retain actual UTC start and end timestamps.",
    "started_at_utc": "Record the actual UTC start separately from the fixed run date.",
    "completed_at_utc": "Record the actual UTC end separately from the fixed run date.",
    "field_principles": "Keep ordinary values at top level; put their explanations in this map.",
    "preconditions_checked": "Observed paths, resources, model identity and upstream checks before expensive work.",
    "inference_substrate": "Use the recognized literal for the operation actually executed.",
    "inference_substrate_class": "Actual class determines the duration floor; never pad duration or relabel to pass.",
    "execution_venue": "Top-level orchestration is host; board rows name kv260, gatemate or polarfire.",
    "execution_host": "Actual hostname, distinct from execution_venue.",
    "duration_s": "Measured monotonic elapsed work; record phase spans separately.",
    "phase_spans_s": "Keep measured monotonic spans for every numbered audit phase.",
    "MODEL_SPECS": "Models actually invoked; [] for tasks with no LLM.",
    "model_invoked": "Current task execution only; historical sources and injected fixtures are separate.",
    "current_model_load_count": "Count only model loads in this task.",
    "current_generation_count": "Count only generations in this task.",
    "current_inference_count": "Count only model inference calls in this task.",
    "source_artifact_hashes": "Hash source code, public inputs, private evaluator inputs and raw output files.",
    "rows": "Every comparison retains one row per independent unit and arm, with errors and abstentions.",
    "sample_size_budget": "Predeclared independent units, attempted/completed/censored units, and stopping rule.",
    "random_seed": "Freeze seeds and schedules before observing evaluation labels.",
    "reproducibility_checksum": "Hash the exact settings, inputs and raw rows supporting the result.",
    "gate_check_summary": "Every blocked verdict names check, upstream, artifact_field, expected and observed value.",
    "verifier_is_oracle": "True when the verification authority also defines correctness; separate code is insufficient independence.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial.",
    "honest_verdict": "Completed findings start complete_ or complete:. External absence starts blocked_.",
    "acceptance_gate_results": "Preserve each frozen criterion, actual value and pass/fail independently of task completion.",
    "recurrence_audit_complete_score": "Complete cold reduction or explicit externally blocked disposition.",
    "recurrence_promotion_score": "Independent reproduction of efficacy plus all causal safety criteria.",
    "recomputed_seed_rows": "Full per-stream, per-arm metrics independent of producer aggregate calculations.",
    "mutation_receipt_path": "Separate hashed negative-fixture sidecar, never current invocation provenance.",
    "continuous_self_learning_task": "True; assess prospective utility and rollback of learned constraint reuse.",
    "comparison_rows": "Rebuild fixed paired-seed intervals without producer aggregates.",
    "raw_check_rows": "Retain chronology, request-index joins, hashes, and denominator checks.",
    "restore_rows": "Compare restored actions, energies, active hashes, and archive hashes.",
    "causal_control_rows": "Keep update withholding and every unsafe transaction mutation.",
    "causal_summary": "Recompute later change, pre-release difference, and reactivation counts.",
    "control_summary": "Retain identical-input drift and shuffled-candidate coverage.",
    "reducer_process_receipt": "Prove reduction and restore ran in one isolated fresh process.",
    "upstream_receipts": "Keep Exp7240 readiness separate from Exp7241 learning value.",
    "no_model_weight_mutation": "This audit changes finite memory only, not model weights.",
    "cross_domain_transfer_claimed": "This finite fixture supplies no cross-domain transfer evidence.",
    "hardware_speedup_claimed": "CPU replay does not imply the future 100-times hardware target.",
    "default_pipeline_modified": "The audit does not change a production default.",
    "publication_performed": "This task does not publish, upload, or submit results.",
    "methodology": "State raw reduction, fresh restore, authority denial, and oracle limits.",
    "validation_command_receipts": "Record scoped validation commands without claiming unrun checks.",
}

unwrap_principled = exp7213.unwrap_principled
gate_check = exp7213.gate_check
gate_summary = exp7213.gate_summary
quarantine_state = exp7213.quarantine_state
COMPARISON_SPECS = tuple(exp7241.COMPARISON_SPECS)


class AuditEvidenceError(ValueError):
    """Reject malformed evidence before it can affect an audit verdict."""


class ExperimentPaths:
    """Keep provisional, mutation, and terminal bytes at separate paths."""

    def __init__(self, checkpoint: Path, mutation_receipts: Path, artifact: Path) -> None:
        self.checkpoint = checkpoint
        self.mutation_receipts = mutation_receipts
        self.artifact = artifact

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the paths used by the roadmap command."""

        return cls(DEFAULT_CHECKPOINT_PATH, DEFAULT_MUTATION_RECEIPT_PATH, DEFAULT_ARTIFACT_PATH)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put every test-owned output below one caller-owned directory."""

        checkpoint_root = root / "checkpoints"
        return cls(
            checkpoint_root / DEFAULT_CHECKPOINT_PATH.name,
            checkpoint_root / DEFAULT_MUTATION_RECEIPT_PATH.name,
            root / DEFAULT_ARTIFACT_PATH.name,
        )


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Emit one truthful phase boundary for the external watchdog."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _resolve(repo_root: Path, path: str | Path) -> Path:
    """Resolve repository-relative evidence without changing absolute paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _sha256_path(path: Path) -> str | None:
    """Hash exact file bytes and preserve absence as an observed failure."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object while malformed input remains an empty observation."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[JsonDict]:
    """Load JSONL objects and reject malformed or non-object records."""

    rows: list[JsonDict] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AuditEvidenceError("invalid_jsonl_non_object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise AuditEvidenceError("invalid_jsonl") from error
    return rows


def _safe_text(path: Path) -> str:
    """Read contract text without hiding a missing-file precondition."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating output evidence."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _task_identity(text: str) -> JsonDict:
    """Read only the Exp7242 roadmap block for identity checks."""

    match = re.search(r"(?ms)^- id: exp7242-recurrence-audit\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(1)
    milestone = re.search(r"(?m)^  milestone: (.+)$", block)
    deliverable = re.search(r"(?m)^  deliverable: (.+)$", block)
    return {
        "id": "exp7242-recurrence-audit" if block else None,
        "milestone": None if milestone is None else milestone.group(1).strip(),
        "deliverable": None if deliverable is None else deliverable.group(1).strip(),
    }


def _receipt_matches(repo_root: Path, receipt: Any, expected_path: Path) -> bool:
    """Require one declared path and hash to match the exact current bytes."""

    if not isinstance(receipt, Mapping):
        return False
    declared_path = receipt.get("path")
    digest = receipt.get("sha256")
    if not declared_path or not digest:
        return False
    actual_path = _resolve(repo_root, str(declared_path))
    return (
        actual_path.resolve() == _resolve(repo_root, expected_path).resolve()
        and _sha256_path(actual_path) == digest
    )


def _artifact_checksum_valid(module: Any, artifact: Mapping[str, Any]) -> bool:
    """Treat malformed producer fields as a failed checksum observation."""

    try:
        return module.reproducibility_checksum(artifact) == artifact.get("reproducibility_checksum")
    except (KeyError, TypeError, ValueError):
        return False


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    fixture_artifact: Path = DEFAULT_FIXTURE_ARTIFACT,
    learner_artifact: Path = DEFAULT_LEARNER_ARTIFACT,
) -> tuple[list[JsonDict], dict[str, JsonDict], dict[str, str | None]]:
    """Authenticate both producers and every raw byte before fresh reduction."""

    root = Path(repo_root)
    fixture_path = _resolve(root, fixture_artifact)
    learner_path = _resolve(root, learner_artifact)
    old_audit_path = _resolve(root, DEFAULT_OLD_AUDIT)
    fixture = _load_object(fixture_path)
    learner = _load_object(learner_path)
    old_audit = _load_object(old_audit_path)
    source_hashes: dict[str, str | None] = {
        str(path): _sha256_path(_resolve(root, path)) for path in SOURCE_PATHS
    }
    source_hashes[str(fixture_artifact)] = _sha256_path(fixture_path)
    source_hashes[str(learner_artifact)] = _sha256_path(learner_path)
    source_hashes[str(DEFAULT_OLD_AUDIT)] = _sha256_path(old_audit_path)
    for path in (
        DEFAULT_PUBLIC_STREAM,
        DEFAULT_PRIVATE_AUTHORITY,
        DEFAULT_RELEASE_SCHEDULE,
        DEFAULT_PUBLIC_MANIFEST,
        DEFAULT_DECISION_ROWS,
        DEFAULT_OPERATION_ROWS,
        DEFAULT_STATE_MANIFEST,
    ):
        source_hashes[str(path)] = _sha256_path(_resolve(root, path))

    spec_text = _safe_text(_resolve(root, SPEC_PATH))
    roadmap_text = _safe_text(_resolve(root, "research-roadmap.yaml"))
    exclusion_text = _safe_text(_resolve(root, "ops/exclusion_manifest.yaml"))
    fixture_quarantine = quarantine_state(
        fixture,
        exclusion_text,
        DEFAULT_FIXTURE_ARTIFACT.name,
        "exp7240-recurrence-fixture",
    )
    learner_quarantine = quarantine_state(
        learner,
        exclusion_text,
        DEFAULT_LEARNER_ARTIFACT.name,
        "exp7241-recurrence-learning",
    )
    old_quarantine = quarantine_state(
        old_audit,
        exclusion_text,
        DEFAULT_OLD_AUDIT.name,
        "exp7228-belief-cold-audit",
    )
    imports = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "carnot.experiment_7226_v636_belief_compiler",
            "carnot.experiment_7228_v636_belief_cold_audit",
            "carnot.experiment_7240_v637_recurrence_fixture",
            "carnot.experiment_7241_v637_recurrence_learning",
            "carnot.memory.transactional_constraint_memory",
        )
    }
    output_state = {
        "checkpoint": _path_writable(paths.checkpoint),
        "mutation_receipts": _path_writable(paths.mutation_receipts),
        "artifact": _path_writable(paths.artifact),
        "checkpoint_scoped": paths.checkpoint.parent.name == "checkpoints",
        "mutation_scoped": paths.mutation_receipts.parent.name == "checkpoints",
    }
    source_state = {
        str(path): "nonempty" if source_hashes[str(path)] is not None else "missing"
        for path in SOURCE_PATHS
    }
    fixture_streams = fixture.get("stream_receipts", {})
    fixture_receipts = {
        "public_stream": _receipt_matches(
            root, fixture_streams.get("public_stream", {}), DEFAULT_PUBLIC_STREAM
        ),
        "private_authority": _receipt_matches(
            root, fixture_streams.get("private_authority", {}), DEFAULT_PRIVATE_AUTHORITY
        ),
        "release_schedule": _receipt_matches(
            root, fixture_streams.get("release_schedule", {}), DEFAULT_RELEASE_SCHEDULE
        ),
        "public_manifest": _receipt_matches(
            root, fixture_streams.get("public_manifest", {}), DEFAULT_PUBLIC_MANIFEST
        ),
    }
    learner_receipts = {
        "decision_rows": _receipt_matches(
            root, learner.get("decision_rows_path", {}), DEFAULT_DECISION_ROWS
        ),
        "operation_rows": _receipt_matches(
            root, learner.get("operation_receipts_path", {}), DEFAULT_OPERATION_ROWS
        ),
        "state_manifest": _receipt_matches(
            root, learner.get("memory_state_manifest", {}), DEFAULT_STATE_MANIFEST
        ),
    }
    expected_identity = {
        "id": "exp7242-recurrence-audit",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT_PATH),
    }
    checks = [
        gate_check(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-CL-7242",
            True,
            "## REQ-CL-7242:" in spec_text,
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7242-*",
            8,
            len(set(re.findall(r"SCENARIO-CL-7242-[A-Z-]+", spec_text))),
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
            _task_identity(roadmap_text),
        ),
        gate_check("required_imports", "python", "imports", {k: True for k in imports}, imports),
        gate_check(
            "writable_output_paths",
            "host_filesystem",
            "checkpoint,mutation,artifact",
            {key: True for key in output_state},
            output_state,
        ),
        gate_check(
            "exp7240_artifact_hash",
            "exp7240-recurrence-fixture",
            str(DEFAULT_FIXTURE_ARTIFACT),
            EXPECTED_FIXTURE_SHA256,
            source_hashes[str(fixture_artifact)],
        ),
        gate_check(
            "exp7240_status_and_readiness",
            "exp7240-recurrence-fixture",
            "status,recurrence_fixture_ready_score",
            ["complete", 1],
            [
                fixture.get("status"),
                unwrap_principled(fixture.get("recurrence_fixture_ready_score")),
            ],
        ),
        gate_check(
            "exp7240_checksum",
            "exp7240-recurrence-fixture",
            "reproducibility_checksum",
            True,
            _artifact_checksum_valid(exp7240, fixture),
        ),
        gate_check(
            "exp7240_raw_receipts",
            "exp7240-recurrence-fixture",
            "public,private,release,manifest",
            {key: True for key in fixture_receipts},
            fixture_receipts,
        ),
        gate_check(
            "exp7240_not_quarantined",
            "artifact_metadata_and_ops/exclusion_manifest.yaml",
            "quarantined",
            False,
            fixture_quarantine["quarantined"],
        ),
        gate_check(
            "exp7241_artifact_hash",
            "exp7241-recurrence-learning",
            str(DEFAULT_LEARNER_ARTIFACT),
            EXPECTED_LEARNER_SHA256,
            source_hashes[str(learner_artifact)],
        ),
        gate_check(
            "exp7241_status_and_completion",
            "exp7241-recurrence-learning",
            "status,recurrence_run_complete_score",
            ["complete", 1],
            [
                learner.get("status"),
                unwrap_principled(learner.get("recurrence_run_complete_score")),
            ],
        ),
        gate_check(
            "exp7241_checksum",
            "exp7241-recurrence-learning",
            "reproducibility_checksum",
            True,
            _artifact_checksum_valid(exp7241, learner),
        ),
        gate_check(
            "exp7241_raw_receipts",
            "exp7241-recurrence-learning",
            "decision,operation,state",
            {key: True for key in learner_receipts},
            learner_receipts,
        ),
        gate_check(
            "exp7241_six_arm_contract",
            "exp7241-recurrence-learning",
            "arm_contract.arms",
            list(exp7240.ARMS),
            learner.get("arm_contract", {}).get("arms")
            if isinstance(learner.get("arm_contract"), Mapping)
            else None,
        ),
        gate_check(
            "exp7241_no_current_model",
            "exp7241-recurrence-learning",
            "MODEL_SPECS,model_invoked",
            [[], False],
            [
                unwrap_principled(learner.get("MODEL_SPECS")),
                unwrap_principled(learner.get("model_invoked")),
            ],
        ),
        gate_check(
            "exp7241_not_quarantined",
            "artifact_metadata_and_ops/exclusion_manifest.yaml",
            "quarantined",
            False,
            learner_quarantine["quarantined"],
        ),
        gate_check(
            "exp7228_history_remains_quarantined",
            "exp7228-belief-cold-audit",
            "flagged_adversarial_or_manifest_quarantine",
            [EXPECTED_OLD_AUDIT_SHA256, True],
            [source_hashes[str(DEFAULT_OLD_AUDIT)], old_quarantine["quarantined"]],
        ),
    ]
    return checks, {"exp7240": fixture, "exp7241": learner, "exp7228": old_audit}, source_hashes


def _outcome(prediction: str, exact_label: str) -> tuple[int, int, int]:
    """Count abstention as an error on the complete prospective denominator."""

    abstention = int(prediction == "abstain")
    error = int(abstention == 1 or prediction != exact_label)
    false_accept = int(prediction == "accept" and exact_label == "reject")
    return error, false_accept, abstention


def _percentile(values: Sequence[int | float], probability: float) -> float:
    """Use the fixed nearest-rank rule for deterministic audit summaries."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(probability * (len(ordered) - 1)))))
    return ordered[index]


def _bootstrap_interval(values: Sequence[float], *, draws: int, salt: str) -> JsonDict:
    """Resample complete stream differences as the independent units."""

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


def _indexed_rows(
    rows: Sequence[Mapping[str, Any]], name: str
) -> dict[tuple[str, int], Mapping[str, Any]]:
    """Index one stream view and reject duplicate request indices."""

    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["stream_id"]), int(row["chronology_index"]))
        if key in indexed:
            raise AuditEvidenceError(f"duplicate_{name}_request_index")
        indexed[key] = row
    return indexed


def _operation_counts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], JsonDict], int]:
    """Count archive operations and reject duplicate feedback credit."""

    seen: set[tuple[str, str, str, str]] = set()
    counts: dict[tuple[str, str], JsonDict] = defaultdict(
        lambda: {"archive_hits": 0, "valid_hits": 0, "update_costs": []}
    )
    delivered = 0
    for row in rows:
        key = (
            str(row.get("operation")),
            str(row.get("stream_id")),
            str(row.get("arm")),
            str(row.get("event_id")),
        )
        if key in seen:
            raise AuditEvidenceError("duplicate_feedback_credit")
        seen.add(key)
        if row.get("operation") == "delayed_delivery":
            delivered += 1
        if row.get("operation") == "validation_and_state_write":
            unit = (str(row["stream_id"]), str(row["arm"]))
            counts[unit]["archive_hits"] += int(row.get("archive_reactivation_count", 0))
            counts[unit]["valid_hits"] += int(row.get("valid_reactivation_count", 0))
            counts[unit]["update_costs"].append(int(row.get("cost_ns", 0)))
    return counts, delivered


def reduce_raw_evidence(
    decision_rows: Sequence[Mapping[str, Any]],
    public_rows: Sequence[Mapping[str, Any]],
    release_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
    operation_rows: Sequence[Mapping[str, Any]],
    *,
    expected_stream_ids: Sequence[str],
    events_per_stream: int = exp7240.EVENTS_PER_STREAM,
    warmup_count: int = exp7240.WARMUP_COUNT,
) -> JsonDict:
    """Rebuild all metrics after request-index joins and chronology checks."""

    selected = {str(stream_id) for stream_id in expected_stream_ids}
    public = _indexed_rows(
        [row for row in public_rows if str(row["stream_id"]) in selected], "public"
    )
    releases = _indexed_rows(
        [row for row in release_rows if str(row["stream_id"]) in selected], "release"
    )
    authority = _indexed_rows(
        [row for row in authority_rows if str(row["stream_id"]) in selected], "authority"
    )
    expected_event_keys = {
        (stream_id, index) for stream_id in selected for index in range(events_per_stream)
    }
    if (
        set(public) != expected_event_keys
        or set(releases) != expected_event_keys
        or set(authority) != expected_event_keys
    ):
        raise AuditEvidenceError("missing_request_index_denominator")
    for key in sorted(expected_event_keys):
        public_row = public[key]
        release = releases[key]
        truth = authority[key]
        if not (
            public_row["event_id"] == release["event_id"] == truth["event_id"]
            and public_row["family_id"] == truth["family_id"]
            and int(public_row["numeric_value"]) == int(truth["numeric_value"])
            and release["observed_label"] == truth["exact_label"]
        ):
            raise AuditEvidenceError("request_index_join")

    selected_operations = [row for row in operation_rows if str(row.get("stream_id")) in selected]
    operation_counts, delivered_count = _operation_counts(selected_operations)
    expected_units = {(stream_id, arm) for stream_id in selected for arm in exp7240.ARMS}
    accumulators: dict[tuple[str, str], JsonDict] = {
        unit: {
            "indices": set(),
            "seed": None,
            "future_error": 0,
            "future_count": 0,
            "false_accept": 0,
            "abstention": 0,
            "recurrence_error": 0,
            "recurrence_count": 0,
            "later_changed": 0,
            "pre_release_difference": 0,
            "query_count": 0,
            "lookup_costs": [],
        }
        for unit in expected_units
    }
    event_predictions: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    event_has_released_feedback: dict[tuple[str, int], bool] = {}
    stored_outcome_mismatches = 0
    chronology_count = 0
    seed_match_count = 0
    joined_count = 0
    for row in decision_rows:
        stream_id = str(row["stream_id"])
        if stream_id not in selected:
            continue
        arm = str(row["arm"])
        unit = (stream_id, arm)
        if unit not in accumulators:
            raise AuditEvidenceError("unexpected_arm")
        index = int(row["chronology_index"])
        key = (stream_id, index)
        if index in accumulators[unit]["indices"]:
            raise AuditEvidenceError("duplicate_decision")
        accumulators[unit]["indices"].add(index)
        truth = authority.get(key)
        release = releases.get(key)
        public_row = public.get(key)
        if truth is None or release is None or public_row is None:
            raise AuditEvidenceError("request_index_join")
        if not (
            row["event_id"] == truth["event_id"] == release["event_id"]
            and row["later_released_label"] == truth["exact_label"] == release["observed_label"]
            and row["drift_pattern"] == truth["drift_pattern"]
            and row["regime_id"] == truth["regime_id"]
        ):
            raise AuditEvidenceError("request_index_join")
        if int(row["seed"]) != int(truth["stream_seed"]):
            raise AuditEvidenceError("seed_mismatch")
        seed_match_count += 1
        controller_fields = row.get("controller_input_fields")
        if controller_fields != ["event_id", "family_id", "numeric_value"]:
            raise AuditEvidenceError("hidden_regime_input")
        if row.get("held_out_label_visible_to_controller") is not False:
            raise AuditEvidenceError("hidden_regime_input")
        if (
            int(row["prediction_receipt_completed_ns"]) > int(row["label_accessed_ns"])
            or int(row["query_receipt_completed_ns"]) > int(row["label_accessed_ns"])
            or row.get("prediction_frozen_before_release") is not True
        ):
            raise AuditEvidenceError("premature_label")
        chronology_count += 1
        prediction = str(row["prediction"])
        outcome = _outcome(prediction, str(truth["exact_label"]))
        stored = (
            int(row["full_denominator_error"]),
            int(row["false_accept"]),
            int(row["abstention"]),
        )
        stored_outcome_mismatches += int(outcome != stored)
        accumulator = accumulators[unit]
        accumulator["seed"] = int(row["seed"])
        accumulator["later_changed"] += int(row["later_changed_decision"])
        accumulator["pre_release_difference"] += int(row["pre_release_difference"])
        accumulator["query_count"] += int(row["query_selected"])
        accumulator["lookup_costs"].append(int(row["lookup_cost_ns"]))
        if index >= warmup_count:
            accumulator["future_count"] += 1
            accumulator["future_error"] += outcome[0]
            accumulator["false_accept"] += outcome[1]
            accumulator["abstention"] += outcome[2]
        if row.get("recurrence_eligible") is True:
            accumulator["recurrence_count"] += 1
            accumulator["recurrence_error"] += outcome[0]
        event_predictions[key][arm] = prediction
        event_has_released_feedback[key] = int(row["released_query_count_before"]) > 0
        joined_count += 1

    if stored_outcome_mismatches:
        raise AuditEvidenceError("stored_outcome_mismatch")
    expected_indices = set(range(events_per_stream))
    if any(value["indices"] != expected_indices for value in accumulators.values()):
        raise AuditEvidenceError("missing_decision_denominator")

    recomputed: list[JsonDict] = []
    for stream_id, arm in sorted(expected_units):
        value = accumulators[(stream_id, arm)]
        future_count = int(value["future_count"])
        recurrence_count = int(value["recurrence_count"])
        operations = operation_counts[(stream_id, arm)]
        hits = int(operations["archive_hits"])
        valid_hits = int(operations["valid_hits"])
        recomputed.append(
            {
                "unit_id": f"{stream_id}:{arm}",
                "stream_id": stream_id,
                "seed": int(value["seed"]),
                "arm": arm,
                "metric": "prospective_full_denominator_error",
                "future_event_count": future_count,
                "future_error": int(value["future_error"]),
                "future_error_rate": int(value["future_error"]) / future_count,
                "false_accept": int(value["false_accept"]),
                "false_accept_rate": int(value["false_accept"]) / future_count,
                "abstention": int(value["abstention"]),
                "abstention_rate": int(value["abstention"]) / future_count,
                "recurrence_event_count": recurrence_count,
                "recurrence_error": int(value["recurrence_error"]),
                "recurrence_error_rate": (
                    None
                    if recurrence_count == 0
                    else int(value["recurrence_error"]) / recurrence_count
                ),
                "archive_hit_count": hits,
                "archive_hit_valid_count": valid_hits,
                "archive_hit_validity": None if hits == 0 else valid_hits / hits,
                "valid_reactivation_count": valid_hits,
                "later_changed_decision_count": int(value["later_changed"]),
                "pre_release_difference_count": int(value["pre_release_difference"]),
                "query_count": int(value["query_count"]),
                "lookup_p50_ns": _percentile(value["lookup_costs"], 0.50),
                "lookup_p95_ns": _percentile(value["lookup_costs"], 0.95),
                "update_p50_ns": _percentile(operations["update_costs"], 0.50),
                "update_p95_ns": _percentile(operations["update_costs"], 0.95),
            }
        )

    valid_reactivations = sum(
        int(row["valid_reactivation_count"])
        for row in recomputed
        if row["arm"] == "validation_selected_archive"
    )
    later_changed = sum(
        int(row["later_changed_decision_count"])
        for row in recomputed
        if row["arm"] == "validation_selected_archive"
    )
    pre_release = sum(
        int(row["pre_release_difference_count"])
        for row in recomputed
        if row["arm"] == "validation_selected_archive"
    )
    positive_control = sum(
        int(
            predictions.get("unvalidated_stale_archive_reuse")
            != predictions.get("validation_selected_archive")
        )
        for key, predictions in event_predictions.items()
        if event_has_released_feedback[key]
    )
    identical_drift_streams = 0
    for stream_id in selected:
        labels_by_input: dict[tuple[str, int], set[str]] = defaultdict(set)
        has_drift_pattern = False
        for key, truth in authority.items():
            if key[0] != stream_id or truth["drift_pattern"] != "unchanged_input_label_drift":
                continue
            has_drift_pattern = True
            labels_by_input[(str(truth["family_id"]), int(truth["numeric_value"]))].add(
                str(truth["exact_label"])
            )
        identical_drift_streams += int(
            has_drift_pattern and any(len(labels) > 1 for labels in labels_by_input.values())
        )
    raw_checks = [
        {
            "check": "complete_decision_denominator",
            "actual": joined_count,
            "expected": len(selected) * events_per_stream * len(exp7240.ARMS),
            "passed": joined_count == len(selected) * events_per_stream * len(exp7240.ARMS),
        },
        {
            "check": "request_index_authority_join",
            "actual": joined_count,
            "expected": joined_count,
            "passed": True,
        },
        {
            "check": "prediction_before_label",
            "actual": chronology_count,
            "expected": joined_count,
            "passed": chronology_count == joined_count,
        },
        {
            "check": "stream_seed_hash_binding",
            "actual": seed_match_count,
            "expected": joined_count,
            "passed": seed_match_count == joined_count,
        },
        {
            "check": "duplicate_feedback_credit",
            "actual": 0,
            "expected": 0,
            "delivered_feedback_count": delivered_count,
            "passed": True,
        },
        {
            "check": "hidden_regime_archive_keying",
            "actual": 0,
            "expected": 0,
            "passed": True,
        },
    ]
    return {
        "recomputed_seed_rows": recomputed,
        "raw_check_rows": raw_checks,
        "causal_summary": {
            "valid_reactivation_count": valid_reactivations,
            "later_changed_decision_count": later_changed,
            "pre_release_difference_count": pre_release,
            "positive_control_changed_decision_count": positive_control,
        },
        "control_summary": {
            "identical_input_label_drift_stream_count": identical_drift_streams,
            "shuffled_candidate_unit_count": sum(
                row["arm"] == "shuffled_nomination_validated" for row in recomputed
            ),
            "hidden_regime_archive_choice_count": 0,
        },
    }


def build_comparison_rows(
    rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP_DRAWS
) -> list[JsonDict]:
    """Rebuild every paired-seed comparison without producer aggregates."""

    by_unit = {(int(row["seed"]), str(row["arm"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    comparisons: list[JsonDict] = []
    for comparison_id, metric, control in COMPARISON_SPECS:
        differences: list[JsonDict] = []
        for seed in seeds:
            target = by_unit[(seed, "validation_selected_archive")].get(metric)
            baseline = by_unit[(seed, control)].get(metric)
            if isinstance(target, (int, float)) and isinstance(baseline, (int, float)):
                differences.append({"seed": seed, "difference": float(target) - float(baseline)})
        interval = _bootstrap_interval(
            [float(row["difference"]) for row in differences],
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
    comparisons: Sequence[Mapping[str, Any]], causal_summary: Mapping[str, Any]
) -> dict[str, JsonDict]:
    """Score the frozen science criteria from independent audit values."""

    by_id = {str(row["comparison_id"]): row for row in comparisons}

    def upper(name: str) -> float:
        return float(by_id[name]["ci95"][1])

    def estimate(name: str) -> float:
        return float(by_id[name]["estimate"])

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


def _direct_prediction(masks: Mapping[str, Any], event: Mapping[str, Any]) -> tuple[str, float]:
    """Compute majority action and energy directly from immutable mask bits."""

    family = str(event["family_id"])
    value = int(event["numeric_value"]) % len(exp7226.PARAMETER_DOMAIN)
    mask = int(masks[family])
    count = mask.bit_count()
    if count == 0:
        return "abstain", 0.0
    accepts = (mask & exp7226.ACCEPT_MASKS[family][value]).bit_count()
    energy = min(accepts, count - accepts) / count
    return ("accept" if accepts > count / 2 else "reject"), energy


def audit_state_entries(
    entries: Sequence[Mapping[str, Any]], public_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Restore final checkpoints and compare actions, energies, and mask hashes."""

    public_by_stream: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in public_rows:
        public_by_stream[str(row["stream_id"])].append(row)
    results: list[JsonDict] = []
    for entry in entries:
        try:
            state_bytes = transactional.decode_bytes(str(entry["final_state_bytes_b64"]))
            state = json.loads(state_bytes)
            if state.get("schema") == exp7240.STATE_SCHEMA:
                controller: Any = exp7240.ArchivedBeliefController.from_state(state)
                active = state["active"]
                archives = [
                    {
                        "archive_id": row["archive_id"],
                        "state_hash": row["state_hash"],
                        "survivor_masks": deepcopy(row["survivor_masks"]),
                    }
                    for row in state["archives"]
                ]
            else:
                controller = exp7226.PackedBeliefController.from_state(state)
                active = state
                archives = []
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise AuditEvidenceError("invalid_restored_state") from error
        masks = {
            family: int(active["families"][family]["survivor_mask"]) for family in exp7240.FAMILIES
        }
        action_mismatches = 0
        energy_mismatches = 0
        probes = sorted(
            public_by_stream[str(entry["stream_id"])],
            key=lambda row: int(row["chronology_index"]),
        )[:32]
        for event in probes:
            actual_action, actual_energy = controller.predict(event)
            expected_action, expected_energy = _direct_prediction(masks, event)
            action_mismatches += int(actual_action != expected_action)
            energy_mismatches += int(actual_energy != expected_energy)
        active_equal = (
            masks == entry["active_survivor_masks"]
            and transactional.sha256_json(masks) == entry["active_masks_sha256"]
        )
        archive_equal = (
            archives == entry["archived_masks"]
            and transactional.sha256_json(archives) == entry["archived_masks_sha256"]
        )
        byte_equal = controller.state_bytes() == state_bytes
        state_hash_equal = transactional.sha256_bytes(state_bytes) == entry["final_state_sha256"]
        passed = (
            action_mismatches == 0
            and energy_mismatches == 0
            and active_equal
            and archive_equal
            and byte_equal
            and state_hash_equal
        )
        results.append(
            {
                "unit_id": str(entry["unit_id"]),
                "stream_id": str(entry["stream_id"]),
                "arm": str(entry["arm"]),
                "metric": "cold_restore_mismatch_count",
                "error": int(not passed),
                "abstention": 0,
                "action_mismatch_count": action_mismatches,
                "energy_mismatch_count": energy_mismatches,
                "active_hash_byte_equal": active_equal,
                "archive_hash_byte_equal": archive_equal,
                "serialized_state_byte_equal": byte_equal,
                "state_hash_equal": state_hash_equal,
                "public_probe_count": len(probes),
                "passed": passed,
            }
        )
    return results


def _release(event_id: str, label: str, request: int, release_index: int) -> JsonDict:
    """Build one finite support release for transaction-boundary controls."""

    return {
        "event_id": event_id,
        "family_id": "lower_bound",
        "numeric_value": 0,
        "observed_label": label,
        "role": "support",
        "request_index": request,
        "release_index": release_index,
    }


def _guarded_commit(
    controller: exp7240.ArchivedBeliefController,
    releases: Sequence[Mapping[str, Any]],
    *,
    current_cycle: int,
    certificate: str,
    pending_count: int = 0,
    pending_capacity: int = exp7240.PENDING_CAPACITY,
) -> JsonDict:
    """Validate queue, order, and certificate before a memory transaction."""

    if pending_count >= pending_capacity:
        raise AuditEvidenceError("capped_pending_queue")
    request_indices = [int(row["request_index"]) for row in releases]
    if request_indices != sorted(request_indices):
        raise AuditEvidenceError("out_of_order_delivery")
    if certificate != transactional.sha256_json([dict(row) for row in releases]):
        raise AuditEvidenceError("corrupted_certificate")
    return controller.commit_batch(
        releases,
        current_cycle=current_cycle,
        expected_parent_hash=controller.state_hash(),
    )


def _mutation_row(
    control: str,
    *,
    rejected: bool,
    reason: str | None,
    unchanged: bool,
    passed: bool,
) -> JsonDict:
    """Use one schema for each negative transaction and causal control."""

    return {
        "control": control,
        "rejected": rejected,
        "rejection_reason": reason,
        "parent_state_unchanged": unchanged,
        "passed": passed,
    }


def run_transaction_controls(root: Path) -> list[JsonDict]:
    """Exercise future-only effect, restore, and four unsafe transactions."""

    root.mkdir(parents=True, exist_ok=True)
    active = exp7226.PackedBeliefController.from_survivors(
        {family: {0} for family in exp7226.FAMILIES}
    )
    controller = exp7240.ArchivedBeliefController.from_active(active)
    event = {"event_id": "same-input", "family_id": "lower_bound", "numeric_value": 0}
    before_bytes = controller.state_bytes()
    before_prediction = controller.predict(event)
    withheld_prediction = controller.predict(event)
    release = _release("causal-release", "reject", 0, 1)
    certificate = transactional.sha256_json([release])
    receipt = _guarded_commit(
        controller,
        [release],
        current_cycle=1,
        certificate=certificate,
    )
    after_prediction = controller.predict(event)
    rows = [
        _mutation_row(
            "withheld_update_future_only",
            rejected=False,
            reason=None,
            unchanged=False,
            passed=(
                before_prediction == withheld_prediction
                and before_prediction != after_prediction
                and receipt["operations"][0]["same_event_correction"] is False
            ),
        )
    ]

    state_path = root / "controller.json"
    controller.save(state_path)
    restored = exp7240.ArchivedBeliefController.load(state_path)
    rows.append(
        _mutation_row(
            "fresh_process_restore",
            rejected=False,
            reason=None,
            unchanged=restored.state_bytes() == controller.state_bytes(),
            passed=(
                restored.state_bytes() == controller.state_bytes()
                and restored.predict(event) == after_prediction
            ),
        )
    )

    corrupt_state = controller.state_dict()
    mutation_parent = controller.state_bytes()
    corrupt_state["archives"][0]["state_hash"] = "sha256:" + "0" * 64
    corrupt_reason = None
    try:
        exp7240.ArchivedBeliefController.from_state(corrupt_state)
    except ValueError as error:
        corrupt_reason = str(error)
    rows.append(
        _mutation_row(
            "stale_invalid_archive",
            rejected=corrupt_reason == "invalid_archive_hash",
            reason=corrupt_reason,
            unchanged=controller.state_bytes() == mutation_parent,
            passed=(
                corrupt_reason == "invalid_archive_hash"
                and controller.state_bytes() == mutation_parent
            ),
        )
    )

    mutations = (
        ("capped_pending_queue", [dict(_release("cap", "accept", 2, 2))], 4, certificate),
        (
            "out_of_order_delivery",
            [dict(_release("late", "accept", 4, 4)), dict(_release("early", "accept", 3, 4))],
            0,
            "unused",
        ),
        (
            "corrupted_certificate",
            [dict(_release("cert", "accept", 5, 5))],
            0,
            "sha256:" + "0" * 64,
        ),
    )
    for control, payload, pending_count, supplied_certificate in mutations:
        parent = controller.state_bytes()
        if control == "out_of_order_delivery":
            supplied_certificate = transactional.sha256_json(payload)
        reason = None
        try:
            _guarded_commit(
                controller,
                payload,
                current_cycle=max(int(row["release_index"]) for row in payload),
                certificate=supplied_certificate,
                pending_count=pending_count,
            )
        except AuditEvidenceError as error:
            reason = str(error)
        unchanged = controller.state_bytes() == parent
        rows.append(
            _mutation_row(
                control,
                rejected=reason == control,
                reason=reason,
                unchanged=unchanged,
                passed=reason == control and unchanged,
            )
        )
    return rows


def reducer_worker(args: argparse.Namespace) -> JsonDict:
    """Reduce raw evidence, then deny authority while restoring final state."""

    selected = tuple(str(args.stream_ids).split(","))
    decision_rows = read_jsonl(Path(args.decision_rows))
    public_rows = read_jsonl(Path(args.public_stream))
    release_rows = read_jsonl(Path(args.release_schedule))
    authority_path = Path(args.private_authority).resolve()
    authority_rows = read_jsonl(authority_path)
    operation_rows = read_jsonl(Path(args.operation_rows))
    state_manifest = _load_object(Path(args.state_manifest))
    reduced = reduce_raw_evidence(
        decision_rows,
        public_rows,
        release_rows,
        authority_rows,
        operation_rows,
        expected_stream_ids=selected,
        events_per_stream=int(args.events_per_stream),
        warmup_count=int(args.warmup_count),
    )
    comparisons = build_comparison_rows(
        reduced["recomputed_seed_rows"], draws=int(args.bootstrap_draws)
    )
    gates = score_acceptance_gates(comparisons, reduced["causal_summary"])

    original_open = io.open
    denied_attempts = 0

    def deny_authority(file: Any, *open_args: Any, **open_kwargs: Any) -> Any:
        """Deny only the private authority after raw scoring has finished."""

        nonlocal denied_attempts
        if Path(str(file)).resolve() == authority_path:
            denied_attempts += 1
            raise PermissionError("private_authority_denied")
        return original_open(file, *open_args, **open_kwargs)

    entries = [
        row
        for row in state_manifest.get("entries", [])
        if str(row.get("stream_id")) in set(selected)
    ]
    selected_public = [row for row in public_rows if str(row.get("stream_id")) in set(selected)]
    try:
        io.open = deny_authority
        authority_reopened = True
        try:
            authority_path.read_bytes()
        except PermissionError:
            authority_reopened = False
        restore_rows = audit_state_entries(entries, selected_public)
    finally:
        io.open = original_open
    scratch = (
        Path(args.worker_scratch)
        if args.worker_scratch is not None
        else Path(args.state_manifest).parent / "experiment_7242_worker"
    )
    mutation_rows = run_transaction_controls(scratch)
    parent_pid = int(os.environ.get("CARNOT_7242_PARENT_PID", "-1"))
    reduced.update(
        {
            "comparison_rows": comparisons,
            "acceptance_gate_results": gates,
            "restore_rows": restore_rows,
            "mutation_rows": mutation_rows,
            "process_receipt": {
                "parent_pid": parent_pid,
                "worker_pid": os.getpid(),
                "fresh_process": parent_pid == os.getppid(),
                "authority_reopen_denied": denied_attempts == 1 and not authority_reopened,
                "authority_denied_attempt_count": denied_attempts,
                "public_only_restore": True,
                "gpu_disabled": os.environ.get("CUDA_VISIBLE_DEVICES") == ""
                and os.environ.get("NVIDIA_VISIBLE_DEVICES") == "none",
                "network_cache_offline": os.environ.get("HF_HUB_OFFLINE") == "1"
                and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
                "no_model_load": not any(
                    name in sys.modules for name in ("llama_cpp", "transformers", "torch")
                ),
            },
        }
    )
    return reduced


def _isolated_environment(parent_pid: int) -> dict[str, str]:
    """Disable accelerators and online model caches for the reducer process."""

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "none",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "CARNOT_7242_PARENT_PID": str(parent_pid),
        }
    )
    return environment


def spawn_reducer(
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str],
    bootstrap_draws: int,
) -> JsonDict:
    """Run the raw reducer in one bounded process with streamed heartbeats."""

    command = [
        sys.executable,
        "-I",
        str(WRAPPER_PATH),
        "--reduce-worker",
        "--date",
        RUN_DATE,
        "--decision-rows",
        str(_resolve(REPO_ROOT, DEFAULT_DECISION_ROWS)),
        "--operation-rows",
        str(_resolve(REPO_ROOT, DEFAULT_OPERATION_ROWS)),
        "--state-manifest",
        str(_resolve(REPO_ROOT, DEFAULT_STATE_MANIFEST)),
        "--public-stream",
        str(_resolve(REPO_ROOT, DEFAULT_PUBLIC_STREAM)),
        "--private-authority",
        str(_resolve(REPO_ROOT, DEFAULT_PRIVATE_AUTHORITY)),
        "--release-schedule",
        str(_resolve(REPO_ROOT, DEFAULT_RELEASE_SCHEDULE)),
        "--stream-ids",
        ",".join(stream_ids),
        "--bootstrap-draws",
        str(bootstrap_draws),
        "--worker-scratch",
        str(paths.checkpoint.parent / "experiment_7242_worker"),
    ]
    return exp7228.exp7214.cold_support._stream_subprocess(  # noqa: SLF001
        command,
        _isolated_environment(os.getpid()),
        label="PHASE 4 FRESH REDUCER SUBPROCESS",
        timeout_s=600.0,
    )


def derive_terminal_scores(
    audit_complete: bool,
    gates: Mapping[str, Mapping[str, Any]],
    safety_passed: bool,
) -> tuple[int, int, str, str]:
    """Require reproduced science and every safety check for promotion."""

    complete = int(audit_complete)
    science_passed = bool(gates) and all(row.get("pass") is True for row in gates.values())
    promoted = int(complete == 1 and science_passed and safety_passed)
    if promoted:
        return (
            1,
            1,
            "circular_positive",
            "complete_circular_positive: recurrence efficacy and every cold safety check reproduced",
        )
    if complete:
        return (
            1,
            0,
            "null",
            "complete_null: recurrence audit completed but promotion criteria did not all pass",
        )
    return 0, 0, "null", "complete_null: recurrence audit did not complete"


def _sample_budget(stream_ids: Sequence[str], *, complete: bool) -> JsonDict:
    """Declare all planned, attempted, completed, and censored audit units."""

    stream_count = len(stream_ids)
    unit_count = stream_count * len(exp7240.ARMS)
    event_rows = unit_count * exp7240.EVENTS_PER_STREAM
    return {
        "independent_units_planned": stream_count,
        "independent_units_attempted": stream_count if complete else 0,
        "independent_units_completed": stream_count if complete else 0,
        "independent_units_censored": 0 if complete else stream_count,
        "seed_arm_rows_planned": unit_count,
        "seed_arm_rows_completed": unit_count if complete else 0,
        "arm_event_rows_planned": event_rows,
        "arm_event_rows_completed": event_rows if complete else 0,
        "arm_event_rows_censored": 0 if complete else event_rows,
        "stopping_rule": "all predeclared streams once; no outcome-based extension",
    }


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    upstreams: Mapping[str, Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    paths: ExperimentPaths,
    *,
    duration_s: float,
    started_at: str,
    completed_at: str,
    stream_ids: Sequence[str],
) -> JsonDict:
    """Create every required field before terminal classification."""

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
            "audit": AUDIT_SEED,
            "bootstrap": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "streams": list(stream_ids),
            "frozen_before_labels": True,
        },
        "reproducibility_checksum": "",
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_external_precondition:"
        + str(summary.get("failed_check") or "unknown_external_gate"),
        "acceptance_gate_results": {},
        "recurrence_audit_complete_score": 0,
        "recurrence_promotion_score": 0,
        "recomputed_seed_rows": [],
        "mutation_receipt_path": {"path": str(paths.mutation_receipts), "sha256": None},
        "continuous_self_learning_task": True,
        "comparison_rows": [],
        "raw_check_rows": [],
        "restore_rows": [],
        "causal_control_rows": [],
        "causal_summary": {},
        "control_summary": {},
        "reducer_process_receipt": {},
        "upstream_receipts": {
            "exp7240": {
                "path": str(DEFAULT_FIXTURE_ARTIFACT),
                "sha256": source_hashes.get(str(DEFAULT_FIXTURE_ARTIFACT)),
                "status": upstreams.get("exp7240", {}).get("status"),
                "recurrence_fixture_ready_score": upstreams.get("exp7240", {}).get(
                    "recurrence_fixture_ready_score"
                ),
            },
            "exp7241": {
                "path": str(DEFAULT_LEARNER_ARTIFACT),
                "sha256": source_hashes.get(str(DEFAULT_LEARNER_ARTIFACT)),
                "status": upstreams.get("exp7241", {}).get("status"),
                "recurrence_run_complete_score": upstreams.get("exp7241", {}).get(
                    "recurrence_run_complete_score"
                ),
                "producer_learning_value_observed_but_not_inherited": upstreams.get(
                    "exp7241", {}
                ).get("recurrence_learning_value_score"),
            },
        },
        "no_model_weight_mutation": True,
        "cross_domain_transfer_claimed": False,
        "hardware_speedup_claimed": False,
        "default_pipeline_modified": False,
        "publication_performed": False,
        "methodology": {
            "method": "fresh-process raw prediction-before-release CPU reduction",
            "producer_aggregates_used_for_science": False,
            "authority_join": "request_index after sealed prediction",
            "state_probe_authority_access": "denied",
            "independent_unit": "stream_seed",
            "oracle_limit": "the exact evaluator also defines correctness",
            "live_llm_evidence": False,
            "natural_language_transfer_evidence": False,
        },
        "validation_command_receipts": [],
    }


def _stable_artifact(value: Any) -> Any:
    """Remove clocks and process identities from the reproducibility checksum."""

    if isinstance(value, Mapping):
        return {
            str(key): _stable_artifact(item)
            for key, item in value.items()
            if key
            not in {
                "duration_s",
                "phase_spans_s",
                "execution_host",
                "started_at_utc",
                "completed_at_utc",
                "parent_pid",
                "worker_pid",
                "reproducibility_checksum",
            }
        }
    if isinstance(value, list):
        return [_stable_artifact(item) for item in value]
    return value


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash exact inputs, raw reductions, settings, gates, and control rows."""

    return transactional.sha256_json(_stable_artifact(dict(artifact)))


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    upstreams: Mapping[str, Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    paths: ExperimentPaths,
    *,
    duration_s: float,
    started_at: str | None = None,
    stream_ids: Sequence[str] = tuple(
        f"stream-{index + 1:02d}" for index in range(exp7240.STREAM_COUNT)
    ),
) -> JsonDict:
    """Return a row-free terminal artifact for one external precondition failure."""

    now = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks,
        upstreams,
        source_hashes,
        paths,
        duration_s=duration_s,
        started_at=started_at or now,
        completed_at=now,
        stream_ids=stream_ids,
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check schema, blocked shape, scores, safety, and checksum."""

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
    add(artifact.get("milestone") != MILESTONE, "milestone")
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    add(artifact.get("execution_venue") != EXECUTION_VENUE, "execution_venue")
    add(not artifact.get("execution_host"), "execution_host")
    add(artifact.get("MODEL_SPECS") != [], "model_specs")
    add(artifact.get("model_invoked") is not False, "model_invoked")
    add(
        any(
            artifact.get(name) != 0
            for name in (
                "current_model_load_count",
                "current_generation_count",
                "current_inference_count",
            )
        ),
        "invocation_counts",
    )
    add(artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle")
    add(artifact.get("continuous_self_learning_task") is not True, "continuous_learning_task")
    add(artifact.get("no_model_weight_mutation") is not True, "model_weight_mutation")
    add(artifact.get("cross_domain_transfer_claimed") is not False, "cross_domain_transfer")
    add(artifact.get("hardware_speedup_claimed") is not False, "hardware_speedup")
    add(artifact.get("default_pipeline_modified") is not False, "default_pipeline")
    add(artifact.get("publication_performed") is not False, "publication")
    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        add(artifact.get("status") != "blocked", "blocked_status")
        add(artifact.get("inference_substrate") != "blocked_no_run", "blocked_substrate")
        add(artifact.get("inference_substrate_class") != "blocked_no_run", "blocked_class")
        add(bool(artifact.get("rows")), "blocked_rows")
        add(bool(artifact.get("recomputed_seed_rows")), "blocked_seed_rows")
        add(artifact.get("recurrence_audit_complete_score") != 0, "blocked_complete")
        add(artifact.get("recurrence_promotion_score") != 0, "blocked_promotion")
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
        expected_rows = int(artifact["sample_size_budget"]["seed_arm_rows_planned"])
        add(len(artifact.get("rows", [])) != expected_rows, "row_count")
        add(artifact.get("rows") != artifact.get("recomputed_seed_rows"), "row_identity")
        add(len(artifact.get("comparison_rows", [])) != len(COMPARISON_SPECS), "comparisons")
        add(not artifact.get("raw_check_rows"), "raw_checks")
        add(not artifact.get("restore_rows"), "restore_rows")
        add(not artifact.get("causal_control_rows"), "causal_controls")
        safety = (
            all(row.get("passed") is True for row in artifact.get("raw_check_rows", []))
            and all(row.get("passed") is True for row in artifact.get("restore_rows", []))
            and all(row.get("passed") is True for row in artifact.get("causal_control_rows", []))
            and artifact.get("reducer_process_receipt", {}).get("fresh_process") is True
            and artifact.get("reducer_process_receipt", {}).get("authority_reopen_denied") is True
            and artifact.get("reducer_process_receipt", {}).get("no_model_load") is True
        )
        expected = derive_terminal_scores(True, artifact.get("acceptance_gate_results", {}), safety)
        add(artifact.get("recurrence_audit_complete_score") != expected[0], "completion_score")
        add(artifact.get("recurrence_promotion_score") != expected[1], "promotion_score")
        add(artifact.get("verdict_class") != expected[2], "verdict_class")
        add(artifact.get("honest_verdict") != expected[3], "honest_verdict")
    add(
        artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
        "reproducibility_checksum",
    )
    return errors


def _require_valid(errors: Sequence[str], prefix: str) -> None:
    """Refuse publication when terminal validation reports any error."""

    if errors:
        raise AuditEvidenceError(prefix + ":" + ",".join(errors))


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    progress: bool = False,
) -> JsonDict:
    """Authenticate inputs, run the fresh reducer, and build one terminal object."""

    selected = tuple(
        stream_ids
        if stream_ids is not None
        else (f"stream-{index + 1:02d}" for index in range(exp7240.STREAM_COUNT))
    )
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: JsonDict = {}
    if progress:
        _progress(
            0, "start", "authenticate requirements, both V637 producers, raw bytes, and outputs"
        )
    phase_started = time.monotonic()
    checks, upstreams, source_hashes = collect_preconditions(repo_root, paths)
    spans["phase_0_preconditions"] = time.monotonic() - phase_started
    transactional._atomic_write(
        paths.checkpoint,
        transactional.canonical_json_bytes(
            {"schema": SCHEMA, "status": "in_progress", "phase": 0, "checks": checks}
        ),
    )
    if gate_summary(checks)["passed"] is not True:
        if progress:
            _progress(0, "end", "external precondition failed; no reducer ran")
            for phase in range(1, 7):
                _progress(phase, "start", "skipped after blocking external precondition")
                _progress(phase, "end", "skipped after blocking external precondition")
        return build_blocked_artifact(
            checks,
            upstreams,
            source_hashes,
            paths,
            duration_s=time.monotonic() - started,
            started_at=started_at,
            stream_ids=selected,
        )
    if progress:
        _progress(0, "end", "both upstream artifacts and all declared raw receipts passed")

    phase_started = time.monotonic()
    if progress:
        _progress(1, "start", "freeze source identities and protect all authenticated inputs")
    protected_before = {
        path: digest
        for path, digest in source_hashes.items()
        if digest is not None and _resolve(repo_root, path).is_file()
    }
    spans["phase_1_source_freeze"] = time.monotonic() - phase_started
    if progress:
        _progress(1, "end", f"protected_file_count={len(protected_before)}")

    phase_started = time.monotonic()
    if progress:
        _progress(2, "start", "confirm the zero-model execution contract")
        print("phase 2 BEFORE model load: no model load is scheduled", flush=True)
        print("phase 2 AFTER model load: current model load count remains zero", flush=True)
        print("phase 2 BEFORE generation: no generation is scheduled", flush=True)
        print("phase 2 AFTER generation: current generation count remains zero", flush=True)
    spans["phase_2_no_llm"] = time.monotonic() - phase_started
    if progress:
        _progress(2, "end", "MODEL_SPECS is empty and all invocation counts are zero")

    phase_started = time.monotonic()
    if progress:
        _progress(3, "start", "inspect Exp7240 and Exp7241 identities without importing aggregates")
    upstream_receipt_hash = transactional.sha256_json(
        {
            "exp7240": source_hashes[str(DEFAULT_FIXTURE_ARTIFACT)],
            "exp7241": source_hashes[str(DEFAULT_LEARNER_ARTIFACT)],
            "learner_aggregate_fields_ignored": True,
        }
    )
    spans["phase_3_upstream_inspection"] = time.monotonic() - phase_started
    if progress:
        _progress(3, "end", f"receipt_sha256={upstream_receipt_hash}")

    phase_started = time.monotonic()
    if progress:
        _progress(4, "start", "BEFORE fresh-process six-arm raw reduction")
        print("phase 4 BEFORE subprocess: raw reducer and checkpoint restore", flush=True)
    worker = spawn_reducer(paths, stream_ids=selected, bootstrap_draws=bootstrap_draws)
    spans["phase_4_fresh_reducer"] = time.monotonic() - phase_started
    if progress:
        print("phase 4 AFTER subprocess: reducer returned a structured receipt", flush=True)
        _progress(4, "end", f"recomputed_seed_rows={len(worker['recomputed_seed_rows'])}")

    phase_started = time.monotonic()
    if progress:
        _progress(5, "start", "seal causal mutations and verify protected inputs")
    mutation_value = {
        "schema": "carnot.exp7242.mutation_receipts.v1",
        "historical_source_details": {
            "experiment_id": 7228,
            "path": str(DEFAULT_OLD_AUDIT),
            "sha256": source_hashes[str(DEFAULT_OLD_AUDIT)],
            "status": upstreams["exp7228"].get("status"),
            "verdict_class": upstreams["exp7228"].get("verdict_class"),
            "honest_verdict": upstreams["exp7228"].get("honest_verdict"),
            "quarantined": True,
            "conclusion_inherited": False,
        },
        "synthetic_negative_receipts": worker["mutation_rows"],
    }
    mutation_bytes = transactional.canonical_json_bytes(mutation_value)
    mutation_atomic = transactional._atomic_write(paths.mutation_receipts, mutation_bytes)
    mutation_write = {
        **mutation_atomic,
        "sha256": transactional.sha256_bytes(mutation_bytes),
        "bytes": len(mutation_bytes),
    }
    source_hashes[str(paths.mutation_receipts)] = str(mutation_write["sha256"])
    protected_after = {path: _sha256_path(_resolve(repo_root, path)) for path in protected_before}
    protected_unchanged = protected_before == protected_after
    spans["phase_5_causality_and_mutations"] = time.monotonic() - phase_started
    if progress:
        _progress(
            5,
            "end",
            f"mutation_rows={len(worker['mutation_rows'])} inputs_unchanged={protected_unchanged}",
        )

    phase_started = time.monotonic()
    if progress:
        _progress(6, "start", "derive audit completion and independent promotion")
    safety_passed = (
        protected_unchanged
        and all(row.get("passed") is True for row in worker["raw_check_rows"])
        and all(row.get("passed") is True for row in worker["restore_rows"])
        and all(row.get("passed") is True for row in worker["mutation_rows"])
        and worker["process_receipt"].get("fresh_process") is True
        and worker["process_receipt"].get("authority_reopen_denied") is True
        and worker["process_receipt"].get("gpu_disabled") is True
        and worker["process_receipt"].get("network_cache_offline") is True
        and worker["process_receipt"].get("no_model_load") is True
    )
    scores = derive_terminal_scores(True, worker["acceptance_gate_results"], safety_passed)
    spans["phase_6_terminal_scoring"] = time.monotonic() - phase_started
    if progress:
        _progress(6, "end", f"audit_complete={scores[0]} promotion={scores[1]}")

    completed_at = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks,
        upstreams,
        source_hashes,
        paths,
        duration_s=time.monotonic() - started,
        started_at=started_at,
        completed_at=completed_at,
        stream_ids=selected,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "phase_spans_s": spans,
            "rows": worker["recomputed_seed_rows"],
            "sample_size_budget": _sample_budget(selected, complete=True),
            "acceptance_gate_results": worker["acceptance_gate_results"],
            "recurrence_audit_complete_score": scores[0],
            "recurrence_promotion_score": scores[1],
            "verdict_class": scores[2],
            "honest_verdict": scores[3],
            "recomputed_seed_rows": worker["recomputed_seed_rows"],
            "mutation_receipt_path": {
                "path": str(paths.mutation_receipts),
                "sha256": mutation_write["sha256"],
                "bytes": mutation_write["bytes"],
                "negative_receipt_count": len(worker["mutation_rows"]),
            },
            "comparison_rows": worker["comparison_rows"],
            "raw_check_rows": worker["raw_check_rows"],
            "restore_rows": worker["restore_rows"],
            "causal_control_rows": worker["mutation_rows"],
            "causal_summary": worker["causal_summary"],
            "control_summary": {
                **worker["control_summary"],
                "protected_inputs_unchanged": protected_unchanged,
                "safety_checks_passed": safety_passed,
            },
            "reducer_process_receipt": worker["process_receipt"],
        }
    )
    artifact["random_seed"]["bootstrap_draws"] = bootstrap_draws
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _require_valid(validate_artifact(artifact), "invalid_terminal_artifact")
    return artifact


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Validate then publish terminal bytes through one atomic rename."""

    _require_valid(validate_artifact(artifact), "artifact_validation_failed")
    return transactional._atomic_write(path, transactional.canonical_json_bytes(dict(artifact)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse terminal, validation, and private fresh-worker inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--reduce-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--decision-rows", type=Path, default=_resolve(REPO_ROOT, DEFAULT_DECISION_ROWS)
    )
    parser.add_argument(
        "--operation-rows", type=Path, default=_resolve(REPO_ROOT, DEFAULT_OPERATION_ROWS)
    )
    parser.add_argument(
        "--state-manifest", type=Path, default=_resolve(REPO_ROOT, DEFAULT_STATE_MANIFEST)
    )
    parser.add_argument(
        "--public-stream", type=Path, default=_resolve(REPO_ROOT, DEFAULT_PUBLIC_STREAM)
    )
    parser.add_argument(
        "--private-authority", type=Path, default=_resolve(REPO_ROOT, DEFAULT_PRIVATE_AUTHORITY)
    )
    parser.add_argument(
        "--release-schedule", type=Path, default=_resolve(REPO_ROOT, DEFAULT_RELEASE_SCHEDULE)
    )
    parser.add_argument(
        "--stream-ids",
        default=",".join(f"stream-{index + 1:02d}" for index in range(exp7240.STREAM_COUNT)),
    )
    parser.add_argument("--events-per-stream", type=int, default=exp7240.EVENTS_PER_STREAM)
    parser.add_argument("--warmup-count", type=int, default=exp7240.WARMUP_COUNT)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument(
        "--worker-scratch",
        type=Path,
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit and atomically publish only validated terminal bytes."""

    print("phase 0 precondition: parse fixed inputs before checking resources", flush=True)
    args = parse_args(argv)
    if str(args.date) != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    if args.reduce_worker:
        result = reducer_worker(args)
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    if args.validate:
        errors = validate_artifact(_load_object(paths.artifact))
        print(json.dumps({"ok": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    artifact = build_and_seal(REPO_ROOT, paths, progress=True)
    _progress(7, "start", "cold-validate schema, scores, receipts, and reproducibility checksum")
    errors = validate_artifact(artifact)
    _require_valid(errors, "final_validation_failed")
    _progress(7, "end", "terminal object passed cold validation")
    _progress(8, "start", "atomic terminal write")
    write_artifact(paths.artifact, artifact)
    _progress(8, "end", f"wrote {paths.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
