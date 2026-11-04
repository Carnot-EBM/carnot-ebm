"""Measure witnessed predicate refinement with charged feedback.

The learner reads only public events and released labels. Exact authority stays
outside prediction and query selection. Durable committed predicates and the
shipped version-space predictor receive matched witness information.

Spec refs: REQ-CL-7213 and SCENARIO-CL-7213-*.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import socket
import statistics
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_7199_v634_bounded_acquisition as exp7199
from carnot import experiment_7212_v635_refinement_fixture as exp7212
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7213
SCHEMA = "carnot.exp7213.v635_refinement_learning.v1"
MILESTONE = "2026.09.635"
RUN_DATE = "20260911"
BOOTSTRAP_SEED = 7_213_001
BOOTSTRAP_DRAWS = 10_000
STREAM_SEEDS = tuple(exp7212.STREAM_SEEDS)
EVENTS_PER_SEED = exp7212.EVENTS_PER_SEED
FAMILIES = tuple(exp7212.FAMILIES)
PARAMETER_DOMAIN = tuple(exp7212.PARAMETER_DOMAIN)
WARMUP_COUNT = exp7212.WARMUP_COUNT
QUERY_BUDGET = exp7212.QUERY_BUDGET
FITTING_BUDGET = exp7212.FITTING_BUDGET
VALIDATION_BUDGET = exp7212.VALIDATION_BUDGET
VALIDATION_PER_FAMILY = exp7212.VALIDATION_PER_FAMILY
PENDING_CAPACITY = exp7212.PENDING_CAPACITY
ARMS = tuple(exp7212.ARMS)
COMMITTED_ARMS = (
    "passive_query_committed",
    "random_query_committed",
    "witness_query_committed",
)
LEARNING_ARMS = (*COMMITTED_ARMS, "witness_query_version_space")
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = (
    "CPU finite-domain bitset elimination, charged witness replay, transactional "
    "predicate commits, and direct compiled predicate dispatch; no LLM invocation"
)
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM_ARTIFACT = Path("results/experiment_7212_v635_refinement_fixture.json")
EXPECTED_UPSTREAM_SHA256 = "sha256:f818e720c53a28443418333c14f8dd4b04ab82324e6a6d91077cd81ead37a546"
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/experiment_7213_v635_refinement_learning.json")
DEFAULT_ARTIFACT_PATH = Path("results/experiment_7213_v635_refinement_learning.json")
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
    Path("python/carnot/experiment_7198_v634_feedback_capacity_stream.py"),
    Path("python/carnot/experiment_7199_v634_bounded_acquisition.py"),
    Path("python/carnot/experiment_7200_v634_acquisition_cold_audit.py"),
    Path("python/carnot/experiment_7212_v635_refinement_fixture.py"),
    Path("python/carnot/experiment_7213_v635_refinement_learning.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("scripts/experiments/experiment_7213_v635_refinement_learning.py"),
    Path("tests/python/test_experiment_7213_v635_refinement_learning.py"),
    SPEC_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "field_principles",
    "status",
    "run_date",
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
    "refinement_run_complete_score",
    "refinement_value_score",
    "continuous_self_learning_task",
    "no_model_weight_mutation",
    "acceptance_gate_learning",
    "decision_rows",
    "query_rows",
    "commit_deletion_rows",
    "checkpoint_path",
    "checkpoint_hash",
    "latency_summary",
    "future_hardware_path",
    "MODEL_SPECS",
    "model_invoked",
    "comparison_rows",
    "panel_conformance_errors",
    "upstream_receipt",
    "raw_rows_checksum",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "A versioned schema makes incompatible readers fail closed.",
    "experiment_id": "A fixed ID prevents another task from supplying this evidence.",
    "milestone": "The milestone binds the measurement to the V635 contract.",
    "field_principles": "Echo the reason for each field beside its actual evidence.",
    "status": "Write a terminal artifact only after completion or a diagnosed external block.",
    "run_date": "Use 20260911; do not substitute an upstream experiment date.",
    "preconditions_checked": "Record the actual resource, code and gate observations.",
    "inference_substrate": "Describe executed computation, not the intended workload.",
    "inference_substrate_class": "The actual operation determines its duration floor.",
    "execution_venue": "Use exactly host, kv260, gatemate or polarfire; this task uses host.",
    "execution_host": "Put the actual hostname here, never inside execution_venue.",
    "duration_s": "Measure monotonic work time; do not pad it to pass a floor.",
    "source_artifact_hashes": "Bind source code, input data and frozen contracts to the claim.",
    "rows": "Keep unit_id, arm, seed, metric, error and abstention for each comparison.",
    "sample_size_budget": "Retain planned, attempted, completed, censored and unit counts.",
    "random_seed": "Freeze stochastic choices before held-out outcomes are read.",
    "reproducibility_checksum": "Hash the inputs, code, settings and timing-free raw rows.",
    "gate_check_summary": "Every blocked verdict names the failed gate and exact values.",
    "verifier_is_oracle": "The same correctness authority remains circular.",
    "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
    "honest_verdict": "Use complete findings or blocked external absence; readiness is not value.",
    "refinement_run_complete_score": "An honest null still supplies a complete learning run.",
    "refinement_value_score": "One requires prospective benefit and causal template dependence.",
    "continuous_self_learning_task": "Queried feedback creates and revokes deployed constraints.",
    "no_model_weight_mutation": "This run changes constraint memory, not model weights.",
    "acceptance_gate_learning": "Each fixed criterion retains its interval and outcome.",
    "decision_rows": "Every prediction, authority result, and information boundary is retained.",
    "query_rows": "Fitting, validation, delay, and pending capacity are charged consistently.",
    "commit_deletion_rows": "Deleting committed predicates tests the deployed decisions directly.",
    "checkpoint_path": "Full state stays under results/checkpoints for cold replay.",
    "checkpoint_hash": "The checkpoint digest binds every reloadable final state.",
    "latency_summary": "Acquisition, validation, and deployment costs are all included.",
    "future_hardware_path": "Measured CPU work stays separate from optional FPGA work.",
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is only cited.",
    "comparison_rows": "Paired stream intervals preserve the independent sampling unit.",
    "panel_conformance_errors": "An empty list proves chronology and resource bounds passed.",
    "upstream_receipt": "Exact producer and stream hashes authenticate the frozen fixture.",
    "raw_rows_checksum": "One digest binds the complete measured row payload.",
}


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep checkpoint state separate from the terminal measurement."""

    checkpoint: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the repository paths used by the required command."""

        return cls(DEFAULT_CHECKPOINT_PATH, DEFAULT_ARTIFACT_PATH)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put test outputs under one caller-owned private directory."""

        return cls(
            root / "checkpoints" / "experiment_7213_v635_refinement_learning.json",
            root / "experiment_7213_v635_refinement_learning.json",
        )


@dataclass(frozen=True)
class FixtureViews:
    """Hold public and authority bytes in separate indexed structures."""

    fixture: JsonDict
    public_by_seed: dict[int, list[JsonDict]]
    authority_by_id: dict[str, JsonDict]
    warmup_by_seed: dict[int, list[JsonDict]]
    manifest: JsonDict


@dataclass
class PendingRecord:
    """Keep unreleased labels outside all learner-visible policy inputs."""

    event: JsonDict
    role: str
    request_index: int
    release_index: int
    observed_label: str
    exact_label: str
    poisoned: bool
    query_row: JsonDict


@dataclass
class ArmRuntime:
    """Own one independently initialized arm for one frozen stream."""

    arm: str
    seed: int
    fallback: exp7212.FrozenWarmupFallback
    controller: exp7212.CandidateRefinementController | None = None
    version_controller: exp7199.VersionSpaceController | None = None
    memories: dict[str, transactional.TransactionalConstraintMemory] = field(default_factory=dict)
    compiled_parameters: dict[str, int] = field(default_factory=dict)
    pending: list[PendingRecord] = field(default_factory=list)
    query_count: int = 0
    fitting_count: int = 0
    validation_count: int = 0
    validation_by_family: dict[str, int] = field(default_factory=dict)
    max_pending: int = 0
    max_memory_bytes: int = 0
    operation_index: int = 0
    costs: dict[str, list[int]] = field(default_factory=dict)
    operation_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class LearningPanel:
    """Retain raw decisions, feedback, interventions, and final states."""

    rows: list[JsonDict]
    decision_rows: list[JsonDict]
    query_rows: list[JsonDict]
    commit_deletion_rows: list[JsonDict]
    state_rows: list[JsonDict]
    latency_values: dict[str, list[int]]


def _resolve(repo_root: Path, path: Path | str) -> Path:
    """Resolve repository-relative evidence without changing absolute paths."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _sha256_path(path: Path) -> str | None:
    """Hash exact file bytes in bounded chunks and preserve absence as None."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def _atomic_write(path: Path, payload: bytes) -> None:
    """Publish complete bytes with one rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def unwrap_principled(value: Any) -> Any:
    """Unwrap only the exact two-key principle and value representation."""

    if (
        isinstance(value, Mapping)
        and set(value) == {"principle", "value"}
        and isinstance(value.get("principle"), str)
    ):
        return value["value"]
    return value


def gate_check(
    check: str,
    upstream: str,
    field_name: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    """Keep exact expected and observed values beside one prerequisite."""

    actual = unwrap_principled(observed)
    return {
        "check": check,
        "upstream": upstream,
        "field": field_name,
        "expected_value": expected,
        "observed_value": actual,
        "passed": actual == expected,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose the first failure without discarding the complete gate ledger."""

    failed = next((row for row in checks if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "checks": [dict(row) for row in checks],
        "failed_check": None if failed is None else failed["check"],
        "upstream": None if failed is None else failed["upstream"],
        "field": None if failed is None else failed["field"],
        "expected_value": None if failed is None else failed["expected_value"],
        "observed_value": None if failed is None else failed["observed_value"],
    }


_QUARANTINE_KEYS = (
    "artifact_quarantined",
    "upstream_quarantined",
    "quarantine_flag",
    "quarantined",
    "excluded_from_use",
    "flagged_adversarial",
)


def quarantine_state(
    upstream: Mapping[str, Any],
    exclusion_text: str,
    artifact_name: str,
    task_id: str,
) -> JsonDict:
    """Treat metadata and the independent manifest as separate quarantine gates."""

    flags = {key: unwrap_principled(upstream[key]) for key in _QUARANTINE_KEYS if key in upstream}
    matches = [marker for marker in (artifact_name, task_id) if marker in exclusion_text]
    return {
        "quarantined": any(value is True for value in flags.values()) or bool(matches),
        "declared_flags": flags,
        "exclusion_manifest_matches": matches,
    }


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating evidence early."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _task_identity(text: str) -> JsonDict:
    """Extract only the Exp7213 block from the executable roadmap."""

    match = re.search(r"(?ms)^- id: exp7213-refinement-learning\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(0)
    return {
        "id": "exp7213-refinement-learning" if block else None,
        "milestone": MILESTONE if f"milestone: {MILESTONE}" in block else None,
        "deliverable": (
            str(DEFAULT_ARTIFACT_PATH) if f"deliverable: {DEFAULT_ARTIFACT_PATH}" in block else None
        ),
    }


def _fixture_paths(fixture: Mapping[str, Any]) -> tuple[Path, ...]:
    """Return only declared fixture data and checkpoint paths."""

    keys = (
        "public_stream_path",
        "authority_sidecar_path",
        "split_feedback_manifest_path",
        "released_warmup_path",
        "controller_serialization_path",
        "checkpoint_path",
    )
    return tuple(Path(str(unwrap_principled(fixture.get(key, "")))) for key in keys)


def _load_object(path: Path) -> JsonDict:
    """Decode one JSON object while malformed evidence stays a failed gate."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    upstream_artifact: Path = DEFAULT_UPSTREAM_ARTIFACT,
) -> tuple[list[JsonDict], JsonDict, dict[str, str | None]]:
    """Authenticate the fixture before any learner sees an authority row."""

    root = Path(repo_root)
    upstream_path = _resolve(root, upstream_artifact)
    fixture = _load_object(upstream_path)
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    roadmap_path = root / "research-roadmap.yaml"
    roadmap_text = roadmap_path.read_text(encoding="utf-8") if roadmap_path.is_file() else ""
    references_path = root / "research-references.md"
    references_text = (
        references_path.read_text(encoding="utf-8") if references_path.is_file() else ""
    )
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion_text = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    source_hashes: dict[str, str | None] = {
        str(path): _sha256_path(root / path) for path in SOURCE_PATHS
    }
    source_hashes[str(upstream_artifact)] = _sha256_path(upstream_path)
    declared_hashes = unwrap_principled(fixture.get("source_artifact_hashes", {}))
    fixture_paths = _fixture_paths(fixture)
    actual_fixture_hashes = {
        str(path): _sha256_path(_resolve(root, path)) for path in fixture_paths
    }
    for path, value in actual_fixture_hashes.items():
        source_hashes[path] = value
    stream_hash_match = isinstance(declared_hashes, Mapping) and all(
        declared_hashes.get(path) == value
        for path, value in actual_fixture_hashes.items()
        if "checkpoint" not in path
    )
    checkpoint_hash = _sha256_path(_resolve(root, fixture_paths[-1])) if fixture_paths else None
    fixture_gate = unwrap_principled(fixture.get("gate_check_summary", {}))
    quarantine = quarantine_state(
        fixture,
        exclusion_text,
        DEFAULT_UPSTREAM_ARTIFACT.name,
        "exp7212-refinement-fixture",
    )
    own_sizes = {
        str(path): (root / path).stat().st_size if (root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    imports = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "carnot.experiment_7199_v634_bounded_acquisition",
            "carnot.experiment_7212_v635_refinement_fixture",
            "carnot.memory.transactional_constraint_memory",
        )
    }
    tools = {
        "python": Path(sys.executable).is_file(),
        "jq": shutil.which("jq") is not None,
        "sha256sum": shutil.which("sha256sum") is not None,
    }
    destinations = {
        "checkpoint": _path_writable(paths.checkpoint),
        "artifact": _path_writable(paths.artifact),
    }
    expected_identity = {
        "id": "exp7213-refinement-learning",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT_PATH),
    }
    checksum_valid = bool(fixture) and fixture.get(
        "reproducibility_checksum"
    ) == exp7212.reproducibility_checksum(fixture)
    checks = [
        gate_check(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-CL-7213",
            True,
            "## REQ-CL-7213:" in spec_text,
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7213-*",
            8,
            spec_text.count("### SCENARIO-CL-7213-"),
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            {str(path): "nonempty" for path in SOURCE_PATHS},
            {key: "nonempty" if value else None for key, value in own_sizes.items()},
        ),
        gate_check(
            "required_source_hashes",
            "repository",
            "SOURCE_PATHS.sha256",
            True,
            all(
                re.fullmatch(r"sha256:[0-9a-f]{64}", value or "")
                for value in source_hashes.values()
            ),
        ),
        gate_check(
            "cited_source_bytes",
            "research-references.md",
            "arXiv:2509.24489",
            True,
            "2509.24489" in references_text
            and "Query-Driven Interactive Refinement for Constraint Acquisition" in references_text,
        ),
        gate_check(
            "v635_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            expected_identity,
            _task_identity(roadmap_text),
        ),
        gate_check(
            "required_imports", "python", "imports", {key: True for key in imports}, imports
        ),
        gate_check(
            "required_tools", "host", "python,jq,sha256sum", {key: True for key in tools}, tools
        ),
        gate_check(
            "output_destinations",
            "host_filesystem",
            "checkpoint,artifact",
            {key: True for key in destinations},
            destinations,
        ),
        gate_check(
            "exp7212_artifact_hash",
            "exp7212-refinement-fixture",
            str(DEFAULT_UPSTREAM_ARTIFACT),
            EXPECTED_UPSTREAM_SHA256,
            source_hashes[str(upstream_artifact)],
        ),
        gate_check(
            "exp7212_status",
            "exp7212-refinement-fixture",
            "status",
            "complete",
            fixture.get("status"),
        ),
        gate_check(
            "exp7212_run_date",
            "exp7212-refinement-fixture",
            "run_date",
            RUN_DATE,
            fixture.get("run_date"),
        ),
        gate_check(
            "exp7212_readiness",
            "exp7212-refinement-fixture",
            "refinement_fixture_ready_score",
            1,
            fixture.get("refinement_fixture_ready_score"),
        ),
        gate_check(
            "exp7212_gate",
            "exp7212-refinement-fixture",
            "gate_check_summary.passed",
            True,
            fixture_gate.get("passed") if isinstance(fixture_gate, Mapping) else None,
        ),
        gate_check(
            "exp7212_no_model",
            "exp7212-refinement-fixture",
            "MODEL_SPECS,model_invoked",
            {"MODEL_SPECS": [], "model_invoked": False},
            {
                "MODEL_SPECS": unwrap_principled(fixture.get("MODEL_SPECS")),
                "model_invoked": unwrap_principled(fixture.get("model_invoked")),
            },
        ),
        gate_check(
            "exp7212_not_quarantined",
            "artifact_metadata_and_ops/exclusion_manifest.yaml",
            "quarantined",
            False,
            quarantine["quarantined"],
        ),
        gate_check(
            "exp7212_reproducibility_checksum",
            "exp7212-refinement-fixture",
            "reproducibility_checksum",
            True,
            checksum_valid,
        ),
        gate_check(
            "exp7212_stream_hashes",
            "exp7212.source_artifact_hashes",
            "declared_paths.sha256",
            True,
            stream_hash_match,
        ),
        gate_check(
            "exp7212_checkpoint_hash",
            "exp7212-refinement-fixture",
            "checkpoint_hash",
            fixture.get("checkpoint_hash"),
            checkpoint_hash,
        ),
    ]
    return checks, fixture, source_hashes


def load_fixture_views(repo_root: Path, upstream_artifact: Path) -> FixtureViews:
    """Load frozen rows only after callers have completed producer gates."""

    root = Path(repo_root)
    fixture = _load_object(_resolve(root, upstream_artifact))
    public_path, authority_path, manifest_path, warmup_path, _, _ = _fixture_paths(fixture)
    public = exp7212.read_jsonl(_resolve(root, public_path))
    authority = exp7212.read_jsonl(_resolve(root, authority_path))
    warmup = exp7212.read_jsonl(_resolve(root, warmup_path))
    manifest = _load_object(_resolve(root, manifest_path))
    public_by_seed = {
        seed: [row for row in public if int(row["seed"]) == seed] for seed in STREAM_SEEDS
    }
    authority_by_id = {
        str(row["event_id"]): row for row in authority if row.get("row_type") == "event_authority"
    }
    warmup_by_seed = {
        seed: [row for row in warmup if int(row["seed"]) == seed] for seed in STREAM_SEEDS
    }
    return FixtureViews(fixture, public_by_seed, authority_by_id, warmup_by_seed, manifest)


def _stable_rank(*parts: Any) -> int:
    """Derive deterministic local ranks without Python's salted hash."""

    return int(transactional.sha256_json(list(parts)).removeprefix("sha256:")[:16], 16)


def _reserved_by_seed(manifest: Mapping[str, Any], seed: int) -> dict[str, set[int]]:
    """Restore the validation split that the fixture froze before fitting."""

    return {
        str(row["family_id"]): {int(value) for value in row["reserved_validation_x"]}
        for row in manifest["validation_partitions"]
        if int(row["seed"]) == seed
    }


def _prime_candidate(
    controller: exp7212.CandidateRefinementController,
    warmup: Sequence[Mapping[str, Any]],
    memories: Mapping[str, transactional.TransactionalConstraintMemory],
) -> None:
    """Give private fitting state the same already released warmup evidence."""

    for release in warmup:
        family = str(release["family_id"])
        value = int(release["numeric_value"])
        label = str(release["observed_label"])
        state = controller.families[family]
        state.hypotheses = {
            parameter
            for parameter in state.hypotheses
            if exp7212.exact_label(family, value, parameter) == label
        }
        state.support_ids.append(str(release["event_id"]))
    for family, state in controller.families.items():
        if len(state.hypotheses) == 1:
            state.candidate_parameter = next(iter(state.hypotheses))
            state.expected_parent_hash = memories[family].state_hash()


def _prime_version_space(
    controller: exp7199.VersionSpaceController,
    warmup: Sequence[Mapping[str, Any]],
) -> None:
    """Give the strong predictor every released warmup label."""

    for release in sorted(
        warmup, key=lambda row: (int(row["release_index"]), int(row["request_index"]))
    ):
        controller.observe(
            release,
            observed_label=str(release["observed_label"]),
            role="support",
            request_index=int(release["request_index"]),
            release_index=int(release["release_index"]),
        )


def _new_runtime(
    arm: str,
    seed: int,
    warmup: Sequence[Mapping[str, Any]],
    reserved: Mapping[str, set[int]],
    state_root: Path,
) -> ArmRuntime:
    """Create one arm without sharing mutable state with another arm or stream."""

    fallback = exp7212.FrozenWarmupFallback.from_releases(warmup)
    costs = {
        name: []
        for name in (
            "lookup_ns",
            "selection_ns",
            "queue_ns",
            "score_ns",
            "update_ns",
            "validation_ns",
            "commit_ns",
        )
    }
    runtime = ArmRuntime(
        arm=arm,
        seed=seed,
        fallback=fallback,
        validation_by_family={family: 0 for family in FAMILIES},
        costs=costs,
        operation_counts={name: 0 for name in costs},
    )
    if arm in COMMITTED_ARMS:
        runtime.memories = {
            family: transactional.TransactionalConstraintMemory(
                state_root / str(seed) / arm / family
            )
            for family in FAMILIES
        }
        runtime.controller = exp7212.CandidateRefinementController(
            stream_id=f"stream-{seed}",
            fallback=fallback,
            reserved_validation=reserved,
        )
        _prime_candidate(runtime.controller, warmup, runtime.memories)
    elif arm == "witness_query_version_space":
        runtime.version_controller = exp7199.VersionSpaceController()
        _prime_version_space(runtime.version_controller, warmup)
    return runtime


def _runtime_memory_bytes(runtime: ArmRuntime) -> int:
    """Charge controller, memory, compiled dispatch, and private queue bytes."""

    controller_state: Any = None
    if runtime.controller is not None:
        controller_state = runtime.controller.state_dict()
    elif runtime.version_controller is not None:
        controller_state = runtime.version_controller.state_dict()
    pending = [
        {
            "event_id": row.event["event_id"],
            "role": row.role,
            "request_index": row.request_index,
            "release_index": row.release_index,
            "observed_label": row.observed_label,
            "exact_label": row.exact_label,
            "poisoned": row.poisoned,
        }
        for row in runtime.pending
    ]
    memory_bytes = sum(len(memory.state_bytes()) for memory in runtime.memories.values())
    return (
        len(transactional.canonical_json_bytes(controller_state))
        + memory_bytes
        + len(transactional.canonical_json_bytes(runtime.compiled_parameters))
        + len(transactional.canonical_json_bytes(pending))
    )


def _compiled_prediction(runtime: ArmRuntime, event: Mapping[str, Any]) -> tuple[str, str, int]:
    """Dispatch a committed exact predicate or the immutable fallback directly."""

    family = str(event["family_id"])
    parameter = runtime.compiled_parameters.get(family)
    if parameter is None:
        return runtime.fallback.predict(event), "frozen_warmup_fallback", 0
    version = json.loads(runtime.memories[family].state_bytes())["version"]
    return (
        exp7212.exact_label(family, int(event["numeric_value"]), parameter),
        "committed_exact_predicate",
        int(version),
    )


def _predict(runtime: ArmRuntime, event: Mapping[str, Any]) -> tuple[str, str, int]:
    """Use only the deployed surface assigned to this matched arm."""

    if runtime.arm == "warmup_frozen":
        return runtime.fallback.predict(event), "frozen_warmup_fallback", 0
    if runtime.version_controller is not None:
        prediction, _ = runtime.version_controller.predict(event)
        return prediction, "online_version_space_majority", 0
    return _compiled_prediction(runtime, event)


def _random_fitting_ids(
    seed: int,
    events: Sequence[Mapping[str, Any]],
    reserved: Mapping[str, set[int]],
) -> set[str]:
    """Freeze 48 random fitting slots from public bytes before labels are read."""

    eligible = [
        row
        for row in events
        if int(row["chronology_index"]) >= WARMUP_COUNT
        and int(row["numeric_value"]) not in reserved[str(row["family_id"])]
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            _stable_rank("exp7213-random-query", BOOTSTRAP_SEED, seed, row["event_id"]),
            str(row["event_id"]),
        ),
    )
    return {str(row["event_id"]) for row in ranked[:FITTING_BUDGET]}


def _query_decision(
    runtime: ArmRuntime,
    event: Mapping[str, Any],
    random_ids: set[str],
) -> tuple[str, str] | None:
    """Select a role from public bytes and released learner state only."""

    if runtime.arm == "warmup_frozen" or runtime.query_count >= QUERY_BUDGET:
        return None
    if int(event["chronology_index"]) < WARMUP_COUNT:
        return None
    family = str(event["family_id"])
    value = int(event["numeric_value"])
    if runtime.arm == "witness_query_version_space":
        return None
    if runtime.controller is None:
        return None
    state = runtime.controller.families[family]
    reserved = runtime.controller.reserved_validation[family]
    if (
        state.candidate_parameter is not None
        and state.active_commit_receipt is None
        and value in reserved
        and runtime.validation_count < VALIDATION_BUDGET
        and runtime.validation_by_family[family] < VALIDATION_PER_FAMILY
    ):
        return "validation", "reserved_post_singleton_validation"
    if value in reserved or runtime.fitting_count >= FITTING_BUDGET:
        return None
    if runtime.arm == "passive_query_committed":
        return "fitting", "passive_next_eligible_public_event"
    if runtime.arm == "random_query_committed":
        if str(event["event_id"]) in random_ids:
            return "fitting", "frozen_random_public_rank"
        return None
    if state.active_commit_receipt is not None:
        return None
    witness = exp7212.choose_witness(family, state.hypotheses, reserved_values=reserved)
    if witness is not None and value == witness:
        return "fitting", "maximally_balanced_lowest_witness"
    return None


def _refresh_compiled(runtime: ArmRuntime, family: str) -> None:
    """Compile only the certified record that transactional memory exposes."""

    key = f"predicate:stream-{runtime.seed}:{family}"
    record = next(
        (row for row in runtime.memories[family].records() if row.get("key") == key),
        None,
    )
    if record is None or record.get("certified") is not True:
        runtime.compiled_parameters.pop(family, None)
    else:
        runtime.compiled_parameters[family] = int(record["parameter"])


def _apply_committed_release(runtime: ArmRuntime, pending: PendingRecord) -> JsonDict:
    """Apply one released label and restore a fresh epoch after contradiction."""

    if runtime.controller is None:
        raise RuntimeError("committed_release_requires_controller")
    family = str(pending.event["family_id"])
    memory = runtime.memories[family]
    before_state = runtime.controller.families[family]
    active_receipt = deepcopy(before_state.active_commit_receipt)
    started = time.perf_counter_ns()
    if pending.role == "validation":
        result = runtime.controller.observe_validation(
            pending.event,
            pending.observed_label,
            memory=memory,
        )
    else:
        result = runtime.controller.observe_fitting(
            pending.event,
            pending.observed_label,
            memory=memory,
        )
        if result["operation"] == "candidate_set_empty":
            rollback = memory.rollback(active_receipt) if active_receipt is not None else None
            runtime.controller._reset_family(family)
            reset = runtime.controller.families[family]
            value = int(pending.event["numeric_value"])
            reset.hypotheses = {
                parameter
                for parameter in PARAMETER_DOMAIN
                if exp7212.exact_label(family, value, parameter) == pending.observed_label
            }
            reset.support_ids = [str(pending.event["event_id"])]
            if len(reset.hypotheses) == 1:
                reset.candidate_parameter = next(iter(reset.hypotheses))
                reset.expected_parent_hash = memory.state_hash()
            result["operation"] = (
                "revoke_on_fitting_contradiction"
                if active_receipt is not None
                else "refit_after_empty"
            )
            result["rollback_receipt"] = rollback
            result["hypotheses_after"] = sorted(reset.hypotheses)
    elapsed = time.perf_counter_ns() - started
    _refresh_compiled(runtime, family)
    result["update_ns"] = elapsed
    result["support_ids"] = list(runtime.controller.families[family].support_ids)
    result["validation_ids"] = list(runtime.controller.families[family].validation_ids)
    result["memory_version"] = int(json.loads(memory.state_bytes())["version"])
    return result


def _apply_version_release(runtime: ArmRuntime, pending: PendingRecord) -> JsonDict:
    """Update the shipped majority predictor from the same witness release."""

    if runtime.version_controller is None:
        raise RuntimeError("version_release_requires_controller")
    started = time.perf_counter_ns()
    result = runtime.version_controller.observe(
        pending.event,
        observed_label=pending.observed_label,
        role="support" if pending.role == "fitting" else "validation",
        request_index=pending.request_index,
        release_index=pending.release_index,
    )
    result["update_ns"] = time.perf_counter_ns() - started
    result["memory_version"] = int(result["epoch"])
    return result


def _apply_release(runtime: ArmRuntime, pending: PendingRecord) -> JsonDict:
    """Route a release only to a runtime that owns a learning controller."""

    if runtime.controller is not None:
        return _apply_committed_release(runtime, pending)
    if runtime.version_controller is not None:
        return _apply_version_release(runtime, pending)
    raise RuntimeError("released_feedback_requires_learning_arm")


def _state_hash(runtime: ArmRuntime) -> str:
    """Hash acquisition state without treating host timing as learned state."""

    if runtime.controller is not None:
        state: Any = runtime.controller.state_dict()
    elif runtime.version_controller is not None:
        state = runtime.version_controller.state_dict()
    else:
        state = runtime.fallback.state_dict()
    return transactional.sha256_json(state)


def _checkpoint_state(runtime: ArmRuntime) -> JsonDict:
    """Serialize all state that can affect a later action without hidden labels."""

    if runtime.controller is not None:
        controller_state: Any = runtime.controller.state_dict()
    elif runtime.version_controller is not None:
        controller_state = runtime.version_controller.state_dict()
    else:
        controller_state = None
    memory_states = {
        family: {
            "state": json.loads(memory.state_bytes()),
            "state_bytes_b64": base64.b64encode(memory.state_bytes()).decode("ascii"),
            "state_hash": memory.state_hash(),
        }
        for family, memory in runtime.memories.items()
    }
    return {
        "unit_id": f"{runtime.seed}:{runtime.arm}",
        "seed": runtime.seed,
        "arm": runtime.arm,
        "fallback": runtime.fallback.state_dict(),
        "controller_state": controller_state,
        "memory_states": memory_states,
        "compiled_parameters": dict(runtime.compiled_parameters),
        "pending_public_state": [
            {
                "event_id": row.event["event_id"],
                "family_id": row.event["family_id"],
                "numeric_value": row.event["numeric_value"],
                "role": row.role,
                "request_index": row.request_index,
                "release_index": row.release_index,
            }
            for row in runtime.pending
        ],
        "query_count": runtime.query_count,
        "fitting_count": runtime.fitting_count,
        "validation_count": runtime.validation_count,
        "state_hash": _state_hash(runtime),
        "reloadable": True,
    }


def _record_cost(runtime: ArmRuntime, name: str, elapsed: int) -> None:
    """Retain each measured CPU cost and its operation count."""

    runtime.costs[name].append(elapsed)
    runtime.operation_counts[name] += 1


def _aggregate_row(runtime: ArmRuntime, decisions: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce one full stream while keeping prospective and phase denominators."""

    prospective = [row for row in decisions if row["prospective"]]
    recurrence = [row for row in decisions if row["phase"] == "recurrence"]
    drift = [row for row in decisions if row["phase"] == "drift"]
    poison = [row for row in decisions if row["phase"] == "poison"]
    error = sum(int(row["error"]) for row in prospective)
    false_accept = sum(int(row["false_accept"]) for row in prospective)
    abstention = sum(int(row["abstention"]) for row in prospective)
    total_cost = sum(sum(values) for values in runtime.costs.values())
    return {
        "unit_id": f"{runtime.seed}:{runtime.arm}",
        "arm": runtime.arm,
        "seed": runtime.seed,
        "metric": "prospective_full_denominator_error",
        "error": error,
        "abstention": abstention,
        "event_count": len(prospective),
        "error_rate": error / len(prospective),
        "false_accept": false_accept,
        "false_accept_rate": false_accept / len(prospective),
        "recurrence_error": sum(int(row["error"]) for row in recurrence),
        "recurrence_error_rate": sum(int(row["error"]) for row in recurrence) / len(recurrence),
        "drift_error": sum(int(row["error"]) for row in drift),
        "drift_error_rate": sum(int(row["error"]) for row in drift) / len(drift),
        "poison_error": sum(int(row["error"]) for row in poison),
        "poison_error_rate": sum(int(row["error"]) for row in poison) / len(poison),
        "query_count": runtime.query_count,
        "fitting_query_count": runtime.fitting_count,
        "validation_query_count": runtime.validation_count,
        "released_query_count": 0,
        "pending_at_end": len(runtime.pending),
        "max_pending": runtime.max_pending,
        "max_memory_bytes": runtime.max_memory_bytes,
        "total_cost_ns": total_cost,
        "amortized_throughput_events_per_s": (
            0.0 if total_cost == 0 else EVENTS_PER_SEED * 1_000_000_000 / total_cost
        ),
        "operation_counts": dict(runtime.operation_counts),
    }


def run_learning_panel(
    views: FixtureViews,
    *,
    state_root: Path,
    seeds: Sequence[int] = STREAM_SEEDS,
    progress: bool = False,
) -> LearningPanel:
    """Replay every matched arm while authority access follows sealed actions."""

    started = time.monotonic()
    rows: list[JsonDict] = []
    decision_rows: list[JsonDict] = []
    query_rows: list[JsonDict] = []
    deletion_rows: list[JsonDict] = []
    state_rows: list[JsonDict] = []
    latency_values: dict[str, list[int]] = {
        name: []
        for name in (
            "lookup_ns",
            "selection_ns",
            "queue_ns",
            "score_ns",
            "update_ns",
            "validation_ns",
            "commit_ns",
        )
    }
    for seed_number, seed in enumerate(seeds, start=1):
        events = views.public_by_seed[seed]
        warmup = views.warmup_by_seed[seed]
        reserved = _reserved_by_seed(views.manifest, seed)
        random_ids = _random_fitting_ids(seed, events, reserved)
        runtimes = {arm: _new_runtime(arm, seed, warmup, reserved, state_root) for arm in ARMS}
        unit_decisions: dict[str, list[JsonDict]] = {arm: [] for arm in ARMS}
        for runtime in runtimes.values():
            runtime.max_memory_bytes = _runtime_memory_bytes(runtime)

        for event in events:
            event_id = str(event["event_id"])
            chronology_index = int(event["chronology_index"])
            drafts: dict[str, JsonDict] = {}
            witness_decision: tuple[str, str] | None = None
            witness_selection_ns = 0
            for arm in ARMS:
                runtime = runtimes[arm]
                runtime.operation_index += 1
                lookup_started = time.perf_counter_ns()
                prediction, source, memory_version = _predict(runtime, event)
                lookup_ns = time.perf_counter_ns() - lookup_started
                _record_cost(runtime, "lookup_ns", lookup_ns)
                prediction_operation = runtime.operation_index

                runtime.operation_index += 1
                if arm == "witness_query_version_space":
                    query_decision = witness_decision
                    selection_ns = witness_selection_ns
                else:
                    selection_started = time.perf_counter_ns()
                    query_decision = _query_decision(runtime, event, random_ids)
                    selection_ns = time.perf_counter_ns() - selection_started
                    if arm == "witness_query_committed":
                        witness_decision = query_decision
                        witness_selection_ns = selection_ns
                _record_cost(runtime, "selection_ns", selection_ns)
                query_operation = runtime.operation_index
                query_charged = False
                if query_decision is not None:
                    role, reason = query_decision
                    pending_before = len(runtime.pending)
                    query_row: JsonDict = {
                        "unit_id": f"{seed}:{arm}",
                        "arm": arm,
                        "seed": seed,
                        "event_id": event_id,
                        "family_id": event["family_id"],
                        "query_value": event["numeric_value"],
                        "role": role,
                        "reason": reason,
                        "request_index": chronology_index,
                        "request_operation_index": query_operation,
                        "release_index": None,
                        "release_operation_index": None,
                        "pending_before": pending_before,
                        "pending_after": pending_before,
                        "query_charged": False,
                        "request_status": "capacity_full",
                        "released": False,
                        "observed_label": None,
                        "poisoned": None,
                        "version_before": memory_version,
                        "version_after": memory_version,
                        "support_ids": [],
                        "validation_ids": [],
                        "commit_event": None,
                        "revoke_event": None,
                        "prediction_before_release": prediction,
                        "prediction_after_release": prediction,
                        "prediction_changed": False,
                        "selection_ns": selection_ns,
                        "queue_ns": 0,
                        "update_ns": 0,
                        "validation_ns": 0,
                        "commit_ns": 0,
                        "predicted_before_request": prediction_operation < query_operation,
                        "future_label_visible_to_selector": False,
                        "hidden_parameter_visible_to_selector": False,
                        "hidden_audit_visible_to_selector": False,
                        "validation_used_for_elimination": False,
                    }
                    if len(runtime.pending) < PENDING_CAPACITY:
                        truth = views.authority_by_id[event_id]
                        queue_started = time.perf_counter_ns()
                        query_row["release_index"] = int(truth["release_index"])
                        query_row["request_status"] = "admitted"
                        query_row["query_charged"] = True
                        runtime.pending.append(
                            PendingRecord(
                                event=dict(event),
                                role=role,
                                request_index=chronology_index,
                                release_index=int(truth["release_index"]),
                                observed_label=str(truth["observed_label"]),
                                exact_label=str(truth["exact_label"]),
                                poisoned=bool(truth["poisoned"]),
                                query_row=query_row,
                            )
                        )
                        queue_ns = time.perf_counter_ns() - queue_started
                        query_row["queue_ns"] = queue_ns
                        runtime.query_count += 1
                        runtime.fitting_count += int(role == "fitting")
                        runtime.validation_count += int(role == "validation")
                        runtime.validation_by_family[str(event["family_id"])] += int(
                            role == "validation"
                        )
                        query_row["pending_after"] = len(runtime.pending)
                        query_charged = True
                        _record_cost(runtime, "queue_ns", queue_ns)
                    query_rows.append(query_row)
                runtime.max_pending = max(runtime.max_pending, len(runtime.pending))
                runtime.max_memory_bytes = max(
                    runtime.max_memory_bytes, _runtime_memory_bytes(runtime)
                )
                drafts[arm] = {
                    "unit_id": f"{seed}:{arm}",
                    "arm": arm,
                    "seed": seed,
                    "event_id": event_id,
                    "chronology_index": chronology_index,
                    "family_id": event["family_id"],
                    "numeric_value": event["numeric_value"],
                    "prediction": prediction,
                    "prediction_source": source,
                    "memory_version": memory_version,
                    "prediction_operation_index": prediction_operation,
                    "query_decision_operation_index": query_operation,
                    "authority_score_operation_index": None,
                    "query_charged": query_charged,
                    "prospective": chronology_index >= WARMUP_COUNT,
                    "lookup_ns": lookup_ns,
                    "selection_ns": selection_ns,
                    "future_label_visible_to_decision": False,
                    "hidden_parameter_visible_to_decision": False,
                    "hidden_audit_visible_to_decision": False,
                }

            truth = views.authority_by_id[event_id]
            for arm in ARMS:
                runtime = runtimes[arm]
                draft = drafts[arm]
                runtime.operation_index += 1
                score_started = time.perf_counter_ns()
                exact_label = str(truth["exact_label"])
                prediction = str(draft["prediction"])
                error = int(prediction == "abstain" or prediction != exact_label)
                false_accept = int(prediction == "accept" and exact_label == "reject")
                abstention = int(prediction == "abstain")
                score_ns = time.perf_counter_ns() - score_started
                _record_cost(runtime, "score_ns", score_ns)
                draft.update(
                    {
                        "phase": truth["phase"],
                        "exact_label": exact_label,
                        "observed_feedback_label": truth["observed_label"],
                        "poisoned_feedback": bool(truth["poisoned"]),
                        "error": error,
                        "false_accept": false_accept,
                        "abstention": abstention,
                        "score_ns": score_ns,
                        "authority_score_operation_index": runtime.operation_index,
                    }
                )
                decision_rows.append(draft)
                unit_decisions[arm].append(draft)
                if arm == "witness_query_committed":
                    acquisition_hash = _state_hash(runtime)
                    deleted_prediction = runtime.fallback.predict(event)
                    deleted_error = int(
                        deleted_prediction == "abstain" or deleted_prediction != exact_label
                    )
                    deletion_rows.append(
                        {
                            "unit_id": f"{seed}:{arm}",
                            "arm": arm,
                            "seed": seed,
                            "event_id": event_id,
                            "chronology_index": chronology_index,
                            "prospective": chronology_index >= WARMUP_COUNT,
                            "prediction": prediction,
                            "error": error,
                            "deleted_prediction": deleted_prediction,
                            "deleted_error": deleted_error,
                            "full_reset_prediction": runtime.fallback.predict(event),
                            "template_count_before": len(runtime.compiled_parameters),
                            "template_count_after": 0,
                            "deletion_changed_prediction": prediction != deleted_prediction,
                            "acquisition_state_hash_before": acquisition_hash,
                            "acquisition_state_hash_after": acquisition_hash,
                            "acquisition_state_preserved": True,
                            "full_reset_is_separate": True,
                            "shadow_read_only": True,
                        }
                    )

            for arm in ARMS:
                runtime = runtimes[arm]
                released = [
                    pending
                    for pending in runtime.pending
                    if pending.release_index <= chronology_index
                ]
                for pending in released:
                    runtime.operation_index += 1
                    before_prediction = _predict(runtime, pending.event)[0]
                    update = _apply_release(runtime, pending)
                    update_ns = int(update["update_ns"])
                    _record_cost(runtime, "update_ns", update_ns)
                    if pending.role == "validation":
                        _record_cost(runtime, "validation_ns", update_ns)
                    commit_decision = update.get("commit_receipt")
                    commit_event = None
                    commit_ns = 0
                    if isinstance(commit_decision, Mapping) and commit_decision.get("admitted"):
                        commit_event = commit_decision["commit_receipt"]["event_id"]
                        commit_ns = update_ns
                        _record_cost(runtime, "commit_ns", commit_ns)
                    revoke_event = (
                        str(pending.event["event_id"])
                        if str(update.get("operation", "")).startswith("revoke")
                        else None
                    )
                    after_prediction = _predict(runtime, pending.event)[0]
                    pending.query_row.update(
                        {
                            "released": True,
                            "release_operation_index": runtime.operation_index,
                            "observed_label": pending.observed_label,
                            "poisoned": pending.poisoned,
                            "version_after": update.get("memory_version", 0),
                            "support_ids": list(update.get("support_ids", [])),
                            "validation_ids": list(update.get("validation_ids", [])),
                            "commit_event": commit_event,
                            "revoke_event": revoke_event,
                            "update_operation": update.get("operation"),
                            "prediction_after_release": after_prediction,
                            "prediction_changed": before_prediction != after_prediction,
                            "update_ns": update_ns,
                            "validation_ns": update_ns if pending.role == "validation" else 0,
                            "commit_ns": commit_ns,
                        }
                    )
                released_ids = {pending.event["event_id"] for pending in released}
                runtime.pending = [
                    pending
                    for pending in runtime.pending
                    if pending.event["event_id"] not in released_ids
                ]
                runtime.max_memory_bytes = max(
                    runtime.max_memory_bytes, _runtime_memory_bytes(runtime)
                )

        for arm, runtime in runtimes.items():
            row = _aggregate_row(runtime, unit_decisions[arm])
            row["released_query_count"] = sum(
                int(query["released"])
                for query in query_rows
                if query["unit_id"] == row["unit_id"] and query["query_charged"]
            )
            rows.append(row)
            state_rows.append(_checkpoint_state(runtime))
            for name, values in runtime.costs.items():
                latency_values[name].extend(values)
        if progress:
            print(
                f"PHASE 3 PROGRESS: completed stream {seed_number}/{len(seeds)}; "
                f"decisions={len(decision_rows)}; elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
    return LearningPanel(
        rows,
        decision_rows,
        query_rows,
        deletion_rows,
        state_rows,
        latency_values,
    )


def panel_conformance_errors(
    panel: LearningPanel,
    *,
    seeds: Sequence[int] = STREAM_SEEDS,
) -> list[str]:
    """Check chronology, full denominators, matched schedules, and query bounds."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    expected_units = len(seeds) * len(ARMS)
    add(len(panel.rows) != expected_units, "aggregate_row_count")
    add(
        len(panel.decision_rows) != expected_units * EVENTS_PER_SEED,
        "decision_row_count",
    )
    add(
        len(panel.commit_deletion_rows) != len(seeds) * EVENTS_PER_SEED,
        "deletion_row_count",
    )
    add(
        any(
            int(row["prediction_operation_index"]) >= int(row["query_decision_operation_index"])
            or int(row["query_decision_operation_index"])
            >= int(row["authority_score_operation_index"])
            for row in panel.decision_rows
        ),
        "prediction_authority_chronology",
    )
    add(
        any(
            row["future_label_visible_to_decision"] is not False
            or row["hidden_parameter_visible_to_decision"] is not False
            or row["hidden_audit_visible_to_decision"] is not False
            for row in panel.decision_rows
        ),
        "decision_authority_access",
    )
    add(
        any(row["abstention"] == 1 and row["error"] != 1 for row in panel.decision_rows),
        "abstention_denominator",
    )
    add(any(int(row["query_count"]) > QUERY_BUDGET for row in panel.rows), "query_budget")
    add(
        any(int(row["fitting_query_count"]) > FITTING_BUDGET for row in panel.rows),
        "fitting_budget",
    )
    add(
        any(int(row["validation_query_count"]) > VALIDATION_BUDGET for row in panel.rows),
        "validation_budget",
    )
    add(
        any(int(row["max_pending"]) > PENDING_CAPACITY for row in panel.rows),
        "pending_capacity",
    )
    add(
        any(
            row["validation_used_for_elimination"] is not False
            or row["predicted_before_request"] is not True
            or row["future_label_visible_to_selector"] is not False
            or row["hidden_parameter_visible_to_selector"] is not False
            or row["hidden_audit_visible_to_selector"] is not False
            for row in panel.query_rows
        ),
        "validation_or_selector_access",
    )
    add(
        any(
            row["released"]
            and int(row["release_operation_index"]) <= int(row["request_operation_index"])
            for row in panel.query_rows
        ),
        "request_release_chronology",
    )
    for seed in seeds:
        witness = [
            (row["event_id"], row["role"], row["release_index"])
            for row in panel.query_rows
            if row["seed"] == seed
            and row["arm"] == "witness_query_committed"
            and row["query_charged"]
        ]
        version = [
            (row["event_id"], row["role"], row["release_index"])
            for row in panel.query_rows
            if row["seed"] == seed
            and row["arm"] == "witness_query_version_space"
            and row["query_charged"]
        ]
        add(witness != version, "witness_schedule_parity")
    add(
        any(
            row["acquisition_state_hash_before"] != row["acquisition_state_hash_after"]
            or row["template_count_after"] != 0
            or row["acquisition_state_preserved"] is not True
            or row["shadow_read_only"] is not True
            for row in panel.commit_deletion_rows
        ),
        "template_deletion_intervention",
    )
    return errors


def _percentile(values: Sequence[float | int], probability: float) -> float:
    """Return one deterministic nearest-rank percentile."""

    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(probability * len(ordered))))
    return ordered[index]


def _bootstrap_interval(
    values: Mapping[int, float],
    *,
    comparison_id: str,
    draws: int,
) -> JsonDict:
    """Resample complete stream seeds as the only independent units."""

    seeds = sorted(values)
    point = statistics.fmean(values.values()) if values else 0.0
    rng = random.Random(_stable_rank("exp7213-bootstrap", BOOTSTRAP_SEED, comparison_id))
    samples = [
        statistics.fmean(values[rng.choice(seeds)] for _ in seeds) if seeds else 0.0
        for _ in range(draws)
    ]
    return {
        "comparison_id": comparison_id,
        "estimate": point,
        "ci95_lower": _percentile(samples, 0.025),
        "ci95_upper": _percentile(samples, 0.975),
        "independent_stream_count": len(seeds),
        "bootstrap_draws": draws,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def build_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    deletion_rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> list[JsonDict]:
    """Build every preregistered paired stream comparison."""

    by_arm_seed = {(str(row["arm"]), int(row["seed"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})

    def difference(left: str, right: str, field_name: str) -> dict[int, float]:
        return {
            seed: float(by_arm_seed[(left, seed)][field_name])
            - float(by_arm_seed[(right, seed)][field_name])
            for seed in seeds
        }

    comparisons = [
        _bootstrap_interval(
            difference("witness_query_committed", "warmup_frozen", "error_rate"),
            comparison_id="future_error_change_vs_warmup_frozen",
            draws=draws,
        ),
        _bootstrap_interval(
            difference("witness_query_committed", "random_query_committed", "error_rate"),
            comparison_id="future_error_change_vs_random_query_committed",
            draws=draws,
        ),
        _bootstrap_interval(
            difference("witness_query_committed", "passive_query_committed", "error_rate"),
            comparison_id="future_error_change_vs_passive_query_committed",
            draws=draws,
        ),
        _bootstrap_interval(
            difference("witness_query_committed", "witness_query_version_space", "error_rate"),
            comparison_id="future_error_change_vs_version_space",
            draws=draws,
        ),
        _bootstrap_interval(
            difference("witness_query_committed", "warmup_frozen", "false_accept_rate"),
            comparison_id="false_accept_change_vs_warmup_frozen",
            draws=draws,
        ),
        _bootstrap_interval(
            difference("witness_query_committed", "warmup_frozen", "recurrence_error_rate"),
            comparison_id="recurrence_error_change_vs_warmup_frozen",
            draws=draws,
        ),
    ]
    accuracy_values = {
        seed: -difference("witness_query_committed", "witness_query_version_space", "error_rate")[
            seed
        ]
        for seed in seeds
    }
    comparisons.append(
        _bootstrap_interval(
            accuracy_values,
            comparison_id="compiled_accuracy_change_vs_version_space",
            draws=draws,
        )
    )
    throughput_values = {
        seed: (
            float(by_arm_seed[("witness_query_version_space", seed)]["total_cost_ns"])
            / max(
                1.0,
                float(by_arm_seed[("witness_query_committed", seed)]["total_cost_ns"]),
            )
        )
        for seed in seeds
    }
    comparisons.append(
        _bootstrap_interval(
            throughput_values,
            comparison_id="compiled_throughput_ratio_vs_version_space",
            draws=draws,
        )
    )
    deletion_values: dict[int, float] = {}
    for seed in seeds:
        selected = [
            row
            for row in deletion_rows
            if int(row["seed"]) == seed
            and row.get("prospective") is True
            and row.get("arm", "witness_query_committed") == "witness_query_committed"
        ]
        deletion_values[seed] = (
            statistics.fmean(float(row["deleted_error"]) - float(row["error"]) for row in selected)
            if selected
            else 0.0
        )
    comparisons.append(
        _bootstrap_interval(
            deletion_values,
            comparison_id="prospective_error_increase_after_template_deletion",
            draws=draws,
        )
    )
    return comparisons


def score_acceptance_gate(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    violation_count: int,
) -> JsonDict:
    """Apply the fixed primary learning gate and independent deployment gate."""

    by_id = {str(row["comparison_id"]): row for row in comparisons}

    def criterion(name: str, observed: Any, relation: str, threshold: Any) -> JsonDict:
        if relation == "<":
            passed = observed < threshold
        elif relation == "<=":
            passed = observed <= threshold
        elif relation == ">":
            passed = observed > threshold
        elif relation == ">=":
            passed = observed >= threshold
        else:
            passed = observed == threshold
        return {
            "criterion": name,
            "observed": observed,
            "relation": relation,
            "threshold": threshold,
            "passed": passed,
        }

    primary = [
        criterion(
            "future_error_upper_vs_warmup_below_zero",
            by_id["future_error_change_vs_warmup_frozen"]["ci95_upper"],
            "<",
            0.0,
        ),
        criterion(
            "future_error_upper_vs_random_below_zero",
            by_id["future_error_change_vs_random_query_committed"]["ci95_upper"],
            "<",
            0.0,
        ),
        criterion(
            "false_accept_upper_vs_warmup_nonpositive",
            by_id["false_accept_change_vs_warmup_frozen"]["ci95_upper"],
            "<=",
            0.0,
        ),
        criterion(
            "recurrence_error_upper_increase_bounded",
            by_id["recurrence_error_change_vs_warmup_frozen"]["ci95_upper"],
            "<=",
            0.02,
        ),
        criterion("capacity_and_validation_violations", violation_count, "==", 0),
        criterion(
            "template_deletion_strictly_increases_error",
            by_id["prospective_error_increase_after_template_deletion"]["estimate"],
            ">",
            0.0,
        ),
    ]
    secondary = [
        criterion(
            "compiled_accuracy_noninferiority_lower",
            by_id["compiled_accuracy_change_vs_version_space"]["ci95_lower"],
            ">=",
            -0.02,
        ),
        criterion(
            "compiled_amortized_throughput_ratio",
            by_id["compiled_throughput_ratio_vs_version_space"]["estimate"],
            ">=",
            2.0,
        ),
    ]
    return {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": int(comparisons[0]["bootstrap_draws"]),
        "primary_criteria": primary,
        "primary_learning_gate_passed": all(row["passed"] for row in primary),
        "secondary_deployment_criteria": secondary,
        "secondary_compiled_deployment_gate_passed": all(row["passed"] for row in secondary),
        "secondary_can_rescue_primary": False,
        "nfr_01_satisfied_by_secondary": False,
        "version_space_superiority_claimed": False,
    }


def _latency_summary(panel: LearningPanel) -> JsonDict:
    """Reduce every measured CPU operation without hiding total acquisition cost."""

    summary: JsonDict = {}
    for name, values in panel.latency_values.items():
        summary[name.removesuffix("_ns")] = {
            "operation_count": len(values),
            "p50_ns": int(_percentile(values, 0.50)),
            "p95_ns": int(_percentile(values, 0.95)),
            "total_ns": int(sum(values)),
        }
    summary["max_memory_bytes"] = max(int(row["max_memory_bytes"]) for row in panel.rows)
    summary["old_update_target_ns"] = 1_000
    summary["old_update_target_claimed_met"] = (
        summary["update"]["p95_ns"] < summary["old_update_target_ns"]
    )
    summary["cost_scope"] = [
        "query_selection",
        "queue_storage",
        "fitting",
        "validation",
        "transactional_commit",
        "compiled_lookup",
    ]
    return summary


def _future_hardware_path() -> JsonDict:
    """Separate measured host kernels from optional later acceleration."""

    return {
        "measured_now": [
            "CPU integer counters",
            "CPU finite-domain bitset-equivalent set intersections",
            "CPU direct compiled predicate dispatch",
            "CPU transactional JSON memory",
        ],
        "optional_after_usefulness": [
            "FPGA committed-template matching",
            "FPGA packed bitset intersection",
        ],
        "fpga_measured": False,
        "fpga_speedup_claimed": False,
        "deployment_condition": "Only after measured CPU usefulness passes its own gate.",
    }


def _stable_value(value: Any) -> Any:
    """Remove host timing while retaining all scientific choices and outcomes."""

    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in value.items()
            if not str(key).endswith("_ns")
            and key not in {"duration_s", "execution_host", "checkpoint_hash"}
        }
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    return value


def raw_rows_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all raw row bytes, including measured host timing."""

    return transactional.sha256_json(
        {
            "rows": artifact.get("rows", []),
            "decision_rows": artifact.get("decision_rows", []),
            "query_rows": artifact.get("query_rows", []),
            "commit_deletion_rows": artifact.get("commit_deletion_rows", []),
        }
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash sources, settings, gates, and timing-free raw scientific rows."""

    fields = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "random_seed",
        "gate_check_summary",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
        "refinement_run_complete_score",
        "refinement_value_score",
        "continuous_self_learning_task",
        "no_model_weight_mutation",
        "acceptance_gate_learning",
        "decision_rows",
        "query_rows",
        "commit_deletion_rows",
        "comparison_rows",
        "panel_conformance_errors",
        "upstream_receipt",
        "future_hardware_path",
    )
    return transactional.sha256_json(_stable_value({name: artifact.get(name) for name in fields}))


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    *,
    source_hashes: Mapping[str, Any],
    paths: ExperimentPaths,
    duration_s: float,
) -> JsonDict:
    """Build a schema-complete object before terminal classification."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "blocked",
        "run_date": RUN_DATE,
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": "no qualifying learning computation ran after an external gate failed",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": socket.gethostname(),
        "duration_s": duration_s,
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": {
            "planned_events_per_arm": len(STREAM_SEEDS) * EVENTS_PER_SEED,
            "planned_decisions": len(STREAM_SEEDS) * EVENTS_PER_SEED * len(ARMS),
            "attempted_decisions": 0,
            "completed_decisions": 0,
            "censored_decisions": len(STREAM_SEEDS) * EVENTS_PER_SEED * len(ARMS),
            "independent_units_planned": len(STREAM_SEEDS),
            "independent_units_attempted": 0,
            "independent_units_completed": 0,
            "independent_units_censored": len(STREAM_SEEDS),
            "bootstrap_resamples_planned": BOOTSTRAP_DRAWS,
            "bootstrap_resamples_completed": 0,
        },
        "random_seed": {
            "bootstrap": BOOTSTRAP_SEED,
            "stream_seeds": list(STREAM_SEEDS),
            "random_query_derivation": "sha256(exp7213-random-query,seed,event_id)",
        },
        "reproducibility_checksum": None,
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_external:unknown_precondition",
        "refinement_run_complete_score": 0,
        "refinement_value_score": 0,
        "continuous_self_learning_task": True,
        "no_model_weight_mutation": True,
        "acceptance_gate_learning": {},
        "decision_rows": [],
        "query_rows": [],
        "commit_deletion_rows": [],
        "checkpoint_path": str(paths.checkpoint),
        "checkpoint_hash": None,
        "latency_summary": {},
        "future_hardware_path": _future_hardware_path(),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "comparison_rows": [],
        "panel_conformance_errors": [],
        "upstream_receipt": {
            "artifact_path": str(DEFAULT_UPSTREAM_ARTIFACT),
            "artifact_hash": source_hashes.get(str(DEFAULT_UPSTREAM_ARTIFACT)),
            "refinement_fixture_ready_score": None,
            "stream_hashes_authenticated": False,
            "known_v634_null_promoted": False,
        },
        "raw_rows_checksum": None,
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    *,
    source_hashes: Mapping[str, Any],
    paths: ExperimentPaths,
    duration_s: float,
) -> JsonDict:
    """Return row-free terminal evidence for an unchanged external block."""

    artifact = _base_artifact(
        checks,
        source_hashes=source_hashes,
        paths=paths,
        duration_s=duration_s,
    )
    failed = artifact["gate_check_summary"]["failed_check"] or "unknown_precondition"
    artifact["honest_verdict"] = f"blocked_external:{failed}"
    artifact["raw_rows_checksum"] = raw_rows_checksum(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    check_files: bool = False,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    """Cold-check fields, raw rows, gates, sources, and terminal class."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return ["missing_fields:" + ",".join(missing)]
    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS), "field_principles")
    add(artifact["schema"] != SCHEMA, "schema")
    add(artifact["experiment_id"] != EXPERIMENT_ID, "experiment_id")
    add(artifact["milestone"] != MILESTONE, "milestone")
    add(artifact["run_date"] != RUN_DATE, "run_date")
    add(artifact["execution_venue"] != EXECUTION_VENUE, "execution_venue")
    add(not artifact["execution_host"], "execution_host")
    add(artifact["MODEL_SPECS"] != [], "model_specs")
    add(artifact["model_invoked"] is not False, "model_invoked")
    add(artifact["continuous_self_learning_task"] is not True, "continuous_learning")
    add(artifact["no_model_weight_mutation"] is not True, "weight_mutation")
    add(artifact["verifier_is_oracle"] is not True, "oracle_classification")
    blocked = artifact["verdict_class"] == "blocked"
    if blocked:
        add(artifact["status"] != "blocked", "blocked_status")
        add(artifact["inference_substrate_class"] != "blocked_no_run", "blocked_substrate")
        add(artifact["refinement_run_complete_score"] != 0, "blocked_complete_score")
        add(artifact["refinement_value_score"] != 0, "blocked_value_score")
        add(bool(artifact["rows"]), "blocked_rows")
        add(bool(artifact["decision_rows"]), "blocked_decision_rows")
        add(bool(artifact["query_rows"]), "blocked_query_rows")
        add(bool(artifact["commit_deletion_rows"]), "blocked_deletion_rows")
        summary = artifact["gate_check_summary"]
        add(summary.get("passed") is not False, "blocked_gate_passed")
        for name in ("failed_check", "upstream", "field", "expected_value", "observed_value"):
            add(summary.get(name) is None, f"blocked_gate_{name}")
    else:
        add(artifact["status"] != "complete", "status")
        add(
            artifact["inference_substrate"] != INFERENCE_SUBSTRATE,
            "inference_substrate",
        )
        add(
            artifact["inference_substrate_class"] != INFERENCE_SUBSTRATE_CLASS,
            "inference_substrate_class",
        )
        add(artifact["gate_check_summary"].get("passed") is not True, "preconditions")
        add(artifact["refinement_run_complete_score"] != 1, "complete_score")
        add(len(artifact["rows"]) != len(STREAM_SEEDS) * len(ARMS), "row_count")
        add(
            len(artifact["decision_rows"]) != len(STREAM_SEEDS) * len(ARMS) * EVENTS_PER_SEED,
            "decision_row_count",
        )
        add(
            any(
                not {"unit_id", "arm", "seed", "metric", "error", "abstention"} <= set(row)
                for row in artifact["rows"]
            ),
            "row_schema",
        )
        add(bool(artifact["panel_conformance_errors"]), "panel_conformance")
        gate = artifact["acceptance_gate_learning"]
        expected_value = int(gate.get("primary_learning_gate_passed") is True)
        add(artifact["refinement_value_score"] != expected_value, "value_score")
        expected_class = "circular_positive" if expected_value else "null"
        add(artifact["verdict_class"] != expected_class, "verdict_class")
        add(
            not str(artifact["honest_verdict"]).startswith("complete"),
            "honest_verdict",
        )
        add(
            artifact["sample_size_budget"].get("bootstrap_resamples_completed") != BOOTSTRAP_DRAWS,
            "bootstrap_draws",
        )
        if check_files:
            add(
                _sha256_path(_resolve(repo_root, artifact["checkpoint_path"]))
                != artifact["checkpoint_hash"],
                "checkpoint_hash",
            )
            for path_text, expected_hash in artifact["source_artifact_hashes"].items():
                add(
                    _sha256_path(_resolve(repo_root, path_text)) != expected_hash,
                    "source_hashes",
                )
    add(artifact["raw_rows_checksum"] != raw_rows_checksum(artifact), "raw_rows_checksum")
    add(
        artifact["reproducibility_checksum"] != reproducibility_checksum(artifact),
        "reproducibility_checksum",
    )
    return errors


def _require_valid(errors: Sequence[str]) -> None:
    """Stop publication when any cold artifact check fails."""

    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    upstream_artifact: Path = DEFAULT_UPSTREAM_ARTIFACT,
    duration_s: float | None = None,
    progress: bool = False,
) -> JsonDict:
    """Run gates, matched learning, causal replay, bootstrap, and cold checks."""

    started = time.monotonic()
    if progress:
        print(
            "PHASE 0 CHECK: verify spec, sources, tools, fixture fields, hashes, and quarantine",
            flush=True,
        )
    checks, fixture, source_hashes = collect_preconditions(
        repo_root,
        paths,
        upstream_artifact=upstream_artifact,
    )
    if not all(row["passed"] for row in checks):
        elapsed = time.monotonic() - started if duration_s is None else duration_s
        if progress:
            print(
                "PHASE 0 END: external prerequisite failed; no learning replay ran",
                flush=True,
            )
        return build_blocked_artifact(
            checks,
            source_hashes=source_hashes,
            paths=paths,
            duration_s=elapsed,
        )
    if progress:
        print("PHASE 0 END: exact Exp7212 gates and frozen hashes passed", flush=True)
        print("PHASE 1 START: load separated public and authority fixture views", flush=True)
    views = load_fixture_views(repo_root, upstream_artifact)
    if progress:
        print("PHASE 1 END: frozen views loaded without changing their bytes", flush=True)
        print("PHASE 2 START: initialize 20 independent matched stream states", flush=True)
    with tempfile.TemporaryDirectory(prefix="carnot-exp7213-state-") as temporary:
        if progress:
            print("PHASE 2 END: isolated transactional state roots are ready", flush=True)
            print("PHASE 3 START: run five-arm CPU refinement benchmark", flush=True)
        panel = run_learning_panel(
            views,
            state_root=Path(temporary),
            progress=progress,
        )
    if progress:
        print("PHASE 3 END: all 102400 predictions and charged releases completed", flush=True)
        print("PHASE 4 START: cold-check rows and run 10000 paired stream resamples", flush=True)
    conformance = panel_conformance_errors(panel)
    comparisons = build_comparison_rows(panel.rows, panel.commit_deletion_rows)
    acceptance = score_acceptance_gate(comparisons, violation_count=len(conformance))
    if progress:
        print("PHASE 4 END: paired learning and deployment gates are fixed", flush=True)
        print("PHASE 5 START: persist reloadable full state under results/checkpoints", flush=True)
    checkpoint = {
        "schema": "carnot.exp7213.learning_checkpoint.v1",
        "run_date": RUN_DATE,
        "completed_streams": list(STREAM_SEEDS),
        "state_rows": panel.state_rows,
        "decision_row_count": len(panel.decision_rows),
        "query_row_count": len(panel.query_rows),
        "raw_science_hash": transactional.sha256_json(
            _stable_value(
                {
                    "rows": panel.rows,
                    "decisions": panel.decision_rows,
                    "queries": panel.query_rows,
                    "deletions": panel.commit_deletion_rows,
                }
            )
        ),
    }
    _atomic_write(paths.checkpoint, transactional.canonical_json_bytes(checkpoint))
    source_hashes[str(paths.checkpoint)] = _sha256_path(paths.checkpoint)
    elapsed = time.monotonic() - started if duration_s is None else duration_s
    artifact = _base_artifact(
        checks,
        source_hashes=source_hashes,
        paths=paths,
        duration_s=elapsed,
    )
    value_score = int(acceptance["primary_learning_gate_passed"])
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "rows": panel.rows,
            "sample_size_budget": {
                "planned_events_per_arm": len(STREAM_SEEDS) * EVENTS_PER_SEED,
                "attempted_decisions": len(panel.decision_rows),
                "completed_decisions": len(panel.decision_rows),
                "censored_decisions": 0,
                "independent_units_planned": len(STREAM_SEEDS),
                "independent_units_attempted": len(STREAM_SEEDS),
                "independent_units_completed": len(STREAM_SEEDS),
                "independent_units_censored": 0,
                "bootstrap_resamples_planned": BOOTSTRAP_DRAWS,
                "bootstrap_resamples_completed": BOOTSTRAP_DRAWS,
                "query_rows_completed": len(panel.query_rows),
                "template_deletion_rows_completed": len(panel.commit_deletion_rows),
            },
            "verdict_class": "circular_positive" if value_score else "null",
            "honest_verdict": (
                "complete: witnessed committed predicates passed the fixed primary learning gate"
                if value_score
                else "complete_null: witnessed committed predicates did not pass the fixed primary learning gate"
            ),
            "refinement_run_complete_score": 1,
            "refinement_value_score": value_score,
            "acceptance_gate_learning": acceptance,
            "decision_rows": panel.decision_rows,
            "query_rows": panel.query_rows,
            "commit_deletion_rows": panel.commit_deletion_rows,
            "checkpoint_hash": _sha256_path(paths.checkpoint),
            "latency_summary": _latency_summary(panel),
            "comparison_rows": comparisons,
            "panel_conformance_errors": conformance,
            "upstream_receipt": {
                "artifact_path": str(upstream_artifact),
                "artifact_hash": source_hashes[str(upstream_artifact)],
                "expected_artifact_hash": EXPECTED_UPSTREAM_SHA256,
                "refinement_fixture_ready_score": unwrap_principled(
                    fixture.get("refinement_fixture_ready_score")
                ),
                "stream_hashes_authenticated": True,
                "known_v634_null_promoted": False,
            },
        }
    )
    artifact["raw_rows_checksum"] = raw_rows_checksum(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact, check_files=True, repo_root=repo_root)
    _require_valid(errors)
    if progress:
        print("PHASE 5 END: checkpoint and complete artifact object passed cold checks", flush=True)
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed execution date and explicit private test paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--upstream-artifact", type=Path, default=DEFAULT_UPSTREAM_ARTIFACT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the real gates and atomically publish one terminal artifact."""

    print("PHASE 0 START: parse fixed execution inputs before all checks", flush=True)
    args = _parse_args(argv)
    if str(args.date) != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    artifact = build_and_seal(
        REPO_ROOT,
        paths,
        upstream_artifact=args.upstream_artifact,
        progress=True,
    )
    print("PHASE 6 START: cold-check the terminal object before publication", flush=True)
    errors = validate_artifact(
        artifact,
        check_files=artifact["verdict_class"] != "blocked",
        repo_root=REPO_ROOT,
    )
    _require_valid(errors)
    print("PHASE 6 END: terminal object passed cold validation", flush=True)
    print("FINAL ATOMIC WRITE START", flush=True)
    _atomic_write(paths.artifact, transactional.canonical_json_bytes(artifact))
    print("FINAL ATOMIC WRITE END", flush=True)
    print("PHASE 7 END: terminal deliverable is stable", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper owns execution.
    raise SystemExit(main())
