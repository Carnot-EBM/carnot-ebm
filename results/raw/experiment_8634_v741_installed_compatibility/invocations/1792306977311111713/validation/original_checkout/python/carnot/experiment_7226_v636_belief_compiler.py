"""Compile a finite version space into lossless packed belief state.

The compiler stores every surviving hypothesis, not one chosen predicate.
Predictions read frozen bytes. Released feedback changes state only at an
explicit transaction boundary after the query cycle ends.

Spec refs: REQ-CL-7226 and SCENARIO-CL-7226-*.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_7198_v634_feedback_capacity_stream as exp7198
from carnot import experiment_7199_v634_bounded_acquisition as exp7199
from carnot import experiment_7212_v635_refinement_fixture as exp7212
from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7226
SCHEMA = "carnot.exp7226.v636_belief_compiler.v1"
STATE_SCHEMA = "carnot.packed_belief_state.v1"
MILESTONE = "2026.09.636"
RUN_DATE = "20260911"
RANDOM_SEED = 7_226_000
STREAM_SEEDS = tuple(range(7_226_001, 7_226_021))
REPLAY_SEEDS = tuple(range(7_226_101, 7_226_109))
EVENTS_PER_SEED = 1_024
WARMUP_COUNT = 128
PENDING_CAPACITY = 4
QUERY_CEILING = 128
BLOCK_SIZE = 4
DELAY_SUPPORT = (0, 4, 16, 32)
FAMILIES = tuple(exp7198.FAMILIES)
PARAMETER_DOMAIN = tuple(exp7198.PARAMETER_DOMAIN)
FULL_MASK = (1 << len(PARAMETER_DOMAIN)) - 1
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"
EXPECTED_UPSTREAM_SHA256 = "sha256:62e68c63b8fc36179ebdc8dca46e102ee52f784cf770d561a39aa4d66595e265"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STREAM_ROOT = Path("results/streams/experiment_7226")
DEFAULT_CHECKPOINT_PATH = Path(
    "results/checkpoints/experiment_7226_v636_belief_compiler_state.json"
)
DEFAULT_ARTIFACT_PATH = Path("results/experiment_7226_v636_belief_compiler.json")
DEFAULT_UPSTREAM_ARTIFACT = Path("results/experiment_7213_v635_refinement_learning.json")
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
    Path("python/carnot/experiment_7212_v635_refinement_fixture.py"),
    Path("python/carnot/experiment_7213_v635_refinement_learning.py"),
    Path("python/carnot/experiment_7226_v636_belief_compiler.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("scripts/experiments/experiment_7226_v636_belief_compiler.py"),
    Path("tests/python/test_experiment_7226_v636_belief_compiler.py"),
    SPEC_PATH,
)

FORBIDDEN_AUTHORITY_FIELDS = {
    "audit_label",
    "change_time",
    "exact_label",
    "future_label",
    "hidden_parameter",
    "hidden_seed",
    "observed_label",
    "poisoned",
    "regime",
    "stable_parameter",
    "shifted_parameter",
    "target_label",
    "target_parameter",
}

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
    "belief_compiler_ready_score",
    "stream_manifest_path",
    "compiler_state_path",
    "parity_rows",
    "mutation_rows",
    "hypothesis_domain_contract",
    "future_hardware_path",
    "stream_conformance_errors",
    "compiler_claim_scope",
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
    "run_date": "Use 20260911 and record actual UTC timestamps, never copy an upstream run date.",
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
    "MODEL_SPECS": "Only models actually invoked; [] for this CPU compiler task.",
    "model_invoked": "True only for actual model execution; upstream model outputs are cached evidence.",
    "belief_compiler_ready_score": "Exact supported-domain semantics and fresh sealed stream ready.",
    "stream_manifest_path": "Public, authority and release-manifest paths with hashes.",
    "compiler_state_path": "Serializable state, version and parent hash.",
    "parity_rows": "Prediction/query/energy/update parity per tested state.",
    "mutation_rows": "Future labels and invalid commits fail at the correct boundary.",
    "hypothesis_domain_contract": "Finite supported families and resource limits.",
    "future_hardware_path": "CPU bit operations; optional Rust SIMD/FPGA table path, with sizes.",
    "stream_conformance_errors": "An empty list proves split and information checks passed.",
    "compiler_claim_scope": "Readiness does not imply learning benefit, speed, or paper reproduction.",
}

exact_label = exp7198.exact_label
independent_exact_label = exp7198.independent_exact_label
unwrap_principled = exp7213.unwrap_principled
gate_check = exp7213.gate_check
gate_summary = exp7213.gate_summary
ImmutableSealError = exp7212.ImmutableSealError
write_immutable = exp7212.write_immutable


def _accept_masks() -> dict[str, tuple[int, ...]]:
    """Compile each predicate table once so prediction only counts bits."""

    return {
        family: tuple(
            sum(
                1 << parameter
                for parameter in PARAMETER_DOMAIN
                if exact_label(family, value, parameter) == "accept"
            )
            for value in PARAMETER_DOMAIN
        )
        for family in FAMILIES
    }


ACCEPT_MASKS = _accept_masks()


class CommitRejected(ValueError):
    """Report a transaction that failed before any state byte changed."""


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep sealed streams, compiler state, and the terminal result separate."""

    public_stream: Path
    authority_sidecar: Path
    release_manifest: Path
    stream_manifest: Path
    compiler_state: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the task-owned repository destinations."""

        return cls.from_root(REPO_ROOT / "results")

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put every test output below one caller-owned directory."""

        return cls.from_root(root)

    @classmethod
    def from_root(cls, root: Path) -> ExperimentPaths:
        """Derive every output without a hard-coded checkout path."""

        stream_root = root / "streams" / "experiment_7226"
        return cls(
            stream_root / "public_stream.jsonl",
            stream_root / "evaluator_sidecar.jsonl",
            stream_root / "release_manifest.jsonl",
            stream_root / "public_stream_manifest.json",
            root / "checkpoints" / DEFAULT_CHECKPOINT_PATH.name,
            root / DEFAULT_ARTIFACT_PATH.name,
        )


@dataclass(frozen=True)
class StreamViews:
    """Keep public, scheduling, and correctness authority bytes distinct."""

    public: list[JsonDict]
    authority: list[JsonDict]
    releases: list[JsonDict]
    manifest: JsonDict


def _votes_for(family: str, survivor_mask: int) -> list[int]:
    """Count accept votes for all residues from one packed survivor mask."""

    return [(survivor_mask & mask).bit_count() for mask in ACCEPT_MASKS[family]]


def _new_family_state(survivor_mask: int = FULL_MASK) -> JsonDict:
    """Create one complete family state without a target parameter."""

    return {
        "survivor_mask": survivor_mask,
        "vote_counts": [],
        "epoch": 0,
        "provenance": [],
    }


class PackedBeliefController:
    """Keep every finite hypothesis in masks with exact cached vote counts."""

    def __init__(self) -> None:
        self._state: JsonDict = {
            "schema": STATE_SCHEMA,
            "version": 0,
            "parent_hash": None,
            "families": {family: _new_family_state() for family in FAMILIES},
        }
        self._refresh_votes()

    @classmethod
    def from_survivors(cls, survivors: Mapping[str, set[int]]) -> PackedBeliefController:
        """Build an audit state from explicit surviving hypothesis identities."""

        controller = cls()
        for family, parameters in survivors.items():
            if family not in FAMILIES or any(
                parameter not in PARAMETER_DOMAIN for parameter in parameters
            ):
                raise ValueError("invalid_survivor_subset")
            controller._state["families"][family]["survivor_mask"] = sum(
                1 << parameter for parameter in parameters
            )
        controller._refresh_votes()
        return controller

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> PackedBeliefController:
        """Load state only after its schema, masks, and vote cache agree."""

        if value.get("schema") != STATE_SCHEMA:
            raise ValueError("invalid_state_schema")
        if set(value.get("families", {})) != set(FAMILIES):
            raise ValueError("invalid_state_families")
        version = value.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 0:
            raise ValueError("invalid_state_version")
        parent_hash = value.get("parent_hash")
        if (
            parent_hash is not None
            and re.fullmatch(r"sha256:[0-9a-f]{64}", str(parent_hash)) is None
        ):
            raise ValueError("invalid_parent_hash")
        state = deepcopy(dict(value))
        for family in FAMILIES:
            row = state["families"][family]
            mask = row.get("survivor_mask") if isinstance(row, Mapping) else None
            if not isinstance(mask, int) or isinstance(mask, bool) or not 0 <= mask <= FULL_MASK:
                raise ValueError("invalid_survivor_mask")
            if row.get("vote_counts") != _votes_for(family, mask):
                raise ValueError("invalid_vote_cache")
            if not isinstance(row.get("epoch"), int) or int(row["epoch"]) < 0:
                raise ValueError("invalid_epoch")
            if not isinstance(row.get("provenance"), list):
                raise ValueError("invalid_provenance")
        controller = cls.__new__(cls)
        controller._state = state
        return controller

    @classmethod
    def load(cls, path: Path) -> PackedBeliefController:
        """Restore a durable state and reject malformed JSON or cache bytes."""

        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("invalid_state_object")
        return cls.from_state(value)

    def _refresh_votes(self) -> None:
        """Refresh the exact table after an admitted mask change."""

        for family in FAMILIES:
            row = self._state["families"][family]
            row["vote_counts"] = _votes_for(family, int(row["survivor_mask"]))

    def state_dict(self) -> JsonDict:
        """Return detached bytes so callers cannot mutate a live controller."""

        return deepcopy(self._state)

    def state_bytes(self) -> bytes:
        """Use the shared transaction memory canonical JSON encoding."""

        return transactional.canonical_json_bytes(self._state)

    def state_hash(self) -> str:
        """Identify every mask, vote, provenance, version, and parent byte."""

        return transactional.sha256_bytes(self.state_bytes())

    def save(self, path: Path) -> JsonDict:
        """Publish one complete state through the existing atomic writer."""

        return transactional._atomic_write(path, self.state_bytes())

    def family_state(self, family: str) -> JsonDict:
        """Expose a detached family row for audit without write access."""

        return deepcopy(self._state["families"][family])

    def survivors(self, family: str) -> set[int]:
        """Expand mask bits only for reference comparison and audit output."""

        mask = int(self._state["families"][family]["survivor_mask"])
        return {parameter for parameter in PARAMETER_DOMAIN if mask & (1 << parameter)}

    @staticmethod
    def _public_coordinates(public_event: Mapping[str, Any]) -> tuple[str, int] | None:
        """Normalize a supported finite input without reading authority fields."""

        family = public_event.get("family_id")
        value = public_event.get("numeric_value")
        if family not in FAMILIES or not isinstance(value, int) or isinstance(value, bool):
            return None
        return str(family), value % len(PARAMETER_DOMAIN)

    def predict(self, public_event: Mapping[str, Any]) -> tuple[str, float]:
        """Return the shipped majority decision and disagreement without writes."""

        coordinates = self._public_coordinates(public_event)
        if coordinates is None:
            return "abstain", 0.0
        family, value = coordinates
        row = self._state["families"][family]
        count = int(row["survivor_mask"]).bit_count()
        if count == 0:
            return "abstain", 0.0
        accepts = int(row["vote_counts"][value])
        disagreement = min(accepts, count - accepts) / count
        return ("accept" if accepts > count / 2 else "reject"), disagreement

    def energy(self, label: str, public_event: Mapping[str, Any]) -> JsonDict:
        """Return disagreement energy, never a claim that a label is correct."""

        if label not in {"accept", "reject"}:
            return {
                "status": "unknown_label",
                "value": None,
                "survivor_count": None,
                "disagree_count": None,
            }
        coordinates = self._public_coordinates(public_event)
        if coordinates is None:
            return {
                "status": "unknown_input",
                "value": None,
                "survivor_count": None,
                "disagree_count": None,
            }
        family, value = coordinates
        row = self._state["families"][family]
        count = int(row["survivor_mask"]).bit_count()
        if count == 0:
            return {
                "status": "empty",
                "value": None,
                "survivor_count": 0,
                "disagree_count": None,
            }
        accepts = int(row["vote_counts"][value])
        disagrees = count - accepts if label == "accept" else accepts
        return {
            "status": "known",
            "value": disagrees / count,
            "survivor_count": count,
            "disagree_count": disagrees,
        }

    def select_request(
        self,
        block: Sequence[Mapping[str, Any]],
        tie_ranks: Mapping[str, int],
    ) -> Mapping[str, Any]:
        """Use the shipped disagreement selector through its public interface."""

        return exp7199.select_request(block, "priority_admission", tie_ranks, self)

    @staticmethod
    def _validate_release(release: Mapping[str, Any], current_cycle: int) -> JsonDict:
        """Reject malformed, hidden, duplicate-later, or premature feedback."""

        if set(release) & (FORBIDDEN_AUTHORITY_FIELDS - {"observed_label"}):
            raise CommitRejected("authority_field")
        coordinates = PackedBeliefController._public_coordinates(release)
        if coordinates is None or not release.get("event_id"):
            raise CommitRejected("invalid_public_event")
        label = release.get("observed_label")
        if label not in {"accept", "reject"}:
            raise CommitRejected("invalid_label")
        role = release.get("role")
        if role not in {"support", "validation"}:
            raise CommitRejected("invalid_role")
        request_index = release.get("request_index")
        release_index = release.get("release_index")
        if not isinstance(request_index, int) or not isinstance(release_index, int):
            raise CommitRejected("invalid_release_index")
        if release_index < request_index:
            raise CommitRejected("release_before_request")
        if release_index > current_cycle:
            raise CommitRejected("future_release")
        family, value = coordinates
        return {
            "event_id": str(release["event_id"]),
            "family_id": family,
            "numeric_value": value,
            "observed_label": str(label),
            "role": str(role),
            "request_index": request_index,
            "release_index": release_index,
        }

    def commit_batch(
        self,
        releases: Sequence[Mapping[str, Any]],
        *,
        current_cycle: int,
        expected_parent_hash: str,
        state_path: Path | None = None,
    ) -> JsonDict:
        """Apply one ordered due batch after a query cycle and publish atomically."""

        parent_hash = self.state_hash()
        if expected_parent_hash != parent_hash:
            raise CommitRejected("stale_parent")
        try:
            self.from_state(self._state)
        except ValueError as error:
            raise CommitRejected("corrupt_live_state") from error
        if state_path is not None and state_path.exists():
            try:
                durable_hash = self.load(state_path).state_hash()
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise CommitRejected("corrupt_durable_state") from error
            if durable_hash != parent_hash:
                raise CommitRejected("stale_durable_parent")
        normalized = [self._validate_release(row, current_cycle) for row in releases]
        existing_ids = {
            str(item["event_id"])
            for family in FAMILIES
            for item in self._state["families"][family]["provenance"]
        }
        batch_ids = [str(row["event_id"]) for row in normalized]
        if len(set(batch_ids)) != len(batch_ids) or existing_ids.intersection(batch_ids):
            raise CommitRejected("duplicate_release")
        candidate = deepcopy(self._state)
        candidate["version"] = int(candidate["version"]) + 1
        candidate["parent_hash"] = parent_hash
        operations: list[JsonDict] = []
        for release in normalized:
            family = str(release["family_id"])
            row = candidate["families"][family]
            before_mask = int(row["survivor_mask"])
            after_mask = before_mask
            reset = False
            if release["role"] == "support":
                accept_mask = ACCEPT_MASKS[family][int(release["numeric_value"])]
                matching = (
                    accept_mask
                    if release["observed_label"] == "accept"
                    else FULL_MASK ^ accept_mask
                )
                after_mask = before_mask & matching
                if after_mask == 0:
                    after_mask = FULL_MASK & matching
                    row["epoch"] = int(row["epoch"]) + 1
                    reset = True
                row["survivor_mask"] = after_mask
            row["provenance"].append(deepcopy(release))
            operations.append(
                {
                    "event_id": release["event_id"],
                    "family_id": family,
                    "role": release["role"],
                    "survivor_mask_before": before_mask,
                    "survivor_mask_after": after_mask,
                    "empty_reset": reset,
                    "validation_used_for_elimination": False,
                }
            )
        for family in FAMILIES:
            row = candidate["families"][family]
            row["vote_counts"] = _votes_for(family, int(row["survivor_mask"]))
        admitted = type(self).from_state(candidate)
        new_bytes = admitted.state_bytes()
        receipt = {
            "parent_hash": parent_hash,
            "new_state_hash": transactional.sha256_bytes(new_bytes),
            "parent_bytes_b64": transactional.encode_bytes(self.state_bytes()),
            "new_state_bytes_b64": transactional.encode_bytes(new_bytes),
            "state_version": candidate["version"],
            "release_count": len(normalized),
            "release_order": batch_ids,
            "operations": operations,
            "atomic_write": None,
        }
        if state_path is not None:
            receipt["atomic_write"] = transactional._atomic_write(state_path, new_bytes)
        self._state = admitted._state
        return receipt

    def rollback(self, receipt: Mapping[str, Any], *, state_path: Path | None = None) -> JsonDict:
        """Restore an admitted transaction only from its exact child state."""

        if self.state_hash() != receipt.get("new_state_hash"):
            raise CommitRejected("stale_rollback")
        try:
            parent_bytes = transactional.decode_bytes(str(receipt["parent_bytes_b64"]))
            parent_value = json.loads(parent_bytes)
            parent = type(self).from_state(parent_value)
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            raise CommitRejected("corrupt_rollback_receipt") from error
        if parent.state_hash() != receipt.get("parent_hash"):
            raise CommitRejected("corrupt_rollback_parent")
        atomic = None
        if state_path is not None:
            atomic = transactional._atomic_write(state_path, parent_bytes)
        self._state = parent._state
        return {
            "parent_hash": receipt["parent_hash"],
            "restored_hash": self.state_hash(),
            "byte_identical": self.state_bytes() == parent_bytes,
            "atomic_write": atomic,
        }


def _resolve(repo_root: Path, path: Path | str) -> Path:
    """Resolve evidence paths without changing absolute caller paths."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _sha256_path(path: Path) -> str | None:
    """Hash one file in bounded chunks and preserve a missing identity."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating evidence bytes."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _load_object(path: Path) -> JsonDict:
    """Decode one JSON object while malformed evidence remains a failed gate."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _task_identity(text: str) -> JsonDict:
    """Extract only the Exp7226 block from the executable roadmap."""

    match = re.search(r"(?ms)^- id: exp7226-belief-compiler\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(0)
    return {
        "id": "exp7226-belief-compiler" if block else None,
        "milestone": MILESTONE if f"milestone: {MILESTONE}" in block else None,
        "deliverable": (
            str(DEFAULT_ARTIFACT_PATH) if f"deliverable: {DEFAULT_ARTIFACT_PATH}" in block else None
        ),
    }


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    upstream_artifact: Path = DEFAULT_UPSTREAM_ARTIFACT,
) -> tuple[list[JsonDict], JsonDict, dict[str, str | None]]:
    """Authenticate the diagnostic artifact before any new stream is generated."""

    root = Path(repo_root)
    upstream_path = _resolve(root, upstream_artifact)
    upstream = _load_object(upstream_path)
    source_hashes: dict[str, str | None] = {
        str(path): _sha256_path(root / path) for path in SOURCE_PATHS
    }
    source_hashes[str(upstream_artifact)] = _sha256_path(upstream_path)
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    roadmap_text = (root / "research-roadmap.yaml").read_text(encoding="utf-8")
    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    quarantine = exp7213.quarantine_state(
        upstream,
        exclusion_text,
        DEFAULT_UPSTREAM_ARTIFACT.name,
        "exp7213-refinement-learning",
    )
    imports = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "carnot.experiment_7199_v634_bounded_acquisition",
            "carnot.experiment_7212_v635_refinement_fixture",
            "carnot.experiment_7213_v635_refinement_learning",
            "carnot.memory.transactional_constraint_memory",
        )
    }
    tools = {
        "python": Path(sys.executable).is_file(),
        "jq": shutil.which("jq") is not None,
        "sha256sum": shutil.which("sha256sum") is not None,
    }
    destinations = {
        name: _path_writable(path)
        for name, path in (
            ("public_stream", paths.public_stream),
            ("authority_sidecar", paths.authority_sidecar),
            ("release_manifest", paths.release_manifest),
            ("stream_manifest", paths.stream_manifest),
            ("compiler_state", paths.compiler_state),
            ("artifact", paths.artifact),
        )
    }
    comparison = next(
        (
            row
            for row in upstream.get("comparison_rows", [])
            if row.get("comparison_id") == "future_error_change_vs_version_space"
        ),
        {},
    )
    upstream_gate = unwrap_principled(upstream.get("gate_check_summary", {}))
    checksum_valid = bool(upstream) and upstream.get(
        "reproducibility_checksum"
    ) == exp7213.reproducibility_checksum(upstream)
    expected_identity = {
        "id": "exp7226-belief-compiler",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT_PATH),
    }
    checks = [
        gate_check(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-CL-7226",
            True,
            "## REQ-CL-7226:" in spec_text,
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7226-*",
            8,
            spec_text.count("### SCENARIO-CL-7226-"),
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            {str(path): "nonempty" for path in SOURCE_PATHS},
            {
                str(path): "nonempty"
                if (root / path).is_file() and (root / path).stat().st_size
                else None
                for path in SOURCE_PATHS
            },
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
            "v636_task_identity",
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
            "raw,checkpoint,artifact",
            {key: True for key in destinations},
            destinations,
        ),
        gate_check(
            "exp7213_artifact_hash",
            "exp7213-refinement-learning",
            str(DEFAULT_UPSTREAM_ARTIFACT),
            EXPECTED_UPSTREAM_SHA256,
            source_hashes[str(upstream_artifact)],
        ),
        gate_check(
            "exp7213_status",
            "exp7213-refinement-learning",
            "status",
            "complete",
            upstream.get("status"),
        ),
        gate_check(
            "exp7213_run_complete",
            "exp7213-refinement-learning",
            "refinement_run_complete_score",
            1,
            upstream.get("refinement_run_complete_score"),
        ),
        gate_check(
            "exp7213_null_retained",
            "exp7213-refinement-learning",
            "refinement_value_score",
            0,
            upstream.get("refinement_value_score"),
        ),
        gate_check(
            "exp7213_diagnostic_gap",
            "exp7213-refinement-learning",
            "future_error_change_vs_version_space.estimate",
            0.09828629032258064,
            comparison.get("estimate"),
        ),
        gate_check(
            "exp7213_gate",
            "exp7213-refinement-learning",
            "gate_check_summary.passed",
            True,
            upstream_gate.get("passed") if isinstance(upstream_gate, Mapping) else None,
        ),
        gate_check(
            "exp7213_no_model",
            "exp7213-refinement-learning",
            "MODEL_SPECS,model_invoked",
            {"MODEL_SPECS": [], "model_invoked": False},
            {
                "MODEL_SPECS": unwrap_principled(upstream.get("MODEL_SPECS")),
                "model_invoked": unwrap_principled(upstream.get("model_invoked")),
            },
        ),
        gate_check(
            "exp7213_not_quarantined",
            "artifact_metadata_and_ops/exclusion_manifest.yaml",
            "quarantined",
            False,
            quarantine["quarantined"],
        ),
        gate_check(
            "exp7213_reproducibility_checksum",
            "exp7213-refinement-learning",
            "reproducibility_checksum",
            True,
            checksum_valid,
        ),
    ]
    return checks, upstream, source_hashes


def _stable_seed(*parts: Any) -> int:
    """Derive local deterministic choices without Python's salted hash."""

    return int(transactional.sha256_json(list(parts)).removeprefix("sha256:")[:16], 16)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode immutable rows with the shared canonical JSON format."""

    return b"".join(transactional.canonical_json_bytes(dict(row)) for row in rows)


def evaluator_worker(paths: ExperimentPaths, *, seeds: Sequence[int] = STREAM_SEEDS) -> int:
    """Generate public and authority views inside the evaluator-only process."""

    print("EVALUATOR PHASE START: freeze fresh public, release, and authority views", flush=True)
    public: list[JsonDict] = []
    authority: list[JsonDict] = []
    releases: list[JsonDict] = []
    for seed_number, seed in enumerate(seeds, start=1):
        family_orders: dict[str, list[str]] = {}
        for window in exp7198.WINDOWS:
            order = list(FAMILIES) * (int(window["count"]) // len(FAMILIES))
            random.Random(_stable_seed("exp7226-family-order", seed, window["name"])).shuffle(order)
            family_orders[str(window["name"])] = order
        offsets = {str(window["name"]): 0 for window in exp7198.WINDOWS}
        for chronology_index in range(EVENTS_PER_SEED):
            window = next(
                row
                for row in exp7198.WINDOWS
                if int(row["start"]) <= chronology_index < int(row["stop"])
            )
            window_name = str(window["name"])
            family = family_orders[window_name][offsets[window_name]]
            offsets[window_name] += 1
            family_index = FAMILIES.index(family)
            value = (seed * 17 + chronology_index * 19 + family_index * 7) % 10_000
            stable, shifted = exp7198._parameters_for(seed, family)
            regime = str(window["regime"])
            parameter = stable if regime in {"stable", "recurrent"} else shifted
            label = exact_label(family, value, parameter)
            independent = independent_exact_label(family, value, parameter)
            poisoned = window_name == "poison_rollback" and exp7198._is_poisoned(
                seed, chronology_index
            )
            observed = ("reject" if label == "accept" else "accept") if poisoned else label
            delay = exp7198._burst_delay(seed, chronology_index)
            event_id = f"exp7226-{seed}-{chronology_index:04d}"
            public.append(
                {
                    "event_id": event_id,
                    "stream_id": f"stream-{seed}",
                    "seed": seed,
                    "chronology_index": chronology_index,
                    "family_id": family,
                    "numeric_value": value,
                    "public_input": f"family={family};value={value}",
                    "public_grammar_id": exp7198.FAMILY_GRAMMAR["grammar_version"],
                    "split": "warmup" if chronology_index < WARMUP_COUNT else "prospective",
                }
            )
            releases.append(
                {
                    "event_id": event_id,
                    "seed": seed,
                    "request_index": chronology_index,
                    "delay": delay,
                    "release_index": chronology_index + delay,
                    "queue_authority_only": True,
                }
            )
            authority.append(
                {
                    "event_id": event_id,
                    "seed": seed,
                    "chronology_index": chronology_index,
                    "family_id": family,
                    "numeric_value": value,
                    "window": window_name,
                    "regime": regime,
                    "change_time": int(window["start"]),
                    "hidden_parameter": parameter,
                    "stable_parameter": stable,
                    "shifted_parameter": shifted,
                    "exact_label": label,
                    "independent_exact_label": independent,
                    "poisoned": poisoned,
                    "observed_label": observed,
                }
            )
        if seed_number % 5 == 0 or seed_number == len(seeds):
            print(
                f"EVALUATOR PROGRESS: completed streams {seed_number}/{len(seeds)}",
                flush=True,
            )
    manifest = {
        "schema": "carnot.exp7226.public_stream_manifest.v1",
        "frozen_before_algorithm_evaluation": True,
        "selection_method": "all_preregistered_seeds_no_outcome_selection",
        "seeds": list(seeds),
        "events_per_seed": EVENTS_PER_SEED,
        "total_events": len(seeds) * EVENTS_PER_SEED,
        "splits": [
            {"name": "warmup", "start": 0, "stop": WARMUP_COUNT},
            {"name": "prospective", "start": WARMUP_COUNT, "stop": EVENTS_PER_SEED},
        ],
        "windows": deepcopy(list(exp7198.WINDOWS)),
        "families": list(FAMILIES),
        "parameter_domain": list(PARAMETER_DOMAIN),
        "pending_capacity": PENDING_CAPACITY,
        "feedback_delay_support": list(DELAY_SUPPORT),
        "query_ceiling_per_stream_arm": QUERY_CEILING,
        "block_size": BLOCK_SIZE,
        "release_rule": "after_query_cycle_when_release_index_is_due_in_request_order",
        "arms": [
            "warmup_frozen",
            "reference_version_space",
            "packed_belief",
            "committed_predicate",
            "packed_feedback_withheld",
        ],
        "authority_fields_available_to_learner": False,
    }
    write_immutable(paths.public_stream, _jsonl_bytes(public))
    write_immutable(paths.authority_sidecar, _jsonl_bytes(authority))
    write_immutable(paths.release_manifest, _jsonl_bytes(releases))
    write_immutable(paths.stream_manifest, transactional.canonical_json_bytes(manifest))
    print("EVALUATOR PHASE END: immutable fresh stream views are sealed", flush=True)
    return 0


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read one chronological JSON object per line."""

    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("jsonl_row_not_object")
            rows.append(value)
    return rows


def load_stream_views(paths: ExperimentPaths) -> StreamViews:
    """Load the three separated views after evaluator completion."""

    return StreamViews(
        _read_jsonl(paths.public_stream),
        _read_jsonl(paths.authority_sidecar),
        _read_jsonl(paths.release_manifest),
        _load_object(paths.stream_manifest),
    )


def _nested_keys(value: Any) -> set[str]:
    """Collect nested keys so a hidden authority field cannot evade checks."""

    if isinstance(value, Mapping):
        return set(map(str, value)) | set().union(
            *(_nested_keys(item) for item in value.values()), set()
        )
    if isinstance(value, list):
        return set().union(*(_nested_keys(item) for item in value), set())
    return set()


def public_leakage_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Name public rows that contain any evaluator-only field."""

    return [
        str(row.get("event_id", f"row-{index}"))
        for index, row in enumerate(rows)
        if _nested_keys(row) & FORBIDDEN_AUTHORITY_FIELDS
    ]


def stream_conformance_errors(
    views: StreamViews, *, expected_seeds: Sequence[int] = STREAM_SEEDS
) -> list[str]:
    """Check counts, chronology, grounding, splits, balance, and separation."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    expected_count = len(expected_seeds) * EVENTS_PER_SEED
    add(len(views.public) != expected_count, "public_event_count")
    add(len(views.authority) != expected_count, "authority_event_count")
    add(len(views.releases) != expected_count, "release_event_count")
    add(bool(public_leakage_errors(views.public)), "public_authority_leak")
    public_ids = [str(row.get("event_id")) for row in views.public]
    authority_ids = [str(row.get("event_id")) for row in views.authority]
    release_ids = [str(row.get("event_id")) for row in views.releases]
    add(public_ids != authority_ids or public_ids != release_ids, "event_identity")
    add(len(set(public_ids)) != expected_count, "event_identity")
    add(views.manifest.get("seeds") != list(expected_seeds), "manifest_seeds")
    add(views.manifest.get("events_per_seed") != EVENTS_PER_SEED, "manifest_event_count")
    add(views.manifest.get("pending_capacity") != PENDING_CAPACITY, "pending_capacity")
    add(views.manifest.get("feedback_delay_support") != list(DELAY_SUPPORT), "delay_support")
    add(views.manifest.get("query_ceiling_per_stream_arm") != QUERY_CEILING, "query_ceiling")
    add(views.manifest.get("frozen_before_algorithm_evaluation") is not True, "split_not_frozen")
    add(
        views.manifest.get("authority_fields_available_to_learner") is not False,
        "authority_contract",
    )
    public_by_id = {str(row["event_id"]): row for row in views.public}
    release_by_id = {str(row["event_id"]): row for row in views.releases}
    for truth in views.authority:
        public = public_by_id.get(str(truth.get("event_id")), {})
        release = release_by_id.get(str(truth.get("event_id")), {})
        add(
            public.get("family_id") != truth.get("family_id")
            or public.get("numeric_value") != truth.get("numeric_value"),
            "public_authority_join",
        )
        try:
            extracted = exp7198.extract_public_input(str(public.get("public_input", "")))
        except ValueError:
            extracted = {}
        add(
            extracted
            != {
                "family_id": public.get("family_id"),
                "numeric_value": public.get("numeric_value"),
            },
            "public_extraction",
        )
        add(
            truth.get("exact_label")
            != independent_exact_label(
                str(truth.get("family_id")),
                int(truth.get("numeric_value", -1)),
                int(truth.get("hidden_parameter", -1)),
            )
            or truth.get("independent_exact_label") != truth.get("exact_label"),
            "independent_grounding",
        )
        add(
            release.get("delay") not in DELAY_SUPPORT
            or release.get("release_index")
            != int(release.get("request_index", -1)) + int(release.get("delay", -1)),
            "release_schedule",
        )
        add(
            bool(_nested_keys(release) & {"exact_label", "observed_label", "hidden_parameter"}),
            "release_label_leak",
        )
    for seed in expected_seeds:
        seed_public = [row for row in views.public if row.get("seed") == seed]
        add(
            [int(row.get("chronology_index", -1)) for row in seed_public]
            != list(range(EVENTS_PER_SEED)),
            "chronology",
        )
        add(
            sum(row.get("split") == "warmup" for row in seed_public) != WARMUP_COUNT, "warmup_split"
        )
        add(
            sum(row.get("split") == "prospective" for row in seed_public)
            != EVENTS_PER_SEED - WARMUP_COUNT,
            "prospective_split",
        )
        seed_truth = [row for row in views.authority if row.get("seed") == seed]
        for window in exp7198.WINDOWS:
            selected = [row for row in seed_truth if row.get("window") == window["name"]]
            family_counts = [
                sum(row.get("family_id") == family for row in selected) for family in FAMILIES
            ]
            add(
                len(selected) != int(window["count"]) or len(set(family_counts)) != 1,
                "window_family_balance",
            )
    return errors


def _small_subsets(max_size: int) -> list[tuple[int, ...]]:
    """Freeze all empty-through-small hypothesis subsets before parity checks."""

    return [
        subset
        for size in range(max_size + 1)
        for subset in itertools.combinations(PARAMETER_DOMAIN, size)
    ]


def _finite_parity(max_subset_size: int) -> tuple[JsonDict, int]:
    """Exhaustively compare supported inputs and query blocks for small states."""

    mismatches = 0
    prediction_cases = 0
    energy_cases = 0
    query_cases = 0
    state_count = 0
    subsets = _small_subsets(max_subset_size)
    blocks = [
        [
            {
                "event_id": f"q-{start}-{offset}",
                "family_id": "",
                "numeric_value": (start + offset) % 33,
            }
            for offset in range(BLOCK_SIZE)
        ]
        for start in range(0, len(PARAMETER_DOMAIN), BLOCK_SIZE)
    ]
    for family in FAMILIES:
        for subset in subsets:
            state_count += 1
            survivor_set = set(subset)
            packed = PackedBeliefController.from_survivors({family: survivor_set})
            reference = exp7199.VersionSpaceController()
            reference.families[family].hypotheses = survivor_set
            for value in PARAMETER_DOMAIN:
                event = {
                    "event_id": f"p-{family}-{value}",
                    "family_id": family,
                    "numeric_value": value,
                }
                prediction_cases += 1
                mismatches += int(packed.predict(event) != reference.predict(event))
                for label in ("accept", "reject"):
                    energy_cases += 1
                    energy = packed.energy(label, event)
                    expected = None
                    if survivor_set:
                        expected = sum(
                            exact_label(family, value, parameter) != label
                            for parameter in survivor_set
                        ) / len(survivor_set)
                    mismatches += int(energy["value"] != expected)
            for block_number, block in enumerate(blocks):
                family_block = [dict(row, family_id=family) for row in block]
                ties = exp7199.seeded_tie_ranks(RANDOM_SEED, block_number, family_block)
                query_cases += 1
                packed_id = packed.select_request(family_block, ties)["event_id"]
                reference_id = exp7199.select_request(
                    family_block, "priority_admission", ties, reference
                )["event_id"]
                mismatches += int(packed_id != reference_id)
    return (
        {
            "check": "finite_subset_prediction_energy_query",
            "families": list(FAMILIES),
            "maximum_subset_size": max_subset_size,
            "state_count": state_count,
            "prediction_case_count": prediction_cases,
            "energy_case_count": energy_cases,
            "query_case_count": query_cases,
            "mismatch_count": mismatches,
            "passed": mismatches == 0,
        },
        mismatches,
    )


def _random_replay(replay_steps: int) -> tuple[JsonDict, int]:
    """Replay delayed removals, resets, and rollbacks with fixed local seeds."""

    mismatches = 0
    prediction_cases = 0
    query_cases = 0
    release_count = 0
    reset_count = 0
    rollback_count = 0
    for seed in REPLAY_SEEDS:
        rng = random.Random(seed)
        packed = PackedBeliefController()
        reference = exp7199.VersionSpaceController()
        pending: list[JsonDict] = []
        queries = 0
        for block_start in range(0, replay_steps, BLOCK_SIZE):
            block = [
                {
                    "event_id": f"replay-{seed}-{index}",
                    "family_id": FAMILIES[rng.randrange(len(FAMILIES))],
                    "numeric_value": rng.randrange(33),
                }
                for index in range(block_start, min(block_start + BLOCK_SIZE, replay_steps))
            ]
            ties = {str(row["event_id"]): rank for rank, row in enumerate(reversed(block))}
            for event in block:
                prediction_cases += 1
                mismatches += int(packed.predict(event) != reference.predict(event))
            query_cases += 1
            selected = packed.select_request(block, ties)
            expected = exp7199.select_request(block, "priority_admission", ties, reference)
            mismatches += int(selected["event_id"] != expected["event_id"])
            block_end = block_start + len(block) - 1
            if queries < QUERY_CEILING and len(pending) < PENDING_CAPACITY:
                parameter = rng.randrange(33)
                delay = DELAY_SUPPORT[rng.randrange(len(DELAY_SUPPORT))]
                pending.append(
                    {
                        **dict(selected),
                        "observed_label": exact_label(
                            str(selected["family_id"]), int(selected["numeric_value"]), parameter
                        ),
                        "role": "support",
                        "request_index": block_end,
                        "release_index": block_end + delay,
                    }
                )
                queries += 1
            due = [row for row in pending if int(row["release_index"]) <= block_end]
            if due:
                before_reference = deepcopy(reference)
                parent = packed.state_hash()
                receipt = packed.commit_batch(
                    due,
                    current_cycle=block_end,
                    expected_parent_hash=parent,
                )
                for release in due:
                    update = reference.observe(
                        release,
                        observed_label=str(release["observed_label"]),
                        role="support",
                        request_index=int(release["request_index"]),
                        release_index=int(release["release_index"]),
                    )
                    reset_count += int(bool(update["rollback_hash"]))
                release_count += len(due)
                if block_start % (BLOCK_SIZE * 8) == 0:
                    packed.rollback(receipt)
                    rollback_count += 1
                    reference = before_reference
                    receipt = packed.commit_batch(
                        due,
                        current_cycle=block_end,
                        expected_parent_hash=packed.state_hash(),
                    )
                    for release in due:
                        reference.observe(
                            release,
                            observed_label=str(release["observed_label"]),
                            role="support",
                            request_index=int(release["request_index"]),
                            release_index=int(release["release_index"]),
                        )
                mismatches += int(receipt["new_state_hash"] != packed.state_hash())
                for family in FAMILIES:
                    mismatches += int(
                        packed.survivors(family) != reference.families[family].hypotheses
                    )
                due_ids = {str(row["event_id"]) for row in due}
                pending = [row for row in pending if str(row["event_id"]) not in due_ids]
    return (
        {
            "check": "delayed_random_replay",
            "replay_seeds": list(REPLAY_SEEDS),
            "steps_per_replay": replay_steps,
            "prediction_case_count": prediction_cases,
            "query_case_count": query_cases,
            "release_count": release_count,
            "reset_count": reset_count,
            "rollback_count": rollback_count,
            "mismatch_count": mismatches,
            "passed": mismatches == 0,
        },
        mismatches,
    )


def run_parity_audit(
    *, max_subset_size: int = 2, replay_steps: int = 256
) -> tuple[list[JsonDict], JsonDict]:
    """Run exhaustive small-state parity and longer fixed delayed replays."""

    finite, finite_mismatches = _finite_parity(max_subset_size)
    replay, replay_mismatches = _random_replay(replay_steps)
    rows = [finite, replay]
    return rows, {
        "mismatch_count": finite_mismatches + replay_mismatches,
        "prediction_case_count": finite["prediction_case_count"] + replay["prediction_case_count"],
        "energy_case_count": finite["energy_case_count"],
        "query_case_count": finite["query_case_count"] + replay["query_case_count"],
        "tested_state_count": finite["state_count"],
    }


def run_mutation_controls(root: Path) -> list[JsonDict]:
    """Check future, order, stale, corrupt, and rollback transaction boundaries."""

    root.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []
    future = PackedBeliefController()
    future_before = future.state_bytes()
    future_rejected = False
    try:
        future.commit_batch(
            [_release_for_control("future", label="accept", release_index=12)],
            current_cycle=8,
            expected_parent_hash=future.state_hash(),
        )
    except CommitRejected as error:
        future_rejected = str(error) == "future_release"
    rows.append(
        {
            "mutation_id": "future_label",
            "boundary": "release_time",
            "passed": future_rejected and future.state_bytes() == future_before,
        }
    )

    first = _release_for_control("order-a", value=0, label="accept")
    second = _release_for_control("order-b", value=0, label="reject")
    forward = PackedBeliefController()
    reverse = PackedBeliefController()
    forward.commit_batch(
        [first, second], current_cycle=8, expected_parent_hash=forward.state_hash()
    )
    reverse.commit_batch(
        [second, first], current_cycle=8, expected_parent_hash=reverse.state_hash()
    )
    rows.append(
        {
            "mutation_id": "release_order",
            "boundary": "ordered_batch",
            "passed": forward.state_hash() != reverse.state_hash()
            and forward.survivors("lower_bound") != reverse.survivors("lower_bound"),
        }
    )

    stale = PackedBeliefController()
    stale_before = stale.state_bytes()
    stale_rejected = False
    try:
        stale.commit_batch(
            [_release_for_control("stale")],
            current_cycle=8,
            expected_parent_hash="sha256:" + "0" * 64,
        )
    except CommitRejected as error:
        stale_rejected = str(error) == "stale_parent"
    rows.append(
        {
            "mutation_id": "stale_parent",
            "boundary": "parent_hash",
            "passed": stale_rejected and stale.state_bytes() == stale_before,
        }
    )

    corrupt_path = root / "corrupt-state.json"
    corrupt = PackedBeliefController()
    corrupt.save(corrupt_path)
    corrupt_value = json.loads(corrupt_path.read_text(encoding="utf-8"))
    corrupt_value["families"]["lower_bound"]["vote_counts"][0] += 1
    transactional._atomic_write(corrupt_path, transactional.canonical_json_bytes(corrupt_value))
    corrupt_rejected = False
    try:
        corrupt.commit_batch(
            [_release_for_control("corrupt")],
            current_cycle=8,
            expected_parent_hash=corrupt.state_hash(),
            state_path=corrupt_path,
        )
    except CommitRejected as error:
        corrupt_rejected = str(error) == "corrupt_durable_state"
    rows.append(
        {
            "mutation_id": "corrupt_vote_cache",
            "boundary": "durable_state_load",
            "passed": corrupt_rejected,
        }
    )

    rollback_path = root / "rollback-state.json"
    rollback = PackedBeliefController()
    rollback.save(rollback_path)
    parent_bytes = rollback.state_bytes()
    receipt = rollback.commit_batch(
        [_release_for_control("rollback")],
        current_cycle=8,
        expected_parent_hash=rollback.state_hash(),
        state_path=rollback_path,
    )
    rollback_row = rollback.rollback(receipt, state_path=rollback_path)
    rows.append(
        {
            "mutation_id": "rollback",
            "boundary": "inverse_transaction",
            "passed": rollback_row["byte_identical"] is True
            and rollback_path.read_bytes() == parent_bytes,
        }
    )
    return rows


def _release_for_control(
    event_id: str,
    *,
    value: int = 16,
    label: str = "accept",
    release_index: int = 8,
) -> JsonDict:
    """Build one public release row for isolated mutation controls."""

    return {
        "event_id": event_id,
        "family_id": "lower_bound",
        "numeric_value": value,
        "observed_label": label,
        "role": "support",
        "request_index": 4,
        "release_index": release_index,
    }


def _spawn_evaluator(paths: ExperimentPaths, *, progress: bool) -> None:
    """Run the authority generator in a bounded child process with live output."""

    if progress:
        print("PHASE 1 SUBPROCESS START: launch separate stream evaluator", flush=True)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--evaluator-output-root",
        str(paths.artifact.parent),
    ]
    completed = subprocess.run(command, check=False, timeout=180)
    if completed.returncode != 0:
        raise RuntimeError(f"evaluator_subprocess_failed:{completed.returncode}")
    if progress:
        print("PHASE 1 SUBPROCESS END: separate stream evaluator completed", flush=True)


def _stream_path_receipt(paths: ExperimentPaths) -> JsonDict:
    """Bind each separated stream view to its exact file hash."""

    return {
        name: {"path": str(path), "sha256": _sha256_path(path)}
        for name, path in (
            ("public_stream", paths.public_stream),
            ("authority_sidecar", paths.authority_sidecar),
            ("release_manifest", paths.release_manifest),
            ("public_manifest", paths.stream_manifest),
        )
    }


def _hypothesis_domain_contract(controller: PackedBeliefController) -> JsonDict:
    """State exact finite limits and measured serialized memory bytes."""

    return {
        "families": list(FAMILIES),
        "family_count": len(FAMILIES),
        "input_residue_domain": list(PARAMETER_DOMAIN),
        "hypothesis_parameter_domain": list(PARAMETER_DOMAIN),
        "hypotheses_per_family": len(PARAMETER_DOMAIN),
        "survivor_mask_bits_per_family": len(PARAMETER_DOMAIN),
        "survivor_mask_bytes_per_family": (len(PARAMETER_DOMAIN) + 7) // 8,
        "cached_vote_count_entries": len(FAMILIES) * len(PARAMETER_DOMAIN),
        "cached_vote_entry_bytes": 1,
        "packed_kernel_bytes": len(FAMILIES) * ((len(PARAMETER_DOMAIN) + 7) // 8)
        + len(FAMILIES) * len(PARAMETER_DOMAIN),
        "initial_canonical_state_bytes": len(controller.state_bytes()),
        "numeric_inputs_use_modulo_33_residue": True,
        "prediction_tie": "reject",
        "empty_prediction": "abstain",
        "energy_empty_or_unknown": None,
        "target_label_or_hidden_parameter_stored": False,
        "pending_capacity": PENDING_CAPACITY,
        "feedback_delay_support": list(DELAY_SUPPORT),
        "query_ceiling_per_stream_arm": QUERY_CEILING,
    }


def _future_hardware_path() -> JsonDict:
    """Describe bounded table mappings without claiming measured acceleration."""

    kernel_bytes = len(FAMILIES) * ((len(PARAMETER_DOMAIN) + 7) // 8) + len(FAMILIES) * len(
        PARAMETER_DOMAIN
    )
    return {
        "measured_now": "CPU packed integer AND, popcount, and cached vote lookup",
        "cpu_packed_kernel_bytes": kernel_bytes,
        "rust_simd_path": "four 64-bit survivor masks plus 132 one-byte cached vote counts",
        "fpga_table_path": "four 33-bit masks and four 33-entry vote-count tables",
        "fpga_table_bits": kernel_bytes * 8,
        "native_or_fpga_measured": False,
        "runtime_acceleration_claimed": False,
        "broader_model_expansion": "deferred",
        "mpmmine_external_validation": "deferred",
    }


def _compiler_checkpoint(paths: ExperimentPaths) -> JsonDict:
    """Write reloadable initial states that contain no evaluator authority."""

    controller = PackedBeliefController()
    states = [
        {
            "seed": seed,
            "state": controller.state_dict(),
            "state_hash": controller.state_hash(),
        }
        for seed in STREAM_SEEDS
    ]
    checkpoint = {
        "schema": "carnot.exp7226.compiler_checkpoint.v1",
        "run_date": RUN_DATE,
        "state_version": 0,
        "parent_hash": None,
        "controller_count": len(states),
        "controllers": states,
        "contains_hidden_parameter": False,
        "contains_target_label": False,
    }
    transactional._atomic_write(
        paths.compiler_state, transactional.canonical_json_bytes(checkpoint)
    )
    restored = _load_object(paths.compiler_state)
    if any(
        PackedBeliefController.from_state(row["state"]).state_hash() != row["state_hash"]
        for row in restored.get("controllers", [])
    ):
        raise ValueError("compiler_checkpoint_restore_mismatch")
    return {
        "path": str(paths.compiler_state),
        "sha256": _sha256_path(paths.compiler_state),
        "state_version": checkpoint["state_version"],
        "parent_hash": checkpoint["parent_hash"],
        "controller_count": checkpoint["controller_count"],
        "reload_match": True,
    }


def _stable_value(value: Any) -> Any:
    """Remove host and wall-time values from the scientific checksum."""

    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in value.items()
            if key not in {"duration_s", "execution_host", "started_at_utc", "completed_at_utc"}
        }
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    return value


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash exact sources, settings, stream receipts, and raw audit rows."""

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
        "belief_compiler_ready_score",
        "stream_manifest_path",
        "compiler_state_path",
        "parity_rows",
        "mutation_rows",
        "hypothesis_domain_contract",
        "future_hardware_path",
        "stream_conformance_errors",
        "compiler_claim_scope",
    )
    return transactional.sha256_json(
        _stable_value({field: artifact.get(field) for field in fields})
    )


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    paths: ExperimentPaths,
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Create a schema-complete blocked object before any readiness claim."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": socket.gethostname(),
        "duration_s": duration_s,
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": {
            "planned_streams": len(STREAM_SEEDS),
            "attempted_streams": 0,
            "completed_streams": 0,
            "censored_streams": len(STREAM_SEEDS),
            "planned_events": len(STREAM_SEEDS) * EVENTS_PER_SEED,
            "attempted_events": 0,
            "completed_events": 0,
            "censored_events": len(STREAM_SEEDS) * EVENTS_PER_SEED,
            "independent_units_planned": len(STREAM_SEEDS),
            "independent_units_completed": 0,
        },
        "random_seed": {
            "compiler": RANDOM_SEED,
            "streams": list(STREAM_SEEDS),
            "random_replays": list(REPLAY_SEEDS),
        },
        "reproducibility_checksum": None,
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_external:unknown_precondition",
        "MODEL_SPECS": [],
        "model_invoked": False,
        "belief_compiler_ready_score": 0,
        "stream_manifest_path": {
            "public_stream": {"path": str(paths.public_stream), "sha256": None},
            "authority_sidecar": {"path": str(paths.authority_sidecar), "sha256": None},
            "release_manifest": {"path": str(paths.release_manifest), "sha256": None},
            "public_manifest": {"path": str(paths.stream_manifest), "sha256": None},
        },
        "compiler_state_path": {
            "path": str(paths.compiler_state),
            "sha256": None,
            "state_version": None,
            "parent_hash": None,
            "controller_count": 0,
            "reload_match": False,
        },
        "parity_rows": [],
        "mutation_rows": [],
        "hypothesis_domain_contract": _hypothesis_domain_contract(PackedBeliefController()),
        "future_hardware_path": _future_hardware_path(),
        "stream_conformance_errors": [],
        "compiler_claim_scope": {
            "learning_efficacy_claimed": False,
            "runtime_acceleration_claimed": False,
            "ecai_template_learner_reproduced": False,
            "compact_representation_is_implementation_cost_hypothesis": True,
        },
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    paths: ExperimentPaths,
    *,
    duration_s: float,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> JsonDict:
    """Return row-free terminal evidence for an unchanged external block."""

    now = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks,
        source_hashes,
        paths,
        started_at=started_at or now,
        completed_at=completed_at or now,
        duration_s=duration_s,
    )
    failed = artifact["gate_check_summary"]["failed_check"] or "unknown_precondition"
    artifact["honest_verdict"] = f"blocked_external:{failed}"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    check_files: bool = False,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    """Cold-check schema, readiness gates, source bytes, and stable checksum."""

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
    add(
        not isinstance(artifact["duration_s"], (int, float)) or artifact["duration_s"] < 0,
        "duration",
    )
    add(artifact["MODEL_SPECS"] != [], "model_specs")
    add(artifact["model_invoked"] is not False, "model_invoked")
    add(artifact["verifier_is_oracle"] is not True, "oracle_classification")
    for field in ("started_at_utc", "completed_at_utc"):
        try:
            datetime.fromisoformat(str(artifact[field]))
        except ValueError:
            add(True, field)
    blocked = artifact["verdict_class"] == "blocked"
    if blocked:
        add(artifact["status"] != "blocked", "blocked_status")
        add(artifact["inference_substrate"] != "blocked_no_run", "blocked_substrate")
        add(artifact["inference_substrate_class"] != "blocked_no_run", "blocked_substrate_class")
        add(artifact["belief_compiler_ready_score"] != 0, "blocked_ready")
        add(bool(artifact["rows"]), "blocked_rows")
        summary = artifact["gate_check_summary"]
        add(summary.get("passed") is not False, "blocked_gate")
        for name in ("failed_check", "upstream", "field", "expected_value", "observed_value"):
            add(summary.get(name) is None, f"blocked_gate_{name}")
    else:
        add(artifact["status"] != "complete", "status")
        add(artifact["inference_substrate"] != INFERENCE_SUBSTRATE, "inference_substrate")
        add(
            artifact["inference_substrate_class"] != INFERENCE_SUBSTRATE_CLASS,
            "inference_substrate_class",
        )
        add(artifact["gate_check_summary"].get("passed") is not True, "preconditions")
        add(artifact["belief_compiler_ready_score"] != 1, "ready_score")
        add(artifact["verdict_class"] != "circular_positive", "verdict_class")
        add(not str(artifact["honest_verdict"]).startswith("complete"), "honest_verdict")
        add(len(artifact["rows"]) != len(STREAM_SEEDS), "row_count")
        add(
            any(
                not {"unit_id", "arm", "seed", "metric", "error", "abstention"} <= set(row)
                for row in artifact["rows"]
            ),
            "row_schema",
        )
        add(bool(artifact["stream_conformance_errors"]), "stream_conformance")
        add(any(row.get("passed") is not True for row in artifact["parity_rows"]), "parity")
        add(any(row.get("passed") is not True for row in artifact["mutation_rows"]), "mutations")
        add(artifact["compiler_state_path"].get("reload_match") is not True, "compiler_reload")
        budget = artifact["sample_size_budget"]
        add(budget.get("completed_streams") != len(STREAM_SEEDS), "completed_streams")
        add(budget.get("censored_streams") != 0, "censored_streams")
        if check_files:
            for path_text, expected_hash in artifact["source_artifact_hashes"].items():
                add(_sha256_path(_resolve(repo_root, path_text)) != expected_hash, "source_hashes")
            for receipt in artifact["stream_manifest_path"].values():
                add(
                    _sha256_path(_resolve(repo_root, receipt["path"])) != receipt["sha256"],
                    "stream_hashes",
                )
            compiler = artifact["compiler_state_path"]
            add(
                _sha256_path(_resolve(repo_root, compiler["path"])) != compiler["sha256"],
                "compiler_hash",
            )
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
    progress: bool = False,
) -> JsonDict:
    """Run gates, isolated generation, parity, mutation, and cold validation."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    if progress:
        print("PHASE 0 START: check code, spec, imports, upstream fields, and outputs", flush=True)
    checks, _, source_hashes = collect_preconditions(
        repo_root, paths, upstream_artifact=upstream_artifact
    )
    if not all(row["passed"] for row in checks):
        if progress:
            print("PHASE 0 END: external prerequisite failed; no CPU compiler ran", flush=True)
        return build_blocked_artifact(
            checks,
            source_hashes,
            paths,
            duration_s=time.monotonic() - started,
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
        )
    if progress:
        print("PHASE 0 END: all authenticated preconditions passed", flush=True)
        print("PHASE 1 START: generate 20 streams in the separate evaluator process", flush=True)
    _spawn_evaluator(paths, progress=progress)
    if progress:
        print("PHASE 1 END: fresh stream generation completed", flush=True)
        print(
            "PHASE 2 START: load views and check splits, grounding, and authority separation",
            flush=True,
        )
    views = load_stream_views(paths)
    stream_errors = stream_conformance_errors(views)
    if progress:
        print(
            f"PHASE 2 END: stream checks completed; errors={len(stream_errors)}",
            flush=True,
        )
        print(
            "PHASE 3 BENCHMARK START: exhaustive finite parity and delayed random replay",
            flush=True,
        )
    parity_rows, parity_summary = run_parity_audit()
    if progress:
        print(
            "PHASE 3 BENCHMARK END: "
            f"states={parity_summary['tested_state_count']}; "
            f"mismatches={parity_summary['mismatch_count']}",
            flush=True,
        )
        print("PHASE 4 START: run future, order, stale, corrupt, and rollback controls", flush=True)
    with tempfile.TemporaryDirectory(prefix="carnot-exp7226-mutations-") as directory:
        mutation_rows = run_mutation_controls(Path(directory))
    if progress:
        print(
            f"PHASE 4 END: mutation controls completed; passed={sum(row['passed'] for row in mutation_rows)}/{len(mutation_rows)}",
            flush=True,
        )
        print(
            "PHASE 5 START: serialize and reload packed state under results/checkpoints", flush=True
        )
    compiler_receipt = _compiler_checkpoint(paths)
    stream_receipt = _stream_path_receipt(paths)
    for receipt in stream_receipt.values():
        source_hashes[str(receipt["path"])] = receipt["sha256"]
    source_hashes[str(paths.compiler_state)] = compiler_receipt["sha256"]
    if progress:
        print("PHASE 5 END: compiler checkpoint reload matched exact state hashes", flush=True)
        print("PHASE 6 START: assemble full-denominator readiness evidence", flush=True)
    rows = [
        {
            "unit_id": f"{seed}:packed_belief_fixture",
            "arm": "packed_belief_fixture",
            "seed": seed,
            "metric": "fresh_stream_readiness",
            "error": 0,
            "abstention": 0,
            "event_count": EVENTS_PER_SEED,
            "warmup_event_count": WARMUP_COUNT,
            "prospective_event_count": EVENTS_PER_SEED - WARMUP_COUNT,
            "query_ceiling": QUERY_CEILING,
            "pending_capacity": PENDING_CAPACITY,
        }
        for seed in STREAM_SEEDS
    ]
    ready = (
        not stream_errors
        and parity_summary["mismatch_count"] == 0
        and all(row["passed"] is True for row in mutation_rows)
        and compiler_receipt["reload_match"] is True
    )
    completed_at = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks,
        source_hashes,
        paths,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=time.monotonic() - started,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "rows": rows,
            "sample_size_budget": {
                "planned_streams": len(STREAM_SEEDS),
                "attempted_streams": len(STREAM_SEEDS),
                "completed_streams": len(STREAM_SEEDS),
                "censored_streams": 0,
                "planned_events": len(STREAM_SEEDS) * EVENTS_PER_SEED,
                "attempted_events": len(views.public),
                "completed_events": len(views.public),
                "censored_events": 0,
                "independent_units_planned": len(STREAM_SEEDS),
                "independent_units_completed": len(STREAM_SEEDS),
                "parity_states_completed": parity_summary["tested_state_count"],
                "parity_prediction_cases_completed": parity_summary["prediction_case_count"],
                "parity_energy_cases_completed": parity_summary["energy_case_count"],
                "parity_query_cases_completed": parity_summary["query_case_count"],
                "mutation_controls_completed": len(mutation_rows),
            },
            "verdict_class": "circular_positive" if ready else "null",
            "honest_verdict": (
                "complete: packed belief compiler matches the circular finite reference; no learning or speed claim"
                if ready
                else "complete_null: one or more packed compiler readiness checks failed"
            ),
            "belief_compiler_ready_score": int(ready),
            "stream_manifest_path": stream_receipt,
            "compiler_state_path": compiler_receipt,
            "parity_rows": parity_rows,
            "mutation_rows": mutation_rows,
            "stream_conformance_errors": stream_errors,
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _require_valid(validate_artifact(artifact, check_files=True, repo_root=repo_root))
    if progress:
        print("PHASE 6 END: complete artifact object passed cold validation", flush=True)
    return artifact


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Publish a validated terminal artifact through one atomic rename."""

    _require_valid(validate_artifact(artifact, check_files=artifact["verdict_class"] != "blocked"))
    return transactional._atomic_write(path, transactional.canonical_json_bytes(dict(artifact)))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed execution date and private evaluator mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--upstream-artifact", type=Path, default=DEFAULT_UPSTREAM_ARTIFACT)
    parser.add_argument("--evaluator-output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run isolated generation and atomically publish one terminal result."""

    print("PHASE 0 PRECONDITION: parse inputs before checking any resource", flush=True)
    args = _parse_args(argv)
    if args.evaluator_output_root is not None:
        return evaluator_worker(ExperimentPaths.under(args.evaluator_output_root))
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
    print(
        "PHASE 7 FINAL VALIDATION START: check terminal object and exact file receipts", flush=True
    )
    _require_valid(
        validate_artifact(
            artifact,
            check_files=artifact["verdict_class"] != "blocked",
            repo_root=REPO_ROOT,
        )
    )
    print("PHASE 7 FINAL VALIDATION END: terminal object is valid", flush=True)
    print("FINAL ATOMIC WRITE START", flush=True)
    transactional._atomic_write(paths.artifact, transactional.canonical_json_bytes(artifact))
    print("FINAL ATOMIC WRITE END", flush=True)
    print("PHASE 8 END: terminal deliverable is stable", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
