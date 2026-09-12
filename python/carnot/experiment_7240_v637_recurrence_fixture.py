"""Build a bounded archived-belief controller and sealed recurrence fixture.

The controller keeps one active packed finite-hypothesis state. It saves up to
four immutable survivor-mask snapshots when released evidence contradicts that
active state. A saved snapshot can return only after recent released evidence
passes the fixed validation rule. Predictions happen before any release at the
same event, so a write can affect only a later prediction.

Spec refs: REQ-CL-7240 and SCENARIO-CL-7240-*.
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
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7240
SCHEMA = "carnot.exp7240.v637_recurrence_fixture.v1"
STATE_SCHEMA = "carnot.archived_belief_controller.v1"
MILESTONE = "2026.09.637"
RUN_DATE = "20260912"
RANDOM_SEED = 7_240_000
STREAM_SEEDS = tuple(range(7_240_001, 7_240_033))
STREAM_COUNT = 32
EVENTS_PER_STREAM = 1_024
WARMUP_COUNT = 128
PENDING_CAPACITY = 4
QUERY_CEILING = 128
QUERY_BLOCK_SIZE = 8
DELAY_SUPPORT = (0, 4, 16, 32)
ARCHIVE_CAP = 4
VALIDATION_WINDOW = 16
MIN_VALIDATION_WITNESSES = 8
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"
EXPECTED_UPSTREAM_SHA256 = "sha256:ada61ddef40c2b1875666da974a319e12c1d0b3b4864efeec6aa4e2a57ff7eb6"

FAMILIES = tuple(exp7226.FAMILIES)
PARAMETER_DOMAIN = tuple(exp7226.PARAMETER_DOMAIN)
FULL_MASK = exp7226.FULL_MASK
PATTERNS = (
    "aba_recurrence",
    "abc_unseen_drift",
    "gradual_drift",
    "unchanged_input_label_drift",
)
ARMS = (
    "frozen_warmup",
    "destructive_packed_learner",
    "reset_relearn_no_archive",
    "unvalidated_stale_archive_reuse",
    "validation_selected_archive",
    "shuffled_nomination_validated",
)
ADAPTIVE_ARMS = ARMS[1:]
NOMINATION_MODES = {"none", "stale", "validated", "shuffled_validated"}
FORBIDDEN_PUBLIC_FIELDS = {
    "delay",
    "drift_pattern",
    "exact_label",
    "hidden_parameter",
    "observed_label",
    "regime_id",
    "release_index",
    "segment",
    "stream_seed",
    "target_label",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM_ARTIFACT = Path("results/experiment_7227_v636_belief_learning.json")
DEFAULT_STREAM_ROOT = Path("results/streams/experiment_7240")
DEFAULT_PUBLIC_MANIFEST = DEFAULT_STREAM_ROOT / "public_manifest.json"
DEFAULT_RAW_ROWS = Path("results/checkpoints/experiment_7240_v637_event_rows.jsonl")
DEFAULT_CHECKPOINT = Path("results/checkpoints/experiment_7240_v637_controller_states.json")
DEFAULT_CONTROL_RECEIPTS = Path("results/checkpoints/experiment_7240_v637_control_receipts.json")
DEFAULT_ARTIFACT = Path("results/experiment_7240_v637_recurrence_fixture.json")
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
    Path("python/carnot/experiment_7226_v636_belief_compiler.py"),
    Path("python/carnot/experiment_7227_v636_belief_learning.py"),
    Path("python/carnot/experiment_7240_v637_recurrence_fixture.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("scripts/experiments/experiment_7240_v637_recurrence_fixture.py"),
    Path("tests/python/test_experiment_7240_v637_recurrence_fixture.py"),
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
    "MODEL_SPECS",
    "model_invoked",
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
    "recurrence_fixture_ready_score",
    "stream_manifest_path",
    "controller_contract",
    "arm_contract",
    "continuous_self_learning_task",
    "hardware_operation_map",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Version the artifact and bind experiment_id and milestone to this task.",
    "experiment_id": "A fixed identifier prevents another task from supplying this evidence.",
    "milestone": "The milestone binds the result to the V637 recurrence contract.",
    "status": "Terminal complete or blocked only; unfinished work uses a separate checkpoint path.",
    "run_date": "Use 20260912; retain actual UTC start and end timestamps.",
    "started_at_utc": "Keep the actual UTC start distinct from the fixed execution date.",
    "completed_at_utc": "Keep the actual UTC completion distinct from the fixed date.",
    "field_principles": "Keep ordinary values at top level; put their explanations in this map.",
    "preconditions_checked": "Observed paths, resources, model identity and upstream checks before expensive work.",
    "inference_substrate": "Use the recognized literal for the operation actually executed.",
    "inference_substrate_class": "Actual class determines the duration floor; never pad duration or relabel to pass.",
    "execution_venue": "Top-level orchestration is host; board rows name kv260, gatemate or polarfire.",
    "execution_host": "Actual hostname, distinct from execution_venue.",
    "duration_s": "Measured monotonic elapsed work; record phase spans separately.",
    "MODEL_SPECS": "Models actually invoked; [] for tasks with no LLM.",
    "model_invoked": "Current task execution only; historical sources and injected fixtures are separate.",
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
    "recurrence_fixture_ready_score": "One means controller prototype, sealed streams and positive controls are runnable.",
    "stream_manifest_path": "The public manifest path is fixed; private labels are separately hashed.",
    "controller_contract": "Archive cap, update rule, 16-release window, eight-witness rule, ties, query limit and rollback.",
    "arm_contract": "Six explicit arms with matched observable feedback and resource budgets.",
    "continuous_self_learning_task": "Candidate constraints are added, deactivated and reactivated across queries.",
    "hardware_operation_map": "CPU counters and bitsets now; Rust SIMD and FPGA BRAM matching later, with measured bytes.",
}

unwrap_principled = exp7213.unwrap_principled
gate_check = exp7213.gate_check
gate_summary = exp7213.gate_summary


class ArchiveCommitRejected(ValueError):
    """Report a rejected archive transaction before live bytes change."""


class ImmutableSealError(RuntimeError):
    """Report an attempt to replace already sealed evidence with changed bytes."""


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep public, private, checkpoint, raw, and terminal files separate."""

    public_stream: Path
    private_authority: Path
    release_schedule: Path
    public_manifest: Path
    raw_rows: Path
    checkpoint: Path
    control_receipts: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the task-owned paths below the repository result directory."""

        return cls.from_results_root(REPO_ROOT / "results")

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put all test evidence below one caller-owned temporary directory."""

        return cls.from_results_root(root)

    @classmethod
    def from_results_root(cls, root: Path) -> ExperimentPaths:
        """Derive every task path without assuming the checkout location."""

        stream_root = root / "streams" / "experiment_7240"
        checkpoint_root = root / "checkpoints"
        return cls(
            stream_root / "public_stream.jsonl",
            stream_root / "private_evaluator.jsonl",
            stream_root / "release_schedule.jsonl",
            stream_root / "public_manifest.json",
            checkpoint_root / DEFAULT_RAW_ROWS.name,
            checkpoint_root / DEFAULT_CHECKPOINT.name,
            checkpoint_root / DEFAULT_CONTROL_RECEIPTS.name,
            root / DEFAULT_ARTIFACT.name,
        )


@dataclass(frozen=True)
class StreamViews:
    """Hold learner-visible rows apart from private correctness authority."""

    public: list[JsonDict]
    authority: list[JsonDict]
    releases: list[JsonDict]
    manifest: JsonDict


@dataclass(frozen=True)
class FixturePanel:
    """Retain raw event evidence, aggregate rows, and restartable final states."""

    event_rows: list[JsonDict]
    rows: list[JsonDict]
    query_rows: list[JsonDict]
    final_states: list[JsonDict]
    maximum_controller_bytes: int


def _sha256_path(path: Path) -> str | None:
    """Hash exact file bytes while preserving a missing path as an observation."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def _resolve(repo_root: Path, path: str | Path) -> Path:
    """Resolve repository-relative evidence without altering absolute paths."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and keep malformed evidence from becoming a gate pass."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating evidence bytes."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode chronological rows with the shared canonical JSON representation."""

    return b"".join(transactional.canonical_json_bytes(dict(row)) for row in rows)


def _atomic_write(path: Path, payload: bytes) -> None:
    """Publish complete bytes through one flushed rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_immutable(path: Path, payload: bytes) -> None:
    """Accept an identical prior seal and reject changed replacement bytes."""

    if path.exists():
        if path.read_bytes() != payload:
            raise ImmutableSealError(f"immutable_path_conflict:{path}")
        return
    _atomic_write(path, payload)


def _task_identity(text: str) -> JsonDict:
    """Extract only the Exp7240 block from the executable roadmap."""

    match = re.search(r"(?ms)^- id: exp7240-recurrence-fixture\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(1)
    milestone = re.search(r"(?m)^  milestone: (.+)$", block)
    deliverable = re.search(r"(?m)^  deliverable: (.+)$", block)
    return {
        "id": "exp7240-recurrence-fixture" if block else None,
        "milestone": None if milestone is None else milestone.group(1).strip(),
        "deliverable": None if deliverable is None else deliverable.group(1).strip(),
    }


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    upstream_artifact: Path = DEFAULT_UPSTREAM_ARTIFACT,
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict]:
    """Authenticate task identity, source bytes, and exact Exp7227 evidence."""

    upstream_path = _resolve(repo_root, upstream_artifact)
    upstream = _load_object(upstream_path)
    hashes = {str(path): _sha256_path(_resolve(repo_root, path)) for path in SOURCE_PATHS}
    hashes[str(upstream_path)] = _sha256_path(upstream_path)
    spec_text = _resolve(repo_root, SPEC_PATH).read_text(encoding="utf-8")
    roadmap_text = _resolve(repo_root, "research-roadmap.yaml").read_text(encoding="utf-8")
    exclusion_text = _resolve(repo_root, "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    identity = _task_identity(roadmap_text)
    imports = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "carnot.experiment_7199_v634_bounded_acquisition",
            "carnot.experiment_7226_v636_belief_compiler",
            "carnot.experiment_7227_v636_belief_learning",
            "carnot.memory.transactional_constraint_memory",
        )
    }
    outputs = {
        "public": _path_writable(paths.public_stream),
        "private": _path_writable(paths.private_authority),
        "raw": _path_writable(paths.raw_rows),
        "checkpoint": _path_writable(paths.checkpoint),
        "artifact": _path_writable(paths.artifact),
    }
    quarantine = exp7213.quarantine_state(
        upstream,
        exclusion_text,
        upstream_path.name,
        "exp7227-belief-learning",
    )
    raw_receipts = {
        "decision_rows": upstream.get("decision_rows_path", {}),
        "state": {
            "path": "results/checkpoints/experiment_7227_v636_belief_learning_state.json",
            "sha256": upstream.get("source_artifact_hashes", {}).get(
                "results/checkpoints/experiment_7227_v636_belief_learning_state.json"
            )
            if isinstance(upstream.get("source_artifact_hashes"), Mapping)
            else None,
        },
    }
    raw_matches: dict[str, bool] = {}
    for name, receipt in raw_receipts.items():
        receipt = unwrap_principled(receipt)
        if not isinstance(receipt, Mapping):
            raw_matches[name] = False
            continue
        raw_path = _resolve(repo_root, str(receipt.get("path", "")))
        observed = _sha256_path(raw_path)
        hashes[str(raw_path)] = observed
        raw_matches[name] = observed is not None and observed == receipt.get("sha256")
    source_state = {
        str(path): "nonempty" if hashes[str(path)] is not None else "missing"
        for path in SOURCE_PATHS
    }
    expected_identity = {
        "id": "exp7240-recurrence-fixture",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT),
    }
    try:
        checksum_valid = bool(upstream) and exp7227.reproducibility_checksum(
            upstream
        ) == upstream.get("reproducibility_checksum")
    except (AttributeError, KeyError, TypeError, ValueError):
        checksum_valid = False
    checks = [
        gate_check(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-CL-7240",
            True,
            "REQ-CL-7240" in spec_text,
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7240-*",
            8,
            len(set(re.findall(r"SCENARIO-CL-7240-[A-Z-]+", spec_text))),
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
            "public,private,raw,checkpoint,artifact",
            dict.fromkeys(outputs, True),
            outputs,
        ),
        gate_check(
            "exp7227_artifact_hash",
            "exp7227-belief-learning",
            str(upstream_artifact),
            EXPECTED_UPSTREAM_SHA256,
            hashes[str(upstream_path)],
        ),
        gate_check(
            "exp7227_status",
            "exp7227-belief-learning",
            "status",
            "complete",
            upstream.get("status"),
        ),
        gate_check(
            "exp7227_run_complete",
            "exp7227-belief-learning",
            "belief_run_complete_score",
            1,
            upstream.get("belief_run_complete_score"),
        ),
        gate_check(
            "exp7227_fixed_recurrence_limit_failed",
            "exp7227-belief-learning",
            "belief_learning_value_score",
            0,
            upstream.get("belief_learning_value_score"),
        ),
        gate_check(
            "exp7227_checksum",
            "exp7227-belief-learning",
            "reproducibility_checksum",
            True,
            checksum_valid,
        ),
        gate_check(
            "exp7227_raw_receipts",
            "exp7227-belief-learning",
            "decision_rows,state",
            {"decision_rows": True, "state": True},
            raw_matches,
        ),
        gate_check(
            "exp7227_not_quarantined",
            "artifact_metadata_and_ops/exclusion_manifest.yaml",
            "quarantined",
            False,
            quarantine["quarantined"],
        ),
    ]
    return checks, hashes, upstream


def _survivor_masks(controller: exp7226.PackedBeliefController) -> dict[str, int]:
    """Project an active packed state down to its immutable survivor masks."""

    return {family: int(controller.family_state(family)["survivor_mask"]) for family in FAMILIES}


def _controller_from_masks(masks: Mapping[str, Any]) -> exp7226.PackedBeliefController:
    """Restore the existing packed representation from checked mask integers."""

    survivors = {
        family: {
            parameter for parameter in PARAMETER_DOMAIN if int(masks[family]) & (1 << parameter)
        }
        for family in FAMILIES
    }
    return exp7226.PackedBeliefController.from_survivors(survivors)


def _mask_hash(masks: Mapping[str, Any]) -> str:
    """Bind archive identity only to immutable finite survivor masks."""

    return transactional.sha256_json({family: int(masks[family]) for family in sorted(FAMILIES)})


def _prediction_from_masks(masks: Mapping[str, Any], witness: Mapping[str, Any]) -> str:
    """Evaluate an archived mask on one already released public witness."""

    family = str(witness["family_id"])
    numeric_value = int(witness["numeric_value"])
    mask = int(masks[family])
    if mask == 0:
        return "abstain"
    accept_mask = exp7226.ACCEPT_MASKS[family][numeric_value % len(PARAMETER_DOMAIN)]
    accepts = (mask & accept_mask).bit_count()
    return "accept" if accepts > mask.bit_count() / 2 else "reject"


def _active_contradiction(
    controller: exp7226.PackedBeliefController, release: Mapping[str, Any]
) -> bool:
    """Return true only when every active hypothesis conflicts with a release."""

    family = str(release["family_id"])
    value = int(release["numeric_value"]) % len(PARAMETER_DOMAIN)
    mask = int(controller.family_state(family)["survivor_mask"])
    accept_mask = exp7226.ACCEPT_MASKS[family][value]
    matching = accept_mask if release["observed_label"] == "accept" else FULL_MASK ^ accept_mask
    return mask != 0 and mask & matching == 0


class ArchivedBeliefController:
    """Add bounded immutable archive reuse around the shipped packed controller."""

    def __init__(
        self,
        *,
        archive_cap: int = ARCHIVE_CAP,
        nomination_mode: str = "validated",
    ) -> None:
        if not 0 <= archive_cap <= ARCHIVE_CAP:
            raise ValueError("invalid_archive_cap")
        if nomination_mode not in NOMINATION_MODES:
            raise ValueError("invalid_nomination_mode")
        self._state: JsonDict = {
            "schema": STATE_SCHEMA,
            "version": 0,
            "parent_hash": None,
            "archive_cap": archive_cap,
            "nomination_mode": nomination_mode,
            "active": exp7226.PackedBeliefController().state_dict(),
            "archives": [],
            "next_creation_order": 0,
            "release_window": [],
            "release_ids": [],
            "stale_reuse_count": 0,
            "last_nomination_receipt": {
                "candidate_count": 0,
                "validation_window_size": 0,
                "minimum_witnesses": MIN_VALIDATION_WITNESSES,
                "final_gate": "at_least_8_released_witnesses_and_zero_contradictions",
                "selected_archive_id": None,
                "candidates": [],
            },
        }

    @classmethod
    def from_active(
        cls,
        active: exp7226.PackedBeliefController,
        *,
        archive_cap: int = ARCHIVE_CAP,
        nomination_mode: str = "validated",
    ) -> ArchivedBeliefController:
        """Create the opt-in wrapper without mutating the supplied packed state."""

        controller = cls(archive_cap=archive_cap, nomination_mode=nomination_mode)
        controller._state["active"] = active.state_dict()
        return controller

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> ArchivedBeliefController:
        """Reject malformed active bytes, archive hashes, windows, and modes."""

        if value.get("schema") != STATE_SCHEMA:
            raise ValueError("invalid_archive_state_schema")
        archive_cap = value.get("archive_cap")
        mode = value.get("nomination_mode")
        if not isinstance(archive_cap, int) or isinstance(archive_cap, bool):
            raise ValueError("invalid_archive_cap")
        if not 0 <= archive_cap <= ARCHIVE_CAP or mode not in NOMINATION_MODES:
            raise ValueError("invalid_archive_contract")
        exp7226.PackedBeliefController.from_state(value.get("active", {}))
        archives = value.get("archives")
        release_window = value.get("release_window")
        release_ids = value.get("release_ids")
        if not isinstance(archives, list) or len(archives) > archive_cap:
            raise ValueError("invalid_archives")
        if not isinstance(release_window, list) or len(release_window) > VALIDATION_WINDOW:
            raise ValueError("invalid_release_window")
        if not isinstance(release_ids, list) or len(set(release_ids)) != len(release_ids):
            raise ValueError("invalid_release_ids")
        orders: list[int] = []
        for archive in archives:
            masks = archive.get("survivor_masks", {}) if isinstance(archive, Mapping) else {}
            if set(masks) != set(FAMILIES) or any(
                not isinstance(masks[family], int)
                or isinstance(masks[family], bool)
                or not 0 <= masks[family] <= FULL_MASK
                for family in FAMILIES
            ):
                raise ValueError("invalid_archive_masks")
            if archive.get("state_hash") != _mask_hash(masks):
                raise ValueError("invalid_archive_hash")
            orders.append(int(archive.get("creation_order", -1)))
        if len(set(orders)) != len(orders) or any(order < 0 for order in orders):
            raise ValueError("invalid_archive_order")
        candidate = deepcopy(dict(value))
        controller = cls.__new__(cls)
        controller._state = candidate
        return controller

    @classmethod
    def load(cls, path: Path) -> ArchivedBeliefController:
        """Restore one durable controller and reject non-object JSON state."""

        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("invalid_archive_state_object")
        return cls.from_state(value)

    def state_dict(self) -> JsonDict:
        """Return detached state so outside callers cannot mutate active bytes."""

        return deepcopy(self._state)

    def state_bytes(self) -> bytes:
        """Serialize every active, archive, release, and parent byte canonically."""

        return transactional.canonical_json_bytes(self._state)

    def state_hash(self) -> str:
        """Identify the full durable controller state."""

        return transactional.sha256_bytes(self.state_bytes())

    def save(self, path: Path) -> JsonDict:
        """Atomically publish a restartable controller state."""

        return transactional._atomic_write(path, self.state_bytes())

    def archives(self) -> list[JsonDict]:
        """Return detached immutable archive receipts for audit."""

        return deepcopy(self._state["archives"])

    def stale_reuse_count(self) -> int:
        """Report how often the unsafe control reused an unvalidated archive."""

        return int(self._state["stale_reuse_count"])

    def last_nomination_receipt(self) -> JsonDict:
        """Expose the most recent released-only candidate validation receipt."""

        return deepcopy(self._state["last_nomination_receipt"])

    def _active(self) -> exp7226.PackedBeliefController:
        """Restore the active packed state through its existing validator."""

        return self._active_from_state(self._state["active"])

    def _active_from_state(self, value: Mapping[str, Any]) -> exp7226.PackedBeliefController:
        """Create the packed backend while archive policy stays in this class."""

        return exp7226.PackedBeliefController.from_state(value)

    def _active_from_masks(self, masks: Mapping[str, Any]) -> exp7226.PackedBeliefController:
        """Restore one nominated archive through the selected packed backend."""

        return _controller_from_masks(masks)

    def _validate_live_state(self, value: Mapping[str, Any]) -> None:
        """Validate archive bytes independently from the selected packed backend."""

        ArchivedBeliefController.from_state(value)

    def _admit_state(self, value: Mapping[str, Any]) -> ArchivedBeliefController:
        """Create the next controller without changing archive selection semantics."""

        return type(self).from_state(value)

    def _load_durable_state(self, path: Path) -> ArchivedBeliefController:
        """Load durable bytes through the backend-aware controller constructor."""

        return type(self).load(path)

    def predict(self, public_event: Mapping[str, Any]) -> tuple[str, float]:
        """Read the frozen active state without changing any controller byte."""

        return self._active().predict(public_event)

    def energy(self, label: str, public_event: Mapping[str, Any]) -> JsonDict:
        """Read active disagreement energy through the selected packed backend."""

        return self._active().energy(label, public_event)

    def select_request(
        self,
        block: Sequence[Mapping[str, Any]],
        tie_ranks: Mapping[str, int],
    ) -> Mapping[str, Any]:
        """Use the existing public-only query acquisition on the active state."""

        return self._active().select_request(block, tie_ranks)

    @staticmethod
    def _append_archive(state: JsonDict, active: exp7226.PackedBeliefController) -> str | None:
        """Append new immutable masks and evict the oldest state at the fixed cap."""

        if int(state["archive_cap"]) == 0:
            return None
        masks = _survivor_masks(active)
        state_hash = _mask_hash(masks)
        existing = next(
            (row for row in state["archives"] if row["state_hash"] == state_hash),
            None,
        )
        if existing is not None:
            return str(existing["archive_id"])
        creation_order = int(state["next_creation_order"])
        archive_id = f"archive-{creation_order:06d}"
        state["archives"].append(
            {
                "archive_id": archive_id,
                "creation_order": creation_order,
                "state_hash": state_hash,
                "survivor_masks": masks,
            }
        )
        state["next_creation_order"] = creation_order + 1
        while len(state["archives"]) > int(state["archive_cap"]):
            oldest = min(state["archives"], key=lambda row: int(row["creation_order"]))
            state["archives"].remove(oldest)
        return archive_id

    @staticmethod
    def _nominate(state: JsonDict) -> tuple[JsonDict, JsonDict | None]:
        """Apply the same final released-witness gate to every nominated archive."""

        archives = list(state["archives"])
        mode = str(state["nomination_mode"])
        window = list(state["release_window"])
        if mode == "shuffled_validated":
            digest = transactional.sha256_json(window)
            archives.sort(key=lambda row: transactional.sha256_json([digest, row["archive_id"]]))
        else:
            archives.sort(key=lambda row: int(row["creation_order"]))
        rows: list[JsonDict] = []
        for nomination_order, archive in enumerate(archives):
            applicable = 0
            contradictions = 0
            for witness in window:
                prediction = _prediction_from_masks(archive["survivor_masks"], witness)
                if prediction != "abstain":
                    applicable += 1
                    contradictions += int(prediction != witness["observed_label"])
            rows.append(
                {
                    "archive_id": archive["archive_id"],
                    "creation_order": archive["creation_order"],
                    "nomination_order": nomination_order,
                    "released_witness_count": applicable,
                    "validation_loss": contradictions,
                    "contradiction_count": contradictions,
                    "gate_passed": (applicable >= MIN_VALIDATION_WITNESSES and contradictions == 0),
                }
            )
        eligible = [row for row in rows if row["gate_passed"]]
        selected_row = min(
            eligible,
            key=lambda row: (int(row["validation_loss"]), int(row["creation_order"])),
            default=None,
        )
        selected = (
            None
            if selected_row is None
            else next(
                row for row in state["archives"] if row["archive_id"] == selected_row["archive_id"]
            )
        )
        receipt = {
            "candidate_count": len(rows),
            "validation_window_size": len(window),
            "minimum_witnesses": MIN_VALIDATION_WITNESSES,
            "final_gate": "at_least_8_released_witnesses_and_zero_contradictions",
            "selected_archive_id": None if selected is None else selected["archive_id"],
            "candidates": rows,
        }
        return receipt, selected

    def commit_batch(
        self,
        releases: Sequence[Mapping[str, Any]],
        *,
        current_cycle: int,
        expected_parent_hash: str,
        state_path: Path | None = None,
    ) -> JsonDict:
        """Archive, validate, and commit one released batch after prediction."""

        parent_bytes = self.state_bytes()
        parent_hash = self.state_hash()
        if expected_parent_hash != parent_hash:
            raise ArchiveCommitRejected("stale_parent")
        try:
            self._validate_live_state(self._state)
        except ValueError as error:
            raise ArchiveCommitRejected("corrupt_live_state") from error
        if state_path is not None and state_path.exists():
            try:
                durable_hash = self._load_durable_state(state_path).state_hash()
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise ArchiveCommitRejected("corrupt_durable_state") from error
            if durable_hash != parent_hash:
                raise ArchiveCommitRejected("stale_durable_parent")
        try:
            normalized = [
                exp7226.PackedBeliefController._validate_release(row, current_cycle)
                for row in releases
            ]
        except exp7226.CommitRejected as error:
            raise ArchiveCommitRejected(str(error)) from error
        batch_ids = [str(row["event_id"]) for row in normalized]
        if len(set(batch_ids)) != len(batch_ids) or set(batch_ids) & set(
            self._state["release_ids"]
        ):
            raise ArchiveCommitRejected("duplicate_release")

        candidate = deepcopy(self._state)
        operations: list[JsonDict] = []
        for release in normalized:
            active = self._active_from_state(candidate["active"])
            before_active_hash = active.state_hash()
            contradiction = _active_contradiction(active, release)
            archived_id = self._append_archive(candidate, active) if contradiction else None
            candidate["release_window"].append(deepcopy(release))
            candidate["release_window"] = candidate["release_window"][-VALIDATION_WINDOW:]

            active.commit_batch(
                [release],
                current_cycle=current_cycle,
                expected_parent_hash=active.state_hash(),
            )
            reactivated: JsonDict | None = None
            nomination_receipt: JsonDict
            if candidate["nomination_mode"] == "stale" and contradiction:
                choices = [row for row in candidate["archives"] if row["archive_id"] != archived_id]
                reactivated = min(
                    choices,
                    key=lambda row: int(row["creation_order"]),
                    default=None,
                )
                nomination_receipt = {
                    "candidate_count": len(candidate["archives"]),
                    "validation_window_size": len(candidate["release_window"]),
                    "minimum_witnesses": MIN_VALIDATION_WITNESSES,
                    "final_gate": "not_applied_stale_control",
                    "selected_archive_id": (
                        None if reactivated is None else reactivated["archive_id"]
                    ),
                    "candidates": [],
                }
                candidate["stale_reuse_count"] = int(candidate["stale_reuse_count"]) + int(
                    reactivated is not None
                )
            elif candidate["nomination_mode"] in {"validated", "shuffled_validated"}:
                nomination_receipt, reactivated = self._nominate(candidate)
            else:
                nomination_receipt = {
                    "candidate_count": 0,
                    "validation_window_size": len(candidate["release_window"]),
                    "minimum_witnesses": MIN_VALIDATION_WITNESSES,
                    "final_gate": "archive_reuse_disabled",
                    "selected_archive_id": None,
                    "candidates": [],
                }
            if reactivated is not None:
                active = self._active_from_masks(reactivated["survivor_masks"])
                candidate["archives"] = [
                    row
                    for row in candidate["archives"]
                    if row["archive_id"] != reactivated["archive_id"]
                ]
            candidate["active"] = active.state_dict()
            candidate["last_nomination_receipt"] = nomination_receipt
            candidate["release_ids"].append(str(release["event_id"]))
            operations.append(
                {
                    "event_id": release["event_id"],
                    "active_hash_before": before_active_hash,
                    "active_hash_after": active.state_hash(),
                    "active_contradiction": contradiction,
                    "archived_before_reset": contradiction and archived_id is not None,
                    "archived_state_id": archived_id,
                    "reactivated_archive_id": (
                        None if reactivated is None else reactivated["archive_id"]
                    ),
                    "prediction_frozen_before_release": True,
                    "same_event_correction": False,
                    "nomination_receipt": nomination_receipt,
                }
            )
        candidate["version"] = int(candidate["version"]) + 1
        candidate["parent_hash"] = parent_hash
        admitted = self._admit_state(candidate)
        new_bytes = admitted.state_bytes()
        receipt = {
            "parent_hash": parent_hash,
            "new_state_hash": transactional.sha256_bytes(new_bytes),
            "parent_bytes_b64": transactional.encode_bytes(parent_bytes),
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
        """Restore exact parent bytes only from the transaction's exact child."""

        if self.state_hash() != receipt.get("new_state_hash"):
            raise ArchiveCommitRejected("stale_rollback")
        try:
            parent_bytes = transactional.decode_bytes(str(receipt["parent_bytes_b64"]))
            value = json.loads(parent_bytes)
            restored = type(self).from_state(value)
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            raise ArchiveCommitRejected("invalid_rollback_receipt") from error
        if restored.state_hash() != receipt.get("parent_hash"):
            raise ArchiveCommitRejected("rollback_parent_hash")
        if state_path is not None:
            transactional._atomic_write(state_path, parent_bytes)
        self._state = restored._state
        return {"restored_state_hash": self.state_hash(), "byte_identical": True}


def _stable_parameter(stream_index: int, family: str) -> int:
    """Freeze one evaluator-only base parameter before outcomes are generated."""

    return (4 + stream_index * 5 + FAMILIES.index(family) * 3) % len(PARAMETER_DOMAIN)


def _regime_parameter(pattern: str, chronology_index: int, base: int) -> tuple[str, int]:
    """Return evaluator-only regime identity and its finite hidden parameter."""

    shifted = (base + 11) % len(PARAMETER_DOMAIN)
    unseen = (base + 22) % len(PARAMETER_DOMAIN)
    if pattern == "aba_recurrence":
        return ("A", base) if chronology_index < 384 or chronology_index >= 768 else ("B", shifted)
    if pattern == "abc_unseen_drift":
        if chronology_index < 384:
            return "A", base
        return ("B", shifted) if chronology_index < 768 else ("C", unseen)
    if pattern == "gradual_drift":
        step = max(0, chronology_index - WARMUP_COUNT) // 112
        return f"G{step}", (base + min(step, 7) * 2) % len(PARAMETER_DOMAIN)
    return ("A", base) if chronology_index < 512 else ("B", shifted)


def build_stream_views() -> StreamViews:
    """Generate all 32 streams before any controller can inspect an outcome."""

    public: list[JsonDict] = []
    authority: list[JsonDict] = []
    releases: list[JsonDict] = []
    pattern_counts = dict.fromkeys(PATTERNS, 0)
    for stream_index, stream_seed in enumerate(STREAM_SEEDS):
        pattern = PATTERNS[stream_index % len(PATTERNS)]
        pattern_counts[pattern] += 1
        for chronology_index in range(EVENTS_PER_STREAM):
            repeated_index = chronology_index % 256
            family = FAMILIES[repeated_index % len(FAMILIES)]
            numeric_value = (
                repeated_index * 19 + FAMILIES.index(family) * 7 + stream_index * 13
            ) % 10_000
            base = _stable_parameter(stream_index, family)
            regime_id, parameter = _regime_parameter(pattern, chronology_index, base)
            exact_label = exp7226.exact_label(family, numeric_value, parameter)
            event_id = f"exp7240-s{stream_index + 1:02d}-e{chronology_index:04d}"
            delay = DELAY_SUPPORT[(stream_index + chronology_index) % len(DELAY_SUPPORT)]
            public.append(
                {
                    "event_id": event_id,
                    "stream_id": f"stream-{stream_index + 1:02d}",
                    "chronology_index": chronology_index,
                    "family_id": family,
                    "numeric_value": numeric_value,
                    "public_input": f"family={family};value={numeric_value}",
                }
            )
            authority.append(
                {
                    "event_id": event_id,
                    "stream_id": f"stream-{stream_index + 1:02d}",
                    "stream_seed": stream_seed,
                    "chronology_index": chronology_index,
                    "family_id": family,
                    "numeric_value": numeric_value,
                    "drift_pattern": pattern,
                    "regime_id": regime_id,
                    "hidden_parameter": parameter,
                    "exact_label": exact_label,
                }
            )
            releases.append(
                {
                    "event_id": event_id,
                    "stream_id": f"stream-{stream_index + 1:02d}",
                    "chronology_index": chronology_index,
                    "delay": delay,
                    "observed_label": exact_label,
                }
            )
    manifest = {
        "schema": "carnot.exp7240.public_stream_manifest.v1",
        "stream_count": STREAM_COUNT,
        "events_per_stream": EVENTS_PER_STREAM,
        "warmup_events": WARMUP_COUNT,
        "pending_feedback_capacity": PENDING_CAPACITY,
        "query_ceiling_per_stream": QUERY_CEILING,
        "query_block_size": QUERY_BLOCK_SIZE,
        "delay_support": list(DELAY_SUPPORT),
        "patterns": pattern_counts,
        "public_fields": sorted(public[0]),
        "controller_input_fields": ["event_id", "family_id", "numeric_value"],
        "boundaries_visible_to_controller": False,
        "regimes_visible_to_controller": False,
        "identical_moment_recurrence": True,
        "frozen_before_controller_execution": True,
    }
    return StreamViews(public, authority, releases, manifest)


def _nested_keys(value: Any) -> set[str]:
    """Collect nested field names so hidden authority cannot evade inspection."""

    if isinstance(value, Mapping):
        return set(value) | set().union(*(_nested_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_nested_keys(item) for item in value), set())
    return set()


def public_leakage_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Name public rows that expose private labels, schedules, or regimes."""

    return [
        str(row.get("event_id", "missing_event_id"))
        for row in rows
        if _nested_keys(row) & FORBIDDEN_PUBLIC_FIELDS
    ]


def stream_conformance_errors(views: StreamViews) -> list[str]:
    """Check complete counts, chronology, labels, separation, and moment matching."""

    errors: list[str] = []
    expected = STREAM_COUNT * EVENTS_PER_STREAM
    if not (len(views.public) == len(views.authority) == len(views.releases) == expected):
        errors.append("event_count")
    public_ids = [row.get("event_id") for row in views.public]
    authority_ids = [row.get("event_id") for row in views.authority]
    release_ids = [row.get("event_id") for row in views.releases]
    if public_ids != authority_ids or public_ids != release_ids or len(set(public_ids)) != expected:
        errors.append("event_identity")
    if public_leakage_errors(views.public):
        errors.append("public_authority_leakage")
    if views.manifest.get("patterns") != dict.fromkeys(PATTERNS, 8):
        errors.append("pattern_balance")
    for stream_index in range(STREAM_COUNT):
        start = stream_index * EVENTS_PER_STREAM
        stop = start + EVENTS_PER_STREAM
        stream_public = views.public[start:stop]
        if [row.get("chronology_index") for row in stream_public] != list(range(EVENTS_PER_STREAM)):
            errors.append("chronology")
            break
    for public, truth, release in zip(views.public, views.authority, views.releases):
        if (
            public.get("family_id") != truth.get("family_id")
            or public.get("numeric_value") != truth.get("numeric_value")
            or truth.get("exact_label")
            != exp7226.independent_exact_label(
                str(truth["family_id"]),
                int(truth["numeric_value"]),
                int(truth["hidden_parameter"]),
            )
            or release.get("observed_label") != truth.get("exact_label")
            or release.get("delay") not in DELAY_SUPPORT
        ):
            errors.append("authority_grounding")
            break
    for stream_index in range(0, STREAM_COUNT, len(PATTERNS)):
        rows = views.public[
            stream_index * EVENTS_PER_STREAM : (stream_index + 1) * EVENTS_PER_STREAM
        ]
        first = sorted((row["family_id"], row["numeric_value"]) for row in rows[128:384])
        recurrent = sorted((row["family_id"], row["numeric_value"]) for row in rows[768:1024])
        if first != recurrent:
            errors.append("identical_moment_recurrence")
            break
    return errors


def _stream_receipt(path: Path) -> JsonDict:
    """Bind one separated view to its exact path and hash."""

    return {"path": str(path), "sha256": _sha256_path(path)}


def seal_streams(paths: ExperimentPaths, views: StreamViews) -> dict[str, JsonDict]:
    """Seal public, authority, release, and manifest bytes independently."""

    _write_immutable(paths.public_stream, _jsonl_bytes(views.public))
    _write_immutable(paths.private_authority, _jsonl_bytes(views.authority))
    _write_immutable(paths.release_schedule, _jsonl_bytes(views.releases))
    _write_immutable(paths.public_manifest, transactional.canonical_json_bytes(views.manifest))
    return {
        "public_stream": _stream_receipt(paths.public_stream),
        "private_authority": _stream_receipt(paths.private_authority),
        "release_schedule": _stream_receipt(paths.release_schedule),
        "public_manifest": _stream_receipt(paths.public_manifest),
    }


def _packed_prediction(
    controller: exp7226.PackedBeliefController, event: Mapping[str, Any]
) -> tuple[str, float]:
    """Read one existing packed state through its public prediction API."""

    return controller.predict(event)


def _controller_prediction(
    controller: ArchivedBeliefController, event: Mapping[str, Any]
) -> tuple[str, float]:
    """Read one archived wrapper through the same finite prediction contract."""

    return controller.predict(event)


def _support_release(pending: Mapping[str, Any]) -> JsonDict:
    """Expose one due label without its evaluator regime or generator seed."""

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


def run_fixture_panel(views: StreamViews, *, progress: bool = False) -> FixturePanel:
    """Replay all six arms while every decision remains ahead of current release."""

    started = time.monotonic()
    event_rows: list[JsonDict] = []
    rows: list[JsonDict] = []
    query_rows: list[JsonDict] = []
    final_states: list[JsonDict] = []
    maximum_controller_bytes = 0
    authority = {str(row["event_id"]): row for row in views.authority}
    release_schedule = {str(row["event_id"]): row for row in views.releases}

    for stream_offset, stream_seed in enumerate(STREAM_SEEDS):
        stream_id = f"stream-{stream_offset + 1:02d}"
        events = [row for row in views.public if row["stream_id"] == stream_id]
        frozen = exp7226.PackedBeliefController()
        destructive = exp7226.PackedBeliefController()
        reset = ArchivedBeliefController(archive_cap=0, nomination_mode="none")
        stale = ArchivedBeliefController(nomination_mode="stale")
        validated = ArchivedBeliefController(nomination_mode="validated")
        shuffled = ArchivedBeliefController(nomination_mode="shuffled_validated")
        controllers: dict[str, Any] = {
            "frozen_warmup": frozen,
            "destructive_packed_learner": destructive,
            "reset_relearn_no_archive": reset,
            "unvalidated_stale_archive_reuse": stale,
            "validation_selected_archive": validated,
            "shuffled_nomination_validated": shuffled,
        }
        metrics = {
            arm: {"error": 0, "abstention": 0, "false_accept": 0, "final_error": 0} for arm in ARMS
        }
        pending: list[JsonDict] = []
        actual_queries = 0
        intended_queries = 0
        released_queries = 0
        max_pending = 0
        selected_by_event: dict[str, bool] = {}
        for block_index, offset in enumerate(range(0, EVENTS_PER_STREAM, QUERY_BLOCK_SIZE)):
            block = events[offset : offset + QUERY_BLOCK_SIZE]
            tie_ranks = exp7199.seeded_tie_ranks(stream_seed, block_index, block)
            selected = destructive.select_request(block, tie_ranks)
            selected_by_event[str(selected["event_id"])] = True
            intended_queries += 1
            for event in block:
                event_id = str(event["event_id"])
                chronology_index = int(event["chronology_index"])
                truth = authority[event_id]
                scheduled = release_schedule[event_id]
                will_query = (
                    selected_by_event.get(event_id, False)
                    and actual_queries < QUERY_CEILING
                    and len(pending) < PENDING_CAPACITY
                )
                release_count_before = released_queries
                for arm, controller in controllers.items():
                    state_hash = controller.state_hash()
                    if isinstance(controller, ArchivedBeliefController):
                        prediction, energy = _controller_prediction(controller, event)
                        archive_count = len(controller.archives())
                    else:
                        prediction, energy = _packed_prediction(controller, event)
                        archive_count = 0
                    error = int(prediction != truth["exact_label"])
                    abstention = int(prediction == "abstain")
                    false_accept = int(prediction == "accept" and truth["exact_label"] == "reject")
                    if chronology_index >= WARMUP_COUNT:
                        metrics[arm]["error"] += error
                        metrics[arm]["abstention"] += abstention
                        metrics[arm]["false_accept"] += false_accept
                        if chronology_index >= 768:
                            metrics[arm]["final_error"] += error
                    event_rows.append(
                        {
                            "unit_id": f"{stream_id}:{arm}",
                            "stream_id": stream_id,
                            "arm": arm,
                            "chronology_index": chronology_index,
                            "event_id": event_id,
                            "prediction": prediction,
                            "prediction_energy": energy,
                            "state_hash_before_release": state_hash,
                            "archive_count_before_release": archive_count,
                            "query_selected": will_query,
                            "released_query_count_before": release_count_before,
                            "error": error,
                            "false_accept": false_accept,
                            "abstention": abstention,
                            "prediction_frozen_before_release": True,
                        }
                    )
                    maximum_controller_bytes = max(
                        maximum_controller_bytes, len(controller.state_bytes())
                    )
                if will_query:
                    actual_queries += 1
                    pending.append(
                        {
                            "public": deepcopy(event),
                            "observed_label": truth["exact_label"],
                            "request_index": chronology_index,
                            "release_index": chronology_index + int(scheduled["delay"]),
                        }
                    )
                due = sorted(
                    [row for row in pending if row["release_index"] <= chronology_index],
                    key=lambda row: (row["release_index"], row["request_index"]),
                )
                if due:
                    payload = [_support_release(row) for row in due]
                    if chronology_index < WARMUP_COUNT:
                        frozen.commit_batch(
                            payload,
                            current_cycle=chronology_index,
                            expected_parent_hash=frozen.state_hash(),
                        )
                    destructive.commit_batch(
                        payload,
                        current_cycle=chronology_index,
                        expected_parent_hash=destructive.state_hash(),
                    )
                    for controller in (reset, stale, validated, shuffled):
                        controller.commit_batch(
                            payload,
                            current_cycle=chronology_index,
                            expected_parent_hash=controller.state_hash(),
                        )
                    released_queries += len(due)
                    pending = [row for row in pending if row not in due]
                max_pending = max(max_pending, len(pending))
        pattern = str(authority[events[0]["event_id"]]["drift_pattern"])
        for arm in ARMS:
            metric = metrics[arm]
            rows.append(
                {
                    "unit_id": f"{stream_id}:{arm}",
                    "stream_id": stream_id,
                    "seed": stream_seed,
                    "arm": arm,
                    "metric": "prospective_full_denominator_error",
                    "error": metric["error"],
                    "abstention": metric["abstention"],
                    "false_accept": metric["false_accept"],
                    "event_count": EVENTS_PER_STREAM - WARMUP_COUNT,
                    "error_rate": metric["error"] / (EVENTS_PER_STREAM - WARMUP_COUNT),
                    "final_segment_error": metric["final_error"],
                    "final_segment_event_count": 256,
                    "intended_query_count": intended_queries,
                    "actual_query_count": actual_queries,
                    "released_query_count": released_queries,
                    "pending_at_end": len(pending),
                    "max_pending": max_pending,
                    "drift_pattern": pattern,
                }
            )
        query_rows.append(
            {
                "unit_id": stream_id,
                "stream_id": stream_id,
                "seed": stream_seed,
                "intended_query_count": intended_queries,
                "actual_query_count": actual_queries,
                "released_query_count": released_queries,
                "pending_at_end": len(pending),
                "max_pending": max_pending,
                "adaptive_arms": list(ADAPTIVE_ARMS),
                "matched_selected_feedback": True,
            }
        )
        final_states.append(
            {
                "stream_id": stream_id,
                "validation_selected_archive": validated.state_dict(),
                "shuffled_nomination_validated": shuffled.state_dict(),
            }
        )
        if progress:
            print(
                f"phase 5 benchmark unit {stream_offset + 1}/{STREAM_COUNT} "
                f"elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
    return FixturePanel(event_rows, rows, query_rows, final_states, maximum_controller_bytes)


def _control_release(
    event_id: str,
    family: str,
    numeric_value: int,
    label: str,
    index: int,
) -> JsonDict:
    """Build one released-only support row for isolated transaction controls."""

    return {
        "event_id": event_id,
        "family_id": family,
        "numeric_value": numeric_value,
        "observed_label": label,
        "role": "support",
        "request_index": index,
        "release_index": index,
    }


def _singleton(parameter: int = 0) -> exp7226.PackedBeliefController:
    """Create one exact active hypothesis per finite family for controls."""

    return exp7226.PackedBeliefController.from_survivors(
        {family: {parameter} for family in FAMILIES}
    )


def _commit_control(
    controller: ArchivedBeliefController,
    release: Mapping[str, Any],
    index: int,
    state_path: Path | None = None,
) -> JsonDict:
    """Commit one control against its exact current parent."""

    return controller.commit_batch(
        [release],
        current_cycle=index,
        expected_parent_hash=controller.state_hash(),
        state_path=state_path,
    )


def _validation_control(mode: str) -> tuple[ArchivedBeliefController, JsonDict]:
    """Run one A-B-A sequence whose archive qualifies only after 16 A releases."""

    controller = ArchivedBeliefController.from_active(_singleton(), nomination_mode=mode)
    _commit_control(controller, _control_release(f"{mode}-shift", "lower_bound", 0, "reject", 1), 1)
    receipt: JsonDict = {}
    for index in range(2, 18):
        receipt = _commit_control(
            controller,
            _control_release(f"{mode}-return-{index}", "lower_bound", 0, "accept", index),
            index,
        )
    return controller, receipt


def run_controller_controls(root: Path) -> list[JsonDict]:
    """Apply E2E-007 future-only, restart, retention, validation, and rollback checks."""

    root.mkdir(parents=True, exist_ok=True)
    event = {"event_id": "control-probe", "family_id": "lower_bound", "numeric_value": 0}
    controller = ArchivedBeliefController.from_active(_singleton())
    prediction_before = controller.predict(event)
    parent_hash = controller.state_hash()
    receipt = _commit_control(
        controller,
        _control_release("delayed-contradiction", "lower_bound", 0, "reject", 4),
        4,
    )
    prediction_after = controller.predict(event)
    delayed_row = {
        "control": "future_only_delayed_contradiction",
        "passed": (
            prediction_before[0] == "accept"
            and prediction_after[0] == "reject"
            and receipt["parent_hash"] == parent_hash
            and receipt["operations"][0]["same_event_correction"] is False
        ),
        "prediction_before_release": prediction_before[0],
        "prediction_after_commit": prediction_after[0],
        "parent_hash": parent_hash,
        "child_hash": controller.state_hash(),
    }

    cap_controller = ArchivedBeliefController.from_active(_singleton())
    probes = (
        ("lower_bound", 0, "reject"),
        ("upper_bound", 32, "accept"),
        ("modular_equals", 0, "reject"),
        ("cyclic_window", 0, "reject"),
        ("lower_bound", 0, "accept"),
    )
    for index, (family, value, label) in enumerate(probes, start=1):
        _commit_control(
            cap_controller,
            _control_release(f"cap-{index}", family, value, label, index),
            index,
        )
    cap_row = {
        "control": "archive_cap",
        "passed": len(cap_controller.archives()) == ARCHIVE_CAP,
        "archive_count": len(cap_controller.archives()),
        "archive_cap": ARCHIVE_CAP,
    }

    state_path = root / "restart_controller.json"
    cap_controller.save(state_path)
    restarted = ArchivedBeliefController.load(state_path)
    restart_row = {
        "control": "restart_retention",
        "passed": restarted.state_bytes() == cap_controller.state_bytes(),
        "expected_hash": cap_controller.state_hash(),
        "observed_hash": restarted.state_hash(),
        "archive_count": len(restarted.archives()),
    }

    rollback_parent = cap_controller.state_bytes()
    rollback_prediction = cap_controller.predict(event)
    rollback_receipt = _commit_control(
        cap_controller,
        _control_release("rollback-control", "upper_bound", 32, "reject", 8),
        8,
        state_path,
    )
    rollback_result = cap_controller.rollback(rollback_receipt, state_path=state_path)
    rollback_row = {
        "control": "hash_preserving_rollback",
        "passed": (
            cap_controller.state_bytes() == rollback_parent
            and cap_controller.predict(event) == rollback_prediction
            and state_path.read_bytes() == rollback_parent
        ),
        "parent_hash": rollback_receipt["parent_hash"],
        "restored_hash": rollback_result["restored_state_hash"],
        "byte_identical": rollback_result["byte_identical"],
    }

    validated, validated_receipt = _validation_control("validated")
    shuffled, shuffled_receipt = _validation_control("shuffled_validated")
    validated_nomination = validated.last_nomination_receipt()
    shuffled_nomination = shuffled.last_nomination_receipt()
    validation_row = {
        "control": "released_only_validation_positive",
        "passed": (
            validated_receipt["operations"][0]["reactivated_archive_id"] is not None
            and validated_nomination["validation_window_size"] == VALIDATION_WINDOW
            and validated_nomination["minimum_witnesses"] == MIN_VALIDATION_WITNESSES
        ),
        "selected_archive_id": validated_nomination["selected_archive_id"],
        "validation_window_size": validated_nomination["validation_window_size"],
        "minimum_witnesses": validated_nomination["minimum_witnesses"],
    }
    shuffled_row = {
        "control": "shuffled_nomination_same_final_gate",
        "passed": (
            shuffled_receipt["operations"][0]["reactivated_archive_id"] is not None
            and shuffled_nomination["candidate_count"] == validated_nomination["candidate_count"]
            and shuffled_nomination["final_gate"] == validated_nomination["final_gate"]
        ),
        "validated_candidate_count": validated_nomination["candidate_count"],
        "shuffled_candidate_count": shuffled_nomination["candidate_count"],
        "final_gate": shuffled_nomination["final_gate"],
    }
    return [delayed_row, cap_row, restart_row, rollback_row, validation_row, shuffled_row]


def _controller_contract(maximum_bytes: int) -> JsonDict:
    """Describe the exact opt-in memory and causal update boundary."""

    return {
        "active_state_count": 1,
        "archive_cap": ARCHIVE_CAP,
        "archive_payload": "immutable_survivor_masks",
        "archive_before_contradiction_reset": True,
        "global_inactive_hypothesis_deletion": False,
        "validation_window_releases": VALIDATION_WINDOW,
        "minimum_released_witnesses": MIN_VALIDATION_WITNESSES,
        "maximum_contradictions_for_reactivation": 0,
        "tie_break": ["validation_loss", "creation_order"],
        "query_ceiling_per_stream": QUERY_CEILING,
        "prediction_state": "frozen_before_feedback_release",
        "update_effect": "later_events_only",
        "rollback": "exact_parent_bytes_and_hash",
        "forbidden_regime_signals": [
            "filename",
            "event_number",
            "generator_seed",
            "evaluator_sidecar",
        ],
        "maximum_measured_controller_bytes": maximum_bytes,
    }


def _arm_contract() -> JsonDict:
    """State the six controls and their matched observable information budget."""

    return {
        "arms": list(ARMS),
        "adaptive_arms": list(ADAPTIVE_ARMS),
        "shared_public_events": True,
        "shared_selected_feedback": True,
        "pending_capacity": PENDING_CAPACITY,
        "query_ceiling": QUERY_CEILING,
        "delay_support": list(DELAY_SUPPORT),
        "shuffled_same_archive_content": True,
        "shuffled_same_nomination_count": True,
        "shuffled_same_final_validation_gate": True,
        "science_gate_prepassed": False,
    }


def _hardware_operation_map(maximum_bytes: int) -> JsonDict:
    """Separate measured CPU state from unmeasured future native mappings."""

    return {
        "current": {
            "venue": "host",
            "operations": ["integer_bitset_intersection", "bit_count", "exact_counter"],
            "maximum_controller_bytes": maximum_bytes,
            "measured": True,
        },
        "future": [
            {
                "target": "rust_simd",
                "mapping": "packed survivor masks and vote counts",
                "measured": False,
            },
            {
                "target": "fpga_bram",
                "mapping": "four archive masks plus one active mask",
                "measured": False,
            },
        ],
    }


def _stable_artifact_projection(artifact: Mapping[str, Any]) -> JsonDict:
    """Remove host clocks while retaining every byte that supports readiness."""

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
        "recurrence_fixture_ready_score",
        "stream_manifest_path",
        "stream_receipts",
        "raw_rows_receipt",
        "checkpoint_receipt",
        "control_receipts_path",
        "controller_contract",
        "arm_contract",
        "continuous_self_learning_task",
        "hardware_operation_map",
        "historical_source_receipt",
    )
    return {key: deepcopy(artifact.get(key)) for key in keys}


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash exact settings, sources, separated inputs, raw rows, and aggregates."""

    return transactional.sha256_json(_stable_artifact_projection(artifact))


def _sample_budget(*, complete: bool) -> JsonDict:
    """Return the predeclared attempt, completion, censoring, and stopping rule."""

    planned_events = STREAM_COUNT * EVENTS_PER_STREAM
    planned_arm_rows = planned_events * len(ARMS)
    return {
        "independent_units_planned": STREAM_COUNT,
        "independent_units_attempted": STREAM_COUNT if complete else 0,
        "independent_units_completed": STREAM_COUNT if complete else 0,
        "independent_units_censored": 0,
        "planned_events": planned_events,
        "attempted_events": planned_events if complete else 0,
        "completed_events": planned_events if complete else 0,
        "planned_arm_event_rows": planned_arm_rows,
        "completed_arm_event_rows": planned_arm_rows if complete else 0,
        "stopping_rule": "all 32 predeclared streams once; no outcome-based extension",
    }


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    upstream: Mapping[str, Any],
    paths: ExperimentPaths,
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Create complete provenance before readiness or blocked classification."""

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
        "sample_size_budget": _sample_budget(complete=False),
        "random_seed": {
            "root": RANDOM_SEED,
            "streams": list(STREAM_SEEDS),
            "delays": list(DELAY_SUPPORT),
            "query_tie_schedule": "existing_exp7199_seeded_tie_ranks",
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
        "recurrence_fixture_ready_score": 0,
        "stream_manifest_path": str(paths.public_manifest),
        "stream_receipts": {},
        "raw_rows_receipt": {},
        "checkpoint_receipt": {},
        "control_receipts_path": {},
        "controller_contract": _controller_contract(0),
        "arm_contract": _arm_contract(),
        "continuous_self_learning_task": True,
        "hardware_operation_map": _hardware_operation_map(0),
        "historical_source_receipt": {
            "experiment_id": "exp7227-belief-learning",
            "path": str(DEFAULT_UPSTREAM_ARTIFACT),
            "sha256": source_hashes.get(str(_resolve(REPO_ROOT, DEFAULT_UPSTREAM_ARTIFACT))),
            "status": upstream.get("status"),
            "honest_verdict": upstream.get("honest_verdict"),
            "belief_learning_value_score": upstream.get("belief_learning_value_score"),
        },
        "default_pipeline_modified": False,
        "publication_performed": False,
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    upstream: Mapping[str, Any],
    paths: ExperimentPaths,
) -> JsonDict:
    """Return a terminal row-free artifact for an unchanged external block."""

    now = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks,
        source_hashes,
        upstream,
        paths,
        started_at=now,
        completed_at=now,
        duration_s=0.0,
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _receipt_matches(repo_root: Path, receipt: Mapping[str, Any]) -> bool:
    """Compare one declared evidence hash with the current exact file bytes."""

    path = receipt.get("path")
    expected = receipt.get("sha256")
    return isinstance(path, str) and _sha256_path(_resolve(repo_root, path)) == expected


def validate_artifact(artifact: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> list[str]:
    """Cold-check schema, sources, rows, controls, gates, and stable checksum."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append("required_fields:" + ",".join(missing))
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity")
    if artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE:
        errors.append("date_or_milestone")
    principles = artifact.get("field_principles", {})
    if not isinstance(principles, Mapping) or any(
        field not in principles for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field_principles")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_invocation")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("oracle_declaration")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    for path, expected in artifact.get("source_artifact_hashes", {}).items():
        if expected is None or _sha256_path(_resolve(repo_root, path)) != expected:
            errors.append("source_hash:" + str(path))
            break
    status = artifact.get("status")
    if status == "blocked":
        if (
            artifact.get("rows") != []
            or artifact.get("recurrence_fixture_ready_score") != 0
            or artifact.get("inference_substrate") != "blocked_no_run"
            or artifact.get("inference_substrate_class") != "blocked_no_run"
            or artifact.get("verdict_class") != "blocked"
            or not str(artifact.get("honest_verdict", "")).startswith("blocked_")
            or artifact.get("gate_check_summary", {}).get("passed") is not False
        ):
            errors.append("blocked_contract")
        return errors
    if status != "complete":
        errors.append("status")
        return errors
    if (
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS
        or artifact.get("verdict_class") != "circular_positive"
        or artifact.get("recurrence_fixture_ready_score") != 1
        or artifact.get("gate_check_summary", {}).get("passed") is not True
        or not str(artifact.get("honest_verdict", "")).startswith("complete_")
    ):
        errors.append("complete_contract")
    rows = artifact.get("rows", [])
    if len(rows) != STREAM_COUNT * len(ARMS) or {
        (row.get("stream_id"), row.get("arm")) for row in rows
    } != {(f"stream-{index + 1:02d}", arm) for index in range(STREAM_COUNT) for arm in ARMS}:
        errors.append("aggregate_rows")
    if any(
        row.get("event_count") != EVENTS_PER_STREAM - WARMUP_COUNT
        or row.get("intended_query_count") != QUERY_CEILING
        or row.get("max_pending", PENDING_CAPACITY + 1) > PENDING_CAPACITY
        for row in rows
    ):
        errors.append("row_contract")
    gates = artifact.get("acceptance_gate_results", {})
    readiness = [row for name, row in gates.items() if name != "science_efficacy"]
    if not readiness or any(row.get("pass") is not True for row in readiness):
        errors.append("acceptance_gates")
    if gates.get("science_efficacy", {}).get("pass", False) is not None:
        errors.append("science_gate_prepassed")
    for key in ("stream_receipts",):
        if not artifact.get(key) or not all(
            _receipt_matches(repo_root, receipt) for receipt in artifact[key].values()
        ):
            errors.append(key)
    for key in ("raw_rows_receipt", "checkpoint_receipt", "control_receipts_path"):
        receipt = artifact.get(key, {})
        if not isinstance(receipt, Mapping) or not _receipt_matches(repo_root, receipt):
            errors.append(key)
    budget = artifact.get("sample_size_budget", {})
    if budget.get("completed_arm_event_rows") != STREAM_COUNT * EVENTS_PER_STREAM * len(ARMS):
        errors.append("sample_size_budget")
    return errors


def _require_valid(errors: Sequence[str]) -> None:
    """Prevent publication when any cold validation check fails."""

    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    progress: bool = False,
) -> JsonDict:
    """Run preconditions, generation, six-arm replay, controls, and cold checks."""

    monotonic_start = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: JsonDict = {}
    phase_start = time.monotonic()
    checks, source_hashes, upstream = collect_preconditions(repo_root, paths)
    spans["preconditions"] = time.monotonic() - phase_start
    summary = gate_summary(checks)
    if summary["passed"] is not True:
        artifact = build_blocked_artifact(checks, source_hashes, upstream, paths)
        _require_valid(validate_artifact(artifact, repo_root=repo_root))
        return artifact

    if progress:
        print("phase 3 BEFORE stream generation", flush=True)
    phase_start = time.monotonic()
    views = build_stream_views()
    stream_errors = stream_conformance_errors(views)
    if stream_errors:
        raise ValueError("stream_conformance_failed:" + ",".join(stream_errors))
    stream_receipts = seal_streams(paths, views)
    spans["stream_generation_and_seal"] = time.monotonic() - phase_start
    if progress:
        print("phase 3 AFTER stream generation", flush=True)
        print("phase 4 no model load or generation; invocation counters remain zero", flush=True)
        print("phase 5 BEFORE six-arm CPU benchmark", flush=True)
    phase_start = time.monotonic()
    panel = run_fixture_panel(views, progress=progress)
    spans["six_arm_benchmark"] = time.monotonic() - phase_start
    if progress:
        print("phase 5 AFTER six-arm CPU benchmark", flush=True)
        print("phase 6 BEFORE transactional positive controls", flush=True)
    phase_start = time.monotonic()
    controls = run_controller_controls(paths.checkpoint.parent / "experiment_7240_controls")
    spans["transaction_controls"] = time.monotonic() - phase_start
    if progress:
        print("phase 6 AFTER transactional positive controls", flush=True)

    _write_immutable(paths.raw_rows, _jsonl_bytes(panel.event_rows))
    _atomic_write(
        paths.checkpoint,
        transactional.canonical_json_bytes(
            {
                "schema": "carnot.exp7240.controller_checkpoints.v1",
                "states": panel.final_states,
            }
        ),
    )
    _atomic_write(
        paths.control_receipts,
        transactional.canonical_json_bytes(
            {"schema": "carnot.exp7240.control_receipts.v1", "rows": controls}
        ),
    )
    for path in (
        paths.public_stream,
        paths.private_authority,
        paths.release_schedule,
        paths.public_manifest,
        paths.raw_rows,
        paths.checkpoint,
        paths.control_receipts,
    ):
        source_hashes[str(path)] = _sha256_path(path)

    matched_queries = all(
        len({row["actual_query_count"] for row in panel.rows if row["stream_id"] == stream_id}) == 1
        for stream_id in {row["stream_id"] for row in panel.rows}
    )
    acceptance = {
        "sealed_streams": {
            "expected": 0,
            "actual": len(stream_errors),
            "pass": len(stream_errors) == 0,
        },
        "authority_isolation": {
            "expected": 0,
            "actual": len(public_leakage_errors(views.public)),
            "pass": not public_leakage_errors(views.public),
        },
        "six_arm_rows": {
            "expected": STREAM_COUNT * len(ARMS),
            "actual": len(panel.rows),
            "pass": len(panel.rows) == STREAM_COUNT * len(ARMS),
        },
        "matched_query_budget": {
            "expected": True,
            "actual": matched_queries,
            "pass": matched_queries,
        },
        "archive_and_transaction_controls": {
            "expected": 0,
            "actual": sum(int(row["passed"] is not True) for row in controls),
            "pass": all(row["passed"] is True for row in controls),
        },
        "science_efficacy": {
            "expected": "not_scored_by_fixture",
            "actual": "not_scored_by_fixture",
            "pass": None,
        },
    }
    ready = int(
        all(row["pass"] is True for name, row in acceptance.items() if name != "science_efficacy")
    )
    completed_at = datetime.now(UTC).isoformat()
    duration_s = time.monotonic() - monotonic_start
    artifact = _base_artifact(
        checks,
        source_hashes,
        upstream,
        paths,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "phase_spans_s": spans,
            "rows": panel.rows,
            "sample_size_budget": _sample_budget(complete=True),
            "verdict_class": "circular_positive" if ready else "null",
            "honest_verdict": (
                "complete_circular_positive: recurrence fixture, controller, and positive controls are runnable"
                if ready
                else "complete_null: one or more recurrence fixture readiness controls failed"
            ),
            "acceptance_gate_results": acceptance,
            "recurrence_fixture_ready_score": ready,
            "stream_manifest_path": str(paths.public_manifest),
            "stream_receipts": stream_receipts,
            "raw_rows_receipt": {
                "path": str(paths.raw_rows),
                "sha256": _sha256_path(paths.raw_rows),
                "row_count": len(panel.event_rows),
            },
            "checkpoint_receipt": {
                "path": str(paths.checkpoint),
                "sha256": _sha256_path(paths.checkpoint),
                "state_count": len(panel.final_states),
            },
            "control_receipts_path": {
                "path": str(paths.control_receipts),
                "sha256": _sha256_path(paths.control_receipts),
                "row_count": len(controls),
            },
            "controller_contract": _controller_contract(panel.maximum_controller_bytes),
            "arm_contract": _arm_contract(),
            "hardware_operation_map": _hardware_operation_map(panel.maximum_controller_bytes),
            "query_budget_rows": panel.query_rows,
            "stream_conformance_errors": stream_errors,
            "positive_control_summary": {
                "row_count": len(controls),
                "passed_count": sum(int(row["passed"] is True) for row in controls),
                "failed_count": sum(int(row["passed"] is not True) for row in controls),
            },
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _require_valid(validate_artifact(artifact, repo_root=repo_root))
    return artifact


def write_artifact(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Cold-validate and publish the terminal artifact with one atomic rename."""

    _require_valid(validate_artifact(artifact, repo_root=repo_root))
    _atomic_write(path, transactional.canonical_json_bytes(artifact))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed execution date and optional private result root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CPU fixture and atomically write one terminal result."""

    args = _parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"run_date_must_be_{RUN_DATE}")
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    print("phase 0 BEFORE precondition authentication", flush=True)
    artifact = build_and_seal(REPO_ROOT, paths, progress=True)
    print("phase 0 AFTER precondition authentication", flush=True)
    print("phase 7 BEFORE final cold validation", flush=True)
    _require_valid(validate_artifact(artifact, repo_root=REPO_ROOT))
    print("phase 7 AFTER final cold validation", flush=True)
    print("phase 8 BEFORE atomic terminal write", flush=True)
    write_artifact(paths.artifact, artifact, repo_root=REPO_ROOT)
    print("phase 8 AFTER atomic terminal write", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
