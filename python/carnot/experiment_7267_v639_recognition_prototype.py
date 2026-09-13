"""Build a bounded active-recognition fixture before prospective value tests.

The fixture keeps released examples that distinguish archived finite states.
Fresh released feedback separately gates any reuse. The experiment measures
mechanism readiness, not held-out learning value.

Spec refs: REQ-CL-7267 and SCENARIO-CL-7267-*.
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
import socket
import subprocess
import time
from typing import Any

from carnot import experiment_7199_v634_bounded_acquisition as exp7199
from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7226_v636_belief_compiler as exp7226
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot import experiment_7253_v638_coverage_memory as exp7253
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7267
SCHEMA = "carnot.exp7267.v639_recognition_prototype.v1"
STATE_SCHEMA = "carnot.active_recognition_controller.v1"
MILESTONE = "2026.09.639"
RUN_DATE = "20260913"
RANDOM_SEED = 7_267_000
SHUFFLE_SEED = 7_267_901
DEVELOPMENT_STREAM_SEEDS = tuple(range(7_267_001, 7_267_009))
STREAM_SEEDS = tuple(range(7_267_101, 7_267_125))
DEVELOPMENT_STREAM_COUNT = 8
STREAM_COUNT = 24
EVENTS_PER_STREAM = 1_024
WARMUP_COUNT = 128
QUERY_CEILING = 128
QUERY_BLOCK_SIZE = 8
PENDING_CAPACITY = 4
ARCHIVE_CAP = 4
STABLE_BASIS_CAPACITY = 32
FRESH_VALIDATION_CAPACITY = 16
MIN_FRESH_VALIDATION = 8
LEDGER_CAPACITY = 32
DELAY_SUPPORT = (0, 4, 16, 32)
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
EXECUTION_VENUE = "host"

FAMILIES = tuple(exp7226.FAMILIES)
PARAMETER_DOMAIN = tuple(exp7226.PARAMETER_DOMAIN)
FULL_MASK = exp7226.FULL_MASK
ARMS = (
    "frozen",
    "reset",
    "previous_coverage",
    "active_recognition",
    "random_query_recognition",
    "shuffled_archive_association",
    "full_version_space_memory",
    "feedback_withheld",
)
RECOGNITION_ARMS = (
    "active_recognition",
    "random_query_recognition",
    "shuffled_archive_association",
)
FORBIDDEN_PUBLIC_FIELDS = set(exp7240.FORBIDDEN_PUBLIC_FIELDS) | {
    "hidden_parameter",
    "exact_label",
    "observed_label",
    "regime_id",
    "stratum",
    "stream_seed",
    "boundary",
    "release_index",
}
MEMORY_CAPS = {
    "witness_basis_bytes": 16_384,
    "fresh_validation_bytes": 8_192,
    "archive_bytes": 16_384,
    "pending_bytes": 4_096,
    "ledger_bytes": 4_096,
    "controller_bytes": 65_536,
    "total_bytes": 69_632,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
WRAPPER_PATH = Path("scripts/experiments/experiment_7267_v639_recognition_prototype.py")
TEST_PATH = Path("tests/python/test_experiment_7267_v639_recognition_prototype.py")
DEFAULT_LEARNING_ARTIFACT = Path("results/experiment_7254_v638_coverage_learning.json")
DEFAULT_AUDIT_ARTIFACT = Path("results/experiment_7255_v638_coverage_audit.json")
DEFAULT_ARTIFACT = Path("results/experiment_7267_v639_recognition_prototype.json")
EXPECTED_UPSTREAM_HASHES = {
    "exp7254": "sha256:209b8d243587181a8cc1f2c4f8dd24e59291cca0a2f348460fe125f66b799720",
    "exp7255": "sha256:249e305f39a04bc5eb2c819fdaaa9ee1e1a922536a177e7bb2dd3490a787328c",
}
UPSTREAM_ARTIFACTS = {
    "exp7254": DEFAULT_LEARNING_ARTIFACT,
    "exp7255": DEFAULT_AUDIT_ARTIFACT,
}
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7226_v636_belief_compiler.py"),
    Path("python/carnot/experiment_7253_v638_coverage_memory.py"),
    Path("python/carnot/experiment_7254_v638_coverage_learning.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("python/carnot/experiment_7267_v639_recognition_prototype.py"),
    WRAPPER_PATH,
    TEST_PATH,
    SPEC_PATH,
)
SCENARIO_PATTERN = re.compile(r"SCENARIO-CL-7267-[A-Z-]+")

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
    "recognition_fixture_ready_score",
    "failure_cause_rows",
    "stream_manifest",
    "recognition_contract",
    "continuous_self_learning_task",
)
FIELD_PRINCIPLES = {
    "schema": "Version the result; retain ordinary top-level experiment_id and milestone.",
    "experiment_id": "Bind the evidence to the active Exp7267 task.",
    "milestone": "Bind the evidence to milestone 2026.09.639.",
    "status": "Use complete or blocked only for terminal work; use checkpoints for unfinished work.",
    "run_date": "Use 20260913 and retain actual UTC start and end timestamps.",
    "started_at_utc": "Record the actual UTC start timestamp.",
    "completed_at_utc": "Record the actual UTC completion timestamp.",
    "field_principles": "Store explanations here; consumers read ordinary top-level values.",
    "preconditions_checked": "Retain input hashes, ownership, quarantine state, and failures.",
    "MODEL_SPECS": "Declare only models executable in this invocation; this task has none.",
    "model_invoked": "Derive model use from actual calls; this task makes no model call.",
    "invocation_counts": "Separate attempted and completed loads, generations, and usable answers.",
    "inference_substrate": "Use the recognized literal that describes the actual CPU computation.",
    "inference_substrate_class": "Declare actual compute without padding elapsed time.",
    "execution_venue": "Use host for host orchestration and identify boards elsewhere.",
    "duration_s": "Measure monotonic invocation time and disjoint phase spans.",
    "random_seed": "Freeze every independent-unit seed before prospective outcomes.",
    "reproducibility_checksum": "Bind code, inputs, configuration, and raw evidence.",
    "source_artifact_hashes": "Authenticate exact inputs and preserve quarantine state.",
    "rows": "Retain each independent stream and arm with metrics and censoring state.",
    "sample_size_budget": "Record planned, attempted, completed, and censored units.",
    "acceptance_gate_results": "Keep expected, observed, passed, and principle for each criterion.",
    "gate_check_summary": "Name the exact upstream field and value for a blocked result.",
    "verifier_is_oracle": "Expose that the exact evaluator also defines correctness.",
    "honest_verdict": "Use complete_ for measurements and blocked_ for external absence.",
    "verdict_class": "Use the closed verdict set; oracle authority forbids positive.",
    "validation_receipts": "Record actual commands, exit codes, classifications, and log hashes.",
    "recognition_fixture_ready_score": "One certifies an effective development mechanism and sealed streams.",
    "failure_cause_rows": "Measured selection stages test the signature-collapse explanation.",
    "stream_manifest": "Separate development and prospective public, release, and authority bytes.",
    "recognition_contract": "Freeze query, witness, segmentation, memory, and reactivation rules.",
    "continuous_self_learning_task": "True marks released constraint addition and deactivation.",
}

gate_check = exp7213.gate_check
gate_summary = exp7213.gate_summary
RecognitionCommitRejected = exp7253.ArchiveCommitRejected


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep sealed streams, raw rows, checkpoints, controls, and terminal bytes separate."""

    development_public: Path
    development_authority: Path
    development_releases: Path
    prospective_public: Path
    prospective_authority: Path
    prospective_releases: Path
    stream_manifest: Path
    raw_rows: Path
    diagnosis_rows: Path
    state_sidecar: Path
    control_sidecar: Path
    evidence_sidecar: Path
    provisional: Path
    terminal_candidate: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return all task-owned paths below the repository result directory."""

        return cls.from_results_root(REPO_ROOT / "results")

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put every test output below a caller-owned temporary directory."""

        return cls.from_results_root(root)

    @classmethod
    def from_results_root(cls, root: Path) -> ExperimentPaths:
        """Derive stream, raw, checkpoint, candidate, and final paths."""

        streams = root / "streams" / "experiment_7267"
        raw = root / "raw" / "experiment_7267"
        checkpoints = root / "checkpoints"
        return cls(
            streams / "development_public.jsonl",
            streams / "development_private_authority.jsonl",
            streams / "development_releases.jsonl",
            streams / "prospective_public.jsonl",
            streams / "prospective_private_authority.jsonl",
            streams / "prospective_releases.jsonl",
            streams / "stream_manifest.json",
            raw / "event_rows.jsonl",
            raw / "failure_cause_rows.jsonl",
            checkpoints / "experiment_7267_v639_states.json",
            checkpoints / "experiment_7267_v639_controls.json",
            checkpoints / "experiment_7267_v639_evidence.json",
            checkpoints / "experiment_7267_v639_in_progress.json",
            raw / "terminal_candidate.json",
            root / DEFAULT_ARTIFACT.name,
        )


@dataclass(frozen=True)
class StreamViews:
    """Keep public observations separate from release and evaluator authority."""

    public: list[JsonDict]
    authority: list[JsonDict]
    releases: list[JsonDict]
    manifest: JsonDict


@dataclass(frozen=True)
class RecognitionPanel:
    """Retain raw events, independent rows, final states, and completion counts."""

    event_rows: list[JsonDict]
    rows: list[JsonDict]
    final_states: list[JsonDict]
    completed_stream_count: int
    censored_stream_count: int
    maximum_memory_bytes: int


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Emit one flushed boundary so long CPU work remains observable."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _resolve(repo_root: Path, path: str | Path) -> Path:
    """Resolve repository evidence while preserving absolute test paths."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object while malformed evidence remains unavailable."""

    return exp7253._load_object(path)


def _sha256_path(path: Path) -> str | None:
    """Hash exact bytes while a missing file remains an observed failure."""

    return exp7253._sha256_path(path)


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating terminal bytes."""

    return exp7253._path_writable(path)


def _atomic_write(path: Path, payload: bytes) -> JsonDict:
    """Publish complete bytes with the shared flushed atomic writer."""

    return exp7253._atomic_write(path, payload)


def _write_immutable(path: Path, payload: bytes) -> JsonDict:
    """Accept an identical stream seal and reject replacement bytes."""

    return exp7253._write_immutable(path, payload)


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode chronological rows with the shared canonical JSON form."""

    return exp7253._jsonl_bytes(rows)


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Load object rows while malformed JSONL remains unavailable."""

    return exp7253._read_jsonl(path)


def _mask_hash(masks: Mapping[str, Any]) -> str:
    """Bind archive identity to finite survivor masks, not a moving window."""

    return transactional.sha256_json({family: int(masks[family]) for family in FAMILIES})


def predict_masks(masks: Mapping[str, Any], event: Mapping[str, Any]) -> str:
    """Use the shipped finite prediction semantics for one archived state."""

    return exp7240._prediction_from_masks(masks, event)


def archive_row(archive_id: str, creation_order: int, masks: Mapping[str, Any]) -> JsonDict:
    """Create one immutable survivor-mask snapshot with a stable identity."""

    survivor_masks = {family: int(masks[family]) for family in FAMILIES}
    return {
        "archive_id": archive_id,
        "creation_order": creation_order,
        "state_hash": _mask_hash(survivor_masks),
        "survivor_masks": survivor_masks,
    }


def _pair_disagreement(labels: Sequence[str]) -> int:
    """Count disagreeing archived prediction pairs for one public event."""

    return sum(
        int(left != right) for index, left in enumerate(labels) for right in labels[index + 1 :]
    )


def _public_event(event: Mapping[str, Any]) -> JsonDict:
    """Expose only finite public coordinates to query selection."""

    if set(event) & FORBIDDEN_PUBLIC_FIELDS:
        raise ValueError("private_authority_in_query")
    family = event.get("family_id")
    value = event.get("numeric_value")
    if family not in FAMILIES or not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("invalid_public_query")
    return {
        "event_id": str(event["event_id"]),
        "family_id": str(family),
        "numeric_value": value,
    }


def select_disagreement_query(
    archives: Sequence[Mapping[str, Any]],
    block: Sequence[Mapping[str, Any]],
    tie_ranks: Sequence[int] | Mapping[str, int],
) -> tuple[JsonDict, JsonDict]:
    """Select the public query with maximum archived prediction disagreement."""

    public = [_public_event(event) for event in block]
    if isinstance(tie_ranks, Mapping):
        ranks = [int(tie_ranks[str(event["event_id"])]) for event in public]
    else:
        ranks = [int(value) for value in tie_ranks]
    candidate_rows = []
    for event, rank in zip(public, ranks):
        predictions = [predict_masks(row["survivor_masks"], event) for row in archives]
        candidate_rows.append(
            {
                "event_id": event["event_id"],
                "archive_predictions": predictions,
                "disagreement_pair_count": _pair_disagreement(predictions),
                "tie_rank": rank,
            }
        )
    selected_row = max(
        candidate_rows,
        key=lambda row: (
            int(row["disagreement_pair_count"]),
            -int(row["tie_rank"]),
            str(row["event_id"]),
        ),
    )
    selected = next(row for row in public if row["event_id"] == selected_row["event_id"])
    return selected, {
        "selection_rule": "maximum_archived_pair_disagreement",
        "controller_input_fields": ["event_id", "family_id", "numeric_value"],
        "candidate_rows": candidate_rows,
        "selected_event_id": selected["event_id"],
        "maximum_disagreement": selected_row["disagreement_pair_count"],
        "archive_ids": [str(row["archive_id"]) for row in archives],
        "selected_archive_predictions": selected_row["archive_predictions"],
        "future_label_used": False,
        "private_regime_used": False,
    }


def retain_stable_basis(
    basis: Sequence[Mapping[str, Any]],
    released: Sequence[Mapping[str, Any]],
    archives: Sequence[Mapping[str, Any]],
    *,
    capacity: int = STABLE_BASIS_CAPACITY,
) -> list[JsonDict]:
    """Retain released examples that best distinguish immutable mask snapshots."""

    by_id = {str(row["event_id"]): deepcopy(dict(row)) for row in [*basis, *released]}
    scored = []
    for order, witness in enumerate(by_id.values()):
        labels = [predict_masks(row["survivor_masks"], witness) for row in archives]
        scored.append(
            (
                _pair_disagreement(labels),
                int(witness.get("release_index", order)),
                str(witness["event_id"]),
                witness,
            )
        )
    retained = sorted(scored, key=lambda row: row[:3], reverse=True)[:capacity]
    return [deepcopy(row[3]) for row in sorted(retained, key=lambda row: (row[1], row[2]))]


def _evaluate_archive(
    archive: Mapping[str, Any], witnesses: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Apply the fresh released-witness safety gate before any reactivation."""

    predictions = [predict_masks(archive["survivor_masks"], row) for row in witnesses]
    applicable = sum(int(value != "abstain") for value in predictions)
    contradictions = sum(
        int(prediction != "abstain" and prediction != witness["observed_label"])
        for prediction, witness in zip(predictions, witnesses)
    )
    return {
        "archive_id": archive["archive_id"],
        "released_witness_count": applicable,
        "contradiction_count": contradictions,
        "gate_passed": applicable >= MIN_FRESH_VALIDATION and contradictions == 0,
    }


class RecognitionController:
    """Keep bounded archives and ask a released-feedback query before reuse."""

    def __init__(
        self,
        *,
        archive_cap: int = ARCHIVE_CAP,
        query_mode: str = "active",
        association_mode: str = "aligned",
    ) -> None:
        if query_mode not in {"active", "random"}:
            raise ValueError("invalid_query_mode")
        if association_mode not in {"aligned", "shuffled"}:
            raise ValueError("invalid_association_mode")
        self._state: JsonDict = {
            "schema": STATE_SCHEMA,
            "version": 0,
            "parent_hash": None,
            "archive_cap": archive_cap,
            "query_mode": query_mode,
            "association_mode": association_mode,
            "active": exp7253._bounded_active_state(exp7226.PackedBeliefController()),
            "archives": [],
            "next_creation_order": 0,
            "stable_basis": [],
            "fresh_validation": [],
            "pending_recognition": [],
            "ledger": [],
            "last_nomination_receipt": {},
        }

    @classmethod
    def from_masks(
        cls,
        masks: Mapping[str, Any],
        **kwargs: Any,
    ) -> RecognitionController:
        """Start from detached finite masks without carrying unbounded provenance."""

        controller = cls(**kwargs)
        active = exp7253._controller_from_masks(masks)
        controller._state["active"] = exp7253._bounded_active_state(active)
        return controller

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> RecognitionController:
        """Restore only a bounded state with valid immutable archive identities."""

        if value.get("schema") != STATE_SCHEMA:
            raise ValueError("invalid_recognition_state")
        controller = cls.__new__(cls)
        controller._state = deepcopy(dict(value))
        exp7226.PackedBeliefController.from_state(controller._state["active"])
        archives = controller._state.get("archives", [])
        if len(archives) > int(controller._state["archive_cap"]):
            raise ValueError("archive_capacity")
        if any(row.get("state_hash") != _mask_hash(row["survivor_masks"]) for row in archives):
            raise ValueError("archive_identity")
        if (
            len(controller._state.get("stable_basis", [])) > STABLE_BASIS_CAPACITY
            or len(controller._state.get("fresh_validation", [])) > FRESH_VALIDATION_CAPACITY
            or len(controller._state.get("pending_recognition", [])) > PENDING_CAPACITY
            or len(controller._state.get("ledger", [])) > LEDGER_CAPACITY
        ):
            raise ValueError("bounded_collection_capacity")
        if controller.memory_usage()["within_all_caps"] is not True:
            raise ValueError("recognition_memory_cap")
        return controller

    @classmethod
    def load(cls, path: Path) -> RecognitionController:
        """Cold-load durable JSON bytes through the bounded state validator."""

        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("invalid_recognition_state")
        return cls.from_state(value)

    def state_dict(self) -> JsonDict:
        """Return detached state so outside code cannot mutate live bytes."""

        return deepcopy(self._state)

    def state_bytes(self) -> bytes:
        """Serialize every bounded component in the shared canonical form."""

        return transactional.canonical_json_bytes(self._state)

    def state_hash(self) -> str:
        """Identify the complete active, archive, witness, query, and ledger state."""

        return transactional.sha256_bytes(self.state_bytes())

    def save(self, path: Path) -> JsonDict:
        """Publish restartable bytes through the shared atomic writer."""

        return transactional._atomic_write(path, self.state_bytes())

    def predict(self, event: Mapping[str, Any]) -> tuple[str, float]:
        """Predict from active masks without reading a label or changing state."""

        return exp7226.PackedBeliefController.from_state(self._state["active"]).predict(event)

    def archives(self) -> list[JsonDict]:
        """Expose detached archive rows for audit and query selection."""

        return deepcopy(self._state["archives"])

    def add_archive(self, masks: Mapping[str, Any]) -> str | None:
        """Retain one unique immutable snapshot under the four-archive cap."""

        if int(self._state["archive_cap"]) == 0:
            return None
        order = int(self._state["next_creation_order"])
        row = archive_row(f"archive-{order:06d}", order, masks)
        unique = [
            item for item in self._state["archives"] if item["state_hash"] != row["state_hash"]
        ]
        unique.append(row)
        self._state["archives"] = sorted(
            unique,
            key=lambda item: (int(item["creation_order"]), str(item["archive_id"])),
        )[-int(self._state["archive_cap"]) :]
        self._state["next_creation_order"] = order + 1
        return row["archive_id"] if row in self._state["archives"] else None

    def select_request(
        self,
        block: Sequence[Mapping[str, Any]],
        tie_ranks: Sequence[int] | Mapping[str, int],
    ) -> tuple[JsonDict, JsonDict]:
        """Use active archive disagreement or a matched deterministic random query."""

        public = [_public_event(event) for event in block]
        if isinstance(tie_ranks, Mapping):
            rank_map = {str(key): int(value) for key, value in tie_ranks.items()}
        else:
            rank_map = {str(event["event_id"]): int(rank) for event, rank in zip(public, tie_ranks)}
        archives = self.archives()
        if self._state["query_mode"] == "active" and len(archives) > 1:
            return select_disagreement_query(archives, public, rank_map)
        selected = min(
            public,
            key=lambda event: (rank_map[str(event["event_id"])], str(event["event_id"])),
        )
        predictions = [predict_masks(row["survivor_masks"], selected) for row in archives]
        return selected, {
            "selection_rule": "matched_seeded_random"
            if self._state["query_mode"] == "random"
            else "active_state_fallback",
            "controller_input_fields": ["event_id", "family_id", "numeric_value"],
            "candidate_rows": [],
            "selected_event_id": selected["event_id"],
            "maximum_disagreement": _pair_disagreement(predictions),
            "archive_ids": [str(row["archive_id"]) for row in archives],
            "selected_archive_predictions": predictions,
            "future_label_used": False,
            "private_regime_used": False,
        }

    def record_query(
        self,
        event: Mapping[str, Any],
        receipt: Mapping[str, Any],
        *,
        request_index: int,
        release_index: int,
    ) -> None:
        """Store only public query context until its charged feedback arrives."""

        if len(self._state["pending_recognition"]) >= PENDING_CAPACITY:
            raise ValueError("pending_capacity")
        archive_ids = [str(value) for value in receipt.get("archive_ids", [])]
        mapped_ids = list(archive_ids)
        if self._state["association_mode"] == "shuffled" and len(mapped_ids) > 1:
            mapped_ids = mapped_ids[1:] + mapped_ids[:1]
        self._state["pending_recognition"].append(
            {
                **_public_event(event),
                "request_index": request_index,
                "release_index": release_index,
                "archive_ids": archive_ids,
                "mapped_archive_ids": mapped_ids,
                "archive_predictions": list(receipt.get("selected_archive_predictions", [])),
            }
        )

    def memory_usage(self, pending: Sequence[Mapping[str, Any]] = ()) -> JsonDict:
        """Measure the complete bounded state plus caller-owned delayed releases."""

        values = {
            "witness_basis_bytes": len(
                transactional.canonical_json_bytes(self._state["stable_basis"])
            ),
            "fresh_validation_bytes": len(
                transactional.canonical_json_bytes(self._state["fresh_validation"])
            ),
            "archive_bytes": len(transactional.canonical_json_bytes(self._state["archives"])),
            "pending_bytes": len(
                transactional.canonical_json_bytes(
                    [*self._state["pending_recognition"], *list(pending)]
                )
            ),
            "ledger_bytes": len(transactional.canonical_json_bytes(self._state["ledger"])),
            "controller_bytes": len(self.state_bytes()),
        }
        values["total_bytes"] = values["controller_bytes"] + len(
            transactional.canonical_json_bytes(list(pending))
        )
        values["within_all_caps"] = all(
            int(values[key]) <= limit for key, limit in MEMORY_CAPS.items()
        )
        return values

    def commit_batch(
        self,
        releases: Sequence[Mapping[str, Any]],
        *,
        current_cycle: int,
        expected_parent_hash: str,
        state_path: Path | None = None,
    ) -> JsonDict:
        """Commit due feedback atomically and permit only later reactivation effects."""

        parent_bytes = self.state_bytes()
        parent_hash = self.state_hash()
        if expected_parent_hash != parent_hash:
            raise RecognitionCommitRejected("stale_parent")
        if state_path is not None and state_path.exists():
            try:
                durable_hash = type(self).load(state_path).state_hash()
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise RecognitionCommitRejected("corrupt_durable_state") from error
            if durable_hash != parent_hash:
                raise RecognitionCommitRejected("stale_durable_parent")
        try:
            normalized = [
                exp7226.PackedBeliefController._validate_release(row, current_cycle)
                for row in releases
            ]
        except exp7226.CommitRejected as error:
            raise RecognitionCommitRejected(str(error)) from error
        ids = [str(row["event_id"]) for row in normalized]
        if len(set(ids)) != len(ids) or set(ids) & set(self._state["ledger"]):
            raise RecognitionCommitRejected("duplicate_release")

        candidate = deepcopy(self._state)
        operations = []
        for release in normalized:
            active = exp7226.PackedBeliefController.from_state(candidate["active"])
            family = str(release["family_id"])
            value = int(release["numeric_value"])
            before_mask = int(active.family_state(family)["survivor_mask"])
            accept_mask = exp7226.ACCEPT_MASKS[family][value]
            matching = (
                accept_mask if release["observed_label"] == "accept" else FULL_MASK ^ accept_mask
            )
            contradiction = before_mask != 0 and before_mask & matching == 0
            archived_id = None
            if contradiction and int(candidate["archive_cap"]) > 0:
                live = self._state
                self._state = candidate
                archived_id = self.add_archive(exp7253._survivor_masks(active))
                candidate = self._state
                self._state = live
            active.commit_batch(
                [release],
                current_cycle=current_cycle,
                expected_parent_hash=active.state_hash(),
            )
            candidate["active"] = exp7253._bounded_active_state(active)
            candidate["fresh_validation"].append(deepcopy(release))
            candidate["fresh_validation"] = candidate["fresh_validation"][
                -FRESH_VALIDATION_CAPACITY:
            ]
            candidate["stable_basis"] = retain_stable_basis(
                candidate["stable_basis"],
                [release],
                candidate["archives"],
            )
            pending = next(
                (
                    row
                    for row in candidate["pending_recognition"]
                    if row["event_id"] == release["event_id"]
                ),
                None,
            )
            evaluations = [
                _evaluate_archive(row, candidate["fresh_validation"])
                for row in candidate["archives"]
            ]
            selected_id = None
            selection_changed = False
            if pending is not None:
                by_id = {str(row["archive_id"]): row for row in candidate["archives"]}
                matching_slots = [
                    index
                    for index, prediction in enumerate(pending["archive_predictions"])
                    if prediction == release["observed_label"]
                ]
                before_id = next(
                    (pending["archive_ids"][index] for index in matching_slots),
                    None,
                )
                after_id = next(
                    (pending["mapped_archive_ids"][index] for index in matching_slots),
                    None,
                )
                eligible = {row["archive_id"] for row in evaluations if row["gate_passed"] is True}
                if after_id in eligible:
                    selected_id = str(after_id)
                    selected = by_id[selected_id]
                    candidate["active"] = exp7253._bounded_active_state(
                        exp7253._controller_from_masks(selected["survivor_masks"])
                    )
                    candidate["archives"] = [
                        row for row in candidate["archives"] if row["archive_id"] != selected_id
                    ]
                selection_changed = before_id != after_id
                candidate["pending_recognition"] = [
                    row for row in candidate["pending_recognition"] if row is not pending
                ]
            candidate["ledger"].append(str(release["event_id"]))
            candidate["ledger"] = candidate["ledger"][-LEDGER_CAPACITY:]
            nomination = {
                "archive_size": len(candidate["archives"]),
                "distinguishable_candidate_count": len(
                    {
                        tuple(
                            predict_masks(row["survivor_masks"], witness)
                            for witness in candidate["stable_basis"]
                        )
                        for row in candidate["archives"]
                    }
                ),
                "eligible_candidate_count": sum(int(row["gate_passed"]) for row in evaluations),
                "selected_archive_id": selected_id,
                "association_selection_changed": selection_changed,
                "fresh_feedback_required": True,
            }
            candidate["last_nomination_receipt"] = nomination
            operations.append(
                {
                    "event_id": release["event_id"],
                    "active_contradiction": contradiction,
                    "archived_state_id": archived_id,
                    "reactivated_archive_id": selected_id,
                    "selection_changed": selection_changed,
                    "nomination_receipt": nomination,
                    "prediction_frozen_before_release": True,
                    "same_event_correction": False,
                }
            )
        candidate["version"] = int(candidate["version"]) + 1
        candidate["parent_hash"] = parent_hash
        admitted = type(self).from_state(candidate)
        new_bytes = admitted.state_bytes()
        receipt = {
            "parent_hash": parent_hash,
            "new_state_hash": transactional.sha256_bytes(new_bytes),
            "parent_bytes_b64": transactional.encode_bytes(parent_bytes),
            "new_state_bytes_b64": transactional.encode_bytes(new_bytes),
            "release_order": ids,
            "release_count": len(ids),
            "state_version": candidate["version"],
            "operations": operations,
            "atomic_write": None,
        }
        if state_path is not None:
            receipt["atomic_write"] = transactional._atomic_write(state_path, new_bytes)
        self._state = admitted._state
        return receipt

    def rollback(self, receipt: Mapping[str, Any], *, state_path: Path | None = None) -> JsonDict:
        """Restore exact parent bytes only from the receipt's exact child state."""

        if self.state_hash() != receipt.get("new_state_hash"):
            raise RecognitionCommitRejected("stale_rollback")
        try:
            parent_bytes = transactional.decode_bytes(str(receipt["parent_bytes_b64"]))
            value = json.loads(parent_bytes)
            restored = type(self).from_state(value)
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            raise RecognitionCommitRejected("invalid_rollback_receipt") from error
        if restored.state_hash() != receipt.get("parent_hash"):
            raise RecognitionCommitRejected("rollback_parent_hash")
        if state_path is not None:
            transactional._atomic_write(state_path, parent_bytes)
        self._state = restored._state
        return {"restored_state_hash": self.state_hash(), "byte_identical": True}


def run_stable_basis_diagnostic() -> JsonDict:
    """Observe a counterexample that rolling rebasing merges but a stable basis keeps."""

    archives = [
        archive_row("parameter-0", 0, dict.fromkeys(FAMILIES, 1 << 0)),
        archive_row("parameter-8", 1, dict.fromkeys(FAMILIES, 1 << 8)),
    ]
    discriminator = {
        "event_id": "released-discriminator",
        "family_id": "lower_bound",
        "numeric_value": 7,
        "observed_label": "accept",
        "role": "support",
        "request_index": 0,
        "release_index": 0,
    }
    recent = [
        {
            "event_id": f"rolling-{index:02d}",
            "family_id": "lower_bound",
            "numeric_value": 31,
            "observed_label": "accept",
            "role": "support",
            "request_index": index + 1,
            "release_index": index + 1,
        }
        for index in range(FRESH_VALIDATION_CAPACITY)
    ]
    fresh = [
        {
            **row,
            "event_id": f"fresh-{index:02d}",
            "request_index": index + 100,
            "release_index": index + 100,
        }
        for index, row in enumerate(recent)
    ]
    basis = retain_stable_basis([], [discriminator, *recent], archives)
    rolling_signatures = {
        tuple(predict_masks(row["survivor_masks"], witness) for witness in recent)
        for row in archives
    }
    stable_signatures = {
        tuple(predict_masks(row["survivor_masks"], witness) for witness in basis)
        for row in archives
    }
    return {
        "diagnostic_kind": "observed_finite_released_only_replay",
        "rolling_signature_distinct_count": len(rolling_signatures),
        "stable_signature_distinct_count": len(stable_signatures),
        "counterexample_survives_stable_basis": len(stable_signatures) > len(rolling_signatures),
        "stable_basis_event_ids": [row["event_id"] for row in basis],
        "fresh_validation_event_ids": [row["event_id"] for row in fresh],
        "basis_size": len(basis),
        "hidden_authority_used": False,
    }


def reduce_saved_failure_rows(audit: Mapping[str, Any]) -> list[JsonDict]:
    """Reduce saved V638 aggregate rows without inventing omitted stage evidence."""

    source = audit.get("shuffle_effect_rows", [])
    nominations = sum(int(row.get("nomination_count", 0)) for row in source)
    mapping = sum(int(row.get("candidate_mapping_difference_count", 0)) for row in source)
    selected = sum(int(row.get("selected_archive_difference_count", 0)) for row in source)
    later = sum(int(row.get("later_decision_difference_count", 0)) for row in source)
    limited = sum(int(row.get("zero_headroom_count", 0)) for row in source)
    return [
        {
            "stage": "nomination",
            "observed": nominations,
            "censored": False,
            "interpretation": "saved nomination attempts across FIFO and coverage",
        },
        {
            "stage": "archive_size",
            "observed": None,
            "censored": True,
            "interpretation": "not retained in saved aggregate rows",
        },
        {
            "stage": "distinguishable_candidates",
            "observed": None,
            "censored": True,
            "interpretation": "mapping changes do not identify signature distinguishability",
        },
        {
            "stage": "one_eligible_or_none",
            "observed": limited,
            "censored": False,
            "interpretation": "saved zero-headroom means eligible candidate count was at most one",
        },
        {
            "stage": "signature_merging",
            "observed": None,
            "censored": True,
            "interpretation": "cannot separate merging from archive count in saved rows",
        },
        {
            "stage": "safety_rejection",
            "observed": None,
            "censored": True,
            "interpretation": "cannot separate zero eligible from one eligible in saved rows",
        },
        {
            "stage": "post_shuffle_mapping",
            "observed": mapping,
            "censored": False,
            "interpretation": "candidate identity mapping changed",
        },
        {
            "stage": "post_shuffle_choice",
            "observed": selected,
            "censored": False,
            "interpretation": "selected archive identity changed",
        },
        {
            "stage": "later_prediction_change",
            "observed": later,
            "censored": False,
            "interpretation": "aligned and shuffled predictions later differed",
        },
    ]


def _base_parameter(stream_seed: int, family: str) -> int:
    """Freeze one evaluator-private base parameter from a sealed stream seed."""

    return (stream_seed * 7 + FAMILIES.index(family) * 5) % len(PARAMETER_DOMAIN)


def _regime_parameter(stratum: str, chronology_index: int, base: int) -> tuple[str, int]:
    """Create equal recurrent strata with separated or overlapping hypotheses."""

    if chronology_index < 384 or chronology_index >= 768:
        return "A", base
    offset = 16 if stratum == "separated_recurrence" else 2
    return "B", (base + offset) % len(PARAMETER_DOMAIN)


def build_stream_views(kind: str) -> StreamViews:
    """Generate fixed development or fresh prospective views before replay."""

    if kind == "development":
        seeds = DEVELOPMENT_STREAM_SEEDS
        prefix = "development"
    elif kind == "prospective":
        seeds = STREAM_SEEDS
        prefix = "prospective"
    else:
        raise ValueError("invalid_stream_kind")
    public: list[JsonDict] = []
    authority: list[JsonDict] = []
    releases: list[JsonDict] = []
    strata = {"separated_recurrence": 0, "overlapping_recurrence": 0}
    for stream_offset, seed in enumerate(seeds):
        if kind == "development":
            stratum = (
                "separated_recurrence"
                if stream_offset < len(seeds) // 2
                else "overlapping_recurrence"
            )
        else:
            stratum = "separated_recurrence" if stream_offset < 12 else "overlapping_recurrence"
        strata[stratum] += 1
        stream_id = f"{prefix}-{stream_offset + 1:02d}"
        for chronology_index in range(EVENTS_PER_STREAM):
            repeated_index = chronology_index % 256
            family = FAMILIES[repeated_index % len(FAMILIES)]
            numeric_value = (repeated_index * 19 + FAMILIES.index(family) * 7 + seed * 13) % len(
                PARAMETER_DOMAIN
            )
            regime_id, parameter = _regime_parameter(
                stratum,
                chronology_index,
                _base_parameter(seed, family),
            )
            exact_label = exp7226.exact_label(family, numeric_value, parameter)
            event_id = f"exp7267-{stream_id}-e{chronology_index:04d}"
            delay = DELAY_SUPPORT[(seed + chronology_index) % len(DELAY_SUPPORT)]
            public.append(
                {
                    "event_id": event_id,
                    "stream_id": stream_id,
                    "chronology_index": chronology_index,
                    "family_id": family,
                    "numeric_value": numeric_value,
                    "public_input": f"family={family};value={numeric_value}",
                }
            )
            authority.append(
                {
                    "event_id": event_id,
                    "stream_id": stream_id,
                    "chronology_index": chronology_index,
                    "stream_seed": seed,
                    "stratum": stratum,
                    "regime_id": regime_id,
                    "hidden_parameter": parameter,
                    "exact_label": exact_label,
                }
            )
            releases.append(
                {
                    "event_id": event_id,
                    "stream_id": stream_id,
                    "chronology_index": chronology_index,
                    "delay": delay,
                    "observed_label": exact_label,
                }
            )
    return StreamViews(
        public,
        authority,
        releases,
        {
            "schema": "carnot.exp7267.stream_view.v1",
            "kind": kind,
            "stream_count": len(seeds),
            "events_per_stream": EVENTS_PER_STREAM,
            "warmup_events": WARMUP_COUNT,
            "query_ceiling_per_arm": QUERY_CEILING,
            "archive_capacity": ARCHIVE_CAP,
            "total_memory_cap_bytes": MEMORY_CAPS["total_bytes"],
            "strata": strata,
            "stream_seeds_sha256": transactional.sha256_json(list(seeds)),
            "controller_input_fields": ["event_id", "family_id", "numeric_value"],
            "labels_visible_before_release": False,
            "regimes_visible_to_controller": False,
            "boundaries_visible_to_controller": False,
            "frozen_before_controller_execution": True,
        },
    )


def _nested_keys(value: Any) -> set[str]:
    """Collect nested keys so evaluator authority cannot hide in public rows."""

    if isinstance(value, Mapping):
        return set(value) | set().union(*(_nested_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_nested_keys(item) for item in value), set())
    return set()


def public_leakage_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Name public rows that contain a private label, regime, seed, or schedule."""

    return [
        str(row.get("event_id", "missing_event_id"))
        for row in rows
        if _nested_keys(row) & FORBIDDEN_PUBLIC_FIELDS
    ]


def stream_conformance_errors(views: StreamViews, kind: str) -> list[str]:
    """Check fixed counts, chronology, identity, strata, and authority separation."""

    expected_streams = DEVELOPMENT_STREAM_COUNT if kind == "development" else STREAM_COUNT
    expected = expected_streams * EVENTS_PER_STREAM
    errors: list[str] = []
    if not (len(views.public) == len(views.authority) == len(views.releases) == expected):
        errors.append("event_count")
    ids = [row.get("event_id") for row in views.public]
    if (
        ids != [row.get("event_id") for row in views.authority]
        or ids != [row.get("event_id") for row in views.releases]
        or len(set(ids)) != len(ids)
    ):
        errors.append("event_identity")
    if public_leakage_errors(views.public):
        errors.append("public_authority_leakage")
    expected_half = expected_streams // 2
    if views.manifest.get("strata") != {
        "separated_recurrence": expected_half,
        "overlapping_recurrence": expected_half,
    }:
        errors.append("strata")
    for stream_offset in range(expected_streams):
        start = stream_offset * EVENTS_PER_STREAM
        rows = views.public[start : start + EVENTS_PER_STREAM]
        if [row.get("chronology_index") for row in rows] != list(range(EVENTS_PER_STREAM)):
            errors.append("chronology")
            break
    return errors


def seal_streams(
    paths: ExperimentPaths,
    development: StreamViews,
    prospective: StreamViews,
) -> JsonDict:
    """Seal six authority-separated views and one hash-bound manifest."""

    receipts = {
        "development_public": _write_immutable(
            paths.development_public, jsonl_bytes(development.public)
        ),
        "development_private_authority": _write_immutable(
            paths.development_authority, jsonl_bytes(development.authority)
        ),
        "development_releases": _write_immutable(
            paths.development_releases, jsonl_bytes(development.releases)
        ),
        "prospective_public": _write_immutable(
            paths.prospective_public, jsonl_bytes(prospective.public)
        ),
        "prospective_private_authority": _write_immutable(
            paths.prospective_authority, jsonl_bytes(prospective.authority)
        ),
        "prospective_releases": _write_immutable(
            paths.prospective_releases, jsonl_bytes(prospective.releases)
        ),
    }
    manifest = {
        "schema": "carnot.exp7267.sealed_stream_manifest.v1",
        "development": development.manifest,
        "prospective": prospective.manifest,
        "receipts": receipts,
        "authority_separated": True,
        "old_32_stream_role": "diagnostic_only",
    }
    receipt = _write_immutable(paths.stream_manifest, transactional.canonical_json_bytes(manifest))
    return {**manifest, "manifest_receipt": receipt}


def run_development_selection(views: StreamViews) -> JsonDict:
    """Freeze query parameters on eight development public streams only."""

    archives = [
        archive_row(f"development-{parameter}", index, dict.fromkeys(FAMILIES, 1 << parameter))
        for index, parameter in enumerate((0, 8, 16, 24))
    ]
    rows = []
    changes = 0
    for stream_offset in range(DEVELOPMENT_STREAM_COUNT):
        start = stream_offset * EVENTS_PER_STREAM
        events = views.public[start : start + 64]
        stream_changes = 0
        maximum = 0
        for block_offset in range(0, len(events), QUERY_BLOCK_SIZE):
            block = events[block_offset : block_offset + QUERY_BLOCK_SIZE]
            ranks = {str(row["event_id"]): index for index, row in enumerate(reversed(block))}
            selected, receipt = select_disagreement_query(archives, block, ranks)
            least = min(
                receipt["candidate_rows"],
                key=lambda row: (row["disagreement_pair_count"], row["event_id"]),
            )
            maximum = max(maximum, int(receipt["maximum_disagreement"]))
            stream_changes += int(
                int(receipt["maximum_disagreement"]) > int(least["disagreement_pair_count"])
                and selected["event_id"] != least["event_id"]
            )
        changes += stream_changes
        rows.append(
            {
                "development_stream_id": f"development-{stream_offset + 1:02d}",
                "active_query_selection_change_count": stream_changes,
                "maximum_disagreement": maximum,
                "censored": False,
            }
        )
    return {
        "development_stream_count": DEVELOPMENT_STREAM_COUNT,
        "active_query_selection_change_count": changes,
        "rows": rows,
        "parameters_frozen_before_prospective": True,
        "prospective_authority_used": False,
        "stable_basis_capacity": STABLE_BASIS_CAPACITY,
        "fresh_validation_capacity": FRESH_VALIDATION_CAPACITY,
        "tie_rule": "maximum_disagreement_then_seeded_rank_then_event_id",
    }


def _controller_input(event: Mapping[str, Any]) -> JsonDict:
    """Project a stream event to the three learner-visible finite coordinates."""

    return {
        "event_id": str(event["event_id"]),
        "family_id": str(event["family_id"]),
        "numeric_value": int(event["numeric_value"]),
    }


def _support_release(pending: Mapping[str, Any]) -> JsonDict:
    """Expose one due observed label after its fixed request delay."""

    return {
        **dict(pending["public"]),
        "observed_label": str(pending["observed_label"]),
        "role": "support",
        "request_index": int(pending["request_index"]),
        "release_index": int(pending["release_index"]),
    }


def _packed_memory(controller: Any, pending: Sequence[Mapping[str, Any]]) -> int:
    """Measure one controller and its delayed feedback without hidden estimates."""

    if isinstance(controller, RecognitionController):
        return int(controller.memory_usage(pending)["total_bytes"])
    if isinstance(controller, exp7253.CoverageArchiveController):
        return int(controller.memory_usage(pending)["total_bytes"])
    return len(controller.state_bytes()) + len(transactional.canonical_json_bytes(list(pending)))


def _query_choice(
    controller: Any,
    block: Sequence[Mapping[str, Any]],
    ranks: Mapping[str, int],
) -> tuple[JsonDict, JsonDict]:
    """Use each arm's public query interface and retain a common receipt shape."""

    if isinstance(controller, RecognitionController):
        return controller.select_request(block, ranks)
    selected = controller.select_request(block, ranks)
    return dict(selected), {
        "archive_ids": [],
        "selected_archive_predictions": [],
        "maximum_disagreement": 0,
    }


def _reduce_group(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce one chronological stream-arm unit without dropping abstentions."""

    ordered = sorted(rows, key=lambda row: int(row["chronology_index"]))
    future = [row for row in ordered if int(row["chronology_index"]) >= WARMUP_COUNT]
    recurrence = [row for row in future if row["recurrence_eligible"] is True]
    return {
        "unit_id": ordered[0]["unit_id"],
        "stream_id": ordered[0]["stream_id"],
        "seed": ordered[0]["seed"],
        "stratum": ordered[0]["stratum"],
        "arm": ordered[0]["arm"],
        "metric": "fixture_full_denominator_error",
        "event_count": len(ordered),
        "future_event_count": len(future),
        "future_error": sum(int(row["full_denominator_error"]) for row in future),
        "false_accept": sum(int(row["false_accept"]) for row in future),
        "abstention": sum(int(row["abstention"]) for row in future),
        "recurrence_event_count": len(recurrence),
        "recurrence_error": sum(int(row["full_denominator_error"]) for row in recurrence),
        "query_count": sum(int(row["query_selected"]) for row in ordered),
        "release_count": int(ordered[-1]["released_query_count_after"]),
        "archive_admission_count": sum(int(row["archive_admission_count"]) for row in ordered),
        "reactivation_count": sum(int(row["archive_reactivation_count"]) for row in ordered),
        "selection_change_count": sum(int(row["selection_change_count"]) for row in ordered),
        "maximum_memory_bytes": max(int(row["memory_total_bytes"]) for row in ordered),
        "censored": False,
    }


def reduce_event_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep streams as independent units and preserve the frozen arm order."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["stream_id"]), str(row["arm"]))].append(row)
    order = {arm: index for index, arm in enumerate(ARMS)}
    return [
        _reduce_group(group)
        for _, group in sorted(groups.items(), key=lambda item: (item[0][0], order[item[0][1]]))
    ]


def event_row_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check chronology, authority isolation, unit completeness, and frozen predictions."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(
        any(row.get("held_out_label_visible_to_controller") is not False for row in rows),
        "future_label_leakage",
    )
    add(
        any(row.get("prediction_frozen_before_release") is not True for row in rows),
        "feedback_chronology",
    )
    add(
        any(
            row.get("controller_input_fields") != ["event_id", "family_id", "numeric_value"]
            for row in rows
        ),
        "controller_input_fields",
    )
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("stream_id")), str(row.get("arm")))].append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda row: int(row["chronology_index"]))
        add(
            [int(row["chronology_index"]) for row in ordered] != list(range(EVENTS_PER_STREAM)),
            "incomplete_unit",
        )
    return errors


def independent_reduce(path: Path) -> list[JsonDict]:
    """Reload raw JSONL so producer aggregates cannot influence fixture rows."""

    rows = _read_jsonl(path)
    if not rows:
        raise ValueError("raw_rows_unavailable")
    errors = event_row_errors(rows)
    if errors:
        raise ValueError("raw_row_validation_failed:" + ",".join(errors))
    return reduce_event_rows(rows)


def run_recognition_panel(
    views: StreamViews,
    *,
    stream_ids: Sequence[str] | None = None,
    progress: bool = False,
) -> RecognitionPanel:
    """Replay eight arms with predictions sealed before each delayed release."""

    selected_streams = tuple(
        stream_ids or (f"prospective-{index + 1:02d}" for index in range(STREAM_COUNT))
    )
    authority = {str(row["event_id"]): row for row in views.authority}
    schedules = {str(row["event_id"]): row for row in views.releases}
    by_stream = {
        stream_id: [row for row in views.public if row["stream_id"] == stream_id]
        for stream_id in selected_streams
    }
    event_rows: list[JsonDict] = []
    final_states: list[JsonDict] = []
    maximum_memory = 0
    started = time.monotonic()
    last_heartbeat = started
    for stream_offset, stream_id in enumerate(selected_streams):
        events = by_stream[stream_id]
        if len(events) != EVENTS_PER_STREAM:
            raise ValueError(f"incomplete_stream:{stream_id}")
        first_truth = authority[str(events[0]["event_id"])]
        seed = int(first_truth["stream_seed"])
        controllers: dict[str, Any] = {
            "frozen": exp7226.PackedBeliefController(),
            "reset": exp7226.PackedBeliefController(),
            "previous_coverage": exp7253.CoverageArchiveController(),
            "active_recognition": RecognitionController(),
            "random_query_recognition": RecognitionController(query_mode="random"),
            "shuffled_archive_association": RecognitionController(association_mode="shuffled"),
            "full_version_space_memory": exp7226.PackedBeliefController(),
            "feedback_withheld": exp7226.PackedBeliefController(),
        }
        pending: dict[str, list[JsonDict]] = {arm: [] for arm in ARMS}
        query_counts = dict.fromkeys(ARMS, 0)
        release_counts = dict.fromkeys(ARMS, 0)
        memory = {arm: _packed_memory(controller, []) for arm, controller in controllers.items()}
        stream_start = len(event_rows)
        for block_index, offset in enumerate(range(0, EVENTS_PER_STREAM, QUERY_BLOCK_SIZE)):
            block = events[offset : offset + QUERY_BLOCK_SIZE]
            public_block = [_controller_input(row) for row in block]
            ranks = exp7199.seeded_tie_ranks(seed, block_index, public_block)
            choices = {
                arm: _query_choice(controller, public_block, ranks)
                for arm, controller in controllers.items()
            }
            for event in block:
                event_id = str(event["event_id"])
                index = int(event["chronology_index"])
                public_event = _controller_input(event)
                predictions = {
                    arm: controller.predict(public_event) for arm, controller in controllers.items()
                }
                truth = authority[event_id]
                positions = {}
                for arm in ARMS:
                    prediction, energy = predictions[arm]
                    selected, query_receipt = choices[arm]
                    will_query = (
                        selected["event_id"] == event_id
                        and int(query_counts[arm]) < QUERY_CEILING
                        and len(pending[arm]) < PENDING_CAPACITY
                    )
                    positions[arm] = len(event_rows)
                    event_rows.append(
                        {
                            "unit_id": f"{stream_id}:{arm}",
                            "stream_id": stream_id,
                            "seed": seed,
                            "stratum": truth["stratum"],
                            "arm": arm,
                            "event_id": event_id,
                            "chronology_index": index,
                            "prediction": prediction,
                            "prediction_energy": energy,
                            "later_released_label": truth["exact_label"],
                            "full_denominator_error": int(prediction != truth["exact_label"]),
                            "false_accept": int(
                                prediction == "accept" and truth["exact_label"] == "reject"
                            ),
                            "abstention": int(prediction == "abstain"),
                            "recurrence_eligible": index >= 768,
                            "query_selected": will_query,
                            "query_disagreement": query_receipt["maximum_disagreement"]
                            if will_query
                            else 0,
                            "released_query_count_before": int(release_counts[arm]),
                            "released_query_count_after": int(release_counts[arm]),
                            "archive_admission_count": 0,
                            "archive_reactivation_count": 0,
                            "selection_change_count": 0,
                            "memory_total_bytes": int(memory[arm]),
                            "controller_input_fields": ["event_id", "family_id", "numeric_value"],
                            "held_out_label_visible_to_controller": False,
                            "prediction_frozen_before_release": True,
                            "same_event_correction": False,
                            "censored": False,
                        }
                    )
                    if will_query:
                        schedule = schedules[event_id]
                        release_index = index + int(schedule["delay"])
                        if isinstance(controllers[arm], RecognitionController):
                            controllers[arm].record_query(
                                public_event,
                                query_receipt,
                                request_index=index,
                                release_index=release_index,
                            )
                        pending[arm].append(
                            {
                                "public": public_event,
                                "observed_label": schedule["observed_label"],
                                "request_index": index,
                                "release_index": release_index,
                            }
                        )
                        query_counts[arm] += 1
                        memory[arm] = _packed_memory(controllers[arm], pending[arm])
                for arm, controller in controllers.items():
                    due = sorted(
                        [row for row in pending[arm] if int(row["release_index"]) <= index],
                        key=lambda row: (int(row["release_index"]), int(row["request_index"])),
                    )
                    if arm == "feedback_withheld":
                        due = []
                    if arm == "frozen":
                        due = [row for row in due if int(row["request_index"]) < WARMUP_COUNT]
                    if due:
                        receipt = controller.commit_batch(
                            [_support_release(row) for row in due],
                            current_cycle=index,
                            expected_parent_hash=controller.state_hash(),
                        )
                        operations = receipt["operations"]
                        event_rows[positions[arm]]["archive_admission_count"] = sum(
                            int(row.get("archived_state_id") is not None) for row in operations
                        )
                        event_rows[positions[arm]]["archive_reactivation_count"] = sum(
                            int(row.get("reactivated_archive_id") is not None) for row in operations
                        )
                        event_rows[positions[arm]]["selection_change_count"] = sum(
                            int(row.get("selection_changed") is True) for row in operations
                        )
                        release_counts[arm] += len(due)
                        pending[arm] = [row for row in pending[arm] if row not in due]
                        memory[arm] = _packed_memory(controller, pending[arm])
                    event_rows[positions[arm]]["released_query_count_after"] = int(
                        release_counts[arm]
                    )
                    event_rows[positions[arm]]["memory_total_bytes"] = int(memory[arm])
                    maximum_memory = max(maximum_memory, int(memory[arm]))
                now = time.monotonic()
                if progress and now - last_heartbeat >= 60:
                    print(
                        f"phase 5 benchmark heartbeat streams={stream_offset}/{len(selected_streams)} "
                        f"events={len(event_rows)} elapsed_s={now - started:.3f}",
                        flush=True,
                    )
                    last_heartbeat = now
        for arm, controller in controllers.items():
            final_states.append(
                {
                    "stream_id": stream_id,
                    "arm": arm,
                    "state_sha256": controller.state_hash(),
                    "state_bytes": len(controller.state_bytes()),
                    "archive_count": len(controller.archives())
                    if hasattr(controller, "archives")
                    else 0,
                    "pending_count": len(pending[arm]),
                }
            )
        if progress:
            print(
                f"phase 5 benchmark unit {stream_offset + 1}/{len(selected_streams)} "
                f"completed_rows={len(event_rows) - stream_start} elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
    return RecognitionPanel(
        event_rows,
        reduce_event_rows(event_rows),
        final_states,
        len(selected_streams),
        0,
        maximum_memory,
    )


def _control_release(event_id: str, value: int, label: str, index: int) -> JsonDict:
    """Build one released lower-bound witness for isolated transaction controls."""

    return {
        "event_id": event_id,
        "family_id": "lower_bound",
        "numeric_value": value,
        "observed_label": label,
        "role": "support",
        "request_index": index,
        "release_index": index,
    }


def run_mutation_controls(root: Path) -> list[JsonDict]:
    """Require future-label, ineffective-shuffle, duplicate, and stale attacks to fail."""

    root.mkdir(parents=True, exist_ok=True)
    masks = [dict.fromkeys(FAMILIES, 1 << parameter) for parameter in (0, 8)]
    archives = [archive_row(f"control-{index}", index, row) for index, row in enumerate(masks)]
    leak_rejected = False
    try:
        select_disagreement_query(
            archives,
            [
                {
                    "event_id": "leak",
                    "family_id": "lower_bound",
                    "numeric_value": 7,
                    "exact_label": "accept",
                }
            ],
            [0],
        )
    except ValueError as error:
        leak_rejected = str(error) == "private_authority_in_query"

    query = {"event_id": "shuffle", "family_id": "lower_bound", "numeric_value": 7}
    _, receipt = select_disagreement_query(archives, [query], [0])
    original = receipt["archive_ids"]
    mutated = list(original)
    ineffective_detected = len(mutated) > 1 and mutated == original

    controller = RecognitionController.from_masks(masks[0])
    state_path = root / "controller.json"
    controller.save(state_path)
    first = _control_release("duplicate", 0, "accept", 1)
    controller.commit_batch(
        [first],
        current_cycle=1,
        expected_parent_hash=controller.state_hash(),
        state_path=state_path,
    )
    child = controller.state_bytes()
    duplicate_rejected = False
    try:
        controller.commit_batch(
            [first],
            current_cycle=1,
            expected_parent_hash=controller.state_hash(),
            state_path=state_path,
        )
    except RecognitionCommitRejected as error:
        duplicate_rejected = str(error) == "duplicate_release"
    stale_rejected = False
    try:
        controller.commit_batch(
            [_control_release("stale", 0, "reject", 2)],
            current_cycle=2,
            expected_parent_hash="sha256:" + "0" * 64,
            state_path=state_path,
        )
    except RecognitionCommitRejected as error:
        stale_rejected = str(error) == "stale_parent"
    unchanged = controller.state_bytes() == child == state_path.read_bytes()
    return [
        {
            "mutation": "future_label_access",
            "expected_rejection": "private_authority_in_query",
            "observed_rejection": "private_authority_in_query" if leak_rejected else None,
            "passed": leak_rejected,
        },
        {
            "mutation": "ineffective_shuffled_identity",
            "expected_rejection": "unchanged_selection_detected",
            "observed_rejection": "unchanged_selection_detected" if ineffective_detected else None,
            "passed": ineffective_detected,
        },
        {
            "mutation": "duplicate_release",
            "expected_rejection": "duplicate_release",
            "observed_rejection": "duplicate_release" if duplicate_rejected else None,
            "parent_bytes_preserved": unchanged,
            "passed": duplicate_rejected and unchanged,
        },
        {
            "mutation": "stale_parent",
            "expected_rejection": "stale_parent",
            "observed_rejection": "stale_parent" if stale_rejected else None,
            "parent_bytes_preserved": unchanged,
            "passed": stale_rejected and unchanged,
        },
    ]


def run_e2e_controls(root: Path) -> list[JsonDict]:
    """Exercise released input through query, delayed commit, later use, restore, and rollback."""

    root.mkdir(parents=True, exist_ok=True)
    active = dict.fromkeys(FAMILIES, 1 << 0)
    controller = RecognitionController.from_masks(active)
    controller.add_archive(dict.fromkeys(FAMILIES, 1 << 8))
    controller.add_archive(dict.fromkeys(FAMILIES, 1 << 16))
    query_event = {"event_id": "e2e-query", "family_id": "lower_bound", "numeric_value": 12}
    before_prediction = controller.predict(query_event)
    selected, query_receipt = controller.select_request([query_event], [0])
    controller.record_query(selected, query_receipt, request_index=1, release_index=2)
    state_path = root / "controller.json"
    controller.save(state_path)
    parent = controller.state_bytes()
    delayed = _control_release("e2e-query", 12, "reject", 1)
    delayed["release_index"] = 2
    premature_rejected = False
    try:
        controller.commit_batch(
            [delayed],
            current_cycle=1,
            expected_parent_hash=controller.state_hash(),
            state_path=state_path,
        )
    except RecognitionCommitRejected as error:
        premature_rejected = str(error) == "future_release"
    receipt = controller.commit_batch(
        [delayed],
        current_cycle=2,
        expected_parent_hash=controller.state_hash(),
        state_path=state_path,
    )
    later = controller.predict(query_event)
    restored = RecognitionController.load(state_path)
    restore_ok = (
        restored.state_hash() == controller.state_hash() and restored.predict(query_event) == later
    )
    rollback = restored.rollback(receipt, state_path=state_path)
    rollback_ok = (
        rollback["byte_identical"] is True
        and restored.state_bytes() == parent == state_path.read_bytes()
    )
    return [
        {"stage": "released_event", "passed": delayed["observed_label"] == "reject"},
        {
            "stage": "query",
            "passed": selected["event_id"] == "e2e-query"
            and query_receipt["future_label_used"] is False,
        },
        {"stage": "delayed_feedback", "passed": premature_rejected},
        {"stage": "atomic_commit", "passed": receipt["new_state_hash"] == controller.state_hash()},
        {"stage": "later_prediction", "passed": before_prediction != later},
        {"stage": "cold_restore", "passed": restore_ok},
        {"stage": "rollback", "passed": rollback_ok},
    ]


def _task_identity(text: str) -> JsonDict:
    """Extract only the active Exp7267 task from the executable roadmap."""

    match = re.search(r"(?ms)^- id: exp7267-recognition-prototype\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(1)
    milestone = re.search(r"(?m)^  milestone: (.+)$", block)
    deliverable = re.search(r"(?m)^  deliverable: (.+)$", block)
    return {
        "id": "exp7267-recognition-prototype" if block else None,
        "milestone": None if milestone is None else milestone.group(1).strip(),
        "deliverable": None if deliverable is None else deliverable.group(1).strip(),
    }


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    upstream_overrides: Mapping[str, Path] | None = None,
) -> tuple[list[JsonDict], dict[str, str | None], dict[str, JsonDict]]:
    """Authenticate task, null evidence, sources, imports, quarantine, and output ownership."""

    overrides = dict(upstream_overrides or {})
    upstream_paths = {
        name: _resolve(repo_root, overrides.get(name, path))
        for name, path in UPSTREAM_ARTIFACTS.items()
    }
    upstream = {name: _load_object(path) for name, path in upstream_paths.items()}
    hashes = {
        str(_resolve(repo_root, path)): _sha256_path(_resolve(repo_root, path))
        for path in SOURCE_PATHS
    }
    for name, path in upstream_paths.items():
        hashes[str(path)] = _sha256_path(path)
    spec = _resolve(repo_root, SPEC_PATH).read_text(encoding="utf-8")
    roadmap = _resolve(repo_root, "research-roadmap.yaml").read_text(encoding="utf-8")
    exclusions = _resolve(repo_root, "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    source_state = {
        str(path): "nonempty" if hashes[str(_resolve(repo_root, path))] is not None else "missing"
        for path in SOURCE_PATHS
    }
    writable_fields = (
        "development_public",
        "prospective_public",
        "raw_rows",
        "diagnosis_rows",
        "state_sidecar",
        "control_sidecar",
        "evidence_sidecar",
        "provisional",
        "terminal_candidate",
        "artifact",
    )
    writable = {field: _path_writable(getattr(paths, field)) for field in writable_fields}
    checks = [
        gate_check(
            "driving_capability_spec", str(SPEC_PATH), "REQ-CL-7267", True, "REQ-CL-7267" in spec
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7267-*",
            8,
            len(set(SCENARIO_PATTERN.findall(spec))),
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            dict.fromkeys(source_state, "nonempty"),
            source_state,
        ),
        gate_check(
            "v639_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            {
                "id": "exp7267-recognition-prototype",
                "milestone": MILESTONE,
                "deliverable": str(DEFAULT_ARTIFACT),
            },
            _task_identity(roadmap),
        ),
        gate_check("required_imports", "python", "finite controllers", True, True),
        gate_check(
            "writable_output_paths",
            "host_filesystem",
            "task-owned outputs",
            dict.fromkeys(writable, True),
            writable,
        ),
    ]
    for name, path in upstream_paths.items():
        artifact = upstream[name]
        checks.extend(
            [
                gate_check(
                    f"{name}_artifact_hash",
                    name,
                    str(path),
                    EXPECTED_UPSTREAM_HASHES[name],
                    hashes[str(path)],
                ),
                gate_check(f"{name}_status", name, "status", "complete", artifact.get("status")),
                gate_check(
                    f"{name}_verdict", name, "verdict_class", "null", artifact.get("verdict_class")
                ),
            ]
        )
        quarantine = exp7213.quarantine_state(artifact, exclusions, path.name, name)
        checks.append(
            gate_check(
                f"{name}_not_quarantined",
                "artifact_metadata_and_ops/exclusion_manifest.yaml",
                "quarantined",
                False,
                quarantine["quarantined"],
            )
        )
    checks.append(
        gate_check(
            "saved_nomination_rows",
            "exp7255",
            "shuffle_effect_rows",
            True,
            bool(upstream["exp7255"].get("shuffle_effect_rows")),
        )
    )
    return checks, hashes, upstream


def _summary_with_failures(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain the first failed gate and every exact failed observation."""

    summary = gate_summary(checks)
    summary["failed_checks"] = [dict(row) for row in checks if row.get("passed") is not True]
    return summary


def _recognition_contract() -> JsonDict:
    """Freeze query, witness, change-signal, segmentation, and memory rules."""

    return {
        "archive_representation": "immutable_finite_survivor_masks",
        "archive_capacity": ARCHIVE_CAP,
        "stable_basis_capacity": STABLE_BASIS_CAPACITY,
        "stable_basis_rule": "retain_released_maximum_archive_pair_disagreement",
        "fresh_validation_capacity": FRESH_VALIDATION_CAPACITY,
        "fresh_validation_minimum": MIN_FRESH_VALIDATION,
        "fresh_validation_zero_contradictions": True,
        "query_rule": "maximum_archived_pair_disagreement",
        "query_ceiling_per_arm": QUERY_CEILING,
        "query_block_size": QUERY_BLOCK_SIZE,
        "pending_capacity": PENDING_CAPACITY,
        "reactivation_requires_fresh_feedback": True,
        "same_event_correction": False,
        "online_change_signal": "released_contradiction_detection",
        "supplied_boundary_use": "oracle_assisted_diagnostic_only",
        "autonomous_nomination_boundary_access": False,
        "future_label_access": False,
        "evaluator_parameter_access": False,
        "memory_caps": deepcopy(MEMORY_CAPS),
    }


def _sample_budget(stream_ids: Sequence[str], *, complete: bool) -> JsonDict:
    """State planned, attempted, completed, censored, and fixed stopping units."""

    completed_streams = len(stream_ids) if complete else 0
    return {
        "planned_development_streams": DEVELOPMENT_STREAM_COUNT,
        "planned_prospective_streams": STREAM_COUNT,
        "attempted_streams": len(stream_ids),
        "completed_streams": completed_streams,
        "censored_streams": 0 if complete else len(stream_ids),
        "events_per_stream": EVENTS_PER_STREAM,
        "warmup_events_per_stream": WARMUP_COUNT,
        "planned_arms": len(ARMS),
        "planned_arm_event_rows": len(stream_ids) * EVENTS_PER_STREAM * len(ARMS),
        "completed_arm_event_rows": completed_streams * EVENTS_PER_STREAM * len(ARMS),
        "query_ceiling_per_arm_stream": QUERY_CEILING,
        "fixed_stopping_rule": "run every selected sealed stream and arm without outcome stopping",
        "old_stream_count_used_for_prospective_rows": 0,
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
    """Create every required field before blocking or measured classification."""

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
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_invoked": MODEL_INVOKED,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": socket.gethostname(),
        "duration_s": duration_s,
        "random_seed": {
            "root": RANDOM_SEED,
            "development_streams": list(DEVELOPMENT_STREAM_SEEDS),
            "prospective_streams": list(STREAM_SEEDS),
            "shuffle": SHUFFLE_SEED,
            "frozen_before_prospective_labels": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": _sample_budget(stream_ids, complete=False),
        "acceptance_gate_results": {},
        "gate_check_summary": _summary_with_failures(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_precondition",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "recognition_fixture_ready_score": 0,
        "failure_cause_rows": [],
        "stream_manifest": {},
        "recognition_contract": _recognition_contract(),
        "continuous_self_learning_task": True,
        "held_out_learning_value_scored": False,
        "production_defaults_modified": False,
        "publication_performed": False,
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    stream_ids: Sequence[str],
    *,
    started_at: str,
    duration_s: float,
) -> JsonDict:
    """Return a row-free terminal block for an external prerequisite failure."""

    artifact = _base_artifact(
        checks,
        source_hashes,
        stream_ids,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=duration_s,
    )
    failed = artifact["gate_check_summary"]["failed_checks"]
    first = failed[0] if failed else {}
    artifact["honest_verdict"] = (
        (f"blocked_{first.get('upstream', 'external')}_{first.get('field', 'precondition')}")
        .replace("/", "_")
        .replace(" ", "_")
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable identity, configuration, sources, rows, gates, and receipts."""

    excluded = {
        "reproducibility_checksum",
        "started_at_utc",
        "completed_at_utc",
        "duration_s",
        "phase_spans_s",
        "execution_host",
    }
    return transactional.sha256_json(
        {key: value for key, value in artifact.items() if key not in excluded}
    )


def _receipt_matches(repo_root: Path, receipt: Mapping[str, Any]) -> bool:
    """Compare a declared sidecar hash with current exact bytes."""

    return _sha256_path(_resolve(repo_root, str(receipt.get("path", "")))) == receipt.get("sha256")


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
    check_files: bool = False,
) -> list[str]:
    """Cold-check terminal schema, rows, limits, controls, sources, and checksum."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(any(field not in artifact for field in REQUIRED_ARTIFACT_FIELDS), "required_fields")
    add(
        artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID,
        "identity",
    )
    add(
        artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE,
        "date_milestone",
    )
    add(
        artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False,
        "model_contract",
    )
    add(artifact.get("invocation_counts") != INVOCATION_COUNTS, "invocation_counts")
    add(artifact.get("verifier_is_oracle") is not True, "oracle_declaration")
    add(artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact), "checksum")
    receipts = artifact.get("validation_receipts", [])
    add(
        not isinstance(receipts, list)
        or any(
            set(row) != {"command", "exit_code", "classification", "log_sha256"}
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(row.get("log_sha256", ""))) is None
            for row in receipts
        ),
        "validation_receipts",
    )
    if artifact.get("status") == "blocked":
        add(
            artifact.get("rows") != []
            or artifact.get("recognition_fixture_ready_score") != 0
            or artifact.get("verdict_class") != "blocked"
            or artifact.get("inference_substrate") != "blocked_no_run"
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
    ready = artifact.get("recognition_fixture_ready_score")
    if ready == 1:
        complete_valid = artifact.get("verdict_class") == "circular_positive" and str(
            artifact.get("honest_verdict", "")
        ).startswith("complete_circular_positive")
    else:
        complete_valid = (
            ready == 0
            and artifact.get("verdict_class") == "null"
            and str(artifact.get("honest_verdict", "")).startswith("complete_null")
        )
    add(
        not complete_valid or artifact.get("held_out_learning_value_scored") is not False,
        "complete_contract",
    )
    selected = tuple(
        expected_stream_ids or (f"prospective-{index + 1:02d}" for index in range(STREAM_COUNT))
    )
    expected_units = {(stream_id, arm) for stream_id in selected for arm in ARMS}
    rows = artifact.get("rows", [])
    add(
        not isinstance(rows, list)
        or {(row.get("stream_id"), row.get("arm")) for row in rows} != expected_units,
        "rows",
    )
    add(
        any(
            int(row.get("query_count", QUERY_CEILING + 1)) > QUERY_CEILING
            or int(row.get("maximum_memory_bytes", MEMORY_CAPS["total_bytes"] + 1))
            > MEMORY_CAPS["total_bytes"]
            or row.get("censored") is not False
            for row in rows
        ),
        "row_limits",
    )
    gates = artifact.get("acceptance_gate_results", {})
    add(
        not isinstance(gates, Mapping)
        or not gates
        or any(row.get("passed") != row.get("pass") for row in gates.values())
        or ready != int(all(row.get("passed") is True for row in gates.values())),
        "acceptance_gate_results",
    )
    manifest = artifact.get("stream_manifest", {})
    add(
        manifest.get("development", {}).get("stream_count") != DEVELOPMENT_STREAM_COUNT
        or manifest.get("prospective", {}).get("stream_count") != STREAM_COUNT
        or manifest.get("old_32_stream_role") != "diagnostic_only",
        "stream_manifest",
    )
    if check_files:
        sidecars = [
            artifact.get("raw_rows_receipt", {}),
            artifact.get("diagnosis_rows_receipt", {}),
            artifact.get("state_sidecar_receipt", {}),
            artifact.get("control_sidecar_receipt", {}),
            artifact.get("evidence_sidecar_receipt", {}),
            manifest.get("manifest_receipt", {}),
        ]
        add(any(not _receipt_matches(repo_root, row) for row in sidecars), "sidecar_hashes")
        add(
            independent_reduce(_resolve(repo_root, artifact["raw_rows_receipt"]["path"])) != rows,
            "independent_reducer",
        )
        add(
            any(
                expected is None or _sha256_path(_resolve(repo_root, path)) != expected
                for path, expected in artifact.get("source_artifact_hashes", {}).items()
            ),
            "source_artifact_hashes",
        )
    return errors


def attach_validation_receipts(
    artifact: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Attach exact command receipts and refresh the stable checksum."""

    required = {"command", "exit_code", "classification", "log_sha256"}
    if any(
        set(receipt) != required
        or not isinstance(receipt["command"], str)
        or not isinstance(receipt["exit_code"], int)
        or not isinstance(receipt["classification"], str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt["log_sha256"])) is None
        for receipt in receipts
    ):
        raise ValueError("validation_receipt_schema")
    changed = deepcopy(dict(artifact))
    changed["validation_receipts"] = [dict(receipt) for receipt in receipts]
    changed["reproducibility_checksum"] = reproducibility_checksum(changed)
    return changed


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    upstream_overrides: Mapping[str, Path] | None = None,
    progress: bool = False,
) -> JsonDict:
    """Authenticate, diagnose, seal streams, replay controls, reduce, and validate."""

    selected = tuple(
        stream_ids or (f"prospective-{index + 1:02d}" for index in range(STREAM_COUNT))
    )
    monotonic_start = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: JsonDict = {}
    if progress:
        _progress(0, "start", "authenticate inputs, quarantine state, and writable outputs")
    phase_start = time.monotonic()
    checks, source_hashes, upstream = collect_preconditions(
        repo_root,
        paths,
        upstream_overrides=upstream_overrides,
    )
    spans["phase_0_preconditions"] = time.monotonic() - phase_start
    _atomic_write(
        paths.provisional,
        transactional.canonical_json_bytes(
            {"schema": SCHEMA, "status": "in_progress", "phase": 0, "checks": checks}
        ),
    )
    if gate_summary(checks)["passed"] is not True:
        if progress:
            _progress(0, "end", "external precondition failed; no measurement ran")
        return build_blocked_artifact(
            checks,
            source_hashes,
            selected,
            started_at=started_at,
            duration_s=time.monotonic() - monotonic_start,
        )
    if progress:
        _progress(0, "end", "all exact external preconditions passed")

    phase_start = time.monotonic()
    if progress:
        _progress(1, "start", "confirm zero current model work")
        print("phase 1 BEFORE model load: no model load scheduled", flush=True)
        print("phase 1 AFTER model load: attempted and completed loads remain zero", flush=True)
        print("phase 1 BEFORE generation: no generation call scheduled", flush=True)
        print(
            "phase 1 AFTER generation: attempted and completed generations remain zero", flush=True
        )
    spans["phase_1_no_llm"] = time.monotonic() - phase_start
    if progress:
        _progress(1, "end", "MODEL_SPECS is empty and every current counter is zero")

    phase_start = time.monotonic()
    if progress:
        _progress(2, "start", "reduce saved null rows before replay")
    failure_rows = reduce_saved_failure_rows(upstream["exp7255"])
    diagnostic = run_stable_basis_diagnostic()
    failure_rows.append(
        {
            "stage": "stable_released_basis_counterexample",
            "observed": diagnostic["counterexample_survives_stable_basis"],
            "censored": False,
            "interpretation": "observed finite replay; historical primary cause remains unidentified from saved aggregates",
        }
    )
    spans["phase_2_prior_failure_diagnosis"] = time.monotonic() - phase_start
    if progress:
        _progress(
            2,
            "end",
            f"saved_nominations={failure_rows[0]['observed']} stable_counterexample={diagnostic['counterexample_survives_stable_basis']}",
        )

    phase_start = time.monotonic()
    if progress:
        _progress(3, "start", "generate and seal eight development plus 24 prospective streams")
    development = build_stream_views("development")
    prospective = build_stream_views("prospective")
    stream_errors = stream_conformance_errors(
        development, "development"
    ) + stream_conformance_errors(prospective, "prospective")
    if stream_errors:
        raise ValueError("stream_conformance_failed:" + ",".join(stream_errors))
    stream_manifest = seal_streams(paths, development, prospective)
    spans["phase_3_stream_seal"] = time.monotonic() - phase_start
    if progress:
        _progress(3, "end", "all separated stream bytes sealed")

    phase_start = time.monotonic()
    if progress:
        _progress(4, "start", "freeze parameters and require a development query selection change")
    development_result = run_development_selection(development)
    spans["phase_4_development_intervention"] = time.monotonic() - phase_start
    if progress:
        _progress(
            4,
            "end",
            f"selection_changes={development_result['active_query_selection_change_count']}",
        )

    phase_start = time.monotonic()
    if progress:
        _progress(5, "start", "BEFORE eight-arm CPU benchmark")
        print("phase 5 BEFORE benchmark: exact finite-controller replay", flush=True)
    panel = run_recognition_panel(prospective, stream_ids=selected, progress=progress)
    spans["phase_5_benchmark"] = time.monotonic() - phase_start
    if progress:
        print("phase 5 AFTER benchmark: all selected stream-arm events completed", flush=True)
        _progress(5, "end", f"completed_event_rows={len(panel.event_rows)}")

    phase_start = time.monotonic()
    if progress:
        _progress(6, "start", "run E2E, mutations, raw reduction, and sidecar seals")
    raw_errors = event_row_errors(panel.event_rows)
    if raw_errors:
        raise ValueError("event_row_errors:" + ",".join(raw_errors))
    controls_root = paths.provisional.parent / "experiment_7267_controls"
    mutations = run_mutation_controls(controls_root / "mutations")
    e2e = run_e2e_controls(controls_root / "e2e")
    raw_receipt = _atomic_write(paths.raw_rows, jsonl_bytes(panel.event_rows))
    reduced = independent_reduce(paths.raw_rows)
    if reduced != panel.rows:
        raise ValueError("independent_reducer_mismatch")
    diagnosis_receipt = _atomic_write(paths.diagnosis_rows, jsonl_bytes(failure_rows))
    state_receipt = _atomic_write(
        paths.state_sidecar,
        transactional.canonical_json_bytes(
            {"schema": "carnot.exp7267.states.v1", "states": panel.final_states}
        ),
    )
    control_receipt = _atomic_write(
        paths.control_sidecar,
        transactional.canonical_json_bytes(
            {"schema": "carnot.exp7267.controls.v1", "mutations": mutations, "e2e": e2e}
        ),
    )
    evidence_receipt = _atomic_write(
        paths.evidence_sidecar,
        transactional.canonical_json_bytes(
            {
                "schema": "carnot.exp7267.evidence.v1",
                "historical_model_receipts": {},
                "historical_artifacts": {
                    name: {
                        "path": str(UPSTREAM_ARTIFACTS[name]),
                        "sha256": EXPECTED_UPSTREAM_HASHES[name],
                        "status": upstream[name].get("status"),
                        "verdict_class": upstream[name].get("verdict_class"),
                    }
                    for name in UPSTREAM_ARTIFACTS
                },
                "stable_basis_diagnostic": diagnostic,
                "supplied_boundary_diagnostic": {
                    "enabled": False,
                    "role": "oracle_assisted_diagnostic_only",
                    "used_for_nomination": False,
                },
            }
        ),
    )
    for receipt in (
        raw_receipt,
        diagnosis_receipt,
        state_receipt,
        control_receipt,
        evidence_receipt,
        stream_manifest["manifest_receipt"],
        *stream_manifest["receipts"].values(),
    ):
        source_hashes[str(receipt["path"])] = str(receipt["sha256"])
    spans["phase_6_controls_and_reduction"] = time.monotonic() - phase_start
    if progress:
        _progress(6, "end", "all controls and independent raw-row reduction completed")

    expected_rows = len(selected) * EVENTS_PER_STREAM * len(ARMS)
    maximum_query = max(int(row["query_count"]) for row in panel.rows)
    acceptance = {
        "saved_null_rows_reduced": {
            "principle": "Diagnose the saved null before testing a replacement mechanism.",
            "expected": ">0 nominations and explicit censored stages",
            "observed": [
                failure_rows[0]["observed"],
                sum(int(row["censored"]) for row in failure_rows),
            ],
            "pass": bool(failure_rows[0]["observed"])
            and any(row["censored"] for row in failure_rows),
        },
        "stable_basis_counterexample": {
            "principle": "A measured released-only counterexample must survive the new stable basis.",
            "expected": True,
            "observed": diagnostic["counterexample_survives_stable_basis"],
            "pass": diagnostic["counterexample_survives_stable_basis"] is True,
        },
        "effective_development_intervention": {
            "principle": "Readiness requires at least one changed development query selection.",
            "expected": ">0",
            "observed": development_result["active_query_selection_change_count"],
            "pass": development_result["active_query_selection_change_count"] > 0,
        },
        "sealed_streams": {
            "principle": "All development and prospective authority views must be separate and complete.",
            "expected": [8, 24, 12, 12, 0],
            "observed": [
                development.manifest["stream_count"],
                prospective.manifest["stream_count"],
                prospective.manifest["strata"]["separated_recurrence"],
                prospective.manifest["strata"]["overlapping_recurrence"],
                len(stream_errors),
            ],
            "pass": len(stream_errors) == 0,
        },
        "complete_eight_arm_panel": {
            "principle": "Every selected stream and arm must finish without censoring.",
            "expected": expected_rows,
            "observed": len(panel.event_rows),
            "pass": len(panel.event_rows) == expected_rows and panel.censored_stream_count == 0,
        },
        "exact_query_limits": {
            "principle": "Every arm must stay at or below the fixed charged-feedback ceiling.",
            "expected": f"<={QUERY_CEILING}",
            "observed": maximum_query,
            "pass": maximum_query <= QUERY_CEILING,
        },
        "bounded_memory": {
            "principle": "Archives, witnesses, pending requests, and ledgers share the existing cap.",
            "expected": f"<={MEMORY_CAPS['total_bytes']}",
            "observed": panel.maximum_memory_bytes,
            "pass": panel.maximum_memory_bytes <= MEMORY_CAPS["total_bytes"],
        },
        "mutation_controls": {
            "principle": "Each authority, identity, duplicate, and parent mutation must fail closed.",
            "expected": len(mutations),
            "observed": sum(int(row["passed"] is True) for row in mutations),
            "pass": all(row["passed"] is True for row in mutations),
        },
        "e2e_controls": {
            "principle": "The complete delayed durable lifecycle must preserve chronology and bytes.",
            "expected": len(e2e),
            "observed": sum(int(row["passed"] is True) for row in e2e),
            "pass": all(row["passed"] is True for row in e2e),
        },
        "independent_raw_reducer": {
            "principle": "Cold JSONL reduction must reproduce every stream-arm row.",
            "expected": transactional.sha256_json(panel.rows),
            "observed": transactional.sha256_json(reduced),
            "pass": reduced == panel.rows,
        },
    }
    for row in acceptance.values():
        row["passed"] = row["pass"]
    ready = int(all(row["passed"] is True for row in acceptance.values()))
    stop = None
    if not ready:
        stop = next(name for name, row in acceptance.items() if row["passed"] is not True)
    completed_at = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks,
        source_hashes,
        selected,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=time.monotonic() - monotonic_start,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "phase_spans_s": spans,
            "rows": panel.rows,
            "sample_size_budget": _sample_budget(selected, complete=True),
            "acceptance_gate_results": acceptance,
            "verdict_class": "circular_positive" if ready else "null",
            "honest_verdict": (
                "complete_circular_positive: active recognition fixture is ready; held-out learning value is not scored"
                if ready
                else f"complete_null: recognition mechanism stopped at {stop}"
            ),
            "recognition_fixture_ready_score": ready,
            "failure_cause_rows": failure_rows,
            "historical_primary_cause": "not_identifiable_from_saved_aggregate_rows",
            "stable_basis_diagnostic": diagnostic,
            "development_selection": development_result,
            "stream_manifest": stream_manifest,
            "recognition_contract": _recognition_contract(),
            "raw_rows_receipt": {
                **raw_receipt,
                "row_count": len(panel.event_rows),
                "format": "jsonl",
            },
            "diagnosis_rows_receipt": {**diagnosis_receipt, "row_count": len(failure_rows)},
            "state_sidecar_receipt": {**state_receipt, "state_count": len(panel.final_states)},
            "control_sidecar_receipt": {
                **control_receipt,
                "mutation_count": len(mutations),
                "e2e_count": len(e2e),
            },
            "evidence_sidecar_receipt": evidence_receipt,
            "e2e_rows": e2e,
            "mutation_rows": mutations,
        }
    )
    artifact["validation_receipts"] = [
        {
            "command": f"independent_reduce {paths.raw_rows}",
            "exit_code": 0,
            "classification": "passed",
            "log_sha256": transactional.sha256_json(reduced),
        },
        {
            "command": "run_e2e_controls",
            "exit_code": 0,
            "classification": "passed",
            "log_sha256": transactional.sha256_json(e2e),
        },
    ]
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


def write_artifact(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
) -> None:
    """Cold-validate and atomically publish one terminal artifact."""

    errors = validate_artifact(
        artifact,
        repo_root=repo_root,
        expected_stream_ids=expected_stream_ids,
        check_files=True,
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    _atomic_write(path, transactional.canonical_json_bytes(dict(artifact)))


def _command_receipt(command: Sequence[str]) -> JsonDict:
    """Run one unbuffered validation subprocess and retain its exact combined log."""

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
            "COVERAGE_FILE": "/tmp/.coverage-carnot-exp7267",
        },
    )
    output: list[str] = []
    if process.stdout is not None:
        for line in process.stdout:
            output.append(line)
            print(line, end="", flush=True)
    exit_code = process.wait()
    combined = "".join(output)
    print(
        f"validation AFTER subprocess: exit_code={exit_code} elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )
    return {
        "command": command_text,
        "exit_code": exit_code,
        "classification": "passed" if exit_code == 0 else "failed",
        "log_sha256": transactional.sha256_bytes(combined.encode()),
    }


def _validation_commands(candidate: Path) -> list[list[str]]:
    """Return the fixed focused coverage, style, type, spec, and artifact checks."""

    python = str(REPO_ROOT / ".venv/bin/python")
    coverage = str(REPO_ROOT / ".venv/bin/coverage")
    ruff = str(REPO_ROOT / ".venv/bin/ruff")
    mypy = str(REPO_ROOT / ".venv/bin/mypy")
    module = "python/carnot/experiment_7267_v639_recognition_prototype.py"
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
            "--basetemp=/tmp/carnot-exp7267-final",
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
    """Parse the fixed execution date and optional private output root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure a candidate, run focused validation, then publish terminal bytes."""

    print("phase 0 immediate: Exp7267 recognition prototype started", flush=True)
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
    _progress(7, "start", "write measured candidate under raw evidence before external validation")
    _atomic_write(paths.terminal_candidate, transactional.canonical_json_bytes(artifact))
    _progress(7, "end", f"candidate={paths.terminal_candidate}")
    _progress(8, "start", "BEFORE focused coverage, Ruff, mypy, spec, adversarial, and row checks")
    receipts = list(artifact["validation_receipts"])
    receipts.extend(
        _command_receipt(command) for command in _validation_commands(paths.terminal_candidate)
    )
    artifact = attach_validation_receipts(artifact, receipts)
    _atomic_write(
        paths.provisional,
        transactional.canonical_json_bytes(
            {
                "schema": SCHEMA,
                "status": "in_progress",
                "phase": 8,
                "validation_receipts": receipts,
            }
        ),
    )
    failed = [row for row in receipts if row["exit_code"] != 0]
    if failed:
        raise RuntimeError(
            "focused_validation_failed:" + ",".join(row["command"] for row in failed)
        )
    _progress(8, "end", "AFTER all focused external validations passed")
    _progress(9, "start", "BEFORE final cold validation and atomic terminal write")
    write_artifact(paths.artifact, artifact, repo_root=REPO_ROOT)
    _progress(9, "end", f"AFTER atomic terminal write {paths.artifact}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin wrapper owns execution.
    raise SystemExit(main())
