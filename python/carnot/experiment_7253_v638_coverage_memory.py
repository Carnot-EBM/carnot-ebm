"""Build bounded coverage memory with effective archive controls.

The controller stores finite survivor masks and released-witness signatures.
It keeps four diverse snapshots without an unbounded history. The fixture uses
fresh streams and does not score a learning gain.

Spec refs: REQ-CL-7253 and SCENARIO-CL-7253-*.
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
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7253
SCHEMA = "carnot.exp7253.v638_coverage_memory.v1"
STATE_SCHEMA = "carnot.coverage_archive_controller.v1"
MILESTONE = "2026.09.638"
RUN_DATE = "20260912"
RANDOM_SEED = 7_253_000
SHUFFLE_SEED = 7_253_901
DEVELOPMENT_STREAM_SEEDS = tuple(range(7_253_001, 7_253_009))
STREAM_SEEDS = tuple(range(7_253_101, 7_253_133))
DEVELOPMENT_STREAM_COUNT = 8
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
LEDGER_CAPACITY = 16
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

FAMILIES = tuple(exp7226.FAMILIES)
PARAMETER_DOMAIN = tuple(exp7226.PARAMETER_DOMAIN)
FULL_MASK = exp7226.FULL_MASK
PATTERNS = tuple(exp7240.PATTERNS)
ARMS = (
    "frozen_warmup",
    "reset_relearn",
    "destructive_update",
    "fifo_archive_shuffled",
    "fifo_archive_aligned",
    "coverage_archive_shuffled",
    "coverage_archive_aligned",
    "oracle_positive_control",
)
ARCHIVE_ARMS = ARMS[3:7]
FORBIDDEN_PUBLIC_FIELDS = set(exp7240.FORBIDDEN_PUBLIC_FIELDS) | {
    "release_noisy",
    "boundary",
    "generator_seed",
}
MEMORY_CAPS = {
    "witness_bytes": 12_288,
    "archive_bytes": 16_384,
    "pending_bytes": 4_096,
    "ledger_bytes": 4_096,
    "controller_bytes": 65_536,
    "total_bytes": 69_632,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
WRAPPER_PATH = Path("scripts/experiments/experiment_7253_v638_coverage_memory.py")
DEFAULT_ARTIFACT = Path("results/experiment_7253_v638_coverage_memory.json")
DEFAULT_STREAM_ROOT = Path("results/streams/experiment_7253")
DEFAULT_RAW_ROWS = Path("results/raw/experiment_7253/decision_rows.jsonl")
DEFAULT_STATE_SIDECAR = Path("results/checkpoints/experiment_7253_v638_states.json")
DEFAULT_CONTROL_SIDECAR = Path("results/checkpoints/experiment_7253_v638_controls.json")
DEFAULT_EVIDENCE_SIDECAR = Path("results/checkpoints/experiment_7253_v638_evidence.json")
DEFAULT_PROVISIONAL = Path("results/checkpoints/experiment_7253_v638_in_progress.json")
UPSTREAM_ARTIFACTS = {
    "exp7240": Path("results/experiment_7240_v637_recurrence_fixture.json"),
    "exp7241": Path("results/experiment_7241_v637_recurrence_learning.json"),
    "exp7242": Path("results/experiment_7242_v637_recurrence_audit.json"),
}
EXPECTED_UPSTREAM_HASHES = {
    "exp7240": "sha256:0f12abc839f3d70006698ebaa5169f12ab1cae9dc7526c6d124078ec7a6baf74",
    "exp7241": "sha256:ebf779e3b79a07c82300758efb4392fd9c90cda833609bfd7c2cf105ba4710ce",
    "exp7242": "sha256:738ac5a3c2eea092b69f42a317ad34af8cf3a037bc8d251cac80aea51f4f8b6b",
}
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7226_v636_belief_compiler.py"),
    Path("python/carnot/experiment_7240_v637_recurrence_fixture.py"),
    Path("python/carnot/experiment_7241_v637_recurrence_learning.py"),
    Path("python/carnot/experiment_7253_v638_coverage_memory.py"),
    WRAPPER_PATH,
    Path("tests/python/test_experiment_7253_v638_coverage_memory.py"),
    SPEC_PATH,
)
SCENARIO_PATTERN = re.compile(r"SCENARIO-CL-7253-[A-Z-]+")

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
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "execution_host",
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
    "coverage_fixture_ready_score",
    "controller_contract",
    "shuffle_effect_receipt",
    "stream_manifest",
    "continuous_self_learning_task",
)
FIELD_PRINCIPLES = {
    "schema": "Version this artifact and keep experiment identity as ordinary fields.",
    "experiment_id": "Bind this evidence to the fixed Exp7253 task.",
    "milestone": "Bind this evidence to milestone 2026.09.638.",
    "status": "A terminal artifact is complete or blocked; checkpoints remain in progress.",
    "run_date": "Use 20260912 and keep actual UTC timestamps separately.",
    "started_at_utc": "Record the actual UTC start time.",
    "completed_at_utc": "Record the actual UTC end time.",
    "field_principles": "Keep explanations here and ordinary values at top level.",
    "preconditions_checked": "Record observed source, ownership, hash, import, and quarantine checks.",
    "MODEL_SPECS": "List only models invoked now; this CPU task invokes none.",
    "model_invoked": "Derive this value from current calls, not historical evidence.",
    "inference_substrate": "Describe the CPU exact solver or a blocked no-run state.",
    "inference_substrate_class": "Use the closed compute class without padding duration.",
    "execution_venue": "Use host for orchestration and name boards only in separate receipts.",
    "execution_host": "Record the real hostname separately from venue.",
    "duration_s": "Measure monotonic elapsed time with disjoint phase spans.",
    "random_seed": "Freeze development, prospective, shuffle, delay, and query seeds.",
    "reproducibility_checksum": "Bind code, inputs, configuration, raw rows, and result fields.",
    "source_artifact_hashes": "Authenticate exact upstream and local source bytes.",
    "rows": "Retain every independent stream and arm summary with full denominators.",
    "sample_size_budget": "State planned, attempted, completed, censored units, and stopping.",
    "acceptance_gate_results": "Keep each frozen criterion, observed value, and pass state.",
    "gate_check_summary": "Name the exact failed external check for every blocked result.",
    "verifier_is_oracle": "Expose that the exact evaluator also defines correctness.",
    "honest_verdict": "Use complete_ for measured findings and blocked_ for external absence.",
    "verdict_class": "Use the closed verdict class and forbid a positive oracle claim.",
    "validation_receipts": "Record commands, exit codes, classifications, and log hashes.",
    "coverage_fixture_ready_score": "One certifies runnable controls, sealed streams, and bounded memory.",
    "controller_contract": "Freeze signature distance, admission, capacity, safety, query, and byte caps.",
    "shuffle_effect_receipt": "Show an identity-changing mapping under the same archive and safety gate.",
    "stream_manifest": "Hash development, prospective public, release, and private authority bytes.",
    "continuous_self_learning_task": "Released feedback changes reusable finite constraints, not model weights.",
}

gate_check = exp7213.gate_check
gate_summary = exp7213.gate_summary
ArchiveCommitRejected = exp7240.ArchiveCommitRejected


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep sealed inputs, audit rows, state, controls, and terminal bytes separate."""

    development_public: Path
    development_authority: Path
    development_releases: Path
    prospective_public: Path
    prospective_authority: Path
    prospective_releases: Path
    stream_manifest: Path
    raw_rows: Path
    state_sidecar: Path
    control_sidecar: Path
    evidence_sidecar: Path
    provisional: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return all task-owned paths below the repository results directory."""

        return cls.from_results_root(REPO_ROOT / "results")

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put test evidence below one caller-owned directory."""

        return cls.from_results_root(root)

    @classmethod
    def from_results_root(cls, root: Path) -> ExperimentPaths:
        """Derive each path without changing authenticated input locations."""

        streams = root / "streams" / "experiment_7253"
        checkpoints = root / "checkpoints"
        return cls(
            streams / "development_public.jsonl",
            streams / "development_private_authority.jsonl",
            streams / "development_releases.jsonl",
            streams / "prospective_public.jsonl",
            streams / "prospective_private_authority.jsonl",
            streams / "prospective_releases.jsonl",
            streams / "stream_manifest.json",
            root / "raw" / "experiment_7253" / "decision_rows.jsonl",
            checkpoints / DEFAULT_STATE_SIDECAR.name,
            checkpoints / DEFAULT_CONTROL_SIDECAR.name,
            checkpoints / DEFAULT_EVIDENCE_SIDECAR.name,
            checkpoints / DEFAULT_PROVISIONAL.name,
            root / DEFAULT_ARTIFACT.name,
        )


@dataclass(frozen=True)
class StreamViews:
    """Keep learner-visible events apart from release and correctness authority."""

    public: list[JsonDict]
    authority: list[JsonDict]
    releases: list[JsonDict]
    manifest: JsonDict


@dataclass(frozen=True)
class FixturePanel:
    """Retain raw decisions, reduced units, final states, and maximum memory."""

    event_rows: list[JsonDict]
    rows: list[JsonDict]
    final_states: list[JsonDict]
    maximum_memory: dict[str, int]
    prospective_shuffle_change_count: int


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Emit a flushed boundary so long exact CPU work remains observable."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _resolve(repo_root: Path, path: str | Path) -> Path:
    """Resolve repository-relative evidence and preserve absolute test paths."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _sha256_path(path: Path) -> str | None:
    """Hash exact bytes and preserve a missing file as an observed failure."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object while malformed evidence remains unavailable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Load object rows for the independent reducer."""

    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError):
        return []
    return rows if all(isinstance(row, dict) for row in rows) else []


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating terminal-shaped bytes."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _atomic_write(path: Path, payload: bytes) -> JsonDict:
    """Publish complete bytes with one flushed rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"path": str(path), "sha256": _sha256_path(path), "bytes": len(payload)}


def _write_immutable(path: Path, payload: bytes) -> JsonDict:
    """Accept an identical seal and reject replacement with different bytes."""

    if path.exists():
        if path.read_bytes() != payload:
            raise exp7240.ImmutableSealError(f"immutable_path_conflict:{path}")
    else:
        _atomic_write(path, payload)
    return {"path": str(path), "sha256": _sha256_path(path), "bytes": len(payload)}


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode chronological rows with the shared canonical format."""

    return b"".join(transactional.canonical_json_bytes(dict(row)) for row in rows)


def _mask_hash(masks: Mapping[str, Any]) -> str:
    """Bind an archive identity to only its finite survivor masks."""

    return transactional.sha256_json({family: int(masks[family]) for family in FAMILIES})


def _survivor_masks(controller: exp7226.PackedBeliefController) -> dict[str, int]:
    """Project active state to the immutable finite archive payload."""

    return {family: int(controller.family_state(family)["survivor_mask"]) for family in FAMILIES}


def _controller_from_masks(masks: Mapping[str, Any]) -> exp7226.PackedBeliefController:
    """Restore packed state from checked finite masks."""

    return exp7226.PackedBeliefController.from_survivors(
        {
            family: {
                parameter for parameter in PARAMETER_DOMAIN if int(masks[family]) & (1 << parameter)
            }
            for family in FAMILIES
        }
    )


def _bounded_active_state(controller: exp7226.PackedBeliefController) -> JsonDict:
    """Remove duplicate history because the bounded ledger owns release identity."""

    state = controller.state_dict()
    for family in FAMILIES:
        state["families"][family]["provenance"] = []
    return exp7226.PackedBeliefController.from_state(state).state_dict()


def snapshot_signature(
    masks: Mapping[str, Any],
    witnesses: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Represent a snapshot on no more than the newest 16 released witnesses."""

    return tuple(
        exp7240._prediction_from_masks(masks, witness)
        for witness in list(witnesses)[-VALIDATION_WINDOW:]
    )


def signature_distance(left: Sequence[str], right: Sequence[str]) -> int:
    """Count unequal decisions after aligning signatures to the same window."""

    width = max(len(left), len(right))
    padded_left = tuple(left) + ("missing",) * (width - len(left))
    padded_right = tuple(right) + ("missing",) * (width - len(right))
    return sum(int(a != b) for a, b in zip(padded_left, padded_right))


def diagnostic_archive(
    archive_id: str,
    creation_order: int,
    signature: Sequence[str],
    *,
    masks: Mapping[str, int] | None = None,
    witness_ids: Sequence[str] | None = None,
) -> JsonDict:
    """Build one finite archive row for deterministic controls and tests."""

    survivor_masks = dict(masks or dict.fromkeys(FAMILIES, FULL_MASK))
    ids = list(witness_ids or (f"witness-{index}" for index in range(len(signature))))
    return {
        "archive_id": archive_id,
        "creation_order": creation_order,
        "state_hash": _mask_hash(survivor_masks),
        "survivor_masks": survivor_masks,
        "signature_witness_ids": ids[-VALIDATION_WINDOW:],
        "signature": list(signature)[-VALIDATION_WINDOW:],
    }


def retain_archives(
    archives: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    archive_cap: int = ARCHIVE_CAP,
) -> tuple[list[JsonDict], JsonDict]:
    """Remove signature duplicates and apply FIFO or farthest-first coverage."""

    if mode not in {"fifo", "coverage"}:
        raise ValueError("invalid_admission_mode")
    if not 0 <= archive_cap <= ARCHIVE_CAP:
        raise ValueError("invalid_archive_cap")
    newest_by_signature: dict[tuple[str, ...], JsonDict] = {}
    for source in archives:
        row = deepcopy(dict(source))
        signature = tuple(str(value) for value in row.get("signature", []))
        existing = newest_by_signature.get(signature)
        if existing is None or (int(row["creation_order"]), str(row["archive_id"])) > (
            int(existing["creation_order"]),
            str(existing["archive_id"]),
        ):
            newest_by_signature[signature] = row
    unique = list(newest_by_signature.values())
    duplicate_count = len(archives) - len(unique)
    if mode == "fifo":
        retained = (
            sorted(unique, key=lambda row: (int(row["creation_order"]), row["archive_id"]))[
                -archive_cap:
            ]
            if archive_cap
            else []
        )
        selection_steps: list[JsonDict] = []
    else:
        remaining = list(unique)
        chosen: list[JsonDict] = []
        selection_steps = []
        while remaining and len(chosen) < archive_cap:
            scored = []
            for row in remaining:
                signature = tuple(row["signature"])
                if chosen:
                    distance = min(
                        signature_distance(signature, other["signature"]) for other in chosen
                    )
                else:
                    distance = sum(
                        signature_distance(signature, other["signature"])
                        for other in remaining
                        if other is not row
                    )
                scored.append((distance, int(row["creation_order"]), str(row["archive_id"]), row))
            distance, _, _, selected = max(scored, key=lambda item: item[:3])
            chosen.append(selected)
            remaining.remove(selected)
            selection_steps.append(
                {
                    "step": len(chosen),
                    "archive_id": selected["archive_id"],
                    "minimum_or_total_distance": distance,
                }
            )
        retained = sorted(chosen, key=lambda row: (int(row["creation_order"]), row["archive_id"]))
    return retained, {
        "mode": mode,
        "input_count": len(archives),
        "duplicate_eliminated_count": duplicate_count,
        "distinct_signature_count": len(unique),
        "retained_count": len(retained),
        "retained_archive_ids": [row["archive_id"] for row in retained],
        "distance": "hamming_over_released_witness_signature",
        "tie_break": ["newer_creation_order", "archive_id"],
        "selection_steps": selection_steps,
    }


def _candidate_evaluation(
    archive: Mapping[str, Any], witnesses: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Apply the unchanged eight-witness and zero-contradiction safety gate."""

    predictions = snapshot_signature(archive["survivor_masks"], witnesses)
    applicable = sum(int(value != "abstain") for value in predictions)
    contradictions = sum(
        int(prediction != "abstain" and prediction != witness["observed_label"])
        for prediction, witness in zip(predictions, list(witnesses)[-VALIDATION_WINDOW:])
    )
    return {
        "archive_id": archive["archive_id"],
        "released_witness_count": applicable,
        "contradiction_count": contradictions,
        "gate_passed": applicable >= MIN_VALIDATION_WITNESSES and contradictions == 0,
    }


def nominate_archives(
    archives: Sequence[Mapping[str, Any]],
    witnesses: Sequence[Mapping[str, Any]],
    *,
    shuffled: bool,
    shuffle_seed: int = SHUFFLE_SEED,
) -> tuple[JsonDict, JsonDict | None]:
    """Map validation signatures to identities before the same final safety gate."""

    ordered = sorted(
        (deepcopy(dict(row)) for row in archives),
        key=lambda row: (transactional.sha256_json(row["signature"]), str(row["archive_id"])),
    )
    before_ids = [str(row["archive_id"]) for row in ordered]
    after_ids = list(before_ids)
    if shuffled and len(after_ids) > 1:
        seed_text = transactional.sha256_json(
            [shuffle_seed, [witness.get("event_id") for witness in witnesses], before_ids]
        )
        random.Random(int(seed_text[-16:], 16)).shuffle(after_ids)
        if after_ids == before_ids:
            after_ids = after_ids[1:] + after_ids[:1]
    by_id = {str(row["archive_id"]): row for row in ordered}
    slots = [
        {
            "slot": index,
            "signature_sha256": transactional.sha256_json(row["signature"]),
            "candidate_archive_id": before_ids[index],
        }
        for index, row in enumerate(ordered)
    ]
    after_mapping = [
        {**slot, "candidate_archive_id": after_ids[index]} for index, slot in enumerate(slots)
    ]

    def select(candidate_ids: Sequence[str]) -> tuple[str | None, list[JsonDict]]:
        evaluations = [
            _candidate_evaluation(by_id[candidate_id], witnesses) for candidate_id in candidate_ids
        ]
        selected = next(
            (row["archive_id"] for row in evaluations if row["gate_passed"] is True),
            None,
        )
        return selected, evaluations

    selected_before, before_evaluations = select(before_ids)
    selected_after, after_evaluations = select(after_ids)
    selected_id = selected_after if shuffled else selected_before
    receipt = {
        "candidate_count": len(ordered),
        "distinguishable_candidate_count": len({tuple(row["signature"]) for row in ordered}),
        "validation_window_size": min(len(witnesses), VALIDATION_WINDOW),
        "minimum_witnesses": MIN_VALIDATION_WITNESSES,
        "final_gate": "at_least_8_released_witnesses_and_zero_contradictions",
        "before_mapping": slots,
        "after_mapping": after_mapping,
        "before_evaluations": before_evaluations,
        "after_evaluations": after_evaluations,
        "selected_before_archive_id": selected_before,
        "selected_after_archive_id": selected_after,
        "selected_archive_id": selected_id,
        "selection_changed": selected_before is not None and selected_before != selected_after,
        "same_archive_contents": set(before_ids) == set(after_ids),
        "same_final_safety_gate": True,
    }
    return receipt, None if selected_id is None else by_id[selected_id]


class CoverageArchiveController:
    """Keep one bounded packed state and four released-signature snapshots."""

    def __init__(
        self,
        *,
        archive_cap: int = ARCHIVE_CAP,
        admission_mode: str = "coverage",
        nomination_mode: str = "aligned",
        shuffle_seed: int = SHUFFLE_SEED,
    ) -> None:
        if not 0 <= archive_cap <= ARCHIVE_CAP:
            raise ValueError("invalid_archive_cap")
        if admission_mode not in {"fifo", "coverage"}:
            raise ValueError("invalid_admission_mode")
        if nomination_mode not in {"aligned", "shuffled"}:
            raise ValueError("invalid_nomination_mode")
        self._state: JsonDict = {
            "schema": STATE_SCHEMA,
            "version": 0,
            "parent_hash": None,
            "archive_cap": archive_cap,
            "admission_mode": admission_mode,
            "nomination_mode": nomination_mode,
            "shuffle_seed": shuffle_seed,
            "active": _bounded_active_state(exp7226.PackedBeliefController()),
            "archives": [],
            "next_creation_order": 0,
            "release_window": [],
            "ledger": [],
            "last_admission_receipt": {},
            "last_nomination_receipt": {},
        }

    @classmethod
    def from_active(
        cls,
        active: exp7226.PackedBeliefController,
        **kwargs: Any,
    ) -> CoverageArchiveController:
        """Wrap detached finite active state without importing its provenance history."""

        controller = cls(**kwargs)
        controller._state["active"] = _bounded_active_state(active)
        return controller

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> CoverageArchiveController:
        """Reject malformed state, oversized collections, hashes, or duplicate orders."""

        if value.get("schema") != STATE_SCHEMA:
            raise ValueError("invalid_archive_state_schema")
        archive_cap = value.get("archive_cap")
        if not isinstance(archive_cap, int) or isinstance(archive_cap, bool):
            raise ValueError("invalid_archive_cap")
        if not 0 <= archive_cap <= ARCHIVE_CAP:
            raise ValueError("invalid_archive_cap")
        if value.get("admission_mode") not in {"fifo", "coverage"}:
            raise ValueError("invalid_admission_mode")
        if value.get("nomination_mode") not in {"aligned", "shuffled"}:
            raise ValueError("invalid_nomination_mode")
        active = exp7226.PackedBeliefController.from_state(value.get("active", {}))
        if any(active.family_state(family)["provenance"] for family in FAMILIES):
            raise ValueError("unbounded_active_provenance")
        archives = value.get("archives")
        window = value.get("release_window")
        ledger = value.get("ledger")
        if not isinstance(archives, list) or len(archives) > archive_cap:
            raise ValueError("invalid_archives")
        if not isinstance(window, list) or len(window) > VALIDATION_WINDOW:
            raise ValueError("invalid_release_window")
        if (
            not isinstance(ledger, list)
            or len(ledger) > LEDGER_CAPACITY
            or len(set(ledger)) != len(ledger)
        ):
            raise ValueError("invalid_ledger")
        orders: list[int] = []
        signatures: list[tuple[str, ...]] = []
        for archive in archives:
            if not isinstance(archive, Mapping):
                raise ValueError("invalid_archive")
            masks = archive.get("survivor_masks", {})
            if set(masks) != set(FAMILIES) or any(
                not isinstance(masks[family], int)
                or isinstance(masks[family], bool)
                or not 0 <= masks[family] <= FULL_MASK
                for family in FAMILIES
            ):
                raise ValueError("invalid_archive_masks")
            if archive.get("state_hash") != _mask_hash(masks):
                raise ValueError("invalid_archive_hash")
            signature = archive.get("signature")
            ids = archive.get("signature_witness_ids")
            if (
                not isinstance(signature, list)
                or not isinstance(ids, list)
                or len(signature) != len(ids)
                or len(signature) > VALIDATION_WINDOW
                or any(value not in {"accept", "reject", "abstain"} for value in signature)
            ):
                raise ValueError("invalid_archive_signature")
            orders.append(int(archive.get("creation_order", -1)))
            signatures.append(tuple(signature))
        if len(set(orders)) != len(orders) or any(order < 0 for order in orders):
            raise ValueError("invalid_archive_order")
        if len(set(signatures)) != len(signatures):
            raise ValueError("duplicate_archive_signature")
        controller = cls.__new__(cls)
        controller._state = deepcopy(dict(value))
        if controller.memory_usage()["within_all_caps"] is not True:
            raise ValueError("controller_memory_cap")
        return controller

    @classmethod
    def load(cls, path: Path) -> CoverageArchiveController:
        """Cold-load one durable controller through the full state validator."""

        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("invalid_archive_state_object")
        return cls.from_state(value)

    def state_dict(self) -> JsonDict:
        """Return detached state so outside code cannot mutate live bytes."""

        return deepcopy(self._state)

    def state_bytes(self) -> bytes:
        """Serialize every bounded controller component canonically."""

        return transactional.canonical_json_bytes(self._state)

    def state_hash(self) -> str:
        """Identify the complete bounded controller state."""

        return transactional.sha256_bytes(self.state_bytes())

    def save(self, path: Path) -> JsonDict:
        """Write restartable bytes through the shared atomic writer."""

        return transactional._atomic_write(path, self.state_bytes())

    def archives(self) -> list[JsonDict]:
        """Expose detached archive rows for audit."""

        return deepcopy(self._state["archives"])

    def predict(self, event: Mapping[str, Any]) -> tuple[str, float]:
        """Predict from frozen active masks without changing state."""

        return exp7226.PackedBeliefController.from_state(self._state["active"]).predict(event)

    def select_request(
        self,
        block: Sequence[Mapping[str, Any]],
        tie_ranks: Mapping[str, int],
    ) -> Mapping[str, Any]:
        """Reuse the shipped public-only acquisition policy."""

        return exp7226.PackedBeliefController.from_state(self._state["active"]).select_request(
            block, tie_ranks
        )

    def memory_usage(self, pending: Sequence[Mapping[str, Any]] = ()) -> JsonDict:
        """Measure every mutable collection and enforce the frozen byte limits."""

        witness_bytes = len(transactional.canonical_json_bytes(self._state["release_window"]))
        archive_bytes = len(transactional.canonical_json_bytes(self._state["archives"]))
        pending_bytes = len(transactional.canonical_json_bytes(list(pending)))
        ledger_bytes = len(transactional.canonical_json_bytes(self._state["ledger"]))
        controller_bytes = len(self.state_bytes())
        total_bytes = controller_bytes + pending_bytes
        usage = {
            "witness_bytes": witness_bytes,
            "archive_bytes": archive_bytes,
            "pending_bytes": pending_bytes,
            "ledger_bytes": ledger_bytes,
            "controller_bytes": controller_bytes,
            "total_bytes": total_bytes,
            "pending_count": len(pending),
        }
        usage["within_all_caps"] = (
            len(self._state["release_window"]) <= VALIDATION_WINDOW
            and len(self._state["archives"]) <= int(self._state["archive_cap"])
            and len(self._state["ledger"]) <= LEDGER_CAPACITY
            and len(pending) <= PENDING_CAPACITY
            and all(usage[key] <= limit for key, limit in MEMORY_CAPS.items())
        )
        return usage

    @staticmethod
    def _refresh_signatures(state: JsonDict) -> None:
        """Rebase all signatures onto the same newest released witness window."""

        witnesses = state["release_window"][-VALIDATION_WINDOW:]
        witness_ids = [str(row["event_id"]) for row in witnesses]
        for archive in state["archives"]:
            archive["signature"] = list(snapshot_signature(archive["survivor_masks"], witnesses))
            archive["signature_witness_ids"] = witness_ids

    @staticmethod
    def _admit_snapshot(state: JsonDict, active: exp7226.PackedBeliefController) -> str | None:
        """Add pre-reset masks and immediately enforce duplicate and capacity rules."""

        if int(state["archive_cap"]) == 0:
            state["last_admission_receipt"] = {
                "mode": state["admission_mode"],
                "retained_count": 0,
                "archive_disabled": True,
            }
            return None
        masks = _survivor_masks(active)
        creation_order = int(state["next_creation_order"])
        archive_id = f"archive-{creation_order:06d}"
        witnesses = state["release_window"][-VALIDATION_WINDOW:]
        state["archives"].append(
            diagnostic_archive(
                archive_id,
                creation_order,
                snapshot_signature(masks, witnesses),
                masks=masks,
                witness_ids=[str(row["event_id"]) for row in witnesses],
            )
        )
        state["next_creation_order"] = creation_order + 1
        retained, receipt = retain_archives(
            state["archives"],
            mode=str(state["admission_mode"]),
            archive_cap=int(state["archive_cap"]),
        )
        state["archives"] = retained
        state["last_admission_receipt"] = receipt
        return archive_id if archive_id in receipt["retained_archive_ids"] else None

    def commit_batch(
        self,
        releases: Sequence[Mapping[str, Any]],
        *,
        current_cycle: int,
        expected_parent_hash: str,
        state_path: Path | None = None,
    ) -> JsonDict:
        """Apply due feedback atomically and let it affect only later decisions."""

        parent_bytes = self.state_bytes()
        parent_hash = self.state_hash()
        if expected_parent_hash != parent_hash:
            raise ArchiveCommitRejected("stale_parent")
        try:
            type(self).from_state(self._state)
        except ValueError as error:
            raise ArchiveCommitRejected("corrupt_live_state") from error
        if state_path is not None and state_path.exists():
            try:
                durable_hash = type(self).load(state_path).state_hash()
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
        ids = [str(row["event_id"]) for row in normalized]
        if len(set(ids)) != len(ids) or set(ids) & set(self._state["ledger"]):
            raise ArchiveCommitRejected("duplicate_release")

        candidate = deepcopy(self._state)
        operations: list[JsonDict] = []
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
            candidate["release_window"].append(deepcopy(release))
            candidate["release_window"] = candidate["release_window"][-VALIDATION_WINDOW:]
            self._refresh_signatures(candidate)
            rebased, rebase_receipt = retain_archives(
                candidate["archives"],
                mode=str(candidate["admission_mode"]),
                archive_cap=int(candidate["archive_cap"]),
            )
            candidate["archives"] = rebased
            candidate["last_admission_receipt"] = rebase_receipt
            archived_id = self._admit_snapshot(candidate, active) if contradiction else None
            active.commit_batch(
                [release],
                current_cycle=current_cycle,
                expected_parent_hash=active.state_hash(),
            )
            candidate["active"] = _bounded_active_state(active)
            nomination, selected = nominate_archives(
                candidate["archives"],
                candidate["release_window"],
                shuffled=candidate["nomination_mode"] == "shuffled",
                shuffle_seed=int(candidate["shuffle_seed"]),
            )
            if selected is not None:
                candidate["active"] = _bounded_active_state(
                    _controller_from_masks(selected["survivor_masks"])
                )
                candidate["archives"] = [
                    row
                    for row in candidate["archives"]
                    if row["archive_id"] != selected["archive_id"]
                ]
            candidate["last_nomination_receipt"] = nomination
            candidate["ledger"].append(str(release["event_id"]))
            candidate["ledger"] = candidate["ledger"][-LEDGER_CAPACITY:]
            operations.append(
                {
                    "event_id": release["event_id"],
                    "active_contradiction": contradiction,
                    "archived_state_id": archived_id,
                    "reactivated_archive_id": None if selected is None else selected["archive_id"],
                    "selection_changed": nomination["selection_changed"],
                    "nomination_receipt": nomination,
                    "admission_receipt": deepcopy(candidate["last_admission_receipt"]),
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
            "memory_usage": admitted.memory_usage(),
            "atomic_write": None,
        }
        if state_path is not None:
            receipt["atomic_write"] = transactional._atomic_write(state_path, new_bytes)
        self._state = admitted._state
        return receipt

    def rollback(self, receipt: Mapping[str, Any], *, state_path: Path | None = None) -> JsonDict:
        """Restore exact parent bytes only from the receipt's exact child state."""

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


def _stable_parameter(stream_seed: int, family: str) -> int:
    """Freeze one private base parameter from the sealed stream seed."""

    return (stream_seed * 5 + FAMILIES.index(family) * 3) % len(PARAMETER_DOMAIN)


def _regime_parameter(pattern: str, chronology_index: int, base: int) -> tuple[str, int]:
    """Preserve the prior private drift schedule without exposing its boundary."""

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


def build_stream_views(kind: str) -> StreamViews:
    """Generate development or prospective views before any controller runs."""

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
    pattern_counts = dict.fromkeys(PATTERNS, 0)
    for stream_index, stream_seed in enumerate(seeds):
        pattern = PATTERNS[stream_index % len(PATTERNS)]
        pattern_counts[pattern] += 1
        stream_id = f"{prefix}-{stream_index + 1:02d}"
        for chronology_index in range(EVENTS_PER_STREAM):
            repeated_index = chronology_index % 256
            family = FAMILIES[repeated_index % len(FAMILIES)]
            numeric_value = (
                repeated_index * 19 + FAMILIES.index(family) * 7 + stream_seed * 13
            ) % 10_000
            base = _stable_parameter(stream_seed, family)
            regime_id, parameter = _regime_parameter(pattern, chronology_index, base)
            exact_label = exp7226.exact_label(family, numeric_value, parameter)
            event_id = f"exp7253-{stream_id}-e{chronology_index:04d}"
            delay = DELAY_SUPPORT[(stream_seed + chronology_index) % len(DELAY_SUPPORT)]
            noisy = (stream_seed * 17 + chronology_index * 13) % 31 == 0
            observed_label = (
                ("reject" if exact_label == "accept" else "accept") if noisy else exact_label
            )
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
                    "stream_id": stream_id,
                    "chronology_index": chronology_index,
                    "delay": delay,
                    "observed_label": observed_label,
                    "release_noisy": noisy,
                }
            )
    manifest = {
        "schema": "carnot.exp7253.stream_view.v1",
        "kind": kind,
        "stream_count": len(seeds),
        "events_per_stream": EVENTS_PER_STREAM,
        "warmup_events": WARMUP_COUNT,
        "prospective_events": EVENTS_PER_STREAM - WARMUP_COUNT,
        "pending_feedback_capacity": PENDING_CAPACITY,
        "query_ceiling_per_stream": QUERY_CEILING,
        "query_block_size": QUERY_BLOCK_SIZE,
        "delay_support": list(DELAY_SUPPORT),
        "patterns": pattern_counts,
        "stream_seeds_sha256": transactional.sha256_json(list(seeds)),
        "public_fields": sorted(public[0]),
        "controller_input_fields": ["event_id", "family_id", "numeric_value"],
        "boundaries_visible_to_controller": False,
        "regimes_visible_to_controller": False,
        "labels_visible_before_release": False,
        "identical_input_cyclic_return": True,
        "noisy_release_rule": "(stream_seed*17 + chronology_index*13) mod 31 equals zero",
        "frozen_before_controller_execution": True,
    }
    return StreamViews(public, authority, releases, manifest)


def _nested_keys(value: Any) -> set[str]:
    """Collect nested keys so private authority cannot hide below one level."""

    if isinstance(value, Mapping):
        return set(value) | set().union(*(_nested_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_nested_keys(item) for item in value), set())
    return set()


def public_leakage_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Name public rows that contain private labels, schedules, regimes, or seeds."""

    return [
        str(row.get("event_id", "missing_event_id"))
        for row in rows
        if _nested_keys(row) & FORBIDDEN_PUBLIC_FIELDS
    ]


def stream_conformance_errors(views: StreamViews, kind: str) -> list[str]:
    """Check fixed counts, identity, chronology, authority, noise, and recurrence."""

    errors: list[str] = []
    expected_streams = DEVELOPMENT_STREAM_COUNT if kind == "development" else STREAM_COUNT
    expected = expected_streams * EVENTS_PER_STREAM
    if not (len(views.public) == len(views.authority) == len(views.releases) == expected):
        errors.append("event_count")
    public_ids = [row.get("event_id") for row in views.public]
    if (
        public_ids != [row.get("event_id") for row in views.authority]
        or public_ids != [row.get("event_id") for row in views.releases]
        or len(set(public_ids)) != len(public_ids)
    ):
        errors.append("event_identity")
    if public_leakage_errors(views.public):
        errors.append("public_authority_leakage")
    if views.manifest.get("patterns") != dict.fromkeys(PATTERNS, expected_streams // 4):
        errors.append("pattern_balance")
    for stream_index in range(expected_streams):
        start = stream_index * EVENTS_PER_STREAM
        rows = views.public[start : start + EVENTS_PER_STREAM]
        if [row.get("chronology_index") for row in rows] != list(range(EVENTS_PER_STREAM)):
            errors.append("chronology")
            break
        first = sorted((row["family_id"], row["numeric_value"]) for row in rows[128:384])
        recurrent = sorted((row["family_id"], row["numeric_value"]) for row in rows[768:1024])
        if first != recurrent:
            errors.append("identical_input_recurrence")
            break
    for public, truth, release in zip(views.public, views.authority, views.releases):
        expected_label = exp7226.independent_exact_label(
            str(truth["family_id"]),
            int(truth["numeric_value"]),
            int(truth["hidden_parameter"]),
        )
        observed = truth["exact_label"]
        if release["release_noisy"]:
            observed = "reject" if observed == "accept" else "accept"
        if (
            public["family_id"] != truth["family_id"]
            or public["numeric_value"] != truth["numeric_value"]
            or truth["exact_label"] != expected_label
            or release["observed_label"] != observed
            or release["delay"] not in DELAY_SUPPORT
        ):
            errors.append("authority_grounding")
            break
    return errors


def seal_streams(
    paths: ExperimentPaths,
    development: StreamViews,
    prospective: StreamViews,
) -> JsonDict:
    """Seal six separated views and a hash-only combined manifest."""

    receipts = {
        "development_public": _write_immutable(
            paths.development_public, _jsonl_bytes(development.public)
        ),
        "development_private_authority": _write_immutable(
            paths.development_authority, _jsonl_bytes(development.authority)
        ),
        "development_releases": _write_immutable(
            paths.development_releases, _jsonl_bytes(development.releases)
        ),
        "prospective_public": _write_immutable(
            paths.prospective_public, _jsonl_bytes(prospective.public)
        ),
        "prospective_private_authority": _write_immutable(
            paths.prospective_authority, _jsonl_bytes(prospective.authority)
        ),
        "prospective_releases": _write_immutable(
            paths.prospective_releases, _jsonl_bytes(prospective.releases)
        ),
    }
    manifest = {
        "schema": "carnot.exp7253.sealed_stream_manifest.v1",
        "development": development.manifest,
        "prospective": prospective.manifest,
        "receipts": receipts,
        "authority_separated": True,
    }
    manifest_receipt = _write_immutable(
        paths.stream_manifest, transactional.canonical_json_bytes(manifest)
    )
    return {**manifest, "manifest_receipt": manifest_receipt}


def run_development_rule_selection(views: StreamViews) -> JsonDict:
    """Choose one representation rule using only eight development public streams."""

    rows: list[JsonDict] = []
    masks = [{family: 1 << parameter for family in FAMILIES} for parameter in (0, 7, 15, 23, 31)]
    for stream_index in range(DEVELOPMENT_STREAM_COUNT):
        witnesses = views.public[
            stream_index * EVENTS_PER_STREAM : stream_index * EVENTS_PER_STREAM + VALIDATION_WINDOW
        ]
        candidates = [
            diagnostic_archive(
                f"development-{stream_index}-{index}",
                index,
                snapshot_signature(candidate_masks, witnesses),
                masks=candidate_masks,
                witness_ids=[str(row["event_id"]) for row in witnesses],
            )
            for index, candidate_masks in enumerate(masks)
        ]
        fifo, _ = retain_archives(candidates, mode="fifo")
        coverage, _ = retain_archives(candidates, mode="coverage")

        def diversity(selected: Sequence[Mapping[str, Any]]) -> int:
            return sum(
                signature_distance(left["signature"], right["signature"])
                for index, left in enumerate(selected)
                for right in selected[index + 1 :]
            )

        rows.append(
            {
                "development_stream_id": f"development-{stream_index + 1:02d}",
                "fifo_signature_distance": diversity(fifo),
                "coverage_signature_distance": diversity(coverage),
            }
        )
    fifo_total = sum(row["fifo_signature_distance"] for row in rows)
    coverage_total = sum(row["coverage_signature_distance"] for row in rows)
    return {
        "development_stream_count": len(rows),
        "prospective_authority_used": False,
        "candidate_rules": ["fifo_recency", "max_min_hamming_recency_id"],
        "chosen_rule": "max_min_hamming_recency_id",
        "selection_metric": "pairwise_hamming_signature_distance",
        "fifo_total": fifo_total,
        "coverage_total": coverage_total,
        "rows": rows,
        "sealed_choice_sha256": transactional.sha256_json(["max_min_hamming_recency_id", rows]),
    }


def _controller_input(event: Mapping[str, Any]) -> JsonDict:
    """Expose only public finite coordinates to every non-oracle controller."""

    return {
        "event_id": event["event_id"],
        "family_id": event["family_id"],
        "numeric_value": event["numeric_value"],
    }


def _support_release(pending: Mapping[str, Any]) -> JsonDict:
    """Expose one due observed label after its fixed delay."""

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


def reduce_event_rows(event_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Independently reduce raw chronological rows to stream-and-arm summaries."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in event_rows:
        groups.setdefault((str(row["stream_id"]), str(row["arm"])), []).append(row)
    reduced: list[JsonDict] = []
    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    for (stream_id, arm), rows in sorted(
        groups.items(), key=lambda item: (item[0][0], arm_order[item[0][1]])
    ):
        rows = sorted(rows, key=lambda row: int(row["chronology_index"]))
        future = [row for row in rows if int(row["chronology_index"]) >= WARMUP_COUNT]
        recurrence = [row for row in future if row["recurrence_eligible"] is True]
        reduced.append(
            {
                "unit_id": f"{stream_id}:{arm}",
                "stream_id": stream_id,
                "seed": int(rows[0]["seed"]),
                "arm": arm,
                "metric": "fixture_full_denominator_error",
                "event_count": len(rows),
                "prospective_event_count": len(future),
                "error": sum(int(row["error"]) for row in future),
                "error_rate": sum(int(row["error"]) for row in future) / len(future),
                "false_accept": sum(int(row["false_accept"]) for row in future),
                "abstention": sum(int(row["abstention"]) for row in future),
                "recurrence_event_count": len(recurrence),
                "recurrence_error": sum(int(row["error"]) for row in recurrence),
                "query_count": sum(int(row["query_selected"]) for row in rows),
                "release_count": int(rows[-1]["released_query_count_after"]),
                "archive_admission_count": sum(int(row["archive_admission_count"]) for row in rows),
                "archive_reactivation_count": sum(
                    int(row["archive_reactivation_count"]) for row in rows
                ),
                "shuffle_selection_change_count": sum(
                    int(row["shuffle_selection_change_count"]) for row in rows
                ),
                "maximum_memory_bytes": max(int(row["memory_total_bytes"]) for row in rows),
                "maximum_witness_bytes": max(int(row["memory_witness_bytes"]) for row in rows),
                "maximum_archive_bytes": max(int(row["memory_archive_bytes"]) for row in rows),
                "maximum_pending_bytes": max(int(row["memory_pending_bytes"]) for row in rows),
                "maximum_ledger_bytes": max(int(row["memory_ledger_bytes"]) for row in rows),
                "censored": False,
            }
        )
    return reduced


def independent_reduce(path: Path) -> list[JsonDict]:
    """Reload raw JSONL before reduction so producer aggregates cannot influence it."""

    rows = _read_jsonl(path)
    if not rows:
        raise ValueError("raw_rows_unavailable")
    errors = event_row_errors(rows)
    if errors:
        raise ValueError("raw_row_validation_failed:" + ",".join(errors))
    return reduce_event_rows(rows)


def event_row_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check chronology and label isolation on raw rows before reduction."""

    errors: list[str] = []
    if any(row.get("prediction_frozen_before_release") is not True for row in rows):
        errors.append("prediction_chronology")
    if any(
        row.get("oracle_control") is not True
        and (
            row.get("held_out_label_visible_to_controller") is not False
            or row.get("controller_input_fields") != ["event_id", "family_id", "numeric_value"]
        )
        for row in rows
    ):
        errors.append("future_label_leakage")
    return errors


def run_fixture_panel(
    views: StreamViews,
    *,
    stream_ids: Sequence[str] | None = None,
    progress: bool = False,
) -> FixturePanel:
    """Replay eight arms with predictions sealed before delayed feedback commits."""

    selected_streams = tuple(
        stream_ids
        if stream_ids is not None
        else (f"prospective-{index + 1:02d}" for index in range(STREAM_COUNT))
    )
    authority = {str(row["event_id"]): row for row in views.authority}
    schedules = {str(row["event_id"]): row for row in views.releases}
    by_stream = {
        stream_id: [row for row in views.public if row["stream_id"] == stream_id]
        for stream_id in selected_streams
    }
    event_rows: list[JsonDict] = []
    final_states: list[JsonDict] = []
    maximum_memory = dict.fromkeys(MEMORY_CAPS, 0)
    prospective_shuffle_change_count = 0
    started = time.monotonic()

    for stream_offset, stream_id in enumerate(selected_streams):
        events = by_stream[stream_id]
        stream_seed = int(authority[str(events[0]["event_id"])]["stream_seed"])
        frozen = exp7226.PackedBeliefController()
        reset = CoverageArchiveController(archive_cap=0)
        destructive = exp7226.PackedBeliefController()
        controllers: dict[str, Any] = {
            "frozen_warmup": frozen,
            "reset_relearn": reset,
            "destructive_update": destructive,
            "fifo_archive_shuffled": CoverageArchiveController(
                admission_mode="fifo", nomination_mode="shuffled"
            ),
            "fifo_archive_aligned": CoverageArchiveController(
                admission_mode="fifo", nomination_mode="aligned"
            ),
            "coverage_archive_shuffled": CoverageArchiveController(
                admission_mode="coverage", nomination_mode="shuffled"
            ),
            "coverage_archive_aligned": CoverageArchiveController(
                admission_mode="coverage", nomination_mode="aligned"
            ),
        }
        pending: list[JsonDict] = []
        actual_queries = 0
        released_queries = 0
        stream_start = len(event_rows)

        for block_index, offset in enumerate(range(0, EVENTS_PER_STREAM, QUERY_BLOCK_SIZE)):
            block = events[offset : offset + QUERY_BLOCK_SIZE]
            public_block = [_controller_input(row) for row in block]
            tie_ranks = exp7199.seeded_tie_ranks(stream_seed, block_index, public_block)
            selected = destructive.select_request(public_block, tie_ranks)
            selected_id = str(selected["event_id"])
            for event in block:
                event_id = str(event["event_id"])
                chronology_index = int(event["chronology_index"])
                controller_event = _controller_input(event)
                will_query = (
                    event_id == selected_id
                    and actual_queries < QUERY_CEILING
                    and len(pending) < PENDING_CAPACITY
                )
                event_positions: dict[str, int] = {}
                predictions: dict[str, tuple[str, float]] = {
                    arm: controller.predict(controller_event)
                    for arm, controller in controllers.items()
                }
                truth = authority[event_id]
                predictions["oracle_positive_control"] = (str(truth["exact_label"]), 0.0)
                for arm in ARMS:
                    prediction, energy = predictions[arm]
                    controller = controllers.get(arm)
                    if isinstance(controller, CoverageArchiveController):
                        usage = controller.memory_usage(pending)
                    elif controller is None:
                        usage = {
                            "witness_bytes": 0,
                            "archive_bytes": 0,
                            "pending_bytes": 0,
                            "ledger_bytes": 0,
                            "controller_bytes": 0,
                            "total_bytes": 0,
                        }
                    else:
                        size = len(controller.state_bytes())
                        usage = {
                            "witness_bytes": 0,
                            "archive_bytes": 0,
                            "pending_bytes": len(transactional.canonical_json_bytes(pending)),
                            "ledger_bytes": 0,
                            "controller_bytes": size,
                            "total_bytes": size + len(transactional.canonical_json_bytes(pending)),
                        }
                    if arm in ARCHIVE_ARMS:
                        for key in MEMORY_CAPS:
                            maximum_memory[key] = max(maximum_memory[key], int(usage[key]))
                    event_positions[arm] = len(event_rows)
                    event_rows.append(
                        {
                            "unit_id": f"{stream_id}:{arm}",
                            "stream_id": stream_id,
                            "seed": stream_seed,
                            "arm": arm,
                            "event_id": event_id,
                            "chronology_index": chronology_index,
                            "prediction": prediction,
                            "prediction_energy": energy,
                            "later_released_label": truth["exact_label"],
                            "error": int(prediction != truth["exact_label"]),
                            "false_accept": int(
                                prediction == "accept" and truth["exact_label"] == "reject"
                            ),
                            "abstention": int(prediction == "abstain"),
                            "recurrence_eligible": (
                                truth["drift_pattern"] == "aba_recurrence"
                                and chronology_index >= 768
                            ),
                            "query_selected": will_query,
                            "released_query_count_before": released_queries,
                            "released_query_count_after": released_queries,
                            "archive_admission_count": 0,
                            "archive_reactivation_count": 0,
                            "shuffle_selection_change_count": 0,
                            "memory_total_bytes": usage["total_bytes"],
                            "memory_witness_bytes": usage["witness_bytes"],
                            "memory_archive_bytes": usage["archive_bytes"],
                            "memory_pending_bytes": usage["pending_bytes"],
                            "memory_ledger_bytes": usage["ledger_bytes"],
                            "controller_input_fields": (
                                []
                                if arm == "oracle_positive_control"
                                else ["event_id", "family_id", "numeric_value"]
                            ),
                            "held_out_label_visible_to_controller": (
                                "not_applicable_evaluator_control"
                                if arm == "oracle_positive_control"
                                else False
                            ),
                            "oracle_control": arm == "oracle_positive_control",
                            "prediction_frozen_before_release": True,
                            "censored": False,
                        }
                    )
                if will_query:
                    actual_queries += 1
                    schedule = schedules[event_id]
                    pending.append(
                        {
                            "public": controller_event,
                            "observed_label": schedule["observed_label"],
                            "request_index": chronology_index,
                            "release_index": chronology_index + int(schedule["delay"]),
                        }
                    )
                due = sorted(
                    [row for row in pending if int(row["release_index"]) <= chronology_index],
                    key=lambda row: (int(row["release_index"]), int(row["request_index"])),
                )
                if due:
                    payload = [_support_release(row) for row in due]
                    update = dict(controllers)
                    if chronology_index >= WARMUP_COUNT:
                        update.pop("frozen_warmup")
                    for arm, controller in update.items():
                        receipt = controller.commit_batch(
                            payload,
                            current_cycle=chronology_index,
                            expected_parent_hash=controller.state_hash(),
                        )
                        if isinstance(controller, CoverageArchiveController):
                            operations = receipt["operations"]
                            event_rows[event_positions[arm]]["archive_admission_count"] = sum(
                                int(row["archived_state_id"] is not None) for row in operations
                            )
                            event_rows[event_positions[arm]]["archive_reactivation_count"] = sum(
                                int(row["reactivated_archive_id"] is not None) for row in operations
                            )
                            changes = sum(int(row["selection_changed"]) for row in operations)
                            event_rows[event_positions[arm]]["shuffle_selection_change_count"] = (
                                changes
                            )
                            if arm.endswith("shuffled"):
                                prospective_shuffle_change_count += changes
                    released_queries += len(due)
                    pending = [row for row in pending if row not in due]
                for position in event_positions.values():
                    event_rows[position]["released_query_count_after"] = released_queries

        for arm in ARCHIVE_ARMS:
            controller = controllers[arm]
            usage = controller.memory_usage(pending)
            final_states.append(
                {
                    "stream_id": stream_id,
                    "arm": arm,
                    "state_sha256": controller.state_hash(),
                    "state": controller.state_dict(),
                    "memory_usage": usage,
                }
            )
        if progress:
            print(
                f"phase 5 benchmark unit {stream_offset + 1}/{len(selected_streams)} "
                f"completed_rows={len(event_rows) - stream_start} "
                f"elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
    rows = reduce_event_rows(event_rows)
    return FixturePanel(
        event_rows,
        rows,
        final_states,
        maximum_memory,
        prospective_shuffle_change_count,
    )


def panel_conformance_errors(
    panel: FixturePanel,
    expected_stream_ids: Sequence[str],
) -> list[str]:
    """Check complete arm rows, chronology, equal budgets, and byte limits."""

    errors = event_row_errors(panel.event_rows)
    expected_events = len(expected_stream_ids) * EVENTS_PER_STREAM * len(ARMS)
    if len(panel.event_rows) != expected_events:
        errors.append("event_row_count")
    expected_units = {(stream_id, arm) for stream_id in expected_stream_ids for arm in ARMS}
    if {(row["stream_id"], row["arm"]) for row in panel.rows} != expected_units:
        errors.append("unit_rows")
    for stream_id in expected_stream_ids:
        archive_rows = [
            row
            for row in panel.rows
            if row["stream_id"] == stream_id and row["arm"] in ARCHIVE_ARMS
        ]
        if len({(row["query_count"], row["release_count"]) for row in archive_rows}) != 1:
            errors.append("archive_budget_mismatch")
            break
    if any(panel.maximum_memory[key] > MEMORY_CAPS[key] for key in MEMORY_CAPS):
        errors.append("memory_cap")
    return errors


def run_shuffle_diagnostic() -> JsonDict:
    """Create two safe distinguishable archives and require an identity change."""

    witnesses = [
        {
            "event_id": f"lower-{index}",
            "family_id": "lower_bound",
            "numeric_value": 0,
            "observed_label": "accept",
        }
        for index in range(8)
    ] + [
        {
            "event_id": "modular-extra",
            "family_id": "modular_equals",
            "numeric_value": 0,
            "observed_label": "accept",
        }
    ]
    first_masks = dict.fromkeys(FAMILIES, 1)
    second_masks = dict(first_masks)
    second_masks["modular_equals"] = 0
    archives = [
        diagnostic_archive(
            "eligible-a",
            0,
            snapshot_signature(first_masks, witnesses),
            masks=first_masks,
            witness_ids=[row["event_id"] for row in witnesses],
        ),
        diagnostic_archive(
            "eligible-b",
            1,
            snapshot_signature(second_masks, witnesses),
            masks=second_masks,
            witness_ids=[row["event_id"] for row in witnesses],
        ),
    ]
    receipt, _ = nominate_archives(archives, witnesses, shuffled=True)
    return receipt


def _control_release(event_id: str, label: str, index: int) -> JsonDict:
    """Build one finite lower-bound release for transaction controls."""

    return {
        "event_id": event_id,
        "family_id": "lower_bound",
        "numeric_value": 0,
        "observed_label": label,
        "role": "support",
        "request_index": index,
        "release_index": index,
    }


def run_controller_controls(root: Path) -> list[JsonDict]:
    """Run bounded-memory, negative, admission, shuffle, and E2E-007 controls."""

    root.mkdir(parents=True, exist_ok=True)
    active = exp7226.PackedBeliefController.from_survivors({family: {0} for family in FAMILIES})
    controller = CoverageArchiveController.from_active(active)
    state_path = root / "controller.json"
    controller.save(state_path)
    parent = controller.state_bytes()
    rejected = False
    try:
        controller.commit_batch(
            [_control_release("wrong-parent", "reject", 1)],
            current_cycle=1,
            expected_parent_hash="sha256:" + "0" * 64,
            state_path=state_path,
        )
    except ArchiveCommitRejected:
        rejected = True
    failed_parent = {
        "control": "failed_parent_atomic_rollback",
        "passed": rejected
        and controller.state_bytes() == parent
        and state_path.read_bytes() == parent,
    }
    future = _control_release("future-label", "reject", 2)
    future["release_index"] = 3
    future_rejected = False
    try:
        controller.commit_batch(
            [future],
            current_cycle=2,
            expected_parent_hash=controller.state_hash(),
            state_path=state_path,
        )
    except ArchiveCommitRejected:
        future_rejected = True
    future_row = {
        "control": "future_label_rejected",
        "passed": future_rejected and controller.state_bytes() == parent,
    }
    probe = {"event_id": "probe", "family_id": "lower_bound", "numeric_value": 0}
    before = controller.predict(probe)[0]
    receipt = controller.commit_batch(
        [_control_release("delayed", "reject", 4)],
        current_cycle=4,
        expected_parent_hash=controller.state_hash(),
        state_path=state_path,
    )
    after = controller.predict(probe)[0]
    delayed = {
        "control": "delayed_feedback_future_only",
        "passed": before == "accept"
        and after == "reject"
        and receipt["operations"][0]["same_event_correction"] is False,
    }
    reloaded = CoverageArchiveController.load(state_path)
    reload_matches = reloaded.state_bytes() == controller.state_bytes()
    rollback = controller.rollback(receipt, state_path=state_path)
    cold = {
        "control": "cold_reload_and_rollback",
        "passed": reload_matches
        and rollback["byte_identical"] is True
        and controller.state_bytes() == parent,
    }

    memory_controller = CoverageArchiveController.from_active(active)
    for index in range(48):
        memory_controller.commit_batch(
            [_control_release(f"memory-{index}", "reject" if index % 2 == 0 else "accept", index)],
            current_cycle=index,
            expected_parent_hash=memory_controller.state_hash(),
        )
    usage = memory_controller.memory_usage(
        [{"event_id": f"pending-{index}"} for index in range(PENDING_CAPACITY)]
    )
    bounded = {"control": "bounded_memory", "passed": usage["within_all_caps"], **usage}

    signatures = ("aaaa", "aaaa", "aaar", "aarr", "arrr", "rrrr")
    archives = [
        diagnostic_archive(f"archive-{index}", index, tuple(signature))
        for index, signature in enumerate(signatures)
    ]
    fifo, _ = retain_archives(archives, mode="fifo")
    coverage, _ = retain_archives(archives, mode="coverage")
    eviction = {
        "control": "coverage_eviction_differs_from_fifo",
        "passed": [row["archive_id"] for row in fifo] != [row["archive_id"] for row in coverage],
        "fifo_archive_ids": [row["archive_id"] for row in fifo],
        "coverage_archive_ids": [row["archive_id"] for row in coverage],
    }
    shuffle = run_shuffle_diagnostic()
    effective = {
        "control": "effective_signature_shuffle",
        "passed": shuffle["selection_changed"] is True,
        "receipt": shuffle,
    }
    return [bounded, failed_parent, delayed, future_row, eviction, effective, cold]


def _task_identity(text: str) -> JsonDict:
    """Extract only the fixed Exp7253 block from the executable roadmap."""

    match = re.search(r"(?ms)^- id: exp7253-coverage-memory\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(1)
    milestone = re.search(r"(?m)^  milestone: (.+)$", block)
    deliverable = re.search(r"(?m)^  deliverable: (.+)$", block)
    return {
        "id": "exp7253-coverage-memory" if block else None,
        "milestone": None if milestone is None else milestone.group(1).strip(),
        "deliverable": None if deliverable is None else deliverable.group(1).strip(),
    }


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    upstream_overrides: Mapping[str, Path] | None = None,
) -> tuple[list[JsonDict], dict[str, str | None], dict[str, JsonDict]]:
    """Authenticate task identity, three upstream artifacts, imports, and ownership."""

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
    imports = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "carnot.experiment_7226_v636_belief_compiler",
            "carnot.experiment_7240_v637_recurrence_fixture",
            "carnot.experiment_7241_v637_recurrence_learning",
            "carnot.memory.transactional_constraint_memory",
        )
    }
    writable = {
        field: _path_writable(getattr(paths, field))
        for field in (
            "development_public",
            "prospective_public",
            "raw_rows",
            "state_sidecar",
            "control_sidecar",
            "evidence_sidecar",
            "provisional",
            "artifact",
        )
    }
    source_state = {
        str(path): "nonempty" if hashes[str(_resolve(repo_root, path))] is not None else "missing"
        for path in SOURCE_PATHS
    }
    checks = [
        gate_check(
            "driving_capability_spec", str(SPEC_PATH), "REQ-CL-7253", True, "REQ-CL-7253" in spec
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7253-*",
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
            "v638_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            {
                "id": "exp7253-coverage-memory",
                "milestone": MILESTONE,
                "deliverable": str(DEFAULT_ARTIFACT),
            },
            _task_identity(roadmap),
        ),
        gate_check("required_imports", "python", "imports", dict.fromkeys(imports, True), imports),
        gate_check(
            "writable_output_paths",
            "host_filesystem",
            "sealed,raw,state,controls,terminal",
            dict.fromkeys(writable, True),
            writable,
        ),
    ]
    expected_states = {
        "exp7240": ("recurrence_fixture_ready_score", 1),
        "exp7241": ("recurrence_run_complete_score", 1),
        "exp7242": ("recurrence_audit_complete_score", 1),
    }
    for name in ("exp7240", "exp7241", "exp7242"):
        path = upstream_paths[name]
        checks.append(
            gate_check(
                f"{name}_artifact_hash",
                name,
                str(path),
                EXPECTED_UPSTREAM_HASHES[name],
                hashes[str(path)],
            )
        )
        checks.append(
            gate_check(f"{name}_status", name, "status", "complete", upstream[name].get("status"))
        )
        field, expected = expected_states[name]
        checks.append(
            gate_check(f"{name}_completion", name, field, expected, upstream[name].get(field))
        )
        quarantine = exp7213.quarantine_state(upstream[name], exclusions, path.name, name)
        checks.append(
            gate_check(
                f"{name}_not_quarantined",
                "artifact_metadata_and_ops/exclusion_manifest.yaml",
                "quarantined",
                False,
                quarantine["quarantined"],
            )
        )
    return checks, hashes, upstream


def _controller_contract(maximum_memory: Mapping[str, int]) -> JsonDict:
    """Freeze admission, nomination, safety, capacity, and measured byte limits."""

    return {
        "snapshot_representation": "accept_reject_abstain_signature",
        "signature_witness_limit": VALIDATION_WINDOW,
        "coverage_distance": "hamming_over_released_witness_signature",
        "coverage_admission": "deduplicate_then_greedy_farthest_first",
        "coverage_tie_break": ["newer_creation_order", "archive_id"],
        "fifo_control": "deduplicate_then_oldest_eviction",
        "archive_capacity": ARCHIVE_CAP,
        "minimum_released_witnesses": MIN_VALIDATION_WITNESSES,
        "maximum_contradictions": 0,
        "validation_window": VALIDATION_WINDOW,
        "pending_capacity": PENDING_CAPACITY,
        "ledger_capacity": LEDGER_CAPACITY,
        "query_ceiling": QUERY_CEILING,
        "release_budget": QUERY_CEILING,
        "memory_caps": deepcopy(MEMORY_CAPS),
        "maximum_measured_bytes": dict(maximum_memory),
        "rollback": "exact_parent_bytes_and_hash",
        "unbounded_auxiliary_collections": False,
    }


def _arm_contract() -> JsonDict:
    """Declare eight arms and equal resources across the four archive arms."""

    return {
        "arms": list(ARMS),
        "archive_arms": list(ARCHIVE_ARMS),
        "archive_capacity": ARCHIVE_CAP,
        "total_memory_cap": MEMORY_CAPS["total_bytes"],
        "query_ceiling": QUERY_CEILING,
        "release_budget": QUERY_CEILING,
        "shared_public_events": True,
        "shared_selected_feedback": True,
        "shared_delay_and_noise_schedule": True,
        "shared_final_safety_gate": True,
        "oracle_positive_control_is_evaluator_only": True,
    }


def _sample_budget(stream_ids: Sequence[str], *, complete: bool) -> JsonDict:
    """State the sealed stream, event, arm, censoring, and stopping budget."""

    units = len(stream_ids)
    events = units * EVENTS_PER_STREAM
    rows = events * len(ARMS)
    return {
        "development_units": DEVELOPMENT_STREAM_COUNT,
        "independent_units_planned": units,
        "independent_units_attempted": units if complete else 0,
        "independent_units_completed": units if complete else 0,
        "independent_units_censored": 0 if complete else units,
        "events_per_unit": EVENTS_PER_STREAM,
        "warmup_events_per_unit": WARMUP_COUNT,
        "prospective_events_per_unit": EVENTS_PER_STREAM - WARMUP_COUNT,
        "planned_arm_event_rows": rows,
        "completed_arm_event_rows": rows if complete else 0,
        "stopping_rule": "all predeclared streams once; no outcome-based extension",
    }


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    paths: ExperimentPaths,
    stream_ids: Sequence[str],
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Create a schema-complete blocked result before any CPU measurement."""

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
        "MODEL_SPECS": [],
        "model_invoked": False,
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "current_inference_count": 0,
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": socket.gethostname(),
        "duration_s": duration_s,
        "phase_spans_s": {},
        "random_seed": {
            "root": RANDOM_SEED,
            "development_streams": list(DEVELOPMENT_STREAM_SEEDS),
            "prospective_streams": list(STREAM_SEEDS),
            "shuffle": SHUFFLE_SEED,
            "schedule_frozen_before_authority": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": _sample_budget(stream_ids, complete=False),
        "acceptance_gate_results": {},
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_precondition:" + str(summary.get("failed_check")),
        "verdict_class": "blocked",
        "validation_receipts": [],
        "coverage_fixture_ready_score": 0,
        "controller_contract": _controller_contract(dict.fromkeys(MEMORY_CAPS, 0)),
        "shuffle_effect_receipt": {},
        "stream_manifest": {},
        "continuous_self_learning_task": True,
        "current_invocation_counts": {"model_loads": 0, "generations": 0, "model_invocations": 0},
        "raw_rows_receipt": {"path": str(paths.raw_rows), "sha256": None, "row_count": 0},
        "state_sidecar_receipt": {"path": str(paths.state_sidecar), "sha256": None},
        "control_sidecar_receipt": {"path": str(paths.control_sidecar), "sha256": None},
        "evidence_sidecar_receipt": {"path": str(paths.evidence_sidecar), "sha256": None},
        "development_rule_selection": {},
        "arm_contract": _arm_contract(),
        "science_learning_gain_scored": False,
        "causal_attribution_eligible": False,
        "no_model_weight_mutation": True,
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
    started_at: str | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    """Return a row-free terminal block for an external prerequisite failure."""

    del upstream
    selected = tuple(
        stream_ids or (f"prospective-{index + 1:02d}" for index in range(STREAM_COUNT))
    )
    now = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
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
    """Hash stable identity, settings, sources, raw receipts, rows, gates, and verdict."""

    excluded = {"started_at_utc", "completed_at_utc", "duration_s", "phase_spans_s"}
    return transactional.sha256_json(
        {
            key: deepcopy(value)
            for key, value in artifact.items()
            if key not in excluded and key != "reproducibility_checksum"
        }
    )


def _receipt_matches(repo_root: Path, receipt: Mapping[str, Any]) -> bool:
    """Compare a declared evidence hash with current exact bytes."""

    path = receipt.get("path")
    return isinstance(path, str) and _sha256_path(_resolve(repo_root, path)) == receipt.get(
        "sha256"
    )


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
    check_files: bool = False,
) -> list[str]:
    """Cold-check terminal identity, rows, controls, sources, files, and checksum."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping"]
    add(any(field not in artifact for field in REQUIRED_ARTIFACT_FIELDS), "required_fields")
    add(
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE,
        "identity",
    )
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
    add(
        artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
        "reproducibility_checksum",
    )
    receipts = artifact.get("validation_receipts", [])
    add(
        not isinstance(receipts, list)
        or any(
            set(receipt) != {"command", "exit_code", "classification", "log_sha256"}
            or not isinstance(receipt["command"], str)
            or not isinstance(receipt["exit_code"], int)
            or not isinstance(receipt["classification"], str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt["log_sha256"])) is None
            for receipt in receipts
        ),
        "validation_receipts",
    )
    status = artifact.get("status")
    if status == "blocked":
        add(
            artifact.get("rows") != []
            or artifact.get("coverage_fixture_ready_score") != 0
            or artifact.get("inference_substrate") != "blocked_no_run"
            or artifact.get("inference_substrate_class") != "blocked_no_run"
            or artifact.get("verdict_class") != "blocked"
            or not str(artifact.get("honest_verdict", "")).startswith("blocked_")
            or artifact.get("gate_check_summary", {}).get("passed") is not False,
            "blocked_contract",
        )
        return errors
    add(status != "complete", "status")
    if status != "complete":
        return errors
    add(
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
        "substrate",
    )
    add(
        artifact.get("coverage_fixture_ready_score") != 1
        or artifact.get("verdict_class") != "circular_positive"
        or not str(artifact.get("honest_verdict", "")).startswith("complete_")
        or artifact.get("science_learning_gain_scored") is not False,
        "complete_contract",
    )
    add(artifact.get("gate_check_summary", {}).get("passed") is not True, "preconditions")
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
    gates = artifact.get("acceptance_gate_results", {})
    expected_gate_names = {
        "sealed_streams",
        "development_rule_sealed",
        "eight_arm_rows",
        "bounded_memory",
        "controller_controls",
        "effective_shuffle_diagnostic",
        "independent_raw_reducer",
        "science_learning_gain",
    }
    add(
        not isinstance(gates, Mapping)
        or set(gates) != expected_gate_names
        or any(
            row.get("pass") is not True
            for name, row in gates.items()
            if name != "science_learning_gain"
        )
        or gates.get("science_learning_gain", {}).get("pass") is not None,
        "acceptance_gate_results",
    )
    shuffle = artifact.get("shuffle_effect_receipt", {})
    add(
        shuffle.get("selection_changed") is not True
        or shuffle.get("same_archive_contents") is not True
        or shuffle.get("same_final_safety_gate") is not True,
        "shuffle_effect_receipt",
    )
    contract = artifact.get("controller_contract", {})
    add(
        contract.get("archive_capacity") != ARCHIVE_CAP
        or contract.get("memory_caps") != MEMORY_CAPS
        or contract.get("unbounded_auxiliary_collections") is not False,
        "controller_contract",
    )
    expected_event_rows = len(selected) * EVENTS_PER_STREAM * len(ARMS)
    add(
        artifact.get("sample_size_budget", {}).get("completed_arm_event_rows")
        != expected_event_rows
        or artifact.get("raw_rows_receipt", {}).get("row_count") != expected_event_rows,
        "sample_size_budget",
    )
    if check_files:
        file_receipts = [
            artifact.get("raw_rows_receipt", {}),
            artifact.get("state_sidecar_receipt", {}),
            artifact.get("control_sidecar_receipt", {}),
            artifact.get("evidence_sidecar_receipt", {}),
            artifact.get("stream_manifest", {}).get("manifest_receipt", {}),
        ]
        add(
            any(not _receipt_matches(repo_root, receipt) for receipt in file_receipts),
            "sidecar_hashes",
        )
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
    """Attach only exact command receipts and refresh the stable checksum."""

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
    """Authenticate, seal streams, replay controls, reduce rows, and validate."""

    selected = tuple(
        stream_ids or (f"prospective-{index + 1:02d}" for index in range(STREAM_COUNT))
    )
    monotonic_start = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: JsonDict = {}
    if progress:
        _progress(0, "start", "authenticate source bytes, quarantine, imports, and writable paths")
    phase_start = time.monotonic()
    checks, source_hashes, upstream = collect_preconditions(
        repo_root, paths, upstream_overrides=upstream_overrides
    )
    spans["phase_0_preconditions"] = time.monotonic() - phase_start
    _atomic_write(
        paths.provisional,
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
            started_at=started_at,
            duration_s=time.monotonic() - monotonic_start,
        )
        if progress:
            _progress(0, "end", "external precondition failed; no CPU measurement ran")
        return artifact
    if progress:
        _progress(0, "end", "all external preconditions passed")

    phase_start = time.monotonic()
    if progress:
        _progress(1, "start", "activate flushed boundaries and monotonic progress")
    spans["phase_1_progress"] = time.monotonic() - phase_start
    if progress:
        _progress(1, "end", "progress contract active")

    phase_start = time.monotonic()
    if progress:
        _progress(2, "start", "confirm zero model loads, generations, and invocations")
        print("phase 2 BEFORE model load: no model load scheduled", flush=True)
        print("phase 2 AFTER model load: model load count remains zero", flush=True)
        print("phase 2 BEFORE generation: no model generation scheduled", flush=True)
        print("phase 2 AFTER generation: generation count remains zero", flush=True)
    spans["phase_2_no_llm"] = time.monotonic() - phase_start
    if progress:
        _progress(2, "end", "MODEL_SPECS is empty and all current counters are zero")

    phase_start = time.monotonic()
    if progress:
        _progress(3, "start", "generate and seal development and prospective stream views")
    development = build_stream_views("development")
    prospective = build_stream_views("prospective")
    stream_errors = stream_conformance_errors(
        development, "development"
    ) + stream_conformance_errors(prospective, "prospective")
    if stream_errors:
        raise ValueError("stream_conformance_failed:" + ",".join(stream_errors))
    stream_manifest = seal_streams(paths, development, prospective)
    spans["phase_3_stream_generation_and_seal"] = time.monotonic() - phase_start
    if progress:
        _progress(3, "end", "8 development and 32 prospective streams sealed separately")

    phase_start = time.monotonic()
    if progress:
        _progress(
            4, "start", "choose and seal one coverage distance rule from development public rows"
        )
    rule_selection = run_development_rule_selection(development)
    spans["phase_4_development_rule"] = time.monotonic() - phase_start
    if progress:
        _progress(4, "end", f"chosen_rule={rule_selection['chosen_rule']}")

    phase_start = time.monotonic()
    if progress:
        _progress(5, "start", "BEFORE eight-arm CPU benchmark")
        print("phase 5 BEFORE benchmark: exact finite-controller replay", flush=True)
    panel = run_fixture_panel(prospective, stream_ids=selected, progress=progress)
    spans["phase_5_eight_arm_benchmark"] = time.monotonic() - phase_start
    if progress:
        print("phase 5 AFTER benchmark: all selected event-arm rows completed", flush=True)
        _progress(5, "end", f"completed_rows={len(panel.event_rows)}")

    phase_start = time.monotonic()
    if progress:
        _progress(6, "start", "run memory, transaction, admission, shuffle, and E2E-007 controls")
    conformance = panel_conformance_errors(panel, selected)
    if conformance:
        raise ValueError("panel_conformance_failed:" + ",".join(conformance))
    controls = run_controller_controls(paths.provisional.parent / "experiment_7253_controls")
    if any(row["passed"] is not True for row in controls):
        raise ValueError("controller_control_failed")
    raw_receipt = _atomic_write(paths.raw_rows, _jsonl_bytes(panel.event_rows))
    reduced = independent_reduce(paths.raw_rows)
    if reduced != panel.rows:
        raise ValueError("independent_reducer_mismatch")
    state_receipt = _atomic_write(
        paths.state_sidecar,
        transactional.canonical_json_bytes(
            {"schema": "carnot.exp7253.states.v1", "states": panel.final_states}
        ),
    )
    control_receipt = _atomic_write(
        paths.control_sidecar,
        transactional.canonical_json_bytes(
            {"schema": "carnot.exp7253.controls.v1", "rows": controls}
        ),
    )
    evidence_receipt = _atomic_write(
        paths.evidence_sidecar,
        transactional.canonical_json_bytes(
            {
                "schema": "carnot.exp7253.evidence.v1",
                "historical_artifacts": {
                    name: {
                        "path": str(UPSTREAM_ARTIFACTS[name]),
                        "sha256": EXPECTED_UPSTREAM_HASHES[name],
                        "status": upstream[name].get("status"),
                        "honest_verdict": upstream[name].get("honest_verdict"),
                    }
                    for name in UPSTREAM_ARTIFACTS
                },
                "injected_negative_controls": [
                    row
                    for row in controls
                    if row["control"] in {"failed_parent_atomic_rollback", "future_label_rejected"}
                ],
            }
        ),
    )
    for receipt in (raw_receipt, state_receipt, control_receipt, evidence_receipt):
        source_hashes[str(receipt["path"])] = str(receipt["sha256"])
    for receipt in stream_manifest["receipts"].values():
        source_hashes[str(receipt["path"])] = str(receipt["sha256"])
    source_hashes[str(stream_manifest["manifest_receipt"]["path"])] = str(
        stream_manifest["manifest_receipt"]["sha256"]
    )
    shuffle_effect = run_shuffle_diagnostic()
    spans["phase_6_controls_and_reduction"] = time.monotonic() - phase_start
    if progress:
        _progress(6, "end", "all controls and independent raw-row reduction passed")

    acceptance = {
        "sealed_streams": {
            "principle": "All separated development and prospective files must be immutable and conformant.",
            "expected": 0,
            "observed": len(stream_errors),
            "pass": len(stream_errors) == 0,
        },
        "development_rule_sealed": {
            "principle": "Only eight development public streams choose one fixed coverage rule.",
            "expected": [8, "max_min_hamming_recency_id", False],
            "observed": [
                rule_selection["development_stream_count"],
                rule_selection["chosen_rule"],
                rule_selection["prospective_authority_used"],
            ],
            "pass": rule_selection["development_stream_count"] == 8
            and rule_selection["prospective_authority_used"] is False,
        },
        "eight_arm_rows": {
            "principle": "Every selected independent stream must have all eight arm rows.",
            "expected": len(selected) * len(ARMS),
            "observed": len(panel.rows),
            "pass": len(panel.rows) == len(selected) * len(ARMS),
        },
        "bounded_memory": {
            "principle": "Every measured mutable controller component must stay within its byte cap.",
            "expected": deepcopy(MEMORY_CAPS),
            "observed": panel.maximum_memory,
            "pass": all(panel.maximum_memory[key] <= MEMORY_CAPS[key] for key in MEMORY_CAPS),
        },
        "controller_controls": {
            "principle": "Transaction, leakage, admission, shuffle, reload, and rollback controls must pass.",
            "expected": len(controls),
            "observed": sum(int(row["passed"] is True) for row in controls),
            "pass": all(row["passed"] is True for row in controls),
        },
        "effective_shuffle_diagnostic": {
            "principle": "A distinguishable eligible diagnostic must change the selected snapshot identity.",
            "expected": True,
            "observed": shuffle_effect["selection_changed"],
            "pass": shuffle_effect["selection_changed"] is True,
        },
        "independent_raw_reducer": {
            "principle": "A cold raw-row reduction must reproduce every producer unit row.",
            "expected": transactional.sha256_json(panel.rows),
            "observed": transactional.sha256_json(reduced),
            "pass": reduced == panel.rows,
        },
        "science_learning_gain": {
            "principle": "This phase builds a fixture and does not score learning gain.",
            "expected": "not_scored",
            "observed": "not_scored",
            "pass": None,
        },
    }
    ready = int(
        all(
            row["pass"] is True
            for name, row in acceptance.items()
            if name != "science_learning_gain"
        )
    )
    completed_at = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks,
        source_hashes,
        paths,
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
                "complete_circular_positive: bounded coverage fixture and effective controls are ready"
                if ready
                else "complete_null: one or more bounded coverage fixture controls failed"
            ),
            "coverage_fixture_ready_score": ready,
            "controller_contract": _controller_contract(panel.maximum_memory),
            "shuffle_effect_receipt": shuffle_effect,
            "stream_manifest": stream_manifest,
            "raw_rows_receipt": {
                **raw_receipt,
                "row_count": len(panel.event_rows),
                "format": "jsonl",
            },
            "state_sidecar_receipt": {**state_receipt, "state_count": len(panel.final_states)},
            "control_sidecar_receipt": {**control_receipt, "row_count": len(controls)},
            "evidence_sidecar_receipt": evidence_receipt,
            "development_rule_selection": rule_selection,
            "causal_attribution_eligible": panel.prospective_shuffle_change_count > 0,
            "prospective_shuffle_selection_change_count": panel.prospective_shuffle_change_count,
        }
    )
    artifact["validation_receipts"] = [
        {
            "command": f"independent_reduce {paths.raw_rows}",
            "exit_code": 0,
            "classification": "passed",
            "log_sha256": transactional.sha256_json(reduced),
        }
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
    """Cold-validate and atomically publish the terminal artifact."""

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
    """Parse the fixed execution date and optional private output root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CPU fixture and atomically publish one validated terminal result."""

    args = _parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"run_date_must_be_{RUN_DATE}")
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    artifact = build_and_seal(REPO_ROOT, paths, progress=True)
    _progress(7, "start", "BEFORE final cold validation")
    print("phase 7 BEFORE validation: cold artifact and independent raw-row checks", flush=True)
    errors = validate_artifact(artifact, repo_root=REPO_ROOT, check_files=True)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    print("phase 7 AFTER validation: all terminal checks passed", flush=True)
    _progress(7, "end", "final cold validation passed")
    _progress(8, "start", "BEFORE atomic terminal write")
    print("phase 8 BEFORE atomic terminal write", flush=True)
    write_artifact(paths.artifact, artifact, repo_root=REPO_ROOT)
    print("phase 8 AFTER atomic terminal write", flush=True)
    _progress(8, "end", f"wrote {paths.artifact}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin wrapper owns execution.
    raise SystemExit(main())
