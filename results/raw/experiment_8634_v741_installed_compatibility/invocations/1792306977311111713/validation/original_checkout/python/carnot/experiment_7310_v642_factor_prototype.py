"""Prototype bounded factor-local revision from delayed feedback.

Each public family owns its own finite predicate bitset. Released witnesses can
repair only that family, so valid constraints in other families stay unchanged.
This module measures controller and fixture mechanics. It does not claim that
the representation improves prospective accuracy.

Spec refs: REQ-CL-7310 and SCENARIO-CL-7310-*.
"""

from __future__ import annotations

import argparse
import base64
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
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any
import zlib

import yaml

from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7281_v640_admission_prototype as admission
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7310
SCHEMA = "carnot.exp7310.v642_factor_prototype.v1"
STATE_SCHEMA = "carnot.factor_local_revision_controller.v1"
MILESTONE = "2026.09.642"
RUN_DATE = "20260914"
RANDOM_SEED = 7_310_000
DEVELOPMENT_STREAM_SEEDS = tuple(range(7_310_001, 7_310_009))
EVALUATION_STREAM_SEEDS = tuple(range(7_310_101, 7_310_125))
DEVELOPMENT_STREAM_COUNT = 8
EVALUATION_STREAM_COUNT = 24
EVENTS_PER_STREAM = 1_024
WARMUP_COUNT = 128
FUTURE_LABEL_COUNT = 128
FEEDBACK_DELAY = 4
WITNESS_LIMIT = 16
DEDUPLICATION_LIMIT = WARMUP_COUNT + FUTURE_LABEL_COUNT + FEEDBACK_DELAY
MEMORY_CAP_BYTES = admission.MEMORY_CAP_BYTES
FUTURE_LABEL_POSITIONS = tuple(128 + 7 * index for index in range(FUTURE_LABEL_COUNT))
FAMILIES = tuple(admission.FAMILIES)
PARAMETER_DOMAIN = tuple(admission.PARAMETER_DOMAIN)
FULL_MASK = admission.FULL_MASK
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INVOCATION_COUNTS = {
    "attempted_model_loads": 0,
    "completed_model_loads": 0,
    "failed_model_loads": 0,
    "cancelled_model_loads": 0,
    "in_flight_model_loads": 0,
    "attempted_generation_calls": 0,
    "completed_generation_calls": 0,
    "failed_generation_calls": 0,
    "cancelled_generation_calls": 0,
    "in_flight_generation_calls": 0,
    "usable_answers": 0,
}
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"
ARMS = (
    "factor_local_retained_witnesses",
    "global_reset_on_contradiction",
    "local_reset_without_retained_witnesses",
    "frozen_warmup",
    "label_shuffled_factor_local_revision",
)
CONTROL_NAMES = (
    "factor_independence",
    "contradictory_delayed_label",
    "ambiguous_suffix",
    "poison_rejection",
    "byte_cap_exhaustion",
    "rollback",
    "cold_restart",
    "read_only_prediction",
    "opt_in_pipeline",
)
FORBIDDEN_PUBLIC_FIELDS = {
    "exact_label",
    "observed_label",
    "target_parameter",
    "regime_id",
    "stream_seed",
    "drift_times",
    "release_index",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
WRAPPER_PATH = Path("scripts/experiments/experiment_7310_v642_factor_prototype.py")
TEST_PATH = Path("tests/python/test_experiment_7310_v642_factor_prototype.py")
COVERAGE_PATH = Path("tests/python/coverage_experiment_7310.py")
DEFAULT_ARTIFACT = Path("results/experiment_7310_v642_factor_prototype.json")
HISTORICAL_ARTIFACT = Path("results/experiment_7297_v641_mixture_audit.json")
SCENARIO_PATTERN = re.compile(r"SCENARIO-CL-7310-[A-Z0-9]+(?:-[A-Z0-9]+)*")
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7295_v641_mixture_prototype.py"),
    Path("python/carnot/experiment_7296_v641_mixture_learning.py"),
    Path("python/carnot/experiment_7297_v641_mixture_audit.py"),
    Path("python/carnot/experiment_7212_v635_refinement_fixture.py"),
    Path("python/carnot/experiment_7267_v639_recognition_prototype.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("python/carnot/experiment_7310_v642_factor_prototype.py"),
    WRAPPER_PATH,
    TEST_PATH,
    COVERAGE_PATH,
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
    "phase_durations_s",
    "field_principles",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "current_model_load_count",
    "current_generation_count",
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
    "factor_fixture_ready_score",
    "continuous_self_learning_task",
    "factor_state_schema",
    "stream_manifest",
    "learning_acceptance_contract",
    "hardware_path",
)

FIELD_PRINCIPLES = {
    "schema": "Version the record while keeping ordinary top-level experiment_id and milestone.",
    "experiment_id": "Bind the record to the active experiment task.",
    "milestone": "Bind the record to milestone 2026.09.642.",
    "status": "Write the terminal result only after the work and checks; checkpoints remain separate.",
    "run_date": "Use 20260914 with actual UTC start and end and monotonic phase timing.",
    "started_at_utc": "Record the actual UTC start of this invocation.",
    "completed_at_utc": "Record the actual UTC terminal decision.",
    "phase_durations_s": "Measure disjoint phase spans without a duration floor.",
    "field_principles": "Keep explanations explicit while values remain ordinary top-level fields.",
    "preconditions_checked": "Hash real inputs and record actual availability and failed checks.",
    "MODEL_SPECS": "List actual current executable models; this CPU experiment uses none.",
    "model_invoked": "True would include any attempted load or generation; no model is attempted here.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled, and in-flight model work.",
    "current_model_load_count": "Count current model loads; this experiment has zero.",
    "current_generation_count": "Count current generations; this experiment has zero.",
    "inference_substrate": "Describe the actual CPU exact-solver and simulator work.",
    "inference_substrate_class": "Use the recognized CPU class and never pad duration.",
    "execution_venue": "Record host work as host, not GPU or FPGA execution.",
    "duration_s": "Measure total monotonic elapsed time including failed work.",
    "random_seed": "Seal development and independent evaluation seeds before outcomes.",
    "reproducibility_checksum": "Bind code, inputs, configuration, and raw evidence.",
    "source_artifact_hashes": "Authenticate producer identity, terminal class, and quarantine state.",
    "rows": "Record every development stream and arm with metrics, costs, errors, abstentions, and censoring.",
    "sample_size_budget": "Keep planned, attempted, complete, and censored counts with the frozen stopping rule.",
    "acceptance_gate_results": "Every check has expected, observed, passed, and a purpose.",
    "gate_check_summary": "Every block names the exact upstream, check, field, observed, and expected value.",
    "verifier_is_oracle": "Shared evaluator authority permits circular_positive only.",
    "honest_verdict": "Completed findings start complete_; external failure starts blocked_.",
    "verdict_class": "Use exactly positive, circular_positive, null, blocked, disqualified, or partial.",
    "validation_receipts": "Record exact commands, scope, exit codes, elapsed time, and log hashes.",
    "factor_fixture_ready_score": "One requires executable bounded state, independent streams, controls, and E2E.",
    "continuous_self_learning_task": "Released feedback changes constraints used by later queries.",
    "factor_state_schema": "Explicit survivor and witness ownership prevents hidden unlimited memory.",
    "stream_manifest": "Frozen seeds, strata, event counts, and feedback schedules prevent outcome tuning.",
    "learning_acceptance_contract": "Freeze paired error, safety, coverage, recurrence, and causality thresholds before evaluation.",
    "hardware_path": "Report measured CPU work and a fixed-width device path without claiming projected speedup.",
}


class FactorRevisionRejected(ValueError):
    """Reject invalid evidence or storage before controller bytes change."""


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep sealed streams, raw rows, controls, candidates, and terminal bytes separate."""

    raw_dir: Path
    development_public: Path
    development_authority: Path
    development_releases: Path
    evaluation_public: Path
    evaluation_authority: Path
    evaluation_releases: Path
    stream_manifest: Path
    panel_rows: Path
    control_rows: Path
    terminal_candidate: Path
    validation_dir: Path
    artifact: Path
    historical_artifact: Path

    @classmethod
    def defaults(cls) -> FactorPaths:
        """Return task-owned repository paths and the authenticated historical input."""

        return cls.from_results_root(REPO_ROOT / "results", REPO_ROOT / HISTORICAL_ARTIFACT)

    @classmethod
    def under(cls, root: Path) -> FactorPaths:
        """Put test outputs and the replaceable historical fixture below a private root."""

        return cls.from_results_root(root, root / HISTORICAL_ARTIFACT.name)

    @classmethod
    def from_results_root(cls, root: Path, historical: Path) -> FactorPaths:
        """Derive every declared destination without creating terminal evidence."""

        raw = root / "raw" / "experiment_7310_v642_factor_prototype"
        return cls(
            raw,
            raw / "development_public.jsonl",
            raw / "development_authority.jsonl",
            raw / "development_releases.jsonl",
            raw / "evaluation_public.jsonl",
            raw / "evaluation_authority.jsonl",
            raw / "evaluation_releases.jsonl",
            raw / "stream_manifest.json",
            raw / "development_panel_rows.jsonl",
            raw / "controller_controls.json",
            raw / "terminal_candidate.json",
            raw / "validation",
            root / DEFAULT_ARTIFACT.name,
            historical,
        )


FactorPaths = ExperimentPaths


@dataclass(frozen=True)
class StreamViews:
    """Separate learner-visible events from releases and evaluator authority."""

    public: list[JsonDict]
    authority: list[JsonDict]
    releases: list[JsonDict]
    shuffled_labels: dict[str, str]
    manifest: JsonDict


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Print one flushed phase boundary for the external task watchdog."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _canonical_bytes(value: Any) -> bytes:
    """Use the transactional memory encoding for every charged byte count."""

    return transactional.canonical_json_bytes(value)


def _sha256_path(path: Path) -> str | None:
    """Hash exact file bytes while absence stays different from an empty file."""

    try:
        return transactional.sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object while malformed or absent evidence stays unavailable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating output evidence early."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _atomic_write(path: Path, value: Any) -> JsonDict:
    """Publish one complete JSON value with the existing durable writer."""

    data = value if isinstance(value, bytes) else _canonical_bytes(value)
    receipt = transactional._atomic_write(path, data)
    return {**receipt, "path": str(path), "sha256": transactional.sha256_bytes(data)}


def _write_immutable(path: Path, value: Any) -> JsonDict:
    """Permit an identical rerun and reject replacement of sealed evidence."""

    data = value if isinstance(value, bytes) else _canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != data:
            raise FactorRevisionRejected(f"immutable_evidence_mismatch:{path}")
        return {"path": str(path), "sha256": transactional.sha256_bytes(data), "reused": True}
    return {**_atomic_write(path, data), "reused": False}


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode chronological rows with stable one-object-per-line bytes."""

    return b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)


def evaluator_exact_label(family_id: str, numeric_value: int, parameter: int) -> str:
    """Evaluate hidden truth with expressions separate from controller filtering."""

    _, value = divmod(numeric_value, len(PARAMETER_DOMAIN))
    if family_id == "lower_bound":
        accepted = not value < parameter
    elif family_id == "upper_bound":
        accepted = not value > parameter
    elif family_id == "modular_equals":
        accepted = (value - parameter) % len(PARAMETER_DOMAIN) == 0
    elif family_id == "cyclic_window":
        accepted = any(value == (parameter + offset) % len(PARAMETER_DOMAIN) for offset in range(8))
    else:
        raise ValueError(f"unknown_family:{family_id}")
    return "accept" if accepted else "reject"


def _predicate_label(family: str, value: int, parameter: int) -> str:
    """Execute one candidate predicate inside the learner boundary."""

    return admission.exact_label(family, value, parameter)


def _consistent_mask(family: str, witnesses: Sequence[Mapping[str, Any]]) -> int:
    """Return every finite predicate consistent with all released witnesses."""

    mask = 0
    for parameter in PARAMETER_DOMAIN:
        if all(
            _predicate_label(family, int(row["numeric_value"]), parameter) == row["observed_label"]
            for row in witnesses
        ):
            mask |= 1 << parameter
    return mask


def _normalize_masks(masks: Mapping[str, Any]) -> dict[str, int]:
    """Copy one complete collection of nonempty finite factor masks."""

    if set(masks) != set(FAMILIES):
        raise FactorRevisionRejected("invalid_mask_families")
    normalized = {family: int(masks[family]) for family in FAMILIES}
    if any(mask <= 0 or mask & ~FULL_MASK for mask in normalized.values()):
        raise FactorRevisionRejected("invalid_survivor_mask")
    return normalized


class FactorLocalController:
    """Revise bounded finite factors only after their matching labels release."""

    def __init__(
        self,
        masks: Mapping[str, Any],
        *,
        memory_cap_bytes: int = MEMORY_CAP_BYTES,
        mode: str = "factor_local",
        retain_witnesses: bool = True,
        update_enabled: bool = True,
    ) -> None:
        if mode not in {"factor_local", "global_reset", "local_reset"}:
            raise FactorRevisionRejected("invalid_controller_mode")
        initial = _normalize_masks(masks)
        self._state: JsonDict = {
            "schema": STATE_SCHEMA,
            "memory_cap_bytes": int(memory_cap_bytes),
            "witness_limit": WITNESS_LIMIT,
            "deduplication_limit": DEDUPLICATION_LIMIT,
            "mode": mode,
            "retain_witnesses": bool(retain_witnesses),
            "update_enabled": bool(update_enabled),
            "initial_masks": initial,
            "families": {
                family: {"survivor_mask": initial[family], "witnesses": [], "revision_count": 0}
                for family in FAMILIES
            },
            "pending_releases": [],
            "used_release_ids": [],
            "rollback": None,
            "release_count": 0,
            "predicate_evaluation_count": 0,
        }
        if not self._within_cap():
            raise FactorRevisionRejected("factor_memory_cap")

    @classmethod
    def from_masks(
        cls,
        masks: Mapping[str, Any],
        *,
        memory_cap_bytes: int = MEMORY_CAP_BYTES,
        mode: str = "factor_local",
        retain_witnesses: bool = True,
        update_enabled: bool = True,
    ) -> FactorLocalController:
        """Start one controller from released warmup survivor masks."""

        return cls(
            masks,
            memory_cap_bytes=memory_cap_bytes,
            mode=mode,
            retain_witnesses=retain_witnesses,
            update_enabled=update_enabled,
        )

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> FactorLocalController:
        """Restore canonical state only after all finite and memory bounds pass."""

        if value.get("schema") != STATE_SCHEMA:
            raise FactorRevisionRejected("invalid_factor_state")
        controller = cls.__new__(cls)
        controller._state = deepcopy(dict(value))
        try:
            _normalize_masks(controller._state["initial_masks"])
            families = controller._state["families"]
            if set(families) != set(FAMILIES):
                raise FactorRevisionRejected("invalid_factor_state")
            for family in FAMILIES:
                row = families[family]
                mask = int(row["survivor_mask"])
                witnesses = row["witnesses"]
                if mask <= 0 or mask & ~FULL_MASK or len(witnesses) > WITNESS_LIMIT:
                    raise FactorRevisionRejected("invalid_factor_state")
            if len(controller._state["used_release_ids"]) > DEDUPLICATION_LIMIT:
                raise FactorRevisionRejected("invalid_factor_state")
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, FactorRevisionRejected):
                raise
            raise FactorRevisionRejected("invalid_factor_state") from error
        if not controller._within_cap():
            raise FactorRevisionRejected("factor_memory_cap")
        return controller

    @classmethod
    def load(cls, path: Path) -> FactorLocalController:
        """Load one durable controller through the same cold state validator."""

        value = _load_object(path)
        if not value:
            raise FactorRevisionRejected("invalid_factor_state")
        return cls.from_state(value)

    def state_dict(self) -> JsonDict:
        """Return detached state so callers cannot mutate live controller bytes."""

        return deepcopy(self._state)

    def state_bytes(self) -> bytes:
        """Serialize every charged survivor, witness, pending, dedup, and rollback byte."""

        return _canonical_bytes(self._state)

    def state_hash(self) -> str:
        """Bind all charged controller state under one stable identity."""

        return transactional.sha256_bytes(self.state_bytes())

    def _within_cap(self) -> bool:
        return len(self.state_bytes()) <= int(self._state["memory_cap_bytes"])

    def memory_usage(self) -> JsonDict:
        """Report measured serialized bytes and every bounded state category."""

        used = len(self.state_bytes())
        return {
            "serialized_state_bytes": used,
            "cap_bytes": int(self._state["memory_cap_bytes"]),
            "within_cap": used <= int(self._state["memory_cap_bytes"]),
            "survivor_mask_count": len(FAMILIES),
            "witness_count": sum(
                len(self._state["families"][family]["witnesses"]) for family in FAMILIES
            ),
            "pending_release_count": len(self._state["pending_releases"]),
            "deduplication_id_count": len(self._state["used_release_ids"]),
            "rollback_bytes": len(str(self._state.get("rollback") or "").encode("utf-8")),
            "uncharged_parent_pointer_count": 0,
        }

    def family_state(self, family: str) -> JsonDict:
        """Return one detached family for audit without exposing a write handle."""

        if family not in FAMILIES:
            raise FactorRevisionRejected("invalid_family")
        return deepcopy(self._state["families"][family])

    def family_bytes(self, family: str) -> bytes:
        """Encode one family so local revision can prove byte independence."""

        return _canonical_bytes(self.family_state(family))

    @staticmethod
    def _coordinates(event: Mapping[str, Any]) -> tuple[str, int]:
        if set(event) & FORBIDDEN_PUBLIC_FIELDS:
            raise FactorRevisionRejected("private_authority_in_prediction")
        family = event.get("family_id")
        value = event.get("numeric_value")
        if family not in FAMILIES or not isinstance(value, int) or isinstance(value, bool):
            raise FactorRevisionRejected("invalid_public_event")
        return str(family), value

    def predict(self, event: Mapping[str, Any]) -> str:
        """Return a decision only when all surviving predicates agree, without writes."""

        family, value = self._coordinates(event)
        mask = int(self._state["families"][family]["survivor_mask"])
        labels = {
            _predicate_label(family, value, parameter)
            for parameter in PARAMETER_DOMAIN
            if mask & (1 << parameter)
        }
        return next(iter(labels)) if len(labels) == 1 else "abstain"

    def seal_prediction(self, event: Mapping[str, Any], *, release_index: int) -> JsonDict:
        """Charge a prediction receipt before its future label can be accepted."""

        parent = self.state_bytes()
        family, value = self._coordinates(event)
        try:
            event_id = str(event["event_id"])
            source_index = int(event["chronology_index"])
            due = int(release_index)
        except (KeyError, TypeError, ValueError) as error:
            raise FactorRevisionRejected("invalid_public_event") from error
        if not event_id or due < source_index:
            raise FactorRevisionRejected("invalid_release_schedule")
        if any(row["event_id"] == event_id for row in self._state["pending_releases"]):
            raise FactorRevisionRejected("duplicate_pending_release")
        prediction = self.predict(event)
        row = {
            "event_id": event_id,
            "family_id": family,
            "numeric_value": value,
            "source_index": source_index,
            "release_index": due,
            "prediction": prediction,
            "evidence_id": transactional.sha256_json([event_id, family, value, source_index, due]),
        }
        self._state["pending_releases"].append(row)
        self._state["pending_releases"].sort(
            key=lambda item: (item["release_index"], item["event_id"])
        )
        if not self._within_cap():
            self._state = json.loads(parent)
            raise FactorRevisionRejected("factor_memory_cap")
        return deepcopy(row)

    @staticmethod
    def _released_witness(release: Mapping[str, Any]) -> JsonDict:
        try:
            row = {
                "event_id": str(release["event_id"]),
                "family_id": str(release["family_id"]),
                "numeric_value": int(release["numeric_value"]),
                "observed_label": str(release["observed_label"]),
                "source_index": int(release["source_index"]),
                "release_index": int(release["release_index"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise FactorRevisionRejected("invalid_release") from error
        if (
            not row["event_id"]
            or row["family_id"] not in FAMILIES
            or row["observed_label"] not in {"accept", "reject"}
        ):
            raise FactorRevisionRejected("invalid_release")
        return row

    def _install_rollback(self, parent: bytes, receipt_id: str) -> None:
        compressed = zlib.compress(parent, level=9)
        self._state["rollback"] = {
            "receipt_id": receipt_id,
            "parent_hash": transactional.sha256_bytes(parent),
            "parent_bytes_zlib_b64": base64.b64encode(compressed).decode("ascii"),
        }

    def apply_release(self, release: Mapping[str, Any], *, current_index: int) -> JsonDict:
        """Apply one matching due release and rebuild only the contradicted factor."""

        witness = self._released_witness(release)
        pending = next(
            (
                row
                for row in self._state["pending_releases"]
                if row["event_id"] == witness["event_id"]
            ),
            None,
        )
        if pending is None:
            raise FactorRevisionRejected("release_without_sealed_prediction")
        if int(current_index) != witness["release_index"] or pending["release_index"] != int(
            current_index
        ):
            raise FactorRevisionRejected("release_not_due")
        if any(
            pending[key] != witness[key]
            for key in ("event_id", "family_id", "numeric_value", "source_index", "release_index")
        ):
            raise FactorRevisionRejected("release_receipt_mismatch")
        evidence_id = str(pending["evidence_id"])
        if evidence_id in self._state["used_release_ids"]:
            raise FactorRevisionRejected("duplicate_release")

        original = self.state_bytes()
        before_hash = transactional.sha256_bytes(original)
        family = witness["family_id"]
        unaffected = {name: self.family_bytes(name) for name in FAMILIES if name != family}
        started = time.perf_counter_ns()
        contradiction = False
        suffix_length = 0
        predicate_evaluations = 0
        try:
            # The single prior rollback slot expires before the next transaction.
            # This keeps rollback state bounded and prevents recursive parent copies.
            self._state["rollback"] = None
            parent = self.state_bytes()
            self._state["pending_releases"] = [
                row
                for row in self._state["pending_releases"]
                if row["event_id"] != witness["event_id"]
            ]
            self._state["used_release_ids"].append(evidence_id)
            self._state["used_release_ids"] = self._state["used_release_ids"][-DEDUPLICATION_LIMIT:]
            self._state["release_count"] = int(self._state["release_count"]) + 1
            row = self._state["families"][family]
            if self._state["update_enabled"]:
                one_mask = _consistent_mask(family, [witness])
                predicate_evaluations += len(PARAMETER_DOMAIN)
                narrowed = int(row["survivor_mask"]) & one_mask
                contradiction = narrowed == 0
                if contradiction and self._state["mode"] == "global_reset":
                    for name in FAMILIES:
                        target = self._state["families"][name]
                        target["survivor_mask"] = int(self._state["initial_masks"][name])
                        target["witnesses"] = []
                    row = self._state["families"][family]
                    row["survivor_mask"] = one_mask
                    suffix_length = 1
                elif contradiction and self._state["mode"] == "local_reset":
                    row["survivor_mask"] = one_mask
                    row["witnesses"] = []
                    suffix_length = 1
                elif contradiction:
                    candidates = [*row["witnesses"], witness][-WITNESS_LIMIT:]
                    for start in range(len(candidates)):
                        suffix = candidates[start:]
                        rebuilt = _consistent_mask(family, suffix)
                        predicate_evaluations += len(PARAMETER_DOMAIN) * len(suffix)
                        if rebuilt:
                            row["survivor_mask"] = rebuilt
                            row["witnesses"] = suffix if self._state["retain_witnesses"] else []
                            suffix_length = len(suffix)
                            break
                    if suffix_length == 0:
                        raise FactorRevisionRejected("no_consistent_suffix")
                else:
                    row["survivor_mask"] = narrowed
                    if self._state["retain_witnesses"]:
                        row["witnesses"] = [*row["witnesses"], witness][-WITNESS_LIMIT:]
                        suffix_length = len(row["witnesses"])
                row["revision_count"] = int(row["revision_count"]) + 1
            self._state["predicate_evaluation_count"] = (
                int(self._state["predicate_evaluation_count"]) + predicate_evaluations
            )
            receipt_id = transactional.sha256_json(
                [evidence_id, before_hash, family, self._state["release_count"]]
            )
            self._install_rollback(parent, receipt_id)
            if self._state["mode"] != "global_reset" and any(
                self.family_bytes(name) != data for name, data in unaffected.items()
            ):
                raise FactorRevisionRejected("unaffected_factor_changed")
            if not self._within_cap():
                raise FactorRevisionRejected("factor_memory_cap")
        except Exception:
            self._state = json.loads(original)
            raise
        elapsed_ns = time.perf_counter_ns() - started
        after_hash = self.state_hash()
        return {
            "receipt_id": receipt_id,
            "event_id": witness["event_id"],
            "family_id": family,
            "prediction_preceded_release": True,
            "contradiction": contradiction,
            "rebuilt_family": family if contradiction else None,
            "longest_consistent_suffix_length": suffix_length,
            "unaffected_factor_bytes_preserved": self._state["mode"] != "global_reset",
            "state_hash_before": before_hash,
            "state_hash_after": after_hash,
            "predicate_evaluations": predicate_evaluations,
            "update_latency_ns": elapsed_ns,
            "memory_bytes": len(self.state_bytes()),
        }

    def rollback(self, receipt: Mapping[str, Any]) -> JsonDict:
        """Restore the charged parent bytes only for the latest matching receipt."""

        rollback = self._state.get("rollback")
        if not isinstance(rollback, Mapping) or rollback.get("receipt_id") != receipt.get(
            "receipt_id"
        ):
            raise FactorRevisionRejected("stale_rollback")
        try:
            parent = zlib.decompress(
                base64.b64decode(str(rollback["parent_bytes_zlib_b64"]).encode("ascii"))
            )
        except (KeyError, ValueError, zlib.error) as error:
            raise FactorRevisionRejected("invalid_rollback") from error
        if transactional.sha256_bytes(parent) != rollback.get("parent_hash"):
            raise FactorRevisionRejected("invalid_rollback")
        restored = type(self).from_state(json.loads(parent))
        self._state = restored._state
        return {
            "receipt_id": receipt["receipt_id"],
            "restored_hash": self.state_hash(),
            "byte_identical": self.state_bytes() == parent,
            "passed": self.state_bytes() == parent,
        }

    def save(self, path: Path) -> JsonDict:
        """Publish all charged state through the existing atomic memory writer."""

        return _atomic_write(path, self.state_bytes())


class _ControllerMemoryView:
    """Expose the factor file with the state-hash interface used by memory hooks."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path

    def state_hash(self) -> str:
        """Hash the exact durable factor bytes read after an atomic write."""

        return transactional.sha256_bytes(self.state_path.read_bytes())


class FactorPipelineHook:
    """Default-off pipeline hook that persists only the opted-in factor controller."""

    def __init__(
        self,
        state_dir: Path,
        *,
        enabled: bool = False,
        controller: FactorLocalController | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.enabled = bool(enabled)
        self.state_path = self.state_dir / "factor_controller.json"
        self._controller: FactorLocalController | None = None
        if self.enabled:
            if self.state_path.exists():
                self._controller = FactorLocalController.load(self.state_path)
            else:
                self._controller = controller or FactorLocalController.from_masks(
                    dict.fromkeys(FAMILIES, FULL_MASK)
                )
                self._controller.save(self.state_path)
        self.transactional_memory = _ControllerMemoryView(self.state_path)

    def predict(self, event: Mapping[str, Any]) -> str:
        """Return abstain while disabled or read the durable opted-in controller."""

        if not self.enabled or self._controller is None:
            return "abstain"
        return self._controller.predict(event)

    def pre_label(self, event: Mapping[str, Any], *, release_index: int) -> JsonDict:
        """Seal and durably store a prediction before evaluator feedback is visible."""

        if not self.enabled or self._controller is None:
            raise FactorRevisionRejected("factor_hook_disabled")
        receipt = self._controller.seal_prediction(event, release_index=release_index)
        self._controller.save(self.state_path)
        return receipt

    def release(self, release: Mapping[str, Any], *, current_index: int) -> JsonDict:
        """Commit one released witness atomically for later pipeline requests."""

        if not self.enabled or self._controller is None:
            raise FactorRevisionRejected("factor_hook_disabled")
        receipt = self._controller.apply_release(release, current_index=current_index)
        self._controller.save(self.state_path)
        return receipt


def _base_parameter(seed: int, family: str) -> int:
    """Derive one evaluator-private parameter from a frozen independent seed."""

    return (seed * 17 + FAMILIES.index(family) * 11) % len(PARAMETER_DOMAIN)


def _hidden_parameter(
    stratum: str, seed: int, stream_offset: int, family: str, index: int
) -> tuple[str, int, tuple[int, ...]]:
    """Return private drift and recurrence without adding routing data to public rows."""

    base = _base_parameter(seed, family)
    family_index = FAMILIES.index(family)
    if stratum == "stationary_controls":
        return "stationary", base, ()
    if stratum == "isolated_factor_changes":
        first = 272 + seed % 53
        second = 704 + seed % 61
        active = family_index == stream_offset % len(FAMILIES)
        if active and first <= index < second:
            return "changed", (base + 9 + stream_offset) % len(PARAMETER_DOMAIN), (first, second)
        return "recurrent" if index >= second else "base", base, (first, second)
    first = 240 + family_index * 31 + seed % 19
    second = 700 + family_index * 17 + seed % 29
    if first <= index < second:
        return "changed", (base + 7 + family_index * 3) % len(PARAMETER_DOMAIN), (first, second)
    return "recurrent" if index >= second else "base", base, (first, second)


def build_stream_views(kind: str) -> StreamViews:
    """Build frozen public, release, shuffled, and evaluator-only stream views."""

    if kind == "development":
        seeds = DEVELOPMENT_STREAM_SEEDS
        prefix = "development"
        strata = (
            "isolated_factor_changes",
            "overlapping_factor_changes",
            "stationary_controls",
            "isolated_factor_changes",
            "overlapping_factor_changes",
            "stationary_controls",
            "isolated_factor_changes",
            "overlapping_factor_changes",
        )
    elif kind == "evaluation":
        seeds = EVALUATION_STREAM_SEEDS
        prefix = "evaluation"
        strata = tuple(
            name
            for name in (
                "isolated_factor_changes",
                "overlapping_factor_changes",
                "stationary_controls",
            )
            for _ in range(8)
        )
    else:
        raise ValueError("invalid_stream_kind")
    public: list[JsonDict] = []
    authority: list[JsonDict] = []
    releases: list[JsonDict] = []
    labels_by_stream: dict[str, list[tuple[str, str]]] = defaultdict(list)
    selected = set(FUTURE_LABEL_POSITIONS)
    for stream_offset, (seed, stratum) in enumerate(zip(seeds, strata, strict=True)):
        stream_id = f"{prefix}-{stream_offset + 1:02d}"
        for index in range(EVENTS_PER_STREAM):
            family = FAMILIES[(index + stream_offset) % len(FAMILIES)]
            value = (index * 29 + stream_offset * 13 + seed * 7) % len(PARAMETER_DOMAIN)
            regime, parameter, drift_times = _hidden_parameter(
                stratum, seed, stream_offset, family, index
            )
            label = evaluator_exact_label(family, value, parameter)
            event_id = f"exp7310-{stream_id}-e{index:04d}"
            public.append(
                {
                    "event_id": event_id,
                    "stream_id": stream_id,
                    "chronology_index": index,
                    "family_id": family,
                    "numeric_value": value,
                }
            )
            authority.append(
                {
                    "event_id": event_id,
                    "stream_id": stream_id,
                    "chronology_index": index,
                    "stream_seed": seed,
                    "stratum": stratum,
                    "regime_id": regime,
                    "target_parameter": parameter,
                    "drift_times": list(drift_times),
                    "family_id": family,
                    "numeric_value": value,
                    "exact_label": label,
                }
            )
            if index < WARMUP_COUNT or index in selected:
                release_index = index if index < WARMUP_COUNT else index + FEEDBACK_DELAY
                role = "warmup" if index < WARMUP_COUNT else "future_feedback"
                row = {
                    "event_id": event_id,
                    "stream_id": stream_id,
                    "family_id": family,
                    "numeric_value": value,
                    "observed_label": label,
                    "source_index": index,
                    "release_index": release_index,
                    "role": role,
                }
                releases.append(row)
                if role == "future_feedback":
                    labels_by_stream[stream_id].append((event_id, label))
    shuffled: dict[str, str] = {}
    for offset, (stream_id, pairs) in enumerate(sorted(labels_by_stream.items())):
        labels = [label for _, label in pairs]
        random.Random(RANDOM_SEED + 900 + offset).shuffle(labels)
        public_by_id = {
            str(row["event_id"]): row for row in public if row["stream_id"] == stream_id
        }
        for (event_id, original), injected in zip(pairs, labels, strict=True):
            event = public_by_id[event_id]
            candidate = {
                "numeric_value": event["numeric_value"],
                "observed_label": injected,
            }
            shuffled[event_id] = (
                injected if _consistent_mask(str(event["family_id"]), [candidate]) else original
            )
    counts = {name: strata.count(name) for name in dict.fromkeys(strata)}
    return StreamViews(
        public,
        authority,
        releases,
        shuffled,
        {
            "schema": "carnot.exp7310.stream_views.v1",
            "kind": kind,
            "stream_count": len(seeds),
            "events_per_stream": EVENTS_PER_STREAM,
            "warmup_events_per_stream": WARMUP_COUNT,
            "future_feedback_labels_per_stream": FUTURE_LABEL_COUNT,
            "feedback_delay": FEEDBACK_DELAY,
            "future_label_positions": list(FUTURE_LABEL_POSITIONS),
            "stream_seeds": list(seeds),
            "stream_seeds_sha256": transactional.sha256_json(list(seeds)),
            "strata": counts,
            "public_fields": [
                "event_id",
                "stream_id",
                "chronology_index",
                "family_id",
                "numeric_value",
            ],
            "evaluator_owned_transitions": True,
            "frozen_before_controller_execution": True,
        },
    )


def stream_conformance_errors(views: StreamViews, kind: str) -> list[str]:
    """Check fixed counts, schedules, strata, identity, and authority separation."""

    expected_streams = (
        DEVELOPMENT_STREAM_COUNT if kind == "development" else EVALUATION_STREAM_COUNT
    )
    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(kind not in {"development", "evaluation"}, "stream_kind")
    add(len(views.public) != expected_streams * EVENTS_PER_STREAM, "public_count")
    add(len(views.authority) != expected_streams * EVENTS_PER_STREAM, "authority_count")
    add(
        len(views.releases) != expected_streams * (WARMUP_COUNT + FUTURE_LABEL_COUNT),
        "release_count",
    )
    add(any(set(row) & FORBIDDEN_PUBLIC_FIELDS for row in views.public), "public_authority_leakage")
    add(
        [row.get("event_id") for row in views.public]
        != [row.get("event_id") for row in views.authority],
        "event_identity",
    )
    by_stream: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in views.releases:
        by_stream[str(row["stream_id"])].append(row)
    add(len(by_stream) != expected_streams, "stream_count")
    for rows in by_stream.values():
        warmup = [row for row in rows if row["role"] == "warmup"]
        future = [row for row in rows if row["role"] == "future_feedback"]
        add([row["source_index"] for row in warmup] != list(range(WARMUP_COUNT)), "warmup_schedule")
        add(
            [row["source_index"] for row in future] != list(FUTURE_LABEL_POSITIONS)
            or any(row["release_index"] != row["source_index"] + FEEDBACK_DELAY for row in future),
            "future_schedule",
        )
    if kind == "evaluation":
        add(
            views.manifest.get("strata")
            != {
                "isolated_factor_changes": 8,
                "overlapping_factor_changes": 8,
                "stationary_controls": 8,
            },
            "evaluation_strata",
        )
    return errors


def _warmup_masks(releases: Sequence[Mapping[str, Any]], stream_id: str) -> dict[str, int]:
    """Fit one factor mask from exactly the released 128-event warmup."""

    warmup = [row for row in releases if row["stream_id"] == stream_id and row["role"] == "warmup"]
    if len(warmup) != WARMUP_COUNT:
        raise ValueError(f"incomplete_warmup:{stream_id}")
    result = {}
    for family in FAMILIES:
        result[family] = _consistent_mask(
            family, [row for row in warmup if row["family_id"] == family]
        )
        if result[family] == 0:
            raise ValueError(f"inconsistent_warmup:{stream_id}:{family}")
    return result


def _arm_controller(arm: str, masks: Mapping[str, int]) -> FactorLocalController:
    """Construct one of the five frozen bounded controller arms."""

    if arm == "global_reset_on_contradiction":
        return FactorLocalController.from_masks(masks, mode="global_reset")
    if arm == "local_reset_without_retained_witnesses":
        return FactorLocalController.from_masks(masks, mode="local_reset", retain_witnesses=False)
    if arm == "frozen_warmup":
        return FactorLocalController.from_masks(masks, update_enabled=False)
    return FactorLocalController.from_masks(masks)


def _prediction_counts(prediction: str, label: str) -> tuple[int, int, int, int]:
    """Score full-denominator error, false acceptance, abstention, and coverage."""

    return (
        int(prediction != label),
        int(prediction == "accept" and label == "reject"),
        int(prediction == "abstain"),
        int(prediction != "abstain"),
    )


def run_development_panel(views: StreamViews, *, progress: bool = False) -> list[JsonDict]:
    """Replay all five arms once while labels follow one shared delayed schedule."""

    stream_ids = sorted({str(row["stream_id"]) for row in views.public})
    public_by_stream: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    authority_by_id = {str(row["event_id"]): row for row in views.authority}
    releases_by_stream_due: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in views.public:
        public_by_stream[str(row["stream_id"])].append(row)
    for row in views.releases:
        if row["role"] == "future_feedback":
            releases_by_stream_due[(str(row["stream_id"]), int(row["release_index"]))].append(row)
    rows: list[JsonDict] = []
    panel_started = time.monotonic()
    for stream_number, stream_id in enumerate(stream_ids, start=1):
        masks = _warmup_masks(views.releases, stream_id)
        controllers = {arm: _arm_controller(arm, masks) for arm in ARMS}
        frozen = controllers["frozen_warmup"]
        metrics = {
            arm: {
                "errors": 0,
                "non_feedback_errors": 0,
                "non_feedback_count": 0,
                "false_accepts": 0,
                "abstentions": 0,
                "covered": 0,
                "changed": 0,
                "updates": 0,
                "operations": 0,
                "latencies": [],
                "max_bytes": len(controller.state_bytes()),
            }
            for arm, controller in controllers.items()
        }
        for event in public_by_stream[stream_id][WARMUP_COUNT:]:
            index = int(event["chronology_index"])
            truth = str(authority_by_id[str(event["event_id"])]["exact_label"])
            frozen_prediction = frozen.predict(event)
            selected = index in set(FUTURE_LABEL_POSITIONS)
            for arm, controller in controllers.items():
                prediction = controller.predict(event)
                error, false_accept, abstention, covered = _prediction_counts(prediction, truth)
                metric = metrics[arm]
                metric["errors"] += error
                metric["false_accepts"] += false_accept
                metric["abstentions"] += abstention
                metric["covered"] += covered
                metric["changed"] += int(arm != "frozen_warmup" and prediction != frozen_prediction)
                if not selected:
                    metric["non_feedback_count"] += 1
                    metric["non_feedback_errors"] += error
                if selected:
                    controller.seal_prediction(event, release_index=index + FEEDBACK_DELAY)
                    metric["max_bytes"] = max(metric["max_bytes"], len(controller.state_bytes()))
            for release in releases_by_stream_due.get((stream_id, index), []):
                for arm, controller in controllers.items():
                    observed = dict(release)
                    if arm == "label_shuffled_factor_local_revision":
                        observed["observed_label"] = views.shuffled_labels[str(release["event_id"])]
                    receipt = controller.apply_release(observed, current_index=index)
                    metric = metrics[arm]
                    metric["updates"] += 1
                    metric["operations"] += int(receipt["predicate_evaluations"])
                    metric["latencies"].append(int(receipt["update_latency_ns"]))
                    metric["max_bytes"] = max(metric["max_bytes"], int(receipt["memory_bytes"]))
        stratum = str(authority_by_id[str(public_by_stream[stream_id][0]["event_id"])]["stratum"])
        for arm in ARMS:
            metric = metrics[arm]
            future_count = EVENTS_PER_STREAM - WARMUP_COUNT
            latencies = sorted(metric["latencies"])
            rows.append(
                {
                    "stream_id": stream_id,
                    "stream_seed": views.manifest["stream_seeds"][stream_number - 1],
                    "stratum": stratum,
                    "arm": arm,
                    "future_prediction_count": future_count,
                    "feedback_selected_prediction_count": FUTURE_LABEL_COUNT,
                    "non_feedback_future_prediction_count": metric["non_feedback_count"],
                    "future_error_count": metric["errors"],
                    "future_error_rate": metric["errors"] / future_count,
                    "non_feedback_error_rate": metric["non_feedback_errors"]
                    / metric["non_feedback_count"],
                    "false_accept_rate": metric["false_accepts"] / future_count,
                    "abstention_rate": metric["abstentions"] / future_count,
                    "coverage_rate": metric["covered"] / future_count,
                    "legitimate_later_changed_prediction_count": metric["changed"],
                    "released_update_count": metric["updates"],
                    "predicate_evaluation_count": metric["operations"],
                    "maximum_memory_bytes": metric["max_bytes"],
                    "mean_update_latency_ns": (
                        sum(latencies) / len(latencies) if latencies else 0.0
                    ),
                    "p95_update_latency_ns": (
                        latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
                        if latencies
                        else 0
                    ),
                    "byte_limit_violations": int(metric["max_bytes"] > MEMORY_CAP_BYTES),
                    "time_limit_violations": 0,
                    "censored": False,
                }
            )
        if progress:
            _progress(
                4,
                "progress",
                f"development streams={stream_number}/{len(stream_ids)} elapsed={time.monotonic() - panel_started:.3f}s",
            )
    return rows


def _control_release(event_id: str, label: str, *, source: int = 0, due: int = 4) -> JsonDict:
    """Build one released lower-bound witness for isolated controller controls."""

    return {
        "event_id": event_id,
        "family_id": "lower_bound",
        "numeric_value": 10,
        "observed_label": label,
        "source_index": source,
        "release_index": due,
    }


def _control_event(event_id: str, *, source: int = 0) -> JsonDict:
    """Build the matching public lower-bound event without evaluator fields."""

    return {
        "event_id": event_id,
        "family_id": "lower_bound",
        "numeric_value": 10,
        "chronology_index": source,
    }


def run_controller_controls(root: Path) -> list[JsonDict]:
    """Exercise independence, contradiction, poison, bounds, durability, and E2E."""

    root.mkdir(parents=True, exist_ok=True)
    masks = dict.fromkeys(FAMILIES, FULL_MASK)
    masks["lower_bound"] = 1 << 5
    masks["upper_bound"] = 1 << 20
    outcomes: dict[str, bool] = {}

    local = FactorLocalController.from_masks(masks)
    upper = local.family_bytes("upper_bound")
    local.seal_prediction(_control_event("factor"), release_index=4)
    local.apply_release(_control_release("factor", "reject"), current_index=4)
    outcomes["factor_independence"] = local.family_bytes("upper_bound") == upper
    outcomes["contradictory_delayed_label"] = (
        local.family_state("lower_bound")["revision_count"] == 1
    )
    outcomes["ambiguous_suffix"] = (
        local.predict({**_control_event("ambiguous"), "numeric_value": 12}) == "abstain"
    )

    poison = FactorLocalController.from_masks(masks)
    poison.seal_prediction(_control_event("poison"), release_index=4)
    parent = poison.state_bytes()
    try:
        poison.apply_release(_control_release("poison", "invalid"), current_index=4)
    except FactorRevisionRejected:
        pass
    outcomes["poison_rejection"] = poison.state_bytes() == parent

    base = FactorLocalController.from_masks(dict.fromkeys(FAMILIES, FULL_MASK))
    limited = FactorLocalController.from_masks(
        dict.fromkeys(FAMILIES, FULL_MASK), memory_cap_bytes=len(base.state_bytes()) + 8
    )
    parent = limited.state_bytes()
    try:
        limited.seal_prediction(_control_event("cap"), release_index=4)
    except FactorRevisionRejected:
        pass
    outcomes["byte_cap_exhaustion"] = limited.state_bytes() == parent

    reversible = FactorLocalController.from_masks(masks)
    reversible.seal_prediction(_control_event("rollback"), release_index=4)
    parent = reversible.state_bytes()
    receipt = reversible.apply_release(_control_release("rollback", "reject"), current_index=4)
    outcomes["rollback"] = (
        reversible.rollback(receipt)["byte_identical"] and reversible.state_bytes() == parent
    )
    restart_path = root / "restart.json"
    reversible.save(restart_path)
    outcomes["cold_restart"] = (
        FactorLocalController.load(restart_path).state_bytes() == reversible.state_bytes()
    )
    parent = reversible.state_bytes()
    reversible.predict(_control_event("read-only"))
    outcomes["read_only_prediction"] = reversible.state_bytes() == parent

    hook = FactorPipelineHook(
        root / "hook", enabled=True, controller=FactorLocalController.from_masks(masks)
    )
    hook.pre_label(_control_event("e2e"), release_index=4)
    before = hook.predict(_control_event("before"))
    hook.release(_control_release("e2e", "reject"), current_index=4)
    after = FactorPipelineHook(root / "hook", enabled=True).predict(_control_event("after"))
    outcomes["opt_in_pipeline"] = before == "accept" and after == "reject"
    return [
        {"control": name, "passed": outcomes.get(name) is True, "observed": outcomes.get(name)}
        for name in CONTROL_NAMES
    ]


def _task_identity(text: str) -> JsonDict:
    """Read only the active Exp7310 identity from the executable roadmap."""

    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    if not isinstance(value, dict) or value.get("milestone") != MILESTONE:
        return {}
    for task in value.get("tasks", []):
        if isinstance(task, dict) and task.get("id") == "exp7310-factor-prototype":
            return {
                "id": task.get("id"),
                "milestone": task.get("milestone"),
                "deliverable": task.get("deliverable"),
            }
    return {}


def _gate(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep every expected and observed precondition in one auditable row."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve every check and expose the first exact failed observation."""

    failed = [dict(row) for row in checks if row.get("passed") is not True]
    first = failed[0] if failed else None
    return {
        "passed": not failed,
        "checks": [dict(row) for row in checks],
        "failed_checks": failed,
        "first_failure": first,
        "upstream": first.get("upstream") if first else None,
        "check": first.get("check") if first else None,
        "field": first.get("field") if first else None,
        "observed_value": first.get("observed_value") if first else None,
        "expected_value": first.get("expected_value") if first else None,
    }


def collect_preconditions(
    repo_root: Path, paths: ExperimentPaths
) -> tuple[list[JsonDict], dict[str, str | None]]:
    """Authenticate historical evidence, task identity, sources, exclusions, and outputs."""

    historical = _load_object(paths.historical_artifact)
    spec_text = (repo_root / SPEC_PATH).read_text(encoding="utf-8")
    roadmap_path = repo_root / "research-roadmap.yaml"
    roadmap_text = roadmap_path.read_text(encoding="utf-8")
    exclusion_text = (repo_root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    identity = _task_identity(roadmap_text)
    expected_identity = {
        "id": "exp7310-factor-prototype",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT),
    }
    scenario_count = len(set(SCENARIO_PATTERN.findall(spec_text)))
    checks = [
        _gate(
            "driving_capability_spec",
            str(repo_root / SPEC_PATH),
            "REQ-CL-7310",
            True,
            "REQ-CL-7310" in spec_text,
            "REQ-CL-7310" in spec_text,
            "Implementation starts only after its requirement exists.",
        ),
        _gate(
            "scenario_contract",
            str(repo_root / SPEC_PATH),
            "SCENARIO-CL-7310-*",
            7,
            scenario_count,
            scenario_count == 7,
            "Every required behavior has a named scenario.",
        ),
        _gate(
            "v642_task_identity",
            str(roadmap_path),
            "id,milestone,deliverable",
            expected_identity,
            identity,
            identity == expected_identity,
            "Run only the active task and declared output.",
        ),
        _gate(
            "historical_available",
            str(paths.historical_artifact),
            "schema",
            "carnot.exp7297.v641_mixture_audit.v1",
            historical.get("schema"),
            historical.get("schema") == "carnot.exp7297.v641_mixture_audit.v1",
            "The retired mechanism record must be authentic evidence.",
        ),
        _gate(
            "historical_terminal",
            str(paths.historical_artifact),
            "status",
            "complete",
            historical.get("status"),
            historical.get("status") == "complete",
            "An incomplete external record cannot motivate a successor.",
        ),
        _gate(
            "historical_audit_complete",
            str(paths.historical_artifact),
            "mixture_audit_complete_score",
            1,
            historical.get("mixture_audit_complete_score"),
            historical.get("mixture_audit_complete_score") == 1,
            "The prior null must come from a complete audit.",
        ),
        _gate(
            "historical_null",
            str(paths.historical_artifact),
            "verdict_class",
            "null",
            historical.get("verdict_class"),
            historical.get("verdict_class") == "null",
            "The new representation addresses the recorded fixed-share null.",
        ),
        _gate(
            "historical_not_quarantined",
            str(paths.historical_artifact),
            "flagged_adversarial",
            False,
            bool(historical.get("flagged_adversarial")),
            not bool(historical.get("flagged_adversarial")),
            "Quarantined evidence cannot authorize a successor.",
        ),
        _gate(
            "historical_not_disqualified",
            str(paths.historical_artifact),
            "verdict_class",
            "not disqualified",
            historical.get("verdict_class"),
            historical.get("verdict_class") != "disqualified",
            "Disqualified evidence cannot be treated as authenticated history.",
        ),
        _gate(
            "exp7310_not_excluded",
            str(repo_root / "ops/exclusion_manifest.yaml"),
            "experiment_id",
            False,
            bool(re.search(r"experiment_id:\s*(?:exp)?7310\b", exclusion_text)),
            not bool(re.search(r"experiment_id:\s*(?:exp)?7310\b", exclusion_text)),
            "A retired task identifier must fail before work.",
        ),
        _gate(
            "resource_ownership",
            str(paths.artifact),
            "task_owned_outputs_writable",
            True,
            _path_writable(paths.artifact) and _path_writable(paths.raw_dir / "probe"),
            _path_writable(paths.artifact) and _path_writable(paths.raw_dir / "probe"),
            "Only task-owned writable outputs may receive evidence.",
        ),
    ]
    hashes: dict[str, str | None] = {
        str(repo_root / path): _sha256_path(repo_root / path) for path in SOURCE_PATHS
    }
    hashes[str(paths.historical_artifact)] = _sha256_path(paths.historical_artifact)
    sources_available = all(value is not None for value in hashes.values())
    checks.append(
        _gate(
            "source_bytes_available",
            "declared source paths",
            "sha256",
            True,
            sources_available,
            sources_available,
            "All source identities must be hashable before measurement.",
        )
    )
    return checks, hashes


def _seal_streams(
    paths: ExperimentPaths, development: StreamViews, evaluation: StreamViews
) -> JsonDict:
    """Seal all authority-separated views and return one hash-bound manifest."""

    receipts = {
        "development_public": _write_immutable(
            paths.development_public, _jsonl_bytes(development.public)
        ),
        "development_authority": _write_immutable(
            paths.development_authority, _jsonl_bytes(development.authority)
        ),
        "development_releases": _write_immutable(
            paths.development_releases, _jsonl_bytes(development.releases)
        ),
        "evaluation_public": _write_immutable(
            paths.evaluation_public, _jsonl_bytes(evaluation.public)
        ),
        "evaluation_authority": _write_immutable(
            paths.evaluation_authority, _jsonl_bytes(evaluation.authority)
        ),
        "evaluation_releases": _write_immutable(
            paths.evaluation_releases, _jsonl_bytes(evaluation.releases)
        ),
    }
    manifest = {
        "schema": "carnot.exp7310.stream_manifest.v1",
        "frozen": True,
        "development": development.manifest,
        "evaluation": evaluation.manifest,
        "feedback_schedule": {
            "warmup_events": WARMUP_COUNT,
            "future_feedback_labels": FUTURE_LABEL_COUNT,
            "future_label_positions": list(FUTURE_LABEL_POSITIONS),
            "delay": FEEDBACK_DELAY,
            "same_positions_all_arms": True,
        },
        "authority_separated": True,
        "shuffled_label_hashes": {
            "development": transactional.sha256_json(development.shuffled_labels),
            "evaluation": transactional.sha256_json(evaluation.shuffled_labels),
        },
        "receipts": receipts,
    }
    _write_immutable(paths.stream_manifest, manifest)
    return manifest


def _learning_contract() -> JsonDict:
    """Freeze prospective gates without using prototype outcomes for selection."""

    return {
        "frozen_before_evaluation": True,
        "independent_unit": "whole_evaluation_stream",
        "interval": "paired_bootstrap_ci95",
        "bootstrap_draws": 10_000,
        "comparisons": {
            "future_error_vs_global_reset": {"upper_delta": "<0"},
            "future_error_vs_local_reset": {"upper_delta": "<0"},
            "non_feedback_error_vs_global_reset": {"upper_delta": "<0"},
            "non_feedback_error_vs_local_reset": {"upper_delta": "<0"},
            "recurrence_error_vs_frozen": {"upper_delta": "<=0.02"},
            "false_accept_vs_controls": {"upper_delta": "<=0"},
            "coverage_vs_controls": {"lower_delta": ">=-0.02"},
            "legitimate_later_changed_predictions": {"minimum": 24},
            "time_and_byte_violations": {"maximum": 0},
        },
        "prototype_readiness_depends_on_efficacy": False,
        "retired_fixed_share_retrained": False,
    }


def _hardware_path(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report measured CPU mechanics and keep device acceleration unverified."""

    operations = sum(int(row["predicate_evaluation_count"]) for row in rows)
    latencies = [
        float(row["mean_update_latency_ns"]) for row in rows if row["released_update_count"]
    ]
    return {
        "current_implementation": "python_integer_bitsets_and_bounded_counters_on_cpu",
        "measured_predicate_evaluations": operations,
        "maximum_charged_bytes": max(int(row["maximum_memory_bytes"]) for row in rows),
        "measured_mean_cpu_update_latency_ns": sum(latencies) / len(latencies),
        "future_native_layout": {
            "factor_count": len(FAMILIES),
            "survivor_words": "four contiguous uint64 words",
            "witness_table": f"{len(FAMILIES)}x{WITNESS_LIMIT} fixed-width records",
            "targets": ["Rust/PyO3", "FPGA lookup"],
        },
        "future_acceleration_target": "100x",
        "future_acceleration_target_verified": False,
        "hardware_execution_claimed": False,
        "hardware_training_claimed": False,
    }


def _sample_budget(completed: bool) -> JsonDict:
    """State the fixed prototype and sealed future sample budgets."""

    development_units = DEVELOPMENT_STREAM_COUNT * len(ARMS)
    evaluation_units = EVALUATION_STREAM_COUNT * len(ARMS)
    return {
        "development_streams_planned": DEVELOPMENT_STREAM_COUNT,
        "development_streams_attempted": DEVELOPMENT_STREAM_COUNT if completed else 0,
        "development_stream_arm_units_planned": development_units,
        "development_stream_arm_units_complete": development_units if completed else 0,
        "evaluation_streams_sealed": EVALUATION_STREAM_COUNT if completed else 0,
        "evaluation_stream_arm_units_reserved": evaluation_units,
        "evaluation_stream_arm_units_measured": 0,
        "censored_units": 0,
        "events_per_stream": EVENTS_PER_STREAM,
        "warmup_events_per_stream": WARMUP_COUNT,
        "future_feedback_labels_per_stream": FUTURE_LABEL_COUNT,
        "stopping_rule": "fixed counts; prototype stops after eight development streams and seals evaluation without reading efficacy",
    }


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Create every required field before terminal classification is known."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "phase_durations_s": {},
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in checks],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "random_seed": {
            "master": RANDOM_SEED,
            "development": list(DEVELOPMENT_STREAM_SEEDS),
            "evaluation": list(EVALUATION_STREAM_SEEDS),
            "sealed_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": _sample_budget(False),
        "acceptance_gate_results": {},
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_unclassified",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "factor_fixture_ready_score": 0,
        "continuous_self_learning_task": True,
        "factor_state_schema": {
            "schema": STATE_SCHEMA,
            "factor_ownership": {
                family: {"survivor_mask": "uint64", "released_witness_capacity": WITNESS_LIMIT}
                for family in FAMILIES
            },
            "charged_categories": [
                "survivor_masks",
                "released_witness_bytes",
                "pending_releases",
                "deduplication_ids",
                "rollback_state",
            ],
            "memory_cap_bytes": MEMORY_CAP_BYTES,
            "uncharged_parent_pointer_count": 0,
        },
        "stream_manifest": {},
        "learning_acceptance_contract": _learning_contract(),
        "hardware_path": {
            "current_implementation": "cpu_not_run",
            "future_acceleration_target_verified": False,
            "hardware_execution_claimed": False,
            "hardware_training_claimed": False,
        },
        "no_model_weight_mutation": True,
        "production_default_changed": False,
        "repository_health": {},
        "raw_evidence_receipts": {},
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable identity, sources, configuration, rows, gates, and raw evidence."""

    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "MODEL_SPECS",
        "model_invoked",
        "invocation_counts",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "random_seed",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "gate_check_summary",
        "verifier_is_oracle",
        "honest_verdict",
        "verdict_class",
        "factor_fixture_ready_score",
        "factor_state_schema",
        "stream_manifest",
        "learning_acceptance_contract",
        "hardware_path",
        "raw_evidence_receipts",
    )
    return transactional.sha256_json({key: artifact.get(key) for key in keys})


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]], source_hashes: Mapping[str, str | None]
) -> JsonDict:
    """Build row-free terminal evidence for an unchanged external failure."""

    now = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks, source_hashes, started_at=now, completed_at=now, duration_s=0.0
    )
    failure = artifact["gate_check_summary"]["first_failure"]
    if failure is None:
        raise ValueError("blocked_artifact_without_failure")
    artifact["honest_verdict"] = (
        f"blocked_{failure['check']}: upstream={failure['upstream']}; field={failure['field']}; "
        f"observed={failure['observed_value']!r}; expected={failure['expected_value']!r}"
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _gate_result(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Give every mechanical readiness check a complete auditable shape."""

    return {
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def _repository_health(historical: Mapping[str, Any]) -> JsonDict:
    """Preserve the prior repository-wide observation without rerunning its failures."""

    prior = historical.get("global_suite_observation")
    return {
        "classification": "historical_observation_not_current_required_check",
        "source_experiment_id": 7297,
        "observation": deepcopy(prior) if isinstance(prior, Mapping) else prior,
        "waives_affected_test_failure": False,
    }


def build_and_seal(repo_root: Path, paths: ExperimentPaths, *, progress: bool = False) -> JsonDict:
    """Authenticate, seal streams, replay development, run controls, and classify."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    phase_spans: dict[str, float] = {}
    phase = time.monotonic()
    if progress:
        _progress(1, "start", "authenticate historical evidence and outputs")
    checks, hashes = collect_preconditions(repo_root, paths)
    phase_spans["preconditions"] = time.monotonic() - phase
    if progress:
        _progress(1, "end", f"checks={len(checks)} passed={gate_summary(checks)['passed']}")
    if not gate_summary(checks)["passed"]:
        artifact = build_blocked_artifact(checks, hashes)
        artifact["started_at_utc"] = started_at
        artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
        artifact["duration_s"] = time.monotonic() - started
        artifact["phase_durations_s"] = phase_spans
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    phase = time.monotonic()
    if progress:
        _progress(2, "start", "build independent development and evaluation views")
    development = build_stream_views("development")
    evaluation = build_stream_views("evaluation")
    stream_errors = [
        *stream_conformance_errors(development, "development"),
        *stream_conformance_errors(evaluation, "evaluation"),
    ]
    manifest = _seal_streams(paths, development, evaluation)
    phase_spans["stream_freeze"] = time.monotonic() - phase
    if progress:
        _progress(2, "end", f"sealed streams=32 conformance_errors={len(stream_errors)}")

    phase = time.monotonic()
    if progress:
        _progress(3, "start", "run independent controller attack controls")
    with tempfile.TemporaryDirectory(prefix="carnot-exp7310-controls-", dir="/tmp") as temporary:
        controls = run_controller_controls(Path(temporary))
    _write_immutable(paths.control_rows, controls)
    phase_spans["controller_controls"] = time.monotonic() - phase
    if progress:
        _progress(
            3, "end", f"controls={len(controls)} passed={sum(row['passed'] for row in controls)}"
        )

    phase = time.monotonic()
    if progress:
        _progress(4, "start", "benchmark eight development streams across five arms")
    rows = run_development_panel(development, progress=progress)
    _write_immutable(paths.panel_rows, _jsonl_bytes(rows))
    phase_spans["development_panel"] = time.monotonic() - phase
    if progress:
        _progress(4, "end", f"stream-arm rows={len(rows)}")

    controls_pass = len(controls) == len(CONTROL_NAMES) and all(row["passed"] for row in controls)
    maximum_bytes = max(int(row["maximum_memory_bytes"]) for row in rows)
    changed = sum(
        int(row["legitimate_later_changed_prediction_count"])
        for row in rows
        if row["arm"] == "factor_local_retained_witnesses"
    )
    gates = {
        "authenticated_inputs": _gate_result(
            True, True, True, "Only exact non-quarantined historical evidence can start."
        ),
        "stream_conformance": _gate_result(
            [],
            stream_errors,
            not stream_errors,
            "Frozen counts and authority separation prevent outcome-driven changes.",
        ),
        "five_bounded_arms": _gate_result(
            list(ARMS), list(ARMS), True, "Controls remain fixed before prospective evaluation."
        ),
        "controller_controls": _gate_result(
            len(CONTROL_NAMES),
            sum(row["passed"] for row in controls),
            controls_pass,
            "Required attacks and durable paths must all execute.",
        ),
        "bounded_state": _gate_result(
            f"<={MEMORY_CAP_BYTES}",
            maximum_bytes,
            maximum_bytes <= MEMORY_CAP_BYTES,
            "Every durable state category shares the existing cap.",
        ),
        "legitimate_later_changes": _gate_result(
            ">=24",
            changed,
            changed >= 24,
            "Released feedback must cause enough later predictions to test causality.",
        ),
        "time_and_byte_violations": _gate_result(
            0,
            sum(
                int(row["time_limit_violations"]) + int(row["byte_limit_violations"])
                for row in rows
            ),
            all(
                int(row["time_limit_violations"]) == 0 and int(row["byte_limit_violations"]) == 0
                for row in rows
            ),
            "A bounded fixture cannot hide resource violations.",
        ),
        "evaluation_panel_sealed_not_scored": _gate_result(
            24,
            evaluation.manifest["stream_count"],
            evaluation.manifest["stream_count"] == 24,
            "Readiness seals the prospective panel without claiming efficacy.",
        ),
    }
    ready = int(all(row["passed"] for row in gates.values()))
    historical = _load_object(paths.historical_artifact)
    artifact = _base_artifact(
        checks,
        hashes,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
    )
    artifact.update(
        {
            "status": "complete",
            "phase_durations_s": phase_spans,
            "rows": rows,
            "sample_size_budget": _sample_budget(True),
            "acceptance_gate_results": gates,
            "honest_verdict": (
                "complete_circular_positive: bounded factor-local revision fixture is ready; "
                "prospective efficacy remains unmeasured"
                if ready
                else "complete_null: factor-local revision fixture failed one or more mechanical readiness gates"
            ),
            "verdict_class": "circular_positive" if ready else "null",
            "factor_fixture_ready_score": ready,
            "stream_manifest": manifest,
            "hardware_path": _hardware_path(rows),
            "repository_health": _repository_health(historical),
            "raw_evidence_receipts": {
                "stream_manifest": {
                    "path": str(paths.stream_manifest),
                    "sha256": _sha256_path(paths.stream_manifest),
                },
                "panel_rows": {
                    "path": str(paths.panel_rows),
                    "sha256": _sha256_path(paths.panel_rows),
                },
                "control_rows": {
                    "path": str(paths.control_rows),
                    "sha256": _sha256_path(paths.control_rows),
                },
            },
            "validation_receipts": [
                {
                    "command": "internal:independent controller controls and development reduction",
                    "scope": "REQ-CL-7310 mechanics",
                    "exit_code": 0 if ready else 1,
                    "duration_s": phase_spans["controller_controls"]
                    + phase_spans["development_panel"],
                    "log_sha256": transactional.sha256_json([controls, rows]),
                }
            ],
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact, repo_root=repo_root, check_files=True)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    return artifact


def _receipt_error(receipt: Mapping[str, Any]) -> bool:
    """Reject validation receipts without exact command, scope, status, timing, and log hash."""

    return not (
        isinstance(receipt.get("command"), str)
        and bool(receipt.get("command"))
        and isinstance(receipt.get("scope"), str)
        and bool(receipt.get("scope"))
        and isinstance(receipt.get("exit_code"), int)
        and isinstance(receipt.get("duration_s"), (int, float))
        and re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt.get("log_sha256", ""))) is not None
    )


def validate_artifact(
    artifact: Mapping[str, Any], *, repo_root: Path = REPO_ROOT, check_files: bool = False
) -> list[str]:
    """Cold-check identity, bounds, rows, controls, raw hashes, and terminal class."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(any(field not in artifact for field in REQUIRED_ARTIFACT_FIELDS), "required_fields")
    add(
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE,
        "identity",
    )
    add(
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != INVOCATION_COUNTS
        or artifact.get("current_model_load_count") != 0
        or artifact.get("current_generation_count") != 0,
        "model_invocation",
    )
    add(
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS
        or artifact.get("execution_venue") != EXECUTION_VENUE,
        "substrate",
    )
    add(artifact.get("verifier_is_oracle") is not True, "oracle_declaration")
    add(artifact.get("verdict_class") == "positive", "oracle_positive_forbidden")
    add(artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact), "checksum")
    add(
        not isinstance(artifact.get("field_principles"), Mapping)
        or any(
            field not in artifact.get("field_principles", {}) for field in REQUIRED_ARTIFACT_FIELDS
        ),
        "field_principles",
    )
    receipts = artifact.get("validation_receipts", [])
    add(
        not isinstance(receipts, list)
        or any(not isinstance(row, Mapping) or _receipt_error(row) for row in receipts),
        "validation_receipts",
    )
    if artifact.get("status") == "blocked":
        add(
            artifact.get("rows") != []
            or artifact.get("factor_fixture_ready_score") != 0
            or artifact.get("verdict_class") != "blocked"
            or not str(artifact.get("honest_verdict", "")).startswith("blocked_")
            or artifact.get("gate_check_summary", {}).get("passed") is not False
            or artifact.get("gate_check_summary", {}).get("first_failure") is None,
            "blocked_contract",
        )
        return errors
    add(artifact.get("status") != "complete", "status")
    if artifact.get("status") != "complete":
        return errors
    rows = artifact.get("rows", [])
    expected_units = {
        (f"development-{index:02d}", arm)
        for index in range(1, DEVELOPMENT_STREAM_COUNT + 1)
        for arm in ARMS
    }
    add(
        not isinstance(rows, list)
        or {(row.get("stream_id"), row.get("arm")) for row in rows} != expected_units
        or any(
            row.get("censored") is not False
            or row.get("future_prediction_count") != EVENTS_PER_STREAM - WARMUP_COUNT
            or row.get("feedback_selected_prediction_count") != FUTURE_LABEL_COUNT
            or row.get("non_feedback_future_prediction_count")
            != EVENTS_PER_STREAM - WARMUP_COUNT - FUTURE_LABEL_COUNT
            for row in rows
        ),
        "rows",
    )
    gates = artifact.get("acceptance_gate_results", {})
    ready = int(bool(gates) and all(row.get("passed") is True for row in gates.values()))
    add(artifact.get("factor_fixture_ready_score") != ready, "fixture_ready_score")
    add(
        ready == 1
        and (
            artifact.get("verdict_class") != "circular_positive"
            or not str(artifact.get("honest_verdict", "")).startswith("complete_circular_positive:")
        ),
        "ready_verdict",
    )
    add(
        artifact.get("stream_manifest", {}).get("evaluation", {}).get("strata")
        != {
            "isolated_factor_changes": 8,
            "overlapping_factor_changes": 8,
            "stationary_controls": 8,
        },
        "stream_manifest",
    )
    add(
        artifact.get("factor_state_schema", {}).get("memory_cap_bytes") != MEMORY_CAP_BYTES
        or artifact.get("factor_state_schema", {}).get("uncharged_parent_pointer_count") != 0,
        "factor_state_schema",
    )
    add(
        artifact.get("hardware_path", {}).get("future_acceleration_target_verified") is not False
        or artifact.get("hardware_path", {}).get("hardware_execution_claimed") is not False
        or artifact.get("hardware_path", {}).get("hardware_training_claimed") is not False,
        "hardware_claim",
    )
    if check_files:
        add(
            any(
                expected is None or _sha256_path(Path(path)) != expected
                for path, expected in artifact.get("source_artifact_hashes", {}).items()
            ),
            "source_artifact_hashes",
        )
        add(
            any(
                receipt.get("sha256") is None
                or _sha256_path(Path(str(receipt.get("path")))) != receipt.get("sha256")
                for receipt in artifact.get("raw_evidence_receipts", {}).values()
            ),
            "raw_evidence_hashes",
        )
    return errors


def attach_validation_receipts(
    artifact: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Attach exact command evidence and refresh the stable content checksum."""

    if any(not isinstance(row, Mapping) or _receipt_error(row) for row in receipts):
        raise ValueError("validation_receipt_schema")
    updated = deepcopy(dict(artifact))
    updated["validation_receipts"] = [dict(row) for row in receipts]
    updated["reproducibility_checksum"] = reproducibility_checksum(updated)
    return updated


def write_artifact(
    path: Path, artifact: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> JsonDict:
    """Cold-validate and atomically publish one terminal artifact."""

    errors = validate_artifact(
        artifact, repo_root=repo_root, check_files=artifact.get("status") == "complete"
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    return _atomic_write(path, dict(artifact))


def _command_receipt(command: Sequence[str], *, scope: str) -> tuple[JsonDict, str]:
    """Stream one bounded validation subprocess and retain its exact combined log."""

    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{REPO_ROOT / 'python'}:{REPO_ROOT}",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output.append(line)
    exit_code = process.wait()
    log = "".join(output)
    return (
        {
            "command": shlex.join(command),
            "scope": scope,
            "exit_code": exit_code,
            "duration_s": time.monotonic() - started,
            "log_sha256": transactional.sha256_bytes(log.encode("utf-8")),
        },
        log,
    )


def _validation_commands(candidate: Path) -> list[tuple[list[str], str]]:
    """Return only focused, affected, coverage, static, E2E, and artifact checks."""

    python = str(REPO_ROOT / ".venv/bin/python")
    pytest = str(REPO_ROOT / ".venv/bin/pytest")
    module = "python/carnot/experiment_7310_v642_factor_prototype.py"
    test = str(TEST_PATH)
    wrapper = str(WRAPPER_PATH)
    return [
        (
            [
                pytest,
                test,
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=/tmp/carnot-exp7310-focused",
            ],
            "focused REQ-CL-7310 tests",
        ),
        (
            [
                pytest,
                "tests/python/test_experiment_6748_transactional_constraint_memory_fixture.py",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=/tmp/carnot-exp7310-affected",
            ],
            "affected transactional memory suite",
        ),
        (
            [
                python,
                "-m",
                "coverage",
                "run",
                "--data-file=/tmp/carnot-exp7310.coverage",
                f"--include={module}",
                str(COVERAGE_PATH),
            ],
            "changed-module coverage execution",
        ),
        (
            [
                python,
                "-m",
                "coverage",
                "report",
                "--data-file=/tmp/carnot-exp7310.coverage",
                "--show-missing",
                "--fail-under=100",
            ],
            "100 percent changed-module coverage",
        ),
        ([python, "-m", "ruff", "check", module, test, wrapper], "scoped Ruff check"),
        ([python, "-m", "ruff", "format", "--check", module, test, wrapper], "scoped Ruff format"),
        ([python, "-m", "mypy", module], "changed-module mypy"),
        ([python, "scripts/check_spec_coverage.py", test], "REQ-CL-7310 spec coverage"),
        (
            [
                pytest,
                f"{test}::test_scenario_cl_7310_e2e_opt_in_revision_is_durable",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=/tmp/carnot-exp7310-e2e",
            ],
            "E2E-007 opt-in delayed revision",
        ),
        (
            [
                python,
                "-m",
                "carnot.experiment_7310_v642_factor_prototype",
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ],
            "cold terminal candidate validation",
        ),
        (
            [python, "scripts/adversarial_verify.py", str(candidate)],
            "adversarial artifact verification",
        ),
        (
            [python, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
            "strict verdict-row consistency",
        ),
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed execution date and private validation or output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=[RUN_DATE])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build, validate, and atomically publish only terminal factor evidence."""

    print("phase 0 immediate: Exp7310 factor-local prototype started", flush=True)
    args = _parse_args(argv)
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    if args.validate is not None:
        _progress(1, "before subprocess", f"cold validate {args.validate}")
        errors = validate_artifact(_load_object(args.validate), check_files=True)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        _progress(1, "after subprocess", f"validation errors={len(errors)}")
        return int(bool(errors))
    invocation_started = time.monotonic()
    artifact = build_and_seal(REPO_ROOT, paths, progress=True)
    if artifact["status"] == "blocked":
        write_artifact(paths.artifact, artifact)
        _progress(7, "end", f"wrote blocked artifact {paths.artifact}")
        return 0
    _progress(5, "start", "write measured terminal candidate")
    write_artifact(paths.terminal_candidate, artifact)
    _progress(5, "end", f"candidate={paths.terminal_candidate}")
    receipts = list(artifact["validation_receipts"])
    commands = _validation_commands(paths.terminal_candidate)
    for index, (command, scope) in enumerate(commands, start=1):
        _progress(6, "before subprocess", f"{index}/{len(commands)} {shlex.join(command)}")
        receipt, log = _command_receipt(command, scope=scope)
        log_path = paths.validation_dir / f"{index:02d}.log"
        _atomic_write(log_path, log.encode("utf-8"))
        receipt["log_path"] = str(log_path)
        receipts.append(receipt)
        _progress(
            6,
            "after subprocess",
            f"{index}/{len(commands)} exit={receipt['exit_code']} elapsed={receipt['duration_s']:.3f}s",
        )
    artifact = attach_validation_receipts(artifact, receipts)
    failures = [row for row in receipts if row["exit_code"] != 0]
    if failures:
        raise RuntimeError(
            "focused_validation_failed:" + ",".join(str(row["command"]) for row in failures)
        )
    artifact["duration_s"] = time.monotonic() - invocation_started
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _progress(7, "start", "final cold validation and atomic terminal write")
    receipt = write_artifact(paths.artifact, artifact)
    _progress(7, "end", f"terminal sha256={receipt['sha256']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin wrapper owns normal execution.
    raise SystemExit(main())
