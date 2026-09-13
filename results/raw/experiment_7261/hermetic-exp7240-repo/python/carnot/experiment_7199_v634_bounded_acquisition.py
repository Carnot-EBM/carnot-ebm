"""Measure bounded feedback acquisition with an exact finite version space.

The learner updates only from labels that its bounded queue requested and later
released. Validation labels test a frozen singleton but never help fit it. The
experiment replays immutable Exp7198 bytes and does not invoke an LLM.

Spec refs: REQ-CL-7199 and SCENARIO-CL-7199-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
from typing import Any

from carnot import experiment_7184_v633_revocable_template_csl as exp7184
from carnot import experiment_7198_v634_feedback_capacity_stream as exp7198


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7199
SCHEMA = "carnot.exp7199.v634_bounded_acquisition.v1"
MILESTONE = "2026.09.634"
RUN_DATE = "20260911"
RANDOM_SEED = 7_199_202_609_11
STREAM_SEEDS = tuple(exp7198.STREAM_SEEDS)
EVENTS_PER_SEED = exp7198.EVENTS_PER_SEED
PARAMETER_DOMAIN = tuple(exp7198.PARAMETER_DOMAIN)
FAMILIES = tuple(exp7198.FAMILIES)
CAPACITIES = tuple(exp7198.CAPACITIES)
DELAY_SCHEDULES = tuple(exp7198.DELAY_SCHEDULES)
CELLS = tuple(itertools.product(CAPACITIES, DELAY_SCHEDULES))
PRIMARY_CELL = {"capacity": 4, "delay_schedule": "burst"}
BLOCK_SIZE = exp7198.BLOCK_SIZE
WARMUP_COUNT = exp7198.WARMUP_COUNT
WARMUP_REQUEST_STOP = exp7198.WARMUP_REQUEST_STOP
MEMORY_BYTE_BUDGET = exp7198.MEMORY_BYTE_BUDGET
SUPPORT_REQUIRED = 3
VALIDATION_REQUIRED = 8
LABEL_QUOTA = 248
BOOTSTRAP_DRAWS = 2_000
DEPLOYABLE_ARMS = (
    "warmup_frozen",
    "fifo_admission",
    "random_admission",
    "priority_admission",
)
ARMS = (*DEPLOYABLE_ARMS, "all_information_oracle")
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

DEFAULT_UPSTREAM_ARTIFACT_PATH = Path("results/experiment_7198_v634_feedback_capacity_stream.json")
DEFAULT_CHECKPOINT_PATH = Path(
    "results/checkpoints/experiment_7199_v634_bounded_acquisition_progress.json"
)
DEFAULT_ARTIFACT_PATH = Path("results/experiment_7199_v634_bounded_acquisition.json")
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
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("python/carnot/experiment_7184_v633_revocable_template_csl.py"),
    Path("python/carnot/experiment_7185_v633_memory_cold_audit.py"),
    Path("python/carnot/experiment_7198_v634_feedback_capacity_stream.py"),
    Path("python/carnot/experiment_7199_v634_bounded_acquisition.py"),
    Path("scripts/experiments/experiment_7199_v634_bounded_acquisition.py"),
    Path("tests/python/test_experiment_7199_v634_bounded_acquisition.py"),
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
    "acquisition_run_complete_score",
    "acquisition_value_score",
    "continuous_self_learning_task",
    "decision_rows",
    "update_rows",
    "pending_queue_rows",
    "acceptance_gate_learning",
    "future_hardware_path",
    "no_model_weight_mutation",
    "prediction_contract",
    "validation_access_rows",
    "validation_partition_rows",
    "MODEL_SPECS",
    "model_invoked",
    "comparison_rows",
    "source_grounding_rows",
    "latency_summary",
    "benefit_decomposition",
    "priority_specific_benefit_score",
    "memory_capacity_violation_count",
    "checkpoint_path",
    "checkpoint_hash",
    "upstream_receipt",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "A versioned schema makes incompatible readers fail closed.",
    "experiment_id": "A fixed ID prevents another task from supplying this evidence.",
    "milestone": "The milestone binds the run to the V634 contract.",
    "field_principles": "Echo each declared reason beside the actual evidence contract.",
    "status": "Terminal only after completion or a diagnosed external block.",
    "run_date": "Use 20260911, never a historical date.",
    "preconditions_checked": "Record each required resource and its actual observed state.",
    "inference_substrate": "Describe executed computation, not the planned workload.",
    "inference_substrate_class": "Apply the duration floor for the work actually performed.",
    "execution_venue": "Host or device identity limits the scope of the evidence.",
    "duration_s": "Measure monotonic elapsed work; never pad time to pass a floor.",
    "source_artifact_hashes": "Bind code, inputs and frozen contracts to the result.",
    "rows": "Retain unit ID, arm, seed, metric, error and abstention for every comparison.",
    "sample_size_budget": "Record planned and completed counts, independent units and exclusions.",
    "random_seed": "Freeze all stochastic choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash inputs, code, seeds and raw rows.",
    "gate_check_summary": (
        "Every blocked verdict names the failed check, upstream, field, expected and "
        "observed value."
    ),
    "verifier_is_oracle": (
        "True when verification uses the same correctness authority; separate "
        "implementations alone do not remove circularity."
    ),
    "verdict_class": (
        "Use positive | circular_positive | null | blocked | disqualified | partial. "
        "Only incomplete own work can be partial."
    ),
    "honest_verdict": (
        "Use complete_ or complete: for completed findings, including nulls; blocked_* "
        "for external blocks. Never promote infrastructure readiness as scientific benefit."
    ),
    "acquisition_run_complete_score": "A completed null is still auditable downstream.",
    "acquisition_value_score": "One requires the frozen primary-cell gate.",
    "continuous_self_learning_task": "True denotes actual updates from past released feedback.",
    "decision_rows": "Each event records arm, prior state, prediction, cost and eventual outcome.",
    "update_rows": "Support IDs and state hashes prove constraint addition was real.",
    "pending_queue_rows": "Track occupancy, admission, release, eviction and lost labels.",
    "acceptance_gate_learning": "Freeze the primary cell and all error/safety thresholds.",
    "future_hardware_path": (
        "Bitset operations supply a concrete acceleration path without a speed claim."
    ),
    "no_model_weight_mutation": (
        "True prevents CPU memory learning from being labeled LLM training."
    ),
    "prediction_contract": (
        "Majority vote, reject ties, empty-set abstention; abstention counts as "
        "full-denominator error."
    ),
    "validation_access_rows": (
        "Every commit-support and validation label was requested, released and charged "
        "before the decision."
    ),
    "validation_partition_rows": (
        "Roles precede label reveal; held-out validation never eliminates hypotheses "
        "and follows candidate freeze."
    ),
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is cited, not repeated.",
    "comparison_rows": "Every frozen cell uses seed-paired intervals without event resampling.",
    "source_grounding_rows": (
        "Public extraction, predicate execution and independent scoring must agree."
    ),
    "latency_summary": "Measured p50 and p95 costs include lookup, selection, storage and update.",
    "benefit_decomposition": (
        "Version-space decisions and identical template persistence receive separate credit."
    ),
    "priority_specific_benefit_score": (
        "Priority receives separate credit only when it beats matched random admission."
    ),
    "memory_capacity_violation_count": "Zero is required by every value gate.",
    "checkpoint_path": "Real progress is stored under results/checkpoints, not the deliverable.",
    "checkpoint_hash": "The checkpoint digest binds the completed seed units.",
    "upstream_receipt": "Exact hashes bind immutable public, authority and manifest bytes.",
}

canonical_json = exp7198.canonical_json
sha256_bytes = exp7198.sha256_bytes
sha256_json = exp7198.sha256_json
sha256_path = exp7198.sha256_path
gate_check = exp7184.gate_check
gate_summary = exp7184.gate_summary
write_json_atomic = exp7198.write_json_atomic


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep progress bytes separate from the terminal result."""

    checkpoint: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the repository paths used by the public command."""

        return cls(DEFAULT_CHECKPOINT_PATH, DEFAULT_ARTIFACT_PATH)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put test outputs under one caller-owned directory."""

        return cls(
            root / "checkpoints" / "progress.json",
            root / "experiment_7199_v634_bounded_acquisition.json",
        )


@dataclass
class FamilyState:
    """Store the complete bounded state for one predicate family."""

    hypotheses: set[int] = field(default_factory=lambda: set(PARAMETER_DOMAIN))
    support_ids: list[str] = field(default_factory=list)
    validation_ids: list[str] = field(default_factory=list)
    candidate_parameter: int | None = None
    freeze_release_index: int | None = None
    committed_template: JsonDict | None = None
    superseded_templates: list[JsonDict] = field(default_factory=list)
    archive: list[list[Any]] = field(default_factory=list)
    epoch: int = 0

    def as_dict(self) -> JsonDict:
        """Serialize all bytes that can affect a later decision."""

        return {
            "hypotheses": sorted(self.hypotheses),
            "support_ids": list(self.support_ids),
            "validation_ids": list(self.validation_ids),
            "candidate_parameter": self.candidate_parameter,
            "freeze_release_index": self.freeze_release_index,
            "committed_template": deepcopy(self.committed_template),
            "superseded_templates": deepcopy(self.superseded_templates),
            "archive": deepcopy(self.archive),
            "epoch": self.epoch,
        }


def _predicate_name(family_id: str, parameter: int) -> str:
    """Name the exact executable predicate stored after validation."""

    return f"{family_id}(parameter={parameter})"


class VersionSpaceController:
    """Eliminate finite hypotheses using released support labels only."""

    def __init__(self) -> None:
        self.families = {family_id: FamilyState() for family_id in FAMILIES}

    def state_dict(self) -> JsonDict:
        """Return stable state for hashes and the memory byte gate."""

        return {family_id: self.families[family_id].as_dict() for family_id in FAMILIES}

    def state_bytes(self) -> bytes:
        """Encode every persistent byte that can affect future predictions."""

        return canonical_json(self.state_dict())

    def state_hash(self) -> str:
        """Identify one exact controller state."""

        return sha256_bytes(self.state_bytes())

    def predict(self, public_event: Mapping[str, Any]) -> tuple[str, float]:
        """Return majority prediction and public disagreement fraction."""

        family_id = str(public_event["family_id"])
        hypotheses = self.families[family_id].hypotheses
        if not hypotheses:
            return "abstain", 0.0
        labels = [
            exp7198.exact_label(family_id, int(public_event["numeric_value"]), parameter)
            for parameter in hypotheses
        ]
        accepts = labels.count("accept")
        disagreement = min(accepts, len(labels) - accepts) / len(labels)
        return ("accept" if accepts > len(labels) / 2 else "reject"), disagreement

    @staticmethod
    def _retire_template(
        state: FamilyState,
        event_id: str,
        rollback_hash: str,
    ) -> None:
        """Preserve superseded predicate bytes before clearing active state."""

        if state.committed_template is not None:
            retired = deepcopy(state.committed_template)
            retired.update(
                {
                    "superseded_at_event_id": event_id,
                    "rollback_hash": rollback_hash,
                    "superseded_epoch": state.epoch,
                }
            )
            state.superseded_templates.append(retired)
            state.committed_template = None

    @classmethod
    def _reset_epoch(
        cls,
        state: FamilyState,
        event_id: str,
        rollback_hash: str,
    ) -> None:
        """Open a fresh epoch without reusing old support or validation credit."""

        cls._retire_template(state, event_id, rollback_hash)
        state.epoch += 1
        state.hypotheses = set(PARAMETER_DOMAIN)
        state.support_ids = []
        state.validation_ids = []
        state.candidate_parameter = None
        state.freeze_release_index = None

    def observe(
        self,
        public_event: Mapping[str, Any],
        *,
        observed_label: str,
        role: str,
        request_index: int,
        release_index: int,
    ) -> JsonDict:
        """Apply one charged release while preserving role separation."""

        if role not in {"support", "validation"}:
            raise ValueError(f"invalid_feedback_role:{role}")
        family_id = str(public_event["family_id"])
        event_id = str(public_event["event_id"])
        state = self.families[family_id]
        before_hash = self.state_hash()
        hypotheses_before = sorted(state.hypotheses)
        prediction_before = self.predict(public_event)[0]
        state.archive.append(
            [
                event_id,
                role,
                int(request_index),
                int(release_index),
                str(observed_label),
                state.epoch,
            ]
        )
        rollback_hash: str | None = None
        operation = "validation_archived_before_freeze"

        if role == "support":
            survivors = {
                parameter
                for parameter in state.hypotheses
                if exp7198.exact_label(family_id, int(public_event["numeric_value"]), parameter)
                == observed_label
            }
            if not survivors:
                rollback_hash = before_hash
                had_template = state.committed_template is not None
                self._reset_epoch(state, event_id, rollback_hash)
                state.hypotheses = {
                    parameter
                    for parameter in state.hypotheses
                    if exp7198.exact_label(family_id, int(public_event["numeric_value"]), parameter)
                    == observed_label
                }
                state.support_ids.append(event_id)
                operation = "revoke_reset_support" if had_template else "empty_reset_support"
            else:
                state.hypotheses = survivors
                if event_id not in state.support_ids:
                    state.support_ids.append(event_id)
                operation = "support_eliminate"
            if (
                len(state.hypotheses) == 1
                and len(state.support_ids) >= SUPPORT_REQUIRED
                and state.candidate_parameter is None
            ):
                state.candidate_parameter = next(iter(state.hypotheses))
                state.freeze_release_index = release_index
                state.validation_ids = []
                operation = "freeze_singleton"
        elif state.candidate_parameter is not None:
            candidate_label = exp7198.exact_label(
                family_id,
                int(public_event["numeric_value"]),
                state.candidate_parameter,
            )
            requested_after_freeze = (
                state.freeze_release_index is not None
                and request_index > state.freeze_release_index
                and release_index > state.freeze_release_index
            )
            if requested_after_freeze and candidate_label != observed_label:
                rollback_hash = before_hash
                self._reset_epoch(state, event_id, rollback_hash)
                operation = "candidate_rejected_reset"
            elif requested_after_freeze and event_id not in state.validation_ids:
                state.validation_ids.append(event_id)
                operation = "validation_pass"
                if (
                    len(state.validation_ids) >= VALIDATION_REQUIRED
                    and state.committed_template is None
                ):
                    parameter = state.candidate_parameter
                    state.committed_template = {
                        "family_id": family_id,
                        "parameter": parameter,
                        "predicate": _predicate_name(family_id, parameter),
                        "epoch": state.epoch,
                        "support_ids": list(state.support_ids),
                        "validation_ids": list(state.validation_ids),
                        "freeze_release_index": state.freeze_release_index,
                        "commit_release_index": release_index,
                    }
                    operation = "commit_template"
            else:
                operation = "validation_archived_not_post_freeze"

        prediction_after = self.predict(public_event)[0]
        result = {
            "event_id": event_id,
            "family_id": family_id,
            "role": role,
            "request_index": request_index,
            "release_index": release_index,
            "operation": operation,
            "epoch": state.epoch,
            "state_hash_before": before_hash,
            "state_hash_after": self.state_hash(),
            "hypotheses_before": hypotheses_before,
            "hypotheses_after": sorted(state.hypotheses),
            "support_ids": list(state.support_ids),
            "validation_ids": list(state.validation_ids),
            "candidate_parameter": state.candidate_parameter,
            "freeze_release_index": state.freeze_release_index,
            "committed_template": deepcopy(state.committed_template),
            "rollback_hash": rollback_hash,
            "validation_used_for_elimination": False,
            "prediction_changed_by_commit": (
                False if operation == "commit_template" else prediction_before != prediction_after
            ),
        }
        return result


@dataclass(frozen=True)
class PendingRecord:
    """Keep delivery authority outside the public admission callback."""

    event_id: str
    family_id: str
    numeric_value: int
    role: str
    request_index: int
    release_index: int
    observed_label: str
    exact_label: str
    poisoned: bool


@dataclass(frozen=True)
class AcquisitionPanel:
    """Retain every row needed for a later independent cold replay."""

    rows: list[JsonDict]
    decision_rows: list[JsonDict]
    update_rows: list[JsonDict]
    pending_queue_rows: list[JsonDict]
    validation_access_rows: list[JsonDict]
    validation_partition_rows: list[JsonDict]


def partition_roles(block: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Freeze two support and two validation roles from public bytes only."""

    if len(block) != BLOCK_SIZE:
        raise ValueError("role_partition_requires_four_events")
    ranked = sorted(
        block,
        key=lambda row: sha256_json(
            [
                "role",
                row["event_id"],
                row["family_id"],
                row["numeric_value"],
                row["public_input"],
            ]
        ),
    )
    support_ids = {str(row["event_id"]) for row in ranked[::2]}
    return {
        str(row["event_id"]): ("support" if str(row["event_id"]) in support_ids else "validation")
        for row in block
    }


def seeded_tie_ranks(
    seed: int,
    block_index: int,
    block: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Reuse Exp7198's stable tie permutation for both adaptive selectors."""

    return exp7198.seeded_tie_ranks(seed, block_index, block)


def select_request(
    block: Sequence[Mapping[str, Any]],
    arm: str,
    tie_ranks: Mapping[str, int],
    controller: VersionSpaceController,
) -> Mapping[str, Any]:
    """Select one public event without labels, regimes, delays or change times."""

    if arm == "fifo_admission":
        return block[0]
    if arm == "random_admission":
        return min(block, key=lambda row: tie_ranks[str(row["event_id"])])
    if arm == "priority_admission":
        return min(
            block,
            key=lambda row: (
                -controller.predict(row)[1],
                tie_ranks[str(row["event_id"])],
            ),
        )
    raise ValueError(f"unsupported_admission_arm:{arm}")


def _resolve(repo_root: Path, path: Path | str) -> Path:
    """Resolve an evidence path against the selected checkout."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _load_object(path: Path) -> JsonDict:
    """Parse one JSON object and preserve malformed input as failed evidence."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Load chronological immutable rows without changing their order."""

    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError):
        return []
    return [row for row in rows if isinstance(row, dict)]


def load_upstream_views(
    repo_root: Path,
    upstream_artifact_path: Path,
) -> exp7198.StreamViews:
    """Load sealed stream rows only after the caller has passed quarantine gates."""

    upstream = _load_object(_resolve(repo_root, upstream_artifact_path))
    public_path = _resolve(repo_root, str(upstream.get("public_stream_path", "missing")))
    authority_path = _resolve(repo_root, str(upstream.get("authority_sidecar_path", "missing")))
    manifest_path = _resolve(repo_root, str(upstream.get("stream_manifest_path", "missing")))
    views = exp7198.StreamViews(
        _read_jsonl(public_path),
        _read_jsonl(authority_path),
        _load_object(manifest_path),
    )
    errors = exp7198.stream_conformance_errors(views)
    if errors:
        raise ValueError("upstream_stream_conformance_failed:" + ",".join(errors))
    return views


def _percentile(values: Sequence[int], probability: float) -> int:
    """Return a deterministic nearest-rank latency percentile."""

    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(probability * (len(ordered) - 1)))
    return int(ordered[index])


def _window_metrics() -> dict[str, JsonDict]:
    """Create full-denominator counters for every frozen stream window."""

    return {
        str(window["name"]): {
            "events": 0,
            "error": 0,
            "false_accept": 0,
            "abstention": 0,
        }
        for window in exp7198.WINDOWS
    }


def _memory_bytes(
    controller: VersionSpaceController,
    pending: Sequence[PendingRecord],
    staging: Sequence[Mapping[str, Any]],
) -> int:
    """Charge controller, pending records and the four-event staging block."""

    payload = {
        "controller": controller.state_dict(),
        "pending": [record.__dict__ for record in pending],
        "staging": [dict(row) for row in staging],
    }
    return len(canonical_json(payload))


def _finish_window_metrics(metrics: Mapping[str, Mapping[str, int]]) -> JsonDict:
    """Add rates while keeping abstentions in the error denominator."""

    result: JsonDict = {}
    for name, raw in metrics.items():
        events = int(raw["events"])
        result[name] = {
            **dict(raw),
            "error_rate": raw["error"] / events,
            "false_accept_rate": raw["false_accept"] / events,
            "abstention_rate": raw["abstention"] / events,
        }
    return result


def _partition_rows(
    public_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    seeds: Sequence[int],
) -> list[JsonDict]:
    """Record the public role freeze once because all cells reuse it."""

    rows: list[JsonDict] = []
    for seed in seeds:
        events = public_by_seed[seed]
        for block_start in range(0, EVENTS_PER_SEED, BLOCK_SIZE):
            block = events[block_start : block_start + BLOCK_SIZE]
            roles = partition_roles(block)
            for event in block:
                event_id = str(event["event_id"])
                rows.append(
                    {
                        "event_id": event_id,
                        "seed": seed,
                        "block_index": block_start // BLOCK_SIZE,
                        "role": roles[event_id],
                        "public_role_hash": sha256_json(
                            [
                                "role",
                                event_id,
                                event["family_id"],
                                event["numeric_value"],
                                event["public_input"],
                            ]
                        ),
                        "role_frozen_before_label": True,
                        "label_visible_when_frozen": False,
                    }
                )
    return rows


def run_acquisition_panel(
    views: exp7198.StreamViews,
    *,
    seeds: Sequence[int] = STREAM_SEEDS,
    cells: Sequence[tuple[int, str]] = CELLS,
    checkpoint_path: Path | None = None,
    progress: bool = False,
) -> AcquisitionPanel:
    """Replay matched controllers and queues over every requested stream cell."""

    started = time.monotonic()
    authority = {str(row["event_id"]): row for row in views.authority_events}
    public_by_seed = {
        seed: [row for row in views.public_events if int(row["seed"]) == seed] for seed in seeds
    }
    partition_rows = _partition_rows(public_by_seed, seeds)
    rows: list[JsonDict] = []
    decision_rows: list[JsonDict] = []
    update_rows: list[JsonDict] = []
    pending_rows: list[JsonDict] = []
    access_rows: list[JsonDict] = []

    for seed_number, seed in enumerate(seeds, start=1):
        seed_events = public_by_seed[seed]
        for capacity, delay_schedule in cells:
            controllers = {arm: VersionSpaceController() for arm in DEPLOYABLE_ARMS}
            pending: dict[str, list[PendingRecord]] = {arm: [] for arm in DEPLOYABLE_ARMS}
            operation_index = {arm: 0 for arm in ARMS}
            metrics = {arm: _window_metrics() for arm in ARMS}
            requests = {arm: 0 for arm in ARMS}
            releases = {arm: 0 for arm in ARMS}
            dropped = {arm: 0 for arm in ARMS}
            max_pending = {arm: 0 for arm in ARMS}
            max_memory = {arm: 0 for arm in ARMS}
            commit_counts = {arm: 0 for arm in ARMS}
            revocation_counts = {arm: 0 for arm in ARMS}
            lookup_costs: dict[str, list[int]] = {arm: [] for arm in ARMS}
            update_costs: dict[str, list[int]] = {arm: [] for arm in ARMS}
            selection_costs: dict[str, list[int]] = {arm: [] for arm in ARMS}
            storage_costs: dict[str, list[int]] = {arm: [] for arm in ARMS}
            warmup_hashes: dict[str, str | None] = {arm: None for arm in ARMS}

            for block_start in range(0, EVENTS_PER_SEED, BLOCK_SIZE):
                block = seed_events[block_start : block_start + BLOCK_SIZE]
                block_end = block_start + BLOCK_SIZE - 1
                block_index = block_start // BLOCK_SIZE
                roles = partition_roles(block)
                tie_ranks = seeded_tie_ranks(seed, block_index, block)

                for arm in ARMS:
                    controller = controllers.get(arm)
                    drafts: list[JsonDict] = []
                    for public_event in block:
                        operation_index[arm] += 1
                        lookup_started = time.perf_counter_ns()
                        if arm == "all_information_oracle":
                            prediction = str(
                                authority[str(public_event["event_id"])]["exact_label"]
                            )
                            disagreement = 0.0
                            prior_state_hash = "all_information_oracle"
                        else:
                            if (
                                controller is None
                            ):  # pragma: no cover - fixed arm table prevents this.
                                raise RuntimeError("missing_controller")
                            prior_state_hash = controller.state_hash()
                            prediction, disagreement = controller.predict(public_event)
                        lookup_ns = time.perf_counter_ns() - lookup_started
                        lookup_costs[arm].append(lookup_ns)
                        drafts.append(
                            {
                                "unit_id": f"{seed}:{capacity}:{delay_schedule}:{arm}",
                                "event_id": str(public_event["event_id"]),
                                "arm": arm,
                                "seed": seed,
                                "capacity": capacity,
                                "delay_schedule": delay_schedule,
                                "prior_state_hash": prior_state_hash,
                                "prediction": prediction,
                                "disagreement": disagreement,
                                "lookup_ns": lookup_ns,
                                "prediction_index": operation_index[arm],
                            }
                        )

                    occupancy_before = 0 if arm == "all_information_oracle" else len(pending[arm])
                    eligible = block_end < WARMUP_REQUEST_STOP or (
                        block_start >= WARMUP_COUNT
                        and arm not in {"warmup_frozen", "all_information_oracle"}
                    )
                    selected: Mapping[str, Any] | None = None
                    request_status = "not_requested"
                    selection_ns = 0
                    storage_ns = 0
                    if eligible and controller is not None:
                        selector = "priority_admission" if block_start < WARMUP_COUNT else arm
                        selection_started = time.perf_counter_ns()
                        selected = select_request(block, selector, tie_ranks, controller)
                        selection_ns = time.perf_counter_ns() - selection_started
                        selection_costs[arm].append(selection_ns)
                        if occupancy_before >= capacity:
                            dropped[arm] += 1
                            request_status = "dropped_capacity_full"
                        elif requests[arm] >= LABEL_QUOTA:
                            request_status = "quota_exhausted"
                        else:
                            truth = authority[str(selected["event_id"])]
                            storage_started = time.perf_counter_ns()
                            pending[arm].append(
                                PendingRecord(
                                    event_id=str(selected["event_id"]),
                                    family_id=str(selected["family_id"]),
                                    numeric_value=int(selected["numeric_value"]),
                                    role=roles[str(selected["event_id"])],
                                    request_index=block_end,
                                    release_index=(
                                        block_end + int(truth["delay_by_schedule"][delay_schedule])
                                    ),
                                    observed_label=str(truth["observed_label"]),
                                    exact_label=str(truth["exact_label"]),
                                    poisoned=bool(truth["poisoned"]),
                                )
                            )
                            storage_ns = time.perf_counter_ns() - storage_started
                            storage_costs[arm].append(storage_ns)
                            requests[arm] += 1
                            request_status = "admitted"

                    if controller is not None:
                        max_pending[arm] = max(max_pending[arm], len(pending[arm]))
                        max_memory[arm] = max(
                            max_memory[arm],
                            _memory_bytes(controller, pending[arm], block),
                        )

                    for draft in drafts:
                        operation_index[arm] += 1
                        truth = authority[draft["event_id"]]
                        prediction = str(draft["prediction"])
                        error = int(prediction == "abstain" or prediction != truth["exact_label"])
                        false_accept = int(
                            prediction == "accept" and truth["exact_label"] == "reject"
                        )
                        abstention = int(prediction == "abstain")
                        window = str(truth["window"])
                        metric = metrics[arm][window]
                        metric["events"] += 1
                        metric["error"] += error
                        metric["false_accept"] += false_accept
                        metric["abstention"] += abstention
                        draft.update(
                            {
                                "window": window,
                                "exact_label": truth["exact_label"],
                                "error": error,
                                "false_accept": false_accept,
                                "abstention": abstention,
                                "outcome_score_index": operation_index[arm],
                            }
                        )
                        decision_rows.append(draft)

                    released: list[PendingRecord] = []
                    if controller is not None:
                        released = [
                            record for record in pending[arm] if record.release_index <= block_end
                        ]
                        for record in released:
                            operation_index[arm] += 1
                            public_release = {
                                "event_id": record.event_id,
                                "family_id": record.family_id,
                                "numeric_value": record.numeric_value,
                            }
                            update_started = time.perf_counter_ns()
                            update = controller.observe(
                                public_release,
                                observed_label=record.observed_label,
                                role=record.role,
                                request_index=record.request_index,
                                release_index=record.release_index,
                            )
                            update_ns = time.perf_counter_ns() - update_started
                            update_costs[arm].append(update_ns)
                            update.update(
                                {
                                    "unit_id": f"{seed}:{capacity}:{delay_schedule}:{arm}",
                                    "arm": arm,
                                    "seed": seed,
                                    "capacity": capacity,
                                    "delay_schedule": delay_schedule,
                                    "update_ns": update_ns,
                                    "release_operation_index": operation_index[arm],
                                    "requested": True,
                                    "released": True,
                                    "quota_charged": True,
                                    "capacity_charged": True,
                                }
                            )
                            update_rows.append(update)
                            access_rows.append(
                                {
                                    "unit_id": update["unit_id"],
                                    "event_id": record.event_id,
                                    "arm": arm,
                                    "seed": seed,
                                    "capacity": capacity,
                                    "delay_schedule": delay_schedule,
                                    "role": record.role,
                                    "request_index": record.request_index,
                                    "release_index": record.release_index,
                                    "release_operation_index": operation_index[arm],
                                    "requested": True,
                                    "released": True,
                                    "quota_charged": True,
                                    "capacity_charged": True,
                                    "role_frozen_before_label": True,
                                    "validation_used_for_elimination": False,
                                    "candidate_freeze_release_index": update[
                                        "freeze_release_index"
                                    ],
                                    "operation": update["operation"],
                                }
                            )
                            releases[arm] += 1
                            commit_counts[arm] += int(update["operation"] == "commit_template")
                            revocation_counts[arm] += int(
                                update["operation"]
                                in {"revoke_reset_support", "candidate_rejected_reset"}
                                and update["rollback_hash"] is not None
                            )
                        released_ids = {record.event_id for record in released}
                        pending[arm] = [
                            record for record in pending[arm] if record.event_id not in released_ids
                        ]
                        max_memory[arm] = max(
                            max_memory[arm],
                            _memory_bytes(controller, pending[arm], []),
                        )

                    pending_rows.append(
                        {
                            "unit_id": f"{seed}:{capacity}:{delay_schedule}:{arm}",
                            "arm": arm,
                            "seed": seed,
                            "capacity": capacity,
                            "delay_schedule": delay_schedule,
                            "block_index": block_index,
                            "chronology_index": block_end,
                            "predictions_committed": BLOCK_SIZE,
                            "selected_event_id": None if selected is None else selected["event_id"],
                            "request_status": request_status,
                            "occupancy_before_selection": occupancy_before,
                            "occupancy_after_request": (
                                0
                                if controller is None
                                else occupancy_before + int(request_status == "admitted")
                            ),
                            "release_count": len(released),
                            "occupancy_after_release": 0
                            if controller is None
                            else len(pending[arm]),
                            "pending_eviction_count": 0,
                            "lost_label_count": int(request_status == "dropped_capacity_full"),
                            "future_delay_visible_to_selector": False,
                            "label_visible_to_selector": False,
                            "selection_after_predictions": True,
                            "release_after_selection": True,
                            "selection_ns": selection_ns,
                            "storage_ns": storage_ns,
                        }
                    )

                    if block_end == WARMUP_COUNT - 1:
                        warmup_hashes[arm] = (
                            "all_information_oracle"
                            if controller is None
                            else controller.state_hash()
                        )

            for arm in ARMS:
                window_metrics = _finish_window_metrics(metrics[arm])
                total_error = sum(int(value["error"]) for value in window_metrics.values())
                total_abstention = sum(
                    int(value["abstention"]) for value in window_metrics.values()
                )
                total_cost = (
                    sum(lookup_costs[arm])
                    + sum(update_costs[arm])
                    + sum(selection_costs[arm])
                    + sum(storage_costs[arm])
                )
                controller = controllers.get(arm)
                rows.append(
                    {
                        "unit_id": f"{seed}:{capacity}:{delay_schedule}:{arm}",
                        "arm": arm,
                        "seed": seed,
                        "capacity": capacity,
                        "delay_schedule": delay_schedule,
                        "metric": "full_stream_error_rate",
                        "error": total_error,
                        "abstention": total_abstention,
                        "event_count": EVENTS_PER_SEED,
                        "error_rate": total_error / EVENTS_PER_SEED,
                        "requests": requests[arm],
                        "releases": releases[arm],
                        "dropped_requests": dropped[arm],
                        "lost_labels_at_stream_end": (
                            0 if controller is None else len(pending[arm])
                        ),
                        "label_quota": LABEL_QUOTA,
                        "max_pending": max_pending[arm],
                        "max_memory_bytes": max_memory[arm],
                        "memory_byte_budget": MEMORY_BYTE_BUDGET,
                        "pending_eviction_count": 0,
                        "warmup_state_hash": warmup_hashes[arm],
                        "final_state_hash": (
                            "all_information_oracle"
                            if controller is None
                            else controller.state_hash()
                        ),
                        "template_commit_count": commit_counts[arm],
                        "template_revocation_count": revocation_counts[arm],
                        "lookup_p50_ns": _percentile(lookup_costs[arm], 0.50),
                        "lookup_p95_ns": _percentile(lookup_costs[arm], 0.95),
                        "update_p50_ns": _percentile(update_costs[arm], 0.50),
                        "update_p95_ns": _percentile(update_costs[arm], 0.95),
                        "selection_p50_ns": _percentile(selection_costs[arm], 0.50),
                        "selection_p95_ns": _percentile(selection_costs[arm], 0.95),
                        "storage_p50_ns": _percentile(storage_costs[arm], 0.50),
                        "storage_p95_ns": _percentile(storage_costs[arm], 0.95),
                        "end_to_end_cost_ns": total_cost,
                        "window_metrics": window_metrics,
                    }
                )

        if checkpoint_path is not None:
            checkpoint = {
                "schema": "carnot.exp7199.progress.v1",
                "completed_seeds": list(seeds[:seed_number]),
                "completed_unit_count": len(rows),
                "decision_row_count": len(decision_rows),
                "update_row_count": len(update_rows),
                "structural_hash": sha256_json(
                    {
                        "rows": _stable_timing_value(rows),
                        "completed_seeds": list(seeds[:seed_number]),
                    }
                ),
            }
            write_json_atomic(checkpoint_path, checkpoint)
        if progress:
            print(
                f"PHASE 3 PROGRESS: completed seed {seed_number}/{len(seeds)}; "
                f"terminal_units={len(rows)}; elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )

    return AcquisitionPanel(
        rows,
        decision_rows,
        update_rows,
        pending_rows,
        access_rows,
        partition_rows,
    )


def panel_conformance_errors(
    panel: AcquisitionPanel,
    *,
    seeds: Sequence[int] = STREAM_SEEDS,
    cells: Sequence[tuple[int, str]] = CELLS,
) -> list[str]:
    """Check chronology, full denominators, role separation, quotas and bytes."""

    errors: list[str] = []
    expected_units = len(seeds) * len(cells) * len(ARMS)
    expected_decisions = expected_units * EVENTS_PER_SEED
    expected_pending = expected_units * (EVENTS_PER_SEED // BLOCK_SIZE)
    expected_identities = {
        (seed, capacity, delay, arm) for seed in seeds for capacity, delay in cells for arm in ARMS
    }
    identities = {
        (row["seed"], row["capacity"], row["delay_schedule"], row["arm"]) for row in panel.rows
    }
    if len(panel.rows) != expected_units or identities != expected_identities:
        errors.append("terminal_unit_panel")
    if len(panel.decision_rows) != expected_decisions:
        errors.append("decision_row_count")
    if len(panel.pending_queue_rows) != expected_pending:
        errors.append("pending_queue_row_count")
    if any(
        int(row["prediction_index"]) >= int(row["outcome_score_index"])
        for row in panel.decision_rows
    ):
        errors.append("prediction_outcome_chronology")
    if any(row["abstention"] == 1 and row["error"] != 1 for row in panel.decision_rows):
        errors.append("abstention_denominator")
    if any(int(row["max_pending"]) > int(row["capacity"]) for row in panel.rows):
        errors.append("pending_capacity")
    if any(int(row["max_memory_bytes"]) > MEMORY_BYTE_BUDGET for row in panel.rows):
        errors.append("memory_budget")
    if any(int(row["pending_eviction_count"]) != 0 for row in panel.rows):
        errors.append("pending_eviction")
    if any(int(row["requests"]) > LABEL_QUOTA for row in panel.rows):
        errors.append("label_quota")
    if any(
        row["future_delay_visible_to_selector"] is not False
        or row["label_visible_to_selector"] is not False
        or row["selection_after_predictions"] is not True
        or row["release_after_selection"] is not True
        or int(row["pending_eviction_count"]) != 0
        for row in panel.pending_queue_rows
    ):
        errors.append("queue_information_chronology")
    if any(
        int(row["request_index"]) > int(row["release_index"])
        or row["requested"] is not True
        or row["released"] is not True
        or row["quota_charged"] is not True
        or row["capacity_charged"] is not True
        for row in panel.validation_access_rows
    ):
        errors.append("feedback_access")
    if any(
        row["role"] == "validation" and row["validation_used_for_elimination"] is not False
        for row in panel.validation_access_rows
    ):
        errors.append("validation_fitted_hypothesis")
    expected_partitions = len(seeds) * EVENTS_PER_SEED
    if len(panel.validation_partition_rows) != expected_partitions:
        errors.append("role_partition_count")
    for seed in seeds:
        seed_rows = [row for row in panel.validation_partition_rows if row["seed"] == seed]
        for block_index in range(EVENTS_PER_SEED // BLOCK_SIZE):
            roles = [row["role"] for row in seed_rows if row["block_index"] == block_index]
            if roles.count("support") != 2 or roles.count("validation") != 2:
                errors.append("role_partition_balance")
                break
    for seed in seeds:
        for capacity, delay in cells:
            warmup = [
                row
                for row in panel.rows
                if row["seed"] == seed
                and row["capacity"] == capacity
                and row["delay_schedule"] == delay
                and row["arm"] in DEPLOYABLE_ARMS
            ]
            if len({row["warmup_state_hash"] for row in warmup}) != 1:
                errors.append("warmup_state_mismatch")
                break
    return list(dict.fromkeys(errors))


def paired_seed_bootstrap(
    left: Mapping[int, float],
    right: Mapping[int, float],
    *,
    seed: int,
) -> JsonDict:
    """Resample paired stream seeds while keeping all event rows dependent."""

    seed_ids = sorted(set(left) & set(right))
    if not seed_ids:
        raise ValueError("paired_bootstrap_requires_seeds")
    differences = [float(left[item]) - float(right[item]) for item in seed_ids]
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [differences[rng.randrange(len(differences))] for _ in seed_ids]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    return {
        "difference": round(sum(differences) / len(differences), 12),
        "ci95_low": round(low, 12),
        "ci95_high": round(high, 12),
    }


_PAIRWISE_COMPARISONS = (
    ("fifo_admission", "warmup_frozen"),
    ("random_admission", "warmup_frozen"),
    ("priority_admission", "warmup_frozen"),
    ("random_admission", "fifo_admission"),
    ("priority_admission", "fifo_admission"),
    ("priority_admission", "random_admission"),
)


def build_comparison_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Compute every preregistered pair with stream seed as the only bootstrap unit."""

    comparisons: list[JsonDict] = []
    for capacity, delay_schedule in CELLS:
        cell_rows = [
            row
            for row in rows
            if row["capacity"] == capacity and row["delay_schedule"] == delay_schedule
        ]
        for arm, control in _PAIRWISE_COMPARISONS:
            for window in ("prospective", "recurrence"):
                for metric in ("error_rate", "false_accept_rate", "abstention_rate"):
                    left = {
                        int(row["seed"]): float(row["window_metrics"][window][metric])
                        for row in cell_rows
                        if row["arm"] == arm
                    }
                    right = {
                        int(row["seed"]): float(row["window_metrics"][window][metric])
                        for row in cell_rows
                        if row["arm"] == control
                    }
                    estimate = paired_seed_bootstrap(
                        left,
                        right,
                        seed=int(
                            sha256_json(
                                [
                                    RANDOM_SEED,
                                    capacity,
                                    delay_schedule,
                                    arm,
                                    control,
                                    window,
                                    metric,
                                ]
                            ).removeprefix("sha256:")[:16],
                            16,
                        ),
                    )
                    comparisons.append(
                        {
                            "capacity": capacity,
                            "delay_schedule": delay_schedule,
                            "arm": arm,
                            "control": control,
                            "window": window,
                            "metric": metric,
                            **estimate,
                            "bootstrap_unit": "stream_seed",
                            "independent_unit_count": len(left),
                            "bootstrap_draws": BOOTSTRAP_DRAWS,
                        }
                    )
    return comparisons


def score_primary_gate(
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    violation_count: int,
) -> tuple[int, int]:
    """Apply the fixed capacity-four burst thresholds without cell selection."""

    primary = [
        row
        for row in comparison_rows
        if row.get("capacity") == PRIMARY_CELL["capacity"]
        and row.get("delay_schedule") == PRIMARY_CELL["delay_schedule"]
        and row.get("arm") == "priority_admission"
    ]

    def row_for(control: str, window: str, metric: str) -> Mapping[str, Any] | None:
        return next(
            (
                row
                for row in primary
                if row.get("control") == control
                and row.get("window") == window
                and row.get("metric") == metric
            ),
            None,
        )

    def high(control: str, window: str, metric: str) -> float:
        row = row_for(control, window, metric)
        return float("inf") if row is None else float(row["ci95_high"])

    safe = violation_count == 0
    beats_frozen_controls = all(
        high(control, "prospective", "error_rate") < 0
        and high(control, "prospective", "false_accept_rate") <= 0
        and high(control, "recurrence", "error_rate") <= 0.02
        for control in ("warmup_frozen", "fifo_admission")
    )
    priority_specific = high("random_admission", "prospective", "error_rate") < 0
    return int(safe and beats_frozen_controls), int(safe and priority_specific)


def terminal_classification(
    *,
    completion_score: int,
    value_score: int,
) -> tuple[str, str, str]:
    """Keep completed null evidence separate from incomplete own work."""

    if completion_score != 1:
        return (
            "complete",
            "disqualified",
            "complete_disqualified: bounded acquisition panel failed its owned contract",
        )
    if value_score == 1:
        return (
            "complete",
            "positive",
            "complete_positive: bounded acquisition passed the frozen primary-cell gate",
        )
    return (
        "complete",
        "null",
        "complete_null: bounded acquisition did not pass the frozen primary-cell gate",
    )


def _task_block(text: str) -> str:
    """Extract only Exp7199 so a different roadmap task cannot satisfy identity."""

    match = re.search(r"(?ms)^- id: exp7199-bounded-acquisition\n(.*?)(?=^- id:|\Z)", text)
    return "" if match is None else match.group(0)


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without writing measurement bytes."""

    candidate = path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.is_dir() and os.access(candidate, os.W_OK)


_QUARANTINE_KEYS = (
    "artifact_quarantined",
    "upstream_quarantined",
    "quarantine_flag",
    "quarantined",
    "excluded_from_use",
    "flagged_adversarial",
)


def _quarantine_state(upstream: Mapping[str, Any], exclusion_text: str) -> JsonDict:
    """Combine explicit artifact flags with the independent exclusion manifest."""

    declared = {key: upstream[key] for key in _QUARANTINE_KEYS if key in upstream}
    markers = (
        "experiment_7198_v634_feedback_capacity_stream.json",
        "exp7198-feedback-capacity-stream",
    )
    manifest_matches = [marker for marker in markers if marker in exclusion_text]
    return {
        "quarantined": any(value is True for value in declared.values()) or bool(manifest_matches),
        "declared_flags": declared,
        "exclusion_manifest_matches": manifest_matches,
    }


def _source_hashes(
    repo_root: Path,
    upstream_artifact_path: Path,
    upstream: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Bind code, tests, contracts, upstream artifact and immutable stream bytes."""

    hashes = {str(path): sha256_path(repo_root / path) for path in SOURCE_PATHS}
    hashes[str(upstream_artifact_path)] = sha256_path(_resolve(repo_root, upstream_artifact_path))
    if upstream:
        for field_name in (
            "public_stream_path",
            "authority_sidecar_path",
            "stream_manifest_path",
        ):
            value = upstream.get(field_name)
            if isinstance(value, str):
                hashes[value] = sha256_path(_resolve(repo_root, value))
    return hashes


def collect_preconditions(
    repo_root: Path,
    upstream_artifact_path: Path,
    paths: ExperimentPaths,
) -> tuple[list[JsonDict], JsonDict]:
    """Check all external gates before any upstream stream row is parsed."""

    upstream = _load_object(_resolve(repo_root, upstream_artifact_path))
    spec_path = repo_root / SPEC_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    roadmap_path = repo_root / "research-roadmap.yaml"
    roadmap_text = roadmap_path.read_text(encoding="utf-8") if roadmap_path.is_file() else ""
    task = _task_block(roadmap_text)
    exclusion_path = repo_root / "ops/exclusion_manifest.yaml"
    exclusion_text = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    quarantine = _quarantine_state(upstream, exclusion_text)
    sizes = {
        str(path): (repo_root / path).stat().st_size if (repo_root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    hashes = _source_hashes(repo_root, upstream_artifact_path, upstream)
    identity = {
        "id": "exp7199-bounded-acquisition" if task else None,
        "milestone": MILESTONE if f"milestone: {MILESTONE}" in task else None,
        "deliverable": (
            str(DEFAULT_ARTIFACT_PATH) if f"deliverable: {DEFAULT_ARTIFACT_PATH}" in task else None
        ),
    }
    expected_identity = {
        "id": "exp7199-bounded-acquisition",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT_PATH),
    }
    stream_hashes = {
        "public_view_hash": sha256_path(
            _resolve(repo_root, str(upstream.get("public_stream_path", "missing")))
        ),
        "authority_sidecar_hash": sha256_path(
            _resolve(repo_root, str(upstream.get("authority_sidecar_path", "missing")))
        ),
        "stream_manifest_hash": sha256_path(
            _resolve(repo_root, str(upstream.get("stream_manifest_path", "missing")))
        ),
    }
    expected_stream_hashes = {
        key: upstream.get(key)
        for key in ("public_view_hash", "authority_sidecar_hash", "stream_manifest_hash")
    }
    prior_null = upstream.get("prior_null_gate_receipt", {})
    expected_prior_null = {"null_observed": 0, "promoted_as_positive": False}
    observed_prior_null = {
        "null_observed": prior_null.get("null_observed"),
        "promoted_as_positive": prior_null.get("promoted_as_positive"),
    }
    tools = {
        "python_executable": Path(sys.executable).is_file(),
        "sha256sum": shutil.which("sha256sum") is not None,
    }
    destinations = {
        "checkpoint_parent": _path_writable(_resolve(repo_root, paths.checkpoint)),
        "artifact_parent": _path_writable(_resolve(repo_root, paths.artifact)),
    }
    expected_manifest = {
        "seeds": list(STREAM_SEEDS),
        "capacities": list(CAPACITIES),
        "delay_schedules": list(DELAY_SCHEDULES),
        "block_size": BLOCK_SIZE,
        "memory_byte_budget": MEMORY_BYTE_BUDGET,
    }
    manifest = upstream.get("stream_manifest", {})
    observed_manifest = {key: manifest.get(key) for key in expected_manifest}
    return [
        gate_check(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-CL-7199",
            True,
            "## REQ-CL-7199:" in spec_text,
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7199-*",
            6,
            spec_text.count("### SCENARIO-CL-7199-"),
            spec_text.count("### SCENARIO-CL-7199-") >= 6,
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            {str(path): "nonempty" for path in SOURCE_PATHS},
            sizes,
            all(isinstance(size, int) and size > 0 for size in sizes.values()),
        ),
        gate_check(
            "required_source_hashes",
            "repository_and_exp7198",
            "SOURCE_PATHS.sha256",
            "sha256:<64 hex> for every source and immutable upstream byte file",
            hashes,
            all(
                isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                for value in hashes.values()
            ),
        ),
        gate_check(
            "v634_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            expected_identity,
            identity,
        ),
        gate_check(
            "upstream_status",
            "exp7198-feedback-capacity-stream",
            "status",
            "complete",
            upstream.get("status"),
        ),
        gate_check(
            "upstream_completion_gate",
            "exp7198-feedback-capacity-stream",
            "stream_capacity_ready_score",
            1,
            upstream.get("stream_capacity_ready_score"),
        ),
        gate_check(
            "upstream_verdict_class",
            "exp7198-feedback-capacity-stream",
            "verdict_class",
            "circular_positive",
            upstream.get("verdict_class"),
        ),
        gate_check(
            "upstream_run_date",
            "exp7198-feedback-capacity-stream",
            "run_date",
            RUN_DATE,
            upstream.get("run_date"),
        ),
        gate_check(
            "upstream_no_model_replay",
            "exp7198-feedback-capacity-stream",
            "MODEL_SPECS,model_invoked",
            {"MODEL_SPECS": [], "model_invoked": False},
            {
                "MODEL_SPECS": upstream.get("MODEL_SPECS"),
                "model_invoked": upstream.get("model_invoked"),
            },
        ),
        gate_check(
            "upstream_gate_summary",
            "exp7198-feedback-capacity-stream",
            "gate_check_summary.passed",
            True,
            upstream.get("gate_check_summary", {}).get("passed"),
        ),
        gate_check(
            "upstream_not_quarantined",
            "artifact_metadata_and_ops/exclusion_manifest.yaml",
            "quarantined",
            False,
            quarantine["quarantined"],
        ),
        gate_check(
            "upstream_quarantine_receipt",
            "artifact_metadata_and_ops/exclusion_manifest.yaml",
            "declared_flags,exclusion_manifest_matches",
            {"declared_flags": {}, "exclusion_manifest_matches": []},
            {
                "declared_flags": quarantine["declared_flags"],
                "exclusion_manifest_matches": quarantine["exclusion_manifest_matches"],
            },
        ),
        gate_check(
            "known_prior_null_not_promoted",
            "exp7198.prior_null_gate_receipt",
            "null_observed,promoted_as_positive",
            expected_prior_null,
            observed_prior_null,
        ),
        gate_check(
            "upstream_stream_contract",
            "exp7198.stream_manifest",
            "seeds,capacities,delay_schedules,block_size,memory_byte_budget",
            expected_manifest,
            observed_manifest,
        ),
        gate_check(
            "upstream_immutable_byte_hashes",
            "exp7198 sealed files",
            "public_view_hash,authority_sidecar_hash,stream_manifest_hash",
            expected_stream_hashes,
            stream_hashes,
        ),
        gate_check(
            "required_local_tools",
            "host",
            "python,sha256sum",
            {key: True for key in tools},
            tools,
        ),
        gate_check(
            "output_destinations_writable",
            "host_filesystem",
            "checkpoint,artifact",
            {key: True for key in destinations},
            destinations,
        ),
    ], upstream


def _source_grounding_rows(
    views: exp7198.StreamViews,
    seeds: Sequence[int],
) -> list[JsonDict]:
    """Run public extraction through two exact implementations before scoring."""

    public = {str(row["event_id"]): row for row in views.public_events}
    rows: list[JsonDict] = []
    for seed in seeds:
        truths = [row for row in views.authority_events if int(row["seed"]) == seed]
        agreement = 0
        for truth in truths:
            parsed = exp7198.extract_public_input(public[str(truth["event_id"])]["public_input"])
            first = exp7198.exact_label(
                str(parsed["family_id"]),
                int(parsed["numeric_value"]),
                int(truth["hidden_parameter"]),
            )
            second = exp7198.independent_exact_label(
                str(parsed["family_id"]),
                int(parsed["numeric_value"]),
                int(truth["hidden_parameter"]),
            )
            agreement += int(first == second == truth["exact_label"])
        rows.append(
            {
                "seed": seed,
                "public_input_count": len(truths),
                "extraction_count": len(truths),
                "execution_count": len(truths),
                "independent_score_count": len(truths),
                "agreement_count": agreement,
                "passed": agreement == len(truths),
            }
        )
    return rows


def _acceptance_gate() -> JsonDict:
    """Expose every threshold before any comparison can be interpreted."""

    return {
        "primary_cell": deepcopy(PRIMARY_CELL),
        "prospective_error_ci95_high_against_warmup_frozen": "<0",
        "prospective_error_ci95_high_against_fifo": "<0",
        "prospective_false_accept_ci95_high_against_both_controls": "<=0",
        "recurrence_error_ci95_high_against_both_controls": "<=0.02",
        "memory_capacity_violation_count": 0,
        "priority_specific_prospective_error_ci95_high_against_random": "<0",
        "bootstrap_unit": "stream_seed",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "cell_selection_after_outcomes": False,
    }


def _prediction_contract() -> JsonDict:
    """Expose the exact deployed vote and scoring rules."""

    return {
        "hypothesis_domain": list(PARAMETER_DOMAIN),
        "prediction": "majority_vote_over_surviving_hypotheses",
        "tie_action": "reject",
        "empty_set_action": "abstain",
        "abstention_scoring": "error_on_full_denominator",
        "support_eliminates_hypotheses": True,
        "validation_eliminates_hypotheses": False,
        "support_required_before_singleton_freeze": SUPPORT_REQUIRED,
        "post_freeze_validation_required_before_commit": VALIDATION_REQUIRED,
    }


def _future_hardware_path() -> JsonDict:
    """Describe a concrete acceleration path without inventing measurements."""

    return {
        "current": "CPU bitset intersections and bounded counters",
        "future": "FPGA lookup and bitset logic",
        "lookup_target_ms": 1.0,
        "unmeasured_speedup_claimed": False,
        "claimed_speedup": None,
    }


def _empty_evidence() -> JsonDict:
    """Keep blocked artifacts schema-complete without invented measurements."""

    return {
        "rows": [],
        "decision_rows": [],
        "update_rows": [],
        "pending_queue_rows": [],
        "validation_access_rows": [],
        "validation_partition_rows": [],
        "comparison_rows": [],
        "source_grounding_rows": [],
    }


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    upstream: Mapping[str, Any],
    *,
    repo_root: Path,
    upstream_artifact_path: Path,
    paths: ExperimentPaths,
    run_date: str,
    duration_s: float,
) -> JsonDict:
    """Build shared provenance before any completion or value claim."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "blocked",
        "run_date": run_date,
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": "blocked before qualifying computation",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "source_artifact_hashes": _source_hashes(repo_root, upstream_artifact_path, upstream),
        "sample_size_budget": {
            "planned_events": len(STREAM_SEEDS) * EVENTS_PER_SEED,
            "completed_events": 0,
            "independent_stream_units_planned": len(STREAM_SEEDS),
            "independent_stream_units_completed": 0,
            "capacity_delay_cells_planned": len(CELLS),
            "capacity_delay_cells_completed": 0,
            "terminal_units_planned": len(STREAM_SEEDS) * len(CELLS) * len(ARMS),
            "terminal_units_completed": 0,
            "exclusions": [],
        },
        "random_seed": {
            "master": RANDOM_SEED,
            "stream_seeds": list(STREAM_SEEDS),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "selection_derivation": "sha256 canonical public tuples",
        },
        "reproducibility_checksum": None,
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_external: bounded acquisition did not run",
        "acquisition_run_complete_score": 0,
        "acquisition_value_score": 0,
        "continuous_self_learning_task": False,
        "acceptance_gate_learning": _acceptance_gate(),
        "future_hardware_path": _future_hardware_path(),
        "no_model_weight_mutation": True,
        "prediction_contract": _prediction_contract(),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "latency_summary": {
            "lookup_p50_ns": None,
            "lookup_p95_ns": None,
            "update_p50_ns": None,
            "update_p95_ns": None,
            "lookup_target_ms": 1.0,
            "lookup_target_met": None,
        },
        "benefit_decomposition": {
            "version_space_acquisition_benefit": None,
            "template_commit_accuracy_benefit": None,
            "commit_is_persistence_only": True,
        },
        "priority_specific_benefit_score": 0,
        "memory_capacity_violation_count": 0,
        "checkpoint_path": str(paths.checkpoint),
        "checkpoint_hash": None,
        "upstream_receipt": {
            "artifact_path": str(upstream_artifact_path),
            "artifact_hash": sha256_path(_resolve(repo_root, upstream_artifact_path)),
            "completion_field": "stream_capacity_ready_score",
            "completion_expected": 1,
            "completion_observed": upstream.get("stream_capacity_ready_score"),
            "known_prior_null_field": "prior_null_gate_receipt.null_observed",
            "known_prior_null_observed": upstream.get("prior_null_gate_receipt", {}).get(
                "null_observed"
            ),
            "known_prior_null_promoted": False,
            "public_view_hash": upstream.get("public_view_hash"),
            "authority_sidecar_hash": upstream.get("authority_sidecar_hash"),
            "stream_manifest_hash": upstream.get("stream_manifest_hash"),
        },
        **_empty_evidence(),
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    upstream: Mapping[str, Any],
    *,
    repo_root: Path,
    upstream_artifact_path: Path,
    paths: ExperimentPaths,
    run_date: str,
    duration_s: float,
) -> JsonDict:
    """Return a terminal row-free result for an external prerequisite failure."""

    artifact = _base_artifact(
        checks,
        upstream,
        repo_root=repo_root,
        upstream_artifact_path=upstream_artifact_path,
        paths=paths,
        run_date=run_date,
        duration_s=duration_s,
    )
    failed = artifact["gate_check_summary"]["failed_check"] or "unknown_precondition"
    artifact["honest_verdict"] = f"blocked_external:{failed}"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


_TIMING_KEYS = {
    "lookup_ns",
    "update_ns",
    "selection_ns",
    "storage_ns",
    "lookup_p50_ns",
    "lookup_p95_ns",
    "update_p50_ns",
    "update_p95_ns",
    "selection_p50_ns",
    "selection_p95_ns",
    "storage_p50_ns",
    "storage_p95_ns",
    "end_to_end_cost_ns",
}


def _stable_timing_value(value: Any) -> Any:
    """Remove host timing while retaining all scientific decisions and counters."""

    if isinstance(value, Mapping):
        return {
            key: _stable_timing_value(item)
            for key, item in value.items()
            if key not in _TIMING_KEYS
        }
    if isinstance(value, list):
        return [_stable_timing_value(item) for item in value]
    return value


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash source identities, seeds, contracts and all timing-free raw rows."""

    digest = hashlib.sha256()
    digest.update(b"carnot.exp7199.reproducibility.v1\n")
    scalar_fields = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "source_artifact_hashes",
        "random_seed",
        "gate_check_summary",
        "acceptance_gate_learning",
        "prediction_contract",
        "upstream_receipt",
        "sample_size_budget",
        "benefit_decomposition",
        "acquisition_run_complete_score",
        "acquisition_value_score",
        "priority_specific_benefit_score",
        "memory_capacity_violation_count",
        "verdict_class",
        "honest_verdict",
    )
    row_fields = (
        "rows",
        "decision_rows",
        "update_rows",
        "pending_queue_rows",
        "validation_access_rows",
        "validation_partition_rows",
        "comparison_rows",
        "source_grounding_rows",
    )
    for name in scalar_fields:
        digest.update(canonical_json([name, _stable_timing_value(artifact.get(name))]))
        digest.update(b"\n")
    for name in row_fields:
        digest.update(canonical_json([name, len(artifact.get(name, []))]))
        digest.update(b"\n")
        for row in artifact.get(name, []):
            digest.update(canonical_json(_stable_timing_value(row)))
            digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def _artifact_panel(artifact: Mapping[str, Any]) -> AcquisitionPanel:
    """Project stored raw evidence into the panel conformance checker."""

    return AcquisitionPanel(
        list(artifact["rows"]),
        list(artifact["decision_rows"]),
        list(artifact["update_rows"]),
        list(artifact["pending_queue_rows"]),
        list(artifact["validation_access_rows"]),
        list(artifact["validation_partition_rows"]),
    )


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    check_files: bool = False,
) -> list[str]:
    """Cold-check schema, raw chronology, gates, source bytes and checksum."""

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
    add(artifact["MODEL_SPECS"] != [], "model_specs")
    add(artifact["model_invoked"] is not False, "model_invoked")
    add(artifact["no_model_weight_mutation"] is not True, "weight_mutation")
    add(artifact["verifier_is_oracle"] is not True, "verifier_authority")
    add(artifact["acceptance_gate_learning"] != _acceptance_gate(), "acceptance_gate")
    add(artifact["prediction_contract"] != _prediction_contract(), "prediction_contract")
    blocked = artifact["verdict_class"] == "blocked"
    if blocked:
        add(artifact["status"] != "blocked", "blocked_status")
        add(artifact["inference_substrate_class"] != "blocked_no_run", "blocked_substrate")
        add(artifact["acquisition_run_complete_score"] != 0, "blocked_completion")
        add(artifact["acquisition_value_score"] != 0, "blocked_value")
        add(artifact["continuous_self_learning_task"] is not False, "blocked_learning_claim")
        add(any(artifact[field] for field in _empty_evidence()), "blocked_rows")
        summary = artifact["gate_check_summary"]
        add(summary.get("passed") is not False, "blocked_gate_passed")
        add(not summary.get("failed_check"), "blocked_failed_check")
        add(not summary.get("upstream"), "blocked_failed_upstream")
        add(not summary.get("field"), "blocked_failed_field")
    else:
        add(artifact["status"] != "complete", "status")
        add(artifact["inference_substrate"] != INFERENCE_SUBSTRATE, "inference_substrate")
        add(
            artifact["inference_substrate_class"] != INFERENCE_SUBSTRATE_CLASS,
            "inference_substrate_class",
        )
        add(artifact["execution_venue"] != EXECUTION_VENUE, "execution_venue")
        add(artifact["gate_check_summary"].get("passed") is not True, "precondition_gate")
        panel = _artifact_panel(artifact)
        conformance = panel_conformance_errors(panel)
        add(bool(conformance), "panel_conformance:" + ",".join(conformance))
        violation_count = sum(
            int(row["max_pending"] > row["capacity"])
            + int(row["max_memory_bytes"] > MEMORY_BYTE_BUDGET)
            + int(row["pending_eviction_count"] != 0)
            for row in artifact["rows"]
        )
        add(
            artifact["memory_capacity_violation_count"] != violation_count,
            "violation_count",
        )
        value, priority = score_primary_gate(
            artifact["comparison_rows"], violation_count=violation_count
        )
        add(artifact["acquisition_value_score"] != value, "value_score")
        add(artifact["priority_specific_benefit_score"] != priority, "priority_score")
        expected_complete = int(not conformance and len(artifact["rows"]) == 600)
        add(artifact["acquisition_run_complete_score"] != expected_complete, "completion_score")
        expected_status, expected_class, expected_honest = terminal_classification(
            completion_score=expected_complete,
            value_score=value,
        )
        add(artifact["status"] != expected_status, "terminal_status")
        add(artifact["verdict_class"] != expected_class, "terminal_class")
        add(artifact["honest_verdict"] != expected_honest, "terminal_honest_verdict")
        add(not artifact["update_rows"], "missing_updates")
        add(artifact["continuous_self_learning_task"] is not True, "learning_claim")
        add(
            any(row.get("passed") is not True for row in artifact["source_grounding_rows"]),
            "source_grounding",
        )
        add(
            artifact["latency_summary"].get("lookup_p95_ns", 1_000_001) >= 1_000_000,
            "lookup_target",
        )
        if check_files:
            root = repo_root or Path(__file__).resolve().parents[2]
            upstream_path = Path(str(artifact["upstream_receipt"]["artifact_path"]))
            add(
                artifact["source_artifact_hashes"]
                != _source_hashes(root, upstream_path, _load_object(_resolve(root, upstream_path))),
                "source_hashes",
            )
            checkpoint_path = _resolve(root, str(artifact["checkpoint_path"]))
            add(sha256_path(checkpoint_path) != artifact["checkpoint_hash"], "checkpoint_hash")
    add(
        artifact["reproducibility_checksum"] != reproducibility_checksum(artifact),
        "reproducibility_checksum",
    )
    return errors


def _latency_summary(panel: AcquisitionPanel) -> JsonDict:
    """Reduce measured lookup and update costs across all executed rows."""

    lookup = [int(row["lookup_ns"]) for row in panel.decision_rows]
    updates = [int(row["update_ns"]) for row in panel.update_rows]
    lookup_p50 = _percentile(lookup, 0.50)
    lookup_p95 = _percentile(lookup, 0.95)
    return {
        "lookup_p50_ns": lookup_p50,
        "lookup_p95_ns": lookup_p95,
        "update_p50_ns": _percentile(updates, 0.50),
        "update_p95_ns": _percentile(updates, 0.95),
        "lookup_target_ms": 1.0,
        "lookup_target_met": lookup_p95 < 1_000_000,
    }


def _primary_difference(
    comparisons: Sequence[Mapping[str, Any]],
    control: str,
) -> float | None:
    """Return the preregistered primary error point estimate for one control."""

    row = next(
        (
            item
            for item in comparisons
            if item["capacity"] == 4
            and item["delay_schedule"] == "burst"
            and item["arm"] == "priority_admission"
            and item["control"] == control
            and item["window"] == "prospective"
            and item["metric"] == "error_rate"
        ),
        None,
    )
    return None if row is None else float(row["difference"])


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    upstream_artifact_path: Path = DEFAULT_UPSTREAM_ARTIFACT_PATH,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    progress: bool = False,
) -> JsonDict:
    """Run preflight, exact replay, seed bootstrap and cold artifact validation."""

    started = time.monotonic()
    if progress:
        print(
            "PHASE 0 START: verify spec, sources, exact upstream gates, quarantine, tools, and paths",
            flush=True,
        )
    checks, upstream = collect_preconditions(repo_root, upstream_artifact_path, paths)
    if not all(row["passed"] for row in checks):
        elapsed = time.monotonic() - started if duration_s is None else duration_s
        if progress:
            print(
                "PHASE 0 END: external gate failed; building terminal blocked artifact", flush=True
            )
        return build_blocked_artifact(
            checks,
            upstream,
            repo_root=repo_root,
            upstream_artifact_path=upstream_artifact_path,
            paths=paths,
            run_date=run_date,
            duration_s=elapsed,
        )
    if progress:
        print("PHASE 0 END: all external gates passed and prior null stayed unpromoted", flush=True)
        print(
            "PHASE 1 START: load immutable public, authority, and stream-manifest bytes", flush=True
        )
    views = load_upstream_views(repo_root, upstream_artifact_path)
    if progress:
        print("PHASE 1 END: sealed Exp7198 stream bytes passed conformance", flush=True)
        print(
            "PHASE 2 START: run public extraction, execution, and independent scoring", flush=True
        )
    grounding_rows = _source_grounding_rows(views, STREAM_SEEDS)
    if progress:
        print("PHASE 2 END: all 10,240 source rows agree", flush=True)
        print("PHASE 3 START: benchmark all five arms, ten seeds, and twelve cells", flush=True)
    checkpoint_path = _resolve(repo_root, paths.checkpoint)
    panel = run_acquisition_panel(
        views,
        checkpoint_path=checkpoint_path,
        progress=progress,
    )
    if progress:
        print("PHASE 3 END: all 600 terminal units and event ledgers completed", flush=True)
        print(
            "PHASE 4 START: bootstrap all frozen prospective and recurrence comparisons", flush=True
        )
    comparisons = build_comparison_rows(panel.rows)
    if progress:
        print("PHASE 4 END: paired intervals use stream seed as the independent unit", flush=True)
        print(
            "PHASE 5 START: derive bounds, latency, benefit decomposition, and verdict", flush=True
        )
    conformance = panel_conformance_errors(panel)
    violation_count = sum(
        int(row["max_pending"] > row["capacity"])
        + int(row["max_memory_bytes"] > MEMORY_BYTE_BUDGET)
        + int(row["pending_eviction_count"] != 0)
        for row in panel.rows
    )
    value_score, priority_score = score_primary_gate(
        comparisons,
        violation_count=violation_count,
    )
    completion_score = int(not conformance and len(panel.rows) == 600)
    status, verdict_class, honest_verdict = terminal_classification(
        completion_score=completion_score,
        value_score=value_score,
    )
    elapsed = time.monotonic() - started if duration_s is None else duration_s
    artifact = _base_artifact(
        checks,
        upstream,
        repo_root=repo_root,
        upstream_artifact_path=upstream_artifact_path,
        paths=paths,
        run_date=run_date,
        duration_s=elapsed,
    )
    primary_vs_frozen = _primary_difference(comparisons, "warmup_frozen")
    artifact.update(
        {
            "status": status,
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "rows": panel.rows,
            "decision_rows": panel.decision_rows,
            "update_rows": panel.update_rows,
            "pending_queue_rows": panel.pending_queue_rows,
            "validation_access_rows": panel.validation_access_rows,
            "validation_partition_rows": panel.validation_partition_rows,
            "comparison_rows": comparisons,
            "source_grounding_rows": grounding_rows,
            "sample_size_budget": {
                "planned_events": len(STREAM_SEEDS) * EVENTS_PER_SEED,
                "completed_events": len(STREAM_SEEDS) * EVENTS_PER_SEED,
                "materialized_arm_event_rows": len(panel.decision_rows),
                "independent_stream_units_planned": len(STREAM_SEEDS),
                "independent_stream_units_completed": len(STREAM_SEEDS),
                "capacity_delay_cells_planned": len(CELLS),
                "capacity_delay_cells_completed": len(CELLS),
                "terminal_units_planned": 600,
                "terminal_units_completed": len(panel.rows),
                "exclusions": [],
            },
            "verdict_class": verdict_class,
            "honest_verdict": honest_verdict,
            "acquisition_run_complete_score": completion_score,
            "acquisition_value_score": value_score,
            "priority_specific_benefit_score": priority_score,
            "continuous_self_learning_task": bool(panel.update_rows),
            "latency_summary": _latency_summary(panel),
            "benefit_decomposition": {
                "version_space_acquisition_benefit": primary_vs_frozen,
                "template_commit_accuracy_benefit": 0.0,
                "commit_is_persistence_only": True,
                "committed_singleton_prediction_change_count": sum(
                    int(
                        row["operation"] == "commit_template"
                        and row["prediction_changed_by_commit"] is True
                    )
                    for row in panel.update_rows
                ),
            },
            "memory_capacity_violation_count": violation_count,
            "checkpoint_hash": sha256_path(checkpoint_path),
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact, repo_root=repo_root, check_files=True)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    if progress:
        print(
            "PHASE 5 END: completion and frozen value gates are independently classified",
            flush=True,
        )
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed execution date and optional private output root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--upstream-artifact-path",
        type=Path,
        default=DEFAULT_UPSTREAM_ARTIFACT_PATH,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run real file gates and atomically publish one terminal artifact."""

    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    artifact = build_and_seal(
        repo_root,
        paths,
        upstream_artifact_path=args.upstream_artifact_path,
        run_date=str(args.date),
        progress=True,
    )
    print("PHASE 6 START: cold-check principles, rows, gates, hashes, and verdict", flush=True)
    errors = validate_artifact(
        artifact,
        repo_root=repo_root,
        check_files=artifact["verdict_class"] != "blocked",
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    print("PHASE 6 END: terminal evidence passed cold validation", flush=True)
    print("PHASE 7 START: atomically write the terminal deliverable", flush=True)
    print("FINAL ATOMIC WRITE START", flush=True)
    write_json_atomic(_resolve(repo_root, paths.artifact), artifact)
    print("FINAL ATOMIC WRITE END", flush=True)
    print("PHASE 7 END: terminal artifact is stable", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper owns this path.
    raise SystemExit(main())
