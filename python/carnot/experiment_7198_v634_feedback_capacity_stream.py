"""Build the V634 stream for bounded delayed-feedback acquisition.

The public file contains only data that a deployable scheduler can inspect.
The authority sidecar keeps parameters, labels, changes, poison markers, and
delays. This module measures fixture readiness, not learning benefit.

Spec refs: REQ-CL-7198 and SCENARIO-CL-7198-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
from typing import Any

from carnot import experiment_7183_v633_supersession_stream as exp7183
from carnot import experiment_7184_v633_revocable_template_csl as exp7184


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7198
SCHEMA = "carnot.exp7198.v634_feedback_capacity_stream.v1"
MILESTONE = "2026.09.634"
RUN_DATE = "20260911"
RANDOM_SEED = 7_198_202_609_11
STREAM_SEEDS = (
    71_980_011,
    71_980_019,
    71_980_031,
    71_980_043,
    71_980_057,
    71_980_069,
    71_980_081,
    71_980_093,
    71_980_107,
    71_980_121,
)
EVENTS_PER_SEED = 1_024
WARMUP_COUNT = 128
WARMUP_REQUEST_STOP = 96
PARAMETER_DOMAIN = tuple(range(33))
FAMILIES = ("lower_bound", "upper_bound", "modular_equals", "cyclic_window")
WINDOWS: tuple[JsonDict, ...] = (
    {"name": "warmup", "start": 0, "stop": 128, "count": 128, "regime": "stable"},
    {
        "name": "online_validation",
        "start": 128,
        "stop": 256,
        "count": 128,
        "regime": "stable",
    },
    {
        "name": "prospective",
        "start": 256,
        "stop": 768,
        "count": 512,
        "regime": "shifted",
    },
    {
        "name": "recurrence",
        "start": 768,
        "stop": 896,
        "count": 128,
        "regime": "recurrent",
    },
    {
        "name": "poison_rollback",
        "start": 896,
        "stop": 1_024,
        "count": 128,
        "regime": "shifted_after_recurrence",
    },
)
CAPACITIES = (1, 4, 16)
DELAY_SCHEDULES = ("constant_0", "constant_4", "constant_16", "burst")
ARMS = ("static_frozen", "random_admission", "disagreement_admission")
MEMORY_BYTE_BUDGET = 64 * 1_024
BLOCK_SIZE = 4
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = (
    "deterministic CPU finite-domain predicates, bounded pending-queue scheduling, "
    "and exact replay; no model invocation"
)
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

DEFAULT_PUBLIC_STREAM_PATH = Path("results/streams/experiment_7198_v634_public_stream.jsonl")
DEFAULT_AUTHORITY_SIDECAR_PATH = Path(
    "results/streams/experiment_7198_v634_authority_sidecar.jsonl"
)
DEFAULT_MANIFEST_PATH = Path("results/streams/experiment_7198_v634_stream_manifest.json")
DEFAULT_CHECKPOINT_PATH = Path(
    "results/checkpoints/experiment_7198_v634_feedback_capacity_stream_progress.json"
)
DEFAULT_ARTIFACT_PATH = Path("results/experiment_7198_v634_feedback_capacity_stream.json")
DEFAULT_UPSTREAM_ARTIFACT_PATH = Path("results/experiment_7184_v633_revocable_template_csl.json")
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
    Path("python/carnot/experiment_7183_v633_supersession_stream.py"),
    Path("python/carnot/experiment_7184_v633_revocable_template_csl.py"),
    Path("python/carnot/experiment_7198_v634_feedback_capacity_stream.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("scripts/experiments/experiment_7198_v634_feedback_capacity_stream.py"),
    Path("tests/python/test_experiment_7198_v634_feedback_capacity_stream.py"),
    SPEC_PATH,
)

FAMILY_GRAMMAR: JsonDict = {
    "grammar_version": "finite_numeric_predicate.v1",
    "parameter_domain": [0, 32],
    "families": {
        "lower_bound": "accept iff numeric_value mod 33 is at least hidden_parameter",
        "upper_bound": "accept iff numeric_value mod 33 is at most hidden_parameter",
        "modular_equals": "accept iff numeric_value mod 33 equals hidden_parameter",
        "cyclic_window": ("accept iff (numeric_value mod 33 - hidden_parameter) mod 33 is below 8"),
    },
}

FORBIDDEN_PUBLIC_FIELDS = {
    "parameter",
    "parameters",
    "hidden_parameter",
    "label",
    "exact_label",
    "independent_exact_label",
    "observed_label",
    "future_label",
    "regime",
    "change_time",
    "change_times",
    "delay",
    "delays",
    "delay_by_schedule",
    "release_index",
    "future_release",
    "poisoned",
    "poison_state",
    "authority",
}

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
    "stream_capacity_ready_score",
    "stream_manifest",
    "public_stream_path",
    "authority_sidecar_path",
    "information_budget_rows",
    "headroom_rows",
    "MODEL_SPECS",
    "model_invoked",
    "public_view_hash",
    "authority_sidecar_hash",
    "stream_manifest_path",
    "stream_manifest_hash",
    "capacity_delay_contract",
    "warmup_state_rows",
    "source_grounding_rows",
    "mutation_audit_rows",
    "prior_null_gate_receipt",
    "checkpoint_path",
    "checkpoint_hash",
    "pending_queue_rows",
    "panel_hash",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "A versioned schema makes incompatible readers fail closed.",
    "experiment_id": "A fixed ID prevents another task from supplying this evidence.",
    "milestone": "The milestone binds the stream to the V634 contract.",
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
    "stream_capacity_ready_score": (
        "Readiness certifies information and timing contracts, not learnability."
    ),
    "stream_manifest": "Freeze all ten seeds, 1024 events and delay/capacity cells.",
    "public_stream_path": "Learner-visible fields omit hidden parameters and future labels.",
    "authority_sidecar_path": "The scorer alone holds full truth and feedback timing.",
    "information_budget_rows": "The frozen and adaptive controls start with equal information.",
    "headroom_rows": "A zero-error oracle is an upper bound, not an attainable accuracy target.",
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is cited, not repeated.",
    "public_view_hash": "The public-view digest detects post-freeze learner-input changes.",
    "authority_sidecar_hash": "The authority digest binds hidden truth and timing.",
    "stream_manifest_path": "A separate manifest lets later tasks inspect frozen contracts.",
    "stream_manifest_hash": "The manifest digest exposes outcome-driven contract edits.",
    "capacity_delay_contract": "Explicit queue rules make every capacity cell replayable.",
    "warmup_state_rows": "State hashes prove equal information at the freeze boundary.",
    "source_grounding_rows": "Extraction and two exact scorers bind public input to truth.",
    "mutation_audit_rows": "Named attacks must fail before readiness can become one.",
    "prior_null_gate_receipt": "The old zero-value result stays a null input, not headroom.",
    "checkpoint_path": "Real progress is stored outside the terminal deliverable.",
    "checkpoint_hash": "The checkpoint digest binds completed computation units.",
    "pending_queue_rows": "Boundary receipts expose admission, occupancy, and release order.",
    "panel_hash": "One digest binds all scheduler comparisons and budget receipts.",
}

canonical_json = exp7183.canonical_json
sha256_bytes = exp7183.sha256_bytes
sha256_json = exp7183.sha256_json
sha256_path = exp7183.sha256_path
write_immutable_jsonl = exp7183.write_immutable_jsonl
write_immutable_json = exp7183.write_immutable_json
write_json_atomic = exp7183.write_json_atomic
ImmutableSealError = exp7183.ImmutableSealError
gate_check = exp7183.gate_check
gate_summary = exp7183.gate_summary


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep stream, authority, checkpoint, and terminal paths distinct."""

    public_stream: Path
    authority_sidecar: Path
    manifest: Path
    checkpoint: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the repository paths required by the public command."""

        return cls(
            DEFAULT_PUBLIC_STREAM_PATH,
            DEFAULT_AUTHORITY_SIDECAR_PATH,
            DEFAULT_MANIFEST_PATH,
            DEFAULT_CHECKPOINT_PATH,
            DEFAULT_ARTIFACT_PATH,
        )

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put private test evidence under one caller-owned directory."""

        return cls(
            root / "streams" / "public_stream.jsonl",
            root / "streams" / "authority_sidecar.jsonl",
            root / "streams" / "stream_manifest.json",
            root / "checkpoints" / "progress.json",
            root / "experiment_7198_v634_feedback_capacity_stream.json",
        )


@dataclass(frozen=True)
class StreamViews:
    """Hold public and authority rows separately before either file is sealed."""

    public_events: list[JsonDict]
    authority_events: list[JsonDict]
    manifest: JsonDict


@dataclass(frozen=True)
class PendingRecord:
    """Keep hidden delivery data outside the learner-visible scheduler input."""

    event_id: str
    family_id: str
    numeric_value: int
    request_index: int
    release_index: int
    observed_label: str
    exact_label: str
    poisoned: bool


@dataclass(frozen=True)
class CapacityPanel:
    """Keep row panels explicit so cold validators can recompute their digest."""

    rows: list[JsonDict]
    information_budget_rows: list[JsonDict]
    warmup_state_rows: list[JsonDict]
    pending_queue_rows: list[JsonDict]
    headroom_rows: list[JsonDict]


def _stable_seed(*parts: Any) -> int:
    """Derive local randomness without depending on Python's salted object hash."""

    return int(sha256_json(list(parts)).removeprefix("sha256:")[:16], 16)


def _window_for(index: int) -> Mapping[str, Any]:
    """Return the one evaluator window that owns a chronology index."""

    return next(window for window in WINDOWS if window["start"] <= index < window["stop"])


def _parameters_for(seed: int, family_id: str) -> tuple[int, int]:
    """Freeze stable and shifted parameters before any outcome is generated."""

    family_index = FAMILIES.index(family_id)
    stable = 4 + _stable_seed("parameter", seed, family_id) % 25
    shifted = (stable + 7 + family_index * 3) % len(PARAMETER_DOMAIN)
    if shifted == stable:  # pragma: no cover - arithmetic prevents this defensive case.
        shifted = (shifted + 1) % len(PARAMETER_DOMAIN)
    return stable, shifted


def exact_label(family_id: str, numeric_value: int, parameter: int) -> str:
    """Execute one finite public predicate with its evaluator-only parameter."""

    residue = numeric_value % 33
    if family_id == "lower_bound":
        accepted = residue >= parameter
    elif family_id == "upper_bound":
        accepted = residue <= parameter
    elif family_id == "modular_equals":
        accepted = residue == parameter
    elif family_id == "cyclic_window":
        accepted = (residue - parameter) % 33 < 8
    else:
        raise ValueError(f"unknown_family:{family_id}")
    return "accept" if accepted else "reject"


def independent_exact_label(family_id: str, numeric_value: int, parameter: int) -> str:
    """Recompute the same label with alternate integer expressions."""

    _, residue = divmod(numeric_value, 33)
    if family_id == "lower_bound":
        accepted = not residue < parameter
    elif family_id == "upper_bound":
        accepted = not residue > parameter
    elif family_id == "modular_equals":
        accepted = (residue - parameter) % 33 == 0
    elif family_id == "cyclic_window":
        accepted = any(residue == (parameter + offset) % 33 for offset in range(8))
    else:
        raise ValueError(f"unknown_family:{family_id}")
    return "accept" if accepted else "reject"


_PUBLIC_INPUT = re.compile(
    r"^family=(lower_bound|upper_bound|modular_equals|cyclic_window);value=(-?[0-9]+)$"
)


def extract_public_input(value: str) -> JsonDict:
    """Parse the fixed public text without adding authority-side information."""

    match = _PUBLIC_INPUT.fullmatch(value)
    if match is None:
        raise ValueError("invalid_public_input")
    return {"family_id": match.group(1), "numeric_value": int(match.group(2))}


def _burst_delay(seed: int, chronology_index: int) -> int:
    """Return one frozen burst delay that no admission callback receives."""

    return (0, 4, 16, 32)[_stable_seed("burst", seed, chronology_index) % 4]


def _is_poisoned(seed: int, chronology_index: int) -> bool:
    """Select sixteen poison rows per seed with a fixed evaluator-only shuffle."""

    indices = list(range(896, 1_024))
    random.Random(_stable_seed("poison", seed)).shuffle(indices)
    return chronology_index in set(indices[:16])


def _stream_manifest() -> JsonDict:
    """Return the complete frozen contract without reading generated outcomes."""

    cells = [
        {"capacity": capacity, "delay_schedule": delay}
        for capacity in CAPACITIES
        for delay in DELAY_SCHEDULES
    ]
    return {
        "schema": "carnot.exp7198.stream_manifest.v1",
        "frozen_before_learner": True,
        "seeds": list(STREAM_SEEDS),
        "events_per_seed": EVENTS_PER_SEED,
        "total_events": len(STREAM_SEEDS) * EVENTS_PER_SEED,
        "windows": deepcopy(list(WINDOWS)),
        "families": list(FAMILIES),
        "family_grammar": deepcopy(FAMILY_GRAMMAR),
        "parameter_domain": list(PARAMETER_DOMAIN),
        "capacities": list(CAPACITIES),
        "delay_schedules": list(DELAY_SCHEDULES),
        "capacity_delay_cells": cells,
        "arms": list(ARMS),
        "block_size": BLOCK_SIZE,
        "request_limit_per_block": 1,
        "memory_byte_budget": MEMORY_BYTE_BUDGET,
        "pending_policy": "drop_new_never_evict_pending",
        "release_capacity_rule": "available_at_next_four_event_boundary",
        "oracle": {
            "arm_status": "unattainable_upper_bound",
            "learner_visible": False,
            "full_information": True,
        },
    }


def build_stream_views() -> StreamViews:
    """Generate all fixed data before any scheduler can select a request."""

    public_events: list[JsonDict] = []
    authority_events: list[JsonDict] = []
    for seed in STREAM_SEEDS:
        family_orders: dict[str, list[str]] = {}
        for window in WINDOWS:
            order = list(FAMILIES) * (int(window["count"]) // len(FAMILIES))
            random.Random(_stable_seed("family_order", seed, window["name"])).shuffle(order)
            family_orders[str(window["name"])] = order
        window_offsets = {str(window["name"]): 0 for window in WINDOWS}
        for chronology_index in range(EVENTS_PER_SEED):
            window = _window_for(chronology_index)
            window_name = str(window["name"])
            offset = window_offsets[window_name]
            window_offsets[window_name] += 1
            family_id = family_orders[window_name][offset]
            family_index = FAMILIES.index(family_id)
            numeric_value = (seed * 17 + chronology_index * 19 + family_index * 7) % 10_000
            stable_parameter, shifted_parameter = _parameters_for(seed, family_id)
            regime = str(window["regime"])
            hidden_parameter = (
                stable_parameter if regime in {"stable", "recurrent"} else shifted_parameter
            )
            event_id = f"exp7198-{seed}-{chronology_index:04d}"
            label = exact_label(family_id, numeric_value, hidden_parameter)
            independent = independent_exact_label(family_id, numeric_value, hidden_parameter)
            poisoned = (
                _is_poisoned(seed, chronology_index) if window_name == "poison_rollback" else False
            )
            observed_label = label
            if poisoned:
                observed_label = "reject" if label == "accept" else "accept"
            public_events.append(
                {
                    "event_id": event_id,
                    "seed": seed,
                    "chronology_index": chronology_index,
                    "entity_id": f"entity-{seed}-{chronology_index:04d}",
                    "family_id": family_id,
                    "public_grammar_id": FAMILY_GRAMMAR["grammar_version"],
                    "numeric_value": numeric_value,
                    "public_input": f"family={family_id};value={numeric_value}",
                }
            )
            authority_events.append(
                {
                    "event_id": event_id,
                    "seed": seed,
                    "chronology_index": chronology_index,
                    "family_id": family_id,
                    "numeric_value": numeric_value,
                    "window": window_name,
                    "regime": regime,
                    "change_time": int(window["start"]),
                    "hidden_parameter": hidden_parameter,
                    "stable_parameter": stable_parameter,
                    "shifted_parameter": shifted_parameter,
                    "exact_label": label,
                    "independent_exact_label": independent,
                    "poisoned": poisoned,
                    "observed_label": observed_label,
                    "delay_by_schedule": {
                        "constant_0": 0,
                        "constant_4": 4,
                        "constant_16": 16,
                        "burst": _burst_delay(seed, chronology_index),
                    },
                }
            )
    return StreamViews(public_events, authority_events, _stream_manifest())


def _nested_keys(value: Any) -> set[str]:
    """Collect nested keys so hidden fields cannot evade an access check."""

    if isinstance(value, Mapping):
        return set(value) | set().union(*(_nested_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_nested_keys(item) for item in value), set())
    return set()


def public_leakage_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Name public rows that expose any evaluator-only field."""

    return [
        str(row.get("event_id", "missing_event_id"))
        for row in rows
        if _nested_keys(row) & FORBIDDEN_PUBLIC_FIELDS
    ]


def _windows_valid(windows: Sequence[Mapping[str, Any]]) -> bool:
    """Require exact non-overlapping coverage of all 1,024 chronology indices."""

    return (
        len(windows) == len(WINDOWS)
        and [int(row["start"]) for row in windows] == [0, 128, 256, 768, 896]
        and [int(row["stop"]) for row in windows] == [128, 256, 768, 896, 1_024]
        and all(int(row["stop"]) - int(row["start"]) == int(row["count"]) for row in windows)
    )


def stream_conformance_errors(views: StreamViews) -> list[str]:
    """Return stable failures for balance, separation, and exact grounding."""

    errors: list[str] = []
    expected_count = len(STREAM_SEEDS) * EVENTS_PER_SEED
    if len(views.public_events) != expected_count:
        errors.append("public_event_count")
    if len(views.authority_events) != expected_count:
        errors.append("authority_event_count")
    public_ids = [str(row.get("event_id")) for row in views.public_events]
    authority_ids = [str(row.get("event_id")) for row in views.authority_events]
    if public_ids != authority_ids or len(set(public_ids)) != expected_count:
        errors.append("event_identity")
    if public_leakage_errors(views.public_events):
        errors.append("public_access_leakage")
    if not _windows_valid(views.manifest.get("windows", [])):
        errors.append("window_partition")
    public_by_id = {str(row["event_id"]): row for row in views.public_events}
    for truth in views.authority_events:
        public = public_by_id.get(str(truth["event_id"]), {})
        if public.get("family_id") != truth.get("family_id") or public.get(
            "numeric_value"
        ) != truth.get("numeric_value"):
            errors.append("public_authority_join")
            break
        try:
            extracted = extract_public_input(str(public.get("public_input", "")))
        except ValueError:
            errors.append("public_extraction")
            break
        if extracted != {
            "family_id": public["family_id"],
            "numeric_value": public["numeric_value"],
        }:
            errors.append("public_extraction")
            break
        if truth.get("exact_label") != truth.get("independent_exact_label"):
            errors.append("exact_label_disagreement")
            break
        observed_matches = truth.get("observed_label") == truth.get("exact_label")
        if observed_matches == bool(truth.get("poisoned")):
            errors.append("poison_witness")
            break
    for seed in STREAM_SEEDS:
        seed_rows = [row for row in views.authority_events if row.get("seed") == seed]
        for window in WINDOWS:
            rows = [row for row in seed_rows if row.get("window") == window["name"]]
            family_counts = [
                sum(row.get("family_id") == family for row in rows) for family in FAMILIES
            ]
            if len(rows) != window["count"] or len(set(family_counts)) != 1:
                errors.append("family_balance")
                return list(dict.fromkeys(errors))
    return list(dict.fromkeys(errors))


def public_event_iterator(rows: Sequence[Mapping[str, Any]]) -> Iterator[JsonDict]:
    """Yield learner inputs without sharing mutable authority references."""

    for row in rows:
        if _nested_keys(row) & FORBIDDEN_PUBLIC_FIELDS:
            raise ValueError("public_event_contains_authority_field")
        yield deepcopy(dict(row))


def feedback_release_iterator(
    pending: Sequence[PendingRecord], chronology_index: int
) -> Iterator[JsonDict]:
    """Yield labels only when the private queue says their delay has elapsed."""

    for record in pending:
        if record.release_index <= chronology_index:
            yield {
                "event_id": record.event_id,
                "family_id": record.family_id,
                "numeric_value": record.numeric_value,
                "request_index": record.request_index,
                "release_index": record.release_index,
                "observed_label": record.observed_label,
                "exact_label": record.exact_label,
                "poisoned": record.poisoned,
            }


def disagreement_fraction(
    public_event: Mapping[str, Any],
    hypotheses: Mapping[str, set[int]] | None = None,
) -> float:
    """Score only public predictions, with no label or delivery-time input."""

    family_id = str(public_event["family_id"])
    candidates = PARAMETER_DOMAIN if hypotheses is None else tuple(hypotheses[family_id])
    if not candidates:
        return 0.0
    labels = [
        exact_label(family_id, int(public_event["numeric_value"]), parameter)
        for parameter in candidates
    ]
    accepts = labels.count("accept")
    return min(accepts, len(labels) - accepts) / len(labels)


def seeded_tie_ranks(
    seed: int, block_index: int, block: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """Give random and disagreement admission the same frozen tie order."""

    event_ids = [str(row["event_id"]) for row in block]
    random.Random(_stable_seed("tie", seed, block_index)).shuffle(event_ids)
    return {event_id: rank for rank, event_id in enumerate(event_ids)}


def select_request(
    block: Sequence[Mapping[str, Any]],
    arm: str,
    tie_ranks: Mapping[str, int],
    hypotheses: Mapping[str, set[int]] | None = None,
) -> Mapping[str, Any]:
    """Select one request from public rows without seeing authority timing."""

    if arm == "random_admission":
        return min(block, key=lambda row: tie_ranks[str(row["event_id"])])
    if arm == "disagreement_admission":
        return min(
            block,
            key=lambda row: (
                -disagreement_fraction(row, hypotheses),
                tie_ranks[str(row["event_id"])],
            ),
        )
    raise ValueError(f"unsupported_admission_arm:{arm}")


def _state_hash(hypotheses: Mapping[str, set[int]]) -> str:
    """Hash only learner-visible finite version spaces."""

    return sha256_json({family: sorted(hypotheses[family]) for family in FAMILIES})


def _prediction(public: Mapping[str, Any], hypotheses: Mapping[str, set[int]]) -> str:
    """Return a majority prediction or abstain when no hypothesis survives."""

    family_id = str(public["family_id"])
    labels = [
        exact_label(family_id, int(public["numeric_value"]), parameter)
        for parameter in hypotheses[family_id]
    ]
    if not labels:
        return "abstain"
    accepts = labels.count("accept")
    return "accept" if accepts > len(labels) / 2 else "reject"


def _apply_release(hypotheses: dict[str, set[int]], release: Mapping[str, Any]) -> None:
    """Update the fixture version space only from a released observed label."""

    family_id = str(release["family_id"])
    numeric_value = int(release["numeric_value"])
    observed_label = str(release["observed_label"])
    hypotheses[family_id] = {
        parameter
        for parameter in hypotheses[family_id]
        if exact_label(family_id, numeric_value, parameter) == observed_label
    }


def _memory_bytes(
    hypotheses: Mapping[str, set[int]],
    released_event_ids: Sequence[str],
    pending: Sequence[PendingRecord],
    staging: Sequence[Mapping[str, Any]],
) -> int:
    """Charge version spaces, evidence IDs, private pending rows, and staging."""

    state = {
        "hypotheses": {family: sorted(hypotheses[family]) for family in FAMILIES},
        "released_event_ids": list(released_event_ids),
        "pending": [record.__dict__ for record in pending],
        "staging": [dict(row) for row in staging],
    }
    return len(canonical_json(state))


def _window_metric_template() -> dict[str, JsonDict]:
    """Create counters for every frozen evaluator window."""

    return {str(window["name"]): {"events": 0, "errors": 0, "abstentions": 0} for window in WINDOWS}


def _panel_hash(panel: CapacityPanel) -> str:
    """Bind all stable comparison rows without host timing."""

    return sha256_json(
        {
            "rows": panel.rows,
            "information_budget_rows": panel.information_budget_rows,
            "warmup_state_rows": panel.warmup_state_rows,
            "pending_queue_rows": panel.pending_queue_rows,
            "headroom_rows": panel.headroom_rows,
        }
    )


def run_capacity_panel(views: StreamViews, *, progress: bool = False) -> CapacityPanel:
    """Replay every capacity-delay cell with label-blind admission callbacks."""

    started = time.monotonic()
    authority = {str(row["event_id"]): row for row in views.authority_events}
    public_by_seed = {
        seed: [row for row in views.public_events if row["seed"] == seed] for seed in STREAM_SEEDS
    }
    rows: list[JsonDict] = []
    information_rows: list[JsonDict] = []
    warmup_rows: list[JsonDict] = []
    pending_rows: list[JsonDict] = []
    headroom_rows: list[JsonDict] = [
        {
            "slice": "exp7184_static_future",
            "source_experiment": 7184,
            "static_error_rate": 0.0,
            "oracle_error_rate": 0.0,
            "headroom": 0.0,
            "headroom_class": "no_headroom",
            "oracle_is_unattainable": True,
        }
    ]
    for seed_number, seed in enumerate(STREAM_SEEDS, start=1):
        seed_public = public_by_seed[seed]
        for capacity in CAPACITIES:
            for delay_schedule in DELAY_SCHEDULES:
                hypotheses = {
                    arm: {family: set(PARAMETER_DOMAIN) for family in FAMILIES} for arm in ARMS
                }
                pending = {arm: [] for arm in ARMS}
                released_ids = {arm: [] for arm in ARMS}
                requests = {arm: 0 for arm in ARMS}
                releases = {arm: 0 for arm in ARMS}
                dropped = {arm: 0 for arm in ARMS}
                validation_requests = {arm: 0 for arm in ARMS}
                validation_releases = {arm: 0 for arm in ARMS}
                max_pending = {arm: 0 for arm in ARMS}
                max_memory = {arm: 0 for arm in ARMS}
                metrics = {arm: _window_metric_template() for arm in ARMS}
                eligible_blocks = {arm: 0 for arm in ARMS}
                for block_start in range(0, EVENTS_PER_SEED, BLOCK_SIZE):
                    block = seed_public[block_start : block_start + BLOCK_SIZE]
                    block_end = block_start + BLOCK_SIZE - 1
                    block_index = block_start // BLOCK_SIZE
                    tie_ranks = seeded_tie_ranks(seed, block_index, block)
                    for arm in ARMS:
                        for public in block:
                            truth = authority[str(public["event_id"])]
                            prediction = _prediction(public, hypotheses[arm])
                            window_metric = metrics[arm][str(truth["window"])]
                            window_metric["events"] += 1
                            if prediction == "abstain":
                                window_metric["abstentions"] += 1
                                window_metric["errors"] += 1
                            elif prediction != truth["exact_label"]:
                                window_metric["errors"] += 1
                        occupancy_before = len(pending[arm])
                        warmup_eligible = block_end < WARMUP_REQUEST_STOP
                        online_eligible = block_start >= WARMUP_COUNT and arm != "static_frozen"
                        eligible = warmup_eligible or online_eligible
                        selected: Mapping[str, Any] | None = None
                        request_status = "not_requested"
                        if eligible:
                            eligible_blocks[arm] += 1
                            selector = (
                                "disagreement_admission" if block_start < WARMUP_COUNT else arm
                            )
                            selected = select_request(
                                block,
                                selector,
                                tie_ranks,
                                hypotheses[arm],
                            )
                            if occupancy_before >= capacity:
                                dropped[arm] += 1
                                request_status = "dropped_capacity_full"
                            else:
                                truth = authority[str(selected["event_id"])]
                                delay = int(truth["delay_by_schedule"][delay_schedule])
                                pending[arm].append(
                                    PendingRecord(
                                        event_id=str(selected["event_id"]),
                                        family_id=str(selected["family_id"]),
                                        numeric_value=int(selected["numeric_value"]),
                                        request_index=block_end,
                                        release_index=block_end + delay,
                                        observed_label=str(truth["observed_label"]),
                                        exact_label=str(truth["exact_label"]),
                                        poisoned=bool(truth["poisoned"]),
                                    )
                                )
                                requests[arm] += 1
                                if truth["window"] == "online_validation":
                                    validation_requests[arm] += 1
                                request_status = "admitted"
                        max_pending[arm] = max(max_pending[arm], len(pending[arm]))
                        staging_bytes = _memory_bytes(
                            hypotheses[arm], released_ids[arm], pending[arm], block
                        )
                        max_memory[arm] = max(max_memory[arm], staging_bytes)
                        released = list(feedback_release_iterator(pending[arm], block_end))
                        released_event_ids = {str(row["event_id"]) for row in released}
                        for release in released:
                            if arm != "static_frozen" or block_end < WARMUP_COUNT:
                                _apply_release(hypotheses[arm], release)
                            released_ids[arm].append(str(release["event_id"]))
                            releases[arm] += 1
                            truth = authority[str(release["event_id"])]
                            if truth["window"] == "online_validation":
                                validation_releases[arm] += 1
                        pending[arm] = [
                            record
                            for record in pending[arm]
                            if record.event_id not in released_event_ids
                        ]
                        max_memory[arm] = max(
                            max_memory[arm],
                            _memory_bytes(hypotheses[arm], released_ids[arm], pending[arm], []),
                        )
                        if seed == STREAM_SEEDS[0]:
                            pending_rows.append(
                                {
                                    "seed": seed,
                                    "capacity": capacity,
                                    "delay_schedule": delay_schedule,
                                    "arm": arm,
                                    "block_index": block_index,
                                    "chronology_index": block_end,
                                    "predictions_committed": len(block),
                                    "selection_after_predictions": True,
                                    "future_delay_visible_to_selector": False,
                                    "occupancy_before_selection": occupancy_before,
                                    "selected_event_id": (
                                        None if selected is None else selected["event_id"]
                                    ),
                                    "request_status": request_status,
                                    "released_after_selection_count": len(released),
                                    "release_after_selection": True,
                                    "occupancy_after_release": len(pending[arm]),
                                    "pending_eviction_count": 0,
                                }
                            )
                        if block_end == WARMUP_COUNT - 1:
                            warmup_rows.append(
                                {
                                    "seed": seed,
                                    "capacity": capacity,
                                    "delay_schedule": delay_schedule,
                                    "arm": arm,
                                    "state_hash": _state_hash(hypotheses[arm]),
                                    "released_event_ids": list(released_ids[arm]),
                                    "pending_count": len(pending[arm]),
                                    "frozen_static_policy": arm == "static_frozen",
                                }
                            )
                for arm in ARMS:
                    window_metrics = metrics[arm]
                    total_errors = sum(int(value["errors"]) for value in window_metrics.values())
                    total_abstentions = sum(
                        int(value["abstentions"]) for value in window_metrics.values()
                    )
                    unit_id = f"{seed}:{capacity}:{delay_schedule}:{arm}"
                    row = {
                        "unit_id": unit_id,
                        "arm": arm,
                        "seed": seed,
                        "capacity": capacity,
                        "delay_schedule": delay_schedule,
                        "metric": "full_stream_error_rate",
                        "error": total_errors,
                        "abstention": total_abstentions,
                        "event_count": EVENTS_PER_SEED,
                        "error_rate": total_errors / EVENTS_PER_SEED,
                        "requests": requests[arm],
                        "releases": releases[arm],
                        "dropped_requests": dropped[arm],
                        "eligible_block_count": eligible_blocks[arm],
                        "max_pending": max_pending[arm],
                        "max_memory_bytes": max_memory[arm],
                        "pending_eviction_count": 0,
                        "window_metrics": window_metrics,
                    }
                    rows.append(row)
                    warmup = next(
                        receipt
                        for receipt in warmup_rows
                        if receipt["seed"] == seed
                        and receipt["capacity"] == capacity
                        and receipt["delay_schedule"] == delay_schedule
                        and receipt["arm"] == arm
                    )
                    information_rows.append(
                        {
                            "unit_id": unit_id,
                            "arm": arm,
                            "seed": seed,
                            "capacity": capacity,
                            "delay_schedule": delay_schedule,
                            "warmup_state_hash": warmup["state_hash"],
                            "warmup_released_count": len(warmup["released_event_ids"]),
                            "warmup_pending_count": warmup["pending_count"],
                            "requests": requests[arm],
                            "released": releases[arm],
                            "dropped_capacity_full": dropped[arm],
                            "online_validation_requested": validation_requests[arm],
                            "online_validation_released": validation_releases[arm],
                            "pending_labels_evicted": 0,
                            "memory_byte_budget": MEMORY_BYTE_BUDGET,
                            "max_memory_bytes": max_memory[arm],
                        }
                    )
                    if arm == "static_frozen":
                        prospective = window_metrics["prospective"]
                        static_error_rate = prospective["errors"] / prospective["events"]
                        headroom_rows.append(
                            {
                                "slice": "prospective_shifted",
                                "seed": seed,
                                "capacity": capacity,
                                "delay_schedule": delay_schedule,
                                "static_error_rate": static_error_rate,
                                "oracle_error_rate": 0.0,
                                "headroom": static_error_rate,
                                "headroom_class": (
                                    "positive_control_headroom"
                                    if static_error_rate > 0
                                    else "no_headroom"
                                ),
                                "oracle_is_unattainable": True,
                            }
                        )
        if progress:
            print(
                f"PHASE 4 PROGRESS: completed seed {seed_number}/{len(STREAM_SEEDS)}; "
                f"elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
    return CapacityPanel(rows, information_rows, warmup_rows, pending_rows, headroom_rows)


def _panel_conformance_errors(panel: CapacityPanel) -> list[str]:
    """Check matched warmup state, request quotas, queue bounds, and bytes."""

    errors: list[str] = []
    expected = len(STREAM_SEEDS) * len(CAPACITIES) * len(DELAY_SCHEDULES) * len(ARMS)
    identities = {
        (row["seed"], row["capacity"], row["delay_schedule"], row["arm"]) for row in panel.rows
    }
    expected_identities = {
        (seed, capacity, delay, arm)
        for seed in STREAM_SEEDS
        for capacity in CAPACITIES
        for delay in DELAY_SCHEDULES
        for arm in ARMS
    }
    if len(panel.rows) != expected or identities != expected_identities:
        errors.append("row_panel")
    if any(row["requests"] > row["eligible_block_count"] for row in panel.rows):
        errors.append("request_budget")
    if any(row["max_pending"] > row["capacity"] for row in panel.rows):
        errors.append("pending_capacity")
    if any(row["max_memory_bytes"] > MEMORY_BYTE_BUDGET for row in panel.rows):
        errors.append("memory_budget")
    if any(row["pending_eviction_count"] != 0 for row in panel.rows):
        errors.append("pending_eviction")
    if len(panel.information_budget_rows) != expected:
        errors.append("information_budget_count")
    for seed in STREAM_SEEDS:
        for capacity in CAPACITIES:
            for delay in DELAY_SCHEDULES:
                warmup = [
                    row
                    for row in panel.warmup_state_rows
                    if row["seed"] == seed
                    and row["capacity"] == capacity
                    and row["delay_schedule"] == delay
                ]
                if (
                    len(warmup) != len(ARMS)
                    or len({row["state_hash"] for row in warmup}) != 1
                    or len({tuple(row["released_event_ids"]) for row in warmup}) != 1
                    or any(row["pending_count"] != 0 for row in warmup)
                ):
                    errors.append("warmup_information_mismatch")
                    return list(dict.fromkeys(errors))
    if any(
        row["predictions_committed"] != BLOCK_SIZE
        or row["selection_after_predictions"] is not True
        or row["future_delay_visible_to_selector"] is not False
        or row["release_after_selection"] is not True
        or row["pending_eviction_count"] != 0
        for row in panel.pending_queue_rows
    ):
        errors.append("queue_chronology")
    return list(dict.fromkeys(errors))


def _source_grounding_rows(views: StreamViews) -> list[JsonDict]:
    """Summarize complete public parse and independent exact-score agreement."""

    rows: list[JsonDict] = []
    public = {str(row["event_id"]): row for row in views.public_events}
    for seed in STREAM_SEEDS:
        truth_rows = [row for row in views.authority_events if row["seed"] == seed]
        agreement_count = 0
        for truth in truth_rows:
            extracted = extract_public_input(public[str(truth["event_id"])]["public_input"])
            first = exact_label(**extracted, parameter=int(truth["hidden_parameter"]))
            second = independent_exact_label(**extracted, parameter=int(truth["hidden_parameter"]))
            agreement_count += int(first == second == truth["exact_label"])
        rows.append(
            {
                "seed": seed,
                "public_input_count": len(truth_rows),
                "extraction_count": len(truth_rows),
                "execution_count": len(truth_rows),
                "independent_score_count": len(truth_rows),
                "agreement_count": agreement_count,
                "passed": agreement_count == len(truth_rows),
            }
        )
    return rows


def _mutation_audit_rows(views: StreamViews, panel: CapacityPanel) -> list[JsonDict]:
    """Execute one isolated change for each preregistered readiness failure."""

    changed_seed = deepcopy(views.manifest)
    changed_seed["seeds"][0] += 1
    leaked_public = deepcopy(views.public_events[:1])
    leaked_public[0]["hidden_parameter"] = 1
    overlap = deepcopy(views.manifest["windows"])
    overlap[1]["start"] = 127
    extra_request = deepcopy(panel.rows[:1])
    extra_request[0]["requests"] = extra_request[0]["eligible_block_count"] + 1
    pending_eviction = deepcopy(panel.rows[:1])
    pending_eviction[0]["pending_eviction_count"] = 1
    memory_overflow = deepcopy(panel.rows[:1])
    memory_overflow[0]["max_memory_bytes"] = MEMORY_BYTE_BUDGET + 1
    return [
        {
            "mutation": "changed_seed",
            "detector": "manifest_seed_identity",
            "detected": changed_seed != _stream_manifest(),
        },
        {
            "mutation": "public_hidden_field",
            "detector": "public_access_isolation",
            "detected": bool(public_leakage_errors(leaked_public)),
        },
        {
            "mutation": "overlapping_window",
            "detector": "disjoint_window_partition",
            "detected": not _windows_valid(overlap),
        },
        {
            "mutation": "extra_request",
            "detector": "one_request_per_block",
            "detected": extra_request[0]["requests"] > extra_request[0]["eligible_block_count"],
        },
        {
            "mutation": "pending_eviction",
            "detector": "drop_new_queue_policy",
            "detected": pending_eviction[0]["pending_eviction_count"] != 0,
        },
        {
            "mutation": "memory_overflow",
            "detector": "bounded_memory_bytes",
            "detected": memory_overflow[0]["max_memory_bytes"] > MEMORY_BYTE_BUDGET,
        },
    ]


def _resolve(repo_root: Path, path: Path | str) -> Path:
    """Resolve evidence relative to the selected checkout."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _load_object(path: Path) -> JsonDict:
    """Decode one object while keeping missing or malformed evidence explicit."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _task_block(text: str) -> str:
    """Extract only Exp7198 so another roadmap task cannot satisfy identity gates."""

    match = re.search(r"(?ms)^- id: exp7198-feedback-capacity-stream\n(.*?)(?=^- id:|\Z)", text)
    return "" if match is None else match.group(0)


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating task evidence early."""

    candidate = path if path.suffix == "" else path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.is_dir() and os.access(candidate, os.W_OK)


def _source_hashes(repo_root: Path, upstream_artifact_path: Path) -> JsonDict:
    """Bind task code, contracts, tests, and the exact prior-null artifact."""

    hashes = {str(path): sha256_path(repo_root / path) for path in SOURCE_PATHS}
    hashes[str(upstream_artifact_path)] = sha256_path(_resolve(repo_root, upstream_artifact_path))
    return hashes


_QUARANTINE_KEYS = (
    "artifact_quarantined",
    "upstream_quarantined",
    "quarantine_flag",
    "excluded_from_use",
)


def _quarantine_state(upstream: Mapping[str, Any], exclusion_text: str) -> JsonDict:
    """Combine artifact metadata with the independent exclusion manifest."""

    declared = {key: upstream[key] for key in _QUARANTINE_KEYS if key in upstream}
    markers = (
        "experiment_7184_v633_revocable_template_csl.json",
        "exp7184-revocable-template-csl",
    )
    manifest_matches = [marker for marker in markers if marker in exclusion_text]
    return {
        "quarantined": any(value is True for value in declared.values()) or bool(manifest_matches),
        "declared_flags": declared,
        "exclusion_manifest_matches": manifest_matches,
    }


def collect_preconditions(
    repo_root: Path,
    upstream_artifact_path: Path,
    paths: ExperimentPaths,
) -> tuple[list[JsonDict], JsonDict]:
    """Check external evidence before generating or scheduling any event."""

    upstream_path = _resolve(repo_root, upstream_artifact_path)
    upstream = _load_object(upstream_path)
    spec_text = (
        (repo_root / SPEC_PATH).read_text(encoding="utf-8")
        if (repo_root / SPEC_PATH).is_file()
        else ""
    )
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
    hashes = _source_hashes(repo_root, upstream_artifact_path)
    identity = {
        "id": "exp7198-feedback-capacity-stream" if task else None,
        "milestone": MILESTONE if f"milestone: {MILESTONE}" in task else None,
        "deliverable": (
            str(DEFAULT_ARTIFACT_PATH) if f"deliverable: {DEFAULT_ARTIFACT_PATH}" in task else None
        ),
    }
    expected_identity = {
        "id": "exp7198-feedback-capacity-stream",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT_PATH),
    }
    upstream_errors = exp7184.validate_artifact(upstream) if upstream else ["missing_upstream"]
    tools = {
        "python_executable": Path(sys.executable).is_file(),
        "sha256sum": shutil.which("sha256sum") is not None,
    }
    destinations = {
        "public_stream_parent": _path_writable(_resolve(repo_root, paths.public_stream)),
        "authority_sidecar_parent": _path_writable(_resolve(repo_root, paths.authority_sidecar)),
        "manifest_parent": _path_writable(_resolve(repo_root, paths.manifest)),
        "checkpoint_parent": _path_writable(_resolve(repo_root, paths.checkpoint)),
        "artifact_parent": _path_writable(_resolve(repo_root, paths.artifact)),
    }
    return [
        gate_check(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-CL-7198",
            True,
            "## REQ-CL-7198:" in spec_text,
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7198-*",
            9,
            spec_text.count("### SCENARIO-CL-7198-"),
            spec_text.count("### SCENARIO-CL-7198-") >= 9,
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
            "repository",
            "SOURCE_PATHS.sha256",
            "sha256:<64 hex> for every source and upstream artifact",
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
            "exp7184-revocable-template-csl",
            "status",
            "complete",
            upstream.get("status"),
        ),
        gate_check(
            "upstream_completion_gate",
            "exp7184-revocable-template-csl",
            "memory_run_complete_score",
            1,
            upstream.get("memory_run_complete_score"),
        ),
        gate_check(
            "known_upstream_null_preserved",
            "exp7184-revocable-template-csl",
            "memory_value_score",
            0,
            upstream.get("memory_value_score"),
        ),
        gate_check(
            "upstream_verdict_class",
            "exp7184-revocable-template-csl",
            "verdict_class",
            "null",
            upstream.get("verdict_class"),
        ),
        gate_check(
            "upstream_milestone",
            "exp7184-revocable-template-csl",
            "milestone",
            "2026.09.633",
            upstream.get("milestone"),
        ),
        gate_check(
            "upstream_cold_validation",
            "exp7184-revocable-template-csl",
            "cold_validation_errors",
            [],
            upstream_errors,
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
            "required_local_tools",
            "host",
            "python,sha256sum",
            {key: True for key in tools},
            tools,
        ),
        gate_check(
            "output_destinations_writable",
            "host_filesystem",
            "stream,authority,manifest,checkpoint,artifact",
            {key: True for key in destinations},
            destinations,
        ),
    ], upstream


def _empty_evidence() -> JsonDict:
    """Keep blocked output schema-complete without inventing measurements."""

    return {
        "rows": [],
        "information_budget_rows": [],
        "headroom_rows": [],
        "warmup_state_rows": [],
        "source_grounding_rows": [],
        "mutation_audit_rows": [],
        "pending_queue_rows": [],
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all stable evidence while excluding measured host duration."""

    stable = deepcopy(dict(artifact))
    stable["duration_s"] = None
    stable["reproducibility_checksum"] = None
    return sha256_json(stable)


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
    """Build provenance fields before any readiness or terminal claim."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "blocked",
        "run_date": run_date,
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": "blocked before qualifying stream computation",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "source_artifact_hashes": _source_hashes(repo_root, upstream_artifact_path),
        "sample_size_budget": {
            "planned_events": len(STREAM_SEEDS) * EVENTS_PER_SEED,
            "completed_events": 0,
            "independent_stream_units_planned": len(STREAM_SEEDS),
            "independent_stream_units_completed": 0,
            "capacity_delay_cells_planned": len(CAPACITIES) * len(DELAY_SCHEDULES),
            "capacity_delay_cells_completed": 0,
            "exclusions": [],
        },
        "random_seed": {
            "master": RANDOM_SEED,
            "stream_seeds": list(STREAM_SEEDS),
            "derivation": "sha256 canonical tuple prefixes",
        },
        "reproducibility_checksum": None,
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_external: bounded feedback stream did not run",
        "stream_capacity_ready_score": 0,
        "stream_manifest": _stream_manifest(),
        "public_stream_path": str(paths.public_stream),
        "authority_sidecar_path": str(paths.authority_sidecar),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "public_view_hash": None,
        "authority_sidecar_hash": None,
        "stream_manifest_path": str(paths.manifest),
        "stream_manifest_hash": None,
        "capacity_delay_contract": {
            "capacities": list(CAPACITIES),
            "delay_schedules": list(DELAY_SCHEDULES),
            "block_size": BLOCK_SIZE,
            "request_limit_per_block": 1,
            "memory_byte_budget": MEMORY_BYTE_BUDGET,
            "full_queue_action": "drop_new",
            "pending_eviction_allowed": False,
            "release_capacity_available": "next_boundary",
        },
        "prior_null_gate_receipt": {
            "upstream": "exp7184-revocable-template-csl",
            "artifact_path": str(upstream_artifact_path),
            "artifact_hash": sha256_path(_resolve(repo_root, upstream_artifact_path)),
            "completion_field": "memory_run_complete_score",
            "completion_expected": 1,
            "completion_observed": upstream.get("memory_run_complete_score"),
            "null_field": "memory_value_score",
            "null_expected": 0,
            "null_observed": upstream.get("memory_value_score"),
            "promoted_as_positive": False,
        },
        "checkpoint_path": str(paths.checkpoint),
        "checkpoint_hash": None,
        "panel_hash": None,
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
    """Return a terminal row-free artifact for an external failed check."""

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


def _artifact_panel(artifact: Mapping[str, Any]) -> CapacityPanel:
    """Project stored rows into the same structure used by cold checks."""

    return CapacityPanel(
        list(artifact["rows"]),
        list(artifact["information_budget_rows"]),
        list(artifact["warmup_state_rows"]),
        list(artifact["pending_queue_rows"]),
        list(artifact["headroom_rows"]),
    )


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    check_files: bool = False,
) -> list[str]:
    """Cold-check schema, sources, deterministic rows, budgets, and verdict."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return ["missing_fields:" + ",".join(missing)]
    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(
        set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS),
        "field_principles_mismatch",
    )
    add(artifact["schema"] != SCHEMA, "schema_mismatch")
    add(artifact["experiment_id"] != EXPERIMENT_ID, "experiment_id_mismatch")
    add(artifact["milestone"] != MILESTONE, "milestone_mismatch")
    add(artifact["run_date"] != RUN_DATE, "run_date_mismatch")
    add(artifact["MODEL_SPECS"] != [], "model_specs_mismatch")
    add(artifact["model_invoked"] is not False, "model_invoked_mismatch")
    add(artifact["verifier_is_oracle"] is not True, "verifier_is_oracle_mismatch")
    add(type(artifact["stream_capacity_ready_score"]) is not int, "ready_score_not_bare_integer")
    blocked = artifact["verdict_class"] == "blocked"
    if blocked:
        add(artifact["status"] != "blocked", "blocked_status_mismatch")
        add(artifact["inference_substrate_class"] != "blocked_no_run", "blocked_substrate_mismatch")
        add(artifact["stream_capacity_ready_score"] != 0, "blocked_ready_score_mismatch")
        add(bool(artifact["rows"]), "blocked_rows_present")
        add(
            artifact["gate_check_summary"].get("passed") is not False,
            "blocked_gate_summary_mismatch",
        )
        add(not artifact["gate_check_summary"].get("failed_check"), "blocked_failed_check_missing")
        add(not artifact["gate_check_summary"].get("upstream"), "blocked_upstream_missing")
        add(not artifact["gate_check_summary"].get("field"), "blocked_field_missing")
    else:
        add(artifact["status"] != "complete", "status_mismatch")
        add(artifact["inference_substrate"] != INFERENCE_SUBSTRATE, "inference_substrate_mismatch")
        add(
            artifact["inference_substrate_class"] != INFERENCE_SUBSTRATE_CLASS,
            "inference_substrate_class_mismatch",
        )
        add(artifact["execution_venue"] != EXECUTION_VENUE, "execution_venue_mismatch")
        add(artifact["verdict_class"] != "circular_positive", "verdict_class_mismatch")
        add("no learning benefit" not in artifact["honest_verdict"], "honest_verdict_mismatch")
        add(artifact["stream_manifest"] != _stream_manifest(), "stream_manifest_mismatch")
        sample = artifact["sample_size_budget"]
        add(sample.get("planned_events") != 10_240, "sample_budget_planned_mismatch")
        add(sample.get("completed_events") != 10_240, "sample_budget_completed_mismatch")
        add(sample.get("exclusions") != [], "sample_budget_exclusions_mismatch")
        panel = _artifact_panel(artifact)
        panel_errors = _panel_conformance_errors(panel)
        add(bool(panel_errors), "row_panel_mismatch")
        add(artifact["panel_hash"] != _panel_hash(panel), "panel_hash_mismatch")
        add(
            {row.get("mutation") for row in artifact["mutation_audit_rows"]}
            != {
                "changed_seed",
                "public_hidden_field",
                "overlapping_window",
                "extra_request",
                "pending_eviction",
                "memory_overflow",
            }
            or any(row.get("detected") is not True for row in artifact["mutation_audit_rows"]),
            "mutation_audit_mismatch",
        )
        add(
            any(row.get("passed") is not True for row in artifact["source_grounding_rows"]),
            "source_grounding_mismatch",
        )
        add(
            artifact["prior_null_gate_receipt"].get("null_observed") != 0
            or artifact["prior_null_gate_receipt"].get("promoted_as_positive") is not False,
            "prior_null_gate_mismatch",
        )
        if not errors:
            expected_views = build_stream_views()
            expected_public_hash = sha256_bytes(exp7183._jsonl_bytes(expected_views.public_events))
            expected_authority_hash = sha256_bytes(
                exp7183._jsonl_bytes(expected_views.authority_events)
            )
            expected_manifest_hash = sha256_bytes(canonical_json(expected_views.manifest) + b"\n")
            add(artifact["public_view_hash"] != expected_public_hash, "public_view_hash_mismatch")
            add(
                artifact["authority_sidecar_hash"] != expected_authority_hash,
                "authority_sidecar_hash_mismatch",
            )
            add(
                artifact["stream_manifest_hash"] != expected_manifest_hash,
                "stream_manifest_hash_mismatch",
            )
            if not errors:
                expected_panel = run_capacity_panel(expected_views)
                add(
                    artifact["panel_hash"] != _panel_hash(expected_panel),
                    "deterministic_panel_mismatch",
                )
        expected_ready = int(not errors)
        add(
            artifact["stream_capacity_ready_score"] != expected_ready,
            "stream_capacity_ready_score_mismatch",
        )
        if check_files and not errors:
            root = repo_root or Path(__file__).resolve().parents[2]
            add(
                sha256_path(_resolve(root, str(artifact["public_stream_path"])))
                != artifact["public_view_hash"],
                "public_view_file_mismatch",
            )
            add(
                sha256_path(_resolve(root, str(artifact["authority_sidecar_path"])))
                != artifact["authority_sidecar_hash"],
                "authority_sidecar_file_mismatch",
            )
            add(
                sha256_path(_resolve(root, str(artifact["stream_manifest_path"])))
                != artifact["stream_manifest_hash"],
                "stream_manifest_file_mismatch",
            )
            add(
                sha256_path(_resolve(root, str(artifact["checkpoint_path"])))
                != artifact["checkpoint_hash"],
                "checkpoint_file_mismatch",
            )
            upstream_path = Path(artifact["prior_null_gate_receipt"]["artifact_path"])
            add(
                artifact["source_artifact_hashes"] != _source_hashes(root, upstream_path),
                "source_artifact_hashes_mismatch",
            )
    add(
        artifact["reproducibility_checksum"] != reproducibility_checksum(artifact),
        "reproducibility_checksum_mismatch",
    )
    return errors


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    upstream_artifact_path: Path = DEFAULT_UPSTREAM_ARTIFACT_PATH,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    progress: bool = False,
) -> JsonDict:
    """Run preconditions, freeze streams, benchmark queues, and validate evidence."""

    started = time.monotonic()
    if progress:
        print(
            "PHASE 0 START: verify specifications, sources, upstream gates, quarantine, tools, and paths",
            flush=True,
        )
    checks, upstream = collect_preconditions(repo_root, upstream_artifact_path, paths)
    if not all(row["passed"] for row in checks):
        elapsed = time.monotonic() - started if duration_s is None else duration_s
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
        print("PHASE 0 END: external preconditions passed and prior null preserved", flush=True)
        print(
            "PHASE 1 START: freeze seeds, windows, family grammar, capacities, and delays",
            flush=True,
        )
        print("PHASE 1 END: no-model stream contract frozen", flush=True)
        print("PHASE 2 START: generate all public and authority rows before scheduling", flush=True)
    views = build_stream_views()
    stream_errors = stream_conformance_errors(views)
    if progress:
        print("PHASE 2 END: generated 10,240 events with balanced family windows", flush=True)
        print("PHASE 3 START: seal public, authority, and manifest bytes", flush=True)
    public_hash = write_immutable_jsonl(
        _resolve(repo_root, paths.public_stream), views.public_events
    )
    authority_hash = write_immutable_jsonl(
        _resolve(repo_root, paths.authority_sidecar), views.authority_events
    )
    manifest_hash = write_immutable_json(_resolve(repo_root, paths.manifest), views.manifest)
    if progress:
        print("PHASE 3 END: immutable view and manifest hashes recorded", flush=True)
        print("PHASE 4 START: benchmark all bounded pending-feedback cells", flush=True)
    panel = run_capacity_panel(views, progress=progress)
    panel_errors = _panel_conformance_errors(panel)
    if progress:
        print("PHASE 4 END: all capacity-delay-arm units completed", flush=True)
        print(
            "PHASE 5 START: verify public access isolation, chronology, and grounding", flush=True
        )
    grounding_rows = _source_grounding_rows(views)
    if progress:
        print("PHASE 5 END: public extraction and independent exact scoring agree", flush=True)
        print("PHASE 6 START: execute deterministic leakage and budget mutations", flush=True)
    mutation_rows = _mutation_audit_rows(views, panel)
    if progress:
        print("PHASE 6 END: all preregistered mutations were detected", flush=True)
        print("PHASE 7 START: checkpoint completed units and derive readiness", flush=True)
    panel_hash = _panel_hash(panel)
    checkpoint = {
        "schema": "carnot.exp7198.progress.v1",
        "completed_events": len(views.public_events),
        "completed_panel_units": len(panel.rows),
        "public_view_hash": public_hash,
        "authority_sidecar_hash": authority_hash,
        "stream_manifest_hash": manifest_hash,
        "panel_hash": panel_hash,
    }
    checkpoint_path = _resolve(repo_root, paths.checkpoint)
    write_json_atomic(checkpoint_path, checkpoint)
    checkpoint_hash = sha256_path(checkpoint_path)
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
    readiness = int(
        not stream_errors
        and not panel_errors
        and all(row["passed"] for row in grounding_rows)
        and all(row["detected"] for row in mutation_rows)
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "rows": deepcopy(panel.rows),
            "sample_size_budget": {
                "planned_events": 10_240,
                "completed_events": len(views.public_events),
                "independent_stream_units_planned": len(STREAM_SEEDS),
                "independent_stream_units_completed": len(STREAM_SEEDS),
                "capacity_delay_cells_planned": 12,
                "capacity_delay_cells_completed": 12,
                "comparison_units_planned": 360,
                "comparison_units_completed": len(panel.rows),
                "exclusions": [],
            },
            "verdict_class": "circular_positive" if readiness else "disqualified",
            "honest_verdict": (
                "complete: bounded feedback stream ready; no learning benefit measured"
                if readiness
                else "complete_disqualified: bounded feedback stream contract failed"
            ),
            "stream_capacity_ready_score": readiness,
            "stream_manifest": deepcopy(views.manifest),
            "public_view_hash": public_hash,
            "authority_sidecar_hash": authority_hash,
            "stream_manifest_hash": manifest_hash,
            "information_budget_rows": deepcopy(panel.information_budget_rows),
            "headroom_rows": deepcopy(panel.headroom_rows),
            "warmup_state_rows": deepcopy(panel.warmup_state_rows),
            "source_grounding_rows": grounding_rows,
            "mutation_audit_rows": mutation_rows,
            "checkpoint_hash": checkpoint_hash,
            "pending_queue_rows": deepcopy(panel.pending_queue_rows),
            "panel_hash": panel_hash,
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact, repo_root=repo_root, check_files=True)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    if progress:
        print(
            "PHASE 7 END: cold validation passed with stream readiness separated from value",
            flush=True,
        )
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed date and optional private evidence paths."""

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
    """Run the real file gates and publish the terminal artifact atomically."""

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
    print(
        "PHASE 8 START: verify principle fields and atomically write terminal artifact", flush=True
    )
    errors = validate_artifact(
        artifact,
        repo_root=repo_root,
        check_files=artifact["verdict_class"] != "blocked",
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    print("FINAL ATOMIC WRITE START", flush=True)
    write_json_atomic(_resolve(repo_root, paths.artifact), artifact)
    print("FINAL ATOMIC WRITE END", flush=True)
    print("PHASE 8 END: terminal artifact is stable", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper owns this path.
    raise SystemExit(main())
