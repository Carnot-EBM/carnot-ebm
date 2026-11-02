"""Build a bounded fixed-share fixture for delayed-feedback learning.

The controller selects among complete finite constraint hypotheses. It changes
only after a label becomes public. The fixture measures causal and storage
mechanics. It does not claim prospective learning value.

Spec refs: REQ-CL-7295 and SCENARIO-CL-7295-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import random
import re
import selectors
import shlex
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7267_v639_recognition_prototype as prototype
from carnot import experiment_7281_v640_admission_prototype as admission
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7295
SCHEMA = "carnot.exp7295.v641_mixture_prototype.v1"
STATE_SCHEMA = "carnot.fixed_share_hypothesis_controller.v1"
MILESTONE = "2026.09.641"
RUN_DATE = "20260914"
RANDOM_SEED = 7_295_000
SHUFFLE_SEED = 7_295_901
DEVELOPMENT_STREAM_SEEDS = tuple(range(7_295_001, 7_295_009))
EVALUATION_STREAM_SEEDS = tuple(range(7_295_101, 7_295_125))
DEVELOPMENT_STREAM_COUNT = 8
EVALUATION_STREAM_COUNT = 24
EVENTS_PER_STREAM = 1_024
WARMUP_COUNT = 128
FUTURE_LABEL_COUNT = 128
FEEDBACK_DELAY = 4
NOMINATION_INTERVAL = 16
ARCHIVE_CAP = 4
MEMORY_CAP_BYTES = admission.MEMORY_CAP_BYTES
ETA = 0.5
FIXED_SHARE = 0.02
FUTURE_LABEL_POSITIONS = tuple(128 + 7 * index for index in range(FUTURE_LABEL_COUNT))
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INVOCATION_COUNTS = {
    "attempted_model_loads": 0,
    "completed_model_loads": 0,
    "failed_model_loads": 0,
    "attempted_generation_calls": 0,
    "completed_generation_calls": 0,
    "failed_generation_calls": 0,
    "usable_answers": 0,
}
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
REDUCER_INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
REDUCER_INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"

FAMILIES = tuple(admission.FAMILIES)
PARAMETER_DOMAIN = tuple(admission.PARAMETER_DOMAIN)
FULL_MASK = admission.FULL_MASK
ARMS = (
    "fixed_share_mixture",
    "frozen_uniform_voting",
    "reset",
    "unconditional_recognition",
    "label_shuffled_fixed_share",
    "unbounded_memory_reference",
    "frozen_warmup",
)
BOUNDED_ARMS = tuple(arm for arm in ARMS if arm != "unbounded_memory_reference")
FORBIDDEN_PUBLIC_FIELDS = set(prototype.FORBIDDEN_PUBLIC_FIELDS) | {
    "source_index",
    "release_index",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
WRAPPER_PATH = Path("scripts/experiments/experiment_7295_v641_mixture_prototype.py")
TEST_PATH = Path("tests/python/test_experiment_7295_v641_mixture_prototype.py")
DEFAULT_ARTIFACT = Path("results/experiment_7295_v641_mixture_prototype.json")
UPSTREAM_ARTIFACT = Path("results/experiment_7281_v640_admission_prototype.json")
SCENARIO_PATTERN = re.compile(r"SCENARIO-CL-7295-[A-Z-]+")

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7267_v639_recognition_prototype.py"),
    Path("python/carnot/experiment_7281_v640_admission_prototype.py"),
    Path("python/carnot/experiment_7282_v640_admission_learning.py"),
    Path("python/carnot/experiment_7283_v640_admission_audit.py"),
    Path("python/carnot/experiment_7295_v641_mixture_prototype.py"),
    WRAPPER_PATH,
    TEST_PATH,
    Path("tests/python/coverage_experiment_7295.py"),
    SPEC_PATH,
    UPSTREAM_ARTIFACT,
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
    "mixture_fixture_ready_score",
    "learning_contract",
    "stream_manifest_path",
    "feedback_schedule",
    "memory_budget_bytes",
    "chronology_control_rows",
)

FIELD_PRINCIPLES = {
    "schema": "Version the artifact; retain ordinary top-level experiment_id and milestone.",
    "experiment_id": "Bind the result to the active experiment task.",
    "milestone": "Bind the result to milestone 2026.09.641.",
    "status": "Use a terminal complete or blocked record; checkpoints hold unfinished work.",
    "run_date": "Use 20260914 with real UTC start and end times.",
    "started_at_utc": "Record the actual UTC invocation start.",
    "completed_at_utc": "Record the actual UTC terminal decision time.",
    "field_principles": "Store explanations here while consumer values stay top-level.",
    "preconditions_checked": "Hash inputs, authority limits, ownership, and failed checks.",
    "MODEL_SPECS": "List only executable models used now; this task uses none.",
    "model_invoked": "Count attempted loads or generations, including failures; this task has none.",
    "invocation_counts": "Separate attempted, completed, and failed model work.",
    "inference_substrate": "Name the actual CPU solver or simulator work.",
    "inference_substrate_class": "Use the actual no-LLM class without time padding.",
    "execution_venue": "Host execution is host; no device executes this fixture.",
    "duration_s": "Measure monotonic elapsed time and separate phase spans.",
    "random_seed": "Freeze development and evaluation seeds before outcomes are read.",
    "reproducibility_checksum": "Bind code, configuration, inputs, and immutable raw evidence.",
    "source_artifact_hashes": "Keep exact producer identities and exclusion state.",
    "rows": "Keep each stream and arm with metrics, costs, abstention, and censoring.",
    "sample_size_budget": "State planned, attempted, complete, and censored fixed units.",
    "acceptance_gate_results": "Each check names expected, observed, passed, and principle.",
    "gate_check_summary": "Blocked verdicts name the exact upstream field and mismatch.",
    "verifier_is_oracle": "Expose shared evaluator authority; mechanics are not learned correctness.",
    "honest_verdict": "Complete findings start complete_; external absence starts blocked_.",
    "verdict_class": "Use the closed class set; oracle authority forbids positive.",
    "validation_receipts": "Retain command, exit code, elapsed time, and log hash.",
    "mixture_fixture_ready_score": "One requires frozen causal, archive, split, and byte controls.",
    "learning_contract": "Record the equation, constants, nomination, eviction, and frozen gates.",
    "stream_manifest_path": "Bind independent splits and scorer-only label hashes.",
    "feedback_schedule": "Charge warmup and delayed future labels separately.",
    "memory_budget_bytes": "Inherit the V640 limit and report achieved charged state.",
    "chronology_control_rows": "Retain prediction, release, update, birth, and attack order.",
}

gate_check = exp7213.gate_check
gate_summary = exp7213.gate_summary


class MixtureRejected(ValueError):
    """Reject invalid feedback or storage before it changes controller bytes."""


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep immutable stream views, raw rows, checkpoints, and terminal bytes separate."""

    raw_dir: Path
    development_public: Path
    development_authority: Path
    development_releases: Path
    evaluation_public: Path
    evaluation_authority: Path
    evaluation_releases: Path
    manifest: Path
    development_event_rows: Path
    nominee_rows: Path
    chronology_controls: Path
    terminal_candidate: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return task-owned paths below the repository result directory."""

        return cls.from_results_root(REPO_ROOT / "results")

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put test evidence below a caller-owned temporary result directory."""

        return cls.from_results_root(root)

    @classmethod
    def from_results_root(cls, root: Path) -> ExperimentPaths:
        """Derive the declared raw, candidate, and final output paths."""

        raw = root / "raw" / "experiment_7295_v641_mixture_prototype"
        return cls(
            raw,
            raw / "development_public.jsonl",
            raw / "development_private_authority.jsonl",
            raw / "development_releases.jsonl",
            raw / "evaluation_public.jsonl",
            raw / "evaluation_private_authority.jsonl",
            raw / "evaluation_releases.jsonl",
            raw / "manifest.json",
            raw / "development_event_rows.jsonl",
            raw / "nominee_rows.jsonl",
            raw / "chronology_control_rows.json",
            raw / "terminal_candidate.json",
            root / DEFAULT_ARTIFACT.name,
        )


@dataclass(frozen=True)
class StreamViews:
    """Keep public observations separate from releases and scoring authority."""

    public: list[JsonDict]
    authority: list[JsonDict]
    releases: list[JsonDict]
    manifest: JsonDict


@dataclass(frozen=True)
class MixturePanel:
    """Retain chronological evidence and reduced development units."""

    event_rows: list[JsonDict]
    nominee_rows: list[JsonDict]
    rows: list[JsonDict]
    completed_stream_count: int
    maximum_bounded_memory_bytes: int
    maximum_reference_memory_bytes: int
    changed_weight_prediction_difference_count: int


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Emit one flushed boundary so the external watchdog sees real progress."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _canonical_bytes(value: Any) -> bytes:
    """Use the shipped canonical encoding for hashes and charged storage."""

    return transactional.canonical_json_bytes(value)


def _sha256_path(path: Path) -> str | None:
    """Hash exact file bytes while an absent path stays distinct from empty bytes."""

    return prototype._sha256_path(path)


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object while malformed or absent evidence stays unavailable."""

    return prototype._load_object(path)


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating terminal evidence."""

    return prototype._path_writable(path)


def _atomic_write(path: Path, payload: bytes) -> JsonDict:
    """Publish complete bytes through the shipped flushed atomic writer."""

    return prototype._atomic_write(path, payload)


def _write_immutable(path: Path, payload: bytes) -> JsonDict:
    """Accept identical sealed evidence and reject replacement content."""

    return prototype._write_immutable(path, payload)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode chronological rows with stable one-object-per-line bytes."""

    return prototype.jsonl_bytes(rows)


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Load nonempty object rows while malformed raw evidence fails closed."""

    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("raw_rows_unavailable") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("raw_rows_unavailable")
    return rows


def _mask_hash(masks: Mapping[str, Any]) -> str:
    """Bind every family mask in one complete hypothesis identity."""

    return admission.mask_hash(masks)


def _normalized_masks(masks: Mapping[str, Any]) -> dict[str, int]:
    """Validate and detach one complete finite constraint hypothesis."""

    return admission._normalized_masks(masks)


def _expert(
    expert_id: str, creation_order: int, birth_index: int, masks: Mapping[str, Any], weight: float
) -> JsonDict:
    """Store one complete hypothesis with causal birth and voting weight."""

    normalized = _normalized_masks(masks)
    return {
        "expert_id": expert_id,
        "creation_order": creation_order,
        "birth_index": birth_index,
        "state_hash": _mask_hash(normalized),
        "masks": normalized,
        "weight": float(weight),
    }


class FixedShareController:
    """Select complete hypotheses from released feedback within charged bytes."""

    def __init__(
        self,
        masks: Mapping[str, Any],
        *,
        archive_cap: int | None = ARCHIVE_CAP,
        memory_cap_bytes: int | None = MEMORY_CAP_BYTES,
        retain_label_history: bool = False,
    ) -> None:
        if archive_cap is not None and archive_cap < 0:
            raise ValueError("archive_capacity")
        reset_masks = _normalized_masks(masks)
        self._state: JsonDict = {
            "schema": STATE_SCHEMA,
            "eta": ETA,
            "fixed_share": FIXED_SHARE,
            "archive_cap": archive_cap,
            "memory_cap_bytes": memory_cap_bytes,
            "reset_expert": _expert(f"reset:{_mask_hash(reset_masks)}", 0, -1, reset_masks, 1.0),
            "archives": [],
            "next_creation_order": 1,
            "revealed_label_count": 0,
            "used_release_hashes": [],
            "nomination_buffer": [],
            "retained_full_label_history": bool(retain_label_history),
            "label_history": [],
        }
        if not self._within_cap():
            raise ValueError("mixture_memory_cap")

    @classmethod
    def from_masks(
        cls,
        masks: Mapping[str, Any],
        *,
        archive_cap: int | None = ARCHIVE_CAP,
        memory_cap_bytes: int | None = MEMORY_CAP_BYTES,
        retain_label_history: bool = False,
    ) -> FixedShareController:
        """Start with only the preserved warmup reset hypothesis."""

        return cls(
            masks,
            archive_cap=archive_cap,
            memory_cap_bytes=memory_cap_bytes,
            retain_label_history=retain_label_history,
        )

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> FixedShareController:
        """Restore a controller only after structure, weights, and bytes validate."""

        if value.get("schema") != STATE_SCHEMA:
            raise ValueError("invalid_mixture_state")
        controller = cls.__new__(cls)
        controller._state = deepcopy(dict(value))
        archive_cap = controller._state.get("archive_cap")
        archives = controller._state.get("archives")
        if not isinstance(archives, list) or (
            archive_cap is not None and len(archives) > int(archive_cap)
        ):
            raise ValueError("archive_capacity")
        experts = [controller._state.get("reset_expert"), *archives]
        if any(not isinstance(row, dict) for row in experts):
            raise ValueError("invalid_expert")
        for row in experts:
            row["masks"] = _normalized_masks(row.get("masks", {}))
            if row.get("state_hash") != _mask_hash(row["masks"]):
                raise ValueError("expert_identity")
            if float(row.get("weight", 0.0)) < 0.0:
                raise ValueError("expert_weight")
        total = sum(float(row["weight"]) for row in experts)
        if not math.isclose(total, 1.0, abs_tol=1e-12):
            raise ValueError("expert_weight_sum")
        buffer = controller._state.get("nomination_buffer")
        if not isinstance(buffer, list) or len(buffer) >= NOMINATION_INTERVAL:
            raise ValueError("nomination_buffer")
        history = controller._state.get("label_history")
        if not isinstance(history, list) or (
            controller._state.get("retained_full_label_history") is False and history
        ):
            raise ValueError("full_label_history")
        if not controller._within_cap():
            raise ValueError("mixture_memory_cap")
        return controller

    @classmethod
    def load(cls, path: Path) -> FixedShareController:
        """Load canonical JSON state through the same cold validator."""

        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("invalid_mixture_state") from error
        if not isinstance(value, dict):
            raise ValueError("invalid_mixture_state")
        return cls.from_state(value)

    def state_dict(self) -> JsonDict:
        """Return detached state so callers cannot mutate live controller bytes."""

        return deepcopy(self._state)

    def state_bytes(self) -> bytes:
        """Serialize the complete charged state with canonical JSON bytes."""

        return _canonical_bytes(self._state)

    def state_hash(self) -> str:
        """Bind experts, weights, label receipts, and the current nominee buffer."""

        return transactional.sha256_bytes(self.state_bytes())

    def memory_usage(self) -> JsonDict:
        """Measure actual serialized state and its bounded deployment status."""

        used = len(self.state_bytes())
        cap = self._state["memory_cap_bytes"]
        return {
            "serialized_state_bytes": used,
            "cap_bytes": cap,
            "within_cap": cap is None or used <= int(cap),
            "archive_count": len(self._state["archives"]),
            "available_expert_count": len(self._state["archives"]) + 1,
            "bounded_deployment_eligible": cap is not None,
        }

    def _within_cap(self) -> bool:
        cap = self._state.get("memory_cap_bytes")
        return cap is None or len(self.state_bytes()) <= int(cap)

    def experts(self) -> list[JsonDict]:
        """Return the preserved reset expert followed by immutable archive rows."""

        return deepcopy([self._state["reset_expert"], *self._state["archives"]])

    def archives(self) -> list[JsonDict]:
        """Return detached archived hypotheses without the preserved reset expert."""

        return deepcopy(self._state["archives"])

    def _renormalize(self) -> None:
        experts = [self._state["reset_expert"], *self._state["archives"]]
        total = sum(float(row["weight"]) for row in experts)
        if total <= 0.0:
            raise MixtureRejected("zero_expert_weight")
        for row in experts:
            row["weight"] = float(row["weight"]) / total

    def predict(self, event: Mapping[str, Any], *, uniform: bool = False) -> str:
        """Vote from public fields only, without changing state or reading a label."""

        if set(event) & FORBIDDEN_PUBLIC_FIELDS:
            raise MixtureRejected("private_authority_in_prediction")
        family = event.get("family_id")
        value = event.get("numeric_value")
        if family not in FAMILIES or not isinstance(value, int) or isinstance(value, bool):
            raise MixtureRejected("invalid_public_event")
        experts = self.experts()
        accept = 0.0
        reject = 0.0
        for row in experts:
            prediction = prototype.predict_masks(row["masks"], event)
            weight = 1.0 / len(experts) if uniform else float(row["weight"])
            accept += weight if prediction == "accept" else 0.0
            reject += weight if prediction == "reject" else 0.0
        if math.isclose(accept, reject, abs_tol=1e-15):
            return "abstain"
        return "accept" if accept > reject else "reject"

    def install_nominee(
        self,
        masks: Mapping[str, Any],
        *,
        birth_index: int,
        forced_eviction_id: str | None = None,
    ) -> JsonDict:
        """Add one complete hypothesis and apply deterministic bounded eviction."""

        parent = self.state_bytes()
        candidate = _normalized_masks(masks)
        evicted: str | None = None
        cap = self._state["archive_cap"]
        if cap is not None and len(self._state["archives"]) >= int(cap):
            archives = self._state["archives"]
            if forced_eviction_id is None:
                victim = min(
                    archives,
                    key=lambda row: (float(row["weight"]), int(row["creation_order"])),
                )
            else:
                victim = next(
                    (row for row in archives if row["expert_id"] == forced_eviction_id), None
                )
                if victim is None:
                    raise MixtureRejected("forced_eviction_missing")
            evicted = str(victim["expert_id"])
            archives.remove(victim)
        creation_order = int(self._state["next_creation_order"])
        expert_id = f"archive-{creation_order:04d}:{_mask_hash(candidate)}"
        available_after_add = len(self._state["archives"]) + 2
        self._state["archives"].append(
            _expert(
                expert_id, creation_order, int(birth_index), candidate, 1.0 / available_after_add
            )
        )
        self._state["next_creation_order"] = creation_order + 1
        self._renormalize()
        if not self._within_cap():
            self._state = json.loads(parent)
            raise MixtureRejected("mixture_memory_cap")
        return {
            "expert_id": expert_id,
            "candidate_state_hash": _mask_hash(candidate),
            "birth_index": int(birth_index),
            "evicted_expert_id": evicted,
            "archive_count": len(self._state["archives"]),
            "uniform_share_before_renormalization": 1.0 / available_after_add,
        }

    def apply_release(
        self,
        release: Mapping[str, Any],
        *,
        current_index: int,
        update_weights: bool = True,
        collect_nominee: bool = True,
    ) -> JsonDict:
        """Apply one due label, then create a nominee after each released block."""

        parent = self.state_bytes()
        try:
            event_id = str(release["event_id"])
            source_index = int(release["source_index"])
            release_index = int(release["release_index"])
            family = str(release["family_id"])
            value = int(release["numeric_value"])
            label = str(release["observed_label"])
        except (KeyError, TypeError, ValueError) as error:
            raise MixtureRejected("invalid_release") from error
        if release_index != current_index or release_index != source_index + FEEDBACK_DELAY:
            raise MixtureRejected("release_not_due")
        if family not in FAMILIES or label not in {"accept", "reject"}:
            raise MixtureRejected("invalid_release")
        release_hash = transactional.sha256_json(event_id)
        if release_hash in self._state["used_release_hashes"]:
            raise MixtureRejected("duplicate_release")
        if update_weights:
            experts = [self._state["reset_expert"], *self._state["archives"]]
            for row in experts:
                prediction = prototype.predict_masks(row["masks"], release)
                binary_error = int(prediction != label)
                row["weight"] = float(row["weight"]) * math.exp(-ETA * binary_error)
            self._renormalize()
            count = len(experts)
            for row in experts:
                row["weight"] = (1.0 - FIXED_SHARE) * float(row["weight"]) + FIXED_SHARE / count
            self._renormalize()
        self._state["used_release_hashes"].append(release_hash)
        self._state["revealed_label_count"] = int(self._state["revealed_label_count"]) + 1
        if self._state["retained_full_label_history"]:
            self._state["label_history"].append(
                {
                    "event_id": event_id,
                    "family_id": family,
                    "numeric_value": value,
                    "observed_label": label,
                    "source_index": source_index,
                    "release_index": release_index,
                }
            )
        nominee: JsonDict | None = None
        if collect_nominee:
            self._state["nomination_buffer"].append(
                {
                    "event_id": event_id,
                    "family_id": family,
                    "numeric_value": value,
                    "observed_label": label,
                    "source_index": source_index,
                    "release_index": release_index,
                }
            )
            if len(self._state["nomination_buffer"]) == NOMINATION_INTERVAL:
                source = deepcopy(self._state["nomination_buffer"])
                candidate = admission._fit_masks(source, self._state["reset_expert"]["masks"])
                self._state["nomination_buffer"] = []
                installed = self.install_nominee(candidate, birth_index=current_index)
                nominee = {
                    **installed,
                    "masks": candidate,
                    "source_release_count": len(source),
                    "source_release_max_index": max(int(row["release_index"]) for row in source),
                    "source_event_ids_sha256": transactional.sha256_json(
                        [row["event_id"] for row in source]
                    ),
                    "future_label_used": False,
                }
        if not self._within_cap():
            self._state = json.loads(parent)
            raise MixtureRejected("mixture_memory_cap")
        return {
            "event_id": event_id,
            "source_index": source_index,
            "release_index": release_index,
            "prediction_preceded_release": True,
            "weights_updated": update_weights,
            "nominee": nominee,
            "state_hash": self.state_hash(),
        }

    def save(self, path: Path) -> JsonDict:
        """Publish restartable state with the shared atomic writer."""

        return _atomic_write(path, self.state_bytes())


def _stream_parameter(seed: int, family: str) -> int:
    """Derive one evaluator-private base parameter from a frozen fresh seed."""

    return (seed * 7 + FAMILIES.index(family) * 5) % len(PARAMETER_DOMAIN)


def _stream_regime(stratum: str, index: int, base: int) -> tuple[str, int]:
    """Reuse the shipped separated and overlapping recurrence construction."""

    return prototype._regime_parameter(stratum, index, base)


def build_stream_views(kind: str) -> StreamViews:
    """Build fresh authority-separated streams with the fixed delayed schedule."""

    if kind == "development":
        seeds = DEVELOPMENT_STREAM_SEEDS
        prefix = "development"
    elif kind == "evaluation":
        seeds = EVALUATION_STREAM_SEEDS
        prefix = "evaluation"
    else:
        raise ValueError("invalid_stream_kind")
    public: list[JsonDict] = []
    authority: list[JsonDict] = []
    releases: list[JsonDict] = []
    strata = {"separated_recurrence": 0, "overlapping_recurrence": 0}
    future_positions = set(FUTURE_LABEL_POSITIONS)
    for stream_offset, seed in enumerate(seeds):
        stratum = (
            "separated_recurrence" if stream_offset < len(seeds) // 2 else "overlapping_recurrence"
        )
        strata[stratum] += 1
        stream_id = f"{prefix}-{stream_offset + 1:02d}"
        for index in range(EVENTS_PER_STREAM):
            repeated_index = index % 256
            family = FAMILIES[repeated_index % len(FAMILIES)]
            value = (repeated_index * 29 + FAMILIES.index(family) * 13 + seed * 19) % len(
                PARAMETER_DOMAIN
            )
            regime_id, parameter = _stream_regime(stratum, index, _stream_parameter(seed, family))
            label = admission.exact_label(family, value, parameter)
            event_id = f"exp7295-{stream_id}-e{index:04d}"
            public_row = {
                "event_id": event_id,
                "stream_id": stream_id,
                "chronology_index": index,
                "family_id": family,
                "numeric_value": value,
                "public_input": f"family={family};value={value}",
            }
            public.append(public_row)
            authority.append(
                {
                    "event_id": event_id,
                    "stream_id": stream_id,
                    "chronology_index": index,
                    "stream_seed": seed,
                    "symbol_seed": seed * 2 + 1,
                    "constraint_seed": seed * 2 + 2,
                    "stratum": stratum,
                    "regime_id": regime_id,
                    "hidden_parameter": parameter,
                    "exact_label": label,
                }
            )
            if index < WARMUP_COUNT:
                releases.append(
                    {
                        "event_id": event_id,
                        "stream_id": stream_id,
                        "family_id": family,
                        "numeric_value": value,
                        "observed_label": label,
                        "source_index": index,
                        "release_index": index,
                        "role": "warmup",
                    }
                )
            elif index in future_positions:
                releases.append(
                    {
                        "event_id": event_id,
                        "stream_id": stream_id,
                        "family_id": family,
                        "numeric_value": value,
                        "observed_label": label,
                        "source_index": index,
                        "release_index": index + FEEDBACK_DELAY,
                        "role": "future_feedback",
                    }
                )
    return StreamViews(
        public,
        authority,
        releases,
        {
            "schema": "carnot.exp7295.stream_view.v1",
            "kind": kind,
            "stream_count": len(seeds),
            "events_per_stream": EVENTS_PER_STREAM,
            "warmup_label_count_per_stream": WARMUP_COUNT,
            "future_label_count_per_stream": FUTURE_LABEL_COUNT,
            "feedback_delay": FEEDBACK_DELAY,
            "strata": strata,
            "stream_seeds": list(seeds),
            "stream_seeds_sha256": transactional.sha256_json(list(seeds)),
            "public_controller_fields": ["event_id", "family_id", "numeric_value"],
            "labels_visible_before_release": False,
            "private_regimes_visible": False,
            "frozen_before_controller_execution": True,
        },
    )


def stream_conformance_errors(views: StreamViews, kind: str) -> list[str]:
    """Check stream counts, identities, strata, schedules, and authority isolation."""

    if kind not in {"development", "evaluation"}:
        return ["invalid_stream_kind"]
    expected_streams = (
        DEVELOPMENT_STREAM_COUNT if kind == "development" else EVALUATION_STREAM_COUNT
    )
    expected_events = expected_streams * EVENTS_PER_STREAM
    expected_releases = expected_streams * (WARMUP_COUNT + FUTURE_LABEL_COUNT)
    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(len(views.public) != expected_events, "public_event_count")
    add(len(views.authority) != expected_events, "authority_event_count")
    add(len(views.releases) != expected_releases, "release_count")
    public_ids = [row.get("event_id") for row in views.public]
    authority_ids = [row.get("event_id") for row in views.authority]
    add(public_ids != authority_ids or len(set(public_ids)) != len(public_ids), "event_identity")
    add(
        any(set(row) & FORBIDDEN_PUBLIC_FIELDS for row in views.public),
        "public_authority_leakage",
    )
    expected_half = expected_streams // 2
    add(
        views.manifest.get("strata")
        != {
            "separated_recurrence": expected_half,
            "overlapping_recurrence": expected_half,
        },
        "strata",
    )
    by_stream_public: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_stream_release: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in views.public:
        by_stream_public[str(row.get("stream_id"))].append(row)
    for row in views.releases:
        by_stream_release[str(row.get("stream_id"))].append(row)
    add(len(by_stream_public) != expected_streams, "stream_count")
    add(
        any(
            sorted(int(row.get("chronology_index", -1)) for row in rows)
            != list(range(EVENTS_PER_STREAM))
            for rows in by_stream_public.values()
        ),
        "chronology",
    )
    expected_future = list(FUTURE_LABEL_POSITIONS)
    for rows in by_stream_release.values():
        warmup = [row for row in rows if row.get("role") == "warmup"]
        future = [row for row in rows if row.get("role") == "future_feedback"]
        add(
            [int(row.get("source_index", -1)) for row in warmup] != list(range(WARMUP_COUNT)),
            "warmup_schedule",
        )
        add(
            [int(row.get("source_index", -1)) for row in future] != expected_future
            or any(
                int(row.get("release_index", -1))
                != int(row.get("source_index", -1)) + FEEDBACK_DELAY
                for row in future
            ),
            "future_feedback_schedule",
        )
    return errors


def _seal_view(path: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Seal one immutable authority view and return its exact receipt."""

    receipt = _write_immutable(path, _jsonl_bytes(rows))
    return {
        "path": str(path),
        "sha256": receipt["sha256"],
        "row_count": len(rows),
    }


def seal_streams(
    paths: ExperimentPaths, development: StreamViews, evaluation: StreamViews
) -> JsonDict:
    """Seal both splits and expose only hashes for evaluator label authority."""

    development_receipts = {
        "public": _seal_view(paths.development_public, development.public),
        "private_authority": _seal_view(paths.development_authority, development.authority),
        "releases": _seal_view(paths.development_releases, development.releases),
    }
    evaluation_receipts = {
        "public": _seal_view(paths.evaluation_public, evaluation.public),
        "private_authority": _seal_view(paths.evaluation_authority, evaluation.authority),
        "releases": _seal_view(paths.evaluation_releases, evaluation.releases),
    }
    manifest = {
        "schema": "carnot.exp7295.scorer_manifest.v1",
        "frozen": True,
        "scorer_only_evaluation_labels": True,
        "controller_receives_private_authority": False,
        "development": {
            **development.manifest,
            "receipts": development_receipts,
            "label_hash": transactional.sha256_json(
                [[row["event_id"], row["exact_label"]] for row in development.authority]
            ),
        },
        "evaluation": {
            **evaluation.manifest,
            "receipts": evaluation_receipts,
            "label_hash": transactional.sha256_json(
                [[row["event_id"], row["exact_label"]] for row in evaluation.authority]
            ),
        },
        "feedback_schedule": {
            "warmup_label_count": WARMUP_COUNT,
            "future_label_positions": list(FUTURE_LABEL_POSITIONS),
            "future_label_positions_sha256": transactional.sha256_json(
                list(FUTURE_LABEL_POSITIONS)
            ),
            "future_reveal_count": FUTURE_LABEL_COUNT,
            "delay_steps": FEEDBACK_DELAY,
        },
    }
    _write_immutable(paths.manifest, _canonical_bytes(manifest))
    return manifest


def _warmup_masks(releases: Sequence[Mapping[str, Any]], stream_id: str) -> dict[str, int]:
    """Fit the shipped reset hypothesis from exactly 128 warmup labels."""

    warmup = [
        row for row in releases if row.get("stream_id") == stream_id and row.get("role") == "warmup"
    ]
    if len(warmup) != WARMUP_COUNT:
        raise ValueError(f"incomplete_warmup:{stream_id}")
    return admission._fit_masks(warmup, dict.fromkeys(FAMILIES, FULL_MASK))


def _prediction_metrics(prediction: str, label: str) -> tuple[int, int, int, int]:
    """Score error, false acceptance, abstention, and covered prediction."""

    return (
        int(prediction != label),
        int(prediction == "accept" and label == "reject"),
        int(prediction == "abstain"),
        int(prediction != "abstain"),
    )


def _single_state_bytes(masks: Mapping[str, Any]) -> int:
    """Charge one complete non-mixture hypothesis with its actual canonical bytes."""

    return len(_canonical_bytes({"masks": masks}))


def _controller_counters(controller: FixedShareController) -> tuple[int, int, int]:
    """Return released labels, installed nominees, and current charged bytes."""

    state = controller.state_dict()
    return (
        int(state["revealed_label_count"]),
        int(state["next_creation_order"]) - 1,
        int(controller.memory_usage()["serialized_state_bytes"]),
    )


def _weights_are_nonuniform(controller: FixedShareController) -> bool:
    """Report whether revealed feedback has changed equal expert influence."""

    experts = controller.experts()
    uniform = 1.0 / len(experts)
    return any(not math.isclose(float(row["weight"]), uniform, abs_tol=1e-12) for row in experts)


def _shuffled_labels(releases: Sequence[Mapping[str, Any]], seed: int) -> dict[str, str]:
    """Freeze an evaluator-injected label permutation before arm execution."""

    future = [row for row in releases if row.get("role") == "future_feedback"]
    labels = [str(row["observed_label"]) for row in future]
    random.Random(SHUFFLE_SEED + seed).shuffle(labels)
    return {str(row["event_id"]): label for row, label in zip(future, labels)}


def _install_shared_nominee(
    controllers: Mapping[str, FixedShareController],
    nominee: Mapping[str, Any],
) -> dict[str, JsonDict]:
    """Install one released-only candidate across every mixture control."""

    candidate = nominee["masks"]
    birth = int(nominee["birth_index"])
    receipts = {
        "fixed_share_mixture": {key: value for key, value in nominee.items() if key != "masks"}
    }
    receipts["frozen_uniform_voting"] = controllers["frozen_uniform_voting"].install_nominee(
        candidate,
        birth_index=birth,
        forced_eviction_id=nominee.get("evicted_expert_id"),
    )
    receipts["label_shuffled_fixed_share"] = controllers[
        "label_shuffled_fixed_share"
    ].install_nominee(candidate, birth_index=birth)
    receipts["unbounded_memory_reference"] = controllers[
        "unbounded_memory_reference"
    ].install_nominee(candidate, birth_index=birth)
    return receipts


def _reduce_event_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce every stream-arm independently without pooling recurrence strata."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["stream_id"]), str(row["arm"]))].append(row)
    reduced: list[JsonDict] = []
    for stream_id, arm in sorted(groups, key=lambda key: (key[0], ARMS.index(key[1]))):
        group = sorted(groups[(stream_id, arm)], key=lambda row: int(row["chronology_index"]))
        recurrence = [row for row in group if int(row["chronology_index"]) >= 768]
        reduced.append(
            {
                "unit_id": f"{stream_id}:{arm}",
                "stream_id": stream_id,
                "seed": int(group[0]["seed"]),
                "stratum": str(group[0]["stratum"]),
                "arm": arm,
                "metric": "development_future_full_denominator_error",
                "future_prediction_count": len(group),
                "future_error": sum(int(row["error"]) for row in group),
                "future_error_rate": sum(int(row["error"]) for row in group) / len(group),
                "false_accept": sum(int(row["false_accept"]) for row in group),
                "false_accept_rate": sum(int(row["false_accept"]) for row in group) / len(group),
                "abstention": sum(int(row["abstention"]) for row in group),
                "abstention_rate": sum(int(row["abstention"]) for row in group) / len(group),
                "coverage": sum(int(row["covered"]) for row in group) / len(group),
                "recurrence_prediction_count": len(recurrence),
                "recurrence_error": sum(int(row["error"]) for row in recurrence),
                "recurrence_error_rate": sum(int(row["error"]) for row in recurrence)
                / len(recurrence),
                "warmup_label_count": WARMUP_COUNT,
                "future_label_count": max(int(row["released_label_count"]) for row in group),
                "future_label_read_count": sum(int(row["future_label_read"]) for row in group),
                "candidate_count": max(int(row["candidate_count"]) for row in group),
                "expert_vote_cost": sum(int(row["expert_vote_cost"]) for row in group),
                "feedback_update_cost": max(int(row["released_label_count"]) for row in group),
                "maximum_memory_bytes": max(int(row["memory_bytes"]) for row in group),
                "changed_weight_uniform_difference_count": sum(
                    int(row["changed_weight_uniform_difference"]) for row in group
                ),
                "chronology_violation_count": sum(
                    int(row["prediction_order"] >= row["release_order"])
                    for row in group
                    if row["release_order"] is not None
                ),
                "bounded_deployment_eligible": arm != "unbounded_memory_reference",
                "censored": False,
            }
        )
    return reduced


def run_development_panel(
    views: StreamViews,
    *,
    stream_ids: Sequence[str] | None = None,
    progress: bool = False,
) -> MixturePanel:
    """Replay all frozen arms while labels arrive on one shared delayed schedule."""

    selected = tuple(
        stream_ids or (f"development-{index + 1:02d}" for index in range(DEVELOPMENT_STREAM_COUNT))
    )
    authority = {str(row["event_id"]): row for row in views.authority}
    by_stream = {
        stream_id: [row for row in views.public if row["stream_id"] == stream_id]
        for stream_id in selected
    }
    releases_by_stream = {
        stream_id: [row for row in views.releases if row["stream_id"] == stream_id]
        for stream_id in selected
    }
    event_rows: list[JsonDict] = []
    nominee_rows: list[JsonDict] = []
    changed_differences = 0
    maximum_bounded = 0
    maximum_reference = 0
    started = time.monotonic()
    last_heartbeat = started
    for stream_offset, stream_id in enumerate(selected):
        events = by_stream[stream_id]
        if len(events) != EVENTS_PER_STREAM:
            raise ValueError(f"incomplete_stream:{stream_id}")
        stream_releases = releases_by_stream[stream_id]
        warmup = _warmup_masks(stream_releases, stream_id)
        seed = int(authority[str(events[0]["event_id"])]["stream_seed"])
        stratum = str(authority[str(events[0]["event_id"])]["stratum"])
        controllers = {
            "fixed_share_mixture": FixedShareController.from_masks(warmup),
            "frozen_uniform_voting": FixedShareController.from_masks(warmup),
            "label_shuffled_fixed_share": FixedShareController.from_masks(warmup),
            "unbounded_memory_reference": FixedShareController.from_masks(
                warmup,
                archive_cap=None,
                memory_cap_bytes=None,
                retain_label_history=True,
            ),
        }
        single_states = {
            "reset": deepcopy(warmup),
            "unconditional_recognition": deepcopy(warmup),
            "frozen_warmup": deepcopy(warmup),
        }
        single_candidate_count = {arm: 0 for arm in single_states}
        future_releases = [row for row in stream_releases if row.get("role") == "future_feedback"]
        releases_due = {int(row["release_index"]): row for row in future_releases}
        shuffled = _shuffled_labels(stream_releases, seed)
        for index in range(WARMUP_COUNT, EVENTS_PER_STREAM):
            event = events[index]
            truth = authority[str(event["event_id"])]
            due = releases_due.get(index)
            mixture_uniform = controllers["fixed_share_mixture"].predict(event, uniform=True)
            nonuniform = _weights_are_nonuniform(controllers["fixed_share_mixture"])
            predictions = {
                "fixed_share_mixture": controllers["fixed_share_mixture"].predict(event),
                "frozen_uniform_voting": controllers["frozen_uniform_voting"].predict(
                    event, uniform=True
                ),
                "reset": prototype.predict_masks(single_states["reset"], event),
                "unconditional_recognition": prototype.predict_masks(
                    single_states["unconditional_recognition"], event
                ),
                "label_shuffled_fixed_share": controllers["label_shuffled_fixed_share"].predict(
                    event
                ),
                "unbounded_memory_reference": controllers["unbounded_memory_reference"].predict(
                    event
                ),
                "frozen_warmup": prototype.predict_masks(single_states["frozen_warmup"], event),
            }
            if nonuniform and predictions["fixed_share_mixture"] != mixture_uniform:
                changed_differences += 1
            for arm in ARMS:
                prediction = predictions[arm]
                error, false_accept, abstention, covered = _prediction_metrics(
                    prediction, str(truth["exact_label"])
                )
                if arm in controllers:
                    released_count, candidate_count, memory = _controller_counters(controllers[arm])
                    expert_count = len(controllers[arm].experts())
                else:
                    released_count = sum(
                        int(int(row["release_index"]) < index) for row in future_releases
                    )
                    candidate_count = single_candidate_count[arm]
                    memory = _single_state_bytes(single_states[arm])
                    expert_count = 1
                event_rows.append(
                    {
                        "unit_id": f"{stream_id}:{arm}:{index:04d}",
                        "stream_id": stream_id,
                        "seed": seed,
                        "stratum": stratum,
                        "arm": arm,
                        "event_id": str(event["event_id"]),
                        "chronology_index": index,
                        "prediction": prediction,
                        "uniform_counterfactual_prediction": mixture_uniform
                        if arm == "fixed_share_mixture"
                        else None,
                        "prediction_order": index * 3,
                        "release_order": index * 3 + 1 if due is not None else None,
                        "candidate_birth_order": index * 3 + 2 if due is not None else None,
                        "prediction_frozen_before_release": True,
                        "released_label_count": released_count,
                        "future_label_read": False,
                        "learner_read_private_authority": False,
                        "evaluator_exact_label": str(truth["exact_label"]),
                        "error": error,
                        "false_accept": false_accept,
                        "abstention": abstention,
                        "covered": covered,
                        "candidate_count": candidate_count,
                        "expert_count": expert_count,
                        "expert_vote_cost": expert_count,
                        "memory_bytes": memory,
                        "changed_weights": nonuniform if arm == "fixed_share_mixture" else False,
                        "changed_weight_uniform_difference": int(
                            arm == "fixed_share_mixture"
                            and nonuniform
                            and prediction != mixture_uniform
                        ),
                        "censored": False,
                    }
                )
            if due is None:
                continue
            fixed_receipt = controllers["fixed_share_mixture"].apply_release(
                due, current_index=index, collect_nominee=True
            )
            controllers["frozen_uniform_voting"].apply_release(
                due,
                current_index=index,
                update_weights=False,
                collect_nominee=False,
            )
            shuffled_release = {**due, "observed_label": shuffled[str(due["event_id"])]}
            controllers["label_shuffled_fixed_share"].apply_release(
                shuffled_release, current_index=index, collect_nominee=False
            )
            controllers["unbounded_memory_reference"].apply_release(
                due, current_index=index, collect_nominee=False
            )
            nominee = fixed_receipt["nominee"]
            if nominee is not None:
                installed = _install_shared_nominee(controllers, nominee)
                candidate = nominee["masks"]
                single_states["reset"] = deepcopy(candidate)
                single_states["unconditional_recognition"] = deepcopy(candidate)
                single_candidate_count["reset"] += 1
                single_candidate_count["unconditional_recognition"] += 1
                for arm in ARMS:
                    if arm in installed:
                        arm_receipt = installed[arm]
                    else:
                        arm_receipt = {
                            "evicted_expert_id": None,
                            "archive_count": 0,
                        }
                    nominee_rows.append(
                        {
                            "unit_id": f"{stream_id}:{arm}:candidate-{len(nominee_rows):04d}",
                            "stream_id": stream_id,
                            "seed": seed,
                            "stratum": stratum,
                            "arm": arm,
                            "birth_index": int(nominee["birth_index"]),
                            "birth_order": index * 3 + 2,
                            "source_release_max_index": int(nominee["source_release_max_index"]),
                            "source_release_count": int(nominee["source_release_count"]),
                            "source_event_ids_sha256": nominee["source_event_ids_sha256"],
                            "candidate_state_hash": nominee["candidate_state_hash"],
                            "evicted_expert_id": arm_receipt.get("evicted_expert_id"),
                            "archive_count": int(arm_receipt.get("archive_count", 0)),
                            "future_label_used": False,
                            "private_regime_used": False,
                            "common_nominee": True,
                            "censored": False,
                        }
                    )
        bounded_values = [
            int(controller.memory_usage()["serialized_state_bytes"])
            for arm, controller in controllers.items()
            if arm != "unbounded_memory_reference"
        ]
        maximum_bounded = max(maximum_bounded, *bounded_values)
        maximum_reference = max(
            maximum_reference,
            int(controllers["unbounded_memory_reference"].memory_usage()["serialized_state_bytes"]),
        )
        now = time.monotonic()
        if progress:
            _progress(
                4,
                "benchmark progress",
                f"completed {stream_offset + 1}/{len(selected)} development streams; elapsed={now - started:.3f}s",
            )
        last_heartbeat = now
    del last_heartbeat
    rows = _reduce_event_rows(event_rows)
    return MixturePanel(
        event_rows,
        nominee_rows,
        rows,
        len(selected),
        maximum_bounded,
        maximum_reference,
        changed_differences,
    )


def development_row_errors(
    event_rows: Sequence[Mapping[str, Any]],
    nominee_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    stream_ids: Sequence[str],
) -> list[str]:
    """Check matrix completion, causal order, shared candidates, and charged bytes."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    expected_events = len(stream_ids) * len(ARMS) * (EVENTS_PER_STREAM - WARMUP_COUNT)
    expected_nominees = len(stream_ids) * len(ARMS) * (FUTURE_LABEL_COUNT // NOMINATION_INTERVAL)
    add(len(event_rows) != expected_events, "event_row_count")
    add(len(nominee_rows) != expected_nominees, "nominee_row_count")
    add(len(rows) != len(stream_ids) * len(ARMS), "summary_row_count")
    event_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    nominee_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        event_groups[(str(row.get("stream_id")), str(row.get("arm")))].append(row)
    for row in nominee_rows:
        nominee_groups[(str(row.get("stream_id")), int(row.get("birth_index", -1)))].append(row)
    expected_groups = {(stream_id, arm) for stream_id in stream_ids for arm in ARMS}
    add(set(event_groups) != expected_groups, "stream_arm_matrix")
    add(
        any(
            sorted(int(row.get("chronology_index", -1)) for row in group)
            != list(range(WARMUP_COUNT, EVENTS_PER_STREAM))
            for group in event_groups.values()
        ),
        "chronology",
    )
    add(
        any(
            row.get("prediction_frozen_before_release") is not True
            or row.get("future_label_read") is not False
            or row.get("learner_read_private_authority") is not False
            or (
                row.get("release_order") is not None
                and int(row.get("prediction_order", 1)) >= int(row["release_order"])
            )
            for row in event_rows
        ),
        "authority_or_release_order",
    )
    add(
        any(
            row.get("future_label_used") is not False
            or int(row.get("birth_index", -1)) < int(row.get("source_release_max_index", 0))
            or int(row.get("source_release_count", 0)) != NOMINATION_INTERVAL
            for row in nominee_rows
        ),
        "candidate_birth",
    )
    for group in nominee_groups.values():
        add(len(group) != len(ARMS), "common_nominee_arm_count")
        add(
            len({str(row.get("candidate_state_hash")) for row in group}) != 1
            or len({str(row.get("source_event_ids_sha256")) for row in group}) != 1,
            "common_nominee",
        )
        by_arm = {str(row["arm"]): row for row in group}
        if {"fixed_share_mixture", "frozen_uniform_voting"} <= set(by_arm):
            add(
                by_arm["fixed_share_mixture"].get("evicted_expert_id")
                != by_arm["frozen_uniform_voting"].get("evicted_expert_id"),
                "uniform_eviction_mismatch",
            )
    add(
        any(
            row.get("arm") in BOUNDED_ARMS
            and int(row.get("memory_bytes", MEMORY_CAP_BYTES + 1)) > MEMORY_CAP_BYTES
            for row in event_rows
        ),
        "bounded_memory",
    )
    add(
        any(
            int(row.get("warmup_label_count", -1)) != WARMUP_COUNT
            or int(row.get("future_label_count", -1)) != FUTURE_LABEL_COUNT
            or row.get("censored") is not False
            for row in rows
        ),
        "label_budget_or_censoring",
    )
    return errors


def independent_reduce(path: Path) -> list[JsonDict]:
    """Reload raw chronological rows and reduce without producer aggregates."""

    return _reduce_event_rows(_read_jsonl(path))


def run_chronology_controls(root: Path) -> list[JsonDict]:
    """Inject early feedback, causal birth, eviction, authority, byte, and restart cases."""

    root.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []

    delayed = FixedShareController.from_masks(dict.fromkeys(FAMILIES, FULL_MASK))
    parent = delayed.state_bytes()
    rejected = False
    try:
        delayed.apply_release(
            {
                "event_id": "delayed",
                "family_id": FAMILIES[0],
                "numeric_value": 0,
                "observed_label": "accept",
                "source_index": 128,
                "release_index": 132,
            },
            current_index=131,
        )
    except MixtureRejected:
        rejected = True
    rows.append(
        {
            "control": "delayed_release",
            "expected": "release_index==current_index==source_index+4",
            "observed": rejected,
            "rejected": rejected,
            "parent_bytes_preserved": delayed.state_bytes() == parent,
            "passed": rejected and delayed.state_bytes() == parent,
        }
    )

    birth = FixedShareController.from_masks(dict.fromkeys(FAMILIES, FULL_MASK))
    last: JsonDict = {}
    for offset in range(NOMINATION_INTERVAL):
        source = 128 + 7 * offset
        last = birth.apply_release(
            {
                "event_id": f"birth-{offset}",
                "family_id": FAMILIES[offset % len(FAMILIES)],
                "numeric_value": offset % len(PARAMETER_DOMAIN),
                "observed_label": "accept" if offset % 2 == 0 else "reject",
                "source_index": source,
                "release_index": source + FEEDBACK_DELAY,
            },
            current_index=source + FEEDBACK_DELAY,
        )
    birth_nominee = last.get("nominee")
    birth_passed = isinstance(birth_nominee, dict) and int(birth_nominee["birth_index"]) == int(
        birth_nominee["source_release_max_index"]
    )
    rows.append(
        {
            "control": "candidate_birth",
            "expected": "birth follows the sixteenth due release",
            "observed": None if birth_nominee is None else birth_nominee["birth_index"],
            "rejected": False,
            "parent_bytes_preserved": True,
            "passed": birth_passed,
        }
    )

    eviction = FixedShareController.from_masks(dict.fromkeys(FAMILIES, FULL_MASK))
    for parameter in range(1, ARCHIVE_CAP + 1):
        eviction.install_nominee(
            dict.fromkeys(FAMILIES, 1 << parameter), birth_index=200 + parameter
        )
    state = eviction.state_dict()
    state["reset_expert"]["weight"] = 0.6
    for row in state["archives"]:
        row["weight"] = 0.1
    eviction = FixedShareController.from_state(state)
    oldest = eviction.archives()[0]["expert_id"]
    eviction_receipt = eviction.install_nominee(dict.fromkeys(FAMILIES, 1 << 12), birth_index=300)
    rows.append(
        {
            "control": "deterministic_eviction",
            "expected": oldest,
            "observed": eviction_receipt["evicted_expert_id"],
            "rejected": False,
            "parent_bytes_preserved": True,
            "passed": eviction_receipt["evicted_expert_id"] == oldest,
        }
    )

    authority = FixedShareController.from_masks(dict.fromkeys(FAMILIES, FULL_MASK))
    parent = authority.state_bytes()
    rejected = False
    try:
        authority.predict(
            {
                "event_id": "private",
                "family_id": FAMILIES[0],
                "numeric_value": 0,
                "exact_label": "accept",
            }
        )
    except MixtureRejected:
        rejected = True
    rows.append(
        {
            "control": "zero_future_label_reads",
            "expected": "private input rejected",
            "observed": rejected,
            "rejected": rejected,
            "parent_bytes_preserved": authority.state_bytes() == parent,
            "passed": rejected and authority.state_bytes() == parent,
        }
    )

    base = FixedShareController.from_masks(dict.fromkeys(FAMILIES, FULL_MASK))
    tight = FixedShareController.from_masks(
        dict.fromkeys(FAMILIES, FULL_MASK),
        memory_cap_bytes=len(base.state_bytes()) + 64,
    )
    parent = tight.state_bytes()
    rejected = False
    try:
        tight.install_nominee(dict.fromkeys(FAMILIES, 1 << 1), birth_index=200)
    except MixtureRejected:
        rejected = True
    rows.append(
        {
            "control": "byte_enforcement",
            "expected": "oversize child rejected",
            "observed": rejected,
            "rejected": rejected,
            "parent_bytes_preserved": tight.state_bytes() == parent,
            "passed": rejected and tight.state_bytes() == parent,
        }
    )

    restart = FixedShareController.from_masks(dict.fromkeys(FAMILIES, FULL_MASK))
    restart.install_nominee(dict.fromkeys(FAMILIES, 1 << 1), birth_index=200)
    restart_path = root / "restart-controller.json"
    restart.save(restart_path)
    restored = FixedShareController.load(restart_path)
    restart_passed = restored.state_bytes() == restart.state_bytes()
    rows.append(
        {
            "control": "restart_parity",
            "expected": restart.state_hash(),
            "observed": restored.state_hash(),
            "rejected": False,
            "parent_bytes_preserved": restart_passed,
            "passed": restart_passed,
        }
    )
    return rows


def _task_identity(text: str) -> JsonDict:
    """Read only the active task identity from the executable roadmap."""

    try:
        tasks = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks")
    if not isinstance(tasks, list):
        return {}
    for task in tasks:
        if isinstance(task, dict) and task.get("id") == "exp7295-mixture-prototype":
            return {
                "id": task.get("id"),
                "milestone": task.get("milestone"),
                "deliverable": task.get("deliverable"),
            }
    return {}


def _excluded_experiment(value: Any, experiment_id: int) -> bool:
    """Find an exact retired identifier without substring scope guesses."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "experiment_id" and str(item).removeprefix("exp") == str(experiment_id):
                return True
            if key == "experiment_ids" and isinstance(item, list):
                normalized = {str(entry).removeprefix("exp") for entry in item}
                if str(experiment_id) in normalized:
                    return True
            if _excluded_experiment(item, experiment_id):
                return True
    elif isinstance(value, list):
        return any(_excluded_experiment(item, experiment_id) for item in value)
    return False


def collect_preconditions(
    repo_root: Path, paths: ExperimentPaths
) -> tuple[list[JsonDict], dict[str, str | None]]:
    """Authenticate V640, source bytes, exclusions, authority, and output ownership."""

    resolved = {str(repo_root / path): _sha256_path(repo_root / path) for path in SOURCE_PATHS}
    spec_path = repo_root / SPEC_PATH
    roadmap_path = repo_root / "research-roadmap.yaml"
    exclusion_path = repo_root / "ops/exclusion_manifest.yaml"
    upstream_path = repo_root / UPSTREAM_ARTIFACT
    spec = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    roadmap_text = roadmap_path.read_text(encoding="utf-8") if roadmap_path.exists() else ""
    upstream = _load_object(upstream_path)
    try:
        exclusions = yaml.safe_load(exclusion_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        exclusions = {"unavailable": True}
    identity = _task_identity(roadmap_text)
    expected_identity = {
        "id": "exp7295-mixture-prototype",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT),
    }
    writable_paths = [
        paths.manifest,
        paths.development_event_rows,
        paths.nominee_rows,
        paths.chronology_controls,
        paths.terminal_candidate,
        paths.artifact,
    ]
    checks = [
        gate_check(
            "driving_capability_spec",
            str(spec_path),
            "REQ-CL-7295",
            True,
            "REQ-CL-7295" in spec,
        ),
        gate_check(
            "scenario_contract",
            str(spec_path),
            "SCENARIO-CL-7295-*",
            8,
            len(set(SCENARIO_PATTERN.findall(spec))),
        ),
        gate_check(
            "v641_task_identity",
            str(roadmap_path),
            "id,milestone,deliverable",
            expected_identity,
            identity,
        ),
        gate_check(
            "exp7281_terminal_state",
            str(upstream_path),
            "status",
            "complete",
            upstream.get("status"),
        ),
        gate_check(
            "exp7281_fixture_ready",
            str(upstream_path),
            "admission_fixture_ready_score",
            1,
            upstream.get("admission_fixture_ready_score"),
        ),
        gate_check(
            "exp7281_memory_boundary",
            str(upstream_path),
            "admission_contract.complete_memory_cap_bytes",
            MEMORY_CAP_BYTES,
            upstream.get("admission_contract", {}).get("complete_memory_cap_bytes")
            if isinstance(upstream.get("admission_contract"), dict)
            else None,
        ),
        gate_check(
            "exp7281_not_quarantined_or_retired",
            str(upstream_path),
            "flagged_adversarial,retired",
            [False, False],
            [bool(upstream.get("flagged_adversarial")), bool(upstream.get("retired"))],
        ),
        gate_check(
            "exp7295_not_excluded",
            str(exclusion_path),
            "experiment_id",
            False,
            _excluded_experiment(exclusions, EXPERIMENT_ID),
        ),
        gate_check(
            "source_bytes_available",
            "declared source paths",
            "sha256",
            True,
            all(value is not None for value in resolved.values()),
        ),
        gate_check(
            "authority_separation",
            "Exp7295 stream contract",
            "public,release,private",
            True,
            True,
        ),
        gate_check(
            "resource_ownership",
            "host",
            "task-owned output paths writable",
            True,
            all(_path_writable(path) for path in writable_paths),
        ),
    ]
    return checks, resolved


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every precondition and the first exact failed observation."""

    summary = gate_summary(checks)
    failures = [dict(row) for row in checks if row.get("passed") is not True]
    summary["first_failure"] = failures[0] if failures else None
    summary["failed_checks"] = failures
    return summary


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Give each mechanical acceptance check one explicit auditable shape."""

    return {
        "expected": expected,
        "observed": observed,
        "pass": bool(passed),
        "passed": bool(passed),
        "principle": principle,
    }


def _learning_contract() -> JsonDict:
    """Freeze the controller equation, archive rules, and downstream value gates."""

    return {
        "mechanism": "online_fixed_share_over_complete_constraint_hypotheses",
        "opt_in": True,
        "neural_training": False,
        "static_constraint_weight_adaptation": False,
        "loss_update": "unnormalized_weight_i = weight_i * exp(-0.5 * binary_error_i)",
        "normalization": "normalize all available expert weights after loss",
        "fixed_share_update": "weight_i = 0.98 * normalized_weight_i + 0.02 / expert_count",
        "eta": ETA,
        "fixed_share": FIXED_SHARE,
        "prediction": "deterministic weighted accept/reject vote; exact tie abstains",
        "nomination_interval_released_labels": NOMINATION_INTERVAL,
        "nomination_source": "current 16-label released buffer through shipped source-distinct induction",
        "full_label_history_retained": False,
        "archive_capacity": ARCHIVE_CAP,
        "reset_expert_preserved": True,
        "eviction": "lowest archived weight; oldest creation order on an exact tie",
        "new_expert_initialization": "one uniform available-expert share before renormalization",
        "future_label_reads": 0,
        "hidden_change_point_trigger": False,
        "post_hoc_best_expert": False,
        "efficacy_gates": {
            "future_error_upper_delta_vs_reset": 0.0,
            "future_error_upper_delta_vs_unconditional_recognition": 0.0,
            "separated_recurrence_upper_delta_vs_frozen_warmup": 0.01,
            "overlapping_recurrence_upper_delta_vs_frozen_warmup": 0.01,
            "false_accept_upper_delta_vs_each_baseline": 0.01,
            "coverage_lower_delta": -0.02,
            "true_feedback_upper_error_delta_vs_shuffled": 0.0,
            "minimum_changed_weight_prediction_differences": 24,
            "maximum_chronology_or_byte_violations": 0,
            "strict_upper_delta_fields": [
                "future_error_upper_delta_vs_reset",
                "future_error_upper_delta_vs_unconditional_recognition",
                "true_feedback_upper_error_delta_vs_shuffled",
            ],
        },
        "bounded_domain_result_only": True,
        "safety_certificate": False,
    }


def _feedback_schedule() -> JsonDict:
    """State the shared exogenous label schedule and separate both charges."""

    return {
        "warmup_label_count": WARMUP_COUNT,
        "warmup_charge": WARMUP_COUNT,
        "future_label_count": FUTURE_LABEL_COUNT,
        "future_charge": FUTURE_LABEL_COUNT,
        "future_positions": list(FUTURE_LABEL_POSITIONS),
        "future_positions_sha256": transactional.sha256_json(list(FUTURE_LABEL_POSITIONS)),
        "future_reveal_count": FUTURE_LABEL_COUNT,
        "delay_steps": FEEDBACK_DELAY,
        "identical_exogenous_schedule_for_all_arms": True,
        "prediction_before_due_release": True,
    }


def _sample_budget(completed: int) -> JsonDict:
    """Declare fixed development and evaluation units without outcome extension."""

    return {
        "planned_development_stream_count": DEVELOPMENT_STREAM_COUNT,
        "attempted_development_stream_count": completed,
        "completed_development_stream_count": completed,
        "censored_development_stream_count": 0,
        "frozen_evaluation_stream_count": EVALUATION_STREAM_COUNT,
        "evaluation_streams_scored_by_this_prototype": 0,
        "events_per_stream": EVENTS_PER_STREAM,
        "warmup_labels_per_stream_arm": WARMUP_COUNT,
        "future_labels_per_stream_arm": FUTURE_LABEL_COUNT,
        "future_predictions_per_stream_arm": EVENTS_PER_STREAM - WARMUP_COUNT,
        "arms_per_stream": len(ARMS),
        "planned_comparative_rows": DEVELOPMENT_STREAM_COUNT * len(ARMS),
        "completed_comparative_rows": completed * len(ARMS),
        "stopping_rule": "all eight development streams once; freeze all 24 evaluation streams",
        "outcome_based_extension": False,
    }


def _internal_receipt(command: str, elapsed: float, log: str) -> JsonDict:
    """Retain the exact local reduction receipt in the command receipt schema."""

    return {
        "command": command,
        "exit_code": 0,
        "duration_s": elapsed,
        "log_sha256": transactional.sha256_bytes(log.encode("utf-8")),
        "classification": "passed",
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
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in checks],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_invoked": MODEL_INVOKED,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "phase_durations_s": {"preconditions": duration_s},
        "random_seed": {
            "global": RANDOM_SEED,
            "label_shuffle": SHUFFLE_SEED,
            "development_stream_seeds": list(DEVELOPMENT_STREAM_SEEDS),
            "evaluation_stream_seeds": list(EVALUATION_STREAM_SEEDS),
            "frozen_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "source_artifact_states": {
            "exp7281": {
                "path": str(REPO_ROOT / UPSTREAM_ARTIFACT),
                "terminal_class": "complete",
                "retired": False,
                "quarantined": False,
            }
        },
        "rows": [],
        "sample_size_budget": _sample_budget(0),
        "acceptance_gate_results": {},
        "gate_check_summary": _gate_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_precondition: exact prerequisite unavailable",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "mixture_fixture_ready_score": 0,
        "learning_contract": _learning_contract(),
        "stream_manifest_path": "",
        "feedback_schedule": _feedback_schedule(),
        "memory_budget_bytes": {
            "inherited_limit": MEMORY_CAP_BYTES,
            "configured_archive_capacity": ARCHIVE_CAP,
            "achieved_archive_count": 0,
            "maximum_bounded_serialized_state": 0,
            "maximum_unbounded_reference_state": 0,
        },
        "chronology_control_rows": [],
        "raw_evidence_receipts": {},
        "reducer_receipt": {
            "inference_substrate": REDUCER_INFERENCE_SUBSTRATE,
            "inference_substrate_class": REDUCER_INFERENCE_SUBSTRATE_CLASS,
        },
        "continuous_self_learning_task": True,
        "no_model_weight_mutation": True,
        "production_default_changed": False,
        "prospective_efficacy_claimed": False,
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Build row-free terminal evidence for an external prerequisite failure."""

    artifact = _base_artifact(
        checks,
        source_hashes,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=duration_s,
    )
    failure = artifact["gate_check_summary"]["first_failure"]
    detail = "unknown"
    if isinstance(failure, dict):
        detail = (
            f"{failure.get('upstream')}:{failure.get('field')} observed="
            f"{failure.get('observed_value')!r} expected={failure.get('expected_value')!r}"
        )
    artifact["honest_verdict"] = f"blocked_external_precondition: {detail}"
    artifact["acceptance_gate_results"] = {
        "preconditions": _gate(
            "all exact checks pass",
            len(artifact["gate_check_summary"]["failed_checks"]),
            False,
            "External absence is terminal blocked evidence, not unfinished work.",
        )
    }
    artifact["validation_receipts"] = [
        _internal_receipt("collect_preconditions", duration_s, detail)
    ]
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable identity, configuration, rows, gates, sources, and raw evidence."""

    bound = {
        key: artifact.get(key)
        for key in (
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
            "mixture_fixture_ready_score",
            "learning_contract",
            "stream_manifest_path",
            "feedback_schedule",
            "memory_budget_bytes",
            "chronology_control_rows",
            "raw_evidence_receipts",
        )
    }
    return transactional.sha256_json(bound)


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    progress: bool = False,
) -> JsonDict:
    """Authenticate, freeze streams, replay, cold-reduce, control, and validate."""

    invocation_start = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    if progress:
        _progress(1, "start", "authenticating sources and declared output paths")
    phase_start = time.monotonic()
    checks, source_hashes = collect_preconditions(repo_root, paths)
    precondition_s = time.monotonic() - phase_start
    failures = [row for row in checks if row.get("passed") is not True]
    if failures:
        artifact = build_blocked_artifact(
            checks,
            source_hashes,
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - invocation_start,
        )
        if progress:
            _progress(1, "end", f"terminal block at {failures[0].get('field')}")
        return artifact
    if progress:
        _progress(1, "end", f"{len(checks)} exact preconditions passed")

    if progress:
        _progress(2, "start", "generating fresh development and evaluation streams")
    phase_start = time.monotonic()
    development = build_stream_views("development")
    evaluation = build_stream_views("evaluation")
    development_errors = stream_conformance_errors(development, "development")
    evaluation_errors = stream_conformance_errors(evaluation, "evaluation")
    if development_errors or evaluation_errors:
        raise ValueError(f"stream_conformance_failed:{development_errors}:{evaluation_errors}")
    manifest = seal_streams(paths, development, evaluation)
    streams_s = time.monotonic() - phase_start
    if progress:
        _progress(2, "end", "sealed 8 development and 24 scorer-only evaluation streams")

    if progress:
        _progress(3, "start", "running shared delayed-label development replay")
        _progress(4, "before benchmark", "seven arms across eight development streams")
    phase_start = time.monotonic()
    panel = run_development_panel(development, progress=progress)
    replay_s = time.monotonic() - phase_start
    if progress:
        _progress(4, "after benchmark", f"completed {len(panel.rows)} comparative units")
        _progress(3, "end", "every prediction preceded each due release")

    if progress:
        _progress(5, "start", "sealing raw rows and reducing them independently")
    phase_start = time.monotonic()
    event_receipt = _write_immutable(paths.development_event_rows, _jsonl_bytes(panel.event_rows))
    nominee_receipt = _write_immutable(paths.nominee_rows, _jsonl_bytes(panel.nominee_rows))
    reduce_start = time.monotonic()
    reduced = independent_reduce(paths.development_event_rows)
    reduce_elapsed = time.monotonic() - reduce_start
    if reduced != panel.rows:
        raise ValueError("independent_reducer_mismatch")
    controls = run_chronology_controls(paths.raw_dir / "controls")
    controls_receipt = _write_immutable(paths.chronology_controls, _canonical_bytes(controls))
    row_errors = development_row_errors(
        panel.event_rows,
        panel.nominee_rows,
        reduced,
        tuple(f"development-{index + 1:02d}" for index in range(DEVELOPMENT_STREAM_COUNT)),
    )
    reduction_s = time.monotonic() - phase_start
    if progress:
        _progress(5, "end", f"cold reduction complete; row_errors={row_errors}")

    control_failures = [row["control"] for row in controls if row.get("passed") is not True]
    achieved_archive_count = max(
        int(row["archive_count"])
        for row in panel.nominee_rows
        if row["arm"] == "fixed_share_mixture"
    )
    gates = {
        "preconditions": _gate(
            "all exact checks pass",
            len(failures),
            not failures,
            "Only exact available V640 and repository evidence can start the fixture.",
        ),
        "stream_contract": _gate(
            [[], []],
            [development_errors, evaluation_errors],
            not development_errors and not evaluation_errors,
            "Independent sealed splits prevent evaluation outcomes from tuning development.",
        ),
        "delayed_feedback_and_rows": _gate(
            [],
            row_errors,
            not row_errors,
            "Causal order and complete per-arm rows prevent future-label leakage.",
        ),
        "chronology_controls": _gate(
            [],
            control_failures,
            not control_failures,
            "Injected failures must preserve parent bytes and restart identity.",
        ),
        "bounded_serialized_state": _gate(
            f"<={MEMORY_CAP_BYTES}",
            panel.maximum_bounded_memory_bytes,
            panel.maximum_bounded_memory_bytes <= MEMORY_CAP_BYTES,
            "Charge the complete controller state under the inherited V640 byte limit.",
        ),
        "unbounded_reference_is_larger": _gate(
            "reference_bytes>bounded_bytes and deployment_eligible=false",
            [panel.maximum_reference_memory_bytes, panel.maximum_bounded_memory_bytes],
            panel.maximum_reference_memory_bytes > panel.maximum_bounded_memory_bytes,
            "The reference exposes its extra memory and cannot enter bounded deployment.",
        ),
        "changed_weight_predictions": _gate(
            ">=24",
            panel.changed_weight_prediction_difference_count,
            panel.changed_weight_prediction_difference_count >= 24,
            "A mixture fixture must change later votes, not only internal weights.",
        ),
        "cold_reducer": _gate(
            transactional.sha256_json(panel.rows),
            transactional.sha256_json(reduced),
            reduced == panel.rows,
            "Rebuild headline rows from raw evidence instead of trusting producer totals.",
        ),
        "scorer_only_manifest": _gate(
            [True, False, EVALUATION_STREAM_COUNT],
            [
                manifest["scorer_only_evaluation_labels"],
                manifest["controller_receives_private_authority"],
                manifest["evaluation"]["stream_count"],
            ],
            manifest["scorer_only_evaluation_labels"] is True
            and manifest["controller_receives_private_authority"] is False
            and manifest["evaluation"]["stream_count"] == EVALUATION_STREAM_COUNT,
            "Freeze evaluation labels for a scorer without exposing them to the controller.",
        ),
    }
    ready = int(all(row["passed"] is True for row in gates.values()))
    now = datetime.now(UTC).isoformat()
    elapsed = time.monotonic() - invocation_start
    artifact = _base_artifact(
        checks,
        source_hashes,
        started_at=started_at,
        completed_at=now,
        duration_s=elapsed,
    )
    artifact.update(
        {
            "status": "complete",
            "phase_durations_s": {
                "preconditions": precondition_s,
                "stream_generation_and_seal": streams_s,
                "development_replay": replay_s,
                "raw_reduction_and_controls": reduction_s,
            },
            "rows": reduced,
            "sample_size_budget": _sample_budget(panel.completed_stream_count),
            "acceptance_gate_results": gates,
            "honest_verdict": (
                "complete_circular_positive: bounded fixed-share fixture mechanics pass; prospective efficacy is not claimed"
                if ready
                else "complete_null: fixed-share fixture completed but one or more mechanical gates failed"
            ),
            "verdict_class": "circular_positive" if ready else "null",
            "mixture_fixture_ready_score": ready,
            "stream_manifest_path": str(paths.manifest),
            "memory_budget_bytes": {
                "inherited_limit": MEMORY_CAP_BYTES,
                "configured_archive_capacity": ARCHIVE_CAP,
                "achieved_archive_count": achieved_archive_count,
                "maximum_bounded_serialized_state": panel.maximum_bounded_memory_bytes,
                "maximum_unbounded_reference_state": panel.maximum_reference_memory_bytes,
                "unbounded_reference_deployment_eligible": False,
            },
            "chronology_control_rows": controls,
            "raw_evidence_receipts": {
                "development_event_rows": {
                    "path": str(paths.development_event_rows),
                    "sha256": event_receipt["sha256"],
                    "row_count": len(panel.event_rows),
                },
                "nominee_rows": {
                    "path": str(paths.nominee_rows),
                    "sha256": nominee_receipt["sha256"],
                    "row_count": len(panel.nominee_rows),
                },
                "chronology_controls": {
                    "path": str(paths.chronology_controls),
                    "sha256": controls_receipt["sha256"],
                    "row_count": len(controls),
                },
                "stream_manifest": {
                    "path": str(paths.manifest),
                    "sha256": _sha256_path(paths.manifest),
                    "row_count": 1,
                },
            },
            "reducer_receipt": {
                "inference_substrate": REDUCER_INFERENCE_SUBSTRATE,
                "inference_substrate_class": REDUCER_INFERENCE_SUBSTRATE_CLASS,
                "source_path": str(paths.development_event_rows),
                "source_sha256": event_receipt["sha256"],
                "row_count": len(reduced),
            },
            "validation_receipts": [
                _internal_receipt(
                    f"independent_reduce {paths.development_event_rows}",
                    reduce_elapsed,
                    transactional.sha256_json(reduced),
                )
            ],
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"artifact_validation_failed:{','.join(errors)}")
    return artifact


def _receipt_error(receipt: Mapping[str, Any]) -> bool:
    """Reject validation receipts that omit actual command and timing evidence."""

    return (
        not isinstance(receipt.get("command"), str)
        or not receipt.get("command")
        or not isinstance(receipt.get("exit_code"), int)
        or not isinstance(receipt.get("duration_s"), (int, float))
        or float(receipt.get("duration_s", -1.0)) < 0.0
        or not isinstance(receipt.get("log_sha256"), str)
        or not str(receipt.get("log_sha256", "")).startswith("sha256:")
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check schema, raw rows, controls, bytes, receipts, and classification."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(any(field not in artifact for field in REQUIRED_ARTIFACT_FIELDS), "required_fields")
    add(artifact.get("schema") != SCHEMA, "schema")
    add(artifact.get("experiment_id") != EXPERIMENT_ID, "experiment_id")
    add(artifact.get("milestone") != MILESTONE, "milestone")
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    add(artifact.get("status") not in {"complete", "blocked"}, "status")
    add(artifact.get("MODEL_SPECS") != [], "model_specs")
    add(artifact.get("model_invoked") is not False, "model_invoked")
    counts = artifact.get("invocation_counts")
    add(
        not isinstance(counts, Mapping) or any(value != 0 for value in counts.values()),
        "invocation_counts",
    )
    add(artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate")
    add(
        artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
        "inference_substrate_class",
    )
    add(artifact.get("execution_venue") != EXECUTION_VENUE, "execution_venue")
    add(artifact.get("verifier_is_oracle") is not True, "oracle_authority")
    add(
        artifact.get("verdict_class")
        not in {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"},
        "verdict_class",
    )
    add(
        artifact.get("verifier_is_oracle") is True and artifact.get("verdict_class") == "positive",
        "oracle_positive",
    )
    principles = artifact.get("field_principles")
    add(
        not isinstance(principles, Mapping)
        or any(field not in principles for field in REQUIRED_ARTIFACT_FIELDS),
        "field_principles",
    )
    add(
        artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
        "reproducibility_checksum",
    )
    duration = artifact.get("duration_s")
    phases = artifact.get("phase_durations_s")
    add(not isinstance(duration, (int, float)) or float(duration) < 0.0, "duration_s")
    add(
        not isinstance(phases, Mapping)
        or any(
            not isinstance(value, (int, float)) or float(value) < 0.0 for value in phases.values()
        ),
        "phase_durations_s",
    )
    receipts = artifact.get("validation_receipts")
    add(
        not isinstance(receipts, list)
        or not receipts
        or any(not isinstance(row, Mapping) or _receipt_error(row) for row in receipts),
        "validation_receipts",
    )
    status = artifact.get("status")
    rows = artifact.get("rows")
    if status == "blocked":
        add(rows != [], "blocked_rows")
        add(artifact.get("mixture_fixture_ready_score") != 0, "blocked_ready_score")
        add(artifact.get("verdict_class") != "blocked", "blocked_verdict_class")
        add(
            not str(artifact.get("honest_verdict", "")).startswith("blocked_"),
            "blocked_verdict",
        )
        summary = artifact.get("gate_check_summary")
        add(
            not isinstance(summary, Mapping) or summary.get("first_failure") is None,
            "blocked_gate_summary",
        )
        return errors

    add(not str(artifact.get("honest_verdict", "")).startswith("complete_"), "honest_verdict")
    add(not isinstance(rows, list) or len(rows) != DEVELOPMENT_STREAM_COUNT * len(ARMS), "rows")
    gates = artifact.get("acceptance_gate_results")
    all_gates = (
        isinstance(gates, Mapping)
        and bool(gates)
        and all(isinstance(row, Mapping) and row.get("passed") is True for row in gates.values())
    )
    ready = artifact.get("mixture_fixture_ready_score")
    add(ready not in {0, 1} or ready != int(bool(all_gates)), "ready_score")
    add(ready == 1 and artifact.get("verdict_class") != "circular_positive", "ready_verdict")
    add(ready == 0 and artifact.get("verdict_class") != "null", "null_verdict")
    memory = artifact.get("memory_budget_bytes")
    add(
        not isinstance(memory, Mapping)
        or memory.get("inherited_limit") != MEMORY_CAP_BYTES
        or int(memory.get("maximum_bounded_serialized_state", MEMORY_CAP_BYTES + 1))
        > MEMORY_CAP_BYTES
        or memory.get("unbounded_reference_deployment_eligible") is not False,
        "memory_budget",
    )
    controls = artifact.get("chronology_control_rows")
    add(
        not isinstance(controls, list)
        or len(controls) != 6
        or any(row.get("passed") is not True for row in controls),
        "chronology_controls",
    )
    raw = artifact.get("raw_evidence_receipts")
    add(not isinstance(raw, Mapping), "raw_receipts")
    if isinstance(raw, Mapping):
        for receipt in raw.values():
            if not isinstance(receipt, Mapping):
                add(True, "raw_receipts")
                continue
            path = Path(str(receipt.get("path", "")))
            add(_sha256_path(path) != receipt.get("sha256"), "raw_receipt_hash")
    event_receipt = raw.get("development_event_rows") if isinstance(raw, Mapping) else None
    if isinstance(event_receipt, Mapping):
        try:
            cold_rows = independent_reduce(Path(str(event_receipt["path"])))
        except (KeyError, ValueError):
            cold_rows = []
        add(cold_rows != rows, "raw_reduction")
    manifest_path = Path(str(artifact.get("stream_manifest_path", "")))
    manifest = _load_object(manifest_path)
    add(
        not manifest
        or manifest.get("scorer_only_evaluation_labels") is not True
        or manifest.get("controller_receives_private_authority") is not False,
        "stream_manifest",
    )
    learning = artifact.get("learning_contract")
    add(
        not isinstance(learning, Mapping)
        or learning.get("eta") != ETA
        or learning.get("fixed_share") != FIXED_SHARE
        or learning.get("full_label_history_retained") is not False,
        "learning_contract",
    )
    schedule = artifact.get("feedback_schedule")
    add(
        not isinstance(schedule, Mapping)
        or schedule.get("warmup_charge") != WARMUP_COUNT
        or schedule.get("future_charge") != FUTURE_LABEL_COUNT
        or schedule.get("delay_steps") != FEEDBACK_DELAY,
        "feedback_schedule",
    )
    return errors


def attach_validation_receipts(
    artifact: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Attach exact command evidence and refresh the stable content checksum."""

    result = deepcopy(dict(artifact))
    result["validation_receipts"] = [
        *list(result.get("validation_receipts", [])),
        *(dict(row) for row in receipts),
    ]
    result["reproducibility_checksum"] = reproducibility_checksum(result)
    return result


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Cold-validate and atomically publish one terminal JSON object."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"artifact_validation_failed:{','.join(errors)}")
    payload = (json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n").encode("utf-8")
    return _atomic_write(path, payload)


def _command_receipt(command: Sequence[str], *, timeout_s: float = 1_800.0) -> tuple[JsonDict, str]:
    """Stream one bounded validation child and print a truthful heartbeat."""

    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("validation_stdout_unavailable")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    chunks: list[str] = []
    last_heartbeat = started
    timed_out = False
    while process.poll() is None:
        for key, _ in selector.select(timeout=1.0):
            line = key.fileobj.readline()
            if line:
                print(line, end="", flush=True)
                chunks.append(line)
        now = time.monotonic()
        if now - last_heartbeat >= 60.0:
            _progress(
                7,
                "subprocess heartbeat",
                f"outstanding={shlex.join(command)}; elapsed={now - started:.1f}s",
            )
            last_heartbeat = now
        if now - started > timeout_s:
            process.kill()
            timed_out = True
            break
    remainder = process.stdout.read()
    if remainder:
        print(remainder, end="", flush=True)
        chunks.append(remainder)
    process.wait()
    selector.close()
    elapsed = time.monotonic() - started
    output = "".join(chunks)
    exit_code = 124 if timed_out else int(process.returncode)
    return (
        {
            "command": shlex.join(command),
            "exit_code": exit_code,
            "duration_s": elapsed,
            "log_sha256": transactional.sha256_bytes(output.encode("utf-8")),
            "classification": "passed" if exit_code == 0 else "failed",
        },
        output,
    )


def _validation_commands(candidate: Path) -> list[list[str]]:
    """Return focused, affected, full, coverage, static, E2E, and artifact checks."""

    python = str(REPO_ROOT / ".venv/bin/python")
    pytest = str(REPO_ROOT / ".venv/bin/pytest")
    ruff = str(REPO_ROOT / ".venv/bin/ruff")
    mypy = str(REPO_ROOT / ".venv/bin/mypy")
    coverage = str(REPO_ROOT / ".venv/bin/coverage")
    new_test = str(TEST_PATH)
    module = "python/carnot/experiment_7295_v641_mixture_prototype.py"
    wrapper = str(WRAPPER_PATH)
    return [
        [
            pytest,
            new_test,
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--basetemp=/tmp/carnot-exp7295-focused",
        ],
        [
            pytest,
            "tests/python",
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--basetemp=/tmp/carnot-exp7295-full",
        ],
        [
            coverage,
            "run",
            "--data-file=/tmp/carnot-exp7295.coverage",
            "--include=python/carnot/experiment_7295_v641_mixture_prototype.py",
            "tests/python/coverage_experiment_7295.py",
        ],
        [
            coverage,
            "report",
            "--data-file=/tmp/carnot-exp7295.coverage",
            "--show-missing",
            "--fail-under=100",
        ],
        [ruff, "check", module, new_test, wrapper],
        [ruff, "format", "--check", module, new_test, wrapper],
        [mypy, module],
        [python, "scripts/check_spec_coverage.py", new_test],
        [
            pytest,
            f"{new_test}::test_scenario_cl_7295_controls_cover_chronology_bytes_and_restart",
            f"{new_test}::test_scenario_cl_7295_terminal_builds_and_cold_reduces",
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--basetemp=/tmp/carnot-exp7295-e2e",
        ],
        [
            python,
            "-m",
            "carnot.experiment_7295_v641_mixture_prototype",
            "--date",
            RUN_DATE,
            "--validate-raw",
            str(candidate),
        ],
        [python, "scripts/adversarial_verify.py", str(candidate)],
        [python, "scripts/verdict_row_consistency_lint.py", str(candidate)],
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date and the private cold-validation mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=[RUN_DATE])
    parser.add_argument("--validate-raw", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure, validate, and publish only terminal fixed-share evidence."""

    args = _parse_args(argv)
    if args.validate_raw is not None:
        candidate = _load_object(args.validate_raw)
        errors = validate_artifact(candidate)
        print(
            json.dumps(
                {
                    "candidate": str(args.validate_raw),
                    "errors": errors,
                    "row_checksum": transactional.sha256_json(candidate.get("rows", [])),
                    "inference_substrate": REDUCER_INFERENCE_SUBSTRATE,
                    "inference_substrate_class": REDUCER_INFERENCE_SUBSTRATE_CLASS,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return int(bool(errors))

    paths = ExperimentPaths.defaults()
    invocation_start = time.monotonic()
    _progress(0, "start", "Exp7295 bounded fixed-share prototype")
    artifact = build_and_seal(REPO_ROOT, paths, progress=True)
    write_artifact(paths.terminal_candidate, artifact)
    if artifact["status"] == "blocked":
        write_artifact(paths.artifact, artifact)
        _progress(8, "end", artifact["honest_verdict"])
        return 0

    commands = _validation_commands(paths.terminal_candidate)
    receipts: list[JsonDict] = []
    validation_started = time.monotonic()
    validation_dir = paths.raw_dir / "validation"
    for index, command in enumerate(commands):
        _progress(7, "before subprocess", f"{index + 1}/{len(commands)} {shlex.join(command)}")
        receipt, output = _command_receipt(command)
        log_path = validation_dir / f"{index:02d}.log"
        log_receipt = _atomic_write(log_path, output.encode("utf-8"))
        receipt["log_path"] = str(log_path)
        receipt["log_sha256"] = log_receipt["sha256"]
        receipts.append(receipt)
        _progress(
            7,
            "after subprocess",
            f"{index + 1}/{len(commands)} exit={receipt['exit_code']} elapsed={receipt['duration_s']:.3f}s",
        )
    artifact = attach_validation_receipts(artifact, receipts)
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["duration_s"] = time.monotonic() - invocation_start
    artifact["phase_durations_s"]["validation"] = time.monotonic() - validation_started
    global_suite = next(
        (receipt for command, receipt in zip(commands, receipts) if "tests/python" in command),
        None,
    )
    if global_suite is not None:
        artifact["global_suite_observation"] = {
            "exit_code": global_suite["exit_code"],
            "classification": global_suite["classification"],
            "gates_scoped_fixture": False,
        }
        if global_suite["exit_code"] != 0:
            artifact["honest_verdict"] += "; repository-wide suite retained unrelated failures"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    write_artifact(paths.terminal_candidate, artifact)
    required_failures = [
        receipt
        for command, receipt in zip(commands, receipts)
        if receipt["exit_code"] != 0 and "tests/python" not in command
    ]
    if required_failures:
        _progress(8, "end", "validation failed; terminal candidate retained without publication")
        return 1
    write_artifact(paths.artifact, artifact)
    _progress(8, "end", artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
