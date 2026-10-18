"""Build the V635 query-refinement fixture and commit-only runtime.

The evaluator worker owns every hidden parameter and outcome. The learner gets
public events, charged warmup releases, and a frozen query contract. Deployed
predictions use only warmup bytes or exact predicates in transactional memory.

Spec refs: REQ-CL-7212 and SCENARIO-CL-7212-*.
"""

from __future__ import annotations

import argparse
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
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_7198_v634_feedback_capacity_stream as exp7198
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7212
SCHEMA = "carnot.exp7212.v635_refinement_fixture.v1"
MILESTONE = "2026.09.635"
RUN_DATE = "20260911"
RANDOM_SEED = 7_212_000
STREAM_SEEDS = tuple(range(7_212_001, 7_212_021))
EVENTS_PER_SEED = 1_024
WARMUP_COUNT = 32
PARAMETER_DOMAIN = tuple(exp7198.PARAMETER_DOMAIN)
FAMILIES = tuple(exp7198.FAMILIES)
PHASES: tuple[JsonDict, ...] = (
    {"name": "stable", "public_name": "phase_0", "start": 0, "stop": 256},
    {"name": "drift", "public_name": "phase_1", "start": 256, "stop": 512},
    {"name": "recurrence", "public_name": "phase_2", "start": 512, "stop": 768},
    {"name": "poison", "public_name": "phase_3", "start": 768, "stop": 1_024},
)
ARMS = (
    "warmup_frozen",
    "passive_query_committed",
    "random_query_committed",
    "witness_query_committed",
    "witness_query_version_space",
)
QUERY_BUDGET = 64
FITTING_BUDGET = 48
VALIDATION_BUDGET = 16
VALIDATION_PER_FAMILY = 4
PENDING_CAPACITY = 4
POISON_COUNT_PER_SEED = 16
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = (
    "deterministic CPU finite-domain witness selection, exact predicate replay, "
    "and transactional commit and rollback; no model invocation"
)
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STREAM_ROOT = Path("results/streams/experiment_7212")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/experiment_7212_v635_refinement_fixture.json")
DEFAULT_ARTIFACT_PATH = Path("results/experiment_7212_v635_refinement_fixture.json")
DEFAULT_UPSTREAM_7199 = Path("results/experiment_7199_v634_bounded_acquisition.json")
DEFAULT_UPSTREAM_7200 = Path("results/experiment_7200_v634_acquisition_cold_audit.json")
DEFAULT_AUTHENTICATOR_7204 = Path("results/experiment_7204_v634_capstone.json")
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
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("python/carnot/experiment_7212_v635_refinement_fixture.py"),
    Path("scripts/experiments/experiment_7212_v635_refinement_fixture.py"),
    Path("tests/python/test_experiment_7212_v635_refinement_fixture.py"),
    DEFAULT_AUTHENTICATOR_7204,
    SPEC_PATH,
)

FORBIDDEN_PUBLIC_FIELDS = {
    "authority",
    "audit_label",
    "delay",
    "exact_label",
    "future_label",
    "hidden_parameter",
    "hidden_seed",
    "observed_label",
    "parameter",
    "poisoned",
    "release_index",
    "regime",
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
    "refinement_fixture_ready_score",
    "public_stream_path",
    "authority_sidecar_path",
    "query_budget_contract",
    "commit_path_receipt",
    "validation_partition_rows",
    "MODEL_SPECS",
    "model_invoked",
    "split_feedback_manifest_path",
    "released_warmup_path",
    "controller_serialization_path",
    "stream_hashes",
    "stream_conformance_errors",
    "source_citation_receipt",
    "upstream_gate_receipt",
    "checkpoint_path",
    "checkpoint_hash",
    "no_model_weight_mutation",
    "certificate_scope",
    "controller_contract",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "A versioned schema makes incompatible readers fail closed.",
    "experiment_id": "A fixed ID prevents another task from supplying this evidence.",
    "milestone": "The milestone binds the fixture to the V635 contract.",
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
    "sample_size_budget": "Retain planned, attempted, completed, censored and independent-unit counts.",
    "random_seed": "Freeze stochastic choices before held-out outcomes are read.",
    "reproducibility_checksum": "Hash the inputs, code, settings and raw rows.",
    "gate_check_summary": "Every blocked verdict names failed check, upstream, field, expected and observed value.",
    "verifier_is_oracle": "Same correctness authority remains circular even with a separate implementation.",
    "verdict_class": "Use exactly positive | circular_positive | null | blocked | disqualified | partial.",
    "honest_verdict": "Use complete_ or complete: for findings and blocked_* for external absence.",
    "refinement_fixture_ready_score": "One certifies the input and causal runtime contract, not learning value.",
    "public_stream_path": "Learning reads only events and released queried feedback.",
    "authority_sidecar_path": "Hidden parameters and final audit labels never enter the learner.",
    "query_budget_contract": "All fitting and validation queries consume the same bounded resource.",
    "commit_path_receipt": "Only committed predicates can change the new deployed arm.",
    "validation_partition_rows": "Disjoint validation cannot silently train the candidate.",
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is only cited.",
    "split_feedback_manifest_path": "Frozen public roles and delay law prevent outcome-selected partitions.",
    "released_warmup_path": "Only charged released warmup labels may shape the frozen fallback.",
    "controller_serialization_path": "Reloadable public state binds the downstream learner contract.",
    "stream_hashes": "Independent hashes expose changes to each sealed stream view.",
    "stream_conformance_errors": "An empty list proves all owned stream checks passed.",
    "source_citation_receipt": "The local checked source bytes bind the adapted method and its limits.",
    "upstream_gate_receipt": "Prior completion and failed value are separate authenticated facts.",
    "checkpoint_path": "Progress bytes stay under results/checkpoints and outside the deliverable.",
    "checkpoint_hash": "The checkpoint hash binds the completed unit count.",
    "no_model_weight_mutation": "CPU constraint memory is not model-weight training.",
    "certificate_scope": "Partial finite observations support empirical validation, not universal proof.",
    "controller_contract": "Private fitting state cannot become a deployed majority vote.",
}

exact_label = exp7198.exact_label
independent_exact_label = exp7198.independent_exact_label
canonical_json_bytes = transactional.canonical_json_bytes
sha256_bytes = transactional.sha256_bytes
sha256_json = transactional.sha256_json


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep immutable stream, checkpoint, and terminal destinations distinct."""

    public_stream: Path
    authority_sidecar: Path
    split_feedback_manifest: Path
    released_warmup: Path
    controller_serialization: Path
    checkpoint: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the repository destinations required by the public command."""

        return cls.from_stream_root(
            DEFAULT_STREAM_ROOT,
            DEFAULT_CHECKPOINT_PATH,
            DEFAULT_ARTIFACT_PATH,
        )

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put test evidence under one caller-owned private directory."""

        return cls.from_stream_root(
            root / "streams" / "experiment_7212",
            root / "checkpoints" / "experiment_7212_v635_refinement_fixture.json",
            root / "experiment_7212_v635_refinement_fixture.json",
        )

    @classmethod
    def from_stream_root(
        cls,
        stream_root: Path,
        checkpoint: Path | None = None,
        artifact: Path | None = None,
    ) -> ExperimentPaths:
        """Build every stream path from one explicit immutable directory."""

        return cls(
            stream_root / "public_stream.jsonl",
            stream_root / "evaluator_sidecar.jsonl",
            stream_root / "split_feedback_manifest.json",
            stream_root / "released_warmup.jsonl",
            stream_root / "controller_serialization.json",
            checkpoint or stream_root.parent.parent / "checkpoints" / "experiment_7212.json",
            artifact or stream_root.parent.parent / "experiment_7212.json",
        )


class ImmutableSealError(RuntimeError):
    """Report an attempted change to already sealed scientific input bytes."""


def _resolve(repo_root: Path, path: Path | str) -> Path:
    """Resolve repository-relative evidence without changing caller-owned paths."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _sha256_path(path: Path) -> str | None:
    """Hash a file in bounded chunks or retain a missing identity as None."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def _atomic_write(path: Path, payload: bytes) -> None:
    """Publish bytes with one rename so readers never observe a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_immutable(path: Path, payload: bytes) -> None:
    """Keep an identical prior seal and reject any changed replacement bytes."""

    if path.exists():
        if path.read_bytes() != payload:
            raise ImmutableSealError(f"immutable_path_conflict:{path}")
        return
    _atomic_write(path, payload)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode rows with the same canonical JSON form used by state hashes."""

    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def read_jsonl(path: Path) -> list[JsonDict]:
    """Read one sealed JSON object per line without reordering the chronology."""

    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_row_not_object:{path}")
            rows.append(value)
    return rows


def _stable_seed(*parts: Any) -> int:
    """Derive repeatable local randomness without Python's salted object hash."""

    return int(sha256_json(list(parts)).removeprefix("sha256:")[:16], 16)


def _phase_for(index: int) -> Mapping[str, Any]:
    """Return the single frozen evaluator phase for a chronology index."""

    return next(phase for phase in PHASES if phase["start"] <= index < phase["stop"])


def _hidden_parameters(seed: int, family_id: str) -> tuple[int, int, int]:
    """Create private stable and drift parameters only inside evaluator work."""

    authority_seed = _stable_seed("exp7212-authority", seed, family_id)
    rng = random.Random(authority_seed)
    stable = rng.randrange(33)
    drift = rng.randrange(33)
    while drift == stable:
        drift = rng.randrange(33)
    return authority_seed, stable, drift


def _reserved_values(seed: int, family_id: str) -> list[int]:
    """Reserve four public values before any fitting outcome can be read."""

    rng = random.Random(_stable_seed("exp7212-validation", seed, family_id))
    return sorted(rng.sample(list(PARAMETER_DOMAIN), VALIDATION_PER_FAMILY))


def _burst_delay(seed: int, request_index: int) -> int:
    """Reuse the fixed Exp7198 burst support with fresh task-owned seeds."""

    return (0, 4, 16, 32)[_stable_seed("exp7212-burst", seed, request_index) % 4]


def _poison_indices(seed: int) -> set[int]:
    """Freeze sixteen corrupted feedback rows in the final evaluator phase."""

    candidates = list(range(768, 1_024))
    random.Random(_stable_seed("exp7212-poison", seed)).shuffle(candidates)
    return set(candidates[:POISON_COUNT_PER_SEED])


def _family_order(seed: int, phase: Mapping[str, Any]) -> list[str]:
    """Balance each phase while keeping public event order deterministic."""

    if phase["name"] == "stable":
        prefix = list(FAMILIES) * 8
        tail = list(FAMILIES) * 56
        random.Random(_stable_seed("exp7212-family-tail", seed)).shuffle(tail)
        return prefix + tail
    order = list(FAMILIES) * 64
    random.Random(_stable_seed("exp7212-family", seed, phase["public_name"])).shuffle(order)
    return order


def _numeric_value(
    seed: int,
    chronology_index: int,
    family_id: str,
    family_occurrence: int,
    reserved: Mapping[str, Sequence[int]],
) -> int:
    """Create public values and keep warmup observations outside validation."""

    if chronology_index < WARMUP_COUNT:
        allowed = [value for value in PARAMETER_DOMAIN if value not in reserved[family_id]]
        return allowed[family_occurrence % len(allowed)]
    return _stable_seed("exp7212-public-x", seed, chronology_index, family_id) % 33


@dataclass(frozen=True)
class FrozenWarmupFallback:
    """Keep the only non-memory predictor fixed after released warmup labels."""

    label_rows: tuple[tuple[str, int, str], ...]
    defaults: tuple[tuple[str, str], ...]
    evidence_ids: tuple[str, ...]

    @classmethod
    def from_releases(cls, releases: Sequence[Mapping[str, Any]]) -> FrozenWarmupFallback:
        """Freeze value labels and family defaults from released evidence only."""

        labels: dict[tuple[str, int], str] = {}
        counts = {family: {"accept": 0, "reject": 0} for family in FAMILIES}
        evidence_ids: list[str] = []
        for row in releases:
            family = str(row["family_id"])
            value = int(row["numeric_value"])
            label = str(row["observed_label"])
            labels[(family, value)] = label
            counts[family][label] += 1
            evidence_ids.append(str(row["event_id"]))
        defaults = tuple(
            (
                family,
                "accept" if counts[family]["accept"] > counts[family]["reject"] else "reject",
            )
            for family in FAMILIES
        )
        label_rows = tuple(
            (family, value, label) for (family, value), label in sorted(labels.items())
        )
        return cls(label_rows, defaults, tuple(evidence_ids))

    def predict(self, public_event: Mapping[str, Any]) -> str:
        """Use an observed warmup value or the frozen family-majority fallback."""

        family = str(public_event["family_id"])
        value = int(public_event["numeric_value"])
        labels = {(item[0], item[1]): item[2] for item in self.label_rows}
        defaults = dict(self.defaults)
        return labels.get((family, value), defaults[family])

    def state_dict(self) -> JsonDict:
        """Serialize all fixed bytes that can affect a fallback prediction."""

        return {
            "label_rows": [list(row) for row in self.label_rows],
            "defaults": {key: value for key, value in self.defaults},
            "evidence_ids": list(self.evidence_ids),
            "frozen": True,
        }


@dataclass
class CandidateFamilyState:
    """Hold acquisition-private fitting and validation state for one family."""

    hypotheses: set[int] = field(default_factory=lambda: set(PARAMETER_DOMAIN))
    support_ids: list[str] = field(default_factory=list)
    validation_ids: list[str] = field(default_factory=list)
    candidate_parameter: int | None = None
    expected_parent_hash: str | None = None
    active_commit_receipt: JsonDict | None = None

    def state_dict(self) -> JsonDict:
        """Serialize private acquisition state without making it deployable."""

        return {
            "hypotheses": sorted(self.hypotheses),
            "support_ids": list(self.support_ids),
            "validation_ids": list(self.validation_ids),
            "candidate_parameter": self.candidate_parameter,
            "expected_parent_hash": self.expected_parent_hash,
            "active_commit_receipt": deepcopy(self.active_commit_receipt),
        }


def choose_witness(
    family_id: str,
    hypotheses: set[int],
    *,
    reserved_values: set[int],
) -> int | None:
    """Choose the lowest maximally balanced public disagreement value."""

    if len(hypotheses) < 2:
        return None
    ranked: list[tuple[int, int]] = []
    for value in PARAMETER_DOMAIN:
        if value in reserved_values:
            continue
        labels = [exact_label(family_id, value, parameter) for parameter in hypotheses]
        score = min(labels.count("accept"), labels.count("reject"))
        if score > 0:
            ranked.append((score, value))
    if not ranked:
        return None
    return max(ranked, key=lambda item: (item[0], -item[1]))[1]


class CandidateRefinementController:
    """Fit candidates privately while deployment reads committed memory only."""

    def __init__(
        self,
        *,
        stream_id: str,
        fallback: FrozenWarmupFallback,
        reserved_validation: Mapping[str, set[int]],
    ) -> None:
        self.stream_id = stream_id
        self.fallback = fallback
        self.reserved_validation = {
            family: set(reserved_validation.get(family, set())) for family in FAMILIES
        }
        self.families = {family: CandidateFamilyState() for family in FAMILIES}
        self.fitting_queries = 0
        self.validation_queries = 0

    def state_dict(self) -> JsonDict:
        """Return reloadable private state and an explicit deployment boundary."""

        return {
            "schema": "carnot.exp7212.controller.v1",
            "stream_id": self.stream_id,
            "fallback": self.fallback.state_dict(),
            "reserved_validation": {
                family: sorted(values) for family, values in self.reserved_validation.items()
            },
            "families": {family: self.families[family].state_dict() for family in FAMILIES},
            "fitting_queries": self.fitting_queries,
            "validation_queries": self.validation_queries,
            "deployment_reads_private_hypotheses": False,
        }

    def deployed_predict(
        self,
        public_event: Mapping[str, Any],
        memory: transactional.TransactionalConstraintMemory,
    ) -> JsonDict:
        """Read a durable exact predicate or use the frozen warmup fallback."""

        family = str(public_event["family_id"])
        key = f"predicate:{self.stream_id}:{family}"
        snapshot = memory.begin_episode(str(public_event["event_id"]))
        lookup = memory.lookup(snapshot, key)
        memory.end_episode()
        record = lookup["record"]
        if lookup["found"] and lookup["safe"] and isinstance(record, Mapping):
            parameter = int(record["parameter"])
            prediction = exact_label(family, int(public_event["numeric_value"]), parameter)
            source = "committed_exact_predicate"
        else:
            prediction = self.fallback.predict(public_event)
            source = "frozen_warmup_fallback"
        return {
            "prediction": prediction,
            "source": source,
            "memory_version": snapshot["version"],
            "uncommitted_hypotheses_consulted": False,
        }

    def observe_fitting(
        self,
        public_event: Mapping[str, Any],
        observed_label: str,
        *,
        memory: transactional.TransactionalConstraintMemory,
    ) -> JsonDict:
        """Eliminate candidates only from a charged released fitting label."""

        if self.fitting_queries >= FITTING_BUDGET:
            raise ValueError("fitting_query_budget_exhausted")
        family = str(public_event["family_id"])
        value = int(public_event["numeric_value"])
        if value in self.reserved_validation[family]:
            raise ValueError("reserved_validation_used_for_fitting")
        self.fitting_queries += 1
        state = self.families[family]
        before = sorted(state.hypotheses)
        state.hypotheses = {
            parameter
            for parameter in state.hypotheses
            if exact_label(family, value, parameter) == observed_label
        }
        state.support_ids.append(str(public_event["event_id"]))
        if len(state.hypotheses) == 1 and state.candidate_parameter is None:
            state.candidate_parameter = next(iter(state.hypotheses))
            state.expected_parent_hash = memory.state_hash()
            operation = "freeze_singleton"
        elif state.hypotheses:
            operation = "fit_eliminate"
        else:
            operation = "candidate_set_empty"
        return {
            "operation": operation,
            "family_id": family,
            "event_id": public_event["event_id"],
            "hypotheses_before": before,
            "hypotheses_after": sorted(state.hypotheses),
            "validation_used_for_elimination": False,
            "query_charged": True,
        }

    def _reset_family(self, family: str) -> None:
        """Discard contradicted acquisition state without reading hidden truth."""

        self.families[family] = CandidateFamilyState()

    def commit_candidate(
        self,
        family: str,
        *,
        memory: transactional.TransactionalConstraintMemory,
        exact_authority: bool = True,
    ) -> JsonDict:
        """Commit one validated singleton only against its unchanged parent hash."""

        state = self.families[family]
        before_hash = memory.state_hash()
        if state.candidate_parameter is None or len(state.validation_ids) < VALIDATION_PER_FAMILY:
            return {
                "admitted": False,
                "reason": "candidate_not_empirically_validated",
                "state_hash_before": before_hash,
                "state_hash_after": before_hash,
            }
        if state.expected_parent_hash != before_hash:
            return {
                "admitted": False,
                "reason": "stale_expected_parent_hash",
                "expected_parent_hash": state.expected_parent_hash,
                "state_hash_before": before_hash,
                "state_hash_after": memory.state_hash(),
            }
        parameter = state.candidate_parameter
        predicate = f"{family}(parameter={parameter})"
        event = {
            "event_id": f"commit:{self.stream_id}:{family}:{len(state.validation_ids)}",
            "scope": self.stream_id,
            "facts": {
                "constraint_family": family,
                "scope": self.stream_id,
                "validation_ids": list(state.validation_ids),
            },
            "exact_label": exact_authority,
            "certified_repair": predicate,
        }
        key = f"predicate:{self.stream_id}:{family}"
        proposal = {
            "key": key,
            "scope": self.stream_id,
            "repair": predicate,
            "source_event_id": event["event_id"],
            "evidence_hash": transactional.event_evidence_hash(event),
            "future_use_eligible": True,
            "expires_after": EVENTS_PER_SEED,
            "content_hash": sha256_json({"key": key, "scope": self.stream_id, "repair": predicate}),
            "family_id": family,
            "parameter": parameter,
            "stream_id": self.stream_id,
            "validation_ids": list(state.validation_ids),
            "certificate_scope": "empirically_validated_partial_finite_domain",
        }
        decision = memory.admit(proposal, event, boundary_index=1)
        decision = deepcopy(decision)
        decision["state_hash_before"] = before_hash
        decision["state_hash_after"] = memory.state_hash()
        if decision["admitted"]:
            state.active_commit_receipt = deepcopy(decision["commit_receipt"])
        return decision

    def observe_validation(
        self,
        public_event: Mapping[str, Any],
        observed_label: str,
        *,
        memory: transactional.TransactionalConstraintMemory,
    ) -> JsonDict:
        """Validate without fitting and revoke a contradicted committed version."""

        if self.validation_queries >= VALIDATION_BUDGET:
            raise ValueError("validation_query_budget_exhausted")
        family = str(public_event["family_id"])
        value = int(public_event["numeric_value"])
        event_id = str(public_event["event_id"])
        if value not in self.reserved_validation[family]:
            raise ValueError("validation_value_not_reserved")
        state = self.families[family]
        if event_id in state.validation_ids:
            raise ValueError("duplicate_validation_evidence")
        self.validation_queries += 1
        hypotheses_before = sorted(state.hypotheses)
        if state.candidate_parameter is None:
            return {
                "operation": "validation_charged_without_candidate",
                "validation_used_for_elimination": False,
                "hypotheses_before": hypotheses_before,
                "hypotheses_after": hypotheses_before,
            }
        candidate_label = exact_label(family, value, state.candidate_parameter)
        if candidate_label != observed_label:
            rollback = None
            if state.active_commit_receipt is not None:
                rollback = memory.rollback(state.active_commit_receipt)
            self._reset_family(family)
            return {
                "operation": "revoke_contradicted_candidate",
                "validation_used_for_elimination": False,
                "hypotheses_before": hypotheses_before,
                "hypotheses_after": list(PARAMETER_DOMAIN),
                "rollback_receipt": rollback,
            }
        state.validation_ids.append(event_id)
        commit = None
        if (
            len(state.validation_ids) >= VALIDATION_PER_FAMILY
            and state.active_commit_receipt is None
        ):
            commit = self.commit_candidate(family, memory=memory)
        return {
            "operation": "validation_pass" if commit is None else "validation_commit",
            "validation_used_for_elimination": False,
            "hypotheses_before": hypotheses_before,
            "hypotheses_after": sorted(state.hypotheses),
            "commit_receipt": commit,
        }


def _controller_serialization(
    warmup_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    seeds: Sequence[int],
) -> JsonDict:
    """Build initial learner state only from released warmup and public splits."""

    partitions = {
        (int(row["seed"]), str(row["family_id"])): set(row["reserved_validation_x"])
        for row in manifest["validation_partitions"]
    }
    controllers = []
    for seed in seeds:
        released = [row for row in warmup_rows if int(row["seed"]) == seed]
        fallback = FrozenWarmupFallback.from_releases(released)
        reserved = {family: partitions[(seed, family)] for family in FAMILIES}
        controller = CandidateRefinementController(
            stream_id=f"stream-{seed}",
            fallback=fallback,
            reserved_validation=reserved,
        )
        controllers.append({"seed": seed, "state": controller.state_dict()})
    return {
        "schema": "carnot.exp7212.controller_serialization.v1",
        "derived_only_from_released_warmup": True,
        "controller_count": len(controllers),
        "controllers": controllers,
    }


def evaluator_worker(paths: ExperimentPaths, *, seeds: Sequence[int] = STREAM_SEEDS) -> int:
    """Generate hidden parameters and labels inside the evaluator-only boundary."""

    print("EVALUATOR PHASE START: generate sealed public and authority views", flush=True)
    public_rows: list[JsonDict] = []
    authority_rows: list[JsonDict] = []
    warmup_rows: list[JsonDict] = []
    validation_partitions: list[JsonDict] = []
    for seed_number, seed in enumerate(seeds, start=1):
        reserved = {family: _reserved_values(seed, family) for family in FAMILIES}
        parameters = {family: _hidden_parameters(seed, family) for family in FAMILIES}
        for family in FAMILIES:
            validation_partitions.append(
                {
                    "seed": seed,
                    "family_id": family,
                    "reserved_validation_x": reserved[family],
                    "fitting_x_domain": [
                        value for value in PARAMETER_DOMAIN if value not in reserved[family]
                    ],
                    "reserved_before_fitting": True,
                    "validation_used_for_elimination": False,
                }
            )
        poison_indices = _poison_indices(seed)
        occurrence = {family: 0 for family in FAMILIES}
        for phase in PHASES:
            family_order = _family_order(seed, phase)
            for phase_offset, family in enumerate(family_order):
                chronology_index = int(phase["start"]) + phase_offset
                value = _numeric_value(
                    seed,
                    chronology_index,
                    family,
                    occurrence[family],
                    reserved,
                )
                occurrence[family] += 1
                authority_seed, stable_parameter, drift_parameter = parameters[family]
                parameter = (
                    stable_parameter
                    if phase["name"] in {"stable", "recurrence"}
                    else drift_parameter
                )
                label = exact_label(family, value, parameter)
                independent = independent_exact_label(family, value, parameter)
                poisoned = chronology_index in poison_indices
                observed = "reject" if poisoned and label == "accept" else label
                if poisoned and label == "reject":
                    observed = "accept"
                event_id = f"exp7212-{seed}-{chronology_index:04d}"
                public_rows.append(
                    {
                        "event_id": event_id,
                        "stream_id": f"stream-{seed}",
                        "seed": seed,
                        "chronology_index": chronology_index,
                        "family_id": family,
                        "numeric_value": value,
                        "public_input": f"family={family};value={value}",
                        "public_grammar_id": "finite_numeric_predicate.v1",
                    }
                )
                delay = _burst_delay(seed, chronology_index)
                authority_rows.append(
                    {
                        "row_type": "event_authority",
                        "event_id": event_id,
                        "seed": seed,
                        "chronology_index": chronology_index,
                        "phase": phase["name"],
                        "family_id": family,
                        "numeric_value": value,
                        "hidden_seed": authority_seed,
                        "hidden_parameter": parameter,
                        "stable_parameter": stable_parameter,
                        "drift_parameter": drift_parameter,
                        "exact_label": label,
                        "independent_exact_label": independent,
                        "poisoned": poisoned,
                        "observed_label": observed,
                        "delay": delay,
                        "release_index": chronology_index + delay,
                    }
                )
                if chronology_index < WARMUP_COUNT:
                    warmup_rows.append(
                        {
                            "source": "released_queried_feedback",
                            "event_id": event_id,
                            "seed": seed,
                            "family_id": family,
                            "numeric_value": value,
                            "observed_label": observed,
                            "request_index": chronology_index,
                            "release_index": chronology_index + delay,
                            "query_charged": True,
                        }
                    )
            for family in FAMILIES:
                _, stable_parameter, drift_parameter = parameters[family]
                parameter = (
                    stable_parameter
                    if phase["name"] in {"stable", "recurrence"}
                    else drift_parameter
                )
                for value in PARAMETER_DOMAIN:
                    authority_rows.append(
                        {
                            "row_type": "audit_label",
                            "seed": seed,
                            "phase": phase["name"],
                            "family_id": family,
                            "numeric_value": value,
                            "hidden_parameter": parameter,
                            "audit_label": exact_label(family, value, parameter),
                            "independent_audit_label": independent_exact_label(
                                family, value, parameter
                            ),
                            "scoring_only": True,
                        }
                    )
        if seed_number % 5 == 0 or seed_number == len(seeds):
            print(
                f"EVALUATOR PROGRESS: completed streams {seed_number}/{len(seeds)}",
                flush=True,
            )
    warmup_rows.sort(
        key=lambda row: (int(row["seed"]), int(row["release_index"]), int(row["request_index"]))
    )
    manifest = {
        "schema": "carnot.exp7212.split_feedback_manifest.v1",
        "frozen_before_fitting": True,
        "seeds": list(seeds),
        "events_per_seed": EVENTS_PER_SEED,
        "public_intervals": [
            {"name": phase["public_name"], "start": phase["start"], "stop": phase["stop"]}
            for phase in PHASES
        ],
        "families": list(FAMILIES),
        "parameter_domain": list(PARAMETER_DOMAIN),
        "warmup_released_observations": WARMUP_COUNT,
        "query_budget": QUERY_BUDGET,
        "fitting_budget": FITTING_BUDGET,
        "validation_budget": VALIDATION_BUDGET,
        "pending_capacity": PENDING_CAPACITY,
        "burst_delay_support": [0, 4, 16, 32],
        "arms": list(ARMS),
        "validation_partitions": validation_partitions,
        "authority_fields_available_to_learner": False,
    }
    controllers = _controller_serialization(warmup_rows, manifest, seeds)
    write_immutable(paths.public_stream, _jsonl_bytes(public_rows))
    write_immutable(paths.authority_sidecar, _jsonl_bytes(authority_rows))
    write_immutable(paths.released_warmup, _jsonl_bytes(warmup_rows))
    write_immutable(paths.split_feedback_manifest, canonical_json_bytes(manifest))
    write_immutable(paths.controller_serialization, canonical_json_bytes(controllers))
    print("EVALUATOR PHASE END: all immutable views are sealed", flush=True)
    return 0


def _nested_keys(value: Any) -> set[str]:
    """Collect nested keys so authority data cannot hide below one public object."""

    if isinstance(value, Mapping):
        keys = {str(key) for key in value}
        for item in value.values():
            keys.update(_nested_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_nested_keys(item))
        return keys
    return set()


def public_leakage_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Name public events that contain evaluator-only keys at any depth."""

    return [
        str(row.get("event_id", f"row-{index}"))
        for index, row in enumerate(rows)
        if _nested_keys(row) & FORBIDDEN_PUBLIC_FIELDS
    ]


def stream_conformance_errors(
    public: Sequence[Mapping[str, Any]],
    authority: Sequence[Mapping[str, Any]],
    warmup: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    expected_seeds: Sequence[int] = STREAM_SEEDS,
) -> list[str]:
    """Check balance, separation, exact labels, audit coverage, and public splits."""

    errors: list[str] = []
    event_authority = [row for row in authority if row.get("row_type") == "event_authority"]
    audit = [row for row in authority if row.get("row_type") == "audit_label"]
    expected_events = len(expected_seeds) * EVENTS_PER_SEED
    if len(public) != expected_events or len(event_authority) != expected_events:
        errors.append("event_count")
    if public_leakage_errors(public):
        errors.append("public_authority_leak")
    if len({str(row.get("event_id")) for row in public}) != len(public):
        errors.append("public_event_identity")
    authority_by_id = {str(row["event_id"]): row for row in event_authority}
    if set(authority_by_id) != {str(row.get("event_id")) for row in public}:
        errors.append("public_authority_identity")
    if any(
        row.get("exact_label")
        != independent_exact_label(
            str(row.get("family_id")),
            int(row.get("numeric_value", -1)),
            int(row.get("hidden_parameter", -1)),
        )
        or row.get("independent_exact_label") != row.get("exact_label")
        for row in event_authority
    ):
        errors.append("event_exact_grounding")
    if any(
        row.get("audit_label")
        != independent_exact_label(
            str(row.get("family_id")),
            int(row.get("numeric_value", -1)),
            int(row.get("hidden_parameter", -1)),
        )
        or row.get("independent_audit_label") != row.get("audit_label")
        or row.get("scoring_only") is not True
        for row in audit
    ):
        errors.append("audit_exact_grounding")
    if len(audit) != len(expected_seeds) * len(PHASES) * len(FAMILIES) * 33:
        errors.append("audit_panel_count")
    if len(warmup) != len(expected_seeds) * WARMUP_COUNT:
        errors.append("warmup_count")
    if any(
        row.get("source") != "released_queried_feedback"
        or row.get("query_charged") is not True
        or str(row.get("observed_label"))
        != str(authority_by_id.get(str(row.get("event_id")), {}).get("observed_label"))
        for row in warmup
    ):
        errors.append("warmup_release")
    partitions = manifest.get("validation_partitions", [])
    if len(partitions) != len(expected_seeds) * len(FAMILIES):
        errors.append("validation_partition_count")
    if any(
        len(row.get("reserved_validation_x", [])) != VALIDATION_PER_FAMILY
        or not set(row.get("reserved_validation_x", [])).isdisjoint(row.get("fitting_x_domain", []))
        or row.get("validation_used_for_elimination") is not False
        for row in partitions
    ):
        errors.append("validation_partition")
    for seed in expected_seeds:
        seed_public = [row for row in public if int(row.get("seed", -1)) == seed]
        seed_authority = [row for row in event_authority if int(row.get("seed", -1)) == seed]
        if len(seed_public) != EVENTS_PER_SEED:
            errors.append("seed_event_count")
            continue
        for phase in PHASES:
            phase_rows = [
                row
                for row in seed_public
                if int(phase["start"]) <= int(row["chronology_index"]) < int(phase["stop"])
            ]
            if len(phase_rows) != 256 or any(
                sum(row["family_id"] == family for row in phase_rows) != 64 for family in FAMILIES
            ):
                errors.append("phase_family_balance")
        if sum(bool(row.get("poisoned")) for row in seed_authority) != POISON_COUNT_PER_SEED:
            errors.append("poison_count")
    return list(dict.fromkeys(errors))


def run_commit_path_probe(root: Path) -> JsonDict:
    """Exercise insertion, deletion, stale-parent, and poison behavior on disk."""

    fallback = FrozenWarmupFallback.from_releases(
        [
            {
                "event_id": f"probe-warm-{index}",
                "family_id": "lower_bound",
                "numeric_value": index,
                "observed_label": "reject",
            }
            for index in range(8)
        ]
    )
    reserved = {family: set() for family in FAMILIES}
    reserved["lower_bound"] = {20, 21, 22, 23}
    witness = {
        "event_id": "probe-witness",
        "family_id": "lower_bound",
        "numeric_value": 20,
    }
    memory = transactional.TransactionalConstraintMemory(root / "causal")
    controller = CandidateRefinementController(
        stream_id="probe-stream",
        fallback=fallback,
        reserved_validation=reserved,
    )
    state = controller.families["lower_bound"]
    state.hypotheses = {10}
    state.candidate_parameter = 10
    state.support_ids = ["support-1"]
    state.validation_ids = ["validation-1", "validation-2", "validation-3", "validation-4"]
    state.expected_parent_hash = memory.state_hash()
    parent_bytes = memory.state_bytes()
    before = controller.deployed_predict(witness, memory)
    insertion = controller.commit_candidate("lower_bound", memory=memory)
    after = controller.deployed_predict(witness, memory)
    rollback = memory.rollback(insertion["commit_receipt"])
    deleted = controller.deployed_predict(witness, memory)

    stale_memory = transactional.TransactionalConstraintMemory(root / "stale")
    stale = CandidateRefinementController(
        stream_id="stale-stream",
        fallback=fallback,
        reserved_validation=reserved,
    )
    stale_state = stale.families["lower_bound"]
    stale_state.hypotheses = {10}
    stale_state.candidate_parameter = 10
    stale_state.validation_ids = ["s1", "s2", "s3", "s4"]
    stale_state.expected_parent_hash = "sha256:" + "0" * 64
    stale_before = stale_memory.state_hash()
    stale_result = stale.commit_candidate("lower_bound", memory=stale_memory)

    poison_memory = transactional.TransactionalConstraintMemory(root / "poison")
    poison = CandidateRefinementController(
        stream_id="poison-stream",
        fallback=fallback,
        reserved_validation=reserved,
    )
    poison_state = poison.families["lower_bound"]
    poison_state.hypotheses = {10}
    poison_state.candidate_parameter = 10
    poison_state.validation_ids = ["p1", "p2", "p3", "p4"]
    poison_state.expected_parent_hash = poison_memory.state_hash()
    poison_before = poison_memory.state_hash()
    poison_result = poison.commit_candidate(
        "lower_bound", memory=poison_memory, exact_authority=False
    )
    return {
        "fallback_prediction": before["prediction"],
        "committed_prediction": after["prediction"],
        "deleted_prediction": deleted["prediction"],
        "insertion_changed_decision": before["prediction"] != after["prediction"],
        "deletion_changed_decision": after["prediction"] != deleted["prediction"],
        "uncommitted_hypotheses_consulted": any(
            row["uncommitted_hypotheses_consulted"] for row in (before, after, deleted)
        ),
        "transaction_admitted": insertion["admitted"],
        "transaction_parent_unchanged": insertion["commit_receipt"]["parent_hash"]
        == sha256_bytes(parent_bytes),
        "rollback_byte_identical": rollback["byte_identical"]
        and memory.state_bytes() == parent_bytes,
        "wrong_version_rejected": stale_result["reason"] == "stale_expected_parent_hash"
        and stale_memory.state_hash() == stale_before,
        "poisoned_transaction_rejected": poison_result["admitted"] is False
        and poison_memory.state_hash() == poison_before,
        "poison_quarantine_written": bool(poison_memory.quarantine_entries()),
        "certificate_scope": "empirically_validated_partial_finite_domain",
    }


def unwrap_principled(value: Any) -> Any:
    """Unwrap only the exact two-key principle/value field representation."""

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
    """Keep exact expected and observed values beside every prerequisite."""

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
    """Expose the first failed prerequisite without losing the complete ledger."""

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
    """Check artifact flags and the exclusion manifest as independent gates."""

    flags = {key: unwrap_principled(upstream[key]) for key in _QUARANTINE_KEYS if key in upstream}
    matches = [marker for marker in (artifact_name, task_id) if marker in exclusion_text]
    return {
        "quarantined": any(value is True for value in flags.values()) or bool(matches),
        "declared_flags": flags,
        "exclusion_manifest_matches": matches,
    }


def _read_summary(path: Path) -> JsonDict:
    """Use bounded jq parsing so a large producer row panel is not retained."""

    if not path.is_file():
        return {}
    expression = (
        "{schema,experiment_id,milestone,status,run_date,gate_check_summary,"
        "acquisition_run_complete_score,acquisition_value_score,"
        "acquisition_audit_complete_score,memory_promotion_score,MODEL_SPECS,"
        "model_invoked,source_artifact_hashes,reproducibility_checksum,"
        "artifact_quarantined,upstream_quarantined,quarantine_flag,quarantined,"
        "excluded_from_use,flagged_adversarial} | with_entries(select(.value != null))"
    )
    try:
        completed = subprocess.run(
            ["jq", "-c", expression, str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        value = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating evidence early."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _task_identity(text: str) -> JsonDict:
    """Extract only the Exp7212 block from the executable V635 contract."""

    match = re.search(r"(?ms)^- id: exp7212-refinement-fixture\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(0)
    return {
        "id": "exp7212-refinement-fixture" if block else None,
        "milestone": MILESTONE if f"milestone: {MILESTONE}" in block else None,
        "deliverable": (
            str(DEFAULT_ARTIFACT_PATH) if f"deliverable: {DEFAULT_ARTIFACT_PATH}" in block else None
        ),
    }


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    upstream_7199: Path = DEFAULT_UPSTREAM_7199,
    upstream_7200: Path = DEFAULT_UPSTREAM_7200,
) -> tuple[list[JsonDict], JsonDict, JsonDict, dict[str, str | None]]:
    """Verify sources and exact producer gates before generating hidden data."""

    root = Path(repo_root)
    resolved_7199 = _resolve(root, upstream_7199)
    resolved_7200 = _resolve(root, upstream_7200)
    summary_7199 = _read_summary(resolved_7199)
    summary_7200 = _read_summary(resolved_7200)
    summary_7204 = _read_summary(root / DEFAULT_AUTHENTICATOR_7204)
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
    source_hashes[str(upstream_7199)] = _sha256_path(resolved_7199)
    source_hashes[str(upstream_7200)] = _sha256_path(resolved_7200)
    source_sizes = {
        str(path): (root / path).stat().st_size if (root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    quarantine_7199 = quarantine_state(
        summary_7199,
        exclusion_text,
        DEFAULT_UPSTREAM_7199.name,
        "exp7199-bounded-acquisition",
    )
    quarantine_7200 = quarantine_state(
        summary_7200,
        exclusion_text,
        DEFAULT_UPSTREAM_7200.name,
        "exp7200-acquisition-cold-audit",
    )
    gate_7199 = unwrap_principled(summary_7199.get("gate_check_summary", {}))
    gate_7200 = unwrap_principled(summary_7200.get("gate_check_summary", {}))
    source_map_7200 = unwrap_principled(summary_7200.get("source_artifact_hashes", {}))
    source_map_7204 = unwrap_principled(summary_7204.get("source_artifact_hashes", {}))
    linked_7199_hash = (
        source_map_7200.get(str(DEFAULT_UPSTREAM_7199))
        if isinstance(source_map_7200, Mapping)
        else None
    )
    linked_7200_hash = (
        source_map_7204.get(str(DEFAULT_UPSTREAM_7200))
        if isinstance(source_map_7204, Mapping)
        else None
    )
    expected_identity = {
        "id": "exp7212-refinement-fixture",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT_PATH),
    }
    tools = {
        "python": Path(sys.executable).is_file(),
        "jq": shutil.which("jq") is not None,
        "sha256sum": shutil.which("sha256sum") is not None,
    }
    imports = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "carnot.experiment_7198_v634_feedback_capacity_stream",
            "carnot.experiment_7199_v634_bounded_acquisition",
            "carnot.experiment_7200_v634_acquisition_cold_audit",
            "carnot.memory.transactional_constraint_memory",
        )
    }
    destinations = {
        "stream": _path_writable(paths.public_stream),
        "checkpoint": _path_writable(paths.checkpoint),
        "artifact": _path_writable(paths.artifact),
    }
    citation_present = (
        "2509.24489" in references_text
        and "Query-Driven Interactive Refinement for Constraint Acquisition" in references_text
    )
    checks = [
        gate_check(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-CL-7212",
            True,
            "## REQ-CL-7212:" in spec_text,
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7212-*",
            9,
            spec_text.count("### SCENARIO-CL-7212-"),
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            {str(path): "nonempty" for path in SOURCE_PATHS},
            {key: "nonempty" if value else None for key, value in source_sizes.items()},
        ),
        gate_check(
            "required_source_hashes",
            "repository_and_upstreams",
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
            citation_present,
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
            "stream,checkpoint,artifact",
            {key: True for key in destinations},
            destinations,
        ),
        gate_check(
            "exp7199_status",
            "exp7199-bounded-acquisition",
            "status",
            "complete",
            summary_7199.get("status"),
        ),
        gate_check(
            "exp7199_run_date",
            "exp7199-bounded-acquisition",
            "run_date",
            RUN_DATE,
            summary_7199.get("run_date"),
        ),
        gate_check(
            "exp7199_completion",
            "exp7199-bounded-acquisition",
            "acquisition_run_complete_score",
            1,
            summary_7199.get("acquisition_run_complete_score"),
        ),
        gate_check(
            "exp7199_known_null",
            "exp7199-bounded-acquisition",
            "acquisition_value_score",
            0,
            summary_7199.get("acquisition_value_score"),
        ),
        gate_check(
            "exp7199_gate",
            "exp7199-bounded-acquisition",
            "gate_check_summary.passed",
            True,
            gate_7199.get("passed") if isinstance(gate_7199, Mapping) else None,
        ),
        gate_check(
            "exp7199_no_model",
            "exp7199-bounded-acquisition",
            "MODEL_SPECS,model_invoked",
            {"MODEL_SPECS": [], "model_invoked": False},
            {
                "MODEL_SPECS": unwrap_principled(summary_7199.get("MODEL_SPECS")),
                "model_invoked": unwrap_principled(summary_7199.get("model_invoked")),
            },
        ),
        gate_check(
            "exp7199_not_quarantined",
            "artifact_metadata_and_ops/exclusion_manifest.yaml",
            "quarantined",
            False,
            quarantine_7199["quarantined"],
        ),
        gate_check(
            "exp7200_status",
            "exp7200-acquisition-cold-audit",
            "status",
            "complete",
            summary_7200.get("status"),
        ),
        gate_check(
            "exp7200_run_date",
            "exp7200-acquisition-cold-audit",
            "run_date",
            RUN_DATE,
            summary_7200.get("run_date"),
        ),
        gate_check(
            "exp7200_completion",
            "exp7200-acquisition-cold-audit",
            "acquisition_audit_complete_score",
            1,
            summary_7200.get("acquisition_audit_complete_score"),
        ),
        gate_check(
            "exp7200_known_null",
            "exp7200-acquisition-cold-audit",
            "memory_promotion_score",
            0,
            summary_7200.get("memory_promotion_score"),
        ),
        gate_check(
            "exp7200_gate",
            "exp7200-acquisition-cold-audit",
            "gate_check_summary.passed",
            True,
            gate_7200.get("passed") if isinstance(gate_7200, Mapping) else None,
        ),
        gate_check(
            "exp7200_no_model",
            "exp7200-acquisition-cold-audit",
            "MODEL_SPECS,model_invoked",
            {"MODEL_SPECS": [], "model_invoked": False},
            {
                "MODEL_SPECS": unwrap_principled(summary_7200.get("MODEL_SPECS")),
                "model_invoked": unwrap_principled(summary_7200.get("model_invoked")),
            },
        ),
        gate_check(
            "exp7200_not_quarantined",
            "artifact_metadata_and_ops/exclusion_manifest.yaml",
            "quarantined",
            False,
            quarantine_7200["quarantined"],
        ),
        gate_check(
            "exp7199_hash_authenticated",
            "exp7200.source_artifact_hashes",
            str(DEFAULT_UPSTREAM_7199),
            source_hashes[str(upstream_7199)],
            linked_7199_hash,
        ),
        gate_check(
            "exp7200_hash_authenticated",
            "exp7204.source_artifact_hashes",
            str(DEFAULT_UPSTREAM_7200),
            source_hashes[str(upstream_7200)],
            linked_7200_hash,
        ),
        gate_check(
            "known_failed_values_not_promoted",
            "exp7199_and_exp7200",
            "acquisition_value_score,memory_promotion_score,promoted",
            {"acquisition_value_score": 0, "memory_promotion_score": 0, "promoted": False},
            {
                "acquisition_value_score": unwrap_principled(
                    summary_7199.get("acquisition_value_score")
                ),
                "memory_promotion_score": unwrap_principled(
                    summary_7200.get("memory_promotion_score")
                ),
                "promoted": False,
            },
        ),
    ]
    return checks, summary_7199, summary_7200, source_hashes


def _spawn_evaluator(paths: ExperimentPaths) -> None:
    """Run authority generation in a bounded process with inherited live output."""

    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "carnot.experiment_7212_v635_refinement_fixture",
            "--evaluator-worker",
            "--stream-root",
            str(paths.public_stream.parent),
        ],
        check=True,
        timeout=180,
        env=environment,
    )


def _source_hashes(
    repo_root: Path,
    paths: ExperimentPaths,
    upstream_7199: Path,
    upstream_7200: Path,
) -> dict[str, str | None]:
    """Bind repository inputs, producer artifacts, and all sealed stream views."""

    hashes = {str(path): _sha256_path(repo_root / path) for path in SOURCE_PATHS}
    for path in (
        upstream_7199,
        upstream_7200,
        paths.public_stream,
        paths.authority_sidecar,
        paths.split_feedback_manifest,
        paths.released_warmup,
        paths.controller_serialization,
    ):
        hashes[str(path)] = _sha256_path(_resolve(repo_root, path))
    return hashes


def _query_budget_contract() -> JsonDict:
    """Expose the matched charged resource shared by every downstream arm."""

    return {
        "warmup_released_observations": WARMUP_COUNT,
        "additional_query_limit": QUERY_BUDGET,
        "fitting_query_limit": FITTING_BUDGET,
        "validation_query_limit": VALIDATION_BUDGET,
        "validation_queries_per_family": VALIDATION_PER_FAMILY,
        "pending_capacity": PENDING_CAPACITY,
        "delay_protocol": "fixed_burst_0_4_16_32",
        "all_arms_pay_query_and_delay_costs": True,
        "validation_consumes_total_budget": True,
    }


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    *,
    source_hashes: Mapping[str, Any],
    paths: ExperimentPaths,
    duration_s: float,
) -> JsonDict:
    """Build schema-complete evidence before terminal classification."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "blocked",
        "run_date": RUN_DATE,
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": "no qualifying computation ran because an external prerequisite failed",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": socket.gethostname(),
        "duration_s": duration_s,
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": {
            "planned_events": len(STREAM_SEEDS) * EVENTS_PER_SEED,
            "attempted_events": 0,
            "completed_events": 0,
            "censored_events": len(STREAM_SEEDS) * EVENTS_PER_SEED,
            "independent_units_planned": len(STREAM_SEEDS),
            "independent_units_attempted": 0,
            "independent_units_completed": 0,
            "independent_units_censored": len(STREAM_SEEDS),
        },
        "random_seed": {"fixture": RANDOM_SEED, "stream_seeds": list(STREAM_SEEDS)},
        "reproducibility_checksum": None,
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_external:unknown_precondition",
        "refinement_fixture_ready_score": 0,
        "public_stream_path": str(paths.public_stream),
        "authority_sidecar_path": str(paths.authority_sidecar),
        "query_budget_contract": _query_budget_contract(),
        "commit_path_receipt": {},
        "validation_partition_rows": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "split_feedback_manifest_path": str(paths.split_feedback_manifest),
        "released_warmup_path": str(paths.released_warmup),
        "controller_serialization_path": str(paths.controller_serialization),
        "stream_hashes": {},
        "stream_conformance_errors": [],
        "source_citation_receipt": {
            "citation": "arXiv:2509.24489",
            "local_path": "research-references.md",
            "local_hash": source_hashes.get("research-references.md"),
            "adaptation": "deterministic disagreement witnesses for finite numeric predicates",
            "reproduction_claimed": False,
        },
        "upstream_gate_receipt": {
            "exp7199_acquisition_run_complete_score": None,
            "exp7199_acquisition_value_score": None,
            "exp7200_acquisition_audit_complete_score": None,
            "exp7200_memory_promotion_score": None,
            "known_failed_value_promoted": False,
        },
        "checkpoint_path": str(paths.checkpoint),
        "checkpoint_hash": None,
        "no_model_weight_mutation": True,
        "certificate_scope": "empirically_validated_partial_finite_domain",
        "controller_contract": {
            "fitting_state_private": True,
            "deployed_sources": ["frozen_warmup_fallback", "committed_exact_predicate"],
            "uncommitted_majority_vote_allowed": False,
        },
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    *,
    source_hashes: Mapping[str, Any],
    paths: ExperimentPaths,
    duration_s: float,
) -> JsonDict:
    """Return terminal row-free evidence for an unchanged external block."""

    artifact = _base_artifact(
        checks,
        source_hashes=source_hashes,
        paths=paths,
        duration_s=duration_s,
    )
    failed = artifact["gate_check_summary"]["failed_check"] or "unknown_precondition"
    artifact["honest_verdict"] = f"blocked_external:{failed}"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable inputs, settings, receipts, and raw comparison rows."""

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
        "refinement_fixture_ready_score",
        "query_budget_contract",
        "commit_path_receipt",
        "validation_partition_rows",
        "stream_hashes",
        "stream_conformance_errors",
        "source_citation_receipt",
        "upstream_gate_receipt",
        "no_model_weight_mutation",
        "certificate_scope",
        "controller_contract",
    )
    return sha256_json({name: artifact.get(name) for name in fields})


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    check_files: bool = False,
) -> list[str]:
    """Cold-check schema, causal receipts, budgets, hashes, and terminal class."""

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
    add(artifact["no_model_weight_mutation"] is not True, "weight_mutation")
    add(artifact["verifier_is_oracle"] is not True, "oracle_classification")
    add(artifact["query_budget_contract"] != _query_budget_contract(), "query_budget_contract")
    blocked = artifact["verdict_class"] == "blocked"
    if blocked:
        add(artifact["status"] != "blocked", "blocked_status")
        add(artifact["inference_substrate_class"] != "blocked_no_run", "blocked_substrate")
        add(artifact["refinement_fixture_ready_score"] != 0, "blocked_readiness")
        add(bool(artifact["rows"]), "blocked_rows")
        summary = artifact["gate_check_summary"]
        add(summary.get("passed") is not False, "blocked_gate_passed")
        for key in ("failed_check", "upstream", "field", "expected_value", "observed_value"):
            add(summary.get(key) is None, f"blocked_gate_{key}")
    else:
        add(artifact["status"] != "complete", "status")
        add(artifact["verdict_class"] != "circular_positive", "verdict_class")
        add(artifact["inference_substrate"] != INFERENCE_SUBSTRATE, "inference_substrate")
        add(
            artifact["inference_substrate_class"] != INFERENCE_SUBSTRATE_CLASS,
            "inference_substrate_class",
        )
        add(artifact["gate_check_summary"].get("passed") is not True, "precondition_gate")
        add(len(artifact["rows"]) != len(STREAM_SEEDS) * len(ARMS), "row_count")
        add(
            any(
                not {"unit_id", "arm", "seed", "metric", "error", "abstention"} <= set(row)
                for row in artifact["rows"]
            ),
            "row_schema",
        )
        add(bool(artifact["stream_conformance_errors"]), "stream_conformance")
        receipt = artifact["commit_path_receipt"]
        add(
            not all(
                receipt.get(name) is True
                for name in (
                    "insertion_changed_decision",
                    "deletion_changed_decision",
                    "transaction_admitted",
                    "transaction_parent_unchanged",
                    "rollback_byte_identical",
                    "wrong_version_rejected",
                    "poisoned_transaction_rejected",
                    "poison_quarantine_written",
                )
            )
            or receipt.get("uncommitted_hypotheses_consulted") is not False,
            "commit_path",
        )
        add(
            len(artifact["validation_partition_rows"]) != len(STREAM_SEEDS) * len(FAMILIES),
            "validation_partition_count",
        )
        add(
            any(
                not set(row["reserved_validation_x"]).isdisjoint(row["fitting_x_domain"])
                or row["validation_used_for_elimination"] is not False
                for row in artifact["validation_partition_rows"]
            ),
            "validation_partition",
        )
        add(
            artifact["upstream_gate_receipt"].get("known_failed_value_promoted") is not False,
            "prior_null_promoted",
        )
        expected_ready = int(not errors)
        add(artifact["refinement_fixture_ready_score"] != expected_ready, "readiness")
        if check_files:
            root = repo_root or REPO_ROOT
            for path_text, expected_hash in artifact["stream_hashes"].items():
                add(_sha256_path(_resolve(root, path_text)) != expected_hash, "stream_hashes")
            add(
                _sha256_path(_resolve(root, artifact["checkpoint_path"]))
                != artifact["checkpoint_hash"],
                "checkpoint_hash",
            )
            for path_text, expected_hash in artifact["source_artifact_hashes"].items():
                add(_sha256_path(_resolve(root, path_text)) != expected_hash, "source_hashes")
    add(
        artifact["reproducibility_checksum"] != reproducibility_checksum(artifact),
        "reproducibility_checksum",
    )
    return errors


def _comparison_rows() -> list[JsonDict]:
    """Record fixture integrity per arm without inventing learning benefit."""

    return [
        {
            "unit_id": f"{seed}:{arm}",
            "arm": arm,
            "seed": seed,
            "metric": "fixture_contract_pass",
            "error": 0,
            "abstention": 0,
            "event_count": EVENTS_PER_SEED,
            "learning_value_measured": False,
        }
        for seed in STREAM_SEEDS
        for arm in ARMS
    ]


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    upstream_7199: Path = DEFAULT_UPSTREAM_7199,
    upstream_7200: Path = DEFAULT_UPSTREAM_7200,
    duration_s: float | None = None,
    progress: bool = False,
) -> JsonDict:
    """Run preflight, isolated generation, causal probes, and cold validation."""

    started = time.monotonic()
    if progress:
        print(
            "PHASE 0 CHECK: verify spec, sources, tools, gates, quarantine, and paths", flush=True
        )
    checks, summary_7199, summary_7200, initial_hashes = collect_preconditions(
        repo_root,
        paths,
        upstream_7199=upstream_7199,
        upstream_7200=upstream_7200,
    )
    if not all(row["passed"] for row in checks):
        elapsed = time.monotonic() - started if duration_s is None else duration_s
        if progress:
            print(
                "PHASE 0 END: external prerequisite failed; no fixture computation ran", flush=True
            )
        return build_blocked_artifact(
            checks,
            source_hashes=initial_hashes,
            paths=paths,
            duration_s=elapsed,
        )
    if progress:
        print(
            "PHASE 0 END: exact producer gates passed and known nulls stayed unpromoted", flush=True
        )
        print("PHASE 1 START: launch bounded evaluator-only stream process", flush=True)
    _spawn_evaluator(paths)
    if progress:
        print("PHASE 1 END: evaluator-only process completed and sealed its views", flush=True)
        print(
            "PHASE 2 START: cold-check stream separation, balance, labels, and budgets", flush=True
        )
    public = read_jsonl(paths.public_stream)
    authority = read_jsonl(paths.authority_sidecar)
    warmup = read_jsonl(paths.released_warmup)
    manifest = json.loads(paths.split_feedback_manifest.read_text(encoding="utf-8"))
    conformance = stream_conformance_errors(public, authority, warmup, manifest)
    if conformance:
        raise ValueError("stream_conformance_failed:" + ",".join(conformance))
    if progress:
        print("PHASE 2 END: 20 sealed streams and hidden audit panels passed", flush=True)
        print(
            "PHASE 3 START: exercise committed insertion, deletion, stale parent, and poison",
            flush=True,
        )
    probe_root = paths.checkpoint.parent / "experiment_7212_transaction_probe"
    commit_receipt = run_commit_path_probe(probe_root)
    if progress:
        print(
            "PHASE 3 END: committed memory is causally active and both attacks reject", flush=True
        )
        print("PHASE 4 START: write checkpoint and assemble per-arm fixture rows", flush=True)
    rows = _comparison_rows()
    checkpoint = {
        "schema": "carnot.exp7212.progress.v1",
        "completed_streams": list(STREAM_SEEDS),
        "completed_event_count": len(public),
        "completed_unit_count": len(rows),
        "stream_contract_hash": sha256_json(
            {
                "public": _sha256_path(paths.public_stream),
                "authority": _sha256_path(paths.authority_sidecar),
                "manifest": _sha256_path(paths.split_feedback_manifest),
            }
        ),
    }
    _atomic_write(paths.checkpoint, canonical_json_bytes(checkpoint))
    source_hashes = _source_hashes(repo_root, paths, upstream_7199, upstream_7200)
    elapsed = time.monotonic() - started if duration_s is None else duration_s
    artifact = _base_artifact(
        checks,
        source_hashes=source_hashes,
        paths=paths,
        duration_s=elapsed,
    )
    stream_paths = (
        paths.public_stream,
        paths.authority_sidecar,
        paths.split_feedback_manifest,
        paths.released_warmup,
        paths.controller_serialization,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "rows": rows,
            "sample_size_budget": {
                "planned_events": len(public),
                "attempted_events": len(public),
                "completed_events": len(public),
                "censored_events": 0,
                "independent_units_planned": len(STREAM_SEEDS),
                "independent_units_attempted": len(STREAM_SEEDS),
                "independent_units_completed": len(STREAM_SEEDS),
                "independent_units_censored": 0,
                "hidden_audit_rows_completed": sum(
                    row.get("row_type") == "audit_label" for row in authority
                ),
            },
            "verdict_class": "circular_positive",
            "honest_verdict": "complete: refinement fixture ready; learning value is unmeasured and V634 nulls remain unpromoted",
            "refinement_fixture_ready_score": 1,
            "commit_path_receipt": commit_receipt,
            "validation_partition_rows": list(manifest["validation_partitions"]),
            "stream_hashes": {str(path): _sha256_path(path) for path in stream_paths},
            "stream_conformance_errors": conformance,
            "source_citation_receipt": {
                "citation": "arXiv:2509.24489",
                "local_path": "research-references.md",
                "local_hash": source_hashes["research-references.md"],
                "adaptation": "deterministic disagreement witnesses for finite numeric predicates",
                "reproduction_claimed": False,
            },
            "upstream_gate_receipt": {
                "exp7199_acquisition_run_complete_score": unwrap_principled(
                    summary_7199.get("acquisition_run_complete_score")
                ),
                "exp7199_acquisition_value_score": unwrap_principled(
                    summary_7199.get("acquisition_value_score")
                ),
                "exp7200_acquisition_audit_complete_score": unwrap_principled(
                    summary_7200.get("acquisition_audit_complete_score")
                ),
                "exp7200_memory_promotion_score": unwrap_principled(
                    summary_7200.get("memory_promotion_score")
                ),
                "known_failed_value_promoted": False,
            },
            "checkpoint_hash": _sha256_path(paths.checkpoint),
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact, repo_root=repo_root, check_files=True)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    if progress:
        print("PHASE 4 END: checkpoint and 100 non-benefit fixture rows completed", flush=True)
        print("PHASE 5 END: cold artifact validation passed", flush=True)
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed date and explicit evaluator or test destinations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--upstream-7199", type=Path, default=DEFAULT_UPSTREAM_7199)
    parser.add_argument("--upstream-7200", type=Path, default=DEFAULT_UPSTREAM_7200)
    parser.add_argument("--evaluator-worker", action="store_true")
    parser.add_argument("--stream-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run real gates and publish one validated terminal artifact atomically."""

    print("PHASE 0 START: parse fixed execution inputs before all checks", flush=True)
    args = _parse_args(argv)
    if str(args.date) != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    if args.evaluator_worker:
        if args.stream_root is None:
            raise ValueError("evaluator_worker_requires_stream_root")
        return evaluator_worker(ExperimentPaths.from_stream_root(args.stream_root))
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    artifact = build_and_seal(
        REPO_ROOT,
        paths,
        upstream_7199=args.upstream_7199,
        upstream_7200=args.upstream_7200,
        progress=True,
    )
    print("PHASE 5 START: cold-check the complete terminal object before publication", flush=True)
    errors = validate_artifact(
        artifact,
        repo_root=REPO_ROOT,
        check_files=artifact["verdict_class"] != "blocked",
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    print("PHASE 5 END: terminal object passed cold validation", flush=True)
    print("PHASE 6 START: final atomic artifact write", flush=True)
    print("FINAL ATOMIC WRITE START", flush=True)
    _atomic_write(paths.artifact, canonical_json_bytes(artifact))
    print("FINAL ATOMIC WRITE END", flush=True)
    print("PHASE 6 END: terminal deliverable is stable", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper owns normal execution.
    raise SystemExit(main())
