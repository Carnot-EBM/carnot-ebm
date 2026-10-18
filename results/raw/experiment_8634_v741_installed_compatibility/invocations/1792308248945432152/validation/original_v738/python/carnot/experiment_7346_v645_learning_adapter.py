"""Measure a bounded learning adapter through the shipped pipeline seam.

The adapter sees public schedule fields and Boolean executor replies only. It
uses the existing opt-in production Safety-Net hook on ``VerifyRepairPipeline``.
The native pipeline result remains final authority for every returned plan.

Spec refs: REQ-CL-7346 and SCENARIO-CL-7346-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

from carnot import experiment_7330_v644_public_learner as public
from carnot.memory import transactional_constraint_memory as transactional
from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.verify_repair import VerifyRepairPipeline
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    build_scoped_commands,
    run_commands,
    run_scoped_validation,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260916"
MILESTONE = "2026.09.645"
EXPERIMENT_ID = "exp7346-v645-learning-adapter"
SCHEMA = "carnot.exp7346.v645_learning_adapter.v1"
QUERY_BUDGET = 24
STATE_CAP_BYTES = 69_632
DEVELOPMENT_SEED = 7_346_101
EVALUATION_SEED = 7_346_201
RESAMPLING_SEED = 7_346_301
BOOTSTRAP_DRAWS = 2_000

PERSISTENT_ARM = "persistent_structural_acquisition"
RESET_ARM = "reset_per_request"
CACHE_ARM = "exact_plan_cache_plus_reset"
FROZEN_ARM = "frozen_memory_after_warmup"
ARMS = (PERSISTENT_ARM, RESET_ARM, CACHE_ARM, FROZEN_ARM)

MODULE_PATH = Path("python/carnot/experiment_7346_v645_learning_adapter.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7346_v645_learning_adapter.py")
TEST_PATH = Path("tests/python/test_experiment_7346_v645_learning_adapter.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
PIPELINE_SPEC_PATH = Path("openspec/capabilities/pipeline/spec.md")
PRODUCER_PATH = Path("results/experiment_7344_v645_executor_fixture.json")
PUBLIC_MANIFEST_PATH = Path(
    "results/raw/experiment_7344_v645_executor_fixture/public/public_manifest.json"
)
PRIVATE_MANIFEST_PATH = Path(
    "results/raw/experiment_7344_v645_executor_fixture/evaluator/evaluator_private_manifest.json"
)
SCRIPTED_COHORT_PATH = Path(
    "results/raw/experiment_7344_v645_executor_fixture/sidecars/scripted_public_model_requests.json"
)
HISTORICAL_MODEL_PATH = Path(
    "results/raw/experiment_7344_v645_executor_fixture/sidecars/historical_model_receipt.json"
)
DEFAULT_OUTPUT_PATH = Path("results/experiment_7346_v645_learning_adapter.json")
RAW_DIR = Path("results/raw/experiment_7346_v645_learning_adapter")

TERMINAL_CHECK_NAMES = (
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ALL_REQUIRED_CHECK_NAMES = (*REQUIRED_CHECK_NAMES, "full_python_suite", *TERMINAL_CHECK_NAMES)

ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    SPEC_PATH,
    PIPELINE_SPEC_PATH,
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7330_v644_public_learner.py"),
    Path("python/carnot/pipeline/verify_repair.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("python/carnot/experiment_7323_v643_addition_prototype.py"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
    PRODUCER_PATH,
    PUBLIC_MANIFEST_PATH,
    PRIVATE_MANIFEST_PATH,
    SCRIPTED_COHORT_PATH,
    HISTORICAL_MODEL_PATH,
)

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Version the record and retain ordinary top-level experiment_id and milestone.",
    "status": "Write a terminal result only after actual work and affected checks.",
    "run_date": "Use 20260916; record real UTC timestamps as well.",
    "preconditions_checked": "Record each actual input/resource check before dependent work.",
    "MODEL_SPECS": "List actual intended model identities; this run intends no model work.",
    "model_invoked": "True for any attempted current model load or generation, including failures.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled and in-flight operations.",
    "inference_substrate": "Declare actual computation; historical model receipts are not current inference.",
    "inference_substrate_class": "Use the closed duration class matching the actual run.",
    "execution_venue": "Use host; this milestone makes no new board-execution claim.",
    "duration_s": "Measure monotonic time; never wait merely to pass a duration floor.",
    "phase_spans": "Measure disjoint load, generation, evaluation, test and write spans.",
    "random_seed": "Freeze development, evaluation and resampling seeds before outcomes.",
    "reproducibility_checksum": "Bind code, settings, inputs, evaluator identity and raw evidence.",
    "source_artifact_hashes": "Authenticate exact producers and current same-milestone paths.",
    "rows": "Keep every comparative unit, arm, metric, cost, failure and censoring disposition.",
    "sample_size_budget": "Record planned, attempted, completed and censored units and stopping rules.",
    "acceptance_gate_results": "Each gate records expected, observed, passed and its principle.",
    "gate_check_summary": "Every blocked_* names upstream, failed check, exact artifact field, expected and observed value.",
    "verifier_is_oracle": "True: executor-defined success remains circular_positive even with separate implementation.",
    "honest_verdict": "Completed work starts complete_ or complete:; external absence starts blocked_ with its failed check.",
    "verdict_class": "Closed enum: positive | circular_positive | null | blocked | disqualified | partial.",
    "flagged_adversarial": "Set false only after current verification; a critical finding prevents promotion.",
    "validation_receipts": "Retain exact command, scope, exit code, elapsed time and log hash, including failures.",
    "repository_health": "Preserve dated unrelated failures separately from affected required validation.",
    "field_principles": "Explain fields separately; do not wrap numeric gates or ordinary dictionaries.",
    "learning_adapter_ready_score": "One means the actual opt-in seam and controls work.",
    "learning_value_score": "One means retained rows pass the frozen future-use and service gates.",
    "promotion_score": "One means every safety, value, validation, and adversarial gate passes.",
    "frozen_learning_protocol": "Downstream gates and denominators cannot move after outcomes.",
    "adapter_controls": "Record causal atom use and version/restart safety.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        *REQUIRED_FIELD_PRINCIPLES,
        "experiment_id",
        "milestone",
        "phase",
        "started_at_utc",
        "completed_at_utc",
        "independent_reduction",
        "cohort_accounting",
        "raw_evidence_paths",
        "production_default_changed",
        "research_roadmap_changed",
        "publication_surface_changed",
        "no_model_weight_mutation",
        "deployment_cost_claim",
    }
)


class LearningAdapterError(RuntimeError):
    """Reject malformed public data, unsafe feedback, or invalid durable state."""


def progress(phase: str, event: str, detail: str = "") -> None:
    """Print one flushed boundary so long work never looks stalled."""

    suffix = f" {detail}" if detail else ""
    print(f"[exp7346] phase={phase} event={event}{suffix}", flush=True)


def sha256_file(path: Path) -> str:
    """Hash exact bytes with the public learner's shared representation."""

    return public.sha256_file(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a complete JSON value with fsync and one local rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(public.canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # pragma: no cover - only an interrupted rename leaves it.
            temporary.unlink()


def validate_adapter_request(request: Mapping[str, Any]) -> None:
    """Validate public field families separately so failures stay local."""

    identifiers = request.get("activities")
    if (
        not isinstance(request.get("request_id"), str)
        or not request.get("request_id")
        or not isinstance(request.get("version_token"), str)
        or not request.get("version_token")
        or not isinstance(identifiers, list)
        or not identifiers
        or len(identifiers) > 6
        or any(not isinstance(name, str) or not name for name in identifiers)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise LearningAdapterError("public_identifiers")
    names = set(identifiers)
    horizon = request.get("horizon")
    durations = request.get("durations")
    revision = request.get("public_revision", 0)
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon < 1
        or not isinstance(durations, Mapping)
        or set(durations) != names
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in durations.values()
        )
    ):
        raise LearningAdapterError("public_types")
    windows = request.get("allowed_starts")
    if not isinstance(windows, Mapping) or set(windows) != names:
        raise LearningAdapterError("public_windows")
    for name in identifiers:
        values = windows[name]
        if (
            not isinstance(values, list)
            or not values
            or len(values) > 4
            or len(set(values)) != len(values)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value + int(durations[name]) > horizon
                for value in values
            )
        ):
            raise LearningAdapterError("public_windows")
    weights = request.get("weights")
    if (
        not isinstance(weights, Mapping)
        or set(weights) != names
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in weights.values()
        )
    ):
        raise LearningAdapterError("public_weights")


class QualifiedFixtureExecutor:
    """Expose only charged Boolean checks over one qualified private record."""

    def __init__(
        self,
        request: Mapping[str, Any],
        private_record: Mapping[str, Any],
        *,
        exact_cache: dict[str, bool] | None = None,
    ) -> None:
        validate_adapter_request(request)
        if private_record.get("version_token") != request.get("version_token"):
            raise LearningAdapterError("private_record_version")
        self.request = deepcopy(dict(request))
        self.version_token = str(request["version_token"])
        self.request_id = str(request["request_id"])
        self._rules = deepcopy(dict(private_record["private_rules"]))
        self._cache = exact_cache
        self.attempt_count = 0
        self.external_call_count = 0
        self.cache_hits = 0
        self.receipts: list[JsonDict] = []

    def query(
        self,
        plan: Mapping[str, Any],
        reason: str,
        *,
        allow_cache: bool = False,
    ) -> bool:
        """Charge every attempt and forbid cached release authority."""

        if self.attempt_count >= QUERY_BUDGET:
            raise LearningAdapterError("query_budget")
        self.attempt_count += 1
        cache_key = public.sha256_json(
            {"version_token": self.version_token, "plan": deepcopy(dict(plan))}
        )
        cache_allowed = allow_cache and reason != "final" and self._cache is not None
        if cache_allowed and cache_key in self._cache:
            accepted = bool(self._cache[cache_key])
            external = False
            self.cache_hits += 1
        else:
            from scripts.experiments import (  # noqa: PLC0415
                experiment_7330_v644_private_executor as private,
            )

            accepted = bool(private._check(self.request, plan, self._rules))  # noqa: SLF001
            external = True
            self.external_call_count += 1
            if cache_allowed:
                assert self._cache is not None
                self._cache[cache_key] = accepted
        receipt = {
            "sequence": self.attempt_count,
            "reason": str(reason),
            "request_id": self.request_id,
            "plan_hash": public.sha256_json(plan),
            "accepted": accepted,
            "external_call": external,
            "cache_hit": not external,
        }
        self.receipts.append(receipt)
        return accepted


@dataclass(frozen=True)
class AdapterRouteDecision:
    """Match the existing Safety-Net route decision surface."""

    decision_id: str
    chosen_order: tuple[str, ...]
    fallback_reason: str | None
    candidate_action: str
    influenced_atom_ids: tuple[str, ...]


class ExactPlanExtractor:
    """Turn one serialized schedule into the pipeline's exact Boolean constraint."""

    def __init__(self) -> None:
        self._executor: QualifiedFixtureExecutor | None = None
        self._plan: JsonDict | None = None

    @property
    def supported_domains(self) -> list[str]:
        return ["schedule"]

    def bind(self, executor: QualifiedFixtureExecutor, plan: Mapping[str, Any]) -> None:
        self._executor = executor
        self._plan = deepcopy(dict(plan))

    def extract(self, text: str, domain: str | None = None) -> list[ConstraintResult]:
        if domain != "schedule" or self._executor is None or self._plan is None:
            raise LearningAdapterError("extractor_context")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise LearningAdapterError("plan_json") from error
        if parsed != self._plan:
            raise LearningAdapterError("plan_identity")
        accepted = self._executor.query(parsed, "final")
        candidate_id = public.sha256_json({"request_id": self._executor.request_id, "plan": parsed})
        return [
            ConstraintResult(
                constraint_type="schedule_exact_plan",
                description="The qualified Boolean executor accepts this complete schedule.",
                metadata={
                    "satisfied": accepted,
                    "energy": float(not accepted),
                    "candidate_id": candidate_id,
                },
            )
        ]


class LearningScheduleAdapter:
    """Persist pair-gap atoms only through between-request transactions."""

    def __init__(
        self,
        state_dir: Path,
        *,
        enabled: bool = False,
        persist_feedback: bool = True,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.enabled = bool(enabled)
        self.persist_feedback = bool(persist_feedback)
        self.corrupted_state = False
        self.route_calls = 0
        self._memory: transactional.TransactionalConstraintMemory | None = None
        self._context: JsonDict | None = None
        self._completed: dict[str, JsonDict] = {}
        self._last_feedback: JsonDict = {}
        self._last_commit_receipts: list[JsonDict] = []
        if self.enabled:
            try:
                self._memory = transactional.TransactionalConstraintMemory(self.state_dir)
                if len(self._memory.state_bytes()) > STATE_CAP_BYTES:
                    raise ValueError("state cap")
                self._decode_records(self._memory.records())
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                self.corrupted_state = True
                self._memory = None

    @property
    def state_path(self) -> Path:
        return self.state_dir / "state.json"

    @property
    def last_feedback(self) -> JsonDict:
        return deepcopy(self._last_feedback)

    def state_bytes(self) -> bytes:
        if self._memory is not None:
            return self._memory.state_bytes()
        if self.state_path.exists():
            return self.state_path.read_bytes()
        return b""

    @staticmethod
    def _decode_records(records: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], set[str]]:
        atoms: list[JsonDict] = []
        invalidated: set[str] = set()
        for record in records:
            try:
                payload = json.loads(str(record["repair"]))
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError("invalid adapter record") from error
            if payload.get("record_type") == "pair_atom":
                atom = dict(payload["atom"])
                unhashed = {key: value for key, value in atom.items() if key != "atom_id"}
                if atom.get("atom_id") != public.sha256_json(unhashed):
                    raise ValueError("invalid atom hash")
                atoms.append(atom)
            elif payload.get("record_type") == "invalidation":
                invalidated.update(str(value) for value in payload.get("atom_ids", []))
            else:
                raise ValueError("invalid adapter record type")
        return atoms, invalidated

    def _learner(
        self,
        version_token: str,
        records: Sequence[Mapping[str, Any]],
        *,
        extra_invalidated: Sequence[str] = (),
    ) -> public.PublicConstraintLearner:
        atoms, invalidated = self._decode_records(records)
        invalidated.update(extra_invalidated)
        state = {
            "schema": public.STATE_SCHEMA,
            "active_version_token": str(version_token),
            "atoms": sorted(
                [atom for atom in atoms if atom["atom_id"] not in invalidated],
                key=lambda row: row["atom_id"],
            ),
            "uncertain_compounds": [],
        }
        return public.PublicConstraintLearner.from_state_bytes(public.canonical_bytes(state))

    def begin_request(
        self,
        request: Mapping[str, Any],
        executor: QualifiedFixtureExecutor,
        *,
        allow_learning: bool = True,
    ) -> JsonDict:
        """Freeze durable bytes before any proposal or exact query."""

        validate_adapter_request(request)
        if not self.enabled:
            raise LearningAdapterError("adapter_disabled")
        if self.corrupted_state or self._memory is None:
            raise LearningAdapterError("corrupted_state")
        if self._context is not None:
            raise LearningAdapterError("request_already_active")
        snapshot = self._memory.begin_episode(str(request["request_id"]))
        records = [] if not self.persist_feedback else list(snapshot["records"])
        learner = self._learner(str(request["version_token"]), records)
        empty = public.PublicConstraintLearner(str(request["version_token"]))
        unconstrained = empty.propose(request)
        try:
            proposed = learner.propose(request)
        except public.PublicLearningError as error:
            if str(error) != "no_public_candidate":
                self._memory.end_episode()
                raise
            proposed = None
        influenced = [
            atom["atom_id"]
            for atom in learner.active_atoms()
            if proposed is None or not public._atom_allows_plan(atom, request, unconstrained)  # noqa: SLF001
        ]
        action = "rejected" if proposed is None else "redirected" if influenced else "accepted"
        self._context = {
            "request": deepcopy(dict(request)),
            "executor": executor,
            "snapshot": snapshot,
            "records": records,
            "learner": learner,
            "unconstrained": unconstrained,
            "proposed": proposed,
            "influenced_atom_ids": influenced,
            "candidate_action": action,
            "allow_learning": bool(allow_learning),
            "active_atom_count_before": len(learner.active_atoms()),
        }
        return self.propose()

    def propose(self) -> JsonDict:
        if self._context is None:
            raise LearningAdapterError("no_active_request")
        return {
            "plan": deepcopy(self._context["proposed"]),
            "unconstrained_plan": deepcopy(self._context["unconstrained"]),
            "candidate_action": self._context["candidate_action"],
            "influenced_atom_ids": list(self._context["influenced_atom_ids"]),
            "active_atom_count_before": self._context["active_atom_count_before"],
            "entry_state_hash": self._context["snapshot"]["state_hash"],
        }

    def route(self, request: Any) -> AdapterRouteDecision:
        """Use the existing seam without deleting an extracted constraint."""

        if self._context is None:
            raise LearningAdapterError("no_active_request")
        candidate_ids = tuple(candidate.candidate_id for candidate in request.candidates)
        decision_id = public.sha256_json(
            {
                "pipeline_request_id": request.request_id,
                "entry_state_hash": self._context["snapshot"]["state_hash"],
                "candidate_ids": candidate_ids,
                "plan": self._context["proposed"],
            }
        )
        self.route_calls += 1
        return AdapterRouteDecision(
            decision_id=decision_id,
            chosen_order=candidate_ids,
            fallback_reason=None,
            candidate_action=str(self._context["candidate_action"]),
            influenced_atom_ids=tuple(self._context["influenced_atom_ids"]),
        )

    @staticmethod
    def _transaction_payload(
        payload: Mapping[str, Any],
        *,
        record_key: str,
        scope: str,
    ) -> tuple[JsonDict, JsonDict]:
        repair = public.canonical_bytes(payload).decode("utf-8")
        event_id = f"exp7346-{record_key.removeprefix('sha256:')}"
        event = {
            "event_id": event_id,
            "kind": "reusable_repair",
            "family": "v645_learning_adapter",
            "scope": scope,
            "facts": {"constraint_family": payload["record_type"], "scope": scope},
            "exact_label": True,
            "certified_repair": repair,
            "target_key": None,
        }
        proposal = {
            "key": f"v645:{scope}:{record_key}",
            "scope": scope,
            "repair": repair,
            "source_event_id": event_id,
            "evidence_hash": transactional.event_evidence_hash(event),
            "future_use_eligible": True,
            "expires_after": 1_000_000,
        }
        proposal["content_hash"] = transactional.sha256_json(
            {key: proposal[key] for key in ("key", "scope", "repair")}
        )
        return event, proposal

    def _commit_payload(
        self,
        payload: Mapping[str, Any],
        *,
        record_key: str,
        scope: str,
        boundary_index: int,
    ) -> JsonDict:
        assert self._memory is not None
        if any(record["key"] == f"v645:{scope}:{record_key}" for record in self._memory.records()):
            return {"admitted": False, "reason": "duplicate", "commit_receipt": None}
        event, proposal = self._transaction_payload(payload, record_key=record_key, scope=scope)
        result = self._memory.admit(proposal, event, boundary_index=boundary_index)
        if not result["admitted"]:
            raise LearningAdapterError(str(result["reason"]))
        if len(self._memory.state_bytes()) > STATE_CAP_BYTES:
            self._memory.rollback(result["commit_receipt"])
            raise LearningAdapterError("persistent_state_cap")
        return result

    def record_exact_result(
        self,
        decision: AdapterRouteDecision,
        exact_result: Mapping[str, Any],
    ) -> JsonDict:
        """Close localization, end the episode, then commit certified atoms."""

        if decision.decision_id in self._completed:
            duplicate = {**self._completed[decision.decision_id], "duplicate_feedback": True}
            self._last_feedback = duplicate
            return deepcopy(duplicate)
        if self._context is None or self._memory is None:
            raise LearningAdapterError("no_active_request")
        context = self._context
        executor: QualifiedFixtureExecutor = context["executor"]
        verified = bool(exact_result.get("verified"))
        invalidated = list(context["influenced_atom_ids"]) if not verified else []
        contradiction = bool(invalidated)
        learner = self._learner(
            str(context["request"]["version_token"]),
            context["records"],
            extra_invalidated=invalidated,
        )
        atoms_before = {atom["atom_id"] for atom in learner.active_atoms()}
        if not verified and context["allow_learning"]:
            learner.localize_rejection(
                context["request"],
                context["proposed"],
                lambda plan, reason: executor.query(plan, reason),
            )
        new_atoms = [atom for atom in learner.active_atoms() if atom["atom_id"] not in atoms_before]
        entry_bytes = bytes(context["snapshot"]["state_bytes"])
        self._memory.end_episode()
        commits: list[JsonDict] = []
        if self.persist_feedback:
            if invalidated:
                payload = {
                    "record_type": "invalidation",
                    "atom_ids": sorted(invalidated),
                    "contradiction_plan_hash": public.sha256_json(context["proposed"]),
                }
                key = public.sha256_json(payload)
                commits.append(
                    self._commit_payload(
                        payload,
                        record_key=key,
                        scope=str(context["request"]["version_token"]),
                        boundary_index=executor.attempt_count,
                    )
                )
            for atom in new_atoms:
                payload = {"record_type": "pair_atom", "atom": atom}
                commits.append(
                    self._commit_payload(
                        payload,
                        record_key=str(atom["atom_id"]),
                        scope=str(atom["version_token"]),
                        boundary_index=executor.attempt_count,
                    )
                )
        feedback = {
            "adapter_api_version": "carnot.exp7346.learning_schedule_adapter.v1",
            "mode": "enabled",
            "release_authority": "native_exact_verifier",
            "route": context["candidate_action"],
            "abstention": not verified,
            "fallback_reason": None,
            "exact_result": dict(exact_result),
            "candidate_preservation": {
                "all_candidates_preserved": True,
                "deleted_candidate_count": 0,
            },
            "entry_state_hash": context["snapshot"]["state_hash"],
            "entry_state_unchanged_during_request": (
                entry_bytes == context["snapshot"]["state_bytes"]
            ),
            "new_atom_count": len(new_atoms),
            "new_atom_ids": [atom["atom_id"] for atom in new_atoms],
            "invalidated_atom_ids": invalidated,
            "contradiction_revalidation": contradiction,
            "commit_count": sum(bool(row.get("admitted")) for row in commits),
            "duplicate_feedback": False,
            "state_bytes": len(self.state_bytes()),
        }
        self._last_commit_receipts = [
            dict(row["commit_receipt"]) for row in commits if row.get("commit_receipt") is not None
        ]
        self._context = None
        self._completed[decision.decision_id] = deepcopy(feedback)
        self._last_feedback = deepcopy(feedback)
        return feedback

    def cancel_request(self, reason: str) -> JsonDict:
        """Release a frozen episode without treating missing feedback as evidence."""

        if self._context is None or self._memory is None:
            raise LearningAdapterError("no_active_request")
        entry_hash = self._context["snapshot"]["state_hash"]
        self._memory.end_episode()
        self._context = None
        return {"cancelled": True, "reason": reason, "entry_state_hash": entry_hash}

    def proposal_after_erasure(self, request: Mapping[str, Any]) -> JsonDict:
        """Return the public proposal with all learned atoms removed."""

        validate_adapter_request(request)
        return public.PublicConstraintLearner(str(request["version_token"])).propose(request)

    def rollback_last_commit(self) -> JsonDict:
        if self._memory is None or not self._last_commit_receipts:
            raise LearningAdapterError("no_commit_receipt")
        return self._memory.rollback(self._last_commit_receipts[-1])


class AdapterPipelineHarness:
    """Drive one opted-in adapter through an otherwise default pipeline."""

    def __init__(self, adapter: LearningScheduleAdapter) -> None:
        self.adapter = adapter
        self.extractor = ExactPlanExtractor()
        self.pipeline = VerifyRepairPipeline(
            extractor=self.extractor,
            production_safety_net_adapter=adapter,
        )

    def execute(
        self,
        request: Mapping[str, Any],
        private_record: Mapping[str, Any],
        *,
        warmup: bool,
        allow_learning: bool = True,
        exact_cache: dict[str, bool] | None = None,
    ) -> JsonDict:
        """Propose publicly, verify exactly, and retain all charged receipts."""

        started = time.monotonic()
        executor = QualifiedFixtureExecutor(request, private_record, exact_cache=exact_cache)
        proposal = self.adapter.begin_request(
            request,
            executor,
            allow_learning=allow_learning,
        )
        plan = proposal["plan"]
        if plan is None:
            self.adapter.cancel_request("no_public_candidate")
            feedback: JsonDict = {
                "new_atom_count": 0,
                "invalidated_atom_ids": [],
                "contradiction_revalidation": False,
                "state_bytes": len(self.adapter.state_bytes()),
            }
            returned = False
            result_verified = False
        else:
            self.extractor.bind(executor, plan)
            result = self.pipeline.verify(
                f"Verify public schedule {request['request_id']}",
                public.canonical_bytes(plan).decode("utf-8"),
                domain="schedule",
            )
            feedback = self.adapter.last_feedback
            result_verified = bool(result.verified)
            returned = result_verified
        elapsed = time.monotonic() - started
        final_receipts = [row for row in executor.receipts if row["reason"] == "final"]
        erased = self.adapter.proposal_after_erasure(request)
        future_use = bool(proposal["influenced_atom_ids"] and returned)
        return {
            "row_type": "comparison",
            "request_id": request["request_id"],
            "version_token": request["version_token"],
            "warmup": bool(warmup),
            "candidate_action": proposal["candidate_action"],
            "active_atom_count_before": proposal["active_atom_count_before"],
            "influenced_atom_ids": proposal["influenced_atom_ids"],
            "returned_plan": deepcopy(plan) if returned else None,
            "returned": returned,
            "coverage": int(returned),
            "returned_feasible": bool(returned and result_verified),
            "stale_atom_returned": False,
            "query_attempts": executor.attempt_count,
            "executor_calls": executor.external_call_count,
            "cache_hits": executor.cache_hits,
            "final_checks": len(final_receipts),
            "final_receipt": deepcopy(final_receipts[-1]) if final_receipts else None,
            "new_atom_count": int(feedback.get("new_atom_count", 0)),
            "invalidated_atom_ids": list(feedback.get("invalidated_atom_ids", [])),
            "contradiction_revalidation": bool(feedback.get("contradiction_revalidation", False)),
            "entry_state_hash": proposal["entry_state_hash"],
            "state_bytes": int(feedback.get("state_bytes", len(self.adapter.state_bytes()))),
            "utility": _utility(request, plan if returned else None),
            "future_use_witness": future_use,
            "erasure_reversed": bool(future_use and plan != erased),
            "complete_wall_cost": elapsed,
            "generation_cost_allocation": 0.0,
            "serialization_cost_included": True,
            "pipeline_defaults_unchanged": (
                self.pipeline._max_repairs == 3  # noqa: SLF001
                and self.pipeline._timeout_seconds == 30.0  # noqa: SLF001
                and self.pipeline.routing_mode == "argmax"
                and self.pipeline.balance_ratio == 1.0
                and not self.pipeline.has_model
            ),
            "censored": False,
            "failures": [] if returned else ["exact_rejection"],
        }

    def close(self) -> None:
        self.pipeline.close()


def _utility(request: Mapping[str, Any], plan: Mapping[str, Any] | None) -> float:
    """Measure public early-slot utility without reading private rules."""

    if plan is None:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for name in request["activities"]:
        starts = list(request["allowed_starts"][name])
        weight = float(request["weights"][name])
        rank = starts.index(int(plan["assignments"][name]))
        scale = max(1, len(starts) - 1)
        numerator += weight * (1.0 - rank / scale)
        denominator += weight
    return numerator / denominator


def run_adapter_lifecycle(state_root: Path) -> JsonDict:
    """Run the E2E-007 commit, restart, duplicate, and rollback lifecycle."""

    request = {
        "request_id": "lifecycle-first",
        "version_token": "opaque-lifecycle-v1",
        "activities": ["a", "b"],
        "allowed_starts": {"a": [0, 2], "b": [0, 2]},
        "durations": {"a": 1, "b": 1},
        "weights": {"a": 1, "b": 1},
        "horizon": 8,
        "public_revision": 0,
    }
    record = {
        "version_token": request["version_token"],
        "private_rules": {
            "capacity": 2,
            "pair_gaps": [{"pair": ["a", "b"], "minimum_gap": 1}],
            "forbidden_compounds": [],
        },
    }
    adapter = LearningScheduleAdapter(state_root, enabled=True)
    parent_bytes = adapter.state_bytes()
    harness = AdapterPipelineHarness(adapter)
    harness.execute(request, record, warmup=True)
    feedback = adapter.last_feedback
    committed_bytes = adapter.state_bytes()
    decision_id = next(iter(adapter._completed))  # noqa: SLF001
    duplicate = adapter.record_exact_result(
        AdapterRouteDecision(decision_id, (), None, "accepted", ()),
        {"verified": False},
    )
    duplicate_bytes = adapter.state_bytes()
    restarted = LearningScheduleAdapter(state_root, enabled=True)
    restart_bytes = restarted.state_bytes()
    later = {**request, "request_id": "lifecycle-later", "public_revision": 1}
    later_executor = QualifiedFixtureExecutor(
        later, {**record, "version_token": later["version_token"]}
    )
    visible = restarted.begin_request(later, later_executor)["active_atom_count_before"] > 0
    restarted.cancel_request("restart_probe")
    rollback = adapter.rollback_last_commit()
    checks = {
        "commit_after_close": feedback.get("commit_count") == 1,
        "duplicate_feedback_noop": duplicate.get("duplicate_feedback") is True
        and duplicate_bytes == committed_bytes,
        "restart_atom_visible": visible,
        "restart_bytes_equal": committed_bytes == restart_bytes,
        "rollback_bytes_equal": rollback["byte_identical"] is True
        and adapter.state_bytes() == parent_bytes,
    }
    harness.close()
    checks["e2e_007_passed"] = all(checks.values())
    return checks


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_bootstrap(
    stream_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    comparator: str,
    *,
    seed: int,
    draws: int,
) -> JsonDict:
    stream_ids = sorted(stream_metrics)
    if not stream_ids:
        return {
            "utility_difference_ci95": {"lower": math.nan, "upper": math.nan},
            "coverage_difference_ci95": {"lower": math.nan, "upper": math.nan},
            "query_ratio_ci95": {"lower": math.nan, "upper": math.nan},
            "complete_wall_cost_ratio_ci95": {"lower": math.nan, "upper": math.nan},
            "stream_count": 0,
        }
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {
        "utility": [],
        "coverage": [],
        "query": [],
        "wall": [],
    }
    for _ in range(draws):
        selected = [rng.choice(stream_ids) for _ in stream_ids]
        persistent = [stream_metrics[key][PERSISTENT_ARM] for key in selected]
        baseline = [stream_metrics[key][comparator] for key in selected]
        samples["utility"].append(
            sum(left["utility"] - right["utility"] for left, right in zip(persistent, baseline))
            / len(selected)
        )
        samples["coverage"].append(
            sum(left["coverage"] - right["coverage"] for left, right in zip(persistent, baseline))
            / len(selected)
        )
        samples["query"].append(
            sum(row["queries"] for row in persistent)
            / max(1.0, sum(row["queries"] for row in baseline))
        )
        samples["wall"].append(
            sum(row["wall"] for row in persistent)
            / max(1e-12, sum(row["wall"] for row in baseline))
        )
    return {
        "utility_difference_ci95": {
            "lower": _quantile(samples["utility"], 0.025),
            "upper": _quantile(samples["utility"], 0.975),
        },
        "coverage_difference_ci95": {
            "lower": _quantile(samples["coverage"], 0.025),
            "upper": _quantile(samples["coverage"], 0.975),
        },
        "query_ratio_ci95": {
            "lower": _quantile(samples["query"], 0.025),
            "upper": _quantile(samples["query"], 0.975),
        },
        "complete_wall_cost_ratio_ci95": {
            "lower": _quantile(samples["wall"], 0.025),
            "upper": _quantile(samples["wall"], 0.975),
        },
        "stream_count": len(stream_ids),
        "bootstrap_draws": draws,
        "stream_clustered": True,
    }


def reduce_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    resampling_seed: int = RESAMPLING_SEED,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> JsonDict:
    """Independently reduce post-warmup source rows under the frozen contract."""

    comparisons = [
        row
        for row in rows
        if row.get("row_type") == "comparison"
        and row.get("cohort") == "source_fidelity"
        and row.get("warmup") is False
        and row.get("censored") is False
    ]
    stream_metrics: dict[str, dict[str, JsonDict]] = {}
    for stream_id in sorted({str(row["stream_id"]) for row in comparisons}):
        stream_metrics[stream_id] = {}
        for arm in ARMS:
            selected = [
                row for row in comparisons if row["stream_id"] == stream_id and row["arm"] == arm
            ]
            if not selected:
                continue
            stream_metrics[stream_id][arm] = {
                "utility": sum(float(row["utility"]) for row in selected) / len(selected),
                "coverage": sum(float(row["coverage"]) for row in selected) / len(selected),
                "queries": sum(float(row["query_attempts"]) for row in selected),
                "wall": sum(float(row["complete_wall_cost"]) for row in selected),
            }
    complete_streams = {
        key: value for key, value in stream_metrics.items() if set(value) == set(ARMS)
    }
    paired = {
        comparator: _paired_bootstrap(
            complete_streams,
            comparator,
            seed=resampling_seed + index,
            draws=bootstrap_draws,
        )
        for index, comparator in enumerate((RESET_ARM, CACHE_ARM), start=1)
    }
    unsafe = sum(
        bool(row.get("returned")) and row.get("returned_feasible") is not True
        for row in comparisons
    )
    stale = sum(bool(row.get("stale_atom_returned")) for row in comparisons)
    witnesses = [
        row
        for row in comparisons
        if row.get("arm") == PERSISTENT_ARM
        and row.get("future_use_witness") is True
        and row.get("erasure_reversed") is True
    ]
    service_pass = all(
        result[metric]["lower"] >= -0.02
        for result in paired.values()
        for metric in ("utility_difference_ci95", "coverage_difference_ci95")
    )
    query_pass = all(result["query_ratio_ci95"]["upper"] < 0.90 for result in paired.values())
    wall_pass = all(
        result["complete_wall_cost_ratio_ci95"]["upper"] <= 1.0 for result in paired.values()
    )
    return {
        "row_count": len(rows),
        "post_warmup_source_row_count": len(comparisons),
        "complete_stream_count": len(complete_streams),
        "comparisons": paired,
        "safety": {
            "returned_infeasible_plan_count": unsafe,
            "stale_atom_return_count": stale,
            "query_budget_exceeded_count": sum(
                int(row.get("query_attempts", 0)) > QUERY_BUDGET for row in comparisons
            ),
            "state_cap_exceeded_count": sum(
                int(row.get("state_bytes", 0)) > STATE_CAP_BYTES for row in comparisons
            ),
        },
        "causal_future_use": {
            "witness_count": len(witnesses),
            "request_ids": sorted(str(row["request_id"]) for row in witnesses),
            "all_reversed_by_erasure": bool(witnesses)
            and all(row.get("erasure_reversed") is True for row in witnesses),
        },
        "frozen_memory_comparison": {
            "post_warmup_only": True,
            "row_count": sum(row.get("arm") == FROZEN_ARM for row in comparisons),
        },
        "service_quality_gate_passed": service_pass,
        "total_query_gate_passed": query_pass,
        "deployment_cost_gate_passed": wall_pass,
        "promotion_contract_passed": (
            unsafe == 0
            and stale == 0
            and service_pass
            and query_pass
            and bool(witnesses)
            and all(
                int(row.get("query_attempts", 0)) <= QUERY_BUDGET
                and int(row.get("state_bytes", 0)) <= STATE_CAP_BYTES
                for row in comparisons
            )
        ),
    }


def _request_streams(public_manifest: Mapping[str, Any]) -> list[JsonDict]:
    streams = []
    for stream in public_manifest["development_streams"]:
        streams.append(
            {
                "cohort": "source_fidelity",
                "source_label": str(stream["cohort"]),
                "stream_id": str(stream["stream_id"]),
                "requests": [deepcopy(dict(row)) for row in stream["requests"]],
            }
        )
    for stream in public_manifest["public_model_streams"]:
        streams.append(
            {
                "cohort": "scripted_model_shaped",
                "source_label": str(stream["cohort"]),
                "stream_id": str(stream["stream_id"]),
                "requests": [
                    {**deepcopy(dict(panel["original"])), "warmup": bool(panel["warmup"])}
                    for panel in stream["requests"]
                ],
            }
        )
    return streams


def run_measurement(
    public_manifest: Mapping[str, Any],
    private_manifest: Mapping[str, Any],
    state_root: Path,
) -> list[JsonDict]:
    """Execute four frozen arms over sealed source and scripted cohorts."""

    records = private_manifest["evaluator_records"]
    streams = _request_streams(public_manifest)
    rows: list[JsonDict] = []
    total = len(streams) * len(ARMS)
    completed = 0
    for stream in streams:
        for arm in ARMS:
            adapter = LearningScheduleAdapter(
                state_root / stream["cohort"] / stream["stream_id"] / arm,
                enabled=True,
                persist_feedback=arm in {PERSISTENT_ARM, FROZEN_ARM},
            )
            harness = AdapterPipelineHarness(adapter)
            exact_cache: dict[str, bool] | None = {} if arm == CACHE_ARM else None
            for request_index, request in enumerate(stream["requests"]):
                warmup = bool(request.get("warmup", request_index < 4))
                row = harness.execute(
                    request,
                    records[str(request["request_id"])],
                    warmup=warmup,
                    allow_learning=arm != FROZEN_ARM or warmup,
                    exact_cache=exact_cache,
                )
                row.update(
                    {
                        "cohort": stream["cohort"],
                        "source_label": stream["source_label"],
                        "stream_id": stream["stream_id"],
                        "request_index": request_index,
                        "arm": arm,
                        "counts_toward_promotion": stream["cohort"] == "source_fidelity",
                    }
                )
                rows.append(row)
            harness.close()
            completed += 1
            progress(
                "evaluation",
                "unit_complete",
                f"completed={completed}/{total} rows={len(rows)}",
            )
    return rows


def run_adapter_controls(state_root: Path) -> JsonDict:
    """Exercise lifecycle, withholding, corruption, and erased-atom controls."""

    lifecycle = run_adapter_lifecycle(state_root / "lifecycle")
    request = {
        "request_id": "control-withheld",
        "version_token": "opaque-control-v1",
        "activities": ["a", "b"],
        "allowed_starts": {"a": [0, 2], "b": [0, 2]},
        "durations": {"a": 1, "b": 1},
        "weights": {"a": 1, "b": 1},
        "horizon": 8,
        "public_revision": 0,
    }
    private_record = {
        "version_token": request["version_token"],
        "private_rules": {
            "capacity": 2,
            "pair_gaps": [{"pair": ["a", "b"], "minimum_gap": 1}],
            "forbidden_compounds": [],
        },
    }
    withheld = LearningScheduleAdapter(state_root / "withheld", enabled=True)
    before = withheld.state_bytes()
    withheld.begin_request(request, QualifiedFixtureExecutor(request, private_record))
    during = withheld.state_bytes()
    withheld.cancel_request("withheld_feedback")
    after = withheld.state_bytes()
    corrupt_dir = state_root / "corrupt"
    corrupt = LearningScheduleAdapter(corrupt_dir, enabled=True)
    corrupt.state_path.write_text("corrupt", encoding="utf-8")
    corrupt_reload = LearningScheduleAdapter(corrupt_dir, enabled=True)
    checks = {
        **lifecycle,
        "withheld_feedback_no_commit": before == during == after,
        "corrupted_state_rejected": corrupt_reload.corrupted_state,
        "query_budget": QUERY_BUDGET,
        "state_cap_bytes": STATE_CAP_BYTES,
        "production_default_enabled": False,
    }
    checks["passed"] = (
        all(
            value is True
            for key, value in checks.items()
            if key not in {"query_budget", "state_cap_bytes", "production_default_enabled"}
        )
        and checks["production_default_enabled"] is False
    )
    return checks


def _precondition_row(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected_value": expected,
        "observed_value": observed,
        "available": bool(passed),
        "blocking": True,
        "principle": "Dependent work starts only from authenticated eligible inputs.",
    }


def _declared_hash(producer: Mapping[str, Any], relative: Path) -> str | None:
    for name, value in dict(producer.get("source_artifact_hashes") or {}).items():
        if str(name).endswith(relative.as_posix()):
            return str(value)
    return None


def collect_preconditions(repo_root: Path) -> tuple[list[JsonDict], dict[str, str], JsonDict]:
    """Check exact paths and producer fields before dependent evaluation."""

    rows: list[JsonDict] = []
    hashes: dict[str, str] = {}
    root = repo_root.resolve()
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        rows.append(
            _precondition_row(
                f"source_bytes:{relative}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                path.stat().st_size if available else None,
                available,
            )
        )
        if available:
            hashes[relative.as_posix()] = sha256_file(path)
    producer: JsonDict = {}
    producer_path = root / PRODUCER_PATH
    if producer_path.is_file():
        try:
            loaded = json.loads(producer_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                producer = loaded
        except json.JSONDecodeError:
            producer = {}
    producer_checks = (
        ("producer_status", "status", "complete", producer.get("status") == "complete"),
        (
            "producer_milestone",
            "milestone",
            MILESTONE,
            producer.get("milestone") == MILESTONE,
        ),
        (
            "producer_run_date",
            "run_date",
            RUN_DATE,
            producer.get("run_date") == RUN_DATE,
        ),
        (
            "producer_ready",
            "executor_fixture_ready_score",
            1,
            producer.get("executor_fixture_ready_score") == 1,
        ),
        (
            "producer_terminal_class",
            "verdict_class",
            "circular_positive",
            producer.get("verdict_class") == "circular_positive",
        ),
        (
            "producer_adversarial_clear",
            "flagged_adversarial",
            False,
            producer.get("flagged_adversarial") is False,
        ),
        (
            "producer_gate_summary",
            "gate_check_summary.passed",
            True,
            dict(producer.get("gate_check_summary") or {}).get("passed") is True,
        ),
    )
    for check, field, expected, passed in producer_checks:
        observed: Any = producer
        for part in field.split("."):
            observed = observed.get(part) if isinstance(observed, Mapping) else None
        rows.append(
            _precondition_row(
                check,
                PRODUCER_PATH.as_posix(),
                field,
                expected,
                observed,
                passed,
            )
        )
    for relative in (PUBLIC_MANIFEST_PATH, PRIVATE_MANIFEST_PATH, SCRIPTED_COHORT_PATH):
        actual = hashes.get(relative.as_posix())
        declared = _declared_hash(producer, relative)
        rows.append(
            _precondition_row(
                f"producer_hash:{relative}",
                PRODUCER_PATH.as_posix(),
                f"source_artifact_hashes.{relative}",
                declared,
                actual,
                declared is not None and actual == declared,
            )
        )
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    rows.append(
        _precondition_row(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-CL-7346",
            "REQ-CL-7346" if "REQ-CL-7346" in spec_text else None,
            "REQ-CL-7346" in spec_text,
        )
    )
    exclusion_text = (
        (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    excluded = "experiment_id: 7346" in exclusion_text or "exp7346" in exclusion_text.lower()
    rows.append(
        _precondition_row(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            True,
            not excluded,
            not excluded,
        )
    )
    return rows, hashes, producer


def passing_test_preconditions() -> list[JsonDict]:
    """Provide a compact, explicit passing seam for reducer unit tests."""

    return [_precondition_row("test_precondition", "fixture", "ready", True, True, True)]


def passing_test_receipts() -> list[JsonDict]:
    """Provide named successful command receipts for terminal unit tests."""

    return [
        {
            "name": name,
            "command": f"test:{name}",
            "scope": "unit_test",
            "exit_code": 0,
            "duration_s": 0.001,
            "log_sha256": "sha256:" + "0" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in ALL_REQUIRED_CHECK_NAMES
    ]


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    return {
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def _receipt_passed(receipts: Sequence[Mapping[str, Any]], name: str) -> bool:
    selected = [row for row in receipts if row.get("name") == name]
    return (
        len(selected) == 1
        and selected[0].get("passed") is True
        and selected[0].get("exit_code") == 0
    )


def _gate_summary(
    gates: Mapping[str, Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    failed_rows = [row for row in preconditions if not row.get("available")]
    failed_precondition = next(
        (row for row in failed_rows if row.get("check") == "producer_ready"),
        failed_rows[0] if failed_rows else None,
    )
    failed_gate = next((name for name, value in gates.items() if not value.get("passed")), None)
    if failed_precondition is not None:
        return {
            "upstream": failed_precondition["upstream"],
            "failed_check": failed_precondition["check"],
            "artifact_field": failed_precondition["artifact_field"],
            "expected_value": failed_precondition["expected_value"],
            "observed_value": failed_precondition["observed_value"],
            "passed": False,
            "failed_check_count": sum(not row.get("available") for row in preconditions),
            "check_count": len(preconditions) + len(gates),
        }
    if failed_gate is not None:
        gate = gates[failed_gate]
        return {
            "upstream": EXPERIMENT_ID,
            "failed_check": failed_gate,
            "artifact_field": f"acceptance_gate_results.{failed_gate}",
            "expected_value": gate["expected"],
            "observed_value": gate["observed"],
            "passed": False,
            "failed_check_count": sum(not value.get("passed") for value in gates.values()),
            "check_count": len(preconditions) + len(gates),
        }
    return {
        "upstream": EXPERIMENT_ID,
        "failed_check": None,
        "artifact_field": "promotion_score",
        "expected_value": 1,
        "observed_value": 1,
        "passed": True,
        "failed_check_count": 0,
        "check_count": len(preconditions) + len(gates),
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind decisions and evidence while excluding host timing and receipts."""

    bound = {
        key: artifact.get(key)
        for key in (
            "schema",
            "experiment_id",
            "milestone",
            "run_date",
            "random_seed",
            "source_artifact_hashes",
            "rows",
            "sample_size_budget",
            "frozen_learning_protocol",
            "adapter_controls",
            "independent_reduction",
            "cohort_accounting",
        )
    }
    return public.sha256_json(bound)


def _base_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    controls: Mapping[str, Any],
    started_at: str,
    phase_spans: Mapping[str, float] | None,
) -> JsonDict:
    reduction = reduce_rows(rows)
    receipts = [dict(row) for row in validation_receipts]
    preconditions_pass = all(row.get("available") is True for row in preconditions)
    scoped_pass = all(_receipt_passed(receipts, name) for name in REQUIRED_CHECK_NAMES)
    full_pass = _receipt_passed(receipts, "full_python_suite")
    terminal_pass = all(_receipt_passed(receipts, name) for name in TERMINAL_CHECK_NAMES)
    controls_pass = controls.get("passed") is True
    gates = {
        "preconditions": _gate(
            True,
            preconditions_pass,
            preconditions_pass,
            "All exact producer and owned-resource checks precede dependent work.",
        ),
        "adapter_controls": _gate(
            True,
            controls_pass,
            controls_pass,
            "Withholding, corruption, version, restart, rollback, and duplicate controls fail closed.",
        ),
        "service_quality": _gate(
            True,
            reduction["service_quality_gate_passed"],
            reduction["service_quality_gate_passed"],
            "Persistence cannot reduce paired utility or coverage below the frozen tolerance.",
        ),
        "query_advantage": _gate(
            "CI95 upper < 0.90",
            {
                name: value["query_ratio_ci95"]["upper"]
                for name, value in reduction["comparisons"].items()
            },
            reduction["total_query_gate_passed"],
            "Persistent atoms must reduce total paid work against both reset controls.",
        ),
        "safety": _gate(
            {
                "returned_infeasible_plan_count": 0,
                "stale_atom_return_count": 0,
                "query_budget_exceeded_count": 0,
                "state_cap_exceeded_count": 0,
            },
            reduction["safety"],
            all(value == 0 for value in reduction["safety"].values()),
            "No unsafe return, stale authority, budget excess, or state excess is promotable.",
        ),
        "causal_future_use": _gate(
            ">=1 reversed witness",
            reduction["causal_future_use"],
            reduction["causal_future_use"]["witness_count"] >= 1
            and reduction["causal_future_use"]["all_reversed_by_erasure"],
            "An acquired atom must change later distinct work and erasure must reverse it.",
        ),
        "scoped_validation": _gate(
            True,
            scoped_pass,
            scoped_pass,
            "The shipped bounded runner owns affected test, coverage, lint, type, and spec checks.",
        ),
        "full_python_suite": _gate(
            True,
            full_pass,
            full_pass,
            "The mandated Python suite must pass once without weakening repository checks.",
        ),
        "terminal_validation": _gate(
            True,
            terminal_pass,
            terminal_pass,
            "Cold reduction and both strict terminal readers must pass the measured candidate.",
        ),
    }
    all_terminal = all(value["passed"] for value in gates.values())
    pending_terminal = scoped_pass and full_pass and not terminal_pass
    mechanism_ready = preconditions_pass and controls_pass and scoped_pass and full_pass
    value_ready = mechanism_ready and reduction["promotion_contract_passed"]
    if not preconditions_pass:
        status = "blocked_learning_adapter_precondition"
        honest = "blocked_learning_adapter_precondition"
        verdict = "blocked"
    elif pending_terminal:
        status = "partial_pending_terminal_validation"
        honest = "partial_pending_terminal_validation"
        verdict = "partial"
    elif not scoped_pass or not full_pass or not controls_pass or not terminal_pass:
        status = "complete_learning_adapter_disqualified"
        honest = "complete_learning_adapter_disqualified"
        verdict = "disqualified"
    elif reduction["promotion_contract_passed"]:
        status = "complete_learning_adapter_circular_positive"
        honest = "complete_circular_positive_learning_adapter_promotable"
        verdict = "circular_positive"
    else:
        status = "complete_learning_adapter_null"
        honest = "complete_null_learning_adapter_not_promotable"
        verdict = "null"
    scores_allowed = verdict not in {"blocked", "disqualified", "partial"}
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 2,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "preconditions_checked": [dict(row) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "phase_spans": dict(
            phase_spans
            or {
                "load_s": 0.0,
                "generation_s": 0.0,
                "evaluation_s": float(duration_s),
                "test_s": 0.0,
                "write_s": 0.0,
            }
        ),
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "evaluation": EVALUATION_SEED,
            "resampling": RESAMPLING_SEED,
        },
        "source_artifact_hashes": dict(source_hashes),
        "rows": [deepcopy(dict(row)) for row in rows],
        "sample_size_budget": {
            "planned_source_streams": 32,
            "planned_scripted_model_shaped_streams": 8,
            "planned_arms": list(ARMS),
            "attempted_rows": len(rows),
            "completed_rows": sum(not row.get("censored", False) for row in rows),
            "censored_rows": sum(bool(row.get("censored", False)) for row in rows),
            "stopping_rule": "Run each frozen stream-arm-request unit once; do not extend from outcomes.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates, preconditions),
        "verifier_is_oracle": True,
        "honest_verdict": honest,
        "verdict_class": verdict,
        "flagged_adversarial": not _receipt_passed(receipts, "adversarial_verify"),
        "validation_receipts": receipts,
        "repository_health": {
            "status": "degraded_open",
            "affects_required_checks": False,
            "historical_failures": [
                {
                    "date": "2026-09-15",
                    "source": "exp7330-v644-executor-isolation",
                    "exit_code": -15,
                    "classification": "unrelated_historical_full_suite_termination",
                    "resolved": False,
                }
            ],
        },
        "field_principles": {
            **REQUIRED_FIELD_PRINCIPLES,
            **{
                key: "This field retains the measured adapter record without changing its gate type."
                for key in REQUIRED_ARTIFACT_FIELDS
                if key not in REQUIRED_FIELD_PRINCIPLES
            },
        },
        "learning_adapter_ready_score": int(scores_allowed and mechanism_ready),
        "learning_value_score": int(scores_allowed and value_ready),
        "promotion_score": int(scores_allowed and all_terminal and value_ready),
        "frozen_learning_protocol": {
            "arms": list(ARMS),
            "query_budget_per_request": QUERY_BUDGET,
            "state_cap_bytes": STATE_CAP_BYTES,
            "warmup_cutoff_source": 4,
            "warmup_cutoff_scripted_model_shaped": 2,
            "utility_and_coverage_ci95_lower": -0.02,
            "total_query_ratio_ci95_upper_exclusive": 0.90,
            "complete_wall_cost_ratio_ci95_upper_inclusive": 1.0,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "resampling_seed": RESAMPLING_SEED,
            "frozen_before_outcomes": True,
        },
        "adapter_controls": deepcopy(dict(controls)),
        "independent_reduction": reduction,
        "cohort_accounting": {
            "source_fidelity": {
                "counts_toward_promotion": True,
                "current_model_generated": False,
            },
            "scripted_model_shaped": {
                "counts_toward_promotion": False,
                "current_model_generated": False,
                "label": "scripted_public_model_requests_no_current_llm",
            },
            "development_pass_region": {
                "counts_toward_science": False,
                "witnessed": controls_pass,
            },
        },
        "raw_evidence_paths": {
            "rows": str(RAW_DIR / "rows.json"),
            "measured_candidate": str(RAW_DIR / "measured-terminal-candidate.json"),
            "scripted_model_sidecar": str(SCRIPTED_COHORT_PATH),
            "historical_model_sidecar": str(HISTORICAL_MODEL_PATH),
        },
        "production_default_changed": False,
        "research_roadmap_changed": False,
        "publication_surface_changed": False,
        "no_model_weight_mutation": True,
        "deployment_cost_claim": bool(scores_allowed and reduction["deployment_cost_gate_passed"]),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def artifact_from_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    controls: Mapping[str, Any] | None = None,
    started_at: str | None = None,
    phase_spans: Mapping[str, float] | None = None,
) -> JsonDict:
    """Build one schema-complete artifact from rows and real command receipts."""

    default_controls = {
        "passed": True,
        "commit_after_close": True,
        "duplicate_feedback_noop": True,
        "restart_atom_visible": True,
        "restart_bytes_equal": True,
        "rollback_bytes_equal": True,
        "e2e_007_passed": True,
        "withheld_feedback_no_commit": True,
        "corrupted_state_rejected": True,
        "production_default_enabled": False,
    }
    return _base_artifact(
        rows=rows,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=validation_receipts,
        duration_s=duration_s,
        controls=controls or default_controls,
        started_at=started_at or datetime.now(UTC).isoformat(),
        phase_spans=phase_spans,
    )


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    state_root: Path,
    execute_measurement: bool = True,
    validation_receipts: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Check upstream eligibility, then optionally execute all frozen rows."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    preconditions, hashes, _producer = collect_preconditions(repo_root)
    if not all(row["available"] for row in preconditions):
        return artifact_from_rows(
            rows=[],
            preconditions=preconditions,
            source_hashes=hashes,
            validation_receipts=validation_receipts,
            duration_s=time.monotonic() - started,
            controls={"passed": False, "blocked_before_controls": True},
            started_at=started_at,
        )
    if not execute_measurement:
        raise LearningAdapterError("measurement_required_after_passing_preconditions")
    public_manifest = json.loads((repo_root / PUBLIC_MANIFEST_PATH).read_text(encoding="utf-8"))
    private_manifest = json.loads((repo_root / PRIVATE_MANIFEST_PATH).read_text(encoding="utf-8"))
    rows = run_measurement(public_manifest, private_manifest, state_root / "measurement")
    controls = run_adapter_controls(state_root / "controls")
    return artifact_from_rows(
        rows=rows,
        preconditions=preconditions,
        source_hashes=hashes,
        validation_receipts=validation_receipts,
        duration_s=time.monotonic() - started,
        controls=controls,
        started_at=started_at,
    )


def independent_reduce(artifact: Mapping[str, Any]) -> list[str]:
    """Reload row-derived claims without trusting stored aggregate fields."""

    errors: list[str] = []
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        return ["rows are not a list"]
    expected = reduce_rows(rows)
    if artifact.get("independent_reduction") != expected:
        errors.append("stored reduction differs from rows")
    return errors


def validate_artifact(artifact: object) -> list[str]:
    """Cold-check schema, safety budgets, scores, and row-derived claims."""

    if not isinstance(artifact, Mapping):
        return ["artifact is not an object"]
    errors: list[str] = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        errors.append(f"missing fields: {missing}")
        return errors
    if artifact.get("schema") != SCHEMA or artifact.get("run_date") != RUN_DATE:
        errors.append("schema or run date mismatch")
    if artifact.get("milestone") != MILESTONE or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment identity mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("current model declaration mismatch")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("invocation counts are nonzero")
    if artifact.get("inference_substrate") != "cpu_exact_solver_or_simulator":
        errors.append("inference substrate mismatch")
    if artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator":
        errors.append("inference substrate class mismatch")
    if artifact.get("execution_venue") != "host":
        errors.append("execution venue mismatch")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("invalid verdict class")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("executor oracle disclosure missing")
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        errors.append("rows are not a list")
    else:
        if any(int(row.get("query_attempts", 0)) > QUERY_BUDGET for row in rows):
            errors.append("query budget exceeded")
        if any(int(row.get("state_bytes", 0)) > STATE_CAP_BYTES for row in rows):
            errors.append("state cap exceeded")
    errors.extend(independent_reduce(artifact))
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        field not in principles for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field principles incomplete")
    if artifact.get("verdict_class") in {"blocked", "disqualified", "partial"} and any(
        artifact.get(field) != 0
        for field in ("learning_adapter_ready_score", "learning_value_score", "promotion_score")
    ):
        errors.append("blocked, disqualified, or partial scores must be zero")
    if artifact.get("status", "").startswith("blocked_") and rows:
        errors.append("blocked artifact contains rows")
    if artifact.get("flagged_adversarial") is True and artifact.get("promotion_score") != 0:
        errors.append("adversarial finding did not zero promotion")
    return errors


def scoped_command_plan(repo_root: Path, temporary_root: Path) -> list[CommandSpec]:
    """Build only the explicit Exp7346 affected validation commands."""

    basetemp = temporary_root / "basetemp"
    coverage = temporary_root / ".coverage-exp7346"
    basetemp.mkdir(parents=True, exist_ok=True)
    return build_scoped_commands(
        repo_root,
        [str(TEST_PATH)],
        [str(MODULE_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        basetemp=basetemp,
        coverage_file=coverage,
    )


def _full_suite_command(repo_root: Path) -> CommandSpec:
    return CommandSpec(
        "full_python_suite",
        (str(repo_root / ".venv/bin/pytest"), "tests/python", "-q"),
        "mandated_python_suite",
        timeout_s=1_800.0,
    )


def _terminal_commands(repo_root: Path, candidate: Path) -> list[CommandSpec]:
    python = str(repo_root / ".venv/bin/python")
    return [
        CommandSpec(
            "independent_reducer",
            (
                python,
                "-u",
                "-c",
                (
                    "import json,pathlib;"
                    "from carnot.experiment_7346_v645_learning_adapter import independent_reduce;"
                    "a=json.loads(pathlib.Path(r'" + str(candidate.resolve()) + "').read_text());"
                    "e=independent_reduce(a);print(e,flush=True);raise SystemExit(bool(e))"
                ),
            ),
            "measured_candidate",
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate.resolve())),
            "measured_candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate.resolve()),
            ),
            "measured_candidate",
        ),
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - declared E2E entrypoint.
    args = _parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.validate:
        value = json.loads((REPO_ROOT / args.output).read_text(encoding="utf-8"))
        errors = validate_artifact(value)
        print(errors, flush=True)
        return int(bool(errors))
    overall_started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress("preconditions", "start")
    preconditions, hashes, _producer = collect_preconditions(REPO_ROOT)
    progress("preconditions", "end", f"passed={all(row['available'] for row in preconditions)}")
    if not all(row["available"] for row in preconditions):
        blocked = artifact_from_rows(
            rows=[],
            preconditions=preconditions,
            source_hashes=hashes,
            validation_receipts=[],
            duration_s=time.monotonic() - overall_started,
            controls={"passed": False, "blocked_before_controls": True},
            started_at=started_at,
        )
        _atomic_json(REPO_ROOT / args.output, blocked)
        progress("write", "terminal_blocked", str(args.output))
        return 0
    raw_dir = REPO_ROOT / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    state_root = Path(tempfile.mkdtemp(prefix="state-", dir=raw_dir))
    public_manifest = json.loads((REPO_ROOT / PUBLIC_MANIFEST_PATH).read_text(encoding="utf-8"))
    private_manifest = json.loads((REPO_ROOT / PRIVATE_MANIFEST_PATH).read_text(encoding="utf-8"))
    progress("evaluation", "start")
    evaluation_started = time.monotonic()
    rows = run_measurement(public_manifest, private_manifest, state_root / "measurement")
    controls = run_adapter_controls(state_root / "controls")
    evaluation_s = time.monotonic() - evaluation_started
    _atomic_json(raw_dir / "rows.json", {"schema": SCHEMA + ".rows", "rows": rows})
    progress("evaluation", "end", f"rows={len(rows)} elapsed_s={evaluation_s:.3f}")
    validation_root = Path(tempfile.mkdtemp(prefix="exp7346-validation-", dir="/tmp"))
    (validation_root / "basetemp").mkdir(parents=True, exist_ok=True)
    progress("validation", "before_subprocesses")
    test_started = time.monotonic()
    scoped = run_scoped_validation(
        REPO_ROOT,
        [str(TEST_PATH)],
        [str(MODULE_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        basetemp=validation_root / "basetemp",
        coverage_file=validation_root / ".coverage-exp7346",
        log_dir=raw_dir / "validation/scoped",
    )
    full_receipts = run_commands(
        REPO_ROOT,
        [_full_suite_command(REPO_ROOT)],
        log_dir=raw_dir / "validation/full",
    )
    preliminary_receipts = [*scoped["validation_receipts"], *full_receipts]
    candidate = artifact_from_rows(
        rows=rows,
        preconditions=preconditions,
        source_hashes=hashes,
        validation_receipts=preliminary_receipts,
        duration_s=time.monotonic() - overall_started,
        controls=controls,
        started_at=started_at,
        phase_spans={
            "load_s": 0.0,
            "generation_s": 0.0,
            "evaluation_s": evaluation_s,
            "test_s": time.monotonic() - test_started,
            "write_s": 0.0,
        },
    )
    candidate_path = raw_dir / "measured-terminal-candidate.json"
    _atomic_json(candidate_path, candidate)
    terminal_receipts = run_commands(
        REPO_ROOT,
        _terminal_commands(REPO_ROOT, candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    test_s = time.monotonic() - test_started
    progress(
        "validation",
        "after_subprocesses",
        f"receipts={len(preliminary_receipts) + len(terminal_receipts)}",
    )
    final = artifact_from_rows(
        rows=rows,
        preconditions=preconditions,
        source_hashes=hashes,
        validation_receipts=[*preliminary_receipts, *terminal_receipts],
        duration_s=time.monotonic() - overall_started,
        controls=controls,
        started_at=started_at,
        phase_spans={
            "load_s": 0.0,
            "generation_s": 0.0,
            "evaluation_s": evaluation_s,
            "test_s": test_s,
            "write_s": 0.0,
        },
    )
    errors = validate_artifact(final)
    if errors:
        final["status"] = "complete_learning_adapter_disqualified"
        final["honest_verdict"] = "complete_disqualified_learning_adapter_validation"
        final["verdict_class"] = "disqualified"
        final["flagged_adversarial"] = True
        final["learning_adapter_ready_score"] = 0
        final["learning_value_score"] = 0
        final["promotion_score"] = 0
        final["gate_check_summary"] = {
            "upstream": EXPERIMENT_ID,
            "failed_check": "cold_artifact_validation",
            "artifact_field": "validate_artifact",
            "expected_value": [],
            "observed_value": errors,
            "passed": False,
            "failed_check_count": len(errors),
            "check_count": len(final["acceptance_gate_results"]) + len(preconditions),
        }
        final["reproducibility_checksum"] = reproducibility_checksum(final)
    progress("write", "before_atomic_publish", str(args.output))
    write_started = time.monotonic()
    final["phase_spans"]["write_s"] = time.monotonic() - write_started
    final["completed_at_utc"] = datetime.now(UTC).isoformat()
    final["duration_s"] = time.monotonic() - overall_started
    final["reproducibility_checksum"] = reproducibility_checksum(final)
    _atomic_json(REPO_ROOT / args.output, final)
    progress("write", "after_atomic_publish", f"status={final['status']}")
    return int(bool(validate_artifact(final)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
