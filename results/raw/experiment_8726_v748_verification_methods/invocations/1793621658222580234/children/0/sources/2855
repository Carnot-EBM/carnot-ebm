"""Measure held-out value from bounded versioned structural constraint addition.

The learner receives public scheduling requests, an authenticated executor
version, and Boolean query answers. A separate process owns all private rules
and exact optimum diagnostics. This boundary prevents benchmark truth from
becoming learner input.

Spec refs: REQ-CL-7324 and SCENARIO-CL-7324-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

from carnot import experiment_7323_v643_addition_prototype as prototype
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    run_commands,
    run_scoped_validation,
)


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7324
SCHEMA = "carnot.experiment_7324.v643_addition_learning.v1"
MILESTONE = "2026.09.643"
RUN_DATE = "20260915"
UPSTREAM_ARTIFACT = Path("results/experiment_7323_v643_addition_prototype.json")
DEFAULT_ARTIFACT = Path("results/experiment_7324_v643_addition_learning.json")
MODULE_PATH = Path("python/carnot/experiment_7324_v643_addition_learning.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7324_v643_addition_learning.py")
TEST_PATH = Path("tests/python/test_experiment_7324_v643_addition_learning.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
REPO_ROOT = Path(__file__).resolve().parents[2]

ARMS = prototype.PRIMARY_ARMS
PERSISTENT_ARM = "persistent_structural_acquisition"
QUERY_BUDGET = prototype.QUERY_BUDGET
STATE_CAP_BYTES = prototype.STATE_CAP_BYTES
STREAM_JOB_LIMIT_S = 2_400.0
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 7_324_901
INTERVENTION_SEED = 7_324_733
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INVOCATION_COUNTS = {
    "loads": {"attempted": 0, "completed": 0, "failed": 0, "cancelled": 0, "in_flight": 0},
    "generations": {
        "attempted": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
        "in_flight": 0,
    },
}

CAPTURE_GATES = (
    "authenticated_inputs",
    "complete_row_capture",
    "cold_reducer_parity",
    "query_and_state_accounting",
    "interventions_complete",
    "required_scoped_validation",
)
VALUE_GATES = (
    "oracle_work_vs_reset",
    "oracle_work_vs_cache",
    "utility_vs_reset",
    "utility_vs_cache",
    "coverage_vs_reset",
    "coverage_vs_cache",
    "returned_infeasible_plans",
    "stale_version_atoms",
    "causal_later_decisions",
    "frozen_arm_retained",
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "status",
    "run_date",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "phase_spans",
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
    "repository_health",
    "field_principles",
    "addition_capture_complete_score",
    "addition_value_score",
    "continuous_self_learning_task",
    "per_stream_results",
    "query_ledger",
    "constraint_update_rows",
    "no_model_weight_mutation",
)


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw evidence, checkpoints, validation logs, and terminal bytes apart."""

    results_root: Path
    raw_dir: Path
    checkpoint_dir: Path
    evaluator_dir: Path
    rows: Path
    query_ledger: Path
    interventions: Path
    terminal_candidate: Path
    validation_dir: Path
    artifact: Path
    upstream_artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the declared production paths under the repository results tree."""

        return cls.from_results_root(REPO_ROOT / "results", REPO_ROOT / UPSTREAM_ARTIFACT)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put test outputs and a replaceable upstream copy below one private root."""

        return cls.from_results_root(root, root / UPSTREAM_ARTIFACT.name)

    @classmethod
    def from_results_root(cls, root: Path, upstream: Path) -> ExperimentPaths:
        """Derive task-owned paths without creating a success-shaped terminal file."""

        raw = root / "raw" / "experiment_7324_v643_addition_learning"
        return cls(
            root,
            raw,
            root / "checkpoints" / "experiment_7324_v643_addition_learning",
            raw / "evaluator",
            raw / "request_rows.jsonl",
            raw / "query_ledger.jsonl",
            raw / "intervention_rows.jsonl",
            raw / "terminal_candidate.json",
            raw / "validation",
            root / DEFAULT_ARTIFACT.name,
            upstream,
        )


@dataclass
class PanelResult:
    """Return complete raw and reduced evidence from the bounded held-out panel."""

    rows: list[JsonDict]
    query_ledger: list[JsonDict]
    per_stream_results: list[JsonDict]
    intervention_rows: list[JsonDict]
    completed_stream_ids: list[str]
    censored_stream_ids: list[str]
    optimum_diagnostics: list[JsonDict]


def _atomic_bytes(path: Path, data: bytes) -> None:
    """Replace one task-owned file only after its full bytes reach stable storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # pragma: no cover - only an interrupted replace leaves this file.
            temporary.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    """Write canonical JSON so hashes bind the same evidence on every replay."""

    _atomic_bytes(path, prototype.canonical_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write complete JSONL evidence with one atomic replacement."""

    data = b"".join(prototype.canonical_bytes(dict(row)) + b"\n" for row in rows)
    _atomic_bytes(path, data)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Persist a query boundary before the learner can observe its response."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(prototype.canonical_bytes(dict(row)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[JsonDict]:
    """Load object rows and reject malformed scalar evidence."""

    if not path.is_file():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("invalid_jsonl_row")
        rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    """Expose the shared project file hash for terminal write receipts."""

    return prototype.sha256_file(path)


def _evaluator_worker(  # pragma: no cover - tests observe this isolated child through receipts.
    environment_id: str,
    stratum: str,
    log_root: str,
    requests: Any,
    responses: Any,
) -> None:
    """Own private rules and return only public views or due Boolean results."""

    environment = prototype._environment(environment_id, stratum)
    log_path = Path(log_root) / "evaluator_requests.jsonl"
    responses.put({"kind": "ready", "public_environment": environment})
    while True:
        message = requests.get()
        operation = message["operation"]
        if operation == "close":
            responses.put({"kind": "closed"})
            return
        request_index = int(message["request_index"])
        executor = prototype._executor_for(environment, request_index)
        request = environment["public_requests"][request_index]
        if operation == "query":
            query_id = str(message["query_id"])
            persisted = {
                "event": "query_persisted",
                "query_id": query_id,
                "request_id": request["request_id"],
                "request_index": request_index,
                "executor_version": executor.version,
                "plan_hash": prototype.sha256_json(message["plan"]),
            }
            _append_jsonl(log_path, persisted)
            started = time.perf_counter()
            accepted = executor.check(request, message["plan"])
            duration = time.perf_counter() - started
            _append_jsonl(
                log_path,
                {
                    **persisted,
                    "event": "response_exposed",
                    "accepted": accepted,
                    "executor_duration_s": duration,
                },
            )
            responses.put(
                {
                    "kind": "query_response",
                    "query_id": query_id,
                    "accepted": accepted,
                    "executor_duration_s": duration,
                }
            )
            continue
        if operation == "optimum":
            started = time.perf_counter()
            names = list(request["activities"])
            best = 0.0
            call_count = 0
            for values in itertools.product(*(request["allowed_slots"][name] for name in names)):
                candidate = prototype._plan(
                    str(request["request_id"]), dict(zip(names, values, strict=True))
                )
                call_count += 1
                if executor.check(request, candidate):
                    best = max(best, prototype._utility(request, candidate))
            responses.put(
                {
                    "kind": "optimum_response",
                    "request_index": request_index,
                    "complete": True,
                    "utility": best,
                    "diagnostic_executor_calls": call_count,
                    "duration_s": time.perf_counter() - started,
                }
            )
            continue
        responses.put({"kind": "error", "error": f"unknown_operation:{operation}"})


class EvaluatorClient:
    """Talk to the private evaluator without importing its rules into learner state."""

    def __init__(self, environment_id: str, stratum: str, log_root: Path) -> None:
        self.environment_id = environment_id
        self.stratum = stratum
        self.log_root = log_root
        self.public_environment: JsonDict = {}
        self._context = multiprocessing.get_context("spawn")
        self._requests: Any = self._context.Queue()
        self._responses: Any = self._context.Queue()
        self._process: Any = None

    def __enter__(self) -> EvaluatorClient:
        """Start a fresh process and accept only its public environment receipt."""

        self.log_root.mkdir(parents=True, exist_ok=True)
        log_path = self.log_root / "evaluator_requests.jsonl"
        if log_path.exists():
            log_path.unlink()
        self._process = self._context.Process(
            target=_evaluator_worker,
            args=(
                self.environment_id,
                self.stratum,
                str(self.log_root),
                self._requests,
                self._responses,
            ),
        )
        self._process.start()
        ready = self._responses.get(timeout=30.0)
        if ready.get("kind") != "ready":  # pragma: no cover - defensive child failure.
            raise RuntimeError("evaluator_start_failed")
        self.public_environment = deepcopy(ready["public_environment"])
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the worker without leaving evaluator work in flight."""

        if self._process is None:  # pragma: no cover - context misuse has no process to close.
            return
        self._requests.put({"operation": "close"})
        closed = self._responses.get(timeout=30.0)
        if closed.get("kind") != "closed":  # pragma: no cover - defensive child failure.
            raise RuntimeError("evaluator_close_failed")
        self._process.join(timeout=30.0)
        if self._process.is_alive():  # pragma: no cover - hard stop for an unresponsive child.
            self._process.terminate()
            self._process.join(timeout=5.0)

    def query(self, request_index: int, plan: Mapping[str, Any], query_id: str) -> JsonDict:
        """Submit one already-persisted plan and expose only its due Boolean response."""

        self._requests.put(
            {
                "operation": "query",
                "request_index": request_index,
                "plan": deepcopy(dict(plan)),
                "query_id": query_id,
            }
        )
        response = self._responses.get(timeout=30.0)
        if response.get("kind") != "query_response" or response.get("query_id") != query_id:
            raise RuntimeError("evaluator_query_mismatch")  # pragma: no cover - child corruption.
        return dict(response)

    def optimum(self, request_index: int) -> JsonDict:
        """Request a benchmark-only optimum summary after learner work finishes."""

        self._requests.put({"operation": "optimum", "request_index": request_index})
        response = self._responses.get(timeout=30.0)
        if response.get("kind") != "optimum_response":
            raise RuntimeError("evaluator_optimum_mismatch")  # pragma: no cover
        return dict(response)


class ExecutorProxy:
    """Present the small executor interface while retaining only public identity."""

    def __init__(
        self,
        evaluator: EvaluatorClient,
        request: Mapping[str, Any],
        version: str,
        arm: str,
        *,
        response_transform: Callable[[bool, int], bool] | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.request = deepcopy(dict(request))
        self.version = str(version)
        self.arm = arm
        self.identity = prototype.sha256_json(
            {"environment": evaluator.environment_id, "version": self.version}
        )
        self.response_transform = response_transform
        self.executor_call_count = 0
        self.executor_duration_s = 0.0
        self.pending_query_id: str | None = None

    def check(self, request: Mapping[str, Any], plan: Mapping[str, Any]) -> bool:
        """Expose one due Boolean and retain the private executor duration only."""

        if request.get("request_id") != self.request.get("request_id"):
            return False
        query_id = self.pending_query_id or prototype.sha256_json(
            {"arm": self.arm, "plan": plan, "sequence": self.executor_call_count + 1}
        )
        response = self.evaluator.query(
            int(str(request["request_id"]).rsplit("-", 1)[-1]), plan, query_id
        )
        self.executor_call_count += 1
        self.executor_duration_s += float(response["executor_duration_s"])
        accepted = bool(response["accepted"])
        if self.response_transform is not None:
            accepted = self.response_transform(accepted, self.executor_call_count)
        return accepted


class PersistingOracle(prototype.ChargedOracle):
    """Seal each prediction and query before any Boolean reaches the learner."""

    def __init__(
        self,
        executor: ExecutorProxy,
        request: Mapping[str, Any],
        ledger_path: Path,
        arm: str,
        request_index: int,
        *,
        exact_cache: dict[str, bool] | None = None,
        diagnostic_only: bool = False,
    ) -> None:
        super().__init__(executor, request, exact_cache=exact_cache)
        self.ledger_path = ledger_path
        self.arm = arm
        self.request_index = request_index
        self.diagnostic_only = diagnostic_only

    def query(self, plan: Mapping[str, Any], reason: str, *, allow_cache: bool = False) -> bool:
        """Persist request bytes before delegating to cache or private execution."""

        sequence = self.attempt_count + 1
        query_id = prototype.sha256_json(
            {
                "arm": self.arm,
                "request_id": self.request["request_id"],
                "request_index": self.request_index,
                "sequence": sequence,
                "reason": reason,
                "plan": plan,
            }
        )
        submitted = {
            "event": "query_submitted_before_response",
            "query_id": query_id,
            "stream_id": self.executor.evaluator.environment_id,
            "request_id": self.request["request_id"],
            "request_index": self.request_index,
            "arm": self.arm,
            "reason": reason,
            "plan_hash": prototype.sha256_json(plan),
            "executor_version": self.executor.version,
            "diagnostic_only": self.diagnostic_only,
        }
        _append_jsonl(self.ledger_path, submitted)
        self.executor.pending_query_id = query_id
        accepted = super().query(plan, reason, allow_cache=allow_cache)
        self.executor.pending_query_id = None
        receipt = self.receipts[-1]
        receipt.update(
            {
                "query_id": query_id,
                "stream_id": self.executor.evaluator.environment_id,
                "request_id": self.request["request_id"],
                "request_index": self.request_index,
                "arm": self.arm,
                "diagnostic_only": self.diagnostic_only,
            }
        )
        _append_jsonl(
            self.ledger_path,
            {**submitted, "event": "query_resolved", "accepted": accepted, **receipt},
        )
        return accepted


def _clone_learner(
    state: bytes, version: str, *, frozen: bool = False
) -> prototype.StructuralAdditionLearner:
    """Restore an exact saved prefix for causal interventions."""

    learner = prototype.StructuralAdditionLearner(version, frozen=frozen)
    learner._state = json.loads(state)  # noqa: SLF001 - exact prefix replay is the intervention.
    learner.activate_version(version)
    return learner


def _state_accounting(
    learner: prototype.StructuralAdditionLearner,
    cache: Mapping[str, bool],
    rollback: bytes,
    pending_ids: Sequence[str],
) -> JsonDict:
    """Charge all persistent and transactional bytes under one sealed limit."""

    categories = {
        "learner_memory_bytes": len(learner.state_bytes()),
        "exact_cache_bytes": len(prototype.canonical_bytes(cache)),
        "pending_query_id_bytes": len(prototype.canonical_bytes(list(pending_ids))),
        "rollback_image_bytes": len(rollback),
    }
    categories["sealed_state_bytes"] = sum(categories.values())
    return categories


def _request_row(
    evaluator: EvaluatorClient,
    learner: prototype.StructuralAdditionLearner,
    cache: dict[str, bool],
    request: Mapping[str, Any],
    request_index: int,
    arm: str,
    ledger_path: Path,
    *,
    response_transform: Callable[[bool, int], bool] | None = None,
    diagnostic_only: bool = False,
) -> tuple[JsonDict, list[JsonDict], JsonDict]:
    """Run one bounded request and keep learner and benchmark costs separate."""

    lookup_started = time.perf_counter()
    version = str(evaluator.public_environment["observable_executor_versions"][request_index])
    learner.activate_version(version)
    lookup_duration = time.perf_counter() - lookup_started
    rollback = learner.state_bytes()
    proxy = ExecutorProxy(
        evaluator,
        request,
        version,
        arm,
        response_transform=response_transform,
    )
    oracle = PersistingOracle(
        proxy,
        request,
        ledger_path,
        arm,
        request_index,
        exact_cache=cache if arm == "exact_plan_cache_reset_learner" else None,
        diagnostic_only=diagnostic_only,
    )
    started = time.perf_counter()
    censored = False
    censor_reason: str | None = None
    try:
        result = learner.run_request(
            request,
            oracle,
            allow_cache=arm == "exact_plan_cache_reset_learner",
        )
    except prototype.AdditionRejected as error:  # A capped failure stays in the denominator.
        censored = True
        censor_reason = str(error)
        result = {
            "returned_plan": None,
            "final_accepted": False,
            "oracle_calls": oracle.call_count,
            "oracle_attempts": oracle.attempt_count,
            "failed_calls": oracle.failed_call_count,
            "cache_hits": oracle.cache_hits,
            "cache_misses": oracle.cache_misses,
            "final_checks": sum(row["reason"] == "final" for row in oracle.receipts),
            "new_atoms": [],
            "active_atom_count": len(learner.active_atoms()),
            "influenced_atom_ids": [],
            "solve_duration_s": 0.0,
            "state_bytes": len(learner.state_bytes()),
            "state_hash": learner.state_hash(),
        }
    request_duration = time.perf_counter() - started
    optimum = evaluator.optimum(request_index)
    returned = result["returned_plan"]
    utility = prototype._utility(request, returned)
    attainable = float(optimum["utility"]) if optimum["complete"] else None
    pending_receipt_ids = [str(oracle.receipts[-1]["query_id"])] if oracle.receipts else []
    accounting = _state_accounting(learner, cache, rollback, pending_receipt_ids)
    update_duration = max(
        0.0,
        request_duration - float(result["solve_duration_s"]) - proxy.executor_duration_s,
    )
    active = learner.active_atoms()
    row = {
        "stream_id": evaluator.environment_id,
        "stratum": evaluator.stratum,
        "request_id": request["request_id"],
        "request_index": request_index,
        "primary_window": request_index >= 4,
        "arm": arm,
        "diagnostic_only": diagnostic_only,
        "executor_version": version,
        "prediction_persisted_before_response": True,
        "oracle_calls": int(result["oracle_calls"]),
        "oracle_attempts": int(result["oracle_attempts"]),
        "failed_calls": int(result["failed_calls"]),
        "cache_hits": int(result["cache_hits"]),
        "cache_misses": int(result["cache_misses"]),
        "final_checks": int(result["final_checks"]),
        "returned": returned is not None,
        "abstained": returned is None,
        "returned_infeasible": returned is not None and not bool(result["final_accepted"]),
        "returned_plan_hash": prototype.sha256_json(returned) if returned is not None else None,
        "utility": utility,
        "attainable_reward": attainable,
        "optimum_censored": not bool(optimum["complete"]),
        "utility_fraction": utility / attainable if attainable else 0.0,
        "active_atom_count": int(result["active_atom_count"]),
        "new_atoms": deepcopy(result["new_atoms"]),
        "new_atom_count": len(result["new_atoms"]),
        "influenced_atom_ids": list(result["influenced_atom_ids"]),
        "stale_version_atom_count": sum(row["version"] != version for row in active),
        "lookup_duration_s": lookup_duration,
        "update_duration_s": update_duration,
        "solver_duration_s": float(result["solve_duration_s"]),
        "executor_duration_s": proxy.executor_duration_s,
        "request_duration_s": request_duration + lookup_duration,
        **accounting,
        "censored": censored,
        "censor_reason": censor_reason,
    }
    diagnostic = {
        "stream_id": evaluator.environment_id,
        "request_id": request["request_id"],
        "request_index": request_index,
        "arm": arm,
        "complete": bool(optimum["complete"]),
        "diagnostic_executor_calls": int(optimum["diagnostic_executor_calls"]),
        "diagnostic_duration_s": float(optimum["duration_s"]),
        "learner_input": False,
    }
    return row, [deepcopy(dict(item)) for item in oracle.receipts], diagnostic


def _stream_interventions(
    evaluator: EvaluatorClient,
    prefix: bytes,
    ledger_path: Path,
    primary_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Replay three declared controls from one exact post-warmup prefix."""

    rows: list[JsonDict] = []
    ledger: list[JsonDict] = []
    diagnostics: list[JsonDict] = []
    prefix_hash = prototype.sha256_bytes(prefix)
    persistent_by_index = {
        int(row["request_index"]): row
        for row in primary_rows
        if row["arm"] == PERSISTENT_ARM and int(row["request_index"]) >= 4
    }
    names = ("feedback_withheld", "label_shuffled", "learned_atom_erasure")
    for intervention in names:
        version = str(evaluator.public_environment["observable_executor_versions"][4])
        learner = _clone_learner(prefix, version, frozen=intervention == "feedback_withheld")
        if intervention == "learned_atom_erasure":
            learner._state["atoms"] = []  # noqa: SLF001 - this is the declared erasure control.
            learner.frozen = True
        for request_index in range(4, 24):
            request = evaluator.public_environment["public_requests"][request_index]
            transform: Callable[[bool, int], bool] | None = None
            if intervention == "label_shuffled":
                transform = lambda accepted, sequence, index=request_index: (
                    not accepted if (index + sequence + INTERVENTION_SEED) % 2 else accepted
                )
            row, receipts, diagnostic = _request_row(
                evaluator,
                learner,
                {},
                request,
                request_index,
                f"intervention:{intervention}",
                ledger_path,
                response_transform=transform,
                diagnostic_only=True,
            )
            reference = persistent_by_index[request_index]
            rows.append(
                {
                    "stream_id": evaluator.environment_id,
                    "stratum": evaluator.stratum,
                    "intervention": intervention,
                    "request_id": request["request_id"],
                    "request_index": request_index,
                    "same_prefix_hash": prefix_hash,
                    "decision_changed": row["returned_plan_hash"]
                    != reference["returned_plan_hash"],
                    "oracle_calls": row["oracle_calls"],
                    "returned": row["returned"],
                    "state_hash": learner.state_hash(),
                    "diagnostic_only": True,
                }
            )
            ledger.extend(receipts)
            diagnostics.append(diagnostic)
    return rows, ledger, diagnostics


def _run_stream(
    evaluator: EvaluatorClient,
    paths: ExperimentPaths,
    *,
    progress: bool,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict], bool]:
    """Run four arms on identical requests and checkpoint every request boundary."""

    environment = evaluator.public_environment
    first_version = str(environment["observable_executor_versions"][0])
    learners = {arm: prototype.StructuralAdditionLearner(first_version) for arm in ARMS}
    caches: dict[str, dict[str, bool]] = {arm: {} for arm in ARMS}
    rows: list[JsonDict] = []
    ledger: list[JsonDict] = []
    diagnostics: list[JsonDict] = []
    prefix: bytes | None = None
    stream_started = time.monotonic()
    censored = False
    local_ledger = evaluator.log_root / "learner_query_events.jsonl"
    if local_ledger.exists():
        local_ledger.unlink()
    for request_index, request in enumerate(environment["public_requests"]):
        if time.monotonic() - stream_started > STREAM_JOB_LIMIT_S:
            censored = True  # pragma: no cover - the bounded fixture finishes in seconds.
            break
        for arm in ARMS:
            reset = arm in {"reset_each_request_acquisition", "exact_plan_cache_reset_learner"}
            learner = (
                prototype.StructuralAdditionLearner(
                    str(environment["observable_executor_versions"][request_index])
                )
                if reset
                else learners[arm]
            )
            learner.frozen = arm == "frozen_after_four_request_warmup" and request_index >= 4
            row, receipts, optimum = _request_row(
                evaluator,
                learner,
                caches[arm],
                request,
                request_index,
                arm,
                local_ledger,
            )
            rows.append(row)
            ledger.extend(receipts)
            diagnostics.append(optimum)
            if not reset:
                learners[arm] = learner
        if request_index == 3:
            prefix = learners[PERSISTENT_ARM].state_bytes()
            _atomic_bytes(
                paths.checkpoint_dir / f"{evaluator.environment_id}-warmup-prefix.bin", prefix
            )
        _atomic_json(
            paths.checkpoint_dir / f"{evaluator.environment_id}-{request_index:02d}.json",
            {
                "schema": SCHEMA,
                "status": "complete",
                "stream_id": evaluator.environment_id,
                "request_index": request_index,
                "completed_arm_count": len(ARMS),
                "row_hash": prototype.sha256_json(rows[-len(ARMS) :]),
                "pending_query_ids": [],
            },
        )
        if progress:
            print(
                f"[exp7324] phase=stream_request event=end stream={evaluator.environment_id} "
                f"unit={request_index + 1}/24 elapsed_s={time.monotonic() - stream_started:.3f}",
                flush=True,
            )
    if prefix is None:  # pragma: no cover - only a pre-warmup timeout can reach this state.
        return rows, ledger, [], diagnostics, True
    intervention_rows, intervention_ledger, intervention_diagnostics = _stream_interventions(
        evaluator, prefix, local_ledger, rows
    )
    ledger.extend(intervention_ledger)
    diagnostics.extend(intervention_diagnostics)
    return rows, ledger, intervention_rows, diagnostics, censored


def reduce_stream_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce post-warmup request rows while preserving each stream and arm."""

    keys = sorted({(str(row["stream_id"]), str(row["arm"])) for row in rows})
    reduced: list[JsonDict] = []
    for stream_id, arm in keys:
        selected = [
            row
            for row in rows
            if row["stream_id"] == stream_id and row["arm"] == arm and row["primary_window"]
        ]
        if not selected:
            continue
        reduced.append(
            {
                "stream_id": stream_id,
                "stratum": selected[0]["stratum"],
                "arm": arm,
                "primary_request_count": len(selected),
                "primary_oracle_calls": sum(int(row["oracle_calls"]) for row in selected),
                "primary_oracle_attempts": sum(int(row["oracle_attempts"]) for row in selected),
                "mean_utility_fraction": sum(float(row["utility_fraction"]) for row in selected)
                / len(selected),
                "coverage_rate": sum(bool(row["returned"]) for row in selected) / len(selected),
                "returned_infeasible_count": sum(
                    bool(row["returned_infeasible"]) for row in selected
                ),
                "stale_version_atom_count": sum(
                    int(row["stale_version_atom_count"]) for row in selected
                ),
                "censored_request_count": sum(bool(row["censored"]) for row in selected),
                "maximum_sealed_state_bytes": max(
                    int(row["sealed_state_bytes"]) for row in selected
                ),
                "lookup_duration_s": sum(float(row["lookup_duration_s"]) for row in selected),
                "update_duration_s": sum(float(row["update_duration_s"]) for row in selected),
                "solver_duration_s": sum(float(row["solver_duration_s"]) for row in selected),
                "executor_duration_s": sum(float(row["executor_duration_s"]) for row in selected),
                "request_duration_s": sum(float(row["request_duration_s"]) for row in selected),
            }
        )
    return reduced


def bootstrap_ci95(values: Sequence[float], salt: str) -> JsonDict:
    """Resample paired stream statistics with one sealed deterministic seed."""

    if not values:
        raise ValueError("paired_streams_unavailable")
    seed_offset = int(prototype.sha256_bytes(salt.encode()).split(":", 1)[1][:12], 16)
    generator = random.Random(BOOTSTRAP_SEED + seed_offset)
    draws = []
    count = len(values)
    for _ in range(BOOTSTRAP_DRAWS):
        draws.append(sum(values[generator.randrange(count)] for _ in range(count)) / count)
    draws.sort()
    return {
        "estimate": sum(values) / count,
        "ci95_lower": draws[int(0.025 * BOOTSTRAP_DRAWS)],
        "ci95_upper": draws[int(0.975 * BOOTSTRAP_DRAWS) - 1],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "paired_stream_count": count,
        "independent_unit": "stream",
    }


def build_comparison_rows(per_stream: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build overall and stratum intervals from paired stream summaries."""

    by_key = {(row["stream_id"], row["arm"]): row for row in per_stream}
    stream_ids = sorted({str(row["stream_id"]) for row in per_stream})
    strata = ("overall", "stationary", "announced_version_change", "return_to_known_version")
    specs = (
        ("oracle_work_vs_reset", "reset_each_request_acquisition", "ratio", "primary_oracle_calls"),
        ("oracle_work_vs_cache", "exact_plan_cache_reset_learner", "ratio", "primary_oracle_calls"),
        (
            "oracle_work_vs_frozen",
            "frozen_after_four_request_warmup",
            "ratio",
            "primary_oracle_calls",
        ),
        (
            "utility_vs_reset",
            "reset_each_request_acquisition",
            "difference",
            "mean_utility_fraction",
        ),
        (
            "utility_vs_cache",
            "exact_plan_cache_reset_learner",
            "difference",
            "mean_utility_fraction",
        ),
        ("coverage_vs_reset", "reset_each_request_acquisition", "difference", "coverage_rate"),
        ("coverage_vs_cache", "exact_plan_cache_reset_learner", "difference", "coverage_rate"),
    )
    comparisons: list[JsonDict] = []
    for stratum in strata:
        selected_ids = [
            stream_id
            for stream_id in stream_ids
            if stratum == "overall" or by_key[(stream_id, PERSISTENT_ARM)]["stratum"] == stratum
        ]
        if not selected_ids:
            continue
        for comparison_id, control, operation, metric in specs:
            values = []
            for stream_id in selected_ids:
                learned = float(by_key[(stream_id, PERSISTENT_ARM)][metric])
                baseline = float(by_key[(stream_id, control)][metric])
                values.append(
                    learned / baseline if operation == "ratio" and baseline else learned - baseline
                )
            comparisons.append(
                {
                    "comparison_id": comparison_id,
                    "stratum": stratum,
                    "learned_arm": PERSISTENT_ARM,
                    "control_arm": control,
                    "metric": metric,
                    "operation": operation,
                    **bootstrap_ci95(values, f"{comparison_id}:{stratum}"),
                }
            )
    return comparisons


def build_constraint_update_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Link each admitted atom to only later distinct requests it influenced."""

    updates: list[JsonDict] = []
    persistent = [row for row in rows if row["arm"] == PERSISTENT_ARM]
    for row in persistent:
        for atom in row["new_atoms"]:
            affected = [
                later["request_id"]
                for later in persistent
                if later["stream_id"] == row["stream_id"]
                and int(later["request_index"]) > int(row["request_index"])
                and atom["atom_id"] in later["influenced_atom_ids"]
            ]
            updates.append(
                {
                    "stream_id": row["stream_id"],
                    "source_request_id": row["request_id"],
                    "source_request_index": row["request_index"],
                    "atom_id": atom["atom_id"],
                    "atom_kind": atom["kind"],
                    "executor_version": atom["version"],
                    "witness": deepcopy(atom["witness"]),
                    "query_receipts": deepcopy(atom["query_receipts"]),
                    "affected_future_request_ids": affected,
                    "affected_future_request_count": len(affected),
                    "structural_change_only": not affected,
                }
            )
    return updates


def cold_reduce(rows_path: Path, query_path: Path) -> JsonDict:
    """Reload immutable raw rows and independently count the charged evidence."""

    rows = read_jsonl(rows_path)
    ledger = read_jsonl(query_path)
    primary_ledger = [row for row in ledger if row.get("diagnostic_only") is False]
    return {
        "row_count": len(rows),
        "row_hash": prototype.sha256_json(rows),
        "query_row_count": len(ledger),
        "primary_query_row_count": len(primary_ledger),
        "query_hash": prototype.sha256_json(ledger),
        "primary_oracle_attempts": sum(int(row["oracle_attempts"]) for row in rows),
        "primary_ledger_attempts": len(primary_ledger),
        "per_stream_results": reduce_stream_rows(rows),
    }


def run_panel(
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    progress: bool = True,
) -> PanelResult:
    """Materialize each held-out stream privately and run the fixed panel once."""

    manifest = prototype.build_stream_manifest()
    environments = {row["environment_id"]: row for row in manifest["held_out"]["environments"]}
    selected = list(stream_ids or sorted(environments))
    unknown = [stream_id for stream_id in selected if stream_id not in environments]
    if unknown:
        raise ValueError("unknown_stream:" + ",".join(unknown))
    all_rows: list[JsonDict] = []
    all_ledger: list[JsonDict] = []
    all_interventions: list[JsonDict] = []
    all_diagnostics: list[JsonDict] = []
    completed: list[str] = []
    censored_ids: list[str] = []
    for unit, stream_id in enumerate(selected, start=1):
        environment = environments[stream_id]
        if progress:
            print(
                f"[exp7324] phase=held_out_stream event=start unit={unit}/{len(selected)} "
                f"stream={stream_id}",
                flush=True,
            )
        with EvaluatorClient(
            stream_id, str(environment["stratum"]), paths.evaluator_dir / stream_id
        ) as evaluator:
            observed = evaluator.public_environment
            if observed["public_view_hash"] != environment["public_view_hash"]:
                raise ValueError("evaluator_public_view_hash")
            rows, ledger, interventions, diagnostics, censored = _run_stream(
                evaluator, paths, progress=progress
            )
        all_rows.extend(rows)
        all_ledger.extend(ledger)
        all_interventions.extend(interventions)
        all_diagnostics.extend(diagnostics)
        if censored:
            censored_ids.append(stream_id)
        else:
            completed.append(stream_id)
        _write_jsonl(paths.rows, all_rows)
        _write_jsonl(paths.query_ledger, all_ledger)
        _write_jsonl(paths.interventions, all_interventions)
        if progress:
            print(
                f"[exp7324] phase=held_out_stream event=end unit={unit}/{len(selected)} "
                f"stream={stream_id} rows={len(rows)} censored={censored}",
                flush=True,
            )
    return PanelResult(
        all_rows,
        all_ledger,
        reduce_stream_rows(all_rows),
        all_interventions,
        completed,
        censored_ids,
        all_diagnostics,
    )


def precondition_gate(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    principle: str,
) -> JsonDict:
    """Keep an external check and its exact observed value in one row."""

    if isinstance(expected, str) and expected.startswith("not "):
        passed = observed != expected.removeprefix("not ")
    else:
        passed = observed == expected
    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
        "principle": principle,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve every check and the first failure without replacing its values."""

    failures = [deepcopy(dict(row)) for row in checks if row.get("passed") is not True]
    return {
        "passed": not failures,
        "check_count": len(checks),
        "failed_check_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "checks": [deepcopy(dict(row)) for row in checks],
    }


def _load_object(path: Path) -> JsonDict:
    """Load an object artifact while treating malformed external bytes as absent."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate Exp7323, current sources, exclusion state, and output ownership."""

    root = repo_root.resolve()
    upstream = _load_object(paths.upstream_artifact)
    upstream_path = str(paths.upstream_artifact)
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    sources = (
        SPEC_PATH,
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("research-program.md"),
        Path("research-references.md"),
        Path("python/carnot/experiment_7323_v643_addition_prototype.py"),
        Path("python/carnot/memory/transactional_constraint_memory.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    source_hashes = {
        path.as_posix(): prototype.sha256_file(root / path)
        for path in sources
        if (root / path).is_file()
    }
    hashes: JsonDict = {
        "current_sources": source_hashes,
        "producer_artifact": {
            "path": upstream_path,
            "sha256": prototype.sha256_file(paths.upstream_artifact)
            if paths.upstream_artifact.is_file()
            else None,
            "schema": upstream.get("schema"),
            "status": upstream.get("status"),
            "verdict_class": upstream.get("verdict_class"),
            "addition_fixture_ready_score": upstream.get("addition_fixture_ready_score"),
        },
        "historical_diagnostics": {
            "results/experiment_7311_v642_factor_learning.json": {
                "sha256": prototype.sha256_file(
                    root / "results/experiment_7311_v642_factor_learning.json"
                ),
                "readiness_authority": False,
            },
            "results/experiment_7199_v634_bounded_acquisition.json": {
                "sha256": prototype.sha256_file(
                    root / "results/experiment_7199_v634_bounded_acquisition.json"
                ),
                "readiness_authority": False,
            },
        },
    }
    checks = [
        precondition_gate(
            "driving_capability_spec",
            str(root / SPEC_PATH),
            "REQ-CL-7324",
            True,
            "REQ-CL-7324" in spec_text,
            "Implementation starts only after its specific requirement exists.",
        ),
        precondition_gate(
            "scenario_contract",
            str(root / SPEC_PATH),
            "SCENARIO-CL-7324-*",
            7,
            spec_text.count("### SCENARIO-CL-7324-"),
            "Every required behavior has a named scenario.",
        ),
        precondition_gate(
            "upstream_available",
            upstream_path,
            "path",
            True,
            paths.upstream_artifact.is_file(),
            "A missing producer cannot authorize held-out work.",
        ),
        precondition_gate(
            "upstream_schema",
            upstream_path,
            "schema",
            prototype.SCHEMA,
            upstream.get("schema"),
            "Only the declared addition prototype schema can authorize evaluation.",
        ),
        precondition_gate(
            "upstream_terminal",
            upstream_path,
            "status",
            "complete",
            upstream.get("status"),
            "Incomplete external work cannot authorize a prospective claim.",
        ),
        precondition_gate(
            "addition_fixture_ready",
            upstream_path,
            "addition_fixture_ready_score",
            1,
            upstream.get("addition_fixture_ready_score"),
            "Only the complete bounded addition fixture can authorize evaluation.",
        ),
        precondition_gate(
            "upstream_not_quarantined",
            upstream_path,
            "flagged_adversarial",
            False,
            bool(upstream.get("flagged_adversarial", False)),
            "Quarantined evidence cannot authorize a later experiment.",
        ),
        *(
            precondition_gate(
                f"upstream_not_{verdict}",
                upstream_path,
                "verdict_class",
                f"not {verdict}",
                upstream.get("verdict_class"),
                "A numeric readiness score cannot override a failed terminal class.",
            )
            for verdict in ("disqualified", "blocked", "partial")
        ),
        precondition_gate(
            "upstream_cold_validation",
            upstream_path,
            "validate_artifact",
            [],
            prototype.validate_artifact(upstream, require_validation=True)
            if upstream
            else ["absent"],
            "The producer checksum and required validation receipts must remain valid.",
        ),
        precondition_gate(
            "current_task_not_excluded",
            str(root / "ops/exclusion_manifest.yaml"),
            "experiment_7324_v643_addition_learning",
            False,
            "experiment_7324_v643_addition_learning" in exclusion_text,
            "An excluded current task stops before measurement.",
        ),
        precondition_gate(
            "source_bytes_available",
            "declared current sources",
            "sha256",
            len(sources),
            len(source_hashes),
            "Every executable and evidence identity must be hashable before measurement.",
        ),
    ]
    return checks, hashes, upstream


def gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Record one frozen acceptance rule with both compatibility pass keys."""

    return {
        "expected": expected,
        "observed": observed,
        "pass": bool(passed),
        "passed": bool(passed),
        "principle": principle,
    }


def _comparison(
    comparisons: Sequence[Mapping[str, Any]], comparison_id: str, stratum: str = "overall"
) -> Mapping[str, Any]:
    """Select one preregistered interval or fail instead of inventing a zero."""

    return next(
        row
        for row in comparisons
        if row["comparison_id"] == comparison_id and row["stratum"] == stratum
    )


def _scientific_gates(
    per_stream: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Apply all sealed work, utility, coverage, safety, and version gates jointly."""

    ratio_reset = _comparison(comparisons, "oracle_work_vs_reset")
    ratio_cache = _comparison(comparisons, "oracle_work_vs_cache")
    utility_reset = _comparison(comparisons, "utility_vs_reset")
    utility_cache = _comparison(comparisons, "utility_vs_cache")
    coverage_reset = _comparison(comparisons, "coverage_vs_reset")
    coverage_cache = _comparison(comparisons, "coverage_vs_cache")
    infeasible = sum(int(row["returned_infeasible_count"]) for row in per_stream)
    stale = sum(int(row["stale_version_atom_count"]) for row in per_stream)
    affected = sum(int(row["affected_future_request_count"]) for row in updates)
    frozen_present = len([row for row in per_stream if row["arm"] == ARMS[3]])
    stream_count = len({row["stream_id"] for row in per_stream})
    return {
        "oracle_work_vs_reset": gate(
            "paired CI95 upper<0.90",
            ratio_reset["ci95_upper"],
            float(ratio_reset["ci95_upper"]) < 0.90,
            "Persistent structure must reduce total executor work against reset.",
        ),
        "oracle_work_vs_cache": gate(
            "paired CI95 upper<0.90",
            ratio_cache["ci95_upper"],
            float(ratio_cache["ci95_upper"]) < 0.90,
            "Persistent structure must beat an exact-cache control.",
        ),
        "utility_vs_reset": gate(
            "paired CI95 lower>=-0.02",
            utility_reset["ci95_lower"],
            float(utility_reset["ci95_lower"]) >= -0.02,
            "Work savings cannot remove attainable utility.",
        ),
        "utility_vs_cache": gate(
            "paired CI95 lower>=-0.02",
            utility_cache["ci95_lower"],
            float(utility_cache["ci95_lower"]) >= -0.02,
            "Work savings cannot remove attainable utility against caching.",
        ),
        "coverage_vs_reset": gate(
            "paired CI95 lower>=-0.02",
            coverage_reset["ci95_lower"],
            float(coverage_reset["ci95_lower"]) >= -0.02,
            "Work savings cannot come from extra abstention.",
        ),
        "coverage_vs_cache": gate(
            "paired CI95 lower>=-0.02",
            coverage_cache["ci95_lower"],
            float(coverage_cache["ci95_lower"]) >= -0.02,
            "Work savings cannot come from extra abstention against caching.",
        ),
        "returned_infeasible_plans": gate(
            0,
            infeasible,
            infeasible == 0,
            "Every returned plan needs its charged final Boolean acceptance.",
        ),
        "stale_version_atoms": gate(
            0,
            stale,
            stale == 0,
            "Atoms from another executor version have no active authority.",
        ),
        "causal_later_decisions": gate(
            ">=1",
            affected,
            affected >= 1,
            "Only later distinct requests affected by an admitted atom count as learning.",
        ),
        "frozen_arm_retained": gate(
            stream_count,
            frozen_present,
            frozen_present == stream_count,
            "The frozen arm keeps forgetting and opportunity visible.",
        ),
    }


def derive_terminal_scores(gates: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Keep complete capture separate from a positive or null scientific result."""

    capture_failures = [
        name for name in CAPTURE_GATES if gates.get(name, {}).get("passed") is not True
    ]
    value_failures = [name for name in VALUE_GATES if gates.get(name, {}).get("passed") is not True]
    capture = not capture_failures
    value = capture and not value_failures
    if not capture:
        return {
            "addition_capture_complete_score": 0,
            "addition_value_score": 0,
            "verdict_class": "partial",
            "honest_verdict": "partial: held-out capture or required validation is incomplete: "
            + ",".join(capture_failures),
        }
    if not value:
        return {
            "addition_capture_complete_score": 1,
            "addition_value_score": 0,
            "verdict_class": "null",
            "honest_verdict": "complete_null: structural addition capture completed but frozen value gates failed: "
            + ",".join(value_failures),
        }
    return {
        "addition_capture_complete_score": 1,
        "addition_value_score": 1,
        "verdict_class": "circular_positive",
        "honest_verdict": "complete: persistent structural addition reduced held-out executor work under shared exact executor authority",
    }


def _field_principles() -> JsonDict:
    """Explain why each required field exists without wrapping executable values."""

    return {
        "schema": "Version this artifact; keep ordinary top-level experiment_id and milestone.",
        "status": "Write terminal output only after current work and required validation.",
        "run_date": "Use 20260915; preserve actual UTC timestamps and monotonic phase spans.",
        "preconditions_checked": "Record input identities, availability, and the exact failed check.",
        "MODEL_SPECS": "Current executable identities only; no LLM work exists in this CPU experiment.",
        "model_invoked": "True for any actual attempted load or generation, including unusable results.",
        "invocation_counts": "Separate attempted, completed, failed, cancelled, and in-flight operations.",
        "inference_substrate": "Describe actual computation using the recognized substrate literal.",
        "inference_substrate_class": "Use the closed class that matches actual computation.",
        "execution_venue": "Use host for this milestone; historical board work is not current execution.",
        "duration_s": "Measure real elapsed time; never sleep or inflate counts.",
        "phase_spans": "Record disjoint spans, units, checkpoint boundaries, and pending operations.",
        "random_seed": "Seal independent development, evaluation, bootstrap, and intervention seeds.",
        "reproducibility_checksum": "Bind code, public inputs, evaluator identity, settings, and raw evidence.",
        "source_artifact_hashes": "Authenticate exact producers; diagnostics cannot authorize readiness.",
        "rows": "Emit every comparative request and arm with costs, failures, abstentions, and censoring.",
        "sample_size_budget": "Record planned, attempted, complete, and censored counts and stopping rule.",
        "acceptance_gate_results": "Keep each expected value, observation, pass state, and principle together.",
        "gate_check_summary": "Name upstream, check, field, expected value, and observed value.",
        "verifier_is_oracle": "Shared executor authority forbids a positive scientific class.",
        "honest_verdict": "Completed findings use a complete prefix; external absence uses blocked_.",
        "verdict_class": "Use the closed terminal evidence enum.",
        "validation_receipts": "Keep commands, scopes, exits, times, and log hashes, including failures.",
        "repository_health": "Preserve unrelated failures without passing a required current check.",
        "field_principles": "Explain why each field exists without wrapping its executable value.",
        "addition_capture_complete_score": "One requires complete streams, raw rows, and valid scoped checks.",
        "addition_value_score": "One requires every frozen safety, coverage, utility, and work gate.",
        "continuous_self_learning_task": "True because feedback changes constraints used on later requests.",
        "per_stream_results": "Keep each stream and arm with work, utility, feasibility, costs, and memory.",
        "query_ledger": "Count all main and auxiliary calls without selective exclusions.",
        "constraint_update_rows": "Link each new atom to witnesses and later request effects.",
        "no_model_weight_mutation": "True because this experiment learns constraints, not model weights.",
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind scientific evidence while excluding host timing and mutable validation logs."""

    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "MODEL_SPECS",
        "inference_substrate",
        "random_seed",
        "source_artifact_hashes",
        "learner_settings",
        "rows",
        "per_stream_results",
        "query_ledger",
        "constraint_update_rows",
        "intervention_results",
        "comparison_rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "verifier_is_oracle",
    )
    return prototype.sha256_json({key: artifact.get(key) for key in keys})


def _base_artifact(checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, Any]) -> JsonDict:
    """Create one schema-complete record before blocked or measured classification."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 3,
        "status": "complete",
        "run_date": RUN_DATE,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": MODEL_SPECS,
        "model_invoked": MODEL_INVOKED,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": {
            "development": prototype.DEVELOPMENT_SEED,
            "evaluation": prototype.EVALUATION_SEED,
            "bootstrap": BOOTSTRAP_SEED,
            "interventions": INTERVENTION_SEED,
            "sealed_before_results": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "rows": [],
        "sample_size_budget": {},
        "acceptance_gate_results": {},
        "gate_check_summary": gate_check_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "",
        "verdict_class": "partial",
        "validation_receipts": [],
        "required_checks_passed": False,
        "repository_health": prototype.repository_health(),
        "field_principles": _field_principles(),
        "addition_capture_complete_score": 0,
        "addition_value_score": 0,
        "addition_readiness_score": 0,
        "addition_promotion_score": 0,
        "continuous_self_learning_task": True,
        "per_stream_results": [],
        "query_ledger": [],
        "constraint_update_rows": [],
        "no_model_weight_mutation": True,
        "production_default_changed": False,
        "publication_surface_changed": False,
        "learner_settings": {
            "arms": list(ARMS),
            "requests_per_stream": 24,
            "warmup_requests": 4,
            "primary_requests": 20,
            "query_cap_per_request": QUERY_BUDGET,
            "state_cap_bytes": STATE_CAP_BYTES,
            "stream_job_limit_s": STREAM_JOB_LIMIT_S,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "executor_response": "Boolean only",
        },
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, Any]
) -> JsonDict:
    """Return a row-free terminal record for unchanged external failure."""

    artifact = _base_artifact(checks, hashes)
    failure = artifact["gate_check_summary"]["first_failure"]
    if failure is None:
        raise ValueError("blocked_artifact_without_failure")
    artifact.update(
        {
            "status": "blocked",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "verdict_class": "blocked",
            "honest_verdict": f"blocked_{failure['check']}: {failure['field']} expected {failure['expected_value']!r}; observed {failure['observed_value']!r}",
            "sample_size_budget": {
                "planned_stream_count": 24,
                "attempted_stream_count": 0,
                "completed_stream_count": 0,
                "censored_stream_count": 0,
                "planned_request_count": 576,
                "planned_arm_request_rows": 2304,
                "stopping_rule": "external failure is terminal blocked",
            },
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _raw_receipt(path: Path, row_count: int) -> JsonDict:
    """Bind a raw evidence file and its independently countable row total."""

    return {
        "path": str(path),
        "sha256": prototype.sha256_file(path),
        "row_count": row_count,
    }


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    progress: bool = True,
) -> JsonDict:
    """Run the bounded measurement, cold-reduce raw rows, and build a candidate."""

    started = time.monotonic()
    checks, hashes, _upstream = collect_preconditions(repo_root, paths)
    if not gate_check_summary(checks)["passed"]:
        return build_blocked_artifact(checks, hashes)
    artifact = _base_artifact(checks, hashes)
    selected = tuple(stream_ids or (f"held-out-{index:02d}" for index in range(24)))
    phase_started = time.monotonic()
    if progress:
        print(f"[exp7324] phase=panel event=start streams={len(selected)}", flush=True)
    panel = run_panel(paths, stream_ids=selected, progress=progress)
    panel_end = time.monotonic()
    if progress:
        print(
            f"[exp7324] phase=panel event=end streams={len(panel.completed_stream_ids)} "
            f"rows={len(panel.rows)}",
            flush=True,
        )
    cold_started = time.monotonic()
    cold = cold_reduce(paths.rows, paths.query_ledger)
    if cold["row_hash"] != prototype.sha256_json(panel.rows):
        raise ValueError("independent_reducer_mismatch")
    comparisons = build_comparison_rows(panel.per_stream_results)
    updates = build_constraint_update_rows(panel.rows)
    full_capture = len(selected) == 24 and not panel.censored_stream_ids
    primary_ledger = [row for row in panel.query_ledger if row["diagnostic_only"] is False]
    query_parity = len(primary_ledger) == sum(int(row["oracle_attempts"]) for row in panel.rows)
    bounded = all(
        int(row["oracle_calls"]) <= QUERY_BUDGET
        and int(row["sealed_state_bytes"]) <= STATE_CAP_BYTES
        for row in panel.rows
    )
    intervention_names = {row["intervention"] for row in panel.intervention_rows}
    gates: dict[str, JsonDict] = {
        "authenticated_inputs": gate(
            True,
            artifact["gate_check_summary"]["passed"],
            bool(artifact["gate_check_summary"]["passed"]),
            "Only exact eligible Exp7323 evidence can start the panel.",
        ),
        "complete_row_capture": gate(
            [24, 2304, 0],
            [len(panel.completed_stream_ids), len(panel.rows), len(panel.censored_stream_ids)],
            full_capture and len(panel.rows) == 2304,
            "Every planned stream, request, and arm stays in the fixed denominator.",
        ),
        "cold_reducer_parity": gate(
            prototype.sha256_json(panel.rows),
            cold["row_hash"],
            cold["row_hash"] == prototype.sha256_json(panel.rows),
            "Reloaded raw rows, not producer aggregates, own the terminal counts.",
        ),
        "query_and_state_accounting": gate(
            True,
            query_parity and bounded,
            query_parity and bounded,
            "Every query and all sealed state categories stay charged within their caps.",
        ),
        "interventions_complete": gate(
            ["feedback_withheld", "label_shuffled", "learned_atom_erasure"],
            sorted(intervention_names),
            intervention_names == {"feedback_withheld", "label_shuffled", "learned_atom_erasure"},
            "All declared controls start from saved warmup prefixes.",
        ),
        "required_scoped_validation": gate(
            True,
            False,
            False,
            "Capture becomes complete only after every scoped required check passes.",
        ),
        **_scientific_gates(panel.per_stream_results, comparisons, updates),
    }
    scores = derive_terminal_scores(gates)
    artifact.update(
        {
            "rows": panel.rows,
            "per_stream_results": panel.per_stream_results,
            "query_ledger": panel.query_ledger,
            "constraint_update_rows": updates,
            "intervention_results": panel.intervention_rows,
            "comparison_rows": comparisons,
            "optimum_diagnostics": panel.optimum_diagnostics,
            "sample_size_budget": {
                "planned_stream_count": 24,
                "attempted_stream_count": len(selected),
                "completed_stream_count": len(panel.completed_stream_ids),
                "censored_stream_count": len(panel.censored_stream_ids),
                "censored_stream_ids": panel.censored_stream_ids,
                "requests_per_stream": 24,
                "warmup_requests_per_stream": 4,
                "primary_requests_per_stream": 20,
                "arms_per_request": 4,
                "planned_request_count": 576,
                "attempted_request_count": len(selected) * 24,
                "complete_request_count": len(panel.rows) // len(ARMS),
                "planned_arm_request_rows": 2304,
                "complete_arm_request_rows": len(panel.rows),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "outcome_based_extension": False,
                "stopping_rule": "run all 24 sealed held-out streams once with no outcome-based extension",
            },
            "acceptance_gate_results": gates,
            "raw_evidence_receipts": {
                "rows": _raw_receipt(paths.rows, len(panel.rows)),
                "query_ledger": _raw_receipt(paths.query_ledger, len(panel.query_ledger)),
                "interventions": _raw_receipt(paths.interventions, len(panel.intervention_rows)),
            },
            "evaluator_process": {
                "separate_process_per_stream": True,
                "learner_visible_fields": ["public_requests", "executor_version", "due_boolean"],
                "private_rules_visible_to_learner": False,
                "optimum_visible_to_learner": False,
                "evaluator_identity": prototype.sha256_json(
                    {
                        "producer_module": hashes["current_sources"].get(
                            "python/carnot/experiment_7323_v643_addition_prototype.py"
                        ),
                        "worker_module": hashes["current_sources"].get(MODULE_PATH.as_posix()),
                        "manifest_hash": prototype.build_stream_manifest()["manifest_hash"],
                    }
                ),
            },
            "phase_spans": [
                {
                    "phase": "held_out_panel",
                    "start_s": phase_started - started,
                    "end_s": panel_end - started,
                    "units": len(panel.rows),
                    "checkpoint_boundaries": len(selected) * 24,
                    "pending_operations": [],
                },
                {
                    "phase": "cold_reduction",
                    "start_s": cold_started - started,
                    "end_s": time.monotonic() - started,
                    "units": len(panel.rows),
                    "checkpoint_boundaries": 1,
                    "pending_operations": [],
                },
            ],
            "duration_s": time.monotonic() - started,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            **scores,
        }
    )
    artifact["addition_readiness_score"] = artifact["addition_capture_complete_score"]
    artifact["addition_promotion_score"] = artifact["addition_value_score"]
    artifact["validation_receipts"] = [
        {
            "name": "cold_raw_reduction",
            "command": f"cold_reduce {paths.rows} {paths.query_ledger}",
            "scope": "SCENARIO-CL-7324-REDUCTION immutable raw rows",
            "exit_code": 0,
            "duration_s": time.monotonic() - cold_started,
            "log_sha256": prototype.sha256_json(cold),
            "passed": True,
            "timed_out": False,
        }
    ]
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact, expected_stream_count=len(selected), check_files=True)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    return artifact


def _receipt_schema_error(receipt: Mapping[str, Any]) -> bool:
    """Reject validation rows that omit command, scope, exit, time, or log identity."""

    return not (
        isinstance(receipt.get("command"), str)
        and bool(receipt.get("command"))
        and isinstance(receipt.get("scope"), str)
        and bool(receipt.get("scope"))
        and isinstance(receipt.get("exit_code"), int)
        and isinstance(receipt.get("duration_s"), (int, float))
        and str(receipt.get("log_sha256", "")).startswith("sha256:")
    )


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_stream_count: int = 24,
    check_files: bool = False,
) -> list[str]:
    """Cold-check identity, boundaries, row counts, raw bytes, gates, and scores."""

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
        "model_boundary",
    )
    add(
        artifact.get("continuous_self_learning_task") is not True
        or artifact.get("model_invoked") is not False
        or artifact.get("no_model_weight_mutation") is not True
        or artifact.get("production_default_changed") is not False
        or artifact.get("publication_surface_changed") is not False,
        "learning_boundary",
    )
    add(
        artifact.get("inference_substrate") != "cpu_exact_solver_or_simulator"
        or artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
        or artifact.get("execution_venue") != "host",
        "substrate",
    )
    add(artifact.get("verifier_is_oracle") is not True, "oracle_declaration")
    add(artifact.get("verdict_class") == "positive", "oracle_positive_forbidden")
    add(
        artifact.get("verdict_class")
        not in {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"},
        "verdict_class",
    )
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
        or any(not isinstance(row, Mapping) or _receipt_schema_error(row) for row in receipts),
        "validation_receipts",
    )
    if artifact.get("status") == "blocked":
        add(
            artifact.get("rows") != []
            or artifact.get("per_stream_results") != []
            or artifact.get("query_ledger") != []
            or artifact.get("constraint_update_rows") != []
            or artifact.get("addition_capture_complete_score") != 0
            or artifact.get("addition_value_score") != 0
            or artifact.get("addition_readiness_score") != 0
            or artifact.get("addition_promotion_score") != 0
            or artifact.get("verdict_class") != "blocked"
            or not str(artifact.get("honest_verdict", "")).startswith("blocked_")
            or artifact.get("gate_check_summary", {}).get("first_failure") is None,
            "blocked_contract",
        )
        return errors
    add(artifact.get("status") != "complete", "status")
    rows = artifact.get("rows", [])
    add(
        not isinstance(rows, list) or len(rows) != expected_stream_count * 24 * len(ARMS),
        "row_count",
    )
    if isinstance(rows, list):
        add(
            any(int(row.get("oracle_calls", QUERY_BUDGET + 1)) > QUERY_BUDGET for row in rows),
            "row_query_cap",
        )
        add(
            any(
                int(row.get("sealed_state_bytes", STATE_CAP_BYTES + 1)) > STATE_CAP_BYTES
                for row in rows
            ),
            "row_state_cap",
        )
        add(
            any(row.get("prediction_persisted_before_response") is not True for row in rows),
            "prediction_order",
        )
    per_stream = artifact.get("per_stream_results", [])
    add(
        not isinstance(per_stream, list)
        or len(per_stream) != expected_stream_count * len(ARMS)
        or any(row.get("primary_request_count") != 20 for row in per_stream),
        "per_stream_results",
    )
    gates = artifact.get("acceptance_gate_results", {})
    add(
        not isinstance(gates, Mapping)
        or any(name not in gates for name in (*CAPTURE_GATES, *VALUE_GATES))
        or any(
            row.get("pass") != row.get("passed")
            or not {"expected", "observed", "pass", "passed", "principle"} <= set(row)
            for row in gates.values()
        ),
        "acceptance_gate_results",
    )
    if isinstance(gates, Mapping):
        scores = derive_terminal_scores(gates)
        if artifact.get("verdict_class") == "disqualified":
            add(
                artifact.get("addition_capture_complete_score") != 0
                or artifact.get("addition_value_score") != 0
                or not str(artifact.get("honest_verdict", "")).startswith("complete_disqualified:"),
                "disqualified_contract",
            )
        else:
            add(
                artifact.get("addition_capture_complete_score")
                != scores["addition_capture_complete_score"],
                "addition_capture_complete_score",
            )
            add(
                artifact.get("addition_value_score") != scores["addition_value_score"],
                "addition_value_score",
            )
            add(
                artifact.get("verdict_class") != scores["verdict_class"]
                or artifact.get("honest_verdict") != scores["honest_verdict"],
                "terminal_classification",
            )
    add(
        artifact.get("verdict_class") in {"blocked", "disqualified"}
        and (
            artifact.get("addition_capture_complete_score") != 0
            or artifact.get("addition_value_score") != 0
            or artifact.get("addition_readiness_score") != 0
            or artifact.get("addition_promotion_score") != 0
        ),
        "failed_scores",
    )
    comparisons = artifact.get("comparison_rows", [])
    add(
        not isinstance(comparisons, list)
        or any(
            row.get("bootstrap_draws") != BOOTSTRAP_DRAWS or row.get("independent_unit") != "stream"
            for row in comparisons
        ),
        "comparison_rows",
    )
    if check_files:
        raw = artifact.get("raw_evidence_receipts", {})
        try:
            raw_rows = Path(raw["rows"]["path"])
            raw_queries = Path(raw["query_ledger"]["path"])
            cold = cold_reduce(raw_rows, raw_queries)
            add(cold["row_hash"] != prototype.sha256_json(rows), "cold_reducer")
            add(cold["per_stream_results"] != per_stream, "cold_per_stream")
            add(prototype.sha256_file(raw_rows) != raw["rows"]["sha256"], "raw_row_hash")
            add(
                prototype.sha256_file(raw_queries) != raw["query_ledger"]["sha256"],
                "raw_query_hash",
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            add(True, "raw_evidence_receipts")
    return errors


def write_artifact(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    expected_stream_count: int = 24,
) -> JsonDict:
    """Validate and publish one terminal object with an atomic rename."""

    errors = validate_artifact(
        artifact,
        expected_stream_count=expected_stream_count,
        check_files=artifact.get("status") == "complete",
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    _atomic_json(path, artifact)
    return {"path": str(path), "sha256": prototype.sha256_file(path)}


def _attach_validation(artifact: Mapping[str, Any], validation: Mapping[str, Any]) -> JsonDict:
    """Attach scoped outcomes, then derive terminal capture and scientific scores."""

    changed = deepcopy(dict(artifact))
    changed.update(
        {
            key: deepcopy(value)
            for key, value in validation.items()
            if key
            in {
                "required_checks_passed",
                "missing_required_commands",
                "failed_required_commands",
                "duplicate_required_commands",
                "repository_health",
            }
        }
    )
    changed["validation_receipts"] = [
        *changed.get("validation_receipts", []),
        *deepcopy(validation.get("validation_receipts", [])),
    ]
    scoped_passed = bool(validation.get("required_checks_passed"))
    changed["acceptance_gate_results"]["required_scoped_validation"] = gate(
        True,
        scoped_passed,
        scoped_passed,
        "Capture becomes complete only after every scoped required check passes.",
    )
    scores = derive_terminal_scores(changed["acceptance_gate_results"])
    changed.update(scores)
    changed["addition_readiness_score"] = changed["addition_capture_complete_score"]
    changed["addition_promotion_score"] = changed["addition_value_score"]
    changed["reproducibility_checksum"] = reproducibility_checksum(changed)
    return changed


def _mark_disqualified(artifact: Mapping[str, Any], failed_names: Sequence[str]) -> JsonDict:
    """Retain measured rows while setting all readiness and promotion scores to zero."""

    changed = deepcopy(dict(artifact))
    changed.update(
        {
            "addition_capture_complete_score": 0,
            "addition_value_score": 0,
            "addition_readiness_score": 0,
            "addition_promotion_score": 0,
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified: current affected validation failed: "
            + ",".join(failed_names),
        }
    )
    changed["reproducibility_checksum"] = reproducibility_checksum(changed)
    return changed


def _run_full_suite(repo_root: Path, log_dir: Path) -> JsonDict:  # pragma: no cover
    """Run the mandated full Python suite once as a repository health observation."""

    command = CommandSpec(
        "full_python_suite",
        (str(repo_root / ".venv/bin/pytest"), "tests/python", "-q"),
        "repository_wide_health_observation_required_by_task",
        timeout_s=4_000.0,
    )
    return run_commands(repo_root, [command], log_dir=log_dir)[0]


def _terminal_validators(
    repo_root: Path, candidate: Path, log_dir: Path
) -> list[JsonDict]:  # pragma: no cover
    """Run adversarial and strict row validation on the measured candidate."""

    commands = [
        CommandSpec(
            "adversarial_verify",
            (
                str(repo_root / ".venv/bin/python"),
                "-u",
                "scripts/adversarial_verify.py",
                str(candidate),
            ),
            "measured_terminal_candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                str(repo_root / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "measured_terminal_candidate",
        ),
    ]
    return run_commands(repo_root, commands, log_dir=log_dir)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed date plus private validation and output-root test seams."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=[RUN_DATE])
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Measure, validate, classify, and atomically publish terminal evidence."""

    print("[exp7324] phase=startup event=start", flush=True)
    args = _parse_args(argv)
    if args.validate is not None:
        print("[exp7324] phase=cold_validation event=start", flush=True)
        artifact = _load_object(args.validate)
        errors = validate_artifact(artifact, check_files=True)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        print(f"[exp7324] phase=cold_validation event=end errors={len(errors)}", flush=True)
        return int(bool(errors))
    paths = (
        ExperimentPaths.from_results_root(
            args.output_root.resolve() / "results", REPO_ROOT / UPSTREAM_ARTIFACT
        )
        if args.output_root
        else ExperimentPaths.defaults()
    )
    invocation_started = time.monotonic()
    print("[exp7324] phase=preconditions event=start", flush=True)
    artifact = build_and_seal(REPO_ROOT, paths, progress=True)
    print(f"[exp7324] phase=measurement event=end status={artifact['status']}", flush=True)
    if artifact["status"] == "blocked":
        write_artifact(paths.artifact, artifact)
        print(f"[exp7324] phase=terminal_write event=end path={paths.artifact}", flush=True)
        return 0

    write_artifact(paths.terminal_candidate, artifact)
    print("[exp7324] phase=scoped_validation event=start", flush=True)
    scoped_started = time.monotonic()
    scoped_basetemp = Path("/tmp/carnot-exp7324-scoped")
    scoped_basetemp.mkdir(parents=True, exist_ok=True)
    validation = run_scoped_validation(
        REPO_ROOT,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        basetemp=scoped_basetemp,
        coverage_file=paths.raw_dir / ".coverage",
        log_dir=paths.validation_dir / "scoped",
        historical_failures=prototype.repository_health()["historical_failures"],
    )
    artifact = _attach_validation(artifact, validation)
    scoped_ended = time.monotonic()
    artifact["phase_spans"].append(
        {
            "phase": "scoped_validation",
            "start_s": scoped_started - invocation_started,
            "end_s": scoped_ended - invocation_started,
            "units": len(REQUIRED_CHECK_NAMES),
            "checkpoint_boundaries": len(REQUIRED_CHECK_NAMES),
            "pending_operations": [],
        }
    )
    write_artifact(paths.terminal_candidate, artifact)
    print(
        f"[exp7324] phase=scoped_validation event=end passed={artifact['required_checks_passed']}",
        flush=True,
    )

    print("[exp7324] phase=terminal_validators event=start", flush=True)
    validators_started = time.monotonic()
    terminal_receipts = _terminal_validators(
        REPO_ROOT, paths.terminal_candidate, paths.validation_dir / "terminal"
    )
    artifact["validation_receipts"].extend(terminal_receipts)
    validators_ended = time.monotonic()
    artifact["phase_spans"].append(
        {
            "phase": "terminal_validators",
            "start_s": validators_started - invocation_started,
            "end_s": validators_ended - invocation_started,
            "units": len(terminal_receipts),
            "checkpoint_boundaries": len(terminal_receipts),
            "pending_operations": [],
        }
    )
    print(
        f"[exp7324] phase=terminal_validators event=end units={len(terminal_receipts)}",
        flush=True,
    )

    print("[exp7324] phase=full_python_suite event=start", flush=True)
    full_started = time.monotonic()
    full_receipt = _run_full_suite(REPO_ROOT, paths.validation_dir / "full_suite")
    full_ended = time.monotonic()
    artifact["phase_spans"].append(
        {
            "phase": "full_python_suite",
            "start_s": full_started - invocation_started,
            "end_s": full_ended - invocation_started,
            "units": 1,
            "checkpoint_boundaries": 1,
            "pending_operations": [],
        }
    )
    print(
        f"[exp7324] phase=full_python_suite event=end exit={full_receipt['exit_code']}",
        flush=True,
    )
    artifact["repository_health"] = {
        **artifact.get("repository_health", {}),
        "current_observation": full_receipt,
        "status": "healthy" if full_receipt["passed"] else "degraded_open",
        "affects_required_checks": (
            not full_receipt["passed"]
            and TEST_PATH.name in str(full_receipt.get("output_tail", ""))
        ),
    }
    failed = [
        str(row.get("name"))
        for row in artifact["validation_receipts"]
        if row.get("passed") is False
    ]
    if artifact["repository_health"]["affects_required_checks"]:
        failed.append("affected_full_python_suite")
    if failed or not artifact.get("required_checks_passed"):
        artifact = _mark_disqualified(artifact, failed or ["scoped_required_checks"])
    artifact["duration_s"] = time.monotonic() - invocation_started
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact, check_files=True)
    if errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(errors))
    print("[exp7324] phase=terminal_write event=start", flush=True)
    write_artifact(paths.artifact, artifact)
    print(f"[exp7324] phase=terminal_write event=end path={paths.artifact}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin script is the supported entrypoint.
    raise SystemExit(main())
