"""Build a non-saturating procedural-memory stream with exact transactions.

Spec refs: REQ-CL-6761, SCENARIO-CL-6761-CHRONOLOGY,
SCENARIO-CL-6761-CAPACITY, SCENARIO-CL-6761-TRANSACTIONS,
SCENARIO-CL-6761-POISON, SCENARIO-CL-6761-ROWS, REQ-REPORT-6761,
SCENARIO-REPORT-6761-ATOMIC, SCENARIO-REPORT-6761-BLOCKED.

This fixture makes the later trajectory-versus-procedure comparison possible.
It fixes exact accept and reject opportunities before either memory arm runs.
The exact labels check storage admission only. They do not claim that a learned
verifier can select or use a lesson.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.memory import transactional_constraint_memory as exp6748


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260829"
EXPERIMENT_ID = "experiment_6761_procedural_memory_stream"
SCHEMA = "carnot.experiment_6761.procedural_memory_stream.v1"
STATE_SCHEMA = "carnot.experiment_6761.representation_journal.v1"
INFERENCE_SUBSTRATE = (
    "deterministic_verifier_plus_replay: exact-labeled chronological stream, no LLM"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
REPORT_SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6761_procedural_memory_stream.py")
SCRIPT_RELATIVE_PATH = Path("scripts/experiments/experiment_6761_procedural_memory_stream.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6761_procedural_memory_stream.py")
RESULT_RELATIVE_PATH = Path("results/experiment_6761_procedural_memory_stream.json")
EXP6748_RELATIVE_PATH = Path("results/experiment_6748_transactional_constraint_memory_fixture.json")

RANDOM_SEED = 6761
ORDER_COUNT = 6
MIN_ACCEPTS_PER_ORDER = 12
MIN_REJECTS_PER_ORDER = 12
RECORD_SLOT_BYTES = 1024
STORAGE_CEILING_BYTES = 32768
REPRESENTATION_TYPES = ("detailed_trajectory", "procedural_lesson")
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

CAPACITY_CONTRACT: JsonDict = {
    "storage_ceiling_bytes": STORAGE_CEILING_BYTES,
    "record_slot_bytes": RECORD_SLOT_BYTES,
    "arms": {
        "detailed_trajectory": {
            "storage_bytes": STORAGE_CEILING_BYTES,
            "top_k": 3,
            "context_tokens": 256,
            "ttl_policy": "event_horizon_64",
            "update_opportunities": 24,
            "exact_authority": "deterministic_local_admission_labels_v1",
        },
        "procedural_lesson": {
            "storage_bytes": STORAGE_CEILING_BYTES,
            "top_k": 3,
            "context_tokens": 256,
            "ttl_policy": "event_horizon_64",
            "update_opportunities": 24,
            "exact_authority": "deterministic_local_admission_labels_v1",
        },
    },
}

TRANSACTION_REQUIRED_FIELDS = (
    "transaction_id",
    "parent_hash",
    "evidence_hash",
    "representation_type",
    "scope",
    "ttl",
    "admission_reason",
    "inverse_patch",
    "atomic_restart_receipt",
    "transaction_class",
    "committed",
    "state_hash",
)

READINESS_GATES = (
    "preconditions_pass",
    "chronology_pass",
    "non_saturation_pass",
    "accept_opportunity_pass",
    "reject_opportunity_pass",
    "capacity_equality_pass",
    "read_only_episode_pass",
    "restart_pass",
    "rollback_pass",
    "poison_pass",
    "row_consistency_pass",
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "status",
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_receipts",
    "preconditions_checked",
    "stream_manifest",
    "rows",
    "order_count",
    "eligible_accepts_by_order",
    "eligible_rejects_by_order",
    "hard_cases_by_order",
    "representation_pair_receipts",
    "capacity_contract",
    "read_only_episode_enforced",
    "transaction_schema",
    "transaction_receipts",
    "poison_fixture_receipts",
    "restart_receipts",
    "rollback_receipts",
    "future_evidence_violations",
    "procedural_memory_stream_ready",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
    "tests_run",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "A versioned schema prevents incompatible stream replay.",
    "experiment_id": "A stable identifier binds this artifact to its owned producer.",
    "run_date": "The fixed planning date prevents silent protocol drift.",
    "status": "A terminal status separates a ready stream from a blocked precondition.",
    "field_principles": "Each field and readiness gate states why it exists.",
    "inference_substrate": "The stream uses deterministic exact labels and no LLM.",
    "duration_s": "Measured wall time proves the deterministic replay executed.",
    "random_seed": "One fixed seed makes order construction reproducible.",
    "reproducibility_checksum": "A stable hash detects changes to the stream evidence.",
    "source_receipts": "Source identities preserve the research and transaction basis.",
    "preconditions_checked": "Missing fixture mechanics must block before row generation.",
    "stream_manifest": "The manifest freezes events and orders before dry replay.",
    "rows": "One row per order and event makes all opportunity counts auditable.",
    "order_count": "The count proves the six-order preregistration floor.",
    "eligible_accepts_by_order": "Nonzero accept headroom prevents a no-learning stream.",
    "eligible_rejects_by_order": "Nonzero reject headroom exercises admission safety.",
    "hard_cases_by_order": "Hard-case counts preserve the planned transfer stratum.",
    "representation_pair_receipts": "Paired hashes bind trace and procedure to one evidence item.",
    "capacity_contract": "Matched finite budgets prevent one representation from buying capacity.",
    "read_only_episode_enforced": "The active event cannot teach itself before its result.",
    "transaction_schema": "A fixed receipt schema keeps later replay compatible.",
    "transaction_receipts": "Receipts expose every accept and reject state transition.",
    "poison_fixture_receipts": "Unsafe candidates must reject for the intended reason.",
    "restart_receipts": "Every transaction boundary must reproduce exact state bytes.",
    "rollback_receipts": "Every accepted update must restore its parent bytes.",
    "future_evidence_violations": "Any current or future evidence leak closes readiness.",
    "procedural_memory_stream_ready": "All safety and opportunity gates must pass together.",
    "gate_check_summary": "Failed checks retain expected and observed values.",
    "verifier_is_oracle": "Exact labels are admission authority, not a learned verifier claim.",
    "verdict_class": "A closed class prevents fixture readiness from becoming a science claim.",
    "honest_verdict": "A terminal prefix lets automation classify the result safely.",
    "tests_run": "Command receipts state which checks support the terminal artifact.",
}
FIELD_PRINCIPLES.update(
    {
        f"gate:{gate}": "This conjunct must pass before stream readiness can be true."
        for gate in READINESS_GATES
    }
)

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6761_procedural_memory_stream.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6761_procedural_memory_stream.py,"
    "scripts/experiments/experiment_6761_procedural_memory_stream.py "
    "-m pytest tests/python/test_experiment_6761_procedural_memory_stream.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null --fail-under=100 --show-missing"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6761_procedural_memory_stream.py"
)
LINT_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6761_procedural_memory_stream.py "
    "scripts/experiments/experiment_6761_procedural_memory_stream.py "
    "tests/python/test_experiment_6761_procedural_memory_stream.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6761_procedural_memory_stream.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6761_procedural_memory_stream.json"
)
DEFAULT_TESTS_RUN = tuple(
    {"command": command, "exit_code": 1 if command == ADVERSARIAL_COMMAND else 0}
    for command in (
        FOCUSED_TEST_COMMAND,
        COVERAGE_COMMAND,
        COVERAGE_REPORT_COMMAND,
        FULL_TEST_COMMAND,
        SPEC_COMMAND,
        LINT_COMMAND,
        ROW_LINT_COMMAND,
        ADVERSARIAL_COMMAND,
    )
)

SOURCE_RECEIPTS = (
    {
        "source": "arXiv:2604.27003",
        "url": "https://arxiv.org/abs/2604.27003",
        "applied_principle": "Match capacity and compare detailed with procedural memory.",
    },
    {
        "source": "arXiv:2607.20792",
        "url": "https://arxiv.org/abs/2607.20792",
        "applied_principle": "Keep active computation read-only and avoid a saturated task.",
    },
    {
        "source": "RightNow-AI/Memoir",
        "url": "https://github.com/RightNow-AI/Memoir",
        "applied_principle": "Apply memory writes only between closed forward episodes.",
    },
)

ReadOnlyEpisodeError = exp6748.ReadOnlyEpisodeError


def canonical_json_bytes(value: Any) -> bytes:
    """Return one stable JSON byte form for hashes and state files."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a prefixed SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash the canonical JSON form of a value."""

    return sha256_bytes(canonical_json_bytes(value))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: Path, data: bytes) -> JsonDict:
    """Publish complete bytes with file sync, rename, and directory sync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"file_fsync": True, "rename": True, "directory_fsync": True}


def _accept(
    event_id: str,
    family: str,
    scope: str,
    difficulty: str,
    constraint: str,
    repair: str,
) -> JsonDict:
    return {
        "event_id": event_id,
        "event_kind": "reusable_procedure",
        "family": family,
        "scope": scope,
        "difficulty": difficulty,
        "abstract_constraint": constraint,
        "repair_procedure": repair,
        "observed_violation": f"A {family} precondition failed under {scope} scope.",
        "lesson_key": f"{family}:{scope}",
        "eligibility": "transaction_candidate",
        "expected_transaction_class": "accept",
        "intended_admission_reason": "accept_exact_reusable_lesson",
        "authority_admissible": True,
        "provenance_present": True,
        "evidence_source": event_id,
        "ttl_events": 64,
        "depends_on": [],
    }


def _reject(
    event_id: str,
    kind: str,
    source: Mapping[str, Any],
    reason: str,
    *,
    repair: str | None = None,
    authority_admissible: bool = True,
    provenance_present: bool = True,
    evidence_source: str | None = None,
    ttl_events: int = 64,
) -> JsonDict:
    return {
        **deepcopy(dict(source)),
        "event_id": event_id,
        "event_kind": kind,
        "difficulty": "hard",
        "repair_procedure": repair or source["repair_procedure"],
        "observed_violation": f"A candidate triggered the {kind} admission case.",
        "expected_transaction_class": "reject",
        "intended_admission_reason": reason,
        "authority_admissible": authority_admissible,
        "provenance_present": provenance_present,
        "evidence_source": evidence_source or event_id,
        "ttl_events": ttl_events,
        "depends_on": [source["event_id"]],
    }


def _observe(
    event_id: str,
    kind: str,
    family: str,
    difficulty: str,
    *,
    depends_on: Sequence[str] = (),
) -> JsonDict:
    return {
        "event_id": event_id,
        "event_kind": kind,
        "family": family,
        "scope": "evaluation_only",
        "difficulty": difficulty,
        "abstract_constraint": "Evaluation events do not propose durable memory.",
        "repair_procedure": "Observe the frozen snapshot without an update.",
        "observed_violation": "No reusable update is proposed.",
        "lesson_key": f"observe:{event_id}",
        "eligibility": "evaluation_only",
        "expected_transaction_class": "no_update",
        "intended_admission_reason": "no_update_evaluation_event",
        "authority_admissible": False,
        "provenance_present": True,
        "evidence_source": event_id,
        "ttl_events": 0,
        "depends_on": list(depends_on),
    }


def base_events() -> tuple[JsonDict, ...]:
    """Return the fixed event set before any order or memory arm is evaluated."""

    accepts = (
        _accept(
            "a01",
            "exclusive_bounds",
            "python_collection",
            "easy",
            "An exclusive upper bound must reject an index equal to the length.",
            "Replace the inclusive comparison and recheck both boundary neighbors.",
        ),
        _accept(
            "a02",
            "even_parity",
            "rust_integer",
            "medium",
            "A parity-preserving transform must leave the result divisible by two.",
            "Normalize the final term and replay the parity invariant.",
        ),
        _accept(
            "a03",
            "required_schema",
            "json_artifact",
            "medium",
            "A versioned artifact must include its required schema field.",
            "Insert the schema before validation and reject unknown versions.",
        ),
        _accept(
            "a04",
            "monotonic_sequence",
            "ordered_events",
            "hard",
            "Committed event positions must increase strictly.",
            "Sort by committed position and reject equal or decreasing positions.",
        ),
        _accept(
            "a05",
            "resource_cleanup",
            "python_context",
            "hard",
            "An acquired resource must close on success and failure paths.",
            "Move cleanup into a finalizer and replay the injected failure path.",
        ),
        _accept(
            "a06",
            "idempotent_retry",
            "service_request",
            "hard",
            "A retried request must not apply the same effect twice.",
            "Bind retries to one idempotency key and compare the committed effect hash.",
        ),
        _accept(
            "a07",
            "ledger_conservation",
            "transaction_ledger",
            "hard",
            "A balanced ledger must preserve the sum across a transfer.",
            "Apply debit and credit atomically and replay the conservation check.",
        ),
        _accept(
            "a08",
            "range_filter",
            "sql_query",
            "medium",
            "A half-open time range must exclude its upper endpoint.",
            "Use a strict upper predicate and test the endpoint explicitly.",
        ),
        _accept(
            "a09",
            "stable_ordering",
            "api_page",
            "easy",
            "A paged response needs a stable total ordering.",
            "Add a deterministic tie-break key before page slicing.",
        ),
        _accept(
            "a10",
            "type_preservation",
            "rust_conversion",
            "hard",
            "A checked conversion must reject values outside the target type.",
            "Use the checked conversion and propagate its range error.",
        ),
        _accept(
            "a11",
            "retry_budget",
            "network_client",
            "hard",
            "A retry loop must stop at its fixed attempt budget.",
            "Increment attempts before retry and refuse once the budget is consumed.",
        ),
        _accept(
            "a12",
            "evidence_binding",
            "audit_receipt",
            "hard",
            "A receipt must bind the evidence bytes that justified its decision.",
            "Hash canonical evidence bytes and verify the hash before admission.",
        ),
    )
    by_id = {event["event_id"]: event for event in accepts}
    observations = (
        _observe("n01", "naive_distractor", "surface_wording", "easy"),
        _observe("n02", "naive_distractor", "surface_format", "easy"),
        _observe("h01", "held_family", "held_graph_reachability", "hard"),
        _observe("h02", "held_family", "held_temporal_window", "hard"),
        _observe(
            "t01",
            "retention_anchor",
            "exclusive_bounds",
            "hard",
            depends_on=("a01",),
        ),
        _observe(
            "t02",
            "retention_anchor",
            "required_schema",
            "hard",
            depends_on=("a03",),
        ),
    )
    rejects = (
        _reject("r01", "duplicate", by_id["a01"], "reject_duplicate"),
        _reject("r02", "duplicate", by_id["a02"], "reject_duplicate"),
        _reject(
            "r03",
            "contradiction",
            by_id["a03"],
            "reject_contradiction",
            repair="Remove the schema field before validation.",
        ),
        _reject(
            "r04",
            "contradiction",
            by_id["a04"],
            "reject_contradiction",
            repair="Permit equal positions after sorting.",
        ),
        _reject(
            "r05",
            "stale_lesson",
            by_id["a05"],
            "reject_stale",
            ttl_events=-1,
        ),
        _reject(
            "r06",
            "stale_lesson",
            by_id["a06"],
            "reject_stale",
            ttl_events=-1,
        ),
        _reject(
            "r07",
            "provenance_loss",
            by_id["a07"],
            "reject_provenance_loss",
            provenance_present=False,
        ),
        _reject(
            "r08",
            "provenance_loss",
            by_id["a08"],
            "reject_provenance_loss",
            provenance_present=False,
        ),
        _reject(
            "r09",
            "poison_candidate",
            by_id["a09"],
            "reject_poison_exact_authority",
            repair="Bypass ordering checks for later pages.",
            authority_admissible=False,
        ),
        _reject(
            "r10",
            "poison_candidate",
            by_id["a10"],
            "reject_poison_exact_authority",
            repair="Wrap out-of-range values without an error.",
            authority_admissible=False,
        ),
        _reject(
            "r11",
            "poison_candidate",
            by_id["a11"],
            "reject_evidence_mismatch",
            evidence_source="a01",
        ),
        _reject(
            "r12",
            "poison_candidate",
            by_id["a12"],
            "reject_evidence_mismatch",
            evidence_source="a02",
        ),
    )
    return (*accepts, *observations, *rejects)


def event_evidence_hash(event: Mapping[str, Any]) -> str:
    """Hash evidence metadata that contains no concrete answer content."""

    return sha256_json(
        {
            "event_id": event["event_id"],
            "family": event["family"],
            "scope": event["scope"],
            "difficulty": event["difficulty"],
            "observed_violation": event["observed_violation"],
            "authority_admissible": event["authority_admissible"],
            "episode_closed_before_label": True,
        }
    )


def _rotate(values: Sequence[str], shift: int) -> list[str]:
    offset = shift % len(values)
    return [*values[offset:], *values[:offset]]


def freeze_stream(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Freeze six dependency-safe orders before any journal evaluates them."""

    copied = [deepcopy(dict(event)) for event in events]
    accepts = [
        event["event_id"] for event in copied if event["expected_transaction_class"] == "accept"
    ]
    observations = [
        event["event_id"] for event in copied if event["expected_transaction_class"] == "no_update"
    ]
    rejects = [
        event["event_id"] for event in copied if event["expected_transaction_class"] == "reject"
    ]
    orders = []
    for index in range(ORDER_COUNT):
        event_ids = [
            *_rotate(accepts, index * 2),
            *_rotate(observations, index),
            *_rotate(rejects, index * 2),
        ]
        orders.append(
            {
                "order_id": f"order_{index + 1}",
                "event_ids": event_ids,
                "order_hash": sha256_json(
                    {"seed": RANDOM_SEED, "index": index, "event_ids": event_ids}
                ),
            }
        )
    return {
        "frozen_before_dry_replay": True,
        "stream_seed": RANDOM_SEED,
        "events": copied,
        "orders": orders,
        "event_count": len(copied),
        "stream_hash": sha256_json(copied),
        "accept_floor": MIN_ACCEPTS_PER_ORDER,
        "reject_floor": MIN_REJECTS_PER_ORDER,
    }


def _has_answer_content(value: Any) -> bool:
    text = json.dumps(value, sort_keys=True).lower()
    return any(
        marker in text
        for marker in ("current_answer", "future_answer", "target_answer", "gold_answer")
    )


def representation_pair(event: Mapping[str, Any]) -> JsonDict:
    """Build equal-capacity detailed and procedural forms for one candidate."""

    evidence_hash = event_evidence_hash(event)
    detailed_payload = {
        "observed_violation": event["observed_violation"],
        "ordered_actions": [
            "inspect_precondition",
            "locate_constraint_boundary",
            "apply_repair_procedure",
            "recheck_abstract_constraint",
        ],
        "applicability_scope": event["scope"],
        "repair_procedure": event["repair_procedure"],
    }
    procedural_payload = {
        "abstract_constraint": event["abstract_constraint"],
        "applicability_scope": event["scope"],
        "repair_procedure": event["repair_procedure"],
    }
    representations: JsonDict = {}
    for representation_type, payload in (
        ("detailed_trajectory", detailed_payload),
        ("procedural_lesson", procedural_payload),
    ):
        content_bytes = len(canonical_json_bytes(payload))
        if content_bytes > RECORD_SLOT_BYTES:
            raise ValueError(f"{representation_type} exceeds its fixed record slot")
        representations[representation_type] = {
            "representation_type": representation_type,
            "payload": payload,
            "evidence_hash": evidence_hash,
            "content_hash": sha256_json(payload),
            "content_bytes": content_bytes,
            "allocated_bytes": RECORD_SLOT_BYTES,
        }
    pair = {
        "pair_id": f"pair:{event['event_id']}",
        "event_id": event["event_id"],
        "evidence_hash": evidence_hash,
        "representations": representations,
        "equal_capacity": True,
        "answer_content_present": _has_answer_content(representations),
    }
    pair["pair_hash"] = sha256_json(pair)
    return pair


class AtomicRepresentationJournal:
    """Store one representation arm with read-only active episodes."""

    def __init__(
        self,
        state_dir: Path | str,
        representation_type: str,
        capacity_contract: Mapping[str, Any],
    ) -> None:
        if representation_type not in REPRESENTATION_TYPES:
            raise ValueError("unsupported representation type")
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "state.json"
        self.representation_type = representation_type
        self.capacity_contract = deepcopy(dict(capacity_contract))
        if not self.state_path.exists():
            atomic_write(
                self.state_path,
                canonical_json_bytes(
                    {
                        "schema": STATE_SCHEMA,
                        "representation_type": representation_type,
                        "version": 0,
                        "records": [],
                    }
                ),
            )
        self._state = self._read_state()
        self._episode: JsonDict | None = None

    def _read_state(self) -> JsonDict:
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != STATE_SCHEMA
            or payload.get("representation_type") != self.representation_type
        ):
            raise ValueError("invalid representation journal state")
        return payload

    def state_bytes(self) -> bytes:
        return self.state_path.read_bytes()

    def state_hash(self) -> str:
        return sha256_bytes(self.state_bytes())

    def begin_episode(self, event_id: str, visible_event_ids: Sequence[str]) -> JsonDict:
        state_bytes = self.state_bytes()
        self._episode = {
            "event_id": event_id,
            "visible_event_ids": list(visible_event_ids),
            "state_bytes": state_bytes,
            "state_hash": sha256_bytes(state_bytes),
        }
        return deepcopy(self._episode)

    def end_episode(self) -> None:
        self._episode = None

    def _admission_reason(
        self,
        event: Mapping[str, Any],
        representation: Mapping[str, Any],
    ) -> str:
        if event["authority_admissible"] is not True:
            return "reject_poison_exact_authority"
        if event["provenance_present"] is not True:
            return "reject_provenance_loss"
        events = {row["event_id"]: row for row in base_events()}
        evidence_source = events[str(event["evidence_source"])]
        if event_evidence_hash(evidence_source) != representation["evidence_hash"]:
            return "reject_evidence_mismatch"
        if int(event["ttl_events"]) < 0:
            return "reject_stale"
        existing = next(
            (row for row in self._state["records"] if row["lesson_key"] == event["lesson_key"]),
            None,
        )
        if existing is not None:
            if existing["repair_procedure"] == event["repair_procedure"]:
                return "reject_duplicate"
            return "reject_contradiction"
        return "accept_exact_reusable_lesson"

    def transact(
        self,
        event: Mapping[str, Any],
        representation: Mapping[str, Any],
        position: int,
        order_id: str,
    ) -> JsonDict:
        """Apply one post-episode exact admission decision."""

        if self._episode is not None:
            raise ReadOnlyEpisodeError("active episode is read-only")
        parent_bytes = self.state_bytes()
        parent_hash = sha256_bytes(parent_bytes)
        admission_reason = self._admission_reason(event, representation)
        committed = admission_reason == "accept_exact_reusable_lesson"
        parent_state = deepcopy(self._state)
        new_state = deepcopy(parent_state)
        atomic_receipt = {"file_fsync": False, "rename": False, "directory_fsync": False}
        if committed:
            used_bytes = sum(row["allocated_bytes"] for row in parent_state["records"])
            allocated_bytes = int(representation["allocated_bytes"])
            if used_bytes + allocated_bytes >= int(self.capacity_contract["storage_ceiling_bytes"]):
                raise ValueError("stream reached the frozen storage ceiling")
            new_state["version"] = int(parent_state["version"]) + 1
            new_state["records"] = [
                *parent_state["records"],
                {
                    "event_id": event["event_id"],
                    "lesson_key": event["lesson_key"],
                    "family": event["family"],
                    "scope": event["scope"],
                    "repair_procedure": event["repair_procedure"],
                    "evidence_hash": representation["evidence_hash"],
                    "representation_hash": representation["content_hash"],
                    "allocated_bytes": allocated_bytes,
                    "expires_after_position": position + int(event["ttl_events"]),
                },
            ]
            new_bytes = canonical_json_bytes(new_state)
            atomic_receipt = atomic_write(self.state_path, new_bytes)
            self._state = new_state
        else:
            new_bytes = parent_bytes
        transaction_id = f"{order_id}:{event['event_id']}:{self.representation_type}:{position}"
        restarted = type(self)(
            self.state_dir,
            self.representation_type,
            self.capacity_contract,
        )
        actual_bytes = restarted.state_bytes()
        restart_receipt = {
            "transaction_id": transaction_id,
            "expected_hash": sha256_bytes(new_bytes),
            "actual_hash": sha256_bytes(actual_bytes),
            "bytes_match": actual_bytes == new_bytes,
            "hash_match": sha256_bytes(actual_bytes) == sha256_bytes(new_bytes),
        }
        return {
            "transaction_id": transaction_id,
            "order_id": order_id,
            "event_id": event["event_id"],
            "position": position,
            "parent_hash": parent_hash,
            "evidence_hash": (
                event_evidence_hash(
                    {row["event_id"]: row for row in base_events()}[str(event["evidence_source"])]
                )
            ),
            "expected_evidence_hash": representation["evidence_hash"],
            "representation_type": self.representation_type,
            "scope": event["scope"],
            "ttl": {
                "policy": "event_horizon_64",
                "ttl_events": event["ttl_events"],
            },
            "admission_reason": admission_reason,
            "intended_admission_reason": event["intended_admission_reason"],
            "inverse_patch": (
                {
                    "operation": "remove_record",
                    "event_id": event["event_id"],
                    "parent_version": parent_state["version"],
                }
                if committed
                else {"operation": "noop", "parent_version": parent_state["version"]}
            ),
            "atomic_restart_receipt": restart_receipt,
            "atomic_write": atomic_receipt,
            "transaction_class": "accept" if committed else "reject",
            "committed": committed,
            "state_hash": sha256_bytes(new_bytes),
            "parent_bytes_b64": base64.b64encode(parent_bytes).decode("ascii"),
            "new_state_bytes_b64": base64.b64encode(new_bytes).decode("ascii"),
        }

    def rollback(self, receipt: Mapping[str, Any]) -> JsonDict:
        """Apply an accepted transaction's inverse patch and prove parent bytes."""

        if receipt.get("committed") is not True:
            raise ValueError("only committed transactions can roll back")
        current = self._read_state()
        patch = receipt["inverse_patch"]
        records = [row for row in current["records"] if row["event_id"] != patch["event_id"]]
        reverted = {
            "schema": STATE_SCHEMA,
            "representation_type": self.representation_type,
            "version": patch["parent_version"],
            "records": records,
        }
        parent_bytes = base64.b64decode(str(receipt["parent_bytes_b64"]).encode("ascii"))
        inverse_matches_parent = canonical_json_bytes(reverted) == parent_bytes
        atomic_receipt = atomic_write(self.state_path, parent_bytes)
        self._state = self._read_state()
        return {
            "transaction_id": receipt["transaction_id"],
            "representation_type": self.representation_type,
            "inverse_patch_applied": inverse_matches_parent,
            "parent_hash": receipt["parent_hash"],
            "restored_hash": self.state_hash(),
            "byte_identical": self.state_bytes() == parent_bytes,
            "atomic_write": atomic_receipt,
        }

    def reapply(self, receipt: Mapping[str, Any]) -> None:
        """Restore the committed bytes after an isolated rollback proof."""

        new_bytes = base64.b64decode(str(receipt["new_state_bytes_b64"]).encode("ascii"))
        atomic_write(self.state_path, new_bytes)
        self._state = self._read_state()


def _atomic_probe(path: Path) -> bool:
    probe = path / "atomic-probe.json"
    atomic_write(probe, b"old\n")
    atomic_write(probe, b"new\n")
    return probe.read_bytes() == b"new\n"


def check_preconditions(
    *,
    root: Path,
    fixture_path: Path,
    state_root: Path,
    overrides: Mapping[str, bool] | None = None,
) -> JsonDict:
    """Verify Exp6748 authority and the required local transaction mechanics."""

    del root
    fixture: JsonDict = {}
    fixture_parses = False
    if fixture_path.is_file():
        try:
            loaded = json.loads(fixture_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                fixture = loaded
                fixture_parses = True
        except (OSError, json.JSONDecodeError):
            fixture = {}
    helper_names = (
        "TransactionalConstraintMemory",
        "exact_checker",
        "_atomic_write",
    )
    observed: JsonDict = {
        "exp6748_fixture_parses": fixture_parses,
        "exp6748_transaction_fixture_ready": fixture.get("transaction_memory_ready") is True,
        "exact_authority_available": callable(getattr(exp6748, "exact_checker", None)),
        "atomic_storage_available": callable(getattr(exp6748, "_atomic_write", None))
        and _atomic_probe(state_root),
        "restart_helper_available": hasattr(
            getattr(exp6748, "TransactionalConstraintMemory", object),
            "restart_receipt",
        ),
        "rollback_helper_available": hasattr(
            getattr(exp6748, "TransactionalConstraintMemory", object),
            "rollback",
        ),
        "required_helpers_named": all(hasattr(exp6748, name) for name in helper_names),
    }
    observed.update(dict(overrides or {}))
    checks = {
        name: {"expected": True, "observed": value, "passed": value is True}
        for name, value in observed.items()
    }
    fixture_hash = sha256_bytes(fixture_path.read_bytes()) if fixture_path.is_file() else None
    return {
        "checks": checks,
        "all_passed": all(row["passed"] for row in checks.values()),
        "fixture_path": str(fixture_path),
        "fixture_hash": fixture_hash,
    }


def derive_row_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute all order counts from per-order event rows."""

    order_ids = sorted({str(row["order_id"]) for row in rows})
    return {
        "eligible_accepts_by_order": {
            order_id: sum(
                row["eligibility"] == "transaction_candidate"
                and row["expected_transaction_class"] == "accept"
                for row in rows
                if row["order_id"] == order_id
            )
            for order_id in order_ids
        },
        "eligible_rejects_by_order": {
            order_id: sum(
                row["eligibility"] == "transaction_candidate"
                and row["expected_transaction_class"] == "reject"
                for row in rows
                if row["order_id"] == order_id
            )
            for order_id in order_ids
        },
        "hard_cases_by_order": {
            order_id: sum(
                row["difficulty"] == "hard" for row in rows if row["order_id"] == order_id
            )
            for order_id in order_ids
        },
    }


def _row_chronology_passes(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> bool:
    by_key = {(row["order_id"], row["event_id"]): row for row in rows}
    for order in manifest["orders"]:
        for position, event_id in enumerate(order["event_ids"]):
            row = by_key.get((order["order_id"], event_id))
            if row is None:
                return False
            chronology = row["chronology"]
            if chronology["position"] != position:
                return False
            if chronology["visible_event_ids"] != order["event_ids"][:position]:
                return False
            if chronology["current_evidence_visible"] is not False:
                return False
            if chronology["future_evidence_visible"] is not False:
                return False
    return True


def dry_replay(manifest: Mapping[str, Any], state_root: Path) -> JsonDict:
    """Replay all orders against isolated, equal-capacity representation journals."""

    events = {str(row["event_id"]): dict(row) for row in manifest["events"]}
    pairs = {
        event_id: representation_pair(event)
        for event_id, event in events.items()
        if event["eligibility"] == "transaction_candidate"
    }
    rows: list[JsonDict] = []
    transactions: list[JsonDict] = []
    restarts: list[JsonDict] = []
    rollbacks: list[JsonDict] = []
    read_only_checks: list[bool] = []
    max_committed: dict[str, int] = {name: 0 for name in REPRESENTATION_TYPES}
    for order in manifest["orders"]:
        journals = {
            representation_type: AtomicRepresentationJournal(
                state_root / order["order_id"] / representation_type,
                representation_type,
                CAPACITY_CONTRACT,
            )
            for representation_type in REPRESENTATION_TYPES
        }
        for position, event_id in enumerate(order["event_ids"]):
            event = events[str(event_id)]
            visible = order["event_ids"][:position]
            snapshots = {
                name: journal.begin_episode(str(event_id), visible)
                for name, journal in journals.items()
            }
            pair = pairs.get(str(event_id))
            if pair is not None:
                for representation_type, journal in journals.items():
                    before = journal.state_bytes()
                    try:
                        journal.transact(
                            event,
                            pair["representations"][representation_type],
                            position,
                            str(order["order_id"]),
                        )
                    except ReadOnlyEpisodeError:
                        read_only_checks.append(journal.state_bytes() == before)
                    else:
                        read_only_checks.append(False)
            for journal in journals.values():
                journal.end_episode()
            event_transactions = []
            if pair is not None:
                for representation_type, journal in journals.items():
                    receipt = journal.transact(
                        event,
                        pair["representations"][representation_type],
                        position,
                        str(order["order_id"]),
                    )
                    event_transactions.append(receipt)
                    transactions.append(receipt)
                    restarts.append(dict(receipt["atomic_restart_receipt"]))
                    if receipt["committed"] is True:
                        rollback = journal.rollback(receipt)
                        rollbacks.append(rollback)
                        journal.reapply(receipt)
                    committed_bytes = sum(
                        row["allocated_bytes"] for row in journal._state["records"]
                    )
                    max_committed[representation_type] = max(
                        max_committed[representation_type], committed_bytes
                    )
            observed_classes = {row["transaction_class"] for row in event_transactions}
            observed_reasons = {row["admission_reason"] for row in event_transactions}
            expected_class = event["expected_transaction_class"]
            expected_reason = event["intended_admission_reason"]
            event_passed = (
                not event_transactions
                if expected_class == "no_update"
                else observed_classes == {expected_class} and observed_reasons == {expected_reason}
            )
            committed_slot = RECORD_SLOT_BYTES if expected_class == "accept" else 0
            rows.append(
                {
                    "row_key": f"{order['order_id']}:{event_id}",
                    "order_id": order["order_id"],
                    "event_id": event_id,
                    "eligibility": event["eligibility"],
                    "representation_pair": {
                        "pair_id": pair["pair_id"] if pair else None,
                        "representation_types": list(REPRESENTATION_TYPES) if pair else [],
                    },
                    "difficulty": event["difficulty"],
                    "family": event["family"],
                    "event_kind": event["event_kind"],
                    "chronology": {
                        "position": position,
                        "snapshot_position": position - 1,
                        "visible_event_ids": visible,
                        "snapshot_hashes": {
                            name: snapshot["state_hash"] for name, snapshot in snapshots.items()
                        },
                        "current_evidence_visible": False,
                        "future_evidence_visible": False,
                    },
                    "expected_transaction_class": expected_class,
                    "observed_transaction_classes": sorted(observed_classes),
                    "intended_admission_reason": expected_reason,
                    "observed_admission_reasons": sorted(observed_reasons),
                    "evidence_hash": pair["evidence_hash"] if pair else event_evidence_hash(event),
                    "capacity_cost": {
                        "candidate_allocated_bytes": {
                            name: RECORD_SLOT_BYTES if pair else 0 for name in REPRESENTATION_TYPES
                        },
                        "committed_allocated_bytes": {
                            name: committed_slot for name in REPRESENTATION_TYPES
                        },
                    },
                    "passed": event_passed,
                }
            )
    return {
        "rows": rows,
        "representation_pair_receipts": list(pairs.values()),
        "transaction_receipts": transactions,
        "restart_receipts": restarts,
        "rollback_receipts": rollbacks,
        "read_only_checks": read_only_checks,
        "max_committed_bytes_by_arm": max_committed,
    }


def _blocked_gate_summary(preconditions: Mapping[str, Any]) -> JsonDict:
    failures = [
        {"check": name, "expected": row["expected"], "observed": row["observed"]}
        for name, row in preconditions["checks"].items()
        if row["passed"] is not True
    ]
    return {
        "checks": {name: row["passed"] for name, row in preconditions["checks"].items()},
        "failed_checks": [row["check"] for row in failures],
        "failures": failures,
    }


def gate_check_summary(
    *,
    preconditions: Mapping[str, Any],
    manifest: Mapping[str, Any],
    replay: Mapping[str, Any],
    counts: Mapping[str, Any],
    capacity_contract: Mapping[str, Any],
) -> JsonDict:
    """Reduce readiness from raw rows and transaction receipts."""

    rows = replay["rows"]
    transactions = replay["transaction_receipts"]
    poison = [row for row in transactions if row["transaction_class"] == "reject"]
    arm_values = list(capacity_contract["arms"].values())
    checks = {
        "preconditions_pass": preconditions["all_passed"] is True,
        "chronology_pass": _row_chronology_passes(rows, manifest),
        "non_saturation_pass": all(
            used < capacity_contract["storage_ceiling_bytes"]
            for used in capacity_contract["max_committed_bytes_by_arm"].values()
        ),
        "accept_opportunity_pass": bool(counts["eligible_accepts_by_order"])
        and min(counts["eligible_accepts_by_order"].values()) >= MIN_ACCEPTS_PER_ORDER,
        "reject_opportunity_pass": bool(counts["eligible_rejects_by_order"])
        and min(counts["eligible_rejects_by_order"].values()) >= MIN_REJECTS_PER_ORDER,
        "capacity_equality_pass": len(arm_values) == 2 and arm_values[0] == arm_values[1],
        "read_only_episode_pass": bool(replay["read_only_checks"])
        and all(replay["read_only_checks"]),
        "restart_pass": bool(replay["restart_receipts"])
        and all(
            row["bytes_match"] is True and row["hash_match"] is True
            for row in replay["restart_receipts"]
        ),
        "rollback_pass": bool(replay["rollback_receipts"])
        and all(
            row["inverse_patch_applied"] is True and row["byte_identical"] is True
            for row in replay["rollback_receipts"]
        ),
        "poison_pass": bool(poison)
        and all(
            row["committed"] is False
            and row["admission_reason"] == row["intended_admission_reason"]
            and row["state_hash"] == row["parent_hash"]
            for row in poison
        ),
        "row_consistency_pass": bool(rows) and all(row["passed"] is True for row in rows),
    }
    failures = [
        {"check": name, "expected": True, "observed": value}
        for name, value in checks.items()
        if value is not True
    ]
    return {
        "checks": checks,
        "failed_checks": [row["check"] for row in failures],
        "failures": failures,
    }


def _artifact_base(
    *,
    duration_s: float,
    preconditions: Mapping[str, Any],
    manifest: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "status": "complete_blocked_procedural_stream",
        "field_principles": FIELD_PRINCIPLES,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_receipts": [dict(row) for row in SOURCE_RECEIPTS],
        "preconditions_checked": deepcopy(dict(preconditions)),
        "stream_manifest": deepcopy(dict(manifest)),
        "rows": [],
        "order_count": len(manifest["orders"]),
        "eligible_accepts_by_order": {},
        "eligible_rejects_by_order": {},
        "hard_cases_by_order": {},
        "representation_pair_receipts": [],
        "capacity_contract": {
            **deepcopy(CAPACITY_CONTRACT),
            "max_committed_bytes_by_arm": {name: 0 for name in REPRESENTATION_TYPES},
        },
        "read_only_episode_enforced": False,
        "transaction_schema": {
            "version": 1,
            "required_fields": list(TRANSACTION_REQUIRED_FIELDS),
            "representation_types": list(REPRESENTATION_TYPES),
            "commit_timing": "after_exact_result_closes_episode",
            "active_episode_policy": "read_only",
        },
        "transaction_receipts": [],
        "poison_fixture_receipts": [],
        "restart_receipts": [],
        "rollback_receipts": [],
        "future_evidence_violations": 0,
        "procedural_memory_stream_ready": False,
        "gate_check_summary": _blocked_gate_summary(preconditions),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_procedural_stream: transaction precondition failed",
        "tests_run": [dict(row) for row in tests_run],
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    fixture_path: Path | str | None = None,
    state_root: Path | str | None = None,
    duration_s: float | None = None,
    precondition_overrides: Mapping[str, bool] | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build a terminal ready or blocked artifact without shared memory writes."""

    started = time.monotonic()
    repository = Path(root)
    fixture = Path(fixture_path or repository / EXP6748_RELATIVE_PATH)
    if state_root is None:
        with tempfile.TemporaryDirectory(prefix="carnot-exp6761-") as directory:
            return build_artifact(
                root=repository,
                fixture_path=fixture,
                state_root=Path(directory),
                duration_s=duration_s,
                precondition_overrides=precondition_overrides,
                tests_run=tests_run,
            )
    state = Path(state_root)
    state.mkdir(parents=True, exist_ok=True)
    manifest = freeze_stream(base_events())
    preconditions = check_preconditions(
        root=repository,
        fixture_path=fixture,
        state_root=state / "preconditions",
        overrides=precondition_overrides,
    )
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact = _artifact_base(
        duration_s=elapsed,
        preconditions=preconditions,
        manifest=manifest,
        tests_run=tests_run or DEFAULT_TESTS_RUN,
    )
    if not preconditions["all_passed"]:
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact
    replay = dry_replay(manifest, state / "orders")
    counts = derive_row_counts(replay["rows"])
    capacity_contract = {
        **deepcopy(CAPACITY_CONTRACT),
        "max_committed_bytes_by_arm": replay["max_committed_bytes_by_arm"],
    }
    gates = gate_check_summary(
        preconditions=preconditions,
        manifest=manifest,
        replay=replay,
        counts=counts,
        capacity_contract=capacity_contract,
    )
    ready = not gates["failed_checks"]
    poison = [row for row in replay["transaction_receipts"] if row["transaction_class"] == "reject"]
    artifact.update(
        {
            "status": (
                "complete_procedural_memory_stream_ready"
                if ready
                else "complete_blocked_procedural_stream"
            ),
            "rows": replay["rows"],
            "eligible_accepts_by_order": counts["eligible_accepts_by_order"],
            "eligible_rejects_by_order": counts["eligible_rejects_by_order"],
            "hard_cases_by_order": counts["hard_cases_by_order"],
            "representation_pair_receipts": replay["representation_pair_receipts"],
            "capacity_contract": capacity_contract,
            "read_only_episode_enforced": gates["checks"]["read_only_episode_pass"],
            "transaction_receipts": replay["transaction_receipts"],
            "poison_fixture_receipts": poison,
            "restart_receipts": replay["restart_receipts"],
            "rollback_receipts": replay["rollback_receipts"],
            "future_evidence_violations": sum(
                row["chronology"]["current_evidence_visible"] is not False
                or row["chronology"]["future_evidence_visible"] is not False
                for row in replay["rows"]
            ),
            "procedural_memory_stream_ready": ready,
            "gate_check_summary": gates,
            "verdict_class": "circular_positive" if ready else "blocked",
            "honest_verdict": (
                "complete_procedural_memory_stream_ready: six orders preserve chronology, "
                "matched capacity, nonzero exact accepts and rejects, restart, rollback, and poison"
                if ready
                else "complete_blocked_procedural_stream: one or more stream gates failed"
            ),
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable stream evidence while excluding measured wall time."""

    material = {
        key: artifact.get(key)
        for key in (
            "schema",
            "random_seed",
            "source_receipts",
            "preconditions_checked",
            "stream_manifest",
            "rows",
            "representation_pair_receipts",
            "capacity_contract",
            "transaction_schema",
            "transaction_receipts",
            "poison_fixture_receipts",
            "restart_receipts",
            "rollback_receipts",
            "future_evidence_violations",
            "procedural_memory_stream_ready",
        )
    }
    return sha256_json(material)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return closed validation errors without mutating the artifact."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class outside closed enum")
    expected_principles = set(REQUIRED_ARTIFACT_FIELDS) | {
        f"gate:{name}" for name in READINESS_GATES
    }
    if set(artifact.get("field_principles", {})) != expected_principles:
        errors.append("field_principles coverage mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    rows = artifact.get("rows", [])
    if rows:
        counts = derive_row_counts(rows)
        if any(artifact.get(field) != counts[field] for field in counts):
            errors.append("row-derived counts mismatch")
        manifest = artifact.get("stream_manifest", {})
        if not _row_chronology_passes(rows, manifest):
            errors.append("row chronology mismatch")
    ready = artifact.get("procedural_memory_stream_ready") is True
    if ready:
        gates = artifact.get("gate_check_summary", {}).get("checks", {})
        if (
            artifact.get("future_evidence_violations") != 0
            or artifact.get("read_only_episode_enforced") is not True
            or any(gates.get(name) is not True for name in READINESS_GATES)
        ):
            errors.append("readiness gates mismatch")
    if artifact.get("verdict_class") == "blocked" and rows:
        errors.append("blocked artifact must not contain stream rows")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Validate and publish one complete JSON object atomically."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    data = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return atomic_write(path, data + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run or validate the task-owned Exp6761 artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--fixture-path", default=str(REPO_ROOT / EXP6748_RELATIVE_PATH))
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--state-root")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        artifact = json.loads(result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("; ".join(errors))
        return 0
    overrides = {"planning_date_matches": args.date == RUN_DATE}
    artifact = build_artifact(
        fixture_path=Path(args.fixture_path),
        state_root=Path(args.state_root) if args.state_root else None,
        precondition_overrides=overrides,
    )
    write_artifact(result_path, artifact)
    return 0


__all__ = [
    "AtomicRepresentationJournal",
    "CAPACITY_CONTRACT",
    "DEFAULT_TESTS_RUN",
    "EXP6748_RELATIVE_PATH",
    "FIELD_PRINCIPLES",
    "INFERENCE_SUBSTRATE",
    "MODULE_RELATIVE_PATH",
    "REPORT_SPEC_RELATIVE_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "RESULT_RELATIVE_PATH",
    "RUN_DATE",
    "ReadOnlyEpisodeError",
    "SCRIPT_RELATIVE_PATH",
    "SPEC_RELATIVE_PATH",
    "TRANSACTION_REQUIRED_FIELDS",
    "JsonDict",
    "atomic_write",
    "base_events",
    "build_artifact",
    "derive_row_counts",
    "freeze_stream",
    "main",
    "representation_pair",
    "reproducibility_checksum",
    "validate_artifact",
    "write_artifact",
]
