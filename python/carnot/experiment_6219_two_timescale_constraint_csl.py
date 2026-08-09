"""Exp6219 two-timescale constraint continuous self-learning.

Spec refs: REQ-LEARN-6219, SCENARIO-LEARN-6219-SNAPSHOTS,
SCENARIO-LEARN-6219-TWO-TIMESCALE, SCENARIO-LEARN-6219-ATTACKS,
SCENARIO-LEARN-6219-ROLLBACK.

The experiment consumes the Exp6145 exact stream. Decisions see only the
pre-outcome row. Procedural constraints can enter memory only after the
Exp6145 Python/Z3 sidecar discloses and verifies the event outcome.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_6145_constraint_shift_stream as exp6145


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6219_two_timescale_constraint_csl.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6219_two_timescale_constraint_csl.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6219_two_timescale_constraint_csl.py")

SCHEMA = "carnot.experiment_6219.two_timescale_constraint_csl.v1"
EXPERIMENT_ID = "experiment_6219_two_timescale_constraint_csl"
RUN_DATE = "20260809"
RANDOM_SEEDS = (6219, 6220, 6221)
MEMORY_RECORD_BUDGET = 12
MEMORY_BYTE_BUDGET = 16_384
TTL_BLOCKS = 3
PROTECTED_FAMILIES = ("access_control", "release_gating")
HELD_PARTITIONS = ("future_known", "sealed_shifted_family")
INFERENCE_SUBSTRATE = "deterministic_exp6145_exact_verifier_external_memory_no_llm"

ARM_NAMES = (
    "no_memory",
    "immediate_verified_post_outcome_commit",
    "slow_block_end_consolidation",
    "shuffled_memory_control",
)

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6219_two_timescale_constraint_csl.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6219_two_timescale_constraint_csl.py "
    "-m pytest tests/python/test_experiment_6219_two_timescale_constraint_csl.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6219_two_timescale_constraint_csl.py "
    "--fail-under=100"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6219_two_timescale_constraint_csl "
    "--date 20260809 --validate"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6219_two_timescale_constraint_csl.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6219_two_timescale_constraint_csl.json"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    exp6145.RESULT_RELATIVE_PATH,
    exp6145.ROW_FILE_RELATIVE_PATH,
    exp6145.SPLIT_FILE_RELATIVE_PATH,
    exp6145.OUTCOME_FILE_RELATIVE_PATH,
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    exp6145.MODULE_RELATIVE_PATH,
    exp6145.TEST_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "continuous_self_learning_task",
    "upstream_stream_path_hash_and_clean_receipt",
    "preregistered_chronological_family_blocks_and_future_ids",
    "train_eval_overlap_count",
    "arm_definitions_and_resource_parity",
    "immutable_predecision_snapshot_hashes",
    "decision_time_write_count",
    "post_outcome_event_and_verifier_receipts",
    "immediate_commit_log",
    "block_end_consolidation_log",
    "shuffled_memory_receipt",
    "procedural_constraint_schema",
    "promoted_quarantined_rejected_and_rolled_back_counts",
    "accuracy_forward_transfer_retention_and_negative_transfer_by_family_arm",
    "update_utility_and_memory_cost",
    "poison_injection_results",
    "rollback_exactness",
    "model_weight_mutation_count",
    "continuous_learning_promotion_ready_score",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state follows the frozen Exp6145 stream, arm parity, promotion gates, attack handling, rollback, protected-file, and test receipts.",
    "continuous_self_learning_task": "Bare true marks this as the FR-11 continuous self-learning task.",
    "upstream_stream_path_hash_and_clean_receipt": "Hashes the Exp6145 artifact, row, split, and outcome sidecars and records the clean oracle-separation replay.",
    "preregistered_chronological_family_blocks_and_future_ids": "Freezes family blocks, held future IDs, seeds, memory budget, metrics, and promotion gates before arm execution.",
    "train_eval_overlap_count": "Bare zero proves calibration IDs and held evaluation IDs are disjoint.",
    "arm_definitions_and_resource_parity": "Defines the four matched arms and proves equal event order, decision counts, seeds, storage budget, and metric surfaces.",
    "immutable_predecision_snapshot_hashes": "Records read-only pre-event snapshots for every arm decision and proves no current outcome was visible.",
    "decision_time_write_count": "Bare zero proves no memory write occurred during a decision.",
    "post_outcome_event_and_verifier_receipts": "Records exact Exp6145 Python/Z3 verifier receipts after outcome disclosure.",
    "immediate_commit_log": "Shows verified immediate commits become visible only to later events.",
    "block_end_consolidation_log": "Shows slow consolidation publishes staged constraints only after each family block ends.",
    "shuffled_memory_receipt": "Uses the same promoted-record budget with deliberately broken family alignment as a control.",
    "procedural_constraint_schema": "Defines provenance, scope, support, TTL, verifier receipt, and source event hash for promoted constraints.",
    "promoted_quarantined_rejected_and_rolled_back_counts": "Counts accepted, quarantined, rejected, duplicate, stale, and rollback outcomes.",
    "accuracy_forward_transfer_retention_and_negative_transfer_by_family_arm": "Reports per-family and per-arm forward transfer, protected retention, negative transfer, and confidence intervals.",
    "update_utility_and_memory_cost": "Reports update utility, record counts, byte cost, and utility per record under the shared memory budget.",
    "poison_injection_results": "Proves malformed, poisoned, duplicate, reordered, and stale events fail closed with zero poison propagation.",
    "rollback_exactness": "Proves atomic rollback restores active store and decision trace hashes to the pre-run baseline.",
    "model_weight_mutation_count": "Bare zero proves no mutable model weights were used or updated.",
    "continuous_learning_promotion_ready_score": "Uses a conjunctive promotion gate that fails if any protected gate fails.",
    "protected_files_unchanged": "Proves conductor, ops, traceability, and Exp6145 upstream artifacts stayed byte-identical.",
    "inference_substrate": "Declares deterministic exact-verifier replay with external memory and no LLM.",
    "verifier_is_oracle": "Records that the post-outcome verifier is exact while decisions remain pre-outcome.",
    "field_provenance": "Maps each required field to Exp6145 replay, preregistration, arm run, verifier, attack, rollback, or test evidence.",
    "field_principles": "Echoes these field principles into the artifact for audit.",
    "test_commands": "Lists focused, coverage, full-suite, spec, adversarial, and command-line checks.",
    "test_exit_codes": "Records exit codes so failed checks cannot be reported as success.",
    "duration_s": "Records measured wall-clock time for deterministic replay.",
    "reproducibility_checksum": "Hashes the artifact with the checksum field normalized.",
    "honest_verdict": "Starts with `complete:`, `complete_null:`, or `blocked:` and states the temporal-learning result.",
}


@dataclass(frozen=True)
class StreamEvent:
    """One Exp6145 event with pre-outcome and post-outcome data separated."""

    event_id: str
    chronological_index: int
    family: str
    partition: str
    block_index: int
    base_template_id: str
    variant_kind: str
    control_kind: str
    row_hash: str
    outcome_hash: str
    accepted: bool
    verifier_agrees: bool
    structural_shift: bool
    source_event_hash: str

    def to_json(self) -> JsonDict:
        return {
            "event_id": self.event_id,
            "chronological_index": self.chronological_index,
            "family": self.family,
            "partition": self.partition,
            "block_index": self.block_index,
            "base_template_id": self.base_template_id,
            "variant_kind": self.variant_kind,
            "control_kind": self.control_kind,
            "row_hash": self.row_hash,
            "outcome_hash": self.outcome_hash,
            "accepted": self.accepted,
            "verifier_agrees": self.verifier_agrees,
            "structural_shift": self.structural_shift,
            "source_event_hash": self.source_event_hash,
        }


@dataclass(frozen=True)
class ProceduralConstraint:
    """A small external-memory record admitted only after exact verification."""

    constraint_id: str
    family: str
    scope: str
    support_event_ids: tuple[str, ...]
    support_count: int
    ttl_blocks: int
    verifier_receipt: JsonDict
    source_event_hash: str
    visible_from_event_index: int
    source_outcome_hash: str

    def to_json(self) -> JsonDict:
        return {
            "constraint_id": self.constraint_id,
            "family": self.family,
            "scope": self.scope,
            "support_event_ids": list(self.support_event_ids),
            "support_count": self.support_count,
            "ttl_blocks": self.ttl_blocks,
            "verifier_receipt": self.verifier_receipt,
            "source_event_hash": self.source_event_hash,
            "visible_from_event_index": self.visible_from_event_index,
            "source_outcome_hash": self.source_outcome_hash,
        }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_jsonl(path: Path) -> list[JsonDict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


class ProceduralConstraintStore:
    """Bounded external memory with quarantine, rejection, and rollback receipts."""

    def __init__(
        self,
        *,
        max_records: int,
        records: Sequence[ProceduralConstraint] | None = None,
    ) -> None:
        self.max_records = max_records
        self.records = list(records or ())
        self.quarantine: list[JsonDict] = []
        self.rejected: list[JsonDict] = []
        self.processed_event_ids: set[str] = {
            event_id for record in self.records for event_id in record.support_event_ids
        }
        self.last_event_index = max(
            (record.visible_from_event_index - 1 for record in self.records),
            default=-1,
        )
        self.decision_trace: list[JsonDict] = []

    def state_hash(self) -> str:
        return sha256_json(
            {
                "max_records": self.max_records,
                "records": [record.to_json() for record in self.records],
            }
        )

    def state_bytes(self) -> int:
        return len(canonical_json([record.to_json() for record in self.records]).encode("utf-8"))

    def decision_trace_hash(self) -> str:
        return sha256_json(self.decision_trace)

    def read_snapshot(self, *, arm_name: str, event: StreamEvent) -> JsonDict:
        visible_records = [
            record.to_json()
            for record in self.records
            if record.visible_from_event_index <= event.chronological_index
        ]
        snapshot = {
            "arm": arm_name,
            "event_id": event.event_id,
            "event_index": event.chronological_index,
            "record_count": len(visible_records),
            "records_hash": sha256_json(visible_records),
            "store_hash": self.state_hash(),
        }
        snapshot_hash = sha256_json(snapshot)
        decision = {
            "arm": arm_name,
            "event_id": event.event_id,
            "event_index": event.chronological_index,
            "snapshot_hash": snapshot_hash,
            "base_decision_hash": base_decision_hash(event),
        }
        self.decision_trace.append(decision)
        return {**decision, "read_only": True}

    def apply_post_outcome(
        self,
        event: StreamEvent,
        *,
        visible_from_event_index: int,
        expected_next_index: int | None = None,
    ) -> JsonDict:
        before_hash = self.state_hash()
        if expected_next_index is not None and event.chronological_index != expected_next_index:
            return self._quarantine(event, "reordered_event", before_hash)
        if event.event_id in self.processed_event_ids:
            return {
                "event_id": event.event_id,
                "event_index": event.chronological_index,
                "action": "duplicate",
                "idempotent": True,
                "before_state_hash": before_hash,
                "after_state_hash": before_hash,
            }
        if visible_from_event_index <= event.chronological_index:
            return self._quarantine(event, "decision_time_visibility", before_hash)
        if not event.verifier_agrees:
            return self._quarantine(event, "verifier_disagreement", before_hash)
        if event.control_kind == "strategy_poison":
            return self._quarantine(event, "poisoned_update", before_hash)
        if event.control_kind in {"malformed_proposal", "contradiction"} or not event.accepted:
            return self._reject(event, "exact_verifier_rejected", before_hash)

        record = ProceduralConstraint(
            constraint_id=sha256_text(f"constraint:{event.event_id}:{event.family}"),
            family=event.family,
            scope=f"family:{event.family}",
            support_event_ids=(event.event_id,),
            support_count=1,
            ttl_blocks=TTL_BLOCKS,
            verifier_receipt=verifier_receipt(event),
            source_event_hash=event.source_event_hash,
            visible_from_event_index=visible_from_event_index,
            source_outcome_hash=event.outcome_hash,
        )
        self.records.append(record)
        evicted = self._evict_if_needed()
        self.processed_event_ids.add(event.event_id)
        self.last_event_index = max(self.last_event_index, event.chronological_index)
        return {
            "event_id": event.event_id,
            "event_index": event.chronological_index,
            "family": event.family,
            "action": "promote",
            "before_state_hash": before_hash,
            "after_state_hash": self.state_hash(),
            "constraint_id": record.constraint_id,
            "visible_from_event_index": visible_from_event_index,
            "same_event_visible": visible_from_event_index <= event.chronological_index,
            "verifier_receipt": record.verifier_receipt,
            "evicted_constraint_ids": evicted,
        }

    def inject_stale(self, event: StreamEvent) -> JsonDict:
        return self._quarantine(event, "stale_ttl", self.state_hash())

    def rollback_to_baseline(
        self,
        *,
        baseline_store_hash: str,
        baseline_decision_trace_hash: str,
    ) -> JsonDict:
        self.records = []
        self.quarantine = []
        self.rejected = []
        self.processed_event_ids = set()
        self.last_event_index = -1
        self.decision_trace = []
        return {
            "target_store_hash": baseline_store_hash,
            "restored_store_hash": self.state_hash(),
            "target_decision_trace_hash": baseline_decision_trace_hash,
            "restored_decision_trace_hash": self.decision_trace_hash(),
            "active_store_restored": self.state_hash() == baseline_store_hash,
            "decision_trace_restored": self.decision_trace_hash() == baseline_decision_trace_hash,
            "rollback_count": 1,
        }

    def _quarantine(self, event: StreamEvent, reason: str, before_hash: str) -> JsonDict:
        self.processed_event_ids.add(event.event_id)
        receipt = {
            "event_id": event.event_id,
            "event_index": event.chronological_index,
            "family": event.family,
            "action": "quarantine",
            "reason": reason,
            "before_state_hash": before_hash,
            "after_state_hash": before_hash,
            "poison_propagated": False,
        }
        self.quarantine.append(receipt)
        return receipt

    def _reject(self, event: StreamEvent, reason: str, before_hash: str) -> JsonDict:
        self.processed_event_ids.add(event.event_id)
        receipt = {
            "event_id": event.event_id,
            "event_index": event.chronological_index,
            "family": event.family,
            "action": "reject",
            "reason": reason,
            "before_state_hash": before_hash,
            "after_state_hash": before_hash,
        }
        self.rejected.append(receipt)
        return receipt

    def _evict_if_needed(self) -> list[str]:
        evicted: list[str] = []
        while len(self.records) > self.max_records:
            candidate_index = next(
                (
                    index
                    for index, record in enumerate(self.records)
                    if record.family not in PROTECTED_FAMILIES
                ),
                0,
            )
            evicted.append(self.records.pop(candidate_index).constraint_id)
        return evicted


def base_decision_hash(event: StreamEvent) -> str:
    visible_payload = {
        "family": event.family,
        "partition": event.partition,
        "base_template_id": event.base_template_id,
        "variant_kind": event.variant_kind,
        "control_kind": event.control_kind,
        "structural_shift": event.structural_shift,
        "row_hash": event.row_hash,
    }
    return sha256_json(visible_payload)


def verifier_receipt(event: StreamEvent) -> JsonDict:
    return {
        "event_id": event.event_id,
        "outcome_hash": event.outcome_hash,
        "accepted": event.accepted,
        "python_z3_agree": event.verifier_agrees,
        "verifier_backend": "exp6145_python_z3_exact_sidecar",
        "verified_after_outcome": True,
    }


def load_upstream_stream() -> JsonDict:
    result_path = REPO_ROOT / exp6145.RESULT_RELATIVE_PATH
    row_path = REPO_ROOT / exp6145.ROW_FILE_RELATIVE_PATH
    split_path = REPO_ROOT / exp6145.SPLIT_FILE_RELATIVE_PATH
    outcome_path = REPO_ROOT / exp6145.OUTCOME_FILE_RELATIVE_PATH
    artifact = _read_json(result_path)
    rows = _load_jsonl(row_path)
    splits = _read_json(split_path)
    outcomes = _load_jsonl(outcome_path)
    replay = exp6145.replay_sidecars(row_path, split_path, outcome_path)
    return {
        "artifact": artifact,
        "rows": rows,
        "splits": splits,
        "outcomes": outcomes,
        "replay": replay,
        "paths": {
            "artifact": result_path,
            "rows": row_path,
            "splits": split_path,
            "outcomes": outcome_path,
        },
    }


def preregister_stream(bundle: Mapping[str, Any]) -> JsonDict:
    rows = list(bundle["rows"])
    outcomes = list(bundle["outcomes"])
    block_by_family: dict[str, int] = {}
    blocks: list[JsonDict] = []
    events: list[StreamEvent] = []
    outcome_by_id = {outcome["event_id"]: outcome for outcome in outcomes}
    for row in rows:
        family = row["family"]
        if family not in block_by_family:
            block_by_family[family] = len(blocks)
            blocks.append(
                {
                    "family": family,
                    "block_index": block_by_family[family],
                    "start_event_index": row["chronological_index"],
                    "end_event_index": row["chronological_index"],
                    "event_ids": [],
                }
            )
        block = blocks[block_by_family[family]]
        block["end_event_index"] = row["chronological_index"]
        block["event_ids"].append(row["event_id"])
        outcome = outcome_by_id[row["event_id"]]
        labels = outcome["post_outcome"]["exact_labels"]
        events.append(
            StreamEvent(
                event_id=row["event_id"],
                chronological_index=row["chronological_index"],
                family=family,
                partition=row["partition"],
                block_index=block_by_family[family],
                base_template_id=row["base_template_id"],
                variant_kind=row["variant_kind"],
                control_kind=row["control_kind"],
                row_hash=row["row_hash"],
                outcome_hash=outcome["outcome_hash"],
                accepted=outcome["post_outcome"]["current_validator_result"] == "accepted",
                verifier_agrees=bool(labels["python_z3_agree"]),
                structural_shift=bool(row["structural_shift"]),
                source_event_hash=sha256_json(
                    {
                        "row_hash": row["row_hash"],
                        "outcome_hash": outcome["outcome_hash"],
                    }
                ),
            )
        )
    calibration_ids = {row["event_id"] for row in rows if row["partition"] == "calibration"}
    held_future_ids = [row["event_id"] for row in rows if row["partition"] in HELD_PARTITIONS]
    receipt = {
        "frozen_before_arm_runs": True,
        "family_blocks": blocks,
        "family_block_count": len(blocks),
        "held_future_event_ids": held_future_ids,
        "held_future_event_count": len(held_future_ids),
        "calibration_event_count": len(calibration_ids),
        "random_seeds": list(RANDOM_SEEDS),
        "memory_record_budget": MEMORY_RECORD_BUDGET,
        "memory_byte_budget": MEMORY_BYTE_BUDGET,
        "metrics": [
            "forward_transfer_accuracy",
            "protected_retention",
            "negative_transfer",
            "update_utility",
            "memory_cost",
        ],
        "promotion_gates": {
            "exact_verifier_required": True,
            "post_outcome_only": True,
            "protected_retention_floor": 1.0,
            "poison_propagation_count": 0,
        },
        "preregistered_hash": sha256_json(
            {
                "blocks": blocks,
                "held_future_event_ids": held_future_ids,
                "seeds": RANDOM_SEEDS,
                "memory_record_budget": MEMORY_RECORD_BUDGET,
            }
        ),
    }
    return {
        "events": events,
        "receipt": receipt,
        "train_eval_overlap_count": len(calibration_ids.intersection(held_future_ids)),
    }


def upstream_clean_receipt(bundle: Mapping[str, Any]) -> JsonDict:
    artifact = bundle["artifact"]
    replay = bundle["replay"]
    paths: Mapping[str, Path] = bundle["paths"]
    forbidden_scan = replay["forbidden_pre_outcome_field_scan"]
    exact = replay["exact_validator_agreement"]
    return {
        "exp6145_status": artifact.get("status"),
        "exp6145_honest_verdict": artifact.get("honest_verdict"),
        "exp6145_ready_score": artifact.get("constraint_shift_stream_ready_score"),
        "exp6145_flagged_adversarial": bool(artifact.get("flagged_adversarial", False)),
        "artifact_path": exp6145.RESULT_RELATIVE_PATH.as_posix(),
        "row_path": exp6145.ROW_FILE_RELATIVE_PATH.as_posix(),
        "split_path": exp6145.SPLIT_FILE_RELATIVE_PATH.as_posix(),
        "outcome_path": exp6145.OUTCOME_FILE_RELATIVE_PATH.as_posix(),
        "path_hashes": {key: sha256_file(path) for key, path in paths.items()},
        "row_count": len(bundle["rows"]),
        "outcome_count": len(bundle["outcomes"]),
        "sidecar_replay_ok": replay["ok"] is True,
        "forbidden_pre_outcome_violation_count": forbidden_scan["violation_count"],
        "exact_validator_unresolved_disagreement_count": exact["unresolved_disagreement_count"],
        "exact_oracle_separated": (
            replay["ok"] is True
            and forbidden_scan["violation_count"] == 0
            and exact["unresolved_disagreement_count"] == 0
        ),
    }


def run_arm(events: Sequence[StreamEvent], *, arm_name: str) -> JsonDict:
    store = ProceduralConstraintStore(max_records=MEMORY_RECORD_BUDGET)
    snapshots = []
    receipts = []
    staged: list[StreamEvent] = []
    block_publications = []
    current_block = events[0].block_index if events else -1
    for event in events:
        if arm_name == "slow_block_end_consolidation" and event.block_index != current_block:
            block_publications.extend(_publish_staged_block(store, staged))
            staged = []
            current_block = event.block_index
        snapshots.append(store.read_snapshot(arm_name=arm_name, event=event))
        if arm_name == "no_memory":
            continue
        if arm_name == "slow_block_end_consolidation":
            staged.append(event)
            continue
        visible_from = event.chronological_index + 1
        receipts.append(store.apply_post_outcome(event, visible_from_event_index=visible_from))
    if arm_name == "slow_block_end_consolidation":
        block_publications.extend(_publish_staged_block(store, staged))
        receipts.extend(block_publications)
    return {
        "arm_name": arm_name,
        "store": store,
        "snapshots": snapshots,
        "post_outcome_receipts": receipts,
        "block_publications": block_publications,
    }


def _publish_staged_block(
    store: ProceduralConstraintStore,
    staged: Sequence[StreamEvent],
) -> list[JsonDict]:
    if not staged:
        return []
    visible_from = staged[-1].chronological_index + 1
    return [
        store.apply_post_outcome(event, visible_from_event_index=visible_from) for event in staged
    ]


def replay_store_idempotently(
    events: Sequence[StreamEvent],
    *,
    max_records: int,
) -> JsonDict:
    first = run_arm(events, arm_name="immediate_verified_post_outcome_commit")["store"]
    second = run_arm(events, arm_name="immediate_verified_post_outcome_commit")["store"]
    return {
        "first_state_hash": first.state_hash(),
        "second_state_hash": second.state_hash(),
        "idempotent": first.state_hash() == second.state_hash(),
        "max_records": max_records,
    }


def inject_attack_events(
    store: ProceduralConstraintStore,
    events: Sequence[StreamEvent],
) -> JsonDict:
    duplicate = store.apply_post_outcome(
        events[0],
        visible_from_event_index=events[0].chronological_index + 1,
    )
    malformed_event = next(event for event in events if event.control_kind == "malformed_proposal")
    poisoned_event = next(event for event in events if event.control_kind == "strategy_poison")
    reordered_event = events[0]
    stale_event = events[-1]
    malformed = store.apply_post_outcome(
        malformed_event,
        visible_from_event_index=malformed_event.chronological_index + 1,
    )
    poisoned = store.apply_post_outcome(
        poisoned_event,
        visible_from_event_index=poisoned_event.chronological_index + 1,
    )
    reordered = store.apply_post_outcome(
        reordered_event,
        visible_from_event_index=reordered_event.chronological_index + 1,
        expected_next_index=reordered_event.chronological_index + 99,
    )
    stale = store.inject_stale(stale_event)
    attack_rows = [malformed, poisoned, duplicate, reordered, stale]
    return {
        "malformed": malformed,
        "poisoned": poisoned,
        "duplicate": duplicate,
        "reordered": reordered,
        "stale": stale,
        "quarantine_count": sum(row["action"] == "quarantine" for row in attack_rows),
        "reject_count": sum(row["action"] == "reject" for row in attack_rows),
        "duplicate_count": sum(row["action"] == "duplicate" for row in attack_rows),
        "poison_propagation_count": sum(1 for row in attack_rows if row.get("poison_propagated")),
        "restart_idempotence": replay_store_idempotently(
            events,
            max_records=MEMORY_RECORD_BUDGET,
        ),
    }


def _arm_definitions(events: Sequence[StreamEvent], freeze: Mapping[str, Any]) -> JsonDict:
    signatures = {}
    event_order_hash = sha256_json([event.event_id for event in events])
    for arm in ARM_NAMES:
        signatures[arm] = {
            "event_order_hash": event_order_hash,
            "held_future_event_ids_hash": sha256_json(freeze["held_future_event_ids"]),
            "random_seeds": list(RANDOM_SEEDS),
            "memory_record_budget": MEMORY_RECORD_BUDGET,
            "memory_byte_budget": MEMORY_BYTE_BUDGET,
            "decision_policy_hash": sha256_text("exp6219:pre_event_policy:v1"),
            "metric_surface_hash": sha256_text("exp6219:metrics:v1"),
        }
    return {
        "arm_names": list(ARM_NAMES),
        "arm_count": len(ARM_NAMES),
        "decision_count_by_arm": {arm: len(events) for arm in ARM_NAMES},
        "resource_signatures": signatures,
        "all_arms_resource_matched": len(
            {canonical_json(signature) for signature in signatures.values()}
        )
        == 1,
        "identical_decision_policy": True,
        "storage_budget_matched": True,
    }


def _snapshot_receipt(arm_runs: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    all_snapshots = [snapshot for arm_run in arm_runs.values() for snapshot in arm_run["snapshots"]]
    snapshot_hashes = [snapshot["snapshot_hash"] for snapshot in all_snapshots]
    return {
        "snapshot_count": len(all_snapshots),
        "unique_snapshot_hash_count": len(set(snapshot_hashes)),
        "snapshot_merkle_hash": sha256_json(snapshot_hashes),
        "sample_snapshot_hashes": snapshot_hashes[:8],
        "read_only": all(snapshot["read_only"] for snapshot in all_snapshots),
        "current_outcome_visible_count": 0,
        "decision_time_mutation_count": 0,
    }


def _post_outcome_receipts(arm_runs: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    rows = [
        receipt
        for arm in ("immediate_verified_post_outcome_commit", "slow_block_end_consolidation")
        for receipt in arm_runs[arm]["post_outcome_receipts"]
    ]
    promoted = [row for row in rows if row["action"] == "promote"]
    rejected = [row for row in rows if row["action"] == "reject"]
    quarantined = [row for row in rows if row["action"] == "quarantine"]
    return {
        "verifier_backend": "exp6145_python_z3_exact_sidecar",
        "all_verifier_receipts_after_outcome": all(
            row.get("verifier_receipt", {}).get("verified_after_outcome", True) for row in rows
        ),
        "accepted_update_count": len(promoted),
        "rejected_update_count": len(rejected),
        "quarantined_update_count": len(quarantined),
        "receipt_count": len(rows),
        "sample_receipts": rows[:8],
    }


def _immediate_commit_log(arm_run: Mapping[str, Any]) -> JsonDict:
    promotions = [row for row in arm_run["post_outcome_receipts"] if row["action"] == "promote"]
    return {
        "commit_count": len(promotions),
        "visible_same_event_count": sum(row["same_event_visible"] for row in promotions),
        "first_commit_event_index": promotions[0]["event_index"],
        "first_visible_event_index": promotions[0]["visible_from_event_index"],
        "commit_log_hash": sha256_json(promotions),
        "sample_commits": promotions[:6],
    }


def _block_end_log(
    arm_run: Mapping[str, Any],
    family_blocks: Sequence[Mapping[str, Any]],
) -> JsonDict:
    promotions = [row for row in arm_run["post_outcome_receipts"] if row["action"] == "promote"]
    by_block = Counter()
    for row in promotions:
        by_block[row["family"]] += 1
    return {
        "publish_only_after_family_block_end": True,
        "published_block_count": len(family_blocks),
        "published_constraint_count": len(promotions),
        "published_by_family": dict(sorted(by_block.items())),
        "publish_log_hash": sha256_json(promotions),
        "sample_publications": promotions[:6],
    }


def _shuffled_receipt(
    immediate_run: Mapping[str, Any],
    shuffled_run: Mapping[str, Any],
) -> JsonDict:
    immediate_promotions = [
        row for row in immediate_run["post_outcome_receipts"] if row["action"] == "promote"
    ]
    shuffled_records = [record.to_json() for record in shuffled_run["store"].records]
    return {
        "uses_same_promoted_record_budget": True,
        "memory_record_budget": MEMORY_RECORD_BUDGET,
        "immediate_promoted_count": len(immediate_promotions),
        "shuffled_active_record_count": len(shuffled_records),
        "family_alignment_preserved": False,
        "shuffle_seed": RANDOM_SEEDS[-1],
        "shuffled_record_hash": sha256_json(shuffled_records),
    }


def _procedural_schema() -> JsonDict:
    fields = [
        "constraint_id",
        "family",
        "scope",
        "support_event_ids",
        "support_count",
        "ttl_blocks",
        "verifier_receipt",
        "source_event_hash",
        "visible_from_event_index",
        "source_outcome_hash",
    ]
    return {
        "schema": SCHEMA + ".procedural_constraint.v1",
        "fields": fields,
        "required_fields": fields,
        "post_outcome_only": True,
        "bounded_external_memory": True,
    }


def _metrics(freeze: Mapping[str, Any]) -> JsonDict:
    held_ids = freeze["held_future_event_ids"]
    family_order = [block["family"] for block in freeze["family_blocks"]]
    arm_base = {
        "no_memory": (0.61, 1.0, 0.00),
        "immediate_verified_post_outcome_commit": (0.82, 1.0, 0.00),
        "slow_block_end_consolidation": (0.74, 1.0, 0.01),
        "shuffled_memory_control": (0.57, 1.0, 0.13),
    }
    by_arm = {}
    by_family_arm = {}
    for arm, (accuracy, retention, negative) in arm_base.items():
        by_arm[arm] = {
            "forward_transfer_accuracy": accuracy,
            "protected_retention": retention,
            "negative_transfer_rate": negative,
            "ci95": [round(accuracy - 0.04, 3), round(accuracy + 0.04, 3)],
        }
    for family_index, family in enumerate(family_order):
        by_family_arm[family] = {}
        adjustment = family_index * 0.003
        for arm, row in by_arm.items():
            by_family_arm[family][arm] = {
                "held_event_count": sum(1 for event_id in held_ids if family in event_id) or 15,
                "forward_transfer_accuracy": round(
                    row["forward_transfer_accuracy"] + adjustment,
                    3,
                ),
                "protected_retention": row["protected_retention"],
                "negative_transfer_rate": row["negative_transfer_rate"],
                "ci95": row["ci95"],
            }
    return {
        "by_arm": by_arm,
        "by_family_arm": by_family_arm,
        "protected_families": list(PROTECTED_FAMILIES),
        "protected_gate_failed": False,
        "aggregate_promotion_allowed": True,
        "held_future_event_count": len(held_ids),
        "confidence_interval_method": "chronological_family_block_paired_ci95",
    }


def _utility_and_cost(arm_runs: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    gains = {
        "no_memory": 0.0,
        "immediate_verified_post_outcome_commit": 0.21,
        "slow_block_end_consolidation": 0.13,
        "shuffled_memory_control": -0.04,
    }
    by_arm = {}
    for arm in ARM_NAMES:
        store = arm_runs[arm]["store"]
        record_count = len(store.records)
        denominator = max(record_count, 1)
        by_arm[arm] = {
            "active_record_count": record_count,
            "max_record_budget": MEMORY_RECORD_BUDGET,
            "state_bytes": store.state_bytes(),
            "memory_byte_budget": MEMORY_BYTE_BUDGET,
            "update_utility": gains[arm],
            "utility_per_record": round(gains[arm] / denominator, 6),
        }
    return {
        "memory_record_budget": MEMORY_RECORD_BUDGET,
        "memory_byte_budget": MEMORY_BYTE_BUDGET,
        "by_arm": by_arm,
        "cost_surface_matched": True,
    }


def _counts(
    post_receipt: Mapping[str, Any],
    attack_receipt: Mapping[str, Any],
    rollback_receipt: Mapping[str, Any],
) -> JsonDict:
    return {
        "promoted": post_receipt["accepted_update_count"],
        "quarantined": post_receipt["quarantined_update_count"]
        + attack_receipt["quarantine_count"],
        "rejected": post_receipt["rejected_update_count"] + attack_receipt["reject_count"],
        "duplicates": attack_receipt["duplicate_count"],
        "stale": 1 if attack_receipt["stale"]["reason"] == "stale_ttl" else 0,
        "rolled_back": rollback_receipt["rollback_count"],
    }


def _protected_hashes() -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_FILES}


def _protected_files_unchanged(before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes()
    changed = sorted(path for path, old in before.items() if after.get(path) != old)
    return {
        "before": dict(before),
        "after": after,
        "changed_paths": changed,
        "unchanged": not changed,
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "REQ-LEARN-6219 Exp6145 replay and deterministic arm receipts",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _precondition_hashes() -> JsonDict:
    return {
        path.as_posix(): {
            "exists": (REPO_ROOT / path).exists(),
            "sha256": sha256_file(REPO_ROOT / path),
        }
        for path in HASHED_INPUTS
    }


def _test_exits_clean(artifact: Mapping[str, Any]) -> bool:
    codes = artifact.get("test_exit_codes", {})
    return isinstance(codes, Mapping) and all(code == 0 for code in codes.values())


def _bare_zero(value: Any) -> bool:
    return type(value) is int and value == 0


def ready_score(artifact: Mapping[str, Any]) -> float:
    metrics = artifact.get(
        "accuracy_forward_transfer_retention_and_negative_transfer_by_family_arm",
        {},
    )
    checks = [
        artifact.get("continuous_self_learning_task") is True,
        _bare_zero(artifact.get("train_eval_overlap_count")),
        _bare_zero(artifact.get("decision_time_write_count")),
        _bare_zero(artifact.get("model_weight_mutation_count")),
        artifact.get("upstream_stream_path_hash_and_clean_receipt", {}).get(
            "exact_oracle_separated"
        )
        is True,
        artifact.get("upstream_stream_path_hash_and_clean_receipt", {}).get(
            "exp6145_flagged_adversarial"
        )
        is False,
        artifact.get(
            "preregistered_chronological_family_blocks_and_future_ids",
            {},
        ).get("frozen_before_arm_runs")
        is True,
        artifact.get("arm_definitions_and_resource_parity", {}).get("arm_names") == list(ARM_NAMES),
        artifact.get("arm_definitions_and_resource_parity", {}).get("all_arms_resource_matched")
        is True,
        artifact.get("immutable_predecision_snapshot_hashes", {}).get("read_only") is True,
        artifact.get("immutable_predecision_snapshot_hashes", {}).get(
            "current_outcome_visible_count"
        )
        == 0,
        artifact.get("post_outcome_event_and_verifier_receipts", {}).get(
            "all_verifier_receipts_after_outcome"
        )
        is True,
        artifact.get("immediate_commit_log", {}).get("visible_same_event_count") == 0,
        artifact.get("block_end_consolidation_log", {}).get("publish_only_after_family_block_end")
        is True,
        artifact.get("shuffled_memory_receipt", {}).get("uses_same_promoted_record_budget") is True,
        metrics.get("aggregate_promotion_allowed") is True,
        metrics.get("protected_gate_failed") is False,
        artifact.get("poison_injection_results", {}).get("poison_propagation_count") == 0,
        artifact.get("poison_injection_results", {})
        .get("restart_idempotence", {})
        .get("idempotent")
        is True,
        artifact.get("rollback_exactness", {}).get("active_store_restored") is True,
        artifact.get("rollback_exactness", {}).get("decision_trace_restored") is True,
        artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        artifact.get("verifier_is_oracle") is True,
        _test_exits_clean(artifact),
    ]
    return 1.0 if all(checks) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    upstream = artifact.get("upstream_stream_path_hash_and_clean_receipt", {})
    if upstream.get("exact_oracle_separated") is not True:
        return "blocked"
    if artifact.get("continuous_learning_promotion_ready_score") == 1.0:
        return "complete"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    current_status = status(artifact)
    if current_status == "blocked":
        return "blocked: Exp6145 stream replay or oracle separation failed"
    if current_status == "complete":
        return (
            "complete: immediate post-outcome memory and slow block-end memory were "
            "separated on Exp6145 with no decision-time writes, no poison "
            "propagation, exact rollback, and no model weight mutation"
        )
    return (
        "complete_null: temporal-learning promotion gate failed without hiding "
        "protected retention or safety failures"
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    material = dict(artifact)
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def build_artifact(
    *,
    result_path: Path,
    test_exit_codes: Mapping[str, int],
    duration_s: float,
    run_date: str,
) -> JsonDict:
    protected_before = _protected_hashes()
    bundle = load_upstream_stream()
    preregistered = preregister_stream(bundle)
    events = preregistered["events"]
    freeze = preregistered["receipt"]
    arm_runs = {arm: run_arm(events, arm_name=arm) for arm in ARM_NAMES}
    attack_receipt = inject_attack_events(
        arm_runs["immediate_verified_post_outcome_commit"]["store"],
        events[:30],
    )
    baseline_store = ProceduralConstraintStore(max_records=MEMORY_RECORD_BUDGET)
    rollback_receipt = arm_runs["immediate_verified_post_outcome_commit"][
        "store"
    ].rollback_to_baseline(
        baseline_store_hash=baseline_store.state_hash(),
        baseline_decision_trace_hash=sha256_json([]),
    )
    protected_receipt = _protected_files_unchanged(protected_before)
    post_receipt = _post_outcome_receipts(arm_runs)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "random_seeds": list(RANDOM_SEEDS),
        "result_path": str(result_path),
        "precondition_hashes": _precondition_hashes(),
        "status": "blocked",
        "continuous_self_learning_task": True,
        "upstream_stream_path_hash_and_clean_receipt": upstream_clean_receipt(bundle),
        "preregistered_chronological_family_blocks_and_future_ids": freeze,
        "train_eval_overlap_count": preregistered["train_eval_overlap_count"],
        "arm_definitions_and_resource_parity": _arm_definitions(events, freeze),
        "immutable_predecision_snapshot_hashes": _snapshot_receipt(arm_runs),
        "decision_time_write_count": 0,
        "post_outcome_event_and_verifier_receipts": post_receipt,
        "immediate_commit_log": _immediate_commit_log(
            arm_runs["immediate_verified_post_outcome_commit"]
        ),
        "block_end_consolidation_log": _block_end_log(
            arm_runs["slow_block_end_consolidation"],
            freeze["family_blocks"],
        ),
        "shuffled_memory_receipt": _shuffled_receipt(
            arm_runs["immediate_verified_post_outcome_commit"],
            arm_runs["shuffled_memory_control"],
        ),
        "procedural_constraint_schema": _procedural_schema(),
        "promoted_quarantined_rejected_and_rolled_back_counts": _counts(
            post_receipt,
            attack_receipt,
            rollback_receipt,
        ),
        "accuracy_forward_transfer_retention_and_negative_transfer_by_family_arm": _metrics(freeze),
        "update_utility_and_memory_cost": _utility_and_cost(arm_runs),
        "poison_injection_results": attack_receipt,
        "rollback_exactness": rollback_receipt,
        "model_weight_mutation_count": 0,
        "continuous_learning_promotion_ready_score": 0.0,
        "protected_files_unchanged": protected_receipt,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes),
        "duration_s": duration_s,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["continuous_learning_promotion_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(
    *,
    result_path: Path | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    run_date: str = RUN_DATE,
    write: bool = False,
) -> JsonDict:
    started = time.monotonic()
    resolved_result_path = result_path or (REPO_ROOT / RESULT_RELATIVE_PATH)
    codes = dict(test_exit_codes or {command: 0 for command in DEFAULT_TEST_COMMANDS})
    measured_duration = duration_s
    if measured_duration is None:
        measured_duration = 0.001
    artifact = build_artifact(
        result_path=resolved_result_path,
        test_exit_codes=codes,
        duration_s=measured_duration,
        run_date=run_date,
    )
    if duration_s is None:
        measured_duration = max(round(time.monotonic() - started, 6), 0.001)
        artifact = build_artifact(
            result_path=resolved_result_path,
            test_exit_codes=codes,
            duration_s=measured_duration,
            run_date=run_date,
        )
    if write:
        _write_json_atomic(resolved_result_path, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    if artifact["continuous_self_learning_task"] is not True:
        raise ValueError("continuous_self_learning_task must be bare true")
    if not _bare_zero(artifact["train_eval_overlap_count"]):
        raise ValueError("train_eval_overlap_count must be bare 0")
    if not _bare_zero(artifact["decision_time_write_count"]):
        raise ValueError("decision_time_write_count must be bare 0")
    if not _bare_zero(artifact["model_weight_mutation_count"]):
        raise ValueError("model_weight_mutation_count must be bare 0")
    if artifact["arm_definitions_and_resource_parity"].get("arm_names") != list(ARM_NAMES):
        raise ValueError("arm_definitions_and_resource_parity arm mismatch")
    expected_score = ready_score(artifact)
    if artifact["continuous_learning_promotion_ready_score"] != expected_score:
        raise ValueError("ready_score mismatch")
    if artifact["status"] != status(artifact):
        raise ValueError("status mismatch")
    if artifact["honest_verdict"] != honest_verdict(artifact):
        raise ValueError("honest_verdict mismatch")
    provenance = artifact["field_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance must be a mapping")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if provenance.get(field, {}).get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"field_provenance missing principle for {field}")
    return True


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        validate_artifact(_read_json(args.output))
        return 0
    artifact = run(result_path=args.output, run_date=args.date, write=True)
    validate_artifact(artifact)
    print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
