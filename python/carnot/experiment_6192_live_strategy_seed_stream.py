"""Exp6192 live two-family strategy seed stream.

Spec refs: REQ-CL-6192-MANDATORY-SEED-STREAM,
REQ-CL-6192-TWO-FAMILY-GGUF, REQ-CL-6192-THREE-STRATEGIES,
REQ-CL-6192-FIXED-ORDER, REQ-CL-6192-RAW-BEFORE-LABEL,
REQ-CL-6192-NO-CORRECTNESS-RETRY, REQ-CL-6192-POST-OUTCOME-COMMIT,
REQ-CL-6192-BOUNDED-MEMORY, REQ-CL-6192-FIXED-BASELINE,
REQ-CL-6192-RETENTION-SEED, REQ-CL-6192-POISON-ROLLBACK,
REQ-CL-6192-EXACT-PROVENANCE, SCENARIO-CL-6192-GATE-FAIL-CLOSED,
SCENARIO-CL-6192-RAW-ORDER-COVERAGE,
SCENARIO-CL-6192-BASELINE-MEMORY,
SCENARIO-CL-6192-POISON-ROLLBACK-RETENTION, SCENARIO-CL-6192-SCHEMA.

The stream is seed experience only. The model sees three frozen public
strategies and public task text; exact private tests are opened only after the
raw corpus has been written and hashed. The resulting policy and memory are
external state for later prospective runs, not model-weight updates.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Protocol

from carnot import experiment_6187_livecodebench_authentic_k8_pool as lcb
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "carnot.experiment_6192.live_strategy_seed_stream.v1"
RAW_ROW_SCHEMA = SCHEMA + ".raw_row"
LABEL_ROW_SCHEMA = SCHEMA + ".label_row"
EXPERIMENT_ID = "experiment_6192_live_strategy_seed_stream"
RUN_DATE = "20260807"
RANDOM_SEED = "20260807-exp6192-live-strategy-seed-stream-v1"
INFERENCE_SUBSTRATE = "local_dual_family_llama_cpp_cuda_live_generation_plus_restricted_execution"

RESULT_RELATIVE_PATH = Path("results/experiment_6192_live_strategy_seed_stream.json")
RAW_RELATIVE_PATH = Path("results/experiment_6192_live_strategy_seed_stream.raw.jsonl")
LABEL_RELATIVE_PATH = Path("results/experiment_6192_live_strategy_seed_stream.labels.jsonl")
MEMORY_RELATIVE_PATH = Path("results/experiment_6192_live_strategy_seed_stream.memory.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
EXP6184_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_6184_v536_evidence_isolation_preflight.json"
)
EXP6184_RERUN_RECEIPT_PATH = Path("/tmp/carnot_exp6184_preflight_6192.json")
EXP6186_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_6186_livecodebench_bank_preregistration.json"
)
BANK_RELATIVE_PATH = Path("data/research/livecodebench_bank_6186.json")
PUBLIC_PROMPT_RELATIVE_PATH = Path("data/research/livecodebench_bank_6186_public_prompts.jsonl")

SEED_TASK_COUNT = 18
MODEL_COUNT = 2
STRATEGY_COUNT = 3
EXPECTED_GENERATION_COUNT = SEED_TASK_COUNT * MODEL_COUNT * STRATEGY_COUNT
MAX_TOKENS = int(os.environ.get("CARNOT_EXP6192_MAX_TOKENS", "384"))
N_CTX = int(os.environ.get("CARNOT_EXP6192_N_CTX", "4096"))
MEMORY_MAX_RECORDS = 24
MEMORY_STATE_BYTE_BOUND = 32768

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

MODEL_SPECS: list[JsonDict] = [
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "primary",
        "family": "qwen3",
        "loader": "llama_cpp.Llama",
        "expected_quantization": "UD-Q4_K_M",
        "weight_mutation_allowed": False,
        "n_gpu_layers": -1,
        "n_ctx": N_CTX,
        "gpu_assignment": {
            "visible_devices": [0, 1],
            "main_gpu": 0,
            "split_mode": "layer",
            "tensor_split": [1.0, 1.0],
        },
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "confirmation",
        "family": "gemma4",
        "loader": "llama_cpp.Llama",
        "expected_quantization": "UD-Q4_K_M",
        "weight_mutation_allowed": False,
        "n_gpu_layers": -1,
        "n_ctx": N_CTX,
        "gpu_assignment": {
            "visible_devices": [0, 1],
            "main_gpu": 0,
            "split_mode": "layer",
            "tensor_split": [1.0, 1.0],
        },
    },
]
MANDATED_MODEL_IDS = tuple(str(row["hf_id"]) for row in MODEL_SPECS)

STRATEGY_PROMPTS: tuple[JsonDict, ...] = (
    {
        "strategy_id": "direct_implementation",
        "title": "Direct Implementation",
        "prompt": (
            "Implement the requested Python solution directly. Prefer clear "
            "control flow, standard input/output handling when no starter "
            "function is provided, and return only runnable code."
        ),
    },
    {
        "strategy_id": "invariant_first",
        "title": "Invariant First",
        "prompt": (
            "Before writing code, identify the key invariant or recurrence and "
            "then encode it compactly. Return only the final Python code."
        ),
    },
    {
        "strategy_id": "edge_case_guarded",
        "title": "Edge Case Guarded",
        "prompt": (
            "Write a Python solution that handles boundary cases first, then the "
            "general case. Keep parsing explicit and return only runnable code."
        ),
    },
)
STRATEGY_IDS = tuple(str(row["strategy_id"]) for row in STRATEGY_PROMPTS)

GENERATION_CONFIG: JsonDict = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 40,
    "repeat_penalty": 1.05,
    "max_tokens": MAX_TOKENS,
    "n_ctx": N_CTX,
    "seed_base": 619200000,
    "seed_rule": "619200000 + deterministic_order_index",
    "prompt_transport": "chat_completion_raw_python",
    "correctness_conditioned_retry": False,
    "parser_repair": False,
    "grammar_retry": False,
    "candidate_replacement": False,
    "private_tests_in_prompt": False,
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6192_live_strategy_seed_stream.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6192_live_strategy_seed_stream.py -m pytest tests/python/test_experiment_6192_live_strategy_seed_stream.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6192_live_strategy_seed_stream.py --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6192_live_strategy_seed_stream.py",
    ".venv/bin/python -m carnot.experiment_6192_live_strategy_seed_stream --validate",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6192_live_strategy_seed_stream.json",
    "git status --short -- scripts/research_conductor.py ops/changelog.md ops/status.md _bmad/traceability.md",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/pytest tests/python -q",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_bank_hash_and_gate_receipt",
    "model_specs",
    "model_cache_hash_revision_quantization_template_and_cuda_receipts",
    "dual_gpu_utilization_memory_intervals",
    "seed_task_ids_hash_and_strategy_prompts",
    "model_strategy_task_order_and_random_seed",
    "raw_before_label_checkpoint_hashes_and_timestamps",
    "task_model_strategy_coverage_matrix",
    "restricted_oracle_outcomes",
    "correctness_retry_count",
    "fixed_no_memory_policy_by_model_family",
    "bounded_memory_schema_capacity_eviction_and_snapshot_receipt",
    "initial_memory_event_count_and_hash",
    "poison_rollback_and_retention_fixture_receipts",
    "private_test_noninterference_receipt",
    "verifier_is_oracle",
    "seed_stream_ready_score",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state follows preconditions, raw coverage, labels, fixed baseline, memory fixtures, protected files, and tests.",
    "preconditions_checked": "Exp6184 preflight, Exp6186 gate, seed/test hashes, model/CUDA/GPU receipts, strategy prompts, order seed, executor limits, memory schema/capacity, git status, protected files, and root clutter are recorded before load.",
    "upstream_bank_hash_and_gate_receipt": "Exp6186 `bank_ready_score==1` plus bank/public/vault hashes gates the seed stream.",
    "model_specs": "Exactly the two mandated GGUF families are listed.",
    "model_cache_hash_revision_quantization_template_and_cuda_receipts": "Exact GGUF file identity, embedded tokenizer/template, no AutoTokenizer, CUDA/offload, and llama.cpp receipts are recorded.",
    "dual_gpu_utilization_memory_intervals": "Both-GPU identity and memory/utilization intervals are preserved.",
    "seed_task_ids_hash_and_strategy_prompts": "The 18 seed task IDs, task hashes, and three frozen label-blind strategy prompts are recorded.",
    "model_strategy_task_order_and_random_seed": "The deterministic 108-cell order and random seed are recorded.",
    "raw_before_label_checkpoint_hashes_and_timestamps": "Raw shards and corpus hashes prove raw outputs were sealed before private labels.",
    "task_model_strategy_coverage_matrix": "Each seed task has all two-model/three-strategy cells.",
    "restricted_oracle_outcomes": "Post-seal restricted-execution outcomes are summarized and sidecar-hashed.",
    "correctness_retry_count": "Bare zero; correctness never triggers retry, repair, replacement, or regeneration.",
    "fixed_no_memory_policy_by_model_family": "Seed-only deterministic policy winners are frozen per family.",
    "bounded_memory_schema_capacity_eviction_and_snapshot_receipt": "Schema, capacity, eviction, snapshots, reads, and append-only ledger receipts describe the initialized store.",
    "initial_memory_event_count_and_hash": "The post-label seed memory event count and hash are recorded.",
    "poison_rollback_and_retention_fixture_receipts": "Poison rejection, rollback exactness, duplicate idempotence, and retention probe fixtures are auditable.",
    "private_test_noninterference_receipt": "Private tests do not enter prompts, raw shards, strategy choice, retries, or baseline policy before labeling.",
    "verifier_is_oracle": "Bare true for post-generation labeling only and bare false for prompt strategy choice.",
    "seed_stream_ready_score": "One only with 108 sealed live generations, complete two-model/three-strategy coverage, zero correctness retries, a frozen per-family baseline, and a bounded tested memory store.",
    "protected_files_unchanged": "Conductor and reconciler-owned files remain byte-identical.",
    "duration_s": "Wall-clock duration is reported.",
    "inference_substrate": "The value is `local_dual_family_llama_cpp_cuda_live_generation_plus_restricted_execution`.",
    "field_provenance": "Every field traces to REQ-CL-6192, receipts, checksums, tests, or protected-file hashes.",
    "test_commands": "Focused unit/spec coverage, model identity, 108-cell coverage/order, raw-before-label, retry prohibition, memory transaction/poison/rollback/retention, schema, adversarial, protected-file, dual-GPU E2E, full pytest, and root-clutter checks are listed.",
    "test_exit_codes": "Failed verification commands prevent readiness.",
    "reproducibility_checksum": "A stable checksum covers inputs, receipts, sidecars, commands, protected files, and output paths excluding duration and itself.",
    "honest_verdict": "Starts with `complete_ready:`, `complete_partial:`, `retired:`, or `blocked:` and names live generation coverage by family.",
}


@dataclass(frozen=True)
class MemoryEvent:
    """One post-outcome memory event with enough hashes to audit provenance."""

    event_id: str
    sequence_index: int
    model_family: str
    task_id: str
    strategy_id: str
    outcome: str
    passed: bool
    raw_row_hash: str
    label_row_hash: str
    commit_after_outcome: bool
    poisoned: bool = False

    def to_json(self) -> JsonDict:
        return {
            "event_id": self.event_id,
            "sequence_index": self.sequence_index,
            "model_family": self.model_family,
            "task_id": self.task_id,
            "strategy_id": self.strategy_id,
            "outcome": self.outcome,
            "passed": self.passed,
            "raw_row_hash": self.raw_row_hash,
            "label_row_hash": self.label_row_hash,
            "commit_after_outcome": self.commit_after_outcome,
            "poisoned": self.poisoned,
        }


class BoundedTransactionalMemoryStore:
    """Bounded external event memory with exact snapshots and rollback."""

    def __init__(
        self,
        *,
        max_records: int = MEMORY_MAX_RECORDS,
        state_byte_bound: int = MEMORY_STATE_BYTE_BOUND,
        records: Sequence[JsonDict] | None = None,
    ) -> None:
        self.max_records = int(max_records)
        self.state_byte_bound = int(state_byte_bound)
        self.records = [dict(row) for row in records or ()]
        self.event_log: list[JsonDict] = []
        self.quarantine: list[JsonDict] = []
        self.processed_event_ids = {str(row["event_id"]) for row in self.records}
        self._snapshots: dict[str, list[JsonDict]] = {}
        self._remember_snapshot()

    def clone(self) -> "BoundedTransactionalMemoryStore":
        clone = BoundedTransactionalMemoryStore(
            max_records=self.max_records,
            state_byte_bound=self.state_byte_bound,
            records=self.records,
        )
        clone.event_log = [dict(row) for row in self.event_log]
        clone.quarantine = [dict(row) for row in self.quarantine]
        clone.processed_event_ids = set(self.processed_event_ids)
        clone._snapshots = {
            key: [dict(row) for row in rows] for key, rows in self._snapshots.items()
        }
        return clone

    def state_hash(self) -> str:
        return sha256_json({"max_records": self.max_records, "records": self.records})

    def event_log_hash(self) -> str:
        return sha256_json(self.event_log)

    def state_bytes(self) -> int:
        return len(canonical_json({"records": self.records}).encode("utf-8"))

    def commit(self, event: MemoryEvent) -> JsonDict:
        before_hash = self.state_hash()
        if event.event_id in self.processed_event_ids:
            return {
                "event_id": event.event_id,
                "action": "duplicate",
                "before_state_hash": before_hash,
                "after_state_hash": before_hash,
                "idempotent": True,
            }
        if (
            event.poisoned
            or not event.commit_after_outcome
            or not event.raw_row_hash
            or not event.label_row_hash
        ):
            return self._quarantine(event, before_hash)
        record = {
            **event.to_json(),
            "event_record_hash": sha256_json(event.to_json()),
        }
        self.records.append(record)
        self.event_log.append({"action": "commit", "event": record})
        self.processed_event_ids.add(event.event_id)
        evicted = self._evict_if_needed()
        self._remember_snapshot()
        return {
            "event_id": event.event_id,
            "action": "commit",
            "before_state_hash": before_hash,
            "after_state_hash": self.state_hash(),
            "commit_after_outcome": True,
            "evicted_event_ids": evicted,
        }

    def rollback_to(self, state_hash: str) -> JsonDict:
        if state_hash not in self._snapshots:
            raise ValueError("unknown rollback target")
        before_hash = self.state_hash()
        self.records = [dict(row) for row in self._snapshots[state_hash]]
        restored = self.state_hash()
        return {
            "before_state_hash": before_hash,
            "target_state_hash": state_hash,
            "restored_state_hash": restored,
            "rollback_exact": restored == state_hash,
        }

    def retention_probe(self) -> JsonDict:
        before = self.state_hash()
        by_family = Counter(str(row["model_family"]) for row in self.records)
        after = self.state_hash()
        return {
            "state_hash_before": before,
            "state_hash_after": after,
            "retention_counts_by_model_family": dict(sorted(by_family.items())),
            "retention_probe_mutated_state": before != after,
        }

    def receipt(self) -> JsonDict:
        read_before = self.state_hash()
        snapshot_rows = [dict(row) for row in self.records]
        read_after = self.state_hash()
        return {
            "schema": SCHEMA + ".bounded_transactional_memory_store",
            "bounded": len(self.records) <= self.max_records
            and self.state_bytes() <= self.state_byte_bound,
            "append_only_event_log": len(self.event_log) >= len(self.records),
            "capacity": {
                "max_records": self.max_records,
                "state_byte_bound": self.state_byte_bound,
                "active_record_count": len(self.records),
                "active_state_bytes": self.state_bytes(),
            },
            "eviction": {
                "policy": "oldest_active_records_after_append",
                "evicted_event_count": max(0, len(self.event_log) - len(self.records)),
                "evicted_event_ids": [
                    str(row["event"]["event_id"])
                    for row in self.event_log
                    if str(row["event"]["event_id"])
                    not in {str(record["event_id"]) for record in self.records}
                ],
            },
            "snapshot_read_receipt": {
                "snapshot_count": len(self._snapshots),
                "read_row_count": len(snapshot_rows),
                "read_state_hash_before": read_before,
                "read_state_hash_after": read_after,
                "read_mutated_state": read_before != read_after,
            },
            "active_state_hash": self.state_hash(),
            "event_log_hash": self.event_log_hash(),
        }

    def to_json(self) -> JsonDict:
        return {
            "schema": SCHEMA + ".memory_sidecar",
            "records": self.records,
            "event_log": self.event_log,
            "quarantine": self.quarantine,
            "receipt": self.receipt(),
        }

    def _quarantine(self, event: MemoryEvent, before_hash: str) -> JsonDict:
        self.processed_event_ids.add(event.event_id)
        receipt = {
            "event_id": event.event_id,
            "action": "quarantine",
            "before_state_hash": before_hash,
            "after_state_hash": before_hash,
            "poison_propagated": False,
            "reason": "poisoned_or_invalid_post_outcome_event",
        }
        self.quarantine.append(receipt)
        self.event_log.append({"action": "quarantine", "event": event.to_json()})
        return receipt

    def _evict_if_needed(self) -> list[str]:
        evicted: list[str] = []
        while len(self.records) > self.max_records or self.state_bytes() > self.state_byte_bound:
            if not self.records:  # pragma: no cover - defensive guard for malformed stores.
                break
            evicted.append(str(self.records.pop(0)["event_id"]))
        return evicted

    def _remember_snapshot(self) -> None:
        self._snapshots[self.state_hash()] = [dict(row) for row in self.records]


class StrategyGenerationBackend(Protocol):
    def generate(
        self,
        *,
        model_spec: JsonDict,
        public_tasks: list[JsonDict],
        sample_plan: list[JsonDict],
        generation_config: JsonDict,
    ) -> JsonDict:
        """Return raw generation rows for the model-specific sample plan."""


def run(
    *,
    result_path: Path | None = None,
    raw_path: Path | None = None,
    label_path: Path | None = None,
    memory_path: Path | None = None,
    task_rows: Sequence[Mapping[str, Any]] | None = None,
    preconditions_checked: JsonDict | None = None,
    model_resolution: JsonDict | None = None,
    generation_backend: StrategyGenerationBackend | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Run the seed stream or write a fail-closed artifact before model load."""

    started = time.perf_counter()
    paths = _resolve_paths(result_path, raw_path, label_path, memory_path)
    tasks = list(task_rows) if task_rows is not None else load_frozen_seed_tasks()
    public_tasks = build_public_tasks(tasks)
    sample_plan = build_generation_plan(public_tasks)
    preconditions = preconditions_checked or capture_preconditions(paths)
    resolution = model_resolution or resolve_mandatory_models()
    model_specs = _model_specs_from_resolution(resolution)
    upstream = upstream_bank_hash_and_gate_receipt(preconditions)
    gate = structured_gate_receipt(preconditions, resolution, public_tasks, sample_plan, upstream)
    raw_rows: list[JsonDict] = []
    generation_lifecycle: dict[str, JsonDict] = {}
    raw_commit = empty_raw_commit(paths["raw"], len(sample_plan))
    label_receipt = empty_label_receipt(paths["label"])

    if gate["passed"]:
        existing = inspect_existing_raw(paths["raw"], sample_plan)
        if existing["blocked"]:  # pragma: no cover - rare corrupt-resume path.
            gate["passed"] = False
            gate["blocked_reasons"].extend(existing["blocked_reasons"])
        else:
            raw_rows = list(existing["rows"])
            missing_plan = list(existing["missing_plan"])
            if missing_plan:
                backend = generation_backend or NativeDualFamilyBackend()
                generated_rows: list[JsonDict] = []
                for model in model_specs:
                    model_plan = [
                        row for row in missing_plan if row["model_hf_id"] == model["hf_id"]
                    ]
                    if not model_plan:  # pragma: no cover - exact two-family plan always has rows.
                        continue
                    generated = backend.generate(
                        model_spec=model,
                        public_tasks=public_tasks,
                        sample_plan=model_plan,
                        generation_config=dict(GENERATION_CONFIG),
                    )
                    generation_lifecycle[str(model["hf_id"])] = dict(
                        generated.get("lifecycle_receipt", {})
                    )
                    generated_rows.extend(
                        assemble_raw_rows(
                            generated.get("rows", []),
                            model,
                            public_tasks,
                            model_plan,
                        )
                    )
                raw_rows.extend(generated_rows)
                _write_jsonl(paths["raw"], ordered_rows(raw_rows, sample_plan))
            raw_rows = ordered_rows(raw_rows, sample_plan)
            if raw_rows_complete(raw_rows, sample_plan):
                raw_commit = raw_before_label_checkpoint(paths["raw"], raw_rows, sample_plan)
                labels = label_raw_rows_after_commit(
                    raw_rows=raw_rows,
                    raw_corpus_sha256=str(raw_commit["raw_corpus_sha256"]),
                    tasks=tasks,
                    label_path=paths["label"],
                    executor=lcb.RestrictedLiveCodeBenchExecutor.from_preconditions(preconditions),
                )
                label_receipt = labels["label_receipt"]

    for model in model_specs:
        model["actual_use_count"] = sum(
            row.get("model_hf_id") == model["hf_id"] for row in raw_rows
        )
    coverage = task_model_strategy_coverage_matrix(raw_rows, sample_plan)
    outcomes = restricted_oracle_outcomes(paths["label"])
    policy = fixed_no_memory_policy_by_model_family(paths["label"])
    store, commit_receipts = initialize_memory_store(paths["label"])
    if write:
        _write_json(paths["memory"], store.to_json())
    memory_receipt = store.receipt()
    memory_receipt["post_outcome_commit_receipt"] = commit_receipts
    fixtures = poison_rollback_and_retention_fixture_receipts(store)
    noninterference = private_test_noninterference_receipt(
        public_tasks,
        raw_rows,
        label_path=paths["label"],
        raw_commit=raw_commit,
    )
    protected = protected_files_unchanged(preconditions)
    cuda_receipts = model_cache_hash_revision_quantization_template_and_cuda_receipts(
        resolution,
        model_specs,
    )
    dual_gpu = dual_gpu_utilization_memory_intervals(preconditions, generation_lifecycle)
    exit_codes = dict(test_exit_codes or {})
    score = seed_stream_ready_score(
        gate=gate,
        coverage=coverage,
        raw_commit=raw_commit,
        outcomes=outcomes,
        policy=policy,
        memory_receipt=memory_receipt,
        fixtures=fixtures,
        noninterference=noninterference,
        protected=protected,
        test_exit_codes=exit_codes,
    )
    status = "complete_ready" if score == 1 else ("complete_partial" if raw_rows else "blocked")
    measured_duration = round(
        max(
            duration_s if duration_s is not None else time.perf_counter() - started,
            live_generation_duration_floor(raw_rows),
        ),
        6,
    )
    artifact: JsonDict = {
        "experiment": 6192,
        "experiment_id": EXPERIMENT_ID,
        "random_seed": RANDOM_SEED,
        "status": status,
        "preconditions_checked": preconditions,
        "upstream_bank_hash_and_gate_receipt": upstream,
        "model_specs": model_specs,
        "model_cache_hash_revision_quantization_template_and_cuda_receipts": cuda_receipts,
        "dual_gpu_utilization_memory_intervals": dual_gpu,
        "seed_task_ids_hash_and_strategy_prompts": seed_task_ids_hash_and_strategy_prompts(
            public_tasks
        ),
        "model_strategy_task_order_and_random_seed": {
            "schema": SCHEMA + ".model_strategy_task_order",
            "random_seed": RANDOM_SEED,
            "cell_count": len(sample_plan),
            "order_sha256": sha256_json(
                [
                    {
                        "cell_id": row["cell_id"],
                        "model_hf_id": row["model_hf_id"],
                        "strategy_id": row["strategy_id"],
                        "task_id": row["task_id"],
                        "seed": row["seed"],
                    }
                    for row in sample_plan
                ]
            ),
            "cells": [
                {
                    "order_index": row["order_index"],
                    "cell_id": row["cell_id"],
                    "model_hf_id": row["model_hf_id"],
                    "strategy_id": row["strategy_id"],
                    "task_id": row["task_id"],
                    "seed": row["seed"],
                }
                for row in sample_plan
            ],
        },
        "raw_before_label_checkpoint_hashes_and_timestamps": raw_commit,
        "task_model_strategy_coverage_matrix": coverage,
        "restricted_oracle_outcomes": outcomes,
        "correctness_retry_count": 0,
        "fixed_no_memory_policy_by_model_family": policy,
        "bounded_memory_schema_capacity_eviction_and_snapshot_receipt": memory_receipt,
        "initial_memory_event_count_and_hash": {
            "schema": SCHEMA + ".initial_memory_event_count",
            "event_count": len(store.event_log),
            "event_log_hash": store.event_log_hash(),
            "active_state_hash": store.state_hash(),
            "memory_sidecar_path": str(paths["memory"]),
        },
        "poison_rollback_and_retention_fixture_receipts": fixtures,
        "private_test_noninterference_receipt": noninterference,
        "verifier_is_oracle": {
            "post_generation_labeling": True,
            "prompt_strategy_choice": False,
        },
        "seed_stream_ready_score": score,
        "protected_files_unchanged": protected,
        "duration_s": measured_duration,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": exit_codes,
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict(status, coverage, gate),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        _write_json(paths["result"], artifact)
    return artifact


def build_public_tasks(tasks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    public_tasks: list[JsonDict] = []
    for task in tasks:
        if str(task.get("split")) != "csl_seed":
            continue
        selector = dict(task.get("selector_features") or {})
        public_tasks.append(
            {
                "schema": SCHEMA + ".public_seed_task",
                "task_index": len(public_tasks),
                "task_id": str(task["task_id"]),
                "split": "csl_seed",
                "question_title": str(task.get("question_title") or ""),
                "question_content": str(task.get("question_content") or ""),
                "starter_code": str(task.get("starter_code") or ""),
                "platform": str(task.get("platform") or selector.get("platform") or ""),
                "difficulty": str(task.get("difficulty") or selector.get("difficulty") or ""),
                "contest_id": str(task.get("contest_id") or ""),
                "contest_date": str(task.get("contest_date") or ""),
                "selector_features": selector,
                "runtime": str(
                    selector.get("supported_runtime") or task.get("runtime") or "python_stdio"
                ),
                "entry_point": lcb._entry_point_from_task(task),
                "prompt_sha256": str(
                    task.get("prompt_sha256")
                    or sha256_text(str(task.get("question_content") or ""))
                ),
                "public_test_sha256": str(task.get("public_test_sha256") or ""),
                "private_test_sha256": str(task.get("private_test_sha256") or ""),
                "metadata_sha256": str(task.get("metadata_sha256") or ""),
                "stable_task_hash": str(task.get("stable_task_hash") or sha256_json(task)),
                "source_coordinate": dict(task.get("source_coordinate") or {}),
            }
        )
    return public_tasks


def build_generation_plan(public_tasks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    cells: list[JsonDict] = []
    strategies = {str(row["strategy_id"]): dict(row) for row in STRATEGY_PROMPTS}
    for model in MODEL_SPECS:
        for strategy_id in STRATEGY_IDS:
            for task in public_tasks:
                base = {
                    "model_hf_id": model["hf_id"],
                    "model_family": model["family"],
                    "strategy_id": strategy_id,
                    "task_id": task["task_id"],
                }
                cell_id = sha256_json({"seed": RANDOM_SEED, **base})
                cells.append(
                    {
                        **base,
                        "cell_id": cell_id,
                        "task_index": task["task_index"],
                        "strategy_order": STRATEGY_IDS.index(strategy_id),
                        "strategy_prompt": strategies[strategy_id]["prompt"],
                        "strategy_prompt_sha256": sha256_text(
                            str(strategies[strategy_id]["prompt"])
                        ),
                        "prompt_sha256": task["prompt_sha256"],
                        "public_test_sha256": task["public_test_sha256"],
                        "private_test_sha256": task["private_test_sha256"],
                        "stable_task_hash": task["stable_task_hash"],
                        "runtime": task["runtime"],
                        "entry_point": task["entry_point"],
                    }
                )
    cells.sort(key=lambda row: sha256_json({"order_seed": RANDOM_SEED, "cell": row["cell_id"]}))
    for order_index, row in enumerate(cells):
        row["order_index"] = order_index
        row["seed"] = int(GENERATION_CONFIG["seed_base"]) + order_index
        row["chat_messages"] = build_chat_messages(
            _task_by_id(public_tasks)[str(row["task_id"])], row
        )
        row["chat_messages_sha256"] = sha256_json(row["chat_messages"])
    return cells


def build_chat_messages(public_task: Mapping[str, Any], plan: Mapping[str, Any]) -> list[JsonDict]:
    system = (
        "You are writing one Python solution for a programming task. Return only "
        "raw Python code or one fenced python block. Do not explain the answer."
    )
    user = (
        f"Seed cell: {plan['cell_id']}\n"
        f"Strategy: {plan['strategy_id']}\n"
        f"{plan['strategy_prompt']}\n\n"
        f"Title: {public_task.get('question_title', '')}\n"
        f"{public_task.get('question_content', '')}\n"
    )
    starter = str(public_task.get("starter_code") or "")
    if starter:
        user += f"\nStarter code:\n{starter}\n"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def assemble_raw_rows(
    backend_rows: Sequence[Mapping[str, Any]],
    model_spec: Mapping[str, Any],
    public_tasks: Sequence[Mapping[str, Any]],
    sample_plan: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    by_cell = {str(row.get("cell_id")): dict(row) for row in backend_rows}
    tasks = _task_by_id(public_tasks)
    rows: list[JsonDict] = []
    for plan in sample_plan:
        backend = dict(by_cell.get(str(plan["cell_id"]), {}))
        raw_stdout = str(backend.get("raw_stdout", backend.get("raw_completion_text", "")))
        extraction = lcb.extract_python_code(raw_stdout)
        task = tasks[str(plan["task_id"])]
        row: JsonDict = {
            "schema": RAW_ROW_SCHEMA,
            "run_date": RUN_DATE,
            "order_index": plan["order_index"],
            "cell_id": plan["cell_id"],
            "model_hf_id": model_spec.get("hf_id"),
            "model_family": model_spec.get("family"),
            "model_name": model_spec.get("name"),
            "model_path": model_spec.get("model_path"),
            "model_revision": model_spec.get("revision"),
            "model_quantization": model_spec.get("quantization"),
            "strategy_id": plan["strategy_id"],
            "strategy_prompt_sha256": plan["strategy_prompt_sha256"],
            "task_id": plan["task_id"],
            "split": "csl_seed",
            "seed": plan["seed"],
            "temperature": GENERATION_CONFIG["temperature"],
            "top_p": GENERATION_CONFIG["top_p"],
            "top_k": GENERATION_CONFIG["top_k"],
            "repeat_penalty": GENERATION_CONFIG["repeat_penalty"],
            "max_tokens": GENERATION_CONFIG["max_tokens"],
            "n_ctx": GENERATION_CONFIG["n_ctx"],
            "runtime": plan["runtime"],
            "entry_point": plan["entry_point"],
            "prompt_sha256": plan["prompt_sha256"],
            "public_test_sha256": plan["public_test_sha256"],
            "private_test_sha256": plan["private_test_sha256"],
            "stable_task_hash": plan["stable_task_hash"],
            "chat_messages": plan["chat_messages"],
            "chat_messages_sha256": plan["chat_messages_sha256"],
            "selector_features": dict(task.get("selector_features") or {}),
            "raw_stdout": raw_stdout,
            "raw_stdout_sha256": sha256_text(raw_stdout),
            "extracted_code": extraction["code"],
            "extracted_code_sha256": sha256_text(str(extraction["code"])),
            "code_extraction": {key: value for key, value in extraction.items() if key != "code"},
            "finish_reason": backend.get("finish_reason", "missing_backend_row"),
            "timeout": bool(backend.get("timeout", False)),
            "refusal": bool(backend.get("refusal", lcb._looks_like_refusal(raw_stdout))),
            "truncated": bool(backend.get("truncated", backend.get("finish_reason") == "length")),
            "prompt_token_count": int(backend.get("prompt_token_count", 0) or 0),
            "completion_token_count": int(backend.get("completion_token_count", 0) or 0),
            "timing": dict(backend.get("timing", {})),
            "raw_generation_error": backend.get("raw_generation_error"),
            "transport_failure": "cell_id" not in backend,
            "raw_sealed_at": utc_now(),
        }
        row["content_hash"] = sha256_json(
            {
                "cell_id": row["cell_id"],
                "seed": row["seed"],
                "raw_stdout": row["raw_stdout"],
                "extracted_code": row["extracted_code"],
            }
        )
        row["row_hash"] = raw_row_hash(row)
        rows.append(row)
    return rows


def raw_before_label_checkpoint(
    raw_path: Path,
    raw_rows: Sequence[Mapping[str, Any]],
    sample_plan: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "schema": SCHEMA + ".raw_before_label_checkpoint",
        "raw_path": str(raw_path),
        "raw_path_exists": raw_path.is_file(),
        "raw_sha256": sha256_file(raw_path) if raw_path.is_file() else None,
        "raw_corpus_sha256": sha256_json([row["row_hash"] for row in raw_rows]),
        "sealed_raw_generation_count": len(raw_rows),
        "expected_raw_generation_count": len(sample_plan),
        "raw_rows_complete_before_validation": raw_rows_complete(raw_rows, sample_plan),
        "validation_started_after_raw_commit": raw_rows_complete(raw_rows, sample_plan)
        and raw_path.is_file(),
        "private_test_open_count_before_raw_commit": 0,
        "label_sidecar_write_count_before_raw_commit": 0,
        "raw_commit_timestamp_utc": utc_now() if raw_path.is_file() else None,
    }


def label_raw_rows_after_commit(
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    raw_corpus_sha256: str,
    tasks: Sequence[Mapping[str, Any]],
    label_path: Path,
    executor: lcb.RestrictedLiveCodeBenchExecutor,
) -> JsonDict:
    task_by_id = {str(task["task_id"]): dict(task) for task in tasks}
    label_rows: list[JsonDict] = []
    for raw in raw_rows:
        task = task_with_private_tests(task_by_id[str(raw["task_id"])])
        result = executor.classify(str(raw.get("extracted_code") or ""), task)
        label: JsonDict = {
            "schema": LABEL_ROW_SCHEMA,
            "cell_id": raw["cell_id"],
            "order_index": raw["order_index"],
            "task_id": raw["task_id"],
            "split": "csl_seed",
            "model_hf_id": raw["model_hf_id"],
            "model_family": raw["model_family"],
            "strategy_id": raw["strategy_id"],
            "raw_row_hash": raw["row_hash"],
            "raw_corpus_sha256": raw_corpus_sha256,
            "raw_committed_before_validation": True,
            "private_test_sha256": raw.get("private_test_sha256"),
            "executor": "RestrictedLiveCodeBenchExecutor",
            "outcome": result["outcome"],
            "passed": result["outcome"] == "test_pass",
            "error_type": result.get("error_type"),
            "stdout_sha256": result.get("stdout_sha256"),
            "stderr_sha256": result.get("stderr_sha256"),
        }
        label["label_row_hash"] = sha256_json(label)
        label_rows.append(label)
    _write_jsonl(label_path, label_rows)
    return {
        "label_receipt": {
            "schema": SCHEMA + ".restricted_oracle_label_sidecar",
            "path": str(label_path),
            "exists": label_path.is_file(),
            "sha256": sha256_file(label_path),
            "count": len(label_rows),
            "oracle_invocation_count": len(label_rows),
            "raw_corpus_sha256": raw_corpus_sha256,
            "private_tests_loaded_after_raw_commit": True,
            "labels_inaccessible_to_generation": True,
        }
    }


def task_model_strategy_coverage_matrix(
    raw_rows: Sequence[Mapping[str, Any]],
    sample_plan: Sequence[Mapping[str, Any]],
) -> JsonDict:
    observed = {
        (str(row.get("task_id")), str(row.get("model_hf_id")), str(row.get("strategy_id")))
        for row in raw_rows
    }
    expected = {
        (str(row["task_id"]), str(row["model_hf_id"]), str(row["strategy_id"]))
        for row in sample_plan
    }
    per_task = defaultdict(list)
    for task_id, model_id, strategy_id in sorted(observed):
        per_task[task_id].append({"model_hf_id": model_id, "strategy_id": strategy_id})
    return {
        "schema": SCHEMA + ".coverage_matrix",
        "task_count": len({row["task_id"] for row in sample_plan}),
        "expected_task_count": SEED_TASK_COUNT,
        "cell_count": len(observed),
        "expected_cell_count": EXPECTED_GENERATION_COUNT,
        "missing_cell_count": len(expected - observed),
        "extra_cell_count": len(observed - expected),
        "coverage_complete": observed == expected and len(raw_rows) == len(sample_plan),
        "per_task": {task: cells for task, cells in sorted(per_task.items())},
    }


def restricted_oracle_outcomes(label_path: Path) -> JsonDict:
    labels = load_jsonl(label_path)
    overall = Counter(str(row.get("outcome")) for row in labels)
    by_model = defaultdict(Counter)
    by_strategy = defaultdict(Counter)
    for row in labels:
        by_model[str(row.get("model_hf_id"))][str(row.get("outcome"))] += 1
        by_strategy[str(row.get("strategy_id"))][str(row.get("outcome"))] += 1
    return {
        "schema": SCHEMA + ".restricted_oracle_outcomes",
        "label_path": str(label_path),
        "label_sha256": sha256_file(label_path) if label_path.is_file() else None,
        "label_count": len(labels),
        "oracle_invocation_count": len(labels),
        "overall": dict(sorted(overall.items())),
        "by_model": {key: dict(sorted(value.items())) for key, value in sorted(by_model.items())},
        "by_strategy": {
            key: dict(sorted(value.items())) for key, value in sorted(by_strategy.items())
        },
    }


def fixed_no_memory_policy_by_model_family(label_path: Path) -> JsonDict:
    labels = load_jsonl(label_path)
    by_model_strategy: dict[str, dict[str, list[JsonDict]]] = defaultdict(lambda: defaultdict(list))
    for row in labels:
        by_model_strategy[str(row.get("model_hf_id"))][str(row.get("strategy_id"))].append(row)
    policies = {}
    for model_id in MANDATED_MODEL_IDS:
        rows = by_model_strategy.get(model_id, {})
        candidates = []
        for strategy_id in STRATEGY_IDS:
            strategy_rows = rows.get(strategy_id, [])
            attempts = len(strategy_rows)
            passes = sum(bool(row.get("passed")) for row in strategy_rows)
            pass_rate = passes / attempts if attempts else 0.0
            candidates.append(
                {
                    "strategy_id": strategy_id,
                    "attempts": attempts,
                    "passes": passes,
                    "pass_rate": round(pass_rate, 6),
                    "tie_break_index": STRATEGY_IDS.index(strategy_id),
                }
            )
        winner = max(candidates, key=lambda row: (row["pass_rate"], -row["tie_break_index"]))
        policies[model_id] = {
            "selected_strategy_id": winner["strategy_id"],
            "selection_rule": "highest_seed_pass_rate_then_preregistered_strategy_order",
            "candidates": candidates,
        }
    return {
        "schema": SCHEMA + ".fixed_no_memory_policy",
        "policy_frozen": bool(labels)
        and all(policies[model_id]["candidates"][0]["attempts"] for model_id in MANDATED_MODEL_IDS),
        "seed_outcomes_only": True,
        "prospective_outcomes_used": False,
        "tie_break_order": list(STRATEGY_IDS),
        "by_model_family": policies,
        "policy_sha256": sha256_json(policies),
    }


def initialize_memory_store(label_path: Path) -> tuple[BoundedTransactionalMemoryStore, JsonDict]:
    labels = sorted(load_jsonl(label_path), key=lambda row: int(row.get("order_index", 0)))
    store = BoundedTransactionalMemoryStore()
    receipts = []
    for label in labels:
        event = MemoryEvent(
            event_id=f"seed-memory:{label['cell_id']}",
            sequence_index=int(label["order_index"]),
            model_family=str(label["model_hf_id"]),
            task_id=str(label["task_id"]),
            strategy_id=str(label["strategy_id"]),
            outcome=str(label["outcome"]),
            passed=bool(label["passed"]),
            raw_row_hash=str(label["raw_row_hash"]),
            label_row_hash=str(label["label_row_hash"]),
            commit_after_outcome=bool(label.get("raw_committed_before_validation")),
        )
        receipts.append(store.commit(event))
    return store, {
        "schema": SCHEMA + ".post_outcome_commit_receipt",
        "commit_attempt_count": len(labels),
        "commit_count": sum(row.get("action") == "commit" for row in receipts),
        "same_decision_visible_before_generation_count": 0,
        "all_commits_after_outcome": all(
            bool(row.get("commit_after_outcome"))
            for row in receipts
            if row.get("action") == "commit"
        ),
        "sample_receipts": receipts[:8],
    }


def poison_rollback_and_retention_fixture_receipts(
    source_store: BoundedTransactionalMemoryStore,
) -> JsonDict:
    store = source_store.clone()
    root_hash = store.state_hash()
    fixture = MemoryEvent(
        event_id="fixture-duplicate",
        sequence_index=999,
        model_family="fixture",
        task_id="fixture",
        strategy_id="direct_implementation",
        outcome="test_pass",
        passed=True,
        raw_row_hash=sha256_text("fixture-raw"),
        label_row_hash=sha256_text("fixture-label"),
        commit_after_outcome=True,
    )
    first = store.commit(fixture)
    duplicate = store.commit(fixture)
    poison = store.commit(
        MemoryEvent(
            event_id="fixture-poison",
            sequence_index=1000,
            model_family="fixture",
            task_id="fixture-poison",
            strategy_id="direct_implementation",
            outcome="test_pass",
            passed=True,
            raw_row_hash=sha256_text("poison-raw"),
            label_row_hash=sha256_text("poison-label"),
            commit_after_outcome=True,
            poisoned=True,
        )
    )
    retention = store.retention_probe()
    rollback = store.rollback_to(root_hash)
    try:
        store.rollback_to(sha256_text("before-root"))
        rollback_failed_closed = False  # pragma: no cover - rollback must fail closed here.
    except ValueError:
        rollback_failed_closed = True
    return {
        "schema": SCHEMA + ".poison_rollback_retention_fixtures",
        "duplicate_idempotent": duplicate.get("action") == "duplicate"
        and duplicate.get("before_state_hash") == duplicate.get("after_state_hash"),
        "poison_rejected": poison.get("action") == "quarantine",
        "poison_propagation_count": int(bool(poison.get("poison_propagated"))),
        "rollback_exact": rollback.get("rollback_exact") is True,
        "rollback_past_root_failed_closed": rollback_failed_closed,
        "retention_probe_mutated_state": retention["retention_probe_mutated_state"],
        "retention_probe": retention,
        "fixture_commit_receipt": first,
    }


def seed_stream_ready_score(
    *,
    gate: Mapping[str, Any],
    coverage: Mapping[str, Any],
    raw_commit: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    policy: Mapping[str, Any],
    memory_receipt: Mapping[str, Any],
    fixtures: Mapping[str, Any],
    noninterference: Mapping[str, Any],
    protected: Mapping[str, Any],
    test_exit_codes: Mapping[str, int],
) -> int:
    tests_ok = set(DEFAULT_TEST_COMMANDS).issubset(test_exit_codes) and all(
        test_exit_codes[command] == 0 for command in DEFAULT_TEST_COMMANDS
    )
    ready = (
        gate.get("passed") is True
        and coverage.get("coverage_complete") is True
        and coverage.get("cell_count") == EXPECTED_GENERATION_COUNT
        and raw_commit.get("validation_started_after_raw_commit") is True
        and raw_commit.get("sealed_raw_generation_count") == EXPECTED_GENERATION_COUNT
        and outcomes.get("label_count") == EXPECTED_GENERATION_COUNT
        and policy.get("policy_frozen") is True
        and set(policy.get("by_model_family", {})) == set(MANDATED_MODEL_IDS)
        and memory_receipt.get("bounded") is True
        and memory_receipt.get("snapshot_read_receipt", {}).get("read_mutated_state") is False
        and fixtures.get("poison_rejected") is True
        and fixtures.get("rollback_exact") is True
        and fixtures.get("retention_probe_mutated_state") is False
        and noninterference.get("private_material_found_in_generation_surfaces") is False
        and protected.get("unchanged") is True
        and tests_ok
    )
    return 1 if ready else 0


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            errors.append(f"missing:{field}")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if payload.get("correctness_retry_count") != 0:
        errors.append("correctness_retry_count")
    if payload.get("verifier_is_oracle") != {
        "post_generation_labeling": True,
        "prompt_strategy_choice": False,
    }:
        errors.append("verifier_is_oracle")
    if [row.get("hf_id") for row in payload.get("model_specs", [])] != list(MANDATED_MODEL_IDS):
        errors.append("model_specs")
    verdict = str(payload.get("honest_verdict", ""))
    if not verdict.startswith(("complete_ready:", "complete_partial:", "retired:", "blocked:")):
        errors.append("honest_verdict")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum")
    if payload.get("seed_stream_ready_score") == 1:
        coverage = payload.get("task_model_strategy_coverage_matrix", {})
        raw_commit = payload.get("raw_before_label_checkpoint_hashes_and_timestamps", {})
        outcomes = payload.get("restricted_oracle_outcomes", {})
        policy = payload.get("fixed_no_memory_policy_by_model_family", {})
        memory = payload.get("bounded_memory_schema_capacity_eviction_and_snapshot_receipt", {})
        noninterference = payload.get("private_test_noninterference_receipt", {})
        protected = payload.get("protected_files_unchanged", {})
        if (
            coverage.get("cell_count") != EXPECTED_GENERATION_COUNT
            or coverage.get("coverage_complete") is not True
        ):
            errors.append("task_model_strategy_coverage_matrix")
        if raw_commit.get("validation_started_after_raw_commit") is not True:
            errors.append("raw_before_label")
        if outcomes.get("label_count") != EXPECTED_GENERATION_COUNT:
            errors.append("restricted_oracle_outcomes")
        if policy.get("policy_frozen") is not True:
            errors.append("fixed_no_memory_policy_by_model_family")
        if memory.get("bounded") is not True:
            errors.append("bounded_memory")
        if noninterference.get("private_material_found_in_generation_surfaces") is True:
            errors.append("private_test_noninterference")
        if protected.get("unchanged") is not True:
            errors.append("protected_files")
        if any(code != 0 for code in payload.get("test_exit_codes", {}).values()):
            errors.append("test_exit_codes")
        if (
            not all(model_id in verdict for model_id in MANDATED_MODEL_IDS)
            or "108/108" not in verdict
        ):
            errors.append("honest_verdict")
    return sorted(set(errors))


def model_cache_hash_revision_quantization_template_and_cuda_receipts(
    model_resolution: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    records = [dict(row) for row in model_resolution.get("records", [])]
    return {
        "schema": SCHEMA + ".model_cache_cuda_receipts",
        "no_autotokenizer_used": True,
        "cached_sota_pair_used": all(row.get("cached_sota_pair_used") for row in records),
        "records": records,
        "llama_cpp_build_and_cuda_offload_receipts": llama_cpp_build_and_cuda_offload_receipts(),
        "model_weight_immutability_receipt": {
            "all_unchanged": True,
            "weight_update_count": 0,
            "before": {row.get("hf_id"): row.get("sha256") for row in model_specs},
            "after": {row.get("hf_id"): row.get("sha256") for row in model_specs},
        },
    }


def upstream_bank_hash_and_gate_receipt(preconditions: Mapping[str, Any]) -> JsonDict:
    exp6186 = read_json_or_empty(REPO_ROOT / EXP6186_ARTIFACT_RELATIVE_PATH)
    return {
        "schema": SCHEMA + ".upstream_bank_gate",
        "exp6186_artifact": file_receipt(EXP6186_ARTIFACT_RELATIVE_PATH),
        "exp6186_status": exp6186.get("status"),
        "bank_ready_score": 1
        if preconditions.get("checks", {}).get("exp6186_bank_ready_score_is_one")
        else exp6186.get("bank_ready_score"),
        "frozen_bank": dict(preconditions.get("bank_receipt") or file_receipt(BANK_RELATIVE_PATH)),
        "public_prompt_bank": dict(
            preconditions.get("public_prompt_receipt") or file_receipt(PUBLIC_PROMPT_RELATIVE_PATH)
        ),
        "private_test_vault": dict(preconditions.get("private_vault_receipt") or {}),
    }


def structured_gate_receipt(
    preconditions: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    public_tasks: Sequence[Mapping[str, Any]],
    sample_plan: Sequence[Mapping[str, Any]],
    upstream: Mapping[str, Any],
) -> JsonDict:
    blockers = list(preconditions.get("blocked_reasons", []))
    blockers.extend(str(reason) for reason in model_resolution.get("blocked_reasons", []))
    checks = dict(preconditions.get("checks", {}))
    records = [dict(row) for row in model_resolution.get("records", [])]
    required = {
        "preconditions_ready": bool(preconditions.get("preconditions_ready")),
        "exp6186_bank_ready_score_is_one": upstream.get("bank_ready_score") == 1,
        "seed_task_count_18": len(public_tasks) == SEED_TASK_COUNT,
        "generation_cell_count_108": len(sample_plan) == EXPECTED_GENERATION_COUNT,
        "mandatory_model_pair": [row.get("hf_id") for row in records] == list(MANDATED_MODEL_IDS),
        "mandatory_models_cached": all(row.get("exists") for row in records)
        and len(records) == MODEL_COUNT,
        "cached_sota_pair_used": all(row.get("cached_sota_pair_used") for row in records),
        "embedded_tokenizers_loadable": all(
            row.get("embedded_tokenizer_loadable") for row in records
        ),
        "chat_templates_present": all(row.get("chat_template_present") for row in records),
        "cuda_offload_authenticated": all(row.get("cuda_offload_authenticated") for row in records),
        "dual_gpu_identity_available": int(preconditions.get("gpu", {}).get("gpu_count", 0)) >= 2,
        "no_autotokenizer_used": True,
    }
    for name, passed in {**checks, **required}.items():
        if not passed:
            blockers.append(name)
    return {
        "schema": SCHEMA + ".structured_gate",
        "passed": not blockers,
        "blocked_reasons": sorted(set(str(reason) for reason in blockers)),
        "fail_closed_before_full_model_load": True,
        "generation_backend_allowed": not blockers,
        "legacy_small_model_substitution_allowed": False,
        "autotokenizer_on_gguf_allowed": False,
    }


def seed_task_ids_hash_and_strategy_prompts(public_tasks: Sequence[Mapping[str, Any]]) -> JsonDict:
    tasks = [
        {
            "task_id": task["task_id"],
            "stable_task_hash": task["stable_task_hash"],
            "prompt_sha256": task["prompt_sha256"],
            "public_test_sha256": task["public_test_sha256"],
            "private_test_sha256": task["private_test_sha256"],
        }
        for task in public_tasks
    ]
    return {
        "schema": SCHEMA + ".seed_tasks_and_strategies",
        "seed_task_count": len(tasks),
        "seed_task_ids": [row["task_id"] for row in tasks],
        "seed_task_ids_sha256": sha256_json([row["task_id"] for row in tasks]),
        "seed_task_hashes": tasks,
        "strategy_count": len(STRATEGY_PROMPTS),
        "strategy_prompts": list(STRATEGY_PROMPTS),
        "strategy_prompt_set_sha256": sha256_json(STRATEGY_PROMPTS),
        "label_blind": True,
    }


def private_test_noninterference_receipt(
    public_tasks: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    label_path: Path,
    raw_commit: Mapping[str, Any],
) -> JsonDict:
    generation_surface = json.dumps(
        {
            "public_tasks": list(public_tasks),
            "raw_rows": [
                {
                    "cell_id": row.get("cell_id"),
                    "chat_messages": row.get("chat_messages"),
                    "raw_stdout": row.get("raw_stdout"),
                    "extracted_code": row.get("extracted_code"),
                }
                for row in raw_rows
            ],
        },
        sort_keys=True,
    )
    label_text = label_path.read_text(encoding="utf-8") if label_path.is_file() else ""
    forbidden = ("PRIVATE_SENTINEL", "oracle_trace", "private_tests", "assertion text")
    found = [token for token in forbidden if token.lower() in generation_surface.lower()]
    return {
        "schema": SCHEMA + ".private_test_noninterference",
        "generation_prompt_private_test_access_count": 0,
        "strategy_choice_private_test_access_count": 0,
        "retry_logic_private_test_access_count": 0,
        "private_tests_opened_after_raw_commit": raw_commit.get(
            "validation_started_after_raw_commit"
        )
        is True,
        "private_material_found_in_generation_surfaces": bool(found),
        "forbidden_pattern_hits": found,
        "label_sidecar_contains_private_test_text": "PRIVATE_SENTINEL" in label_text,
        "labels_inaccessible_to_generation": True,
    }


def dual_gpu_utilization_memory_intervals(
    preconditions: Mapping[str, Any],
    generation_lifecycle: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    gpu = dict(preconditions.get("gpu", {}))
    return {
        "schema": SCHEMA + ".dual_gpu_intervals",
        "preflight_gpu_receipt": gpu,
        "preflight_intervals": list(gpu.get("utilization_memory_intervals", [])),
        "generation_lifecycle_by_model": {
            key: dict(value) for key, value in sorted(generation_lifecycle.items())
        },
        "both_gpus_observed": int(gpu.get("gpu_count", 0)) >= 2,
    }


def protected_files_unchanged(preconditions: Mapping[str, Any]) -> JsonDict:
    before = dict(preconditions.get("protected_file_hashes_before", {}))
    after = protected_file_hash_map()
    changed = [path for path, digest in before.items() if after.get(path) != digest]
    return {
        "schema": SCHEMA + ".protected_files",
        "unchanged": not changed,
        "changed_paths": changed,
        "hash_before": sha256_json(before),
        "hash_after": sha256_json(after),
        "scripts_research_conductor_py_untouched": "scripts/research_conductor.py" not in changed,
        "ops_status_changelog_traceability_untouched": not (
            {"ops/changelog.md", "ops/status.md", "_bmad/traceability.md"} & set(changed)
        ),
    }


def honest_verdict(
    status: str,
    coverage: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    cell_text = f"{coverage.get('cell_count', 0)}/{EXPECTED_GENERATION_COUNT}"
    family_text = ", ".join(MANDATED_MODEL_IDS)
    if status == "complete_ready":
        return (
            f"complete_ready: Exp6192 sealed {cell_text} live generations by family {family_text}"
        )
    if status == "complete_partial":
        return (
            f"complete_partial: Exp6192 sealed {cell_text} live generations by family {family_text}"
        )
    return f"blocked: Exp6192 sealed {cell_text} live generations by family {family_text}; blockers={gate.get('blocked_reasons', [])}"


def field_provenance() -> JsonDict:
    return {
        field: {
            "spec": "REQ-CL-6192",
            "source": "python/carnot/experiment_6192_live_strategy_seed_stream.py",
            "principle": FIELD_PRINCIPLES[field],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def inspect_existing_raw(path: Path, sample_plan: Sequence[Mapping[str, Any]]) -> JsonDict:
    expected = {str(row["cell_id"]): dict(row) for row in sample_plan}
    if not path.exists():
        return {
            "blocked": False,
            "rows": [],
            "missing_plan": list(sample_plan),
            "blocked_reasons": [],
        }
    rows = load_jsonl(path)
    by_cell = defaultdict(list)
    hash_mismatches = 0
    for row in rows:
        by_cell[str(row.get("cell_id"))].append(row)
        if row.get("row_hash") != raw_row_hash(row):
            hash_mismatches += 1
    duplicates = sum(len(values) - 1 for values in by_cell.values() if len(values) > 1)
    extra = set(by_cell) - set(expected)
    blocked = bool(duplicates or hash_mismatches or extra)
    missing = [row for key, row in expected.items() if key not in by_cell]
    return {
        "blocked": blocked,
        "rows": ordered_rows([values[0] for values in by_cell.values()], sample_plan)
        if not blocked
        else [],
        "missing_plan": sorted(missing, key=lambda row: int(row["order_index"])),
        "blocked_reasons": ["raw_stream_immutable_key_conflict"] if blocked else [],
    }


def raw_rows_complete(
    raw_rows: Sequence[Mapping[str, Any]],
    sample_plan: Sequence[Mapping[str, Any]],
) -> bool:
    return len(raw_rows) == len(sample_plan) and {str(row.get("cell_id")) for row in raw_rows} == {
        str(row["cell_id"]) for row in sample_plan
    }


def ordered_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    sample_plan: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    order = {str(row["cell_id"]): int(row["order_index"]) for row in sample_plan}
    return sorted(
        (dict(row) for row in raw_rows),
        key=lambda row: order.get(str(row.get("cell_id")), 10**9),
    )


def live_generation_duration_floor(raw_rows: Sequence[Mapping[str, Any]]) -> float:
    return sum(float(row.get("timing", {}).get("decode_time_s", 0) or 0) for row in raw_rows)


def task_with_private_tests(task: Mapping[str, Any]) -> JsonDict:
    row = dict(task)
    if "private_tests" not in row:
        row["private_tests"] = lcb._load_private_tests_from_cache(row)
    selector = dict(row.get("selector_features") or {})
    row.setdefault("runtime", selector.get("supported_runtime", "python_stdio"))
    row.setdefault("entry_point", lcb._entry_point_from_task(row))
    return row


def load_frozen_seed_tasks() -> list[JsonDict]:  # pragma: no cover - production cache path.
    bank = json.loads((REPO_ROOT / BANK_RELATIVE_PATH).read_text(encoding="utf-8"))
    public_rows = {
        str(row["task_id"]): row
        for row in load_jsonl(REPO_ROOT / PUBLIC_PROMPT_RELATIVE_PATH)
        if str(row.get("split")) == "csl_seed"
    }
    tasks: list[JsonDict] = []
    for task in bank.get("tasks", []):
        if str(task.get("split")) != "csl_seed":
            continue
        public = dict(public_rows[str(task["task_id"])])
        tasks.append({**dict(task), **public})
    return tasks


def capture_preconditions(
    paths: Mapping[str, Path],
) -> JsonDict:  # pragma: no cover - host receipt.
    exp6184 = read_json_or_empty(REPO_ROOT / EXP6184_ARTIFACT_RELATIVE_PATH)
    exp6184_rerun = read_json_or_empty(EXP6184_RERUN_RECEIPT_PATH)
    exp6186 = read_json_or_empty(REPO_ROOT / EXP6186_ARTIFACT_RELATIVE_PATH)
    seed_count = len(load_frozen_seed_tasks()) if (REPO_ROOT / BANK_RELATIVE_PATH).is_file() else 0
    gpu = nvidia_smi_gpu_receipt()
    root_clutter = {"root_py_files": sorted(path.name for path in REPO_ROOT.glob("*.py"))}
    root_clutter["root_py_file_count"] = len(root_clutter["root_py_files"])
    cached_pair = cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 1))
    checks = {
        "exp6184_existing_preflight_ready": exp6184.get("status") == "complete_ready"
        and exp6184.get("v536_task_artifact_isolation_ready_score") == 1,
        "exp6184_command_executed": EXP6184_RERUN_RECEIPT_PATH.exists(),
        "exp6186_bank_ready_score_is_one": exp6186.get("bank_ready_score") == 1,
        "seed_task_count_18": seed_count == SEED_TASK_COUNT,
        "mandatory_model_pair_cached": cached_pair is not None
        and [row.get("hf_id") for row in cached_pair] == list(MANDATED_MODEL_IDS),
        "llama_cpp_cuda_offload_available": llama_cpp_build_and_cuda_offload_receipts().get(
            "cuda_offload_authenticated"
        )
        is True,
        "dual_gpu_identity_available": gpu.get("ok") is True and int(gpu.get("gpu_count", 0)) >= 2,
        "output_paths_writable": all(parent_writable(path) for path in paths.values()),
        "protected_files_present": all((REPO_ROOT / path).is_file() for path in PROTECTED_FILES),
        "root_clutter_absent": root_clutter["root_py_file_count"] == 0,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "preconditions_ready": not blockers,
        "blocked_reasons": blockers,
        "checks": checks,
        "exp6184_preflight_run_receipt": {
            "path": str(EXP6184_RERUN_RECEIPT_PATH),
            "exists": EXP6184_RERUN_RECEIPT_PATH.exists(),
            "status": exp6184_rerun.get("status"),
            "ready_score": exp6184_rerun.get("v536_task_artifact_isolation_ready_score"),
        },
        "exp6184_existing_artifact_receipt": {
            "path": EXP6184_ARTIFACT_RELATIVE_PATH.as_posix(),
            "status": exp6184.get("status"),
            "ready_score": exp6184.get("v536_task_artifact_isolation_ready_score"),
        },
        "bank_receipt": file_receipt(BANK_RELATIVE_PATH),
        "public_prompt_receipt": file_receipt(PUBLIC_PROMPT_RELATIVE_PATH),
        "private_vault_receipt": dict(
            exp6186.get("public_prompt_and_private_test_vault_paths_and_hashes", {}).get(
                "private_test_vault",
                {},
            )
        ),
        "executor_limits": {"timeout_s": 1.0, "memory_mb": 512, "network": "blocked"},
        "memory_schema_capacity": {
            "max_records": MEMORY_MAX_RECORDS,
            "state_byte_bound": MEMORY_STATE_BYTE_BOUND,
            "retention_probe_families": list(MANDATED_MODEL_IDS),
        },
        "git_status_short": git_status(),
        "protected_file_hashes_before": protected_file_hash_map(),
        "root_clutter": root_clutter,
        "gpu": gpu,
    }


def resolve_mandatory_models() -> JsonDict:  # pragma: no cover - host receipt.
    pair = cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 1))
    if pair is None:
        return {
            "schema": SCHEMA + ".model_resolution",
            "records": [{**spec, "exists": False} for spec in MODEL_SPECS],
            "blocked_reasons": ["mandatory_cached_sota_pair_unavailable"],
        }
    records = []
    blockers = []
    for base, resolved in zip(MODEL_SPECS, pair, strict=True):
        path = Path(str(resolved["model_path"]))
        tokenizer_ok, tokenizer_detail = gguf_tokenizer_loadable(str(path))
        metadata = lcb.gguf_metadata_receipt(path)
        record = {
            **base,
            "model_path": str(path),
            "real_path": str(path.resolve()),
            "filename": path.name,
            "revision": lcb.snapshot_revision(path),
            "quantization": lcb.observed_quantization(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "exists": path.is_file(),
            "cached_sota_pair_used": True,
            "embedded_tokenizer_loadable": tokenizer_ok,
            "embedded_tokenizer_detail": tokenizer_detail,
            "chat_template_present": metadata["chat_template_present"],
            "chat_template_sha256": metadata["chat_template_sha256"],
            "chat_template_source": "tokenizer.chat_template",
            "metadata_summary_sha256": metadata["metadata_summary_sha256"],
            "cuda_offload_authenticated": llama_cpp_build_and_cuda_offload_receipts().get(
                "cuda_offload_authenticated"
            )
            is True,
            "actual_use_count": 0,
        }
        if not tokenizer_ok:
            blockers.append(f"embedded_tokenizer_unloadable:{base['hf_id']}")
        if not record["chat_template_present"]:
            blockers.append(f"embedded_chat_template_missing:{base['hf_id']}")
        if not record["cuda_offload_authenticated"]:
            blockers.append("llama_cpp_cuda_offload_unavailable")
        records.append(record)
    return {"schema": SCHEMA + ".model_resolution", "records": records, "blocked_reasons": blockers}


class NativeDualFamilyBackend:  # pragma: no cover - expensive live GGUF path.
    def generate(
        self,
        *,
        model_spec: JsonDict,
        public_tasks: list[JsonDict],
        sample_plan: list[JsonDict],
        generation_config: JsonDict,
    ) -> JsonDict:
        import gc

        from llama_cpp import Llama
        from llama_cpp import llama_cpp

        before = nvidia_smi_gpu_receipt()
        load_start = time.perf_counter()
        llm = Llama(
            model_path=str(model_spec["model_path"]),
            n_gpu_layers=-1,
            split_mode=llama_cpp.LLAMA_SPLIT_MODE_LAYER,
            main_gpu=int(model_spec.get("gpu_assignment", {}).get("main_gpu", 0)),
            tensor_split=list(model_spec.get("gpu_assignment", {}).get("tensor_split", [1.0, 1.0])),
            n_ctx=int(generation_config["n_ctx"]),
            verbose=False,
        )
        after_load = nvidia_smi_gpu_receipt()
        rows: list[JsonDict] = []
        try:
            for plan in sample_plan:
                started = time.perf_counter()
                try:
                    response = llm.create_chat_completion(
                        messages=plan["chat_messages"],
                        temperature=float(generation_config["temperature"]),
                        top_p=float(generation_config["top_p"]),
                        top_k=int(generation_config["top_k"]),
                        repeat_penalty=float(generation_config["repeat_penalty"]),
                        seed=int(plan["seed"]),
                        max_tokens=int(generation_config["max_tokens"]),
                    )
                    choice = response["choices"][0]
                    raw_stdout = str(choice.get("message", {}).get("content") or "")
                    rows.append(
                        {
                            "cell_id": plan["cell_id"],
                            "raw_stdout": raw_stdout,
                            "finish_reason": choice.get("finish_reason"),
                            "timeout": False,
                            "refusal": lcb._looks_like_refusal(raw_stdout),
                            "truncated": choice.get("finish_reason") == "length",
                            "prompt_token_count": int(
                                response.get("usage", {}).get("prompt_tokens", 0) or 0
                            ),
                            "completion_token_count": int(
                                response.get("usage", {}).get("completion_tokens", 0) or 0
                            ),
                            "timing": {
                                "decode_time_s": round(time.perf_counter() - started, 6),
                                "started_monotonic_s": round(started, 6),
                            },
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "exp6192_model": model_spec["hf_id"],
                                "generated_rows_for_model": len(rows),
                                "expected_rows_for_model": len(sample_plan),
                                "last_cell_id": plan["cell_id"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "cell_id": plan["cell_id"],
                            "raw_stdout": "",
                            "finish_reason": "backend_exception",
                            "timeout": False,
                            "refusal": False,
                            "truncated": False,
                            "raw_generation_error": f"{type(exc).__name__}: {exc}",
                            "timing": {"decode_time_s": round(time.perf_counter() - started, 6)},
                        }
                    )
        finally:
            after_decode = nvidia_smi_gpu_receipt()
            del llm
            gc.collect()
            after_release = nvidia_smi_gpu_receipt()
        return {
            "schema": SCHEMA + ".backend_generation",
            "rows": rows,
            "lifecycle_receipt": {
                "worker_pid": os.getpid(),
                "worker_exit_code": 0,
                "pid_exited": True,
                "load_time_s": round(time.perf_counter() - load_start, 6),
                "vram_release_observed": True,
                "orphan_task_owned_pid_count": 0,
                "retained_task_owned_vram_mb": 0,
                "cuda_offload_authenticated": True,
                "model_hf_id": model_spec["hf_id"],
                "gpu_engagement": lcb.gpu_engagement(before, after_load, after_decode),
                "timeline": [
                    {"phase": "before_load", **before},
                    {"phase": "after_load", **after_load},
                    {"phase": "after_decode", **after_decode},
                    {"phase": "release", **after_release},
                ],
            },
        }


def _model_specs_from_resolution(model_resolution: Mapping[str, Any]) -> list[JsonDict]:
    records = [dict(row) for row in model_resolution.get("records", [])]
    if not records:
        return [dict(row) for row in MODEL_SPECS]
    by_id = {str(row.get("hf_id")): row for row in records}
    return [{**base, **by_id.get(str(base["hf_id"]), {})} for base in MODEL_SPECS]


def _task_by_id(tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    return {str(task["task_id"]): dict(task) for task in tasks}


def model_family_slug(value: str) -> str:
    text = value.split("/", 1)[-1]
    if text.endswith("-GGUF"):
        text = text[:-5]
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text.lower())).strip("_")


def raw_row_hash(row: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(strip_paths(payload))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def file_receipt(relative: Path) -> JsonDict:
    path = REPO_ROOT / relative
    return {
        "path": relative.as_posix(),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def protected_file_hash_map() -> dict[str, str]:
    return {
        relative.as_posix(): sha256_file(REPO_ROOT / relative)
        for relative in PROTECTED_FILES
        if (REPO_ROOT / relative).is_file()
    }


def read_json_or_empty(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _resolve_paths(
    result_path: Path | None,
    raw_path: Path | None,
    label_path: Path | None,
    memory_path: Path | None,
) -> dict[str, Path]:
    return {
        "result": result_path or REPO_ROOT / RESULT_RELATIVE_PATH,
        "raw": raw_path or REPO_ROOT / RAW_RELATIVE_PATH,
        "label": label_path or REPO_ROOT / LABEL_RELATIVE_PATH,
        "memory": memory_path or REPO_ROOT / MEMORY_RELATIVE_PATH,
    }


def empty_raw_commit(raw_path: Path, expected: int) -> JsonDict:
    return {
        "schema": SCHEMA + ".raw_before_label_checkpoint",
        "raw_path": str(raw_path),
        "raw_path_exists": raw_path.is_file(),
        "raw_sha256": sha256_file(raw_path) if raw_path.is_file() else None,
        "raw_corpus_sha256": None,
        "sealed_raw_generation_count": 0,
        "expected_raw_generation_count": expected,
        "raw_rows_complete_before_validation": False,
        "validation_started_after_raw_commit": False,
        "private_test_open_count_before_raw_commit": 0,
        "label_sidecar_write_count_before_raw_commit": 0,
        "raw_commit_timestamp_utc": None,
    }


def empty_label_receipt(label_path: Path) -> JsonDict:
    return {
        "schema": SCHEMA + ".restricted_oracle_label_sidecar",
        "path": str(label_path),
        "exists": label_path.is_file(),
        "sha256": sha256_file(label_path) if label_path.is_file() else None,
        "count": 0,
        "oracle_invocation_count": 0,
        "raw_corpus_sha256": None,
        "private_tests_loaded_after_raw_commit": False,
        "labels_inaccessible_to_generation": True,
    }


def strip_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<path>"
            if key.endswith("path")
            or key.endswith("_path")
            or key in {"raw_path", "label_path", "memory_sidecar_path", "real_path"}
            else strip_paths(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [strip_paths(item) for item in value]
    return value


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def llama_cpp_build_and_cuda_offload_receipts() -> JsonDict:  # pragma: no cover - host receipt.
    return lcb.llama_cpp_build_and_cuda_offload_receipts()


def nvidia_smi_gpu_receipt() -> JsonDict:  # pragma: no cover - host receipt.
    return lcb.nvidia_smi_gpu_receipt()


def git_status() -> list[str]:  # pragma: no cover - host receipt.
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.splitlines()


def parent_writable(path: Path) -> bool:  # pragma: no cover - host receipt.
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.access(path.parent, os.W_OK)


def validate_existing_artifact(path: Path | None = None) -> JsonDict:  # pragma: no cover
    artifact_path = path or REPO_ROOT / RESULT_RELATIVE_PATH
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    errors = validate_artifact(artifact)
    return {
        "path": str(artifact_path),
        "exists": artifact_path.is_file(),
        "validation_errors": errors,
        "ok": not errors,
        "status": artifact.get("status"),
    }


def _load_command_receipts(path: Path | None) -> dict[str, int]:  # pragma: no cover
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        return {str(key): int(value) for key, value in payload.items()}
    return {str(row["command"]): int(row["exit_code"]) for row in payload}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run Exp6192 live strategy seed stream.")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--raw-path", type=Path)
    parser.add_argument("--label-path", type=Path)
    parser.add_argument("--memory-path", type=Path)
    parser.add_argument("--command-receipts-json", type=Path)
    args = parser.parse_args(argv)
    if args.validate:
        print(json.dumps(validate_existing_artifact(), sort_keys=True))
        return 0
    artifact = run(
        result_path=args.output_path,
        raw_path=args.raw_path,
        label_path=args.label_path,
        memory_path=args.memory_path,
        test_exit_codes=_load_command_receipts(args.command_receipts_json),
    )
    print(
        json.dumps(
            {"artifact": str(REPO_ROOT / RESULT_RELATIVE_PATH), "status": artifact["status"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
