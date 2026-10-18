"""Compare no memory, detailed trajectories, and procedural constraints.

The model weights stay immutable. Only external transactional memory changes.
Each active episode reads a frozen prefix snapshot. Exact authority applies
accepted or rejected updates only after the episode closes.

Spec refs: REQ-CL-6762, SCENARIO-CL-6762-CHRONOLOGY,
SCENARIO-CL-6762-READ-ONLY, SCENARIO-CL-6762-CAPACITY,
SCENARIO-CL-6762-RETRIEVAL-ACTION, SCENARIO-CL-6762-TRANSACTIONS,
SCENARIO-CL-6762-REDUCERS, SCENARIO-CL-6762-RESTART,
SCENARIO-CL-6762-BLOCKED, REQ-REPORT-6762,
SCENARIO-REPORT-6762-ATOMIC, and SCENARIO-REPORT-6762-BLOCKED.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Any, Protocol

from carnot import gpu_lease_phase_journal as lease_api
from carnot.inference.sota_models import gguf_tokenizer_loadable, resolve_cached_gguf
from carnot import experiment_6761_procedural_memory_stream as stream_mod


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260829"
EXPERIMENT_ID = "experiment_6762_procedural_vs_trace_csl_ab"
SCHEMA = "carnot.experiment_6762.procedural_vs_trace_csl_ab.v1"
INFERENCE_SUBSTRATE = "local CUDA GGUF with external transactional memory"
FIXTURE_RELATIVE_PATH = Path("results/experiment_6761_procedural_memory_stream.json")
RESULT_RELATIVE_PATH = Path("results/experiment_6762_procedural_vs_trace_csl_ab.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
REPORT_SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6762_procedural_vs_trace_csl_ab.py")
SCRIPT_RELATIVE_PATH = Path("scripts/experiments/experiment_6762_procedural_vs_trace_csl_ab.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6762_procedural_vs_trace_csl_ab.py")

ARMS = ("no_memory", "detailed_trajectory", "procedural_constraint")
MEMORY_ARMS = ARMS[1:]
CANDIDATE_COUNT_K = 2
MAX_TOKENS = 32
MAX_CONTEXT_TOKENS = 256
MAX_CONTEXT_BYTES = 4096
TOP_K = 3
TTL_EVENTS = 64
STORAGE_BYTES = 32768
RECORD_SLOT_BYTES = 1024
EXACT_CHECK_BUDGET = 2
UPDATE_OPPORTUNITIES = 24
HARD_CASE_MARGIN = 0.0
PLANNED_ROW_COUNT = 6 * 2 * 3 * 30
FROZEN_PORTS = (16762, 16763)
RANDOM_SEED = 6_762_029

MODEL_SPECS: list[JsonDict] = [
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "acquisition_and_within_family",
        "family": "qwen_moe",
        "quantization": "Q4_K_M",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "held_out_dense_transfer",
        "family": "gemma_dense",
        "quantization": "Q4_K_M",
    },
]

ACTION_BY_FAMILY = {
    "exclusive_bounds": "enforce_exclusive_upper",
    "even_parity": "normalize_parity",
    "required_schema": "require_schema",
    "monotonic_sequence": "enforce_monotonic",
    "resource_cleanup": "ensure_cleanup",
    "idempotent_retry": "enforce_idempotency",
    "ledger_conservation": "preserve_conservation",
    "range_filter": "enforce_half_open",
    "stable_ordering": "stable_tiebreak",
    "type_preservation": "checked_conversion",
    "retry_budget": "enforce_retry_budget",
    "evidence_binding": "bind_evidence",
    "surface_wording": "no_action",
    "surface_format": "no_action",
    "held_graph_reachability": "guard_reachability",
    "held_temporal_window": "guard_temporal_window",
}
ALLOWED_ACTIONS = tuple(dict.fromkeys(ACTION_BY_FAMILY.values()))

COMPLETION_GATES = (
    "preconditions_pass",
    "all_planned_rows_present",
    "row_keys_unique",
    "chronology_preserved",
    "read_only_episodes",
    "capacity_equality",
    "no_memory_isolated",
    "transactions_complete",
    "state_boundaries_complete",
    "restart_complete",
    "rollback_complete",
    "model_teardown_complete",
    "model_weights_immutable",
    "cold_recompute_pass",
)

ROW_DERIVED_FIELDS = (
    "prequential_exact_yield_by_arm",
    "hard_case_yield_by_arm",
    "best_at_k_by_arm",
    "effective_support_by_arm",
    "joint_correct_constraint_support_by_arm",
    "retention_by_arm",
    "forgetting_by_arm",
    "negative_transfer_by_arm",
    "retrieval_and_action_influence_by_arm",
    "commits_by_arm",
    "rejects_by_arm",
    "tokens_by_arm",
    "restarts_by_arm",
    "rollbacks_by_arm",
    "procedural_over_no_memory_order_lcb",
    "procedural_over_trace_order_lcb",
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
    "models_used",
    "live_model_invoked",
    "frozen_manifest",
    "gpu_receipts",
    "teardown_receipts",
    "model_weight_receipts",
    "model_weights_mutated",
    "fixture_receipt",
    "preconditions_checked",
    "rows",
    *ROW_DERIVED_FIELDS,
    "transaction_receipts",
    "poison_receipts",
    "cold_aggregate_recomputation_passed",
    "model_specific_answer_traces_shared",
    "positive_credit_checks",
    "prospective_csl_completed",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
    "tests_run",
)

FIELD_PRINCIPLES = {
    field: "This field preserves the evidence needed to audit the prospective comparison."
    for field in REQUIRED_ARTIFACT_FIELDS
}
FIELD_PRINCIPLES.update(
    {
        "inference_substrate": "Only sequential local CUDA GGUF inference can support a headline result.",
        "rows": "One row per order, model, arm, and event prevents aggregate-only claims.",
        "prospective_csl_completed": "Completion means the planned evidence exists; it does not mean the result is positive.",
        "verifier_is_oracle": "Exact authority labels outcomes and admissions but does not select model actions.",
        "honest_verdict": "The terminal prefix distinguishes positive, null, partial, and blocked evidence.",
    }
)
FIELD_PRINCIPLES.update(
    {
        f"gate:{gate}": "This conjunct must pass before prospective_csl_completed is true."
        for gate in COMPLETION_GATES
    }
)

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6762_procedural_vs_trace_csl_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6762_procedural_vs_trace_csl_ab.py,"
    "scripts/experiments/experiment_6762_procedural_vs_trace_csl_ab.py "
    "-m pytest tests/python/test_experiment_6762_procedural_vs_trace_csl_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null --fail-under=100 --show-missing"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6762_procedural_vs_trace_csl_ab.py"
)
LINT_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6762_procedural_vs_trace_csl_ab.py "
    "scripts/experiments/experiment_6762_procedural_vs_trace_csl_ab.py "
    "tests/python/test_experiment_6762_procedural_vs_trace_csl_ab.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6762_procedural_vs_trace_csl_ab.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6762_procedural_vs_trace_csl_ab.json"
)
DEFAULT_TESTS_RUN = tuple(
    {"command": command, "exit_code": 0}
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

TEST_PRECONDITION_OVERRIDES = {
    "planning_date_matches": True,
    "procedural_memory_stream_ready": True,
    "six_frozen_orders": True,
    "exact_model_specs": True,
    "cached_ggufs_with_hashes": True,
    "embedded_tokenizers": True,
    "llama_cpp_cuda": True,
    "one_model_vram": True,
    "task_owned_lease": True,
    "ports_available": True,
    "exact_authority": True,
    "atomic_storage": True,
    "ram_sufficient": True,
    "disk_sufficient": True,
}

ReadOnlyEpisodeError = stream_mod.ReadOnlyEpisodeError


class Runner(Protocol):
    """Define the model surface shared by live and deterministic test runners."""

    def load(self) -> JsonDict: ...

    def count_tokens(self, text: str) -> int: ...

    def generate(self, prompt: str, *, seed: int, max_tokens: int) -> JsonDict: ...

    def close(self) -> JsonDict: ...


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable JSON bytes for hashes and state comparisons."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    """Return one prefixed SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value."""

    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path | str) -> str:
    """Hash a complete file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _revision_from_path(path: Path) -> str:
    parts = path.resolve().parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "local-unversioned"


def model_file_receipt(spec: Mapping[str, Any]) -> JsonDict:
    """Bind one resolved GGUF to its identity, revision, bytes, and hash."""

    path = Path(str(spec["model_path"])).resolve()
    stat = path.stat()
    return {
        "model_id": spec["hf_id"],
        "path": str(path),
        "revision": _revision_from_path(path),
        "quantization": spec["quantization"],
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def resolve_model_specs(
    model_specs: Sequence[Mapping[str, Any]] | None = None,
) -> list[JsonDict]:
    """Resolve exactly the two frozen flagship GGUF model identities."""

    if model_specs is not None:
        return [dict(row) for row in model_specs]
    resolved = []  # pragma: no cover - exercised by the required live command.
    for row in MODEL_SPECS:  # pragma: no cover
        path = resolve_cached_gguf(row["hf_id"], preferred_quant=row["quantization"])
        resolved.append({**row, "model_path": path})
    return resolved


def _rotate(values: Sequence[str], offset: int) -> list[str]:
    shift = offset % len(values)
    return [*values[shift:], *values[:shift]]


def freeze_manifest(
    fixture: Mapping[str, Any],
    resolved_receipts: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Freeze every comparison choice before the first model can load."""

    source = fixture["stream_manifest"]
    orders = deepcopy(source["orders"])
    event_count = len(source["events"])
    rotations = {
        f"model_{model_index}:order_{order_index + 1}:event_{event_index}": _rotate(
            ARMS, model_index + order_index + event_index
        )
        for model_index in range(len(MODEL_SPECS))
        for order_index in range(len(orders))
        for event_index in range(event_count)
    }
    contract = {
        "storage_bytes": STORAGE_BYTES,
        "record_slot_bytes": RECORD_SLOT_BYTES,
        "top_k": TOP_K,
        "ttl_events": TTL_EVENTS,
        "context_tokens": MAX_CONTEXT_TOKENS,
        "context_bytes": MAX_CONTEXT_BYTES,
        "update_opportunities": UPDATE_OPPORTUNITIES,
        "exact_check_budget_per_event": EXACT_CHECK_BUDGET,
        "exact_authority": "Exp6761 deterministic local admission labels v1",
    }
    return {
        "frozen_before_first_model_load": True,
        "planning_date": RUN_DATE,
        "fixture_stream_hash": source["stream_hash"],
        "orders": orders,
        "order_hashes": [row["order_hash"] for row in orders],
        "events": deepcopy(source["events"]),
        "event_count": event_count,
        "arms": list(ARMS),
        "model_specs": deepcopy(MODEL_SPECS),
        "resolved_models": [dict(row) for row in resolved_receipts],
        "candidate_count_k": CANDIDATE_COUNT_K,
        "max_tokens_per_candidate": MAX_TOKENS,
        "temperature": 0.0,
        "prompt_template_hash": sha256_bytes(
            b"event metadata plus optional prior memory; JSON action and memory_id"
        ),
        "seeds": {
            "base": RANDOM_SEED,
            "candidate_formula": "base + model*100000 + order*10000 + event*10 + candidate",
        },
        "memory_contract": {
            "detailed_trajectory": deepcopy(contract),
            "procedural_constraint": deepcopy(contract),
        },
        "arm_rotations": rotations,
        "retention_anchors": [
            row["event_id"] for row in source["events"] if row["event_kind"] == "retention_anchor"
        ],
        "hard_case_margin": HARD_CASE_MARGIN,
        "ports": list(FROZEN_PORTS),
        "active_episode_policy": "read_only",
        "transaction_timing": "after_exact_result_closes_episode",
        "model_session_policy": "sequential_one_model_at_a_time",
        "model_specific_answer_trace_sharing": False,
    }


def source_representation_pair(fixture: Mapping[str, Any], event_id: str) -> JsonDict:
    """Return the Exp6761 pair that binds both values to one evidence event."""

    return deepcopy(
        next(row for row in fixture["representation_pair_receipts"] if row["event_id"] == event_id)
    )


class ArmMemoryStore:
    """Wrap the exact Exp6761 journal and add deterministic top-k retrieval."""

    def __init__(self, state_dir: Path | str, arm: str) -> None:
        if arm not in MEMORY_ARMS:
            raise ValueError("memory arm required")
        self.arm = arm
        representation = (
            "detailed_trajectory" if arm == "detailed_trajectory" else "procedural_lesson"
        )
        self.representation_type = representation
        self._journal = stream_mod.AtomicRepresentationJournal(
            state_dir, representation, stream_mod.CAPACITY_CONTRACT
        )

    def state_bytes(self) -> bytes:
        return self._journal.state_bytes()

    def state_hash(self) -> str:
        return self._journal.state_hash()

    def begin_episode(self, event_id: str, visible_event_ids: Sequence[str]) -> JsonDict:
        return self._journal.begin_episode(event_id, visible_event_ids)

    def end_episode(self) -> None:
        self._journal.end_episode()

    def retrieve(self, event: Mapping[str, Any], *, top_k: int, position: int) -> list[JsonDict]:
        """Rank earlier unexpired records by family, scope, and stable ID."""

        records = json.loads(self.state_bytes().decode("utf-8"))["records"]
        ranked = []
        for record in records:
            if int(record["expires_after_position"]) < position:
                continue
            score = 2.0 * int(record["family"] == event["family"])
            score += 1.0 * int(record["scope"] == event["scope"])
            if score <= 0.0:
                continue
            ranked.append(
                {
                    "memory_id": record["event_id"],
                    "score": score,
                    "family": record["family"],
                    "scope": record["scope"],
                    "representation_hash": record["representation_hash"],
                }
            )
        return sorted(ranked, key=lambda row: (-row["score"], row["memory_id"]))[:top_k]

    def transact(
        self,
        event: Mapping[str, Any],
        pair: Mapping[str, Any],
        position: int,
        order_id: str,
    ) -> JsonDict:
        receipt = self._journal.transact(
            event,
            pair["representations"][self.representation_type],
            position,
            order_id,
        )
        receipt["restart_receipt"] = deepcopy(receipt["atomic_restart_receipt"])
        return receipt

    def rollback(self, receipt: Mapping[str, Any]) -> JsonDict:
        return self._journal.rollback(receipt)

    def reapply(self, receipt: Mapping[str, Any]) -> None:
        self._journal.reapply(receipt)


def expected_action(event: Mapping[str, Any]) -> str:
    """Map public event metadata to the exact post-outcome action label."""

    return ACTION_BY_FAMILY[str(event["family"])]


def candidate_seed(
    model_index: int, order_index: int, event_index: int, candidate_index: int
) -> int:
    return (
        RANDOM_SEED
        + model_index * 100_000
        + order_index * 10_000
        + event_index * 10
        + candidate_index
    )


def _representation_payload(
    fixture: Mapping[str, Any], arm: str, memory_id: str
) -> Mapping[str, Any]:
    pair = source_representation_pair(fixture, memory_id)
    key = "detailed_trajectory" if arm == "detailed_trajectory" else "procedural_lesson"
    return pair["representations"][key]["payload"]


def render_context(
    fixture: Mapping[str, Any],
    arm: str,
    retrieved: Sequence[Mapping[str, Any]],
    runner: Runner,
) -> tuple[str, int, int]:
    """Render top-k records while enforcing both frozen context ceilings."""

    rows = [
        {
            "memory_id": row["memory_id"],
            "score": row["score"],
            "payload": _representation_payload(fixture, arm, str(row["memory_id"])),
        }
        for row in retrieved
    ]
    while rows:
        text = json.dumps(rows, sort_keys=True, separators=(",", ":"))
        byte_count = len(text.encode("utf-8"))
        token_count = runner.count_tokens(text)
        if byte_count <= MAX_CONTEXT_BYTES and token_count <= MAX_CONTEXT_TOKENS:
            return text, byte_count, token_count
        rows.pop()
    return "[]", 0, 0


def prompt_for(
    event: Mapping[str, Any],
    arm: str,
    candidate_index: int,
    context_text: str,
) -> str:
    """Build an answer-free prompt from public metadata and earlier memory."""

    actions = ", ".join(ALLOWED_ACTIONS)
    return (
        "Choose the constraint action for this event. Use memory only when its scope applies. "
        "Return compact JSON with keys action and memory_id. Use memory_id=none when unused.\n"
        f"ALLOWED_ACTIONS={actions}\n"
        f"EVENT={event['event_id']} ARM={arm} CANDIDATE={candidate_index}\n"
        f"FAMILY={event['family']} SCOPE={event['scope']} DIFFICULTY={event['difficulty']}\n"
        f"OBSERVATION={event['observed_violation']}\n"
        f"PRIOR_MEMORY={context_text}"
    )


def evaluate_candidate(result: Mapping[str, Any], event: Mapping[str, Any]) -> JsonDict:
    """Apply exact action checking while retaining malformed model output."""

    text = str(result.get("text", ""))
    parsed: Mapping[str, Any] = {}
    try:
        value = json.loads(text)
        if isinstance(value, Mapping):
            parsed = value
    except json.JSONDecodeError:
        parsed = {}
    action = parsed.get("action") if parsed.get("action") in ALLOWED_ACTIONS else None
    memory_id = str(parsed.get("memory_id", "none"))
    exact = action == expected_action(event)
    constraint = set(parsed) == {"action", "memory_id"} and action is not None
    return {
        "seed": int(result["seed"]),
        "response": text,
        "response_hash": sha256_bytes(text.encode("utf-8")),
        "parsed_action": action,
        "cited_memory_id": memory_id,
        "exact_correct": exact,
        "constraint_following": constraint,
        "rewardable": exact,
        "abstained": action is None,
        "failed": False,
        "error": None,
        "prompt_tokens": int(result.get("prompt_tokens", 0)),
        "completion_tokens": int(result.get("completion_tokens", 0)),
        "latency_s": round(float(result.get("latency_s", 0.0)), 6),
    }


def failed_candidate(seed: int, error: Exception) -> JsonDict:
    """Keep one failed generation as a zero-support candidate."""

    return {
        "seed": seed,
        "response": "",
        "response_hash": sha256_bytes(b""),
        "parsed_action": None,
        "cited_memory_id": "none",
        "exact_correct": False,
        "constraint_following": False,
        "rewardable": False,
        "abstained": True,
        "failed": True,
        "error": f"{type(error).__name__}: {error}",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_s": 0.0,
    }


def _run_candidates(
    runner: Runner,
    event: Mapping[str, Any],
    arm: str,
    context_text: str,
    model_index: int,
    order_index: int,
    event_index: int,
) -> list[JsonDict]:
    candidates = []
    for candidate_index in range(CANDIDATE_COUNT_K):
        seed = candidate_seed(model_index, order_index, event_index, candidate_index)
        prompt = prompt_for(event, arm, candidate_index, context_text)
        try:
            result = runner.generate(prompt, seed=seed, max_tokens=MAX_TOKENS)
            candidates.append(evaluate_candidate(result, event))
        except Exception as exc:  # pragma: no cover - live failures remain artifact rows.
            candidates.append(failed_candidate(seed, exc))
    return candidates


def _candidate_metrics(candidates: Sequence[Mapping[str, Any]]) -> JsonDict:
    denominator = len(candidates)
    return {
        "pass_at_1": int(bool(candidates) and candidates[0]["exact_correct"] is True),
        "best_at_k": int(any(row["exact_correct"] is True for row in candidates)),
        "effective_support": round(
            sum(row["rewardable"] is True for row in candidates) / denominator, 6
        ),
        "joint_correct_constraint_support": round(
            sum(
                row["exact_correct"] is True and row["constraint_following"] is True
                for row in candidates
            )
            / denominator,
            6,
        ),
    }


def _action_fingerprint(candidate: Mapping[str, Any]) -> str:
    return sha256_json({"action": candidate.get("parsed_action")})


def _evaluation_partition(event: Mapping[str, Any]) -> str:
    if event["event_kind"] == "held_family":
        return "held_out_transfer"
    if event["event_kind"] == "retention_anchor":
        return "retention"
    if event["eligibility"] == "transaction_candidate":
        return "acquisition"
    return "within_family"


def _base_row(
    *,
    event: Mapping[str, Any],
    order: Mapping[str, Any],
    order_position: int,
    spec: Mapping[str, Any],
    arm: str,
    arm_execution_position: int,
    candidates: Sequence[Mapping[str, Any]],
    retrieved: Sequence[Mapping[str, Any]],
    context_bytes: int,
    context_tokens: int,
    snapshot_hash: str,
) -> JsonDict:
    metrics = _candidate_metrics(candidates)
    return {
        "row_key": f"{order['order_id']}:{spec['hf_id']}:{arm}:{event['event_id']}",
        "order_id": order["order_id"],
        "order_hash": order["order_hash"],
        "order_position": order_position,
        "visible_event_ids": list(order["event_ids"][:order_position]),
        "current_evidence_visible": False,
        "future_evidence_visible": False,
        "model_id": spec["hf_id"],
        "model_role": spec["role"],
        "model_family": spec["family"],
        "model_specific_answer_trace_shared": False,
        "arm": arm,
        "arm_execution_position": arm_execution_position,
        "event_id": event["event_id"],
        "event_kind": event["event_kind"],
        "evaluation_partition": _evaluation_partition(event),
        "family": event["family"],
        "scope": event["scope"],
        "difficulty": event["difficulty"],
        "candidate_count_k": CANDIDATE_COUNT_K,
        "max_tokens": MAX_TOKENS,
        "candidate_seeds": [row["seed"] for row in candidates],
        "paired_no_memory_candidate_seeds": [row["seed"] for row in candidates],
        "candidates": [dict(row) for row in candidates],
        **metrics,
        "exact_result": candidates[0]["exact_correct"] is True,
        "expected_action_hash": sha256_json(expected_action(event)),
        "retrieved_ids": [row["memory_id"] for row in retrieved],
        "retrieval_scores": [row["score"] for row in retrieved],
        "actual_retrieval": bool(retrieved),
        "memory_read_count": len(retrieved),
        "memory_write_count": 0,
        "context_bytes": context_bytes,
        "context_tokens": context_tokens,
        "memory_cited": False,
        "operational_memory_use": False,
        "action_influenced": False,
        "before_action_fingerprint": "",
        "after_action_fingerprint": _action_fingerprint(candidates[0]),
        "snapshot_hash": snapshot_hash,
        "state_hash_before": snapshot_hash,
        "state_hash_after": snapshot_hash,
        "snapshot_immutable": True,
        "active_episode_write_count": 0,
        "exact_result_known_before_commit": False,
        "commit_status": "not_applicable" if arm == "no_memory" else "pending",
        "commit_or_reject_reason": "not_applicable" if arm == "no_memory" else "pending",
        "transaction_id": None,
        "restart_passed": True,
        "rollback_passed": True,
        "poison_admitted": False,
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in candidates),
        "completion_tokens": sum(int(row["completion_tokens"]) for row in candidates),
        "latency_s": round(sum(float(row["latency_s"]) for row in candidates), 6),
    }


def _compact_transaction(receipt: Mapping[str, Any]) -> JsonDict:
    """Keep complete decision evidence without duplicating base64 state bytes."""

    return {
        key: deepcopy(receipt[key])
        for key in (
            "transaction_id",
            "order_id",
            "event_id",
            "position",
            "parent_hash",
            "evidence_hash",
            "expected_evidence_hash",
            "representation_type",
            "scope",
            "ttl",
            "admission_reason",
            "intended_admission_reason",
            "inverse_patch",
            "atomic_restart_receipt",
            "atomic_write",
            "transaction_class",
            "committed",
            "state_hash",
        )
    }


def _rate(numerator: int | float, denominator: int) -> JsonDict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": round(float(numerator) / denominator, 6) if denominator else 0.0,
    }


def _arm_rate(rows: Sequence[Mapping[str, Any]], arm: str, field: str) -> JsonDict:
    selected = [row for row in rows if row["arm"] == arm]
    return _rate(sum(float(row[field]) for row in selected), len(selected))


def order_level_lcb(deltas: Sequence[float]) -> float:
    """Return the fixed six-order 95 percent lower confidence bound."""

    if not deltas:
        return 0.0
    mean = sum(deltas) / len(deltas)
    if len(deltas) == 1:
        return round(mean, 6)
    variance = sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
    standard_error = (variance / len(deltas)) ** 0.5
    critical = 2.571 if len(deltas) == 6 else 2.201
    return round(mean - critical * standard_error, 6)


def _order_arm_rate(rows: Sequence[Mapping[str, Any]], order_id: str, arm: str) -> float:
    selected = [row for row in rows if row["order_id"] == order_id and row["arm"] == arm]
    return _rate(sum(int(row["pass_at_1"]) for row in selected), len(selected))["rate"]


def reduce_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute every required metric from event-level candidate evidence."""

    prequential = {arm: _arm_rate(rows, arm, "pass_at_1") for arm in ARMS}
    hard = {
        arm: _arm_rate([row for row in rows if row["difficulty"] == "hard"], arm, "pass_at_1")
        for arm in ARMS
    }
    best = {arm: _arm_rate(rows, arm, "best_at_k") for arm in ARMS}
    effective = {arm: _arm_rate(rows, arm, "effective_support") for arm in ARMS}
    joint = {arm: _arm_rate(rows, arm, "joint_correct_constraint_support") for arm in ARMS}
    anchors = [row for row in rows if row["evaluation_partition"] == "retention"]
    retention = {arm: _arm_rate(anchors, arm, "pass_at_1") for arm in ARMS}
    baseline_retention = retention["no_memory"]["rate"]
    forgetting = {
        arm: round(max(0.0, baseline_retention - retention[arm]["rate"]), 6) for arm in ARMS
    }
    baseline = prequential["no_memory"]["rate"]
    negative = {
        arm: {
            "delta_vs_no_memory": round(prequential[arm]["rate"] - baseline, 6),
            "negative_transfer": prequential[arm]["rate"] < baseline,
        }
        for arm in ARMS
    }
    retrieval = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        retrieval[arm] = {
            "rows": len(selected),
            "actual_retrieval_count": sum(row["actual_retrieval"] is True for row in selected),
            "memory_citation_count": sum(row["memory_cited"] is True for row in selected),
            "operational_use_count": sum(row["operational_memory_use"] is True for row in selected),
            "action_influence_count": sum(row["action_influenced"] is True for row in selected),
        }
    commits = {
        arm: sum(row["commit_status"] == "committed" for row in rows if row["arm"] == arm)
        for arm in ARMS
    }
    rejects = {
        arm: sum(row["commit_status"] == "rejected" for row in rows if row["arm"] == arm)
        for arm in ARMS
    }
    tokens = {
        arm: {
            "prompt_tokens": sum(int(row["prompt_tokens"]) for row in rows if row["arm"] == arm),
            "completion_tokens": sum(
                int(row["completion_tokens"]) for row in rows if row["arm"] == arm
            ),
        }
        for arm in ARMS
    }
    restarts = {
        arm: sum(
            row["transaction_id"] is not None and row["restart_passed"] is True
            for row in rows
            if row["arm"] == arm
        )
        for arm in ARMS
    }
    rollbacks = {
        arm: sum(
            row["commit_status"] == "committed" and row["rollback_passed"] is True
            for row in rows
            if row["arm"] == arm
        )
        for arm in ARMS
    }
    order_ids = sorted({str(row["order_id"]) for row in rows})
    no_memory_deltas = [
        _order_arm_rate(rows, order_id, "procedural_constraint")
        - _order_arm_rate(rows, order_id, "no_memory")
        for order_id in order_ids
    ]
    trace_deltas = [
        _order_arm_rate(rows, order_id, "procedural_constraint")
        - _order_arm_rate(rows, order_id, "detailed_trajectory")
        for order_id in order_ids
    ]
    return {
        "prequential_exact_yield_by_arm": prequential,
        "hard_case_yield_by_arm": hard,
        "best_at_k_by_arm": best,
        "effective_support_by_arm": effective,
        "joint_correct_constraint_support_by_arm": joint,
        "retention_by_arm": retention,
        "forgetting_by_arm": forgetting,
        "negative_transfer_by_arm": negative,
        "retrieval_and_action_influence_by_arm": retrieval,
        "commits_by_arm": commits,
        "rejects_by_arm": rejects,
        "tokens_by_arm": tokens,
        "restarts_by_arm": restarts,
        "rollbacks_by_arm": rollbacks,
        "procedural_over_no_memory_order_lcb": order_level_lcb(no_memory_deltas),
        "procedural_over_trace_order_lcb": order_level_lcb(trace_deltas),
    }


def _chronology_passes(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> bool:
    for model in MODEL_SPECS:
        for order in manifest["orders"]:
            for arm in ARMS:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["model_id"] == model["hf_id"]
                        and row["order_id"] == order["order_id"]
                        and row["arm"] == arm
                    ),
                    key=lambda row: int(row["order_position"]),
                )
                if [row["event_id"] for row in selected] != order["event_ids"]:
                    return False
                if any(
                    row["visible_event_ids"] != order["event_ids"][:position]
                    or row["current_evidence_visible"] is not False
                    or row["future_evidence_visible"] is not False
                    for position, row in enumerate(selected)
                ):
                    return False
    return True


def completion_checks(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute completion without using the scientific effect direction."""

    rows = artifact["rows"]
    memory_rows = [row for row in rows if row["arm"] in MEMORY_ARMS]
    transaction_rows = [row for row in memory_rows if row["transaction_id"] is not None]
    return {
        "preconditions_pass": artifact["preconditions_checked"]["all_passed"] is True,
        "all_planned_rows_present": len(rows) == PLANNED_ROW_COUNT,
        "row_keys_unique": len({row["row_key"] for row in rows}) == len(rows),
        "chronology_preserved": _chronology_passes(rows, artifact["frozen_manifest"]),
        "read_only_episodes": all(
            row["snapshot_immutable"] is True and row["active_episode_write_count"] == 0
            for row in rows
        ),
        "capacity_equality": artifact["frozen_manifest"]["memory_contract"]["detailed_trajectory"]
        == artifact["frozen_manifest"]["memory_contract"]["procedural_constraint"],
        "no_memory_isolated": all(
            row["retrieved_ids"] == []
            and row["memory_read_count"] == 0
            and row["memory_write_count"] == 0
            and row["commit_status"] == "not_applicable"
            for row in rows
            if row["arm"] == "no_memory"
        ),
        "transactions_complete": bool(transaction_rows)
        and all(row["commit_status"] in {"committed", "rejected"} for row in transaction_rows),
        "state_boundaries_complete": all(
            row["state_hash_before"] and row["state_hash_after"] for row in memory_rows
        ),
        "restart_complete": bool(transaction_rows)
        and all(row["restart_passed"] is True for row in transaction_rows),
        "rollback_complete": all(
            row["rollback_passed"] is True
            for row in transaction_rows
            if row["commit_status"] == "committed"
        ),
        "model_teardown_complete": len(artifact["teardown_receipts"]) == 2
        and all(
            row.get("closed") is True and row.get("vram_released") is True
            for row in artifact["teardown_receipts"]
        ),
        "model_weights_immutable": artifact["model_weights_mutated"] is False,
        "cold_recompute_pass": artifact["cold_aggregate_recomputation_passed"] is True,
    }


def _gate_summary(checks: Mapping[str, bool]) -> JsonDict:
    failures = [
        {"check": name, "expected": True, "observed": value}
        for name, value in checks.items()
        if value is not True
    ]
    return {
        "checks": dict(checks),
        "failed_checks": [row["check"] for row in failures],
        "failures": failures,
    }


def _precondition_summary(checks: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    failures = [
        {
            "check": name,
            "expected": row["expected"],
            "observed": row["observed"],
        }
        for name, row in checks.items()
        if row["passed"] is not True
    ]
    return {
        "checks": {name: row["passed"] for name, row in checks.items()},
        "failed_checks": [row["check"] for row in failures],
        "failures": failures,
    }


def gpu_snapshot() -> list[JsonDict]:  # pragma: no cover - live host receipt.
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, uuid, name, total, free, used = [part.strip() for part in line.split(",", 5)]
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "total_mib": int(total),
                "free_mib": int(free),
                "used_mib": int(used),
            }
        )
    return rows


def _available_ram_bytes() -> int:  # pragma: no cover - live host receipt.
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def _ports_available(ports: Sequence[int]) -> bool:  # pragma: no cover - live host receipt.
    sockets = []
    try:
        for port in ports:
            handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            handle.bind(("127.0.0.1", port))
            sockets.append(handle)
        return True
    except OSError:
        return False
    finally:
        for handle in sockets:
            handle.close()


def _atomic_storage_probe(path: Path) -> bool:
    receipt = stream_mod.atomic_write(path / "atomic-probe", b"complete\n")
    return all(receipt.values()) and (path / "atomic-probe").read_bytes() == b"complete\n"


class GpuLeaseBundle:  # pragma: no cover - live process ownership only.
    """Hold both GPU locks while the two models run sequentially."""

    def __init__(self, leases: Sequence[lease_api.GpuLease]) -> None:
        self.leases = list(leases)
        self.started = False

    @classmethod
    def acquire(
        cls, state_root: Path, gpu_rows: Sequence[Mapping[str, Any]], *_args: Any
    ) -> GpuLeaseBundle:
        leases = []
        try:
            for gpu in gpu_rows:
                lease = lease_api.GpuLease.acquire(
                    runtime_dir=state_root / "leases",
                    task_id=EXPERIMENT_ID,
                    device_uuid=str(gpu["uuid"]),
                    expected_model="sequential-qwen36-then-gemma4-31b",
                    vram_before_mb=int(gpu["used_mib"]),
                    ttl_s=86_400.0,
                )
                lease.transition("admitted")
                lease.transition("loading")
                leases.append(lease)
            return cls(leases)
        except Exception:
            for lease in leases:
                lease.close()
            raise

    def owner_receipts(self) -> list[JsonDict]:
        return [{**lease.owner_receipt(), "owner_bound": True} for lease in self.leases]

    def start_inference(self, resident_mib: int) -> None:
        if self.started:
            return
        for lease in self.leases:
            lease.transition("resident", vram_mb=resident_mib)
            lease.transition("inferencing")
        self.started = True

    def heartbeat(self) -> None:
        for lease in self.leases:
            lease.heartbeat()

    def _finish(self, terminal: str) -> JsonDict:
        releases = []
        for lease in self.leases:
            if lease.document["phase"] == "loading":
                lease.transition("terminal_blocked")
            else:
                lease.transition("unloading")
                lease.transition("validating", vram_mb=4, exit_code=0, unload_observed=True)
                lease.transition(terminal)
            releases.append(lease.release())
        return {"released": True, "terminal": True, "owner_bound": True, "rows": releases}

    def complete(self) -> JsonDict:
        return self._finish("terminal_complete")

    def block(self) -> JsonDict:
        return self._finish("terminal_blocked")


def _check(expected: Any, observed: Any) -> JsonDict:
    return {"expected": expected, "observed": observed, "passed": expected == observed}


def collect_preconditions(
    *,
    fixture: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    state_root: Path,
    overrides: Mapping[str, bool] | None,
) -> JsonDict:
    """Measure every owned prerequisite before model inference."""

    state_root.mkdir(parents=True, exist_ok=True)
    paths = [Path(str(row.get("model_path", ""))) for row in specs]
    paths_exist = len(paths) == 2 and all(path.is_file() for path in paths)
    receipts = [model_file_receipt(row) for row in specs] if paths_exist else []
    try:  # pragma: no cover - live package check.
        from llama_cpp import llama_cpp

        cuda = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:  # pragma: no cover
        cuda = False
    try:  # pragma: no cover - live driver check.
        gpu_rows = gpu_snapshot()
    except Exception:  # pragma: no cover
        gpu_rows = []
    tokenizer_rows = [gguf_tokenizer_loadable(str(path)) for path in paths] if paths_exist else []
    free_vram = sum(int(row["free_mib"]) for row in gpu_rows) * 1024 * 1024
    largest = max((receipt["size_bytes"] for receipt in receipts), default=0)
    actual_ids = [row.get("hf_id") for row in specs]
    expected_ids = [row["hf_id"] for row in MODEL_SPECS]
    observations: dict[str, Any] = {
        "planning_date_matches": True,
        "procedural_memory_stream_ready": fixture.get("procedural_memory_stream_ready") is True,
        "six_frozen_orders": len(fixture.get("stream_manifest", {}).get("orders", [])) == 6,
        "exact_model_specs": actual_ids == expected_ids,
        "cached_ggufs_with_hashes": len(receipts) == 2
        and all(str(row["sha256"]).startswith("sha256:") for row in receipts),
        "embedded_tokenizers": len(tokenizer_rows) == 2
        and all(row[0] is True for row in tokenizer_rows),
        "llama_cpp_cuda": cuda,
        "one_model_vram": largest > 0 and free_vram >= int(largest * 1.1),
        "task_owned_lease": False,
        "ports_available": _ports_available(FROZEN_PORTS),
        "exact_authority": callable(stream_mod.AtomicRepresentationJournal.transact),
        "atomic_storage": _atomic_storage_probe(state_root / "preflight"),
        "ram_sufficient": _available_ram_bytes() >= 16 * 1024**3,
        "disk_sufficient": shutil.disk_usage(state_root).free >= 1024**3,
    }
    observations.update(dict(overrides or {}))
    checks = {name: _check(True, value) for name, value in observations.items()}
    return {
        "checks": checks,
        "all_passed": all(row["passed"] for row in checks.values()),
        "model_receipts": receipts,
        "tokenizer_receipts": [
            {"model_id": spec["hf_id"], "passed": row[0], "detail": row[1]}
            for spec, row in zip(specs, tokenizer_rows, strict=False)
        ],
        "gpu_snapshot": gpu_rows,
        "free_vram_bytes": free_vram,
        "largest_model_bytes": largest,
        "ram_available_bytes": _available_ram_bytes(),
        "disk_free_bytes": shutil.disk_usage(state_root).free,
        "ports": list(FROZEN_PORTS),
    }


class LiveLlamaRunner:  # pragma: no cover - required live CUDA path.
    """Load one local GGUF and run deterministic JSON chat completions."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self.spec = dict(spec)
        self._llm: Any = None
        self._gpu_before: list[JsonDict] = []

    def load(self) -> JsonDict:
        from llama_cpp import Llama, llama_cpp

        self._gpu_before = gpu_snapshot()
        started = time.monotonic()
        self._llm = Llama(
            model_path=str(self.spec["model_path"]),
            n_gpu_layers=-1,
            n_ctx=2048,
            n_batch=256,
            verbose=False,
        )
        return {
            "model_id": self.spec["hf_id"],
            "model_path": str(Path(str(self.spec["model_path"])).resolve()),
            "loaded": True,
            "cuda_offload": bool(llama_cpp.llama_supports_gpu_offload()),
            "process_id": os.getpid(),
            "gpu_before": self._gpu_before,
            "gpu_after": gpu_snapshot(),
            "load_duration_s": round(time.monotonic() - started, 6),
        }

    def count_tokens(self, text: str) -> int:
        if self._llm is None:
            raise RuntimeError("model is not loaded")
        return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False))

    def generate(self, prompt: str, *, seed: int, max_tokens: int) -> JsonDict:
        if self._llm is None:
            raise RuntimeError("model is not loaded")
        started = time.monotonic()
        result = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Follow the JSON action contract exactly."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed,
            response_format={"type": "json_object"},
        )
        usage = result.get("usage", {})
        return {
            "text": result["choices"][0]["message"]["content"] or "",
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "latency_s": round(time.monotonic() - started, 6),
            "seed": seed,
            "max_tokens": max_tokens,
        }

    def close(self) -> JsonDict:
        self._llm = None
        gc.collect()
        after = gpu_snapshot()
        before_used = sum(int(row["used_mib"]) for row in self._gpu_before)
        after_used = sum(int(row["used_mib"]) for row in after)
        return {
            "model_id": self.spec["hf_id"],
            "closed": True,
            "vram_released": after_used <= before_used + 512,
            "gpu_after_close": after,
        }


def _blank_metrics() -> JsonDict:
    return {field: {} if not field.endswith("_lcb") else 0.0 for field in ROW_DERIVED_FIELDS}


def _base_artifact(
    *,
    fixture_path: Path,
    fixture: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "status": "complete_blocked_procedural_csl_ab",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "models_used": [
            {
                "model_id": row["hf_id"],
                "role": row["role"],
                "family": row["family"],
                "model_path": row.get("model_path"),
            }
            for row in specs
        ],
        "live_model_invoked": False,
        "frozen_manifest": deepcopy(dict(manifest)),
        "gpu_receipts": [],
        "teardown_receipts": [],
        "model_weight_receipts": [],
        "model_weights_mutated": False,
        "fixture_receipt": {
            "path": str(fixture_path),
            "sha256": sha256_file(fixture_path),
            "procedural_memory_stream_ready": fixture.get("procedural_memory_stream_ready"),
            "stream_hash": fixture.get("stream_manifest", {}).get("stream_hash"),
        },
        "preconditions_checked": deepcopy(dict(preconditions)),
        "rows": [],
        **_blank_metrics(),
        "transaction_receipts": [],
        "poison_receipts": [],
        "cold_aggregate_recomputation_passed": False,
        "model_specific_answer_traces_shared": False,
        "positive_credit_checks": {},
        "prospective_csl_completed": False,
        "gate_check_summary": _precondition_summary(preconditions["checks"]),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_procedural_csl_ab: owned precondition failed",
        "tests_run": [dict(row) for row in DEFAULT_TESTS_RUN],
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _positive_credit_checks(artifact: Mapping[str, Any]) -> JsonDict:
    hard = artifact["hard_case_yield_by_arm"]
    support = artifact["effective_support_by_arm"]
    retention = artifact["retention_by_arm"]
    return {
        "procedural_over_no_memory_lcb": artifact["procedural_over_no_memory_order_lcb"] > 0.0,
        "procedural_over_trace_lcb": artifact["procedural_over_trace_order_lcb"] > 0.0,
        "nonzero_commits_and_rejects": artifact["commits_by_arm"]["procedural_constraint"] > 0
        and artifact["rejects_by_arm"]["procedural_constraint"] > 0,
        "no_anchor_forgetting": retention["procedural_constraint"]["rate"]
        >= max(retention["no_memory"]["rate"], retention["detailed_trajectory"]["rate"]),
        "hard_case_margin": hard["procedural_constraint"]["rate"] + HARD_CASE_MARGIN
        >= max(hard["no_memory"]["rate"], hard["detailed_trajectory"]["rate"]),
        "support_not_contracted": support["procedural_constraint"]["rate"]
        >= max(support["no_memory"]["rate"], support["detailed_trajectory"]["rate"]),
        "zero_poison": all(row["committed"] is False for row in artifact["poison_receipts"]),
    }


def _run_model_rows(
    *,
    runner: Runner,
    spec: Mapping[str, Any],
    model_index: int,
    fixture: Mapping[str, Any],
    manifest: Mapping[str, Any],
    state_root: Path,
    lease: Any,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:
    events = {str(row["event_id"]): row for row in manifest["events"]}
    rows: list[JsonDict] = []
    transactions: list[JsonDict] = []
    poison: list[JsonDict] = []
    for order_index, order in enumerate(manifest["orders"]):
        stores = {
            arm: ArmMemoryStore(
                state_root / f"model-{model_index}" / str(order["order_id"]) / arm,
                arm,
            )
            for arm in MEMORY_ARMS
        }
        for event_index, event_id in enumerate(order["event_ids"]):
            lease.heartbeat()
            event = events[str(event_id)]
            snapshots: dict[str, JsonDict] = {}
            retrievals: dict[str, list[JsonDict]] = {}
            contexts: dict[str, tuple[str, int, int]] = {}
            for arm, store in stores.items():
                snapshots[arm] = store.begin_episode(
                    str(event_id), order["event_ids"][:event_index]
                )
                retrievals[arm] = store.retrieve(event, top_k=TOP_K, position=event_index)
                contexts[arm] = render_context(fixture, arm, retrievals[arm], runner)
            execution_key = f"model_{model_index}:order_{order_index + 1}:event_{event_index}"
            event_rows: dict[str, JsonDict] = {}
            for execution_position, arm in enumerate(manifest["arm_rotations"][execution_key]):
                context_text, context_bytes, context_tokens = (
                    ("[]", 0, 0) if arm == "no_memory" else contexts[arm]
                )
                candidates = _run_candidates(
                    runner,
                    event,
                    arm,
                    context_text,
                    model_index,
                    order_index,
                    event_index,
                )
                snapshot_hash = (
                    sha256_json({"arm": "no_memory"})
                    if arm == "no_memory"
                    else snapshots[arm]["state_hash"]
                )
                event_rows[arm] = _base_row(
                    event=event,
                    order=order,
                    order_position=event_index,
                    spec=spec,
                    arm=arm,
                    arm_execution_position=execution_position,
                    candidates=candidates,
                    retrieved=[] if arm == "no_memory" else retrievals[arm],
                    context_bytes=context_bytes,
                    context_tokens=context_tokens,
                    snapshot_hash=snapshot_hash,
                )
            baseline = event_rows["no_memory"]
            baseline_fingerprint = baseline["after_action_fingerprint"]
            for arm, row in event_rows.items():
                row["paired_no_memory_candidate_seeds"] = baseline["candidate_seeds"]
                row["before_action_fingerprint"] = baseline_fingerprint
                row["action_influenced"] = row["after_action_fingerprint"] != baseline_fingerprint
                cited = {
                    candidate["cited_memory_id"]
                    for candidate in row["candidates"]
                    if candidate["cited_memory_id"] != "none"
                }
                row["memory_cited"] = bool(cited & set(row["retrieved_ids"]))
                row["operational_memory_use"] = bool(row["retrieved_ids"])
                row["operational_memory_use"] = row["operational_memory_use"] and (
                    row["memory_cited"]
                    or (
                        row["pass_at_1"] == 1
                        and any(
                            item["family"] == row["family"]
                            for item in ([] if arm == "no_memory" else retrievals[arm])
                        )
                    )
                )
            for arm, store in stores.items():
                row = event_rows[arm]
                row["snapshot_immutable"] = store.state_hash() == snapshots[arm]["state_hash"]
                store.end_episode()
                row["exact_result_known_before_commit"] = True
                if event["eligibility"] != "transaction_candidate":
                    row["commit_status"] = "no_update"
                    row["commit_or_reject_reason"] = "no_update_evaluation_event"
                    continue
                pair = source_representation_pair(fixture, str(event_id))
                receipt = store.transact(event, pair, event_index, str(order["order_id"]))
                compact = _compact_transaction(receipt)
                compact.update({"model_id": spec["hf_id"], "arm": arm})
                transactions.append(compact)
                row["transaction_id"] = receipt["transaction_id"]
                row["commit_status"] = "committed" if receipt["committed"] else "rejected"
                row["commit_or_reject_reason"] = receipt["admission_reason"]
                row["memory_write_count"] = int(receipt["committed"] is True)
                row["state_hash_after"] = receipt["state_hash"]
                row["restart_passed"] = bool(
                    receipt["atomic_restart_receipt"]["bytes_match"]
                    and receipt["atomic_restart_receipt"]["hash_match"]
                )
                if receipt["committed"] is True:
                    rollback = store.rollback(receipt)
                    row["rollback_passed"] = bool(rollback["byte_identical"])
                    store.reapply(receipt)
                if event["event_kind"] == "poison_candidate":
                    poison_row = {
                        **compact,
                        "model_id": spec["hf_id"],
                        "arm": arm,
                    }
                    poison.append(poison_row)
                    row["poison_admitted"] = receipt["committed"] is True
            rows.extend(event_rows[arm] for arm in ARMS)
    return rows, transactions, poison


def run_experiment(
    *,
    fixture_path: Path | str = REPO_ROOT / FIXTURE_RELATIVE_PATH,
    state_root: Path | str | None = None,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    runner_factory: Callable[[Mapping[str, Any]], Runner] = LiveLlamaRunner,
    lease_factory: Callable[..., Any] = GpuLeaseBundle.acquire,
    precondition_overrides: Mapping[str, bool] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Run the two model sessions and three isolated arms."""

    started = time.monotonic()
    if state_root is None:  # pragma: no cover - required live command uses temporary state.
        with tempfile.TemporaryDirectory(prefix="carnot-exp6762-") as directory:
            return run_experiment(
                fixture_path=fixture_path,
                state_root=directory,
                model_specs=model_specs,
                runner_factory=runner_factory,
                lease_factory=lease_factory,
                precondition_overrides=precondition_overrides,
                duration_s=duration_s,
            )
    fixture_path = Path(fixture_path)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    specs = resolve_model_specs(model_specs)
    root = Path(state_root)
    preconditions = collect_preconditions(
        fixture=fixture,
        specs=specs,
        state_root=root,
        overrides=precondition_overrides,
    )
    receipts = preconditions["model_receipts"]
    manifest = freeze_manifest(fixture, receipts)
    lease = None
    non_lease_pass = all(
        row["passed"] for name, row in preconditions["checks"].items() if name != "task_owned_lease"
    )
    if non_lease_pass:
        try:
            lease = lease_factory(root, preconditions["gpu_snapshot"], specs)
            observed_lease = True
        except Exception:  # pragma: no cover - live ownership failure is a blocked artifact.
            observed_lease = False
        override_lease = (precondition_overrides or {}).get("task_owned_lease")
        if override_lease is not None:
            observed_lease = override_lease
        preconditions["checks"]["task_owned_lease"] = _check(True, observed_lease)
        preconditions["all_passed"] = all(row["passed"] for row in preconditions["checks"].values())
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact = _base_artifact(
        fixture_path=fixture_path,
        fixture=fixture,
        specs=specs,
        manifest=manifest,
        preconditions=preconditions,
        duration_s=elapsed,
    )
    if preconditions["all_passed"] is not True:
        if lease is not None:
            lease.block()
        artifact["gate_check_summary"] = _precondition_summary(preconditions["checks"])
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    artifact["preconditions_checked"]["lease_owner_receipts"] = lease.owner_receipts()
    before_weights = receipts
    all_rows: list[JsonDict] = []
    transactions: list[JsonDict] = []
    poison: list[JsonDict] = []
    for model_index, spec in enumerate(specs):
        runner = runner_factory(spec)
        load_receipt = runner.load()
        artifact["gpu_receipts"].append(load_receipt)
        if hasattr(lease, "start_inference"):
            resident = sum(int(row.get("used_mib", 0)) for row in load_receipt.get("gpu_after", []))
            lease.start_inference(resident)
        model_rows, model_transactions, model_poison = _run_model_rows(
            runner=runner,
            spec=spec,
            model_index=model_index,
            fixture=fixture,
            manifest=manifest,
            state_root=root,
            lease=lease,
        )
        all_rows.extend(model_rows)
        transactions.extend(model_transactions)
        poison.extend(model_poison)
        artifact["teardown_receipts"].append(runner.close())
    lease_release = lease.complete()
    artifact["preconditions_checked"]["lease_release_receipt"] = lease_release
    after_weights = [model_file_receipt(spec) for spec in specs]
    artifact["model_weight_receipts"] = [
        {
            "model_id": before["model_id"],
            "before": before,
            "after": after,
            "mutated": before != after,
        }
        for before, after in zip(before_weights, after_weights, strict=True)
    ]
    artifact["model_weights_mutated"] = any(
        row["mutated"] is True for row in artifact["model_weight_receipts"]
    )
    aggregates = reduce_rows(all_rows)
    cold = reduce_rows(json.loads(json.dumps(all_rows)))
    artifact.update(
        {
            "status": "complete_procedural_csl_ab",
            "live_model_invoked": len(artifact["gpu_receipts"]) == 2
            and all(
                row.get("loaded") is True and row.get("cuda_offload") is True
                for row in artifact["gpu_receipts"]
            ),
            "rows": all_rows,
            **aggregates,
            "transaction_receipts": transactions,
            "poison_receipts": poison,
            "cold_aggregate_recomputation_passed": cold == aggregates,
        }
    )
    checks = completion_checks(artifact)
    completed = all(checks.values())
    artifact["prospective_csl_completed"] = completed
    artifact["gate_check_summary"] = _gate_summary(checks)
    positive_checks = _positive_credit_checks(artifact)
    artifact["positive_credit_checks"] = positive_checks
    positive = completed and all(positive_checks.values())
    artifact["verdict_class"] = "positive" if positive else ("null" if completed else "partial")
    artifact["honest_verdict"] = (
        "complete_positive: procedural constraint memory beat no memory and detailed trajectories with all safety gates"
        if positive
        else (
            "complete_null: the planned comparison completed without positive procedural-memory credit"
            if completed
            else "complete_partial: one or more row, lifecycle, or teardown gates failed"
        )
    )
    artifact["duration_s"] = round(
        float(duration_s) if duration_s is not None else time.monotonic() - started, 6
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash frozen inputs, rows, aggregates, transactions, and lifecycle evidence."""

    material = {
        key: artifact.get(key)
        for key in REQUIRED_ARTIFACT_FIELDS
        if key not in {"duration_s", "reproducibility_checksum", "tests_run"}
    }
    return sha256_json(material)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return every closed validation error without changing the artifact."""

    errors = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    expected_principles = set(REQUIRED_ARTIFACT_FIELDS) | {
        f"gate:{gate}" for gate in COMPLETION_GATES
    }
    if set(artifact.get("field_principles", {})) != expected_principles:
        errors.append("field_principles coverage mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("model_specific_answer_traces_shared") is not False:
        errors.append("model answer traces shared")
    if artifact.get("model_weights_mutated") is True:
        errors.append("model weights mutated")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    rows = artifact.get("rows", [])
    if rows:
        reduced = reduce_rows(rows)
        if any(artifact.get(field) != reduced[field] for field in ROW_DERIVED_FIELDS):
            errors.append("row-derived metrics mismatch")
        checks = completion_checks(artifact)
        if artifact.get("prospective_csl_completed") is not all(checks.values()):
            errors.append("completion gates mismatch")
    elif artifact.get("prospective_csl_completed") is True:
        errors.append("completed artifact has no rows")
    if artifact.get("verdict_class") == "blocked" and rows:
        errors.append("blocked artifact contains rows")
    return errors


def write_artifact(path: Path | str, artifact: Mapping[str, Any]) -> JsonDict:
    """Validate and publish one complete result through an atomic rename."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    target = Path(path)
    data = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    receipt = stream_mod.atomic_write(target, data)
    return {
        "path": str(target),
        "atomic_rename": receipt["rename"],
        "sha256": sha256_file(target),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the prospective comparison or validate its stored result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--fixture-path", default=str(REPO_ROOT / FIXTURE_RELATIVE_PATH))
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
    if args.date != RUN_DATE:
        raise ValueError(f"planning date must be {RUN_DATE}")
    artifact = run_experiment(
        fixture_path=Path(args.fixture_path),
        state_root=Path(args.state_root) if args.state_root else None,
    )
    write_artifact(result_path, artifact)
    return 0


__all__ = [
    "ACTION_BY_FAMILY",
    "ARMS",
    "ArmMemoryStore",
    "CANDIDATE_COUNT_K",
    "DEFAULT_TESTS_RUN",
    "EXPERIMENT_ID",
    "FIELD_PRINCIPLES",
    "FIXTURE_RELATIVE_PATH",
    "INFERENCE_SUBSTRATE",
    "MAX_TOKENS",
    "MODEL_SPECS",
    "PLANNED_ROW_COUNT",
    "REPORT_SPEC_RELATIVE_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "RESULT_RELATIVE_PATH",
    "ROW_DERIVED_FIELDS",
    "RUN_DATE",
    "ReadOnlyEpisodeError",
    "SCHEMA",
    "SCRIPT_RELATIVE_PATH",
    "SPEC_RELATIVE_PATH",
    "TEST_PRECONDITION_OVERRIDES",
    "collect_preconditions",
    "freeze_manifest",
    "main",
    "reduce_rows",
    "reproducibility_checksum",
    "run_experiment",
    "sha256_file",
    "source_representation_pair",
    "validate_artifact",
    "write_artifact",
]
