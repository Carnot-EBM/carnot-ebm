"""Compare four frozen-Qwen external-memory policies chronologically.

Spec refs: REQ-SELF-7142 and SCENARIO-SELF-7142-*.

The model always acts before the exact checker opens the current outcome. The
checker controls only later external-memory writes. It never changes or ranks
the current action, and this module never changes model weights.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from typing import Any

from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
ModelCall = Callable[..., JsonDict]
ExactScore = Callable[[Mapping[str, Any], Mapping[str, Any]], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_7142_v627_flowbalance_memory_csl"
SCHEMA = "carnot.experiment_7142.v627_flowbalance_memory_csl.v1"
RUN_DATE = "20260908"
RANDOM_SEED = 7_142_202_609_08
QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
RESULT_PATH = REPO_ROOT / "results/experiment_7142_v627_flowbalance_memory_csl.json"
SOURCE_PATH = REPO_ROOT / "results/experiment_7141_v627_csl_event_stream.json"
BANK_PATH = REPO_ROOT / "results/experiment_7129_v626_sota_constraint_bank.json"
TRANSACTION_ROOT = REPO_ROOT / "results/raw/experiment_7142_v627_flowbalance_memory_csl"
EXPECTED_EVENT_COUNT = 108
PROMPT_BUDGET_BYTES = 2_048
MEMORY_BUDGET_BYTES = 512
MAX_GENERATION_TOKENS = 48
MAX_MEMORY_RECORDS = 8
EPISODE_SIZE = 9
LOCAL_GPU_USD_PER_HOUR = 0.35
INFERENCE_SUBSTRATE = "live_llm_inference_with_frozen_policy_state"
INFERENCE_SUBSTRATE_CLASS = "model_full_generation"
EXECUTION_VENUE = "host"

ARMS = (
    "no_memory",
    "equal_budget_raw_trace",
    "delayed_procedural_memory",
    "verifier_balanced_strategy_memory",
)

MODEL_SPECS: list[JsonDict] = [
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": QWEN_ID,
        "gpu": 0,
        "quantization": "Q4_K_M",
        "chat_template_source": "embedded_gguf",
        "remote_allowed": False,
    }
]

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "MODEL_SPECS",
    "model_identity_rows",
    "model_load_receipts",
    "arm_rows",
    "event_rows",
    "action_receipt_rows",
    "outcome_reveal_rows",
    "advantage_rows",
    "memory_candidate_rows",
    "transaction_rows",
    "signature_rows",
    "conflict_rows",
    "rollback_rows",
    "refresh_rows",
    "learning_curve_rows",
    "future_success_rows",
    "stale_memory_rows",
    "protected_retention_rows",
    "negative_transfer_rows",
    "abstention_rows",
    "latency_rows",
    "token_rows",
    "cost_rows",
    "paired_interval_rows",
    "model_weights_changed",
    "flowbalance_training_reproduction",
    "flowbalance_memory_csl_complete_score",
    "future_uplift_supported_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "A reason for each required field makes missing evidence visible.",
    "preconditions_checked": "Exact preflight rows prevent a missing resource from becoming invented evidence.",
    "run_date": "The fixed execution date separates this run from later replays.",
    "inference_substrate": "The substrate states that a frozen local model produced every action.",
    "inference_substrate_class": "The closed class separates full generation from a blocked no-run.",
    "execution_venue": "The host venue prevents unattributed remote compute claims.",
    "duration_s": "Measured wall time exposes interruption and implausible model work.",
    "source_artifact_hashes": "Hashes bind the result to exact stream, model, code, and solver bytes.",
    "rows": "The complete event-arm denominator prevents aggregate-only claims.",
    "MODEL_SPECS": "The cached model declaration prevents silent model or quantization substitution.",
    "model_identity_rows": "Path, revision, hash, backend, device, and template identify the invoked model.",
    "model_load_receipts": "Live load and smoke receipts prove the declared model could generate.",
    "arm_rows": "Frozen arm summaries prevent a treatment from disappearing after outcomes open.",
    "event_rows": "Per-event actions support independent chronology and metric replay.",
    "action_receipt_rows": "A durable action seal proves each outcome opened after all four actions.",
    "outcome_reveal_rows": "Delayed exact outcomes keep the verifier outside current action selection.",
    "advantage_rows": "Paired exact differences supply the sign for balanced updates.",
    "memory_candidate_rows": "Candidate rows expose retained, reversed, and rejected guidance.",
    "transaction_rows": "Staged terminal receipts make every external state change auditable.",
    "signature_rows": "Signatures detect any change to a memory record after admission.",
    "conflict_rows": "Conflict evidence exposes deterministic replacement instead of hidden overwrite.",
    "rollback_rows": "Parent restoration proves a failed publish cannot corrupt active memory.",
    "refresh_rows": "Sealed boundary rows prove refresh is bounded and chronological.",
    "learning_curve_rows": "Prefix success rates show when any future value appears or decays.",
    "future_success_rows": "Held-future exact success is the main value measurement.",
    "stale_memory_rows": "Harm after memory reuse reveals obsolete guidance.",
    "protected_retention_rows": "Protected outcomes prevent future gains from hiding forgetting.",
    "negative_transfer_rows": "Paired losses expose cases where memory hurts a baseline success.",
    "abstention_rows": "No-write rows prove ties and missing preferences do not update memory.",
    "latency_rows": "Per-arm time shows the service cost of external memory.",
    "token_rows": "Prompt and completion counts prove matched budgets and model dose.",
    "cost_rows": "An explicit local cost rule makes cost per event reproducible.",
    "paired_interval_rows": "Paired intervals keep uncertainty attached to future uplift claims.",
    "model_weights_changed": "False isolates external memory from policy training.",
    "flowbalance_training_reproduction": "False prevents an external update rule from becoming a training claim.",
    "flowbalance_memory_csl_complete_score": "One requires all planned actions and terminal receipts, not uplift.",
    "future_uplift_supported_score": "One requires a positive paired lower bound with retained protected performance.",
    "random_seed": "One fixed seed binds event-level decoding and deterministic conflict rules.",
    "reproducibility_checksum": "A canonical digest detects any later artifact mutation.",
    "gate_check_summary": "The first failed expected-observed check makes a block actionable.",
    "verifier_is_oracle": "False states that the exact checker reveals outcomes but selects no action.",
    "verdict_class": "A closed class separates complete null evidence from support or disqualification.",
    "honest_verdict": "A class-matching prefix prevents completion from implying scientific uplift.",
}

HIDDEN_EVENT_KEYS = frozenset(
    {
        "exact_success",
        "exact_correct",
        "exact_outcome",
        "exact_label",
        "parse_success",
        "parsed",
        "witness",
        "objective_matches",
        "future_success",
    }
)


class PreconditionError(RuntimeError):
    """Carry one failed external check into a schema-complete artifact."""

    def __init__(self, check: str, expected: Any, observed: Any):
        super().__init__(check)
        self.check = check
        self.expected = deepcopy(expected)
        self.observed = deepcopy(observed)


class ProtocolError(RuntimeError):
    """Reject a chronology, budget, receipt, or transaction violation."""


def canonical_json(value: Any) -> str:
    """Serialize evidence with one stable UTF-8 spelling."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes and retain the digest algorithm in the value."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash the exact UTF-8 bytes of one string."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash one canonical JSON projection."""

    return sha256_text(canonical_json(value))


def sha256_path(path: Path) -> str | None:
    """Stream a file hash so large cached model files do not enter memory."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind every artifact field except the digest that stores this value."""

    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def gate_row(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Record one gate with exact expected and observed values."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain all checks and promote the first failure for automation."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Replace one JSON file only after its complete bytes reach storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _empty_artifact(run_date: str) -> JsonDict:
    """Return the complete schema before any external check can run."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": {"all_passed": False, "checks": []},
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_identity_rows": [],
        "model_load_receipts": [],
        "arm_rows": [],
        "event_rows": [],
        "action_receipt_rows": [],
        "outcome_reveal_rows": [],
        "advantage_rows": [],
        "memory_candidate_rows": [],
        "transaction_rows": [],
        "signature_rows": [],
        "conflict_rows": [],
        "rollback_rows": [],
        "refresh_rows": [],
        "learning_curve_rows": [],
        "future_success_rows": [],
        "stale_memory_rows": [],
        "protected_retention_rows": [],
        "negative_transfer_rows": [],
        "abstention_rows": [],
        "latency_rows": [],
        "token_rows": [],
        "cost_rows": [],
        "paired_interval_rows": [],
        "model_weights_changed": False,
        "flowbalance_training_reproduction": False,
        "flowbalance_memory_csl_complete_score": 0,
        "future_uplift_supported_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(
            [gate_row("experiment_initialized", True, "checks_not_started", False)]
        ),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_initial_schema_written_before_checks",
        "status_receipt_rows": [],
        "frozen_plan": {},
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def initialize_artifact(path: Path, run_date: str) -> JsonDict:
    """Write every required field before a gate, model, stream, or store check."""

    artifact = _empty_artifact(run_date)
    write_json_atomic(path, artifact)
    return artifact


def _nested_keys(value: Any) -> set[str]:
    """List nested keys so an action cannot receive hidden outcome evidence."""

    if isinstance(value, Mapping):
        return {str(key) for key in value} | {
            nested for child in value.values() for nested in _nested_keys(child)
        }
    if isinstance(value, (list, tuple)):
        return {nested for child in value for nested in _nested_keys(child)}
    return set()


def resolve_model_specs(
    *, cached_pair_func: Callable[..., Sequence[Mapping[str, Any]] | None] = cached_sota_pair
) -> list[JsonDict]:
    """Resolve the cached Qwen Q4 file through the shared SOTA helper."""

    pair = cached_pair_func(gpu_indices=(0, 1), preferred_quant="Q4_K_M", model_indices=(0, 2))
    qwen = next((dict(row) for row in pair or [] if row.get("hf_id") == QWEN_ID), None)
    if qwen is None:
        raise PreconditionError("cached_qwen_q4", "cached Q4_K_M file", None)
    path = Path(str(qwen.get("model_path") or ""))
    if not path.is_file() or "q4_k_m" not in path.name.lower():
        raise PreconditionError(
            "cached_qwen_q4",
            {"exists": True, "quantization": "Q4_K_M"},
            {"path": str(path), "exists": path.is_file()},
        )
    return [
        {
            **qwen,
            "quantization": "Q4_K_M",
            "chat_template_source": "embedded_gguf",
            "revision": path.parent.name,
            "model_sha256": sha256_path(path),
            "backend": "llama_cpp_cuda",
            "device": [0, 1],
            "remote_allowed": False,
        }
    ]


def _fit_bytes(text: str, size: int) -> str:
    """Return valid UTF-8 text whose encoded length is exactly one budget."""

    encoded = text.encode("utf-8")
    if len(encoded) > size:
        encoded = encoded[:size]
        while True:
            try:
                text = encoded.decode("utf-8")
                break
            except UnicodeDecodeError:
                encoded = encoded[:-1]
    else:
        text = encoded.decode("utf-8")
    return text + " " * (size - len(text.encode("utf-8")))


def build_prompt_rows(event: Mapping[str, Any], memories: Mapping[str, str]) -> list[JsonDict]:
    """Build four label-free prompts with equal context and decode budgets."""

    leaked = _nested_keys(event) & HIDDEN_EVENT_KEYS
    if leaked:
        raise ProtocolError(f"outcome_leakage:{sorted(leaked)}")
    if set(memories) != set(ARMS):
        raise ProtocolError("arm_memory_roster_mismatch")
    rows = []
    event_index = int(event["chronology_index"])
    seed = RANDOM_SEED + event_index
    for arm in ARMS:
        memory = _fit_bytes(str(memories[arm]), MEMORY_BUDGET_BYTES)
        body = (
            "/no_think\nReturn only the direct JSON object requested by the task.\n"
            f"Prior external guidance:\n{memory}\n"
            f"Current task:\n{str(event['prompt']).rstrip()}"
        )
        prompt = _fit_bytes(body, PROMPT_BUDGET_BYTES)
        rows.append(
            {
                "event_id": event["event_id"],
                "chronology_index": event_index,
                "arm": arm,
                "prompt": prompt,
                "prompt_hash": sha256_text(prompt),
                "prompt_bytes": len(prompt.encode("utf-8")),
                "memory_text": memory,
                "memory_hash": sha256_text(memory),
                "memory_bytes": len(memory.encode("utf-8")),
                "seed": seed,
                "max_tokens": MAX_GENERATION_TOKENS,
            }
        )
    return rows


def seal_action_receipts(
    event: Mapping[str, Any],
    prompt_rows: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Seal all four model actions before an exact outcome can open."""

    prompts = {str(row.get("arm")): row for row in prompt_rows}
    if set(prompts) != set(ARMS) or set(responses) != set(ARMS):
        raise ProtocolError("all_action_receipts_required")
    receipts = []
    previous = "sha256:" + "0" * 64
    for arm in ARMS:
        payload = {
            "event_id": event["event_id"],
            "chronology_index": event["chronology_index"],
            "event_content_hash": event["event_content_hash"],
            "arm": arm,
            "prompt_hash": prompts[arm]["prompt_hash"],
            "response_hash": sha256_text(str(responses[arm].get("raw_text", ""))),
            "terminal_state": responses[arm].get("terminal_state"),
            "previous_receipt_hash": previous,
        }
        payload["receipt_hash"] = sha256_json(payload)
        previous = payload["receipt_hash"]
        receipts.append(payload)
    return receipts


def reveal_outcomes(
    event: Mapping[str, Any],
    responses: Mapping[str, Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    exact_score: ExactScore,
) -> list[JsonDict]:
    """Open exact results only after every arm has a valid action receipt."""

    by_arm = {str(row.get("arm")): row for row in receipts}
    if set(by_arm) != set(ARMS) or set(responses) != set(ARMS):
        raise ProtocolError("all_action_receipts_required")
    for arm, receipt in by_arm.items():
        payload = {key: value for key, value in receipt.items() if key != "receipt_hash"}
        if receipt.get("receipt_hash") != sha256_json(payload):
            raise ProtocolError(f"action_receipt_invalid:{arm}")
    rows = []
    for arm in ARMS:
        try:
            score = dict(exact_score(event, responses[arm]))
        except Exception as exc:  # noqa: BLE001 - a checker failure remains an exact failure row.
            score = {
                "exact_success": False,
                "parse_success": False,
                "failed_constraint_classes": [f"exact_score_error:{type(exc).__name__}"],
            }
        payload = {
            "event_id": event["event_id"],
            "chronology_index": event["chronology_index"],
            "arm": arm,
            "action_receipt_hash": by_arm[arm]["receipt_hash"],
            "exact_success": score.get("exact_success"),
            "parse_success": score.get("parse_success"),
            "failed_constraint_classes": list(score.get("failed_constraint_classes") or []),
        }
        payload["outcome_receipt_hash"] = sha256_json(payload)
        rows.append(payload)
    return rows


def flowbalance_update(
    candidate: Mapping[str, Any],
    *,
    balanced_success: bool | None,
    baseline_success: bool | None,
) -> JsonDict:
    """Retain positive guidance, reverse negative guidance, and abstain on no preference."""

    if balanced_success is None or baseline_success is None:
        return {
            "advantage": None,
            "preference": "missing",
            "direction": "abstain",
            "write": False,
            "record": None,
        }
    advantage = int(balanced_success) - int(baseline_success)
    if advantage == 0:
        return {
            "advantage": 0,
            "preference": "tie",
            "direction": "abstain",
            "write": False,
            "record": None,
        }
    record = deepcopy(dict(candidate))
    direction = "retain" if advantage > 0 else "reverse"
    if direction == "reverse":
        record["strategy"] = "REVERSE: " + str(record.get("strategy", ""))
    return {
        "advantage": advantage,
        "preference": "balanced" if advantage > 0 else "baseline",
        "direction": direction,
        "write": True,
        "record": record,
    }


def sign_memory_record(
    *,
    arm: str,
    event: Mapping[str, Any],
    payload: Mapping[str, Any],
    admitted_for_event_index: int,
    direction: str,
) -> JsonDict:
    """Sign every field that controls one future-visible memory record."""

    record: JsonDict = {
        "record_id": sha256_json(
            {
                "arm": arm,
                "event_id": event["event_id"],
                "admitted_for_event_index": admitted_for_event_index,
                "direction": direction,
                "payload": payload,
            }
        ),
        "arm": arm,
        "source_event_id": event["event_id"],
        "source_event_index": int(event["chronology_index"]),
        "admitted_for_event_index": int(admitted_for_event_index),
        "direction": direction,
        "payload": deepcopy(dict(payload)),
    }
    record["signature"] = sha256_json(record)
    return record


def signature_valid(record: Mapping[str, Any]) -> bool:
    """Recompute the signature over all signed record fields."""

    return record.get("signature") == sha256_json(
        {key: value for key, value in record.items() if key != "signature"}
    )


class TransactionalMemory:
    """Store signed external records with atomic commit and exact rollback."""

    def __init__(self, root: Path, name: str, *, max_records: int = MAX_MEMORY_RECORDS):
        self.root = root / name
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.max_records = max_records
        self._records: list[JsonDict] = []
        write_json_atomic(self.state_path, {"records": self._records})

    def state_hash(self) -> str:
        """Hash exact active state bytes, including formatting and order."""

        return sha256_bytes(self.state_path.read_bytes())

    def retrieve(self, family: str, *, current_event_index: int) -> list[JsonDict]:
        """Return only records admitted by a strictly earlier event."""

        return [
            deepcopy(row)
            for row in reversed(self._records)
            if row["source_event_index"] < current_event_index
            and row["admitted_for_event_index"] <= current_event_index
            and row["payload"].get("family") == family
        ][: self.max_records]

    def commit(
        self,
        record: Mapping[str, Any],
        *,
        current_event_index: int,
        force_failure: bool = False,
    ) -> JsonDict:
        """Stage a signed child, publish atomically, or restore exact parent bytes."""

        candidate = deepcopy(dict(record))
        if not signature_valid(candidate):
            raise ProtocolError("memory_signature_invalid")
        if current_event_index <= int(candidate["source_event_index"]):
            raise ProtocolError("same_event_write")
        if int(candidate["admitted_for_event_index"]) != current_event_index:
            raise ProtocolError("admission_index_mismatch")
        parent_bytes = self.state_path.read_bytes()
        parent_hash = sha256_bytes(parent_bytes)
        next_records = deepcopy(self._records)
        family = candidate["payload"].get("family")
        same = [row for row in next_records if row["payload"].get("family") == family]
        conflict = None
        if same:
            replaced = min(same, key=lambda row: (row["source_event_index"], row["record_id"]))
            next_records.remove(replaced)
            conflict = {
                "family": family,
                "incoming_record_id": candidate["record_id"],
                "replaced_record_id": replaced["record_id"],
                "resolution": "replace_older_same_family",
            }
        next_records.append(candidate)
        next_records.sort(key=lambda row: (row["source_event_index"], row["record_id"]))
        evicted = []
        while len(next_records) > self.max_records:
            evicted.append(next_records.pop(0)["record_id"])
        staged_path = self.root / f"staged-{candidate['record_id'][7:23]}.json"
        write_json_atomic(staged_path, {"records": next_records})
        staged_hash = sha256_bytes(staged_path.read_bytes())
        if force_failure:
            self.state_path.write_bytes(parent_bytes)
            staged_path.unlink()
            return {
                "record_id": candidate["record_id"],
                "arm": candidate["arm"],
                "source_event_id": candidate["source_event_id"],
                "terminal_state": "rolled_back",
                "parent_state_hash": parent_hash,
                "staged_state_hash": staged_hash,
                "final_state_hash": self.state_hash(),
                "parent_restored": self.state_path.read_bytes() == parent_bytes,
                "conflict": conflict,
                "evicted_record_ids": evicted,
            }
        staged_path.replace(self.state_path)
        self._records = next_records
        return {
            "record_id": candidate["record_id"],
            "arm": candidate["arm"],
            "source_event_id": candidate["source_event_id"],
            "terminal_state": "committed",
            "parent_state_hash": parent_hash,
            "staged_state_hash": staged_hash,
            "final_state_hash": self.state_hash(),
            "parent_restored": None,
            "conflict": conflict,
            "evicted_record_ids": evicted,
        }

    def refresh(self, *, boundary_event_index: int) -> JsonDict:
        """Refresh only at a sealed boundary and retain the fixed record bound."""

        before = len(self._records)
        eligible = [
            row for row in self._records if row["source_event_index"] < boundary_event_index
        ]
        eligible = eligible[-self.max_records :]
        self._records = deepcopy(eligible)
        write_json_atomic(self.state_path, {"records": self._records})
        return {
            "store": self.root.name,
            "boundary_event_index": boundary_event_index,
            "record_count_before": before,
            "record_count_after": len(self._records),
            "max_records": self.max_records,
            "future_record_visible": any(
                row["source_event_index"] >= boundary_event_index for row in self._records
            ),
            "state_hash": self.state_hash(),
        }


def _render_memory(records: Sequence[Mapping[str, Any]]) -> str:
    """Render bounded records without exposing signatures or hidden labels."""

    if not records:
        return "No prior external guidance."
    return "\n".join(
        f"{row['direction'].upper()}: {row['payload'].get('strategy', row['payload'].get('trace', ''))}"
        for row in records
    )


def _initial_strategy(family: str) -> str:
    """Give the balanced arm one frozen strategy before it learns a record."""

    strategies = {
        "sat_logic": "Candidate strategy: assign every Boolean symbol, then check each clause.",
        "graph_coloring": "Candidate strategy: assign every node, then check each listed edge.",
        "bounded_scheduling": "Candidate strategy: assign every start, then check precedence, overlap, horizon, and makespan.",
    }
    return strategies.get(family, "Candidate strategy: check every requested field and constraint.")


def _memory_candidate(
    arm: str, event: Mapping[str, Any], outcome: Mapping[str, Any], response: Mapping[str, Any]
) -> JsonDict:
    """Build one later-readable candidate from the arm's sealed action and outcome."""

    family = str(event["constraint_family"])
    success = outcome.get("exact_success") is True
    if arm == "equal_budget_raw_trace":
        return {
            "family": family,
            "trace": str(response.get("raw_text", ""))[:384],
            "strategy": f"Prior exact {'success' if success else 'failure'} trace: {str(response.get('raw_text', ''))[:320]}",
        }
    if arm == "delayed_procedural_memory":
        strategy = (
            "Return one complete direct JSON object with every requested symbol."
            if success
            else "Check the JSON schema, every symbol, every constraint, and the requested objective before answering."
        )
        return {"family": family, "strategy": strategy}
    return {
        "family": family,
        "strategy": str(response.get("reasoning_text") or response.get("raw_text") or "")[:384],
    }


def _rate(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Return exact-success rate, or no value for an empty slice."""

    return sum(row.get("exact_success") is True for row in rows) / len(rows) if rows else None


def _paired_interval(deltas: Sequence[int]) -> tuple[float | None, float | None, float | None]:
    """Return a two-sided normal interval for paired binary differences."""

    if not deltas:
        return None, None, None
    mean = statistics.fmean(deltas)
    if len(deltas) == 1:
        return mean, mean, mean
    half = 1.96 * statistics.stdev(deltas) / math.sqrt(len(deltas))
    return mean, mean - half, mean + half


def _reduce_metrics(
    event_rows: Sequence[Mapping[str, Any]], advantage_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Reduce all value, harm, retention, latency, token, and cost rows."""

    by_arm = {arm: [row for row in event_rows if row["arm"] == arm] for arm in ARMS}
    future = {
        arm: [row for row in rows if row["split"] == "future"] for arm, rows in by_arm.items()
    }
    protected = {
        arm: [row for row in rows if row["split"] == "protected_retention"]
        for arm, rows in by_arm.items()
    }
    baseline_future = {row["event_id"]: row for row in future["no_memory"]}
    baseline_all = {row["event_id"]: row for row in by_arm["no_memory"]}
    baseline_protected_rate = _rate(protected["no_memory"])
    future_success_rows = [
        {
            "arm": arm,
            "event_count": len(future[arm]),
            "exact_success_count": sum(row["exact_success"] is True for row in future[arm]),
            "exact_success_rate": _rate(future[arm]),
        }
        for arm in ARMS
    ]
    paired_interval_rows = []
    for arm in ARMS[1:]:
        deltas = [
            int(row["exact_success"] is True)
            - int(baseline_future[row["event_id"]]["exact_success"] is True)
            for row in future[arm]
        ]
        mean, lower, upper = _paired_interval(deltas)
        paired_interval_rows.append(
            {
                "comparison": f"{arm}_vs_no_memory",
                "arm": arm,
                "split": "future",
                "paired_event_count": len(deltas),
                "mean_delta": mean,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "method": "normal_interval_on_paired_binary_differences",
            }
        )
    protected_retention_rows = [
        {
            "arm": arm,
            "event_count": len(protected[arm]),
            "exact_success_rate": _rate(protected[arm]),
            "no_memory_exact_success_rate": baseline_protected_rate,
            "retention_delta_vs_no_memory": (
                None
                if _rate(protected[arm]) is None or baseline_protected_rate is None
                else _rate(protected[arm]) - baseline_protected_rate
            ),
        }
        for arm in ARMS
    ]
    negative_transfer_rows = []
    stale_memory_rows = []
    for arm in ARMS[1:]:
        rows = by_arm[arm]
        negative = [
            row
            for row in rows
            if baseline_all[row["event_id"]]["exact_success"] is True
            and row["exact_success"] is not True
        ]
        stale = [row for row in negative if row["memory_record_ids"]]
        negative_transfer_rows.append(
            {
                "arm": arm,
                "negative_transfer_count": len(negative),
                "event_ids": [row["event_id"] for row in negative],
                "rate": len(negative) / len(rows) if rows else None,
            }
        )
        stale_memory_rows.append(
            {
                "arm": arm,
                "memory_hit_count": sum(bool(row["memory_record_ids"]) for row in rows),
                "stale_harm_count": len(stale),
                "event_ids": [row["event_id"] for row in stale],
            }
        )
    latency_rows = []
    token_rows = []
    cost_rows = []
    arm_rows = []
    for arm, rows in by_arm.items():
        seconds = sum(float(row["latency_s"]) for row in rows)
        prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
        completion_tokens = sum(int(row["completion_tokens"]) for row in rows)
        cost = seconds * LOCAL_GPU_USD_PER_HOUR / 3_600.0
        latency_rows.append(
            {
                "arm": arm,
                "event_count": len(rows),
                "total_latency_s": seconds,
                "mean_latency_s": seconds / len(rows) if rows else None,
                "latency_per_event_s": seconds / len(rows) if rows else None,
            }
        )
        token_rows.append(
            {
                "arm": arm,
                "event_count": len(rows),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens_per_event": (prompt_tokens + completion_tokens) / len(rows)
                if rows
                else None,
            }
        )
        cost_rows.append(
            {
                "arm": arm,
                "event_count": len(rows),
                "assumed_local_gpu_usd_per_hour": LOCAL_GPU_USD_PER_HOUR,
                "total_cost_usd": cost,
                "cost_per_event_usd": cost / len(rows) if rows else None,
            }
        )
        arm_rows.append(
            {
                "arm": arm,
                "event_count": len(rows),
                "exact_success_count": sum(row["exact_success"] is True for row in rows),
                "exact_success_rate": _rate(rows),
                "prompt_budget_bytes": PROMPT_BUDGET_BYTES,
                "memory_budget_bytes": MEMORY_BUDGET_BYTES,
                "max_generation_tokens": MAX_GENERATION_TOKENS,
            }
        )
    abstention_rows = [
        {
            "event_id": row["event_id"],
            "chronology_index": row["chronology_index"],
            "preference": row["preference"],
            "advantage": row["advantage"],
            "write": False,
        }
        for row in advantage_rows
        if row.get("write") is False
    ]
    return {
        "arm_rows": arm_rows,
        "future_success_rows": future_success_rows,
        "stale_memory_rows": stale_memory_rows,
        "protected_retention_rows": protected_retention_rows,
        "negative_transfer_rows": negative_transfer_rows,
        "abstention_rows": abstention_rows,
        "latency_rows": latency_rows,
        "token_rows": token_rows,
        "cost_rows": cost_rows,
        "paired_interval_rows": paired_interval_rows,
    }


def _future_uplift_supported(metrics: Mapping[str, Any]) -> int:
    """Derive support only from the balanced paired interval and retention rows."""

    interval = next(
        row
        for row in metrics["paired_interval_rows"]
        if row["arm"] == "verifier_balanced_strategy_memory"
    )
    retention = next(
        row
        for row in metrics["protected_retention_rows"]
        if row["arm"] == "verifier_balanced_strategy_memory"
    )
    return int(
        interval["mean_delta"] is not None
        and interval["mean_delta"] > 0
        and interval["ci95_lower"] > 0
        and retention["retention_delta_vs_no_memory"] is not None
        and retention["retention_delta_vs_no_memory"] >= 0
    )


def _blocked_artifact(
    artifact: JsonDict, error: PreconditionError, *, duration_s: float
) -> JsonDict:
    """Finish a failed prerequisite without dropping any required field."""

    check = gate_row(error.check, error.expected, error.observed, False)
    artifact["duration_s"] = duration_s
    artifact["preconditions_checked"] = {"all_passed": False, "checks": [check]}
    artifact["gate_check_summary"] = gate_summary([check])
    artifact["inference_substrate_class"] = "blocked_no_run"
    artifact["flowbalance_memory_csl_complete_score"] = 0
    artifact["future_uplift_supported_score"] = 0
    artifact["verdict_class"] = "blocked"
    artifact["honest_verdict"] = f"blocked_{error.check}"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _status(rows: list[JsonDict], path: Path, phase: str, state: str, **details: Any) -> None:
    """Print and persist one flushed phase, request, transaction, or checkpoint receipt."""

    row = {"sequence": len(rows), "phase": phase, "state": state, **details}
    rows.append(row)
    print(
        "exp7142 " + " ".join(f"{key}={value}" for key, value in row.items()),
        flush=True,
    )
    write_json_atomic(path, {"experiment_id": EXPERIMENT_ID, "status_receipt_rows": rows})


def _load_source(path: Path) -> JsonDict:
    """Load one upstream object and preserve exact diagnostics on failure."""

    if not path.is_file():
        raise PreconditionError("experiment_7141_v627_csl_event_stream.exists", True, False)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreconditionError(
            "experiment_7141_v627_csl_event_stream.valid_json",
            "JSON object",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    if not isinstance(value, dict):
        raise PreconditionError(
            "experiment_7141_v627_csl_event_stream.valid_json", "JSON object", type(value).__name__
        )
    return value


def _default_exact_scorer(
    bank_path: Path = BANK_PATH,
) -> ExactScore:  # pragma: no cover - live source boundary.
    """Build an exact checker from immutable solver receipts, never model labels."""

    from carnot.experiment_7129_v626_sota_constraint_bank import (
        parse_model_text,
        verify_direct_answer,
    )

    bank = _load_source(bank_path)
    receipts = {str(row["instance_id"]): row for row in bank.get("solver_receipt_rows", [])}

    def score(event: Mapping[str, Any], response: Mapping[str, Any]) -> JsonDict:
        parsed = parse_model_text(
            str(response.get("raw_text", "")), str(event["constraint_family"])
        )
        if parsed["parse_success"] is not True:
            return {
                "exact_success": False,
                "parse_success": False,
                "failed_constraint_classes": [f"parse:{parsed['parse_error']}"],
            }
        receipt = receipts.get(str(event["instance_id"]))
        if receipt is None:
            raise PreconditionError("exact_solver_receipt", "present", event["instance_id"])
        exact = verify_direct_answer(receipt, parsed["parsed"])
        return {
            "exact_success": bool(exact["exact_correct"]),
            "parse_success": True,
            "failed_constraint_classes": []
            if exact["exact_correct"]
            else ["exact_constraint_or_objective"],
        }

    return score


class _LiveSession:  # pragma: no cover - live llama.cpp and CUDA boundary.
    """Keep one frozen llama.cpp model resident for all chronological requests."""

    def __init__(self, model_path: Path):
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=PROMPT_BUDGET_BYTES,
            n_gpu_layers=-1,
            n_batch=512,
            n_ubatch=512,
            main_gpu=0,
            split_mode=1,
            tensor_split=[0.5, 0.5],
            seed=RANDOM_SEED,
            use_mmap=True,
            verbose=False,
        )

    def call(
        self,
        *,
        event: Mapping[str, Any],
        arm: str,
        prompt: str,
        seed: int,
        max_tokens: int,
    ) -> JsonDict:
        del event, arm
        started = time.perf_counter()
        try:
            result = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "Return only a direct JSON answer. Do not use tools or answer IDs.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                seed=seed,
                response_format={"type": "json_object"},
            )
            choice = (list(result.get("choices") or [{}]) or [{}])[0]
            message = dict(choice.get("message") or {})
            usage = dict(result.get("usage") or {})
            return {
                "raw_text": str(message.get("content") or ""),
                "reasoning_text": str(
                    message.get("reasoning_content") or message.get("reasoning") or ""
                ),
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "duration_s": time.perf_counter() - started,
                "terminal_state": "complete",
            }
        except Exception as exc:  # noqa: BLE001 - the failed request must remain a row.
            return {
                "raw_text": "",
                "reasoning_text": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "duration_s": time.perf_counter() - started,
                "terminal_state": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def close(self) -> None:
        """Release model state without changing the cached weight file."""

        close = getattr(self.llm, "close", None)
        if callable(close):
            close()
        gc.collect()


def _nvidia_rows() -> list[JsonDict]:  # pragma: no cover - live host boundary.
    """Read the two local CUDA devices without using a network service."""

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 6:
            rows.append(
                {
                    "index": int(values[0]),
                    "uuid": values[1],
                    "name": values[2],
                    "memory_total_mb": int(values[3]),
                    "memory_used_mb": int(values[4]),
                    "utilization_gpu_pct": int(values[5]),
                }
            )
    return rows


def live_preflight(  # pragma: no cover - live model, GPU, and filesystem boundary.
    *,
    model_specs: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    transaction_root: Path,
    source_artifact: Mapping[str, Any],
) -> JsonDict:
    """Check every external prerequisite and keep the live model for the run."""

    from llama_cpp import llama_cpp

    checks = []
    model = dict(model_specs[0])
    model_path = Path(str(model["model_path"]))
    checks.append(gate_row("cached_qwen_q4", True, model_path.is_file(), model_path.is_file()))
    outcomes = list(source_artifact.get("outcome_reveal_rows") or [])
    event_ids = [str(row.get("event_id")) for row in events]
    outcome_ids = [str(row.get("event_id")) for row in outcomes]
    exact_reveal_ok = len(outcomes) == len(events) and outcome_ids == event_ids
    checks.append(gate_row("exact_outcome_reveal", True, exact_reveal_ok, exact_reveal_ok))
    devices = _nvidia_rows()
    gpu_ok = len(devices) == 2 and all("RTX 3090" in row["name"] for row in devices)
    checks.append(gate_row("dual_rtx_3090_gpu", 2, devices, gpu_ok))
    backend_ok = bool(llama_cpp.llama_supports_gpu_offload())
    checks.append(gate_row("cuda_llama_cpp_backend", True, backend_ok, backend_ok))
    try:
        transaction_root.mkdir(parents=True, exist_ok=True)
        probe = transaction_root / ".write-probe"
        probe.write_bytes(b"transactional-store-probe")
        storage_ok = probe.read_bytes() == b"transactional-store-probe"
        probe.unlink()
    except OSError as exc:
        storage_ok = False
        checks.append(gate_row("writable_transactional_store", True, str(exc), False))
    else:
        checks.append(gate_row("writable_transactional_store", True, storage_ok, storage_ok))
    if not all(row["passed"] for row in checks):
        return {
            "all_passed": False,
            "checks": checks,
            "model_identity_rows": [],
            "model_load_receipts": [],
        }
    loaded_at = time.perf_counter()
    try:
        session = _LiveSession(model_path)
        metadata = dict(getattr(session.llm, "metadata", {}) or {})
        templates = {
            str(key): value
            for key, value in metadata.items()
            if "chat_template" in str(key).lower()
        }
        template_ok = bool(templates)
        checks.append(gate_row("embedded_chat_template", True, template_ok, template_ok))
        smoke = session.call(
            event=events[0],
            arm="preflight",
            prompt='/no_think Return only {"ok":true}.',
            seed=RANDOM_SEED,
            max_tokens=16,
        )
        smoke_ok = smoke["terminal_state"] == "complete" and bool(smoke["raw_text"])
        checks.append(gate_row("one_live_generation", True, smoke_ok, smoke_ok))
    except Exception as exc:  # noqa: BLE001 - exact load failure belongs in the gate.
        checks.append(gate_row("one_live_generation", True, f"{type(exc).__name__}: {exc}", False))
        return {
            "all_passed": False,
            "checks": checks,
            "model_identity_rows": [],
            "model_load_receipts": [],
        }
    identity = {
        "model_id": QWEN_ID,
        "model_path": str(model_path),
        "revision": model["revision"],
        "model_sha256": model["model_sha256"],
        "backend": "llama_cpp_cuda",
        "device": [row["index"] for row in devices],
        "chat_template_source": "embedded_gguf",
        "chat_template_hash": sha256_json(templates),
    }
    receipt = {
        "request_kind": "preflight_smoke",
        "terminal_state": smoke["terminal_state"],
        "response_hash": sha256_text(smoke["raw_text"]),
        "prompt_tokens": smoke["prompt_tokens"],
        "completion_tokens": smoke["completion_tokens"],
        "load_and_smoke_duration_s": time.perf_counter() - loaded_at,
    }
    return {
        "all_passed": all(row["passed"] for row in checks),
        "checks": checks,
        "model_identity_rows": [identity],
        "model_load_receipts": [receipt],
        "_session": session,
        "_model_call": session.call,
        "_exact_score": _default_exact_scorer(),
    }


def _source_hashes(
    source_path: Path, model_specs: Sequence[Mapping[str, Any]]
) -> JsonDict:  # pragma: no cover - live repository boundary.
    """Bind the run to the requested source, code, tests, spec, and model."""

    paths = (
        source_path,
        BANK_PATH,
        REPO_ROOT / "python/carnot/experiment_7142_v627_flowbalance_memory_csl.py",
        REPO_ROOT / "tests/python/test_experiment_7142_v627_flowbalance_memory_csl.py",
        REPO_ROOT / "openspec/capabilities/self-learning/spec.md",
        REPO_ROOT / "research-program.md",
        REPO_ROOT / "research-references.md",
        REPO_ROOT / "results/experiment_7106_v623_procedural_memory_csl.json",
        REPO_ROOT / "results/experiment_6978_transactional_constraint_self_learning.json",
    )
    result = {
        (
            str(path.relative_to(REPO_ROOT))
            if path.is_relative_to(REPO_ROOT)
            else f"external_source:{path.name}"
        ): sha256_path(path)
        for path in paths
    }
    result["model:unsloth/Qwen3.6-35B-A3B-GGUF"] = model_specs[0].get("model_sha256")
    return result


def run_experiment(
    *,
    run_date: str = RUN_DATE,
    result_path: Path = RESULT_PATH,
    transaction_root: Path = TRANSACTION_ROOT,
    source_path: Path = SOURCE_PATH,
    events: Sequence[Mapping[str, Any]] | None = None,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    preflight_func: Callable[..., Mapping[str, Any]] = live_preflight,
    model_call: ModelCall | None = None,
    exact_score: ExactScore | None = None,
    expected_event_count: int = EXPECTED_EVENT_COUNT,
) -> JsonDict:
    """Initialize, gate, act, reveal, transact, reduce, validate, and publish."""

    started = time.perf_counter()
    print("exp7142 phase=initialize state=start", flush=True)
    artifact = initialize_artifact(result_path, run_date)
    status_rows: list[JsonDict] = []
    status_path = transaction_root / "status.json"
    _status(status_rows, status_path, "initialize", "complete", result_path=str(result_path))
    try:
        _status(status_rows, status_path, "source_gate", "start")
        source = _load_source(source_path)
        ready = source.get("csl_event_stream_ready_score")
        if ready != 1:
            raise PreconditionError(
                "experiment_7141_v627_csl_event_stream.csl_event_stream_ready_score", 1, ready
            )
        event_values = [deepcopy(dict(row)) for row in (events or source.get("event_rows") or [])]
        if len(event_values) != expected_event_count:
            raise PreconditionError("frozen_event_count", expected_event_count, len(event_values))
        if [row.get("chronology_index") for row in event_values] != list(
            range(expected_event_count)
        ):
            raise PreconditionError(
                "frozen_event_chronology",
                list(range(expected_event_count)),
                [row.get("chronology_index") for row in event_values],
            )
        _status(status_rows, status_path, "source_gate", "complete", event_count=len(event_values))
        specs = (
            [deepcopy(dict(row)) for row in model_specs]
            if model_specs is not None
            else resolve_model_specs()
        )
        _status(status_rows, status_path, "preflight", "start")
        preflight = dict(
            preflight_func(
                model_specs=specs,
                events=event_values,
                transaction_root=transaction_root,
                source_artifact=source,
            )
        )
        if preflight.get("all_passed") is not True:
            failed = next(
                (row for row in preflight.get("checks", []) if row.get("passed") is not True),
                gate_row("preflight", True, False, False),
            )
            raise PreconditionError(
                str(failed["check"]), failed.get("expected_value"), failed.get("observed_value")
            )
        active_call = model_call or preflight.get("_model_call")
        active_score = exact_score or preflight.get("_exact_score")
        if not callable(active_call) or not callable(active_score):
            raise PreconditionError("live_model_and_exact_checker", True, False)
        _status(status_rows, status_path, "preflight", "complete")
    except PreconditionError as exc:
        artifact = _blocked_artifact(artifact, exc, duration_s=time.perf_counter() - started)
        _status(status_rows, status_path, "terminal", "blocked", failed_check=exc.check)
        artifact["status_receipt_rows"] = deepcopy(status_rows)
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        write_json_atomic(result_path, artifact)
        return artifact

    artifact["MODEL_SPECS"] = specs
    artifact["model_identity_rows"] = [
        deepcopy(dict(row)) for row in preflight.get("model_identity_rows", [])
    ]
    artifact["model_load_receipts"] = [
        deepcopy(dict(row)) for row in preflight.get("model_load_receipts", [])
    ]
    artifact["frozen_plan"] = {
        "event_ids": [row["event_id"] for row in event_values],
        "event_hash": sha256_json([row["event_content_hash"] for row in event_values]),
        "arms": list(ARMS),
        "arm_order": list(ARMS),
        "prompt_budget_bytes": PROMPT_BUDGET_BYTES,
        "memory_budget_bytes": MEMORY_BUDGET_BYTES,
        "max_generation_tokens": MAX_GENERATION_TOKENS,
        "random_seed": RANDOM_SEED,
        "split_boundaries": [
            {
                "split": split,
                "first_index": min(
                    row["chronology_index"] for row in event_values if row["split"] == split
                ),
                "last_index": max(
                    row["chronology_index"] for row in event_values if row["split"] == split
                ),
            }
            for split in dict.fromkeys(row["split"] for row in event_values)
        ],
    }

    stores = {
        "equal_budget_raw_trace": TransactionalMemory(
            transaction_root, "raw", max_records=MAX_MEMORY_RECORDS
        ),
        "delayed_procedural_memory": TransactionalMemory(
            transaction_root, "procedural", max_records=MAX_MEMORY_RECORDS
        ),
        "verifier_balanced_strategy_memory": TransactionalMemory(
            transaction_root, "balanced", max_records=MAX_MEMORY_RECORDS
        ),
    }
    event_rows: list[JsonDict] = []
    action_rows: list[JsonDict] = []
    outcome_rows: list[JsonDict] = []
    advantage_rows: list[JsonDict] = []
    candidate_rows: list[JsonDict] = []
    transaction_rows: list[JsonDict] = []
    signature_rows: list[JsonDict] = []
    conflict_rows: list[JsonDict] = []
    rollback_rows: list[JsonDict] = []
    refresh_rows: list[JsonDict] = []
    learning_rows: list[JsonDict] = []
    rollback_probe_done = False
    _status(
        status_rows,
        status_path,
        "chronological_run",
        "start",
        planned_requests=len(event_values) * len(ARMS),
    )
    try:
        for event in event_values:
            index = int(event["chronology_index"])
            family = str(event["constraint_family"])
            visible = {
                "no_memory": [],
                **{
                    arm: store.retrieve(family, current_event_index=index)
                    for arm, store in stores.items()
                },
            }
            memory_text = {arm: _render_memory(visible[arm]) for arm in ARMS}
            if not visible["verifier_balanced_strategy_memory"]:
                memory_text["verifier_balanced_strategy_memory"] = _initial_strategy(family)
            prompts = build_prompt_rows(event, memory_text)
            responses: JsonDict = {}
            prompt_by_arm = {row["arm"]: row for row in prompts}
            for arm in ARMS:
                request_number = len(event_rows) + len(responses) + 1
                _status(
                    status_rows,
                    status_path,
                    "model_request",
                    "start",
                    request=request_number,
                    event=index,
                    arm=arm,
                )
                prompt_row = prompt_by_arm[arm]
                response = dict(
                    active_call(
                        event=event,
                        arm=arm,
                        prompt=prompt_row["prompt"],
                        seed=prompt_row["seed"],
                        max_tokens=prompt_row["max_tokens"],
                    )
                )
                responses[arm] = response
                _status(
                    status_rows,
                    status_path,
                    "model_response",
                    str(response.get("terminal_state")),
                    request=request_number,
                    event=index,
                    arm=arm,
                    response_hash=sha256_text(str(response.get("raw_text", ""))),
                )
            receipts = seal_action_receipts(event, prompts, responses)
            reveals = reveal_outcomes(event, responses, receipts, active_score)
            action_rows.extend(receipts)
            outcome_rows.extend(reveals)
            reveal_by_arm = {row["arm"]: row for row in reveals}
            for arm in ARMS:
                prompt_row = prompt_by_arm[arm]
                response = responses[arm]
                reveal = reveal_by_arm[arm]
                event_rows.append(
                    {
                        "event_id": event["event_id"],
                        "chronology_index": index,
                        "split": event["split"],
                        "constraint_family": family,
                        "hardness": event.get("hardness"),
                        "arm": arm,
                        "model_id": QWEN_ID,
                        "prompt_hash": prompt_row["prompt_hash"],
                        "prompt_bytes": prompt_row["prompt_bytes"],
                        "memory_hash": prompt_row["memory_hash"],
                        "memory_bytes": prompt_row["memory_bytes"],
                        "memory_record_ids": [row["record_id"] for row in visible[arm]],
                        "seed": prompt_row["seed"],
                        "max_tokens": prompt_row["max_tokens"],
                        "raw_text": str(response.get("raw_text", "")),
                        "reasoning_text": str(response.get("reasoning_text", "")),
                        "response_hash": sha256_text(str(response.get("raw_text", ""))),
                        "terminal_state": response.get("terminal_state"),
                        "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(response.get("completion_tokens", 0) or 0),
                        "latency_s": float(response.get("duration_s", 0.0) or 0.0),
                        "action_receipt_hash": next(
                            row["receipt_hash"] for row in receipts if row["arm"] == arm
                        ),
                        "outcome_receipt_hash": reveal["outcome_receipt_hash"],
                        "exact_success": reveal["exact_success"],
                        "parse_success": reveal["parse_success"],
                        "failed_constraint_classes": reveal["failed_constraint_classes"],
                    }
                )
            for arm in ("equal_budget_raw_trace", "delayed_procedural_memory"):
                candidate = _memory_candidate(arm, event, reveal_by_arm[arm], responses[arm])
                candidate_rows.append(
                    {
                        "event_id": event["event_id"],
                        "arm": arm,
                        "direction": "retain",
                        "write": True,
                        "payload": candidate,
                    }
                )
                record = sign_memory_record(
                    arm=arm,
                    event=event,
                    payload=candidate,
                    admitted_for_event_index=index + 1,
                    direction="retain",
                )
                signature_rows.append(
                    {"record": record, "signature_valid": signature_valid(record)}
                )
                receipt = stores[arm].commit(record, current_event_index=index + 1)
                transaction_rows.append(receipt)
                if receipt["conflict"]:
                    conflict_rows.append(
                        {"arm": arm, "event_id": event["event_id"], **receipt["conflict"]}
                    )
                _status(
                    status_rows,
                    status_path,
                    "transaction",
                    receipt["terminal_state"],
                    arm=arm,
                    event=index,
                    record_id=record["record_id"],
                )
            balanced_candidate = _memory_candidate(
                "verifier_balanced_strategy_memory",
                event,
                reveal_by_arm["verifier_balanced_strategy_memory"],
                responses["verifier_balanced_strategy_memory"],
            )
            update = flowbalance_update(
                balanced_candidate,
                balanced_success=reveal_by_arm["verifier_balanced_strategy_memory"][
                    "exact_success"
                ],
                baseline_success=reveal_by_arm["no_memory"]["exact_success"],
            )
            advantage = {
                "event_id": event["event_id"],
                "chronology_index": index,
                "balanced_exact_success": reveal_by_arm["verifier_balanced_strategy_memory"][
                    "exact_success"
                ],
                "no_memory_exact_success": reveal_by_arm["no_memory"]["exact_success"],
                **{key: update[key] for key in ("advantage", "preference", "direction", "write")},
            }
            advantage_rows.append(advantage)
            candidate_rows.append(
                {
                    "event_id": event["event_id"],
                    "arm": "verifier_balanced_strategy_memory",
                    "payload": balanced_candidate,
                    **{
                        key: update[key]
                        for key in ("advantage", "preference", "direction", "write")
                    },
                }
            )
            if update["write"]:
                record = sign_memory_record(
                    arm="verifier_balanced_strategy_memory",
                    event=event,
                    payload=update["record"],
                    admitted_for_event_index=index + 1,
                    direction=update["direction"],
                )
                signature_rows.append(
                    {"record": record, "signature_valid": signature_valid(record)}
                )
                receipt = stores["verifier_balanced_strategy_memory"].commit(
                    record, current_event_index=index + 1
                )
                transaction_rows.append(receipt)
                if receipt["conflict"]:
                    conflict_rows.append(
                        {
                            "arm": "verifier_balanced_strategy_memory",
                            "event_id": event["event_id"],
                            **receipt["conflict"],
                        }
                    )
                _status(
                    status_rows,
                    status_path,
                    "transaction",
                    receipt["terminal_state"],
                    arm="verifier_balanced_strategy_memory",
                    event=index,
                    record_id=record["record_id"],
                )
            if not rollback_probe_done:
                probe = sign_memory_record(
                    arm="verifier_balanced_strategy_memory",
                    event=event,
                    payload={"family": "rollback_probe", "strategy": "must never publish"},
                    admitted_for_event_index=index + 1,
                    direction="reverse",
                )
                signature_rows.append(
                    {
                        "record": probe,
                        "signature_valid": signature_valid(probe),
                        "rollback_probe": True,
                    }
                )
                rolled = stores["verifier_balanced_strategy_memory"].commit(
                    probe, current_event_index=index + 1, force_failure=True
                )
                transaction_rows.append(rolled)
                rollback_rows.append(rolled)
                rollback_probe_done = True
                _status(
                    status_rows,
                    status_path,
                    "transaction",
                    "rolled_back",
                    arm="verifier_balanced_strategy_memory",
                    event=index,
                    record_id=probe["record_id"],
                )
            if (index + 1) % EPISODE_SIZE == 0 or index + 1 == len(event_values):
                for arm, store in stores.items():
                    refresh = store.refresh(boundary_event_index=index + 1)
                    refresh_rows.append({"arm": arm, **refresh})
                    _status(
                        status_rows,
                        status_path,
                        "checkpoint",
                        "complete",
                        arm=arm,
                        boundary=index + 1,
                    )
            for arm in ARMS:
                prefix = [
                    row
                    for row in event_rows
                    if row["arm"] == arm and row["chronology_index"] <= index
                ]
                learning_rows.append(
                    {
                        "arm": arm,
                        "through_event_index": index,
                        "event_count": len(prefix),
                        "cumulative_exact_success_rate": _rate(prefix),
                    }
                )
    finally:
        session = preflight.get("_session")
        if session is not None:
            session.close()

    metrics = _reduce_metrics(event_rows, advantage_rows)
    supported = _future_uplift_supported(metrics)
    source_checks = [
        gate_row(
            "experiment_7141_v627_csl_event_stream.csl_event_stream_ready_score",
            1,
            source.get("csl_event_stream_ready_score"),
            source.get("csl_event_stream_ready_score") == 1,
        ),
        gate_row(
            "planned_event_arm_rows",
            expected_event_count * len(ARMS),
            len(event_rows),
            len(event_rows) == expected_event_count * len(ARMS),
        ),
        gate_row(
            "planned_action_receipts",
            expected_event_count * len(ARMS),
            len(action_rows),
            len(action_rows) == expected_event_count * len(ARMS),
        ),
        gate_row(
            "planned_outcome_receipts",
            expected_event_count * len(ARMS),
            len(outcome_rows),
            len(outcome_rows) == expected_event_count * len(ARMS),
        ),
        gate_row(
            "model_requests_terminal",
            True,
            all(row["terminal_state"] == "complete" for row in event_rows),
            all(row["terminal_state"] == "complete" for row in event_rows),
        ),
        gate_row(
            "model_load_receipts_terminal",
            True,
            bool(artifact["model_load_receipts"])
            and all(
                row.get("terminal_state") == "complete" for row in artifact["model_load_receipts"]
            ),
            bool(artifact["model_load_receipts"])
            and all(
                row.get("terminal_state") == "complete" for row in artifact["model_load_receipts"]
            ),
        ),
        gate_row(
            "transaction_receipts_terminal",
            True,
            bool(transaction_rows)
            and all(
                row["terminal_state"] in {"committed", "rolled_back"} for row in transaction_rows
            ),
            bool(transaction_rows)
            and all(
                row["terminal_state"] in {"committed", "rolled_back"} for row in transaction_rows
            ),
        ),
        gate_row(
            "signatures_valid",
            True,
            all(row["signature_valid"] for row in signature_rows),
            all(row["signature_valid"] for row in signature_rows),
        ),
        gate_row(
            "rollback_parent_restored",
            True,
            bool(rollback_rows) and all(row["parent_restored"] is True for row in rollback_rows),
            bool(rollback_rows) and all(row["parent_restored"] is True for row in rollback_rows),
        ),
        gate_row(
            "matched_prompt_and_memory_budgets",
            True,
            all(
                row["prompt_bytes"] == PROMPT_BUDGET_BYTES
                and row["memory_bytes"] == MEMORY_BUDGET_BYTES
                for row in event_rows
            ),
            all(
                row["prompt_bytes"] == PROMPT_BUDGET_BYTES
                and row["memory_bytes"] == MEMORY_BUDGET_BYTES
                for row in event_rows
            ),
        ),
        gate_row("model_weights_changed", False, False, True),
    ]
    checks = [*source_checks, *[deepcopy(dict(row)) for row in preflight.get("checks", [])]]
    complete = int(all(row["passed"] for row in checks))
    artifact.update(
        {
            "preconditions_checked": {"all_passed": bool(complete), "checks": checks},
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "duration_s": time.perf_counter() - started,
            "source_artifact_hashes": _source_hashes(source_path, specs),
            "rows": event_rows,
            "event_rows": event_rows,
            "action_receipt_rows": action_rows,
            "outcome_reveal_rows": outcome_rows,
            "advantage_rows": advantage_rows,
            "memory_candidate_rows": candidate_rows,
            "transaction_rows": transaction_rows,
            "signature_rows": signature_rows,
            "conflict_rows": conflict_rows,
            "rollback_rows": rollback_rows,
            "refresh_rows": refresh_rows,
            "learning_curve_rows": learning_rows,
            **metrics,
            "model_weights_changed": False,
            "flowbalance_training_reproduction": False,
            "flowbalance_memory_csl_complete_score": complete,
            "future_uplift_supported_score": supported if complete else 0,
            "gate_check_summary": gate_summary(checks),
            "verifier_is_oracle": False,
            "verdict_class": "positive"
            if complete and supported
            else ("null" if complete else "disqualified"),
            "honest_verdict": (
                "positive_supported_future_uplift_with_protected_retention"
                if complete and supported
                else (
                    "null_complete_without_supported_future_uplift"
                    if complete
                    else "disqualified_incomplete_or_inconsistent_receipts"
                )
            ),
            "status_receipt_rows": deepcopy(status_rows),
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact, expected_event_count=expected_event_count)
    if errors:
        validation = gate_row("artifact_validation", [], errors, False)
        checks.append(validation)
        artifact["preconditions_checked"] = {"all_passed": False, "checks": checks}
        artifact["gate_check_summary"] = gate_summary(checks)
        artifact["flowbalance_memory_csl_complete_score"] = 0
        artifact["future_uplift_supported_score"] = 0
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "disqualified_artifact_validation_failed"
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _status(
        status_rows,
        status_path,
        "terminal",
        artifact["verdict_class"],
        complete=artifact["flowbalance_memory_csl_complete_score"],
    )
    artifact["status_receipt_rows"] = deepcopy(status_rows)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    write_json_atomic(result_path, artifact)
    print(
        canonical_json(
            {
                "result_path": str(result_path),
                "verdict_class": artifact["verdict_class"],
                "honest_verdict": artifact["honest_verdict"],
            }
        ),
        flush=True,
    )
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any], *, expected_event_count: int = EXPECTED_EVENT_COUNT
) -> list[str]:
    """Cold-check schema, budgets, receipts, transactions, metrics, and verdict."""

    errors: list[str] = []
    principles = dict(artifact.get("field_principles") or {})
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"required_field_missing:{field}")
        if not str(principles.get(field, "")).strip():
            errors.append(f"field_principle_missing:{field}")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    verdict = str(artifact.get("verdict_class"))
    if verdict not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not str(artifact.get("honest_verdict", "")).startswith(verdict):
        errors.append("honest_verdict_class_mismatch")
    if artifact.get("model_weights_changed") is not False:
        errors.append("model_weights_changed")
    if artifact.get("flowbalance_training_reproduction") is not False:
        errors.append("flowbalance_training_reproduction")
    gate = dict(artifact.get("gate_check_summary") or {})
    if verdict == "blocked":
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_class_mismatch")
        if gate.get("passed") is not False or not gate.get("failed_check"):
            errors.append("blocked_gate_diagnostic_missing")
        if "expected_value" not in gate or "observed_value" not in gate:
            errors.append("blocked_gate_values_missing")
        if artifact.get("flowbalance_memory_csl_complete_score") != 0:
            errors.append("blocked_completion_nonzero")
        return list(dict.fromkeys(errors))
    event_rows = list(artifact.get("event_rows") or [])
    actions = list(artifact.get("action_receipt_rows") or [])
    outcomes = list(artifact.get("outcome_reveal_rows") or [])
    expected_rows = expected_event_count * len(ARMS)
    if artifact.get("rows") != event_rows:
        errors.append("rows_event_rows_mismatch")
    if len(event_rows) != expected_rows:
        errors.append("event_row_count_mismatch")
    if len(actions) != expected_rows:
        errors.append("action_receipt_count_mismatch")
    if len(outcomes) != expected_rows:
        errors.append("outcome_receipt_count_mismatch")
    for receipt in actions:
        payload = {key: value for key, value in receipt.items() if key != "receipt_hash"}
        if receipt.get("receipt_hash") != sha256_json(payload):
            errors.append("action_receipt_hash_mismatch")
    action_by_key = {(row.get("event_id"), row.get("arm")): row for row in actions}
    outcome_by_key = {(row.get("event_id"), row.get("arm")): row for row in outcomes}
    if any(
        (action := action_by_key.get((row.get("event_id"), row.get("arm")))) is None
        or row.get("response_hash") != sha256_text(str(row.get("raw_text", "")))
        or action.get("chronology_index") != row.get("chronology_index")
        or action.get("prompt_hash") != row.get("prompt_hash")
        or action.get("response_hash") != row.get("response_hash")
        or action.get("terminal_state") != row.get("terminal_state")
        or action.get("receipt_hash") != row.get("action_receipt_hash")
        for row in event_rows
    ):
        errors.append("event_action_link_mismatch")
    for receipt in outcomes:
        payload = {key: value for key, value in receipt.items() if key != "outcome_receipt_hash"}
        if receipt.get("outcome_receipt_hash") != sha256_json(payload):
            errors.append("outcome_receipt_hash_mismatch")
    if any(
        (outcome := outcome_by_key.get((row.get("event_id"), row.get("arm")))) is None
        or outcome.get("chronology_index") != row.get("chronology_index")
        or outcome.get("action_receipt_hash") != row.get("action_receipt_hash")
        or outcome.get("outcome_receipt_hash") != row.get("outcome_receipt_hash")
        or outcome.get("exact_success") != row.get("exact_success")
        or outcome.get("parse_success") != row.get("parse_success")
        or outcome.get("failed_constraint_classes") != row.get("failed_constraint_classes")
        for row in event_rows
    ):
        errors.append("event_outcome_link_mismatch")
    if any(row.get("prompt_bytes") != PROMPT_BUDGET_BYTES for row in event_rows):
        errors.append("prompt_budget_mismatch")
    if any(row.get("memory_bytes") != MEMORY_BUDGET_BYTES for row in event_rows):
        errors.append("memory_budget_mismatch")
    if any(row.get("max_tokens") != MAX_GENERATION_TOKENS for row in event_rows):
        errors.append("generation_budget_mismatch")
    if any(
        any(
            source_index >= int(row["chronology_index"])
            for source_index in [
                next(
                    (
                        int(signature["record"]["source_event_index"])
                        for signature in artifact.get("signature_rows", [])
                        if signature.get("record", {}).get("record_id") == record_id
                    ),
                    -1,
                )
            ]
        )
        for row in event_rows
        for record_id in row.get("memory_record_ids", [])
    ):
        errors.append("future_memory_visible")
    if any(
        row.get("terminal_state") not in {"committed", "rolled_back"}
        for row in artifact.get("transaction_rows", [])
    ):
        errors.append("transaction_not_terminal")
    if any(
        row.get("signature_valid") is not True or not signature_valid(row.get("record", {}))
        for row in artifact.get("signature_rows", [])
    ):
        errors.append("signature_invalid")
    if any(row.get("parent_restored") is not True for row in artifact.get("rollback_rows", [])):
        errors.append("rollback_parent_not_restored")
    if any(
        row.get("record_count_after", MAX_MEMORY_RECORDS + 1) > MAX_MEMORY_RECORDS
        or row.get("future_record_visible") is not False
        for row in artifact.get("refresh_rows", [])
    ):
        errors.append("refresh_bound_or_chronology_mismatch")
    expected_advantages = []
    for row in artifact.get("advantage_rows", []):
        balanced = outcome_by_key.get(
            (row.get("event_id"), "verifier_balanced_strategy_memory"), {}
        ).get("exact_success")
        baseline = outcome_by_key.get((row.get("event_id"), "no_memory"), {}).get("exact_success")
        update = flowbalance_update({}, balanced_success=balanced, baseline_success=baseline)
        expected_advantages.append(
            {
                "event_id": row.get("event_id"),
                "chronology_index": row.get("chronology_index"),
                "balanced_exact_success": balanced,
                "no_memory_exact_success": baseline,
                **{key: update[key] for key in ("advantage", "preference", "direction", "write")},
            }
        )
    if list(artifact.get("advantage_rows") or []) != expected_advantages:
        errors.append("advantage_rule_mismatch")
    expected_metrics = _reduce_metrics(event_rows, expected_advantages)
    metric_fields = tuple(expected_metrics)
    if any(artifact.get(field) != expected_metrics[field] for field in metric_fields):
        errors.append("reduced_metric_rows_mismatch")
    structural_complete = (
        len(event_rows) == expected_rows
        and len(actions) == expected_rows
        and len(outcomes) == expected_rows
        and bool(artifact.get("model_load_receipts"))
        and all(
            row.get("terminal_state") == "complete"
            for row in artifact.get("model_load_receipts", [])
        )
        and all(row.get("terminal_state") == "complete" for row in event_rows)
        and bool(artifact.get("transaction_rows"))
        and all(
            row.get("terminal_state") in {"committed", "rolled_back"}
            for row in artifact.get("transaction_rows", [])
        )
    )
    row_complete = structural_complete and gate.get("passed") is True
    if artifact.get("flowbalance_memory_csl_complete_score") != int(row_complete):
        errors.append("completion_score_mismatch")
    paired = list(artifact.get("paired_interval_rows") or [])
    if len(paired) != len(ARMS) - 1:
        errors.append("paired_interval_count_mismatch")
    future = list(artifact.get("future_success_rows") or [])
    if {row.get("arm") for row in future} != set(ARMS):
        errors.append("future_success_arm_roster_mismatch")
    expected_supported = _future_uplift_supported(expected_metrics) if row_complete else 0
    if artifact.get("future_uplift_supported_score") != expected_supported:
        errors.append("future_uplift_score_mismatch")
    supported = expected_supported == 1
    expected_verdict = (
        "positive" if row_complete and supported else ("null" if row_complete else "disqualified")
    )
    if verdict != expected_verdict:
        errors.append("verdict_class_mismatch")
    if gate.get("passed") is not bool(row_complete):
        errors.append("gate_completion_mismatch")
    return list(dict.fromkeys(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command boundary.
    """Run the dated experiment or validate one existing deliverable."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--transaction-root", type=Path, default=TRANSACTION_ROOT)
    parser.add_argument("--source-path", type=Path, default=SOURCE_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        artifact = json.loads(args.result_path.read_text(encoding="utf-8"))
    else:
        artifact = run_experiment(
            run_date=args.date,
            result_path=args.result_path,
            transaction_root=args.transaction_root,
            source_path=args.source_path,
        )
    errors = validate_artifact(artifact)
    print(
        canonical_json({"result_path": str(args.result_path), "validation_errors": errors}),
        flush=True,
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover - command wrapper owns direct execution.
    raise SystemExit(main())
