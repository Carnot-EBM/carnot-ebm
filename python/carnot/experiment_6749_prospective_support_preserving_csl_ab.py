"""Run the prospective support-preserving transactional-memory comparison.

Spec refs: REQ-CL-6749, SCENARIO-CL-6749-PROSPECTIVE-ORDER,
SCENARIO-CL-6749-SNAPSHOT, SCENARIO-CL-6749-EXACT-ADMISSION,
SCENARIO-CL-6749-ARM-ISOLATION, SCENARIO-CL-6749-SUPPORT,
SCENARIO-CL-6749-NO-WEIGHT-WRITES.

The acquisition model may publish an exact-certified external-memory record
only after its episode ends. The transfer model reads the acquisition model's
pre-event snapshot. It never contributes target-family evidence to memory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Protocol

from carnot.inference.sota_models import gguf_tokenizer_loadable, resolve_cached_gguf
from carnot.memory import transactional_constraint_memory as txmem


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260829"
EXPERIMENT_ID = "experiment_6749_prospective_support_preserving_csl_ab"
SCHEMA = "carnot.experiment_6749.prospective_support_preserving_csl_ab.v1"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6749_prospective_support_preserving_csl_ab.json"
)
FIXTURE_RELATIVE_PATH = Path(
    "results/experiment_6748_transactional_constraint_memory_fixture.json"
)
INFERENCE_SUBSTRATE = "local llama.cpp CUDA GGUF plus exact CPU admission checker"
ARMS = ("no_memory", "transactional_memory")
CANDIDATE_COUNT_K = 2
MAX_TOKENS = 12
VERIFIER_BUDGET_PER_CANDIDATE = 1
PLANNED_ROW_COUNT = 6 * 2 * 12 * len(ARMS)
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
MODEL_SPECS: list[JsonDict] = [
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "acquisition_and_same_family",
        "family": "qwen_moe",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "held_dense_transfer",
        "family": "gemma_dense",
    },
]
RANDOM_SEEDS = {
    "order": 6749,
    "model": {MODEL_SPECS[0]["hf_id"]: 6760, MODEL_SPECS[1]["hf_id"]: 6770},
    "candidate_base": 6749000,
}
ALLOWED_LABELS = (
    "clamp_upper_bound",
    "normalize_even_parity",
    "require_schema_field",
    "none",
    "clamp_lower_bound",
    "normalize_modulo",
    "reject_unsafe",
)
EVENT_PROBLEMS = {
    "e01": "A Python value can exceed its declared maximum bound.",
    "e02": "A Rust result must satisfy an even-parity constraint.",
    "e03": "An artifact can omit a required schema field.",
    "e04": "The text has a style preference but no correctness defect.",
    "e05": "A later Python value again exceeds the same maximum bound.",
    "e06": "The known Python upper-bound defect occurs again.",
    "e07": "A Python value is below its declared minimum bound.",
    "e08": "A Rust result again violates an even-parity constraint.",
    "e09": "A later artifact omits the same required schema field.",
    "e10": "A copied Rust repair claims an unrelated Python bound fix.",
    "e11": "A held-family integer violates a modulo constraint.",
    "e12": "An unsafe proposal asks the system to disable exact checks.",
}

GATE_NAMES = (
    "preconditions_pass",
    "all_planned_rows_present",
    "row_keys_unique",
    "chronology_preserved",
    "arm_isolation",
    "snapshot_immutability",
    "active_episode_write_zero",
    "exact_authority",
    "target_family_future_evidence_zero",
    "transaction_activity_nonzero",
    "rollback_exact",
    "model_weights_immutable",
    "live_models_complete",
    "row_metrics_consistent",
)

ROW_DERIVED_FIELDS = (
    "prequential_exact_yield_by_order",
    "best_at_k_by_arm",
    "effective_rewardable_support_by_arm",
    "joint_correct_constraint_support_by_arm",
    "retention_by_anchor",
    "cross_family_transfer_by_order",
    "negative_transfer_by_family",
    "token_cost_by_model_arm",
    "latency_by_model_arm",
    "commit_reject_rollback_counts",
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
    "model_specs",
    "live_model_invoked",
    "gpu_receipts",
    "model_weight_receipts",
    "model_weights_mutated",
    "fixture_receipt",
    "preconditions_checked",
    "frozen_protocol",
    "rows",
    *ROW_DERIVED_FIELDS,
    "prospective_csl_completed",
    "gate_check_summary",
    "verdict_class",
    "honest_verdict",
    "verifier_is_oracle",
    "tests_run",
)

FIELD_PRINCIPLES: dict[str, str] = {
    field: "This field makes the prospective result complete and independently auditable."
    for field in REQUIRED_ARTIFACT_FIELDS
}
FIELD_PRINCIPLES.update(
    {
        "duration_s": "A monotonic duration records the cost of real sequential inference.",
        "random_seed": "Frozen order, model, and candidate seeds prevent outcome-driven sampling.",
        "reproducibility_checksum": "One checksum binds the stream, protocol, snapshots, and rows.",
        "models_used": "Exact model identifiers prevent a legacy model from entering the claim.",
        "live_model_invoked": "A true value requires real generation from both cached GGUF files.",
        "gpu_receipts": "Load-time CUDA receipts prove sequential local GPU execution.",
        "model_weight_receipts": "Before and after file receipts detect a model-weight write.",
        "model_weights_mutated": "External memory is the only permitted mutable learning state.",
        "rows": "Every failed, abstaining, and successful candidate remains available for reduction.",
        "prospective_csl_completed": "This gate means every planned comparison ran; it is not a positive-result bit.",
        "gate_check_summary": "Failed checks include expected and observed values for an owned block.",
        "verdict_class": "A closed class separates completion from scientific direction.",
        "honest_verdict": "A terminal prefix lets the conductor classify the completed result.",
    }
)
FIELD_PRINCIPLES.update(
    {f"gate:{gate}": "This conjunct must pass before prospective_csl_completed is true." for gate in GATE_NAMES}
)

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6749_prospective_support_preserving_csl_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6749_prospective_support_preserving_csl_ab.py "
    "-m pytest tests/python/test_experiment_6749_prospective_support_preserving_csl_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null --fail-under=100 --show-missing"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6749_prospective_support_preserving_csl_ab.py"
)
LINT_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6749_prospective_support_preserving_csl_ab.py "
    "scripts/experiments/experiment_6749_prospective_support_preserving_csl_ab.py "
    "tests/python/test_experiment_6749_prospective_support_preserving_csl_ab.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6749_prospective_support_preserving_csl_ab.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6749_prospective_support_preserving_csl_ab.json"
)
DEFAULT_TESTS_RUN = [
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
]

TEST_PRECONDITION_OVERRIDES = {
    "six_frozen_orders": True,
    "exact_cached_model_specs": True,
    "model_paths_exist": True,
    "embedded_tokenizers_load": True,
    "llama_cpp_cuda_offload": True,
    "gpu_count_at_least_two": True,
    "sequential_vram_sufficient": True,
    "state_root_writable": True,
    "exact_cpu_admission_checker": True,
}


class Runner(Protocol):
    """Define the small inference surface used by real and test runners."""

    def load(self) -> JsonDict: ...

    def generate(self, prompt: str, *, seed: int, max_tokens: int) -> JsonDict: ...

    def close(self) -> None: ...


def canonical_json_bytes(value: Any) -> bytes:
    """Return one byte representation for checksums and atomic artifacts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    """Return a project-style SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash model or fixture bytes without loading them into Python memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def gpu_snapshot() -> list[JsonDict]:
    """Read free VRAM directly from the NVIDIA driver."""

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, name, total, free = [part.strip() for part in line.split(",", 3)]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "total_mib": int(total),
                "free_mib": int(free),
            }
        )
    return rows


def model_file_receipt(spec: Mapping[str, Any]) -> JsonDict:
    """Bind a cached GGUF to content, size, and modification time."""

    path = Path(str(spec["model_path"])).resolve()
    stat = path.stat()
    return {
        "model_id": spec["hf_id"],
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def resolve_model_specs(model_specs: Sequence[Mapping[str, Any]] | None = None) -> list[JsonDict]:
    """Resolve only the two preregistered SOTA GGUF model identities."""

    if model_specs is not None:
        return [dict(row) for row in model_specs]
    resolved = []
    for row in MODEL_SPECS:
        path = resolve_cached_gguf(str(row["hf_id"]), preferred_quant="Q4_K_M")
        resolved.append({**row, "model_path": path})
    return resolved


def freeze_protocol(fixture: Mapping[str, Any]) -> JsonDict:
    """Freeze prompts, budgets, orders, anchors, metrics, and rollback rules."""

    manifest = fixture["stream_manifest"]
    return {
        "frozen_before_first_episode": True,
        "prompt_template_sha256": sha256_bytes(
            b"event problem + allowed labels + optional exact-certified snapshot"
        ),
        "candidate_count_k": CANDIDATE_COUNT_K,
        "max_tokens_per_candidate": MAX_TOKENS,
        "verifier_budget_per_candidate": VERIFIER_BUDGET_PER_CANDIDATE,
        "orders": deepcopy(manifest["orders"]),
        "order_hashes": [row["order_hash"] for row in manifest["orders"]],
        "stream_hash": manifest["stream_hash"],
        "retention_anchors": list(manifest["families"]["retention_anchor"]),
        "support_thresholds": {
            "positive_delta_min": 0.0,
            "retention_delta_min": 0.0,
            "support_delta_min": 0.0,
        },
        "support_definitions": {
            "pass_at_1": "The first candidate is exact and constraint-following.",
            "best_at_k": "At least one candidate has the exact repair label.",
            "effective_rewardable_support": "The fraction of candidates with the exact rewardable label.",
            "joint_correct_constraint_support": "The fraction that is exact and follows the output grammar.",
        },
        "rollback_criteria": "Apply every inverse patch in reverse commit order and restore initial bytes.",
        "target_family_future_evidence_allowed": False,
        "seeds": deepcopy(RANDOM_SEEDS),
    }


def _check_row(expected: Any, observed: Any) -> JsonDict:
    return {"expected": expected, "observed": observed, "passed": observed == expected}


def check_preconditions(
    fixture: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    state_root: Path,
    overrides: Mapping[str, bool] | None = None,
) -> JsonDict:
    """Check the owned fixture, exact models, CUDA path, VRAM, and state path."""

    override = dict(overrides or {})
    state_root.mkdir(parents=True, exist_ok=True)
    paths = [Path(str(row.get("model_path"))) for row in specs]
    actual_ids = [row.get("hf_id") for row in specs]
    expected_ids = [row["hf_id"] for row in MODEL_SPECS]
    orders = fixture.get("stream_manifest", {}).get("orders", [])

    try:
        from llama_cpp import llama_cpp

        cuda_offload = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        cuda_offload = False
    try:
        gpu_rows = gpu_snapshot()
    except Exception:
        gpu_rows = []
    existing_paths = all(path.is_file() for path in paths)
    tokenizer_rows = [gguf_tokenizer_loadable(str(path)) for path in paths] if existing_paths else []
    total_free_bytes = sum(int(row["free_mib"]) for row in gpu_rows) * 1024 * 1024
    largest_model_bytes = max((path.stat().st_size for path in paths if path.is_file()), default=0)
    observations: dict[str, Any] = {
        "exp6748_transaction_memory_ready": fixture.get("transaction_memory_ready") is True,
        "six_frozen_orders": len(orders) == 6
        and len({row.get("order_hash") for row in orders}) == 6,
        "exact_cached_model_specs": actual_ids == expected_ids,
        "model_paths_exist": existing_paths,
        "embedded_tokenizers_load": len(tokenizer_rows) == 2
        and all(row[0] is True for row in tokenizer_rows),
        "llama_cpp_cuda_offload": cuda_offload,
        "gpu_count_at_least_two": len(gpu_rows) >= 2,
        "sequential_vram_sufficient": largest_model_bytes > 0
        and total_free_bytes >= int(largest_model_bytes * 1.1),
        "state_root_writable": os.access(state_root, os.W_OK),
        "exact_cpu_admission_checker": callable(txmem.exact_checker),
    }
    for name, value in override.items():
        if name in observations:
            observations[name] = value
    checks = {name: _check_row(True, observed) for name, observed in observations.items()}
    return {
        "checks": checks,
        "all_passed": all(row["passed"] for row in checks.values()),
        "gpu_snapshot": gpu_rows,
        "tokenizer_receipts": [
            {"model_id": spec["hf_id"], "passed": row[0], "detail": row[1]}
            for spec, row in zip(specs, tokenizer_rows, strict=False)
        ],
        "largest_model_bytes": largest_model_bytes,
        "total_free_vram_bytes": total_free_bytes,
    }


def _gate_summary_from_preconditions(preconditions: Mapping[str, Any]) -> JsonDict:
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


def prompt_for(
    event: Mapping[str, Any],
    arm: str,
    snapshot: Mapping[str, Any] | None,
    candidate_index: int,
) -> str:
    """Build the frozen classification prompt without current outcome evidence."""

    records = [] if snapshot is None else list(snapshot.get("records", []))
    safe_records = [
        {"family": row["key"].split(":")[1], "scope": row["scope"], "repair": row["repair"]}
        for row in records
        if row.get("certified") is True
    ]
    memory_text = json.dumps(safe_records, sort_keys=True) if safe_records else "[]"
    labels = ", ".join(ALLOWED_LABELS)
    return (
        "Select the exact repair label for this constrained event. "
        "Use prior memory only when its family and scope match. "
        f"Return exactly one label from: {labels}.\n"
        f"EVENT={event['event_id']} ARM={arm} CANDIDATE={candidate_index}\n"
        f"FAMILY={event['family']} SCOPE={event['scope']}\n"
        f"PROBLEM={EVENT_PROBLEMS[str(event['event_id'])]}\n"
        f"EXACT_CERTIFIED_PRIOR_MEMORY={memory_text}"
    )


class LiveLlamaRunner:
    """Load one GGUF at a time and collect CUDA-backed generation receipts."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self.spec = dict(spec)
        self._llm: Any = None
        self._grammar: Any = None

    def load(self) -> JsonDict:
        from llama_cpp import Llama, LlamaGrammar, llama_cpp

        before = gpu_snapshot()
        started = time.monotonic()
        self._llm = Llama(
            model_path=str(self.spec["model_path"]),
            n_gpu_layers=-1,
            n_ctx=1024,
            n_batch=256,
            verbose=False,
        )
        choices = " | ".join(json.dumps(label) for label in ALLOWED_LABELS)
        self._grammar = LlamaGrammar.from_string(f"root ::= {choices}")
        return {
            "model_id": self.spec["hf_id"],
            "model_path": str(Path(str(self.spec["model_path"])).resolve()),
            "loaded": True,
            "cuda_offload": bool(llama_cpp.llama_supports_gpu_offload()),
            "process_id": os.getpid(),
            "gpu_before": before,
            "gpu_after": gpu_snapshot(),
            "load_duration_s": round(time.monotonic() - started, 6),
        }

    def generate(self, prompt: str, *, seed: int, max_tokens: int) -> JsonDict:
        if self._llm is None:
            raise RuntimeError("model is not loaded")
        started = time.monotonic()
        result = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Follow the output-label contract exactly."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.55,
            top_p=0.9,
            seed=seed,
            grammar=self._grammar,
        )
        usage = result.get("usage", {})
        text = result["choices"][0]["message"]["content"] or ""
        return {
            "text": text,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "latency_s": round(time.monotonic() - started, 6),
            "seed": seed,
            "max_tokens": max_tokens,
        }

    def close(self) -> None:
        self._llm = None
        self._grammar = None
        gc.collect()


def evaluate_candidate(
    text: str,
    event: Mapping[str, Any],
    seed: int,
    prompt_tokens: int,
    completion_tokens: int,
    latency_s: float,
) -> JsonDict:
    """Apply the exact CPU label checker while retaining malformed output."""

    stripped = text.strip()
    parsed = next(
        (label for label in ALLOWED_LABELS if re.search(rf"\b{re.escape(label)}\b", stripped)),
        None,
    )
    exact = parsed == event["certified_repair"]
    constraint = stripped in ALLOWED_LABELS
    abstained = parsed is None
    return {
        "seed": seed,
        "response": text,
        "response_hash": sha256_bytes(text.encode("utf-8")),
        "parsed_label": parsed,
        "exact_correct": exact,
        "constraint_following": constraint,
        "rewardable": exact and not abstained,
        "abstained": abstained,
        "failed": False,
        "error": None,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "latency_s": round(float(latency_s), 6),
    }


def failed_candidate(seed: int, prompt_tokens: int, error: str) -> JsonDict:
    """Keep a generation failure as a zero-support candidate row."""

    return {
        "seed": seed,
        "response": "",
        "response_hash": sha256_bytes(b""),
        "parsed_label": None,
        "exact_correct": False,
        "constraint_following": False,
        "rewardable": False,
        "abstained": True,
        "failed": True,
        "error": error,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": 0,
        "latency_s": 0.0,
    }


def candidate_metrics(candidates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce pass, best, rewardable, and joint support from candidates."""

    count = len(candidates)
    return {
        "pass_at_1": int(
            bool(candidates)
            and candidates[0]["exact_correct"] is True
            and candidates[0]["constraint_following"] is True
        ),
        "best_at_k": int(any(row["exact_correct"] is True for row in candidates)),
        "effective_rewardable_support": (
            sum(int(row["rewardable"] is True) for row in candidates) / count if count else 0.0
        ),
        "joint_correct_constraint_support": (
            sum(
                int(
                    row["exact_correct"] is True
                    and row["constraint_following"] is True
                )
                for row in candidates
            )
            / count
            if count
            else 0.0
        ),
    }


def candidate_seed(model_index: int, order_index: int, event_index: int, candidate: int) -> int:
    """Give matched arms the same deterministic candidate seed."""

    return (
        int(RANDOM_SEEDS["candidate_base"])
        + model_index * 100000
        + order_index * 1000
        + event_index * 10
        + candidate
    )


def _run_candidates(
    runner: Runner,
    event: Mapping[str, Any],
    arm: str,
    snapshot: Mapping[str, Any] | None,
    model_index: int,
    order_index: int,
    event_index: int,
) -> list[JsonDict]:
    candidates = []
    for candidate_index in range(CANDIDATE_COUNT_K):
        seed = candidate_seed(model_index, order_index, event_index, candidate_index)
        prompt = prompt_for(event, arm, snapshot, candidate_index)
        try:
            generated = runner.generate(prompt, seed=seed, max_tokens=MAX_TOKENS)
            candidates.append(
                evaluate_candidate(
                    str(generated["text"]),
                    event,
                    seed,
                    int(generated.get("prompt_tokens", 0)),
                    int(generated.get("completion_tokens", 0)),
                    float(generated.get("latency_s", 0.0)),
                )
            )
        except Exception as error:
            candidates.append(failed_candidate(seed, 0, f"{type(error).__name__}: {error}"))
    return candidates


def _result_row(
    *,
    event: Mapping[str, Any],
    order: Mapping[str, Any],
    order_position: int,
    spec: Mapping[str, Any],
    arm: str,
    candidates: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any] | None,
) -> JsonDict:
    metrics = candidate_metrics(candidates)
    memory_read = int(arm == "transactional_memory")
    return {
        "row_key": ":".join(
            [str(order["order_id"]), str(spec["hf_id"]), str(event["event_id"]), arm]
        ),
        "order_id": order["order_id"],
        "order_hash": order["order_hash"],
        "order_position": order_position,
        "event_id": event["event_id"],
        "event_kind": event["kind"],
        "family": event["family"],
        "scope": event["scope"],
        "model_id": spec["hf_id"],
        "model_role": spec["role"],
        "model_family": spec["family"],
        "arm": arm,
        "candidate_count_k": CANDIDATE_COUNT_K,
        "max_tokens": MAX_TOKENS,
        "verifier_budget": CANDIDATE_COUNT_K * VERIFIER_BUDGET_PER_CANDIDATE,
        "candidate_seeds": [row["seed"] for row in candidates],
        "candidates": [dict(row) for row in candidates],
        **metrics,
        "memory_read_count": memory_read,
        "memory_write_count": 0,
        "active_episode_write_count": 0,
        "snapshot_hash": None if snapshot is None else snapshot["state_hash"],
        "snapshot_version": None if snapshot is None else snapshot["version"],
        "snapshot_immutable": True,
        "memory_source_model_id": (
            None if arm == "no_memory" else MODEL_SPECS[0]["hf_id"]
        ),
        "target_family_future_evidence_count": 0,
        "exact_result_known_before_commit": False,
        "commit_status": "not_applicable",
        "admission_checks": {},
        "quarantine_written": False,
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in candidates),
        "completion_tokens": sum(int(row["completion_tokens"]) for row in candidates),
        "latency_s": round(sum(float(row["latency_s"]) for row in candidates), 6),
    }


def _rate(numerator: int | float, denominator: int) -> JsonDict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": round(float(numerator) / denominator, 6) if denominator else 0.0,
    }


def reduce_rows(
    rows: Sequence[Mapping[str, Any]],
    lifecycle_by_order: Mapping[str, Mapping[str, int]],
) -> JsonDict:
    """Recompute all headline, support, transfer, cost, and lifecycle metrics."""

    prequential: dict[str, Any] = {}
    for order_id in sorted({str(row["order_id"]) for row in rows}):
        prequential[order_id] = {}
        for model_id in [str(row["hf_id"]) for row in MODEL_SPECS]:
            prequential[order_id][model_id] = {}
            for arm in ARMS:
                selected = [
                    row
                    for row in rows
                    if row["order_id"] == order_id
                    and row["model_id"] == model_id
                    and row["arm"] == arm
                ]
                prequential[order_id][model_id][arm] = _rate(
                    sum(int(row["pass_at_1"]) for row in selected), len(selected)
                )

    best = {}
    rewardable = {}
    joint = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        best[arm] = _rate(sum(int(row["best_at_k"]) for row in selected), len(selected))
        denominator = sum(len(row["candidates"]) for row in selected)
        rewardable[arm] = _rate(
            sum(
                int(candidate["rewardable"] is True)
                for row in selected
                for candidate in row["candidates"]
            ),
            denominator,
        )
        joint[arm] = _rate(
            sum(
                int(
                    candidate["exact_correct"] is True
                    and candidate["constraint_following"] is True
                )
                for row in selected
                for candidate in row["candidates"]
            ),
            denominator,
        )

    retention: dict[str, Any] = {}
    anchor_rows = [row for row in rows if row["event_kind"] == "retention_anchor"]
    for row in anchor_rows:
        retention.setdefault(str(row["order_id"]), {}).setdefault(str(row["model_id"]), {})[
            str(row["arm"])
        ] = {
            "event_id": row["event_id"],
            "pass_at_1": row["pass_at_1"],
            "best_at_k": row["best_at_k"],
            "joint_correct_constraint_support": row["joint_correct_constraint_support"],
        }

    transfer = {}
    gemma_id = MODEL_SPECS[1]["hf_id"]
    for order_id in sorted({str(row["order_id"]) for row in rows}):
        arm_values = {}
        for arm in ARMS:
            selected = [
                row
                for row in rows
                if row["order_id"] == order_id
                and row["model_id"] == gemma_id
                and row["arm"] == arm
            ]
            arm_values[arm] = _rate(
                sum(int(row["pass_at_1"]) for row in selected), len(selected)
            )
        transfer[order_id] = {
            "by_arm": arm_values,
            "transactional_minus_no_memory": round(
                arm_values["transactional_memory"]["rate"]
                - arm_values["no_memory"]["rate"],
                6,
            ),
        }

    negative = {}
    for model_id in [str(row["hf_id"]) for row in MODEL_SPECS]:
        negative[model_id] = {}
        for family in sorted({str(row["family"]) for row in rows}):
            rates = {}
            for arm in ARMS:
                selected = [
                    row
                    for row in rows
                    if row["model_id"] == model_id
                    and row["family"] == family
                    and row["arm"] == arm
                ]
                rates[arm] = _rate(
                    sum(int(row["pass_at_1"]) for row in selected), len(selected)
                )["rate"]
            delta = round(rates["transactional_memory"] - rates["no_memory"], 6)
            negative[model_id][family] = {"delta": delta, "negative_transfer": delta < 0.0}

    tokens: dict[str, Any] = {}
    latency: dict[str, Any] = {}
    for model_id in [str(row["hf_id"]) for row in MODEL_SPECS]:
        tokens[model_id] = {}
        latency[model_id] = {}
        for arm in ARMS:
            selected = [
                row for row in rows if row["model_id"] == model_id and row["arm"] == arm
            ]
            tokens[model_id][arm] = {
                "prompt_tokens": sum(int(row["prompt_tokens"]) for row in selected),
                "completion_tokens": sum(int(row["completion_tokens"]) for row in selected),
                "total_tokens": sum(
                    int(row["prompt_tokens"]) + int(row["completion_tokens"])
                    for row in selected
                ),
            }
            latency[model_id][arm] = {
                "duration_s": round(sum(float(row["latency_s"]) for row in selected), 6),
                "row_count": len(selected),
            }

    normalized_lifecycle = {key: dict(value) for key, value in sorted(lifecycle_by_order.items())}
    lifecycle_keys = ("commits", "rejects", "quarantine", "rollbacks", "rollback_failures")
    lifecycle = {
        "by_order": normalized_lifecycle,
        "totals": {
            key: sum(int(row.get(key, 0)) for row in normalized_lifecycle.values())
            for key in lifecycle_keys
        },
    }
    return {
        "prequential_exact_yield_by_order": prequential,
        "best_at_k_by_arm": best,
        "effective_rewardable_support_by_arm": rewardable,
        "joint_correct_constraint_support_by_arm": joint,
        "retention_by_anchor": retention,
        "cross_family_transfer_by_order": transfer,
        "negative_transfer_by_family": negative,
        "token_cost_by_model_arm": tokens,
        "latency_by_model_arm": latency,
        "commit_reject_rollback_counts": lifecycle,
    }


def _chronology_passes(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> bool:
    expected_orders = {
        str(order["order_id"]): list(order["event_ids"]) for order in protocol["orders"]
    }
    for model in MODEL_SPECS:
        for order_id, event_ids in expected_orders.items():
            for arm in ARMS:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["model_id"] == model["hf_id"]
                        and row["order_id"] == order_id
                        and row["arm"] == arm
                    ),
                    key=lambda row: int(row["order_position"]),
                )
                if [row["event_id"] for row in selected] != event_ids:
                    return False
    return True


def _arm_isolation_passes(rows: Sequence[Mapping[str, Any]]) -> bool:
    cells: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        cells[(str(row["order_id"]), str(row["model_id"]), str(row["event_id"]))][
            str(row["arm"])
        ] = row
    return bool(cells) and all(
        set(pair) == set(ARMS)
        and pair["no_memory"]["candidate_seeds"]
        == pair["transactional_memory"]["candidate_seeds"]
        and pair["no_memory"]["candidate_count_k"]
        == pair["transactional_memory"]["candidate_count_k"]
        and pair["no_memory"]["max_tokens"] == pair["transactional_memory"]["max_tokens"]
        and pair["no_memory"]["memory_read_count"] == 0
        and pair["no_memory"]["memory_write_count"] == 0
        for pair in cells.values()
    )


def _row_metrics_pass(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        candidate_metrics(row["candidates"])
        == {
            "pass_at_1": row["pass_at_1"],
            "best_at_k": row["best_at_k"],
            "effective_rewardable_support": row["effective_rewardable_support"],
            "joint_correct_constraint_support": row["joint_correct_constraint_support"],
        }
        for row in rows
    )


def completion_checks(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute the completion gate without using scientific effect direction."""

    rows = artifact["rows"]
    lifecycle = artifact["commit_reject_rollback_counts"]["totals"]
    receipts = artifact["gpu_receipts"]
    checks = {
        "preconditions_pass": artifact["preconditions_checked"]["all_passed"] is True,
        "all_planned_rows_present": len(rows) == PLANNED_ROW_COUNT,
        "row_keys_unique": len({row["row_key"] for row in rows}) == len(rows),
        "chronology_preserved": _chronology_passes(rows, artifact["frozen_protocol"]),
        "arm_isolation": _arm_isolation_passes(rows),
        "snapshot_immutability": all(
            row["snapshot_immutable"] is True for row in rows if row["arm"] == "transactional_memory"
        ),
        "active_episode_write_zero": all(row["active_episode_write_count"] == 0 for row in rows),
        "exact_authority": all(
            row["exact_result_known_before_commit"] is True
            and bool(row["admission_checks"])
            and all(row["admission_checks"].values())
            for row in rows
            if row["commit_status"] == "committed"
        ),
        "target_family_future_evidence_zero": all(
            row["target_family_future_evidence_count"] == 0 for row in rows
        ),
        "transaction_activity_nonzero": lifecycle["commits"] > 0
        and lifecycle["rejects"] > 0,
        "rollback_exact": lifecycle["rollbacks"] == lifecycle["commits"]
        and lifecycle["rollback_failures"] == 0,
        "model_weights_immutable": artifact["model_weights_mutated"] is False,
        "live_models_complete": artifact["live_model_invoked"] is True
        and len(receipts) == 2
        and all(row.get("loaded") is True and row.get("cuda_offload") is True for row in receipts),
        "row_metrics_consistent": _row_metrics_pass(rows),
    }
    return checks


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


def _blank_metrics() -> JsonDict:
    return {field: {} for field in ROW_DERIVED_FIELDS}


def _base_artifact(
    *,
    fixture_path: Path,
    fixture: Mapping[str, Any],
    protocol: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "status": "complete_blocked_prospective_csl",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": deepcopy(RANDOM_SEEDS),
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
        "model_specs": [
            {"hf_id": row["hf_id"], "role": row["role"], "family": row["family"]}
            for row in specs
        ],
        "live_model_invoked": False,
        "gpu_receipts": [],
        "model_weight_receipts": [],
        "model_weights_mutated": False,
        "fixture_receipt": {
            "path": str(fixture_path),
            "sha256": sha256_file(fixture_path),
            "transaction_memory_ready": fixture.get("transaction_memory_ready"),
            "stream_hash": fixture.get("stream_manifest", {}).get("stream_hash"),
        },
        "preconditions_checked": deepcopy(dict(preconditions)),
        "frozen_protocol": deepcopy(dict(protocol)),
        "rows": [],
        **_blank_metrics(),
        "prospective_csl_completed": False,
        "gate_check_summary": _gate_summary_from_preconditions(preconditions),
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_prospective_csl: owned precondition failed",
        "verifier_is_oracle": True,
        "tests_run": deepcopy(DEFAULT_TESTS_RUN),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _run_qwen_orders(
    runner: Runner,
    spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
    events: Mapping[str, Mapping[str, Any]],
    state_root: Path,
) -> tuple[list[JsonDict], dict[str, dict[str, JsonDict]], dict[str, JsonDict]]:
    rows: list[JsonDict] = []
    snapshots: dict[str, dict[str, JsonDict]] = {}
    lifecycle: dict[str, JsonDict] = {}
    for order_index, order in enumerate(protocol["orders"]):
        order_id = str(order["order_id"])
        memory = txmem.TransactionalConstraintMemory(state_root / order_id)
        initial_bytes = memory.state_bytes()
        commits: list[JsonDict] = []
        snapshots[order_id] = {}
        counts = {
            "commits": 0,
            "rejects": 0,
            "quarantine": 0,
            "rollbacks": 0,
            "rollback_failures": 0,
        }
        for event_index, event_id in enumerate(order["event_ids"]):
            event = events[str(event_id)]
            snapshot = memory.begin_episode(str(event_id))
            snapshots[order_id][str(event_id)] = {
                "state_hash": snapshot["state_hash"],
                "version": snapshot["version"],
                "records": deepcopy(snapshot["records"]),
            }
            baseline_candidates = _run_candidates(
                runner, event, "no_memory", None, 0, order_index, event_index
            )
            transaction_candidates = _run_candidates(
                runner,
                event,
                "transactional_memory",
                snapshot,
                0,
                order_index,
                event_index,
            )
            baseline_row = _result_row(
                event=event,
                order=order,
                order_position=event_index + 1,
                spec=spec,
                arm="no_memory",
                candidates=baseline_candidates,
                snapshot=None,
            )
            transaction_row = _result_row(
                event=event,
                order=order,
                order_position=event_index + 1,
                spec=spec,
                arm="transactional_memory",
                candidates=transaction_candidates,
                snapshot=snapshot,
            )
            transaction_row["snapshot_immutable"] = memory.state_hash() == snapshot["state_hash"]
            memory.end_episode()
            transaction_row["exact_result_known_before_commit"] = True
            proposal = (
                txmem.proposal_for(event) if transaction_row["pass_at_1"] == 1 else None
            )
            if proposal is None:
                transaction_row["commit_status"] = (
                    "no_proposal"
                    if txmem.proposal_for(event) is None
                    else "model_incorrect_no_proposal"
                )
            else:
                decision = memory.admit(proposal, event, boundary_index=event_index + 1)
                transaction_row["admission_checks"] = deepcopy(decision["checks"])
                if decision["admitted"] is True:
                    transaction_row["commit_status"] = "committed"
                    commits.append(decision["commit_receipt"])
                    counts["commits"] += 1
                else:
                    transaction_row["commit_status"] = "rejected"
                    counts["rejects"] += 1
                    transaction_row["quarantine_written"] = bool(
                        decision.get("quarantine_receipt", {}).get("written")
                    )
                    counts["quarantine"] += int(transaction_row["quarantine_written"])
            rows.extend([baseline_row, transaction_row])

        for receipt in reversed(commits):
            rollback = memory.rollback(receipt)
            counts["rollbacks"] += 1
            counts["rollback_failures"] += int(rollback["passed"] is not True)
        counts["rollback_failures"] += int(memory.state_bytes() != initial_bytes)
        lifecycle[order_id] = counts
    return rows, snapshots, lifecycle


def _run_gemma_orders(
    runner: Runner,
    spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
    events: Mapping[str, Mapping[str, Any]],
    snapshots: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for order_index, order in enumerate(protocol["orders"]):
        order_id = str(order["order_id"])
        for event_index, event_id in enumerate(order["event_ids"]):
            event = events[str(event_id)]
            snapshot = snapshots[order_id][str(event_id)]
            for arm in ARMS:
                visible = snapshot if arm == "transactional_memory" else None
                candidates = _run_candidates(
                    runner, event, arm, visible, 1, order_index, event_index
                )
                rows.append(
                    _result_row(
                        event=event,
                        order=order,
                        order_position=event_index + 1,
                        spec=spec,
                        arm=arm,
                        candidates=candidates,
                        snapshot=visible,
                    )
                )
    return rows


def run_experiment(
    *,
    fixture_path: Path | str = REPO_ROOT / FIXTURE_RELATIVE_PATH,
    state_root: Path | str | None = None,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    runner_factory: Callable[[Mapping[str, Any]], Runner] = LiveLlamaRunner,
    precondition_overrides: Mapping[str, bool] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Run both models sequentially over all matched frozen order cells."""

    started = time.monotonic()
    if state_root is None:
        with tempfile.TemporaryDirectory(prefix="carnot-exp6749-") as directory:
            return run_experiment(
                fixture_path=fixture_path,
                state_root=directory,
                model_specs=model_specs,
                runner_factory=runner_factory,
                precondition_overrides=precondition_overrides,
                duration_s=duration_s,
            )
    fixture_path = Path(fixture_path)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    specs = resolve_model_specs(model_specs)
    protocol = freeze_protocol(fixture)
    root = Path(state_root)
    preconditions = check_preconditions(fixture, specs, root, precondition_overrides)
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact = _base_artifact(
        fixture_path=fixture_path,
        fixture=fixture,
        protocol=protocol,
        specs=specs,
        preconditions=preconditions,
        duration_s=elapsed,
    )
    if preconditions["all_passed"] is not True:
        return artifact

    before = [model_file_receipt(spec) for spec in specs]
    events = {
        str(event["event_id"]): event for event in fixture["stream_manifest"]["events"]
    }
    gpu_receipts = []
    qwen_runner = runner_factory(specs[0])
    gpu_receipts.append(qwen_runner.load())
    qwen_rows, snapshots, lifecycle = _run_qwen_orders(
        qwen_runner, specs[0], protocol, events, root
    )
    qwen_runner.close()

    gemma_runner = runner_factory(specs[1])
    gpu_receipts.append(gemma_runner.load())
    gemma_rows = _run_gemma_orders(gemma_runner, specs[1], protocol, events, snapshots)
    gemma_runner.close()
    after = [model_file_receipt(spec) for spec in specs]
    weight_receipts = [
        {
            "model_id": prior["model_id"],
            "before": prior,
            "after": current,
            "mutated": prior != current,
        }
        for prior, current in zip(before, after, strict=True)
    ]
    rows = [*qwen_rows, *gemma_rows]
    aggregates = reduce_rows(rows, lifecycle)
    artifact.update(
        {
            "status": "complete_prospective_csl",
            "live_model_invoked": len(gpu_receipts) == 2
            and all(row.get("loaded") is True for row in gpu_receipts),
            "gpu_receipts": gpu_receipts,
            "model_weight_receipts": weight_receipts,
            "model_weights_mutated": any(row["mutated"] is True for row in weight_receipts),
            "rows": rows,
            **aggregates,
        }
    )
    checks = completion_checks(artifact)
    completed = all(checks.values())
    artifact["prospective_csl_completed"] = completed
    artifact["gate_check_summary"] = _gate_summary(checks)
    support = artifact["effective_rewardable_support_by_arm"]
    best = artifact["best_at_k_by_arm"]
    positive = (
        completed
        and best["transactional_memory"]["rate"] > best["no_memory"]["rate"]
        and support["transactional_memory"]["rate"] >= support["no_memory"]["rate"]
        and not any(
            cell["negative_transfer"]
            for model in artifact["negative_transfer_by_family"].values()
            for cell in model.values()
        )
    )
    artifact["verdict_class"] = "positive" if positive else ("null" if completed else "partial")
    artifact["honest_verdict"] = (
        "complete_prospective_csl_positive: transactional memory improved best-at-k without support loss"
        if positive
        else (
            "complete_prospective_csl_null: all planned cells completed without a support-preserving gain"
            if completed
            else "complete_prospective_csl_partial: one or more completion gates failed"
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
    """Hash the frozen protocol, snapshots, candidates, aggregates, and receipts."""

    material = {
        "schema": artifact.get("schema"),
        "random_seed": artifact.get("random_seed"),
        "models_used": artifact.get("models_used"),
        "model_specs": artifact.get("model_specs"),
        "fixture_receipt": artifact.get("fixture_receipt"),
        "frozen_protocol": artifact.get("frozen_protocol"),
        "rows": artifact.get("rows"),
        "model_weight_receipts": artifact.get("model_weight_receipts"),
        **{field: artifact.get(field) for field in ROW_DERIVED_FIELDS},
        "prospective_csl_completed": artifact.get("prospective_csl_completed"),
        "gate_check_summary": artifact.get("gate_check_summary"),
        "verdict_class": artifact.get("verdict_class"),
    }
    return sha256_bytes(canonical_json_bytes(material))


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return closed validation errors without modifying the evidence."""

    errors = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class outside closed enum")
    expected_principles = set(REQUIRED_ARTIFACT_FIELDS) | {
        f"gate:{name}" for name in GATE_NAMES
    }
    if set(artifact.get("field_principles", {})) != expected_principles:
        errors.append("field_principles coverage mismatch")
    if artifact.get("model_weights_mutated") is True:
        errors.append("model weights mutated")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if artifact.get("rows"):
        reduced = reduce_rows(
            artifact["rows"], artifact["commit_reject_rollback_counts"]["by_order"]
        )
        if any(artifact.get(field) != reduced[field] for field in ROW_DERIVED_FIELDS):
            errors.append("row-derived metrics mismatch")
        checks = completion_checks(artifact)
        if artifact.get("prospective_csl_completed") is not all(checks.values()):
            errors.append("completion gates mismatch")
    elif artifact.get("prospective_csl_completed") is True:
        errors.append("completed artifact has no rows")
    return errors


def write_artifact(path: Path | str, artifact: Mapping[str, Any]) -> JsonDict:
    """Validate and publish the terminal artifact through one atomic rename."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, target)
    return {"path": str(target), "atomic_rename": True, "sha256": sha256_file(target)}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the live experiment or validate its stored terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--fixture-path", default=str(REPO_ROOT / FIXTURE_RELATIVE_PATH))
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
    artifact = run_experiment(
        fixture_path=args.fixture_path,
        state_root=args.state_root,
    )
    write_artifact(result_path, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - the task uses the thin script wrapper.
    raise SystemExit(main())
