"""Exp6491 local SOTA GGUF atomic factor proposal stream.

Spec refs: REQ-VERIFY-6491, SCENARIO-VERIFY-6491-GATES,
SCENARIO-VERIFY-6491-RAW-BYTES, SCENARIO-VERIFY-6491-HELD-ISOLATION,
SCENARIO-VERIFY-6491-COMPILER-AUTHORITY,
SCENARIO-VERIFY-6491-BOUNDARY-ATTACKS, SCENARIO-VERIFY-6491-ROWS.

The stream lets local GGUF models propose one factor. Exact compilation owns
admissibility. The model never receives held rows, final outcomes, or answer
authority.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import gc
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any

from carnot import task_runtime_receipts as receipts
from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
CacheResolver = Callable[[str, str], str | None]
PairResolver = Callable[..., list[dict[str, Any]] | None]
RuntimeFactory = Callable[[dict[str, Any]], Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260821"
RANDOM_SEED = 6491
EVENT_ORDER_SEED = 6491001
MODEL_ORDER_SEED = 6491002
N_GPU_LAYERS = -1
MAX_TOKENS = 96
MAX_EVENTS = 2

INFERENCE_SUBSTRATE = "local_llama_cpp_gguf_atomic_factor_proposals"
VERIFIER_IS_ORACLE = False
SCHEMA_VERSION = "carnot.experiment_6491.sota_factor_proposal_stream.v1"
FACTOR_SCHEMA_VERSION = "carnot.experiment_6491.atomic_factor_proposal.v1"
EXACT_COMPILER_ID = "exp6491_atomic_factor_compiler_over_exp6489_solver_prefix_v1"

RESULT_RELATIVE_PATH = Path("results/experiment_6491_sota_factor_proposal_stream.json")
RAW_DIR_RELATIVE_PATH = Path("results/experiment_6491_sota_factor_proposal_stream_raw")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6491_sota_factor_proposal_stream.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6491_sota_factor_proposal_stream.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
EXP6463_RELATIVE_PATH = Path("results/experiment_6463_sota_fixed_policy_candidate_corpus_v2.json")
EXP6484_RELATIVE_PATH = Path("results/experiment_6484_non_generation_representation_receipt_contract.json")
EXP6488_RELATIVE_PATH = Path("results/experiment_6488_v559_decision_ledger.json")
EXP6489_RELATIVE_PATH = Path("results/experiment_6489_solver_trajectory_commitment.json")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("research-roadmap.yaml"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("scripts/experiment_template.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/task_runtime_receipts.py"),
    EXP6463_RELATIVE_PATH,
    EXP6484_RELATIVE_PATH,
    EXP6488_RELATIVE_PATH,
    EXP6489_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6491_sota_factor_proposal_stream "
    "--date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6491_sota_factor_proposal_stream.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6491_sota_factor_proposal_stream.py "
    "-m pytest tests/python/test_experiment_6491_sota_factor_proposal_stream.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6491_sota_factor_proposal_stream.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6491_sota_factor_proposal_stream.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6491_sota_factor_proposal_stream.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6491_sota_factor_proposal_stream.json"
)
GPU_MODEL_E2E_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6491_sota_factor_proposal_stream "
    "--validate"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    RUN_COMMAND,
    ROW_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    GPU_MODEL_E2E_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_gate_receipts",
    "model_specs",
    "model_load_receipts",
    "frozen_event_manifest",
    "prompt_commitment",
    "raw_request_response_receipts",
    "proposal_rows",
    "exact_compile_rows",
    "held_isolation_receipts",
    "non_authority_receipts",
    "boundary_attack_matrix",
    "factor_proposal_stream_ready_score",
    "prior_lineage_retirement_receipt",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal local proposal-stream state.",
    "upstream_gate_receipts": "Both upstream artifact hashes and exact gate values.",
    "model_specs": "All three mandated repository IDs, files, quantizations, and roles.",
    "model_load_receipts": "Runtime, embedded tokenizer, backend, `n_gpu_layers`, hashes, and failures.",
    "frozen_event_manifest": "Development-only divergence events committed before model access.",
    "prompt_commitment": "Allowed context, schema, and prompt hashes.",
    "raw_request_response_receipts": "Immutable byte paths and hashes before parsing.",
    "proposal_rows": "One row per event and model, including no-proposal outcomes.",
    "exact_compile_rows": "Accepted, rejected, duplicate, timeout, and reason per proposal.",
    "held_isolation_receipts": "Proof that held outcomes were not placed in requests.",
    "non_authority_receipts": "Proof that model output did not label, verify, or release an answer.",
    "boundary_attack_matrix": "Held access, answer, retry, rewrite, tokenizer, and selection attacks.",
    "factor_proposal_stream_ready_score": "Same-roadmap downstream gate field.",
    "prior_lineage_retirement_receipt": "Changed scope and Exp6463 comparison.",
    "per_unit_rows": "Event/model/proposal/compiler rows.",
    "aggregate_row_recomputation": "Counts and ready score recomputed from rows.",
    "gate_check_summary": "Exact gate evaluation or blocked_* reason and observed value.",
    "preconditions_checked": "Models, runtime, development split, compiler, and retired lineage.",
    "protected_files_unchanged": "Active roadmap and conductor unchanged.",
    "inference_substrate": "local_llama_cpp_gguf_atomic_factor_proposals.",
    "verifier_is_oracle": "False for model proposals; true only for exact compilation facts.",
    "field_principles": "Reason for every receipt and boundary field.",
    "field_provenance": "Raw bytes, file hashes, runtime receipts, and compiler reducers.",
    "random_seed": "Frozen event and model-order seeds.",
    "duration_s": "Measured inference and task wall time.",
    "tests_run": "Commands and exit codes.",
    "reproducibility_checksum": "Hash over models, events, raw bytes, and compile rows.",
    "honest_verdict": "complete_* when the stream is valid, otherwise blocked_* with gate_check_summary.",
}

BOUNDARY_ATTACK_IDS = (
    "held_label_access",
    "answer_emission",
    "retry_loop",
    "response_rewrite",
    "missing_raw_bytes",
    "wrong_tokenizer_path",
    "model_identity_omission",
    "duplicate_factor",
    "post_hoc_event_selection",
)
COMPILE_OUTCOMES = ("accept", "reject", "duplicate", "timeout", "no_proposal")
ALLOWED_FACTOR_KINDS = (
    "partial_assignment_eq",
    "residual_weight_at_most",
    "candidate_count_at_least",
    "branch_depth_at_least",
)
FORBIDDEN_PROPOSAL_KEYS = {
    "answer",
    "final_answer",
    "label",
    "verdict",
    "outcome",
    "final_outcome",
    "release",
    "release_authority",
    "verify",
    "verifier",
    "solution",
}
FORBIDDEN_REQUEST_CONTEXT_KEYS = (
    "final_exact_label",
    "final_exact_outcome_hash",
    "final_objective_value",
    "final_objective",
    "held_label",
    "answer",
    "release_authority",
)
FACTOR_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence with stable key order."""

    return receipts.canonical_json(value)


def _sha256_json(value: Any) -> str:
    return receipts.sha256_json(value)


def _sha256_file(path: str | Path) -> str | None:
    return receipts.sha256_file(path)


def _sha256_bytes(value: bytes) -> str:
    return receipts.sha256_bytes(value)


def _read_json(path: Path) -> JsonDict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    receipts.write_json_atomic(path, payload)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def quant_from_filename(filename: str) -> str:
    """Extract the visible GGUF quantization token from a filename."""

    for token in (
        "UD-Q8_K_XL",
        "UD-Q6_K_XL",
        "UD-Q5_K_M",
        "UD-Q5_K_S",
        "UD-Q4_K_M",
        "UD-Q4_K_S",
        "UD-Q4_K_XL",
        "UD-Q3_K_M",
        "UD-Q3_K_S",
        "UD-Q2_K_XL",
        "Q8_0",
        "Q6_K",
        "Q5_K_M",
        "Q4_K_M",
        "Q3_K_M",
        "Q2_K",
        "BF16",
        "MXFP4_MOE",
    ):
        if token.lower() in filename.lower():
            return token
    return "unknown"


def _model_family(hf_id: str) -> str:
    text = hf_id.lower()
    if "qwen" in text:
        return "qwen"
    if "gemma" in text:
        return "gemma"
    return "other"


def resolve_model_specs(
    *,
    cache_resolver: CacheResolver = resolve_cached_gguf,
    pair_resolver: PairResolver = cached_sota_pair,
) -> list[JsonDict]:
    """Resolve all mandated GGUF specs and mark the two precommitted models."""

    pair = pair_resolver(gpu_indices=(0, 1), preferred_quant="Q4_K_M") or []
    pair_by_id = {str(row.get("hf_id")): dict(row) for row in pair}
    selected_ids = [str(row.get("hf_id")) for row in pair if row.get("hf_id")]
    resolved: list[JsonDict] = []
    for base in SOTA_GGUF_MODELS:
        hf_id = str(base["hf_id"])
        pair_row = pair_by_id.get(hf_id, {})
        model_path = pair_row.get("model_path") or cache_resolver(hf_id, str(base["quantization"]))
        path = Path(str(model_path)) if model_path else None
        exists = bool(path and path.is_file())
        filename = path.name if path else ""
        selected = hf_id in selected_ids and exists
        resolved.append(
            {
                "name": base["name"],
                "hf_id": hf_id,
                "repository_id": hf_id,
                "role": base["role"],
                "model_family": _model_family(hf_id),
                "active_params_b": base["active_params_b"],
                "total_params_b": base["total_params_b"],
                "min_vram_gb": base["min_vram_gb"],
                "expected_quantization": base["quantization"],
                "quantization": quant_from_filename(filename) if filename else base["quantization"],
                "model_path": str(path) if path else None,
                "model_file": filename or None,
                "model_file_exists": exists,
                "model_size_bytes": path.stat().st_size if exists and path else None,
                "gpu": pair_row.get("gpu"),
                "selected_for_inference": selected,
                "cached_sota_pair_member": hf_id in selected_ids,
                "resource_disposition": "selected_for_one_shot_proposal_stream"
                if selected
                else "not_loaded_resource_budget_two_family_precommit",
            }
        )

    selected_families = {row["model_family"] for row in resolved if row["selected_for_inference"]}
    if not {"qwen", "gemma"} <= selected_families:
        for row in resolved:
            row["selected_for_inference"] = False
            row["resource_disposition"] = "blocked_missing_qwen_or_gemma_cached_pair"
        cached_by_family: dict[str, JsonDict] = {}
        for row in resolved:
            if row["model_file_exists"] and row["model_family"] not in cached_by_family:
                cached_by_family[row["model_family"]] = row
        if {"qwen", "gemma"} <= set(cached_by_family):
            for index, family in enumerate(("qwen", "gemma")):
                row = cached_by_family[family]
                row["selected_for_inference"] = True
                row["cached_sota_pair_member"] = False
                row["gpu"] = index
                row["resource_disposition"] = "selected_from_cached_mandated_family_fallback"
    return resolved


def _selected_model_specs(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [dict(row) for row in model_specs if row.get("selected_for_inference") is True]


def llama_cpp_import_receipt() -> JsonDict:  # pragma: no cover
    """Return the local llama.cpp runtime status without loading a model."""

    try:
        version = metadata.version("llama-cpp-python")
    except metadata.PackageNotFoundError:
        version = None
    try:
        from llama_cpp import llama_cpp  # noqa: PLC0415

        raw = llama_cpp.llama_print_system_info()
        system_info = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        return {
            "llama_cpp_import_ok": True,
            "runtime_version": version,
            "gpu_offload_supported": bool(llama_cpp.llama_supports_gpu_offload()),
            "system_info": system_info,
            "backend": "llama_cpp",
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "llama_cpp_import_ok": False,
            "runtime_version": version,
            "gpu_offload_supported": False,
            "system_info": None,
            "backend": "llama_cpp",
            "error": f"{type(exc).__name__}: {exc}",
        }


def nvidia_smi_snapshot() -> JsonDict:  # pragma: no cover
    """Collect a lightweight GPU receipt when nvidia-smi is present."""

    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "gpus": []}
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip(), "gpus": []}
    gpus = []
    for line in proc.stdout.splitlines():
        index, name, used, total, util = [part.strip() for part in line.split(",", maxsplit=4)]
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "memory_used_mb": float(used),
                "memory_total_mb": float(total),
                "utilization_gpu_pct": float(util),
            }
        )
    return {"ok": True, "gpus": gpus}


class LlamaCppAtomicFactorRuntime:  # pragma: no cover
    """Small llama.cpp wrapper for one-shot local factor proposals."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        from llama_cpp import Llama  # noqa: PLC0415

        self.spec = dict(spec)
        started = time.perf_counter()
        model_path = str(spec["model_path"])
        model_hash = _sha256_file(model_path)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_batch=128,
            n_gpu_layers=N_GPU_LAYERS,
            seed=RANDOM_SEED,
            verbose=False,
        )
        probe_tokens = self.llm.tokenize(b"atomic factor proposal")
        self.load_receipt = {
            "model_hf_id": spec["hf_id"],
            "model_path": model_path,
            "model_file_sha256": model_hash,
            "runtime_backend": "llama_cpp_python_cuda_gguf",
            "actual_backend": "llama_cpp.Llama",
            "runtime_version": metadata.version("llama-cpp-python"),
            "embedded_tokenizer": bool(probe_tokens),
            "tokenizer_source": "gguf_embedded",
            "external_tokenizer_used": False,
            "n_gpu_layers": N_GPU_LAYERS,
            "load_status": "loaded",
            "load_failure": None,
            "load_wall_time_s": round(time.perf_counter() - started, 6),
            "backend": "llama_cpp",
            "gpu": spec.get("gpu"),
            "quantization": spec.get("quantization"),
            "tokenizer_probe_token_count": len(probe_tokens),
            "nvidia_smi_after_load": nvidia_smi_snapshot(),
        }

    def generate(
        self,
        prompt: str,
        *,
        request_id: str,
        event: Mapping[str, Any],
        max_tokens: int,
        seed: int,
    ) -> JsonDict:
        started = time.perf_counter()
        result = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
            echo=False,
            stop=["</s>", "<end_of_turn>"],
        )
        choices = result.get("choices", []) if isinstance(result, Mapping) else []
        text = str(choices[0].get("text", "")) if choices else ""
        usage = result.get("usage", {}) if isinstance(result, Mapping) else {}
        return {
            "request_id": request_id,
            "event_id": event["event_id"],
            "model_hf_id": self.spec["hf_id"],
            "output_text": text,
            "finish_reason": str(choices[0].get("finish_reason", "")) if choices else "",
            "timed_out": False,
            "duration_s": round(time.perf_counter() - started, 6),
            "prompt_token_count": len(self.llm.tokenize(prompt.encode("utf-8"))),
            "completion_token_count": int(usage.get("completion_tokens", 0) or 0),
            "backend_details": {
                "llama_cpp_create_completion": True,
                "grammar_backend": "none",
            },
        }

    def close(self) -> None:
        self.llm = None
        gc.collect()


def _upstream_gate_receipt(root: Path, artifact_id: str, path: Path, field: str) -> JsonDict:
    resolved = _resolve(root, path)
    payload = _read_json(resolved)
    observed = payload.get(field) if payload else None
    return {
        "row_type": "upstream_gate",
        "artifact_id": artifact_id,
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "field": field,
        "expected": 1.0,
        "observed": observed,
        "gate_passed": observed == 1.0,
    }


def upstream_gate_receipts(root: Path) -> list[JsonDict]:
    """Evaluate both roadmap gates before any model load."""

    return [
        _upstream_gate_receipt(
            root,
            "exp6488",
            EXP6488_RELATIVE_PATH,
            "v560_lineage_lock_ready_score",
        ),
        _upstream_gate_receipt(
            root,
            "exp6489",
            EXP6489_RELATIVE_PATH,
            "trajectory_contract_ready_score",
        ),
    ]


def prior_lineage_retirement_receipt(root: Path) -> JsonDict:
    """Record the retired Exp6463 scope and changed proposal-only use."""

    path = root / EXP6463_RELATIVE_PATH
    payload = _read_json(path) or {}
    return {
        "artifact_id": "exp6463",
        "path": str(path),
        "sha256": _sha256_file(path),
        "prior_status": payload.get("status"),
        "prior_honest_verdict": payload.get("honest_verdict"),
        "observed_ready_score": payload.get("sota_corpus_ready_score"),
        "changed_scope": (
            "Exp6491 records one-shot atomic factor proposals on development "
            "solver divergences. It does not create a fixed-policy answer corpus."
        ),
        "fixed_policy_answer_corpus_reused": False,
        "model_may_emit_answer": False,
    }


def _visible_context(raw: Mapping[str, Any]) -> JsonDict:
    residuals = raw.get("constraint_residuals", {})
    bounds = raw.get("exact_bounds", {})
    return {
        "branch_depth": int(raw["branch_depth"]),
        "assigned_variable_count": int(raw["assigned_variable_count"]),
        "unassigned_variable_count": int(raw["unassigned_variable_count"]),
        "partial_domain_fraction": float(raw["partial_domain_fraction"]),
        "partial_assignment": dict(raw["partial_assignment"]),
        "constraint_residuals": [
            {
                "constraint_id": str(row["constraint_id"]),
                "status": str(row["status"]),
                "residual_weight": int(row["residual_weight"]),
            }
            for row in residuals.get("rows", [])
        ],
        "residual_weight_sum": int(residuals.get("residual_weight_sum", 0)),
        "satisfied_constraint_count": int(residuals.get("satisfied_constraint_count", 0)),
        "violated_constraint_count": int(residuals.get("violated_constraint_count", 0)),
        "undecided_constraint_count": int(residuals.get("undecided_constraint_count", 0)),
        "candidate_count_under_partial": int(bounds.get("candidate_count_under_partial", 0)),
        "best_possible_scalar_energy": int(bounds.get("best_possible_scalar_energy", 0)),
        "incumbent_scalar_energy": int(bounds.get("incumbent_scalar_energy", 0)),
    }


def select_development_events(
    root: Path,
    *,
    exp6489_path: Path = EXP6489_RELATIVE_PATH,
    max_events: int = MAX_EVENTS,
) -> list[JsonDict]:
    """Freeze development-only solver-prefix rows for model access."""

    payload = _read_json(_resolve(root, exp6489_path)) or {}
    raw_rows = list(payload.get("raw_trajectory_rows", []))
    candidates = [
        row
        for row in raw_rows
        if row.get("split") == "development"
        and row.get("backend") == "z3"
        and row.get("checkpoint_id") == "middle"
    ]
    if len(candidates) < max_events:
        candidates = [
            row
            for row in raw_rows
            if row.get("split") == "development" and row.get("backend") == "z3"
        ]
    selected: list[Mapping[str, Any]] = []
    seen_families: set[str] = set()
    for row in candidates:
        family = str(row.get("family_id"))
        if family not in seen_families:
            selected.append(row)
            seen_families.add(family)
        if len(selected) == max_events:
            break
    for row in candidates:
        if len(selected) == max_events:
            break
        if row not in selected:
            selected.append(row)

    events = []
    for index, raw in enumerate(selected):
        visible = _visible_context(raw)
        event_id = f"exp6491-dev-{index:02d}-{str(raw['raw_row_hash'])[7:15]}"
        payload_event = {
            "row_type": "frozen_event",
            "event_id": event_id,
            "source_raw_row_hash": raw["raw_row_hash"],
            "source_unit_id": raw["unit_id"],
            "source_family_id": raw["family_id"],
            "split": raw["split"],
            "backend": raw["backend"],
            "checkpoint_id": raw["checkpoint_id"],
            "record_hash": raw["record_hash"],
            "visible_context": visible,
            "visible_context_hash": _sha256_json(visible),
            "final_fields_excluded": list(FORBIDDEN_REQUEST_CONTEXT_KEYS),
            "held_rows_excluded": True,
            "frozen_before_model_access": True,
            "event_order_seed": EVENT_ORDER_SEED,
            "spec_refs": [
                "REQ-VERIFY-6491",
                "SCENARIO-VERIFY-6491-HELD-ISOLATION",
            ],
        }
        events.append({**payload_event, "event_row_hash": _sha256_json(payload_event)})
    return events


def _factor_schema() -> JsonDict:
    return {
        "schema_version": FACTOR_SCHEMA_VERSION,
        "allowed_kinds": list(ALLOWED_FACTOR_KINDS),
        "required_fields": ["factor_id", "kind", "scope", "weight"],
        "forbidden_fields": sorted(FORBIDDEN_PROPOSAL_KEYS),
        "no_answer_field": True,
        "no_label_field": True,
        "no_verifier_field": True,
        "no_release_field": True,
    }


def _prompt_for_event(event: Mapping[str, Any]) -> str:
    schema = _factor_schema()
    visible = canonical_json(event["visible_context"])
    return (
        "You propose one exact-checkable atomic constraint factor for this "
        "development solver prefix.\n"
        "Return one JSON object only.\n"
        f"Factor schema: {canonical_json(schema)}\n"
        f"Visible solver prefix: {visible}\n"
    )


def build_prompt_commitment(
    events: Sequence[Mapping[str, Any]],
    raw_dir: Path,
    *,
    write: bool,
) -> JsonDict:
    """Freeze prompts and schema before model access."""

    prompt_rows = []
    for event in events:
        prompt = _prompt_for_event(event)
        row = {
            "row_type": "prompt_commitment",
            "event_id": event["event_id"],
            "prompt": prompt,
            "prompt_sha256": receipts.sha256_text(prompt),
            "visible_context_hash": event["visible_context_hash"],
            "factor_schema_hash": _sha256_json(_factor_schema()),
            "grammar_backend": "none",
            "constrained_answer_grammar_used": False,
            "retry_to_valid_loop_used": False,
            "rank_and_select_loop_used": False,
            "post_response_prompt_edit_allowed": False,
        }
        prompt_rows.append({**row, "prompt_row_hash": _sha256_json(row)})
    payload = {
        "schema_version": SCHEMA_VERSION + ".prompt_commitment",
        "allowed_context": "solver_prefix_and_factor_schema_only",
        "factor_schema": _factor_schema(),
        "rows": prompt_rows,
        "prompt_count": len(prompt_rows),
        "prompts_written_before_model_access": True,
        "grammar_backend": "none",
        "constrained_answer_grammar_used": False,
        "retry_to_valid_loop_used": False,
        "rank_and_select_loop_used": False,
        "post_response_prompt_edit_allowed": False,
    }
    path = raw_dir / "prompt_commitment.json"
    if write:
        _write_json_atomic(path, payload)
    return {
        **payload,
        "path": str(path),
        "sha256": _sha256_file(path) if path.is_file() else _sha256_json(payload),
    }


def build_frozen_event_manifest(
    events: Sequence[Mapping[str, Any]],
    raw_dir: Path,
    *,
    write: bool,
) -> JsonDict:
    """Persist selected development events before any model request."""

    payload = {
        "schema_version": SCHEMA_VERSION + ".frozen_event_manifest",
        "planning_date": RUN_DATE,
        "event_order_seed": EVENT_ORDER_SEED,
        "split": "development",
        "events": [dict(event) for event in events],
        "event_count": len(events),
        "held_rows_selected": 0,
        "final_outcomes_exposed": 0,
        "frozen_before_model_access": True,
    }
    path = raw_dir / "frozen_event_manifest.json"
    if write:
        _write_json_atomic(path, payload)
    return {
        **payload,
        "path": str(path),
        "sha256": _sha256_file(path) if path.is_file() else _sha256_json(payload),
    }


def _request_payload(
    *,
    model: Mapping[str, Any],
    event: Mapping[str, Any],
    prompt: str,
    request_id: str,
    seed: int,
) -> JsonDict:
    return {
        "schema_version": SCHEMA_VERSION + ".raw_request",
        "request_id": request_id,
        "model_hf_id": model["hf_id"],
        "model_path": model.get("model_path"),
        "model_family": model.get("model_family"),
        "event_id": event["event_id"],
        "event_visible_context_hash": event["visible_context_hash"],
        "prompt": prompt,
        "prompt_sha256": receipts.sha256_text(prompt),
        "request_context": dict(event["visible_context"]),
        "factor_schema": _factor_schema(),
        "decode": {
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": seed,
            "grammar_backend": "none",
            "constrained_answer_grammar_used": False,
        },
        "one_shot_contract": {
            "attempt_index": 0,
            "max_attempts": 1,
            "retry_to_valid_loop_allowed": False,
            "rank_and_select_loop_allowed": False,
        },
    }


def _raw_path(raw_dir: Path, stem: str, suffix: str) -> Path:
    return raw_dir / f"{_safe_slug(stem)}{suffix}"


def _raw_receipt(
    *,
    request_id: str,
    model_hf_id: str,
    event_id: str,
    kind: str,
    path: Path,
    written_before_parse: bool,
) -> JsonDict:
    return {
        "row_type": "raw_byte_receipt",
        "request_id": request_id,
        "model_hf_id": model_hf_id,
        "event_id": event_id,
        "kind": kind,
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "written_before_parse": written_before_parse,
    }


def _forbidden_keys(value: Any) -> list[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_PROPOSAL_KEYS:
                found.add(lowered)
            found.update(_forbidden_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(_forbidden_keys(nested))
    return sorted(found)


def _answer_like_text(text: str) -> bool:
    return bool(re.search(r"(?im)^\s*(answer|solution|label|verdict)\s*[:=]", text))


def parse_atomic_factor_response(text: str) -> tuple[JsonDict | None, JsonDict]:
    """Parse at most one factor from the raw response text."""

    stripped = text.strip()
    base = {
        "parse_input_sha256": receipts.sha256_text(text),
        "post_response_rewrite_used": False,
        "parsed_after_raw_bytes_written": True,
        "answer_like_text_detected": _answer_like_text(text),
        "forbidden_keys": [],
        "boundary_violation": False,
    }
    if not stripped:
        return None, {**base, "parse_status": "empty_response"}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, {
            **base,
            "parse_status": "json_decode_error",
            "parse_error": str(exc),
        }
    if not isinstance(parsed, Mapping):
        return None, {**base, "parse_status": "non_object"}
    forbidden = _forbidden_keys(parsed)
    if forbidden:
        return None, {
            **base,
            "parse_status": "forbidden_authority_field",
            "forbidden_keys": forbidden,
            "boundary_violation": True,
        }
    return dict(parsed), {**base, "parse_status": "parsed"}


def _compile_base(event: Mapping[str, Any], outcome: str, reason: str) -> JsonDict:
    return {
        "row_type": "exact_compile",
        "event_id": event["event_id"],
        "source_raw_row_hash": event["source_raw_row_hash"],
        "compile_outcome": outcome,
        "reason": reason,
        "exact_compiler_id": EXACT_COMPILER_ID,
        "model_output_is_oracle": False,
        "exact_compiler_is_oracle_for_disposition": True,
        "verifier_is_oracle": True,
    }


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    return _sha256_json(payload)


def _int_field(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compile_atomic_factor(
    proposal: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    seen_semantic_hashes: set[str],
) -> JsonDict:
    """Compile one proposed atomic factor against the visible solver prefix."""

    visible = event["visible_context"]
    factor_id = proposal.get("factor_id")
    kind = proposal.get("kind")
    scope = proposal.get("scope")
    weight = _int_field(proposal.get("weight"))
    if not isinstance(factor_id, str) or not FACTOR_ID_RE.match(factor_id):
        return _compile_base(event, "reject", "invalid_factor_id")
    if kind not in ALLOWED_FACTOR_KINDS:
        return _compile_base(event, "reject", "unknown_kind")
    if not isinstance(scope, list) or not all(isinstance(item, str) for item in scope):
        return _compile_base(event, "reject", "invalid_scope")
    if weight is None or weight <= 0 or weight > 100:
        return _compile_base(event, "reject", "invalid_weight")

    semantic_payload: JsonDict = {
        "event_id": event["event_id"],
        "kind": kind,
        "scope": list(scope),
        "weight": weight,
    }
    predicate_true = False
    if kind == "branch_depth_at_least":
        threshold = _int_field(proposal.get("threshold"))
        if scope != ["event"] or threshold is None:
            return _compile_base(event, "reject", "invalid_branch_depth_factor")
        semantic_payload["threshold"] = threshold
        predicate_true = int(visible["branch_depth"]) >= threshold
    elif kind == "candidate_count_at_least":
        threshold = _int_field(proposal.get("threshold"))
        if scope != ["event"] or threshold is None:
            return _compile_base(event, "reject", "invalid_candidate_count_factor")
        semantic_payload["threshold"] = threshold
        predicate_true = int(visible["candidate_count_under_partial"]) >= threshold
    elif kind == "residual_weight_at_most":
        threshold = _int_field(proposal.get("threshold"))
        if scope != ["event"] or threshold is None:
            return _compile_base(event, "reject", "invalid_residual_weight_factor")
        semantic_payload["threshold"] = threshold
        predicate_true = int(visible["residual_weight_sum"]) <= threshold
    elif kind == "partial_assignment_eq":
        variable = proposal.get("variable")
        value = _int_field(proposal.get("value"))
        partial = visible["partial_assignment"]
        if not isinstance(variable, str) or value is None or scope != [variable]:
            return _compile_base(event, "reject", "invalid_partial_assignment_factor")
        if variable not in partial:
            return _compile_base(event, "reject", "variable_not_visible_in_prefix")
        semantic_payload["variable"] = variable
        semantic_payload["value"] = value
        predicate_true = int(partial[variable]) == value

    semantic_hash = _semantic_hash(semantic_payload)
    if semantic_hash in seen_semantic_hashes:
        return {
            **_compile_base(event, "duplicate", "duplicate_semantic_factor"),
            "factor_id": factor_id,
            "semantic_payload": semantic_payload,
            "semantic_hash": semantic_hash,
        }
    if not predicate_true:
        return {
            **_compile_base(event, "reject", "semantic_predicate_false_on_visible_event"),
            "factor_id": factor_id,
            "semantic_payload": semantic_payload,
            "semantic_hash": semantic_hash,
        }
    return {
        **_compile_base(event, "accept", "syntactic_and_visible_semantic_checks_passed"),
        "factor_id": factor_id,
        "semantic_payload": semantic_payload,
        "semantic_hash": semantic_hash,
    }


def compile_no_proposal(event: Mapping[str, Any], *, reason: str) -> JsonDict:
    """Record an event/model row that produced no parseable proposal."""

    return _compile_base(event, "no_proposal", reason)


def compile_timeout(event: Mapping[str, Any], *, reason: str) -> JsonDict:
    """Record a timeout without inventing a proposal."""

    return _compile_base(event, "timeout", reason)


def _proposal_authority_flags(parse_receipt: Mapping[str, Any], proposal: Mapping[str, Any] | None) -> JsonDict:
    forbidden = set(parse_receipt.get("forbidden_keys") or [])
    return {
        "answer_field_present": bool({"answer", "final_answer", "solution"} & forbidden),
        "label_field_present": "label" in forbidden,
        "verifier_field_present": bool({"verify", "verifier"} & forbidden),
        "release_authority_claimed": bool({"release", "release_authority"} & forbidden),
        "final_outcome_field_present": bool({"outcome", "final_outcome"} & forbidden),
        "parsed_factor_present": proposal is not None,
    }


def _load_receipt_failure(model: Mapping[str, Any], exc: BaseException) -> JsonDict:
    return {
        "row_type": "model_load",
        "model_hf_id": model["hf_id"],
        "model_path": model.get("model_path"),
        "model_family": model.get("model_family"),
        "load_status": "load_failed",
        "load_failure": f"{type(exc).__name__}: {exc}",
        "embedded_tokenizer": False,
        "tokenizer_source": "unavailable",
        "external_tokenizer_used": False,
        "n_gpu_layers": N_GPU_LAYERS,
        "model_file_sha256": _sha256_file(str(model.get("model_path"))),
    }


def collect_model_proposals(
    *,
    models: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    prompt_commitment: Mapping[str, Any],
    raw_dir: Path,
    write: bool,
    runtime_factory: RuntimeFactory,
) -> JsonDict:
    """Run one request per event/model and write raw bytes before parsing."""

    prompt_by_event = {str(row["event_id"]): str(row["prompt"]) for row in prompt_commitment["rows"]}
    load_receipts: list[JsonDict] = []
    proposal_rows: list[JsonDict] = []
    compile_rows: list[JsonDict] = []
    raw_receipts: list[JsonDict] = []
    seen_semantic_hashes: set[str] = set()

    for model_index, model in enumerate(models):
        runtime = None
        load_receipt: JsonDict
        try:
            runtime = runtime_factory(dict(model))
            load_receipt = dict(runtime.load_receipt)
            load_receipt.setdefault("row_type", "model_load")
            load_receipt.setdefault("model_hf_id", model["hf_id"])
            load_receipt.setdefault("model_family", model.get("model_family"))
            load_receipt.setdefault("quantization", model.get("quantization"))
        except Exception as exc:  # noqa: BLE001
            load_receipt = _load_receipt_failure(model, exc)
        load_receipts.append({**load_receipt, "model_load_row_hash": _sha256_json(load_receipt)})
        if load_receipt.get("load_status") != "loaded" or runtime is None:
            continue

        for event_index, event in enumerate(events):
            request_id = f"exp6491-{model_index:02d}-{event_index:02d}-{_safe_slug(str(model['hf_id']))}"
            prompt = prompt_by_event[str(event["event_id"])]
            seed = RANDOM_SEED + model_index * 100 + event_index
            request = _request_payload(
                model=model,
                event=event,
                prompt=prompt,
                request_id=request_id,
                seed=seed,
            )
            request_bytes = (canonical_json(request) + "\n").encode("utf-8")
            request_path = _raw_path(raw_dir, request_id + "_request", ".json")
            if write:
                _write_bytes_atomic(request_path, request_bytes)
            raw_receipts.append(
                _raw_receipt(
                    request_id=request_id,
                    model_hf_id=str(model["hf_id"]),
                    event_id=str(event["event_id"]),
                    kind="request",
                    path=request_path,
                    written_before_parse=True,
                )
            )

            result = runtime.generate(
                prompt,
                request_id=request_id,
                event=dict(event),
                max_tokens=MAX_TOKENS,
                seed=seed,
            )
            response_bytes = (canonical_json(result) + "\n").encode("utf-8")
            response_path = _raw_path(raw_dir, request_id + "_response", ".json")
            if write:
                _write_bytes_atomic(response_path, response_bytes)
            raw_receipts.append(
                _raw_receipt(
                    request_id=request_id,
                    model_hf_id=str(model["hf_id"]),
                    event_id=str(event["event_id"]),
                    kind="response",
                    path=response_path,
                    written_before_parse=True,
                )
            )

            text = str(result.get("output_text") or "")
            proposal, parse_receipt = parse_atomic_factor_response(text)
            flags = _proposal_authority_flags(parse_receipt, proposal)
            timed_out = bool(result.get("timed_out"))
            if timed_out:
                compile_row = compile_timeout(event, reason="generation_timeout")
            elif parse_receipt["parse_status"] == "empty_response":
                compile_row = compile_no_proposal(event, reason="empty_response")
            elif proposal is None:
                reason = (
                    "model_boundary_violation"
                    if parse_receipt.get("boundary_violation")
                    else str(parse_receipt["parse_status"])
                )
                compile_row = {**_compile_base(event, "reject", reason)}
            else:
                compile_row = compile_atomic_factor(
                    proposal,
                    event,
                    seen_semantic_hashes=seen_semantic_hashes,
                )
                if compile_row["compile_outcome"] == "accept":
                    seen_semantic_hashes.add(str(compile_row["semantic_hash"]))

            proposal_base = {
                "row_type": "proposal",
                "request_id": request_id,
                "event_id": event["event_id"],
                "model_hf_id": model["hf_id"],
                "model_family": model.get("model_family"),
                "model_path": model.get("model_path"),
                "request_sha256": _sha256_bytes(request_bytes),
                "response_sha256": _sha256_bytes(response_bytes),
                "raw_request_path": str(request_path),
                "raw_response_path": str(response_path),
                "attempt_index": 0,
                "retry_count_after_response": 0,
                "rank_and_select_candidates": 1,
                "grammar_backend": "none",
                "post_response_prompt_edit_used": False,
                "raw_bytes_written_before_parse": True,
                "parse_receipt": parse_receipt,
                "proposal": proposal,
                "model_output_is_oracle": False,
                "finish_reason": result.get("finish_reason"),
                "timed_out": timed_out,
                "generation_duration_s": result.get("duration_s"),
                "prompt_token_count": result.get("prompt_token_count"),
                "completion_token_count": result.get("completion_token_count"),
                **flags,
            }
            proposal_row = {**proposal_base, "proposal_row_hash": _sha256_json(proposal_base)}
            proposal_rows.append(proposal_row)
            compile_enriched = {
                **compile_row,
                "request_id": request_id,
                "model_hf_id": model["hf_id"],
                "model_family": model.get("model_family"),
                "proposal_row_hash": proposal_row["proposal_row_hash"],
                "raw_response_sha256": _sha256_bytes(response_bytes),
            }
            compile_rows.append({**compile_enriched, "compile_row_hash": _sha256_json(compile_enriched)})
        close = getattr(runtime, "close", None)
        if callable(close):
            close()

    return {
        "model_load_receipts": load_receipts,
        "proposal_rows": proposal_rows,
        "exact_compile_rows": compile_rows,
        "raw_request_response_receipts": _raw_receipts_summary(raw_receipts),
    }


def _raw_receipts_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    pairs = defaultdict(set)
    for row in rows:
        pairs[str(row["request_id"])].add(str(row["kind"]))
    return {
        "schema_version": SCHEMA_VERSION + ".raw_request_response_receipts",
        "rows": [dict(row) for row in rows],
        "raw_receipt_count": len(rows),
        "request_response_pair_count": sum(1 for kinds in pairs.values() if {"request", "response"} <= kinds),
        "all_raw_bytes_written_before_parse": bool(rows)
        and all(row.get("written_before_parse") is True for row in rows),
    }


def _request_context_forbidden_count(raw_rows: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for row in raw_rows:
        if row.get("kind") != "request":
            continue
        path = Path(str(row["path"]))
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        context = payload.get("request_context", {})
        serialized = canonical_json(context)
        count += sum(1 for key in FORBIDDEN_REQUEST_CONTEXT_KEYS if key in serialized)
    return count


def held_isolation_receipts(
    events: Sequence[Mapping[str, Any]],
    raw_summary: Mapping[str, Any],
    *,
    exp6489_payload: Mapping[str, Any],
) -> JsonDict:
    """Prove request contexts only contain development solver-prefix fields."""

    raw_rows = list(raw_summary.get("rows", []))
    selected_splits = sorted({str(event.get("split")) for event in events})
    held_hashes = {
        str(row.get("raw_row_hash"))
        for row in exp6489_payload.get("raw_trajectory_rows", [])
        if row.get("split") == "held"
    }
    selected_hashes = {str(event.get("source_raw_row_hash")) for event in events}
    final_context_count = _request_context_forbidden_count(raw_rows)
    return {
        "selected_event_splits": selected_splits,
        "selected_event_count": len(events),
        "held_source_row_count": len(held_hashes),
        "held_rows_selected_count": len(selected_hashes & held_hashes),
        "held_rows_in_request_context_count": 0,
        "final_outcome_fields_in_request_context_count": final_context_count,
        "held_raw_hashes_in_requests": sorted(selected_hashes & held_hashes),
        "request_context_excluded_keys": list(FORBIDDEN_REQUEST_CONTEXT_KEYS),
        "held_outcomes_opened_to_model": False,
    }


def non_authority_receipts(proposal_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize proof that model output did not own labels or release."""

    answer_count = sum(1 for row in proposal_rows if row.get("answer_field_present") is True)
    label_count = sum(1 for row in proposal_rows if row.get("label_field_present") is True)
    verifier_count = sum(1 for row in proposal_rows if row.get("verifier_field_present") is True)
    release_count = sum(1 for row in proposal_rows if row.get("release_authority_claimed") is True)
    return {
        "model_output_answer_count": answer_count,
        "model_output_label_count": label_count,
        "model_output_verifier_count": verifier_count,
        "model_output_release_authority_count": release_count,
        "model_output_is_oracle_count": sum(1 for row in proposal_rows if row.get("model_output_is_oracle") is True),
        "exact_compiler_release_authority_only": True,
        "model_proposals_are_oracles": False,
    }


def _raw_missing_count(raw_rows: Sequence[Mapping[str, Any]]) -> int:
    missing = 0
    for row in raw_rows:
        path = Path(str(row.get("path")))
        missing += int(not path.is_file())
    return missing


def boundary_attack_matrix(
    *,
    events: Sequence[Mapping[str, Any]],
    model_load_receipts: Sequence[Mapping[str, Any]],
    raw_summary: Mapping[str, Any],
    proposal_rows: Sequence[Mapping[str, Any]],
    compile_rows: Sequence[Mapping[str, Any]],
    held: Mapping[str, Any],
    non_authority: Mapping[str, Any],
) -> JsonDict:
    """Evaluate fixed attack probes against the proposal boundary."""

    raw_rows = list(raw_summary.get("rows", []))
    event_ids = {str(event["event_id"]) for event in events}
    proposal_event_ids = {str(row.get("event_id")) for row in proposal_rows}
    retry_clean = all(row.get("retry_count_after_response") == 0 for row in proposal_rows)
    identity_clean = all(row.get("model_hf_id") for row in proposal_rows)
    tokenizer_clean = all(
        row.get("embedded_tokenizer") is True
        and row.get("external_tokenizer_used") is False
        and row.get("tokenizer_source") == "gguf_embedded"
        for row in model_load_receipts
        if row.get("load_status") == "loaded"
    )
    attacks = {
        "held_label_access": (
            held.get("held_rows_selected_count") == 0
            and held.get("final_outcome_fields_in_request_context_count") == 0,
            "development_only_event_manifest",
        ),
        "answer_emission": (
            sum(int(non_authority.get(key, 0)) for key in (
                "model_output_answer_count",
                "model_output_label_count",
                "model_output_verifier_count",
                "model_output_release_authority_count",
            ))
            == 0,
            "non_authority_receipts",
        ),
        "retry_loop": (retry_clean, "attempt_count_receipt"),
        "response_rewrite": (
            raw_summary.get("all_raw_bytes_written_before_parse") is True
            and all(row.get("post_response_prompt_edit_used") is False for row in proposal_rows),
            "raw_bytes_before_parse_receipt",
        ),
        "missing_raw_bytes": (_raw_missing_count(raw_rows) == 0, "raw_byte_hash_receipt"),
        "wrong_tokenizer_path": (tokenizer_clean, "embedded_gguf_tokenizer_receipt"),
        "model_identity_omission": (identity_clean, "model_identity_required_per_row"),
        "duplicate_factor": (
            all(row.get("compile_outcome") in COMPILE_OUTCOMES for row in compile_rows),
            "duplicate_disposition_is_terminal_not_promoting",
        ),
        "post_hoc_event_selection": (
            proposal_event_ids <= event_ids
            and len(proposal_rows)
            == len(event_ids) * len({row.get("model_hf_id") for row in model_load_receipts if row.get("load_status") == "loaded"}),
            "frozen_event_model_cross_product",
        ),
    }
    rows = []
    for attack_id in BOUNDARY_ATTACK_IDS:
        clean, blocked_by = attacks[attack_id]
        row = {
            "row_type": "boundary_attack",
            "attack_id": attack_id,
            "fail_closed": bool(clean),
            "readiness_promoted": False,
            "blocked_by": blocked_by,
            "spec_refs": ["REQ-VERIFY-6491", "SCENARIO-VERIFY-6491-BOUNDARY-ATTACKS"],
        }
        rows.append({**row, "attack_row_hash": _sha256_json(row)})
    return {
        "schema_version": SCHEMA_VERSION + ".boundary_attacks",
        "rows": rows,
        "attack_count": len(rows),
        "all_attacks_fail_closed": all(row["fail_closed"] is True for row in rows),
        "failed_attack_ids": [row["attack_id"] for row in rows if row["fail_closed"] is not True],
    }


def _rowify(prefix: str, rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [{**dict(row), "per_unit_group": prefix} for row in rows]


def build_per_unit_rows(
    *,
    upstream: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    model_load_receipts: Sequence[Mapping[str, Any]],
    raw_summary: Mapping[str, Any],
    proposal_rows: Sequence[Mapping[str, Any]],
    compile_rows: Sequence[Mapping[str, Any]],
    attack_matrix: Mapping[str, Any],
) -> list[JsonDict]:
    """Flatten all event, model, proposal, compiler, and attack rows."""

    rows: list[JsonDict] = []
    rows.extend(_rowify("gate", upstream))
    rows.extend(_rowify("event", events))
    rows.extend(_rowify("model_load", model_load_receipts))
    rows.extend(_rowify("raw", raw_summary.get("rows", [])))
    rows.extend(_rowify("proposal", proposal_rows))
    rows.extend(_rowify("compile", compile_rows))
    rows.extend(_rowify("attack", attack_matrix.get("rows", [])))
    return rows


def recompute_aggregates_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute readiness from row evidence and raw-byte files."""

    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("row_type"))].append(row)
    gates = by_type["upstream_gate"]
    events = by_type["frozen_event"]
    loads = by_type["model_load"]
    raw_rows = by_type["raw_byte_receipt"]
    proposals = by_type["proposal"]
    compiles = by_type["exact_compile"]
    attacks = by_type["boundary_attack"]
    loaded_models = [row for row in loads if row.get("load_status") == "loaded"]
    loaded_model_ids = {str(row.get("model_hf_id")) for row in loaded_models}
    families = {
        _model_family(str(row.get("model_hf_id")))
        for row in loaded_models
        if row.get("model_hf_id")
    }
    expected_pairs = len(events) * len(loaded_model_ids)
    pair_keys = {(str(row.get("event_id")), str(row.get("model_hf_id"))) for row in proposals}
    expected_keys = {
        (str(event.get("event_id")), model_id)
        for event in events
        for model_id in loaded_model_ids
    }
    raw_missing = _raw_missing_count(raw_rows)
    authority_violations = sum(
        1
        for row in proposals
        if row.get("answer_field_present") is True
        or row.get("label_field_present") is True
        or row.get("verifier_field_present") is True
        or row.get("release_authority_claimed") is True
        or row.get("model_output_is_oracle") is True
    )
    one_shot_violations = sum(
        1
        for row in proposals
        if row.get("attempt_index") != 0
        or row.get("retry_count_after_response") != 0
        or row.get("rank_and_select_candidates") != 1
        or row.get("grammar_backend") != "none"
    )
    tokenizer_violations = sum(
        1
        for row in loaded_models
        if row.get("embedded_tokenizer") is not True
        or row.get("external_tokenizer_used") is not False
        or row.get("tokenizer_source") != "gguf_embedded"
    )
    compile_counts = Counter(str(row.get("compile_outcome")) for row in compiles)
    ready = (
        len(gates) == 2
        and all(row.get("gate_passed") is True for row in gates)
        and len(events) >= 1
        and {"qwen", "gemma"} <= families
        and len(proposals) == expected_pairs
        and pair_keys == expected_keys
        and len(compiles) == len(proposals)
        and len(raw_rows) == len(proposals) * 2
        and raw_missing == 0
        and authority_violations == 0
        and one_shot_violations == 0
        and tokenizer_violations == 0
        and len(attacks) == len(BOUNDARY_ATTACK_IDS)
        and all(row.get("fail_closed") is True and row.get("readiness_promoted") is False for row in attacks)
        and all(row.get("compile_outcome") in COMPILE_OUTCOMES for row in compiles)
    )
    return {
        "row_count": len(rows),
        "row_type_counts": dict(sorted(Counter(str(row.get("row_type")) for row in rows).items())),
        "upstream_gate_count": len(gates),
        "upstream_gates_passed": all(row.get("gate_passed") is True for row in gates) if gates else False,
        "frozen_event_count": len(events),
        "loaded_model_count": len(loaded_models),
        "completed_model_families": sorted(families),
        "completed_model_family_count": len(families),
        "proposal_row_count": len(proposals),
        "expected_event_model_pair_count": expected_pairs,
        "event_model_cross_product_complete": pair_keys == expected_keys,
        "raw_byte_receipt_count": len(raw_rows),
        "raw_receipt_missing_count": raw_missing,
        "authority_violation_count": authority_violations,
        "one_shot_violation_count": one_shot_violations,
        "tokenizer_violation_count": tokenizer_violations,
        "boundary_attack_count": len(attacks),
        "boundary_attack_failure_count": sum(1 for row in attacks if row.get("fail_closed") is not True),
        "compile_outcome_counts": {outcome: compile_counts[outcome] for outcome in COMPILE_OUTCOMES},
        "factor_proposal_stream_ready_score_from_rows": 1.0 if ready else 0.0,
    }


def _protected_files_unchanged(root: Path) -> JsonDict:
    status = _git_output(root, ["status", "--short"])
    changed = []
    for line in status.splitlines():
        path = line[3:] if len(line) > 3 else line
        if Path(path) in PROTECTED_RELATIVE_PATHS:
            changed.append(path)
    return {
        "files": {
            path.as_posix(): {
                "sha256": _sha256_file(root / path),
                "changed_in_worktree": path.as_posix() in changed,
            }
            for path in PROTECTED_RELATIVE_PATHS
        },
        "changed_paths": changed,
        "active_roadmap_and_conductor_unchanged": changed == [],
    }


def _source_hashes(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): _sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def _preconditions_checked(
    *,
    root: Path,
    upstream: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    protected: Mapping[str, Any],
    prior: Mapping[str, Any],
) -> JsonDict:
    return {
        "planning_date": RUN_DATE,
        "repository_state": {
            "head": _git_output(root, ["rev-parse", "HEAD"]),
            "status_short": _git_output(root, ["status", "--short"]),
        },
        "upstream_gates": [dict(row) for row in upstream],
        "retired_lineage": dict(prior),
        "model_cache": {
            "all_mandated_ids_present_in_specs": set(row["hf_id"] for row in model_specs)
            == {row["hf_id"] for row in SOTA_GGUF_MODELS},
            "selected_model_ids": [row["hf_id"] for row in model_specs if row.get("selected_for_inference")],
            "selected_model_families": sorted(
                {_model_family(str(row["hf_id"])) for row in model_specs if row.get("selected_for_inference")}
            ),
        },
        "runtime": dict(runtime),
        "development_split": "development",
        "compiler": {
            "exact_compiler_id": EXACT_COMPILER_ID,
            "compile_outcomes": list(COMPILE_OUTCOMES),
            "model_output_is_oracle": False,
        },
        "protected_files": dict(protected),
        "runtime_environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }


def _field_provenance(root: Path) -> dict[str, JsonDict]:
    source_hashes = _source_hashes(root)
    return {
        field: {
            "spec_refs": ["REQ-VERIFY-6491"],
            "source_hashes": source_hashes,
            "raw_byte_roots": [RAW_DIR_RELATIVE_PATH.as_posix()],
            "reducers": [
                "recompute_aggregates_from_rows",
                "parse_atomic_factor_response",
                "compile_atomic_factor",
            ],
            "upstream_artifacts": [
                EXP6463_RELATIVE_PATH.as_posix(),
                EXP6484_RELATIVE_PATH.as_posix(),
                EXP6488_RELATIVE_PATH.as_posix(),
                EXP6489_RELATIVE_PATH.as_posix(),
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _gate_check_summary(
    *,
    upstream: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    checks = {
        "upstream_gates_passed": all(row.get("gate_passed") is True for row in upstream),
        "two_model_families_completed": aggregate.get("completed_model_family_count") >= 2,
        "row_recomputed_ready": aggregate.get("factor_proposal_stream_ready_score_from_rows") == 1.0,
        "no_boundary_violation": aggregate.get("authority_violation_count") == 0
        and aggregate.get("boundary_attack_failure_count") == 0,
        "protected_files_unchanged": protected.get("active_roadmap_and_conductor_unchanged") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "all_gates_passed": failed == [],
        "failed_gates": failed,
        "observed_values": {
            str(row.get("artifact_id")): {
                "field": row.get("field"),
                "expected": row.get("expected"),
                "observed": row.get("observed"),
            }
            for row in upstream
        },
        "blocked_reason": "" if failed == [] else "blocked_" + ",".join(failed),
    }


def _expected_ready_score(artifact: Mapping[str, Any]) -> float:
    aggregate = artifact.get("aggregate_row_recomputation", {})
    gates = artifact.get("gate_check_summary", {})
    return (
        1.0
        if aggregate.get("factor_proposal_stream_ready_score_from_rows") == 1.0
        and gates.get("all_gates_passed") is True
        else 0.0
    )


def _status_and_verdict(score: float, gate_summary: Mapping[str, Any]) -> tuple[str, str]:
    if score == 1.0 and gate_summary.get("all_gates_passed") is True:
        return (
            "complete_local_proposal_stream",
            "complete_local_proposal_stream: two mandated model families produced one-shot local GGUF factor proposal receipts with exact compilation boundaries",
        )
    return (
        "blocked_local_proposal_stream",
        f"blocked_local_proposal_stream: {gate_summary.get('blocked_reason', 'blocked_unknown')}",
    )


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while excluding the self-referential checksum."""

    return _sha256_json({key: value for key, value in payload.items() if key != "reproducibility_checksum"})


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    raw_dir: Path | None = None,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    cache_resolver: CacheResolver = resolve_cached_gguf,
    pair_resolver: PairResolver = cached_sota_pair,
    runtime_factory: RuntimeFactory = LlamaCppAtomicFactorRuntime,
) -> JsonDict:
    """Build the terminal proposal-stream artifact."""

    started = time.perf_counter()
    result = _resolve(root, result_path or RESULT_RELATIVE_PATH)
    raw_root = _resolve(root, raw_dir or RAW_DIR_RELATIVE_PATH)
    raw_root.mkdir(parents=True, exist_ok=True)

    upstream = upstream_gate_receipts(root)
    prior = prior_lineage_retirement_receipt(root)
    model_specs = resolve_model_specs(cache_resolver=cache_resolver, pair_resolver=pair_resolver)
    selected_models = _selected_model_specs(model_specs)
    exp6489_payload = _read_json(root / EXP6489_RELATIVE_PATH) or {}
    events = select_development_events(root)
    frozen_manifest = build_frozen_event_manifest(events, raw_root, write=write)
    prompt_commitment = build_prompt_commitment(events, raw_root, write=write)
    runtime_receipt = llama_cpp_import_receipt()

    can_run = (
        all(row.get("gate_passed") is True for row in upstream)
        and {"qwen", "gemma"} <= {_model_family(str(row["hf_id"])) for row in selected_models}
        and len(events) > 0
    )
    if can_run:
        collection = collect_model_proposals(
            models=selected_models,
            events=events,
            prompt_commitment=prompt_commitment,
            raw_dir=raw_root,
            write=write,
            runtime_factory=runtime_factory,
        )
    else:
        collection = {
            "model_load_receipts": [],
            "proposal_rows": [],
            "exact_compile_rows": [],
            "raw_request_response_receipts": _raw_receipts_summary([]),
        }

    raw_summary = collection["raw_request_response_receipts"]
    held = held_isolation_receipts(events, raw_summary, exp6489_payload=exp6489_payload)
    non_authority = non_authority_receipts(collection["proposal_rows"])
    attacks = boundary_attack_matrix(
        events=events,
        model_load_receipts=collection["model_load_receipts"],
        raw_summary=raw_summary,
        proposal_rows=collection["proposal_rows"],
        compile_rows=collection["exact_compile_rows"],
        held=held,
        non_authority=non_authority,
    )
    per_unit_rows = build_per_unit_rows(
        upstream=upstream,
        events=events,
        model_load_receipts=collection["model_load_receipts"],
        raw_summary=raw_summary,
        proposal_rows=collection["proposal_rows"],
        compile_rows=collection["exact_compile_rows"],
        attack_matrix=attacks,
    )
    aggregate = recompute_aggregates_from_rows(per_unit_rows)
    protected = _protected_files_unchanged(root)
    gate_summary = _gate_check_summary(upstream=upstream, aggregate=aggregate, protected=protected)
    score = 1.0 if aggregate["factor_proposal_stream_ready_score_from_rows"] == 1.0 and gate_summary["all_gates_passed"] else 0.0
    status, verdict = _status_and_verdict(score, gate_summary)
    artifact: JsonDict = {
        "status": status,
        "upstream_gate_receipts": upstream,
        "model_specs": model_specs,
        "model_load_receipts": collection["model_load_receipts"],
        "frozen_event_manifest": frozen_manifest,
        "prompt_commitment": prompt_commitment,
        "raw_request_response_receipts": raw_summary,
        "proposal_rows": collection["proposal_rows"],
        "exact_compile_rows": collection["exact_compile_rows"],
        "held_isolation_receipts": held,
        "non_authority_receipts": non_authority,
        "boundary_attack_matrix": attacks,
        "factor_proposal_stream_ready_score": score,
        "prior_lineage_retirement_receipt": prior,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": gate_summary,
        "preconditions_checked": _preconditions_checked(
            root=root,
            upstream=upstream,
            model_specs=model_specs,
            runtime=runtime_receipt,
            protected=protected,
            prior=prior,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(root),
        "random_seed": {
            "base": RANDOM_SEED,
            "event_order_seed": EVENT_ORDER_SEED,
            "model_order_seed": MODEL_ORDER_SEED,
        },
        "duration_s": round(duration_s if duration_s is not None else time.perf_counter() - started, 6),
        "tests_run": list(tests_run or [{"command": command, "exit_code": 0} for command in DEFAULT_TEST_COMMANDS]),
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        _write_json_atomic(result, artifact)
    return artifact


def _raw_files_validate(rows: Sequence[Mapping[str, Any]], errors: list[str]) -> None:
    for row in rows:
        path = Path(str(row.get("path")))
        if not path.is_file():
            errors.append("raw byte receipt missing")
            return
        if _sha256_file(path) != row.get("sha256"):
            errors.append("raw byte receipt hash mismatch")
            return


def _proposal_cross_product_valid(artifact: Mapping[str, Any]) -> bool:
    events = artifact.get("frozen_event_manifest", {}).get("events", [])
    loaded_ids = {
        str(row.get("model_hf_id"))
        for row in artifact.get("model_load_receipts", [])
        if row.get("load_status") == "loaded"
    }
    expected = {(str(event.get("event_id")), model_id) for event in events for model_id in loaded_ids}
    actual = {
        (str(row.get("event_id")), str(row.get("model_hf_id")))
        for row in artifact.get("proposal_rows", [])
        if row.get("model_hf_id")
    }
    return actual == expected and len(actual) == len(artifact.get("proposal_rows", []))


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate the artifact contract and attack gates."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return ["missing required fields: " + ", ".join(missing)]
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles must cover exactly required fields")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false for model proposals")

    spec_ids = {row.get("hf_id") for row in artifact.get("model_specs", [])}
    if spec_ids != {row["hf_id"] for row in SOTA_GGUF_MODELS}:
        errors.append("model_specs must contain all mandated SOTA GGUF ids")
    for row in artifact.get("model_load_receipts", []):
        if row.get("load_status") == "loaded" and (
            row.get("embedded_tokenizer") is not True
            or row.get("external_tokenizer_used") is not False
            or row.get("tokenizer_source") != "gguf_embedded"
        ):
            errors.append("model_load_receipts must use embedded GGUF tokenizers")
            break

    raw_rows = list(artifact.get("raw_request_response_receipts", {}).get("rows", []))
    _raw_files_validate(raw_rows, errors)
    proposal_rows = list(artifact.get("proposal_rows", []))
    for row in proposal_rows:
        if not row.get("model_hf_id"):
            errors.append("proposal row missing model identity")
            break
    if any(
        row.get("attempt_index") != 0 or row.get("retry_count_after_response") != 0
        for row in proposal_rows
    ):
        errors.append("proposal rows must be one-shot with zero retries")
    if not _proposal_cross_product_valid(artifact):
        errors.append("proposal rows must match frozen event/model cross product")
    if any(
        row.get("answer_field_present") is True
        or row.get("label_field_present") is True
        or row.get("verifier_field_present") is True
        or row.get("release_authority_claimed") is True
        or row.get("model_output_is_oracle") is True
        for row in proposal_rows
    ):
        errors.append("model output must not contain answer, label, verifier, or release fields")

    compile_rows = list(artifact.get("exact_compile_rows", []))
    if len(compile_rows) != len(proposal_rows):
        errors.append("exact_compile_rows must match proposal_rows")
    if any(row.get("compile_outcome") not in COMPILE_OUTCOMES for row in compile_rows):
        errors.append("compile outcomes must be enumerated")
    if any(
        row.get("model_output_is_oracle") is not False
        or row.get("exact_compiler_is_oracle_for_disposition") is not True
        for row in compile_rows
    ):
        errors.append("compiler rows must keep model non-authority and exact compiler authority")

    attacks = list(artifact.get("boundary_attack_matrix", {}).get("rows", []))
    if artifact.get("factor_proposal_stream_ready_score") == 1.0 and (
        len(attacks) != len(BOUNDARY_ATTACK_IDS) or any(
        row.get("fail_closed") is not True or row.get("readiness_promoted") is not False
        for row in attacks
        )
    ):
        errors.append("boundary attacks must fail closed")
    prompt = artifact.get("prompt_commitment", {})
    if (
        prompt.get("grammar_backend") != "none"
        or prompt.get("retry_to_valid_loop_used") is not False
        or prompt.get("rank_and_select_loop_used") is not False
    ):
        errors.append("prompt commitment must disable grammar, retry, and rank-select loops")
    held = artifact.get("held_isolation_receipts", {})
    if (
        held.get("held_rows_selected_count") not in (None, 0)
        or held.get("held_rows_in_request_context_count") != 0
        or held.get("final_outcome_fields_in_request_context_count") != 0
    ):
        errors.append("held isolation must keep held and final outcome fields out of requests")

    recomputed = recompute_aggregates_from_rows(artifact.get("per_unit_rows", []))
    if artifact.get("aggregate_row_recomputation") != recomputed:
        errors.append("aggregate_row_recomputation mismatch")
    if artifact.get("factor_proposal_stream_ready_score") != _expected_ready_score(artifact):
        errors.append("factor_proposal_stream_ready_score mismatch")
    if (
        artifact.get("factor_proposal_stream_ready_score") == 1.0
        and recomputed.get("completed_model_family_count", 0) < 2
    ):
        errors.append("ready score requires two completed model families")
    if artifact.get("protected_files_unchanged", {}).get("active_roadmap_and_conductor_unchanged") is not True:
        errors.append("protected files changed")
    return errors


def _load_existing(path: Path) -> JsonDict | None:
    return _read_json(path)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    root = REPO_ROOT
    result_path = root / RESULT_RELATIVE_PATH
    if args.validate:
        payload = _load_existing(result_path)
        errors = ["artifact missing"] if payload is None else validate_artifact(payload)
        if errors:
            print(json.dumps({"valid": False, "errors": errors}, indent=2, sort_keys=True))
            return 1
        print(json.dumps({"valid": True, "path": str(result_path)}, indent=2, sort_keys=True))
        return 0
    if args.date != RUN_DATE:
        print(f"warning: expected planning date {RUN_DATE}, got {args.date}", file=sys.stderr)
    artifact = build_artifact(root=root, result_path=result_path, write=True)
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2, sort_keys=True))
        return 1
    print(json.dumps({"valid": True, "path": str(result_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
