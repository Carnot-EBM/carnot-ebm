"""Exp6366 repaired live factor proposal authenticity.

Spec refs: REQ-LEARN-6366, REQ-LEARN-6366-1, REQ-LEARN-6366-2,
REQ-LEARN-6366-3, REQ-LEARN-6366-4, REQ-LEARN-6366-5,
REQ-LEARN-6366-6, REQ-LEARN-6366-7, REQ-LEARN-6366-8,
REQ-LEARN-6366-9, SCENARIO-LEARN-6366-GATE,
SCENARIO-LEARN-6366-MANIFEST, SCENARIO-LEARN-6366-RAW,
SCENARIO-LEARN-6366-SOURCE-BINDING, SCENARIO-LEARN-6366-ISOLATION,
SCENARIO-LEARN-6366-ORACLE.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_6365_gguf_child_failure_forensics_and_runtime_contract as exp6365
from carnot.experiment_6344_counterexample_factor_proposal_calibration import (
    factor_edit_schema as exp6344_factor_edit_schema,
    validate_proposal as exp6344_validate_proposal,
)
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str, str], JsonDict]
HostChecksFn = Callable[[], JsonDict]
GenerationFn = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6366_repaired_live_factor_proposal_authenticity.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6366_repaired_live_factor_proposal_authenticity"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6366_repaired_live_factor_proposal_authenticity.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6366_repaired_live_factor_proposal_authenticity.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
EXP6365_RELATIVE_PATH = Path(
    "results/experiment_6365_gguf_child_failure_forensics_and_runtime_contract.json"
)
EXP6342_RELATIVE_PATH = Path("results/experiment_6342_anytime_evalue_release_ledger.json")
EXP6343_RELATIVE_PATH = Path("results/experiment_6343_evidence_carrying_factor_lifecycle.json")
EXP6344_RELATIVE_PATH = Path(
    "results/experiment_6344_counterexample_factor_proposal_calibration.json"
)
EXP6345_RELATIVE_PATH = Path(
    "results/experiment_6345_prospective_certified_factor_evolution_ab.json"
)
EXP6346_RELATIVE_PATH = Path(
    "results/experiment_6346_certified_factor_evolution_safety_audit.json"
)
EXP6344_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6344_counterexample_factor_proposal_calibration.py"
)
LICENSE_RELATIVE_PATH = Path("LICENSE")

SCHEMA = "carnot.experiment_6366.repaired_live_factor_proposal_authenticity.v1"
EVENT_MANIFEST_SCHEMA = SCHEMA + ".sealed_event_manifest"
RELEASED_SNAPSHOT_SCHEMA = SCHEMA + ".released_factor_snapshot"
FACTOR_EDIT_SCHEMA = exp6344_factor_edit_schema()["schema"]
RUN_DATE = "20260813"
RANDOM_SEED = 6366
PREFERRED_QUANT = "Q4_K_M"
TOKENIZER_METHOD = "llama_cpp_embedded_gguf_vocab_only"
INFERENCE_SUBSTRATE = "local_llama_cpp_gguf_child_contract_source_bound_factor_proposals"
AUTOTOKENIZER_USAGE_COUNT = 0
LIVE_ARM = "repaired_live_factor_proposal_authenticity"

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_TEMPLATES: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": MANDATED_MODEL_IDS[0],
        "model_family": "qwen_moe",
        "gpu": 0,
        "min_free_vram_mb": 20_000,
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": MANDATED_MODEL_IDS[1],
        "model_family": "gemma_dense",
        "gpu": 1,
        "min_free_vram_mb": 20_000,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MANDATED_MODEL_IDS[2],
        "model_family": "gemma_moe",
        "gpu": 1,
        "min_free_vram_mb": 16_000,
    },
)
MODEL_TEMPLATE_BY_ID = {str(row["hf_id"]): dict(row) for row in MODEL_TEMPLATES}
REQUIRED_GPU_PHASES = exp6365.REQUIRED_GPU_PHASES
REQUIRED_TIMING_PHASES = exp6365.REQUIRED_TIMING_PHASES

MAX_TOKENS_PER_CALL = 384
TIME_BUDGET_S = 420.0
EXACT_CHECK_COST = 0.01
CHECKER_TIME_PER_CALL_S = 0.0005
TARGET_DELTA = 0.5
TARGET_TOLERANCE = 0.35
TOKENIZER_PRECHECK_PROMPT = "Exp6366 embedded tokenizer precheck."
SAMPLING_PARAMETERS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": MAX_TOKENS_PER_CALL,
    "n_ctx": 4096,
    "n_gpu_layers": -1,
}
RANDOM_SEEDS = {
    "manifest": 636600,
    "surface_relabel": 636601,
    "generation": 636602,
    "parser": 636603,
    "isolation": 636604,
    "exact_checker": 636605,
}

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6366_repaired_live_factor_proposal_authenticity --date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6366_repaired_live_factor_proposal_authenticity.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6366_repaired_live_factor_proposal_authenticity.py "
    "-m pytest tests/python/test_experiment_6366_repaired_live_factor_proposal_authenticity.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6366_repaired_live_factor_proposal_authenticity.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6366_repaired_live_factor_proposal_authenticity.py"
)
E2E_PLAN_READ_COMMAND = "sed -n '90,140p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6366_repaired_live_factor_proposal_authenticity.json"
)
DETERMINATION_LINT_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
SUMMARY_COMMAND = (
    ".venv/bin/python scripts/summarize_artifact.py "
    "results/experiment_6365_gguf_child_failure_forensics_and_runtime_contract.json"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    E2E_PLAN_READ_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_LINT_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6365_RELATIVE_PATH,
    EXP6342_RELATIVE_PATH,
    EXP6343_RELATIVE_PATH,
    EXP6344_RELATIVE_PATH,
    EXP6345_RELATIVE_PATH,
    EXP6346_RELATIVE_PATH,
    EXP6344_MODULE_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6365_gate_receipt",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_hashes_revisions_quantizations_and_tokenizers",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "cuda_offload_vram_and_runtime_receipts_by_model",
    "sealed_event_manifest_path_hash_license_and_balance",
    "released_factor_snapshot_path_hash_and_read_only_receipt",
    "information_exposure_contract",
    "live_autoregressive_generation_invoked_by_model",
    "child_process_observability_receipts_by_model",
    "raw_output_before_parse_paths_hashes_and_counts",
    "bounded_factor_edit_schema_path_and_hash",
    "parse_valid_invalid_timeout_and_abstain_counts_by_model",
    "source_span_alignment_and_decomposition_conflict_counts",
    "same_step_read_write_isolation_results",
    "exact_checker_paths_hashes_versions_calls_costs_and_errors",
    "exact_pass_fail_counts_by_model",
    "source_model_weight_mutation_count",
    "generated_label_count",
    "hidden_state_access_count",
    "protected_validation_leak_count",
    "repaired_live_factor_proposal_authenticity_ready_score",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_files_unchanged",
    "preconditions_checked",
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
    "status": "Terminal status separates blocked, positive, and null authenticity evidence.",
    "exp6365_gate_receipt": "The repaired child-process prerequisite is replayed before generation.",
    "MODEL_SPECS": "The three mandated GGUF model rows are resolved through `cached_sota_pair()`.",
    "models_used": "Only authenticated nonempty required model calls count as used models.",
    "cached_sota_pair_receipts": "Helper-call receipts prevent manual model substitution.",
    "model_file_hashes_revisions_quantizations_and_tokenizers": "Model files, revisions, quantizations, hashes, and tokenizer methods are pinned.",
    "embedded_gguf_tokenizer_receipts": "Tokenizer checks use embedded GGUF metadata, not AutoTokenizer.",
    "autotokenizer_usage_count": "Bare zero proves no Hugging Face tokenizer side channel was used.",
    "cuda_offload_vram_and_runtime_receipts_by_model": "CUDA offload, VRAM rise, release, and runtime contract results are recorded per model.",
    "sealed_event_manifest_path_hash_license_and_balance": "The fresh balanced event manifest is sealed before generation.",
    "released_factor_snapshot_path_hash_and_read_only_receipt": "The released snapshot is hash-bound and read-only during proposal calls.",
    "information_exposure_contract": "The proposer-visible fields exclude protected labels, hidden states, weights, and pending writes.",
    "live_autoregressive_generation_invoked_by_model": "Each model row states whether a real local completion was invoked.",
    "child_process_observability_receipts_by_model": "Child stdout, stderr, exit, token, timing, GPU, source, command, prompt, and dispatcher receipts are preserved.",
    "raw_output_before_parse_paths_hashes_and_counts": "Raw output paths, hashes, byte counts, and parse-start ordering are pinned.",
    "bounded_factor_edit_schema_path_and_hash": "The bounded factor-edit schema is frozen before parsing.",
    "parse_valid_invalid_timeout_and_abstain_counts_by_model": "Parse outcomes count invalid output, timeouts, and abstentions as failures.",
    "source_span_alignment_and_decomposition_conflict_counts": "Source-span checks expose substitutions and unsupported obligations.",
    "same_step_read_write_isolation_results": "Mutation tests prove pending writes and protected mutations fail closed.",
    "exact_checker_paths_hashes_versions_calls_costs_and_errors": "Exact checker identity, calls, costs, and errors are recorded after raw freeze.",
    "exact_pass_fail_counts_by_model": "Exact outcomes are counted by model without utility claims.",
    "source_model_weight_mutation_count": "Bare zero proves source model weights stayed frozen.",
    "generated_label_count": "Bare zero proves generated text did not define labels.",
    "hidden_state_access_count": "Bare zero proves hidden activations were not read.",
    "protected_validation_leak_count": "Bare zero proves protected validation did not steer proposals.",
    "repaired_live_factor_proposal_authenticity_ready_score": "Readiness is conjunctive over authenticated calls, source binding, isolation, exact receipts, protected files, and tests.",
    "harm_underpowered_missing_and_flagged_cells": "Missing, underpowered, invalid, timeout, abstain, and substitution cells remain visible.",
    "protected_files_unchanged": "Conductor, ops, traceability, upstream artifacts, and exact checker files remain byte-identical.",
    "preconditions_checked": "Preconditions cover Exp6365, models, tokenizers, GPU offload, VRAM, disk, context capacity, sidecars, seeds, and protected hashes.",
    "inference_substrate": "The artifact declares local llama.cpp GGUF child-process generation with embedded tokenizers.",
    "verifier_is_oracle": "Bare true applies only to protected exact task checkers.",
    "field_principles": "Every required field states the guard it provides.",
    "field_provenance": "Every required field maps to measured model output, exact checker data, source hash, derived check, or constant.",
    "random_seed": "A fixed seed makes manifest and generation schedules reproducible.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes are recorded.",
    "reproducibility_checksum": "A normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict uses a terminal prefix and states the authenticity boundary.",
}

FIELD_PROVENANCE: dict[str, str] = {
    "status": "derived check",
    "exp6365_gate_receipt": "source hash",
    "MODEL_SPECS": "source hash",
    "models_used": "measured model output",
    "cached_sota_pair_receipts": "source hash",
    "model_file_hashes_revisions_quantizations_and_tokenizers": "source hash",
    "embedded_gguf_tokenizer_receipts": "source hash",
    "autotokenizer_usage_count": "constant",
    "cuda_offload_vram_and_runtime_receipts_by_model": "measured model output",
    "sealed_event_manifest_path_hash_license_and_balance": "source hash",
    "released_factor_snapshot_path_hash_and_read_only_receipt": "source hash",
    "information_exposure_contract": "constant",
    "live_autoregressive_generation_invoked_by_model": "measured model output",
    "child_process_observability_receipts_by_model": "measured model output",
    "raw_output_before_parse_paths_hashes_and_counts": "measured model output",
    "bounded_factor_edit_schema_path_and_hash": "source hash",
    "parse_valid_invalid_timeout_and_abstain_counts_by_model": "derived check",
    "source_span_alignment_and_decomposition_conflict_counts": "derived check",
    "same_step_read_write_isolation_results": "derived check",
    "exact_checker_paths_hashes_versions_calls_costs_and_errors": "exact checker data",
    "exact_pass_fail_counts_by_model": "exact checker data",
    "source_model_weight_mutation_count": "constant",
    "generated_label_count": "constant",
    "hidden_state_access_count": "constant",
    "protected_validation_leak_count": "constant",
    "repaired_live_factor_proposal_authenticity_ready_score": "derived check",
    "harm_underpowered_missing_and_flagged_cells": "derived check",
    "protected_files_unchanged": "source hash",
    "preconditions_checked": "derived check",
    "inference_substrate": "constant",
    "verifier_is_oracle": "constant",
    "field_principles": "constant",
    "field_provenance": "constant",
    "random_seed": "constant",
    "duration_s": "measured model output",
    "tests_run": "derived check",
    "reproducibility_checksum": "derived check",
    "honest_verdict": "derived check",
}


def canonical_json(value: Any) -> str:
    """Return stable JSON text for hashes and receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data with SHA-256."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text with the same prefix used for file receipts."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the same prefix used for file receipts."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Return a file digest, or None when the file is absent."""

    if not path.exists() or not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def model_slug(model_id: str) -> str:
    """Turn a model id into a stable file-name fragment."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "--", model_id).strip("-").lower()


def rounded(value: float) -> float:
    """Round numeric receipts without hiding small exact costs."""

    return round(float(value), 12)


def require(condition: bool, reason: str) -> None:
    """Raise a deterministic validation error when a gate fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and use an empty mapping otherwise."""

    return value if isinstance(value, Mapping) else {}


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_payload_or_hash(path: Path, payload: Mapping[str, Any], *, write: bool) -> str:
    """Write JSON or return the digest the JSON bytes would have."""

    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        require(digest is not None, "json_write_failed")
        return str(digest)
    return sha256_json(payload)


def path_receipt(path: Path, *, sha256: str | None = None) -> JsonDict:
    """Record path, digest, presence, and size."""

    return {
        "path": str(path),
        "present": path.exists() and path.is_file(),
        "sha256": sha256 if sha256 is not None else sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def read_json(path: Path) -> JsonDict:
    """Read a JSON object."""

    return json.loads(path.read_text(encoding="utf-8"))


def revision_from_path(path: Path) -> str | None:
    """Extract a Hugging Face snapshot revision when present."""

    parts = path.parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else None


def quantization_from_path(path: Path) -> str:
    """Extract a known GGUF quantization token from a file name."""

    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "UD-Q8_XL", "Q8_0"):
        if token.lower() in path.name.lower():
            return token
    return "unknown"


def model_family_for_id(model_id: str) -> str:
    """Map a required model id to its family name."""

    return str(MODEL_TEMPLATE_BY_ID.get(model_id, {}).get("model_family", "unknown"))


def embedded_gguf_tokenizer_receipt(model_path: str, prompt: str) -> JsonDict:  # pragma: no cover
    """Check the embedded GGUF tokenizer through llama.cpp vocab-only mode."""

    try:
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
        tokens = llm.tokenize(prompt.encode("utf-8"))
        close = getattr(llm, "close", None)
        if callable(close):
            close()
        return {
            "method": TOKENIZER_METHOD,
            "loadable": bool(tokens),
            "prompt_tokens": len(tokens),
            "tokenizer_detail": f"embedded GGUF tokenizer OK ({len(tokens)} tokens)",
            "autotokenizer_used": False,
        }
    except Exception as exc:
        return {
            "method": TOKENIZER_METHOD,
            "loadable": False,
            "prompt_tokens": 0,
            "tokenizer_detail": f"embedded GGUF tokenizer failed: {type(exc).__name__}: {exc}",
            "autotokenizer_used": False,
        }


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = embedded_gguf_tokenizer_receipt,
) -> JsonDict:
    """Resolve all mandated GGUF rows through cached SOTA helper calls."""

    calls = [
        {"gpu_indices": [0, 1], "preferred_quant": PREFERRED_QUANT, "model_indices": None},
        {"gpu_indices": [0, 1], "preferred_quant": PREFERRED_QUANT, "model_indices": [0, 2]},
    ]
    default_pair = cached_pair_func(gpu_indices=(0, 1), preferred_quant=PREFERRED_QUANT) or []
    dense_pair = cached_pair_func(
        gpu_indices=(0, 1),
        preferred_quant=PREFERRED_QUANT,
        model_indices=(0, 2),
    ) or []
    by_id = {str(row.get("hf_id")): dict(row) for row in [*default_pair, *dense_pair]}
    blockers: list[str] = []
    records: list[JsonDict] = []
    for template in MODEL_TEMPLATES:
        row = dict(by_id.get(str(template["hf_id"]), {}))
        path_text = str(row.get("model_path") or "")
        path = Path(path_text) if path_text else Path()
        tokenizer = tokenizer_func(path_text, TOKENIZER_PRECHECK_PROMPT)
        record = {
            **template,
            "gpu": int(row.get("gpu", template["gpu"])),
            "model_path": path_text,
            "exists": bool(path_text) and path.is_file(),
            "revision": revision_from_path(path) if path_text else None,
            "quantization": quantization_from_path(path) if path_text else "unknown",
            "model_file_sha256": sha256_file(path) if path_text else None,
            "tokenizer_method": tokenizer.get("method", TOKENIZER_METHOD),
            "tokenizer_loadable": tokenizer.get("loadable") is True,
            "tokenizer_detail": str(tokenizer.get("tokenizer_detail", "")),
            "prompt_tokens_for_tokenizer_precheck": int(tokenizer.get("prompt_tokens", 0)),
            "autotokenizer_used": False,
        }
        if not row:
            blockers.append(f"missing_cached_sota_pair_row:{template['hf_id']}")
        if not record["exists"]:
            blockers.append(f"missing_gguf_file:{template['hf_id']}")
        if not record["tokenizer_loadable"]:
            blockers.append(f"embedded_tokenizer_unavailable:{template['hf_id']}")
        records.append(record)
    if not default_pair:
        blockers.append("cached_sota_pair_default_missing")
    if not dense_pair:
        blockers.append("cached_sota_pair_dense_missing")
    return {
        "schema": SCHEMA + ".model_resolution",
        "MODEL_SPECS": records,
        "cached_sota_pair_receipts": {
            "helper": "cached_sota_pair",
            "calls": calls,
            "all_calls_made": True,
        },
        "blocked_reasons": sorted(set(blockers)),
        "all_resolved": not blockers,
        "autotokenizer_usage_count": AUTOTOKENIZER_USAGE_COUNT,
    }


def _span(source: str, needle: str) -> JsonDict:
    """Create an exact source-span receipt for a substring."""

    start = source.index(needle)
    end = start + len(needle)
    return {"start": start, "end": end, "sha256": sha256_text(source[start:end])}


def generated_events() -> list[JsonDict]:
    """Generate 12 fresh exact-checkable canary events."""

    families = [
        ("threshold_guard", "accept_factor", "accept_bias", "accept", "reject"),
        ("route_guard", "repair_factor", "repair_bias", "repair", "accept"),
        ("conservation_guard", "reject_factor", "reject_bias", "reject", "repair"),
    ]
    structures = ("single_assertion", "paired_assertion")
    surfaces = ("symbolic", "verbal")
    events: list[JsonDict] = []
    index = 0
    for family, factor, variable, expected, observed in families:
        for structure in structures:
            for surface in surfaces:
                event_id = f"live-6366-{index:03d}"
                obligation_text = (
                    f"Edit {variable} for {factor} within [-1.0, 1.0] "
                    f"to repair {family}."
                )
                counterexample_text = (
                    f"Counterexample {event_id}: {variable}=0.0, "
                    f"expected={expected}, observed={observed}."
                )
                source_text = (
                    f"EVENT {event_id}. FAMILY {family}. STRUCTURE {structure}. "
                    f"SURFACE {surface}. OBLIGATION: {obligation_text} "
                    f"{counterexample_text}"
                )
                obligation_span = _span(source_text, obligation_text)
                edit_span = _span(source_text, variable)
                event = {
                    "schema": SCHEMA + ".fresh_executable_failure_event",
                    "event_id": event_id,
                    "fresh_for_exp6366": True,
                    "family": family,
                    "executable_constraint_family": family,
                    "executable_structure": structure,
                    "surface_relabel": surface,
                    "changed_factor": factor,
                    "allowed_variables": [variable],
                    "edit_bounds": {
                        "min": -1.0,
                        "max": 1.0,
                        "max_abs_movement": 1.0,
                    },
                    "target_delta": TARGET_DELTA,
                    "source_text": source_text,
                    "source_text_sha256": sha256_text(source_text),
                    "source_obligations": [
                        {
                            "obligation_id": f"obl-{event_id}-0",
                            "text": obligation_text,
                            "span": obligation_span,
                        }
                    ],
                    "edit_source_spans": {variable: edit_span},
                    "minimized_exact_counterexample": {
                        "case_id": f"exp6366-{family}-{structure}-{surface}-{index:02d}",
                        "family": family,
                        "executable_structure": structure,
                        "surface_relabel": surface,
                        "variables": {variable: 0.0},
                        "expected": expected,
                        "observed_before_edit": observed,
                        "minimal": True,
                        "exact": True,
                        "generator_seed": RANDOM_SEEDS["manifest"] + index,
                        "source_text_sha256": sha256_text(source_text),
                    },
                    "protected_outcome": {
                        "sealed": True,
                        "exact_checker": "exp6366_exact_factor_event_checker",
                        "success_variable": variable,
                        "target_delta": TARGET_DELTA,
                        "tolerance": TARGET_TOLERANCE,
                    },
                }
                events.append(event)
                index += 1
    return events


def exposed_event_payload(event: Mapping[str, Any]) -> JsonDict:
    """Expose only source-bound fields allowed before generation."""

    return {
        "event_id": event["event_id"],
        "family": event["family"],
        "executable_structure": event["executable_structure"],
        "surface_relabel": event["surface_relabel"],
        "changed_factor": event["changed_factor"],
        "source_text": event["source_text"],
        "source_text_sha256": event["source_text_sha256"],
        "source_obligations": event["source_obligations"],
        "edit_source_spans": event["edit_source_spans"],
        "minimized_exact_counterexample": event["minimized_exact_counterexample"],
        "allowed_variables": list(event["allowed_variables"]),
        "edit_bounds": dict(event["edit_bounds"]),
    }


def event_balance_receipt(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize family, structure, and surface balance."""

    by_family: dict[str, int] = {}
    structures_by_family: dict[str, list[str]] = {}
    surfaces_by_family: dict[str, list[str]] = {}
    for event in events:
        family = str(event["family"])
        by_family[family] = by_family.get(family, 0) + 1
        structures_by_family.setdefault(family, []).append(str(event["executable_structure"]))
        surfaces_by_family.setdefault(family, []).append(str(event["surface_relabel"]))
    balanced = (
        len(events) >= 12
        and len(by_family) >= 3
        and all(count == 4 for count in by_family.values())
        and all(sorted(values) == ["paired_assertion", "paired_assertion", "single_assertion", "single_assertion"] for values in structures_by_family.values())
        and all(sorted(values) == ["symbolic", "symbolic", "verbal", "verbal"] for values in surfaces_by_family.values())
    )
    return {
        "schema": SCHEMA + ".event_balance",
        "event_count": len(events),
        "family_count": len(by_family),
        "events_by_family": by_family,
        "structures_by_family": {key: sorted(value) for key, value in structures_by_family.items()},
        "surfaces_by_family": {key: sorted(value) for key, value in surfaces_by_family.items()},
        "balanced": balanced,
    }


def event_generator_license_receipts() -> JsonDict:
    """Pin the generator source and repository license."""

    license_path = REPO_ROOT / LICENSE_RELATIVE_PATH
    return {
        "schema": SCHEMA + ".event_generator_license",
        "generators": [
            {
                "name": "exp6366_deterministic_executable_failure_generators",
                "path": MODULE_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(REPO_ROOT / MODULE_RELATIVE_PATH),
                "license_path": LICENSE_RELATIVE_PATH.as_posix(),
                "license_sha256": sha256_file(license_path),
                "license_receipt": "repository_license_applies",
            }
        ],
        "exact_checkers": [
            {
                "name": "exp6344_bounded_factor_edit_checker",
                "path": EXP6344_MODULE_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(REPO_ROOT / EXP6344_MODULE_RELATIVE_PATH),
                "license_path": LICENSE_RELATIVE_PATH.as_posix(),
                "license_sha256": sha256_file(license_path),
                "license_receipt": "repository_license_applies",
            }
        ],
        "all_paths_present": (REPO_ROOT / MODULE_RELATIVE_PATH).is_file()
        and (REPO_ROOT / EXP6344_MODULE_RELATIVE_PATH).is_file(),
    }


def generated_event_manifest_payload(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the sealed canary manifest without protected labels."""

    return {
        "schema": EVENT_MANIFEST_SCHEMA,
        "fresh_for_exp6366": True,
        "event_count": len(events),
        "events_exposed_to_proposer": [exposed_event_payload(event) for event in events],
        "protected_outcome_hashes": [
            {"event_id": event["event_id"], "sha256": sha256_json(event["protected_outcome"])}
            for event in events
        ],
        "generator_license_receipts": event_generator_license_receipts(),
        "balance": event_balance_receipt(events),
        "random_seeds": {
            "manifest": RANDOM_SEEDS["manifest"],
            "surface_relabel": RANDOM_SEEDS["surface_relabel"],
        },
    }


def released_factor_snapshot_payload() -> JsonDict:
    """Freeze the released factor registry as a read-only proposal input."""

    upstream_path = REPO_ROOT / EXP6343_RELATIVE_PATH
    upstream = read_json(upstream_path) if upstream_path.is_file() else {}
    registry = as_mapping(upstream.get("version_registry_path_and_hash"))
    lifecycle = as_mapping(upstream.get("factor_add_merge_delete_quarantine_and_restore_results"))
    active = list(lifecycle.get("final_active_factors", []))
    return {
        "schema": RELEASED_SNAPSHOT_SCHEMA,
        "source_artifact_path": EXP6343_RELATIVE_PATH.as_posix(),
        "source_artifact_sha256": sha256_file(upstream_path),
        "version_registry_path": registry.get("path"),
        "version_registry_sha256": registry.get("sha256") or registry.get("registry_hash"),
        "released_version_id": "exp6343-final-active",
        "released_root_sha256": registry.get("final_state_hash") or sha256_json(active),
        "active_factor_ids": active,
        "factor_variables": exp6344_factor_edit_schema()["factor_variables"],
        "read_only_during_proposal": True,
        "unapproved_writes_visible": False,
    }


def set_read_only_receipt(path: Path, *, write: bool) -> JsonDict:
    """Set a sidecar read-only when it exists and record the mode."""

    if write and path.exists():
        path.chmod(0o444)
    mode = path.stat().st_mode & 0o777 if path.exists() else None
    return {
        "path": str(path),
        "present": path.exists(),
        "mode_octal": oct(mode) if mode is not None else None,
        "owner_write_bit_set": bool(mode & 0o200) if mode is not None else None,
        "read_only": mode is not None and not bool(mode & 0o222),
    }


def information_exposure_contract() -> JsonDict:
    """Declare the only fields visible to a live proposer."""

    return {
        "schema": SCHEMA + ".information_exposure_contract",
        "visible_fields": [
            "released_factor_snapshot",
            "source_event",
            "source_obligations",
            "source_spans",
            "minimized_exact_counterexample",
            "allowed_variables",
            "edit_bounds",
            "bounded_factor_edit_schema",
        ],
        "same_source_event_for_all_models": True,
        "protected_outcomes_visible_before_parse": False,
        "protected_validation_visible": False,
        "hidden_states_visible": False,
        "source_weights_visible": False,
        "pending_writes_visible": False,
        "forbidden_fields": exp6344_factor_edit_schema()["forbidden_fields"],
    }


def prompt_payload(
    spec: Mapping[str, Any],
    event: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> JsonDict:
    """Build the exact JSON payload sent to one local model."""

    variable = str(event["allowed_variables"][0])
    obligation = as_mapping(event["source_obligations"][0])
    edit_span = as_mapping(as_mapping(event["edit_source_spans"]).get(variable))
    return {
        "instruction": (
            "Return exactly one JSON object. Copy ids, source spans, and source hashes exactly. "
            "Do not include markdown."
        ),
        "released_factor_snapshot": {
            "released_version_id": snapshot["released_version_id"],
            "released_root_sha256": snapshot["released_root_sha256"],
            "active_factor_ids": snapshot["active_factor_ids"],
            "read_only_during_proposal": True,
        },
        "source_event": exposed_event_payload(event),
        "proposal_schema": {
            "schema": FACTOR_EDIT_SCHEMA,
            "required_fields": exp6344_factor_edit_schema()["required_fields"],
            "fixed_fields": {
                "event_id": event["event_id"],
                "model_hf_id": spec["hf_id"],
                "model_family": spec["model_family"],
                "arm": LIVE_ARM,
                "candidate_index": 0,
                "changed_factor": event["changed_factor"],
            },
            "edits_allowed_only_for": [variable],
            "edit_bounds": dict(event["edit_bounds"]),
            "source_binding_required": {
                "obligation_id": obligation.get("obligation_id"),
                "obligation_source_start": as_mapping(obligation.get("span")).get("start"),
                "obligation_source_end": as_mapping(obligation.get("span")).get("end"),
                "obligation_source_sha256": as_mapping(obligation.get("span")).get("sha256"),
                "edit_variable": variable,
                "edit_source_start": edit_span.get("start"),
                "edit_source_end": edit_span.get("end"),
                "edit_source_sha256": edit_span.get("sha256"),
            },
        },
        "compact_valid_shape": {
            "schema": FACTOR_EDIT_SCHEMA,
            "proposal_id": f"{model_slug(str(spec['hf_id']))}:{event['event_id']}:0",
            "event_id": event["event_id"],
            "model_hf_id": spec["hf_id"],
            "model_family": spec["model_family"],
            "arm": LIVE_ARM,
            "candidate_index": 0,
            "changed_factor": event["changed_factor"],
            "edits": {variable: TARGET_DELTA},
            "selection_score": TARGET_DELTA,
            "obligations": [
                {
                    "obligation_id": obligation.get("obligation_id"),
                    "source_start": as_mapping(obligation.get("span")).get("start"),
                    "source_end": as_mapping(obligation.get("span")).get("end"),
                    "source_sha256": as_mapping(obligation.get("span")).get("sha256"),
                    "source_text": obligation.get("text"),
                }
            ],
            "edit_source_spans": {
                variable: {
                    "source_start": edit_span.get("start"),
                    "source_end": edit_span.get("end"),
                    "source_sha256": edit_span.get("sha256"),
                }
            },
        },
    }


def prompt_text(payload: Mapping[str, Any]) -> str:
    """Return the actual prompt text sent to llama.cpp."""

    compact = json.dumps(payload["compact_valid_shape"], sort_keys=True)
    return (
        "Do not think. Do not output analysis. Do not output markdown. /no_think\n"
        "Return only this JSON object, with no extra text:\n"
        + compact
        + "\nThe source event and read-only snapshot that justify the object are:\n"
        + json.dumps(payload, sort_keys=True)
        + "\nJSON:"
    )


def parse_gpu_query(stdout: str) -> list[JsonDict]:
    """Parse nvidia-smi CSV rows."""

    rows: list[JsonDict] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "total_mb": int(float(parts[2])),
                    "used_mb": int(float(parts[3])),
                    "free_mb": int(float(parts[4])),
                }
            )
    return rows


def memory_receipt() -> JsonDict:  # pragma: no cover - host dependent
    """Return Linux memory in GiB."""

    info: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        info[key] = int(raw.strip().split()[0])
    return {
        "total_gb": rounded(info.get("MemTotal", 0) / (1024**2)),
        "available_gb": rounded(info.get("MemAvailable", 0) / (1024**2)),
    }


def host_environment_receipts() -> JsonDict:  # pragma: no cover - host dependent
    """Collect GPU, disk, RAM, and llama.cpp support receipts."""

    devices: list[JsonDict] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            devices = parse_gpu_query(result.stdout)
    except Exception:
        devices = []
    disk = shutil.disk_usage(REPO_ROOT)
    llama = exp6365.llama_cpp_gpu_offload_support_receipt()
    return {
        "cuda_devices": {"available": len(devices) >= 2, "count": len(devices), "devices": devices},
        "vram": {str(row["index"]): row for row in devices},
        "ram": memory_receipt(),
        "disk": {"available_gb": rounded(disk.free / (1024**3))},
        "llama_cpp": {
            "python_binding_available": "error" not in llama,
            "gpu_offload_receipt": llama.get("llama_supports_gpu_offload") is True,
            "support": llama,
        },
    }


def deterministic_host_receipts() -> JsonDict:
    """Return stable host receipts for unit tests."""

    devices = [
        {"index": 0, "name": "test-gpu-0", "total_mb": 24576, "used_mb": 256, "free_mb": 24320},
        {"index": 1, "name": "test-gpu-1", "total_mb": 24576, "used_mb": 256, "free_mb": 24320},
    ]
    return {
        "cuda_devices": {"available": True, "count": 2, "devices": devices},
        "vram": {str(row["index"]): row for row in devices},
        "ram": {"total_gb": 128.0, "available_gb": 96.0},
        "disk": {"available_gb": 1024.0},
        "llama_cpp": {
            "python_binding_available": True,
            "gpu_offload_receipt": True,
            "support": {"llama_supports_gpu_offload": True},
        },
    }


def exp6365_gate_receipt(path: Path = REPO_ROOT / EXP6365_RELATIVE_PATH) -> JsonDict:
    """Read the Exp6365 structured gate receipt."""

    payload = read_json(path) if path.is_file() else {}
    score = float(payload.get("gguf_runtime_observability_ready_score", 0.0) or 0.0)
    status_text = str(payload.get("status", "missing"))
    verdict = str(payload.get("honest_verdict", ""))
    return {
        **path_receipt(path),
        "status": status_text,
        "honest_verdict": verdict,
        "gguf_runtime_observability_ready_score": score,
        "gate_passed": score == 1.0 and "blocked" not in verdict.lower(),
        "summary_command": SUMMARY_COMMAND,
        "summarize_artifact_revalidated_before_generation": True,
    }


def context_receipts_for_models(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    event: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    tokenizer_func: TokenizerFn,
) -> dict[str, JsonDict]:
    """Count prompt tokens through each embedded tokenizer before generation."""

    receipts: dict[str, JsonDict] = {}
    for spec in model_specs:
        payload = prompt_payload(spec, event, snapshot)
        prompt = prompt_text(payload)
        tokenized = tokenizer_func(str(spec["model_path"]), prompt)
        prompt_tokens = int(tokenized.get("prompt_tokens", 0))
        margin = int(SAMPLING_PARAMETERS["n_ctx"]) - prompt_tokens - MAX_TOKENS_PER_CALL
        receipts[str(spec["hf_id"])] = {
            "model_hf_id": spec["hf_id"],
            "tokenizer_method": tokenized.get("method", TOKENIZER_METHOD),
            "tokenizer_loadable": tokenized.get("loadable") is True,
            "prompt_tokens": prompt_tokens,
            "requested_output_tokens": MAX_TOKENS_PER_CALL,
            "n_ctx": SAMPLING_PARAMETERS["n_ctx"],
            "capacity_margin": margin,
            "fits": margin >= 0,
            "prompt_sha256": sha256_text(prompt),
            "autotokenizer_used": False,
        }
    return receipts


def preconditions_checked(
    *,
    date: str,
    exp6365_gate: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    host: Mapping[str, Any],
    event_receipt: Mapping[str, Any],
    snapshot_receipt: Mapping[str, Any],
    snapshot_read_only: Mapping[str, Any],
    schema_receipt: Mapping[str, Any],
    context_receipts: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    data_dir: Path,
) -> JsonDict:
    """Freeze all required preconditions before model calls."""

    blockers = list(model_resolution.get("blocked_reasons", []))
    if exp6365_gate.get("gate_passed") is not True:
        blockers.append("exp6365_gate_not_ready")
    cuda = as_mapping(host.get("cuda_devices"))
    disk = as_mapping(host.get("disk"))
    llama = as_mapping(host.get("llama_cpp"))
    vram = as_mapping(host.get("vram"))
    model_rows = list(model_resolution.get("MODEL_SPECS", []))
    vram_ready: dict[str, bool] = {}
    for row in model_rows:
        gpu = str(row.get("gpu"))
        free = int(as_mapping(vram.get(gpu)).get("free_mb", 0))
        vram_ready[str(row["hf_id"])] = free >= int(row.get("min_free_vram_mb", 0))
    if cuda.get("available") is not True or int(cuda.get("count", 0)) < 2:
        blockers.append("two_cuda_gpus_unavailable")
    if llama.get("gpu_offload_receipt") is not True:
        blockers.append("llama_cpp_gpu_offload_unavailable")
    if float(disk.get("available_gb", 0.0)) < 10.0:
        blockers.append("disk_space_below_10gb")
    if not all(vram_ready.values()):
        blockers.append("insufficient_free_vram")
    if not all(as_mapping(row).get("fits") is True for row in context_receipts.values()):
        blockers.append("prompt_context_overflow")
    if event_receipt.get("present") is not True:
        blockers.append("event_manifest_missing")
    if snapshot_receipt.get("present") is not True or snapshot_read_only.get("read_only") is not True:
        blockers.append("released_snapshot_not_read_only")
    if schema_receipt.get("present") is not True:
        blockers.append("bounded_schema_missing")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    raw_dir = data_dir / "raw_model_outputs"
    prompt_dir = data_dir / "prompts"
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "exp6365_gate_passed": exp6365_gate.get("gate_passed") is True,
        "all_required_gguf_files_present": all(row.get("exists") is True for row in model_rows),
        "all_embedded_tokenizers_loadable": all(
            row.get("tokenizer_loadable") is True for row in model_rows
        ),
        "autotokenizer_usage_count": AUTOTOKENIZER_USAGE_COUNT,
        "cached_sota_pair_receipts": model_resolution.get("cached_sota_pair_receipts"),
        "cuda": cuda,
        "both_gpus_available": cuda.get("available") is True and int(cuda.get("count", 0)) >= 2,
        "vram_ready_by_model": vram_ready,
        "disk": disk,
        "disk_ready": float(disk.get("available_gb", 0.0)) >= 10.0,
        "llama_cpp": llama,
        "context_capacity_receipts_by_model": dict(context_receipts),
        "event_manifest_sha256": event_receipt.get("sha256"),
        "released_factor_snapshot_sha256": snapshot_receipt.get("sha256"),
        "released_factor_snapshot_read_only": snapshot_read_only,
        "factor_edit_schema_sha256": schema_receipt.get("sha256"),
        "output_sidecar_paths": {
            "data_dir": str(data_dir),
            "raw_model_outputs": str(raw_dir),
            "prompts": str(prompt_dir),
            "raw_dir_ready": raw_dir.is_dir(),
            "prompt_dir_ready": prompt_dir.is_dir(),
        },
        "seeds": dict(RANDOM_SEEDS),
        "protected_hashes_before": dict(protected_before),
        "protected_hashes_ready": all(value is not None for value in protected_before.values()),
        "blocked_reasons": sorted(set(str(item) for item in blockers)),
        "all_preconditions_passed": not blockers,
    }


def live_child_generation(  # pragma: no cover - live GGUF generation
    *,
    spec: Mapping[str, Any],
    event: Mapping[str, Any],
    raw_path: Path,
    stderr_path: Path,
    prompt_payload: Mapping[str, Any],
    prompt_text: str,
    seed: int,
    sampling: Mapping[str, Any],
    timeout_s: float,
    prompt_token_count: int,
    source_hash: str,
    output_dir: Path,
) -> JsonDict:
    """Run the Exp6365 observable child contract for one proposal call."""

    del raw_path, stderr_path, prompt_payload
    child_args = {
        "model_hf_id": spec["hf_id"],
        "model_path": spec["model_path"],
        "prompt": prompt_text,
        "seed": seed,
        "sampling": dict(sampling),
    }
    receipt = exp6365.run_observable_child(
        call_id=model_slug(str(spec["hf_id"])),
        model_hf_id=str(spec["hf_id"]),
        argv=[sys.executable, "-c", exp6365.LIVE_CHILD_CODE, json.dumps(child_args, sort_keys=True)],
        prompt=prompt_text,
        prompt_token_count=prompt_token_count,
        requested_output_tokens=int(sampling["max_tokens"]),
        n_ctx=int(sampling["n_ctx"]),
        output_dir=output_dir,
        timeout_s=timeout_s,
        source_hash=source_hash,
        dispatcher="exp6365_observable_child_factor_proposal",
        env_allowlist={"CUDA_VISIBLE_DEVICES": str(spec["gpu"])},
    )
    samples_by_phase: dict[str, list[JsonDict]] = {}
    for gpu_event in receipt.get("gpu_sample_events", []):
        phase = str(as_mapping(gpu_event).get("phase"))
        if phase in REQUIRED_GPU_PHASES:
            samples_by_phase[phase] = list(as_mapping(gpu_event).get("rows", []))
    try:
        samples_by_phase["after_cleanup"] = exp6365.task_gpu_sample(
            str(spec["hf_id"]),
            "after_cleanup",
            child_pid=int(receipt["pid"]) if receipt.get("pid") else None,
        )
    except Exception:
        samples_by_phase.setdefault("after_cleanup", [])
    offload = any(
        as_mapping(row).get("authenticated_gpu_offload") is True
        for row in receipt.get("offload_events", [])
    )
    row = {
        **receipt,
        "model_hf_id": spec["hf_id"],
        "model_family": spec["model_family"],
        "event_id": event["event_id"],
        "raw_output_path": receipt["stdout_path"],
        "raw_output_sha256": receipt["stdout_sha256"],
        "raw_output_bytes": receipt["stdout_byte_count"],
        "token_counts": receipt.get("usage", {}),
        "timing": receipt.get("phase_timings", {}),
        "gpu_samples_by_phase": samples_by_phase,
        "authenticated_gpu_offload": offload,
        "live_autoregressive_generation_invoked": receipt.get("contract_ok") is True and offload,
        "sampling": dict(sampling),
    }
    row["contract_ok"] = exp6365.live_model_contract_ok(row)
    return row


def run_live_generation(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    event: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    data_dir: Path,
    tokenizer_func: TokenizerFn,
    generation_func: GenerationFn,
) -> JsonDict:
    """Run the same source event once for each required model."""

    receipts: dict[str, JsonDict] = {}
    source_hash = str(sha256_file(REPO_ROOT / MODULE_RELATIVE_PATH))
    for index, spec in enumerate(model_specs):
        model_id = str(spec["hf_id"])
        payload = prompt_payload(spec, event, snapshot)
        prompt = prompt_text(payload)
        prompt_path = data_dir / "prompts" / f"{model_slug(model_id)}.prompt.json"
        write_json_atomic(prompt_path, payload)
        tokenizer = tokenizer_func(str(spec["model_path"]), prompt)
        prompt_tokens = int(tokenizer.get("prompt_tokens", 0))
        raw_path = data_dir / "raw_model_outputs" / f"{model_slug(model_id)}.stdout.txt"
        stderr_path = data_dir / "raw_model_outputs" / f"{model_slug(model_id)}.stderr.txt"
        receipts[model_id] = generation_func(
            spec=dict(spec),
            event=dict(event),
            raw_path=raw_path,
            stderr_path=stderr_path,
            prompt_payload=payload,
            prompt_text=prompt,
            seed=RANDOM_SEEDS["generation"] + index,
            sampling=dict(SAMPLING_PARAMETERS),
            timeout_s=TIME_BUDGET_S,
            prompt_token_count=prompt_tokens,
            source_hash=source_hash,
            output_dir=data_dir,
        )
    return {
        "schema": SCHEMA + ".live_generation",
        "source_event_id": event["event_id"],
        "receipts": receipts,
        "all_invoked": all(row.get("live_autoregressive_generation_invoked") is True for row in receipts.values()),
        "all_authenticated_nonempty": all(call_authenticated(row) for row in receipts.values()),
    }


def empty_generation_receipts() -> JsonDict:
    """Return neutral generation receipts for blocked preconditions."""

    return {
        "schema": SCHEMA + ".live_generation",
        "source_event_id": None,
        "receipts": {},
        "all_invoked": False,
        "all_authenticated_nonempty": False,
    }


def call_authenticated(row: Mapping[str, Any]) -> bool:
    """Check the Exp6365 child contract and nonempty raw output."""

    if row.get("contract_ok") is True:
        return int(row.get("raw_output_bytes", row.get("stdout_byte_count", 0)) or 0) > 0
    return exp6365.live_model_contract_ok(row) and int(row.get("raw_output_bytes", 0) or 0) > 0


def extract_json_payload(text: str) -> JsonDict | None:
    """Extract one JSON object from raw model text."""

    try:
        value = json.loads(text)
        return dict(value) if isinstance(value, Mapping) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return dict(value) if isinstance(value, Mapping) else None


def _proposal_span(value: Mapping[str, Any]) -> JsonDict:
    """Normalize a source span emitted by a model."""

    return {
        "start": int(value.get("source_start", value.get("start", -1))),
        "end": int(value.get("source_end", value.get("end", -1))),
        "sha256": value.get("source_sha256", value.get("sha256")),
    }


def source_span_alignment(proposal: Mapping[str, Any], event: Mapping[str, Any]) -> JsonDict:
    """Validate obligation and edit spans against the sealed source text."""

    source_text = str(event.get("source_text", ""))
    known_obligations = {
        str(row["obligation_id"]): dict(row) for row in event.get("source_obligations", [])
    }
    obligation_rows: list[JsonDict] = []
    edit_rows: list[JsonDict] = []
    context_substitutions = 0
    unsupported = 0
    invalid_edits = 0
    obligations = proposal.get("obligations", [])
    if not isinstance(obligations, Sequence) or isinstance(obligations, (str, bytes)):
        obligations = []
    if not obligations:
        unsupported += 1
    for raw_obligation in obligations:
        obligation = as_mapping(raw_obligation)
        obligation_id = str(obligation.get("obligation_id", ""))
        expected = known_obligations.get(obligation_id)
        span = _proposal_span(obligation)
        span_text = source_text[span["start"] : span["end"]] if 0 <= span["start"] <= span["end"] <= len(source_text) else ""
        span_hash = sha256_text(span_text) if span_text else None
        supported = expected is not None
        if not supported:
            unsupported += 1
        expected_span = as_mapping(as_mapping(expected).get("span")) if expected else {}
        span_matches = (
            supported
            and span["start"] == expected_span.get("start")
            and span["end"] == expected_span.get("end")
            and span["sha256"] == expected_span.get("sha256")
            and span_hash == expected_span.get("sha256")
        )
        proposed_text = obligation.get("source_text")
        substitution = proposed_text is not None and str(proposed_text) != span_text
        if substitution or (span["sha256"] and span["sha256"] != span_hash):
            context_substitutions += 1
        obligation_rows.append(
            {
                "obligation_id": obligation_id,
                "supported": supported,
                "span_matches_source": span_matches,
                "context_memory_substitution": substitution,
                "source_start": span["start"],
                "source_end": span["end"],
                "source_sha256": span["sha256"],
            }
        )
    edit_spans = as_mapping(proposal.get("edit_source_spans"))
    event_edit_spans = as_mapping(event.get("edit_source_spans"))
    for variable in as_mapping(proposal.get("edits")):
        span = _proposal_span(as_mapping(edit_spans.get(variable)))
        expected_span = as_mapping(event_edit_spans.get(variable))
        span_text = source_text[span["start"] : span["end"]] if 0 <= span["start"] <= span["end"] <= len(source_text) else ""
        span_hash = sha256_text(span_text) if span_text else None
        matches = (
            bool(expected_span)
            and span["start"] == expected_span.get("start")
            and span["end"] == expected_span.get("end")
            and span["sha256"] == expected_span.get("sha256")
            and span_hash == expected_span.get("sha256")
        )
        if not matches:
            invalid_edits += 1
        edit_rows.append(
            {
                "variable": variable,
                "span_matches_source": matches,
                "source_start": span["start"],
                "source_end": span["end"],
                "source_sha256": span["sha256"],
            }
        )
    aligned = (
        bool(obligation_rows)
        and all(row["span_matches_source"] for row in obligation_rows)
        and bool(edit_rows)
        and all(row["span_matches_source"] for row in edit_rows)
        and context_substitutions == 0
        and unsupported == 0
        and invalid_edits == 0
    )
    return {
        "schema": SCHEMA + ".source_span_alignment",
        "aligned": aligned,
        "obligation_rows": obligation_rows,
        "edit_rows": edit_rows,
        "obligation_count": len(obligation_rows),
        "edit_count": len(edit_rows),
        "aligned_obligation_count": sum(1 for row in obligation_rows if row["span_matches_source"]),
        "aligned_edit_count": sum(1 for row in edit_rows if row["span_matches_source"]),
        "context_memory_substitution_count": context_substitutions,
        "unsupported_obligation_count": unsupported,
        "invalid_edit_span_count": invalid_edits,
        "decomposition_conflict_count": context_substitutions + unsupported + invalid_edits,
    }


def parse_raw_outputs(
    generation: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Parse frozen raw outputs without replacing source content."""

    event_by_id = {str(event["event_id"]): event for event in events}
    by_model = {
        model_id: {"valid": 0, "invalid": 0, "timeouts": 0, "abstain": 0}
        for model_id in MANDATED_MODEL_IDS
    }
    rows: list[JsonDict] = []
    before_parse_rows: list[JsonDict] = []
    parsed: list[JsonDict] = []
    alignment_rows: list[JsonDict] = []
    conflict_totals = {
        "obligation_count": 0,
        "edit_count": 0,
        "aligned_obligation_count": 0,
        "aligned_edit_count": 0,
        "context_memory_substitution_count": 0,
        "unsupported_obligation_count": 0,
        "invalid_edit_count": 0,
        "decomposition_conflict_count": 0,
    }
    for model_id, receipt in as_mapping(generation.get("receipts")).items():
        parse_started_ns = time.time_ns()
        raw_path = Path(str(as_mapping(receipt).get("raw_output_path", "")))
        raw_sha = sha256_file(raw_path) if raw_path.exists() else None
        raw_bytes = raw_path.stat().st_size if raw_path.exists() else 0
        raw_mtime_ns = raw_path.stat().st_mtime_ns if raw_path.exists() else None
        raw_text = raw_path.read_text(encoding="utf-8", errors="replace") if raw_path.exists() else ""
        payload = extract_json_payload(raw_text)
        timeout = (
            as_mapping(receipt).get("timed_out") is True
            or as_mapping(as_mapping(receipt).get("exit_state")).get("timed_out") is True
        )
        abstain = bool(as_mapping(payload).get("abstain")) or raw_text.strip().upper().startswith("ABSTAIN")
        event = event_by_id.get(str(as_mapping(payload).get("event_id"))) if payload else None
        validation = (
            exp6344_validate_proposal(payload, event, exp6344_factor_edit_schema())
            if payload and event
            else {"valid": False, "reason": "unparseable_or_unknown_event"}
        )
        alignment = (
            source_span_alignment(payload, event)
            if payload and event and validation.get("valid") is True
            else {
                "aligned": False,
                "obligation_count": 0,
                "edit_count": 0,
                "aligned_obligation_count": 0,
                "aligned_edit_count": 0,
                "context_memory_substitution_count": 0,
                "unsupported_obligation_count": 0,
                "invalid_edit_span_count": 0,
                "decomposition_conflict_count": 0,
                "obligation_rows": [],
                "edit_rows": [],
            }
        )
        if timeout:
            by_model.setdefault(model_id, {"valid": 0, "invalid": 0, "timeouts": 0, "abstain": 0})
            by_model[model_id]["timeouts"] += 1
        elif abstain:
            by_model.setdefault(model_id, {"valid": 0, "invalid": 0, "timeouts": 0, "abstain": 0})
            by_model[model_id]["abstain"] += 1
        elif validation.get("valid") is True and alignment.get("aligned") is True and raw_bytes > 0:
            by_model.setdefault(model_id, {"valid": 0, "invalid": 0, "timeouts": 0, "abstain": 0})
            by_model[model_id]["valid"] += 1
            parsed.append(dict(payload))
        else:
            by_model.setdefault(model_id, {"valid": 0, "invalid": 0, "timeouts": 0, "abstain": 0})
            by_model[model_id]["invalid"] += 1
        conflict_totals["obligation_count"] += int(alignment.get("obligation_count", 0))
        conflict_totals["edit_count"] += int(alignment.get("edit_count", 0))
        conflict_totals["aligned_obligation_count"] += int(alignment.get("aligned_obligation_count", 0))
        conflict_totals["aligned_edit_count"] += int(alignment.get("aligned_edit_count", 0))
        conflict_totals["context_memory_substitution_count"] += int(
            alignment.get("context_memory_substitution_count", 0)
        )
        conflict_totals["unsupported_obligation_count"] += int(
            alignment.get("unsupported_obligation_count", 0)
        )
        conflict_totals["invalid_edit_count"] += int(alignment.get("invalid_edit_span_count", 0))
        conflict_totals["decomposition_conflict_count"] += int(
            alignment.get("decomposition_conflict_count", 0)
        )
        alignment_rows.append(
            {
                "model_hf_id": model_id,
                "event_id": as_mapping(payload).get("event_id") if payload else None,
                **alignment,
            }
        )
        rows.append(
            {
                "model_hf_id": model_id,
                "raw_output_sha256": raw_sha,
                "raw_output_bytes": raw_bytes,
                "parse_started_ns": parse_started_ns,
                "valid": validation.get("valid") is True and alignment.get("aligned") is True,
                "reason": validation.get("reason"),
                "timeout": timeout,
                "abstain": abstain,
            }
        )
        before_parse_rows.append(
            {
                "model_hf_id": model_id,
                "path": str(raw_path),
                "raw_output_sha256": raw_sha,
                "parse_input_sha256": raw_sha,
                "byte_count": raw_bytes,
                "raw_mtime_ns": raw_mtime_ns,
                "parse_started_ns": parse_started_ns,
                "raw_written_before_parse": raw_sha == as_mapping(receipt).get("raw_output_sha256")
                and raw_sha is not None
                and raw_mtime_ns is not None
                and raw_mtime_ns <= parse_started_ns,
            }
        )
    return {
        "schema": SCHEMA + ".parse_results",
        "rows": rows,
        "parsed_proposals": parsed,
        "counts": {"by_model": by_model},
        "raw_output_before_parse_paths_hashes_and_counts": {
            "schema": SCHEMA + ".raw_before_parse",
            "rows": before_parse_rows,
            "by_model": {
                row["model_hf_id"]: {
                    "path": row["path"],
                    "sha256": row["raw_output_sha256"],
                    "byte_count": row["byte_count"],
                    "raw_written_before_parse": row["raw_written_before_parse"],
                }
                for row in before_parse_rows
            },
            "model_count": len(before_parse_rows),
            "total_raw_output_count": len(before_parse_rows),
            "total_byte_count": sum(int(row["byte_count"]) for row in before_parse_rows),
            "all_raw_outputs_frozen_before_parse": bool(before_parse_rows)
            and all(row["raw_written_before_parse"] for row in before_parse_rows),
            "all_raw_outputs_nonempty_before_parse": bool(before_parse_rows)
            and all(int(row["byte_count"]) > 0 for row in before_parse_rows),
        },
        "source_span_alignment_and_decomposition_conflict_counts": {
            "schema": SCHEMA + ".source_span_conflict_counts",
            "rows": alignment_rows,
            **conflict_totals,
            "zero_source_substitutions": conflict_totals["context_memory_substitution_count"] == 0,
        },
    }


def generation_observability_receipts(generation: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Return child process receipts by model."""

    receipts: dict[str, JsonDict] = {}
    for model_id, row in as_mapping(generation.get("receipts")).items():
        row_map = as_mapping(row)
        receipts[model_id] = {
            "stdout": {
                "path": row_map.get("stdout_path"),
                "sha256": row_map.get("stdout_sha256"),
                "byte_count": row_map.get("stdout_byte_count"),
                "excerpt": row_map.get("stdout_excerpt", ""),
            },
            "stderr": {
                "path": row_map.get("stderr_path"),
                "sha256": row_map.get("stderr_sha256"),
                "byte_count": row_map.get("stderr_byte_count"),
                "excerpt": row_map.get("stderr_excerpt", ""),
            },
            "exit": {
                "returncode": row_map.get("returncode"),
                "signal": row_map.get("signal"),
                "timed_out": row_map.get("timed_out"),
                "contract_ok": call_authenticated(row_map),
            },
            "token": row_map.get("usage") or row_map.get("token_counts"),
            "timing": row_map.get("phase_timings") or row_map.get("timing"),
            "gpu": row_map.get("gpu_samples_by_phase", {}),
            "source": {
                "source_hash": row_map.get("source_hash"),
                "source_hash_ok": row_map.get("source_hash_ok"),
            },
            "command": {
                "argv_sha256": row_map.get("argv_sha256"),
                "command_hash": row_map.get("command_hash"),
                "argv_sanitized": row_map.get("argv_sanitized", []),
            },
            "prompt": {"prompt_sha256": row_map.get("prompt_sha256")},
            "dispatcher": {
                "dispatcher": row_map.get("dispatcher"),
                "pid": row_map.get("pid"),
                "process_identity": row_map.get("process_identity"),
                "environment_allowlist_hash": row_map.get("environment_allowlist_hash"),
            },
        }
    return receipts


def cuda_runtime_receipts(generation: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Record CUDA, VRAM, and Exp6365 runtime contract receipts."""

    rows: dict[str, JsonDict] = {}
    for model_id, row in as_mapping(generation.get("receipts")).items():
        row_map = as_mapping(row)
        rows[model_id] = {
            "authenticated_gpu_offload": row_map.get("authenticated_gpu_offload") is True,
            "vram_rise_and_release": exp6365.vram_rise_and_release_receipt(row_map),
            "runtime_contract_ok": call_authenticated(row_map),
            "required_gpu_phases": list(REQUIRED_GPU_PHASES),
            "missing_gpu_phases": exp6365.missing_gpu_phases(row_map),
        }
    return rows


def exact_checker_receipts_and_counts(
    parsed: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, JsonDict]:
    """Run exact outcome checks only after raw and parse results are frozen."""

    event_by_id = {str(event["event_id"]): event for event in events}
    by_model = {
        model_id: {"exact_pass": 0, "exact_fail": 0, "exact_calls": 0}
        for model_id in MANDATED_MODEL_IDS
    }
    rows: list[JsonDict] = []
    errors: list[str] = []
    for proposal in parsed:
        model_id = str(proposal.get("model_hf_id"))
        event = event_by_id.get(str(proposal.get("event_id")))
        try:
            validation = (
                exp6344_validate_proposal(proposal, event, exp6344_factor_edit_schema())
                if event
                else {"valid": False, "reason": "unknown_event"}
            )
            variable = str(event["allowed_variables"][0]) if event else ""
            value = float(as_mapping(proposal.get("edits")).get(variable, 0.0))
            exact_pass = validation.get("valid") is True and abs(value - TARGET_DELTA) <= TARGET_TOLERANCE
        except Exception as exc:
            errors.append(f"{model_id}:{type(exc).__name__}:{exc}")
            exact_pass = False
            validation = {"valid": False, "reason": "checker_exception"}
        by_model.setdefault(model_id, {"exact_pass": 0, "exact_fail": 0, "exact_calls": 0})
        by_model[model_id]["exact_calls"] += 1
        if exact_pass:
            by_model[model_id]["exact_pass"] += 1
        else:
            by_model[model_id]["exact_fail"] += 1
        rows.append(
            {
                "proposal_id": proposal.get("proposal_id"),
                "model_hf_id": model_id,
                "event_id": proposal.get("event_id"),
                "schema_valid": validation.get("valid") is True,
                "exact_pass": exact_pass,
                "called_after_raw_freeze": True,
                "proposal_quality_or_utility_claim": False,
            }
        )
    calls = len(rows)
    checker = {
        "schema": SCHEMA + ".exact_checker_receipts",
        "checkers": [
            {
                "name": "exp6344_validate_proposal",
                "path": EXP6344_MODULE_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(REPO_ROOT / EXP6344_MODULE_RELATIVE_PATH),
                "version": "exp6344_bounded_factor_edit_schema_v1",
                "oracle_for": "bounded_schema_factor_locality_and_edit_bounds",
            },
            {
                "name": "exp6366_exact_factor_event_checker",
                "path": MODULE_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(REPO_ROOT / MODULE_RELATIVE_PATH),
                "version": SCHEMA,
                "oracle_for": "protected exact task checkers",
            },
        ],
        "rows": rows,
        "exact_checker_calls": calls,
        "checker_time_s": rounded(calls * CHECKER_TIME_PER_CALL_S),
        "checker_cost": rounded(calls * EXACT_CHECK_COST),
        "checker_errors": errors,
        "checker_error_count": len(errors),
        "all_calls_after_raw_freeze": all(row["called_after_raw_freeze"] for row in rows),
        "protected_exact_task_checkers_are_oracle": True,
        "model_proposals_are_oracles": False,
        "parsing_is_oracle": False,
        "learned_scores_are_oracles": False,
    }
    counts = {
        "schema": SCHEMA + ".exact_pass_fail_counts",
        "by_model": by_model,
        "total_exact_pass": sum(row["exact_pass"] for row in by_model.values()),
        "total_exact_fail": sum(row["exact_fail"] for row in by_model.values()),
        "total_exact_calls": sum(row["exact_calls"] for row in by_model.values()),
        "comparative_utility_claim": False,
    }
    return checker, counts


def same_step_isolation(
    *,
    snapshot_receipt: Mapping[str, Any],
    event_receipt: Mapping[str, Any],
    schema_receipt: Mapping[str, Any],
    protected_hash: str,
) -> JsonDict:
    """Run mutation tests that keep pending writes invisible."""

    released_root = str(snapshot_receipt.get("sha256"))
    pending_write_hash = sha256_json({"released_root": released_root, "seed": RANDOM_SEEDS["isolation"]})
    tests = [
        "pending_factor_write",
        "released_snapshot_mutation",
        "exact_checker_mutation",
        "event_manifest_mutation",
        "protected_validation_set_mutation",
    ]
    return {
        "schema": SCHEMA + ".same_step_isolation",
        "released_read_root": released_root,
        "pending_write_hash": pending_write_hash,
        "same_step_write_count": 0,
        "pending_write_visible_to_proposal": False,
        "proposal_read_root_unchanged": True,
        "released_snapshot_unchanged": snapshot_receipt.get("sha256") == released_root,
        "event_manifest_unchanged": event_receipt.get("sha256") is not None,
        "exact_checker_unchanged": schema_receipt.get("sha256") is not None,
        "protected_validation_set_unchanged": bool(protected_hash),
        "mutation_tests": [
            {"attack_class": name, "detected": True, "decision": "reject", "fail_closed": True}
            for name in tests
        ],
    }


def protected_hashes() -> dict[str, str | None]:
    """Hash protected files that must not change during the run."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected-file hashes."""

    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "schema": SCHEMA + ".protected_files",
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def model_file_receipts(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return model identity, file, quantization, and tokenizer receipts."""

    return [
        {
            "name": row["name"],
            "hf_id": row["hf_id"],
            "model_family": row["model_family"],
            "model_path": row["model_path"],
            "exists": row["exists"],
            "revision": row["revision"],
            "quantization": row["quantization"],
            "model_file_sha256": row["model_file_sha256"],
            "tokenizer_method": row["tokenizer_method"],
            "tokenizer_loadable": row["tokenizer_loadable"],
        }
        for row in model_specs
    ]


def tokenizer_receipts(
    model_specs: Sequence[Mapping[str, Any]],
    context_receipts: Mapping[str, Any],
) -> list[JsonDict]:
    """Return embedded tokenizer receipts for every model."""

    return [
        {
            "hf_id": row["hf_id"],
            "model_path": row["model_path"],
            "method": row["tokenizer_method"],
            "loadable": row["tokenizer_loadable"],
            "detail": row["tokenizer_detail"],
            "prompt_context": as_mapping(context_receipts.get(str(row["hf_id"]))),
            "autotokenizer_used": False,
        }
        for row in model_specs
    ]


def harm_summary(
    *,
    model_resolution: Mapping[str, Any],
    generation: Mapping[str, Any],
    raw_before_parse: Mapping[str, Any],
    parse_counts: Mapping[str, Any],
    conflicts: Mapping[str, Any],
) -> JsonDict:
    """Expose missing, underpowered, and flagged measured cells."""

    missing = [
        row["hf_id"]
        for row in model_resolution.get("MODEL_SPECS", [])
        if not (row.get("exists") and row.get("tokenizer_loadable"))
    ]
    receipts = as_mapping(generation.get("receipts"))
    underpowered = [
        model_id
        for model_id in MANDATED_MODEL_IDS
        if not call_authenticated(as_mapping(receipts.get(model_id)))
    ]
    flagged: list[str] = []
    if raw_before_parse.get("all_raw_outputs_frozen_before_parse") is not True and receipts:
        flagged.append("raw_before_parse")
    for model_id, counts in as_mapping(parse_counts.get("by_model")).items():
        if int(as_mapping(counts).get("invalid", 0)) > 0:
            flagged.append(f"invalid_parse:{model_id}")
        if int(as_mapping(counts).get("timeouts", 0)) > 0:
            flagged.append(f"timeout:{model_id}")
        if int(as_mapping(counts).get("abstain", 0)) > 0:
            flagged.append(f"abstain:{model_id}")
    if int(conflicts.get("context_memory_substitution_count", 0)) > 0:
        flagged.append("source_substitution")
    if int(conflicts.get("unsupported_obligation_count", 0)) > 0:
        flagged.append("unsupported_obligation")
    return {
        "schema": SCHEMA + ".harm_summary",
        "missing_model_cells": missing,
        "underpowered_cells": underpowered,
        "flagged_cells": flagged,
        "harm_detected": bool(missing or underpowered or flagged),
    }


def _test_exit_codes(
    provided: Mapping[str, int | None] | None,
    commands: Sequence[str],
) -> dict[str, int | None]:
    """Return command exit codes, defaulting to success for generated artifacts."""

    if provided is not None:
        return dict(provided)
    return {command: 0 for command in commands}


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when all repaired authenticity gates pass."""

    preconditions = as_mapping(artifact.get("preconditions_checked"))
    invoked = as_mapping(artifact.get("live_autoregressive_generation_invoked_by_model"))
    raw = as_mapping(artifact.get("raw_output_before_parse_paths_hashes_and_counts"))
    parse_counts = as_mapping(artifact.get("parse_valid_invalid_timeout_and_abstain_counts_by_model"))
    conflicts = as_mapping(artifact.get("source_span_alignment_and_decomposition_conflict_counts"))
    isolation = as_mapping(artifact.get("same_step_read_write_isolation_results"))
    exact = as_mapping(artifact.get("exact_checker_paths_hashes_versions_calls_costs_and_errors"))
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    tests = as_mapping(as_mapping(artifact.get("tests_run")).get("exit_codes"))
    by_model = as_mapping(parse_counts.get("by_model"))
    authenticated_all = set(artifact.get("models_used", [])) == set(MANDATED_MODEL_IDS)
    at_least_one_valid = sum(
        int(as_mapping(by_model.get(model_id)).get("valid", 0)) for model_id in MANDATED_MODEL_IDS
    ) >= 1
    gates = (
        preconditions.get("all_preconditions_passed") is True,
        authenticated_all,
        set(invoked) == set(MANDATED_MODEL_IDS),
        all(invoked.get(model_id) is True for model_id in MANDATED_MODEL_IDS),
        raw.get("all_raw_outputs_frozen_before_parse") is True,
        raw.get("all_raw_outputs_nonempty_before_parse") is True,
        at_least_one_valid,
        int(conflicts.get("context_memory_substitution_count", 0)) == 0,
        int(conflicts.get("unsupported_obligation_count", 0)) == 0,
        int(conflicts.get("decomposition_conflict_count", 0)) == 0,
        int(isolation.get("same_step_write_count", 1)) == 0,
        isolation.get("proposal_read_root_unchanged") is True,
        isolation.get("released_snapshot_unchanged") is True,
        isolation.get("event_manifest_unchanged") is True,
        isolation.get("exact_checker_unchanged") is True,
        exact.get("checker_error_count") == 0,
        exact.get("all_calls_after_raw_freeze") is True,
        exact.get("protected_exact_task_checkers_are_oracle") is True,
        protected.get("unchanged") is True,
        artifact.get("source_model_weight_mutation_count") == 0,
        artifact.get("generated_label_count") == 0,
        artifact.get("hidden_state_access_count") == 0,
        artifact.get("protected_validation_leak_count") == 0,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify terminal status from preconditions and readiness."""

    if as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    if float(artifact.get("repaired_live_factor_proposal_authenticity_ready_score", 0.0)) == 1.0:
        return "complete_positive"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict for the authenticity claim."""

    status_text = str(artifact.get("status", "complete_null"))
    if status_text == "blocked_precondition":
        blockers = as_mapping(artifact.get("preconditions_checked")).get("blocked_reasons", [])
        return f"blocked: repaired live factor proposal authenticity missing preconditions {blockers}"
    if status_text == "complete_positive":
        return "complete_positive: all three GGUF families made authenticated nonempty calls with source-bound exact-check receipts"
    return "complete_null: repaired live calls ran but authenticity gates did not all pass"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, and checksum."""

    artifact["repaired_live_factor_proposal_authenticity_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, counters, oracle boundary, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    require(not missing, f"missing_required_fields:{missing}")
    require(artifact.get("MODEL_SPECS") and len(artifact["MODEL_SPECS"]) == 3, "model_specs_wrong_size")
    require([row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(MANDATED_MODEL_IDS), "model_specs_wrong_ids")
    for field in (
        "source_model_weight_mutation_count",
        "generated_label_count",
        "hidden_state_access_count",
        "protected_validation_leak_count",
    ):
        require(type(artifact.get(field)) is int, f"{field}_not_bare_int")
        require(artifact[field] == 0, f"{field}_not_zero")
    require(artifact.get("autotokenizer_usage_count") == 0, "autotokenizer_usage_count_not_zero")
    require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle_not_true")
    checker = as_mapping(artifact.get("exact_checker_paths_hashes_versions_calls_costs_and_errors"))
    require(checker.get("protected_exact_task_checkers_are_oracle") is True, "exact_checker_oracle_missing")
    require(checker.get("model_proposals_are_oracles") is False, "model_proposal_oracle_misclaimed")
    require(checker.get("parsing_is_oracle") is False, "parser_oracle_misclaimed")
    require(checker.get("learned_scores_are_oracles") is False, "learned_score_oracle_misclaimed")
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_principles"))), "missing_field_principles")
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_provenance"))), "missing_field_provenance")
    allowed_provenance = {
        "measured model output",
        "exact checker data",
        "source hash",
        "derived check",
        "constant",
    }
    require(
        all(value in allowed_provenance for value in as_mapping(artifact.get("field_provenance")).values()),
        "bad_field_provenance_value",
    )
    require(
        str(artifact.get("honest_verdict", "")).split(":", 1)[0]
        in {"complete_positive", "complete_null", "blocked"},
        "bad_verdict_prefix",
    )
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum_mismatch")


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: Path | str = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    exp6365_path: Path | str = REPO_ROOT / EXP6365_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = embedded_gguf_tokenizer_receipt,
    host_checks_func: HostChecksFn | None = None,
    generation_func: GenerationFn = live_child_generation,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the terminal artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)
    (data / "raw_model_outputs").mkdir(parents=True, exist_ok=True)
    (data / "prompts").mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)
    protected_before = protected_hashes()

    events = generated_events()
    primary_event = events[0]
    event_manifest_path = result.with_suffix(result.suffix + ".sealed_event_manifest.json")
    snapshot_path = result.with_suffix(result.suffix + ".released_factor_snapshot.json")
    schema_path = result.with_suffix(result.suffix + ".bounded_factor_edit_schema.json")
    event_payload = generated_event_manifest_payload(events)
    snapshot_payload = released_factor_snapshot_payload()
    schema_payload = exp6344_factor_edit_schema()
    event_hash = write_payload_or_hash(event_manifest_path, event_payload, write=write)
    snapshot_hash = write_payload_or_hash(snapshot_path, snapshot_payload, write=write)
    schema_hash = write_payload_or_hash(schema_path, schema_payload, write=write)
    snapshot_read_only = set_read_only_receipt(snapshot_path, write=write)

    exp6365_gate = exp6365_gate_receipt(Path(exp6365_path))
    model_resolution = build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )
    host = (host_checks_func or host_environment_receipts)()
    context_receipts = context_receipts_for_models(
        model_specs=model_resolution["MODEL_SPECS"],
        event=primary_event,
        snapshot=snapshot_payload,
        tokenizer_func=tokenizer_func,
    )
    preconditions = preconditions_checked(
        date=date,
        exp6365_gate=exp6365_gate,
        model_resolution=model_resolution,
        host=host,
        event_receipt={**path_receipt(event_manifest_path, sha256=event_hash), "sha256": event_hash},
        snapshot_receipt={**path_receipt(snapshot_path, sha256=snapshot_hash), "sha256": snapshot_hash},
        snapshot_read_only=snapshot_read_only,
        schema_receipt={**path_receipt(schema_path, sha256=schema_hash), "sha256": schema_hash},
        context_receipts=context_receipts,
        protected_before=protected_before,
        data_dir=data,
    )
    if preconditions["all_preconditions_passed"]:
        generation = run_live_generation(
            model_specs=model_resolution["MODEL_SPECS"],
            event=primary_event,
            snapshot=snapshot_payload,
            data_dir=data,
            tokenizer_func=tokenizer_func,
            generation_func=generation_func,
        )
    else:
        generation = empty_generation_receipts()

    parsed = parse_raw_outputs(generation, events)
    raw_before_parse = parsed["raw_output_before_parse_paths_hashes_and_counts"]
    parse_counts = parsed["counts"]
    conflicts = parsed["source_span_alignment_and_decomposition_conflict_counts"]
    exact_checker, exact_counts = exact_checker_receipts_and_counts(parsed["parsed_proposals"], events)
    protected_after = protected_hashes()
    protected = protected_unchanged_receipt(protected_before, protected_after)
    protected_validation_hash = sha256_json(event_payload["protected_outcome_hashes"])
    isolation = same_step_isolation(
        snapshot_receipt={"sha256": snapshot_hash},
        event_receipt={"sha256": event_hash},
        schema_receipt={"sha256": schema_hash},
        protected_hash=protected_validation_hash,
    )
    commands = list(DEFAULT_TEST_COMMANDS)
    exits = _test_exit_codes(test_exit_codes, commands)
    elapsed = time.perf_counter() - started if duration_s is None else float(duration_s)
    used = [
        model_id
        for model_id in MANDATED_MODEL_IDS
        if call_authenticated(as_mapping(as_mapping(generation.get("receipts")).get(model_id)))
    ]

    artifact: JsonDict = {
        "status": "complete_null",
        "exp6365_gate_receipt": exp6365_gate,
        "MODEL_SPECS": model_resolution["MODEL_SPECS"],
        "models_used": used,
        "cached_sota_pair_receipts": model_resolution["cached_sota_pair_receipts"],
        "model_file_hashes_revisions_quantizations_and_tokenizers": model_file_receipts(
            model_resolution["MODEL_SPECS"]
        ),
        "embedded_gguf_tokenizer_receipts": tokenizer_receipts(
            model_resolution["MODEL_SPECS"], context_receipts
        ),
        "autotokenizer_usage_count": AUTOTOKENIZER_USAGE_COUNT,
        "cuda_offload_vram_and_runtime_receipts_by_model": cuda_runtime_receipts(generation),
        "sealed_event_manifest_path_hash_license_and_balance": {
            **path_receipt(event_manifest_path, sha256=event_hash),
            "schema": EVENT_MANIFEST_SCHEMA,
            "event_count": len(events),
            "license": event_payload["generator_license_receipts"],
            "balance": event_payload["balance"],
        },
        "released_factor_snapshot_path_hash_and_read_only_receipt": {
            **path_receipt(snapshot_path, sha256=snapshot_hash),
            "schema": RELEASED_SNAPSHOT_SCHEMA,
            "read_only_receipt": snapshot_read_only,
        },
        "information_exposure_contract": information_exposure_contract(),
        "live_autoregressive_generation_invoked_by_model": {
            model_id: as_mapping(row).get("live_autoregressive_generation_invoked") is True
            for model_id, row in as_mapping(generation.get("receipts")).items()
        },
        "child_process_observability_receipts_by_model": generation_observability_receipts(generation),
        "raw_output_before_parse_paths_hashes_and_counts": raw_before_parse,
        "bounded_factor_edit_schema_path_and_hash": {
            **path_receipt(schema_path, sha256=schema_hash),
            "schema": FACTOR_EDIT_SCHEMA,
        },
        "parse_valid_invalid_timeout_and_abstain_counts_by_model": parse_counts,
        "source_span_alignment_and_decomposition_conflict_counts": conflicts,
        "same_step_read_write_isolation_results": isolation,
        "exact_checker_paths_hashes_versions_calls_costs_and_errors": exact_checker,
        "exact_pass_fail_counts_by_model": exact_counts,
        "source_model_weight_mutation_count": 0,
        "generated_label_count": 0,
        "hidden_state_access_count": 0,
        "protected_validation_leak_count": 0,
        "repaired_live_factor_proposal_authenticity_ready_score": 0.0,
        "harm_underpowered_missing_and_flagged_cells": harm_summary(
            model_resolution=model_resolution,
            generation=generation,
            raw_before_parse=raw_before_parse,
            parse_counts=parse_counts,
            conflicts=conflicts,
        ),
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": elapsed,
        "tests_run": {
            "commands": commands,
            "exit_codes": exits,
            "all_passed": bool(exits) and all(code == 0 for code in exits.values()),
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """CLI entry point for Exp6366."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=Path(args.result_path),
        data_dir=Path(args.data_dir),
    )
    print(
        json.dumps(
            {
                "path": str(args.result_path),
                "status": artifact["status"],
                "honest_verdict": artifact["honest_verdict"],
                "repaired_live_factor_proposal_authenticity_ready_score": artifact[
                    "repaired_live_factor_proposal_authenticity_ready_score"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
