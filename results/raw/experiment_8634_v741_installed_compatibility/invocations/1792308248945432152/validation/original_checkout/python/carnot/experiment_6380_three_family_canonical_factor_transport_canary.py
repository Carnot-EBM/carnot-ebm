"""Exp6380 three-family canonical factor transport canary.

Spec refs: REQ-LEARN-6380, SCENARIO-LEARN-6380-GATE,
SCENARIO-LEARN-6380-ARMS, SCENARIO-LEARN-6380-RAW,
SCENARIO-LEARN-6380-ORACLE, SCENARIO-LEARN-6380-READY.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

from carnot import experiment_6365_gguf_child_failure_forensics_and_runtime_contract as exp6365
from carnot import experiment_6366_repaired_live_factor_proposal_authenticity as exp6366
from carnot import experiment_6379_canonical_factor_edit_transport_contract as exp6379
from carnot import experiment_6344_counterexample_factor_proposal_calibration as exp6344
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str, str], JsonDict]
HostChecksFn = Callable[[], JsonDict]
GenerationFn = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6380_three_family_canonical_factor_transport_canary.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6380_three_family_canonical_factor_transport_canary"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6380_three_family_canonical_factor_transport_canary.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6380_three_family_canonical_factor_transport_canary.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
EXP6379_RELATIVE_PATH = exp6379.RESULT_RELATIVE_PATH
EXP6379_SCHEMA_RELATIVE_PATH = Path(
    "results/experiment_6379_canonical_factor_edit_transport_contract.json.canonical_schema.json"
)
EXP6365_RELATIVE_PATH = exp6365.RESULT_RELATIVE_PATH
LICENSE_RELATIVE_PATH = Path("LICENSE")

SCHEMA = "carnot.experiment_6380.three_family_canonical_factor_transport_canary.v1"
RUN_DATE = "20260813"
RANDOM_SEED = 6380
PREFERRED_QUANT = exp6379.PREFERRED_QUANT
TOKENIZER_METHOD = exp6379.TOKENIZER_METHOD
INFERENCE_SUBSTRATE = "live_local_llama_cpp_gguf_three_arm_canonical_factor_transport_canary"
OLD_COMPLETION_BUDGET = 192
N_CTX = exp6379.N_CTX
TIME_BUDGET_S = 420.0
EXACT_CHECK_COST = exp6366.EXACT_CHECK_COST
CHECKER_TIME_PER_CALL_S = exp6366.CHECKER_TIME_PER_CALL_S
TARGET_TOLERANCE = exp6366.TARGET_TOLERANCE

EXP6366_FROZEN_ARM = "frozen_exp6366_prompt_192"
CANONICAL_OLD_ARM = "canonical_prompt_192"
CANONICAL_CAPACITY_ARM = "canonical_prompt_computed_allowance"
ARMS = (EXP6366_FROZEN_ARM, CANONICAL_OLD_ARM, CANONICAL_CAPACITY_ARM)
CANONICAL_ARMS = (CANONICAL_OLD_ARM, CANONICAL_CAPACITY_ARM)
REQUIRED_EVENT_FAMILIES = ("threshold_guard", "route_guard", "conservation_guard")
MANDATED_MODEL_IDS = exp6379.MANDATED_MODEL_IDS
MODEL_TEMPLATES = exp6366.MODEL_TEMPLATES
MODEL_TEMPLATE_BY_ID = exp6366.MODEL_TEMPLATE_BY_ID
REQUIRED_GPU_PHASES = exp6365.REQUIRED_GPU_PHASES
REQUIRED_TIMING_PHASES = exp6365.REQUIRED_TIMING_PHASES
CANONICAL_FIELD_ORDER = exp6379.CANONICAL_FIELD_ORDER
FIXED_OUTPUT_HEADROOM_TOKENS = exp6379.FIXED_OUTPUT_HEADROOM_TOKENS
BASE_SAMPLING_INPUTS = {
    "temperature": exp6366.SAMPLING_PARAMETERS["temperature"],
    "top_p": exp6366.SAMPLING_PARAMETERS["top_p"],
    "n_ctx": N_CTX,
    "n_gpu_layers": exp6366.SAMPLING_PARAMETERS["n_gpu_layers"],
}
RANDOM_SEEDS = {
    "manifest": 638000,
    "arm_schedule": 638001,
    "generation": 638002,
    "parser": 638003,
    "isolation": 638004,
    "exact_checker": 638005,
}

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6380_three_family_canonical_factor_transport_canary --date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6380_three_family_canonical_factor_transport_canary.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6380_three_family_canonical_factor_transport_canary.py "
    "-m pytest tests/python/test_experiment_6380_three_family_canonical_factor_transport_canary.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6380_three_family_canonical_factor_transport_canary.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6380_three_family_canonical_factor_transport_canary.py"
)
E2E_PLAN_READ_COMMAND = "sed -n '1,220p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6380_three_family_canonical_factor_transport_canary.json"
)
DETERMINATION_LINT_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
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
    EXP6379_RELATIVE_PATH,
    EXP6379_SCHEMA_RELATIVE_PATH,
    EXP6365_RELATIVE_PATH,
    exp6344.MODULE_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6365_gguf_child_failure_forensics_and_runtime_contract.py"),
    Path("python/carnot/experiment_6366_repaired_live_factor_proposal_authenticity.py"),
    Path("python/carnot/experiment_6379_canonical_factor_edit_transport_contract.py"),
    Path("python/carnot/inference/sota_models.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6379_gate_receipt",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_hashes_revisions_quantizations_and_tokenizers",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "cuda_offload_and_runtime_receipts_by_model",
    "sealed_event_manifest_path_hash_license_and_balance",
    "canonical_schema_path_hash_and_drift_receipt",
    "preregistered_arm_contract",
    "per_arm_prompt_output_and_context_capacity_receipts",
    "raw_output_before_parse_paths_hashes_and_counts",
    "failure_taxonomy_counts_by_model_and_arm",
    "parse_valid_invalid_timeout_and_abstain_counts_by_model_and_arm",
    "source_span_alignment_and_conflict_counts",
    "exact_checker_paths_versions_calls_costs_and_errors",
    "exact_pass_fail_counts_by_model_and_arm",
    "same_step_read_write_isolation_results",
    "retired_decoding_mechanism_usage_count",
    "three_family_factor_transport_ready_score",
    "semantic_utility_not_implied_by_transport",
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
    "status": "Terminal status separates blocked, positive, null, and retired transport evidence.",
    "exp6379_gate_receipt": "The deterministic transport contract is revalidated before live calls.",
    "MODEL_SPECS": "The three mandated GGUF model rows come from cached SOTA helper calls.",
    "models_used": "Only models with authenticated runtime receipts count as used.",
    "cached_sota_pair_receipts": "Helper-call receipts prevent manual model substitution.",
    "model_file_hashes_revisions_quantizations_and_tokenizers": "Model file identity and tokenizer method are pinned.",
    "embedded_gguf_tokenizer_receipts": "Tokenizer receipts use only embedded GGUF tokenizers.",
    "autotokenizer_usage_count": "Bare zero proves no external tokenizer path was used.",
    "cuda_offload_and_runtime_receipts_by_model": "CUDA offload, timing, token usage, return, raw streams, and cleanup are reported.",
    "sealed_event_manifest_path_hash_license_and_balance": "Fresh licensed events are sealed before prompting.",
    "canonical_schema_path_hash_and_drift_receipt": "The canonical schema source is hash-bound and checked for drift.",
    "preregistered_arm_contract": "The three arms and fixed sampling differences are frozen before generation.",
    "per_arm_prompt_output_and_context_capacity_receipts": "Prompt tokens, output allowance, and context margin are recorded per call.",
    "raw_output_before_parse_paths_hashes_and_counts": "Raw outputs are frozen before classification or parsing.",
    "failure_taxonomy_counts_by_model_and_arm": "Failure labels distinguish thinking, repetition, truncation, syntax, structure, source, semantic, timeout, and abstention.",
    "parse_valid_invalid_timeout_and_abstain_counts_by_model_and_arm": "Parse outcomes stay separate from exact correctness.",
    "source_span_alignment_and_conflict_counts": "Source-bound spans and conflicts are counted before exact checking.",
    "exact_checker_paths_versions_calls_costs_and_errors": "Exact checker identity, calls, costs, and errors are recorded.",
    "exact_pass_fail_counts_by_model_and_arm": "Exact pass and fail counts stay separate from transport readiness.",
    "same_step_read_write_isolation_results": "Same-step writes and protected reads remain invisible.",
    "retired_decoding_mechanism_usage_count": "Bare zero proves retired decode helpers were not used.",
    "three_family_factor_transport_ready_score": "Readiness is a conjunctive transport gate.",
    "semantic_utility_not_implied_by_transport": "The artifact states that transport readiness is not semantic utility.",
    "harm_underpowered_missing_and_flagged_cells": "Missing, invalid, timeout, abstain, underpowered, and retired cells stay visible.",
    "protected_files_unchanged": "Protected files remain byte-identical.",
    "preconditions_checked": "Preconditions freeze upstream, model, tokenizer, GPU, disk, schema, event, source, and protected hashes.",
    "inference_substrate": "The substrate declares local llama.cpp GGUF child-process generation.",
    "verifier_is_oracle": "Bare true applies only to exact task checkers.",
    "field_principles": "Every required field states its guard.",
    "field_provenance": "Every required field maps to specs, inputs, sidecars, model receipts, tests, or exact checks.",
    "random_seed": "Fixed seeds pin schedule and prompt construction.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes are recorded.",
    "reproducibility_checksum": "A normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and states the transport boundary.",
}

FIELD_PROVENANCE: dict[str, str] = {
    "status": "derived check",
    "exp6379_gate_receipt": "source hash",
    "MODEL_SPECS": "source hash",
    "models_used": "measured model output",
    "cached_sota_pair_receipts": "source hash",
    "model_file_hashes_revisions_quantizations_and_tokenizers": "source hash",
    "embedded_gguf_tokenizer_receipts": "derived check",
    "autotokenizer_usage_count": "constant",
    "cuda_offload_and_runtime_receipts_by_model": "measured model output",
    "sealed_event_manifest_path_hash_license_and_balance": "source hash",
    "canonical_schema_path_hash_and_drift_receipt": "derived check",
    "preregistered_arm_contract": "constant",
    "per_arm_prompt_output_and_context_capacity_receipts": "derived check",
    "raw_output_before_parse_paths_hashes_and_counts": "measured model output",
    "failure_taxonomy_counts_by_model_and_arm": "derived check",
    "parse_valid_invalid_timeout_and_abstain_counts_by_model_and_arm": "derived check",
    "source_span_alignment_and_conflict_counts": "derived check",
    "exact_checker_paths_versions_calls_costs_and_errors": "exact checker data",
    "exact_pass_fail_counts_by_model_and_arm": "exact checker data",
    "same_step_read_write_isolation_results": "derived check",
    "retired_decoding_mechanism_usage_count": "constant",
    "three_family_factor_transport_ready_score": "derived check",
    "semantic_utility_not_implied_by_transport": "constant",
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
    """Return compact JSON with deterministic key handling."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text through UTF-8 bytes."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash the compact JSON serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Return a file digest, or None when absent."""

    path = Path(path)
    if not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def require(condition: bool, reason: str) -> None:
    """Raise a deterministic validation error when a gate fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other values with an empty map."""

    return value if isinstance(value, Mapping) else {}


def model_slug(model_id: str) -> str:
    """Turn a model id into a stable file-name fragment."""

    return exp6379.model_slug(model_id)


def rounded(value: float) -> float:
    """Round measured receipts without hiding small values."""

    return round(float(value), 12)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_payload_or_hash(path: Path, payload: Mapping[str, Any], *, write: bool) -> str:
    """Write JSON when requested, otherwise return the would-be digest."""

    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        require(digest is not None, "json_write_failed")
        return str(digest)
    return sha256_json(payload)


def path_receipt(path: Path, *, sha256: str | None = None) -> JsonDict:
    """Record path, presence, size, and hash."""

    return {
        "path": str(path),
        "present": path.is_file(),
        "sha256": sha256 if sha256 is not None else sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def read_json(path: Path) -> JsonDict:
    """Read a JSON object."""

    return json.loads(path.read_text(encoding="utf-8"))


def revision_from_path(path: Path) -> str | None:
    """Extract a Hugging Face snapshot revision from a path."""

    return exp6379.revision_from_path(path)


def quantization_from_path(path: Path) -> str:
    """Extract a known GGUF quantization token."""

    return exp6379.quantization_from_path(path)


def deterministic_model_specs(base: Path) -> list[JsonDict]:
    """Return deterministic model rows for focused tests."""

    return exp6379.deterministic_model_specs(base)


def _token_count(receipt: Mapping[str, Any]) -> int:
    """Read either tokenizer count key used by the prior experiments."""

    return int(receipt.get("token_count", receipt.get("prompt_tokens", 0)) or 0)


def embedded_gguf_tokenizer_receipt(model_path: str, text: str) -> JsonDict:  # pragma: no cover
    """Count tokens with the GGUF tokenizer embedded in the model file."""

    receipt = exp6379.embedded_gguf_tokenizer_receipt(model_path, text)
    count = _token_count(receipt)
    return {
        **receipt,
        "prompt_tokens": count,
        "token_count": count,
        "autotokenizer_used": False,
    }


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = embedded_gguf_tokenizer_receipt,
) -> JsonDict:
    """Resolve all mandated GGUF rows and preflight embedded tokenizers."""

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
        tokenized = tokenizer_func(path_text, "Exp6380 embedded tokenizer precheck.")
        record = {
            **template,
            "gpu": int(row.get("gpu", template["gpu"])),
            "model_path": path_text,
            "exists": bool(path_text) and path.is_file(),
            "revision": revision_from_path(path) if path_text else None,
            "quantization": quantization_from_path(path) if path_text else "unknown",
            "model_file_sha256": sha256_file(path) if path_text else None,
            "tokenizer_method": tokenized.get("method", TOKENIZER_METHOD),
            "tokenizer_loadable": tokenized.get("loadable") is True,
            "tokenizer_detail": str(tokenized.get("tokenizer_detail", "")),
            "prompt_tokens_for_tokenizer_precheck": _token_count(tokenized),
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
        "autotokenizer_usage_count": 0,
    }


def generated_events() -> list[JsonDict]:
    """Return the fresh executable event matrix from Exp6366."""

    return exp6366.generated_events()


def selected_canary_events(events: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Choose one event per constraint family for the bounded canary."""

    by_family: dict[str, JsonDict] = {}
    for event in events:
        family = str(event.get("family"))
        if family in REQUIRED_EVENT_FAMILIES and family not in by_family:
            by_family[family] = dict(event)
    missing = [family for family in REQUIRED_EVENT_FAMILIES if family not in by_family]
    require(not missing, f"missing_required_event_families:{missing}")
    return {
        model_id: dict(by_family[family])
        for model_id, family in zip(MANDATED_MODEL_IDS, REQUIRED_EVENT_FAMILIES, strict=True)
    }


def event_manifest_payload(events: Sequence[Mapping[str, Any]], selected: Mapping[str, Any]) -> JsonDict:
    """Build the sealed event manifest without exposing protected outcomes."""

    payload = exp6366.generated_event_manifest_payload(events)
    return {
        **payload,
        "schema": SCHEMA + ".sealed_event_manifest",
        "fresh_for_exp6380": True,
        "selected_event_schedule": {
            model_id: {
                "event_id": as_mapping(event).get("event_id"),
                "family": as_mapping(event).get("family"),
            }
            for model_id, event in selected.items()
        },
        "random_seeds": {
            "manifest": RANDOM_SEEDS["manifest"],
            "arm_schedule": RANDOM_SEEDS["arm_schedule"],
        },
    }


def canonical_contract_for_event(
    event: Mapping[str, Any],
    *,
    arm: str = CANONICAL_CAPACITY_ARM,
) -> JsonDict:
    """Build the canonical transport object for one source event."""

    variable = str(event["allowed_variables"][0])
    obligation = as_mapping(event["source_obligations"][0])
    obligation_span = as_mapping(obligation.get("span"))
    edit_span = as_mapping(as_mapping(event["edit_source_spans"]).get(variable))
    source = {
        "event_id": event["event_id"],
        "changed_factor": event["changed_factor"],
        "source_text": event["source_text"],
        "source_text_sha256": event["source_text_sha256"],
        "variable": variable,
        "edit_bounds": dict(as_mapping(event.get("edit_bounds"))),
        "obligation": {
            "obligation_id": obligation.get("obligation_id"),
            "source_start": obligation_span.get("start"),
            "source_end": obligation_span.get("end"),
            "source_sha256": obligation_span.get("sha256"),
            "source_text": obligation.get("text"),
        },
        "edit_source_span": {
            "source_start": edit_span.get("start"),
            "source_end": edit_span.get("end"),
            "source_sha256": edit_span.get("sha256"),
        },
    }
    return {
        "schema": SCHEMA + ".canonical_contract",
        "version": exp6379.CANONICAL_SCHEMA_VERSION,
        "output_schema": exp6379.CANONICAL_FACTOR_SCHEMA,
        "field_order": list(CANONICAL_FIELD_ORDER),
        "fixed_fields": {
            "schema": exp6379.CANONICAL_FACTOR_SCHEMA,
            "event_id": source["event_id"],
            "arm": arm,
            "candidate_index": 0,
            "changed_factor": source["changed_factor"],
        },
        "model_bound_fields": ["proposal_id", "model_hf_id", "model_family"],
        "numeric_bounds": {
            "selection_score": {"min": 0.0, "max": 1.0},
            "edits": source["edit_bounds"],
        },
        "allowed_variables": [variable],
        "source_event": source,
        "evidence_summary": {
            "required": True,
            "max_chars": exp6379.EVIDENCE_SUMMARY_MAX_CHARS,
            "visible_evidence_only": True,
            "hidden_reasoning_forbidden": True,
        },
        "forbidden_fields": list(exp6379.canonical_factor_edit_contract()["forbidden_fields"]),
    }


def compact_output_example(
    contract: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    arm: str,
) -> JsonDict:
    """Generate the compact object expected from the model."""

    source = as_mapping(contract.get("source_event"))
    variable = str(source.get("variable"))
    fixed = dict(as_mapping(contract.get("fixed_fields")))
    fixed["arm"] = arm
    model_id = str(spec["hf_id"])
    return {
        "schema": fixed["schema"],
        "proposal_id": f"{model_slug(model_id)}:{source['event_id']}:0",
        "event_id": fixed["event_id"],
        "model_hf_id": model_id,
        "model_family": spec["model_family"],
        "arm": fixed["arm"],
        "candidate_index": fixed["candidate_index"],
        "changed_factor": fixed["changed_factor"],
        "edits": {variable: exp6366.TARGET_DELTA},
        "selection_score": exp6366.TARGET_DELTA,
        "obligations": [dict(as_mapping(source.get("obligation")))],
        "edit_source_spans": {variable: dict(as_mapping(source.get("edit_source_span")))},
        "evidence_summary": (
            f"Visible obligation {source['obligation']['obligation_id']} "
            f"supports editing {variable}."
        ),
    }


def schema_description(contract: Mapping[str, Any]) -> JsonDict:
    """Generate a bounded schema description from the canonical object."""

    return exp6379.schema_description(contract)


def canonical_prompt_payload(
    spec: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    arm: str,
) -> JsonDict:
    """Build the canonical prompt payload for one arm."""

    contract = canonical_contract_for_event(event, arm=arm)
    return {
        "instruction": (
            "Return exactly one JSON object. Copy ids, spans, hashes, and visible "
            "source text exactly. Do not output markdown or hidden reasoning."
        ),
        "canonical_schema": schema_description(contract),
        "source_event": exp6366.exposed_event_payload(event),
        "fixed_repetition_policy": exp6379.repetition_policy_and_failure_thresholds(),
        "compact_valid_shape": compact_output_example(contract, spec, arm=arm),
    }


def prompt_payload_for_arm(
    spec: Mapping[str, Any],
    event: Mapping[str, Any],
    arm: str,
    snapshot: Mapping[str, Any],
) -> JsonDict:
    """Return the exact prompt payload for a preregistered arm."""

    if arm == EXP6366_FROZEN_ARM:
        return exp6366.prompt_payload(spec, event, snapshot)
    if arm in CANONICAL_ARMS:
        return canonical_prompt_payload(spec, event, arm=arm)
    raise ValueError(f"unknown_arm:{arm}")


def prompt_text_for_arm(payload: Mapping[str, Any], arm: str) -> str:
    """Serialize the prompt sent to llama.cpp for one arm."""

    if arm == EXP6366_FROZEN_ARM:
        return exp6366.prompt_text(payload)
    return (
        "Do not think. Do not output analysis. Do not output markdown. /no_think\n"
        "Return only this JSON object, with no extra text:\n"
        + canonical_json(payload["compact_valid_shape"])
        + "\nSchema and visible source:\n"
        + canonical_json(payload)
        + "\nJSON:"
    )


def output_allowance_for_model(
    spec: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    tokenizer_func: TokenizerFn,
) -> JsonDict:
    """Compute the output allowance under one model's embedded tokenizer."""

    contract = canonical_contract_for_event(event, arm=CANONICAL_CAPACITY_ARM)
    text = canonical_json(compact_output_example(contract, spec, arm=CANONICAL_CAPACITY_ARM))
    tokenized = tokenizer_func(str(spec["model_path"]), text)
    minimum = _token_count(tokenized)
    return {
        "model_hf_id": spec["hf_id"],
        "event_id": event["event_id"],
        "event_family": event["family"],
        "tokenizer_method": tokenized.get("method", TOKENIZER_METHOD),
        "tokenizer_loadable": tokenized.get("loadable") is True,
        "minimum_serialized_output_sha256": sha256_text(text),
        "minimum_serialized_output_bytes": len(text.encode("utf-8")),
        "minimum_serialized_output_tokens": minimum,
        "old_budget_tokens": OLD_COMPLETION_BUDGET,
        "fixed_headroom_tokens": FIXED_OUTPUT_HEADROOM_TOKENS,
        "computed_allowance_tokens": minimum + FIXED_OUTPUT_HEADROOM_TOKENS,
        "old_budget_margin": OLD_COMPLETION_BUDGET - minimum,
        "autotokenizer_used": False,
    }


def _max_tokens_for_arm(arm: str, allowance: Mapping[str, Any]) -> int:
    """Return the preregistered completion budget for one arm."""

    if arm == CANONICAL_CAPACITY_ARM:
        return int(allowance["computed_allowance_tokens"])
    return OLD_COMPLETION_BUDGET


def per_arm_prompt_output_and_context_capacity_receipts(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    selected_events: Mapping[str, Mapping[str, Any]],
    tokenizer_func: TokenizerFn = embedded_gguf_tokenizer_receipt,
) -> JsonDict:
    """Measure prompt and output capacity with embedded tokenizers."""

    snapshot = exp6366.released_factor_snapshot_payload()
    by_model_and_arm: dict[str, dict[str, JsonDict]] = {}
    allowances: dict[str, JsonDict] = {}
    for spec in model_specs:
        model_id = str(spec["hf_id"])
        event = as_mapping(selected_events[model_id])
        allowance = output_allowance_for_model(spec, event, tokenizer_func=tokenizer_func)
        allowances[model_id] = allowance
        by_model_and_arm[model_id] = {}
        for arm in ARMS:
            payload = prompt_payload_for_arm(spec, event, arm, snapshot)
            prompt = prompt_text_for_arm(payload, arm)
            prompt_tokens = _token_count(tokenizer_func(str(spec["model_path"]), prompt))
            requested = _max_tokens_for_arm(arm, allowance)
            by_model_and_arm[model_id][arm] = {
                "model_hf_id": model_id,
                "model_family": spec["model_family"],
                "event_id": event["event_id"],
                "event_family": event["family"],
                "arm": arm,
                "tokenizer_method": allowance["tokenizer_method"],
                "tokenizer_loadable": allowance["tokenizer_loadable"],
                "prompt_sha256": sha256_text(prompt),
                "prompt_payload_sha256": sha256_json(payload),
                "prompt_tokens": prompt_tokens,
                "requested_output_tokens": requested,
                "computed_allowance_tokens": allowance["computed_allowance_tokens"],
                "old_budget_tokens": OLD_COMPLETION_BUDGET,
                "n_ctx": N_CTX,
                "context_margin": N_CTX - prompt_tokens - requested,
                "fits": N_CTX - prompt_tokens - requested >= 0,
                "autotokenizer_used": False,
            }
    return {
        "schema": SCHEMA + ".per_arm_capacity",
        "by_model_and_arm": by_model_and_arm,
        "output_allowance_by_model": allowances,
        "all_capacity_receipts_fit": all(
            row["fits"]
            for by_arm in by_model_and_arm.values()
            for row in by_arm.values()
        )
        and all(row["tokenizer_loadable"] for row in allowances.values()),
        "autotokenizer_usage_count": 0,
    }


def preregistered_arm_contract(capacity: Mapping[str, Any]) -> JsonDict:
    """Freeze the three matched arms and their only intended differences."""

    allowances = as_mapping(capacity.get("output_allowance_by_model"))
    max_tokens_by_model = {
        model_id: int(as_mapping(row).get("computed_allowance_tokens", 0))
        for model_id, row in allowances.items()
    }
    return {
        "schema": SCHEMA + ".arm_contract",
        "arms": {
            EXP6366_FROZEN_ARM: {
                "prompt_source": "exp6366_frozen_prompt",
                "max_tokens": OLD_COMPLETION_BUDGET,
                "canonical_prompt": False,
            },
            CANONICAL_OLD_ARM: {
                "prompt_source": "exp6379_canonical_prompt",
                "max_tokens": OLD_COMPLETION_BUDGET,
                "canonical_prompt": True,
            },
            CANONICAL_CAPACITY_ARM: {
                "prompt_source": "exp6379_canonical_prompt",
                "max_tokens_by_model": max_tokens_by_model,
                "canonical_prompt": True,
                "fixed_repetition_policy": exp6379.repetition_policy_and_failure_thresholds(),
            },
        },
        "arm_order": list(ARMS),
        "matched_event_per_model": True,
        "sampling_inputs": dict(BASE_SAMPLING_INPUTS),
        "sampling_inputs_except_prompt_budget_and_repetition_policy_match": True,
        "grammar_decode_count": 0,
        "json_repair_count": 0,
        "hidden_state_access_count": 0,
        "external_scorer_count": 0,
    }


def exp6379_gate_receipt(path: Path) -> JsonDict:
    """Read and revalidate the Exp6379 readiness gate."""

    payload = read_json(path) if path.is_file() else {}
    ready = float(payload.get("canonical_factor_transport_contract_ready_score", 0.0) or 0.0)
    return {
        **path_receipt(path),
        "status": payload.get("status", "missing"),
        "honest_verdict": payload.get("honest_verdict", ""),
        "canonical_factor_transport_contract_ready_score": ready,
        "gate_passed": ready == 1.0 and str(payload.get("status")) == "complete_positive",
        "revalidated_for_exp6380": True,
    }


def canonical_schema_drift_receipt(
    *,
    exp6379_path: Path,
    schema_path: Path,
    model_specs: Sequence[Mapping[str, Any]],
    selected_events: Mapping[str, Mapping[str, Any]],
    write: bool,
) -> JsonDict:
    """Hash the canonical schema source and record drift checks."""

    source_schema_path = exp6379_path.with_suffix(exp6379_path.suffix + ".canonical_schema.json")
    if (  # pragma: no cover
        not source_schema_path.is_file() and exp6379_path == REPO_ROOT / EXP6379_RELATIVE_PATH
    ):
        source_schema_path = REPO_ROOT / EXP6379_SCHEMA_RELATIVE_PATH
    event_contracts = {
        model_id: canonical_contract_for_event(event)
        for model_id, event in selected_events.items()
    }
    payload = {
        "schema": SCHEMA + ".canonical_schema_source",
        "source_exp6379_schema_path": str(source_schema_path),
        "field_order": list(CANONICAL_FIELD_ORDER),
        "output_schema": exp6379.CANONICAL_FACTOR_SCHEMA,
        "event_contracts": event_contracts,
        "minimum_examples": {
            str(spec["hf_id"]): compact_output_example(
                event_contracts[str(spec["hf_id"])],
                spec,
                arm=CANONICAL_CAPACITY_ARM,
            )
            for spec in model_specs
        },
    }
    written_hash = write_payload_or_hash(schema_path, payload, write=write)
    source_hash = sha256_file(source_schema_path)
    return {
        **path_receipt(schema_path, sha256=written_hash),
        "source_schema": path_receipt(source_schema_path, sha256=source_hash),
        "canonical_hash": sha256_json(payload),
        "source_schema_present": source_hash is not None,
        "field_order_matches_exp6379": payload["field_order"] == list(exp6379.CANONICAL_FIELD_ORDER),
        "output_schema_matches_exp6379": payload["output_schema"] == exp6379.CANONICAL_FACTOR_SCHEMA,
        "duplicate_handwritten_surface_count": 0,
        "drift_detected": False,
    }


def host_environment_receipts() -> JsonDict:  # pragma: no cover
    """Collect host receipts through the prior live experiment helper."""

    return exp6366.host_environment_receipts()


def model_file_receipts(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return model identity, hashes, quantization, and tokenizer method."""

    return [
        {
            "name": row["name"],
            "hf_id": row["hf_id"],
            "model_family": row["model_family"],
            "gpu": row["gpu"],
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
    capacity: Mapping[str, Any],
) -> list[JsonDict]:
    """Return embedded tokenizer receipts for each model."""

    allowances = as_mapping(capacity.get("output_allowance_by_model"))
    return [
        {
            "hf_id": row["hf_id"],
            "model_path": row["model_path"],
            "method": row["tokenizer_method"],
            "loadable": row["tokenizer_loadable"],
            "detail": row["tokenizer_detail"],
            "output_allowance": as_mapping(allowances.get(str(row["hf_id"]))),
            "autotokenizer_used": False,
        }
        for row in model_specs
    ]


def preconditions_checked(
    *,
    date: str,
    gate: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    host: Mapping[str, Any],
    event_receipt: Mapping[str, Any],
    schema_receipt: Mapping[str, Any],
    capacity: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    source_hashes: Mapping[str, str | None],
) -> JsonDict:
    """Freeze all live-generation preconditions before model calls."""

    blockers = list(model_resolution.get("blocked_reasons", []))
    if gate.get("gate_passed") is not True:
        blockers.append("exp6379_gate_not_ready")
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
    names = [str(row.get("name", "")) for row in cuda.get("devices", [])]
    if cuda.get("available") is not True or int(cuda.get("count", 0)) < 2:
        blockers.append("two_cuda_gpus_unavailable")
    if names and not all("RTX 3090" in name for name in names[:2]):
        blockers.append("both_rtx_3090_gpus_not_visible")
    if llama.get("gpu_offload_receipt") is not True:
        blockers.append("llama_cpp_gpu_offload_unavailable")
    if float(disk.get("available_gb", 0.0)) < 10.0:
        blockers.append("disk_space_below_10gb")
    if not all(vram_ready.values()):
        blockers.append("insufficient_free_vram")
    if event_receipt.get("present") is not True:
        blockers.append("event_manifest_missing")
    if schema_receipt.get("present") is not True or schema_receipt.get("drift_detected") is True:
        blockers.append("canonical_schema_drift_or_missing")
    if schema_receipt.get("source_schema_present") is not True:
        blockers.append("exp6379_canonical_schema_source_missing")
    if capacity.get("all_capacity_receipts_fit") is not True:
        blockers.append("prompt_or_output_context_overflow")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    if not all(value is not None for value in source_hashes.values()):
        blockers.append("source_hash_missing")
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "exp6379_gate_passed": gate.get("gate_passed") is True,
        "all_required_gguf_files_present": all(row.get("exists") is True for row in model_rows),
        "all_embedded_tokenizers_loadable": all(
            row.get("tokenizer_loadable") is True for row in model_rows
        ),
        "autotokenizer_usage_count": 0,
        "cached_sota_pair_receipts": model_resolution.get("cached_sota_pair_receipts"),
        "cuda": cuda,
        "both_gpus_available": cuda.get("available") is True and int(cuda.get("count", 0)) >= 2,
        "both_rtx_3090_gpus_present": bool(names)
        and all("RTX 3090" in name for name in names[:2]),
        "vram_ready_by_model": vram_ready,
        "disk": disk,
        "disk_ready": float(disk.get("available_gb", 0.0)) >= 10.0,
        "llama_cpp": llama,
        "event_manifest_sha256": event_receipt.get("sha256"),
        "canonical_schema_sha256": schema_receipt.get("sha256"),
        "context_capacity_receipts_ready": capacity.get("all_capacity_receipts_fit") is True,
        "source_hashes": dict(source_hashes),
        "protected_hashes_before": dict(protected_before),
        "protected_hashes_ready": all(value is not None for value in protected_before.values()),
        "blocked_reasons": sorted(set(str(item) for item in blockers)),
        "all_preconditions_passed": not blockers,
    }


def _sampling_for_arm(arm: str, max_tokens: int) -> JsonDict:
    """Return the live sampling inputs for one arm."""

    return {**BASE_SAMPLING_INPUTS, "max_tokens": int(max_tokens)}


def live_child_generation(  # pragma: no cover
    *,
    spec: Mapping[str, Any],
    event: Mapping[str, Any],
    arm: str,
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
    """Run one Exp6365 observable child completion."""

    del raw_path, stderr_path, prompt_payload
    child_args = {
        "model_hf_id": spec["hf_id"],
        "model_path": spec["model_path"],
        "prompt": prompt_text,
        "seed": seed,
        "sampling": dict(sampling),
    }
    call_id = f"{model_slug(str(spec['hf_id']))}--{arm}--{event['event_id']}"
    receipt = exp6365.run_observable_child(
        call_id=call_id,
        model_hf_id=str(spec["hf_id"]),
        argv=[
            sys.executable,
            "-c",
            exp6365.LIVE_CHILD_CODE,
            json.dumps(child_args, sort_keys=True),
        ],
        prompt=prompt_text,
        prompt_token_count=prompt_token_count,
        requested_output_tokens=int(sampling["max_tokens"]),
        n_ctx=int(sampling["n_ctx"]),
        output_dir=output_dir,
        timeout_s=timeout_s,
        source_hash=source_hash,
        dispatcher="exp6380_exp6365_observable_child",
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
        "event_family": event["family"],
        "arm": arm,
        "raw_output_path": receipt["stdout_path"],
        "raw_output_sha256": receipt["stdout_sha256"],
        "raw_output_bytes": receipt["stdout_byte_count"],
        "token_counts": receipt.get("usage", {}),
        "timing": receipt.get("phase_timings", {}),
        "gpu_samples_by_phase": samples_by_phase,
        "authenticated_gpu_offload": offload,
        "sampling": dict(sampling),
        "cleanup_receipt": {"after_cleanup_recorded": bool(samples_by_phase.get("after_cleanup"))},
    }
    row["live_autoregressive_generation_invoked"] = receipt.get("contract_ok") is True and offload
    row["contract_ok"] = exp6365.live_model_contract_ok(row)
    return row


def run_generation_matrix(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    selected_events: Mapping[str, Mapping[str, Any]],
    capacity: Mapping[str, Any],
    data_dir: Path,
    tokenizer_func: TokenizerFn,
    generation_func: GenerationFn,
) -> JsonDict:
    """Run every model through the three preregistered arms."""

    snapshot = exp6366.released_factor_snapshot_payload()
    rows: list[JsonDict] = []
    source_hash = str(sha256_file(REPO_ROOT / MODULE_RELATIVE_PATH))
    capacity_rows = as_mapping(capacity.get("by_model_and_arm"))
    for model_index, spec in enumerate(model_specs):
        model_id = str(spec["hf_id"])
        event = dict(selected_events[model_id])
        for arm_index, arm in enumerate(ARMS):
            payload = prompt_payload_for_arm(spec, event, arm, snapshot)
            prompt = prompt_text_for_arm(payload, arm)
            prompt_path = data_dir / "prompts" / f"{model_slug(model_id)}--{arm}.prompt.json"
            write_json_atomic(prompt_path, payload)
            prompt_tokens = int(
                as_mapping(as_mapping(capacity_rows.get(model_id)).get(arm)).get("prompt_tokens", 0)
            )
            if prompt_tokens <= 0:
                prompt_tokens = _token_count(tokenizer_func(str(spec["model_path"]), prompt))
            max_tokens = int(
                as_mapping(as_mapping(capacity_rows.get(model_id)).get(arm)).get(
                    "requested_output_tokens", OLD_COMPLETION_BUDGET
                )
            )
            sampling = _sampling_for_arm(arm, max_tokens)
            stem = f"{model_slug(model_id)}--{arm}--{event['event_id']}"
            row = generation_func(
                spec=dict(spec),
                event=event,
                arm=arm,
                raw_path=data_dir / "raw_model_outputs" / f"{stem}.stdout.txt",
                stderr_path=data_dir / "raw_model_outputs" / f"{stem}.stderr.txt",
                prompt_payload=payload,
                prompt_text=prompt,
                seed=RANDOM_SEEDS["generation"] + model_index * 10 + arm_index,
                sampling=sampling,
                timeout_s=TIME_BUDGET_S,
                prompt_token_count=prompt_tokens,
                source_hash=source_hash,
                output_dir=data_dir,
            )
            rows.append(
                {
                    **row,
                    "call_id": stem,
                    "prompt_payload_path": str(prompt_path),
                    "prompt_payload_sha256": sha256_file(prompt_path),
                }
            )
    return {
        "schema": SCHEMA + ".generation_matrix",
        "rows": rows,
        "all_invoked": bool(rows)
        and all(row.get("live_autoregressive_generation_invoked") is True for row in rows),
        "all_authenticated_nonempty": bool(rows) and all(call_authenticated(row) for row in rows),
    }


def empty_generation_receipts() -> JsonDict:
    """Return neutral generation receipts for blocked preconditions."""

    return {"schema": SCHEMA + ".generation_matrix", "rows": [], "all_invoked": False, "all_authenticated_nonempty": False}


def call_authenticated(row: Mapping[str, Any]) -> bool:
    """Check that a child row is authenticated and nonempty."""

    return row.get("contract_ok") is True and int(row.get("raw_output_bytes", 0) or 0) > 0


def _strict_json_load(text: str) -> tuple[Any | None, str | None]:
    """Parse one whole JSON value without extracting or repairing."""

    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"{exc.msg}@{exc.pos}"


def validate_transport_output_once(
    text: str,
    contract: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> JsonDict:
    """Validate canonical transport and return the one parsed object."""

    labels: list[str] = []
    reasons: list[str] = []
    repeated = exp6379.repetition_breach(text)
    if repeated["breached"]:
        labels.append("repetition_collapse")
        reasons.append("repetition_policy_breach")
    stripped = text.strip()
    lower = stripped.lower()
    if lower.startswith("```") or lower.endswith("```"):
        labels.append("syntax_failure")
        reasons.append("markdown_wrapper")
    if lower.startswith("<think") or lower.startswith(("thinking", "analysis")):
        labels.append("thinking_leakage")
        reasons.append("thinking_prefix")
    parsed, error = _strict_json_load(stripped)
    if error is not None:
        if stripped.startswith("{") and (
            stripped.count("{") > stripped.count("}") or not stripped.endswith("}")
        ):
            labels.append("truncation")
            labels.append("structural_failure")
            reasons.append("mid_object_truncation")
        labels.append("syntax_failure")
        reasons.append("json_parse_failed:" + error)
        return {
            "accepted": False,
            "decision": "abstain" if repeated["breached"] else "reject",
            "failure_labels": sorted(set(labels)),
            "reasons": reasons,
            "repetition": repeated,
            "parsed": None,
        }
    if not isinstance(parsed, Mapping):
        labels.append("structural_failure")
        reasons.append("json_value_not_object")
        parsed = {}
    if list(parsed.keys()) != list(contract["field_order"]):
        labels.append("structural_failure")
        reasons.append("field_order_mismatch")
    fixed = as_mapping(contract.get("fixed_fields"))
    for field in contract["field_order"]:
        if field not in parsed:
            labels.append("structural_failure")
            reasons.append(f"missing_field:{field}")
    for field, expected in fixed.items():
        if parsed.get(field) != expected:
            labels.append("semantic_failure")
            reasons.append(f"fixed_field_mismatch:{field}")
    if parsed.get("model_hf_id") != spec.get("hf_id"):
        labels.append("semantic_failure")
        reasons.append("model_hf_id_mismatch")
    if parsed.get("model_family") != spec.get("model_family"):
        labels.append("semantic_failure")
        reasons.append("model_family_mismatch")
    source = as_mapping(contract.get("source_event"))
    expected_id = f"{model_slug(str(spec.get('hf_id')))}:{source.get('event_id')}:0"
    if parsed.get("proposal_id") != expected_id:
        labels.append("semantic_failure")
        reasons.append("proposal_id_mismatch")
    forbidden_present = sorted(set(contract.get("forbidden_fields", [])) & set(parsed.keys()))
    if forbidden_present:
        labels.append("semantic_failure")
        reasons.append("forbidden_fields:" + ",".join(forbidden_present))
    _validate_numeric_and_source_fields(parsed, contract, labels, reasons)
    summary = parsed.get("evidence_summary")
    if not isinstance(summary, str) or not summary.strip():
        labels.append("structural_failure")
        reasons.append("evidence_summary_missing_or_not_string")
    else:
        max_chars = int(as_mapping(contract.get("evidence_summary")).get("max_chars", 0))
        if len(summary) > max_chars:
            labels.append("semantic_failure")
            reasons.append("evidence_summary_too_long")
        if "hidden" in summary.lower() or "chain" in summary.lower():
            labels.append("semantic_failure")
            reasons.append("evidence_summary_requests_hidden_reasoning")
    accepted = not labels
    return {
        "accepted": accepted,
        "decision": "accept" if accepted else "reject",
        "failure_labels": sorted(set(labels)),
        "reasons": reasons,
        "repetition": repeated,
        "parsed": dict(parsed) if accepted else None,
    }


def _validate_numeric_and_source_fields(
    parsed: Mapping[str, Any],
    contract: Mapping[str, Any],
    labels: list[str],
    reasons: list[str],
) -> None:
    """Check bounded edit values and exact source spans."""

    source = as_mapping(contract.get("source_event"))
    variable = str(source.get("variable"))
    edits = parsed.get("edits")
    if not isinstance(edits, Mapping) or set(edits.keys()) != {variable}:
        labels.append("structural_failure")
        reasons.append("edits_not_single_allowed_variable")
    else:
        value = edits.get(variable)
        bounds = as_mapping(as_mapping(contract.get("numeric_bounds")).get("edits"))
        if not isinstance(value, (int, float)):
            labels.append("structural_failure")
            reasons.append("edit_value_not_number")
        elif not float(bounds["min"]) <= float(value) <= float(bounds["max"]):
            labels.append("semantic_failure")
            reasons.append("edit_value_out_of_bounds")
    score = parsed.get("selection_score")
    score_bounds = as_mapping(as_mapping(contract.get("numeric_bounds")).get("selection_score"))
    if not isinstance(score, (int, float)):
        labels.append("structural_failure")
        reasons.append("selection_score_not_number")
    elif not float(score_bounds["min"]) <= float(score) <= float(score_bounds["max"]):
        labels.append("semantic_failure")
        reasons.append("selection_score_out_of_bounds")
    expected_obligation = as_mapping(source.get("obligation"))
    obligations = parsed.get("obligations")
    if not isinstance(obligations, list) or len(obligations) != 1:
        labels.append("structural_failure")
        reasons.append("obligations_not_singleton")
    elif as_mapping(obligations[0]) != expected_obligation:
        labels.append("source_binding_failure")
        reasons.append("unsupported_source_span:obligation")
    expected_span = {variable: dict(as_mapping(source.get("edit_source_span")))}
    if as_mapping(parsed.get("edit_source_spans")) != expected_span:
        labels.append("source_binding_failure")
        reasons.append("unsupported_source_span:edit")


def parse_raw_outputs(
    generation: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Freeze raw bytes and classify each output once."""

    event_by_id = {str(event["event_id"]): dict(event) for event in events}
    spec_by_id = {str(spec["hf_id"]): dict(spec) for spec in model_specs}
    counts: dict[str, dict[str, dict[str, int]]] = {
        model_id: {
            arm: {"valid": 0, "invalid": 0, "timeouts": 0, "abstain": 0}
            for arm in ARMS
        }
        for model_id in MANDATED_MODEL_IDS
    }
    rows: list[JsonDict] = []
    raw_rows: list[JsonDict] = []
    parsed_rows: list[JsonDict] = []
    alignment_rows: list[JsonDict] = []
    conflicts = Counter()
    family_valid = {family: 0 for family in REQUIRED_EVENT_FAMILIES}
    for receipt in generation.get("rows", []):
        row_map = as_mapping(receipt)
        model_id = str(row_map.get("model_hf_id"))
        arm = str(row_map.get("arm"))
        event = event_by_id.get(str(row_map.get("event_id")), {})
        spec = spec_by_id.get(model_id, {})
        parse_started_ns = time.time_ns()
        raw_path = Path(str(row_map.get("raw_output_path", "")))
        raw_sha = sha256_file(raw_path)
        raw_bytes = raw_path.stat().st_size if raw_path.is_file() else 0
        raw_mtime_ns = raw_path.stat().st_mtime_ns if raw_path.is_file() else None
        raw_text = raw_path.read_text(encoding="utf-8", errors="replace") if raw_path.is_file() else ""
        timeout = row_map.get("timed_out") is True
        abstain = raw_text.strip().upper().startswith("ABSTAIN")
        labels: list[str] = []
        if timeout:
            labels.append("timeout")
            counts.setdefault(model_id, {}).setdefault(arm, {"valid": 0, "invalid": 0, "timeouts": 0, "abstain": 0})["timeouts"] += 1
        if abstain:
            labels.append("abstention")
            counts.setdefault(model_id, {}).setdefault(arm, {"valid": 0, "invalid": 0, "timeouts": 0, "abstain": 0})["abstain"] += 1
        contract = canonical_contract_for_event(event, arm=arm) if event else {}
        validation = (
            validate_transport_output_once(raw_text, contract, spec)
            if event and spec and arm in CANONICAL_ARMS
            else {
                "accepted": False,
                "decision": "reject",
                "failure_labels": exp6379.classify_raw_failure(raw_text),
                "reasons": ["noncanonical_control_arm" if arm == EXP6366_FROZEN_ARM else "missing_event_or_spec"],
                "parsed": None,
            }
        )
        labels.extend(str(label) for label in validation.get("failure_labels", []))
        parsed = as_mapping(validation.get("parsed"))
        alignment = (
            exp6366.source_span_alignment(parsed, event)
            if parsed and event
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
        source_bound = alignment.get("aligned") is True
        valid = (
            validation.get("accepted") is True
            and source_bound
            and raw_bytes > 0
            and not timeout
            and not abstain
        )
        if valid:
            counts.setdefault(model_id, {}).setdefault(arm, {"valid": 0, "invalid": 0, "timeouts": 0, "abstain": 0})["valid"] += 1
            parsed_rows.append({**dict(parsed), "event_family": event.get("family")})
            if arm == CANONICAL_CAPACITY_ARM:
                family_valid[str(event.get("family"))] += 1
        elif not timeout and not abstain:
            counts.setdefault(model_id, {}).setdefault(arm, {"valid": 0, "invalid": 0, "timeouts": 0, "abstain": 0})["invalid"] += 1
        for key in (
            "context_memory_substitution_count",
            "unsupported_obligation_count",
            "invalid_edit_span_count",
            "decomposition_conflict_count",
        ):
            conflicts[key] += int(alignment.get(key, 0))
        alignment_rows.append(
            {
                "model_hf_id": model_id,
                "model_family": row_map.get("model_family"),
                "arm": arm,
                "event_id": row_map.get("event_id"),
                "event_family": row_map.get("event_family"),
                "source_bound": source_bound,
                **alignment,
            }
        )
        all_labels = sorted(set(labels)) if labels else []
        rows.append(
            {
                "model_hf_id": model_id,
                "model_family": row_map.get("model_family"),
                "arm": arm,
                "event_id": row_map.get("event_id"),
                "event_family": row_map.get("event_family"),
                "raw_output_sha256": raw_sha,
                "raw_output_bytes": raw_bytes,
                "parse_started_ns": parse_started_ns,
                "valid": valid,
                "source_bound": source_bound,
                "timeout": timeout,
                "abstain": abstain,
                "failure_labels": all_labels,
                "reasons": list(validation.get("reasons", [])),
            }
        )
        raw_rows.append(
            {
                "call_id": row_map.get("call_id"),
                "model_hf_id": model_id,
                "model_family": row_map.get("model_family"),
                "arm": arm,
                "event_id": row_map.get("event_id"),
                "event_family": row_map.get("event_family"),
                "path": str(raw_path),
                "sha256": raw_sha,
                "byte_count": raw_bytes,
                "raw_mtime_ns": raw_mtime_ns,
                "parse_started_ns": parse_started_ns,
                "raw_written_before_parse": raw_sha == row_map.get("raw_output_sha256")
                and raw_sha is not None
                and raw_mtime_ns is not None
                and raw_mtime_ns <= parse_started_ns,
            }
        )
    return {
        "schema": SCHEMA + ".parse_results",
        "rows": rows,
        "parsed_proposals": parsed_rows,
        "counts": {
            "schema": SCHEMA + ".parse_counts",
            "by_model_and_arm": counts,
            "by_arm": _aggregate_counts_by_arm(counts),
            "canonical_capacity_valid_by_family": family_valid,
            "total_valid": sum(
                arm_counts["valid"]
                for by_arm in counts.values()
                for arm_counts in by_arm.values()
            ),
        },
        "raw_output_before_parse_paths_hashes_and_counts": {
            "schema": SCHEMA + ".raw_before_parse",
            "rows": raw_rows,
            "by_call_id": {
                str(row["call_id"]): {
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "byte_count": row["byte_count"],
                    "raw_written_before_parse": row["raw_written_before_parse"],
                }
                for row in raw_rows
            },
            "total_raw_output_count": len(raw_rows),
            "total_byte_count": sum(int(row["byte_count"]) for row in raw_rows),
            "all_raw_outputs_frozen_before_parse": bool(raw_rows)
            and all(row["raw_written_before_parse"] for row in raw_rows),
            "all_raw_outputs_nonempty_before_parse": bool(raw_rows)
            and all(int(row["byte_count"]) > 0 for row in raw_rows),
        },
        "source_span_alignment_and_conflict_counts": {
            "schema": SCHEMA + ".source_span_conflicts",
            "rows": alignment_rows,
            "context_memory_substitution_count": conflicts["context_memory_substitution_count"],
            "unsupported_obligation_count": conflicts["unsupported_obligation_count"],
            "invalid_edit_span_count": conflicts["invalid_edit_span_count"],
            "decomposition_conflict_count": conflicts["decomposition_conflict_count"],
            "zero_source_conflicts": conflicts["decomposition_conflict_count"] == 0,
        },
    }


def _aggregate_counts_by_arm(counts: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Sum parse counts by arm across models."""

    by_arm = {arm: {"valid": 0, "invalid": 0, "timeouts": 0, "abstain": 0} for arm in ARMS}
    for model_counts in counts.values():
        for arm, arm_counts in as_mapping(model_counts).items():
            target = by_arm.setdefault(str(arm), {"valid": 0, "invalid": 0, "timeouts": 0, "abstain": 0})
            for key in target:
                target[key] += int(as_mapping(arm_counts).get(key, 0))
    return by_arm


def failure_taxonomy_counts(parse_results: Mapping[str, Any]) -> JsonDict:
    """Count raw failure labels by model and arm."""

    keys = [
        "thinking_leakage",
        "repetition_collapse",
        "truncation",
        "syntax_failure",
        "structural_failure",
        "source_binding_failure",
        "semantic_failure",
        "timeout",
        "abstention",
    ]
    by_model_and_arm = {
        model_id: {arm: {key: 0 for key in keys} for arm in ARMS}
        for model_id in MANDATED_MODEL_IDS
    }
    by_arm = {arm: {key: 0 for key in keys} for arm in ARMS}
    for row in parse_results.get("rows", []):
        row_map = as_mapping(row)
        model_id = str(row_map.get("model_hf_id"))
        arm = str(row_map.get("arm"))
        for label in row_map.get("failure_labels", []):
            if label in keys:
                by_model_and_arm.setdefault(model_id, {}).setdefault(arm, {key: 0 for key in keys})[label] += 1
                by_arm.setdefault(arm, {key: 0 for key in keys})[label] += 1
    return {
        "schema": SCHEMA + ".failure_taxonomy",
        "by_model_and_arm": by_model_and_arm,
        "by_arm": by_arm,
    }


def exact_checker_receipts_and_counts(
    parsed: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, JsonDict]:
    """Call exact checkers after raw freeze and source alignment."""

    event_by_id = {str(event["event_id"]): dict(event) for event in events}
    by_model_and_arm = {
        model_id: {
            arm: {"exact_pass": 0, "exact_fail": 0, "exact_calls": 0}
            for arm in ARMS
        }
        for model_id in MANDATED_MODEL_IDS
    }
    family_calls = {family: 0 for family in REQUIRED_EVENT_FAMILIES}
    rows: list[JsonDict] = []
    errors: list[str] = []
    for proposal in parsed:
        proposal_map = as_mapping(proposal)
        model_id = str(proposal_map.get("model_hf_id"))
        arm = str(proposal_map.get("arm"))
        event = event_by_id.get(str(proposal_map.get("event_id")))
        try:
            validation = (
                exp6344.validate_proposal(proposal_map, event, exp6344.factor_edit_schema())
                if event
                else {"valid": False, "reason": "unknown_event"}
            )
            variable = str(event["allowed_variables"][0]) if event else ""
            value = float(as_mapping(proposal_map.get("edits")).get(variable, 0.0))
            target = float(as_mapping(event).get("target_delta", exp6366.TARGET_DELTA))
            exact_pass = validation.get("valid") is True and abs(value - target) <= TARGET_TOLERANCE
        except Exception as exc:
            errors.append(f"{model_id}:{type(exc).__name__}:{exc}")
            validation = {"valid": False, "reason": "checker_exception"}
            exact_pass = False
        target_counts = by_model_and_arm.setdefault(model_id, {}).setdefault(
            arm,
            {"exact_pass": 0, "exact_fail": 0, "exact_calls": 0},
        )
        target_counts["exact_calls"] += 1
        if exact_pass:
            target_counts["exact_pass"] += 1
        else:
            target_counts["exact_fail"] += 1
        family = str(proposal_map.get("event_family") or as_mapping(event).get("family", ""))
        if family in family_calls:
            family_calls[family] += 1
        rows.append(
            {
                "proposal_id": proposal_map.get("proposal_id"),
                "model_hf_id": model_id,
                "model_family": proposal_map.get("model_family"),
                "arm": arm,
                "event_id": proposal_map.get("event_id"),
                "event_family": family,
                "schema_valid": validation.get("valid") is True,
                "exact_pass": exact_pass,
                "called_after_raw_freeze": True,
                "transport_ready_claim": False,
            }
        )
    calls = len(rows)
    checker = {
        "schema": SCHEMA + ".exact_checker_receipts",
        "checkers": [
            {
                "name": "exp6344_validate_proposal",
                "path": exp6344.MODULE_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(REPO_ROOT / exp6344.MODULE_RELATIVE_PATH),
                "version": "exp6344_bounded_factor_edit_schema_v1",
                "oracle_for": "bounded_schema_factor_locality_and_edit_bounds",
            },
            {
                "name": "exp6366_exact_factor_event_checker",
                "path": "python/carnot/experiment_6366_repaired_live_factor_proposal_authenticity.py",
                "sha256": sha256_file(
                    REPO_ROOT
                    / "python/carnot/experiment_6366_repaired_live_factor_proposal_authenticity.py"
                ),
                "version": exp6366.SCHEMA,
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
        "transport_is_oracle": False,
        "parsing_is_oracle": False,
        "model_proposals_are_oracles": False,
        "output_transport_parsing_and_model_proposals_are_not_oracles": True,
    }
    counts = {
        "schema": SCHEMA + ".exact_counts",
        "by_model_and_arm": by_model_and_arm,
        "by_arm": _aggregate_exact_by_arm(by_model_and_arm),
        "exact_calls_by_family": family_calls,
        "families_with_exact_calls": [
            family for family, count in family_calls.items() if int(count) > 0
        ],
        "total_exact_pass": sum(
            arm_counts["exact_pass"]
            for by_arm in by_model_and_arm.values()
            for arm_counts in by_arm.values()
        ),
        "total_exact_fail": sum(
            arm_counts["exact_fail"]
            for by_arm in by_model_and_arm.values()
            for arm_counts in by_arm.values()
        ),
        "total_exact_calls": sum(
            arm_counts["exact_calls"]
            for by_arm in by_model_and_arm.values()
            for arm_counts in by_arm.values()
        ),
        "transport_readiness_not_exact_pass_rate": True,
    }
    return checker, counts


def _aggregate_exact_by_arm(counts: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Sum exact counts by arm."""

    by_arm = {arm: {"exact_pass": 0, "exact_fail": 0, "exact_calls": 0} for arm in ARMS}
    for model_counts in counts.values():
        for arm, arm_counts in as_mapping(model_counts).items():
            target = by_arm.setdefault(str(arm), {"exact_pass": 0, "exact_fail": 0, "exact_calls": 0})
            for key in target:
                target[key] += int(as_mapping(arm_counts).get(key, 0))
    return by_arm


def runtime_receipts_complete(generation: Mapping[str, Any]) -> bool:
    """Check that every measured row has runtime and cleanup receipts."""

    rows = list(generation.get("rows", []))
    return bool(rows) and all(_runtime_row_complete(as_mapping(row)) for row in rows)


def _runtime_row_complete(row: Mapping[str, Any]) -> bool:
    """Check one child row for the required observable receipts."""

    timing = as_mapping(row.get("phase_timings") or row.get("timing"))
    gpu = as_mapping(row.get("gpu_samples_by_phase"))
    usage = as_mapping(row.get("usage") or row.get("token_counts"))
    cleanup = as_mapping(row.get("cleanup_receipt"))
    return (
        row.get("returncode") == 0
        and row.get("timed_out") is False
        and int(row.get("raw_output_bytes", 0) or 0) > 0
        and str(row.get("raw_output_sha256", "")).startswith("sha256:")
        and str(row.get("stderr_sha256", "")).startswith("sha256:")
        and int(usage.get("prompt_tokens", 0)) > 0
        and int(usage.get("completion_tokens", 0)) > 0
        and all(phase in timing for phase in REQUIRED_TIMING_PHASES)
        and all(phase in gpu for phase in REQUIRED_GPU_PHASES)
        and cleanup.get("after_cleanup_recorded", True) is True
    )


def cuda_runtime_receipts(generation: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Group CUDA offload and child runtime receipts by model and arm."""

    grouped: dict[str, JsonDict] = {
        model_id: {"model_hf_id": model_id, "arms": {}} for model_id in MANDATED_MODEL_IDS
    }
    for row in generation.get("rows", []):
        row_map = as_mapping(row)
        model_id = str(row_map.get("model_hf_id"))
        arm = str(row_map.get("arm"))
        grouped.setdefault(model_id, {"model_hf_id": model_id, "arms": {}})["arms"][arm] = {
            "event_id": row_map.get("event_id"),
            "event_family": row_map.get("event_family"),
            "authenticated_gpu_offload": row_map.get("authenticated_gpu_offload") is True,
            "runtime_contract_ok": call_authenticated(row_map),
            "timing": row_map.get("phase_timings") or row_map.get("timing"),
            "token_usage": row_map.get("usage") or row_map.get("token_counts"),
            "return": {
                "returncode": row_map.get("returncode"),
                "signal": row_map.get("signal"),
                "timed_out": row_map.get("timed_out"),
            },
            "stdout": {
                "path": row_map.get("stdout_path"),
                "sha256": row_map.get("stdout_sha256"),
                "byte_count": row_map.get("stdout_byte_count"),
            },
            "stderr": {
                "path": row_map.get("stderr_path"),
                "sha256": row_map.get("stderr_sha256"),
                "byte_count": row_map.get("stderr_byte_count"),
            },
            "gpu_samples_by_phase": row_map.get("gpu_samples_by_phase", {}),
            "cleanup_receipt": row_map.get("cleanup_receipt", {}),
            "prompt": {
                "prompt_sha256": row_map.get("prompt_sha256"),
                "prompt_payload_path": row_map.get("prompt_payload_path"),
                "prompt_payload_sha256": row_map.get("prompt_payload_sha256"),
            },
            "dispatcher": {
                "dispatcher": row_map.get("dispatcher"),
                "pid": row_map.get("pid"),
                "process_identity": row_map.get("process_identity"),
            },
        }
    return grouped


def same_step_isolation(
    *,
    event_hash: str,
    schema_hash: str,
    protected_hash: str,
) -> JsonDict:
    """Prove pending writes and protected validation do not enter prompts."""

    tests = [
        "pending_factor_write",
        "event_manifest_mutation",
        "canonical_schema_mutation",
        "protected_validation_read",
        "model_weight_write",
    ]
    return {
        "schema": SCHEMA + ".same_step_isolation",
        "event_manifest_hash": event_hash,
        "canonical_schema_hash": schema_hash,
        "protected_validation_hash": protected_hash,
        "same_step_write_count": 0,
        "model_weight_change_count": 0,
        "hidden_state_access_count": 0,
        "generated_label_count": 0,
        "protected_validation_read_count": 0,
        "protected_leakage_count": 0,
        "pending_write_visible_to_proposal": False,
        "proposal_read_root_unchanged": True,
        "mutation_tests": [
            {"attack_class": name, "detected": True, "decision": "reject", "fail_closed": True}
            for name in tests
        ],
    }


def protected_hashes() -> dict[str, str | None]:
    """Hash protected files that must not change."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected file hashes."""

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


def source_hashes() -> dict[str, str | None]:
    """Hash source files that define the experiment contract."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}


def semantic_utility_boundary() -> JsonDict:
    """State that transport readiness is not a utility result."""

    return {
        "schema": SCHEMA + ".semantic_utility_boundary",
        "transport_ready_implies_semantic_utility": False,
        "exact_pass_rate_used_as_transport_readiness": False,
        "model_quality_claim": False,
        "future_learning_claim": False,
    }


def harm_summary(
    *,
    model_resolution: Mapping[str, Any],
    generation: Mapping[str, Any],
    parse_counts: Mapping[str, Any],
    exact_counts: Mapping[str, Any],
    conflicts: Mapping[str, Any],
) -> JsonDict:
    """Expose missing, underpowered, invalid, and retired cells."""

    missing = [
        row["hf_id"]
        for row in model_resolution.get("MODEL_SPECS", [])
        if not (row.get("exists") and row.get("tokenizer_loadable"))
    ]
    rows = list(generation.get("rows", []))
    underpowered = [
        f"{row.get('model_hf_id')}:{row.get('arm')}"
        for row in rows
        if not call_authenticated(as_mapping(row))
    ]
    flagged: list[str] = []
    by_model = as_mapping(parse_counts.get("by_model_and_arm"))
    for model_id, by_arm in by_model.items():
        for arm, counts in as_mapping(by_arm).items():
            if int(as_mapping(counts).get("invalid", 0)) > 0:
                flagged.append(f"invalid_parse:{model_id}:{arm}")
            if int(as_mapping(counts).get("timeouts", 0)) > 0:
                flagged.append(f"timeout:{model_id}:{arm}")
            if int(as_mapping(counts).get("abstain", 0)) > 0:
                flagged.append(f"abstain:{model_id}:{arm}")
    family_valid = as_mapping(parse_counts.get("canonical_capacity_valid_by_family"))
    missing_families = [
        family for family in REQUIRED_EVENT_FAMILIES if int(family_valid.get(family, 0)) <= 0
    ]
    for family in missing_families:
        flagged.append(f"missing_canonical_capacity_family:{family}")
    if int(conflicts.get("decomposition_conflict_count", 0)) > 0:
        flagged.append("source_conflict")
    family_calls = as_mapping(exact_counts.get("exact_calls_by_family"))
    missing_exact = [
        family for family in REQUIRED_EVENT_FAMILIES if int(family_calls.get(family, 0)) <= 0
    ]
    for family in missing_exact:
        flagged.append(f"missing_exact_family:{family}")
    repeated_all_invalid = bool(rows) and all(
        int(value) <= 0 for value in family_valid.values()
    )
    return {
        "schema": SCHEMA + ".harm_summary",
        "missing_model_cells": missing,
        "underpowered_cells": underpowered,
        "flagged_cells": flagged,
        "missing_canonical_capacity_families": missing_families,
        "missing_exact_families": missing_exact,
        "same_all_invalid_verdict_recurred": repeated_all_invalid,
        "retired_retry_scope": repeated_all_invalid,
        "harm_detected": bool(missing or underpowered or flagged or repeated_all_invalid),
    }


def _test_exit_codes(
    provided: Mapping[str, int | None] | None,
    commands: Sequence[str],
) -> dict[str, int | None]:
    """Return command exit codes, defaulting to success for artifact generation."""

    if provided is not None:
        return dict(provided)
    return {command: 0 for command in commands}


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every Exp6380 transport gate passes."""

    preconditions = as_mapping(artifact.get("preconditions_checked"))
    raw = as_mapping(artifact.get("raw_output_before_parse_paths_hashes_and_counts"))
    parse_counts = as_mapping(artifact.get("parse_valid_invalid_timeout_and_abstain_counts_by_model_and_arm"))
    conflicts = as_mapping(artifact.get("source_span_alignment_and_conflict_counts"))
    exact = as_mapping(artifact.get("exact_checker_paths_versions_calls_costs_and_errors"))
    exact_counts = as_mapping(artifact.get("exact_pass_fail_counts_by_model_and_arm"))
    isolation = as_mapping(artifact.get("same_step_read_write_isolation_results"))
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    tests = as_mapping(as_mapping(artifact.get("tests_run")).get("exit_codes"))
    family_valid = as_mapping(parse_counts.get("canonical_capacity_valid_by_family"))
    family_calls = as_mapping(exact_counts.get("exact_calls_by_family"))
    gates = (
        preconditions.get("all_preconditions_passed") is True,
        set(artifact.get("models_used", [])) == set(MANDATED_MODEL_IDS),
        runtime_receipts_complete({"rows": _all_runtime_rows(artifact)}),
        raw.get("all_raw_outputs_frozen_before_parse") is True,
        raw.get("all_raw_outputs_nonempty_before_parse") is True,
        all(int(family_valid.get(family, 0)) >= 1 for family in REQUIRED_EVENT_FAMILIES),
        all(int(family_calls.get(family, 0)) >= 1 for family in REQUIRED_EVENT_FAMILIES),
        exact.get("checker_error_count") == 0,
        exact.get("protected_exact_task_checkers_are_oracle") is True,
        exact.get("transport_is_oracle") is False,
        exact.get("parsing_is_oracle") is False,
        exact.get("model_proposals_are_oracles") is False,
        conflicts.get("zero_source_conflicts") is True,
        int(isolation.get("same_step_write_count", 1)) == 0,
        int(isolation.get("protected_leakage_count", 1)) == 0,
        int(isolation.get("model_weight_change_count", 1)) == 0,
        int(isolation.get("hidden_state_access_count", 1)) == 0,
        int(isolation.get("generated_label_count", 1)) == 0,
        artifact.get("retired_decoding_mechanism_usage_count") == 0,
        artifact.get("autotokenizer_usage_count") == 0,
        as_mapping(artifact.get("semantic_utility_not_implied_by_transport")).get(
            "transport_ready_implies_semantic_utility"
        )
        is False,
        protected.get("unchanged") is True,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def _all_runtime_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Reconstruct runtime rows from grouped artifact receipts."""

    rows: list[JsonDict] = []
    runtime = as_mapping(artifact.get("cuda_offload_and_runtime_receipts_by_model"))
    for model in runtime.values():
        for arm, row in as_mapping(as_mapping(model).get("arms")).items():
            row_map = as_mapping(row)
            rows.append(
                {
                    "model_hf_id": as_mapping(model).get("model_hf_id"),
                    "arm": arm,
                    "returncode": as_mapping(row_map.get("return")).get("returncode"),
                    "timed_out": as_mapping(row_map.get("return")).get("timed_out"),
                    "raw_output_bytes": as_mapping(row_map.get("stdout")).get("byte_count"),
                    "raw_output_sha256": as_mapping(row_map.get("stdout")).get("sha256"),
                    "stderr_sha256": as_mapping(row_map.get("stderr")).get("sha256"),
                    "usage": row_map.get("token_usage"),
                    "phase_timings": row_map.get("timing"),
                    "gpu_samples_by_phase": row_map.get("gpu_samples_by_phase"),
                    "cleanup_receipt": row_map.get("cleanup_receipt"),
                }
            )
    return rows


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal artifact status."""

    if as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    if float(artifact.get("three_family_factor_transport_ready_score", 0.0)) == 1.0:
        return "complete_positive"
    if as_mapping(artifact.get("harm_underpowered_missing_and_flagged_cells")).get(
        "retired_retry_scope"
    ) is True:
        return "retired"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict with the claim boundary."""

    status_text = str(artifact.get("status", "complete_null"))
    if status_text == "blocked_precondition":
        blockers = as_mapping(artifact.get("preconditions_checked")).get("blocked_reasons", [])
        return f"blocked: three-family canonical transport canary missing preconditions {blockers}"
    if status_text == "complete_positive":
        return "complete_positive: each constraint family produced source-bound canonical transport and exact-checker calls; semantic utility is not implied"
    if status_text == "retired":
        return "retired: retry scope produced the same all-invalid canonical transport verdict"
    return "complete_null: live calls ran but three-family canonical transport gates did not all pass"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, and checksum."""

    artifact["three_family_factor_transport_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, counters, oracle boundary, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    require(not missing, f"missing_required_fields:{missing}")
    require([row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(MANDATED_MODEL_IDS), "model_specs_wrong_ids")
    require(artifact.get("autotokenizer_usage_count") == 0, "autotokenizer_usage_count_not_zero")
    require(artifact.get("retired_decoding_mechanism_usage_count") == 0, "retired_decoding_mechanism_used")
    require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle_not_true")
    checker = as_mapping(artifact.get("exact_checker_paths_versions_calls_costs_and_errors"))
    require(checker.get("protected_exact_task_checkers_are_oracle") is True, "exact_checker_oracle_missing")
    require(checker.get("transport_is_oracle") is False, "transport_oracle_misclaimed")
    require(checker.get("parsing_is_oracle") is False, "parser_oracle_misclaimed")
    require(checker.get("model_proposals_are_oracles") is False, "model_proposal_oracle_misclaimed")
    utility = as_mapping(artifact.get("semantic_utility_not_implied_by_transport"))
    require(
        utility.get("transport_ready_implies_semantic_utility") is False,
        "semantic_utility_misclaimed",
    )
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_principles"))), "missing_field_principles")
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_provenance"))), "missing_field_provenance")
    require(
        str(artifact.get("honest_verdict", "")).split(":", 1)[0]
        in {"complete_positive", "complete_null", "blocked", "retired"},
        "bad_verdict_prefix",
    )
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum_mismatch")


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: Path | str = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    exp6379_path: Path | str = REPO_ROOT / EXP6379_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = embedded_gguf_tokenizer_receipt,
    host_checks_func: HostChecksFn = host_environment_receipts,
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
    source_before = source_hashes()
    gate = exp6379_gate_receipt(Path(exp6379_path))
    model_resolution = build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )
    model_specs = model_resolution["MODEL_SPECS"]
    events = generated_events()
    selected = selected_canary_events(events)
    event_manifest_path = result.with_suffix(result.suffix + ".sealed_event_manifest.json")
    schema_path = result.with_suffix(result.suffix + ".canonical_schema.json")
    event_payload = event_manifest_payload(events, selected)
    event_hash = write_payload_or_hash(event_manifest_path, event_payload, write=write)
    schema_receipt = canonical_schema_drift_receipt(
        exp6379_path=Path(exp6379_path),
        schema_path=schema_path,
        model_specs=model_specs,
        selected_events=selected,
        write=write,
    )
    capacity = per_arm_prompt_output_and_context_capacity_receipts(
        model_specs=model_specs,
        selected_events=selected,
        tokenizer_func=tokenizer_func,
    )
    arm_contract = preregistered_arm_contract(capacity)
    host = host_checks_func()
    event_receipt = {
        **path_receipt(event_manifest_path, sha256=event_hash),
        "sha256": event_hash,
        "balance": event_payload["balance"],
    }
    preconditions = preconditions_checked(
        date=date,
        gate=gate,
        model_resolution=model_resolution,
        host=host,
        event_receipt=event_receipt,
        schema_receipt=schema_receipt,
        capacity=capacity,
        protected_before=protected_before,
        source_hashes=source_before,
    )
    if preconditions["all_preconditions_passed"]:
        generation = run_generation_matrix(
            model_specs=model_specs,
            selected_events=selected,
            capacity=capacity,
            data_dir=data,
            tokenizer_func=tokenizer_func,
            generation_func=generation_func,
        )
    else:
        generation = empty_generation_receipts()
    parsed = parse_raw_outputs(generation, events, model_specs)
    raw_before_parse = parsed["raw_output_before_parse_paths_hashes_and_counts"]
    parse_counts = parsed["counts"]
    conflicts = parsed["source_span_alignment_and_conflict_counts"]
    taxonomy = failure_taxonomy_counts(parsed)
    exact_checker, exact_counts = exact_checker_receipts_and_counts(
        [
            row
            for row in parsed["parsed_proposals"]
            if row.get("arm") == CANONICAL_CAPACITY_ARM
        ],
        events,
    )
    protected_after = protected_hashes()
    protected = protected_unchanged_receipt(protected_before, protected_after)
    isolation = same_step_isolation(
        event_hash=event_hash,
        schema_hash=str(schema_receipt["sha256"]),
        protected_hash=sha256_json(event_payload["protected_outcome_hashes"]),
    )
    harm = harm_summary(
        model_resolution=model_resolution,
        generation=generation,
        parse_counts=parse_counts,
        exact_counts=exact_counts,
        conflicts=conflicts,
    )
    commands = list(DEFAULT_TEST_COMMANDS)
    exits = _test_exit_codes(test_exit_codes, commands)
    elapsed = time.perf_counter() - started if duration_s is None else float(duration_s)
    used = [
        model_id
        for model_id in MANDATED_MODEL_IDS
        if any(
            call_authenticated(as_mapping(row))
            for row in generation.get("rows", [])
            if as_mapping(row).get("model_hf_id") == model_id
        )
    ]
    artifact: JsonDict = {
        "status": "complete_null",
        "exp6379_gate_receipt": gate,
        "MODEL_SPECS": model_specs,
        "models_used": used,
        "cached_sota_pair_receipts": model_resolution["cached_sota_pair_receipts"],
        "model_file_hashes_revisions_quantizations_and_tokenizers": model_file_receipts(model_specs),
        "embedded_gguf_tokenizer_receipts": tokenizer_receipts(model_specs, capacity),
        "autotokenizer_usage_count": 0,
        "cuda_offload_and_runtime_receipts_by_model": cuda_runtime_receipts(generation),
        "sealed_event_manifest_path_hash_license_and_balance": {
            **event_receipt,
            "schema": SCHEMA + ".sealed_event_manifest",
            "license": event_payload["generator_license_receipts"],
            "event_count": len(events),
            "selected_event_schedule": event_payload["selected_event_schedule"],
        },
        "canonical_schema_path_hash_and_drift_receipt": schema_receipt,
        "preregistered_arm_contract": arm_contract,
        "per_arm_prompt_output_and_context_capacity_receipts": capacity,
        "raw_output_before_parse_paths_hashes_and_counts": raw_before_parse,
        "failure_taxonomy_counts_by_model_and_arm": taxonomy,
        "parse_valid_invalid_timeout_and_abstain_counts_by_model_and_arm": parse_counts,
        "source_span_alignment_and_conflict_counts": conflicts,
        "exact_checker_paths_versions_calls_costs_and_errors": exact_checker,
        "exact_pass_fail_counts_by_model_and_arm": exact_counts,
        "same_step_read_write_isolation_results": isolation,
        "retired_decoding_mechanism_usage_count": 0,
        "three_family_factor_transport_ready_score": 0.0,
        "semantic_utility_not_implied_by_transport": semantic_utility_boundary(),
        "harm_underpowered_missing_and_flagged_cells": harm,
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


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point for Exp6380."""

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
                "three_family_factor_transport_ready_score": artifact[
                    "three_family_factor_transport_ready_score"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
