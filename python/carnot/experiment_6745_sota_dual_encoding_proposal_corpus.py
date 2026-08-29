"""Build the complete three-model dual-encoding certificate corpus.

The experiment asks each mandated local GGUF model for a small SAT or UNSAT
certificate on every frozen Exp6744 CNF. It preserves every attempt. Two
independent encoders translate parseable output before a model-independent CNF
checker assigns the closed diagnosis.

Spec refs: REQ-VERIFY-6745 and SCENARIO-VERIFY-6745-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from carnot import experiment_6649_exact_certificate_proposal_corpus as prior_runner
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf
from carnot.verify import dual_certificate_encoder_a as encoder_a
from carnot.verify import dual_certificate_encoder_b as encoder_b


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_6745_sota_dual_encoding_proposal_corpus.json")
UPSTREAM_PATH = Path("results/experiment_6744_hardness_controlled_certificate_stream.json")
SCHEMA = "carnot.experiment_6745.sota_dual_encoding_proposal_corpus.v1"
ROW_SCHEMA = f"{SCHEMA}.row"
MANIFEST_SCHEMA = f"{SCHEMA}.manifest"
INFERENCE_SUBSTRATE = "local_llama_cpp_cuda_gguf_dual_encoding_exact_cpu_checking"
RANDOM_SEED = 6_745_000

MODEL_SPECS = [
    {
        "family_id": "qwen36_flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe",
        "device_index": 0,
        "headline_eligible": True,
    },
    {
        "family_id": "gemma4_31b_flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense",
        "device_index": 1,
        "headline_eligible": True,
    },
    {
        "family_id": "gemma4_26b_middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle_moe",
        "device_index": 0,
        "headline_eligible": True,
    },
]

# This is the same proven llama-server contract used by Exp6649. Keeping one
# immutable copy here binds every model and instance to identical budgets.
DECODE_CONFIG = deepcopy(prior_runner.DECODE_PARAMETERS)
DIAGNOSES = (
    "exact_valid",
    "malformed_certificate",
    "translation_disagreement",
    "reasoning_error",
    "abstention",
)
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}

ARTIFACT_FIELDS = (
    "schema",
    "experiment",
    "title",
    "run_date",
    "status",
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "models_used",
    "live_model_invoked",
    "gpu_receipts",
    "planned_row_count",
    "rows",
    "diagnosis_counts",
    "encoder_agreement_matrix",
    "exact_success_by_model",
    "dual_encoding_corpus_ready",
    "gate_check_summary",
    "verdict_class",
    "honest_verdict",
    "preconditions_checked",
    "frozen_manifest",
)
FIELD_PRINCIPLES = {
    "schema": "A versioned shape lets downstream code reject incompatible corpus rows.",
    "experiment": "The numeric identity binds the result to the planned task.",
    "title": "The title states the narrow dual-encoding corpus purpose.",
    "run_date": "The planning date makes the frozen execution window explicit.",
    "status": "The status separates a complete corpus from a complete blocked artifact.",
    "field_principles": "Every field explains its evidence role and downstream effect.",
    "inference_substrate": "The value declares local llama.cpp CUDA GGUF inference and CPU checks.",
    "duration_s": "Monotonic wall time makes live model work visible.",
    "random_seed": "The fixed schedule gives every model the same instance seed.",
    "reproducibility_checksum": "The hash binds the prompt, config, stream, models, and rows.",
    "models_used": "Exact IDs, roles, paths, and file hashes preserve model attribution.",
    "live_model_invoked": "The boolean prevents cached or synthetic rows from posing as inference.",
    "gpu_receipts": "One authentic CUDA load receipt is required for each headline model.",
    "planned_row_count": "The frozen denominator cannot shrink after failures or low accuracy.",
    "rows": "Every model-instance attempt retains raw, parser, budget, encoder, and checker data.",
    "diagnosis_counts": "The closed taxonomy is recomputed from retained rows.",
    "encoder_agreement_matrix": "Semantic agreement separates translation faults from reasoning faults.",
    "exact_success_by_model": "Per-model exact yield derives from rows without controlling readiness.",
    "dual_encoding_corpus_ready": "This downstream gate requires complete attributable evidence, not accuracy.",
    "gate_check_summary": "A failed gate names its expected and observed value.",
    "verdict_class": "The closed project vocabulary makes terminal outcomes machine-readable.",
    "honest_verdict": "A terminal prefix states whether the corpus completed or blocked.",
    "preconditions_checked": "The receipt proves upstream, model, CUDA, VRAM, and checker availability.",
    "frozen_manifest": "The stream, prompt, config, stop rules, seeds, and denominator freeze before output.",
}

_SAT_TERM = re.compile(r"x[1-9][0-9]*=[01]")
_UNSAT_TERM = re.compile(r"c[1-9][0-9]*")


def canonical_json(value: Any) -> str:
    """Return deterministic compact JSON for evidence hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text in the artifact receipt format."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value after canonical serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash an existing file without loading a multi-gigabyte GGUF into RAM."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def manifest_checksum(manifest: Mapping[str, Any]) -> str:
    """Hash the frozen manifest without its self-referential field."""

    return sha256_json({key: value for key, value in manifest.items() if key != "manifest_sha256"})


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the terminal artifact without its self-referential field."""

    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def parse_certificate_dsl(raw_output: str) -> JsonDict:
    """Parse the frozen one-line DSL without adding missing model evidence."""

    base = {
        "parser_status": "malformed",
        "claim": None,
        "terms": [],
        "parse_failure": None,
        "abstention": False,
    }
    text = raw_output.strip()
    if not text:
        return {**base, "parse_failure": "empty_output"}
    if "```" in text:
        return {**base, "parse_failure": "code_fence_not_allowed"}
    parts = text.split()
    claim = parts[0]
    if claim == "ABSTAIN":
        if len(parts) != 1:
            return {**base, "parse_failure": "abstention_has_terms"}
        return {**base, "parser_status": "abstention", "abstention": True}
    if claim not in {"SAT", "UNSAT"}:
        return {**base, "parse_failure": "unknown_claim"}
    if len(parts) == 1:
        return {**base, "parse_failure": "missing_terms"}
    terms = parts[1:] if claim == "SAT" else " ".join(parts[1:]).replace(",", " ").split()
    pattern = _SAT_TERM if claim == "SAT" else _UNSAT_TERM
    if any(pattern.fullmatch(term) is None for term in terms):
        failure = "invalid_sat_term" if claim == "SAT" else "invalid_unsat_term"
        return {**base, "parse_failure": failure}
    return {
        **base,
        "parser_status": "parseable",
        "claim": claim,
        "terms": terms,
    }


def _literal_satisfied(literal: int, assignment: Mapping[int, bool]) -> bool:
    """Evaluate one signed CNF literal under a complete assignment."""

    value = assignment[abs(literal)]
    return value if literal > 0 else not value


def exact_check_constraints(cnf: Mapping[str, Any], constraints: Mapping[str, Any]) -> JsonDict:
    """Check normalized SAT evidence without reading an Exp6744 answer label."""

    n_vars = int(cnf["n_vars"])
    clauses = [[int(literal) for literal in clause] for clause in cnf["clauses"]]
    base = {
        "attempted": True,
        "authority_available": True,
        "valid": False,
        "reason": None,
        "checked_assignment_count": 0,
    }
    claim = constraints.get("claim")
    if claim == "SAT":
        assignment: dict[int, bool] = {}
        bindings = constraints.get("bindings", [])
        for binding in bindings:
            variable = binding.get("variable")
            values = binding.get("values")
            if not isinstance(variable, int) or not isinstance(values, list) or len(values) != 1:
                return {**base, "reason": "assignment_incomplete"}
            if not isinstance(values[0], bool):
                return {**base, "reason": "assignment_value_invalid"}
            assignment[variable] = values[0]
        if set(assignment) != set(range(1, n_vars + 1)):
            return {**base, "reason": "assignment_incomplete"}
        valid = all(
            any(_literal_satisfied(literal, assignment) for literal in clause) for clause in clauses
        )
        return {
            **base,
            "valid": valid,
            "reason": "satisfying_assignment" if valid else "clause_unsatisfied",
            "checked_assignment_count": 1,
        }
    if claim == "UNSAT":
        core = constraints.get("core_clause_indices", [])
        if not isinstance(core, list) or not core:
            return {**base, "reason": "empty_core"}
        if any(not isinstance(index, int) or index < 1 or index > len(clauses) for index in core):
            return {**base, "reason": "core_index_out_of_range"}
        selected = [clauses[index - 1] for index in core]
        checked = 0
        for values in product((False, True), repeat=n_vars):
            checked += 1
            assignment = {index + 1: value for index, value in enumerate(values)}
            if all(
                any(_literal_satisfied(literal, assignment) for literal in clause)
                for clause in selected
            ):
                return {
                    **base,
                    "reason": "core_satisfiable",
                    "checked_assignment_count": checked,
                }
        return {
            **base,
            "valid": True,
            "reason": "core_unsatisfiable",
            "checked_assignment_count": checked,
        }
    return {**base, "reason": "unsupported_claim"}


def build_prompt(source: Mapping[str, Any]) -> str:
    """Build the one frozen proof-carrying prompt without an answer label."""

    cnf = source["cnf"]
    clause_lines = [
        f"c{index}: " + " ".join(str(literal) for literal in clause)
        for index, clause in enumerate(cnf["clauses"], start=1)
    ]
    return "\n".join(
        [
            "Determine whether this CNF is SAT or UNSAT.",
            "Return exactly one line and no explanation. Use one of these forms:",
            "SAT x1=0 x2=1 ... (bind every variable exactly once)",
            "UNSAT c1,c2,... (name a nonempty clause subset that is itself UNSAT)",
            "ABSTAIN",
            "Clauses use signed integers. Clause numbers are one-based.",
            f"n_vars: {cnf['n_vars']}",
            *clause_lines,
        ]
    )


def build_frozen_manifest(stream: Mapping[str, Any]) -> JsonDict:
    """Freeze the stream, prompt, model order, seeds, and budgets before inference."""

    stream_checksum = stream.get("deterministic_replay_receipt", {}).get("first_stream_sha256")
    instances = []
    for index, source in enumerate(stream.get("rows", []), start=1):
        prompt = build_prompt(source)
        instances.append(
            {
                **deepcopy(dict(source)),
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "generation_seed": RANDOM_SEED + index,
            }
        )
    manifest: JsonDict = {
        "schema": MANIFEST_SCHEMA,
        "stream_checksum": stream_checksum,
        "stream_row_count": len(instances),
        "ordered_source_row_ids": [source["row_id"] for source in instances],
        "ordered_model_family_ids": [model["family_id"] for model in MODEL_SPECS],
        "decode_config": deepcopy(DECODE_CONFIG),
        "instances": instances,
        "planned_row_count": len(instances) * len(MODEL_SPECS),
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = manifest_checksum(manifest)
    return manifest


def _empty_encoder_receipt(encoder_id: str) -> JsonDict:
    """Return the explicit not-applicable encoder shape for non-parseable rows."""

    return {
        "attempted": False,
        "encoder_id": encoder_id,
        "normalized_constraints": None,
        "error": None,
        "exact_check": {
            "attempted": False,
            "authority_available": True,
            "valid": None,
            "reason": "not_applicable",
            "checked_assignment_count": 0,
        },
    }


def build_attempt_row(
    source: Mapping[str, Any],
    model: Mapping[str, Any],
    generation: Mapping[str, Any],
    gpu_receipt: Mapping[str, Any],
    generation_seed: int,
) -> JsonDict:
    """Retain one attempt and derive its diagnosis from independent checks."""

    raw_value = generation.get("raw_output", "")
    raw_output = (
        raw_value.decode("utf-8", "replace") if isinstance(raw_value, bytes) else str(raw_value)
    )
    parsed = parse_certificate_dsl(raw_output)
    encoded_a = _empty_encoder_receipt(encoder_a.ENCODER_ID)
    encoded_b = _empty_encoder_receipt(encoder_b.ENCODER_ID)
    agreement: bool | None = None
    row_blocked = False
    if parsed["parser_status"] == "abstention":
        diagnosis: str | None = "abstention"
    elif parsed["parser_status"] != "parseable":
        diagnosis = "malformed_certificate"
    else:
        try:
            first = encoder_a.encode_certificate(parsed)
            second = encoder_b.encode_certificate(parsed)
            first_check = exact_check_constraints(source["cnf"], first["normalized_constraints"])
            second_check = exact_check_constraints(source["cnf"], second["normalized_constraints"])
            encoded_a = {"attempted": True, "error": None, **first, "exact_check": first_check}
            encoded_b = {"attempted": True, "error": None, **second, "exact_check": second_check}
            agreement = first["normalized_constraints"] == second["normalized_constraints"]
            if not agreement:
                diagnosis = "translation_disagreement"
            elif first_check["valid"] is True and second_check["valid"] is True:
                diagnosis = "exact_valid"
            else:
                diagnosis = "reasoning_error"
        except Exception as error:  # Exact authority failures must not receive a guessed diagnosis.
            row_blocked = True
            diagnosis = None
            message = f"{type(error).__name__}: {error}"
            encoded_a["error"] = message
            encoded_b["error"] = message
            encoded_a["exact_check"]["authority_available"] = False
            encoded_b["exact_check"]["authority_available"] = False
    finish_reason = str(generation.get("finish_reason", "unknown"))
    failure = generation.get("failure_kind")
    timed_out = "timeout" in (finish_reason + " " + str(failure)).lower()
    row: JsonDict = {
        "schema": ROW_SCHEMA,
        "row_id": f"{model['family_id']}|{source['row_id']}",
        "model_family_id": model["family_id"],
        "model_hf_id": model["hf_id"],
        "model_role": model["role"],
        "model_path": model.get("model_path"),
        "model_sha256": model.get("model_sha256"),
        "headline_eligible": model.get("headline_eligible") is True,
        "source_row_id": source["row_id"],
        "source_row_sha256": source.get("row_sha256"),
        "pair_id": source.get("pair_id"),
        "pair_role": source.get("pair_role"),
        "family": source.get("family"),
        "size_bin": source.get("size_bin"),
        "split": source.get("split"),
        "prompt_sha256": source.get("prompt_sha256") or sha256_text(build_prompt(source)),
        "generation_seed": int(generation_seed),
        "decode_budget": deepcopy(DECODE_CONFIG),
        "raw_output": raw_output,
        "raw_output_sha256": sha256_text(raw_output),
        "raw_api_response_sha256": generation.get("raw_api_response_sha256"),
        "parser_status": parsed["parser_status"],
        "parse_failure": parsed["parse_failure"],
        "abstention": parsed["abstention"],
        "parsed_certificate": {
            "claim": parsed["claim"],
            "terms": parsed["terms"],
        }
        if parsed["parser_status"] == "parseable"
        else None,
        "timed_out": timed_out,
        "stop_reason": finish_reason,
        "generation_failure_kind": failure,
        "http_status": int(generation.get("http_status", 0) or 0),
        "prompt_tokens": int(generation.get("prompt_tokens", 0) or 0),
        "generated_tokens": int(generation.get("generated_tokens", 0) or 0),
        "latency_s": round(float(generation.get("latency_s", 0.0) or 0.0), 9),
        "started_monotonic_ns": int(generation.get("started_monotonic_ns", 0) or 0),
        "finished_monotonic_ns": int(generation.get("finished_monotonic_ns", 0) or 0),
        "gpu_receipt_id": gpu_receipt.get("session_id") or gpu_receipt.get("model_family_id"),
        "cuda_attributed": gpu_receipt.get("authentic") is True,
        "encoder_a": encoded_a,
        "encoder_b": encoded_b,
        "encoder_agreement": agreement,
        "model_self_judgment_used": False,
        "row_blocked": row_blocked,
        "diagnosis": diagnosis,
        "row_sha256": "",
    }
    row["row_sha256"] = sha256_json(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )
    return row


def recompute_aggregates(
    rows: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    gpu_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Rebuild readiness and all headline summaries from retained rows."""

    expected_ids = {
        f"{model['family_id']}|{source['row_id']}" for model in models for source in sources
    }
    observed_ids = [str(row.get("row_id")) for row in rows]
    diagnosis_counts = {diagnosis: 0 for diagnosis in DIAGNOSES}
    diagnosis_counts.update(Counter(row.get("diagnosis") for row in rows if row.get("diagnosis")))
    matrix = {
        "agree_both_valid": 0,
        "agree_both_invalid": 0,
        "agree_mixed_exact_outcome": 0,
        "disagree": 0,
        "not_applicable": 0,
    }
    for row in rows:
        if row.get("encoder_a", {}).get("attempted") is not True:
            matrix["not_applicable"] += 1
        elif row.get("encoder_agreement") is False:
            matrix["disagree"] += 1
        else:
            valid_a = row.get("encoder_a", {}).get("exact_check", {}).get("valid") is True
            valid_b = row.get("encoder_b", {}).get("exact_check", {}).get("valid") is True
            if valid_a and valid_b:
                matrix["agree_both_valid"] += 1
            elif valid_a != valid_b:
                matrix["agree_mixed_exact_outcome"] += 1
            else:
                matrix["agree_both_invalid"] += 1
    exact_by_model = {}
    for model in models:
        model_rows = [row for row in rows if row.get("model_family_id") == model["family_id"]]
        exact = sum(row.get("diagnosis") == "exact_valid" for row in model_rows)
        exact_by_model[str(model["family_id"])] = {
            "hf_id": model["hf_id"],
            "attempts": len(model_rows),
            "exact_valid": exact,
            "exact_success_rate": exact / len(model_rows) if model_rows else 0.0,
        }
    receipt_ids = {
        str(receipt.get("model_family_id"))
        for receipt in gpu_receipts
        if receipt.get("authentic") is True
    }
    required_receipts = {str(model["family_id"]) for model in models}
    complete_rows = all(
        row.get("model_hf_id")
        and row.get("model_role")
        and row.get("source_row_id")
        and row.get("prompt_sha256")
        and isinstance(row.get("generation_seed"), int)
        and row.get("decode_budget") == DECODE_CONFIG
        and row.get("raw_output_sha256") == sha256_text(str(row.get("raw_output", "")))
        and row.get("diagnosis") in DIAGNOSES
        and row.get("row_blocked") is False
        for row in rows
    )
    applicable_complete = all(
        row.get("parser_status") != "parseable"
        or (
            row.get("encoder_a", {}).get("attempted") is True
            and row.get("encoder_b", {}).get("attempted") is True
            and row.get("encoder_a", {}).get("exact_check", {}).get("attempted") is True
            and row.get("encoder_b", {}).get("exact_check", {}).get("attempted") is True
            and row.get("encoder_a", {}).get("exact_check", {}).get("authority_available") is True
            and row.get("encoder_b", {}).get("exact_check", {}).get("authority_available") is True
        )
        for row in rows
    )
    ready = bool(
        len(rows) == len(expected_ids)
        and len(observed_ids) == len(set(observed_ids))
        and set(observed_ids) == expected_ids
        and complete_rows
        and applicable_complete
        and receipt_ids == required_receipts
    )
    return {
        "planned_row_count": len(expected_ids),
        "observed_row_count": len(rows),
        "diagnosis_counts": diagnosis_counts,
        "encoder_agreement_matrix": matrix,
        "exact_success_by_model": exact_by_model,
        "dual_encoding_corpus_ready": ready,
    }


def build_artifact(
    *,
    date: str,
    duration_s: float,
    manifest: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    gpu_receipts: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    """Build one complete artifact from row-derived aggregates."""

    reduction = recompute_aggregates(rows, manifest.get("instances", []), models, gpu_receipts)
    ready = reduction["dual_encoding_corpus_ready"] is True
    exact_total = sum(row.get("diagnosis") == "exact_valid" for row in rows)
    verdict_class = "positive" if ready and exact_total > 0 else "null" if ready else "partial"
    checks = [
        {
            "check": "dual_encoding_corpus_ready",
            "expected": True,
            "observed": ready,
            "passed": ready,
        }
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 6745,
        "title": "Three-model SOTA dual-encoding proposal corpus",
        "run_date": date,
        "status": "complete" if ready else "complete_partial",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": {
            "base": RANDOM_SEED,
            "decode_seed_schedule": [
                source.get("generation_seed") for source in manifest.get("instances", [])
            ],
            "same_instance_seed_for_all_models": True,
        },
        "reproducibility_checksum": "",
        "models_used": [deepcopy(dict(model)) for model in models],
        "live_model_invoked": any(receipt.get("authentic") is True for receipt in gpu_receipts),
        "gpu_receipts": [deepcopy(dict(receipt)) for receipt in gpu_receipts],
        "planned_row_count": reduction["planned_row_count"],
        "rows": [deepcopy(dict(row)) for row in rows],
        "diagnosis_counts": reduction["diagnosis_counts"],
        "encoder_agreement_matrix": reduction["encoder_agreement_matrix"],
        "exact_success_by_model": reduction["exact_success_by_model"],
        "dual_encoding_corpus_ready": ready,
        "gate_check_summary": {"all_passed": ready, "checks": checks},
        "verdict_class": verdict_class,
        "honest_verdict": (
            f"complete: all {len(rows)} attributable rows are ready; exact-valid is "
            f"{exact_total}/{len(rows)} and does not control readiness"
            if ready
            else "complete_partial: one or more corpus completeness checks failed"
        ),
        "preconditions_checked": deepcopy(dict(preconditions)),
        "frozen_manifest": deepcopy(dict(manifest)),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_blocked_artifact(
    *,
    date: str,
    duration_s: float,
    failed_check: str,
    expected: Any,
    observed: Any,
    models: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    """Build the full terminal schema for one owned precondition block."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 6745,
        "title": "Three-model SOTA dual-encoding proposal corpus",
        "run_date": date,
        "status": "complete_blocked_proposal_corpus",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": {"base": RANDOM_SEED, "decode_seed_schedule": []},
        "reproducibility_checksum": "",
        "models_used": [deepcopy(dict(model)) for model in models],
        "live_model_invoked": False,
        "gpu_receipts": [],
        "planned_row_count": int(manifest.get("planned_row_count", 0) or 0),
        "rows": [],
        "diagnosis_counts": {diagnosis: 0 for diagnosis in DIAGNOSES},
        "encoder_agreement_matrix": {
            "agree_both_valid": 0,
            "agree_both_invalid": 0,
            "agree_mixed_exact_outcome": 0,
            "disagree": 0,
            "not_applicable": 0,
        },
        "exact_success_by_model": {},
        "dual_encoding_corpus_ready": False,
        "gate_check_summary": {
            "all_passed": False,
            "failed_check": failed_check,
            "expected": expected,
            "observed": observed,
        },
        "verdict_class": "blocked",
        "honest_verdict": (
            f"complete_blocked_proposal_corpus: {failed_check} expected {expected!r}, "
            f"observed {observed!r}"
        ),
        "preconditions_checked": deepcopy(dict(preconditions)),
        "frozen_manifest": deepcopy(dict(manifest)),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject schema, principle, aggregate, verdict, or checksum drift."""

    errors = []
    missing = sorted(set(ARTIFACT_FIELDS) - set(artifact))
    if missing:
        return ["missing_required_fields:" + ",".join(missing)]
    if set(artifact) != set(artifact.get("field_principles", {})):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if artifact.get("status") == "complete_blocked_proposal_corpus":
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith(
            "complete_blocked_proposal_corpus"
        ):
            errors.append("blocked_verdict_prefix_mismatch")
        if artifact.get("rows") or artifact.get("dual_encoding_corpus_ready") is not False:
            errors.append("blocked_rows_or_gate_invalid")
        return errors
    reduction = recompute_aggregates(
        artifact.get("rows", []),
        artifact.get("frozen_manifest", {}).get("instances", []),
        artifact.get("models_used", []),
        artifact.get("gpu_receipts", []),
    )
    for key in (
        "planned_row_count",
        "diagnosis_counts",
        "encoder_agreement_matrix",
        "exact_success_by_model",
        "dual_encoding_corpus_ready",
    ):
        if artifact.get(key) != reduction[key]:
            errors.append("aggregate_recomputation_mismatch")
            break
    return errors


def write_json_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Validate, synchronize, and atomically replace the result file."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(";".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".exp6745-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def resolve_models() -> list[JsonDict]:  # pragma: no cover - exercised by the required live run
    """Resolve the mandated Qwen, dense Gemma, and middle Gemma GGUF files."""

    pair = cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 2)) or []
    by_id = {str(model.get("hf_id")): model for model in pair}
    middle_id = MODEL_SPECS[2]["hf_id"]
    middle_path = resolve_cached_gguf(middle_id, "Q4_K_M")
    if middle_path:
        by_id[middle_id] = {"hf_id": middle_id, "model_path": middle_path}
    resolved = []
    for defined in MODEL_SPECS:
        path = str(by_id.get(defined["hf_id"], {}).get("model_path", ""))
        exists = bool(path and Path(path).is_file())
        resolved.append(
            {
                **deepcopy(defined),
                "model_path": path,
                "model_sha256": sha256_file(path) if exists else "missing",
                "model_size_bytes": Path(path).stat().st_size if exists else 0,
                "resolved": exists,
            }
        )
    return resolved


def collect_preconditions(
    stream: Mapping[str, Any], models: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> JsonDict:  # pragma: no cover - exercised by the required live run
    """Check upstream, cache, CUDA, free VRAM, and exact authority before inference."""

    llama = prior_runner.canaries.llama_cpp_receipt()
    gpus = prior_runner.canaries.gpu_inventory()
    gpu_by_index = {int(gpu["index"]): gpu for gpu in gpus}
    vram_rows = []
    for model in models:
        gpu = gpu_by_index.get(int(model["device_index"]), {})
        required_mb = int(model.get("model_size_bytes", 0)) // (1024 * 1024) + 2048
        free_mb = int(gpu.get("memory_free_mb", 0) or 0)
        vram_rows.append(
            {
                "model_family_id": model["family_id"],
                "device_index": model["device_index"],
                "required_mb": required_mb,
                "free_mb": free_mb,
                "passed": free_mb >= required_mb,
            }
        )
    exact_smoke = (
        exact_check_constraints(
            {"n_vars": 1, "clauses": [[1]]},
            encoder_a.encode_certificate(parse_certificate_dsl("SAT x1=1"))[
                "normalized_constraints"
            ],
        )["valid"]
        is True
        and exact_check_constraints(
            {"n_vars": 1, "clauses": [[1], [-1]]},
            encoder_b.encode_certificate(parse_certificate_dsl("UNSAT c1,c2"))[
                "normalized_constraints"
            ],
        )["valid"]
        is True
    )
    checks = [
        {
            "check": "exp6744_hardness_stream_ready",
            "expected": True,
            "observed": stream.get("hardness_stream_ready"),
            "passed": stream.get("hardness_stream_ready") is True,
        },
        {
            "check": "stream_checksum_frozen",
            "expected": "sha256:*",
            "observed": manifest.get("stream_checksum"),
            "passed": str(manifest.get("stream_checksum", "")).startswith("sha256:"),
        },
        {
            "check": "all_three_exact_cached_paths",
            "expected": True,
            "observed": len(models) == 3 and all(model.get("resolved") is True for model in models),
            "passed": len(models) == 3 and all(model.get("resolved") is True for model in models),
        },
        {
            "check": "llama_cpp_cuda_offload",
            "expected": True,
            "observed": llama.get("cuda_linked"),
            "passed": llama.get("cuda_linked") is True,
        },
        {
            "check": "sequential_load_vram",
            "expected": True,
            "observed": all(row["passed"] for row in vram_rows),
            "passed": all(row["passed"] for row in vram_rows),
        },
        {
            "check": "dual_encoder_exact_authority",
            "expected": True,
            "observed": exact_smoke,
            "passed": exact_smoke,
        },
    ]
    return {
        "all_passed": all(check["passed"] for check in checks),
        "checks": checks,
        "vram_rows": vram_rows,
        "llama_cpp": llama,
        "gpu_inventory": gpus,
        "remote_models_allowed": False,
        "legacy_headline_models_allowed": False,
    }


def _first_failed(preconditions: Mapping[str, Any]) -> tuple[str, Any]:
    """Return the first failed precondition and its observed value."""

    for check in preconditions.get("checks", []):
        if check.get("passed") is not True:
            return str(check.get("check")), check.get("observed")
    return "preconditions", preconditions.get("all_passed")


def run(date: str, root: Path = REPO_ROOT) -> JsonDict:  # pragma: no cover - live E2E path
    """Run preconditions, three sequential model sessions, and one atomic write."""

    started = time.monotonic()
    stream = json.loads((root / UPSTREAM_PATH).read_text(encoding="utf-8"))
    manifest = build_frozen_manifest(stream)
    models = resolve_models()
    preconditions = collect_preconditions(stream, models, manifest)
    if preconditions["all_passed"] is not True:
        failed, observed = _first_failed(preconditions)
        artifact = build_blocked_artifact(
            date=date,
            duration_s=time.monotonic() - started,
            failed_check=failed,
            expected=True,
            observed=observed,
            models=models,
            manifest=manifest,
            preconditions=preconditions,
        )
        write_json_atomic(root / RESULT_PATH, artifact)
        return artifact
    rows = []
    gpu_receipts = []
    for model in models:
        generations, receipt = prior_runner._run_model_session(  # noqa: SLF001
            root, model, manifest["instances"]
        )
        receipt = {**receipt, "model_family_id": model["family_id"], "model_hf_id": model["hf_id"]}
        receipt["session_id"] = str(receipt.get("session_id", "")).replace("exp6649", "exp6745", 1)
        gpu_receipts.append(receipt)
        rows.extend(
            build_attempt_row(
                source,
                model,
                generation,
                receipt,
                int(source["generation_seed"]),
            )
            for source, generation in zip(manifest["instances"], generations, strict=True)
        )
    artifact = build_artifact(
        date=date,
        duration_s=time.monotonic() - started,
        manifest=manifest,
        models=models,
        rows=rows,
        gpu_receipts=gpu_receipts,
        preconditions=preconditions,
    )
    write_json_atomic(root / RESULT_PATH, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed planning date used in the terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260829")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    """Run Exp6745 and return zero only for a validated terminal artifact."""

    args = parse_args(argv)
    artifact = run(str(args.date))
    errors = validate_artifact(artifact)
    if errors:
        raise SystemExit(";".join(errors))
    print(json.dumps({"status": artifact["status"], "artifact": str(RESULT_PATH)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
