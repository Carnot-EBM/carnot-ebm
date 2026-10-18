"""Cold-reparse frozen GGUF certificate output without invoking an LLM.

This experiment removes only a proven bytes transport envelope. It preserves
the paid-for output and measures parser, encoder, and exact-check outcomes as
separate evidence.

Spec refs: REQ-VERIFY-6755, SCENARIO-VERIFY-6755-*, REQ-REPORT-6755,
and SCENARIO-REPORT-6755-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from carnot import experiment_6745_sota_dual_encoding_proposal_corpus as frozen
from carnot.inference.gguf_output_text import (
    OutputTextNormalizationError,
    normalize_gguf_output_text,
)
from carnot.verify import dual_certificate_encoder_a as encoder_a
from carnot.verify import dual_certificate_encoder_b as encoder_b


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_PROPOSAL_PATH = Path("results/experiment_6745_sota_dual_encoding_proposal_corpus.json")
UPSTREAM_STREAM_PATH = Path("results/experiment_6744_hardness_controlled_certificate_stream.json")
RESULT_PATH = Path("results/experiment_6755_lossless_gguf_output_reparse.json")
MODULE_PATH = Path("python/carnot/experiment_6755_lossless_gguf_output_reparse.py")
BOUNDARY_PATH = Path("python/carnot/inference/gguf_output_text.py")
FROZEN_PARSER_PATH = Path("python/carnot/experiment_6745_sota_dual_encoding_proposal_corpus.py")
ENCODER_A_PATH = Path("python/carnot/verify/dual_certificate_encoder_a.py")
ENCODER_B_PATH = Path("python/carnot/verify/dual_certificate_encoder_b.py")
CODE_PATHS = (
    MODULE_PATH,
    BOUNDARY_PATH,
    FROZEN_PARSER_PATH,
    ENCODER_A_PATH,
    ENCODER_B_PATH,
)

SCHEMA = "carnot.experiment_6755.lossless_gguf_output_reparse.v1"
ROW_SCHEMA = f"{SCHEMA}.row"
INFERENCE_SUBSTRATE = "frozen_local_row_replay_no_llm"
RANDOM_SEED = 6_755_000
EXPECTED_ROW_COUNT = 216
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

ROW_DERIVED_FIELDS = (
    "replayed_row_count",
    "pre_diagnosis_counts",
    "post_diagnosis_counts",
    "normalization_kind_counts",
    "bytes_envelope_rows",
    "invalid_variable_reference_rows",
    "invalid_clause_reference_rows",
    "non_binary_value_rows",
    "duplicate_rows",
    "incomplete_evidence_rows",
    "invalid_typed_symbol_rows",
    "false_parseable_proof_rows",
    "environment_grammar_targetable_rows",
    "exact_valid_rows",
    "semantic_edits_performed",
    "transport_reparse_ready",
)
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
    "reproducibility_receipt",
    "source_artifact_sha256",
    "source_stream_artifact_sha256",
    "rows",
    *ROW_DERIVED_FIELDS,
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "schema": "A versioned shape lets readers reject incompatible replay receipts.",
    "experiment": "The numeric identity binds this result to the planned frozen reparse.",
    "title": "The title states that this result tests transport rather than model quality.",
    "run_date": "The fixed planning date identifies the intended evidence window.",
    "status": "The status separates a complete replay from a task-owned input block.",
    "field_principles": "This map explains every field, including why this map exists.",
    "inference_substrate": "The value states that local frozen rows were replayed with no LLM.",
    "duration_s": "Monotonic wall time records the real local replay cost.",
    "random_seed": "The fixed seed identifies this deterministic replay protocol.",
    "reproducibility_checksum": "The hash binds input, code, seed, and row-derived evidence.",
    "reproducibility_receipt": "Component hashes show which input, code, and rows form the checksum.",
    "source_artifact_sha256": "The file hash binds every preserved Exp6745 receipt.",
    "source_stream_artifact_sha256": "The file hash binds the exact CNFs used for checking.",
    "rows": "Every frozen proposal keeps its original evidence and cold replay result.",
    "replayed_row_count": "The count proves that the paid-for denominator did not shrink.",
    "pre_diagnosis_counts": "Stored diagnoses show the failure state before transport recovery.",
    "post_diagnosis_counts": "Cold diagnoses separate syntax recovery from exact correctness.",
    "normalization_kind_counts": "Kinds show which rows used legacy-envelope recovery.",
    "bytes_envelope_rows": "This count measures the transport defect found in frozen text.",
    "invalid_variable_reference_rows": "This count measures SAT symbols outside each CNF environment.",
    "invalid_clause_reference_rows": "This count measures UNSAT clause IDs outside each CNF environment.",
    "non_binary_value_rows": "This count measures assignment values outside the binary domain.",
    "duplicate_rows": "This count measures evidence that violates uniqueness constraints.",
    "incomplete_evidence_rows": "This count measures missing required assignment or core evidence.",
    "invalid_typed_symbol_rows": "This count measures terms outside the certificate symbol grammar.",
    "false_parseable_proof_rows": "This count keeps syntactic success separate from false proofs.",
    "environment_grammar_targetable_rows": "This row union bounds errors addressable by environment grammar.",
    "exact_valid_rows": "This count reports semantic proof success without controlling transport readiness.",
    "semantic_edits_performed": "Zero proves that replay removed transport encoding only.",
    "transport_reparse_ready": "This gate requires 216 lossless replays and matching row reduction.",
    "gate_check_summary": "Each gate records its expected and observed value.",
    "verifier_is_oracle": "False separates the exact evaluator from the transport mechanism.",
    "verdict_class": "The closed class makes the terminal result machine-readable.",
    "honest_verdict": "The terminal text distinguishes transport recovery from exact proof success.",
}

_SAT_BINDING = re.compile(r"x([0-9]+)=([^\s]+)")
_UNSAT_REFERENCE = re.compile(r"c([0-9]+)")


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for evidence hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text in the source artifact format."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value after canonical serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    """Hash file bytes and retain an explicit missing state."""

    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json_object(path: Path) -> JsonDict:
    """Load one JSON object and reject scalar or array substitutes."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON object required")
    return value


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Build one stable precondition row with its observed value."""

    return {"check": check, "expected": expected, "observed": observed, "passed": passed}


def evaluate_preconditions(proposal: Mapping[str, Any], stream: Mapping[str, Any]) -> JsonDict:
    """Check the frozen denominator, hashes, ready stream, and source coverage."""

    rows = proposal.get("rows", [])
    row_ids = [str(row.get("row_id", "")) for row in rows if isinstance(row, Mapping)]
    hashes_present = len(rows) == len(row_ids) and all(
        isinstance(row.get("raw_output_sha256"), str) and bool(row["raw_output_sha256"])
        for row in rows
        if isinstance(row, Mapping)
    )
    proposal_stream = proposal.get("frozen_manifest", {}).get("stream_checksum")
    exact_stream = stream.get("deterministic_replay_receipt", {}).get("first_stream_sha256")
    source_ids = {
        str(row.get("row_id")) for row in stream.get("rows", []) if isinstance(row, Mapping)
    }
    requested_source_ids = {
        str(row.get("source_row_id")) for row in rows if isinstance(row, Mapping)
    }
    missing_sources = sorted(requested_source_ids - source_ids)
    stream_link = {
        "hardness_stream_ready": stream.get("hardness_stream_ready"),
        "proposal_stream_checksum": proposal_stream,
        "exact_stream_checksum": exact_stream,
    }
    checks = [
        _check("exp6745_row_count", EXPECTED_ROW_COUNT, len(rows), len(rows) == EXPECTED_ROW_COUNT),
        _check(
            "exp6745_unique_row_ids",
            EXPECTED_ROW_COUNT,
            len(set(row_ids)),
            len(row_ids) == EXPECTED_ROW_COUNT and len(set(row_ids)) == EXPECTED_ROW_COUNT,
        ),
        _check(
            "exp6745_original_output_hashes",
            EXPECTED_ROW_COUNT,
            sum(
                isinstance(row, Mapping)
                and isinstance(row.get("raw_output_sha256"), str)
                and bool(row.get("raw_output_sha256"))
                for row in rows
            ),
            hashes_present and len(rows) == EXPECTED_ROW_COUNT,
        ),
        _check(
            "exp6744_ready_stream_link",
            {"hardness_stream_ready": True, "checksums_match": True},
            stream_link,
            stream.get("hardness_stream_ready") is True
            and isinstance(proposal_stream, str)
            and proposal_stream.startswith("sha256:")
            and proposal_stream == exact_stream,
        ),
        _check("exp6744_source_row_coverage", [], missing_sources, not missing_sources),
    ]
    return {"all_passed": all(row["passed"] for row in checks), "checks": checks}


def first_failed_check(summary: Mapping[str, Any]) -> JsonDict:
    """Return the first failed gate, or one explicit all-passed receipt."""

    for row in summary.get("checks", []):
        if row.get("passed") is not True:
            return deepcopy(dict(row))
    return {"check": "all_preconditions", "expected": True, "observed": True, "passed": True}


def analyze_grammar_failures(text: str, cnf: Mapping[str, Any]) -> JsonDict:
    """Measure only errors a typed environment grammar can prevent."""

    n_vars = int(cnf["n_vars"])
    n_clauses = len(cnf["clauses"])
    parts = text.strip().split()
    claim = parts[0] if parts else ""
    terms = parts[1:]
    invalid_variables: list[int] = []
    invalid_clauses: list[int] = []
    non_binary_terms: list[str] = []
    duplicate_symbols: list[str] = []
    invalid_terms: list[str] = []
    incomplete = False

    if claim == "SAT":
        seen: list[int] = []
        complete_variables: set[int] = set()
        for term in terms:
            match = _SAT_BINDING.fullmatch(term)
            if match is None:
                invalid_terms.append(term)
                continue
            variable = int(match.group(1))
            value = match.group(2)
            seen.append(variable)
            if variable < 1 or variable > n_vars:
                invalid_variables.append(variable)
            if value not in {"0", "1"}:
                non_binary_terms.append(term)
            if 1 <= variable <= n_vars and value in {"0", "1"}:
                complete_variables.add(variable)
        duplicate_symbols = [f"x{value}" for value, count in Counter(seen).items() if count > 1]
        incomplete = complete_variables != set(range(1, n_vars + 1))
    elif claim == "UNSAT":
        references = " ".join(terms).replace(",", " ").split()
        seen_clauses: list[int] = []
        valid_clauses: list[int] = []
        for term in references:
            match = _UNSAT_REFERENCE.fullmatch(term)
            if match is None:
                invalid_terms.append(term)
                continue
            clause = int(match.group(1))
            seen_clauses.append(clause)
            if clause < 1 or clause > n_clauses:
                invalid_clauses.append(clause)
            else:
                valid_clauses.append(clause)
        duplicate_symbols = [
            f"c{value}" for value, count in Counter(seen_clauses).items() if count > 1
        ]
        incomplete = not valid_clauses
    else:
        invalid_terms = parts

    result = {
        "invalid_variable_reference": bool(invalid_variables),
        "invalid_clause_reference": bool(invalid_clauses),
        "non_binary_value": bool(non_binary_terms),
        "duplicate": bool(duplicate_symbols),
        "incomplete_evidence": incomplete,
        "invalid_typed_symbol": bool(invalid_terms),
        "invalid_variable_references": sorted(set(invalid_variables)),
        "invalid_clause_references": sorted(set(invalid_clauses)),
        "non_binary_terms": non_binary_terms,
        "duplicate_symbols": duplicate_symbols,
        "invalid_terms": invalid_terms,
    }
    result["environment_grammar_targetable"] = any(
        result[key]
        for key in (
            "invalid_variable_reference",
            "invalid_clause_reference",
            "non_binary_value",
            "duplicate",
            "incomplete_evidence",
            "invalid_typed_symbol",
        )
    )
    return result


def _empty_encoder_receipt(encoder_id: str) -> JsonDict:
    """Keep a complete non-applicable encoder shape for failed syntax."""

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


def _not_attempted_parse(reason: str) -> JsonDict:
    """Represent a parser that correctly did not receive rejected transport."""

    return {
        "parser_status": "not_attempted",
        "claim": None,
        "terms": [],
        "parse_failure": reason,
        "abstention": False,
    }


def replay_row(proposal: Mapping[str, Any], source: Mapping[str, Any]) -> JsonDict:
    """Replay one frozen output without reading its source answer label."""

    raw_value = proposal.get("raw_output", "")
    original_text = raw_value if isinstance(raw_value, str) else repr(raw_value)
    original_hash = str(proposal.get("raw_output_sha256", ""))
    pre_parse = frozen.parse_certificate_dsl(original_text)
    encoded_a = _empty_encoder_receipt(encoder_a.ENCODER_ID)
    encoded_b = _empty_encoder_receipt(encoder_b.ENCODER_ID)
    normalized_text: str | None = None
    normalized_hash: str | None = None
    normalization_kind = "rejected"
    normalization_error: str | None = None
    semantic_edits = 0
    encoder_agreement: bool | None = None
    exact_outcome = "not_applicable"
    failure_reason: str | None = None
    post_diagnosis: str | None = None

    try:
        normalization = normalize_gguf_output_text(raw_value, unwrap_legacy_envelope=True)
    except OutputTextNormalizationError as error:
        normalization_error = error.reason
        failure_reason = error.reason
        post_parse = _not_attempted_parse(error.reason)
        grammar = analyze_grammar_failures("", source["cnf"])
        post_diagnosis = "transport_normalization_failed"
    else:
        normalized_text = str(normalization["text"])
        normalized_hash = str(normalization["normalized_text_sha256"])
        normalization_kind = str(normalization["normalization_kind"])
        semantic_edits = int(normalization["semantic_edits_performed"])
        post_parse = frozen.parse_certificate_dsl(normalized_text)
        grammar = analyze_grammar_failures(normalized_text, source["cnf"])
        if post_parse["parser_status"] == "abstention":
            post_diagnosis = "abstention"
            failure_reason = "model_abstention"
        elif post_parse["parser_status"] != "parseable":
            post_diagnosis = "malformed_certificate"
            failure_reason = str(post_parse["parse_failure"])
        else:
            try:
                first = encoder_a.encode_certificate(post_parse)
                second = encoder_b.encode_certificate(post_parse)
                first_check = frozen.exact_check_constraints(
                    source["cnf"], first["normalized_constraints"]
                )
                second_check = frozen.exact_check_constraints(
                    source["cnf"], second["normalized_constraints"]
                )
                encoded_a = {"attempted": True, "error": None, **first, "exact_check": first_check}
                encoded_b = {
                    "attempted": True,
                    "error": None,
                    **second,
                    "exact_check": second_check,
                }
                encoder_agreement = (
                    first["normalized_constraints"] == second["normalized_constraints"]
                )
                if not encoder_agreement:
                    post_diagnosis = "translation_disagreement"
                    exact_outcome = "translation_disagreement"
                    failure_reason = "normalized_constraints_disagree"
                elif first_check["valid"] is True and second_check["valid"] is True:
                    post_diagnosis = "exact_valid"
                    exact_outcome = "exact_valid"
                else:
                    post_diagnosis = "reasoning_error"
                    exact_outcome = "false_parseable_proof"
                    reasons = [str(first_check["reason"]), str(second_check["reason"])]
                    failure_reason = reasons[0] if reasons[0] == reasons[1] else "|".join(reasons)
            except Exception as error:  # Exact authority failure stays explicit and unclassified.
                message = f"{type(error).__name__}: {error}"
                encoded_a["error"] = message
                encoded_b["error"] = message
                encoded_a["exact_check"]["authority_available"] = False
                encoded_b["exact_check"]["authority_available"] = False
                post_diagnosis = "exact_authority_failed"
                exact_outcome = "authority_failure"
                failure_reason = message

    original_hash_matches = bool(original_hash) and original_hash == sha256_text(original_text)
    replay_complete = normalization_error is None and (
        post_parse["parser_status"] != "parseable"
        or (
            encoded_a["attempted"] is True
            and encoded_b["attempted"] is True
            and encoded_a["exact_check"]["attempted"] is True
            and encoded_b["exact_check"]["attempted"] is True
        )
    )
    evidence_preserved = (
        original_hash_matches
        and normalized_text is not None
        and normalized_hash == sha256_text(normalized_text)
    )
    row: JsonDict = {
        "schema": ROW_SCHEMA,
        "row_id": str(proposal.get("row_id")),
        "model": {
            "family_id": proposal.get("model_family_id"),
            "hf_id": proposal.get("model_hf_id"),
            "role": proposal.get("model_role"),
        },
        "family": proposal.get("family") or source.get("family"),
        "source_row": {
            "row_id": proposal.get("source_row_id"),
            "row_sha256": proposal.get("source_row_sha256"),
            "exact_stream_row_sha256": source.get("row_sha256"),
        },
        "source_artifact_row_sha256": proposal.get("row_sha256"),
        "original_raw_api_response_sha256": proposal.get("raw_api_response_sha256"),
        "original_output_text": original_text,
        "original_output_sha256": original_hash,
        "original_output_hash_matches": original_hash_matches,
        "normalized_output_text": normalized_text,
        "normalized_output_sha256": normalized_hash,
        "normalization_kind": normalization_kind,
        "normalization_error": normalization_error,
        "pre_diagnosis": proposal.get("diagnosis"),
        "pre_parse_result": pre_parse,
        "post_diagnosis": post_diagnosis,
        "post_parse_result": post_parse,
        "encoder_a": encoded_a,
        "encoder_b": encoded_b,
        "encoder_agreement": encoder_agreement,
        "exact_outcome": exact_outcome,
        "failure_reason": failure_reason,
        "grammar_failures": grammar,
        "semantic_edits_performed": semantic_edits,
        "replay_complete": replay_complete,
        "evidence_preserved": evidence_preserved,
        "row_sha256": "",
    }
    row["row_sha256"] = sha256_json(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )
    return row


def _counts(rows: Sequence[Mapping[str, Any]], field: str) -> JsonDict:
    """Count one row field and sort keys for stable artifacts."""

    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def recompute_aggregates(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Rebuild every headline value from retained proposal rows."""

    ids = [str(row.get("row_id")) for row in rows]
    row_hashes_match = all(
        row.get("row_sha256")
        == sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
        for row in rows
    )
    reduction: JsonDict = {
        "replayed_row_count": len(rows),
        "pre_diagnosis_counts": _counts(rows, "pre_diagnosis"),
        "post_diagnosis_counts": _counts(rows, "post_diagnosis"),
        "normalization_kind_counts": _counts(rows, "normalization_kind"),
        "bytes_envelope_rows": sum(
            row.get("normalization_kind") == "legacy_python_bytes_literal" for row in rows
        ),
        "invalid_variable_reference_rows": sum(
            row.get("grammar_failures", {}).get("invalid_variable_reference") is True
            for row in rows
        ),
        "invalid_clause_reference_rows": sum(
            row.get("grammar_failures", {}).get("invalid_clause_reference") is True for row in rows
        ),
        "non_binary_value_rows": sum(
            row.get("grammar_failures", {}).get("non_binary_value") is True for row in rows
        ),
        "duplicate_rows": sum(
            row.get("grammar_failures", {}).get("duplicate") is True for row in rows
        ),
        "incomplete_evidence_rows": sum(
            row.get("grammar_failures", {}).get("incomplete_evidence") is True for row in rows
        ),
        "invalid_typed_symbol_rows": sum(
            row.get("grammar_failures", {}).get("invalid_typed_symbol") is True for row in rows
        ),
        "false_parseable_proof_rows": sum(
            row.get("exact_outcome") == "false_parseable_proof" for row in rows
        ),
        "environment_grammar_targetable_rows": sum(
            row.get("grammar_failures", {}).get("environment_grammar_targetable") is True
            for row in rows
        ),
        "exact_valid_rows": sum(row.get("exact_outcome") == "exact_valid" for row in rows),
        "semantic_edits_performed": sum(
            int(row.get("semantic_edits_performed", 0) or 0) for row in rows
        ),
    }
    reduction["transport_reparse_ready"] = (
        len(rows) == EXPECTED_ROW_COUNT
        and len(set(ids)) == EXPECTED_ROW_COUNT
        and row_hashes_match
        and all(row.get("replay_complete") is True for row in rows)
        and all(row.get("evidence_preserved") is True for row in rows)
        and reduction["semantic_edits_performed"] == 0
    )
    return reduction


def _code_file_hashes() -> JsonDict:
    """Bind every implementation used by normalization and exact replay."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in CODE_PATHS}


def build_reproducibility_receipt(
    rows: Sequence[Mapping[str, Any]],
    source_artifact_sha256: str,
    source_stream_artifact_sha256: str,
    *,
    code_files: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Bind the frozen inputs, implementation code, seed, and replay rows."""

    files = deepcopy(dict(code_files)) if code_files is not None else _code_file_hashes()
    input_hash = sha256_json(
        {
            "proposal_artifact": source_artifact_sha256,
            "exact_stream_artifact": source_stream_artifact_sha256,
        }
    )
    code_hash = sha256_json(files)
    rows_hash = sha256_json(rows)
    value = sha256_json(
        {
            "input_sha256": input_hash,
            "code_sha256": code_hash,
            "rows_sha256": rows_hash,
            "random_seed": RANDOM_SEED,
        }
    )
    return {
        "algorithm": "sha256",
        "input_sha256": input_hash,
        "code_sha256": code_hash,
        "rows_sha256": rows_hash,
        "code_files": files,
        "value": value,
    }


def _base_artifact(date: str, duration_s: float) -> JsonDict:
    """Build fields shared by completed and blocked terminal receipts."""

    return {
        "schema": SCHEMA,
        "experiment": 6755,
        "title": "Lossless GGUF output boundary and frozen 216-row reparse",
        "run_date": date,
        "status": "",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "reproducibility_receipt": {},
        "source_artifact_sha256": "missing",
        "source_stream_artifact_sha256": "missing",
        "rows": [],
        **recompute_aggregates([]),
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "",
    }


def _set_reproducibility(artifact: JsonDict) -> None:
    """Populate the component receipt and its top-level stable checksum."""

    receipt = build_reproducibility_receipt(
        artifact["rows"],
        str(artifact["source_artifact_sha256"]),
        str(artifact["source_stream_artifact_sha256"]),
    )
    artifact["reproducibility_receipt"] = receipt
    artifact["reproducibility_checksum"] = receipt["value"]


def build_blocked_artifact(
    *,
    date: str,
    duration_s: float,
    failed_check: str,
    expected: Any,
    observed: Any,
    source_artifact_sha256: str,
    source_stream_artifact_sha256: str = "not_checked",
) -> JsonDict:
    """Build the full terminal schema for one frozen-input block."""

    artifact = _base_artifact(date, duration_s)
    artifact.update(
        {
            "status": "complete_blocked_reparse_input",
            "source_artifact_sha256": source_artifact_sha256,
            "source_stream_artifact_sha256": source_stream_artifact_sha256,
            "gate_check_summary": {
                "all_passed": False,
                "failed_check": failed_check,
                "expected": expected,
                "observed": observed,
            },
            "verdict_class": "blocked",
            "honest_verdict": (
                f"complete_blocked_reparse_input: {failed_check} expected {expected!r}, "
                f"observed {observed!r}"
            ),
        }
    )
    _set_reproducibility(artifact)
    return artifact


def build_artifact(
    *,
    date: str,
    duration_s: float,
    rows: Sequence[Mapping[str, Any]],
    source_artifact_sha256: str,
    source_stream_artifact_sha256: str,
    preconditions: Mapping[str, Any],
) -> JsonDict:
    """Build one complete artifact from cold row reductions."""

    retained_rows = [deepcopy(dict(row)) for row in rows]
    reduction = recompute_aggregates(retained_rows)
    cold_rows = json.loads(canonical_json(retained_rows))
    cold_reduction = recompute_aggregates(cold_rows)
    aggregate_match = reduction == cold_reduction
    ready = (
        preconditions.get("all_passed") is True
        and reduction["transport_reparse_ready"] is True
        and aggregate_match
    )
    reduction["transport_reparse_ready"] = ready
    artifact = _base_artifact(date, duration_s)
    artifact.update(
        {
            "status": "complete" if ready else "complete_partial",
            "source_artifact_sha256": source_artifact_sha256,
            "source_stream_artifact_sha256": source_stream_artifact_sha256,
            "rows": retained_rows,
            **reduction,
            "gate_check_summary": {
                "all_passed": ready,
                "checks": [
                    *deepcopy(list(preconditions.get("checks", []))),
                    _check(
                        "lossless_row_replay",
                        EXPECTED_ROW_COUNT,
                        sum(
                            row.get("replay_complete") is True
                            and row.get("evidence_preserved") is True
                            for row in retained_rows
                        ),
                        reduction["transport_reparse_ready"] is True,
                    ),
                    _check(
                        "cold_aggregate_recomputation",
                        True,
                        aggregate_match,
                        aggregate_match,
                    ),
                ],
            },
            "verdict_class": "positive" if ready else "partial",
            "honest_verdict": (
                f"complete: lossless transport replay recovered {len(retained_rows)}/"
                f"{EXPECTED_ROW_COUNT} rows; {reduction['exact_valid_rows']}/"
                f"{EXPECTED_ROW_COUNT} exact-valid is a separate semantic outcome"
                if ready
                else "complete_partial: frozen transport replay did not preserve every row"
            ),
        }
    )
    _set_reproducibility(artifact)
    return artifact


def _validate_reproducibility(artifact: Mapping[str, Any]) -> bool:
    """Recompute the stable checksum from recorded component hashes and rows."""

    receipt = artifact.get("reproducibility_receipt", {})
    if not isinstance(receipt, Mapping):
        return False
    expected = build_reproducibility_receipt(
        artifact.get("rows", []),
        str(artifact.get("source_artifact_sha256")),
        str(artifact.get("source_stream_artifact_sha256")),
        code_files=receipt.get("code_files", {}),
    )
    return (
        dict(receipt) == expected and artifact.get("reproducibility_checksum") == expected["value"]
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject field, aggregate, verdict, row, or checksum drift."""

    missing = sorted(set(ARTIFACT_FIELDS) - set(artifact))
    if missing:
        return ["missing_required_fields:" + ",".join(missing)]
    errors: list[str] = []
    if set(artifact) != set(artifact.get("field_principles", {})):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    if not _validate_reproducibility(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if artifact.get("status") == "complete_blocked_reparse_input":
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith("complete_blocked_reparse_input"):
            errors.append("blocked_verdict_prefix_mismatch")
        if artifact.get("rows"):
            errors.append("blocked_rows_invalid")
        return errors

    reduction = recompute_aggregates(artifact.get("rows", []))
    if any(artifact.get(field) != reduction[field] for field in ROW_DERIVED_FIELDS):
        errors.append("aggregate_recomputation_mismatch")
    ready = artifact.get("transport_reparse_ready") is True
    if ready != bool(artifact.get("gate_check_summary", {}).get("all_passed")):
        errors.append("readiness_gate_mismatch")
    if ready and artifact.get("verdict_class") != "positive":
        errors.append("ready_verdict_class_mismatch")
    return errors


def write_json_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Validate, synchronize, and atomically replace one result artifact."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(";".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".exp6755-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():  # pragma: no cover - OS failure cleanup
            temporary.unlink()


def _write_blocked(
    *,
    root: Path,
    date: str,
    started: float,
    failed_check: str,
    expected: Any,
    observed: Any,
    proposal_hash: str,
    stream_hash: str = "not_checked",
) -> JsonDict:
    """Publish one task-owned blocked artifact and stop the replay path."""

    artifact = build_blocked_artifact(
        date=date,
        duration_s=time.monotonic() - started,
        failed_check=failed_check,
        expected=expected,
        observed=observed,
        source_artifact_sha256=proposal_hash,
        source_stream_artifact_sha256=stream_hash,
    )
    write_json_atomic(root / RESULT_PATH, artifact)
    return artifact


def _public_source_rows(stream: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Project CNF inputs without copying or reading any answer label."""

    fields = ("row_id", "row_sha256", "family", "cnf")
    return {
        str(row.get("row_id")): {field: deepcopy(row.get(field)) for field in fields}
        for row in stream.get("rows", [])
        if isinstance(row, Mapping)
    }


def run(date: str, root: Path = REPO_ROOT) -> JsonDict:
    """Check frozen inputs, replay all rows, and publish one atomic artifact."""

    started = time.monotonic()
    proposal_path = root / UPSTREAM_PROPOSAL_PATH
    stream_path = root / UPSTREAM_STREAM_PATH
    proposal_hash = sha256_file(proposal_path)
    try:
        proposal = load_json_object(proposal_path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        return _write_blocked(
            root=root,
            date=date,
            started=started,
            failed_check="exp6745_json_object",
            expected="parseable JSON object",
            observed=f"{type(error).__name__}: {error}",
            proposal_hash=proposal_hash,
        )
    stream_hash = sha256_file(stream_path)
    try:
        stream = load_json_object(stream_path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        return _write_blocked(
            root=root,
            date=date,
            started=started,
            failed_check="exp6744_json_object",
            expected="parseable JSON object",
            observed=f"{type(error).__name__}: {error}",
            proposal_hash=proposal_hash,
            stream_hash=stream_hash,
        )
    preconditions = evaluate_preconditions(proposal, stream)
    if preconditions["all_passed"] is not True:
        failed = first_failed_check(preconditions)
        return _write_blocked(
            root=root,
            date=date,
            started=started,
            failed_check=str(failed["check"]),
            expected=failed["expected"],
            observed=failed["observed"],
            proposal_hash=proposal_hash,
            stream_hash=stream_hash,
        )

    sources = _public_source_rows(stream)
    rows = [replay_row(row, sources[str(row["source_row_id"])]) for row in proposal["rows"]]
    artifact = build_artifact(
        date=date,
        duration_s=time.monotonic() - started,
        rows=rows,
        source_artifact_sha256=proposal_hash,
        source_stream_artifact_sha256=stream_hash,
        preconditions=preconditions,
    )
    write_json_atomic(root / RESULT_PATH, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed planning date for the terminal receipt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260829")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    """Run Exp6755 and return zero only for a validated terminal artifact."""

    args = parse_args(argv)
    artifact = run(str(args.date))
    errors = validate_artifact(artifact)
    if errors:
        raise SystemExit(";".join(errors))
    print(json.dumps({"status": artifact["status"], "artifact": str(RESULT_PATH)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
