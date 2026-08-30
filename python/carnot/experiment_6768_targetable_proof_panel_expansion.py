"""Build a frozen, answer-blind panel of exact-invalid proof mutations.

The mutation operators use only certificate text and the source CNF. Exact
checks evaluate completed mutations, but they never choose mutation content.

Spec refs: REQ-VERIFY-6768, SCENARIO-VERIFY-6768-*, REQ-REPORT-6768,
and SCENARIO-REPORT-6768-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_6745_sota_dual_encoding_proposal_corpus as frozen
from carnot.verify import dual_certificate_encoder_a as encoder_a
from carnot.verify import dual_certificate_encoder_b as encoder_b


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REPLAY_PATH = Path("results/experiment_6755_lossless_gguf_output_reparse.json")
UPSTREAM_PROPOSAL_PATH = Path("results/experiment_6745_sota_dual_encoding_proposal_corpus.json")
UPSTREAM_STREAM_PATH = Path("results/experiment_6744_hardness_controlled_certificate_stream.json")
RESULT_PATH = Path("results/experiment_6768_targetable_proof_panel_expansion.json")
MODULE_PATH = Path("python/carnot/experiment_6768_targetable_proof_panel_expansion.py")
FROZEN_PARSER_PATH = Path("python/carnot/experiment_6745_sota_dual_encoding_proposal_corpus.py")
ENCODER_A_PATH = Path("python/carnot/verify/dual_certificate_encoder_a.py")
ENCODER_B_PATH = Path("python/carnot/verify/dual_certificate_encoder_b.py")
CODE_PATHS = (MODULE_PATH, FROZEN_PARSER_PATH, ENCODER_A_PATH, ENCODER_B_PATH)

SCHEMA = "carnot.experiment_6768.targetable_proof_panel_expansion.v1"
ROW_SCHEMA = f"{SCHEMA}.row"
PARSER_ID = "carnot.exp6768.environment_structural_parser.v1"
INFERENCE_SUBSTRATE = "deterministic_local_certificate_mutation_no_llm"
RANDOM_SEED = 6_768_000
EXPECTED_REPLAY_ROW_COUNT = 216
EXPECTED_TARGETABLE_SOURCE_COUNT = 21
MINIMUM_PANEL_ROWS = 36
MINIMUM_ROWS_PER_STRATUM = 2
HELD_FAMILIES = ("expander_tseitin", "ladder_tseitin", "pigeonhole_anchor")
ERROR_CLASSES = (
    "undefined_variable",
    "invalid_clause",
    "non_binary_value",
    "duplicate_evidence",
    "missing_evidence",
    "premature_terminal",
)
OPERATOR_NAMES = {
    "undefined_variable": "answer_blind_append_undefined_variable_v1",
    "invalid_clause": "answer_blind_replace_with_out_of_range_clause_v1",
    "non_binary_value": "answer_blind_replace_binary_value_with_two_v1",
    "duplicate_evidence": "answer_blind_append_conflicting_duplicate_v1",
    "missing_evidence": "answer_blind_remove_last_required_binding_v1",
    "premature_terminal": "answer_blind_truncate_after_first_binding_v1",
}
FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "answer",
        "answer_label",
        "label",
        "exact_outcome",
        "exact_valid",
        "solver_trace",
        "ground_truth_certificate",
    }
)
FUTURE_OR_ANSWER_FEATURES_READ: list[str] = []
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
ROW_DERIVED_FIELDS = (
    "targetable_row_count",
    "counts_by_family",
    "counts_by_error_class",
    "exact_valid_mutations",
    "duplicate_rows",
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
    "source_proposal_artifact_sha256",
    "source_stream_artifact_sha256",
    "source_targetable_row_ids",
    "mutation_operators",
    "rows",
    *ROW_DERIVED_FIELDS,
    "proof_preserving_relabel_receipts",
    "future_or_answer_features_read",
    "cold_replay_receipt",
    "targetable_panel_ready",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "schema": "A versioned shape lets readers reject incompatible panel receipts.",
    "experiment": "The numeric identity binds this result to the planned panel task.",
    "title": "The title states that this result is a fixture, not a model claim.",
    "run_date": "The fixed date identifies the planned evidence window.",
    "status": "The status separates a ready fixture from a visible blocked result.",
    "field_principles": "This map explains every field, including why this map exists.",
    "inference_substrate": "The value declares deterministic local mutation with no LLM.",
    "duration_s": "Monotonic wall time records the real panel build cost.",
    "random_seed": "The fixed seed binds operator order and deterministic fallbacks.",
    "reproducibility_checksum": "The hash binds frozen inputs, code, seed, and rows.",
    "reproducibility_receipt": "Component hashes identify all checksum inputs.",
    "source_artifact_sha256": "The file hash binds the immutable Exp6755 input.",
    "source_proposal_artifact_sha256": "The file hash binds Exp6745 output lineage.",
    "source_stream_artifact_sha256": "The file hash binds the unchanged source CNFs.",
    "source_targetable_row_ids": "The row-derived IDs freeze the 21 source candidates.",
    "mutation_operators": "The row-derived map freezes each preregistered operator.",
    "rows": "Each mutation keeps its source, local change, and exact failure evidence.",
    "targetable_row_count": "The row count measures the cold-audited panel size.",
    "counts_by_family": "Family counts prove coverage across every held proof family.",
    "counts_by_error_class": "Class counts prove coverage across every target error.",
    "exact_valid_mutations": "Zero prevents valid proofs from entering an invalid panel.",
    "duplicate_rows": "Zero proves each source and operator identity occurs once.",
    "proof_preserving_relabel_receipts": "Pair receipts prove relabel rows stay aligned.",
    "future_or_answer_features_read": "An empty list records the answer-blind feature boundary.",
    "cold_replay_receipt": "A fresh process rechecks parsing, encoding, and exact failure.",
    "targetable_panel_ready": "This exact Exp6769 gate reports fixture readiness only.",
    "gate_check_summary": "Each fixture gate records its expected and observed value.",
    "verifier_is_oracle": "False states that exact checks evaluate but do not generate rows.",
    "verdict_class": "The closed class makes the terminal result machine-readable.",
    "honest_verdict": "The terminal text reports readiness without a quality claim.",
}

_SAT_TERM = re.compile(r"x([1-9][0-9]*)=(-?[0-9]+)")
_UNSAT_TERM = re.compile(r"c([1-9][0-9]*)")


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically so evidence hashes replay exactly."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text in the repository's explicit receipt format."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value after canonical serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    """Hash file bytes and keep missing input visible to blocked artifacts."""

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
    """Build one stable gate row with its actual observed value."""

    return {"check": check, "expected": expected, "observed": observed, "passed": passed}


def _serialized_artifact_sha256(artifact: Mapping[str, Any]) -> str:
    """Rebuild the sorted JSON form used by Exp6745's atomic writer."""

    encoded = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return sha256_text(encoded)


def _targetable_rows(replay: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Select only the frozen grammar-targetable rows without exact outcomes."""

    return [
        row
        for row in replay.get("rows", [])
        if isinstance(row, Mapping)
        and row.get("grammar_failures", {}).get("environment_grammar_targetable") is True
    ]


def evaluate_preconditions(
    replay: Mapping[str, Any],
    proposal: Mapping[str, Any],
    stream: Mapping[str, Any],
) -> JsonDict:
    """Check denominator, hashes, target count, and both source lineages."""

    rows = [row for row in replay.get("rows", []) if isinstance(row, Mapping)]
    row_ids = [str(row.get("row_id", "")) for row in rows]
    hash_rows = sum(
        isinstance(row.get("original_output_text"), str)
        and row.get("original_output_sha256") == sha256_text(str(row["original_output_text"]))
        and isinstance(row.get("normalized_output_text"), str)
        and row.get("normalized_output_sha256") == sha256_text(str(row["normalized_output_text"]))
        and row.get("original_output_hash_matches") is True
        and row.get("evidence_preserved") is True
        for row in rows
    )
    targetable = _targetable_rows(replay)
    proposal_rows = {
        str(row.get("row_id")): row for row in proposal.get("rows", []) if isinstance(row, Mapping)
    }
    stream_rows = {
        str(row.get("row_id")): row for row in stream.get("rows", []) if isinstance(row, Mapping)
    }
    proposal_lineage_failures: list[str] = []
    stream_lineage_failures: list[str] = []
    for row in rows:
        row_id = str(row.get("row_id"))
        source_id = str(row.get("source_row", {}).get("row_id"))
        proposal_row = proposal_rows.get(row_id)
        stream_row = stream_rows.get(source_id)
        if (
            proposal_row is None
            or row.get("source_artifact_row_sha256") != proposal_row.get("row_sha256")
            or row.get("original_output_text") != proposal_row.get("raw_output")
            or row.get("original_output_sha256") != proposal_row.get("raw_output_sha256")
            or source_id != str(proposal_row.get("source_row_id"))
            or row.get("source_row", {}).get("row_sha256") != proposal_row.get("source_row_sha256")
        ):
            proposal_lineage_failures.append(row_id)
        if (
            stream_row is None
            or row.get("source_row", {}).get("exact_stream_row_sha256")
            != stream_row.get("row_sha256")
            or row.get("source_row", {}).get("row_sha256") != stream_row.get("row_sha256")
        ):
            stream_lineage_failures.append(row_id)
    expected_proposal_hash = _serialized_artifact_sha256(proposal)
    checks = [
        _check(
            "exp6755_transport_reparse_ready",
            True,
            replay.get("transport_reparse_ready"),
            replay.get("transport_reparse_ready") is True,
        ),
        _check(
            "exp6755_row_count",
            EXPECTED_REPLAY_ROW_COUNT,
            len(rows),
            len(rows) == EXPECTED_REPLAY_ROW_COUNT,
        ),
        _check(
            "exp6755_unique_row_ids",
            EXPECTED_REPLAY_ROW_COUNT,
            len(set(row_ids)),
            len(row_ids) == EXPECTED_REPLAY_ROW_COUNT
            and len(set(row_ids)) == EXPECTED_REPLAY_ROW_COUNT,
        ),
        _check(
            "exp6755_raw_and_normalized_hashes",
            EXPECTED_REPLAY_ROW_COUNT,
            hash_rows,
            hash_rows == EXPECTED_REPLAY_ROW_COUNT,
        ),
        _check(
            "exp6755_targetable_source_rows",
            EXPECTED_TARGETABLE_SOURCE_COUNT,
            len(targetable),
            len(targetable) == EXPECTED_TARGETABLE_SOURCE_COUNT,
        ),
        _check(
            "exp6745_artifact_hash_lineage",
            expected_proposal_hash,
            replay.get("source_artifact_sha256"),
            replay.get("source_artifact_sha256") == expected_proposal_hash,
        ),
        _check(
            "exp6745_source_row_lineage",
            [],
            proposal_lineage_failures,
            not proposal_lineage_failures,
        ),
        _check(
            "exp6744_cnf_lineage",
            [],
            stream_lineage_failures,
            not stream_lineage_failures,
        ),
    ]
    return {"all_passed": all(row["passed"] for row in checks), "checks": checks}


def first_failed_check(summary: Mapping[str, Any]) -> JsonDict:
    """Return the first failed gate or one explicit all-passed receipt."""

    for row in summary.get("checks", []):
        if row.get("passed") is not True:
            return deepcopy(dict(row))
    return {"check": "all_preconditions", "expected": True, "observed": True, "passed": True}


def project_source_candidates(
    replay: Mapping[str, Any],
    proposal: Mapping[str, Any],
    stream: Mapping[str, Any],
) -> list[JsonDict]:
    """Copy only mutation-safe fields into the operator feature boundary."""

    proposal_rows = {
        str(row.get("row_id")): row for row in proposal.get("rows", []) if isinstance(row, Mapping)
    }
    stream_rows = {
        str(row.get("row_id")): row for row in stream.get("rows", []) if isinstance(row, Mapping)
    }
    candidates: list[JsonDict] = []
    for row in sorted(_targetable_rows(replay), key=lambda value: str(value.get("row_id"))):
        row_id = str(row["row_id"])
        source_row_id = str(row["source_row"]["row_id"])
        proposal_row = proposal_rows[row_id]
        stream_row = stream_rows[source_row_id]
        cnf = deepcopy(stream_row["cnf"])
        candidates.append(
            {
                "source_panel_row_id": row_id,
                "source_replay_row_sha256": row["row_sha256"],
                "source_proposal_row_sha256": proposal_row["row_sha256"],
                "source_stream_row_id": source_row_id,
                "source_stream_row_sha256": stream_row["row_sha256"],
                "model": deepcopy(row["model"]),
                "family": row["family"],
                "size": proposal_row["size_bin"],
                "split": proposal_row["split"],
                "pair_id": proposal_row["pair_id"],
                "pair_role": proposal_row["pair_role"],
                "source_raw_output_text": row["original_output_text"],
                "source_raw_output_sha256": row["original_output_sha256"],
                "source_output_text": row["normalized_output_text"],
                "source_output_sha256": row["normalized_output_sha256"],
                "cnf": cnf,
                "source_problem_sha256": sha256_json(cnf),
            }
        )
    return candidates


def _fallback_bit(source: Mapping[str, Any], variable: int) -> int:
    """Fill a missing structural slot from frozen public provenance only."""

    digest = hashlib.sha256(
        f"{RANDOM_SEED}|{source['source_panel_row_id']}|{variable}".encode()
    ).digest()
    return digest[0] & 1


def _binary_base(source: Mapping[str, Any]) -> tuple[str, dict[int, int]]:
    """Build a complete answer-blind SAT shape from observed assignment tokens."""

    observed: dict[int, int] = {}
    for match in _SAT_TERM.finditer(str(source["source_output_text"])):
        variable = int(match.group(1))
        if variable not in observed:
            observed[variable] = abs(int(match.group(2))) % 2
    n_vars = int(source["cnf"]["n_vars"])
    values = {
        variable: observed.get(variable, _fallback_bit(source, variable))
        for variable in range(1, n_vars + 1)
    }
    text = "SAT " + " ".join(f"x{variable}={values[variable]}" for variable in values)
    return text, values


def _region(
    *,
    kind: str,
    before_span: tuple[int, int],
    after_span: tuple[int, int],
    before_fragment: str,
    after_fragment: str,
) -> JsonDict:
    """Describe the smallest changed certificate region for later audit."""

    return {
        "kind": kind,
        "before_span": list(before_span),
        "after_span": list(after_span),
        "before_fragment": before_fragment,
        "after_fragment": after_fragment,
        "attributable": True,
        "smallest_responsible_region": True,
    }


def build_operator_mutations(source: Mapping[str, Any]) -> list[JsonDict]:
    """Apply all six deterministic operators without consulting exact authority."""

    before_sat, values = _binary_base(source)
    n_vars = int(source["cnf"]["n_vars"])
    n_clauses = len(source["cnf"]["clauses"])

    undefined_term = f"x{n_vars + 1}=0"
    undefined_after = f"{before_sat} {undefined_term}"
    undefined_region = _region(
        kind="appended_variable_term",
        before_span=(len(before_sat), len(before_sat)),
        after_span=(len(before_sat) + 1, len(undefined_after)),
        before_fragment="",
        after_fragment=undefined_term,
    )

    clause_seed = int(
        sha256_text(str(source["source_output_sha256"])).removeprefix("sha256:")[:8], 16
    )
    valid_clause = 1 + clause_seed % n_clauses
    invalid_before = f"UNSAT c{valid_clause}"
    invalid_term = f"c{n_clauses + 1}"
    invalid_after = f"UNSAT {invalid_term}"
    before_clause_start = invalid_before.index("c")
    after_clause_start = invalid_after.index("c")
    invalid_region = _region(
        kind="clause_reference",
        before_span=(before_clause_start, len(invalid_before)),
        after_span=(after_clause_start, len(invalid_after)),
        before_fragment=invalid_before[before_clause_start:],
        after_fragment=invalid_term,
    )

    value_token = f"x1={values[1]}"
    value_start = before_sat.index(value_token) + len("x1=")
    non_binary_after = before_sat[:value_start] + "2" + before_sat[value_start + 1 :]
    non_binary_region = _region(
        kind="assignment_value",
        before_span=(value_start, value_start + 1),
        after_span=(value_start, value_start + 1),
        before_fragment=str(values[1]),
        after_fragment="2",
    )

    duplicate_term = f"x1={1 - values[1]}"
    duplicate_after = f"{before_sat} {duplicate_term}"
    duplicate_region = _region(
        kind="appended_duplicate_term",
        before_span=(len(before_sat), len(before_sat)),
        after_span=(len(before_sat) + 1, len(duplicate_after)),
        before_fragment="",
        after_fragment=duplicate_term,
    )

    terms = before_sat.split()
    removed_last = terms[-1]
    missing_after = " ".join(terms[:-1])
    missing_start = before_sat.rindex(removed_last)
    missing_region = _region(
        kind="missing_required_slot",
        before_span=(missing_start, len(before_sat)),
        after_span=(len(missing_after), len(missing_after)),
        before_fragment=removed_last,
        after_fragment="",
    )

    premature_after = " ".join(terms[:2])
    removed_suffix = before_sat[len(premature_after) + 1 :]
    premature_region = _region(
        kind="premature_terminal_boundary",
        before_span=(len(premature_after) + 1, len(before_sat)),
        after_span=(len(premature_after), len(premature_after)),
        before_fragment=removed_suffix,
        after_fragment="",
    )
    values_by_class = {
        "undefined_variable": (before_sat, undefined_after, undefined_region),
        "invalid_clause": (invalid_before, invalid_after, invalid_region),
        "non_binary_value": (before_sat, non_binary_after, non_binary_region),
        "duplicate_evidence": (before_sat, duplicate_after, duplicate_region),
        "missing_evidence": (before_sat, missing_after, missing_region),
        "premature_terminal": (before_sat, premature_after, premature_region),
    }
    return [
        {
            "mutation_operator": OPERATOR_NAMES[error_class],
            "error_class": error_class,
            "before_certificate": values_by_class[error_class][0],
            "after_certificate": values_by_class[error_class][1],
            "target_region": values_by_class[error_class][2],
            "operator_seed": RANDOM_SEED,
        }
        for error_class in ERROR_CLASSES
    ]


def parse_counterfactual_certificate(text: str) -> JsonDict:
    """Parse typed certificate structure while retaining invalid environments."""

    base: JsonDict = {
        "parser_id": PARSER_ID,
        "parser_status": "malformed",
        "claim": None,
        "terms": [],
        "sat_bindings": [],
        "core_clause_indices": [],
        "parse_failure": None,
    }
    stripped = text.strip()
    if not stripped:
        return {**base, "parse_failure": "empty_output"}
    parts = stripped.split()
    claim = parts[0]
    if claim not in {"SAT", "UNSAT"}:
        return {**base, "parse_failure": "unknown_claim"}
    if len(parts) == 1:
        return {**base, "parse_failure": "missing_terms"}
    terms = parts[1:] if claim == "SAT" else " ".join(parts[1:]).replace(",", " ").split()
    if claim == "SAT":
        matches = [_SAT_TERM.fullmatch(term) for term in terms]
        if any(match is None for match in matches):
            return {**base, "claim": claim, "terms": terms, "parse_failure": "invalid_sat_term"}
        bindings = [
            {"variable": int(match.group(1)), "value": int(match.group(2))}
            for match in matches
            if match is not None
        ]
        core: list[int] = []
    else:
        unsat_matches = [_UNSAT_TERM.fullmatch(term) for term in terms]
        if any(match is None for match in unsat_matches):
            return {
                **base,
                "claim": claim,
                "terms": terms,
                "parse_failure": "invalid_unsat_term",
            }
        bindings = []
        core = [int(match.group(1)) for match in unsat_matches if match is not None]
    return {
        **base,
        "parser_status": "parseable",
        "claim": claim,
        "terms": terms,
        "sat_bindings": bindings,
        "core_clause_indices": core,
    }


def detect_error_classes(cnf: Mapping[str, Any], parsed: Mapping[str, Any]) -> list[str]:
    """Classify environment failures without using an answer or exact result."""

    found: set[str] = set()
    if parsed.get("parser_status") != "parseable":
        return []
    if parsed.get("claim") == "SAT":
        bindings = parsed.get("sat_bindings", [])
        variables = [int(binding["variable"]) for binding in bindings]
        values = [int(binding["value"]) for binding in bindings]
        n_vars = int(cnf["n_vars"])
        if any(variable < 1 or variable > n_vars for variable in variables):
            found.add("undefined_variable")
        if any(value not in {0, 1} for value in values):
            found.add("non_binary_value")
        if any(count > 1 for count in Counter(variables).values()):
            found.add("duplicate_evidence")
        missing = set(range(1, n_vars + 1)) - set(variables)
        if missing:
            found.add("premature_terminal" if len(bindings) == 1 else "missing_evidence")
    else:
        clauses = [int(value) for value in parsed.get("core_clause_indices", [])]
        if any(value < 1 or value > len(cnf["clauses"]) for value in clauses):
            found.add("invalid_clause")
        if any(count > 1 for count in Counter(clauses).values()):
            found.add("duplicate_evidence")
    return [error_class for error_class in ERROR_CLASSES if error_class in found]


def _structural_constraints(parsed: Mapping[str, Any]) -> JsonDict:
    """Preserve invalid typed values so the exact checker can reject them."""

    if parsed["claim"] == "SAT":
        grouped: dict[int, set[Any]] = defaultdict(set)
        for binding in parsed["sat_bindings"]:
            value = int(binding["value"])
            grouped[int(binding["variable"])].add(bool(value) if value in {0, 1} else value)
        bindings = [
            {"variable": variable, "values": sorted(values)}
            for variable, values in sorted(grouped.items())
        ]
        core: list[int] = []
    else:
        bindings = []
        core = sorted(set(int(value) for value in parsed["core_clause_indices"]))
    return {
        "claim": parsed["claim"],
        "bindings": bindings,
        "core_clause_indices": core,
    }


def _encoder_receipt(
    module: Any,
    parsed: Mapping[str, Any],
    cnf: Mapping[str, Any],
    error_class: str,
) -> JsonDict:
    """Run one frozen encoder and retain either constraints or its rejection."""

    receipt: JsonDict = {
        "attempted": True,
        "encoder_id": module.ENCODER_ID,
        "accepted": False,
        "normalized_constraints": None,
        "error": None,
        "rejection_class": None,
        "exact_check": {
            "attempted": False,
            "authority_available": True,
            "valid": None,
            "reason": "encoder_rejected",
            "checked_assignment_count": 0,
        },
    }
    try:
        encoded = module.encode_certificate(parsed)
    except ValueError as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["rejection_class"] = error_class
        return receipt
    exact = frozen.exact_check_constraints(cnf, encoded["normalized_constraints"])
    receipt.update(
        {
            "accepted": True,
            "normalized_constraints": encoded["normalized_constraints"],
            "exact_check": exact,
        }
    )
    return receipt


def _target_region_is_valid(before: str, after: str, target_region: Mapping[str, Any]) -> bool:
    """Verify that recorded spans reproduce both changed fragments exactly."""

    try:
        before_start, before_end = [int(value) for value in target_region["before_span"]]
        after_start, after_end = [int(value) for value in target_region["after_span"]]
    except (KeyError, TypeError, ValueError):
        return False
    return (
        0 <= before_start <= before_end <= len(before)
        and 0 <= after_start <= after_end <= len(after)
        and before[before_start:before_end] == target_region.get("before_fragment")
        and after[after_start:after_end] == target_region.get("after_fragment")
        and target_region.get("attributable") is True
        and target_region.get("smallest_responsible_region") is True
    )


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash a mutation row without its self-referential checksum field."""

    return sha256_json({key: value for key, value in row.items() if key != "row_sha256"})


def evaluate_mutation(source: Mapping[str, Any], mutation: Mapping[str, Any]) -> JsonDict:
    """Parse, encode, and exact-check one already generated mutation."""

    before = str(mutation["before_certificate"])
    after = str(mutation["after_certificate"])
    parser_receipt = parse_counterfactual_certificate(after)
    if parser_receipt["parser_status"] != "parseable":
        raise ValueError("unparsable_mutation")
    detected = detect_error_classes(source["cnf"], parser_receipt)
    if detected != [mutation["error_class"]]:
        raise ValueError(f"mutation_error_classes:{detected!r}")
    if not _target_region_is_valid(before, after, mutation["target_region"]):
        raise ValueError("unattributable_target_region")
    receipt_a = _encoder_receipt(
        encoder_a, parser_receipt, source["cnf"], str(mutation["error_class"])
    )
    receipt_b = _encoder_receipt(
        encoder_b, parser_receipt, source["cnf"], str(mutation["error_class"])
    )
    structural_constraints = _structural_constraints(parser_receipt)
    exact = frozen.exact_check_constraints(source["cnf"], structural_constraints)
    if exact["valid"] is True:
        raise ValueError("exact_valid_mutation")
    problem_hash = sha256_json(source["cnf"])
    candidate_identity = sha256_json(
        {
            "source_panel_row_id": source["source_panel_row_id"],
            "source_problem_sha256": problem_hash,
            "mutation_operator": mutation["mutation_operator"],
            "after_certificate": after,
        }
    )
    row: JsonDict = {
        "schema": ROW_SCHEMA,
        "row_id": f"{source['source_panel_row_id']}|{mutation['error_class']}",
        "candidate_identity_sha256": candidate_identity,
        "family": source["family"],
        "size": source["size"],
        "split": source["split"],
        "pair_id": source["pair_id"],
        "pair_role": source["pair_role"],
        "cnf": deepcopy(source["cnf"]),
        "source_problem_sha256": source["source_problem_sha256"],
        "source_problem_after_sha256": problem_hash,
        "source_problem_unchanged": source["source_problem_sha256"] == problem_hash,
        "source_panel_row_id": source["source_panel_row_id"],
        "source_replay_row_sha256": source["source_replay_row_sha256"],
        "source_proposal_row_sha256": source["source_proposal_row_sha256"],
        "source_stream_row_id": source["source_stream_row_id"],
        "source_stream_row_sha256": source["source_stream_row_sha256"],
        "source_model": deepcopy(source["model"]),
        "source_raw_output_text": source["source_raw_output_text"],
        "source_raw_output_sha256": source["source_raw_output_sha256"],
        "source_output_text": source["source_output_text"],
        "source_output_sha256": source["source_output_sha256"],
        "source_candidate": deepcopy(dict(source)),
        "mutation_operator": mutation["mutation_operator"],
        "operator_seed": mutation["operator_seed"],
        "error_class": mutation["error_class"],
        "before_certificate": before,
        "after_certificate": after,
        "target_region": deepcopy(mutation["target_region"]),
        "parser_receipt": parser_receipt,
        "detected_error_classes": detected,
        "encoder_a_receipt": receipt_a,
        "encoder_b_receipt": receipt_b,
        "encoder_agreement": (
            receipt_a["accepted"] is True
            and receipt_b["accepted"] is True
            and receipt_a["normalized_constraints"] == receipt_b["normalized_constraints"]
        ),
        "exact_failure_receipt": {
            **exact,
            "checker_id": "carnot.exp6745.exact_check_constraints",
            "normalized_constraints": structural_constraints,
        },
        "exact_valid": False,
        "row_sha256": "",
    }
    row["row_sha256"] = row_checksum(row)
    return row


def _counts(rows: Sequence[Mapping[str, Any]], field: str) -> JsonDict:
    """Count one retained row field and sort keys for stable artifacts."""

    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def recompute_row_aggregates(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Rebuild all panel counts and uniqueness values from rows only."""

    identities = [str(row.get("candidate_identity_sha256")) for row in rows]
    return {
        "targetable_row_count": len(rows),
        "counts_by_family": _counts(rows, "family"),
        "counts_by_error_class": _counts(rows, "error_class"),
        "exact_valid_mutations": sum(row.get("exact_valid") is True for row in rows),
        "duplicate_rows": len(identities) - len(set(identities)),
    }


def build_relabel_receipts(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Derive complete base/relabel invariance evidence from mutation rows."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("pair_id")), str(row.get("error_class")))].append(row)
    receipts: list[JsonDict] = []
    for (pair_id, error_class), pair_rows in sorted(grouped.items()):
        by_role = {str(row.get("pair_role")): row for row in pair_rows}
        if set(by_role) != {"base", "relabel"} or len(pair_rows) != 2:
            continue
        base = by_role["base"]
        relabel = by_role["relabel"]
        same_split = base.get("split") == relabel.get("split")
        before_match = base.get("before_certificate") == relabel.get("before_certificate")
        after_match = base.get("after_certificate") == relabel.get("after_certificate")
        mutation_signature_match = (
            base.get("mutation_operator") == relabel.get("mutation_operator")
            and base.get("error_class") == relabel.get("error_class")
            and base.get("target_region", {}).get("kind")
            == relabel.get("target_region", {}).get("kind")
            and base.get("parser_receipt", {}).get("parser_status")
            == relabel.get("parser_receipt", {}).get("parser_status")
            and base.get("exact_failure_receipt", {}).get("reason")
            == relabel.get("exact_failure_receipt", {}).get("reason")
        )
        receipts.append(
            {
                "pair_id": pair_id,
                "error_class": error_class,
                "mutation_operator": base.get("mutation_operator"),
                "base_row_id": base.get("row_id"),
                "relabel_row_id": relabel.get("row_id"),
                "split": base.get("split"),
                "same_split": same_split,
                "before_certificates_match": before_match,
                "after_certificates_match": after_match,
                "mutation_signature_match": mutation_signature_match,
                "pair_invariance_passed": (
                    same_split and before_match and after_match and mutation_signature_match
                ),
            }
        )
    return receipts


def cold_replay_rows(rows: Sequence[Mapping[str, Any]], producer_pid: int) -> JsonDict:
    """Rebuild every row in this process and report any receipt mismatch."""

    mismatches: list[JsonDict] = []
    for row in rows:
        mutation = {
            "mutation_operator": row.get("mutation_operator"),
            "error_class": row.get("error_class"),
            "before_certificate": row.get("before_certificate"),
            "after_certificate": row.get("after_certificate"),
            "target_region": row.get("target_region"),
            "operator_seed": row.get("operator_seed"),
        }
        try:
            replayed = evaluate_mutation(row["source_candidate"], mutation)
        except (KeyError, TypeError, ValueError) as error:
            mismatches.append(
                {"row_id": row.get("row_id"), "reason": f"{type(error).__name__}: {error}"}
            )
            continue
        if replayed != dict(row):
            mismatches.append({"row_id": row.get("row_id"), "reason": "receipt_mismatch"})
    cold_pid = os.getpid()
    return {
        "producer_pid": int(producer_pid),
        "cold_pid": cold_pid,
        "fresh_process": cold_pid != int(producer_pid),
        "replayed_row_count": len(rows),
        "rows_sha256": sha256_json(rows),
        "mismatches": mismatches,
        "all_passed": not mismatches and cold_pid != int(producer_pid),
    }


def _write_json_payload_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Synchronize and atomically replace one local JSON receipt."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".exp6768-", delete=False
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


def cold_replay_main(input_path: Path, output_path: Path) -> int:
    """Run the cold worker contract without reading any upstream artifact."""

    payload = load_json_object(input_path)
    receipt = cold_replay_rows(payload.get("rows", []), int(payload["producer_pid"]))
    _write_json_payload_atomic(output_path, receipt)
    return 0


def run_cold_replay(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Launch a fresh Python process for parser, encoder, and exact replay."""

    producer_pid = os.getpid()
    with tempfile.TemporaryDirectory(prefix="carnot-exp6768-") as directory:
        work = Path(directory)
        input_path = work / "input.json"
        output_path = work / "output.json"
        _write_json_payload_atomic(input_path, {"producer_pid": producer_pid, "rows": rows})
        command = [
            sys.executable,
            "-m",
            "carnot.experiment_6768_targetable_proof_panel_expansion",
            "--cold-replay-input",
            str(input_path),
            "--cold-replay-output",
            str(output_path),
        ]
        environment = os.environ.copy()
        python_path = str(REPO_ROOT / "python")
        environment["PYTHONPATH"] = python_path + (
            os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
        )
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0 or not output_path.is_file():
            return {
                "producer_pid": producer_pid,
                "cold_pid": None,
                "fresh_process": False,
                "replayed_row_count": 0,
                "rows_sha256": sha256_json(rows),
                "mismatches": [
                    {
                        "row_id": None,
                        "reason": f"cold_worker_exit_{completed.returncode}",
                    }
                ],
                "all_passed": False,
                "stderr": completed.stderr[-2000:],
            }
        receipt = load_json_object(output_path)
        receipt["worker_exit_code"] = completed.returncode
        return receipt


def _code_file_hashes() -> JsonDict:
    """Bind every implementation used by mutation and exact replay."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in CODE_PATHS}


def build_reproducibility_receipt(
    rows: Sequence[Mapping[str, Any]],
    source_artifact_sha256: str,
    source_proposal_artifact_sha256: str,
    source_stream_artifact_sha256: str,
    *,
    code_files: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Bind frozen inputs, code, fixed seed, and retained mutation rows."""

    files = deepcopy(dict(code_files)) if code_files is not None else _code_file_hashes()
    input_hash = sha256_json(
        {
            "replay_artifact": source_artifact_sha256,
            "proposal_artifact": source_proposal_artifact_sha256,
            "stream_artifact": source_stream_artifact_sha256,
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
    """Build the complete field shape shared by ready and blocked results."""

    return {
        "schema": SCHEMA,
        "experiment": 6768,
        "title": "Exact-invalid targetable proof panel expansion",
        "run_date": date,
        "status": "",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "reproducibility_receipt": {},
        "source_artifact_sha256": "missing",
        "source_proposal_artifact_sha256": "not_checked",
        "source_stream_artifact_sha256": "not_checked",
        "source_targetable_row_ids": [],
        "mutation_operators": {},
        "rows": [],
        **recompute_row_aggregates([]),
        "proof_preserving_relabel_receipts": [],
        "future_or_answer_features_read": [],
        "cold_replay_receipt": {},
        "targetable_panel_ready": False,
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "",
    }


def _set_reproducibility(artifact: JsonDict) -> None:
    """Populate the stable component receipt and top-level checksum."""

    receipt = build_reproducibility_receipt(
        artifact["rows"],
        str(artifact["source_artifact_sha256"]),
        str(artifact["source_proposal_artifact_sha256"]),
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
    source_proposal_artifact_sha256: str = "not_checked",
    source_stream_artifact_sha256: str = "not_checked",
) -> JsonDict:
    """Build a full terminal artifact for one frozen-input failure."""

    artifact = _base_artifact(date, duration_s)
    artifact.update(
        {
            "status": "complete_blocked_targetable_panel",
            "source_artifact_sha256": source_artifact_sha256,
            "source_proposal_artifact_sha256": source_proposal_artifact_sha256,
            "source_stream_artifact_sha256": source_stream_artifact_sha256,
            "gate_check_summary": {
                "all_passed": False,
                "failed_check": failed_check,
                "expected": expected,
                "observed": observed,
            },
            "verdict_class": "blocked",
            "honest_verdict": (
                f"complete_blocked_targetable_panel: {failed_check} expected {expected!r}, "
                f"observed {observed!r}"
            ),
        }
    )
    _set_reproducibility(artifact)
    return artifact


def _source_ids_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Freeze source IDs through the rows that actually entered the panel."""

    return sorted({str(row.get("source_panel_row_id")) for row in rows})


def _operators_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Derive the one class-to-operator map from retained rows."""

    values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        values[str(row.get("error_class"))].add(str(row.get("mutation_operator")))
    return {key: sorted(items)[0] for key, items in sorted(values.items()) if len(items) == 1}


def build_artifact(
    *,
    date: str,
    duration_s: float,
    rows: Sequence[Mapping[str, Any]],
    source_artifact_sha256: str,
    source_proposal_artifact_sha256: str,
    source_stream_artifact_sha256: str,
    preconditions: Mapping[str, Any],
    cold_replay_receipt: Mapping[str, Any],
) -> JsonDict:
    """Build a complete artifact and derive every fixture gate from receipts."""

    retained_rows = [deepcopy(dict(row)) for row in rows]
    reduction = recompute_row_aggregates(retained_rows)
    pair_receipts = build_relabel_receipts(retained_rows)
    source_ids = _source_ids_from_rows(retained_rows)
    operators = _operators_from_rows(retained_rows)
    family_coverage = (
        set(reduction["counts_by_family"]) == set(HELD_FAMILIES)
        and min(reduction["counts_by_family"].values(), default=0) >= MINIMUM_ROWS_PER_STRATUM
    )
    class_coverage = (
        set(reduction["counts_by_error_class"]) == set(ERROR_CLASSES)
        and min(reduction["counts_by_error_class"].values(), default=0) >= MINIMUM_ROWS_PER_STRATUM
    )
    row_hashes_match = all(row.get("row_sha256") == row_checksum(row) for row in retained_rows)
    row_contracts_pass = all(
        row.get("parser_receipt", {}).get("parser_status") == "parseable"
        and row.get("detected_error_classes") == [row.get("error_class")]
        and row.get("exact_valid") is False
        and row.get("source_problem_unchanged") is True
        and _target_region_is_valid(
            str(row.get("before_certificate")),
            str(row.get("after_certificate")),
            row.get("target_region", {}),
        )
        for row in retained_rows
    )
    provenance_pass = len(source_ids) == EXPECTED_TARGETABLE_SOURCE_COUNT and all(
        row.get("source_raw_output_sha256") == sha256_text(str(row.get("source_raw_output_text")))
        and row.get("source_output_sha256") == sha256_text(str(row.get("source_output_text")))
        and row.get("source_problem_sha256") == sha256_json(row.get("cnf"))
        for row in retained_rows
    )
    pair_pass = bool(pair_receipts) and all(
        receipt["pair_invariance_passed"] is True for receipt in pair_receipts
    )
    cold_pass = (
        cold_replay_receipt.get("all_passed") is True
        and cold_replay_receipt.get("fresh_process") is True
        and cold_replay_receipt.get("replayed_row_count") == len(retained_rows)
        and cold_replay_receipt.get("rows_sha256") == sha256_json(retained_rows)
    )
    checks = [
        *deepcopy(list(preconditions.get("checks", []))),
        _check(
            "minimum_panel_rows",
            f">={MINIMUM_PANEL_ROWS}",
            len(retained_rows),
            len(retained_rows) >= MINIMUM_PANEL_ROWS,
        ),
        _check(
            "held_family_coverage",
            {family: f">={MINIMUM_ROWS_PER_STRATUM}" for family in HELD_FAMILIES},
            reduction["counts_by_family"],
            family_coverage,
        ),
        _check(
            "target_error_class_coverage",
            {error: f">={MINIMUM_ROWS_PER_STRATUM}" for error in ERROR_CLASSES},
            reduction["counts_by_error_class"],
            class_coverage,
        ),
        _check(
            "answer_and_future_feature_leakage",
            [],
            FUTURE_OR_ANSWER_FEATURES_READ,
            not FUTURE_OR_ANSWER_FEATURES_READ,
        ),
        _check(
            "exact_valid_mutations",
            0,
            reduction["exact_valid_mutations"],
            reduction["exact_valid_mutations"] == 0,
        ),
        _check("duplicate_rows", 0, reduction["duplicate_rows"], reduction["duplicate_rows"] == 0),
        _check(
            "row_hash_and_local_error_contract",
            True,
            row_hashes_match and row_contracts_pass,
            row_hashes_match and row_contracts_pass,
        ),
        _check("source_problem_and_output_provenance", True, provenance_pass, provenance_pass),
        _check("proof_preserving_relabel_invariance", True, pair_pass, pair_pass),
        _check("fresh_process_cold_replay", True, cold_pass, cold_pass),
    ]
    ready = preconditions.get("all_passed") is True and all(row["passed"] for row in checks)
    artifact = _base_artifact(date, duration_s)
    artifact.update(
        {
            "status": "complete" if ready else "complete_partial_targetable_panel",
            "source_artifact_sha256": source_artifact_sha256,
            "source_proposal_artifact_sha256": source_proposal_artifact_sha256,
            "source_stream_artifact_sha256": source_stream_artifact_sha256,
            "source_targetable_row_ids": source_ids,
            "mutation_operators": operators,
            "rows": retained_rows,
            **reduction,
            "proof_preserving_relabel_receipts": pair_receipts,
            "future_or_answer_features_read": deepcopy(FUTURE_OR_ANSWER_FEATURES_READ),
            "cold_replay_receipt": deepcopy(dict(cold_replay_receipt)),
            "targetable_panel_ready": ready,
            "gate_check_summary": {"all_passed": ready, "checks": checks},
            "verdict_class": "positive" if ready else "partial",
            "honest_verdict": (
                f"complete: targetable exact-invalid fixture ready with {len(retained_rows)} "
                "cold-replayed rows; this is not a model-quality claim"
                if ready
                else "complete_partial_targetable_panel: one or more fixture gates failed"
            ),
        }
    )
    _set_reproducibility(artifact)
    return artifact


def _validate_reproducibility(artifact: Mapping[str, Any]) -> bool:
    """Recompute the checksum from recorded component hashes and rows."""

    receipt = artifact.get("reproducibility_receipt", {})
    if not isinstance(receipt, Mapping):
        return False
    expected = build_reproducibility_receipt(
        artifact.get("rows", []),
        str(artifact.get("source_artifact_sha256")),
        str(artifact.get("source_proposal_artifact_sha256")),
        str(artifact.get("source_stream_artifact_sha256")),
        code_files=receipt.get("code_files", {}),
    )
    return (
        dict(receipt) == expected and artifact.get("reproducibility_checksum") == expected["value"]
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject field, row, aggregate, readiness, or checksum drift."""

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
    if artifact.get("future_or_answer_features_read") != []:
        errors.append("answer_feature_leakage")
    if not _validate_reproducibility(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if artifact.get("status") == "complete_blocked_targetable_panel":
        if artifact.get("rows"):
            errors.append("blocked_rows_invalid")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith(
            "complete_blocked_targetable_panel"
        ):
            errors.append("blocked_verdict_prefix_mismatch")
        if artifact.get("targetable_panel_ready") is not False:
            errors.append("blocked_readiness_mismatch")
        return errors

    rows = artifact.get("rows", [])
    reduction = recompute_row_aggregates(rows)
    if any(artifact.get(field) != reduction[field] for field in ROW_DERIVED_FIELDS):
        errors.append("aggregate_recomputation_mismatch")
    if any(row.get("row_sha256") != row_checksum(row) for row in rows):
        errors.append("row_checksum_mismatch")
    if artifact.get("source_targetable_row_ids") != _source_ids_from_rows(rows):
        errors.append("source_row_id_recomputation_mismatch")
    if artifact.get("mutation_operators") != _operators_from_rows(rows):
        errors.append("operator_recomputation_mismatch")
    if artifact.get("proof_preserving_relabel_receipts") != build_relabel_receipts(rows):
        errors.append("relabel_recomputation_mismatch")
    ready = artifact.get("targetable_panel_ready") is True
    if ready != bool(artifact.get("gate_check_summary", {}).get("all_passed")):
        errors.append("readiness_gate_mismatch")
    if ready and artifact.get("verdict_class") != "positive":
        errors.append("ready_verdict_class_mismatch")
    return errors


def write_json_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Validate, synchronize, and atomically replace the result artifact."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(";".join(errors))
    _write_json_payload_atomic(path, artifact)


def _write_blocked(
    *,
    root: Path,
    date: str,
    started: float,
    failed_check: str,
    expected: Any,
    observed: Any,
    replay_hash: str,
    proposal_hash: str = "not_checked",
    stream_hash: str = "not_checked",
) -> JsonDict:
    """Publish one task-owned blocked artifact and stop expansion."""

    artifact = build_blocked_artifact(
        date=date,
        duration_s=time.monotonic() - started,
        failed_check=failed_check,
        expected=expected,
        observed=observed,
        source_artifact_sha256=replay_hash,
        source_proposal_artifact_sha256=proposal_hash,
        source_stream_artifact_sha256=stream_hash,
    )
    write_json_atomic(root / RESULT_PATH, artifact)
    return artifact


def run(date: str, root: Path = REPO_ROOT) -> JsonDict:
    """Check frozen inputs, build mutations, cold-replay, and publish once."""

    started = time.monotonic()
    replay_path = root / UPSTREAM_REPLAY_PATH
    proposal_path = root / UPSTREAM_PROPOSAL_PATH
    stream_path = root / UPSTREAM_STREAM_PATH
    replay_hash = sha256_file(replay_path)
    try:
        replay = load_json_object(replay_path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        return _write_blocked(
            root=root,
            date=date,
            started=started,
            failed_check="exp6755_json_object",
            expected="parseable JSON object",
            observed=f"{type(error).__name__}: {error}",
            replay_hash=replay_hash,
        )
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
            replay_hash=replay_hash,
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
            replay_hash=replay_hash,
            proposal_hash=proposal_hash,
            stream_hash=stream_hash,
        )
    preconditions = evaluate_preconditions(replay, proposal, stream)
    if preconditions["all_passed"] is not True:
        failed = first_failed_check(preconditions)
        return _write_blocked(
            root=root,
            date=date,
            started=started,
            failed_check=str(failed["check"]),
            expected=failed["expected"],
            observed=failed["observed"],
            replay_hash=replay_hash,
            proposal_hash=proposal_hash,
            stream_hash=stream_hash,
        )
    sources = project_source_candidates(replay, proposal, stream)
    try:
        rows = [
            evaluate_mutation(source, mutation)
            for source in sources
            for mutation in build_operator_mutations(source)
        ]
    except (KeyError, TypeError, ValueError) as error:
        return _write_blocked(
            root=root,
            date=date,
            started=started,
            failed_check="mutation_expansion",
            expected="all preregistered mutations retained",
            observed=f"{type(error).__name__}: {error}",
            replay_hash=replay_hash,
            proposal_hash=proposal_hash,
            stream_hash=stream_hash,
        )
    cold_receipt = run_cold_replay(rows)
    artifact = build_artifact(
        date=date,
        duration_s=time.monotonic() - started,
        rows=rows,
        source_artifact_sha256=replay_hash,
        source_proposal_artifact_sha256=proposal_hash,
        source_stream_artifact_sha256=stream_hash,
        preconditions=preconditions,
        cold_replay_receipt=cold_receipt,
    )
    write_json_atomic(root / RESULT_PATH, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed planning date and private cold-worker paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260830")
    parser.add_argument("--cold-replay-input", type=Path)
    parser.add_argument("--cold-replay-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    """Run the producer or its fresh-process cold replay worker."""

    args = parse_args(argv)
    if args.cold_replay_input is not None or args.cold_replay_output is not None:
        if args.cold_replay_input is None or args.cold_replay_output is None:
            raise SystemExit("both cold replay paths are required")
        return cold_replay_main(args.cold_replay_input, args.cold_replay_output)
    artifact = run(str(args.date))
    errors = validate_artifact(artifact)
    if errors:
        raise SystemExit(";".join(errors))
    print(json.dumps({"status": artifact["status"], "artifact": str(RESULT_PATH)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
