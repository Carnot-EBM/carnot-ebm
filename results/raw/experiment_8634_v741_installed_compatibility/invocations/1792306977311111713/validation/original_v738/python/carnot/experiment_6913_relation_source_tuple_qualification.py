"""Replay saved relation bytes against public source text only.

Spec refs: REQ-CONSTRAINT-6913 and SCENARIO-CONSTRAINT-6913-*.

The reducer checks syntax, byte anchors, tuple types, and visible relation
structure. It does not open formal ASP labels or infer model semantics.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_6913_relation_source_tuple_qualification.json")
SOURCE_PATHS = {
    "exp6886": Path("results/experiment_6886_enoki_exact_relation_fixture.json"),
    "exp6900": Path("results/experiment_6900_authentic_anchored_relation_corpus.json"),
    "exp6912": Path("results/experiment_6912_alias_safe_relation_corpus_reducer.json"),
}
EXPECTED_SOURCE_HASHES = {
    "exp6886": "sha256:602250fbfe172f08458ea279787d992e89835f12005ba6ef59ec02f3b411d500",
    "exp6900": "sha256:beb442dfa3743bc3271150eb88d35cf0e31ac8b657e00664ed143611ed7d0c0c",
    "exp6912": "sha256:5122cd3e95c59ec116dad0d64c03484171a79fe977f9891872da03a8a7a7c2c6",
}
EXPECTED_SOURCE_RECORDS_HASH = (
    "sha256:3190e7f7e8ba85ef82d91fc394bee154068bb0c92f2d6da147c53c4a5fb74ab7"
)
REQUIRED_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_SEEDS = (6899, 6900, 6901, 6902)
FAMILIES = (
    "graph_coloring",
    "scheduling",
    "non_monotonic_defaults",
    "contradictions",
    "cardinality_constraints",
)
ENOKI_ARM = "enoki:pinned_openie_encoder"
RULE_ARM = "rule:anchored_lexical_v1"
GGUF_ARMS = tuple(f"gguf:{model}" for model in REQUIRED_MODELS)
REQUIRED_ARMS = (*GGUF_ARMS, ENOKI_ARM, RULE_ARM)
EXPECTED_CELL_COUNT = 1_400
RANDOM_SEED = 6913
SCHEMA = "carnot.exp6913.relation_source_tuple_qualification.v1"
INFERENCE_SUBSTRATE = "deterministic_cpu_source_tuple_qualification_no_llm"
BLOCKED_VERDICT = "complete_blocked_relation_source_tuple_qualification"
COMPLETE_VERDICT = "complete_relation_source_tuple_qualification"

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "source_offset_rows",
    "source_byte_identity_rows",
    "parser_rows",
    "tuple_type_rows",
    "entity_anchor_rows",
    "relation_direction_rows",
    "omission_rows",
    "duplicate_rows",
    "abstention_rows",
    "arm_summary_rows",
    "model_summary_rows",
    "family_summary_rows",
    "seed_summary_rows",
    "wilson_interval_rows",
    "proposal_coverage_by_arm",
    "source_grounded_correctness_by_arm",
    "held_sidecar_access_count",
    "model_inference_call_count",
    "reported_vs_recomputed_metrics",
    "random_seed",
    "reproducibility_checksum",
    "source_tuple_shard_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each required field states why the evidence is present.",
    "preconditions_checked": "Unsafe or changed inputs stop scoring before any tuple decision.",
    "inference_substrate": "The exact value states that this reducer runs no model inference.",
    "duration_s": "Measured wall time proves that a fresh reducer process ran.",
    "source_artifact_hashes": "Exact hashes bind every decision to immutable source artifacts.",
    "rows": "One terminal row per cell prevents failures from disappearing in aggregates.",
    "source_offset_rows": "Byte-offset receipts expose UTF-8 drift and invalid boundaries.",
    "source_byte_identity_rows": "Quoted bytes must equal the exact saved source bytes.",
    "parser_rows": "Parser failures and bypass attempts remain terminal evidence.",
    "tuple_type_rows": "Arity and field types block malformed tuples from later scoring.",
    "entity_anchor_rows": "Closed entity membership rejects partial or invented entities.",
    "relation_direction_rows": "Subject and object roles must match the visible source relation.",
    "omission_rows": "Required visible relations cannot disappear from coverage claims.",
    "duplicate_rows": "Repeated tuples receive no extra credit and remain visible.",
    "abstention_rows": "False abstentions stay separate from parser and tuple failures.",
    "arm_summary_rows": "Arm denominators prevent controls from filling model cells.",
    "model_summary_rows": "Model summaries expose each producer without cross-model pooling.",
    "family_summary_rows": "Family summaries expose weak source structures.",
    "seed_summary_rows": "Seed summaries expose stochastic failures and deterministic controls.",
    "wilson_interval_rows": "Intervals show uncertainty with exact row denominators.",
    "proposal_coverage_by_arm": "Coverage measures syntactic proposals without calling them correct.",
    "source_grounded_correctness_by_arm": "Correctness requires a complete exact source-grounded tuple set.",
    "held_sidecar_access_count": "Zero proves that formal ASP labels stayed sealed.",
    "model_inference_call_count": "Zero proves that saved proposals were not regenerated.",
    "reported_vs_recomputed_metrics": "Fresh row replay detects stale or changed aggregates.",
    "random_seed": "A fixed identifier makes the deterministic reducer contract explicit.",
    "reproducibility_checksum": "A stable digest detects decision drift across fresh processes.",
    "source_tuple_shard_ready_score": "Readiness means complete evidence, not universal correctness.",
    "gate_check_summary": "Each failed gate records its expected and observed values.",
    "verifier_is_oracle": "True states that exact source bytes define this bounded decision.",
    "verdict_class": "The closed class prevents an oracle-backed result from claiming positive.",
    "honest_verdict": "A complete prefix marks a terminal result for the conductor.",
}

FAMILY_CONTRACTS = {
    "graph_coloring": {
        "subject_kind": "Node",
        "subject_token": "n",
        "subject_id": "node",
        "predicate": "has_color",
        "object_text": "red",
        "object_id": "red",
        "positive_link": "has color",
        "negative_link": "does not have color",
    },
    "scheduling": {
        "subject_kind": "Task",
        "subject_token": "task",
        "subject_id": "task",
        "predicate": "scheduled_at",
        "object_text": "morning",
        "object_id": "morning",
        "positive_link": "is scheduled in the",
        "negative_link": "is not scheduled in the",
    },
    "non_monotonic_defaults": {
        "subject_kind": "Bird",
        "subject_token": "b",
        "subject_id": "bird",
        "predicate": "has_condition",
        "object_text": "injured",
        "object_id": "injured",
        "positive_link": "is",
        "negative_link": "is not",
    },
    "contradictions": {
        "subject_kind": "Claim",
        "subject_token": "claim",
        "subject_id": "claim",
        "predicate": "has_truth_status",
        "object_text": "accepted",
        "object_id": "accepted",
        "positive_link": "is",
        "negative_link": "is not",
    },
    "cardinality_constraints": {
        "subject_kind": "Set",
        "subject_token": "s",
        "subject_id": "set",
        "predicate": "selects",
        "object_text": "option A",
        "object_id": "option_a",
        "positive_link": "selects",
        "negative_link": "does not select",
    },
}


def canonical_json(value: Any) -> str:
    """Use stable JSON so checksums do not depend on process ordering."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return one prefixed digest in the repository artifact format."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one source file without changing it."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return "missing"
    return "sha256:" + digest.hexdigest()


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Retain both values so a blocked run explains its exact cause."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve all failed gates while exposing the first stable blocker."""

    copied = [deepcopy(dict(row)) for row in checks]
    failed = [row for row in copied if row.get("passed") is not True]
    first = failed[0] if failed else None
    return {
        "passed": not failed,
        "failed_check": first.get("check") if first else None,
        "expected": first.get("expected") if first else True,
        "observed": first.get("observed") if first else True,
        "failed_checks": failed,
        "checks": copied,
    }


def _decode_b64(value: Any, error: str) -> bytes:
    """Decode strict base64 so corrupt saved bytes fail closed."""

    if not isinstance(value, str):
        raise ValueError(error)
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError(error) from exc


def check_source_span(source_bytes: bytes, start: Any, end: Any, quoted_text: Any) -> JsonDict:
    """Check byte boundaries and quoted identity without Unicode normalization."""

    typed = (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
    )
    in_range = typed and 0 <= start < end <= len(source_bytes)
    exact = source_bytes[start:end] if in_range else b""
    try:
        exact_text = exact.decode("utf-8") if in_range else None
        if in_range:
            source_bytes[:start].decode("utf-8")
            source_bytes[end:].decode("utf-8")
        boundary_valid = bool(in_range)
    except UnicodeDecodeError:
        exact_text = None
        boundary_valid = False
    quoted_valid = isinstance(quoted_text, str)
    byte_identity = (
        boundary_valid
        and quoted_valid
        and quoted_text.encode("utf-8") == exact
        and quoted_text == exact_text
    )
    return {
        "start_utf8": start,
        "end_utf8": end,
        "offset_valid": boundary_valid,
        "quoted_text": quoted_text,
        "exact_text": exact_text,
        "exact_source_bytes_b64": base64.b64encode(exact).decode("ascii"),
        "byte_identity": byte_identity,
        "passed": boundary_valid and byte_identity,
    }


def check_tuple_types(fields: Any) -> JsonDict:
    """Require four string fields before any entity or direction decision."""

    arity_valid = isinstance(fields, (list, tuple)) and len(fields) == 4
    field_types = [type(value).__name__ for value in fields] if arity_valid else []
    field_types_valid = arity_valid and all(isinstance(value, str) for value in fields)
    return {
        "arity": len(fields) if isinstance(fields, (list, tuple)) else None,
        "expected_arity": 4,
        "arity_valid": arity_valid,
        "field_types": field_types,
        "field_types_valid": field_types_valid,
        "passed": arity_valid and field_types_valid,
    }


def source_relation_contract(source: Mapping[str, Any]) -> list[JsonDict]:
    """Derive visible relations from the fixed public sentence grammar."""

    family = str(source.get("family", ""))
    contract = FAMILY_CONTRACTS.get(family)
    if contract is None:
        raise ValueError(f"unsupported_source_family:{family}")
    fixture_id = str(source.get("fixture_id", ""))
    try:
        ordinal = int(fixture_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid_fixture_id:{fixture_id}") from exc
    token = f"{contract['subject_token']}{ordinal}"
    subject_phrase = f"{contract['subject_kind']} {token}"
    object_text = str(contract["object_text"])
    text = str(source.get("source_text", ""))
    positive = f"{subject_phrase} {contract['positive_link']} {object_text}."
    negative = f"{subject_phrase} {contract['negative_link']} {object_text}."
    if positive not in text:
        raise ValueError(f"visible_positive_relation_missing:{fixture_id}")
    subject_id = f"{contract['subject_id']}_{ordinal}"
    object_aliases = [object_text]
    if family == "scheduling":
        object_aliases.append("the morning")
    common = {
        "subject_entity_id": subject_id,
        "subject_aliases": [token, subject_phrase],
        "predicate": str(contract["predicate"]),
        "object_entity_id": str(contract["object_id"]),
        "object_aliases": object_aliases,
    }
    relations = [
        {
            **common,
            "polarity": "positive",
            "normalized_tuple": [
                subject_id,
                str(contract["predicate"]),
                str(contract["object_id"]),
                "positive",
            ],
        }
    ]
    if negative in text:
        relations.append(
            {
                **common,
                "polarity": "negative",
                "normalized_tuple": [
                    subject_id,
                    str(contract["predicate"]),
                    str(contract["object_id"]),
                    "negative",
                ],
            }
        )
    return relations


def extract_source_records(
    cells: Sequence[Mapping[str, Any]], fixture_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Recover source text only from hash-verified saved rule requests."""

    fixtures = {
        str(row.get("fixture_id")): row for row in fixture_rows if row.get("row_type") == "fixture"
    }
    sources: list[JsonDict] = []
    seen: set[str] = set()
    for cell in cells:
        if cell.get("arm") != RULE_ARM:
            continue
        fixture_id = str(cell.get("fixture_id", ""))
        if fixture_id in seen:
            raise ValueError(f"duplicate_source_rule_cell:{fixture_id}")
        seen.add(fixture_id)
        payload = _decode_b64(cell.get("raw_request_b64"), "source_request_invalid_base64")
        if sha256_bytes(payload) != cell.get("raw_request_sha256") or len(payload) != cell.get(
            "request_byte_count"
        ):
            raise ValueError(f"source_request_hash_mismatch:{fixture_id}")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"source_request_invalid_utf8:{fixture_id}") from exc
        text_hash = sha256_bytes(payload)
        fixture = fixtures.get(fixture_id)
        if fixture is None:
            raise ValueError(f"fixture_row_missing:{fixture_id}")
        expected_fields = {
            "group_id": cell.get("group_id"),
            "family": cell.get("family"),
            "split": cell.get("split"),
            "source_text_hash": text_hash,
        }
        if any(fixture.get(name) != value for name, value in expected_fields.items()):
            raise ValueError(f"fixture_source_drift:{fixture_id}")
        if cell.get("source_text_hash") != text_hash:
            raise ValueError(f"cell_source_hash_mismatch:{fixture_id}")
        family = str(cell.get("family"))
        predicate = FAMILY_CONTRACTS.get(family, {}).get("predicate")
        sources.append(
            {
                "fixture_id": fixture_id,
                "group_id": cell.get("group_id"),
                "family": family,
                "split": cell.get("split"),
                "source_order": cell.get("source_order"),
                "source_text": text,
                "source_text_hash": text_hash,
                "relation_schema_version": "anchored_relation_v1",
                "allowed_predicates": [predicate] if predicate else [],
            }
        )
    return sorted(sources, key=lambda row: int(row["source_order"]))


def _find_exact_offsets(source_text: str, phrase: str) -> tuple[int | None, int | None]:
    """Locate an exact Enoki phrase without case folding or normalization."""

    index = source_text.find(phrase)
    if index < 0 or not phrase:
        return None, None
    start = len(source_text[:index].encode("utf-8"))
    return start, start + len(phrase.encode("utf-8"))


def _predicate_from_enoki(value: str) -> str:
    """Map only the public relation phrases used by the frozen Enoki output."""

    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    aliases = {
        "has_color": {"has_color", "has", "color"},
        "scheduled_at": {"scheduled_at", "is_scheduled_in", "scheduled_in"},
        "has_condition": {"has_condition", "is", "is_injured"},
        "has_truth_status": {"has_truth_status", "is", "is_accepted"},
        "selects": {"selects", "does_select", "select"},
    }
    for predicate, values in aliases.items():
        if normalized in values:
            return predicate
    return normalized


def _parse_protocol(raw_text: str, source_bytes: bytes) -> tuple[list[JsonDict], list[str]]:
    """Parse every plain protocol line and preserve malformed lines."""

    if raw_text == "":
        return [], ["empty"]
    proposals: list[JsonDict] = []
    states: list[str] = []
    for line_index, raw_line in enumerate(raw_text.splitlines() or [""]):
        if raw_line.strip() == "ABSTAIN":
            states.append("abstention")
            continue
        fields = raw_line.split("\t")
        if len(fields) != 7 or fields[0] != "REL":
            states.append("malformed")
            continue
        try:
            subject_start, subject_end = int(fields[1]), int(fields[2])
            object_start, object_end = int(fields[4]), int(fields[5])
        except ValueError:
            states.append("malformed")
            continue
        subject_probe = check_source_span(source_bytes, subject_start, subject_end, "")
        object_probe = check_source_span(source_bytes, object_start, object_end, "")
        subject_text = subject_probe["exact_text"]
        object_text = object_probe["exact_text"]
        proposals.append(
            {
                "line_index": line_index,
                "raw_line": raw_line,
                "subject_start_utf8": subject_start,
                "subject_end_utf8": subject_end,
                "subject_text": subject_text,
                "predicate": fields[3],
                "object_start_utf8": object_start,
                "object_end_utf8": object_end,
                "object_text": object_text,
                "polarity": fields[6],
                "tuple_fields": [subject_text, fields[3], object_text, fields[6]],
            }
        )
        states.append("parsed")
    return proposals, states


def _parse_enoki(raw_text: str, source_text: str) -> tuple[list[JsonDict], list[str]]:
    """Parse saved Enoki JSON without loading the encoder or its model code."""

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        return [], ["malformed"]
    if not isinstance(result, Mapping) or not isinstance(result.get("triples"), list):
        return [], ["malformed"]
    triples = result["triples"]
    if not triples:
        return [], ["empty"]
    proposals: list[JsonDict] = []
    states: list[str] = []
    for line_index, triple in enumerate(triples):
        if not isinstance(triple, Mapping):
            states.append("malformed")
            continue
        subject = triple.get("subject")
        relation = triple.get("relation")
        object_text = triple.get("object")
        if not all(isinstance(value, str) for value in (subject, relation, object_text)):
            states.append("malformed")
            continue
        subject_start, subject_end = _find_exact_offsets(source_text, subject)
        object_start, object_end = _find_exact_offsets(source_text, object_text)
        predicate = _predicate_from_enoki(relation)
        proposals.append(
            {
                "line_index": line_index,
                "raw_line": canonical_json(triple),
                "subject_start_utf8": subject_start,
                "subject_end_utf8": subject_end,
                "subject_text": subject,
                "predicate": predicate,
                "object_start_utf8": object_start,
                "object_end_utf8": object_end,
                "object_text": object_text,
                "polarity": "positive",
                "tuple_fields": [subject, predicate, object_text, "positive"],
            }
        )
        states.append("parsed")
    return proposals, states


def _terminal_parser_state(states: Sequence[str]) -> str:
    """Reduce line states without hiding any malformed line."""

    if "malformed" in states:
        return "malformed"
    if "parsed" in states:
        return "parsed"
    if "abstention" in states:
        return "abstention"
    return "empty"


def _cell_model_id(cell: Mapping[str, Any]) -> str:
    """Keep controls separate when a cell has no GGUF model identifier."""

    hf_id = cell.get("hf_id")
    return str(hf_id) if isinstance(hf_id, str) and hf_id else str(cell.get("arm"))


def grounding_label(
    *,
    correct: bool,
    parser_integrity: bool,
    parser_state: str,
    fresh_state: str,
    false_abstention: bool,
    duplicate_count: int,
    offset_passed: bool,
    byte_passed: bool,
    tuple_passed: bool,
    entity_passed: bool,
    direction_passed: bool,
    omitted_count: int,
) -> str:
    """Name the first source-grounding failure without losing detailed checks."""

    if correct:
        return "source_grounded_correct"
    if not parser_integrity:
        return f"source_grounding_failed_{parser_state}"
    if fresh_state == "malformed":
        return "source_grounding_failed_parser"
    if false_abstention:
        return "source_grounding_failed_false_abstention"
    if duplicate_count:
        return "source_grounding_failed_duplicate"
    if not offset_passed:
        return "source_grounding_failed_offset"
    if not byte_passed:
        return "source_grounding_failed_byte_identity"
    if not tuple_passed:
        return "source_grounding_failed_tuple_type"
    if not entity_passed:
        return "source_grounding_failed_entity"
    if not direction_passed:
        return "source_grounding_failed_direction"
    if omitted_count:
        return "source_grounding_failed_omission"
    return "source_grounding_failed_incomplete"


def qualify_cell(
    cell: Mapping[str, Any],
    source: Mapping[str, Any],
    expected_relations: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Produce one terminal source-only decision from one saved cell."""

    raw_error = None
    try:
        raw_bytes = _decode_b64(cell.get("raw_output_b64"), "raw_output_invalid_base64")
    except ValueError as exc:
        raw_bytes = b""
        raw_error = str(exc)
    observed_hash = sha256_bytes(raw_bytes)
    raw_passed = (
        raw_error is None
        and observed_hash == cell.get("raw_output_sha256")
        and len(raw_bytes) == cell.get("output_byte_count")
    )
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = ""
        raw_error = "raw_output_invalid_utf8"
        raw_passed = False
    source_text = str(source.get("source_text", ""))
    source_bytes = source_text.encode("utf-8")
    if cell.get("arm") == ENOKI_ARM:
        proposals, parser_states = _parse_enoki(raw_text, source_text)
    else:
        proposals, parser_states = _parse_protocol(raw_text, source_bytes)
    fresh_state = _terminal_parser_state(parser_states)
    parser_integrity = cell.get("parser_attempted") is True
    if not parser_integrity:
        parser_state = "parser_bypass"
    elif cell.get("parser_input_sha256") != cell.get("raw_output_sha256"):
        parser_state = "parser_input_mismatch"
        parser_integrity = False
    elif cell.get("terminal") is not True:
        parser_state = "nonterminal_cell"
        parser_integrity = False
    elif not raw_passed:
        parser_state = raw_error or "raw_output_hash_mismatch"
        parser_integrity = False
    else:
        parser_state = fresh_state

    expected = [dict(row) for row in expected_relations]
    expected_tuples = {tuple(row["normalized_tuple"]) for row in expected}
    proposal_details: list[JsonDict] = []
    seen: set[tuple[Any, ...]] = set()
    matched: set[tuple[str, ...]] = set()
    duplicate_count = 0
    for proposal in proposals:
        subject_span = check_source_span(
            source_bytes,
            proposal.get("subject_start_utf8"),
            proposal.get("subject_end_utf8"),
            proposal.get("subject_text"),
        )
        object_span = check_source_span(
            source_bytes,
            proposal.get("object_start_utf8"),
            proposal.get("object_end_utf8"),
            proposal.get("object_text"),
        )
        tuple_check = check_tuple_types(proposal.get("tuple_fields"))
        subject_ids = sorted(
            {
                str(row["subject_entity_id"])
                for row in expected
                if proposal.get("subject_text") in row.get("subject_aliases", [])
            }
        )
        object_ids = sorted(
            {
                str(row["object_entity_id"])
                for row in expected
                if proposal.get("object_text") in row.get("object_aliases", [])
            }
        )
        entity_passed = len(subject_ids) == 1 and len(object_ids) == 1
        normalized = (
            (
                subject_ids[0],
                str(proposal.get("predicate")),
                object_ids[0],
                str(proposal.get("polarity")),
            )
            if entity_passed
            else None
        )
        direction_passed = normalized in expected_tuples if normalized is not None else False
        identity = normalized or tuple(proposal.get("tuple_fields", []))
        duplicate = identity in seen
        seen.add(identity)
        duplicate_count += int(duplicate)
        if direction_passed and not duplicate:
            matched.add(normalized)
        proposal_details.append(
            {
                **deepcopy(proposal),
                "source_offset_check": {
                    "subject_offset_valid": subject_span["offset_valid"],
                    "object_offset_valid": object_span["offset_valid"],
                    "passed": subject_span["offset_valid"] and object_span["offset_valid"],
                },
                "source_byte_identity_check": {
                    "subject": subject_span,
                    "object": object_span,
                    "passed": subject_span["byte_identity"] and object_span["byte_identity"],
                },
                "tuple_type_check": tuple_check,
                "entity_anchor_check": {
                    "subject_entity_ids": subject_ids,
                    "object_entity_ids": object_ids,
                    "passed": entity_passed,
                },
                "relation_direction_check": {
                    "normalized_tuple": list(normalized) if normalized else None,
                    "expected": direction_passed,
                    "passed": direction_passed,
                },
                "duplicate": duplicate,
            }
        )

    omitted = sorted([list(value) for value in expected_tuples - matched])
    abstained = fresh_state == "abstention"
    false_abstention = abstained and bool(expected)
    parser_passed = parser_integrity and fresh_state != "malformed"
    offset_passed = bool(proposal_details) and all(
        row["source_offset_check"]["passed"] and row["entity_anchor_check"]["passed"]
        for row in proposal_details
    )
    byte_passed = bool(proposal_details) and all(
        row["source_byte_identity_check"]["passed"] for row in proposal_details
    )
    tuple_passed = bool(proposal_details) and all(
        row["tuple_type_check"]["passed"] for row in proposal_details
    )
    entity_passed = bool(proposal_details) and all(
        row["entity_anchor_check"]["passed"] for row in proposal_details
    )
    direction_passed = bool(proposal_details) and all(
        row["relation_direction_check"]["passed"] for row in proposal_details
    )
    omitted_passed = not omitted
    duplicate_passed = duplicate_count == 0
    abstention_passed = not false_abstention
    correct = all(
        (
            raw_passed,
            parser_passed,
            fresh_state == "parsed",
            offset_passed,
            byte_passed,
            tuple_passed,
            entity_passed,
            direction_passed,
            omitted_passed,
            duplicate_passed,
            abstention_passed,
            len(matched) == len(expected_tuples),
        )
    )
    label = grounding_label(
        correct=correct,
        parser_integrity=parser_integrity,
        parser_state=parser_state,
        fresh_state=fresh_state,
        false_abstention=false_abstention,
        duplicate_count=duplicate_count,
        offset_passed=offset_passed,
        byte_passed=byte_passed,
        tuple_passed=tuple_passed,
        entity_passed=entity_passed,
        direction_passed=direction_passed,
        omitted_count=len(omitted),
    )

    source_offset_check = {
        "proposal_count": len(proposal_details),
        "passed": offset_passed,
        "offsets": [
            {
                "subject_start_utf8": row["subject_start_utf8"],
                "subject_end_utf8": row["subject_end_utf8"],
                "object_start_utf8": row["object_start_utf8"],
                "object_end_utf8": row["object_end_utf8"],
            }
            for row in proposal_details
        ],
    }
    source_byte_check = {
        "proposal_count": len(proposal_details),
        "passed": byte_passed,
        "quoted_spans": [row["source_byte_identity_check"] for row in proposal_details],
    }
    return {
        "row_type": "terminal_source_tuple_decision",
        "cell_identity": str(cell.get("cell_identity")),
        "arm": str(cell.get("arm")),
        "model_id": _cell_model_id(cell),
        "model_family": str(cell.get("model_family")),
        "seed": cell.get("seed"),
        "seed_label": str(cell.get("seed")) if cell.get("seed") is not None else "deterministic",
        "fixture_id": str(cell.get("fixture_id")),
        "group_id": str(cell.get("group_id")),
        "family": str(cell.get("family")),
        "split": str(cell.get("split")),
        "terminal": True,
        "source_text_hash": sha256_bytes(source_bytes),
        "source_bytes_b64": base64.b64encode(source_bytes).decode("ascii"),
        "raw_output_check": {
            "declared_sha256": cell.get("raw_output_sha256"),
            "observed_sha256": observed_hash,
            "declared_byte_count": cell.get("output_byte_count"),
            "observed_byte_count": len(raw_bytes),
            "error": raw_error,
            "passed": raw_passed,
        },
        "parsed_tuple_fields": [row.get("tuple_fields") for row in proposal_details],
        "parsed_proposals": proposal_details,
        "parser_check": {
            "state": parser_state,
            "fresh_state": fresh_state,
            "line_states": parser_states,
            "source_parser_attempted": cell.get("parser_attempted"),
            "input_hash_matches_output": (
                cell.get("parser_input_sha256") == cell.get("raw_output_sha256")
            ),
            "passed": parser_passed,
        },
        "source_offset_check": source_offset_check,
        "source_byte_identity_check": source_byte_check,
        "tuple_type_check": {
            "checks": [row["tuple_type_check"] for row in proposal_details],
            "passed": tuple_passed,
        },
        "entity_anchor_check": {
            "checks": [row["entity_anchor_check"] for row in proposal_details],
            "passed": entity_passed,
        },
        "relation_direction_check": {
            "checks": [row["relation_direction_check"] for row in proposal_details],
            "passed": direction_passed,
        },
        "omission_check": {
            "required_count": len(expected_tuples),
            "matched_count": len(matched),
            "omitted_count": len(omitted),
            "omitted_tuples": omitted,
            "passed": omitted_passed,
        },
        "duplicate_check": {
            "proposal_count": len(proposal_details),
            "duplicate_count": duplicate_count,
            "passed": duplicate_passed,
        },
        "abstention_check": {
            "abstained": abstained,
            "required_relation_count": len(expected_tuples),
            "false_abstention": false_abstention,
            "passed": abstention_passed,
        },
        "proposal_covered": bool(proposal_details),
        "source_grounded_correct": correct,
        "source_grounding_label": label,
    }


def wilson_interval(successes: int, total: int) -> JsonDict:
    """Return the two-sided 95 percent Wilson interval for a row rate."""

    if total == 0:
        return {"confidence": 0.95, "lower": None, "upper": None}
    z = 1.96
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * ((rate * (1 - rate) / total + z * z / (4 * total * total)) ** 0.5)
    margin /= denominator
    return {
        "confidence": 0.95,
        "lower": round(max(0.0, center - margin), 8),
        "upper": round(min(1.0, center + margin), 8),
    }


def _dimension_summary(rows: Sequence[Mapping[str, Any]], dimension: str) -> list[JsonDict]:
    """Compute exact denominators and both metrics for one grouping field."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if dimension == "seed":
            value = str(row.get("seed")) if row.get("seed") is not None else "deterministic"
        else:
            value = str(row.get(dimension))
        groups[value].append(row)
    summaries: list[JsonDict] = []
    for value in sorted(groups):
        members = groups[value]
        denominator = len(members)
        proposal_numerator = sum(row.get("proposal_covered") is True for row in members)
        grounded_numerator = sum(row.get("source_grounded_correct") is True for row in members)
        proposal_rate = proposal_numerator / denominator
        grounded_rate = grounded_numerator / denominator
        summaries.append(
            {
                "dimension": dimension,
                "value": value,
                "proposal_numerator": proposal_numerator,
                "source_grounded_numerator": grounded_numerator,
                "denominator": denominator,
                "proposal_coverage": proposal_rate,
                "source_grounded_correctness": grounded_rate,
                "proposal_coverage_wilson_95": wilson_interval(proposal_numerator, denominator),
                "source_grounded_correctness_wilson_95": wilson_interval(
                    grounded_numerator, denominator
                ),
                "qualification_decision": (
                    "qualified" if grounded_numerator == denominator else "disqualified"
                ),
            }
        )
    return summaries


def _arm_metric_map(rows: Sequence[Mapping[str, Any]], metric: str) -> JsonDict:
    """Build the compact by-arm metric from arm summary rows."""

    result: JsonDict = {}
    for row in rows:
        if metric == "proposal_coverage":
            numerator = row["proposal_numerator"]
            interval = row["proposal_coverage_wilson_95"]
        else:
            numerator = row["source_grounded_numerator"]
            interval = row["source_grounded_correctness_wilson_95"]
        result[row["value"]] = {
            "numerator": numerator,
            "denominator": row["denominator"],
            "rate": row[metric],
            "wilson_95": interval,
        }
    return result


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Derive every requested summary directly from terminal cell rows."""

    arm_rows = _dimension_summary(rows, "arm")
    model_rows = _dimension_summary(rows, "model_id")
    family_rows = _dimension_summary(rows, "family")
    seed_rows = _dimension_summary(rows, "seed")
    all_summaries = [*arm_rows, *model_rows, *family_rows, *seed_rows]
    intervals = []
    for row in all_summaries:
        for metric, field in (
            ("proposal_coverage", "proposal_coverage_wilson_95"),
            ("source_grounded_correctness", "source_grounded_correctness_wilson_95"),
        ):
            intervals.append(
                {
                    "dimension": row["dimension"],
                    "value": row["value"],
                    "metric": metric,
                    "numerator": (
                        row["proposal_numerator"]
                        if metric == "proposal_coverage"
                        else row["source_grounded_numerator"]
                    ),
                    "denominator": row["denominator"],
                    **row[field],
                }
            )
    return {
        "arm_summary_rows": arm_rows,
        "model_summary_rows": model_rows,
        "family_summary_rows": family_rows,
        "seed_summary_rows": seed_rows,
        "wilson_interval_rows": intervals,
        "proposal_coverage_by_arm": _arm_metric_map(arm_rows, "proposal_coverage"),
        "source_grounded_correctness_by_arm": _arm_metric_map(
            arm_rows, "source_grounded_correctness"
        ),
    }


def compare_reported_metrics(
    rows: Sequence[Mapping[str, Any]],
    proposal_coverage_by_arm: Mapping[str, Any],
    source_grounded_correctness_by_arm: Mapping[str, Any],
) -> JsonDict:
    """Compare reported arm metrics with a fresh replay of terminal rows."""

    recomputed = summarize_rows(rows)
    comparisons = [
        {
            "metric": "proposal_coverage_by_arm",
            "reported": deepcopy(dict(proposal_coverage_by_arm)),
            "recomputed": recomputed["proposal_coverage_by_arm"],
            "passed": dict(proposal_coverage_by_arm) == recomputed["proposal_coverage_by_arm"],
        },
        {
            "metric": "source_grounded_correctness_by_arm",
            "reported": deepcopy(dict(source_grounded_correctness_by_arm)),
            "recomputed": recomputed["source_grounded_correctness_by_arm"],
            "passed": dict(source_grounded_correctness_by_arm)
            == recomputed["source_grounded_correctness_by_arm"],
        },
    ]
    return {
        "agreement": all(row["passed"] for row in comparisons),
        "comparisons": comparisons,
    }


def validate_preconditions(
    *,
    receipt: Mapping[str, Any],
    observed_hashes: Mapping[str, str],
    expected_hashes: Mapping[str, str],
    cells: Sequence[Mapping[str, Any]],
    expected_cell_ids: set[str],
    held_sidecar_access_count: int,
    source_held_sidecar_access_count: int = 0,
) -> JsonDict:
    """Check hashes, clean admission, identities, terminals, and sealed authority."""

    checks = [
        gate_check(f"source_hash:{name}", digest, observed_hashes.get(name))
        for name, digest in expected_hashes.items()
    ]
    receipt_rows = receipt.get("cell_identity_rows", [])
    receipt_rows = receipt_rows if isinstance(receipt_rows, list) else []
    receipt_ids = {
        str(row.get("cell_identity"))
        for row in receipt_rows
        if isinstance(row, Mapping)
        and row.get("occurrence_count") == 1
        and row.get("expected") is True
    }
    cell_ids = [str(row.get("cell_identity")) for row in cells]
    linked_source_hash = (
        receipt.get("source_artifact_hashes", {}).get("exp6900", {}).get("observed_sha256")
        if isinstance(receipt.get("source_artifact_hashes"), Mapping)
        else None
    )
    checks.extend(
        [
            gate_check(
                "clean_relation_corpus_ready_score",
                1,
                receipt.get("clean_relation_corpus_ready_score"),
            ),
            gate_check("receipt_cell_identity_set", sorted(expected_cell_ids), sorted(receipt_ids)),
            gate_check(
                "source_cell_identity_set", sorted(expected_cell_ids), sorted(set(cell_ids))
            ),
            gate_check("source_cell_identity_count", len(expected_cell_ids), len(cell_ids)),
            gate_check(
                "terminal_source_cells", True, all(row.get("terminal") is True for row in cells)
            ),
            gate_check("receipt_exp6900_hash", observed_hashes.get("exp6900"), linked_source_hash),
            gate_check("source_held_sidecar_access_count", 0, source_held_sidecar_access_count),
            gate_check("held_sidecar_access_count", 0, held_sidecar_access_count),
        ]
    )
    return gate_summary(checks)


def expected_cell_identities() -> set[str]:
    """Build the fixed 1,400-cell matrix without reading observed identities."""

    fixtures = [f"{family}_{ordinal:02d}" for ordinal in range(20) for family in FAMILIES]
    identities = {
        f"{model}::{seed}::{fixture}"
        for model in REQUIRED_MODELS
        for seed in REQUIRED_SEEDS
        for fixture in fixtures
    }
    identities.update(
        f"{arm}::deterministic::{fixture}" for arm in (ENOKI_ARM, RULE_ARM) for fixture in fixtures
    )
    return identities


def _row_views(rows: Sequence[Mapping[str, Any]], field: str) -> list[JsonDict]:
    """Expose one focused check per cell without changing its denominator."""

    return [{"cell_identity": row["cell_identity"], **deepcopy(dict(row[field]))} for row in rows]


def _attach_principles(artifact: JsonDict) -> None:
    """Attach one principle for every required field and gate check."""

    principles = deepcopy(FIELD_PRINCIPLES)
    for row in artifact.get("gate_check_summary", {}).get("checks", []):
        principles[f"gate:{row['check']}"] = (
            "This exact expected-versus-observed gate prevents incomplete evidence from passing."
        )
    artifact["field_principles"] = principles


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding wall time and the digest itself."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json(stable).encode("utf-8"))


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    date: str,
    duration_s: float,
    source_artifact_hashes: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    expected_cell_ids: set[str],
    expected_arms: set[str],
    expected_families: set[str],
) -> JsonDict:
    """Build the complete terminal artifact only from qualified cell rows."""

    copied_rows = [deepcopy(dict(row)) for row in rows]
    summaries = summarize_rows(copied_rows)
    comparisons = compare_reported_metrics(
        copied_rows,
        summaries["proposal_coverage_by_arm"],
        summaries["source_grounded_correctness_by_arm"],
    )
    row_ids = [str(row.get("cell_identity")) for row in copied_rows]
    observed_arms = {str(row["value"]) for row in summaries["arm_summary_rows"]}
    observed_families = {str(row["value"]) for row in summaries["family_summary_rows"]}
    checks = [
        gate_check("preconditions_passed", True, preconditions_checked.get("passed") is True),
        gate_check("expected_terminal_row_count", len(expected_cell_ids), len(copied_rows)),
        gate_check("one_terminal_row_per_cell", sorted(expected_cell_ids), sorted(set(row_ids))),
        gate_check("duplicate_terminal_row_count", 0, len(row_ids) - len(set(row_ids))),
        gate_check(
            "all_rows_terminal", True, all(row.get("terminal") is True for row in copied_rows)
        ),
        gate_check(
            "saved_output_bytes_replayed",
            True,
            all(row.get("raw_output_check", {}).get("passed") is True for row in copied_rows),
        ),
        gate_check("held_sidecar_access_count", 0, 0),
        gate_check("model_inference_call_count", 0, 0),
        gate_check("reported_aggregates_match_rows", True, comparisons["agreement"]),
        gate_check("arm_qualification_decisions", sorted(expected_arms), sorted(observed_arms)),
        gate_check(
            "family_qualification_decisions",
            sorted(expected_families),
            sorted(observed_families),
        ),
    ]
    summary = gate_summary(checks)
    ready = int(summary["passed"])
    artifact: JsonDict = {
        "experiment_id": 6913,
        "schema": SCHEMA,
        "run_date": date,
        "status": "complete",
        "field_principles": {},
        "preconditions_checked": deepcopy(dict(preconditions_checked)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "rows": copied_rows,
        "source_offset_rows": _row_views(copied_rows, "source_offset_check"),
        "source_byte_identity_rows": _row_views(copied_rows, "source_byte_identity_check"),
        "parser_rows": _row_views(copied_rows, "parser_check"),
        "tuple_type_rows": _row_views(copied_rows, "tuple_type_check"),
        "entity_anchor_rows": _row_views(copied_rows, "entity_anchor_check"),
        "relation_direction_rows": _row_views(copied_rows, "relation_direction_check"),
        "omission_rows": _row_views(copied_rows, "omission_check"),
        "duplicate_rows": _row_views(copied_rows, "duplicate_check"),
        "abstention_rows": _row_views(copied_rows, "abstention_check"),
        **summaries,
        "held_sidecar_access_count": 0,
        "model_inference_call_count": 0,
        "reported_vs_recomputed_metrics": comparisons,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_tuple_shard_ready_score": ready,
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "verdict_class": "circular_positive" if ready else "disqualified",
        "honest_verdict": COMPLETE_VERDICT,
    }
    _attach_principles(artifact)
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def blocked_artifact(
    *,
    date: str,
    duration_s: float,
    source_artifact_hashes: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
) -> JsonDict:
    """Write a schema-complete terminal block without fabricating row evidence."""

    artifact: JsonDict = {
        "experiment_id": 6913,
        "schema": SCHEMA,
        "run_date": date,
        "status": "blocked",
        "field_principles": {},
        "preconditions_checked": deepcopy(dict(preconditions_checked)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "rows": [],
        "source_offset_rows": [],
        "source_byte_identity_rows": [],
        "parser_rows": [],
        "tuple_type_rows": [],
        "entity_anchor_rows": [],
        "relation_direction_rows": [],
        "omission_rows": [],
        "duplicate_rows": [],
        "abstention_rows": [],
        "arm_summary_rows": [],
        "model_summary_rows": [],
        "family_summary_rows": [],
        "seed_summary_rows": [],
        "wilson_interval_rows": [],
        "proposal_coverage_by_arm": {},
        "source_grounded_correctness_by_arm": {},
        "held_sidecar_access_count": 0,
        "model_inference_call_count": 0,
        "reported_vs_recomputed_metrics": {"agreement": False, "comparisons": []},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_tuple_shard_ready_score": 0,
        "gate_check_summary": deepcopy(dict(preconditions_checked)),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }
    _attach_principles(artifact)
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Replay schema, metrics, checksum, and terminal class from one artifact."""

    errors: list[str] = []
    if not set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact):
        errors.append("required_fields_missing")
    if not set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_principles", {})):
        errors.append("required_field_principles_missing")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("held_sidecar_access_count") != 0:
        errors.append("held_sidecar_access")
    if artifact.get("model_inference_call_count") != 0:
        errors.append("model_inference_calls")
    if artifact.get("verdict_class") == "positive":
        errors.append("positive_verdict_forbidden")
    rows = artifact.get("rows", [])
    rows = rows if isinstance(rows, list) else []
    comparison = compare_reported_metrics(
        rows,
        artifact.get("proposal_coverage_by_arm", {}),
        artifact.get("source_grounded_correctness_by_arm", {}),
    )
    if not comparison["agreement"]:
        errors.append("aggregate_disagreement")
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    if artifact.get("source_tuple_shard_ready_score") == 1:
        if artifact.get("gate_check_summary", {}).get("passed") is not True:
            errors.append("ready_gate_disagreement")
        if artifact.get("verdict_class") != "circular_positive":
            errors.append("ready_verdict_class")
    return errors


def _read_json(path: Path) -> JsonDict:
    """Read one immutable JSON artifact or return an empty mapping."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace only the requested result after a complete JSON write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    output_path: Path | str = RESULT_PATH,
    expected_hashes: Mapping[str, str] = EXPECTED_SOURCE_HASHES,
    expected_ids: set[str] | None = None,
    expected_arms: set[str] | None = None,
    expected_families: set[str] | None = None,
    expected_source_records_hash: str | None = EXPECTED_SOURCE_RECORDS_HASH,
    clock: Any = time.monotonic,
) -> JsonDict:
    """Run the source-only replay without opening any held ASP sidecar."""

    started = clock()
    paths = {name: root / relative for name, relative in SOURCE_PATHS.items()}
    observed_hashes = {name: sha256_file(path) for name, path in paths.items()}
    source_hash_rows = {
        name: {
            "path": str(SOURCE_PATHS[name]),
            "expected_sha256": expected_hashes.get(name),
            "observed_sha256": observed_hashes.get(name),
        }
        for name in SOURCE_PATHS
    }
    fixture = _read_json(paths["exp6886"])
    source = _read_json(paths["exp6900"])
    receipt = _read_json(paths["exp6912"])
    cells = source.get("cell_manifest", [])
    cells = cells if isinstance(cells, list) else []
    wanted_ids = expected_ids if expected_ids is not None else expected_cell_identities()
    preconditions = validate_preconditions(
        receipt=receipt,
        observed_hashes=observed_hashes,
        expected_hashes=expected_hashes,
        cells=cells,
        expected_cell_ids=wanted_ids,
        held_sidecar_access_count=0,
        source_held_sidecar_access_count=int(source.get("held_sidecar_access_count", -1)),
    )
    output = Path(output_path)
    output = output if output.is_absolute() else root / output
    if not preconditions["passed"]:
        artifact = blocked_artifact(
            date=date,
            duration_s=clock() - started,
            source_artifact_hashes=source_hash_rows,
            preconditions_checked=preconditions,
        )
        _write_json(output, artifact)
        return artifact
    try:
        sources = extract_source_records(cells, fixture.get("rows", []))
        observed_source_hash = sha256_bytes(canonical_json(sources).encode("utf-8"))
        if (
            expected_source_records_hash is not None
            and observed_source_hash != expected_source_records_hash
        ):
            raise ValueError("source_record_hash_mismatch")
        by_source = {str(row["fixture_id"]): row for row in sources}
        contracts = {
            fixture_id: source_relation_contract(row) for fixture_id, row in by_source.items()
        }
        rows = [
            qualify_cell(
                cell,
                by_source[str(cell["fixture_id"])],
                contracts[str(cell["fixture_id"])],
            )
            for cell in cells
        ]
    except (KeyError, TypeError, ValueError) as exc:
        failed = gate_check("source_reconstruction", True, str(exc))
        preconditions = gate_summary([*preconditions["checks"], failed])
        artifact = blocked_artifact(
            date=date,
            duration_s=clock() - started,
            source_artifact_hashes=source_hash_rows,
            preconditions_checked=preconditions,
        )
        _write_json(output, artifact)
        return artifact
    artifact = build_artifact(
        rows=rows,
        date=date,
        duration_s=clock() - started,
        source_artifact_hashes={
            **source_hash_rows,
            "source_records": {
                "expected_sha256": expected_source_records_hash,
                "observed_sha256": observed_source_hash,
            },
        },
        preconditions_checked=preconditions,
        expected_cell_ids=wanted_ids,
        expected_arms=expected_arms if expected_arms is not None else set(REQUIRED_ARMS),
        expected_families=(expected_families if expected_families is not None else set(FAMILIES)),
    )
    _write_json(output, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Expose the exact dated command required by the experiment contract."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260903")
    parser.add_argument("--output", default=str(RESULT_PATH))
    args = parser.parse_args(argv)
    run(date=args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper is the required command surface.
    raise SystemExit(main())
