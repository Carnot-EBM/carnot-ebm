"""Freeze lossless evidence-link inputs over historically exposed source groups.

This module selects a smaller fixed roster from the authenticated V662 groups.
It does not call a model. Later experiments can add sentence-to-source evidence,
but they cannot change these texts, roles, labels, or selection rules.

Spec refs: REQ-VERIFY-7588 and SCENARIO-VERIFY-7588-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from carnot import experiment_7575_v662_cached_learning_protocol as v662
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260924"
MILESTONE = "2026.09.663"
EXPERIMENT_ID = "exp7588-v663-evidence-protocol"
SCHEMA = "carnot.exp7588.v663.evidence_protocol.v1"
RESULT_PATH = Path("results/experiment_7588_v663_evidence_protocol.json")
RAW_DIR = Path("results/raw/experiment_7588_v663_evidence_protocol")
MODULE_PATH = Path("python/carnot/experiment_7588_v663_evidence_protocol.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7588_v663_evidence_protocol.py")
TEST_PATH = Path("tests/python/test_experiment_7588_v663_evidence_protocol.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
SOURCE_PATH = Path("results/experiment_7575_v662_cached_learning_protocol.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SELECTION_SALT = "v663-evidence-20260924"
RANDOM_SEED = 7_588_001
HISTORICAL_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
SOURCE_ROLE_COUNTS = {"fit": 160, "tune": 40, "policy": 40, "online": 160, "test": 80}
ROLE_COUNTS = {"fit": 80, "tune": 20, "policy": 20, "online": 80, "evaluation": 40}
ROLE_SOURCE = {
    "fit": "fit",
    "tune": "tune",
    "policy": "policy",
    "online": "online",
    "evaluation": "test",
}
PILOT_COUNT = 8
SCORED_GROUPS = sum(ROLE_COUNTS.values())
PROTOCOL_ARMS = ("evidence_link", "evidence_erasure", "within_role_derangement")
EVIDENCE_FEATURE_NAMES = (
    "supported_sentence_fraction",
    "contradicted_fraction",
    "unknown_fraction",
    "valid_link_fraction",
    "exact_named_entity_overlap",
    "numeric_value_agreement",
    "negation_mismatch_indicator",
    "extraction_censor_indicator",
)
ZERO_INVOCATION_COUNTS = {
    name: {state: 0 for state in ("attempted", "completed", "failed", "cancelled")}
    for name in ("model_loads", "forward_calls", "generation_calls", "tokens")
}
AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def _text_sha256(value: str) -> str:
    """Hash exact UTF-8 bytes because every pointer depends on unchanged text."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ambiguous_block(value: str) -> bool:
    """Keep code and tables whole because punctuation is not a sentence boundary there."""

    lines = value.splitlines()
    return (
        "```" in value
        or any(line.count("|") >= 2 for line in lines)
        or any(line.startswith(("    ", "\t")) for line in lines if line.strip())
    )


def segment_lossless(text: str, prefix: str) -> list[JsonDict]:
    """Number whole sentences or ambiguous blocks without dropping any UTF-8 byte."""

    if not isinstance(text, str) or not text or prefix not in {"R", "S"}:
        raise ValueError("complete_text_absent")
    spans: list[tuple[int, int, str]] = []
    for block_match in re.finditer(r".*?(?:\n[ \t]*\n|\Z)", text, flags=re.DOTALL):
        start, end = block_match.span()
        if start == end:
            continue
        block = text[start:end]
        if _ambiguous_block(block):
            spans.append((start, end, "ambiguous_whole_block"))
            continue
        cursor = start
        for boundary in re.finditer(r"(?<=[.!?])\s+", block):
            stop = start + boundary.end()
            if stop > cursor:
                spans.append((cursor, stop, "whole_sentence"))
                cursor = stop
        if cursor < end:
            spans.append((cursor, end, "whole_sentence"))
    rows: list[JsonDict] = []
    for index, (start, end, kind) in enumerate(spans, 1):
        value = text[start:end]
        rows.append(
            {
                "sentence_id": f"{prefix}{index:03d}",
                "byte_start": len(text[:start].encode("utf-8")),
                "byte_end": len(text[:end].encode("utf-8")),
                "text": value,
                "text_sha256": _text_sha256(value),
                "boundary_kind": kind,
            }
        )
    roundtrip_segments(text, rows)
    return rows


def roundtrip_segments(text: str, segments: Sequence[Mapping[str, Any]]) -> str:
    """Rebuild exact bytes and reject gaps, overlap, altered text, or bad offsets."""

    original = text.encode("utf-8")
    cursor = 0
    rebuilt = bytearray()
    for row in segments:
        start = row.get("byte_start")
        end = row.get("byte_end")
        value = row.get("text")
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(value, str):
            raise ValueError("segment_roundtrip_invalid")
        encoded = value.encode("utf-8")
        if start != cursor or end != start + len(encoded) or original[start:end] != encoded:
            raise ValueError("segment_roundtrip_invalid")
        rebuilt.extend(encoded)
        cursor = end
    if cursor != len(original) or bytes(rebuilt) != original:
        raise ValueError("segment_roundtrip_invalid")
    return rebuilt.decode("utf-8")


def build_input_contract(row: Mapping[str, Any]) -> JsonDict:
    """Project one historical group into a label-free, lossless transport input."""

    component = str(row.get("source_id") or row.get("component_hash") or "")
    response = row.get("response")
    source = row.get("context")
    if (
        not component
        or not isinstance(response, str)
        or not response
        or not isinstance(source, str)
        or not source
    ):
        raise ValueError("complete_text_absent")
    if row.get("response_sha256") not in (None, canonical_hash(response)):
        raise ValueError("response_hash_invalid")
    if row.get("context_sha256") not in (None, canonical_hash(source)):
        raise ValueError("source_hash_invalid")
    response_hash = _text_sha256(response)
    source_hash = _text_sha256(source)
    role = str(row.get("role") or "")
    return {
        "component_hash": component,
        "role": role,
        "source_role": str(row.get("source_role") or role),
        "official_split": str(row.get("official_split") or ""),
        "complete_response": response,
        "complete_source": source,
        "response_sha256": response_hash,
        "source_sha256": source_hash,
        "response_sentences": segment_lossless(response, "R"),
        "source_sentences": segment_lossless(source, "S"),
        "designated_response_sha256": response_hash,
        "designation_rank": _text_sha256(f"{SELECTION_SALT}:{component}:{response_hash}"),
        "labels_accessible": False,
        "generated_replacement_text_allowed": False,
        "source_truncation_allowed": False,
    }


_OUTPUT_FIELDS = {
    "response_sentence_id",
    "source_sentence_ids",
    "relation",
    "entity_type",
    "abstention_reason",
}
_RELATIONS = {"supports", "contradicts", "unknown"}
_ENTITY_TYPES = {"person", "organization", "location", "date", "numeric", "code", "other", "none"}


def normalize_evidence_output(
    contract: Mapping[str, Any], proposals: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Validate pointer-only model output and make every omitted sentence unknown."""

    if len(proposals) > 6:
        raise ValueError("evidence_link_budget_exceeded")
    response_ids = {str(row["sentence_id"]) for row in contract.get("response_sentences", [])}
    source_ids = {str(row["sentence_id"]) for row in contract.get("source_sentences", [])}
    normalized: dict[str, JsonDict] = {}
    for proposal in proposals:
        if set(proposal) != _OUTPUT_FIELDS:
            raise ValueError("evidence_output_fields_invalid")
        response_id = str(proposal.get("response_sentence_id") or "")
        if response_id not in response_ids or response_id in normalized:
            raise ValueError("response_sentence_id_invalid")
        links = proposal.get("source_sentence_ids")
        if (
            not isinstance(links, list)
            or len(links) != len(set(links))
            or any(str(source_id) not in source_ids for source_id in links)
        ):
            raise ValueError("source_sentence_id_invalid")
        relation = str(proposal.get("relation") or "")
        if relation not in _RELATIONS:
            raise ValueError("evidence_relation_invalid")
        entity_type = str(proposal.get("entity_type") or "")
        abstention = proposal.get("abstention_reason")
        if entity_type not in _ENTITY_TYPES or not isinstance(abstention, str):
            raise ValueError("evidence_output_value_invalid")
        if relation == "unknown":
            if links or not abstention:
                raise ValueError("unknown_relation_contract_invalid")
        elif not links or abstention:
            raise ValueError("linked_relation_contract_invalid")
        normalized[response_id] = {
            "response_sentence_id": response_id,
            "source_sentence_ids": [str(value) for value in links],
            "relation": relation,
            "entity_type": entity_type,
            "abstention_reason": abstention,
            "proposed_link": True,
        }
    for response_id in response_ids - normalized.keys():
        normalized[response_id] = {
            "response_sentence_id": response_id,
            "source_sentence_ids": [],
            "relation": "unknown",
            "entity_type": "none",
            "abstention_reason": "unlinked_by_extractor",
            "proposed_link": False,
        }
    return [normalized[key] for key in sorted(normalized)]


def _sentence_text(contract: Mapping[str, Any], side: str) -> dict[str, str]:
    return {
        str(row["sentence_id"]): str(row["text"]) for row in contract.get(f"{side}_sentences", [])
    }


def _named_entities(value: str) -> set[str]:
    """Return exact multi-token names; overlap is lexical evidence, not truth."""

    return set(re.findall(r"\b[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+)+\b", value))


def _numbers(value: str) -> set[str]:
    return set(re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", value))


def _agreement(left: set[str], right: set[str]) -> float | None:
    if not left or not right:
        return None
    return len(left & right) / len(left | right)


def reduce_evidence_features(
    contract: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    *,
    raw_probability: float,
) -> JsonDict:
    """Reduce validated pointers to eight bounded values plus a separate offset."""

    if not math.isfinite(raw_probability) or not 0.0 <= raw_probability <= 1.0:
        raise ValueError("raw_probability_invalid")
    response_text = _sentence_text(contract, "response")
    source_text = _sentence_text(contract, "source")
    if len(evidence) != len(response_text):
        raise ValueError("evidence_sentence_roster_invalid")
    relations = Counter(str(row.get("relation")) for row in evidence)
    denominator = len(response_text)
    linked = [row for row in evidence if row.get("relation") in {"supports", "contradicts"}]
    response_link_text = " ".join(response_text[str(row["response_sentence_id"])] for row in linked)
    source_link_text = " ".join(
        source_text[str(source_id)]
        for row in linked
        for source_id in row.get("source_sentence_ids", [])
    )
    entity_overlap = _agreement(
        _named_entities(response_link_text), _named_entities(source_link_text)
    )
    numeric_agreement = _agreement(_numbers(response_link_text), _numbers(source_link_text))
    response_negated = bool(re.search(r"\b(?:no|not|never|without)\b", response_link_text, re.I))
    source_negated = bool(re.search(r"\b(?:no|not|never|without)\b", source_link_text, re.I))
    values: dict[str, float | None] = {
        "supported_sentence_fraction": relations["supports"] / denominator,
        "contradicted_fraction": relations["contradicts"] / denominator,
        "unknown_fraction": relations["unknown"] / denominator,
        "valid_link_fraction": len(linked) / denominator,
        "exact_named_entity_overlap": entity_overlap,
        "numeric_value_agreement": numeric_agreement,
        "negation_mismatch_indicator": float(response_negated != source_negated),
        "extraction_censor_indicator": float(
            sum(bool(row.get("proposed_link")) for row in evidence) >= 6
        ),
    }
    if tuple(values) != EVIDENCE_FEATURE_NAMES:  # pragma: no cover - local literal invariant.
        raise ValueError("evidence_feature_schema_invalid")
    if any(  # pragma: no cover - ratios and Boolean indicators are bounded by construction.
        value is not None and not 0.0 <= value <= 1.0 for value in values.values()
    ):
        raise ValueError("evidence_feature_out_of_bounds")
    return {
        "features": values,
        "missing_values": {name: value is None for name, value in values.items()},
        "raw_probability_offset": float(raw_probability),
        "imputation_status": "unimputed_fit_only",
        "imputation_rule": "median_of_observed_fit_role_only_then_freeze",
        "lexical_agreement_is_semantic_proof": False,
    }


def erase_evidence(contract: Mapping[str, Any]) -> list[JsonDict]:
    """Create the deterministic evidence-erasure control over unchanged text."""

    return [
        {
            "response_sentence_id": str(row["sentence_id"]),
            "source_sentence_ids": [],
            "relation": "unknown",
            "entity_type": "none",
            "abstention_reason": "deterministic_evidence_erasure",
            "proposed_link": False,
        }
        for row in contract.get("response_sentences", [])
    ]


def _rank(component: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{component}".encode()).hexdigest()


_PROMPT_PREFIX = (
    "System: Judge only whether the complete answer is grounded in the complete "
    "tool output. Do not execute code.\n\nTool output:\n"
)
_QUESTION_MARKER = "\n\nUser request:\n"
_ANSWER_MARKER = "\n\nAnswer:\n"
_OPTION_MARKER = "\n\n A. The answer is supported by the supplied tool output.\n"
_PROMPT_SUFFIX = (
    " B. The answer contains unsupported content.\nReturn the better option.\nAssistant:"
)


def restore_original_text(row: Mapping[str, Any]) -> JsonDict:
    """Recover exact public source and answer bytes from the frozen scorer prompt."""

    restored = deepcopy(dict(row))
    if restored.get("response"):
        return restored
    prompt = restored.get("context")
    if not isinstance(prompt, str) or not prompt.startswith(_PROMPT_PREFIX):
        raise ValueError("original_text_contract_invalid")
    source_and_rest = prompt[len(_PROMPT_PREFIX) :]
    source, question_marker, question_and_answer = source_and_rest.partition(_QUESTION_MARKER)
    question, answer_marker, answer_and_options = question_and_answer.partition(_ANSWER_MARKER)
    answer, option_marker, suffix = answer_and_options.rpartition(_OPTION_MARKER)
    if (
        not question_marker
        or not answer_marker
        or not option_marker
        or suffix != _PROMPT_SUFFIX
        or not source
        or not question
        or not answer
    ):
        raise ValueError("original_text_contract_invalid")
    restored["scorer_prompt"] = prompt
    restored["scorer_prompt_sha256"] = canonical_hash(prompt)
    restored["context"] = source
    restored["response"] = answer
    restored["question"] = question
    restored["context_sha256"] = canonical_hash(source)
    restored["response_sha256"] = canonical_hash(answer)
    restored["original_text_extracted_from_authenticated_prompt"] = True
    return restored


def select_roles(roles: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    """Select fixed role-local groups without reading labels or prediction scores."""

    seen: set[str] = set()
    prepared: dict[str, list[JsonDict]] = {}
    for source_role, expected in SOURCE_ROLE_COUNTS.items():
        rows = list(roles.get(source_role, []))
        if len(rows) != expected:
            raise ValueError(f"source_role_count_invalid:{source_role}")
        prepared[source_role] = []
        for row in rows:
            component = str(row.get("source_id") or "")
            if not component or component in seen:
                raise ValueError("component_duplicate")
            if row.get("role") != source_role:
                raise ValueError("source_role_identity_invalid")
            restored = restore_original_text(row)
            if (
                not restored.get("response")
                or not restored.get("context")
                or restored.get("label") not in {0, 1}
            ):
                raise ValueError("source_group_incomplete")
            seen.add(component)
            prepared[source_role].append(restored)
    scored: dict[str, list[JsonDict]] = {}
    for role, count in ROLE_COUNTS.items():
        source_role = ROLE_SOURCE[role]
        ranked = sorted(prepared[source_role], key=lambda row: _rank(str(row["source_id"])))
        scored[role] = [
            {**deepcopy(dict(row)), "source_role": source_role, "role": role}
            for row in ranked[:count]
        ]
    ranked_fit = sorted(prepared["fit"], key=lambda row: _rank(str(row["source_id"])))
    pilot = [
        {**deepcopy(dict(row)), "source_role": "fit", "role": "pilot"}
        for row in ranked_fit[ROLE_COUNTS["fit"] : ROLE_COUNTS["fit"] + PILOT_COUNT]
    ]
    selected_ids = {role: [str(row["source_id"]) for row in rows] for role, rows in scored.items()}
    selected_ids["pilot"] = [str(row["source_id"]) for row in pilot]
    return {
        "scored": scored,
        "pilot": pilot,
        "selected_ids": selected_ids,
        "salt": SELECTION_SALT,
        "selection_used_labels": False,
        "selection_used_scores": False,
    }


def within_role_derangement(
    scored: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, str]:
    """Rotate salted source identities inside each role with no fixed points."""

    mapping: dict[str, str] = {}
    for role in ROLE_COUNTS:
        rows = sorted(scored.get(role, []), key=lambda row: _rank(str(row["source_id"])))
        if len(rows) < 2:
            raise ValueError(f"derangement_role_too_small:{role}")
        ids = [str(row["source_id"]) for row in rows]
        mapping.update(dict(zip(ids, ids[1:] + ids[:1], strict=True)))
    return mapping


def build_isolated_readers(selected: Mapping[str, Any]) -> tuple[list[JsonDict], list[JsonDict]]:
    """Build label-free extraction inputs and a separate evaluator-only label reader."""

    rows = [row for role in ROLE_COUNTS for row in selected.get("scored", {}).get(role, [])] + list(
        selected.get("pilot", [])
    )
    inputs: list[JsonDict] = []
    labels: list[JsonDict] = []
    seen: set[str] = set()
    for row in rows:
        contract = build_input_contract(row)
        component = str(contract["component_hash"])
        if component in seen:
            raise ValueError("component_duplicate")
        seen.add(component)
        contract["read_only"] = row.get("role") in {"evaluation", "pilot"}
        contract["feedback_lag"] = 8 if row.get("role") == "online" else None
        inputs.append(contract)
        labels.append(
            {
                "component_hash": component,
                "role": str(row["role"]),
                "label": int(row["label"]),
            }
        )
    return inputs, labels


def build_protocol(selected: Mapping[str, Any]) -> JsonDict:
    """Freeze role, schema, feature, control, cost, and reader rules."""

    scored = selected["scored"]
    all_rows = [row for role in ROLE_COUNTS for row in scored[role]] + list(selected["pilot"])
    ids = [str(row["source_id"]) for row in all_rows]
    overlap_count = len(ids) - len(set(ids))
    roster = [
        {
            "component_hash": str(row["source_id"]),
            "role": str(row["role"]),
            "source_role": str(row["source_role"]),
            "official_split": str(row["official_split"]),
            "response_sha256": _text_sha256(str(row["response"])),
            "source_sha256": _text_sha256(str(row["context"])),
            "historically_exposed": True,
        }
        for row in all_rows
    ]
    return {
        "schema": "carnot.exp7588.v663.evidence_protocol_manifest.v1",
        "dataset_id": v662.PINNED_DATASET_ID,
        "dataset_revision": v662.PINNED_DATASET_REVISION,
        "selection_salt": SELECTION_SALT,
        "selection_rule": "sha256(salt:component_hash)_ascending_within_source_role",
        "selection_used_labels": False,
        "selection_used_scores": False,
        "role_counts": {**ROLE_COUNTS, "pilot": PILOT_COUNT},
        "source_role_counts": SOURCE_ROLE_COUNTS,
        "scored_group_count": SCORED_GROUPS,
        "pilot_group_count": PILOT_COUNT,
        "roster": roster,
        "roster_sha256": canonical_hash(roster),
        "overlap_audit": {
            "unique_components": len(set(ids)),
            "total_components": len(ids),
            "overlap_count": overlap_count,
            "pilot_outside_scored_roles": overlap_count == 0,
        },
        "input_contract": {
            "complete_response_required": True,
            "complete_source_required": True,
            "lossless_utf8_byte_offsets": True,
            "whole_ambiguous_blocks": True,
            "source_truncation_allowed": False,
            "generated_replacement_text_allowed": False,
        },
        "output_contract": {
            "fields": sorted(_OUTPUT_FIELDS),
            "relations": sorted(_RELATIONS),
            "maximum_proposed_links": 6,
            "unlinked_response_sentence_relation": "unknown",
        },
        "evidence_feature_names": list(EVIDENCE_FEATURE_NAMES),
        "feature_imputation": {
            "missing_entity_and_numeric_values": "explicit_null",
            "learn_on_role": "fit",
            "rule": "median_of_observed_fit_role_only_then_freeze",
        },
        "raw_probability_is_separate_offset": True,
        "lexical_agreement_is_semantic_proof": False,
        "controls": ["evidence_erasure", "within_role_derangement"],
        "within_role_derangement": within_role_derangement(scored),
        "controls_reuse_complete_text": True,
        "reader_isolation": {
            "extraction_fields_include_label": False,
            "model_transport_fields_include_label": False,
            "label_reader_separate": True,
            "evaluation_role_read_only": True,
        },
        "response_designation": "lowest_hash_among_complete_group_response_variants",
        "costs": {"accept": "5*y", "reject": "1-y", "escalate": 0.2},
        "non_escalation_floor": 0.10,
        "delayed_feedback_lag": 8,
        "evaluation_retention_role": "evaluation",
        "fresh_confirmatory_claim_allowed": False,
        "claim_scope": "descriptive_reuse",
    }


def _path_label(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:  # pragma: no cover - production files stay below the declared root.
        return str(path.resolve())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], root: Path) -> JsonDict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return {
        "path": _path_label(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
    }


def build_protocol_rows(selected: Mapping[str, Any], protocol: Mapping[str, Any]) -> list[JsonDict]:
    """Emit one descriptive byte-custody row for every unit and protocol arm."""

    scored = selected["scored"]
    rows = [row for role in ROLE_COUNTS for row in scored[role]] + list(selected["pilot"])
    by_id = {str(row["source_id"]): row for row in rows}
    derangement = dict(protocol["within_role_derangement"])
    output: list[JsonDict] = []
    for row in rows:
        unit_id = str(row["source_id"])
        for arm in PROTOCOL_ARMS:
            donor_id = (
                derangement.get(unit_id, unit_id) if arm == "within_role_derangement" else unit_id
            )
            donor = by_id[donor_id]
            response_bytes = len(str(row["response"]).encode("utf-8"))
            source_bytes = len(str(donor["context"]).encode("utf-8"))
            output.append(
                {
                    "unit_id": unit_id,
                    "role": str(row["role"]),
                    "arm": arm,
                    "absolute_metrics": {
                        "response_bytes": response_bytes,
                        "source_bytes": source_bytes,
                    },
                    "raw_numerator": response_bytes + source_bytes,
                    "raw_denominator": 1,
                    "seed": RANDOM_SEED,
                    "metric_direction": "descriptive_no_benefit_direction",
                    "missingness": {"response": False, "source": False, "label": False},
                    "censored": False,
                    "sign": "not_applicable_protocol_only",
                    "response_sha256": _text_sha256(str(row["response"])),
                    "source_sha256": _text_sha256(str(donor["context"])),
                    "donor_unit_id": donor_id,
                    "provenance": "authenticated_v662_complete_text",
                }
            )
    return output


def reduce_protocol_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute row uniqueness, arm completeness, counts, and byte totals."""

    seen: set[tuple[str, str]] = set()
    arms: dict[str, set[str]] = defaultdict(set)
    role_units: dict[str, set[str]] = defaultdict(set)
    totals: dict[str, int] = defaultdict(int)
    for row in rows:
        key = (str(row.get("unit_id")), str(row.get("arm")))
        if key in seen:
            raise ValueError("protocol_row_duplicate")
        seen.add(key)
        if row.get("raw_denominator") != 1 or not isinstance(row.get("raw_numerator"), int):
            raise ValueError("protocol_row_arithmetic_invalid")
        absolute = row.get("absolute_metrics")
        if not isinstance(absolute, Mapping) or row["raw_numerator"] != sum(
            int(absolute.get(name, -1)) for name in ("response_bytes", "source_bytes")
        ):
            raise ValueError("protocol_row_arithmetic_invalid")
        unit_id, arm = key
        arms[unit_id].add(arm)
        role_units[str(row.get("role"))].add(unit_id)
        totals[arm] += int(row["raw_numerator"])
    expected_arms = set(PROTOCOL_ARMS)
    return {
        "row_count": len(rows),
        "unique_units": len(arms),
        "role_unit_counts": {role: len(values) for role, values in sorted(role_units.items())},
        "arm_byte_numerators": dict(sorted(totals.items())),
        "all_arms_complete": bool(arms)
        and all(values == expected_arms for values in arms.values()),
        "duplicate_rows": 0,
        "censored_rows": sum(bool(row.get("censored")) for row in rows),
    }


def freeze_protocol_files(raw_dir: Path, selected: Mapping[str, Any], *, root: Path) -> JsonDict:
    """Write separate readers and the protocol only after every text round-trips."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    inputs, labels = build_isolated_readers(selected)
    roundtrip_failures: list[str] = []
    for row in inputs:
        try:
            roundtrip_segments(row["complete_response"], row["response_sentences"])
            roundtrip_segments(row["complete_source"], row["source_sentences"])
        except ValueError:  # pragma: no cover - contracts were round-tripped when built.
            roundtrip_failures.append(str(row["component_hash"]))
    offsets = [
        {
            "component_hash": str(row["source_id"]),
            "role": str(row["role"]),
            "raw_probability_offset": float(row["probability"]),
        }
        for role in ROLE_COUNTS
        for row in selected["scored"][role]
    ] + [
        {
            "component_hash": str(row["source_id"]),
            "role": "pilot",
            "raw_probability_offset": float(row["probability"]),
        }
        for row in selected["pilot"]
    ]
    input_receipt = _write_jsonl(raw_dir / "extraction_inputs.jsonl", inputs, root)
    label_receipt = _write_jsonl(raw_dir / "human_labels.jsonl", labels, root)
    offset_receipt = _write_jsonl(raw_dir / "raw_probability_offsets.jsonl", offsets, root)
    protocol = build_protocol(selected)
    protocol["reader_sidecars"] = {
        "extraction_inputs": input_receipt,
        "human_labels": label_receipt,
        "raw_probability_offsets": offset_receipt,
    }
    protocol["labels_in_extraction_sidecar"] = False
    protocol["roundtrip_failure_count"] = len(roundtrip_failures)
    protocol_path = raw_dir / "protocol.json"
    atomic_json(protocol_path, protocol)
    rows = build_protocol_rows(selected, protocol)
    return {
        "protocol": protocol,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_path_label": _path_label(protocol_path, root),
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_bytes": protocol_path.stat().st_size,
        "inputs_path": str((raw_dir / "extraction_inputs.jsonl").resolve()),
        "labels_path": str((raw_dir / "human_labels.jsonl").resolve()),
        "raw_sidecars": protocol["reader_sidecars"],
        "rows": rows,
        "row_reduction": reduce_protocol_rows(rows),
        "roundtrip_failures": roundtrip_failures,
    }


def precondition(
    check: str,
    upstream: str,
    path: Path,
    field: str,
    expected: Any,
    observed: Any,
    *,
    op: str = "eq",
) -> JsonDict:
    """Record one exact external operand so a block is independently actionable."""

    if op == "eq":
        passed = observed == expected
    elif op == "in":
        passed = observed in expected
    else:
        raise ValueError("precondition_operator_invalid")
    return {
        "check": check,
        "upstream": upstream,
        "path": str(path.resolve()),
        "field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def _load_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_preconditions(  # pragma: no cover - production source boundary.
    root: Path,
) -> tuple[list[JsonDict], list[JsonDict], dict[str, list[JsonDict]]]:
    """Authenticate V662 terminal bytes, all role sidecars, labels, and ownership."""

    root = root.resolve()
    checks: list[JsonDict] = []
    hashes: list[JsonDict] = []
    source_path = root / SOURCE_PATH
    spec_path = root / SPEC_PATH
    exclusion_path = root / EXCLUSION_PATH
    for name, path in (
        ("v662_terminal", source_path),
        ("capability_spec", spec_path),
        ("exclusion_manifest", exclusion_path),
    ):
        exists = path.is_file() and path.stat().st_size > 0
        checks.append(
            precondition(
                "source_exists",
                name,
                path,
                "path",
                "readable_nonempty_file",
                "readable_nonempty_file" if exists else "missing",
            )
        )
        if exists:
            hashes.append(
                {
                    "producer": name,
                    "path": _path_label(path, root),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "source_class": "authenticated_producer"
                    if name == "v662_terminal"
                    else "conductor_pre_gate",
                }
            )
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    checks.append(
        precondition(
            "driving_requirement",
            "verification_spec",
            spec_path,
            "REQ-*",
            "REQ-VERIFY-7588",
            "REQ-VERIFY-7588" if "REQ-VERIFY-7588" in spec_text else "missing",
        )
    )
    exclusion_text = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    quarantined = (
        "experiment_id: 7588" in exclusion_text or "exp7588-evidence-protocol" in exclusion_text
    )
    checks.append(
        precondition(
            "resource_ownership",
            "ops_exclusion_manifest",
            exclusion_path,
            "current_task_quarantined",
            False,
            quarantined,
        )
    )
    source = _load_object(source_path)
    for check, field, expected in (
        ("upstream_verdict", "verdict_class", ("null", "positive")),
        ("upstream_adversarial", "flagged_adversarial", False),
        ("upstream_cached_roles", "cached_roles_ready_score", 1),
        ("upstream_online_protocol", "online_protocol_ready_score", 1),
        ("historical_exposure", "fresh_confirmatory_claim_allowed", False),
    ):
        checks.append(
            precondition(
                check,
                "exp7575",
                source_path,
                field,
                expected,
                source.get(field),
                op="in" if isinstance(expected, tuple) else "eq",
            )
        )
    frozen = (
        source.get("frozen_protocol") if isinstance(source.get("frozen_protocol"), Mapping) else {}
    )
    exposure = (
        source.get("exposure_manifest")
        if isinstance(source.get("exposure_manifest"), Mapping)
        else {}
    )
    checks.extend(
        [
            precondition(
                "source_role_manifest",
                "exp7575",
                source_path,
                "frozen_protocol.role_counts",
                SOURCE_ROLE_COUNTS,
                frozen.get("role_counts"),
            ),
            precondition(
                "dataset_revision",
                "exp7575",
                source_path,
                "exposure_manifest.dataset_revision",
                v662.PINNED_DATASET_REVISION,
                exposure.get("dataset_revision"),
            ),
        ]
    )
    roles: dict[str, list[JsonDict]] = {}
    if source:
        try:
            v662.validate_artifact(source, root=root, require_validation=True)
            sidecar_hashes: list[JsonDict] = []
            roles = v662.load_cached_roles(root, sidecar_hashes)
            hashes.extend(
                {
                    **row,
                    "producer": "v661_original_human_labels_and_complete_text",
                    "source_class": "authenticated_producer",
                }
                for row in sidecar_hashes
            )
            custody_observed = {name: len(rows) for name, rows in roles.items()}
            labels_valid = all(
                row.get("label") in {0, 1} for rows in roles.values() for row in rows
            )
        except (OSError, KeyError, TypeError, ValueError):
            custody_observed = {}
            labels_valid = False
        checks.append(
            precondition(
                "cached_role_custody",
                "exp7575",
                source_path,
                "loaded_role_counts",
                SOURCE_ROLE_COUNTS,
                custody_observed,
            )
        )
        checks.append(
            precondition(
                "original_human_labels",
                "exp7575",
                source_path,
                "all_labels_binary_and_joined",
                True,
                labels_valid,
            )
        )
    return checks, hashes, roles


REQUIRED_PRINCIPLE_FIELDS = (
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "gate_check_summary",
    "acceptance_gate_results",
    "rows",
    "sample_size_budget",
    "inference_substrate",
    "inference_substrate_class",
    "MODEL_SPECS",
    "invocation_counts",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "validation_receipts",
    "verifier_is_oracle",
    "field_principles",
    "evidence_protocol_ready_score",
    "protocol_path",
    "fresh_confirmatory_claim_allowed",
    "evidence_feature_names",
    "role_counts",
)


def field_principles() -> dict[str, str]:
    """Carry each requested audit principle into the terminal artifact."""

    return {
        "honest_verdict": "Use a complete_ terminal prefix; execution completion does not prove benefit.",
        "verdict_class": "Use exactly one closed class; only unfinished owned work is partial.",
        "flagged_adversarial": "Persist the terminal reader outcome; flagged evidence cannot open a gate.",
        "gate_check_summary": "A blocked result names check, upstream, path, field, operator, expected, and observed values.",
        "acceptance_gate_results": "Keep validity, readiness, benefit, retention, and freshness separate with a principle per gate.",
        "rows": "Each unit and arm retains absolute metrics, numerator, denominator, seed, direction, missingness, and provenance.",
        "sample_size_budget": "Count independent source groups; seeds, sentences, and arms do not multiply them.",
        "inference_substrate": "State current execution; inherited GPU evidence is not current inference.",
        "inference_substrate_class": "Record actual and planned classes separately; blocked_no_run means zero model work.",
        "MODEL_SPECS": "Cached-only work declares no current model and records historical identity separately.",
        "invocation_counts": "Count current loads, forwards, generations, and tokens independently from history.",
        "duration_s": "Measure monotonic current work with phase spans; do not pad or inherit time.",
        "random_seed": "Record explicit seeds for each stochastic stage; deterministic salts remain named separately.",
        "reproducibility_checksum": "Bind immutable raw evidence, configuration, and terminal reduction.",
        "source_artifact_hashes": "Distinguish authenticated producers, missing producers, and conductor pre-gate sources.",
        "validation_receipts": "Bind command, worktree, exit code, and raw log hash, including terminal readers.",
        "verifier_is_oracle": "Exact labels and constructed controls cannot support an oracle-distinct claim.",
        "field_principles": "Keep these one-line reasons inside the emitted artifact.",
        "evidence_protocol_ready_score": "One requires complete lossless inputs, all 240 roles, eight pilots, no overlap, and isolated readers.",
        "protocol_path": "The immutable role, schema, control, and cost manifest hash is required downstream.",
        "fresh_confirmatory_claim_allowed": "Previously exposed source groups always remain non-fresh.",
        "evidence_feature_names": "Exactly eight predeclared features use explicit missingness and fit-only imputation.",
        "role_counts": "Freeze 80 fit, 20 tune, 20 policy, 80 online, 40 evaluation, and eight pilots.",
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    principle: str,
    *,
    op: str = "eq",
) -> JsonDict:
    if op == "eq":
        passed = observed == expected
    else:
        raise ValueError("gate_operator_invalid")
    return {
        "check": check,
        "category": category,
        "op": op,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": principle,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [str(row["check"]) for row in failed],
        "first_failure": failed[0] if failed else None,
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    stable = deepcopy(dict(value))
    stable.pop("reproducibility_checksum", None)
    return canonical_hash(stable)


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    return bool(receipts) and all(
        row.get("passed") is True and row.get("exit_code") == 0 and row.get("timed_out") is not True
        for row in receipts
    )


def build_artifact(
    bundle: Mapping[str, Any],
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    require_validation: bool = True,
) -> JsonDict:
    """Build a complete protocol-ready null without claiming evidence benefit."""

    protocol = bundle["protocol"]
    role_counts = dict(protocol["role_counts"])
    preconditions_ok = bool(preconditions) and all(
        row.get("passed") is True for row in preconditions
    )
    validation_ok = _validation_passed(validation_receipts) if require_validation else True
    row_reduction = dict(bundle["row_reduction"])
    custody_ok = (
        role_counts == {**ROLE_COUNTS, "pilot": PILOT_COUNT}
        and row_reduction.get("unique_units") == SCORED_GROUPS + PILOT_COUNT
        and row_reduction.get("all_arms_complete") is True
    )
    lossless = not bundle.get("roundtrip_failures") and protocol.get("roundtrip_failure_count") == 0
    overlap_ok = protocol["overlap_audit"]["overlap_count"] == 0
    isolated = (
        protocol.get("labels_in_extraction_sidecar") is False
        and protocol["reader_isolation"]["extraction_fields_include_label"] is False
        and protocol["reader_isolation"]["model_transport_fields_include_label"] is False
    )
    gates = [
        _gate(
            "authenticated_preconditions",
            "validity",
            True,
            preconditions_ok,
            "Only exact V662 bytes, labels, revision, requirement, and ownership may seed the protocol.",
        ),
        _gate(
            "scoped_and_terminal_validation",
            "validity",
            True,
            validation_ok,
            "Every affected and terminal reader must pass before publication.",
        ),
        _gate(
            "exact_role_roster",
            "readiness",
            True,
            custody_ok,
            "The registered source-group counts prevent score-driven replacement.",
        ),
        _gate(
            "lossless_complete_inputs",
            "readiness",
            True,
            lossless,
            "Byte round-trip prevents omitted qualifiers or reconstructed text.",
        ),
        _gate(
            "zero_component_overlap",
            "readiness",
            True,
            overlap_ok,
            "Independent roles and pilots cannot share a source component.",
        ),
        _gate(
            "isolated_readers",
            "readiness",
            True,
            isolated,
            "Labels cannot influence extraction or model transport.",
        ),
        _gate(
            "evaluation_retention_and_lag",
            "retention",
            True,
            protocol.get("evaluation_retention_role") == "evaluation"
            and protocol.get("delayed_feedback_lag") == 8,
            "Evaluation labels stay read-only and online feedback remains delayed by eight.",
        ),
        _gate(
            "predictive_benefit",
            "benefit",
            "measured_positive",
            "not_measured_protocol_only",
            "Protocol readiness cannot establish predictive or decision benefit.",
        ),
        _gate(
            "fresh_confirmatory_claim",
            "freshness",
            False,
            protocol.get("fresh_confirmatory_claim_allowed"),
            "Runtime separation cannot erase prior exposure of these source groups.",
        ),
    ]
    ready = all(
        row["passed"]
        for row in gates
        if row["category"] in {"validity", "readiness", "retention", "freshness"}
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "title": "V663 lossless evidence-link protocol",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "complete": True,
        "honest_verdict": (
            "complete_null_evidence_protocol_ready_benefit_unmeasured"
            if ready
            else "complete_disqualified_evidence_protocol_validation_failed"
        ),
        "verdict_class": "null" if ready else "disqualified",
        "flagged_adversarial": False,
        "positive_claim": False,
        "predictive_benefit_measured": False,
        "fresh_confirmatory_claim_allowed": False,
        "claim_scope": "descriptive_reuse",
        "verifier_is_oracle": False,
        "MODEL_SPECS": [],
        "model_specs": [],
        "no_model_load": True,
        "model_invoked": False,
        "historical_model_id": HISTORICAL_MODEL_ID,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "random_seed": RANDOM_SEED,
        "random_seeds_used": {"protocol_rows": RANDOM_SEED},
        "selection_salt": SELECTION_SALT,
        "duration_s": float(duration_s),
        "protocol_path": str(bundle["protocol_path_label"]),
        "protocol_sha256": str(bundle["protocol_sha256"]),
        "protocol_bytes": int(bundle["protocol_bytes"]),
        "role_counts": role_counts,
        "evidence_feature_names": list(EVIDENCE_FEATURE_NAMES),
        "feature_imputation": deepcopy(protocol["feature_imputation"]),
        "raw_probability_is_separate_offset": True,
        "overlap_audit": deepcopy(protocol["overlap_audit"]),
        "reader_isolation": deepcopy(protocol["reader_isolation"]),
        "rows": deepcopy(list(bundle["rows"])),
        "independent_row_reduction": row_reduction,
        "sample_size_budget": {
            "intended_source_groups": sum(SOURCE_ROLE_COUNTS.values()),
            "intended_scored_independent_units": SCORED_GROUPS,
            "intended_pilot_independent_units": PILOT_COUNT,
            "observed_independent_units": SCORED_GROUPS + PILOT_COUNT,
            "excluded_independent_units": sum(SOURCE_ROLE_COUNTS.values())
            - SCORED_GROUPS
            - PILOT_COUNT,
            "censored_independent_units": 0,
            "failed_independent_units": 0,
            "seeds_or_windows_multiply_units": False,
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "evidence_protocol_ready_score": int(ready),
        "applicable_numbered_e2e": [],
        "capability_e2e": {
            "operations": ["freeze", "persist", "reload", "duplicate_rejection"],
            "passed": ready,
        },
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "raw_sidecars": deepcopy(dict(bundle["raw_sidecars"])),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "field_principles": field_principles(),
        "external_publication_authorized": False,
        "generator_weight_change_authorized": False,
        "default_promotion_authorized": False,
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> JsonDict:
    """Publish external absence without inventing protocol rows or owned work."""

    failed = [deepcopy(dict(row)) for row in checks if row.get("passed") is not True]
    first = (
        failed[0]
        if failed
        else {
            "check": "unknown_precondition",
            "upstream": "unknown",
            "path": "unknown",
            "field": "unknown",
            "op": "eq",
            "expected": True,
            "observed": False,
            "passed": False,
        }
    )
    reason = re.sub(r"[^a-z0-9]+", "_", str(first["check"]).lower()).strip("_")
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "title": "V663 lossless evidence-link protocol",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "complete": True,
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "positive_claim": False,
        "fresh_confirmatory_claim_allowed": False,
        "claim_scope": "descriptive_reuse",
        "verifier_is_oracle": False,
        "MODEL_SPECS": [],
        "model_specs": [],
        "no_model_load": True,
        "model_invoked": False,
        "historical_model_id": HISTORICAL_MODEL_ID,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "blocked_before_cached_aggregation",
        "inference_substrate_class": "blocked_no_run",
        "planned_inference_substrate_class": "no_model_load",
        "random_seed": RANDOM_SEED,
        "selection_salt": SELECTION_SALT,
        "duration_s": float(duration_s),
        "protocol_path": None,
        "protocol_sha256": None,
        "role_counts": {**{role: 0 for role in ROLE_COUNTS}, "pilot": 0},
        "evidence_feature_names": list(EVIDENCE_FEATURE_NAMES),
        "rows": [],
        "sample_size_budget": {
            "intended_source_groups": sum(SOURCE_ROLE_COUNTS.values()),
            "intended_scored_independent_units": SCORED_GROUPS,
            "intended_pilot_independent_units": PILOT_COUNT,
            "observed_independent_units": 0,
            "excluded_independent_units": 0,
            "censored_independent_units": 0,
            "failed_independent_units": 0,
            "unstarted_independent_units": SCORED_GROUPS + PILOT_COUNT,
        },
        "acceptance_gate_results": failed,
        "gate_check_summary": {
            "passed": False,
            "failed_count": len(failed),
            "failed_checks": [str(row["check"]) for row in failed],
            "first_failure": first,
        },
        "evidence_protocol_ready_score": 0,
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "raw_sidecars": {},
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "validation_receipts": [],
        "field_principles": field_principles(),
        "external_publication_authorized": False,
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _resolve(root: Path, label: str) -> Path:
    path = Path(label)
    return path if path.is_absolute() else root / path


def _receipt_valid(root: Path, receipt: Mapping[str, Any]) -> bool:
    path = _resolve(root, str(receipt.get("path") or ""))
    return path.is_file() and sha256_file(path) == receipt.get("sha256")


def validate_artifact(
    value: object,
    *,
    root: Path = REPO_ROOT,
    require_validation: bool = True,
) -> JsonDict:
    """Cold-check terminal identity, hashes, rows, readers, and no-model claims."""

    if not isinstance(value, Mapping):
        raise ValueError("artifact_object_required")
    artifact = dict(value)
    errors: list[str] = []
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("checksum_invalid")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_specs") != []:
        errors.append("model_specs_not_empty")
    if artifact.get("no_model_load") is not True or artifact.get("model_invoked") is not False:
        errors.append("model_invocation_claim_invalid")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("invocation_counts_nonzero")
    if artifact.get("fresh_confirmatory_claim_allowed") is not False:
        errors.append("freshness_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not set(REQUIRED_PRINCIPLE_FIELDS) <= set(principles):
        errors.append("field_principles_invalid")
    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        failure = artifact.get("gate_check_summary", {}).get("first_failure")
        required = {"check", "upstream", "path", "field", "op", "expected", "observed"}
        if not str(artifact.get("honest_verdict") or "").startswith("complete_blocked_"):
            errors.append("blocked_verdict_invalid")
        if not isinstance(failure, Mapping) or not required <= set(failure):
            errors.append("blocked_gate_summary_invalid")
        if (
            artifact.get("rows") != []
            or artifact.get("inference_substrate_class") != "blocked_no_run"
        ):
            errors.append("blocked_measurement_invalid")
    else:
        if artifact.get("verdict_class") not in {"null", "disqualified"}:
            errors.append("verdict_class_invalid")
        if artifact.get("inference_substrate_class") != "no_model_load":
            errors.append("substrate_invalid")
        protocol_path = _resolve(root, str(artifact.get("protocol_path") or ""))
        if not protocol_path.is_file() or sha256_file(protocol_path) != artifact.get(
            "protocol_sha256"
        ):
            errors.append("protocol_hash_invalid")
            protocol: Mapping[str, Any] = {}
        else:
            protocol_value = _load_object(protocol_path)
            protocol = protocol_value
        expected_counts = {**ROLE_COUNTS, "pilot": PILOT_COUNT}
        if (
            artifact.get("role_counts") != expected_counts
            or protocol.get("role_counts") != expected_counts
        ):
            errors.append("role_counts_invalid")
        if artifact.get("evidence_feature_names") != list(EVIDENCE_FEATURE_NAMES):
            errors.append("feature_schema_invalid")
        if protocol and (
            protocol.get("roundtrip_failure_count") != 0
            or protocol.get("overlap_audit", {}).get("overlap_count") != 0
            or protocol.get("fresh_confirmatory_claim_allowed") is not False
        ):
            errors.append("protocol_readiness_invalid")
        rows = artifact.get("rows")
        if not isinstance(rows, list) or not rows:
            errors.append("rows_missing")
        else:
            try:
                reduced = reduce_protocol_rows(rows)
            except (KeyError, TypeError, ValueError):
                errors.append("row_reduction_invalid")
            else:
                if reduced != artifact.get("independent_row_reduction"):
                    errors.append("row_reduction_mismatch")
                if reduced.get("unique_units") != SCORED_GROUPS + PILOT_COUNT:
                    errors.append("row_unit_count_invalid")
        if artifact.get("evidence_protocol_ready_score") not in {0, 1}:
            errors.append("ready_score_invalid")
        sidecars = artifact.get("raw_sidecars")
        if (
            not isinstance(sidecars, Mapping)
            or not sidecars
            or not all(
                isinstance(receipt, Mapping) and _receipt_valid(root, receipt)
                for receipt in sidecars.values()
            )
        ):
            errors.append("raw_sidecar_invalid")
        if require_validation and not _validation_passed(artifact.get("validation_receipts", [])):
            errors.append("validation_invalid")
    if errors:
        raise ValueError(";".join(errors))
    return {
        "valid": True,
        "blocked": blocked,
        "ready": artifact.get("evidence_protocol_ready_score") == 1,
        "row_count": len(artifact.get("rows", [])),
    }


def cold_replay(
    path: Path,
    *,
    root: Path = REPO_ROOT,
    require_validation: bool = True,
) -> JsonDict:
    """Reload and validate one exact candidate without model work."""

    return validate_artifact(_load_object(path), root=root, require_validation=require_validation)


def independent_reduce_artifact(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Independently reduce every candidate row and authenticate the protocol bytes."""

    artifact = _load_object(path)
    validation = validate_artifact(artifact, root=root, require_validation=False)
    reduction = reduce_protocol_rows(artifact.get("rows", [])) if artifact.get("rows") else None
    return {
        "passed": validation["valid"],
        "row_reduction_sha256": canonical_hash(reduction) if reduction is not None else None,
        "protocol_sha256": artifact.get("protocol_sha256"),
        "row_count": validation["row_count"],
    }


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase boundary with current monotonic elapsed time."""

    payload = {
        "phase": phase,
        "event": event,
        "elapsed_s": round(time.monotonic() - started, 3),
        **details,
    }
    print("[exp7588-progress] " + json.dumps(payload, sort_keys=True), flush=True)


def _terminal_commands(candidate: Path) -> list[validation_scope.CommandSpec]:  # pragma: no cover
    """Build fresh readers for the exact unpublished candidate."""

    python = ".venv/bin/python"
    return [
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), "--verify-artifact", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_reduction",
            (python, "-u", WRAPPER_PATH.as_posix(), "--independent-reduce", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
            300.0,
        ),
    ]


def _run_specs(  # pragma: no cover - subprocess boundary.
    root: Path,
    commands: Sequence[validation_scope.CommandSpec],
    log_dir: Path,
) -> list[JsonDict]:
    planned = [PlannedCommand(command, "required_validation", True) for command in commands]
    rows = run_categorized_commands(root, planned, log_dir=log_dir, heartbeat_s=60.0)
    for row in rows:
        row["worktree"] = str(root.resolve())
    return rows


def _publish_blocked(
    root: Path,
    checks: Sequence[Mapping[str, Any]],
    hashes: Sequence[Mapping[str, Any]],
    started: float,
) -> int:  # pragma: no cover
    artifact = build_blocked_artifact(checks, hashes, duration_s=time.monotonic() - started)
    atomic_json(root / RESULT_PATH, artifact)
    progress(
        started,
        "publish",
        "blocked_after",
        path=str(root / RESULT_PATH),
        verdict=artifact["honest_verdict"],
    )
    return 0


def run_experiment(root: Path, run_date: str) -> int:  # pragma: no cover - declared E2E.
    """Authenticate, validate, freeze, cold-read, and atomically publish the protocol."""

    started = time.monotonic()
    root = root.resolve()
    if root != REPO_ROOT.resolve() or run_date != RUN_DATE:
        raise ValueError("root_or_date_invalid")
    progress(started, "preconditions", "before")
    checks, source_hashes, roles = collect_preconditions(root)
    progress(started, "preconditions", "after", completed=len(checks))
    if any(row["passed"] is not True for row in checks):
        return _publish_blocked(root, checks, source_hashes, started)
    try:
        selected = select_roles(roles)
    except ValueError as exc:
        checks.append(
            precondition(
                "selected_role_roster",
                "exp7575",
                root / SOURCE_PATH,
                "salted_selection",
                "240_scored_plus_8_disjoint_pilot",
                str(exc),
            )
        )
        return _publish_blocked(root, checks, source_hashes, started)
    checks.append(
        precondition(
            "selected_role_roster",
            "exp7575",
            root / SOURCE_PATH,
            "salted_selection",
            "240_scored_plus_8_disjoint_pilot",
            "240_scored_plus_8_disjoint_pilot",
        )
    )
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        raw_dir / "affected_validation_manifest.json",
        {
            "experiment_id": AFFECTED_MANIFEST.experiment_id,
            "test_paths": list(AFFECTED_MANIFEST.test_paths),
            "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
            "static_paths": list(AFFECTED_MANIFEST.static_paths),
        },
    )
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7588-"))
    commands = build_command_plan(root, AFFECTED_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, AFFECTED_MANIFEST, commands)
    if plan_errors:
        raise RuntimeError("validation_plan_invalid:" + ",".join(plan_errors))
    progress(started, "affected_validation", "before", commands=len(commands))
    affected = _run_specs(root, commands, private_root / "logs" / "affected")
    progress(started, "affected_validation", "after", passed=_validation_passed(affected))
    if not _validation_passed(affected):
        raise RuntimeError("affected_validation_failed")
    progress(started, "protocol_freeze", "before", units=SCORED_GROUPS + PILOT_COUNT)
    bundle = freeze_protocol_files(raw_dir, selected, root=root)
    progress(
        started,
        "protocol_freeze",
        "after",
        units=SCORED_GROUPS + PILOT_COUNT,
        rows=len(bundle["rows"]),
    )
    candidate_path = raw_dir / "terminal_candidate.json"
    provisional = build_artifact(
        bundle,
        preconditions=checks,
        source_hashes=source_hashes,
        validation_receipts=affected,
        duration_s=time.monotonic() - started,
    )
    atomic_json(candidate_path, provisional)
    progress(started, "terminal_validation", "before", commands=4)
    terminal = _run_specs(
        root,
        _terminal_commands(candidate_path),
        private_root / "logs" / "terminal",
    )
    progress(started, "terminal_validation", "after", passed=_validation_passed(terminal))
    if not _validation_passed(terminal):
        raise RuntimeError("terminal_validation_failed")
    final = build_artifact(
        bundle,
        preconditions=checks,
        source_hashes=source_hashes,
        validation_receipts=[*affected, *terminal],
        duration_s=time.monotonic() - started,
    )
    atomic_json(candidate_path, final)
    progress(started, "exact_terminal_validation", "before", commands=4)
    exact = _run_specs(
        root,
        _terminal_commands(candidate_path),
        private_root / "logs" / "exact_terminal",
    )
    progress(started, "exact_terminal_validation", "after", passed=_validation_passed(exact))
    if not _validation_passed(exact):
        raise RuntimeError("exact_terminal_validation_failed")
    atomic_json(root / RESULT_PATH, final)
    progress(
        started,
        "publish",
        "after",
        path=str(root / RESULT_PATH),
        verdict=final["honest_verdict"],
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse production and the two read-only terminal reader modes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--verify-artifact", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def _argument_path(path: Path, root: Path) -> Path:  # pragma: no cover
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the producer or inspect one exact candidate without model work."""

    args = parse_args(argv)
    root = args.root.resolve()
    if args.verify_artifact is not None:
        result = cold_replay(
            _argument_path(args.verify_artifact, root),
            root=root,
            require_validation=True,
        )
        print(json.dumps({"mode": "cold_replay", **result}, sort_keys=True), flush=True)
        return 0
    if args.independent_reduce is not None:
        result = independent_reduce_artifact(
            _argument_path(args.independent_reduce, root), root=root
        )
        print(json.dumps({"mode": "independent_reduce", **result}, sort_keys=True), flush=True)
        return int(result["passed"] is not True)
    return run_experiment(root, args.date)
