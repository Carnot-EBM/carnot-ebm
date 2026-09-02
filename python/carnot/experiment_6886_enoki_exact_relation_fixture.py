"""Build the pinned Enoki source cache and exact anchored-relation fixture.

Spec refs: REQ-CONSTRAINT-6886 and SCENARIO-CONSTRAINT-6886-*.

The Enoki assets are provenance inputs. This module does not load the encoder.
It uses synthetic bounded fixtures because the public EnokiQA train split has
no hallucination labels. Clingo remains the independent exact authority.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
from typing import Any

from carnot import asp_energy


JsonDict = dict[str, Any]
Solver = Callable[[asp_energy.ASPProgram], list[list[str]]]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
RESULT_PATH = Path("results/experiment_6886_enoki_exact_relation_fixture.json")
DEFAULT_CACHE_ROOT = Path(
    os.environ.get(
        "CARNOT_EXP6886_CACHE",
        str(Path.home() / ".cache" / "carnot" / "exp6886_enoki_exact_relation_fixture"),
    )
)
RANDOM_SEED = 6886
RELATION_SCHEMA_VERSION = "anchored_relation_v1"
INFERENCE_SUBSTRATE = "deterministic_enoki_asset_and_exact_asp_fixture_no_llm"
ENCODER_REPO = "s-nlp/enoki-openie-encoder"
ENCODER_REVISION = "3be7767049d8db73ede6eab5c27c0d98faddeaf8"
ENOKIQA_REPO = "s-nlp/EnokiQA"
ENOKIQA_REVISION = "d764e01aa55ab90ca623b3a5fda24e122155b61e"
ENOKIQA_PARQUET_PATH = "data/train-00000-of-00001.parquet"
ENOKIQA_PARQUET_SHA256 = "sha256:e34d5539971bc74ce36fc990ddb971426f8287efa6c21a5c484e9235ebef82cd"
ENOKIQA_SHARD_SIZE = 64
ENOKIQA_SHARD_COLUMNS = (
    "id",
    "title",
    "question",
    "paragraph_context",
    "context_id",
    "wiki_url",
)
CACHE_SCHEMA = "carnot.exp6886.enoki_asset_cache.v1"
CACHE_MANIFEST_NAME = "asset_manifest.json"
MIN_FREE_BYTES = 2_000_000_000
SOLVER_TIMEOUT_S = 2.0
FAMILIES = (
    "graph_coloring",
    "scheduling",
    "non_monotonic_defaults",
    "contradictions",
    "cardinality_constraints",
)
CASES = ("valid", "omitted", "contradictory", "malformed", "abstain")

# These hashes bind the exact functional encoder snapshot. The banner is not
# functional model data and is deliberately excluded from the cache contract.
ENCODER_FILE_SHA256: dict[str, str] = {
    "README.md": "sha256:f63ab2792aa0b554a0775fd1962cdbe2e1a0dcc6234f19ecc2023d5a5defd897",
    "config.json": "sha256:b20e6438565b01e96c65829e59116d7a922fdce30b3b1a11a207ad6c5c4fab18",
    "configuration_enoki.py": "sha256:761fbafe4bc915844925506bd8a68166e547bf8960059374545bbde05da90687",
    "conversion_metadata.json": "sha256:af1611b406b186693d2fef3f4111143e184f717baf9b9efe2e817d98d9bc9a06",
    "inference.py": "sha256:298d97f86fcd11a6493cea39f41f3cfff03029aceeeb1abb6976798b13bb4734",
    "model.safetensors": "sha256:4045f55966726f49cd87760a0388f72987687854874c90321c1a65327888bd8e",
    "modeling_enoki.py": "sha256:8954e4e1631e82d5db9b658195ab9dfdbddf65a25e8299a4b8554e6e6cc9df22",
    "requirements.txt": "sha256:dd6d600981bfedbd92df452a8cc240a54c95fae64d3d991e1821b6f972ede629",
    "special_tokens_map.json": "sha256:2acd87ac94a342f188c92355fde28d2bc6878c43d804eb06a560a04da0bbb719",
    "tokenizer.json": "sha256:55d9646d5701fbb3acb4e5ec8bdd2a6cb3bcd91e5176570ddcddd4522d434dc9",
    "tokenizer_config.json": "sha256:a42cb78a999160feed7d8943a8db7b33645bc1fcacb06d0624f561334953ae8b",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "enoki_asset_receipts",
    "asset_revision_rows",
    "asset_license_rows",
    "relation_schema_version",
    "closed_vocabulary_manifest",
    "rows",
    "fixture_family_counts",
    "calibration_group_manifest",
    "sealed_held_group_manifest",
    "split_overlap_count",
    "relation_to_atom_rows",
    "atom_collision_rows",
    "unsupported_rows",
    "solver_parity_rows",
    "rule_violation_receipts",
    "prompt_nonexposure_results",
    "independent_solver_receipts",
    "random_seed",
    "reproducibility_checksum",
    "relation_fixture_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "A reason per field keeps every gate understandable during replay.",
    "preconditions_checked": "Exact resource checks prevent fabricated asset or solver evidence.",
    "inference_substrate": "The fixed declaration proves that no encoder or LLM inference ran.",
    "duration_s": "Measured wall time proves that the deterministic fixture builder executed.",
    "source_artifact_hashes": "Source hashes bind the result to the qualified compiler and code.",
    "enoki_asset_receipts": "Asset receipts pin external files without placing their bytes in Git.",
    "asset_revision_rows": "Immutable revisions prevent a moving repository head from changing results.",
    "asset_license_rows": "License rows preserve declarations and disclose missing declarations.",
    "relation_schema_version": "A versioned relation shape rejects incompatible future records.",
    "closed_vocabulary_manifest": "The closed map prevents free-form text from becoming an ASP atom.",
    "rows": "Typed rows keep assets, fixtures, relations, and exact checks independently auditable.",
    "fixture_family_counts": "Family counts prevent one easy family from carrying readiness.",
    "calibration_group_manifest": "The calibration manifest freezes commissioning identities.",
    "sealed_held_group_manifest": "The held manifest freezes future identities without labels.",
    "split_overlap_count": "A zero overlap count prevents calibration access to held groups.",
    "relation_to_atom_rows": "Per-record mapping rows expose every acceptance and abstention.",
    "atom_collision_rows": "Collision rows prove that distinct meanings do not share one atom.",
    "unsupported_rows": "Rejected rows preserve fail-closed reasons instead of disappearing.",
    "solver_parity_rows": "Per-fixture set equality keeps clingo as independent authority.",
    "rule_violation_receipts": "Local receipts explain each non-zero energy contribution.",
    "prompt_nonexposure_results": "Mechanical checks keep formal labels and sidecars out of prompts.",
    "independent_solver_receipts": "Solver receipts name the exact authority and all failures.",
    "random_seed": "The fixed seed identifies the deterministic generation contract.",
    "reproducibility_checksum": "A stable content hash detects silent result drift.",
    "relation_fixture_ready_score": "The downstream gate opens only when every exact check passes.",
    "gate_check_summary": "Expected and observed values make blocked results actionable.",
    "verifier_is_oracle": "True discloses that exact ASP execution defines correctness here.",
    "verdict_class": "The closed class prevents an oracle-backed check from posing as independent proof.",
    "honest_verdict": "A terminal prefix lets the conductor classify the completed run safely.",
}


class AssetUnavailable(RuntimeError):
    """Report an exact asset or cache precondition that did not pass."""


class RelationValidationError(ValueError):
    """Report why an anchored relation cannot enter the ASP compiler."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        suffix = f":{detail}" if detail else ""
        super().__init__(f"{code}{suffix}")


class SolverTimeoutError(TimeoutError):
    """Report that independent solver authority was unavailable within its bound."""


def canonical_json(value: Any) -> bytes:
    """Return stable UTF-8 bytes for hashing and cache identities."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    """Return one explicit SHA-256 identity."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data without depending on formatting."""

    return sha256_bytes(canonical_json(value))


def sha256_path(path: Path | str) -> str:
    """Hash a local file without loading external model code."""

    return sha256_bytes(Path(path).read_bytes())


def relation_provenance_hash(record: Mapping[str, Any]) -> str:
    """Hash every relation field except the self-referential provenance hash."""

    payload = {key: deepcopy(value) for key, value in record.items() if key != "provenance_hash"}
    return sha256_json(payload)


def _byte_span(source_text: str, needle: str, *, start_utf8: int = 0) -> JsonDict:
    source = source_text.encode("utf-8")
    target = needle.encode("utf-8")
    start = source.find(target, start_utf8)
    if start < 0:
        raise ValueError(f"anchor_not_found:{needle}")
    return {"start_utf8": start, "end_utf8": start + len(target), "text": needle}


def build_anchored_relation(
    *,
    record_id: str,
    source_text: str,
    evidence_text: str,
    subject_id: str,
    subject_text: str,
    predicate: str,
    object_id: str,
    object_text: str,
    polarity: str,
    asp_atom: str,
) -> JsonDict:
    """Build one record with exact byte anchors into the provided source text."""

    evidence_span = _byte_span(source_text, evidence_text)
    subject_span = _byte_span(
        source_text,
        subject_text,
        start_utf8=int(evidence_span["start_utf8"]),
    )
    object_span = _byte_span(
        source_text,
        object_text,
        start_utf8=int(evidence_span["start_utf8"]),
    )
    record: JsonDict = {
        "record_id": record_id,
        "source_text_hash": sha256_bytes(source_text.encode("utf-8")),
        "source_span": {
            "start_utf8": 0,
            "end_utf8": len(source_text.encode("utf-8")),
            "text": source_text,
        },
        "evidence_span": evidence_span,
        "subject": {"entity_id": subject_id, "text": subject_text, "span": subject_span},
        "predicate": predicate,
        "object": {"entity_id": object_id, "text": object_text, "span": object_span},
        "polarity": polarity,
        "normalized_tuple": [subject_id, predicate, object_id, polarity],
        "asp_atom": asp_atom,
        "provenance_hash": "",
    }
    record["provenance_hash"] = relation_provenance_hash(record)
    return record


def _family_definition(family: str, ordinal: int) -> JsonDict:
    if family == "graph_coloring":
        return {
            "subject_id": f"node_{ordinal}",
            "subject_text": f"n{ordinal}",
            "predicate": "has_color",
            "object_id": "red",
            "object_text": "red",
            "positive_atom": f"gc_{ordinal}_red",
            "negative_atom": f"gc_{ordinal}_not_red",
            "positive_evidence": f"Node n{ordinal} has color red.",
            "negative_evidence": f"Node n{ordinal} does not have color red.",
            "base_program": (
                f"1 {{gc_{ordinal}_red; gc_{ordinal}_blue}} 1.\n"
                f":- gc_{ordinal}_red, gc_{ordinal}_not_red.\n"
            ),
        }
    if family == "scheduling":
        return {
            "subject_id": f"task_{ordinal}",
            "subject_text": f"task{ordinal}",
            "predicate": "scheduled_at",
            "object_id": "morning",
            "object_text": "morning",
            "positive_atom": f"sc_{ordinal}_morning",
            "negative_atom": f"sc_{ordinal}_not_morning",
            "positive_evidence": f"Task task{ordinal} is scheduled in the morning.",
            "negative_evidence": f"Task task{ordinal} is not scheduled in the morning.",
            "base_program": (
                f"1 {{sc_{ordinal}_morning; sc_{ordinal}_evening}} 1.\n"
                f":- sc_{ordinal}_morning, sc_{ordinal}_not_morning.\n"
            ),
        }
    if family == "non_monotonic_defaults":
        return {
            "subject_id": f"bird_{ordinal}",
            "subject_text": f"b{ordinal}",
            "predicate": "has_condition",
            "object_id": "injured",
            "object_text": "injured",
            "positive_atom": f"df_{ordinal}_injured",
            "negative_atom": f"df_{ordinal}_not_injured",
            "positive_evidence": f"Bird b{ordinal} is injured.",
            "negative_evidence": f"Bird b{ordinal} is not injured.",
            "base_program": (
                f"df_{ordinal}_bird.\n"
                f"df_{ordinal}_flies :- df_{ordinal}_bird, not df_{ordinal}_injured.\n"
                f":- df_{ordinal}_injured, df_{ordinal}_not_injured.\n"
            ),
        }
    if family == "contradictions":
        return {
            "subject_id": f"claim_{ordinal}",
            "subject_text": f"claim{ordinal}",
            "predicate": "has_truth_status",
            "object_id": "accepted",
            "object_text": "accepted",
            "positive_atom": f"ct_{ordinal}_accepted",
            "negative_atom": f"ct_{ordinal}_not_accepted",
            "positive_evidence": f"Claim claim{ordinal} is accepted.",
            "negative_evidence": f"Claim claim{ordinal} is not accepted.",
            "base_program": (
                f"0 {{ct_{ordinal}_accepted; ct_{ordinal}_not_accepted}} 2.\n"
                f":- ct_{ordinal}_accepted, ct_{ordinal}_not_accepted.\n"
            ),
        }
    if family == "cardinality_constraints":
        return {
            "subject_id": f"set_{ordinal}",
            "subject_text": f"s{ordinal}",
            "predicate": "selects",
            "object_id": "option_a",
            "object_text": "option A",
            "positive_atom": f"cd_{ordinal}_a",
            "negative_atom": f"cd_{ordinal}_not_a",
            "positive_evidence": f"Set s{ordinal} selects option A.",
            "negative_evidence": f"Set s{ordinal} does not select option A.",
            "base_program": (
                f"1 {{cd_{ordinal}_a; cd_{ordinal}_b; cd_{ordinal}_c}} 2.\n"
                f":- cd_{ordinal}_a, cd_{ordinal}_not_a.\n"
            ),
        }
    raise ValueError(f"unsupported_family:{family}")


def build_closed_vocabulary(family: str, ordinal: int) -> list[JsonDict]:
    """Return the two explicit polarity mappings for one bounded fixture."""

    definition = _family_definition(family, ordinal)
    rows = []
    for polarity, atom in (
        ("positive", definition["positive_atom"]),
        ("negative", definition["negative_atom"]),
    ):
        normalized = [
            definition["subject_id"],
            definition["predicate"],
            definition["object_id"],
            polarity,
        ]
        rows.append(
            {
                "family": family,
                "subject_id": definition["subject_id"],
                "predicate": definition["predicate"],
                "object_id": definition["object_id"],
                "polarity": polarity,
                "normalized_tuple": normalized,
                "asp_atom": atom,
            }
        )
    return rows


def validate_vocabulary(vocabulary: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return every non-injective atom collision in a closed vocabulary."""

    by_atom: dict[str, list[list[str]]] = {}
    for row in vocabulary:
        atom = str(row["asp_atom"])
        normalized = [str(value) for value in row["normalized_tuple"]]
        if normalized not in by_atom.setdefault(atom, []):
            by_atom[atom].append(normalized)
    return [
        {"asp_atom": atom, "normalized_tuples": tuples, "passed": False}
        for atom, tuples in sorted(by_atom.items())
        if len(tuples) > 1
    ]


def _span_bounds(source_text: str, span: Mapping[str, Any]) -> tuple[int, int]:
    source = source_text.encode("utf-8")
    start = span.get("start_utf8")
    end = span.get("end_utf8")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end <= start
        or end > len(source)
    ):
        raise RelationValidationError("invalid_span")
    return start, end


def _slice_span(source_text: str, span: Mapping[str, Any]) -> str:
    source = source_text.encode("utf-8")
    start, end = _span_bounds(source_text, span)
    try:
        value = source[start:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RelationValidationError("invalid_span", "utf8_boundary") from exc
    if value != span.get("text"):
        if source_text[start:end] == span.get("text"):
            raise RelationValidationError("invalid_span", "character_offsets")
        raise RelationValidationError("span_text_mismatch")
    return value


def _inside(inner: Mapping[str, Any], outer: Mapping[str, Any]) -> bool:
    return int(outer["start_utf8"]) <= int(inner["start_utf8"]) and int(inner["end_utf8"]) <= int(
        outer["end_utf8"]
    )


def validate_relation(
    record: Mapping[str, Any],
    source_text: str,
    vocabulary: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Validate exact anchors and return one accepted relation-to-atom row."""

    required = {
        "record_id",
        "source_text_hash",
        "source_span",
        "evidence_span",
        "subject",
        "predicate",
        "object",
        "polarity",
        "normalized_tuple",
        "asp_atom",
        "provenance_hash",
    }
    missing = sorted(required - set(record))
    if missing:
        raise RelationValidationError("missing_field", ",".join(missing))
    if record["source_text_hash"] != sha256_bytes(source_text.encode("utf-8")):
        raise RelationValidationError("source_text_hash")
    _span_bounds(source_text, record["source_span"])
    _span_bounds(source_text, record["evidence_span"])
    _span_bounds(source_text, record["subject"]["span"])
    _span_bounds(source_text, record["object"]["span"])
    if not _inside(record["evidence_span"], record["source_span"]):
        raise RelationValidationError("span_outside_source")
    if not _inside(record["subject"]["span"], record["evidence_span"]) or not _inside(
        record["object"]["span"], record["evidence_span"]
    ):
        raise RelationValidationError("span_outside_evidence")
    source_value = _slice_span(source_text, record["source_span"])
    _slice_span(source_text, record["evidence_span"])
    subject = record["subject"]
    obj = record["object"]
    _slice_span(source_text, subject["span"])
    _slice_span(source_text, obj["span"])
    if source_value != source_text:
        raise RelationValidationError("source_span_not_complete")
    if subject["span"]["text"] != subject.get("text") or obj["span"]["text"] != obj.get("text"):
        raise RelationValidationError("span_text_mismatch")
    expected_tuple = [
        str(subject.get("entity_id")),
        str(record["predicate"]),
        str(obj.get("entity_id")),
        str(record["polarity"]),
    ]
    if list(record["normalized_tuple"]) != expected_tuple:
        raise RelationValidationError("normalized_tuple")
    if record["polarity"] not in {"positive", "negative"}:
        raise RelationValidationError("invalid_polarity")
    subjects = {str(row["subject_id"]) for row in vocabulary}
    objects = {str(row["object_id"]) for row in vocabulary}
    predicates = {str(row["predicate"]) for row in vocabulary}
    if expected_tuple[0] not in subjects or expected_tuple[2] not in objects:
        raise RelationValidationError("unknown_entity")
    if expected_tuple[1] not in predicates:
        raise RelationValidationError("unsupported_predicate")
    mapping = {tuple(str(value) for value in row["normalized_tuple"]): row for row in vocabulary}
    expected_row = mapping.get(tuple(expected_tuple))
    if expected_row is None:
        raise RelationValidationError("unmapped_tuple")
    if record["asp_atom"] != expected_row["asp_atom"]:
        raise RelationValidationError("asp_atom_mismatch")
    if record["provenance_hash"] != relation_provenance_hash(record):
        raise RelationValidationError("provenance_hash")
    return {
        "record_id": str(record["record_id"]),
        "normalized_tuple": expected_tuple,
        "asp_atom": str(record["asp_atom"]),
        "polarity": str(record["polarity"]),
        "provenance_hash": str(record["provenance_hash"]),
        "accepted": True,
        "reason": "accepted_exact_anchor_and_closed_map",
    }


def validate_relation_set(
    records: Sequence[Mapping[str, Any]],
    source_text: str,
    vocabulary: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Map a record set while preserving duplicate and rejection receipts."""

    rows: list[JsonDict] = []
    record_ids: set[str] = set()
    tuples: set[tuple[str, ...]] = set()
    atoms: dict[str, tuple[str, ...]] = {}
    for record in records:
        record_id = str(record.get("record_id", "missing"))
        normalized = tuple(str(value) for value in record.get("normalized_tuple", []))
        if record_id in record_ids:
            rows.append(
                {"record_id": record_id, "accepted": False, "reason": "duplicate_record_id"}
            )
            continue
        record_ids.add(record_id)
        if normalized in tuples:
            rows.append(
                {"record_id": record_id, "accepted": False, "reason": "duplicate_normalized_tuple"}
            )
            continue
        tuples.add(normalized)
        try:
            row = validate_relation(record, source_text, vocabulary)
        except (KeyError, TypeError, RelationValidationError) as exc:
            reason = exc.code if isinstance(exc, RelationValidationError) else "malformed_record"
            rows.append({"record_id": record_id, "accepted": False, "reason": reason})
            continue
        prior = atoms.get(row["asp_atom"])
        current = tuple(row["normalized_tuple"])
        if prior is not None and prior != current:
            rows.append({"record_id": record_id, "accepted": False, "reason": "atom_collision"})
            continue
        atoms[row["asp_atom"]] = current
        rows.append(row)
    return rows


def _fixture_source(definition: Mapping[str, Any], expected_case: str, ordinal: int) -> str:
    prefix = "Café evidence: " if ordinal % 2 == 0 else "Evidence: "
    positive = str(definition["positive_evidence"])
    negative = str(definition["negative_evidence"])
    if expected_case == "contradictory":
        return f"{prefix}{positive} {negative}"
    return f"{prefix}{positive}"


def _relation_for_definition(
    definition: Mapping[str, Any],
    source_text: str,
    *,
    fixture_id: str,
    polarity: str,
) -> JsonDict:
    evidence = str(definition[f"{polarity}_evidence"])
    return build_anchored_relation(
        record_id=f"{fixture_id}-{polarity}",
        source_text=source_text,
        evidence_text=evidence,
        subject_id=str(definition["subject_id"]),
        subject_text=str(definition["subject_text"]),
        predicate=str(definition["predicate"]),
        object_id=str(definition["object_id"]),
        object_text=str(definition["object_text"]),
        polarity=polarity,
        asp_atom=str(definition[f"{polarity}_atom"]),
    )


def build_fixtures() -> list[JsonDict]:
    """Build 150 balanced, group-disjoint exact fixtures."""

    fixtures: list[JsonDict] = []
    for ordinal in range(30):
        group_id = f"relation_group_{ordinal:02d}"
        split = "calibration" if ordinal < 15 else "held"
        expected_case = CASES[ordinal % len(CASES)]
        for family in FAMILIES:
            fixture_id = f"{family}_{ordinal:02d}"
            definition = _family_definition(family, ordinal)
            source_text = _fixture_source(definition, expected_case, ordinal)
            proposals: list[JsonDict] = []
            if expected_case in {"valid", "malformed", "abstain"}:
                proposals.append(
                    _relation_for_definition(
                        definition,
                        source_text,
                        fixture_id=fixture_id,
                        polarity="positive",
                    )
                )
            elif expected_case == "contradictory":
                proposals.extend(
                    [
                        _relation_for_definition(
                            definition,
                            source_text,
                            fixture_id=fixture_id,
                            polarity="positive",
                        ),
                        _relation_for_definition(
                            definition,
                            source_text,
                            fixture_id=fixture_id,
                            polarity="negative",
                        ),
                    ]
                )
            if expected_case == "malformed":
                proposals[0]["evidence_span"]["end_utf8"] = proposals[0]["evidence_span"][
                    "start_utf8"
                ]
            if expected_case == "abstain":
                proposals[0]["predicate"] = "unsupported_relation"
                proposals[0]["normalized_tuple"][1] = "unsupported_relation"
                proposals[0]["provenance_hash"] = relation_provenance_hash(proposals[0])
            fixtures.append(
                {
                    "fixture_id": fixture_id,
                    "group_id": group_id,
                    "split": split,
                    "family": family,
                    "expected_case": expected_case,
                    "source_text": source_text,
                    "source_text_hash": sha256_bytes(source_text.encode("utf-8")),
                    "vocabulary": build_closed_vocabulary(family, ordinal),
                    "proposals": proposals,
                    "base_program": str(definition["base_program"]),
                }
            )
    if len(fixtures) != 150:
        raise ValueError("fixture_count")  # pragma: no cover - fixed loop invariant
    return fixtures


def public_prompt_view(fixture: Mapping[str, Any]) -> JsonDict:
    """Return the only fixture fields that a future proposal arm may read."""

    return {
        "fixture_id": fixture["fixture_id"],
        "group_id": fixture["group_id"],
        "family": fixture["family"],
        "source_text": fixture["source_text"],
        "source_text_hash": fixture["source_text_hash"],
        "relation_schema_version": RELATION_SCHEMA_VERSION,
        "allowed_predicates": sorted({row["predicate"] for row in fixture["vocabulary"]}),
    }


def _mapping_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_mapping_keys(child))
    return keys


def _hidden_tokens(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if len(value) >= 8 else []
    if isinstance(value, Mapping):
        return [token for child in value.values() for token in _hidden_tokens(child)]
    if isinstance(value, list):
        return [token for child in value for token in _hidden_tokens(child)]
    return []


def audit_prompt_nonexposure(
    prompt_view: Mapping[str, Any],
    hidden_payload: Mapping[str, Any],
    sidecar_paths: Sequence[str],
) -> list[JsonDict]:
    """Check hidden keys, distinctive values, and sidecar paths mechanically."""

    serialized = canonical_json(prompt_view).decode("utf-8")
    keys = _mapping_keys(prompt_view)
    rows: list[JsonDict] = []
    for field, value in hidden_payload.items():
        leaked_tokens = sorted({token for token in _hidden_tokens(value) if token in serialized})
        rows.append(
            {
                "check": f"hidden_field:{field}",
                "passed": field not in keys and not leaked_tokens,
                "key_present": field in keys,
                "leaked_value_hashes": [
                    sha256_bytes(token.encode("utf-8")) for token in leaked_tokens
                ],
            }
        )
    for path in sidecar_paths:
        rows.append(
            {
                "check": f"sidecar_path:{sha256_bytes(path.encode('utf-8'))}",
                "passed": path not in serialized,
                "key_present": False,
                "leaked_value_hashes": []
                if path not in serialized
                else [sha256_bytes(path.encode())],
            }
        )
    return rows


def solve_with_timeout(
    program: asp_energy.ASPProgram,
    *,
    timeout_s: float,
    solver: Solver = asp_energy.solve_with_clingo,
) -> list[list[str]]:
    """Bound one synchronous clingo call without hiding timeout failures."""

    if timeout_s <= 0:
        raise SolverTimeoutError(f"solver_timeout:{timeout_s}")

    def alarm_handler(_signum: int, _frame: Any) -> None:
        raise SolverTimeoutError(f"solver_timeout:{timeout_s}")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return solver(program)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _local_violation_receipts(
    fixture_id: str, compiled: asp_energy.CompiledASPProgram
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    states = compiled.enumerate_states()
    for term in compiled.energy_terms:
        for state in states:
            receipt = compiled.decompose_state(state)
            row = next(item for item in receipt["terms"] if item["rule_id"] == term.rule_id)
            if int(row["energy"]) > 0:
                rows.append(
                    {
                        "fixture_id": fixture_id,
                        "rule_id": term.rule_id,
                        "kind": term.kind,
                        "energy": int(row["energy"]),
                        "violation": row["violation"],
                        "state_hash": sha256_json(state),
                    }
                )
                break
    return rows


def evaluate_fixture(
    fixture: Mapping[str, Any],
    *,
    solver: Solver = asp_energy.solve_with_clingo,
    timeout_s: float = SOLVER_TIMEOUT_S,
) -> JsonDict:
    """Map relations, compile accepted atoms, and compare exact answer sets."""

    relation_rows = validate_relation_set(
        fixture["proposals"], fixture["source_text"], fixture["vocabulary"]
    )
    accepted_atoms = sorted(
        {row["asp_atom"] for row in relation_rows if row.get("accepted") is True}
    )
    fact_text = "".join(f"{atom}.\n" for atom in accepted_atoms)
    program_text = f"{fixture['base_program']}{fact_text}"
    compiled = asp_energy.compile_program(program_text, program_id=str(fixture["fixture_id"]))
    zero_states = compiled.zero_energy_states()
    started = time.perf_counter()
    solver_error: str | None = None
    try:
        answer_sets = solve_with_timeout(compiled.program, timeout_s=timeout_s, solver=solver)
    except SolverTimeoutError as exc:
        answer_sets = []
        solver_error = str(exc)
    solver_duration = time.perf_counter() - started
    parity = solver_error is None and answer_sets == zero_states
    parity_row = {
        "fixture_id": fixture["fixture_id"],
        "family": fixture["family"],
        "solver_answer_set_count": len(answer_sets),
        "zero_energy_state_count": len(zero_states),
        "solver_answer_sets_hash": sha256_json(answer_sets),
        "zero_energy_states_hash": sha256_json(zero_states),
        "semantic_parity": parity,
        "solver_error": solver_error,
        "passed": parity,
    }
    return {
        "fixture": dict(fixture),
        "relation_rows": relation_rows,
        "accepted_atoms": accepted_atoms,
        "asp_program": program_text,
        "answer_sets": answer_sets,
        "zero_energy_states": zero_states,
        "solver_receipt": {
            "name_version": asp_energy.solver_name_version(),
            "duration_observed": solver_duration >= 0.0,
            "error": solver_error,
            "answer_sets_hash": sha256_json(answer_sets),
        },
        "solver_parity_row": parity_row,
        "rule_violation_receipts": _local_violation_receipts(str(fixture["fixture_id"]), compiled),
    }


def _case_passed(report: Mapping[str, Any]) -> bool:
    fixture = report["fixture"]
    expected = fixture["expected_case"]
    relation_rows = report["relation_rows"]
    accepted = [row for row in relation_rows if row.get("accepted") is True]
    rejected = [row for row in relation_rows if row.get("accepted") is not True]
    if expected == "valid":
        return len(accepted) == 1 and not rejected
    if expected == "omitted":
        return not relation_rows
    if expected == "contradictory":
        return len(accepted) == 2 and report["solver_parity_row"]["solver_answer_set_count"] == 0
    if expected == "malformed":
        return len(rejected) == 1 and rejected[0]["reason"] == "invalid_span"
    return len(rejected) == 1 and rejected[0]["reason"] == "unsupported_predicate"


def _manifest_payload() -> JsonDict:
    return {
        "schema": CACHE_SCHEMA,
        "encoder_revision": ENCODER_REVISION,
        "enokiqa_revision": ENOKIQA_REVISION,
        "shard_size": ENOKIQA_SHARD_SIZE,
        "shard_columns": list(ENOKIQA_SHARD_COLUMNS),
    }


def check_cache_identity(cache_root: Path | str) -> JsonDict:
    """Reject an existing cache target with a different immutable identity."""

    path = Path(cache_root) / CACHE_MANIFEST_NAME
    expected = _manifest_payload()
    if not path.is_file():
        return {"compatible": True, "manifest_present": False, "path": str(path)}
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetUnavailable(f"cache_manifest_unreadable:{exc}") from exc
    for field, value in expected.items():
        if observed.get(field) != value:
            raise AssetUnavailable(f"cache_hash_drift:{field}")
    return {"compatible": True, "manifest_present": True, "path": str(path)}


def read_enokiqa_source_rows() -> list[JsonDict]:
    """Read only the first bounded source columns from the exact Parquet revision."""

    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    uri = f"hf://datasets/{ENOKIQA_REPO}@{ENOKIQA_REVISION}/{ENOKIQA_PARQUET_PATH}"
    filesystem = HfFileSystem()
    with filesystem.open(uri, "rb") as source:
        parquet = pq.ParquetFile(source)
        table = parquet.read_row_group(0, columns=list(ENOKIQA_SHARD_COLUMNS))
    return [dict(row) for row in table.slice(0, ENOKIQA_SHARD_SIZE).to_pylist()]


def prepare_enoki_assets(
    cache_root: Path | str,
    *,
    hub_download: Callable[..., str] | None = None,
    dataset_reader: Callable[[], list[JsonDict]] | None = None,
) -> JsonDict:
    """Cache exact encoder files plus one bounded source-only dataset shard."""

    root = Path(cache_root)
    check_cache_identity(root)
    root.mkdir(parents=True, exist_ok=True)
    if hub_download is None:
        from huggingface_hub import hf_hub_download

        hub_download = hf_hub_download
    reader = dataset_reader or read_enokiqa_source_rows
    encoder_files: list[JsonDict] = []
    try:
        for filename, expected_hash in sorted(ENCODER_FILE_SHA256.items()):
            path = Path(
                hub_download(
                    ENCODER_REPO,
                    filename,
                    revision=ENCODER_REVISION,
                    repo_type=None,
                )
            )
            observed_hash = sha256_path(path)
            if observed_hash != expected_hash:
                raise AssetUnavailable(f"cache_hash_drift:{filename}")
            encoder_files.append({"path": str(path), "filename": filename, "sha256": observed_hash})
        rows = reader()
    except AssetUnavailable:
        raise
    except Exception as exc:
        raise AssetUnavailable(f"asset_unavailable:{type(exc).__name__}:{exc}") from exc
    if len(rows) != ENOKIQA_SHARD_SIZE:
        raise AssetUnavailable(f"bounded_shard_size:{len(rows)}")
    projected = [{column: row[column] for column in ENOKIQA_SHARD_COLUMNS} for row in rows]
    row_hashes = [sha256_json(row) for row in projected]
    shard_path = root / "enokiqa_source_shard.jsonl"
    shard_path.write_text(
        "".join(canonical_json(row).decode("utf-8") + "\n" for row in projected),
        encoding="utf-8",
    )
    shard_hash = sha256_path(shard_path)
    manifest = {
        **_manifest_payload(),
        "encoder_file_hashes": {row["filename"]: row["sha256"] for row in encoder_files},
        "source_parquet_sha256": ENOKIQA_PARQUET_SHA256,
        "shard_sha256": shard_hash,
        "row_hashes": row_hashes,
    }
    manifest_path = root / CACHE_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "hashes_valid": True,
        "asset_receipts": [
            {
                "asset_id": "enoki_encoder",
                "source_url": f"https://huggingface.co/{ENCODER_REPO}/tree/{ENCODER_REVISION}",
                "revision": ENCODER_REVISION,
                "cache_path": str(Path(encoder_files[0]["path"]).parent),
                "files": encoder_files,
                "model_loaded": False,
            },
            {
                "asset_id": "enokiqa_source_shard",
                "source_url": (
                    f"https://huggingface.co/datasets/{ENOKIQA_REPO}/resolve/"
                    f"{ENOKIQA_REVISION}/{ENOKIQA_PARQUET_PATH}"
                ),
                "revision": ENOKIQA_REVISION,
                "cache_path": str(shard_path),
                "files": [
                    {
                        "path": str(shard_path),
                        "sha256": shard_hash,
                        "source_parquet_sha256": ENOKIQA_PARQUET_SHA256,
                    }
                ],
                "projected_columns": list(ENOKIQA_SHARD_COLUMNS),
                "source_row_count": ENOKIQA_SHARD_SIZE,
            },
        ],
        "revision_rows": [
            {
                "asset_id": "enoki_encoder",
                "expected_revision": ENCODER_REVISION,
                "observed_revision": ENCODER_REVISION,
                "passed": True,
            },
            {
                "asset_id": "enokiqa_source_shard",
                "expected_revision": ENOKIQA_REVISION,
                "observed_revision": ENOKIQA_REVISION,
                "passed": True,
            },
        ],
        "license_rows": [
            {
                "asset_id": "enoki_encoder",
                "declared_license": None,
                "base_model": "answerdotai/ModernBERT-large",
                "base_model_license": "apache-2.0",
                "license_status": "model_card_has_no_derivative_license_declaration",
            },
            {
                "asset_id": "enokiqa_source_shard",
                "declared_license": "cc-by-sa-4.0",
                "license_status": "declared",
            },
        ],
        "row_hashes": row_hashes,
        "shard_size": len(projected),
        "cache_manifest_path": str(manifest_path),
    }


def _source_artifact_hashes() -> JsonDict:
    paths = {
        "exp6274_artifact": Path("results/experiment_6274_asp_energy_semantic_compiler.json"),
        "exp6275_artifact": Path(
            "results/experiment_6275_flagship_asp_constraint_verification_benchmark.json"
        ),
        "exp6875_artifact": Path("results/experiment_6875_text_anchored_relation_asp_fixture.json"),
        "asp_energy_compiler": Path("python/carnot/asp_energy.py"),
        "exp6274_harness": Path("python/carnot/experiment_6274_asp_energy_semantic_compiler.py"),
        "module": Path("python/carnot/experiment_6886_enoki_exact_relation_fixture.py"),
        "wrapper": Path("scripts/experiments/experiment_6886_enoki_exact_relation_fixture.py"),
        "tests": Path("tests/python/test_experiment_6886_enoki_exact_relation_fixture.py"),
        "spec": SPEC_PATH,
    }
    return {
        name: {"path": path.as_posix(), "sha256": sha256_path(REPO_ROOT / path)}
        for name, path in paths.items()
        if (REPO_ROOT / path).is_file()
    }


def _compiler_qualified() -> JsonDict:
    path = REPO_ROOT / "results/experiment_6274_asp_energy_semantic_compiler.json"
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "observed": f"{type(exc).__name__}:{exc}"}
    passed = (
        artifact.get("status") == "complete"
        and artifact.get("asp_energy_semantic_ready_score") == 1.0
        and artifact.get("parity_failure_count") == 0
    )
    return {
        "available": passed,
        "path": str(path),
        "sha256": sha256_path(path),
        "observed": {
            "status": artifact.get("status"),
            "asp_energy_semantic_ready_score": artifact.get("asp_energy_semantic_ready_score"),
            "parity_failure_count": artifact.get("parity_failure_count"),
        },
    }


def _preconditions(cache_root: Path, asset_bundle: Mapping[str, Any]) -> JsonDict:
    compiler = _compiler_qualified()
    solver_version = asp_energy.solver_name_version()
    disk = shutil.disk_usage(cache_root.parent if cache_root.parent.exists() else REPO_ROOT)
    cache = check_cache_identity(cache_root)
    return {
        "qualified_exp6274_compiler": compiler,
        "independent_solver": {
            "available": not solver_version.endswith(":missing"),
            "name_version": solver_version,
        },
        "enoki_assets": {
            "available": bool(asset_bundle.get("hashes_valid")),
            "encoder_revision": ENCODER_REVISION,
            "enokiqa_revision": ENOKIQA_REVISION,
        },
        "disk": {
            "available": disk.free >= MIN_FREE_BYTES,
            "free_bytes": disk.free,
            "minimum_bytes": MIN_FREE_BYTES,
        },
        "cache_identity": cache,
        "no_llm_or_encoder_inference": True,
    }


def _check(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    return {"check": name, "expected": expected, "observed": observed, "passed": bool(passed)}


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return all readiness checks plus the first exact failure."""

    failures = [dict(row) for row in checks if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "checks": [dict(row) for row in checks],
        "passed": not failures,
        "failed_check": first.get("check") if first else None,
        "expected": first.get("expected") if first else "all checks pass",
        "observed": first.get("observed") if first else "all checks pass",
        "failed_checks": failures,
    }


def _write_sidecars(reports: Sequence[Mapping[str, Any]], cache_root: Path) -> JsonDict:
    cache_root.mkdir(parents=True, exist_ok=True)
    receipts: JsonDict = {}
    for split in ("calibration", "held"):
        rows = []
        for report in reports:
            fixture = report["fixture"]
            if fixture["split"] != split:
                continue
            rows.append(
                {
                    "fixture_id": fixture["fixture_id"],
                    "group_id": fixture["group_id"],
                    "expected_case": fixture["expected_case"],
                    "asp_program": report["asp_program"],
                    "answer_sets": report["answer_sets"],
                    "zero_energy_states": report["zero_energy_states"],
                    "solver_receipt": report["solver_receipt"],
                }
            )
        path = cache_root / f"sealed_{split}_formal_sidecar.json"
        payload = {
            "schema": "carnot.exp6886.sealed_formal_sidecar.v1",
            "split": split,
            "rows": rows,
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        receipts[split] = {"path": str(path), "sha256": sha256_path(path), "row_count": len(rows)}
    return receipts


def _split_manifest(fixtures: Sequence[Mapping[str, Any]], split: str) -> JsonDict:
    groups = sorted({str(row["group_id"]) for row in fixtures if row["split"] == split})
    fixture_ids = sorted(str(row["fixture_id"]) for row in fixtures if row["split"] == split)
    return {
        "split": split,
        "group_ids": groups,
        "group_count": len(groups),
        "fixture_count": len(fixture_ids),
        "public_fixture_manifest_hash": sha256_json(fixture_ids),
        "labels_in_manifest": False,
    }


def _fixture_row(report: Mapping[str, Any]) -> JsonDict:
    fixture = report["fixture"]
    prompt = public_prompt_view(fixture)
    return {
        "row_type": "fixture",
        "fixture_id": fixture["fixture_id"],
        "group_id": fixture["group_id"],
        "split": fixture["split"],
        "family": fixture["family"],
        "source_text_hash": fixture["source_text_hash"],
        "prompt_view_hash": sha256_json(prompt),
        "case_contract_passed": _case_passed(report),
    }


def _blocked_artifact(
    *,
    date: str,
    duration_s: float,
    failed_check: str,
    observed: Any,
) -> JsonDict:
    checks = [_check(failed_check, True, observed, False)]
    artifact: JsonDict = {
        "schema": "carnot.exp6886.enoki_exact_relation_fixture.v1",
        "experiment_id": 6886,
        "run_date": date,
        "status": "blocked",
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": {"failed_before_fixture_build": True},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": _source_artifact_hashes(),
        "enoki_asset_receipts": [],
        "asset_revision_rows": [],
        "asset_license_rows": [],
        "relation_schema_version": RELATION_SCHEMA_VERSION,
        "closed_vocabulary_manifest": {},
        "rows": [],
        "fixture_family_counts": {},
        "calibration_group_manifest": {},
        "sealed_held_group_manifest": {},
        "split_overlap_count": 0,
        "relation_to_atom_rows": [],
        "atom_collision_rows": [],
        "unsupported_rows": [],
        "solver_parity_rows": [],
        "rule_violation_receipts": [],
        "prompt_nonexposure_results": [],
        "independent_solver_receipts": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "relation_fixture_ready_score": 0,
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_enoki_exact_relation_fixture",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def build_artifact(
    *,
    date: str,
    cache_root: Path | str,
    asset_bundle: Mapping[str, Any],
    duration_s: float,
    fixtures: Sequence[Mapping[str, Any]] | None = None,
    solver: Solver = asp_energy.solve_with_clingo,
    timeout_s: float = SOLVER_TIMEOUT_S,
) -> JsonDict:
    """Build and validate the complete exact fixture artifact."""

    root = Path(cache_root)
    fixture_rows = list(fixtures or build_fixtures())
    reports = [evaluate_fixture(row, solver=solver, timeout_s=timeout_s) for row in fixture_rows]
    sidecars = _write_sidecars(reports, root)
    calibration = _split_manifest(fixture_rows, "calibration")
    held = _split_manifest(fixture_rows, "held")
    overlap = sorted(set(calibration["group_ids"]) & set(held["group_ids"]))
    vocabulary = [entry for fixture in fixture_rows for entry in fixture["vocabulary"]]
    collisions = validate_vocabulary(vocabulary)
    relation_rows = [
        {"fixture_id": report["fixture"]["fixture_id"], **row}
        for report in reports
        for row in report["relation_rows"]
    ]
    unsupported = [row for row in relation_rows if row.get("accepted") is not True]
    solver_rows = [dict(report["solver_parity_row"]) for report in reports]
    violation_rows = [row for report in reports for row in report["rule_violation_receipts"]]
    nonexposure: list[JsonDict] = []
    sidecar_paths = [str(receipt["path"]) for receipt in sidecars.values()]
    for report in reports:
        fixture = report["fixture"]
        hidden = {
            "expected_case": fixture["expected_case"],
            "asp_program": report["asp_program"],
            "answer_sets": report["answer_sets"],
            "solver_receipt": report["solver_receipt"],
        }
        for row in audit_prompt_nonexposure(public_prompt_view(fixture), hidden, sidecar_paths):
            nonexposure.append({"fixture_id": fixture["fixture_id"], **row})
    family_counts = dict(sorted(Counter(row["family"] for row in fixture_rows).items()))
    case_counts = Counter((row["family"], row["expected_case"]) for row in fixture_rows)
    preconditions = _preconditions(root, asset_bundle)
    schema_passed = all(_case_passed(report) for report in reports)
    family_floor_passed = all(family_counts.get(family, 0) >= 30 for family in FAMILIES) and all(
        case_counts[(family, case)] >= 6 for family in FAMILIES for case in CASES
    )
    row_hash_count = len(asset_bundle.get("row_hashes", []))
    asset_hashes_passed = (
        bool(asset_bundle.get("hashes_valid"))
        and row_hash_count == ENOKIQA_SHARD_SIZE
        and len(asset_bundle.get("asset_receipts", [])) == 2
        and len(asset_bundle.get("revision_rows", [])) == 2
        and all(row.get("passed") is True for row in asset_bundle.get("revision_rows", []))
    )
    checks = [
        _check("asset_hashes", True, asset_hashes_passed, asset_hashes_passed),
        _check(
            "qualified_exp6274_compiler",
            True,
            preconditions["qualified_exp6274_compiler"]["available"],
            bool(preconditions["qualified_exp6274_compiler"]["available"]),
        ),
        _check(
            "independent_solver_available",
            True,
            preconditions["independent_solver"]["available"],
            bool(preconditions["independent_solver"]["available"]),
        ),
        _check("relation_schema_checks", True, schema_passed, schema_passed),
        _check("family_and_case_floors", True, family_floor_passed, family_floor_passed),
        _check("split_overlap_count", 0, len(overlap), not overlap),
        _check(
            "prompt_and_sidecar_nonexposure",
            True,
            all(row["passed"] for row in nonexposure),
            all(row["passed"] for row in nonexposure),
        ),
        _check("injective_relation_map", 0, len(collisions), not collisions),
        _check(
            "exact_solver_parity",
            0,
            sum(1 for row in solver_rows if row["passed"] is not True),
            all(row["passed"] for row in solver_rows),
        ),
    ]
    summary = gate_summary(checks)
    ready_score = 1 if summary["passed"] else 0
    verdict_class = "circular_positive" if ready_score == 1 else "disqualified"
    honest_verdict = (
        "complete_enoki_exact_relation_fixture_ready_no_accuracy_claim"
        if ready_score == 1
        else "complete_enoki_exact_relation_fixture_disqualified"
    )
    rows: list[JsonDict] = [
        {"row_type": "asset", **row} for row in asset_bundle.get("asset_receipts", [])
    ]
    rows.extend(_fixture_row(report) for report in reports)
    rows.extend({"row_type": "relation", **row} for row in relation_rows)
    rows.extend({"row_type": "exact_check", **row} for row in solver_rows)
    artifact: JsonDict = {
        "schema": "carnot.exp6886.enoki_exact_relation_fixture.v1",
        "experiment_id": 6886,
        "run_date": date,
        "status": "complete" if ready_score == 1 else "disqualified",
        "claim_boundary": {
            "encoder_loaded": False,
            "llm_inference_count": 0,
            "enoki_accuracy_claimed": False,
            "enokiqa_public_split_is_unannotated": True,
        },
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": _source_artifact_hashes(),
        "enoki_asset_receipts": [dict(row) for row in asset_bundle["asset_receipts"]],
        "asset_revision_rows": [dict(row) for row in asset_bundle["revision_rows"]],
        "asset_license_rows": [dict(row) for row in asset_bundle["license_rows"]],
        "relation_schema_version": RELATION_SCHEMA_VERSION,
        "closed_vocabulary_manifest": {
            "schema": "carnot.exp6886.closed_relation_vocabulary.v1",
            "supported_asp_subset": "Exp6274 propositional atoms with explicit polarity atoms",
            "predicates": sorted({row["predicate"] for row in vocabulary}),
            "entry_count": len(vocabulary),
            "entries_hash": sha256_json(vocabulary),
            "entries": vocabulary,
            "injective": not collisions,
        },
        "rows": rows,
        "fixture_family_counts": family_counts,
        "calibration_group_manifest": calibration,
        "sealed_held_group_manifest": held,
        "split_overlap_count": len(overlap),
        "relation_to_atom_rows": relation_rows,
        "atom_collision_rows": collisions,
        "unsupported_rows": unsupported,
        "solver_parity_rows": solver_rows,
        "rule_violation_receipts": violation_rows,
        "prompt_nonexposure_results": nonexposure,
        "independent_solver_receipts": {
            "name_version": asp_energy.solver_name_version(),
            "fixture_calls": len(reports),
            "timeout_count": sum(row["solver_error"] is not None for row in solver_rows),
            "disagreement_count": sum(row["semantic_parity"] is not True for row in solver_rows),
            "sidecar_hashes": {key: value["sha256"] for key, value in sidecars.items()},
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "relation_fixture_ready_score": ready_score,
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable artifact content while excluding elapsed wall time."""

    stable = deepcopy(dict(artifact))
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject summary drift, positive overclaims, and incomplete principles."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"required_fields:{','.join(missing)}")
    principles = artifact.get("field_principles", {})
    if not isinstance(principles, Mapping) or not set(REQUIRED_ARTIFACT_FIELDS) <= set(principles):
        raise ValueError("field_principles")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    if artifact.get("verdict_class") not in {
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        raise ValueError("verdict_class")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        raise ValueError("honest_verdict")
    summary = artifact.get("gate_check_summary", {})
    expected_score = 1 if summary.get("passed") is True else 0
    if artifact.get("relation_fixture_ready_score") != expected_score:
        raise ValueError("ready_score")
    if expected_score == 1 and artifact.get("verdict_class") != "circular_positive":
        raise ValueError("verdict_class")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum")


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_PATH,
    cache_root: Path | str = DEFAULT_CACHE_ROOT,
    asset_loader: Callable[[Path], Mapping[str, Any]] = prepare_enoki_assets,
    write: bool = True,
) -> JsonDict:
    """Run preconditions, build the fixture, and optionally write one result."""

    started = time.perf_counter()
    root = Path(cache_root)
    try:
        bundle = asset_loader(root)
        artifact = build_artifact(
            date=date,
            cache_root=root,
            asset_bundle=bundle,
            duration_s=time.perf_counter() - started,
        )
    except AssetUnavailable as exc:
        artifact = _blocked_artifact(
            date=date,
            duration_s=time.perf_counter() - started,
            failed_check="enoki_assets_available",
            observed=str(exc),
        )
    if write:
        path = Path(result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the required date and execute the deterministic builder."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    artifact = run(date=args.date)
    print(json.dumps({"honest_verdict": artifact["honest_verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - wrapper is the supported entry point
    raise SystemExit(main())
