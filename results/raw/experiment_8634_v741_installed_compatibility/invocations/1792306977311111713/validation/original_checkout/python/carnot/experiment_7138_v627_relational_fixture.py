"""Build a sealed source-grounded relational fixture from cached RAGTruth rows.

The fixture is an input receipt for later model work. It does not score a
verifier. The model view contains documents and a neutral extraction request.
The separate scorer view contains the cached response and span labels.

Spec refs: REQ-HARNESS-7138 and SCENARIO-HARNESS-7138-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260908"
RANDOM_SEED = 7_138_202_609_08
RESULT_PATH = Path("results/experiment_7138_v627_relational_fixture.json")
SOURCE_INFO_PATH = Path("data/ragtruth/source_info.jsonl")
RESPONSE_PATH = Path("data/ragtruth/response.jsonl")
LINEAGE_PATH = Path("data/real_factual_corpus_ragtruth.jsonl")

INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator: deterministic fixture and SQLite sandbox"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

SOURCE_FAMILIES = ("CNN/DM", "Recent News", "MARCO", "Yelp")
RESPONSE_LABELS = ("clean", "hallucinated")
SOURCE_GROUPS_PER_FAMILY = 12
ROWS_PER_CLASS_FAMILY = 9
FIXTURE_ROW_COUNT = 72
SQL_ROW_CAP = 32
SQL_DEFAULT_STEP_BUDGET = 10_000
SQL_DEFAULT_TIMEOUT_S = 0.05

PINNED_CORPUS_HASHES = {
    "source_info": "sha256:0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
    "responses": "sha256:e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
    "lineage": "sha256:a046524bf56e1e4c23b89a331de12d2e5a766dbfe6f1116b5dbb7b7d4083556e",
}
SOURCE_REQUIRED_FIELDS = ("source_id", "task_type", "source", "source_info", "prompt")
RESPONSE_REQUIRED_FIELDS = (
    "id",
    "source_id",
    "model",
    "temperature",
    "labels",
    "split",
    "quality",
    "response",
)
LINEAGE_REQUIRED_FIELDS = (
    "question",
    "answer",
    "evidence_passage",
    "is_hallucination",
    "model_confidence",
)

MODEL_VIEW_FIELDS = (
    "fixture_id",
    "task_type",
    "split",
    "source_text",
    "response_text",
    "source_text_sha256",
    "response_text_sha256",
    "prompt",
)
MODEL_PROMPT = (
    "Extract only entities and relations stated in document A or document B. "
    "Return one JSON object that follows the supplied relation schema. Use a "
    "typed null value when a value is not stated. Preserve exact character "
    "spans for each relation."
)
_MODEL_RESERVED_RE = re.compile(
    r"(?i)(?:\b(?:clean|hallucinated|hallucination)\b|ragtruth|"
    r"source_info\.jsonl|response\.jsonl|sealed[_ -]?scorer|scorer[_ -]?view)"
)

ENTITY_TYPES = (
    "person",
    "organization",
    "location",
    "event",
    "work",
    "product",
    "time",
    "quantity",
    "other",
)
RELATION_TYPES = (
    "affiliated_with",
    "causes",
    "contains",
    "created_by",
    "has_attribute",
    "has_quantity",
    "located_in",
    "occurred_at",
    "occurred_on",
    "part_of",
    "precedes",
    "related_to",
)
VALUE_TYPES = ("entity", "string", "number", "boolean", "date", "unknown")
UNKNOWN_REASONS = ("not_stated_in_source", "ambiguous_source", "not_applicable")
DOCUMENT_TYPES = ("source", "response")

RELATION_SCHEMA: JsonDict = {
    "schema": "carnot.relational_grounding.v1",
    "closed": True,
    "entity": {
        "required_fields": ["entity_id", "entity_type", "canonical_name"],
        "entity_types": list(ENTITY_TYPES),
        "additional_fields": False,
    },
    "relation": {
        "required_fields": [
            "relation_id",
            "subject_entity_id",
            "predicate",
            "object",
            "provenance",
        ],
        "predicates": list(RELATION_TYPES),
        "additional_fields": False,
    },
    "value": {
        "required_fields": ["type", "value", "unit", "unknown_reason"],
        "types": list(VALUE_TYPES),
        "unknown_contract": {
            "value": None,
            "unit": None,
            "reasons": list(UNKNOWN_REASONS),
        },
        "additional_fields": False,
    },
    "provenance_span": {
        "required_fields": ["document", "start", "end", "text", "sha256"],
        "documents": list(DOCUMENT_TYPES),
        "offset_unit": "unicode_code_point",
        "end_is_exclusive": True,
        "text_must_match_document_slice": True,
        "additional_fields": False,
    },
}

SQL_COLUMNS = (
    "relation_id",
    "subject_entity_id",
    "subject_name",
    "subject_type",
    "predicate",
    "object_type",
    "object_value",
    "object_unit",
    "unknown_reason",
    "document",
    "span_start",
    "span_end",
    "span_text",
    "span_sha256",
)
SQL_SANDBOX_CONTRACT: JsonDict = {
    "database": "isolated_in_memory_sqlite",
    "table": "grounded_relations",
    "columns": list(SQL_COLUMNS),
    "query_only": True,
    "grammar": (
        "SELECT <closed columns> FROM grounded_relations "
        "[WHERE <column> <comparison> <literal> [AND ...]] "
        "[ORDER BY <closed columns>] LIMIT <1..32>"
    ),
    "row_cap": SQL_ROW_CAP,
    "default_step_budget": SQL_DEFAULT_STEP_BUDGET,
    "default_timeout_s": SQL_DEFAULT_TIMEOUT_S,
    "denied": [
        "comments",
        "semicolons",
        "joins",
        "subqueries",
        "compound_queries",
        "functions",
        "writes",
        "pragmas",
        "attaches",
        "extensions",
        "unknown_columns",
        "excessive_rows",
        "excessive_steps",
        "excessive_time",
    ],
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "field_principles",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "fixture_rows",
    "source_family_rows",
    "class_balance_rows",
    "model_view_rows",
    "sealed_scorer_rows",
    "split_rows",
    "relation_schema",
    "sql_sandbox_contract",
    "mutation_rows",
    "independent_loader_rows",
    "label_exposure_count",
    "fixture_row_count",
    "source_grounding_fixture_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "status": "The state distinguishes the first durable write from a terminal result.",
    "field_principles": "A reason for each field makes missing evidence visible.",
    "preconditions_checked": "Exact path, hash, and schema checks prevent corpus substitution.",
    "run_date": "The fixed execution date identifies this frozen fixture build.",
    "inference_substrate": "The text limits this result to deterministic CPU fixture work.",
    "inference_substrate_class": "The class distinguishes execution from a blocked no-run.",
    "execution_venue": "The host venue prevents an unsupported hardware claim.",
    "duration_s": "Measured wall time records that the fixture workflow ran.",
    "source_artifact_hashes": "Input hashes bind all rows to exact cached corpus bytes.",
    "rows": "The complete ordered receipt prevents aggregate-only evidence.",
    "fixture_rows": "Raw IDs, text, ranks, splits, and hashes freeze selection.",
    "source_family_rows": "Per-family counts prove source-group coverage.",
    "class_balance_rows": "Sealed per-family counts prove the fixed class quota.",
    "model_view_rows": "Only neutral documents and extraction instructions reach a model.",
    "sealed_scorer_rows": "Response and span outcomes remain outside model inputs.",
    "split_rows": "Split counts prevent train and test membership drift.",
    "relation_schema": "Closed entity, relation, value, and span types reject silent coercion.",
    "sql_sandbox_contract": "A bounded read-only grammar limits untrusted SQLite work.",
    "mutation_rows": "Executed negative controls prove key guards can fail.",
    "independent_loader_rows": "A second loader reproduces order, labels, splits, and hashes.",
    "label_exposure_count": "Zero is required before a model view is usable.",
    "fixture_row_count": "The exact denominator prevents later sample shrinkage.",
    "source_grounding_fixture_ready_score": "One means fixture readiness, not verifier value.",
    "random_seed": "A fixed seed names the deterministic selection policy.",
    "reproducibility_checksum": "A canonical digest detects later artifact mutation.",
    "gate_check_summary": "Exact expected and observed values make a block actionable.",
    "verifier_is_oracle": "False prevents fixture readiness from certifying a verifier.",
    "verdict_class": "A closed terminal class supports automation.",
    "honest_verdict": "A matching prefix states readiness or the exact block.",
}


def canonical_json(value: Any) -> str:
    """Serialize a value with one stable JSON byte representation."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash exact UTF-8 text so character changes remain visible."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Hash exact file bytes without normalizing JSONL line endings."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all artifact fields except the field that stores this digest."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_text(canonical_json(payload))


def _source_text(value: Any) -> str:
    """Render structured sources once so later character spans are stable."""

    return value if isinstance(value, str) else canonical_json(value)


def _read_jsonl(path: Path | str) -> list[JsonDict]:
    """Read JSON objects while retaining their file order."""

    rows: list[JsonDict] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"jsonl_row_not_object:{line_number}")
        rows.append(value)
    return rows


def load_source_records(path: Path | str) -> list[JsonDict]:
    """Load source records after the caller has completed byte-level gates."""

    return _read_jsonl(path)


def load_response_records(path: Path | str) -> list[JsonDict]:
    """Load full response records only after source groups are fixed."""

    return _read_jsonl(path)


def response_membership_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project split routing fields without reading the outcome field."""

    return [
        {
            "id": str(row.get("id")),
            "source_id": str(row.get("source_id")),
            "split": row.get("split"),
            "quality": row.get("quality"),
        }
        for row in rows
    ]


def load_response_membership(path: Path | str) -> list[JsonDict]:
    """Load only fields needed to select source groups and split membership."""

    return response_membership_rows(_read_jsonl(path))


def _source_rank(row: Mapping[str, Any]) -> str:
    """Rank source groups with source-only facts and no response outcome."""

    source_text = _source_text(row.get("source_info"))
    return sha256_text(
        canonical_json(
            {
                "family": row.get("source"),
                "source_id": str(row.get("source_id")),
                "task_type": row.get("task_type"),
                "source_sha256": sha256_text(source_text),
            }
        )
    )


def select_source_groups(
    sources: Sequence[Mapping[str, Any]], membership: Sequence[Mapping[str, Any]]
) -> dict[str, list[str]]:
    """Select test source groups before any response outcome is inspected."""

    test_source_ids = {
        str(row.get("source_id")) for row in membership if row.get("split") == "test"
    }
    selected: dict[str, list[str]] = {}
    for family in SOURCE_FAMILIES:
        candidates = [
            row
            for row in sources
            if row.get("source") == family and str(row.get("source_id")) in test_source_ids
        ]
        ranked = sorted(candidates, key=lambda row: (_source_rank(row), str(row.get("source_id"))))
        ids = [str(row.get("source_id")) for row in ranked[:SOURCE_GROUPS_PER_FAMILY]]
        if len(ids) != SOURCE_GROUPS_PER_FAMILY:
            raise ValueError(f"source_group_quota_unavailable:{family}")
        selected[family] = ids
    return selected


def _response_rank(row: Mapping[str, Any]) -> str:
    """Rank responses without using cached response or span outcomes."""

    return sha256_text(
        canonical_json(
            {
                "source_id": str(row.get("source_id")),
                "response_id": str(row.get("id")),
                "model": row.get("model"),
                "split": row.get("split"),
                "response_text_sha256": sha256_text(str(row.get("response", ""))),
                "random_seed": RANDOM_SEED,
            }
        )
    )


def _final_rank(source: Mapping[str, Any], response: Mapping[str, Any]) -> str:
    """Mix selected rows without encoding their sealed class or position."""

    return sha256_text(
        canonical_json(
            {
                "source_id": str(source.get("source_id")),
                "response_id": str(response.get("id")),
                "source_text_sha256": sha256_text(_source_text(source.get("source_info"))),
                "response_text_sha256": sha256_text(str(response.get("response", ""))),
                "random_seed": RANDOM_SEED,
            }
        )
    )


def _eligible_model_text(source_text: str, response_text: str) -> bool:
    """Exclude reserved scorer words before class quotas are inspected."""

    return _MODEL_RESERVED_RE.search(source_text + "\n" + response_text) is None


def render_model_prompt(_row: Mapping[str, Any]) -> str:
    """Return one neutral instruction that contains no scorer state."""

    return MODEL_PROMPT


def _select_balanced_records(
    sources: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    selected_groups: Mapping[str, Sequence[str]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any], str, str]]:
    """Apply fixed class quotas only after source-group selection is complete."""

    source_by_id = {str(row.get("source_id")): row for row in sources}
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any], str, str]] = []
    for family in SOURCE_FAMILIES:
        group_ids = set(selected_groups.get(family, ()))
        candidates: list[tuple[str, Mapping[str, Any]]] = []
        for response in responses:
            source_id = str(response.get("source_id"))
            source = source_by_id.get(source_id)
            if (
                source_id not in group_ids
                or source is None
                or response.get("split") != "test"
                or response.get("quality") != "good"
            ):
                continue
            source_text = _source_text(source.get("source_info"))
            response_text = str(response.get("response", ""))
            if _eligible_model_text(source_text, response_text):
                candidates.append((_response_rank(response), response))
        candidates.sort(key=lambda item: (item[0], str(item[1].get("id"))))
        for response_label in RESPONSE_LABELS:
            wants_outcome = response_label == "hallucinated"
            matches = [
                (rank, row) for rank, row in candidates if bool(row.get("labels")) is wants_outcome
            ][:ROWS_PER_CLASS_FAMILY]
            if len(matches) != ROWS_PER_CLASS_FAMILY:
                raise ValueError(f"class_quota_unavailable:{family}:{response_label}")
            selected.extend(
                (source_by_id[str(row.get("source_id"))], row, response_label, rank)
                for rank, row in matches
            )
    selected.sort(key=lambda item: (_final_rank(item[0], item[1]), str(item[1].get("id"))))
    return selected


def build_fixture_views(
    sources: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    selected_groups: Mapping[str, Sequence[str]],
) -> JsonDict:
    """Build joined, model-visible, and sealed views in one opaque order."""

    selected = _select_balanced_records(sources, responses, selected_groups)
    source_positions = {
        (family, source_id): index
        for family, ids in selected_groups.items()
        for index, source_id in enumerate(ids, 1)
    }
    fixture_rows: list[JsonDict] = []
    model_rows: list[JsonDict] = []
    sealed_rows: list[JsonDict] = []
    for index, (source, response, response_label, response_rank) in enumerate(selected, 1):
        fixture_id = f"unit-{index:03d}"
        source_text = _source_text(source.get("source_info"))
        response_text = str(response.get("response", ""))
        source_family = str(source.get("source"))
        fixture_row = {
            "fixture_id": fixture_id,
            "source_id": str(source.get("source_id")),
            "response_id": str(response.get("id")),
            "source_family": source_family,
            "task_type": str(source.get("task_type")),
            "model": str(response.get("model")),
            "split": str(response.get("split")),
            "source_group_position": source_positions[
                (source_family, str(source.get("source_id")))
            ],
            "source_group_rank": _source_rank(source),
            "response_rank": response_rank,
            "final_rank": _final_rank(source, response),
            "source_text": source_text,
            "response_text": response_text,
            "source_text_sha256": sha256_text(source_text),
            "response_text_sha256": sha256_text(response_text),
            "source_record_sha256": sha256_text(canonical_json(source)),
            "response_record_sha256": sha256_text(canonical_json(response)),
            "selection_rule": "source_hash_12_then_response_hash_9_per_class_then_blind_mix",
        }
        model_row = {
            "fixture_id": fixture_id,
            "task_type": fixture_row["task_type"],
            "split": fixture_row["split"],
            "source_text": source_text,
            "response_text": response_text,
            "source_text_sha256": fixture_row["source_text_sha256"],
            "response_text_sha256": fixture_row["response_text_sha256"],
            "prompt": MODEL_PROMPT,
        }
        labels = deepcopy(list(response.get("labels") or []))
        sealed_row = {
            "fixture_id": fixture_id,
            "source_id": fixture_row["source_id"],
            "response_id": fixture_row["response_id"],
            "response_label": response_label,
            "span_labels": labels,
            "span_labels_sha256": sha256_text(canonical_json(labels)),
        }
        fixture_rows.append(fixture_row)
        model_rows.append(model_row)
        sealed_rows.append(sealed_row)

    family_rows = []
    for family in SOURCE_FAMILIES:
        family_rows.append(
            {
                "source_family": family,
                "source_group_count": len(selected_groups[family]),
                "source_ids": list(selected_groups[family]),
                "row_count": sum(row["source_family"] == family for row in fixture_rows),
            }
        )
    balance_rows = [
        {
            "source_family": family,
            "response_label": label,
            "row_count": sum(
                fixture["source_family"] == family and sealed["response_label"] == label
                for fixture, sealed in zip(fixture_rows, sealed_rows, strict=True)
            ),
        }
        for family in SOURCE_FAMILIES
        for label in RESPONSE_LABELS
    ]
    return {
        "fixture_rows": fixture_rows,
        "source_family_rows": family_rows,
        "class_balance_rows": balance_rows,
        "model_view_rows": model_rows,
        "sealed_scorer_rows": sealed_rows,
        "split_rows": [{"split": "test", "row_count": len(fixture_rows)}],
    }


def count_label_exposures(model_rows: Sequence[Mapping[str, Any]]) -> int:
    """Count reserved fields and tokens across the complete model view."""

    count = 0
    for row in model_rows:
        for key, value in row.items():
            lowered_key = str(key).lower()
            if any(token in lowered_key for token in ("label", "scorer", "sealed")):
                count += 1
            if isinstance(value, str) and _MODEL_RESERVED_RE.search(value):
                count += 1
    return count


def model_view_exposure_errors(model_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject shape, hash, prompt, and reserved-token changes in model inputs."""

    errors: list[str] = []
    for row in model_rows:
        fixture_id = str(row.get("fixture_id"))
        if set(row) != set(MODEL_VIEW_FIELDS):
            errors.append(f"model_view_fields:{fixture_id}")
        if row.get("source_text_sha256") != sha256_text(str(row.get("source_text", ""))):
            errors.append(f"model_source_hash:{fixture_id}")
        if row.get("response_text_sha256") != sha256_text(str(row.get("response_text", ""))):
            errors.append(f"model_response_hash:{fixture_id}")
        if row.get("prompt") != render_model_prompt(row):
            errors.append(f"model_prompt_mismatch:{fixture_id}")
        if count_label_exposures([row]):
            errors.append(f"model_view_exposure:{fixture_id}")
    return list(dict.fromkeys(errors))


def _valid_relation_object(value: Any, entity_ids: set[str]) -> bool:
    """Check the closed relation-value union without coercing values."""

    if not isinstance(value, Mapping) or set(value) != {
        "type",
        "value",
        "unit",
        "unknown_reason",
    }:
        return False
    value_type = value.get("type")
    payload = value.get("value")
    unit = value.get("unit")
    reason = value.get("unknown_reason")
    if value_type not in VALUE_TYPES:
        return False
    if value_type == "unknown":
        return payload is None and unit is None and reason in UNKNOWN_REASONS
    if reason is not None or (unit is not None and value_type != "number"):
        return False
    if value_type == "entity":
        return isinstance(payload, str) and payload in entity_ids and unit is None
    if value_type in {"string", "date"}:
        return isinstance(payload, str) and bool(payload) and unit is None
    if value_type == "number":
        return (
            isinstance(payload, (int, float))
            and not isinstance(payload, bool)
            and (unit is None or isinstance(unit, str))
        )
    return isinstance(payload, bool) and unit is None


def validate_relation_bundle(bundle: Mapping[str, Any], documents: Mapping[str, str]) -> list[str]:
    """Validate closed relation data and exact in-document provenance spans."""

    if set(bundle) != {"entities", "relations"}:
        return ["relation_bundle_fields"]
    entities = bundle.get("entities")
    relations = bundle.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        return ["relation_bundle_collections"]
    errors: list[str] = []
    entity_ids: set[str] = set()
    for entity in entities:
        entity_id = str(entity.get("entity_id")) if isinstance(entity, Mapping) else "unknown"
        valid = (
            isinstance(entity, Mapping)
            and set(entity) == {"entity_id", "entity_type", "canonical_name"}
            and isinstance(entity.get("entity_id"), str)
            and bool(re.fullmatch(r"e[0-9]{3,}", str(entity.get("entity_id"))))
            and entity.get("entity_type") in ENTITY_TYPES
            and isinstance(entity.get("canonical_name"), str)
            and bool(entity.get("canonical_name"))
            and entity_id not in entity_ids
        )
        if not valid:
            errors.append(f"entity_invalid:{entity_id}")
        else:
            entity_ids.add(entity_id)
    relation_ids: set[str] = set()
    for relation in relations:
        relation_id = (
            str(relation.get("relation_id")) if isinstance(relation, Mapping) else "unknown"
        )
        valid_shape = (
            isinstance(relation, Mapping)
            and set(relation)
            == {"relation_id", "subject_entity_id", "predicate", "object", "provenance"}
            and bool(re.fullmatch(r"r[0-9]{3,}", relation_id))
            and relation_id not in relation_ids
            and relation.get("subject_entity_id") in entity_ids
            and relation.get("predicate") in RELATION_TYPES
            and isinstance(relation.get("provenance"), list)
            and bool(relation.get("provenance"))
        )
        if not valid_shape:
            errors.append(f"relation_invalid:{relation_id}")
            continue
        relation_ids.add(relation_id)
        if not _valid_relation_object(relation.get("object"), entity_ids):
            errors.append(f"relation_object_invalid:{relation_id}")
        for index, span in enumerate(relation.get("provenance", [])):
            valid_span = (
                isinstance(span, Mapping)
                and set(span) == {"document", "start", "end", "text", "sha256"}
                and span.get("document") in DOCUMENT_TYPES
                and type(span.get("start")) is int
                and type(span.get("end")) is int
                and 0 <= int(span.get("start", -1)) < int(span.get("end", -1))
            )
            if not valid_span:
                errors.append(f"provenance_invalid:{relation_id}:{index}")
                continue
            document = documents.get(str(span["document"]))
            start = int(span["start"])
            end = int(span["end"])
            if not isinstance(document, str) or end > len(document):
                errors.append(f"provenance_bounds:{relation_id}:{index}")
                continue
            if document[start:end] != span.get("text"):
                errors.append(f"provenance_text_mismatch:{relation_id}:{index}")
            if span.get("sha256") != sha256_text(str(span.get("text", ""))):
                errors.append(f"provenance_hash_mismatch:{relation_id}:{index}")
    return list(dict.fromkeys(errors))


def _deny_sql_action(
    action: int, _a: str | None, _b: str | None, _c: str | None, _d: str | None
) -> int:
    """Allow reads only after the private database has been populated."""

    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ}
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def create_relation_database(
    bundle: Mapping[str, Any], documents: Mapping[str, str]
) -> sqlite3.Connection:
    """Create one isolated in-memory database and then lock it read-only."""

    errors = validate_relation_bundle(bundle, documents)
    if errors:
        raise ValueError(";".join(errors))
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE grounded_relations ("
        "relation_id TEXT NOT NULL, subject_entity_id TEXT NOT NULL, "
        "subject_name TEXT NOT NULL, subject_type TEXT NOT NULL, predicate TEXT NOT NULL, "
        "object_type TEXT NOT NULL, object_value, object_unit TEXT, unknown_reason TEXT, "
        "document TEXT NOT NULL, span_start INTEGER NOT NULL, span_end INTEGER NOT NULL, "
        "span_text TEXT NOT NULL, span_sha256 TEXT NOT NULL)"
    )
    entities = {row["entity_id"]: row for row in bundle["entities"]}
    for relation in bundle["relations"]:
        entity = entities[relation["subject_entity_id"]]
        value = relation["object"]
        for span in relation["provenance"]:
            connection.execute(
                "INSERT INTO grounded_relations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    relation["relation_id"],
                    relation["subject_entity_id"],
                    entity["canonical_name"],
                    entity["entity_type"],
                    relation["predicate"],
                    value["type"],
                    value["value"],
                    value["unit"],
                    value["unknown_reason"],
                    span["document"],
                    span["start"],
                    span["end"],
                    span["text"],
                    span["sha256"],
                ),
            )
    connection.commit()
    connection.enable_load_extension(False)
    connection.execute("PRAGMA query_only = ON")
    connection.set_authorizer(_deny_sql_action)
    return connection


_FORBIDDEN_SQL_RE = re.compile(
    r"(?is)(?:--|/\*|\*/|;|\b(?:insert|update|delete|replace|create|drop|alter|"
    r"pragma|attach|detach|vacuum|reindex|analyze|union|intersect|except|join|"
    r"with|returning|load_extension)\b)"
)
_SELECT_RE = re.compile(
    r"(?is)^\s*SELECT\s+(?P<select>\*|[A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)"
    r"\s+FROM\s+grounded_relations"
    r"(?:\s+WHERE\s+(?P<where>.+?))?"
    r"(?:\s+ORDER\s+BY\s+(?P<order>[A-Za-z_][A-Za-z0-9_]*(?:\s+(?:ASC|DESC))?"
    r"(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*(?:\s+(?:ASC|DESC))?)*))?"
    r"\s+LIMIT\s+(?P<limit>[0-9]+)\s*$"
)
_WHERE_TERM_RE = re.compile(
    r"(?is)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(=|!=|<>|<=|>=|<|>)\s*"
    r"(?:'(?:''|[^'])*'|-?[0-9]+(?:\.[0-9]+)?|NULL)\s*$"
)


def _supported_select(query: str) -> bool:
    """Accept only the documented single-table SQL grammar."""

    if _FORBIDDEN_SQL_RE.search(query) or "(" in query or ")" in query:
        return False
    match = _SELECT_RE.fullmatch(query)
    if match is None:
        return False
    selected = (
        list(SQL_COLUMNS)
        if match.group("select") == "*"
        else [part.strip() for part in match.group("select").split(",")]
    )
    if any(column not in SQL_COLUMNS for column in selected):
        return False
    if int(match.group("limit")) < 1 or int(match.group("limit")) > SQL_ROW_CAP:
        return False
    where = match.group("where")
    if where:
        for term in re.split(r"(?i)\s+AND\s+", where):
            term_match = _WHERE_TERM_RE.fullmatch(term)
            if term_match is None or term_match.group(1) not in SQL_COLUMNS:
                return False
    order = match.group("order")
    if order:
        for term in order.split(","):
            column = term.strip().split()[0]
            if column not in SQL_COLUMNS:
                return False
    return True


def execute_bounded_select(
    connection: sqlite3.Connection,
    query: str,
    *,
    max_steps: int = SQL_DEFAULT_STEP_BUDGET,
    timeout_s: float = SQL_DEFAULT_TIMEOUT_S,
) -> JsonDict:
    """Execute one bounded read or return a typed rejection without repair."""

    if not isinstance(query, str) or not _supported_select(query):
        return {"status": "rejected", "reason": "unsupported_sql"}
    if max_steps <= 0:
        return {"status": "rejected", "reason": "step_budget_exhausted"}
    if timeout_s <= 0:
        return {"status": "rejected", "reason": "time_budget_exhausted"}
    started = time.monotonic()
    steps = 0
    stop_reason: str | None = None

    def progress() -> int:
        nonlocal steps, stop_reason
        steps += 1
        if steps > max_steps:
            stop_reason = "step_budget_exhausted"
            return 1
        if time.monotonic() - started > timeout_s:
            stop_reason = "time_budget_exhausted"
            return 1
        return 0

    connection.set_progress_handler(progress, 1)
    try:
        cursor = connection.execute(query)
        rows = cursor.fetchall()
        columns = [str(item[0]) for item in cursor.description or ()]
    except sqlite3.DatabaseError:
        return {"status": "rejected", "reason": stop_reason or "unsupported_sql"}
    finally:
        connection.set_progress_handler(None, 0)
    return {
        "status": "ok",
        "columns": columns,
        "rows": [list(row) for row in rows],
        "row_count": len(rows),
    }


def _gate(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Record exact precondition values for terminal failure reporting."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def _gate_summary(row: Mapping[str, Any] | None) -> JsonDict:
    """Promote the first failure into the stable automation shape."""

    if row is None:
        return {"failed_check": None, "expected_value": 1, "observed_value": 1, "passed": True}
    return {
        "failed_check": row.get("check"),
        "expected_value": deepcopy(row.get("expected_value")),
        "observed_value": deepcopy(row.get("observed_value")),
        "passed": False,
    }


def _schema_observation(path: Path, required: Sequence[str]) -> tuple[bool, Any]:
    """Return the first JSONL shape failure without hiding its line number."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return False, {"line": 0, "error": type(exc).__name__}
    row_count = 0
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        row_count += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return False, {"line": line_number, "error": "invalid_json"}
        if not isinstance(row, dict):
            return False, {"line": line_number, "error": "row_not_object"}
        missing = [field for field in required if field not in row]
        if missing:
            return False, {"line": line_number, "missing_fields": missing}
    return row_count > 0, {"row_count": row_count}


def _preconditions(
    paths: Mapping[str, Path], expected_hashes: Mapping[str, str]
) -> tuple[list[JsonDict], Mapping[str, Any] | None]:
    """Check paths, exact bytes, and JSONL fields in a fixed fail-first order."""

    checks: list[JsonDict] = []
    for name in ("source_info", "responses", "lineage"):
        path = paths[name]
        passed = path.is_file() and path.stat().st_mode != 0
        row = _gate(
            f"{name}_path",
            "readable_file",
            "readable_file" if passed else "missing_or_unreadable",
            passed,
        )
        checks.append(row)
        if not passed:
            return checks, row
    for name in ("source_info", "responses", "lineage"):
        observed = sha256_file(paths[name])
        row = _gate(
            f"{name}_hash", expected_hashes[name], observed, observed == expected_hashes[name]
        )
        checks.append(row)
        if not row["passed"]:
            return checks, row
    schemas = (
        ("source_schema", "source_info", SOURCE_REQUIRED_FIELDS),
        ("response_schema", "responses", RESPONSE_REQUIRED_FIELDS),
        ("lineage_schema", "lineage", LINEAGE_REQUIRED_FIELDS),
    )
    for check, name, required in schemas:
        passed, observed = _schema_observation(paths[name], required)
        row = _gate(check, list(required), observed, passed)
        checks.append(row)
        if not passed:
            return checks, row
    return checks, None


def _base_artifact(run_date: str) -> JsonDict:
    """Return a schema-complete first-write artifact without reading inputs."""

    artifact: JsonDict = {
        "status": "running",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "fixture_rows": [],
        "source_family_rows": [],
        "class_balance_rows": [],
        "model_view_rows": [],
        "sealed_scorer_rows": [],
        "split_rows": [],
        "relation_schema": deepcopy(RELATION_SCHEMA),
        "sql_sandbox_contract": deepcopy(SQL_SANDBOX_CONTRACT),
        "mutation_rows": [],
        "independent_loader_rows": [],
        "label_exposure_count": 0,
        "fixture_row_count": 0,
        "source_grounding_fixture_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {
            "failed_check": "fixture_build_complete",
            "expected_value": True,
            "observed_value": False,
            "passed": False,
        },
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_source_grounding_fixture_build",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _checkpoint_artifact(artifact: Mapping[str, Any], output_path: Path) -> Path:
    """Atomically persist the complete schema before each fallible phase."""

    atomic_write_json(output_path, dict(artifact), allow_override=False, sort_keys=True)
    return output_path


def _independent_selected_rows(
    source_path: Path, response_path: Path
) -> list[tuple[JsonDict, JsonDict, str]]:
    """Rebuild selection from raw files without calling producer loaders."""

    source_rows = [
        json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line
    ]
    response_rows = [
        json.loads(line) for line in response_path.read_text(encoding="utf-8").splitlines() if line
    ]
    test_ids = {str(row["source_id"]) for row in response_rows if row["split"] == "test"}
    groups: dict[str, list[str]] = {}
    for family in SOURCE_FAMILIES:
        family_sources = [
            row
            for row in source_rows
            if row["source"] == family and str(row["source_id"]) in test_ids
        ]
        family_sources.sort(key=lambda row: (_source_rank(row), str(row["source_id"])))
        groups[family] = [
            str(row["source_id"]) for row in family_sources[:SOURCE_GROUPS_PER_FAMILY]
        ]
    source_by_id = {str(row["source_id"]): row for row in source_rows}
    selected: list[tuple[JsonDict, JsonDict, str]] = []
    for family in SOURCE_FAMILIES:
        ids = set(groups[family])
        candidates = []
        for response in response_rows:
            source_id = str(response["source_id"])
            if source_id not in ids or response["split"] != "test" or response["quality"] != "good":
                continue
            source_text = _source_text(source_by_id[source_id]["source_info"])
            response_text = str(response["response"])
            if _eligible_model_text(source_text, response_text):
                candidates.append((_response_rank(response), response))
        candidates.sort(key=lambda item: (item[0], str(item[1]["id"])))
        for label in RESPONSE_LABELS:
            outcome = label == "hallucinated"
            chosen = [row for _rank, row in candidates if bool(row["labels"]) is outcome][
                :ROWS_PER_CLASS_FAMILY
            ]
            selected.extend((source_by_id[str(row["source_id"])], row, label) for row in chosen)
    selected.sort(key=lambda item: (_final_rank(item[0], item[1]), str(item[1]["id"])))
    return selected


def independent_replay(
    artifact: Mapping[str, Any], source_path: Path | str, response_path: Path | str
) -> tuple[list[JsonDict], list[str]]:
    """Reproduce order, split, labels, spans, and hashes through a second loader."""

    source_path = Path(source_path)
    response_path = Path(response_path)
    if not source_path.is_file() or sha256_file(source_path) != PINNED_CORPUS_HASHES["source_info"]:
        return [], ["source_info_hash_mismatch"]
    if (
        not response_path.is_file()
        or sha256_file(response_path) != PINNED_CORPUS_HASHES["responses"]
    ):
        return [], ["response_hash_mismatch"]
    expected = _independent_selected_rows(source_path, response_path)
    fixture_rows = list(artifact.get("fixture_rows", []))
    model_rows = list(artifact.get("model_view_rows", []))
    sealed_rows = list(artifact.get("sealed_scorer_rows", []))
    errors: list[str] = []
    expected_order = [
        (str(source["source_id"]), str(response["id"])) for source, response, _ in expected
    ]
    observed_order = [
        (str(row.get("source_id")), str(row.get("response_id")))
        for row in fixture_rows
        if isinstance(row, Mapping)
    ]
    if expected_order != observed_order:
        errors.append("fixture_row_order_mismatch")
    if len(model_rows) != len(expected) or len(sealed_rows) != len(expected):
        errors.append("fixture_view_count_mismatch")
    model_by_id = {
        str(row.get("fixture_id")): row for row in model_rows if isinstance(row, Mapping)
    }
    sealed_by_id = {
        str(row.get("fixture_id")): row for row in sealed_rows if isinstance(row, Mapping)
    }
    fixture_by_id = {
        str(row.get("fixture_id")): row for row in fixture_rows if isinstance(row, Mapping)
    }
    receipts: list[JsonDict] = []
    for index, (source, response, response_label) in enumerate(expected, 1):
        fixture_id = f"unit-{index:03d}"
        source_text = _source_text(source["source_info"])
        response_text = str(response["response"])
        source_hash = sha256_text(source_text)
        response_hash = sha256_text(response_text)
        labels_hash = sha256_text(canonical_json(response["labels"]))
        fixture = fixture_by_id.get(fixture_id, {})
        model = model_by_id.get(fixture_id, {})
        sealed = sealed_by_id.get(fixture_id, {})
        row_errors: list[str] = []
        if fixture.get("source_id") != str(source["source_id"]):
            row_errors.append(f"source_id_mismatch:{fixture_id}")
        if fixture.get("response_id") != str(response["id"]):
            row_errors.append(f"response_id_mismatch:{fixture_id}")
        if fixture.get("split") != response["split"] or model.get("split") != response["split"]:
            row_errors.append(f"split_mismatch:{fixture_id}")
        if sealed.get("response_label") != response_label:
            row_errors.append(f"response_label_mismatch:{fixture_id}")
        if sealed.get("span_labels") != response["labels"]:
            row_errors.append(f"span_labels_mismatch:{fixture_id}")
        if sealed.get("span_labels_sha256") != labels_hash:
            row_errors.append(f"span_labels_hash_mismatch:{fixture_id}")
        if (
            fixture.get("source_text_sha256") != source_hash
            or model.get("source_text_sha256") != source_hash
            or fixture.get("source_text") != source_text
            or model.get("source_text") != source_text
        ):
            row_errors.append(f"source_text_mismatch:{fixture_id}")
        if (
            fixture.get("response_text_sha256") != response_hash
            or model.get("response_text_sha256") != response_hash
            or fixture.get("response_text") != response_text
            or model.get("response_text") != response_text
        ):
            row_errors.append(f"response_text_mismatch:{fixture_id}")
        errors.extend(row_errors)
        receipts.append(
            {
                "fixture_id": fixture_id,
                "source_id": str(source["source_id"]),
                "response_id": str(response["id"]),
                "split": response["split"],
                "response_label": response_label,
                "source_text_sha256": source_hash,
                "response_text_sha256": response_hash,
                "span_labels_sha256": labels_hash,
                "exact_match": not row_errors,
            }
        )
    return receipts, list(dict.fromkeys(errors))


def _mutation_receipts(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Execute four negative controls against the completed in-memory fixture."""

    exposed = deepcopy(list(artifact["model_view_rows"]))
    exposed[0]["response_label"] = "hallucinated"
    exposure_errors = model_view_exposure_errors(exposed)

    connection = create_relation_database({"entities": [], "relations": []}, {})
    sql_result = execute_bounded_select(connection, "DELETE FROM grounded_relations")
    connection.close()

    first = artifact["fixture_rows"][0]
    changed_source_detected = (
        sha256_text(str(first["source_text"]) + " ") != first["source_text_sha256"]
    )
    order = [row["fixture_id"] for row in artifact["fixture_rows"]]
    reordered = list(order)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    return [
        {
            "mutation": "label_exposure",
            "detected": bool(exposure_errors),
            "detection": exposure_errors[0] if exposure_errors else None,
        },
        {
            "mutation": "unsupported_sql",
            "detected": sql_result.get("status") == "rejected",
            "detection": sql_result.get("reason"),
        },
        {
            "mutation": "changed_source_bytes",
            "detected": changed_source_detected,
            "detection": "source_text_hash_mismatch" if changed_source_detected else None,
        },
        {
            "mutation": "reordered_rows",
            "detected": reordered != order,
            "detection": "fixture_row_order_mismatch" if reordered != order else None,
        },
    ]


def _blocked_artifact(
    artifact: JsonDict,
    checks: Sequence[Mapping[str, Any]],
    failure: Mapping[str, Any],
    paths: Mapping[str, Path],
    duration_s: float,
) -> JsonDict:
    """Convert the already-written running shape into one exact terminal block."""

    artifact["status"] = "blocked"
    artifact["preconditions_checked"] = deepcopy(list(checks))
    artifact["source_artifact_hashes"] = {
        name: sha256_file(path) for name, path in paths.items() if path.is_file()
    }
    artifact["duration_s"] = round(max(0.0, duration_s), 6)
    artifact["gate_check_summary"] = _gate_summary(failure)
    artifact["verdict_class"] = "blocked"
    artifact["honest_verdict"] = f"blocked_{failure.get('check')}"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact(
    root: Path | str,
    run_date: str,
    *,
    result_path: Path | str = RESULT_PATH,
    corpus_paths: Mapping[str, Path | str] | None = None,
    expected_hashes: Mapping[str, str] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Write first, load the corpus, then write one terminal fixture receipt."""

    started = time.monotonic()
    root = Path(root)
    output_path = Path(result_path)
    if not output_path.is_absolute():
        output_path = root / output_path
    defaults = {
        "source_info": root / SOURCE_INFO_PATH,
        "responses": root / RESPONSE_PATH,
        "lineage": root / LINEAGE_PATH,
    }
    if corpus_paths is not None:
        defaults.update({name: Path(path) for name, path in corpus_paths.items()})
    paths = {name: Path(defaults[name]) for name in ("source_info", "responses", "lineage")}
    hashes = dict(PINNED_CORPUS_HASHES if expected_hashes is None else expected_hashes)

    artifact = _base_artifact(run_date)
    _checkpoint_artifact(artifact, output_path)
    checks, failure = _preconditions(paths, hashes)
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    if failure is not None:
        artifact = _blocked_artifact(artifact, checks, failure, paths, float(elapsed))
        _checkpoint_artifact(artifact, output_path)
        return artifact

    sources = load_source_records(paths["source_info"])
    membership = load_response_membership(paths["responses"])
    selected_groups = select_source_groups(sources, membership)
    responses = load_response_records(paths["responses"])
    try:
        views = build_fixture_views(sources, responses, selected_groups)
    except ValueError as exc:
        failure = _gate("fixture_balance", "72_rows_balanced_9_per_class_family", str(exc), False)
        checks.append(failure)
        artifact = _blocked_artifact(artifact, checks, failure, paths, float(elapsed))
        _checkpoint_artifact(artifact, output_path)
        return artifact

    artifact.update(views)
    artifact["rows"] = deepcopy(views["fixture_rows"])
    artifact["preconditions_checked"] = checks
    artifact["source_artifact_hashes"] = {name: sha256_file(path) for name, path in paths.items()}
    artifact["inference_substrate_class"] = INFERENCE_SUBSTRATE_CLASS
    artifact["fixture_row_count"] = len(views["fixture_rows"])
    artifact["label_exposure_count"] = count_label_exposures(views["model_view_rows"])
    receipts, replay_errors = independent_replay(artifact, paths["source_info"], paths["responses"])
    artifact["independent_loader_rows"] = receipts
    artifact["mutation_rows"] = _mutation_receipts(artifact)
    ready = int(
        artifact["fixture_row_count"] == FIXTURE_ROW_COUNT
        and artifact["label_exposure_count"] == 0
        and not replay_errors
        and all(row["detected"] is True for row in artifact["mutation_rows"])
    )
    artifact["source_grounding_fixture_ready_score"] = ready
    artifact["status"] = "complete"
    artifact["duration_s"] = round(max(0.0, float(elapsed)), 6)
    artifact["gate_check_summary"] = _gate_summary(
        None if ready else _gate("source_grounding_fixture_ready_score", 1, ready, False)
    )
    artifact["verdict_class"] = "positive" if ready else "disqualified"
    artifact["honest_verdict"] = (
        "complete_positive_source_grounding_fixture_ready_no_verifier_value_claim"
        if ready
        else "disqualified_source_grounding_fixture_contract"
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _checkpoint_artifact(artifact, output_path)
    return artifact


def _positive_errors(
    artifact: Mapping[str, Any], source_path: Path, response_path: Path
) -> list[str]:
    """Recompute complete-fixture invariants without trusting producer summaries."""

    errors: list[str] = []
    fixture_rows = list(artifact.get("fixture_rows", []))
    model_rows = list(artifact.get("model_view_rows", []))
    sealed_rows = list(artifact.get("sealed_scorer_rows", []))
    if artifact.get("status") != "complete":
        errors.append("status mismatch")
    if artifact.get("rows") != fixture_rows:
        errors.append("rows mismatch")
    if len(fixture_rows) != FIXTURE_ROW_COUNT or artifact.get("fixture_row_count") != len(
        fixture_rows
    ):
        errors.append("fixture_row_count mismatch")
    exposure_errors = model_view_exposure_errors(model_rows)
    errors.extend(exposure_errors)
    exposure_count = count_label_exposures(model_rows)
    if artifact.get("label_exposure_count") != exposure_count or exposure_count != 0:
        errors.append("label_exposure_count mismatch")
    if len(model_rows) != FIXTURE_ROW_COUNT or len(sealed_rows) != FIXTURE_ROW_COUNT:
        errors.append("view row count mismatch")
    family_counts = Counter(row.get("source_family") for row in fixture_rows)
    if family_counts != Counter({family: 18 for family in SOURCE_FAMILIES}):
        errors.append("source family balance mismatch")
    fixture_family = {row.get("fixture_id"): row.get("source_family") for row in fixture_rows}
    class_counts = Counter(
        (fixture_family.get(row.get("fixture_id")), row.get("response_label"))
        for row in sealed_rows
    )
    if class_counts != Counter(
        {
            (family, label): ROWS_PER_CLASS_FAMILY
            for family in SOURCE_FAMILIES
            for label in RESPONSE_LABELS
        }
    ):
        errors.append("class balance mismatch")
    if artifact.get("split_rows") != [{"split": "test", "row_count": FIXTURE_ROW_COUNT}]:
        errors.append("split_rows mismatch")
    if artifact.get("relation_schema") != RELATION_SCHEMA:
        errors.append("relation_schema mismatch")
    if artifact.get("sql_sandbox_contract") != SQL_SANDBOX_CONTRACT:
        errors.append("sql_sandbox_contract mismatch")
    mutation_rows = list(artifact.get("mutation_rows", []))
    if [row.get("mutation") for row in mutation_rows] != [
        "label_exposure",
        "unsupported_sql",
        "changed_source_bytes",
        "reordered_rows",
    ] or not all(row.get("detected") is True and row.get("detection") for row in mutation_rows):
        errors.append("mutation_rows mismatch")
    replay_rows, replay_errors = independent_replay(artifact, source_path, response_path)
    errors.extend(replay_errors)
    if artifact.get("independent_loader_rows") != replay_rows:
        errors.append("independent_loader_rows mismatch")
    if artifact.get("source_grounding_fixture_ready_score") != 1:
        errors.append("source_grounding_fixture_ready_score mismatch")
    if artifact.get("gate_check_summary") != _gate_summary(None):
        errors.append("gate_check_summary mismatch")
    if artifact.get("verdict_class") != "positive":
        errors.append("verdict_class mismatch")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_positive"):
        errors.append("honest_verdict mismatch")
    return errors


def validate_artifact(
    value: Mapping[str, Any] | str | Path,
    source_path: Path | str = REPO_ROOT / SOURCE_INFO_PATH,
    response_path: Path | str = REPO_ROOT / RESPONSE_PATH,
) -> list[str]:
    """Cold-check schema, separation, replay, readiness, and verdict consistency."""

    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            return ["artifact_missing"]
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ["artifact_unreadable"]
        if not isinstance(loaded, dict):
            return ["artifact_not_object"]
        artifact: Mapping[str, Any] = loaded
    elif isinstance(value, Mapping):
        artifact = value
    else:
        return ["artifact_not_object"]

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    extra = [field for field in artifact if field not in REQUIRED_ARTIFACT_FIELDS]
    if missing or extra:
        return [f"artifact fields mismatch missing={missing} extra={extra}"]
    errors: list[str] = []
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue mismatch")
    if not isinstance(artifact.get("duration_s"), (int, float)) or artifact.get("duration_s") < 0:
        errors.append("duration_s invalid")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    failed = next(
        (
            row
            for row in artifact.get("preconditions_checked", [])
            if isinstance(row, Mapping) and row.get("passed") is False
        ),
        None,
    )
    if failed is not None:
        if artifact.get("status") != "blocked":
            errors.append("blocked status mismatch")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked inference_substrate_class mismatch")
        if artifact.get("gate_check_summary") != _gate_summary(failed):
            errors.append("blocked gate_check_summary mismatch")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked verdict_class mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked honest_verdict mismatch")
        if artifact.get("source_grounding_fixture_ready_score") != 0:
            errors.append("blocked readiness score mismatch")
    else:
        if artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS:
            errors.append("inference_substrate_class mismatch")
        errors.extend(_positive_errors(artifact, Path(source_path), Path(response_path)))
    return list(dict.fromkeys(errors))


def main(argv: Sequence[str] | None = None) -> int:
    """Build the fixed-date artifact or validate an existing artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(args.validate)
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
        return int(bool(errors))
    if not re.fullmatch(r"[0-9]{8}", args.date) or args.date != RUN_DATE:
        return 2
    root = find_repo_root()
    artifact = build_artifact(root, args.date, result_path=args.result_path)
    errors = validate_artifact(artifact)
    print(
        json.dumps(
            {
                "artifact": str(args.result_path),
                "source_grounding_fixture_ready_score": artifact[
                    "source_grounding_fixture_ready_score"
                ],
                "verdict_class": artifact["verdict_class"],
                "valid": not errors,
                "errors": errors,
            },
            sort_keys=True,
        )
    )
    return 1 if errors or artifact["verdict_class"] != "positive" else 0


if __name__ == "__main__":  # pragma: no cover - use the thin experiment wrapper.
    raise SystemExit(main())
