"""Bounded prompt policy for verifier-grounded continual constraint learning.

The model sees only public formulations and policies learned from earlier exact
outcomes. The writer receives a smaller post-outcome object. This separation
prevents confidence text or future labels from becoming update authority.

Spec refs: REQ-LEARN-6978 and SCENARIO-LEARN-6978-CHRONOLOGY/NO-FUTURE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping, Sequence

from carnot.learning.constraint_policy_store import sha256_bytes


JsonDict = dict[str, Any]
ARMS = ("frozen", "read_only", "transactional_write")
FORBIDDEN_FIELDS = {
    "confidence",
    "model_confidence",
    "rationale",
    "future_label",
    "future_labels",
    "later_outcome",
    "later_outcomes",
    "expected_label",
    "held_future_label",
}

CONSTRAINT_IR_SCHEMA: JsonDict = {
    "type": "object",
    "properties": {
        "schema_version": {
            "type": "string",
            "const": "carnot.constraint_ir.mapping.v1",
        },
        "variable_map": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "scale": {"type": "string"},
                    "offset": {"type": "string"},
                },
                "required": ["source", "target", "scale", "offset"],
                "additionalProperties": False,
            },
        },
        "objective_map": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["same", "reversed"]},
                "scale": {"type": "string"},
                "offset": {"type": "string"},
            },
            "required": ["direction", "scale", "offset"],
            "additionalProperties": False,
        },
    },
    "required": ["schema_version", "variable_map", "objective_map"],
    "additionalProperties": False,
}

_POLICY_TEXT = {
    "parse:malformed_json": (
        "Emit one complete JSON object only. Include schema_version, variable_map, "
        "and objective_map. Do not emit prose."
    ),
    "schema:rejected": (
        "Use only source, target, scale, and offset in each variable_map row. "
        "Include each source and target variable exactly once."
    ),
    "domain_correspondence": (
        "Derive each target variable as scale times source plus offset. Check the "
        "bounded domains in both directions."
    ),
    "objective:direction": (
        "Set objective direction to reversed only when the affine substitution "
        "reverses the source objective order."
    ),
    "objective:order": (
        "Substitute every variable map into the target objective. Report exact "
        "rational scale and offset, then preserve weak order and ties."
    ),
    "exact:relation": (
        "Check domain correspondence and the affine objective relation together. "
        "A valid JSON shape alone is not an equivalence certificate."
    ),
}


def _canonical_text(value: Any) -> str:
    """Serialize prompt data without depending on dictionary insertion order."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def find_forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    """Find denied key names at any depth without scanning harmless values."""

    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in FORBIDDEN_FIELDS:
                paths.append(path)
            paths.extend(find_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            paths.extend(find_forbidden_paths(child, path))
    return paths


def visible_predecessors(events: Sequence[Mapping[str, Any]], event_ordinal: int) -> list[JsonDict]:
    """Return only frozen event records that precede the active ordinal."""

    ordered = sorted((deepcopy(dict(row)) for row in events), key=lambda row: row["event_ordinal"])
    expected = list(range(len(ordered)))
    observed = [int(row["event_ordinal"]) for row in ordered]
    if observed != expected:
        raise ValueError("event ordinals are not a complete chronological sequence")
    if event_ordinal < 0 or event_ordinal >= len(ordered):
        raise ValueError("event ordinal is outside the frozen stream")
    return ordered[:event_ordinal]


def initial_memory_records() -> list[JsonDict]:
    """Return the small protected policy shared by read and write controls."""

    return [
        {
            "policy_key": "initial:complete_json",
            "scope": "global",
            "error_class": "initial",
            "policy_text": _POLICY_TEXT["parse:malformed_json"],
            "source_event_id": None,
            "source_event_ordinal": -1,
            "exact_certificate_digest": None,
            "outcome": None,
            "schedule_id": "direct",
            "protected": True,
        },
        {
            "policy_key": "initial:exact_affine_objective",
            "scope": "global",
            "error_class": "initial",
            "policy_text": _POLICY_TEXT["objective:order"],
            "source_event_id": None,
            "source_event_ordinal": -1,
            "exact_certificate_digest": None,
            "outcome": None,
            "schedule_id": "direct",
            "protected": True,
        },
    ]


def lookup_memory(
    records: Sequence[Mapping[str, Any]], *, formulation_family: str, limit: int
) -> list[JsonDict]:
    """Select bounded global or same-family policies in stable newest-first order."""

    eligible = [
        deepcopy(dict(row)) for row in records if row.get("scope") in {"global", formulation_family}
    ]
    eligible.sort(
        key=lambda row: (
            int(row.get("source_event_ordinal", -1)),
            str(row.get("policy_key", "")),
        ),
        reverse=True,
    )
    return eligible[: max(0, int(limit))]


def build_prompt(
    *,
    event: Mapping[str, Any],
    pair: Mapping[str, Any],
    memory_records: Sequence[Mapping[str, Any]],
    schedule_id: str,
    predecessor_ids: Sequence[str] = (),
) -> tuple[str, JsonDict]:
    """Render one direct-schedule prompt from public and predecessor-only data."""

    if schedule_id != "direct":
        raise ValueError("Exp6978 accepts only the selected direct schedule")
    public_pair = {
        "pair_id": pair["pair_id"],
        "source_formulation": deepcopy(pair["source_formulation"]),
        "target_formulation": deepcopy(pair["target_formulation"]),
    }
    policy_rows = [
        {
            "policy_key": row["policy_key"],
            "scope": row["scope"],
            "policy_text": row["policy_text"],
        }
        for row in memory_records
    ]
    visibility = {
        "active_event_id": event["event_id"],
        "active_event_ordinal": event["event_ordinal"],
        "visible_predecessor_ids": list(predecessor_ids),
        "public_pair": public_pair,
        "policy_memory": policy_rows,
    }
    forbidden = find_forbidden_paths(visibility)
    if forbidden:
        raise ValueError(f"forbidden prompt fields:{forbidden}")
    prompt = (
        "Infer one semantic correspondence between the public bounded formulations. "
        "Use rational strings for scale and offset. Emit only one ConstraintIR JSON object.\n"
        f"SCHEDULE=direct constrained from token zero\n"
        f"VISIBLE_PREDECESSOR_IDS={_canonical_text(list(predecessor_ids))}\n"
        f"POLICY_MEMORY={_canonical_text(policy_rows)}\n"
        f"PUBLIC_PAIR={_canonical_text(public_pair)}\n"
        f"CONSTRAINT_IR_JSON_SCHEMA={_canonical_text(CONSTRAINT_IR_SCHEMA)}"
    )
    return prompt, {
        "prompt_hash": sha256_bytes(prompt.encode("utf-8")),
        "forbidden_paths": forbidden,
        "visible_predecessor_ids": list(predecessor_ids),
        "memory_record_keys": [row["policy_key"] for row in policy_rows],
    }


def build_writer_input(
    *,
    atomic_error_class: str,
    exact_certificate_digest: str,
    schedule_metadata: Mapping[str, Any],
    outcome: str,
) -> JsonDict:
    """Build the only evidence object allowed to reach the policy writer."""

    value = {
        "atomic_error_class": str(atomic_error_class),
        "exact_certificate_digest": str(exact_certificate_digest),
        "schedule_metadata": deepcopy(dict(schedule_metadata)),
        "outcome": str(outcome),
    }
    forbidden = find_forbidden_paths(value)
    if forbidden:
        raise ValueError(f"forbidden writer fields:{forbidden}")
    return value


def classify_error(outcome: Mapping[str, Any]) -> str:
    """Reduce exact evaluator fields to one atomic policy error class."""

    if outcome.get("exact_success") is True:
        return "none"
    if outcome.get("parse_success") is False:
        return f"parse:{outcome.get('parse_reason') or 'malformed_json'}"
    if outcome.get("schema_outcome") == "rejected":
        return "schema:rejected"
    if outcome.get("domain_correspondence_outcome") == "failed":
        return "domain_correspondence"
    if outcome.get("objective_direction_outcome") == "failed":
        return "objective:direction"
    if outcome.get("objective_order_outcome") == "failed":
        return "objective:order"
    return "exact:relation"


def build_update_proposal(
    writer_input: Mapping[str, Any],
    *,
    event_id: str,
    event_ordinal: int,
    formulation_family: str,
) -> JsonDict:
    """Convert allowed post-outcome evidence into one bounded policy record."""

    forbidden = find_forbidden_paths(writer_input)
    if forbidden:
        raise ValueError(f"forbidden writer fields:{forbidden}")
    error_class = str(writer_input["atomic_error_class"])
    policy_text = _POLICY_TEXT.get(error_class, _POLICY_TEXT["exact:relation"])
    schedule = dict(writer_input["schedule_metadata"])
    return {
        "policy_key": f"learned:{formulation_family}:{error_class}",
        "scope": formulation_family,
        "error_class": error_class,
        "policy_text": policy_text,
        "source_event_id": event_id,
        "source_event_ordinal": int(event_ordinal),
        "exact_certificate_digest": str(writer_input["exact_certificate_digest"]),
        "outcome": str(writer_input["outcome"]),
        "schedule_id": str(schedule["schedule_id"]),
        "protected": False,
    }


def replay_safety_score(records: Sequence[Mapping[str, Any]]) -> float:
    """Score policies for explicit instructions that erase prior successes."""

    if not records:
        return 1.0
    harmful_terms = ("ignore prior exact", "accept malformed", "skip exact")
    harmful = sum(
        any(term in str(row.get("policy_text", "")).lower() for term in harmful_terms)
        for row in records
    )
    return (len(records) - harmful) / len(records)
