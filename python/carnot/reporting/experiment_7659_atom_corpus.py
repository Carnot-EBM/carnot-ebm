"""Reusable, label-blind atom features for REQ-REPORT-7659.

The input is the original predictor record. A checked line means only that a
visible statement has the named syntax; surrounding prose remains unverified.
"""

from __future__ import annotations

import re
from typing import Any

from carnot.reporting.experiment_7646_source_features import ARMS, validate_model_row
from carnot.verify.tool_source_atoms import digest, parse_atoms, verify_answer


LINE_CLAIM = re.compile(r"`(?P<name>[A-Za-z_]\w*)`[^.]{0,140}?\b(?:at|on) lines? (?P<line>\d+)")


def validate_input(row: dict[str, Any], expected: dict[str, Any], role: str) -> None:
    """Reject changed predictor bytes, role, and any evaluator label field."""

    validate_model_row(row)
    if row["role"] != role or expected["role"] != role:
        raise ValueError("role_mismatch")
    for field in ("component_hash", "source_sha256", "answer_sha256", "learning_partition"):
        if row[field] != expected[field]:
            raise ValueError(f"{field}_mismatch")
    for field in ("complete_source", "complete_answer", "answer_sentences"):
        if row[field] != expected[field]:
            raise ValueError(f"{field}_mismatch")


def _line_witnesses(source: str, answer: str, atoms: list[dict]) -> list[dict]:
    """Check a quoted symbol only against visible definition or raise syntax."""

    found = []
    for claim in LINE_CLAIM.finditer(answer):
        name, number = claim.group("name"), int(claim.group("line"))
        candidates = [atom for atom in atoms if atom["line"] == number]
        structural = [
            atom
            for atom in candidates
            if re.search(rf"\b(?:def|class|raise)\s+{re.escape(name)}\b", atom["text"])
        ]
        lexical = [atom for atom in candidates if name in atom["text"]]
        status = "observed" if structural else "ambiguous" if lexical else "unknown"
        match = (structural or lexical or [None])[0]
        found.append(
            {
                "kind": "line_statement",
                "name": name,
                "line": number,
                "status": status,
                "byte_start": len(answer[: claim.start()].encode()),
                "byte_end": len(answer[: claim.end()].encode()),
                "evidence_span": [match["byte_start"], match["byte_end"]] if match else None,
                "source_complete": bool(match and match["complete"]),
            }
        )
    return found


def extract_group(
    row: dict[str, Any], source: str, arm: str, source_group_id: str | None
) -> dict[str, Any]:
    """Keep one denominator row even when no claim grammar covers its prose."""

    validate_model_row(row)
    if arm not in ARMS:
        raise ValueError("unknown_arm")
    atoms = parse_atoms(source)
    answer = row["complete_answer"]
    native = verify_answer(source, answer)
    extra = _line_witnesses(source, answer, atoms)
    witnesses = native["witnesses"] + extra
    unique = {
        (witness["kind"], witness["byte_start"], witness["byte_end"]): witness
        for witness in witnesses
    }
    witnesses = list(unique.values())
    checked = sum(
        witness["status"] in {"observed", "scoped_contradiction"} for witness in witnesses
    )
    lexical = sum(
        name in atom["text"]
        for atom in atoms
        for name in set(re.findall(r"`([A-Za-z_]\w*)`", answer))
    )
    unknown = sum(witness["status"] == "unknown" for witness in witnesses)
    ambiguity = sum(witness["status"] == "ambiguous" for witness in witnesses)
    contradictions = sum(witness["status"] == "scoped_contradiction" for witness in witnesses)
    return {
        "unit_id": row["component_hash"],
        "role": row["role"],
        "partition": row["learning_partition"],
        "arm": arm,
        "source_group_id": source_group_id,
        "source_sha256": digest(source),
        "original_source_sha256": row["source_sha256"],
        "answer_sha256": row["answer_sha256"],
        "original_sentences": row["answer_sentences"],
        "historically_exposed": row["historically_exposed"],
        "official_split": row["official_split"],
        "source_role": row["source_role"],
        "injection_label_provenance": "evaluator_sidecar_isolated",
        "dialects": sorted({atom["dialect"] for atom in atoms}),
        "source_atom_count": len(atoms),
        "checked_structural_propositions": checked,
        "lexical_membership": lexical,
        "scoped_contradictions": contradictions,
        "ambiguity": ambiguity,
        "unknown_claims": unknown or int(not witnesses),
        "unsupported_semantic_spans": len(row["answer_sentences"]),
        "witnesses": witnesses,
        "whole_answer_certified": False,
        "numerator": checked,
        "denominator": len(row["answer_sentences"]),
        "raw_metrics": {"checked_propositions": checked, "lexical_membership": lexical},
        "excluded": False,
        "censored": checked == 0,
        "provenance": "Exp7602 isolated predictor store",
    }


def build_role_rows(rows: list[dict], role: str, derangement: dict[str, str]) -> list[dict]:
    """Use every inherited group once per arm without consulting labels."""

    groups = [row["component_hash"] for row in rows]
    if len(groups) != len(set(groups)) or any(row["role"] != role for row in rows):
        raise ValueError("duplicate_or_wrong_role")
    if set(derangement) != set(groups) or set(derangement.values()) != set(groups):
        raise ValueError("derangement_roster_mismatch")
    if any(key == value for key, value in derangement.items()):
        raise ValueError("derangement_fixed_point")
    lookup = {row["component_hash"]: row for row in rows}
    result = []
    for row in rows:
        validate_input(row, row, role)
        group = row["component_hash"]
        for arm in ARMS:
            source_row = row if arm == "original_source" else lookup[derangement[group]]
            source = "" if arm == "evidence_erasure" else source_row["complete_source"]
            source_id = None if arm == "evidence_erasure" else source_row["component_hash"]
            result.append(extract_group(row, source, arm, source_id))
    return result
