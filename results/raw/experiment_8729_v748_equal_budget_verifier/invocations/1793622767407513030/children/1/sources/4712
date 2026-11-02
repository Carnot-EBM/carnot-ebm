"""Public-family selection and label-blind relation rows for REQ-REPORT-7673.

The caller supplies only public shard columns. Outcome fields are rejected at
the predictor boundary, so source relations cannot silently learn annotations.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import re
from typing import Any

from carnot.verify.tool_source_relations import verify_relations


def digest(value: str) -> str:
    """Bind exact UTF-8 text; even whitespace changes invalidate custody."""
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def stable_hash(value: Any) -> str:
    """Hash ordered JSON with one encoding for role and family receipts."""
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")))


def normalized(value: str) -> str:
    """Fold case and spacing so trivially edited copies share a family."""
    return " ".join(value.casefold().split())


def source_template(value: str) -> str:
    """Collapse line numbers and changing identifiers in tool records."""
    return re.sub(r"\b\d+\b", "<n>", normalized(value))


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    """Read public text and instance ID, without interpreting label metadata."""
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        match = re.search(r'"instance_id"\s*:\s*"([^"\\]+)"', metadata)
        instance = match.group(1) if match else ""
    elif isinstance(metadata, Mapping):
        instance = str(metadata.get("instance_id", ""))
    else:
        instance = ""
    values = (
        row.get("official_split"),
        row.get("context"),
        row.get("question") or "",
        row.get("answer"),
        instance,
    )
    if (
        not all(isinstance(value, str) for value in values)
        or not values[0]
        or not values[1]
        or not values[3]
    ):
        raise ValueError("public_row_incomplete")
    return values  # type: ignore[return-value]


def cluster_public_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Join sibling answers, source copies, and source templates first.

    Transitive union prevents one copied source from receiving two roles. A
    family touching both official splits is excluded from both populations.
    """
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        parents[find(right)] = find(left)

    seen: dict[str, int] = {}
    identities = [_identity(row) for row in rows]
    for index, (_, source, _, answer, instance) in enumerate(identities):
        keys = (
            "instance:" + instance if instance else "",
            "source:" + normalized(source),
            "template:" + source_template(source),
            "answer:" + normalized(answer),
        )
        for key in keys:
            if not key:
                continue
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        grouped[find(index)].append(index)
    families, collisions = [], []
    for members in grouped.values():
        splits = {identities[index][0] for index in members}
        source_hashes = sorted({digest(normalized(identities[index][1])) for index in members})
        answer_hashes = sorted({digest(normalized(identities[index][3])) for index in members})
        family_id = stable_hash([source_hashes, answer_hashes])
        record = {
            "family_id": family_id,
            "member_count": len(members),
            "instance_ids": sorted(
                {identities[index][4] for index in members if identities[index][4]}
            ),
            "source_hashes": source_hashes,
            "answer_hashes": answer_hashes,
            "template_hashes": sorted(
                {digest(source_template(identities[index][1])) for index in members}
            ),
            "exact_source_hashes": sorted({digest(identities[index][1]) for index in members}),
            "exact_answer_hashes": sorted({digest(identities[index][3]) for index in members}),
        }
        if len(splits) != 1:
            collisions.append(
                {
                    **record,
                    "reason": "cross_official_split_family",
                    "official_splits": sorted(splits),
                }
            )
            continue
        complete = [index for index in members if identities[index][2]]
        if not complete:
            collisions.append(
                {**record, "reason": "complete_question_missing", "official_splits": sorted(splits)}
            )
            continue
        choice = min(complete, key=lambda i: stable_hash(identities[i]))
        families.append({**record, "official_split": splits.pop(), "view": dict(rows[choice])})
    return sorted(families, key=lambda x: x["family_id"]), sorted(
        collisions, key=lambda x: x["family_id"]
    )


def subtract_exposure(
    families: Sequence[dict], ledger: Mapping[str, set[str]]
) -> tuple[list[dict], list[dict]]:
    """Exclude every family matching an exposed source, answer, or identity."""
    eligible, excluded = [], []
    for family in families:
        exposed = (
            family["family_id"] in ledger["family_ids"]
            or bool(set(family["source_hashes"]) & ledger["source_hashes"])
            or bool(set(family.get("answer_hashes", [])) & ledger["answer_hashes"])
            or bool(set(family.get("template_hashes", [])) & ledger.get("template_hashes", set()))
            or bool(
                set(family.get("exact_source_hashes", []))
                & ledger.get("exact_source_hashes", set())
            )
            or bool(
                set(family.get("exact_answer_hashes", []))
                & ledger.get("exact_answer_hashes", set())
            )
            or bool(set(family.get("instance_ids", [])) & ledger.get("instance_ids", set()))
        )
        if exposed:
            excluded.append({"family_id": family["family_id"], "reason": "prior_exposure"})
        else:
            eligible.append(family)
    return eligible, excluded


def assign_roles(families: Sequence[dict], counts: Mapping[str, int], salt: str) -> list[dict]:
    """Hash-sort the public eligible roster and require exact planned counts."""
    if len({family["family_id"] for family in families}) != len(families):
        raise ValueError("duplicate_family")
    selected = []
    for split in ("train", "test"):
        roles = [role for role in counts if (role == "evaluation") == (split == "test")]
        pool = sorted(
            (family for family in families if family["official_split"] == split),
            key=lambda family: digest(salt + ":" + family["family_id"]),
        )
        needed = sum(counts[role] for role in roles)
        if len(pool) < needed:
            raise ValueError(f"underfilled_{split}:{len(pool)}<{needed}")
        offset = 0
        for role in roles:
            selected.extend(
                {**family, "role": role} for family in pool[offset : offset + counts[role]]
            )
            offset += counts[role]
    return selected


PREDICTOR_FIELDS = frozenset(
    {
        "component_hash",
        "role",
        "learning_partition",
        "official_split",
        "complete_source",
        "complete_question",
        "complete_answer",
        "source_sha256",
        "answer_sha256",
        "source_role",
        "historically_exposed",
        "labels_accessible",
        "raw_probability_accessible",
        "answer_sentences",
    }
)
ARMS = ("original_source", "evidence_erasure", "within_role_derangement")


def predictor_view(row: Mapping[str, Any], family_id: str, role: str) -> dict:
    """Project only public bytes; no original metadata survives projection."""
    split, source, question, answer, _ = _identity(row)
    if (role == "evaluation") != (split == "test"):
        raise ValueError("role_split_mismatch")
    if role not in {
        "fit",
        "retention",
        "tune",
        "policy",
        "online_update",
        "online_admission",
        "evaluation",
    }:
        raise ValueError("role_invalid")
    from carnot.experiment_7588_v663_evidence_protocol import segment_lossless

    return {
        "component_hash": family_id,
        "role": role,
        "learning_partition": role,
        "official_split": split,
        "complete_source": source,
        "complete_question": question,
        "complete_answer": answer,
        "source_sha256": digest(source),
        "answer_sha256": digest(answer),
        "answer_sentences": segment_lossless(answer, "R"),
        "source_role": "tool_output",
        "historically_exposed": False,
        "labels_accessible": False,
        "raw_probability_accessible": False,
    }


def validate_predictor(row: Mapping[str, Any], role: str) -> None:
    """Refuse labels, extra fields, role drift, and altered text bytes."""
    if set(row) != PREDICTOR_FIELDS or any(
        "label" in key and key != "labels_accessible" for key in row
    ):
        raise ValueError("label_or_field_access_rejected")
    if row["labels_accessible"] is not False or row["raw_probability_accessible"] is not False:
        raise ValueError("label_access_rejected")
    if row["role"] != role or row["learning_partition"] != role:
        raise ValueError("role_mismatch")
    if row["source_sha256"] != digest(row["complete_source"]):
        raise ValueError("source_hash_mismatch")
    if row["answer_sha256"] != digest(row["complete_answer"]):
        raise ValueError("answer_hash_mismatch")
    from carnot.experiment_7588_v663_evidence_protocol import roundtrip_segments

    roundtrip_segments(row["complete_answer"], row["answer_sentences"])


def feature_rows(rows: Sequence[dict], role: str) -> list[dict]:
    """Produce three source views with the same answer and family weight."""
    if len(rows) < 2:
        raise ValueError("derangement_underfilled")
    by_id = {row["component_hash"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate_family")
    for row in rows:
        validate_predictor(row, role)
    ordered = sorted(rows, key=lambda row: row["component_hash"])
    donors = {
        row["component_hash"]: ordered[(i + 1) % len(ordered)] for i, row in enumerate(ordered)
    }
    output = []
    for row in ordered:
        for arm in ARMS:
            donor = row if arm == "original_source" else donors[row["component_hash"]]
            source = "" if arm == "evidence_erasure" else donor["complete_source"]
            source_group_id = None if arm == "evidence_erasure" else donor["component_hash"]
            relations = verify_relations(source, row["complete_answer"])
            findings = relations["findings"]
            checked = sum(item["status"] != "unknown" for item in findings)
            output.append(
                {
                    "unit_id": row["component_hash"],
                    "role": role,
                    "arm": arm,
                    "source_group_id": source_group_id,
                    "source_sha256": digest(source),
                    "original_source_sha256": row["source_sha256"],
                    "answer_sha256": row["answer_sha256"],
                    "source_atom_count": len(relations["relations"]),
                    "checked_relations": checked,
                    "unknown_claims": sum(item["status"] == "unknown" for item in findings),
                    "findings": findings,
                    "relations": relations["relations"],
                    "raw_metrics": {
                        "checked_relations": checked,
                        "source_atom_count": len(relations["relations"]),
                    },
                    "numerator": checked,
                    "denominator": len(findings),
                    "excluded": False,
                    "censored": checked == 0,
                    "whole_answer_certified": False,
                    "provenance": "pinned LettuceDetect tool-output public view",
                }
            )
    return output
