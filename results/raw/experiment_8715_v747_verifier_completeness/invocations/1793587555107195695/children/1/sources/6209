"""Keep calibration labels in their original roles to prevent evaluator leakage.

REQ-VERIFY-7968. Human spans remain fallible exposed development evidence.
The existing exact-byte join is the only annotation engine.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any
import unicodedata

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.source_boundary_7892 import EIGHT_ROLES
from carnot.verify import response_targets_7955 as targets
from carnot.verify.sentence_labels_7942 import digest, index_unique

Json = dict[str, Any]


def partition(
    public: list[Json],
    audit: list[Json],
    roster: list[Json],
    rows: list[Json],
    annotations: list[Json],
) -> tuple[Json, Json]:
    """Export every original slot so label imbalance cannot change admission."""
    frozen, boundaries = targets.freeze(public)
    indexed = targets.check_roles(frozen, roster)
    if Counter(row["role"] for row in roster) != EIGHT_ROLES:
        raise ValueError("original_role_roster")
    clusters: dict[str, str] = {}
    for row in frozen:
        text = bytes.fromhex(row["source_bytes"]).decode("utf-8")
        normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()
        cluster = digest(normalized.encode())
        role = indexed[row["family_id"]]["role"]
        if cluster in clusters and clusters[cluster] != role:
            raise ValueError("normalized_source_role")
        clusters[cluster] = role
    union = index_unique(rows, "family_id")
    if set(union) != set(indexed) or boundaries != audit:
        raise ValueError("union_drift")
    for row in frozen:
        label = union[row["family_id"]]
        role = indexed[row["family_id"]]
        if label["role"] != role["role"]:
            raise ValueError("role_drift")
        if label["source_cluster_id"] != role["source_cluster_id"] or label[
            "response_sha256"
        ] != digest(bytes.fromhex(row["answer_bytes"])):
            raise ValueError("union_drift")
    views, counts, classes, sources = {}, {}, {}, {}
    for role in EIGHT_ROLES:
        labels = [r for r in rows if r["role"] == role]
        ids = {r["family_id"] for r in labels}
        eligible = [r for r in labels if r["y"] is not None]
        classes[role] = {str(y): sum(r["y"] == y for r in labels) for y in (0, 1)}
        classes[role]["unknown"] = len(labels) - len(eligible)
        counts[role] = len(labels)
        sources[role] = dict(
            intended=len({r["source_cluster_id"] for r in labels}),
            eligible=len({r["source_cluster_id"] for r in eligible}),
        )
        views[role] = dict(
            public=dict(
                role=role,
                request_rows=[r for r in frozen if r["family_id"] in ids],
                boundaries=[r for r in audit if r["family_id"] in ids],
            ),
            evaluator=dict(
                role=role,
                rows=labels,
                annotation_rows=[r for r in annotations if r["family_id"] in ids],
            ),
        )
    completed = sum(sum(c[str(y)] for y in (0, 1)) for c in classes.values())
    return views, dict(
        response_roles_ready_score=1,
        role_counts=counts,
        class_counts_by_role=classes,
        source_cluster_counts_by_role=sources,
        cross_role_overlap_count=0,
        rows_sha256=canonical_hash(rows),
        exclusion_rows=[r for r in rows if r["y"] is None],
        sample_size_budget=dict(
            unit="complete_response_family",
            intended=640,
            eligible=completed,
            started=640,
            completed=completed,
            failed=0,
            censored=0,
            excluded=640 - completed,
            independent=len({r["source_cluster_id"] for r in rows if r["y"] is not None}),
        ),
    )


def authorize(role: str, purpose: str, seals: Json) -> None:
    """Require a consumer's declared purpose before exposing evaluator labels."""
    allowed = dict(
        capture=set(EIGHT_ROLES),
        fitting={"fit", "tune"},
        threshold_design={"policy_design"},
        evaluation=set(EIGHT_ROLES) - {"fit", "tune", "policy_design"},
    )
    if role not in allowed.get(purpose, set()):
        raise ValueError("role_access")
    if purpose == "evaluation":
        if set(seals) != {"heads", "policies"}:
            raise ValueError("seals_required")
        from carnot.experiment_7942_v689_sentence_labels import checked_reference

        for item in seals.values():
            checked_reference(item)


def read_view(path: Path, expected_hash: str, role: str, purpose: str, seals: Json) -> Json:
    """A capture route opens one public file and never opens evaluator files."""
    import json

    from carnot.experiment_7942_v689_sentence_labels import checked_reference

    authorize(role, purpose, seals)
    value: Json = json.loads(
        checked_reference(dict(path=str(path), sha256=expected_hash)).read_text()
    )
    keys = (
        {"role", "request_rows", "boundaries"}
        if purpose == "capture"
        else {"role", "rows", "annotation_rows"}
    )
    if set(value) != keys or value["role"] != role:
        raise ValueError("role_view_fields")
    if purpose == "capture":
        if targets.freeze(value["request_rows"])[1] != value["boundaries"]:
            raise ValueError("public_drift")
    elif any(row["role"] != role for row in value["rows"]):
        raise ValueError("role_drift")
    return value
