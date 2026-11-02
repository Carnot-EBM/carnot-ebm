"""Qualify exposed source families before opening evaluator labels.

The public join contains original bytes only. Keeping label checks in this
module prevents a response label from silently becoming a sentence label.
REQ-REPORT-7892-V685.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from carnot.reporting import source_boundary_7866, source_boundary_7880
from carnot.verify import source_projection

ORIGINAL_ROLES = source_boundary_7866.ROLES
EIGHT_ROLES = {
    "fit": 256,
    "tune": 64,
    "policy_design": 32,
    "calibration_replay": 32,
    "online_update": 96,
    "online_admission": 64,
    "evaluation": 64,
    "retention": 32,
}
IDENTITY = {
    "experiment_id": 7892,
    "task_id": "exp7892-source-boundary",
    "milestone": "2026.09.685",
    "execution_venue": "host",
}


def require_identity(artifact: dict[str, Any]) -> None:
    """Stop an old producer or venue from being published under this task."""
    for field, expected in IDENTITY.items():
        if artifact.get(field) != expected:
            raise ValueError(f"identity:{field}")


def qualify(
    rows: list[dict[str, Any]], public: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Keep the original roster, split policy by ID, and isolate labels."""
    if len(rows) != 640 or len(public) != 640:
        raise ValueError("family_count")
    if Counter(row.get("role") for row in rows) != ORIGINAL_ROLES:
        raise ValueError("role_count")
    if len({row.get("family_id") for row in rows}) != 640:
        raise ValueError("family_duplicate")
    source_boundary_7866.check_role_groups(rows)
    if len({row["source_group"] for row in rows}) != 640:
        raise ValueError("duplicate_group")
    split = source_boundary_7880.policy_subroles(rows)
    qualified: list[dict[str, Any]] = []
    evaluators: list[dict[str, Any]] = []
    for row, source in zip(rows, public, strict=True):
        if set(source) != source_projection.PUBLIC_KEYS or source["family_id"] != row["family_id"]:
            raise ValueError("public_metadata_or_join")
        label = row["label_provenance"].get("human_label")
        if label not in (0, 1):
            raise ValueError("missing_human_label")
        role = split[row["family_id"]] if row["role"] == "policy" else row["role"]
        qualified.append(
            {
                "family_id": row["family_id"],
                "role": role,
                "source_group": row["source_group"],
                "source_cluster_id": row["source_sha256"],
                "source_sha256": row["source_sha256"],
                "status": row["status"],
                "arm": "public_projection",
                "seed": 68592,
            }
        )
        evaluators.append(
            {
                "family_id": row["family_id"],
                "role": role,
                "human_label": label,
                "observed": 1,
                "label_scope": "response",
                "annotation_byte_offsets": row["label_provenance"]["annotation_byte_offsets"],
                "response_id": row["label_provenance"]["response_id"]
                if "response_id" in row["label_provenance"]
                else None,
            }
        )
    counts = dict(Counter(row["role"] for row in qualified))
    if counts != EIGHT_ROLES:
        raise ValueError("policy_role_count")
    return qualified, evaluators, counts


def budget(rows: list[dict[str, Any]], intended: int) -> dict[str, int]:
    """Expose every terminal unit state so readers can recompute totals."""
    states = Counter(row["status"] for row in rows)
    return {
        "intended": intended,
        "eligible": states["completed"],
        "started": len(rows),
        "completed": states["completed"],
        "failed": states["failed"],
        "censored": states["censored"],
        "excluded": states["excluded"],
        "independent": len({row["family_id"] for row in rows}),
    }
