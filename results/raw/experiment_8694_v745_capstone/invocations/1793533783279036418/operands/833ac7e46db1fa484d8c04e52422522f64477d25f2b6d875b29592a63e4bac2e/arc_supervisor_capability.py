"""REQ-VERIFY-8536: availability is an operand of selection, never an outcome.

The opt-in snapshot keeps legacy serialized snapshots unchanged. Each action
reads the actual policy again, so changing a runtime gate cannot leave a mask
cached from a previous window or level.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from carnot.agentic.arc_trajectory_supervisor import TrajectorySnapshot

FLAG = "CARNOT_ARC_SUPERVISOR_CAPABILITY_MASK"
LLM_ARMS = frozenset({"allow_reinduction", "tool_loop_reinduction"})


@dataclass(frozen=True)
class CapabilitySnapshot(TrajectorySnapshot):
    """Only explicitly enabled selection adds capability fields to receipts."""

    llm_reinduction_available: bool | None = None
    capability_mask_enabled: bool = True


def with_capabilities(policy: Any, snapshot: TrajectorySnapshot) -> CapabilitySnapshot:
    """Ask the current runtime; a missing or unreadable declaration stays unknown."""
    reader = getattr(policy, "trajectory_runtime_capabilities", None)
    try:
        capabilities = reader() if callable(reader) else None
    except (AttributeError, TypeError, ValueError):
        capabilities = None
    value = capabilities.get("llm_reinduction") if isinstance(capabilities, dict) else None
    return CapabilitySnapshot(
        **asdict(snapshot), llm_reinduction_available=value if type(value) is bool else None
    )


def filter_rows(rows: list[dict[str, Any]], snapshot: Any) -> list[dict[str, Any]]:
    """Retain the curated table and every baseline predicate in its original order."""
    available = snapshot.llm_reinduction_available
    result = []
    for row in rows:
        requires_model = row["arm"] in LLM_ARMS
        capability = available if requires_model else True
        result.append(
            dict(
                row,
                baseline_eligible=row["eligible"],
                capability_eligible=capability,
                capability_reason=(
                    "runtime_unknown"
                    if capability is None
                    else "runtime_explicitly_disabled"
                    if capability is False
                    else "runtime_available"
                    if requires_model
                    else "no_model_required"
                ),
                eligible=bool(row["eligible"] and capability is not False),
            )
        )
    return result


def annotate_window(supervisor: Any, snapshot: Any, row: dict[str, Any]) -> None:
    """Unavailable arms are not attempts and cannot certify successful exhaustion."""
    predicates = supervisor.arm_eligibility(snapshot)
    row.update(
        capability_predicates=predicates,
        selection_status=(
            "all_arms_attempted"
            if set(row["arms_enabled"]) <= set(row["arms_used"])
            else "no_currently_executable_arm"
        ),
    )
