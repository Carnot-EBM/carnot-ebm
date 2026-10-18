"""Current V676 custody and readiness checks for the scored ARC runner.

The scored actions stay in the qualified V675 runner. These checks make it
harder to count invented probes or silently omit an affected validation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from carnot.experiment_7763_v675_arc_runner_qualification import ARMS, cold_reduce, schedule_rows


def reduce_probe_evidence(
    schedule: list[dict[str, Any]], probes: list[dict[str, Any]]
) -> dict[str, int]:
    """Recount raw observations only when every probe belongs to the frozen panel."""
    if schedule != schedule_rows():
        raise ValueError("schedule_changed")
    scheduled = {str(row["episode_id"]) for row in schedule}
    scheduled.update(
        f"{game}:{seed}:{arm}"
        for game, seed in (("fixture", 67500), ("r11l", 67501))
        for arm in ARMS
    )
    if any(str(probe["episode_id"]) not in scheduled for probe in probes):
        raise ValueError("unscheduled_probe")
    return cold_reduce(schedule, probes)


def gate_decision(
    receipts: Sequence[Mapping[str, Any]], required: Sequence[str], *, sdk_ok: bool
) -> tuple[bool, list[str]]:
    """Open readiness only for one passing receipt per required affected check."""
    failed = [
        name
        for name in required
        if len(matches := [row for row in receipts if row.get("name") == name]) != 1
        or matches[0].get("exit_code") != 0
        or matches[0].get("passed") is not True
        or matches[0].get("timed_out") is True
    ]
    if not sdk_ok:
        failed.append("sdk_transport")
    return not failed, failed
