"""REQ-REPORT-8314: reuse the qualified receipt reader with stricter cell support.

The existing parser authenticates native environment receipts. A descriptive
arm choice needs support in every game, because pooled firings hide sparse cells.
"""

from pathlib import Path
from types import FunctionType
from typing import Any, cast

from carnot.reporting import arc_outcome_frontier_8300 as baseline

ROOT = baseline.ROOT
FRONTIER = baseline.FRONTIER
PINNED_FRONTIER = baseline.PINNED_FRONTIER
TASK_ID = "exp8314-arc-coverage-frontier"
QUALIFIED_CODE = baseline.QUALIFIED_CODE
qualified = baseline.qualified
authority = baseline.authority
signature = baseline.signature
original_source = baseline.original_source
Json = dict[str, Any]


def summarize(rows: list[Json], prior: Json) -> Json:
    """Keep native denominators while forbidding a candidate from sparse cells."""
    value = baseline.summarize(rows, prior)
    supported = (
        value["arm_overlap_games"] >= 3
        and len(value["shared_arms"]) >= 2
        and all(
            c["fired"] >= 5 for c in value["per_game_arm_rows"] if c["arm"] in value["shared_arms"]
        )
    )
    if not supported:
        value.update(
            proposed_arm_change=None,
            proposed_generalization_change=None,
            descriptive_leave_one_game_out=[],
            support_status="insufficient_support",
        )
    return value


def inspect(locator: Path, frontier: Path, adapter: Path) -> Json:
    """Bind the current identity without changing historical parser or source bytes."""
    adapted = FunctionType(
        original_source.inspect.__code__,
        vars(original_source) | globals() | {"baseline": original_source.baseline},
    )
    return cast(Json, adapted(locator, frontier, adapter))


def unqualified(locator: Path, frontier: Path, adapter: Path) -> Json:
    """Represent an unmeasured delta when owned coverage has failed, never fake rows."""
    return dict(
        summarize([], {}),
        failures=[],
        authority_locator_rows=[],
        source_artifact_hashes={},
        receipt_frontier=dict(path=str(frontier), sha256=None),
        frontier_sha256=None,
        frontier_hashes={},
        historical_excluded_count=None,
        unchanged_authority=False,
        qualification_not_ready=True,
    )
