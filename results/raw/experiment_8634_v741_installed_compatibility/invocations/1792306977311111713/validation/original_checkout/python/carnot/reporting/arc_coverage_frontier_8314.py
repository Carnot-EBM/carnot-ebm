"""REQ-REPORT-8314: reuse the qualified receipt reader with stricter cell support.

The existing parser authenticates native environment receipts. A descriptive
arm choice needs support in every game, because pooled firings hide sparse cells.
"""

from pathlib import Path
from types import FunctionType
from typing import Any, cast

from carnot.reporting import arc_outcome_frontier_8300 as baseline
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

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
    value = cast(Json, adapted(locator, frontier, adapter))
    audit = adapter / "authority_checks.json"
    atomic_json(audit, value["authority_locator_rows"])
    value["authority_locator_rows"] = [compact(r) for r in value["authority_locator_rows"]]
    value["failures"] = [compact(r) for r in value["failures"]]
    value["rows"] = [compact(r) for r in value["rows"]]
    value["authority_projection_evidence_path"] = str(audit)
    value["source_artifact_hashes"][str(audit)] = sha256_file(audit)
    return value


def compact(row: Json) -> Json:
    """Keep full comparisons in the raw audit and scalar hashes in the primary.

    Historical primaries contain their own nested evidence. Repeating those
    complete objects makes a small current audit needlessly expensive to verify.
    Original decisions stay unchanged and replay recomputes the comparisons.
    """
    if any(isinstance(row.get(k), (dict, list)) for k in ("expected", "observed")):
        return dict(
            row,
            expected=canonical_hash(row.get("expected")),
            observed=canonical_hash(row.get("observed")),
            artifact_field=row["artifact_field"] + ".canonical_sha256",
        )
    return row


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
