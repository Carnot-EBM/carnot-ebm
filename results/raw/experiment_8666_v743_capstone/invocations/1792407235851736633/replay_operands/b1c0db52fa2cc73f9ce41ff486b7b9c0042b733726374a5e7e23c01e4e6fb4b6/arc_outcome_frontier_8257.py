"""REQ-REPORT-8257: adapt authenticated history to the unchanged qualified reader.

The private projection changes interface identity only. Its source signature,
clock and receipt frontier come from sealed Exp8243 bytes, never invented events.
"""

import json
from pathlib import Path
from typing import Any

from carnot.reporting import arc_supervisor_frontier_8243 as qualified
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar

ROOT = qualified.ROOT
FRONTIER = ROOT / "results/experiment_8243_v712_arc_supervisor_frontier.json"
PINNED_FRONTIER = "sha256:b7222147212f0bf6b74a186f83dd7d05fb0a31348f950c0247a64e3024ec0ec3"
TASK_ID = "exp8257-arc-outcome-frontier"
QUALIFIED_CODE = [
    "python/carnot/reporting/arc_supervisor_frontier_8243.py",
    "python/carnot/reporting/arc_authoritative_frontier_8215.py",
    "python/carnot/reporting/arc_outcome_delta_8229.py",
]
authority = qualified.authority
Json = dict[str, Any]


def summarize(rows: list[Json], prior: Json) -> Json:
    """Small samples cannot order arms, even when a descriptive rate is computable."""
    value = qualified.summarize(rows, prior)
    supported = (
        value["arm_overlap_games"] >= 3
        and len(value["shared_arms"]) >= 2
        and all(value["per_arm_results"][a]["fired"] >= 5 for a in value["shared_arms"])
    )
    if not supported:
        value["descriptive_leave_one_game_out"] = []
    proposal = (
        dict(
            status="candidate_only",
            compared_arms=value["shared_arms"],
            leave_one_game_out=value["descriptive_leave_one_game_out"],
            causal_superiority=False,
            live_priority_modified=False,
            specification="Prospectively compare the leave-one-game-out preferred arm against the current arm in matched game/context windows; reject the change unless environment resolution rate improves with a confidence interval excluding zero.",
        )
        if supported
        else None
    )
    value.update(
        proposed_generalization_change=proposal,
        proposed_arm_change=proposal,
        support_status="observational_candidate" if supported else "insufficient_support",
        reopen_condition="New authenticated post-Exp8243 producer bytes and environment receipts; ordering requires three shared games, two arms and five firings per arm.",
    )
    return value


def inspect(locator: Path, frontier: Path, adapter: Path) -> Json:
    """Authenticate the real primary first; a projection never replaces missing authority."""
    checks: list[Json] = []
    hashes: dict[str, str] = {}

    def check(path: Path, field: str, expected: Any, observed: Any) -> None:
        digest = sha256_file(path) if path.is_file() else None
        if digest:
            hashes[str(path)] = digest
        checks.append(
            dict(
                authority.operand(path, field, expected, observed, digest),
                passed=expected == observed,
            )
        )

    previous: Json = {}
    check(frontier, "is_file", True, frontier.is_file())
    if frontier.is_file():
        if frontier == FRONTIER:
            check(frontier, "sha256", PINNED_FRONTIER, sha256_file(frontier))
        try:
            previous = json.loads(frontier.read_text())
            if not isinstance(previous, dict):
                raise ValueError("frontier_not_object")
            sidecar = (
                frontier.parent
                / "raw"
                / frontier.stem
                / "validators"
                / (sha256_file(frontier)[7:] + ".json")
            )
            report = read_bound_sidecar(frontier, sidecar)
            check(sidecar, "report.passed", True, report.get("report", {}).get("passed"))
            check(sidecar, "primary_path", str(frontier.absolute()), report.get("primary_path"))
        except (ValueError, OSError) as error:
            previous = {}
            check(frontier, "bound_frontier", True, str(error))
    for key, expected in dict(
        experiment_id=8243,
        task_id="exp8243-arc-supervisor-frontier",
        schema="arc-supervisor-frontier-v712",
        required_checks_passed=True,
        arc_delta_ready_score=1,
        verdict_class="null",
    ).items():
        check(frontier, key, expected, previous.get(key))
    for label in QUALIFIED_CODE:
        path = ROOT / label
        expected = previous.get("code_config_hashes", {}).get(label)
        check(path, "qualified_code_hash_present", True, expected is not None)
        check(
            path, "qualified_code_sha256", expected, sha256_file(path) if path.is_file() else None
        )
    prior = previous.get("current_frontier", {})
    projection = adapter / qualified.FRONTIER.name
    projected = dict(
        previous,
        experiment_id=8229,
        task_id="exp8229-arc-outcome-delta",
        schema="arc-outcome-delta-v711",
        qualification_projection_origin=dict(path=str(frontier), sha256=hashes.get(str(frontier))),
    )
    if not any(not r["passed"] for r in checks):
        if not projection.is_file():
            publish_primary(
                projection,
                projected,
                lambda _: dict(passed=True, origin=projected["qualification_projection_origin"]),
            )
        check(projection, "qualification_projection", projected, json.loads(projection.read_text()))
    failures = [r for r in checks if not r["passed"]]
    delta = qualified.inspect(locator, projection) if not failures else {}
    failures += delta.get("failures", [])
    rows = (
        delta.get("rows", [])
        if not failures
        else [
            dict(
                r,
                status="failed",
                reason="external_authority",
                condition="authority_operand",
                metric="operand_authenticated",
                numerator=0,
                denominator=1,
            )
            for r in failures
        ]
    )
    return dict(
        delta,
        **summarize(rows, prior),
        failures=failures,
        authority_locator_rows=[*checks, *delta.get("authority_locator_rows", [])],
        source_artifact_hashes=dict(delta.get("source_artifact_hashes", {}), **hashes),
        frontier_sha256=hashes.get(str(frontier)),
        frontier_hashes=delta.get("frontier_hashes", {}),
        historical_excluded_count=previous.get("excluded_count"),
        unchanged_authority=delta.get("unchanged_authority", False),
        receipt_frontier=dict(
            path=str(frontier),
            sha256=hashes.get(str(frontier)),
            after_finished_at=previous.get("finished_at"),
            **prior,
        ),
        adapter_path=str(adapter),
    )
