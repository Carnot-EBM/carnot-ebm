"""REQ-REPORT-8272: adapt sealed history without replacing receipt authority.

Only the private interface identity changes. The producer bytes, environment
join and support calculation remain those qualified by the previous audit.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import Any

from carnot.reporting import arc_outcome_frontier_8257 as baseline
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar

# Copy namespaces so changing the invocation date cannot mutate historical readers.
dated = SimpleNamespace(**vars(baseline.qualified))


def event_order(row: Json, cutoff: Json, current: str) -> str:
    """The receipt join stays unchanged; only this audit's final date advances."""
    return baseline.qualified.event_order(row, cutoff, "20261008")


dated.inspect = FunctionType(
    baseline.qualified.inspect.__code__, dict(vars(baseline.qualified), event_order=event_order)
)
qualified = SimpleNamespace(**vars(baseline))
qualified.inspect = FunctionType(baseline.inspect.__code__, dict(vars(baseline), qualified=dated))
ROOT = baseline.ROOT
FRONTIER = ROOT / "results/experiment_8257_v713_arc_outcome_frontier.json"
PINNED_FRONTIER = "sha256:f9951075753d6bd3d67397315efab5eb00dd16208495c7971dc0b3d581108670"
TASK_ID = "exp8272-arc-outcome-frontier"
QUALIFIED_CODE = [
    *qualified.QUALIFIED_CODE,
    "python/carnot/reporting/arc_outcome_frontier_8257.py",
    "python/carnot/reporting/arc_outcome_execution_8257.py",
    "scripts/experiments/experiment_8257_v713_arc_outcome_frontier.py",
]
authority = qualified.authority
signature = qualified.qualified.signature
summarize = qualified.summarize
Json = dict[str, Any]


def inspect(locator: Path, frontier: Path, adapter: Path) -> Json:
    """Authenticate history before projecting its identity into the qualified interface."""
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
        experiment_id=8257,
        task_id=qualified.TASK_ID,
        schema="arc-outcome-frontier-v713",
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
        experiment_id=8243,
        task_id="exp8243-arc-supervisor-frontier",
        schema="arc-supervisor-frontier-v712",
        qualification_projection_origin=dict(path=str(frontier), sha256=hashes.get(str(frontier))),
    )
    if not any(not row["passed"] for row in checks):
        if not projection.is_file():
            publish_primary(
                projection,
                projected,
                lambda _: dict(passed=True, origin=projected["qualification_projection_origin"]),
            )
        check(projection, "qualification_projection", projected, json.loads(projection.read_text()))
    failures = [row for row in checks if not row["passed"]]
    delta = qualified.inspect(locator, projection, adapter / "qualified") if not failures else {}
    failures += delta.get("failures", [])
    rows = (
        delta.get("rows", [])
        if not failures
        else [
            dict(
                row,
                status="failed",
                reason="external_authority",
                condition="authority_operand",
                metric="operand_authenticated",
                numerator=0,
                denominator=1,
            )
            for row in failures
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
