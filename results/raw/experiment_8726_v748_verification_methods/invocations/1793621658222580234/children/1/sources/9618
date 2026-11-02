#!/usr/bin/env python3
"""Read the qualified supervisor delta without new game work. REQ-REPORT-8014.

The content frontier comes from the qualified receipt, so old source bytes do
not become new evidence. Associations only suggest a future controlled test.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any

# A cold child starts outside the checkout and still needs the shared CLI.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.reporting import arc_supervisor_qualification as qualified  # noqa: E402
from carnot.reporting import arc_supervisor_v690_delta as runner  # noqa: E402
from carnot.reporting.current_work_receipt import sha256_file  # noqa: E402
from scripts.experiments.experiment_7962_v690_arc_supervisor_delta import main as run  # noqa: E402

CLI = "scripts/experiments/experiment_8014_v694_arc_supervisor_delta.py"
PRIOR = qualified.OUTPUT
INVENTORY = PRIOR.parent / "raw" / PRIOR.stem / "receipt_inventory.json"
scope = SimpleNamespace(
    ROOT=runner.ROOT,
    PRIOR=PRIOR,
    INVENTORY=INVENTORY,
    REGISTRY=runner.REGISTRY,
    PINNED={
        str(PRIOR): "sha256:cda6822317a6c00c269fcac80273c10b63e6aa85b8095937ec8f78a297ac88aa",
        str(INVENTORY): "sha256:5fb137de2b64dd372c0198ef97c793c9c6a7d2418dd5842afea3b6736ff4ba91",
        str(runner.REGISTRY): runner.PINNED[str(runner.REGISTRY)],
    },
    EXPERIMENT_ID=8014,
    PRIOR_ID=8001,
    MILESTONE="2026.10.694",
    RUN_DATE="20261002",
    OUTPUT=runner.ROOT / "results/experiment_8014_v694_arc_supervisor_delta.json",
    MODULE=CLI,
    CLI=CLI,
    TEST="tests/python/test_arc_supervisor_delta_8014.py",
    ADDED=[CLI],
    INCLUDE="*/" + CLI,
    CONSUMERS=[qualified.TEST, qualified.history.TEST, runner.TEST, *runner.CONSUMERS],
    COVERAGE_TESTS=[],
    APPLICABLE_E2E=["E2E-017"],
    FREEZE_SOURCES=True,
    previous=runner.previous,
    scan=qualified.scan,
)


def inputs() -> dict[str, Any]:
    """Require qualified identity and matching outcome hashes before scanning."""
    checked = runner.inputs(scope)
    expected = dict(
        experiment_id=8001,
        task_id="exp8001-arc-supervisor-delta",
        run_date="20261002",
        honest_verdict="complete_null_no_new_outcomes",
        no_new_outcomes=True,
        outcome_hashes=checked["inventory"].get("outcome_hashes", "missing"),
    )
    checked["checks"].extend(
        runner.previous.operand(
            scope.PRIOR,
            key,
            value,
            checked["prior"].get(key, "missing"),
            scope.PINNED[str(scope.PRIOR)],
        )
        for key, value in expected.items()
    )
    checked["failures"] = [r for r in checked["checks"] if r["expected"] != r["observed"]]
    return checked


def commands(private: Path) -> list[dict[str, Any]]:
    """Reuse private routes and run the applicable small end-to-end check."""
    specs = [s for s in runner.commands(private, scope) if not s["name"].startswith("e2e_016")]
    specs.insert(
        0,
        dict(
            name="e2e_017",
            argv=[
                str(scope.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_arc_supervisor_delta_7874.py",
                f"--basetemp={private / 'e2e017'}",
            ],
            expected_exit=0,
            expected_text=None,
            deadline_s=60,
            classification="required",
        ),
    )
    return specs


def artifact_fields(value: dict[str, Any], checked: dict[str, Any]) -> dict[str, Any]:
    """Separate reader headroom, durable custody and observational suggestions."""
    fields = qualified.history.artifact_fields(value, checked)
    state = value.get("verdict_class", "null")
    rows = value["new_event_rows"]
    control = qualified.positive_control()
    for arms in fields["per_game_arm_statistics"].values():
        for cell in arms.values():
            cell["resolved_by_levelup"] = cell["helped"]
    fields.update(
        honest_verdict="complete_null_no_new_outcomes"
        if state == "null" and not rows
        else "complete_null_observational_outcomes"
        if state == "null"
        else "complete_" + state + "_supervisor_delta",
        no_new_outcomes=not rows,
        frontier=dict(
            upstream_id="exp8001-arc-supervisor-delta",
            path=str(scope.INVENTORY),
            sha256=scope.PINNED[str(scope.INVENTORY)],
            role="qualified_content_frontier",
            producer_invocation_date="20261002",
            receipt_inventory=value.get("receipt_inventory", {}),
            outcome_hashes=value.get("outcome_hashes", []),
        ),
        positive_control_results=control,
        genuine_headroom=dict(
            reader_accepts_supported_progress=control["genuine_headroom"],
            claim_scope="protocol_fixture_only",
            natural_evidence_count=0,
            scientific_benefit=None,
        ),
        chronology_unknown_rows=[r for r in rows if r["chronology"] == "unknown"],
        proposed_generalization_refinement=dict(
            action="future_controlled_generalization_test",
            existing_arms=sorted({r["arm"] for r in rows}),
            causal_gain_claimed=False,
            selection_callsite="E3AgentPolicy._maybe_supervise_trajectory -> time_supervisor_selection",
            application_callsite="E3AgentPolicy._apply_trajectory_redirect",
            eligible_selection_change="Randomize eligible existing arms against unchanged control with matched budgets.",
        )
        if fields["sample_interpretation"] == "directional_observational"
        else None,
        provenance_scope="referenced_receipts_only",
        registry_precheck_receipt=dict(
            path=str(scope.REGISTRY),
            sha256=scope.PINNED[str(scope.REGISTRY)],
            artifact_field="games[*].levels_reproduced",
            live_path_receipts=rows,
            new_credit=0,
        ),
        checkpoint_references=[
            dict(path=p, sha256=h, role="current_durable_primitive_rows")
            for p, h in value.get("source_artifact_hashes", {}).items()
            if p.endswith("/receipt_inventory.json")
        ],
        cited_upstream_artifacts=[
            dict(
                experiment_id=8001,
                path=str(p),
                sha256=scope.PINNED[str(p)],
                fields_imported=[
                    "receipt_inventory",
                    "seen_receipt_hashes",
                    "outcome_hashes",
                    "source_artifact_hashes",
                ],
            )
            for p in (scope.PRIOR, scope.INVENTORY)
        ],
    )
    return fields


def replay(value: dict[str, Any]) -> list[str]:
    """Cold reduction rejects supplemental drift without claiming fixture science."""
    expected = artifact_fields(value, {})
    return runner.replay(value) + [
        key
        for key in (
            "no_new_outcomes",
            "per_game_arm_statistics",
            "chronology_unknown_rows",
            "positive_control_results",
            "genuine_headroom",
            "proposed_generalization_refinement",
        )
        if key in value and value[key] != expected[key]
    ]


def execute(output: Path, private: Path) -> int:
    """Keep publication and validation in the qualified shared implementation."""
    return runner.execute(output, private, scope)


def terminal(candidate: Path, private: Path, durable: Path) -> dict[str, Any]:
    """Run exact-byte validators and a fresh child outside repository scratch."""
    copy = private / "terminal-candidate.json"
    shutil.copyfile(candidate, copy)
    report = runner.previous.terminal(copy, private, durable)
    launcher = "import os, runpy, sys; os.chdir(sys.argv[1]); sys.argv = sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    row = runner.previous.run(
        dict(
            name="cold_primary_replay",
            argv=[
                "env",
                "-u",
                "PYTHONPATH",
                str(scope.ROOT / ".venv/bin/python"),
                "-u",
                "-c",
                launcher,
                str(private),
                str(scope.ROOT / CLI),
                "--cold-replay",
                str(copy),
            ],
            deadline_s=60,
            expected_exit=0,
            expected_text=None,
            classification="required",
        ),
        private,
        durable,
    )
    report["reports"].append(row)
    report["passed"] = (
        report["passed"] and row["passed"] and sha256_file(copy) == sha256_file(candidate)
    )
    return report


scope.inputs = inputs
scope.commands = commands
scope.artifact_fields = artifact_fields
scope.replay = replay
scope.execute = execute
scope.terminal = terminal


def main(argv: list[str] | None = None) -> int:
    """Use existing tested argument and fixture routes for this invocation."""
    return run(argv, scope=scope)


if __name__ == "__main__":
    raise SystemExit(main())
