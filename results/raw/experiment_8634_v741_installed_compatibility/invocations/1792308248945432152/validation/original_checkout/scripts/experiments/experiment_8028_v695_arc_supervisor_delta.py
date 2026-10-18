#!/usr/bin/env python3
"""Read Exp8014's frontier without game or model work. REQ-REPORT-8028.

The shared runner owns authentication, validation and publication. This scope
keeps calendar changes from becoming evidence or new solve credit.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

# Cold replay must resolve the shared entry point outside the checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.reporting.current_work_receipt import atomic_json, sha256_file  # noqa: E402
from scripts.experiments import experiment_8014_v694_arc_supervisor_delta as baseline  # noqa: E402
from scripts.experiments.experiment_7962_v690_arc_supervisor_delta import main as run  # noqa: E402

runner = baseline.runner
CLI = "scripts/experiments/experiment_8028_v695_arc_supervisor_delta.py"
PRIOR = baseline.scope.OUTPUT
INVENTORY = PRIOR.parent / "raw" / PRIOR.stem / "receipt_inventory.json"
scope = SimpleNamespace(**vars(baseline.scope))
scope.__dict__.update(
    PRIOR=PRIOR,
    INVENTORY=INVENTORY,
    PINNED={
        str(PRIOR): "sha256:5a4e0346fba28c44d5321ae5e37fc157c766c12c685c4bcfaae24b395efac504",
        str(INVENTORY): "sha256:41228c90c759867d700c22a239cae13e47b135dee417cd5afac4883a4db55b10",
        str(runner.REGISTRY): runner.PINNED[str(runner.REGISTRY)],
    },
    EXPERIMENT_ID=8028,
    PRIOR_ID=8014,
    MILESTONE="2026.10.695",
    OUTPUT=runner.ROOT / "results/experiment_8028_v695_arc_supervisor_delta.json",
    MODULE=CLI,
    CLI=CLI,
    TEST="tests/python/test_arc_supervisor_delta_8028.py",
    ADDED=[CLI],
    INCLUDE="*/" + CLI,
    CONSUMERS=[baseline.scope.TEST, *baseline.scope.CONSUMERS],
)


def inputs() -> dict[str, Any]:
    """Bind the prior content frontier before imported observations are eligible."""
    checked = runner.inputs(scope)
    expected = dict(
        experiment_id=8014,
        task_id="exp8014-arc-supervisor-delta",
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
    for row in checked["checks"]:
        row["passed"] = row["expected"] == row["observed"]
    checked["failures"] = [r for r in checked["checks"] if not r["passed"]]
    return checked


def commands(private: Path) -> list[dict[str, Any]]:
    """Keep existing owned checks and one bounded repository-health diagnostic."""
    specs = [s for s in runner.commands(private, scope) if not s["name"].startswith("e2e_016")]
    return [baseline.commands(private)[0], *specs]


def scan(
    root: Path, producers: list[Path], checked: dict[str, Any], private: Path, *, current_date: str
) -> dict[str, Any]:
    """Use qualified authentication; a missing event clock cannot support refinement."""
    value = baseline.qualified.scan(root, producers, checked, private, current_date=current_date)
    for row in value["rows"]:
        if row["status"] in {"completed", "censored", "unknown"} and row["chronology"] == "unknown":
            row.update(status="excluded", reason="unknown_chronology")
    value.update(runner.previous.reduce(value["rows"]))
    atomic_json(private / "primitive_rows.json", value)
    return value


def artifact_fields(value: dict[str, Any], checked: dict[str, Any]) -> dict[str, Any]:
    """Give downstream readers exact readiness while separating science from custody."""
    fields = baseline.artifact_fields(value, checked)
    fields["frontier"].update(
        upstream_id="exp8014-arc-supervisor-delta",
        path=str(scope.INVENTORY),
        sha256=scope.PINNED[str(scope.INVENTORY)],
    )
    raw = scope.OUTPUT.parent / "raw" / scope.OUTPUT.stem
    fields.update(
        arc_delta_ready_score=int(value.get("verdict_class", "null") == "null"),
        current_game_runs=0,
        solve_provenance=sorted({r["solve_provenance"] for r in value["new_event_rows"]}) or None,
        cited_upstream_artifacts=[
            dict(
                experiment_id=8014,
                path=str(p),
                sha256=scope.PINNED[str(p)],
                fields_imported=["receipt_inventory", "seen_receipt_hashes", "outcome_hashes"],
            )
            for p in (scope.PRIOR, scope.INVENTORY)
        ],
        checkpoint_references=[
            dict(path=p, sha256=h, role="current_durable_primitive_rows")
            for p, h in value.get("source_artifact_hashes", {}).items()
            if Path(p).is_relative_to(raw) and p.endswith("/receipt_inventory.json")
        ],
    )
    return fields


def replay(value: dict[str, Any]) -> list[str]:
    """Reject altered summary fields and changed durable primitive checkpoints."""
    expected = artifact_fields(value, {})
    errors = baseline.replay(value) + [
        key
        for key in ("arc_delta_ready_score", "current_game_runs", "solve_provenance")
        if key in value and value[key] != expected[key]
    ]
    for row in value.get("checkpoint_references", []):
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            errors.append("checkpoint_sha256")
    return errors


def execute(output: Path, private: Path) -> int:
    """End science at the delta and publish using existing validation orchestration."""
    return runner.execute(output, private, scope)


def terminal(candidate: Path, private: Path, durable: Path) -> dict[str, Any]:
    """Run shared byte validators and this scope's reducer in a fresh child."""
    report = runner.previous.terminal(candidate, private, durable)
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
                str(candidate),
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
    report["passed"] = report["passed"] and row["passed"]
    return report


scope.__dict__.update(
    inputs=inputs,
    commands=commands,
    scan=scan,
    artifact_fields=artifact_fields,
    replay=replay,
    execute=execute,
    terminal=terminal,
)


def main(argv: list[str] | None = None) -> int:
    """Reuse the shared parser and its isolated fixture and publication routes."""
    return run(argv, scope=scope)


if __name__ == "__main__":
    raise SystemExit(main())
