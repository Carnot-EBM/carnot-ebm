"""Bind current evidence to qualified original bytes. Spec: REQ-REPORT-8054.

The shared reader authenticates live outcomes. This scope preserves its empty
frontier and validation custody without changing the agent or running games.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

from carnot.reporting import arc_supervisor_v696_delta as baseline
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

runner = baseline.runner
MODULE = "python/carnot/reporting/arc_supervisor_v697_delta.py"
CLI = "scripts/experiments/experiment_8054_v697_arc_supervisor_delta.py"
PRIOR = baseline.scope.OUTPUT
INVENTORY = PRIOR.parent / "raw" / PRIOR.stem / "receipt_inventory.json"
SIDECAR = INVENTORY.parent / "published_terminal_reports.json"
SCANNER = runner.ROOT / "results/experiment_8001_v693_arc_supervisor_qualification.json"
AUTH_CODE = (
    "python/carnot/reporting/arc_supervisor_qualification.py",
    "python/carnot/reporting/arc_supervisor_v688_receipts.py",
    "python/carnot/reporting/arc_supervisor_v689_delta.py",
    "python/carnot/reporting/v686_contract_validation.py",
    "python/carnot/reporting/primary_publication.py",
    "python/carnot/reporting/current_work_receipt.py",
    "tests/python/test_arc_supervisor_qualification_8001.py",
)
scope = SimpleNamespace(**vars(baseline.scope))
scope.__dict__.update(
    PRIOR=PRIOR,
    INVENTORY=INVENTORY,
    PINNED={
        str(PRIOR): "sha256:0873749a8cf561646d0be4671384351a849a792f284d19c1f09d6dbbe84f2647",
        str(INVENTORY): "sha256:26e9727044cdd1f15e34059ee91bd91e4fc67415ecb1acb60618b9097663bc4f",
        str(SIDECAR): "sha256:0b6864be547565657e76d3a6b7e644b79718cb42d41adc24c3779405a78f6743",
        str(SCANNER): "sha256:cda6822317a6c00c269fcac80273c10b63e6aa85b8095937ec8f78a297ac88aa",
        str(runner.REGISTRY): runner.PINNED[str(runner.REGISTRY)],
    },
    EXPERIMENT_ID=8054,
    PRIOR_ID=8041,
    MILESTONE="2026.10.697",
    RUN_DATE="20261003",
    OUTPUT=runner.ROOT / "results/experiment_8054_v697_arc_supervisor_delta.json",
    MODULE=MODULE,
    CLI=CLI,
    TEST="tests/python/test_arc_supervisor_delta_8054.py",
    ADDED=[MODULE, CLI],
    INCLUDE=",".join("*/" + p for p in (MODULE, CLI)),
    CONSUMERS=[baseline.scope.TEST, *baseline.scope.CONSUMERS],
)


def inputs() -> dict[str, Any]:
    """Qualified primary bytes bind both the outcome frontier and validation code."""
    checked = runner.inputs(scope)
    expected = dict(
        experiment_id=8041,
        task_id="exp8041-arc-supervisor-delta",
        run_date="20261002",
        honest_verdict="complete_null_no_new_outcomes",
        arc_delta_ready_score=1,
        outcome_hashes=checked["inventory"].get("outcome_hashes", "missing"),
    )
    checks = checked["checks"]
    checks.extend(
        runner.previous.operand(
            scope.PRIOR, k, v, checked["prior"].get(k, "missing"), scope.PINNED[str(scope.PRIOR)]
        )
        for k, v in expected.items()
    )
    for path in (SIDECAR, SCANNER):
        actual = sha256_file(path) if path.is_file() else "missing"
        checks.append(
            runner.previous.operand(path, "sha256", scope.PINNED[str(path)], actual, actual)
        )
        document = json.loads(path.read_text()) if actual == scope.PINNED[str(path)] else {}
        fields = (
            {"primary_sha256": scope.PINNED[str(scope.PRIOR)], "report.passed": True}
            if path == SIDECAR
            else {"experiment_id": 8001, "arc_evidence_ready_score": 1}
        )
        for field, target in fields.items():
            observed = (
                document.get("report", {}).get("passed", "missing")
                if field == "report.passed"
                else document.get(field, "missing")
            )
            checks.append(runner.previous.operand(path, field, target, observed, actual))
    hashes = checked["prior"].get("source_artifact_hashes", {})
    for label in AUTH_CODE:
        path = scope.ROOT / label
        actual = sha256_file(path) if path.is_file() else "missing"
        checks.append(
            runner.previous.operand(
                path, "sha256", hashes.get(label, "qualified_code_hash_required"), actual, actual
            )
        )
    for row in checks:
        row.update(
            passed=row["expected"] == row["observed"],
            check_name="authenticate_" + row["artifact_field"],
        )
    checked["additional_source_hashes"] = {
        row["path"]: row["sha256"]
        for row in checks
        if row["passed"] and row["artifact_field"] == "sha256"
    }
    checked["failures"] = [row for row in checks if not row["passed"]]
    return checked


def commands(private: Path) -> list[dict[str, Any]]:
    """Owned acceptance stays separate from one bounded whole-repository diagnostic."""
    specs = [s for s in runner.commands(private, scope) if not s["name"].startswith("e2e_016")]
    return [baseline.commands(private)[0], *specs]


def artifact_fields(value: dict[str, Any], checked: dict[str, Any]) -> dict[str, Any]:
    """An authenticated empty delta needs no new policy or scientific benefit claim."""
    fields = baseline.artifact_fields(value, checked)
    fields["frontier"].update(
        upstream_id="exp8041-arc-supervisor-delta",
        path=str(scope.INVENTORY),
        sha256=scope.PINNED[str(scope.INVENTORY)],
        producer_invocation_date=scope.RUN_DATE,
    )
    raw = scope.OUTPUT.parent / "raw" / scope.OUTPUT.stem
    failures = value.get("gate_check_summary", [])
    for row in failures:
        row.update(check_name="validate_" + row["artifact_field"], passed=False)
    fields.update(
        event_frontier=fields["frontier"],
        checkpoint_references=[
            dict(path=p, sha256=h, role="current_durable_primitive_rows")
            for p, h in value.get("source_artifact_hashes", {}).items()
            if Path(p).is_relative_to(raw) and p.endswith("/receipt_inventory.json")
        ],
        cited_upstream_artifacts=[
            dict(
                experiment_id=8041,
                path=str(p),
                sha256=scope.PINNED[str(p)],
                fields_imported=[
                    "receipt_inventory",
                    "seen_receipt_hashes",
                    "outcome_hashes",
                    "finished_at",
                ],
            )
            for p in (scope.PRIOR, scope.INVENTORY)
        ],
        unavailable_input_count=sum(r.get("observed") == "missing" for r in failures),
        tampered_input_count=sum(
            r.get("artifact_field") == "sha256" and r.get("observed") != "missing" for r in failures
        ),
        claim_scope="This 20261003 invocation reduces only authenticated content beyond Exp8041; exposed observational evidence grants no causal or solve credit.",
    )
    if value.get("verdict_class") == "blocked" and failures:
        fields["honest_verdict"] = "complete_blocked_" + Path(failures[0]["path"]).stem
    return fields


def replay(value: dict[str, Any]) -> list[str]:
    """Fresh reduction rejects invented counts and altered durable checkpoints."""
    expected = artifact_fields(value, {})
    errors = runner.replay(value) + [
        k
        for k in expected
        if k in value
        and value[k] != expected[k]
        and k
        not in {
            "honest_verdict",
            "code_config_hashes",
            "raw_shard_hashes",
            "cited_upstream_artifacts",
        }
    ]
    for row in value.get("checkpoint_references", []):
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            errors.append("checkpoint_sha256")
    return errors


def terminal(candidate: Path, private: Path, durable: Path) -> dict[str, Any]:
    """Validate identical candidate bytes and cold-reduce them outside the checkout."""
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


def execute(output: Path, private: Path) -> int:
    """The shared publisher exposes one primary only after terminal validation."""
    status = int(runner.execute(output, private, scope))
    durable = output.parent / "raw" / output.stem
    report = terminal(output, private, durable)
    atomic_json(
        durable / "published_terminal_reports.json",
        dict(primary_path=str(output), primary_sha256=sha256_file(output), report=report),
    )
    assert report["passed"], "published_byte_validation_failed"
    return status


scope.__dict__.update(
    inputs=inputs,
    commands=commands,
    scan=baseline.scan,
    artifact_fields=artifact_fields,
    replay=replay,
    terminal=terminal,
    execute=execute,
)
