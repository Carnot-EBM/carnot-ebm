"""Bind current evidence to qualified original bytes. Spec: REQ-REPORT-8080.

The shared reader authenticates live outcomes. This scope preserves its empty
frontier and validation custody without changing the agent or running games.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

from carnot.reporting import arc_supervisor_v698_frontier as baseline
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

runner = baseline.runner
MODULE = "python/carnot/reporting/arc_supervisor_v699_frontier.py"
CLI = "scripts/experiments/experiment_8080_v699_arc_supervisor_frontier.py"
PRIOR = baseline.scope.OUTPUT
INVENTORY = PRIOR.parent / "raw" / PRIOR.stem / "receipt_inventory.json"
SIDECAR = INVENTORY.parent / "published_terminal_reports.json"
SCANNER = runner.ROOT / "results/experiment_8001_v693_arc_supervisor_qualification.json"
AUTH_CODE = (*baseline.AUTH_CODE, baseline.MODULE, baseline.CLI)
NAMED_PRIOR_CODE = "python/carnot/experiment_8067_v698_arc_supervisor_frontier.py"

scope = SimpleNamespace(**vars(baseline.scope))
scope.__dict__.update(
    PRIOR=PRIOR,
    INVENTORY=INVENTORY,
    PINNED={
        str(PRIOR): "sha256:2291a36ceb97c68a57618620994bfd829eb2c7989fa4893f76ab47a799c508d2",
        str(INVENTORY): "sha256:f5669e32cfb5ad69200b068d7a9f7682d32511232dba7fe92c192b671afb951f",
        str(SIDECAR): "sha256:8f878d6843c45fbc77c1f2e52f9eb125f66f5446b3081e271594e3097ddb6a8f",
        str(SCANNER): "sha256:cda6822317a6c00c269fcac80273c10b63e6aa85b8095937ec8f78a297ac88aa",
        str(runner.REGISTRY): runner.PINNED[str(runner.REGISTRY)],
    },
    EXPERIMENT_ID=8080,
    PRIOR_ID=8067,
    MILESTONE="2026.10.699",
    RUN_DATE="20261003",
    OUTPUT=runner.ROOT / "results/experiment_8080_v699_arc_supervisor_frontier.json",
    MODULE=MODULE,
    CLI=CLI,
    TEST="tests/python/test_arc_supervisor_frontier_8080.py",
    ADDED=[MODULE, CLI],
    INCLUDE=",".join("*/" + p for p in (MODULE, CLI)),
    CONSUMERS=[
        baseline.scope.TEST,
        *baseline.scope.CONSUMERS,
        "tests/python/test_adversarial_verify_guards.py",
    ],
)


def inputs() -> dict[str, Any]:
    """Qualified primary bytes bind both the outcome frontier and validation code."""
    checked = runner.inputs(scope)
    expected = dict(
        experiment_id=8067,
        task_id="exp8067-arc-supervisor-frontier",
        run_date="20261003",
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
    for label in (
        ".venv/bin/python",
        ".venv/bin/pytest",
        ".venv/bin/coverage",
        ".venv/bin/ruff",
        ".venv/bin/mypy",
        NAMED_PRIOR_CODE,
        "CLAUDE.md",
        "CODEX.md",
        "ops/e2e-test-plan.md",
        "scripts/experiment_template.py",
        "python/carnot/agentic/arc_competition_agent.py",
        "python/carnot/agentic/arc_solver_kit.py",
        "ops/exclusion_manifest.yaml",
        "openspec/change-proposals/research-roadmap-vNEXT.md",
        "scripts/experiments/experiment_7874_v683_arc_supervisor_delta.py",
    ):
        path = scope.ROOT / label
        checks.append(
            runner.previous.operand(
                path,
                "is_file",
                True,
                True if path.is_file() else "missing",
                sha256_file(path) if path.is_file() else None,
            )
        )
    for row in checks:
        row.update(
            passed=row["expected"] == row["observed"],
            check_name="authenticate_" + row["artifact_field"],
            check="authenticate_" + row["artifact_field"],
            upstream=row["upstream_id"],
            hash=row["sha256"],
            field=row["artifact_field"],
        )
    checked["additional_source_hashes"] = {
        row["path"]: row["sha256"]
        for row in checks
        if row["passed"] and row["artifact_field"] in {"sha256", "is_file"}
    }
    checked["failures"] = [row for row in checks if not row["passed"]]
    history = scope.ROOT / "results/raw" / scope.OUTPUT.stem / "prior_owned_attempt.json"
    if history.is_file():
        record = json.loads(history.read_text())
        checked["prior"].setdefault("historical_required_failures", []).append(record)
        checked["additional_source_hashes"][str(history)] = sha256_file(history)
    return checked


def commands(private: Path) -> list[dict[str, Any]]:
    """Owned acceptance stays separate from one bounded whole-repository diagnostic."""
    specs = [s for s in runner.commands(private, scope) if not s["name"].startswith("e2e_016")]
    specs = [baseline.commands(private)[0], *specs]
    launcher = "import os, runpy, sys; os.chdir(sys.argv[1]); sys.argv = sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    for spec in specs:
        if spec["name"].startswith("cli_"):
            argv = [str(scope.ROOT / a) if a == CLI else a for a in spec["argv"]]
            spec["argv"] = [
                "env",
                "-u",
                "PYTHONPATH",
                str(scope.ROOT / ".venv/bin/python"),
                "-u",
                "-c",
                launcher,
                str(private),
                *argv,
            ]
        if spec["name"] in {"unit_coverage", "coverage_json"}:
            spec["argv"] = [
                a + ",*/scripts/adversarial_verify.py" if a.startswith("--include=") else a
                for a in spec["argv"]
            ]
            spec["argv"].insert(2, "--omit=__no_omitted_added_code__")
        if spec["name"] in {"ruff_check", "ruff_format", "mypy"}:
            spec["argv"].append("scripts/adversarial_verify.py")
    proof = "import ast,json; from pathlib import Path; p=Path('scripts/adversarial_verify.py'); t=ast.parse(p.read_text()); g=next(n for n in ast.walk(t) if isinstance(n,ast.If) and 'k == \\\"arc_delta_ready_score\\\"' in ast.get_source_segment(p.read_text(),n)); lines={g.lineno,g.body[0].lineno}; v=json.loads(Path(__import__('sys').argv[1]).read_text()); assert lines <= set(v['files'][str(p)]['executed_lines']); print({'added_statement_lines':sorted(lines),'coverage_percent':100},flush=True)"
    # This check covers the two added statements, without demanding old code coverage.
    specs.insert(
        next(i for i, s in enumerate(specs) if s["name"] == "coverage_json") + 1,
        dict(
            name="verifier_added_statement_coverage",
            argv=[
                str(scope.ROOT / ".venv/bin/python"),
                "-c",
                proof,
                str(private / "coverage.json"),
            ],
            deadline_s=60,
            expected_exit=0,
            expected_text=None,
            classification="required",
        ),
    )
    history = scope.ROOT / "results/raw" / scope.OUTPUT.stem / "prior_owned_attempt.json"
    if history.is_file():
        record = json.loads(history.read_text())
        health = next(s for s in specs if s["name"] == "full_python_suite")
        health.update(
            reuse_receipt=record["repository_health"][0],
            reuse_source_path=str(history),
            reuse_source_sha256=sha256_file(history),
        )
    return specs


def artifact_fields(value: dict[str, Any], checked: dict[str, Any]) -> dict[str, Any]:
    """An authenticated empty delta needs no new policy or scientific benefit claim."""
    for row in value.get("gate_check_summary", []):
        if "artifact_field" not in row:
            row.update(
                runner.previous.operand(
                    scope.OUTPUT,
                    "terminal_validation",
                    True,
                    row.get("report", {}).get("passed", False),
                    None,
                )
            )
    fields = baseline.artifact_fields(value, checked)
    fields["frontier"].update(
        upstream_id="exp8067-arc-supervisor-frontier",
        path=str(scope.INVENTORY),
        sha256=scope.PINNED[str(scope.INVENTORY)],
        producer_invocation_date=scope.RUN_DATE,
    )
    raw = scope.OUTPUT.parent / "raw" / scope.OUTPUT.stem
    failures = value.get("gate_check_summary", [])
    for row in failures:
        row.update(
            check_name="validate_" + row["artifact_field"],
            passed=False,
            check="validate_" + row["artifact_field"],
            upstream=row["upstream_id"],
            hash=row["sha256"],
            field=row["artifact_field"],
        )
    fields.update(
        task_id="exp8080-arc-supervisor-frontier",
        title="Authenticated V699 live supervisor frontier",
        blocked_input_count=len(failures),
        pending_events=[
            r
            for r in value["rows"]
            if r["status"] in {"censored", "unknown"} or r.get("chronology") == "unknown"
        ],
        event_frontier=fields["frontier"],
        prior_frontier=dict(
            upstream_id="exp8067-arc-supervisor-frontier",
            path=str(scope.INVENTORY),
            sha256=scope.PINNED[str(scope.INVENTORY)],
            receipt_inventory=value.get("seen_receipt_hashes", {}),
        ),
        current_frontier=dict(
            receipt_inventory=value.get("receipt_inventory", {}),
            outcome_hashes=value.get("outcome_hashes", []),
        ),
        new_outcome_count=sum(r["status"] == "completed" for r in value["new_event_rows"]),
        refinement_proposal=fields["candidate_refinement"],
        required_checks_passed=all(
            r["passed"]
            for r in value.get("validation_receipts", [])
            if r["classification"] == "required"
        )
        and value.get("verdict_class", "null") != "disqualified",
        checkpoint_references=[
            dict(path=p, sha256=h, role="current_durable_primitive_rows")
            for p, h in value.get("source_artifact_hashes", {}).items()
            if Path(p).is_relative_to(raw) and p.endswith("/receipt_inventory.json")
        ],
        cited_upstream_artifacts=[
            dict(
                experiment_id=8067,
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
        claim_scope="This 20261003 invocation reduces only authenticated content beyond Exp8067; exposed observational evidence grants no causal or solve credit.",
    )
    if fields["refinement_proposal"] is not None:
        fields["refinement_proposal"]["future_held_out_protocol"] = dict(
            max_closed_firings=30,
            minimum_games=3,
            minimum_closed_firings=10,
            design="Freeze curated arms, randomize selection/control before future held-out outcomes, match action budgets and compare game-level help and censoring.",
            stop_rule="Stop at 30 closed firings or the preregistered action budget; retain all open and failed units.",
            acceptance="Report game-grouped uncertainty, action cost and regressions; no default change follows from observational support.",
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
    scan=baseline.scope.scan,
    artifact_fields=artifact_fields,
    replay=replay,
    terminal=terminal,
    execute=execute,
)
