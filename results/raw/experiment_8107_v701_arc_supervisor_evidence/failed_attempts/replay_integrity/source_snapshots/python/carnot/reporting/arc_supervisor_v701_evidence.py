"""Qualify the real reader before inspecting redirects. Spec: REQ-REPORT-8107.

Reuse the authenticated scanner and publication runner so private controls
cannot become natural game evidence or earn generalization credit.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from carnot.reporting import arc_supervisor_v698_frontier as baseline
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

runner = baseline.runner
MODULE = "python/carnot/reporting/arc_supervisor_v701_evidence.py"
CLI = "scripts/experiments/experiment_8107_v701_arc_supervisor_evidence.py"
PRIOR = baseline.scope.OUTPUT
INVENTORY = PRIOR.parent / "raw" / PRIOR.stem / "receipt_inventory.json"
SIDECAR = INVENTORY.parent / "published_terminal_reports.json"
scope = SimpleNamespace(**vars(baseline.scope))
scope.__dict__.update(
    PRIOR=PRIOR,
    INVENTORY=INVENTORY,
    EXPERIMENT_ID=8107,
    PRIOR_ID=8067,
    MILESTONE="2026.10.701",
    RUN_DATE="20261004",
    MODULE=MODULE,
    CLI=CLI,
    OUTPUT=runner.ROOT / "results/experiment_8107_v701_arc_supervisor_evidence.json",
    TEST="tests/python/test_arc_supervisor_evidence_8107.py",
    ADDED=[MODULE, CLI],
    INCLUDE=",".join("*/" + p for p in (MODULE, CLI)),
    COVERAGE_TESTS=[],
    CONSUMERS=[baseline.scope.TEST, *baseline.scope.CONSUMERS],
    QUALIFY_BEFORE_SCAN=True,
    APPLICABLE_E2E=["E2E-017"],
    PINNED={
        str(PRIOR): "sha256:2291a36ceb97c68a57618620994bfd829eb2c7989fa4893f76ab47a799c508d2",
        str(INVENTORY): "sha256:f5669e32cfb5ad69200b068d7a9f7682d32511232dba7fe92c192b671afb951f",
        str(SIDECAR): "sha256:8f878d6843c45fbc77c1f2e52f9eb125f66f5446b3081e271594e3097ddb6a8f",
        str(runner.REGISTRY): runner.PINNED[str(runner.REGISTRY)],
    },
)


def inputs() -> dict[str, Any]:
    """Resolve the original CLI import; expected bytes come from qualified custody."""
    checked = runner.inputs(scope)
    checked["checks"].extend(baseline.inputs()["checks"])
    tree = ast.parse((scope.ROOT / baseline.CLI).read_text())
    imported = next(
        n.module
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and any(a.name == "scope" for a in n.names)
    )
    assert imported is not None
    spec = importlib.util.find_spec(imported)
    assert spec is not None and spec.origin is not None
    resolved = Path(spec.origin).resolve()
    checked["resolved_reader_path"] = str(resolved)
    checks = checked["checks"]
    checks.append(
        runner.previous.operand(
            resolved,
            "resolved_import",
            baseline.MODULE,
            str(resolved.relative_to(scope.ROOT)),
            sha256_file(resolved),
        )
    )
    hashes = checked["prior"].get("source_artifact_hashes", {})
    for label in (*baseline.AUTH_CODE, baseline.MODULE, baseline.CLI):
        path = scope.ROOT / label
        actual = sha256_file(path) if path.is_file() else "missing"
        checks.append(
            runner.previous.operand(
                path, "sha256", hashes.get(label, "qualified_hash_required"), actual, actual
            )
        )
    actual = sha256_file(SIDECAR) if SIDECAR.is_file() else "missing"
    checks.append(
        runner.previous.operand(SIDECAR, "sha256", scope.PINNED[str(SIDECAR)], actual, actual)
    )
    document = json.loads(SIDECAR.read_text()) if actual == scope.PINNED[str(SIDECAR)] else {}
    for field, expected, observed in (
        (
            "primary_sha256",
            scope.PINNED[str(scope.PRIOR)],
            document.get("primary_sha256", "missing"),
        ),
        ("report.passed", True, document.get("report", {}).get("passed", "missing")),
        ("experiment_id", 8067, checked["prior"].get("experiment_id", "missing")),
        (
            "outcome_hashes",
            checked["inventory"].get("outcome_hashes", "missing"),
            checked["prior"].get("outcome_hashes", "missing"),
        ),
    ):
        path = SIDECAR if field in {"primary_sha256", "report.passed"} else scope.PRIOR
        checks.append(runner.previous.operand(path, field, expected, observed, actual))
    for row in checks:
        row.update(
            check="authenticate_" + row["artifact_field"],
            upstream=row["upstream_id"],
            hash=row["sha256"],
            field=row["artifact_field"],
            passed=row["expected"] == row["observed"],
        )
    checked["failures"] = [r for r in checks if not r["passed"]]
    checked["additional_source_hashes"] = {
        r["path"]: r["sha256"] for r in checks if r["passed"] and r["artifact_field"] == "sha256"
    }
    return checked


def commands(private: Path) -> list[dict[str, Any]]:
    """Freeze existing controls with private cwd and coverage of only new code."""
    specs = [baseline.commands(private)[0], *runner.commands(private, scope)]
    specs = [s for s in specs if not s["name"].startswith("e2e_016")]
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
    return specs


def scan(
    root: Path, producers: list[Path], checked: dict[str, Any], private: Path, *, current_date: str
) -> dict[str, Any]:
    """Preserve original receipt operands while naming each observational unit."""
    value = baseline.scope.scan(root, producers, checked, private, current_date=current_date)
    for row in value["rows"]:
        row.update(
            source_id=row.get("source_sha256"),
            unit_id=row.get("receipt_id"),
            condition="observational_live_redirect",
            issued_state=row.get("chronology"),
            metric="observed_levelup_after_redirect",
            numerator=int(row.get("resolved_by_levelup") is True),
            denominator=1,
            exclusion_reason=row.get("reason"),
        )
    atomic_json(private / "primitive_rows.json", value)
    return value


def artifact_fields(value: dict[str, Any], checked: dict[str, Any]) -> dict[str, Any]:
    """An empty qualified reader is ready; observed help is never causal credit."""
    fields = baseline.artifact_fields(value, checked)
    ready = fields["required_checks_passed"] and value.get("verdict_class", "null") not in {
        "blocked",
        "disqualified",
    }
    redirects = [
        dict(r, fired=1, helped=int(r["resolved_by_levelup"] is True))
        for r in value["new_event_rows"]
    ]
    frontier = dict(
        upstream_id="exp8067-arc-supervisor-frontier",
        path=str(scope.INVENTORY),
        sha256=scope.PINNED[str(scope.INVENTORY)],
        producer_invocation_date=scope.RUN_DATE,
    )
    raw = scope.OUTPUT.parent / "raw" / scope.OUTPUT.stem
    fields.update(
        task_id="exp8107-arc-supervisor-evidence",
        title="Qualified V701 supervisor evidence",
        claim_scope=0,
        exposure_scope=0,
        verifier_is_oracle=0,
        supervisor_reader_ready_score=int(ready),
        new_outcome_count=sum(r["status"] == "completed" for r in redirects),
        redirect_rows=redirects,
        new_solve_credit=0,
        leaderboard_change=0,
        frontier=frontier,
        event_frontier=frontier,
        prior_frontier=dict(frontier, receipt_inventory=value.get("seen_receipt_hashes", {})),
        frontier_hashes=dict(
            prior_primary=scope.PINNED[str(scope.PRIOR)],
            prior_inventory=scope.PINNED[str(scope.INVENTORY)],
            current_outcomes=canonical_hash(value.get("outcome_hashes", [])),
        ),
        recommended_generalization_change=fields["candidate_refinement"],
        solve_provenance="live_agent_self_discovery"
        if any(r["helped"] for r in redirects)
        else "not_applicable_no_solve_claim",
        acceptance_gates=dict(
            reader_qualified=int(ready),
            new_redirect_outcomes=len(redirects),
            causal_benefit_claimed=False,
            independent_generalization_credit=0,
        ),
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
                fields_imported=["receipt_inventory", "outcome_hashes", "finished_at"],
            )
            for p in (scope.PRIOR, scope.INVENTORY)
        ],
    )
    for row in value.get("gate_check_summary", []):
        row.update(
            check="validate_" + row["artifact_field"],
            upstream=row["upstream_id"],
            hash=row["sha256"],
            field=row["artifact_field"],
        )
    return fields


def replay(value: dict[str, Any]) -> list[str]:
    """Recompute claims and reject altered durable bytes in a fresh process."""
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
    """Both validators and a cold external CLI must accept identical candidate bytes."""
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


def execute(output: Path, private: Path) -> int:
    """The qualified shared runner owns validation and atomic primary publication."""
    return int(runner.execute(output, private, scope))


scope.__dict__.update(
    inputs=inputs,
    commands=commands,
    scan=scan,
    artifact_fields=artifact_fields,
    replay=replay,
    terminal=terminal,
    execute=execute,
)
