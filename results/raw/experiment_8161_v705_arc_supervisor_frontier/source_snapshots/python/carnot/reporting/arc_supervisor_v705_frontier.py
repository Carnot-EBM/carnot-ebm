"""Reuse a qualified reader without changing its authority. REQ-REPORT-8161.

Saved hashes authorize the reader and event frontier. Private controls check
execution only; an empty authentic ledger completes this task without policy
changes, model loads or new game runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from carnot.reporting import arc_supervisor_v704_renewal as baseline
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

runner = baseline.runner
MODULE = "python/carnot/reporting/arc_supervisor_v705_frontier.py"
CLI = "scripts/experiments/experiment_8161_v705_arc_supervisor_frontier.py"
PRIOR = baseline.scope.OUTPUT
INVENTORY = PRIOR.parent / "raw" / PRIOR.stem / "receipt_inventory.json"
SIDECAR = INVENTORY.parent / "terminal_reports.json"
METHODS = runner.ROOT / "openspec/change-proposals/research-roadmap-v704-preserved-20261005.md"
scope = SimpleNamespace(**vars(baseline.scope))
scope.__dict__.update(
    EXPERIMENT_ID=8161,
    PRIOR_ID=8147,
    MILESTONE="2026.10.705",
    RUN_DATE="20261005",
    PRIOR=PRIOR,
    INVENTORY=INVENTORY,
    MODULE=MODULE,
    CLI=CLI,
    OUTPUT=runner.ROOT / "results/experiment_8161_v705_arc_supervisor_frontier.json",
    TEST="tests/python/test_arc_supervisor_frontier_8161.py",
    ADDED=[MODULE, CLI],
    INCLUDE=",".join("*/" + p for p in (MODULE, CLI)),
    CONSUMERS=[baseline.scope.TEST, *baseline.scope.CONSUMERS],
    QUALIFY_BEFORE_SCAN=False,
    PINNED={
        str(PRIOR): "sha256:e888104dc7920fd6404464302428779f0473ac42799ea43977164c9544112fef",
        str(INVENTORY): "sha256:b642a4643a5d481409ef26efc7e5774bee78d56233e6c306570065012a917708",
        str(SIDECAR): "sha256:de1e922c05a4c427e3befbbcacc38782135d3479309ee7627084782ba088a355",
        str(runner.REGISTRY): runner.PINNED[str(runner.REGISTRY)],
        str(METHODS): "sha256:92dca5a4ce979d47a47da2e9622da8e1cdd306a8a45c051f2737c3ee1e984f28",
    },
)


def inputs() -> dict[str, Any]:
    """Expected reader bytes come only from the authenticated upstream receipt."""
    checked = runner.inputs(scope)
    checks = checked["checks"]
    for path in (METHODS,):
        actual = sha256_file(path) if path.is_file() else "missing"
        checks.append(
            runner.previous.operand(path, "sha256", scope.PINNED[str(path)], actual, actual)
        )
    for label in (
        *scope.ADDED,
        scope.TEST,
        *[".venv/bin/" + n for n in ("python", "pytest", "coverage", "ruff", "mypy")],
        "CLAUDE.md",
        "CODEX.md",
        "ops/e2e-test-plan.md",
        "scripts/experiment_template.py",
        "python/carnot/agentic/arc_competition_agent.py",
        "python/carnot/agentic/arc_solver_kit.py",
        "ops/exclusion_manifest.yaml",
        "openspec/capabilities/verification/spec.md",
        "openspec/capabilities/arc-world-model-trust-energy/spec.md",
    ):
        path = scope.ROOT / label
        checks.append(
            runner.previous.operand(
                path, "is_file", True, path.is_file(), sha256_file(path) if path.is_file() else None
            )
        )
    actual = sha256_file(SIDECAR) if SIDECAR.is_file() else "missing"
    sidecar = json.loads(SIDECAR.read_text()) if actual == scope.PINNED[str(SIDECAR)] else {}
    for path, field, expected, observed in (
        (SIDECAR, "sha256", scope.PINNED[str(SIDECAR)], actual),
        (SIDECAR, "primary_sha256", scope.PINNED[str(scope.PRIOR)], sidecar.get("primary_sha256")),
        (SIDECAR, "report.passed", True, sidecar.get("report", {}).get("passed")),
        (
            scope.PRIOR,
            "supervisor_reader_ready_score",
            1,
            checked["prior"].get("supervisor_reader_ready_score"),
        ),
        (
            scope.PRIOR,
            "required_checks_passed",
            True,
            checked["prior"].get("required_checks_passed"),
        ),
    ):
        checks.append(runner.previous.operand(path, field, expected, observed, actual))
    hashes = checked["prior"].get("reader_receipt", {}).get("current_code_hashes", {})
    checks.append(
        runner.previous.operand(
            scope.PRIOR, "reader_hashes_present", True, bool(hashes), scope.PINNED[str(scope.PRIOR)]
        )
    )
    for label, expected in hashes.items():
        path = scope.ROOT / label
        actual = sha256_file(path) if path.is_file() else "missing"
        checks.append(runner.previous.operand(path, "reader_code_sha256", expected, actual, actual))
    for row in checks:
        row.update(
            check="authenticate_" + row["artifact_field"],
            upstream=row["upstream_id"],
            hash=row["sha256"],
            passed=row["expected"] == row["observed"],
        )
    checked["failures"] = [r for r in checks if not r["passed"]]
    checked["additional_source_hashes"] = {
        r["path"]: r["sha256"]
        for r in checks
        if r["passed"] and r["artifact_field"] in {"sha256", "reader_code_sha256", "is_file"}
    }
    return checked


def commands(private: Path) -> list[dict[str, Any]]:
    """Freeze owned checks and retain one separate whole-repository health run."""
    specs = [baseline.commands(private)[0], *runner.commands(private, scope)]
    specs = [s for s in specs if not s["name"].startswith("e2e_016")]
    launcher = "import os,runpy,sys; os.chdir(sys.argv[1]); sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name='__main__')"
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
        if spec["name"] == "affected_pytest":
            spec["deadline_s"] = 180
    return specs


def scan(
    root: Path, producers: list[Path], checked: dict[str, Any], private: Path, *, current_date: str
) -> dict[str, Any]:
    """Reuse original assertions and identity filtering while preserving old files."""
    original = baseline.scope.OUTPUT
    baseline.scope.OUTPUT = scope.OUTPUT
    try:
        return baseline.scan(root, producers, checked, private, current_date=current_date)
    finally:
        baseline.scope.OUTPUT = original


def artifact_fields(value: dict[str, Any], checked: dict[str, Any]) -> dict[str, Any]:
    """Execution readiness is separate from new observational outcomes and benefit."""
    fields = baseline.artifact_fields(value, checked)
    ready = fields["supervisor_reader_ready_score"]
    count = len(value["new_event_rows"])
    sources = value.get("source_artifact_hashes", {})
    hashes = runner.validation.dependency_hashes(
        scope.ROOT, paths=[*scope.ADDED, scope.TEST, *scope.CONSUMERS]
    )
    shards = {p: h for p, h in sources.items() if scope.OUTPUT.stem in p and "/raw/" in p}
    frontier = dict(
        path=str(scope.INVENTORY),
        sha256=scope.PINNED[str(scope.INVENTORY)],
        after_finished_at=checked.get("prior", {}).get("finished_at")
        or value.get("event_frontier", {}).get("after_finished_at"),
        preserved_event_ids=sorted(
            r["event_id"] for r in checked.get("inventory", {}).get("rows", []) if "event_id" in r
        )
        or value.get("event_frontier", {}).get("preserved_event_ids", []),
    )
    fields.update(
        task_id="exp8161-arc-supervisor-frontier",
        title="V705 new live supervisor outcome frontier",
        historical_methods=dict(path=str(METHODS), sha256=scope.PINNED[str(METHODS)]),
        event_frontier=frontier,
        code_config_hashes=hashes,
        raw_shard_hashes=shards,
        checkpoint_references=[
            dict(path=p, sha256=h)
            for p, h in shards.items()
            if p.endswith("receipt_inventory.json")
        ],
        reader_receipt=dict(
            qualified_upstream_path=str(PRIOR),
            qualified_upstream_sha256=scope.PINNED[str(PRIOR)],
            original_frontier_sha256=scope.PINNED[str(INVENTORY)],
            current_code_hashes=hashes,
        ),
        historical_model_provenance=checked.get("prior", {}).get(
            "historical_model_provenance", value.get("historical_model_provenance", [])
        ),
        new_outcome_ready_score=int(
            bool(ready) and any(r["status"] == "completed" for r in value["new_event_rows"])
        ),
        solve_provenance="live_agent_self_discovery" if count else "not_applicable_no_solve_claim",
        cited_upstream_artifacts=[
            dict(
                path=str(p),
                sha256=scope.PINNED[str(p)],
                fields_imported=[
                    "reader_receipt",
                    "receipt_inventory",
                    "rows",
                    "finished_at",
                    "historical_required_failures",
                    "historical_model_provenance",
                ],
            )
            for p in (PRIOR, INVENTORY, SIDECAR)
        ],
    )
    if value.get("verdict_class") == "blocked" and value.get("gate_check_summary"):
        for row in value["gate_check_summary"]:
            row.update(
                check="authenticate_" + row["artifact_field"],
                upstream=row["upstream_id"],
                hash=row["sha256"],
                passed=False,
            )
        fields["honest_verdict"] = "complete_blocked_" + value["gate_check_summary"][0]["check"]
    return fields


def replay(value: dict[str, Any]) -> list[str]:
    """Independent reduction and rehashing prevent summaries from inventing work."""
    import tempfile

    errors = runner.replay(value)
    if value.get("task_id") != "exp8161-arc-supervisor-frontier":
        return errors
    expected = artifact_fields(value, {})
    errors.extend(k for k in expected if value.get(k) != expected[k])
    for label, digest in value["source_artifact_hashes"].items():
        path = Path(label)
        path = path if path.is_absolute() else scope.ROOT / path
        if not path.is_file() or sha256_file(path) != digest:
            errors.append("source_sha256:" + label)
    for row in value["validation_receipts"]:
        if "log_path" in row and (
            not Path(row["log_path"]).is_file()
            or sha256_file(Path(row["log_path"])) != row["log_sha256"]
        ):
            errors.append("validation_log_sha256")
    with tempfile.TemporaryDirectory(prefix="carnot-8161-cold-", dir="/tmp") as scratch:
        errors.extend(baseline.control_errors(value["reader_conformance_rows"], Path(scratch)))
    return errors


def terminal(candidate: Path, private: Path, durable: Path) -> dict[str, Any]:
    """Unchanged validators and an outside-checkout process check identical bytes."""
    report = runner.previous.terminal(candidate, private, durable)
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
                "import os,runpy,sys; os.chdir(sys.argv[1]); sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name='__main__')",
                str(private),
                str(scope.ROOT / CLI),
                "--cold-replay",
                str(candidate),
            ],
            expected_exit=0,
            expected_text=None,
            deadline_s=60,
            classification="required",
        ),
        private,
        durable,
    )
    report["reports"].append(row)
    report["passed"] = (
        report["passed"] and row["passed"] and not replay(json.loads(candidate.read_text()))
    )
    return report


def execute(output: Path, private: Path) -> int:
    """The qualified runner owns atomic publication after the frozen checks pass."""
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
