"""Read only new authenticated outcomes. Spec: REQ-REPORT-8120.

The qualified reader already checks applied redirects against wrapper and frame
evidence. Reusing it keeps private controls separate from natural observations.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from carnot.reporting import arc_supervisor_v701_evidence as baseline
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

runner = baseline.runner
MODULE = "python/carnot/reporting/arc_supervisor_v702_delta.py"
CLI = "scripts/experiments/experiment_8120_v702_arc_supervisor_delta.py"
PRIOR = baseline.scope.OUTPUT
INVENTORY = PRIOR.parent / "raw" / PRIOR.stem / "receipt_inventory.json"
SIDECAR = INVENTORY.parent / "terminal_reports.json"
HEALTH = (
    runner.ROOT
    / "results/raw/experiment_8120_v702_arc_supervisor_delta/failed_attempts/schema_missing/repository_health.json"
)
scope = SimpleNamespace(**vars(baseline.scope))
scope.__dict__.update(
    PRIOR=PRIOR,
    INVENTORY=INVENTORY,
    EXPERIMENT_ID=8120,
    PRIOR_ID=8107,
    MILESTONE="2026.10.702",
    RUN_DATE="20261004",
    MODULE=MODULE,
    CLI=CLI,
    OUTPUT=runner.ROOT / "results/experiment_8120_v702_arc_supervisor_delta.json",
    TEST="tests/python/test_arc_supervisor_delta_8120.py",
    ADDED=[MODULE, CLI],
    INCLUDE=",".join("*/" + p for p in (MODULE, CLI)),
    CONSUMERS=[baseline.scope.TEST, *baseline.scope.CONSUMERS],
    PINNED={
        str(PRIOR): "sha256:b10df04286afecfc5dd1c10725c400f32e8e8664307ac9ed21443b7e55b5c016",
        str(INVENTORY): "sha256:47f62b3b0ae318a7c22422bf3a77b9e9a4689e1b8759821b587f10ebe17e98d1",
        str(SIDECAR): "sha256:1e6c93c279dceb4e5232dcfc5169b97fb2b0fb87adc4155c178941898add7088",
        str(runner.REGISTRY): runner.PINNED[str(runner.REGISTRY)],
    },
)


def inputs() -> dict[str, Any]:
    """Exact qualified bytes authorize the frontier; absent bytes name a block."""
    checked = runner.inputs(scope)
    actual = sha256_file(SIDECAR) if SIDECAR.is_file() else "missing"
    sidecar = json.loads(SIDECAR.read_text()) if actual == scope.PINNED[str(SIDECAR)] else {}
    checks = checked["checks"]
    for field, expected, observed in (
        ("sha256", scope.PINNED[str(SIDECAR)], actual),
        ("primary_sha256", scope.PINNED[str(PRIOR)], sidecar.get("primary_sha256", "missing")),
        ("report.passed", True, sidecar.get("report", {}).get("passed", "missing")),
    ):
        checks.append(runner.previous.operand(SIDECAR, field, expected, observed, actual))
    for field, expected in dict(
        experiment_id=8107,
        required_checks_passed=True,
        supervisor_reader_ready_score=1,
        outcome_hashes=checked["inventory"].get("outcome_hashes", "missing"),
    ).items():
        checks.append(
            runner.previous.operand(
                PRIOR,
                field,
                expected,
                checked["prior"].get(field, "missing"),
                scope.PINNED[str(PRIOR)],
            )
        )
    hashes = checked["prior"].get("source_artifact_hashes", {})
    for label in (
        *baseline.baseline.AUTH_CODE,
        baseline.baseline.MODULE,
        baseline.baseline.CLI,
        baseline.MODULE,
        baseline.CLI,
    ):
        path = scope.ROOT / label
        actual = sha256_file(path) if path.is_file() else "missing"
        checks.append(
            runner.previous.operand(
                path, "sha256", hashes.get(label, "qualified_code_hash_required"), actual, actual
            )
        )
    for label in (
        *scope.ADDED,
        scope.TEST,
        *scope.CONSUMERS,
        *[".venv/bin/" + n for n in ("python", "pytest", "coverage", "ruff", "mypy")],
    ):
        path = scope.ROOT / label
        checks.append(
            runner.previous.operand(
                path, "is_file", True, path.is_file(), sha256_file(path) if path.is_file() else None
            )
        )
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
        if r["passed"] and r["artifact_field"] in {"sha256", "is_file"}
    }
    if HEALTH.is_file():
        checked["owned_repository_health"] = [
            dict(
                r,
                reused=True,
                reuse_source_path=str(HEALTH),
                reuse_source_sha256=sha256_file(HEALTH),
            )
            for r in json.loads(HEALTH.read_text())["checks"]
        ]
        checked["additional_source_hashes"][str(HEALTH)] = sha256_file(HEALTH)
    return checked


def commands(private: Path) -> list[dict[str, Any]]:
    """Freeze bounded private CLI routes and only the new statement denominator."""
    specs = [baseline.commands(private)[0], *runner.commands(private, scope)]
    specs = [s for s in specs if not s["name"].startswith("e2e_016")]
    if HEALTH.is_file():
        specs = [s for s in specs if s["name"] != "full_python_suite"]
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
    """The qualified reader skips old receipt hashes; games remain source groups."""
    value = baseline.scan(root, producers, checked, private, current_date=current_date)
    for row in value["rows"]:
        row["source_cluster_id"] = row.get("game") or row.get("producer_sha256")
    atomic_json(private / "primitive_rows.json", value)
    return value


def artifact_fields(value: dict[str, Any], checked: dict[str, Any]) -> dict[str, Any]:
    """Ready empty science earns no benefit, model invocation or solve credit."""
    fields = baseline.artifact_fields(value, checked)
    frontier = dict(
        upstream_id="exp8107-arc-supervisor-evidence",
        path=str(INVENTORY),
        sha256=scope.PINNED[str(INVENTORY)],
        producer_invocation_date=scope.RUN_DATE,
    )
    raw = scope.OUTPUT.parent / "raw" / scope.OUTPUT.stem
    fields.update(
        task_id="exp8120-arc-supervisor-delta",
        title="V702 authenticated supervisor delta",
        frontier=frontier,
        event_frontier=frontier,
        prior_frontier=dict(frontier, receipt_inventory=value.get("seen_receipt_hashes", {})),
        frontier_hashes=dict(
            prior_primary=scope.PINNED[str(PRIOR)],
            prior_inventory=scope.PINNED[str(INVENTORY)],
            prior_terminal=scope.PINNED[str(SIDECAR)],
            current_outcomes=canonical_hash(value.get("outcome_hashes", [])),
        ),
        arc_delta_ready_score=fields["supervisor_reader_ready_score"],
        independent_generalization_score=0,
        call_ledger=[],
        checkpoint_references=[
            dict(path=p, sha256=h, role="current_durable_primitive_rows")
            for p, h in value.get("source_artifact_hashes", {}).items()
            if Path(p).is_relative_to(raw) and p.endswith("/receipt_inventory.json")
        ],
        cited_upstream_artifacts=[
            dict(
                experiment_id=8107,
                path=str(p),
                sha256=scope.PINNED[str(p)],
                fields_imported=["receipt_inventory", "outcome_hashes", "finished_at"],
            )
            for p in (PRIOR, INVENTORY, SIDECAR)
        ],
    )
    return fields


def replay(value: dict[str, Any]) -> list[str]:
    """Recompute row reductions and retain the immutable frontier in cold replay."""
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
    if value.get("task_id") == "exp8120-arc-supervisor-delta":
        errors.extend(k for k in expected if k not in value)
    identities = [
        r["event_id"] for r in value["rows"] if r["status"] in {"completed", "censored", "unknown"}
    ]
    if len(identities) != len(set(identities)):
        errors.append("duplicate_event_identity")
    for row in value.get("checkpoint_references", []):
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            errors.append("checkpoint_sha256")
        elif json.loads(path.read_text()).get("rows") != value["rows"]:
            errors.append("checkpoint_rows")
    return errors


def terminal(candidate: Path, private: Path, durable: Path) -> dict[str, Any]:
    """Validate identical bytes and require normal cold replay outside the checkout."""
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
    report["replay_errors"] = replay(json.loads(candidate.read_text()))
    report["passed"] = report["passed"] and row["passed"] and not report["replay_errors"]
    return report


def execute(output: Path, private: Path) -> int:
    """The shared runner owns frozen checks, custody and atomic publication."""
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
