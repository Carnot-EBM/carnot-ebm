"""Advance authenticated events without running games. REQ-REPORT-8202.

The qualified reader retains its original assertions. This adapter binds the
new custody frontier and removes file clocks from discovery authority.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

from carnot.reporting import arc_supervisor_v707_frontier as previous
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

runner = previous.runner
MODULE = "python/carnot/reporting/arc_supervisor_v708_frontier.py"
CLI = "scripts/experiments/experiment_8202_v708_arc_supervisor_frontier.py"
PRIOR = previous.scope.OUTPUT
INVENTORY = PRIOR.parent / "raw" / PRIOR.stem / "receipt_inventory.json"
SIDECAR = INVENTORY.parent / "terminal_reports.json"
STATE = runner.ROOT / "ops/arc_live_agent_state.json"
scope = SimpleNamespace(**vars(previous.scope))
scope.__dict__.update(
    EXPERIMENT_ID=8202,
    PRIOR_ID=8189,
    MILESTONE="2026.10.708",
    RUN_DATE="20261006",
    PRIOR=PRIOR,
    INVENTORY=INVENTORY,
    MODULE=MODULE,
    CLI=CLI,
    OUTPUT=runner.ROOT / "results/experiment_8202_v708_arc_supervisor_frontier.json",
    TEST="tests/python/test_arc_supervisor_frontier_8202.py",
    ADDED=[MODULE, CLI],
    INCLUDE=",".join("*/" + p for p in (MODULE, CLI)),
    CONSUMERS=[previous.scope.TEST, *previous.scope.CONSUMERS],
    APPLICABLE_E2E=["E2E-017"],
    PINNED={
        str(PRIOR): "sha256:bc298ca8b5e58f082e8cb714b8ffe1cf808432cbb4f54e5490fdc6f5468d75c0",
        str(INVENTORY): "sha256:f0ac50659c9c568823d68caac31c5031ae7dc3efd2bc812c87eccc6bc4f2338b",
        str(SIDECAR): "sha256:226f79a1fa1addb5968fffb278aef8b3f02b1961906d13186e848c0c070fb207",
        str(runner.REGISTRY): runner.PINNED[str(runner.REGISTRY)],
        str(previous.METHODS): previous.scope.PINNED[str(previous.METHODS)],
    },
)


@contextmanager
def bound() -> Iterator[None]:
    """Bind one synchronous adapter call and restore historical module authority."""
    names = dict(scope=scope, PRIOR=PRIOR, INVENTORY=INVENTORY, SIDECAR=SIDECAR, CLI=CLI)
    saved = {name: getattr(previous, name) for name in names}
    previous.__dict__.update(names)
    try:
        yield
    finally:
        previous.__dict__.update(saved)


def inputs() -> dict[str, Any]:
    """Check registry custody first, then the external state and actual host runtime."""
    print("[exp8202] phase=registry_precheck completed=0 pending=1", flush=True)
    registry_hash = sha256_file(scope.REGISTRY) if scope.REGISTRY.is_file() else "missing"
    print(
        f"[exp8202] phase=registry_authenticated completed=1 pending=0 hash={registry_hash} "
        f"passed={registry_hash == scope.PINNED[str(scope.REGISTRY)]}",
        flush=True,
    )
    with bound():
        checked = previous.inputs()
    print(
        f"[exp8202] phase=registry_checked completed={len(checked['registry_precheck'])} pending=0",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="carnot-8202-precheck-", dir="/tmp") as scratch:
        private = Path(scratch)
        probe = private / "writable"
        probe.write_text("storage precheck")
        operands = [
            (STATE, "live_state_is_file", True, STATE.is_file()),
            (Path(sys.executable), "python_runtime", True, sys.version_info >= (3, 12)),
            (
                private,
                "private_writable_storage",
                True,
                probe.read_text() == "storage precheck"
                and private.stat().st_mode & 0o077 == 0
                and shutil.disk_usage(private).free > 1048576,
            ),
        ]
        for path, field, expected, observed in operands:
            digest = sha256_file(path) if path.is_file() else None
            row = runner.previous.operand(path, field, expected, observed, digest)
            row.update(
                check="authenticate_" + field,
                upstream=row["upstream_id"],
                hash=digest,
                passed=expected == observed,
            )
            checked["checks"].append(row)
            if path.is_file():
                checked["additional_source_hashes"][str(path)] = digest
    checked["failures"] = [r for r in checked["checks"] if not r["passed"]]
    return checked


def commands(private: Path) -> list[dict[str, Any]]:
    """Reuse frozen owned argv, including direct script execution outside the checkout."""
    with bound():
        return previous.commands(private)


def scan(
    root: Path, producers: list[Path], checked: dict[str, Any], private: Path, *, current_date: str
) -> dict[str, Any]:
    """Use qualified identities and receipt hashes without filtering producer mtimes."""
    original = previous.baseline.scope.OUTPUT
    previous.baseline.scope.OUTPUT = scope.OUTPUT
    try:
        value = previous.baseline.scan(root, producers, checked, private, current_date=current_date)
    finally:
        previous.baseline.scope.OUTPUT = original
    for row in value["rows"]:
        row.update(
            condition="observational_applied_redirect",
            metric="resolved_by_levelup",
            numerator=int(row.get("resolved_by_levelup") is True),
            denominator=1,
            fired=True,
            helped=row.get("resolved_by_levelup") is True,
        )
    return value


def artifact_fields(value: dict[str, Any], checked: dict[str, Any]) -> dict[str, Any]:
    """Bind this frontier while keeping imported provenance apart from current calls."""
    with bound():
        fields = previous.artifact_fields(value, checked)
    arms: dict[str, Any] = {}
    counted: set[tuple[str, str]] = set()
    for row in value["new_event_rows"]:
        arm = arms.setdefault(
            row["arm"],
            dict(
                fired=0,
                helped=0,
                resolved_by_levelup=0,
                actions_to_levelup=[],
                stagnations_unredirected=0,
            ),
        )
        arm["fired"] += 1
        arm["helped"] += int(row["resolved_by_levelup"] is True)
        arm["resolved_by_levelup"] += int(row["resolved_by_levelup"] is True)
        arm["actions_to_levelup"].append(row["actions_to_levelup"])
        identity = (row["arm"], row["receipt_id"])
        if identity not in counted:
            arm["stagnations_unredirected"] += row.get("stagnations_unredirected") or 0
            counted.add(identity)
    fields.update(
        task_id="exp8202-arc-supervisor-frontier",
        title="V708 authenticated supervisor frontier",
        frontier_hashes=dict(
            prior_primary=scope.PINNED[str(PRIOR)],
            prior_inventory=scope.PINNED[str(INVENTORY)],
            current_outcomes=canonical_hash(value["new_event_rows"]),
        ),
        new_outcome_count=sum(r["resolved_by_levelup"] is True for r in value["new_event_rows"]),
        per_arm_results=arms,
        new_level_solves_claimed=False,
        solve_provenance="live_agent_self_discovery"
        if value["new_event_rows"]
        else "development_proxy",
        arc_evidence_ready_score=fields["supervisor_reader_ready_score"],
    )
    if value.get("verdict_class") == "null" and not fields["new_outcome_count"]:
        fields["honest_verdict"] = "complete_null_no_new_outcomes"
    return fields


def replay(value: dict[str, Any]) -> list[str]:
    """Recompute owned fields and verify primitive custody in a fresh process."""
    errors = runner.replay(value)
    if value.get("task_id") != "exp8202-arc-supervisor-frontier":
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
    with tempfile.TemporaryDirectory(prefix="carnot-8202-cold-", dir="/tmp") as scratch:
        errors.extend(
            previous.baseline.control_errors(value["reader_conformance_rows"], Path(scratch))
        )
    return errors


def terminal(candidate: Path, private: Path, durable: Path) -> dict[str, Any]:
    """Validate identical bytes with the existing adversarial and row validators."""
    with bound():
        report = previous.terminal(candidate, private, durable)
    report["passed"] = report["passed"] and not replay(json.loads(candidate.read_text()))
    return report


def execute(output: Path, private: Path) -> int:
    """Reuse checked primary publication after all frozen owned checks finish."""
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
