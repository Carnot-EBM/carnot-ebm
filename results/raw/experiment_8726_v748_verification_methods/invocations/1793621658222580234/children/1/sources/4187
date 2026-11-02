"""REQ-VERIFY-8720: reuse bounded validation without reopening historical measurements."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, cast
from unittest.mock import patch

from carnot.reporting import arc_supervisor_frontier_8720 as e
from carnot.reporting import arc_decision_qualification_runner_8705 as retained
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.gatemate_continuity_8400 import budget
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
retained_validate, retained_terminal = retained.validate, retained.terminal


def measure(raw: Path, run_date: str) -> Json:
    """Check disk scratch and seal the inputs before reading the new identity delta."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    e.progress("preconditions_before")
    with TemporaryDirectory(prefix="arc8720-", dir="/var/tmp") as directory:
        pre = resources(Path(directory))
        pre["write_probe_passed"] = (
            Path(directory) / "write_probe"
        ).read_bytes() == b"private disk scratch"
    pre.update(optional_devices_required=False, language_model_calls=0)
    gates: list[Json] = []
    failures: list[Json] = []
    refs: list[Json] = []
    contract = e.authority.authority(e.ROOT)
    failures.extend(contract["gate_check_summary"])
    upstream: Json = {}
    before: Json = {}
    after: Json = {}
    rows: list[Json] = []
    path, field = e.ROOT / e.UPSTREAM, "arc_decision_ready_score"
    gate = e.original.observe(path, "exp8705-arc-decision-qualification", field, e.PIN)
    gates.append(gate)
    try:
        if not gate["passed"]:
            raise ValueError("closed_producer_unavailable")
        upstream, refs = e.authenticate()
        path, field = e.ROOT / e.BASELINE, "retained_ledger_bytes"
        baseline = e.read(dict(path=str(path), sha256=e.LEDGER_PIN))
        refs.append(e.reference(path))
        path, field = e.ROOT.joinpath(*e.DEFAULT_LEDGER_PARTS), "ledger_authority"
        ledger = json.loads(path.read_bytes())
        refs.append(e.reference(path))
        before = e.frontier(upstream, baseline["entries"])
        after = before
        e.progress("authenticate_complete_freeze_before_measurement", 0, len(ledger["entries"]))
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            dict(
                upstream="closed_decision_and_live_ledger",
                path=str(path),
                sha256=e.sha256_file(path) if path.is_file() else None,
                check="authenticated_frontier_operand",
                field=field,
                operator="==",
                expected="authenticated",
                observed=str(error),
                passed=False,
                operand_state="missing_file"
                if not path.is_file()
                else "changed_or_invalid_operand",
            )
        )
    if not (
        pre["disk_backed"]
        and pre["tools_available"]
        and pre["write_probe_passed"]
        and min(pre["free_disk_bytes"], pre["memory_available_bytes"]) >= 2**30
    ):
        failures.append(
            dict(
                upstream="host",
                path="/var/tmp",
                sha256=None,
                check="cpu_resources",
                field="disk_memory_tools",
                operator="==",
                expected=True,
                observed=pre,
                operand_state="resource_unavailable",
                passed=False,
            )
        )
    closure = set(
        e.OWNED
        + e.TESTS
        + [
            e.authority.DESIGN,
            e.authority.ACTIVE,
            "pyproject.toml",
            "tests/python/conftest.py",
            "openspec/capabilities/research-reporting/spec.md",
            "openspec/capabilities/verification/spec.md",
            "ops/exclusion_manifest.yaml",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "docs/research-notes/v747-arc-supervisor-frontier.md",
        ]
    )
    closure.update(
        str(Path(cast(str, m.__file__)).resolve().relative_to(e.ROOT))
        for m in list(sys.modules.values())
        if getattr(m, "__file__", None)
        and Path(cast(str, m.__file__)).suffix == ".py"
        and Path(cast(str, m.__file__)).is_file()
        and Path(cast(str, m.__file__)).resolve().is_relative_to(e.ROOT)
        and ".venv" not in Path(cast(str, m.__file__)).parts
    )
    source_refs = refs + contract["references"]
    for index, name in enumerate(sorted(closure)):
        source = e.ROOT / name
        target = raw / "source_closure" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_refs += [e.reference(source), e.reference(target)]
        if index % 25 == 0:
            e.progress("freeze_sources", index, len(closure) - index)
    if not failures:
        try:
            rows, after = e.delta(baseline, ledger, upstream)
            for row in rows:
                source_refs += [row["source_reference"], row["producer_reference"]]
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            failures.append(
                dict(
                    upstream="live_redirect_ledger",
                    path=str(path),
                    sha256=e.sha256_file(path),
                    check="new_outcome_authority",
                    field="new_events",
                    operator="==",
                    expected="authenticated",
                    observed=str(error),
                    passed=False,
                )
            )
    end = time.monotonic_ns()
    if (end - start) / 1e9 >= 3600:
        raise TimeoutError("measurement_deadline")
    work = dict(
        raw=str(raw),
        run_date=run_date,
        started_at_utc=wall,
        ended_at_utc=datetime.now(UTC).isoformat(),
        start_ns=start,
        end_ns=end,
        duration_s=(end - start) / 1e9,
        preconditions=pre,
        gates=gates,
        failures=failures,
        source_refs=source_refs,
        code_refs=[e.reference(e.ROOT / p) for p in e.OWNED + e.TESTS],
        upstream_refs=refs,
        authority=contract,
        frontier_before=before,
        frontier_after=after,
        new_event_rows=rows,
        episode_refs=[],
        historical_model_provenance=upstream.get("historical_model_provenance", {}),
        registry_precheck=upstream.get("registry_precheck", {}),
        policy_source_reference=e.reference(
            e.ROOT / "python/carnot/agentic/arc_trajectory_supervisor.py"
        ),
        phase_spans=[
            dict(
                phase="authenticate_freeze_delta",
                started_monotonic_ns=start,
                ended_monotonic_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
    )
    e.atomic_json(raw / "measurement.json", work)
    e.atomic_json(
        raw.parents[1] / "seals" / (raw.name + ".json"),
        dict(measurement=e.reference(raw / "measurement.json"), code_refs=work["code_refs"]),
    )
    e.progress("measurement_complete", len(rows), 0)
    return work


def validate(raw: Path) -> list[Json]:
    """Reuse original disjoint collection and keep consumer coverage separate."""
    with patch.object(retained, "e", e):
        return list(retained_validate(raw))


def terminal(path: Path, raw: Path) -> Json:
    """Unchanged cold replay and scanners validate the exact candidate bytes."""
    with patch.object(retained, "e", e):
        return dict(retained_terminal(path, raw))


def replay(path: Path) -> None:
    """Reuse the independent measurement seal and exact source/log checks."""
    with patch.object(retained, "e", e):
        retained.replay(path)


def run(output: Path, date: str, private: bool, failed_child: bool) -> int:
    """Preserve existing private-output guards and atomic publication semantics."""
    checkout = e.ROOT.resolve()
    if (private or failed_child) and (
        output.absolute().is_relative_to(checkout) or output.resolve().is_relative_to(checkout)
    ):
        raise ValueError("private controls require output outside checkout")
    output = output.resolve()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = measure(raw, date)
    receipts = (
        [
            dict(
                name="private_control_only",
                passed=True,
                scope="private_fixture_not_production_validation",
            )
        ]
        if private
        else validate(raw)
    )
    if failed_child:
        receipts.append(
            child(
                "failed_child",
                [sys.executable, "-c", "raise SystemExit(1)"],
                raw / "validation",
                deadline=10,
            )
        )
    e.atomic_json(raw / "validation.json", dict(receipts=receipts))
    value = e.build(work, receipts, output)
    publication = publish_primary(output, value, lambda path: terminal(path, raw / "terminal"))
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="arc_frontier_ready_score",
        expected=value["arc_frontier_ready_score"],
    )
    e.atomic_json(
        raw / "terminal_validation.json",
        dict(publication=publication, consumer=consumer, passed=consumer["passed"]),
    )
    e.progress(value["honest_verdict"], value["completed_count"], 0)
    return 0 if value["required_checks_passed"] and consumer["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Parse dates before measuring; controls stay outside the checkout."""
    e.progress("start_no_model_load")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261102")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--failed-child", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            replay(args.cold_replay)
            return 0
        if len(args.date) != 8:
            raise ValueError("calendar_date")
        datetime.strptime(args.date, "%Y%m%d")
        with budget(4800):
            return run(args.output.absolute(), args.date, args.private_control, args.failed_child)
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
        print(f"[exp8720] error={error}", file=sys.stderr, flush=True)
        return 1
