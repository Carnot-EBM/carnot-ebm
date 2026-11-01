"""REQ-VERIFY-8691: seal one finite reader invocation before validation and publication."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, UTC
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import arc_censoring_decision_8691 as e
from carnot.reporting import v743_capstone as validation
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.gatemate_continuity_8400 import budget
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def measure(raw: Path, run_date: str) -> Json:
    """Read only authenticated upstream receipts; optional devices cannot veto CPU work."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    e.progress("preconditions_before", 0, 16)
    with tempfile.TemporaryDirectory(prefix="arc8691-", dir="/var/tmp") as directory:
        pre = resources(Path(directory))
    pre.update(optional_devices_required=False, language_model_calls=0)
    ready = (
        pre["disk_backed"]
        and pre["tools_available"]
        and pre["free_disk_bytes"] >= 2**30
        and pre["memory_available_bytes"] >= 2**30
    )
    episodes: list[Json] = []
    gates: list[Json] = []
    refs: list[Json] = []
    failures: list[Json] = []
    try:
        episodes, gates, refs = e.authenticate()
        failures.extend(r for r in gates if not r["passed"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            dict(
                upstream="exp8677/exp8678",
                check="producer_authentication",
                path=str(e.ROOT / "results"),
                sha256=None,
                field="original_bytes_and_terminal_validation",
                operator="==",
                expected="qualified",
                observed=str(error),
                operand_state="missing_file"
                if isinstance(error, FileNotFoundError)
                else "changed_or_invalid_operand",
                passed=False,
            )
        )
    if not ready:
        failures.append(
            dict(
                upstream="host",
                check="cpu_resources",
                path="/var/tmp",
                sha256=None,
                field="disk_memory_tools",
                operator="==",
                expected=True,
                observed=pre,
                operand_state="resource_unavailable",
                passed=False,
            )
        )
    if failures:
        episodes = []
    e.progress("inputs_authenticated_before_reduction", len(episodes), 16 - len(episodes))
    source_refs = refs.copy()
    source_refs.extend(
        e.reference(p) for p in raw.parents[1].glob("invocations/*/failure_summary.json")
    )
    dependencies = [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "ops/e2e-test-plan.md",
        "ops/exclusion_manifest.yaml",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/verification/spec.md",
        "scripts/experiment_template.py",
        "python/carnot/reporting/primary_publication.py",
        "openspec/change-proposals/research-roadmap-vNEXT.md",
        "openspec/change-proposals/research-roadmap-v744-preserved-20261101.md",
        "openspec/change-proposals/v744-methods-protocol.json",
        "research-references.md",
        "python/carnot/agentic/arc_competition_agent.py",
        "python/carnot/agentic/arc_solver_kit.py",
        "python/carnot/agentic/arc_trajectory_supervisor.py",
        "ops/arc_solve_registry.yaml",
        "python/carnot/reporting/v744_capstone.py",
        "python/carnot/reporting/v743_capstone.py",
        "python/carnot/reporting/v709_execution.py",
        "docs/research-notes/v745-arc-censoring-decision.md",
    ]
    code_refs = [e.reference(e.ROOT / p) for p in e.OWNED + e.TESTS]
    for name in dependencies + e.OWNED + e.TESTS:
        target = raw / "source_closure" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(e.ROOT / name, target)
        source_refs.extend([e.reference(e.ROOT / name), e.reference(target)])
    episode_refs = []
    for index, episode in enumerate(episodes):
        path = raw / "episodes" / f"{index:02}.json"
        e.atomic_json(
            path,
            {
                k: episode[k]
                for k in ("unit", "actions", "delivery_rows", "reset_provenance", "observer_rows")
            },
        )
        episode_refs.append(e.reference(path))
    e.reduce(episodes)
    end = time.monotonic_ns()
    if (end - start) / 1e9 >= 3600:
        raise TimeoutError("measurement_deadline")
    upstream = (
        e.read(
            next(
                r
                for r in refs
                if r["path"].endswith("experiment_8678_v744_arc_observer_transfer.json")
            )
        )
        if episodes
        else {}
    )
    work: Json = dict(
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
        episode_refs=episode_refs,
        source_refs=source_refs,
        code_refs=code_refs,
        upstream_refs=refs,
        historical_model_provenance=upstream.get("historical_model_provenance", {}),
        registry_precheck=upstream.get("registry_precheck", {}),
        policy_source_reference=e.reference(
            e.ROOT / "python/carnot/agentic/arc_trajectory_supervisor.py"
        ),
        phase_spans=[
            dict(
                phase="authenticate_and_reduce",
                started_monotonic_ns=start,
                ended_monotonic_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
    )
    e.atomic_json(raw / "measurement.json", work)
    seal = dict(measurement=e.reference(raw / "measurement.json"), code_refs=code_refs)
    e.atomic_json(raw / "measurement_seal.json", seal)
    e.atomic_json(raw.parents[1] / "seals" / (raw.name + ".json"), seal)
    e.progress("measurement_complete", len(episodes), 0)
    return work


def validate(raw: Path) -> list[Json]:
    """Reuse disjoint collection, separate consumer coverage and retained SQLite custody."""
    with patch.object(validation, "e", e):
        return list(validation.validate(raw))


def terminal(path: Path, raw: Path) -> Json:
    """Preserve unchanged scanners and their actual severity on final candidate bytes."""
    with patch.object(validation, "e", e):
        return dict(validation.terminal(path, raw))


def replay(path: Path) -> None:
    """An independently located seal rejects a rehashed candidate or local replacement."""
    value = json.loads(path.read_bytes())
    output = Path(value["publication_output"])
    anchor = output.parent / "raw" / e.NAME
    raw = anchor / "invocations" / value["invocation_id"]
    seal = e.read(e.reference(anchor / "seals" / (raw.name + ".json")))
    if (
        seal["measurement"] != value["primitive_reference"]
        or Path(value["primitive_reference"]["path"]) != raw / "measurement.json"
    ):
        raise ValueError("independent_measurement_seal")
    work = e.read(value["primitive_reference"])
    if work["raw"] != str(raw) or work["code_refs"] != seal["code_refs"]:
        raise ValueError("independent_source_seal")
    for ref in work["source_refs"] + work["code_refs"] + work["episode_refs"]:
        if e.sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("operand_changed_bytes")
    if work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9:
        raise ValueError("clock_drift")
    refs = e.read(value["validation_reference"])
    if refs["receipts"] != value["validation_receipts"]:
        raise ValueError("validation_drift")
    for receipt in refs["receipts"]:
        for stream in ("stdout", "stderr"):
            if (
                stream + "_path" in receipt
                and e.sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]
            ):
                raise ValueError("child_log_drift")
    if e.build(work, refs["receipts"], output) != value:
        raise ValueError("report_drift_false_promotion")
    e.progress("cold_replay_passed", 16, 0)


def run(output: Path, date: str, private: bool, failed_child: bool) -> int:
    """Expose only bytes accepted by cold replay and unchanged final-byte validators."""
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
        field="arc_decision_ready_score",
        expected=value["arc_decision_ready_score"],
    )
    e.atomic_json(
        raw / "terminal_validation.json",
        dict(publication=publication, consumer=consumer, passed=consumer["passed"]),
    )
    e.progress(
        "complete_" + value["verdict_class"],
        value["completed_count"],
        16 - value["completed_count"],
    )
    return 0 if value["required_checks_passed"] and consumer["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Private controls stay outside the checkout and cannot impersonate production validation."""
    e.progress("start_no_model_load", 0, 16)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261101")
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
        print(f"[exp8691] error={error}", file=sys.stderr, flush=True)
        return 1
