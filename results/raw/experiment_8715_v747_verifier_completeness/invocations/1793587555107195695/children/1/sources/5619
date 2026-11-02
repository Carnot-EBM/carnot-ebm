"""REQ-VERIFY-8694: bounded reconciliation publishes independently checked bytes."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v745_capstone_inputs as e
from carnot.reporting import v745_capstone_reduce as reduction
from carnot.reporting import v745_capstone_report as report
from carnot.reporting import v743_capstone as previous
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.gatemate_continuity_8400 import budget
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def measure(raw: Path, run_date: str) -> Json:
    """Freeze authority and inputs before reducing any admitted scientific rows."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    e.progress("preconditions_before", 0, 14)
    with tempfile.TemporaryDirectory(prefix="exp8694-", dir="/var/tmp") as directory:
        pre = resources(Path(directory))
        ready = bool(
            pre["disk_backed"]
            and pre["tools_available"]
            and pre["free_disk_bytes"] >= 2**30
            and pre["memory_available_bytes"] >= 2**30
        )
        pre.update(optional_devices_required=False, language_model_calls=0)
    p, contract = e.protocol(), e.methods.authority(e.ROOT)
    log = e.freeze(e.ROOT / "ops/conductor-log.md", raw / "sources")
    lines = Path(log["snapshot_path"]).read_text().splitlines()
    slots, refs, gates = [], [log], []
    by_id = {t["id"]: t for t in p["task_map_v745"]}
    for index, task in enumerate(p["task_map_v745"][:-1]):
        e.progress("slot_before_" + task["id"], index, 14 - index)
        row = e.resolve(task, raw / "primaries", lines)
        slots.append(row)
        refs.extend(
            [row["declared_reference"], row["primary_reference"], *row["evidence_references"]]
        )
        for gate in task.get("gated_on", []):
            upstream = by_id[gate["upstream"]]
            gates.append(
                dict(
                    e.observe(e.locate(upstream), upstream, gate["artifact_field"], gate["value"]),
                    downstream=task["id"],
                )
            )
        e.progress("slot_after_" + task["id"], index + 1, 13 - index)
    prior = {}
    for task in p["task_map_v745"]:
        for registration in task.get("prior_failures", []):
            identity = registration["experiment_id"]
            found = list((e.ROOT / "results").glob(f"experiment_{identity[3:7]}_*.json"))
            if len(found) == 1:
                ref = e.freeze(found[0], raw / "history")
                refs.append(ref)
                v = e.read(ref)
                auth = e.custody(found[0], dict(id=identity))
                prior[identity] = dict(
                    honest_verdict=v.get("honest_verdict"), terminal_sealed=auth["sealed"]
                )
    history_ref = dict(path=str(e.ROOT / e.HISTORY), sha256=e.HISTORY_PIN)
    e.read(history_ref)
    refs.append(e.freeze(Path(history_ref["path"]), raw / "history"))
    dependencies = [
        e.methods.PROTOCOL,
        e.methods.DESIGN,
        e.methods.ACTIVE,
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "ops/exclusion_manifest.yaml",
        "ops/e2e-test-plan.md",
        "ops/verifier_gaps.md",
        "research-references.md",
        "research-studying.md",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/verification/spec.md",
        "openspec/change-proposals/research-roadmap-v744-preserved-20261101.md",
        "docs/research-notes/v745-capstone.md",
        "scripts/experiment_template.py",
        "scripts/publication_gate.py",
        "scripts/summarize_artifact.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "ops/publication_gate_state.json",
        "docs/technical-report.md",
        "docs/arxiv-paper/main.tex",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/v743_capstone.py",
        "python/carnot/reporting/v745_contract_methods.py",
        "python/carnot/reporting/v745_private_inputs.py",
        "python/carnot/reporting/v709_execution.py",
        "python/carnot/reporting/spline_precision_study_8692.py",
        "python/carnot/reporting/arc_observer_transfer_8678.py",
    ]
    refs.extend(e.freeze(e.ROOT / name, raw / "sources") for name in dependencies)
    code = [e.freeze(e.ROOT / name, raw / "code") for name in e.OWNED + e.TESTS]
    plan = dict(
        owned=e.OWNED,
        tests=e.TESTS,
        serial_flags=["-n", "0", "-o", "addopts=", "--no-cov"],
        minimum_new_statement_coverage=100,
        measurement_cap_s=3600,
        task_cap_s=4800,
    )
    e.atomic_json(raw / "configuration.json", plan)
    code.append(e.reference(raw / "configuration.json"))
    work: Json = dict(
        raw=str(raw),
        invocation_id=raw.name,
        run_date=run_date,
        started_at_utc=wall,
        protocol=p,
        contract=contract,
        slots=slots,
        prior=prior,
        gate_rows=gates,
        preconditions=pre,
        preconditions_passed=ready,
        history_reference=history_ref,
        code_refs=code,
        source_refs=refs,
        historical_models={
            str(r["experiment_id"]): e.read(r["primary_reference"]).get(
                "historical_model_provenance", []
            )
            for r in slots
            if r["producer_executed"]
        },
    )
    closure: dict[str, Json] = {}
    for row in slots:
        if row["producer_executed"]:
            previous.e.closure(e.read(row["primary_reference"]), raw / "operands", closure)
    work["operand_refs"] = list(closure.values())
    e.atomic_json(
        raw / "premeasurement_seal.json",
        dict(code_refs=code, source_refs=refs, operand_refs=work["operand_refs"]),
    )
    e.progress("sources_inputs_frozen_before_reduction", 13, 1)
    work["reductions"] = reduction.reduce(work)
    publication = child(
        "publication_gate",
        [sys.executable, "-u", "scripts/publication_gate.py", "--json"],
        raw / "publication",
        deadline=60,
    )
    if not publication["passed"]:
        raise ValueError("publication_gate_execution")
    work["publication_receipt"] = publication
    work["publication"] = json.loads(Path(publication["stdout_path"]).read_bytes())
    work["external_checks"] = [
        dict(
            upstream=r["task_id"],
            check="qualified_producer",
            path=r["primary_reference"]["path"],
            sha256=r["primary_reference"]["sha256"],
            field="producer_qualified",
            operator="==",
            expected=True,
            observed=r["producer_qualified"],
            operand_state="stamped_disqualified"
            if r["flagged_adversarial"]
            else "producer_not_executed",
            passed=False,
        )
        for r in slots
        if not r["producer_qualified"]
    ]
    end = time.monotonic_ns()
    if end - start >= 3600_000_000_000:
        raise TimeoutError("measurement_deadline")
    work.update(
        start_ns=start,
        end_ns=end,
        duration_s=(end - start) / 1e9,
        ended_at_utc=datetime.now(UTC).isoformat(),
        phase_spans=[
            dict(
                phase="authenticate_freeze_reduce",
                start_ns=start,
                end_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
    )
    e.atomic_json(raw / "measurement.json", work)
    seal = dict(measurement=e.reference(raw / "measurement.json"), code_refs=code)
    e.atomic_json(raw.parents[1] / "seals" / (raw.name + ".json"), seal)
    (raw / "measurement.json").chmod(0o444)
    e.progress("measurement_complete", 13, 1)
    return work


def validate(raw: Path) -> list[Json]:
    """Reuse disjoint serial collection and keep consumer coverage separate from ownership."""
    with patch.object(previous, "e", e):
        return list(previous.validate(raw))


def replay(path: Path) -> None:
    """An independent seal prevents a candidate from replacing its own evidence."""
    value = json.loads(path.read_bytes())
    output = Path(value["publication_output"])
    anchor = output.parent / "raw" / e.NAME
    raw = anchor / "invocations" / value["invocation_id"]
    seal = e.read(e.reference(anchor / "seals" / (value["invocation_id"] + ".json")))
    if (
        seal["measurement"] != value["primitive_reference"]
        or Path(value["primitive_reference"]["path"]) != raw / "measurement.json"
    ):
        raise ValueError("independent_measurement_seal")
    work = e.read(value["primitive_reference"])
    if str(raw) != work["raw"] or seal["code_refs"] != work["code_refs"]:
        raise ValueError("independent_source_seal")
    for ref in work["source_refs"] + work["code_refs"] + work["operand_refs"]:
        e.verify(ref)
    if (
        work["protocol"] != e.protocol()
        or work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9
    ):
        raise ValueError("authority_or_clock_drift")
    validation = e.read(value["validation_reference"])
    if validation["receipts"] != value["validation_receipts"]:
        raise ValueError("validation_drift")
    for receipt in validation["receipts"] + [work["publication_receipt"]]:
        for stream in ("stdout", "stderr"):
            if (
                stream + "_path" in receipt
                and e.sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]
            ):
                raise ValueError("child_log_drift")
    if reduction.reduce(work) != work["reductions"]:
        raise ValueError("cold_reduction_drift")
    if report.build(work, validation["receipts"], output) != value:
        raise ValueError("report_drift")
    e.progress("cold_replay_passed", 14, 0)


def terminal(path: Path, raw: Path) -> Json:
    """Preserve actual scanner severity and require the unchanged terminal consumers."""
    with patch.object(previous, "e", e):
        return dict(previous.terminal(path, raw))


def run(output: Path, run_date: str, private: bool, failed_child: bool) -> int:
    """Only checked final bytes become visible; failed owned children remove readiness."""
    if (private or failed_child) and (
        output.absolute().is_relative_to(e.ROOT) or output.resolve().is_relative_to(e.ROOT)
    ):
        raise ValueError("private controls require output outside checkout")
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = measure(raw, run_date)
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
    value = report.build(work, receipts, output)
    publication = publish_primary(output, value, lambda path: terminal(path, raw / "terminal"))
    reader = reader_receipt(
        e.TASK,
        output.parent,
        field="reconciliation_ready_score",
        expected=value["reconciliation_ready_score"],
    )
    e.atomic_json(
        raw / "terminal_validation.json",
        dict(publication=publication, consumer=reader, passed=reader["passed"]),
    )
    e.progress("complete_" + value["verdict_class"], 14, 0)
    return 0 if value["required_checks_passed"] and reader["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Private controls must remain private through lexical paths and symlink aliases."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero", 0, 14)
    os.chdir(e.ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
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
        print(f"[exp8694] error={error}", file=sys.stderr, flush=True)
        return 1
