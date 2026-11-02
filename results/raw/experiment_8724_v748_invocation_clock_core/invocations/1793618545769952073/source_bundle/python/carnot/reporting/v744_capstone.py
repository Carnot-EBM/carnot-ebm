"""REQ-VERIFY-8680: bounded readers publish only independently checked final bytes."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v744_capstone_inputs as e
from carnot.reporting import v744_capstone_reduce as reduction
from carnot.reporting import v744_capstone_report as report
from carnot.reporting import v743_capstone as previous
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.gatemate_continuity_8400 import budget
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
HISTORY = "results/experiment_8666_v743_capstone.json"
HISTORY_PIN = "sha256:e5b711df34bea27c35d2879d110bdba208dc8c6c3fbc985cb13243d4934d61d5"


def measure(raw: Path, run_date: str) -> Json:
    """Authenticate CPU resources, exact authority and gate operands before reduction."""
    started = time.monotonic_ns()
    e.progress("preconditions_before", 0, 14)
    scratch = Path(tempfile.mkdtemp(prefix="exp8680-", dir="/var/tmp"))
    pre = resources(scratch)
    ready = (
        pre["disk_backed"]
        and pre["tools_available"]
        and pre["free_disk_bytes"] >= 2**30
        and pre["memory_available_bytes"] >= 1500 * 1024**2
        and (scratch / "write_probe").read_bytes() == b"private disk scratch"
    )
    pre.update(optional_cuda_required=False, optional_boards_required=False)
    p = e.protocol()
    contract = e.authority(raw / "authority", p)
    log = e.freeze(e.ROOT / "ops/conductor-log.md", raw / "sources")
    lines = Path(log["snapshot_path"]).read_text().splitlines()
    slots, values, refs, gates = [], {}, [log, *contract["references"]], []
    by_id = {t["id"]: t for t in p["task_map_v744"]}
    for index, task in enumerate(p["task_map_v744"][:-1]):
        e.progress("slot_before_" + task["id"], index, 14 - index)
        row = e.resolve(task, raw / "primaries", lines)
        slots.append(row)
        refs.extend(
            [row["declared_reference"], row["primary_reference"], *row["evidence_references"]]
        )
        if row["producer_executed"]:
            values[str(row["experiment_id"])] = e.read(row["primary_reference"])
        for gate in task.get("gated_on", []):
            observed = e.observe(
                e.locate(by_id[gate["upstream"]]),
                gate["upstream"],
                gate["artifact_field"],
                gate["value"],
            )
            gates.append(dict(observed, downstream=task["id"]))
        e.progress("slot_after_" + task["id"], index + 1, 13 - index)
    external = []
    for ref in p["protocol_operand_references"]:
        path = Path(ref["path"])
        digest = e.sha256_file(path) if path.is_file() else None
        if digest != ref["sha256"]:
            external.append(
                dict(
                    upstream="exp8667-contract-methods",
                    check="sealed_original_source",
                    path=str(path),
                    sha256=digest,
                    field="sha256",
                    operator="==",
                    expected=ref["sha256"],
                    observed=digest,
                    operand_state="changed_bytes" if path.is_file() else "missing_file",
                    passed=False,
                )
            )
    for number in ("8677", "8678"):
        if not next(r for r in slots if str(r["experiment_id"]) == number)["eligible"]:
            raise ValueError("unqualified_current_primitive:" + number)
    reduced, operands = reduction.reduce(values, p, raw / "operands")
    publication_receipt = child(
        "publication_gate",
        [sys.executable, "-u", "scripts/publication_gate.py", "--json"],
        raw / "publication",
        deadline=60,
    )
    if not publication_receipt["passed"]:
        raise ValueError("publication_gate_execution")
    publication = json.loads(Path(publication_receipt["stdout_path"]).read_bytes())
    history_ref = dict(path=str(e.ROOT / HISTORY), sha256=HISTORY_PIN)
    history = e.read(history_ref)
    prior_values = {}
    e.progress("freeze_prior_sources_before", 0, 14)
    for index, task in enumerate(p["task_map_v744"]):
        for registration in task.get("prior_failures", []):
            number = registration["experiment_id"].split("-", 1)[0][3:]
            paths = list((e.ROOT / "results").glob(f"experiment_{number}_*.json"))
            if len(paths) == 1:
                ref = e.freeze(paths[0], raw / "history")
                prior_values[registration["experiment_id"]] = e.read(ref)
                refs.append(ref)
        e.progress("freeze_prior_sources", index + 1, 13 - index)
    dependencies = [
        e.methods.PROTOCOL,
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/verification/spec.md",
        "openspec/change-proposals/research-roadmap-vNEXT.md",
        "openspec/change-proposals/research-roadmap-v743-preserved-20261101.md",
        "openspec/change-proposals/v743-methods-protocol.json",
        "python/carnot/reporting/v744_contract_methods.py",
        "python/carnot/reporting/v743_capstone_inputs.py",
        "docs/research-notes/v744-capstone.md",
        "scripts/publication_gate.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/v709_execution.py",
        "python/carnot/reporting/v743_capstone.py",
        "python/carnot/reporting/v743_capstone_reduce.py",
        "python/carnot/reporting/arc_observation_8677.py",
        "python/carnot/reporting/arc_observer_transfer_8678.py",
        "python/carnot/reporting/hardware_boundary_costs_8679.py",
        "scripts/experiment_template.py",
        "scripts/summarize_artifact.py",
        "ops/exclusion_manifest.yaml",
        "ops/verifier_gaps.md",
        "research-references.md",
        "research-studying.md",
        "ops/publication_gate_state.json",
        "docs/technical-report.md",
        "docs/arxiv-paper/main.tex",
    ]
    refs.extend(e.freeze(e.ROOT / name, raw / "sources") for name in dependencies)
    refs.extend(e.freeze(Path(ref["path"]), raw / "operands") for ref in operands)
    refs.append(e.freeze(Path(history_ref["path"]), raw / "history"))
    code = [e.freeze(e.ROOT / name, raw / "code") for name in e.OWNED + e.TESTS]
    ended = time.monotonic_ns()
    if ended - started >= 3600_000_000_000:
        raise TimeoutError("measurement_deadline")
    work: Json = dict(
        raw=str(raw),
        invocation_id=raw.name,
        run_date=run_date,
        start_ns=started,
        end_ns=ended,
        duration_s=(ended - started) / 1e9,
        phase_spans=[
            dict(
                phase="authenticate_and_reduce",
                start_ns=started,
                end_ns=ended,
                duration_s=(ended - started) / 1e9,
            )
        ],
        protocol=p,
        contract=contract,
        slots=slots,
        gate_rows=gates,
        external_checks=external,
        reductions=reduced,
        preconditions=pre,
        preconditions_passed=bool(ready),
        publication=publication,
        publication_receipt=publication_receipt,
        history={
            k: history[k]
            for k in ("preserved_nulls", "historical_quarantines", "preserved_v737_supervisor_null")
        },
        prior_values={
            k: {
                "honest_verdict": v.get("honest_verdict"),
                "verdict_class": v.get("verdict_class"),
                "flagged_adversarial": v.get("flagged_adversarial"),
            }
            for k, v in prior_values.items()
        },
        source_refs=refs,
        operand_refs=operands,
        code_refs=code,
        historical_models={n: v.get("historical_model_provenance", []) for n, v in values.items()},
    )
    e.atomic_json(raw / "measurement.json", work)
    seal = dict(measurement=e.reference(raw / "measurement.json"), code_refs=code)
    e.atomic_json(raw / "measurement_seal.json", seal)
    e.atomic_json(raw.parents[1] / "seals" / (raw.name + ".json"), seal)
    (raw / "measurement.json").chmod(0o444)
    e.progress("measurement_complete", 13, 1)
    return work


def validate(raw: Path) -> list[Json]:
    """Reuse the bounded collector; instrument consumers separately from owned code."""
    with patch.object(previous, "e", e):
        return list(previous.validate(raw))


def replay(path: Path) -> None:
    """Locate the independent seal outside the candidate's replaceable primitive envelope."""
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
    for ref in work["source_refs"] + work["code_refs"]:
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
    values = {
        str(row["experiment_id"]): e.read(row["primary_reference"])
        for row in work["slots"]
        if row["producer_executed"]
    }
    with tempfile.TemporaryDirectory(prefix="exp8680-replay-", dir="/var/tmp") as directory:
        reduced, _ = reduction.reduce(values, work["protocol"], Path(directory))
    if reduced != work["reductions"]:
        raise ValueError("cold_reduction_drift")
    if report.build(work, validation["receipts"], output) != value:
        raise ValueError("report_drift")
    e.progress("cold_replay_passed", 14, 0)


def terminal(path: Path, raw: Path) -> Json:
    """Keep actual scanner severity while using the unchanged final-byte consumers."""
    with patch.object(previous, "e", e):
        return dict(previous.terminal(path, raw))


def run(output: Path, run_date: str, private: bool, failed_child: bool) -> int:
    """Freeze one invocation; only an accepted final candidate becomes reader-visible."""
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
    """Private controls cannot write into the checkout or impersonate production checks."""
    e.progress("start_no_model_load_current_LLM_calls_0", 0, 14)
    os.chdir(e.ROOT)
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
        if (args.private_control or args.failed_child) and args.output.absolute().is_relative_to(
            e.ROOT
        ):
            raise ValueError("private controls require output outside checkout")
        with budget(4800):
            return run(args.output.absolute(), args.date, args.private_control, args.failed_child)
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
        print(f"[exp8680] error={error}", file=sys.stderr, flush=True)
        return 1
