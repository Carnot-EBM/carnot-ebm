"""REQ-VERIFY-8666: bounded validation precedes exact-byte publication."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import v743_capstone_inputs as e
from carnot.reporting import v743_capstone_reduce as reduction
from carnot.reporting import v743_capstone_report as report
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.current_validator_8580 import xml_nodes
from carnot.reporting.gatemate_continuity_8400 import budget
from carnot.reporting.lean_runner_8608 import E2E, COMMON
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def measure(raw: Path, run_date: str) -> Json:
    """Optional devices cannot suppress independent CPU reconciliation."""
    start = time.monotonic_ns()
    e.progress("preconditions_before", 0, 14)
    scratch = Path(tempfile.mkdtemp(prefix="exp8666-", dir="/var/tmp"))
    pre = resources(scratch)
    ready = (
        pre["disk_backed"]
        and pre["tools_available"]
        and pre["free_disk_bytes"] >= 1024**3
        and pre["memory_available_bytes"] >= 1500 * 1024**2
        and (scratch / "write_probe").read_bytes() == b"private disk scratch"
    )
    pre.update(optional_cuda_required=False, optional_boards_required=False)
    protocol = e.methods.protocol()
    contract = e.authority(raw / "original_authority")
    log = e.freeze(e.ROOT / "ops/conductor-log.md", raw / "sources")
    lines = Path(log["snapshot_path"]).read_text().splitlines()
    slots, values, refs, gate_rows = [], {}, [log], []
    by_id = {t["id"]: t for t in protocol["task_map_v743"]}
    for index, task in enumerate(protocol["task_map_v743"][:-1]):
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
            observed["downstream"] = task["id"]
            gate_rows.append(observed)
        if not row["declared_reference"]["exists"]:
            gate_rows.append(
                e.observe(
                    Path(row["declared_reference"]["path"]),
                    task["id"],
                    "declared_producer_exists",
                    True,
                )
            )
        e.progress("slot_after_" + task["id"], index + 1, 13 - index)
    reductions, unions, operands = reduction.reduce(values, protocol, raw / "operands")
    pub = child(
        "publication_gate",
        [sys.executable, "-u", "scripts/publication_gate.py", "--json"],
        raw / "publication",
        deadline=60,
    )
    if not pub["passed"]:
        raise ValueError("publication_gate_execution")
    publication = json.loads(Path(pub["stdout_path"]).read_bytes())
    history_ref = e.freeze(e.ROOT / "results/experiment_8652_v742_capstone.json", raw / "sources")
    history = e.read(history_ref)
    for name in [
        e.methods.PROTOCOL,
        e.methods.DESIGN,
        e.methods.ACTIVE,
        "results/experiment_8639_v741_capstone.json",
        "ops/exclusion_manifest.yaml",
        "scripts/publication_gate.py",
        "scripts/conductor_gates.py",
        "scripts/recurring_blocker_ledger.py",
        "scripts/experiment_template.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/roadmap_contract.py",
        "research-references.md",
        "research-studying.md",
        "openspec/change-proposals/research-roadmap-v742-preserved-20261018.md",
        "ops/north-star.md",
        "ops/publication_gate_state.json",
    ]:
        refs.append(e.freeze(e.ROOT / name, raw / "sources"))
    refs.append(history_ref)
    for task in protocol["task_map_v743"]:
        for prior in task.get("prior_failures", []):
            number = prior["experiment_id"].split("-", 1)[0][3:]
            for path in (e.ROOT / "results").glob(f"experiment_{number}_*.json"):
                refs.append(e.freeze(path, raw / "history"))
    refs.extend(e.freeze(Path(r["path"]), raw / "operands") for r in operands)
    end = time.monotonic_ns()
    if end - start >= 3600_000_000_000:
        raise TimeoutError("measurement_deadline")
    work = dict(
        raw=str(raw),
        run_date=run_date,
        start_ns=start,
        end_ns=end,
        duration_s=(end - start) / 1e9,
        phase_spans=[
            dict(
                phase="authenticate_and_cold_reduce",
                start_ns=start,
                end_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
        protocol=protocol,
        contract=contract,
        slots=slots,
        gate_rows=gate_rows,
        reductions=reductions,
        unions=unions,
        preconditions=pre,
        preconditions_passed=bool(ready),
        publication=publication,
        publication_receipt=pub,
        history=history,
        history_reference=history_ref,
        source_refs=refs,
        operand_refs=operands,
        code_refs=[e.reference(e.ROOT / p) for p in e.OWNED + e.TESTS],
        historical_models={k: v.get("historical_model_provenance", []) for k, v in values.items()},
    )
    e.atomic_json(raw / "measurement.json", work)
    e.atomic_json(
        raw / "measurement_seal.json",
        dict(measurement=e.reference(raw / "measurement.json"), code_refs=work["code_refs"]),
    )
    e.progress("measurement_complete", 13, 1)
    return work


def validate(raw: Path) -> list[Json]:
    """Preserve disjoint original nodes and raw child SQLite before combining copies."""
    scratch = Path(tempfile.mkdtemp(prefix="exp8666-validation-", dir="/var/tmp"))
    configs = {}
    for role, include in [
        ("owned", [str(e.ROOT / p) for p in e.OWNED]),
        ("consumers", [str(e.ROOT / "python/carnot/*"), str(e.ROOT / "scripts/*")]),
    ]:
        folder = scratch / role
        folder.mkdir()
        config = folder / "coverage.ini"
        config.write_text(
            "[run]\nparallel=true\npatch=subprocess\ndata_file="
            + str(folder / ".coverage")
            + "\ninclude=\n"
            + "".join("    " + p + "\n" for p in include)
            + "[report]\nexclude_lines=\n"
        )
        configs[role] = config
    collection = child(
        "collect_once",
        [sys.executable, "-m", "pytest", *COMMON, "--collect-only", "-q", *e.TESTS, *E2E],
        raw / "validation",
        deadline=90,
    )
    nodes = [
        s.strip()
        for s in Path(collection["stdout_path"]).read_text().splitlines()
        if s.startswith("tests/") and "::" in s
    ]
    receipts, outcomes = [collection], []
    for index, suite in enumerate([*e.TESTS, *E2E]):
        selected = [n for n in nodes if n == suite or n.startswith(suite + "::")]
        if not selected:
            receipts.append(dict(name=f"empty_shard_{index}", passed=False))
            continue
        role = "owned" if suite in e.TESTS else "consumers"
        config = configs[role]
        xml = raw / "validation" / f"nodes-{index}.xml"
        receipt = child(
            f"tests_{index}",
            [
                "env",
                "COVERAGE_RCFILE=" + str(config),
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *COMMON,
                "-q",
                *selected,
                "--junitxml=" + str(xml),
                "--basetemp=" + str(scratch / f"suite-{index}"),
            ],
            raw / "validation",
            deadline=600,
        )
        receipt["coverage_role"] = role
        receipts.append(receipt)
        if receipt["passed"]:
            outcomes.extend(xml_nodes(xml))
    receipts.append(
        dict(
            name="original_disjoint_union",
            passed=bool(nodes)
            and len(nodes) == len(set(nodes))
            and sorted(nodes) == sorted(outcomes),
            original_node_ids=nodes,
            executed_node_ids=outcomes,
        )
    )
    shards = []
    for role, config in configs.items():
        copies = config.parent / "copies"
        copies.mkdir()
        for source in config.parent.glob(".coverage.*"):
            target = raw / "original_child_coverage" / role / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            shutil.copyfile(target, copies / target.name)
            shards.append(dict(role=role, reference=e.reference(target)))
    receipts.append(
        dict(
            name="preserved_raw_child_sqlite",
            passed=bool(shards),
            shards=shards,
            source_hashes=[e.reference(e.ROOT / p) for p in e.OWNED],
        )
    )
    config = configs["owned"]
    plan = [
        (
            "coverage_combine",
            [
                sys.executable,
                "-m",
                "coverage",
                "combine",
                "--rcfile=" + str(config),
                "--keep",
                str(config.parent / "copies"),
            ],
        ),
        (
            "coverage_report",
            [
                sys.executable,
                "-m",
                "coverage",
                "report",
                "--rcfile=" + str(config),
                "--fail-under=100",
                "--show-missing",
            ],
        ),
        (
            "coverage_json",
            [
                sys.executable,
                "-m",
                "coverage",
                "json",
                "--rcfile=" + str(config),
                "-o",
                str(raw / "coverage.json"),
            ],
        ),
        ("ruff_check", [str(e.ROOT / ".venv/bin/ruff"), "check", *e.OWNED, *e.TESTS]),
        ("ruff_format", [str(e.ROOT / ".venv/bin/ruff"), "format", "--check", *e.OWNED, *e.TESTS]),
        (
            "strict_mypy",
            [
                str(e.ROOT / ".venv/bin/mypy"),
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=skip",
                "--ignore-missing-imports",
                *e.OWNED,
            ],
        ),
        ("spec_coverage", [sys.executable, "scripts/check_spec_coverage.py", "--files", *e.TESTS]),
    ]
    receipts.extend(child(name, argv, raw / "validation", deadline=120) for name, argv in plan)
    return receipts


def replay(path: Path) -> None:
    """Rebuild the report from original operands rather than trusting rehashed aggregates."""
    value = json.loads(path.read_bytes())
    work = e.read(value["primitive_reference"])
    raw = Path(work["raw"])
    seal = e.read(e.reference(raw / "measurement_seal.json"))
    if (
        seal["measurement"] != value["primitive_reference"]
        or seal["code_refs"] != work["code_refs"]
        or Path(value["primitive_reference"]["path"]) != raw / "measurement.json"
    ):
        raise ValueError("independent_measurement_seal")
    for ref in work["source_refs"] + work["code_refs"]:
        e.verify(ref)
    if (
        work["protocol"] != e.methods.protocol()
        or work["contract"] != e.authority(Path(work["contract"]["authority_root"]))
        or work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9
    ):
        raise ValueError("authority_or_clock_drift")
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
    values = {
        str(row["experiment_id"]): e.read(row["primary_reference"])
        for row in work["slots"]
        if row["producer_executed"]
    }
    reduced, unions, _ = reduction.reduce(values, work["protocol"], raw / "replay_operands")
    if reduced != work["reductions"] or unions != work["unions"]:
        raise ValueError("cold_reduction_drift")
    expected = report.build(work, refs["receipts"], Path(value["publication_output"]))
    if expected != value:
        raise ValueError("report_drift")
    e.progress("cold_replay_passed", 14, 0)


def terminal(path: Path, raw: Path) -> Json:
    """Keep the unchanged scanners' actual severity on the candidate's final bytes."""
    receipts = [
        child(name, [sys.executable, "-u", *argv], raw, deadline=240)
        for name, argv in [
            ("cold_replay", [e.CLI, "--cold-replay", str(path)]),
            ("adversarial", ["scripts/adversarial_verify.py", "--json", str(path)]),
            ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)]),
        ]
    ]
    findings = json.loads(Path(receipts[1]["stdout_path"]).read_bytes())
    return dict(
        passed=all(r["passed"] for r in receipts) and findings["flagged_count"] == 0,
        receipts=receipts,
        actual_findings=findings,
    )


def run(output: Path, run_date: str, private: bool, failed_child: bool) -> int:
    """Use one immutable invocation and expose only bytes accepted by final readers."""
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
    publication = publish_primary(
        output, value, lambda path: terminal(path, raw / "terminal_checks")
    )
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
    """Private controls cannot write into the checkout or impersonate measured validation."""
    e.progress("start_no_model_load_current_LLM_calls_0", 0, 14)
    os.chdir(e.ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261019")
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
        print(f"[exp8666] error={error}", file=sys.stderr, flush=True)
        return 1
