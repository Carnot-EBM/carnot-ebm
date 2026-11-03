"""REQ-REPORT-8755: freeze before numerical work and publish only cold-checked bytes."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, UTC
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

from coverage import Coverage
import yaml

from carnot.reporting import verifier_family_surface_8755 as m
from carnot.reporting import verifier_family_report_8755 as report
from carnot.reporting import verifier_family_analysis_8755 as analysis
from carnot.reporting import verifier_surface_runner_8754 as original
from carnot.reporting import v750_archive_runner as validation_original
from carnot.reporting import v749_contract_runner
from carnot.reporting.collection_contract import collect
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
validation_runner = namespace(validation_original, "family_validation")
validation_runner.a = m
terminal_runner = namespace(v749_contract_runner, "family_terminal")
terminal_runner.CLI = m.CLI
terminal = terminal_runner.terminal
validate = validation_runner.validate


def validation_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """New modules get strict types without legacy repository relaxations."""
    if name == "types":
        argv = [
            argv[0],
            "--config-file=/dev/null",
            "--strict",
            "--follow-imports=silent",
            "--ignore-missing-imports",
            *m.OWNED,
        ]
    return child(name, argv, logs, **kwargs)


validation_runner.child = validation_child


def validation_collect(
    private: Path, raw: Path, suites: list[str], owned: list[str], run_child: Callable[..., Json]
) -> tuple[list[Json], Json]:
    """Measure new files in owned children without profiling the unchanged scanners."""
    plan, receipt = collect(private, raw, suites, owned, run_child)
    for spec in plan:
        if spec["role"] == "owned":
            config = Path(spec["config"]["path"])
            include = "include=\n" + "\n".join("    " + str(m.ROOT / p) for p in m.OWNED) + "\n"
            config.write_text(config.read_text().replace("[run]\n", "[run]\n" + include))
            spec["config"] = m.reference(config)
    return plan, receipt


validation_runner.collect = validation_collect


def prepare(raw: Path, private: Path, upstream: Path, private_control: bool) -> Json:
    """Reuse method custody while binding this producer's code and exact node inventory."""
    work = original.prepare(raw, private, m.methods.PROTOCOL, private_control)
    admitted = m.inputs(upstream)
    work["input_ready"] = admitted["passed"]
    work["gate_check_summary"] += admitted["gate_check_summary"]
    work["references"] += [
        r
        for r in admitted["references"]
        if Path(r["path"]).is_file() and m.reference(Path(r["path"]))["sha256"] == r["sha256"]
    ]
    work["sources"] += [
        m.reference(m.ROOT / p)
        for p in m.OWNED
        + m.TESTS
        + [
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/v709_execution.py",
            "python/carnot/reporting/v746_retained_evidence.py",
            "python/carnot/reporting/verifier_surface_methods_8754.py",
            "python/carnot/reporting/verifier_surface_report_8754.py",
            "python/carnot/reporting/verifier_surface_runner_8754.py",
        ]
    ]
    manifest = yaml.safe_load((m.ROOT / "ops/exclusion_manifest.yaml").read_bytes())
    retired = any(
        str(r.get("experiment_id")) == "8755"
        for r in manifest["retired"] + manifest.get("retired_experiments", [])
        if isinstance(r, dict)
    )
    work["preconditions"]["exclusion_check_passed"] = not retired
    if retired:
        work["input_ready"] = False
        work["gate_check_summary"].append(
            dict(
                upstream=m.TASK,
                **m.reference(m.ROOT / "ops/exclusion_manifest.yaml"),
                check="task_not_retired",
                field="retired_experiment_ids",
                operator="not_contains",
                expected=8755,
                observed=8755,
                passed=False,
            )
        )
    work["upstream_path"] = str(upstream)
    if admitted["passed"]:
        work["canonical_workload"]["reference"], work["transformed_workload"]["reference"] = (
            admitted["references"][1:3]
        )
        work["workload_references"] += admitted["references"][1:3]
    health = m.ROOT / "results/raw" / m.NAME / "startup/full_python_suite.receipt.json"
    if health.is_file():
        receipt = m.load(m.reference(health))
        work["repository_health_reference"]["current_invocation_diagnostic"] = m.reference(health)
        work["references"] += [m.reference(health)] + [
            dict(path=receipt[s + "_path"], sha256=receipt[s + "_sha256"])
            for s in ["stdout", "stderr"]
        ]
    plan, receipt = collect(private, raw, m.TESTS + m.CONSUMERS, m.TESTS, child)
    nodes = [n for p in plan for n in p["nodes"]]
    atomic_json(
        raw / "premeasurement_nodes.json",
        dict(node_ids=nodes, node_ids_sha256=canonical_hash(nodes), receipt=receipt),
    )
    work["frozen_node_reference"] = m.reference(raw / "premeasurement_nodes.json")
    work["references"].append(work["frozen_node_reference"])
    work["frozen_at_utc"] = datetime.now(UTC).isoformat()
    atomic_json(raw / "frozen_inputs.json", work)
    return work


def replay(path: Path) -> bool:
    """Cold reconstruction rejects changed inputs, row attacks and rehashed aggregates."""
    try:
        value = json.loads(path.read_bytes())
        raw = Path(value["primitive_reference"]["path"]).parent
        seal = json.loads((raw / "independent_seal.json").read_bytes())
        if seal != {k: value[k] for k in ["primitive_reference", "validation_reference"]}:
            return False
        work, validation = (
            m.load(value["primitive_reference"]),
            m.load(value["validation_reference"]),
        )
        refs = (
            work["sources"]
            + work["references"]
            + work["workload_references"]
            + [work["measurement_reference"]]
            + validation.get("raw_references", [])
        )
        if any(m.reference(Path(r["path"]))["sha256"] != r["sha256"] for r in refs):
            return False
        if work["input_ready"]:
            admitted = m.inputs(Path(work["upstream_path"]))
            if not admitted["passed"] or not m.audit(
                admitted["workloads"], m.load(work["measurement_reference"])
            ):
                return False
        return bool(value == report.build(work, validation, Path(value["publication_output"])))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(argv: list[str] | None = None) -> int:
    """Private controls exercise real CLI paths but never grant production readiness."""
    m.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261103"], default="20261103")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--upstream", type=Path, default=m.UPSTREAM)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        m.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.resolve().is_relative_to(m.ROOT.resolve()):
        parser.error("private outputs must remain outside the checkout")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    tracer = Coverage(data_file=str(raw / "producer.sqlite"), config_file=False)
    if not args.private_control:
        tracer.start()
    with TemporaryDirectory(prefix="exp8755-private-") as directory:
        work = prepare(raw, Path(directory), args.upstream, args.private_control)
        work["numerical_started_at_utc"] = datetime.now(UTC).isoformat()
        measured = (
            m.measure(m.inputs(args.upstream)["workloads"], raw)
            if work["input_ready"] and all(work["preconditions"]["tools"].values())
            else dict(rows=[], call_cost_rows=[])
        )
        atomic_json(raw / "measurement.json", measured)
        work["measurement_reference"] = m.reference(raw / "measurement.json")
        work["row_audit_passed"] = bool(
            work["input_ready"] and m.audit(m.inputs(args.upstream)["workloads"], measured)
        )
        atomic_json(raw / "analysis.json", analysis.summarize(measured["rows"]))
        work["analysis_reference"] = m.reference(raw / "analysis.json")
        work["numerical_ended_at_utc"] = datetime.now(UTC).isoformat()
        if args.private_control:
            validation = dict(
                receipts=[dict(name="private_control", passed=False)],
                union=dict(passed=False),
                coverage=dict(complete=False),
                ended_at_utc=datetime.now(UTC).isoformat(),
            )
        else:
            m.progress("before_scoped_validation")
            validation = validate(raw, tracer, [])
            m.progress("after_scoped_validation")
        atomic_json(raw / "work.json", work)
        atomic_json(raw / "validation.json", validation)
        atomic_json(
            raw / "independent_seal.json",
            dict(
                primitive_reference=m.reference(raw / "work.json"),
                validation_reference=m.reference(raw / "validation.json"),
            ),
        )
        value = report.build(work, validation, output)
        m.progress("before_terminal_validation")
        published = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
        atomic_json(raw / "publication_receipt.json", published)
        atomic_json(
            raw / "reader_receipt.json",
            reader_receipt(
                m.TASK,
                output.parent,
                field="verifier_transfer_rows_ready_score",
                expected=value["verifier_transfer_rows_ready_score"],
            ),
        )
        m.progress("published_" + value["verdict_class"], len(measured["rows"]), 0)
    return 0
