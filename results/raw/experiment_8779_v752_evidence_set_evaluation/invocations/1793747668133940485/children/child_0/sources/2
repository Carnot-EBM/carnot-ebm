"""REQ-VERIFY-8779: bounded input-only measurement and private serial validation."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from coverage import Coverage
import yaml

from carnot.reporting import evidence_set_evaluation_8779 as e
from carnot.reporting import evidence_set_methods_8768 as original
from carnot.reporting import evidence_evaluation_report_8779 as report
from carnot.reporting import evidence_qualification_runner_8777 as previous
from carnot.reporting import v746_retained_evidence as evidence
from carnot.reporting import v749_contract_runner as historical_terminal
from carnot.reporting.collection_contract import collect
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
terminal = namespace(historical_terminal, "evidence_evaluation_terminal")
terminal.CLI = e.CLI


def preconditions(private: Path) -> Json:
    """Check scratch, tools, retirement and CPU resource needs before any measurement."""
    private.mkdir(parents=True, exist_ok=True)
    probe = private / "writable"
    probe.write_text("private scratch")
    tools = all((e.ROOT / ".venv/bin" / p).is_file() for p in ["python", "pytest", "ruff", "mypy"])
    manifest = yaml.safe_load((e.ROOT / "ops/exclusion_manifest.yaml").read_bytes())
    retired = any(
        str(r.get("experiment_id")) == "8779"
        for key in ["retired", "retired_experiments"]
        for r in manifest.get(key, [])
        if isinstance(r, dict)
    )
    return dict(
        passed=tools and not retired and probe.read_text() == "private scratch",
        tools_available=tools,
        not_retired=not retired,
        private_scratch_writable=True,
        free_disk_bytes=shutil.disk_usage(private).free,
        resource_projection=dict(
            worlds=240,
            units=2880,
            arm_rows=8640,
            llm_calls=0,
            model_loads=0,
            maximum_calls_per_arm_unit=5,
            estimated_cpu_measurement_seconds=30,
            reserved_validation_seconds=900,
            memory_bytes=256 * 1024 * 1024,
        ),
    )


def validate(raw: Path, private: Path) -> Json:
    """Preserve exact node outcomes and actual unit/CLI SQLite before scratch cleanup."""
    suites = e.TESTS + original.CONSUMERS
    plan, collection = collect(private, raw, suites, e.TESTS, child)
    receipts, children, consumers = [collection], [], []
    sources = [evidence.reference(e.ROOT / p) for p in e.OWNED + e.TESTS + original.CONSUMERS]
    for spec in plan:
        config = Path(spec["config"]["path"])
        config.write_text(
            config.read_text().replace(
                "[run]\n",
                "[run]\ninclude=\n" + "\n".join("    " + str(e.ROOT / p) for p in e.OWNED) + "\n",
            )
        )
        spec["config"] = evidence.reference(config)
    atomic_json(raw / "frozen_test_plan.json", dict(plan=plan, collection_receipt=collection))
    for spec in plan:
        config = Path(spec["config"]["path"])
        receipt = child(
            spec["name"],
            ["env", "COVERAGE_RCFILE=" + str(config), *spec["argv"]],
            raw / "checks",
            deadline=600,
        )
        receipts.append(receipt)
        retained = evidence.retain(
            receipt,
            spec["nodes"],
            Path(spec["scratch"]),
            raw / "children" / spec["name"],
            sources,
            spec["role"],
        )
        children.append(retained)
        if spec["role"] == "consumer":
            consumers.append(retained)
    for name, argv in [
        ("ruff", [".venv/bin/ruff", "check", *e.OWNED, *e.TESTS]),
        ("format", [".venv/bin/ruff", "format", "--check", *e.OWNED, *e.TESTS]),
        (
            "types",
            [
                ".venv/bin/mypy",
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=silent",
                "--ignore-missing-imports",
                *e.OWNED,
            ],
        ),
        ("spec", [sys.executable, "scripts/check_spec_coverage.py", *e.TESTS]),
    ]:
        receipts.append(child(name, argv, raw / "checks", deadline=120))
    union = evidence.aggregate(children, [n for spec in plan for n in spec["nodes"]])
    consumer_union = evidence.aggregate(
        consumers, [n for spec in plan if spec["role"] == "consumer" for n in spec["nodes"]]
    )
    combined = evidence.combine(children, raw / "combined")
    cov = Coverage(data_file=combined["combined_reference"]["path"], config_file=False)
    cov.load()
    files: list[Json] = []
    for name in e.OWNED:
        _, statements, excluded, missing, _ = cov.analysis2(str(e.ROOT / name))
        files.append(
            dict(path=name, statements=len(statements), excluded=excluded, missing=missing)
        )
    count = sum(r["statements"] for r in files)
    missing_count = sum(len(r["missing"]) for r in files)
    coverage = dict(
        complete=count > 0 and missing_count == 0 and not any(r["excluded"] for r in files),
        percent=100 * (count - missing_count) / count,
        files=files,
        combined_reference=combined["combined_reference"],
    )
    refs = (
        children
        + [r for p in children for r in evidence.load(p)["sqlite_references"]]
        + [coverage["combined_reference"]]
    )
    refs += [
        dict(path=r[s + "_path"], sha256=r[s + "_sha256"])
        for r in receipts
        for s in ["stdout", "stderr"]
    ]
    refs += [evidence.reference(raw / "frozen_test_plan.json")] + [spec["config"] for spec in plan]
    return dict(
        receipts=receipts,
        union=union,
        original_node_union=consumer_union,
        coverage=coverage,
        raw_references=refs,
    )


def main(argv: list[str] | None = None) -> int:
    """Publish only checked terminal bytes, with real clocks and no model or training calls."""
    e.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261103"], default="20261103")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--upstream", type=Path, default=e.UPSTREAM)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--decide-input", type=Path)
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = report.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    if args.decide_input:
        if args.decisions is None:
            parser.error("--decisions is required with --decide-input")
        payloads = json.loads(
            args.decide_input.read_bytes(),
            object_pairs_hook=original.unique_object,
            parse_constant=original.reject_constant,
        )["rows"]
        atomic_json(
            args.decisions,
            dict(rows=e.measure(payloads), oracle_file_reads=0, feature_label_reads=0),
        )
        return 0
    output = args.output.absolute()
    if args.private_control and output.resolve().is_relative_to(e.ROOT.resolve()):
        parser.error("private outputs must remain outside the checkout")
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    started, start_ns = datetime.now(UTC).isoformat(), time.monotonic_ns()
    with TemporaryDirectory(prefix="exp8779-private-") as directory:
        private = Path(directory)
        pre = preconditions(private)
        e.progress("before_freeze")
        frozen = e.freeze(args.upstream)
        atomic_json(raw / "frozen.json", frozen)
        atomic_json(raw / "arm_inputs.json", dict(rows=[p["input"] for p in frozen["inputs"]]))
        frozen_at = datetime.now(UTC).isoformat()
        receipt = child(
            "input_only_decisions",
            [
                sys.executable,
                e.CLI,
                "--decide-input",
                str(raw / "arm_inputs.json"),
                "--decisions",
                str(raw / "decisions.json"),
            ],
            raw / "measurement",
            deadline=180,
        )
        observed = json.loads((raw / "decisions.json").read_bytes())
        pre["input_only_child_passed"] = (
            receipt["passed"]
            and observed["oracle_file_reads"] == 0
            and observed["feature_label_reads"] == 0
        )
        pre["passed"] = pre["passed"] and pre["input_only_child_passed"]
        measured_at = datetime.now(UTC).isoformat()
        e.progress("before_scoped_validation")
        validation = (
            previous.private_validation(raw) if args.private_control else validate(raw, private)
        )
        e.progress("after_scoped_validation")
        ended = datetime.now(UTC).isoformat()
        sources = [
            evidence.reference(e.ROOT / p)
            for p in e.OWNED
            + e.TESTS
            + original.OWNED
            + [
                "python/carnot/reporting/evidence_qualification_8777.py",
                "python/carnot/reporting/evidence_qualification_report_8777.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/v749_verifier_protocol.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
                "scripts/experiment_template.py",
                "ops/exclusion_manifest.yaml",
                "openspec/capabilities/research-reporting/spec.md",
                "openspec/capabilities/verification/spec.md",
                "docs/research-notes/v752-evidence-set-evaluation.md",
            ]
        ]
        health = e.ROOT / "results/raw" / e.NAME / "startup/full_python_suite.receipt.json"
        work: Json = dict(
            raw=str(raw),
            run_date=args.date,
            invocation_id=raw.name,
            upstream_path=str(args.upstream.absolute()),
            preconditions_checked=pre,
            frozen_reference=evidence.reference(raw / "frozen.json"),
            decisions_reference=evidence.reference(raw / "decisions.json"),
            sources=sources,
            shards=[
                evidence.reference(raw / name)
                for name in ["frozen.json", "arm_inputs.json", "decisions.json"]
            ],
            measurement_receipt=receipt,
            repository_health_reference=dict(
                reference=evidence.reference(health) if health.is_file() else None,
                full_suite_pass=False,
                scope="one bounded diagnostic; separate from scoped acceptance",
            ),
            started_at_utc=started,
            ended_at_utc=ended,
            duration_s=(time.monotonic_ns() - start_ns) / 1e9,
            phase_spans=[
                dict(phase=p, start_utc=a, end_utc=b)
                for p, a, b in [
                    ("freeze", started, frozen_at),
                    ("measure", frozen_at, measured_at),
                    ("validation", measured_at, ended),
                ]
            ],
        )
        work["shards"] += [
            dict(path=receipt[s + "_path"], sha256=receipt[s + "_sha256"])
            for s in ["stdout", "stderr"]
        ]
        atomic_json(raw / "work.json", work)
        atomic_json(raw / "validation.json", validation)
        value = report.build(work, validation, output)
        atomic_json(
            raw / "independent_seal.json",
            {
                k: value[k]
                for k in ["primitive_reference", "validation_reference", "frozen_input_reference"]
            },
        )
        e.progress("before_terminal_publication")
        published = publish_primary(
            output, value, lambda candidate: terminal.terminal(candidate, raw)
        )
        atomic_json(raw / "publication_receipt.json", published)
        atomic_json(
            raw / "reader_receipt.json",
            reader_receipt(
                e.TASK,
                output.parent,
                field="evidence_rows_ready_score",
                expected=value["evidence_rows_ready_score"],
            ),
        )
        e.progress("terminal_published_" + value["verdict_class"], len(value["rows"]), 0)
    return 0
