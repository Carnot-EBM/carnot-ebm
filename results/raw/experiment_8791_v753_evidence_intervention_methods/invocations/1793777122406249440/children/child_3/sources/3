"""REQ-VERIFY-8791: private model-free qualification, retained child evidence and atomic publication."""

import argparse
from datetime import datetime, UTC
import importlib
import json
from pathlib import Path
import resource
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

import yaml

from carnot.reporting import evidence_intervention_methods_8791 as m
from carnot.reporting import evidence_intervention_corpus_8791 as c
from carnot.reporting import evidence_intervention_report_8791 as report
from carnot.reporting import evidence_evaluation_runner_8779 as base
from carnot.reporting import evidence_qualification_runner_8777 as previous
from carnot.reporting import evidence_set_methods_8768 as original
from carnot.reporting import v749_contract_runner as historical_terminal
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v746_retained_evidence import reference
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
validator = namespace(base, "intervention_validation")
validator.e = m
terminal = namespace(historical_terminal, "intervention_terminal")
terminal.CLI = m.CLI


def preconditions(private: Path) -> Json:
    """Measure a small construction probe before science; reserve validation time independently."""
    private.mkdir(parents=True, exist_ok=True)
    probe = private / "writable"
    probe.write_text("private8791")
    dependencies = [importlib.import_module(name).__name__ for name in ["coverage", "yaml"]]
    tools = all(
        (m.ROOT / ".venv/bin" / name).is_file() for name in ["python", "pytest", "ruff", "mypy"]
    )
    manifest = yaml.safe_load((m.ROOT / "ops/exclusion_manifest.yaml").read_bytes())
    retired = any(
        str(row.get("experiment_id")) in {"8791", m.TASK}
        for key in ["retired", "retired_experiments"]
        for row in manifest.get(key, [])
        if isinstance(row, dict)
    )
    start = time.monotonic_ns()
    sample = c.units(((0, 1), (0, 2)), "projection-probe")
    seconds = (time.monotonic_ns() - start) / 1e9
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    free = shutil.disk_usage(private).free
    return dict(
        passed=tools
        and not retired
        and probe.read_text() == "private8791"
        and free > 128 * 1024 * 1024,
        dependencies=dependencies,
        tools_available=tools,
        not_retired=not retired,
        private_scratch_writable=True,
        free_disk_bytes=free,
        resource_projection=dict(
            measured_probe_seconds=seconds,
            probe_units=len(sample),
            measured_peak_rss_bytes=rss,
            projected_construction_seconds=seconds * 120 * 8,
            projected_memory_bytes=rss + 128 * 1024 * 1024,
            reserved_validation_seconds=900,
            frozen_worlds=120,
            measured_worlds=1,
            llm_calls=0,
            model_loads=0,
            scientific_acceptance_separate=True,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """A thin CLI seals methods before input-only decisions and publishes only validated terminal bytes."""
    m.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261104"], default="20261104")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--upstream-method", type=Path, default=m.UPSTREAM_METHOD)
    parser.add_argument("--upstream-rows", type=Path, default=m.UPSTREAM_ROWS)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--decide-input", type=Path)
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = report.replay(args.cold_replay)
        m.progress("replay_passed" if passed else "replay_rejected")
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
            dict(rows=m.measure(payloads), oracle_file_reads=0, feature_label_reads=0),
        )
        return 0
    output = args.output.absolute()
    if args.private_control and output.resolve().is_relative_to(m.ROOT.resolve()):
        parser.error("private outputs must remain outside the checkout")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    started, start_ns = datetime.now(UTC).isoformat(), time.monotonic_ns()
    with TemporaryDirectory(prefix="exp8791-private-") as directory:
        private = Path(directory)
        try:
            pre = preconditions(private)
        except (OSError, ImportError) as error:
            pre = dict(passed=False, resource_error=str(error))
        m.progress("authenticate_history")
        authentication = m.authenticate(args.upstream_method, args.upstream_rows)
        atomic_json(raw / "method.json", m.method())
        runnable = pre["passed"] and authentication["passed"]
        plan = (
            c.freeze()
            if runnable
            else dict(inputs=[], labels=[], length_balance_rows=[], split_manifest={})
        )
        fixtures = c.fixtures() if runnable else []
        frozen = dict(authentication=authentication, plan=plan, fixtures=fixtures)
        atomic_json(raw / "frozen.json", frozen)
        atomic_json(raw / "arm_inputs.json", dict(rows=[r["input"] for r in fixtures]))
        frozen_at = datetime.now(UTC).isoformat()
        m.progress("methods_and_labels_sealed", len(plan["inputs"]), 0)
        receipts = []
        if runnable:
            receipt = child(
                "input_only_tiny_controls",
                [
                    sys.executable,
                    m.CLI,
                    "--decide-input",
                    str(raw / "arm_inputs.json"),
                    "--decisions",
                    str(raw / "decisions.json"),
                ],
                raw / "measurement",
                deadline=60,
            )
            receipts.append(receipt)
            if not (raw / "decisions.json").is_file():
                atomic_json(raw / "decisions.json", dict(rows=[]))
        else:
            atomic_json(raw / "decisions.json", dict(rows=[]))
        measured_at = datetime.now(UTC).isoformat()
        m.progress("before_scoped_validation")
        validation = (
            previous.private_validation(raw)
            if args.private_control
            else validator.validate(raw, private)
        )
        m.progress("after_scoped_validation")
        ended = datetime.now(UTC).isoformat()
        sources = [
            reference(m.ROOT / p)
            for p in m.OWNED
            + m.TESTS
            + [
                "scripts/experiment_template.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
                "python/carnot/reporting/primary_publication.py",
                "ops/exclusion_manifest.yaml",
                "openspec/capabilities/research-reporting/spec.md",
                "openspec/capabilities/verification/spec.md",
            ]
        ]
        health = m.ROOT / "results/raw" / m.NAME / "startup/full_python_suite.receipt.json"
        work = dict(
            raw=str(raw),
            run_date=args.date,
            invocation_id=raw.name,
            upstream_method=str(args.upstream_method.absolute()),
            upstream_rows=str(args.upstream_rows.absolute()),
            preconditions_checked=pre,
            method_reference=reference(raw / "method.json"),
            frozen_reference=reference(raw / "frozen.json"),
            decisions_reference=reference(raw / "decisions.json"),
            sources=sources,
            shards=[
                reference(raw / name)
                for name in ["method.json", "frozen.json", "arm_inputs.json", "decisions.json"]
            ],
            measurement_receipts=receipts,
            repository_health_reference=dict(
                reference=reference(health) if health.is_file() else None,
                full_suite_pass=False,
                scope="one whole-repository diagnostic; separate from scoped acceptance",
            ),
            started_at_utc=started,
            ended_at_utc=ended,
            duration_s=(time.monotonic_ns() - start_ns) / 1e9,
            phase_spans=[
                dict(phase=p, start_utc=a, end_utc=b)
                for p, a, b in [
                    ("preconditions_and_freeze", started, frozen_at),
                    ("tiny_controls", frozen_at, measured_at),
                    ("validation", measured_at, ended),
                ]
            ],
        )
        atomic_json(raw / "work.json", work)
        atomic_json(raw / "validation.json", validation)
        value = report.build(work, validation, output)
        atomic_json(
            raw / "independent_seal.json",
            {k: value[k] for k in ["primitive_reference", "validation_reference"]},
        )
        m.progress("before_terminal_publication")
        published = publish_primary(
            output, value, lambda candidate: terminal.terminal(candidate, raw)
        )
        atomic_json(raw / "publication_receipt.json", published)
        atomic_json(
            raw / "reader_receipt.json",
            reader_receipt(
                m.TASK,
                output.parent,
                field="evidence_interventions_ready_score",
                expected=value["evidence_interventions_ready_score"],
            ),
        )
        m.progress("terminal_published_" + value["verdict_class"], len(value["rows"]), 0)
    return 0
