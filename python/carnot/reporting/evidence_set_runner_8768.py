"""REQ-VERIFY-8768: seal input-only children and publish cold-replayed terminal bytes."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, UTC
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any

from coverage import Coverage
import yaml

from carnot.reporting import evidence_set_methods_8768 as m
from carnot.reporting import evidence_set_corpus_8768 as corpus
from carnot.reporting import evidence_set_analysis_8768 as analysis
from carnot.reporting import evidence_set_report_8768 as report
from carnot.reporting import v750_archive_runner as validation_original
from carnot.reporting import v749_contract_runner as terminal_original
from carnot.reporting.collection_contract import collect
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
validation_runner = namespace(validation_original, "evidence_set_validation")
validation_runner.a = SimpleNamespace(
    ROOT=m.ROOT, OWNED=m.OWNED, TESTS=m.TESTS, CONSUMERS=m.CONSUMERS, reference=reference
)
terminal_runner = namespace(terminal_original, "evidence_set_terminal")
terminal_runner.CLI = m.CLI
terminal = terminal_runner.terminal


def validation_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """New code uses strict type checks independent of historical module relaxations."""
    if name == "types":
        argv = [
            argv[0],
            "--config-file=/dev/null",
            "--strict",
            "--follow-imports=silent",
            "--ignore-missing-imports",
            *m.OWNED,
        ]
    if name.startswith("child_"):
        config = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--rcfile="))
        argv = ["env", "COVERAGE_RCFILE=" + config, *argv]
    return child(name, argv, logs, **kwargs)


def validation_collect(
    private: Path, raw: Path, suites: list[str], owned: list[str], run_child: Callable[..., Json]
) -> tuple[list[Json], Json]:
    """Instrument owned statements and actual CLI children without profiling old scanners."""
    plan, receipt = collect(private, raw, suites, owned, run_child)
    for spec in plan:
        config = Path(spec["config"]["path"])
        include = "include=\n" + "\n".join("    " + str(m.ROOT / p) for p in m.OWNED) + "\n"
        config.write_text(config.read_text().replace("[run]\n", "[run]\n" + include))
        spec["config"] = reference(config)
    return plan, receipt


validation_runner.child = validation_child
validation_runner.collect = validation_collect


def validate(raw: Path, tracer: Coverage) -> Json:
    """Reuse the retained exact-node/JUnit/SQLite union rather than inventing validation."""
    return dict(validation_runner.validate(raw, tracer, []))


def prepare(raw: Path, private: Path, private_control: bool) -> Json:
    """Freeze methods, exact nodes, resources and source identities before measurement."""
    private.mkdir(parents=True, exist_ok=True)
    probe = private / "writable"
    probe.write_text("private scratch")
    tools = {
        name: (m.ROOT / ".venv/bin" / name).is_file()
        for name in ["python", "pytest", "ruff", "mypy"]
    }
    manifest = yaml.safe_load((m.ROOT / "ops/exclusion_manifest.yaml").read_bytes())
    retired = any(
        str(row.get("experiment_id")) == "8768"
        for key in ["retired", "retired_experiments"]
        for row in manifest.get(key, [])
        if isinstance(row, dict)
    )
    preconditions = dict(
        private_scratch_writable=probe.read_text() == "private scratch",
        tools=tools,
        not_retired=not retired,
        no_model_needed=True,
        free_disk_bytes=shutil.disk_usage(private).free,
    )
    preconditions["passed"] = (
        preconditions["private_scratch_writable"] and all(tools.values()) and not retired
    )
    m.progress("freeze_methods_and_nodes")
    plan, receipt = collect(private, raw, m.TESTS + m.CONSUMERS, m.TESTS, child)
    nodes = [n for p in plan for n in p["nodes"]]
    atomic_json(
        raw / "nodes.json", dict(node_ids=nodes, sha256=canonical_hash(nodes), receipt=receipt)
    )
    atomic_json(raw / "method.json", m.method())
    upstream_paths = [
        "results/experiment_8729_v748_equal_budget_verifier.json",
        "results/experiment_8737_v749_contract_methods.json",
        "results/experiment_8755_v750_verifier_family_surface.json",
        "openspec/change-proposals/research-roadmap-vNEXT.md",
        "openspec/change-proposals/research-roadmap-v750-preserved-20261103.md",
        "openspec/change-proposals/research-roadmap-v749-preserved-20261102.md",
        "ops/exclusion_manifest.yaml",
        "research-references.md",
    ]
    shared = [
        "primary_publication",
        "current_work_receipt",
        "v709_execution",
        "v746_retained_evidence",
        "v749_verifier_protocol",
        "verifier_family_surface_8755",
        "v750_archive_runner",
        "v749_contract_runner",
        "collection_contract",
        "relational_evidence_8698",
        "relational_capacity_8712",
        "verification_methods_8726",
        "equal_budget_verifier_8729",
        "v749_contract_methods",
        "verifier_family_analysis_8755",
    ]
    sources = m.OWNED + m.TESTS + m.CONSUMERS + [f"python/carnot/reporting/{p}.py" for p in shared]
    sources += [
        "scripts/experiment_template.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "pyproject.toml",
        "CODEX.md",
        "CLAUDE.md",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/verification/spec.md",
        "docs/research-notes/v751-evidence-set-methods.md",
    ]
    references = [reference(m.ROOT / p) for p in upstream_paths]
    health = m.ROOT / "results/raw" / m.NAME / "startup/full_python_suite.receipt.json"
    health_ref = reference(health) if health.is_file() else None
    health_logs = (
        []
        if health_ref is None
        else [
            dict(
                path=str((m.ROOT / load(health_ref)[s + "_path"]).absolute()),
                sha256=load(health_ref)[s + "_sha256"],
            )
            for s in ["stdout", "stderr"]
        ]
    )
    references += ([] if health_ref is None else [health_ref]) + health_logs
    references += [reference(p) for p in sorted(health.parent.glob("development-*"))]
    return dict(
        raw=str(raw),
        invocation_id=raw.name,
        private_control=private_control,
        preconditions_checked=preconditions,
        sources=[reference(m.ROOT / p) for p in sources],
        references=references,
        nodes_reference=reference(raw / "nodes.json"),
        method_reference=reference(raw / "method.json"),
        repository_health_reference=dict(
            reference=health_ref,
            log_references=health_logs,
            scope="bounded separate diagnostic; no full-suite pass",
            timeout_s=120,
        ),
        historical_regression_reference=dict(
            reference=references[2],
            status="closed V750 schedule not rerun; regression history only",
            included_in_N=False,
        ),
    )


def replay(path: Path) -> bool:
    """Independently reconstruct corpus, oracle, decisions and terminal report from frozen bytes."""
    try:
        value = json.loads(path.read_bytes())
        raw = Path(value["primitive_reference"]["path"]).parent
        seal = json.loads((raw / "independent_seal.json").read_bytes())
        if seal != {k: value[k] for k in ["primitive_reference", "validation_reference"]}:
            return False
        work, validation = load(value["primitive_reference"]), load(value["validation_reference"])
        refs = work["sources"] + work["references"] + validation.get("raw_references", [])
        refs += [
            work[k]
            for k in [
                "method_reference",
                "corpus_reference",
                "oracle_reference",
                "source_records_reference",
                "decisions_reference",
                "nodes_reference",
            ]
        ]
        if any(reference(Path(r["path"]))["sha256"] != r["sha256"] for r in refs):
            return False
        inputs, labels, sources = corpus.freeze()
        if (
            load(work["method_reference"]) != m.method()
            or load(work["corpus_reference"])["rows"] != inputs
            or load(work["oracle_reference"])["rows"] != labels
            or load(work["source_records_reference"])["rows"] != sources
            or not analysis.audit(inputs, load(work["decisions_reference"])["rows"])
        ):
            return False
        return bool(value == report.build(work, validation, Path(value["publication_output"])))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(argv: list[str] | None = None) -> int:
    """A private run exercises actual CLI paths while production owns qualification."""
    m.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261103"], default="20261103")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--decide-input", type=Path)
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        m.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    if args.decide_input:
        if args.decisions is None:
            parser.error("--decisions is required with --decide-input")
        inputs = json.loads(
            args.decide_input.read_bytes(),
            object_pairs_hook=m.unique_object,
            parse_constant=m.reject_constant,
        )["rows"]
        for unit in inputs:
            m.decode(json.dumps(unit["input"]))
        atomic_json(args.decisions, dict(rows=analysis.measure(inputs), oracle_file_reads=0))
        return 0
    output = args.output.absolute()
    if args.private_control and output.resolve().is_relative_to(m.ROOT.resolve()):
        parser.error("private outputs must remain outside the checkout")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    started, start_ns = datetime.now(UTC).isoformat(), time.monotonic_ns()
    tracer = Coverage(
        data_file=str(raw / "producer.sqlite"),
        config_file=False,
        include=[str(m.ROOT / p) for p in m.OWNED],
    )
    if not args.private_control:
        tracer.start()
    with TemporaryDirectory(prefix="exp8768-private-") as directory:
        work = prepare(raw, Path(directory), args.private_control)
        inputs, labels, sources = corpus.freeze()
        for name, data in [("corpus", inputs), ("oracle", labels), ("source_records", sources)]:
            atomic_json(raw / (name + ".json"), dict(rows=data))
            work[name + "_reference"] = reference(raw / (name + ".json"))
        work["disjoint_splits"] = corpus.disjoint(inputs)
        frozen_at = datetime.now(UTC).isoformat()
        work["oracle_access_rows"] = [
            dict(
                phase="independent_oracle_seal",
                oracle_reads=len(inputs),
                consumer="oracle enumerator",
                feature_label_reads=0,
            )
        ]
        atomic_json(raw / "frozen_inputs.json", work)
        decisions, process_receipts = [], []
        for split in ["development", "evaluation"]:
            selected = [p for p in inputs if p["split"] == split]
            atomic_json(raw / (split + "_inputs.json"), dict(rows=selected))
            destination = raw / (split + "_decisions.json")
            receipt = child(
                split + "_input_only",
                [
                    sys.executable,
                    m.CLI,
                    "--decide-input",
                    str(raw / (split + "_inputs.json")),
                    "--decisions",
                    str(destination),
                ],
                raw / "measurement_processes",
                deadline=180,
            )
            process_receipts.append(receipt)
            observed = json.loads(destination.read_bytes())
            if (
                not receipt["passed"]
                or observed["oracle_file_reads"] != 0
                or not analysis.audit(selected, observed["rows"])
            ):
                raise ValueError("input_only_child_rejected")
            decisions += observed["rows"]
            work["oracle_access_rows"].append(
                dict(
                    phase=split + "_decisions",
                    oracle_reads=0,
                    feature_label_reads=0,
                    consumer="input-only CLI child",
                    receipt=receipt,
                    input_reference=reference(raw / (split + "_inputs.json")),
                    decision_reference=reference(destination),
                )
            )
        atomic_json(raw / "decisions.json", dict(rows=decisions))
        work["decisions_reference"] = reference(raw / "decisions.json")
        work["oracle_access_rows"].append(
            dict(
                phase="score_after_decision_seal",
                oracle_reads=len(labels),
                consumer="report aggregator only",
                feature_label_reads=0,
            )
        )
        numerical_end = datetime.now(UTC).isoformat()
        m.progress("before_scoped_validation")
        if args.private_control:
            validation = dict(
                receipts=[dict(name="private_control_no_readiness", passed=False)],
                union=dict(passed=False),
                coverage=dict(complete=False),
                raw_references=[],
            )
        else:
            validation = validate(raw, tracer)
        validation["measurement_receipts"] = process_receipts
        atomic_json(raw / "validation.json", validation)
        work["validation_reference"] = reference(raw / "validation.json")
        ended = datetime.now(UTC).isoformat()
        work.update(
            started_at_utc=started,
            ended_at_utc=ended,
            duration_s=(time.monotonic_ns() - start_ns) / 1e9,
            phase_spans=[
                dict(phase=p, start_utc=a, end_utc=b)
                for p, a, b in [
                    ("freeze", started, frozen_at),
                    ("decisions", frozen_at, numerical_end),
                    ("validation", numerical_end, ended),
                ]
            ],
        )
        atomic_json(raw / "work.json", work)
        value = report.build(work, validation, output)
        atomic_json(
            raw / "independent_seal.json",
            {k: value[k] for k in ["primitive_reference", "validation_reference"]},
        )
        m.progress("before_terminal_publication")
        published = publish_primary(output, value, lambda p: terminal(p, raw))
        consumer = reader_receipt(
            m.TASK,
            output.parent,
            field="evidence_set_methods_ready_score",
            expected=value["evidence_set_methods_ready_score"],
        )
        atomic_json(raw / "consumer.json", consumer)
        if not consumer["passed"]:
            raise ValueError("reader_identity")
        m.progress("terminal_published", len(value["rows"]), 0)
        print(
            json.dumps(
                dict(
                    published,
                    verdict=value["honest_verdict"],
                    ready=value["evidence_set_methods_ready_score"],
                )
            ),
            flush=True,
        )
    return 0
