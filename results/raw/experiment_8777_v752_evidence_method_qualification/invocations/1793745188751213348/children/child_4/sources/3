"""REQ-VERIFY-8777: freeze private authority, exact nodes and real CLI qualification."""

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

from carnot.reporting import evidence_qualification_8777 as q
from carnot.reporting import evidence_archive_8777 as archive
from carnot.reporting import evidence_qualification_report_8777 as report
from carnot.reporting import evidence_set_methods_8768 as original
from carnot.reporting import v746_retained_evidence as evidence
from carnot.reporting import v749_contract_runner as historical_terminal
from carnot.reporting.collection_contract import collect
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
terminal = namespace(historical_terminal, "evidence_qualification_terminal")
terminal.CLI = q.CLI


def available() -> bool:
    """Dependencies are checked before allocating measurement work or loading any model."""
    return all((q.ROOT / ".venv/bin" / p).is_file() for p in ["python", "pytest", "ruff", "mypy"])


def preconditions(private: Path) -> Json:
    """A writable private root and explicit resource projection make the CPU-only scope auditable."""
    private.mkdir(parents=True, exist_ok=True)
    probe = private / "writable"
    probe.write_text("private scratch")
    tools = available()
    manifest = yaml.safe_load((q.ROOT / "ops/exclusion_manifest.yaml").read_bytes())
    not_retired = not any(
        str(row.get("experiment_id")) == "8777"
        for key in ["retired", "retired_experiments"]
        for row in manifest.get(key, [])
        if isinstance(row, dict)
    )
    return dict(
        passed=tools and not_retired and probe.read_text() == "private scratch",
        not_retired=not_retired,
        tools_available=tools,
        private_scratch_writable=True,
        free_disk_bytes=shutil.disk_usage(private).free,
        resource_projection=dict(
            worlds=432,
            units=5184,
            arm_rows=15552,
            llm_calls=0,
            model_loads=0,
            maximum_calls_per_arm_unit=5,
            estimated_cpu_measurement_seconds=60,
            reserved_validation_seconds=900,
        ),
    )


def collect_plan(private: Path, raw: Path, root: Path) -> tuple[list[Json], Json]:
    """Freeze one exact node inventory before measuring any new evaluation world."""
    suite = os.path.relpath(root / "original_tests" / Path(archive.ORIGINAL_TESTS[0]).name, q.ROOT)
    plan, receipt = collect(
        private, raw, q.TESTS + original.CONSUMERS + [suite], q.TESTS + [suite], child
    )
    for spec in plan:
        config = Path(spec["config"]["path"])
        include = (
            "include=\n"
            + "\n".join("    " + str(q.ROOT / p) for p in q.OWNED + archive.ORIGINAL_OWNED)
            + "\n"
        )
        config.write_text(config.read_text().replace("[run]\n", "[run]\n" + include))
        spec["config"] = evidence.reference(config)
        if spec["nodes"][0].startswith(suite + "::"):
            spec["role"] = "original"
    atomic_json(raw / "frozen_test_plan.json", dict(plan=plan, collection_receipt=receipt))
    return plan, receipt


def private_validation(raw: Path) -> Json:
    """Private CLI controls retain a genuine failed check and cannot grant readiness."""
    source = raw / "test_private_failure.py"
    source.write_text(
        '"""REQ-VERIFY-8777: owned failure."""\ndef test_failure():\n    assert False\n'
    )
    receipt = child(
        "private_owned_failure",
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            "-o",
            "addopts=",
            "--no-cov",
            "-q",
            str(source),
        ],
        raw / "checks",
        deadline=60,
    )
    return dict(
        receipts=[receipt],
        union=dict(passed=False, node_ids=[]),
        coverage=dict(complete=False),
        raw_references=[
            dict(path=receipt[s + "_path"], sha256=receipt[s + "_sha256"])
            for s in ["stdout", "stderr"]
        ],
    )


def validate(raw: Path, tracer: Any, root: Path) -> Json:
    """Retain real serial children, then combine only invocation-owned unit and CLI SQLite."""
    frozen = json.loads((raw / "frozen_test_plan.json").read_bytes())
    plan = frozen["plan"]
    receipts, children, originals = [frozen["collection_receipt"]], [], []
    for spec in plan:
        config = spec["config"]["path"]
        receipt = child(
            spec["name"],
            ["env", "COVERAGE_RCFILE=" + config, *spec["argv"]],
            raw / "checks",
            deadline=600,
        )
        receipts.append(receipt)
        retained = evidence.retain(
            receipt,
            spec["nodes"],
            Path(spec["scratch"]),
            raw / "children" / spec["name"],
            [
                evidence.reference(q.ROOT / p)
                for p in q.OWNED + q.TESTS + archive.ORIGINAL_OWNED + archive.ORIGINAL_TESTS
            ],
            spec["role"],
        )
        children.append(retained)
        if spec["role"] in {"original", "consumer"}:
            originals.append(retained)
    for name, argv in [
        ("ruff", [".venv/bin/ruff", "check", *q.OWNED, *q.TESTS]),
        ("format", [".venv/bin/ruff", "format", "--check", *q.OWNED, *q.TESTS]),
        (
            "types",
            [
                ".venv/bin/mypy",
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=silent",
                "--ignore-missing-imports",
                *q.OWNED,
            ],
        ),
        (
            "spec",
            [sys.executable, "scripts/check_spec_coverage.py", *q.TESTS, *archive.ORIGINAL_TESTS],
        ),
    ]:
        receipts.append(child(name, argv, raw / "checks", deadline=120))
    union = evidence.aggregate(children, [n for s in plan for n in s["nodes"]])
    original_union = evidence.aggregate(
        originals, [n for s in plan if s["role"] in {"original", "consumer"} for n in s["nodes"]]
    )
    combined = evidence.combine(children, raw / "combined")
    cov = Coverage(data_file=combined["combined_reference"]["path"], config_file=False)
    cov.load()
    control = raw / "failure" / "control.json"
    if control.is_file():
        original_data = Coverage(
            data_file=json.loads(control.read_bytes())["producer_sqlite_reference"]["path"],
            config_file=False,
        )
        original_data.load()
        cov.get_data().update(original_data.get_data())
    cov.save()
    files: list[Json] = []
    for name in q.OWNED + archive.ORIGINAL_OWNED:
        _, statements, excluded, missing, _ = cov.analysis2(str(q.ROOT / name))
        files.append(
            dict(
                path=name,
                statements=len(statements),
                excluded=excluded,
                missing=missing,
                role="new" if name in q.OWNED else "original",
            )
        )
    count, missing_count = (
        sum(f["statements"] for f in files),
        sum(len(f["missing"]) for f in files),
    )
    coverage = dict(
        complete=count > 0 and missing_count == 0 and not any(f["excluded"] for f in files),
        percent=100 * (count - missing_count) / count,
        files=files,
        combined_reference=evidence.reference(Path(combined["combined_reference"]["path"])),
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
    return dict(
        receipts=receipts,
        union=union,
        original_node_union=original_union,
        coverage=coverage,
        raw_references=refs,
    )


def main(argv: list[str] | None = None) -> int:
    """Publish exact terminal bytes after measuring a frozen, model-free qualification scope."""
    q.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261103"], default="20261103")
    parser.add_argument("--output", type=Path, default=q.ROOT / "results" / (q.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--decide-input", type=Path)
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = report.replay(args.cold_replay)
        q.progress("replay_passed" if passed else "replay_rejected")
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
            dict(rows=q.measure(payloads), oracle_file_reads=0, feature_label_reads=0),
        )
        return 0
    output = args.output.absolute()
    if args.private_control and output.resolve().is_relative_to(q.ROOT.resolve()):
        parser.error("private outputs must remain outside the checkout")
    raw = output.parent / "raw" / q.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    started, start_ns = datetime.now(UTC).isoformat(), time.monotonic_ns()
    with TemporaryDirectory(prefix="exp8777-private-") as directory:
        private = Path(directory)
        pre = preconditions(private)
        q.progress("freeze_private_v751_authority")
        root = archive.freeze_archive(raw / "v751_authority")
        plan, collected = collect_plan(private, raw, root)
        inputs, labels, manifest = q.freeze()
        old = json.loads((q.ROOT / archive.PRIMARY).read_bytes())
        upstream = [
            evidence.reference(q.ROOT / archive.PRIMARY),
            old["method_reference"],
            old["corpus_reference"],
            old["oracle_reference"],
            evidence.reference(q.ROOT / archive.V751),
            evidence.reference(q.ROOT / "ops/exclusion_manifest.yaml"),
        ]
        old_inputs = evidence.load(old["corpus_reference"])["rows"]
        for unit in old_inputs:
            if unit["split"] == "evaluation":
                unit["split"] = "replication"
            unit["input"] = q.projection(unit["input"])
        authenticated = (
            evidence.load(old["method_reference"]) == original.method()
            and inputs[:2160] == old_inputs
        )
        authenticated = authenticated and all(
            a["unit_id"] == b["unit_id"] == p["unit_id"]
            and a["decision"] == b["decision"]
            and q.projection(a["truth_records"]) == p["input"]
            for a, b, p in zip(
                evidence.load(old["oracle_reference"])["rows"],
                labels[:2160],
                old_inputs,
                strict=True,
            )
        )
        authenticated = authenticated and all(
            evidence.reference(q.ROOT / p)["sha256"]
            == next(r["sha256"] for r in old["code_config_hashes"] if Path(r["path"]) == q.ROOT / p)
            for p in archive.ORIGINAL_OWNED + archive.ORIGINAL_TESTS
        )
        work: Json = dict(
            raw=str(raw),
            invocation_id=raw.name,
            run_date=args.date,
            preconditions_checked=pre,
            original_method_authenticated=authenticated,
            original_owned=archive.ORIGINAL_OWNED,
            original_tests=archive.ORIGINAL_TESTS,
            expected_original_nodes=old["retained_child_custody"]["node_ids"],
            consumers=original.CONSUMERS,
            references=upstream,
            sources=[
                evidence.reference(q.ROOT / p)
                for p in q.OWNED
                + q.TESTS
                + archive.ORIGINAL_OWNED
                + archive.ORIGINAL_TESTS
                + original.CONSUMERS
                + [
                    "pyproject.toml",
                    "scripts/experiment_template.py",
                    "scripts/adversarial_verify.py",
                    "scripts/verdict_row_consistency_lint.py",
                    "openspec/capabilities/research-reporting/spec.md",
                    "openspec/capabilities/verification/spec.md",
                    "docs/research-notes/v752-evidence-method-qualification.md",
                ]
            ],
            shards=[],
            original_primary_preserved=dict(
                reference=upstream[0], verdict_class=old["verdict_class"], promoted=False
            ),
        )
        frozen_method = dict(
            original_method=original.method(),
            original_method_reference=old["method_reference"],
            maximum_evaluator_invocations=5,
            local_over_budget="first five premise checks; any refutation refutes; otherwise insufficient",
            MODEL_SPECS=[],
        )
        for name, data in [
            ("method", frozen_method),
            ("corpus", dict(rows=inputs)),
            ("oracle", dict(rows=labels)),
            ("manifest", manifest),
        ]:
            target = raw / (name + ".json")
            atomic_json(target, data)
            work[name + "_reference"] = evidence.reference(target)
            work["shards"].append(evidence.reference(target))
        work["shards"] += [
            evidence.reference(root / "archive_seal.json"),
            evidence.reference(raw / "frozen_test_plan.json"),
        ]
        work["shards"] += [spec["config"] for spec in plan]
        health = q.ROOT / "results/raw" / q.NAME / "startup/full_python_suite.receipt.json"
        work["repository_health_reference"] = dict(
            reference=evidence.reference(health) if health.is_file() else None,
            full_suite_pass=False,
            scope="one bounded full-suite diagnostic; separate from scoped acceptance",
        )
        health_ref = work["repository_health_reference"]["reference"]
        if health_ref is not None:
            health_receipt = evidence.load(health_ref)
            work["references"] += [health_ref] + [
                dict(
                    path=str(q.ROOT / health_receipt[s + "_path"]),
                    sha256=health_receipt[s + "_sha256"],
                )
                for s in ["stdout", "stderr"]
            ]
        work["oracle_access_rows"] = [
            dict(
                phase="independent_boolean_oracle_seal",
                oracle_reads=len(labels),
                feature_label_reads=0,
                consumer="Boolean completion enumerator",
            ),
            dict(
                phase="input_only_arm_process",
                oracle_reads=0,
                feature_label_reads=0,
                consumer="strict facts/query CLI projection",
                os_access_sandbox=False,
            ),
        ]
        frozen_at = datetime.now(UTC).isoformat()
        atomic_json(
            raw / "frozen_inputs.json",
            {k: v for k, v in work.items() if k not in ["shards", "oracle_access_rows"]},
        )
        work["frozen_input_reference"] = evidence.reference(raw / "frozen_inputs.json")
        atomic_json(raw / "arm_inputs.json", dict(rows=[p["input"] for p in inputs]))
        q.progress("before_input_only_measurement", 0, len(inputs) * 3)
        receipt = child(
            "input_only_decisions",
            [
                sys.executable,
                q.CLI,
                "--decide-input",
                str(raw / "arm_inputs.json"),
                "--decisions",
                str(raw / "decisions.json"),
            ],
            raw / "measurement",
            deadline=180,
        )
        observed = json.loads((raw / "decisions.json").read_bytes())
        if (
            not receipt["passed"]
            or observed["oracle_file_reads"] != 0
            or not q.audit(inputs, observed["rows"])
        ):
            raise ValueError("input_only_child_rejected")
        work["decisions_reference"] = evidence.reference(raw / "decisions.json")
        work["shards"] += [work["decisions_reference"], evidence.reference(raw / "arm_inputs.json")]
        work["oracle_access_rows"].append(
            dict(
                phase="score_after_decision_seal",
                oracle_reads=len(labels),
                feature_label_reads=0,
                consumer="aggregator only",
                receipt=receipt,
            )
        )
        measured_at = datetime.now(UTC).isoformat()
        q.progress("before_original_failure_publication")
        work["failure_control"] = (
            dict(passed=False, private_control=True)
            if args.private_control
            else archive.failure_control(root, raw / "failure")
        )
        q.progress("before_scoped_validation")
        validation = private_validation(raw) if args.private_control else validate(raw, None, root)
        ended = datetime.now(UTC).isoformat()
        work.update(
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
        q.progress("before_terminal_publication")
        published = publish_primary(
            output, value, lambda candidate: terminal.terminal(candidate, raw)
        )
        atomic_json(raw / "publication_receipt.json", published)
        atomic_json(
            raw / "reader_receipt.json",
            reader_receipt(
                q.TASK,
                output.parent,
                field="evidence_methods_ready_score",
                expected=value["evidence_methods_ready_score"],
            ),
        )
        q.progress("terminal_published_" + value["verdict_class"], len(value["rows"]), 0)
    return 0
