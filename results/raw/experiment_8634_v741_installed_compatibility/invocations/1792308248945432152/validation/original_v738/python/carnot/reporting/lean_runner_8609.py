"""REQ-VERIFY-8609: bounded original nodes and unchanged publication validators."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import signal
import shutil
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import lean_qualification_8609 as e
from carnot.reporting import lean_resident_8608 as prototype
from carnot.reporting import resident_validation_8593 as historical
from carnot.reporting.lean_original_8609 import checkout
from carnot.reporting.current_validator_8580 import xml_nodes
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt, read_bound_sidecar
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov"]
monotonic = time.monotonic


def validate(work: Json, raw: Path) -> list[Json]:
    """Collect unchanged node families before disjoint children with original fixtures."""
    scratch = Path(tempfile.mkdtemp(prefix="carnot8609-validation-", dir="/var/tmp"))
    config = scratch / "coverage.ini"
    observed = [
        "python/carnot/inference/resident_energy.py",
        "python/carnot/inference/resident_protocol.py",
        "scripts/experiments/resident_worker_8608.py",
    ]
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(scratch / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED + observed)
        + "[report]\nexclude_lines=\n"
    )
    producer, _ = e.upstream(e.ROOT, historical.e.NAME)
    original_nodes = producer["original_node_union"]["collected"]
    suites = list(
        dict.fromkeys(
            e.TESTS
            + prototype.TESTS
            + [node.split("::", 1)[0] for node in original_nodes]
            + [historical.e.TEST]
        )
    )
    collection = child(
        "collection",
        [sys.executable, "-m", "pytest", *COMMON, "--collect-only", "-q", *suites],
        raw / "validation",
        deadline=90,
    )
    nodes = [
        line.strip()
        for line in Path(collection["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    shards = [[node for node in nodes if node.startswith(suite + "::")] for suite in suites]
    shards = [selected for selected in shards if selected]
    atomic_json(
        raw / "original_nodes.json",
        dict(
            nodes=nodes,
            shards=shards,
            suites=suites,
            authority="unchanged original assertions and historical CLI dates",
        ),
    )
    receipts, outcomes = [collection], []
    original_checkout = checkout(scratch / "original_v738")
    work["historical_authority_replay"] = original_checkout
    atomic_json(raw / "historical_authority_replay.json", original_checkout)
    junit_shards = []
    for i, selected in enumerate(shards):
        xml = scratch / f"nodes-{i}.xml"
        command = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            "-m",
            "pytest",
        ]
        if selected[0].startswith(historical.e.TEST + "::"):
            original_root = original_checkout["root"]
            command = [
                "env",
                "PYTHONPATH=" + original_root + "/python:" + original_root,
                "CARNOT_REPO_ROOT=" + original_root,
                sys.executable,
                "-c",
                f"import os,sys;os.chdir({original_root!r});from coverage.cmdline import main;raise SystemExit(main(sys.argv[1:]))",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
            ]
        receipt = child(
            f"tests_{i}",
            [
                "env",
                "COVERAGE_RCFILE=" + str(config),
                *command,
                *COMMON,
                "-q",
                *selected,
                "--junitxml=" + str(xml),
                "--basetemp=" + str(scratch / f"suite-{i}"),
            ],
            raw / "validation",
            deadline=480,
        )
        receipts.append(receipt)
        if receipt["passed"]:
            outcomes.extend(xml_nodes(xml))
            retained = raw / f"nodes-{i}.xml"
            shutil.copyfile(xml, retained)
            junit_shards.append(e.reference(retained))
    union = dict(
        name="exact_disjoint_node_union",
        passed=bool(nodes)
        and len(nodes) == len(set(nodes))
        and sorted(nodes) == sorted(outcomes)
        and set(original_nodes) <= set(nodes),
        collected=nodes,
        passed_node_ids=outcomes,
        shards=shards,
        original_authority="unchanged original fixtures and historical dates",
        historical_v738_replay=original_checkout,
        original_published_union=original_nodes,
        junit_shards=junit_shards,
    )
    atomic_json(raw / "node_union.json", union)
    receipts.append(union)
    work["original_node_union"] = union
    if "started_ns" in work:
        probe_work = dict(work, raw=str(raw / "cold_union_snapshot"))
        atomic_json(Path(probe_work["raw"]) / "work.json", probe_work)
        probe = raw / "cold_union_probe.json"
        atomic_json(probe, e.artifact(probe_work, receipts, raw / (e.NAME + ".json")))
        receipts.append(
            child(
                "cold_union_coverage",
                [
                    "env",
                    "COVERAGE_RCFILE=" + str(config),
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "--rcfile=" + str(config),
                    e.CLI,
                    "--cold-replay",
                    str(probe),
                ],
                raw / "validation",
                deadline=90,
            )
        )
    report = scratch / "coverage.json"
    plan = [
        (
            "coverage_combine",
            [sys.executable, "-m", "coverage", "combine", "--rcfile=" + str(config), "--keep"],
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
                str(report),
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
                "--include=" + ",".join(str(e.ROOT / p) for p in e.OWNED),
                "--fail-under=100",
                "--show-missing",
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
    receipts.extend(child(name, argv, raw / "validation", deadline=90) for name, argv in plan)
    coverage = json.loads(report.read_bytes())
    atomic_json(raw / "coverage.json", coverage)
    owned = {str((e.ROOT / p).resolve()) for p in e.OWNED}
    measured = {str(Path(p).resolve()): row for p, row in coverage["files"].items()}
    passed = owned <= set(measured) and all(
        not measured[p]["missing_lines"] and measured[p]["summary"]["num_statements"] > 0
        for p in owned
    )
    work["child_statement_coverage"] = dict(
        passed=passed,
        owned_files=e.OWNED,
        report=e.reference(raw / "coverage.json"),
        files=coverage["files"],
        subprocess_patch="coverage patch=subprocess; actual worker children retained",
    )
    receipts.append(
        dict(
            name="new_owned_statement_coverage_100",
            passed=passed,
            report=e.reference(raw / "coverage.json"),
        )
    )
    return receipts


def terminal(path: Path, raw: Path) -> Json:
    """Actual unchanged exits and findings determine whether candidate bytes publish."""
    receipts = [
        child(name, [sys.executable, "-u", *argv], raw, deadline=60)
        for name, argv in [
            ("cold_replay", [e.CLI, "--cold-replay", str(path)]),
            ("adversarial", ["scripts/adversarial_verify.py", "--json", str(path)]),
            ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)]),
        ]
    ]
    findings = (
        json.loads(Path(receipts[1]["stdout_path"]).read_text())
        if Path(receipts[1]["stdout_path"]).read_text().strip().startswith("{")
        else {}
    )
    return dict(
        passed=all(r["passed"] for r in receipts), receipts=receipts, actual_findings=findings
    )


def run(output: Path, run_date: str) -> int:
    """Make a private reader-visible candidate before publishing identical checked bytes."""
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    began = monotonic()
    work = e.measure(raw, run_date)
    if monotonic() - began >= 3600:
        work["owned_deadline_failed"] = True
    receipts = validate(work, raw)
    receipts.append(
        dict(name="measurement_deadline", passed=not work.get("owned_deadline_failed", False))
    )
    atomic_json(Path(work["raw"]) / "work.json", work)
    value = e.artifact(work, receipts, output)
    cases = [("cold", value, 0), ("missing", None, 1)]
    changed = deepcopy(value)
    changed["completed_count"] += 1
    failed = e.artifact(work, [*receipts, dict(name="owned_failure", passed=False)], output)
    failed["lean_measurement_ready_score"] = 1
    cases.extend([("rehashed_tamper", changed, 1), ("owned_failure", failed, 1)])
    for name, candidate, expected in cases:
        path = raw / "controls" / (name + ".json")
        if candidate is not None:
            candidate["reproducibility_checksum"] = e.checksum(candidate)
            atomic_json(path, candidate)
        receipts.append(
            child(
                name,
                [sys.executable, "-u", e.CLI, "--cold-replay", str(path)],
                raw / "controls",
                expected=expected,
                deadline=60,
            )
        )
    value = e.artifact(work, receipts, output)
    private_output = raw / "private" / output.name
    private = publish_primary(private_output, value, lambda p: terminal(p, raw / "private_checks"))
    reader = reader_receipt(
        e.TASK,
        private_output.parent,
        field="lean_measurement_ready_score",
        expected=value["lean_measurement_ready_score"],
    )
    receipts.append(
        dict(
            name="private_publication_readers",
            passed=reader["passed"],
            publication=private,
            reader=reader,
        )
    )
    value = e.artifact(work, receipts, output)
    published = publish_primary(output, value, lambda p: terminal(p, raw / "terminal_checks"))
    atomic_json(
        raw / "terminal_validation.json",
        read_bound_sidecar(output, Path(published["sidecar_path"])),
    )
    progress("8609_complete_" + value["verdict_class"], 1, 0)
    return 0 if value["required_checks_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Both standalone replay and runtime dates use one bounded public entry point."""
    progress("8609_start_no_model_load_MODEL_SPECS=[]_current_llm_calls=0")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        if args.cold_replay:
            e.replay(args.cold_replay)
            return 0
        if datetime.strptime(args.date, "%Y%m%d").strftime("%Y%m%d") != args.date:
            raise ValueError("date_width")
        return run(args.output, args.date)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp8609] error={error}", file=sys.stderr, flush=True)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)
