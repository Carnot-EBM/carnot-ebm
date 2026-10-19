"""REQ-VERIFY-8656: bound separate historical, current and consumer child execution."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import streaming_adapter_8656 as a
from carnot.reporting import streaming_execution_8642 as legacy
from carnot.reporting import streaming_execution_8656 as e
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v738_contract_runner import date_value, exact_union

Json = dict[str, Any]
CONSUMERS = {
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py": [
        "python/carnot/reporting/v685_authority_lifecycle.py",
        "scripts/experiments/experiment_7891_v685_authority_lifecycle.py",
    ],
    "tests/python/test_primary_publication_7928.py": [
        "python/carnot/reporting/primary_publication.py",
        "scripts/conductor_gates.py",
        "scripts/in_process_doc_reconcile.py",
    ],
    "tests/python/test_restricted_decision_audit_8210.py": [
        "python/carnot/verify/restricted_decision_audit_8210.py",
        "python/carnot/verify/restricted_decision_rule_8210.py",
        "scripts/experiments/experiment_8210_v709_restricted_decision_audit.py",
    ],
}


def child_suite(
    suite: str,
    role: str,
    sources: list[str],
    view: list[Json],
    raw: Path,
    scratch: Path,
    label: str,
) -> tuple[list[Json], list[Json], list[Json]]:
    """Collect once per authority and capture each bounded disjoint child before the next."""
    inventory = a.role_inventory(suite, role, [e.ROOT / p for p in sources], raw)
    overlay = {r["origin_path"]: r["reference"] for r in view}
    for row in inventory["sources"]:
        if row["source"] in overlay:
            row.update(
                reference=overlay[row["source"]], source_sha256=overlay[row["source"]]["sha256"]
            )
    config = scratch / (label + ".ini")
    config.write_text(
        "[run]\nparallel = true\npatch = subprocess\ninclude =\n"
        + "\n".join("    " + r["source"] for r in inventory["sources"])
        + "\n"
    )
    prefix = [sys.executable, "-m", "pytest", "-n", "0", "-o", "addopts=", "--no-cov"]
    argv = prefix + ["--collect-only", "-q", suite]
    collection = child(
        "collect_" + label,
        a.private_command(view, argv, scratch) if view else argv,
        raw / "checks",
        deadline=90,
    )
    nodes = [
        line
        for line in Path(collection["stdout_path"]).read_text().splitlines()
        if line.startswith(suite + "::")
    ]
    plan = [dict(name="collect_" + label, collected_node_ids=nodes)]
    receipts, manifests = [collection], []
    size = 1 if "8210" in suite else len(nodes) or 1
    for i in range(0, len(nodes), size):
        name = label + "_" + str(i)
        private = scratch / name
        private.mkdir()
        base, junit = private / ".coverage", private / "junit.xml"
        group = nodes[i : i + size]
        argv = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            "--context=" + name,
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "-q",
            *group,
            "--junitxml=" + str(junit),
            "--basetemp=" + str(private / "temp"),
        ]
        with patch.dict(os.environ, COVERAGE_FILE=str(base), COVERAGE_RCFILE=str(config)):
            receipt = child(
                name,
                a.private_command(view, argv, scratch) if view else argv,
                raw / "checks",
                deadline=300,
            )
        receipts.append(receipt)
        try:
            shards = a.capture(base, raw, inventory)
            passed = receipt["passed"] and receipt["normal_exit"]
        except (OSError, ValueError, a.DataError) as error:
            shards, passed = [], False
            receipt["capture_error"] = str(error)
        manifests.append(
            dict(
                name=name,
                role=inventory,
                node_ids=group,
                receipt=receipt,
                shards=shards,
                passed=passed,
            )
        )
        plan.append(dict(name=name, collected_node_ids=group, junit_path=str(junit)))
        e.progress("captured_" + label, min(i + size, len(nodes)), max(0, len(nodes) - i - size))
    return receipts, manifests, plan


def run_checks(raw: Path, history: Json) -> Json:
    """Require original/current outcomes while consumer statements retain their own role."""
    receipts, manifests, refs = [], [], []
    original_plan, current_plan = [], []
    with TemporaryDirectory(prefix="exp8656-checks-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        specs = [
            (
                e.old.TEST,
                "historical_driver",
                e.old.OWNED,
                history.get("views", {}).get("8629", []),
                "original8629",
            ),
            (
                legacy.TEST,
                "historical_driver",
                list(dict.fromkeys(legacy.OWNED + e.old.OWNED)),
                history.get("views", {}).get("8642", []),
                "original8642",
            ),
            (e.old.TEST, "streaming_producer", e.old.OWNED, [], "current8629"),
            (
                legacy.TEST,
                "current_adapter",
                list(dict.fromkeys(legacy.OWNED + e.old.OWNED)),
                [],
                "current8642",
            ),
            (e.TEST, "current_adapter", e.OWNED, [], "current8656"),
            *[
                (suite, "historical_consumer", sources, [], "consumer" + str(i))
                for i, (suite, sources) in enumerate(CONSUMERS.items())
            ],
        ]
        for suite, role, sources, view, label in specs:
            r, m, p = child_suite(suite, role, sources, view, raw, scratch, label)
            receipts += r
            manifests += m
            if label.startswith("original") or label.startswith("consumer"):
                original_plan += p
            if not label.startswith("original"):
                current_plan += p
        original, current = exact_union(original_plan), exact_union(current_plan)
        for union in (original, current):
            union["junit_references"] = [
                a.c.freeze(Path(r["path"]), raw) for r in union["junit_references"]
            ]
            refs += union["junit_references"]
        receipts += [original, current]
        coverage = coverage_gates(manifests, raw, scratch)
        receipts.append(dict(name="role_coverage", passed=coverage["passed"]))
        regression = a.c.retention_regression(raw)
        reproduced = reproduce_failures(history, raw, scratch)
        reproduced["unsuffixed_retention"] = regression
        receipts += [
            r
            for r in [
                reproduced.get("driver_failure"),
                reproduced.get("consumer_failure"),
                regression.get("receipt"),
            ]
            if r
        ]
        refs += reproduced.get("empty_consumer_sqlite", [])
        refs += [r["reference"] for r in regression.get("direct_capture", [])]
        refs += [regression["source_reference"]] if "source_reference" in regression else []
        receipts.append(
            dict(
                name="legacy_failures_reproduced",
                passed=regression["passed"] and reproduced["passed"],
            )
        )
        receipts += static_checks(raw)
        for manifest in manifests:
            refs += [r["reference"] for r in manifest["shards"]]
        for receipt in receipts:
            for key in ("stdout_path", "stderr_path"):
                if key in receipt:
                    refs.append(a.c.freeze(Path(receipt[key]), raw))
    return dict(
        receipts=receipts,
        child_shard_manifest=manifests,
        original_node_union=original,
        current_node_union=current,
        coverage=coverage,
        legacy_failures_reproduced=regression["passed"] and reproduced["passed"],
        legacy_failure_reproduction=reproduced,
        immutable_refs=refs,
        scratch_cleanup=not scratch.exists(),
    )


def coverage_gates(manifests: list[Json], raw: Path, scratch: Path) -> Json:
    """Combine only role-matched current children twice, preserving original SQLite bytes."""
    reports = []
    try:
        for label, role in [
            ("current8656", "current_adapter"),
            ("current8629", "streaming_producer"),
        ]:
            selected = [m for m in manifests if m["name"].startswith(label + "_")]
            sources = (
                [Path(r["source"]) for r in selected[0]["role"]["sources"]] if selected else []
            )
            first = a.combine(selected, role, sources, scratch / (label + "-combine-1"))
            second = a.combine(selected, role, sources, scratch / (label + "-combine-2"))
            if role == "streaming_producer":
                first["files"] = [
                    r for r in first["files"] if Path(r["source"]).name == "streaming_evidence.py"
                ]
            first.update(
                role=role,
                passed=bool(first["files"]) and all(not r["missing"] for r in first["files"]),
                second_combine_identical=first["statement_count"] == second["statement_count"],
            )
            reports.append(first)
        return dict(
            passed=all(r["passed"] and r["second_combine_identical"] for r in reports),
            role_reports=reports,
            original_shards_immutable=all(
                a.c.authenticate(r["reference"]) for m in manifests for r in m["shards"]
            ),
        )
    except (ValueError, OSError, a.DataError) as error:
        return dict(passed=False, role_reports=reports, error=str(error))


def reproduce_failures(history: Json, raw: Path, scratch: Path) -> Json:
    """Re-execute recorded failures without changing original assertions or blessing the old producer."""
    view = [
        r
        for r in history.get("views", {}).get("8642", [])
        if not r["origin_path"].endswith("streaming_source_runner_8629.py")
    ]
    nodes = history.get("failed_owned_test_identities", [])
    if len(nodes) != 2:
        return dict(passed=False, missing_reason="missing_historical_authority")
    argv = [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        "0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        *nodes,
        "--basetemp=" + str(scratch / "reproduce-original"),
    ]
    receipt = child(
        "original_driver_mismatch",
        a.private_command(view, argv, scratch),
        raw / "regressions",
        expected=1,
        deadline=240,
    )
    stdout = Path(receipt["stdout_path"]).read_text()
    retained = all("FAILED " + node in stdout for node in nodes) and len(nodes) == 2
    base = scratch / "consumer-wrong.coverage"
    argv = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--rcfile=/dev/null",
        "--include=" + str(e.ROOT / e.old.OWNED[0]),
        "-m",
        "pytest",
        "-n",
        "0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        "tests/python/test_primary_publication_7928.py::test_nested_reports_and_terminal_revalidation",
        "--basetemp=" + str(scratch / "reproduce-consumer"),
    ]
    with patch.dict(os.environ, COVERAGE_FILE=str(base)):
        consumer = child("consumer_wrong_inventory", argv, raw / "regressions", deadline=90)
    missing = False
    try:
        a.c.capture(base, raw, [e.ROOT / p for p in e.old.OWNED])
    except ValueError as error:
        missing = str(error) == "missing_child_statements"
    refs = [a.c.freeze(p, raw) for p in base.parent.glob(base.name + "*") if p.is_file()]
    return dict(
        passed=receipt["passed"] and retained and consumer["passed"] and missing,
        failed_test_identities=nodes,
        driver_failure=receipt,
        consumer_failure=consumer,
        consumer_missing_statements=missing,
        empty_consumer_sqlite=refs,
    )


def static_checks(raw: Path) -> list[Json]:
    """Keep exact tool exits and logs instead of interpreting lint silence as a pass."""
    commands = [
        ("ruff_check", [str(e.ROOT / ".venv/bin/ruff"), "check", *e.OWNED, e.TEST]),
        ("ruff_format", [str(e.ROOT / ".venv/bin/ruff"), "format", "--check", *e.OWNED, e.TEST]),
        (
            "strict_mypy",
            [str(e.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *e.OWNED],
        ),
        ("spec_coverage", [sys.executable, "scripts/check_spec_coverage.py", "--files", e.TEST]),
    ]
    return [child(name, argv, raw / "checks", deadline=120) for name, argv in commands]


def validate_candidate(path: Path, raw: Path) -> Json:
    """Fresh replay and unchanged shared validators inspect exactly the prospective bytes."""
    commands = [
        ("cold_replay", [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)]),
        ("adversarial", [sys.executable, "scripts/adversarial_verify.py", "--json", str(path)]),
        (
            "row_lint",
            [sys.executable, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)],
        ),
    ]
    receipts = [
        child(name, argv, raw / "terminal_checks" / e.sha256_file(path).split(":")[1], deadline=120)
        for name, argv in commands
    ]
    findings = json.loads(Path(receipts[1]["stdout_path"]).read_bytes())
    return dict(
        passed=all(r["passed"] for r in receipts), receipts=receipts, adversarial_findings=findings
    )


def main(argv: list[str] | None = None) -> int:
    """Keep evidence private until final replay and unchanged readers validate publication bytes."""
    e.progress("start_no_model_load_calls_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date_value, default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--measure-only", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        return int(not e.replay(args.cold_replay))
    previous = signal.setitimer(signal.ITIMER_REAL, 3600)
    started = time.monotonic()
    try:
        output = args.output.absolute()
        parent = (
            args.measure_only.parent if args.measure_only else output.parent / "raw" / output.stem
        )
        raw = parent / "invocations" / str(time.time_ns())
        work = e.measure(args.root, raw, args.date)
        if args.measure_only:
            e.atomic_json(args.measure_only, work)
            return 0
        ref = e.seal(work)
        signal.setitimer(signal.ITIMER_REAL, max(1, 4800 - (time.monotonic() - started)))
        validation = e.seal_validation(run_checks(raw, work["history"]), raw)
        initial = e.build(ref, validation, [], {}, output)
        controls = []
        for mode in ("missing", "rehashed"):
            changed = deepcopy(initial)
            if mode == "missing":
                changed["immutable_measurement_reference"]["path"] = str(raw / "absent")
            else:
                changed["streaming_extractor_ready_score"] = 999
                changed.pop("reproducibility_checksum")
                changed["reproducibility_checksum"] = e.canonical_hash(changed)
            path = raw / ("reject_" + mode + ".json")
            e.atomic_json(path, changed)
            controls.append(
                child(
                    "reject_" + mode,
                    [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
                    raw / "controls",
                    expected=1,
                    deadline=60,
                )
            )
        candidate = raw / "private_candidate.json"
        e.atomic_json(candidate, e.build(ref, validation, controls, {}, output))
        checked = validate_candidate(candidate, raw)
        value = e.build(
            ref, validation, controls + checked["receipts"], checked["adversarial_findings"], output
        )

        def terminal(path: Path) -> Json:
            """Owned failure may publish a zero-readiness disqualification with actual severity retained."""
            report = validate_candidate(path, raw)
            safe = value["verdict_class"] == "disqualified" and all(
                r["passed"] for r in report["receipts"] if r["name"] != "adversarial"
            )
            return dict(
                report,
                passed=report["passed"] or safe,
                failed_result_publication=safe,
                validation_passed=report["passed"],
            )

        result = publish_primary(output, value, terminal)
        readers = reader_receipt(
            e.TASK,
            output.parent,
            field="streaming_extractor_ready_score",
            expected=value["streaming_extractor_ready_score"],
        )
        e.atomic_json(raw / "terminal_validation.json", dict(result, reader_receipt=readers))
        e.progress("published", value["completed_count"], value["censored_count"])
        return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
