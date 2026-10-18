"""REQ-REPORT-8642: supervise bounded children and publish only checked bytes."""

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

from carnot.reporting import streaming_custody_8642 as c
from carnot.reporting import streaming_execution_8642 as e
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v738_contract_runner import date_value, exact_union

Json = dict[str, Any]


def run_checks(raw: Path) -> Json:
    """Collect once, run disjoint children, and capture each data file before the next child."""
    receipts, plan, manifests, refs = [], [], [], []
    with TemporaryDirectory(prefix="exp8642-checks-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        config = scratch / "coverage.ini"
        config.write_text(
            "[run]\nparallel = true\npatch = subprocess\ninclude =\n"
            + "\n".join("    " + str(p) for p in e.SOURCES)
            + "\n"
        )
        collection = child(
            "collection",
            [
                sys.executable,
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--collect-only",
                "-q",
                *e.SUITES,
            ],
            raw / "checks",
            deadline=90,
        )
        receipts.append(collection)
        nodes = [
            line
            for line in Path(collection["stdout_path"]).read_text().splitlines()
            if any(line.startswith(s + "::") for s in e.SUITES)
        ]
        plan.append(dict(name="collect_original", collected_node_ids=nodes))
        for suite in e.SUITES:
            original = [n for n in nodes if n.startswith(suite + "::")]
            size = 1 if "8210" in suite else len(original) or 1
            for i in range(0, len(original), size):
                group = original[i : i + size]
                name = Path(suite).stem + f"_{i}"
                private = scratch / name
                private.mkdir()
                base, junit = private / ".coverage", private / "junit.xml"
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
                with patch.dict(os.environ, COVERAGE_FILE=str(base)):
                    receipt = child(name, argv, raw / "checks", deadline=240)
                receipts.append(receipt)
                try:
                    shards = c.capture(base, raw, e.SOURCES)
                    passed = receipt["passed"] and receipt["normal_exit"]
                except ValueError as error:
                    shards, passed = [], False
                    receipt["capture_error"] = str(error)
                manifests.append(
                    dict(name=name, node_ids=group, receipt=receipt, shards=shards, passed=passed)
                )
                plan.append(dict(name=name, collected_node_ids=group, junit_path=str(junit)))
                refs += [r["reference"] for r in shards]
                if junit.is_file():
                    refs.append(c.freeze(junit, raw))
                e.progress("child_captured", len(manifests), len(e.SUITES))
        union = exact_union(plan)
        receipts.append(union)
        shards = [r for m in manifests for r in m["shards"]]
        required = [e.ROOT / p for p in [*e.OWNED, "python/carnot/verify/streaming_evidence.py"]]
        aggregates, stable = [], []
        for i in range(2):
            aggregates.append(c.combine(shards, e.SOURCES, scratch / f"combine-{i}"))
            stable.append(
                dict(
                    name=f"immutable_combine_{i}",
                    passed=all(c.authenticate(r) for r in refs),
                    raw_shard_hashes=refs,
                )
            )
        coverage = dict(files=[r for r in aggregates[0]["files"] if Path(r["source"]) in required])
        coverage["passed"] = all(not r["missing"] for r in coverage["files"])
        coverage["statement_count"] = sum(r["statement_count"] for r in coverage["files"])
        receipts += [dict(name="owned_statement_coverage", passed=coverage["passed"]), *stable]
        for name, argv in [
            ("ruff_check", [str(e.ROOT / ".venv/bin/ruff"), "check", *e.OWNED, e.TEST]),
            (
                "ruff_format",
                [str(e.ROOT / ".venv/bin/ruff"), "format", "--check", *e.OWNED, e.TEST],
            ),
            (
                "strict_mypy",
                [str(e.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *e.OWNED],
            ),
            (
                "spec_coverage",
                [sys.executable, "scripts/check_spec_coverage.py", "--files", e.TEST],
            ),
        ]:
            receipts.append(child(name, argv, raw / "checks", deadline=120))
        for receipt in receipts:
            for key in ("stdout_path", "stderr_path"):
                if key in receipt:
                    refs.append(c.freeze(Path(receipt[key]), raw))
        refs.append(c.freeze(config, raw))
        e.atomic_json(raw / "execution_plan.json", dict(plan=plan))
        refs.append(c.freeze(raw / "execution_plan.json", raw))
    return dict(
        receipts=receipts,
        child_shard_manifest=manifests,
        original_node_union=union,
        coverage=coverage,
        immutable_revalidation_rows=stable,
        immutable_refs=refs,
        scratch_cleanup=not scratch.exists(),
    )


def validate_candidate(path: Path, raw: Path) -> Json:
    """Run unchanged readers on these exact bytes and keep their actual severity."""
    logs = raw / "terminal_checks" / e.sha256_file(path).split(":")[1]
    commands = [
        ("cold_replay", [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)]),
        ("adversarial", [sys.executable, "scripts/adversarial_verify.py", "--json", str(path)]),
        (
            "row_lint",
            [sys.executable, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)],
        ),
    ]
    receipts = [child(name, argv, logs, deadline=120) for name, argv in commands]
    findings = json.loads(Path(receipts[1]["stdout_path"]).read_bytes())
    return dict(
        passed=all(r["passed"] for r in receipts), receipts=receipts, adversarial_findings=findings
    )


def main(argv: list[str] | None = None) -> int:
    """Keep all evidence private until final validation has checked the publication bytes."""
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
        signal.setitimer(signal.ITIMER_REAL, max(1, 4800 - (time.monotonic() - started)))
        validation = run_checks(raw)
        work["validation"] = validation
        work["immutable_refs"] += validation.get("immutable_refs", [])
        ref = e.seal(work)
        candidate = raw / "private_candidate.json"
        initial = e.build(ref, [], {}, output)
        controls = []
        for mode in ("missing", "rehashed"):
            changed = deepcopy(initial)
            if mode == "missing":
                changed["measurement_reference"]["path"] = str(raw / "absent")
            else:
                changed["streaming_extractor_ready_score"] = 999
                changed.pop("reproducibility_checksum")
                changed["reproducibility_checksum"] = e.canonical_hash(changed)
            control = raw / ("reject_" + mode + ".json")
            e.atomic_json(control, changed)
            controls.append(
                child(
                    "reject_" + mode,
                    [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(control)],
                    raw / "controls",
                    expected=1,
                    deadline=60,
                )
            )
        e.atomic_json(candidate, e.build(ref, controls, {}, output))
        checked = validate_candidate(candidate, raw)
        value = e.build(
            ref, controls + checked["receipts"], checked["adversarial_findings"], output
        )

        def terminal(path: Path) -> Json:
            """An owned failure can publish its disqualification when structural readers agree."""
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
