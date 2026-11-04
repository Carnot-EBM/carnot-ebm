"""REQ-VERIFY-8814: bounded checks qualify only this administrative producer."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

from coverage import Coverage

from carnot.reporting import v755_branch_contract as c
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def run_check(
    name: str, argv: list[str], raw: Path, deadline: float = 120, scope: str = "owned"
) -> Json:
    """Forward child streams while the existing supervisor enforces its original deadline."""
    offsets = dict.fromkeys(["stdout", "stderr"], 0)
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(child, name, argv, raw, deadline=deadline, scope=scope, heartbeat=30)
        while not future.done():
            for stream in offsets:
                path = raw / (name + "." + stream)
                if path.is_file():
                    data = path.read_bytes()
                    if len(data) > offsets[stream]:
                        print(data[offsets[stream] :].decode(errors="replace"), end="", flush=True)
                        offsets[stream] = len(data)
            c.progress(f"child_{name}_elapsed_{int(time.monotonic() - start)}")
            time.sleep(1)
        result: Json = future.result()
    for stream in offsets:
        data = Path(result[stream + "_path"]).read_bytes()
        print(data[offsets[stream] :].decode(errors="replace"), end="", flush=True)
    return result


def coverage_result(path: Path) -> Json:
    """Require actual measured statements for every owned file with no excluded lines."""
    report = json.loads(path.read_bytes())
    files = []
    for name in c.OWNED:
        found = next((v for k, v in report["files"].items() if k.endswith(name)), {})
        files.append(
            dict(
                path=name,
                statements=found.get("summary", {}).get("num_statements", 0),
                missing=found.get("missing_lines", ["unmeasured"]),
                excluded=found.get("excluded_lines", []),
            )
        )
    complete = all(f["statements"] and not f["missing"] and not f["excluded"] for f in files)
    return dict(
        complete=bool(complete),
        percent=100 if complete else report["totals"]["percent_covered"],
        files=files,
        report_reference=c.reference(path),
    )


def validate(raw: Path, tracer: Coverage) -> Json:
    """Run owned tests and private E2E once; broad repository debt is separate evidence."""
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\ninclude =\n"
        + "".join("    " + str(c.ROOT / p) + "\n" for p in c.OWNED)
        + "[report]\nexclude_lines =\n"
    )
    python = sys.executable
    checks = raw / "checks"
    receipts = [
        run_check(
            "owned_tests",
            [
                python,
                "-m",
                "coverage",
                "run",
                "--rcfile",
                str(config),
                "--data-file",
                str(raw / ".coverage"),
                "--parallel-mode",
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                c.TEST,
            ],
            checks,
            180,
        )
    ]
    receipts.append(
        run_check(
            "private_e2e018_021",
            [
                python,
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "tests/python/test_restricted_decision_audit_8210.py",
            ],
            checks,
            180,
        )
    )
    for name, args in [
        ("ruff_check", ["ruff", "check", *c.OWNED, c.TEST]),
        ("ruff_format", ["ruff", "format", "--check", *c.OWNED, c.TEST]),
        ("mypy", ["mypy", "--strict", *c.OWNED]),
        ("spec_coverage", [python, "scripts/check_spec_coverage.py", c.TEST]),
    ]:
        if args[0] in {"ruff", "mypy"}:
            args[0] = str(c.ROOT / ".venv/bin" / args[0])
        receipts.append(run_check(name, args, checks, 120))
    tracer.stop()
    tracer.save()
    combined = Coverage(data_file=str(raw / "combined.coverage"), config_file=str(config))
    combined.combine(data_paths=[str(raw)], strict=True, keep=True)
    combined.save()
    report = raw / "coverage.json"
    combined.json_report(outfile=str(report))
    coverage = coverage_result(report)
    receipts.append(
        dict(
            name="owned_statement_coverage",
            passed=coverage["complete"],
            reference=c.reference(report),
        )
    )
    debt_path = c.ROOT / "results/raw" / c.NAME / "development/global_python_suite.receipt.json"
    debt = (
        json.loads(debt_path.read_bytes())
        if debt_path.is_file()
        else dict(passed=False, reason="not_recorded")
    )
    return dict(receipts=receipts, coverage=coverage, global_test_debt=debt)


def terminal(candidate: Path, raw: Path) -> Json:
    """The same candidate passes scanners and a fresh interpreter outside the checkout."""
    checks = [
        run_check(
            "adversarial",
            [sys.executable, "scripts/adversarial_verify.py", str(candidate), "--json"],
            raw,
        ),
        run_check(
            "row_consistency",
            [sys.executable, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
            raw,
        ),
        run_check(
            "cold_replay",
            [
                "env",
                "-u",
                "PYTHONPATH",
                "-C",
                "/var/tmp",
                sys.executable,
                "-u",
                str(c.ROOT / c.CLI),
                "--cold-replay",
                str(candidate),
            ],
            raw,
        ),
    ]
    result = dict(passed=all(r["passed"] for r in checks), receipts=checks)
    c.atomic_json(raw / "terminal_validation.json", result)
    return result


def run(output: Path, raw: Path, root: Path, run_date: str) -> int:
    """Retain rejected candidates and publish only bytes accepted by the shared publisher."""
    raw.mkdir(parents=True, exist_ok=True)
    tracer = Coverage(
        data_file=str(raw / ".coverage.parent"),
        include=[str(c.ROOT / p) for p in c.OWNED],
        config_file=False,
    )
    tracer.set_option("report:exclude_lines", [])
    tracer.start()
    work = c.freeze(root, raw, run_date)
    c.progress("validation_before")
    validation = validate(raw, tracer)
    c.atomic_json(raw / "validation.json", validation)
    c.progress("validation_after")
    value = c.build(work, validation)
    value["raw_shard_hashes"].append(c.reference(raw / "validation.json"))
    value["reproducibility_checksum"] = c.checksum(value)
    c.progress("publication_before")
    try:
        receipt = publish_primary(output, value, lambda p: terminal(p, raw / "terminal"))
    except ValueError as error:
        c.progress("publication_rejected_" + str(error))
        return 1
    c.atomic_json(raw / "publication.json", receipt)
    readers = reader_receipt(
        c.TASK, output.parent, field="contract_ready_score", expected=value["contract_ready_score"]
    )
    c.atomic_json(raw / "reader_receipt.json", readers)
    c.progress("publication_after", 13)
    print(
        json.dumps(
            dict(
                path=str(output),
                verdict=value["honest_verdict"],
                contract_ready_score=value["contract_ready_score"],
            )
        ),
        flush=True,
    )
    return 0 if value["required_checks_passed"] and readers["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Private paths support qualification without changing the active task queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261104")
    parser.add_argument("--output", type=Path, default=c.ROOT / "results" / (c.NAME + ".json"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--root", type=Path, default=c.ROOT)
    parser.add_argument("--cold-replay", "--replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        result = c.replay(args.cold_replay)
        print(json.dumps(result), flush=True)
        return 0 if result["passed"] else 1
    if not re.fullmatch(r"\d{8}", args.date):
        print("invalid_date", flush=True)
        return 2
    raw = args.raw_dir or args.output.parent / "raw" / c.NAME / "invocations" / str(time.time_ns())
    return run(args.output.absolute(), raw.absolute(), args.root, args.date)
