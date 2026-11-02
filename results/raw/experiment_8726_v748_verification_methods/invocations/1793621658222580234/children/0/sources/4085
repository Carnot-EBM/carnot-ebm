"""REQ-VERIFY-8519: bounded children qualify current readers without new timing."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import append_import_qualification_8519 as q
from carnot.reporting import append_service_cost_8494 as reader
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute
from carnot.reporting.v729_contract_runner import collect

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Original collected node unions retain assertions and private historical CLI dates."""
    private.mkdir(parents=True, exist_ok=True)
    py, pytest, cov, ruff, mypy = [
        str(q.ROOT / ".venv/bin" / n) for n in ("python", "pytest", "coverage", "ruff", "mypy")
    ]
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    suites = [
        "tests/python/test_append_store_8493.py",
        "tests/python/test_append_service_cost_8494.py",
        "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
        "tests/python/test_restricted_decision_audit_8210.py",
        "tests/python/test_primary_publication_7928.py",
    ]
    plan, collections = [], []
    for suite in suites:
        nodes, receipt = collect(private, suite)
        collections.append(dict(suite=suite, nodes=nodes, receipt=receipt))
        size = 1 if "8210" in suite else 8
        adapter = (
            ["-p", "carnot.reporting.append_import_history_8519"] if "append_" in suite else []
        )
        for i in range(0, len(nodes), size):
            plan.append(
                dict(
                    name=f"historical_{Path(suite).stem}_{i}",
                    argv=[
                        pytest,
                        *common,
                        *adapter,
                        *nodes[i : i + size],
                        "--basetemp=" + str(private / f"{Path(suite).stem}-{i}"),
                    ],
                    expected=0,
                    deadline=300,
                    scope="owned",
                    nodes=nodes[i : i + size],
                )
            )
    atomic_json(private / "collections.json", dict(collections=collections))
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess,_exit\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join(
            "    " + str(q.ROOT / p) + "\n"
            for p in [*q.OWNED, "python/carnot/reporting/append_service_cost_8494.py"]
        )
        + "[report]\nexclude_lines=\n"
    )
    rc = "--rcfile=" + str(config)
    specs = [
        (
            "historical_union",
            [
                py,
                "-c",
                "import json; from pathlib import Path; "
                "c=json.loads(Path(" + repr(str(private / "collections.json")) + ").read_text()); "
                "p=json.loads(Path("
                + repr(str(private / "plan.json"))
                + ").read_text())['commands']; "
                "a=[n for x in c['collections'] for n in x['nodes']]; "
                "b=[n for x in p if x['name'].startswith('historical_') for n in x.get('nodes',[])]; "
                "assert a==b and len(a)==len(set(a)); print(len(a), 'original nodes preserved')",
            ],
            30,
        ),
        (
            "owned_tests",
            [
                cov,
                "run",
                rc,
                "-m",
                "pytest",
                *common,
                q.TEST,
                "--basetemp=" + str(private / "owned"),
            ],
            300,
        ),
        ("coverage_combine", [cov, "combine", rc, "--keep"], 30),
        (
            "coverage_preserve",
            [
                py,
                "-c",
                "from pathlib import Path; import shutil; "
                "p=Path(" + repr(str(private)) + "); d=p/'coverage_shards'; d.mkdir(); "
                "[shutil.copyfile(s,d/s.name) for s in p.glob('.coverage*') if s.is_file()]",
            ],
            30,
        ),
        (
            "coverage_report",
            [cov, "report", rc, "--show-missing", "--fail-under=100", *q.OWNED],
            30,
        ),
        ("coverage_json", [cov, "json", rc, "-o", str(private / "coverage.json"), *q.OWNED], 30),
        (
            "repair_line_coverage",
            [
                py,
                "-c",
                "import ast,json; from pathlib import Path; from coverage import CoverageData; "
                "p=Path('python/carnot/reporting/append_service_cost_8494.py').absolute(); "
                "t=ast.parse(p.read_text()); lines={n.lineno for n in ast.walk(t) if "
                "(isinstance(n,ast.ImportFrom) and n.module=='carnot.verify.append_store_8493') or "
                "(isinstance(n,ast.Name) and n.id=='Store')}; "
                "d=CoverageData(" + repr(str(private / ".coverage")) + "); d.read(); "
                "covered=set(d.lines(str(p)) or []); proof=dict(owned_lines=sorted(lines),covered_lines=sorted(lines & covered)); "
                "Path("
                + repr(str(private / "repair_coverage.json"))
                + ").write_text(json.dumps(proof)); "
                "assert len(lines)==2 and lines<=covered; print(proof)",
            ],
            30,
        ),
        ("ruff_check", [ruff, "check", *q.OWNED, q.TEST, *reader.OWNED], 30),
        ("ruff_format", [ruff, "format", "--check", *q.OWNED, q.TEST], 30),
        (
            "strict_mypy",
            [
                mypy,
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=skip",
                "--ignore-missing-imports",
                *q.OWNED,
                *reader.OWNED,
            ],
            60,
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", "--files", q.TEST], 30),
    ]
    plan += [dict(name=n, argv=a, deadline=d, expected=0, scope="owned") for n, a, d in specs]
    atomic_json(private / "plan.json", dict(commands=plan))
    return plan


def run(root: Path, output: Path, private: bool, scratch: Path) -> int:
    """Retain real failures and publish only bytes accepted by unchanged consumers."""
    start = time.monotonic_ns()
    raw = output.parent / "raw" / q.NAME / "invocations" / str(time.time_ns())
    work = q.preflight(root, raw)
    receipts = []
    if not work["gate_check_summary"]:
        history_output = raw / "historical_replay.json"
        receipt = child(
            "historical_replay",
            [
                sys.executable,
                "-u",
                str(q.ROOT / q.CLI),
                "--history-root",
                str(root),
                "--history-output",
                str(history_output),
            ],
            raw / "history_logs",
            deadline=180,
        )
        receipts.append(receipt)
        if receipt["passed"]:
            work["history"] = q.read(q.reference(history_output))
        else:
            work["owned_failure"] = "historical_replay"
    plan = (
        []
        if private or not all(work["preconditions_checked"]["tools"].values())
        else manifest(scratch)
    )
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            commands=plan,
            owned=q.OWNED,
            full_repository_suite="not_run",
            measurement_cap_s=3600,
            task_cap_s=4800,
            validation_reserve_s=900,
            child_poll_s=30,
        ),
    )
    receipts += execute(plan, raw / "validation_logs")
    validation = raw / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    for path in scratch.iterdir():
        if path.is_file() and path.name.startswith(
            ("coverage", ".coverage", "repair_coverage", "collections", "plan")
        ):
            shutil.copyfile(path, validation / path.name)
    for name in ("coverage_shards", "collection"):
        if (scratch / name).is_dir():
            shutil.copytree(scratch / name, validation / name)
    work["validation_references"] = [
        q.reference(p)
        for p in sorted((raw / "validation").rglob("*"))
        if p.is_file()
        and (
            p.name.startswith(("coverage", ".coverage", "repair_coverage", "collections", "plan"))
            or p.parent.name == "collection"
        )
    ]
    if (raw / "validation/coverage.json").is_file():
        work["owned_coverage"] = q.read(q.reference(raw / "validation/coverage.json"))["totals"]
    work["duration_s"] = (time.monotonic_ns() - start) / 1e9
    candidate = raw / "preterminal.json"
    atomic_json(candidate, q.build(work, receipts, output))
    argv = [sys.executable, "-u", str(q.ROOT / q.CLI)]
    for name, args, expected in [
        ("private_cli", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("error", ["--deliberate-error"], 7),
    ]:
        receipts.append(child(name, argv + args, raw / "controls", expected=expected, deadline=180))
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["append_runner_ready_score"] = 999
    changed["reproducibility_checksum"] = q.checksum(changed)
    tamper = raw / "rehashed_tamper.json"
    atomic_json(tamper, changed)
    receipts.append(
        child(
            "tamper",
            argv + ["--cold-replay", str(tamper)],
            raw / "controls",
            expected=1,
            deadline=180,
        )
    )
    work["duration_s"] = (time.monotonic_ns() - start) / 1e9
    work["phase_spans"] = [dict(phase=r["name"], duration_s=r["duration_s"]) for r in receipts]
    value = q.build(work, receipts, output)
    reports: list[Json] = []

    def validate(path: Path) -> Json:
        for name, args in [
            ("terminal_cold", [str(q.ROOT / q.CLI), "--cold-replay", str(path)]),
            ("adversarial", ["scripts/adversarial_verify.py", "--json", str(path)]),
            ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)]),
        ]:
            reports.append(
                child(name, [sys.executable, "-u", *args], raw / "terminal", deadline=180)
            )
        return dict(passed=all(r["passed"] for r in reports), receipts=reports)

    try:
        publication = publish_primary(output, value, validate)
    except ValueError:
        work["owned_failure"] = "terminal_validation"
        work["adversarial_findings"] = [
            flag
            for r in reports
            if r["name"] == "adversarial"
            for report in json.loads(Path(r["stdout_path"]).read_bytes())["reports"]
            for flag in report.get("flags", [])
        ]
        receipts += reports
        value = q.build(work, receipts, output)
        reports = []
        publication = publish_primary(output, value, validate)
    consumer = reader_receipt(
        q.TASK,
        output.parent,
        field="append_runner_ready_score",
        expected=value["append_runner_ready_score"],
    )
    findings = [
        flag
        for report in json.loads(Path(reports[-2]["stdout_path"]).read_bytes())["reports"]
        for flag in report.get("flags", [])
    ]
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            receipts=reports,
            consumer=consumer,
            adversarial_findings=findings,
            passed=consumer["passed"] and not findings,
        ),
    )
    q.progress("finished", value["completed_count"], value["censored_count"])
    return int(not consumer["passed"] or bool(findings))


def main(argv: list[str] | None = None) -> int:
    """The direct CLI owns a finite deadline and keeps private outputs outside the checkout."""
    q.progress("start_no_model_load_MODEL_SPECS_empty_current_llm_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261014"], default="20261014")
    parser.add_argument("--root", type=Path, default=q.ROOT)
    parser.add_argument("--output", type=Path, default=q.ROOT / "results" / (q.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--history-root", type=Path)
    parser.add_argument("--history-output", type=Path)
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = q.replay(args.cold_replay)
        q.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    if args.history_root:
        if not args.history_output:
            parser.error("history output required")
        with TemporaryDirectory(dir="/var/tmp", prefix="exp8519-history-") as d:
            atomic_json(args.history_output, q.history(args.history_root, Path(d)))
        return 0
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(q.ROOT):
        parser.error("private outputs must be outside the repository")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(dir="/var/tmp", prefix="exp8519-validation-") as d:
            return run(args.root, output, args.private_control, Path(d))
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
