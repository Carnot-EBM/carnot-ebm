"""REQ-VERIFY-8625: bounded private checks precede validation of the published bytes."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import hardware_operation_boundary_8625 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.current_validator_8580 import xml_nodes
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v726_feature_panel_runner import scan

Json = dict[str, Any]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov"]
SUITES = [
    e.TEST,
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
    "tests/python/test_primary_publication_7928.py",
]


def validate(raw: Path) -> list[Json]:
    """Collect unchanged nodes first, then execute disjoint children with real subprocess coverage."""
    folder = raw / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    config, data = folder / "coverage.ini", folder / ".coverage"
    config.write_text(
        "[run]\nparallel = true\npatch = subprocess\ninclude = "
        + ",".join(str(e.ROOT / p) for p in e.OWNED)
        + "\n[report]\nexclude_lines =\n"
    )
    env = ["env", f"COVERAGE_RCFILE={config}", f"COVERAGE_FILE={data}"]
    receipts, collected, passed = [], [], []
    for i, suite in enumerate(SUITES):
        receipt = child(
            f"collect{i}",
            [sys.executable, "-m", "pytest", *COMMON, "--collect-only", "-q", suite],
            folder,
            deadline=60,
        )
        receipts.append(receipt)
        collected.extend(
            s
            for s in Path(receipt["stdout_path"]).read_text().splitlines()
            if s.startswith(suite + "::")
        )
    atomic_json(
        folder / "original_nodes.json",
        dict(
            original_node_ids=collected, suite_references=[e.reference(e.ROOT / p) for p in SUITES]
        ),
    )
    for i in range(0, len(collected), 4):
        xml = folder / f"tests{i}.xml"
        receipts.append(
            child(
                f"tests{i}",
                [
                    *env,
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "-m",
                    "pytest",
                    *COMMON,
                    "-q",
                    *collected[i : i + 4],
                    f"--junitxml={xml}",
                ],
                folder,
                deadline=180,
            )
        )
        if xml.exists():
            passed.extend(xml_nodes(xml))
    receipts.append(
        dict(
            name="exact_node_union",
            passed=bool(collected)
            and sorted(collected) == sorted(passed)
            and len(set(collected)) == len(collected),
            original_node_ids=collected,
            passed_node_ids=passed,
        )
    )
    receipts.append(
        child(
            "combine",
            [*env, sys.executable, "-m", "coverage", "combine", str(folder)],
            folder,
            deadline=60,
        )
    )
    report = folder / "coverage.json"
    receipts.append(
        child(
            "coverage_report",
            [*env, sys.executable, "-m", "coverage", "json", "-o", str(report)],
            folder,
            deadline=60,
        )
    )
    files = json.loads(report.read_bytes()).get("files", {}) if report.exists() else {}
    receipts.append(
        dict(
            name="owned_coverage",
            passed=set(e.OWNED) <= set(files)
            and all(
                not files[p]["missing_lines"] and files[p]["summary"]["num_statements"] > 0
                for p in e.OWNED
                if p in files
            ),
            report=e.reference(report) if report.exists() else None,
            files=files,
        )
    )
    for name, args in [
        ("ruff_check", ["ruff", "check", *e.OWNED, e.TEST]),
        ("ruff_format", ["ruff", "format", "--check", *e.OWNED, e.TEST]),
        ("strict_mypy", ["mypy", "--strict", "--follow-imports=silent", *e.OWNED]),
        ("spec_coverage", ["python", "scripts/check_spec_coverage.py", "--files", e.TEST]),
    ]:
        receipts.append(
            child(name, [str(e.ROOT / ".venv/bin" / args[0]), *args[1:]], folder, deadline=180)
        )
    atomic_json(folder / "receipts.json", dict(receipts=receipts))
    return receipts


def terminal(path: Path, raw: Path) -> Json:
    """A fresh bounded process reconstructs evidence before the unchanged shared readers inspect it."""
    cold = child(
        "cold",
        [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
        raw,
        deadline=60,
    )
    checks, findings = scan(path, raw)
    return dict(
        passed=cold["passed"] and all(r["passed"] for r in checks) and not findings,
        receipts=[cold, *checks],
        findings=findings,
    )


def run(
    root: Path, output: Path, private: Path, date: str, control: bool, owned_failure: bool
) -> int:
    """Keep failure receipts and expose only the exact candidate accepted by cold replay and both validators."""
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = e.measure(root, raw, private, date)
    receipts = (
        [
            child(
                "private_child",
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "raise SystemExit(7)"
                    if owned_failure
                    else "print('actual private child',flush=True)",
                ],
                raw / "validation",
                deadline=30,
            )
        ]
        if control
        else validate(raw)
    )
    work["exact_union_receipt"] = next(
        (r for r in receipts if r["name"] == "exact_node_union"),
        dict(passed=control, scope="private_control"),
    )
    work["owned_coverage"] = next((r for r in receipts if r["name"] == "owned_coverage"), None)
    work["refs"] += [e.reference(Path(r["report"]["path"])) for r in receipts if r.get("report")]
    work["refs"] += [
        e.reference(Path(r[key]))
        for r in receipts
        for key in ["stdout_path", "stderr_path"]
        if key in r
    ]
    work["phase_spans"] += [
        dict(phase=r["name"], duration_s=r["duration_s"]) for r in receipts if "duration_s" in r
    ]
    work["duration_s"] += sum(r.get("duration_s", 0) for r in receipts)
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, e.build(work, receipts, raw, output))
    checked, findings = scan(candidate, raw / "preterminal")
    receipts += checked
    work["adversarial_findings"] = findings
    atomic_json(raw / "validation_receipts.json", dict(receipts=receipts))
    work["validation_reference"] = e.reference(raw / "validation_receipts.json")
    work["refs"].append(work["validation_reference"])
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, raw, output)
    publication = publish_primary(output, value, lambda path: terminal(path, raw / "terminal"))
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="hardware_boundary_ready_score",
        expected=value["hardware_boundary_ready_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            consumer=consumer,
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    e.progress(
        "published_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Bind the requested date while private controls remain outside reader-visible results."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261018")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--owned-failure", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if len(args.date) != 8:
            return 2
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        return 2
    if args.cold_replay:
        return int(not e.replay(args.cold_replay))
    if args.private_control and args.output.resolve().is_relative_to(e.ROOT):
        parser.error("private output requires scratch outside checkout")
    timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8625-private-", dir="/var/tmp") as directory:
            return run(
                args.root,
                args.output.absolute(),
                Path(directory),
                args.date,
                args.private_control,
                args.owned_failure,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)
