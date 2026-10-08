"""REQ-VERIFY-8304: actual bounded children qualify one atomic terminal primary."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import v717_contract_methods as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child, execute
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]


def preflight(plan: list[Json]) -> list[Json]:
    """Missing executables are external operands, checked before measurement or children."""
    failures = []
    for binary in dict.fromkeys(spec["argv"][0] for spec in plan):
        resolved = shutil.which(binary)
        if resolved is None:
            failures.append(m.failure(Path(binary), "executable_available", True, None))
    return failures


def manifest(private: Path) -> list[Json]:
    """Freeze invocation-only coverage and private paths before reading any outcomes."""
    py, cov, pytest, ruff, mypy = [
        str(m.ROOT / ".venv/bin" / n) for n in ["python", "coverage", "pytest", "ruff", "mypy"]
    ]
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(m.ROOT / p) + "\n" for p in m.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    rc = "--rcfile=" + str(config)
    specs = [
        (
            "owned_tests",
            [
                cov,
                "run",
                rc,
                "-m",
                "pytest",
                *common,
                m.TEST,
                "--basetemp=" + str(private / "owned-tests"),
            ],
            180,
        ),
        (
            "private_E2E018_consumers",
            [
                pytest,
                *common,
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_current_work_receipt.py",
                "--basetemp=" + str(private / "consumers"),
            ],
            240,
        ),
        ("coverage_combine", [cov, "combine", rc], 30),
        ("coverage_report", [cov, "report", rc, "--show-missing", "--fail-under=100"], 30),
        ("coverage_json", [cov, "json", rc, "-o", str(private / "coverage.json")], 30),
        ("ruff_check", [ruff, "check", *m.OWNED, m.TEST], 30),
        ("ruff_format", [ruff, "format", "--check", *m.OWNED, m.TEST], 30),
        (
            "strict_mypy",
            [
                mypy,
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=skip",
                "--ignore-missing-imports",
                *m.OWNED,
            ],
            60,
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", "--files", m.TEST], 30),
    ]
    return [dict(name=n, argv=a, expected=0, deadline=d, scope="owned") for n, a, d in specs]


def controls(value: Json, raw: Path) -> list[Json]:
    """Fresh children must accept valid bytes and reject both missing and rehashed tampering."""
    receipts = []
    for name in ["valid", "negative", "rehashed_tamper"]:
        changed = deepcopy(value)
        if name == "negative":
            changed["reproducibility_checksum"] = "invalid"
        if name == "rehashed_tamper":
            changed["current_contract_ready_score"] = 1 - changed["current_contract_ready_score"]
            changed.pop("reproducibility_checksum")
            changed["reproducibility_checksum"] = canonical_hash(changed)
        path = raw / (name + ".json")
        atomic_json(path, changed)
        receipts.append(
            child(
                "cold_" + name,
                [
                    str(m.ROOT / ".venv/bin/python"),
                    "-u",
                    str(m.ROOT / m.CLI),
                    "--cold-replay",
                    str(path),
                ],
                raw / "cold",
                expected=int(name != "valid"),
                deadline=60,
                heartbeat=20,
            )
        )
    return receipts


def publish(value: Json, output: Path, raw: Path) -> None:
    """Retain failed owned validation, clear readiness, then validate the honest failure."""
    reports: list[Json] = []

    def validate(candidate: Path) -> Json:
        attempt_logs = raw / "terminal" / str(len(reports) // 3)
        for name, argv in [
            ("terminal_cold", [str(m.ROOT / m.CLI), "--cold-replay", str(candidate)]),
            ("adversarial", ["scripts/adversarial_verify.py", "--json", str(candidate)]),
            (
                "strict_rows",
                ["scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
            ),
        ]:
            reports.append(
                child(
                    name,
                    [str(m.ROOT / ".venv/bin/python"), "-u", *argv],
                    attempt_logs,
                    deadline=60,
                    heartbeat=20,
                )
            )
        return dict(passed=all(r["passed"] for r in reports[-3:]), checks=reports[-3:])

    try:
        publication = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        atomic_json(raw / "rejected_candidate.json", value)
        atomic_json(raw / "rejected_checks.json", dict(rows=reports))
        work = json.loads((raw / "measurement.json").read_bytes())
        failed = deepcopy(reports)
        value = m.build(work, value["validation_receipts"] + failed, raw, output)
        publication = publish_primary(output, value, validate)
    atomic_json(
        output.parent / "raw" / output.stem / "terminal_validation.json",
        dict(publication=publication, checks=reports, normal_process_completion=True),
    )
    m.progress("published", 1, 0)


def main(argv: list[str] | None = None) -> int:
    """Private CLI operands exercise real execution without replacing historical primaries."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    m.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261008"], default="20261008")
    parser.add_argument("--root", type=Path, default=m.ROOT)
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-fixture", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = m.replay(args.cold_replay)
        m.progress("replay_passed" if passed else "reduction_drift")
        return int(not passed)
    output = args.output.absolute()
    if args.private_fixture and output.is_relative_to(m.ROOT):
        parser.error("fixture outputs must be private")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="exp8304-owned-") as directory:
        private = Path(directory)
        plan = manifest(private)
        if args.private_fixture:
            plan = [
                dict(
                    name="private_child",
                    argv=[sys.executable, "-c", "print('private CLI control')"],
                    expected=0,
                    deadline=10,
                    scope="owned",
                )
            ]
        atomic_json(
            raw / "execution_manifest.json",
            dict(
                owned=m.OWNED,
                private_scratch=str(private),
                commands=plan,
                child_heartbeat_seconds=20,
                terminal_deadline_seconds=60,
                fixture=args.private_fixture,
            ),
        )
        resource_failures = preflight(plan)
        work = m.measure(args.root, raw)
        work["failures"].extend(resource_failures)
        work["owned_validation_complete"] = not resource_failures
        receipts = execute(plan, raw / "checks") if not resource_failures else []
        if (private / "coverage.json").is_file():
            atomic_json(
                raw / "owned_coverage.json", json.loads((private / "coverage.json").read_bytes())
            )
            work["owned_coverage_reference"] = dict(
                path=str(raw / "owned_coverage.json"),
                sha256=sha256_file(raw / "owned_coverage.json"),
            )
        work["execution_manifest_reference"] = dict(
            path=str(raw / "execution_manifest.json"),
            sha256=sha256_file(raw / "execution_manifest.json"),
        )
        work["invocation_argv"] = [str(m.ROOT / m.CLI), *(sys.argv[1:] if argv is None else argv)]
        work["ended_monotonic_ns"] = time.monotonic_ns()
        value = normalize_artifact_for_template_write(m.build(work, receipts, raw, output))
        value["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in value.items() if k != "reproducibility_checksum"}
        )
        cold = controls(value, raw)
        value = normalize_artifact_for_template_write(m.build(work, receipts + cold, raw, output))
        value["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in value.items() if k != "reproducibility_checksum"}
        )
        atomic_json(
            raw / "execution_manifest_binding.json",
            dict(
                path=str(raw / "execution_manifest.json"),
                sha256=sha256_file(raw / "execution_manifest.json"),
            ),
        )
        publish(value, output, raw)
    return 0
