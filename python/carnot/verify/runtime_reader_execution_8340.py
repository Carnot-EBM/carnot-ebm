"""REQ-VERIFY-8340: freeze bounded child recipes before runtime observations."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import os
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import yaml  # type: ignore[import-untyped]

from carnot.reporting import sentence_spline_execution_8334 as qualified
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.v709_execution import execute
from carnot.verify import runtime_reader_8340 as q

Json = dict[str, Any]
PY = str(q.ROOT / ".venv/bin/python")


def authority_controls(private: Path) -> list[Json]:
    """Four real reader invocations test old/current authority and reject mutations."""
    commands = []
    for name, milestone, expected in [
        ("valid_current", q.MILESTONE, 0),
        ("valid_old", "2026.10.716", 0),
        ("wrong_milestone", q.MILESTONE, 1),
        ("corrupt_contract", q.MILESTONE, 1),
    ]:
        root = private / name
        design = q.design(root, milestone)
        design.parent.mkdir(parents=True, exist_ok=True)
        design.write_bytes(q.design(q.ROOT, milestone).read_bytes())
        tasks = q.parse_design(design.read_text(), milestone=milestone)[1]
        if name == "corrupt_contract":
            tasks[0]["prompt"] += " corrupted private control"
        (root / "research-roadmap.yaml").write_text(
            yaml.safe_dump(
                dict(milestone="wrong" if name == "wrong_milestone" else milestone, tasks=tasks)
            )
        )
        commands.append(
            dict(
                name=name,
                argv=[PY, "-u", str(q.ROOT / q.CLI), "--authority", milestone, "--root", str(root)],
                deadline=60,
                expected=expected,
                scope="owned",
            )
        )
    return commands


def manifest(private: Path) -> list[Json]:
    """Scope coverage to added code and retain immutable historical consumers."""
    config = private / "coverage.ini"
    config.write_text(
        f"[run]\nparallel = True\npatch = subprocess\ndata_file = {private / '.coverage'}\n[report]\nexclude_lines =\n"
    )
    pytest = [str(q.ROOT / ".venv/bin/pytest"), "-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    specs = [
        (
            "tools_and_private_scratch",
            [
                PY,
                "-c",
                'import coverage,pytest,ruff,mypy,shutil,pathlib,sys;p=pathlib.Path(sys.argv[1]);p.write_bytes(b"private");assert p.read_bytes()==b"private";assert shutil.disk_usage(p.parent).free>10000000;print(sys.version)',
                str(private / "probe"),
            ],
            20,
        ),
        (
            "owned_coverage",
            [
                PY,
                "-m",
                "coverage",
                "run",
                "-p",
                "--rcfile",
                str(config),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                q.TEST,
                "--basetemp=" + str(private / "owned"),
            ],
            360,
        ),
        (
            "coverage_combine",
            [PY, "-m", "coverage", "combine", "--rcfile", str(config), "--keep"],
            20,
        ),
        (
            "coverage_json",
            [
                PY,
                "-m",
                "coverage",
                "json",
                "--rcfile",
                str(config),
                "--include=" + ",".join(str(q.ROOT / p) for p in q.OWNED),
                "--fail-under=100",
                "-o",
                str(private / "coverage.json"),
            ],
            20,
        ),
        (
            "E2E018_runtime_consumers",
            [
                *pytest,
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "tests/python/test_runtime_localization_8290.py",
                "tests/python/test_runtime_change_boundary_8307.py",
                "tests/python/test_primary_publication_7928.py",
                "--basetemp=" + str(private / "consumers"),
            ],
            360,
        ),
        ("ruff_check", [str(q.ROOT / ".venv/bin/ruff"), "check", *q.OWNED, q.TEST], 20),
        (
            "ruff_format",
            [str(q.ROOT / ".venv/bin/ruff"), "format", "--check", *q.OWNED, q.TEST],
            20,
        ),
        (
            "strict_mypy",
            [
                str(q.ROOT / ".venv/bin/mypy"),
                "--strict",
                "--follow-imports=silent",
                "--config-file=/dev/null",
                *q.OWNED,
            ],
            60,
        ),
        ("spec_coverage", [PY, "scripts/check_spec_coverage.py", "--files", q.TEST], 20),
    ]
    return [
        dict(
            name=n,
            argv=a,
            deadline=d,
            expected=0,
            scope="external_preconditions" if n == "tools_and_private_scratch" else "owned",
        )
        for n, a, d in specs
    ] + authority_controls(private / "authority_controls")


def publish(value: Json, output: Path, raw: Path) -> None:
    """Use unchanged validators and retain a checked disqualification on failure."""
    with patch.object(qualified, "e", q):
        qualified.publish(value, output, raw)


def main(argv: list[str] | None = None) -> int:
    """A private mechanics run never substitutes for current environment evidence."""
    q.progress("start_no_model_load")
    os.environ["PYTHONUNBUFFERED"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261009"], default="20261009")
    parser.add_argument("--root", type=Path, default=q.ROOT)
    parser.add_argument("--output", type=Path, default=q.ROOT / "results" / (q.NAME + ".json"))
    parser.add_argument("--private-run", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--authority", choices=[q.MILESTONE, "2026.10.716"])
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = q.replay(args.cold_replay)
        q.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    if args.authority:
        with TemporaryDirectory(prefix="exp8340-authority-") as directory:
            auth = q.authority(args.root, Path(directory), args.authority)
        print(
            json.dumps(
                dict(activated=auth["activated"], gate_check_summary=auth["gate_check_summary"])
            ),
            flush=True,
        )
        return int(not auth["activated"])
    output = args.output.absolute()
    if args.private_run and output.is_relative_to(q.ROOT / "results"):
        parser.error("private run requires private output")
    began, wall = time.monotonic_ns(), time.time_ns()
    raw = output.parent / "raw" / q.NAME / "invocations" / str(wall)
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    with TemporaryDirectory(prefix="carnot8340-") as directory:
        private = Path(directory)
        plan = manifest(private)
        atomic_json(
            raw / "execution_manifest.json",
            dict(
                commands=plan,
                frozen_before_measurement=True,
                terminal_argv=[
                    [
                        PY,
                        "-u",
                        str(q.ROOT / q.CLI),
                        "--cold-replay",
                        str(output.parent / "raw" / q.NAME / "terminal_candidate.json"),
                    ],
                    [
                        PY,
                        "scripts/adversarial_verify.py",
                        "--json",
                        str(output.parent / "raw" / q.NAME / "terminal_candidate.json"),
                    ],
                    [
                        PY,
                        "scripts/verdict_row_consistency_lint.py",
                        "--strict",
                        str(output.parent / "raw" / q.NAME / "terminal_candidate.json"),
                    ],
                ],
                terminal_deadline_s=60,
                heartbeat_s=20,
                negative_argv=[
                    [
                        PY,
                        "-u",
                        str(q.ROOT / q.CLI),
                        "--cold-replay",
                        str(private / (name + ".json")),
                    ]
                    for name in ("negative_checksum", "rehashed_tamper")
                ],
            ),
        )
        receipts = execute(plan[:1], raw / "preconditions")
        work = q.measure(args.root, raw)
        earlier = q.ROOT / "results/raw" / q.NAME
        work["prior_validation_attempts"] = [
            dict(path=str(path), receipt=json.loads(path.read_bytes()))
            for directory in (
                "reproduction",
                "development_coverage",
                "development_coverage2",
                "repository_verification",
            )
            for path in (earlier / directory).glob("*.receipt.json")
        ]
        work["refs"].extend(
            reference(Path(row["path"])) for row in work["prior_validation_attempts"]
        )
        work["checks"] += [
            q.old.gate(Path(r["stdout_path"]), "required_tools", True, r["passed"])
            for r in receipts
            if not r["passed"]
        ]
        q.progress("owned_validation_before")
        if not args.private_run:
            receipts += execute(plan[1:], raw / "validation")
        q.progress("owned_validation_after")
        coverage = private / "coverage.json"
        if coverage.is_file():
            atomic_json(raw / "owned_coverage.json", json.loads(coverage.read_bytes()))
            work["owned_statement_coverage"] = json.loads(coverage.read_bytes())["totals"]
            work["refs"].append(reference(raw / "owned_coverage.json"))
        findings = qualified.qualify_findings(raw / "finding_controls")
        receipts += findings["receipts"]
        work["finding_audits"] = findings["audits"]
        work["refs"].extend(
            [findings["primitive_reference"], reference(raw / "execution_manifest.json")]
        )
        work.update(
            private_run=args.private_run,
            duration_s=(time.monotonic_ns() - began) / 1e9,
            phase_spans=[
                dict(
                    phase="qualification",
                    started_monotonic_ns=began,
                    ended_monotonic_ns=time.monotonic_ns(),
                    started_wall_ns=wall,
                    duration_s=(time.monotonic_ns() - began) / 1e9,
                )
            ],
        )
        atomic_json(raw / "measurement.json", work)
        provisional = q.build(work, raw, receipts)
        controls = []
        for name in ("negative_checksum", "rehashed_tamper"):
            bad = deepcopy(provisional)
            bad["runtime_changed_score"] = 1 - bad["runtime_changed_score"]
            bad["reproducibility_checksum"] = (
                q.checksum(bad) if name == "rehashed_tamper" else "invalid"
            )
            path = private / (name + ".json")
            atomic_json(path, bad)
            controls.append(
                dict(
                    name=name,
                    argv=[PY, "-u", str(q.ROOT / q.CLI), "--cold-replay", str(path)],
                    deadline=60,
                    expected=1,
                    scope="owned",
                )
            )
        receipts += execute(controls, raw / "cold_controls")
        publish(q.build(work, raw, receipts), output, raw)
    return 0
