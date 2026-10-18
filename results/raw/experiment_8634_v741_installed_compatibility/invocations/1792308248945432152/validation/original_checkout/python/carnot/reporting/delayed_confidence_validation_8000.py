"""REQ-REPORT-8000: sealed validation commands retain actual private receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

Json = dict[str, Any]


def freeze(raw: Path, scratch: Path) -> Json:
    """Name every required check before opening natural evaluator labels."""
    from carnot import experiment_8000_v693_delayed_confidence as e
    from carnot.verify.delayed_confidence_8000 import CONFIG

    py, cov = str(e.ROOT / ".venv/bin/python"), str(e.ROOT / ".venv/bin/coverage")
    include = ",".join(str(e.ROOT / p) for p in e.OWNED)
    prefix = [
        cov,
        "run",
        "--parallel-mode",
        "--data-file=" + str(scratch / ".coverage"),
        "--include=" + include,
    ]
    src, out = scratch / "fixture.json", scratch / "success" / (e.NAME + ".json")
    atomic_json(
        src,
        dict(
            calibration=[
                dict(family_id=f"c{i}", source_cluster_id=f"c{i}", p=0.1, y=0, status="completed")
                for i in range(64)
            ],
            stream=[
                dict(family_id=f"s{i}", source_cluster_id=f"s{i}", p=0.1, status="completed")
                for i in range(256)
            ],
            targets={f"s{i}": int(i % 10 == 0) for i in range(256)},
        ),
    )
    commands: list[Json] = []

    def add(
        name: str, argv: list[str], required: bool = True, cwd: Path = e.ROOT, deadline: int = 600
    ) -> None:
        commands.append(
            dict(name=name, argv=argv, required=required, cwd=str(cwd), deadline_s=deadline)
        )

    add(
        "owned_unit_and_consumers",
        prefix
        + [
            "-m",
            "pytest",
            "-n0",
            "-o",
            "addopts=",
            "--no-cov",
            "-q",
            "--basetemp=" + str(scratch / "pytest"),
            *e.TESTS,
            "tests/python/test_qwen_energy_calibration_7972.py",
            "tests/python/test_primary_publication_7928.py",
        ],
    )
    cli = str(e.ROOT / e.OWNED[-1])
    add(
        "private_success_cli",
        prefix + [cli, "--fixture-input", str(src), "--validation-worker", "--output", str(out)],
    )
    add(
        "private_blocked_cli",
        prefix
        + [
            cli,
            "--root",
            str(scratch / "absent"),
            "--validation-worker",
            "--output",
            str(scratch / "blocked" / (e.NAME + ".json")),
        ],
    )
    add(
        "cold_replay",
        ["/usr/bin/env", "-u", "PYTHONPATH", *prefix, cli, "--cold-replay", str(out)],
        cwd=scratch,
    )
    add(
        "E2E-019",
        [
            py,
            "-m",
            "pytest",
            "-n0",
            "-o",
            "addopts=",
            "--no-cov",
            "-q",
            "--basetemp=" + str(scratch / "e2e"),
            "tests/python/test_experiment_7942_v689_sentence_labels.py",
        ],
    )
    add(
        "coverage_combine",
        [cov, "combine", "--data-file=" + str(scratch / ".coverage"), str(scratch)],
    )
    add(
        "coverage_report",
        [
            cov,
            "report",
            "--data-file=" + str(scratch / ".coverage"),
            "--include=" + include,
            "--fail-under=100",
        ],
    )
    add(
        "coverage_json",
        [
            cov,
            "json",
            "--data-file=" + str(scratch / ".coverage"),
            "--include=" + include,
            "-o",
            str(scratch / "coverage.json"),
        ],
    )
    add("ruff_check", [str(e.ROOT / ".venv/bin/ruff"), "check", *e.OWNED, *e.TESTS])
    add("ruff_format", [str(e.ROOT / ".venv/bin/ruff"), "format", "--check", *e.OWNED, *e.TESTS])
    add(
        "strict_mypy",
        [str(e.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *e.OWNED[:-1]],
    )
    add("spec_coverage", [py, "-u", str(e.ROOT / "scripts/check_spec_coverage.py"), *e.TESTS])
    add(
        "full_pytest",
        [str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
        required=False,
        deadline=900,
    )
    manifest = dict(
        commands=commands,
        config=CONFIG,
        environment=dict(
            JAX_PLATFORMS="cpu",
            PYTHONUNBUFFERED="1",
            OPENBLAS_NUM_THREADS="1",
            COVERAGE_FILE=str(scratch / ".coverage-full"),
            CARNOT_AUDIT_COVERAGE_FILE=str(scratch / ".coverage"),
        ),
    )
    atomic_json(raw / "validation_manifest.json", manifest)
    return manifest


def execute(manifest: Json, raw: Path) -> list[Json]:
    """Supervise each child with thirty-second heartbeats and bounded deadlines."""
    receipts = []
    for item in manifest["commands"]:
        r = run_commands(
            Path(item["cwd"]),
            [
                CommandSpec(
                    item["name"],
                    tuple(item["argv"]),
                    "owned" if item["required"] else "repository_health",
                    item["deadline_s"],
                )
            ],
            log_dir=raw / "validation_logs" / item["name"],
            extra_env=manifest["environment"],
            heartbeat_s=30,
        )[0]
        r["required"] = item["required"]
        receipts.append(r)
    atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
    return receipts


def coverage_counts(scratch: Path) -> Json:
    """Statement denominators include the real CLI and cannot be empty."""
    from carnot import experiment_8000_v693_delayed_confidence as e

    report = json.loads((scratch / "coverage.json").read_text())
    counts = {}
    for path, value in report["files"].items():
        relative = str(Path(path).relative_to(e.ROOT)) if Path(path).is_absolute() else path
        if relative in e.OWNED:
            counts[relative] = value["summary"]
    return counts
