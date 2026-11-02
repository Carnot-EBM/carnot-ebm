"""REQ-REPORT-7996: freeze checks before measurement and retain real child logs.

All mutable test data lives outside the checkout. The existing supervisor
prints elapsed-time heartbeats while a child is silent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

Json = dict[str, Any]


def freeze(raw: Path, scratch: Path) -> Json:
    """Pin explicit commands and fixture bytes before any numerical measurement."""
    from carnot import experiment_7996_v693_sparse_energy_training as e
    from carnot.reporting.multivariate_validation_7982 import fixture_data
    from carnot.verify.sparse_energy_7996 import CONFIG

    py, cov = str(e.ROOT / ".venv/bin/python"), str(e.ROOT / ".venv/bin/coverage")
    cli = str(e.ROOT / e.OWNED[-1])
    include = ",".join(str(e.ROOT / p) for p in e.OWNED)
    prefix = [
        cov,
        "run",
        "--parallel-mode",
        "--data-file=" + str(scratch / ".coverage"),
        "--include=" + include,
    ]
    src = scratch / "fixture.json"
    atomic_json(src, {k: v for k, v in fixture_data().items() if k in ("fit", "tune")})
    out = scratch / "success" / (e.NAME + ".json")
    commands: list[Json] = []

    def add(
        name: str, argv: list[str], required: bool = True, cwd: Path = e.ROOT, deadline: int = 300
    ) -> None:
        commands.append(
            dict(name=name, argv=argv, required=required, cwd=str(cwd), deadline_s=deadline)
        )

    tests = e.TESTS + [
        "tests/python/test_multivariate_energy_7982.py",
        "tests/python/test_qwen_energy_calibration_7972.py",
        "tests/python/test_evidence_features_7980.py",
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_experiment_7942_v689_sentence_labels.py",
    ]
    add(
        "unit_consumers_e2e015_019",
        prefix
        + [
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--basetemp=" + str(scratch / "pytest"),
            *tests,
            "-q",
        ],
        deadline=600,
    )
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
        "private_historical_cli",
        prefix
        + [
            cli,
            "--validation-worker",
            "--output",
            str(scratch / "historical" / (e.NAME + ".json")),
        ],
    )
    add(
        "cold_replay",
        ["/usr/bin/env", "-u", "PYTHONPATH", *prefix, cli, "--cold-replay", str(out)],
        cwd=scratch,
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
        owned_files=e.OWNED,
        tests=tests,
        config=CONFIG,
        fixture_sha256=sha256_file(src),
        environment=dict(
            COVERAGE_FILE=str(scratch / ".coverage-full"),
            JAX_PLATFORMS="cpu",
            PYTHONUNBUFFERED="1",
            OPENBLAS_NUM_THREADS="1",
        ),
    )
    atomic_json(raw / "validation_manifest.json", manifest)
    return manifest


def execute(manifest: Json, raw: Path) -> list[Json]:
    """Run bounded children sequentially so receipt order matches the frozen plan."""
    receipts = []
    for item in manifest["commands"]:
        print(f"[exp7996] subprocess_begin={item['name']}", flush=True)
        receipt = run_commands(
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
        receipt["required"] = item["required"]
        receipts.append(receipt)
        print(f"[exp7996] subprocess_end={item['name']} exit={receipt['exit_code']}", flush=True)
    atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
    return receipts


def coverage_counts(scratch: Path) -> Json:
    """Retain nonempty denominators for every changed implementation file."""
    from carnot import experiment_7996_v693_sparse_energy_training as e

    report = json.loads((scratch / "coverage.json").read_text())
    result = {}
    for path, value in report["files"].items():
        relative = str(Path(path).relative_to(e.ROOT)) if Path(path).is_absolute() else path
        if relative in e.OWNED:
            result[relative] = value["summary"]
    return result
