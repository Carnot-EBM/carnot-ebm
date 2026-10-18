"""REQ-REPORT-7998: freeze real commands and private output paths before labels.

The shared child supervisor prints elapsed time at most thirty seconds apart.
Full repository health is retained separately from checks owned by this task.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference, checked
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.typed_validation_7997 import fixture_data

Json = dict[str, Any]


def freeze(raw: Path, scratch: Path, prior_health: Path | None = None) -> Json:
    """Pin every validation command before measurement and isolate all writes."""
    from carnot import experiment_7998_v693_selective_feedback_learning as e
    from carnot.verify.selective_feedback_7998 import CONFIG

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
    src, out = scratch / "fixture.json", scratch / "success" / (e.NAME + ".json")
    fixture = fixture_data()
    for sources in fixture["public"].values():
        for row in sources:
            row["q"] = 0.5
    atomic_json(src, fixture)
    commands: list[Json] = []

    def add(
        name: str, argv: list[str], required: bool = True, cwd: Path = e.ROOT, deadline: int = 300
    ) -> None:
        commands.append(
            dict(name=name, argv=argv, required=required, cwd=str(cwd), deadline_s=deadline)
        )

    consumers = [
        "tests/python/test_sparse_energy_7996.py",
        "tests/python/test_experiment_7996_v693_sparse_energy_training.py",
        "tests/python/test_typed_development_7997.py",
        "tests/python/test_primary_publication_7928.py",
    ]
    add(
        "owned_unit_and_consumers",
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
            *e.TESTS,
            *consumers,
            "-q",
        ],
        deadline=600,
    )
    for name, test in [
        ("E2E-015", "tests/python/test_source_boundary_7852.py"),
        ("E2E-019", "tests/python/test_experiment_7942_v689_sentence_labels.py"),
    ]:
        add(
            name,
            [
                py,
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(scratch / name),
                test,
                "-q",
            ],
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
    if prior_health:
        add("prior_repository_health_audit", [py, cli, "--check-health-receipt", str(prior_health)])
    else:
        add(
            "full_pytest",
            [str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
            required=False,
            deadline=900,
        )
    manifest = dict(
        commands=commands,
        config=CONFIG,
        code_paths=e.OWNED,
        environment=dict(
            COVERAGE_FILE=str(scratch / ".coverage-full"),
            JAX_PLATFORMS="cpu",
            PYTHONUNBUFFERED="1",
            OPENBLAS_NUM_THREADS="1",
        ),
    )
    if prior_health:
        manifest["prior_health_reference"] = reference(prior_health)
    atomic_json(raw / "validation_manifest.json", manifest)
    return manifest


def execute(manifest: Json, raw: Path) -> list[Json]:
    """Save actual child exit codes and logs; elapsed heartbeats bound silence."""
    receipts = []
    for item in manifest["commands"]:
        print(f"[exp7998] subprocess_begin={item['name']}", flush=True)
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
        print(f"[exp7998] subprocess_end={item['name']} exit={receipt['exit_code']}", flush=True)
    if "prior_health_reference" in manifest:
        from carnot import experiment_7998_v693_selective_feedback_learning as e

        prior = e.health_receipt(checked(manifest["prior_health_reference"]))
        prior.update(required=False, name="prior_full_pytest")
        receipts.append(prior)
    atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
    return receipts


def coverage_counts(scratch: Path) -> Json:
    """Count only new files with a nonempty statement denominator."""
    from carnot import experiment_7998_v693_selective_feedback_learning as e

    report = json.loads((scratch / "coverage.json").read_text())
    result = {}
    for path, value in report["files"].items():
        relative = str(Path(path).relative_to(e.ROOT)) if Path(path).is_absolute() else path
        if relative in e.OWNED:
            result[relative] = value["summary"]
    return result
