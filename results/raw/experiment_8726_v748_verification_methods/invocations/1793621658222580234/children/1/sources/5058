"""REQ-REPORT-7982: freeze bounded checks and retain observed child receipts.

Mutable coverage and pytest outputs stay outside the checkout. Logs are copied
only after each child exits; failed checks remain part of the declared ledger.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

Json = dict[str, Any]


def fixture_data() -> Json:
    """Separable labels exercise wiring and are explicitly circular evidence."""
    return {
        role: [
            dict(
                family_id=f"{role}-{i}",
                source_cluster_id=f"{role}-{i}",
                q=0.5,
                features=[float(i % 2)] * 8,
                y=i % 2,
                status="completed",
            )
            for i in range(n)
        ]
        for role, n in dict(fit=128, tune=32, policy_design=32).items()
    }


def freeze(raw: Path, scratch: Path) -> Json:
    """Manifest identity precedes every child fit and the primary measurement."""
    from carnot import experiment_7982_v692_multivariate_energy as e

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
    atomic_json(src, fixture_data())
    out = scratch / "success" / f"{e.NAME}.json"
    commands: list[Json] = []

    def add(
        name: str, argv: list[str], required: bool = True, cwd: Path = e.ROOT, deadline: int = 300
    ) -> None:
        commands.append(
            dict(name=name, argv=argv, required=required, cwd=str(cwd), deadline_s=deadline)
        )

    tests = e.TESTS + [
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
    )
    add(
        "private_fitting_cli",
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
            str(scratch / "blocked" / f"{e.NAME}.json"),
        ],
    )
    add(
        "private_live_cli",
        prefix + [cli, "--validation-worker", "--output", str(scratch / "live" / f"{e.NAME}.json")],
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
        fixture_sha256=sha256_file(src),
        environment=dict(COVERAGE_FILE=str(scratch / ".coverage-full"), JAX_PLATFORMS="cpu"),
    )
    atomic_json(raw / "validation_manifest.json", manifest)
    return manifest


def execute(manifest: Json, raw: Path) -> list[Json]:
    """The qualified supervisor prints heartbeats while a bounded child is silent."""
    receipts = []
    for item in manifest["commands"]:
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
            extra_env=manifest.get("environment", {}),
            heartbeat_s=30,
        )[0]
        receipt["required"] = item["required"]
        receipts.append(receipt)
    atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
    return receipts


def coverage_counts(scratch: Path) -> Json:
    """Nonempty per-file statement counts distinguish coverage from an empty report."""
    from carnot import experiment_7982_v692_multivariate_energy as e

    report = json.loads((scratch / "coverage.json").read_text())
    result = {}
    for path, value in report["files"].items():
        relative = str(Path(path).relative_to(e.ROOT)) if Path(path).is_absolute() else path
        if relative in e.OWNED:
            result[relative] = value["summary"]
    return result
