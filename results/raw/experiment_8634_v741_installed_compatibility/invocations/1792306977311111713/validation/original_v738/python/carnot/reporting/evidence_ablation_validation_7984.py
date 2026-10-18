"""REQ-REPORT-7984: freeze bounded validation before any science measurement.

Mutable pytest and coverage files stay in private scratch. The existing child
supervisor provides heartbeats and archives logs only after each child exits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.verify import evidence_features_7980 as features

Json = dict[str, Any]


def fixture() -> Json:
    """Synthetic disjoint sources test wiring and carry no natural-data evidence."""
    public, data = {}, {}
    for role, count in dict(fit=128, tune=32, policy_design=32, evaluation=64).items():
        public[role], data[role] = [], []
        for i in range(count):
            source = f"Only {role} item {i:03d} has {i % 2} units. No other item does."
            answer = f"Only {role} item {i:03d} has {i % 2} units."
            row = dict(
                family_id=f"{role}-{i}",
                source_bytes=source.encode().hex(),
                answer_bytes=answer.encode().hex(),
            )
            public[role].append(row)
            data[role].append(
                dict(
                    family_id=row["family_id"],
                    source_cluster_id=features.normalized(source.encode()),
                    q=0.5,
                    features=features.extract(row)["values"],
                    y=i % 2,
                    status="completed",
                )
            )
    return dict(public=public, data=data)


def freeze(raw: Path, scratch: Path) -> Json:
    """Declare actual commands, paths and scope before fitting or benchmark work."""
    from carnot import experiment_7984_v692_evidence_ablation as e

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
    atomic_json(src, fixture())
    commands: list[Json] = []

    def add(
        name: str,
        argv: list[str],
        *,
        required: bool = True,
        cwd: Path = e.ROOT,
        deadline: int = 300,
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
            str(scratch / "blocked" / (e.NAME + ".json")),
        ],
    )
    add(
        "private_live_cli",
        prefix
        + [cli, "--validation-worker", "--output", str(scratch / "live" / (e.NAME + ".json"))],
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
            "--show-missing",
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
        environment=dict(COVERAGE_FILE=str(scratch / ".coverage-health"), JAX_PLATFORMS="cpu"),
    )
    atomic_json(raw / "validation_manifest.json", manifest)
    return manifest


def execute(manifest: Json, raw: Path) -> list[Json]:
    """Retain observed exits for every declared check, including failed health checks."""
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
            extra_env=manifest["environment"],
            heartbeat_s=30,
        )[0]
        receipt["required"] = item["required"]
        receipts.append(receipt)
    atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
    return receipts


def coverage_counts(scratch: Path) -> Json:
    """Archive nonempty statement counts after coverage children have exited."""
    from carnot import experiment_7984_v692_evidence_ablation as e

    report = json.loads((scratch / "coverage.json").read_text())
    result = {}
    for path, value in report["files"].items():
        relative = str(Path(path).relative_to(e.ROOT)) if Path(path).is_absolute() else path
        if relative in e.OWNED:
            result[relative] = value["summary"]
    return result
