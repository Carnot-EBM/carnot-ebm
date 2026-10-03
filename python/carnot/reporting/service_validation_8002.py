"""REQ-REPORT-8002: freeze owned checks and supervise their actual executions.

Private CLI runs contribute real statement coverage. Repository health remains
a separately named diagnostic so old failures cannot become current successes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.sparse_validation_7996 import execute

Json = dict[str, Any]


def freeze(raw: Path, scratch: Path) -> Json:
    """Name commands and fixture bytes before measurement begins."""
    from carnot import experiment_8002_v693_service_cost as e
    from carnot.reporting import service_cost_8002 as s

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
    atomic_json(src, s.fixture())
    commands: list[Json] = []

    def add(
        name: str, argv: list[str], required: bool = True, cwd: Path = e.ROOT, deadline: int = 300
    ) -> None:
        commands.append(
            dict(name=name, argv=argv, required=required, cwd=str(cwd), deadline_s=deadline)
        )

    consumers = [
        "tests/python/test_service_cost_7989.py",
        "tests/python/test_experiment_7989_v692_service_cost.py",
        "tests/python/test_sparse_energy_7996.py",
        "tests/python/test_selective_feedback_7998.py",
        "tests/python/test_primary_publication_7928.py",
    ]
    add(
        "owned_unit_consumers",
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
    add(
        "E2E-019",
        [
            py,
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--basetemp=" + str(scratch / "E2E-019"),
            "tests/python/test_experiment_7942_v689_sentence_labels.py",
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
            str(scratch / "blocked" / out.name),
        ],
    )
    add(
        "cold_replay_cli",
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
        config=s.CONFIG,
        code_paths=e.OWNED,
        environment=dict(
            COVERAGE_FILE=str(scratch / ".coverage-full"),
            JAX_PLATFORMS="cpu",
            PYTHONUNBUFFERED="1",
            OPENBLAS_NUM_THREADS="1",
        ),
    )
    atomic_json(raw / "validation_manifest.json", manifest)
    return manifest


def coverage_counts(scratch: Path) -> Json:
    """Retain exact nonempty covered and executable statement counts."""
    report = json.loads((scratch / "coverage.json").read_bytes())
    return {k: v["summary"] for k, v in report["files"].items()}
