"""REQ-REPORT-8003: frozen private checks include real CLI statement coverage.

One repository health run remains a diagnostic. Owned unit, static, coverage
and E2E checks remain required regardless of historical failures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file

Json = dict[str, Any]


def freeze(raw: Path, scratch: Path, health_receipt: Path | None = None) -> Json:
    """Fix argv, fixture bytes and private output paths before numerical work."""
    from carnot import experiment_8003_v693_hardware_sparse_boundary as e
    from carnot.reporting.hardware_sparse_8003 import fixture
    from carnot.verify.fixedpoint_sparse_8003 import CONFIG

    py, cov = str(e.ROOT / ".venv/bin/python"), str(e.ROOT / ".venv/bin/coverage")
    include = ",".join(str(e.ROOT / p) for p in e.OWNED)
    prefix = [
        cov,
        "run",
        "--parallel-mode",
        "--data-file=" + str(scratch / ".coverage"),
        "--include=" + include,
    ]
    cli = str(e.ROOT / e.OWNED[-1])
    unit_temp = Path("/tmp/carnot-8003-validation") / scratch.name / "pytest"
    unit_temp.parent.mkdir(parents=True, exist_ok=True)
    src = scratch / "fixture.json"
    atomic_json(src, fixture())
    out = scratch / "success" / (e.NAME + ".json")
    commands: list[Json] = []

    def add(
        name: str, argv: list[str], required: bool = True, cwd: Path = e.ROOT, deadline: int = 300
    ) -> None:
        commands.append(
            dict(name=name, argv=argv, required=required, cwd=str(cwd), deadline_s=deadline)
        )

    tests = e.TESTS + [
        "tests/python/test_experiment_7977_v691_hardware_evidence.py",
        "tests/python/test_experiment_7990_v692_hardware_evidence.py",
        "tests/python/test_sparse_energy_7996.py",
        "tests/python/test_service_cost_8002.py",
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
            "--basetemp=" + str(unit_temp),
            *tests,
            "-q",
        ],
        deadline=600,
    )
    for name, test in (
        ("E2E-015", "tests/python/test_source_boundary_7852.py"),
        ("E2E-019", "tests/python/test_experiment_7942_v689_sentence_labels.py"),
    ):
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
            str(scratch / "blocked" / out.name),
        ],
    )
    add(
        "private_historical_cli",
        prefix + [cli, "--validation-worker", "--output", str(scratch / "historical" / out.name)],
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
        [str(e.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *e.OWNED],
    )
    add("spec_coverage", [py, "-u", str(e.ROOT / "scripts/check_spec_coverage.py"), *e.TESTS])
    if health_receipt is None:
        add(
            "full_pytest",
            [str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
            required=False,
            deadline=900,
        )
    manifest = dict(
        commands=commands,
        config=CONFIG,
        owned_files=e.OWNED,
        environment=dict(
            COVERAGE_FILE=str(scratch / ".coverage-full"),
            JAX_PLATFORMS="cpu",
            PYTHONUNBUFFERED="1",
            OPENBLAS_NUM_THREADS="1",
        ),
    )
    if health_receipt is not None:
        manifest["historical_health_receipt"] = dict(
            path=str(health_receipt),
            sha256=sha256_file(health_receipt),
            scope="earlier_actual_health_invocation_not_an_owned_passing_check",
        )
    atomic_json(raw / "validation_manifest.json", manifest)
    return manifest


def normalize_receipts(receipts: list[Json], manifest: Json) -> None:
    """A receipt log remains readable when its child used a private working directory."""
    directories = {r["name"]: Path(r["cwd"]) for r in manifest["commands"]}
    for receipt in receipts:
        if receipt.get("log_path") and not Path(receipt["log_path"]).is_absolute():
            receipt["log_path"] = str(directories[receipt["name"]] / receipt["log_path"])


def coverage_counts(scratch: Path) -> Json:
    """Require nonempty statement counts for each added implementation file."""
    from carnot import experiment_8003_v693_hardware_sparse_boundary as e

    report = scratch / "coverage.json"
    if not report.is_file():
        return {}
    values = json.loads(report.read_bytes())
    return {
        str(Path(path).relative_to(e.ROOT)) if Path(path).is_absolute() else path: item["summary"]
        for path, item in values["files"].items()
    }
