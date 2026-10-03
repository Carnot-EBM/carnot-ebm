"""REQ-REPORT-7999: freeze owned commands and keep repository health separate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

Json = dict[str, Any]


def fixture_data(raw: Path) -> Json:
    """Generate producer-side protocol fixtures separately from the independent reducer."""
    from carnot.reporting.typed_validation_7997 import fixture_data as initial
    from carnot.verify import selective_feedback_7998 as producer
    from carnot.verify.learning_causal_audit_7999 import ARMS

    data = initial()
    head, sources = data["heads"]["spline"][0], data["public"]["stream"]
    for row in sources:
        row["q"] = 0.5
    labels = {r["family_id"]: r["y"] for r in data["targets"]["stream"]}
    acquisition = producer.acquire(head, sources, 101)
    trajectories = {}
    for arm in ARMS:
        schedule = [r for r in acquisition if r["arm"] == arm]
        directory = raw / "fixture_states" / arm
        trajectories[arm + "-101"] = dict(
            trajectory=producer.stream(head, sources, schedule, labels.__getitem__, directory),
            state_directory=str(directory),
        )
    public = [
        dict(sources[i], family_id=f"retention-{i}", source_cluster_id=f"retention-{i}")
        for i in range(64)
    ]
    return dict(
        head=head,
        sources=sources,
        seeds=[101],
        acquisition=acquisition,
        trajectories=trajectories,
        stream_targets=labels,
        retention_public=public,
        retention_targets={r["family_id"]: i % 2 for i, r in enumerate(public)},
    )


def freeze(raw: Path, scratch: Path) -> Json:
    """Seal explicit validation commands before natural labels or measurements."""
    from carnot import experiment_7999_v693_learning_causal_audit as e
    from carnot.verify.learning_causal_audit_7999 import CONFIG

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
    atomic_json(src, fixture_data(raw))
    commands: list[Json] = []

    def add(
        name: str, argv: list[str], required: bool = True, cwd: Path = e.ROOT, deadline: int = 300
    ) -> None:
        commands.append(
            dict(name=name, argv=argv, required=required, cwd=str(cwd), deadline_s=deadline)
        )

    consumers = [
        "tests/python/test_selective_feedback_7998.py",
        "tests/python/test_experiment_7998_v693_selective_feedback_learning.py",
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_qwen_energy_calibration_7972.py",
    ]
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
            *consumers,
        ],
        deadline=600,
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
            "--basetemp=" + str(scratch / "E2E-019"),
            "tests/python/test_experiment_7942_v689_sentence_labels.py",
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
    """Keep actual exit codes, logs and elapsed heartbeats for every named child."""
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
    """Require nonempty owned-statement denominators rather than historical coverage."""
    from carnot import experiment_7999_v693_learning_causal_audit as e

    report = json.loads((scratch / "coverage.json").read_text())
    counts = {}
    for path, value in report["files"].items():
        relative = str(Path(path).relative_to(e.ROOT)) if Path(path).is_absolute() else path
        if relative in e.OWNED:
            counts[relative] = value["summary"]
    return counts
