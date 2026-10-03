"""REQ-REPORT-7997: freeze commands before science and keep real child receipts.

Private scratch holds all mutable fixtures and coverage. The existing child
supervisor prints elapsed-time heartbeats every thirty seconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

Json = dict[str, Any]


def fixture_data() -> Json:
    """Separate scripted labels exercise CLI custody, never natural benefit."""
    heads = {
        a: [
            dict(
                arm=a,
                seed=17,
                parameters=[0.0] * n,
                decay_scale=1.0,
                temperature=1.0,
                scaler=dict(minimum=[0.0] * 9, maximum=[1.0] * 9),
            )
        ]
        for a, n in dict(spline=109, logistic=10, mlp=89).items()
    }
    heads["scalar"] = [dict(arm="platt", seed=17, parameters=[2.0, -1.0], temperature=1.0)]
    public, targets = {}, {}
    for role, n in [("calibration", 64), ("stream", 256)]:
        public[role] = [
            dict(
                family_id=f"{role}-{i}",
                source_cluster_id=f"{role}-{i}",
                q=float(i % 2),
                features=[0.5] * 8,
                status="completed",
            )
            for i in range(n)
        ]
        targets[role] = [dict(family_id=f"{role}-{i}", y=i % 2) for i in range(n)]
    return dict(heads=heads, public=public, targets=targets)


def freeze(raw: Path, scratch: Path, prior_exposure: Path | None = None) -> Json:
    """Pin validation commands and fixture bytes before any target evaluation."""
    from carnot import experiment_7997_v693_typed_development_decisions as e
    from carnot.verify.typed_development_7997 import CONFIG

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
    out = scratch / "success" / (e.NAME + ".json")
    commands: list[Json] = []

    def add(
        name: str, argv: list[str], required: bool = True, cwd: Path = e.ROOT, deadline: int = 300
    ) -> None:
        commands.append(
            dict(name=name, argv=argv, required=required, cwd=str(cwd), deadline_s=deadline)
        )

    tests = e.TESTS + [
        "tests/python/test_sparse_energy_7996.py",
        "tests/python/test_experiment_7996_v693_sparse_energy_training.py",
        "tests/python/test_qwen_energy_calibration_7972.py",
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_experiment_7942_v689_sentence_labels.py",
    ]
    add(
        "unit_consumers_e2e019",
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
        "cold_replay",
        ["/usr/bin/env", "-u", "PYTHONPATH", *prefix, cli, "--cold-replay", str(out)],
        cwd=scratch,
    )
    if prior_exposure:
        add("stream_target_exposure_order", prefix + [cli, "--check-exposure", str(prior_exposure)])
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
        code_paths=e.OWNED,
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
    """Sequential children preserve actual order, exit codes and bounded silence."""
    receipts = []
    for item in manifest["commands"]:
        print(f"[exp7997] subprocess_begin={item['name']}", flush=True)
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
        print(f"[exp7997] subprocess_end={item['name']} exit={receipt['exit_code']}", flush=True)
    atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
    return receipts


def coverage_counts(scratch: Path) -> Json:
    """Only new files count toward this invocation's coverage denominator."""
    from carnot import experiment_7997_v693_typed_development_decisions as e

    report = json.loads((scratch / "coverage.json").read_text())
    result = {}
    for path, value in report["files"].items():
        relative = str(Path(path).relative_to(e.ROOT)) if Path(path).is_absolute() else path
        if relative in e.OWNED:
            result[relative] = value["summary"]
    return result
