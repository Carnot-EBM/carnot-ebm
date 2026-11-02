"""REQ-VERIFY-8509: bounded validation precedes exact-byte publication."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

import yaml

from carnot.reporting import arc_redirect_effect_8509 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
TESTS = [e.TEST, "tests/python/test_arc_redirect_runner_8509.py"]
HISTORICAL = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]


def gate(check: str, path: Path, field: str, expected: Any, observed: Any) -> Json:
    """An exact comparison preserves missing external operands separately from zero."""
    return dict(
        check=check,
        upstream=e.TASK,
        path=str(path),
        hash=sha256_file(path) if path.is_file() else None,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def preconditions(root: Path, private: Path) -> tuple[list[Json], Json, Json]:
    """The explicit task invocation and independently pinned activated object authorize this CPU audit."""
    e.progress("preconditions_before")
    private.mkdir(parents=True, exist_ok=True)
    failures, hashes = [], {}
    active = root / "research-roadmap.yaml"
    observed = None
    if active.is_file():
        plan = yaml.safe_load(active.read_bytes())
        task = next((t for t in plan["tasks"] if t["id"] == e.TASK), None)
        observed = canonical_hash(task) if task else None
        hashes[str(active)] = sha256_file(active)
    if observed != e.TASK_PIN:
        failures.append(
            gate("activated_task", active, "canonical_complete_task_sha256", e.TASK_PIN, observed)
        )
    for name in [
        *e.OWNED,
        *TESTS,
        "ops/exclusion_manifest.yaml",
        "ops/arc_solve_registry.yaml",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/verification/spec.md",
    ]:
        path = e.ROOT / name
        hashes[str(path)] = sha256_file(path)
    tools = {
        name: (e.ROOT / ".venv/bin" / name).is_file()
        for name in ["python", "pytest", "ruff", "mypy"]
    }
    available = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    disk = shutil.disk_usage(private).free
    mounts = [line.split() for line in Path("/proc/mounts").read_text().splitlines()]
    filesystem = max(
        (m for m in mounts if str(private.absolute()).startswith(m[1].rstrip("/") + "/")),
        key=lambda m: len(m[1]),
    )[2]
    for field, passed, expected, actual in [
        ("tools", all(tools.values()), True, tools),
        ("available_memory_bytes", available >= 2**30, ">=1073741824", available),
        ("disk_backed_scratch", filesystem not in ["tmpfs", "ramfs"], True, filesystem),
        ("free_disk_bytes", disk >= 2**30, ">=1073741824", disk),
    ]:
        if not passed:
            failures.append(gate(field, private, field, expected, actual))
    resources = dict(
        task_sha256=observed,
        task_authorized_by="explicit_user_invocation_and_pinned_activated_task",
        tools=tools,
        available_memory_bytes=available,
        free_disk_bytes=disk,
        filesystem=filesystem,
        private_scratch=str(private),
        optional_cuda_required=False,
        optional_board_required=False,
        measurement_cap_s=3600,
        validation_reserve_s=900,
        task_cap_s=4800,
        child_heartbeat_s=30,
        invocation_argv=list(sys.argv),
        full_V731_design_required_for_this_ungated_task=False,
    )
    e.progress("preconditions_after", len(hashes), 0)
    return failures, resources, hashes


def validate_checks(raw: Path, private: Path) -> list[Json]:
    """Genuine owned coverage and disjoint historical shards retain unchanged assertions."""
    logs = raw / "validation"
    logs.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    config = private / "coverage.ini"
    data = private / ".coverage"
    config.write_text(
        "[run]\nparallel = True\npatch = subprocess\ndata_file = "
        + str(data)
        + "\ninclude =\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED)
        + "[report]\nfail_under = 100\nshow_missing = True\n"
    )
    receipts = []
    union = {}
    # Collect before execution so each historical assertion has one owned shard.
    for index, suite in enumerate(HISTORICAL):
        receipt = child(
            f"collect_{index}",
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                suite,
            ],
            logs,
            deadline=90,
        )
        receipts.append(receipt)
        if not receipt["passed"]:
            atomic_json(logs / "historical_node_union.json", union)
            return receipts
        nodes = [
            line.strip()
            for line in Path(receipt["stdout_path"]).read_text().splitlines()
            if line.startswith("tests/python/") and "::" in line
        ]
        shards = [nodes[i : i + 4] for i in range(0, len(nodes), 4)]
        union[suite] = dict(
            collected=nodes,
            shards=shards,
            union_equals_collection=bool(nodes)
            and sorted(n for s in shards for n in s) == sorted(nodes),
            disjoint=len(nodes) == len(set(nodes)),
        )
        for shard_index, shard in enumerate(shards):
            receipts.append(
                child(
                    f"historical_{index}_{shard_index}",
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-n",
                        "0",
                        "-o",
                        "addopts=",
                        "--no-cov",
                        "-q",
                        *shard,
                        "--basetemp=" + str(private / f"history{index}-{shard_index}"),
                    ],
                    logs,
                    deadline=180,
                )
            )
    atomic_json(logs / "historical_node_union.json", union)
    commands = [
        (
            "owned_tests",
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                *TESTS,
                "--basetemp=" + str(private / "owned"),
            ],
            450,
        ),
        ("ruff_check", [str(e.ROOT / ".venv/bin/ruff"), "check", *e.OWNED, *TESTS], 60),
        (
            "ruff_format",
            [str(e.ROOT / ".venv/bin/ruff"), "format", "--check", *e.OWNED, *TESTS],
            60,
        ),
        (
            "strict_mypy",
            [str(e.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=skip", *e.OWNED],
            120,
        ),
        (
            "spec_coverage",
            [sys.executable, "scripts/check_spec_coverage.py", "--files", *TESTS],
            60,
        ),
        (
            "coverage_combine",
            [
                sys.executable,
                "-m",
                "coverage",
                "combine",
                "--rcfile=" + str(config),
                "--keep",
                str(private),
            ],
            60,
        ),
        (
            "coverage_report",
            [
                sys.executable,
                "-m",
                "coverage",
                "report",
                "--rcfile=" + str(config),
                "--fail-under=100",
            ],
            60,
        ),
        (
            "coverage_json",
            [
                sys.executable,
                "-m",
                "coverage",
                "json",
                "--rcfile=" + str(config),
                "-o",
                str(logs / "owned_coverage.json"),
            ],
            60,
        ),
    ]
    for name, argv, deadline in commands:
        receipts.append(child(name, argv, logs, deadline=deadline))
    coverage_dir = logs / "coverage_shards"
    coverage_dir.mkdir(exist_ok=True)
    for shard in private.glob(".coverage*"):
        shutil.copyfile(shard, coverage_dir / shard.name)
    return receipts


def controls(value: Json, raw: Path) -> list[Json]:
    """Fresh processes distinguish intentional failure from successful rejection of changed evidence."""
    rows = []
    for name, arguments, expected in [
        ("missing", ["--cold-replay", str(raw / "missing.json")], 1),
        ("intentional", ["--deliberate-error"], 7),
    ]:
        rows.append(
            child(
                name,
                [sys.executable, "-u", str(e.ROOT / e.CLI), *arguments],
                raw / "controls",
                deadline=60,
                expected=expected,
            )
        )
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    changed = deepcopy(work)
    if changed.get("source"):
        changed["source"]["primary"]["rows"][0]["supervisor_outcome_rows"][0][
            "resolved_by_levelup"
        ] = True
    else:
        changed["failures"] = []
    path = raw / "tampered_work.json"
    atomic_json(path, changed)
    output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (e.NAME + ".json")
    candidate = e.build(path, value["validation_receipts"], output)
    tampered = raw / "self_consistently_rehashed_tamper.json"
    atomic_json(tampered, candidate)
    rows.append(
        child(
            "rehashed_tamper",
            [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(tampered)],
            raw / "controls",
            deadline=90,
            expected=1,
        )
    )
    return rows


def publish(value: Json, output: Path, raw: Path) -> Json:
    """The unchanged validators check exact candidate bytes; rejected attempts stay on disk."""
    reports = []

    def validate(candidate: Path) -> Json:
        checks = []
        for name, argv in [
            (
                "terminal_replay",
                [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(candidate)],
            ),
            (
                "adversarial",
                [sys.executable, "scripts/adversarial_verify.py", "--json", str(candidate)],
            ),
            (
                "strict_rows",
                [
                    sys.executable,
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ],
            ),
        ]:
            checks.append(child(name, argv, raw / "terminal", deadline=90))
        report = dict(passed=all(r["passed"] for r in checks), checks=checks)
        reports.append(report)
        atomic_json(raw / "publication_attempts.json", dict(attempts=reports))
        return report

    publication = publish_primary(output, value, validate)
    read_bound_sidecar(output, Path(publication["sidecar_path"]))
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="generalization_diagnostic_ready_score",
        expected=value["generalization_diagnostic_ready_score"],
    )
    side = dict(
        publication=publication,
        attempts=reports,
        consumer=consumer,
        passed=reports[-1]["passed"] and consumer["passed"],
    )
    atomic_json(Path(value["terminal_validation_sidecar_path"]), side)
    return side


def main(argv: list[str] | None = None) -> int:
    """Own only this task's deadlines and outputs; historical primaries remain untouched."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_LLM_calls_zero", 0, 32)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261014"], default="20261014")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        try:
            passed = e.replay(json.loads(args.cold_replay.read_bytes()))
        except (OSError, ValueError):
            passed = False
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    started = time.monotonic()
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    output = args.output.absolute()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(prefix="carnot-exp8509-", dir="/var/tmp") as directory:
            private = Path(directory)
            failures, resources, hashes = preconditions(args.root, private)
            source = None
            e.progress("historical_authentication_before", 0, 16)
            try:
                source = e.authenticate(args.root, raw / "inputs")
            except (OSError, ValueError, KeyError) as exc:
                failures.append(
                    gate(
                        "original_sealed_producer",
                        args.root / e.UPSTREAM,
                        "original_sealed_authority",
                        e.UPSTREAM_PIN,
                        str(exc),
                    )
                )
            e.progress("historical_authentication_after", 16 if source else 0, 0 if source else 16)
            fixture_rows = [
                r
                for mode in ["shadow", "applied"]
                for r in e.fixtures(raw / "fixtures" / mode, mode)
            ]
            measurement = time.monotonic() - started
            if measurement >= 3600:
                raise TimeoutError("measurement_deadline")
            work = dict(
                source=source,
                fixture_rows=fixture_rows,
                failures=failures,
                preconditions_checked=resources,
                duration_s=measurement,
                phase_spans=[
                    dict(phase="authentication_and_private_mechanisms", duration_s=measurement)
                ],
                source_artifact_hashes=hashes,
                code_config_hashes={
                    str(e.ROOT / n): sha256_file(e.ROOT / n)
                    for n in [
                        *e.OWNED,
                        "python/carnot/agentic/arc_competition_agent.py",
                        "python/carnot/agentic/arc_trajectory_supervisor.py",
                        "python/carnot/agentic/arc_arm_eligibility.py",
                        "python/carnot/agentic/arc_live_panel_runtime_8384.py",
                        "python/carnot/experiment_7526_v658_arc_eligibility.py",
                    ]
                },
            )
            path = raw / "work.json"
            atomic_json(path, work)
            e.progress("validation_before", 0, 1)
            receipts = validate_checks(raw, private / "validation")
            value = e.build(path, receipts, output)
            receipts.extend(controls(value, raw))
            work["duration_s"] = time.monotonic() - started
            work["phase_spans"].append(
                dict(phase="validation", duration_s=work["duration_s"] - measurement)
            )
            atomic_json(path, work)
            value = e.build(path, receipts, output)
            e.progress("publication_before", 32 if source else 0, 0)
            side = publish(value, output, raw)
            e.progress("publication_after", value["completed_count"], 0)
            return int(not side["passed"])
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
