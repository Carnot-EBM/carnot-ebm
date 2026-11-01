"""REQ-VERIFY-8205: freeze validation, replay primitives and publish checked bytes.

The private parent remains alive around every pytest child. Production checks
are explicit; fixture CLI controls never count as new scientific evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import v709_execution as x
from carnot.reporting import v709_qualification as q
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import build_scoped_commands
from carnot.reporting.primary_publication import publish_primary
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = q.ROOT
NAME = "experiment_8205_v709_contract_consumer_qualification"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v709_qualification_8205.py"
OWNED = [
    "python/carnot/reporting/v709_qualification.py",
    "python/carnot/reporting/v709_execution.py",
    "python/carnot/reporting/v709_runner.py",
    CLI,
]


def commands(private: Path) -> list[Json]:
    """Reuse the scoped plan and include real CLI coverage through its subprocess hook."""
    (private / "pytest").mkdir(parents=True, exist_ok=True)
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    specs = build_scoped_commands(
        ROOT,
        [TEST],
        OWNED[:-1],
        static_paths=[CLI],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage",
    )
    cov = str(ROOT / ".venv/bin/coverage")
    replacements = dict(
        changed_module_coverage=[
            cov,
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
            TEST,
            "--basetemp=" + str(private / "covered"),
        ],
        changed_module_coverage_report=[cov, "combine", "--rcfile=" + str(config)],
        changed_module_mypy=[
            str(ROOT / ".venv/bin/mypy"),
            "--strict",
            "--follow-imports=skip",
            *OWNED,
        ],
        scoped_spec_coverage=[
            str(ROOT / ".venv/bin/python"),
            "scripts/check_spec_coverage.py",
            "--files",
            TEST,
        ],
    )
    plan = [
        dict(
            name=s.name,
            argv=replacements.get(s.name, list(s.argv)),
            deadline=240,
            expected=0,
            scope="owned",
        )
        for s in specs
    ]
    plan.extend(
        [
            dict(
                name="coverage_report",
                argv=[
                    cov,
                    "json",
                    "--rcfile=" + str(config),
                    "--fail-under=100",
                    "-o",
                    str(private / "coverage.json"),
                ],
                deadline=30,
                expected=0,
                scope="owned",
            ),
            dict(
                name="consumers_E2E018",
                argv=[
                    str(ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    "tests/python/test_primary_publication_7928.py",
                    "tests/python/test_conductor_gates.py",
                    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                    "--basetemp=" + str(private / "consumers"),
                ],
                deadline=240,
                expected=0,
                scope="owned",
            ),
            dict(
                name="full_python_suite",
                argv=[str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
                deadline=180,
                expected=0,
                scope="repository_health",
            ),
        ]
    )
    return plan


def replay(path: Path) -> Json:
    """Rehash custody and reduce original primitives in a fresh process."""
    value = json.loads(path.read_bytes())
    for ref in [
        value["work_reference"],
        *value["raw_shard_hashes"],
        *value["code_config_hashes"],
        *value["source_artifact_hashes"],
    ]:
        named = ref.get("snapshot_path", ref.get("path"))
        if ref.get("exists", True) and sha256_file(Path(named)) != ref["sha256"]:
            raise ValueError("evidence_hash_drift")
    for receipt in value["validation_receipts"]:
        for label in ("stdout", "stderr"):
            if sha256_file(Path(receipt[label + "_path"])) != receipt[label + "_sha256"]:
                raise ValueError("log_hash_drift")
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    rebuilt = q.build(work, value["validation_receipts"], fixture=value["fixture_validation_scope"])
    for key, observed in rebuilt.items():
        if value[key] != observed:
            raise ValueError("primitive_reduction_drift:" + key)
    snaps = work["contract"]["authority_snapshots"]
    with tempfile.TemporaryDirectory(prefix="carnot8205-replay-") as directory:
        private = Path(directory)
        paths = [
            Path(snaps[k].get("snapshot_path", private / k)) for k in ("design", "staged", "active")
        ]
        contract = q.assess(*paths, private / "authority")
        for key in ("activated", "contract_rows", "canonical_tasks_sha256"):
            if contract[key] != work["contract"][key]:
                raise ValueError("authority_reduction_drift:" + key)
    for ref in value["source_artifact_hashes"]:
        if (
            not work["precondition_failures"]
            and ref.get("exists")
            and Path(ref["source_path"]).name == "research-complete.yaml"
        ):
            import yaml

            archive = yaml.safe_load(Path(ref["snapshot_path"]).read_bytes())
            finished = next(m for m in archive["milestones"] if m["id"] == "2026.10.708")
            if [r["original_archive_row"] for r in value["task_dispositions"]] != finished["tasks"]:
                raise ValueError("historical_reduction_drift")
    return dict(passed=True, rows_checksum=canonical_hash(rebuilt["rows"]))


def terminal_plan(path: Path) -> list[Json]:
    """Freeze exact terminal operands before any measurement begins."""
    py = str(ROOT / ".venv/bin/python")
    specs = [
        dict(name=name, argv=argv, deadline=180, expected=0, scope="terminal")
        for name, argv in (
            ("cold_replay", [py, "-u", str(ROOT / CLI), "--cold-replay", str(path)]),
            (
                "adversarial_verify",
                [py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)],
            ),
            (
                "strict_rows",
                [py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)],
            ),
        )
    ]
    return specs


def terminal_checks(path: Path, logs: Path) -> Json:
    """Require fresh replay and both unchanged terminal auditors before publication."""
    rows = x.execute(terminal_plan(path), logs)
    return dict(passed=all(r["passed"] for r in rows), checks=rows)


def precondition_command() -> Json:
    """Record literal Python and tool availability before dependent work starts."""
    return dict(
        name="environment_preconditions",
        argv=[
            str(ROOT / ".venv/bin/python"),
            "-c",
            "import json,sys,os,pytest,coverage,yaml; from pathlib import Path; p=Path(sys.executable).parent; tools={n:os.access(p/n,os.X_OK) for n in ['python','pytest','coverage','ruff','mypy']}; print(json.dumps({'python':sys.version,'executable':sys.executable,'pytest':pytest.__version__,'coverage':coverage.__version__,'yaml':yaml.__version__,'tools':tools})); raise SystemExit(int(not all(tools.values())))",
        ],
        deadline=30,
        expected=0,
        scope="preconditions",
    )


def main(argv: list[str] | None = None) -> int:
    """Retain invocation scratch until all child and terminal validation finishes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261006"], default="20261006")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--private-fixture", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    x.progress("start", 0, 1)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or ROOT / "results" / (NAME + ".json")).absolute()
        if args.private_fixture and output.is_relative_to(ROOT / "results"):
            raise ValueError("private_fixture_requires_private_output")
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        start = time.monotonic_ns()
        with tempfile.TemporaryDirectory(prefix="carnot8205-invocation-") as directory:
            private = Path(directory)
            probe = private / "writable_probe"
            probe.write_bytes(b"actual private storage")
            runtime = dict(
                python=os.sys.version,
                executable=os.sys.executable,
                private_parent=str(private),
                mode=oct(private.stat().st_mode & 0o777),
                writable=probe.read_bytes() == b"actual private storage",
            )
            plan = [] if args.private_fixture else commands(private)
            control_plan = x.pytest_plan(private / "child_controls")
            terminal_specs = terminal_plan(
                output.parent / "raw" / output.stem / "terminal_candidate.json"
            )
            preflight = precondition_command()
            atomic_json(
                raw / "validation_manifest.json",
                dict(
                    commands=plan,
                    owned=OWNED,
                    terminal_commands=terminal_specs,
                    child_controls=control_plan,
                    precondition_command=preflight,
                    runtime=runtime,
                    frozen_before_measurement_ns=time.monotonic_ns(),
                ),
            )
            precondition_receipt = x.execute([preflight], raw / "precondition_logs")[0]
            x.progress("measurement_before", 0, 13)
            work = q.measure(
                args.root,
                raw,
                initial_failures=[]
                if precondition_receipt["passed"]
                else [
                    dict(
                        upstream_id=q.TASK,
                        path=precondition_receipt["stderr_path"],
                        hash=precondition_receipt["stderr_sha256"],
                        artifact_field="environment_preconditions.exit_code",
                        op="==",
                        expected=0,
                        observed=precondition_receipt["exit_code"],
                    )
                ],
            )
            x.progress("measurement_after", len(work["task_dispositions"]), 0)
            work["runtime_preconditions"] = runtime
            validations = []
            health = []
            if not work["precondition_failures"]:
                validations = x.execute(control_plan, raw / "child_logs")
                if not args.private_fixture:
                    os.environ["CARNOT_8205_COVERAGE_CONFIG"] = str(private / "coverage.ini")
                    validations.extend(
                        x.execute(
                            [s for s in plan if s["scope"] == "owned"], raw / "validation_logs"
                        )
                    )
                    health = x.execute(
                        [s for s in plan if s["scope"] == "repository_health"], raw / "health_logs"
                    )
                    del os.environ["CARNOT_8205_COVERAGE_CONFIG"]
            value = q.build(work, validations, fixture=args.private_fixture)
            atomic_json(raw / "work.json", work)
            atomic_json(raw / "primitive_rows.json", value["rows"])
            coverage_path = private / "coverage.json"
            coverage_value = (
                json.loads(coverage_path.read_bytes()) if coverage_path.is_file() else {}
            )
            atomic_json(raw / "coverage.json", coverage_value)
            value.update(
                duration_s=(time.monotonic_ns() - start) / 1e9,
                precondition_command_receipt=precondition_receipt,
                measurement_clocks=dict(
                    started_monotonic_ns=start,
                    ended_monotonic_ns=time.monotonic_ns(),
                    owner_pid=os.getpid(),
                ),
                runtime_preconditions=runtime,
                repository_health=dict(owned=False, receipts=health),
                coverage_statement_counts=coverage_value.get("files", {}),
                code_config_hashes=[
                    dict(path=str(ROOT / p), sha256=sha256_file(ROOT / p)) for p in [*OWNED, TEST]
                ],
                work_reference=dict(
                    path=str(raw / "work.json"), sha256=sha256_file(raw / "work.json")
                ),
                raw_shard_hashes=[
                    dict(path=str(p), sha256=sha256_file(p))
                    for p in sorted(raw.rglob("*"))
                    if p.is_file()
                ],
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
            )
            value["reproducibility_checksum"] = canonical_hash(
                [value["work_reference"], value["code_config_hashes"]]
            )
            value["field_principles"] = {
                k: "Exact current execution evidence cannot repair failed historical science."
                for k in value
            }
            value = normalize_artifact_for_template_write(value)
            x.progress("publication_before", 13, 1)
            atomic_json(raw / "candidate.json", value)

            def validate(_candidate: Path) -> Json:
                rows = x.execute(terminal_specs, raw / "terminal_logs")
                return dict(passed=all(r["passed"] for r in rows), checks=rows)

            publication = publish_primary(output, value, validate)
            atomic_json(
                raw / "terminal_validation.json",
                dict(
                    publication=publication, required_checks_passed=value["required_checks_passed"]
                ),
            )
        x.progress("complete", 13, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
