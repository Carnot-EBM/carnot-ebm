"""REQ-VERIFY-8262: run normally exited validation and retain its actual operands.

Private scratch owns temporary coverage output. Durable copies and a fresh
reader must qualify before the unchanged publisher exposes a primary.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import coverage_custody_8262 as custody
from carnot.reporting import v714_coverage_custody as q
from carnot.reporting.v685_authority_lifecycle import assess_authorities
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v709_execution import child
from carnot.reporting.v710_contract_replay import require_reference, snapshot
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries so a supervisor can see bounded progress."""
    print(f"[exp8262] phase={phase} completed={completed} pending={pending}", flush=True)


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze scoped tests and exact JSON operands before any measurement begins."""
    py, cov, pytest, ruff, mypy = [
        str(q.ROOT / ".venv/bin" / n) for n in ["python", "coverage", "pytest", "ruff", "mypy"]
    ]
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(q.ROOT / n) + "\n" for n in q.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    rc = "--rcfile=" + str(config)
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    rows = [
        ("coverage_custody_tests", [cov, "run", rc, "-m", "pytest", *common, q.TEST], 180),
        ("current_contract_tests", [pytest, *common, q.TEST, "-k", "authority"], 120),
        (
            "consumer_E2E018",
            [
                pytest,
                *common,
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_current_work_receipt.py",
            ],
            360,
        ),
        ("coverage_combine", [cov, "combine", rc], 30),
        ("coverage_report", [cov, "report", rc, "--show-missing", "--fail-under=100"], 30),
        ("coverage_json", [cov, "json", rc, "-o", str(private / "coverage.json")], 30),
        ("ruff_check", [ruff, "check", *q.OWNED, q.TEST], 30),
        ("ruff_format", [ruff, "format", "--check", *q.OWNED, q.TEST], 30),
        (
            "strict_mypy",
            [
                mypy,
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=skip",
                "--ignore-missing-imports",
                *q.OWNED,
            ],
            60,
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", "--files", q.TEST], 30),
    ]
    return dict(
        owned=q.OWNED,
        commands=[dict(name=n, argv=a, expected=0, deadline=d, scope="owned") for n, a, d in rows],
        repository_health=dict(
            name="repository_full_suite",
            argv=[pytest, "tests/python", "-q"],
            expected=0,
            deadline=120,
            scope="diagnostic",
        ),
    )


def execute(spec: Json, logs: Path) -> Json:
    """Reuse process-group deadlines and separate durable stdout and stderr hashes."""
    return dict(
        child(
            spec["name"],
            spec["argv"],
            logs,
            expected=spec["expected"],
            deadline=spec["deadline"],
            heartbeat=20,
            scope=spec["scope"],
        )
    )


def cold_controls(binding: Json, raw: Path) -> list[Json]:
    """Use fresh readers for valid, missing and rehashed contradictory primitives."""
    rows = []
    for name in ["positive", "missing", "rehashed_tamper"]:
        value = deepcopy(binding)
        if name == "missing":
            value["report_path"] = str(raw / "absent.json")
        if name == "rehashed_tamper":
            report = json.loads(Path(value["report_path"]).read_bytes())
            next(iter(report["files"].values()))["summary"]["num_statements"] += 1
            target = raw / "tampered_report.json"
            atomic_json(target, report)
            value.update(report_path=str(target), report_sha256=sha256_file(target))
            saved = json.loads(Path(value["receipt_path"]).read_bytes())
            saved.update(
                {k: v for k, v in value.items() if k not in {"receipt_path", "receipt_sha256"}}
            )
            receipt_path = raw / "tampered_receipt.json"
            atomic_json(receipt_path, saved)
            value.update(receipt_path=str(receipt_path), receipt_sha256=sha256_file(receipt_path))
        path = raw / (name + "_binding.json")
        atomic_json(path, value)
        receipt = execute(
            dict(
                name="coverage_cold_" + name,
                argv=[
                    str(q.ROOT / ".venv/bin/python"),
                    "-u",
                    str(q.ROOT / q.CLI),
                    "--coverage-replay",
                    str(path),
                ],
                expected=0 if name == "positive" else 1,
                deadline=60,
                scope="owned",
            ),
            raw / "logs",
        )
        rows.append(dict(control=name, **receipt))
    return rows


def replay(path: Path) -> Json:
    """Rebuild readiness from primitive reports and independently parsed authorities."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    if checksum != canonical_hash(value):
        raise ValueError("candidate_checksum")
    for ref in [
        value["work_reference"],
        *value["source_artifact_hashes"],
        *value["code_config_hashes"],
        *value["raw_shard_hashes"],
    ]:
        require_reference(ref)
    for receipt in value["validation_receipts"] + value["cold_replay_rows"]:
        for prefix in ["stdout", "stderr"]:
            if sha256_file(Path(receipt[prefix + "_path"])) != receipt[prefix + "_sha256"]:
                raise ValueError("validation_stream_hash")
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    measured = (
        custody.replay(value["coverage_command_receipt"])
        if value["coverage_command_receipt"]
        else {}
    )
    if (
        measured != value["owned_statement_counts"]
        or not value["scratch_removed"]
        or Path(value["scratch_path"]).exists()
    ):
        raise ValueError("durable_coverage_reduction")
    rebuilt = q.reduce(
        work,
        value["validation_receipts"],
        bool(measured),
        bool(value["cold_replay_rows"]) and all(r["passed"] for r in value["cold_replay_rows"]),
    )
    if any(value[k] != v for k, v in rebuilt.items()):
        raise ValueError("primitive_readiness_drift")
    refs = work["refs"]
    for row in work["history"]:
        ref = next(r for r in refs if r["path"] == row["path"])
        source = json.loads(Path(ref["snapshot_path"]).read_bytes()) if ref["exists"] else {}
        if row["honest_verdict"] != source.get("honest_verdict") or row["source_counts"] != {
            k: source.get(k) for k in row["source_counts"]
        }:
            raise ValueError("historical_primitive_drift")
    with TemporaryDirectory(prefix="carnot8262-replay-") as directory:
        private = Path(directory)
        paths = [Path(r.get("snapshot_path", private / str(i))) for i, r in enumerate(refs[:3])]
        if work["contract"]["authority_snapshots"]:
            checked = assess_authorities(
                *paths, private / "check", milestone=q.MILESTONE, first_id=8262, count=14
            )
            for key in ["activated", "canonical_tasks_sha256", "contract_rows"]:
                if checked[key] != work["contract"][key]:
                    raise ValueError("authority_reduction_drift")
            _, tasks = parse_design(paths[0].read_text(), milestone=q.MILESTONE)
            if tasks != work["tasks"]:
                raise ValueError("full_task_primitive_drift")
    if value["MODEL_SPECS"] or value["model_invocation_counts"] != ZERO_INVOCATION_COUNTS:
        raise ValueError("current_model_provenance")
    return dict(passed=True, owned_statement_counts=measured)


def build(
    work: Json,
    raw: Path,
    receipts: list[Json],
    binding: Json,
    cold: list[Json],
    scratch: Path,
    health: Json,
) -> Json:
    """Bind actual execution bytes and explain why administrative scores grant no benefit."""
    measured = custody.replay(binding) if binding else {}
    value = q.reduce(work, receipts, bool(measured), bool(cold) and all(r["passed"] for r in cold))
    execution = raw / "execution_contract.json"
    atomic_json(execution, work["execution_contract"])
    value.update(
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["runtime_checks"],
        duration_s=work["duration_s"]
        + sum(r["duration_s"] for r in receipts + cold)
        + health.get("duration_s", 0),
        random_seed=101,
        invocation_argv=work["invocation_argv"],
        source_artifact_hashes=work["refs"],
        cited_upstream_artifacts=[
            dict(
                r,
                fields_imported=r.get("fields_imported", []),
            )
            for r in work["refs"]
        ],
        code_config_hashes=[
            snapshot(q.ROOT / p, raw / "code", str(i))
            for i, p in enumerate(q.OWNED + [q.TEST] + q.REUSED)
        ],
        work_reference=dict(
            path=str(raw / "measurement.json"), sha256=sha256_file(raw / "measurement.json")
        ),
        raw_shard_hashes=[
            dict(path=str(raw / n), sha256=sha256_file(raw / n))
            for n in [
                "validation_commands.json",
                "validation_receipts.json",
                "execution_contract.json",
            ]
        ],
        phase_spans=[
            dict(
                phase="preconditions_and_authority", duration_s=work["duration_s"], **work["clock"]
            )
        ]
        + [
            dict(
                phase=r["name"],
                duration_s=r["duration_s"],
                started_monotonic_ns=r["started_monotonic_ns"],
                ended_monotonic_ns=r["ended_monotonic_ns"],
                started_wall_ns=r["started_wall_ns"],
            )
            for r in receipts + cold
        ],
        execution_contract_path=str(execution),
        execution_contract_sha256=sha256_file(execution),
        coverage_report_path=binding.get("report_path"),
        coverage_report_sha256=binding.get("report_sha256"),
        coverage_command_receipt=binding,
        owned_statement_counts=measured,
        scratch_removed=not scratch.exists(),
        scratch_path=str(scratch),
        cold_replay_rows=cold,
        repository_health=health,
        scientific_benefit_measured=False,
        external_publication_authorized=False,
        methodology_note="Authenticate immutable science and actual full current authority. Read the explicit measured coverage JSON operand, preserve report, command and code bytes before temporary cleanup, then independently recompute statements in new processes. Private negative controls qualify custody only. H1 and H2 are unmeasured; historical missing outputs grant no scientific benefit.",
        reconciliation_note="Conductor owns ops and traceability reconciliation.",
    )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Bind actual current invocation evidence; execution readiness grants no scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        coverage_command_receipt="Explicit argv, actual exits, clocks, stream hashes and durable report identity.",
        owned_statement_counts="Counts recomputed from primitive coverage line sets and exact owned source bytes.",
        cold_replay_rows="New processes read durable coverage after original scratch deletion, including rehashed tamper.",
        historical_dispositions="Keep missing output distinct from measured zero; never invent historical producer verdicts.",
        current_contract_ready_score="Administrative activation and independently tested full task agreement only.",
        coverage_custody_ready_score="Independently tested durable coverage custody only.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return dict(value)


def terminal(candidate: Path, raw: Path) -> Json:
    """Check a private candidate with unchanged adversarial and strict row validators."""
    with TemporaryDirectory(prefix="carnot8262-terminal-") as directory:
        private = Path(directory) / (q.NAME + ".json")
        private.write_bytes(candidate.read_bytes())
        py = str(q.ROOT / ".venv/bin/python")
        checks = [
            execute(
                dict(name=name, argv=argv, expected=0, deadline=120, scope="terminal"),
                raw / "terminal_logs" / str(time.time_ns()),
            )
            for name, argv in [
                ("cold_replay", [py, "-u", str(q.ROOT / q.CLI), "--cold-replay", str(private)]),
                (
                    "adversarial_verify",
                    [py, str(q.ROOT / "scripts/adversarial_verify.py"), "--json", str(private)],
                ),
                (
                    "strict_rows",
                    [
                        py,
                        str(q.ROOT / "scripts/verdict_row_consistency_lint.py"),
                        "--strict",
                        str(private),
                    ],
                ),
            ]
        ]
    return dict(passed=all(r["passed"] for r in checks), checks=checks)


def run(root: Path, output: Path) -> Json:
    """Run the real branch, remove scratch, cold-read custody and publish checked bytes."""
    progress("start", 0, 14)
    raw = output.absolute().parent / "raw" / q.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    binding: Json = {}
    health: Json = {}
    with TemporaryDirectory(prefix="carnot8262-private-") as directory:
        private = Path(directory)
        specs = manifest(private, raw / "candidate.json")
        health_source = os.environ.get("CARNOT8262_HEALTH_RECEIPT")
        if health_source:
            specs["repository_health_receipt"] = dict(
                path=health_source, sha256=sha256_file(Path(health_source))
            )
        atomic_json(raw / "validation_commands.json", specs)
        probe = private / "write_probe"
        probe.write_bytes(b"private scratch")
        runtime: list[Json] = [
            dict(
                path=str(probe),
                artifact_field="private_scratch_writable",
                expected=True,
                observed=probe.read_bytes() == b"private scratch",
                passed=True,
            )
        ]
        for name in ["python", "coverage", "pytest", "ruff", "mypy"]:
            path = q.ROOT / ".venv/bin" / name
            runtime.append(
                dict(
                    path=str(path),
                    artifact_field="required_tool",
                    expected=True,
                    observed=path.is_file(),
                    passed=path.is_file(),
                )
            )
        progress("preconditions_before")
        work = q.measure(root, raw)
        work.update(runtime_checks=runtime, invocation_argv=list(sys.argv))
        work["failures"].extend(
            q.failure(Path(r["path"]), r["artifact_field"], True, r["observed"])
            for r in runtime
            if not r["passed"]
        )
        progress("preconditions_after", 14)
        receipts = []
        for index, spec in enumerate(
            specs["commands"] if all(r["passed"] for r in runtime) else []
        ):
            progress("validation", index, len(specs["commands"]) - index)
            receipt = execute(spec, raw / "logs")
            receipts.append(receipt)
            if spec["name"] == "coverage_json":
                try:
                    binding = custody.preserve(
                        q.ROOT, spec, receipt, specs["owned"], raw / "coverage"
                    )
                except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                    source = (
                        Path(spec["argv"][spec["argv"].index("-o") + 1])
                        if "-o" in spec["argv"]
                        else raw / "missing-coverage-operand"
                    )
                    work.setdefault("owned_failures", []).append(
                        q.failure(source, "durable_measured_coverage", True, str(error))
                    )
                    snapshot(source, raw / "failed_coverage", "report")
                    atomic_json(raw / "failed_coverage/command_receipt.json", receipt)
                    receipts.append(
                        dict(
                            receipt,
                            name="coverage_custody_failure",
                            passed=False,
                            custody_failure=str(error),
                        )
                    )
        if health_source:
            health = json.loads(Path(health_source).read_bytes())
            if health["argv"] != [str(q.ROOT / ".venv/bin/pytest"), "tests/python", "-q"]:
                raise ValueError("repository_health_argv")
            for prefix in ["stdout", "stderr"]:
                source = Path(health[prefix + "_path"])
                if sha256_file(source) != health[prefix + "_sha256"]:
                    raise ValueError("repository_health_stream")
                target = raw / "health" / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                health[prefix + "_path"] = str(target)
            health["log_path"] = health["stdout_path"]
            atomic_json(raw / "health/receipt.json", health)
        elif specs.get("repository_health") and all(r["passed"] for r in runtime):
            health = execute(specs["repository_health"], raw / "health")
        atomic_json(raw / "measurement.json", work)
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
    progress("scratch_removed", 1)
    cold = cold_controls(binding, raw / "cold") if binding else []
    value = build(work, raw, receipts, binding, cold, private, health)
    if (
        root == q.ROOT
        and output.absolute().is_relative_to(q.ROOT / "results")
        and work["execution_contract"]
    ):
        atomic_json(q.ROOT / q.EXECUTION, work["execution_contract"])
    terminal_reports: list[Json] = []

    def validate(candidate: Path) -> Json:
        report = terminal(candidate, raw)
        terminal_reports.append(report)
        return report

    try:
        publication = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        atomic_json(raw / "failed_terminal_candidate.json", value)
        atomic_json(raw / "failed_terminal_report.json", terminal_reports[-1])
        receipts.extend(terminal_reports[-1]["checks"])
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
        value = build(work, raw, receipts, binding, cold, private, health)
        publication = publish_primary(output, value, validate)
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            normal_process_exit=True,
            owned_checks_passed=value["required_checks_passed"],
        ),
    )
    progress("publication_complete", 1)
    return value


def main(argv: list[str] | None = None) -> int:
    """Expose the thin direct CLI and fresh reader without model loading or fixture shortcuts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261008"], default="20261008")
    parser.add_argument("--root", type=Path, default=q.ROOT)
    parser.add_argument("--output", type=Path, default=q.ROOT / "results" / (q.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--coverage-replay", type=Path)
    args = parser.parse_args(argv)
    progress("start")
    try:
        if args.coverage_replay:
            print(
                json.dumps(
                    dict(
                        passed=True,
                        owned_statement_counts=custody.replay(
                            json.loads(args.coverage_replay.read_bytes())
                        ),
                    )
                ),
                flush=True,
            )
        elif args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
        else:
            run(args.root, args.output)
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
