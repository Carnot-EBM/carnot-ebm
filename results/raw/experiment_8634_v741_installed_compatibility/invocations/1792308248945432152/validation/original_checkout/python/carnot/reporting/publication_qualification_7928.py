"""Qualify publication mechanics without changing numerical evidence.

REQ-REPORT-7928: private adversarial layouts drive the existing consumers.
Historical custody and owned validation remain separate from scientific benefit.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt

ROOT = Path(__file__).resolve().parents[3]
HELPER = "python/carnot/reporting/primary_publication.py"
MODULE = "python/carnot/reporting/publication_qualification_7928.py"
CLI = "scripts/experiments/experiment_7928_v688_primary_publication.py"
OWNED = (HELPER, MODULE, CLI)
INCLUDES = ",".join("*/" + name for name in OWNED)
TESTS = (
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_publication_qualification_7928.py",
)
CONSUMERS = (
    "tests/python/test_conductor_gates.py",
    "tests/python/test_in_process_doc_reconcile.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_source_boundary_7852.py",
    "tests/python/test_experiment_7868_v683_intervention_protocol.py",
)
DEPENDENCIES = (
    *OWNED,
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "scripts/conductor_gates.py",
    "scripts/in_process_doc_reconcile.py",
    "scripts/adversarial_verify.py",
    "scripts/verdict_row_consistency_lint.py",
    "scripts/check_spec_coverage.py",
    "python/carnot/verify/training_qualification_7916.py",
    "scripts/experiment_template.py",
    "pyproject.toml",
    *TESTS,
    *CONSUMERS,
)


def progress(phase: str, started: float, units: int = 0) -> None:
    """Flush phase boundaries so quiet children cannot hide stalled work."""
    print(
        f"[exp7928] phase={phase} completed={units} elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )


def dependency_hashes() -> dict[str, str]:
    """Bind each current callable and test to the bytes used for qualification."""
    return {name: sha256_file(ROOT / name) for name in DEPENDENCIES}


def fixture(output: Path) -> dict[str, Any]:
    """Exercise publication faults in isolated directories and keep primitive observations."""
    started = time.monotonic()
    base = {
        "experiment_id": 7928,
        "task_id": "exp7928-primary-publication",
        "honest_verdict": "complete_null_fixture",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "artifact_resolution_ready_score": 1,
    }
    rows: list[dict[str, Any]] = []
    for name in (
        "nested",
        "legacy",
        "missing",
        "conflict",
        "stale",
        "malformed",
        "disqualified",
        "absent",
        "concurrent",
        "terminal",
    ):
        directory = output.parent / (output.stem + "-cases") / name
        directory.mkdir(parents=True, exist_ok=True)
        primary = directory / "experiment_7928_fixture.json"
        expected = name in {"nested", "concurrent", "terminal"}
        details: dict[str, Any] = {}
        if name in {"nested", "stale", "terminal"}:
            details = publish_primary(
                primary, base, lambda p: {"passed": True, "checked_sha256": sha256_file(p)}
            )
            sidecar = Path(details["sidecar_path"])
            os.utime(sidecar, ns=(primary.stat().st_mtime_ns + 10**9,) * 2)
            if name == "stale":
                atomic_json(primary, {**base, "revision": 1})
                try:
                    read_bound_sidecar(primary, sidecar)
                except ValueError as exc:
                    details["failure_reason"] = str(exc)
            elif name == "terminal":
                details["revalidation"] = publish_primary(
                    primary, base, lambda p: {"passed": True, "checked_sha256": sha256_file(p)}
                )
        elif name == "legacy":
            atomic_json(primary, base)
            sidecar = primary.with_suffix(".json.validators.json")
            atomic_json(sidecar, {"candidate_sha256": sha256_file(primary)})
            os.utime(sidecar, ns=(primary.stat().st_mtime_ns + 10**9,) * 2)
        elif name == "conflict":
            atomic_json(primary, base)
            atomic_json(directory / "experiment_7928_other.json", base)
            try:
                publish_primary(primary, base, lambda p: {"passed": True})
            except ValueError as exc:
                details["failure_reason"] = str(exc)
        elif name == "malformed":
            primary.write_text("{")
        elif name in {"disqualified", "absent"}:
            atomic_json(
                primary,
                {**base, "verdict_class": "disqualified", "artifact_resolution_ready_score": 0}
                if name == "disqualified"
                else {"experiment_id": 7928},
            )
        elif name == "concurrent":

            def publish(index: int) -> dict[str, Any]:
                return publish_primary(
                    primary, {**base, "revision": index}, lambda p: {"passed": True}
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                details["replacements"] = list(pool.map(publish, range(2)))
        reading = reader_receipt(base["task_id"], directory)
        observed = reading["passed"]
        if name in {"stale", "conflict"}:
            observed = "failure_reason" not in details
        rows.append(
            {
                "unit": name,
                "arm": "publication_layout",
                "seed": 0,
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
                "reading": reading,
                "details": details,
            }
        )
        progress("fixture_" + name, started, len(rows))
    result = {
        "rows": rows,
        "rows_sha256": canonical_hash(rows),
        "passed": all(row["passed"] for row in rows),
    }
    atomic_json(output, result)
    return result


def replay(path: Path) -> dict[str, Any]:
    """Cold-reduce observations and re-read selected bytes rather than trust a score."""
    value = json.loads(path.read_text())
    rows = value["rows"]
    valid = bool(rows) and all(row["passed"] and row["observed"] == row["expected"] for row in rows)
    for row in rows:
        reading = row.get("reading")
        if reading and reading["gate_path"]:
            selected = Path(reading["gate_path"])
            actual = reader_receipt("exp7928-primary-publication", selected.parent)
            valid = (
                valid
                and actual["gate_sha256"] == reading["gate_sha256"]
                and actual["document_path"] == reading["document_path"]
            )
    return {
        "passed": valid and canonical_hash(rows) == value["rows_sha256"],
        "completed": len(rows),
    }


def history(
    results: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Freeze current historical selection without rewriting past dispatch claims."""
    sources: list[dict[str, Any]] = []
    readings: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for identity, slug in (
        (7904, "v686_training_qualification"),
        (7916, "v687_training_qualification"),
        (7918, "energy_fit"),
    ):
        primary = results / f"experiment_{identity}_{slug}.json"
        paths = (
            [primary, primary.with_suffix(".json.validators.json")]
            if identity != 7918
            else [primary]
        )
        for path in paths:
            sources.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "mtime_ns": path.stat().st_mtime_ns,
                    "role": "historical_primary" if path == primary else "historical_validator",
                    "exposure": "exposed_development",
                }
            )
        data = json.loads(primary.read_text())
        failures.extend(
            {"experiment_id": identity, **row}
            for row in data.get("validation_receipts", [])
            if not row["passed"]
        )
        if identity == 7916:
            failures.extend(data["historical_required_failures"])
        if identity != 7918:
            reading = reader_receipt(
                f"exp{identity}-training-qualification",
                results,
                field="training_runtime_ready_score",
            )
            reading["historical_primary_verdict"] = data["honest_verdict"]
            reading["primary_path"] = str(primary)
            reading["primary_sha256"] = sha256_file(primary)
            reading["all_gate_operands"] = [
                reader_receipt(
                    f"exp{identity}-training-qualification",
                    results,
                    field=field,
                    expected=expected,
                    op=op,
                )
                for field, op, expected in (
                    ("training_runtime_ready_score", "==", 1),
                    ("flagged_adversarial", "==", False),
                    ("verdict_class", "in", ["positive", "circular_positive", "null"]),
                )
            ]
            reading["historical_dispatch_claim"] = (
                "current selection only; no inference about past dispatch"
            )
            readings.append(reading)
    for failure in failures:
        log = Path(failure["log_path"])
        actual = sha256_file(log)
        if actual != failure["log_sha256"]:
            raise FileNotFoundError("historical_log_hash_mismatch", str(log))
    return sources, readings, failures


def freeze(private: Path, raw: Path) -> tuple[Path, list[CommandSpec]]:
    """Fix command scope and custody before any qualification result exists."""
    commands = build_scoped_commands(
        ROOT,
        (*TESTS, *CONSUMERS),
        OWNED[:2],
        static_paths=(CLI,),
        basetemp=private,
        coverage_file=private / ".coverage",
    )
    commands = [
        replace(
            row,
            timeout_s=600 if "pytest" in row.name or row.name == "changed_module_coverage" else 180,
        )
        for row in commands
    ]
    index = next(i for i, row in enumerate(commands) if row.name == "changed_module_coverage")
    commands[index] = CommandSpec(
        "changed_module_coverage",
        (
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            f"--data-file={private / '.coverage'}",
            f"--include={INCLUDES}",
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            f"--basetemp={private / 'coverage'}",
            *TESTS,
            "-q",
        ),
        "owned_units_and_cli",
        600,
    )
    index += 1
    commands.insert(
        index,
        CommandSpec(
            "combine_coverage",
            (
                sys.executable,
                "-m",
                "coverage",
                "combine",
                f"--data-file={private / '.coverage'}",
                str(private),
            ),
            "owned_statements",
            60,
        ),
    )
    index = next(
        i for i, row in enumerate(commands) if row.name == "changed_module_coverage_report"
    )
    commands[index] = CommandSpec(
        "changed_module_coverage_report",
        (
            sys.executable,
            "-m",
            "coverage",
            "json",
            f"--data-file={private / '.coverage'}",
            f"--include={INCLUDES}",
            "--fail-under=100",
            "-o",
            str(private / "coverage.json"),
        ),
        "owned_statements",
        60,
    )
    index = next(i for i, row in enumerate(commands) if row.name == "changed_module_mypy")
    commands[index] = replace(
        commands[index],
        argv=(str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED),
    )
    index = next(i for i, row in enumerate(commands) if row.name == "scoped_spec_coverage")
    commands[index] = replace(
        commands[index], argv=(sys.executable, "scripts/check_spec_coverage.py", *TESTS)
    )
    intervention = "scripts/experiments/experiment_7868_v683_intervention_protocol.py"
    commands.extend(
        CommandSpec(
            "e2e_016_" + mode,
            (
                sys.executable,
                intervention,
                "--date",
                "20260929",
                flag,
                str(private / "e2e016.json"),
            ),
            "historical_fixture",
            180,
        )
        for mode, flag in (("fixture", "--fixture-e2e"), ("replay", "--cold-replay"))
    )
    commands.append(
        CommandSpec(
            "full_repository_pytest",
            (
                str(ROOT / ".venv/bin/pytest"),
                "tests/python",
                "-q",
                f"--basetemp={private / 'full'}",
            ),
            "repository_health",
            600,
        )
    )
    manifest = raw / "validation_command_manifest.json"
    atomic_json(
        manifest,
        {
            "commands": [
                {
                    **asdict(row),
                    "expected_exit": 0,
                    "expected_reason": "required owned check"
                    if row.scope != "repository_health"
                    else "repository-wide debt recorded separately",
                    "deadline_s": row.timeout_s,
                }
                for row in commands
            ],
            "coverage_includes": INCLUDES,
            "affected_dependencies": dependency_hashes(),
            "execution_date": "20260930",
            "historical_fixture_date": "20260929",
            "fixture_expected_failures": [
                "legacy_shadow",
                "missing_primary",
                "conflicting_primary",
                "stale_primary_hash",
                "malformed_primary",
                "disqualified_primary",
                "absent_field",
            ],
        },
    )
    return manifest, commands


def run_checks(commands: list[CommandSpec], raw: Path, private: Path) -> list[dict[str, Any]]:
    """Reuse the supervisor and seal only logs from children that have exited."""
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=private / "logs",
        heartbeat_s=30,
        extra_env={
            "CARNOT7928_COVERAGE": "1",
            "COVERAGE_FILE": str(private / ".coverage"),
            "JAX_PLATFORMS": "cpu",
        },
    )
    for row, spec in zip(receipts, commands, strict=True):
        log = Path(row["log_path"])
        if not log.is_absolute():
            log = ROOT / log
        durable = raw / "logs" / (sha256_file(log).split(":")[1] + ".log")
        durable.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(log, durable)
        row.update(
            {
                "log_path": str(durable),
                "argv": list(spec.argv),
                "expected_exit": 0,
                "actual_exit": row["exit_code"],
                "deadline_s": spec.timeout_s,
                "expected_reason": "required owned check",
                "measured_files": list(OWNED),
            }
        )
    return receipts


def coverage_counts(private: Path) -> dict[str, Any]:
    """Keep empty or absent coverage visible as failure rather than infer execution."""
    path = private / "coverage.json"
    data = json.loads(path.read_text()).get("files", {}) if path.is_file() else {}
    return {name: value["summary"] for name, value in data.items()}


def terminal_checks(candidate: Path, raw: Path) -> dict[str, Any]:
    """Run both terminal validators on the same candidate and inspect their reports."""
    commands = [
        CommandSpec(
            "adversarial",
            (sys.executable, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(candidate)),
            "terminal",
            180,
        ),
        CommandSpec(
            "row_consistency",
            (
                sys.executable,
                str(ROOT / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(candidate),
            ),
            "terminal",
            60,
        ),
    ]
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=raw / "terminal_logs" / sha256_file(candidate).split(":")[1],
        heartbeat_s=30,
    )
    report = json.loads(Path(receipts[0]["log_path"]).read_text())
    return {
        "passed": all(row["passed"] for row in receipts),
        "flagged_adversarial": report["flagged_count"] > 0,
        "candidate_sha256": sha256_file(candidate),
        "receipts": receipts,
    }


def qualify(date: str) -> int:
    """Publish a mechanical receipt only after owned checks and exact-byte validation."""
    started_ns = time.monotonic_ns()
    started = time.monotonic()
    progress("start", started)
    private = Path(tempfile.mkdtemp(prefix="carnot7928-"))
    output = ROOT / "results/experiment_7928_v688_primary_publication.json"
    raw = output.parent / "raw" / output.stem
    raw.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "experiment_id": 7928,
        "task_id": "exp7928-primary-publication",
        "milestone": "2026.09.688",
        "run_date": date,
        "honest_verdict": "complete_circular_positive_primary_publication",
        "verdict_class": "circular_positive",
        "flagged_adversarial": False,
        "gate_check_summary": [],
        "rows": [],
        "artifact_resolution_ready_score": 0,
        "acceptance_gate_results": {
            "validity": False,
            "readiness": 0,
            "calibration": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "source_artifact_hashes": [],
        "resolution_rows": [],
        "layout_mutation_rows": [],
        "historical_required_failures": [],
        "validation_receipts": [],
        "observed_child_commands": [],
        "coverage_statement_counts": {},
        "validation_command_manifest_path": None,
        "publication_helper_path": HELPER,
        "publication_helper_callable": "carnot.reporting.primary_publication.publish_primary",
        "preconditions_checked": {"no_numerical_requalification": True},
        "repository_health": {"affects_owned_readiness": False, "retire_if_same_verdict": True},
        "verifier_is_oracle": True,
        "claim_scope": "exposed_development; private fixture agreement only; no independent scientific benefit",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": None,
        "model_invocation_counts": dict(ZERO_INVOCATION_COUNTS),
        "trained_head_specs": [],
        "random_seed": 0,
        "methodology": "Read exact historical primaries for custody. Reproduce current mtime-selected sidecars and qualify new primary-only layout through unmodified consumers. No model or benchmark executes.",
    }
    try:
        sources, historical, failures = history(ROOT / "results")
        result.update(
            source_artifact_hashes=sources,
            resolution_rows=historical,
            historical_required_failures=failures,
        )
        manifest, commands = freeze(private, raw)
        result["validation_command_manifest_path"] = str(manifest)
        progress("frozen_inputs_commands", started)
        fixture_path = private / "layout_fixture.json"
        matrix = fixture(fixture_path)
        result["rows"] = matrix["rows"]
        result["rows_sha256"] = matrix["rows_sha256"]
        result["layout_mutation_rows"] = matrix["rows"]
        cold = run_checks(
            [
                CommandSpec(
                    "cold_fixture",
                    (sys.executable, str(ROOT / CLI), "--cold-replay", str(fixture_path)),
                    "fixture",
                    60,
                ),
                CommandSpec(
                    "terminal_fixture",
                    (sys.executable, str(ROOT / CLI), "--terminal-recheck", str(fixture_path)),
                    "fixture",
                    60,
                ),
            ],
            raw,
            private,
        )
        receipts = cold + run_checks(commands, raw, private)
        result["validation_receipts"] = [
            row for row in receipts if row.get("scope") != "repository_health"
        ]
        result["repository_health"]["current_full_suite"] = [
            row for row in receipts if row.get("scope") == "repository_health"
        ]
        result["observed_child_commands"] = receipts
        counts = coverage_counts(private)
        result["coverage_statement_counts"] = counts
        covered = all(
            counts.get(name, {}).get("num_statements", 0) > 0 and counts[name]["missing_lines"] == 0
            for name in OWNED
        )
        passed = (
            matrix["passed"]
            and all(row["passed"] for row in result["validation_receipts"])
            and covered
        )
        result["artifact_resolution_ready_score"] = int(passed)
        result["acceptance_gate_results"].update(validity=passed, readiness=int(passed))
        if not passed:
            result.update(
                honest_verdict="complete_disqualified_primary_publication",
                verdict_class="disqualified",
            )
            result["gate_check_summary"] = [
                {
                    "upstream_id": "owned_validation",
                    "artifact_path": str(manifest),
                    "artifact_sha256": sha256_file(manifest),
                    "field": row["name"],
                    "op": "==",
                    "expected": 0,
                    "observed": row["exit_code"],
                }
                for row in result["validation_receipts"]
                if not row["passed"]
            ]
            if not covered:
                result["gate_check_summary"].append(
                    {
                        "upstream_id": "owned_coverage",
                        "artifact_path": str(private / "coverage.json"),
                        "artifact_sha256": None,
                        "field": "statement_coverage",
                        "op": "==",
                        "expected": 100,
                        "observed": counts,
                    }
                )
    except FileNotFoundError as exc:
        result.update(
            honest_verdict="complete_blocked_publication_prerequisite", verdict_class="blocked"
        )
        result["gate_check_summary"] = [
            {
                "upstream_id": "historical_artifact",
                "artifact_path": str(exc.filename or exc),
                "artifact_sha256": None,
                "field": "artifact",
                "op": "exists",
                "expected": True,
                "observed": None,
            }
        ]
    progress("qualification_reduced", started, len(result["rows"]))
    ended_ns = time.monotonic_ns()
    duration = (ended_ns - started_ns) / 1e9
    result["duration_s"] = duration
    result["phase_spans"] = [
        {"phase": "owned_qualification", "start_s": 0.0, "end_s": duration, "duration_s": duration}
    ]
    result["current_work_receipt"] = build_current_work_receipt(
        run_id=result["task_id"],
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=result["inference_substrate"],
        inference_substrate_details={"mechanical_publication_only": True},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        phase_spans=result["phase_spans"],
    )
    result["resolved_imports"] = {
        name: str(Path(importlib.import_module(name).__file__).resolve())
        for name in (
            "carnot.reporting.primary_publication",
            "carnot.reporting.publication_qualification_7928",
            "scripts.conductor_gates",
            "scripts.in_process_doc_reconcile",
        )
    }
    result["reproducibility_checksum"] = canonical_hash(
        {
            "code": dependency_hashes(),
            "sources": result["source_artifact_hashes"],
            "config": {"date": date, "coverage_includes": INCLUDES},
        }
    )
    size = len(result["rows"])
    result["sample_size_budget"] = {
        "intended": 10,
        "eligible": size,
        "started": size,
        "completed": size,
        "failed": sum(not row["passed"] for row in result["rows"]),
        "censored": 0,
        "excluded": 10 - size,
        "independent": 0,
    }
    result["primary_resolution_receipt"] = {
        "path": str(raw / "primary_resolution_receipt.json"),
        "binding": "external receipt records final reader-selected primary hash; avoids self-hash recursion",
        "fixture_readers": [
            row["reading"] for row in result["rows"] if row["unit"] in {"nested", "terminal"}
        ],
    }
    result["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    result["field_principles"] = {
        key: "Bind executing producer, exact bytes, measured scope and primitive evidence; fixture agreement cannot establish scientific benefit."
        for key in result
    }
    result["field_principles"].update(
        artifact_resolution_ready_score="Only real reader identity plus every owned check can open mechanical readiness.",
        primary_resolution_receipt="External byte-bound receipts avoid circular self-hash claims.",
        source_artifact_hashes="Historical qualification requires exact primary custody, not newest-match selection.",
    )
    candidate = raw / "terminal_candidate.json"
    atomic_json(candidate, result)
    report = terminal_checks(candidate, raw)
    if not report["passed"]:
        result.update(
            honest_verdict="complete_disqualified_primary_publication",
            verdict_class="disqualified",
            flagged_adversarial=report["flagged_adversarial"],
            artifact_resolution_ready_score=0,
        )
        result["acceptance_gate_results"].update(validity=False, readiness=0)
        result["gate_check_summary"].append(
            {
                "upstream_id": "owned_terminal_validator",
                "artifact_path": str(candidate),
                "artifact_sha256": sha256_file(candidate),
                "field": "terminal_checks",
                "op": "==",
                "expected": True,
                "observed": False,
            }
        )
        atomic_json(candidate, result)
        report = terminal_checks(candidate, raw)
    progress("terminal_bytes_checked", started, size)
    validated_hash = sha256_file(candidate)
    receipt = publish_primary(
        output,
        result,
        lambda p: {"passed": sha256_file(p) == validated_hash, "terminal_checks": report},
    )
    selected = reader_receipt(
        result["task_id"], output.parent, expected=result["artifact_resolution_ready_score"]
    )
    selected["passed"] = (
        selected["passed"]
        and selected["gate_path"] == str(output)
        and selected["gate_sha256"] == receipt["primary_sha256"]
    )
    atomic_json(
        raw / "primary_resolution_receipt.json",
        {"primary_sha256": receipt["primary_sha256"], "reader_receipt": selected},
    )
    atomic_json(
        raw / "terminal_validation.json",
        {
            "primary_sha256": receipt["primary_sha256"],
            "hashed_sidecar_path": receipt["sidecar_path"],
            "report": report,
            "reader_receipt": selected,
        },
    )
    progress("complete_publication", started, size)
    return int(not selected["passed"] or result["verdict_class"] in {"blocked", "disqualified"})
