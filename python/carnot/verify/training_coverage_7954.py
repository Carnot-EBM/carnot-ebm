"""Qualify fitting exception routes before natural execution (REQ-VERIFY-7954)."""

from __future__ import annotations

import ast
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from typing import Any

from coverage import CoverageData

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.publication_qualification_7928 import terminal_checks
from carnot.verify import energy_fit_7943 as fit
from carnot.verify import training_publication_7941 as publication
from carnot.verify import training_publication_7941_run as prior_run

ROOT = fit.ROOT
CLI = "scripts/experiments/experiment_7954_v690_training_coverage.py"
TASK = "exp7954-training-coverage"
OWNED = ("python/carnot/verify/training_coverage_7954.py", CLI)
COVERED = (*OWNED, *fit.OWNED)
INCLUDES = ",".join("*/" + name for name in COVERED)
TESTS = ("tests/python/test_training_coverage_7954.py", "tests/python/test_energy_fit_7943.py")
CONSUMERS = (*prior_run.TESTS, *prior_run.CONSUMERS)
SCORES = ("training_coverage_ready_score", "prefit_coverage_ready_score", "runtime_ready_score")
NEGATIVES = {
    "wrapper_malformed": "Expecting property name",
    "publication_collision": "conflicting_primary",
    "malformed_replay": "Expecting property name",
    "missing_replay": "No such file or directory",
    "missing_key_replay": "terminal_validation_sidecar_path",
    "expected_failure": "source evidence blocked",
}
START = time.monotonic()


def progress(phase: str, units: int = 0) -> None:
    """Real elapsed work and flushed boundaries keep supervised children observable."""
    print(
        f"[exp7954] phase={phase} completed={units} elapsed_s={time.monotonic() - START:.3f}",
        flush=True,
    )


def route(raw: Path, name: str) -> Path:
    """Each scenario has its own directory so the unchanged publisher retains uniqueness."""
    return raw / "routes" / name / "experiment_7943_fixture.json"


def freeze(raw: Path) -> tuple[Path, list[CommandSpec]]:
    """Bind commands and callable bytes before tests or a small fixture can run."""
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "attempts").mkdir(exist_ok=True)
    py = str(ROOT / ".venv/bin/python")
    cov = (py, "-m", "coverage")
    measure = (
        *cov,
        "run",
        "--parallel-mode",
        f"--data-file={raw / '.coverage'}",
        f"--include={INCLUDES}",
    )
    common = ("-n", "0", "-o", "addopts=", "--no-cov", "-q")
    commands = [
        CommandSpec(
            "affected_unit",
            (
                *measure,
                "-m",
                "pytest",
                *common,
                f"--basetemp={raw / 'attempts/unit'}",
                *TESTS,
                *CONSUMERS,
            ),
            "required",
            300,
        )
    ]
    inputs = raw / "inputs"
    inputs.mkdir(exist_ok=True)
    (inputs / "experiment_7943_malformed.json").write_text("{")
    atomic_json(inputs / "experiment_7943_missing_key.json", {})
    collision = route(raw, "publication_collision").parent
    for name in ("a", "b"):
        atomic_json(
            collision / f"experiment_7943_conflict_{name}.json", {"historical_sibling": name}
        )
    commands += [
        CommandSpec(
            "wrapper_malformed",
            (
                *measure,
                CLI,
                "--date",
                "20260930",
                "--cold-replay",
                str(inputs / "experiment_7943_malformed.json"),
            ),
            "exception",
            60,
        ),
        CommandSpec(
            "publication_collision",
            (
                *measure,
                fit.CLI,
                "--date",
                "20260930",
                "--fixture",
                "--publication",
                str(inputs / "absent_publication.json"),
                "--output",
                str(route(raw, "publication_collision")),
            ),
            "control",
            90,
        ),
    ]
    for name, filename in (
        ("malformed_replay", "malformed"),
        ("missing_replay", "missing"),
        ("missing_key_replay", "missing_key"),
    ):
        commands.append(
            CommandSpec(
                name,
                (
                    *measure,
                    fit.CLI,
                    "--date",
                    "20260930",
                    "--cold-replay",
                    str(inputs / f"experiment_7943_{filename}.json"),
                    "--output",
                    str(route(raw, name)),
                ),
                "exception",
                60,
            )
        )
    for name in ("expected_failure", "blocked", "success"):
        args = (
            *measure,
            fit.CLI,
            "--date",
            "20260930",
            "--fixture",
            "--output",
            str(route(raw, name)),
        )
        if name != "success":
            args += ("--publication", str(inputs / "absent_publication.json"))
        if name != "blocked":
            args += ("--assert-ready",)
        commands.append(CommandSpec(name, args, "route", 90))
        commands.append(
            CommandSpec(
                name + "_replay",
                (*measure, fit.CLI, "--date", "20260930", "--cold-replay", str(route(raw, name))),
                "route",
                90,
            )
        )
    commands.append(
        CommandSpec(
            "terminal_recheck",
            (
                *measure,
                fit.CLI,
                "--date",
                "20260930",
                "--terminal-recheck",
                str(route(raw, "success")),
            ),
            "route",
            90,
        )
    )
    commands += [
        CommandSpec(
            "e2e_015",
            (
                str(ROOT / ".venv/bin/pytest"),
                *common,
                f"--basetemp={raw / 'attempts/e2e015'}",
                "tests/python/test_source_boundary_7852.py",
            ),
            "required",
            180,
        )
    ]
    commands += [
        CommandSpec(
            "e2e_016_" + name,
            (
                py,
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                flag,
                str(raw / "e2e016/fixture.json"),
            ),
            "required",
            180,
        )
        for name, flag in (("fixture", "--fixture-e2e"), ("replay", "--cold-replay"))
    ]
    commands += [
        CommandSpec(
            "ruff_check",
            (str(ROOT / ".venv/bin/ruff"), "check", *OWNED, *TESTS[:1]),
            "required",
            60,
        ),
        CommandSpec(
            "ruff_format",
            (str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, *TESTS[:1]),
            "required",
            60,
        ),
        CommandSpec(
            "mypy",
            (str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED),
            "required",
            120,
        ),
        CommandSpec(
            "spec_coverage",
            (py, "scripts/check_spec_coverage.py", *TESTS, *CONSUMERS),
            "required",
            60,
        ),
        CommandSpec(
            "coverage_combine",
            (*cov, "combine", f"--data-file={raw / '.coverage'}", str(raw)),
            "required",
            60,
        ),
        CommandSpec(
            "coverage_json",
            (
                *cov,
                "json",
                f"--data-file={raw / '.coverage'}",
                f"--include={INCLUDES}",
                "--fail-under=100",
                "-o",
                str(raw / "coverage.json"),
            ),
            "required",
            60,
        ),
    ]
    dependencies = {
        **publication.library().dependency_hashes(),
        **{
            name: sha256_file(ROOT / name)
            for name in (
                *COVERED,
                *TESTS,
                *CONSUMERS,
                "scripts/experiment_template.py",
                "scripts/check_spec_coverage.py",
                "pyproject.toml",
            )
        },
    }
    path = raw / "validation_command_manifest.json"
    atomic_json(
        path,
        {
            "commands": [
                {
                    **asdict(c),
                    "expected_exit": 2 if c.name in NEGATIVES else 0,
                    "expected_reason": NEGATIVES.get(c.name, "command must pass"),
                    "deadline_s": c.timeout_s,
                }
                for c in commands
            ],
            "coverage_includes": INCLUDES,
            "coverage_file": str(raw / ".coverage"),
            "dependencies": dependencies,
            "affected_files": OWNED,
            "transitive_consumers": CONSUMERS,
            "execution_date": "20260930",
            "historical_fixture_date": "20260929",
            "natural_fitting_authorized": False,
            "checkpoint_policy": "exact code/config/input hashes",
        },
    )
    return path, commands


def execute(commands: list[CommandSpec], raw: Path) -> list[dict[str, Any]]:
    """Expected failures pass only with both their actual exit and flushed reason."""
    rows = run_commands(
        ROOT,
        commands,
        log_dir=raw / "logs",
        heartbeat_s=30,
        extra_env={"CARNOT7954_COVERAGE_FILE": str(raw / ".coverage"), "JAX_PLATFORMS": "cpu"},
    )
    for spec, row in zip(commands, rows, strict=True):
        expected = 2 if spec.name in NEGATIVES else 0
        row.update(
            argv=list(spec.argv),
            actual_exit=row["exit_code"],
            expected_exit=expected,
            expected_reason=NEGATIVES.get(spec.name, "command must pass"),
            deadline_s=spec.timeout_s,
            measured_files=list(COVERED),
        )
        row["passed"] = (
            not row["timed_out"]
            and row["exit_code"] == expected
            and (spec.name not in NEGATIVES or NEGATIVES[spec.name] in row["output_tail"])
        )
    return rows


def coverage_ready(counts: dict[str, Any]) -> bool:
    """An empty measurement or one missed statement cannot qualify executable code."""
    return all(
        any(
            name.endswith(owned) and item["num_statements"] > 0 and item["missing_lines"] == 0
            for name, item in counts.items()
        )
        for owned in COVERED
    )


def ready(
    receipts: list[dict[str, Any]], counts: dict[str, Any], selected: list[dict[str, Any]]
) -> bool:
    """Mechanics require all owned checks plus distinct successful and blocked readers."""
    return (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and coverage_ready(counts)
        and len(selected) == 3
        and all(r["passed"] for r in selected)
    )


def selections(raw: Path) -> list[dict[str, Any]]:
    """Both unchanged readers must choose the private primary rather than newer sidecars."""
    return [
        {
            **reader_receipt(
                fit.TASK,
                route(raw, name).parent,
                field="fixture_ready_score",
                expected=int(name == "success"),
            ),
            "route": name,
        }
        for name in ("success", "blocked", "expected_failure")
        if route(raw, name).is_file()
    ]


def negative_mutation(raw: Path) -> dict[str, Any]:
    """Delete the exception test in a private copy; other files retain their measured custody.

    Clear the target CLI's coverage so prior exception routes cannot conceal the
    deletion. Dispatch tests and real cold replay rebuild its normal paths.
    """
    progress("coverage_mutation_before")
    directory = raw / "mutation"
    directory.mkdir(exist_ok=True)
    source = (ROOT / TESTS[0]).read_text()
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "test_real_exception_routes"
    )
    lines = source.splitlines(keepends=True)
    copied = directory / "test_deleted_exception.py"
    copied.write_text("".join(lines[: node.lineno - 1] + lines[node.end_lineno :]))
    original = CoverageData(basename=str(raw / ".coverage"))
    original.read()
    seeded = CoverageData(basename=str(directory / ".coverage"))
    seeded.add_lines(
        {
            name: original.lines(name) or []
            for name in original.measured_files()
            if not name.endswith(fit.CLI)
        }
    )
    seeded.write()
    py = str(ROOT / ".venv/bin/python")
    cov = (py, "-m", "coverage")
    measure = (
        *cov,
        "run",
        "--parallel-mode",
        f"--data-file={directory / '.coverage'}",
        f"--include={INCLUDES}",
    )
    commands = [
        CommandSpec(
            "deleted_exception_test",
            (
                *measure,
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                str(copied),
                "-k",
                "main_dispatch",
                f"--basetemp={directory / 'unit'}",
            ),
            "control",
            90,
        ),
        CommandSpec(
            "mutation_cold_replay",
            (*measure, fit.CLI, "--date", "20260930", "--cold-replay", str(route(raw, "success"))),
            "control",
            90,
        ),
        CommandSpec(
            "mutation_combine",
            (
                *cov,
                "combine",
                "--append",
                "--keep",
                f"--data-file={directory / '.coverage'}",
                str(directory),
            ),
            "control",
            60,
        ),
        CommandSpec(
            "mutation_report",
            (
                *cov,
                "json",
                f"--data-file={directory / '.coverage'}",
                f"--include={INCLUDES}",
                "--fail-under=100",
                "-o",
                str(directory / "coverage.json"),
            ),
            "control",
            60,
        ),
    ]
    atomic_json(
        directory / "command_manifest.json",
        {
            "commands": [asdict(c) for c in commands],
            "coverage_includes": INCLUDES,
            "deleted_test": "test_real_exception_routes",
            "seed_policy": "retain measured non-target files; clear entire target CLI",
        },
    )
    rows = run_commands(
        ROOT,
        commands,
        log_dir=directory / "logs",
        heartbeat_s=30,
        extra_env={
            "CARNOT7954_COVERAGE_FILE": str(directory / ".coverage"),
            "JAX_PLATFORMS": "cpu",
        },
    )
    report = json.loads((directory / "coverage.json").read_text())
    missing = next(
        v["missing_lines"] for name, v in report["files"].items() if name.endswith(fit.CLI)
    )
    passed = (
        all(r["exit_code"] == 0 and not r["timed_out"] for r in rows[:-1])
        and rows[-1]["exit_code"] == 2
        and missing == [51, 52, 53]
    )
    result = {
        "passed": passed,
        "deleted_test": "test_real_exception_routes",
        "mutated_test_path": str(copied),
        "missing_cli_lines": missing,
        "receipts": rows,
        "coverage_report_path": str(directory / "coverage.json"),
    }
    atomic_json(directory / "receipt.json", result)
    progress("coverage_mutation_after", len(rows))
    return result


def base(
    sources: list[dict[str, Any]], failures: list[dict[str, Any]], upstream: dict[str, Any]
) -> dict[str, Any]:
    """Current qualification owns its identity while inherited facts keep their producers."""
    value = publication.base(sources, failures)
    value.update(
        experiment_id=7954,
        task_id=TASK,
        milestone="2026.09.690",
        run_date="20260930",
        honest_verdict="complete_blocked_prerequisite"
        if failures
        else "complete_circular_positive_training_coverage",
        verdict_class="blocked" if failures else "circular_positive",
        training_coverage_ready_score=0,
        prefit_coverage_ready_score=0,
        runtime_ready_score=0,
        exception_route_rows=[],
        qualified_runner_manifest={},
        training_entrypoint={
            "path": fit.CLI,
            "callable": "carnot.verify.energy_fit_7943.driver",
            "numerical_callable": "carnot.verify.energy_fit_7930.fit_heads",
        },
        cited_upstream_artifacts=[],
        coverage_statement_counts={},
        methodology="Current unit and actual CLI exception coverage, private fixture fitting and cold replay. No natural fit. Fixture agreement is circular; September 28 oracle-distinct corrigendum remains in force.",
    )
    value["historical_required_failures"] = list(upstream.get("historical_required_failures", []))
    value["acceptance_gate_results"].update(calibration=None)
    value["resolved_imports"].update(
        {
            "carnot.verify.training_coverage_7954": str(ROOT / OWNED[0]),
            "carnot.verify.energy_fit_7943": str(Path(fit.__file__).resolve()),
        }
    )
    value["field_principles"].update(
        {
            k: "Current nonempty full coverage, fixture fit, cold replay and all owned checks must pass; historical readiness never substitutes."
            for k in SCORES
        }
    )
    return value


def seal(value: dict[str, Any], output: Path) -> None:
    """Only inspected final bytes become visible; the terminal sidecar binds their hash."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["primary_resolution_receipt"] = {
        "path": str(raw / "primary_resolution.json"),
        "binding": "external final hash",
    }
    value["duration_s"] = time.monotonic() - START
    for key in value:
        value["field_principles"].setdefault(
            key,
            "Bind producer, primitive evidence, exact bytes and exposed roles; running successfully is not independent benefit.",
        )
    candidate = raw / "candidate.json"
    atomic_json(candidate, value)
    report = terminal_checks(candidate, raw)
    if not report["passed"] or report["flagged_adversarial"]:
        value.update(
            honest_verdict="complete_disqualified_terminal_validation",
            verdict_class="disqualified",
            flagged_adversarial=report["flagged_adversarial"],
            **dict.fromkeys(SCORES, 0),
        )
        value["acceptance_gate_results"].update(validity=False, readiness=0)
        atomic_json(candidate, value)
        value["flagged_adversarial"] = terminal_checks(candidate, raw)["flagged_adversarial"]
    receipt = publish_primary(output, value, lambda p: terminal_checks(p, raw))
    atomic_json(
        raw / "terminal_validation.json",
        {
            "candidate_sha256": receipt["primary_sha256"],
            "validator_sidecar_path": receipt["sidecar_path"],
        },
    )
    os.utime(receipt["sidecar_path"], ns=(output.stat().st_mtime_ns + 10**9,) * 2)
    selected = reader_receipt(TASK, output.parent, field=SCORES[0], expected=value[SCORES[0]])
    if not selected["passed"] or selected["gate_sha256"] != receipt["primary_sha256"]:
        raise ValueError("primary consumer mismatch")
    atomic_json(raw / "primary_resolution.json", selected)
    progress("published", len(value["rows"]))


def qualify(
    output: Path, publication_path: Path = fit.PUBLICATION, runtime: Path = fit.RUNTIME
) -> int:
    """Qualify only bounded fixture mechanics; natural 27-head fitting remains downstream."""
    progress("start")
    began = time.monotonic()
    raw = output.parent / "raw" / output.stem
    manifest, commands = freeze(raw)
    progress("authenticate")
    try:
        sources, dependencies, upstream = fit.authenticate(publication_path, runtime, fit.UPSTREAM)
    except fit.old.custody.InputBlocked as exc:
        value = base([], exc.operands, {})
        progress("external_prerequisite_blocked")
    else:
        value = base(sources, [], upstream)
        receipts = execute(commands, raw)
        coverage = (
            json.loads((raw / "coverage.json").read_text())["files"]
            if (raw / "coverage.json").is_file()
            else {}
        )
        counts = {name: item["summary"] for name, item in coverage.items()}
        mutation = (
            negative_mutation(raw)
            if (raw / "coverage.json").is_file()
            else {"passed": False, "reason": "owned coverage report absent"}
        )
        selected = selections(raw)
        fixture = (
            json.loads(route(raw, "success").read_text()) if route(raw, "success").is_file() else {}
        )
        passed = (
            ready(receipts, counts, selected)
            and mutation["passed"]
            and bool(fixture.get("trained_head_specs"))
            and fixture.get("fixture_ready_score") == 1
            and fixture.get("energy_fit_ready_score") == 0
            and len(fixture.get("rows", [])) == 3
        )
        frozen = json.loads(manifest.read_text())
        passed = (
            passed
            and [r["name"] for r in receipts] == [c.name for c in commands]
            and all(
                sha256_file(ROOT / name) == digest
                for name, digest in frozen["dependencies"].items()
            )
        )
        routes = [r for r in receipts if r["scope"] in ("route", "exception")]
        value.update(
            honest_verdict="complete_circular_positive_training_coverage"
            if passed
            else "complete_disqualified_required_checks",
            verdict_class="circular_positive" if passed else "disqualified",
            **dict.fromkeys(SCORES, int(passed)),
            validation_receipts=receipts,
            observed_child_commands=[r["argv"] for r in receipts],
            isolated_route_rows=routes,
            exception_route_rows=[r for r in routes if r["scope"] == "exception"],
            coverage_statement_counts=counts,
            consumer_selection_rows=selected,
            training_dependency_hashes={**dependencies, **frozen["dependencies"]},
            fixture_checkpoint_manifest={
                "path": fixture.get("checkpoint_manifest_path"),
                "sha256": fixture.get("checkpoint_manifest_sha256"),
                "executing_producer": 7943,
            },
            trained_head_specs=fixture.get("trained_head_specs", []),
            negative_mutation_receipt=mutation,
            rows=[
                {
                    "unit": r["name"],
                    "arm": "cli_qualification",
                    "seed": 67801,
                    "passed": r["passed"],
                    "expected_exit": r["expected_exit"],
                    "actual_exit": r["actual_exit"],
                }
                for r in routes
            ],
        )
        value["sample_size_budget"].update(
            {k: len(routes) for k in ("intended", "eligible", "started", "completed")},
            failed=sum(not r["passed"] for r in routes),
            censored=0,
            excluded=0,
            independent=0,
            unit="CLI scenario; no independent scientific samples",
        )
        for identity, path, fields in (
            (
                7941,
                publication_path,
                [
                    "runtime_ready_score",
                    "training_publication_ready_score",
                    "training_runtime_ready_score",
                ],
            ),
            (7916, runtime, ["training_runtime_ready_score"]),
            (
                7943,
                ROOT / "results/experiment_7943_v689_energy_fit.json",
                ["honest_verdict", "coverage_statement_counts"],
            ),
            (7930, ROOT / "results/experiment_7930_v688_energy_fit.json", ["honest_verdict"]),
        ):
            old = json.loads(path.read_text())
            value["cited_upstream_artifacts"].append(
                {
                    "experiment_id": identity,
                    "fields_imported": fields,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "historical_fields": {k: old.get(k) for k in fields},
                }
            )
            value["source_artifact_hashes"].append(
                {"path": str(path), "sha256": sha256_file(path), "role": "inherited_not_promoted"}
            )
            value["historical_required_failures"] += [
                {"experiment_id": identity, **r}
                for r in old.get("validation_receipts", [])
                if not r["passed"] and r["scope"] != "repository_health"
            ]
            value["repository_health"][str(identity)] = {
                "scope": "historical_only",
                "receipts": [
                    r
                    for r in old.get("validation_receipts", [])
                    if r["scope"] == "repository_health"
                ],
            }
        value["source_artifact_hashes"] += fixture.get("source_artifact_hashes", [])
        value["preconditions_checked"] = {
            "publication_sha256": sha256_file(publication_path),
            "runtime_sha256": sha256_file(runtime),
            "current_runtime_ready_score": upstream["runtime_ready_score"],
            "historical_training_runtime_ready_score": upstream["training_runtime_ready_score"],
            "natural_fitting_performed": False,
        }
        value["acceptance_gate_results"].update(validity=passed, readiness=int(passed))
        runner = raw / "qualified_runner_manifest.json"
        qualified = {
            "callable": value["training_entrypoint"],
            "qualified": passed,
            "dependencies": value["training_dependency_hashes"],
            "coverage_includes": INCLUDES,
            "coverage_statement_counts": counts,
            "validation_manifest_sha256": sha256_file(manifest),
            "exception_handler_lines": [51, 52, 53],
            "current_producer": 7954,
            "natural_fitting_performed": False,
        }
        atomic_json(runner, qualified)
        value["qualified_runner_manifest"] = {"path": str(runner), "sha256": sha256_file(runner)}
    value["validation_command_manifest_path"] = str(manifest)
    for prior in (raw / "attempts").glob("*/experiment_7954_*.json"):
        previous = json.loads(prior.read_text())
        value["source_artifact_hashes"].append(
            {
                "path": str(prior),
                "sha256": sha256_file(prior),
                "role": "historical_owned_attempt_retired_after_code_configuration_change",
            }
        )
        value["historical_required_failures"] += [
            {
                "experiment_id": 7954,
                **r,
                "scope": "historical_owned_attempt",
                "retired_by": "corrected temporary directory setup and mutation coverage configuration",
            }
            for r in previous["validation_receipts"]
            if not r["passed"]
        ]
    value["source_artifact_hashes"].append(
        {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
            "role": "frozen_current_configuration",
        }
    )
    elapsed = time.monotonic() - began
    value["phase_spans"] = [
        {"phase": "coverage_qualification", "start_s": 0, "end_s": elapsed, "duration_s": elapsed}
    ]
    value["reproducibility_checksum"] = canonical_hash(
        {"manifest": sha256_file(manifest), "sources": value["source_artifact_hashes"]}
    )
    seal(value, output)
    replay(output)
    return 0


def replay(path: Path) -> None:
    """Cold reduction checks primitive hashes and current readiness rather than inherited claims."""
    value = json.loads(path.read_text())
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    read_bound_sidecar(path, Path(terminal["validator_sidecar_path"]))
    for row in [*value["validation_receipts"], *value["source_artifact_hashes"]]:
        source = Path(row.get("log_path", row.get("path", "")))
        if not source.is_absolute():
            source = ROOT / source
        if sha256_file(source) != row.get("log_sha256", row.get("sha256")):
            raise ValueError("primitive hash drift")
    if value[SCORES[0]]:
        counts = {
            name: item["summary"]
            for name, item in json.loads(
                (path.parent / "raw" / path.stem / "coverage.json").read_text()
            )["files"].items()
        }
        if (
            not ready(value["validation_receipts"], counts, value["consumer_selection_rows"])
            or counts != value["coverage_statement_counts"]
            or value["sample_size_budget"]["completed"] != len(value["isolated_route_rows"])
            or value["sample_size_budget"]["independent"] != 0
        ):
            raise ValueError("readiness drift")
        for selected in value["consumer_selection_rows"]:
            if sha256_file(Path(selected["gate_path"])) != selected["gate_sha256"]:
                raise ValueError("route primary hash drift")
            fit.replay(Path(selected["gate_path"]), True)
    progress("cold_replay_passed", len(value["rows"]))
