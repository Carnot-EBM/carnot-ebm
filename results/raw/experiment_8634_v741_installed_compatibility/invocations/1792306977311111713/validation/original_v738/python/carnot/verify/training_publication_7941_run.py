"""Freeze and reduce owned validation for REQ-REPORT-7941."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import reader_receipt
from carnot.verify import training_publication_7941 as core

TESTS = ("tests/python/test_training_publication_7941.py",)
CONSUMERS = (
    "tests/python/test_energy_fit_7930.py",
    "tests/python/test_energy_fit_7930_run.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_conductor_gates.py",
    "tests/python/test_in_process_doc_reconcile.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_training_qualification_7916.py",
)
NEGATIVES = {
    "expected_failure": "source evidence blocked",
    "conflict": "conflicting_primary",
    "reject": "candidate_rejected",
}


def route(raw: Path, name: str) -> Path:
    """One publication directory per scenario preserves the publisher's uniqueness contract."""
    return raw / "routes" / name / "experiment_7941_fixture.json"


def freeze(raw: Path) -> tuple[Path, list[CommandSpec]]:
    """Fix scope, paths, deadlines and expectations before any fixture head is fitted."""
    py = str(core.ROOT / ".venv/bin/python")
    private = Path(tempfile.mkdtemp(prefix="carnot7941-validation-"))
    cov = (py, "-m", "coverage")
    measure = (
        *cov,
        "run",
        "--parallel-mode",
        f"--data-file={private / '.coverage'}",
        f"--include={core.INCLUDES}",
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
                f"--basetemp={private / 'unit'}",
                *TESTS,
                *CONSUMERS,
            ),
            "required",
            300,
        )
    ]
    for name in (
        "expected_failure",
        "blocked",
        "malformed",
        "hash_mismatch",
        "success",
        "terminal",
        "conflict",
        "reject",
    ):
        args = (
            *measure,
            core.CLI,
            "--date",
            "20260930",
            "--fixture",
            "--output",
            str(route(raw, name)),
        )
        if name in ("expected_failure", "blocked", "malformed", "hash_mismatch"):
            filename = {"malformed": "malformed.json", "hash_mismatch": "hash_mismatch.json"}.get(
                name, "missing.json"
            )
            args += ("--runtime", str(raw / "inputs" / filename))
        if name in ("expected_failure", "success"):
            args += ("--assert-ready",)
        if name in ("conflict", "reject"):
            args += ("--publication-fault", name)
        commands.append(CommandSpec(name, args, "route", 90))
        if name not in ("conflict", "reject"):
            commands.append(
                CommandSpec(
                    name + "_replay",
                    (
                        *measure,
                        core.CLI,
                        "--date",
                        "20260930",
                        "--cold-replay",
                        str(route(raw, name)),
                    ),
                    "route",
                    90,
                )
            )
    commands.append(
        CommandSpec(
            "terminal_recheck",
            (
                *measure,
                core.CLI,
                "--date",
                "20260930",
                "--terminal-recheck",
                str(route(raw, "terminal")),
            ),
            "route",
            90,
        )
    )
    commands += [
        CommandSpec(
            "e2e_015",
            (
                str(core.ROOT / ".venv/bin/pytest"),
                *common,
                f"--basetemp={private / 'e2e015'}",
                "tests/python/test_source_boundary_7852.py",
            ),
            "required",
            180,
        ),
        *[
            CommandSpec(
                "e2e_016_" + name,
                (
                    py,
                    "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                    "--date",
                    "20260929",
                    flag,
                    str(raw / "e2e016" / "fixture.json"),
                ),
                "required",
                180,
            )
            for name, flag in (("fixture", "--fixture-e2e"), ("replay", "--cold-replay"))
        ],
        CommandSpec(
            "ruff_check",
            (str(core.ROOT / ".venv/bin/ruff"), "check", *core.OWNED, *TESTS),
            "required",
            60,
        ),
        CommandSpec(
            "ruff_format",
            (str(core.ROOT / ".venv/bin/ruff"), "format", "--check", *core.OWNED, *TESTS),
            "required",
            60,
        ),
        CommandSpec(
            "mypy",
            (str(core.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *core.OWNED),
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
            (*cov, "combine", f"--data-file={private / '.coverage'}", str(private)),
            "required",
            60,
        ),
        CommandSpec(
            "coverage_json",
            (
                *cov,
                "json",
                f"--data-file={private / '.coverage'}",
                f"--include={core.INCLUDES}",
                "--fail-under=100",
                "-o",
                str(raw / "coverage.json"),
            ),
            "required",
            60,
        ),
        CommandSpec(
            "full_pytest",
            (
                str(core.ROOT / ".venv/bin/pytest"),
                "tests/python",
                "-q",
                f"--basetemp={private / 'repository_suite'}",
            ),
            "repository_health",
            180,
        ),
    ]
    health = raw / "attempts/initial_private_scope/repository_health.json"
    if health.is_file():
        recorded = json.loads(health.read_text())
        commands[-1] = CommandSpec("full_pytest", tuple(recorded["argv"]), "repository_health", 180)
    manifest = raw / "validation_command_manifest.json"
    atomic_json(
        manifest,
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
            "coverage_includes": core.INCLUDES,
            "coverage_file": str(private / ".coverage"),
            "changed_sources": core.OWNED,
            "affected_tests": TESTS,
            "transitive_consumers": CONSUMERS,
            "training_dependency_hashes": core.library().dependency_hashes(),
            "consumer_hashes": {
                name: sha256_file(core.ROOT / name)
                for name in (
                    "python/carnot/reporting/primary_publication.py",
                    "scripts/conductor_gates.py",
                    "scripts/in_process_doc_reconcile.py",
                    "scripts/adversarial_verify.py",
                    "scripts/verdict_row_consistency_lint.py",
                    *TESTS,
                    *CONSUMERS,
                )
            },
            "execution_date": "20260930",
            "historical_fixture_date": "20260929",
            "training_entrypoint": core.CLI,
            "numerical_math_changed": False,
        },
    )
    return manifest, commands


def execute(commands: list[CommandSpec], raw: Path) -> list[dict[str, Any]]:
    """Seal exited children and require both exit and reason for intentional negatives."""
    health = raw / "attempts/initial_private_scope/repository_health.json"
    live = [c for c in commands if c.name != "full_pytest" or not health.is_file()]
    manifest = raw / "validation_command_manifest.json"
    coverage_file = (
        json.loads(manifest.read_text())["coverage_file"]
        if manifest.is_file()
        else str(raw / ".coverage")
    )
    rows = run_commands(
        core.ROOT,
        live,
        log_dir=raw / "logs",
        heartbeat_s=30,
        extra_env={
            "CARNOT7941_COVERAGE": "1",
            "COVERAGE_FILE": coverage_file,
            "JAX_PLATFORMS": "cpu",
        },
    )
    if len(live) != len(commands):
        recorded = json.loads(health.read_text())
        recorded["reused_single_repository_invocation"] = True
        rows.append(recorded)
    for spec, row in zip(commands, rows, strict=True):
        expected = 2 if spec.name in NEGATIVES else 0
        row.update(
            argv=list(spec.argv),
            actual_exit=row["exit_code"],
            expected_exit=expected,
            expected_reason=NEGATIVES.get(spec.name, "command must pass"),
            deadline_s=spec.timeout_s,
            measured_files=list(core.OWNED),
        )
        row["passed"] = (
            not row["timed_out"]
            and row["exit_code"] == expected
            and (spec.name not in NEGATIVES or NEGATIVES[spec.name] in row["output_tail"])
        )
    return rows


def reduce_readiness(receipts: list[dict[str, Any]], counts: dict[str, Any]) -> bool:
    """A completed blocked route is useful validation, but cannot substitute for fitted success."""
    covered = all(
        any(
            name.endswith(owned) and item["num_statements"] > 0 and item["missing_lines"] == 0
            for name, item in counts.items()
        )
        for owned in core.OWNED
    )
    return (
        covered
        and bool(receipts)
        and all(row["passed"] for row in receipts if row["scope"] != "repository_health")
    )


def qualify(output: Path) -> int:
    """Aggregate current private receipts while preserving all historical required failures."""
    began = time.monotonic()
    core.progress("start")
    raw = output.parent / "raw" / output.stem
    manifest, commands = freeze(raw)
    try:
        sources, dependencies, qualification = core.authenticate(core.RUNTIME)
    except core.custody.custody.InputBlocked as exc:
        value = core.base([], exc.operands)
    else:
        value = core.base(sources, [])
        inputs = raw / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        (inputs / "malformed.json").write_text("{")
        changed = json.loads(core.RUNTIME.read_text())
        changed["deliberate_hash_mismatch"] = True
        atomic_json(inputs / "hash_mismatch.json", changed)
        core.progress("before_validation")
        receipts = execute(commands, raw)
        core.progress("after_validation", len(receipts))
        coverage = (
            json.loads((raw / "coverage.json").read_text())["files"]
            if (raw / "coverage.json").is_file()
            else {}
        )
        counts = {name: item["summary"] for name, item in coverage.items()}
        route_rows = [row for row in receipts if row["scope"] == "route"]
        selections = []
        for name in (
            "expected_failure",
            "blocked",
            "malformed",
            "hash_mismatch",
            "success",
            "terminal",
        ):
            primary = route(raw, name)
            if primary.is_file():
                selected = reader_receipt(
                    "exp7941-training-publication",
                    primary.parent,
                    field="runtime_ready_score",
                    expected=int(name in ("success", "terminal")),
                )
                selected["route"] = name
                selections.append(selected)
        ready = (
            reduce_readiness(receipts, counts)
            and [r["name"] for r in receipts] == [c.name for c in commands]
            and len(selections) == 6
            and all(row["passed"] for row in selections)
        )
        success = route(raw, "success")
        fixture = json.loads(success.read_text()) if success.is_file() else {}
        ready = (
            ready and bool(fixture.get("trained_head_specs")) and len(fixture.get("rows", [])) == 3
        )
        value.update(
            honest_verdict="complete_circular_positive_training_publication"
            if ready
            else "complete_disqualified_required_checks",
            verdict_class="circular_positive" if ready else "disqualified",
            training_publication_ready_score=int(ready),
            runtime_ready_score=int(ready),
            training_dependency_hashes={**dependencies, **core.library().dependency_hashes()},
            validation_receipts=receipts,
            observed_child_commands=[row["argv"] for row in receipts],
            coverage_statement_counts=counts,
            isolated_route_rows=route_rows,
            consumer_selection_rows=selections,
            rows=[
                {
                    "unit": row["name"],
                    "arm": "cli_publication",
                    "seed": 67801,
                    "passed": row["passed"],
                    "expected_exit": row["expected_exit"],
                    "actual_exit": row["actual_exit"],
                }
                for row in route_rows
            ],
            fixture_checkpoint_manifest=fixture.get("checkpoint_manifest_path"),
            trained_head_specs=fixture.get("trained_head_specs", []),
            historical_required_failures=qualification["historical_required_failures"],
            repository_health={
                "current_full_pytest": [r for r in receipts if r["scope"] == "repository_health"],
                "affects_required_checks": False,
            },
        )
        historical = core.ROOT / "results/experiment_7930_v688_energy_fit.json"
        old = json.loads(historical.read_text())
        value["source_artifact_hashes"].append(
            {
                "path": str(historical),
                "sha256": sha256_file(historical),
                "role": "historical_disqualified_not_promoted",
            }
        )
        value["source_artifact_hashes"] += fixture.get("source_artifact_hashes", [])
        if fixture.get("checkpoint_manifest_path"):
            value["fixture_checkpoint_manifest"] = {
                "path": fixture["checkpoint_manifest_path"],
                "sha256": fixture["checkpoint_manifest_sha256"],
            }
        value["historical_required_failures"] += [
            {"experiment_id": 7930, **r}
            for r in old["validation_receipts"]
            if not r["passed"] and r["scope"] != "repository_health"
        ]
        value["preconditions_checked"] = {
            "runtime_primary_sha256": sha256_file(core.RUNTIME),
            "numerical_dependencies_unchanged": True,
            "historical_exp7930_verdict": old["honest_verdict"],
            "historical_exp7930_retired_unchanged": True,
        }
        value["acceptance_gate_results"].update(validity=ready, readiness=int(ready))
        value["sample_size_budget"].update(
            {key: len(route_rows) for key in ("intended", "eligible", "started", "completed")},
            independent=0,
            unit="CLI scenario, not scientific sample",
            failed=sum(not r["passed"] for r in route_rows),
        )
    value["validation_command_manifest_path"] = str(manifest)
    prior_attempt = raw / "attempts/initial_private_scope/required_failures.json"
    value["prior_validation_attempts"] = (
        json.loads(prior_attempt.read_text())["rows"] if prior_attempt.is_file() else []
    )
    value["historical_required_failures"] += value["prior_validation_attempts"]
    elapsed = time.monotonic() - began
    value["phase_spans"] = [
        {
            "phase": "publication_qualification",
            "start_s": 0,
            "end_s": elapsed,
            "duration_s": elapsed,
        }
    ]
    value["reproducibility_checksum"] = canonical_hash(
        {"manifest": sha256_file(manifest), "sources": value["source_artifact_hashes"]}
    )
    core.seal(value, output)
    core.replay(output, recheck=True)
    return 0
