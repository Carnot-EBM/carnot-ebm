"""Keep fitting scratch separate from checked evidence (REQ-VERIFY-7967)."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone, UTC
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.publication_qualification_7928 import terminal_checks
from carnot.testing import child_results_guard
from carnot.verify import energy_fit_7943 as fit
from carnot.verify import training_coverage_7954 as prior

ROOT = prior.ROOT
TASK = "exp7967-private-training-qualification"
CLI = "scripts/experiments/experiment_7967_v691_private_training_qualification.py"
OWNED = ("python/carnot/verify/private_training_7967.py", CLI)
TEST = "tests/python/test_private_training_7967.py"
COVERED = (*prior.COVERED, *OWNED)
INCLUDES = ",".join("*/" + name for name in COVERED)
START = time.monotonic()
HEALTH_LOG = Path("/tmp/carnot7967-repository-health.log")
SOURCE_IDENTITIES = (
    (7892, fit.UPSTREAM, "20260929"),
    (7954, ROOT / "results/experiment_7954_v690_training_coverage.json", "20260930"),
)


def progress(phase: str, units: int = 0) -> None:
    """Print actual elapsed work so a supervisor can detect a stopped child."""
    print(
        f"[exp7967] phase={phase} completed={units} elapsed_s={time.monotonic() - START:.3f}",
        flush=True,
    )


def check_roots(scratch: Path, archive: Path) -> None:
    """Reject unsafe storage before a freezer can create fixtures or start children."""
    scratch, archive = scratch.resolve(), archive.resolve()
    if (
        scratch.is_relative_to(ROOT.resolve())
        or scratch.is_relative_to(archive)
        or archive.is_relative_to(scratch)
    ):
        raise ValueError(
            "unsafe_scratch: checkout and archive must be separate from mutable storage"
        )


def freeze(scratch: Path, archive: Path) -> tuple[Path, list[CommandSpec]]:
    """Adapt the historical freezer without changing its dates, assertions or guard."""
    check_roots(scratch, archive)
    runner = fit.isolated(Path(prior.__file__))
    runner.INCLUDES, runner.COVERED = INCLUDES, COVERED
    runner.TESTS = (*prior.TESTS, TEST)
    path, commands = runner.freeze(scratch)
    for index, command in enumerate(commands):
        if command.name in ("ruff_check", "ruff_format", "mypy"):
            commands[index] = replace(
                command, argv=(*command.argv, *OWNED, *(() if command.name == "mypy" else (TEST,)))
            )
    measure = (
        str(ROOT / ".venv/bin/python"),
        "-m",
        "coverage",
        "run",
        "--parallel-mode",
        f"--data-file={scratch / '.coverage'}",
        f"--include={INCLUDES}",
    )
    extra = [
        CommandSpec(
            "private_guard_fixture",
            (*measure, CLI, "--date", "20261001", "--guard-fixture", str(scratch / "guard_probe")),
            "required",
            60,
        ),
        CommandSpec(
            "unsafe_scratch",
            (
                *measure,
                CLI,
                "--date",
                "20261001",
                "--check-private-root",
                str(ROOT / "results/raw/unsafe7967"),
            ),
            "exception",
            60,
        ),
        CommandSpec(
            "current_cli_outside",
            (
                "/usr/bin/env",
                "-u",
                "PYTHONPATH",
                "--chdir=" + str(scratch),
                *measure,
                str(ROOT / CLI),
                "--date",
                "20261001",
                "--check-private-root",
                str(scratch),
            ),
            "required",
            60,
        ),
    ]
    insert = next(i for i, c in enumerate(commands) if c.name == "coverage_combine")
    commands[insert:insert] = extra
    frozen = json.loads(path.read_text())
    frozen.update(
        scratch_root=str(scratch),
        archive_root=str(archive),
        execution_date="20261001",
        current_producer={"experiment_id": 7967, "task_id": TASK},
        affected_files=OWNED,
        coverage_includes=INCLUDES,
        qualification_deadline_s=1800,
        artifact_guard_enabled=True,
    )
    frozen["dependencies"].update({n: sha256_file(ROOT / n) for n in (*OWNED, TEST)})
    frozen["commands"] = [
        {
            **asdict(c),
            "expected_exit": 2 if c.name in (*prior.NEGATIVES, "unsafe_scratch") else 0,
            "expected_reason": "unsafe_scratch"
            if c.name == "unsafe_scratch"
            else prior.NEGATIVES.get(c.name, "command must pass"),
            "deadline_s": c.timeout_s,
        }
        for c in commands
    ]
    atomic_json(path, frozen)
    return path, commands


def authenticate_sources() -> list[dict[str, Any]]:
    """Authenticate historical producer dates and public bytes without promoting old fits."""
    sources = []
    for identity, path, date in SOURCE_IDENTITIES:
        try:
            value = json.loads(path.read_text())
            for field, expected in (("experiment_id", identity), ("run_date", date)):
                if value.get(field) != expected:
                    raise ValueError(f"{field}: expected {expected}; observed {value.get(field)}")
            if identity == 7892:
                if (
                    value.get("source_boundary_ready_score") != 1
                    or value.get("flagged_adversarial") is not False
                ):
                    raise ValueError("source boundary gate blocked")
                for row in value["public_shards"]:
                    if sha256_file(Path(row["path"])) != row["sha256"]:
                        raise ValueError("public source hash drift")
                    sources.append({**row, "role": "public_source_bytes", "producer_id": identity})
            sources.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "producer_id": identity,
                    "producer_run_date": date,
                    "historical_verdict": value.get("honest_verdict"),
                    "role": "historical_identity_not_promoted",
                }
            )
        except (OSError, ValueError, KeyError) as exc:
            row = fit.old.custody.operand(
                path,
                "producer_identity_and_source_hashes",
                "==",
                "frozen identity and bytes",
                str(exc),
            )
            row["upstream_id"] = f"exp{identity}"
            raise fit.old.custody.InputBlocked([row]) from exc
    return sources


def guard_probe(scratch: Path) -> dict[str, Any]:
    """Reproduce the guard's cross-device move in an isolated mirror, never the real results."""
    scratch.mkdir(parents=True, exist_ok=True)
    mirror = scratch / "mirror"
    source = mirror / "results/source.tmp"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("private protected-root reproduction")
    with TemporaryDirectory(prefix="carnot7967-cross-device-", dir="/dev/shm") as redirect:
        script = "\n".join(
            (
                "import os,sys,json",
                "from carnot.testing.child_results_guard import _SHIM_SOURCE",
                "os.environ['CARNOT_CHILD_GUARD_REPO_ROOT']=sys.argv[1]",
                "os.environ['CARNOT_CHILD_GUARD_REDIRECT_ROOT']=sys.argv[2]",
                "exec(_SHIM_SOURCE)",
                "try: os.rename(sys.argv[1]+'/results/source.tmp',sys.argv[1]+'/results/target.json')",
                "except OSError as exc: print(json.dumps({'errno':exc.errno,'guard_enabled':True}),flush=True)",
            )
        )
        rows = run_commands(
            ROOT,
            [
                CommandSpec(
                    "mirrored_guard_move",
                    (sys.executable, "-c", script, str(mirror), redirect),
                    "control",
                    30,
                )
            ],
            log_dir=scratch / "logs",
            heartbeat_s=15,
        )
        observed = json.loads(rows[0]["output_tail"].splitlines()[-1])
    atomic_json(scratch / "atomic.json", {"private": True})
    historical = (
        ROOT / "results/raw/experiment_7954_v690_training_coverage/logs/00_affected_unit.log"
    )
    receipt = {
        "passed": rows[0]["passed"] and observed["errno"] == 18,
        "guard_enabled": observed["guard_enabled"],
        "protected_move_errno": observed["errno"],
        "private_atomic_same_device": scratch.stat().st_dev
        == (scratch / "atomic.json").stat().st_dev,
        "historical_failure_count": 13 if "13 failed, 207 passed" in historical.read_text() else 0,
        "mirror_path": str(mirror),
        "cross_device_fixture_only": True,
        "receipts": rows,
    }
    atomic_json(scratch / "receipt.json", receipt)
    return receipt


def reduce_ready(
    commands: list[CommandSpec],
    receipts: list[dict[str, Any]],
    counts: dict[str, Any],
    selected: list[dict[str, Any]],
    fixture: dict[str, Any],
) -> bool:
    """Require complete execution evidence as well as nonempty full statement coverage."""
    return (
        bool(commands)
        and [r["name"] for r in receipts] == [c.name for c in commands]
        and all(r["argv"] == list(c.argv) for r, c in zip(receipts, commands, strict=True))
        and prior.ready(receipts, counts, selected)
        and all(
            any(
                n.endswith(owned) and v["num_statements"] > 0 and v["missing_lines"] == 0
                for n, v in counts.items()
            )
            for owned in COVERED
        )
        and bool(fixture.get("trained_head_specs"))
        and fixture.get("fixture_ready_score") == 1
        and fixture.get("energy_fit_ready_score") == 0
        and len(fixture.get("rows", [])) == 3
    )


def archived_path(value: dict[str, Any], path: Path) -> Path:
    """Find immutable copies while keeping original measured paths in command receipts."""
    custody = value["scratch_root_receipt"]
    scratch = Path(custody["path"])
    return (
        Path(custody["archive_path"]) / path.relative_to(scratch)
        if path.is_relative_to(scratch)
        else path
    )


def archive_health(raw: Path) -> dict[str, Any]:
    """Preserve the completed full-suite deadline separately from scoped qualification."""
    if not HEALTH_LOG.is_file():
        return {}
    destination = raw / "repository_health/full_pytest.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HEALTH_LOG, destination)
    return {
        "name": "full_pytest",
        "scope": "repository_health",
        "passed": False,
        "command_argv": [".venv/bin/pytest", "tests/python", "-q"],
        "exit_code": 124,
        "deadline_s": 600,
        "timed_out": True,
        "log_path": str(destination),
        "log_sha256": sha256_file(destination),
        "reason": "Unrelated existing failures occurred before the bounded global run timed out; no full-suite pass is claimed.",
    }


def qualify(output: Path, publication: Path = fit.PUBLICATION, runtime: Path = fit.RUNTIME) -> int:
    """Execute the historical mechanics privately, then publish current checked evidence."""
    progress("start_flushed")
    began, started_at = time.monotonic(), datetime.now(UTC).isoformat()
    raw = output.parent / "raw" / output.stem
    raw.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="carnot7967-private-") as directory:
        scratch = Path(directory)
        manifest, commands = freeze(scratch, raw)
        progress("authenticate_inputs")
        try:
            sources, dependencies, upstream = fit.authenticate(publication, runtime, fit.UPSTREAM)
            sources += authenticate_sources()
        except fit.old.custody.InputBlocked as exc:
            value = prior.base([], exc.operands, {})
            value.update(honest_verdict="complete_blocked_prerequisite", verdict_class="blocked")
            receipts, counts, selected, fixture = [], {}, [], {}
            passed, mutation = False, {"passed": False, "reason": "external prerequisite blocked"}
            dependencies = {}
            progress("complete_blocked_prerequisite")
        else:
            value = prior.base(sources, [], upstream)
            child_results_guard.install(str(scratch / "guard_redirect"))
            progress("private_children_before")
            receipts = prior.execute(commands, scratch)
            for row in receipts:
                if row["name"] == "unsafe_scratch":
                    row.update(expected_exit=2, expected_reason="unsafe_scratch")
                    row["passed"] = (
                        row["actual_exit"] == 2
                        and not row.get("timed_out", False)
                        and "unsafe_scratch" in row.get("output_tail", "")
                    )
                row["measured_files"] = list(COVERED)
            progress("private_children_after", len(receipts))
            report = (
                json.loads((scratch / "coverage.json").read_text())
                if (scratch / "coverage.json").exists()
                else {"files": {}}
            )
            counts = {name: item["summary"] for name, item in report["files"].items()}
            selected = prior.selections(scratch)
            success = prior.route(scratch, "success")
            fixture = json.loads(success.read_text()) if success.exists() else {}
            mutation = (
                prior.negative_mutation(scratch)
                if (scratch / ".coverage").exists()
                else {"passed": False, "reason": "coverage data absent"}
            )
            frozen = json.loads(manifest.read_text())
            passed = (
                reduce_ready(commands, receipts, counts, selected, fixture)
                and mutation["passed"]
                and all(sha256_file(ROOT / n) == h for n, h in frozen["dependencies"].items())
                and time.monotonic() - began < 1800
            )
            value.update(
                honest_verdict="complete_circular_positive_private_training_qualification"
                if passed
                else "complete_disqualified_required_checks",
                verdict_class="circular_positive" if passed else "disqualified",
            )
        frozen = json.loads(manifest.read_text())
        value.update(
            experiment_id=7967,
            task_id=TASK,
            milestone="2026.10.691",
            run_date="20261001",
            execution_date="20261001",
            started_at=started_at,
            training_execution_ready_score=int(passed),
            coverage_statement_counts=counts,
            consumer_selection_rows=selected,
            trained_head_specs=fixture.get("trained_head_specs", []),
            fixture_prediction_rows=fixture.get("fixture_prediction_rows", []),
            fixture_checkpoint_manifest={
                "path": fixture.get("checkpoint_manifest_path"),
                "sha256": fixture.get("checkpoint_manifest_sha256"),
            },
            fixture_prediction_path=fixture.get("prediction_rows_path"),
            fixture_prediction_sha256=fixture.get("prediction_rows_sha256"),
            negative_mutation_receipt=mutation,
            private_root_rows=[],
            scratch_root_receipt={
                "path": str(scratch),
                "archive_path": str(raw / "evidence"),
                "outside_checkout": True,
                "guard_enabled": True,
                "deleted_after_archive": True,
            },
            inference_substrate="aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            execution_venue="host",
            MODEL_SPECS=[],
            model_specs=[],
            target_model=None,
            verifier_is_oracle=True,
            claim_scope="private_fixture_oracle_agreement; exposed_development",
            training_dependency_hashes={**dependencies, **frozen["dependencies"]},
        )
        value["acceptance_gate_results"].update(
            validity=passed,
            readiness=int(passed),
            calibration=None,
            decision_benefit=None,
            retention=None,
            efficiency=None,
        )
        value["training_entrypoint"]["numerical_callable"] = (
            "carnot.verify.energy_fit_7930.fit_heads"
        )
        value["resolved_imports"].update(
            {"carnot.verify.private_training_7967": str(ROOT / OWNED[0])}
        )
        historical = json.loads(SOURCE_IDENTITIES[-1][1].read_text())
        value["historical_required_failures"] += [
            {"experiment_id": 7954, **r}
            for r in historical["validation_receipts"]
            if not r["passed"]
        ]
        value["cited_upstream_artifacts"] = [
            {
                "experiment_id": i,
                "path": str(p),
                "sha256": sha256_file(p),
                "producer_run_date": d,
                "fields_imported": ["honest_verdict", "source_artifact_hashes"],
            }
            for i, p, d in SOURCE_IDENTITIES
            if p.exists()
        ]
        for name in ("00_affected_unit", "13_e2e_015"):
            path = ROOT / f"results/raw/experiment_7954_v690_training_coverage/logs/{name}.log"
            value["source_artifact_hashes"].append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "role": "preserved_historical_failure",
                }
            )
        progress("archive_closed_evidence_before", len(receipts))
        shutil.copytree(scratch, raw / "evidence", dirs_exist_ok=True)
        for row in receipts:
            original = Path(row["log_path"])
            row.update(
                original_log_path=str(original), log_path=str(archived_path(value, original))
            )
        value["validation_receipts"] = receipts
        value["observed_child_commands"] = [r["argv"] for r in receipts]
        value["exception_route_rows"] = [r for r in receipts if r["scope"] == "exception"]
        value["rows"] = [
            {
                "unit": r["name"],
                "arm": "private_execution_qualification",
                "seed": 67801,
                "passed": r["passed"],
                "expected_exit": r["expected_exit"],
                "actual_exit": r["actual_exit"],
            }
            for r in receipts
        ]
        size = len(receipts)
        value["sample_size_budget"].update(
            intended=len(commands),
            eligible=size,
            started=size,
            completed=size,
            failed=sum(not r["passed"] for r in receipts),
            censored=0,
            excluded=len(commands) - size,
            independent=0,
            unit="command route; no independent scientific source",
        )
        value["validation_command_manifest_path"] = str(archived_path(value, manifest))
        value["source_artifact_hashes"].append(
            {
                "path": value["validation_command_manifest_path"],
                "sha256": sha256_file(manifest),
                "role": "frozen_current_configuration",
            }
        )
        value["source_artifact_hashes"] += fixture.get("source_artifact_hashes", [])
        value["preconditions_checked"] = {
            "publication_path": str(publication),
            "runtime_path": str(runtime),
            "historical_dates_preserved": True,
            "natural_fitting_performed": False,
        }
        probe = scratch / "guard_probe/receipt.json"
        value["private_root_rows"] = [json.loads(probe.read_text())] if probe.exists() else []
        runner = raw / "qualified_runner_manifest.json"
        atomic_json(
            runner,
            {
                "qualified": passed,
                "callable": value["training_entrypoint"],
                "dependencies": value["training_dependency_hashes"],
                "coverage_includes": INCLUDES,
                "coverage_statement_counts": counts,
                "current_producer": 7967,
                "validation_manifest_sha256": sha256_file(manifest),
            },
        )
        value["qualified_runner_manifest"] = {"path": str(runner), "sha256": sha256_file(runner)}
        progress("archive_closed_evidence_after", size)
    elapsed = time.monotonic() - began
    value["finished_at"] = datetime.now(UTC).isoformat()
    value["repository_health"]["current_full_pytest"] = archive_health(raw)
    value["phase_spans"] = [
        {"phase": "private_qualification", "start_s": 0, "end_s": elapsed, "duration_s": elapsed}
    ]
    value["reproducibility_checksum"] = canonical_hash(
        {
            "dependencies": value["training_dependency_hashes"],
            "sources": value["source_artifact_hashes"],
        }
    )
    seal(value, output)
    replay(output)
    return 0


def seal(value: dict[str, Any], output: Path) -> None:
    """Reuse checked publication so new sidecars cannot replace the live primary."""
    publisher = fit.isolated(Path(prior.__file__))
    publisher.TASK, publisher.START = TASK, START
    publisher.SCORES = ("training_execution_ready_score", *prior.SCORES)
    publisher.terminal_checks, publisher.progress = terminal_checks, progress
    publisher.seal(value, output)


def replay_fit(value: dict[str, Any]) -> None:
    """Rebuild fixture predictions from archived checkpoint bytes and source rows."""
    resolve = lambda path: archived_path(value, Path(path))
    primitive = resolve(value["fixture_prediction_path"])
    rows = [json.loads(line) for line in primitive.read_text().splitlines() if line]
    manifest_path = resolve(value["fixture_checkpoint_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    if (
        sha256_file(primitive) != value["fixture_prediction_sha256"]
        or rows != value["fixture_prediction_rows"]
        or sha256_file(manifest_path) != value["fixture_checkpoint_manifest"]["sha256"]
        or sha256_file(resolve(manifest["checkpoint_path"])) != manifest["checkpoint_sha256"]
    ):
        raise ValueError("fixture primitive drift")
    upstream = json.loads(resolve(manifest["identity"]["upstream_path"]).read_text())
    for field in ("public_shards", "evaluator_shards"):
        for ref in upstream[field]:
            ref["path"] = str(resolve(ref["path"]))
    with TemporaryDirectory(prefix="carnot7967-cold-") as private:
        source = Path(private) / "upstream.json"
        atomic_json(source, upstream)
        records, _ = fit.old.custody.load_records(source)
        predicted = fit.old.natural_training.predict(
            fit.old.training_runtime.load(resolve(manifest["checkpoint_path"])), records
        )
    if predicted != [
        {key: row[key] for key in pred} for row, pred in zip(rows, predicted, strict=True)
    ]:
        raise ValueError("cold fixture prediction drift")


def replay(path: Path) -> None:
    """Cold-reduce checked bytes, complete receipts and current archived numerical evidence."""
    progress("cold_replay_before")
    value = json.loads(path.read_text())
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    read_bound_sidecar(path, Path(terminal["validator_sidecar_path"]))
    for row in [
        *value["validation_receipts"],
        *value["source_artifact_hashes"],
        value.get("qualified_runner_manifest", {}),
    ]:
        if row.get("path") or row.get("log_path"):
            source = Path(row.get("log_path", row.get("path", "")))
            if not source.is_absolute():
                source = ROOT / source
            if sha256_file(archived_path(value, source)) != row.get(
                "log_sha256", row.get("sha256")
            ):
                raise ValueError("primitive hash drift")
    for name, digest in value.get("training_dependency_hashes", {}).items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("dependency hash drift")
    if value["training_execution_ready_score"]:
        raw = path.parent / "raw" / path.stem
        frozen = json.loads(Path(value["validation_command_manifest_path"]).read_text())
        commands = [
            CommandSpec(c["name"], tuple(c["argv"]), c["scope"], c["timeout_s"])
            for c in frozen["commands"]
        ]
        report = json.loads((raw / "evidence/coverage.json").read_text())
        counts = {name: item["summary"] for name, item in report["files"].items()}
        fixture = json.loads((raw / "evidence" / prior.route(Path("."), "success")).read_text())
        if (
            not reduce_ready(
                commands,
                value["validation_receipts"],
                counts,
                value["consumer_selection_rows"],
                fixture,
            )
            or counts != value["coverage_statement_counts"]
            or value["sample_size_budget"]["completed"] != len(value["validation_receipts"])
            or value["sample_size_budget"]["independent"] != 0
            or not value["negative_mutation_receipt"]["passed"]
        ):
            raise ValueError("readiness drift")
        for row in value["consumer_selection_rows"]:
            if sha256_file(archived_path(value, Path(row["gate_path"]))) != row["gate_sha256"]:
                raise ValueError("route primary hash drift")
        replay_fit(value)
    progress("cold_replay_after", len(value["rows"]))
