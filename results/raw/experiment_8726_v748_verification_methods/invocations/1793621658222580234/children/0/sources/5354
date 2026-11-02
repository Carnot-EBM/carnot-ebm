"""REQ-REPORT-7978-V691: freeze bounded validation and publish exact checked bytes."""

from datetime import UTC, datetime
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import yaml

from carnot.reporting import v690_capstone_validation as prior
from carnot.reporting import v691_capstone as cap
from carnot.reporting.primary_publication import publish_primary, reader_receipt

ROOT = cap.ROOT
OWNED = (
    "python/carnot/reporting/v691_capstone.py",
    "python/carnot/reporting/v691_capstone_validation.py",
    "scripts/experiments/experiment_7978_v691_capstone.py",
)
TEST = "tests/python/test_experiment_7978_v691_capstone.py"
run_check = prior.run_check


def manifest(private: Path, *, health: dict[str, Any] | None = None) -> dict[str, Any]:
    """Reuse qualified command construction with current fixtures and exact includes."""
    private.mkdir(parents=True, exist_ok=True)
    value = prior.manifest(private)
    fixture = private / "fixture"
    for name, target in (("design.md", "design.md"), ("active.yaml", "research-roadmap.yaml")):
        (fixture / target).write_bytes(
            gzip.decompress((ROOT / f"tests/fixtures/v691/{name}.gz").read_bytes())
        )
    (fixture / "research-roadmap-next.yaml").write_bytes(
        (fixture / "research-roadmap.yaml").read_bytes()
    )
    tasks = yaml.safe_load((fixture / "research-roadmap.yaml").read_bytes())["tasks"]
    for task in tasks[:-1]:
        gates = {
            g["artifact_field"]: g["value"]
            for other in tasks
            for g in other.get("gated_on", [])
            if g["upstream"] == task["id"] and g["op"] == "=="
        }
        cap.atomic_json(
            fixture / task["deliverable"],
            dict(
                experiment_id=int(task["id"][3:7]),
                task_id=task["id"],
                milestone="2026.10.691",
                run_date="20260930",
                execution_date="20260930",
                current_run_id=task["id"] + "-20260930",
                started_at="2026-09-30T23:59:59Z",
                finished_at="2026-10-01T00:00:01Z",
                MODEL_SPECS=[],
                honest_verdict="complete_null_fixture",
                verdict_class="null",
                flagged_adversarial=False,
                rows=[],
            )
            | gates,
        )
    before = "--include=" + ",".join(str(ROOT / name) for name in prior.OWNED)
    after = "--include=" + ",".join(str(ROOT / name) for name in OWNED)
    for row in value["commands"]:
        row["argv"] = [
            arg.replace(before, after)
            .replace(prior.TEST, TEST)
            .replace(prior.OWNED[-1], OWNED[-1])
            .replace("experiment_7965_capstone.json", "experiment_7978_capstone.json")
            for arg in row["argv"]
        ]
        if row["name"] in {"ruff", "format", "mypy"}:
            row["argv"] = row["argv"][: 3 if row["name"] == "format" else 2] + list(OWNED)
            if row["name"] != "mypy":
                row["argv"].append(TEST)
        if row["name"] == "current_authority_consumers":
            row["argv"][1] = "tests/python/test_experiment_7966_v691_contract_methods.py"
    replay = next(r for r in value["commands"] if r["name"] == "cli_replay")
    negative = json.loads(json.dumps(replay))
    negative.update(
        name="cli_negative_replay",
        expected_exit=1,
        failure_reason="sample_size_budget_changed",
        fixture_mutation=dict(
            source=str(private / "success/experiment_7978_capstone.json"),
            output=str(private / "negative/experiment_7978_capstone.json"),
        ),
    )
    negative["argv"] = [
        a.replace(
            "success/experiment_7978_capstone.json", "negative/experiment_7978_capstone.json"
        ).replace("replay.coverage", "negative.coverage")
        for a in negative["argv"]
    ]
    blocked = json.loads(json.dumps(replay))
    blocked.update(name="cli_blocked_replay")
    blocked["argv"] = [
        a.replace(str(fixture), str(private / "absent"))
        .replace("success/experiment_7978_capstone.json", "block/experiment_7978_capstone.json")
        .replace("replay.coverage", "blocked-replay.coverage")
        for a in blocked["argv"]
    ]
    position = next(i for i, r in enumerate(value["commands"]) if r["name"] == "coverage_combine")
    value["commands"][position:position] = [negative, blocked]
    next(r for r in value["commands"] if r["name"] == "coverage_combine")["argv"] += [
        str(private / "negative.coverage"),
        str(private / "blocked-replay.coverage"),
    ]
    py, pytest = str(ROOT / ".venv/bin/python"), str(ROOT / ".venv/bin/pytest")
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    for name, tests in (
        ("e2e019", ["tests/python/test_experiment_7942_v689_sentence_labels.py"]),
        (
            "checkpoint_consumers",
            [
                "tests/python/test_experiment_7968_v691_response_role_targets.py",
                "tests/python/test_experiment_7972_v691_qwen_energy_calibration.py",
                "tests/python/test_service_cost_7976.py",
            ],
        ),
    ):
        value["commands"].append(
            dict(
                name=name,
                argv=[pytest, *tests, *common, f"--basetemp={private / name}"],
                expected_exit=0,
                failure_reason=None,
                deadline_s=300,
                classification="required",
            )
        )
    value["commands"].append(
        dict(
            name="repository_full_suite",
            argv=[pytest, "tests/python", "-q", f"--basetemp={private / 'repository-health'}"],
            expected_exit=0,
            failure_reason=None,
            deadline_s=600,
            classification="diagnostic",
        )
    )
    specs = sorted(
        {
            arg
            for row in value["commands"]
            for arg in row["argv"]
            if arg.startswith("tests/") and arg.endswith(".py")
        }
    )
    next(r for r in value["commands"] if r["name"] == "spec")["argv"] = [
        py,
        "scripts/check_spec_coverage.py",
        *specs,
    ]
    value.update(
        coverage_includes=list(OWNED),
        execution_date="20261001",
        producer_invocations=cap.freeze_producers(fixture, fixture / "research-roadmap.yaml"),
        scratch_root=str(private),
        dependency_hashes=prior.prior.dependency_hashes(ROOT, paths=[*OWNED, TEST, *specs]),
    )
    value["dependency_hashes"].update(
        {
            name: cap.sha256_file(ROOT / name)
            for name in (
                "pyproject.toml",
                "openspec/capabilities/research-reporting/spec.md",
                "ops/exclusion_manifest.yaml",
                "tests/fixtures/v691/active.yaml.gz",
                "tests/fixtures/v691/design.md.gz",
            )
        }
    )
    value["prior_attempt"] = health
    if health and health["value"]["repository_health"].get("current_full_suite"):
        value["commands"] = [r for r in value["commands"] if r["name"] != "repository_full_suite"]
    return value


def archive_attempt(output: Path) -> dict[str, Any]:
    """Keep the exact failed primary before replacing its task-owned publication path."""
    digest = cap.sha256_file(output)
    archived = output.parent / "raw" / output.stem / "attempts" / (digest[7:] + ".json")
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_bytes(output.read_bytes())
    return dict(path=str(archived), sha256=digest, value=json.loads(archived.read_bytes()))


def disqualify(
    value: dict[str, Any], reason: str, path: Path, expected: Any, observed: Any
) -> None:
    """Owned failures remove readiness even when science already lacks prerequisites."""
    value.update(
        honest_verdict="complete_disqualified_" + reason,
        verdict_class="disqualified",
        capstone_execution_ready_score=0,
    )
    value["acceptance_gate_results"].update(validity=False, readiness=0)
    value["gate_check_summary"].append(
        cap.prior.prior.shared.operand(
            path,
            cap.sha256_file(path) if path.is_file() else None,
            "exp7978-owned-validation",
            reason,
            expected,
            observed,
        )
    )


def terminal(value: dict[str, Any], output: Path, private: Path, durable: Path) -> None:
    """Recheck changed verdict bytes and bind both live consumers to the final primary."""
    sidecar, candidate = durable / "terminal-validation.json", private / "candidate.json"
    value["terminal_validation_sidecar_path"] = str(sidecar)
    for attempt in range(3):
        cap.atomic_json(candidate, value)
        receipts = [
            run_check(
                ROOT,
                dict(
                    name=name,
                    argv=[str(ROOT / ".venv/bin/python"), script, flag, str(candidate)],
                    expected_exit=0,
                    deadline_s=60,
                    classification="terminal",
                ),
                private / f"terminal-{attempt}",
                durable / "logs",
            )
            for name, script, flag in (
                ("adversarial_verify", "scripts/adversarial_verify.py", "--json"),
                ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
            )
        ]
        receipts.append(
            run_check(
                ROOT,
                dict(
                    name="terminal_cold_replay",
                    argv=[
                        str(ROOT / ".venv/bin/python"),
                        str(ROOT / OWNED[-1]),
                        "--date",
                        "20261001",
                        "--root",
                        str(Path(value["rows"][0]["path"]).parent.parent),
                        "--design",
                        value["authority_snapshots"]["design"]["source_path"],
                        "--active",
                        value["authority_snapshots"]["active"]["source_path"],
                        "--cold-replay",
                        str(candidate),
                    ],
                    expected_exit=0,
                    deadline_s=120,
                    classification="terminal",
                ),
                private / f"terminal-{attempt}",
                durable / "logs",
            )
        )
        try:
            report = json.loads(Path(receipts[0]["log_path"]).read_text())
            if type(report["flagged_count"]) is not int:
                raise ValueError("invalid_report")
        except (ValueError, KeyError, TypeError):
            report = dict(flagged_count=None, parse_error="invalid_validator_report")
            receipts[0]["passed"] = False
        flagged = bool(report["flagged_count"])
        failed = flagged or any(not r["passed"] for r in receipts)
        digest = cap.sha256_file(candidate)
        cap.atomic_json(
            sidecar,
            dict(
                candidate_sha256=digest,
                primary_path=str(output),
                adversarial_report=report,
                strict_row_report=Path(receipts[1]["log_path"]).read_text(),
                validator_receipts=receipts,
            ),
        )
        if failed and attempt == 0:
            disqualify(value, "terminal_verification", candidate, "unflagged and passing", report)
            value["flagged_adversarial"] = flagged
            continue
        if value["flagged_adversarial"] != flagged:
            value["flagged_adversarial"] = flagged
            continue
        break
    else:
        raise ValueError("terminal_flags_unstable")
    publish_primary(
        output,
        value,
        lambda path: dict(
            passed=cap.sha256_file(path) == digest,
            terminal_sidecar=str(sidecar),
            terminal_checks_passed=not failed,
        ),
    )
    newer = output.stat().st_mtime + 2
    os.utime(sidecar, (newer, newer))
    reading = reader_receipt(
        "exp7978-capstone",
        output.parent,
        field="capstone_execution_ready_score",
        expected=value["capstone_execution_ready_score"],
    )
    if (
        not reading["passed"]
        or reading["gate_sha256"] != digest
        or reading["document_sha256"] != digest
    ):
        raise ValueError("primary_reader_drift")
    cap.atomic_json(Path(value["primary_resolution_receipt"]["path"]), reading)


def qualify(root: Path, design: Path, active: Path, date: str, output: Path) -> int:
    """Archive owned receipts after children exit and qualify only the frozen new scope."""
    started, start_ns = time.monotonic(), time.monotonic_ns()
    cap.progress(started, "qualification_start", 0)
    with tempfile.TemporaryDirectory(prefix="carnot-7978-validation-") as directory:
        private = Path(directory)
        previous = archive_attempt(output) if output.is_file() else None
        frozen = manifest(private, health=previous)
        invocations = cap.freeze_producers(root, active)
        frozen["current_producer_invocations"] = invocations
        durable = (
            root / "results/raw/experiment_7978_v691_capstone" / cap.canonical_hash(frozen)[7:]
        )
        manifest_path = durable / "validation-command-manifest.json"
        cap.atomic_json(manifest_path, frozen)
        value = cap.build_candidate(
            root, design, active, date, snapshots=durable / "authority", invocations=invocations
        )
        value.update(
            started_monotonic_ns=start_ns,
            validation_command_manifest_path=str(manifest_path),
            scratch_root_receipt=dict(
                path=str(private), private=True, outside_checkout=True, cleaned_on_exit=True
            ),
        )
        cap.atomic_json(
            durable / "input-checkpoint.json",
            dict(
                manifest_sha256=cap.sha256_file(manifest_path),
                input_sha256=cap.canonical_hash(value["source_artifact_hashes"]),
                status="code_config_input_frozen",
            ),
        )
        receipts = []
        for spec in frozen["commands"]:
            if spec.get("fixture_mutation"):
                mutation = spec["fixture_mutation"]
                tampered = json.loads(Path(mutation["source"]).read_bytes())
                tampered["sample_size_budget"]["completed"] = 99
                cap.atomic_json(Path(mutation["output"]), tampered)
            row = run_check(ROOT, spec, private, durable / "logs")
            if row.get("failure_reason") and row["expected_exit"] != 0:
                row["passed"] = (
                    row["passed"] and row["failure_reason"] in Path(row["log_path"]).read_text()
                )
            if row["name"] == "publication_gate":
                prior.publication_result(value, row)
            receipts.append(row)
        report = private / "coverage.json"
        counts = json.loads(report.read_text())["files"] if report.is_file() else {}
        value["coverage_statement_counts"] = {
            name: item["summary"] for name, item in counts.items()
        }
        complete = prior.prior.coverage_complete(report, includes=list(OWNED))
        receipts.append(
            dict(
                name="nonempty_complete_statement_coverage",
                argv=["coverage_json_reduction", str(report)],
                expected_exit=0,
                actual_exit=int(not complete),
                passed=complete,
                deadline_s=0,
                classification="required",
            )
        )
        if report.is_file():
            shutil.copyfile(report, durable / "coverage.json")
        for shard in private.glob("*.coverage*"):
            if shard.is_file():
                shutil.copyfile(shard, durable / shard.name)
        value.update(
            validation_receipts=receipts, observed_child_commands=[r["argv"] for r in receipts]
        )
        value["repository_health"]["current_full_suite"] = [
            r for r in receipts if r["classification"] == "diagnostic"
        ]
        if previous:
            value["source_artifact_hashes"].append(
                dict(
                    path=previous["path"],
                    sha256=previous["sha256"],
                    role="earlier_owned_failed_attempt",
                    exposure="administrative",
                )
            )
            value["repository_health"]["prior_owned_health"] = previous["value"][
                "repository_health"
            ]
            value["historical_required_failures"].append(
                dict(
                    path=previous["path"],
                    sha256=previous["sha256"],
                    honest_verdict=previous["value"]["honest_verdict"],
                    current_pass=False,
                    required_failures=[
                        r
                        for r in previous["value"]["validation_receipts"]
                        if r["classification"] != "diagnostic" and not r["passed"]
                    ],
                )
            )
        prior_path = root / "results/experiment_7965_v690_capstone.json"
        old, old_hash = cap.prior.prior.shared.read(prior_path)
        value["repository_health"]["historical_receipts"] = [
            dict(
                path=str(prior_path),
                sha256=old_hash,
                repository_health=old.get("repository_health"),
                current_pass=False,
            )
        ]
        failures = [r for r in receipts if r["classification"] != "diagnostic" and not r["passed"]]
        for row in failures:
            disqualify(
                value,
                "required_validation",
                Path(row.get("log_path", str(report))),
                row["expected_exit"],
                row["actual_exit"],
            )
        if not failures:
            value["capstone_execution_ready_score"] = 1
            value["acceptance_gate_results"].update(validity=True, readiness=1)
        value["source_artifact_hashes"].extend(
            dict(
                path=str(ROOT / path),
                sha256=digest,
                role="frozen_dependency",
                exposure="administrative",
            )
            for path, digest in frozen["dependency_hashes"].items()
        )
        value["resolved_imports"][__name__] = str(Path(__file__).resolve())
        cap.atomic_json(
            durable / "primitive-reductions.json", dict(rows=value["independent_reduction_rows"])
        )
        errors = cap.cold_replay(value, root, design, active)
        if errors:
            disqualify(value, "cold_replay", durable / "primitive-reductions.json", [], errors)
        value["ended_monotonic_ns"] = time.monotonic_ns()
        value["duration_s"] = (value["ended_monotonic_ns"] - start_ns) / 1e9
        value["finished_at"] = datetime.now(UTC).isoformat()
        value["phase_spans"].append(
            dict(
                phase="owned_validation",
                start_s=value["phase_spans"][0]["end_s"],
                end_s=value["duration_s"],
                completed_units=len(receipts),
            )
        )
        value["field_principles"].update(
            {
                key: "Bind this field to measured owned work and frozen input identity."
                for key in value
            }
        )
        terminal(value, output, private / "terminal", durable / "terminal")
        cap.atomic_json(durable / "planning-handoff.json", value["planning_handoff"])
    cap.progress(started, "published_checked_bytes", 13)
    return 0
