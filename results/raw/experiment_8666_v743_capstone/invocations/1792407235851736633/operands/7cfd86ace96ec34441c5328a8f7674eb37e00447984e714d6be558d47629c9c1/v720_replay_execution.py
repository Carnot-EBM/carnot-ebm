"""REQ-VERIFY-8358: freeze checks, supervise children and publish checked bytes."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import v720_terminal_replay as e
from carnot.reporting import v720_frozen_input_contract as contract
from carnot.reporting import v718_replay_history as policy
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Freeze only owned coverage plus unchanged consumer and private E2E commands."""
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\npatch = subprocess\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    */" + p + "\n" for p in e.OWNED)
    )
    py, coverage = str(e.ROOT / ".venv/bin/python"), str(e.ROOT / ".venv/bin/coverage")
    commands = [
        (
            "owned_tests",
            [
                coverage,
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
                e.TEST,
                "--basetemp=" + str(private / "unit"),
            ],
        ),
        (
            "coverage_combine",
            [coverage, "combine", "--data-file=" + str(private / ".coverage"), str(private)],
        ),
        (
            "owned_coverage",
            [
                coverage,
                "json",
                "--data-file=" + str(private / ".coverage"),
                "--include=" + ",".join("*/" + p for p in e.OWNED),
                "--fail-under=100",
                "-o",
                str(private / "coverage.json"),
            ],
        ),
        ("ruff_check", [str(e.ROOT / ".venv/bin/ruff"), "check", *e.OWNED, e.TEST]),
        ("ruff_format", [str(e.ROOT / ".venv/bin/ruff"), "format", "--check", *e.OWNED, e.TEST]),
        (
            "strict_mypy",
            [str(e.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *e.OWNED],
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", "--files", e.TEST]),
    ]
    for name, tests in [
        ("private_E2E018", ["tests/python/test_experiment_7891_v685_authority_lifecycle.py"]),
        ("private_E2E021", ["tests/python/test_restricted_decision_audit_8210.py"]),
        (
            "unchanged_consumers",
            [
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_arc_supervisor_frontier_8355.py",
                "tests/python/test_gatemate_change_ledger_8357.py",
            ],
        ),
    ]:
        commands.append(
            (
                name,
                [
                    str(e.ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    *tests,
                    "--basetemp=" + str(private / name),
                ],
            )
        )
    return [
        dict(
            name=name,
            argv=argv,
            expected=0,
            deadline=600 if name == "owned_tests" else 300,
            scope="owned",
        )
        for name, argv in commands
    ]


def preflight(private: Path) -> list[Json]:
    """Check real scratch and executables before reading any producer evidence."""
    probe = private / "write_probe"
    probe.write_bytes(b"private scratch")
    checks = [
        e.gate(str(private), "private_scratch_rw", True, probe.read_bytes() == b"private scratch"),
        e.gate(
            str(private),
            "private_scratch_capacity",
            True,
            shutil.disk_usage(private).free >= 1_000_000_000,
        ),
    ]
    checks.extend(
        e.gate(
            str(e.ROOT / ".venv/bin" / tool),
            "executable",
            True,
            os.access(e.ROOT / ".venv/bin" / tool, os.X_OK),
        )
        for tool in ["python", "pytest", "coverage", "ruff", "mypy"]
    )
    e.progress("preconditions_after", len(checks), 0)
    return checks


def freeze(paths: list[Path], raw: Path) -> list[Json]:
    """Copy small current configuration files once and retain their original labels."""
    refs = []
    for source in paths:
        if source.is_file():
            target = raw / (sha256_file(source)[7:] + source.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            target.chmod(0o400)
            refs.append(dict(reference(target), original_path=str(source.absolute())))
    return refs


def controls(value: Json, raw: Path) -> list[Json]:
    """Use real children for cold success, changed claims and repaired-hash attacks."""
    receipts = []
    for name in ["valid", "changed-disposition", "wrong-authority", "rehashed-tamper"]:
        candidate = deepcopy(value)
        if name == "changed-disposition":
            candidate["verdict_class"] = "positive"
        elif name == "wrong-authority":
            candidate["authority_task_sha256"] = "wrong"
        elif name == "rehashed-tamper":
            candidate["branch_replay_ready_score"] = 1 - value["branch_replay_ready_score"]
        candidate["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in candidate.items() if k != "reproducibility_checksum"}
        )
        path = raw / (name + ".json")
        atomic_json(path, candidate)
        receipts.append(
            child(
                "terminal_" + name,
                [
                    str(e.ROOT / ".venv/bin/python"),
                    "-u",
                    str(e.ROOT / e.CLI),
                    "--cold-replay",
                    str(path),
                ],
                raw / "logs",
                expected=int(name != "valid"),
                deadline=240,
                heartbeat=20,
            )
        )
    return receipts


def validate(path: Path, raw: Path) -> Json:
    """Keep every finding and use the shipped typed consumer without new exemptions."""
    adversarial = child(
        "adversarial",
        [
            str(e.ROOT / ".venv/bin/python"),
            "-u",
            "scripts/adversarial_verify.py",
            "--json",
            str(path),
        ],
        raw / "validators",
        deadline=60,
    )
    try:
        report = json.loads(Path(adversarial["stdout_path"]).read_bytes())
    except (ValueError, OSError):
        report = {}
    report.update(candidate_sha256=sha256_file(path), verifier_sha256=policy.verifier_hash())
    finding = policy.consume(
        report,
        path,
        adversarial["exit_code"],
        dict(recomputed=False, deliberate_error_rejected=False),
    )
    rows = child(
        "row_consistency",
        [
            str(e.ROOT / ".venv/bin/python"),
            "-u",
            "scripts/verdict_row_consistency_lint.py",
            "--strict",
            str(path),
        ],
        raw / "validators",
        deadline=60,
    )
    cold = child(
        "terminal_cold",
        [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
        raw / "validators",
        deadline=240,
        heartbeat=20,
    )
    result = dict(
        passed=finding["passed"] and rows["passed"] and cold["passed"],
        findings=finding,
        checks=[adversarial, rows, cold],
    )
    atomic_json(raw / "candidate_validation.json", result)
    return result


def operand_controls(private: Path, raw: Path) -> list[Json]:
    """Qualify both dispositions and each closure attack without replacing natural data."""
    receipts = []
    for failed in [False, True]:
        output = private / str(failed) / (e.NAME + ".json")
        value = e.private_producer(output, failed)
        bundle = e.c.capture(output, sha256_file(output), raw / str(failed) / "closure", {})
        for name in [
            "valid",
            "missing-source",
            "changed-source",
            "wrong-authority",
            "changed-disposition",
            "rehashed-tamper",
        ]:
            request = dict(
                bundle=deepcopy(bundle), authority=e.TASK_PIN, disposition=value["verdict_class"]
            )
            if name == "wrong-authority":
                request["authority"] = "wrong"
            elif name == "changed-disposition":
                request["disposition"] = "positive"
            elif name != "valid":
                ref = request["bundle"]["rows"][-1]["reference"]
                target = raw / str(failed) / (name + ".bin")
                ref["path"] = str(target)
                if name != "missing-source":
                    target.write_bytes(b"{}")
                if name == "rehashed-tamper":
                    ref["sha256"] = sha256_file(target)
                    request["bundle"]["rows"][-1]["expected_sha256"] = ref["sha256"]
            result, receipt = e.invoke(
                request, raw / str(failed) / name, name, expected=int(name != "valid")
            )
            receipt["control_result"] = result
            receipts.append(receipt)
    return receipts


def run(output: Path, private: Path, *, fixture: bool = False) -> int:
    """Own qualification and atomic publication while the conductor owns ops updates."""
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    checks = preflight(private)
    authority_refs: list[Json] = []
    authority: Json = {}
    source_checks: list[Json] = []
    if not fixture:
        for identity, pin in e.c.PINS.items():
            primary = e.ROOT / "results" / (e.PRODUCERS[identity] + ".json")
            observed = sha256_file(primary) if primary.is_file() else None
            source_checks.append(e.gate(str(primary), "producer.sha256", pin, observed, observed))
            if observed == pin:
                value = json.loads(primary.read_bytes())
                side = primary.parent / "raw" / primary.stem / "validators" / (pin[7:] + ".json")
                terminal = Path(value["terminal_validation_sidecar_path"])
                for path, seal in zip([side, terminal], e.c.TERMINAL_PINS[identity], strict=True):
                    observed = sha256_file(path) if path.is_file() else None
                    source_checks.append(
                        e.gate(str(path), "terminal.sha256", seal, observed, observed)
                    )
        authority_refs = freeze(
            [
                e.ROOT / p
                for p in [contract.DESIGN, contract.ACTIVE, contract.STAGED, contract.PROTOCOL]
            ],
            raw / "authority",
        )
        try:
            actual = contract.authority(e.ROOT, private / "authority")
            task = next(t for t in actual["tasks"] if t["id"] == e.TASK)
            authority = dict(activated=actual["activated"], task=task)
            checks.append(
                e.gate(
                    str(e.ROOT / contract.DESIGN), "task.sha256", e.TASK_PIN, canonical_hash(task)
                )
            )
            checks.append(
                e.gate(
                    str(e.ROOT / contract.ACTIVE), "authority.activated", True, actual["activated"]
                )
            )
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            checks.append(
                e.gate(str(e.ROOT / contract.DESIGN), "authority_available", True, str(error))
            )
    plan = (
        manifest(private)
        if not fixture
        else [
            dict(
                name="private_check",
                argv=[
                    str(e.ROOT / ".venv/bin/python"),
                    "-u",
                    "-c",
                    "print('private qualification', flush=True)",
                ],
                expected=0,
                deadline=5,
                scope="owned",
            )
        ]
    )
    atomic_json(
        raw / "command_manifest.json",
        dict(commands=plan, frozen_before_measurement_ns=time.monotonic_ns()),
    )
    code_refs = freeze(
        [
            e.ROOT / p
            for p in [
                *e.OWNED,
                e.TEST,
                "python/carnot/testing/pytest_memory_watchdog.py",
                "python/carnot/reporting/primary_publication.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        raw / "code",
    )
    code_refs.extend(freeze([raw / "command_manifest.json"], raw / "code"))
    receipts = execute(plan, raw / "validation")
    coverage_path = private / "coverage.json"
    if coverage_path.is_file():
        code_refs.extend(freeze([coverage_path], raw / "coverage"))
    receipts.extend(operand_controls(private / "mechanical_controls", raw / "controls"))
    work = e.measure(raw, private=fixture)
    health_path = e.ROOT / "results/raw" / e.NAME / "repository_health/attempt.json"
    if not fixture and health_path.is_file():
        health = json.loads(health_path.read_bytes())
        receipts.append(health)
        work["repository_suite_attempt"] = health
        work["source_refs"].extend(freeze([health_path], raw / "repository_health"))
        if not health["passed"]:
            work["diagnostics"].append(
                e.gate(
                    str(health_path),
                    "repository_suite.exit_code",
                    0,
                    health["exit_code"],
                    sha256_file(health_path),
                )
            )
    work.update(
        authority_refs=authority_refs,
        authority=authority,
        code_refs=code_refs,
        preconditions_checked=checks + source_checks,
        frozen_validation_receipts=receipts,
    )
    work["diagnostics"].extend(c for c in checks if not c["passed"])
    if not all(c["passed"] for c in checks):
        receipts.append(dict(name="precondition_failure", passed=False, scope="owned"))
    value = normalize_artifact_for_template_write(e.build(work, receipts, raw, output))
    cold_controls = controls(value, raw / "terminal_controls")
    receipts.extend(cold_controls)
    work["frozen_validation_receipts"] = receipts
    value = normalize_artifact_for_template_write(e.build(work, receipts, raw, output))
    candidate = raw / "private_candidate.json"
    atomic_json(candidate, value)
    report = validate(candidate, raw / "private_candidate")
    work["adversarial_findings"] = report["findings"]["findings"]
    if not report["passed"]:
        receipts.append(dict(name="private_candidate_rejected", passed=False, scope="owned"))
    work["frozen_validation_receipts"] = receipts
    value = normalize_artifact_for_template_write(e.build(work, receipts, raw, output))
    publication = publish_primary(output, value, lambda p: validate(p, raw / "publication"))
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            private_candidate_validation=report,
            reader=reader_receipt(
                e.TASK,
                output.parent,
                field="branch_replay_ready_score",
                expected=value["branch_replay_ready_score"],
            ),
        ),
    )
    e.progress("publication_after", 1, 0)
    return int(value["verdict_class"] == "disqualified")
