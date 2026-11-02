"""REQ-VERIFY-8004: freeze checks and publish only independently checked bytes.

The shared supervisor owns child timing and heartbeats. This task adds its own
small validation plan rather than copying the failed capstone validation stack.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import v693_capstone as cap
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v686_contract_validation import run_check, coverage_complete

Json = dict[str, Any]
CONSUMERS = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_conductor_gates.py",
    "tests/python/test_in_process_doc_reconcile.py",
]


def manifest(private: Path) -> Json:
    """Bind exact commands, exits and coverage includes before reducing any outcomes."""
    py, cov, pytest, ruff, mypy = [
        str(cap.ROOT / ".venv/bin" / n) for n in ("python", "coverage", "pytest", "ruff", "mypy")
    ]
    prefix = [
        cov,
        "run",
        "--parallel-mode",
        f"--data-file={private / '.coverage'}",
        "--include=" + ",".join(str(cap.ROOT / n) for n in cap.OWNED),
    ]
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    commands: list[Json] = []

    def add(
        name: str,
        argv: list[str],
        expected: int = 0,
        classification: str = "required",
        deadline: int = 300,
    ) -> None:
        commands.append(
            dict(
                name=name,
                argv=argv,
                expected_exit=expected,
                classification=classification,
                deadline_s=deadline,
            )
        )

    add("publication_gate", [py, "scripts/publication_gate.py", "--json"], deadline=60)
    add(
        "owned_tests",
        prefix + ["-m", "pytest", *common, cap.TEST, f"--basetemp={private / 'unit'}"],
    )
    add(
        "E2E-018_and_consumers",
        [pytest, *common, *CONSUMERS, f"--basetemp={private / 'consumers'}"],
    )
    cli = [str(cap.ROOT / cap.OWNED[-1]), "--date", "20261002"]
    fixture = private / "fixture"
    out = private / "success/experiment_8004_v693_capstone.json"
    args = [
        "--root",
        str(fixture),
        "--active",
        str(fixture / "research-roadmap.yaml"),
        "--design",
        str(fixture / "design.md"),
        "--output",
        str(out),
        "--evidence-only",
    ]
    add("private_success", prefix + cli + args)
    add(
        "private_blocked",
        prefix
        + cli
        + [
            a.replace("success/", "blocked/").replace(
                str(fixture), str(private / "blocked_fixture")
            )
            for a in args
        ]
        + ["--missing-producer-fixture"],
    )
    add(
        "private_cold_replay",
        ["/usr/bin/env", "-u", "PYTHONPATH", *prefix, *cli, "--cold-replay", str(out)],
    )
    add(
        "expected_negative_replay",
        prefix + cli + ["--cold-replay", str(private / "absent.json"), "--expect-rejection"],
    )
    add("coverage_combine", [cov, "combine", f"--data-file={private / '.coverage'}", str(private)])
    add(
        "coverage_json",
        [cov, "json", f"--data-file={private / '.coverage'}", "-o", str(private / "coverage.json")],
    )
    add(
        "coverage_report",
        [
            cov,
            "report",
            f"--data-file={private / '.coverage'}",
            "--show-missing",
            "--fail-under=100",
        ],
    )
    add("ruff_check", [ruff, "check", *cap.OWNED, cap.TEST])
    add("ruff_format", [ruff, "format", "--check", *cap.OWNED, cap.TEST])
    add("strict_mypy", [mypy, "--strict", "--follow-imports=silent", *cap.OWNED])
    add("spec_coverage", [py, "scripts/check_spec_coverage.py", cap.TEST, *CONSUMERS])
    add("full_suite", [pytest, "tests/python", "-q"], classification="diagnostic", deadline=900)
    return dict(
        commands=commands,
        coverage_includes=cap.OWNED,
        dependency_hashes={
            n: sha256_file(cap.ROOT / n) for n in cap.OWNED + [cap.TEST] + CONSUMERS
        },
        scratch_root=str(private),
        execution_date="20261002",
        negative_replay_expected_exit=0,
        policy="owned checks required; repository health diagnostic only",
    )


def apply_checks(value: Json, receipts: list[Json], counts: Json) -> None:
    """Only observed owned checks qualify execution; scientific gaps stay separate."""
    required = [r for r in receipts if r.get("classification") != "diagnostic"]
    valid = bool(required) and all(r["passed"] for r in required)
    value.update(validation_receipts=receipts, coverage_statement_counts=counts)
    value["acceptance_gate_results"].update(
        validity=valid, readiness=int(valid and value["verdict_class"] == "null")
    )
    value["capstone_execution_ready_score"] = int(valid and value["verdict_class"] == "null")
    if not valid:
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_v693_owned_checks",
            science_ready=False,
            capstone_execution_ready_score=0,
        )
        for r in required:
            if not r["passed"]:
                value["gate_check_summary"].append(
                    cap.operand(
                        Path(r.get("log_path", "/missing")),
                        "exp8004-owned-validation",
                        r["name"],
                        r.get("expected_exit", 0),
                        r["actual_exit"],
                    )
                )
    cap.terminal_disposition(value)


def publish(value: Json, output: Path, private: Path, durable: Path) -> None:
    """Validators and both readers must agree on exactly the prospective primary bytes."""
    private.mkdir(parents=True, exist_ok=True)
    sidecar = durable / "terminal_validation.json"
    value["terminal_validation_sidecar_path"] = str(sidecar)
    candidate = private / output.name
    atomic_json(candidate, value)
    checks = [
        dict(
            name=name,
            argv=[str(cap.ROOT / ".venv/bin/python"), str(cap.ROOT / script), flag, str(candidate)],
            expected_exit=0,
            deadline_s=60,
            classification="required",
        )
        for name, script, flag in (
            ("adversarial_verify", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        )
    ]
    checks.append(
        dict(
            name="independent_cold_replay",
            argv=[
                "/usr/bin/env",
                "-u",
                "PYTHONPATH",
                str(cap.ROOT / ".venv/bin/python"),
                str(cap.ROOT / cap.OWNED[-1]),
                "--date",
                "20261002",
                "--cold-replay",
                str(candidate),
            ],
            expected_exit=0,
            deadline_s=60,
            classification="required",
        )
    )
    receipts = [
        run_check(private, spec, private / spec["name"], durable / "logs") for spec in checks
    ]
    if not all(r["passed"] for r in receipts):
        atomic_json(durable / "failed_terminal_receipts.json", dict(receipts=receipts))
        raise ValueError("terminal_validation_failed")
    adversarial = json.loads(Path(receipts[0]["log_path"]).read_text())
    if adversarial["flagged_count"]:
        raise ValueError("critical_adversarial_flag")
    digest = sha256_file(candidate)
    checked = private / "readers" / output.name
    atomic_json(checked, value)
    selected = reader_receipt(
        value["task_id"],
        checked.parent,
        field="capstone_execution_ready_score",
        expected=value["capstone_execution_ready_score"],
    )
    if (
        not selected["passed"]
        or selected["gate_sha256"] != digest
        or selected["document_sha256"] != digest
    ):
        raise ValueError("primary_reader_drift")
    atomic_json(
        sidecar,
        dict(
            passed=True,
            primary_path=str(output),
            primary_sha256=digest,
            receipts=receipts,
            private_readers=selected,
        ),
    )
    publication = publish_primary(
        output,
        value,
        lambda p: dict(passed=sha256_file(p) == digest, terminal_sidecar=str(sidecar)),
    )
    final_readers = reader_receipt(
        value["task_id"],
        output.parent,
        field="capstone_execution_ready_score",
        expected=value["capstone_execution_ready_score"],
    )
    if final_readers["gate_sha256"] != digest or final_readers["document_sha256"] != digest:
        raise ValueError("published_reader_drift")
    atomic_json(
        sidecar,
        dict(
            passed=True,
            **publication,
            receipts=receipts,
            private_readers=selected,
            final_readers=final_readers,
        ),
    )


def qualify(root: Path, active: Path, design: Path, date: str, output: Path) -> int:
    """Execute each frozen check once and retain unrelated repository-health failures."""
    started = time.monotonic_ns()
    started_at = datetime.now(UTC).isoformat()
    cap.progress("qualification_start")
    with tempfile.TemporaryDirectory(prefix="carnot-8004-") as directory:
        private = Path(directory)
        cap.prepare_fixture(private / "fixture")
        cap.prepare_fixture(private / "blocked_fixture")
        frozen = manifest(private)
        durable = output.parent / "raw" / output.stem / canonical_hash(frozen)[7:]
        manifest_path = durable / "validation_command_manifest.json"
        atomic_json(manifest_path, frozen)
        value = cap.build(root, active, design, date, durable / "authority")
        value["phase_spans"][0]["end_s"] = (time.monotonic_ns() - started) / 1e9
        value["validation_command_manifest_path"] = str(manifest_path)
        receipts = [
            run_check(cap.ROOT, spec, private / spec["name"], durable / "logs")
            for spec in frozen["commands"]
        ]
        publication = json.loads(Path(receipts[0]["log_path"]).read_text())
        value.update(
            publication_gate_results=publication,
            paper_ready=publication["paper_ready"],
            unmet_gates=publication["unmet_gates"],
            **{f"g{i}": publication["gates"][f"G{i}"]["pass"] for i in range(1, 5)},
        )
        report = private / "coverage.json"
        counts = json.loads(report.read_text())["files"] if report.is_file() else {}
        complete = coverage_complete(report, includes=cap.OWNED)
        receipts.append(
            dict(
                name="nonempty_100_percent_changed_statements",
                passed=complete,
                actual_exit=int(not complete),
                expected_exit=0,
                classification="required",
                argv=["coverage_json_reduction", str(report)],
            )
        )
        atomic_json(durable / "coverage_statement_counts.json", counts)
        apply_checks(value, receipts, {k: v["summary"] for k, v in counts.items()})
        ended = time.monotonic_ns()
        value.update(
            duration_s=(ended - started) / 1e9,
            duration_scope="frozen configuration, evidence reduction and owned checks; terminal timing retained in sidecar receipts",
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
            started_monotonic_timestamp_ns=started,
            ended_monotonic_timestamp_ns=ended,
            started_monotonic_ns=started,
            ended_monotonic_ns=ended,
            scratch_root_receipt=dict(
                path=str(private), outside_checkout=True, cleaned_on_exit=True
            ),
        )
        value["phase_spans"].append(
            dict(
                phase="owned_validation",
                start_s=value["phase_spans"][0]["end_s"],
                end_s=value["duration_s"],
                completed_units=len(receipts),
            )
        )
        atomic_json(
            durable / "primitive_rows.json",
            dict(
                rows=value["rows"], independent_reduction_rows=value["independent_reduction_rows"]
            ),
        )
        value["field_principles"].update(
            {
                k: "Retain actual current receipts; historical paper readiness supplies no new science."
                for k in value
            }
        )
        publish(value, output, private / "terminal", durable / "terminal")
        if root == cap.ROOT:
            cap.append_retirements(root, value["retirement_rows"])
    cap.progress("published_checked_primary", 13)
    return 0
