"""REQ-VERIFY-7991-V692: keep owned checks bounded and exact negative exits honest.

The shared child supervisor emits heartbeats and archives logs after child exit.
Historical repository failures remain diagnostics outside this qualification.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from carnot.reporting import v692_capstone as cap
from carnot.reporting.v686_contract_validation import run_check, coverage_complete
from carnot.reporting.primary_publication import publish_primary, reader_receipt

ROOT = cap.ROOT
OWNED = [
    "python/carnot/reporting/v692_capstone.py",
    "python/carnot/reporting/v692_capstone_reduction.py",
    "python/carnot/reporting/v692_capstone_validation.py",
    "scripts/experiments/experiment_7991_v692_capstone.py",
]
TEST = "tests/python/test_experiment_7991_v692_capstone.py"
CONSUMERS = [
    "tests/python/test_experiment_7979_v692_contract_methods.py",
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_conductor_gates.py",
    "tests/python/test_in_process_doc_reconcile.py",
]
Json = dict[str, Any]


def manifest(private: Path) -> Json:
    """Freeze exact files and private CLI coverage before any source reduction."""
    py, cov, pytest, ruff, mypy = (
        str(ROOT / ".venv/bin" / n) for n in ("python", "coverage", "pytest", "ruff", "mypy")
    )
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    commands = []

    def add(name: str, argv: list[str], deadline: int = 300) -> None:
        commands.append(
            dict(
                name=name,
                argv=argv,
                expected_exit=0,
                deadline_s=deadline,
                classification="required",
            )
        )

    add("publication_gate", [py, "scripts/publication_gate.py", "--json"], 60)
    add(
        "owned_unit_and_real_private_cli",
        [
            "/usr/bin/env",
            f"CARNOT_7991_COVERAGE_DIR={private}",
            cov,
            "run",
            "--parallel-mode",
            f"--data-file={private / 'owned.coverage'}",
            "--include=" + ",".join(str(ROOT / n) for n in OWNED),
            "-m",
            "pytest",
            TEST,
            *common,
            f"--basetemp={private / 'unit'}",
        ],
    )
    add(
        "affected_consumers_e2e018",
        [pytest, *CONSUMERS, *common, f"--basetemp={private / 'consumers'}"],
    )
    add(
        "coverage_combine",
        [cov, "combine", f"--data-file={private / 'owned.coverage'}", str(private)],
        60,
    )
    add(
        "coverage_json",
        [
            cov,
            "json",
            f"--data-file={private / 'owned.coverage'}",
            "-o",
            str(private / "coverage.json"),
        ],
        60,
    )
    add(
        "changed_statement_coverage",
        [cov, "report", f"--data-file={private / 'owned.coverage'}", "--fail-under=100"],
        60,
    )
    add("ruff_check", [ruff, "check", *OWNED, TEST], 60)
    add("ruff_format", [ruff, "format", "--check", *OWNED, TEST], 60)
    add("strict_mypy", [mypy, "--strict", "--follow-imports=silent", *OWNED], 60)
    add("spec_coverage", [py, "scripts/check_spec_coverage.py", TEST, *CONSUMERS], 60)
    return dict(
        commands=commands,
        coverage_includes=OWNED,
        execution_date="20261001",
        scratch_root=str(private),
        dependency_hashes={n: cap.sha256_file(ROOT / n) for n in [*OWNED, TEST, *CONSUMERS]},
        required_negative_routes=[
            "missing_upstream",
            "authority_drift",
            "malformed_rows",
            "aggregate_drift",
        ],
        legitimate_blocked_cold_replay_expected_exit=0,
        negative_wrapper_expected_exit=0,
        full_suite_policy="retained historical diagnostic receipts; no unchanged rerun",
    )


def disqualify(value: Json, reason: str, receipt: Json) -> None:
    """Owned failures always remove readiness, even when science was already blocked."""
    value.update(
        verdict_class="disqualified",
        honest_verdict="complete_disqualified_" + reason,
        capstone_execution_ready_score=0,
        science_ready=False,
    )
    value["acceptance_gate_results"].update(validity=False, readiness=0)
    value["gate_check_summary"].append(
        cap.reduction.operand(
            Path(receipt.get("log_path", "/missing")),
            "exp7991-owned-validation",
            receipt["name"],
            receipt.get("expected_exit", 0),
            receipt["actual_exit"],
        )
    )


def terminal(value: Json, output: Path, private: Path, durable: Path) -> None:
    """Check exactly the prospective final bytes and retain both consumer identities."""
    sidecar = durable / "terminal-validation.json"
    value["terminal_validation_sidecar_path"] = str(sidecar)
    candidate = private / "experiment_7991_v692_capstone.json"
    for attempt in range(2):
        cap.atomic_json(candidate, value)
        specs = [
            dict(
                name=name,
                argv=[str(ROOT / ".venv/bin/python"), str(ROOT / script), flag, str(candidate)],
                expected_exit=0,
                deadline_s=60,
                classification="required",
            )
            for name, script, flag in (
                ("adversarial_verify", "scripts/adversarial_verify.py", "--json"),
                ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
            )
        ]
        specs.append(
            dict(
                name="terminal_cold_replay",
                argv=[
                    "/usr/bin/env",
                    "-u",
                    "PYTHONPATH",
                    str(ROOT / ".venv/bin/python"),
                    str(ROOT / OWNED[-1]),
                    "--date",
                    "20261001",
                    "--root",
                    value["input_root"],
                    "--design",
                    value["authority_snapshots"]["design"]["source_path"],
                    "--active",
                    value["authority_snapshots"]["active"]["source_path"],
                    "--cold-replay",
                    str(candidate),
                ],
                expected_exit=0,
                deadline_s=180,
                classification="required",
            )
        )
        receipts = [
            run_check(
                private, spec, private / f"attempt-{attempt}-{spec['name']}", durable / "logs"
            )
            for spec in specs
        ]
        try:
            report = json.loads(Path(receipts[0]["log_path"]).read_text())
        except ValueError:
            report = dict(flagged_count=None, parse_error=True)
            receipts[0]["passed"] = False
        flagged = report.get("flagged_count") != 0
        failures = [r for r in receipts if not r["passed"]]
        if flagged or failures:
            if attempt == 0:
                receipt = (
                    failures[0]
                    if failures
                    else dict(
                        name="adversarial_flag",
                        actual_exit=1,
                        log_path=receipts[0]["log_path"],
                        passed=False,
                        classification="required",
                    )
                )
                value["validation_receipts"].append(receipt)
                disqualify(value, "terminal_validation", receipt)
                value["flagged_adversarial"] = flagged
                continue
            raise ValueError("terminal_validation_failed")
        digest = cap.sha256_file(candidate)
        cap.atomic_json(
            sidecar,
            dict(
                candidate_sha256=digest,
                primary_path=str(output),
                validator_receipts=receipts,
                passed=True,
            ),
        )
        publish_primary(
            output,
            value,
            lambda p: dict(passed=cap.sha256_file(p) == digest, terminal_sidecar=str(sidecar)),
        )
        selected = reader_receipt(
            "exp7991-capstone",
            output.parent,
            field="capstone_execution_ready_score",
            expected=value["capstone_execution_ready_score"],
        )
        if (
            not selected["passed"]
            or selected["gate_sha256"] != digest
            or selected["document_sha256"] != digest
        ):
            raise ValueError("primary_reader_drift")
        cap.atomic_json(Path(value["primary_resolution_receipt"]["path"]), selected)
        return


def qualify(root: Path, design: Path, active: Path, date: str, output: Path) -> int:
    """Run the frozen checks once, preserving all source results and failed receipts."""
    started = time.monotonic_ns()
    cap.progress("owned_qualification_start")
    with tempfile.TemporaryDirectory(prefix="carnot-7991-") as directory:
        private = Path(directory)
        frozen = manifest(private)
        durable = output.parent / "raw" / output.stem / cap.canonical_hash(frozen)[7:]
        manifest_path = durable / "validation-command-manifest.json"
        cap.atomic_json(manifest_path, frozen)
        value = cap.build_candidate(root, design, active, date, snapshots=durable / "authority")
        value.update(
            input_root=str(root),
            validation_command_manifest_path=str(manifest_path),
            scratch_root_receipt=dict(
                path=str(private), private=True, outside_checkout=True, cleaned_on_exit=True
            ),
        )
        receipts = [
            run_check(ROOT, spec, private / spec["name"], durable / "logs")
            for spec in frozen["commands"]
        ]
        for source in sorted((private / "cli-receipts").glob("*.json")):
            target = durable / "cli-receipts" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            receipts.append(
                dict(
                    json.loads(target.read_bytes()),
                    log_path=str(target),
                    log_sha256=cap.sha256_file(target),
                )
            )
        publication = json.loads(Path(receipts[0]["log_path"]).read_text())
        value.update(
            publication_gate_results=publication,
            **cap.prior.prior.publication_operands(publication),
        )
        complete = coverage_complete(private / "coverage.json", includes=OWNED)
        receipts.append(
            dict(
                name="nonempty_per_file_100_percent_coverage",
                actual_exit=int(not complete),
                expected_exit=0,
                passed=complete,
                classification="required",
            )
        )
        if (private / "coverage.json").is_file():
            shutil.copyfile(private / "coverage.json", durable / "coverage.json")
            value["coverage_statement_counts"] = {
                k: v["summary"]
                for k, v in json.loads((private / "coverage.json").read_text())["files"].items()
            }
        value["validation_receipts"] = receipts
        for receipt in receipts:
            if not receipt["passed"]:
                disqualify(value, "required_validation", receipt)
        if all(r["passed"] for r in receipts):
            value["acceptance_gate_results"].update(
                validity=True, readiness=int(value["science_ready"])
            )
            value["capstone_execution_ready_score"] = int(value["science_ready"])
        value.update(
            observed_child_commands=[r["argv"] for r in receipts if "argv" in r],
            duration_s=(time.monotonic_ns() - started) / 1e9,
            finished_at=datetime.now(UTC).isoformat(),
            started_monotonic_timestamp_ns=started,
            ended_monotonic_timestamp_ns=time.monotonic_ns(),
        )
        value["phase_spans"].append(
            dict(
                phase="owned_validation",
                start_s=value["phase_spans"][0]["end_s"],
                end_s=value["duration_s"],
                completed_units=len(receipts),
            )
        )
        value["field_principles"].update(
            {k: "Retain observed owned checks and exact frozen evidence." for k in value}
        )
        terminal(value, output, private / "terminal", durable / "terminal")
    cap.progress("published_checked_primary", 13)
    return 0
