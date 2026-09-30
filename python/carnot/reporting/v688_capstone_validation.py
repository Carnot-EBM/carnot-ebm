"""REQ-REPORT-7939-V688: reuse bounded commands and the primary publication layout."""

import gzip
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import yaml

from carnot.reporting import v687_capstone_validation as prior
from carnot.reporting import v688_capstone as cap
from carnot.reporting.primary_publication import publish_primary, reader_receipt

ROOT = cap.ROOT
OWNED = (
    "python/carnot/reporting/v688_capstone.py",
    "python/carnot/reporting/v688_capstone_validation.py",
    "scripts/experiments/experiment_7939_v688_capstone.py",
)
TEST = "tests/python/test_experiment_7939_v688_capstone.py"
run_check = prior.run_check
publication_result = prior.publication_result


def manifest(private: Path) -> dict[str, Any]:
    """Adapt the qualified command plan without copying its historical dispatcher."""
    value = prior.manifest(private)
    for name, target in (("design.md", "design.md"), ("active.yaml", "research-roadmap.yaml")):
        (private / "fixture" / target).write_bytes(
            gzip.decompress((ROOT / f"tests/fixtures/v688/{name}.gz").read_bytes())
        )
    before = "--include=" + ",".join(str(ROOT / name) for name in prior.OWNED)
    after = "--include=" + ",".join(str(ROOT / name) for name in OWNED)
    tasks = yaml.safe_load((private / "fixture/research-roadmap.yaml").read_bytes())["tasks"]
    for task in tasks[:-1]:
        gates = {
            g["artifact_field"]: g["value"]
            for other in tasks
            for g in other.get("gated_on", [])
            if g["upstream"] == task["id"] and g["op"] == "=="
        }
        cap.atomic_json(
            private / "fixture" / task["deliverable"],
            {
                "experiment_id": int(task["id"][3:7]),
                "task_id": task["id"],
                "milestone": "2026.09.688",
                "run_date": "20260930",
                "MODEL_SPECS": task["MODEL_SPECS"],
                "honest_verdict": "complete_null_fixture",
                "verdict_class": "null",
                "flagged_adversarial": False,
                "rows": [],
                **gates,
            },
        )
    for row in value["commands"]:
        row["argv"] = [
            arg.replace(before, after)
            .replace(prior.TEST, TEST)
            .replace(prior.OWNED[-1], OWNED[-1])
            .replace("success.json", "success/experiment_7939_capstone.json")
            .replace("block.json", "block/experiment_7939_capstone.json")
            .replace("checked.json", "checked/experiment_7939_capstone.json")
            for arg in row["argv"]
        ]
        if row["name"] in {"ruff", "format", "mypy"}:
            row["argv"] = row["argv"][: 3 if row["name"] == "format" else 2] + list(OWNED)
            if row["name"] != "mypy":
                row["argv"].append(TEST)
        if row["name"] == "consumers_e2e018":
            row["argv"].insert(1, prior.TEST)
            row["argv"][1:1] = [
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_conductor_gates.py",
                "tests/python/test_in_process_doc_reconcile.py",
            ]
    value.update(
        coverage_includes=list(OWNED),
        dependency_hashes=prior.dependency_hashes(ROOT, paths=[*OWNED, TEST, prior.TEST]),
    )
    value["dependency_hashes"].update(
        {
            name: cap.sha256_file(ROOT / name)
            for name in (
                "pyproject.toml",
                "ops/exclusion_manifest.yaml",
                "openspec/capabilities/research-reporting/spec.md",
                "tests/fixtures/v688/design.md.gz",
                "tests/fixtures/v688/active.yaml.gz",
            )
        }
    )
    return value


def disqualify(
    value: dict[str, Any], reason: str, path: Path, expected: Any, observed: Any
) -> None:
    """A required owned failure invalidates readiness even when external science is absent."""
    value.update(
        honest_verdict="complete_disqualified_" + reason,
        verdict_class="disqualified",
        capstone_execution_ready_score=0,
    )
    value["acceptance_gate_results"].update(validity=False, readiness=0)
    value["gate_check_summary"].append(
        cap.prior.shared.operand(
            path,
            cap.sha256_file(path) if path.is_file() else None,
            "exp7939-owned-validation",
            reason,
            expected,
            observed,
        )
    )


def terminal(value: dict[str, Any], output: Path, private: Path, durable: Path) -> None:
    """Check final bytes after verdict changes, then publish a single locked primary."""
    sidecar = durable / "terminal-validation.json"
    value["terminal_validation_sidecar_path"] = str(sidecar)
    value["field_principles"]["terminal_validation_sidecar_path"] = (
        "Bind actual terminal reports to the final candidate hash."
    )
    candidate = private / "candidate.json"
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
        try:
            report = json.loads(Path(receipts[0]["log_path"]).read_text())
            if not isinstance(report, dict) or type(report["flagged_count"]) is not int:
                raise ValueError("invalid_validator_report")
        except (ValueError, KeyError, TypeError):
            report = dict(flagged_count=None, parse_error="invalid_validator_report")
            receipts[0]["passed"] = False
        flagged = bool(report.get("flagged_count", 0))
        failed = flagged or any(not r["passed"] for r in receipts)
        digest = cap.sha256_file(candidate)
        cap.atomic_json(
            sidecar,
            dict(
                candidate_sha256=digest,
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
        "exp7939-capstone",
        output.parent,
        field="capstone_execution_ready_score",
        expected=value["capstone_execution_ready_score"],
    )
    if reading["gate_sha256"] != digest or reading["document_sha256"] != digest:
        raise ValueError("primary_reader_drift")
    cap.atomic_json(Path(value["primary_resolution_receipt"]["path"]), reading)


def qualify(root: Path, design: Path, active: Path, date: str, output: Path) -> int:
    """Freeze checks before results and distinguish owned failures from repository debt."""
    started = time.monotonic()
    cap.progress(started, "qualification_start", 0)
    private = Path(tempfile.mkdtemp(prefix="carnot-7939-validation-"))
    frozen = manifest(private)
    durable = root / "results/raw/experiment_7939_v688_capstone" / cap.canonical_hash(frozen)[7:]
    manifest_path = durable / "validation-command-manifest.json"
    cap.atomic_json(manifest_path, frozen)
    value = cap.build_candidate(root, design, active, date, snapshots=durable / "authority")
    cap.atomic_json(
        durable / "checkpoint.json",
        dict(
            status="inputs_and_commands_frozen",
            manifest_sha256=cap.sha256_file(manifest_path),
            input_sha256=cap.canonical_hash(value["source_artifact_hashes"]),
        ),
    )
    receipts = []
    for spec in frozen["commands"]:
        row = run_check(ROOT, spec, private, durable / "logs")
        if row.get("failure_reason") and row["expected_exit"] != 0:
            row["passed"] = (
                row["passed"] and row["failure_reason"] in Path(row["log_path"]).read_text()
            )
        if row["name"] == "publication_gate":
            publication_result(value, row)
        receipts.append(row)
    report = private / "coverage.json"
    counts = json.loads(report.read_text())["files"] if report.is_file() else {}
    value["coverage_statement_counts"] = {name: item["summary"] for name, item in counts.items()}
    complete = prior.coverage_complete(report, includes=list(OWNED))
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
    value.update(
        validation_receipts=receipts,
        validation_command_manifest_path=str(manifest_path),
        observed_child_commands=[r["argv"] for r in receipts],
    )
    value["repository_health"]["current_full_suite"] = [
        r for r in receipts if r["classification"] == "diagnostic"
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
    value["resolved_imports"]["carnot.reporting.v688_capstone_validation"] = str(
        Path(__file__).resolve()
    )
    cap.atomic_json(
        durable / "primitive-reductions.json", dict(rows=value["independent_reduction_rows"])
    )
    replay_errors = cap.cold_replay(value, root, design, active)
    if replay_errors:
        disqualify(value, "cold_replay", durable / "primitive-reductions.json", [], replay_errors)
    value["ended_monotonic_timestamp_ns"] = time.monotonic_ns()
    value["duration_s"] = (
        value["ended_monotonic_timestamp_ns"] - value["started_monotonic_timestamp_ns"]
    ) / 1e9
    value["phase_spans"].append(
        dict(
            phase="owned_validation",
            start_s=value["phase_spans"][0]["end_s"],
            end_s=value["duration_s"],
            completed_units=len(receipts),
        )
    )
    terminal(value, output, private / "terminal", durable / "terminal")
    final = json.loads(output.read_bytes())
    report_path = root / final["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# V688 capstone\n\nVerdict: "
        + final["honest_verdict"]
        + ".\n\n"
        + "\n".join(
            f"- {r['task_id']}: {r['status']} ({r['path']})." for r in final["outcome_rows"]
        )
        + "\n\n"
        + "\n".join(
            f"- {gap}: {d['decision']}. Continue: {d['continue_if']} Retire: {d['retire_if']}"
            for gap, d in final["gap_decisions"].items()
        )
        + "\n\nAudit readiness is separate from scientific benefit and FoVer publication readiness. GAP-ORACLE-DISTINCT remains open after the September 28 correction. DiffusionGemma remains pending. Historical required failures remain preserved. Historical E2E-016 uses 20260929 on both routes; current execution uses 20260930. No model was loaded. Ops and traceability reconciliation belongs to the conductor.\n"
    )
    cap.progress(started, "published_checked_bytes", 12)
    return 0
