"""Reuse bounded validation tools without copying a historical dispatcher.

REQ-REPORT-7927-V687. Private scratch protects historical results from fixtures.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from carnot.reporting import v687_capstone as cap
from carnot.reporting.v686_contract_validation import (
    coverage_complete,
    dependency_hashes,
    run_check,
)

ROOT = cap.ROOT
OWNED = (
    "python/carnot/reporting/v687_capstone.py",
    "python/carnot/reporting/v687_capstone_validation.py",
    "scripts/experiments/experiment_7927_v687_capstone.py",
)
TEST = "tests/python/test_experiment_7927_v687_capstone.py"


def manifest(private: Path) -> dict[str, Any]:
    """Freeze explicit files and routes before observing any validation results."""
    fixture = private / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    for name, target in (("design.md", "design.md"), ("active.yaml", "research-roadmap.yaml")):
        (fixture / target).write_bytes(
            gzip.decompress((ROOT / f"tests/fixtures/v687/{name}.gz").read_bytes())
        )
    py, cov, pytest, ruff, mypy = (
        str(ROOT / ".venv/bin" / name) for name in ("python", "coverage", "pytest", "ruff", "mypy")
    )
    include = "--include=" + ",".join(str(ROOT / name) for name in OWNED)
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    base = ["--date", "20260930", "--root", str(fixture), "--design", str(fixture / "design.md")]
    commands = []

    def add(
        name: str,
        argv: list[str],
        expected: int = 0,
        reason: str | None = None,
        deadline: int = 180,
        classification: str = "required",
    ) -> None:
        commands.append(
            dict(
                name=name,
                argv=argv,
                expected_exit=expected,
                failure_reason=reason,
                deadline_s=deadline,
                classification=classification,
            )
        )

    add("publication_gate", [py, "scripts/publication_gate.py", "--json"], deadline=60)
    add(
        "unit",
        [
            cov,
            "run",
            f"--data-file={private / 'unit.coverage'}",
            include,
            "-m",
            "pytest",
            TEST,
            *common,
            f"--basetemp={private / 'unit-temp'}",
        ],
    )
    consumers = [
        "tests/python/test_experiment_7914_v686_capstone.py",
        "tests/python/test_experiment_7902_v685_capstone.py",
        "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
        "tests/python/test_experiment_7915_v687_contract_methods.py",
    ]
    add("consumers_e2e018", [pytest, *consumers, *common, f"--basetemp={private / 'consumers'}"])
    routes = {
        "success": [*base, "--output", str(private / "success.json"), "--evidence-only"],
        "block": [
            "--date",
            "20260930",
            "--root",
            str(private / "absent"),
            "--output",
            str(private / "block.json"),
            "--evidence-only",
        ],
        "failure": ["--evidence-only"],
        "replay": [*base, "--cold-replay", str(private / "success.json")],
        "terminal": [
            *base,
            "--terminal-recheck",
            str(private / "success.json"),
            "--output",
            str(private / "checked.json"),
        ],
    }
    for name, args in routes.items():
        add(
            "cli_" + name,
            [
                cov,
                "run",
                f"--data-file={private / f'{name}.coverage'}",
                include,
                str(ROOT / OWNED[-1]),
                *args,
            ],
            2 if name == "failure" else 0,
            "--date" if name == "failure" else None,
        )
    for name, test in (
        ("e2e015", "tests/python/test_source_boundary_7852.py"),
        ("e2e017", "tests/python/test_arc_supervisor_delta_7874.py"),
    ):
        add(name, [pytest, test, *common, f"--basetemp={private / name}"])
    historical = "scripts/experiments/experiment_7868_v683_intervention_protocol.py"
    for name, route in (("fixture", "--fixture-e2e"), ("replay", "--cold-replay")):
        add(
            "e2e016_" + name,
            [py, historical, "--date", "20260929", route, str(private / "e2e016.json")],
        )
    add(
        "e2e016_wrong_date",
        [py, historical, "--date", "20260930", "--fixture-e2e", str(private / "wrong-date.json")],
        1,
        "run_date_mismatch",
        60,
    )
    add(
        "coverage_combine",
        [
            cov,
            "combine",
            "--keep",
            f"--data-file={private / 'combined.coverage'}",
            *[str(private / f"{name}.coverage") for name in ("unit", *routes)],
        ],
    )
    add(
        "coverage_report",
        [
            cov,
            "report",
            f"--data-file={private / 'combined.coverage'}",
            include,
            "--show-missing",
            "--fail-under=100",
        ],
    )
    add(
        "coverage_json",
        [
            cov,
            "json",
            f"--data-file={private / 'combined.coverage'}",
            include,
            "-o",
            str(private / "coverage.json"),
        ],
    )
    add("ruff", [ruff, "check", *OWNED, TEST])
    add("format", [ruff, "format", "--check", *OWNED, TEST])
    add("mypy", [mypy, "--strict", *OWNED])
    add("spec", [py, "scripts/check_spec_coverage.py", TEST, *consumers])
    add(
        "repository_full_suite",
        [pytest, "tests/python", "-q"],
        deadline=600,
        classification="diagnostic",
    )
    dependencies = dependency_hashes(ROOT, paths=[*OWNED, TEST, *consumers])
    for name in (
        "pyproject.toml",
        "ops/exclusion_manifest.yaml",
        "openspec/capabilities/research-reporting/spec.md",
        "tests/fixtures/v687/design.md.gz",
        "tests/fixtures/v687/active.yaml.gz",
        "scripts/publication_gate.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
    ):
        dependencies[name] = cap.sha256_file(ROOT / name)
    return dict(
        commands=commands,
        coverage_includes=list(OWNED),
        dependency_hashes=dependencies,
        historical_fixture_date="20260929",
        execution_date="20260930",
        e2e018="Historical publishing inapplicable after rollover; private current and historical assertions execute.",
    )


def publication_result(value: dict[str, Any], receipt: dict[str, Any]) -> None:
    """Invalid output or a failed command cannot invent a passing publication gate."""
    if not receipt["passed"]:
        return
    try:
        parsed = json.loads(Path(receipt["log_path"]).read_text())
        if not isinstance(parsed, dict) or any(
            type(parsed["gates"][key]["pass"]) is not bool for key in cap.GATES
        ):
            raise ValueError("invalid_gates")
    except (ValueError, KeyError, TypeError):
        receipt.update(passed=False, failure_reason="invalid_publication_output")
        return
    value.update(publication_gate_results=parsed, **cap.publication_operands(parsed))


def terminal(value: dict[str, Any], output: Path, private: Path, durable: Path) -> None:
    """A changed verdict is checked again so sidecars describe the published bytes."""
    private.mkdir(parents=True, exist_ok=True)
    sidecar = durable / "terminal-validation.json"
    value["terminal_validation_sidecar_path"] = str(sidecar)
    value["field_principles"]["terminal_validation_sidecar_path"] = (
        "The candidate hash binds actual validator reports to final bytes."
    )
    candidate = private / "candidate.json"
    for attempt in range(3):
        cap.atomic_json(candidate, value)
        commands = [
            dict(
                name=name,
                argv=[str(ROOT / ".venv/bin/python"), script, flag, str(candidate)],
                expected_exit=0,
                deadline_s=60,
                classification="terminal",
            )
            for name, script, flag in (
                ("adversarial_verify", "scripts/adversarial_verify.py", "--json"),
                ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
            )
        ]
        receipts = [
            run_check(ROOT, spec, private / f"terminal-{attempt}", durable) for spec in commands
        ]
        try:
            report = json.loads(Path(receipts[0]["log_path"]).read_text())
            if not isinstance(report, dict) or type(report["flagged_count"]) is not int:
                raise ValueError("invalid_validator_report")
        except (ValueError, TypeError, KeyError):
            report = dict(flagged_count=None, parse_error="invalid_validator_report")
            receipts[0]["passed"] = False
        flagged = bool(report.get("flagged_count", 0))
        failed = flagged or any(not row["passed"] for row in receipts)
        cap.atomic_json(
            sidecar,
            dict(
                candidate_sha256=cap.sha256_file(candidate),
                adversarial_report=report,
                strict_row_report=Path(receipts[1]["log_path"]).read_text(),
                validator_receipts=receipts,
            ),
        )
        if failed and attempt == 0:
            value.update(
                honest_verdict="complete_disqualified_terminal_verification",
                verdict_class="disqualified",
                capstone_execution_ready_score=0,
                flagged_adversarial=flagged,
            )
            value["acceptance_gate_results"].update(validity=False, readiness=0)
            value["gate_check_summary"].append(
                cap.shared.operand(
                    candidate,
                    cap.sha256_file(candidate),
                    "exp7927-owned-validation",
                    "terminal_validators",
                    "unflagged and passing",
                    report,
                )
            )
            continue
        if value["flagged_adversarial"] != flagged:
            value["flagged_adversarial"] = flagged
            continue
        break
    else:
        raise ValueError("terminal_flags_unstable")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".checked")
    shutil.copyfile(candidate, temporary)
    temporary.replace(output)


def qualify(root: Path, design: Path, active: Path, date: str, output: Path) -> int:
    """Freeze commands once and qualify current audit work without retrying old science."""
    started = time.monotonic()
    cap.progress(started, "qualification_start", 0)
    private = Path(tempfile.mkdtemp(prefix="carnot-7927-validation-"))
    frozen = manifest(private)
    durable = root / "results/raw/experiment_7927_v687_capstone" / cap.canonical_hash(frozen)[7:]
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
    receipts = [run_check(ROOT, spec, private, durable / "logs") for spec in frozen["commands"]]
    for row in receipts:
        if row.get("failure_reason") and row["expected_exit"] != 0:
            row["passed"] = (
                row["passed"] and row["failure_reason"] in Path(row["log_path"]).read_text()
            )
        if row["name"] == "publication_gate":
            publication_result(value, row)
    report = private / "coverage.json"
    counts = json.loads(report.read_text())["files"] if report.is_file() else {}
    value["coverage_statement_counts"] = {name: item["summary"] for name, item in counts.items()}
    complete = coverage_complete(report, includes=list(OWNED))
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
        value["gate_check_summary"].append(
            cap.shared.operand(
                Path(row.get("log_path", str(report))),
                row.get("log_sha256"),
                "exp7927-owned-validation",
                row["name"],
                row["expected_exit"],
                row["actual_exit"],
            )
        )
    if failures:
        value.update(
            honest_verdict="complete_disqualified_required_validation",
            verdict_class="disqualified",
            capstone_execution_ready_score=0,
        )
        value["acceptance_gate_results"].update(validity=False, readiness=0)
    else:
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
    cap.atomic_json(durable / "primitive-rows.json", value["independent_reduction_rows"])
    replay_errors = cap.cold_replay(value, root, design, active)
    if replay_errors:
        value.update(
            honest_verdict="complete_disqualified_cold_replay",
            verdict_class="disqualified",
            capstone_execution_ready_score=0,
        )
        value["acceptance_gate_results"].update(validity=False, readiness=0)
        value["gate_check_summary"].append(
            cap.shared.operand(
                durable / "primitive-rows.json",
                cap.sha256_file(durable / "primitive-rows.json"),
                "exp7927-owned-validation",
                "cold_replay",
                [],
                replay_errors,
            )
        )
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
    value["field_principles"].update(
        {
            key: "Seal exact evidence bytes after owned validation; external science remains separate."
            for key in value
            if key not in value["field_principles"]
        }
    )
    terminal(value, output, private / "terminal", durable / "terminal")
    final = json.loads(output.read_bytes())
    report_path = root / final["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# V687 capstone\n\nVerdict: "
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
        + "\n\nAudit readiness is separate from scientific utility and FoVer publication readiness. GAP-ORACLE-DISTINCT remains open after the September 28 corrigendum. DiffusionGemma remains pending. Historical failures remain preserved. Historical E2E-016 uses 20260929; execution uses 20260930. No model was loaded by this capstone.\n"
    )
    cap.progress(started, "published_checked_bytes", 13)
    return 0
