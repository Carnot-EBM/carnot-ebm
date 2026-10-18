#!/usr/bin/env python3
"""Run the V685 capstone from exact authority and producer paths.

REQ-REPORT-7902-V685. Current validation is recorded separately from old debt.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file  # noqa: E402
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands  # noqa: E402
from carnot.reporting.v685_capstone import build_candidate, cold_replay  # noqa: E402


MODEL_SPECS: list[str] = []
OWNED = (
    "python/carnot/reporting/v685_capstone.py",
    "scripts/experiments/experiment_7902_v685_capstone.py",
)
TESTS = (
    "tests/python/test_experiment_7902_v685_capstone.py",
    "tests/python/test_experiment_7890_v684_capstone.py",
    "tests/python/test_experiment_7877_v683_independent_audit.py",
    "tests/python/test_experiment_7878_v683_capstone.py",
)
START = time.monotonic()


def progress(phase: str, completed: int) -> None:
    """A monotonic timestamp makes long child work observable."""
    print(
        f"[exp7902] {phase} elapsed_s={time.monotonic() - START:.3f} completed_units={completed}",
        flush=True,
    )


def _private_fixture(directory: Path) -> tuple[Path, Path]:
    """Copy frozen V685 bytes so validation cannot edit live authority."""
    directory.mkdir(parents=True, exist_ok=True)
    design = directory / "design.md"
    active = directory / "active.yaml"
    design.write_bytes((ROOT / "tests/fixtures/v685/design.md").read_bytes())
    active.write_bytes(gzip.decompress((ROOT / "tests/fixtures/v685/active.yaml.gz").read_bytes()))
    (directory / "results").mkdir(exist_ok=True)
    return design, active


def commands(private: Path) -> list[CommandSpec]:
    """Freeze exact affected argv and coverage files before observations."""
    for name in (*OWNED, *TESTS):
        if not (ROOT / name).is_file():
            raise ValueError(f"affected_scope_missing:{name}")
    design, active = _private_fixture(private / "fixture")
    py = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    cov = str(ROOT / ".venv/bin/coverage")
    ruff = str(ROOT / ".venv/bin/ruff")
    mypy = str(ROOT / ".venv/bin/mypy")
    include = ",".join(str(ROOT / file) for file in OWNED)
    script = str(ROOT / OWNED[1])
    success = private / "fixture/success.json"
    fixture_args = (
        "--date",
        "20260929",
        "--root",
        str(private / "fixture"),
        "--design",
        str(design),
        "--active",
        str(active),
    )
    pytest_args = ("-n", "0", "-o", "addopts=", "--no-cov", "-q")
    e2e_file = private / "e2e016.json"
    raw: list[tuple[str, tuple[str, ...], str, float]] = [
        ("publication_gate", (py, "scripts/publication_gate.py", "--json"), "required", 60),
        ("affected_pytest", (pytest, *TESTS, *pytest_args), "required", 300),
        (
            "unit_coverage",
            (
                cov,
                "run",
                f"--data-file={private / 'unit.coverage'}",
                f"--include={include}",
                "-m",
                "pytest",
                TESTS[0],
                *pytest_args,
            ),
            "required",
            180,
        ),
        (
            "cli_success",
            (
                cov,
                "run",
                f"--data-file={private / 'success.coverage'}",
                f"--include={include}",
                script,
                *fixture_args,
                "--output",
                str(success),
                "--evidence-only",
            ),
            "required",
            60,
        ),
        (
            "cli_missing_date",
            (
                cov,
                "run",
                f"--data-file={private / 'failure.coverage'}",
                f"--include={include}",
                script,
                "--root",
                str(private / "fixture"),
                "--output",
                str(private / "failure.json"),
                "--evidence-only",
            ),
            "expected_failure",
            60,
        ),
        (
            "cli_cold_replay",
            (
                cov,
                "run",
                f"--data-file={private / 'replay.coverage'}",
                f"--include={include}",
                script,
                *fixture_args,
                "--cold-replay",
                str(success),
            ),
            "required",
            60,
        ),
        (
            "e2e015",
            (pytest, "tests/python/test_source_boundary_7852.py", *pytest_args),
            "required",
            180,
        ),
        (
            "e2e016_fixture",
            (
                py,
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--fixture-e2e",
                str(e2e_file),
            ),
            "required",
            180,
        ),
        (
            "e2e016_replay",
            (
                py,
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--cold-replay",
                str(e2e_file),
            ),
            "required",
            180,
        ),
        (
            "e2e017",
            (pytest, "tests/python/test_arc_supervisor_delta_7874.py", *pytest_args),
            "required",
            180,
        ),
        ("ruff_check", (ruff, "check", *OWNED, TESTS[0]), "required", 60),
        ("ruff_format", (ruff, "format", "--check", *OWNED, TESTS[0]), "required", 60),
        ("mypy_strict", (mypy, "--strict", *OWNED), "required", 120),
        ("spec_coverage", (py, "scripts/check_spec_coverage.py", TESTS[0]), "required", 120),
        (
            "coverage_combine",
            (
                cov,
                "combine",
                "--keep",
                f"--data-file={private / 'combined.coverage'}",
                str(private / "unit.coverage"),
                str(private / "success.coverage"),
                str(private / "failure.coverage"),
                str(private / "replay.coverage"),
            ),
            "required",
            60,
        ),
        (
            "coverage_report",
            (
                cov,
                "report",
                f"--data-file={private / 'combined.coverage'}",
                f"--include={include}",
                "--show-missing",
                "--fail-under=100",
            ),
            "required",
            60,
        ),
    ]
    return [CommandSpec(name, argv, scope, deadline) for name, argv, scope, deadline in raw]


def _report(value: dict[str, Any]) -> str:
    """State decisions plainly without upgrading fixtures to natural evidence."""
    lines = [
        "# V685 capstone: independent evidence reduction",
        "",
        f"Verdict: {value['honest_verdict']}.",
        "",
        "The administrative reduction is complete. Missing or disqualified",
        "science leaves product benefit unmeasured. The qualified source",
        "boundary is development evidence, not an independent benefit.",
        "",
        "## Twelve outcomes",
        "",
    ]
    for row in value["outcome_rows"]:
        lines.append(f"- {row['upstream_id']}: {row['status']} ({row['path']}).")
    lines += ["", "## Product gaps", ""]
    for name, decision in value["gap_decisions"].items():
        lines.append(f"- {name}: {decision['decision']}.")
    lines += [
        "",
        "## Publication gate",
        "",
        f"G1={value['G1']}, G2={value['G2']}, G3={value['G3']}, G4={value['G4']}.",
        f"Paper ready: {value['paper_ready']}.",
        "",
        "The next science roadmap waits for the owned source and validation",
        "repairs. Historical failure scopes remain in the JSON ledger.",
        "",
    ]
    return "\n".join(lines)


def _terminal(root: Path, candidate: Path, private: Path) -> tuple[list[dict[str, Any]], bool]:
    """Seal validator output after each child exits and read its real flags."""
    py = str(root / ".venv/bin/python")
    checks = [
        CommandSpec(
            "adversarial_verify",
            (py, "scripts/adversarial_verify.py", "--json", str(candidate)),
            "terminal",
            60,
        ),
        CommandSpec(
            "strict_rows",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal",
            60,
        ),
    ]
    receipts = run_commands(root, checks, log_dir=private / "terminal_logs", heartbeat_s=30)
    for item in receipts:
        item["classification"] = "terminal"
    try:
        adversarial = json.loads((root / receipts[0]["log_path"]).read_text())
    except (OSError, ValueError):
        adversarial = {"flagged_count": 1, "error": "validator_report_unreadable"}
    flagged = bool(adversarial.get("flagged_count", 0)) or not all(r["passed"] for r in receipts)
    sidecar = private / f"terminal-{sha256_file(candidate)[7:]}.json"
    atomic_json(
        sidecar,
        {
            "candidate_path": str(candidate),
            "candidate_sha256": sha256_file(candidate),
            "adversarial_report": adversarial,
            "validator_receipts": receipts,
        },
    )
    return receipts, flagged


def run_current(root: Path, design: Path, active: Path, date: str, output: Path) -> dict[str, Any]:
    """Run frozen checks and publish only the candidate those checks saw."""
    progress("preconditions", 0)
    identity = canonical_hash(
        {
            "design": sha256_file(design),
            "active": sha256_file(active),
            "code": [sha256_file(ROOT / name) for name in OWNED],
            "run": time.monotonic_ns(),
        }
    )
    private = root / "results/raw/experiment_7902_v685_capstone" / f"attempt-{identity}"
    private.mkdir(parents=True)
    manifest = commands(private)
    manifest_path = private / "validation_command_manifest.json"
    atomic_json(
        manifest_path,
        {
            "changed_files": list(OWNED),
            "affected_tests": list(TESTS),
            "source_hashes": {name: sha256_file(ROOT / name) for name in (*OWNED, *TESTS)},
            "coverage_includes": [str(ROOT / name) for name in OWNED],
            "applicable_e2e": ["E2E-015", "E2E-016", "E2E-017"],
            "inapplicable_e2e": [f"E2E-{n:03d}" for n in range(1, 15)],
            "commands": [
                {
                    "name": c.name,
                    "argv": list(c.argv),
                    "classification": c.scope,
                    "expected_exit": 2 if c.name == "cli_missing_date" else 0,
                    "expected_error": "--date" if c.name == "cli_missing_date" else None,
                    "deadline_s": c.timeout_s,
                }
                for c in manifest
            ],
        },
    )
    progress("before_validation", 0)
    receipts = run_commands(root, manifest, log_dir=private / "logs", heartbeat_s=30)
    for item in receipts:
        item["classification"] = item["scope"]
        if item["name"] == "cli_missing_date":
            item["passed"] = (
                item["exit_code"] == 2 and "--date" in item["output_tail"] and not item["timed_out"]
            )
    progress("after_validation", len(receipts))
    publication_log = root / receipts[0]["log_path"]
    try:
        publication = json.loads(publication_log.read_text())
    except (OSError, ValueError):
        publication = {}
    value = build_candidate(root, design, active, date, publication)
    value["validation_receipts"] = receipts
    value["validation_command_manifest_path"] = str(manifest_path)
    value["observed_child_commands"] = [item["command_argv"] for item in receipts]
    failures = [item["name"] for item in receipts if not item["passed"]]
    if failures:
        value["honest_verdict"] = "complete_disqualified_required_checks"
        value["verdict_class"] = "disqualified"
        value["capstone_execution_ready_score"] = 0
        value["acceptance_gate_results"]["validity"] = False
        value["acceptance_gate_results"]["readiness"] = 0
        value["repository_health"]["owned_failed_checks"] = failures
    value["duration_s"] = time.monotonic() - START
    value["phase_spans"].append(
        {
            "phase": "required_validation",
            "duration_s": sum(r["duration_s"] for r in receipts),
            "completed_units": len(receipts),
        }
    )
    value["field_principles"]["terminal_validation_report_paths"] = (
        "Sidecars bind terminal reports to the candidate without self-hashing."
    )
    value["terminal_validation_report_paths"] = []
    report = root / value["report_path"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report(value))
    candidate = private / "terminal_candidate.json"
    atomic_json(candidate, value)
    if cold_replay(candidate, root, design, active):
        raise ValueError("terminal_cold_replay_failed")
    progress("before_terminal_validation", len(receipts))
    terminal_receipts, flagged = _terminal(root, candidate, private)
    value["flagged_adversarial"] = flagged
    if flagged:
        value["honest_verdict"] = "complete_disqualified_terminal_validation"
        value["verdict_class"] = "disqualified"
        value["capstone_execution_ready_score"] = 0
        value["acceptance_gate_results"]["readiness"] = 0
    if value != json.loads(candidate.read_text()):
        atomic_json(candidate, value)
        terminal_receipts, flagged = _terminal(root, candidate, private)
        if flagged:
            value["flagged_adversarial"] = True
        atomic_json(candidate, value)
    if not all(item["passed"] for item in terminal_receipts):
        raise ValueError("terminal_validation_failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, json.loads(candidate.read_text()))
    progress("published", 12 + len(receipts) + len(terminal_receipts))
    return value


def main() -> int:
    """Keep fixture evidence and cold replay separate from the terminal run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--active", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args()
    progress("start", 0)
    root = args.root.resolve()
    design = args.design or root / "openspec/change-proposals/research-roadmap-vNEXT.md"
    active = args.active or root / "research-roadmap.yaml"
    if args.cold_replay:
        errors = cold_replay(args.cold_replay, root, design, active)
        print(json.dumps({"cold_replay_errors": errors}), flush=True)
        return int(bool(errors))
    output = args.output or root / "results/experiment_7902_v685_capstone.json"
    if args.evidence_only:
        atomic_json(output, build_candidate(root, design, active, args.date))
        progress("evidence_only_complete", 12)
        return 0
    run_current(root, design, active, args.date, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
