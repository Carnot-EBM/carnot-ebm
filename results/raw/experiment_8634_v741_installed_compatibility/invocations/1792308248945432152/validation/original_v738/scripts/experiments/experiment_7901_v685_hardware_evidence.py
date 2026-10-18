"""Publish current board custody after bounded validation.

The driver reads dated receipts and never sends a board command.
Spec ref: REQ-REPORT-7901-V685 and SCENARIO-REPORT-7901-CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from coverage import CoverageData

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7901_v685_hardware_evidence import cold_reduce, read_evidence
from scripts.experiments.experiment_7889_v684_hardware_evidence import (
    repository_health_from_artifact,
    run_child,
    terminal_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = "results/experiment_7901_v685_hardware_evidence.json"
MODULE = "python/carnot/reporting/experiment_7901_v685_hardware_evidence.py"
SCRIPT = "scripts/experiments/experiment_7901_v685_hardware_evidence.py"
TEST = "tests/python/test_experiment_7901_v685_hardware_evidence.py"
PRIOR = "results/experiment_7889_v684_hardware_evidence.json"
CONSUMERS = (
    "tests/python/test_experiment_7889_v684_hardware_evidence.py",
    "tests/python/test_experiment_7876_v683_hardware_evidence.py",
    "tests/python/test_experiment_7862_v682_hardware_evidence.py",
)
LIBRARIES = (
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "python/carnot/reporting/experiment_7847_v681_hardware_evidence.py",
    "python/carnot/reporting/experiment_7862_v682_hardware_evidence.py",
    "python/carnot/reporting/experiment_7876_v683_hardware_evidence.py",
    "python/carnot/reporting/experiment_7889_v684_hardware_evidence.py",
    "scripts/experiments/experiment_7889_v684_hardware_evidence.py",
)


def progress(started: float, phase: str, event: str, units: int) -> None:
    """A visible boundary lets an operator tell progress from a hung child."""
    print(
        f"[exp7901] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def check_coverage_shards(paths: list[Path]) -> None:
    """SCENARIO-REPORT-7901-COVERAGE: reject existing zero-line data."""
    owned = {(ROOT / MODULE).resolve(), (ROOT / SCRIPT).resolve()}
    for path in paths:
        if not path.is_file():
            raise ValueError(f"empty_coverage_shard:{path}")
        data = CoverageData(basename=str(path))
        data.read()
        measured = {Path(name).resolve() for name in data.measured_files()}
        if not any(data.lines(str(name)) for name in measured & owned):
            raise ValueError(f"empty_coverage_shard:{path}")


def manifest(private: Path) -> list[dict[str, Any]]:
    """Freeze every required argv and exit rule before any child starts."""
    py = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    cov = str(ROOT / ".venv/bin/coverage")
    ruff = str(ROOT / ".venv/bin/ruff")
    mypy = str(ROOT / ".venv/bin/mypy")
    include = ",".join(str(ROOT / name) for name in (MODULE, SCRIPT))
    common = ["-n", "0", "-o", "addopts=", "--no-cov"]
    unit = private / "unit.coverage"
    success = private / "success.coverage"
    missing = private / "missing.coverage"
    replay = private / "replay.coverage"
    negative = private / "negative.coverage"
    combined = private / "combined.coverage"
    shards = [unit, success, missing, replay, negative]
    fixture = private / "e2e-016-fixture.json"
    pytest_temp = Path("/tmp/carnot-7901-pytest") / private.name
    raw = [
        (
            "affected_pytest",
            [pytest, TEST, *CONSUMERS, "-q", *common, f"--basetemp={pytest_temp / 'affected'}"],
            180,
            0,
            None,
        ),
        (
            "unit_coverage",
            [
                cov,
                "run",
                f"--data-file={unit}",
                f"--include={include}",
                "-m",
                "pytest",
                TEST,
                "-q",
                *common,
                f"--basetemp={pytest_temp / 'coverage'}",
            ],
            180,
            0,
            None,
        ),
        (
            "cli_success",
            [
                cov,
                "run",
                f"--data-file={success}",
                f"--include={include}",
                str(ROOT / SCRIPT),
                "--date",
                "20260929",
                "--root",
                str(ROOT),
                "--output",
                str(private / "success.json"),
                "--evidence-only",
            ],
            30,
            0,
            None,
        ),
        (
            "cli_missing_input",
            [
                cov,
                "run",
                f"--data-file={missing}",
                f"--include={include}",
                str(ROOT / SCRIPT),
                "--date",
                "20260929",
                "--root",
                str(private / "absent"),
                "--output",
                str(private / "missing.json"),
                "--evidence-only",
            ],
            30,
            0,
            None,
        ),
        (
            "cold_replay",
            [
                cov,
                "run",
                f"--data-file={replay}",
                f"--include={include}",
                str(ROOT / SCRIPT),
                "--root",
                str(ROOT),
                "--cold-replay",
                str(private / "success.json"),
            ],
            30,
            0,
            None,
        ),
        (
            "negative_replay",
            [
                cov,
                "run",
                f"--data-file={negative}",
                f"--include={include}",
                str(ROOT / SCRIPT),
                "--root",
                str(ROOT),
                "--cold-replay",
                str(private / "changed.json"),
            ],
            30,
            1,
            "rows_changed",
        ),
        (
            "coverage_shards",
            [
                py,
                "-u",
                "-c",
                "from pathlib import Path; import sys; from scripts.experiments.experiment_7901_v685_hardware_evidence import check_coverage_shards; check_coverage_shards([Path(x) for x in sys.argv[1:]])",
                *(str(path) for path in shards),
            ],
            30,
            0,
            None,
        ),
        (
            "coverage_combine",
            [cov, "combine", f"--data-file={combined}", *(str(path) for path in shards)],
            30,
            0,
            None,
        ),
        (
            "changed_coverage",
            [
                cov,
                "report",
                f"--data-file={combined}",
                f"--include={include}",
                "--show-missing",
                "--fail-under=100",
            ],
            30,
            0,
            None,
        ),
        ("ruff_check", [ruff, "check", MODULE, SCRIPT, TEST], 30, 0, None),
        ("ruff_format", [ruff, "format", "--check", MODULE, SCRIPT, TEST], 30, 0, None),
        ("mypy", [mypy, "--strict", MODULE, SCRIPT], 60, 0, None),
        ("scoped_spec", [py, "-u", "scripts/check_spec_coverage.py", TEST], 30, 0, None),
        (
            "e2e_016_fixture",
            [
                py,
                "-u",
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--fixture-e2e",
                str(fixture),
            ],
            180,
            0,
            None,
        ),
        (
            "e2e_016_replay",
            [
                py,
                "-u",
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--cold-replay",
                str(fixture),
            ],
            180,
            0,
            None,
        ),
    ]
    return [
        {
            "name": name,
            "argv": argv,
            "deadline_s": deadline,
            "classification": "required",
            "expected_exit": expected,
            "required_reason": reason,
        }
        for name, argv, deadline, expected, reason in raw
    ]


def main(argv: list[str] | None = None) -> int:
    """Validate current evidence and publish only the checked candidate."""
    started = time.monotonic()
    progress(started, "entrypoint", "start", 0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.cold_replay:
        progress(started, "cold_replay", "before_reduction", 0)
        print(
            json.dumps(cold_reduce(root, json.loads(args.cold_replay.read_text())), sort_keys=True),
            flush=True,
        )
        progress(started, "cold_replay", "after_reduction", 1)
        return 0
    progress(started, "preconditions", "before_read", 0)
    result = read_evidence(root, args.date)
    progress(started, "preconditions", "after_read", len(result["rows"]))
    output = args.output or root / OUTPUT
    if args.evidence_only:
        atomic_json(output, result)
        progress(started, "evidence_only", "written", len(result["rows"]))
        return 0
    if root != ROOT:
        raise ValueError("full validation requires the worktree root")
    health = repository_health_from_artifact(ROOT / PRIOR)
    closure = {
        name: sha256_file(ROOT / name) for name in (MODULE, SCRIPT, TEST, *CONSUMERS, *LIBRARIES)
    }
    source_identity = {
        name: row["sha256"] for name, row in result["source_artifact_hashes"].items()
    }
    key = canonical_hash(
        {"closure": closure, "sources": source_identity, "date": args.date}
    ).removeprefix("sha256:")
    private = ROOT / "results/raw/experiment_7901_v685_hardware_evidence/attempts" / key
    private.mkdir(parents=True, exist_ok=True)
    commands = manifest(private)
    terminal_checks = terminal_manifest(private)
    frozen = {
        "task_id": result["task_id"],
        "commands": commands,
        "terminal_validators": terminal_checks,
        "source_closure": source_identity,
        "code_closure": {name: closure[name] for name in (MODULE, SCRIPT, *LIBRARIES)},
        "test_closure": {name: closure[name] for name in (TEST, *CONSUMERS)},
        "input_configuration_hash": key,
        "repository_health_diagnostic": {
            "source_artifact_path": health["source_artifact_path"],
            "source_artifact_hash": health["source_artifact_hash"],
            "classification": "retained_diagnostic",
        },
        "e2e_applicability": {
            **{
                f"E2E-{number:03d}": "inapplicable: another producer or physical execution"
                for number in range(1, 19)
                if number != 16
            },
            "E2E-016": "applicable: dated fixture and cold replay",
            "task_cli_e2e": "applicable: success, blocked input, and cold replay",
        },
    }
    manifest_path = private / "validation_command_manifest.json"
    atomic_json(manifest_path, frozen)
    result["validation_command_manifest_path"] = str(manifest_path)
    result["validation_command_manifest_sha256"] = sha256_file(manifest_path)
    result["field_principles"]["validation_command_manifest_sha256"] = (
        "Authenticate the frozen validation plan."
    )
    checkpoint = private / "completed_units.json"
    previous = json.loads(checkpoint.read_text()).get("checks", []) if checkpoint.is_file() else []
    receipts: list[dict[str, Any]] = []
    for spec in commands:
        index = len(receipts)
        if index < len(previous):
            receipt = previous[index]
            log = Path(receipt["log_path"])
            if (
                any(
                    receipt.get(field) != spec[field]
                    for field in ("name", "argv", "classification")
                )
                or not log.is_file()
                or sha256_file(log) != receipt["log_sha256"]
            ):
                raise ValueError("checkpoint_receipt_changed")
            progress(started, spec["name"], "resumed", index + 1)
        else:
            if spec["name"] == "negative_replay":
                changed = json.loads((private / "success.json").read_text())
                changed["board_rows"][0]["k_max"] = 6
                atomic_json(private / "changed.json", changed)
            receipt = run_child(spec, private, started, index)
            receipt["raw_passed"] = receipt["passed"]
            reason = spec["required_reason"]
            receipt["passed"] = (
                receipt["exit_code"] == spec["expected_exit"]
                and not receipt["timed_out"]
                and (reason is None or reason in Path(receipt["log_path"]).read_text())
            )
            atomic_json(checkpoint, {"checks": [*receipts, receipt]})
        receipts.append(receipt)
        result["observed_child_commands"].append(
            {
                "name": receipt["name"],
                "argv": receipt["argv"],
                "classification": receipt["classification"],
                "expected_exit": spec["expected_exit"],
                "actual_exit": receipt["exit_code"],
            }
        )
    required_passed = all(row["passed"] for row in receipts)
    result["validation_receipts"] = {"checks": receipts, "required_checks_passed": required_passed}
    result["repository_health"] = health
    result["resolved_imports"] = {
        "carnot.reporting.experiment_7901_v685_hardware_evidence": str((ROOT / MODULE).resolve()),
        "scripts.experiments.experiment_7901_v685_hardware_evidence": str(
            (ROOT / SCRIPT).resolve()
        ),
    }
    if not required_passed:
        result["honest_verdict"] = "complete_disqualified_required_checks"
        result["verdict_class"] = "disqualified"
        result["hardware_evidence_ready_score"] = 0
    result["acceptance_gate_results"]["readiness"] = int(
        required_passed and result["hardware_evidence_ready_score"] == 1
    )
    result["duration_s"] = time.monotonic() - started
    result["phase_spans"]["validation_s"] = (
        result["duration_s"] - result["phase_spans"]["evidence_read_s"]
    )
    result["reproducibility_checksum"] = canonical_hash(
        {
            "base": result["reproducibility_checksum"],
            "manifest": result["validation_command_manifest_sha256"],
            "seed": 0,
        }
    )
    candidate = private / "terminal-candidate.json"
    atomic_json(candidate, result)
    reports = [
        run_child(spec, private, started, len(receipts) + index)
        for index, spec in enumerate(terminal_checks)
    ]
    try:
        flags = json.loads(Path(reports[0]["log_path"]).read_text())["flagged_count"]
    except (ValueError, KeyError, TypeError):
        flags = 1
    if flags or not all(report["passed"] for report in reports):
        result["flagged_adversarial"] = bool(flags)
        result["honest_verdict"] = "complete_disqualified_terminal_validation"
        result["verdict_class"] = "disqualified"
        result["hardware_evidence_ready_score"] = 0
        result["acceptance_gate_results"]["readiness"] = 0
        atomic_json(candidate, result)
        reports = [
            run_child(spec, private, started, len(receipts) + index)
            for index, spec in enumerate(terminal_checks)
        ]
    atomic_json(
        private / "terminal_validation_reports.json",
        {"candidate_sha256": sha256_file(candidate), "reports": reports},
    )
    if not all(report["passed"] for report in reports):
        raise ValueError("terminal_validation_failed")
    atomic_json(output, result)
    progress(started, "final", "written", len(receipts) + len(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
