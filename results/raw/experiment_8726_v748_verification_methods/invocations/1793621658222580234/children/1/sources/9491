#!/usr/bin/env python3
"""Run the V678 independent evidence audit (REQ-REPORT-7807)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from carnot.experiment_7807_v678_independent_evidence_audit import (
    build_artifact,
    cold_replay,
    inspect_sources,
    read_branches,
)
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting import experiment_7303_validation_scope as checks


def progress(start: float, phase: str, event: str, units: int) -> None:
    """Print each phase boundary with measured elapsed time and units."""
    print(
        f"phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def run_experiment(root: Path, date: str, output: Path) -> dict[str, Any]:
    """Run current custody, explicit checks, and terminal readers once."""
    root = root.resolve()
    start = time.monotonic()
    spans: list[dict[str, Any]] = []
    phase_start = 0.0

    def finish(phase: str, units: int) -> None:
        nonlocal phase_start
        end = time.monotonic() - start
        spans.append(
            {
                "phase": phase,
                "start_s": phase_start,
                "end_s": end,
                "duration_s": end - phase_start,
                "completed_units": units,
            }
        )
        phase_start = end
        progress(start, phase, "complete", units)

    progress(start, "preconditions", "start", 0)
    sources, failures = inspect_sources(root)
    scope_path = (
        root / "results/raw/experiment_7807_v678_independent_evidence_audit/frozen_scope.json"
    )
    scope = json.loads(scope_path.read_bytes())
    raw_dir = root / "results/raw/experiment_7807_v678_independent_evidence_audit"
    finish("preconditions", len(sources))
    progress(start, "reduction", "start", 0)
    branches, raw_failures = read_branches(root, sources)
    artifact = build_artifact(root, date, sources, failures + raw_failures, branches)
    artifact["validation_receipts"]["frozen_affected_scope"] = scope
    finish("reduction", len(artifact["rows"]))

    private = Path("/tmp/exp7807-validation")
    private.mkdir(parents=True, exist_ok=True)
    (private / "basetemp").mkdir(parents=True, exist_ok=True)
    (private / "coverage.data").unlink(missing_ok=True)
    commands = checks.build_scoped_commands(
        root,
        scope["direct_tests"] + scope["transitive_consumers"],
        scope["changed_modules"],
        static_paths=scope["static_paths"],
        basetemp=private / "basetemp",
        coverage_file=private / "coverage.data",
    )
    frozen = [(item["name"], item["argv"]) for item in scope["commands"]]
    if [(item.name, list(item.argv)) for item in commands] != frozen:
        raise ValueError("frozen_validation_argv_changed")
    progress(start, "affected_validation", "start", 0)
    receipts = checks.run_commands(
        root,
        commands,
        log_dir=raw_dir / "validation/affected",
        extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / "coverage.data")},
        heartbeat_s=30,
    )
    artifact["validation_receipts"]["required_commands"] = receipts
    artifact["validation_receipts"].update(checks.reduce_required_checks(receipts))
    finish("affected_validation", len(receipts))

    progress(start, "full_python_suite", "start", 0)
    full = checks.CommandSpec(
        "full_python_suite",
        (
            str(root / ".venv/bin/pytest"),
            "tests/python",
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            f"--basetemp={private / 'basetemp/full'}",
        ),
        "repository_health",
        1800,
    )
    cached_path = raw_dir / "validation/full/receipt.json"
    if cached_path.is_file():
        cached = json.loads(cached_path.read_bytes())
        log_path = root / cached["log_path"]
        if (
            cached["command_argv"] != list(full.argv)
            or sha256_file(log_path) != cached["log_sha256"]
        ):
            raise ValueError("invalid_full_suite_receipt")
        full_receipts = [cached]
    else:
        full_receipts = checks.run_commands(
            root,
            [full],
            log_dir=raw_dir / "validation/full",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
    artifact["validation_receipts"]["full_python_suite"] = full_receipts
    artifact["validation_receipts"]["repository_health_passed"] = full_receipts[0]["passed"]
    finish("full_python_suite", 1)
    if not artifact["validation_receipts"]["required_checks_passed"]:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["acceptance_gate_results"].update(validity=False, readiness=0)
        artifact["independent_evidence_ready_score"] = 0

    candidate = raw_dir / "terminal_candidate.json"
    artifact["phase_spans"] = list(spans)
    artifact["duration_s"] = time.monotonic() - start
    atomic_json(candidate, artifact)
    progress(start, "terminal_readers", "start", 0)
    python = str(root / ".venv/bin/python")
    readers = [
        checks.CommandSpec(
            "cold_replay",
            (
                python,
                "-u",
                "scripts/experiments/experiment_7807_v678_independent_evidence_audit.py",
                "--cold",
                str(candidate),
            ),
            "exact_candidate",
            180,
        ),
        checks.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)),
            "exact_candidate",
            180,
        ),
        checks.CommandSpec(
            "strict_row_consistency",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
            180,
        ),
    ]
    terminal = checks.run_commands(
        root,
        readers,
        log_dir=raw_dir / "validation/terminal",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=30,
    )
    artifact["validation_receipts"]["terminal_readers"] = terminal
    artifact["validation_receipts"]["exact_candidate_sha256"] = sha256_file(candidate)
    artifact["validation_receipts"]["e2e_checks"] = [
        {"name": "real_entrypoint_cold_replay", "passed": terminal[0]["passed"]}
    ]
    report = (
        json.loads((root / terminal[1]["log_path"]).read_text()) if terminal[1]["passed"] else {}
    )
    artifact["flagged_adversarial"] = bool(report.get("flagged_count", 1))
    if not all(item["passed"] for item in terminal) or artifact["flagged_adversarial"]:
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["verdict_class"] = "disqualified"
        artifact["acceptance_gate_results"].update(validity=False, readiness=0)
        artifact["independent_evidence_ready_score"] = 0
    finish("terminal_readers", len(terminal))
    progress(start, "publication", "start", 0)
    finish("publication", 1)
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - start
    atomic_json(output, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Choose the task run or exact candidate cold replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--cold", type=Path)
    args = parser.parse_args(argv)
    if args.cold is not None:
        failures = cold_replay(args.cold)
        print(json.dumps({"cold_replay_failures": failures}), flush=True)
        return int(bool(failures))
    print("Exp7807 start elapsed_s=0.000 completed_units=0", flush=True)
    root = Path(__file__).resolve().parents[2]
    result = run_experiment(
        root, args.date, root / "results/experiment_7807_v678_independent_evidence_audit.json"
    )
    print(
        json.dumps(
            {"honest_verdict": result["honest_verdict"], "verdict_class": result["verdict_class"]}
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
