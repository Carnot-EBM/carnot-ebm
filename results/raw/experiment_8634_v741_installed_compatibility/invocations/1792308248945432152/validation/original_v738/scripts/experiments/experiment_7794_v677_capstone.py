"""Execute V677's evidence-only capstone (REQ-REPORT-7794)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from carnot.experiment_7794_v677_capstone import (  # noqa: E402
    CLI,
    MODULE,
    OUTPUT,
    ROOT,
    TEST,
    account,
    authority,
    build_artifact,
    cold_replay,
)
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file  # noqa: E402
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands  # noqa: E402


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Emit each boundary with measured time and completed units."""
    print(
        f"[exp7794] {phase} {event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def span(name: str, begin: float, end: float, units: int) -> dict[str, Any]:
    """Record a measured, disjoint monotonic phase."""
    return dict(
        phase=name,
        start_s=begin,
        end_s=end,
        duration_s=end - begin,
        completed_units=units,
        run_date="20260928",
        heartbeat_times=[],
    )


def publication(start: float, root: Path, logs: Path) -> dict[str, Any]:
    """Invoke the stable FoVer gate and retain its exact output bytes."""
    command = CommandSpec(
        "publication_gate",
        (str(root / ".venv/bin/python"), "scripts/publication_gate.py", "--json"),
        "stable_publication_gate",
        120,
    )
    progress(start, "publication", "before_subprocess")
    receipt = run_commands(root, [command], log_dir=logs, heartbeat_s=60)[0]
    progress(start, "publication", "after_subprocess", 1)
    value = json.loads(Path(receipt["log_path"]).read_text())
    value.update({name: value["gates"][name]["pass"] for name in ("G1", "G2", "G3", "G4")})
    value.update(
        command_argv=receipt["command_argv"],
        command_exit=receipt["exit_code"],
        source_hash=receipt["log_sha256"],
        publication_performed=False,
    )
    return value


def read_candidate(path: Path, root: Path) -> list[str]:
    """Cold-check the exact candidate and its separately saved rows."""
    value = json.loads(path.read_text())
    errors = cold_replay(value, root)
    rows = path.parent / "rows.json"
    if not rows.is_file() or canonical_hash(json.loads(rows.read_text())) != canonical_hash(
        value.get("rows")
    ):
        errors.append("raw_rows")
    return sorted(set(errors))


def validation_commands(root: Path, private: Path) -> list[CommandSpec]:
    """Freeze complete affected checks and one broad repository diagnostic."""
    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    ruff = str(root / ".venv/bin/ruff")
    mypy = str(root / ".venv/bin/mypy")
    coverage = str(root / ".venv/bin/coverage")
    tests = (
        str(TEST),
        "tests/python/test_experiment_7781_v677_contract_methods.py",
        "tests/python/test_experiment_7787_v677_qwen_event_confidence.py",
    )
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    include = f"{MODULE},{CLI}"
    return [
        CommandSpec(
            "affected_pytest",
            (pytest, *common, f"--basetemp={private / 'pytest/affected'}", *tests, "-q"),
            "frozen_affected",
            1800,
        ),
        CommandSpec(
            "changed_module_coverage",
            (
                coverage,
                "run",
                f"--data-file={private / '.coverage'}",
                f"--include={include}",
                "-m",
                "pytest",
                *common,
                f"--basetemp={private / 'pytest/coverage'}",
                *tests,
                "-q",
            ),
            "new_code",
            1800,
        ),
        CommandSpec(
            "changed_module_coverage_report",
            (
                coverage,
                "report",
                f"--data-file={private / '.coverage'}",
                f"--include={include}",
                "--show-missing",
                "--fail-under=100",
            ),
            "new_code",
            120,
        ),
        CommandSpec(
            "ruff_check", (ruff, "check", str(MODULE), str(CLI), str(TEST)), "changed_files", 120
        ),
        CommandSpec(
            "ruff_format",
            (ruff, "format", "--check", str(MODULE), str(CLI), str(TEST)),
            "changed_files",
            120,
        ),
        CommandSpec("changed_module_mypy", (mypy, str(MODULE), str(CLI)), "changed_files", 180),
        CommandSpec(
            "scoped_spec_coverage",
            (python, "scripts/check_spec_coverage.py", *tests),
            "frozen_tests",
            120,
        ),
        CommandSpec(
            "full_python_suite",
            (pytest, "tests/python", "-q", *common, f"--basetemp={private / 'pytest/broad'}"),
            "repository_health_diagnostic",
            1800,
        ),
    ]


def run_experiment(root: Path, run_date: str, output: Path) -> dict[str, Any]:
    """Validate the owned work, test a fresh candidate, then publish atomically."""
    if run_date != "20260928":
        raise ValueError("V677 run date must be 20260928")
    root = root.resolve()
    start = time.monotonic()
    progress(start, "preflight", "start")
    contract = authority(root)
    rows, _, _ = account(root, contract["tasks"])
    if not contract["comparison"]["passed"]:
        raise ValueError("V677 authority mismatch")
    private = Path(tempfile.mkdtemp(prefix="exp7794-", dir="/tmp"))
    for name in ("affected", "coverage", "broad"):
        (private / "pytest" / name).mkdir(parents=True)
    logs = root / "results/raw/experiment_7794_v677_capstone/validation"
    scope = json.loads(Path("/tmp/exp7794/frozen_scope.json").read_text())
    if scope["changed_modules"] != [str(MODULE), str(CLI)] or scope["direct_tests"] != [str(TEST)]:
        raise ValueError("frozen affected scope mismatch")
    pub = publication(start, root, logs / "publication")
    preflight_end = time.monotonic() - start
    spans = [span("preflight", 0.0, preflight_end, len(rows))]
    progress(start, "preflight", "end", len(rows))
    commands = validation_commands(root, private)
    progress(start, "validation", "before_subprocesses")
    receipts = run_commands(
        root,
        commands,
        log_dir=logs / "validation",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60,
    )
    progress(start, "validation", "after_subprocesses", len(receipts))
    validation_end = time.monotonic() - start
    spans.append(span("validation", preflight_end, validation_end, len(receipts)))
    owned = receipts
    candidate = build_artifact(root, pub, owned, spans, validation_end)
    candidate["validation_receipts"].update(
        repository_collection=receipts[-1],
        frozen_affected_scope=scope,
        frozen_scope_sha256=sha256_file(Path("/tmp/exp7794/frozen_scope.json")),
    )
    atomic_json(private / "rows.json", candidate["rows"])
    candidate_path = private / "candidate.json"
    atomic_json(candidate_path, candidate)
    terminal = [
        CommandSpec(
            "cold_replay",
            (
                str(root / ".venv/bin/python"),
                "-u",
                str(CLI),
                "--root",
                str(root),
                "--cold-validate",
                str(candidate_path),
            ),
            "exact_candidate",
            900,
        ),
        CommandSpec(
            "adversarial_verify",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/adversarial_verify.py",
                "--json",
                str(candidate_path),
            ),
            "exact_candidate",
            900,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate_path),
            ),
            "exact_candidate",
            900,
        ),
    ]
    progress(start, "terminal", "before_subprocesses")
    terminal_receipts = run_commands(
        root,
        terminal,
        log_dir=logs / "terminal",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60,
    )
    progress(start, "terminal", "after_subprocesses", len(terminal_receipts))
    finish = time.monotonic() - start
    spans.append(span("terminal", validation_end, finish, len(terminal_receipts)))
    final = build_artifact(root, pub, owned, spans, finish)
    final["validation_receipts"].update(
        repository_collection=receipts[-1],
        frozen_affected_scope=scope,
        frozen_scope_sha256=sha256_file(Path("/tmp/exp7794/frozen_scope.json")),
        terminal_readers=terminal_receipts,
        cold_reduction=terminal_receipts[0]["passed"],
        exact_candidate_sha256=sha256_file(candidate_path),
    )
    final["flagged_adversarial"] = not terminal_receipts[1]["passed"]
    if not all(r["passed"] for r in terminal_receipts):
        final["verdict_class"] = "disqualified"
        final["honest_verdict"] = "complete_disqualified_v677_capstone_validation"
        final["capstone_complete_score"] = 0
        final["acceptance_gate_results"].update(validity=False, readiness=0)
        for receipt in terminal_receipts:
            if not receipt["passed"]:
                final["gate_check_summary"].append(
                    dict(
                        upstream_id="Exp7794",
                        artifact_path=receipt["log_path"],
                        artifact_hash=receipt["log_sha256"],
                        field=f"terminal.{receipt['name']}.exit_code",
                        operator="==",
                        expected=0,
                        observed=receipt["exit_code"],
                    )
                )
        final["rows"][-1].update(
            verdict_class=final["verdict_class"],
            honest_verdict=final["honest_verdict"],
            flagged_adversarial=final["flagged_adversarial"],
        )
        final["task_dispositions"][-1].update(
            verdict_class=final["verdict_class"], honest_verdict=final["honest_verdict"]
        )
    destination = output if output.is_absolute() else root / output
    progress(start, "artifact", "before_atomic")
    atomic_json(destination, final)
    progress(start, "artifact", "after_atomic", 1)
    return final


def main(argv: list[str] | None = None) -> int:
    """Run this capstone or cold-validate an existing candidate."""
    print("[exp7794] startup flushed", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cold-validate", type=Path)
    args = parser.parse_args(argv)
    if args.cold_validate:
        errors = read_candidate(args.cold_validate, args.root.resolve())
        print(json.dumps({"cold_errors": errors}), flush=True)
        return int(bool(errors))
    run_experiment(args.root, args.date, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
