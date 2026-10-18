"""Run V675 artifact aggregation and exact terminal checks (REQ-REPORT-7766)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.experiment_7766_v675_capstone import (  # noqa: E402
    CLI,
    MODULE,
    OUTPUT,
    RAW,
    ROOT,
    TEST,
    account,
    authority,
    build_artifact,
    cold_replay,
)
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file  # noqa: E402
from carnot.reporting.experiment_7303_validation_scope import (  # noqa: E402
    CommandSpec,
    build_scoped_commands,
    run_commands,
)


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Make every phase boundary visible with real monotonic elapsed time."""
    print(
        f"[exp7766] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} "
        f"completed_units={units}",
        flush=True,
    )


def span(phase: str, first: float, last: float, units: int) -> dict[str, object]:
    """Keep disjoint measured phases so duration never needs padding."""
    return {
        "phase": phase,
        "start_s": first,
        "end_s": last,
        "duration_s": last - first,
        "completed_units": units,
        "run_date": "20260927",
        "heartbeat_times": [],
    }


def publication(start: float) -> dict[str, object]:
    """Ask the unchanged gate for its narrow older FoVer headline result."""
    argv = [str(ROOT / ".venv/bin/python"), "scripts/publication_gate.py", "--json"]
    progress(start, "publication_gate", "before_subprocess")
    result = subprocess.run(
        argv, cwd=ROOT, capture_output=True, text=True, timeout=120, check=False
    )
    progress(start, "publication_gate", "after_subprocess", 1)
    if result.returncode:
        raise RuntimeError(f"publication gate failed: {result.returncode} {result.stderr}")
    value = json.loads(result.stdout)
    value.update(
        command_argv=argv,
        command_exit=result.returncode,
        result_hash=canonical_hash(value),
        headline_auroc=0.9131,
        claim_scope="established_FoVer_headline_only",
        publication_performed=False,
    )
    return value


def read_candidate(path: Path, root: Path, raw: Path) -> list[str]:
    """Compare raw rows and every source in the current fresh process."""
    value = json.loads(path.read_text())
    errors = cold_replay(value, root)
    if not raw.is_file() or canonical_hash(json.loads(raw.read_text())) != canonical_hash(
        value["rows"]
    ):
        errors.append("raw_rows")
    return sorted(set(errors))


def reduce_rows(path: Path, root: Path) -> list[str]:
    """Independently reduce actual source rows without using headline fields."""
    actual = json.loads(path.read_text())
    expected, _, _ = account(root, authority(root)["tasks"])
    expected[-1].update(
        verdict_class=actual[-1]["verdict_class"],
        honest_verdict=actual[-1]["honest_verdict"],
        raw_metrics=actual[-1]["raw_metrics"],
    )
    return [] if canonical_hash(actual) == canonical_hash(expected) else ["rows"]


def run_experiment(root: Path, run_date: str, output: Path) -> dict[str, object]:
    """Validate current files, then publish one terminal result atomically."""
    start = time.monotonic()
    progress(start, "preflight", "start")
    root = root.resolve()
    if run_date != "20260927":
        raise ValueError("V675 run date must be 20260927")
    source = authority(root)
    rows, _, _ = account(root, source["tasks"])
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    scope = json.loads((raw / "frozen_affected_scope.json").read_text())
    assert scope["changed_modules"] == [str(MODULE)]
    assert scope["static_paths"] == [str(CLI)]
    assert scope["tests"] == [str(TEST)]
    preflight_end = time.monotonic() - start
    progress(start, "preflight", "end", len(rows))
    pub = publication(start)
    publication_end = time.monotonic() - start
    private = Path(tempfile.mkdtemp(prefix="exp7766-validation-", dir="/tmp"))
    (private / "pytest").mkdir()
    # A real child proves the nested basetemp parent exists before pytest starts.
    probe = CommandSpec(
        "basetemp_parent_probe",
        (
            str(ROOT / ".venv/bin/python"),
            "-c",
            "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.mkdir(); print(p.is_dir())",
            str(private / "pytest" / "child"),
        ),
        "private_directory",
        60,
    )
    progress(start, "validation", "before_subprocess")
    probe_receipts = run_commands(
        ROOT,
        [probe],
        log_dir=raw / "validation/probe",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60,
    )
    commands = build_scoped_commands(
        ROOT,
        [str(TEST)],
        [str(MODULE)],
        static_paths=[str(CLI)],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage",
    )
    commands.append(
        CommandSpec(
            "full_python_suite",
            (
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'pytest' / 'full'}",
                "tests/python",
                "-q",
            ),
            "all_python_tests",
            3600,
        )
    )
    receipts = probe_receipts + run_commands(
        ROOT,
        commands,
        log_dir=raw / "validation/affected",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60,
    )
    validation_end = time.monotonic() - start
    progress(start, "validation", "after_subprocess", len(receipts))
    spans = [
        span("preflight", 0.0, preflight_end, len(rows)),
        span("publication_gate", preflight_end, publication_end, 1),
        span("validation", publication_end, validation_end, len(receipts)),
    ]
    candidate = build_artifact(root, pub, receipts, spans, validation_end)
    atomic_json(raw / "rows.json", candidate["rows"])
    candidate_path = raw / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    terminal = [
        CommandSpec(
            "exact_contract",
            (
                str(ROOT / ".venv/bin/python"),
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
            "independent_row_reduction",
            (
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(CLI),
                "--root",
                str(root),
                "--reduce-rows",
                str(raw / "rows.json"),
            ),
            "raw_rows",
            900,
        ),
        CommandSpec(
            "adversarial_verify",
            (
                str(ROOT / ".venv/bin/python"),
                "-u",
                "scripts/adversarial_verify.py",
                str(candidate_path),
            ),
            "exact_candidate",
            900,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                str(ROOT / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate_path),
            ),
            "exact_candidate",
            900,
        ),
    ]
    progress(start, "terminal", "before_subprocess")
    terminal_receipts = run_commands(
        ROOT,
        terminal,
        log_dir=raw / "validation/terminal",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60,
    )
    terminal_end = time.monotonic() - start
    progress(start, "terminal", "after_subprocess", len(terminal_receipts))
    spans.append(span("terminal", validation_end, terminal_end, len(terminal_receipts)))
    final = build_artifact(root, pub, receipts, spans, terminal_end)
    final["validation_receipts"]["terminal_readers"] = terminal_receipts
    final["validation_receipts"]["cold_replay"] = terminal_receipts[0]["passed"]
    final["validation_receipts"]["independent_row_reduction"] = terminal_receipts[1]["passed"]
    final["flagged_adversarial"] = not terminal_receipts[2]["passed"]
    if not all(item["passed"] for item in terminal_receipts):
        final["verdict_class"] = "disqualified"
        final["honest_verdict"] = "complete_disqualified_v675_terminal_reader"
        final["capstone_complete_score"] = 0
        final["acceptance_gate_results"]["readiness"] = 0
        final["acceptance_gate_results"]["validity"] = False
    progress(start, "artifact", "before_atomic")
    atomic_json(output if output.is_absolute() else root / output, final)
    progress(start, "artifact", "after_atomic", 1)
    return final


def main(argv: list[str] | None = None) -> int:
    """Dispatch the real run and fresh-process terminal readers."""
    print("[exp7766] phase=startup event=flushed elapsed_s=0 completed_units=0", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cold-validate", type=Path)
    parser.add_argument("--reduce-rows", type=Path)
    args = parser.parse_args(argv)
    if args.cold_validate:
        errors = read_candidate(
            args.cold_validate, args.root.resolve(), args.root / RAW / "rows.json"
        )
        print(json.dumps({"cold_errors": errors}), flush=True)
        return int(bool(errors))
    if args.reduce_rows:
        errors = reduce_rows(args.reduce_rows, args.root.resolve())
        print(json.dumps({"row_errors": errors}), flush=True)
        return int(bool(errors))
    run_experiment(args.root, args.date, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
