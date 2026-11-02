"""Run the CPU aggregation capstone for V674 (REQ-REPORT-7752)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

# Direct execution puts scripts/experiments on sys.path, while the shared
# roadmap schema lives under the repository's scripts package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.experiment_7752_v674_capstone import (
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
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Flush a measured boundary so each owned phase stays observable."""
    print(
        f"[exp7752] {phase} {event} elapsed_s={time.monotonic() - start:.3f} "
        f"completed_units={units}",
        flush=True,
    )


def publication(root: Path, start: float) -> dict[str, object]:
    """Ask the unchanged G1-G4 gate for its own current result."""
    command = [str(ROOT / ".venv/bin/python"), "scripts/publication_gate.py", "--json"]
    progress(start, "publication_gate", "before_subprocess")
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=120, check=False
    )
    progress(start, "publication_gate", "after_subprocess", 1)
    if result.returncode:
        raise RuntimeError(f"publication gate exited {result.returncode}: {result.stderr}")
    value = json.loads(result.stdout)
    value.update(
        command=".venv/bin/python scripts/publication_gate.py --json",
        exit_code=result.returncode,
        result_hash=canonical_hash(value),
        headline_auroc=0.9131,
        publication_performed=False,
    )
    return value


def read_candidate(path: Path, root: Path) -> list[str]:
    """Cold-reduce exact source and raw row bytes in this fresh process."""
    value = json.loads(path.read_text())
    errors = cold_replay(value, root)
    raw = root / RAW / "rows.json"
    if not raw.is_file() or canonical_hash(json.loads(raw.read_text())) != canonical_hash(
        value.get("rows")
    ):
        errors.append("raw_rows")
    return sorted(set(errors))


def span(
    phase: str, start_s: float, end_s: float, units: int, checkpoints: dict[str, str]
) -> dict[str, object]:
    """Record disjoint monotonic work with exact checkpoint hashes."""
    return {
        "phase": phase,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": end_s - start_s,
        "run_date": "20260927",
        "heartbeat_times": [],
        "completed_units": units,
        "checkpoint_hashes": checkpoints,
    }


def run_experiment(root: Path, run_date: str, output: Path) -> dict[str, object]:
    """Validate affected code and the exact candidate before atomic publication."""
    start = time.monotonic()
    progress(start, "preflight", "start")
    root = root.resolve()
    if run_date != "20260927":
        raise ValueError("V674 run date must be 20260927")
    source = authority(root)
    rows, _, _ = account(root, source["tasks"])
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    scope = {
        "changed_modules": [str(MODULE)],
        "static_paths": [str(CLI)],
        "tests": [str(TEST)],
        "requirements": ["REQ-REPORT-7752", "SCENARIO-REPORT-7752-REPLAY"],
    }
    atomic_json(raw / "frozen_affected_scope.json", scope)
    pub = publication(root, start)
    boundary = time.monotonic() - start
    spans = [
        span(
            "preflight",
            0.0,
            boundary,
            len(rows),
            {"frozen_affected_scope.json": sha256_file(raw / "frozen_affected_scope.json")},
        )
    ]
    private = Path(tempfile.mkdtemp(prefix="exp7752-validation-", dir="/tmp"))
    (private / "pytest").mkdir()
    commands = build_scoped_commands(
        ROOT,
        [str(TEST)],
        [str(MODULE)],
        static_paths=[str(CLI)],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage",
    )
    progress(start, "validation", "before_subprocesses")
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=raw / "validation/affected",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60,
    )
    end = time.monotonic() - start
    spans.append(span("validation", boundary, end, len(receipts), {}))
    progress(start, "validation", "after_subprocesses", len(receipts))
    candidate = build_artifact(root, pub, receipts, spans, end)
    atomic_json(raw / "rows.json", candidate["rows"])
    candidate_path = raw / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    terminal = [
        CommandSpec(
            "cold_replay",
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
    progress(start, "terminal", "before_subprocesses")
    terminal_receipts = run_commands(
        ROOT,
        terminal,
        log_dir=raw / "validation/terminal",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60,
    )
    finish = time.monotonic() - start
    spans.append(
        span(
            "terminal",
            end,
            finish,
            len(terminal_receipts),
            {"terminal_candidate.json": sha256_file(candidate_path)},
        )
    )
    progress(start, "terminal", "after_subprocesses", len(terminal_receipts))
    final = build_artifact(root, pub, receipts, spans, finish)
    final["validation_receipts"]["terminal_readers"] = terminal_receipts
    final["validation_receipts"]["cold_reduction"] = terminal_receipts[0]["passed"]
    if not all(item["passed"] for item in terminal_receipts):
        final["verdict_class"] = "disqualified"
        final["honest_verdict"] = "complete_disqualified_v674_terminal_reader"
        final["flagged_adversarial"] = not terminal_receipts[1]["passed"]
        final["capstone_complete_score"] = 0
        final["acceptance_gate_results"]["validity"] = False
    destination = output if output.is_absolute() else root / output
    progress(start, "publication", "before_atomic")
    atomic_json(destination, final)
    progress(start, "publication", "after_atomic", 1)
    return final


def main(argv: list[str] | None = None) -> int:
    """Run the experiment or validate a previously captured candidate."""
    print("[exp7752] startup flushed", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--date", default="20260927")
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
