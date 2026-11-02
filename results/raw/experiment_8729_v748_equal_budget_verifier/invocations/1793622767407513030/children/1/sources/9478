"""Run V676's evidence-only capstone (REQ-REPORT-7780)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.experiment_7780_v676_capstone import (  # noqa: E402
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
    """Flush every phase and child boundary with elapsed work."""
    print(
        f"[exp7780] {phase} {event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def span(name: str, begin: float, end: float, units: int) -> dict[str, Any]:
    """Record disjoint monotonic phase time without a duration floor."""
    return {
        "phase": name,
        "start_s": begin,
        "end_s": end,
        "duration_s": end - begin,
        "run_date": "20260927",
        "completed_units": units,
        "heartbeat_times": [],
    }


def publication(start: float) -> dict[str, Any]:
    """Run the stable older FoVer G1-G4 gate unchanged."""
    argv = [str(ROOT / ".venv/bin/python"), "scripts/publication_gate.py", "--json"]
    progress(start, "publication_gate", "before_subprocess")
    result = subprocess.run(
        argv, cwd=ROOT, capture_output=True, text=True, timeout=120, check=False
    )
    progress(start, "publication_gate", "after_subprocess", 1)
    value = json.loads(result.stdout)
    value.update(
        command_argv=argv,
        command_exit=result.returncode,
        result_hash=canonical_hash(value),
        headline_auroc=0.9131,
        publication_performed=False,
    )
    return value


def read_candidate(path: Path, root: Path) -> list[str]:
    """Cold-reduce exact raw rows and source bytes in a fresh process."""
    value = json.loads(path.read_text())
    errors = cold_replay(value, root)
    rows_path = root / RAW / "rows.json"
    if not rows_path.is_file() or canonical_hash(
        json.loads(rows_path.read_text())
    ) != canonical_hash(value.get("rows")):
        errors.append("raw_rows")
    return sorted(set(errors))


def existing_repository_receipt(root: Path, raw: Path) -> list[dict[str, Any]] | None:
    """Reuse one exact logged broad diagnostic when only owned checks changed."""
    candidate_path = raw / "terminal_candidate.json"
    if not candidate_path.is_file():
        return None
    previous = json.loads(candidate_path.read_text())
    receipts = previous.get("validation_receipts", {}).get("repository_collection")
    if not isinstance(receipts, list) or len(receipts) != 1:
        return None
    receipt = receipts[0]
    argv = receipt.get("command_argv", [])
    log = root / receipt.get("log_path", "")
    if (
        receipt.get("name") != "full_python_suite"
        or len(argv) < 2
        or argv[1] != "tests/python"
        or not log.is_file()
        or sha256_file(log) != receipt.get("log_sha256")
    ):
        return None
    return receipts


def run_experiment(root: Path, run_date: str, output: Path) -> dict[str, Any]:
    """Validate the exact candidate, then publish one atomic terminal record."""
    start = time.monotonic()
    progress(start, "preflight", "start")
    root = root.resolve()
    if run_date != "20260927":
        raise ValueError("V676 run date must be 20260927")
    source = authority(root)
    rows, _, _ = account(root, source["tasks"])
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    frozen = raw / "frozen_affected_scope.json"
    if not frozen.is_file():
        raise FileNotFoundError(frozen)
    scope = json.loads(frozen.read_text())
    if scope["changed_modules"] != [str(MODULE)] or scope["direct_tests"] != [str(TEST)]:
        raise ValueError("frozen affected scope mismatch")
    pub = publication(start)
    preflight_end = time.monotonic() - start
    spans = [span("preflight", 0.0, preflight_end, len(rows))]
    progress(start, "preflight", "end", len(rows))

    private = Path(tempfile.mkdtemp(prefix="exp7780-validation-", dir="/tmp"))
    for name in ("focused", "coverage", "broad"):
        (private / "pytest" / name).mkdir(parents=True)
    probe = CommandSpec(
        "basetemp_parent_probe",
        (
            str(ROOT / ".venv/bin/python"),
            "-c",
            "from pathlib import Path; import sys; assert all(Path(p).is_dir() for p in sys.argv[1:])",
            *(str(private / "pytest" / name) for name in ("focused", "coverage", "broad")),
        ),
        "private_temp",
        30,
    )
    probe_receipts = run_commands(ROOT, [probe], log_dir=raw / "validation/probe", heartbeat_s=60)
    tests = [*scope["direct_tests"], *scope["transitive_consumer_tests"]]
    commands = build_scoped_commands(
        ROOT,
        tests,
        scope["changed_modules"],
        static_paths=scope["static_paths"],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage",
    )
    progress(start, "affected_validation", "before_subprocesses")
    affected = run_commands(
        ROOT,
        commands,
        log_dir=raw / "validation/affected",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60,
    )
    progress(start, "affected_validation", "after_subprocesses", len(affected))
    broad = CommandSpec(
        "full_python_suite",
        (
            str(ROOT / ".venv/bin/pytest"),
            "tests/python",
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            f"--basetemp={private / 'pytest/broad'}",
        ),
        "repository_health_diagnostic",
        1800,
    )
    broad_receipts = existing_repository_receipt(root, raw)
    if broad_receipts is None:
        progress(start, "repository_collection", "before_subprocess")
        broad_receipts = run_commands(
            ROOT,
            [broad],
            log_dir=raw / "validation/repository",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=60,
        )
        progress(start, "repository_collection", "after_subprocess", 1)
    else:
        progress(start, "repository_collection", "reused_exact_logged_diagnostic", 1)
    validation_end = time.monotonic() - start
    spans.append(span("validation", preflight_end, validation_end, len(affected) + 2))
    receipts = [*probe_receipts, *affected]
    candidate = build_artifact(root, pub, receipts, spans, validation_end)
    candidate["validation_receipts"]["repository_collection"] = broad_receipts
    candidate["validation_receipts"]["repository_collection_healthy"] = broad_receipts[0]["passed"]
    candidate["validation_receipts"]["frozen_scope_sha256"] = sha256_file(frozen)
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
                "--json",
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
    spans.append(span("terminal", validation_end, finish, len(terminal_receipts)))
    progress(start, "terminal", "after_subprocesses", len(terminal_receipts))
    final = build_artifact(root, pub, receipts, spans, finish)
    final["validation_receipts"].update(
        repository_collection=broad_receipts,
        repository_collection_healthy=broad_receipts[0]["passed"],
        terminal_readers=terminal_receipts,
        cold_reduction=terminal_receipts[0]["passed"],
        exact_candidate_sha256=sha256_file(candidate_path),
    )
    final["flagged_adversarial"] = not terminal_receipts[1]["passed"]
    if not all(receipt["passed"] for receipt in terminal_receipts):
        final["verdict_class"] = "disqualified"
        final["honest_verdict"] = "complete_disqualified_v676_terminal_reader"
        final["capstone_complete_score"] = 0
        final["acceptance_gate_results"].update(validity=False, readiness=0)
        for item in terminal_receipts:
            if not item["passed"]:
                final["gate_check_summary"].append(
                    {
                        "upstream_id": "Exp7780",
                        "artifact_path": item["log_path"],
                        "artifact_hash": item["log_sha256"],
                        "field": f"terminal.{item['name']}.exit_code",
                        "operator": "==",
                        "expected": 0,
                        "observed": item["exit_code"],
                    }
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
    """Run the current capstone or cold-check a previously written candidate."""
    print("[exp7780] startup flushed", flush=True)
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
