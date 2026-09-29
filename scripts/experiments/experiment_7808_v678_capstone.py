"""Run the V678 capstone with bounded checks and durable receipts.

REQ-REPORT-7808; SCENARIO-REPORT-7808-TERMINAL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from carnot.experiment_7808_v678_capstone import (  # noqa: E402
    CLI,
    DESIGN,
    MODULE,
    OUTPUT,
    ROADMAP,
    ROOT,
    TEST,
    account_tasks,
    build_artifact,
    check_authority,
    replay_candidate,
)
from carnot.reporting.current_work_receipt import atomic_json, sha256_file  # noqa: E402
from carnot.reporting.experiment_7303_validation_scope import (  # noqa: E402
    CommandSpec,
    run_commands,
    run_scoped_validation,
)

RAW = Path("results/raw/experiment_7808_v678_capstone")
TESTS = (
    str(TEST),
    "tests/python/test_experiment_7795_v678_contract_methods.py",
    "tests/python/test_experiment_7807_v678_independent_evidence_audit.py",
    "tests/python/test_experiment_7303_v642_validation_scope.py",
)


def progress(start: float, phase: str, event: str, units: int) -> None:
    """Show real elapsed time and completed work at every phase boundary."""
    print(
        f"[exp7808] {phase} {event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def span(name: str, start: float, stop: float, units: int) -> dict[str, Any]:
    """Store measured, disjoint monotonic phase time."""
    return dict(
        phase=name,
        start_s=start,
        end_s=stop,
        duration_s=stop - start,
        completed_units=units,
        run_date="20260928",
        heartbeat_times=[],
    )


def _run(root: Path, name: str, argv: tuple[str, ...], timeout: float) -> dict[str, Any]:
    """Run one owned child with a heartbeat and hash its exact output."""
    return run_commands(
        root,
        [CommandSpec(name, argv, "V678_required", timeout)],
        log_dir=root / RAW / "validation" / name,
        heartbeat_s=30,
    )[0]


def run_experiment(root: Path, run_date: str) -> dict[str, Any]:
    """Validate the current inputs, then atomically publish one terminal result."""
    if run_date != "20260928":
        raise ValueError("V678 run date must be 20260928")
    start = time.monotonic()
    spans: list[dict[str, Any]] = []
    progress(start, "preflight", "start", 0)
    phase = time.monotonic()
    design, roadmap = (root / DESIGN).read_bytes(), (root / ROADMAP).read_bytes()
    authority = check_authority(design, roadmap)
    if not authority["passed"]:
        raise ValueError(f"V678 authority mismatch: {authority['errors']}")
    tasks = __import__("yaml").safe_load(roadmap)["tasks"]
    rows, _, failures = account_tasks(root, tasks)
    preflight = dict(
        design_path=str(DESIGN),
        design_sha256=sha256_file(root / DESIGN),
        roadmap_path=str(ROADMAP),
        roadmap_sha256=sha256_file(root / ROADMAP),
        authority_match=True,
        declared_paths=[r["producer_path"] for r in rows],
        input_presence=[
            dict(task_id=r["task_id"], availability=r["availability"], sha256=r["producer_hash"])
            for r in rows
        ],
        failed_operands=failures,
        backend="host_cpu_aggregation",
        resources="CPU, local files, no model or board load",
    )
    atomic_json(root / RAW / "preflight.json", preflight)
    print(json.dumps({"preconditions_checked": preflight}, sort_keys=True), flush=True)
    spans.append(span("preflight", phase, time.monotonic(), len(rows)))
    progress(start, "preflight", "complete", len(rows))

    phase = time.monotonic()
    progress(start, "summaries", "start", 0)
    summaries = []
    for index, row in enumerate(rows[:-1], 1):
        if row["availability"] == "producer":
            receipt = _run(
                root,
                f"summary_{row['experiment_id']}",
                (
                    str(root / ".venv/bin/python"),
                    "scripts/summarize_artifact.py",
                    row["producer_path"],
                ),
                120,
            )
            summaries.append(receipt)
        progress(start, "summaries", "unit", index)
    spans.append(span("summaries", phase, time.monotonic(), len(summaries)))
    progress(start, "summaries", "complete", len(summaries))

    phase = time.monotonic()
    progress(start, "publication", "start", 0)
    publication_receipt = _run(
        root,
        "publication_gate",
        (str(root / ".venv/bin/python"), "scripts/publication_gate.py", "--json"),
        120,
    )
    publication = json.loads((root / publication_receipt["log_path"]).read_text())
    publication.update(
        {name: publication["gates"][name]["pass"] for name in ("G1", "G2", "G3", "G4")}
    )
    publication.update(
        command_argv=publication_receipt["command_argv"],
        command_exit=publication_receipt["exit_code"],
        source_hash=publication_receipt["log_sha256"],
        publication_performed=False,
    )
    spans.append(span("publication", phase, time.monotonic(), 1))
    progress(start, "publication", "complete", 1)

    phase = time.monotonic()
    progress(start, "validation", "start", 0)
    private = Path(tempfile.mkdtemp(prefix="exp7808_", dir="/tmp"))
    (private / "pytest").mkdir()
    scoped = run_scoped_validation(
        root,
        TESTS,
        (str(MODULE),),
        static_paths=(str(CLI),),
        basetemp=private / "pytest",
        coverage_file=private / ".coverage",
        log_dir=root / RAW / "validation" / "affected",
    )
    coverage = str(root / ".venv/bin/coverage")
    include = f"{MODULE},{CLI}"
    shard = private / ".coverage_cli"
    combined = private / ".coverage_combined"
    extra = [
        _run(
            root,
            "cli_coverage",
            (
                coverage,
                "run",
                f"--data-file={shard}",
                f"--include={include}",
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'pytest/cli_coverage'}",
                str(TEST),
                "-q",
            ),
            180,
        ),
        _run(
            root,
            "coverage_combine",
            (
                coverage,
                "combine",
                "--keep",
                f"--data-file={combined}",
                str(private / ".coverage"),
                str(shard),
            ),
            120,
        ),
        _run(
            root,
            "all_changed_code_coverage_report",
            (
                coverage,
                "report",
                f"--data-file={combined}",
                f"--include={include}",
                "--show-missing",
                "--fail-under=100",
            ),
            120,
        ),
    ]
    previous = root / RAW / "terminal_candidate.json"
    full = None
    if previous.is_file():
        old = (
            json.loads(previous.read_bytes())
            .get("validation_receipts", {})
            .get("full_python_suite")
        )
        if (
            old is not None
            and (root / old["log_path"]).is_file()
            and sha256_file(root / old["log_path"]) == old["log_sha256"]
        ):
            full = old
    if full is None:
        full = _run(
            root,
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
                f"--basetemp={private / 'full'}",
            ),
            600,
        )
    validation = dict(
        frozen_scope_path=str(RAW / "frozen_scope.json"),
        affected=scoped,
        coverage_shards=extra,
        full_python_suite=full,
        repository_health=dict(
            status="healthy" if full["passed"] else "degraded_open",
            full_suite_exit=full["exit_code"],
            affects_required_checks=False,
        ),
        summaries=summaries,
        publication_gate=publication_receipt,
        required_checks_passed=scoped["required_checks_passed"] and all(r["passed"] for r in extra),
    )
    spans.append(
        span("validation", phase, time.monotonic(), len(scoped["validation_receipts"]) + 1)
    )
    progress(start, "validation", "complete", len(scoped["validation_receipts"]) + 1)

    phase = time.monotonic()
    progress(start, "terminal", "start", 0)
    candidate = build_artifact(root, publication, validation, spans, time.monotonic() - start)
    candidate_path = root / RAW / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    cold = _run(
        root,
        "cold_replay",
        (
            str(root / ".venv/bin/python"),
            "-c",
            "import json,sys; from pathlib import Path; from carnot.experiment_7808_v678_capstone import replay_candidate; "
            "p=Path(sys.argv[1]); e=replay_candidate(json.loads(p.read_text()),Path.cwd()); print(e,flush=True); sys.exit(bool(e))",
            str(candidate_path),
        ),
        120,
    )
    adversarial = _run(
        root,
        "adversarial_verify",
        (
            str(root / ".venv/bin/python"),
            "scripts/adversarial_verify.py",
            "--json",
            str(candidate_path),
        ),
        120,
    )
    strict = _run(
        root,
        "strict_row_consistency",
        (
            str(root / ".venv/bin/python"),
            "scripts/verdict_row_consistency_lint.py",
            "--strict",
            str(candidate_path),
        ),
        120,
    )
    validation["terminal_readers"] = [cold, adversarial, strict]
    validation["required_checks_passed"] = validation["required_checks_passed"] and all(
        receipt["passed"] for receipt in validation["terminal_readers"]
    )
    candidate = build_artifact(root, publication, validation, spans, time.monotonic() - start)
    candidate["flagged_adversarial"] = not adversarial["passed"]
    if not validation["required_checks_passed"]:
        candidate["verdict_class"] = "disqualified"
        candidate["honest_verdict"] = "complete_disqualified_required_validation"
        candidate["acceptance_gate_results"] = {
            key: 0 for key in candidate["acceptance_gate_results"]
        }
        candidate["rows"][-1]["verdict_class"] = "disqualified"
        candidate["rows"][-1]["honest_verdict"] = candidate["honest_verdict"]
        candidate["task_dispositions"][-1] = dict(candidate["rows"][-1])
    spans.append(span("terminal", phase, time.monotonic(), 3))
    candidate["phase_spans"] = spans
    candidate["duration_s"] = time.monotonic() - start
    atomic_json(root / OUTPUT, candidate)
    progress(start, "terminal", "complete", 3)
    return candidate


def main() -> int:
    """Expose the production run and one read-only cold replay path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--cold-replay")
    args = parser.parse_args()
    if args.cold_replay:
        errors = replay_candidate(json.loads(Path(args.cold_replay).read_text()), ROOT)
        print(json.dumps({"errors": errors}), flush=True)
        return int(bool(errors))
    result = run_experiment(ROOT, args.date)
    print(
        json.dumps(
            {"honest_verdict": result["honest_verdict"], "verdict_class": result["verdict_class"]}
        ),
        flush=True,
    )
    return 0 if result["verdict_class"] != "disqualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
