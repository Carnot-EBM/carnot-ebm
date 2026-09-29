#!/usr/bin/env python3
"""Run the V684 administrative capstone. REQ-REPORT-7890-V684."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from carnot.reporting.current_work_receipt import atomic_json, sha256_file  # noqa: E402
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands  # noqa: E402
from carnot.reporting.v684_capstone import build_candidate, cold_replay  # noqa: E402


MODEL_SPECS: list[str] = []
OWNED = (
    "python/carnot/reporting/v684_capstone.py",
    "scripts/experiments/experiment_7890_v684_capstone.py",
)
TESTS = (
    "tests/python/test_experiment_7890_v684_capstone.py",
    "tests/python/test_experiment_7879_v684_contract_methods.py",
    "tests/python/test_experiment_7889_v684_hardware_evidence.py",
    "tests/python/test_experiment_7303_v642_validation_scope.py",
    "tests/python/test_experiment_7837_v681_contract_methods.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_publication_gate.py",
)
START = time.monotonic()


def progress(phase: str, completed: int) -> None:
    """Make an outstanding CPU child visible to the conductor."""
    print(
        f"[exp7890] {phase} elapsed_s={time.monotonic() - START:.3f} completed_units={completed}",
        flush=True,
    )


def commands(private: Path) -> list[CommandSpec]:
    """Freeze actual file paths and exact argv before observing child results."""
    for name in (*OWNED, *TESTS):
        if not (ROOT / name).is_file():
            raise ValueError(f"affected_scope_missing:{name}")
    py = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    cov = str(ROOT / ".venv/bin/coverage")
    ruff = str(ROOT / ".venv/bin/ruff")
    mypy = str(ROOT / ".venv/bin/mypy")
    include = ",".join(str(ROOT / name) for name in OWNED)
    common = ["-q", "-n", "0", "-o", "addopts=", "--no-cov"]
    unit = private / "unit.coverage"
    success = private / "success.coverage"
    missing = private / "missing.coverage"
    replay = private / "replay.coverage"
    combined = private / "combined.coverage"
    script = str(ROOT / OWNED[1])
    shard_code = (
        "from coverage import Coverage; import sys; "
        "files=sys.argv[1:]; "
        "assert all(__import__('pathlib').Path(p).is_file() for p in files); "
        "data=[Coverage(data_file=p) for p in files]; "
        "[item.load() for item in data]; "
        "assert all(any(item.get_data().lines(name) for name in item.get_data().measured_files()) for item in data)"
    )
    import_code = (
        "from pathlib import Path; import json; "
        "from carnot.reporting.v684_capstone import resolve_imports; "
        f"print(json.dumps(resolve_imports(Path({str(ROOT)!r}))))"
    )
    raw: list[tuple[str, list[str], float, str]] = [
        ("package_root_imports", [py, "-u", "-c", import_code], 30, "required"),
        (
            "affected_pytest",
            [pytest, *TESTS, *common, f"--basetemp={private / 'affected'}"],
            300,
            "required",
        ),
        (
            "coverage_unit",
            [
                cov,
                "run",
                f"--data-file={unit}",
                f"--include={include}",
                "-m",
                "pytest",
                TESTS[0],
                *common,
                f"--basetemp={private / 'covered'}",
            ],
            180,
            "required",
        ),
        (
            "cli_success",
            [
                cov,
                "run",
                f"--data-file={success}",
                f"--include={include}",
                script,
                "--date",
                "20260929",
                "--output",
                str(private / "success.json"),
                "--evidence-only",
            ],
            60,
            "required",
        ),
        (
            "cli_missing_input",
            [
                cov,
                "run",
                f"--data-file={missing}",
                f"--include={include}",
                script,
                "--date",
                "20260929",
                "--root",
                str(private / "absent"),
                "--output",
                str(private / "missing.json"),
                "--evidence-only",
            ],
            60,
            "expected_failure",
        ),
        (
            "cold_replay",
            [
                cov,
                "run",
                f"--data-file={replay}",
                f"--include={include}",
                script,
                "--cold-replay",
                str(private / "success.json"),
            ],
            60,
            "required",
        ),
        (
            "coverage_shards",
            [py, "-u", "-c", shard_code, str(unit), str(success), str(missing), str(replay)],
            30,
            "required",
        ),
        (
            "coverage_combine",
            [
                cov,
                "combine",
                "--keep",
                f"--data-file={combined}",
                str(unit),
                str(success),
                str(missing),
                str(replay),
            ],
            30,
            "required",
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
            "required",
        ),
        ("ruff_check", [ruff, "check", *OWNED, TESTS[0]], 30, "required"),
        ("ruff_format", [ruff, "format", "--check", *OWNED, TESTS[0]], 30, "required"),
        ("strict_mypy", [mypy, "--strict", *OWNED], 90, "required"),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", TESTS[0]], 30, "required"),
        (
            "e2e_016_fixture",
            [
                py,
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--fixture-e2e",
                str(private / "e2e016.json"),
            ],
            180,
            "required",
        ),
        (
            "e2e_016_replay",
            [
                py,
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--cold-replay",
                str(private / "e2e016.json"),
            ],
            90,
            "required",
        ),
        (
            "e2e_017",
            [
                pytest,
                "tests/python/test_arc_supervisor_delta_7874.py",
                *common,
                f"--basetemp={private / 'e2e017'}",
            ],
            180,
            "required",
        ),
    ]
    return [CommandSpec(name, tuple(argv), scope, deadline) for name, argv, deadline, scope in raw]


def seal_receipts(receipts: list[dict[str, Any]], private: Path) -> None:
    """Move closed child logs to immutable hash-addressed names."""
    sealed = private / "sealed_logs"
    sealed.mkdir(exist_ok=True)
    for receipt in receipts:
        source = Path(receipt["log_path"])
        digest = sha256_file(source).removeprefix("sha256:")
        target = sealed / f"{receipt['name']}_{digest}.log"
        if target.is_file() and sha256_file(target) != receipt["log_sha256"]:
            raise ValueError("sealed_log_collision")
        if not target.is_file():
            shutil.copyfile(source, target)
        receipt["log_path"] = str(target)
        receipt["log_sha256"] = sha256_file(target)
        receipt["command_argv"] = receipt.get("command_argv", [])
        receipt["classification"] = receipt.pop("scope")


def report_text(value: dict[str, Any]) -> str:
    """Leave falsifiable next measurements beside the terminal artifact."""
    lines = [
        "# Milestone 2026.09.684 decisions",
        "",
        "Date: 2026-09-29.",
        "",
        f"Verdict: `{value['honest_verdict']}`. Six current scientific producers are required.",
        "",
        "The source corpus is exposed development data. A fixture is circular evidence.",
        "GAP-ORACLE-DISTINCT remains open. The September 28 retractions for",
        "Exp4245, Exp5160 and Exp5171 remain in force. DiffusionGemma stays gated.",
        "",
    ]
    for gap, item in value["prd_gap_decisions"].items():
        lines.extend(
            [
                f"## {gap}",
                "",
                f"Decision: {item['decision']}.",
                "",
                "Benefit, retention and efficiency are null until qualified measurements exist.",
                f"Falsifiable next question: {item['next_question']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope and prerequisites",
            "",
            "The six scientific producers are Exp7882, Exp7883, Exp7884, Exp7885, Exp7886 and Exp7888.",
            "A same-verdict retirement needs the identical scope and technique. The new",
            "calibration estimand is distinct from the skipped scheduling experiment.",
            "Historical failures remain in the JSON ledger. FoVer G1-G4 are unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def run_current(root: Path, date: str, output: Path, private: Path) -> dict[str, Any]:
    """Freeze current checks, supervise children, then check exact terminal bytes."""
    progress("preflight_before_compute", 0)
    manifest = commands(private)
    manifest_path = private / "validation_command_manifest.json"
    atomic_json(
        manifest_path,
        {
            "affected_scope": list(TESTS),
            "changed_files": list(OWNED),
            "closure_reason": "New ledger and CLI plus their imported contract, receipt, gate, scope, and current V684 consumers. V683 runners require absent staged V683 authority and are historical debt.",
            "inapplicable_e2e": [f"E2E-{n:03d}" for n in range(1, 16)],
            "applicable_e2e": ["E2E-016", "E2E-017"],
            "commands": [
                {
                    "name": c.name,
                    "argv": list(c.argv),
                    "classification": c.scope,
                    "deadline_s": c.timeout_s,
                }
                for c in manifest
            ],
        },
    )
    value = build_candidate(root, date)
    value["validation_command_manifest_path"] = str(manifest_path)
    progress("before_required_subprocesses", 12)
    receipts = run_commands(root, manifest, log_dir=private / "logs", heartbeat_s=30)
    seal_receipts(receipts, private)
    for item in receipts:
        if item["name"] == "cli_missing_input":
            item["passed"] = item["exit_code"] != 0 and "FileNotFoundError" in item["output_tail"]
    value["validation_receipts"] = receipts
    value["observed_child_commands"] = [item["command_argv"] for item in receipts]
    failed = [
        item["name"]
        for item in receipts
        if item["classification"] == "required" and not item["passed"]
    ]
    failed.extend(
        item["name"]
        for item in receipts
        if item["classification"] == "expected_failure" and not item["passed"]
    )
    if failed:
        value["honest_verdict"] = "complete_disqualified_required_v684_validation"
        value["verdict_class"] = "disqualified"
        value["capstone_execution_ready_score"] = 0
        value["acceptance_gate_results"]["validity"] = False
        value["repository_health"]["owned_failed_checks"] = failed
    else:
        value["capstone_execution_ready_score"] = 1
        value["acceptance_gate_results"]["validity"] = True
        value["acceptance_gate_results"]["readiness"] = 1
    value["duration_s"] = time.monotonic() - START
    value["phase_spans"].append(
        {
            "phase": "required_validation",
            "duration_s": sum(r["duration_s"] for r in receipts),
            "completed_units": len(receipts),
        }
    )
    value["field_principles"]["terminal_validation_report_paths"] = (
        "Terminal reports are sidecars to avoid self-hashing."
    )
    value["terminal_validation_report_paths"] = {
        "adversarial": str(private / "terminal_adversarial.json"),
        "strict_rows": str(private / "terminal_strict_rows.log"),
    }
    report = root / value["report_path"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(report_text(value))
    progress("before_terminal_validation", 12 + len(receipts))
    candidate = private / "terminal_candidate.json"
    atomic_json(candidate, value)
    if cold_replay(candidate, root):
        raise ValueError("terminal_cold_replay_failed")
    py = str(root / ".venv/bin/python")
    terminal = [
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
    checked = run_commands(root, terminal, log_dir=private / "terminal_logs", heartbeat_s=30)
    seal_receipts(checked, private)
    adversarial = json.loads(Path(checked[0]["log_path"]).read_text())
    value["flagged_adversarial"] = adversarial.get("flagged_count", 0) != 0
    if value["flagged_adversarial"]:
        value["honest_verdict"] = "complete_disqualified_adversarial_flags"
        value["verdict_class"] = "disqualified"
        value["capstone_execution_ready_score"] = 0
        value["acceptance_gate_results"]["readiness"] = 0
    if not all(item["passed"] for item in checked):
        value["honest_verdict"] = "complete_disqualified_terminal_validation"
        value["verdict_class"] = "disqualified"
        value["capstone_execution_ready_score"] = 0
        value["acceptance_gate_results"]["readiness"] = 0
    # Reports have no candidate hash, so their paths remain outside its checksum.
    Path(value["terminal_validation_report_paths"]["adversarial"]).write_text(
        json.dumps(adversarial, indent=2)
    )
    Path(value["terminal_validation_report_paths"]["strict_rows"]).write_bytes(
        Path(checked[1]["log_path"]).read_bytes()
    )
    if value != json.loads(candidate.read_text()):
        atomic_json(candidate, value)
        checked = run_commands(
            root, terminal, log_dir=private / "terminal_recheck_logs", heartbeat_s=30
        )
        if not all(item["passed"] for item in checked):
            raise ValueError("terminal_recheck_failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, json.loads(candidate.read_text()))
    progress("after_terminal_publication", 12 + len(receipts) + len(checked))
    return value


def main() -> int:
    """Provide a private evidence route and a supervised terminal run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args()
    progress("start", 0)
    if args.cold_replay:
        errors = cold_replay(args.cold_replay, args.root)
        print(json.dumps({"cold_replay_errors": errors}), flush=True)
        return int(bool(errors))
    output = args.output or args.root / "results/experiment_7890_v684_capstone.json"
    if args.evidence_only:
        atomic_json(output, build_candidate(args.root, args.date))
        progress("evidence_only_complete", 12)
        return 0
    private = Path(tempfile.mkdtemp(prefix="carnot-7890-v684-"))
    run_current(args.root, args.date, output, private)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
