"""Produce the V683 terminal capstone (REQ-REPORT-7878)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from carnot.reporting.current_work_receipt import atomic_json, sha256_file  # noqa: E402
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands  # noqa: E402
from carnot.reporting import v683_capstone as capstone  # noqa: E402


OUTPUT = ROOT / "results/experiment_7878_v683_capstone.json"
DEFAULT_ROOT = Path("/tmp/carnot-7878-v683-20260929")
MODULE = "python/carnot/reporting/v683_capstone.py"
SCRIPT = "scripts/experiments/experiment_7878_v683_capstone.py"
TEST = "tests/python/test_experiment_7878_v683_capstone.py"


def progress(start: float, phase: str, done: int) -> None:
    """Let the supervisor see each bounded phase and completed unit."""
    print(
        f"exp7878 {phase} elapsed_s={time.monotonic() - start:.3f} completed_units={done}",
        flush=True,
    )


def manifest(output_root: Path) -> dict[str, Any]:
    """Freeze exact argv, scope and deadline before any evidence reduction."""
    py = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    cov = str(ROOT / ".venv/bin/coverage")
    ruff = str(ROOT / ".venv/bin/ruff")
    mypy = str(ROOT / ".venv/bin/mypy")
    base = ["-q", "-n", "0", "-o", "addopts=", "--no-cov"]
    include = "*/v683_capstone.py"
    unit = output_root / ".coverage.unit"
    covered = output_root / ".coverage"
    checks = [
        ("publication_gate", [py, "scripts/publication_gate.py", "--json"], "required", 60),
        (
            "worktree_imports",
            [
                py,
                "-c",
                "import importlib,json; names=['carnot.reporting.v683_capstone','carnot.reporting.v683_independent_audit','carnot.reporting.current_work_receipt','scripts.publication_gate']; print(json.dumps({'resolved_imports':{n:importlib.import_module(n).__file__ for n in names}}))",
            ],
            "required",
            60,
        ),
        (
            "affected_pytest",
            [pytest, TEST, *base, f"--basetemp={output_root / 'pytest-affected'}"],
            "required",
            180,
        ),
        (
            "full_pytest",
            [pytest, "tests/python", *base, f"--basetemp={output_root / 'pytest-full'}"],
            "required",
            900,
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
                TEST,
                *base,
                f"--basetemp={output_root / 'pytest-covered'}",
            ],
            "required",
            180,
        ),
        (
            "coverage_report",
            [
                cov,
                "report",
                f"--data-file={unit}",
                f"--include={include}",
                "--show-missing",
                "--fail-under=100",
            ],
            "required",
            60,
        ),
        ("ruff_check", [ruff, "check", MODULE, SCRIPT, TEST], "required", 60),
        ("ruff_format", [ruff, "format", "--check", MODULE, SCRIPT, TEST], "required", 60),
        ("strict_mypy", [mypy, "--strict", MODULE, SCRIPT], "required", 120),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", TEST], "required", 60),
    ]
    return {
        "schema": "carnot.exp7878.validation.v1",
        "affected_source": [MODULE, SCRIPT],
        "affected_tests": [TEST],
        "e2e_applicability": {
            "private_cli_and_cold_replay": "applicable",
            "E2E-016": "upstream protocol; capstone reads its result",
            "E2E-017": "upstream ARC; capstone reads its result",
        },
        "commands": [
            {"name": name, "argv": argv, "classification": kind, "deadline_s": deadline}
            for name, argv, kind, deadline in checks
        ],
    }


def _seal(receipt: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """Move a closed child log to an immutable path based on its bytes."""
    source = ROOT / receipt["log_path"]
    digest = sha256_file(source)
    target = output_root / "sealed_logs" / f"{receipt['name']}_{digest[7:]}.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != digest:
            raise ValueError("sealed_log_collision")
    else:
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    source.unlink()
    return {**receipt, "log_path": str(target), "log_sha256": digest}


def _run(entry: dict[str, Any], output_root: Path, start: float, done: int) -> dict[str, Any]:
    """Supervise one child with a heartbeat and retain its real exit."""
    progress(start, f"before_subprocess:{entry['name']}", done)
    spec = CommandSpec(
        entry["name"], tuple(entry["argv"]), entry["classification"], entry["deadline_s"]
    )
    raw = run_commands(ROOT, [spec], log_dir=output_root / "transient_logs", heartbeat_s=30)[0]
    receipt = _seal({**raw, "classification": entry["classification"]}, output_root)
    progress(start, f"after_subprocess:{entry['name']}", done + 1)
    return receipt


def _report(value: dict[str, Any]) -> str:
    """Render the exact ledger and bounded research decisions for review."""
    lines = [
        "# V683 research milestone — 2026-09-29",
        "",
        f"Verdict: `{value['honest_verdict']}`. Science complete: {value['milestone_evidence_complete_score']}. Measured benefit: {value['milestone_benefit_score']}.",
        "",
        "## Current evidence",
        "",
        "| Task | Status | Qualified |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {r['task_id']} | {r['status']} | {r['eligible']} |" for r in value["task_evidence_rows"]
    )
    lines.extend(
        [
            "",
            "Six required scientific producers are absent. A skip receipt is administrative. Exposed natural labels and deterministic fixtures do not establish hidden generalization.",
            "",
            "## PRD gaps",
            "",
        ]
    )
    lines.extend(
        f"- {key}: {item['decision']}. {item['reason']} Next: {item['next_question']}"
        for key, item in value["prd_gap_decisions"].items()
    )
    lines.extend(
        [
            "",
            "GAP-ORACLE-DISTINCT remains open. The September 28 retractions of Exp4245, Exp5160 and Exp5171 remain in force. DiffusionGemma is not promoted.",
            "",
            "## Stable publication gates",
            "",
        ]
    )
    lines.extend(
        f"- {gate}: {item['pass']} — {item['detail']}"
        for gate, item in value["publication_gate_results"]["gates"].items()
    )
    lines.extend(
        [
            "",
            "Scientific readiness is separate from release authorization. No publication or production activation occurred.",
            "",
            "## Retirement and continuation",
            "",
        ]
    )
    lines.extend(
        f"- {r['task_id']} versus {r['prior_experiment_id']}: {r['decision']} ({r['current_verdict']})."
        for r in value["retirement_decisions"]
        if r["identical_verdict"]
    )
    lines.extend(["", "## Literature and hardware triggers", ""])
    lines.extend(
        f"- {key}: {decision}." for key, decision in value["literature_adoption_decisions"].items()
    )
    lines.append(
        "- Require a paired live route, independent ARC firings, and authenticated new board execution before deployment claims."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run a private replay or finish one terminal reconciliation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--science-only", action="store_true")
    parser.add_argument("--check-only", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    start = time.monotonic()
    progress(start, "start", 0)
    if args.check_only:
        errors = capstone.cold_replay(args.check_only, ROOT)
        progress(start, f"cold_replay:{errors}", 1)
        return int(bool(errors))
    if args.date != "20260929":
        parser.error("run date must be 20260929")
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    frozen = manifest(output_root)
    manifest_path = output_root / "validation_command_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_bytes()) != frozen:
        raise ValueError("frozen_validation_manifest_changed")
    atomic_json(manifest_path, frozen)
    progress(start, "preconditions_begin", 0)
    value = capstone.build_candidate(ROOT, args.date)
    value["validation_command_manifest_path"] = str(manifest_path)
    value["validation_command_manifest_sha256"] = sha256_file(manifest_path)
    value["field_principles"]["validation_command_manifest_sha256"] = (
        "Bind argv, class and deadline before measurement."
    )
    checkpoint = (
        output_root
        / "checkpoints"
        / value["reproducibility_checksum"].split(":", 1)[1]
        / "ledger.json"
    )
    stable = {
        key: value[key]
        for key in ("task_evidence_rows", "rows", "gate_check_summary", "source_artifact_hashes")
    }
    if checkpoint.is_file() and json.loads(checkpoint.read_bytes()) != stable:
        raise ValueError("hash_bound_checkpoint_changed")
    atomic_json(checkpoint, stable)
    candidate = output_root / "candidate.json"
    atomic_json(candidate, value)
    progress(start, "preconditions_end", 14)
    if args.science_only:
        return 0
    receipts: list[dict[str, Any]] = []
    for entry in frozen["commands"]:
        item = _run(entry, output_root, start, len(receipts))
        receipts.append(item)
        value["validation_receipts"] = receipts
        value["observed_child_commands"] = receipts
        if entry["name"] == "worktree_imports" and item["passed"]:
            resolved = json.loads(Path(item["log_path"]).read_text())["resolved_imports"]
            value["resolved_imports"] = resolved
        atomic_json(candidate, value)
    failed = [r for r in receipts if r["classification"] == "required" and not r["passed"]]
    for item in failed:
        value["gate_check_summary"].append(
            {
                "upstream_id": "Exp7878",
                "path": item["log_path"],
                "hash": item["log_sha256"],
                "artifact_field": f"validation_receipts.{item['name']}",
                "op": "==",
                "expected": 0,
                "observed": item["exit_code"],
            }
        )
    if failed:
        value.update(
            honest_verdict="complete_disqualified_required_v683_validation",
            verdict_class="disqualified",
        )
        value["repository_health"]["status"] = "failed_owned_validation"
        value["acceptance_gate_results"]["readiness"] = 0
    else:
        value["repository_health"]["status"] = "passed"
        value["capstone_execution_ready_score"] = 1
        value["acceptance_gate_results"]["readiness"] = 1
    value["duration_s"] = time.monotonic() - start
    value["phase_spans"].append(
        {
            "phase": "owned_validation",
            "duration_s": value["duration_s"] - value["phase_spans"][0]["duration_s"],
            "completed_units": len(receipts),
        }
    )
    for key in value:
        value["field_principles"].setdefault(
            key, "Keep the terminal result and its scope auditable."
        )
    atomic_json(candidate, value)
    progress(start, "cold_replay_begin", len(receipts))
    replay_errors = capstone.cold_replay(candidate, ROOT)
    if replay_errors:
        value.update(
            honest_verdict="complete_disqualified_cold_replay",
            verdict_class="disqualified",
            capstone_execution_ready_score=0,
        )
        value["acceptance_gate_results"]["readiness"] = 0
        value["cold_replay_errors"] = replay_errors
        value["field_principles"]["cold_replay_errors"] = (
            "A changed input or primitive row disqualifies readiness."
        )
    progress(start, "cold_replay_end", len(receipts) + 1)
    atomic_json(candidate, value)
    terminal = [
        {
            "name": "adversarial_verify",
            "argv": [
                str(ROOT / ".venv/bin/python"),
                "scripts/adversarial_verify.py",
                "--json",
                str(candidate),
            ],
            "classification": "required",
            "deadline_s": 60,
        },
        {
            "name": "strict_rows",
            "argv": [
                str(ROOT / ".venv/bin/python"),
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ],
            "classification": "required",
            "deadline_s": 60,
        },
    ]
    terminal_receipts = [
        _run(entry, output_root, start, len(receipts) + i) for i, entry in enumerate(terminal)
    ]
    value["validation_receipts"].extend(terminal_receipts)
    value["observed_child_commands"] = value["validation_receipts"]
    try:
        report = json.loads(Path(terminal_receipts[0]["log_path"]).read_text())
        value["flagged_adversarial"] = bool(report["flagged_count"])
    except (OSError, ValueError, KeyError):
        value["flagged_adversarial"] = True
    bad_terminal = [r for r in terminal_receipts if not r["passed"]]
    if bad_terminal or value["flagged_adversarial"]:
        value.update(
            honest_verdict="complete_disqualified_terminal_verification",
            verdict_class="disqualified",
            capstone_execution_ready_score=0,
        )
        value["acceptance_gate_results"]["readiness"] = 0
        for item in bad_terminal:
            value["gate_check_summary"].append(
                {
                    "upstream_id": "Exp7878",
                    "path": item["log_path"],
                    "hash": item["log_sha256"],
                    "artifact_field": f"validation_receipts.{item['name']}",
                    "op": "==",
                    "expected": 0,
                    "observed": item["exit_code"],
                }
            )
    value["duration_s"] = time.monotonic() - start
    for key in value:
        value["field_principles"].setdefault(
            key, "Keep the terminal result and its scope auditable."
        )
    atomic_json(candidate, value)
    note = ROOT / value["report_path"]
    note.parent.mkdir(parents=True, exist_ok=True)
    temporary = note.with_name(f".{note.name}.tmp-{os.getpid()}")
    temporary.write_text(_report(value))
    temporary.replace(note)
    atomic_json(OUTPUT, value)
    progress(start, f"published:{value['honest_verdict']}", len(value["validation_receipts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
