"""Publish the V682 capstone after bounded, owned validation.

REQ-REPORT-7864-V682. Science-only and check-only routes support private replay.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from carnot.reporting.current_work_receipt import atomic_json, sha256_file  # noqa: E402
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands  # noqa: E402
from carnot.reporting import v682_capstone as capstone  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/experiment_7864_v682_capstone.json"
SCRATCH = Path("/tmp/carnot-7864")


def progress(start: float, phase: str, completed: int) -> None:
    """Expose phase boundaries and the count so a stalled child is visible."""
    print(
        f"exp7864 {phase} elapsed_s={time.monotonic() - start:.3f} completed_units={completed}",
        flush=True,
    )


def commands(root: Path, scratch: Path) -> list[CommandSpec]:
    """Freeze exact argv and deadlines before reading current outcomes."""
    py = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    cov = str(root / ".venv/bin/coverage")
    ruff = str(root / ".venv/bin/ruff")
    mypy = str(root / ".venv/bin/mypy")
    module = "python/carnot/reporting/v682_capstone.py"
    script = "scripts/experiments/experiment_7864_v682_capstone.py"
    test = "tests/python/test_experiment_7864_v682_capstone.py"
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    include = "*/v682_capstone.py,*/experiment_7864_v682_capstone.py"
    required = [
        ("publication_gate", (py, "scripts/publication_gate.py", "--json"), 60),
        (
            "worktree_imports",
            (
                py,
                "-c",
                "import importlib,json; n=['carnot.reporting.v682_capstone','carnot.reporting.current_work_receipt','carnot.reporting.roadmap_contract']; print(json.dumps({'resolved_imports':{x:importlib.import_module(x).__file__ for x in n}}))",
            ),
            60,
        ),
        (
            "affected_pytest",
            (pytest, *common, test, "-q", f"--basetemp={scratch / 'pytest-affected'}"),
            180,
        ),
        (
            "coverage_unit",
            (
                cov,
                "run",
                f"--data-file={scratch / '.coverage.unit'}",
                f"--include={include}",
                "-m",
                "pytest",
                *common,
                test,
                "-q",
                f"--basetemp={scratch / 'pytest-covered'}",
            ),
            180,
        ),
        (
            "cli_e2e",
            (
                cov,
                "run",
                f"--data-file={scratch / '.coverage.cli'}",
                f"--include={include}",
                script,
                "--date",
                "20260929",
                "--science-only",
                "--output",
                str(scratch / "cli-candidate.json"),
            ),
            120,
        ),
        (
            "coverage_combine",
            (
                cov,
                "combine",
                "--keep",
                f"--data-file={scratch / '.coverage'}",
                str(scratch / ".coverage.unit"),
                str(scratch / ".coverage.cli"),
            ),
            60,
        ),
        (
            "changed_coverage",
            (
                cov,
                "report",
                f"--data-file={scratch / '.coverage'}",
                f"--include={include}",
                "--show-missing",
                "--fail-under=100",
            ),
            60,
        ),
        ("ruff_check", (ruff, "check", module, script, test), 60),
        ("ruff_format", (ruff, "format", "--check", module, script, test), 60),
        ("mypy", (mypy, "--strict", module, script), 120),
        ("scoped_spec", (py, "scripts/check_spec_coverage.py", test), 60),
        ("cold_replay", (py, script, "--check-only", str(scratch / "candidate.json")), 60),
        (
            "adversarial_verify",
            (py, "scripts/adversarial_verify.py", "--json", str(scratch / "candidate.json")),
            60,
        ),
        (
            "strict_rows",
            (
                py,
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(scratch / "candidate.json"),
            ),
            60,
        ),
    ]
    result = [
        CommandSpec(name, tuple(argv), "required", deadline) for name, argv, deadline in required
    ]
    result.append(
        CommandSpec(
            "repository_health_180s",
            (pytest, "tests/python", "-q", *common, f"--basetemp={scratch / 'pytest-health'}"),
            "diagnostic",
            180,
        )
    )
    return result


def seal(receipt: dict[str, Any], scratch: Path, attempt: str) -> dict[str, Any]:
    """Seal a closed log under its content hash after the child exits."""
    original = ROOT / receipt["log_path"]
    digest = sha256_file(original)
    target = scratch / "validation_logs" / attempt / f"{receipt['name']}_{digest[7:]}.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(original.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    original.unlink()
    return {**receipt, "log_path": str(target), "log_sha256": digest}


def run_one(
    spec: CommandSpec, scratch: Path, start: float, count: int, attempt: str
) -> dict[str, Any]:
    """Use the shared child supervisor and record the actual exit and log."""
    progress(start, f"before_subprocess:{spec.name}", count)
    raw = run_commands(
        ROOT, [spec], log_dir=scratch / "transient_logs" / attempt, heartbeat_s=30.0
    )[0]
    receipt = seal({**raw, "classification": spec.scope}, scratch, attempt)
    progress(start, f"after_subprocess:{spec.name}", count + 1)
    return receipt


def write_note(value: dict[str, Any], path: Path) -> None:
    """Render the same fourteen rows used in the machine result."""
    lines = [
        "# V682 capstone — 2026-09-29",
        "",
        f"Verdict: `{value['honest_verdict']}`. Milestone evidence complete: {value['milestone_evidence_complete_score']}; registered benefit: {value['milestone_benefit_score']}.",
        "",
        "The capstone reconciles all fourteen tasks. Missing producers and failed required validation keep the milestone blocked. Fixture agreement is circular; natural annotations remain exposed development evidence. No new LLM weight training, fresh holdout generalization, ARC solve, or board execution is claimed.",
        "",
        "| Task | Status | Exact verdict |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['task_id']} | {row['status']} | {row['honest_verdict'] or 'missing'} |"
        for row in value["task_disposition_rows"]
    )
    lines.extend(["", "## Narrow retirement", ""])
    lines.extend(
        f"- {row['task_id']} repeats {row['prior_experiment_id']} with `{row['current_verdict']}`; retire only the repeated protocol scope."
        for row in value["retirement_rows"]
        if row["identical_verdict"]
    )
    lines.extend(["", "## Next decisions", ""])
    lines.extend(f"- {row['branch']}: {row['decision']}" for row in value["next_decision_rows"])
    lines.extend(["", "## Publication gates", ""])
    lines.extend(
        f"- {row['gate']}: {row['pass']} (stable publication scope; no V682 benefit inference)."
        for row in value["publication_gate_rows"]
    )
    lines.extend(
        [
            "",
            "GAP-ORACLE-DISTINCT remains open. The September 28 corrigendum and pending DiffusionGemma work remain in force. The repository health diagnostic is separate from required checks; historical failed obligations stay open. No publication or roadmap activation occurred.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)


def main() -> int:
    """Run private replay or complete one bounded terminal reconciliation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--science-only", action="store_true")
    parser.add_argument("--check-only", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-root", type=Path, default=SCRATCH)
    args = parser.parse_args()
    start = time.monotonic()
    progress(start, "start", 0)
    if args.check_only:
        errors = capstone.cold_replay(args.check_only, ROOT)
        progress(start, f"cold_replay:{errors}", 1)
        return int(bool(errors))
    if args.date != "20260929":
        parser.error("run date must be 20260929")
    scratch = args.output_root
    scratch.mkdir(parents=True, exist_ok=True)
    frozen = commands(ROOT, scratch)
    manifest = scratch / "validation_command_manifest.json"
    atomic_json(
        manifest,
        {
            "schema": "carnot.exp7864.validation.v1",
            "commands": [
                {
                    "name": spec.name,
                    "argv": list(spec.argv),
                    "classification": spec.scope,
                    "deadline_s": spec.timeout_s,
                }
                for spec in frozen
            ],
        },
    )
    progress(start, "preconditions_begin", 0)
    result = capstone.build_candidate(ROOT, args.date, time.monotonic() - start)
    result["validation_command_manifest_path"] = str(manifest)
    result["validation_command_manifest_sha256"] = sha256_file(manifest)
    result["field_principles"]["validation_command_manifest_sha256"] = (
        "Freeze required argv and deadlines before reading outcomes."
    )
    progress(start, "preconditions_end", len(result["task_disposition_rows"]))
    checkpoint = (
        scratch
        / "checkpoints"
        / result["reproducibility_checksum"].split(":", 1)[1]
        / "ledger.json"
    )
    stable = {
        key: result[key]
        for key in ("task_disposition_rows", "rows", "gate_check_summary", "source_artifact_hashes")
    }
    if checkpoint.exists():
        if json.loads(checkpoint.read_bytes()) != stable:
            raise ValueError("checkpoint_observation_changed")
    else:
        atomic_json(checkpoint, stable)
    candidate = args.output or scratch / "candidate.json"
    atomic_json(candidate, result)
    progress(start, "candidate_written", 14)
    if args.science_only:
        return 0
    attempt = uuid.uuid4().hex
    receipts: list[dict[str, Any]] = []
    required_failed = False
    for spec in frozen:
        if spec.name in {"adversarial_verify", "strict_rows"} and not required_failed:
            result["capstone_execution_ready_score"] = 1
            result["acceptance_gate_results"]["readiness"] = 1
        result["duration_s"] = time.monotonic() - start
        atomic_json(scratch / "candidate.json", result)
        receipt = run_one(spec, scratch, start, len(receipts), attempt)
        receipts.append(receipt)
        result["validation_receipts"] = receipts
        result["observed_child_commands"] = receipts
        if spec.name == "publication_gate" and receipt["passed"]:
            report = json.loads(Path(receipt["log_path"]).read_text())
            result["publication_gate_rows"] = [
                {
                    "gate": gate,
                    "pass": report["gates"][gate]["pass"],
                    "detail": report["gates"][gate]["detail"],
                    "scope": "stable_publication_gate_not_v682_benefit",
                }
                for gate in ("G1", "G2", "G3", "G4")
            ]
        if spec.name == "worktree_imports" and receipt["passed"]:
            result["resolved_imports"].update(
                json.loads(Path(receipt["log_path"]).read_text())["resolved_imports"]
            )
        if spec.name == "adversarial_verify":
            try:
                result["flagged_adversarial"] = bool(
                    json.loads(Path(receipt["log_path"]).read_text())["flagged_count"]
                )
            except (ValueError, KeyError):
                result["flagged_adversarial"] = True
        if spec.name == "repository_health_180s":
            result["repository_health"].update(
                {"status": "passed" if receipt["passed"] else "failed", "receipt": receipt}
            )
        elif not receipt["passed"] or (
            spec.name == "adversarial_verify" and result["flagged_adversarial"]
        ):
            required_failed = True
            result["gate_check_summary"].append(
                {
                    "upstream_id": "Exp7864",
                    "path": receipt["log_path"],
                    "hash": receipt["log_sha256"],
                    "artifact_field": f"validation_receipts.{spec.name}",
                    "op": "==",
                    "expected": 0,
                    "observed": receipt["exit_code"],
                }
            )
    result["duration_s"] = time.monotonic() - start
    result["phase_spans"].append(
        {
            "phase": "owned_validation",
            "duration_s": result["duration_s"] - result["phase_spans"][0]["duration_s"],
            "completed_units": len(receipts),
        }
    )
    if required_failed:
        result.update(
            honest_verdict="complete_disqualified_required_validation",
            verdict_class="disqualified",
            capstone_execution_ready_score=0,
        )
        result["acceptance_gate_results"]["readiness"] = 0
    else:
        result["capstone_execution_ready_score"] = 1
        result["acceptance_gate_results"]["readiness"] = 1
    write_note(result, ROOT / result["capstone_doc_path"])
    atomic_json(OUTPUT, result)
    progress(start, f"complete:{result['honest_verdict']}", len(receipts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
