"""REQ-VERIFY-8373: bounded real validation qualifies a single atomic capstone primary."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time

import yaml
from typing import Any
from unittest.mock import patch

from carnot.reporting import v721_capstone_evidence as e
from carnot.reporting import v717_contract_runner as base
from carnot.reporting import v718_replay_runner as terminal
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """The existing scoped manifest covers new statements and unchanged authority consumers."""
    with patch.object(base, "m", e):
        plan = list(base.manifest(private))
    plan[0]["deadline"] = 900
    for name, tests in [
        ("private_E2E021", ["tests/python/test_restricted_decision_audit_8210.py"]),
        (
            "current_consumers",
            [
                "tests/python/test_v718_contract_replay_8318.py::test_finding_policy",
                "tests/python/test_v721_contract_methods_8360.py::test_private_e2e018",
                "tests/python/test_v717_capstone_8317.py::test_append_only_retirement",
            ],
        ),
    ]:
        plan.append(
            dict(
                name=name,
                argv=[
                    str(e.ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    *tests,
                    "--basetemp=" + str(private / name),
                ],
                expected=0,
                deadline=300,
                scope="owned",
            )
        )
    return plan


def controls(value: Json, raw: Path) -> list[Json]:
    """Real fresh processes reject negative and rehashed authority, result and source claims."""
    receipts = []
    for name in ["valid", "negative", "rehashed_result", "wrong_authority", "changed_source"]:
        changed = deepcopy(value)
        if name == "negative":
            changed["reproducibility_checksum"] = "invalid"
        elif name == "rehashed_result":
            changed["H1"]["intended_count"] = 96
        elif name == "wrong_authority":
            changed["canonical_tasks_sha256"] = "wrong"
        elif name == "changed_source":
            changed["source_artifact_hashes"][0]["sha256"] = "sha256:wrong"
        if name not in ["valid", "negative"]:
            changed["reproducibility_checksum"] = canonical_hash(
                {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
            )
        path = raw / (name + ".json")
        atomic_json(path, changed)
        receipts.append(
            child(
                "cold_" + name,
                [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
                raw / "logs",
                expected=int(name != "valid"),
                deadline=240,
                heartbeat=20,
            )
        )
    return receipts


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, failed: bool = False
) -> int:
    """Missing external inputs block, while actual owned child failures disqualify the capstone."""
    started = time.monotonic_ns()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    plan = manifest(private)
    if control:
        plan = [
            dict(
                name="actual_private_child",
                argv=[
                    sys.executable,
                    "-u",
                    "-c",
                    "print('real mechanical child', flush=True); raise SystemExit("
                    + str(int(failed))
                    + ")",
                ],
                expected=0,
                deadline=10,
                scope="owned",
            )
        ]
    atomic_json(
        raw / "command_manifest.json",
        dict(
            commands=plan,
            task_cap_s=4800,
            heartbeat_s=20,
            frozen_before_measurement_ns=time.monotonic_ns(),
            private_scratch=str(private),
            private_control=control,
        ),
    )
    progress_checks = [
        e.gate(
            str(private),
            "disk_capacity_at_least_1GiB",
            True,
            shutil.disk_usage(private).free >= 1024**3,
        )
    ]
    progress_checks.append(
        e.gate(
            str(private),
            "private_scratch_rw_and_mode",
            True,
            (private / "coverage.ini").is_file() and private.stat().st_mode & 0o077 == 0,
        )
    )
    progress_checks.extend(
        e.gate(
            str(e.ROOT / ".venv/bin" / tool),
            "required_tool_executable",
            True,
            os.access(e.ROOT / ".venv/bin" / tool, os.X_OK),
        )
        for tool in ["python", "pytest", "coverage", "ruff", "mypy"]
    )
    with patch.object(base, "m", e):
        progress_checks.extend(base.preflight(plan))
    work = e.measure(root, raw)
    work["started_monotonic_ns"] = started
    work["checks"].extend(progress_checks)
    work["code_refs"] = [
        e.freeze(e.ROOT / path, raw / "code")
        for path in [
            *e.OWNED,
            e.TEST,
            "python/carnot/reporting/primary_publication.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "scripts/publication_gate.py",
        ]
    ]
    work["code_refs"].append(e.freeze(raw / "command_manifest.json", raw / "code"))
    receipts = execute(plan, raw / "validation")
    coverage = private / "coverage.json"
    if coverage.is_file():
        work["code_refs"].append(e.freeze(coverage, raw / "coverage"))
    publication = child(
        "publication_gate",
        [sys.executable, "-u", str(e.ROOT / "scripts/publication_gate.py"), "--json"],
        raw / "publication",
        deadline=90,
        scope="publication",
    )
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["publication"] = json.loads(Path(publication["stdout_path"]).read_bytes())
    receipts.append(publication)
    value = e.build(work, receipts, raw, output)
    receipts.extend(controls(value, raw / "controls"))
    candidate = raw / "audit_candidate.json"
    atomic_json(candidate, e.build(work, receipts, raw, output))
    with patch.object(terminal, "e", e):
        audit = terminal.audit(candidate, raw / "audit", {})
        work["ended_monotonic_ns"] = time.monotonic_ns()
        work["adversarial_findings"] = audit["findings"]
        receipts.append(audit["receipt"])
        terminal.publish(e.build(work, receipts, raw, output), output, raw)
    e.progress("published", 1, 0)
    return 0


def append_retirements(output: Path, manifest_path: Path) -> None:
    """Append narrow authenticated records while preserving every original manifest byte."""
    value = json.loads(output.read_bytes())
    existing = yaml.safe_load(manifest_path.read_bytes())["retired"]
    additions = [
        dict(
            experiment_id=r["task_id"],
            completed_milestone=e.MILESTONE,
            experiment_scope=r["scope"],
            exact_verdict=r["exact_verdict"],
            prior_task_id=r["prior_task_id"],
            reason="authenticated exact-verdict unchanged-inspection retirement",
            reopening_condition=r["reopening_condition"],
            hypothesis_retired=False,
            retired_by_artifact=str(output),
            retired_by_sha256=sha256_file(output),
        )
        for r in value["retirements"]
        if not any(
            old.get("experiment_id") == r["task_id"]
            and old.get("retired_by_artifact") == str(output)
            for old in existing
        )
    ]
    if additions:
        with manifest_path.open("a") as stream:
            stream.write(yaml.safe_dump(additions, sort_keys=False))


def write_note(output: Path) -> None:
    """A reader can see each continuation condition without confusing custody with benefit."""
    value = json.loads(output.read_bytes())
    path = e.ROOT / "docs/research-notes/v721-capstone.md"
    lines = [
        "# V721 capstone — 2026-10-10",
        "",
        "Verdict: " + value["honest_verdict"] + ".",
        "",
        f"Fourteen slots: {value['actual_executed_task_count']} executed including this capstone; {value['pre_gate_count']} conductor pre-gates; {value['missing_output_count']} absent outputs.",
        "",
        "Qualified Exp8361 is the sole current H1/H2 source. Frozen exposed-development nulls preserve H1 128/97 and H2 88/67 sources, stream96, retention32/23 at windows0/32/64/96. Both generalization scores remain zero. Original Exp8350/8351 audit failures remain disqualified.",
        "",
        "Continue authenticated custody and changed deployment evidence. Retire only the qualified frozen training procedure and delayed update budget. Defer unmeasured atomic recovery, actual table training, native parity and full service costs. No runtime or audit failure retires joint evidence reasoning. Zero action disagreement cannot prove semantic correctness; a CPU kernel cannot establish whole-service benefit.",
        "",
        "| Task | Original disposition | Continue / retire / defer condition |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {row['task_id']} | {row['honest_verdict'] or row['disposition']} | {condition} |"
        for row, condition in zip(value["rows"], e.NEXT, strict=True)
    )
    lines.extend(
        [
            "",
            "Three PRD gaps remain open: useful verified decisions; later learning and retention; complete request-scale deployment.",
            "",
            "ARC needs authenticated cross-game outcomes with overlapping support. KV260 remains quadratic Ising k<=5 only. PolarFire remains board-local Linux CPU with no fabric claim. GateMate requires exact recovered history and dated physical change, IDCODE0x20000001, n16 flash and device sample/hash smoke.",
            "",
            "Exp8356 replay drift and Exp8358 incomplete closure remain historical failures. No current patched replay promotes them. Repository health remains the separately recorded V720 quota failure.",
            "",
            f"Publication: G1={value['g1']}, G2={value['g2']}, G3={value['g3']}, G4={value['g4']}, paper_ready={value['paper_ready']}; unmet={value['unmet_gates']}. No external publication, model calls, generator updates or board retries.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Private controls use real CLI execution without publishing invented scientific evidence."""
    e.progress("start_no_model_load_current_LLM_calls_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261010"], default="20261010")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--failed-child", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--worker-request", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    if args.worker_request:
        return e.worker(args.worker_request, args.worker_output)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private controls must publish outside the repository")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8373-owned-", dir="/var/tmp") as directory:
            result = run(
                args.root,
                output,
                Path(directory),
                control=args.private_control,
                failed=args.failed_child,
            )
        if not args.private_control:
            write_note(output)
            append_retirements(output, e.ROOT / "ops/exclusion_manifest.yaml")
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
