"""REQ-VERIFY-8387: reuse bounded validation and atomically publish terminal evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v721_capstone as old
from carnot.reporting import v722_capstone_evidence as e

Json = dict[str, Any]
MANIFEST = old.manifest


def manifest(private: Path) -> list[Json]:
    """Freeze owned coverage, real consumers, full mutations and one separate global run."""
    with patch.object(old, "e", e):
        plan = MANIFEST(private)
    for item in plan:
        item["argv"] = [
            "tests/python/test_v721_capstone_frozen_aliases_8373.py::test_source_alias_reads_and_writes_use_private_custody"
            if arg == "tests/python/test_v721_contract_methods_8360.py::test_private_e2e018"
            else arg
            for arg in item["argv"]
        ]
    plan.append(
        dict(
            name="full_contract_mutations",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_v722_contract_methods_8374.py",
                "--basetemp=" + str(private / "contracts"),
            ],
            expected=0,
            deadline=300,
            scope="owned",
        )
    )
    plan.append(
        dict(
            name="repository_python_suite",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "tests/python",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(private / "global"),
            ],
            expected=0,
            deadline=900,
            scope="global",
        )
    )
    return list(plan)


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, failed: bool = False
) -> int:
    """Existing execution retains exact receipts, findings and unchanged primary validators."""
    with patch.multiple(old, e=e, manifest=manifest, controls=e.controls):
        return int(old.run(root, output, private, control=control, failed=failed))


def write_note(output: Path) -> None:
    """Each independent continuation remains visible without treating parity as science."""
    value = json.loads(output.read_bytes())
    lines = [
        "# V722 capstone — 2026-10-10",
        "",
        "Verdict: " + value["honest_verdict"] + ".",
        "",
        f"Fourteen slots: {value['actual_executed_task_count']} executed including capstone; {value['pre_gate_count']} pre-gates; {value['cascade_skip_count']} logged cascade skips; {value['missing_output_count']} absent primaries (the skip has no primary).",
        "",
        "Qualified V721 H1=-0.00390625 and H2=0 remain closed exposed-development nulls. The SciPy table proof is unchanged: actual fast-path fraction=0. Direct recovery is independent of that route. Numerical parity is not semantic benefit. Both generalization scores remain zero.",
        "",
        "| Task | Original disposition | Continue / retire / defer condition |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {r['task_id']} | {r['honest_verdict'] or r['disposition']} | {r['continuation_condition']} |"
        for r in value["rows"]
    )
    lines.extend(
        [
            "",
            "Three PRD gaps remain open: independent useful decisions, continuous learning and retention, and full native request cost. Source independence, oracle labels, small samples and exposed public ARC outcomes remain explicit evidence limits.",
            "",
            "Retirement applies only to authenticated exact repeated inspection mechanisms. Missing training or service-cost science is deferred. KV260 stays quadratic Ising k<=5; PolarFire stays board-local Linux CPU; GateMate needs exact source recovery and dated physical change before probing.",
            "",
            f"Publication G1={value['g1']}, G2={value['g2']}, G3={value['g3']}, G4={value['g4']}; paper_ready={value['paper_ready']}; unmet={value['unmet_gates']}. Global repository health is separately recorded. No external publication follows.",
        ]
    )
    (e.ROOT / "docs/research-notes/v722-capstone.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Real private CLI controls exercise failures without exposing invented research primaries."""
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
        return e.worker_process(args.worker_request, args.worker_output)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private controls must publish outside the repository")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8387-owned-", dir="/var/tmp") as directory:
            result = run(
                args.root,
                output,
                Path(directory),
                control=args.private_control,
                failed=args.failed_child,
            )
        if not args.private_control:
            write_note(output)
            with patch.object(old, "e", e):
                old.append_retirements(output, e.ROOT / "ops/exclusion_manifest.yaml")
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
