"""REQ-REPORT-8353: reuse bounded execution after freezing exact owned commands."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any
import time
from unittest.mock import patch

import yaml  # type: ignore[import-untyped]

from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.v709_execution import execute
from carnot.reporting import sentence_spline_execution_8334 as qualified
from carnot.verify import runtime_reader_8353 as q
from carnot.verify import runtime_reader_execution_8340 as execution

Json = dict[str, Any]
PY = str(q.ROOT / ".venv/bin/python")
_manifest = execution.manifest


def manifest(private: Path) -> list[Json]:
    """Retain the complete frozen consumer command and real old/current controls."""
    with (
        patch.object(execution, "q", q),
        patch.object(execution, "authority_controls", return_value=[]),
    ):
        commands = _manifest(private)
    config = private / "coverage.ini"
    config.write_text(
        config.read_text().replace(
            "[report]",
            "include = " + ",".join(str(q.ROOT / path) for path in q.OWNED) + "\n[report]",
        )
    )
    for command in commands:
        if command["name"] in {"ruff_check", "ruff_format", "spec_coverage"}:
            command["argv"].append("tests/python/test_runtime_change_boundary_8307.py")
    for name, milestone, expected in [
        ("valid_current", q.MILESTONE, 0),
        ("valid_old", "2026.10.717", 0),
        ("wrong_milestone", q.MILESTONE, 1),
        ("corrupt_contract", q.MILESTONE, 1),
    ]:
        root = q.private_authority(private / name, milestone)
        path = root / "research-roadmap.yaml"
        active = yaml.safe_load(path.read_bytes())
        if name == "wrong_milestone":
            active["milestone"] = "wrong"
        if name == "corrupt_contract":
            active["tasks"][0]["prompt"] += " corrupted private control"
        path.write_text(yaml.safe_dump(active))
        commands.append(
            dict(
                name=name,
                argv=[PY, "-u", str(q.ROOT / q.CLI), "--authority", milestone, "--root", str(root)],
                deadline=60,
                expected=expected,
                scope="owned",
            )
        )
    return list(commands)


def main(argv: list[str] | None = None) -> int:
    """Finish owned reader checks before any changed-environment CUDA operation."""
    args = list(sys.argv[1:] if argv is None else argv)
    q.progress("start_no_model_load")
    os.environ["PYTHONUNBUFFERED"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261009"], default="20261009")
    parser.add_argument("--root", type=Path, default=q.ROOT)
    parser.add_argument("--output", type=Path, default=q.ROOT / "results" / (q.NAME + ".json"))
    parser.add_argument("--private-run", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--authority", choices=[q.MILESTONE, "2026.10.717"])
    parsed = parser.parse_args(args)
    if parsed.cold_replay:
        passed = q.replay(parsed.cold_replay)
        q.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    if parsed.authority:
        with TemporaryDirectory(prefix="carnot8353-authority-") as directory:
            result = q.authority(parsed.root, Path(directory), parsed.authority)
        print(json.dumps(result), flush=True)
        return int(not result["activated"])
    output = parsed.output.absolute()
    if parsed.private_run and output.is_relative_to(q.ROOT / "results"):
        parser.error("private run requires private output")
    began, wall = time.monotonic_ns(), time.time_ns()
    raw = output.parent / "raw" / q.NAME / "invocations" / str(wall)
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    with TemporaryDirectory(prefix="carnot8353-") as directory:
        private = Path(directory)
        commands = manifest(private)
        frozen = raw / "execution_manifest.json"
        candidate = output.parent / "raw" / q.NAME / "terminal_candidate.json"
        atomic_json(
            frozen,
            dict(
                commands=commands,
                frozen_before_measurement=True,
                heartbeat_s=20,
                terminal_deadline_s=60,
                terminal_argv=[
                    [PY, "-u", str(q.ROOT / q.CLI), "--cold-replay", str(candidate)],
                    [PY, "scripts/adversarial_verify.py", "--json", str(candidate)],
                    [
                        PY,
                        "scripts/verdict_row_consistency_lint.py",
                        "--strict",
                        str(candidate),
                    ],
                ],
            ),
        )
        q.progress("reader_validation_before", 0, len(commands))
        receipts = execute(commands[:1] if parsed.private_run else commands, raw / "validation")
        q.progress("reader_validation_after", len(receipts), 0)
        ready = all(r["passed"] for r in receipts) and not parsed.private_run
        work = q.measure(parsed.root, raw, reader_checks_passed=ready)
        prior = q.ROOT / "results/raw" / q.NAME
        work["prior_validation_attempts"] = [
            dict(path=str(path), receipt=json.loads(path.read_bytes()))
            for path in (prior / "development").glob("*.receipt.json")
        ]
        work["refs"].extend(
            reference(Path(row["path"])) for row in work["prior_validation_attempts"]
        )
        work["refs"].append(reference(frozen))
        work["checks"] += [
            q.old.gate(Path(r["stdout_path"]), "required_tools", True, r["passed"])
            for r in receipts
            if r.get("scope") == "external_preconditions" and not r["passed"]
        ]
        if (private / "coverage.json").is_file():
            coverage = json.loads((private / "coverage.json").read_bytes())
            atomic_json(raw / "owned_coverage.json", coverage)
            work["owned_statement_coverage"] = coverage["totals"]
            work["refs"].append(reference(raw / "owned_coverage.json"))
        findings = qualified.qualify_findings(raw / "finding_controls")
        receipts += findings["receipts"]
        work["finding_audits"] = findings["audits"]
        work["refs"].append(findings["primitive_reference"])
        ended = time.monotonic_ns()
        work.update(
            private_run=parsed.private_run,
            duration_s=(ended - began) / 1e9,
            phase_spans=[
                dict(
                    phase="reader_then_runtime",
                    started_monotonic_ns=began,
                    ended_monotonic_ns=ended,
                    started_wall_ns=wall,
                    duration_s=(ended - began) / 1e9,
                )
            ],
        )
        atomic_json(raw / "measurement.json", work)
        value = q.build(work, raw, receipts)
        controls = []
        for name in ("negative_checksum", "rehashed_tamper"):
            bad = deepcopy(value)
            bad["runtime_changed_score"] = 1 - bad["runtime_changed_score"]
            bad["reproducibility_checksum"] = (
                q.checksum(bad) if name == "rehashed_tamper" else "invalid"
            )
            path = private / (name + ".json")
            atomic_json(path, bad)
            controls.append(
                dict(
                    name=name,
                    argv=[PY, "-u", str(q.ROOT / q.CLI), "--cold-replay", str(path)],
                    deadline=60,
                    expected=1,
                    scope="owned",
                )
            )
        receipts += execute(controls, raw / "cold_controls")
        with patch.object(qualified, "e", q):
            qualified.publish(q.build(work, raw, receipts), output, raw)
    return 0
