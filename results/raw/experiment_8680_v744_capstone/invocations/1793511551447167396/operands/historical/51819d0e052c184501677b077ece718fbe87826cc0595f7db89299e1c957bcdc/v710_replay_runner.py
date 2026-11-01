"""REQ-REPORT-8218: publish only normally validated immutable replay evidence.

The temporary parent outlives every bounded child. Repository health remains a
separate diagnostic and cannot repair a failed owned validation command.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v709_execution as x
from carnot.reporting import v709_runner as qualified
from carnot.reporting import v710_contract_replay as q
from carnot.reporting import v710_replay_history as h
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]


def replay(path: Path) -> Json:
    """A fresh process checks copied operands and recomputes every owned aggregate."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    if canonical_hash(value) != checksum:
        raise ValueError("candidate_checksum_drift")
    for ref in [
        value["work_reference"],
        *value["source_artifact_hashes"],
        *value["raw_shard_hashes"],
        *value["code_config_hashes"],
        *value["immutable_code_snapshots"],
    ]:
        q.require_reference(ref)
    for receipt in value["validation_receipts"]:
        for stream in ("stdout", "stderr"):
            q.require_reference(
                dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
            )
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    rebuilt = q.reduce(work, value["validation_receipts"])
    for key, observed in rebuilt.items():
        if value[key] != observed:
            raise ValueError("primitive_reduction_drift:" + key)
    snapshots = work["contract"]["authority_snapshots"]
    if snapshots["design"]["exists"]:
        with tempfile.TemporaryDirectory(prefix="carnot8218-cold-") as directory:
            private = Path(directory)
            paths = [
                Path(snapshots[k].get("snapshot_path", private / k))
                for k in ("design", "staged", "active")
            ]
            contract = q.assess(*paths, private / "authority")
            for key in ("activated", "canonical_tasks_sha256", "contract_rows"):
                if contract[key] != work["contract"][key]:
                    raise ValueError("authority_reduction_drift:" + key)
    for row in work["historical_dispositions"]:
        refs = [r for r in work["source_artifact_hashes"] if r["path"] == row.get("path")]
        if (
            refs
            and {
                k: v
                for k, v in json.loads(Path(refs[0]["snapshot_path"]).read_bytes()).items()
                if k in row["fields_imported"]
            }
            != row["original_primary"]
        ):
            raise ValueError("historical_disposition_drift")
    audit_rows = [
        r
        for r in work["historical_dispositions"]
        if (r.get("original_primary") or {}).get("experiment_id") == 8210
    ]
    if audit_rows and work["h1_custody"].get("agreement") is not None:
        original = audit_rows[0]["original_primary"]["measurement_reference"]
        ref = next(r for r in work["source_artifact_hashes"] if r["path"] == original["path"])
        from carnot.verify.restricted_decision_audit_8210 import n

        evidence = json.loads(Path(ref["snapshot_path"]).read_bytes())["evidence"]
        if n.reduce(evidence)["H1"] != work["h1_custody"]["H1"]:
            raise ValueError("original_H1_reduction_drift")
    return dict(passed=True, rows_checksum=canonical_hash(rebuilt["rows"]))


def main(argv: list[str] | None = None) -> int:
    """Freeze invocation ownership before work and atomically publish checked bytes."""
    x.progress("8218_start", 0, 14)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261007"], default="20261007")
    parser.add_argument("--root", type=Path, default=q.ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--private-fixture", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or q.ROOT / "results" / (q.NAME + ".json")).absolute()
        if args.private_fixture and (
            output.is_relative_to(q.ROOT / "results") or args.root == q.ROOT
        ):
            raise ValueError("fixture_requires_private_root_and_output")
        raw = output.parent / "raw" / q.NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        start, wall = time.monotonic_ns(), time.time_ns()
        with tempfile.TemporaryDirectory(prefix="carnot8218-", dir="/var/tmp") as directory:
            private = Path(directory)
            probe = private / "writable_probe"
            probe.write_bytes(b"private authenticated scratch")
            runtime = dict(
                python=os.sys.version,
                executable=os.sys.executable,
                scratch_mode=oct(private.stat().st_mode & 0o777),
                writable=probe.read_bytes() == b"private authenticated scratch",
                scratch_free_bytes=os.statvfs(private).f_bavail * os.statvfs(private).f_frsize,
            )
            plan = [] if args.private_fixture else q.commands(private)
            controls = x.pytest_plan(private / "controls")
            preflight = qualified.precondition_command()
            with patch.object(qualified, "CLI", q.CLI):
                terminal = qualified.terminal_plan(
                    output.parent / "raw" / q.NAME / "terminal_candidate.json"
                )
            atomic_json(
                raw / "validation_manifest.json",
                dict(
                    commands=plan,
                    controls=controls,
                    terminal_commands=terminal,
                    precondition=preflight,
                    runtime=runtime,
                    frozen_before_measurement_ns=time.monotonic_ns(),
                ),
            )
            preconditions = x.execute([preflight], raw / "preconditions")
            x.progress("8218_measurement_before", 0, 14)
            work = h.measure(args.root, raw)
            if not preconditions[0]["passed"]:
                work["failures"].append(
                    q.failure(
                        Path(preconditions[0]["stderr_path"]),
                        "python_environment_exit",
                        0,
                        preconditions[0]["exit_code"],
                        preconditions[0]["stderr_sha256"],
                    )
                )
            x.progress("8218_measurement_after", 14, 0)
            receipts = x.execute(controls, raw / "controls_logs") + work["replay_controls"]
            receipts += x.execute(
                [p for p in plan if p["scope"] == "owned"], raw / "validation_logs"
            )
            health = x.execute(
                [p for p in plan if p["scope"] == "repository_health"], raw / "health_logs"
            )
            value = q.reduce(work, receipts)
            atomic_json(raw / "work.json", work)
            coverage = private / "coverage.json"
            atomic_json(
                raw / "coverage.json",
                json.loads(coverage.read_bytes()) if coverage.is_file() else {},
            )
            code = [
                q.snapshot(q.ROOT / p, raw / "owned_code", str(i))
                for i, p in enumerate([*q.OWNED, q.TEST, *q.REUSED])
            ]
            value.update(
                duration_s=(time.monotonic_ns() - start) / 1e9,
                random_seed=7108218,
                phase_spans=work.get("phase_spans", []),
                runtime_preconditions=runtime,
                precondition_receipts=preconditions,
                code_config_hashes=code,
                work_reference=dict(
                    path=str(raw / "work.json"), sha256=h.sha256_file(raw / "work.json")
                ),
                repository_health=dict(owned=False, receipts=health),
                coverage_totals=json.loads((raw / "coverage.json").read_bytes()).get("totals", {}),
                raw_shard_hashes=[
                    dict(path=str(p), sha256=h.sha256_file(p))
                    for p in sorted(raw.rglob("*"))
                    if p.is_file()
                ],
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
                measurement_clocks=dict(
                    started_monotonic_ns=start,
                    started_wall_ns=wall,
                    ended_monotonic_ns=time.monotonic_ns(),
                ),
                cited_upstream_artifacts=[
                    dict(
                        path=r.get("path"),
                        sha256=r.get("sha256"),
                        fields_imported=r.get("fields_imported", []),
                    )
                    for r in work["historical_dispositions"]
                ],
            )
            value = normalize_artifact_for_template_write(value)
            value["field_principles"] = {
                k: f"{k} binds actual invocation operands; "
                "administrative readiness cannot establish scientific benefit."
                for k in value
            }
            value["field_principles"].update(
                contract_ready_score="Exact current full-task authority qualifies independently of historical science.",
                historical_replay_ready_score="Only authentic original code, primitives and normal owned checks qualify replay.",
                historical_dispositions="Imported fields and complete hashed primary copies preserve every prior outcome.",
                immutable_code_snapshots="Producer seals select committed original bytes; unavailable originals remain unavailable.",
                h1_custody_qualification="Registered H1 agreement measures custody only and never enters a science gate.",
                canonical_tasks_sha256="The original design digest binds count, order and all executable task fields.",
            )
            value["reproducibility_checksum"] = canonical_hash(value)
            x.progress("8218_publication_before", 14, 1)

            def validate(candidate: Path) -> Json:
                specs = [
                    dict(
                        s,
                        argv=[
                            str(candidate)
                            if a == str(output.parent / "raw" / q.NAME / "terminal_candidate.json")
                            else a
                            for a in s["argv"]
                        ],
                    )
                    for s in terminal
                ]
                checks = x.execute(specs, raw / "terminal_logs")
                return dict(passed=all(r["passed"] for r in checks), checks=checks)

            publication = publish_primary(output, value, validate)
            atomic_json(
                raw / "terminal_validation.json",
                dict(
                    publication=publication,
                    normal_process_exit=True,
                    required_checks_passed=value["required_checks_passed"],
                ),
            )
        x.progress("8218_complete", 14, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
