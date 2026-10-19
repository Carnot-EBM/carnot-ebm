"""REQ-VERIFY-8234-EXECUTION: frozen bounded checks precede checked publication."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import decision_margin_methods_8234 as q
from carnot.reporting import v709_execution as x
from carnot.reporting import v711_current_runner as qualified
from carnot.reporting import v685_authority_lifecycle as authority
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v710_contract_replay import require_reference
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]


def commands(private: Path) -> list[Json]:
    """Measure only added statements while retaining existing private consumer checks."""
    private.mkdir(parents=True, exist_ok=True)
    with (
        patch.object(qualified.q, "OWNED", q.OWNED),
        patch.object(qualified.q, "TEST", q.TEST),
        patch.object(qualified.q, "CLI", q.CLI),
    ):
        plan = qualified.commands(private)
    for spec in plan:
        if spec["name"] == "changed_module_coverage":
            spec["deadline"] = 300
    plan.append(
        dict(
            name="private_E2E021",
            argv=[
                str(q.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_restricted_decision_audit_8210.py",
                "--basetemp=" + str(private / "e2e021"),
            ],
            expected=0,
            deadline=240,
            scope="owned",
        )
    )
    return list(plan)


def replay(path: Path) -> Json:
    """Rebuild primitives from authenticated originals, not reported aggregate values."""
    v = json.loads(path.read_bytes())
    for receipt in v["validation_receipts"]:
        for stream in ["stdout", "stderr"]:
            if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                raise ValueError("receipt_drift")
    for ref in [
        v["work_reference"],
        *v["source_artifact_hashes"],
        *v["code_config_hashes"],
        *v["raw_shard_hashes"],
    ]:
        require_reference(ref)
    w = json.loads(Path(v["work_reference"]["path"]).read_bytes())
    if v["reproducibility_checksum"] != canonical_hash(
        [v["work_reference"], v["code_config_hashes"], q.PIN]
    ):
        raise ValueError("checksum_drift")
    refs = {r["path"]: r for r in w["refs"]}
    p = q.PROTOCOL_VALUE
    if p != w["protocol"] or v["protocol_sha256"] != q.PIN:
        raise ValueError("protocol_drift")
    expected_failures = []
    for expected in [
        dict(path=str(q.ROOT / q.PROTOCOL), sha256=q.PIN),
        *p["source_artifact_hashes"],
    ]:
        operand = Path(w["root"]) / Path(expected["path"]).relative_to(q.ROOT)
        observed = refs[str(operand)]["sha256"]
        if observed != expected["sha256"]:
            expected_failures.append(
                q.failure(operand, "sha256", expected["sha256"], observed, observed)
            )
    for name in q.INPUTS:
        if not refs[str(Path(w["root"]) / name)]["exists"] and name != q.ACTIVE:
            expected_failures.append(q.failure(Path(w["root"]) / name, "exists", True, None))
    if [f for f in w["failures"] if f["artifact_field"] != "source_schema"] != expected_failures:
        raise ValueError("source_gate_drift")
    if not w["failures"]:
        read = lambda ref: json.loads(
            Path(
                refs[str(Path(w["root"]) / Path(ref["path"]).relative_to(q.ROOT))]["snapshot_path"]
            ).read_bytes()
        )
        if q.primitives(read, p) != w["public_rows"]:
            raise ValueError("source_primitive_drift")
    snapshots = w["contract"]["authority_snapshots"]
    if snapshots:
        with tempfile.TemporaryDirectory(prefix="carnot8234-replay-") as directory:
            raw = Path(directory)
            paths = [
                Path(snapshots[k].get("snapshot_path", raw / ("absent-" + k)))
                for k in ["design", "staged", "active"]
            ]
            contract = authority.assess_authorities(
                *paths, raw, milestone=q.MILESTONE, first_id=8234, count=14
            )
            for key in ["activated", "contract_rows", "canonical_tasks_sha256"]:
                if contract[key] != w["contract"][key]:
                    raise ValueError("authority_drift")
    rebuilt = q.reduce(w, v["validation_receipts"])
    for key, value in rebuilt.items():
        if v[key] != value:
            raise ValueError("primitive_reduction_drift:" + key)
    return dict(passed=True, rows_checksum=canonical_hash(rebuilt["rows"]))


def main(argv: list[str] | None = None) -> int:
    """A private parent outlives children and exposes only validated candidate bytes."""
    q.progress("start", 0, 14)
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
            raise ValueError("private_fixture_requires_private_root_and_output")
        raw = output.parent / "raw" / q.NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, mode=0o700)
        start, wall = time.monotonic_ns(), time.time_ns()
        with tempfile.TemporaryDirectory(prefix="carnot8234-", dir="/tmp") as directory:
            private = Path(directory)
            probe = private / "writable"
            probe.write_bytes(b"current private scratch")
            runtime = dict(
                path=str(private),
                writable=probe.read_bytes() == b"current private scratch",
                mode=oct(private.stat().st_mode & 0o777),
                executable=sys.executable,
                free_bytes=os.statvfs(private).f_bavail * os.statvfs(private).f_frsize,
            )
            plan = [] if args.private_fixture else commands(private)
            controls = x.pytest_plan(private / "controls")
            terminal = qualified.terminal_plan(
                output.parent / "raw" / q.NAME / "terminal_candidate.json"
            )
            terminal[0]["argv"][-3] = str(q.ROOT / q.CLI)
            atomic_json(
                raw / "validation_manifest.json",
                dict(
                    commands=plan,
                    controls=controls,
                    terminal=terminal,
                    owned=q.OWNED,
                    runtime=runtime,
                    frozen_before_measurement_ns=time.monotonic_ns(),
                ),
            )
            q.progress("measurement_before", 0, 14)
            work = q.measure(args.root, raw)
            work["root"] = str(args.root)
            q.progress("measurement_after", 14, 0)
            receipts = x.execute(controls, raw / "control_logs")
            receipts += x.execute(
                [s for s in plan if s["scope"] == "owned"], raw / "validation_logs"
            )
            health = x.execute(
                [s for s in plan if s["scope"] == "repository_health"], raw / "health_logs"
            )
            value = q.reduce(work, receipts)
            atomic_json(raw / "work.json", work)
            cov = private / "coverage.json"
            coverage = json.loads(cov.read_bytes()) if cov.exists() else {}
            atomic_json(raw / "coverage.json", coverage)
            end = time.monotonic_ns()
            value.update(
                duration_s=(end - start) / 1e9,
                random_seed=7128239,
                runtime_preconditions=runtime,
                repository_health=dict(owned=False, receipts=health),
                coverage_totals=coverage.get("totals", {}),
                coverage_statement_counts=coverage.get("files", {}),
                work_reference=dict(
                    path=str(raw / "work.json"), sha256=sha256_file(raw / "work.json")
                ),
                raw_shard_hashes=[
                    dict(path=str(p), sha256=sha256_file(p))
                    for p in sorted(raw.rglob("*"))
                    if p.is_file()
                ],
                phase_spans=[
                    dict(
                        phase="methods_measurement_and_validation",
                        started_monotonic_ns=start,
                        ended_monotonic_ns=end,
                        duration_s=(end - start) / 1e9,
                    )
                ]
                + [
                    dict(
                        phase=r["name"],
                        started_monotonic_ns=r["started_monotonic_ns"],
                        ended_monotonic_ns=r["ended_monotonic_ns"],
                        duration_s=r["duration_s"],
                    )
                    for r in receipts + health
                ],
                invocation_argv=sys.orig_argv,
                measurement_clocks=dict(
                    started_monotonic_ns=start,
                    ended_monotonic_ns=end,
                    started_wall_ns=wall,
                    ended_wall_ns=time.time_ns(),
                ),
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
            )
            value["reproducibility_checksum"] = canonical_hash(
                [value["work_reference"], value["code_config_hashes"], q.PIN]
            )
            value["field_principles"] = {
                k: "Actual invocation bytes and clocks separate execution readiness from unmeasured benefit."
                for k in value
            }
            value["field_principles"].update(
                current_contract_ready_score="Actual active authority and owned checks; no scientific benefit credit.",
                margin_protocol_ready_score="Authenticated executable methods and owned checks; no observed benefit.",
                native_margin_weight_rows="Native input probabilities alone define weights; missing slots remain missing.",
                frozen_roles="Original identities and roles precede fitting; reserved labels remain closed.",
                MODEL_SPECS="No current generator load; historical Qwen provenance is not a current call.",
                repository_health="A bounded full-suite diagnostic retains failures separately from owned readiness.",
            )
            value = normalize_artifact_for_template_write(value)
            atomic_json(raw / "candidate.json", value)
            q.progress("publication_before", 14, 1)

            def validate(candidate: Path) -> Json:
                checked = x.execute(terminal, raw / "terminal_logs")
                return dict(passed=all(r["passed"] for r in checked), checks=checked)

            published = publish_primary(output, value, validate)
            atomic_json(
                raw / "terminal_validation.json",
                dict(publication=published, required_checks_passed=value["required_checks_passed"]),
            )
        q.progress("complete", 14, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
