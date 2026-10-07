"""REQ-REPORT-8242: validate owned work before loading a fresh request server."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time

from carnot.inference import concurrency_runtime_8227 as runtime
from carnot.reporting import independent_concurrent_service_8242 as q
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.request_trace_inventory_8200 import copy_bytes
from carnot.verify import concurrency_canary_8227 as e
from scripts.experiment_template import normalize_artifact_for_template_write

execute = q.recorded.execute


def main(argv: list[str] | None = None) -> int:
    """Publish only normally validated bytes and preserve honest failed attempts."""
    began = time.monotonic()
    e.progress("8242_start", 0, 6)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261007"], default="20261007")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (q.NAME + ".json"))
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = q.replay(args.cold_replay)
        e.progress("8242_cold_replay_after", int(passed), 0)
        return 0 if passed else 1
    output = (args.fixture_e2e or args.output).absolute()
    if args.fixture_e2e and output.resolve().is_relative_to(e.ROOT / "results"):
        parser.error("private fixture output must remain outside results")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-8242-validation-"))
    private.chmod(0o700)
    candidate = private / (q.NAME + ".json")
    plan = q.validation_plan(private)
    health = q.recorded.CommandSpec(
        "repository_health_once",
        (str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
        "repository_health_not_owned",
        120,
    )
    e.atomic_json(
        raw / "validation_commands.json",
        dict(
            commands=[asdict(c) for c in plan],
            terminal=[asdict(c) for c in q.validators(candidate)],
            repository_health=asdict(health),
            frozen_before_measurement=True,
        ),
    )
    e.progress("8242_inputs_before", 0, 1)
    data = q.inputs(args.root, raw / "authenticated")
    if output.is_file():
        copy_bytes(output, raw / "previous_primary")
    scratch = execute(
        [
            q.recorded.CommandSpec(
                "private_scratch",
                (
                    str(e.ROOT / ".venv/bin/python"),
                    "-c",
                    "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.write_text('writable'); print(p.read_text())",
                    str(private / "writable"),
                ),
                "preconditions",
                10,
            )
        ],
        raw / "scratch",
    )
    data["checks"].append(
        q.operand("private_scratch_writable", private / "writable", True, scratch[0]["passed"])
    )
    data["ready"] = data["ready"] and scratch[0]["passed"]
    e.progress("8242_inputs_after", int(data["ready"]), 0)
    receipts = execute(
        plan[:1] if args.root != e.ROOT or args.fixture_e2e else plan, raw / "validation"
    )
    coverage = private / "coverage.json"
    data["coverage_statement_counts"] = (
        json.loads(coverage.read_text())["totals"] if coverage.exists() else {}
    )
    work: q.Json = {}
    if (
        data["ready"]
        and all(r["passed"] and r["normal_exit"] for r in receipts)
        and not args.fixture_e2e
    ):
        e.progress("8242_resources_before", 0, 1)
        resources = runtime.preflight(data["protocol"]["identity"], raw / "resources")
        data["checks"].extend(resources["checks"])
        data["resource_receipts"] = resources["receipts"]
        e.atomic_json(raw / "resources.json", resources)
        e.progress("8242_resources_after", int(all(c["passed"] for c in resources["checks"])), 0)
        if all(c["passed"] for c in resources["checks"]):
            os.environ["CARNOT_FORCE_LIVE"] = "1"
            work = q.measured.measure(data, resources, raw / "measurement")
    e.atomic_json(raw / "data.json", data)
    e.atomic_json(raw / "work.json", work)
    value = q.build(data, work, raw, receipts, time.monotonic() - began)
    e.atomic_json(raw / "request_rows.json", dict(rows=value["rows"]))
    value["repository_health"] = (
        execute([health], raw / "health") if args.root == e.ROOT and not args.fixture_e2e else []
    )
    value["duration_s"] = time.monotonic() - began
    value["raw_shard_hashes"] = [
        e.recorder.reference(p) for p in sorted(raw.rglob("*")) if p.is_file()
    ]
    value = normalize_artifact_for_template_write(value)
    value["field_principles"].update(
        {
            k: "Bind invocation bytes and observed work without importing historical model calls."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = q.checksum(value)
    e.atomic_json(candidate, value)
    terminal = execute(q.validators(candidate), raw / "terminal")
    if not all(r["passed"] and r["normal_exit"] for r in terminal):
        e.atomic_json(raw / "failed_terminal_candidate.json", value)
        return 1
    checked = e.sha256_file(candidate)
    e.atomic_json(raw / "terminal_receipt.json", dict(receipts=terminal, candidate_sha256=checked))
    publication = publish_primary(
        output, value, lambda p: dict(passed=e.sha256_file(p) == checked, receipts=terminal)
    )
    e.atomic_json(raw / "publication_receipt.json", publication)
    e.progress("8242_published", 6, 0)
    return 0
