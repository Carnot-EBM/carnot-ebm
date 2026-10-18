"""REQ-VERIFY-8236: complete owned checks before inspecting available resources.

Publication uses private checked bytes. Repository health is kept separate so
an unrelated failure cannot be reported as a successful complete-suite run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time

from carnot.inference import concurrency_runtime_8227 as runtime
from carnot.reporting import qualified_concurrency_8236 as q
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.request_trace_inventory_8200 import copy_bytes
from carnot.verify import concurrency_canary_8227 as e
from scripts.experiment_template import normalize_artifact_for_template_write

execute = q.old.execute


def main(argv: list[str] | None = None) -> int:
    """Freeze commands, preserve every obligation and publish only terminal bytes."""
    began = time.monotonic()
    e.progress("8236_start", 0, 8)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261007"], default="20261007")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (q.NAME + ".json"))
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = q.replay(args.cold_replay)
        e.progress("8236_cold_replay_after", int(passed), 0)
        return 0 if passed else 1
    fixture = args.fixture_e2e is not None
    output = (args.fixture_e2e or args.output).absolute()
    if fixture and output.resolve().is_relative_to(e.ROOT / "results"):
        parser.error("private fixture output must remain outside results")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-8236-validation-"))
    private.chmod(0o700)
    candidate = private / (q.NAME + ".json")
    plan = q.validation_plan(private)
    health = q.old.CommandSpec(
        "repository_health_once",
        (str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
        "repository_health_not_owned",
        600,
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
    e.progress("8236_inputs_before", 0, 1)
    data = q.inputs(args.root, raw / "authenticated")
    if output.is_file():
        copy_bytes(output, raw / "previous_primary")
    e.progress("8236_inputs_after", int(data["ready"]), 0)
    receipts = execute(plan[:1] if fixture or args.root != e.ROOT else plan, raw / "validation")
    coverage = private / "coverage.json"
    data["coverage_statement_counts"] = (
        json.loads(coverage.read_text())["totals"] if coverage.exists() else {}
    )
    result: q.Json = {}
    if data["ready"]:
        protocol_path = (
            raw / "protocol.json" if fixture or args.root != e.ROOT else e.ROOT / q.BINDINGS
        )
        protocol = data["protocol"]
        owned = all(r["passed"] and r["normal_exit"] for r in receipts)
        if owned:
            e.progress("8236_isolation_before", 0, 1)
            result["qualification"] = e.qualify(raw / "isolation")
            e.progress("8236_isolation_after", int(result["qualification"]["passed"]), 0)
            if result["qualification"]["passed"] and not fixture:
                e.progress("8236_resources_before", 0, 1)
                resources = runtime.preflight(data["identity"], raw / "preflight")
                data["checks"].extend(resources["checks"])
                data["resource_receipts"] = resources["receipts"]
                e.atomic_json(raw / "resources.json", resources)
                ready = all(c["passed"] for c in resources["checks"])
                e.progress("8236_resources_after", int(ready), 0)
                if ready:
                    model, gpu = Path(resources["model"]["model_path"]), resources["gpu"]["index"]
                    protocol["server_argv"] = {
                        name: runtime.command(model, raw / name, gpu)
                        for name in [
                            "canary_serial",
                            "canary_concurrent",
                            *protocol["launch_order"],
                        ]
                    }
        e.atomic_json(protocol_path, protocol)
        data["protocol_path"], data["protocol_sha256"] = (
            str(protocol_path),
            e.sha256_file(protocol_path),
        )
        data["refs"].append(copy_bytes(protocol_path, raw / "sealed_protocol"))
        if owned and result.get("qualification", {}).get("passed") and not fixture and ready:
            os.environ["CARNOT_FORCE_LIVE"] = "1"
            result.update(runtime.live(protocol, resources, raw / "canary", task_id=q.TASK))
    for row in result.get("rows", []):
        row["journal_path"] = str(
            raw / "canary" / ("canary_" + row["arm"]) / "requests/events.jsonl"
        )
    e.atomic_json(raw / "data.json", data)
    e.atomic_json(raw / "result.json", result)
    value = q.build(data, result, raw, receipts, time.monotonic() - began, fixture)
    e.atomic_json(raw / "request_rows.json", dict(rows=value["rows"]))
    value["repository_health"] = (
        execute([health], raw / "health") if not fixture and args.root == e.ROOT else []
    )
    value["duration_s"] = time.monotonic() - began
    value["raw_shard_hashes"] = [
        e.recorder.reference(p) for p in sorted(raw.rglob("*")) if p.is_file()
    ]
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = q.old.checksum(value)
    e.atomic_json(candidate, value)
    terminal = execute(q.validators(candidate), raw / "terminal")
    if not all(r["passed"] and r["normal_exit"] for r in terminal):
        e.atomic_json(raw / "failed_terminal_candidate.json", value)
        return 1
    checked_hash = e.sha256_file(candidate)
    e.atomic_json(
        raw / "terminal_receipt.json", dict(receipts=terminal, candidate_sha256=checked_hash)
    )
    publication = publish_primary(
        output, value, lambda p: dict(passed=e.sha256_file(p) == checked_hash, receipts=terminal)
    )
    e.atomic_json(raw / "publication_receipt.json", publication)
    e.progress("8236_published", 8, 0)
    return 0
