"""REQ-VERIFY-8553: publish only bytes accepted by real cold and publication consumers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import hallmark_release_8553 as e
from carnot.reporting import hallmark_release_inputs_8553 as inputs
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def validate(candidate: Path, raw: Path) -> Json:
    """Existing scanners keep their findings, including findings that prevent publication."""
    checks = []
    for name, args in [
        ("cold", ["--cold-replay", str(candidate)]),
        ("adversarial", ["scripts/adversarial_verify.py", "--json", str(candidate)]),
        ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)]),
    ]:
        prefix = [str(e.ROOT / e.CLI)] if name == "cold" else []
        checks.append(
            child(
                name,
                [str(e.ROOT / ".venv/bin/python"), "-u", *prefix, *args],
                raw,
                deadline=60,
                heartbeat=20,
            )
        )
    return dict(
        passed=all(r["passed"] for r in checks),
        checks=checks,
        adversarial_findings=json.loads(Path(checks[1]["stdout_path"]).read_text()),
    )


def main(argv: list[str] | None = None) -> int:
    """The CLI supports private fixtures so rejection controls never rewrite history."""
    print("[exp8553] start no_model_load MODEL_SPECS=[] LLM_calls=0", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261015"], default="20261015")
    parser.add_argument("--manifest", type=Path, default=inputs.MANIFEST)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    began = time.monotonic()
    try:
        if args.deliberate_error:
            raise ValueError("owned_failure_control")
        if args.cold_replay:
            print(json.dumps(e.replay(args.cold_replay)), flush=True)
            return 0
        print("[exp8553] authenticated_reduction_before completed=0 pending=1", flush=True)
        data = json.loads(args.fixture.read_bytes()) if args.fixture else inputs.load(args.manifest)
        raw = args.output.parent / "raw" / e.NAME
        raw.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="carnot8553-", dir="/var/tmp"))
        probe = scratch / "write_probe"
        probe.write_text("disk-backed")
        probe.unlink()
        data.update(
            code_config_hashes=[
                e.reference(e.ROOT / n)
                for n in [*e.OWNED, "python/carnot/reporting/hallmark_release_inputs_8553.py"]
            ],
            preconditions_checked=json.loads((inputs.RAW / "preconditions.json").read_bytes()),
            phase_spans=[
                dict(phase="authenticated_reduction", duration_s=time.monotonic() - began)
            ],
            duration_s=(
                time.time_ns()
                - json.loads((inputs.RAW / "preconditions.json").read_bytes())["started_wall_ns"]
            )
            / 1e9
            if not args.fixture
            else time.monotonic() - began,
        )
        primitive = raw / "primitive_rows.json"
        atomic_json(primitive, data)
        source = e.reference(primitive)
        receipts = json.loads(args.receipts.read_bytes()) if args.receipts else []
        value = e.build(data, source, receipts, args.output)
        candidate = scratch / (e.NAME + ".json")
        atomic_json(candidate, value)
        print("[exp8553] authenticated_reduction_after completed=1 pending=0", flush=True)
        initial = validate(candidate, raw / "initial_checks")
        receipts.extend(initial["checks"])
        value = e.build(data, source, receipts, args.output)
        value["adversarial_findings"] = initial["adversarial_findings"]
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        publication = publish_primary(
            args.output, value, lambda p: validate(p, raw / "terminal_checks")
        )
        bound = read_bound_sidecar(args.output, Path(publication["sidecar_path"]))
        atomic_json(
            raw / "terminal_validation.json", dict(publication=publication, report=bound["report"])
        )
        consumer = reader_receipt(
            e.TASK,
            args.output.parent,
            field="release_audit_ready_score",
            expected=value["release_audit_ready_score"],
        )
        atomic_json(raw / "downstream_consumer.json", consumer)
        if not consumer["passed"]:
            raise ValueError("downstream_consumer")
        print(
            "[exp8553] published_" + value["verdict_class"] + " completed=1 pending=0", flush=True
        )
        return int(not value["required_checks_passed"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("[exp8553] rejected " + str(error), flush=True)
        return 1
