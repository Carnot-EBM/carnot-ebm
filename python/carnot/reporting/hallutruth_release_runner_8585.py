"""REQ-REPORT-8585-REPLAY: expose a bounded CPU audit through existing consumers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from carnot.reporting import hallutruth_release_8585 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def validate(candidate: Path, raw: Path) -> Json:
    """Real isolated children enforce source replay and unchanged shared scanners."""
    checks = []
    for name, args in [
        ("cold", [e.CLI, "--cold-replay", str(candidate)]),
        ("adversarial", ["scripts/adversarial_verify.py", "--json", str(candidate)]),
        ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)]),
    ]:
        checks.append(
            child(
                name,
                [str(e.ROOT / ".venv/bin/python"), "-u", *args],
                raw,
                deadline=60,
                heartbeat=20,
            )
        )
    return dict(passed=all(r["passed"] for r in checks), checks=checks)


def main(argv: list[str] | None = None) -> int:
    """Private output support tests actual publication without touching other experiments."""
    print("[exp8585] start no_model_load MODEL_SPECS=[] current_LLM_calls=0", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261016"], default="20261016")
    raw_default = e.ROOT / "results" / "raw" / e.NAME
    parser.add_argument("--manifest", type=Path, default=raw_default / "release_manifest.json")
    parser.add_argument("--receipts", type=Path, default=raw_default / "validation_receipts.json")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.deliberate_error:
            raise ValueError("owned_failure_control")
        if args.cold_replay:
            print(json.dumps(e.replay(args.cold_replay)), flush=True)
            return 0
        if (
            args.manifest.is_file()
            and json.loads(args.manifest.read_bytes()).get("commit") == "fixture"
            and args.output.resolve().parent == (e.ROOT / "results").resolve()
        ):
            raise ValueError("fixture_requires_private_output")
        print("[exp8585] phase=preconditions_and_inspection", flush=True)
        source = e.measure(args.manifest, args.output)
        receipts = (
            json.loads(args.receipts.read_bytes())
            if args.receipts.is_file()
            else [dict(name="missing_owned_validation", passed=False, path=str(args.receipts))]
        )
        value = e.build(source, receipts, args.output)
        raw = args.output.parent / "raw" / e.NAME
        candidate = Path(e.read(source)["preconditions_checked"]["scratch"]) / (e.NAME + ".json")
        atomic_json(candidate, value)
        print("[exp8585] phase=initial_validation", flush=True)
        report = validate(candidate, raw / "initial_checks")
        receipts.extend(report["checks"])
        value = e.build(source, receipts, args.output)
        publication = publish_primary(
            args.output, value, lambda path: validate(path, raw / "terminal_checks")
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
            "[exp8585] published_" + value["verdict_class"] + " completed=1 pending=0", flush=True
        )
        return int(not value["required_checks_passed"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("[exp8585] rejected " + str(error), flush=True)
        return 1
